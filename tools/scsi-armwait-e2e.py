#!/usr/bin/env python3
"""End-to-end check that the WD33C93A starts a data phase as soon as the guest
arms the HPC3 SCSI DMA channel (issue 346).

NetBSD's wdsc driver writes TRANSFER_INFO to the WD33C93A and arms the HPC3
channel (dmago) only after it, so the chip reaches its data phase with the
channel still off and must wait for the arming. A chip that polls the channel
finds it at the end of a 100 us sleep at the earliest; a chip woken by the
arming starts as soon as the CPU arms it.

This boots the NetBSD/sgimips install CD on an Indy, leaves sysinst for its
shell, reads the CD raw device with dd, and reads `scsi armwait` from the
monitor: how many data phases waited for the arming, how often the chip
checked the channel, and how long each wait lasted.

PASS when at least 100 data phases waited and most of them ended within one
100 us poll interval.

    tools/scsi-armwait-e2e.py IRIS_BINARY NetBSD-<version>-sgimips.iso

The CD image is opened read-only and the machine has no disk, so nothing is
written but a scratch NVRAM, console log and config in a temporary directory.
Ports: --monitor-port (default 23460); the console is a Unix socket there.
The guest is halted back to the PROM before iris is stopped.
"""
import argparse
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mon import Monitor  # noqa: E402

ANSI = re.compile(rb"\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][0-9A-Za-z]|\x1b[=>]")


class Console:
    """The Indy's console line (SCC channel B) on a raw Unix socket."""

    def __init__(self, path, log, deadline=120):
        end = time.time() + deadline
        while True:
            try:
                self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.s.connect(path)
                break
            except OSError:
                if time.time() > end:
                    raise
                time.sleep(0.5)
        self.s.settimeout(1)
        self.buf = b""
        self.log = log

    def expect(self, pattern, timeout):
        """Wait for `pattern` (bytes regex) in output not yet consumed."""
        end = time.time() + timeout
        rx = re.compile(pattern)
        while True:
            m = rx.search(ANSI.sub(b"", self.buf))
            if m:
                self.buf = b""
                return m
            if time.time() > end:
                tail = ANSI.sub(b"", self.buf)[-600:].decode("latin-1")
                raise TimeoutError(f"no {pattern!r} after {timeout}s; console ends:\n{tail}")
            try:
                chunk = self.s.recv(65536)
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionError("console closed")
            self.log.write(chunk)
            self.log.flush()
            self.buf += chunk

    def send(self, text, cr=True):
        for ch in text.encode() + (b"\r" if cr else b""):
            self.s.sendall(bytes([ch]))
            time.sleep(0.02)


def armwait(mon):
    out = mon.cmd("scsi armwait")
    vals = dict(re.findall(r"(\w+|<\w+|>=\w+)=([0-9.]+)", out))
    return out, vals


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("iris")
    ap.add_argument("iso")
    ap.add_argument("--monitor-port", type=int, default=23460)
    ap.add_argument("--keep", help="directory for the config, console log and iris output")
    a = ap.parse_args()

    work = a.keep or tempfile.mkdtemp(prefix="armwait-")
    os.makedirs(work, exist_ok=True)
    # A Unix socket path is limited to about 100 bytes; keep it short.
    sock = os.path.join(tempfile.mkdtemp(prefix="aw", dir="/tmp"), "con")
    cfg = os.path.join(work, "indy.toml")
    with open(cfg, "w") as f:
        f.write(f"""headless = true
no_audio = true
banks = [128, 128, 0, 0]
nvram = "{work}/nvram.bin"
monitor_port = {a.monitor_port}
devices = [ {{ type = "serial", at = "serial1", backend = "unix:{sock}" }} ]
[machine]
profile = "indy_ip24"
cpu = "r4400_150"
[scsi.4]
path = "{os.path.abspath(a.iso)}"
cdrom = true
""")
    out = open(os.path.join(work, "iris.out"), "wb")
    log = open(os.path.join(work, "console.log"), "wb")
    iris = subprocess.Popen([a.iris, "--config", cfg], stdout=out, stderr=subprocess.STDOUT)
    halted = False
    os_started = False  # before the boot command only the PROM runs: safe to stop
    try:
        con = Console(sock, log)
        con.expect(rb"Option\? ", 300)
        con.send("5")
        con.expect(rb">> ", 60)
        os_started = True
        con.send("boot -f scsi(0)cdrom(4)rdisk(0)partition(8)ip2xboot "
                 "scsi(0)cdrom(4)rdisk(0)partition(0)/ip2x")
        con.expect(rb"Installation messages in English", 900)
        con.send("")
        con.expect(rb"Exit Install System", 120)
        time.sleep(2)
        con.send("x", cr=False)
        time.sleep(1)
        con.send("")
        con.expect(rb"\n# ?$|# $", 120)

        mon = Monitor(port=a.monitor_port)
        boot, _ = armwait(mon)
        print("== NetBSD boot and autoconfiguration")
        print(boot)
        results = []
        for bs, count in (("32k", 256), ("2k", 2000)):
            mon.cmd("scsi armwait reset")
            con.send(f"dd if=/dev/rcd0a of=/dev/null bs={bs} count={count}; echo DD-END")
            m = con.expect(rb"(?s)bytes transferred in ([0-9.]+) secs.*?DD-END", 1200)
            text, vals = armwait(mon)
            print(f"== dd bs={bs} count={count}: {m.group(1).decode()} s guest time")
            print(text)
            results.append(vals)

        con.send("halt")
        con.expect(rb"Option\? ", 300)
        halted = True
    finally:
        if halted or not os_started or iris.poll() is not None:
            if iris.poll() is None:
                iris.send_signal(signal.SIGTERM)
                iris.wait(30)
        else:
            print(f"guest not halted; leaving iris (pid {iris.pid}) running, files in {work}")
        try:
            os.rmdir(os.path.dirname(sock))
        except OSError:
            pass

    late = sum(int(v.get("late", 0)) for v in results)
    fast = sum(int(v.get("<50us", 0)) + int(v.get("<100us", 0)) for v in results)
    if late < 100:
        print(f"RESULT: only {late} data phases waited for their channel; nothing to judge")
        return 2
    verdict = "PASS" if fast * 2 >= late else "FAIL"
    print(f"RESULT: {verdict}: {fast} of {late} waits ended within one 100 us poll interval")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
