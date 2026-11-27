#!/usr/bin/env python3
"""netbsd-cd-boot.py: boot a NetBSD/sgimips install CD to sysinst on an Indy.

Starts its own iris (headless, blank NVRAM, the CD as a read-only SCSI 4
CD-ROM), enters the PROM Command Monitor over the CI socket, boots the CD's
install kernel with ip2xboot and waits for the installer's menu or a kernel
panic. Then it quits iris; nothing but the NVRAM in --out is written.

    tools/netbsd-cd-boot.py NetBSD-11.1-sgimips.iso --cpu r5000

Exit status: 0 the installer menu appeared, 1 the kernel panicked or the boot
stalled, 2 iris or the PROM did not come up. The serial transcript is
<out>/serial.log; iris's own output is <out>/iris.log.

The R5000 run covers techomancer/iris#773: the IP24 PROM cleared Config's
line-size bits and NetBSD stopped with "r5k picache line size 16".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

INSTALLER = b"This menu-driven tool"
PANIC = re.compile(rb"panic: [^\r\n]*")
BOOT = "boot -f dksc(0,4,8)ip2xboot ip2x"


class Ci:
    def __init__(self, path: str):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.f = self.sock.makefile("rw", buffering=1, newline="\n")

    def rpc(self, cmd: str, **args) -> dict:
        self.f.write(json.dumps({"cmd": cmd, "args": args}) + "\n")
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise ConnectionError(f"CI socket closed during {cmd!r}")
        return json.loads(line)


def config(iso: Path, cpu: str, sock: str, port: int) -> str:
    return f"""# Written by tools/netbsd-cd-boot.py.
headless = true
monitor_port = {port}
devices = [ {{ type = "serial", at = "serial2", backend = "tcp:{port + 1}" }}, {{ type = "serial", at = "serial1", backend = "tcp:{port + 2}" }} ]
nvram = "nvram.bin"
serial_log = "serial.log"
serial_log_overwrite = true
ci = true
ci_socket = {json.dumps(sock)}

[audio]
host = "null"

[machine]
profile = "indy_ip24"
cpu = {json.dumps(cpu)}

[scsi.4]
path = {json.dumps(str(iso.resolve()))}
cdrom = true
overlay = true
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("iso", type=Path, help="NetBSD-*-sgimips.iso")
    ap.add_argument("--cpu", default="r5000", help="r4400 or r5000 (default r5000)")
    ap.add_argument("--iris", type=Path, default=Path(__file__).resolve().parent.parent / "target/release/iris")
    ap.add_argument("--out", type=Path, default=None, help="work directory (default: a new temporary one)")
    ap.add_argument("--port", type=int, default=27730, help="monitor port; serial A and B take the next two")
    ap.add_argument("--timeout", type=int, default=900, help="seconds from the boot command to the installer")
    a = ap.parse_args()

    if not a.iso.is_file():
        print(f"netbsd-cd-boot: no ISO at {a.iso}", file=sys.stderr)
        return 2
    out = a.out or Path(tempfile.mkdtemp(prefix="netbsd-cd-boot-"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "nvram.bin").unlink(missing_ok=True)  # a blank NVRAM keeps the PROM on serial, menu up
    sock = os.path.join(tempfile.gettempdir(), f"netbsd-cd-boot-{os.getpid()}.sock")
    (out / "iris.toml").write_text(config(a.iso, a.cpu, sock, a.port))
    log = open(out / "iris.log", "wb")
    iris = subprocess.Popen([str(a.iris), "--config", "iris.toml"], cwd=out,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    seen = b""

    def say(msg: str) -> None:
        print(f"netbsd-cd-boot ({a.cpu}): {msg}", flush=True)

    def wait(ci: Ci, done, secs: int):
        """Drain serial until done(seen) returns something, or secs pass."""
        nonlocal seen
        end = time.monotonic() + secs
        while time.monotonic() < end:
            seen += ci.rpc("serial-read").get("data", "").encode("latin1", "replace")
            hit = done(seen)
            if hit:
                return hit
            if iris.poll() is not None:
                return None
            time.sleep(1)
        return None

    ci = None
    try:
        for _ in range(120):
            try:
                ci = Ci(sock)
                if ci.rpc("ping").get("ok"):
                    break
            except OSError:
                ci = None
            if iris.poll() is not None:
                say(f"iris exited ({iris.returncode}); see {out}/iris.log")
                return 2
            time.sleep(1)
        else:
            say("CI socket never came up")
            return 2
        ci.rpc("start")
        if not wait(ci, lambda s: b"Option?" in s, 300):
            say("PROM menu never appeared on serial")
            return 2
        ci.rpc("serial-send", data="5\r")
        if not wait(ci, lambda s: s.rstrip().endswith(b">>"), 60):
            say("no Command Monitor prompt")
            return 2
        seen = b""
        ci.rpc("serial-send", data=BOOT + "\r")
        say(BOOT)

        def verdict(s: bytes):
            if INSTALLER in s:
                return ("pass", "reached the sysinst menu")
            m = PANIC.search(s)
            if m:
                return ("fail", m.group(0).decode("latin1"))
            return None

        hit = wait(ci, verdict, a.timeout)
        if hit is None:
            tail = seen[-600:].decode("latin1", "replace")
            say(f"no installer and no panic within {a.timeout}s; last output:\n{tail}")
            return 1
        say(f"{hit[0].upper()}: {hit[1]}")
        return 0 if hit[0] == "pass" else 1
    finally:
        # The guest runs from RAM and the CD is read-only: quitting is safe.
        if ci is not None:
            try:
                ci.rpc("quit")
            except (OSError, ConnectionError, ValueError):
                pass
        try:
            iris.wait(timeout=30)
        except subprocess.TimeoutExpired:
            iris.terminate()
            iris.wait(timeout=30)
        log.close()
        try:
            os.unlink(sock)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
