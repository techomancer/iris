#!/usr/bin/env python3
"""End-to-end check that a shared IOC status bit is the OR of its sources.

On an Indigo2 the graphics slot and GIO expansion slot 1 share the IOC's
L0 GRAPHICS bit: Newport's retrace arrives as the graphics slot's GIO
interrupt 1, and a virtio-mmio device in slot 1 drives slot 1's. While either
source holds its line the bit must stay set. This starts a real iris (the
built-in IP22 PROM, Newport, virtio-serial in slot 1, no disk), lets the PROM
bring the display up, stops the CPU, and then works the two sources from the
monitor:

  A  virtio holds its interrupt; the Newport retrace is released (a read of
     REX3 STATUS). L0 GRAPHICS must stay set while virtio's InterruptStatus
     is 1.
  B  Newport holds its retrace (EXT_IO SG_IRQ_1 low); virtio's interrupt is
     acknowledged. L0 GRAPHICS must stay set while EXT_IO shows the graphics
     slot pending (one `ioc status` snapshot, taken under the IOC lock).

Each iteration that sees the bit clear with a source still holding it counts
as a lost interrupt. A retrace landing between the release and the sample
re-raises the bit and hides a loss, so a build without the fix shows losses
in most iterations, not all.

    tools/ioc-shared-bit-e2e.py --iris target/release/iris \\
        --iris-ci target/release/iris-ci [--iterations 100] [--base-port 22620]

Exit status: 0 no interrupt lost, 1 an interrupt was lost, 2 the test could
not be set up. The CPU is stopped in the PROM, no guest OS runs, and the only
disk is a blank image the script creates in the run directory, behind an
overlay.
"""
import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile
import time

VIRTIO = 0xBF600000          # GIO expansion slot 1, KSEG1
REX3_STATUS = 0xBF0F1338     # reading it clears VRINT and releases the retrace
RAM = 0x0C000000             # ring memory (physical); banks = [128, ...]
KSEG1 = 0xA0000000
DESC, AVAIL, USED, DATA = RAM, RAM + 0x200, RAM + 0x400, RAM + 0x800
L0_GRAPHICS = 0x40
EXT_SG_IRQ_1 = 0x0200        # active low


class Monitor:
    def __init__(self, port, timeout=60):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.buf = b""
        self._until_prompt()

    def _until_prompt(self):
        while not re.search(rb"(^|\n)> $", self.buf):
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("monitor closed")
            self.buf += chunk
        out, self.buf = self.buf, b""
        return out.decode("latin-1", "replace")

    def cmd(self, c):
        self.s.sendall((c + "\n").encode())
        out = self._until_prompt()
        return re.sub(r"\n?> $", "", out).replace("\r", "").strip()


def config(base):
    return f"""headless = false
ci = true
ci_socket = "tcp:127.0.0.1:{base + 1}"
monitor_port = {base}
devices = [ {{ type = "serial", at = "serial2", backend = "tcp:{base + 3}" }}, {{ type = "serial", at = "serial1", backend = "tcp:{base + 4}" }} ]
no_audio = true
banks = [128, 0, 0, 0]
nvram = "nvram.bin"
nveeprom = "nveeprom.bin"
serial_log = "console.log"

[machine]
profile = "indigo2_ip22"

[graphics]
board = "newport"
heads = 1

[scsi.1]
path = "blank.raw"
overlay = true

[virtio_serial]
enabled = true
slot = "exp1"

[[virtio_serial.port]]
console = true
"""


class Rig:
    def __init__(self, mon):
        self.m = mon

    def w32(self, addr, val):
        self.m.cmd(f"mw {addr:#x} {val:#x} w")

    def w8(self, addr, val):
        self.m.cmd(f"mw {addr:#x} {val:#x} b")

    def r32(self, addr):
        out = self.m.cmd(f"mem {addr:#x} 1 w")
        # "0xffffffffbf600000: 0x74726976"
        words = re.findall(r"0x([0-9a-fA-F]+)", out.split(":", 1)[-1])
        if not words:
            raise RuntimeError(f"cannot parse mem output: {out!r}")
        return int(words[0], 16)

    def le(self, addr, val, n):
        for i in range(n):
            self.w8(KSEG1 + addr + i, (val >> (8 * i)) & 0xFF)

    def ioc(self):
        out = self.m.cmd("ioc status")
        l0 = re.search(r"L0\s+stat=([0-9a-f]{2})", out)
        ext = re.search(r"EXT_IO=([0-9a-f]{4})", out)
        if not (l0 and ext):
            raise RuntimeError(f"cannot parse ioc status: {out!r}")
        return int(l0.group(1), 16), int(ext.group(1), 16)

    def virtio_up(self):
        v = VIRTIO
        if self.r32(v) != 0x74726976:
            raise RuntimeError("no virtio-mmio device in GIO slot 1")
        self.w32(v + 0x70, 0)                      # reset
        self.w32(v + 0x70, 1 | 2)                  # ACKNOWLEDGE | DRIVER
        self.w32(v + 0x24, 0); self.w32(v + 0x20, 0)
        self.w32(v + 0x24, 1); self.w32(v + 0x20, 1)   # VERSION_1 only
        self.w32(v + 0x70, 1 | 2 | 8)              # FEATURES_OK
        if self.r32(v + 0x70) & 8 == 0:
            raise RuntimeError("virtio refused the features")
        self.w32(v + 0x30, 1)                      # transmitq
        self.w32(v + 0x38, 8)
        self.w32(v + 0x80, DESC); self.w32(v + 0x84, 0)
        self.w32(v + 0x90, AVAIL); self.w32(v + 0x94, 0)
        self.w32(v + 0xA0, USED); self.w32(v + 0xA4, 0)
        self.w32(v + 0x44, 1)
        self.w32(v + 0x70, 1 | 2 | 8 | 4)          # DRIVER_OK
        for off in range(0, 0x600, 4):             # clear the rings
            self.w32(KSEG1 + RAM + off, 0)
        self.le(DESC, DATA, 8)                     # desc 0: one byte at DATA
        self.le(DESC + 8, 1, 4)
        self.w8(KSEG1 + DATA, ord("x"))
        self.avail = 0

    def virtio_raise(self):
        """Hand the device one TX buffer; it completes and raises its line."""
        self.le(AVAIL + 4 + 2 * (self.avail % 8), 0, 2)
        self.avail += 1
        self.le(AVAIL + 2, self.avail & 0xFFFF, 2)
        self.w32(VIRTIO + 0x50, 1)
        deadline = time.time() + 5
        while self.r32(VIRTIO + 0x60) & 1 == 0:
            if time.time() > deadline:
                raise RuntimeError("virtio never raised its interrupt")
        return True

    def virtio_ack(self):
        self.w32(VIRTIO + 0x64, 1)

    def wait_retrace(self, timeout=5):
        deadline = time.time() + timeout
        while True:
            _, ext = self.ioc()
            if ext & EXT_SG_IRQ_1 == 0:
                return
            if time.time() > deadline:
                raise RuntimeError("no Newport retrace on EXT_IO SG_IRQ_1")


def run(args, workdir):
    base = args.base_port
    cfg = os.path.join(workdir, "iris.toml")
    with open(cfg, "w") as f:
        f.write(config(base))
    # A blank disk, so the PROM has a SCSI target and nothing real is attached.
    with open(os.path.join(workdir, "blank.raw"), "wb") as f:
        f.truncate(8 << 20)
    log = open(os.path.join(workdir, "iris.log"), "w")
    proc = subprocess.Popen([os.path.abspath(args.iris), "--config", cfg],
                            cwd=workdir, stdout=log, stderr=subprocess.STDOUT)
    try:
        mon = None
        deadline = time.time() + 120
        while mon is None:
            if proc.poll() is not None:
                raise RuntimeError(f"iris exited ({proc.returncode}); see {log.name}")
            try:
                mon = Monitor(base)
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError("monitor never answered")
                time.sleep(0.5)
        rig = Rig(mon)
        print(mon.cmd("machine-start"))
        # Let the PROM program the VC2 so Newport retraces, then park the CPU
        # so nothing but this script touches the two sources.
        rig.wait_retrace(timeout=args.boot_timeout)
        time.sleep(args.settle)
        print(mon.cmd("stop").splitlines()[0])
        rig.virtio_up()

        lost_a = lost_b = 0
        for _ in range(args.iterations):
            # A: virtio holds, Newport lets go.
            rig.virtio_raise()
            rig.wait_retrace()
            rig.r32(REX3_STATUS)
            l0, _ = rig.ioc()
            if rig.r32(VIRTIO + 0x60) & 1 and l0 & L0_GRAPHICS == 0:
                lost_a += 1
            rig.virtio_ack()
            # B: Newport holds, virtio lets go.
            rig.virtio_raise()
            rig.wait_retrace()
            rig.virtio_ack()
            l0, ext = rig.ioc()
            if ext & EXT_SG_IRQ_1 == 0 and l0 & L0_GRAPHICS == 0:
                lost_b += 1
        n = args.iterations
        print(f"A (virtio holds, retrace released): lost {lost_a}/{n}")
        print(f"B (retrace holds, virtio released): lost {lost_b}/{n}")
        return 1 if lost_a or lost_b else 0
    finally:
        # No guest OS runs and no disk is attached: quit through the CI
        # socket, and only if that fails, terminate.
        subprocess.run([os.path.abspath(args.iris_ci), "--socket",
                        f"tcp:127.0.0.1:{base + 1}", "quit"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=30)
        log.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--iris", required=True)
    p.add_argument("--iris-ci", required=True)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--base-port", type=int, default=22620,
                   help="monitor; +1 CI socket, +3/+4 serial ports")
    p.add_argument("--boot-timeout", type=float, default=180)
    p.add_argument("--settle", type=float, default=5,
                   help="seconds the PROM runs after the first retrace")
    p.add_argument("--workdir", help="run directory (default: a new temp dir)")
    args = p.parse_args()
    workdir = args.workdir or tempfile.mkdtemp(prefix="ioc-e2e-")
    os.makedirs(workdir, exist_ok=True)
    print(f"run directory: {workdir}")
    try:
        return run(args, workdir)
    except (RuntimeError, OSError) as e:
        print(f"setup failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
