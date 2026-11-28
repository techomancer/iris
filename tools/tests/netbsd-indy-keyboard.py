#!/usr/bin/env python3
"""Check that NetBSD/sgimips sysinst on an Indy takes cursor keys and letters.

Boots the NetBSD/sgimips install CD on an R4400 Indy with the Newport console,
waits for sysinst's first menu, then presses keys on the emulated PS/2
keyboard through the monitor and compares screenshots:

  1. Down, Down, Up must each change exactly two adjacent text rows: the row
     the highlight left and the one it moved to, one row further down for the
     second Down and the same two rows again for Up.
  2. A letter (`a`, the first item) must change the screen once, and the
     screen must then stay still: the key was released and did not repeat.

NetBSD's keyboard driver asks the 8042 to translate the keyboard's scan
codes to set 1; without that translation the arrows mean nothing to it and
`a` reads as an Enter that is never released (issue #765).

Only which rows change is used, so the check works whether the console draws
its glyphs (NetBSD 11.99.11) or only its text cursor (NetBSD 11.1 under
`--ci`).

    tools/tests/netbsd-indy-keyboard.py --iso NetBSD-11.1-sgimips.iso \\
        [--iris target/release/iris] [--iris-ci target/release/iris-ci] \\
        [--workdir DIR] [--port-base 27650]

The CD is attached read-only through a copy-on-write overlay. The guest never
mounts a disk, so the run ends with `iris-ci quit`. Exit status: 0 pass,
1 fail, 2 setup error. Screenshots stay in the work directory.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from mon import Monitor  # noqa: E402

BOOT = "boot -f dksc(0,4,8)ip2xboot dksc(0,4,0)/ip2x"


def read_png(path: Path) -> tuple[int, int, list[bytes], int]:
    data = path.read_bytes()
    i, idat = 8, b""
    w = h = ct = 0
    while i < len(data):
        n, = struct.unpack(">I", data[i:i + 4])
        kind, chunk = data[i + 4:i + 8], data[i + 8:i + 8 + n]
        if kind == b"IHDR":
            w, h, _depth, ct = struct.unpack(">IIBB", chunk[:10])
        elif kind == b"IDAT":
            idat += chunk
        i += 12 + n
    raw = zlib.decompress(idat)
    bpp = {2: 3, 6: 4}[ct]
    stride = w * bpp
    rows, prev, o = [], bytearray(stride), 0
    for _ in range(h):
        f, line = raw[o], bytearray(raw[o + 1:o + 1 + stride])
        o += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1:
                line[x] = (line[x] + a) & 255
            elif f == 2:
                line[x] = (line[x] + b) & 255
            elif f == 3:
                line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(bytes(line))
        prev = line
    return w, h, rows, bpp


class Guest:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.workdir)
        self.sock = os.path.join(tempfile.gettempdir(), f"iris-netbsd-kbd-{os.getpid()}.sock")
        self.proc = None
        self.mon = None
        self.shots = 0

    def ci(self, *cmd, timeout=120):
        return subprocess.run([self.args.iris_ci, "--socket", self.sock, *cmd],
                              capture_output=True, text=True, timeout=timeout)

    def start(self):
        b = self.args.port_base
        cfg = self.work / "netbsd-indy.toml"
        cfg.write_text(
            f'prom = "(embedded)"\nnvram = "nvram.bin"\nbanks = [128, 128, 0, 0]\n'
            f'headless = false\nno_audio = true\nmonitor_port = {b}\nci_socket = "{self.sock}"\n\n'
            f'[machine]\nprofile = "indy_ip24"\ncpu = "r4400_150"\n\n'
            f'[graphics]\nboard = "newport"\n\n[network]\nmode = "nat"\n\n'
            f'[scsi.4]\npath = "{Path(self.args.iso).resolve()}"\ncdrom = true\n'
            f'kind = "cdrom"\noverlay = true\n')
        self.log = open(self.work / "iris.log", "w")
        self.proc = subprocess.Popen(
            [self.args.iris, "--config", str(cfg), "--ci",
             "--device", f"serial@serial2,backend=tcp:{b + 1}", "--device", f"serial@serial1,backend=tcp:{b + 2}"],
            cwd=self.work, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 120
        while not os.path.exists(self.sock):
            if self.proc.poll() is not None or time.time() > deadline:
                raise RuntimeError(f"iris did not come up; see {self.work / 'iris.log'}")
            time.sleep(1)
        r = self.ci("status", "--wait-ready", "120")
        if r.returncode:
            raise RuntimeError(r.stderr)
        self.ci("start")
        self.mon = Monitor(port=b)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.ci("quit", timeout=60)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                print("iris did not exit after quit; leaving it running", file=sys.stderr)
        if os.path.exists(self.sock):
            os.unlink(self.sock)

    def key(self, k):
        self.mon.cmd(f"key {k}")

    def shot(self, tag):
        self.shots += 1
        path = self.work / f"{self.shots:03d}-{tag}.png"
        reply = self.mon.cmd(f"screen shot {path}")
        return path if path.exists() else self._no_shot(reply)

    @staticmethod
    def _no_shot(reply):
        print(f"  (no screenshot: {reply})")
        return None

    def lit(self, path):
        """Bounding box of the lit pixels, and the pixel rows themselves."""
        if path is None:
            return None, 0, None
        w, h, rows, bpp = read_png(path)
        xs, ys = [], []
        for y, row in enumerate(rows):
            for x in range(w):
                o = x * bpp
                if row[o] + row[o + 1] + row[o + 2] > 60:
                    xs.append(x)
                    ys.append(y)
        box = (min(xs), min(ys), max(xs), max(ys)) if xs else None
        return box, len(xs), rows


def changed_rows(a, b):
    """First and last pixel row that differ between two screenshots."""
    if a is None or b is None:
        return None
    ys = [y for y, (ra, rb) in enumerate(zip(a, b)) if ra != rb]
    return (ys[0], ys[-1]) if ys else None


def wait_for(cond, timeout, step=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return None


def boot_to_sysinst(g: Guest):
    # Fresh NVRAM: the PROM tries to boot sash from the CD, fails, and waits
    # for a key. Escape answers that, or stops the autoboot if it comes
    # earlier; either way it leads to the maintenance menu, where 5 is the
    # command monitor.
    def non_black():
        box, _, _ = g.lit(g.shot("prom"))
        return box is not None
    if not wait_for(non_black, 120, 5):
        raise RuntimeError("the PROM never drew anything")
    g.mon.cmd("screen wait 5000 120000")
    g.key("Escape")
    time.sleep(8)
    g.key("5")
    time.sleep(6)
    g.mon.cmd(f"key type {BOOT}")
    g.key("Return")

    # sysinst's first menu waits for a key: once the kernel has started, the
    # screen stops changing. Four identical samples (about half a minute).
    time.sleep(30)
    seen = []

    def settled():
        box, _, rows = g.lit(g.shot("boot"))
        seen.append(rows if box else None)
        last = seen[-4:]
        return len(last) == 4 and last[0] is not None and all(r == last[0] for r in last)
    if not wait_for(settled, 900, 5):
        raise RuntimeError("sysinst's menu did not appear")
    return seen[-1]


def check_arrows(g: Guest, screen) -> list[str]:
    bands = []
    for k in ("Down", "Down", "Up"):
        g.key(k)
        time.sleep(3)
        _, _, after = g.lit(g.shot(f"key-{k}"))
        bands.append(changed_rows(screen, after))
        print(f"  {k:5s} changed rows: {bands[-1]}")
        screen = after
    d1, d2, d3 = bands
    if not d1:
        return ["Down changed nothing on the screen: the arrows do not reach sysinst"]
    row_h = (d1[1] - d1[0] + 1) // 2
    fails = []
    if not 8 <= row_h <= 32 or d1[1] - d1[0] + 1 != 2 * row_h:
        fails.append(f"Down changed rows {d1}, not two text rows")
    elif d2 != (d1[0] + row_h, d1[1] + row_h):
        fails.append(f"the second Down changed rows {d2}, want {(d1[0] + row_h, d1[1] + row_h)}")
    elif d3 != d2:
        fails.append(f"Up changed rows {d3}, want {d2}")
    return fails


def check_letter(g: Guest) -> list[str]:
    _, _, before = g.lit(g.shot("before-a"))
    g.key("a")
    time.sleep(4)
    samples = []
    for i in range(6):
        _, _, rows = g.lit(g.shot(f"after-a-{i}"))
        samples.append(rows)
        time.sleep(2)
    fails = []
    if samples[0] == before:
        fails.append("`a` did not change the screen")
    changes = sum(1 for a, b in zip(samples, samples[1:]) if a != b)
    print(f"  a     screen changed: {samples[0] != before}; changes over the next 10 s: {changes}")
    if changes:
        fails.append(f"the screen kept changing after `a` ({changes} of 5 samples): the key repeats")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iso", required=True, help="NetBSD/sgimips install CD image")
    ap.add_argument("--iris", default=str(ROOT / "target/release/iris"))
    ap.add_argument("--iris-ci", default=str(ROOT / "target/release/iris-ci"))
    ap.add_argument("--workdir", help="where the config, log and screenshots go")
    ap.add_argument("--port-base", type=int, default=27650,
                    help="monitor port; serial A and B take the next two")
    args = ap.parse_args()
    if not args.workdir:
        args.workdir = tempfile.mkdtemp(prefix="netbsd-kbd-")
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    for p in (args.iso, args.iris, args.iris_ci):
        if not os.path.exists(p):
            print(f"missing: {p}", file=sys.stderr)
            return 2

    g = Guest(args)
    try:
        g.start()
        screen = boot_to_sysinst(g)
        print("sysinst menu up")
        fails = check_arrows(g, screen) + check_letter(g)
        print("ps2:", g.mon.cmd("ps2 status"))
    except Exception as e:  # setup failure, not a keyboard verdict
        print(f"ERROR: {e}", file=sys.stderr)
        g.stop()
        return 2
    g.stop()
    print(f"screenshots in {args.workdir}")
    if fails:
        print("FAIL: " + "; ".join(fails))
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
