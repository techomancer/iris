#!/usr/bin/env python3
"""Run one glprim case in the IRIX guest and check the result (GR2 GL loop).

Steps: start `gr2 trace` on the IRIS monitor, run /tmp/gltest/glprim with the
given options in the background in the guest (tools/irix-sh.py), wait for it
to draw, probe pixels with `gr2 pix`, dump the framebuffer, crop the glprim
window to <out>/<case>.png, stop the trace (<out>/<case>.log).

Probes are in GL window coordinates (origin bottom-left of the client area):
  --probe 200,120=0000ff      expect rgb (hex, +-tolerance per channel)
  --probe 200,120             just print
Exit status: number of failed probes (0 = pass), 100+ on setup errors.

Example:
  tools/glprim-case.py tri-flat -- --hold 6000
  tools/glprim-case.py tri-smooth --probe 200,120=555555 -- --smooth
"""
import argparse
import os
import re
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
IRIX_SH = os.path.join(HERE, "irix-sh.py")


class Monitor:
    def __init__(self, host="127.0.0.1", port=8888):
        self.s = socket.create_connection((host, port), timeout=30)
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
        return re.sub(r"\n?> $", "", out).strip()


def guest(*cmds, timeout=120):
    r = subprocess.run([sys.executable, IRIX_SH, "--timeout", str(timeout), "run", "-k", *cmds],
                       capture_output=True, text=True, timeout=timeout + 60)
    return r.returncode, r.stdout + r.stderr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case")
    p.add_argument("--out", default=os.environ.get("GLPRIM_OUT", "/tmp/glprim-cases"))
    p.add_argument("--probe", action="append", default=[])
    p.add_argument("--tol", type=int, default=2)
    p.add_argument("--settle", type=float, default=6.0, help="seconds to wait for the draw")
    p.add_argument("--notrace", action="store_true")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    a = p.parse_args(argv[:split])
    gargs = argv[split + 1:]
    if not any(x == "--hold" for x in gargs):
        gargs += ["--hold", "15000"]
    os.makedirs(a.out, exist_ok=True)
    log = os.path.join(a.out, a.case + ".log")
    mon = Monitor()
    if not a.notrace:
        print(mon.cmd(f"gr2 trace {log} all"))
    quoted = " ".join("'" + x.replace("'", "'\\''") + "'" for x in gargs)
    rc, out = guest(f"cd /tmp/gltest && (./glprim {quoted} > /tmp/glprim.out 2>&1 &)",
                    f"sleep {int(a.settle)}; cat /tmp/glprim.out")
    print(out.strip())
    m = re.search(r"window origin (-?\d+),(-?\d+) size (\d+)x(\d+)", out)
    if not m:
        print("glprim did not report a window (see output above)")
        if not a.notrace:
            mon.cmd("gr2 trace off")
        return 101
    ox, oy, w, h = map(int, m.groups())
    fails = 0
    for pr in a.probe:
        pos, _, want = pr.partition("=")
        gx, gy = map(int, pos.split(","))
        dx, dy = ox + gx, oy + h - 1 - gy
        res = mon.cmd(f"gr2 pix {dx} {dy}")
        mm = re.search(r"rgb=([0-9a-f]{6})", res)
        got = mm.group(1) if mm else "??????"
        status = ""
        if want:
            gv = [int(got[i:i + 2], 16) for i in (0, 2, 4)] if mm else None
            wv = [int(want[i:i + 2], 16) for i in (0, 2, 4)]
            ok = gv is not None and all(abs(x - y) <= a.tol for x, y in zip(gv, wv))
            status = "ok  " if ok else "FAIL"
            fails += 0 if ok else 1
        print(f"  {status} gl ({gx:4d},{gy:4d}) screen ({dx:4d},{dy:4d}) rgb={got} {('want ' + want) if want else ''}")
    dump = os.path.join(a.out, a.case + "-fb")
    mon.cmd(f"gr2 fbdump {dump}")
    try:
        from PIL import Image
        im = Image.open(os.path.join(dump, "screen.png"))
        im.crop((max(ox - 8, 0), max(oy - 8, 0), ox + w + 8, oy + h + 8)).save(os.path.join(a.out, a.case + ".png"))
        print(f"  window image: {os.path.join(a.out, a.case + '.png')}")
    except Exception as e:  # noqa: BLE001
        print(f"  (crop failed: {e})")
    if not a.notrace:
        print(mon.cmd("gr2 trace off"))
    # Let glprim exit before the next case.
    guest("sleep 1; ps -ef | grep -v grep | grep glprim >/dev/null && kill `ps -ef | grep -v grep | grep glprim | awk '{print $2}'`; true")
    print(f"{a.case}: {'PASS' if fails == 0 else f'{fails} FAILED'}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
