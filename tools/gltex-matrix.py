#!/usr/bin/env python3
"""IMPACT texture check: glprim --scene tex matrix: probe every texel's block centre on the left
quad (raw framebuffer word via `mgras pix`, so the DAC gamma does not
matter) and compare with the texture. Needs /tmp/gltest/glprim in the guest
and IRIS_ROOT_PASSWORD if root has one.

    tools/gltex-matrix.py                       # the default cases
    tools/gltex-matrix.py "luminance_alpha 64" "rgba 256 --texfilter linear,linear"
"""
import os, re, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mon import Monitor

IRIX = ["python3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "irix-sh.py"), "--timeout", "120", "run", "-k"]


def guest(cmd):
    return subprocess.run(IRIX + [cmd], capture_output=True, text=True, timeout=200).stdout


def expect(fmt, s, t, n):
    v_s, v_t = s * 256 // n, t * 256 // n
    if fmt in ("rgba", "rgb"):
        return (v_s, v_t, 0xA5)
    if fmt in ("luminance", "luminance_alpha", "intensity"):
        return (v_s, v_s, v_s)
    return None


def run(fmt, n, extra, mon):
    w, h = 400, 300
    opts = f"--scene tex --visual 8,8,8,8 --texfmt {fmt} --texsize {n} {extra} --hold 8000"
    out = guest(f"cd /tmp/gltest && (./glprim {opts} > /tmp/gp.out 2>&1 &); sleep 5; cat /tmp/gp.out")
    m = re.search(r"window origin (-?\d+),(-?\d+) size (\d+)x(\d+)", out)
    if not m:
        print(f"{fmt} {n} {extra}: no window\n{out}")
        return
    ox, oy, w, h = map(int, m.groups())
    probes = min(n, 8)
    bad, total, samples = 0, 0, []
    for ti in range(probes):
        for si in range(probes):
            s, t = si * n // probes, ti * n // probes
            gx = int(0.1 * w + 0.35 * w * (s + 0.5) / n)
            gy = int(0.2 * h + 0.6 * h * (t + 0.5) / n)
            r = mon.cmd(f"mgras pix {ox + gx} {oy + h - 1 - gy}")
            mm = re.search(r"main=0x([0-9a-f]+)", r)
            word = int(mm.group(1), 16) if mm else 0
            got = (word & 0xFF, (word >> 8) & 0xFF, (word >> 16) & 0xFF)
            want = expect(fmt, s, t, n)
            total += 1
            if want and any(abs(a - b) > 2 for a, b in zip(got, want)):
                bad += 1
                if len(samples) < 6:
                    samples.append(f"texel ({s},{t}) got {got} want {want}")
    print(f"{fmt:16} {n:4} {extra:28} {bad}/{total} wrong  {'; '.join(samples)}")
    guest("killall glprim 2>/dev/null; true")


mon = Monitor()
cases = sys.argv[1:] or ["rgba 8", "rgb 8", "luminance 8", "luminance_alpha 8", "intensity 8",
                         "rgba 64", "rgba 256", "luminance 64", "luminance_alpha 64"]
for c in cases:
    parts = c.split(maxsplit=2)
    run(parts[0], int(parts[1]), parts[2] if len(parts) > 2 else "", mon)
