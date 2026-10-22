#!/usr/bin/env python3
"""Map how libGLcore decodes GE state words, by making the GE answer chosen
values (`mgras probe`) while test/gltest/glstate queries the state.

  tools/mgras-probe-states.py OUTDIR [--ranges 0x1fa-0x1fc,...] [--only SUBSTR]...

Needs a running iris (this tree's build) with the guest logged in and
glstate built in /tmp/gltest (see clanker/mgras/gl-state-map.md).

Bit runs: run r (0..4) answers mask_r = OR of 1 << k for every k whose bit
r is set; plus all ones and zero. A boolean query that tests bit k of its
word reads true exactly in the runs whose r is set in k, so the five runs
spell k (all-ones vs zero tells "bit 0" from "not from this word").
Enum runs: answers 0..9; each query's results give its decoding table.
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MON = [sys.executable, str(ROOT / "tools/mon.py")]
SH = [sys.executable, str(ROOT / "tools/irix-sh.py")]

DEFAULT_RANGES = "0x1fa-0x1fc,0x23a-0x240,0x29f-0x29f,0x4df-0x4e0,0xafb-0xb00,0xb42-0xb42,0xb8b-0xb8f,0xb9c-0xbc1"
DEFAULT_ONLY = ["enable/", "fog/MODE", "shade/", "lightmodel/LOCAL", "lightmodel/TWO", "colormaterial/",
                "raster/CULL", "raster/FRONT", "raster/POLYGON_MODE", "raster/LINE_STIPPLE", "frag/ALPHA",
                "frag/STENCIL", "frag/DEPTH", "frag/BLEND_FUNC", "frag/BLEND_EQ", "frag/LOGIC", "fb/"]


def mon(*cmds):
    return subprocess.run(MON + list(cmds), capture_output=True, text=True).stdout


def guest(cmd, timeout=900):
    env = dict(os.environ)
    env.setdefault("IRIS_ROOT_PASSWORD", "p00pp00p")
    return subprocess.run(SH + ["--timeout", str(timeout), "run", cmd], capture_output=True, text=True, env=env).stdout


def run(outdir, tag, value, ranges, only):
    mon("mgras probe off")
    for lo, hi in ranges:
        out = mon("mgras probe %#x %#x %#x" % (lo, hi, value))
        if "answer" not in out:
            raise SystemExit("mgras probe failed: " + out)
    # IRIX's telnet session truncates long command lines: a script.
    args = " ".join("--only '%s'" % o for o in only)
    (ROOT / "shared" / "probe-run.sh").write_text("#!/bin/sh\ncd /tmp/gltest && ./glstate -v %s > /shared/probe-$1.txt 2>&1\n" % args)
    guest("sh /shared/probe-run.sh %s" % tag)
    mon("mgras probe off")
    src = ROOT / "shared" / ("probe-%s.txt" % tag)
    dst = Path(outdir) / ("%s.txt" % tag)
    dst.write_text(src.read_text())
    return dst


def parse(path):
    """case name -> 'got' text of its first failing check (or 'PASS')."""
    res = {}
    for line in open(path):
        m = re.match(r"(PASS|FAIL|SKIP|XFAIL|XPASS) #\d+ ([^:\s]+):?(.*)", line.strip())
        if not m:
            continue
        g = re.search(r"got \[?([^\]]*)\]?", m.group(3))
        res[m.group(2)] = "PASS" if m.group(1) in ("PASS", "XPASS") else (g.group(1) if g else m.group(3).strip())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--ranges", default=DEFAULT_RANGES)
    ap.add_argument("--only", action="append")
    ap.add_argument("--enum-max", type=int, default=9)
    a = ap.parse_args()
    ranges = [tuple(int(x, 16) for x in r.split("-")) for r in a.ranges.split(",")]
    only = a.only or DEFAULT_ONLY
    os.makedirs(a.outdir, exist_ok=True)
    masks = [sum(1 << k for k in range(32) if k >> r & 1) for r in range(5)]
    runs = [("bit%d" % r, m) for r, m in enumerate(masks)] + [("ones", 0xFFFFFFFF), ("zero", 0)]
    runs += [("k%d" % k, k) for k in range(1, a.enum_max + 1)]
    results = {}
    for tag, v in runs:
        print("probe %s = %#x" % (tag, v), flush=True)
        results[tag] = parse(run(a.outdir, tag, v, ranges, only))
    # Report: per case, the results across runs.
    names = sorted({n for r in results.values() for n in r})
    with open(Path(a.outdir) / "report.txt", "w") as f:
        for n in names:
            row = ["%s=%s" % (t, results[t].get(n, "-")) for t, _ in runs]
            f.write("%s\n    %s\n" % (n, "  ".join(row)))
    print("report:", Path(a.outdir) / "report.txt")


if __name__ == "__main__":
    main()
