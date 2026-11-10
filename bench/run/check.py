#!/usr/bin/env python3
"""check.py CELL CPU: judge a saved bench result by what could be wrong with it.

CELL is cpu-engine, e.g. r4400-jitv2; CPU is the --cpu the emulator was given.
The run is saved as build/results/CELL.json by `iris-bench run`.

CI gates on accuracy, never on speed: a shared runner's throughput varies by
more than most regressions.
"""
import json
import sys

cell, cpu = sys.argv[1:3]
try:
    run = json.load(open(f"build/results/{cell}.json"))
except OSError:
    sys.exit(f"check: no result for {cell}: the run did not finish")

errs = []

# A build whose features did not take runs the wrong CPU and every result is
# mislabelled. The guest's banner is authoritative because it reads PRId.
want = cpu.split("_")[0].upper()
if run["machine"]["cpu"] != want:
    errs.append(f"guest reports {run['machine']['cpu']}, expected {want}")
if not run["machine"]["timebase"]:
    errs.append("no host time base: every timing is a guess")

# A kernel that faults is stepped over by the exception dispatcher and still
# reports a throughput, so unexpected exceptions are checked as well as the
# checksums themselves.
expect_exc = {"sys/exception", "sys/tlb_miss"}
for r in run["rows"]:
    if r["status"] == "MISMATCH":
        errs.append(f"{r['name']} checksum {r['checksum']} != {r['golden']}")
    if r["exc"] and r["name"] not in expect_exc:
        errs.append(f"{r['name']} took {r['exc']} unexpected exceptions")

mips = run["total_icount"] * 1e3 / max(run["total_ns"], 1)
print(f"check: {cell}: {run['matched']}/{run['checked']} matched, "
      f"{mips:.1f} guest MIPS, {run['total_ns'] / 1e9:.1f} s timed")
sys.exit("\n".join(f"check: {e}" for e in errs) or None)
