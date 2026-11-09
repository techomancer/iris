#!/usr/bin/env python3
"""check.py CELL LOG: judge a cpu-tests run by the names of its failing tests.

CELL is cpu-engine, e.g. r4400-interp. [CELL] in expected-failures.txt lists
the tests that fail today; the run passes when exactly those fail and every
planned test reported.
"""
import os
import re
import sys

cell, path = sys.argv[1:3]
log = open(path, errors="replace").read()

want, section = set(), None
for line in open(os.path.join(os.path.dirname(__file__), "..", "expected-failures.txt")):
    line = line.split("#")[0].strip()
    if line.startswith("["):
        section = line.strip("[]")
    elif line and section == cell:
        want.add(line)

plan = re.search(r"^1\.\.(\d+)$", log, re.M)
results = re.findall(r"^(not )?ok \d+ - (\S+)", log, re.M)
failed = {name for bad, name in results if bad}

errs = []
if not plan:
    errs.append("no plan line: the guest never reached its first test")
else:
    errs += [f"{n} failed and is not an expected failure on {cell}" for n in sorted(failed - want)]
    if len(results) != int(plan.group(1)):
        errs.append(f"{len(results)} of {plan.group(1)} tests reported: the run stopped early")
    else:
        errs += [f"{n} now passes on {cell}: remove it from [{cell}] in expected-failures.txt"
                 for n in sorted(want - failed)]
sys.exit("\n".join(f"check: {e}" for e in errs) or print(f"check: {cell} ok"))
