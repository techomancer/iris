#!/usr/bin/env python3
"""diff-hw.py — classify a real-hardware run against the emulator's own.

Real hardware is the oracle (PLAN.md §6, priority 4). Every per-test difference
falls into one of three buckets, and which bucket decides where it gets written
up:

  IRIS-BUG    fails in IRIS, passes on hardware  -> docs/findings.md, promoted
                                                    from suspected to confirmed
  TEST-BUG    fails on both                      -> docs/gotchas.md, the test is
                                                    wrong and the emulator right
  IRIS-WRONG  passes in IRIS, fails on hardware  -> new finding; IRIS is wrong in
                                                    a way the suite rewards

Both inputs must come from the *same* ELF, or the diff means nothing.

    run/diff-hw.py build/emu-r4400.log cputest-hw-r4400.log
"""
import argparse
import re
import sys

# cp0/count_writable derives from CP0 Count, which IRIS anchors to host
# wallclock rather than to guest instructions, so it flakes under load. It is
# excluded by default: on hardware Count is exactly half the pipeline clock and
# deterministic, so a difference here says nothing about the CPU model.
FLAKY = {"cp0/count_writable"}


def parse(path):
    """Read a suite log into {test: PASS|FAIL|SKIP}.

    A test that prints diagnostics before its verdict puts PASS/FAIL on a later
    line, so the name stays current until a verdict token is seen.
    """
    res, cur = {}, None
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            # The name can appear mid-line, after a diagnostic a test printed
            # before its verdict: "[cop2: ...]excep/exl_set_and_cleared ... PASS"
            m = re.search(r"([A-Za-z0-9_]+/[A-Za-z0-9_]+) \.{3,}\s*(.*)$", line)
            rest = m.group(2).strip() if m else line.strip()
            if m:
                cur = m.group(1)
            if not cur:
                continue
            if rest.endswith("PASS"):
                res[cur], cur = "PASS", None
            elif rest.endswith("FAIL"):
                res[cur], cur = "FAIL", None
            elif rest.startswith("skip"):
                res[cur], cur = "SKIP", None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emulator")
    ap.add_argument("hardware")
    ap.add_argument("--include-flaky", action="store_true")
    args = ap.parse_args()

    emu, hw = parse(args.emulator), parse(args.hardware)
    if not hw:
        sys.exit("no test results in %s" % args.hardware)

    buckets = {"IRIS-BUG": [], "TEST-BUG": [], "IRIS-WRONG": [], "AGREE-PASS": []}
    skipped = []
    for name in sorted(set(emu) | set(hw)):
        e, h = emu.get(name), hw.get(name)
        if e is None or h is None:
            skipped.append((name, "only in %s" % ("hardware" if e is None else "emulator")))
            continue
        if "SKIP" in (e, h):
            continue
        if not args.include_flaky and name in FLAKY:
            skipped.append((name, "known non-deterministic under IRIS"))
            continue
        if e == "FAIL" and h == "PASS":
            buckets["IRIS-BUG"].append(name)
        elif e == "FAIL" and h == "FAIL":
            buckets["TEST-BUG"].append(name)
        elif e == "PASS" and h == "FAIL":
            buckets["IRIS-WRONG"].append(name)
        else:
            buckets["AGREE-PASS"].append(name)

    where = {
        "IRIS-BUG": "confirmed emulator bug -> docs/findings.md",
        "TEST-BUG": "the test is wrong -> docs/gotchas.md",
        "IRIS-WRONG": "NEW: IRIS wrong where the suite passes it",
    }
    for b in ("IRIS-WRONG", "IRIS-BUG", "TEST-BUG"):
        print("%s (%d) — %s" % (b, len(buckets[b]), where[b]))
        for n in buckets[b]:
            print("    %s" % n)
        print()
    print("both pass: %d" % len(buckets["AGREE-PASS"]))
    for n, why in skipped:
        print("not compared: %s (%s)" % (n, why))

    return 1 if buckets["IRIS-WRONG"] else 0


if __name__ == "__main__":
    sys.exit(main())
