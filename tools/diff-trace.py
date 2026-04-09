#!/usr/bin/env python3
"""Compare two IRIS JIT trace files and report the first divergence.

Usage:
    python3 diff-trace.py trace-interp.bin trace-jit.bin

Trace files are produced by setting IRIS_JIT_TRACE=<path> when running iris.
Each record is 48 bytes:
    u64 insn_count
    u64 pc
    u64 cp0_count
    u32 cp0_status
    u32 cp0_cause
    u8  in_delay_slot
    u8[7] pad
    u64 gpr_hash
"""

import struct
import sys

RECORD_SIZE = 48
RECORD_FMT = '<QQQIIBxxxxxxxQ'  # little-endian, 7 pad bytes

assert struct.calcsize(RECORD_FMT) == RECORD_SIZE, f"bad format size: {struct.calcsize(RECORD_FMT)}"

FIELDS = ['insn_count', 'pc', 'cp0_count', 'cp0_status', 'cp0_cause', 'in_delay_slot', 'gpr_hash']

def read_records(path):
    records = []
    with open(path, 'rb') as f:
        while True:
            data = f.read(RECORD_SIZE)
            if len(data) < RECORD_SIZE:
                break
            vals = struct.unpack(RECORD_FMT, data)
            records.append(dict(zip(FIELDS, vals)))
    return records

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <trace-a.bin> <trace-b.bin>")
        sys.exit(1)

    path_a, path_b = sys.argv[1], sys.argv[2]
    recs_a = read_records(path_a)
    recs_b = read_records(path_b)

    print(f"Trace A ({path_a}): {len(recs_a)} records")
    print(f"Trace B ({path_b}): {len(recs_b)} records")

    if not recs_a or not recs_b:
        print("ERROR: one or both traces are empty")
        sys.exit(1)

    # Match records by closest insn_count
    b_idx = 0
    divergences = 0
    max_report = 10

    for a in recs_a:
        # Find closest record in B
        while b_idx < len(recs_b) - 1:
            curr_diff = abs(recs_b[b_idx]['insn_count'] - a['insn_count'])
            next_diff = abs(recs_b[b_idx + 1]['insn_count'] - a['insn_count'])
            if next_diff < curr_diff:
                b_idx += 1
            else:
                break

        b = recs_b[b_idx]

        # Skip if instruction counts are too far apart (> 2x BATCH_SIZE)
        if abs(a['insn_count'] - b['insn_count']) > 20000:
            continue

        # Compare fields (skip insn_count since it won't match exactly)
        mismatches = []
        for field in FIELDS[1:]:  # skip insn_count
            va, vb = a[field], b[field]
            if va != vb:
                mismatches.append(field)

        if mismatches:
            divergences += 1
            if divergences <= max_report:
                print(f"\nDIVERGE at insn_count ~{a['insn_count']} (A={a['insn_count']}, B={b['insn_count']}):")
                for field in FIELDS[1:]:
                    va, vb = a[field], b[field]
                    status = "MISMATCH" if field in mismatches else "ok"
                    if field in ('cp0_status', 'cp0_cause'):
                        print(f"  {field:15s}: A={va:08x}  B={vb:08x}  {status}")
                    elif field == 'in_delay_slot':
                        print(f"  {field:15s}: A={va}  B={vb}  {status}")
                    else:
                        print(f"  {field:15s}: A={va:016x}  B={vb:016x}  {status}")

    if divergences == 0:
        print("\nTraces match (no divergences found)")
    else:
        shown = min(divergences, max_report)
        print(f"\nTotal: {divergences} divergences ({shown} shown)")

    sys.exit(1 if divergences > 0 else 0)

if __name__ == '__main__':
    main()
