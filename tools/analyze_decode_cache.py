#!/usr/bin/env python3
"""Analyze the IRIS decode cache binary to understand instruction word distribution."""

import struct
import sys
import os
from collections import Counter
from pathlib import Path

def load_cache(path):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'IRDP':
            print(f"Bad magic: {magic}", file=sys.stderr)
            sys.exit(1)
        version = struct.unpack('B', f.read(1))[0]
        count = struct.unpack('<I', f.read(4))[0]
        words = struct.unpack(f'<{count}I', f.read(count * 4))
    print(f"Loaded {len(words)} unique instruction words (version {version})")
    return words

def analyze(words):
    words = list(words)
    n = len(words)

    # --- bit distribution ---
    bit_counts = [0] * 32
    for w in words:
        for i in range(32):
            if w & (1 << i):
                bit_counts[i] += 1

    print("\n=== Bit density (bit 31..0, % of words with bit set) ===")
    for i in range(31, -1, -1):
        bar = '#' * int(bit_counts[i] * 40 / n)
        print(f"  bit {i:2d}: {bit_counts[i]:6d} / {n} ({bit_counts[i]*100/n:5.1f}%)  {bar}")

    # --- upper 16 bits as index ---
    hi16 = [w >> 16 for w in words]
    hi16_counts = Counter(hi16)
    unique_hi16 = len(hi16_counts)
    max_bucket = max(hi16_counts.values())
    avg_bucket = n / unique_hi16
    filled_slots = unique_hi16
    total_slots = 65536

    print(f"\n=== Upper 16-bit index (hi16 → bucket) ===")
    print(f"  Unique hi16 values : {unique_hi16} / {total_slots} ({unique_hi16*100/total_slots:.1f}% filled)")
    print(f"  Max bucket size    : {max_bucket}")
    print(f"  Avg bucket size    : {avg_bucket:.2f}")

    bucket_size_dist = Counter(hi16_counts.values())
    print(f"  Bucket size distribution:")
    for size in sorted(bucket_size_dist):
        print(f"    {size:3d} entries: {bucket_size_dist[size]:5d} buckets")

    # --- upper N bits ---
    print(f"\n=== Index width sweep (unique keys / total slots / max bucket) ===")
    for bits in [8, 10, 12, 13, 14, 15, 16, 17, 18]:
        shifted = [w >> (32 - bits) for w in words]
        c = Counter(shifted)
        slots = 1 << bits
        print(f"  top {bits:2d} bits: {len(c):6d} unique / {slots:6d} slots ({len(c)*100/slots:5.1f}% fill)  max_bucket={max(c.values())}  avg={n/len(c):.2f}")

    # --- opcode field (bits 31:26) ---
    opcodes = Counter(w >> 26 for w in words)
    print(f"\n=== Opcode distribution (bits 31:26) ===")
    for op, cnt in sorted(opcodes.items(), key=lambda x: -x[1]):
        bar = '#' * int(cnt * 40 / n)
        print(f"  op 0x{op:02x} ({op:2d}): {cnt:6d} ({cnt*100/n:5.1f}%)  {bar}")

    # --- SPECIAL funct field (bits 5:0) when op==0 ---
    specials = [w & 0x3f for w in words if (w >> 26) == 0]
    if specials:
        sc = Counter(specials)
        print(f"\n=== SPECIAL funct (bits 5:0, op=0), {len(specials)} words ===")
        for funct, cnt in sorted(sc.items(), key=lambda x: -x[1])[:20]:
            print(f"  funct 0x{funct:02x} ({funct:2d}): {cnt:5d} ({cnt*100/len(specials):5.1f}%)")

    # --- load/store immediate spread (bits 15:0 when op is load/store) ---
    LOAD_STORE_OPS = {0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x2a,0x2b,0x2c,0x2d,0x2e,0x2f,
                      0x30,0x31,0x35,0x37,0x38,0x39,0x3d,0x3f}
    ls_imms = [w & 0xffff for w in words if (w >> 26) in LOAD_STORE_OPS]
    if ls_imms:
        unique_imms = len(set(ls_imms))
        print(f"\n=== Load/store immediate (bits 15:0): {len(ls_imms)} words, {unique_imms} unique values ===")

    # --- how many words share hi16 with another word ---
    shared = sum(v for v in hi16_counts.values() if v > 1)
    print(f"\n=== Hi16 sharing ===")
    print(f"  Words in a shared hi16 bucket : {shared} ({shared*100/n:.1f}%)")
    print(f"  Words alone in their bucket   : {n - shared} ({(n-shared)*100/n:.1f}%)")

    # --- perfect hash candidate: would hi16 table + linear scan work? ---
    print(f"\n=== Hi16 table + linear scan feasibility ===")
    print(f"  Table size   : {total_slots} entries × 8B (ptr+len) = {total_slots*8//1024} KB overhead")
    print(f"  Max scan len : {max_bucket} (worst case comparisons per lookup)")
    print(f"  99th pct scan: ", end='')
    sizes = sorted(hi16_counts.values())
    p99 = sizes[int(len(sizes)*0.99)]
    print(f"{p99}")

def analyze_hash(words, label, hashfn, table_size):
    buckets = Counter(hashfn(w) % table_size for w in words)
    n = len(words)
    sizes = sorted(buckets.values())
    empty = table_size - len(buckets)
    max_chain = max(sizes)
    avg_chain = n / len(buckets)
    p50 = sizes[len(sizes)//2]
    p90 = sizes[int(len(sizes)*0.90)]
    p99 = sizes[int(len(sizes)*0.99)]
    # Expected comparisons per lookup (avg chain length for occupied slots,
    # weighted by lookup frequency assuming uniform access)
    # E[comparisons] = sum(k * (k/n)) for each bucket of size k = sum(k^2)/n
    expected_cmps = sum(v*v for v in buckets.values()) / n
    print(f"\n=== Hash: {label}, table={table_size} ===")
    print(f"  Occupied slots : {len(buckets)} / {table_size} ({len(buckets)*100/table_size:.1f}%)")
    print(f"  Empty slots    : {empty} ({empty*100/table_size:.1f}%)")
    print(f"  Max chain      : {max_chain}")
    print(f"  Avg chain      : {avg_chain:.2f}")
    print(f"  p50/p90/p99    : {p50} / {p90} / {p99}")
    print(f"  E[comparisons] : {expected_cmps:.2f}  (lower=better, 1.0=perfect)")

if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / '.iris' / 'decode-cache.bin'
    words = load_cache(path)
    analyze(words)

    print("\n" + "="*60)
    print("HASH QUALITY ANALYSIS")
    print("="*60)

    hashes = [
        ("raw ^ (raw>>16)",          lambda w: w ^ (w >> 16),                    65536),
        ("raw ^ (raw>>16) 128K",     lambda w: w ^ (w >> 16),                   131072),
        ("(raw>>16) only",           lambda w: w >> 16,                           65536),
        ("raw * 0x9e3779b9 >> 16",   lambda w: (w * 0x9e3779b9 & 0xffffffff) >> 16, 65536),
        ("raw * 0x9e3779b9 >> 15",   lambda w: (w * 0x9e3779b9 & 0xffffffff) >> 15, 131072),
        ("(raw^(raw>>13)^(raw>>26))",lambda w: (w ^ (w>>13) ^ (w>>26)) & 0xffff, 65536),
        ("fibhash32 >> 16",          lambda w: ((w * 0x61C88647) & 0xffffffff) >> 16, 65536),
    ]
    for label, fn, size in hashes:
        analyze_hash(words, label, fn, size)
