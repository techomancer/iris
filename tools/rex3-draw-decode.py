#!/usr/bin/env python3
"""Decode a REX3 DRAWMODE0/DRAWMODE1 register pair into human-readable fields.

Bit layout mirrors the `DrawMode0`/`DrawMode1` bitfields in src/dev/ng1/rex3.rs — keep
in sync if those change.

Usage:
    python3 rex3-draw-decode.py <dm0> <dm1> [<dm0> <dm1> ...]

    Values may be hex (0x... or bare hex like 000000c6) or decimal.

Examples:
    python3 rex3-draw-decode.py 0xc6 0x3009d7b1
    python3 rex3-draw-decode.py 000000c6 3009d7b1 000000c2 3009d7b1
"""

import sys

OPCODES  = {0: 'NOOP', 1: 'READ', 2: 'DRAW', 3: 'SCR2SCR'}
ADRMODES = {0: 'SPAN', 1: 'BLOCK', 2: 'ILINE', 3: 'FLINE', 4: 'ALINE'}
PLANES   = {0: 'NONE', 1: 'RGB', 2: 'RGBA', 4: 'OLAY', 5: 'PUP', 6: 'CID'}
DEPTHS   = {0: '4bpp', 1: '8bpp', 2: '12bpp', 3: '24bpp'}
HOSTDEPTHS = {0: '12bpp', 1: '8bpp', 2: '4bpp', 3: '32bpp'}
LOGICOPS = {0: 'ZERO', 1: 'AND', 2: 'ANDR', 3: 'SRC', 4: 'ANDI', 5: 'DST', 6: 'XOR', 7: 'OR',
            8: 'NOR', 9: 'XNOR', 10: 'NDST', 11: 'ORR', 12: 'NSRC', 13: 'ORI', 14: 'NAND', 15: 'ONE'}
SF_NAMES = ['ZERO', 'ONE', 'DCOL', '1-DCOL', 'DALPHA', '1-DALPHA', 'SALPHA', '1-SALPHA']


def bit(v, n):
    return (v >> n) & 1


def bits(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def decode_dm0(v):
    op  = OPCODES.get(bits(v, 1, 0), '?')
    adr = ADRMODES.get(bits(v, 4, 2), '?')
    flags = []
    if bit(v, 5):  flags.append('DOSETUP')
    if bit(v, 6):  flags.append('COLORHOST')
    if bit(v, 7):  flags.append('ALPHAHOST')
    if bit(v, 8):  flags.append('STOPONX')
    if bit(v, 9):  flags.append('STOPONY')
    if bit(v, 10): flags.append('SKIPFIRST')
    if bit(v, 11): flags.append('SKIPLAST')
    if bit(v, 12): flags.append('ENZPAT')
    if bit(v, 13): flags.append('ENLSPAT')
    if bit(v, 14): flags.append('LSADVLAST')
    if bit(v, 15): flags.append('LEN32')
    if bit(v, 16): flags.append('ZPOPAQUE')
    if bit(v, 17): flags.append('LSOPAQUE')
    if bit(v, 18): flags.append('SHADE')
    if bit(v, 19): flags.append('LRONLY')
    if bit(v, 20): flags.append('XYOFFSET')
    if bit(v, 21): flags.append('CICLAMP')
    if bit(v, 22): flags.append('ENDPTFILT')
    if bit(v, 23): flags.append('YSTRIDE')
    return f"{op} {adr} {' '.join(flags)}".rstrip()


def decode_dm1(v):
    planes   = PLANES.get(bits(v, 2, 0), '?')
    depth    = DEPTHS.get(bits(v, 4, 3), '?')
    hdepth   = HOSTDEPTHS.get(bits(v, 9, 8), '?')
    cmp_     = bits(v, 14, 12)
    logicop  = LOGICOPS.get(bits(v, 31, 28), '?')
    sf = SF_NAMES[bits(v, 21, 19)]
    df = SF_NAMES[bits(v, 24, 22)]
    flags = []
    if bit(v, 5):  flags.append('DBLSRC')
    if bit(v, 6):  flags.append('YFLIP')
    if bit(v, 7):  flags.append('RWPACKED')
    if bit(v, 10): flags.append('RWDOUBLE')
    if bit(v, 11): flags.append('SWAPEND')
    flags.append('RGB' if bit(v, 15) else 'CI')
    if bit(v, 16): flags.append('DITHER')
    if bit(v, 17): flags.append('FASTCLEAR')
    if bit(v, 18): flags.append(f'BLEND({sf}+{df})')
    if bit(v, 25): flags.append('BACKBLEND')
    if bit(v, 26): flags.append('PREFETCH')
    if bit(v, 27): flags.append('BLENDALPHA')
    return f"planes={planes} {depth} host:{hdepth} cmp:{cmp_} logicop={logicop} {' '.join(flags)}".rstrip()


def parse_val(s):
    return int(s, 16) if s.lower().startswith('0x') else int(s, 16) if any(c in 'abcdefABCDEF' for c in s) else int(s)


def main():
    args = sys.argv[1:]
    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    for i in range(0, len(args), 2):
        dm0 = parse_val(args[i])
        dm1 = parse_val(args[i + 1])
        print(f"DM0={dm0:#010x}  DM1={dm1:#010x}")
        print(f"  dm0: {decode_dm0(dm0)}")
        print(f"  dm1: {decode_dm1(dm1)}")
        print()


if __name__ == '__main__':
    main()
