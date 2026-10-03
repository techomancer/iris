#!/usr/bin/env python3
"""Decode REX JIT compiled/dispatch log lines.

Usage:
    # Pipe emulator output:
    iris ... 2>&1 | python3 decode_jit_entries.py

    # Or pass a log file:
    python3 decode_jit_entries.py iris.log

Reads lines matching "REX JIT: compiled dm0=... dm1=..." or
"REX JIT dispatch: dm0=... dm1=..." and prints decoded drawmode fields.
Also accepts bare hex pairs as arguments:
    python3 decode_jit_entries.py 0x00001106 0x30097011
"""

import re, sys

OPCODES  = {0:'NOOP', 1:'READ', 2:'DRAW', 3:'SCR2SCR'}
ADRMODES = {0:'SPAN', 1:'BLOCK', 2:'ILINE', 3:'FLINE', 4:'ALINE'}
PLANES   = {0:'NONE', 1:'RGB', 2:'RGBA', 4:'OLAY', 5:'PUP', 6:'CID'}
DEPTHS   = {0:'4bpp', 1:'8bpp', 2:'12bpp', 3:'24bpp'}
LOGICOPS = {0:'ZERO',1:'AND',2:'ANDR',3:'SRC',4:'ANDI',5:'DST',6:'XOR',7:'OR',
            8:'NOR',9:'XNOR',10:'NDST',11:'ORR',12:'NSRC',13:'ORI',14:'NAND',15:'ONE'}
SF_NAMES = ['ZERO','ONE','DCOL','1-DCOL','DALPHA','1-DALPHA','SALPHA','1-SALPHA']


def decode_dm0(v):
    op  = OPCODES.get(v & 3, '?')
    adr = ADRMODES.get((v >> 2) & 7, '?')
    flags = []
    if v & (1 << 5):  flags.append('DOSETUP')
    if v & (1 << 6):  flags.append('COLORHOST')
    if v & (1 << 7):  flags.append('ALPHAHOST')
    if v & (1 << 8):  flags.append('STOPONX')
    if v & (1 << 9):  flags.append('STOPONY')
    if v & (1 << 10): flags.append('SKIPFIRST')
    if v & (1 << 11): flags.append('SKIPLAST')
    if v & (1 << 12): flags.append('ENZPAT')
    if v & (1 << 13): flags.append('ENLSPAT')
    if v & (1 << 14): flags.append('LSADVLAST')
    if v & (1 << 15): flags.append('LEN32')
    if v & (1 << 16): flags.append('ZPOPAQUE')
    if v & (1 << 17): flags.append('LSOPAQUE')
    if v & (1 << 18): flags.append('SHADE')
    if v & (1 << 19): flags.append('LRONLY')
    if v & (1 << 20): flags.append('XYOFFSET')
    if v & (1 << 21): flags.append('CICLAMP')
    if v & (1 << 23): flags.append('YSTRIDE')
    return f"{op} {adr}" + (f" [{' '.join(flags)}]" if flags else "")


def decode_dm1(v):
    planes = PLANES.get(v & 7, '?')
    depth  = DEPTHS.get((v >> 3) & 3, '?')
    hdepth = {0:'12bpp', 1:'8bpp', 2:'4bpp', 3:'32bpp'}.get((v >> 9) & 3, '?')
    logop  = LOGICOPS.get((v >> 28) & 0xf, '?')
    flags = []
    if v & (1 << 5):  flags.append('DBLSRC')
    if v & (1 << 15): flags.append('RGB')
    if v & (1 << 16): flags.append('DITHER')
    if v & (1 << 17): flags.append('FASTCLEAR')
    if v & (1 << 18): flags.append('BLEND')
    if v & (1 << 25): flags.append('BACKBLEND')
    sf = SF_NAMES[(v >> 19) & 7]
    df = SF_NAMES[(v >> 22) & 7]
    flag_str = (' [' + ' '.join(flags) + ']') if flags else ''
    return f"planes={planes} {depth} host:{hdepth} logop={logop}{flag_str} sf={sf} df={df}"


def decode_pair(dm0, dm1, prefix=''):
    print(f"{prefix}dm0={dm0:#010x} dm1={dm1:#010x}")
    print(f"  DM0: {decode_dm0(dm0)}")
    print(f"  DM1: {decode_dm1(dm1)}")


# --- bare hex pair mode ---
args = sys.argv[1:]
if len(args) >= 2 and args[0].startswith('0x'):
    try:
        dm0 = int(args[0], 16)
        dm1 = int(args[1], 16)
        decode_pair(dm0, dm1)
        sys.exit(0)
    except ValueError:
        pass

# --- log file / stdin mode ---
pattern = re.compile(
    r'REX JIT.*?dm0=(0x[0-9a-fA-F]+)\s+dm1=(0x[0-9a-fA-F]+)'
    r'(?:\s+\((\d+)B,\s*total:\s*(\d+)\))?'
)

src = open(args[0]) if args else sys.stdin
seen = []
for line in src:
    m = pattern.search(line)
    if not m:
        continue
    dm0 = int(m.group(1), 16)
    dm1 = int(m.group(2), 16)
    size = m.group(3)
    total = m.group(4)
    prefix = f"[#{total} {size}B] " if total else ''
    decode_pair(dm0, dm1, prefix)
    seen.append((dm0, dm1))
    print()

if len(seen) > 1:
    print(f"=== {len(seen)} entries ===")
