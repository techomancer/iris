#!/usr/bin/env python3
"""Decode a rex3 bus log (produced by `rex buslog on`) and summarise GO draws.

Usage:
    python3 decode_rex3_log.py menu.log [--pup] [--rgb N] [--all]

Options:
    --pup      Show PUP/OLAY/CID draws (default: shown)
    --rgb N    Show first N RGB draws (default: 40)
    --all      Show all draws regardless of plane
"""

import re, sys
from collections import Counter

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "menu.log"

OPCODES  = {0:'NOOP', 1:'READ', 2:'DRAW', 3:'SCR2SCR'}
ADRMODES = {0:'SPAN', 1:'BLOCK', 2:'ILINE', 3:'FLINE', 4:'ALINE'}
PLANES   = {0:'NONE', 1:'RGB', 2:'RGBA', 4:'OLAY', 5:'PUP', 6:'CID'}
DEPTHS   = {0:'4bpp', 1:'8bpp', 2:'12bpp', 3:'24bpp'}
LOGICOPS = {0:'ZERO',1:'AND',2:'ANDR',3:'SRC',4:'ANDI',5:'DST',6:'XOR',7:'OR',
            8:'NOR',9:'XNOR',10:'NDST',11:'ORR',12:'NSRC',13:'ORI',14:'NAND',15:'ONE'}
# Per REX3 spec Tables 13/14. SFACTOR 010/011 = dest color; DFACTOR 010/011 = source
# color; 100/101 = source alpha in both. 110/111 undefined.
SFACTOR_NAMES = ['ZERO','ONE','DC','MDC','SA','MSA','?6','?7']
DFACTOR_NAMES = ['ZERO','ONE','SC','MSC','SA','MSA','?6','?7']


def _bit(v, n):
    return (v >> n) & 1


def _bits(v, hi, lo):
    return (v >> lo) & ((1 << (hi - lo + 1)) - 1)


def decode_dm0(v):
    # Bit layout mirrors DrawMode0 in src/dev/ng1/rex3.rs — keep in sync.
    op  = OPCODES.get(_bits(v, 1, 0), '?')
    adr = ADRMODES.get(_bits(v, 4, 2), '?')
    flags = []
    if _bit(v, 5):  flags.append('DOSETUP')
    if _bit(v, 6):  flags.append('COLORHOST')
    if _bit(v, 7):  flags.append('ALPHAHOST')
    if _bit(v, 8):  flags.append('STOPONX')
    if _bit(v, 9):  flags.append('STOPONY')
    if _bit(v, 10): flags.append('SKIPFIRST')
    if _bit(v, 11): flags.append('SKIPLAST')
    if _bit(v, 12): flags.append('ENZPAT')
    if _bit(v, 13): flags.append('ENLSPAT')
    if _bit(v, 14): flags.append('LSADVLAST')
    if _bit(v, 15): flags.append('LEN32')
    if _bit(v, 16): flags.append('ZPOPAQUE')
    if _bit(v, 17): flags.append('LSOPAQUE')
    if _bit(v, 18): flags.append('SHADE')
    if _bit(v, 19): flags.append('LRONLY')
    if _bit(v, 20): flags.append('XYOFFSET')
    if _bit(v, 21): flags.append('CICLAMP')
    if _bit(v, 22): flags.append('ENDPTFILT')
    if _bit(v, 23): flags.append('YSTRIDE')
    return f"{op} {adr} {' '.join(flags)}"


def decode_dm1(v):
    # Bit layout mirrors DrawMode1 in src/dev/ng1/rex3.rs — keep in sync.
    planes = PLANES.get(_bits(v, 2, 0), '?')
    depth  = DEPTHS.get(_bits(v, 4, 3), '?')
    hdepth = {0: '12bpp', 1: '8bpp', 2: '4bpp', 3: '32bpp'}.get(_bits(v, 9, 8), '?')
    cmp_   = _bits(v, 14, 12)
    logop  = LOGICOPS.get(_bits(v, 31, 28), '?')
    sf = SFACTOR_NAMES[_bits(v, 21, 19)]
    df = DFACTOR_NAMES[_bits(v, 24, 22)]
    flags = []
    if _bit(v, 5):  flags.append('DBLSRC')
    if _bit(v, 6):  flags.append('YFLIP')
    if _bit(v, 7):  flags.append('RWPACKED')
    if _bit(v, 10): flags.append('RWDOUBLE')
    if _bit(v, 11): flags.append('SWAPEND')
    flags.append('RGB' if _bit(v, 15) else 'CI')
    if _bit(v, 16): flags.append('DITHER')
    if _bit(v, 17): flags.append('FASTCLEAR')
    if _bit(v, 18): flags.append(f'BLEND({sf}+{df})')
    if _bit(v, 25): flags.append('BACKBLEND')
    if _bit(v, 26): flags.append('PREFETCH')
    if _bit(v, 27): flags.append('BLENDALPHA')
    return f"planes={planes} {depth} host:{hdepth} cmp:{cmp_} logicop={logop} {' '.join(flags)}"


def parse_log(path):
    state = {}
    draws = []
    with open(path) as f:
        for line in f:
            m = re.search(r'(?:Write32|Process).*Offset ([0-9a-fA-F]+).*Val ([0-9a-fA-F]+)', line)
            if not m:
                continue
            off = int(m.group(1), 16)
            val = int(m.group(2), 16)
            reg = off & ~0x800  # strip GO bit

            # XYSTARTI/XYENDI can appear at offset 0x0150/0x0154 or 0x0950/0x0954 (GO variants)
            state[reg] = val

            if off & 0x800:  # GO bit set
                dm0    = state.get(0x0004, 0)
                dm1    = state.get(0x0000, 0)
                xsi    = state.get(0x0150, 0)
                xei    = state.get(0x0154, 0)
                x0i    = (xsi >> 16) & 0xffff
                y0i    = xsi & 0xffff
                x1i    = (xei >> 16) & 0xffff
                y1i    = xei & 0xffff
                plane  = PLANES.get(dm1 & 7, '?')
                cr     = state.get(0x0200, 0)  # COLORRED
                cg     = state.get(0x0208, 0)  # COLORGRN
                cb     = state.get(0x020c, 0)  # COLORBLUE
                cback  = state.get(0x0018, 0)  # COLORBACK
                cvram  = state.get(0x001c, 0)  # COLORVRAM
                wrmask = state.get(0x0220, 0xffffff)  # WRMASK
                draws.append(dict(
                    dm0=dm0, dm1=dm1,
                    x0=x0i, y0=y0i, x1=x1i, y1=y1i,
                    plane=plane, cr=cr, cg=cg, cb=cb,
                    colorback=cback, colorvram=cvram, wrmask=wrmask,
                ))
    return draws


def main():
    show_all  = '--all'  in sys.argv
    show_pup  = '--pup'  in sys.argv or not show_all
    rgb_limit = 40
    for i, a in enumerate(sys.argv):
        if a == '--rgb' and i+1 < len(sys.argv):
            rgb_limit = int(sys.argv[i+1])

    draws = parse_log(LOG_FILE)

    print(f"Total GO draws: {len(draws)}\n")
    plane_counts = Counter(d['plane'] for d in draws)
    print("By plane:", dict(plane_counts))
    print()

    if show_pup or show_all:
        pup_draws = [d for d in draws if d['plane'] in ('PUP','OLAY','CID')]
        print(f"=== PUP/OLAY/CID draws ({len(pup_draws)}) ===")
        for d in pup_draws:
            print(f"  {d['plane']} ({d['x0']},{d['y0']})-({d['x1']},{d['y1']})  "
                  f"dm0={d['dm0']:#010x}  dm1={d['dm1']:#010x}")
            print(f"    {decode_dm0(d['dm0'])}")
            print(f"    {decode_dm1(d['dm1'])}")
            print(f"    colorvram={d['colorvram']:#010x}  colorback={d['colorback']:#010x}  "
                  f"cr={d['cr']:#010x}  cg={d['cg']:#010x}  cb={d['cb']:#010x}  "
                  f"wrmask={d['wrmask']:#010x}")
        print()

    rgb_draws = [d for d in draws if d['plane'] == 'RGB' or show_all]
    print(f"=== RGB draws (showing first {rgb_limit} of {len(rgb_draws)}) ===")
    for d in rgb_draws[:rgb_limit]:
        print(f"  {d['plane']} ({d['x0']},{d['y0']})-({d['x1']},{d['y1']})  "
              f"dm0={d['dm0']:#010x}  dm1={d['dm1']:#010x}")
        print(f"    {decode_dm0(d['dm0'])}")
        print(f"    {decode_dm1(d['dm1'])}")
        print(f"    colorvram={d['colorvram']:#010x}  colorback={d['colorback']:#010x}  "
              f"cr={d['cr']:#010x}  cg={d['cg']:#010x}  cb={d['cb']:#010x}  "
              f"wrmask={d['wrmask']:#010x}")

    # Summary: unique (dm0, dm1) pairs used
    print()
    unique = sorted(set((d['dm0'], d['dm1']) for d in draws))
    print(f"=== Unique (dm0, dm1) pairs: {len(unique)} ===")
    for dm0, dm1 in unique:
        count = sum(1 for d in draws if d['dm0']==dm0 and d['dm1']==dm1)
        print(f"  [{count:4d}x]  dm0={dm0:#010x}  dm1={dm1:#010x}  "
              f"{decode_dm0(dm0)}  |  {decode_dm1(dm1)}")


if __name__ == '__main__':
    main()
