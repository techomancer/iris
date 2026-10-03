#!/usr/bin/env python3
"""Analyse a `log rex <file>` raw REX3 device log.

Unlike the GFIFO buslog (see analyze_rex3_buslog.py), this log carries both the
CPU-side `Write32` (before the GFIFO) and the `Process` (after it), plus the
emulator's own decoded Draw/Coords lines.  That lets us check whether a draw
executed against the DRAWMODE1 the CPU intended, or against a stale one.

Usage:
    python3 tools/analyze_rex3_raw.py raw.log [--bbox X0 Y0 X1 Y1] [--limit N]
"""
import re, sys
from collections import Counter, defaultdict

LOG = sys.argv[1] if len(sys.argv) > 1 else "raw.log"
def flag(n): return n in sys.argv
def arg(n, d):
    return int(sys.argv[sys.argv.index(n)+1]) if n in sys.argv else d

W  = re.compile(r'REX3 Write32: Offset ([0-9a-f]+) \(Reg ([0-9a-f]+) ([A-Z0-9_]+)\) Val ([0-9a-f]+)')
P  = re.compile(r'REX3 Process: Offset ([0-9a-f]+) \(([A-Z0-9_]+)\) Val ([0-9a-f]+)')
D  = re.compile(r'REX3 Draw: (\S+) (\S+) Mode0=([0-9a-f]+) Mode1=([0-9a-f]+)')
C  = re.compile(r'Coords: Start\(([-\d.]+), ([-\d.]+)\) End\(([-\d.]+), ([-\d.]+)\)')

def bit(v,n): return (v>>n)&1
def bits(v,hi,lo): return (v>>lo)&((1<<(hi-lo+1))-1)
def clamp(c):
    if c & (1<<31): return 0
    v=(c>>11)&0x1FF
    return 0 if v>=0x180 else (0xFF if v>0xFF else v)
def alpha_of(raw): return clamp(raw & 0xFFFFF)

SFN = ['ZERO','ONE','DC','MDC','SA','MSA','?6','?7']
DFN = ['ZERO','ONE','SC','MSC','SA','MSA','?6','?7']
def dm1_str(v):
    s = f"planes={bits(v,2,0)} depth={bits(v,4,3)} cmp={bits(v,14,12)}"
    s += " BLEND(%s+%s)" % (SFN[bits(v,21,19)], DFN[bits(v,24,22)]) if bit(v,18) else " noblend"
    if bit(v,16): s += " DITHER"
    if bit(v,5):  s += " DBLSRC"
    s += f" logicop={bits(v,31,28)}"
    return s

# --- pass 1: walk the log, tracking CPU-side vs processed DRAWMODE1 ---
cpu_dm1 = None       # last DRAWMODE1 the CPU wrote (Write32)
proc_dm1 = None      # last DRAWMODE1 the GFIFO processed
proc_alpha = None    # last COLORALPHA processed
draws = []
stale = 0

for line in open(LOG):
    m = W.search(line)
    if m:
        reg = int(m.group(2),16); val = int(m.group(4),16)
        if reg == 0x0000: cpu_dm1 = val
        continue
    m = P.search(line)
    if m:
        off = int(m.group(1),16); val = int(m.group(3),16)
        if   off == 0x0000: proc_dm1 = val
        elif off == 0x0204: proc_alpha = val
        continue
    m = D.search(line)
    if m:
        adr, op, dm0, dm1 = m.group(1), m.group(2), int(m.group(3),16), int(m.group(4),16)
        draws.append(dict(adr=adr, op=op, dm0=dm0, dm1=dm1,
                          a=alpha_of(proc_alpha) if proc_alpha is not None else None,
                          cpu_dm1=cpu_dm1, x0=None))
        if cpu_dm1 is not None and cpu_dm1 != dm1:
            stale += 1
        continue
    m = C.search(line)
    if m and draws:
        draws[-1]['x0']=float(m.group(1)); draws[-1]['y0']=float(m.group(2))
        draws[-1]['x1']=float(m.group(3)); draws[-1]['y1']=float(m.group(4))

print(f"Draws: {len(draws)}   draws whose Mode1 != last CPU-written DRAWMODE1: {stale}\n")

print("=== Draws by (adrmode, op, Mode1) ===")
for (adr,op,dm1),c in Counter((d['adr'],d['op'],d['dm1']) for d in draws).most_common(12):
    print(f"[{c:7d}x] {adr:8s} {op:8s} dm1={dm1:#010x}  {dm1_str(dm1)}")

print("\n=== Alpha distribution per Mode1 (DRAW only) ===")
per = defaultdict(Counter)
for d in draws:
    if d['op']=='DRAW' and d['a'] is not None:
        per[d['dm1']][d['a']] += 1
for dm1, cnt in sorted(per.items(), key=lambda kv:-sum(kv[1].values())):
    tot=sum(cnt.values()); a0=cnt.get(0,0)
    print(f"dm1={dm1:#010x} n={tot:7d}  alpha==0: {a0:7d} ({100*a0/tot:5.1f}%)  {dm1_str(dm1)}")
    top=', '.join(f"a={a}:{n}" for a,n in cnt.most_common(6))
    print(f"    {top}")

# --- geometry: bounding box per Mode1, on-screen draws only ---
print("\n=== On-screen bounding box per Mode1 ===")
bb=defaultdict(lambda:[1e9,1e9,-1e9,-1e9,0])
for d in draws:
    if d.get('x0') is None: continue
    b=bb[d['dm1']]
    b[0]=min(b[0],d['x0'],d['x1']); b[1]=min(b[1],d['y0'],d['y1'])
    b[2]=max(b[2],d['x0'],d['x1']); b[3]=max(b[3],d['y0'],d['y1']); b[4]+=1
for dm1,b in sorted(bb.items(), key=lambda kv:-kv[1][4]):
    print(f"dm1={dm1:#010x} n={b[4]:7d}  x {b[0]:8.1f}..{b[2]:8.1f}  y {b[1]:8.1f}..{b[3]:8.1f}")
