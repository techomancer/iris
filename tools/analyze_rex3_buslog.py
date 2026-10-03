#!/usr/bin/env python3
"""Analyse a `rex buslog on` log (`reg=XXXX(NAME) val=...` format).

Reconstructs REX3 register state and reports, per GO, the effective DRAWMODE0/1,
span geometry and per-vertex colour/alpha, so you can see what GL asked for.

Internal coordinate format is 21.11 fixed point (see src/dev/ng1/rex3.rs:581-600):
  XSTARTI/XYSTARTI : integer, << 11
  XSTARTF/XENDF    : IEEE float on the wire; HW masks to mantissa bits 22:7
Colours are o12.11 (24-bit), integer part = val >> 11.
"""
import re, sys
from collections import Counter

LOG = sys.argv[1] if len(sys.argv) > 1 else "rex3.log"
def arg(n, d):
    return int(sys.argv[sys.argv.index(n)+1]) if n in sys.argv else d

BIAS = 4096
OPCODES  = {0:'NOOP',1:'READ',2:'DRAW',3:'SCR2SCR'}
ADRMODES = {0:'SPAN',1:'BLOCK',2:'ILINE',3:'FLINE',4:'ALINE'}
PLANES   = {0:'NONE',1:'RGB',2:'RGBA',4:'OLAY',5:'PUP',6:'CID'}
DEPTHS   = {0:'4bpp',1:'8bpp',2:'12bpp',3:'24bpp'}
LOGICOPS = {0:'ZERO',1:'AND',2:'ANDR',3:'SRC',4:'ANDI',5:'DST',6:'XOR',7:'OR',
            8:'NOR',9:'XNOR',10:'NDST',11:'ORR',12:'NSRC',13:'ORI',14:'NAND',15:'ONE'}
# Per REX3 spec Tables 13/14. SFACTOR 010/011 = dest color; DFACTOR 010/011 = source
# color; 100/101 = source alpha in both. 110/111 undefined.
SFACTOR_NAMES = ['ZERO','ONE','DC','MDC','SA','MSA','?6','?7']
DFACTOR_NAMES = ['ZERO','ONE','SC','MSC','SA','MSA','?6','?7']
CMPFN = {0:'never',1:'<',2:'==',3:'<=',4:'>',5:'!=',6:'>=',7:'always'}

def bit(v,n): return (v>>n)&1
def bits(v,hi,lo): return (v>>lo)&((1<<(hi-lo+1))-1)
def from_f(v):   return (v & 0x007fff80)          # 21.11 internal
def from_i(v):   return ((v & 0xffff) - (0x10000 if v & 0x8000 else 0)) << 11
def px(v):       return v / 2048.0                # 21.11 -> pixels (pre-bias)
def col(v):      return (v & 0xffffff) / 2048.0   # o12.11 -> 0..4095-ish

def dec_dm0(v):
    f=[n for b,n in [(5,'DOSETUP'),(6,'COLORHOST'),(7,'ALPHAHOST'),(8,'STOPONX'),
        (9,'STOPONY'),(10,'SKIPFIRST'),(11,'SKIPLAST'),(12,'ENZPAT'),(13,'ENLSPAT'),
        (14,'LSADVLAST'),(15,'LEN32'),(16,'ZPOPAQUE'),(17,'LSOPAQUE'),(18,'SHADE'),
        (19,'LRONLY'),(20,'XYOFFSET'),(21,'CICLAMP'),(22,'ENDPTFILT'),(23,'YSTRIDE')] if bit(v,b)]
    return f"{OPCODES.get(bits(v,1,0),'?')} {ADRMODES.get(bits(v,4,2),'?')} {' '.join(f)}"

def dec_dm1(v):
    f=[n for b,n in [(5,'DBLSRC'),(6,'YFLIP'),(7,'RWPACKED'),(10,'RWDOUBLE'),(11,'SWAPEND')] if bit(v,b)]
    f.append('RGB' if bit(v,15) else 'CI')
    if bit(v,16): f.append('DITHER')
    if bit(v,17): f.append('FASTCLEAR')
    if bit(v,18): f.append(f'BLEND({SFACTOR_NAMES[bits(v,21,19)]}+{DFACTOR_NAMES[bits(v,24,22)]})')
    for b,n in [(25,'BACKBLEND'),(26,'PREFETCH'),(27,'BLENDALPHA')]:
        if bit(v,b): f.append(n)
    return (f"planes={PLANES.get(bits(v,2,0),'?')} {DEPTHS.get(bits(v,4,3),'?')} "
            f"afn:{CMPFN[bits(v,14,12)]} logop={LOGICOPS.get(bits(v,31,28),'?')} {' '.join(f)}")

LINE = re.compile(r'reg=([0-9a-f]+)\(([A-Z0-9_]+)\) val=([0-9a-f]+)')
R = dict(DM1=0x0000, DM0=0x0004, ALPHAREF=0x0020, XSTARTF=0x0138, YSTARTF=0x013c,
         XENDF=0x0140, YENDF=0x0144, XSTARTI=0x0148, XYSTARTI=0x0150,
         RED=0x0200, ALPHA=0x0204, GRN=0x0208, BLUE=0x020c, WRMASK=0x0220)

st, gos, pending = {}, [], []
n_go = 0
lo, hi = arg('--from',0), arg('--to', 1<<62)

for line in open(LOG):
    m = LINE.search(line)
    if not m: continue
    off, name, val = int(m.group(1),16), m.group(2), int(m.group(3),16) & 0xffffffff
    is_go = ' GO ' in line
    st[off] = val
    if not is_go:
        pending.append(name); continue
    n_go += 1
    if lo <= n_go <= hi:
        # x start: XYSTARTI (packed) or XSTARTI wins depending on which came last;
        # both write the same internal xstart, so reconstruct from whichever is set.
        xs = from_i(st.get(R['XSTARTI'],0))
        ys = from_i(st.get(R['XYSTARTI'],0) & 0xffff)
        xe = from_f(st.get(R['XENDF'],0))
        gos.append(dict(n=n_go, dm0=st.get(R['DM0'],0), dm1=st.get(R['DM1'],0),
            xs=xs, ys=ys, xe=xe,
            r=st.get(R['RED'],0), g=st.get(R['GRN'],0), b=st.get(R['BLUE'],0),
            a=st.get(R['ALPHA'],0), aref=st.get(R['ALPHAREF'],0),
            setup=list(pending)))
    pending = []

print(f"Total GOs: {n_go}\n")

print("=== GO by (DRAWMODE0, DRAWMODE1) ===")
for (d0,d1),c in Counter((g['dm0'],g['dm1']) for g in gos).most_common():
    print(f"[{c:7d}x] dm0={d0:#010x} dm1={d1:#010x}")
    print(f"          {dec_dm0(d0)}")
    print(f"          {dec_dm1(d1)}")

print("\n=== DRAW SPAN width in pixels (xend - xstart) ===")
w = Counter()
for g in gos:
    if bits(g['dm0'],1,0)==2 and bits(g['dm0'],4,2)==0:
        w[round(px(g['xe']) - px(g['xs']))] += 1
for width,c in sorted(w.items())[:25]:
    print(f"  {width:6d} px : {c:8d} spans")
print(f"  ({len(w)} distinct widths over {sum(w.values())} spans)")

print("\n=== Register writes preceding each GO (top 12) ===")
for s,c in Counter(','.join(g['setup']) for g in gos).most_common(12):
    print(f"  [{c:7d}x] {s or '(nothing — GO with no new state)'}")

print("\n=== ALPHA values seen at DRAW SPAN (top 15) ===")
for a,c in Counter(g['a'] for g in gos if bits(g['dm0'],1,0)==2).most_common(15):
    print(f"  [{c:7d}x] raw={a:#010x}  alpha={col(a):8.2f}  (/4095 = {col(a)/4095:.3f})")
print("\nALPHAREF values:", {hex(k):v for k,v in Counter(g['aref'] for g in gos).items()})

# --- geometry + mode correlation: which draws cover which screen area ---
print("\n=== Screen coverage by DRAWMODE1 (physical px, bias 4096 removed) ===")
import collections
agg = collections.defaultdict(lambda: [1e9,1e9,-1e9,-1e9,0])
for g in gos:
    if bits(g['dm0'],1,0) != 2: continue
    x0 = px(g['xs']) - BIAS; x1 = px(g['xe']) - BIAS; y = px(g['ys']) - BIAS
    if not (-100 < x0 < 2000 and -100 < y < 1200): continue
    a = agg[g['dm1']]
    a[0]=min(a[0],x0); a[1]=min(a[1],y); a[2]=max(a[2],x1); a[3]=max(a[3],y); a[4]+=1
for dm1,(x0,y0,x1,y1,c) in sorted(agg.items(), key=lambda kv:-kv[1][4]):
    print(f"dm1={dm1:#010x} [{c:7d} spans] bbox x {x0:7.1f}..{x1:7.1f}  y {y0:6.1f}..{y1:6.1f}")
    print(f"     {dec_dm1(dm1)}")

print("\n=== alpha=0 spans that still get drawn (blend on, cmp=always) ===")
n_bad = n_ok = 0
for g in gos:
    if bits(g['dm0'],1,0)!=2: continue
    a_stored = g['a'] & 0xFFFFF
    a_int = (a_stored>>11)&0x1FF
    alpha = 0 if (a_stored & (1<<31)) or a_int>=0x180 else min(a_int,0xFF)
    if alpha != 0: continue
    if bit(g['dm1'],18) and bits(g['dm1'],14,12)==7: n_bad += 1
    else: n_ok += 1
print(f"  alpha==0 spans with blend+cmp:always (rely on blend to vanish): {n_bad}")
print(f"  alpha==0 spans otherwise (alpha test or no blend)             : {n_ok}")
