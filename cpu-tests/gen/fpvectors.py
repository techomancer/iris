#!/usr/bin/env python3
"""fpvectors.py — compute IEEE-754 expectations exactly, and emit them as C.

Every expected value here is computed from the *definition* of IEEE 754
arithmetic using exact rational arithmetic (`fractions.Fraction`), then rounded
to the destination format by hand. Nothing is read off a floating-point unit —
not the host's, and certainly not the emulated one. That is the whole point:
an expectation recorded from a machine proves only that the machine still does
what it used to, and a bit pattern worked out by hand is a second, untested
implementation of floating-point arithmetic (see docs/oracle.md).

Python's own floats are used for *nothing* except pretty-printing comments.

Usage:
    python3 gen/fpvectors.py            # rewrite tests/fpu/fpvectors.{c,h}
    make -C cpu-tests vectors           # the same thing

The generated files are checked in, so building the suite needs only the cross
toolchain — no Python, which keeps CI's dependency list to one line.
"""

import sys
from fractions import Fraction
from math import isqrt

# ── formats ──────────────────────────────────────────────────────────────────

class Fmt:
    def __init__(self, name, p, emax, mbits, width):
        self.name = name        # 's' or 'd'
        self.p = p              # significand bits, including the implicit one
        self.emax = emax        # largest unbiased exponent of a finite number
        self.emin = 1 - emax    # smallest unbiased exponent of a normal number
        self.mbits = mbits      # stored mantissa bits
        self.width = width      # total bits
        self.bias = emax
        self.qnan = (((1 << (width - mbits - 1)) - 1) << mbits) | (1 << (mbits - 1))

    @property
    def inf(self):
        return ((1 << (self.width - self.mbits - 1)) - 1) << self.mbits

    @property
    def maxfinite(self):
        return self.inf - 1

S = Fmt('s', p=24, emax=127, mbits=23, width=32)
D = Fmt('d', p=53, emax=1023, mbits=52, width=64)

# Rounding modes, in FCSR.RM order.
RN, RZ, RP, RM = 0, 1, 2, 3
RM_NAMES = ['rn', 'rz', 'rp', 'rm']

# FCSR exception bits, matching FP_* in harness/iris.h.
F_I, F_U, F_O, F_Z, F_V = 1, 2, 4, 8, 16

# ── values ───────────────────────────────────────────────────────────────────
#
# A value is one of:
#   ('n', Fraction)   a finite number; Fraction(0) carries its sign separately
#   ('z', sign)       zero
#   ('i', sign)       infinity
#   ('q', ())         quiet NaN — accepted as an operand, never emitted as an
#                     expectation (the payload is implementation-defined)
# sign is +1 or -1.

def decode(bits, f):
    """Unpack a bit pattern into a value."""
    sign = -1 if bits >> (f.width - 1) else 1
    exp = (bits >> f.mbits) & ((1 << (f.width - f.mbits - 1)) - 1)
    man = bits & ((1 << f.mbits) - 1)
    if exp == (1 << (f.width - f.mbits - 1)) - 1:
        return ('i', sign) if man == 0 else ('q', ())
    if exp == 0:
        if man == 0:
            return ('z', sign)
        # subnormal
        return ('n', sign * Fraction(man, 1 << f.mbits) * Fraction(2) ** f.emin)
    v = Fraction((1 << f.mbits) | man, 1 << f.mbits) * Fraction(2) ** (exp - f.bias)
    return ('n', sign * v)

def round_to_int(x, rm, negative):
    """Round an exact non-negative Fraction to an integer. `negative` is the
    sign of the value the magnitude came from, since RP/RM are directional."""
    fl = x.numerator // x.denominator
    rem = x - fl
    if rem == 0:
        return fl, False
    if rm == RN:
        if rem > Fraction(1, 2):
            r = fl + 1
        elif rem < Fraction(1, 2):
            r = fl
        else:
            r = fl if fl % 2 == 0 else fl + 1
    elif rm == RZ:
        r = fl
    elif rm == RP:
        r = fl if negative else fl + 1
    else:  # RM
        r = fl + 1 if negative else fl
    return r, True

def encode(v, f, rm=RN):
    """Round an exact value to `f` and return (bits, flags).

    Returns bits=None for anything this generator refuses to predict: a
    subnormal or underflowing result, which on an R4400 is an Unimplemented
    Operation (E) rather than an arithmetic result at all, and whose behaviour
    depends on FCSR.FS. Those cases are hand-written in tests/fpu/fpu_denorm.c
    with the manual quoted next to them, not generated here.
    """
    kind, val = v
    if kind == 'z':
        return (0 if val > 0 else 1 << (f.width - 1)), 0
    if kind == 'i':
        return f.inf | (0 if val > 0 else 1 << (f.width - 1)), 0
    if kind == 'q':
        return None, 0
    if val == 0:
        return 0, 0                          # an exact zero written as ('n', 0)
    neg = val < 0
    a = -val if neg else val
    signbit = (1 << (f.width - 1)) if neg else 0

    # Binade: the integer e with 2^e <= a < 2^(e+1).
    e = 0
    while a >= Fraction(2) ** (e + 1):
        e += 1
    while a < Fraction(2) ** e:
        e -= 1

    if e < f.emin:
        return None, 0                       # subnormal — see the docstring
    m, inexact = round_to_int(a * Fraction(2) ** (f.p - 1 - e), rm, neg)
    if m == 1 << f.p:                        # rounding carried into the next binade
        m >>= 1
        e += 1
    if e > f.emax:                           # overflow: Table 7-1
        if rm == RN or (rm == RP and not neg) or (rm == RM and neg):
            return f.inf | signbit, F_O | F_I
        return f.maxfinite | signbit, F_O | F_I
    bits = signbit | ((e + f.bias) << f.mbits) | (m & ((1 << f.mbits) - 1))
    return bits, (F_I if inexact else 0)

# ── operations ───────────────────────────────────────────────────────────────
#
# Special cases follow the R4000 manual's Invalid Operation list (chapter 7):
# inf-inf, 0*inf, 0/0, inf/inf, and sqrt of a negative.

def roundtrip(v, f):
    """The value as the register will actually hold it.

    An operand written down as an exact rational is not necessarily
    representable: 2^30-1 needs 30 significand bits and a single has 24, so the
    FPU sees 2^30 and converts *that*. Computing an expectation from the
    unrounded value produces a table that no correct implementation can match —
    which is how this function came to exist. Returns None for anything the
    format cannot hold at all.
    """
    b, _ = encode(v, f)
    return None if b is None else decode(b, f)

def _num(v):
    return v[1] if v[0] == 'n' else Fraction(0)

def _iszero(v):
    return v[0] == 'z'

def op_add(x, y):
    if 'q' in (x[0], y[0]):
        return ('q', ()), 0
    if x[0] == 'i' and y[0] == 'i':
        return (x, 0) if x[1] == y[1] else (('q', ()), F_V)
    if x[0] == 'i':
        return x, 0
    if y[0] == 'i':
        return y, 0
    if _iszero(x) and _iszero(y):
        # (+0)+(-0) is +0 in every mode but RM, where it is -0.
        return ('z', 1 if x[1] > 0 or y[1] > 0 else -1), 0
    s = _num(x) + _num(y)
    if s == 0:
        return ('z', 1), 0
    return ('n', s), 0

def op_sub(x, y):
    return op_add(x, _negate(y))

def _negate(v):
    if v[0] == 'q':
        return v
    if v[0] in ('z', 'i'):
        return (v[0], -v[1])
    return ('n', -v[1])

def op_mul(x, y):
    if 'q' in (x[0], y[0]):
        return ('q', ()), 0
    sign = _sign(x) * _sign(y)
    if (x[0] == 'i' and _iszero(y)) or (y[0] == 'i' and _iszero(x)):
        return ('q', ()), F_V
    if x[0] == 'i' or y[0] == 'i':
        return ('i', sign), 0
    if _iszero(x) or _iszero(y):
        return ('z', sign), 0
    return ('n', _num(x) * _num(y)), 0

def op_div(x, y):
    if 'q' in (x[0], y[0]):
        return ('q', ()), 0
    sign = _sign(x) * _sign(y)
    if x[0] == 'i' and y[0] == 'i':
        return ('q', ()), F_V
    if _iszero(x) and _iszero(y):
        return ('q', ()), F_V
    if x[0] == 'i':
        return ('i', sign), 0
    if y[0] == 'i':
        return ('z', sign), 0
    if _iszero(y):
        return ('i', sign), F_Z          # finite nonzero / 0
    if _iszero(x):
        return ('z', sign), 0
    return ('n', _num(x) / _num(y)), 0

def _sign(v):
    if v[0] == 'n':
        return -1 if v[1] < 0 else 1
    if v[0] in ('z', 'i'):
        return v[1]
    return 1

def op_sqrt(x, f, rm=RN):
    """Correctly-rounded square root, computed with integer arithmetic only.

    Returned as (bits, flags) rather than a value, because the exact result is
    irrational in general and only the rounded one is representable.
    """
    if x[0] == 'q':
        return None, 0
    if x[0] == 'z':
        return encode(x, f)[0], 0            # sqrt(±0) = ±0
    if x[0] == 'i':
        if x[1] < 0:
            return f.qnan, F_V               # sqrt(-inf)
        return f.inf, 0
    a = x[1]
    if a < 0:
        return f.qnan, F_V
    # Find e with 2^e <= sqrt(a) < 2^(e+1), then round sqrt(a)*2^(p-1-e).
    e = 0
    while Fraction(2) ** (2 * (e + 1)) <= a:
        e += 1
    while Fraction(2) ** (2 * e) > a:
        e -= 1
    scale = f.p - 1 - e
    scaled = a * Fraction(2) ** (2 * scale)  # we want round(sqrt(scaled))
    num, den = scaled.numerator, scaled.denominator
    n = isqrt(num * den) // den              # floor(sqrt(num/den))
    while Fraction((n + 1) ** 2) <= scaled:  # isqrt of a ratio can land low
        n += 1
    exact = Fraction(n) ** 2 == scaled
    if not exact:
        # Round to nearest-even against the exact midpoint: compare
        # (n + 1/2)^2 with the scaled value, i.e. (2n+1)^2 vs 4*scaled.
        mid = Fraction((2 * n + 1) ** 2, 4)
        if mid < scaled or (mid == scaled and n % 2 == 1):
            n += 1
    if n == 1 << f.p:
        n >>= 1
        e += 1
    bits = ((e + f.bias) << f.mbits) | (n & ((1 << f.mbits) - 1))
    return bits, (0 if exact else F_I)

# ── conversions ──────────────────────────────────────────────────────────────

def to_int(v, rm, bits):
    """Convert a value to a `bits`-wide signed integer, or None if the source
    is out of range / not a number. The out-of-range case is deliberately not
    predicted: the R4000 manual (Table 7-2) makes it an Unimplemented Operation
    while the MIPS IV ISA makes it Invalid with a saturated result, and the two
    parts need not agree — tests/fpu/fpu_breadth.c reports it instead."""
    if v[0] in ('q', 'i'):
        return None
    if v[0] == 'z':
        return 0
    r, _ = round_to_int(abs(v[1]), rm, v[1] < 0)
    r = -r if v[1] < 0 else r
    if not (-(1 << (bits - 1)) <= r < (1 << (bits - 1))):
        return None
    return r

def from_int(i, f, rm=RN):
    if i == 0:
        return 0, 0
    return encode(('n', Fraction(i)), f, rm)

# ── literals ─────────────────────────────────────────────────────────────────

def val(x, f):
    """A value from a Python int (bit pattern) or Fraction/int expression."""
    return decode(x, f) if isinstance(x, int) else ('n', Fraction(x))

def bits_of(x, f, rm=RN):
    b, _ = encode(('n', Fraction(x)) if not isinstance(x, tuple) else x, f, rm)
    if b is None:
        raise ValueError('operand is not representable in %s: %s' % (f.name, x))
    return b

def show(v, f):
    """A short decimal rendering for the comment column. Uses Python floats,
    which is fine: nothing depends on it."""
    kind, x = v
    if kind == 'z':
        return '-0' if x < 0 else '0'
    if kind == 'i':
        return '-inf' if x < 0 else 'inf'
    if kind == 'q':
        return 'nan'
    try:
        return repr(float(x))
    except OverflowError:
        # An intermediate that is out of range for a host double — which is
        # exactly the interesting case, since it is what overflows the target
        # format too. The comment column does not need the digits.
        return ('-' if x < 0 else '') + 'huge'

# ── the vectors ──────────────────────────────────────────────────────────────
#
# Operands are chosen to cover: exact results, results that must round (I),
# results that overflow (O|I), the two zero signs, infinities, and the
# division-by-zero and invalid-operation cases whose *flags* are pinned even
# when their NaN payload is not.

def two_operand_cases(f):
    """(a, b) pairs, as exact values, for add/sub/mul/div."""
    eps = Fraction(1, 1 << (f.p - 1))            # 1 ulp of 1.0
    tiny = Fraction(1, 1 << f.p)                 # half an ulp of 1.0 — rounds away
    big = decode(f.maxfinite, f)                 # largest finite
    return [
        (('n', Fraction(2)), ('n', Fraction(3))),
        (('n', Fraction(5)), ('n', Fraction(3))),
        (('n', Fraction(1)), ('n', eps)),        # exact
        (('n', Fraction(1)), ('n', tiny)),       # inexact: rounds to nearest even
        (('n', Fraction(1)), ('n', Fraction(3))),
        (('n', Fraction(-7, 2)), ('n', Fraction(1, 4))),
        (big, big),                              # overflow for add/mul
        (big, ('n', Fraction(1, 2))),            # overflow for div
        (('n', Fraction(1)), ('z', 1)),          # /0 -> Z
        (('n', Fraction(-1)), ('z', 1)),
        (('z', 1), ('z', -1)),
        (('z', -1), ('z', -1)),
        (('i', 1), ('n', Fraction(1))),
        (('i', 1), ('i', 1)),
        (('i', 1), ('i', -1)),                   # inf-inf -> V for sub
        (('i', 1), ('z', 1)),                    # 0*inf -> V for mul
        (('n', Fraction(1)), ('i', 1)),
    ]

def one_operand_cases(f):
    return [
        ('n', Fraction(4)),
        ('n', Fraction(2)),                      # irrational root — inexact
        ('n', Fraction(1, 4)),
        ('n', Fraction(9, 16)),
        ('z', 1),
        ('z', -1),
        ('i', 1),
        ('n', Fraction(-2)),                     # sqrt(-x) -> V
        ('i', -1),
    ]

def convert_cases(f):
    """Values whose conversion to an integer is in range for both W and L."""
    return [
        ('n', Fraction(5, 2)), ('n', Fraction(7, 2)), ('n', Fraction(-5, 2)),
        ('n', Fraction(3, 2)), ('n', Fraction(1, 2)), ('n', Fraction(-1, 2)),
        ('n', Fraction(0)), ('z', 1), ('z', -1),
        ('n', Fraction(1)), ('n', Fraction(-1)),
        ('n', Fraction(1 << 20) + Fraction(1, 2)),
        ('n', Fraction(-(1 << 20)) - Fraction(1, 2)),
        ('n', Fraction((1 << 30) - 1)),
        ('n', Fraction(-(1 << 30))),
    ]

# ── emission ─────────────────────────────────────────────────────────────────

class Out:
    def __init__(self):
        self.c = []
        self.h = []

    def table(self, ctype, name, rows, comment):
        """Emit `rows` (a list of (fields, comment) tuples) as a C array."""
        self.h.append('extern const struct %s %s[];' % (ctype, name))
        self.h.append('#define %s_N %d' % (name.upper(), len(rows)))
        self.c.append('/* %s */' % comment)
        self.c.append('const struct %s %s[] = {' % (ctype, name))
        for fields, note in rows:
            self.c.append('    { %s },%s' % (', '.join(fields),
                                             (' /* %s */' % note) if note else ''))
        self.c.append('};')
        self.c.append('')

def hexlit(v, f):
    return '0x%08Xu' % v if f.width == 32 else '0x%016XULL' % v

def flagstr(fl):
    if fl == 0:
        return '0'
    names = [(F_V, 'FP_V'), (F_Z, 'FP_Z'), (F_O, 'FP_O'), (F_U, 'FP_U'), (F_I, 'FP_I')]
    return ' | '.join(n for b, n in names if fl & b)

def gen_arith(out, f, opname, fn):
    rows = []
    for a, b in two_operand_cases(f):
        a, b = roundtrip(a, f), roundtrip(b, f)
        if a is None or b is None:
            continue
        ab, _ = encode(a, f)
        bb, _ = encode(b, f)
        r, fl = fn(a, b)
        rb, rfl = encode(r, f)
        if rb is None:
            continue                 # NaN or subnormal result — not generated
        rows.append((
            [hexlit(ab, f), hexlit(bb, f), hexlit(rb, f), flagstr(fl | rfl)],
            '%s %s %s = %s' % (show(a, f), opname, show(b, f), show(r, f))
        ))
    out.table('fpv%s2' % f.name, 'fpv_%s_%s' % (opname, f.name), rows,
              '%s.%s' % (opname, f.name))

def gen_sqrt(out, f):
    rows = []
    for a in one_operand_cases(f):
        a = roundtrip(a, f)
        if a is None:
            continue
        ab, _ = encode(a, f)
        rb, fl = op_sqrt(a, f)
        if rb is None or (fl & F_V):
            continue                 # NaN result — payload is not pinned
        rows.append(([hexlit(ab, f), hexlit(rb, f), flagstr(fl)],
                     'sqrt(%s)' % show(a, f)))
    out.table('fpv%s1' % f.name, 'fpv_sqrt_%s' % f.name, rows, 'sqrt.%s' % f.name)

def gen_rm(out, f, opname, fn, cases):
    """The same operation under all four rounding modes.

    Rounding is only ever tested on conversions in this suite; arithmetic
    rounds too, and the overflow rows are Table 7-1's default actions written
    out — round-to-nearest gives an infinity, round-to-zero the largest finite
    number, and the two directed modes give one or the other depending on the
    sign.
    """
    rows = []
    for a, b in cases:
        a, b = roundtrip(a, f), roundtrip(b, f)
        if a is None or b is None:
            continue
        r, _ = fn(a, b)
        outs = [encode(r, f, rm) for rm in (RN, RZ, RP, RM)]
        if any(o[0] is None for o in outs):
            continue
        rows.append(([hexlit(encode(a, f)[0], f), hexlit(encode(b, f)[0], f)]
                     + [hexlit(o[0], f) for o in outs],
                     '%s %s %s' % (show(a, f), opname, show(b, f))))
    out.table('fpvrm%s' % f.name, 'fpv_rm_%s_%s' % (opname, f.name), rows,
              '%s.%s under RN, RZ, RP, RM' % (opname, f.name))

def rm_cases(f):
    big = decode(f.maxfinite, f)
    return {
        'div': [(('n', Fraction(1)), ('n', Fraction(3))),
                (('n', Fraction(-1)), ('n', Fraction(3))),
                (('n', Fraction(2)), ('n', Fraction(3))),
                (('n', Fraction(-2)), ('n', Fraction(3)))],
        'mul': [(big, big), (_negate(big), big)],
    }

def gen_cvt_int(out, f, width):
    """cvt.w/cvt.l under all four rounding modes — which also gives
    round/trunc/ceil/floor, since those are RN/RZ/RP/RM with the mode fixed."""
    rows = []
    for v in convert_cases(f):
        v = roundtrip(v, f)
        if v is None:
            continue
        vb, _ = encode(v, f)
        outs = [to_int(v, rm, width) for rm in (RN, RZ, RP, RM)]
        if any(o is None for o in outs):
            continue
        suffix = 'LL' if width == 64 else ''
        rows.append(([hexlit(vb, f)] + ['%d%s' % (o, suffix) for o in outs],
                     '%s -> %s' % (show(v, f), ','.join(str(o) for o in outs))))
    out.table('fpvcvt_%s%d' % (f.name, width), 'fpv_cvt_%s%d' % (f.name, width),
              rows, 'cvt.%s.%s / round / trunc / ceil / floor'
                    % ('w' if width == 32 else 'l', f.name))

def gen_from_int(out, width):
    """cvt.s.w / cvt.d.w / cvt.s.l / cvt.d.l."""
    if width == 32:
        vals = [0, 1, -1, 42, -42, 1 << 20, -(1 << 20), (1 << 31) - 1, -(1 << 31),
                0x00FFFFFF, 0x01000001]     # the last needs rounding in single
    else:
        vals = [0, 1, -1, 42, (1 << 40) + 1, -(1 << 40) - 1,
                (1 << 62) + (1 << 10), (1 << 53) + 1]
    rows = []
    for i in vals:
        sb, sfl = from_int(i, S)
        db, dfl = from_int(i, D)
        suffix = 'LL' if width == 64 else ''
        rows.append((['%d%s' % (i, suffix), hexlit(sb, S), flagstr(sfl),
                      hexlit(db, D), flagstr(dfl)], str(i)))
    out.table('fpvfromi%d' % width, 'fpv_from_i%d' % width, rows,
              'cvt.s/d from a %d-bit integer' % width)


# ── self-check ───────────────────────────────────────────────────────────────
#
# The tables above are computed from the definition of IEEE 754. This compares
# them against a completely different implementation of the same arithmetic —
# the host's own FPU, via Python floats — as a cross-check, in the sense
# docs/oracle.md §4 means: a disagreement is a reason to go back to the
# standard, not to copy the host's answer.
#
# Double-rounding is not a hazard here. A single-precision result computed in
# double and then rounded to single is correctly rounded whenever the format
# has at least 2p+2 = 50 bits, and a double has 53.

import math
import struct

def _tofloat(v):
    kind, x = v
    if kind == 'z':
        return -0.0 if x < 0 else 0.0
    if kind == 'i':
        return float('-inf') if x < 0 else float('inf')
    if kind == 'q':
        return float('nan')
    return int(x.numerator) / int(x.denominator)

def _host_bits(x, f):
    """Round a host double to `f` and return its bit pattern, or None if the
    host cannot represent the intermediate at all."""
    if f is D:
        return struct.unpack('>Q', struct.pack('>d', x))[0]
    try:
        return struct.unpack('>I', struct.pack('>f', x))[0]
    except OverflowError:
        return S.inf | (1 << 31 if x < 0 else 0)

def check():
    bad = 0
    ops = [('add', op_add, lambda a, b: a + b),
           ('sub', op_sub, lambda a, b: a - b),
           ('mul', op_mul, lambda a, b: a * b),
           ('div', op_div, lambda a, b: a / b)]
    for f in (S, D):
        for name, fn, hostfn in ops:
            for a, b in two_operand_cases(f):
                a, b = roundtrip(a, f), roundtrip(b, f)
                if a is None or b is None:
                    continue
                mine, _ = encode(fn(a, b)[0], f)
                if mine is None:
                    continue
                try:
                    host = _host_bits(hostfn(_tofloat(a), _tofloat(b)), f)
                except (ZeroDivisionError, ValueError, OverflowError):
                    continue     # the host raises where the FPU sets a flag
                if host != mine:
                    # An intermediate that overflows a host double cannot be
                    # cross-checked in single: the host already lost it.
                    if f is S and abs(_tofloat(a)) > 1e308:
                        continue
                    sys.stderr.write('MISMATCH %s.%s %s %s: mine=%x host=%x\n'
                                     % (name, f.name, show(a, f), show(b, f), mine, host))
                    bad += 1
        for a in one_operand_cases(f):
            a = roundtrip(a, f)
            if a is None:
                continue
            mine, _ = op_sqrt(a, f)
            hv = _tofloat(a)
            if hv < 0 or hv != hv:
                continue
            # math.sqrt, not `** 0.5`: Python's pow returns +0.0 for
            # (-0.0) ** 0.5, where IEEE 754 (and math.sqrt) give -0.0.
            host = _host_bits(math.sqrt(hv), f)
            if host != mine:
                sys.stderr.write('MISMATCH sqrt.%s %s: mine=%x host=%x\n'
                                 % (f.name, show(a, f), mine, host))
                bad += 1
    sys.stderr.write('self-check: %d mismatches\n' % bad)
    return 1 if bad else 0

HEADER_TOP = '''/* fpvectors.h — GENERATED by gen/fpvectors.py. Do not edit.
 *
 * IEEE-754 expectations computed with exact rational arithmetic on the host,
 * never read off any FPU. Regenerate with `make vectors`. See docs/oracle.md
 * for why the suite insists on this.
 *
 * `flags` is the set of FCSR exception bits (FP_I/FP_U/FP_O/FP_Z/FP_V) the
 * operation must raise. Results that are NaN or subnormal are deliberately
 * absent: NaN payloads are implementation-defined, and a subnormal result is
 * an Unimplemented Operation on an R4400 rather than an arithmetic result at
 * all — tests/fpu/fpu_denorm.c covers those with the manual quoted alongside.
 */
#ifndef FPVECTORS_H
#define FPVECTORS_H

#include "testlib.h"

struct fpvs2 { u32 a, b, r; u32 flags; };
struct fpvs1 { u32 a, r; u32 flags; };
struct fpvd2 { u64 a, b, r; u32 flags; };
struct fpvd1 { u64 a, r; u32 flags; };
struct fpvcvt_s32 { u32 in; s32 rn, rz, rp, rm; };
struct fpvcvt_s64 { u32 in; s64 rn, rz, rp, rm; };
struct fpvcvt_d32 { u64 in; s32 rn, rz, rp, rm; };
struct fpvcvt_d64 { u64 in; s64 rn, rz, rp, rm; };
struct fpvrms { u32 a, b, rn, rz, rp, rm; };
struct fpvrmd { u64 a, b, rn, rz, rp, rm; };
struct fpvfromi32 { s32 in; u32 s; u32 sflags; u64 d; u32 dflags; };
struct fpvfromi64 { s64 in; u32 s; u32 sflags; u64 d; u32 dflags; };

'''

C_TOP = '''/* fpvectors.c — GENERATED by gen/fpvectors.py. Do not edit.
 * Regenerate with `make vectors`. */

#include "fpvectors.h"

'''

def main():
    out = Out()
    for f in (S, D):
        gen_arith(out, f, 'add', op_add)
        gen_arith(out, f, 'sub', op_sub)
        gen_arith(out, f, 'mul', op_mul)
        gen_arith(out, f, 'div', op_div)
        gen_sqrt(out, f)
        gen_rm(out, f, 'div', op_div, rm_cases(f)['div'])
        gen_rm(out, f, 'mul', op_mul, rm_cases(f)['mul'])
        gen_cvt_int(out, f, 32)
        gen_cvt_int(out, f, 64)
    gen_from_int(out, 32)
    gen_from_int(out, 64)

    hpath = 'tests/fpu/fpvectors.h'
    cpath = 'tests/fpu/fpvectors.c'
    with open(hpath, 'w') as fh:
        fh.write(HEADER_TOP)
        fh.write('\n'.join(out.h))
        fh.write('\n\n#endif /* FPVECTORS_H */\n')
    with open(cpath, 'w') as fh:
        fh.write(C_TOP)
        fh.write('\n'.join(out.c))
        fh.write('\n')
    sys.stderr.write('wrote %s and %s\n' % (hpath, cpath))

if __name__ == '__main__':
    if '--check' in sys.argv:
        sys.exit(check())
    main()
