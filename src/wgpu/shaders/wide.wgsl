// 64- and 128-bit integer arithmetic on 32-bit halves, for models whose
// fixed point goes past 32 bits (IMPACT's iterators are i64 at 2^32, its
// reciprocal unit multiplies them into 96- and 128-bit products). Rust's
// semantics: products are exact before the shift, shifts of signed values
// are arithmetic, a narrowing `as` keeps the low bits.

#include "i64.wgsl"

// a + b + carry (0 or 1): the sum and the carry out.
fn adc32(a: u32, b: u32, carry: u32) -> vec2<u32> {
    let s = a + b;
    let c0 = select(0u, 1u, s < a);
    let t = s + carry;
    let c1 = select(0u, 1u, t < s);
    return vec2<u32>(t, c0 + c1);
}

// a * b as an unsigned 64-bit value.
fn mul32x32(a: u32, b: u32) -> I64 {
    let al = a & 0xffffu;
    let ah = a >> 16u;
    let bl = b & 0xffffu;
    let bh = b >> 16u;
    let ll = al * bl;
    let lh = al * bh;
    let hl = ah * bl;
    let hh = ah * bh;
    let mid = lh + hl;
    let mid_carry = select(0u, 0x10000u, mid < lh);
    let lo = ll + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < ll);
    return I64(lo, hh + (mid >> 16u) + mid_carry + lo_carry);
}

// The unsigned 128-bit product of two unsigned 64-bit values, lowest word
// first.
fn mul64x64_u(a: I64, b: I64) -> vec4<u32> {
    let p00 = mul32x32(a.lo, b.lo);
    let p01 = mul32x32(a.lo, b.hi);
    let p10 = mul32x32(a.hi, b.lo);
    let p11 = mul32x32(a.hi, b.hi);
    var s = adc32(p00.hi, p01.lo, 0u);
    let w1a = s.x;
    var c2 = s.y;
    s = adc32(w1a, p10.lo, 0u);
    let w1 = s.x;
    c2 += s.y;
    s = adc32(p01.hi, p10.hi, 0u);
    var c3 = s.y;
    s = adc32(s.x, p11.lo, 0u);
    c3 += s.y;
    s = adc32(s.x, c2, 0u);
    c3 += s.y;
    let w2 = s.x;
    let w3 = p11.hi + c3;
    return vec4<u32>(p00.lo, w1, w2, w3);
}

fn neg128(x: vec4<u32>) -> vec4<u32> {
    var s = adc32(~x.x, 1u, 0u);
    let w0 = s.x;
    s = adc32(~x.y, 0u, s.y);
    let w1 = s.x;
    s = adc32(~x.z, 0u, s.y);
    let w2 = s.x;
    let w3 = ~x.w + s.y;
    return vec4<u32>(w0, w1, w2, w3);
}

// The signed 128-bit product of two signed 64-bit values.
fn mul64x64_s(a: I64, b: I64) -> vec4<u32> {
    let na = i64_is_neg(a);
    let nb = i64_is_neg(b);
    var ua = a;
    var ub = b;
    if na {
        ua = i64_neg(a);
    }
    if nb {
        ub = i64_neg(b);
    }
    let p = mul64x64_u(ua, ub);
    return select(p, neg128(p), na != nb);
}

// Word i of a 128-bit value, the sign (fill) beyond it.
fn word128(x: vec4<u32>, i: u32, fill: u32) -> u32 {
    switch i {
        case 0u: { return x.x; }
        case 1u: { return x.y; }
        case 2u: { return x.z; }
        case 3u: { return x.w; }
        default: { return fill; }
    }
}

// The low 64 bits of x >> n (n below 128), arithmetic when `signed`.
fn shr128(x: vec4<u32>, n: u32, signed: bool) -> I64 {
    let fill = select(0u, 0xffffffffu, signed && (x.w & 0x80000000u) != 0u);
    let q = n / 32u;
    let r = n % 32u;
    let w0 = word128(x, q, fill);
    let w1 = word128(x, q + 1u, fill);
    let w2 = word128(x, q + 2u, fill);
    if r == 0u {
        return I64(w0, w1);
    }
    return I64((w0 >> r) | (w1 << (32u - r)), (w1 >> r) | (w2 << (32u - r)));
}

// ((a as i128 * b as i128) >> n) as i64.
fn mul_shr_s(a: I64, b: I64, n: u32) -> I64 {
    return shr128(mul64x64_s(a, b), n, true);
}

// a * b, wrapping: the low 64 bits of the product.
fn i64_mul(a: I64, b: I64) -> I64 {
    let p = mul64x64_u(a, b);
    return I64(p.x, p.y);
}

// a >> n, arithmetic, n below 64.
fn i64_shr(a: I64, n: u32) -> I64 {
    let fill = select(0u, 0xffffffffu, i64_is_neg(a));
    if n == 0u {
        return a;
    }
    if n >= 32u {
        let s = n - 32u;
        if s == 0u {
            return I64(a.hi, fill);
        }
        return I64(bitcast<u32>(bitcast<i32>(a.hi) >> s), fill);
    }
    return I64((a.lo >> n) | (a.hi << (32u - n)), bitcast<u32>(bitcast<i32>(a.hi) >> n));
}

// a >> n, logical, n below 64.
fn u64_shr(a: I64, n: u32) -> I64 {
    if n == 0u {
        return a;
    }
    if n >= 32u {
        return I64(a.hi >> (n - 32u), 0u);
    }
    return I64((a.lo >> n) | (a.hi << (32u - n)), a.hi >> n);
}

// a << n, n below 64.
fn u64_shl(a: I64, n: u32) -> I64 {
    if n == 0u {
        return a;
    }
    if n >= 32u {
        return I64(0u, a.lo << (n - 32u));
    }
    return I64(a.lo << n, (a.hi << n) | (a.lo >> (32u - n)));
}

// a < b, unsigned.
fn u64_lt(a: I64, b: I64) -> bool {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

fn i64_eq(a: I64, b: I64) -> bool {
    return a.lo == b.lo && a.hi == b.hi;
}

fn i64_is_zero(a: I64) -> bool {
    return (a.lo | a.hi) == 0u;
}

// 63 - leading zeros: the position of the highest set bit (a nonzero).
fn u64_top_bit(a: I64) -> u32 {
    if a.hi != 0u {
        return 63u - countLeadingZeros(a.hi);
    }
    return 31u - countLeadingZeros(a.lo);
}

// Clamp a signed value to [lo, hi] (32-bit bounds).
fn i64_clamp32(a: I64, lo: i32, hi: i32) -> I64 {
    let l = i64_from_i32(lo);
    let h = i64_from_i32(hi);
    if i64_lt(a, l) {
        return l;
    }
    if i64_lt(h, a) {
        return h;
    }
    return a;
}

// The low 32 bits as a signed value (`as i32`).
fn i64_low_i32(a: I64) -> i32 {
    return bitcast<i32>(a.lo);
}

// a + b * c (b * c the low 64 bits of the product), wrapping: a plane
// stepped c times.
fn i64_madd(a: I64, b: I64, c: i32) -> I64 {
    return i64_add(a, i64_mul(b, i64_from_i32(c)));
}
