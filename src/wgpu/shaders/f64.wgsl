// IEEE double precision on integer halves, for models that compute in f64
// (IMPACT's triangle edges and GL lines walk in doubles), so a kernel can
// reach the same integer the CPU's floor or ceil reaches. A value is its
// bit pattern in an I64. Add, subtract and multiply round to nearest even
// as the host does. Denormal operands are taken as zero and denormal
// results flush to zero: the boards' inputs (f32 positions, s.24 slopes)
// never come within 2^900 of them. NaN and infinity follow IEEE.

#include "wide.wgsl"

const F64_EXP_MASK: u32 = 0x7ffu;

fn f64_exp(a: I64) -> u32 {
    return (a.hi >> 20u) & F64_EXP_MASK;
}

fn f64_sign(a: I64) -> u32 {
    return a.hi & 0x80000000u;
}

// The 53-bit significand with its leading one (a normal).
fn f64_sig(a: I64) -> I64 {
    return I64(a.lo, (a.hi & 0xfffffu) | 0x100000u);
}

fn f64_is_nan(a: I64) -> bool {
    return f64_exp(a) == F64_EXP_MASK && ((a.hi & 0xfffffu) | a.lo) != 0u;
}

fn f64_is_inf(a: I64) -> bool {
    return f64_exp(a) == F64_EXP_MASK && ((a.hi & 0xfffffu) | a.lo) == 0u;
}

// Zero or denormal (taken as zero).
fn f64_is_zero(a: I64) -> bool {
    return f64_exp(a) == 0u;
}

const F64_NAN: I64 = I64(0u, 0x7ff80000u);

fn f64_inf(sign: u32) -> I64 {
    return I64(0u, sign | 0x7ff00000u);
}

fn f64_zero(sign: u32) -> I64 {
    return I64(0u, sign);
}

// x >> n with every bit shifted out or-ed into bit 0 (a "jam").
fn u64_shr_jam(x: I64, n: u32) -> I64 {
    if n == 0u {
        return x;
    }
    if n >= 64u {
        return I64(select(0u, 1u, !i64_is_zero(x)), 0u);
    }
    let r = u64_shr(x, n);
    let back = u64_shl(r, n);
    let lost = !i64_eq(back, x);
    return I64(r.lo | select(0u, 1u, lost), r.hi);
}

// Round and pack: the significand `m` has its leading one at bit 62 (ten
// bits below the 53 kept, the lowest a jam), the value m * 2^(e - 1023 -
// 62).
fn f64_round_pack(sign: u32, e_in: i32, m: I64) -> I64 {
    var e = e_in;
    var keep = u64_shr(m, 10u);
    let rest = m.lo & 0x3ffu;
    let round_bit = (rest & 0x200u) != 0u;
    let sticky = (rest & 0x1ffu) != 0u;
    if round_bit && (sticky || (keep.lo & 1u) != 0u) {
        keep = i64_add(keep, I64(1u, 0u));
        if (keep.hi & 0x200000u) != 0u {
            keep = u64_shr(keep, 1u);
            e += 1;
        }
    }
    if e >= 2047 {
        return f64_inf(sign);
    }
    if e <= 0 {
        return f64_zero(sign);
    }
    return I64(keep.lo, sign | (u32(e) << 20u) | (keep.hi & 0xfffffu));
}

fn f64_mul(a: I64, b: I64) -> I64 {
    let sign = f64_sign(a) ^ f64_sign(b);
    if f64_is_nan(a) || f64_is_nan(b) {
        return F64_NAN;
    }
    if f64_is_inf(a) || f64_is_inf(b) {
        if f64_is_zero(a) || f64_is_zero(b) {
            return F64_NAN;
        }
        return f64_inf(sign);
    }
    if f64_is_zero(a) || f64_is_zero(b) {
        return f64_zero(sign);
    }
    // 53 x 53 bits: the product's leading one is at bit 104 or 105.
    let p = mul64x64_u(f64_sig(a), f64_sig(b));
    var e = i32(f64_exp(a)) + i32(f64_exp(b)) - 1023;
    var shift = 42u;
    if (p.w & 0x200u) != 0u {
        shift = 43u;
        e += 1;
    }
    // To bit 62, the bits below jammed.
    let m = shr128(p, shift, false);
    let low_lost = select(0u, 1u, (p.x | (p.y & ((1u << (shift - 32u)) - 1u))) != 0u);
    return f64_round_pack(sign, e, I64(m.lo | low_lost, m.hi));
}

// |a| < |b| (both finite).
fn f64_mag_lt(a: I64, b: I64) -> bool {
    return u64_lt(I64(a.lo, a.hi & 0x7fffffffu), I64(b.lo, b.hi & 0x7fffffffu));
}

fn f64_add(a_in: I64, b_in: I64) -> I64 {
    if f64_is_nan(a_in) || f64_is_nan(b_in) {
        return F64_NAN;
    }
    if f64_is_inf(a_in) {
        if f64_is_inf(b_in) && f64_sign(a_in) != f64_sign(b_in) {
            return F64_NAN;
        }
        return a_in;
    }
    if f64_is_inf(b_in) {
        return b_in;
    }
    let za = f64_is_zero(a_in);
    let zb = f64_is_zero(b_in);
    if za && zb {
        // -0 + -0 is -0; any other pair of zeros is +0.
        return f64_zero(f64_sign(a_in) & f64_sign(b_in));
    }
    if za {
        return b_in;
    }
    if zb {
        return a_in;
    }
    var a = a_in;
    var b = b_in;
    if f64_mag_lt(a, b) {
        a = b_in;
        b = a_in;
    }
    let ea = f64_exp(a);
    let eb = f64_exp(b);
    let sign = f64_sign(a);
    // Leading ones at bit 62.
    let ma = u64_shl(f64_sig(a), 10u);
    let mb = u64_shr_jam(u64_shl(f64_sig(b), 10u), ea - eb);
    var e = i32(ea);
    if f64_sign(a) == f64_sign(b) {
        var m = i64_add(ma, mb);
        if (m.hi & 0x80000000u) != 0u {
            m = u64_shr_jam(m, 1u);
            e += 1;
        }
        return f64_round_pack(sign, e, m);
    }
    var m = i64_sub(ma, mb);
    if i64_is_zero(m) {
        return f64_zero(0u);
    }
    let top = u64_top_bit(m);
    if top < 62u {
        m = u64_shl(m, 62u - top);
        e -= i32(62u - top);
    }
    return f64_round_pack(sign, e, m);
}

fn f64_sub(a: I64, b: I64) -> I64 {
    return f64_add(a, I64(b.lo, b.hi ^ 0x80000000u));
}

// An i32 as a double, exactly.
fn f64_from_i32(v: i32) -> I64 {
    if v == 0 {
        return f64_zero(0u);
    }
    let sign = select(0u, 0x80000000u, v < 0);
    let mag = select(u32(v), u32(-(v + 1)) + 1u, v < 0);
    let top = 31u - countLeadingZeros(mag);
    // The significand at bit 52.
    let m = u64_shl(I64(mag, 0u), 52u - top);
    return I64(m.lo, sign | ((1023u + top) << 20u) | (m.hi & 0xfffffu));
}

// The integer part of |a| and whether a fraction was dropped, for |a| in
// [1, 2^31).
fn f64_trunc_parts(a: I64) -> vec2<u32> {
    let e = f64_exp(a) - 1023u;
    let s = f64_sig(a);
    let ip = u64_shr(s, 52u - e);
    let back = u64_shl(ip, 52u - e);
    return vec2<u32>(ip.lo, select(0u, 1u, !i64_eq(back, s)));
}

// a.floor() as i32 (Rust: saturating, NaN to 0).
fn f64_floor_i32(a: I64) -> i32 {
    if f64_is_nan(a) {
        return 0;
    }
    let neg = f64_sign(a) != 0u;
    let e = f64_exp(a);
    if e < 1023u {
        return select(0, -1, neg && !f64_is_zero(a));
    }
    if e >= 1023u + 31u {
        return select(2147483647, -2147483647 - 1, neg);
    }
    let t = f64_trunc_parts(a);
    if neg {
        return -i32(t.x) - i32(t.y);
    }
    return i32(t.x);
}

// a.ceil() as i32.
fn f64_ceil_i32(a: I64) -> i32 {
    if f64_is_nan(a) {
        return 0;
    }
    let neg = f64_sign(a) != 0u;
    let e = f64_exp(a);
    if e < 1023u {
        return select(0, 1, !neg && !f64_is_zero(a));
    }
    if e >= 1023u + 31u {
        return select(2147483647, -2147483647 - 1, neg);
    }
    let t = f64_trunc_parts(a);
    if neg {
        return -i32(t.x);
    }
    return i32(t.x + t.y);
}

// a >= b (false with a NaN).
fn f64_ge(a: I64, b: I64) -> bool {
    if f64_is_nan(a) || f64_is_nan(b) {
        return false;
    }
    if f64_is_zero(a) && f64_is_zero(b) {
        return true;
    }
    let na = f64_sign(a) != 0u && !f64_is_zero(a);
    let nb = f64_sign(b) != 0u && !f64_is_zero(b);
    if na != nb {
        return nb;
    }
    if na {
        return !f64_mag_lt(b, a) || i64_eq(a, b);
    }
    return !f64_mag_lt(a, b);
}
