// 64-bit integers on two 32-bit halves, for the boards' exact edge
// functions: WGSL has no i64 everywhere, and a driver's i64 to f32
// conversion is not guaranteed to round as the host's does. Only what
// the kernels need: add, multiply by a small unsigned, compare with 0
// and 1, and a correctly rounded (nearest even) conversion to f32, which
// is what Rust's `as f32` does.

struct I64 {
    lo: u32,
    hi: u32,
}

fn i64_add(a: I64, b: I64) -> I64 {
    let lo = a.lo + b.lo;
    let carry = select(0u, 1u, lo < a.lo);
    return I64(lo, a.hi + b.hi + carry);
}

// a * k for k below 2^16, wrapping as a two's complement product does.
fn i64_mul_u16(a: I64, k: u32) -> I64 {
    let p0 = (a.lo & 0xffffu) * k;
    let p1 = (a.lo >> 16u) * k;
    let lo = p0 + (p1 << 16u);
    let carry = select(0u, 1u, lo < p0);
    return I64(lo, a.hi * k + (p1 >> 16u) + carry);
}

fn i64_is_neg(a: I64) -> bool {
    return (a.hi & 0x80000000u) != 0u;
}

// a >= need, for need 0 or 1.
fn i64_ge_need(a: I64, need: u32) -> bool {
    if i64_is_neg(a) {
        return false;
    }
    return need == 0u || (a.lo | a.hi) != 0u;
}

fn i64_neg(a: I64) -> I64 {
    let lo = ~a.lo + 1u;
    let carry = select(0u, 1u, lo == 0u);
    return I64(lo, ~a.hi + carry);
}

// f32(a), rounded to nearest even.
fn i64_to_f32(a: I64) -> f32 {
    var m = a;
    let neg = i64_is_neg(a);
    if neg {
        m = i64_neg(a);
    }
    if (m.lo | m.hi) == 0u {
        return 0.0;
    }
    // The magnitude's bit length.
    var n: u32;
    if m.hi != 0u {
        n = 64u - countLeadingZeros(m.hi);
    } else {
        n = 32u - countLeadingZeros(m.lo);
    }
    var r: f32;
    if n <= 24u {
        // Exact.
        r = f32(m.lo);
    } else {
        // The top 25 bits (24 to keep and the rounding bit), and whether
        // anything below them is set.
        let s = n - 25u;
        var top: u32;
        var sticky: bool;
        if s >= 32u {
            let t = s - 32u;
            top = m.hi >> t;
            sticky = m.lo != 0u || (t != 0u && (m.hi & ((1u << t) - 1u)) != 0u);
        } else if s == 0u {
            top = m.lo;
            sticky = false;
        } else {
            top = (m.lo >> s) | (m.hi << (32u - s));
            sticky = (m.lo & ((1u << s) - 1u)) != 0u;
        }
        var keep = top >> 1u;
        let round_bit = (top & 1u) != 0u;
        if round_bit && (sticky || (keep & 1u) != 0u) {
            keep += 1u;
        }
        // keep is at most 2^24: exact in f32, and the scale is a power of two.
        r = ldexp(f32(keep), i32(s + 1u));
    }
    return select(r, -r, neg);
}
