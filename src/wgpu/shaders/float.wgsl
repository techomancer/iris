// Floating point the boards' models can be held to. Vulkan, Metal and
// D3D guarantee correctly rounded f32 add, subtract and multiply, but a
// quotient may be off by a few units in the last place (RADV computes
// it as a reciprocal and a multiply), and nothing stops a compiler fusing
// a multiply into an add. The kernels keep a product apart from a sum by
// multiplying it by a one the compiler cannot see (each board's `p()`),
// and divide here: the quotient of the two mantissas as integers, rounded
// to nearest even, which is what the host's division gives.

// a / b, correctly rounded, for normal operands; the hardware's quotient
// for zero, denormal, infinite and NaN ones (the model's range tests
// drop those fragments either way).
fn div_exact(a: f32, b: f32) -> f32 {
    let ia = bitcast<u32>(a);
    let ib = bitcast<u32>(b);
    let ea = (ia >> 23u) & 0xffu;
    let eb = (ib >> 23u) & 0xffu;
    if ea == 0u || eb == 0u || ea == 255u || eb == 255u {
        return a / b;
    }
    let ma = (ia & 0x7fffffu) | 0x800000u;
    let mb = (ib & 0x7fffffu) | 0x800000u;
    // (ma << 26) / mb by restoring division: 26 or 27 quotient bits, the
    // remainder below mb.
    var rem: u32;
    var q: u32;
    if ma >= mb {
        rem = ma - mb;
        q = 1u;
    } else {
        rem = ma;
        q = 0u;
    }
    for (var i = 0u; i < 26u; i++) {
        rem = rem << 1u;
        q = q << 1u;
        if rem >= mb {
            rem -= mb;
            q |= 1u;
        }
    }
    let n = select(26u, 27u, q >= (1u << 26u));
    let drop = n - 24u;
    var m = q >> drop;
    let round_bit = (q >> (drop - 1u)) & 1u;
    let sticky = (q & ((1u << (drop - 1u)) - 1u)) != 0u || rem != 0u;
    if round_bit == 1u && (sticky || (m & 1u) == 1u) {
        m += 1u;
    }
    // a / b = m * 2^(n - 50 + ea - eb); m is at most 2^24, exact in f32,
    // and the scale is a power of two.
    let e = i32(ea) - i32(eb) + i32(n) - 50;
    let sign = (ia ^ ib) & 0x80000000u;
    return bitcast<f32>(bitcast<u32>(ldexp(f32(m), e)) | sign);
}
