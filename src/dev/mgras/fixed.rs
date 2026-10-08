//! The raster subsystem's colour arithmetic: 12.16 fixed point.
//!
//! The RE4 and TE1 are integer machines: iterators are fixed point (the
//! colour registers 12.12, the edge slopes s.24), texels are 4-, 8- or
//! 12-bit integers. Colour from the iterators through texturing, fog and
//! blending is modelled as 12.16 in an `i32`: twelve integer bits of
//! colour (a register's full 0xFFF000 is 4095 << 16) and sixteen of
//! fraction. Products are `(a * b) >> 28`, so the identity is 1 << 28
//! (4096.0) and full white times white loses 1/4096, which no 8-bit pixel
//! shows. Writes shift down to the pixel's size.
//!
//! The interpreter (`rss.rs`, `te1.rs`) and the raster JIT
//! (`rss_jit::compiler`) both compute with these definitions: change one
//! and change the other.

/// The multiplicative identity: 4096.0.
pub const ONE: i32 = 1 << 28;
/// The brightest colour, just below 4096.
pub const MAX: i32 = ONE - 1;
/// A register's 1.0 (0xFFF000, colour 4095) in 12.16.
pub const WHITE: f64 = 4095.0 * 65536.0;

/// `a * b`, 12.16.
#[inline]
pub fn mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> 28) as i32
}

/// Clamp to [0, MAX].
#[inline]
pub fn clamp(v: i32) -> i32 {
    v.clamp(0, MAX)
}

/// A colour plane value (1.0 = a register's full colour) in 12.16.
#[inline]
pub fn colour(v: f64) -> i32 {
    (v * WHITE).round() as i32
}

/// A fraction (fog factor, 1.0 = ONE) in 12.16.
#[inline]
pub fn fraction(v: f64) -> i32 {
    (v * ONE as f64).round() as i32
}

/// A component of `d` nibbles widened to 12 bits by repeating its bits.
#[inline]
pub fn widen(v: u32, d: usize) -> u32 {
    match d {
        1 => v * 0x111,
        2 => v << 4 | v >> 4,
        _ => v,
    }
}

/// A 12-bit colour field of a register in 12.16.
#[inline]
pub fn field12(v: u32) -> i32 {
    ((v & 0xFFF) << 16) as i32
}

/// A clamped colour as a pixel byte.
#[inline]
pub fn to_byte(c: i32) -> u32 {
    (clamp(c) >> 20) as u32
}

/// A clamped colour as a 12-bit colour index.
#[inline]
pub fn to_index(c: i32) -> u32 {
    (clamp(c) >> 16) as u32
}

/// A plane (value, d/dx, d/(-y)) at pixel centres, as the triangle
/// setup's planes are (value at (xs, yref), relative to centres x + 0.5,
/// y + 0.5), in 12.16 by `conv`: the value at pixel (xs, yref) and the
/// integer steps. Pixel (i, j) is then `p[0] + p[1] (i - xs) + p[2]
/// (yref - j)`, wrapping.
pub fn plane(p: [f64; 3], conv: fn(f64) -> i32) -> [i32; 3] {
    [conv(p[0] + 0.5 * p[1] - 0.5 * p[2]), conv(p[1]), conv(p[2])]
}

/// A fixed plane at pixel (i, j) of a triangle with origin (xs, yref).
#[inline]
pub fn at(p: [i32; 3], di: i32, dj: i32) -> i32 {
    p[0].wrapping_add(p[1].wrapping_mul(di)).wrapping_add(p[2].wrapping_mul(dj))
}

/// Depth: z.12 in an `i64`, as the Z registers hold it.
pub const Z_FRAC: u32 = 12;

/// A depth plane (window z units) in z.12, like `plane`.
pub fn zplane(p: [f64; 3]) -> [i64; 3] {
    let c = |v: f64| (v * (1u64 << Z_FRAC) as f64).round() as i64;
    [c(p[0] + 0.5 * p[1] - 0.5 * p[2]), c(p[1]), c(p[2])]
}

/// A depth plane at pixel (xs + di, yref - dj).
#[inline]
pub fn zat(p: [i64; 3], di: i32, dj: i32) -> i64 {
    p[0].wrapping_add(p[1].wrapping_mul(di as i64)).wrapping_add(p[2].wrapping_mul(dj as i64))
}

/// A z.12 depth rounded to the 24-bit buffer value.
#[inline]
pub fn zbuf(z: i64) -> u64 {
    (z.wrapping_add(1 << (Z_FRAC - 1)) >> Z_FRAC).clamp(0, 0xFF_FFFF) as u64
}

// ── texture coordinates ─────────────────────────────────────────────────────

/// S/W, T/W and 1/W: i64 at 2^32, as their registers (`te1::ITER_ONE`).
pub const ITER_FRAC: u32 = 32;

/// An iterator value at 2^32.
#[inline]
pub fn iter(v: f64) -> i64 {
    (v * (1u64 << ITER_FRAC) as f64).round() as i64
}

/// An iterator plane at pixels, like `plane`.
pub fn iter_plane(p: [f64; 3]) -> [i64; 3] {
    [iter(p[0] + 0.5 * p[1] - 0.5 * p[2]), iter(p[1]), iter(p[2])]
}

/// An i64 plane at pixel (xs + di, yref - dj).
#[inline]
pub fn at64(p: [i64; 3], di: i32, dj: i32) -> i64 {
    p[0].wrapping_add(p[1].wrapping_mul(di as i64)).wrapping_add(p[2].wrapping_mul(dj as i64))
}

/// Reciprocal seeds: 1 / (1 + (i + 0.5) / 1024) in Q31, for mantissas
/// whose 10 bits after the leading one are i.
pub static RECIP: [u32; 1024] = {
    let mut t = [0u32; 1024];
    let mut i = 0;
    while i < 1024 {
        let d = 2049 + 2 * i as u64;
        t[i] = (((1u64 << 42) + d / 2) / d) as u32;
        i += 1;
    }
    t
};

/// The reciprocal unit: for `wi` > 0, `(y, n)` with `wi = m 2^n`
/// (1 <= m < 2) and `y` = 1/m in Q31 (a table seed and one Newton step,
/// good to about 2^-22). Then `v / wi` is `(v y) >> (31 + n)`.
#[inline]
pub fn recip(wi: i64) -> (u64, u32) {
    let n = 63 - (wi as u64).leading_zeros();
    // The mantissa in Q30.
    let x = if n >= 30 { (wi as u64) >> (n - 30) } else { (wi as u64) << (30 - n) };
    let y0 = RECIP[((x >> 20) & 0x3FF) as usize] as u64;
    let e = (x * y0) >> 30;
    let y = (y0 * ((1u64 << 32) - e)) >> 31;
    (y, n)
}

/// `v / wi` in Q31, from `recip(wi)`.
#[inline]
pub fn persp(v: i64, y: u64, n: u32) -> i64 {
    ((v as i128 * y as i128) >> n) as i64
}

/// log2(1 + i/256) in Q8 (0..255).
pub fn log2_table() -> &'static [u16; 256] {
    static T: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| std::array::from_fn(|i| ((1.0 + i as f64 / 256.0).log2() * 256.0).round() as u16))
}

/// log2 in Q8 of `v` > 0: the leading one's position and a table of the
/// next 8 bits.
#[inline]
pub fn log2_q8(v: u64) -> i32 {
    let n = 63 - v.leading_zeros();
    let f = if n >= 8 { v >> (n - 8) } else { v << (8 - n) } & 0xFF;
    ((n as i32) << 8) + log2_table()[f as usize] as i32
}

/// The level of detail when the footprint is empty (magnification).
pub const LOD_NONE: i32 = -(1 << 24);

/// A derivative of a perspective-divided coordinate: `a` (2^32, the
/// numerator's step less the coordinate times 1/W's step) over 1/W, in
/// level-0 texels (2^l of them) at Q16, clamped to +-2^31.
#[inline]
pub fn deriv(a: i64, y: u64, n: u32, l: u32) -> i64 {
    (((a as i128 * y as i128) >> (n + 15 - l)) as i64).clamp(-(1 << 31), 1 << 31)
}

/// The level of detail in Q8 from the four footprint derivatives (Q16
/// texels): half the log2 of the larger sum of squares (so no square
/// root), LOD_NONE when it is empty.
#[inline]
pub fn lod_q8(dsx: i64, dtx: i64, dsy: i64, dty: i64) -> i32 {
    let ax = (dsx * dsx) as u64 + (dtx * dtx) as u64;
    let ay = (dsy * dsy) as u64 + (dty * dty) as u64;
    let m = ax.max(ay);
    if m == 0 { LOD_NONE } else { (log2_q8(m) - (32 << 8)) >> 1 }
}

/// A perspective-divided coordinate times `c`'s step, at 2^32: `(c v) >> 31`
/// for `v` in Q31.
#[inline]
pub fn scale31(v: i64, c: i64) -> i64 {
    ((v as i128 * c as i128) >> 31) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_survives_to_a_byte() {
        let w = colour(1.0);
        assert_eq!(w, 4095 << 16);
        assert_eq!(to_byte(mul(w, w)), 255);
        assert_eq!(mul(w, ONE), w);
        assert_eq!(widen(0xFF, 2), 0xFFF);
        assert_eq!(widen(0xF, 1), 0xFFF);
        assert_eq!(to_byte((widen(0x80, 2) << 16) as i32), 0x80);
    }

    #[test]
    fn reciprocal_and_log2_are_close() {
        for k in 1..20_000i64 {
            let wi = k * 7_919_321 + 13;
            let (y, n) = recip(wi);
            let got = persp(1 << 40, y, n) as f64 / (1u64 << 31) as f64;
            let want = (1u64 << 40) as f64 / wi as f64;
            assert!(((got - want) / want).abs() < 1e-6, "{wi}: {got} {want}");
        }
        for k in 1..10_000u64 {
            let v = k * k * 977 + 1;
            assert!((log2_q8(v) as f64 / 256.0 - (v as f64).log2()).abs() < 0.008, "{v}");
        }
        assert_eq!(log2_q8(1 << 32), 32 << 8);
    }
}
