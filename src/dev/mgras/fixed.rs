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
}
