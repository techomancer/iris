//! The shader's integer arithmetic: colour in 12.12, in 32 bits
//! (`clanker/plans/odyssey-quad-pipeline.md`, milestone 7).
//!
//! A colour component is an `i32` with twelve integer and twelve fraction
//! bits: `ONE` (1 << 24) is 4096.0, and the brightest stored colour is
//! `MAX`, just below it, as on IMPACT (`mgras/fixed.rs`). A 12-bit register
//! value v is `v << 12`; an 8-bit plane value is the top eight bits of the
//! twelve. Products keep to 32 bits: each factor gives up low fraction bits
//! first, so a product of two colours is `(a >> 8) * (b >> 8) >> 8`.
//!
//! Interpolation weights are 0.15: three perspective-correct weights that
//! sum to `W_ONE` exactly, so a component equal at every vertex comes out
//! unchanged to its 12.4 (the precision vertex colours interpolate at).

/// 1.0 (4096.0) in 12.12.
pub(crate) const ONE: i32 = 1 << 24;
/// The brightest colour, just below 1.0.
pub(crate) const MAX: i32 = ONE - 1;
/// The weights' 1.0.
pub(crate) const W_ONE: i32 = 1 << 15;

/// Clamp to [0, MAX].
#[inline(always)]
pub(crate) fn clamp(v: i32) -> i32 {
    v.clamp(0, MAX)
}

/// A colour in [0, 1] as the model's doubles carry it, in 12.12: the
/// conversion at the edge of the integer shader.
#[inline(always)]
pub(crate) fn from_f64(c: f64) -> i32 {
    if !(c > 0.0) {
        0
    } else if c >= 1.0 {
        MAX
    } else {
        ((c * ONE as f64) as i32).min(MAX)
    }
}

/// A 12.12 colour as a double, for the paths still in double.
#[inline(always)]
pub(crate) fn to_f64(c: i32) -> f64 {
    c as f64 / ONE as f64
}

/// A 12-bit register field in 12.12 at 1/4096 (the fog colour's reading).
#[inline(always)]
pub(crate) fn field12(v: u32) -> i32 {
    ((v & 0xfff) << 12) as i32
}

/// A 12-bit register field in 12.12 at 1/4095 (the environment colour's
/// reading): v + v/4095 of a step, as widening repeats the top bits.
#[inline(always)]
pub(crate) fn field12_full(v: u32) -> i32 {
    let v = (v & 0xfff) as i32;

    (v << 12) + v
}

/// Two colours' product: 4096.0 times 4096.0 is 4096.0.
#[inline(always)]
pub(crate) fn mul(a: i32, b: i32) -> i32 {
    // Each factor in [0, ONE]: 16 bits each, the product in 32 unsigned.
    (((a as u32 >> 8) * (b as u32 >> 8)) >> 8) as i32
}

/// A signed difference (|d| <= ONE) times a colour or fraction in [0, ONE].
#[inline(always)]
pub(crate) fn smul(d: i32, b: i32) -> i32 {
    // 15 bits and sign times 15 bits: 31 bits.
    ((d >> 9) * (b >> 9)) >> 6
}

/// a + (b - a) f: f in [0, ONE].
#[inline(always)]
pub(crate) fn lerp(a: i32, b: i32, f: i32) -> i32 {
    a + smul(b - a, f)
}

#[cfg(test)]
/// A clamped colour as an 8-bit plane value: floor(256 c), at most 255,
/// as `backend::quantise8` takes a double.
#[inline(always)]
pub(crate) fn to8(c: i32) -> u8 {
    (clamp(c) >> 16) as u8
}

/// The alpha test's 12-bit value: min(4095, floor(4096 a)).
#[inline(always)]
pub(crate) fn alpha12(a: i32) -> u32 {
    (clamp(a) >> 12) as u32
}

#[cfg(test)]
/// Perspective-correct weights of a fragment at barycentrics (b1, b2) of
/// a triangle whose vertices have 1/w inv_w, in 0.15, summing to W_ONE;
/// None where they do not exist (the weights' sum is not positive).
#[inline(always)]
pub(crate) fn weights(b1: f64, b2: f64, inv_w: [f64; 3]) -> Option<[i32; 3]> {
    let w0 = (1.0 - b1 - b2) * inv_w[0];
    let w1 = b1 * inv_w[1];
    let w2 = b2 * inv_w[2];
    let sum = w0 + w1 + w2;

    if !(sum > 0.0) && !(sum < 0.0) {
        return None;
    }
    let k = W_ONE as f64 / sum;
    let p1 = ((w1 * k + 0.5) as i32).clamp(0, W_ONE);
    let p2 = ((w2 * k + 0.5) as i32).clamp(0, W_ONE - p1);

    Some([W_ONE - p1 - p2, p1, p2])
}

/// A component interpolated by weights p from vertex values c in 12.12.
#[inline(always)]
pub(crate) fn interpolate(p: &[i32; 3], c: [i32; 3]) -> i32 {
    // Each vertex value to 12.4 (16 bits) times a 0.15 weight: the sum
    // stays below 2^31; back to 12.12.
    let s = (c[0] >> 8) * p[0] + (c[1] >> 8) * p[1] + (c[2] >> 8) * p[2];

    s >> 7
}

/// log2(1 + i/256) in 1/256, i in 0..256.
fn log2_table() -> &'static [u16; 256] {
    static T: std::sync::OnceLock<[u16; 256]> = std::sync::OnceLock::new();

    T.get_or_init(|| std::array::from_fn(|i| ((1.0 + i as f64 / 256.0).log2() * 256.0).round() as u16))
}

/// log2(v) in 1/256 for v > 0: the leading one's position and a table of
/// the eight bits after it.
#[inline(always)]
pub(crate) fn log2_q8(v: u32) -> i32 {
    let n = 31 - v.leading_zeros();
    let f = if n >= 8 { v >> (n - 8) } else { v << (8 - n) } & 0xff;

    ((n as i32) << 8) + log2_table()[f as usize] as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_and_black_survive_the_arithmetic() {
        assert_eq!(to8(from_f64(1.0)), 255);
        assert_eq!(to8(mul(MAX, MAX)), 255);
        assert_eq!(alpha12(MAX), 4095);
        assert_eq!(alpha12(from_f64(0.5)), 2048);
    }

    #[test]
    fn eight_bit_output_floors_as_the_double_quantiser() {
        for i in 0..=4096 {
            let c = i as f64 / 4096.0;
            assert_eq!(to8(from_f64(c)) as u32, super::super::backend::quantise8(c), "{c}");
        }
    }

    #[test]
    fn texels_widen_to_their_value() {
        // As the texture unit widens an 8-bit texel: to 12 bits by repeating
        // its top bits, then over 4095 into 12.12.
        for v in 0..=255u32 {
            let c = (((v << 4) | (v >> 4)) * 4097) as i32;
            assert!((to_f64(c) - v as f64 / 255.0).abs() < 1.0 / 4096.0, "{v}");
            assert_eq!(to8(c) as u32, v, "{v}");
        }
    }

    #[test]
    fn log2_is_close() {
        for v in 1..200_000u32 {
            assert!((log2_q8(v) as f64 / 256.0 - (v as f64).log2()).abs() < 0.008, "{v}");
        }
        assert_eq!(log2_q8(256), 8 << 8);
    }

    #[test]
    fn weights_sum_to_one_and_keep_equal_values() {
        let p = weights(0.3, 0.25, [1.0, 0.5, 2.0]).unwrap();
        assert_eq!(p.iter().sum::<i32>(), W_ONE);
        let c = from_f64(0.7);
        assert_eq!(interpolate(&p, [c, c, c]), c & !0xff);
        assert!(weights(0.0, 0.0, [0.0, 0.0, 0.0]).is_none());
    }
}
