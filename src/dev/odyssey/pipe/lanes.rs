//! A quad's four pixels on the host's SIMD unit (fearless_simd), for the
//! lane arithmetic LLVM leaves scalar: the saturating float-to-integer
//! conversions of the texture unit and the colour weights, and the walk's
//! depth. Each kernel computes, lane by lane, what the scalar code it
//! stands for computes, bit for bit: the same operations in the same order,
//! no fused multiply-add, and Rust's saturating casts.
//!
//! The kernels are generic over `Simd` and `#[inline(always)]`; their
//! callers dispatch on `level()`. A build whose target has the best level
//! the crate knows (`target-cpu=native` on an AVX2 host, any aarch64) has
//! only that level, and the dispatch folds away; a portable build picks
//! its level at run time.

use std::sync::LazyLock;

use fearless_simd::{f32x4, i32x4, Level, Simd};

static LEVEL: LazyLock<Level> = LazyLock::new(Level::new);

/// The host's SIMD level, detected once.
#[inline(always)]
pub(crate) fn level() -> Level {
    *LEVEL
}

/// `x as i32 - (x < (x as i32) as f32) as i32` at four lanes, the
/// subtraction wrapping: floor(x) for |x| below 2^31; beyond, the cast
/// saturates.
#[inline(always)]
pub(crate) fn floor_i32<S: Simd>(s: S, x: f32x4<S>) -> i32x4<S> {
    let i = s.cvt_i32_precise_f32x4(x);
    let below = s.simd_lt_f32x4(x, s.cvt_f32_i32x4(i));

    s.add_i32x4(i, s.select_i32x4(below, s.splat_i32x4(-1), s.splat_i32x4(0)))
}

/// A lane mask as bits, lane i in bit i.
#[inline(always)]
pub(crate) fn bits<S: Simd>(s: S, m: fearless_simd::mask32x4<S>) -> u8 {
    s.as_array_mask32x4(m).iter().enumerate().fold(0, |b, (i, &v)| b | ((v != 0) as u8) << i)
}

/// Tests: run `$body` with `$s` bound to each SIMD level this host can
/// run, and `$name` to the level's name: the scalar fallback always, SSE4.2
/// and AVX2 on x86 and NEON on aarch64 where the CPU has them. The kernels
/// must give the same answers on every one; the dispatch takes the best.
/// Evaluates to the names of the levels it ran.
#[cfg(test)]
macro_rules! each_level {
    ($s:ident, $name:ident => $body:block) => {{
        #[allow(unused_mut)]
        let mut ran: Vec<&'static str> = Vec::new();
        {
            let ($s, $name) = (fearless_simd::Fallback::new(), "fallback");
            fearless_simd::Simd::vectorize($s, || $body);
            ran.push($name);
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if let Some($s) = fearless_simd::Level::new().as_sse4_2() {
                let $name = "sse4.2";
                fearless_simd::Simd::vectorize($s, || $body);
                ran.push($name);
            }
            if let Some($s) = fearless_simd::Level::new().as_avx2() {
                let $name = "avx2";
                fearless_simd::Simd::vectorize($s, || $body);
                ran.push($name);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if let Some($s) = fearless_simd::Level::new().as_neon() {
                let $name = "neon";
                fearless_simd::Simd::vectorize($s, || $body);
                ran.push($name);
            }
        }
        ran
    }};
}
#[cfg(test)]
pub(crate) use each_level;

#[cfg(test)]
mod tests {
    use super::*;

    /// floor_i32() is the scalar cast and step at every value: halves,
    /// the edges of i32, past them, NaN, the infinities and denormals.
    #[test]
    fn the_lane_floor_is_the_scalar_casts() {
        let scalar = |x: f32| {
            let i = x as i32;
            i.wrapping_sub((x < i as f32) as i32)
        };
        let mut xs = vec![0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 8388607.5, -8388607.5, 2147483520.0, 2147483648.0, -2147483648.0, -2147483904.0, 3e9, -3e9];
        xs.extend([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1e-40, -1e-40, f32::MAX, f32::MIN]);
        let mut seed = 0x9e37_79b9u32;
        for _ in 0..100_000 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            xs.push(f32::from_bits(seed));
        }
        let ran = each_level!(s, level => {
            for c in xs.chunks(4) {
                let mut x = [0.0f32; 4];
                x[..c.len()].copy_from_slice(c);
                let got = s.as_array_i32x4(floor_i32(s, s.load_array_f32x4(x)));
                for i in 0..4 {
                    assert_eq!(got[i], scalar(x[i]), "{level}: {:e}", x[i]);
                }
            }
        });
        // The fallback, and a SIMD level on the hosts the crate knows.
        assert!(ran.len() >= 1 + cfg!(any(target_arch = "x86_64", target_arch = "aarch64")) as usize, "{ran:?}");
    }
}
