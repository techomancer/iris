//! The perspective unit: a filled triangle's interpolants at a quad's four
//! pixels, from their barycentrics (`clanker/plans/odyssey-quad-pipeline.md`,
//! milestone 7).
//!
//! The walk gives every pixel its barycentrics l1, l2 (the edge values over
//! the area, `raster::Triangle`). Every interpolant follows from them and
//! the vertex values: one linear in the window is l0 a0 + l1 a1 + l2 a2;
//! a perspective-correct one is the same sum of a/w over the sum of 1/w.
//! The triangle keeps its vertex values once (`TriTex`, in the triangle's
//! record); no plane is fitted. A quad's texture values go to the texture
//! unit as they are (`texture::TexQuad`), which divides by Q itself; the
//! colour weights are 0.15 (`fixed`).
//!
//! A repeating axis's S/W values are shifted by whole repeats so the
//! triangle's coordinates sit near zero: 32-bit floats then keep the
//! sub-texel bits of a floor repeated hundreds of times, and the shift
//! leaves every wrapped texel where it was.

use super::fixed;
use super::texture::TexQuad;

/// A triangle's texture values at its vertices: S/W, T/W, R/W and Q/W, a
/// repeating axis's shifted by whole repeats.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(crate) struct TriTex {
    v: [[f32; 3]; 4],
}

/// a0 + l1 (a1 - a0) + l2 (a2 - a0).
#[inline(always)]
fn lerp3(a: &[f32; 3], l1: f32, l2: f32) -> f32 {
    a[0] + l1 * (a[1] - a[0]) + l2 * (a[2] - a[0])
}

/// The perspective-correct colour weights at barycentrics (l1, l2) of a
/// triangle whose vertices have 1/w iw, in 0.15, summing to W_ONE; None
/// where they do not exist (the sum of 1/w is zero).
#[inline(always)]
pub(crate) fn weights(l1: f32, l2: f32, iw: &[f32; 3]) -> Option<[i32; 3]> {
    let w1 = l1 * iw[1];
    let w2 = l2 * iw[2];
    let sum = (1.0 - l1 - l2) * iw[0] + w1 + w2;

    if sum == 0.0 || !sum.is_finite() {
        return None;
    }
    let r = fixed::W_ONE as f32 / sum;
    let p1 = ((w1 * r + 0.5) as i32).clamp(0, fixed::W_ONE);
    let p2 = ((w2 * r + 0.5) as i32).clamp(0, fixed::W_ONE - p1);

    Some([fixed::W_ONE - p1 - p2, p1, p2])
}

impl TriTex {
    /// The texture values of a triangle whose vertices have values vals
    /// (S/W, T/W, R/W, Q/W). A repeating axis is shifted by whole repeats
    /// of its value at the first vertex, where Q/W is positive: 32-bit
    /// floats then keep the sub-texel bits of a coordinate repeated hundreds
    /// of times, and the shift leaves every wrapped texel where it was.
    pub fn setup(vals: &[[f64; 4]; 3], repeat: [bool; 2]) -> TriTex {
        let mut v: [[f64; 3]; 4] = std::array::from_fn(|k| vals.map(|a| a[k]));
        let q = v[3];

        for a in 0..2 {
            if repeat[a] && q[0] > 0.0 {
                let k = (v[a][0] / q[0]).floor();
                if k.is_finite() && k.abs() < 1e15 {
                    for j in 0..3 {
                        v[a][j] -= k * q[j];
                    }
                }
            }
        }
        TriTex { v: v.map(|a| a.map(|x| x as f32)) }
    }

    /// The texture coordinates of a quad whose pixels' barycentrics are l1
    /// and l2.
    #[inline(always)]
    pub fn quad(&self, l1: &[f32; 4], l2: &[f32; 4]) -> TexQuad {
        TexQuad { c: std::array::from_fn(|i| self.v.map(|a| lerp3(&a, l1[i], l2[i]))) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The coordinates over Q agree with double arithmetic, modulo whole
    // repeats, over random perspective triangles and barycentrics.
    #[test]
    fn the_unit_agrees_with_double_precision() {
        let mut seed = 0x1234_5678u64;
        let mut rnd = |lo: f64, hi: f64| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (hi - lo) * ((seed >> 11) as f64 / (1u64 << 53) as f64)
        };
        let mut checked = 0;

        for _ in 0..300 {
            let inv_w: [f64; 3] = std::array::from_fn(|_| 1.0 / rnd(0.5, 20.0));
            let st: [[f64; 2]; 3] = std::array::from_fn(|_| [rnd(-40.0, 40.0), rnd(-40.0, 40.0)]);
            let vals: [[f64; 4]; 3] = std::array::from_fn(|i| [st[i][0] * inv_w[i], st[i][1] * inv_w[i], 0.0, inv_w[i]]);
            let tt = TriTex::setup(&vals, [true, true]);

            for _ in 0..20 {
                let mut l1 = [0.0f32; 4];
                let mut l2 = [0.0f32; 4];
                for i in 0..4 {
                    let a = rnd(0.0, 1.0);
                    (l1[i], l2[i]) = (a as f32, rnd(0.0, 1.0 - a) as f32);
                }
                let o = tt.quad(&l1, &l2);
                for i in 0..4 {
                    let (b1, b2) = (l1[i] as f64, l2[i] as f64);
                    let w = [(1.0 - b1 - b2) * inv_w[0], b1 * inv_w[1], b2 * inv_w[2]];
                    let sum = w[0] + w[1] + w[2];
                    for a in 0..2 {
                        let want = (w[0] * st[0][a] + w[1] * st[1][a] + w[2] * st[2][a]) / sum;
                        let got = (o.c[i][a] / o.c[i][3]) as f64;
                        let d = (got - want).rem_euclid(1.0);
                        assert!(d < 1e-3 || d > 1.0 - 1e-3, "{got} vs {want}");
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 10_000, "{checked}");
    }
}
