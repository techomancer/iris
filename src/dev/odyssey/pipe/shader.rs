//! The shader: a fragment's texture application, fog and alpha test, from
//! `ShaderWords` (`clanker/plans/odyssey-quad-pipeline.md`).
//!
//! The texel comes from the texture unit (`texture`); the shader decides
//! what happens to it. The environment, its colour, fog and the alpha test
//! are the words'.

use super::fixed;
use super::state::{Compare, ShaderWords};
use super::super::bits::sextract32;
use super::texture::{Class, Env, TexRegs};
use super::super::geometry::odyssey_f32;

/// Which of a texel's components the environment takes: alpha alone for an
/// alpha texture, and no alpha for one without.
#[inline(always)]
fn takes(class: Class, k: usize) -> bool {
    let alpha = k == 3;

    !((!alpha && class == Class::Alpha) || (alpha && (class == Class::Rgb || class == Class::Luminance)))
}

/// What the shader decides once per draw.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(64))]
pub(crate) struct ShaderState {
    /// The texel class the environment reads.
    pub class: Class,
    pub fog: bool,
    fog_words: [u32; 6],
    fog_colour: [f64; 3],
    /// The alpha test's comparison and 12-bit reference, where it is on.
    pub alpha: Option<(Compare, u32)>,
    /// The texture environment and its colour.
    pub env: Env,
    env_colour: [f64; 4],
    /// The fog and environment colours in 12.12 (`fixed`).
    fog_colour12: [i32; 3],
    env_colour12: [i32; 4],
}

impl ShaderState {
    pub fn of(w: &ShaderWords, t: &TexRegs) -> ShaderState {
        let fc = w.fog_colour();
        let ec = t.env_colour();

        ShaderState {
            class: t.class(),
            fog: w.fog_on(),
            fog_words: w.fog,
            // As the board's fog block carries it: 12 bits over 4096.
            fog_colour: [0, 1, 2].map(|k| (fc[k] as f32 / 4096.0f32) as f64),
            alpha: w.alpha_test_on().then(|| (w.alpha_func(), w.alpha_ref() as u32)),
            env: t.env(),
            // The environment colour: 12 bits over 4095, read only by GL_BLEND.
            env_colour: ec.map(|c| (c as f32 / 4095.0f32) as f64),
            fog_colour12: [0, 1, 2].map(|k| fixed::field12(fc[k] as u32)),
            env_colour12: ec.map(|c| fixed::field12_full(c as u32)),
        }
    }

    /// The fog factor at window depth z from the fog block: a scale and
    /// offset to a parameter t in [0, 1], then a cubic in Bernstein form
    /// through four 12.8 coefficients. The words' layout is established; the
    /// evaluation is the model's reading (`docs/tokens.md`, 0x01a), its sign
    /// and clamps unmeasured.
    pub fn fog_factor(&self, z: f32) -> f32 {
        let w = &self.fog_words;
        let scale = if (w[2] & 0xffffff) != 0 {
            f64::from(odyssey_f32(((w[2] & 0xffffff) << 5).wrapping_add(96u32 << 23)).abs())
        } else {
            0.0
        };
        let mut t = (scale * z as f64 + sextract32(w[3], 0, 12) as f64 / 256.0) * 2.0f64.powi(((w[3] >> 12) & 7) as i32);
        let p = [
            sextract32(w[4] >> 12, 0, 12) as f64 / 256.0,
            sextract32(w[4], 0, 12) as f64 / 256.0,
            sextract32(w[5] >> 12, 0, 12) as f64 / 256.0,
            sextract32(w[5], 0, 12) as f64 / 256.0,
        ];

        t = t.clamp(0.0, 1.0);
        let u = 1.0 - t;
        let f = p[0] * u * u * u + 3.0 * p[1] * u * u * t + 3.0 * p[2] * u * t * t + p[3] * t * t * t;
        (if f >= 1.0 { 1.0 } else if f > 0.0 { f } else { 0.0 }) as f32
    }

    /// A texel by the environment into rgba, for a texture of class
    /// `class`, in double.
    #[inline(always)]
    pub fn apply(&self, class: Class, texel: &[f64; 4], rgba: &mut [f64; 4]) {
        for k in 0..4usize {
            if !takes(class, k) {
                continue;
            }
            let alpha = k == 3;
            match self.env {
                Env::Replace => rgba[k] = texel[k],
                Env::Modulate => rgba[k] *= texel[k],
                Env::Add => rgba[k] = if alpha && class != Class::Intensity { rgba[k] * texel[k] } else { (rgba[k] + texel[k]).min(1.0) },
                Env::Decal => {
                    if !alpha {
                        rgba[k] = if class == Class::Rgb { texel[k] } else { rgba[k] * (1.0 - texel[3]) + texel[k] * texel[3] };
                    }
                }
                Env::Blend => {
                    rgba[k] = if alpha && class != Class::Intensity {
                        rgba[k] * texel[k]
                    } else {
                        rgba[k] * (1.0 - texel[k]) + self.env_colour[k].clamp(0.0, 1.0) * texel[k]
                    };
                }
            }
        }
    }

    /// ShaderState::apply() in 12.12.
    #[inline(always)]
    pub fn apply12(&self, class: Class, texel: &[i32; 4], rgba: &mut [i32; 4]) {
        for k in 0..4usize {
            if !takes(class, k) {
                continue;
            }
            let alpha = k == 3;
            match self.env {
                Env::Replace => rgba[k] = texel[k],
                Env::Modulate => rgba[k] = fixed::mul(rgba[k], texel[k]),
                Env::Add => rgba[k] = if alpha && class != Class::Intensity { fixed::mul(rgba[k], texel[k]) } else { fixed::clamp(rgba[k] + texel[k]) },
                Env::Decal => {
                    if !alpha {
                        rgba[k] = if class == Class::Rgb { texel[k] } else { fixed::lerp(rgba[k], texel[k], texel[3]) };
                    }
                }
                Env::Blend => {
                    rgba[k] = if alpha && class != Class::Intensity {
                        fixed::mul(rgba[k], texel[k])
                    } else {
                        fixed::lerp(rgba[k], self.env_colour12[k], texel[k])
                    };
                }
            }
        }
    }

    /// Shade a fragment of colour rgba in 12.12 at window depth zw: its
    /// texel (from the texture unit, for the draw's texel class)
    /// by the environment, fog and the alpha test. False where the fragment
    /// is dropped.
    #[inline]
    pub fn shade12(&self, textured: bool, rgba: &mut [i32; 4], texel: &[i32; 4], zw: f64) -> bool {
        if textured {
            self.apply12(self.class, texel, rgba);
        }
        if self.fog {
            let f = fixed::from_f64(self.fog_factor(zw as f32) as f64);
            // A factor of 1.0 keeps the colour.
            let f = if f == fixed::MAX { fixed::ONE } else { f };

            for k in 0..3usize {
                rgba[k] = fixed::lerp(self.fog_colour12[k], rgba[k], f);
            }
        }
        if let Some((func, reference)) = self.alpha {
            if !func.test(fixed::alpha12(rgba[3]), reference) {
                return false;
            }
        }
        true
    }

    /// The alpha test's 12-bit value of an alpha: min(4095, floor(4096 a))
    /// (inferred from the vendor frames, graded).
    #[inline(always)]
    fn alpha12(a: f64) -> u32 {
        if !(a > 0.0) {
            0
        } else if a >= 1.0 {
            4095
        } else {
            ((a * 4096.0) as u32).min(4095)
        }
    }

    /// Shade a fragment of a line, a point or a smoothed polygon, of colour
    /// rgba in double at window depth zw: its texel by the environment,
    /// fog, its alpha scaled by its coverage, and the alpha test. False
    /// where the fragment is dropped.
    #[inline]
    pub fn shade(&self, textured: bool, rgba: &mut [f64; 4], texel: &[i32; 4], zw: f64, coverage: f64) -> bool {
        if textured {
            self.apply(self.class, &texel.map(fixed::to_f64), rgba);
        }
        if self.fog {
            let f = self.fog_factor(zw as f32) as f64;
            let f = if f >= 1.0 { 1.0 } else if f > 0.0 { f } else { 0.0 };

            for k in 0..3usize {
                rgba[k] = f * rgba[k] + (1.0 - f) * self.fog_colour[k];
            }
        }
        rgba[3] *= coverage;
        if let Some((func, reference)) = self.alpha {
            // A reference of 0xfff is 1.0 sent clamped: the same 4095.
            if !func.test(ShaderState::alpha12(rgba[3]), reference) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::consts::mode;

    #[test]
    fn the_alpha_test_compares_twelve_bit_values() {
        // GEQUAL 0.5: 0x800.
        let s = ShaderState::of(&ShaderWords { mode: mode::ALPHA_TEST, alpha: (6 << 12) | 0x800, ..Default::default() }, &TexRegs::default());
        let (f, r) = s.alpha.unwrap();

        assert!(f.test(ShaderState::alpha12(0.5), r));
        assert!(!f.test(ShaderState::alpha12(0.4999), r));
        assert_eq!(ShaderState::alpha12(1.0), 4095);
        assert!(ShaderState::of(&ShaderWords::default(), &TexRegs::default()).alpha.is_none());
    }
}
