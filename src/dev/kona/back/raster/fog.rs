//! Fog, per fragment, on plain data.
//!
//! What is known of the pipe: it keeps a fog enable and
//! the mode, density, start, end and colour parameters (generic state rows 0xb60 to 0xb66, established
//! as rows), the fog parameter setter takes five words (established), and the raster side has fog
//! control, colour and depth-offset registers and a fog function table (shadow words 82 to 85 and
//! the table setter, all inferred). The pipe evaluates the function on the eye distance recovered
//! from the window depth, and builds a table of 12-bit unsigned entries indexed by that depth plus
//! an offset (the entry width is established; the depth indexing is inferred). The model keeps the
//! function and the 12-bit entry and approximates the table: it evaluates the function at each
//! fragment's eye distance, interpolated perspective-correctly, instead of at depth steps, then
//! rounds the factor to 12 bits (`ENTRY_ONE` is 1.0; the hardware's conversion constant is
//! unknown). The blend's rounding, the fog index of the colour-index mode and the order of fog
//! among the fragment operations (after the texture environment and before the alpha test,
//! OpenGL's order) are unknown for the hardware and the model's own.
//!
//! The factor is OpenGL's published one, clamped to 0..1: linear `(end - c) / (end - start)`,
//! `exp(-density c)` and `exp(-(density c)^2)`; the fragment's red, green and blue become
//! `f C + (1 - f) Cf` and alpha is untouched. `exp` is not the host's: `exp_neg` is single-precision
//! multiplies and adds only, so a result is the same on every host (this file is held to the same
//! rule as `pixel`).

use crate::dev::kona::back::raster::DrawState;

/// Which curve a draw's fog follows. A draw is run by a function specialised on it once, so a
/// fragment of a draw with fog off or of one curve does not test the others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Curve {
    Linear,
    Exp,
    Exp2,
}

/// Fog as a draw sees it, from the context's parameters when the draw state is built.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fog {
    pub curve: Curve,
    /// Linear: the end distance; the exponential curves: the density.
    pub a: f32,
    /// Linear: 1 / (end - start), or a very large step when they are equal (the equation divides
    /// by zero there; the model takes the limit, 1 before the end and 0 from it); unused otherwise.
    pub b: f32,
    /// The colour in 13-bit units (8191 is full intensity).
    pub colour: [f32; 3],
}

impl Fog {
    /// What a draw with fog off holds in place of fog, for the specialised loop that never reads it.
    pub const OFF: Fog = Fog { curve: Curve::Exp, a: 0.0, b: 0.0, colour: [0.0; 3] };

    /// The fog of `fog`'s parameters, or None with fog off.
    pub fn new(on: bool, mode: u32, density: f32, start: f32, end: f32, colour: [f32; 3]) -> Option<Fog> {
        use crate::dev::kona::defs::gl;
        on.then(|| {
            let curve = match mode {
                gl::LINEAR => Curve::Linear,
                gl::EXP2 => Curve::Exp2,
                _ => Curve::Exp,
            };
            let (a, b) = match curve {
                Curve::Linear => (end, if end == start { 1.0e30 } else { 1.0 / (end - start) }),
                _ => (density, 0.0),
            };
            Fog { curve, a, b, colour: colour.map(|c| c.clamp(0.0, 1.0) * 8191.0) }
        })
    }
}

/// `exp(-x)` for `x` >= 0, from multiplies and adds alone. `x` is split as `k ln 2 + r`; `e^-r` for `r`
/// under `ln 2` is a degree-9 Taylor polynomial (relative error about 1e-7, far below the 13-bit
/// steps the factor is spent on), and 2^-k is built from its exponent. Past 12 the result is under 6e-6 and
/// is taken as 0.
#[inline(always)]
pub fn exp_neg(x: f32) -> f32 {
    const LN2: f32 = 0.693_147_2;
    const LOG2E: f32 = 1.442_695;
    if !(x < 12.0f32) {
        return 0.0;
    }
    let x = x.max(0.0f32);
    let k = (x * LOG2E) as u32;
    let r = x - k as f32 * LN2;
    // 1 - r + r^2/2 - ... - r^9/362880, Horner's form.
    let mut p = -1.0f32 / 362880.0;
    for c in [1.0f32 / 40320.0, -1.0 / 5040.0, 1.0 / 720.0, -1.0 / 120.0, 1.0 / 24.0, -1.0 / 6.0, 0.5, -1.0, 1.0] {
        p = p * r + c;
    }
    p * f32::from_bits((127 - k) << 23)
}

impl Fog {
    /// The factor for fog coordinate `c`, 1 (no fog) to 0 (all fog).
    #[inline(always)]
    pub fn factor<const CURVE: u8>(&self, c: f32) -> f32 {
        let f = match CURVE {
            LINEAR => (self.a - c) * self.b,
            EXP => exp_neg(self.a * c),
            _ => {
                let u = self.a * c;
                exp_neg(u * u)
            }
        };
        // A table entry is 12 bits (approximated: nearest, 1.0 as all ones).
        ((f.clamp(0.0f32, 1.0f32) * ENTRY_ONE + 0.5f32) as u32) as f32 / ENTRY_ONE
    }

    /// Fragment colour `src` (13-bit red, green, blue, alpha) with factor `f` applied.
    #[inline(always)]
    pub fn apply(&self, src: [u32; 4], f: f32) -> [u32; 4] {
        let g = 1.0f32 - f;
        let m = |k: usize| (f * src[k] as f32 + g * self.colour[k] + 0.5f32) as u32;
        [m(0), m(1), m(2), src[3]]
    }
}

/// All ones in a 12-bit fog table entry.
const ENTRY_ONE: f32 = 4095.0;

/// The curve numbers `Fog::factor` is specialised on, and what `curve` names.
pub const LINEAR: u8 = 1;
pub const EXP: u8 = 2;
pub const EXP2: u8 = 3;

/// The curve number of a draw's fog, 0 with fog off.
pub fn curve_of(ds: &DrawState) -> u8 {
    match ds.fog.map(|f| f.curve) {
        None => 0,
        Some(Curve::Linear) => LINEAR,
        Some(Curve::Exp) => EXP,
        Some(Curve::Exp2) => EXP2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::defs::gl;

    #[test]
    fn exp_neg_follows_the_exponential_to_a_part_in_five_million() {
        let mut worst = 0.0f64;
        let mut x = 0.0f32;
        while x < 12.0 {
            let want = (-(x as f64)).exp();
            worst = worst.max(((exp_neg(x) as f64 - want) / want).abs());
            x += 0.0137;
        }
        assert!(worst < 1e-6, "worst relative error {worst}");
        assert_eq!(exp_neg(0.0), 1.0);
        assert_eq!(exp_neg(12.0), 0.0);
        assert_eq!(exp_neg(f32::NAN), 0.0, "not a number is all fog, not a panic");
        assert_eq!(exp_neg(-3.0), 1.0, "a negative argument is no fog");
    }

    fn fog(mode: u32) -> Fog {
        Fog::new(true, mode, 0.5, 2.0, 6.0, [1.0, 0.5, 0.0]).expect("on")
    }

    #[test]
    fn off_is_none_and_the_curves_are_the_published_ones() {
        assert!(Fog::new(false, gl::EXP, 1.0, 0.0, 1.0, [0.0; 3]).is_none());
        let l = fog(gl::LINEAR);
        assert_eq!([l.factor::<LINEAR>(2.0), l.factor::<LINEAR>(6.0), l.factor::<LINEAR>(9.0), l.factor::<LINEAR>(0.0)], [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(l.factor::<LINEAR>(4.0), 2048.0 / ENTRY_ONE, "halfway is entry 2048 of 4095");
        let e = fog(gl::EXP);
        // The factor is a 12-bit entry, so it lands within half a step of the curve.
        let step = 0.5 / ENTRY_ONE + 1e-6;
        assert!((e.factor::<EXP>(2.0) - (-1.0f32).exp()).abs() < step);
        let e2 = fog(gl::EXP2);
        assert!((e2.factor::<EXP2>(2.0) - (-1.0f32).exp()).abs() < step, "exp(-(0.5 * 2)^2) = exp(-1)");
        assert!((e2.factor::<EXP2>(4.0) - (-4.0f32).exp()).abs() < step);
    }

    #[test]
    fn the_factor_is_a_twelve_bit_entry() {
        let l = fog(gl::LINEAR);
        for c in [2.0f32, 2.001, 3.3333, 4.5, 5.99] {
            let f = l.factor::<LINEAR>(c);
            assert_eq!(f * ENTRY_ONE, (f * ENTRY_ONE).round(), "{c}: {f}");
        }
        assert_eq!(l.factor::<LINEAR>(2.0), 1.0, "no fog is all ones");
    }

    #[test]
    fn a_linear_fog_with_start_at_end_steps_at_the_end() {
        let f = Fog::new(true, gl::LINEAR, 1.0, 3.0, 3.0, [0.0; 3]).expect("on");
        assert_eq!([f.factor::<LINEAR>(2.999), f.factor::<LINEAR>(3.0), f.factor::<LINEAR>(3.001)], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn the_blend_moves_red_green_and_blue_toward_the_fog_colour_and_leaves_alpha() {
        let f = fog(gl::LINEAR);
        let c = [8191, 0, 4000, 1234];
        assert_eq!(f.apply(c, 1.0), c);
        assert_eq!(f.apply(c, 0.0), [8191, 4096, 0, 1234], "all fog: the fog colour, 0.5 rounded to 4096");
        assert_eq!(f.apply([0, 8191, 0, 7], 0.5), [4096, 6143, 0, 7]);
    }

    #[test]
    fn the_draw_names_its_curve() {
        let mut ds = DrawState::default();
        assert_eq!(curve_of(&ds), 0);
        for (mode, want) in [(gl::LINEAR, LINEAR), (gl::EXP, EXP), (gl::EXP2, EXP2)] {
            ds.fog = Fog::new(true, mode, 1.0, 0.0, 1.0, [0.0; 3]);
            assert_eq!(curve_of(&ds), want);
        }
    }
}
