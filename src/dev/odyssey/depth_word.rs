//! The board's depth word: a 24-bit unsigned float.
//!
//! Bits 23:18 are the exponent e, bits 17:0 the mantissa m with an implied
//! leading one and no sign, so the value is (1 + m / 2^18) * 2^(e - 31),
//! from 2^-31 to just under 2^33. The stored value is 2S * (d - n) / (f - n)
//! for eye distance d and frustum distances n and f, with 2S = 2^33 - 2^15:
//! linear in eye distance, 0 at the near plane and 2S at the far plane.
//! Words order as unsigned integers do, so a depth compare is an integer
//! compare. Bits 31:24 of a 32-bit word are not depth.
//!
//! What is established is the library's reading of the plane after a DMA
//! read: the encoding (`encode_f64`), the 16-bit form and the polygon-offset
//! field. That the plane holds exactly these bits is INFERRED. The rounding
//! of a rasterised depth into the word is unknown; this model truncates, as
//! the library's encoder does.

use super::consts::ODSY_DEPTH24_MAX;

/// The depth range's far value, 2S = 2^33 - 2^15.
pub(super) const DEPTH_2S: f64 = 8589901824.0;

/// What the library's encoder returns for an exponent of 160 or more. Wider than
/// the word: the plane keeps the low 24 bits.
pub(super) const DEPTH_HOST_SATURATED: u32 = 0x0fffffff;

/// A value below 2^-31 has no exponent; the encoder maps it to 0.
const DEPTH_MIN_VALUE: f64 = 1.0 / 2147483648.0;
const DEPTH_EXP_BIAS: i32 = 31;
const DEPTH_MANTISSA_BITS: u32 = 18;
const DEPTH_MANTISSA_MASK: u32 = (1 << DEPTH_MANTISSA_BITS) - 1;

/// The library's encoder (ESTABLISHED), for a double: the exponent rebased
/// so that 2^-31 is 0, and the top 18 mantissa bits, the rest truncated. A
/// value below 2^-31 gives 0, one of 2^33 or more gives 0x0fffffff, and the
/// sign is ignored.
pub(super) fn encode_f64(v: f64) -> u32 {
    let v = v.abs();

    // NaN fails both tests and is unordered; treat it as below range.
    if !(v >= DEPTH_MIN_VALUE) {
        return 0;
    }
    let bits = v.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1023 + DEPTH_EXP_BIAS;

    if exp >= 64 {
        return DEPTH_HOST_SATURATED;
    }
    ((exp as u32) << DEPTH_MANTISSA_BITS) | ((bits >> (52 - DEPTH_MANTISSA_BITS)) as u32 & DEPTH_MANTISSA_MASK)
}

/// The word for an eye-linear depth fraction z in [0, 1] (0 at the near
/// plane, 1 at the far plane): the encoding of 2S * z. A fraction outside
/// [0, 1] clamps to it.
pub(super) fn word_for_fraction(z: f64) -> u32 {
    let z = if z > 1.0 { 1.0 } else if z > 0.0 { z } else { 0.0 };

    encode_f64(z * DEPTH_2S) & ODSY_DEPTH24_MAX
}

/// The 16-bit form (ESTABLISHED): the exponent in bits 15:10 and the top ten
/// mantissa bits in 9:0, which is the 24-bit word shifted right by eight. The
/// library reads it from windows whose visual is RGBA 12/12/12/12 and the
/// 24-bit form from every other visual with depth. Why those visuals get the
/// shorter word is not shown; the model has no depth plane of that width, so
/// a 16-bit depth read converts the 24-bit word (`ReadPlane::Depth16`).
pub(super) fn word16_from_word24(word: u32) -> u32 {
    (word & ODSY_DEPTH24_MAX) >> 8
}

/// The polygon-offset field (block 0x058) is a signed 25-bit count of steps
/// of the encoded 24-bit word, the limits the library clamps to.
pub(super) const OFFSET_STEPS_MIN: i64 = -(1 << 24);
pub(super) const OFFSET_STEPS_MAX: i64 = (1 << 24) - 1;

/// The word after a polygon offset. INFERRED placement: the board adds the field to the encoded word, so the offset
/// counts mantissa steps and shrinks toward the near plane in proportion to
/// d - n. The sum clamps to 0 and the largest word, a model choice: what the
/// board does at either end is not shown. The field alone is the step count;
/// the library has already folded factor and units into it ("Polygon offset").
pub(super) fn word_with_offset(word: u32, steps: i64) -> u32 {
    (word as i64 + steps).clamp(0, ODSY_DEPTH24_MAX as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::odyssey::raster_bench;
    use crate::dev::odyssey::consts::polygon;
    use crate::dev::odyssey::gl_context::{
        vtx, GL_COLOUR_MASK_ALL, GL_FLAT, GL_LEQUAL, GL_LESS, GL_OFFSET_FILL, GL_TRIANGLES, GlDraw,
        GlDrawState,
    };
    use crate::dev::odyssey::memory::ClipRect;
    use crate::dev::odyssey::raster::{LineRule, NoTexels, RasterRules};




    const WIDTH: usize = 32;

    /// The value a word stands for: (1 + m / 2^18) * 2^(e - 31).
    fn word_value(word: u32) -> f64 {
        let e = (word >> 18) & 0x3f;
        let m = word & 0x3ffff;

        (1.0 + m as f64 / 262144.0) * 2f64.powi(e as i32 - 31)
    }

    #[test]
    fn the_encoder_truncates_to_18_mantissa_bits() {
        assert_eq!(encode_f64(1.0), 31 << 18);
        assert_eq!(encode_f64(1.5), (31 << 18) | (1 << 17));
        // Truncation, not rounding: just below 1 + 2^-18 is still the word for 1.
        assert_eq!(encode_f64(1.0 + 2f64.powi(-18) * 0.999), 31 << 18);
        assert_eq!(encode_f64(1.0 + 2f64.powi(-18)), (31 << 18) | 1);
        // Every word below the saturation point survives a round trip.
        for w in (0..=0xffffffu32).step_by(4099) {
            assert_eq!(encode_f64(word_value(w)), w, "word {w:#x}");
        }
    }

    #[test]
    fn a_value_below_2_to_the_minus_31_encodes_to_zero_and_the_sign_is_ignored() {
        assert_eq!(encode_f64(0.0), 0);
        assert_eq!(encode_f64(2f64.powi(-31) * 0.999), 0, "just under 2^-31");
        assert_eq!(encode_f64(2f64.powi(-31)), 0, "2^-31 is word 0 too");
        assert_eq!(encode_f64(2f64.powi(-31) * (1.0 + 2f64.powi(-18))), 1, "and its first mantissa step is word 1");
        assert_eq!(encode_f64(f64::NAN), 0);
        assert_eq!(encode_f64(-1.0), encode_f64(1.0));
        assert_eq!(encode_f64(-1.0e-12), 0);
    }

    #[test]
    fn a_value_of_2_to_the_33_or_more_saturates_at_0x0fffffff() {
        assert_eq!(encode_f64(8589934592.0), DEPTH_HOST_SATURATED);
        assert_eq!(encode_f64(1.0e20), DEPTH_HOST_SATURATED);
        assert_eq!(encode_f64(f64::INFINITY), DEPTH_HOST_SATURATED);
        assert_eq!(encode_f64(-8589934592.0), DEPTH_HOST_SATURATED);
        // The largest value below 2^33 is a real word, not the saturated one.
        assert_eq!(encode_f64(8589934591.0), 0xffffff);
        // The plane keeps the low 24 bits of a saturated word.
        assert_eq!(DEPTH_HOST_SATURATED & ODSY_DEPTH24_MAX, 0xffffff);
    }

    #[test]
    fn the_near_plane_is_word_zero_and_the_far_plane_is_2s() {
        assert_eq!(word_for_fraction(0.0), 0);
        // 2S = 2^33 - 2^15: exponent 32 (e = 63), mantissa 2^18 - 2.
        assert_eq!(word_for_fraction(1.0), 0xfffffe);
        assert!((word_value(word_for_fraction(1.0)) - DEPTH_2S).abs() < DEPTH_2S * 2f64.powi(-17));
        // Outside the frustum clamps to it.
        assert_eq!(word_for_fraction(-0.5), 0);
        assert_eq!(word_for_fraction(2.0), 0xfffffe);
        // Halfway is linear in eye distance, so the value is S, not a log scale.
        assert!((word_value(word_for_fraction(0.5)) - DEPTH_2S / 2.0).abs() < DEPTH_2S * 2f64.powi(-18));
    }

    #[test]
    fn words_order_as_integers_do_so_depth_compares_without_decoding() {
        let mut last = 0;

        for i in 0..=2000 {
            let w = word_for_fraction(i as f64 / 2000.0);

            assert!(w >= last, "fraction {i}/2000 gave {w:#x} below {last:#x}");
            last = w;
        }
        // Two depths one mantissa step apart in the top binade (relative 2^-18)
        // differ by one word; the same step near the bottom is the same word
        // step, where a fixed-point depth would see no difference.
        let hi = word_for_fraction(0.75);
        let lo = word_for_fraction(1.0e-6);
        assert!(word_for_fraction(0.75 * (1.0 + 2f64.powi(-16))) > hi);
        assert!(word_for_fraction(1.0e-6 * (1.0 + 2f64.powi(-16))) > lo);
    }

    #[test]
    fn the_16_bit_form_is_the_top_16_bits_of_the_word() {
        for w in [0u32, 31 << 18, (31 << 18) | 0x3ffff, 0xfffffe, 0x123456] {
            let w16 = word16_from_word24(w);

            assert_eq!(w16, w >> 8);
            assert_eq!(w16 >> 10, w >> 18, "exponent in bits 15:10");
            assert_eq!(w16 & 0x3ff, (w >> 8) & 0x3ff, "top ten mantissa bits in 9:0");
        }
    }

    #[test]
    fn an_offset_adds_to_the_encoded_word_and_clamps_to_the_words_range() {
        assert_eq!(word_with_offset(1000, 24), 1024);
        assert_eq!(word_with_offset(1000, -1000), 0);
        assert_eq!(word_with_offset(1000, -(1 << 24)), 0);
        assert_eq!(word_with_offset(0xfffffe, (1 << 24) - 1), 0xffffff);
        assert_eq!(OFFSET_STEPS_MIN, -16777216);
        assert_eq!(OFFSET_STEPS_MAX, 16777215);
    }

    // The offset the guest library computes for glPolygonOffset(factor, units)
    // under a frustum: factor * tan(L) * (eye
    // width of a pixel at f) * 2^19 / (f - n) + 2 * units, in steps of the
    // word. It is the guest's arithmetic, not the board's; the register
    // receives the result.
    fn library_offset_steps(factor: f64, units: f64, limit_deg: f64, pixel_at_far: f64, n: f64, f: f64) -> i64 {
        let v = factor * limit_deg.to_radians().tan() * pixel_at_far * 524288.0 / (f - n) + 2.0 * units;

        (v as i64).clamp(OFFSET_STEPS_MIN, OFFSET_STEPS_MAX)
    }

    #[test]
    fn the_library_offset_for_the_default_limit_angle_counts_word_steps_of_the_far_plane() {
        // tan(70 deg) = 2.747: factor 1, a pixel the width of (f - n) / 2^19
        // is 2.747 steps.
        let steps = library_offset_steps(1.0, 0.0, 70.0, 1.0 / 524288.0, 0.0, 1.0);

        assert_eq!(steps, 2);
        assert_eq!(library_offset_steps(0.0, 3.0, 70.0, 1.0, 1.0, 100.0), 6, "units count double");
        assert_eq!(library_offset_steps(1.0e9, 0.0, 70.0, 1.0, 1.0, 2.0), OFFSET_STEPS_MAX, "limited to 24 bits");
        assert_eq!(library_offset_steps(0.0, -1.0e9, 70.0, 1.0, 1.0, 2.0), OFFSET_STEPS_MIN);
    }

    /// Draw one triangle covering the plane at window depth `zw` with the
    /// board's depth rules, depth test `func`, and `steps` of polygon offset
    /// through the register's path (the shared context's bias).
    fn draw_flat(depth: &mut [u32], zw: f64, func: u32, steps: i64) {
        draw_sloped(depth, zw, 0.0, func, steps, (0.0, 0.0));
    }

    /// [`draw_flat`] with the third vertex `tilt` further in window depth, so
    /// the triangle has a depth slope, and the context's glPolygonOffset
    /// factor and units set to `(factor, units)`.
    fn draw_sloped(depth: &mut [u32], zw: f64, tilt: f64, func: u32, steps: i64, (factor, units): (f32, f32)) {
        let rules = RasterRules {
            subpixel_bits: 4,
            color_bits: 8,
            depth_bits: 24,
            line_rule: LineRule::Diamond,
            depth_float: true,
            ..Default::default()
        };
        assert!(rules.rules_supported());
        let mut id = [0.0f32; 16];
        for k in 0..4 {
            id[k * 5] = 1.0;
        }
        let mut st = GlDrawState {
            modelview: id,
            projection: id,
            viewport: [0, 0, WIDTH as i32, WIDTH as i32],
            shade_model: GL_FLAT,
            depth_test: true,
            depth_func: func,
            depth_mask: true,
            depth_range: [0.0, 1.0],
            draw_buffers: 1,
            colour_mask: GL_COLOUR_MASK_ALL,
            polygon_offset: GL_OFFSET_FILL,
            // What the 0x058 write puts in the shared context.
            offset_bias: steps as f32 * polygon::OFFSET_SCALE,
            offset_factor: factor,
            offset_units: units,
            ..Default::default()
        };
        st.drawable.bound = true;
        st.drawable.front = 1;
        st.drawable.clip.n = 1;
        st.drawable.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: WIDTH as i32 - 1, y1: WIDTH as i32 - 1 };
        let mut verts = vec![0.0f32; 3 * vtx::FLOATS];
        let ndc_z = (zw * 2.0 - 1.0) as f32;

        for (i, (x, y)) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0)].iter().enumerate() {
            let v = &mut verts[i * vtx::FLOATS..(i + 1) * vtx::FLOATS];
            let z = if i == 2 { ndc_z + (tilt * 2.0) as f32 } else { ndc_z };
            v[vtx::POSITION..vtx::POSITION + 4].copy_from_slice(&[*x, *y, z, 1.0]);
            v[vtx::COLOR..vtx::COLOR + 4].copy_from_slice(&[1.0; 4]);
        }
        let stipple = [0u32; 32];
        let mut d = GlDraw::new(GL_TRIANGLES, 3, &verts, &st, &stipple);
        let mut b = raster_bench::Bench::empty();

        b.rules = rules;
        b.set_depth(WIDTH as i32, depth);
        let t = b.board.gl_target();
        b.draw(&t, &NoTexels, &mut d);
        depth.copy_from_slice(&b.depth(WIDTH as i32));
    }

    fn plane(initial: u32) -> Vec<u32> {
        vec![initial; WIDTH * WIDTH]
    }

    #[test]
    fn a_rasterised_depth_is_stored_as_the_board_word() {
        let mut fb = plane(ODSY_DEPTH24_MAX);

        draw_flat(&mut fb, 0.25, GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.25)), "{:#x}", fb[0]);
        assert!((word_value(fb[0]) - DEPTH_2S * 0.25).abs() < DEPTH_2S * 2f64.powi(-20));
    }

    #[test]
    fn depth_compares_in_the_board_format_pass_nearer_and_fail_farther() {
        let mut fb = plane(word_for_fraction(0.5));

        draw_flat(&mut fb, 0.75, GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5)), "a farther polygon fails GL_LESS");
        draw_flat(&mut fb, 0.5, GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5)), "an equal one fails GL_LESS");
        draw_flat(&mut fb, 0.5, GL_LEQUAL, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5)));
        draw_flat(&mut fb, 0.25, GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.25)), "a nearer one passes");
        // One mantissa step nearer in the top binade still passes: the
        // format resolves 2^-18 relative at every depth.
        let near = word_for_fraction(0.25) - 1;
        let mut fb = plane(word_for_fraction(0.25));
        draw_flat(&mut fb, word_value(near) / DEPTH_2S, GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == near), "{:#x} against {near:#x}", fb[0]);
    }

    #[test]
    fn a_polygon_offset_adds_the_same_steps_to_the_word_at_any_depth() {
        // INFERRED placement: the field adds to the encoded word. Added to
        // window depth instead, 100 steps would be a different count of words
        // at 0.25 and at 0.75, because a word step is relative.
        for zw in [0.25, 0.75] {
            let mut fb = plane(ODSY_DEPTH24_MAX);

            draw_flat(&mut fb, zw, GL_LESS, 100);
            assert!(fb.iter().all(|&w| w == word_for_fraction(zw) + 100), "zw {zw}: {:#x}", fb[0]);
        }
        let mut fb = plane(ODSY_DEPTH24_MAX);
        draw_flat(&mut fb, 0.25, GL_LESS, -100);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.25) - 100), "a negative offset pulls nearer");
    }

    #[test]
    fn the_offset_field_is_the_whole_offset_whatever_the_slope_factor_and_units() {
        // The library folds factor and 2 units into the one field, and the board
        // has no slope term, so a context carrying a factor and units, on a
        // triangle with a depth slope, offsets exactly as the field says.
        let mut plain = plane(ODSY_DEPTH24_MAX);
        let mut loaded = plane(ODSY_DEPTH24_MAX);

        draw_sloped(&mut plain, 0.25, 0.5, GL_LESS, 40, (0.0, 0.0));
        draw_sloped(&mut loaded, 0.25, 0.5, GL_LESS, 40, (1.5, 2000.0));
        assert_ne!(plain[0], ODSY_DEPTH24_MAX, "the triangle drew");
        assert!(plain.iter().zip(&loaded).all(|(a, b)| a == b), "factor and units changed the stored depth");
        // Same word delta from the unoffset draw at two depths.
        for zw in [0.125, 0.5] {
            let mut off = plane(ODSY_DEPTH24_MAX);
            let mut none = plane(ODSY_DEPTH24_MAX);

            draw_sloped(&mut off, zw, 0.25, GL_LESS, 40, (1.5, 2000.0));
            draw_sloped(&mut none, zw, 0.25, GL_LESS, 0, (0.0, 0.0));
            assert!(off.iter().zip(&none).all(|(a, b)| *b == ODSY_DEPTH24_MAX || *a == *b + 40), "zw {zw}");
        }
    }

    #[test]
    fn a_decal_offset_towards_the_viewer_wins_the_depth_test_over_coplanar_geometry() {
        let mut fb = plane(ODSY_DEPTH24_MAX);

        draw_flat(&mut fb, 0.5, GL_LESS, 0);
        draw_flat(&mut fb, 0.5, GL_LESS, -4);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5) - 4));
        // And a positive one loses.
        draw_flat(&mut fb, 0.5, GL_LESS, 4);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5) - 4));
    }

    #[test]
    fn an_offset_that_would_pass_the_ends_of_the_word_range_clamps() {
        let mut fb = plane(ODSY_DEPTH24_MAX);

        draw_flat(&mut fb, 0.0, GL_LESS, -(1 << 24));
        assert!(fb.iter().all(|&w| w == 0));
        let mut fb = plane(ODSY_DEPTH24_MAX);
        draw_flat(&mut fb, 1.0, GL_LEQUAL, (1 << 24) - 1);
        assert!(fb.iter().all(|&w| w == 0xffffff));
    }
}
