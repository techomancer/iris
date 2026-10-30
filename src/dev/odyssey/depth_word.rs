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
//! What the datasheet's depth section (odyssey.md, "Depth, polygon offset and
//! lighting follow-up") establishes is the library's reading of the plane
//! after a DMA read: the host's encoder and decoder below, the 16-bit form
//! and the polygon-offset field. That the plane holds exactly these bits is
//! INFERRED there. The rounding of a rasterised depth into the word is
//! unknown; this model truncates, as the host encoder does.

use super::consts::ODSY_DEPTH24_MAX;

/// The depth range's far value, 2S = 2^33 - 2^15.
pub const DEPTH_2S: f64 = 8589901824.0;

/// What the host encoder returns for an exponent of 160 or more. Wider than
/// the word: the plane keeps the low 24 bits.
pub const DEPTH_HOST_SATURATED: u32 = 0x0fffffff;

/// A value below 2^-31 has no exponent; the host encoder maps it to 0.
const DEPTH_MIN_VALUE: f64 = 1.0 / 2147483648.0;
const DEPTH_EXP_BIAS: i32 = 31;
const DEPTH_MANTISSA_BITS: u32 = 18;
const DEPTH_MANTISSA_MASK: u32 = (1 << DEPTH_MANTISSA_BITS) - 1;
/// The difference between an IEEE single's exponent and the word's.
const IEEE_EXP_OFFSET: u32 = 96;

/// The host encoder (ESTABLISHED, datasheet depth word): the IEEE single's
/// exponent minus 96 and its top 18 mantissa bits, the rest truncated. An
/// exponent below 96 gives 0, one of 160 or more gives 0x0fffffff, and the
/// sign is ignored.
pub fn encode_host(v: f32) -> u32 {
    encode_f64(v as f64)
}

/// [`encode_host`] for a double, truncating its mantissa to 18 bits. A
/// double's range holds a single's, so the result agrees for every single.
pub fn encode_f64(v: f64) -> u32 {
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

/// The host decoder (ESTABLISHED): the word's exponent plus 96, its mantissa
/// at the top of the single's, and the low five mantissa bits refilled from
/// the top five. Only the low 24 bits are read. Word 0 decodes to 2^-31
/// by this rule, though the encoder maps everything below that to 0; whether
/// the library special-cases it is not shown.
pub fn decode_host(word: u32) -> f32 {
    let e = (word >> DEPTH_MANTISSA_BITS) & 0x3f;
    let m = word & DEPTH_MANTISSA_MASK;

    f32::from_bits(((e + IEEE_EXP_OFFSET) << 23) | (m << 5) | (m >> 13))
}

/// The word for an eye-linear depth fraction z in [0, 1] (0 at the near
/// plane, 1 at the far plane): the encoding of 2S * z. A fraction outside
/// [0, 1] clamps to it.
pub fn word_for_fraction(z: f64) -> u32 {
    let z = if z > 1.0 { 1.0 } else if z > 0.0 { z } else { 0.0 };

    encode_f64(z * DEPTH_2S) & ODSY_DEPTH24_MAX
}

/// The 16-bit form (ESTABLISHED): the exponent in bits 15:10 and the top ten
/// mantissa bits in 9:0, which is the 24-bit word shifted right by eight. The
/// library reads it from windows whose visual is RGBA 12/12/12/12 and the
/// 24-bit form from every other visual with depth. Why those visuals get the
/// shorter word is not shown, and the model has no depth plane of that width
/// or depth DMA read to attach it to, so only the conversion exists.
pub fn word16_from_word24(word: u32) -> u32 {
    (word & ODSY_DEPTH24_MAX) >> 8
}

/// The 16-bit form's decode: the same refill as [`decode_host`], from the
/// ten mantissa bits (their top five fill the low five of the 23).
pub fn decode_host16(word: u32) -> f32 {
    let e = (word >> 10) & 0x3f;
    let m = word & 0x3ff;

    f32::from_bits(((e + IEEE_EXP_OFFSET) << 23) | (m << 13) | (m >> 5))
}

/// The polygon-offset field (block 0x058) is a signed 25-bit count of steps
/// of the encoded 24-bit word, the limits the library clamps to.
pub const OFFSET_STEPS_MIN: i64 = -(1 << 24);
pub const OFFSET_STEPS_MAX: i64 = (1 << 24) - 1;

/// The word after a polygon offset. INFERRED placement (datasheet, "Polygon
/// offset"): the board adds the field to the encoded word, so the offset
/// counts mantissa steps and shrinks toward the near plane in proportion to
/// d - n. APPROXIMATION: the sum clamps to 0 and the largest word; what the
/// board does at either end is not shown.
pub fn word_with_offset(word: u32, steps: i64) -> u32 {
    (word as i64 + steps).clamp(0, ODSY_DEPTH24_MAX as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::odyssey::*;

    const WIDTH: usize = 32;

    #[test]
    fn a_depth_word_decodes_to_one_plus_mantissa_over_2_18_times_a_power_of_two() {
        // e = 31 is exponent 0: 1.0 at m = 0, and half a unit up at the top
        // bit. The refill of the low five mantissa bits from the top five
        // (the library's rule) adds 16 of the single's 2^-23 steps there.
        assert_eq!(decode_host(31 << 18), 1.0);
        assert_eq!(decode_host((31 << 18) | (1 << 17)), 1.5 + 16.0 * 2f32.powi(-23));
        // The extremes: 2^-31 and just under 2^33.
        assert_eq!(decode_host(0), 1.0 / 2147483648.0);
        assert_eq!(decode_host(0xffffff), f32::from_bits(0x4fffffff));
    }

    #[test]
    fn the_host_encoder_truncates_to_18_mantissa_bits_and_decodes_back_within_a_step() {
        for v in [1.0f32, 1.5, 3.0e-5, 0.7, 123_456.79, 5.0e9] {
            let w = encode_host(v);
            let back = decode_host(w);

            // The refill puts the decode inside the word's step, above or below v.
            assert!((v - back).abs() < v * 2f32.powi(-17), "{v} decodes to {back}, further than a step");
            // A word decodes to a value that encodes to the same word.
            assert_eq!(encode_host(back), w, "{v}: the word does not survive a round trip");
        }
        // Truncation, not rounding: just below 1 + 2^-18 is still the word for 1.
        assert_eq!(encode_host(f32::from_bits(0x3f800000 | 0x1f)), 31 << 18);
        assert_eq!(encode_host(f32::from_bits(0x3f800000 | 0x20)), (31 << 18) | 1);
        // Every word below the saturation point round trips exactly.
        for w in (0..=0xffffffu32).step_by(4099) {
            assert_eq!(encode_host(decode_host(w)), w, "word {w:#x}");
        }
    }

    #[test]
    fn a_value_below_2_to_the_minus_31_encodes_to_zero_and_the_sign_is_ignored() {
        assert_eq!(encode_host(0.0), 0);
        assert_eq!(encode_host(f32::from_bits(0x2fffffff)), 0, "just under 2^-31");
        assert_eq!(encode_host(f32::from_bits(0x30000000)), 0, "2^-31 is exponent 96, word 0 too");
        assert_eq!(encode_host(f32::from_bits(0x30000020)), 1, "and its first mantissa step is word 1");
        assert_eq!(encode_host(f32::NAN), 0);
        assert_eq!(encode_host(-1.0), encode_host(1.0));
        assert_eq!(encode_host(-1.0e-12), 0);
    }

    #[test]
    fn a_value_of_2_to_the_33_or_more_saturates_at_0x0fffffff() {
        assert_eq!(encode_host(8589934592.0), DEPTH_HOST_SATURATED);
        assert_eq!(encode_host(1.0e20), DEPTH_HOST_SATURATED);
        assert_eq!(encode_host(f32::INFINITY), DEPTH_HOST_SATURATED);
        assert_eq!(encode_host(-8589934592.0), DEPTH_HOST_SATURATED);
        // The largest single below 2^33 is a real word, not the saturated one.
        assert_eq!(encode_host(f32::from_bits(0x4fffffff)), 0xffffff);
        // The plane keeps the low 24 bits of a saturated word.
        assert_eq!(DEPTH_HOST_SATURATED & ODSY_DEPTH24_MAX, 0xffffff);
    }

    #[test]
    fn the_near_plane_is_word_zero_and_the_far_plane_is_2s() {
        assert_eq!(word_for_fraction(0.0), 0);
        // 2S = 2^33 - 2^15: exponent 32 (e = 63), mantissa 2^18 - 2.
        assert_eq!(word_for_fraction(1.0), 0xfffffe);
        assert!((decode_host(word_for_fraction(1.0)) as f64 - DEPTH_2S).abs() < DEPTH_2S * 2f64.powi(-17));
        // Outside the frustum clamps to it.
        assert_eq!(word_for_fraction(-0.5), 0);
        assert_eq!(word_for_fraction(2.0), 0xfffffe);
        // Halfway is linear in eye distance, so the value is S, not a log scale.
        assert!((decode_host(word_for_fraction(0.5)) as f64 - DEPTH_2S / 2.0).abs() < DEPTH_2S * 2f64.powi(-18));
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
        let step = |w: u32| decode_host(w + 1) as f64 / decode_host(w) as f64 - 1.0;

        assert!(step(hi) < 2f64.powi(-17) && step(lo) < 2f64.powi(-17));
        assert!(word_for_fraction(1.0e-6 * (1.0 + 2f64.powi(-16))) > lo);
    }

    #[test]
    fn the_16_bit_form_is_the_top_16_bits_of_the_word_and_decodes_within_a_coarser_step() {
        for w in [0u32, 31 << 18, (31 << 18) | 0x3ffff, 0xfffffe, 0x123456] {
            let w16 = word16_from_word24(w);

            assert_eq!(w16, w >> 8);
            assert_eq!(w16 >> 10, w >> 18, "exponent in bits 15:10");
            assert_eq!(w16 & 0x3ff, (w >> 8) & 0x3ff, "top ten mantissa bits in 9:0");
            let long = decode_host(w);
            let short = decode_host16(w16);

            assert!((long - short).abs() < long * 2f32.powi(-9), "{w:#x}: {long} against {short}");
        }
        assert_eq!(decode_host16(31 << 10), 1.0);
        assert_eq!(decode_host16((31 << 10) | 0x200), 1.5 + 16.0 * 2f32.powi(-23));
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
    // under a frustum (datasheet, "Polygon offset"): factor * tan(L) * (eye
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

    /// A depth plane of WIDTH by WIDTH words; colour goes nowhere.
    struct Fb<'a> {
        depth: RowsMut<'a, u32>,
    }

    struct FbView<'a> {
        depth: Rows<'a, u32>,
    }

    impl PlaneStore for Fb<'_> {
        fn planes(&mut self) -> Box<dyn Planes + '_> {
            Box::new(FbView { depth: self.depth.view() })
        }
        fn split(&mut self, bands: u32) -> Vec<Box<dyn PlaneStore + '_>> {
            self.depth.split(bands).into_iter().map(|depth| Box::new(Fb { depth }) as Box<dyn PlaneStore + '_>).collect()
        }
    }

    impl Planes for FbView<'_> {
        fn store_colour(&self, _t: ColourTarget, _x: i32, _y: i32, _c: u32) {}
        fn load_colour(&self, _t: ColourTarget, _x: i32, _y: i32) -> u32 {
            0
        }
        fn store_depth(&self, x: i32, y: i32, z: u32) {
            self.depth.get(y as usize, x as usize, 1).unwrap()[0].set(z);
        }
        fn load_depth(&self, x: i32, y: i32) -> u32 {
            self.depth.get(y as usize, x as usize, 1).unwrap()[0].get()
        }
    }

    /// Draw one triangle covering the plane at window depth `zw` with the
    /// board's depth rules, depth test `func`, and `steps` of polygon offset
    /// through the register's path (the shared context's bias).
    fn draw_flat(depth: &mut [u32], zw: f64, func: u32, steps: i64) {
        let rules = RasterRules {
            subpixel_bits: 4,
            color_bits: 8,
            depth_bits: 24,
            rounding: SGI_ROUND_HALF_AWAY as u8,
            tie_rule: SGI_TIE_TOP_LEFT_BOTTOM_ORIGIN as u8,
            arith: SGI_ARITH_DOUBLE as u8,
            line_rule: SGI_LINE_COVERAGE_DOUBLE as u8,
            depth_float: true,
            ..Default::default()
        };
        assert!(rules.rules_supported());
        let target = RasterTarget {
            store_colour: true,
            load_colour: true,
            ..Default::default()
        };
        let mut id = [0.0f32; 16];
        for k in 0..4 {
            id[k * 5] = 1.0;
        }
        let mut st = GlDrawState {
            modelview: id,
            projection: id,
            viewport: [0, 0, WIDTH as i32, WIDTH as i32],
            shade_model: SGI_GL_FLAT,
            depth_test: 1,
            depth_func: func,
            depth_mask: 1,
            depth_range: [0.0, 1.0],
            draw_buffers: 1,
            colour_mask: SGI_GL_COLOUR_MASK_ALL,
            polygon_offset: SGI_GL_OFFSET_FILL,
            // What the 0x058 write puts in the shared context.
            offset_bias: steps as f32 * ODSY_POLYGON_OFFSET_SCALE,
            ..Default::default()
        };
        st.drawable.bound = 1;
        st.drawable.front = 1;
        st.drawable.clip.n = 1;
        st.drawable.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: WIDTH as i32 - 1, y1: WIDTH as i32 - 1 };
        let mut verts = vec![0.0f32; 3 * SGI_GL_VERTEX_FLOATS];
        let ndc_z = (zw * 2.0 - 1.0) as f32;

        for (i, (x, y)) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0)].iter().enumerate() {
            let v = &mut verts[i * SGI_GL_VERTEX_FLOATS..(i + 1) * SGI_GL_VERTEX_FLOATS];
            v[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4].copy_from_slice(&[*x, *y, ndc_z, 1.0]);
            v[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4].copy_from_slice(&[1.0; 4]);
        }
        let stipple = [0u32; 32];
        let mut d = GlDraw::new(SGI_GL_TRIANGLES, 3, &verts, &st, &stipple);
        let mut fb = Fb { depth: RowsMut::new(depth, WIDTH) };

        rules.draw(&target, &mut fb, &NoTexels, &mut d);
    }

    fn plane(initial: u32) -> Vec<u32> {
        vec![initial; WIDTH * WIDTH]
    }

    #[test]
    fn a_rasterised_depth_is_stored_as_the_board_word() {
        let mut fb = plane(ODSY_DEPTH24_MAX);

        draw_flat(&mut fb, 0.25, SGI_GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.25)), "{:#x}", fb[0]);
        assert!((decode_host(fb[0]) as f64 - DEPTH_2S * 0.25).abs() < DEPTH_2S * 2f64.powi(-20));
    }

    #[test]
    fn depth_compares_in_the_board_format_pass_nearer_and_fail_farther() {
        let mut fb = plane(word_for_fraction(0.5));

        draw_flat(&mut fb, 0.75, SGI_GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5)), "a farther polygon fails GL_LESS");
        draw_flat(&mut fb, 0.5, SGI_GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5)), "an equal one fails GL_LESS");
        draw_flat(&mut fb, 0.5, SGI_GL_LEQUAL, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5)));
        draw_flat(&mut fb, 0.25, SGI_GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.25)), "a nearer one passes");
        // One mantissa step nearer in the top binade still passes: the
        // format resolves 2^-18 relative at every depth.
        let near = word_for_fraction(0.25) - 1;
        let mut fb = plane(word_for_fraction(0.25));
        draw_flat(&mut fb, decode_host(near) as f64 / DEPTH_2S, SGI_GL_LESS, 0);
        assert!(fb.iter().all(|&w| w == near), "{:#x} against {near:#x}", fb[0]);
    }

    #[test]
    fn a_polygon_offset_adds_the_same_steps_to_the_word_at_any_depth() {
        // INFERRED placement: the field adds to the encoded word. Added to
        // window depth instead, 100 steps would be a different count of words
        // at 0.25 and at 0.75, because a word step is relative.
        for zw in [0.25, 0.75] {
            let mut fb = plane(ODSY_DEPTH24_MAX);

            draw_flat(&mut fb, zw, SGI_GL_LESS, 100);
            assert!(fb.iter().all(|&w| w == word_for_fraction(zw) + 100), "zw {zw}: {:#x}", fb[0]);
        }
        let mut fb = plane(ODSY_DEPTH24_MAX);
        draw_flat(&mut fb, 0.25, SGI_GL_LESS, -100);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.25) - 100), "a negative offset pulls nearer");
    }

    #[test]
    fn a_decal_offset_towards_the_viewer_wins_the_depth_test_over_coplanar_geometry() {
        let mut fb = plane(ODSY_DEPTH24_MAX);

        draw_flat(&mut fb, 0.5, SGI_GL_LESS, 0);
        draw_flat(&mut fb, 0.5, SGI_GL_LESS, -4);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5) - 4));
        // And a positive one loses.
        draw_flat(&mut fb, 0.5, SGI_GL_LESS, 4);
        assert!(fb.iter().all(|&w| w == word_for_fraction(0.5) - 4));
    }

    #[test]
    fn an_offset_that_would_pass_the_ends_of_the_word_range_clamps() {
        let mut fb = plane(ODSY_DEPTH24_MAX);

        draw_flat(&mut fb, 0.0, SGI_GL_LESS, -(1 << 24));
        assert!(fb.iter().all(|&w| w == 0));
        let mut fb = plane(ODSY_DEPTH24_MAX);
        draw_flat(&mut fb, 1.0, SGI_GL_LEQUAL, (1 << 24) - 1);
        assert!(fb.iter().all(|&w| w == 0xffffff));
    }
}
