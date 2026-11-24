use crate::dev::ng1::rex3::{
    Rex3Context, DRAWMODE0_ADRMODE_A_LINE, DRAWMODE1_DRAWDEPTH_12, DRAWMODE1_DRAWDEPTH_8,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ShadeDda {
    red: u32,
    alpha: u32,
    green: u32,
    blue: u32,
    slope_red: i32,
    slope_alpha: i32,
    slope_green: i32,
    slope_blue: i32,
    rgb_mode: bool,
    ci_clamp: bool,
    draw_depth: u32,
    antialiased: bool,
}

impl ShadeDda {
    pub(crate) fn from_context(context: &Rex3Context) -> Self {
        Self {
            red: context.colorred,
            alpha: context.coloralpha,
            green: context.colorgrn,
            blue: context.colorblue,
            slope_red: context.slopered,
            slope_alpha: context.slopealpha,
            slope_green: context.slopegrn,
            slope_blue: context.slopeblue,
            rgb_mode: context.drawmode1.rgbmode(),
            ci_clamp: context.drawmode0.ciclamp(),
            draw_depth: context.drawmode1.drawdepth(),
            antialiased: context.drawmode0.adrmode() == DRAWMODE0_ADRMODE_A_LINE,
        }
    }

    pub(crate) fn source(self) -> u32 {
        let color = if self.rgb_mode {
            (clamp_component(self.blue) << 16)
                | (clamp_component(self.green) << 8)
                | clamp_component(self.red)
        } else {
            self.red >> 11
        };
        let alpha = if self.antialiased {
            0xff
        } else {
            clamp_component(self.alpha)
        };
        (color & 0x00ff_ffff) | (alpha << 24)
    }

    pub(crate) fn components(self) -> [u32; 4] {
        [self.red, self.alpha, self.green, self.blue]
    }

    pub(crate) fn advance(&mut self) {
        self.red = self.red.wrapping_add(self.slope_red as u32);
        self.alpha = self.alpha.wrapping_add(self.slope_alpha as u32);
        self.green = self.green.wrapping_add(self.slope_green as u32);
        self.blue = self.blue.wrapping_add(self.slope_blue as u32);
        if self.rgb_mode {
            self.red = clamp_accumulator(self.red);
            self.alpha = clamp_accumulator(self.alpha);
            self.green = clamp_accumulator(self.green);
            self.blue = clamp_accumulator(self.blue);
        } else if self.ci_clamp {
            match self.draw_depth {
                DRAWMODE1_DRAWDEPTH_8 if self.red & (1 << 19) != 0 => {
                    self.red = 0x0007_ffff;
                }
                DRAWMODE1_DRAWDEPTH_12 if self.red & (1 << 21) != 0 => {
                    self.red = 0x001f_ffff;
                }
                _ => {}
            }
        }
    }

    /// Whether `advance_by` has a closed form for this DDA, so the GPU can
    /// shade each pixel from the start value and its visit index instead of
    /// from per-32-pixel checkpoints.
    ///
    /// RGB: each component has bit 31 clear, its low 20 bits in
    /// [0, 0x7FFFF], and moves by less than 0x40000 a step. The clamp only
    /// reads bit 31 and the integer bits 11-19, so bits 20-30 (IRIX leaves
    /// junk there) ride along unchanged while the low 20 bits stay in range.
    /// The first step out lands either below 0 (bit 31 set, or a borrow from
    /// bits 20-30 that leaves integer >= 0x180: clamps to 0) or below 0xC0000
    /// (integer 0x100-0x17F: clamps to 0x7FFFF), and every later step lands
    /// there again. So after n steps, with `low = start & 0xFFFFF`, the value
    /// is `start - low + low + n * slope` while that stays in range, else 0
    /// or 0x7FFFF, bit for bit. Colour index without a CI clamp only wraps.
    pub(crate) fn is_linear(&self) -> bool {
        if self.rgb_mode {
            [
                (self.red, self.slope_red),
                (self.alpha, self.slope_alpha),
                (self.green, self.slope_green),
                (self.blue, self.slope_blue),
            ]
            .iter()
            .all(|&(value, slope)| {
                value & (1 << 31) == 0
                    && value & LOW_BITS <= ACCUMULATOR_MAX
                    && slope.unsigned_abs() < LINEAR_SLOPE_LIMIT
            })
        } else {
            !self.ci_clamp
                || !matches!(self.draw_depth, DRAWMODE1_DRAWDEPTH_8 | DRAWMODE1_DRAWDEPTH_12)
        }
    }

    /// `steps` calls of `advance`, without stepping when `is_linear`.
    pub(crate) fn advance_by(&mut self, steps: u64) {
        if self.is_linear() {
            if self.rgb_mode {
                self.red = linear_clamped(self.red, self.slope_red, steps);
                self.alpha = linear_clamped(self.alpha, self.slope_alpha, steps);
                self.green = linear_clamped(self.green, self.slope_green, steps);
                self.blue = linear_clamped(self.blue, self.slope_blue, steps);
            } else {
                // Wrapping arithmetic only needs the step count mod 2^32.
                let steps = steps as u32;
                self.red = self.red.wrapping_add((self.slope_red as u32).wrapping_mul(steps));
                self.alpha =
                    self.alpha.wrapping_add((self.slope_alpha as u32).wrapping_mul(steps));
                self.green =
                    self.green.wrapping_add((self.slope_green as u32).wrapping_mul(steps));
                self.blue = self.blue.wrapping_add((self.slope_blue as u32).wrapping_mul(steps));
            }
            return;
        }
        for _ in 0..steps {
            let before = *self;
            self.advance();
            // A clamped component that maps to itself stays there.
            if *self == before {
                break;
            }
        }
    }

    pub(crate) fn write_back(self, context: &mut Rex3Context) {
        context.colorred = self.red;
        context.coloralpha = self.alpha;
        context.colorgrn = self.green;
        context.colorblue = self.blue;
    }
}

fn clamp_component(value: u32) -> u32 {
    let integer = (value >> 11) & 0x1ff;
    if value & (1 << 31) != 0 || integer >= 0x180 {
        0
    } else {
        integer.min(0xff)
    }
}

const ACCUMULATOR_MAX: u32 = 0x0007_ffff;
const LINEAR_SLOPE_LIMIT: u32 = 0x0004_0000;
const LOW_BITS: u32 = 0x000f_ffff;

fn linear_clamped(start: u32, slope: i32, steps: u64) -> u32 {
    // Past 2^20 steps any non-zero slope has left the range.
    let steps = steps.min(1 << 20) as i64;
    let value = i64::from(start & LOW_BITS) + i64::from(slope) * steps;
    if value < 0 {
        0
    } else if value > i64::from(ACCUMULATOR_MAX) {
        ACCUMULATOR_MAX
    } else {
        (start & !LOW_BITS) | value as u32
    }
}

fn clamp_accumulator(value: u32) -> u32 {
    let integer = (value >> 11) & 0x1ff;
    if value & (1 << 31) != 0 || integer >= 0x180 {
        0
    } else if integer > 0xff {
        0x0007_ffff
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{DrawMode0, DrawMode1};
    use crate::dev::ng1::rex3_generic::iterate_shade;
    use crate::dev::ng1::rex3_shape;

    fn on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("resident-shade-test".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .expect("spawn shade test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    #[test]
    fn advance_by_matches_stepping() {
        let mut random = 0x9e37_79b9u32;
        let mut next = move || {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            random
        };
        // Values and slopes weighted towards the edges of the closed form.
        let edges_value = [0, 1, 0x7f800, 0x7fffe, 0x7ffff, 0x80000, 0xbffff, 0xc0000,
            0x0010_0000, 0x8000_0000, 0xffff_ffff, 0x0090_0000, 0x0090_3d33, 0x7ff7_ffff,
            0x7ff8_0000, 0x0010_0001];
        let edges_slope = [0, 1, -1, 0x800, -0x800, 0x3ffff, -0x3ffff, 0x40000, -0x40000,
            0x7ffff, -0x80000, i32::MAX, i32::MIN];
        let pick_value = |next: &mut dyn FnMut() -> u32| match next() % 4 {
            0 => edges_value[(next() as usize) % edges_value.len()],
            1 => next() & 0x7ffff,
            2 => next() & 0x7ff7_ffff,
            _ => next(),
        };
        let pick_slope = |next: &mut dyn FnMut() -> u32| match next() % 4 {
            0 => edges_slope[(next() as usize) % edges_slope.len()],
            1 => (next() as i32) >> 20,
            2 => (next() as i32) >> 14,
            _ => next() as i32,
        };
        let mut linear = 0;
        for case in 0..30_000u32 {
            let mut dda = ShadeDda {
                red: pick_value(&mut next),
                alpha: pick_value(&mut next),
                green: pick_value(&mut next),
                blue: pick_value(&mut next),
                slope_red: pick_slope(&mut next),
                slope_alpha: pick_slope(&mut next),
                slope_green: pick_slope(&mut next),
                slope_blue: pick_slope(&mut next),
                rgb_mode: case % 4 != 0,
                ci_clamp: next() & 1 != 0,
                draw_depth: next() & 3,
                antialiased: false,
            };
            if case % 2 == 0 && dda.rgb_mode {
                // Mostly in range, so the closed form is what gets exercised.
                for (value, slope) in [
                    (&mut dda.red, &mut dda.slope_red),
                    (&mut dda.alpha, &mut dda.slope_alpha),
                    (&mut dda.green, &mut dda.slope_green),
                    (&mut dda.blue, &mut dda.slope_blue),
                ] {
                    *value &= if *value & 1 != 0 { 0x7ff7_ffff } else { 0x7ffff };
                    *slope %= 0x40000;
                }
            }
            linear += u32::from(dda.is_linear());
            let steps = match next() % 64 {
                _ if case < 16 => 1_100_000 + u64::from(next() % 4096),
                0 => u64::from(next() % 70_000),
                1..=20 => u64::from(next() % 8),
                21..=40 => u64::from(next() % 64),
                _ => u64::from(next() % 2048),
            };
            let mut stepped = dda;
            let mut checked = dda;
            for step in 0..steps {
                stepped.advance();
                if step % 509 == 0 {
                    let mut jumped = dda;
                    jumped.advance_by(step + 1);
                    assert_eq!(jumped, stepped, "case {case} {dda:?} after {} steps", step + 1);
                }
            }
            checked.advance_by(steps);
            assert_eq!(checked, stepped, "case {case} {dda:?} after {steps} steps");
        }
        assert!(linear > 10_000, "only {linear} linear cases");
    }

    #[test]
    fn payload_dda_matches_scalar_shade_iteration() {
        on_large_stack(|| {
            let mut random = 0x243f_6a88u32;
            let mut next = || {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                random
            };
            for rgb_mode in [false, true] {
                for ci_clamp in [false, true] {
                    for draw_depth in 0..=3 {
                        for antialiased in [false, true] {
                            let mut context = Rex3Context::power_on_default();
                            context.drawmode0 = DrawMode0(
                                (1 << 18)
                                    | u32::from(ci_clamp) << 21
                                    | if antialiased {
                                        DRAWMODE0_ADRMODE_A_LINE << 2
                                    } else {
                                        0
                                    },
                            );
                            context.drawmode1 =
                                DrawMode1((draw_depth << 3) | u32::from(rgb_mode) << 15);
                            context.colorred = next();
                            context.coloralpha = next();
                            context.colorgrn = next();
                            context.colorblue = next();
                            context.slopered = next() as i32;
                            context.slopealpha = next() as i32;
                            context.slopegrn = next() as i32;
                            context.slopeblue = next() as i32;
                            let mode = rex3_shape::unpack(
                                context.drawmode0.0,
                                context.drawmode1.0,
                                context.clipmode,
                            );
                            let mut dda = ShadeDda::from_context(&context);
                            for _ in 0..257 {
                                let expected_color = (context.get_colori() & 0x00ff_ffff)
                                    | (if antialiased {
                                        0xff
                                    } else {
                                        Rex3Context::clamp_color_component(context.coloralpha)
                                    } << 24);
                                assert_eq!(dda.source(), expected_color);
                                dda.advance();
                                iterate_shade(&mut context, &mode);
                                let mut actual = context;
                                dda.write_back(&mut actual);
                                assert_eq!(
                                    (
                                        actual.colorred,
                                        actual.coloralpha,
                                        actual.colorgrn,
                                        actual.colorblue,
                                    ),
                                    (
                                        context.colorred,
                                        context.coloralpha,
                                        context.colorgrn,
                                        context.colorblue,
                                    ),
                                );
                            }
                        }
                    }
                }
            }
        });
    }
}
