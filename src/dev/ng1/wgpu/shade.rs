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
