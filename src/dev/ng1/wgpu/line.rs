use crate::dev::ng1::rex3::{
    bres_straight_on_zero, Rex3Context, DRAWMODE0_ADRMODE_A_LINE,
    DRAWMODE0_ADRMODE_F_LINE, DRAWMODE0_ADRMODE_I_LINE, DRAWMODE0_OPCODE_DRAW,
    DRAWMODE1_PLANES_CID, DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP,
    DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA, REX3_BRES_OCTANTS, REX3_COORD_BIAS,
    REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH,
};
use crate::dev::ng1::rex3_flat_fill::FillPlane;
use crate::dev::ng1::rex3_generic::ls_pattern_state_after;

const SKIPPED_PIXEL: u32 = u32::MAX;
const BACKGROUND_PIXEL: u32 = 1 << 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LineKind {
    Integer,
    Fractional,
    Antialiased,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LineCommand {
    pub(super) kind: LineKind,
    pub(super) plane: FillPlane,
    pub(super) min_x: i32,
    pub(super) min_y: i32,
    pub(super) max_x: i32,
    pub(super) max_y: i32,
    pub(super) start_major: i32,
    pub(super) y_major: bool,
    pub(super) major_decreasing: bool,
    pub(super) color: u32,
    pub(super) write_mask: u32,
    pub(super) payload_words: u32,
}

#[derive(Clone, Copy, Debug)]
struct LineGeometry {
    x: i32,
    y: i32,
    d: i32,
    incr1: i32,
    incr2: i32,
    incrx1: i32,
    incrx2: i32,
    incry1: i32,
    incry2: i32,
    y_major: bool,
    straight_on_zero: bool,
    pixel_count: u32,
    skip_first: bool,
    skip_last: bool,
    x_move: i32,
    y_move: i32,
    x_window: i32,
    y_window: i32,
    y_flip: bool,
    z_pattern: u32,
    line_pattern: u32,
    z_pattern_enabled: bool,
    line_pattern_enabled: bool,
    z_opaque: bool,
    line_opaque: bool,
    z_pattern_bit: u8,
    line_pattern_bit: u8,
    line_pattern_count: u32,
    line_pattern_repeat: u32,
    line_pattern_length: u32,
    advance_last_pattern: bool,
}

impl LineGeometry {
    fn pixels(self) -> LinePixels {
        LinePixels {
            geometry: self,
            x: self.x,
            y: self.y,
            d: self.d,
            index: 0,
        }
    }

    fn start_major(self) -> i32 {
        if self.y_major {
            self.physical_y(self.y)
        } else {
            self.x + self.x_move + self.x_window - REX3_COORD_BIAS
        }
    }

    fn major_decreasing(self) -> bool {
        if self.y_major {
            (-self.incry1 < 0) ^ self.y_flip
        } else {
            self.incrx1 < 0
        }
    }

    fn physical_y(self, y: i32) -> i32 {
        let absolute = y + self.y_move + self.y_window;
        if self.y_flip {
            0x23ff - absolute
        } else {
            absolute - REX3_COORD_BIAS
        }
    }

    fn pattern(self, index: u32) -> (bool, bool) {
        let mut background = false;
        if self.z_pattern_enabled {
            let bit = self.z_pattern_bit.wrapping_sub(index as u8) & 31;
            if (self.z_pattern >> bit) & 1 == 0 {
                if self.z_opaque {
                    background = true;
                } else {
                    return (false, false);
                }
            }
        }
        if self.line_pattern_enabled {
            let (_, bit) = ls_pattern_state_after(
                self.line_pattern_count,
                self.line_pattern_repeat,
                self.line_pattern_length,
                self.line_pattern_bit,
                index as u64,
            );
            if (self.line_pattern >> bit) & 1 == 0 {
                if self.line_opaque {
                    background = true;
                } else {
                    return (false, false);
                }
            }
        }
        (true, background)
    }

    fn pattern_advances(self) -> u64 {
        u64::from(self.pixel_count.saturating_sub(1))
            + u64::from(self.advance_last_pattern)
    }
}

#[derive(Clone, Copy, Debug)]
struct LinePixel {
    index: u32,
    x: i32,
    y: i32,
    d: i32,
    physical_x: i32,
    physical_y: i32,
    skip: bool,
}

struct LinePixels {
    geometry: LineGeometry,
    x: i32,
    y: i32,
    d: i32,
    index: u32,
}

impl Iterator for LinePixels {
    type Item = LinePixel;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.geometry.pixel_count {
            return None;
        }
        let last = self.index + 1 == self.geometry.pixel_count;
        let pixel = LinePixel {
            index: self.index,
            x: self.x,
            y: self.y,
            d: self.d,
            physical_x: self.x + self.geometry.x_move + self.geometry.x_window
                - REX3_COORD_BIAS,
            physical_y: self.geometry.physical_y(self.y),
            skip: (self.index == 0 && self.geometry.skip_first)
                || (last && self.geometry.skip_last),
        };
        self.index += 1;
        if !last {
            let straight = self.d < 0 || (self.d == 0 && self.geometry.straight_on_zero);
            if straight {
                self.x += self.geometry.incrx1;
                self.y -= self.geometry.incry1;
                self.d += self.geometry.incr1;
            } else {
                self.x += self.geometry.incrx2;
                self.y -= self.geometry.incry2;
                self.d += self.geometry.incr2;
            }
        }
        Some(pixel)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.geometry.pixel_count - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LinePixels {}

pub(super) struct LinePayload {
    pixels: LinePixels,
}

impl Iterator for LinePayload {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.pixels.next().map(|pixel| {
            let (pattern_draws, background) = self.pixels.geometry.pattern(pixel.index);
            if pixel.skip || !pattern_draws {
                SKIPPED_PIXEL
            } else {
                let minor = if self.pixels.geometry.y_major {
                    pixel.physical_x as u32
                } else {
                    pixel.physical_y as u32
                } & !BACKGROUND_PIXEL;
                minor | if background { BACKGROUND_PIXEL } else { 0 }
            }
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.pixels.size_hint()
    }
}

impl ExactSizeIterator for LinePayload {}

/// One already-set-up line GO. Finishing consumes it so context completion
/// cannot be applied twice.
#[must_use]
#[derive(Debug)]
pub(super) struct PreparedLine {
    line: Option<LineCommand>,
    geometry: LineGeometry,
    final_x: i32,
    final_y: i32,
    final_d: i32,
}

impl PreparedLine {
    pub(super) fn line(&self) -> Option<LineCommand> {
        self.line
    }

    pub(super) fn payload(&self) -> LinePayload {
        LinePayload {
            pixels: self.geometry.pixels(),
        }
    }

    pub(super) fn finish(self, context: &mut Rex3Context) {
        context.xstart = self.final_x << 11;
        context.ystart = self.final_y << 11;
        context.bresd = (self.final_d as u32) & 0x07ff_ffff;
        let advances = self.geometry.pattern_advances();
        if self.geometry.z_pattern_enabled {
            context.zpat_bit = self.geometry.z_pattern_bit
                .wrapping_sub(advances as u8)
                & 31;
        }
        if self.geometry.line_pattern_enabled {
            let (count, bit) = ls_pattern_state_after(
                self.geometry.line_pattern_count,
                self.geometry.line_pattern_repeat,
                self.geometry.line_pattern_length,
                self.geometry.line_pattern_bit,
                advances,
            );
            context.lsmode.set_lsrcount(count);
            context.pat_bit = bit;
        }
        context.host_len = 1;
        context.host_cursor = 0;
    }
}

pub(super) fn prepare_line(context: &Rex3Context) -> Result<PreparedLine, &'static str> {
    let dm0 = context.drawmode0;
    let dm1 = context.drawmode1;
    let kind = match dm0.adrmode() {
        DRAWMODE0_ADRMODE_I_LINE => LineKind::Integer,
        DRAWMODE0_ADRMODE_F_LINE => LineKind::Fractional,
        DRAWMODE0_ADRMODE_A_LINE => LineKind::Antialiased,
        _ => return Err("primitive shape"),
    };
    if dm0.opcode() != DRAWMODE0_OPCODE_DRAW {
        return Err("opcode");
    }
    if !dm0.stoponx() && !dm0.stopony() {
        return Err("continuation line");
    }
    if context.mid_primitive {
        return Err("continuation state");
    }
    if context.host_len > 1 {
        return Err("host transfer");
    }
    if dm0.colorhost()
        || dm0.alphahost()
        || dm0.length32()
        || dm0.shade()
        || dm0.lronly()
        || dm0.ciclamp()
        || dm0.ystride()
    {
        return Err("draw mode 0");
    }
    let plane = match dm1.planes() {
        DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => FillPlane::Rgb,
        DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID => {
            FillPlane::Aux
        }
        _ => return Err("pixel format"),
    };
    if dm1.fastclear() {
        return Err("pixel operation");
    }
    let x = context.xstart >> 11;
    let y = context.ystart >> 11;
    let x_end = context.xend >> 11;
    let y_end = context.yend >> 11;
    let octant = context.bresoctinc1.octant() & 7;
    let (incrx1, incrx2, incry1, incry2, y_major) = REX3_BRES_OCTANTS[octant as usize];
    let major = if kind == LineKind::Integer {
        i64::from((x_end - x).abs().max((y_end - y).abs()))
    } else if y_major {
        i64::from((y_end - y).abs())
    } else {
        i64::from((x_end - x).abs())
    };
    let pixel_count = u32::try_from(major + 1).map_err(|_| "line length")?;
    let endpoint_skip = endpoint_skips(context, kind);
    let x_move = if dm0.xyoffset() {
        (context.xymove >> 16) as i16 as i32
    } else {
        0
    };
    let y_move = if dm0.xyoffset() {
        context.xymove as i16 as i32
    } else {
        0
    };
    let x_window = (context.xywin >> 16) as i16 as i32;
    let y_window = context.xywin as i16 as i32;
    let geometry = LineGeometry {
        x,
        y,
        d: ((context.bresd << 5) as i32) >> 5,
        incr1: context.bresoctinc1.incr1() as i32,
        incr2: ((context.bresrndinc2.incr2() << 11) as i32) >> 11,
        incrx1,
        incrx2,
        incry1,
        incry2,
        y_major,
        straight_on_zero: bres_straight_on_zero(context, octant),
        pixel_count,
        skip_first: dm0.skipfirst() || endpoint_skip.0,
        skip_last: dm0.skiplast() || endpoint_skip.1,
        x_move,
        y_move,
        x_window,
        y_window,
        y_flip: dm1.yflip(),
        z_pattern: context.zpattern,
        line_pattern: context.lspattern,
        z_pattern_enabled: dm0.enzpattern(),
        line_pattern_enabled: dm0.enlspattern(),
        z_opaque: dm0.zpopaque(),
        line_opaque: dm0.lsopaque(),
        z_pattern_bit: context.zpat_bit,
        line_pattern_bit: context.pat_bit,
        line_pattern_count: context.lsmode.lsrcount(),
        line_pattern_repeat: context.lsmode.lsrepeat(),
        line_pattern_length: context.lsmode.lslength(),
        advance_last_pattern: dm0.lsadvlast(),
    };

    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;
    let mut final_x = x;
    let mut final_y = y;
    let mut final_d = geometry.d;
    for pixel in geometry.pixels() {
        final_x = pixel.x;
        final_y = pixel.y;
        final_d = pixel.d;
        if !pixel.skip
            && pixel.physical_x >= 0
            && pixel.physical_x < REX3_SCREEN_WIDTH
            && pixel.physical_y >= 0
            && pixel.physical_y < REX3_SCREEN_HEIGHT
        {
            min_x = min_x.min(pixel.physical_x);
            min_y = min_y.min(pixel.physical_y);
            max_x = max_x.max(pixel.physical_x);
            max_y = max_y.max(pixel.physical_y);
        }
    }

    let line = if min_x > max_x || min_y > max_y || context.wrmask == 0 {
        None
    } else {
        let alpha = if kind == LineKind::Antialiased {
            0xff
        } else {
            Rex3Context::clamp_color_component(context.coloralpha)
        };
        let raw_color = (context.get_colori() & 0x00ff_ffff) | (alpha << 24);
        Some(LineCommand {
            kind,
            plane,
            min_x,
            min_y,
            max_x,
            max_y,
            start_major: geometry.start_major(),
            y_major,
            major_decreasing: geometry.major_decreasing(),
            color: raw_color,
            write_mask: context.wrmask,
            payload_words: pixel_count,
        })
    };
    Ok(PreparedLine {
        line,
        geometry,
        final_x,
        final_y,
        final_d,
    })
}

fn endpoint_skips(context: &Rex3Context, kind: LineKind) -> (bool, bool) {
    if kind != LineKind::Antialiased || !context.drawmode0.endptfilter() {
        return (false, false);
    }
    let start_x = (context.xstart >> 7) & 0xf;
    let start_y = (context.ystart >> 7) & 0xf;
    let end_x = (context.xend >> 7) & 0xf;
    let end_y = (context.yend >> 7) & 0xf;
    let empty_weight = |weights: u32, x: i32, y: i32| {
        if x == 0 && y == 0 {
            return false;
        }
        let index = ((x + y) as u32).min(15);
        weights.checked_shr(index * 4).unwrap_or(0) & 0xf == 0
    };
    (
        empty_weight(context.aweight0, start_x, start_y),
        empty_weight(context.aweight1, end_x, end_y),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{
        DrawMode0, DrawMode1, CLIPMODE_CIDMATCH_SHIFT, DRAWMODE0_ADRMODE_A_LINE_SH, DRAWMODE0_ADRMODE_F_LINE_SH,
        DRAWMODE0_ADRMODE_I_LINE_SH, DRAWMODE1_COMPARE_DISABLE_SH,
        DRAWMODE1_DRAWDEPTH_24, DRAWMODE1_LOGICOP_SRC_SH,
    };
    use crate::dev::ng1::rex3_generic::{draw_with_fb, fline_setup, Framebuffers};
    use crate::dev::ng1::rex3_shape;

    const STRIDE: usize = 2048;
    const PLANE_WORDS: usize = STRIDE * REX3_SCREEN_HEIGHT as usize;

    fn on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("resident-line-test".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .expect("spawn line test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    fn fixed_16(value: i32) -> i32 {
        (REX3_COORD_BIAS << 11) + value * 128
    }

    fn context(kind: LineKind, start: (i32, i32), end: (i32, i32)) -> Rex3Context {
        let mut context = Rex3Context::power_on_default();
        let shape = match kind {
            LineKind::Integer => DRAWMODE0_ADRMODE_I_LINE_SH,
            LineKind::Fractional => DRAWMODE0_ADRMODE_F_LINE_SH,
            LineKind::Antialiased => DRAWMODE0_ADRMODE_A_LINE_SH,
        };
        context.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | shape | (1 << 8) | (1 << 9));
        context.drawmode1 = DrawMode1(
            DRAWMODE1_PLANES_RGB
                | (DRAWMODE1_DRAWDEPTH_24 << 3)
                | (1 << 15)
                | DRAWMODE1_COMPARE_DISABLE_SH
                | DRAWMODE1_LOGICOP_SRC_SH,
        );
        context.clipmode = 0xf << CLIPMODE_CIDMATCH_SHIFT;
        context.wrmask = 0xff0f_ffff;
        context.colorred = 0x31 << 11;
        context.colorgrn = 0x72 << 11;
        context.colorblue = 0xb4 << 11;
        context.coloralpha = 0x56 << 11;
        context.xstart = fixed_16(start.0);
        context.ystart = fixed_16(start.1);
        context.xend = fixed_16(end.0);
        context.yend = fixed_16(end.1);
        context
    }

    fn setup_integer(context: &mut Rex3Context, round: u32) {
        let dx = context.xend - context.xstart;
        let dy = context.yend - context.ystart;
        let adx = dx.abs() >> 11;
        let ady = dy.abs() >> 11;
        let mut octant = 0;
        if dy < 0 {
            octant |= crate::dev::ng1::rex3::OCTANT_YDEC;
        }
        if dx < 0 {
            octant |= crate::dev::ng1::rex3::OCTANT_XDEC;
        }
        if adx > ady {
            octant |= crate::dev::ng1::rex3::OCTANT_XMAJOR;
        }
        let (major, minor) = if adx > ady { (adx, ady) } else { (ady, adx) };
        let incr1 = 2 * minor;
        let incr2 = 2 * (minor - major);
        context.bresoctinc1.set_octant(octant);
        context.bresoctinc1.set_incr1(incr1 as u32 & 0x000f_ffff);
        context.bresrndinc2.set_incr2(incr2 as u32 & 0x001f_ffff);
        context.bresrndinc2.set_rnd(round);
        context.bresd = ((incr1 - major) as u32) & 0x07ff_ffff;
    }

    fn seed(salt: u32) -> Vec<u32> {
        (0..PLANE_WORDS as u32)
            .map(|index| index.wrapping_mul(0x9e37_79b9) ^ salt)
            .collect()
    }

    fn apply_prepared(prepared: &PreparedLine, rgb: &mut [u32], aux: &mut [u32]) {
        let Some(line) = prepared.line() else {
            return;
        };
        assert_eq!(prepared.payload().len(), line.payload_words as usize);
        for (index, word) in prepared.payload().enumerate() {
            if word == SKIPPED_PIXEL {
                continue;
            }
            let minor = (word & !BACKGROUND_PIXEL) as i32;
            let delta = if line.major_decreasing {
                -(index as i32)
            } else {
                index as i32
            };
            let major = line.start_major + delta;
            let (x, y) = if line.y_major {
                (minor, major)
            } else {
                (major, minor)
            };
            if x < 0 || x >= REX3_SCREEN_WIDTH || y < 0 || y >= REX3_SCREEN_HEIGHT {
                continue;
            }
            assert!(x >= line.min_x && x <= line.max_x);
            assert!(y >= line.min_y && y <= line.max_y);
            let plane = match line.plane {
                FillPlane::Rgb => &mut *rgb,
                FillPlane::Aux => &mut *aux,
            };
            let pixel = &mut plane[y as usize * STRIDE + x as usize];
            *pixel = (*pixel & !line.write_mask) | (line.color & line.write_mask);
        }
    }

    fn check(mut actual: Rex3Context, label: &str) {
        let mut expected = actual;
        let mut actual_rgb = seed(0x1234_5678);
        let mut actual_aux = seed(0x8765_4321);
        let mut expected_rgb = actual_rgb.clone();
        let mut expected_aux = actual_aux.clone();

        let prepared = prepare_line(&actual).unwrap_or_else(|reason| panic!("{label}: {reason}"));
        apply_prepared(&prepared, &mut actual_rgb, &mut actual_aux);
        prepared.finish(&mut actual);

        let framebuffers = Framebuffers {
            rgb: expected_rgb.as_mut_ptr(),
            aux: expected_aux.as_mut_ptr(),
        };
        let mode = rex3_shape::unpack(
            expected.drawmode0.0,
            expected.drawmode1.0,
            expected.clipmode,
        );
        draw_with_fb(&mut expected, &framebuffers, &mode);
        expected.hostrw_arm_single();

        assert_eq!(actual_rgb, expected_rgb, "{label}: RGB");
        assert_eq!(actual_aux, expected_aux, "{label}: auxiliary");
        assert_eq!(actual.xstart, expected.xstart, "{label}: xstart");
        assert_eq!(actual.ystart, expected.ystart, "{label}: ystart");
        assert_eq!(actual.bresd, expected.bresd, "{label}: bresd");
        assert_eq!(actual.pat_bit, expected.pat_bit, "{label}: pat_bit");
        assert_eq!(actual.zpat_bit, expected.zpat_bit, "{label}: zpat_bit");
        assert_eq!(
            actual.lsmode.lsrcount(),
            expected.lsmode.lsrcount(),
            "{label}: lsrcount",
        );
        assert_eq!(actual.host_len, expected.host_len, "{label}: host_len");
        assert_eq!(
            actual.host_cursor, expected.host_cursor,
            "{label}: host_cursor"
        );
    }

    #[test]
    fn line_payloads_match_cpu_geometry_and_completion() {
        on_large_stack(|| {
            let deltas = [
                (3, 7),
                (3, -7),
                (-3, 7),
                (-3, -7),
                (7, 3),
                (7, -3),
                (-7, 3),
                (-7, -3),
                (0, 7),
                (7, 0),
                (0, 0),
            ];
            for (index, delta) in deltas.into_iter().enumerate() {
                let mut line = context(
                    LineKind::Integer,
                    (40 * 16, 50 * 16),
                    ((40 + delta.0) * 16, (50 + delta.1) * 16),
                );
                setup_integer(&mut line, if index & 1 == 0 { 0 } else { 0xff });
                if index == 3 {
                    line.drawmode0.0 |= 1 << 10;
                }
                if index == 6 {
                    line.drawmode0.0 |= 1 << 11;
                }
                check(line, &format!("integer octant case {index}"));
            }

            let mut fractional = context(
                LineKind::Fractional,
                (20 * 16 + 3, 30 * 16 + 11),
                (31 * 16 + 13, 36 * 16 + 2),
            );
            fractional.drawmode0.0 |= 1 << 20;
            fractional.xymove = ((-4i16 as u16 as u32) << 16) | 2;
            fractional.xywin = ((2i16 as u16 as u32) << 16) | (-3i16 as u16 as u32);
            fractional.bresrndinc2.set_rnd(0xa5);
            fline_setup(&mut fractional);
            check(fractional, "fractional line");

            let mut patterned = context(
                LineKind::Integer,
                (42 * 16, 64 * 16),
                (79 * 16, 73 * 16),
            );
            patterned.drawmode0.0 |= (1 << 12) | (1 << 13) | (1 << 14);
            patterned.zpattern = 0xd6a5_39c7;
            patterned.lspattern = 0xb5ca_d000;
            patterned.zpat_bit = 27;
            patterned.pat_bit = 25;
            patterned.lsmode.set_lsrcount(4);
            patterned.lsmode.set_lsrepeat(5);
            patterned.lsmode.set_lslength(7);
            setup_integer(&mut patterned, 0x5a);
            check(patterned, "transparent combined-pattern line");

            let mut antialiased = context(
                LineKind::Antialiased,
                (70 * 16 + 7, 80 * 16 + 5),
                (76 * 16 + 9, 91 * 16 + 12),
            );
            antialiased.drawmode0.0 |= (1 << 10) | (1 << 22);
            antialiased.aweight0 = u32::MAX;
            antialiased.aweight1 = 0;
            fline_setup(&mut antialiased);
            check(antialiased, "antialiased endpoint-filtered line");
        });
    }

    #[test]
    fn line_rejection_and_empty_completion_are_transactional() {
        on_large_stack(|| {
            let mut line = context(LineKind::Integer, (10 * 16, 11 * 16), (14 * 16, 15 * 16));
            setup_integer(&mut line, 0);
            line.wrmask = 0;
            let original = line;
            let prepared = prepare_line(&line).unwrap();
            assert!(prepared.line().is_none());
            assert_eq!(line.xstart, original.xstart);
            assert_eq!(line.ystart, original.ystart);
            prepared.finish(&mut line);

            let mut rejected = original;
            rejected.drawmode0.0 |= 1 << 15;
            let unchanged = rejected;
            assert_eq!(prepare_line(&rejected).unwrap_err(), "draw mode 0");
            assert_eq!(rejected.xstart, unchanged.xstart);
            assert_eq!(rejected.ystart, unchanged.ystart);
            assert_eq!(rejected.bresd, unchanged.bresd);
        });
    }
}
