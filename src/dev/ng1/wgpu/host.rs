use crate::dev::ng1::rex3::{
    Rex3Context, DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_SPAN, DRAWMODE0_OPCODE_DRAW,
    DRAWMODE1_PLANES_CID, DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_RGB,
    DRAWMODE1_PLANES_RGBA, HOSTRW_BUF_QWORDS, OCTANT_XDEC, OCTANT_YDEC,
};
use crate::dev::ng1::rex3_flat_fill::FillPlane;
use crate::dev::ng1::rex3_generic::{calculate_fb_address, DynMode, Mode};
use crate::dev::ng1::rex3_shape;

use super::shade::ShadeDda;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum HostKind {
    Block,
    Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostFill {
    pub(super) kind: HostKind,
    pub(super) plane: FillPlane,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostCompletion {
    xstart: i32,
    ystart: i32,
    pat_bit: u8,
    zpat_bit: u8,
    mid_primitive: bool,
    hostcnt: u32,
    host_shifter: u64,
    lsrcount: u32,
    shade: ShadeDda,
}

#[must_use]
#[derive(Debug)]
pub(super) struct PreparedHostFill {
    fill: Option<HostFill>,
    completion: HostCompletion,
    payload_qwords: usize,
    fast_clear: bool,
}

impl PreparedHostFill {
    pub(super) fn fill(&self) -> Option<HostFill> {
        self.fill
    }

    pub(super) fn payload_qwords(&self) -> usize {
        self.payload_qwords
    }

    pub(super) fn fast_clear(&self) -> bool {
        self.fast_clear
    }

    pub(super) fn finish(self, context: &mut Rex3Context) {
        context.xstart = self.completion.xstart;
        context.ystart = self.completion.ystart;
        context.pat_bit = self.completion.pat_bit;
        context.zpat_bit = self.completion.zpat_bit;
        context.mid_primitive = self.completion.mid_primitive;
        context.hostcnt = self.completion.hostcnt;
        context.host_shifter = self.completion.host_shifter;
        context.lsmode.set_lsrcount(self.completion.lsrcount);
        self.completion.shade.write_back(context);
        context.host_len = 1;
        context.host_cursor = 0;
    }
}

#[derive(Clone, Copy)]
struct WalkState {
    xstart: i32,
    ystart: i32,
    pat_bit: u8,
    zpat_bit: u8,
    mid_primitive: bool,
    lsrcount: u32,
    shade: ShadeDda,
}

#[derive(Clone, Copy)]
pub(super) struct HostStream {
    count: u32,
    shifter: u64,
    cursor: u32,
    length: u32,
}

impl HostStream {
    pub(super) fn from_context(context: &Rex3Context) -> Self {
        Self {
            count: context.hostcnt,
            shifter: context.host_shifter,
            cursor: context.host_cursor,
            length: context.host_len,
        }
    }

    pub(super) fn consume(&mut self, context: &Rex3Context) {
        if self.count == 0 {
            let last = self.length.saturating_sub(1);
            let index = self.cursor.min(last) as usize;
            self.shifter = context.hostrw[index.min(HOSTRW_BUF_QWORDS - 1)];
            if context.drawmode1.swapendian() {
                self.shifter = if context.drawmode1.rwdouble() {
                    self.shifter.swap_bytes()
                } else {
                    let high = (self.shifter >> 32) as u32;
                    ((high.swap_bytes() as u64) << 32) | (self.shifter & 0xffff_ffff)
                };
            }
            if self.cursor < self.length {
                self.cursor += 1;
            }
            self.count = rex3_shape::host_count(context.drawmode1.0);
        }

        self.shifter <<= rex3_shape::host_shift(context.drawmode1.0);
        self.count -= 1;
    }

    pub(super) fn drained(self) -> bool {
        self.cursor >= self.length
    }

    pub(super) fn reset_count(&mut self) {
        self.count = 0;
    }

    pub(super) fn count(self) -> u32 {
        self.count
    }

    pub(super) fn shifter(self) -> u64 {
        self.shifter
    }

    pub(super) fn cursor(self) -> u32 {
        self.cursor
    }

    pub(super) fn length(self) -> u32 {
        self.length
    }
}

pub(super) fn prepare_host_fill(context: &Rex3Context) -> Result<PreparedHostFill, &'static str> {
    if context.drawmode0.opcode() != DRAWMODE0_OPCODE_DRAW
        || !(context.drawmode0.colorhost() || context.drawmode0.alphahost())
    {
        return Err("host transfer");
    }
    let kind = match context.drawmode0.adrmode() {
        DRAWMODE0_ADRMODE_BLOCK => HostKind::Block,
        DRAWMODE0_ADRMODE_SPAN => HostKind::Span,
        _ => return Err("host address mode"),
    };
    if context.host_len as usize > HOSTRW_BUF_QWORDS {
        return Err("host payload length");
    }

    let mode = rex3_shape::unpack(context.drawmode0.0, context.drawmode1.0, context.clipmode);
    let fast_clear = mode.fastclear() != 0;
    let plane = match context.drawmode1.planes() {
        DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => Some(FillPlane::Rgb),
        DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID => Some(FillPlane::Aux),
        _ => None,
    };
    let mut walk = WalkState {
        xstart: context.xstart,
        ystart: context.ystart,
        pat_bit: context.pat_bit,
        zpat_bit: context.zpat_bit,
        mid_primitive: context.mid_primitive,
        lsrcount: context.lsmode.lsrcount(),
        shade: ShadeDda::from_context(context),
    };
    let mut stream = HostStream::from_context(context);
    let mut write_address_seen = false;
    run_invocation(
        context,
        &mode,
        kind,
        &mut walk,
        &mut stream,
        &mut write_address_seen,
    );
    while stream.length > 1 && stream.cursor < stream.length {
        let before = stream.cursor;
        stream.count = 0;
        run_invocation(
            context,
            &mode,
            kind,
            &mut walk,
            &mut stream,
            &mut write_address_seen,
        );
        if stream.cursor == before {
            break;
        }
    }

    let fill = if write_address_seen && context.wrmask != 0 {
        plane.map(|plane| HostFill { kind, plane })
    } else {
        None
    };
    Ok(PreparedHostFill {
        fill,
        completion: HostCompletion {
            xstart: walk.xstart,
            ystart: walk.ystart,
            pat_bit: walk.pat_bit,
            zpat_bit: walk.zpat_bit,
            mid_primitive: walk.mid_primitive,
            hostcnt: stream.count,
            host_shifter: stream.shifter,
            lsrcount: walk.lsrcount,
            shade: walk.shade,
        },
        payload_qwords: context.host_len.max(1) as usize,
        fast_clear,
    })
}

fn run_invocation(
    context: &Rex3Context,
    mode: &DynMode,
    kind: HostKind,
    walk: &mut WalkState,
    stream: &mut HostStream,
    write_address_seen: &mut bool,
) {
    match kind {
        HostKind::Block => run_block(context, mode, walk, stream, write_address_seen),
        HostKind::Span => run_span(context, mode, walk, stream, write_address_seen),
    }
}

#[derive(Clone, Copy)]
struct PatternResult {
    draws: bool,
    background: bool,
}

fn pattern_result(context: &Rex3Context, mode: &DynMode, walk: &WalkState) -> PatternResult {
    let mut background = false;
    if mode.enzpattern() != 0 && (context.zpattern >> walk.zpat_bit) & 1 == 0 {
        if mode.zpopaque() != 0 {
            background = true;
        } else {
            return PatternResult {
                draws: false,
                background: false,
            };
        }
    }
    if mode.enlspattern() != 0 && (context.lspattern >> walk.pat_bit) & 1 == 0 {
        if mode.lsopaque() != 0 {
            background = true;
        } else {
            return PatternResult {
                draws: false,
                background: false,
            };
        }
    }
    PatternResult {
        draws: true,
        background,
    }
}

fn visit_pixel(
    context: &Rex3Context,
    mode: &DynMode,
    x: i32,
    y: i32,
    walk: &WalkState,
    stream: &mut HostStream,
    write_address_seen: &mut bool,
) {
    let pattern = pattern_result(context, mode, walk);
    if !pattern.draws {
        return;
    }
    if !pattern.background {
        stream.consume(context);
    }
    *write_address_seen |= calculate_fb_address::<DynMode, true>(mode, x, y, context).is_some();
}

fn advance_pixel(context: &Rex3Context, mode: &DynMode, walk: &mut WalkState) {
    if mode.shade() != 0 {
        walk.shade.advance();
    }
    if mode.enzpattern() != 0 {
        walk.zpat_bit = walk.zpat_bit.wrapping_sub(1) & 31;
    }
    if mode.enlspattern() != 0 {
        let repeat = context.lsmode.lsrepeat().max(1);
        if walk.lsrcount == 0 {
            walk.lsrcount = repeat - 1;
            let wrap = 32 - (context.lsmode.lslength() + 17);
            walk.pat_bit = if u32::from(walk.pat_bit) == wrap {
                31
            } else {
                walk.pat_bit.wrapping_sub(1) & 31
            };
        } else {
            walk.lsrcount -= 1;
        }
    }
}

fn run_block(
    context: &Rex3Context,
    mode: &DynMode,
    walk: &mut WalkState,
    stream: &mut HostStream,
    write_address_seen: &mut bool,
) {
    let stop_on_y = mode.stopony() != 0;
    let color_host = mode.colorhost() != 0;
    let stop_on_word = color_host && !stop_on_y;
    let stop_on_x = mode.stoponx() != 0 || stop_on_word;
    let octant = context.bresoctinc1.octant();
    let x_decreasing = octant & OCTANT_XDEC != 0;
    let y_decreasing = octant & OCTANT_YDEC != 0;
    let x_step = if x_decreasing { -(1 << 11) } else { 1 << 11 };
    let y_stride = if mode.ystride() != 0 { 2 } else { 1 };
    let y_step = if y_decreasing {
        -(y_stride << 11)
    } else {
        y_stride << 11
    };
    let skip_first = mode.skipfirst() != 0;
    let skip_last = mode.skiplast() != 0;
    let lr_skip = mode.lronly() != 0 && x_decreasing;
    let span_len = (context.xend - walk.xstart).abs() >> 11;
    let x_stop = if mode.length32() != 0 && span_len >= 32 {
        Some(walk.xstart + x_step * 32)
    } else {
        None
    };

    walk.mid_primitive = true;
    loop {
        let x = walk.xstart >> 11;
        let y = walk.ystart >> 11;
        walk.xstart += x_step;
        let x_end_reached = if x_decreasing {
            walk.xstart < context.xend
        } else {
            walk.xstart > context.xend
        };
        if !(skip_first || x_end_reached && skip_last || lr_skip) {
            visit_pixel(context, mode, x, y, walk, stream, write_address_seen);
        }
        advance_pixel(context, mode, walk);

        if x_end_reached {
            walk.ystart += y_step;
            walk.xstart = context.xsave;
            walk.pat_bit = 31;
            walk.zpat_bit = 31;
            if !stop_on_y {
                walk.mid_primitive = false;
                break;
            }

            let y_end_reached = if y_decreasing {
                walk.ystart < context.yend
            } else {
                walk.ystart > context.yend
            };
            if y_end_reached {
                walk.mid_primitive = false;
                break;
            }
            if stop_on_word && stream.count > 0 {
                break;
            }
            if color_host && !stop_on_word {
                stream.count = 0;
            }
        } else if let Some(limit) = x_stop {
            let limit_reached = if x_decreasing {
                walk.xstart <= limit
            } else {
                walk.xstart >= limit
            };
            if limit_reached {
                break;
            }
        }

        if stop_on_word && stream.count == 0 && stream.drained() {
            break;
        }
        if !stop_on_x {
            break;
        }
    }
}

fn run_span(
    context: &Rex3Context,
    mode: &DynMode,
    walk: &mut WalkState,
    stream: &mut HostStream,
    write_address_seen: &mut bool,
) {
    if mode.lronly() != 0 && context.bresoctinc1.octant() & OCTANT_XDEC != 0 {
        return;
    }
    let stop_on_word = mode.colorhost() != 0;
    let stop_on_x = mode.stoponx() != 0 || stop_on_word;
    let skip_first = mode.skipfirst() != 0;
    let skip_last = mode.skiplast() != 0;
    let span_len = (context.xend - walk.xstart) >> 11;
    let x_stop = if mode.length32() != 0 && span_len >= 32 {
        Some(walk.xstart + (32 << 11))
    } else {
        None
    };
    let mut first = true;

    walk.mid_primitive = true;
    let completed = loop {
        let x = walk.xstart >> 11;
        let y = walk.ystart >> 11;
        walk.xstart += 1 << 11;
        let x_end_reached = walk.xstart > context.xend;
        if !(first && skip_first || x_end_reached && skip_last) {
            visit_pixel(context, mode, x, y, walk, stream, write_address_seen);
        }
        advance_pixel(context, mode, walk);
        if x_end_reached {
            break true;
        } else if let Some(limit) = x_stop {
            if walk.xstart >= limit {
                break false;
            }
        }
        if stop_on_word && stream.count == 0 && stream.drained() {
            break false;
        }
        if !stop_on_x {
            break false;
        }
        first = false;
    };

    if completed {
        walk.pat_bit = 31;
        walk.zpat_bit = 31;
        walk.mid_primitive = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{
        DrawMode0, DrawMode1, DRAWMODE0_ADRMODE_BLOCK_SH, DRAWMODE0_ADRMODE_SPAN_SH,
        DRAWMODE1_COMPARE_DISABLE_SH, DRAWMODE1_DRAWDEPTH_24, DRAWMODE1_LOGICOP_SRC_SH,
        DRAWMODE1_PLANES_RGB, REX3_COORD_BIAS,
    };
    use crate::dev::ng1::rex3_generic::{draw_block_g, draw_span_g, Framebuffers};

    fn on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("resident-host-test".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .expect("spawn resident host test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    fn context(block: bool, draw_mode: u32, host_mode: u32) -> Rex3Context {
        let mut context = Rex3Context::power_on_default();
        context.drawmode0 = DrawMode0(
            DRAWMODE0_OPCODE_DRAW
                | if block {
                    DRAWMODE0_ADRMODE_BLOCK_SH
                } else {
                    DRAWMODE0_ADRMODE_SPAN_SH
                }
                | draw_mode,
        );
        context.drawmode1 = DrawMode1(
            DRAWMODE1_PLANES_RGB
                | (DRAWMODE1_DRAWDEPTH_24 << 3)
                | (1 << 15)
                | DRAWMODE1_COMPARE_DISABLE_SH
                | DRAWMODE1_LOGICOP_SRC_SH
                | host_mode,
        );
        context.clipmode = 0xf << 9;
        context.wrmask = u32::MAX;
        context.colorred = 0x23 << 11;
        context.colorgrn = 0x45 << 11;
        context.colorblue = 0x67 << 11;
        context.coloralpha = 0x89 << 11;
        context.xstart = (REX3_COORD_BIAS + 17) << 11;
        context.xsave = context.xstart;
        context.ystart = (REX3_COORD_BIAS + 19) << 11;
        context.xend = (REX3_COORD_BIAS + 21) << 11;
        context.yend = (REX3_COORD_BIAS + 21) << 11;
        context.pat_bit = 7;
        context.zpat_bit = 9;
        context.host_len = 3;
        context.host_cursor = 0;
        context.hostrw[0] = 0x0123_4567_89ab_cdef;
        context.hostrw[1] = 0xfedc_ba98_7654_3210;
        context.hostrw[2] = 0x0f1e_2d3c_4b5a_6978;
        context
    }

    fn run_cpu(context: &mut Rex3Context) {
        let mut rgb = vec![0u32; 2048 * 1024];
        let mut aux = vec![0u32; 2048 * 1024];
        let framebuffers = Framebuffers {
            rgb: rgb.as_mut_ptr(),
            aux: aux.as_mut_ptr(),
        };
        let mode = rex3_shape::unpack(context.drawmode0.0, context.drawmode1.0, context.clipmode);
        let draw = |context: &mut Rex3Context| match context.drawmode0.adrmode() {
            DRAWMODE0_ADRMODE_BLOCK => draw_block_g(&framebuffers, context, &mode),
            DRAWMODE0_ADRMODE_SPAN => draw_span_g(&framebuffers, context, &mode),
            _ => unreachable!(),
        };
        draw(context);
        while context.host_len > 1 && context.host_cursor < context.host_len {
            let before = context.host_cursor;
            context.hostcnt = 0;
            draw(context);
            if context.host_cursor == before {
                break;
            }
        }
        context.host_len = 1;
        context.host_cursor = 0;
    }

    fn assert_completion(actual: &Rex3Context, expected: &Rex3Context, label: &str) {
        assert_eq!(actual.xstart, expected.xstart, "{label}: xstart");
        assert_eq!(actual.ystart, expected.ystart, "{label}: ystart");
        assert_eq!(actual.pat_bit, expected.pat_bit, "{label}: pat_bit");
        assert_eq!(actual.zpat_bit, expected.zpat_bit, "{label}: zpat_bit");
        assert_eq!(
            actual.mid_primitive, expected.mid_primitive,
            "{label}: mid_primitive"
        );
        assert_eq!(actual.hostcnt, expected.hostcnt, "{label}: hostcnt");
        assert_eq!(
            actual.host_shifter, expected.host_shifter,
            "{label}: host_shifter"
        );
        assert_eq!(
            actual.host_cursor, expected.host_cursor,
            "{label}: host_cursor"
        );
        assert_eq!(actual.host_len, expected.host_len, "{label}: host_len");
        assert_eq!(actual.lsmode.0, expected.lsmode.0, "{label}: lsmode");
        assert_eq!(actual.colorred, expected.colorred, "{label}: colorred");
        assert_eq!(
            actual.coloralpha, expected.coloralpha,
            "{label}: coloralpha"
        );
        assert_eq!(actual.colorgrn, expected.colorgrn, "{label}: colorgrn");
        assert_eq!(actual.colorblue, expected.colorblue, "{label}: colorblue");
    }

    #[test]
    fn host_fill_completion_matches_cpu_walker() {
        on_large_stack(|| {
            let cases = [
                (
                    "full packed block",
                    true,
                    (1 << 6) | (1 << 8) | (1 << 9),
                    (1 << 7) | (1 << 8),
                ),
                (
                    "row segments",
                    true,
                    (1 << 6) | (1 << 8),
                    (1 << 7) | (2 << 8),
                ),
                (
                    "double swapped span",
                    false,
                    (1 << 6),
                    (1 << 7) | (1 << 10) | (1 << 11),
                ),
                ("alpha-only span", false, (1 << 7) | (1 << 8), (3 << 8)),
                ("alpha batch steps", false, 1 << 7, (1 << 7) | (2 << 8)),
            ];
            for (label, block, draw_mode, host_mode) in cases {
                let context = context(block, draw_mode, host_mode);
                let prepared = prepare_host_fill(&context).expect(label);
                assert_eq!(prepared.payload_qwords(), context.host_len as usize);
                assert!(prepared.fill().is_some(), "{label}");

                let mut actual = context;
                prepared.finish(&mut actual);
                let mut expected = context;
                run_cpu(&mut expected);
                assert_completion(&actual, &expected, label);
            }
        });
    }

    #[test]
    fn host_fill_segments_match_cpu_walker() {
        on_large_stack(|| {
            let mut cases = Vec::new();

            let mut patterned = context(
                false,
                (1 << 6) | (1 << 8) | (1 << 12) | (1 << 13) | (1 << 18),
                (1 << 7) | (1 << 8),
            );
            patterned.zpattern = 0xa000_0000;
            patterned.lspattern = 0xc000_0000;
            patterned.pat_bit = 7;
            patterned.zpat_bit = 9;
            patterned.lsmode.set_lsrepeat(2);
            patterned.lsmode.set_lslength(7);
            patterned.lsmode.set_lsrcount(1);
            patterned.slopered = 0x321;
            patterned.slopealpha = -0x211;
            patterned.slopegrn = 0x123;
            patterned.slopeblue = -0x101;
            cases.push(("patterned shade", patterned, true));

            let mut background = context(
                false,
                (1 << 6) | (1 << 8) | (1 << 12) | (1 << 13) | (1 << 16) | (1 << 17),
                1 << 7,
            );
            background.zpattern = 0;
            background.lspattern = 0;
            cases.push(("opaque background", background, true));

            cases.push((
                "span endpoint skips",
                context(
                    false,
                    (1 << 6) | (1 << 8) | (1 << 10) | (1 << 11),
                    (1 << 7) | (1 << 8),
                ),
                true,
            ));

            let mut length32 = context(false, (1 << 6) | (1 << 8) | (1 << 15), (1 << 7) | (1 << 8));
            length32.xend = (REX3_COORD_BIAS + 90) << 11;
            cases.push(("LENGTH32 span", length32, true));

            let mut reverse_lr = context(true, (1 << 6) | (1 << 8) | (1 << 9) | (1 << 19), 1 << 7);
            reverse_lr.xstart = (REX3_COORD_BIAS + 21) << 11;
            reverse_lr.xsave = reverse_lr.xstart;
            reverse_lr.xend = (REX3_COORD_BIAS + 17) << 11;
            reverse_lr.bresoctinc1.set_octant(OCTANT_XDEC);
            cases.push(("reverse LR-only block", reverse_lr, false));

            let mut span_lr = context(false, (1 << 6) | (1 << 8) | (1 << 19), 1 << 7);
            span_lr.bresoctinc1.set_octant(OCTANT_XDEC);
            cases.push(("reverse LR-only span", span_lr, false));

            cases.push((
                "BLOCK skip-first suppression",
                context(true, (1 << 6) | (1 << 8) | (1 << 9) | (1 << 10), 1 << 7),
                false,
            ));

            for (label, context, writes) in cases {
                let prepared = prepare_host_fill(&context).expect(label);
                assert_eq!(prepared.fill().is_some(), writes, "{label}: fill");
                let mut actual = context;
                prepared.finish(&mut actual);
                let mut expected = context;
                run_cpu(&mut expected);
                assert_completion(&actual, &expected, label);
            }

            let mut fast_clear = context(false, 1 << 7, 1 << 7);
            fast_clear.drawmode1.0 |= 1 << 17;
            fast_clear.colorvram = 0x0065_4321;
            let prepared = prepare_host_fill(&fast_clear).expect("ALPHAHOST fast clear");
            assert!(prepared.fast_clear());
            let mut actual = fast_clear;
            prepared.finish(&mut actual);
            let mut expected = fast_clear;
            run_cpu(&mut expected);
            assert_completion(&actual, &expected, "ALPHAHOST fast clear");

            let mut color_host = context(false, (1 << 6) | (1 << 8), 1 << 7);
            color_host.drawmode1.0 |= 1 << 17;
            assert!(
                !prepare_host_fill(&color_host)
                    .expect("COLORHOST overrides fast clear")
                    .fast_clear()
            );
        });
    }
}
