//! Flat fills shared by Newport's GPU raster implementations.
//!
//! Matching happens after the GO's setup, without changing the context. Keep
//! the prepared command across a batch-full retry and finish it only after
//! accepting it; setup must not be repeated.

use super::rex3::{
    Rex3Context, CLIPMODE_CIDMATCH_MASK, CLIPMODE_CIDMATCH_SHIFT, CLIPMODE_ENSMASK_MASK,
    DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_SPAN, DRAWMODE0_OPCODE_DRAW,
    DRAWMODE1_COMPARE_DISABLE, DRAWMODE1_DRAWDEPTH_24, DRAWMODE1_LOGICOP_SRC, DRAWMODE1_PLANES_CID,
    DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA,
    OCTANT_XDEC, OCTANT_YDEC, REX3_COORD_BIAS, REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH,
};
use super::rex3_generic::{fastclear_color, Mode};
use super::rex3_shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FillPlane {
    Rgb,
    Aux,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FillKind {
    Block,
    Span,
    Clear,
}

/// A nonempty, screen-clipped fill, independent of framebuffer layout.
///
/// The rectangle is `min_x..min_x + width`, `min_y..min_y + height`.
/// Although the planes have 2048 columns, CPU/JIT writes stop at column
/// 1344. Ordinary commands retain the raw source in `color`; clears retain
/// their replicated `COLORVRAM` word. `write_mask` always remains a full word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FlatFill {
    pub(crate) kind: FillKind,
    pub(crate) plane: FillPlane,
    pub(crate) min_x: u32,
    pub(crate) min_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) color: u32,
    pub(crate) write_mask: u32,
}

#[derive(Debug)]
enum ContextAdvance {
    Block { xsave: i32, y_after: i32 },
    Span { xend: i32 },
}

/// A matched GO and its context completion, prepared once after setup.
///
/// `fill()` is `None` for a supported draw with no pixel writes. Such a GO
/// still completes architecturally. This type is intentionally not `Copy`:
/// finishing consumes it so the same prepared GO cannot advance twice.
#[must_use]
#[derive(Debug)]
pub(crate) struct PreparedFlatFill {
    fill: Option<FlatFill>,
    advance: ContextAdvance,
}

impl PreparedFlatFill {
    pub(crate) fn fill(&self) -> Option<FlatFill> {
        self.fill
    }

    /// Complete an accepted GO, including a supported no-op.
    ///
    /// A refused/full batch must leave both this command and the post-setup
    /// context untouched so they can be retried after submission.
    pub(crate) fn finish(self, context: &mut Rex3Context) {
        match self.advance {
            ContextAdvance::Block { xsave, y_after } => {
                context.xstart = xsave;
                context.ystart = y_after;
            }
            ContextAdvance::Span { xend } => {
                context.xstart = xend + (1 << 11);
            }
        }
        context.pat_bit = 31;
        context.zpat_bit = 31;
        context.mid_primitive = false;
        context.host_len = 1;
        context.host_cursor = 0;
    }
}

/// Match and normalize one already-set-up GO, without mutating its context.
/// The rejection strings are the legacy prototype's fallback reasons.
pub(crate) fn prepare_flat_fill(context: &Rex3Context) -> Result<PreparedFlatFill, &'static str> {
    prepare(context, false)
}

/// The resident path also handles simple auxiliary-plane fills. The legacy
/// prototype continues to accept RGB only.
#[cfg(test)]
pub(crate) fn prepare_resident_flat_fill(
    context: &Rex3Context,
) -> Result<PreparedFlatFill, &'static str> {
    prepare(context, true)
}

fn prepare(context: &Rex3Context, allow_aux: bool) -> Result<PreparedFlatFill, &'static str> {
    let dm0 = context.drawmode0;
    let dm1 = context.drawmode1;
    let adrmode = dm0.adrmode();
    let block = adrmode == DRAWMODE0_ADRMODE_BLOCK;
    let full_primitive = match adrmode {
        DRAWMODE0_ADRMODE_SPAN => dm0.stoponx(),
        DRAWMODE0_ADRMODE_BLOCK => dm0.stoponx() && dm0.stopony(),
        _ => false,
    };
    if dm0.opcode() != DRAWMODE0_OPCODE_DRAW {
        return Err("opcode");
    }
    if !full_primitive {
        return Err("primitive shape");
    }
    if context.mid_primitive {
        return Err("continuation state");
    }
    if block && context.xsave != context.xstart {
        return Err("block start state");
    }
    // A multiword host token makes the CPU resume the GO even when this
    // draw reads no host data, so it cannot be treated as one fill.
    if context.host_len > 1 {
        return Err("host transfer");
    }
    let mode = rex3_shape::unpack(dm0.0, dm1.0, context.clipmode);
    let fast_clear = allow_aux && mode.fastclear() != 0;
    if dm0.colorhost()
        || dm0.alphahost()
        || dm0.skipfirst()
        || dm0.skiplast()
        || dm0.length32()
        || dm0.lronly()
        || (!allow_aux && dm0.xyoffset())
        || dm0.endptfilter()
        || dm0.ystride()
    {
        return Err("draw mode 0");
    }
    if !fast_clear
        && (dm0.enzpattern()
            || dm0.enlspattern()
            || dm0.zpopaque()
            || dm0.lsopaque()
            || dm0.shade()
            || dm0.ciclamp())
    {
        return Err("draw mode 0");
    }
    let plane = if allow_aux {
        match dm1.planes() {
            DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => FillPlane::Rgb,
            DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID => {
                FillPlane::Aux
            }
            _ => return Err("pixel format"),
        }
    } else {
        match dm1.planes() {
            DRAWMODE1_PLANES_RGB
                if dm1.drawdepth() == DRAWMODE1_DRAWDEPTH_24 && dm1.rgbmode() =>
            {
                FillPlane::Rgb
            }
            _ => return Err("pixel format"),
        }
    };
    if !allow_aux && dm1.yflip() {
        return Err("pixel operation");
    }
    if !fast_clear {
        let unsupported = if allow_aux {
            dm1.fastclear() || dm1.blend()
        } else {
            dm1.dblsrc()
                || dm1.compare() != DRAWMODE1_COMPARE_DISABLE
                || dm1.dither()
                || dm1.fastclear()
                || dm1.blend()
                || dm1.logicop() != DRAWMODE1_LOGICOP_SRC
        };
        if unsupported {
            return Err("pixel operation");
        }
    }
    if !allow_aux
        && (context.clipmode & CLIPMODE_ENSMASK_MASK != 0
            || context.clipmode & CLIPMODE_CIDMATCH_MASK
                != 0xF << CLIPMODE_CIDMATCH_SHIFT)
    {
        return Err("clip mode");
    }
    if (context.xstart | context.xend | context.ystart | context.yend) & 0x7FF != 0 {
        return Err("fractional coordinates");
    }
    let octant = context.bresoctinc1.octant();
    let x_dec = block && octant & OCTANT_XDEC != 0;
    let y_dec = block && octant & OCTANT_YDEC != 0;
    let x_direction_matches = if x_dec {
        context.xend <= context.xstart
    } else {
        context.xend >= context.xstart
    };
    let y_direction_matches = !block
        || if y_dec {
            context.yend <= context.ystart
        } else {
            context.yend >= context.ystart
        };
    if !x_direction_matches
        || !y_direction_matches
        || (!allow_aux && octant & (OCTANT_XDEC | OCTANT_YDEC) != 0)
    {
        return Err("negative direction");
    }

    let x_window = (context.xywin >> 16) as i16 as i32;
    let y_window = context.xywin as i16 as i32;
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
    let x0 = (context.xstart >> 11) + x_move + x_window - REX3_COORD_BIAS;
    let x1 = (context.xend >> 11) + x_move + x_window - REX3_COORD_BIAS;
    let physical_y = |raw: i32| {
        let absolute = (raw >> 11) + y_move + y_window;
        if dm1.yflip() {
            0x23ff - absolute
        } else {
            absolute - REX3_COORD_BIAS
        }
    };
    let y0 = physical_y(context.ystart);
    let y1 = if adrmode == DRAWMODE0_ADRMODE_BLOCK {
        physical_y(context.yend)
    } else {
        y0
    };

    let min_x = x0.min(x1).max(0);
    let max_x = x0.max(x1).min(REX3_SCREEN_WIDTH - 1);
    let min_y = y0.min(y1).max(0);
    let max_y = y0.max(y1).min(REX3_SCREEN_HEIGHT - 1);
    let fill = if min_x > max_x || min_y > max_y || context.wrmask == 0 {
        None
    } else {
        let color = if fast_clear {
            debug_assert_ne!(mode.fastclear(), 0);
            fastclear_color(context, &mode)
        } else {
            let raw_color = (context.get_colori() & 0x00FF_FFFF)
                | (Rex3Context::clamp_color_component(context.coloralpha) << 24);
            raw_color
        };
        Some(FlatFill {
            kind: if fast_clear {
                FillKind::Clear
            } else if block {
                FillKind::Block
            } else {
                FillKind::Span
            },
            plane,
            min_x: min_x as u32,
            min_y: min_y as u32,
            width: (max_x - min_x + 1) as u32,
            height: (max_y - min_y + 1) as u32,
            color,
            write_mask: context.wrmask,
        })
    };
    let advance = if block {
        ContextAdvance::Block {
            xsave: context.xsave,
            y_after: context.yend + if y_dec { -(1 << 11) } else { 1 << 11 },
        }
    } else {
        ContextAdvance::Span { xend: context.xend }
    };
    Ok(PreparedFlatFill { fill, advance })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{
        DrawMode0, DrawMode1, DRAWMODE0_ADRMODE_BLOCK_SH, DRAWMODE0_ADRMODE_SPAN_SH,
        DRAWMODE1_COMPARE_DISABLE_SH, DRAWMODE1_LOGICOP_SRC_SH,
    };
    use crate::dev::ng1::rex3_generic::{draw_with_fb, Framebuffers};
    use crate::dev::ng1::rex3_shape;

    const STRIDE: usize = 2048;
    const PIXELS: usize = STRIDE * REX3_SCREEN_HEIGHT as usize;

    fn on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("rex3-flat-fill-test".into())
            .stack_size(16 * 1024 * 1024)
            .spawn(f)
            .expect("spawn flat-fill test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    fn context(adrmode: u32, start: (i32, i32), end: (i32, i32)) -> Rex3Context {
        let mut ctx = Rex3Context::power_on_default();
        let stop = if adrmode == DRAWMODE0_ADRMODE_BLOCK_SH {
            (1 << 8) | (1 << 9)
        } else {
            1 << 8
        };
        ctx.drawmode0 = DrawMode0(DRAWMODE0_OPCODE_DRAW | adrmode | stop);
        ctx.drawmode1 = DrawMode1(
            DRAWMODE1_PLANES_RGB
                | (DRAWMODE1_DRAWDEPTH_24 << 3)
                | DRAWMODE1_COMPARE_DISABLE_SH
                | (1 << 15)
                | DRAWMODE1_LOGICOP_SRC_SH,
        );
        ctx.clipmode = 0xF << CLIPMODE_CIDMATCH_SHIFT;
        ctx.wrmask = 0xAA00_00FF;
        ctx.colorred = 0x11 << 11;
        ctx.colorgrn = 0x22 << 11;
        ctx.colorblue = 0x33 << 11;
        ctx.coloralpha = 0x7A << 11;
        ctx.xstart = (start.0 + REX3_COORD_BIAS) << 11;
        ctx.ystart = (start.1 + REX3_COORD_BIAS) << 11;
        ctx.xsave = ctx.xstart;
        ctx.xend = (end.0 + REX3_COORD_BIAS) << 11;
        ctx.yend = (end.1 + REX3_COORD_BIAS) << 11;
        ctx.pat_bit = 7;
        ctx.zpat_bit = 9;
        ctx.host_len = 0;
        ctx.host_cursor = 1;
        ctx
    }

    fn context_state(ctx: &Rex3Context) -> (i32, i32, u8, u8, bool, u32, u32) {
        (
            ctx.xstart,
            ctx.ystart,
            ctx.pat_bit,
            ctx.zpat_bit,
            ctx.mid_primitive,
            ctx.host_len,
            ctx.host_cursor,
        )
    }

    fn apply(fill: FlatFill, rgb: &mut [u32]) {
        for y in fill.min_y..fill.min_y + fill.height {
            for x in fill.min_x..fill.min_x + fill.width {
                let p = &mut rgb[y as usize * STRIDE + x as usize];
                *p = (*p & !fill.write_mask) | (fill.color & fill.write_mask);
            }
        }
    }

    fn assert_plane_eq(actual: &[u32], expected: &[u32], what: &str) {
        assert_eq!(actual.len(), expected.len());
        if let Some(i) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            panic!(
                "{what} at ({}, {}): got {:#010x}, expected {:#010x}",
                i % STRIDE,
                i / STRIDE,
                actual[i],
                expected[i]
            );
        }
    }

    fn compare_with_cpu(
        adrmode: u32,
        start: (i32, i32),
        end: (i32, i32),
        window: (i16, i16),
        mask: u32,
    ) {
        let mut normalized = context(adrmode, start, end);
        normalized.xywin = ((window.0 as u16 as u32) << 16) | window.1 as u16 as u32;
        normalized.wrmask = mask;
        let mut cpu = normalized;
        #[cfg(feature = "rex-jit")]
        let mut jit = normalized;
        let before = context_state(&normalized);
        let command = prepare_flat_fill(&normalized).expect("supported fill");
        assert_eq!(
            context_state(&normalized),
            before,
            "matching must not advance"
        );
        let fill = command.fill();

        let seed: Vec<u32> = (0..PIXELS as u32)
            .map(|i| i.wrapping_mul(0x9e37_79b9) ^ 0x5a5a_0000)
            .collect();
        let mut actual = seed.clone();
        let mut expected = seed.clone();
        let mut aux = vec![0u32; PIXELS];
        let fb = Framebuffers {
            rgb: expected.as_mut_ptr(),
            aux: aux.as_mut_ptr(),
        };
        let mode = rex3_shape::unpack(cpu.drawmode0.0, cpu.drawmode1.0, cpu.clipmode);
        draw_with_fb(&mut cpu, &fb, &mode);
        cpu.hostrw_arm_single();

        if let Some(fill) = fill {
            apply(fill, &mut actual);
        }
        command.finish(&mut normalized);
        assert_plane_eq(
            &actual,
            &expected,
            "normalized fill differed from the CPU walker",
        );
        assert_eq!(context_state(&normalized), context_state(&cpu));

        #[cfg(feature = "rex-jit")]
        {
            let mut jit_rgb = seed.clone();
            let mut compiler = crate::dev::ng1::rex3_jit::compiler::ShaderCompiler::new();
            let (entry, _) = compiler
                .compile_shader(jit.drawmode0.0, jit.drawmode1.0, jit.clipmode)
                .expect("compile flat-fill JIT shader");
            unsafe { entry(&mut jit, jit_rgb.as_mut_ptr(), aux.as_mut_ptr()) };
            jit.hostrw_arm_single();
            assert_plane_eq(&actual, &jit_rgb, "normalized fill differed from the JIT");
            assert_eq!(context_state(&normalized), context_state(&jit));
        }

        // Full storage does not expand the CPU/JIT write boundary.
        for y in 0..REX3_SCREEN_HEIGHT as usize {
            for x in REX3_SCREEN_WIDTH as usize..STRIDE {
                let i = y * STRIDE + x;
                assert_eq!(actual[i], seed[i], "offscreen storage at ({x}, {y})");
            }
        }
    }

    #[test]
    fn normalized_fills_match_cpu_writes_and_context() {
        on_large_stack(|| {
            compare_with_cpu(
                DRAWMODE0_ADRMODE_BLOCK_SH,
                (1338, 1021),
                (1353, 1026),
                (0, 0),
                0xAA00_00FF,
            );
            compare_with_cpu(
                DRAWMODE0_ADRMODE_BLOCK_SH,
                (-4, -2),
                (3, 3),
                (-1, 1),
                0xFF00_0000,
            );
            compare_with_cpu(
                DRAWMODE0_ADRMODE_SPAN_SH,
                (-5, 7),
                (5, 200),
                (2, -3),
                0x00FF_FFFF,
            );
            compare_with_cpu(
                DRAWMODE0_ADRMODE_BLOCK_SH,
                (1344, 2),
                (1351, 4),
                (0, 0),
                u32::MAX,
            );
            compare_with_cpu(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4), (0, 0), 0);
        });
    }

    #[test]
    fn bounds_masks_and_empty_go_are_distinct() {
        on_large_stack(|| {
            let mut ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1342, 1023), (1350, 1028));
            let command = prepare_flat_fill(&ctx).unwrap();
            assert_eq!(
                command.fill(),
                Some(FlatFill {
                    kind: FillKind::Block,
                    plane: FillPlane::Rgb,
                    min_x: 1342,
                    min_y: 1023,
                    width: 2,
                    height: 1,
                    color: 0x7A33_2211,
                    write_mask: 0xAA00_00FF,
                })
            );
            assert_eq!(
                (ctx.xstart, ctx.ystart),
                (
                    (1342 + REX3_COORD_BIAS) << 11,
                    (1023 + REX3_COORD_BIAS) << 11
                )
            );
            command.finish(&mut ctx);
            assert_eq!(ctx.ystart, (1029 + REX3_COORD_BIAS) << 11);

            let mut empty = context(DRAWMODE0_ADRMODE_SPAN_SH, (1344, 0), (1350, 8));
            let command = prepare_flat_fill(&empty).unwrap();
            assert_eq!(command.fill(), None);
            command.finish(&mut empty);
            assert_eq!(empty.xstart, (1351 + REX3_COORD_BIAS) << 11);
            assert_eq!((empty.host_len, empty.host_cursor), (1, 0));
            assert_eq!((empty.pat_bit, empty.zpat_bit), (31, 31));

            let mut offset = context(DRAWMODE0_ADRMODE_BLOCK_SH, (10, 20), (12, 23));
            offset.drawmode0.0 |= 1 << 20;
            offset.xymove = ((-3i16 as u16 as u32) << 16) | 4;
            offset.xywin = ((5i16 as u16 as u32) << 16) | (-2i16 as u16 as u32);
            offset.clipmode = 1 | (0b0101 << CLIPMODE_CIDMATCH_SHIFT);
            assert_eq!(prepare_flat_fill(&offset).unwrap_err(), "draw mode 0");
            let fill = prepare_resident_flat_fill(&offset).unwrap().fill().unwrap();
            assert_eq!(
                (fill.min_x, fill.min_y, fill.width, fill.height),
                (12, 22, 3, 4),
            );
        });
    }

    #[test]
    fn reverse_blocks_and_fast_clears_match_cpu() {
        on_large_stack(|| {
            for fast_clear in [false, true] {
                let mut normalized = context(DRAWMODE0_ADRMODE_BLOCK_SH, (9, 8), (3, 2));
                normalized.bresoctinc1.set_octant(OCTANT_XDEC | OCTANT_YDEC);
                if fast_clear {
                    normalized.drawmode1.0 |= (1 << 16) | (1 << 17) | (1 << 18);
                    normalized.colorvram = 0x00ab_cdef;
                }
                let legacy_error = prepare_flat_fill(&normalized).unwrap_err();
                assert_eq!(
                    legacy_error,
                    if fast_clear {
                        "pixel operation"
                    } else {
                        "negative direction"
                    }
                );

                let mut cpu = normalized;
                let before = context_state(&normalized);
                let command = prepare_resident_flat_fill(&normalized).expect("resident fill");
                assert_eq!(context_state(&normalized), before);
                let fill = command.fill().expect("visible resident fill");
                assert_eq!(
                    fill.kind,
                    if fast_clear {
                        FillKind::Clear
                    } else {
                        FillKind::Block
                    }
                );
                assert_eq!(
                    (fill.min_x, fill.min_y, fill.width, fill.height),
                    (3, 2, 7, 7)
                );
                if fast_clear {
                    assert_eq!(fill.color, 0x00ab_cdef);
                }

                let seed: Vec<u32> = (0..PIXELS as u32)
                    .map(|i| i.wrapping_mul(0x9e37_79b9) ^ 0x5a5a_0000)
                    .collect();
                let mut actual = seed.clone();
                let mut expected = seed.clone();
                let mut aux = vec![0u32; PIXELS];
                apply(fill, &mut actual);
                let fb = Framebuffers {
                    rgb: expected.as_mut_ptr(),
                    aux: aux.as_mut_ptr(),
                };
                let mode = rex3_shape::unpack(cpu.drawmode0.0, cpu.drawmode1.0, cpu.clipmode);
                draw_with_fb(&mut cpu, &fb, &mode);
                cpu.hostrw_arm_single();
                command.finish(&mut normalized);
                assert_plane_eq(&actual, &expected, "resident reverse fill");
                assert_eq!(context_state(&normalized), context_state(&cpu));
                assert_eq!(normalized.xstart, (9 + REX3_COORD_BIAS) << 11);
                assert_eq!(normalized.ystart, (1 + REX3_COORD_BIAS) << 11);
            }
        });
    }

    #[test]
    fn unsupported_draw_does_not_change_context() {
        on_large_stack(|| {
            let mut ctx = context(DRAWMODE0_ADRMODE_BLOCK_SH, (1, 2), (3, 4));
            ctx.host_len = 2;
            let before = context_state(&ctx);
            assert_eq!(prepare_flat_fill(&ctx).unwrap_err(), "host transfer");
            assert_eq!(context_state(&ctx), before);
            ctx.host_len = 0;
            ctx.xstart |= 1;
            ctx.xsave = ctx.xstart;
            let before = context_state(&ctx);
            assert_eq!(
                prepare_flat_fill(&ctx).unwrap_err(),
                "fractional coordinates"
            );
            assert_eq!(context_state(&ctx), before);
        });
    }
}
