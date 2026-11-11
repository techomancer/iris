use crate::dev::ng1::rex3::{
    Rex3Context, DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_SPAN,
    DRAWMODE0_OPCODE_SCR2SCR, DRAWMODE1_PLANES_CID, DRAWMODE1_PLANES_OLAY,
    DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA,
    OCTANT_XDEC, OCTANT_YDEC, REX3_COORD_BIAS, REX3_SCREEN_HEIGHT,
    REX3_SCREEN_WIDTH,
};
use crate::dev::ng1::rex3_flat_fill::FillPlane;
use crate::dev::ng1::rex3_generic::ls_pattern_state_after;

use super::shade::ShadeDda;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CopyKind {
    Block,
    Span,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CopyFill {
    pub(super) kind: CopyKind,
    pub(super) plane: FillPlane,
    pub(super) min_x: i32,
    pub(super) min_y: i32,
    pub(super) max_x: i32,
    pub(super) max_y: i32,
    pub(super) walker_x0: i32,
    pub(super) walker_y0: i32,
    pub(super) walker_x1: i32,
    pub(super) walker_y1: i32,
}

#[derive(Clone, Copy, Debug)]
enum ContextAdvance {
    Block { xsave: i32, y_after: i32 },
    Span { x_after: i32 },
    None,
}

#[must_use]
#[derive(Debug)]
pub(super) struct PreparedCopyFill {
    fill: Option<CopyFill>,
    advance: ContextAdvance,
    final_lsrcount: Option<u32>,
    final_shade: Option<ShadeDda>,
}

impl PreparedCopyFill {
    pub(super) fn fill(&self) -> Option<CopyFill> {
        self.fill
    }

    pub(super) fn finish(self, context: &mut Rex3Context) {
        if let Some(shade) = self.final_shade {
            shade.write_back(context);
        }
        match self.advance {
            ContextAdvance::Block { xsave, y_after } => {
                context.xstart = xsave;
                context.ystart = y_after;
                context.pat_bit = 31;
                context.zpat_bit = 31;
                context.mid_primitive = false;
            }
            ContextAdvance::Span { x_after } => {
                context.xstart = x_after;
                context.pat_bit = 31;
                context.zpat_bit = 31;
                context.mid_primitive = false;
            }
            ContextAdvance::None => {}
        }
        if let Some(count) = self.final_lsrcount {
            context.lsmode.set_lsrcount(count);
        }
        context.host_len = 1;
        context.host_cursor = 0;
    }
}

pub(super) fn prepare_copy_fill(
    context: &Rex3Context,
) -> Result<PreparedCopyFill, &'static str> {
    let dm0 = context.drawmode0;
    let dm1 = context.drawmode1;
    let kind = match dm0.adrmode() {
        DRAWMODE0_ADRMODE_BLOCK => CopyKind::Block,
        DRAWMODE0_ADRMODE_SPAN => CopyKind::Span,
        _ => return Err("primitive shape"),
    };
    if dm0.opcode() != DRAWMODE0_OPCODE_SCR2SCR {
        return Err("opcode");
    }
    let full_primitive = match kind {
        CopyKind::Block => dm0.stoponx() && dm0.stopony(),
        CopyKind::Span => dm0.stoponx(),
    };
    if !full_primitive {
        return Err("continuation state");
    }
    if context.mid_primitive {
        return Err("continuation state");
    }
    if context.host_len > 1 || dm0.colorhost() || dm0.alphahost() {
        return Err("host transfer");
    }
    if dm0.length32() {
        return Err("continuation state");
    }

    let plane = match dm1.planes() {
        DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => Some(FillPlane::Rgb),
        DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID => {
            Some(FillPlane::Aux)
        }
        _ => None,
    };
    let walker_x0 = context.xstart >> 11;
    let walker_y0 = context.ystart >> 11;
    let requested_x1 = context.xend >> 11;
    let requested_y1 = context.yend >> 11;
    let octant = context.bresoctinc1.octant();
    let x_dec = kind == CopyKind::Block && octant & OCTANT_XDEC != 0;
    let y_dec = kind == CopyKind::Block && octant & OCTANT_YDEC != 0;

    if kind == CopyKind::Span && dm0.lronly() && octant & OCTANT_XDEC != 0 {
        return Ok(PreparedCopyFill {
            fill: None,
            advance: ContextAdvance::None,
            final_lsrcount: None,
            final_shade: None,
        });
    }
    if kind == CopyKind::Block {
        let x_matches = if x_dec {
            requested_x1 <= walker_x0
        } else {
            requested_x1 >= walker_x0
        };
        let y_matches = if y_dec {
            requested_y1 <= walker_y0
        } else {
            requested_y1 >= walker_y0
        };
        if !x_matches || !y_matches || context.xsave != context.xstart {
            return Err("block start state");
        }
    }

    let x_visits = match kind {
        CopyKind::Block => i64::from(walker_x0).abs_diff(i64::from(requested_x1)) + 1,
        CopyKind::Span if walker_x0 <= requested_x1 => {
            i64::from(requested_x1 - walker_x0) as u64 + 1
        }
        CopyKind::Span => 1,
    };
    let y_stride = if dm0.ystride() { 2u64 } else { 1 };
    let y_visits = match kind {
        CopyKind::Block => {
            i64::from(walker_y0).abs_diff(i64::from(requested_y1)) / y_stride + 1
        }
        CopyKind::Span => 1,
    };
    let visits = x_visits
        .checked_mul(y_visits)
        .ok_or("copy length")?;
    let x_step = if x_dec { -1 } else { 1 };
    let y_step = if y_dec {
        -(y_stride as i32)
    } else {
        y_stride as i32
    };
    let walker_x1 = walker_x0 + x_step * (x_visits as i32 - 1);
    let walker_y1 = walker_y0 + y_step * (y_visits as i32 - 1);

    let mut final_shade = dm0.shade().then(|| ShadeDda::from_context(context));
    if let Some(shade) = final_shade.as_mut() {
        for _ in 0..visits {
            shade.advance();
        }
    }
    let final_lsrcount = dm0.enlspattern().then(|| {
        ls_pattern_state_after(
            context.lsmode.lsrcount(),
            context.lsmode.lsrepeat(),
            context.lsmode.lslength(),
            context.pat_bit,
            visits,
        )
        .0
    });

    let x_window = (context.xywin >> 16) as i16 as i32;
    let y_window = context.xywin as i16 as i32;
    let x_move = (context.xymove >> 16) as i16 as i32;
    let y_move = context.xymove as i16 as i32;
    let physical_y = |walker_y: i32| {
        let absolute = walker_y + y_window + y_move;
        if dm1.yflip() {
            0x23ff - absolute
        } else {
            absolute - REX3_COORD_BIAS
        }
    };
    let dst_x0 = walker_x0 + x_window + x_move - REX3_COORD_BIAS;
    let dst_x1 = walker_x1 + x_window + x_move - REX3_COORD_BIAS;
    let dst_y0 = physical_y(walker_y0);
    let dst_y1 = physical_y(walker_y1);
    let min_x = dst_x0.min(dst_x1).max(0);
    let max_x = dst_x0.max(dst_x1).min(REX3_SCREEN_WIDTH - 1);
    let min_y = dst_y0.min(dst_y1).max(0);
    let max_y = dst_y0.max(dst_y1).min(REX3_SCREEN_HEIGHT - 1);
    let suppress_all = dm0.skipfirst()
        && kind == CopyKind::Block
        || kind == CopyKind::Block && dm0.lronly() && x_dec;
    let fill = if min_x > max_x
        || min_y > max_y
        || context.wrmask == 0
        || plane.is_none()
        || suppress_all
    {
        None
    } else {
        Some(CopyFill {
            kind,
            plane: plane.unwrap(),
            min_x,
            min_y,
            max_x,
            max_y,
            walker_x0,
            walker_y0,
            walker_x1,
            walker_y1,
        })
    };
    let advance = match kind {
        CopyKind::Block => ContextAdvance::Block {
            xsave: context.xsave,
            y_after: (walker_y0 + y_step * y_visits as i32) << 11,
        },
        CopyKind::Span => ContextAdvance::Span {
            x_after: (walker_x0 + x_visits as i32) << 11,
        },
    };
    Ok(PreparedCopyFill {
        fill,
        advance,
        final_lsrcount,
        final_shade,
    })
}
