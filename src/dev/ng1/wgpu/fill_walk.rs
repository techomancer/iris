use crate::dev::ng1::rex3::{
    Rex3Context, DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_SPAN,
    DRAWMODE0_OPCODE_DRAW, DRAWMODE0_OPCODE_SCR2SCR, DRAWMODE1_PLANES_CID,
    DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP, DRAWMODE1_PLANES_RGB,
    DRAWMODE1_PLANES_RGBA, OCTANT_XDEC, OCTANT_YDEC, REX3_COORD_BIAS,
    REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH,
};
use crate::dev::ng1::rex3_flat_fill::FillPlane;
use crate::dev::ng1::rex3_generic::ls_pattern_state_after;

const FIXED_ONE: i64 = 1 << 11;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FillOperation {
    Draw,
    Copy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FillWalkKind {
    Block,
    Span,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FillCompletion {
    xstart: i32,
    ystart: i32,
    pat_bit: u8,
    zpat_bit: u8,
    lsrcount: Option<u32>,
    mid_primitive: bool,
}

impl FillCompletion {
    pub(crate) fn apply(self, context: &mut Rex3Context) {
        context.xstart = self.xstart;
        context.ystart = self.ystart;
        context.pat_bit = self.pat_bit;
        context.zpat_bit = self.zpat_bit;
        if let Some(count) = self.lsrcount {
            context.lsmode.set_lsrcount(count);
        }
        context.mid_primitive = self.mid_primitive;
        context.host_len = 1;
        context.host_cursor = 0;
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct FillWalk {
    pub(crate) kind: FillWalkKind,
    pub(crate) plane: Option<FillPlane>,
    pub(crate) min_x: i32,
    pub(crate) min_y: i32,
    pub(crate) max_x: i32,
    pub(crate) max_y: i32,
    pub(crate) walker_x0: i32,
    pub(crate) walker_y0: i32,
    pub(crate) walker_xsave: i32,
    pub(crate) walker_x1: i32,
    pub(crate) walker_y1: i32,
    pub(crate) first_row_visits: u32,
    pub(crate) full_row_visits: u32,
    pub(crate) rows: u32,
    pub(crate) first_row_complete: bool,
    pub(crate) visits: u64,
    completion_x: i32,
    completion_y: i32,
    last_row_visits: u32,
    last_row_complete: bool,
    mid_primitive: bool,
}

impl FillWalk {
    pub(crate) fn has_bounds(self) -> bool {
        self.min_x <= self.max_x && self.min_y <= self.max_y
    }

    pub(crate) fn writes_possible(self, context: &Rex3Context) -> bool {
        if self.visits == 0
            || self.plane.is_none()
            || context.wrmask == 0
            || !self.has_bounds()
        {
            return false;
        }
        let dm0 = context.drawmode0;
        let x_dec = self.kind == FillWalkKind::Block
            && context.bresoctinc1.octant() & OCTANT_XDEC != 0;
        if self.kind == FillWalkKind::Block
            && (dm0.skipfirst() || dm0.lronly() && x_dec)
        {
            return false;
        }
        let first_suppressed = u64::from(dm0.skipfirst());
        let first_last_suppressed = u64::from(
            dm0.skiplast()
                && self.first_row_complete
                && !(dm0.skipfirst() && self.first_row_visits == 1),
        );
        let first_writes =
            u64::from(self.first_row_visits) > first_suppressed + first_last_suppressed;
        if first_writes {
            return true;
        }
        self.rows > 1 && (!dm0.skiplast() || self.full_row_visits > 1)
    }

    pub(crate) fn completion(
        self,
        context: &Rex3Context,
        z_pattern_enabled: bool,
        line_pattern_enabled: bool,
    ) -> FillCompletion {
        if self.visits == 0 {
            return FillCompletion {
                xstart: context.xstart,
                ystart: context.ystart,
                pat_bit: context.pat_bit,
                zpat_bit: context.zpat_bit,
                lsrcount: None,
                mid_primitive: context.mid_primitive,
            };
        }

        let before_last = self.visits - u64::from(self.last_row_visits);
        let row_start_bit = if self.rows == 1 { context.pat_bit } else { 31 };
        let row_start_zbit = if self.rows == 1 { context.zpat_bit } else { 31 };
        let row_start_count = if line_pattern_enabled {
            ls_pattern_state_after(
                context.lsmode.lsrcount(),
                context.lsmode.lsrepeat(),
                context.lsmode.lslength(),
                context.pat_bit,
                before_last,
            )
            .0
        } else {
            context.lsmode.lsrcount()
        };
        let (lsrcount, pat_bit) = if self.last_row_complete {
            (
                line_pattern_enabled.then(|| {
                    ls_pattern_state_after(
                        row_start_count,
                        context.lsmode.lsrepeat(),
                        context.lsmode.lslength(),
                        row_start_bit,
                        u64::from(self.last_row_visits),
                    )
                    .0
                }),
                31,
            )
        } else if line_pattern_enabled {
            let (count, bit) = ls_pattern_state_after(
                row_start_count,
                context.lsmode.lsrepeat(),
                context.lsmode.lslength(),
                row_start_bit,
                u64::from(self.last_row_visits),
            );
            (Some(count), bit)
        } else {
            (None, context.pat_bit)
        };
        let zpat_bit = if self.last_row_complete {
            31
        } else if z_pattern_enabled {
            row_start_zbit.wrapping_sub(self.last_row_visits as u8) & 31
        } else {
            context.zpat_bit
        };
        FillCompletion {
            xstart: self.completion_x,
            ystart: self.completion_y,
            pat_bit,
            zpat_bit,
            lsrcount,
            mid_primitive: self.mid_primitive,
        }
    }
}

fn visits_to_end(start: i32, end: i32, step: i64) -> u64 {
    let start = i64::from(start);
    let end = i64::from(end);
    if step > 0 {
        if start > end {
            1
        } else {
            ((end - start) / step) as u64 + 1
        }
    } else if start < end {
        1
    } else {
        ((start - end) / -step) as u64 + 1
    }
}

fn stepped(start: i32, step: i64, advances: u64) -> Result<i32, &'static str> {
    i32::try_from(i64::from(start) + step * advances as i64)
        .map_err(|_| "fill coordinates")
}

pub(crate) fn prepare_fill_walk(
    context: &Rex3Context,
    operation: FillOperation,
) -> Result<FillWalk, &'static str> {
    let dm0 = context.drawmode0;
    let dm1 = context.drawmode1;
    let kind = match dm0.adrmode() {
        DRAWMODE0_ADRMODE_BLOCK => FillWalkKind::Block,
        DRAWMODE0_ADRMODE_SPAN => FillWalkKind::Span,
        _ => return Err("primitive shape"),
    };
    let expected_opcode = match operation {
        FillOperation::Draw => DRAWMODE0_OPCODE_DRAW,
        FillOperation::Copy => DRAWMODE0_OPCODE_SCR2SCR,
    };
    if dm0.opcode() != expected_opcode {
        return Err("opcode");
    }
    if operation == FillOperation::Draw && (dm0.colorhost() || dm0.alphahost()) {
        return Err("host transfer");
    }

    let plane = match dm1.planes() {
        DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => Some(FillPlane::Rgb),
        DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID => {
            Some(FillPlane::Aux)
        }
        _ => None,
    };
    let walker_x0_raw = context.xstart;
    let walker_y0_raw = context.ystart;
    let xsave_raw = context.xsave;
    let xend_raw = context.xend;
    let yend_raw = context.yend;
    let octant = context.bresoctinc1.octant();
    let x_dec = kind == FillWalkKind::Block && octant & OCTANT_XDEC != 0;
    let y_dec = kind == FillWalkKind::Block && octant & OCTANT_YDEC != 0;

    if kind == FillWalkKind::Span && dm0.lronly() && octant & OCTANT_XDEC != 0 {
        return Ok(FillWalk {
            kind,
            plane,
            min_x: 1,
            min_y: 1,
            max_x: 0,
            max_y: 0,
            walker_x0: walker_x0_raw >> 11,
            walker_y0: walker_y0_raw >> 11,
            walker_xsave: xsave_raw >> 11,
            walker_x1: walker_x0_raw >> 11,
            walker_y1: walker_y0_raw >> 11,
            first_row_visits: 0,
            full_row_visits: 0,
            rows: 0,
            first_row_complete: false,
            visits: 0,
            completion_x: walker_x0_raw,
            completion_y: walker_y0_raw,
            last_row_visits: 0,
            last_row_complete: false,
            mid_primitive: context.mid_primitive,
        });
    }

    let x_step = if x_dec { -FIXED_ONE } else { FIXED_ONE };
    let y_stride = if dm0.ystride() { 2 } else { 1 };
    let y_step = if y_dec {
        -FIXED_ONE * i64::from(y_stride)
    } else {
        FIXED_ONE * i64::from(y_stride)
    };
    let first_to_end = visits_to_end(walker_x0_raw, xend_raw, x_step);
    let full_row_visits = visits_to_end(xsave_raw, xend_raw, x_step);
    let stop_on_word = operation == FillOperation::Copy
        && dm0.colorhost()
        && (kind == FillWalkKind::Span || !dm0.stopony());
    let host_stop = stop_on_word && context.hostcnt == 0 && context.hostrw_drained();
    let stop_on_x = dm0.stoponx() || stop_on_word;
    let length_limit = dm0.length32()
        && i64::from(xend_raw).abs_diff(i64::from(walker_x0_raw)) >> 11 >= 32;
    let first_row_visits = if !stop_on_x || host_stop {
        1
    } else if length_limit {
        first_to_end.min(32)
    } else {
        first_to_end
    };
    let first_row_complete = first_row_visits == first_to_end;

    let mut rows = 1u64;
    let mut last_row_visits = first_row_visits;
    let mut last_row_complete = first_row_complete;
    let mut completion_x = stepped(walker_x0_raw, x_step, first_row_visits)?;
    let mut completion_y = walker_y0_raw;
    let mut mid_primitive = !first_row_complete;
    if kind == FillWalkKind::Span && first_row_complete {
        mid_primitive = false;
    }
    if kind == FillWalkKind::Block && first_row_complete {
        completion_x = xsave_raw;
        completion_y = stepped(walker_y0_raw, y_step, 1)?;
        let y_done = if y_dec {
            completion_y < yend_raw
        } else {
            completion_y > yend_raw
        };
        mid_primitive = dm0.stopony() && !y_done;
        if mid_primitive && stop_on_x && !host_stop {
            let remaining_rows = visits_to_end(completion_y, yend_raw, y_step);
            rows = rows
                .checked_add(remaining_rows)
                .ok_or("fill length")?;
            completion_y = stepped(completion_y, y_step, remaining_rows)?;
            last_row_visits = full_row_visits;
            last_row_complete = true;
            mid_primitive = false;
        }
    }

    let visits = first_row_visits
        .checked_add(
            rows.saturating_sub(1)
                .checked_mul(full_row_visits)
                .ok_or("fill length")?,
        )
        .ok_or("fill length")?;
    if visits > u64::from(u32::MAX) {
        return Err("fill length");
    }
    let first_row_visits = u32::try_from(first_row_visits).map_err(|_| "fill length")?;
    let full_row_visits = u32::try_from(full_row_visits).map_err(|_| "fill length")?;
    let rows = u32::try_from(rows).map_err(|_| "fill length")?;
    let last_row_visits = u32::try_from(last_row_visits).map_err(|_| "fill length")?;
    let walker_x0 = walker_x0_raw >> 11;
    let walker_y0 = walker_y0_raw >> 11;
    let walker_xsave = xsave_raw >> 11;
    let walker_x1 = stepped(walker_x0_raw, x_step, u64::from(first_row_visits - 1))? >> 11;
    let walker_y1 = if rows == 1 {
        walker_y0
    } else {
        stepped(walker_y0_raw, y_step, u64::from(rows - 1))? >> 11
    };
    let full_x1 = stepped(xsave_raw, x_step, u64::from(full_row_visits - 1))? >> 11;

    let x_window = (context.xywin >> 16) as i16 as i32;
    let y_window = context.xywin as i16 as i32;
    let moved = operation == FillOperation::Copy || dm0.xyoffset();
    let x_move = if moved {
        (context.xymove >> 16) as i16 as i32
    } else {
        0
    };
    let y_move = if moved {
        context.xymove as i16 as i32
    } else {
        0
    };
    let later_x0 = if rows > 1 { walker_xsave } else { walker_x0 };
    let later_x1 = if rows > 1 { full_x1 } else { walker_x1 };
    let x_values = [walker_x0, walker_x1, later_x0, later_x1];
    let x_min = *x_values.iter().min().unwrap() + x_window + x_move - REX3_COORD_BIAS;
    let x_max = *x_values.iter().max().unwrap() + x_window + x_move - REX3_COORD_BIAS;
    let physical_y = |walker_y: i32| {
        let absolute = walker_y + y_window + y_move;
        if dm1.yflip() {
            0x23ff - absolute
        } else {
            absolute - REX3_COORD_BIAS
        }
    };
    let y0 = physical_y(walker_y0);
    let y1 = physical_y(walker_y1);

    Ok(FillWalk {
        kind,
        plane,
        min_x: x_min.max(0),
        min_y: y0.min(y1).max(0),
        max_x: x_max.min(REX3_SCREEN_WIDTH - 1),
        max_y: y0.max(y1).min(REX3_SCREEN_HEIGHT - 1),
        walker_x0,
        walker_y0,
        walker_xsave,
        walker_x1,
        walker_y1,
        first_row_visits,
        full_row_visits,
        rows,
        first_row_complete,
        visits,
        completion_x,
        completion_y,
        last_row_visits,
        last_row_complete,
        mid_primitive,
    })
}
