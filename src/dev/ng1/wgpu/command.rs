//! Stable plain-data records shared by Newport's resident CPU recorder and
//! raster shader.

use bytemuck::{Pod, Zeroable};

use crate::dev::ng1::rex3::{Rex3Context, HOSTRW_BUF_QWORDS};
use crate::dev::ng1::rex3_flat_fill::{FillKind, FillPlane, FlatFill};
use crate::wgpu::Full;

use super::copy::{CopyFill, CopyKind};
use super::host::{HostFill, HostKind};
use super::line::{LineCommand, LineKind};

pub(super) const COMMAND_BLOCK: u32 = 1;
pub(super) const COMMAND_SPAN: u32 = 2;
pub(super) const COMMAND_CLEAR: u32 = 3;
pub(super) const COMMAND_I_LINE: u32 = 4;
pub(super) const COMMAND_F_LINE: u32 = 5;
pub(super) const COMMAND_A_LINE: u32 = 6;
pub(super) const COMMAND_COPY_BLOCK: u32 = 7;
pub(super) const COMMAND_COPY_SPAN: u32 = 8;
pub(super) const COMMAND_COPY_I_LINE: u32 = 9;
pub(super) const COMMAND_COPY_F_LINE: u32 = 10;
pub(super) const COMMAND_COPY_A_LINE: u32 = 11;
pub(super) const COMMAND_HOST_BLOCK: u32 = 12;
pub(super) const COMMAND_HOST_SPAN: u32 = 13;

pub(super) const FLAG_WRITE_RGB: u32 = 1 << 0;
pub(super) const FLAG_WRITE_AUX: u32 = 1 << 1;
pub(super) const FLAG_READ_RGB: u32 = 1 << 2;
pub(super) const FLAG_READ_AUX: u32 = 1 << 3;
pub(super) const FLAG_BARRIER_BEFORE: u32 = 1 << 4;
pub(super) const FLAG_BARRIER_AFTER: u32 = 1 << 5;
pub(super) const FLAG_LINE_Y_MAJOR: u32 = 1 << 6;
pub(super) const FLAG_LINE_MAJOR_DECREASING: u32 = 1 << 7;
pub(super) const FLAG_SHADED: u32 = 1 << 8;
pub(super) const FLAG_FILL_FIRST_ROW_COMPLETE: u32 = 1 << 9;
pub(super) const FLAG_FILL_BLOCK: u32 = 1 << 10;

pub(super) const STATE_WORDS: usize = 64;
pub(super) const STATE_DRAW_MODE0: usize = 0;
pub(super) const STATE_DRAW_MODE1: usize = 1;
pub(super) const STATE_LINE_MODE: usize = 2;
pub(super) const STATE_CLIP_MODE: usize = 3;
pub(super) const STATE_WRITE_MASK: usize = 4;
pub(super) const STATE_XYWIN: usize = 5;
pub(super) const STATE_COLOR_BACK: usize = 10;
pub(super) const STATE_ALPHA_REF: usize = 12;
pub(super) const STATE_SMASK0X: usize = 13;
pub(super) const STATE_SMASK0Y: usize = 14;
pub(super) const STATE_SMASK1X: usize = 15;
pub(super) const STATE_COLOR_RED: usize = 23;
pub(super) const STATE_COLOR_ALPHA: usize = 24;
pub(super) const STATE_COLOR_GREEN: usize = 25;
pub(super) const STATE_COLOR_BLUE: usize = 26;
pub(super) const STATE_SLOPE_RED: usize = 27;
pub(super) const STATE_SLOPE_ALPHA: usize = 28;
pub(super) const STATE_SLOPE_GREEN: usize = 29;
pub(super) const STATE_SLOPE_BLUE: usize = 30;
pub(super) const STATE_LINE_PATTERN: usize = 7;
pub(super) const STATE_Z_PATTERN: usize = 9;
pub(super) const STATE_PATTERN_BITS: usize = 43;
pub(super) const STATE_XYMOVE: usize = 47;
pub(super) const STATE_HOST_XSTART: usize = 50;
pub(super) const STATE_HOST_YSTART: usize = 51;
pub(super) const STATE_HOST_XEND: usize = 52;
pub(super) const STATE_HOST_YEND: usize = 53;
pub(super) const STATE_HOST_XSAVE: usize = 54;
pub(super) const STATE_HOST_LENGTH: usize = 55;
pub(super) const STATE_HOST_CURSOR: usize = 56;
pub(super) const STATE_SOURCE_COLOR: usize = 63;

/// One maximum HostRW batch, expanded from qwords to shader words.
pub(super) const MAX_PAYLOAD_WORDS: usize = 2 * HOSTRW_BUF_QWORDS;
pub(super) const MAX_PAYLOAD_BYTES: u64 = MAX_PAYLOAD_WORDS as u64 * size_of::<u32>() as u64;

/// Bounds are inclusive. A zero payload length means that its offset is
/// ignored. Tile-list order is the command order for overlapping pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub(super) struct RexCommand {
    kind: u32,
    state_slot: u32,
    flags: u32,
    payload_offset: u32,
    payload_words: u32,
    dst_x0: i32,
    dst_y0: i32,
    dst_x1: i32,
    dst_y1: i32,
    src_x0: i32,
    src_y0: i32,
    src_x1: i32,
    src_y1: i32,
    data: [u32; 3],
}

impl RexCommand {
    pub(super) fn flat_fill(
        state_slot: u32,
        fill: FlatFill,
        payload: PayloadRef,
        shaded: bool,
        cid_test: bool,
        read_destination: bool,
    ) -> Self {
        let kind = match fill.kind {
            FillKind::Block => COMMAND_BLOCK,
            FillKind::Span => COMMAND_SPAN,
            FillKind::Clear => COMMAND_CLEAR,
        };
        let mut flags = match fill.plane {
            FillPlane::Rgb => FLAG_WRITE_RGB,
            FillPlane::Aux => FLAG_WRITE_AUX,
        };
        if cid_test {
            flags |= FLAG_READ_AUX;
        }
        if read_destination {
            flags |= match fill.plane {
                FillPlane::Rgb => FLAG_READ_RGB,
                FillPlane::Aux => FLAG_READ_AUX,
            };
        }
        if shaded {
            flags |= FLAG_SHADED;
        }
        if fill.first_row_complete {
            flags |= FLAG_FILL_FIRST_ROW_COMPLETE;
        }
        if fill.walker_block {
            flags |= FLAG_FILL_BLOCK;
        }
        Self {
            kind,
            state_slot,
            flags,
            payload_offset: payload.offset,
            payload_words: payload.words,
            dst_x0: fill.min_x as i32,
            dst_y0: fill.min_y as i32,
            dst_x1: (fill.min_x + fill.width - 1) as i32,
            dst_y1: (fill.min_y + fill.height - 1) as i32,
            src_x0: fill.walker_x0,
            src_y0: fill.walker_y0,
            src_x1: fill.walker_x1,
            src_y1: fill.walker_y1,
            data: [
                fill.walker_xsave as u32,
                fill.first_row_visits,
                fill.full_row_visits,
            ],
            ..Self::default()
        }
    }

    pub(super) fn line(
        state_slot: u32,
        line: LineCommand,
        payload: PayloadRef,
        cid_test: bool,
        read_destination: bool,
    ) -> Self {
        let kind = match line.kind {
            LineKind::Integer => COMMAND_I_LINE,
            LineKind::Fractional => COMMAND_F_LINE,
            LineKind::Antialiased => COMMAND_A_LINE,
        };
        let mut flags = match line.plane {
            FillPlane::Rgb => FLAG_WRITE_RGB,
            FillPlane::Aux => FLAG_WRITE_AUX,
        };
        if line.y_major {
            flags |= FLAG_LINE_Y_MAJOR;
        }

        if line.major_decreasing {
            flags |= FLAG_LINE_MAJOR_DECREASING;
        }
        if cid_test {
            flags |= FLAG_READ_AUX;
        }
        if read_destination {
            flags |= match line.plane {
                FillPlane::Rgb => FLAG_READ_RGB,
                FillPlane::Aux => FLAG_READ_AUX,
            };
        }
        if line.shaded {
            flags |= FLAG_SHADED;
        }
        Self {
            kind,
            state_slot,
            flags,
            payload_offset: payload.offset,
            payload_words: line.payload_words,
            dst_x0: line.min_x,
            dst_y0: line.min_y,
            dst_x1: line.max_x,
            dst_y1: line.max_y,
            src_x0: line.start_major,
            ..Self::default()
        }
    }

    pub(super) fn copy_fill(
        state_slot: u32,
        fill: CopyFill,
        cid_test: bool,
    ) -> Self {
        let kind = match fill.kind {
            CopyKind::Block => COMMAND_COPY_BLOCK,
            CopyKind::Span => COMMAND_COPY_SPAN,
        };
        let mut flags = FLAG_BARRIER_BEFORE | FLAG_BARRIER_AFTER;
        flags |= match fill.plane {
            FillPlane::Rgb => FLAG_READ_RGB | FLAG_WRITE_RGB,
            FillPlane::Aux => FLAG_READ_AUX | FLAG_WRITE_AUX,
        };
        if cid_test {
            flags |= FLAG_READ_AUX;
        }
        if fill.first_row_complete {
            flags |= FLAG_FILL_FIRST_ROW_COMPLETE;
        }
        if fill.kind == CopyKind::Block {
            flags |= FLAG_FILL_BLOCK;
        }
        Self {
            kind,
            state_slot,
            flags,
            dst_x0: fill.min_x,
            dst_y0: fill.min_y,
            dst_x1: fill.max_x,
            dst_y1: fill.max_y,
            src_x0: fill.walker_x0,
            src_y0: fill.walker_y0,
            src_x1: fill.walker_x1,
            src_y1: fill.walker_y1,
            data: [
                fill.walker_xsave as u32,
                fill.first_row_visits,
                fill.full_row_visits,
            ],
            ..Self::default()
        }
    }

    pub(super) fn copy_line(
        state_slot: u32,
        line: LineCommand,
        payload: PayloadRef,
        cid_test: bool,
    ) -> Self {
        let kind = match line.kind {
            LineKind::Integer => COMMAND_COPY_I_LINE,
            LineKind::Fractional => COMMAND_COPY_F_LINE,
            LineKind::Antialiased => COMMAND_COPY_A_LINE,
        };
        let mut flags = FLAG_BARRIER_BEFORE | FLAG_BARRIER_AFTER;
        flags |= match line.plane {
            FillPlane::Rgb => FLAG_READ_RGB | FLAG_WRITE_RGB,
            FillPlane::Aux => FLAG_READ_AUX | FLAG_WRITE_AUX,
        };
        if line.y_major {
            flags |= FLAG_LINE_Y_MAJOR;
        }
        if line.major_decreasing {
            flags |= FLAG_LINE_MAJOR_DECREASING;
        }
        if cid_test {
            flags |= FLAG_READ_AUX;
        }
        Self {
            kind,
            state_slot,
            flags,
            payload_offset: payload.offset,
            payload_words: line.payload_words,
            dst_x0: line.min_x,
            dst_y0: line.min_y,
            dst_x1: line.max_x,
            dst_y1: line.max_y,
            src_x0: line.start_major,
            ..Self::default()
        }
    }

    pub(super) fn host_fill(
        state_slot: u32,
        fill: HostFill,
        payload: PayloadRef,
        cid_test: bool,
        read_destination: bool,
    ) -> Self {
        let kind = match fill.kind {
            HostKind::Block => COMMAND_HOST_BLOCK,
            HostKind::Span => COMMAND_HOST_SPAN,
        };
        let mut flags = FLAG_BARRIER_BEFORE | FLAG_BARRIER_AFTER;
        flags |= match fill.plane {
            FillPlane::Rgb => FLAG_WRITE_RGB,
            FillPlane::Aux => FLAG_WRITE_AUX,
        };
        if cid_test {
            flags |= FLAG_READ_AUX;
        }
        if read_destination {
            flags |= match fill.plane {
                FillPlane::Rgb => FLAG_READ_RGB,
                FillPlane::Aux => FLAG_READ_AUX,
            };
        }
        Self {
            kind,
            state_slot,
            flags,
            payload_offset: payload.offset,
            payload_words: payload.words,
            ..Self::default()
        }
    }

    #[cfg(test)]
    fn payload(&self) -> PayloadRef {
        PayloadRef {
            offset: self.payload_offset,
            words: self.payload_words,
        }
    }
}

pub(super) fn cid_test(clipmode: u32) -> bool {
    (clipmode >> 9) & 0xf != 0xf
}

/// REX3 state used by coverage and pixel processing. Geometry and variable
/// HostRW data live in `RexCommand` and the payload arena instead.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(super) struct RexStateSlot {
    words: [u32; STATE_WORDS],
}

impl RexStateSlot {
    pub(super) fn for_flat_fill(context: &Rex3Context, fill: FlatFill) -> Self {
        Self::with_source_color(context, fill.color)
    }

    pub(super) fn for_line(context: &Rex3Context, line: LineCommand) -> Self {
        Self::with_source_color(context, line.color)
    }

    pub(super) fn for_copy(context: &Rex3Context) -> Self {
        let mut state = Self::with_source_color(context, 0);
        let mut dm0 = context.drawmode0.0;
        let mut dm1 = context.drawmode1.0;
        if dm1 & (1 << 17) != 0 {
            dm1 &= !(1 << 18);
        } else {
            dm1 &= !(1 << 16);
        }
        dm1 &= !(1 << 17);
        if dm1 & (1 << 18) != 0 {
            dm1 &= !(0xf << 28);
        } else {
            dm1 &= !((7 << 19) | (7 << 22) | (1 << 25) | (1 << 27));
        }
        if dm0 & (1 << 18) == 0 {
            dm0 &= !(1 << 21);
        }
        if dm0 & (1 << 13) == 0 {
            dm0 &= !(1 << 14);
        }
        if dm0 & (3 << 6) == 0 {
            dm1 &= !(0x1f << 7);
        }
        state.words[STATE_DRAW_MODE0] = dm0;
        state.words[STATE_DRAW_MODE1] = dm1;
        state
    }

    pub(super) fn for_host_fill(context: &Rex3Context) -> Self {
        let source = (context.get_colori() & 0x00ff_ffff)
            | (Rex3Context::clamp_color_component(context.coloralpha) << 24);
        Self::with_source_color(context, source)
    }

    fn with_source_color(context: &Rex3Context, source_color: u32) -> Self {
        let mut words = [0; STATE_WORDS];
        words[STATE_DRAW_MODE0] = context.drawmode0.0;
        words[STATE_DRAW_MODE1] = context.drawmode1.0;
        words[STATE_LINE_MODE] = context.lsmode.lsrcount()
            | context.lsmode.lsrepeat() << 8
            | context.lsmode.lsrcntsave() << 16
            | context.lsmode.lslength() << 24;
        words[STATE_CLIP_MODE] = context.clipmode;
        words[STATE_WRITE_MASK] = context.wrmask;
        words[STATE_XYWIN] = context.xywin;
        words[6] = context.topscan;
        words[STATE_LINE_PATTERN] = context.lspattern;
        words[8] = context.lspatsave;
        words[STATE_Z_PATTERN] = context.zpattern;
        words[STATE_COLOR_BACK] = context.colorback;
        words[11] = context.colorvram;
        words[STATE_ALPHA_REF] = context.alpharef;
        words[STATE_SMASK0X] = context.smask0x;
        words[STATE_SMASK0Y] = context.smask0y;
        words[STATE_SMASK1X] = context.smask1x;
        words[16] = context.smask1y;
        words[17] = context.smask2x;
        words[18] = context.smask2y;
        words[19] = context.smask3x;
        words[20] = context.smask3y;
        words[21] = context.smask4x;
        words[22] = context.smask4y;
        words[23] = context.colorred;
        words[24] = context.coloralpha;
        words[25] = context.colorgrn;
        words[26] = context.colorblue;
        words[27] = context.slopered as u32;
        words[28] = context.slopealpha as u32;
        words[29] = context.slopegrn as u32;
        words[30] = context.slopeblue as u32;
        words[31] = context.colorx;
        words[32] = context.bresd;
        words[33] = context.bress1;
        words[34] = context.bresoctinc1.incr1() | context.bresoctinc1.octant() << 24;
        words[35] = context.bresrndinc2.incr2() | context.bresrndinc2.rnd() << 24;
        words[36] = context.brese1;
        words[37] = context.bress2;
        words[38] = context.aweight0;
        words[39] = context.aweight1;
        words[40] = context.stepz;
        words[41] = context.lssave;
        words[42] = context.lsrestore;
        words[STATE_PATTERN_BITS] = context.pat_bit as u32
            | (context.zpat_bit as u32) << 8
            | (context.mid_primitive as u32) << 16;
        words[44] = context.hostcnt;
        words[45] = context.host_shifter as u32;
        words[46] = (context.host_shifter >> 32) as u32;
        words[STATE_XYMOVE] = context.xymove;
        words[48] = context.stall0;
        words[49] = context.stall1;
        words[STATE_HOST_XSTART] = context.xstart as u32;
        words[STATE_HOST_YSTART] = context.ystart as u32;
        words[STATE_HOST_XEND] = context.xend as u32;
        words[STATE_HOST_YEND] = context.yend as u32;
        words[STATE_HOST_XSAVE] = context.xsave as u32;
        words[STATE_HOST_LENGTH] = context.host_len;
        words[STATE_HOST_CURSOR] = context.host_cursor;
        words[STATE_SOURCE_COLOR] = source_color;
        Self { words }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PayloadRef {
    offset: u32,
    words: u32,
}

impl PayloadRef {
    pub(super) const EMPTY: Self = Self {
        offset: 0,
        words: 0,
    };
}

/// CPU recording for the payload buffer associated with the next reusable
/// batch set. Capacity never grows after construction.
pub(super) struct PayloadBatch {
    words: Vec<u32>,
    capacity: usize,
}

impl PayloadBatch {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            words: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub(super) fn append(&mut self, words: &[u32]) -> Result<PayloadRef, Full> {
        if words.is_empty() {
            return Ok(PayloadRef::EMPTY);
        }
        let end = self.words.len().checked_add(words.len()).ok_or(Full)?;
        if end > self.capacity || end > u32::MAX as usize {
            return Err(Full);
        }
        let reference = PayloadRef {
            offset: self.words.len() as u32,
            words: words.len() as u32,
        };
        self.words.extend_from_slice(words);
        Ok(reference)
    }

    pub(super) fn append_iter(
        &mut self,
        words: impl ExactSizeIterator<Item = u32>,
    ) -> Result<PayloadRef, Full> {
        if words.len() == 0 {
            return Ok(PayloadRef::EMPTY);
        }
        let end = self.words.len().checked_add(words.len()).ok_or(Full)?;
        if end > self.capacity || end > u32::MAX as usize {
            return Err(Full);
        }
        let reference = PayloadRef {
            offset: self.words.len() as u32,
            words: words.len() as u32,
        };
        self.words.extend(words);
        Ok(reference)
    }

    pub(super) fn checkpoint(&self) -> usize {
        self.words.len()
    }

    pub(super) fn rollback(&mut self, checkpoint: usize) {
        assert!(checkpoint <= self.words.len());
        self.words.truncate(checkpoint);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    pub(super) fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.words)
    }

    pub(super) fn clear(&mut self) {
        self.words.clear();
    }
}

const _: () = {
    assert!(size_of::<RexCommand>() == 64);
    assert!(align_of::<RexCommand>() == 4);
    assert!(size_of::<RexStateSlot>() == 256);
    assert!(align_of::<RexStateSlot>() == 4);
    assert!(MAX_PAYLOAD_WORDS <= u32::MAX as usize);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::Rex3Context;

    fn on_large_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .name("resident-command-test".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(f)
            .expect("spawn command ABI test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    fn fill() -> FlatFill {
        FlatFill {
            kind: FillKind::Clear,
            plane: FillPlane::Aux,
            min_x: 11,
            min_y: 13,
            width: 17,
            height: 19,
            walker_x0: 101,
            walker_y0: 103,
            walker_xsave: 97,
            walker_x1: 117,
            walker_y1: 121,
            walker_block: true,
            first_row_visits: 17,
            full_row_visits: 21,
            rows: 19,
            first_row_complete: true,
            color: 0x1234_5678,
            write_mask: 0xa5a5_5a5a,
        }
    }

    #[test]
    fn command_layout_is_fixed_plain_data() {
        assert_eq!(
            FLAG_WRITE_RGB
                | FLAG_WRITE_AUX
                | FLAG_READ_RGB
                | FLAG_READ_AUX
                | FLAG_BARRIER_BEFORE
                | FLAG_BARRIER_AFTER
                | FLAG_LINE_Y_MAJOR
                | FLAG_LINE_MAJOR_DECREASING
                | FLAG_SHADED
                | FLAG_FILL_FIRST_ROW_COMPLETE
                | FLAG_FILL_BLOCK,
            0x7ff,
        );
        assert_eq!(size_of::<RexCommand>(), 16 * size_of::<u32>());
        assert_eq!(std::mem::offset_of!(RexCommand, state_slot), 4);
        assert_eq!(std::mem::offset_of!(RexCommand, payload_offset), 12);
        assert_eq!(std::mem::offset_of!(RexCommand, dst_x0), 20);
        assert_eq!(std::mem::offset_of!(RexCommand, src_x0), 36);
        assert_eq!(std::mem::offset_of!(RexCommand, data), 52);
        assert_eq!(size_of::<RexStateSlot>(), STATE_WORDS * size_of::<u32>());
        assert!(size_of::<RexStateSlot>() < size_of::<Rex3Context>() / 1024);
        assert_eq!(bytemuck::bytes_of(&RexCommand::zeroed()).len(), 64);
    }

    #[test]
    fn command_and_state_capture_normalized_fill() {
        on_large_stack(|| {
            let mut context = Rex3Context::power_on_default();
            context.clipmode = 0x1020_3040;
            context.wrmask = 0xa5a5_5a5a;
            context.smask4y = 0x5060_7080;
            context.colorblue = 0x90a0_b0c0;
            context.slopeblue = -17;
            context.host_shifter = 0x0123_4567_89ab_cdef;
            context.xstart = -101;
            context.ystart = 103;
            context.xend = -107;
            context.yend = 109;
            context.xsave = -113;
            context.host_len = 17;
            context.host_cursor = 5;
            context.pat_bit = 7;
            context.zpat_bit = 9;
            context.mid_primitive = true;
            let fill = fill();
            let state = RexStateSlot::for_flat_fill(&context, fill);
            assert_eq!(state.words[3], context.clipmode);
            assert_eq!(state.words[STATE_WRITE_MASK], fill.write_mask);
            assert_eq!(state.words[22], context.smask4y);
            assert_eq!(state.words[26], context.colorblue);
            assert_eq!(state.words[30], context.slopeblue as u32);
            assert_eq!(
                (state.words[45], state.words[46]),
                (0x89ab_cdef, 0x0123_4567)
            );
            assert_eq!(state.words[43], 7 | 9 << 8 | 1 << 16);
            assert_eq!(state.words[STATE_HOST_XSTART], context.xstart as u32);
            assert_eq!(state.words[STATE_HOST_YSTART], context.ystart as u32);
            assert_eq!(state.words[STATE_HOST_XEND], context.xend as u32);
            assert_eq!(state.words[STATE_HOST_YEND], context.yend as u32);
            assert_eq!(state.words[STATE_HOST_XSAVE], context.xsave as u32);
            assert_eq!(state.words[STATE_HOST_LENGTH], context.host_len);
            assert_eq!(state.words[STATE_HOST_CURSOR], context.host_cursor);
            assert_eq!(state.words[STATE_SOURCE_COLOR], fill.color);

            let command =
                RexCommand::flat_fill(23, fill, PayloadRef::EMPTY, false, false, false);
            assert_eq!(command.kind, COMMAND_CLEAR);
            assert_eq!(command.state_slot, 23);
            assert_eq!(
                command.flags,
                FLAG_WRITE_AUX | FLAG_FILL_FIRST_ROW_COMPLETE | FLAG_FILL_BLOCK
            );
            assert_eq!(
                (
                    command.dst_x0,
                    command.dst_y0,
                    command.dst_x1,
                    command.dst_y1
                ),
                (11, 13, 27, 31),
            );
            assert_eq!(command.payload(), PayloadRef::EMPTY);
            assert_eq!(
                RexCommand::flat_fill(23, fill, PayloadRef::EMPTY, false, true, false).flags,
                FLAG_WRITE_AUX
                    | FLAG_READ_AUX
                    | FLAG_FILL_FIRST_ROW_COMPLETE
                    | FLAG_FILL_BLOCK,
            );
            let mut rgb_fill = fill;
            rgb_fill.plane = FillPlane::Rgb;
            assert_eq!(
                RexCommand::flat_fill(23, rgb_fill, PayloadRef::EMPTY, false, false, true).flags,
                FLAG_WRITE_RGB
                    | FLAG_READ_RGB
                    | FLAG_FILL_FIRST_ROW_COMPLETE
                    | FLAG_FILL_BLOCK,
            );
            assert_eq!(
                command.data,
                [
                    fill.walker_xsave as u32,
                    fill.first_row_visits,
                    fill.full_row_visits,
                ],
            );
            assert_eq!(
                (
                    command.src_x0,
                    command.src_y0,
                    command.src_x1,
                    command.src_y1
                ),
                (
                    fill.walker_x0,
                    fill.walker_y0,
                    fill.walker_x1,
                    fill.walker_y1,
                ),
            );

            let mut block = fill;
            block.kind = FillKind::Block;
            assert_eq!(
                RexCommand::flat_fill(0, block, PayloadRef::EMPTY, false, false, false).kind,
                COMMAND_BLOCK
            );
            block.kind = FillKind::Span;
            assert_eq!(
                RexCommand::flat_fill(0, block, PayloadRef::EMPTY, false, false, false).kind,
                COMMAND_SPAN
            );
            let host = HostFill {
                kind: HostKind::Block,
                plane: FillPlane::Rgb,
            };
            let host_command = RexCommand::host_fill(
                7,
                host,
                PayloadRef {
                    offset: 11,
                    words: 14,
                },
                true,
                true,
            );
            assert_eq!(host_command.kind, COMMAND_HOST_BLOCK);
            assert_eq!(host_command.state_slot, 7);
            assert_eq!(
                (host_command.payload_offset, host_command.payload_words),
                (11, 14)
            );
            assert_eq!(
                host_command.flags,
                FLAG_BARRIER_BEFORE
                    | FLAG_BARRIER_AFTER
                    | FLAG_WRITE_RGB
                    | FLAG_READ_RGB
                    | FLAG_READ_AUX,
            );
            let host = HostFill {
                kind: HostKind::Span,
                plane: FillPlane::Aux,
            };
            assert_eq!(
                RexCommand::host_fill(0, host, PayloadRef::EMPTY, false, false).kind,
                COMMAND_HOST_SPAN,
            );
        });
    }

    #[test]
    fn payload_batch_is_bounded_and_transactional() {
        let mut payload = PayloadBatch::new(4);
        let allocated = payload.words.capacity();
        assert_eq!(payload.append(&[]), Ok(PayloadRef::EMPTY));
        let checkpoint = payload.checkpoint();
        assert_eq!(
            payload.append(&[10, 11, 12]),
            Ok(PayloadRef {
                offset: 0,
                words: 3
            }),
        );
        assert_eq!(
            payload.append_iter([13].into_iter()),
            Ok(PayloadRef {
                offset: 3,
                words: 1
            })
        );
        assert_eq!(payload.append(&[14]), Err(Full));
        assert_eq!(
            payload.bytes(),
            bytemuck::cast_slice::<u32, u8>(&[10, 11, 12, 13])
        );
        payload.rollback(checkpoint);
        assert!(payload.is_empty());
        assert_eq!(payload.words.capacity(), allocated);
    }
}
