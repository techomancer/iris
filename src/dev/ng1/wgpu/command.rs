//! Stable plain-data records shared by Newport's resident CPU recorder and
//! raster shader.

use bytemuck::{Pod, Zeroable};

use crate::dev::ng1::rex3::{Rex3Context, HOSTRW_BUF_QWORDS};
use crate::dev::ng1::rex3_flat_fill::{FillPlane, FlatFill};
use crate::wgpu::Full;

pub(super) const COMMAND_FLAT_FILL: u32 = 1;

pub(super) const FLAG_WRITE_RGB: u32 = 1 << 0;
pub(super) const FLAG_WRITE_AUX: u32 = 1 << 1;
pub(super) const FLAG_READ_RGB: u32 = 1 << 2;
pub(super) const FLAG_READ_AUX: u32 = 1 << 3;
pub(super) const FLAG_BARRIER_BEFORE: u32 = 1 << 4;
pub(super) const FLAG_BARRIER_AFTER: u32 = 1 << 5;

pub(super) const STATE_WORDS: usize = 64;
pub(super) const STATE_WRITE_MASK: usize = 4;
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
    pub(super) fn flat_fill(state_slot: u32, fill: FlatFill) -> Self {
        let flags = match fill.plane {
            FillPlane::Rgb => FLAG_WRITE_RGB,
            FillPlane::Aux => FLAG_WRITE_AUX,
        };
        Self {
            kind: COMMAND_FLAT_FILL,
            state_slot,
            flags,
            dst_x0: fill.min_x as i32,
            dst_y0: fill.min_y as i32,
            dst_x1: (fill.min_x + fill.width - 1) as i32,
            dst_y1: (fill.min_y + fill.height - 1) as i32,
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

/// REX3 state used by coverage and pixel processing. Geometry and variable
/// HostRW data live in `RexCommand` and the payload arena instead.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(super) struct RexStateSlot {
    words: [u32; STATE_WORDS],
}

impl RexStateSlot {
    pub(super) fn for_flat_fill(context: &Rex3Context, fill: FlatFill) -> Self {
        let mut words = [0; STATE_WORDS];
        words[0] = context.drawmode0.0;
        words[1] = context.drawmode1.0;
        words[2] = context.lsmode.lsrcount()
            | context.lsmode.lsrepeat() << 8
            | context.lsmode.lsrcntsave() << 16
            | context.lsmode.lslength() << 24;
        words[3] = context.clipmode;
        words[STATE_WRITE_MASK] = context.wrmask;
        words[5] = context.xywin;
        words[6] = context.topscan;
        words[7] = context.lspattern;
        words[8] = context.lspatsave;
        words[9] = context.zpattern;
        words[10] = context.colorback;
        words[11] = context.colorvram;
        words[12] = context.alpharef;
        words[13] = context.smask0x;
        words[14] = context.smask0y;
        words[15] = context.smask1x;
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
        words[43] = context.pat_bit as u32
            | (context.zpat_bit as u32) << 8
            | (context.mid_primitive as u32) << 16;
        words[44] = context.hostcnt;
        words[45] = context.host_shifter as u32;
        words[46] = (context.host_shifter >> 32) as u32;
        words[47] = context.xymove;
        words[48] = context.stall0;
        words[49] = context.stall1;
        words[STATE_SOURCE_COLOR] = fill.color;
        Self { words }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PayloadRef {
    offset: u32,
    words: u32,
}

impl PayloadRef {
    const EMPTY: Self = Self {
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
            plane: FillPlane::Aux,
            min_x: 11,
            min_y: 13,
            width: 17,
            height: 19,
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
                | FLAG_BARRIER_AFTER,
            0x3f,
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
            assert_eq!(state.words[STATE_SOURCE_COLOR], fill.color);

            let command = RexCommand::flat_fill(23, fill);
            assert_eq!(command.kind, COMMAND_FLAT_FILL);
            assert_eq!(command.state_slot, 23);
            assert_eq!(command.flags, FLAG_WRITE_AUX);
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
                (
                    command.src_x0,
                    command.src_y0,
                    command.src_x1,
                    command.src_y1
                ),
                (0, 0, 0, 0),
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
            payload.append(&[13]),
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
