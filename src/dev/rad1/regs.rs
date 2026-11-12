//! RAD1 BAR0 register offsets, the fixed words the model matches, and the
//! status-slot layout, gathered in one place.
//!
//! The names are this project's own and describe what a register does where
//! that is known. A register whose meaning no source gives is named by its
//! offset and marked unsourced. The block is meant to be replaced one for one
//! by a generated definitions header.

// Per-channel sample positions: a read-only bank, one 32-bit word each.
pub(super) const POS_FIRST: u32 = 0x04;
pub(super) const POS_LAST: u32 = 0x2c;
/// The positions of the channels the model runs. The analog input and
/// output share the word at `REG_POS_ANALOG_B`, each in its own bit field.
pub(super) const REG_POS_AES_SUB: u32 = 0x10;
pub(super) const REG_POS_ANALOG_IN_A: u32 = 0x14;
pub(super) const REG_POS_ANALOG_B: u32 = 0x18;
pub(super) const REG_POS_ANALOG_OUT_A: u32 = 0x2c;

/// Free-running counter, read only.
pub(super) const REG_UST_COUNTER: u32 = 0x30;
/// Receiver and sync status, read only; only the status slot carries it.
/// Its sync-present polarity is unsourced, so the model reads it as zero.
pub(super) const REG_SYNC_STATUS: u32 = 0x34;
/// Counter control: the words `UST_CTRL_*` below. Unsourced beyond the
/// values the drivers write.
pub(super) const REG_UST_CONTROL: u32 = 0x40;

// Per-channel enable registers. Bit 0 starts a channel; bit 2 of the AES3
// input word starts its subcode channel.
pub(super) const REG_ADAT_IN_CTRL: u32 = 0x44;
pub(super) const REG_AES_INPUT_CTRL: u32 = 0x48;
pub(super) const REG_ANALOG_INPUT_CTRL: u32 = 0x4c;
pub(super) const REG_ADAT_OUT_CTRL: u32 = 0x50;
pub(super) const REG_AES_OUT_CTRL: u32 = 0x54;
pub(super) const REG_ANALOG_OUT_EN: u32 = 0x58;
pub(super) const REG_STATUS_DMA_CTRL: u32 = 0x5c;

/// Output control: bit 8 reads as an output mute, bit 6 enables the output
/// subcode; both names are inferred.
pub(super) const REG_OUTPUT_CTRL: u32 = 0x70;
/// Written once at start-up with a fixed word each; no source gives a
/// meaning.
pub(super) const REG_UNSOURCED_74: u32 = 0x74;
pub(super) const REG_UNSOURCED_78: u32 = 0x78;
/// Analog gain bytes (input left, input right, output left, output right) on
/// the MediaIO card; a mux word on a card without an analog section.
pub(super) const REG_ANALOG_GAIN: u32 = 0x7c;
/// Block resets: a set bit holds a block in reset, bit 8 + n a clock
/// generator n.
pub(super) const REG_BLOCK_RESET: u32 = 0x80;
/// Two general-purpose outputs. On the Octane card A is the analog output
/// relay (3 releases the mute, 2 applies it) and B the DAC mute (3 mutes, 2
/// unmutes). On the MediaIO card B is the serial data line of the mux word
/// and A its latch strobe, and neither mutes.
pub(super) const REG_GPIO_A: u32 = 0x88;
pub(super) const REG_GPIO_B: u32 = 0x8c;
/// Analog output level or mode, 2 or 3; what they select is unsourced.
pub(super) const REG_ANALOG_OUT_MODE: u32 = 0x90;

// Clock generators: a control word, a remainder word and one master-clock
// select per generator, generator 3's first.
pub(super) const REG_CLOCKGEN_CTRL: u32 = 0xa0;
pub(super) const REG_CLOCKGEN_REM: u32 = 0xa4;
pub(super) const REG_CLOCKGEN_MUX_FIRST: u32 = 0xa8;
pub(super) const REG_CLOCKGEN_MUX_LAST: u32 = 0xb4;
/// Written once at start-up with a fixed word each; no source gives a
/// meaning.
pub(super) const REG_UNSOURCED_B8: u32 = 0xb8;
pub(super) const REG_UNSOURCED_BC: u32 = 0xbc;

/// Descriptor RAM: 16 entries of three words (address low, address high,
/// control) from here.
pub(super) const DMA_BANK_FIRST: u32 = 0x400;
pub(super) const DMA_BANK_LAST: u32 = 0x528;

/// A descriptor set is three registers (address low, address high, control)
/// for set A, then the same for set B.
pub(super) type RingRegs = [u32; 6];
pub(super) const RING_STATUS: RingRegs = [0x500, 0x528, 0x504, 0x4b4, 0x4b8, 0x4bc];
pub(super) const RING_INPUT: RingRegs = [0x4d0, 0x510, 0x4d4, 0x400, 0x404, 0x408];
pub(super) const RING_OUTPUT: RingRegs = [0x4f8, 0x524, 0x4fc, 0x40c, 0x410, 0x414];
pub(super) const RING_AES_SUB: RingRegs = [0x4e0, 0x518, 0x4e4, 0x448, 0x44c, 0x450];
pub(super) const RINGS: [RingRegs; 4] = [RING_STATUS, RING_INPUT, RING_OUTPUT, RING_AES_SUB];
/// Set A's address-low register of the channels the model does not run
/// (ADAT input, AES3 input, ADAT-input subcode, ADAT output, AES3 output):
/// the driver stores a buffer address there at start-up.
pub(super) const REG_ADAT_IN_ADDR: u32 = 0x4c0;
pub(super) const REG_AES_IN_ADDR: u32 = 0x4c8;
pub(super) const REG_ADAT_SUB_ADDR: u32 = 0x4d8;
pub(super) const REG_ADAT_OUT_ADDR: u32 = 0x4e8;
pub(super) const REG_AES_OUT_ADDR: u32 = 0x4f0;

/// The registers that store what is written and read it back, in the order
/// the saved state keeps them.
pub(super) const STARTUP_OFFSETS: [u32; 26] = [
    REG_ADAT_IN_CTRL, REG_AES_INPUT_CTRL, REG_ANALOG_INPUT_CTRL, REG_ADAT_OUT_CTRL, REG_AES_OUT_CTRL,
    REG_OUTPUT_CTRL, REG_UNSOURCED_74, REG_UNSOURCED_78, REG_ANALOG_GAIN, REG_BLOCK_RESET, REG_GPIO_A,
    REG_GPIO_B, REG_ANALOG_OUT_MODE, REG_CLOCKGEN_CTRL, REG_CLOCKGEN_REM,
    REG_CLOCKGEN_MUX_FIRST, REG_CLOCKGEN_MUX_FIRST + 4, REG_CLOCKGEN_MUX_FIRST + 8, REG_CLOCKGEN_MUX_LAST, REG_UNSOURCED_B8, REG_UNSOURCED_BC,
    REG_ADAT_IN_ADDR, REG_AES_IN_ADDR, REG_ADAT_SUB_ADDR, REG_ADAT_OUT_ADDR, REG_AES_OUT_ADDR,
];
/// Enable bits of `STARTUP_OFFSETS` registers whose channel the model does
/// not run; a write that sets one is refused.
pub(super) const REFUSED_ADAT_IN: u32 = 0b101;
pub(super) const REFUSED_AES_IN: u32 = 0b1;
pub(super) const REFUSED_DIGITAL_OUT: u32 = 0b1;

// Output subcode windows: 64-bit words, stored as pairs of 32-bit ones. The
// ADAT output's two 15-word buffers leave a gap after each.
pub(super) const SUBCODE_FIRST: u32 = 0x1000;
pub(super) const SUBCODE_LAST: u32 = 0x1218;
pub(super) const SUBCODE_GAP_A: u32 = 0x1078;
pub(super) const SUBCODE_GAP_B: u32 = 0x10f8;

/// Counter control words: hold, run, stop.
pub(super) const UST_CTRL_HOLD: u32 = 0x0e;
pub(super) const UST_CTRL_RUN: u32 = 0x0a;
pub(super) const UST_CTRL_STOP: u32 = 0x02;

pub(super) const ENABLE_BIT: u32 = 1;
pub(super) const SUBCODE_ENABLE_BIT: u32 = 4;
/// Control word bits 6:3: the index of the next descriptor in the descriptor
/// RAM. The old channel-type tags are those indices.
pub(super) const CTRL_NEXT_SHIFT: u32 = 3;
pub(super) const CTRL_NEXT_MASK: u32 = 0xf;
/// Control word bit 2: end of descriptor; the transfer stops after it.
pub(super) const CTRL_EOD: u32 = 4;
pub(super) const CTRL_LEN_SHIFT: u32 = 7;
pub(super) const CTRL_LEN_MASK: u32 = 0x1ff_ffff;
pub(super) const DMA_UNIT_BYTES: u64 = 128;
/// The control words the model takes to start a channel: the driver's
/// negated length in 128-byte units with its next-descriptor index.
pub(super) const CTRL_START_ANALOG_IN: u32 = 0xffff_f000;
pub(super) const CTRL_START_AES_SUB: u32 = 0xffff_f030;
pub(super) const CTRL_START_STATUS_ONESHOT: u32 = 0xffff_fffc;
pub(super) const CTRL_START_STATUS_RING: u32 = 0xffff_c078;

/// The analog output's enable register: bit 0 starts the channel, bit 29 or
/// 30 picks generator 1 or 2. The "bit 28 + n" pattern is inferred from
/// those two; other bits are unknown.
pub(super) const OUT_SELECT_GEN1: u32 = 1 << 29;
pub(super) const OUT_SELECT_GEN2: u32 = 1 << 30;

/// The relay (`REG_GPIO_A`) and DAC (`REG_GPIO_B`) mute words the Octane
/// PROM and driver write. Other values change no mute state.
pub(super) const RELAY_RELEASE: u32 = 3;
pub(super) const RELAY_MUTE: u32 = 2;
pub(super) const DAC_MUTE: u32 = 3;
pub(super) const DAC_UNMUTE: u32 = 2;

/// Status-DMA control words: stop, a single transfer, and the continuous
/// start with the interval in bits [16:2].
pub(super) const STATUS_START: u32 = 0x40003;
pub(super) const STATUS_ONESHOT_START: u32 = 0x40001;
pub(super) const STATUS_STOP: u32 = 0x1fffc;
pub(super) const STATUS_INTERVAL_SHIFT: u32 = 2;
pub(super) const STATUS_INTERVAL_MASK: u32 = 0x7fff;
pub(super) const STATUS_INTERVAL_FIELD: u32 = STATUS_INTERVAL_MASK << STATUS_INTERVAL_SHIFT;

// Status-DMA slots in host memory: 0x80 bytes apart, at most 0x80 of them.
pub(super) const SLOT_STRIDE: u64 = 0x80;
pub(super) const SLOT_MAX: u64 = 0x80;
/// The limit register holds `SLOT_LIMIT_BASE - count * SLOT_STRIDE`.
pub(super) const SLOT_LIMIT_BASE: i64 = 0x78;
/// The mirror of registers 0x04 to 0x34 starts one word into this block.
pub(super) const SLOT_MIRROR_OFFSET: u64 = 0x40;
pub(super) const SLOT_FLAGS_OFFSET: u64 = 0x78;
pub(super) const SLOT_DONE_OFFSET: u64 = 0x7c;
/// Done, with the AES3 "no input connected" bit set.
pub(super) const SLOT_DONE_WORD: u32 = 0x8000_0000 | 0x0008_0000;
/// Where a slot carries each channel's DMA pointer.
pub(super) const SLOT_PTR_ANALOG_IN: u64 = 0x10;
pub(super) const SLOT_PTR_ANALOG_OUT: u64 = 0x38;
pub(super) const SLOT_PTR_AES_SUB: u64 = 0x20;
/// Pointer words for the channel types the model does not run: written as
/// zero.
pub(super) const SLOT_PTR_UNUSED: [u64; 5] = [0x00, 0x08, 0x18, 0x28, 0x30];
