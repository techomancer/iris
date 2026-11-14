//! SGI Octane onboard RAD audio (PCI 10a9:0005, revision 0xc0) on Bridge
//! slot 3.
//!
//! This is the Octane variant of our RAD1 model: the register file the
//! driver's start-up programs, the free-running UST counter and its control
//! word, the DMA descriptor words, status-DMA control and the analog output
//! enable. Behaviour that no source establishes is kept as the model
//! has it, and flagged below.
//!
//! Approximations kept from the model:
//!
//! * BAR0 is 0x2000 bytes and BAR sizing is a model property, not a
//!   measurement of a physical card.
//! * The PCI class is "not defined" (0) and the card raises no interrupt:
//!   the interrupt pin is 0.
//! * The start-up words, the AES3 subcode window and the descriptor bank
//!   store what was written and return it. No reset values or masks are
//!   established beyond zero; the few writes the model refuses (see
//!   [`Rad1::write`]) are refused because no supported guest makes them.
//! * The analog input and output run at the rates their clock generators
//!   (0 and 1) are programmed for; see [`Rate`]. Until a guest programs
//!   one, its rate is 44.1 kHz: the power-on rate is unknown.
//! * The engines move frames at the sample rate. Frames are moved when
//!   something looks (a position or status read, a status completion) and,
//!   while the analog output runs with host audio enabled, by a 10 ms timer,
//!   so the output reaches the host steadily. The timers are the machine's
//!   `hptimer`, armed by `Device::start` and removed by `Device::stop`. An input ring is filled with
//!   silence: host audio input is not connected.
//! * Output plays as the model's mixer does: each ring frame is two
//!   little-endian words, left then right, each a 24-bit signed sample in
//!   bits [23:0]; the host sample is bits [23:8]. The channel order and the
//!   use of bits [23:8] are the model's inference, not confirmed.
//! * The AES3 subcode channel runs at 48000/192 Hz with empty records: no
//!   capture times them, the rate is inferred from the record layout.
//! * UST ticks every 960 ns, a selected board policy; the physical rate is
//!   unmeasured.
//! * The sample clock, UST and status schedule run on host wall time, less
//!   the time the machine was stopped; they are not tied to the emulated
//!   CPU's progress.
//! * Hold-up approximation, an emulator policy with no hardware
//!   counterpart: while continuous status runs, a completion more than one
//!   interval late means the host held up the timer thread that also
//!   delivers the clock interrupt IRIX refills the ring from, so the card's
//!   clock is set back to the completion's due time (see
//!   [`CardClock::absorb_stall`]). The clock, output rate and UST fall behind
//!   wall time by the hold-ups; on-time completions win 1/32 of an interval
//!   back each, and at most 200 ms is held back. Time the guest has read is
//!   never taken back.
//! * The relay (0x88 = 2) and DAC (0x8c = 3) mute words silence the host
//!   output while the transfer runs on. That they mute is inferred from
//!   when the PROM, IRIX and Linux write them; the power-on state is
//!   unknown and taken as not muted. The Onyx2's MediaIO card does not
//!   take them as mutes: IRIX shifts serial words out through 0x8c and 0x88
//!   there and rewrites 0x8c with 2 and 3 in turn about once a second.
//! * The counter at 0x30 starts on the control word at 0x40 on the Octane
//!   card: the values the Octane driver writes are 0x0e then 0x0a to enable
//!   and 0x02 to disable. Reading them as hold, run and stop, and taking run
//!   only after hold, is the model's; no source gives the bits' meaning. The
//!   MediaIO card's driver never writes the control word and needs the
//!   counter to advance, so there it runs from reset and a card reset does
//!   not stop it.
//! * Registers no source gives a meaning (see the `REG_UNSOURCED_*` names)
//!   store what is written and read it back.
//! * A BAR0 access the model does not recognise is logged and reads zero.
//!   The model fails the bus access instead, which the PCI bus layer here
//!   cannot express.
//!
//! Ported from hw/audio/sgi-rad1-audio.c in jamesbraid/qemu-sgi-origami.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::Mutex;
use std::time::{Duration, Instant};

use crate::config::AudioConfig;
use crate::dev::host_pcm::{CardClock, Sink, PCM_CHANNELS};
use crate::devlog::LogModule;
use crate::host_audio::{HostOut, Meter, Resampler};
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::snapshot::{field, field_bool, field_u32, field_u32s, field_u64, field_u64_vec, field_u64s, field_u8, StateTable};
use crate::traits::{Device, Saveable};
use crate::pci::{Bar, PciConfig, PciFunction, PciFunctionState};

pub const PCI_VENDOR_SGI: u16 = 0x10a9;
pub const PCI_DEVICE_RAD1: u16 = 0x0005;
pub const RAD1_REVISION: u8 = 0xc0;
/// The revision byte of the RAD1 on the Onyx2's MediaIO daughtercard: IRIX
/// lists it as "RAD revision 7.0", the Octane's onboard part as 12.0.
pub const RAD1_REVISION_MEDIAIO: u8 = 0x70;
/// "Not defined" (base class 0), as the model has it.
const PCI_CLASS_NOT_DEFINED: u32 = 0;
pub const BAR0_SIZE: u32 = 0x2000;

mod regs;
use regs::*;

const POS_WORDS: usize = ((POS_LAST - POS_FIRST) / 4 + 1) as usize;

const DMA_BANK_WORDS: usize = ((DMA_BANK_LAST - DMA_BANK_FIRST) / 4 + 1) as usize;

const SUBCODE_WORDS: usize = ((SUBCODE_LAST - SUBCODE_FIRST) / 8 + 1) as usize;

/// The descriptor RAM: 16 entries of three words from 0x400, all stored. A
/// link to an entry still all zero is in no source and stops the output (see
/// [`State::descriptor_end`]).
const DESC_RAM_ENTRY_BYTES: u32 = 12;
const DESC_RAM_ENTRIES: u32 = 16;
const DESC_RAM_LAST: u32 = DMA_BANK_FIRST + DESC_RAM_ENTRIES * DESC_RAM_ENTRY_BYTES - 4;
/// Largest ring walked; only stops a garbage control word naming gigabytes.
const DMA_MAX_BYTES: u64 = 1 << 20;

const UST_NS_PER_TICK: i64 = 960;
const NS_PER_SEC: u64 = 1_000_000_000;
/// The rate of a clock generator no guest has programmed.
const FREQ_ANALOG: u64 = 44100;
const FREQ_AES_SUB: u64 = 48000 / 192;
/// Ring bytes moved per DMA access; a whole number of frames of any type.
const CHUNK_BYTES: u64 = 1024;
/// How often the pump moves output frames to the host.
const PUMP_MS: u64 = 10;
const POS_FIELD_MASK: u32 = 0x3fff;
const UST_LATCH_MASK: u32 = 0x3_ffff;

const STATUS_CLOCK_HZ: u64 = 33_000_000;

/// A 16-bit host sample sits in bits [23:8] of its 24-bit ring sample.
const SAMPLE_SHIFT: u32 = 8;
/// The analog rings' words per frame.
const ANALOG_WORDS: usize = 2;

/// One ring word's 24-bit sample as a 16-bit host sample, bits [23:8].
fn word_to_sample(word: u32) -> i16 {
    (word >> SAMPLE_SHIFT) as i16
}

/// One 16-bit host sample as a ring word: bits [23:8], sign-extended
/// through [31:24], bits [7:0] zero.
fn sample_to_word(sample: i16) -> u32 {
    (i32::from(sample) << SAMPLE_SHIFT) as u32
}

/// Output: `ring` holds whole frames of `words_per_frame` little-endian
/// words; `pcm` receives as many stereo host frames. The first word is left,
/// the second right; further words have no host channel and are dropped.
fn frames_to_pcm(ring: &[u8], pcm: &mut [i16], words_per_frame: usize) {
    let frames = pcm.len() / PCM_CHANNELS;
    for f in 0..frames {
        for ch in 0..PCM_CHANNELS {
            let at = (f * words_per_frame + ch) * 4;
            let word = u32::from_le_bytes(ring[at..at + 4].try_into().unwrap());
            pcm[f * PCM_CHANNELS + ch] = word_to_sample(word);
        }
    }
}

/// Input: stereo host frames to ring frames of `words_per_frame` words. Left
/// fills the first word and right the second; any further word is silence.
fn pcm_to_frames(pcm: &[i16], ring: &mut [u8], words_per_frame: usize) {
    let frames = pcm.len() / PCM_CHANNELS;
    for f in 0..frames {
        for w in 0..words_per_frame {
            let word = if w < PCM_CHANNELS { sample_to_word(pcm[f * PCM_CHANNELS + w]) } else { 0 };
            let at = (f * words_per_frame + w) * 4;
            ring[at..at + 4].copy_from_slice(&word.to_le_bytes());
        }
    }
}

/// Clock generators: 0 clocks the analog input, 1 the analog output, 2 the
/// AES3 output. Generator 3 is not modelled.
const NUM_CG: usize = 4;
/// The control word's generator selects, one bit each from bit 28.
const CG_SELECT_SHIFT: u32 = 28;
/// The control word's divisor less one; bits 9 and 10 above it are flags.
const CG_DIV_MASK: u32 = 0x1ff;
/// The two master clocks a generator divides, each four times its family's
/// base rate.
const MASTER_44K: u64 = 176_400;
const MASTER_48K: u64 = 192_000;

/// A sample rate, `num / den` frames per second.
///
/// A generator divides its master clock by `div + rem / den`. The decode is
/// the public Linux RAD1 driver's (Stanislaw Skowronek's `snd_rad1_set_cg`):
/// the control word at 0xa0 holds the generator select from bit 28 and
/// `div - 1` in its low bits, the word at 0xa4 holds `rem << 16 | (0x10000 -
/// den)`, and the generator's word in 0xa8..0xb4 picks the master: bit 0 set
/// for 192 kHz, clear for 176.4 kHz. IRIX 6.5.30 programs its rates from 8
/// to 48 kHz with words of the same form, all from the 192 kHz master (the
/// tests hold the words it writes). What the flag bits 9 and 10 and the
/// other select values do is unknown; they are ignored. The new rate takes
/// effect at the control word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rate {
    num: u64,
    den: u64,
}

impl Rate {
    const fn hz(hz: u64) -> Self {
        Self { num: hz, den: 1 }
    }

    fn from_clockgen(ctrl: u32, rem_word: u32, mux: u32) -> Self {
        let master = if mux & 1 != 0 { MASTER_48K } else { MASTER_44K };
        let div = u64::from(ctrl & CG_DIV_MASK) + 1;
        let den = 0x1_0000 - u64::from(rem_word & 0xffff);
        let rem = u64::from(rem_word >> 16);
        Self { num: master * den, den: div * den + rem }
    }

    /// Whole frames the clock has produced in `ns`.
    fn frames_in(self, ns: i64) -> u64 {
        (ns as u128 * u128::from(self.num) / (u128::from(self.den) * u128::from(NS_PER_SEC))) as u64
    }

    /// When frame `frames` ends, in ns from the clock's start.
    fn ns_for(self, frames: u64) -> i64 {
        (u128::from(frames) * u128::from(self.den) * u128::from(NS_PER_SEC)).div_ceil(u128::from(self.num)) as i64
    }

    /// The nearest whole rate, for the host stream.
    fn rounded_hz(self) -> u32 {
        ((self.num + self.den / 2) / self.den) as u32
    }
}

/// Channel types, as indices into [`State::chan`]. Only the three the
/// Octane supports are modelled.
const CH_IN: usize = 0;
const CH_OUT: usize = 1;
const CH_AES_SUB: usize = 2;
const NUM_CH: usize = 3;
/// The generator the analog output uses until 0x58 names one.
const OUT_DEFAULT_CG: usize = 1;

struct ChanInfo {
    name: &'static str,
    ring: RingRegs,
    /// The next-descriptor index IRIX writes for this channel type.
    tag: u32,
    /// Bytes in one frame of the ring.
    frame_bytes: u64,
    enable_bit: u32,
    output: bool,
    /// The clock generator setting the rate; with none, `freq` is the
    /// fixed rate in Hz.
    clockgen: Option<usize>,
    freq: u64,
    /// Position register A and the shift of its frame count.
    pos_a: u32,
    pos_a_shift: u32,
    /// Position register B, its shift, and whether it is the parent
    /// channel's (so this channel leaves it alone).
    pos_b: u32,
    pos_b_shift: u32,
    pos_b_parent: bool,
    /// Offset in a status slot of this channel's DMA pointer.
    slot_ptr: u64,
}

const CHAN_NAMES: [&str; NUM_CH] = ["analog in", "analog out", "AES3-input subcode"];

const CHAN_INFO: [ChanInfo; NUM_CH] = [
    ChanInfo {
        name: CHAN_NAMES[CH_IN], ring: RING_INPUT, tag: 0, frame_bytes: 8, enable_bit: ENABLE_BIT,
        output: false, clockgen: Some(0), freq: 0, pos_a: REG_POS_ANALOG_IN_A, pos_a_shift: 18, pos_b: REG_POS_ANALOG_B,
        pos_b_shift: 18, pos_b_parent: false, slot_ptr: SLOT_PTR_ANALOG_IN,
    },
    ChanInfo {
        name: CHAN_NAMES[CH_OUT], ring: RING_OUTPUT, tag: 1, frame_bytes: 8, enable_bit: ENABLE_BIT,
        output: true, clockgen: Some(1), freq: 0, pos_a: REG_POS_ANALOG_OUT_A, pos_a_shift: 18, pos_b: REG_POS_ANALOG_B,
        pos_b_shift: 4, pos_b_parent: false, slot_ptr: SLOT_PTR_ANALOG_OUT,
    },
    ChanInfo {
        name: CHAN_NAMES[CH_AES_SUB], ring: RING_AES_SUB, tag: 6, frame_bytes: 256,
        enable_bit: SUBCODE_ENABLE_BIT, output: false, clockgen: None, freq: FREQ_AES_SUB, pos_a: REG_POS_AES_SUB,
        pos_a_shift: 4, pos_b: REG_POS_AES_SUB, pos_b_shift: 18, pos_b_parent: true, slot_ptr: SLOT_PTR_AES_SUB,
    },
];

#[derive(Default, Clone, Copy)]
struct Chan {
    running: bool,
    /// Ring latched from the working registers at the enable edge.
    base: u64,
    len: u64,
    offset: u64,
    start_ns: i64,
    frames: u64,
    /// End of descriptor: the transfer ends with this descriptor.
    eod: bool,
    /// The next-descriptor index, used when `eod` is clear.
    next: u32,
    /// The entry `next` names holds this very descriptor: loop on it.
    looping: bool,
    /// A link led to a never-programmed descriptor. What the chip does then
    /// is in no source: the channel moves nothing more.
    stalled: bool,
}

/// One decoded descriptor.
#[derive(Clone, Copy)]
struct Desc {
    base: u64,
    len: u64,
    eod: bool,
    next: u32,
    looping: bool,
}

struct State {
    startup: [u32; STARTUP_OFFSETS.len()],
    subcode: [u64; SUBCODE_WORDS],
    dma_bank: [u32; DMA_BANK_WORDS],
    sample_pos: [u32; POS_WORDS],
    analog_out_en: u32,
    /// The clock generator the analog output runs from (0x58 selects it).
    out_cg: usize,
    /// Each generator's last control and remainder words; the master select
    /// (0xa8..0xb4) can arrive after the load.
    cg_loaded: [Option<(u32, u32)>; NUM_CG],
    /// The relay (0x88) and DAC (0x8c) mutes once written. The power-on
    /// state is unknown, so before that the output is not muted.
    relay_muted: Option<bool>,
    dac_muted: Option<bool>,
    status_dma_ctrl: u32,
    status_running: bool,
    chan: [Chan; NUM_CH],
    /// Each clock generator's rate.
    clock: [Rate; NUM_CG],
    ust_count: u32,
    ust_phase: i64,
    ust_enabled: bool,
    ust_armed: bool,
    ust_last_ns: i64,
    /// Bumped whenever the status schedule is replaced, so a stale timer
    /// callback notices and ends.
    status_gen: u64,
    status_interval: u32,
    status_start_ns: i64,
    status_done: u64,
    status_cursor: u32,
    status_slots: u32,
    /// The recurring timer pumping the analog output while it runs with
    /// host audio enabled, and the one-shot for the next status completion.
    audio_timer: Option<TimerId>,
    status_timer: Option<TimerId>,
}

impl State {
    fn new(now: i64, ust_free_running: bool) -> Self {
        Self {
            startup: [0; STARTUP_OFFSETS.len()],
            subcode: [0; SUBCODE_WORDS],
            dma_bank: [0; DMA_BANK_WORDS],
            sample_pos: [0; POS_WORDS],
            analog_out_en: 0,
            out_cg: OUT_DEFAULT_CG,
            cg_loaded: [None; NUM_CG],
            relay_muted: None,
            dac_muted: None,
            status_dma_ctrl: 0,
            status_running: false,
            chan: [Chan::default(); NUM_CH],
            clock: [Rate::hz(FREQ_ANALOG); NUM_CG],
            ust_count: 0,
            ust_phase: 0,
            ust_enabled: ust_free_running,
            ust_armed: false,
            ust_last_ns: now,
            status_gen: 0,
            status_interval: 0,
            status_start_ns: 0,
            status_done: 0,
            status_cursor: 0,
            status_slots: 0,
            audio_timer: None,
            status_timer: None,
        }
    }

    /// The UST counter as it will read at `ns`, for stamping a position.
    fn ust_at(&self, ns: i64) -> u32 {
        let mut ticks = self.ust_phase;
        if self.ust_enabled {
            ticks += ns - self.ust_last_ns;
        }
        self.ust_count.wrapping_add(ticks.div_euclid(UST_NS_PER_TICK) as u32)
    }

    /// A channel's sample rate.
    fn rate(&self, idx: usize) -> Rate {
        match self.cg_of(idx) {
            Some(cg) => self.clock[cg],
            None => Rate::hz(CHAN_INFO[idx].freq),
        }
    }

    /// The clock generator pacing a channel; 0x58 chooses the output's.
    fn cg_of(&self, idx: usize) -> Option<usize> {
        if idx == CH_OUT { Some(self.out_cg) } else { CHAN_INFO[idx].clockgen }
    }

    /// Whether the guest has muted the analog output's relay or DAC.
    fn muted(&self) -> bool {
        self.relay_muted == Some(true) || self.dac_muted == Some(true)
    }

    fn pos(&mut self, reg: u32) -> &mut u32 {
        &mut self.sample_pos[((reg - POS_FIRST) / 4) as usize]
    }

    fn add_field(reg: &mut u32, shift: u32, frames: u32) {
        let mask = POS_FIELD_MASK << shift;
        let field = ((*reg >> shift).wrapping_add(frames)) & POS_FIELD_MASK;
        *reg = (*reg & !mask) | (field << shift);
    }

    /// Credit frames to a channel's position registers. Both count frames.
    /// A holds the count at its shift and, at shift 18, the low 18 bits of
    /// the UST at the frame boundary; B is a 14-bit count the driver's start
    /// polls for change.
    fn advance_pos(&mut self, idx: usize, frames: u32) {
        let info = &CHAN_INFO[idx];
        let c = self.chan[idx];
        Self::add_field(self.pos(info.pos_a), info.pos_a_shift, frames);
        if info.pos_a_shift == 18 {
            let boundary = c.start_ns + self.rate(idx).ns_for(c.frames);
            let ust = self.ust_at(boundary);
            let a = self.pos(info.pos_a);
            *a = (*a & !UST_LATCH_MASK) | (ust & UST_LATCH_MASK);
        }
        if !info.pos_b_parent {
            Self::add_field(self.pos(info.pos_b), info.pos_b_shift, frames);
        }
    }

    fn bank(&self, reg: u32) -> u32 {
        self.dma_bank[((reg - DMA_BANK_FIRST) / 4) as usize]
    }

    /// Advance the UST counter to `now`. It counts only while enabled.
    fn ust_update(&mut self, now: i64) {
        if self.ust_enabled && now > self.ust_last_ns {
            let total = now - self.ust_last_ns + self.ust_phase;
            self.ust_count = self.ust_count.wrapping_add((total / UST_NS_PER_TICK) as u32);
            self.ust_phase = total % UST_NS_PER_TICK;
        }
        self.ust_last_ns = now;
    }

    /// Whether the descriptor set `ring` is programmed for a transfer with
    /// control word `limit`: an address, the control word, and set B equal
    /// to set A. The status ring's whole-memory limit needs no set B.
    fn ring_valid(&self, ring: &RingRegs, is_status: bool, limit: u32) -> bool {
        if (self.bank(ring[0]) == 0 && self.bank(ring[1]) == 0) || self.bank(ring[2]) != limit {
            return false;
        }
        if is_status && limit == CTRL_START_STATUS_ONESHOT {
            return true;
        }
        self.bank(ring[0]) == self.bank(ring[3])
            && self.bank(ring[1]) == self.bank(ring[4])
            && self.bank(ring[5]) == limit
    }

    fn ring_running(&self, ring: &RingRegs) -> bool {
        if *ring == RING_INPUT {
            self.chan[CH_IN].running
        } else if *ring == RING_OUTPUT {
            self.chan[CH_OUT].running
        } else if *ring == RING_AES_SUB {
            self.chan[CH_AES_SUB].running
        } else {
            self.status_running
        }
    }

    /// The words of descriptor RAM entry `index`.
    fn desc_entry(&self, index: u32) -> (u32, u32, u32) {
        let at = DMA_BANK_FIRST + index * DESC_RAM_ENTRY_BYTES;
        (self.bank(at), self.bank(at + 4), self.bank(at + 8))
    }

    /// Whether the analog output's working registers hold a transfer: an
    /// address and a length.
    fn out_programmed(&self) -> bool {
        let ring = &RING_OUTPUT;
        (self.bank(ring[0]) != 0 || self.bank(ring[1]) != 0)
            && (self.bank(ring[2]) >> CTRL_LEN_SHIFT) & CTRL_LEN_MASK != 0
    }

    /// Decode a descriptor for channel `idx`, or say why it cannot run. A
    /// partial frame is refused: what hardware does with one is unknown.
    fn decode_desc(&self, idx: usize, words: (u32, u32, u32)) -> Option<Desc> {
        let info = &CHAN_INFO[idx];
        let (lo, hi, ctrl) = words;
        let base = (u64::from(hi) << 32) | u64::from(lo);
        let neg = (ctrl >> CTRL_LEN_SHIFT) & CTRL_LEN_MASK;
        let units = 0u32.wrapping_sub(neg) & CTRL_LEN_MASK;
        let mut len = u64::from(units) * DMA_UNIT_BYTES;
        let next = (ctrl >> CTRL_NEXT_SHIFT) & CTRL_NEXT_MASK;
        // Only the analog output's end-of-descriptor form is established.
        let eod = info.output && ctrl & CTRL_EOD != 0;

        if base == 0 || len == 0 {
            crate::dlog!(LogModule::Rad1, "rad1: {} enabled with no ring programmed (address {base:#x}, control {ctrl:#x})", info.name);
            return None;
        }
        if len > DMA_MAX_BYTES {
            crate::dlog!(LogModule::Rad1, "rad1: {} ring length {len:#x} exceeds the modelled cap; clamping", info.name);
            len = DMA_MAX_BYTES;
        }
        if !len.is_multiple_of(info.frame_bytes) {
            crate::dlog!(LogModule::Rad1, "rad1: {} ring length {len:#x} is not a whole number of frames", info.name);
            return None;
        }
        if !eod && !info.output && next != info.tag {
            crate::dlog!(LogModule::Rad1, "rad1: {} control word {ctrl:#x} carries tag {next}, not the channel type {}", info.name, info.tag);
        }
        let looping = info.output && !eod && self.desc_entry(next) == words;
        Some(Desc { base, len, eod, next, looping })
    }

    /// The enable edge: latch the working descriptor and start. A ring that
    /// is not programmed, or not a whole number of frames, starts nothing.
    fn chan_start(&mut self, idx: usize, now: i64) {
        let info = &CHAN_INFO[idx];
        let words = (self.bank(info.ring[0]), self.bank(info.ring[1]), self.bank(info.ring[2]));
        let Some(d) = self.decode_desc(idx, words) else { return };
        self.chan[idx] = Chan {
            running: true,
            base: d.base,
            len: d.len,
            offset: 0,
            start_ns: now,
            frames: 0,
            eod: d.eod,
            next: d.next,
            looping: d.looping,
            stalled: false,
        };
    }

    /// The channel has played its descriptor to the end; returns whether it
    /// goes on. End of descriptor stops it and clears the enable bit;
    /// otherwise the entry the index names is loaded, and a never-programmed
    /// one is reported and left standing.
    fn descriptor_end(&mut self, idx: usize) -> bool {
        let c = self.chan[idx];
        if c.eod {
            self.chan[idx].running = false;
            if idx == CH_OUT {
                self.analog_out_en &= !ENABLE_BIT;
            }
            return false;
        }
        if c.looping || idx != CH_OUT {
            self.chan[idx].offset = 0;
            return true;
        }
        let words = self.desc_entry(c.next);
        let linked = if words == (0, 0, 0) { None } else { self.decode_desc(idx, words) };
        match linked {
            Some(d) => {
                self.chan[idx] = Chan {
                    base: d.base,
                    len: d.len,
                    offset: 0,
                    eod: d.eod,
                    next: d.next,
                    looping: d.looping,
                    ..c
                };
                true
            }
            None => {
                crate::dlog!(
                    LogModule::Rad1,
                    "rad1: {} links to descriptor {} which was never programmed (unsupported); output stopped",
                    CHAN_INFO[idx].name,
                    c.next
                );
                self.chan[idx].stalled = true;
                false
            }
        }
    }
}

/// The onboard RAD audio function. The timer callbacks hold a weak
/// reference to the same state.
pub struct Rad1 {
    inner: Arc<Inner>,
}

impl Deref for Rad1 {
    type Target = Inner;
    fn deref(&self) -> &Inner {
        &self.inner
    }
}

impl Default for Rad1 {
    fn default() -> Self {
        Self::new()
    }
}

impl Rad1 {
    pub fn new() -> Self {
        Self::with_clock(CardClock::new())
    }

    /// The card on a sample clock of the caller's.
    pub(crate) fn with_clock(clock: CardClock) -> Self {
        Self { inner: Arc::new_cyclic(|me| Inner::new(me.clone(), clock, RAD1_REVISION)) }
    }

    /// The RAD1 on the Onyx2's MediaIO daughtercard. Its PCI revision byte
    /// differs from the Octane's, and 0x88 and 0x8c do not mute it; the rest
    /// of the model is the same.
    pub fn mediaio() -> Self {
        Self { inner: Arc::new_cyclic(|me| Inner::new(me.clone(), CardClock::new(), RAD1_REVISION_MEDIAIO)) }
    }
}

impl Rad1 {
    /// Name the card's state lock in the lock registry, which holds only a
    /// weak reference to it.
    pub fn register_locks(&self) {
        let me = Arc::downgrade(&self.inner);
        crate::locks::register_lock("rad1::state", move || me.upgrade().is_some_and(|i| i.state.is_locked()));
    }
}

/// Lock order: `tick_pass`, then `stopped_at` (a leaf, taken by `now`; both
/// in [`CardClock`]),
/// then `state`. Rings and status slots are moved with `state` held: the
/// DMA goes down through the Bridge and HEART, which never call back into
/// the card, and a catch-up moves at most two revolutions of a ring.
pub struct Inner {
    pci: PciFunctionState,
    state: Mutex<State>,
    /// The sample clock, its stops and hold-ups, and the timer bookkeeping
    /// shared with other audio cards; reached through `Deref`.
    clock: CardClock,
    timers: OnceLock<Arc<TimerManager>>,
    /// Where the output goes, once the card is enabled. Without a host
    /// stream wanted (tests, `[audio] output = "none"`) the output channel
    /// runs silently.
    sink: Mutex<Option<Sink>>,
    /// Opens the host stream; tests put their own ring behind it.
    open_host: fn(&Inner, &AudioConfig) -> Option<HostOut>,
    /// What the analog channels have moved and the host stream has done.
    meter: Arc<Meter>,
    /// Whether 0x88 and 0x8c mute the output: the Octane's card only.
    mute_words: bool,
    /// Whether the counter at 0x30 runs from reset, with no control write:
    /// the MediaIO card only. Its driver never writes the control word, yet
    /// requires the counter to advance.
    ust_free_running: bool,
    me: Weak<Inner>,
}

/// The host stream.
fn open_host_output(card: &Inner, cfg: &AudioConfig) -> Option<HostOut> {
    crate::host_audio::open_output(cfg, card.meter.clone(), "Octane audio")
}

impl Deref for Inner {
    type Target = CardClock;
    fn deref(&self) -> &CardClock {
        &self.clock
    }
}

impl Inner {
    fn new(me: Weak<Inner>, clock: CardClock, revision: u8) -> Self {
        let ust_free_running = revision == RAD1_REVISION_MEDIAIO;
        Self {
            pci: PciFunctionState::new(PciConfig::new(
                PCI_VENDOR_SGI,
                PCI_DEVICE_RAD1,
                PCI_CLASS_NOT_DEFINED,
                revision,
                [Bar::Mem(BAR0_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                0,
            )),
            state: Mutex::new(State::new(0, ust_free_running)),
            clock,
            timers: OnceLock::new(),
            sink: Mutex::new(None),
            open_host: open_host_output,
            meter: Meter::new(if ust_free_running { "RAD1 (MediaIO)" } else { "RAD1" }, &CHAN_NAMES[..CH_AES_SUB]),
            mute_words: revision != RAD1_REVISION_MEDIAIO,
            ust_free_running,
            me,
        }
    }

    /// Play the analog output through the host with `audio`'s settings. While
    /// the output runs, a timer keeps its frames moving between the guest's
    /// own looks.
    ///
    /// The host stream opens when the machine starts, before the guest
    /// runs, so a host slow to open or to take its first samples is ready
    /// before the guest plays.
    ///
    /// The card's clock runs on wall time: frames falling due before the
    /// host drains its 200 ms ring are dropped. A failed open is retried
    /// when the output starts and then once a second.
    pub fn enable_output(&self, audio: AudioConfig) {
        self.sink.lock().get_or_insert_with(Sink::new).host.enable(audio);
        let mut s = self.state.lock();
        self.arm_audio(&mut s);
    }

    /// Whether timers may be armed: started, on a real clock, with a
    /// manager.
    fn timers_live(&self) -> Option<&Arc<TimerManager>> {
        if !self.armed.load(Ordering::SeqCst) || !self.runs_by_itself() {
            return None;
        }
        self.timers.get()
    }

    /// Run the output pump exactly while the analog output channel runs with
    /// host audio enabled. Called with the state lock held after anything
    /// that can start or stop the channel.
    fn arm_audio(&self, s: &mut State) {
        if !(s.chan[CH_OUT].running && self.armed.load(Ordering::SeqCst)) {
            self.meter.stopped();
        }
        let want = s.chan[CH_OUT].running && self.sink.lock().as_ref().is_some_and(|sk| sk.host.wanted());
        match (want, s.audio_timer) {
            (true, None) => {
                let Some(tm) = self.timers_live() else { return };
                let me = self.me.clone();
                let period = Duration::from_millis(PUMP_MS);
                s.audio_timer = Some(tm.add_recurring(Instant::now() + period, period, (), move |_| {
                    let Some(i) = me.upgrade() else { return TimerReturn::Delete };
                    i.pump()
                }));
            }
            (false, Some(id)) => {
                s.audio_timer = None;
                if let Some(tm) = self.timers.get() {
                    tm.remove(id);
                }
            }
            _ => {}
        }
    }

    /// One tick of the output timer: move the frames the sample clock has
    /// produced, and end the timer when the channel has stopped.
    fn pump(&self) -> TimerReturn {
        let _pass = self.tick_pass.lock();
        if self.stopped() {
            return TimerReturn::Continue;
        }
        // Status completions move the output while they run; a pump read
        // inside a hold-up would keep the late completion from absorbing it.
        let now = (!self.state.lock().status_running).then(|| self.now());
        let mut s = self.state.lock();
        if let Some(now) = now {
            self.sync_chan(&mut s, CH_OUT, now);
        }
        if s.chan[CH_OUT].running {
            if self.sink.lock().as_ref().is_some_and(|sk| sk.host.retry_due(Instant::now())) {
                self.open_sink(false);
            }
            TimerReturn::Continue
        } else {
            self.meter.stopped();
            s.audio_timer = None;
            TimerReturn::Delete
        }
    }

    /// Give the card the machine's timers. Called before `Device::start`.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timers.set(tm);
    }

    /// Open the host stream when output is enabled, and retry a failed open
    /// so an early sound is not lost for the run. `force` skips the wait
    /// between tries.
    fn open_sink(&self, force: bool) {
        let mut sink = self.sink.lock();
        let Some(sk) = sink.as_mut() else { return };
        let now = Instant::now();
        if let Some(opened) = sk.host.open(now, force, |cfg| (self.open_host)(self, cfg), Resampler::following) {
            crate::dlog!(LogModule::Rad1, "rad1: host audio open took {:.1} ms ({})", now.elapsed().as_secs_f64() * 1e3, if opened { "open" } else { "failed" });
        }
    }

    /// Stop the card's clock and hold the timers off guest memory while the
    /// machine is stopped. Once this returns no timer callback is under way.
    fn pause(&self) {
        self.stop_clock();
        self.disarm();
        drop(self.tick_pass.lock());
    }

    /// Run the clock again from where it stopped. The first start opens the
    /// host stream (see [`Inner::enable_output`]).
    fn resume(&self) {
        self.open_sink(false);
        self.start_clock();
        self.armed.store(true, Ordering::SeqCst);
        let mut s = self.state.lock();
        self.arm_audio(&mut s);
        if s.status_running {
            self.arm_status(&mut s);
        }
    }

    /// Remove both timers; they come back with [`Inner::resume`].
    fn disarm(&self) {
        self.armed.store(false, Ordering::SeqCst);
        self.meter.stopped();
        let mut s = self.state.lock();
        for id in [s.audio_timer.take(), s.status_timer.take()].into_iter().flatten() {
            if let Some(tm) = self.timers.get() {
                tm.remove(id);
            }
        }
    }

    /// Back to the power-on state.
    pub fn reset(&self) {
        let now = self.now();
        let mut s = self.state.lock();
        let gen = s.status_gen + 1;
        let (audio, status) = (s.audio_timer.take(), s.status_timer.take());
        for id in [audio, status].into_iter().flatten() {
            if let Some(tm) = self.timers.get() {
                tm.remove(id);
            }
        }
        *s = State::new(now, self.ust_free_running);
        s.status_gen = gen;
        self.meter.stopped();
    }

    /// The analog channels' frames go to the meter; the subcode channel's
    /// are not audio.
    fn moved(&self, idx: usize, frames: u64) {
        if idx != CH_AES_SUB {
            self.meter.moved(idx, frames);
        }
    }

    fn startup_index(off: u32) -> Option<usize> {
        STARTUP_OFFSETS.iter().position(|&o| o == off)
    }

    fn is_descriptor(off: u32) -> bool {
        RINGS.iter().any(|r| r.contains(&off)) || (DMA_BANK_FIRST..=DESC_RAM_LAST).contains(&off)
    }

    fn subcode_access(off: u32, width: usize) -> bool {
        let word = off & !7;
        // The model also takes 8-byte accesses; this bus layer's widest is 4.
        width == 4
            && off & 3 == 0
            && (SUBCODE_FIRST..=SUBCODE_LAST).contains(&word)
            && word != SUBCODE_GAP_A
            && word != SUBCODE_GAP_B
    }

    fn read(&self, off: u32, width: usize) -> u32 {
        let now = self.now();
        let mut s = self.state.lock();
        if width == 4 {
            if let Some(i) = Self::startup_index(off) {
                return s.startup[i];
            }
        }
        if Self::subcode_access(off, width) {
            let w = s.subcode[((off - SUBCODE_FIRST) / 8) as usize];
            return (w >> ((off & 4) * 8)) as u32;
        }
        if width == 4 && off & 3 == 0 {
            if Self::is_descriptor(off) {
                return s.bank(off);
            }
            if (POS_FIRST..=POS_LAST).contains(&off) {
                self.sync_all(&mut s, now);
                return s.sample_pos[((off - POS_FIRST) / 4) as usize];
            }
            if off == REG_ANALOG_OUT_EN {
                // The enable bit clears when a one-shot transfer ends.
                self.sync_chan(&mut s, CH_OUT, now);
                return s.analog_out_en;
            }
            if off == REG_STATUS_DMA_CTRL {
                return s.status_dma_ctrl;
            }
        }
        if off == REG_UST_COUNTER && width == 4 {
            s.ust_update(now);
            return s.ust_count;
        }
        crate::dlog!(LogModule::Rad1, "rad1: unsupported {width}-byte BAR0 read at {off:#x}");
        0
    }

    /// A write the model does not support is logged and dropped. The model
    /// fails the bus cycle; this bus layer cannot, so a refused write leaves
    /// the register unchanged.
    fn write(&self, off: u32, val: u32, width: usize) {
        let now = self.now();
        let mut s = self.state.lock();
        if self.write_locked(&mut s, off, val, width, now).is_none() {
            crate::dlog!(LogModule::Rad1, "rad1: unsupported {width}-byte BAR0 write at {off:#x} value {val:#x}");
        }
        self.arm_audio(&mut s);
    }

    /// `None` when the write is unsupported.
    fn write_locked(&self, s: &mut State, off: u32, val: u32, width: usize, now: i64) -> Option<()> {
        if width == 4 {
            if let Some(i) = Self::startup_index(off) {
                let refused = match off {
                    REG_ADAT_IN_CTRL => REFUSED_ADAT_IN,
                    REG_AES_INPUT_CTRL => REFUSED_AES_IN,
                    REG_ADAT_OUT_CTRL | REG_AES_OUT_CTRL => REFUSED_DIGITAL_OUT,
                    _ => 0,
                };
                if val & refused != 0 {
                    return None;
                }
                if off == REG_ANALOG_INPUT_CTRL {
                    // The analog input's interface word: bit 0 runs it.
                    if val & ENABLE_BIT != 0
                        && !s.chan[CH_IN].running
                        && !s.ring_valid(&RING_INPUT, false, CTRL_START_ANALOG_IN)
                    {
                        return None;
                    }
                    self.enable_write(s, CH_IN, val, now);
                } else if off == REG_AES_INPUT_CTRL {
                    // AES3 input control: bit 2 runs the input subcode channel.
                    if val & SUBCODE_ENABLE_BIT != 0
                        && !s.chan[CH_AES_SUB].running
                        && !s.ring_valid(&RING_AES_SUB, false, CTRL_START_AES_SUB)
                    {
                        return None;
                    }
                    self.enable_write(s, CH_AES_SUB, val, now);
                }
                s.startup[i] = val;
                match off {
                    REG_CLOCKGEN_CTRL => self.clockgen_write(s, val, now),
                    REG_CLOCKGEN_MUX_FIRST..=REG_CLOCKGEN_MUX_LAST => {
                        let cg = NUM_CG - 1 - ((off - REG_CLOCKGEN_MUX_FIRST) / 4) as usize;
                        self.clockgen_reselect(s, cg, now);
                    }
                    REG_GPIO_A if self.mute_words => match val {
                        RELAY_RELEASE => s.relay_muted = Some(false),
                        RELAY_MUTE => s.relay_muted = Some(true),
                        _ => {}
                    },
                    REG_GPIO_B if self.mute_words => match val {
                        DAC_MUTE => s.dac_muted = Some(true),
                        DAC_UNMUTE => s.dac_muted = Some(false),
                        _ => {}
                    },
                    _ => {}
                }
                return Some(());
            }
        }
        if Self::subcode_access(off, width) {
            let word = &mut s.subcode[((off - SUBCODE_FIRST) / 8) as usize];
            let shift = (off & 4) * 8;
            *word = (*word & !(0xffff_ffffu64 << shift)) | (u64::from(val) << shift);
            return Some(());
        }
        if width == 4 && off & 3 == 0 {
            if let Some(ring) = RINGS.iter().find(|r| r.contains(&off)) {
                if !s.ring_running(ring) {
                    s.dma_bank[((off - DMA_BANK_FIRST) / 4) as usize] = val;
                    return Some(());
                }
            } else if (DMA_BANK_FIRST..=DESC_RAM_LAST).contains(&off) {
                // A descriptor RAM entry no supported ring's registers
                // cover: stored for a link to load.
                s.dma_bank[((off - DMA_BANK_FIRST) / 4) as usize] = val;
                return Some(());
            } else if off == REG_ANALOG_OUT_EN {
                return self.analog_out_write(s, val, now);
            } else if off == REG_STATUS_DMA_CTRL {
                self.status_ctrl_write(s, val, now)?;
                s.status_dma_ctrl = val;
                return Some(());
            }
        }
        // The values are the drivers'; hold, run and stop are the model's
        // reading of them (see the module notes).
        if off == REG_UST_CONTROL && width == 4 {
            self.sync_all(s, now);
            match val {
                UST_CTRL_HOLD => {
                    s.ust_update(now);
                    s.ust_enabled = false;
                    s.ust_armed = true;
                    return Some(());
                }
                UST_CTRL_RUN if s.ust_armed => {
                    s.ust_update(now);
                    s.ust_enabled = true;
                    s.ust_armed = false;
                    return Some(());
                }
                UST_CTRL_STOP => {
                    s.ust_update(now);
                    s.ust_enabled = false;
                    s.ust_armed = false;
                    return Some(());
                }
                _ => {}
            }
        }
        None
    }

    /// A clock generator control word: each generator it selects takes the
    /// rate the remainder and master-select words describe.
    fn clockgen_write(&self, s: &mut State, ctrl: u32, now: i64) {
        let rem = s.startup[Self::startup_index(REG_CLOCKGEN_REM).unwrap()];
        for cg in (0..NUM_CG).filter(|cg| ctrl & (1 << (CG_SELECT_SHIFT + *cg as u32)) != 0) {
            s.cg_loaded[cg] = Some((ctrl, rem));
            self.clockgen_apply(s, cg, ctrl, rem, now);
        }
    }

    /// A master-clock select written after its generator was loaded (the
    /// Octane PROM's order) re-evaluates it with the words it was loaded with.
    fn clockgen_reselect(&self, s: &mut State, cg: usize, now: i64) {
        if let Some((ctrl, rem)) = s.cg_loaded[cg] {
            self.clockgen_apply(s, cg, ctrl, rem, now);
        }
    }

    /// Set generator `cg`'s rate from its words. A channel that runs from
    /// it is brought up to `now` at the old rate first, and counts on from
    /// there at the new one.
    fn clockgen_apply(&self, s: &mut State, cg: usize, ctrl: u32, rem: u32, now: i64) {
        let mux_reg = REG_CLOCKGEN_MUX_FIRST + 4 * (NUM_CG - 1 - cg) as u32;
        let mux = s.startup[Self::startup_index(mux_reg).unwrap()];
        let rate = Rate::from_clockgen(ctrl, rem, mux);
        if rate == s.clock[cg] {
            return;
        }
        for idx in 0..NUM_CH {
            if s.cg_of(idx) != Some(cg) {
                continue;
            }
            self.sync_chan(s, idx, now);
            let c = &mut s.chan[idx];
            if c.running {
                c.start_ns = now;
                c.frames = 0;
            }
        }
        crate::dlog!(LogModule::Rad1, "rad1: clock generator {cg} at {:.3} Hz", rate.num as f64 / rate.den as f64);
        s.clock[cg] = rate;
    }

    /// A write to the analog output's enable register: bit 0 runs the
    /// channel, one select bit names its clock generator. Other values are
    /// unsupported and leave the register as it was.
    fn analog_out_write(&self, s: &mut State, val: u32, now: i64) -> Option<()> {
        let cg = match val & !ENABLE_BIT {
            OUT_SELECT_GEN1 => 1,
            OUT_SELECT_GEN2 => 2,
            _ => return None,
        };
        if val & ENABLE_BIT != 0 && !s.chan[CH_OUT].running && !s.out_programmed() {
            return None;
        }
        if cg != s.out_cg {
            self.sync_chan(s, CH_OUT, now);
            s.out_cg = cg;
            let c = &mut s.chan[CH_OUT];
            if c.running {
                c.start_ns = now;
                c.frames = 0;
            }
        }
        self.enable_write(s, CH_OUT, val, now);
        s.analog_out_en = val;
        Some(())
    }

    /// The status-DMA control word: stop, one-shot start, or ring start.
    fn status_ctrl_write(&self, s: &mut State, val: u32, now: i64) -> Option<()> {
        if val == STATUS_STOP {
            self.status_stop(s);
        } else if val == STATUS_ONESHOT_START
            && !s.status_running
            && s.ring_valid(&RING_STATUS, true, CTRL_START_STATUS_ONESHOT)
        {
            // One completion into the ring's base slot, as a continuous
            // completion would fill it.
            let base = Self::status_base(s);
            self.fill_slot(s, base, now);
        } else if val & !STATUS_INTERVAL_FIELD == STATUS_START
            && val & STATUS_INTERVAL_FIELD != 0
            && !s.status_running
            && s.ring_valid(&RING_STATUS, true, CTRL_START_STATUS_RING)
        {
            self.status_start(s, val, now);
        } else {
            return None;
        }
        Some(())
    }

    fn status_stop(&self, s: &mut State) {
        s.status_running = false;
        s.status_gen += 1;
        self.cancel_status(s);
    }

    fn cancel_status(&self, s: &mut State) {
        if let (Some(id), Some(tm)) = (s.status_timer.take(), self.timers.get()) {
            tm.remove(id);
        }
    }

    fn status_base(s: &State) -> u64 {
        (u64::from(s.bank(RING_STATUS[1])) << 32) | u64::from(s.bank(RING_STATUS[0]))
    }

    /// The ring's slot count, recovered from the limit register: the
    /// driver encodes it as `limit = 0x78 - count * 0x80`. A limit that does
    /// not decode to a whole ring falls back to one slot.
    fn status_slots(s: &State) -> u32 {
        let span = SLOT_LIMIT_BASE - i64::from(s.bank(RING_STATUS[2]) as i32);
        if span >= SLOT_STRIDE as i64 {
            ((span as u64) / SLOT_STRIDE).min(SLOT_MAX) as u32
        } else {
            1
        }
    }

    /// Start the continuous engine. The consumer index starts at 0 and the
    /// cursor one past it, the first slot the driver checks; the first
    /// completion is one interval out. A zero interval names no period and
    /// stops the engine. The model has no floor on the interval.
    fn status_start(&self, s: &mut State, ctrl: u32, now: i64) {
        let interval = (ctrl >> STATUS_INTERVAL_SHIFT) & STATUS_INTERVAL_MASK;
        s.status_gen += 1;
        if interval == 0 {
            crate::dlog!(LogModule::Rad1, "rad1: continuous status DMA started with a zero interval (control {ctrl:#x}); stopping");
            s.status_running = false;
            self.cancel_status(s);
            return;
        }
        s.status_slots = Self::status_slots(s);
        s.status_cursor = if s.status_slots > 1 { 1 } else { 0 };
        s.status_interval = interval;
        s.status_start_ns = now;
        s.status_done = 0;
        s.status_running = true;
        self.arm_status(s);
    }

    /// Arm the one-shot for the next completion of the running schedule.
    /// It reschedules itself from each completion, so one is pending at a
    /// time; a replaced schedule's callback ends by its generation.
    fn arm_status(&self, s: &mut State) {
        self.cancel_status(s);
        let Some(tm) = self.timers_live() else { return };
        let gen = s.status_gen;
        let wait = Self::status_deadline(s, s.status_done + 1) - self.now();
        let me = self.me.clone();
        s.status_timer = Some(tm.add_one_shot(Duration::from_nanos(wait.max(0) as u64), (), move |_| {
            let Some(i) = me.upgrade() else { return TimerReturn::Delete };
            i.status_timer_tick(gen)
        }));
    }

    /// The status timer fired: complete one slot and wait for the next.
    fn status_timer_tick(&self, gen: u64) -> TimerReturn {
        let _pass = self.tick_pass.lock();
        if self.stopped() {
            return TimerReturn::Delete;
        }
        match self.status_tick(gen) {
            Some(next) => TimerReturn::RescheduleOneShot(Duration::from_nanos((next - self.now()).max(0) as u64)),
            None => TimerReturn::Delete,
        }
    }

    /// When completion `k` of the current schedule is due.
    fn status_deadline(s: &State, k: u64) -> i64 {
        s.status_start_ns
            + (u128::from(k) * u128::from(s.status_interval) * u128::from(NS_PER_SEC) / u128::from(STATUS_CLOCK_HZ)) as i64
    }

    /// One completion: fill the next slot and return the next deadline, or
    /// `None` when this schedule is over. The cursor advances in ring order
    /// with wrap, so the driver's servicer, which consumes consecutive done
    /// slots, keeps finding fresh ones.
    fn status_tick(&self, gen: u64) -> Option<i64> {
        let (due, interval) = {
            let s = self.state.lock();
            if s.status_gen != gen || !s.status_running {
                return None;
            }
            (Self::status_deadline(&s, s.status_done + 1), Self::status_deadline(&s, 1) - s.status_start_ns)
        };
        let now = self.absorb_stall(due, interval);
        let mut s = self.state.lock();
        if s.status_gen != gen || !s.status_running {
            return None;
        }
        let slot = Self::status_base(&s).checked_add(u64::from(s.status_cursor) * SLOT_STRIDE);
        if !slot.is_some_and(|slot| self.fill_slot(&mut s, slot, now)) {
            // Ring base unmapped: stop rather than spin on a failing DMA.
            s.status_running = false;
            s.status_timer = None;
            return None;
        }
        if s.status_slots != 0 {
            s.status_cursor = (s.status_cursor + 1) % s.status_slots;
        }
        s.status_done += 1;
        // Running behind: restart the schedule rather than fire in a burst.
        if now > Self::status_deadline(&s, s.status_done + u64::from(s.status_slots)) {
            s.status_start_ns = now;
            s.status_done = 0;
        }
        Some(Self::status_deadline(&s, s.status_done + 1))
    }

    /// Fill one status slot, as a completion. Every word is written
    /// little-endian through the card's own DMA path; the Bridge applies the
    /// byte reversal into big-endian memory.
    ///
    /// * +0x44..: registers 0x04..0x34 (the positions, the UST counter and
    ///   the sync status) as they read now, the engines brought up to date
    ///   first.
    /// * each channel's DMA pointer (zero for one not running or unmodelled).
    /// * the flags word, then done last, so the driver never sees a done slot
    ///   with a stale body. No digital input or video reference is connected.
    fn fill_slot(&self, s: &mut State, slot: u64, now: i64) -> bool {
        let Some(dma) = self.pci.dma() else { return false };
        self.sync_all(s, now);

        let mut mirror = Vec::with_capacity(13 * 4);
        for reg in (POS_FIRST..=REG_SYNC_STATUS).step_by(4) {
            let v = match reg {
                POS_FIRST..=POS_LAST => s.sample_pos[((reg - POS_FIRST) / 4) as usize],
                REG_UST_COUNTER => {
                    s.ust_update(now);
                    s.ust_count
                }
                _ => 0, // sync status
            };
            mirror.extend_from_slice(&v.to_le_bytes());
        }
        // The slot address is the guest's: a word past the top of the bus
        // is a failed transfer, like an unmapped one.
        let put = |off: u64, data: &[u8]| slot.checked_add(off).is_some_and(|a| dma.dma_write(a, data).is_ok());
        let ok = put(SLOT_MIRROR_OFFSET + u64::from(POS_FIRST), &mirror);
        let mut ptrs: Vec<(u64, u32)> = SLOT_PTR_UNUSED.iter().map(|&o| (o, 0)).collect();
        for (i, info) in CHAN_INFO.iter().enumerate() {
            let c = &s.chan[i];
            // The slot carries the pointer's low 32 bits.
            ptrs.push((info.slot_ptr, if c.len != 0 { c.base.wrapping_add(c.offset) as u32 } else { 0 }));
        }
        let ok = ok
            && ptrs.iter().all(|&(o, v)| put(o, &v.to_le_bytes()))
            && put(SLOT_FLAGS_OFFSET, &0u32.to_le_bytes())
            && put(SLOT_DONE_OFFSET, &SLOT_DONE_WORD.to_le_bytes());
        if !ok {
            crate::dlog!(LogModule::Rad1, "rad1: status DMA at {slot:#x} failed");
        }
        crate::pf_instant!(crate::perfetto::events::RAD_STATUS, s.status_cursor);
        crate::pf_counter!(crate::perfetto::events::RAD_STATUS_UNREAD, self.status_unread(s));
        ok
    }

    /// For the trace: the ring's slots whose done bit (bit 31 of the done
    /// word) the driver has not yet cleared; it consumes a slot by clearing
    /// that bit.
    #[cfg(feature = "perfetto")]
    fn status_unread(&self, s: &State) -> u32 {
        let Some(dma) = self.pci.dma() else { return 0 };
        let base = Self::status_base(s);
        (0..s.status_slots)
            .filter(|&i| {
                let mut w = [0u8; 4];
                base.checked_add(u64::from(i) * SLOT_STRIDE + SLOT_DONE_OFFSET).is_some_and(|a| dma.dma_read(a, &mut w).is_ok()) && u32::from_le_bytes(w) & 0x8000_0000 != 0
            })
            .count() as u32
    }

    /// A write to a channel's enable register. Its own bit starts and stops
    /// it; the register's other bits never do.
    fn enable_write(&self, s: &mut State, idx: usize, val: u32, now: i64) {
        if val & CHAN_INFO[idx].enable_bit != 0 {
            if !s.chan[idx].running {
                s.chan_start(idx, now);
                if idx == CH_OUT && s.chan[idx].running {
                    self.open_sink(true);
                }
            }
        } else {
            // Stop where the sample clock has got to.
            self.sync_chan(s, idx, now);
            s.chan[idx].running = false;
        }
    }

    /// Move frames between the ring and the host, wrapping at the ring's end.
    /// Output frames are decoded and played (or dropped with no sink); input
    /// frames are silence.
    fn chan_move(&self, s: &mut State, idx: usize, mut frames: u64) -> bool {
        let info = &CHAN_INFO[idx];
        let Some(dma) = self.pci.dma() else { return false };
        // A position read after a long look-away would replay every
        // revolution of the ring it missed, a DMA per chunk on the CPU
        // thread. Past one revolution the ring is left as it was found (an
        // output ring is only read, an input ring is already silence), so
        // whole revolutions beyond that are credited without moving
        // anything. The position fields are 14 bits, so the count's
        // truncation to 32 bits loses nothing.
        let ring_frames = s.chan[idx].len / info.frame_bytes;
        let repeats = idx != CH_OUT || s.chan[idx].looping;
        if repeats && ring_frames != 0 && frames >= 2 * ring_frames {
            let skip = (frames / ring_frames - 1) * ring_frames;
            s.chan[idx].frames += skip;
            self.moved(idx, skip);
            s.advance_pos(idx, skip as u32);
            frames -= skip;
        }
        let mut ring = [0u8; CHUNK_BYTES as usize];
        let mut pcm = [0i16; CHUNK_BYTES as usize / 4];
        while frames != 0 {
            let c = s.chan[idx];
            if !c.running || c.stalled {
                break;
            }
            let bytes = (frames * info.frame_bytes).min(c.len - c.offset);
            let chunk = bytes.min(CHUNK_BYTES);
            let n = chunk / info.frame_bytes;
            let buf = &mut ring[..(n * info.frame_bytes) as usize];
            // A ring the guest based at the top of the bus fails as an
            // unmapped one does.
            let addr = c.base.checked_add(c.offset);
            let ok = addr.is_some_and(|a| {
                if info.output {
                    dma.dma_read(a, buf).is_ok()
                } else if idx == CH_IN {
                    pcm.fill(0); // host input is not connected
                    pcm_to_frames(&pcm[..n as usize * PCM_CHANNELS], buf, ANALOG_WORDS);
                    dma.dma_write(a, buf).is_ok()
                } else {
                    buf.fill(0);
                    dma.dma_write(a, buf).is_ok()
                }
            });
            if !ok {
                crate::dlog!(LogModule::Rad1, "rad1: {} DMA at {:#x} + {:#x} failed", info.name, c.base, c.offset);
                return false;
            }
            if info.output {
                let samples = &mut pcm[..n as usize * PCM_CHANNELS];
                frames_to_pcm(buf, samples, ANALOG_WORDS);
                if let Some(sink) = self.sink.lock().as_mut() {
                    sink.play(samples, s.rate(idx).rounded_hz(), s.muted());
                }
            }
            let c = &mut s.chan[idx];
            c.offset += chunk;
            c.frames += n;
            let done = c.offset >= c.len;
            self.moved(idx, n);
            s.advance_pos(idx, n as u32);
            frames -= n;
            if done && !s.descriptor_end(idx) {
                break;
            }
        }
        true
    }

    /// Bring a channel up to the present: the frames due are those the
    /// sample clock has produced since the enable edge.
    fn sync_chan(&self, s: &mut State, idx: usize, now: i64) {
        let c = s.chan[idx];
        if !c.running || now <= c.start_ns {
            return;
        }
        let due = s.rate(idx).frames_in(now - c.start_ns);
        if due > c.frames && !self.chan_move(s, idx, due - c.frames) {
            s.chan[idx].running = false;
        }
    }

    fn sync_all(&self, s: &mut State, now: i64) {
        for idx in 0..NUM_CH {
            self.sync_chan(s, idx, now);
        }
    }
}

impl Device for Rad1 {
    fn step(&self, _cycles: u64) {}
    fn start(&self) { self.inner.resume() }
    fn stop(&self) { self.inner.pause() }
    fn is_running(&self) -> bool { self.inner.armed.load(Ordering::SeqCst) }
    fn get_clock(&self) -> u64 { 0 }

    /// The counts are as of the last pass of the output timer: a query
    /// moves nothing.
    fn audio_report(&self) -> Option<String> {
        let i = &self.inner;
        let rates = {
            let s = i.state.lock();
            [CH_IN, CH_OUT].map(|idx| s.chan[idx].running.then(|| s.rate(idx).rounded_hz()))
        };
        let sink = i.sink.lock();
        let link = sink.as_ref().map(|sk| &sk.host);
        Some(i.meter.report(&rates, link.and_then(|l| l.out()), link.is_some_and(|l| l.wanted())))
    }
}

impl PciFunction for Rad1 {
    /// The card masters the bus for its status slots and audio rings. It
    /// raises no interrupt, so its state never has a line.
    fn state(&self) -> &PciFunctionState {
        &self.inner.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        self.inner.read(off, width)
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.inner.write(off, val, width)
    }

    fn reset(&self) {
        self.inner.reset();
    }
}

impl State {
    /// Refuse a loaded state the card would divide by, index with or count
    /// down from out of range.
    fn check(&self) -> Result<(), String> {
        for (g, r) in self.clock.iter().enumerate() {
            if r.num == 0 || r.den == 0 {
                return Err(format!("clock: generator {g} has rate {}/{}", r.num, r.den));
            }
        }
        for (info, c) in CHAN_INFO.iter().zip(&self.chan) {
            if c.offset > c.len {
                return Err(format!("chan: {} is {:#x} into a {:#x}-byte ring", info.name, c.offset, c.len));
            }
            if c.next >= DESC_RAM_ENTRIES {
                return Err(format!("chan: {} links to descriptor {}, the card has {DESC_RAM_ENTRIES}", info.name, c.next));
            }
        }
        if u64::from(self.status_slots) > SLOT_MAX {
            return Err(format!("status_slots: {}, at most {SLOT_MAX}", self.status_slots));
        }
        if self.status_cursor >= self.status_slots.max(1) {
            return Err(format!("status_cursor: slot {} of {}", self.status_cursor, self.status_slots));
        }
        Ok(())
    }
}

/// A mute word as saved: 0 not yet written, 1 released, 2 muted.
fn mute_word(m: Option<bool>) -> u8 {
    match m {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }
}

fn mute_from_word(w: u8) -> Result<Option<bool>, String> {
    match w {
        0 => Ok(None),
        1 => Ok(Some(false)),
        2 => Ok(Some(true)),
        _ => Err(format!("mute word {w}: not 0, 1 or 2")),
    }
}

// The card's state for a snapshot: its registers and RAMs, each channel's
// ring position and descriptor link, the clock generators' rates and loads,
// the output's generator, the mutes, UST and the status schedule. The card
// keeps time on its own clock, host time less the time it was stopped and
// the time held back for timer hold-ups; the save records that clock and
// the time held back, and the load sets the stopped time so the clock
// resumes at the saved instant, as the machine's other clocks do. The
// timers are armed again when the card starts. The host audio stream is
// this process's own and is not saved.
impl Saveable for Rad1 {
    fn save_state(&self) -> toml::Value {
        let i = &self.inner;
        let now = i.now();
        let s = i.state.lock();
        let chans = s.chan.iter().map(|c| {
            StateTable::new()
                .bool("running", c.running)
                .u64("base", c.base)
                .u64("len", c.len)
                .u64("offset", c.offset)
                .u64("start_ns", c.start_ns as u64)
                .u64("frames", c.frames)
                .bool("eod", c.eod)
                .u32("next", c.next)
                .bool("looping", c.looping)
                .bool("stalled", c.stalled)
                .build()
        }).collect();
        let rates: Vec<u64> = s.clock.iter().flat_map(|r| [r.num, r.den]).collect();
        // Each generator's last load as (loaded, control, remainder).
        let cg_loaded: Vec<u64> = s.cg_loaded.iter()
            .flat_map(|l| l.map_or([0, 0, 0], |(c, r)| [1, u64::from(c), u64::from(r)]))
            .collect();
        StateTable::new()
            .u64("clock_ns", now as u64)
            .u64("stalled_ns", i.stalled_ns.load(Ordering::SeqCst) as u64)
            .u32s("startup", &s.startup)
            .u64s("subcode", &s.subcode)
            .u32s("dma_bank", &s.dma_bank)
            .u32s("sample_pos", &s.sample_pos)
            .u32("analog_out_en", s.analog_out_en)
            .u64("out_cg", s.out_cg as u64)
            .u64s("cg_loaded", &cg_loaded)
            .u8("relay_muted", mute_word(s.relay_muted))
            .u8("dac_muted", mute_word(s.dac_muted))
            .u32("status_dma_ctrl", s.status_dma_ctrl)
            .bool("status_running", s.status_running)
            .value("chan", toml::Value::Array(chans))
            .u64s("clock", &rates)
            .u32("ust_count", s.ust_count)
            .u64("ust_phase", s.ust_phase as u64)
            .bool("ust_enabled", s.ust_enabled)
            .bool("ust_armed", s.ust_armed)
            .u64("ust_last_ns", s.ust_last_ns as u64)
            .u64("status_gen", s.status_gen)
            .u32("status_interval", s.status_interval)
            .u64("status_start_ns", s.status_start_ns as u64)
            .u64("status_done", s.status_done)
            .u32("status_cursor", s.status_cursor)
            .u32("status_slots", s.status_slots)
            .build()
    }

    /// Called with the card stopped.
    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let i = &self.inner;
        let chans = field(v, "chan")?.as_array().ok_or("chan: not a list")?;
        let rates = field_u64_vec(v, "clock")?;
        if rates.len() != NUM_CG * 2 {
            return Err("clock: not one rate for each generator".into());
        }
        let subcode = field_u64_vec(v, "subcode")?;
        if subcode.len() != SUBCODE_WORDS {
            return Err(format!("subcode: {} words, expected {SUBCODE_WORDS}", subcode.len()));
        }
        let saved_now = field_u64(v, "clock_ns")? as i64;
        let saved_stalled = field_u64(v, "stalled_ns")? as i64;
        let out_cg = field_u64(v, "out_cg")? as usize;
        if out_cg >= NUM_CG {
            return Err(format!("out_cg: generator {out_cg}, the card has {NUM_CG}"));
        }
        let loads: [u64; NUM_CG * 3] = field_u64s(v, "cg_loaded")?;
        let mut cg_loaded = [None; NUM_CG];
        for (slot, &[loaded, ctrl, rem]) in cg_loaded.iter_mut().zip(loads.as_chunks::<3>().0) {
            *slot = match loaded {
                0 => None,
                1 => Some((ctrl as u32, rem as u32)),
                _ => return Err("cg_loaded: not a generator load".into()),
            };
        }
        // Whole structs, so a field added to the card's state cannot be
        // left out of the load without the compiler saying so.
        let mut chan = Vec::with_capacity(NUM_CH);
        for saved in chans {
            chan.push(Chan {
                running: field_bool(saved, "running")?,
                base: field_u64(saved, "base")?,
                len: field_u64(saved, "len")?,
                offset: field_u64(saved, "offset")?,
                start_ns: field_u64(saved, "start_ns")? as i64,
                frames: field_u64(saved, "frames")?,
                eod: field_bool(saved, "eod")?,
                next: field_u32(saved, "next")?,
                looping: field_bool(saved, "looping")?,
                stalled: field_bool(saved, "stalled")?,
            });
        }
        let chan: [Chan; NUM_CH] = chan.try_into().map_err(|_| "chan: not one entry for each channel")?;
        let mut subcode_words = [0u64; SUBCODE_WORDS];
        subcode_words.copy_from_slice(&subcode);
        // A save from before register 0x84 was stored holds one word fewer:
        // 0x84 is the last, and read 0 then, as a reset card's does.
        let startup: [u32; STARTUP_OFFSETS.len()] = if field(v, "startup")?.as_array().is_some_and(|w| w.len() == STARTUP_OFFSETS.len() - 1) {
            let old = field_u32s::<{ STARTUP_OFFSETS.len() - 1 }>(v, "startup")?;
            let mut all = [0; STARTUP_OFFSETS.len()];
            all[..old.len()].copy_from_slice(&old);
            all
        } else {
            field_u32s(v, "startup")?
        };
        let loaded = State {
            startup,
            subcode: subcode_words,
            dma_bank: field_u32s(v, "dma_bank")?,
            sample_pos: field_u32s(v, "sample_pos")?,
            analog_out_en: field_u32(v, "analog_out_en")?,
            out_cg,
            cg_loaded,
            relay_muted: mute_from_word(field_u8(v, "relay_muted")?)?,
            dac_muted: mute_from_word(field_u8(v, "dac_muted")?)?,
            status_dma_ctrl: field_u32(v, "status_dma_ctrl")?,
            status_running: field_bool(v, "status_running")?,
            chan,
            clock: std::array::from_fn(|g| Rate { num: rates[2 * g], den: rates[2 * g + 1] }),
            ust_count: field_u32(v, "ust_count")?,
            ust_phase: field_u64(v, "ust_phase")? as i64,
            ust_enabled: field_bool(v, "ust_enabled")?,
            ust_armed: field_bool(v, "ust_armed")?,
            ust_last_ns: field_u64(v, "ust_last_ns")? as i64,
            status_gen: field_u64(v, "status_gen")?,
            status_interval: field_u32(v, "status_interval")?,
            status_start_ns: field_u64(v, "status_start_ns")? as i64,
            status_done: field_u64(v, "status_done")?,
            status_cursor: field_u32(v, "status_cursor")?,
            status_slots: field_u32(v, "status_slots")?,
            // The card is stopped: its timers were removed, and come back
            // when it starts.
            audio_timer: None,
            status_timer: None,
        };
        loaded.check()?;
        {
            let mut s = i.state.lock();
            if s.audio_timer.is_some() || s.status_timer.is_some() {
                return Err("RAD1 loaded while its timers run".into());
            }
            *s = loaded;
        }
        // The clock reads `saved_now` again, with the same time held back
        // for timer hold-ups, which later on-time completions repay; and
        // nothing later than the saved time has been handed out.
        i.restore(saved_now, saved_stalled);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
