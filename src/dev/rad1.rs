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
//! * A BAR0 access the model does not recognise is logged and reads zero.
//!   The model fails the bus access instead, which the PCI bus layer here
//!   cannot express.
//!
//! Ported from hw/audio/sgi-rad1-audio.c in jamesbraid/qemu-sgi-origami.

use std::ops::Deref;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::Mutex;
use std::time::{Duration, Instant};

use crate::config::AudioConfig;
use crate::devlog::LogModule;
use crate::host_audio::{HostOut, Resampler};
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::traits::Device;
use crate::pci::{Bar, PciBusDevice, PciConfig, PciDma};

pub const PCI_VENDOR_SGI: u16 = 0x10a9;
pub const PCI_DEVICE_RAD1: u16 = 0x0005;
pub const RAD1_REVISION: u8 = 0xc0;
/// "Not defined" (base class 0), as the model has it.
const PCI_CLASS_NOT_DEFINED: u32 = 0;
pub const BAR0_SIZE: u32 = 0x2000;

const REG_UST_COUNTER: u32 = 0x30;
const REG_UST_CONTROL: u32 = 0x40;
const REG_AES_INPUT_CTRL: u32 = 0x48;
const REG_ANALOG_INPUT_CTRL: u32 = 0x4c;
const REG_ANALOG_OUT_EN: u32 = 0x58;
const REG_STATUS_DMA_CTRL: u32 = 0x5c;
const REG_CLOCKGEN_CTRL: u32 = 0xa0;
const REG_CLOCKGEN_REM: u32 = 0xa4;
/// Four master-clock selects, one per clock generator, generator 3 first.
const REG_CLOCKGEN_MUX_FIRST: u32 = 0xa8;
const REG_CLOCKGEN_MUX_LAST: u32 = 0xb4;

const POS_FIRST: u32 = 0x04;
const POS_LAST: u32 = 0x2c;
const POS_WORDS: usize = ((POS_LAST - POS_FIRST) / 4 + 1) as usize;

const DMA_BANK_FIRST: u32 = 0x400;
const DMA_BANK_LAST: u32 = 0x528;
const DMA_BANK_WORDS: usize = ((DMA_BANK_LAST - DMA_BANK_FIRST) / 4 + 1) as usize;

/// A descriptor set is three registers (address low, address high, control)
/// for set A, then the same for set B.
type RingRegs = [u32; 6];
const RING_STATUS: RingRegs = [0x500, 0x528, 0x504, 0x4b4, 0x4b8, 0x4bc];
const RING_INPUT: RingRegs = [0x4d0, 0x510, 0x4d4, 0x400, 0x404, 0x408];
const RING_OUTPUT: RingRegs = [0x4f8, 0x524, 0x4fc, 0x40c, 0x410, 0x414];
const RING_AES_SUB: RingRegs = [0x4e0, 0x518, 0x4e4, 0x448, 0x44c, 0x450];
const RINGS: [RingRegs; 4] = [RING_STATUS, RING_INPUT, RING_OUTPUT, RING_AES_SUB];

/// The start-up words: stored and read back. Unsupported interfaces stay
/// disabled (see the refused values in [`Rad1::write`]).
const STARTUP_OFFSETS: [u32; 26] = [
    0x44, 0x48, 0x4c, 0x50, 0x54, 0x70, 0x74, 0x78, 0x7c, 0x80, 0x88, 0x8c, 0x90, 0xa0, 0xa4,
    0xa8, 0xac, 0xb0, 0xb4, 0xb8, 0xbc, 0x4c0, 0x4c8, 0x4d8, 0x4e8, 0x4f0,
];

const SUBCODE_FIRST: u32 = 0x1000;
const SUBCODE_LAST: u32 = 0x1218;
const SUBCODE_WORDS: usize = ((SUBCODE_LAST - SUBCODE_FIRST) / 8 + 1) as usize;

const ENABLE_BIT: u32 = 1;
/// The analog output's enable register: bit 0 starts the channel, bit 29 or
/// 30 picks generator 1 or 2 (mediaio-rad1-audio, output channel). The
/// "bit 28 + n" pattern is inferred from those two; other bits are unknown.
const OUT_SELECT_GEN1: u32 = 1 << 29;
const OUT_SELECT_GEN2: u32 = 1 << 30;
const SUBCODE_ENABLE_BIT: u32 = 4;
const CTRL_TAG_SHIFT: u32 = 3;
const CTRL_TAG_MASK: u32 = 0xf;
const CTRL_LEN_SHIFT: u32 = 7;
const CTRL_LEN_MASK: u32 = 0x1ff_ffff;
const DMA_UNIT_BYTES: u64 = 128;
/// Largest ring walked; only stops a garbage control word naming gigabytes.
const DMA_MAX_BYTES: u64 = 1 << 20;

const STATUS_START: u32 = 0x40003;
const STATUS_ONESHOT_START: u32 = 0x40001;
const STATUS_STOP: u32 = 0x1fffc;

const UST_NS_PER_TICK: i64 = 960;

const NS_PER_SEC: u64 = 1_000_000_000;
/// The rate of a clock generator no guest has programmed.
const FREQ_ANALOG: u64 = 44100;
/// The WAV capture's rate.
const WAV_RATE: u32 = 44100;
const FREQ_AES_SUB: u64 = 48000 / 192;
/// Ring bytes moved per DMA access; a whole number of frames of any type.
const CHUNK_BYTES: u64 = 1024;
/// How often the pump moves output frames to the host.
const PUMP_MS: u64 = 10;
const POS_FIELD_MASK: u32 = 0x3fff;
const UST_LATCH_MASK: u32 = 0x3_ffff;

/// Status-DMA descriptor slots: 0x80 bytes apart, at most 0x80 of them.
const SLOT_STRIDE: u64 = 0x80;
const SLOT_MAX: u64 = 0x80;
const SLOT_LIMIT_BASE: i64 = 0x78;
const SLOT_MIRROR_OFFSET: u64 = 0x40;
const SLOT_FLAGS_OFFSET: u64 = 0x78;
const SLOT_DONE_OFFSET: u64 = 0x7c;
/// Done, with the AES3 "no input connected" bit set.
const SLOT_DONE_WORD: u32 = 0x8000_0000 | 0x0008_0000;
const STATUS_CLOCK_HZ: u64 = 33_000_000;
const STATUS_INTERVAL_SHIFT: u32 = 2;
const STATUS_INTERVAL_MASK: u32 = 0x7fff;

/// A 16-bit host sample sits in bits [23:8] of its 24-bit ring sample.
const SAMPLE_SHIFT: u32 = 8;
/// Words a host frame carries: the mixer takes two channels.
const PCM_CHANNELS: usize = 2;
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

/// Where decoded output goes: the host stream (through the shared resampler),
/// an optional 44.1 kHz WAV file of what the card played, and for tests a
/// plain buffer of the card's own frames.
struct Sink {
    host: Option<(HostOut, Resampler)>,
    wav: Option<Wav>,
    tap: Option<Arc<Mutex<Vec<i16>>>>,
}

impl Sink {
    /// Play stereo frames the card produced at `rate` Hz.
    fn play(&mut self, pcm: &[i16], rate: u32) {
        if let Some((out, rs)) = self.host.as_mut() {
            if rs.in_rate != rate {
                *rs = Resampler::new(rate, out.stream_rate);
            }
            for p in pcm.chunks_exact(PCM_CHANNELS) {
                rs.push(p[0], p[1], &mut out.producer);
            }
        }
        if let Some(w) = self.wav.as_mut() {
            w.play(pcm, rate);
        }
        if let Some(t) = self.tap.as_ref() {
            t.lock().extend_from_slice(pcm);
        }
    }
}

/// A 44.1 kHz stereo 16-bit WAV of the output, converted from the card's
/// rate, its sizes rewritten after every append so a file from a killed run
/// still plays. A debugging aid: `IRIS_OCTANE_AUDIO_WAV`.
struct Wav {
    file: std::fs::File,
    data_bytes: u32,
    rs: Resampler,
}

impl Wav {
    fn create(path: &std::ffi::OsStr) -> Option<Self> {
        let file = std::fs::File::create(path).ok()?;
        let mut w = Self { file, data_bytes: 0, rs: Resampler::new(WAV_RATE, WAV_RATE) };
        w.header().ok()?;
        Some(w)
    }

    /// Append frames the card produced at `rate` Hz.
    fn play(&mut self, pcm: &[i16], rate: u32) {
        if self.rs.in_rate != rate {
            self.rs = Resampler::new(rate, WAV_RATE);
        }
        if self.rs.passthrough() {
            self.append(pcm);
            return;
        }
        // The resampler writes into a ring; take its output back out.
        let frames = pcm.len() / PCM_CHANNELS;
        let room = (frames * WAV_RATE as usize / rate.max(1) as usize + 2) * PCM_CHANNELS;
        let (mut prod, mut cons) = rtrb::RingBuffer::<i16>::new(room);
        for p in pcm.chunks_exact(PCM_CHANNELS) {
            self.rs.push(p[0], p[1], &mut prod);
        }
        let out: Vec<i16> = std::iter::from_fn(|| cons.pop().ok()).collect();
        self.append(&out);
    }

    fn header(&mut self) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let rate = WAV_RATE;
        let mut h = Vec::with_capacity(44);
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&(36 + self.data_bytes).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&1u16.to_le_bytes());
        h.extend_from_slice(&(PCM_CHANNELS as u16).to_le_bytes());
        h.extend_from_slice(&rate.to_le_bytes());
        h.extend_from_slice(&(rate * 4).to_le_bytes());
        h.extend_from_slice(&4u16.to_le_bytes());
        h.extend_from_slice(&16u16.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&self.data_bytes.to_le_bytes());
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&h)?;
        self.file.seek(SeekFrom::End(0))?;
        Ok(())
    }

    fn append(&mut self, pcm: &[i16]) {
        use std::io::Write;
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        if self.file.write_all(&bytes).is_ok() {
            self.data_bytes = self.data_bytes.wrapping_add(bytes.len() as u32);
            let _ = self.header();
        }
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

/// Pointer words in a status slot for the channel types the Octane does not
/// run: written as zero.
const SLOT_PTR_UNUSED: [u64; 5] = [0x00, 0x08, 0x18, 0x28, 0x30];

const CHAN_INFO: [ChanInfo; NUM_CH] = [
    ChanInfo {
        name: "analog input", ring: RING_INPUT, tag: 0, frame_bytes: 8, enable_bit: ENABLE_BIT,
        output: false, clockgen: Some(0), freq: 0, pos_a: 0x14, pos_a_shift: 18, pos_b: 0x18,
        pos_b_shift: 18, pos_b_parent: false, slot_ptr: 0x10,
    },
    ChanInfo {
        name: "analog output", ring: RING_OUTPUT, tag: 1, frame_bytes: 8, enable_bit: ENABLE_BIT,
        output: true, clockgen: Some(1), freq: 0, pos_a: 0x2c, pos_a_shift: 18, pos_b: 0x18,
        pos_b_shift: 4, pos_b_parent: false, slot_ptr: 0x38,
    },
    ChanInfo {
        name: "AES3-input subcode", ring: RING_AES_SUB, tag: 6, frame_bytes: 256,
        enable_bit: SUBCODE_ENABLE_BIT, output: false, clockgen: None, freq: FREQ_AES_SUB, pos_a: 0x10,
        pos_a_shift: 4, pos_b: 0x10, pos_b_shift: 18, pos_b_parent: true, slot_ptr: 0x20,
    },
];

#[derive(Default, Clone, Copy)]
struct Chan {
    running: bool,
    /// Ring latched from set A at the enable edge.
    base: u64,
    len: u64,
    offset: u64,
    start_ns: i64,
    frames: u64,
}

struct State {
    startup: [u32; 26],
    subcode: [u64; SUBCODE_WORDS],
    dma_bank: [u32; DMA_BANK_WORDS],
    sample_pos: [u32; POS_WORDS],
    analog_out_en: u32,
    /// The clock generator the analog output runs from (0x58 selects it).
    out_cg: usize,
    /// Each generator's last control and remainder words; the master select
    /// (0xa8..0xb4) can arrive after the load.
    cg_loaded: [Option<(u32, u32)>; NUM_CG],
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
    fn new(now: i64) -> Self {
        Self {
            startup: [0; 26],
            subcode: [0; SUBCODE_WORDS],
            dma_bank: [0; DMA_BANK_WORDS],
            sample_pos: [0; POS_WORDS],
            analog_out_en: 0,
            out_cg: OUT_DEFAULT_CG,
            cg_loaded: [None; NUM_CG],
            status_dma_ctrl: 0,
            status_running: false,
            chan: [Chan::default(); NUM_CH],
            clock: [Rate::hz(FREQ_ANALOG); NUM_CG],
            ust_count: 0,
            ust_phase: 0,
            ust_enabled: false,
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
        if is_status && limit == 0xffff_fffc {
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

    /// The enable edge: latch set A and start. A ring that is not
    /// programmed, or not a whole number of frames, starts nothing.
    fn chan_start(&mut self, idx: usize, now: i64) {
        let info = &CHAN_INFO[idx];
        let lo = self.bank(info.ring[0]);
        let hi = self.bank(info.ring[1]);
        let ctrl = self.bank(info.ring[2]);
        let base = (u64::from(hi) << 32) | u64::from(lo);
        let neg = (ctrl >> CTRL_LEN_SHIFT) & CTRL_LEN_MASK;
        let units = 0u32.wrapping_sub(neg) & CTRL_LEN_MASK;
        let mut len = u64::from(units) * DMA_UNIT_BYTES;
        let tag = (ctrl >> CTRL_TAG_SHIFT) & CTRL_TAG_MASK;

        if base == 0 || len == 0 {
            crate::dlog!(LogModule::Rad1, "rad1: {} enabled with no ring programmed (address {base:#x}, control {ctrl:#x})", info.name);
            return;
        }
        if len > DMA_MAX_BYTES {
            crate::dlog!(LogModule::Rad1, "rad1: {} ring length {len:#x} exceeds the modelled cap; clamping", info.name);
            len = DMA_MAX_BYTES;
        }
        // What hardware does with a partial frame is unknown; start nothing.
        if !len.is_multiple_of(info.frame_bytes) {
            crate::dlog!(LogModule::Rad1, "rad1: {} ring length {len:#x} is not a whole number of frames", info.name);
            return;
        }
        if tag != info.tag {
            crate::dlog!(LogModule::Rad1, "rad1: {} control word {ctrl:#x} carries tag {tag}, not the channel type {}", info.name, info.tag);
        }
        self.chan[idx] = Chan { running: true, base, len, offset: 0, start_ns: now, frames: 0 };
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
        Self { inner: Arc::new_cyclic(|me| Inner::new(me.clone())) }
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

/// Lock order: `tick_pass`, then `stopped_at` (a leaf, taken by `now`),
/// then `state`. Rings and status slots are moved with `state` held: the
/// DMA goes down through the Bridge and HEART, which never call back into
/// the card, and a catch-up moves at most two revolutions of a ring.
pub struct Inner {
    pci: PciConfig,
    state: Mutex<State>,
    start: Instant,
    /// When not `i64::MIN`, the clock in nanoseconds, for tests; no timer
    /// is armed then.
    fake_ns: AtomicI64,
    /// Nanoseconds the card's clock stood still while the machine was
    /// stopped, and when the current stop began: the sample clock, the UST
    /// and the status schedule do not run on while the guest cannot see
    /// them.
    stopped_ns: AtomicI64,
    stopped_at: Mutex<Option<Instant>>,
    /// Held across each timer callback's work, so a stop can wait one out.
    tick_pass: Mutex<()>,
    timers: OnceLock<Arc<TimerManager>>,
    /// Between `Device::start` and `Device::stop`: timers are armed only
    /// then.
    armed: AtomicBool,
    dma: OnceLock<Arc<dyn PciDma>>,
    /// `[audio]` settings once host output is enabled; `None` (tests,
    /// `no_audio`) keeps the output channel running silently.
    audio_cfg: Mutex<Option<AudioConfig>>,
    sink: Mutex<Option<Sink>>,
    underruns: Arc<AtomicU64>,
    me: Weak<Inner>,
}

impl Inner {
    fn new(me: Weak<Inner>) -> Self {
        Self {
            pci: PciConfig::new(
                PCI_VENDOR_SGI,
                PCI_DEVICE_RAD1,
                PCI_CLASS_NOT_DEFINED,
                RAD1_REVISION,
                [Bar::Mem(BAR0_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                0,
            ),
            state: Mutex::new(State::new(0)),
            start: Instant::now(),
            fake_ns: AtomicI64::new(i64::MIN),
            stopped_ns: AtomicI64::new(0),
            stopped_at: Mutex::new(None),
            tick_pass: Mutex::new(()),
            timers: OnceLock::new(),
            armed: AtomicBool::new(false),
            dma: OnceLock::new(),
            audio_cfg: Mutex::new(None),
            sink: Mutex::new(None),
            underruns: Arc::new(AtomicU64::new(0)),
            me,
        }
    }

    /// Play the analog output through the host with `audio`'s settings. The
    /// stream opens when the output first starts. While the output runs, a
    /// timer keeps its frames moving between the guest's own looks.
    pub fn enable_output(&self, audio: AudioConfig) {
        *self.audio_cfg.lock() = Some(audio);
        let mut s = self.state.lock();
        self.arm_audio(&mut s);
    }

    /// Whether timers may be armed: started, on a real clock, with a
    /// manager.
    fn timers_live(&self) -> Option<&Arc<TimerManager>> {
        if !self.armed.load(Ordering::SeqCst) || self.fake_ns.load(Ordering::Relaxed) != i64::MIN {
            return None;
        }
        self.timers.get()
    }

    /// Run the output pump exactly while the analog output channel runs with
    /// host audio enabled. Called with the state lock held after anything
    /// that can start or stop the channel.
    fn arm_audio(&self, s: &mut State) {
        let want = s.chan[CH_OUT].running && self.audio_cfg.lock().is_some();
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
        let now = self.now();
        let mut s = self.state.lock();
        self.sync_chan(&mut s, CH_OUT, now);
        if s.chan[CH_OUT].running {
            TimerReturn::Continue
        } else {
            s.audio_timer = None;
            TimerReturn::Delete
        }
    }

    /// Give the card the machine's timers. Called before `Device::start`.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timers.set(tm);
    }

    /// Open the output sink on the first start of the analog output.
    fn open_sink(&self) {
        let mut sink = self.sink.lock();
        if sink.is_some() {
            return;
        }
        let Some(cfg) = self.audio_cfg.lock().clone() else { return };
        let ring = |rate: u32| rate as usize * 2 / 5; // 200 ms of stereo
        let host = crate::host_audio::open_output(
            &cfg,
            self.underruns.clone(),
            Arc::new(AtomicBool::new(true)),
            ring,
            "Octane audio",
            "IRIS_OCTANE_AUDIO_CAPTURE",
        )
        .map(|out| {
            let rs = Resampler::new(FREQ_ANALOG as u32, out.stream_rate);
            (out, rs)
        });
        let wav = std::env::var_os("IRIS_OCTANE_AUDIO_WAV").and_then(|p| Wav::create(&p));
        *sink = Some(Sink { host, wav, tap: None });
    }

    fn now(&self) -> i64 {
        match self.fake_ns.load(Ordering::Relaxed) {
            i64::MIN => {
                let at = self.stopped_at.lock().unwrap_or_else(Instant::now);
                at.duration_since(self.start).as_nanos() as i64 - self.stopped_ns.load(Ordering::SeqCst)
            }
            t => t,
        }
    }

    /// Stop the card's clock and hold the timers off guest memory while the
    /// machine is stopped. Once this returns no timer callback is under way.
    fn pause(&self) {
        {
            let mut at = self.stopped_at.lock();
            if at.is_none() {
                *at = Some(Instant::now());
            }
        }
        self.disarm();
        drop(self.tick_pass.lock());
    }

    /// Run the clock again from where it stopped.
    fn resume(&self) {
        {
            let mut at = self.stopped_at.lock();
            if let Some(t) = at.take() {
                self.stopped_ns.fetch_add(t.elapsed().as_nanos() as i64, Ordering::SeqCst);
            }
        }
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
        let mut s = self.state.lock();
        for id in [s.audio_timer.take(), s.status_timer.take()].into_iter().flatten() {
            if let Some(tm) = self.timers.get() {
                tm.remove(id);
            }
        }
    }

    fn stopped(&self) -> bool {
        self.stopped_at.lock().is_some()
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
        *s = State::new(now);
        s.status_gen = gen;
    }

    fn startup_index(off: u32) -> Option<usize> {
        STARTUP_OFFSETS.iter().position(|&o| o == off)
    }

    fn is_descriptor(off: u32) -> bool {
        RINGS.iter().any(|r| r.contains(&off))
    }

    fn subcode_access(off: u32, width: usize) -> bool {
        let word = off & !7;
        // The model also takes 8-byte accesses; this bus layer's widest is 4.
        width == 4
            && off & 3 == 0
            && (SUBCODE_FIRST..=SUBCODE_LAST).contains(&word)
            && word != 0x1078
            && word != 0x10f8
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
                    0x44 => 5,
                    0x48 | 0x50 | 0x54 => 1,
                    _ => 0,
                };
                if val & refused != 0 {
                    return None;
                }
                if off == REG_ANALOG_INPUT_CTRL {
                    // The analog input's interface word: bit 0 runs it.
                    if val & ENABLE_BIT != 0
                        && !s.chan[CH_IN].running
                        && !s.ring_valid(&RING_INPUT, false, 0xffff_f000)
                    {
                        return None;
                    }
                    self.enable_write(s, CH_IN, val, now);
                } else if off == REG_AES_INPUT_CTRL {
                    // AES3 input control: bit 2 runs the input subcode channel.
                    if val & SUBCODE_ENABLE_BIT != 0
                        && !s.chan[CH_AES_SUB].running
                        && !s.ring_valid(&RING_AES_SUB, false, 0xffff_f030)
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
            } else if off == REG_ANALOG_OUT_EN {
                return self.analog_out_write(s, val, now);
            } else if off == REG_STATUS_DMA_CTRL {
                self.status_ctrl_write(s, val, now)?;
                s.status_dma_ctrl = val;
                return Some(());
            }
        }
        if off == REG_UST_CONTROL && width == 4 {
            self.sync_all(s, now);
            match val {
                0x0e => {
                    s.ust_update(now);
                    s.ust_enabled = false;
                    s.ust_armed = true;
                    return Some(());
                }
                0x0a if s.ust_armed => {
                    s.ust_update(now);
                    s.ust_enabled = true;
                    s.ust_armed = false;
                    return Some(());
                }
                0x02 => {
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
        if val & ENABLE_BIT != 0 && !s.chan[CH_OUT].running && !s.ring_valid(&RING_OUTPUT, false, 0xffff_f008) {
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
            && s.ring_valid(&RING_STATUS, true, 0xffff_fffc)
        {
            // One completion into the ring's base slot, as a continuous
            // completion would fill it.
            let base = Self::status_base(s);
            self.fill_slot(s, base, now);
        } else if val & !0x1fffc == STATUS_START
            && val & 0x1fffc != 0
            && !s.status_running
            && s.ring_valid(&RING_STATUS, true, 0xffff_c078)
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
        let now = self.now();
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
        let Some(dma) = self.dma.get() else { return false };
        self.sync_all(s, now);

        let mut mirror = Vec::with_capacity(13 * 4);
        for reg in (0x04..=0x34).step_by(4) {
            let v = match reg {
                0x04..=0x2c => s.sample_pos[((reg - POS_FIRST) / 4) as usize],
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
        let ok = put(SLOT_MIRROR_OFFSET + 0x04, &mirror);
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
        ok
    }

    /// A write to a channel's enable register. Its own bit starts and stops
    /// it; the register's other bits never do.
    fn enable_write(&self, s: &mut State, idx: usize, val: u32, now: i64) {
        if val & CHAN_INFO[idx].enable_bit != 0 {
            if !s.chan[idx].running {
                s.chan_start(idx, now);
                if idx == CH_OUT && s.chan[idx].running {
                    self.open_sink();
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
        let Some(dma) = self.dma.get() else { return false };
        // A position read after a long look-away would replay every
        // revolution of the ring it missed, a DMA per chunk on the CPU
        // thread. Past one revolution the ring is left as it was found (an
        // output ring is only read, an input ring is already silence), so
        // whole revolutions beyond that are credited without moving
        // anything. The position fields are 14 bits, so the count's
        // truncation to 32 bits loses nothing.
        let ring_frames = s.chan[idx].len / info.frame_bytes;
        if ring_frames != 0 && frames >= 2 * ring_frames {
            let skip = (frames / ring_frames - 1) * ring_frames;
            s.chan[idx].frames += skip;
            s.advance_pos(idx, skip as u32);
            frames -= skip;
        }
        let mut ring = [0u8; CHUNK_BYTES as usize];
        let mut pcm = [0i16; CHUNK_BYTES as usize / 4];
        while frames != 0 {
            let c = s.chan[idx];
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
                    sink.play(samples, s.rate(idx).rounded_hz());
                }
            }
            let c = &mut s.chan[idx];
            c.offset += chunk;
            c.frames += n;
            if c.offset >= c.len {
                c.offset = 0;
            }
            s.advance_pos(idx, n as u32);
            frames -= n;
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
}

impl PciBusDevice for Rad1 {
    fn config(&self) -> &PciConfig {
        &self.inner.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        self.inner.read(off, width)
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.inner.write(off, val, width)
    }

    /// The card masters the bus for its status slots and audio rings. It
    /// raises no interrupt, so there is no `connect_interrupt`.
    fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        let _ = self.inner.dma.set(dma);
    }
}

#[cfg(test)]
mod tests;
