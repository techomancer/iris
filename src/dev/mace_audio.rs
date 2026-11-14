//! The O2's audio: MACE's audio interface and the AD1843 codec behind it.
//!
//! MACE (spec chapter 3, "Audio Codec Interface") has no sample clock of its
//! own. It runs one TDM bus to an AD1843, three stereo DMA channels (one in,
//! two out) on 4 KB rings in main memory, and a register window onto the
//! codec: software names a codec register and a value, and the interface
//! sends it every TDM cycle, reading back the codec's status word.
//!
//! - Registers at MACE + 0x300000 (64-bit, 16 KB block, gaps alias):
//!   0x00 control and status, 0x08 codec address/control/data, 0x10 status
//!   input mask, 0x18 status input, then per channel at 0x20 + 0x20 n: ring
//!   control, read pointer, write pointer, depth.
//! - Rings are the first three 4 KB pages of the peripheral controller's
//!   32 KB ring block (ISA ring base register), 32-byte blocks of four 64-bit
//!   stereo items: left in bits 63:32, right in 31:0, each a 24-bit sample
//!   in bits 31:8 of its half, played as 16 bits with saturation.
//! - Interrupts are the low byte of the ISA interrupt status (status word,
//!   volume, then threshold and error for each channel), on CRIME line 6.
//! - Each channel counts sample pairs (MSC) and stamps UST, in the UST
//!   block at 0x20 + 8 n: MSC in bits 63:32, UST in 31:0.
//!
//! The codec is modelled from the AD1843's register layout (the field map
//! Linux's sgio2audio driver also uses): it reports itself initialised and
//! powered as soon as it is out of reset, and its clock generators, DAC
//! clock sources and mutes decide how each output channel plays.
//!
//! Output goes to the host through [`crate::host_audio`], the stream HAL2
//! plays through, opened when a channel first plays.

use std::sync::Arc;

use crate::snapshot::{get_field, hex_u32, hex_u64, load_u16_slice, toml_u32, toml_u64, u16_slice_to_toml};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;
use parking_lot::Mutex;
use crate::config::AudioConfig;
use crate::dev::crime_mem::O2Memory;
use crate::traits::BusDevice;
use crate::host_audio::{HostOut, Meter, Resampler};

/// Where the audio registers sit, and their 16 KB block.
pub const MACE_AUDIO: u32 = 0x1f30_0000;
pub const MACE_AUDIO_SIZE: u32 = 0x4000;

/// Channel control bits (as the serial DMA's).
const CTRL_RESET: u64 = 0x400;
const CTRL_ENABLE: u64 = 0x200;
const CTRL_MASK: u64 = 0x6e0;
/// Pointers and depths count 32-byte blocks in bits 11:5 of a 4 KB ring.
const PTR_MASK: u32 = 0xfe0;
const BLOCK: u32 = 32;

/// The ISA interrupt status bits this block owns, and its CRIME line.
pub const ISA_AUDIO_MASK: u64 = 0xff;
pub const CRIME_LINE: u32 = 6;

/// AD1843 registers (index, bit, width) the model acts on.
mod ad1843 {
    /// Register 0: INIT (15) and PDNO (14) read 0 once the codec is up.
    pub const STATUS: usize = 0;
    /// Register 9 / 10: DAC1 / DAC2 analog gain (13:8 left, 5:0 right,
    /// 1.5 dB steps, 8 unity, 0 loudest) and mute (bit 15 left, 7 right).
    pub const DAC1_GAIN: usize = 9;
    pub const DAC2_GAIN: usize = 10;
    /// Register 11 / 12: DAC1 / DAC2 digital attenuation (same fields, 0 is
    /// 0 dB) and mute.
    pub const DAC1_MUTE: usize = 11;
    pub const DAC2_MUTE: usize = 12;
    /// Register 15: DAC1 clock source in bits 9:8, DAC2 in 11:10, the left
    /// and right ADCs in 3:2 and 1:0 (0: a fixed 48 kHz; 1-3: clock
    /// generator 1-3).
    pub const CLOCK_SRC: usize = 15;
    /// Registers 17, 20, 23: clock generator 1-3 sample rate, in hertz.
    pub const CLOCK_RATE: [usize; 3] = [17, 20, 23];
    /// Register 27: DAC1 (bit 8) and DAC2 (bit 9) enables.
    pub const ENABLES: usize = 27;
    /// Register 28: clock generator enables (11-13), converter power-down (15).
    pub const POWER: usize = 28;
}

/// The trace: every register access and codec operation, on stderr.
fn tracing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("IRIS_O2_AUDIO_TRACE").is_some())
}

#[derive(Clone, Copy)]
struct Chan {
    ctl: u64,
    rptr: u32,
    wptr: u32,
    /// Items of the block at the hardware pointer already played (0-3).
    sub: u32,
    msc: u32,
    ust: u32,
    /// When the channel started running at `rate`, and the frames it has
    /// moved since.
    start: Option<Instant>,
    done: u64,
    rate: u32,
    /// Output: items played from an empty ring since it last had data.
    dry: u32,
}

impl Chan {
    const RESET: Self = Self { ctl: CTRL_RESET, rptr: 0, wptr: 0, sub: 0, msc: 0, ust: 0, start: None, done: 0, rate: 0, dry: 0 };

    fn running(&self) -> bool {
        self.ctl & CTRL_RESET == 0 && self.ctl & CTRL_ENABLE != 0
    }

    /// Frames due at `rate` since the channel started, at most a quarter
    /// second behind (a host stall is skipped, not replayed). A new rate
    /// starts the count again: amesh moved a running clock from 48 to
    /// 32 kHz, the old count passed the new target, and the difference,
    /// unsigned, wrapped to a loop the service thread never left.
    fn due(&mut self, now: Instant, rate: u32) -> u64 {
        if self.start.is_none() || self.rate != rate {
            self.start = Some(now);
            self.done = 0;
            self.rate = rate;
        }
        let target = (now - self.start.unwrap()).as_micros() as u64 * rate as u64 / 1_000_000;
        let lag = rate as u64 / 4;
        if target > self.done + lag {
            self.done = target - lag;
        }
        let due = target.saturating_sub(self.done);
        self.done = self.done.max(target);
        due
    }

    /// Blocks in the ring: write minus read, in 32-byte units.
    fn depth(&self) -> u32 {
        self.wptr.wrapping_sub(self.rptr) & PTR_MASK
    }

    /// Whether the threshold condition (control bits 7:5) holds. Input
    /// channels count fullness, output channels emptiness, as the spec
    /// words the 25/50/75% levels.
    fn threshold(&self, input: bool) -> bool {
        let blocks = self.depth() / BLOCK; // 0..=127
        let full = PTR_MASK / BLOCK; // 127
        match (self.ctl >> 5) & 7 {
            0 => false,
            n @ 1..=3 => {
                let level = full * n as u32 / 4;
                if input { blocks >= level } else { blocks < level }
            }
            4 => blocks == 0,
            5 => blocks != 0,
            6 => blocks == full,
            _ => blocks != full,
        }
    }
}

struct State {
    /// Control: codec reset (bit 0), volume button latches (24:23).
    ctl: u64,
    codec_ctl: u64,
    mask: u64,
    status_in: u64,
    ch: [Chan; 3],
    codec: [u16; 32],
    /// What the outputs played since the service thread last took it: per
    /// catch-up, each output's pairs and rate.
    played: Vec<([Vec<(i16, i16)>; 2], [u32; 2])>,
}

impl State {
    fn new() -> Self {
        let mut codec = [0u16; 32];
        // Out of reset: converters powered down until software says, and the
        // clock generators at 48 kHz. IRIX resets the codec after the PROM's
        // chime and never writes a rate for its desktop sounds; with the
        // rates cleared the channel never drained, and the program waiting on
        // it (the desktop, restarting) hung polling the read pointer.
        codec[ad1843::POWER] = 0x8000;
        for r in ad1843::CLOCK_RATE {
            codec[r] = 48000;
        }
        Self { ctl: 1, codec_ctl: 1 << 16, mask: 0, status_in: 0, ch: [Chan::RESET; 3], codec, played: Vec::new() }
    }

    fn codec_read(&self, reg: usize) -> u16 {
        match reg {
            // INIT and PDNO clear: initialised, and powered up whenever the
            // converters are (PDNO follows PDNI).
            ad1843::STATUS => (self.codec[0] & 0x3fff) | (self.codec[ad1843::POWER] & 0x8000) >> 1,
            _ => self.codec[reg],
        }
    }

    /// The rate a DAC (1 or 2) plays at, or None while it is off.
    fn dac_rate(&self, dac: usize) -> Option<u32> {
        let c = &self.codec;
        if c[ad1843::POWER] & 0x8000 != 0 || c[ad1843::ENABLES] & (1 << (7 + dac)) == 0 {
            return None;
        }
        Self::clock(c, (c[ad1843::CLOCK_SRC] >> (6 + 2 * dac)) & 3)
    }

    /// The rate a converter clocked from source `src` runs at: 0 is the
    /// fixed 48 kHz, 1-3 a clock generator, if it is on.
    fn clock(c: &[u16; 32], src: u16) -> Option<u32> {
        if src == 0 {
            return Some(48000);
        }
        if c[ad1843::POWER] & (1 << (10 + src)) == 0 {
            return None;
        }
        let rate = c[ad1843::CLOCK_RATE[src as usize - 1]] as u32;
        (4000..=54000).contains(&rate).then_some(rate)
    }

    /// The rate the ADC records at, or None while it is off: the left
    /// channel's clock source (register 15 bits 3:2) and enable (27 bit 0).
    fn adc_rate(&self) -> Option<u32> {
        let c = &self.codec;
        if c[ad1843::POWER] & 0x8000 != 0 || c[ad1843::ENABLES] & 1 == 0 {
            return None;
        }
        Self::clock(c, (c[ad1843::CLOCK_SRC] >> 2) & 3)
    }

    /// A DAC's left and right gains as factors. The analog register's
    /// 6-bit field (13:8 left, 5:0 right) is a gain/attenuation in 1.5 dB
    /// steps with 8 at unity, as the codec's mixer gains are, so 0 (IRIX's
    /// loudest) is +12 dB; the digital register's field (same bits) is an
    /// attenuation from 0 dB. Either register's bit 15 / 7 mutes.
    fn dac_gain(&self, dac: usize) -> (f32, f32) {
        let (g, d) = if dac == 1 { (ad1843::DAC1_GAIN, ad1843::DAC1_MUTE) } else { (ad1843::DAC2_GAIN, ad1843::DAC2_MUTE) };
        let one = |mute: u16, shift: u32| {
            if self.codec[g] & mute != 0 || self.codec[d] & mute != 0 {
                return 0.0;
            }
            let analog = 8.0 - ((self.codec[g] >> shift) & 0x3f) as f32;
            let digital = -(((self.codec[d] >> shift) & 0x3f) as f32);
            10f32.powf(1.5 * (analog + digital) / 20.0)
        };
        (one(0x8000, 8), one(0x80, 0))
    }

    /// The ISA interrupt bits: status word (0), volume (1), then each
    /// channel's threshold (2, 4, 6). The error bits (3, 5, 7) stay clear:
    /// the engine never underruns its FIFO.
    fn isa_bits(&self) -> u64 {
        let mut b = 0;
        if self.status_in & self.mask & 0xffff != 0 {
            b |= 1;
        }
        if self.ctl & (3 << 23) != 0 {
            b |= 2;
        }
        for (n, c) in self.ch.iter().enumerate() {
            if c.ctl & CTRL_RESET == 0 && c.threshold(n == 0) {
                b |= 1 << (2 + 2 * n);
            }
        }
        b
    }
}

/// What playback reaches outside the block.
pub struct AudioIo<'a> {
    pub ram: &'a O2Memory,
    /// The ring block's physical address, from the ISA ring base register.
    pub ring_base: u32,
    /// The MACE UST counter now, for the channels' stamps.
    pub ust: u32,
}

/// The host side: the stream, and a resampler per output channel.
struct Player {
    out: HostOut,
    rs: [Option<Resampler>; 2],
}

pub struct MaceAudio {
    st: Mutex<State>,
    player: Mutex<Option<Player>>,
    config: Mutex<Option<AudioConfig>>,
    tried_open: AtomicBool,
    /// What the outputs have played (silence for a dry ring included) and
    /// the host stream has done.
    meter: Arc<Meter>,
    /// For the trace: non-zero sample pairs each output played, and when
    /// that was last reported.
    heard: [AtomicU64; 2],
    /// For the trace: times each output's ring ran dry between sounds less
    /// than half a second apart (the program fell behind), and the items
    /// those gaps lasted.
    gaps: [AtomicU64; 2],
    gap_items: [AtomicU64; 2],
    last_report: Mutex<Option<Instant>>,
    /// IRIS_O2_AUDIO_DUMP=<file>: every sample each output plays, at the
    /// guest's rate, before resampling (see `dump`).
    dump: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    /// Its bits of MACE's ISA status (0..7).
    irq: std::sync::OnceLock<Vec<crate::irq::IrqLine>>,
}

impl Default for MaceAudio {
    fn default() -> Self {
        Self::new()
    }
}

impl MaceAudio {
    pub fn new() -> Self {
        Self {
            st: Mutex::new(State::new()),
            player: Mutex::new(None),
            config: Mutex::new(None),
            tried_open: AtomicBool::new(false),
            meter: Meter::new("MACE audio", &["analog in", "analog out 1", "analog out 2"]),
            heard: [AtomicU64::new(0), AtomicU64::new(0)],
            gaps: [AtomicU64::new(0), AtomicU64::new(0)],
            gap_items: [AtomicU64::new(0), AtomicU64::new(0)],
            last_report: Mutex::new(None),
            dump: Mutex::new(
                std::env::var_os("IRIS_O2_AUDIO_DUMP")
                    .and_then(|p| std::fs::File::create(p).ok())
                    .map(std::io::BufWriter::new),
            ),
            irq: std::sync::OnceLock::new(),
        }
    }

    /// Play through the host with `audio`'s settings. Without this (tests,
    /// `no_audio`) the channels still run and count, silently.
    pub fn enable_output(&self, audio: AudioConfig) {
        *self.config.lock() = Some(audio);
    }

    pub fn contains(addr: u32) -> bool {
        (MACE_AUDIO..MACE_AUDIO + MACE_AUDIO_SIZE).contains(&addr)
    }

    pub fn read(&self, addr: u32) -> u64 {
        let st = self.st.lock();
        let off = addr & 0x78;
        let v = match off {
            0x00 => {
                // Present: the codec is the TDM master and its serial clock
                // runs whenever it has power, reset or not. The PROM checks
                // this before it touches the codec at all (and, held clear
                // while in reset, never played its chime).
                st.ctl & 1 | 1 << 1
                    | ((st.ch[0].wptr >> 5) as u64) << 2
                    | ((st.ch[1].rptr >> 5) as u64) << 9
                    | ((st.ch[2].rptr >> 5) as u64) << 16
                    | st.ctl & (3 << 23)
            }
            0x08 => st.codec_ctl,
            0x10 => st.mask,
            0x18 => {
                // A read operation repeats every TDM cycle: the buffer holds
                // the named register as it is now.
                if st.codec_ctl & (1 << 16) != 0 && st.ctl & 1 == 0 {
                    st.codec_read(((st.codec_ctl >> 17) & 31) as usize) as u64
                } else {
                    st.status_in
                }
            }
            _ => {
                let c = &st.ch[(off as usize - 0x20) / 0x20];
                match off & 0x18 {
                    0x00 => c.ctl,
                    0x08 => c.rptr as u64,
                    0x10 => c.wptr as u64,
                    _ => c.depth() as u64,
                }
            }
        };
        if tracing() {
            eprintln!("o2audio: read  {off:#04x} = {v:#x}");
        }
        v
    }

    /// A register write. Returns whether the interrupt level may have moved.
    pub fn write(&self, addr: u32, val: u64) {
        let mut st = self.st.lock();
        let off = addr & 0x78;
        if tracing() {
            eprintln!("o2audio: write {off:#04x} = {val:#x}");
        }
        match off {
            0x00 => {
                // Bit 0 is the codec reset; the volume latches clear by
                // writing zeros to them.
                let was_reset = st.ctl & 1 != 0;
                st.ctl = (val & 1) | (st.ctl & val & (3 << 23));
                if st.ctl & 1 != 0 && !was_reset {
                    let (ch, played) = (st.ch, std::mem::take(&mut st.played));
                    *st = State { ch, played, ..State::new() };
                }
            }
            0x08 => {
                st.codec_ctl = val & 0xff_ffff;
                if st.ctl & 1 == 0 {
                    let reg = ((val >> 17) & 31) as usize;
                    if val & (1 << 16) == 0 {
                        st.codec[reg] = val as u16;
                        if tracing() {
                            eprintln!("o2audio: codec reg {reg:2} <- {:#06x} ({})", val as u16, val as u16);
                        }
                    } else {
                        st.status_in = st.codec_read(reg) as u64;
                    }
                }
            }
            0x10 => st.mask = val & 0xffff,
            0x18 => {}
            _ => {
                let n = (off as usize - 0x20) / 0x20;
                let c = &mut st.ch[n];
                match off & 0x18 {
                    0x00 => {
                        c.ctl = val & CTRL_MASK;
                        if c.ctl & CTRL_RESET != 0 {
                            *c = Chan { ctl: c.ctl, ..Chan::RESET };
                        } else if c.ctl & CTRL_ENABLE == 0 {
                            c.start = None;
                        }
                    }
                    // Software owns the input's read pointer and the
                    // outputs' write pointers; the engine owns the others.
                    0x08 if n == 0 => c.rptr = val as u32 & PTR_MASK,
                    0x10 if n != 0 => c.wptr = val as u32 & PTR_MASK,
                    _ => {}
                }
            }
        }
    }

    /// The channel MSC/UST pairs in MACE's UST block (offset 0x20 + 8 n):
    /// MSC in the high word, UST in the low, as the spec's MSC/UST figure
    /// draws it and IRIX's audio library reads it. (Its timer chapter's
    /// table lists the two words the other way round; read that way,
    /// alGetFrameTime ran 17 times fast and amesh drew the same samples
    /// over and over.)
    pub fn msc_ust(&self, n: usize) -> u64 {
        let c = &self.st.lock().ch[n];
        (c.msc as u64) << 32 | c.ust as u64
    }

    pub fn set_msc_ust(&self, n: usize, val: u64) {
        let c = &mut self.st.lock().ch[n];
        c.msc = (val >> 32) as u32;
        c.ust = val as u32;
    }

    /// The low byte of the ISA interrupt status.
    /// Wire the interface's status bits into MACE's ISA interrupt controller.
    pub fn connect_irq(&self, isa: Arc<dyn crate::irq::InterruptController>) {
        let _ = self.irq.set((0..8).map(|b| crate::irq::IrqLine::new(isa.clone(), b)).collect());
    }

    /// Drive the interface's status bits: after a register access and after
    /// the service loop moves the rings.
    pub fn update_irq(&self) {
        let Some(lines) = self.irq.get() else { return };
        let bits = self.isa_bits();
        for l in lines {
            l.set(bits & (1 << l.line()) != 0);
        }
    }

    pub fn isa_bits(&self) -> u64 {
        self.st.lock().isa_bits()
    }

    /// Play what is due on each output channel since the last call, from
    /// its ring to the host. Called from the machine's service thread.
    pub fn poll(&self, io: &AudioIo) {
        self.poll_at(io, Instant::now());
    }

    fn poll_at(&self, io: &AudioIo, now: Instant) {
        self.catch_up_at(io, now);
        let played = std::mem::take(&mut self.st.lock().played);
        for (pcm, rates) in &played {
            self.dump(pcm, rates);
            self.play(pcm, rates);
        }
    }

    /// Move each channel's ring up to now, keeping what the outputs played
    /// for the next [`Self::poll`]. A read of the registers comes here
    /// first, so software sees the pointers as they stand, not as of the
    /// last service pass up to 2 ms ago: IRIX reads the read pointer about
    /// once a millisecond and keeps its write pointer only about 2 ms
    /// ahead of it, so a pass that found the ring that stale played it dry
    /// and filled the tone with zeros.
    pub fn catch_up(&self, io: &AudioIo) {
        self.catch_up_at(io, Instant::now());
    }

    fn catch_up_at(&self, io: &AudioIo, now: Instant) {
        let mut st = self.st.lock();
        let mut pcm: [Vec<(i16, i16)>; 2] = [Vec::new(), Vec::new()];
        let mut rates = [0u32; 2];
        let ram = &*io.ram;
        // The input: no host capture yet, so the ADC delivers silence at its
        // rate, as a quiet microphone would. A recording program (amesh)
        // then sees its ring fill instead of waiting for ever.
        match st.adc_rate() {
            Some(rate) if st.ch[0].running() => {
                let c = &mut st.ch[0];
                let due = c.due(now, rate);
                self.meter.moved(0, due);
                for _ in 0..due {
                    let next = (c.wptr + BLOCK) & PTR_MASK;
                    if c.sub == 0 && next == c.rptr {
                        break; // full: the engine never fills the last slot
                    }
                    let addr = io.ring_base | c.wptr | c.sub * 8;
                    if let Some(o) = ram.dma_offset(addr).map(|o| o as u64) {
                        // Each half: sign extension, the sample, 0xff.
                        ram.phys().write32(o, 0xff);
                        ram.phys().write32(o + 4, 0xff);
                    }
                    c.sub += 1;
                    if c.sub == 4 {
                        c.sub = 0;
                        c.wptr = next;
                    }
                    c.msc = c.msc.wrapping_add(1);
                    c.ust = io.ust;
                }
            }
            _ => st.ch[0].start = None,
        }
        for out in 0..2 {
            let n = out + 1;
            let rate = match st.dac_rate(n) {
                Some(r) if st.ch[n].running() => r,
                _ => {
                    st.ch[n].start = None;
                    continue;
                }
            };
            let (gain_l, gain_r) = st.dac_gain(n);
            let c = &mut st.ch[n];
            rates[out] = rate;
            let due = c.due(now, rate);
            self.meter.moved(out + 1, due);
            for _ in 0..due {
                let mut item = (0i16, 0i16);
                if c.depth() == 0 {
                    c.dry = c.dry.saturating_add(1);
                } else {
                    if c.dry != 0 && c.dry < rate / 2 {
                        self.gaps[out].fetch_add(1, Ordering::Relaxed);
                        self.gap_items[out].fetch_add(c.dry as u64, Ordering::Relaxed);
                    }
                    c.dry = 0;
                    let addr = io.ring_base | (n as u32) << 12 | c.rptr | c.sub * 8;
                    if let Some(o) = ram.dma_offset(addr).map(|o| o as u64) {
                        let half = |w: u32| ((w as i32) >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        let (l, r) = (half(ram.phys().read32(o).data), half(ram.phys().read32(o + 4).data));
                        let scale = |v: i16, g: f32| (v as f32 * g).round() as i16;
                        item = (scale(l, gain_l), scale(r, gain_r));
                        if l != 0 || r != 0 {
                            self.heard[out].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    c.sub += 1;
                    if c.sub == 4 {
                        c.sub = 0;
                        c.rptr = (c.rptr + BLOCK) & PTR_MASK;
                    }
                }
                c.msc = c.msc.wrapping_add(1);
                c.ust = io.ust;
                pcm[out].push(item);
            }
        }
        if rates == [0, 0] {
            self.meter.stopped();
        }
        if pcm.iter().any(|p| !p.is_empty()) {
            st.played.push((pcm, rates));
        }
        drop(st);
        // The pointers moved: a ring may have crossed its threshold.
        self.update_irq();
        if tracing() {
            let mut last = self.last_report.lock();
            if last.is_none_or(|t| t.elapsed().as_secs() >= 1) {
                *last = Some(now);
                let h = [0, 1].map(|k| self.heard[k].swap(0, Ordering::Relaxed));
                if h != [0, 0] {
                    eprintln!("o2audio: non-zero pairs played: out1 {} out2 {}", h[0], h[1]);
                }
                let g = [0, 1].map(|k| self.gaps[k].swap(0, Ordering::Relaxed));
                let gi = [0, 1].map(|k| self.gap_items[k].swap(0, Ordering::Relaxed));
                if g != [0, 0] {
                    eprintln!("o2audio: ring ran dry: out1 {} ({} items) out2 {} ({} items)", g[0], gi[0], g[1], gi[1]);
                }
            }
        }
    }

    /// Append one record per output that played: the output (u8), its rate
    /// and pair count (u32 each), then the pairs, all little-endian. Works
    /// with or without host output, so a muted run can still be checked.
    fn dump(&self, pcm: &[Vec<(i16, i16)>; 2], rates: &[u32; 2]) {
        use std::io::Write;
        let mut dump = self.dump.lock();
        let Some(w) = dump.as_mut() else { return };
        for out in 0..2 {
            if pcm[out].is_empty() {
                continue;
            }
            let mut rec = Vec::with_capacity(9 + pcm[out].len() * 4);
            rec.push(out as u8);
            rec.extend_from_slice(&rates[out].to_le_bytes());
            rec.extend_from_slice(&(pcm[out].len() as u32).to_le_bytes());
            for &(l, r) in &pcm[out] {
                rec.extend_from_slice(&l.to_le_bytes());
                rec.extend_from_slice(&r.to_le_bytes());
            }
            let _ = w.write_all(&rec);
        }
        let _ = w.flush();
    }

    /// Resample each channel to the host rate, mix and push.
    fn play(&self, pcm: &[Vec<(i16, i16)>; 2], rates: &[u32; 2]) {
        let mut player = self.player.lock();
        if player.is_none() && !self.tried_open.swap(true, Ordering::Relaxed) {
            if let Some(cfg) = self.config.lock().clone() {
                let ring = |rate: u32| rate as usize * 2 / 5; // 200 ms of stereo
                *player = crate::host_audio::open_output(&cfg, self.meter.clone(), ring, "O2 audio", "IRIS_O2_AUDIO_CAPTURE")
                    .map(|out| Player { out, rs: [None, None] });
            }
        }
        let Some(p) = player.as_mut() else { return };
        let host = p.out.stream_rate;
        let mut mixed: Vec<i32> = Vec::new();
        for out in 0..2 {
            if pcm[out].is_empty() {
                continue;
            }
            let rs = p.rs[out].get_or_insert_with(|| Resampler::new(rates[out], host));
            if rs.in_rate != rates[out] {
                *rs = Resampler::new(rates[out], host);
            }
            // The resampler writes into a ring; take its output back out.
            let (mut prod, mut cons) = rtrb::RingBuffer::<i16>::new(pcm[out].len() * 2 * (host as usize / rates[out] as usize + 2) + 8);
            for &(l, r) in &pcm[out] {
                let _ = rs.push(l, r, &mut prod); // the scratch ring has room: nothing is refused
            }
            let mut k = 0;
            while let Ok(v) = cons.pop() {
                if k < mixed.len() {
                    mixed[k] += v as i32;
                } else {
                    mixed.push(v as i32);
                }
                k += 1;
            }
        }
        push_mixed(&mut p.out, &mixed);
    }
}

/// Clamp the mixed stereo pairs to 16 bits and push each pair whole or not
/// at all: with one slot left the left sample alone would go in and every
/// later pair would be swapped.
fn push_mixed(out: &mut HostOut, mixed: &[i32]) {
    for pair in mixed.chunks_exact(2) {
        let [l, r] = [pair[0], pair[1]].map(|v| v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        out.push_frame(&[l, r]);
    }
}

/// The bus side: MACE's 64-bit audio registers; a 32-bit read is the half
/// it names, narrower reads the low bits, and every store the value as
/// given.
impl Chan {
    fn save(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("ctl".into(), hex_u64(self.ctl));
        for (k, v) in [("rptr", self.rptr), ("wptr", self.wptr), ("sub", self.sub), ("msc", self.msc),
                       ("ust", self.ust), ("rate", self.rate), ("dry", self.dry)] {
            t.insert(k.into(), hex_u32(v));
        }
        t.insert("done".into(), hex_u64(self.done));
        toml::Value::Table(t)
    }

    /// The pacing clock restarts from the load: frames due are counted
    /// from then, as after a host stall.
    fn load(v: &toml::Value) -> Self {
        let w = |k: &str| get_field(v, k).and_then(toml_u32).unwrap_or(0);
        Self {
            ctl: get_field(v, "ctl").and_then(toml_u64).unwrap_or(CTRL_RESET),
            rptr: w("rptr"),
            wptr: w("wptr"),
            sub: w("sub"),
            msc: w("msc"),
            ust: w("ust"),
            start: None,
            done: get_field(v, "done").and_then(toml_u64).unwrap_or(0),
            rate: w("rate"),
            dry: w("dry"),
        }
    }
}

/// The control registers, the three channels and the codec's registers;
/// loading drives the interrupts again. The host output is the machine's.
impl crate::traits::Saveable for MaceAudio {
    fn save_state(&self) -> toml::Value {
        let st = self.st.lock();
        let mut t = toml::map::Map::new();
        t.insert("ctl".into(), hex_u64(st.ctl));
        t.insert("codec_ctl".into(), hex_u64(st.codec_ctl));
        t.insert("mask".into(), hex_u64(st.mask));
        t.insert("status_in".into(), hex_u64(st.status_in));
        t.insert("channels".into(), toml::Value::Array(st.ch.iter().map(Chan::save).collect()));
        t.insert("codec".into(), u16_slice_to_toml(&st.codec));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        {
            let u = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
            let mut st = self.st.lock();
            *st = State::new();
            st.ctl = u("ctl");
            st.codec_ctl = u("codec_ctl");
            st.mask = u("mask");
            st.status_in = u("status_in");
            if let Some(toml::Value::Array(a)) = get_field(v, "channels") {
                for (c, e) in st.ch.iter_mut().zip(a) {
                    *c = Chan::load(e);
                }
            }
            if let Some(c) = get_field(v, "codec") {
                load_u16_slice(c, &mut st.codec);
            }
        }
        self.update_irq();
        Ok(())
    }
}

impl crate::traits::BusDevice for MaceAudio {
    fn read8(&self, addr: u64) -> crate::traits::BusRead8 { crate::traits::BusRead8::ok(self.read(addr as u32) as u8) }
    fn read16(&self, addr: u64) -> crate::traits::BusRead16 { crate::traits::BusRead16::ok(self.read(addr as u32) as u16) }
    fn read32(&self, addr: u64) -> crate::traits::BusRead32 {
        let v = self.read(addr as u32);
        crate::traits::BusRead32::ok(if addr & 4 == 0 { (v >> 32) as u32 } else { v as u32 })
    }
    fn read64(&self, addr: u64) -> crate::traits::BusRead64 { crate::traits::BusRead64::ok(self.read(addr as u32)) }
    fn write8(&self, addr: u64, val: u8) -> u32 { self.write64(addr, val as u64) }
    fn write16(&self, addr: u64, val: u16) -> u32 { self.write64(addr, val as u64) }
    fn write32(&self, addr: u64, val: u32) -> u32 { self.write64(addr, val as u64) }
    /// A write can start, stop or acknowledge a channel, so the ISA bits
    /// are driven again after it.
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write(addr as u32, val);
        self.update_irq();
        crate::traits::BUS_OK
    }
}

impl crate::traits::Device for MaceAudio {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {
        self.meter.stopped();
    }
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    /// The counts are as of the last service pass: a query moves nothing.
    fn audio_report(&self) -> Option<String> {
        let rates = {
            let st = self.st.lock();
            // The converter's rate, while the channel that feeds it runs.
            let on = |n: usize, rate: Option<u32>| rate.filter(|_| st.ch[n].running());
            [on(0, st.adc_rate()), on(1, st.dac_rate(1)), on(2, st.dac_rate(2))]
        };
        let player = self.player.lock();
        Some(self.meter.report(&rates, player.as_ref().map(|p| &p.out), self.config.lock().is_some()))
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("audio".to_string(), "audio [status]: the audio card's report | audio codec: the AD1843 registers".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "audio" || args != ["codec"] {
            return Err("Usage: audio codec".to_string());
        }
        for (i, v) in self.st.lock().codec.iter().enumerate() {
            writeln!(w, "  AD1843 reg {i:2}: {v:#06x}").map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_codec(a: &MaceAudio, reg: u32, v: u16) {
        a.write(MACE_AUDIO + 0x08, (reg as u64) << 17 | v as u64);
    }

    fn read_codec(a: &MaceAudio, reg: u32) -> u16 {
        a.write(MACE_AUDIO + 0x08, (reg as u64) << 17 | 1 << 16);
        a.read(MACE_AUDIO + 0x18) as u16
    }

    #[test]
    fn a_pair_that_does_not_fit_is_dropped_whole() {
        // Three slots: the first pair fits and leaves one, which the left
        // sample of the second must not take.
        let (p, mut c) = rtrb::RingBuffer::<i16>::new(3);
        let mut out = HostOut::detached(Meter::new("test", &[]), 48000, p);
        push_mixed(&mut out, &[1, -2, 40000, -40000]);
        assert_eq!(std::iter::from_fn(|| c.pop().ok()).collect::<Vec<_>>(), [1, -2]);
        assert_eq!(out.meter.dropped(), 1, "the pair with no room is counted");
        push_mixed(&mut out, &[40000, -40000]);
        assert_eq!(std::iter::from_fn(|| c.pop().ok()).collect::<Vec<_>>(), [i16::MAX, i16::MIN]);
    }

    #[test]
    fn save_load_round_trip() {
        use crate::traits::Saveable;
        let src = MaceAudio::new();
        src.write(MACE_AUDIO, 0);
        write_codec(&src, 28, 0x3800);
        write_codec(&src, 17, 44100);
        {
            let mut st = src.st.lock();
            st.mask = 0x0f;
            st.status_in = 0x3;
            st.ch[1] = Chan { ctl: CTRL_ENABLE, rptr: 0x40, wptr: 0x200, sub: 2, msc: 1234, ust: 5678,
                              start: None, done: 99, rate: 44100, dry: 3 };
        }
        let v1 = src.save_state();
        let dst = MaceAudio::new();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MaceAudio save_state mismatch after load_state round-trip");
        assert_eq!(read_codec(&dst, 17), 44100, "the codec's clock rate is restored");
    }

    /// Out of reset the codec answers, reports itself initialised, and
    /// powers up when told to.
    #[test]
    fn the_codec_comes_up_and_powers_on() {
        let a = MaceAudio::new();
        assert_eq!(a.read(MACE_AUDIO) & 3, 3, "held in reset, codec present");
        a.write(MACE_AUDIO, 0);
        assert_eq!(a.read(MACE_AUDIO) & 3, 2, "out of reset");
        assert_eq!(read_codec(&a, 0) & 0x8000, 0, "INIT clear");
        assert_ne!(read_codec(&a, 0) & 0x4000, 0, "converters powered down");
        write_codec(&a, 28, 0x3800);
        assert_eq!(read_codec(&a, 0) & 0xc000, 0, "powered up");
        assert_eq!(read_codec(&a, 17), 48000, "clock 1 at 48 kHz from reset");
        write_codec(&a, 17, 44100);
        assert_eq!(read_codec(&a, 17), 44100);
    }

    /// Changing a running channel's rate restarts its pacing instead of
    /// wrapping the frame count (amesh, 48 to 32 kHz, hung the machine).
    #[test]
    fn a_rate_change_restarts_the_pacing() {
        let mut c = Chan { ctl: CTRL_ENABLE, ..Chan::RESET };
        let t0 = Instant::now();
        assert_eq!(c.due(t0, 48000), 0);
        let t1 = t0 + std::time::Duration::from_millis(100);
        assert_eq!(c.due(t1, 48000), 4800);
        let t2 = t1 + std::time::Duration::from_millis(1);
        assert_eq!(c.due(t2, 32000), 0, "new rate: count starts again");
        let t3 = t2 + std::time::Duration::from_millis(10);
        assert_eq!(c.due(t3, 32000), 320);
    }

    /// With the ADC running, the input ring fills with silence (0xff in
    /// each half) up to one block short of full.
    #[test]
    fn the_input_ring_fills_with_silence() {
        let ram = O2Memory::single(8 << 20);
        let a = MaceAudio::new();
        a.write(MACE_AUDIO, 0);
        write_codec(&a, 28, 0x1000); // clock 2 on, converters up
        write_codec(&a, 20, 48000);
        write_codec(&a, 15, 2 << 2 | 2); // ADCs from clock 2
        write_codec(&a, 27, 3); // ADCs on
        a.write(MACE_AUDIO + 0x20, 0);
        a.write(MACE_AUDIO + 0x20, CTRL_ENABLE | 6 << 5); // interrupt when full
        let ring_base = 0x0010_0000u32;
        let io = AudioIo { ram: &ram, ring_base, ust: 1 };
        a.poll(&io);
        std::thread::sleep(std::time::Duration::from_millis(30));
        a.poll(&io);
        let wp = a.read(MACE_AUDIO + 0x30) as u32;
        assert!(wp > 0, "the engine wrote");
        let m = &ram;
        assert_eq!(m.get_be(m.dma_offset(ring_base).unwrap(), 4), 0xff);
        std::thread::sleep(std::time::Duration::from_millis(250));
        a.poll(&io);
        assert_eq!(a.read(MACE_AUDIO + 0x38) as u32, PTR_MASK, "full: one block short");
        assert_ne!(a.isa_bits() & 0x04, 0, "the full threshold raised the input bit");
    }

    /// The DAC gain register attenuates by 1.5 dB a step and mutes.
    #[test]
    fn dac_gain_attenuates_and_mutes() {
        let a = MaceAudio::new();
        a.write(MACE_AUDIO, 0);
        let gains = |a: &MaceAudio| a.st.lock().dac_gain(1);
        write_codec(&a, 9, 0x0808); // unity
        assert_eq!(gains(&a), (1.0, 1.0));
        write_codec(&a, 9, 0x0c0c); // 6 dB down
        let (l, r) = gains(&a);
        assert!((l - 0.501).abs() < 0.01 && (r - 0.501).abs() < 0.01);
        write_codec(&a, 9, 0x0000); // IRIX's loudest: +12 dB
        assert!((gains(&a).0 - 3.98).abs() < 0.02);
        write_codec(&a, 9, 0x0808);
        write_codec(&a, 11, 0x0400); // left digitally 6 dB down
        assert!((gains(&a).0 - 0.501).abs() < 0.01 && gains(&a).1 == 1.0);
        write_codec(&a, 9, 0x8808); // left muted
        assert_eq!(gains(&a).0, 0.0);
    }

    /// An output channel plays its ring at its DAC's rate, the hardware
    /// read pointer follows, and the threshold bit says when it runs low.
    #[test]
    fn an_output_ring_plays_at_the_dac_rate() {
        let ram = O2Memory::single(8 << 20);
        let a = MaceAudio::new();
        a.write(MACE_AUDIO, 0);
        write_codec(&a, 28, 0x0800); // clock 1 on, converters up
        write_codec(&a, 17, 8000);
        write_codec(&a, 15, 1 << 8); // DAC1 from clock 1
        write_codec(&a, 27, 1 << 8); // DAC1 on
        let ring_base = 0x0010_0000u32;
        for i in 0..64u32 {
            let s = ((i as i32 * 100) << 8) as u32;
            let m = &ram;
            let o = m.dma_offset(ring_base | 0x1000 | i * 8).unwrap();
            m.put_be(o, 4, s);
        }
        // Channel 1 (output #1): 2 KB of data, interrupt below 50%.
        a.write(MACE_AUDIO + 0x40, 2 << 5);
        a.write(MACE_AUDIO + 0x40, CTRL_ENABLE | 2 << 5);
        a.write(MACE_AUDIO + 0x50, 0x800);
        assert_eq!(a.read(MACE_AUDIO + 0x58), 0x800);
        assert_eq!(a.isa_bits() & 0x10, 0, "half full is not below half");
        let io = AudioIo { ram: &ram, ring_base, ust: 7 };
        a.poll(&io);
        std::thread::sleep(std::time::Duration::from_millis(60));
        a.poll(&io);
        // 8000 Hz for ~60 ms: ~480 pairs, ~120 blocks... capped by the ring.
        let rp = a.read(MACE_AUDIO + 0x48) as u32;
        assert!(rp > 0, "the engine moved its read pointer");
        assert_eq!(a.read(MACE_AUDIO + 0x58) as u32, 0x800u32.wrapping_sub(rp) & PTR_MASK);
        assert_ne!(a.isa_bits() & 0x10, 0, "below half now");
        let mu = a.msc_ust(1);
        assert!((mu >> 32) as u32 > 0 && mu as u32 == 7, "MSC counts in the high word, UST stamped in the low");
    }

    /// Stopping the machine ends playing: the host's silence while it is
    /// stopped is not an underrun.
    #[test]
    fn stopping_the_machine_ends_playing() {
        use crate::traits::Device;
        let a = MaceAudio::new();
        let (prod, mut cons) = rtrb::RingBuffer::<i16>::new(64);
        let mut out = HostOut::detached(a.meter.clone(), 48000, prod);
        out.push_frame(&[1, 2]);
        cons.read_chunk(cons.slots()).unwrap().commit_all();
        let underrun = |cons: &mut rtrb::Consumer<i16>| {
            crate::host_audio::count_underrun(crate::host_audio::pop_frames(cons, 2, &mut [0i16; 2], |v| v), &a.meter);
            a.meter.underruns()
        };
        assert_eq!(underrun(&mut cons), 1, "playing");
        a.stop();
        assert_eq!(underrun(&mut cons), 1, "stopped");
    }

    /// The frames each output played on the pacing clock reach the card's
    /// meter and `audio status`.
    #[test]
    fn audio_status_counts_each_outputs_frames() {
        let ram = O2Memory::single(8 << 20);
        let a = MaceAudio::new();
        a.write(MACE_AUDIO, 0);
        write_codec(&a, 28, 0x0800);
        write_codec(&a, 17, 8000);
        write_codec(&a, 15, 1 << 8);
        write_codec(&a, 27, 1 << 8);
        a.write(MACE_AUDIO + 0x40, CTRL_ENABLE);
        let io = AudioIo { ram: &ram, ring_base: 0x0010_0000, ust: 0 };
        let t0 = Instant::now();
        a.poll_at(&io, t0);
        a.poll_at(&io, t0 + std::time::Duration::from_millis(100));
        assert_eq!(
            a.audio_report().unwrap(),
            "audio: MACE audio\n\
             \x20 analog in: 0 frames moved, guest rate off\n\
             \x20 analog out 1: 800 frames moved, guest rate 8000 Hz\n\
             \x20 analog out 2: 0 frames moved, guest rate off\n\
             \x20 host underruns: 0 frames\n\
             \x20 dropped by the host: 0 frames\n\
             \x20 host output: none (no_audio)\n"
        );
        use crate::traits::Device;
        assert!(a.execute_command("audio", &["codec"], Box::new(std::io::sink())).is_ok());
        assert!(a.execute_command("audio", &["status"], Box::new(std::io::sink())).is_err(), "status is the monitor's");
    }

    /// IRIX feeds an output ring by reading the read pointer about once a
    /// millisecond and setting the write pointer 17 frames past it at
    /// 8 kHz, which the block pointer rounds down to 16 (2 ms). The service
    /// thread passes about every 2.1 ms when the machine is quiet. Reading
    /// a pointer last moved by such a pass, IRIX fell behind: a pass wanted
    /// 16 or 17 frames, sometimes found fewer and played zeros in the middle
    /// of the tone, so 8 and 11 kHz played sharp and long. A register read
    /// brings the ring up to date first, and it never runs dry.
    #[test]
    fn a_register_read_catches_the_ring_up_so_irix_keeps_ahead() {
        let us = std::time::Duration::from_micros;
        let run = |catch_up: bool| {
            let ram = O2Memory::single(8 << 20);
            let a = MaceAudio::new();
            a.write(MACE_AUDIO, 0);
            write_codec(&a, 28, 0x0800);
            write_codec(&a, 17, 8000);
            write_codec(&a, 15, 1 << 8);
            write_codec(&a, 27, 1 << 8);
            let ring_base = 0x0010_0000u32;
            for i in 0..1024u32 {
                let o = ram.dma_offset(ring_base | 0x1000 | i * 4).unwrap();
                ram.put_be(o, 4, 1000 << 8);
            }
            a.write(MACE_AUDIO + 0x40, CTRL_ENABLE);
            let io = AudioIo { ram: &ram, ring_base, ust: 0 };
            let t0 = Instant::now();
            for step in 0..4000u64 {
                let t = t0 + us(step * 50);
                if step % 20 == 0 {
                    if catch_up {
                        a.catch_up_at(&io, t);
                    }
                    let rptr = a.read(MACE_AUDIO + 0x48);
                    a.write(MACE_AUDIO + 0x50, rptr + 17 * 8);
                }
                if step % 42 == 0 {
                    a.poll_at(&io, t);
                }
            }
            (a.gaps[0].load(Ordering::Relaxed), a.msc_ust(1) >> 32)
        };
        let (gaps, _) = run(false);
        assert!(gaps > 0, "a pointer only as fresh as the last pass let the ring run dry");
        let (gaps, frames) = run(true);
        assert_eq!(gaps, 0, "the ring never ran dry");
        assert!((1590..=1600).contains(&frames), "200 ms at 8 kHz played, got {frames}");
    }

    /// The pairs DAC 1's clock played as zeros because the guest let the ring
    /// run empty: a service pass held up for longer than the ring's 10 ms
    /// outruns it. The card moved them, but they carry no sound.
    fn dry_pairs(a: &MaceAudio) -> u64 {
        a.gap_items[0].load(Ordering::Relaxed) + u64::from(a.st.lock().ch[1].dry)
    }

    /// The whole way from guest memory to the host as the O2 runs it: an
    /// output ring the guest keeps full, played by the machine's service
    /// pass on the host's clock, into the host output `enable_output`
    /// opens -- an `[audio] file`, so no sound card -- drained in real time.
    /// At 48 kHz nothing is resampled: what the file holds, less the
    /// underrun silence, is the ring sample for sample with each channel on
    /// its own side. The DAC mute bits silence what follows.
    #[test]
    fn a_tone_in_guest_memory_reaches_the_host_on_the_real_clock() {
        use crate::host_audio::test_signal::*;
        let ram = O2Memory::single(8 << 20);
        let a = MaceAudio::new();
        let (cfg, path) = file_config("o2-tone");
        a.enable_output(cfg);
        a.write(MACE_AUDIO, 0);
        write_codec(&a, 28, 0x0800); // clock 1 on, converters up
        write_codec(&a, 17, 48000);
        write_codec(&a, 15, 1 << 8); // DAC1 from clock 1
        write_codec(&a, 27, 1 << 8); // DAC1 on
        write_codec(&a, 9, 0x0808); // unity gain
        // Five and ten cycles fill the 512-pair ring exactly.
        let (left, right) = (5.0 * 48000.0 / 512.0, 10.0 * 48000.0 / 512.0);
        let tone = tones(512, 48000, left, right, 16000.0);
        let ring_base = 0x0010_0000u32;
        for (i, &v) in tone.iter().enumerate() {
            let o = ram.dma_offset(ring_base | 0x1000 | i as u32 * 4).unwrap();
            ram.put_be(o, 4, (i32::from(v) << 8) as u32);
        }
        a.write(MACE_AUDIO + 0x40, CTRL_ENABLE);
        let io = AudioIo { ram: &ram, ring_base, ust: 0 };
        // The guest keeps the ring full and the service pass plays it. The
        // host can hold this thread up, so what the test expects of the
        // clock is what its passes handed it: the time between one and the
        // next, less the part of a stall the clock skips by design (`Chan::due`
        // moves at most a quarter second of it), not the wall time from
        // start to end.
        let lag = std::time::Duration::from_millis(250);
        let (mut last, mut credit) = (None::<Instant>, std::time::Duration::ZERO);
        let mut run = |ms: u64| {
            let end = Instant::now() + std::time::Duration::from_millis(ms);
            while Instant::now() < end {
                let rptr = a.read(MACE_AUDIO + 0x48) as u32;
                a.write(MACE_AUDIO + 0x50, u64::from(rptr.wrapping_sub(BLOCK) & PTR_MASK));
                let now = Instant::now();
                credit += last.map_or(std::time::Duration::ZERO, |t| (now - t).min(lag));
                last = Some(now);
                a.poll_at(&io, now);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        };
        run(600);
        let (loud, dry_loud, refused_loud) = (a.msc_ust(1) >> 32, dry_pairs(&a), a.meter.dropped());
        write_codec(&a, 9, 0x8888); // both sides muted
        run(300);
        let moved = a.msc_ust(1) >> 32;
        let dropped = a.meter.dropped();
        // Dropping the output completes the file.
        a.player.lock().take();
        let (rate, channels, pcm) = read_wav(&path.0);
        assert_eq!((rate, channels), (48000, 2));
        // Half at least of what the passes allowed: a clock that does not
        // keep to the host's holds the count back.
        assert!(moved as f64 >= 0.5 * credit.as_secs_f64() * 48000.0, "the clock ran: {moved} pairs in {:.3} s of passes", credit.as_secs_f64());
        let heard = sounding(&pcm, 2);
        let frames = (heard.len() / 2) as u64;
        // The ring's ten-millisecond headroom is the guest's to keep; the
        // zeros the clock played when it ran out, and the pairs the host
        // ring refused after a stall, are counted, not heard.
        assert!(frames <= loud, "{frames} pairs heard of {loud} played before the mute");
        assert!(
            frames + dry_loud + refused_loud + 9600 >= loud,
            "{frames} pairs heard of {loud} played before the mute, {dry_loud} of them from an empty ring and {refused_loud} refused by the host ring"
        );
        // A stall of a fifth of a second or more makes the clock's catch-up
        // overflow the host's ring, and the frames it refuses leave a hole
        // in the tone: the order cannot be told then.
        if dropped != 0 {
            eprintln!("not checking the order or the pitch: the host ring refused {dropped} frames after a stall");
            return;
        }
        let want: Vec<i16> = sounding(&tone, 2).into_iter().cycle().take(heard.len()).collect();
        assert_eq!(heard, want, "the ring's samples, in order, on their own channels");
        assert_tones(&heard, 2, 48000, left, right, 15000);
    }

    use crate::host_audio::test_signal::{a_tone_survives_a_pause, tones, CardUnderTest};

    /// The O2 playing a tone, the guest stopping the output for 0.3 s so the
    /// host stream runs dry, then starting it again (see
    /// [`a_tone_survives_a_pause`]). The 512-pair ring holds five and ten
    /// whole cycles, so the tones are 468.75 and 937.5 Hz rather than 440
    /// and 880.
    struct O2 {
        a: MaceAudio,
        ram: O2Memory,
    }

    const O2_RING_BASE: u32 = 0x0010_0000;
    const O2_TONE: (f64, f64) = (5.0 * 48000.0 / 512.0, 10.0 * 48000.0 / 512.0);

    impl CardUnderTest for O2 {
        const NAME: &'static str = "o2-underrun";
        const PAUSE_IS_STALL: bool = false;
        fn build(cfg: &AudioConfig) -> Self {
            let ram = O2Memory::single(8 << 20);
            let a = MaceAudio::new();
            a.enable_output(cfg.clone());
            a.write(MACE_AUDIO, 0);
            write_codec(&a, 28, 0x0800);
            write_codec(&a, 17, 48000);
            write_codec(&a, 15, 1 << 8);
            write_codec(&a, 27, 1 << 8);
            write_codec(&a, 9, 0x0808);
            for (i, &v) in tones(512, 48000, O2_TONE.0, O2_TONE.1, 16000.0).iter().enumerate() {
                let o = ram.dma_offset(O2_RING_BASE | 0x1000 | i as u32 * 4).unwrap();
                ram.put_be(o, 4, (i32::from(v) << 8) as u32);
            }
            Self { a, ram }
        }
        fn tone(&self) -> (f64, f64) {
            O2_TONE
        }
        fn rate(&self) -> f64 {
            48000.0
        }
        fn play(&self) {
            self.a.write(MACE_AUDIO + 0x40, CTRL_ENABLE);
        }
        fn pause(&self) {
            self.a.write(MACE_AUDIO + 0x40, 0);
        }
        /// The guest keeps the ring full and the service pass plays it, on
        /// or off.
        fn wait(&self, ms: u64) {
            let io = AudioIo { ram: &self.ram, ring_base: O2_RING_BASE, ust: 0 };
            let end = Instant::now() + std::time::Duration::from_millis(ms);
            while Instant::now() < end {
                let rptr = self.a.read(MACE_AUDIO + 0x48) as u32;
                self.a.write(MACE_AUDIO + 0x50, u64::from(rptr.wrapping_sub(BLOCK) & PTR_MASK));
                self.a.poll(&io);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }
        /// The pairs of the guest's ring, not the zeros of a pass that
        /// found it empty.
        fn frames(&self) -> f64 {
            self.a.meter.frames(1).saturating_sub(dry_pairs(&self.a)) as f64
        }
        fn meter(&self) -> &Meter {
            &self.a.meter
        }
        fn finish(&self) {
            self.a.player.lock().take();
        }
    }

    #[test]
    fn a_tone_survives_a_pause_into_a_file() {
        a_tone_survives_a_pause::<O2>(crate::host_audio::test_signal::Backend::File);
    }

    #[test]
    fn a_tone_survives_a_pause_on_the_real_device() {
        a_tone_survives_a_pause::<O2>(crate::host_audio::test_signal::Backend::RealDevice);
    }
}
