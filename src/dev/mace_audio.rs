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
use crate::host_audio::{HostOut, Resampler};

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
        Self { ctl: 1, codec_ctl: 1 << 16, mask: 0, status_in: 0, ch: [Chan::RESET; 3], codec }
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
    underruns: Arc<AtomicU64>,
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
            underruns: Arc::new(AtomicU64::new(0)),
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
                    let keep = st.ch;
                    *st = State { ch: keep, ..State::new() };
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
        let mut st = self.st.lock();
        let now = Instant::now();
        let mut pcm: [Vec<(i16, i16)>; 2] = [Vec::new(), Vec::new()];
        let mut rates = [0u32; 2];
        let ram = &*io.ram;
        // The input: no host capture yet, so the ADC delivers silence at its
        // rate, as a quiet microphone would. A recording program (amesh)
        // then sees its ring fill instead of waiting for ever.
        match st.adc_rate() {
            Some(rate) if st.ch[0].running() => {
                let c = &mut st.ch[0];
                for _ in 0..c.due(now, rate) {
                    let next = (c.wptr + BLOCK) & PTR_MASK;
                    if c.sub == 0 && next == c.rptr {
                        break; // full: the engine never fills the last slot
                    }
                    let addr = io.ring_base | c.wptr | c.sub * 8;
                    if let Some(o) = ram.dma_offset(addr).filter(|&o| o + 8 <= ram.len()) {
                        // Each half: sign extension, the sample, 0xff.
                        ram.put_be(o, 4, 0xff);
                        ram.put_be(o + 4, 4, 0xff);
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
            for _ in 0..c.due(now, rate) {
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
                    if let Some(o) = ram.dma_offset(addr).filter(|&o| o + 8 <= ram.len()) {
                        let half = |w: u32| ((w as i32) >> 8).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                        let (l, r) = (half(ram.get_be(o, 4)), half(ram.get_be(o + 4, 4)));
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
        if pcm.iter().all(|p| p.is_empty()) {
            return;
        }
        self.dump(&pcm, &rates);
        self.play(&pcm, &rates);
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
                *player = crate::host_audio::open_output(&cfg, self.underruns.clone(), Arc::new(AtomicBool::new(true)), ring, "O2 audio", "IRIS_O2_AUDIO_CAPTURE")
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
                rs.push(l, r, &mut prod);
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
        for v in mixed {
            let _ = p.out.producer.push(v.clamp(i16::MIN as i32, i16::MAX as i32) as i16);
        }
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
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("audio".to_string(), "MACE audio + AD1843: audio status | audio codec".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write;
        if cmd != "audio" {
            return Err("Command not found".to_string());
        }
        let st = self.st.lock();
        match args.first().copied().unwrap_or("status") {
            "status" => {
                writeln!(w, "MACE audio: control {:#x}, codec control {:#x}, mask {:#x}, host output {}",
                    st.ctl, st.codec_ctl, st.mask, if self.player.lock().is_some() { "open" } else { "closed" }).ok();
                for (n, name) in ["input", "output 1", "output 2"].iter().enumerate() {
                    let c = &st.ch[n];
                    writeln!(w, "  {name}: control {:#x} read {:#05x} write {:#05x} depth {} MSC {} {}",
                        c.ctl, c.rptr, c.wptr, c.depth(), c.msc, if c.running() { "running" } else { "stopped" }).ok();
                }
                let rate = |r: Option<u32>| r.map_or("off".to_string(), |r| format!("{r} Hz"));
                writeln!(w, "  rates: DAC1 {} DAC2 {} ADC {}", rate(st.dac_rate(1)), rate(st.dac_rate(2)), rate(st.adc_rate())).ok();
            }
            "codec" => {
                for (i, v) in st.codec.iter().enumerate() {
                    writeln!(w, "  AD1843 reg {i:2}: {v:#06x}").ok();
                }
            }
            _ => return Err("Usage: audio status | codec".to_string()),
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
}
