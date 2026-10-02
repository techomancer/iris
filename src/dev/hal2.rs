use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::devlog::LogModule;
use std::time::{Duration, Instant};
use std::io::Write;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, Device, DmaClient};
use crate::hptimer::{TimerManager, TimerId, TimerReturn};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::{RingBuffer, Producer};

// HAL2 Register Offsets (relative to 0x1FBD8000)
pub const HAL2_ISR: u32 = 0x10; // Interrupt Status Register
pub const HAL2_REV: u32 = 0x20; // Revision

pub const HAL2_IAR: u32 = 0x30; // Indirect Address Register
pub const HAL2_IDR0: u32 = 0x40; // Indirect Data Register 0
pub const HAL2_IDR1: u32 = 0x50; // Indirect Data Register 1
pub const HAL2_IDR2: u32 = 0x60; // Indirect Data Register 2
pub const HAL2_IDR3: u32 = 0x70; // Indirect Data Register 3

pub mod isr {
    pub const TSTATUS: u8       = 0x01; // r  transaction busy
    pub const USTATUS: u8       = 0x02; // r  utime armed
    pub const CODEC_MODE: u8    = 0x04; // rw 0=indigo, 1=quad
    pub const GLOBAL_RESET_N: u8 = 0x08; // rw 0=reset entire chip
    pub const CODEC_RESET_N: u8  = 0x10; // rw 0=reset codec/synth only
}

pub mod iar {
    pub const WR: u8 = 0x80;
    pub const RD: u8 = 0x40;
    pub const ADDR_MASK: u8 = 0x0F;
}

// Indirect registers
pub const HAL2_I_STATUS: u8 = 0x00;
pub const HAL2_I_CONTROL: u8 = 0x01;
pub const HAL2_I_PBUS_CH1: u8 = 0x02;
pub const HAL2_I_PBUS_CH2: u8 = 0x03;
pub const HAL2_I_PBUS_CH3: u8 = 0x04;
pub const HAL2_I_PBUS_CH4: u8 = 0x05;
pub const HAL2_I_DMA_END_CH1: u8 = 0x06;
pub const HAL2_I_DMA_END_CH2: u8 = 0x07;
pub const HAL2_I_DMA_END_CH3: u8 = 0x08;
pub const HAL2_I_DMA_END_CH4: u8 = 0x09;
pub const HAL2_I_DMA_DRV_CH1: u8 = 0x0A;
pub const HAL2_I_DMA_DRV_CH2: u8 = 0x0B;
pub const HAL2_I_DMA_DRV_CH3: u8 = 0x0C;
pub const HAL2_I_DMA_DRV_CH4: u8 = 0x0D;
pub const HAL2_I_AES_RX: u8 = 0x0E;
pub const HAL2_I_AES_TX: u8 = 0x0F;

pub mod control {
    pub const DEC_RESET_N: u16 = 0x0001;
    pub const INC_RESET_N: u16 = 0x0002;
    pub const SYN_RESET_N: u16 = 0x0004;
    pub const AES_RX_RESET_N: u16 = 0x0008;
    pub const AES_TX_RESET_N: u16 = 0x0010;
    pub const DAC_RESET_N: u16 = 0x0020;
    pub const ADC_RESET_N: u16 = 0x0040;
    pub const UTO_RESET_N: u16 = 0x0080;
}

// IAR field masks
// Bit 7 (0x0080) selects read vs write; NOT bit 15.
// type = bits 15:12, num = bits 11:8, access_sel = bit 7, param = bits 3:2
const IAR_ACCESS_READ: u16  = 0x0080;   // bit 7: 1=read, 0=write
const IAR_TYPE_MASK: u16    = 0xF000;
const IAR_NUM_MASK: u16     = 0x0F00;
const IAR_PARAM_MASK: u16   = 0x000C;   // bits 3:2

// IAR type field values (bits 15:12)
const IAR_TYPE_DMA: u16        = 0x1000; // codec / AES / synth DMA control
const IAR_TYPE_BRES: u16       = 0x2000; // Bresenham clock generators
const IAR_TYPE_GLOBAL_DMA: u16 = 0x9000; // global DMA enable/drive/endian/relay

// IAR num field values for IAR_TYPE_DMA (bits 11:8)
const IAR_NUM_AES_RX: u16  = 0x0200;
const IAR_NUM_AES_TX: u16  = 0x0300;
const IAR_NUM_CODECA: u16  = 0x0400;
const IAR_NUM_CODECB: u16  = 0x0500;

// IAR param values (from bits 3:2 of the IAR word)
// param 0 = relay/endian/special, param 1 = ctrl1, param 2 = ctrl2, param 3 = drive
const IAR_PARAM_0: u16 = 0x00;
const IAR_PARAM_1: u16 = 0x04;   // HAL2_*_CTRL1_W = ...04
const IAR_PARAM_2: u16 = 0x08;   // HAL2_*_CTRL2_W = ...08
const IAR_PARAM_3: u16 = 0x0C;

// DMA enable register bits (HAL2_DMA_ENABLE_W)
const DMA_EN_AES_RX: u16 = 0x02;
const DMA_EN_AES_TX: u16 = 0x04;
const DMA_EN_CODECA: u16 = 0x08;
const DMA_EN_CODECB: u16 = 0x10;

// Codec CTRL1 bitfield positions
const CTRL1_CHAN_MASK:  u16 = 0x0007; // bits 2:0 – HPC3 DMA channel
const CTRL1_CLOCK_SHIFT: u32 = 3;    // bits 4:3 – CLKID: the BRES generator
                                     // NUMBER, 1..3 (0 = none). Not an index.
const CTRL1_CLOCK_MASK:  u16 = 0x0003;
const CTRL1_MODE_SHIFT:  u32 = 8;    // bits 9:8 – channel mode
const CTRL1_MODE_MASK:   u16 = 0x0003;

// Channel mode values (CTRL1 bits 9:8)
const MODE_MONO:   usize = 1;
const MODE_STEREO: usize = 2;
const MODE_QUAD:   usize = 3;

// Pre-buffer: accumulate this many ms of audio samples before pushing to the ring.
// This gives the CPU time to fill its circular DMA buffer before we start draining it,
// preventing initial underrun.
const PREBUF_MS: u64 = 20;
// Ring buffer capacity as a multiple of PREBUF_MS.  Must absorb OS scheduling jitter.
// Expressed as a multiplier of PREBUF_MS stereo samples.
const RING_BUF_MULTIPLIER: usize = 16;

// Consecutive dry reads before giving up prebuf and opening stream anyway.
const DRY_LIMIT: u32 = 100;

/// How often a running channel wakes to move samples. A timer per sample
/// (every 21-91 us) cannot keep real time on a host thread: every late or
/// coalesced tick consumed guest audio more slowly than the codec rate, which
/// played as stretched, ever more delayed sound. Instead each wake moves the
/// number of frames the wall clock says are due.
///
/// It must be well under a millisecond. IRIX refills its 202-frame playback
/// ring from a 1 kHz callback that reads the DMA position each time; at 2 ms
/// the position stood still on every other read, and IRIX let the DMA lap
/// the ring: once a lap it wrote only the lap's remainder, and the rest
/// replayed stale audio (a 199-frame jump back every 20 ms at 11025 Hz, 6 ms
/// at 44.1 kHz -- slow, robotic sound).
const PACE_PERIOD: Duration = Duration::from_micros(250);
/// After a host stall, frames more than this far behind are skipped rather
/// than moved in one burst (which would only add latency).
const PACE_MAX_BACKLOG: Duration = Duration::from_millis(100);
/// How much faster than real time a channel may catch up after a late wake
/// (5/4): see `Pacer::due`.
const PACE_CATCHUP_NUM: u64 = 5;
/// How far (in milliseconds of audio) codec A may read past the DMA position
/// the guest last polled. IRIX refills its 202-frame playback ring from a
/// 1 kHz callback, writing only a little ahead of the position it reads.
/// Emulated, that callback runs late when the host is busy (GLQuake), and a
/// channel paced by the wall clock alone then ran into slots not yet
/// refilled and replayed the previous lap (18 ms chunks at 11025 Hz: the
/// "robot" sound). Coupled to the guest's polls, the channel waits for a
/// late callback instead; the host output buffer covers the wait. A guest
/// that does not poll the position (no poll for 50 ms) is not held back.
const READ_AHEAD_MS: u64 = 1;
const PACE_CATCHUP_DEN: u64 = 4;

/// Wall-clock frame pacing for one channel: how many frames are due now.
struct Pacer {
    start: Instant,
    rate: u64,
    done: u64,
}

impl Pacer {
    fn new(rate: u32) -> Self {
        Self { start: Instant::now(), rate: rate as u64, done: 0 }
    }

    /// Frames to move now, counting them as done: what the wall clock says is
    /// due, but never much more than one period's worth.
    ///
    /// The DMA position must move smoothly, as on the hardware. IRIX writes
    /// its playback ring only a few milliseconds ahead of the position it
    /// polls; when a late wake (a busy host) moved everything due at once,
    /// the position leapt past what IRIX had written, the channel read the
    /// previous lap's samples, and sound under load (GLQuake) repeated in
    /// 18 ms chunks. A late wake now moves at most PACE_CATCHUP times a
    /// period's frames and the backlog drains over the next wakes.
    fn due(&mut self) -> u64 {
        let elapsed = self.start.elapsed();
        let target = (elapsed.as_nanos() as u64).saturating_mul(self.rate) / 1_000_000_000;
        let backlog = target.saturating_sub(self.done);
        let max_backlog = self.rate * PACE_MAX_BACKLOG.as_millis() as u64 / 1000;
        if backlog > max_backlog {
            // Skip the stall rather than replay it.
            self.done = target - max_backlog;
        }
        let per_period = (self.rate * PACE_PERIOD.as_micros() as u64).div_ceil(1_000_000);
        let cap = (per_period * PACE_CATCHUP_NUM).div_ceil(PACE_CATCHUP_DEN).max(per_period + 1);
        let due = target.saturating_sub(self.done).min(cap);
        self.done += due;
        due
    }

    /// Frames `due` handed out but not moved after all: still due.
    fn give_back(&mut self, frames: u64) {
        self.done -= frames;
    }
}

// Sample rates to try when opening the persistent output stream, in order.
const PREFERRED_RATES: &[u32] = &[48000, 44100, 22050];

// ─── Audio output (owned by Codec A, opened once at start, closed at stop) ───

// Opened once at `start()` at the best available host rate; the codec A timer
// pushes i16 stereo pairs through a resampler into the ring buffer producer.
// The stream plays silence when the ring is empty (cpal fills with 0).
struct AudioOut {
    stream_rate: u32,
    producer: Producer<i16>,
    underruns: Arc<AtomicU64>,
    // Set once prebuffering finishes and real samples are flowing; cleared when
    // the codec is disarmed/reset. Gates underrun counting so idle silence
    // (stream open, nothing enabled yet) isn't reported as an underrun.
    playing: Arc<AtomicBool>,
    // Keep stream alive; dropped when AudioOut is dropped at stop().
    _stream: cpal::Stream,
}

// cpal::Stream is !Send/!Sync on some platforms (ALSA uses raw pointers internally),
// but it is safe to hold inside a Mutex.
unsafe impl Send for AudioOut {}
unsafe impl Sync for AudioOut {}

// Simple skip/repeat resampler using a fixed-point accumulator.
// Produces output at `out_rate` from input at `in_rate`.
// Call `push_sample` for every input sample pair; it pushes 0, 1, or 2 pairs to the ring.
/// Converts the codec's rate to the host stream's by Catmull-Rom interpolation.
///
/// It used to repeat the last sample (zero-order hold). From 44.1 or 48 kHz
/// to 48 kHz that is nearly harmless, but Quake plays at 11025 Hz, where each
/// sample became a 4-or-5-sample step: the staircase put loud images of every
/// sound around 11 kHz and its multiples, and the games sounded metallic and
/// robotic. A real DAC at 11025 Hz filters those images out; interpolating
/// through the samples does most of the same.
struct Resampler {
    in_rate: u32,
    out_rate: u32,
    // Position of the next output between h[1] and h[2], in 1/out_rate
    // input-sample units: an exact rational step, so no drift.
    acc: u64,
    // The last four input frames, oldest first; output is interpolated
    // between h[1] and h[2] (two input samples of latency).
    h: [[f32; 2]; 4],
}

impl Resampler {
    fn new(in_rate: u32, out_rate: u32) -> Self {
        Self { in_rate, out_rate, acc: 0, h: [[0.0; 2]; 4] }
    }

    fn passthrough(&self) -> bool { self.in_rate == self.out_rate }

    /// Push one input stereo pair; emits the output pairs it completes.
    fn push(&mut self, l: i16, r: i16, prod: &mut Producer<i16>) {
        if self.passthrough() {
            let _ = prod.push(l);
            let _ = prod.push(r);
            return;
        }
        self.h = [self.h[1], self.h[2], self.h[3], [l as f32, r as f32]];
        let out = self.out_rate as u64;
        while self.acc < out {
            let t = self.acc as f32 / out as f32;
            for c in 0..2 {
                let v = catmull_rom(self.h[0][c], self.h[1][c], self.h[2][c], self.h[3][c], t);
                let _ = prod.push(v.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16);
            }
            self.acc += self.in_rate as u64;
        }
        self.acc -= out;
    }
}

/// The Catmull-Rom spline through p1 and p2 at t in [0, 1).
fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * (2.0 * p1
        + (p2 - p0) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3)
}

// ─── Per-channel mutable state, lives inside a Mutex ─────────────────────────

struct CodecAState {
    // AudioOut is opened once at start() and lives until stop().
    // None only before start() or after stop().
    out: Option<AudioOut>,
    // Resampler from codec rate → stream rate.  Built (or rebuilt) when
    // codec rate first becomes known or changes.
    resampler: Option<Resampler>,
    // True while we're still filling the initial prebuffer before feeding the ring.
    prebuffering: bool,
    prebuf: Vec<i16>,
    dry: u32,
    nonzero_seen: bool,
    timer_id: Option<TimerId>,
    /// The DMA channel the armed timer reads, and what it has done since
    /// it was armed: callbacks, frames read, reads that found no data.
    armed_ch: Option<usize>,
    calls: u64,
    frames: u64,
    dry_reads: u64,
}

impl CodecAState {
    fn new() -> Self {
        Self { out: None, resampler: None, prebuffering: true, prebuf: Vec::new(),
               dry: 0, nonzero_seen: false, timer_id: None,
               armed_ch: None, calls: 0, frames: 0, dry_reads: 0 }
    }
    fn reset_audio(&mut self) {
        // Keep `out` — stream stays open.  Just reset codec-side state.
        self.resampler = None;
        self.prebuffering = true;
        self.prebuf.clear();
        self.dry = 0;
        self.nonzero_seen = false;
        self.armed_ch = None;
        self.calls = 0;
        self.frames = 0;
        self.dry_reads = 0;
        if let Some(o) = &self.out {
            o.playing.store(false, Ordering::Relaxed);
        }
    }
    /// Push interleaved i16 stereo pairs through the resampler into the ring buffer.
    fn push_to_ring(&mut self, samples: &[i16]) {
        if let Some(rs) = &mut self.resampler {
            if let Some(o) = &mut self.out {
                for chunk in samples.chunks_exact(2) {
                    rs.push(chunk[0], chunk[1], &mut o.producer);
                }
            }
        }
    }
}

struct CodecBState {
    timer_id: Option<TimerId>,
}

struct AesTxState {
    timer_id: Option<TimerId>,
}

struct AesRxState {
    loopback: std::collections::VecDeque<u32>,
    timer_id: Option<TimerId>,
}

// ─── HAL2 register state ──────────────────────────────────────────────────────

#[derive(Default)]
struct Hal2State {
    isr: u16,
    iar: u16,
    idr: [u16; 4],
    /// The last indirect accesses (IAR, then IDR0..3 as they stood), for
    /// `hal2 status`.
    iar_log: std::collections::VecDeque<(u16, [u16; 4])>,

    // Internal Registers
    // ctrl[0] = CTRL1 (IDR0), ctrl[1] = CTRL2 IDR0, ctrl[2] = CTRL2 IDR1
    codeca_ctrl: [u16; 3],
    codecb_ctrl: [u16; 3],
    aestx_ctrl: [u16; 3],
    aesrx_ctrl: [u16; 3],

    bres_clock_sel: [u16; 3],
    bres_clock_inc: [u16; 3],
    bres_clock_modctrl: [u16; 3],
    bres_clock_rate: [u32; 3],

    dma_enable: u16,
    dma_drive: u16,
    dma_endian: u16,
    dma_relay: u16,
}

impl Hal2State {
    /// Rate of the generator a codec's CLKID selects. CLKID is the generator
    /// NUMBER (1..3) while `bres_clock_rate` is 0-based, so it needs the shift.
    /// CLKID 0 selects nothing; report 0 rather than inventing a rate.
    fn bres_rate(&self, clk: usize) -> u32 {
        match clk {
            1..=3 => self.bres_clock_rate[clk - 1],
            _ => 0,
        }
    }
    fn codeca_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.codeca_ctrl[0]) }
    fn codecb_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.codecb_ctrl[0]) }
    fn aestx_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.aestx_ctrl[0]) }
    fn aesrx_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.aesrx_ctrl[0]) }
}

/// A clock generator's output rate. It is a Bresenham counter: every master
/// clock adds `inc`, and each time the sum reaches the modulus it ticks and
/// the modulus is taken off -- at most one tick a master clock. The modulus
/// is programmed as `modctrl = inc - mod - 1`. A modulus of 0 ticks on every
/// master clock: IRIX sets 44.1 kHz from the 44.1 kHz master as inc 0,
/// modctrl 0xffff, which is that, not a stopped clock.
fn bres_rate(master: u32, inc: u16, modctrl: u16) -> u32 {
    let inc = inc as u32;
    let modulus = inc.wrapping_sub(modctrl as u32).wrapping_sub(1) & 0xFFFF;
    if modulus == 0 || inc >= modulus {
        master
    } else {
        master * inc / modulus
    }
}

fn decode_ctrl1(ctrl1: u16) -> (usize, usize, usize) {
    let channel = (ctrl1 & CTRL1_CHAN_MASK) as usize;
    let clock   = ((ctrl1 >> CTRL1_CLOCK_SHIFT) & CTRL1_CLOCK_MASK) as usize;
    let mode    = ((ctrl1 >> CTRL1_MODE_SHIFT)  & CTRL1_MODE_MASK)  as usize;
    (channel, clock, mode)
}

// ─── Hal2 public struct ───────────────────────────────────────────────────────

pub struct Hal2 {
    state: Arc<Mutex<Hal2State>>,
    dma_clients: Vec<Arc<dyn DmaClient>>,
    timer_manager: Arc<std::sync::OnceLock<Arc<TimerManager>>>,
    // Per-channel mutable state
    ca_state: Arc<Mutex<CodecAState>>,
    cb_state: Arc<Mutex<CodecBState>>,
    at_state: Arc<Mutex<AesTxState>>,
    ar_state: Arc<Mutex<AesRxState>>,
    // cpal output-callback underrun count (ring buffer empty when the host pulled samples).
    underruns: Arc<AtomicU64>,
}

// ─── cpal helpers ─────────────────────────────────────────────────────────────

fn prebuf_samples(rate: u32) -> usize {
    (rate as usize * 2 * PREBUF_MS as usize) / 1000
}

/// Open a persistent stereo i16 cpal output stream, trying PREFERRED_RATES in order.
/// The stream plays silence when the ring buffer is empty.
fn open_persistent_output(underruns: Arc<AtomicU64>, playing: Arc<AtomicBool>) -> Option<AudioOut> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;

    for &rate in PREFERRED_RATES {
        let config = cpal::StreamConfig {
            channels: 2,
            sample_rate: rate,
            buffer_size: cpal::BufferSize::Default,
        };
        let ring_size = prebuf_samples(rate) * RING_BUF_MULTIPLIER;
        let err_fn = |err: cpal::Error| { eprintln!("HAL2: cpal stream error: {:?}", err); };

        // Try f32 first (macOS CoreAudio native), then i16 (Linux ALSA).
        let (producer, stream) = {
            let (p, mut c) = RingBuffer::<i16>::new(ring_size);
            let underruns_cb = underruns.clone();
            let playing_cb = playing.clone();
            // IRIS_HAL2_CAPTURE=<file>: also write what the host plays, raw
            // interleaved 16-bit little-endian stereo at the stream rate --
            // the end of the whole chain, to check or listen to offline.
            let mut capture = std::env::var_os("IRIS_HAL2_CAPTURE")
                .and_then(|p| std::fs::File::create(p).ok())
                .map(std::io::BufWriter::new);
            let data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for sample in data.iter_mut() {
                    let v = match c.pop() {
                        Ok(v) => v,
                        Err(_) => {
                            if playing_cb.load(Ordering::Relaxed) {
                                underruns_cb.fetch_add(1, Ordering::Relaxed);
                            }
                            0
                        }
                    };
                    if let Some(w) = capture.as_mut() {
                        use std::io::Write;
                        let _ = w.write_all(&v.to_le_bytes());
                    }
                    *sample = v as f32 / 32768.0;
                }
            };
            match device.build_output_stream(config, data_fn, err_fn.clone(), None) {
                Ok(s) => (p, s),
                Err(_) => {
                    // f32 failed, try i16
                    let (p, mut c) = RingBuffer::<i16>::new(ring_size);
                    let underruns_cb = underruns.clone();
                    let playing_cb = playing.clone();
                    let data_fn = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        for sample in data.iter_mut() {
                            *sample = match c.pop() {
                                Ok(v) => v,
                                Err(_) => {
                                    if playing_cb.load(Ordering::Relaxed) {
                                        underruns_cb.fetch_add(1, Ordering::Relaxed);
                                    }
                                    0
                                }
                            };
                        }
                    };
                    match device.build_output_stream(config, data_fn, err_fn.clone(), None) {
                        Ok(s) => (p, s),
                        Err(e) => {
                            eprintln!("HAL2: cpal build_output_stream failed at {}Hz: {:?}", rate, e);
                            continue;
                        }
                    }
                }
            }
        };
        if stream.play().is_err() { continue; }
        // cpal 0.18 dropped DeviceTrait::name(); a Device's Display impl is its name.
        println!("HAL2: audio output: {} via {:?} at {}Hz", device, host.id(), rate);
        return Some(AudioOut {
            stream_rate: rate,
            producer,
            underruns,
            playing,
            _stream: stream,
        });
    }

    eprintln!("HAL2: failed to open audio output at any rate (tried {:?})", PREFERRED_RATES);
    None
}


// ─── impl Hal2 ────────────────────────────────────────────────────────────────

impl Hal2 {
    pub fn new(dma_clients: Vec<Arc<dyn DmaClient>>) -> Self {
        Self {
            state: Arc::new(Mutex::new(Hal2State {
                isr: 0,
                iar: 0,
                idr: [0; 4],
                iar_log: std::collections::VecDeque::new(),
                codeca_ctrl: [0; 3],
                codecb_ctrl: [0; 3],
                aestx_ctrl: [0; 3],
                aesrx_ctrl: [0; 3],
                bres_clock_sel: [1; 3],          // 1 = 44100 Hz master (IP22 boot tune is 44100 Hz)
                bres_clock_inc: [1; 3],           // reset: inc=1
                bres_clock_modctrl: [0xFFFF; 3],  // reset: mod=1, so modctrl = 1-1-1 = 0xFFFF
                bres_clock_rate: [44100; 3],
                dma_enable: 0,
                dma_drive: 0,
                dma_endian: 0,
                dma_relay: 0,
            })),
            dma_clients,
            timer_manager: Arc::new(std::sync::OnceLock::new()),
            ca_state: Arc::new(Mutex::new(CodecAState::new())),
            cb_state: Arc::new(Mutex::new(CodecBState { timer_id: None })),
            at_state: Arc::new(Mutex::new(AesTxState { timer_id: None })),
            ar_state: Arc::new(Mutex::new(AesRxState { loopback: std::collections::VecDeque::new(), timer_id: None })),
            underruns: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn underrun_count(&self) -> u64 {
        self.underruns.load(Ordering::Relaxed)
    }

    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timer_manager.set(tm);
    }

    /// Power-on reset: restore all registers and per-channel state to defaults.
    /// Must only be called after `stop()` has cancelled all timers.
    pub fn power_on(&self) {
        {
            let mut s = self.state.lock();
            s.isr = 0;
            s.iar = 0;
            s.idr = [0; 4];
            s.codeca_ctrl = [0; 3];
            s.codecb_ctrl = [0; 3];
            s.aestx_ctrl  = [0; 3];
            s.aesrx_ctrl  = [0; 3];
            s.bres_clock_sel     = [1; 3];
            s.bres_clock_inc     = [1; 3];
            s.bres_clock_modctrl = [0xFFFF; 3];
            s.bres_clock_rate    = [44100; 3];
            s.dma_enable = 0;
            s.dma_drive  = 0;
            s.dma_endian = 0;
            s.dma_relay  = 0;
        }
        // Per-channel states: timers already stopped by stop(), just clear audio state.
        self.ca_state.lock().reset_audio();
        self.ar_state.lock().loopback.clear();
    }

    // ── Codec A output timer ──────────────────────────────────────────────────

    fn arm_codeca(&self) {
        let Some(tm) = self.timer_manager.get() else { return; };
        self.disarm_codeca();

        let (dma_ch, mode, rate, pitch_rate) = {
            let s = self.state.lock();
            let (ch, clk, mode) = s.codeca_cfg();
            let rate = s.bres_rate(clk);
            let (_, cb_clk, _) = s.codecb_cfg();
            let cb_rate = s.bres_rate(cb_clk);
            // Codec A is the DAC, so playback is paced by codec A's own
            // clock. Codec B is the ADC and has no say in playback pitch.
            let pitch_rate = if rate > 0 { rate } else if cb_rate > 0 { cb_rate } else { 44100 };
            (ch, mode, rate, pitch_rate)
        };

        if rate == 0 || dma_ch >= self.dma_clients.len() { return; }

        let dma_client = self.dma_clients[dma_ch].clone();
        let ca_state = self.ca_state.clone();
        let mut pacer = Pacer::new(pitch_rate);

        self.ca_state.lock().armed_ch = Some(dma_ch);
        let id = tm.add_recurring(Instant::now() + PACE_PERIOD, PACE_PERIOD, (), move |_| {
            let mut due = pacer.due();
            // Keep within READ_AHEAD_MS of the position the guest last polled
            // (see READ_AHEAD_MS); the rest stays due for the next wake.
            if let Some(ahead_words) = dma_client.read_ahead_of_poll() {
                let words_per_frame: u64 = match mode { MODE_MONO => 1, MODE_QUAD => 4, _ => 2 };
                let limit = (pitch_rate as u64 * READ_AHEAD_MS).div_ceil(1000);
                let allowed = limit.saturating_sub(ahead_words / words_per_frame);
                if due > allowed {
                    pacer.give_back(due - allowed);
                    due = allowed;
                }
            }
            if due == 0 {
                return TimerReturn::Continue;
            }
            let mut st = ca_state.lock();
            st.calls += 1;

            // No audio output — still drain DMA so the kernel doesn't hang
            // waiting for PDMA_CTRL_ACT to clear.
            let stream_rate = match st.out.as_ref() {
                Some(o) => o.stream_rate,
                None => {
                    for _ in 0..due {
                        let _ = read_frame_from(&dma_client, mode);
                    }
                    return TimerReturn::Continue;
                }
            };

            // (Re)build resampler if codec rate changed.
            // Use codec B rate as the declared input rate (experiment).
            if st.resampler.as_ref().map_or(true, |r| r.in_rate != pitch_rate) {
                st.resampler = Some(Resampler::new(pitch_rate, stream_rate));
                dlog_dev!(LogModule::Hal2, "HAL2: Codec A resampler {}Hz (pitch={}) → {}Hz", rate, pitch_rate, stream_rate);
            }

            for _ in 0..due {
                let frame = read_frame_from(&dma_client, mode);
                if frame.is_some() { st.frames += 1; } else { st.dry_reads += 1; }
                let was_prebuffering = st.prebuffering;

                match frame {
                    Some((l, r)) => {
                        st.dry = 0;
                        if !st.nonzero_seen && (l != 0 || r != 0) {
                            dlog_dev!(LogModule::Hal2, "HAL2: Codec A first non-zero: l={} r={}", l, r);
                            st.nonzero_seen = true;
                        }
                        if st.prebuffering {
                            // Accumulate before feeding the ring to prevent underrun.
                            st.prebuf.push(l);
                            st.prebuf.push(r);
                            if st.prebuf.len() >= prebuf_samples(rate) {
                                let samples = std::mem::take(&mut st.prebuf);
                                st.push_to_ring(&samples);
                                dlog_dev!(LogModule::Hal2, "HAL2: Codec A prebuf flushed ({} frames)", samples.len() / 2);
                                st.prebuffering = false;
                            }
                        } else {
                            // Active: push directly.
                            st.push_to_ring(&[l, r]);
                        }
                    }
                    None => {
                        st.dry += 1;
                        if st.prebuffering && !st.prebuf.is_empty() && st.dry >= DRY_LIMIT {
                            // Flush whatever we buffered so far rather than waiting forever.
                            let samples = std::mem::take(&mut st.prebuf);
                            st.push_to_ring(&samples);
                            dlog_dev!(LogModule::Hal2, "HAL2: Codec A prebuf flushed (dry) after {} dry reads", st.dry);
                            st.prebuffering = false;
                            st.dry = 0;
                        }
                    }
                }

                // Prebuffering just finished — real audio is now flowing into the ring,
                // so the cpal callback can start treating an empty ring as a genuine underrun.
                if was_prebuffering && !st.prebuffering {
                    if let Some(o) = &st.out {
                        o.playing.store(true, Ordering::Relaxed);
                    }
                }
            }

            TimerReturn::Continue
        });

        self.ca_state.lock().timer_id = Some(id);
    }

    fn disarm_codeca(&self) {
        let id = self.ca_state.lock().timer_id.take();
        if let Some(id) = id {
            if let Some(tm) = self.timer_manager.get() { tm.remove(id); }
        }
        self.ca_state.lock().reset_audio(); // keeps `out` (stream stays open)
    }

    // ── Codec B input (silence writer) timer ─────────────────────────────────

    fn arm_codecb(&self) {
        let Some(tm) = self.timer_manager.get() else { return; };
        self.disarm_codecb();

        let (dma_ch, mode, rate) = {
            let s = self.state.lock();
            let (ch, clk, mode) = s.codecb_cfg();
            let rate = s.bres_rate(clk);
            (ch, mode, rate)
        };

        if rate == 0 || dma_ch >= self.dma_clients.len() { return; }

        let dma_client = self.dma_clients[dma_ch].clone();
        let mut pacer = Pacer::new(rate);

        let id = tm.add_recurring(Instant::now() + PACE_PERIOD, PACE_PERIOD, (), move |_| {
            for _ in 0..pacer.due() {
                let _ = dma_client.write(0, false);
                if mode == MODE_STEREO || mode == MODE_QUAD {
                    let _ = dma_client.write(0, false);
                }
                if mode == MODE_QUAD {
                    let _ = dma_client.write(0, false);
                    let _ = dma_client.write(0, false);
                }
            }
            TimerReturn::Continue
        });

        self.cb_state.lock().timer_id = Some(id);
    }

    fn disarm_codecb(&self) {
        let id = self.cb_state.lock().timer_id.take();
        if let Some(id) = id {
            if let Some(tm) = self.timer_manager.get() { tm.remove(id); }
        }
    }

    // ── AES TX drain timer (no cpal output) ──────────────────────────────────

    fn arm_aestx(&self) {
        let Some(tm) = self.timer_manager.get() else { return; };
        self.disarm_aestx();

        let (dma_ch, rate) = {
            let s = self.state.lock();
            let (ch, clk, _mode) = s.aestx_cfg();
            let rate = s.bres_rate(clk);
            (ch, rate)
        };

        if rate == 0 || dma_ch >= self.dma_clients.len() { return; }

        let dma_client = self.dma_clients[dma_ch].clone();
        let ar_state = self.ar_state.clone();
        let mut pacer = Pacer::new(rate);

        let id = tm.add_recurring(Instant::now() + PACE_PERIOD, PACE_PERIOD, (), move |_| {
            for _ in 0..pacer.due() {
                if let Some((val, st, _)) = dma_client.read() {
                    if !st.refused() {
                        ar_state.lock().loopback.push_back(val);
                    }
                }
            }
            TimerReturn::Continue
        });

        self.at_state.lock().timer_id = Some(id);
    }

    fn disarm_aestx(&self) {
        let id = self.at_state.lock().timer_id.take();
        if let Some(id) = id {
            if let Some(tm) = self.timer_manager.get() { tm.remove(id); }
        }
    }

    // ── AES RX loopback write timer ───────────────────────────────────────────

    fn arm_aesrx(&self) {
        let Some(tm) = self.timer_manager.get() else { return; };
        self.disarm_aesrx();

        let (dma_ch, rate) = {
            let s = self.state.lock();
            let (ch, clk, _mode) = s.aesrx_cfg();
            let rate = s.bres_rate(clk);
            (ch, rate)
        };

        if rate == 0 || dma_ch >= self.dma_clients.len() { return; }

        let dma_client = self.dma_clients[dma_ch].clone();
        let ar_state = self.ar_state.clone();
        let mut pacer = Pacer::new(rate);

        let id = tm.add_recurring(Instant::now() + PACE_PERIOD, PACE_PERIOD, (), move |_| {
            for _ in 0..pacer.due() {
                let val = ar_state.lock().loopback.pop_front().unwrap_or(0);
                let _ = dma_client.write(val, false);
            }
            TimerReturn::Continue
        });

        self.ar_state.lock().timer_id = Some(id);
    }

    fn disarm_aesrx(&self) {
        let id = self.ar_state.lock().timer_id.take();
        if let Some(id) = id {
            if let Some(tm) = self.timer_manager.get() { tm.remove(id); }
        }
        self.ar_state.lock().loopback.clear();
    }

    // ── React to dma_enable changes ───────────────────────────────────────────
    // dma_drive uses physical HPC3 channel bits (unrelated to device indices)
    // so we gate arming only on dma_enable.

    fn apply_dma_enable(&self, old: u16, new: u16) {
        let changed = |bit: u16| (old & bit) != (new & bit);
        let enabled = |bit: u16| (new & bit) != 0;

        if changed(DMA_EN_CODECA) {
            if enabled(DMA_EN_CODECA) {
                dlog_dev!(LogModule::Hal2, "HAL2: Codec A DMA enabled");
                self.arm_codeca();
            } else {
                dlog_dev!(LogModule::Hal2, "HAL2: Codec A DMA disabled");
                self.disarm_codeca();
            }
        }

        if changed(DMA_EN_CODECB) {
            if enabled(DMA_EN_CODECB) {
                dlog_dev!(LogModule::Hal2, "HAL2: Codec B DMA enabled");
                self.arm_codecb();
            } else {
                dlog_dev!(LogModule::Hal2, "HAL2: Codec B DMA disabled");
                self.disarm_codecb();
            }
        }

        if changed(DMA_EN_AES_TX) {
            if enabled(DMA_EN_AES_TX) {
                dlog_dev!(LogModule::Hal2, "HAL2: AES TX DMA enabled");
                self.arm_aestx();
            } else {
                dlog_dev!(LogModule::Hal2, "HAL2: AES TX DMA disabled");
                self.disarm_aestx();
            }
        }

        if changed(DMA_EN_AES_RX) {
            if enabled(DMA_EN_AES_RX) {
                dlog_dev!(LogModule::Hal2, "HAL2: AES RX DMA enabled");
                self.arm_aesrx();
            } else {
                dlog_dev!(LogModule::Hal2, "HAL2: AES RX DMA disabled");
                self.disarm_aesrx();
            }
        }
    }

    fn disarm_all(&self) {
        self.disarm_codeca();
        self.disarm_codecb();
        self.disarm_aestx();
        self.disarm_aesrx();
    }

    // ── Register access ───────────────────────────────────────────────────────

    fn update_rates(state: &mut Hal2State) {
        for i in 0..3 {
            let master = match state.bres_clock_sel[i] {
                0 => 48000u32,
                1 => 44100,
                _ => 48000,
            };
            state.bres_clock_rate[i] = bres_rate(master,
                state.bres_clock_inc[i], state.bres_clock_modctrl[i]);
        }
    }

    fn handle_iar_write(&self, val: u16) {
        let mut state = self.state.lock();
        state.iar = val;
        let idr = state.idr;
        if state.iar_log.len() == 32 {
            state.iar_log.pop_front();
        }
        state.iar_log.push_back((val, idr));

        let is_read  = (val & IAR_ACCESS_READ) != 0;
        let typ      = val & IAR_TYPE_MASK;
        let num      = val & IAR_NUM_MASK;
        let param    = val & IAR_PARAM_MASK;
        let bres_idx = ((val >> 8) & 0xF) as usize;

        if is_read {
            match typ {
                IAR_TYPE_GLOBAL_DMA => match param {
                    IAR_PARAM_0 => state.idr[0] = state.dma_relay,
                    IAR_PARAM_1 => state.idr[0] = state.dma_enable,
                    IAR_PARAM_2 => state.idr[0] = state.dma_endian,
                    IAR_PARAM_3 => state.idr[0] = state.dma_drive,
                    _ => {}
                },
                IAR_TYPE_DMA => match num {
                    IAR_NUM_CODECA => match param {
                        IAR_PARAM_1 => state.idr[0] = state.codeca_ctrl[0],
                        IAR_PARAM_2 => { state.idr[0] = state.codeca_ctrl[1]; state.idr[1] = state.codeca_ctrl[2]; }
                        _ => {}
                    },
                    IAR_NUM_CODECB => match param {
                        IAR_PARAM_1 => state.idr[0] = state.codecb_ctrl[0],
                        IAR_PARAM_2 => { state.idr[0] = state.codecb_ctrl[1]; state.idr[1] = state.codecb_ctrl[2]; }
                        _ => {}
                    },
                    IAR_NUM_AES_TX => match param {
                        IAR_PARAM_1 => state.idr[0] = state.aestx_ctrl[0],
                        IAR_PARAM_2 => { state.idr[0] = state.aestx_ctrl[1]; state.idr[1] = state.aestx_ctrl[2]; }
                        _ => {}
                    },
                    IAR_NUM_AES_RX => match param {
                        IAR_PARAM_1 => state.idr[0] = state.aesrx_ctrl[0],
                        IAR_PARAM_2 => { state.idr[0] = state.aesrx_ctrl[1]; state.idr[1] = state.aesrx_ctrl[2]; }
                        _ => {}
                    },
                    _ => {}
                },
                IAR_TYPE_BRES if bres_idx >= 1 && bres_idx <= 3 => {
                    let idx = bres_idx - 1;
                    match param {
                        IAR_PARAM_1 => state.idr[0] = state.bres_clock_sel[idx],
                        IAR_PARAM_2 => {
                            state.idr[0] = state.bres_clock_inc[idx];
                            state.idr[1] = state.bres_clock_modctrl[idx];
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        } else {
            // Write path — handle DMA enable specially (need old value)
            match typ {
                IAR_TYPE_GLOBAL_DMA => match param {
                    IAR_PARAM_0 => state.dma_relay  = state.idr[0],
                    IAR_PARAM_1 => {
                        let old = state.dma_enable;
                        let new = state.idr[0];
                        state.dma_enable = new;
                        if old != new { dlog_dev!(LogModule::Hal2, "HAL2: DMA enable 0x{:02x} -> 0x{:02x}", old, new); }
                        drop(state);
                        self.apply_dma_enable(old, new);
                        return;
                    }
                    IAR_PARAM_2 => state.dma_endian = state.idr[0],
                    IAR_PARAM_3 => {
                        if state.dma_drive != state.idr[0] {
                            dlog_dev!(LogModule::Hal2, "HAL2: DMA drive 0x{:02x} -> 0x{:02x}", state.dma_drive, state.idr[0]);
                        }
                        state.dma_drive = state.idr[0];
                    }
                    _ => {}
                },
                IAR_TYPE_DMA => {
                    // CTRL1 writes (param=1) may change the channel number or clock index
                    // for an already-active channel — re-arm it if DMA is currently enabled.
                    let rearm = match (num, param) {
                        (IAR_NUM_CODECA, IAR_PARAM_1) => {
                            state.codeca_ctrl[0] = state.idr[0];
                            (state.dma_enable & DMA_EN_CODECA) != 0
                        }
                        (IAR_NUM_CODECA, IAR_PARAM_2) => {
                            state.codeca_ctrl[1] = state.idr[0]; state.codeca_ctrl[2] = state.idr[1]; false
                        }
                        (IAR_NUM_CODECB, IAR_PARAM_1) => {
                            state.codecb_ctrl[0] = state.idr[0];
                            (state.dma_enable & DMA_EN_CODECB) != 0
                        }
                        (IAR_NUM_CODECB, IAR_PARAM_2) => {
                            state.codecb_ctrl[1] = state.idr[0]; state.codecb_ctrl[2] = state.idr[1]; false
                        }
                        (IAR_NUM_AES_TX, IAR_PARAM_1) => {
                            state.aestx_ctrl[0] = state.idr[0];
                            (state.dma_enable & DMA_EN_AES_TX) != 0
                        }
                        (IAR_NUM_AES_TX, IAR_PARAM_2) => {
                            state.aestx_ctrl[1] = state.idr[0]; state.aestx_ctrl[2] = state.idr[1]; false
                        }
                        (IAR_NUM_AES_RX, IAR_PARAM_1) => {
                            state.aesrx_ctrl[0] = state.idr[0];
                            (state.dma_enable & DMA_EN_AES_RX) != 0
                        }
                        (IAR_NUM_AES_RX, IAR_PARAM_2) => {
                            state.aesrx_ctrl[1] = state.idr[0]; state.aesrx_ctrl[2] = state.idr[1]; false
                        }
                        _ => false,
                    };
                    if rearm {
                        drop(state);
                        match num {
                            IAR_NUM_CODECA => self.arm_codeca(),
                            IAR_NUM_CODECB => self.arm_codecb(),
                            IAR_NUM_AES_TX => self.arm_aestx(),
                            IAR_NUM_AES_RX => self.arm_aesrx(),
                            _ => {}
                        }
                        return;
                    }
                }
                IAR_TYPE_BRES if bres_idx >= 1 && bres_idx <= 3 => {
                    let idx = bres_idx - 1;
                    let changed = match param {
                        IAR_PARAM_1 => {
                            state.bres_clock_sel[idx] = state.idr[0];
                            Self::update_rates(&mut state);
                            let master = if state.bres_clock_sel[idx] == 0 { 48000 } else { 44100 };
                            dlog_dev!(LogModule::Hal2, "HAL2: BRES{} sel={} ({}Hz master) → {}Hz",
                                bres_idx, state.bres_clock_sel[idx], master, state.bres_clock_rate[idx]);
                            true
                        }
                        IAR_PARAM_2 => {
                            state.bres_clock_inc[idx]     = state.idr[0];
                            state.bres_clock_modctrl[idx] = state.idr[1];
                            Self::update_rates(&mut state);
                            dlog_dev!(LogModule::Hal2, "HAL2: BRES{} inc={} modctrl={} → {}Hz",
                                bres_idx, state.bres_clock_inc[idx], state.bres_clock_modctrl[idx],
                                state.bres_clock_rate[idx]);
                            true
                        }
                        _ => false,
                    };
                    if changed {
                        drop(state);
                        // reclock_active compares against codec CLKIDs, which
                        // are generator numbers, so pass the 1-based value.
                        self.reclock_active(bres_idx);
                        return;
                    }
                }
                _ => {}
            }
        }
    }

    /// Re-arm any active channels using generator `bres_idx` (1..3, the same
    /// basis as a codec CTRL1 CLKID).
    fn reclock_active(&self, bres_idx: usize) {
        // Read current clock indices and active mask under lock, then re-arm outside lock.
        let (ca_clk, cb_clk, at_clk, ar_clk, dma_enable) = {
            let s = self.state.lock();
            let (_, ca_clk, _) = s.codeca_cfg();
            let (_, cb_clk, _) = s.codecb_cfg();
            let (_, at_clk, _) = s.aestx_cfg();
            let (_, ar_clk, _) = s.aesrx_cfg();
            (ca_clk, cb_clk, at_clk, ar_clk, s.dma_enable)
        };

        // Codec A re-arms when the generator it selected is reprogrammed. It
        // no longer re-arms on a codec B change: its period is its own now.
        if (dma_enable & DMA_EN_CODECA) != 0 && ca_clk == bres_idx {
            self.arm_codeca();
        }
        if (dma_enable & DMA_EN_CODECB) != 0 && cb_clk == bres_idx {
            self.arm_codecb();
        }
        if (dma_enable & DMA_EN_AES_TX) != 0 && at_clk == bres_idx {
            self.arm_aestx();
        }
        if (dma_enable & DMA_EN_AES_RX) != 0 && ar_clk == bres_idx {
            self.arm_aesrx();
        }
    }

    pub fn read(&self, addr: u32) -> BusRead16 {
        let offset = addr & 0xFF;
        let state = self.state.lock();

        let val: u16 = match offset & 0xF0 {
            HAL2_ISR  => state.isr,
            HAL2_REV  => 0x4010,
            HAL2_IAR  => state.iar,
            HAL2_IDR0 => state.idr[0],
            HAL2_IDR1 => state.idr[1],
            HAL2_IDR2 => state.idr[2],
            HAL2_IDR3 => state.idr[3],
            _ => 0,
        };

        dlog_dev!(LogModule::Hal2, "HAL2: Read offset {:02x} -> {:04x}", offset, val);
        BusRead16::ok(val)
    }

    pub fn write(&self, addr: u32, val: u16) -> u32 {
        let offset = addr & 0xFF;

        dlog_dev!(LogModule::Hal2, "HAL2: Write offset {:02x} <- {:04x}", offset, val);

        match offset & 0xF0 {
            HAL2_ISR => {
                        let old_enable;
                {
                    let mut state = self.state.lock();
                    old_enable = state.dma_enable;
                    if (val & (isr::GLOBAL_RESET_N as u16)) == 0 {
                        dlog_dev!(LogModule::Hal2, "HAL2: global reset (ISR=0x{:04x})", val);
                        state.dma_enable = 0;
                        state.dma_drive  = 0;
                        state.codeca_ctrl = [0; 3];
                        state.codecb_ctrl = [0; 3];
                        state.aestx_ctrl  = [0; 3];
                        state.aesrx_ctrl  = [0; 3];
                        state.bres_clock_sel     = [1; 3];
                        state.bres_clock_inc     = [1; 3];
                        state.bres_clock_modctrl = [0xFFFF; 3];
                        state.bres_clock_rate    = [44100; 3];
                    } else if (val & (isr::CODEC_RESET_N as u16)) == 0 {
                        dlog_dev!(LogModule::Hal2, "HAL2: codec reset (ISR=0x{:04x})", val);
                        state.dma_enable &= !(DMA_EN_CODECA | DMA_EN_CODECB | DMA_EN_AES_TX | DMA_EN_AES_RX);
                        state.codeca_ctrl = [0; 3];
                        state.codecb_ctrl = [0; 3];
                        state.aestx_ctrl  = [0; 3];
                        state.aesrx_ctrl  = [0; 3];
                    }
                    state.isr = val;
                }
                // Apply any DMA enable changes that happened during reset
                let new_enable = self.state.lock().dma_enable;
                if old_enable != new_enable {
                    self.apply_dma_enable(old_enable, new_enable);
                }
            }
            HAL2_IAR  => self.handle_iar_write(val),
            HAL2_IDR0 => self.state.lock().idr[0] = val,
            HAL2_IDR1 => self.state.lock().idr[1] = val,
            HAL2_IDR2 => self.state.lock().idr[2] = val,
            HAL2_IDR3 => self.state.lock().idr[3] = val,
            _ => {}
        }
        BUS_OK
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        let me = self.clone(); register_lock_fn("hal2::state",    move || me.state.is_locked());
        let me = self.clone(); register_lock_fn("hal2::ca_state", move || me.ca_state.is_locked());
        let me = self.clone(); register_lock_fn("hal2::cb_state", move || me.cb_state.is_locked());
        let me = self.clone(); register_lock_fn("hal2::at_state", move || me.at_state.is_locked());
        let me = self.clone(); register_lock_fn("hal2::ar_state", move || me.ar_state.is_locked());
    }
}

// ─── DMA read helper (free function to avoid borrow issues in closures) ───────

fn read_frame_from(client: &Arc<dyn DmaClient>, mode: usize) -> Option<(i16, i16)> {
    if mode == MODE_MONO {
        let (v, st, _) = client.read()?;
        if st.refused() { return None; }
        let s = v as i16;
        Some((s, s))
    } else {
        let (lv, lst, _) = client.read()?;
        if lst.refused() { return None; }
        let (rv, rst, _) = match client.read() {
            Some(r) => r,
            None => return Some((lv as i16, lv as i16)),
        };
        let r = if rst.refused() { lv as i16 } else { rv as i16 };
        if mode == MODE_QUAD {
            let _ = client.read();
            let _ = client.read();
        }
        Some((lv as i16, r))
    }
}

// ─── Device impl ──────────────────────────────────────────────────────────────

impl Default for Hal2 {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

impl Device for Hal2 {
    fn step(&self, _cycles: u64) {}

    fn start(&self) {
        // Open persistent audio output once.  Codec A timer will push into it.
        let audio = open_persistent_output(self.underruns.clone(), Arc::new(AtomicBool::new(false)));
        if audio.is_none() {
            eprintln!("HAL2: no audio output available");
        }
        self.ca_state.lock().out = audio;

        // Re-arm any channels that were already enabled (e.g. after a snapshot restore)
        let dma_enable = self.state.lock().dma_enable;
        self.apply_dma_enable(0, dma_enable);
    }

    fn stop(&self) {
        self.disarm_all();
        // Drop the audio output stream.
        self.ca_state.lock().out = None;
    }

    fn is_running(&self) -> bool {
        self.timer_manager.get().is_some()
    }

    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("hal2".to_string(), "HAL2 commands: hal2 status".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd != "hal2" { return Err("Command not found".to_string()); }

        match args.first().map(|s| *s) {
            Some("status") => {
                let s = self.state.lock();

                writeln!(writer, "ISR: 0x{:04x}  global_reset_n={} codec_reset_n={} codec_mode={}",
                    s.isr,
                    (s.isr & isr::GLOBAL_RESET_N as u16 != 0) as u8,
                    (s.isr & isr::CODEC_RESET_N  as u16 != 0) as u8,
                    if s.isr & isr::CODEC_MODE as u16 != 0 { "quad" } else { "indigo" },
                ).unwrap();

                writeln!(writer, "DMA enable: 0x{:02x}  codeca={} codecb={} aes_tx={} aes_rx={}",
                    s.dma_enable,
                    (s.dma_enable & DMA_EN_CODECA != 0) as u8,
                    (s.dma_enable & DMA_EN_CODECB != 0) as u8,
                    (s.dma_enable & DMA_EN_AES_TX != 0) as u8,
                    (s.dma_enable & DMA_EN_AES_RX != 0) as u8,
                ).unwrap();
                // dma_drive uses physical HPC3 channel bits (bit N = PBUS channel N)
                writeln!(writer, "DMA drive:  0x{:02x}  (physical HPC3 channel bitmask)", s.dma_drive).unwrap();

                for i in 0..3 {
                    let master = if s.bres_clock_sel[i] == 0 { 48000u32 } else { 44100 };
                    writeln!(writer, "BRES{}: sel={} ({} Hz master)  inc={}  modctrl={}  → {}Hz",
                        i + 1, s.bres_clock_sel[i], master,
                        s.bres_clock_inc[i], s.bres_clock_modctrl[i],
                        s.bres_clock_rate[i],
                    ).unwrap();
                }

                let mode_str = |m| match m { 1 => "mono", 2 => "stereo", 3 => "quad", _ => "off" };
                // bres=0 means CLKID 0: no generator selected.

                let (ca_ch, ca_clk, ca_mode) = s.codeca_cfg();
                writeln!(writer, "Codec A: ch={} bres={} rate={}Hz mode={}",
                    ca_ch, ca_clk, s.bres_rate(ca_clk), mode_str(ca_mode)).unwrap();
                writeln!(writer, "  ctrl1=0x{:04x} ctrl2=[0x{:04x} 0x{:04x}]",
                    s.codeca_ctrl[0], s.codeca_ctrl[1], s.codeca_ctrl[2]).unwrap();

                let (cb_ch, cb_clk, cb_mode) = s.codecb_cfg();
                writeln!(writer, "Codec B: ch={} bres={} rate={}Hz mode={}",
                    cb_ch, cb_clk, s.bres_rate(cb_clk), mode_str(cb_mode)).unwrap();

                let (at_ch, at_clk, _) = s.aestx_cfg();
                let (ar_ch, ar_clk, _) = s.aesrx_cfg();
                writeln!(writer, "AES TX: ch={} bres={} rate={}Hz", at_ch, at_clk, s.bres_rate(at_clk)).unwrap();
                writeln!(writer, "AES RX: ch={} bres={} rate={}Hz", ar_ch, ar_clk, s.bres_rate(ar_clk)).unwrap();
                drop(s);

                let ca = self.ca_state.lock();
                writeln!(writer, "Codec A out: {}  pitch: {}  prebuf: {}  prebuffering: {}  timer: {}",
                    ca.out.as_ref().map_or("none".to_string(), |o| format!("{}Hz", o.stream_rate)),
                    ca.resampler.as_ref().map_or("none".to_string(), |r| format!("{}Hz", r.in_rate)),
                    ca.prebuf.len() / 2,
                    ca.prebuffering,
                    ca.timer_id.map_or("none".to_string(), |id| format!("{:#x}", id)),
                ).unwrap();
                writeln!(writer, "Codec A timer: reads ch={}  calls={}  frames={}  dry reads={}",
                    ca.armed_ch.map_or("-".to_string(), |c| c.to_string()), ca.calls, ca.frames, ca.dry_reads).unwrap();
                drop(ca);
                writeln!(writer, "cpal underruns (samples): {}", self.underruns.load(Ordering::Relaxed)).unwrap();

                writeln!(writer, "Codec B timer: {}",
                    self.cb_state.lock().timer_id.map_or("none".to_string(), |id| format!("{:#x}", id))).unwrap();
                writeln!(writer, "AES TX timer: {}",
                    self.at_state.lock().timer_id.map_or("none".to_string(), |id| format!("{:#x}", id))).unwrap();
                let ar = self.ar_state.lock();
                writeln!(writer, "AES RX timer: {}  loopback_len={}",
                    ar.timer_id.map_or("none".to_string(), |id| format!("{:#x}", id)),
                    ar.loopback.len(),
                ).unwrap();
                drop(ar);
                let log: Vec<(u16, [u16; 4])> = self.state.lock().iar_log.iter().copied().collect();
                writeln!(writer, "Recent indirect accesses (IAR: IDR0 IDR1 IDR2 IDR3), oldest first:").unwrap();
                for (iar, idr) in log {
                    writeln!(writer, "  {:04x}{}: {:04x} {:04x} {:04x} {:04x}", iar,
                        if iar & IAR_ACCESS_READ != 0 { " (read)" } else { "" },
                        idr[0], idr[1], idr[2], idr[3]).unwrap();
                }
            }
            _ => return Err("Usage: hal2 status".to_string()),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtrb::RingBuffer;

    #[test]
    fn bres_modulus_zero_runs_at_the_master_rate() {
        // What IRIX writes for 44.1 kHz from the 44.1 kHz master, and for
        // 48 kHz from 48 kHz: both are the master rate.
        assert_eq!(bres_rate(44100, 0, 0xFFFF), 44100);
        assert_eq!(bres_rate(48000, 1, 0xFFFF), 48000);
        // inc 4, modulus 8 (modctrl = 4 - 8 - 1): half the master rate.
        assert_eq!(bres_rate(44100, 4, 4u16.wrapping_sub(8).wrapping_sub(1)), 22050);
        // No increment into a nonzero modulus never ticks.
        assert_eq!(bres_rate(48000, 0, 0u16.wrapping_sub(4).wrapping_sub(1)), 0);
    }

    fn resample(in_rate: u32, out_rate: u32, n_frames: usize) -> usize {
        let (mut prod, mut cons) = RingBuffer::<i16>::new(n_frames * 4 + 16);
        let mut r = Resampler::new(in_rate, out_rate);
        for i in 0..n_frames {
            r.push(i as i16, i as i16, &mut prod);
        }
        drop(prod);
        let mut count = 0;
        while cons.pop().is_ok() { count += 1; }
        count / 2  // stereo pairs → frames
    }

    #[test]
    fn resampler_passthrough() {
        // 1:1 — every input frame produces exactly one output frame
        assert_eq!(resample(44100, 44100, 1000), 1000);
    }

    #[test]
    fn resampler_downsample_2x() {
        // 44100 → 22050: every 2 inputs → 1 output, so 1000 in → 500 out
        let out = resample(44100, 22050, 1000);
        assert_eq!(out, 500, "44100→22050: expected 500 frames, got {}", out);
    }

    #[test]
    fn resampler_upsample_2x() {
        // 22050 → 44100: every input → 2 outputs, so 1000 in → 2000 out
        let out = resample(22050, 44100, 1000);
        assert_eq!(out, 2000, "22050→44100: expected 2000 frames, got {}", out);
    }

    #[test]
    fn resampler_upsample_44100_to_48000() {
        // 44100 → 48000: ratio ~1.0884, so 44100 in → 48000 out (over one second of audio)
        let out = resample(44100, 48000, 44100);
        assert_eq!(out, 48000, "44100→48000: expected 48000 frames, got {}", out);
    }

    /// A late wake must not move a burst: after a 10 ms stall at 11025 Hz
    /// the next call moves at most 1.25 periods' worth, and the backlog is
    /// still delivered over the following calls.
    #[test]
    fn pacer_spreads_catch_up_after_a_late_wake() {
        let mut p = Pacer::new(11025);
        p.start -= Duration::from_millis(10);
        let per_period = (11025 * PACE_PERIOD.as_micros() as u64).div_ceil(1_000_000);
        let first = p.due();
        assert!(first <= (per_period * 5).div_ceil(4).max(per_period + 1), "moved {} at once", first);
        let mut total = first;
        for _ in 0..200 { total += p.due(); }
        assert!(total >= 110, "backlog not delivered: {}", total);
    }

    /// A 1 kHz tone at 11025 Hz, resampled to 48 kHz, against the ideal
    /// sine at the best-fitting delay. Repeating samples (the old resampler)
    /// misses by about a fifth of the amplitude; interpolation must be close.
    #[test]
    fn resampler_11025_to_48000_follows_the_waveform() {
        let (inr, outr, f, amp) = (11025u32, 48000u32, 1000.0f64, 16000.0f64);
        let n = 11025;
        let (mut prod, mut cons) = RingBuffer::<i16>::new(n * 12);
        let mut r = Resampler::new(inr, outr);
        for k in 0..n {
            let v = (amp * (2.0 * std::f64::consts::PI * f * k as f64 / inr as f64).sin()) as i16;
            r.push(v, v, &mut prod);
        }
        drop(prod);
        let mut out = Vec::new();
        while let Ok(v) = cons.pop() { out.push(v as f64); }
        let left: Vec<f64> = out.iter().step_by(2).copied().collect();
        let body = &left[1000..left.len() - 1000];
        let mut best = f64::MAX;
        for step in 0..400 {
            let delay = step as f64 * 0.01; // input samples
            let mut err = 0.0;
            for (j, v) in body.iter().enumerate() {
                let t = (j + 1000) as f64 / outr as f64 - delay / inr as f64;
                let want = amp * (2.0 * std::f64::consts::PI * f * t).sin();
                err += (v - want) * (v - want);
            }
            best = best.min((err / body.len() as f64).sqrt() / amp);
        }
        assert!(best < 0.03, "RMS error {:.3} of the amplitude", best);
    }

    #[test]
    fn resampler_downsample_48000_to_44100() {
        // 48000 → 44100: ratio ~0.919, so 48000 in → 44100 out
        let out = resample(48000, 44100, 48000);
        assert_eq!(out, 44100, "48000→44100: expected 44100 frames, got {}", out);
    }
}

#[cfg(test)]
mod clkid_tests {
    use super::*;

    /// Three generators at distinguishable rates, so an off-by-one shows up as
    /// a wrong number rather than a coincidence.
    fn state_with_rates() -> Hal2State {
        let mut s = Hal2State::default();
        s.bres_clock_rate = [48000, 44100, 32000];
        s
    }

    #[test]
    fn clkid_is_a_generator_number_not_an_index() {
        let s = state_with_rates();
        // CLKID n selects BRESn, so it indexes the array at n-1.
        assert_eq!(s.bres_rate(1), 48000, "CLKID 1 is BRES1");
        assert_eq!(s.bres_rate(2), 44100, "CLKID 2 is BRES2");
        assert_eq!(s.bres_rate(3), 32000, "CLKID 3 is BRES3");
        // 0 selects no generator — reporting a rate here is inventing one.
        assert_eq!(s.bres_rate(0), 0, "CLKID 0 selects nothing");
    }

    #[test]
    fn a_codec_reads_the_generator_its_driver_programmed() {
        // What NetBSD's haltwo and Linux's hal2 both do for playback: program
        // BRES1 to the wanted rate, then point the DAC at it with CLKID 1.
        // ctrl1 = 0x0208 is what NetBSD 10.2 and 11.0 actually write.
        let mut s = state_with_rates();
        s.codeca_ctrl[0] = 0x0208;
        let (_, clk, _) = s.codeca_cfg();
        assert_eq!(clk, 1, "ctrl1 0x0208 carries CLKID 1");
        assert_eq!(
            s.bres_rate(clk),
            48000,
            "codec A must read BRES1, the generator the driver configured"
        );
    }

    #[test]
    fn the_adc_reads_its_own_generator_too() {
        // Linux records on BRES2 with CLKID 2 ("2nd Bresenham clock generator
        // for record"), which must not resolve to BRES3.
        let mut s = state_with_rates();
        s.codecb_ctrl[0] = (2 << CTRL1_CLOCK_SHIFT) as u16;
        let (_, clk, _) = s.codecb_cfg();
        assert_eq!(clk, 2);
        assert_eq!(s.bres_rate(clk), 44100, "CLKID 2 is BRES2, not BRES3");
    }
}
