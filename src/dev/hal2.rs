use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::devlog::LogModule;
use std::time::{Duration, Instant};
use std::io::Write;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, Device, DmaClient};
use crate::config::AudioConfig;
use crate::dev::audio_dump::AudioDump;
use crate::hptimer::{TimerManager, TimerId, TimerReturn};
use crate::host_audio::{HostOut as AudioOut, Meter, Resampler, Skip};

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

// CS4216 Codec A CTRL2 bitfields
// IDR0 (codeca_ctrl[1]): bit 10 = MUTE
const CODECA_CTRL2_MUTE: u16 = 0x0400;
// IDR1 (codeca_ctrl[2]): bits 11:7 = LA (left DAC attenuation, 1.5 dB/step),
//                        bits 6:2  = RA (right DAC attenuation, 1.5 dB/step)
const CODECA_CTRL2_LA_SHIFT: u32 = 7;
const CODECA_CTRL2_RA_SHIFT: u32 = 2;
const CODECA_CTRL2_ATTEN_MASK: u16 = 0x001F;

/// CS4216 5-bit DAC attenuator lookup table in Q15 fixed-point (32768 = 0.0 dB).
/// Each step is -1.5 dB: round(32768 * 10^(-1.5 * step / 20)).
const CODEC_ATTEN_Q15: [i32; 32] = [
    32768, 27571, 23198, 19519, 16423, 13818, 11627,  9783,
     8231,  6925,  5827,  4903,  4125,  3471,  2920,  2457,
     2068,  1740,  1464,  1232,  1036,   872,   734,   617,
      519,   437,   368,   309,   260,   219,   184,   155,
];

fn scale_sample(sample: i16, gain_q15_mdac: i64) -> i16 {
    const UNITY: i64 = 32768 * 255;
    if gain_q15_mdac >= UNITY {
        sample
    } else if gain_q15_mdac <= 0 {
        0
    } else {
        ((sample as i64 * gain_q15_mdac) / UNITY)
            .clamp(i16::MIN as i64, i16::MAX as i64) as i16
    }
}

// Pre-buffer: accumulate this many ms of audio samples before pushing to the ring.
// This gives the CPU time to fill its circular DMA buffer before we start draining it,
// preventing initial underrun.
const PREBUF_MS: u64 = 20;

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
    /// Frames of a stall `due` has skipped since the caller last took them.
    skipped: u64,
}

impl Pacer {
    fn new(rate: u32) -> Self {
        Self { start: Instant::now(), rate: rate as u64, done: 0, skipped: 0 }
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
        self.due_at(self.start.elapsed())
    }

    /// [`Pacer::due`] with `elapsed` as the time since the start.
    fn due_at(&mut self, elapsed: Duration) -> u64 {
        let target = (elapsed.as_nanos() as u64).saturating_mul(self.rate) / 1_000_000_000;
        let backlog = target.saturating_sub(self.done);
        let max_backlog = self.rate * PACE_MAX_BACKLOG.as_millis() as u64 / 1000;
        if backlog > max_backlog {
            // Skip the stall rather than replay it.
            self.skipped += target - max_backlog - self.done;
            self.done = target - max_backlog;
        }
        let per_period = (self.rate * PACE_PERIOD.as_micros() as u64).div_ceil(1_000_000);
        let cap = (per_period * PACE_CATCHUP_NUM).div_ceil(PACE_CATCHUP_DEN).max(per_period + 1);
        let due = target.saturating_sub(self.done).min(cap);
        self.done += due;
        due
    }

    /// The frames of stalls `due` has skipped, counted once.
    fn take_skipped(&mut self) -> u64 {
        std::mem::take(&mut self.skipped)
    }

    /// Frames `due` handed out but not moved after all: still due.
    fn give_back(&mut self, frames: u64) {
        self.done -= frames;
    }
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
}

impl CodecAState {
    fn new() -> Self {
        Self { out: None, resampler: None, prebuffering: true, prebuf: Vec::new(), dry: 0, nonzero_seen: false, timer_id: None }
    }
    fn reset_audio(&mut self) {
        // Keep `out` — stream stays open.  Just reset codec-side state.
        self.resampler = None;
        self.prebuffering = true;
        self.prebuf.clear();
        self.dry = 0;
        self.nonzero_seen = false;
        if let Some(o) = &self.out {
            o.meter.stopped();
        }
    }
    /// Push interleaved i16 stereo pairs through the resampler into the ring buffer.
    fn push_to_ring(&mut self, samples: &[i16]) {
        if let Some(rs) = &mut self.resampler {
            if let Some(o) = &mut self.out {
                for frame in samples.chunks_exact(2) {
                    o.push(rs, frame);
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

    /// External 8-bit Multiplying DAC (MDAC) volume registers at HPC3 PBUS
    /// PIO channel 2 (`0x1fbd8800` = `VOLUME_RIGHT`, `0x1fbd8804` = `VOLUME_LEFT`).
    /// Default to `0xFF` (unity gain) until written by PROM or the OS driver.
    vol_left: u8,
    vol_right: u8,
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

    /// Combined output gain `(left_gain, right_gain)` in `0..=(32768 * 255)`,
    /// combining the CS4216 Codec A `MUTE` bit (IDR0 bit 10), 5-bit DAC
    /// attenuators `LA`/`RA` (IDR1 bits 11:7 and 6:2, -1.5 dB/step), and the
    /// external 8-bit MDAC registers (`VOLUME_LEFT`, `VOLUME_RIGHT`).
    fn output_gains(&self) -> (i64, i64) {
        if (self.codeca_ctrl[1] & CODECA_CTRL2_MUTE) != 0 {
            return (0, 0);
        }
        let la = ((self.codeca_ctrl[2] >> CODECA_CTRL2_LA_SHIFT) & CODECA_CTRL2_ATTEN_MASK) as usize;
        let ra = ((self.codeca_ctrl[2] >> CODECA_CTRL2_RA_SHIFT) & CODECA_CTRL2_ATTEN_MASK) as usize;
        let gain_l = CODEC_ATTEN_Q15[la] as i64 * self.vol_left as i64;
        let gain_r = CODEC_ATTEN_Q15[ra] as i64 * self.vol_right as i64;
        (gain_l, gain_r)
    }
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
    meter: Arc<Meter>,
    // [audio]: which host to play through, and its buffer size.
    audio: AudioConfig,
    /// IRIS_AUDIO_DUMP=<file>: every frame codec A plays, at the codec's
    /// own rate, before resampling (see `audio_dump`).
    dump: Option<Arc<crate::dev::audio_dump::AudioDump>>,
}

// ─── cpal helpers ─────────────────────────────────────────────────────────────

fn prebuf_samples(rate: u32) -> usize {
    (rate as usize * 2 * PREBUF_MS as usize) / 1000
}

/// The host output, through the shared `host_audio`.
fn open_persistent_output(audio: &AudioConfig, meter: Arc<Meter>) -> Option<AudioOut> {
    crate::host_audio::open_output(audio, meter, "HAL2")
}

// ─── impl Hal2 ────────────────────────────────────────────────────────────────

impl Hal2 {
    pub fn new(dma_clients: Vec<Arc<dyn DmaClient>>) -> Self {
        Self::with_audio(dma_clients, AudioConfig::default())
    }

    /// HAL2 playing through the host and buffer `[audio]` asks for.
    pub fn with_audio(dma_clients: Vec<Arc<dyn DmaClient>>, audio: AudioConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(Hal2State {
                isr: 0,
                iar: 0,
                idr: [0; 4],
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
                vol_left: 0xFF,
                vol_right: 0xFF,
            })),
            dma_clients,
            timer_manager: Arc::new(std::sync::OnceLock::new()),
            ca_state: Arc::new(Mutex::new(CodecAState::new())),
            cb_state: Arc::new(Mutex::new(CodecBState { timer_id: None })),
            at_state: Arc::new(Mutex::new(AesTxState { timer_id: None })),
            ar_state: Arc::new(Mutex::new(AesRxState { loopback: std::collections::VecDeque::new(), timer_id: None })),
            meter: Meter::new("HAL2", &["analog in", "analog out"]),
            audio,
            dump: crate::dev::audio_dump::AudioDump::from_env(),
        }
    }

    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timer_manager.set(tm);
    }

    pub fn set_volume_left(&self, val: u8) {
        let mut s = self.state.lock();
        if s.vol_left != val {
            dlog_dev!(LogModule::Hal2, "HAL2: VOLUME_LEFT 0x{:02x} -> 0x{:02x}", s.vol_left, val);
            s.vol_left = val;
        }
    }

    pub fn set_volume_right(&self, val: u8) {
        let mut s = self.state.lock();
        if s.vol_right != val {
            dlog_dev!(LogModule::Hal2, "HAL2: VOLUME_RIGHT 0x{:02x} -> 0x{:02x}", s.vol_right, val);
            s.vol_right = val;
        }
    }

    pub fn volume_left(&self) -> u8 {
        self.state.lock().vol_left
    }

    pub fn volume_right(&self) -> u8 {
        self.state.lock().vol_right
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
            s.vol_left   = 0xFF;
            s.vol_right  = 0xFF;
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
        crate::pf_instant!(crate::perfetto::events::HAL2_ARM, 0 << 8 | dma_ch);

        let dma_client = self.dma_clients[dma_ch].clone();
        let state = self.state.clone();
        let ca_state = self.ca_state.clone();
        let dump = self.dump.clone();
        let mut pacer = Pacer::new(pitch_rate);

        let meter = self.meter.clone();
        let id = tm.add_recurring(Instant::now() + PACE_PERIOD, PACE_PERIOD, (), move |_| {
            let mut due = pacer.due();
            meter.skipped(1, Skip::Stale, pacer.take_skipped());
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
            crate::pf_scope!(crate::perfetto::events::HAL2_PACE, due);
            let (gain_l, gain_r) = state.lock().output_gains();
            let mut st = ca_state.lock();

            // No audio output — still drain DMA so the kernel doesn't hang
            // waiting for PDMA_CTRL_ACT to clear.
            let stream_rate = match st.out.as_ref() {
                Some(o) => o.stream_rate,
                None => {
                    drain_without_host(&dma_client, mode, due, &meter, dump.as_deref(), pitch_rate);
                    return TimerReturn::Continue;
                }
            };

            // (Re)build resampler if codec rate changed.
            // Use codec B rate as the declared input rate (experiment).
            if st.resampler.as_ref().map_or(true, |r| r.in_rate != pitch_rate) {
                st.resampler = Some(Resampler::new(pitch_rate, stream_rate, 2));
                dlog_dev!(LogModule::Hal2, "HAL2: Codec A resampler {}Hz (pitch={}) → {}Hz", rate, pitch_rate, stream_rate);
            }

            let (mut got, mut dry) = (0, 0);
            let mut played = dump.as_ref().map(|_| Vec::with_capacity(due as usize));
            for _ in 0..due {
                let frame = read_frame_from(&dma_client, mode);
                match frame {
                    Some((l, r)) => {
                        got += 1;
                        if let Some(p) = played.as_mut() {
                            p.push((l, r));
                        }
                        st.dry = 0;
                        if !st.nonzero_seen && (l != 0 || r != 0) {
                            dlog_dev!(LogModule::Hal2, "HAL2: Codec A first non-zero: l={} r={}", l, r);
                            st.nonzero_seen = true;
                        }
                        let l = scale_sample(l, gain_l);
                        let r = scale_sample(r, gain_r);
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
                        dry += 1;
                        // The HPC3 channel has nothing to read when it is not
                        // active (its chain ended or DMA was disabled) or its
                        // buffer is unreadable: nothing reaches the codec, so
                        // the host's silence is not an underrun.
                        if st.dry == 1 {
                            if let Some(o) = &st.out {
                                o.meter.stopped();
                            }
                        }
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

            }
            if let (Some(d), Some(p)) = (dump.as_ref(), played) {
                d.record(None, pitch_rate, p.into_iter());
            }

            meter.moved(1, got);
            meter.skipped(1, Skip::Dry, dry);
            TimerReturn::Continue
        });

        self.ca_state.lock().timer_id = Some(id);
    }

    fn disarm_codeca(&self) {
        let id = self.ca_state.lock().timer_id.take();
        if let Some(id) = id {
            if let Some(tm) = self.timer_manager.get() { tm.remove(id); }
            crate::pf_instant!(crate::perfetto::events::HAL2_DISARM, 0);
        }
        self.ca_state.lock().reset_audio(); // keeps `out` (stream stays open)
    }

    // ── Codec B input (silence writer) timer ─────────────────────────────────

    fn arm_codecb(&self) {
        let Some(tm) = self.timer_manager.get() else { return; };
        self.disarm_codecb();

        let (dma_ch, mode, rate) = {
            let s = self.state.lock();
            let (ch, clk, raw_mode) = s.codecb_cfg();
            let mode = if raw_mode != 0 {
                raw_mode
            } else if (s.isr & isr::CODEC_MODE as u16) != 0 {
                MODE_QUAD
            } else {
                MODE_STEREO
            };
            let rate = s.bres_rate(clk);
            (ch, mode, rate)
        };

        if rate == 0 || dma_ch >= self.dma_clients.len() { return; }
        crate::pf_instant!(crate::perfetto::events::HAL2_ARM, 1 << 8 | dma_ch);

        let dma_client = self.dma_clients[dma_ch].clone();
        let mut pacer = Pacer::new(rate);
        let meter = self.meter.clone();

        let id = tm.add_recurring(Instant::now() + PACE_PERIOD, PACE_PERIOD, (), move |_| {
            let due = pacer.due();
            meter.skipped(0, Skip::Stale, pacer.take_skipped());
            fill_silence(&dma_client, mode, due, &meter);
            TimerReturn::Continue
        });

        self.cb_state.lock().timer_id = Some(id);
    }

    fn disarm_codecb(&self) {
        let id = self.cb_state.lock().timer_id.take();
        if let Some(id) = id {
            if let Some(tm) = self.timer_manager.get() { tm.remove(id); }
            crate::pf_instant!(crate::perfetto::events::HAL2_DISARM, 1);
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
        crate::pf_instant!(crate::perfetto::events::HAL2_ARM, 2 << 8 | dma_ch);

        let dma_client = self.dma_clients[dma_ch].clone();
        let ar_state = self.ar_state.clone();
        let mut pacer = Pacer::new(rate);

        let id = tm.add_recurring(Instant::now() + PACE_PERIOD, PACE_PERIOD, (), move |_| {
            for _ in 0..pacer.due() {
                if let Some((val, st)) = dma_client.read() {
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
            crate::pf_instant!(crate::perfetto::events::HAL2_DISARM, 2);
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
        crate::pf_instant!(crate::perfetto::events::HAL2_ARM, 3 << 8 | dma_ch);

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
            crate::pf_instant!(crate::perfetto::events::HAL2_DISARM, 3);
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

        let is_read  = (val & IAR_ACCESS_READ) != 0;
        let typ      = val & IAR_TYPE_MASK;
        let num      = val & IAR_NUM_MASK;
        let param    = val & IAR_PARAM_MASK;
        let bres_idx = ((val >> 8) & 0xF) as usize;

        if is_read {
            let subword = val & 1; // 0 = R0 (first 16-bit word), 1 = R1 (second 16-bit word into IDR0)
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
                        IAR_PARAM_2 => if subword == 1 {
                            state.idr[0] = state.codeca_ctrl[2];
                        } else {
                            state.idr[0] = state.codeca_ctrl[1];
                            state.idr[1] = state.codeca_ctrl[2];
                        },
                        _ => {}
                    },
                    IAR_NUM_CODECB => match param {
                        IAR_PARAM_1 => state.idr[0] = state.codecb_ctrl[0],
                        IAR_PARAM_2 => if subword == 1 {
                            state.idr[0] = state.codecb_ctrl[2];
                        } else {
                            state.idr[0] = state.codecb_ctrl[1];
                            state.idr[1] = state.codecb_ctrl[2];
                        },
                        _ => {}
                    },
                    IAR_NUM_AES_TX => match param {
                        IAR_PARAM_1 => state.idr[0] = state.aestx_ctrl[0],
                        IAR_PARAM_2 => if subword == 1 {
                            state.idr[0] = state.aestx_ctrl[2];
                        } else {
                            state.idr[0] = state.aestx_ctrl[1];
                            state.idr[1] = state.aestx_ctrl[2];
                        },
                        _ => {}
                    },
                    IAR_NUM_AES_RX => match param {
                        IAR_PARAM_1 => state.idr[0] = state.aesrx_ctrl[0],
                        IAR_PARAM_2 => if subword == 1 {
                            state.idr[0] = state.aesrx_ctrl[2];
                        } else {
                            state.idr[0] = state.aesrx_ctrl[1];
                            state.idr[1] = state.aesrx_ctrl[2];
                        },
                        _ => {}
                    },
                    _ => {}
                },
                IAR_TYPE_BRES if bres_idx >= 1 && bres_idx <= 3 => {
                    let idx = bres_idx - 1;
                    match param {
                        IAR_PARAM_1 => state.idr[0] = state.bres_clock_sel[idx],
                        IAR_PARAM_2 => if subword == 1 {
                            state.idr[0] = state.bres_clock_modctrl[idx];
                        } else {
                            state.idr[0] = state.bres_clock_inc[idx];
                            state.idr[1] = state.bres_clock_modctrl[idx];
                        },
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
        use crate::locks::register_mutex_fn;
        register_mutex_fn("hal2::state",    self, |h| h.state.is_locked());
        register_mutex_fn("hal2::ca_state", self, |h| h.ca_state.is_locked());
        register_mutex_fn("hal2::cb_state", self, |h| h.cb_state.is_locked());
        register_mutex_fn("hal2::at_state", self, |h| h.at_state.is_locked());
        register_mutex_fn("hal2::ar_state", self, |h| h.ar_state.is_locked());
    }
}

// ─── DMA read helper (free function to avoid borrow issues in closures) ───────

/// With no host output, still read what the channel has so the guest's DMA
/// completes, and count it: the card is playing while the channel has frames.
fn drain_without_host(client: &Arc<dyn DmaClient>, mode: usize, due: u64, meter: &Meter, dump: Option<&AudioDump>, rate: u32) {
    let frames = (0..due).filter_map(|_| read_frame_from(client, mode));
    let got = match dump {
        Some(d) => {
            let played: Vec<_> = frames.collect();
            d.record(None, rate, played.iter().copied());
            played.len() as u64
        }
        None => frames.count() as u64,
    };
    meter.moved(1, got);
    meter.skipped(1, Skip::Dry, due - got);
    if got > 0 {
        meter.started();
    } else {
        meter.stopped();
    }
}

/// Codec B's input: write `due` frames of silence to the channel, counting
/// the frames it took as moved and those it refused as skipped (its ring is
/// full or the channel is not active).
fn fill_silence(client: &Arc<dyn DmaClient>, mode: usize, due: u64, meter: &Meter) {
    let words = match mode { MODE_QUAD => 4, MODE_STEREO => 2, _ => 1 };
    let mut refused = 0;
    for _ in 0..due {
        // Every word is written, so the channel sees the whole frame.
        let ok = (0..words).fold(true, |ok, _| !client.write(0, false).refused() && ok);
        refused += u64::from(!ok);
    }
    meter.moved(0, due - refused);
    meter.skipped(0, Skip::Full, refused);
}

fn read_frame_from(client: &Arc<dyn DmaClient>, mode: usize) -> Option<(i16, i16)> {
    if mode == MODE_MONO {
        let (v, st) = client.read()?;
        if st.refused() { return None; }
        let s = v as i16;
        Some((s, s))
    } else {
        let (lv, lst) = client.read()?;
        if lst.refused() { return None; }
        let (rv, rst) = match client.read() {
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
        let audio = open_persistent_output(&self.audio, self.meter.clone());
        if audio.is_none() && self.audio.plays() {
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

    /// Only codec A plays to the host.
    fn audio_report(&self) -> Option<String> {
        // A codec's rate is the guest's while its timer runs; codec A's also
        // needs its channel to be supplying frames.
        let (armed_b, armed_a) = (self.cb_state.lock().timer_id.is_some(), self.ca_state.lock().timer_id.is_some());
        let rates = {
            let s = self.state.lock();
            let rate = |(_, clk, _): (usize, usize, usize), armed: bool| armed.then(|| s.bres_rate(clk)).filter(|&r| r > 0);
            [rate(s.codecb_cfg(), armed_b), rate(s.codeca_cfg(), armed_a && self.meter.playing())]
        };
        Some(self.meter.report(&rates, self.ca_state.lock().out.as_ref(), self.audio.plays()))
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

    /// What reaches the host ring from codec A, a frame at a time as the
    /// timer pass pushes it and in the burst that flushes the prebuffer, is
    /// fixed.
    #[test]
    fn codec_a_output_is_fixed() {
        use crate::host_audio::test_signal::*;
        for (mode, name) in [(Push::One, "hal2-one"), (Push::Burst, "hal2-burst")] {
            for (tight, run) in [(false, "free"), (true, "tight")] {
                let mut g = Golden::new(&format!("{name}/{run}"));
                let (slots, every) = golden_ring(tight, mode);
                for (inr, outr) in golden_pairs() {
                    for sig in golden_signals(inr) {
                        let (p, mut c) = RingBuffer::<i16>::new(slots);
                        let out = AudioOut::detached(Meter::new("test", &[]), outr, p);
                        let meter = out.meter.clone();
                        let mut st = CodecAState::new();
                        st.out = Some(out);
                        st.resampler = Some(Resampler::new(inr, outr, 2));
                        for (i, chunk) in golden_chunks(&sig, mode).iter().enumerate() {
                            let pcm: Vec<i16> = chunk.iter().flat_map(|&(l, r)| [l, r]).collect();
                            st.push_to_ring(&pcm);
                            if i % every == 0 {
                                g.samples(&golden_drain(&mut c));
                            }
                        }
                        g.samples(&golden_drain(&mut c));
                        g.dropped(meter.dropped());
                    }
                }
                g.check();
            }
        }
    }

    /// Cost of one frame through codec A's push path, as the timer pass pays
    /// it (a frame at a time). Run in a release build:
    /// `cargo test --release -p iris --lib -- --ignored hal2_push_cost --nocapture`
    #[test]
    #[ignore]
    fn hal2_push_cost() {
        const FRAMES: usize = 10_000_000;
        const BATCH: usize = 512;
        for (inr, outr) in [(44100u32, 48000u32), (11025, 48000), (8000, 48000)] {
            let (p, mut c) = RingBuffer::<i16>::new(1 << 16);
            let out = AudioOut::detached(Meter::new("bench", &[]), outr, p);
            let mut st = CodecAState::new();
            st.out = Some(out);
            st.resampler = Some(Resampler::new(inr, outr, 2));
            let mut spent = std::time::Duration::ZERO;
            let mut x = 1i16;
            for _ in 0..FRAMES / BATCH {
                let t = std::time::Instant::now();
                for _ in 0..BATCH {
                    x = x.wrapping_mul(75).wrapping_add(74);
                    st.push_to_ring(std::hint::black_box(&[x, !x]));
                }
                spent += t.elapsed();
                while c.pop().is_ok() {}
            }
            println!("BENCH hal2 push {inr}->{outr}: {:.2} ns per frame", spent.as_nanos() as f64 / FRAMES as f64);
        }
    }

    /// `audio status` reports codec A's frames, rate and the host stream.
    #[test]
    fn audio_status_reports_codec_a_and_the_host_stream() {
        let h = Hal2::new(vec![]);
        {
            let mut s = h.state.lock();
            s.bres_clock_rate[0] = 48000;
            s.codeca_ctrl[0] = (1 << CTRL1_CLOCK_SHIFT | (MODE_STEREO as u16) << CTRL1_MODE_SHIFT) as u16;
        }
        h.meter.moved(1, 12);
        let (prod, _cons) = RingBuffer::<i16>::new(64);
        h.ca_state.lock().out = Some(AudioOut::detached(h.meter.clone(), 44100, prod));
        // The rate is the guest's only while codec A's timer runs and its
        // channel supplies frames.
        h.ca_state.lock().timer_id = Some(1);
        assert!(h.audio_report().unwrap().contains("analog out: 12 frames moved, guest rate off\n"));
        h.ca_state.lock().out.as_mut().unwrap().push_frame(&[1, 2]);
        assert!(h.audio_report().unwrap().contains("analog out: 12 frames moved, guest rate 48000 Hz\n"));
        h.ca_state.lock().timer_id = None;
        h.meter.stopped();
        assert_eq!(
            h.audio_report().as_deref(),
            Some("audio: HAL2\n\
             \x20 analog in: 0 frames moved, guest rate off\n\
             \x20 analog out: 12 frames moved, guest rate off\n\
             \x20 host underruns: 0 frames\n\
             \x20 dropped by the host: 0 frames\n\
             \x20 host output: test at 44100 Hz, 2 channels\n")
        );
    }

    /// A HAL2 with `[audio] output = "none"` opens no host stream, says why in
    /// its report and does not complain that none is available.
    #[test]
    fn output_none_opens_no_stream_and_the_report_says_so() {
        let none = AudioConfig { output: Some(crate::config::AudioOutput::None), ..AudioConfig::default() };
        let h = Hal2::with_audio(vec![], none);
        h.start();
        assert!(h.ca_state.lock().out.is_none());
        assert!(h.audio_report().unwrap().ends_with("host output: none (output = \"none\")\n"));
        // A card that plays but has no stream open is a different state.
        let playing = Hal2::new(vec![]);
        assert!(playing.audio_report().unwrap().ends_with("host output: none (not open)\n"));
    }

    /// Disabling codec A's DMA ends the card's playing: the host's silence
    /// after a sound is not an underrun, and the next frame starts it again.
    #[test]
    fn disabling_codec_a_ends_playing() {
        let h = Hal2::new(vec![]);
        let (prod, mut cons) = RingBuffer::<i16>::new(64);
        h.ca_state.lock().out = Some(AudioOut::detached(h.meter.clone(), 48000, prod));
        let underrun = |h: &Hal2, cons: &mut rtrb::Consumer<i16>| {
            crate::host_audio::count_underrun(crate::host_audio::pop_frames(cons, 2, &mut [0i16; 2], |v| v), &h.meter);
            h.meter.underruns()
        };
        assert_eq!(underrun(&h, &mut cons), 0, "idle");
        h.ca_state.lock().out.as_mut().unwrap().push_frame(&[1, 2]);
        cons.pop().unwrap();
        cons.pop().unwrap();
        assert_eq!(underrun(&h, &mut cons), 1, "playing");
        h.disarm_codeca();
        assert_eq!(underrun(&h, &mut cons), 1, "disabled");
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

    /// A stall past PACE_MAX_BACKLOG is dropped, not replayed, and the
    /// pacer says how many frames that was: due plus skipped is the time
    /// that passed, less the backlog it still owes.
    #[test]
    fn pacer_counts_the_stall_it_skips() {
        let mut p = Pacer::new(11025);
        // The time since the start is handed in, so no hold-up of the test
        // thread can move it.
        let first = p.due_at(Duration::from_secs(5));
        let skipped = p.take_skipped();
        let backlog = 11025 * PACE_MAX_BACKLOG.as_millis() as u64 / 1000;
        // All of the 5 s but the backlog the pacer still owes is skipped;
        // the first handout is a capped part of that backlog.
        assert_eq!(skipped, 11025 * 5 - backlog);
        assert!(first >= 2 && first < backlog, "{first} handed out");
        assert_eq!(p.take_skipped(), 0, "taken once");
        // A frame's time later (90.7 us) the backlog is still inside its
        // bound: no stall. The handout is capped at a period and a quarter,
        // so a look more than that late would be one.
        let _ = p.due_at(Duration::from_secs(5) + Duration::from_micros(90));
        assert_eq!(p.take_skipped(), 0, "no stall, nothing skipped");
    }

}

/// The host end of codec A, as the Indy runs it.
#[cfg(test)]
mod host_tests {
    use super::*;
    use crate::host_audio::test_signal::*;
    use crate::traits::{DmaClient, DmaStatus};

    /// An HPC3 channel whose descriptor chain loops one buffer of 16-bit
    /// samples, each in the low half of its word, as the guest leaves it.
    /// While `late` is set, the guest has fallen behind: from the next
    /// whole frame on, the channel has nothing to give.
    struct Looping {
        words: Vec<u32>,
        at: std::sync::atomic::AtomicUsize,
        late: AtomicBool,
    }

    impl DmaClient for Looping {
        fn read(&self) -> Option<(u32, DmaStatus)> {
            if self.late.load(Ordering::Relaxed) && self.at.load(Ordering::Relaxed) % 2 == 0 {
                return None;
            }
            let i = self.at.fetch_add(1, Ordering::Relaxed);
            Some((self.words[i % self.words.len()], DmaStatus::ok()))
        }
        fn write(&self, _val: u32, _eop: bool) -> DmaStatus {
            DmaStatus::ok()
        }
    }

    /// The whole way from the DMA channel to the host as a running machine
    /// takes it: codec A on BRES1 at 48 kHz, its recurring timer pacing the
    /// channel by the host's clock, the prebuffer, and the host output
    /// `Device::start` opens -- a file output, so no sound card --
    /// drained in real time. At 48 kHz nothing is resampled: what the file
    /// holds, less the underrun silence, is the buffer sample for sample
    /// with each channel on its own side.
    #[test]
    fn a_tone_in_guest_memory_reaches_the_host_on_the_real_clock() {
        // Five and ten cycles fill the 512-frame buffer exactly.
        let (left, right) = (5.0 * 48000.0 / 512.0, 10.0 * 48000.0 / 512.0);
        let tone = tones(512, 48000, left, right, 16000.0);
        let dma = Arc::new(Looping { words: tone.iter().map(|&v| u32::from(v as u16)).collect(), at: Default::default(), late: AtomicBool::new(false) });
        let (cfg, path) = file_config("hal2-tone");
        let h = Hal2::with_audio(vec![dma.clone()], cfg);
        h.set_timer_manager(Arc::new(TimerManager::new()));
        Device::start(&h);
        {
            let mut s = h.state.lock();
            s.bres_clock_rate[0] = 48000;
            // DMA channel 0, CLKID 1, stereo.
            s.codeca_ctrl[0] = (1 << CTRL1_CLOCK_SHIFT | (MODE_STEREO as u16) << CTRL1_MODE_SHIFT) as u16;
            s.dma_enable = DMA_EN_CODECA;
        }
        // The host can hold up the whole process, and the channel skips a
        // stall of more than PACE_MAX_BACKLOG by design. So the test expects
        // of the clock the time between its own looks at it, less the
        // part of a hold-up the clock skips, not the wall time from start to
        // end.
        let (mut last, mut credit) = (Instant::now(), Duration::ZERO);
        h.apply_dma_enable(0, DMA_EN_CODECA);
        let end = last + Duration::from_millis(800);
        while last < end {
            std::thread::sleep(Duration::from_millis(5));
            let now = Instant::now();
            credit += (now - last).min(PACE_MAX_BACKLOG);
            last = now;
        }
        Device::stop(&h);
        let moved = dma.at.load(Ordering::Relaxed) as u64 / 2;
        let dropped = h.meter.dropped();
        let (rate, channels, pcm) = read_wav(&path.0);
        assert_eq!((rate, channels), (48000, 2));
        // Half at least of what the looks allowed: a clock that does not
        // keep to the host's holds the count back.
        assert!(moved as f64 >= 0.5 * credit.as_secs_f64() * 48000.0, "the clock ran: {moved} frames in {:.3} s of looks", credit.as_secs_f64());
        let heard = sounding(&pcm, 2);
        assert!((heard.len() / 2) as u64 + dropped + 9600 >= moved, "{} of {moved} frames reached the file", heard.len() / 2);
        // A frame the host ring refused leaves a hole in the tone: the order
        // cannot be told then.
        if dropped != 0 {
            eprintln!("not checking the order or the pitch: the host ring refused {dropped} frames");
            return;
        }
        let want: Vec<i16> = sounding(&tone, 2).into_iter().cycle().take(heard.len()).collect();
        assert_eq!(heard, want, "the channel's samples, in order, on their own sides");
        assert_tones(&heard, 2, 48000, left, right, 15000);
    }

    /// With no host output the card still counts the frames its channel
    /// supplies, is playing while it supplies them, and stops when it does not.
    #[test]
    fn a_card_with_no_host_output_plays_while_its_channel_has_frames() {
        let dma: Arc<Looping> = Arc::new(Looping { words: vec![0x1234; 8], at: Default::default(), late: AtomicBool::new(false) });
        let client: Arc<dyn DmaClient> = dma.clone();
        let meter = Meter::new("test", &["analog in", "analog out"]);
        drain_without_host(&client, MODE_STEREO, 4, &meter, None, 44100);
        assert_eq!((meter.frames(1), meter.playing()), (4, true));
        dma.late.store(true, Ordering::Relaxed);
        drain_without_host(&client, MODE_STEREO, 4, &meter, None, 44100);
        assert_eq!((meter.frames(1), meter.playing()), (4, false));
        assert_eq!(meter.skips(1, Skip::Dry), 4, "the frames the channel had none for");
    }

    /// Codec B writes silence into its channel; a frame the channel refuses
    /// is skipped, not moved.
    #[test]
    fn codec_b_counts_the_frames_its_channel_refused() {
        struct Refusing(AtomicBool);
        impl DmaClient for Refusing {
            fn read(&self) -> Option<(u32, DmaStatus)> { None }
            fn write(&self, _val: u32, _eop: bool) -> DmaStatus {
                if self.0.load(Ordering::Relaxed) { DmaStatus(DmaStatus::NOT_ACTIVE) } else { DmaStatus::ok() }
            }
        }
        let dma = Arc::new(Refusing(AtomicBool::new(false)));
        let client: Arc<dyn DmaClient> = dma.clone();
        let meter = Meter::new("test", &["analog in", "analog out"]);
        fill_silence(&client, MODE_STEREO, 5, &meter);
        assert_eq!((meter.frames(0), meter.skips(0, Skip::Full)), (5, 0));
        dma.0.store(true, Ordering::Relaxed);
        fill_silence(&client, MODE_STEREO, 3, &meter);
        assert_eq!((meter.frames(0), meter.skips(0, Skip::Full)), (5, 3));
    }

    /// With `IRIS_AUDIO_DUMP` set, codec A records the frames it reads from
    /// its DMA channel at the codec's rate, host stream or not: here none is
    /// open, and the file holds the buffer sample for sample.
    #[test]
    fn the_dump_holds_what_codec_a_played_with_no_host_stream() {
        use crate::dev::audio_dump::{tests::{records, temp}, AudioDump};
        let tone = tones(512, 48000, 5.0 * 48000.0 / 512.0, 10.0 * 48000.0 / 512.0, 16000.0);
        let dma = Arc::new(Looping { words: tone.iter().map(|&v| u32::from(v as u16)).collect(), at: Default::default(), late: AtomicBool::new(false) });
        let path = temp("hal2");
        let mut h = Hal2::new(vec![dma.clone()]);
        h.dump = AudioDump::create(path.as_os_str()).map(Arc::new);
        h.set_timer_manager(Arc::new(TimerManager::new()));
        {
            let mut s = h.state.lock();
            s.bres_clock_rate[0] = 48000;
            s.codeca_ctrl[0] = (1 << CTRL1_CLOCK_SHIFT | (MODE_STEREO as u16) << CTRL1_MODE_SHIFT) as u16;
            s.dma_enable = DMA_EN_CODECA;
        }
        h.apply_dma_enable(0, DMA_EN_CODECA);
        // Wait for a few buffers' worth, not a fixed time: 8 bytes of record
        // header and 4 of sample per frame.
        let deadline = Instant::now() + Duration::from_secs(10);
        while std::fs::metadata(&path).map_or(0, |m| m.len()) < 4 * 2 * 512 + 64 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        Device::stop(&h);
        let recs = records(&std::fs::read(&path).unwrap());
        let _ = std::fs::remove_file(&path);
        assert!(!recs.is_empty() && recs.iter().all(|(rate, _)| *rate == 48000));
        let played: Vec<i16> = recs.iter().flat_map(|(_, f)| f.iter().flat_map(|&(l, r)| [l, r])).collect();
        assert!(played.len() > 2 * 512, "{} samples", played.len());
        let want: Vec<i16> = tone.iter().copied().cycle().take(played.len()).collect();
        assert_eq!(played, want);
    }

    /// Codec A playing a 440 Hz tone on the left and 880 Hz on the right,
    /// its HPC3 channel running out for 0.3 s (a chain that ended, which the
    /// channel reports as nothing to read) so the host stream runs dry, then
    /// refilled (see [`a_tone_survives_a_pause`]).
    struct Codec {
        h: Hal2,
        dma: Arc<Looping>,
        started: AtomicBool,
    }

    impl CardUnderTest for Codec {
        const NAME: &'static str = "hal2-underrun";
        const PAUSE_IS_STALL: bool = false;
        fn build(cfg: &AudioConfig) -> Self {
            // 11 and 22 cycles fill the 1200-frame buffer exactly.
            let tone = tones(1200, 48000, 440.0, 880.0, 16000.0);
            let dma = Arc::new(Looping { words: tone.iter().map(|&v| u32::from(v as u16)).collect(), at: Default::default(), late: AtomicBool::new(false) });
            let h = Hal2::with_audio(vec![dma.clone()], cfg.clone());
            h.set_timer_manager(Arc::new(TimerManager::new()));
            Device::start(&h);
            {
                let mut s = h.state.lock();
                s.bres_clock_rate[0] = 48000;
                s.codeca_ctrl[0] = (1 << CTRL1_CLOCK_SHIFT | (MODE_STEREO as u16) << CTRL1_MODE_SHIFT) as u16;
                s.dma_enable = DMA_EN_CODECA;
            }
            Self { h, dma, started: AtomicBool::new(false) }
        }
        fn tone(&self) -> (f64, f64) {
            (440.0, 880.0)
        }
        fn rate(&self) -> f64 {
            48000.0
        }
        fn play(&self) {
            self.dma.late.store(false, Ordering::Relaxed);
            if !self.started.swap(true, Ordering::Relaxed) {
                self.h.apply_dma_enable(0, DMA_EN_CODECA);
            }
        }
        /// The channel runs out at the next whole frame, so the pause is
        /// exact before the card's count is read.
        fn pause(&self) {
            self.dma.late.store(true, Ordering::Relaxed);
            let mut tries = 0;
            while self.dma.at.load(Ordering::Relaxed) % 2 == 1 {
                tries += 1;
                assert!(tries < 500, "the channel never reached a whole frame");
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        fn wait(&self, ms: u64) {
            std::thread::sleep(Duration::from_millis(ms));
        }
        fn frames(&self) -> f64 {
            self.dma.at.load(Ordering::Relaxed) as f64 / 2.0
        }
        fn meter(&self) -> &Meter {
            &self.h.meter
        }
        fn finish(&self) {
            Device::stop(&self.h);
        }
    }

    #[test]
    fn a_tone_survives_a_pause_into_a_file() {
        a_tone_survives_a_pause::<Codec>(Backend::File);
    }

    #[test]
    fn a_tone_survives_a_pause_on_the_real_device() {
        a_tone_survives_a_pause::<Codec>(Backend::RealDevice);
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

    #[test]
    fn ctrl2_r0_and_r1_read_back_both_words_via_idr0() {
        let hal2 = Hal2::default();
        // Write Codec A CTRL2 (0x1408): IDR0 = 0x03ab, IDR1 = 0x05cd
        hal2.write(HAL2_IDR0, 0x03ab);
        hal2.write(HAL2_IDR1, 0x05cd);
        hal2.write(HAL2_IAR, 0x1408);

        // Clear IDRs, then read R0 (0x1488) and R1 (0x1489) via IDR0
        hal2.write(HAL2_IDR0, 0);
        hal2.write(HAL2_IDR1, 0);
        hal2.write(HAL2_IAR, 0x1488);
        assert_eq!(hal2.read(HAL2_IDR0).data, 0x03ab);

        hal2.write(HAL2_IDR0, 0);
        hal2.write(HAL2_IAR, 0x1489);
        assert_eq!(hal2.read(HAL2_IDR0).data, 0x05cd);

        // Same for BRES1 CTRL2 (0x2108 -> 0x2188 / 0x2189): 32 kHz (inc=2, denom=3, modctrl=0xfffe)
        hal2.write(HAL2_IDR0, 0);
        hal2.write(HAL2_IAR, 0x2104); // 48 kHz master
        hal2.write(HAL2_IDR0, 2);
        hal2.write(HAL2_IDR1, 0xfffe);
        hal2.write(HAL2_IAR, 0x2108);
        assert_eq!(hal2.state.lock().bres_rate(1), 32000);

        hal2.write(HAL2_IDR0, 0);
        hal2.write(HAL2_IAR, 0x2188);
        assert_eq!(hal2.read(HAL2_IDR0).data, 2);
        hal2.write(HAL2_IAR, 0x2189);
        assert_eq!(hal2.read(HAL2_IDR0).data, 0xfffe);
    }

    #[test]
    fn mdac_volume_and_cs4216_dac_attenuation_scale_samples() {
        let hal2 = Hal2::default();
        // Default: unmuted, 0 dB attenuation, MDAC = 0xFF -> exact passthrough
        let (gl, gr) = hal2.state.lock().output_gains();
        assert_eq!(scale_sample(16000, gl), 16000);
        assert_eq!(scale_sample(-16000, gr), -16000);

        // MDAC volume scaling (IRIX a2_dd.o / PROM / NetBSD haltwo)
        hal2.set_volume_left(0);
        hal2.set_volume_right(64);
        let (gl, gr) = hal2.state.lock().output_gains();
        assert_eq!(scale_sample(16000, gl), 0);
        assert_eq!(scale_sample(25500, gr), 6400);

        // Restore MDAC to 255 and test CS4216 5-bit DAC attenuator (4 steps = -6.0 dB, ~0.501x)
        hal2.set_volume_left(255);
        hal2.set_volume_right(255);
        hal2.write(HAL2_IDR0, 0);
        hal2.write(HAL2_IDR1, (4 << CODECA_CTRL2_LA_SHIFT) | (15 << CODECA_CTRL2_RA_SHIFT));
        hal2.write(HAL2_IAR, 0x1408);
        let (gl, gr) = hal2.state.lock().output_gains();
        assert_eq!(scale_sample(32767, gl), 16422);
        assert_eq!(scale_sample(32767, gr), 2456);

        // CS4216 MUTE bit (IDR0 bit 10 = 0x0400) silences both channels
        hal2.write(HAL2_IDR0, CODECA_CTRL2_MUTE);
        hal2.write(HAL2_IDR1, 0);
        hal2.write(HAL2_IAR, 0x1408);
        let (gl, gr) = hal2.state.lock().output_gains();
        assert_eq!((gl, gr), (0, 0));
        assert_eq!(scale_sample(32767, gl), 0);
    }
}
