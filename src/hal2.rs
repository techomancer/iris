use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
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
const CTRL1_CLOCK_SHIFT: u32 = 3;    // bits 4:3 – BRES clock index (0-based → BRES1..3)
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

// Sample rates to try when opening the persistent output stream, in order.
const PREFERRED_RATES: &[u32] = &[48000, 44100, 22050];

// ─── Audio output (owned by Codec A, opened once at start, closed at stop) ───

// Opened once at `start()` at the best available host rate; the codec A timer
// pushes i16 stereo pairs through a resampler into the ring buffer producer.
// The stream plays silence when the ring is empty (cpal fills with 0).
struct AudioOut {
    stream_rate: u32,
    producer: Producer<i16>,
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
struct Resampler {
    in_rate: u32,
    out_rate: u32,
    // Accumulator: tracks fractional position in output-sample units * in_rate.
    // We advance by out_rate each input sample and emit whenever acc >= in_rate.
    acc: u64,
    // Last seen sample (for repeat)
    last_l: i16,
    last_r: i16,
}

impl Resampler {
    fn new(in_rate: u32, out_rate: u32) -> Self {
        Self { in_rate, out_rate, acc: 0, last_l: 0, last_r: 0 }
    }

    fn passthrough(&self) -> bool { self.in_rate == self.out_rate }

    /// Push one input stereo pair. Returns 0, 1, or 2 output pairs via the closure.
    fn push(&mut self, l: i16, r: i16, prod: &mut Producer<i16>) {
        self.last_l = l;
        self.last_r = r;
        if self.passthrough() {
            let _ = prod.push(l);
            let _ = prod.push(r);
            return;
        }
        self.acc += self.out_rate as u64;
        // Emit one output sample for each full in_rate unit accumulated
        while self.acc >= self.in_rate as u64 {
            self.acc -= self.in_rate as u64;
            let _ = prod.push(l);
            let _ = prod.push(r);
        }
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
        Self { out: None, resampler: None, prebuffering: true, prebuf: Vec::new(),
               dry: 0, nonzero_seen: false, timer_id: None }
    }
    fn reset_audio(&mut self) {
        // Keep `out` — stream stays open.  Just reset codec-side state.
        self.resampler = None;
        self.prebuffering = true;
        self.prebuf.clear();
        self.dry = 0;
        self.nonzero_seen = false;
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
}

impl Hal2State {
    fn bres_rate(&self, clk: usize) -> u32 {
        if clk < 3 { self.bres_clock_rate[clk] } else { 44100 }
    }
    fn codeca_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.codeca_ctrl[0]) }
    fn codecb_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.codecb_ctrl[0]) }
    fn aestx_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.aestx_ctrl[0]) }
    fn aesrx_cfg(&self) -> (usize, usize, usize) { decode_ctrl1(self.aesrx_ctrl[0]) }
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
}

// ─── cpal helpers ─────────────────────────────────────────────────────────────

fn prebuf_samples(rate: u32) -> usize {
    (rate as usize * 2 * PREBUF_MS as usize) / 1000
}

/// Open a persistent stereo i16 cpal output stream, trying PREFERRED_RATES in order.
/// The stream plays silence when the ring buffer is empty.
fn open_persistent_output() -> Option<AudioOut> {
    let host = cpal::default_host();
    let device = host.default_output_device()?;

    for &rate in PREFERRED_RATES {
        let config = cpal::StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(rate),
            buffer_size: cpal::BufferSize::Default,
        };
        let ring_size = prebuf_samples(rate) * RING_BUF_MULTIPLIER;
        let err_fn = |err: cpal::StreamError| { eprintln!("HAL2: cpal stream error: {:?}", err); };

        // Try f32 first (macOS CoreAudio native), then i16 (Linux ALSA).
        let (producer, stream) = {
            let (p, mut c) = RingBuffer::<i16>::new(ring_size);
            let data_fn = move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for sample in data.iter_mut() {
                    *sample = c.pop().unwrap_or(0) as f32 / 32768.0;
                }
            };
            match device.build_output_stream(&config, data_fn, err_fn.clone(), None) {
                Ok(s) => (p, s),
                Err(_) => {
                    // f32 failed, try i16
                    let (p, mut c) = RingBuffer::<i16>::new(ring_size);
                    let data_fn = move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                        for sample in data.iter_mut() {
                            *sample = c.pop().unwrap_or(0);
                        }
                    };
                    match device.build_output_stream(&config, data_fn, err_fn.clone(), None) {
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
        println!("HAL2: audio output: {:?} via {:?} at {}Hz",
            device.name().unwrap_or_default(), host.id(), rate);
        return Some(AudioOut {
            stream_rate: rate,
            producer,
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
        }
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
            let pitch_rate = if cb_rate > 0 { cb_rate } else { 44100 };
            (ch, mode, rate, pitch_rate)
        };

        if rate == 0 || dma_ch >= self.dma_clients.len() { return; }

        let period = Duration::from_secs_f64(1.0 / pitch_rate as f64);
        let dma_client = self.dma_clients[dma_ch].clone();
        let ca_state = self.ca_state.clone();

        let id = tm.add_recurring(Instant::now() + period, period, (), move |_| {
            let mut st = ca_state.lock();

            // No audio output — still drain DMA so the kernel doesn't hang
            // waiting for PDMA_CTRL_ACT to clear.
            let stream_rate = match st.out.as_ref() {
                Some(o) => o.stream_rate,
                None => {
                    let _ = read_frame_from(&dma_client, mode);
                    return TimerReturn::Continue;
                }
            };

            // (Re)build resampler if codec rate changed.
            // Use codec B rate as the declared input rate (experiment).
            if st.resampler.as_ref().map_or(true, |r| r.in_rate != pitch_rate) {
                st.resampler = Some(Resampler::new(pitch_rate, stream_rate));
                dlog_dev!(LogModule::Hal2, "HAL2: Codec A resampler {}Hz (pitch={}) → {}Hz", rate, pitch_rate, stream_rate);
            }

            let frame = read_frame_from(&dma_client, mode);

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

        let period = Duration::from_secs_f64(1.0 / rate as f64);
        let dma_client = self.dma_clients[dma_ch].clone();

        let id = tm.add_recurring(Instant::now() + period, period, (), move |_| {
            let _ = dma_client.write(0, false);
            if mode == MODE_STEREO || mode == MODE_QUAD {
                let _ = dma_client.write(0, false);
            }
            if mode == MODE_QUAD {
                let _ = dma_client.write(0, false);
                let _ = dma_client.write(0, false);
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

        let period = Duration::from_secs_f64(1.0 / rate as f64);
        let dma_client = self.dma_clients[dma_ch].clone();
        let ar_state = self.ar_state.clone();

        let id = tm.add_recurring(Instant::now() + period, period, (), move |_| {
            if let Some((val, st, _)) = dma_client.read() {
                if !st.refused() {
                    ar_state.lock().loopback.push_back(val);
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

        let period = Duration::from_secs_f64(1.0 / rate as f64);
        let dma_client = self.dma_clients[dma_ch].clone();
        let ar_state = self.ar_state.clone();

        let id = tm.add_recurring(Instant::now() + period, period, (), move |_| {
            let val = ar_state.lock().loopback.pop_front().unwrap_or(0);
            let _ = dma_client.write(val, false);
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
            let inc = state.bres_clock_inc[i] as u32;
            let modctrl = state.bres_clock_modctrl[i] as u32;
            let mod_val = inc.wrapping_sub(modctrl).wrapping_sub(1) & 0xFFFF;
            if inc == 0 {
                state.bres_clock_rate[i] = 0;
            } else if mod_val > 0 {
                state.bres_clock_rate[i] = (master * inc) / mod_val;
            }
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
                        self.reclock_active(idx);
                        return;
                    }
                }
                _ => {}
            }
        }
    }

    /// Re-arm any active channels that use BRES clock `bres_idx` (0-based).
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

        // Rearm codec A if its own clock changed, or if codec B's clock changed
        // (codec A's timer period uses codec B's rate as pitch_rate).
        if (dma_enable & DMA_EN_CODECA) != 0 && (ca_clk == bres_idx || cb_clk == bres_idx) {
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
        let audio = open_persistent_output();
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

                let (ca_ch, ca_clk, ca_mode) = s.codeca_cfg();
                writeln!(writer, "Codec A: ch={} bres={} rate={}Hz mode={}",
                    ca_ch, ca_clk + 1, s.bres_rate(ca_clk), mode_str(ca_mode)).unwrap();
                writeln!(writer, "  ctrl1=0x{:04x} ctrl2=[0x{:04x} 0x{:04x}]",
                    s.codeca_ctrl[0], s.codeca_ctrl[1], s.codeca_ctrl[2]).unwrap();

                let (cb_ch, cb_clk, cb_mode) = s.codecb_cfg();
                writeln!(writer, "Codec B: ch={} bres={} rate={}Hz mode={}",
                    cb_ch, cb_clk + 1, s.bres_rate(cb_clk), mode_str(cb_mode)).unwrap();

                let (at_ch, at_clk, _) = s.aestx_cfg();
                let (ar_ch, ar_clk, _) = s.aesrx_cfg();
                writeln!(writer, "AES TX: ch={} bres={} rate={}Hz", at_ch, at_clk + 1, s.bres_rate(at_clk)).unwrap();
                writeln!(writer, "AES RX: ch={} bres={} rate={}Hz", ar_ch, ar_clk + 1, s.bres_rate(ar_clk)).unwrap();
                drop(s);

                let ca = self.ca_state.lock();
                writeln!(writer, "Codec A out: {}  pitch: {}  prebuf: {}  prebuffering: {}  timer: {}",
                    ca.out.as_ref().map_or("none".to_string(), |o| format!("{}Hz", o.stream_rate)),
                    ca.resampler.as_ref().map_or("none".to_string(), |r| format!("{}Hz", r.in_rate)),
                    ca.prebuf.len() / 2,
                    ca.prebuffering,
                    ca.timer_id.map_or("none".to_string(), |id| format!("{:#x}", id)),
                ).unwrap();
                drop(ca);

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

    #[test]
    fn resampler_downsample_48000_to_44100() {
        // 48000 → 44100: ratio ~0.919, so 48000 in → 44100 out
        let out = resample(48000, 44100, 48000);
        assert_eq!(out, 44100, "48000→44100: expected 44100 frames, got {}", out);
    }
}
