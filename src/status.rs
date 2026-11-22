//! The machine's status: one structure of atomics that every CPU and device
//! reports into, and the status bar drawn from it.
//!
//! [`MachineStatus`] is built once per machine and handed, as an `Arc`, to
//! everything that reports or reads: the CPUs (their cycle counters and
//! clock-tick counts), the timers (the tick rate the guest programmed), the
//! network and storage controllers (activity bits), the front-panel LEDs,
//! and the graphics boards' display heads. Nothing in it takes a lock on a
//! reporting path: reporters write atomics, readers sample them.
//!
//! The status bar ([`StatusBar`]) is drawn from it by the display layer
//! (`crate::display`), never by a graphics board, and handed to the
//! renderers as finished pixels.
//!
//! The monitor reads the same figures: `speed` (each CPU's MIPS) and
//! `ticks` (each CPU's clock-tick rate against the rate the guest
//! programmed). `watch speed` keeps them on one line.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::cpu::jit_feedback::{JitFeedback, JIT_FEEDBACK};
use crate::cpu::mips_core::CyclesPtr;

/// CPUs a machine can report.
pub const MAX_CPUS: usize = crate::cpu::smp::MAX_CPUS;

/// The heartbeat word's bits: activity lights, which clear once shown, and
/// the front-panel LEDs, which stay as the guest set them.
pub mod hb {
    pub const ENET_TX: u64 = 1 << 0;
    pub const ENET_RX: u64 = 1 << 1;
    /// Front-panel red LED (persistent).
    pub const LED_RED: u64 = 1 << 2;
    /// Front-panel green LED (persistent).
    pub const LED_GREEN: u64 = 1 << 3;
    /// Bits `SCSI_BASE..SCSI_BASE + 16`: SCSI target 0..15 activity, every
    /// controller folded together. See [`super::scsi_blink`].
    pub const SCSI_BASE: u32 = 16;
    pub const SCSI_MASK: u64 = 0xFFFF << SCSI_BASE;
    /// Bits that stay lit between frames.
    pub const PERSISTENT: u64 = LED_RED | LED_GREEN;
}

/// Light the activity light of a storage target. Every controller of the
/// machine shares the lights: the controller is not part of the light, and
/// a target is folded to 0..15 (a Fibre Channel loop ID too). A machine
/// whose controllers are all narrow shows 0..7, folding 8..15 onto them
/// (see [`MachineStatus::set_scsi_wide`]).
#[inline]
pub fn scsi_blink(heartbeat: &AtomicU64, target: u32) {
    heartbeat.fetch_or(1u64 << (hb::SCSI_BASE + (target & 15)), Ordering::Relaxed);
}

/// What one CPU reports.
pub struct CpuStatus {
    /// The CPU's cycle counter and its parked-cycles word (`CyclesPtr`),
    /// null until the CPU exists: boards are built before CPUs.
    cycles: AtomicPtr<u64>,
    parked: AtomicPtr<u64>,
    /// The CPU's clock-tick interrupts, counted by whatever delivers them
    /// (the 8254 on Indy and Indigo2, HEART on the Octane, CP0 Compare
    /// elsewhere).
    ticks: OnceLock<Arc<AtomicU64>>,
    /// The tick rate the guest programmed, in millihertz; 0 until a timer
    /// publishes one. Timers write it through [`tick_rate`](Self::tick_rate).
    tick_rate: Arc<AtomicU64>,
}

impl CpuStatus {
    fn new() -> Self {
        Self {
            cycles: AtomicPtr::new(std::ptr::null_mut()),
            parked: AtomicPtr::new(std::ptr::null_mut()),
            ticks: OnceLock::new(),
            tick_rate: Arc::new(AtomicU64::new(0)),
        }
    }

    fn set_cycles(&self, c: CyclesPtr) {
        self.parked.store(c.raw_parked() as *mut u64, Ordering::Release);
        self.cycles.store(c.raw() as *mut u64, Ordering::Release);
    }

    /// The CPU's cycle counter (reads 0 until the CPU exists).
    pub fn cycles(&self) -> CyclesPtr {
        CyclesPtr::with_parked(self.cycles.load(Ordering::Acquire), self.parked.load(Ordering::Acquire))
    }

    fn ticks(&self) -> u64 {
        self.ticks.get().map_or(0, |t| t.load(Ordering::Relaxed))
    }

    /// Where a timer publishes the tick rate the guest programmed for this
    /// CPU, in millihertz (`hz_to_mhz`). 0 means not known.
    pub fn tick_rate(&self) -> Arc<AtomicU64> {
        self.tick_rate.clone()
    }
}

/// A tick rate in hertz as the millihertz a [`CpuStatus::tick_rate`] holds.
pub fn hz_to_mhz(hz: f64) -> u64 {
    if hz.is_finite() && hz > 0.0 { (hz * 1000.0).round() as u64 } else { 0 }
}

/// Everything the status bar and the `speed`/`ticks` commands show, for one
/// machine. See the module documentation.
pub struct MachineStatus {
    /// LEDs and activity bits ([`hb`]). Devices hold a clone of this `Arc`.
    heartbeat: Arc<AtomicU64>,
    /// Show 16 SCSI lights (some controller is wide) instead of 8.
    scsi_wide: AtomicBool,
    cpus: [CpuStatus; MAX_CPUS],
    /// CPUs the machine has wired (at least 1).
    ncpus: AtomicUsize,
    /// CPU Count rate, for the developer bar.
    count_hz: OnceLock<Arc<AtomicU64>>,
    /// Fetch counters the CPU bumps, read and cleared per frame by the
    /// developer bar. Zero on machines whose CPU does not feed them.
    decoded: Arc<AtomicU64>,
    l1i_hits: Arc<AtomicU64>,
    l1i_fetches: Arc<AtomicU64>,
    uncached: Arc<AtomicU64>,
    /// The JIT's arena and queue gauges. The JIT's compile pool is shared
    /// by the machine's CPUs and reports through a static.
    jit: &'static JitFeedback,
    /// Rates measured over a window, shared by the bar and the monitor.
    sampler: Mutex<Sampler>,
    /// The machine's display refresh thread. It is here because the status
    /// is the one thing every graphics board of a machine is given.
    refresh: Arc<crate::display::Refresh>,
}

impl MachineStatus {
    /// Heartbeat bits that stay lit between frames (the front-panel LEDs).
    pub const PERSISTENT: u64 = hb::PERSISTENT;

    /// A status reporting through `heartbeat`, whose first CPU's clock
    /// ticks are counted in `ticks`.
    pub fn new(heartbeat: Arc<AtomicU64>, ticks: Arc<AtomicU64>) -> Arc<Self> {
        let s = Self::build(heartbeat);
        let _ = s.cpus[0].ticks.set(ticks);
        s
    }

    /// A status whose first CPU's tick counter is wired later, with
    /// [`set_cpu_ticks`](Self::set_cpu_ticks): for boards built before
    /// their CPUs.
    pub fn without_ticks(heartbeat: Arc<AtomicU64>) -> Arc<Self> {
        Self::build(heartbeat)
    }

    fn build(heartbeat: Arc<AtomicU64>) -> Arc<Self> {
        let s = Self {
            heartbeat,
            scsi_wide: AtomicBool::new(false),
            cpus: std::array::from_fn(|_| CpuStatus::new()),
            ncpus: AtomicUsize::new(1),
            count_hz: OnceLock::new(),
            decoded: Arc::new(AtomicU64::new(0)),
            l1i_hits: Arc::new(AtomicU64::new(0)),
            l1i_fetches: Arc::new(AtomicU64::new(0)),
            uncached: Arc::new(AtomicU64::new(0)),
            jit: &JIT_FEEDBACK,
            sampler: Mutex::new(Sampler::default()),
            refresh: crate::display::Refresh::new(),
        };
        Arc::new(s)
    }

    /// A status with its own counters and no CPU, for tests and boards run
    /// outside a machine.
    pub fn detached() -> Arc<Self> {
        Self::new(Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)))
    }

    /// The display refresh thread the machine's boards share.
    pub fn refresh(&self) -> &Arc<crate::display::Refresh> {
        &self.refresh
    }

    /// The heartbeat word devices set bits in.
    pub fn heartbeat(&self) -> Arc<AtomicU64> {
        self.heartbeat.clone()
    }

    /// Show SCSI targets 0..15 rather than 0..7: the machine has a wide
    /// controller.
    pub fn set_scsi_wide(&self, wide: bool) {
        self.scsi_wide.store(wide, Ordering::Relaxed);
    }

    /// The CPU's pre-decoded-instruction, L1-I hit, L1-I fetch and uncached
    /// fetch counters, in that order: the Arcs the CPU is given to bump.
    pub fn fetch_counters(&self) -> [Arc<AtomicU64>; 4] {
        [self.decoded.clone(), self.l1i_hits.clone(), self.l1i_fetches.clone(), self.uncached.clone()]
    }

    /// Wire the first CPU's cycle counter.
    pub fn set_cpu_cycles(&self, ptr: CyclesPtr) {
        self.cpus[0].set_cycles(ptr);
    }

    /// Wire the first CPU's clock-tick counter, unless the status was made
    /// with one.
    pub fn set_cpu_ticks(&self, ticks: Arc<AtomicU64>) {
        let _ = self.cpus[0].ticks.set(ticks);
    }

    /// The first CPU's cycle counter.
    pub fn cycles(&self) -> CyclesPtr {
        self.cpus[0].cycles()
    }

    pub fn set_count_hz(&self, hz: Arc<AtomicU64>) {
        let _ = self.count_hz.set(hz);
    }

    /// Wire one more CPU; returns its index.
    pub fn add_cpu_stats(&self, cycles: CyclesPtr, ticks: Arc<AtomicU64>) -> usize {
        let n = self.ncpus.fetch_add(1, Ordering::AcqRel);
        if n >= MAX_CPUS {
            self.ncpus.store(MAX_CPUS, Ordering::Release);
            return MAX_CPUS - 1;
        }
        self.cpus[n].set_cycles(cycles);
        let _ = self.cpus[n].ticks.set(ticks);
        n
    }

    /// CPUs wired.
    pub fn ncpus(&self) -> usize {
        self.ncpus.load(Ordering::Acquire).clamp(1, MAX_CPUS)
    }

    /// One CPU's slot (panics past [`MAX_CPUS`]).
    pub fn cpu(&self, n: usize) -> &CpuStatus {
        &self.cpus[n]
    }

    fn sample(&self, now: Instant) -> Sample {
        let n = self.ncpus();
        Sample {
            at: now,
            n,
            cpus: std::array::from_fn(|i| {
                if i < n {
                    let c = &self.cpus[i];
                    (c.cycles().retired(), c.ticks(), c.tick_rate.load(Ordering::Relaxed))
                } else {
                    (0, 0, 0)
                }
            }),
        }
    }

    /// Each CPU's rates over the last window (at least [`RATE_WINDOW`]).
    /// Cheap: a caller faster than the window gets the last figures again.
    pub fn rates(&self) -> Rates {
        let now = Instant::now();
        let s = self.sample(now);
        self.sampler.lock().advance(s)
    }

    /// As [`rates`](Self::rates), but measured now: if the last window is
    /// stale (nobody asked for a while, as on a machine with no display),
    /// sample, wait one window and sample again. For the monitor.
    pub fn fresh_rates(&self) -> Rates {
        let stale = self.sampler.lock().last.as_ref()
            .map_or(true, |l| l.at.elapsed() > RATE_WINDOW * 4);
        if stale {
            self.rates();
            std::thread::sleep(RATE_WINDOW);
        }
        self.rates()
    }

    /// One frame's bar inputs. Clears the heartbeat's activity bits and the
    /// fetch counters, so call it once per presented frame.
    pub fn snapshot(&self, gfifo_pending: usize) -> BarStats {
        #[cfg(feature = "developer")]
        let (decoded_delta, l1i_hits, l1i_fetches, uncached, count_hz) = (
            self.decoded.swap(0, Ordering::Relaxed),
            self.l1i_hits.swap(0, Ordering::Relaxed),
            self.l1i_fetches.swap(0, Ordering::Relaxed),
            self.uncached.swap(0, Ordering::Relaxed),
            self.count_hz.get().map_or(crate::cpu::mips_core::DEFAULT_COUNT_HZ, |h| h.load(Ordering::Relaxed)),
        );
        #[cfg(not(feature = "developer"))]
        let (decoded_delta, l1i_hits, l1i_fetches, uncached, count_hz) = (0, 0, 0, 0, 0);
        BarStats {
            rates: self.rates(),
            hb: self.heartbeat.fetch_and(Self::PERSISTENT, Ordering::Relaxed),
            scsi_wide: self.scsi_wide.load(Ordering::Relaxed),
            jit_arena_fill: self.jit.arena_fill.load(Ordering::Relaxed),
            jit_queue_fill: self.jit.queue_fill.load(Ordering::Relaxed),
            jit_flushes: self.jit.flush_events.load(Ordering::Relaxed),
            decoded_delta,
            l1i_hits,
            l1i_fetches,
            uncached,
            count_hz,
            gfifo_pending,
        }
    }

    /// What `speed` prints: each CPU's MIPS, on one line.
    pub fn speed_line(&self) -> String {
        let r = self.fresh_rates();
        let mut s = format!("MIPS {:7.1} total", r.total_mips());
        for (i, c) in r.cpus().iter().enumerate() {
            s.push_str(&format!("  {}:{:6.1}", cpu_label(i), c.mips));
        }
        s
    }

    /// What `ticks` prints: each CPU's clock-tick rate, the rate the guest
    /// programmed, and the light the bar shows for it.
    pub fn ticks_line(&self) -> String {
        let r = self.fresh_rates();
        let mut parts = Vec::new();
        for (i, c) in r.cpus().iter().enumerate() {
            let want = c.expected_hz.map_or("?".to_string(), |e| format!("{e:.0}"));
            parts.push(format!("{}:{:.0}/{} Hz {}", cpu_label(i), c.tick_hz, want, c.tick_state().word()));
        }
        parts.join("  ")
    }
}

impl crate::traits::Device for MachineStatus {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("speed".into(), "speed: each CPU's MIPS over the last half second (`watch speed` keeps it on one line)".into()),
            ("ticks".into(), "ticks: each CPU's clock-tick rate / the rate the guest programmed, ok|off|stopped (`watch ticks`)".into()),
        ]
    }

    fn execute_command(&self, cmd: &str, _args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        let line = match cmd {
            "speed" => self.speed_line(),
            "ticks" => self.ticks_line(),
            _ => return Err(format!("unknown command {cmd}")),
        };
        writeln!(w, "{line}").map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())
    }
}

/// A CPU's label on the bar and in the monitor: 0..9, then A..Z.
pub fn cpu_label(n: usize) -> char {
    match n {
        0..=9 => (b'0' + n as u8) as char,
        10..=35 => (b'A' + (n - 10) as u8) as char,
        _ => '?',
    }
}

/// The shortest window rates are measured over. Long enough that a 100 Hz
/// clock tick does not flicker by one tick in ten.
pub const RATE_WINDOW: Duration = Duration::from_millis(500);

#[derive(Clone)]
struct Sample {
    at: Instant,
    n: usize,
    /// Per CPU: retired cycles, ticks, programmed tick rate (mHz).
    cpus: [(u64, u64, u64); MAX_CPUS],
}

#[derive(Default)]
struct Sampler {
    last: Option<Sample>,
    rates: Rates,
}

impl Sampler {
    fn advance(&mut self, s: Sample) -> Rates {
        let Some(prev) = &self.last else {
            self.last = Some(s);
            return self.rates.clone();
        };
        let dt = s.at.duration_since(prev.at);
        if dt < RATE_WINDOW {
            return self.rates.clone();
        }
        let secs = dt.as_secs_f64();
        let mut cpus = [CpuRate::default(); MAX_CPUS];
        for i in 0..s.n {
            let (pc, pt, _) = prev.cpus[i];
            let (c, t, want) = s.cpus[i];
            cpus[i] = CpuRate {
                mips: (cycles_moved(pc, c) as f64 / secs / 1e6 * 10.0).round() / 10.0,
                tick_hz: t.wrapping_sub(pt) as f64 / secs,
                expected_hz: (want != 0).then(|| want as f64 / 1000.0),
            };
        }
        self.rates = Rates { n: s.n, cpus };
        self.last = Some(s);
        self.rates.clone()
    }
}

/// How far a cycle counter moved between two samples. A counter that went
/// backwards was set to 0 by a CPU reset inside the window, so it moved by
/// its new value; a wrapping difference would show a reset as about 18
/// trillion MIPS.
pub fn cycles_moved(prev: u64, cur: u64) -> u64 {
    if cur >= prev { cur - prev } else { cur }
}

/// One CPU's rates over a window.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct CpuRate {
    pub mips: f64,
    pub tick_hz: f64,
    /// What the guest programmed, if a timer published it.
    pub expected_hz: Option<f64>,
}

/// How a CPU's clock tick compares with what the guest asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TickState {
    /// Within [`TICK_TOLERANCE`] of the programmed rate.
    Ok,
    /// Ticking, but off the programmed rate.
    Off,
    /// No ticks in the window.
    Stopped,
    /// Ticking; no timer published a rate to compare with.
    Unknown,
}

/// How far from the programmed rate a tick may be and still read as on
/// rate.
pub const TICK_TOLERANCE: f64 = 0.10;

impl TickState {
    pub fn word(self) -> &'static str {
        match self {
            TickState::Ok => "ok",
            TickState::Off => "off",
            TickState::Stopped => "stopped",
            TickState::Unknown => "ticking",
        }
    }
}

impl CpuRate {
    pub fn tick_state(&self) -> TickState {
        if self.tick_hz <= 0.0 {
            return TickState::Stopped;
        }
        match self.expected_hz {
            None => TickState::Unknown,
            Some(e) if (self.tick_hz - e).abs() <= e * TICK_TOLERANCE => TickState::Ok,
            Some(_) => TickState::Off,
        }
    }
}

/// Every CPU's rates over the last window.
#[derive(Clone, Default, Debug)]
pub struct Rates {
    n: usize,
    cpus: [CpuRate; MAX_CPUS],
}

impl Rates {
    pub fn cpus(&self) -> &[CpuRate] {
        &self.cpus[..self.n.max(1)]
    }

    pub fn fastest_mips(&self) -> f64 {
        self.cpus().iter().map(|c| c.mips).fold(0.0, f64::max)
    }

    pub fn total_mips(&self) -> f64 {
        self.cpus().iter().map(|c| c.mips).sum()
    }
}

/// One frame's status bar inputs.
#[derive(Clone, Default)]
pub struct BarStats {
    pub rates: Rates,
    /// Heartbeat bits ([`hb`]) set since the last frame.
    pub hb: u64,
    pub scsi_wide: bool,
    pub jit_arena_fill: u8,
    pub jit_queue_fill: u8,
    pub jit_flushes: u32,
    pub decoded_delta: u64,
    pub l1i_hits: u64,
    pub l1i_fetches: u64,
    pub uncached: u64,
    pub count_hz: u64,
    pub gfifo_pending: usize,
}

// ── The bar ─────────────────────────────────────────────────────────────────

/// Height of the status bar in pixels (one VGA glyph row).
pub const STATUS_BAR_HEIGHT: usize = 16;
/// Row stride of a bar strip, in pixels: the widest display.
pub const STRIP_STRIDE: usize = 2048;

const FADE_FRAMES: u8 = 15;

// 0xAABBGGRR.
const BAR_BG: u32 = 0xFF202020;
/// A gauge's extent, drawn behind it so its reach shows when it is empty.
const GAUGE_BG: u32 = 0xFF303030;
const BAR_FG: u32 = 0xFF00CC00;
const BAR_ACTIVE: u32 = 0xFF00FFAA;
const BAR_DIM: u32 = 0xFF004400;
const SPEED: u32 = 0xFF00B000;
const LED_RED_ON: u32 = 0xFF2020FF;
const LED_RED_OFF: u32 = 0xFF000030;
const LED_GREEN_ON: u32 = 0xFF20FF20;
const LED_GREEN_OFF: u32 = 0xFF003000;
const TICK_OK: u32 = 0xFF20E020;
const TICK_OFF: u32 = 0xFF00A0FF;
const TICK_STOPPED: u32 = 0xFF2020F0;
const TICK_UNKNOWN: u32 = 0xFF208020;
/// The JIT gauge right after a flush: an event, not a level.
const JIT_FLUSH_FLASH: u32 = 0xFFFF20C0;

/// Width of a CPU's speed cell and of its tick light.
const CELL_W: usize = 8;
const TICK_W: usize = 4;

/// Green at 0 to red at 255 (`frac` is a 0..=255 fraction).
fn lerp_green_red(frac: u8) -> u32 {
    let f = frac as u32;
    let r = f * 0xE0 / 255;
    let g = 0xE0 - r;
    0xFF000000 | (g << 8) | r
}

/// `c` brightened by `by` per channel: the faint CPU number over a cell.
fn lighten(c: u32, by: u32) -> u32 {
    let ch = |s: u32| (((c >> s) & 0xFF) + by).min(0xFF) << s;
    0xFF000000 | ch(16) | ch(8) | ch(0)
}

/// The bar's state between frames and its pixels.
pub struct StatusBar {
    font: &'static [u8],
    buf: Vec<u32>,
    enet_tx_fade: u8,
    enet_rx_fade: u8,
    scsi_fade: [u8; 16],
    led_red: bool,
    led_green: bool,
    jit_flush_fade: u8,
    jit_flush_seen: u32,
}

impl Default for StatusBar {
    fn default() -> Self { Self::new() }
}

impl StatusBar {
    pub fn new() -> Self {
        Self {
            font: &crate::vga_font::SCREEN_8X16,
            buf: vec![BAR_BG; STRIP_STRIDE * STATUS_BAR_HEIGHT],
            enet_tx_fade: 0,
            enet_rx_fade: 0,
            scsi_fade: [0; 16],
            led_red: false,
            led_green: false,
            jit_flush_fade: 0,
            jit_flush_seen: 0,
        }
    }

    /// Take a frame from `status` and draw the bar `width` pixels wide.
    /// Returns the strip (stride [`STRIP_STRIDE`], [`STATUS_BAR_HEIGHT`]
    /// rows, 0xAABBGGRR).
    pub fn render(&mut self, status: &MachineStatus, gfifo_pending: usize, width: usize) -> &[u32] {
        let stats = status.snapshot(gfifo_pending);
        self.render_stats(&stats, width)
    }

    /// The pixels last drawn.
    pub fn pixels(&self) -> &[u32] {
        &self.buf
    }

    fn update(&mut self, s: &BarStats) {
        if s.hb & hb::ENET_TX != 0 { self.enet_tx_fade = FADE_FRAMES; }
        if s.hb & hb::ENET_RX != 0 { self.enet_rx_fade = FADE_FRAMES; }
        let scsi = (s.hb >> hb::SCSI_BASE) as u16;
        for i in 0..16 {
            if scsi & (1 << i) != 0 {
                self.scsi_fade[i] = FADE_FRAMES;
            }
        }
        self.led_red = s.hb & hb::LED_RED != 0;
        self.led_green = s.hb & hb::LED_GREEN != 0;
        if s.jit_flushes != self.jit_flush_seen {
            self.jit_flush_seen = s.jit_flushes;
            self.jit_flush_fade = FADE_FRAMES;
        }
    }

    fn fade(&mut self) {
        let dec = |f: &mut u8| *f = f.saturating_sub(1);
        dec(&mut self.enet_tx_fade);
        dec(&mut self.enet_rx_fade);
        self.scsi_fade.iter_mut().for_each(dec);
        dec(&mut self.jit_flush_fade);
    }

    /// Draw from an already-taken frame.
    pub fn render_stats(&mut self, s: &BarStats, width: usize) -> &[u32] {
        let width = width.min(STRIP_STRIDE);
        self.update(s);
        for row in 0..STATUS_BAR_HEIGHT {
            let base = row * STRIP_STRIDE;
            self.buf[base..base + width].fill(BAR_BG);
        }
        let mut p = Pen { buf: &mut self.buf, font: self.font, width };

        // The fastest CPU's MIPS, then a cell per CPU: its speed against the
        // fastest, its number, and its clock-tick light.
        let fastest = s.rates.fastest_mips();
        let mut x = p.text(&format!(" {:6.1} ", fastest.min(99999.9)), 0, BAR_FG);
        for (i, c) in s.rates.cpus().iter().enumerate() {
            x = p.cpu_cell(x, i, c.mips, fastest);
            x = p.tick_light(x + 1, c.tick_state()) + 3;
        }

        #[cfg(feature = "developer")]
        {
            let total = s.l1i_fetches + s.uncached;
            let pct = |a: u64, b: u64| if b > 0 { a as f64 / b as f64 * 100.0 } else { 0.0 };
            x = p.text(&format!(" D:{:3.0}% I$:{:3.0}% UC:{:3.0}% cnt:{:5.1}MHz g{:04X}",
                pct(s.decoded_delta, total), pct(s.l1i_hits, s.l1i_fetches), pct(s.uncached, total),
                s.count_hz as f64 / 1e6, s.gfifo_pending), x, BAR_FG);
        }

        let on = |fade: u8| if fade > 0 { BAR_ACTIVE } else { BAR_DIM };
        x = p.text(" NET:", x, BAR_FG);
        x = p.text("TX", x, on(self.enet_tx_fade));
        x = p.text(" ", x, BAR_FG);
        x = p.text("RX", x, on(self.enet_rx_fade));
        x = p.text(" SCSI:", x, BAR_FG);
        let lights = if s.scsi_wide { 16 } else { 8 };
        for i in 0..lights {
            // Narrow: 8..15 fold onto 0..7.
            let fade = if s.scsi_wide { self.scsi_fade[i] } else { self.scsi_fade[i].max(self.scsi_fade[i + 8]) };
            x = p.text(&format!("{:X}", i), x, on(fade));
        }
        x = p.text(" LED:", x, BAR_FG);
        x = p.square(x + 2, if self.led_red { LED_RED_ON } else { LED_RED_OFF });
        x = p.square(x + 4, if self.led_green { LED_GREEN_ON } else { LED_GREEN_OFF });
        #[cfg(feature = "jitv2")]
        {
            x = p.text(" JIT:", x, BAR_FG);
            x = p.jit_gauge(x + 2, s.jit_arena_fill, s.jit_queue_fill, self.jit_flush_fade > 0);
        }
        x = p.text(" G:", x, BAR_FG);
        let _ = p.gfifo_gauge(x + 2, s.gfifo_pending);

        self.fade();
        &self.buf
    }
}

/// Drawing into a bar strip.
struct Pen<'a> {
    buf: &'a mut [u32],
    font: &'static [u8],
    width: usize,
}

impl Pen<'_> {
    fn put(&mut self, x: usize, y: usize, c: u32) {
        if x < self.width && y < STATUS_BAR_HEIGHT {
            self.buf[y * STRIP_STRIDE + x] = c;
        }
    }

    fn get(&self, x: usize, y: usize) -> u32 {
        if x < self.width && y < STATUS_BAR_HEIGHT { self.buf[y * STRIP_STRIDE + x] } else { BAR_BG }
    }

    fn rect(&mut self, x: usize, y: usize, w: usize, h: usize, c: u32) {
        for row in y..y + h {
            for col in x..x + w {
                self.put(col, row, c);
            }
        }
    }

    fn glyph_bits(&self, ch: char, row: usize) -> u8 {
        self.font.get((ch as usize & 0xFF) * 16 + row).copied().unwrap_or(0)
    }

    fn text(&mut self, text: &str, x: usize, c: u32) -> usize {
        let mut cx = x;
        for ch in text.chars() {
            if cx + 8 > self.width {
                break;
            }
            for row in 0..16 {
                let bits = self.glyph_bits(ch, row);
                for col in 0..8 {
                    let lit = (bits >> (7 - col)) & 1 != 0;
                    self.put(cx + col, row, if lit { c } else { BAR_BG });
                }
            }
            cx += 8;
        }
        cx
    }

    /// An 8x16 cell: the CPU's speed as a bar from the bottom, against the
    /// fastest CPU, with the CPU's number faintly over it.
    fn cpu_cell(&mut self, x: usize, n: usize, mips: f64, fastest: f64) -> usize {
        self.rect(x, 0, CELL_W, STATUS_BAR_HEIGHT, GAUGE_BG);
        let h = if fastest > 0.0 {
            ((mips / fastest) * STATUS_BAR_HEIGHT as f64).round().clamp(0.0, STATUS_BAR_HEIGHT as f64) as usize
        } else {
            0
        };
        // Anything running shows at least a line.
        let h = if mips > 0.0 { h.max(1) } else { h };
        self.rect(x, STATUS_BAR_HEIGHT - h, CELL_W, h, SPEED);
        let label = cpu_label(n);
        for row in 0..16 {
            let bits = self.glyph_bits(label, row);
            for col in 0..8 {
                if (bits >> (7 - col)) & 1 != 0 {
                    let under = self.get(x + col, row);
                    self.put(x + col, row, lighten(under, 0x30));
                }
            }
        }
        x + CELL_W
    }

    fn tick_light(&mut self, x: usize, state: TickState) -> usize {
        let c = match state {
            TickState::Ok => TICK_OK,
            TickState::Off => TICK_OFF,
            TickState::Stopped => TICK_STOPPED,
            TickState::Unknown => TICK_UNKNOWN,
        };
        self.rect(x, 2, TICK_W, STATUS_BAR_HEIGHT - 4, c);
        x + TICK_W
    }

    fn square(&mut self, x: usize, c: u32) -> usize {
        const SQ: usize = 10;
        self.rect(x, (STATUS_BAR_HEIGHT - SQ) / 2, SQ, SQ, c);
        x + SQ + 4
    }

    /// A horizontal gauge filled left to right by `fill/255`, over its
    /// extent.
    fn fill_bar(&mut self, x: usize, y: usize, w: usize, h: usize, fill: u8) {
        self.rect(x, y, w, h, GAUGE_BG);
        let filled = w * fill as usize / 255;
        self.rect(x, y, filled, h, lerp_green_red(fill));
    }

    /// 32x16: the code arena's fill over the compile queue's; flashes after
    /// a flush.
    #[cfg_attr(not(feature = "jitv2"), allow(dead_code))]
    fn jit_gauge(&mut self, x: usize, arena: u8, queue: u8, flushed: bool) -> usize {
        const W: usize = 32;
        const H: usize = 7;
        if flushed {
            self.rect(x, 0, W, STATUS_BAR_HEIGHT, JIT_FLUSH_FLASH);
        } else {
            self.rect(x, 0, W, STATUS_BAR_HEIGHT, BAR_BG);
            self.fill_bar(x, 0, W, H, arena);
            self.fill_bar(x, H + 2, W, H, queue);
        }
        x + W + 4
    }

    /// 16x16 GFIFO occupancy, log scale (each doubling the same distance:
    /// the guest only feels back-pressure near the top, and normal traffic
    /// lives in the low hundreds of a 64K queue), filled in columns with
    /// the boundary column partly filled from the bottom.
    fn gfifo_gauge(&mut self, x: usize, pending: usize) -> usize {
        const R: usize = 16;
        let frac = if pending == 0 {
            0u8
        } else {
            let lg = (pending as f64 + 1.0).log2();
            let full = (crate::dev::ng1::rex3::GFIFO_DEPTH as f64).log2();
            ((lg / full) * 255.0).clamp(0.0, 255.0) as u8
        };
        let color = lerp_green_red(frac);
        let units = frac as usize * R * R / 255;
        let (full_cols, partial) = (units / R, units % R);
        self.rect(x, 0, R, R, GAUGE_BG);
        for col in 0..R {
            let rows = if col < full_cols { R } else if col == full_cols { partial } else { 0 };
            self.rect(x + col, R - rows, 1, rows, color);
        }
        x + R + 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr(cycles: &AtomicU64, parked: &AtomicU64) -> CyclesPtr {
        CyclesPtr::with_parked(cycles.as_ptr(), parked.as_ptr())
    }

    /// Rates count retired cycles: a parked CPU adds nothing.
    #[test]
    fn rates_leave_out_parked_cycles() {
        let s = MachineStatus::detached();
        let (c0, p0) = (AtomicU64::new(0), AtomicU64::new(0));
        let (c1, p1) = (AtomicU64::new(0), AtomicU64::new(0));
        s.set_cpu_cycles(ptr(&c0, &p0));
        assert_eq!(s.add_cpu_stats(ptr(&c1, &p1), Arc::new(AtomicU64::new(0))), 1);
        let t0 = Instant::now();
        let mut sampler = Sampler::default();
        sampler.advance(s.sample(t0));
        c0.store(2_000_000, Ordering::Relaxed);
        p0.store(500_000, Ordering::Relaxed);
        c1.store(900_000, Ordering::Relaxed);
        let r = sampler.advance(s.sample(t0 + Duration::from_secs(1)));
        assert_eq!(r.cpus().len(), 2);
        assert_eq!(r.cpus()[0].mips, 1.5);
        assert_eq!(r.cpus()[1].mips, 0.9);
        assert_eq!(r.fastest_mips(), 1.5);
    }

    /// A CPU reset sets the cycle counter back to 0 mid-window: the rate is
    /// what ran since, not a wrapped difference.
    #[test]
    fn a_reset_in_the_window_counts_from_zero() {
        assert_eq!(cycles_moved(900, 40), 40);
        assert_eq!(cycles_moved(40, 900), 860);
    }

    /// A board built before its CPUs takes CPU 0's tick counter later; a
    /// status made with one keeps it.
    #[test]
    fn first_cpu_ticks_wired_later() {
        let late = MachineStatus::without_ticks(Arc::default());
        assert_eq!(late.cpu(0).ticks(), 0);
        let t = Arc::new(AtomicU64::new(42));
        late.set_cpu_ticks(t.clone());
        assert_eq!(late.cpu(0).ticks(), 42);
        let made = MachineStatus::new(Arc::default(), Arc::new(AtomicU64::new(7)));
        made.set_cpu_ticks(t);
        assert_eq!(made.cpu(0).ticks(), 7);
    }

    /// Activity bits clear once shown; the LEDs stay lit.
    #[test]
    fn snapshot_clears_activity_but_keeps_the_leds() {
        let hb = Arc::new(AtomicU64::new(u64::MAX));
        let src = MachineStatus::new(hb.clone(), Arc::new(AtomicU64::new(0)));
        assert_eq!(src.snapshot(0).hb, u64::MAX);
        assert_eq!(src.snapshot(0).hb, MachineStatus::PERSISTENT);
    }

    #[test]
    fn tick_states() {
        let r = |tick_hz, expected_hz| CpuRate { mips: 1.0, tick_hz, expected_hz };
        assert_eq!(r(0.0, Some(100.0)).tick_state(), TickState::Stopped);
        assert_eq!(r(98.0, Some(100.0)).tick_state(), TickState::Ok);
        assert_eq!(r(80.0, Some(100.0)).tick_state(), TickState::Off);
        assert_eq!(r(80.0, None).tick_state(), TickState::Unknown);
    }

    #[test]
    fn cpu_labels_run_on_into_letters() {
        assert_eq!(cpu_label(0), '0');
        assert_eq!(cpu_label(9), '9');
        assert_eq!(cpu_label(10), 'A');
        assert_eq!(cpu_label(35), 'Z');
    }

    /// Every controller's target lands on one of 16 lights; a narrow
    /// machine folds 8..15 onto 0..7.
    #[test]
    fn scsi_targets_fold() {
        let hbw = AtomicU64::new(0);
        scsi_blink(&hbw, 9);
        scsi_blink(&hbw, 0x7D); // a Fibre Channel loop ID
        assert_eq!(hbw.load(Ordering::Relaxed) >> hb::SCSI_BASE, (1 << 9) | (1 << 13));
        let mut bar = StatusBar::new();
        let stats = BarStats { hb: hbw.load(Ordering::Relaxed), ..Default::default() };
        bar.render_stats(&stats, 1280);
        assert!(bar.scsi_fade[9] > 0 && bar.scsi_fade[13] > 0);
    }

    /// The speed cells scale to the fastest CPU, which fills its cell.
    #[test]
    fn the_fastest_cpu_fills_its_cell() {
        let mut rates = Rates { n: 2, ..Default::default() };
        rates.cpus[0].mips = 100.0;
        rates.cpus[1].mips = 50.0;
        let mut bar = StatusBar::new();
        let px = bar.render_stats(&BarStats { rates, ..Default::default() }, 1280).to_vec();
        // " 1234.5 " is 8 characters; CPU 0's cell follows.
        let cell0 = 8 * 8;
        let cell1 = cell0 + CELL_W + 1 + TICK_W + 3;
        let column = |x: usize| (0..STATUS_BAR_HEIGHT).filter(|&y| {
            let c = px[y * STRIP_STRIDE + x];
            c == SPEED || c == lighten(SPEED, 0x30)
        }).count();
        // The column at the cell's left edge carries no digit pixels.
        assert_eq!(column(cell0), STATUS_BAR_HEIGHT);
        assert_eq!(column(cell1), STATUS_BAR_HEIGHT / 2);
    }
}
