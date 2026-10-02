//! Video capture source for the VINO Video-In ASIC.
//!
//! A `VideoSource` produces one video field at a time at the broadcast field
//! rate (NTSC 60 Hz, PAL 50 Hz) in packed YUV 4:2:2 (UYVY) at the signal's
//! native field resolution.  VINO applies clipping, decimation, and pixel
//! format conversion downstream before DMA'ing into system memory.
//!
//! Phase 1 ships `TestPatternSource` only — a self-paced SMPTE-style colour
//! bar generator that needs no host capture hardware.  Real camera sources
//! (macOS AVFoundation, etc.) hang off the same trait in later phases.

use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::Mutex;

/// Broadcast video standard the source emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoStandard {
    /// 525-line / 60-field, two 640×243 fields per 640×486 frame.
    Ntsc,
    /// 625-line / 50-field, two 768×288 fields per 768×576 frame.
    Pal,
}

impl VideoStandard {
    /// Native pixel dimensions of one field.
    pub fn field_size(self) -> (u32, u32) {
        match self {
            VideoStandard::Ntsc => (640, 243),
            VideoStandard::Pal  => (768, 288),
        }
    }

    /// Field period (NTSC ≈ 16.683 ms, PAL = 20.000 ms).
    pub fn field_period(self) -> Duration {
        match self {
            VideoStandard::Ntsc => Duration::from_nanos(16_683_333),
            VideoStandard::Pal  => Duration::from_nanos(20_000_000),
        }
    }
}

/// Which half of an interlaced frame this field belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldParity { Even, Odd }

/// One captured video field, packed as UYVY 4:2:2.
/// `pixels.len() == width as usize * height as usize * 2`.
pub struct Field {
    pub parity: FieldParity,
    pub width:  u32,
    pub height: u32,
    pub pixels: Arc<[u8]>,
}

/// Video source trait.  Implementations are responsible for pacing their
/// own delivery — `next_field` blocks until one field period has elapsed.
pub trait VideoSource: Send + Sync {
    fn standard(&self) -> VideoStandard;
    fn next_field(&self) -> Field;
    /// One-line status string for the monitor `vino status` command.
    fn status(&self) -> String { "no status available".to_string() }
}

// ─── CDMC register → pixel pipeline (Phase 3) ────────────────────────────────

/// Applies IndyCam CDMC gain/balance/saturation to UYVY fields after capture.
pub struct CdmcAdjustedSource {
    inner: Arc<dyn VideoSource>,
    vino:  crate::dev::vino::Vino,
}

impl CdmcAdjustedSource {
    pub fn new(inner: Arc<dyn VideoSource>, vino: crate::dev::vino::Vino) -> Self {
        Self { inner, vino }
    }
}

impl VideoSource for CdmcAdjustedSource {
    fn standard(&self) -> VideoStandard { self.inner.standard() }

    fn next_field(&self) -> Field {
        let mut field = self.inner.next_field();
        let regs = self.vino.cdmc_regs();
        let mut pixels = field.pixels.to_vec();
        crate::dev::cdmc::Cdmc::apply_uyvy_field(&mut pixels, &regs);
        field.pixels = Arc::from(pixels);
        field
    }

    fn status(&self) -> String {
        let regs = self.vino.cdmc_regs();
        format!(
            "{}  cdmc gain={:#04x} red_bal={:#04x} blue_bal={:#04x}",
            self.inner.status(),
            regs[crate::dev::cdmc::reg::GAIN as usize],
            regs[crate::dev::cdmc::reg::RED_BAL as usize],
            regs[crate::dev::cdmc::reg::BLUE_BAL as usize],
        )
    }
}

// ─── Black source: emits solid black fields at the standard's field rate ─────

pub struct BlackSource {
    standard: VideoStandard,
    state:    Mutex<BlackState>,
}

struct BlackState {
    next_due: Instant,
    parity:   FieldParity,
}

impl BlackSource {
    pub fn new(standard: VideoStandard) -> Self {
        Self {
            standard,
            state: Mutex::new(BlackState {
                next_due: Instant::now(),
                parity:   FieldParity::Even,
            }),
        }
    }
}

impl VideoSource for BlackSource {
    fn standard(&self) -> VideoStandard { self.standard }

    fn next_field(&self) -> Field {
        let period = self.standard.field_period();
        let (w, h) = self.standard.field_size();

        let (parity, due) = {
            let mut s  = self.state.lock();
            let due    = s.next_due;
            s.next_due = due + period;
            let parity = s.parity;
            s.parity   = match parity {
                FieldParity::Even => FieldParity::Odd,
                FieldParity::Odd  => FieldParity::Even,
            };
            (parity, due)
        };

        let now = Instant::now();
        if now < due {
            std::thread::sleep(due - now);
        } else if now > due + period * 4 {
            self.state.lock().next_due = now + period;
        }

        // Black in UYVY: U=128, Y=16, V=128, Y=16.
        let mut buf = vec![0u8; (w * h * 2) as usize];
        for c in buf.chunks_mut(4) {
            c[0] = 128; c[1] = 16; c[2] = 128; c[3] = 16;
        }
        Field { parity, width: w, height: h, pixels: Arc::from(buf) }
    }
}

// ─── Test pattern: 100% SMPTE bars + animated luma ramp ──────────────────────

pub struct TestPatternSource {
    standard: VideoStandard,
    state:    Mutex<TestState>,
}

struct TestState {
    next_due: Instant,
    counter:  u32,
    parity:   FieldParity,
}

impl TestPatternSource {
    pub fn new(standard: VideoStandard) -> Self {
        Self {
            standard,
            state: Mutex::new(TestState {
                next_due: Instant::now(),
                counter:  0,
                parity:   FieldParity::Even,
            }),
        }
    }
}

impl VideoSource for TestPatternSource {
    fn standard(&self) -> VideoStandard { self.standard }

    fn next_field(&self) -> Field {
        let period = self.standard.field_period();
        let (w, h) = self.standard.field_size();

        let (counter, parity, due) = {
            let mut s = self.state.lock();
            let due = s.next_due;
            s.next_due = due + period;
            s.counter  = s.counter.wrapping_add(1);
            s.parity   = match s.parity {
                FieldParity::Even => FieldParity::Odd,
                FieldParity::Odd  => FieldParity::Even,
            };
            (s.counter, s.parity, due)
        };

        let now = Instant::now();
        if now < due {
            std::thread::sleep(due - now);
        } else if now > due + period * 4 {
            // Far behind — resync to "now" to avoid burst playback.
            self.state.lock().next_due = now + period;
        }

        Field { parity, width: w, height: h, pixels: render(w, h, counter) }
    }
}

/// BT.601 8-bit Y/Cb/Cr for 100% SMPTE bars (top 2/3 of the field).
/// Bottom 1/3 is an animated 8-bit luma ramp that shifts each field.
fn render(w: u32, h: u32, counter: u32) -> Arc<[u8]> {
    const BARS: [(u8, u8, u8); 8] = [
        (235, 128, 128), // white
        (210,  16, 146), // yellow
        (170, 166,  16), // cyan
        (145,  54,  34), // green
        (106, 202, 222), // magenta
        ( 81,  90, 240), // red
        ( 41, 240, 110), // blue
        ( 16, 128, 128), // black
    ];

    let mut buf  = vec![0u8; (w as usize) * (h as usize) * 2];
    let bar_h    = (h * 2) / 3;
    let bar_w    = (w / 8).max(1);

    for y in 0..h {
        for pair in 0..(w / 2) {
            let x0 = pair * 2;
            let x1 = x0 + 1;

            let (y0v, u, y1v, v) = if y < bar_h {
                let b0 = BARS[((x0 / bar_w).min(7)) as usize];
                let b1 = BARS[((x1 / bar_w).min(7)) as usize];
                let u_avg = ((b0.1 as u16 + b1.1 as u16) / 2) as u8;
                let v_avg = ((b0.2 as u16 + b1.2 as u16) / 2) as u8;
                (b0.0, u_avg, b1.0, v_avg)
            } else {
                let l0 = ((x0.wrapping_add(counter)) & 0xFF) as u8;
                let l1 = ((x1.wrapping_add(counter)) & 0xFF) as u8;
                (l0, 128, l1, 128)
            };

            let i = ((y * w + x0) as usize) * 2;
            buf[i    ] = u;
            buf[i + 1] = y0v;
            buf[i + 2] = v;
            buf[i + 3] = y1v;
        }
    }

    Arc::from(buf)
}
