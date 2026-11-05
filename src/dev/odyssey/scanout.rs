//! An SGI graphics head's scanout mode and retrace. Display line 0 is the
//! top. The retrace is counted on the host's monotonic clock: the
//! deadline (-1 when disarmed) is when the next one is due, and
//! [`RetraceClock::next_retrace`] reports each as it comes due.
//!
//! Ported from include/hw/display/sgi/scanout.h and the shared SGI graphics
//! code in jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;
use super::reader::DisplayShared;
use std::sync::Arc;

pub(super) const SCANOUT_MAX_DIM: u32 = 16384;
/// 1 kHz: well above any SGI display's field rate.
pub(super) const SCANOUT_MIN_PERIOD_NS: i64 = 1000000;

#[derive(Clone, Copy, Default, Debug)]
pub(super) struct ScanoutMode {
    pub width: u32, // visible pixels and lines
    pub height: u32,
    pub frame_clocks: u64, // clocks per frame, blanking included
    pub clock_hz: u32,     // the clock they count; 0: no retrace
}

/// The scanout as the front end sees it: the mode it set. The retrace
/// clock and the rest of what the display keeps are the display's own
/// ([`ScanoutReader`]); the display composes every frame whole from board
/// memory, so no writer marks lines.
#[derive(Clone)]
pub(super) struct Scanout {
    pub mode: ScanoutMode,
    /// Everything the board shares with the display.
    pub shared: Arc<DisplayShared>,
}

/// The retrace clock: the mode's frame period on the host's monotonic
/// clock. `deadline` (-1 when disarmed) is when the next retrace is due,
/// and [`RetraceClock::next_retrace`] reports each as it comes due.
#[derive(Clone, Debug)]
pub(super) struct RetraceClock {
    pub mode: ScanoutMode,
    pub deadline: i64, // virtual ns of the next retrace; -1 disarmed
    pub epoch: i64,    // virtual time of retrace 0
    pub frames: u64,   // retraces since the epoch
}

impl Default for RetraceClock {
    fn default() -> Self {
        RetraceClock { mode: ScanoutMode::default(), deadline: -1, epoch: 0, frames: 0 }
    }
}

impl Scanout {
    /// A scanout over `shared`, with no mode.
    pub fn new(shared: Arc<DisplayShared>) -> Self {
        Scanout { mode: ScanoutMode::default(), shared }
    }

    /// A new mode; the retrace phase restarts when the display next looks.
    pub fn set_mode(&mut self, mode: &ScanoutMode) {
        self.mode = *mode;
        self.mode.sanitize();
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: scanout mode: {}x{} frame {} clocks at {} Hz", self.mode.width, self.mode.height, self.mode.frame_clocks, self.mode.clock_hz);
        self.shared.publish_mode(self.mode);
    }
    /// Restart the retrace phase.
    pub fn reset(&mut self) {
        self.shared.restart_retrace();
    }
}

/// Nanoseconds in a second.
pub(super) const NS_PER_SECOND: u64 = 1_000_000_000;

/// The host's monotonic clock in nanoseconds since the first call: the
/// time base of every scanout's retrace.
pub(super) fn monotonic_ns() -> i64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as i64
}

/// a * b / c without overflowing the product.
pub(super) fn mul_div(a: u64, b: u64, c: u64) -> u64 {
    ((a as u128 * b as u128) / c as u128) as u64
}

impl ScanoutMode {
    /// A mode may be derived from guest-programmed timing tables, so it is
    /// checked before anything is allocated or armed from it. A size past
    /// SCANOUT_MAX_DIM presents nothing, and a period shorter than
    /// SCANOUT_MIN_PERIOD_NS, which would re-arm the retrace faster than
    /// the host can run it, or a frame too long to count, has no retrace.
    /// Returns whether the mode was used as given.
    pub fn sanitize(&mut self) -> bool {
        let mut ok = true;

        if self.width > SCANOUT_MAX_DIM || self.height > SCANOUT_MAX_DIM {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-scanout: {}x{} exceeds the largest mode, {}x{}",
                self.width,
                self.height,
                SCANOUT_MAX_DIM,
                SCANOUT_MAX_DIM
            );
            self.width = 0;
            self.height = 0;
            ok = false;
        }
        if self.clock_hz != 0
            && (self.frame_clocks > u32::MAX as u64
                || (mul_div(self.frame_clocks, NS_PER_SECOND, self.clock_hz as u64) as i64)
                    < SCANOUT_MIN_PERIOD_NS)
        {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-scanout: a frame of {} clocks at {} Hz is outside the retrace periods modelled",
                self.frame_clocks,
                self.clock_hz
            );
            self.clock_hz = 0;
            ok = false;
        }
        ok
    }
}

impl RetraceClock {
    /// Virtual time of retrace n of the current mode, or -1 with no retrace.
    pub fn retrace_time(&self, n: u64) -> i64 {
        let hz = self.mode.clock_hz as u64;
        let fc = self.mode.frame_clocks;

        if hz == 0 || fc == 0 {
            return -1;
        }
        // n * fc / hz seconds, exactly, without overflowing for any n a run
        // reaches: whole seconds' worth of frames first, then the rest, whose
        // product fits because both factors are below 2^32.
        (self.epoch as u64)
            .wrapping_add((n / hz).wrapping_mul(fc).wrapping_mul(NS_PER_SECOND))
            .wrapping_add(mul_div((n % hz).wrapping_mul(fc), NS_PER_SECOND, hz)) as i64
    }

    /// Whether a retrace has come due at `now_ns`. If so it is counted and
    /// the next one armed, and the caller runs it; a caller that has fallen
    /// behind gets each missed retrace in turn.
    pub fn next_retrace(&mut self, now_ns: i64) -> bool {
        if self.deadline < 0 || now_ns < self.deadline {
            return false;
        }
        self.frames += 1;
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: scanout retrace: frame {}", self.frames);
        self.arm();
        true
    }

    pub fn arm(&mut self) {
        let when = self.retrace_time(self.frames + 1);

        self.deadline = if when < 0 { -1 } else { when };
    }

    pub fn restart(&mut self) {
        self.epoch = monotonic_ns();
        self.frames = 0;
        self.arm();
    }
}
