//! An SGI graphics head's scanout: its presented frame, dirty lines and
//! retrace. Display line 0 is the top.
//!
//! `frame` holds the presented frame (0x00RRGGBB, `frame_width` by
//! `frame_height`), which the board's display thread copies to the window;
//! `dirty` marks the lines the board resolves again
//! (Board::scanout_update()). The retrace is counted on the host's
//! monotonic clock: `retrace_deadline` (-1 when disarmed) is when the next
//! one is due, and [`Scanout::next_retrace`] reports each as it comes due.
//!
//! Ported from include/hw/display/sgi/scanout.h and the shared SGI graphics
//! code in jamesbraid/qemu-sgi-origami.
use super::*;

pub const SGI_SCANOUT_MAX_DIM: u32 = 16384;
/// 1 kHz: well above any SGI display's field rate.
pub const SGI_SCANOUT_MIN_PERIOD_NS: i64 = 1000000;

#[derive(Clone, Copy, Default, Debug)]
pub struct ScanoutMode {
    pub width: u32, // visible pixels and lines
    pub height: u32,
    pub frame_clocks: u64, // clocks per frame, blanking included
    pub clock_hz: u32,     // the clock they count; 0: no retrace
}

#[derive(Clone)]
pub struct Scanout {
    pub mode: ScanoutMode,
    pub frame: Vec<u32>,  // frame_width x frame_height of 0x00RRGGBB; empty: none
    pub dirty: Vec<bool>, // one flag per display line
    pub frame_width: u32,
    pub frame_height: u32, // what frame was allocated for

    pub retrace_deadline: i64, // virtual ns of the next retrace; -1 disarmed
    pub epoch: i64,            // virtual time of retrace 0
    pub frames: u64,           // retraces since the epoch
}

impl Default for Scanout {
    fn default() -> Self {
        Scanout {
            mode: ScanoutMode::default(),
            frame: Vec::new(),
            dirty: Vec::new(),
            frame_width: 0,
            frame_height: 0,
            retrace_deadline: -1,
            epoch: 0,
            frames: 0,
        }
    }
}

impl Scanout {
    /// A new mode: new frame, everything dirty, retrace phase restarted now.
    pub fn set_mode(&mut self, mode: &ScanoutMode) {
        self.mode = *mode;
        self.mode.sanitize();
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: scanout mode: {}x{} frame {} clocks at {} Hz", self.mode.width, self.mode.height, self.mode.frame_clocks, self.mode.clock_hz);
        self.alloc();
        self.invalidate_all();
        self.restart();
    }
    /// Repaint and restart the retrace phase.
    pub fn reset(&mut self) {
        self.invalidate_all();
        self.restart();
    }

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
        if self.retrace_deadline < 0 || now_ns < self.retrace_deadline {
            return false;
        }
        self.frames += 1;
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: scanout retrace: frame {}", self.frames);
        self.arm();
        true
    }

    pub fn invalidate(&mut self, first: i32, count: i32) {
        let h: i32 = if !self.frame.is_empty() { self.mode.height as i32 } else { 0 };
        let last = first.wrapping_add(count).wrapping_sub(1);

        let first = first.max(0);
        let last = last.min(h - 1);
        if first > last {
            return;
        }
        for d in &mut self.dirty[first as usize..=last as usize] {
            *d = true;
        }
    }
    pub fn invalidate_all(&mut self) {
        self.invalidate(0, self.mode.height as i32);
    }
}

/// Nanoseconds in a second.
pub const NS_PER_SECOND: u64 = 1_000_000_000;

/// The host's monotonic clock in nanoseconds since the first call: the
/// time base of every scanout's retrace.
pub fn monotonic_ns() -> i64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_nanos() as i64
}

/// a * b / c without overflowing the product.
pub fn mul_div(a: u64, b: u64, c: u64) -> u64 {
    ((a as u128 * b as u128) / c as u128) as u64
}

impl ScanoutMode {
    /// A mode may be derived from guest-programmed timing tables, so it is
    /// checked before anything is allocated or armed from it. A size past
    /// SGI_SCANOUT_MAX_DIM presents nothing, and a period shorter than
    /// SGI_SCANOUT_MIN_PERIOD_NS, which would re-arm the retrace faster than
    /// the host can run it, or a frame too long to count, has no retrace.
    /// Returns whether the mode was used as given.
    pub fn sanitize(&mut self) -> bool {
        let mut ok = true;

        if self.width > SGI_SCANOUT_MAX_DIM || self.height > SGI_SCANOUT_MAX_DIM {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-scanout: {}x{} exceeds the largest mode, {}x{}",
                self.width,
                self.height,
                SGI_SCANOUT_MAX_DIM,
                SGI_SCANOUT_MAX_DIM
            );
            self.width = 0;
            self.height = 0;
            ok = false;
        }
        if self.clock_hz != 0
            && (self.frame_clocks > u32::MAX as u64
                || (mul_div(self.frame_clocks, NS_PER_SECOND, self.clock_hz as u64) as i64)
                    < SGI_SCANOUT_MIN_PERIOD_NS)
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

impl Scanout {
    pub fn arm(&mut self) {
        let when = self.retrace_time(self.frames + 1);

        self.retrace_deadline = if when < 0 { -1 } else { when };
    }

    pub fn restart(&mut self) {
        self.epoch = monotonic_ns();
        self.frames = 0;
        self.arm();
    }

    /// A frame and dirty map for the mode, when its size changed.
    pub fn alloc(&mut self) {
        let w = self.mode.width;
        let h = self.mode.height;

        if !self.frame.is_empty() && w == self.frame_width && h == self.frame_height {
            return;
        }
        self.frame = Vec::new();
        self.dirty = Vec::new();
        self.frame_width = 0;
        self.frame_height = 0;
        if w != 0 && h != 0 {
            self.frame = vec![0u32; w as usize * h as usize];
            self.dirty = vec![false; h as usize];
            self.frame_width = w;
            self.frame_height = h;
        }
    }}
