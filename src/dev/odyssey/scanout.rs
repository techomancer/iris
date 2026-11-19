//! An SGI graphics head's scanout mode. Display line 0 is the top.
//!
//! Ported from include/hw/display/sgi/scanout.h and the shared SGI graphics
//! code in jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;
use super::reader::DisplayShared;
use std::sync::Arc;

pub(super) const SCANOUT_MAX_DIM: u32 = 16384;
/// 1 kHz: well above any SGI display's field rate.
pub(super) const SCANOUT_MIN_PERIOD_NS: i64 = 1000000;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(super) struct ScanoutMode {
    pub width: u32, // visible pixels and lines
    pub height: u32,
    pub frame_clocks: u64, // clocks per frame, blanking included
    pub clock_hz: u32,     // the clock they count; 0: no retrace
    pub line_clocks: u32,  // clocks per line, blanking included
    pub blank_lines: u32,  // lines of vertical blanking per field
    pub fields: u32,       // fields per frame, each with its own retrace
}

impl ScanoutMode {
    /// The mode as the refresh thread's timing; None with no retrace.
    pub fn video_timing(&self) -> Option<crate::display::VideoTiming> {
        (self.clock_hz != 0 && self.line_clocks != 0).then(|| crate::display::VideoTiming {
            h_total: self.line_clocks,
            v_total: (self.frame_clocks / self.line_clocks as u64) as u32,
            pixel_clock_hz: self.clock_hz as u64,
            vblank_lines: self.blank_lines,
            fields: self.fields.max(1),
        })
    }
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
}

/// Nanoseconds in a second.
pub(super) const NS_PER_SECOND: u64 = 1_000_000_000;

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
