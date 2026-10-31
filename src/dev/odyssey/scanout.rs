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

/// What marked display lines dirty, for [`DirtyStats`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirtySite {
    /// A caller that does not say.
    Other,
    /// A new mode or a reset: the whole frame.
    Mode,
    /// A raster job's rows (DrawnRows::take()).
    Raster,
    /// A swap: the rows the window's buffers differ in (Board::flip_rows()).
    Flip,
    /// A frame whose shape changed while it was resolved (Board::scanout_commit()).
    Reshape,
    /// The lookup RAM, a display mode or an overlay mode.
    Lut,
    /// The cursor's old and new place.
    Cursor,
    /// A pixel store from the front end (Board::paint_pixel()).
    Pixel,
    /// An image transfer's row.
    Image,
    /// A solid rectangle fill (Board::fill_rect()).
    FillRect,
    /// A write to the id or overlay planes (Board::clear_planes()).
    ClearPlanes,
    /// A screen-to-screen blit (Board::copy_rect()).
    CopyRect,
    /// A pattern quad (Board::pattern_fill()).
    PatternFill,
    /// A DMA into the colour buffer (Board::dma_to_buffer()).
    Dma,
    /// An integer pixel that goes through a logic operation or to the id planes.
    Integer,
}

impl DirtySite {
    pub const ALL: [DirtySite; 15] = [
        DirtySite::Other,
        DirtySite::Mode,
        DirtySite::Raster,
        DirtySite::Flip,
        DirtySite::Reshape,
        DirtySite::Lut,
        DirtySite::Cursor,
        DirtySite::Pixel,
        DirtySite::Image,
        DirtySite::FillRect,
        DirtySite::ClearPlanes,
        DirtySite::CopyRect,
        DirtySite::PatternFill,
        DirtySite::Dma,
                DirtySite::Integer,
    ];

    pub fn name(self) -> &'static str {
        match self {
            DirtySite::Other => "other",
            DirtySite::Mode => "mode",
            DirtySite::Raster => "raster",
            DirtySite::Flip => "flip",
            DirtySite::Reshape => "reshape",
            DirtySite::Lut => "lut",
            DirtySite::Cursor => "cursor",
            DirtySite::Pixel => "pixel",
            DirtySite::Image => "image",
            DirtySite::FillRect => "fill_rect",
            DirtySite::ClearPlanes => "clear_planes",
            DirtySite::CopyRect => "copy_rect",
            DirtySite::PatternFill => "pattern_fill",
            DirtySite::Dma => "dma",
            DirtySite::Integer => "integer",
        }
    }
}

/// Standing counters of the display's dirty lines, read by the monitor's
/// `vpro dirty`: per site the marks made and the lines they named, and per
/// display frame the lines resolved. Plain integers under the board lock.
#[derive(Clone, Default, Debug)]
pub struct DirtyStats {
    /// By DirtySite: [marks, lines named].
    pub sites: [[u64; 2]; DirtySite::ALL.len()],
    /// Display frames that staged their lines, and those with any line.
    pub frames: u64,
    pub busy_frames: u64,
    /// Lines resolved in all frames, and in the widest one.
    pub rows: u64,
    pub max_rows: u64,
    /// Swaps latched at a retrace.
    pub swaps: u64,
    /// The frame's wait for the raster unit's context while it held the
    /// board lock, and its hold of both for the plane copy: totals and
    /// longest, in nanoseconds.
    pub wait_ns: u64,
    pub wait_max_ns: u64,
    pub copy_ns: u64,
    pub copy_max_ns: u64,
    /// Frames that let the board lock go to wait for a slice, and those
    /// that gave up and waited with it held.
    pub retries: u64,
    pub fallbacks: u64,
}

impl DirtyStats {
    pub fn staged(&mut self, rows: usize) {
        self.frames += 1;
        if rows != 0 {
            self.busy_frames += 1;
            self.rows += rows as u64;
            self.max_rows = self.max_rows.max(rows as u64);
        }
    }

    /// Time one frame took waiting for the planes and copying them.
    pub fn locked(&mut self, wait: std::time::Duration, copy: std::time::Duration) {
        let (wait, copy) = (wait.as_nanos() as u64, copy.as_nanos() as u64);

        self.wait_ns += wait;
        self.wait_max_ns = self.wait_max_ns.max(wait);
        self.copy_ns += copy;
        self.copy_max_ns = self.copy_max_ns.max(copy);
    }

    pub fn report(&self, w: &mut dyn std::io::Write) -> std::io::Result<()> {
        writeln!(w, "frames {} busy {} rows {} max {} swaps {}", self.frames, self.busy_frames, self.rows, self.max_rows, self.swaps)?;
        writeln!(
            w,
            "planes wait_us {} max {} copy_us {} max {} retries {} fallbacks {}",
            self.wait_ns / 1000,
            self.wait_max_ns / 1000,
            self.copy_ns / 1000,
            self.copy_max_ns / 1000,
            self.retries,
            self.fallbacks
        )?;
        for (site, c) in DirtySite::ALL.iter().zip(&self.sites) {
            if c[0] != 0 {
                writeln!(w, "site {} marks {} rows {}", site.name(), c[0], c[1])?;
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Scanout {
    pub mode: ScanoutMode,
    pub frame: Vec<u32>,  // frame_width x frame_height of 0x00RRGGBB; empty: none
    pub dirty: Vec<bool>, // one flag per display line
    pub frame_width: u32,
    pub frame_height: u32, // what frame was allocated for

    /// Per main drawable id, the rows where the two colour buffers may
    /// differ, a bit a row. A swap of a window shows its other buffer, so
    /// only those rows can change (Board::flip_rows()). Writers that name
    /// their rows keep it (DrawnRows::take()). One that names a display
    /// line, but not the buffer or the drawable, sets the line in
    /// `unnamed`, and the next swap measures the buffers in those lines.
    pub differ: Box<[[u64; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]>,
    pub unnamed: [u64; ROW_WORDS],
    /// The displayed-buffer bits the frame was last staged from.
    pub staged_displayed: u32,

    pub stats: DirtyStats,

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
            differ: Box::new([[0; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]),
            unnamed: [0; ROW_WORDS],
            staged_displayed: 0,
            stats: DirtyStats::default(),
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
        self.invalidate_all_for(DirtySite::Mode);
        self.restart();
    }
    /// Repaint and restart the retrace phase.
    pub fn reset(&mut self) {
        self.invalidate_all_for(DirtySite::Mode);
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

    /// Resolve lines again because something other than the colour planes
    /// changed (the lookup RAM, a display mode, the cursor, a swap).
    pub fn repaint(&mut self, first: i32, count: i32) {
        self.repaint_for(DirtySite::Other, first, count);
    }
    pub fn repaint_for(&mut self, site: DirtySite, first: i32, count: i32) {
        let h: i32 = if !self.frame.is_empty() { self.mode.height as i32 } else { 0 };
        let last = first.wrapping_add(count).wrapping_sub(1);

        let first = first.max(0);
        let last = last.min(h - 1);
        if first > last {
            return;
        }
        let c = &mut self.stats.sites[site as usize];
        c[0] += 1;
        c[1] += (last - first + 1) as u64;
        for d in &mut self.dirty[first as usize..=last as usize] {
            *d = true;
        }
    }
    pub fn repaint_all(&mut self) {
        self.repaint_all_for(DirtySite::Other);
    }
    pub fn repaint_all_for(&mut self, site: DirtySite) {
        self.repaint_for(site, 0, self.mode.height as i32);
    }

    /// Resolve lines again because the planes may have changed, in either
    /// colour buffer: the lines, and the next swap measures the buffers.
    pub fn invalidate(&mut self, first: i32, count: i32) {
        self.invalidate_for(DirtySite::Other, first, count);
    }
    pub fn invalidate_for(&mut self, site: DirtySite, first: i32, count: i32) {
        self.repaint_for(site, first, count);
        let last = first.wrapping_add(count).wrapping_sub(1).min(ROW_WORDS as i32 * 64 - 1);
        for y in first.max(0)..=last {
            self.unnamed[y as usize >> 6] |= 1 << (y & 63);
        }
    }
    /// invalidate_for() the lines [y, y + h) of a rectangle, which a caller
    /// has not clipped to the frame.
    pub fn invalidate_rows_for(&mut self, site: DirtySite, y: u32, h: u32) {
        let first = y.min(i32::MAX as u32);
        let end = (y as u64 + h as u64).min(i32::MAX as u64) as u32;

        self.invalidate_for(site, first as i32, (end - first) as i32);
    }
    pub fn invalidate_all(&mut self) {
        self.invalidate_all_for(DirtySite::Other);
    }
    pub fn invalidate_all_for(&mut self, site: DirtySite) {
        self.invalidate_for(site, 0, self.mode.height as i32);
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
