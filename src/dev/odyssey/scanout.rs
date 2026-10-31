//! An SGI graphics head's scanout: its presented frame, dirty lines and
//! retrace. Display line 0 is the top.
//!
//! `frame` holds the presented frame (0x00RRGGBB, `frame_width` by
//! `frame_height`), which the board's display thread copies to the window;
//! `marks` holds the lines the board resolves again
//! (Board::scanout_update()), which any thread that writes a plane sets and
//! the display consumes. The retrace is counted on the host's
//! monotonic clock: `retrace_deadline` (-1 when disarmed) is when the next
//! one is due, and [`Scanout::next_retrace`] reports each as it comes due.
//!
//! Ported from include/hw/display/sgi/scanout.h and the shared SGI graphics
//! code in jamesbraid/qemu-sgi-origami.
use super::*;

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

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
/// `vpro dirty`: per display frame the lines resolved, and the waits for
/// the planes. Plain integers under the board lock; the per-site marks the
/// writers make are atomics in [`RowMarks`].
#[derive(Clone, Default, Debug)]
pub struct DirtyStats {
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

    pub fn report(&self, marks: &RowMarks, w: &mut dyn std::io::Write) -> std::io::Result<()> {
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
        for site in DirtySite::ALL {
            let c = marks.site(site);
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
    pub frame: Vec<u32>, // frame_width x frame_height of 0x00RRGGBB; empty: none
    /// The lines to resolve again, and the lines a writer touched without
    /// naming a buffer: shared with every thread that writes a plane.
    pub marks: Arc<RowMarks>,
    pub frame_width: u32,
    pub frame_height: u32, // what frame was allocated for

    /// Per main drawable id, the rows where the two colour buffers may
    /// differ, a bit a row. A swap of a window shows its other buffer, so
    /// only those rows can change (Board::flip_rows()). Writers that name
    /// their rows keep it (DrawnRows::take()). One that names a display
    /// line, but not the buffer or the drawable, sets the line in
    /// `RowMarks::unnamed`, and the next swap measures the buffers in those
    /// lines.
    pub differ: Box<[[u64; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]>,
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
            marks: Arc::new(RowMarks::default()),
            frame_width: 0,
            frame_height: 0,
            differ: Box::new([[0; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize]),
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
        self.marks.repaint(site, first, count);
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
        self.marks.invalidate(site, first, count);
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

/// Words of the dirty-line bitset: a bit a display line of the largest mode.
pub const DIRTY_WORDS: usize = (SGI_SCANOUT_MAX_DIM as usize).div_ceil(64);

/// The display lines that must be resolved again, and the lines a writer
/// touched without naming a colour buffer, as bitsets any thread may set.
///
/// A writer stores its pixels first and marks them second, with Release; the
/// display takes the marks with `swap(0, Acquire)` and only then reads the
/// planes. A row the display copied while a writer was still storing is
/// therefore marked again after the writer's last store, and the next frame
/// copies it whole. A mark published before its pixels would let the display
/// consume it, copy the old pixels and never hear of the new ones.
///
/// The display takes marks and must put back any it does not apply
/// ([`RowMarks::repaint`]).
pub struct RowMarks {
    dirty: Box<[AtomicU64; DIRTY_WORDS]>,
    unnamed: [AtomicU64; ROW_WORDS],
    /// Lines of the frame the marks are for; 0 with no frame. A mark past it
    /// is dropped.
    height: AtomicU32,
    /// By DirtySite: [marks, lines named].
    sites: [[AtomicU64; 2]; DirtySite::ALL.len()],
}

impl Default for RowMarks {
    fn default() -> Self {
        RowMarks {
            dirty: Box::new(std::array::from_fn(|_| AtomicU64::new(0))),
            unnamed: std::array::from_fn(|_| AtomicU64::new(0)),
            height: AtomicU32::new(0),
            sites: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
        }
    }
}

impl RowMarks {
    /// The frame is `height` lines tall: marks past it are dropped, and a
    /// new height forgets the lines marked for the old one (the caller marks
    /// the new frame whole).
    pub fn set_height(&self, height: u32) {
        if self.height.swap(height, Ordering::AcqRel) != height {
            for w in self.dirty.iter() {
                w.store(0, Ordering::Release);
            }
        }
    }

    /// The lines of the frame, 0 with none.
    pub fn height(&self) -> u32 {
        self.height.load(Ordering::Acquire)
    }

    /// Mark lines [first, first + count) to be resolved again, for a change
    /// that is not in the colour planes (the lookup RAM, the cursor, a
    /// swap) or that the caller has already named in `unnamed`.
    pub fn repaint(&self, site: DirtySite, first: i32, count: i32) {
        let h = self.height() as i32;
        let last = first.wrapping_add(count).wrapping_sub(1);

        let first = first.max(0);
        let last = last.min(h - 1);
        if first > last {
            return;
        }
        let c = &self.sites[site as usize];
        c[0].fetch_add(1, Ordering::Relaxed);
        c[1].fetch_add((last - first + 1) as u64, Ordering::Relaxed);
        set_range(&self.dirty[..], first as usize, last as usize);
    }

    /// Mark lines [first, first + count) after a store to the planes, in
    /// either colour buffer: the lines, and the next swap measures the
    /// buffers there. The caller has stored its pixels already. The lines
    /// are named in `unnamed` before they are marked dirty: the other order
    /// would let a frame take the dirty mark and a swap latch before the
    /// unnamed mark, and the swap would not repaint a line whose buffers
    /// differ, for good.
    pub fn invalidate(&self, site: DirtySite, first: i32, count: i32) {
        self.name_unnamed(first, count);
        self.repaint(site, first, count);
    }

    /// The first half of invalidate(): lines whose buffer the writer did not
    /// name.
    pub fn name_unnamed(&self, first: i32, count: i32) {
        let last = first.wrapping_add(count).wrapping_sub(1).min(ROW_WORDS as i32 * 64 - 1);

        if first.max(0) <= last {
            set_range(&self.unnamed[..], first.max(0) as usize, last as usize);
        }
    }

    /// Take every dirty line below `height`, in order, and clear the marks.
    /// Read the planes only after this.
    pub fn take_dirty(&self, height: usize, rows: &mut Vec<u32>) {
        for (i, word) in self.dirty.iter().enumerate() {
            let mut w = word.swap(0, Ordering::Acquire);

            while w != 0 {
                let y = i * 64 + w.trailing_zeros() as usize;
                w &= w - 1;
                if y < height {
                    rows.push(y as u32);
                }
            }
        }
    }

    /// Whether any line was touched without naming its buffer and not yet
    /// measured.
    pub fn unnamed_pending(&self) -> bool {
        self.unnamed.iter().any(|w| w.load(Ordering::Acquire) != 0)
    }

    /// Take the lines touched without naming a buffer, and clear them. A
    /// write that lands after this sets its line again, and the next swap
    /// measures it.
    pub fn take_unnamed(&self) -> [u64; ROW_WORDS] {
        std::array::from_fn(|i| self.unnamed[i].swap(0, Ordering::Acquire))
    }

    /// [marks, lines named] by `site`.
    pub fn site(&self, site: DirtySite) -> [u64; 2] {
        let c = &self.sites[site as usize];
        [c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed)]
    }

    pub fn reset_sites(&self) {
        for c in &self.sites {
            c[0].store(0, Ordering::Relaxed);
            c[1].store(0, Ordering::Relaxed);
        }
    }

    /// Tests: the lines marked dirty now, not taken.
    #[cfg(test)]
    pub fn dirty_rows(&self) -> Vec<usize> {
        (0..DIRTY_WORDS * 64).filter(|&y| self.dirty[y >> 6].load(Ordering::Acquire) >> (y & 63) & 1 != 0).collect()
    }

    /// Tests: forget every dirty line.
    #[cfg(test)]
    pub fn clear_dirty(&self) {
        for w in self.dirty.iter() {
            w.store(0, Ordering::Release);
        }
    }
}

/// Set bits first..=last of a bitset, a word at a time, with Release.
fn set_range(words: &[AtomicU64], first: usize, last: usize) {
    for i in first >> 6..=last >> 6 {
        let lo = if i == first >> 6 { first & 63 } else { 0 };
        let hi = if i == last >> 6 { last & 63 } else { 63 };
        let mask = (u64::MAX >> (63 - (hi - lo))) << lo;

        words[i].fetch_or(mask, Ordering::Release);
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
        self.frame_width = 0;
        self.frame_height = 0;
        self.marks.set_height(0);
        if w != 0 && h != 0 {
            self.frame = vec![0u32; w as usize * h as usize];
            self.frame_width = w;
            self.frame_height = h;
            self.marks.set_height(h);
        }
    }
}
