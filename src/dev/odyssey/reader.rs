//! What the display thread reads of the board, and nothing else.
//!
//! The display copies the board's planes to the window sixty times a second.
//! It reads the planes the way the raster unit and the front end write them,
//! without a lock, because the writers mark the rows they change after they
//! change them (`RowMarks`) and the display takes the marks before it reads
//! (`ScanoutReader::stage`): a row copied half-written is marked again after
//! the writer's last store and copied whole by the next frame. The rest of
//! what a frame reads is shared through cells of its own, each held only to
//! copy:
//!
//! - the rows the raster unit drew, published at completions, at drain and
//!   at the end of a slice (`RasterCtx::publish`);
//! - the lookup RAM, the display modes and the cursor (`DisplayRegs`), with
//!   a generation that moves when the front end stores to them;
//! - the scanout mode and the retrace restart (`DisplayShared::publish_mode`);
//! - the swap latch and the displayed-buffer bits, which are the host
//!   interface's (`HostInterface::retrace`).
//!
//! Lock order: the board lock, then `DisplayRegs`; the display takes
//! `DisplayRegs`, the mode cell, the published rows and the host interface
//! one at a time and never the board lock. `DisplayRegs` is a leaf: nothing
//! is taken while it is held.
//!
//! The display never calls a `Board` method. Plane memory is fixed at
//! `Board::allocate`, and `PlaneView` holds its base pointers, lengths and
//! geometry from then on; `PlaneView::copy_row` is the one function that
//! reads memory a writer may be storing to.
use super::*;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{Mutex, MutexGuard};

/// The rows of each main drawable id where its two colour buffers may
/// differ, a bit a row (see [`ScanoutReader::differ`]).
pub type Differ = [[u64; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize];

/// The back end's registers the scanout resolves through: the colour
/// lookup RAM, the per-drawable display modes, the overlay modes and the
/// hardware cursor.
#[derive(Clone)]
pub struct DisplayRegs {
    /// The colour lookup RAM (0x2000 entries), one 30-bit colour per entry.
    pub lut: Vec<u32>,
    pub dispmode: [u32; 32],
    /// The overlay ids' display modes, backend entries 0x2920 + id.
    pub overlay_mode: [u32; 8],
    /// The cursor's backend entries 0x2500-0x25ff.
    pub cursor: [u32; 0x100],
    /// Moves on every store through [`SharedRegs::edit`].
    gen: u64,
}

impl Default for DisplayRegs {
    fn default() -> Self {
        DisplayRegs { lut: vec![0; ODSY_LUT_SIZE as usize], dispmode: [0; 32], overlay_mode: [0; 8], cursor: [0; 0x100], gen: 0 }
    }
}

impl DisplayRegs {
    /// Back to the reset values: everything zero, cursor off.
    pub fn reset(&mut self) {
        self.lut.fill(0);
        self.dispmode = [0; 32];
        self.overlay_mode = [0; 8];
        self.cursor = [0; 0x100];
    }

    /// The top eight bits of a lookup-RAM entry's ten-bit channels, as RGB.
    pub fn lut_rgb(&self, entry: u32) -> u32 {
        let word = if entry < ODSY_LUT_SIZE { self.lut[entry as usize] } else { 0 };

        (((word >> ODSY_LUT_RED_SHIFT) & 0xff) << 16) | (((word >> ODSY_LUT_GREEN_SHIFT) & 0xff) << 8) | ((word >> ODSY_LUT_BLUE_SHIFT) & 0xff)
    }

    /// One 8-bit RGB channel through the 1,536-entry gamma table: widened to 12
    /// bits as the X server widens it (c << 4 | c >> 4; the board's own widening
    /// is unknown), mapped to the table in three ranges -- v below 0x200, v / 2 +
    /// 0x100 below 0x400, else v / 4 + 0x200 -- and shown as the top eight bits of
    /// that entry's channel at shift. Inferred: that the table applies to RGB
    /// pixels at all (the blank zeroes it to blacken the screen).
    pub fn gamma(&self, c: u32, shift: u32) -> u32 {
        let v = (c << 4) | (c >> 4);
        let t = if v < 0x200 {
            v
        } else if v < 0x400 {
            v / 2 + 0x100
        } else {
            v / 4 + 0x200
        };

        (self.lut[(ODSY_GAMMA_BASE + t) as usize] >> shift) & 0xff
    }

    /// The colormap group main drawable main_id shows through: bits 8:4 of its
    /// display mode. Below ODSY_GAMMA_GROUP the plane word is an index.
    pub fn display_group(&self, main_id: u32) -> u32 {
        (self.dispmode[main_id as usize] >> ODSY_DISPMODE_GROUP_SHIFT) & ODSY_DISPMODE_GROUP_MASK
    }

    /// The position's components, each a signed 16-bit value (inferred).
    pub fn cursor_x(&self) -> i32 {
        (self.cursor[ODSY_CURSOR_POSITION as usize] >> 16) as i16 as i32
    }

    pub fn cursor_y(&self) -> i32 {
        self.cursor[ODSY_CURSOR_POSITION as usize] as i16 as i32
    }

    /// Whether anything is drawn: bit 0 set and bit 5 clear. APPROXIMATION: on
    /// the second channel the cursor is on a display the model does not have.
    pub fn cursor_shown(&self) -> bool {
        let ctl = self.cursor[ODSY_CURSOR_CONTROL as usize];

        (ctl & ODSY_CURSOR_CTL_ENABLE) != 0 && (ctl & ODSY_CURSOR_CTL_CHANNEL1) == 0
    }

    /// The first and last display line the cursor covers as it stands in a
    /// frame whose last line is `bottom`, or None when it covers none.
    pub fn cursor_span(&self, bottom: i32) -> Option<(i32, i32)> {
        if !self.cursor_shown() {
            return None;
        }
        if self.cursor[ODSY_CURSOR_CONTROL as usize] & ODSY_CURSOR_CTL_CROSSHAIR != 0 {
            // The vertical line crosses every line.
            return Some((0, bottom));
        }
        Some((self.cursor_y().max(0), (self.cursor_y() + ODSY_CURSOR_SIZE as i32 - 1).min(bottom)))
    }

    /// Draw the cursor over display line y, already resolved into dst. With bit
    /// 2 it is a crosshair: APPROXIMATION, a one-pixel horizontal and vertical
    /// line across the whole raster through the position, in colour word 1. The
    /// crosshair's extent, width and colour are unknown.
    pub fn cursor_row(&self, y: u32, dst: &mut [u32], width: u32) {
        let cx = self.cursor_x();
        let cy = self.cursor_y();
        let row = (y as i32).wrapping_sub(cy);

        if !self.cursor_shown() {
            return;
        }
        if self.cursor[ODSY_CURSOR_CONTROL as usize] & ODSY_CURSOR_CTL_CROSSHAIR != 0 {
            let colour = cursor_rgb(self.cursor[ODSY_CURSOR_COLOURS as usize + 1]);

            if row == 0 {
                dst[..width as usize].fill(colour);
            } else if cx >= 0 && cx < width as i32 {
                dst[cx as usize] = colour;
            }
            return;
        }
        if row < 0 || row >= ODSY_CURSOR_SIZE as i32 {
            return;
        }
        for col in 0..ODSY_CURSOR_SIZE as i32 {
            let x = cx.wrapping_add(col);
            let plane = (self.cursor[(ODSY_CURSOR_PLANE as i32 + row * 2 + col / 16) as usize] >> ((col % 16) * 2)) & 3;
            let glyph = (self.cursor[(ODSY_CURSOR_GLYPH as i32 + row * 4 + col / 8) as usize] >> ((col % 8) * 4)) & 0xf;

            if x < 0 || x >= width as i32 || plane != 3 || glyph == 0xf {
                continue;
            }
            dst[x as usize] = cursor_rgb(self.cursor[ODSY_CURSOR_COLOURS as usize + glyph as usize]);
        }
    }
}

/// [`DisplayRegs`] behind a mutex of its own. The front end holds it for
/// the store of a register and marks the lines afterwards; the display holds
/// it to copy.
#[derive(Default)]
pub struct SharedRegs(Mutex<DisplayRegs>);

impl SharedRegs {
    pub fn read(&self) -> MutexGuard<'_, DisplayRegs> {
        self.0.lock()
    }

    /// The registers for a store: the generation moves as the guard is
    /// written through, so a reader that sees it unchanged has seen
    /// every store. Mark the lines the store changes after dropping the
    /// guard.
    pub fn edit(&self) -> RegsEdit<'_> {
        RegsEdit(self.0.lock())
    }
}

pub struct RegsEdit<'a>(MutexGuard<'a, DisplayRegs>);

impl std::ops::Deref for RegsEdit<'_> {
    type Target = DisplayRegs;
    fn deref(&self) -> &DisplayRegs {
        &self.0
    }
}

impl std::ops::DerefMut for RegsEdit<'_> {
    fn deref_mut(&mut self) -> &mut DisplayRegs {
        self.0.gen += 1;
        &mut self.0
    }
}

/// The scanout mode as the front end last set it, and the counts a display
/// compares to notice a new one.
#[derive(Default)]
struct ModeCell {
    mode: ScanoutMode,
    /// Moves when the mode is set.
    gen: u64,
    /// Moves when the retrace phase restarts: a new mode or a reset.
    restart: u64,
}

/// One byte range of board memory, as the display sees it.
#[derive(Clone, Copy)]
struct RawBytes {
    ptr: *const u8,
    len: usize,
}

impl RawBytes {
    fn of(bytes: &[u8]) -> Self {
        RawBytes { ptr: bytes.as_ptr(), len: bytes.len() }
    }
}

/// Which plane a row is copied from.
#[derive(Clone, Copy)]
pub enum RowOf {
    Colour(usize),
    Did,
    Overlay,
}

/// The board's colour, id and overlay planes as the display reaches them:
/// base pointers, lengths and the colour planes' geometry, captured when
/// the board allocates them and valid while the board lives. Plane memory
/// is allocated once and never resized or moved; reset fills it in place.
pub struct PlaneView {
    colour: [RawBytes; 2],
    planes: [Plane; 2],
    did: RawBytes,
    overlay: RawBytes,
}

// SAFETY: the view holds pointers into the board's plane memory, which
// is allocated once (Board::allocate) and lives as long as the board; the
// display thread holds a handle to the board for as long as it runs. The
// view only copies bytes out.
unsafe impl Send for PlaneView {}
unsafe impl Sync for PlaneView {}

impl PlaneView {
    /// The view of `b`'s planes, once they are allocated.
    pub fn of(b: &Board) -> Option<PlaneView> {
        let planes = [b.buffer_plane(0)?, b.buffer_plane(1)?];

        Some(PlaneView {
            colour: [RawBytes::of(&b.mem.bytes), RawBytes::of(&b.buffer1_mem.bytes)],
            planes,
            did: RawBytes::of(&b.did),
            overlay: RawBytes::of(&b.overlay),
        })
    }

    /// Where colour buffer b's line y starts, and whether its words are
    /// big-endian, when `width` pixels of it are whole in a four-byte plane.
    fn colour_at(&self, b: usize, y: u32, width: usize) -> Option<(usize, bool)> {
        let p = &self.planes[b];
        let at = usize::try_from(p.offset(0, y as i32, 4)).ok()?;

        if plane_bpp(p.format) != 4 || !p.contains(width as i32 - 1, y as i32) {
            return None;
        }
        (at.checked_add(width * 4)? <= self.colour[b].len).then_some((at, !p.is_le()))
    }

    /// Copy row y of `what` into `out`: `out.len() / 4` colour words, or
    /// `out.len()` id or overlay bytes. For a colour row, whether its words
    /// are big-endian; None unless the row is whole in a four-byte plane
    /// (and for the id and overlay bytes, unless it is in them).
    ///
    /// This is the one place the display reads memory a writer may be
    /// storing to. The race is tolerated: the writer marks the row after
    /// its last store, so a copy that saw part of a write is made again at
    /// the next frame. Every comparison and resolve works on the copy.
    pub fn copy_row(&self, what: RowOf, y: u32, out: &mut [u8]) -> Option<bool> {
        let (raw, at, be) = match what {
            RowOf::Colour(b) => {
                let (at, be) = self.colour_at(b, y, out.len() / 4)?;
                (self.colour[b], at, be)
            }
            RowOf::Did | RowOf::Overlay => {
                let raw = if let RowOf::Did = what { self.did } else { self.overlay };
                (raw, y as usize * SGI_ODYSSEY_WIDTH as usize, false)
            }
        };
        if at.checked_add(out.len())? > raw.len {
            return None;
        }
        // SAFETY: the range is inside the allocation (checked above), which
        // lives as long as the board and is never moved; `out` is ours.
        unsafe { std::ptr::copy_nonoverlapping(raw.ptr.add(at), out.as_mut_ptr(), out.len()) };
        Some(be)
    }

    /// Colour buffer b's word at (x, y), or 0 outside the plane.
    fn load(&self, b: usize, x: i64, y: i64) -> u32 {
        let p = &self.planes[b];
        let bpp = plane_bpp(p.format);
        let mut bytes = [0u8; 4];

        if x < 0 || y < 0 || x > i32::MAX as i64 || y > i32::MAX as i64 || !p.contains(x as i32, y as i32) {
            return 0;
        }
        let Ok(at) = usize::try_from(p.offset(x as i32, y as i32, bpp)) else { return 0 };
        if at.checked_add(bpp as usize).is_none_or(|end| end > self.colour[b].len) {
            return 0;
        }
        // SAFETY: as copy_row().
        unsafe { std::ptr::copy_nonoverlapping(self.colour[b].ptr.add(at), bytes.as_mut_ptr(), bpp as usize) };
        p.get(bpp, &bytes)
    }

    /// One id or overlay byte, 0 outside.
    fn byte(&self, what: RowOf, at: usize) -> u8 {
        let mut b = [0u8];

        self.copy_row_bytes(what, at, &mut b);
        b[0]
    }

    fn copy_row_bytes(&self, what: RowOf, at: usize, out: &mut [u8]) {
        let raw = if let RowOf::Did = what { self.did } else { self.overlay };

        if at.checked_add(out.len()).is_some_and(|end| end <= raw.len) {
            // SAFETY: as copy_row().
            unsafe { std::ptr::copy_nonoverlapping(raw.ptr.add(at), out.as_mut_ptr(), out.len()) };
        }
    }

    /// resolve_row_generic() on the view: the per-pixel form, which any
    /// plane shape can use. Row y of `width` pixels into `dst`, with
    /// colour buffer (displayed >> id) & 1 showing for each pixel's drawable.
    /// Whether a pixel showed through colour map group 25.
    pub fn resolve_generic(&self, regs: &DisplayRegs, displayed: u32, y: u32, width: u32, dst: &mut [u32]) -> bool {
        let mut saw_unverified = false;

        for x in 0..width {
            let at = (y * SGI_ODYSSEY_WIDTH + x) as usize;
            let did = self.byte(RowOf::Did, at);
            let main_id = ((did as u32) >> ODSY_DID_MAIN_SHIFT) & (ODSY_DISPMODE_COUNT - 1);
            let word = self.load(((displayed >> main_id) & 1) as usize, x as i64, y as i64);
            let group = regs.display_group(main_id);

            let omode = regs.overlay_mode[((did as u32) & ODSY_DID_OVERLAY_MASK) as usize];
            let opixel = self.byte(RowOf::Overlay, at);

            if (omode & ODSY_OVERLAY_MODE_ENABLE) != 0 && opixel != 0 {
                dst[x as usize] = regs.lut_rgb(256 * ((omode >> ODSY_OVERLAY_MODE_GROUP_SHIFT) & ODSY_OVERLAY_MODE_GROUP_MASK) + opixel as u32);
                continue;
            }
            if group >= ODSY_GAMMA_GROUP {
                let mode = regs.dispmode[main_id as usize];
                let r = display_channel(mode, word >> 24);
                let g = display_channel(mode, (word >> 16) & 0xff);
                let b = display_channel(mode, (word >> 8) & 0xff);

                dst[x as usize] = regs.gamma(r, ODSY_LUT_RED_SHIFT) << 16 | regs.gamma(g, ODSY_LUT_GREEN_SHIFT) << 8 | regs.gamma(b, ODSY_LUT_BLUE_SHIFT);
                continue;
            }
            saw_unverified |= group == ODSY_CMAP_UNVERIFIED_GROUP;
            dst[x as usize] = regs.lut_rgb(if group != 0 { 256 * group + (word & 0xff) } else { word & ODSY_INDEX_MASK });
        }
        regs.cursor_row(y, dst, width);
        saw_unverified
    }
}

/// What the board shares with the display thread.
pub struct DisplayShared {
    pub marks: Arc<RowMarks>,
    pub regs: SharedRegs,
    mode: Mutex<ModeCell>,
    pub planes: OnceLock<PlaneView>,
    pub front: Arc<Mutex<HostInterface>>,
    pub raster: Arc<RasterQueue>,
    /// Swaps latched at a retrace.
    pub swaps: AtomicU64,
    /// Colour map group 25 was resolved and reported.
    pub group25: AtomicBool,
}

impl DisplayShared {
    pub fn new(marks: Arc<RowMarks>, front: Arc<Mutex<HostInterface>>, raster: Arc<RasterQueue>) -> Self {
        DisplayShared {
            marks,
            regs: SharedRegs::default(),
            mode: Mutex::new(ModeCell { gen: 1, restart: 1, ..Default::default() }),
            planes: OnceLock::new(),
            front,
            raster,
            swaps: AtomicU64::new(0),
            group25: AtomicBool::new(false),
        }
    }

    /// A new mode: the display reallocates its frame and restarts the
    /// retrace phase when it next looks.
    pub fn publish_mode(&self, mode: ScanoutMode) {
        let mut cell = self.mode.lock();

        cell.mode = mode;
        cell.gen += 1;
        cell.restart += 1;
    }

    /// The retrace phase restarts, as for a reset.
    pub fn restart_retrace(&self) {
        self.mode.lock().restart += 1;
    }

    /// A vertical retrace: the host interface latches the swaps armed and
    /// raises the retrace source (HostInterface::retrace()).
    pub fn retrace(&self) {
        if self.front.lock().retrace() {
            self.swaps.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Report colour map group 25 once.
    pub fn report_group25(&self) {
        if !self.group25.swap(true, Ordering::Relaxed) {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: colormap group 25 (0x1900) resolved as an index table; unverified");
        }
    }
}

/// The display's own state: the presented frame, the retrace clock, the
/// rows where a window's buffers differ, and the lines of the update in
/// progress. Only the display thread (or a test standing in for it) uses it.
#[derive(Default)]
pub struct ScanoutReader {
    shared: Option<Arc<DisplayShared>>,
    /// The presented frame: `frame_width` by `frame_height` of 0x00RRGGBB;
    /// empty with no mode.
    pub frame: Vec<u32>,
    pub frame_width: u32,
    pub frame_height: u32,
    mode_gen: u64,
    restart_gen: u64,
    /// The mode of the frame, and its retrace clock.
    pub clock: RetraceClock,
    /// The frame was just allocated: every line is resolved, whatever the
    /// marks say.
    reshaped: bool,
    /// Per main drawable id, the rows where the two colour buffers may
    /// differ, a bit a row. A swap of a window shows its other buffer, so
    /// only those rows can change (flip_rows()). The rows the raster unit
    /// published join it; one a writer touched without naming its buffer is
    /// measured at the swap (measure_differ()).
    pub differ: Box<Differ>,
    /// The displayed-buffer bits the frame was last staged from.
    staged_displayed: u32,
    /// The displayed-buffer bits of the frame in progress: read once, after
    /// the retrace latched, and used for the flip, the lines staged and the
    /// resolve.
    displayed: u32,
    pub stats: DirtyStats,
    swaps_seen: u64,
    /// The registers of the frame in progress, copied after the marks were
    /// taken.
    snap: DisplayRegs,
    snap_gen: u64,
    stage: ScanoutStage,
    rows: Vec<u32>,
    dirty: Vec<u32>,
    /// The last frame's update, kept to take its buffers back at the start
    /// of the next.
    last: Option<ScanoutJob>,
    scratch: [Vec<u8>; 3],
}

impl ScanoutReader {
    /// Read the board `shared` belongs to.
    pub fn attach(&mut self, shared: &Arc<DisplayShared>) {
        if self.shared.as_ref().is_some_and(|s| Arc::ptr_eq(s, shared)) {
            return;
        }
        self.shared = Some(shared.clone());
        self.mode_gen = 0;
        self.restart_gen = 0;
        self.snap_gen = u64::MAX;
        self.frame = Vec::new();
        self.frame_width = 0;
        self.frame_height = 0;
        self.staged_displayed = 0;
        *self.differ = [[0; ROW_WORDS]; ODSY_DISPMODE_COUNT as usize];
    }

    fn shared(&self) -> Arc<DisplayShared> {
        self.shared.clone().expect("a reader reads a board it is attached to")
    }

    /// Take a new mode and retrace restart the front end published.
    fn sync(&mut self, shared: &DisplayShared) {
        let (mode, gen, restart) = {
            let cell = shared.mode.lock();
            (cell.mode, cell.gen, cell.restart)
        };

        if gen != self.mode_gen {
            self.mode_gen = gen;
            self.clock.mode = mode;
            let (w, h) = (mode.width, mode.height);
            if w != self.frame_width || h != self.frame_height {
                self.frame = Vec::new();
                self.frame_width = 0;
                self.frame_height = 0;
                if w != 0 && h != 0 {
                    self.frame = vec![0u32; w as usize * h as usize];
                    self.frame_width = w;
                    self.frame_height = h;
                }
            }
            self.reshaped = true;
        }
        if restart != self.restart_gen {
            self.restart_gen = restart;
            self.clock.restart();
        }
    }

    /// Every retrace due at `now_ns`, in turn: the host interface latches
    /// the swaps armed and raises the retrace source.
    pub fn poll(&mut self, now_ns: i64) {
        let shared = self.shared();

        self.sync(&shared);
        while self.clock.next_retrace(now_ns) {
            shared.retrace();
        }
    }

    /// The registers as of now, copied when the front end has stored to
    /// them since the last copy.
    fn snapshot(&mut self, shared: &DisplayShared) {
        let regs = shared.regs.read();

        if regs.gen != self.snap_gen {
            self.snap.clone_from(&regs);
            self.snap_gen = regs.gen;
        }
    }

    /// The rows the raster unit published since the last call: the rows
    /// where a window's buffers may differ, and those it shows, to resolve.
    pub fn take_published(&mut self, displayed: u32) {
        let shared = self.shared();

        shared.raster.take_published_rows(displayed, &mut self.differ, &shared.marks);
    }

    /// A frame's update up to the plane reads: take the rows published and
    /// the marks, copy each marked line's plane bytes aside, so that the
    /// resolve itself (`take()`, `ScanoutJob::resolve()`) can run while the
    /// writers go on. A line of an unusual plane shape is resolved at once,
    /// pixel by pixel. A processor's write of the displayed-buffer register
    /// reached only the front: the windows whose buffer it changed repaint
    /// the rows their buffers differ in.
    ///
    /// The marks are taken before the planes are read and the registers
    /// copied after (a lookup store changes the table and then marks every
    /// line), and the displayed bits are read once.
    pub fn stage(&mut self) {
        let shared = self.shared();

        self.sync(&shared);
        let displayed = {
            let mut f = shared.front.lock();
            f.displayed_written = false;
            f.displayed_buffer
        };
        self.displayed = displayed;
        let swaps = shared.swaps.load(Ordering::Relaxed);
        self.stats.swaps += swaps.wrapping_sub(std::mem::replace(&mut self.swaps_seen, swaps));
        self.take_published(displayed);
        let flipped = std::mem::replace(&mut self.staged_displayed, displayed) ^ displayed;
        if flipped != 0 && !self.frame.is_empty() {
            self.flip_rows(&shared, flipped);
        }
        let mut stage = std::mem::take(&mut self.stage);
        stage.lines.clear();
        stage.bytes.clear();
        stage.be.clear();
        if !self.frame.is_empty() {
            let h = self.clock.mode.height as usize;
            let w = self.clock.mode.width as usize;
            let width = self.clock.mode.width.min(SGI_ODYSSEY_WIDTH) as usize;
            let mut dirty = std::mem::take(&mut self.dirty);

            dirty.clear();
            shared.marks.take_dirty(h, &mut dirty);
            if std::mem::take(&mut self.reshaped) {
                dirty.clear();
                dirty.extend(0..h as u32);
            }
            self.snapshot(&shared);
            let view = shared.planes.get();
            let mut row = Vec::new();
            for &y in &dirty {
                let staged = view.is_some_and(|view| stage.push_row(view, y, width));
                if !staged {
                    row.resize(w, 0);
                    if let Some(view) = view {
                        if view.resolve_generic(&self.snap, displayed, y, width as u32, &mut row) {
                            shared.report_group25();
                        }
                    }
                    self.frame[y as usize * w..(y as usize + 1) * w].copy_from_slice(&row);
                }
                stage.lines.push((y, staged));
            }
            self.dirty = dirty;
        }
        self.stats.staged(stage.lines.len());
        self.stage = stage;
    }

    /// Repaint what a flip of the displayed buffers `flipped` (a bit per
    /// main drawable id) changes: the rows where that window's two buffers
    /// differ. A row where they are equal shows the same pixels either way.
    fn flip_rows(&mut self, shared: &DisplayShared, flipped: u32) {
        if self.clock.mode.height > SGI_ODYSSEY_HEIGHT {
            return shared.marks.repaint(DirtySite::Flip, 0, self.clock.mode.height as i32);
        }
        if shared.marks.unnamed_pending() {
            self.measure_differ(shared);
        }
        let mut rows = [0u64; ROW_WORDS];
        for (id, differ) in self.differ.iter().enumerate() {
            if (flipped >> id) & 1 != 0 {
                for (r, d) in rows.iter_mut().zip(differ) {
                    *r |= d;
                }
            }
        }
        for (i, mut w) in rows.into_iter().enumerate() {
            while w != 0 {
                shared.marks.repaint(DirtySite::Flip, (i as u32 * 64 + w.trailing_zeros()) as i32, 1);
                w &= w - 1;
            }
        }
    }

    /// Find, in the lines a writer did not name its buffer or drawable for
    /// (RowMarks::unnamed), the rows where each main drawable's pixels
    /// differ between the two colour buffers. A row the planes do not cover
    /// whole differs for every id.
    ///
    /// The marks are taken first (a write that lands after sets its line
    /// again, for the next swap), the rows are copied, and the result is
    /// set in `differ`, which only the display writes.
    fn measure_differ(&mut self, shared: &DisplayShared) {
        let width = self.clock.mode.width.min(SGI_ODYSSEY_WIDTH) as usize;
        let unnamed = shared.marks.take_unnamed();
        let [l0, l1, did] = &mut self.scratch;

        l0.resize(width * 4, 0);
        l1.resize(width * 4, 0);
        did.resize(width, 0);
        for (i, mut word) in unnamed.into_iter().enumerate() {
            while word != 0 {
                let y = (i * 64 + word.trailing_zeros() as usize) as u32;
                word &= word - 1;
                if y >= self.clock.mode.height {
                    continue;
                }
                let bit = (y as usize >> 6, 1u64 << (y & 63));
                let planes = shared.planes.get().and_then(|view| {
                    let be0 = view.copy_row(RowOf::Colour(0), y, l0)?;
                    let be1 = view.copy_row(RowOf::Colour(1), y, l1)?;
                    view.copy_row(RowOf::Did, y, did)?;
                    (be0 == be1).then_some(())
                });
                // The ids whose pixels differ in this row.
                let mut ids = 0u32;

                match planes {
                    None => ids = u32::MAX,
                    Some(()) if l0 != l1 => {
                        let mut x = 0;
                        while x < width {
                            let d = did[x];
                            let end = x + 1 + did[x + 1..].iter().take_while(|&&v| v == d).count();
                            if l0[x * 4..end * 4] != l1[x * 4..end * 4] {
                                ids |= 1 << ((d >> ODSY_DID_MAIN_SHIFT) as usize & (ODSY_DISPMODE_COUNT as usize - 1));
                            }
                            x = end;
                        }
                    }
                    Some(()) => {}
                }
                for (id, d) in self.differ.iter_mut().enumerate() {
                    d[bit.0] = d[bit.0] & !bit.1 | if (ids >> id) & 1 != 0 { bit.1 } else { 0 };
                }
            }
        }
    }

    /// The staged lines and everything their resolve reads outside the
    /// planes, to resolve (ScanoutJob::resolve()). None when no line was
    /// dirty.
    pub fn take(&mut self) -> Option<ScanoutJob> {
        if self.stage.lines.is_empty() {
            return None;
        }
        let stage = std::mem::take(&mut self.stage);
        let ctx = stage.lines.iter().any(|&(_, staged)| staged).then(|| ResolveCtx::new(&self.snap, self.displayed));
        Some(ScanoutJob {
            stage,
            ctx,
            w: self.clock.mode.width as usize,
            width: self.clock.mode.width.min(SGI_ODYSSEY_WIDTH),
            frame: (self.frame_width, self.frame_height),
            rows: std::mem::take(&mut self.rows),
            saw_unverified: false,
            committed: false,
        })
    }

    /// Put a resolved job's lines into the presented frame, with the cursor
    /// over them. If the frame changed shape meanwhile the lines are dropped
    /// and the whole new frame is dirty. The span of lines changed. Afterwards
    /// the job's rows (ScanoutJob::lines()) are the presented lines, for the
    /// display to convert; hand the job to `finish()` when done.
    pub fn commit(&mut self, job: &mut ScanoutJob) -> Option<(usize, usize)> {
        let shared = self.shared();
        let mut span: Option<(usize, usize)> = None;
        job.committed = false;
        if job.w == 0 || job.frame != (self.frame_width, self.frame_height) || job.w != self.clock.mode.width as usize {
            shared.marks.repaint(DirtySite::Reshape, 0, self.clock.mode.height as i32);
            return None;
        }
        let w = job.w;
        job.rows.resize(job.stage.lines.len() * w, 0);
        for (&(y, staged), row) in job.stage.lines.iter().zip(job.rows.chunks_exact_mut(w)) {
            let y = y as usize;
            if staged {
                self.snap.cursor_row(y as u32, row, job.width);
                self.frame[y * w..(y + 1) * w].copy_from_slice(row);
            } else {
                // Resolved into the frame when it was staged.
                row.copy_from_slice(&self.frame[y * w..(y + 1) * w]);
            }
            span = Some(span.map_or((y, y + 1), |(a, b)| (a.min(y), b.max(y + 1))));
        }
        if job.saw_unverified {
            shared.report_group25();
        }
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: scanout update: {} lines", job.stage.lines.len());
        job.committed = true;
        span
    }

    /// Take a finished job's buffers back for the next frame.
    pub fn recycle(&mut self, job: ScanoutJob) {
        self.stage = job.stage;
        self.rows = job.rows;
    }

    /// Keep the frame's job until the next frame, then take its buffers back.
    pub fn keep(&mut self, job: Option<ScanoutJob>) {
        self.last = job;
    }

    /// The previous frame's job, back to the reader.
    pub fn recycle_last(&mut self) {
        if let Some(done) = self.last.take() {
            self.recycle(done);
        }
    }

    /// The whole update for a caller that holds the planes throughout:
    /// stage, resolve and commit. The span of lines changed.
    pub fn update(&mut self) -> Option<(usize, usize)> {
        self.stage();
        self.resolve()
    }

    /// The rest of an update after stage(): resolve the staged lines into the
    /// presented frame.
    pub fn resolve(&mut self) -> Option<(usize, usize)> {
        let mut job = self.take();
        if let Some(job) = &mut job {
            job.resolve();
        }
        let mut job = job?;
        let span = self.commit(&mut job);
        self.recycle(job);
        span
    }
}
