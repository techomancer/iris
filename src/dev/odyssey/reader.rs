//! What the display thread reads of the board, and nothing else.
//!
//! Each frame the display composes the whole screen from board memory as
//! the scan-out registers describe it (frame.rs), reading memory the
//! raster unit and the front end may be writing, without a lock, as the
//! real scan-out does. The rest of what a frame reads is shared the same
//! way:
//!
//! - the lookup RAM, the display modes, the cursor and the scan-out
//!   geometry (`SharedRegs`), one atomic word an entry, with a generation
//!   that moves after each store;
//! - the scanout mode and the retrace restart (`DisplayShared::publish_mode`);
//! - the swap latch and the displayed-buffer bits, which are the host
//!   interface's (`HostInterface::retrace`): the buffer swap is the
//!   hardware's own synchronisation point, so it keeps its register lock.
//!
//! The display never calls a `Board` method. Board memory is fixed at
//! `Board::allocate`, which hands the display its pointer
//! (`DisplayShared::vram`).

use super::consts::{
    cursor, dispmode, lut, ODSY_GAMMA_BASE,
};
use super::frame::Frame;
use super::vram::VramPtr;
use super::interface::HostInterface;
use super::registers::cursor_rgb;
use super::scanout::{RetraceClock, ScanoutMode};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, Ordering::{Acquire, Relaxed, Release}};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

/// The back end's registers the scanout resolves through: the colour
/// lookup RAM, the per-drawable display modes, the overlay modes and the
/// hardware cursor.
#[derive(Clone)]
pub(super) struct DisplayRegs {
    /// The colour lookup RAM (0x2000 entries), one 30-bit colour per entry.
    pub lut: Vec<u32>,
    pub dispmode: [u32; 32],
    /// The overlay ids' display modes, backend entries 0x2920 + id.
    pub overlay_mode: [u32; 8],
    /// The cursor's backend entries 0x2500-0x25ff.
    pub cursor: [u32; 0x100],
    /// The scan-out geometry, backend entries 0x2800-0x280f (frame::pbj).
    pub pbj: [u32; 16],
}

impl Default for DisplayRegs {
    fn default() -> Self {
        DisplayRegs { lut: vec![0; lut::SIZE as usize], dispmode: [0; 32], overlay_mode: [0; 8], cursor: [0; 0x100], pbj: [0; 16] }
    }
}

impl DisplayRegs {

    /// The top eight bits of a lookup-RAM entry's ten-bit channels, as RGB.
    pub fn lut_rgb(&self, entry: u32) -> u32 {
        let word = if entry < lut::SIZE { self.lut[entry as usize] } else { 0 };

        (((word >> lut::RED_SHIFT) & 0xff) << 16) | (((word >> lut::GREEN_SHIFT) & 0xff) << 8) | ((word >> lut::BLUE_SHIFT) & 0xff)
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
        (self.dispmode[main_id as usize] >> dispmode::GROUP_SHIFT) & dispmode::GROUP_MASK
    }

    /// The position's components, each a signed 16-bit value (inferred).
    pub fn cursor_x(&self) -> i32 {
        (self.cursor[cursor::POSITION as usize] >> 16) as i16 as i32
    }

    pub fn cursor_y(&self) -> i32 {
        self.cursor[cursor::POSITION as usize] as i16 as i32
    }

    /// Whether anything is drawn: bit 0 set and bit 5 clear. APPROXIMATION: on
    /// the second channel the cursor is on a display the model does not have.
    pub fn cursor_shown(&self) -> bool {
        let ctl = self.cursor[cursor::CONTROL as usize];

        (ctl & cursor::CTL_ENABLE) != 0 && (ctl & cursor::CTL_CHANNEL1) == 0
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
        if self.cursor[cursor::CONTROL as usize] & cursor::CTL_CROSSHAIR != 0 {
            let colour = cursor_rgb(self.cursor[cursor::COLOURS as usize + 1]);

            if row == 0 {
                dst[..width as usize].fill(colour);
            } else if cx >= 0 && cx < width as i32 {
                dst[cx as usize] = colour;
            }
            return;
        }
        if row < 0 || row >= cursor::SIZE as i32 {
            return;
        }
        for col in 0..cursor::SIZE as i32 {
            let x = cx.wrapping_add(col);
            let plane = (self.cursor[(cursor::PLANE as i32 + row * 2 + col / 16) as usize] >> ((col % 16) * 2)) & 3;
            let glyph = (self.cursor[(cursor::GLYPH as i32 + row * 4 + col / 8) as usize] >> ((col % 8) * 4)) & 0xf;

            if x < 0 || x >= width as i32 || plane != 3 || glyph == 0xf {
                continue;
            }
            dst[x as usize] = cursor_rgb(self.cursor[cursor::COLOURS as usize + glyph as usize]);
        }
    }
}

/// The back end's registers as the front end stores them and the display
/// reads them: one atomic word per entry, no lock. The board's back end
/// reads its lookup RAM and cursor registers while the front end writes
/// them, so a frame that races a store shows the old or the new entry; the
/// store marks every line it changes afterwards, and the next frame
/// resolves them again.
///
/// `gen` moves after each store. The display reads it before it copies
/// ([`SharedRegs::snapshot`]), so a store that lands during the copy leaves
/// the generation ahead of the one the copy is filed under, and the next
/// frame copies again.
pub(super) struct SharedRegs {
    lut: Box<[AtomicU32]>,
    dispmode: [AtomicU32; 32],
    overlay_mode: [AtomicU32; 8],
    cursor: [AtomicU32; 0x100],
    pbj: [AtomicU32; 16],
    gen: AtomicU64,
    /// Tests: run once the copy is done, where a store can land that the
    /// copy missed.
    #[cfg(test)]
    pub test_after_copy: Mutex<Option<Box<dyn FnMut(&SharedRegs) + Send>>>,
}

impl Default for SharedRegs {
    fn default() -> Self {
        SharedRegs {
            lut: (0..lut::SIZE).map(|_| AtomicU32::new(0)).collect(),
            dispmode: std::array::from_fn(|_| AtomicU32::new(0)),
            overlay_mode: std::array::from_fn(|_| AtomicU32::new(0)),
            cursor: std::array::from_fn(|_| AtomicU32::new(0)),
            pbj: std::array::from_fn(|_| AtomicU32::new(0)),
            gen: AtomicU64::new(0),
            #[cfg(test)]
            test_after_copy: Mutex::new(None),
        }
    }
}

impl SharedRegs {
    /// Store a register. Mark the lines the store changes afterwards.
    pub fn set_lut(&self, index: usize, value: u32) {
        self.lut[index].store(value, Relaxed);
        self.gen.fetch_add(1, Release);
    }

    /// Display mode word `index` (0x2900 + id).
    pub fn dispmode(&self, index: usize) -> u32 {
        self.dispmode[index & 31].load(Relaxed)
    }

    pub fn set_dispmode(&self, index: usize, value: u32) {
        self.dispmode[index].store(value, Relaxed);
        self.gen.fetch_add(1, Release);
    }

    pub fn set_overlay_mode(&self, index: usize, value: u32) {
        self.overlay_mode[index].store(value, Relaxed);
        self.gen.fetch_add(1, Release);
    }

    pub fn set_pbj(&self, index: usize, value: u32) {
        self.pbj[index].store(value, Relaxed);
        self.gen.fetch_add(1, Release);
    }

    pub fn set_cursor(&self, index: usize, value: u32) {
        self.cursor[index].store(value, Relaxed);
        self.gen.fetch_add(1, Release);
    }

    /// Back to the reset values (DisplayRegs::reset()).
    pub fn reset(&self) {
        for w in self.lut.iter().chain(&self.dispmode).chain(&self.overlay_mode).chain(&self.cursor).chain(&self.pbj) {
            w.store(0, Relaxed);
        }
        self.gen.fetch_add(1, Release);
    }

    pub fn cursor(&self, index: usize) -> u32 {
        self.cursor[index].load(Relaxed)
    }

    /// The glyph the display draws, for the host's pointer steering
    /// (`GfxDisplay::hw_cursor`): where it sits and which pixels are
    /// opaque, by the same plane and glyph rules `cursor_row` draws with.
    /// None while the cursor is off, on the second channel, or a crosshair,
    /// which has no glyph.
    pub fn hw_cursor(&self) -> Option<crate::gfx_display::HwCursor> {
        let control = self.cursor(cursor::CONTROL as usize);

        if (control & cursor::CTL_ENABLE) == 0 || (control & (cursor::CTL_CHANNEL1 | cursor::CTL_CROSSHAIR)) != 0 {
            return None;
        }
        let position = self.cursor(cursor::POSITION as usize);
        let mask = (0..cursor::SIZE as usize)
            .map(|row| {
                (0..cursor::SIZE as usize).fold(0u64, |m, col| {
                    let plane = (self.cursor(cursor::PLANE as usize + row * 2 + col / 16) >> ((col % 16) * 2)) & 3;
                    let glyph = (self.cursor(cursor::GLYPH as usize + row * 4 + col / 8) >> ((col % 8) * 4)) & 0xf;

                    m | u64::from(plane == 3 && glyph != 0xf) << col
                })
            })
            .collect();

        Some(crate::gfx_display::HwCursor {
            x: (position >> 16) as i16 as i32,
            y: position as i16 as i32,
            size: cursor::SIZE as usize,
            mask,
        })
    }

    /// The colormap group main drawable `main_id` shows through
    /// (DisplayRegs::display_group()).
    pub fn display_group(&self, main_id: u32) -> u32 {
        (self.dispmode[main_id as usize].load(Relaxed) >> dispmode::GROUP_SHIFT) & dispmode::GROUP_MASK
    }

    /// Copy the registers into `out` when a store has landed since `seen`
    /// (a generation `snapshot` returned), and return the generation the
    /// copy is filed under: the one read before the copy began.
    pub fn snapshot(&self, out: &mut DisplayRegs, seen: u64) -> u64 {
        let gen = self.gen.load(Acquire);

        if gen != seen {
            for (o, w) in out.lut.iter_mut().zip(self.lut.iter()) {
                *o = w.load(Relaxed);
            }
            for (o, w) in out.dispmode.iter_mut().zip(&self.dispmode) {
                *o = w.load(Relaxed);
            }
            for (o, w) in out.overlay_mode.iter_mut().zip(&self.overlay_mode) {
                *o = w.load(Relaxed);
            }
            for (o, w) in out.cursor.iter_mut().zip(&self.cursor) {
                *o = w.load(Relaxed);
            }
            for (o, w) in out.pbj.iter_mut().zip(&self.pbj) {
                *o = w.load(Relaxed);
            }
            #[cfg(test)]
            if let Some(hook) = self.test_after_copy.lock().as_mut() {
                hook(self);
            }
        }
        gen
    }

    /// The registers as they are now: for the tests and the benchmark,
    /// which resolve on a board they hold, and for the monitor's pixel
    /// decode and screen dumps.
    pub fn read(&self) -> DisplayRegs {
        let mut regs = DisplayRegs::default();

        self.snapshot(&mut regs, u64::MAX);
        regs
    }
}

/// The scanout mode as the front end last set it, and the counts a display
/// compares to notice a new one. The fields are separate atomic words: the
/// counts move after the mode is stored and the display reads them before
/// it reads the mode, so a mode read mid-store is read again at the next
/// frame.
struct ModeCell {
    width: AtomicU32,
    height: AtomicU32,
    frame_clocks: AtomicU64,
    clock_hz: AtomicU32,
    /// Moves when the mode is set.
    gen: AtomicU64,
    /// Moves when the retrace phase restarts: a new mode or a reset.
    restart: AtomicU64,
}

impl ModeCell {
    fn new() -> Self {
        ModeCell {
            width: AtomicU32::new(0),
            height: AtomicU32::new(0),
            frame_clocks: AtomicU64::new(0),
            clock_hz: AtomicU32::new(0),
            gen: AtomicU64::new(1),
            restart: AtomicU64::new(1),
        }
    }
}

/// What the board shares with the display thread.
pub(super) struct DisplayShared {
    pub regs: SharedRegs,
    mode: ModeCell,
    /// The board's memory, set when the board allocates it.
    pub vram: OnceLock<VramPtr>,
    pub front: Arc<Mutex<HostInterface>>,
    /// Swaps latched at a retrace.
    pub swaps: AtomicU64,
}

impl DisplayShared {
    pub fn new(front: Arc<Mutex<HostInterface>>) -> Self {
        DisplayShared {
            regs: SharedRegs::default(),
            mode: ModeCell::new(),
            vram: OnceLock::new(),
            front,
            swaps: AtomicU64::new(0),
        }
    }

    /// A new mode: the display reallocates its frame and restarts the
    /// retrace phase when it next looks.
    pub fn publish_mode(&self, mode: ScanoutMode) {
        let cell = &self.mode;

        cell.width.store(mode.width, Relaxed);
        cell.height.store(mode.height, Relaxed);
        cell.frame_clocks.store(mode.frame_clocks, Relaxed);
        cell.clock_hz.store(mode.clock_hz, Relaxed);
        cell.restart.fetch_add(1, Release);
        cell.gen.fetch_add(1, Release);
    }

    /// The retrace phase restarts, as for a reset.
    pub fn restart_retrace(&self) {
        self.mode.restart.fetch_add(1, Release);
    }

    /// A vertical retrace: the host interface latches the swaps armed and
    /// raises the retrace source (HostInterface::retrace()).
    pub fn retrace(&self) {
        if self.front.lock().retrace() {
            self.swaps.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// The display's own state: the presented frame and the retrace clock.
/// Only the display thread (or a test standing in for it) uses it.
#[derive(Default)]
pub(super) struct ScanoutReader {
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
    /// The registers of the frame in progress.
    snap: DisplayRegs,
    snap_gen: u64,
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
    }

    fn shared(&self) -> Arc<DisplayShared> {
        self.shared.clone().expect("a reader reads a board it is attached to")
    }

    /// Take a new mode and retrace restart the front end published.
    fn sync(&mut self, shared: &DisplayShared) {
        let cell = &shared.mode;
        let (gen, restart) = (cell.gen.load(Acquire), cell.restart.load(Acquire));
        let mode = ScanoutMode {
            width: cell.width.load(Relaxed),
            height: cell.height.load(Relaxed),
            frame_clocks: cell.frame_clocks.load(Relaxed),
            clock_hz: cell.clock_hz.load(Relaxed),
        };

        if gen != self.mode_gen {
            self.mode_gen = gen;
            self.clock.mode = mode;
            let (w, h) = (mode.width, mode.height);
            if w != self.frame_width || h != self.frame_height {
                self.frame = vec![0u32; w as usize * h as usize];
                self.frame_width = w;
                self.frame_height = h;
            }
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

    /// Compose the whole frame from board memory as the scan-out registers
    /// describe it (frame.rs); black while they describe none. False with
    /// no scanout mode, when there is no frame.
    pub fn update(&mut self) -> bool {
        let shared = self.shared();

        self.sync(&shared);
        if self.frame.is_empty() {
            return false;
        }
        let displayed = shared.front.lock().displayed_buffer;
        self.snap_gen = shared.regs.snapshot(&mut self.snap, self.snap_gen);
        let (Some(frame), Some(mem)) = (Frame::snapshot(&self.snap, displayed), shared.vram.get()) else {
            // No scan-out programmed (out of reset, before the PROM sets
            // it up): nothing is shown.
            self.frame.fill(0);
            return true;
        };
        // The memory lives as long as the board; the display reads it while
        // the board's units write it, as the scan-out does.
        frame.compose(*mem, &self.snap, &mut self.frame, self.frame_width as usize, self.frame_height as usize);
        true
    }
}
