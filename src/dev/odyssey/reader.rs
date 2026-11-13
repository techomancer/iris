//! What the display thread reads of the board, and nothing else.
//!
//! When anything it reads has changed, the display composes the whole
//! screen from board memory as the scan-out registers describe it
//! (frame.rs), reading memory the raster unit and the front end may be
//! writing, without a lock, as the real scan-out does. A store to the
//! memory (`VramPtr::take_written`) or a move of a register generation
//! below marks the next frame; with neither, the frame on show stays.
//! The rest of what a frame reads is shared the same way:
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
    #[cfg(test)]
    pub fn cursor_row(&self, y: u32, dst: &mut [u32], width: u32) {
        self.cursor_row_in(y, dst, width, false);
    }

    /// As `cursor_row`, in the screen's order when `abgr` (`frame::to_screen`).
    pub fn cursor_row_in(&self, y: u32, dst: &mut [u32], width: u32, abgr: bool) {
        let cursor_rgb = |w: u32| if abgr { super::frame::to_screen(cursor_rgb(w)) } else { cursor_rgb(w) };
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

    /// Scan-out entry 0x2800 + index (frame::pbj).
    pub fn pbj(&self, index: usize) -> u32 {
        self.pbj[index].load(Relaxed)
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
    /// The scan-out on the GPU, set with the GPU raster unit: the display
    /// composes there from the memory the unit draws into, instead of
    /// from the mirror.
    pub gpu_scanout: OnceLock<Mutex<super::gpu_unit::GpuScanout>>,
    /// The command stream position (`CommandFifo::retired`) the raster
    /// unit's submitted work covers: a flip armed after it waits
    /// (`HostInterface::retrace`). The GPU unit publishes it
    /// (`GpuRaster::tick`); the CPU unit leaves it at the maximum, so flips
    /// latch at once as before.
    pub covered: Arc<AtomicU64>,
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
            gpu_scanout: OnceLock::new(),
            covered: Arc::new(AtomicU64::new(u64::MAX)),
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
        if self.front.lock().retrace(self.covered.load(Ordering::Acquire)) {
            self.swaps.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Where `ScanoutReader::update_to` put a frame.
pub(super) enum Composed {
    /// Nothing changed: the last frame stands.
    No,
    /// In `frame` (the GPU scan-out's readback, or no target given).
    InFrame,
    /// In the target, in the order `screen_order` names.
    InTarget,
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
    /// What `frame` was composed from: the mode, the registers and the
    /// displayed-buffer bits. None before the first.
    composed: Option<(u64, u64, u32)>,
    /// The last frame was composed for a store to board memory. The store's
    /// mark is not ordered against its data, so the frame may have missed
    /// the data; one more is composed to catch it.
    settling: bool,
    /// The GPU scan-out failed once (reported), and the CPU composes.
    gpu_failed: bool,
    /// Compose on the CPU in the screen's order, 0xAABBGGRR
    /// (`Frame::snapshot_for_screen`): the display thread's choice, so its
    /// frames go to the screen with no conversion. Tests read 0x00RRGGBB.
    pub screen_order: bool,
    /// Whether `frame` holds the screen's order (else 0x00RRGGBB, as the
    /// GPU scan-out's readback gives it).
    pub frame_abgr: bool,
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
        self.composed = None;
        self.settling = false;
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

    /// As `update`, on the GPU scan-out with no readback: the frame stays
    /// in the scan-out's buffer, returned with its size, for a renderer on
    /// the same device. None when there is no GPU scan-out, no frame, or it
    /// fails (the caller then takes the `update` path).
    #[cfg(feature = "wgpu")]
    pub fn update_on_gpu(&mut self) -> Option<(wgpu::Buffer, u32, u32)> {
        let shared = self.shared();
        let scanout = shared.gpu_scanout.get()?;
        if self.gpu_failed {
            return None;
        }
        self.sync(&shared);
        if self.frame.is_empty() {
            return None;
        }
        let displayed = shared.front.lock().displayed_buffer;
        self.snap_gen = shared.regs.snapshot(&mut self.snap, self.snap_gen);
        let frame = Frame::snapshot(&self.snap, displayed)?;
        let (w, h) = (frame.width.min(self.frame_width), frame.height.min(self.frame_height));
        let composed = scanout.lock().compose(&frame, &self.snap, w, h);
        match composed {
            Ok(buffer) => Some((buffer, w, h)),
            Err(e) => {
                self.gpu_failed = true;
                crate::dlog!(crate::devlog::LogModule::Odyssey, "sgi-odyssey: gpu scanout: {e}; composing on the CPU");
                None
            }
        }
    }

    /// Compose the frame again if anything the scan-out reads has changed
    /// since the last one: a store to board memory, a register (lookup RAM,
    /// display modes, cursor, scan-out geometry), the mode, or the
    /// displayed-buffer bits. It is composed from board memory as the
    /// scan-out registers describe it (frame.rs); black while they describe
    /// none. False when `frame` is as it was, or there is no scanout mode
    /// and so no frame.
    ///
    /// A frame composed for a store is followed by one more, since a store's
    /// mark can be seen before its data.
    ///
    /// The scan-out on the GPU draws into memory with no store the CPU
    /// sees, so there every frame is composed.
    #[cfg(test)]
    pub fn update(&mut self) -> bool {
        !matches!(self.update_to(None), Composed::No)
    }

    /// As `update`, composing on the CPU into `target` (rows of `stride`,
    /// in the order `screen_order` names) instead of `frame`: the display
    /// thread's frames go straight into the screen. The GPU scan-out's
    /// readback still lands in `frame`, as 0x00RRGGBB.
    pub fn update_to(&mut self, target: Option<(&mut [u32], usize)>) -> Composed {
        let shared = self.shared();
        // Taken before anything is read: a store that lands from here on
        // marks the next frame.
        let stored = shared.vram.get().is_none_or(|mem| mem.take_written());

        self.sync(&shared);
        if self.frame.is_empty() {
            return Composed::No;
        }
        let displayed = shared.front.lock().displayed_buffer;
        self.snap_gen = shared.regs.snapshot(&mut self.snap, self.snap_gen);
        let now = Some((self.mode_gen, self.snap_gen, displayed));
        if !stored && !self.settling && self.composed == now && shared.gpu_scanout.get().is_none() {
            return Composed::No;
        }
        self.composed = now;
        self.settling = stored;
        let gpu = shared.gpu_scanout.get().filter(|_| !self.gpu_failed);
        let (w, h) = (self.frame_width as usize, self.frame_height as usize);
        let snapshot = if self.screen_order { Frame::snapshot_for_screen(&self.snap, displayed) } else { Frame::snapshot(&self.snap, displayed) };
        let (Some(frame), Some(mem)) = (snapshot, shared.vram.get()) else {
            // No scan-out programmed (out of reset, before the PROM sets
            // it up): nothing is shown.
            let black = if self.screen_order { super::frame::to_screen(0) } else { 0 };
            return match target {
                Some((out, stride)) => {
                    for row in out.chunks_mut(stride).take(h) {
                        row[..w.min(stride)].fill(black);
                    }
                    Composed::InTarget
                }
                None => {
                    self.frame.fill(black);
                    self.frame_abgr = self.screen_order;
                    Composed::InFrame
                }
            };
        };
        if let Some(scanout) = gpu {
            let (fw, fh) = (frame.width.min(self.frame_width), frame.height.min(self.frame_height));
            // The kernel writes rows of fw; a narrower frame is spread over
            // the presented rows after.
            match scanout.lock().run(&frame, &self.snap, fw, fh, &mut self.frame) {
                Ok(()) => {
                    if fw as usize != w {
                        for y in (1..fh as usize).rev() {
                            self.frame.copy_within(y * fw as usize..(y + 1) * fw as usize, y * w);
                        }
                    }
                    self.frame_abgr = false;
                    return Composed::InFrame;
                }
                Err(e) => {
                    if !self.gpu_failed {
                        self.gpu_failed = true;
                        crate::dlog!(crate::devlog::LogModule::Odyssey, "sgi-odyssey: gpu scanout: {e}; composing on the CPU");
                    }
                }
            }
        }
        // The memory lives as long as the board; the display reads it while
        // the board's units write it, as the scan-out does.
        match target {
            Some((out, stride)) => {
                let rows = h.min(out.len() / stride.max(1));
                frame.compose(*mem, &self.snap, out, stride, rows);
                Composed::InTarget
            }
            None => {
                frame.compose(*mem, &self.snap, &mut self.frame, w, h);
                self.frame_abgr = frame.is_screen_order();
                Composed::InFrame
            }
        }
    }
}
