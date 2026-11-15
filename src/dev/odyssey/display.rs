//! What the board shows: the scanout's colour lookup and gamma, the retrace
//! with its swap latching, and the window iris presents the frame in.

use super::pipe::backend::DestMem;
use super::board::Board;
use super::consts::{
    dispmode, lut, overlay, ODSY_CMAP_UNVERIFIED_GROUP, ODSY_DID_MAIN_SHIFT,
    ODSY_GAMMA_GROUP, ODSY_INDEX_MASK,
};
use super::reader::{DisplayRegs, DisplayShared, ScanoutReader};
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::cpu::mips_core::CyclesPtr;
use crate::dev::ng1::rex3::Renderer;
use crate::disp::{Rex3Screen, FB_STRIDE};
use crate::display::{Composed, Head, HeadHandle, Want};

/// The window the board shows the guest: what `GfxDisplay` hands out.
pub(super) struct Display {
    /// The display channel: screen, renderer slot and frame requests, run
    /// by the machine's refresh thread (`crate::display`).
    pub head: Arc<HeadHandle>,
    /// What the display keeps of the scanout: the presented frame and the
    /// update in progress. Held for each frame.
    pub reader: Mutex<ScanoutReader>,
    /// The machine's status-bar inputs.
    status: Arc<crate::disp::StatusSource>,
    /// The registers the display reads, for the host's cursor query; set
    /// when the board hands the display out.
    shared: OnceLock<Arc<DisplayShared>>,
    verify: Mutex<crate::display::verify::Check>,
}

impl Display {
    pub(super) fn new(status: Arc<crate::disp::StatusSource>) -> Self {
        Self {
            head: status.refresh().head(status.clone()),
            reader: Mutex::new(ScanoutReader::default()),
            status,
            shared: OnceLock::new(),
            verify: Mutex::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn screen(&self) -> &Mutex<Rex3Screen> {
        self.head.screen()
    }

    #[cfg(test)]
    pub(super) fn renderer(&self) -> &Mutex<Option<Box<dyn Renderer>>> {
        self.head.renderer_slot()
    }

    /// Read the cursor from `shared` from now on.
    pub(super) fn attach_registers(&self, shared: Arc<DisplayShared>) {
        let _ = self.shared.set(shared);
    }

    /// If anything the scan-out reads has changed, the whole frame composed
    /// from board memory as the scan-out registers describe it (frame.rs).
    /// On the GPU raster unit with the wgpu renderer on the same device it
    /// is composed there and stays there; otherwise it is composed on the
    /// CPU straight into the screen, in its order (the GPU scan-out's
    /// readback comes back in the reader's `frame` and is converted).
    /// The display takes no board lock and waits for no drawing: it reads
    /// memory while the raster unit and the front end write it, as the
    /// board's own scan-out does.
    pub(super) fn compose(&self, shared: &Arc<DisplayShared>, screen: &mut Rex3Screen, want: Want) -> Composed {
        let mut reader = self.reader.lock();
        reader.attach(shared);
        crate::pf_begin!(crate::perfetto::events::ODSY_FRAME_PLANES);
        #[cfg(feature = "wgpu")]
        if want.gpu {
            if let Some((frame, width, height)) = reader.update_on_gpu() {
                crate::pf_end!(crate::perfetto::events::ODSY_FRAME_PLANES);
                screen.width = (width as usize).min(FB_STRIDE);
                screen.height = height as usize;
                screen.prebuilt = true;
                return Composed::Gpu { frame, width, height };
            }
        }
        let _ = want;
        reader.screen_order = true;
        // Checking the unchanged verdicts (`display::verify`): compose a
        // frame the reader would leave, and compare.
        let verify = crate::display::verify::on() && screen.prebuilt && screen.width != 0;
        if verify {
            self.verify.lock().before(screen);
        }
        reader.force = verify;
        let composed = reader.update_to(Some((&mut screen.fb_rgb, FB_STRIDE)));
        if verify && matches!(composed, super::reader::Composed::InTarget) {
            self.verify.lock().after("vpro", screen, !reader.skipped);
        }
        crate::pf_end!(crate::perfetto::events::ODSY_FRAME_PLANES);
        let (width, height) = (reader.frame_width as usize, reader.frame_height as usize);
        if width == 0 || height == 0 {
            return Composed::Unchanged;
        }
        let (w, h) = (width.min(FB_STRIDE), height.min(screen.fb_rgb.len() / FB_STRIDE));
        match composed {
            super::reader::Composed::No => return Composed::Unchanged,
            super::reader::Composed::InTarget => {}
            super::reader::Composed::InFrame => {
                // 0x00RRGGBB to opaque 0xAABBGGRR: a byte swap with alpha set.
                for (y, src) in reader.frame[..width * height].chunks_exact(width).enumerate().take(h) {
                    let at = y * FB_STRIDE;
                    for (d, &rgb) in screen.fb_rgb[at..at + w].iter_mut().zip(&src[..w]) {
                        *d = super::frame::to_screen(rgb);
                    }
                }
            }
        }
        screen.width = w;
        screen.height = h;
        screen.prebuilt = true;
        Composed::Frame
    }
}

/// The board on the refresh thread.
pub(super) struct Scan {
    pub state: super::BoardLocks,
    pub display: Arc<Display>,
}

impl Head for Scan {
    /// The retrace: the host interface latches the swaps armed and raises
    /// the retrace source. The frame composed after the blank reads the
    /// buffer this latch displays, so nothing is captured before it.
    fn vblank_start(&self, _field: u32, _t_ns: u64) {
        self.state.trace.flush(false);
        let mut reader = self.display.reader.lock();
        reader.attach(&self.state.shared);
        reader.retrace();
        // The retrace runs at the rate of the mode the guest loaded (its
        // frame and line tables at its clock part's frequency).
        if let Some(timing) = reader.clock.mode.video_timing() {
            self.display.head.set_timing(timing);
        }
    }

    /// What the scan-out reads changes without a mark the display could
    /// test for less than `compose` does.
    fn changed(&self) -> bool {
        true
    }

    fn compose(&self, screen: &mut Rex3Screen, want: Want) -> Composed {
        crate::pf_begin!(crate::perfetto::events::ODSY_CAPTURE);
        let composed = self.display.compose(&self.state.shared, screen, want);
        crate::pf_end!(crate::perfetto::events::ODSY_CAPTURE);
        composed
    }

    fn gfifo_pending(&self) -> usize {
        self.state.fifo.level() as usize + self.state.raster.pending() as usize
    }
}

impl Board {
    /// Pixel (x, y)'s main window id: bits 7:3 of its window-id byte in the
    /// buffer the scan-out names (PB&J 0x2800 base, 0x2807 stride).
    pub fn main_id(&mut self, x: u32, y: u32) -> u32 {
        // The id bytes are the front end's own: no wait (Board::id_compare_passes).
        let did = DestMem::of(self.vram_ptr(), &self.dest_views()).did.load1(x as i32, y as i32).unwrap_or(0);
        ((did as u32) >> ODSY_DID_MAIN_SHIFT) & (dispmode::COUNT - 1)
    }

    /// The colormap group main drawable main_id shows through: bits 8:4 of its
    /// display mode. Below ODSY_GAMMA_GROUP the word is an index.
    pub fn display_group(&mut self, main_id: u32) -> u32 {
        self.display.regs.display_group(main_id)
    }

    /// A vertical retrace: the host interface latches the swaps armed and
    /// raises the retrace source (HostInterface::retrace()).
    #[cfg(test)]
    pub fn retrace(&mut self) {
        self.display.retrace();
    }
}

/// The scan-out as the tests and the benchmark run it on a board they hold,
/// without a display thread: the same reader the display uses.
#[cfg(test)]
impl Board {
    fn test_reader(&mut self) -> &mut ScanoutReader {
        self.raster_wait();
        self.reader.attach(&self.display);
        &mut self.reader
    }

    /// Compose the whole frame into `reader.frame`.
    pub fn scanout_update(&mut self) -> bool {
        self.test_reader().update()
    }
}

/// What one main drawable id shows, decided once per update instead of per
/// pixel: which colour buffer, and how its plane word becomes a colour.
#[derive(Clone, Copy, Default)]
pub(super) struct IdShow {
    /// The displayed colour buffer, 0 or 1.
    pub buf: usize,
    /// Whether the plane word is RGB through the gamma table.
    pub rgb: bool,
    /// An RGB mode's channel fold: (c & keep) | (c >> down).
    pub keep: u32,
    pub down: u32,
    /// An index mode: lut entry = base + (word & mask).
    pub base: u32,
    pub mask: u32,
    /// The mode is colour map group 25, reported once when shown.
    pub unverified: bool,
}

/// Everything a row's resolve reads outside the planes, gathered once from a
/// copy of the registers, so none of it can change meanwhile.
pub(super) struct ResolveCtx {
    pub ids: [IdShow; dispmode::COUNT as usize],
    /// Per overlay id: whether it is enabled, and its entry base.
    pub overlay: [(bool, u32); overlay::IDS as usize],
    /// Every lookup-RAM entry as 0x00RRGGBB (DisplayRegs::lut_rgb()).
    pub rgb: Box<[u32; lut::SIZE as usize]>,
    /// DisplayRegs::gamma() of each 8-bit channel value, placed in its byte of
    /// the pixel: red, green, blue.
    pub gamma: Box<[[u32; 256]; 3]>,
}

impl ResolveCtx {
    /// What the registers `regs` make of each drawable, overlay id, lookup
    /// entry and gamma step, with colour buffer (displayed >> id) & 1
    /// showing for each drawable.
    pub(super) fn new(regs: &DisplayRegs, displayed: u32) -> ResolveCtx {
        let mut ids = [IdShow::default(); dispmode::COUNT as usize];
        let mut overlay = [(false, 0u32); overlay::IDS as usize];
        let mut rgb = Box::new([0u32; lut::SIZE as usize]);
        let mut gamma = Box::new([[0u32; 256]; 3]);

        for (i, v) in rgb.iter_mut().enumerate() {
            *v = regs.lut_rgb(i as u32);
        }
        for c in 0..256u32 {
            gamma[0][c as usize] = regs.gamma(c, lut::RED_SHIFT) << 16;
            gamma[1][c as usize] = regs.gamma(c, lut::GREEN_SHIFT) << 8;
            gamma[2][c as usize] = regs.gamma(c, lut::BLUE_SHIFT);
        }
        for (id, show) in ids.iter_mut().enumerate() {
            let mode = regs.dispmode[id];
            let group = regs.display_group(id as u32);

            show.buf = ((displayed >> id) & 1) as usize;
            if group >= ODSY_GAMMA_GROUP {
                show.rgb = true;
                // display_channel() as a mask and a shift; a shift of 8 or
                // more leaves a byte's high part empty.
                (show.keep, show.down) = match (mode >> dispmode::WIDTH_SHIFT) & dispmode::WIDTH_MASK {
                    dispmode::WIDTH_4 => (0xf0, 4),
                    dispmode::WIDTH_5 => (0xf8, 5),
                    _ => (0xff, 8),
                };
            } else if group != 0 {
                show.base = 256 * group;
                show.mask = 0xff;
                show.unverified = group == ODSY_CMAP_UNVERIFIED_GROUP;
            } else {
                show.mask = ODSY_INDEX_MASK;
            }
        }
        for (id, o) in overlay.iter_mut().enumerate() {
            let mode = regs.overlay_mode[id];

            *o = (
                mode & overlay::MODE_ENABLE != 0,
                256 * ((mode >> overlay::MODE_GROUP_SHIFT) & overlay::MODE_GROUP_MASK),
            );
        }
        ResolveCtx { ids, overlay, rgb, gamma }
    }
}

impl crate::gfx_display::GfxDisplay for Display {
    fn display_stop(&self) -> &crate::gfx_display::DisplayStop { self.head.display_stop() }
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { self.head.renderer_slot() }
    fn screen(&self) -> &Mutex<Rex3Screen> { self.head.screen() }
    fn request_screenshot(&self) { self.head.request_screenshot(); }
    fn screenshot_pending(&self) -> bool { self.head.screenshot_pending() }
    fn cycles(&self) -> CyclesPtr { self.status.cycles() }
    fn request_frame(&self) { self.head.request_frame(); }
    fn frame_pending(&self) -> bool { self.head.frame_pending() }
    fn set_capture(&self, on: bool) { self.head.set_capture(on); }
    fn hw_cursor(&self) -> Option<crate::gfx_display::HwCursor> {
        // A racy read of the registers, as the display thread's frame does.
        self.shared.get()?.regs.hw_cursor()
    }
    #[cfg(feature = "wgpu")]
    fn gpu(&self) -> Option<std::sync::Arc<crate::wgpu::Gpu>> {
        Some(self.shared.get()?.gpu_scanout.get()?.lock().gpu())
    }
}
