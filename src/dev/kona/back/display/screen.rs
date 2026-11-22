//! The window the board shows the guest: what `GfxDisplay` hands out. The
//! machine keeps a clone of the `Arc` and returns it from `get_display`.

use crate::cpu::mips_core::CyclesPtr;
use crate::dev::ng1::rex3::Renderer;
use crate::disp::Rex3Screen;
use crate::status::MachineStatus;
use crate::display::HeadHandle;
use crate::gfx_display::{DisplayStop, GfxDisplay};
use parking_lot::Mutex;
use std::sync::Arc;

/// The window the board shows the guest: what `GfxDisplay` hands out. Its
/// screen, renderer slot and frame requests are a head of the machine's
/// display refresh thread (`crate::display`), made when first needed, from
/// the status source the display has by then.
pub struct Display {
    head: std::sync::OnceLock<Arc<HeadHandle>>,
    /// The status source the head was made from, kept so its refresh thread
    /// lives as long as the display.
    home: Mutex<Option<Arc<MachineStatus>>>,
    status: Mutex<Arc<MachineStatus>>,
    tables: std::sync::OnceLock<Arc<Mutex<super::tables::Tables>>>,
}

impl Display {
    pub fn new(status: Arc<MachineStatus>) -> Display {
        Display { head: std::sync::OnceLock::new(), home: Mutex::new(None), status: Mutex::new(status), tables: std::sync::OnceLock::new() }
    }

    /// The machine's status-bar inputs, known once the machine is.
    pub fn set_status(&self, status: Arc<MachineStatus>) {
        if let Some(head) = self.head.get() {
            head.set_status(status.clone());
        }
        *self.status.lock() = status;
    }

    pub fn set_tables(&self, tables: Arc<Mutex<super::tables::Tables>>) {
        let _ = self.tables.set(tables);
    }

    /// A display with no machine behind it, for tests.
    pub fn detached() -> Display {
        Display::new(MachineStatus::new(Arc::default(), Arc::default()))
    }

    pub fn head(&self) -> &Arc<HeadHandle> {
        self.head.get_or_init(|| {
            let status = self.status.lock().clone();
            *self.home.lock() = Some(status.clone());
            let refresh = status.refresh().clone();
            refresh.head(status)
        })
    }
}

impl GfxDisplay for Display {
    fn display_stop(&self) -> &DisplayStop { self.head().display_stop() }
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { self.head().renderer_slot() }
    fn screen(&self) -> &Mutex<Rex3Screen> { self.head().screen() }
    fn request_screenshot(&self) { self.head().request_screenshot(); }
    fn screenshot_pending(&self) -> bool { self.head().screenshot_pending() }
    fn cycles(&self) -> CyclesPtr { self.status.lock().cycles() }
    fn hw_cursor(&self) -> Option<crate::gfx_display::HwCursor> {
        // Read live from the registers, not from the last composed frame.
        self.tables.get()?.lock().snapshot().hw_cursor()
    }
    fn request_frame(&self) { self.head().request_frame(); }
    fn frame_pending(&self) -> bool { self.head().frame_pending() }
    fn set_capture(&self, on: bool) { self.head().set_capture(on); }
}
