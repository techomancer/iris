//! What the UI, CI socket and GUI front-end need from whichever graphics board
//! drives the screen (Newport/REX3 or GR2).
//!
//! Each board owns its display thread and composition; this trait only exposes
//! the hand-off points: the renderer slot the front-end installs into, the
//! last-presented screen (screenshots), and the CPU cycle counter.

use parking_lot::Mutex;

use crate::disp::Rex3Screen;
use crate::mips_core::CyclesPtr;
use crate::rex3::Renderer;

pub trait GfxDisplay: Send + Sync {
    /// Slot the front-end installs its renderer into. The board's display
    /// thread presents every frame through whatever is installed here.
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>>;

    /// Last presented frame state. Screenshots read `rgba` (stride 2048).
    fn screen(&self) -> &Mutex<Rex3Screen>;

    /// Ask the display thread to read the next frame back into `screen().rgba`.
    fn request_screenshot(&self);

    /// CPU cycle counter the board was wired to (GUI's live MIPS estimate).
    fn cycles(&self) -> CyclesPtr;
}
