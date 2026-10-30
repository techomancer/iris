//! The SGI VPro (Odyssey) graphics board in the Octane's graphics port.
//!
//! The board is a crosstalk widget. The processor stores commands into a
//! command FIFO; the board decodes them, draws integer and GL primitives
//! into its own memory, moves pixels to and from guest memory by DMA, and
//! scans the colour planes out to the display. Three threads reach the
//! board: the processor's widget accesses, the display thread, which counts
//! the retrace and copies the scanout's frame to the window 60 times a
//! second, and the board's own thread (worker.rs), which runs the queued
//! command-FIFO stores. interface.rs describes the two locks they share.
//! Interrupt packets go to HEART, the only interrupt target on the Octane:
//! a packet to widget 8 sets or clears the HEART pending bit of its vector.
//!
//! - `board`: the board's state, construction and reset.
//! - `interface`: the host interface: the registers that do not wait for
//!   drawing, interrupts and the queue of command stores.
//! - `mmio`: register accesses, and which of them wait for the drawing.
//! - `fifo`: the command FIFO between the processor and the board's thread.
//! - `worker`: the board's thread, which runs the queued stores.
//! - `registers`: identity, the board NIC, the cursor and the back end's
//!   register file.
//! - `command`: the command-FIFO interpreter.
//! - `context`: graphics context save and switch.
//! - `draw`: buffers, integer drawing and the image path.
//! - `objects`: off-screen objects in board memory.
//! - `dma`: pixel DMA, queries and fences.
//! - `gl`, `geometry`: GL fragment state and texturing; the geometry
//!   program, lighting and the GL primitives.
//! - `gl_context`: the GL state machine the command stream drives.
//! - `raster`: the scalar rasteriser's rules, fragment path and draw entry;
//!   `raster_triangle`, `raster_line`, `raster_pixels`: triangles and
//!   polygons, lines and points, pixel rectangles, bitmaps, clears and
//!   raster ops; `raster_texture`: texture sampling; `raster_bands`:
//!   drawing in bands of rows on host threads; `raster_fixed_texture`: the
//!   fixed-point fast path for textured triangles.
//! - `memory`, `scanout`, `display`: board memory and planes; the scanned
//!   frame and retrace; what reaches the window.
//! - `pixel_dma`, `hooks`, `pi_gfx`: DMA between guest memory and planes,
//!   the hooks the scanout and DMA call back into the board, and graphics
//!   credits for machines whose processor interface counts them.
//! - `consts`: register, command and field constants.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code
//! (hw/display/sgi/) in jamesbraid/qemu-sgi-origami.

#![allow(clippy::all)]

use crate::devlog::LogModule;

pub mod bits;
pub mod board;
pub mod command;
pub mod consts;
pub mod context;
pub mod display;
pub mod dma;
pub mod draw;
pub mod fifo;
pub mod geometry;
pub mod gl;
pub mod gl_context;
pub mod hooks;
pub mod interface;
pub mod memory;
pub mod mmio;
pub mod objects;
pub mod pi_gfx;
pub mod pixel_dma;
pub mod ptr;
pub mod raster;
pub mod raster_bands;
pub mod raster_fixed_texture;
pub mod raster_line;
pub mod raster_pixels;
pub mod raster_texture;
pub mod raster_triangle;
pub mod registers;
pub mod scanout;
pub mod worker;
#[cfg(test)]
mod tests;

pub use bits::*;
pub use board::*;
pub use command::*;
pub use consts::*;
pub use context::*;
pub use display::*;
pub use dma::*;
pub use draw::*;
pub use fifo::*;
pub use geometry::*;
pub use gl::*;
pub use gl_context::*;
pub use hooks::*;
pub use interface::*;
pub use memory::*;
pub use mmio::*;
pub use objects::*;
pub use pi_gfx::*;
pub use pixel_dma::*;
pub use ptr::*;
pub use raster::*;
pub use raster_bands::*;
pub use raster_fixed_texture::*;
pub use raster_line::*;
pub use raster_pixels::*;
pub use raster_texture::*;
pub use raster_triangle::*;
pub use registers::*;
pub use scanout::*;
pub use worker::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use parking_lot::Mutex;

use crate::cpu::mips_core::CyclesPtr;
use crate::dev::ng1::rex3::Renderer;
use crate::disp::{Rex3Screen, FB_STRIDE};
use crate::ip30::{widget_large_base, widget_small_base};
use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, Resettable, BUS_ERR, BUS_OK};

/// The VPro board on widget `slot`: a `BusDevice` for the widget's
/// windows.
pub struct Odyssey {
    small: u64,
    large: u64,
    pub state: BoardLocks,
    pub display: Arc<Display>,
    running: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Odyssey {
    /// A board of `product` (v12, v10, v8, v6) at `widget`: its interrupt
    /// packets reach `irq` (HEART), its DMA reaches `mem`.
    pub fn new(
        widget: u8,
        product: &str,
        irq: Arc<dyn InterruptController>,
        mem: Arc<dyn BusDevice>,
    ) -> Result<Arc<Self>, String> {
        let mut s = Board::new(product, widget as u32)?;
        s.guest = Some(mem);
        s.front.lock().irq = Some(irq);
        Ok(Arc::new(Self {
            small: widget_small_base(widget),
            large: widget_large_base(widget),
            state: BoardLocks::new(s),
            display: Arc::new(Display::new()),
            running: Arc::new(AtomicBool::new(false)),
            thread: Mutex::new(None),
        }))
    }

    pub fn display(&self) -> Arc<dyn crate::gfx_display::GfxDisplay> {
        self.display.clone()
    }

    pub fn set_cpu_cycles(&self, cycles: CyclesPtr) {
        *self.display.cycles.lock() = cycles;
    }

    /// The aperture offset of `addr`, in either window.
    fn offset(&self, addr: u64) -> Option<u64> {
        let off = if addr >= self.large { addr - self.large } else { addr.wrapping_sub(self.small) };
        (off < ODSY_APERTURE as u64).then_some(off)
    }

    fn read(&self, addr: u64, size: u32) -> (u32, u64) {
        match self.offset(addr) {
            Some(off) => {
                (BUS_OK, self.state.read(off, size))
            }
            None => (BUS_ERR, 0),
        }
    }

    fn write(&self, addr: u64, size: u32, val: u64) -> u32 {
        match self.offset(addr) {
            Some(off) => {
                self.state.write(off, val, size);
                BUS_OK
            }
            None => BUS_ERR,
        }
    }

    /// One display frame: the retrace if it is due, then the dirty lines
    /// resolved and the frame presented.
    fn frame(state: &BoardLocks, display: &Display) {
        let (frame, w, h) = {
            let mut s = state.board.lock();
            s.scanout.poll(monotonic_ns());
            s.scanout.update();
            (s.scanout.frame.clone(), s.scanout.frame_width as usize, s.scanout.frame_height as usize)
        };
        if w != 0 && h != 0 && frame.len() >= w * h {
            display.present(&frame, w, h);
        }
    }
}

impl BusDevice for Odyssey {
    fn read8(&self, addr: u64) -> BusRead8 {
        let (status, data) = self.read(addr, 1);
        BusRead8 { status, data: data as u8 }
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        let (status, data) = self.read(addr, 2);
        BusRead16 { status, data: data as u16 }
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        let (status, data) = self.read(addr, 4);
        BusRead32 { status, data: data as u32 }
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        let (status, data) = self.read(addr, 8);
        BusRead64 { status, data }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write(addr, 1, val as u64)
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write(addr, 2, val as u64)
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.write(addr, 4, val as u64)
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write(addr, 8, val)
    }
}

/// The display thread's frame period: the modelled mode's 60 Hz.
const FRAME_PERIOD: std::time::Duration = std::time::Duration::from_micros(16_667);

impl Device for Odyssey {
    fn step(&self, _cycles: u64) {}

    fn start(&self) {
        let mut t = self.thread.lock();
        if t.is_some() {
            return;
        }
        self.state.work_start();
        self.running.store(true, Ordering::Release);
        let (state, display, running) = (self.state.clone(), self.display.clone(), self.running.clone());
        *t = Some(
            std::thread::Builder::new()
                .name("odyssey-display".into())
                .spawn(move || {
                    let mut next = std::time::Instant::now();
                    while running.load(Ordering::Acquire) {
                        Odyssey::frame(&state, &display);
                        next += FRAME_PERIOD;
                        let now = std::time::Instant::now();
                        if next > now {
                            std::thread::sleep(next - now);
                        } else {
                            next = now;
                        }
                    }
                    display.stop();
                })
                .expect("odyssey display thread"),
        );
    }

    fn stop(&self) {
        self.running.store(false, Ordering::Release);
        if let Some(t) = self.thread.lock().take() {
            let _ = t.join();
        }
        self.state.work_stop();
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    fn get_clock(&self) -> u64 {
        0
    }
}

impl Drop for Odyssey {
    fn drop(&mut self) {
        // The work thread holds a clone of the state's handles; stop it
        // so the state can go.
        Device::stop(self);
    }
}

impl Resettable for Odyssey {
    fn power_on(&self) {
        self.state.board.lock().reset();
    }
}
