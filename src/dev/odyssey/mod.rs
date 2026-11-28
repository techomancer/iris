//! The SGI VPro (Odyssey) graphics board in the Octane's graphics port.
//!
//! The board is a crosstalk widget. The processor stores commands into a
//! command FIFO; the board decodes them, draws integer and GL primitives
//! into its own memory, moves pixels to and from guest memory by DMA, and
//! scans the colour planes out to the display. Four threads reach the
//! board: the processor's widget accesses; the board's own thread
//! (worker.rs), BUZZ's front end, which runs the queued command-FIFO stores;
//! BUZZ's raster unit (raster_unit.rs), which draws what the front end sets
//! up; and the display thread, the back end, which counts the retrace and
//! copies the scanout's frame to the window 60 times a second.
//! interface.rs describes how the first two share the board's state.
//! Interrupt packets go to HEART, the only interrupt target on the Octane:
//! a packet to widget 8 sets or clears the HEART pending bit of its vector.
//!
//! - `board`: the board's state, construction and reset.
//! - `interface`: the host interface: the registers that do not wait for
//!   drawing, interrupts and the queue of command stores.
//! - `mmio`: register accesses, and which of them wait for the drawing.
//! - `fifo`: the command FIFO between the processor and the board's thread.
//! - `worker`: the board's thread, which runs the queued stores.
//! - `raster_unit`: the raster unit's thread and the queue that feeds it.
//! - `registers`: identity, the board NIC and the widget registers.
//! - `display_fifo`: the display FIFO, PB&J's register file (cursor, lookup
//!   RAM, display modes, scan-out size, I2C), with its own lock.
//! - `command`: the command-FIFO interpreter.
//! - `context`: graphics context save and switch.
//! - `draw`: buffers, integer drawing and the image path.
//! - `objects`: off-screen objects in board memory.
//! - `dma`: pixel DMA, queries and fences.
//! - `pixel_write`: the library's image-write codes and their pixels.
//! - `gl`, `geometry`: GL fragment state and texturing; the geometry
//!   program, lighting and the GL primitives.
//! - `gl_context`: the GL state machine the command stream drives.
//! - `raster`: the scalar rasteriser's rules, fragment path and draw entry;
//!   `raster_triangle`, `raster_line`, `raster_pixels`: triangles and
//!   polygons, lines and points, pixel rectangles, bitmaps, clears and
//!   raster ops; `raster_texture`: texture sampling; `raster_fixed_texture`:
//!   the fixed-point fast path for textured triangles; `raster_fill`: the row
//!   loops filled triangles take, specialised by draw state and planes;
//!   `raster_bench`: a fill-rate benchmark (tests only).
//! - `memory`, `scanout`, `reader`, `display`: board memory and planes; the
//!   marks writers make and the display thread's own reader, with the scanned
//!   frame and retrace; what reaches the window.
//! - `pixel_dma`: DMA between guest memory and planes.
//! - `depth_word`: the board's 24-bit float depth word and its offset.
//! - `consts`: register, command and field constants.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code
//! (hw/display/sgi/) in jamesbraid/qemu-sgi-origami.

// The crate allows dead code; this board's code does not hide any.
#![warn(dead_code, unused_imports)]

// The rasteriser writes its range tests as !(x > 0.0) and the like so that
// a NaN fails them.
#![allow(clippy::neg_cmp_op_on_partial_ord, reason = "a NaN must fail these tests")]

mod bits;
mod board;
mod command;
mod consts;
mod context;
mod copy;
mod debug;
mod depth_word;
mod display;
#[cfg(test)]
mod display_bench;
mod display_fifo;
mod dma;
mod draw;
mod field;
mod fifo;
mod frame;
#[cfg(test)]
mod golden;
mod geometry;
mod gl;
mod gpu_unit;
mod assembly;
mod interface;
mod clip;
mod logic;
mod mmio;
#[cfg(feature = "vprodiag")]
mod packing;
mod pixel_dma;
mod pipe;
mod pixel_write;
#[cfg(test)]
mod raster_bench;
mod raster_unit;
mod record;
mod reader;
mod registers;
mod scanout;
mod snapshot;
#[cfg(test)]
mod tests;
mod trace;
mod view;
mod vram;
#[cfg(feature = "wgpu")]
mod wgpu;
mod worker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;


use crate::ip30::{widget_large_base, widget_small_base};
use crate::irq::InterruptController;
use crate::traits::{
    BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, Resettable, BUS_BUSY, BUS_ERR,
    BUS_OK,
};

use board::Board;
use consts::ODSY_APERTURE;
use display::Display;
use mmio::BoardLocks;

/// The VPro board on widget `slot`: a `BusDevice` for the widget's
/// windows.
pub struct Odyssey {
    aperture: Aperture,
    state: BoardLocks,
    display: Arc<Display>,
    running: Arc<AtomicBool>,
    /// The board as the refresh thread calls it; the display's head holds
    /// it weakly.
    #[allow(dead_code)]
    scan: Arc<display::Scan>,
}

/// Carries a board interrupt packet into a machine's crosstalk fabric: the
/// destination widget, the address within it, the vector, and whether the
/// source rose or fell. False when nothing at the destination takes it.
pub type PacketSender = Arc<dyn Fn(u8, u64, u8, bool) -> bool + Send + Sync>;

/// How the processor's addresses reach the board, chosen when it is built.
#[derive(Clone, Copy)]
enum Aperture {
    /// The Octane's: the widget's small and large IP30 windows.
    Ip30Windows { small: u64, large: u64 },
    /// Every address is already an aperture offset, as a machine whose I/O
    /// interface hands widgets offsets (Bedrock) presents it.
    Offsets,
}

impl Odyssey {
    /// A board of `product` (v12, v10, v8, v6) at `widget` on the Octane:
    /// the processor reaches it through the IP30 widget windows, its
    /// interrupt packets reach `irq` (HEART), its DMA reaches `mem`. `pool`
    /// is the CPU raster unit's workers and tile size, `raster` which unit
    /// draws and, for the GPU, the wgpu backend (`[graphics]`).
    #[allow(clippy::too_many_arguments, reason = "the board's wiring")]
    pub fn new(
        widget: u8,
        product: &str,
        irq: Arc<dyn InterruptController>,
        mem: Arc<dyn BusDevice>,
        status: Arc<crate::status::MachineStatus>,
        pool: (usize, u32),
        raster: (crate::config::RasterUnit, &str),
    ) -> Result<Arc<Self>, String> {
        let aperture = Aperture::Ip30Windows { small: widget_small_base(widget), large: widget_large_base(widget) };
        Self::build(widget, product, interface::heart_sender(irq), aperture, mem, status, pool, raster)
    }

    /// A board of `product` at `widget` on a machine that hands the board
    /// widget-relative addresses, as Bedrock's I/O interface does, and
    /// carries its interrupt packets through `packets`.
    #[allow(clippy::too_many_arguments, reason = "the board's wiring")]
    pub fn new_widget_relative(
        widget: u8,
        product: &str,
        packets: PacketSender,
        mem: Arc<dyn BusDevice>,
        status: Arc<crate::status::MachineStatus>,
        pool: (usize, u32),
        raster: (crate::config::RasterUnit, &str),
    ) -> Result<Arc<Self>, String> {
        Self::build(widget, product, packets, Aperture::Offsets, mem, status, pool, raster)
    }

    #[allow(clippy::too_many_arguments, reason = "the board's wiring")]
    fn build(
        widget: u8,
        product: &str,
        packets: PacketSender,
        aperture: Aperture,
        mem: Arc<dyn BusDevice>,
        status: Arc<crate::status::MachineStatus>,
        pool: (usize, u32),
        raster: (crate::config::RasterUnit, &str),
    ) -> Result<Arc<Self>, String> {
        let mut s = Board::new(product, widget as u32)?;
        // The fragment pipeline's workers and tile size (`[graphics]`).
        s.raster.set_pool(pool.0, pool.1);
        if raster.0 == crate::config::RasterUnit::Wgpu {
            #[cfg(feature = "wgpu")]
            s.enable_gpu(&crate::wgpu::GpuOptions { backend: raster.1.to_string(), ..Default::default() })?;
            #[cfg(not(feature = "wgpu"))]
            return Err("graphics.raster = \"wgpu\" needs a build with the wgpu feature".into());
        }
        // The recorder's slot is there whether or not a recording runs, so
        // the monitor can start one (`vpro rec`); the DMA path records
        // through it.
        let rec = record::RecSlot::new(None);
        let mem: Arc<dyn BusDevice> = Arc::new(record::RecordingMem { inner: mem, rec: rec.clone() });
        s.guest = Some(mem);
        s.front.lock().packets = Some(packets);
        let mut state = BoardLocks::new(s);
        state.rec = rec;
        let display = Arc::new(Display::new(status));
        let scan = Arc::new(display::Scan { state: state.clone(), display: display.clone() });
        let head: Arc<dyn crate::display::Head> = scan.clone();
        display.head.attach(Arc::downgrade(&head));
        Ok(Arc::new(Self { aperture, state, display, scan, running: Arc::new(AtomicBool::new(false)) }))
    }

    pub fn display(&self) -> Arc<dyn crate::gfx_display::GfxDisplay> {
        // The host's cursor query reads the registers the display reads.
        self.display.attach_registers(self.state.shared.clone());
        self.display.clone()
    }

    /// The aperture offset of `addr`, in either window.
    fn offset(&self, addr: u64) -> Option<u64> {
        let off = match self.aperture {
            Aperture::Ip30Windows { small, large } => if addr >= large { addr - large } else { addr.wrapping_sub(small) },
            Aperture::Offsets => addr,
        };
        (off < ODSY_APERTURE).then_some(off)
    }

    fn read(&self, addr: u64, size: u32) -> (u32, u64) {
        if self.state.gpu_fault().is_some() {
            return (BUS_ERR, 0);
        }
        match self.offset(addr) {
            Some(off) => match self.state.bus_read(off, size) {
                Some(v) if self.state.gpu_fault().is_none() => (BUS_OK, v),
                Some(_) => (BUS_ERR, 0),
                None => (BUS_BUSY, 0),
            },
            None => (BUS_ERR, 0),
        }
    }

    fn write(&self, addr: u64, size: u32, val: u64) -> u32 {
        if self.state.gpu_fault().is_some() {
            return BUS_ERR;
        }
        match self.offset(addr) {
            Some(off) => {
                if self.state.bus_write(off, val, size) && self.state.gpu_fault().is_none() {
                    BUS_OK
                } else if self.state.gpu_fault().is_some() {
                    BUS_ERR
                } else {
                    BUS_BUSY
                }
            }
            None => BUS_ERR,
        }
    }

    /// One display frame with no refresh thread, for tests and benchmarks:
    /// one retrace, then the frame composed into the screen. Whether a new frame was composed.
    #[cfg(test)]
    fn frame(state: &BoardLocks, display: &Display) -> bool {
        let mut reader = display.reader.lock();
        reader.attach(&state.shared);
        reader.retrace();
        drop(reader);
        let want = display.head.want();
        !matches!(display.compose(&state.shared, &mut display.screen().lock(), want), crate::display::Composed::Unchanged)
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

impl Device for Odyssey {
    fn step(&self, _cycles: u64) {}

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(
            "vpro".to_string(),
            "VPro (Odyssey) graphics: status|regs|fifo|state|geom|tex|win|mem|backend|lut|dispmode|cursor|pix|raw|peek|shot|fbdump|hash|unimp|refusals|gpu|trace|rec (vpro help)".to_string(),
        )]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "vpro" {
            return Err(format!("unknown command: {cmd}"));
        }
        self.cmd_vpro(args, &mut *w)
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        self.state.work_start();
        self.display.head.start();
    }

    fn stop(&self) {
        // The renderer is released on the refresh thread, which owns its
        // context; a stop from the main thread leaves it alone
        // (`DisplayStop`).
        self.display.head.stop();
        if !self.running.swap(false, Ordering::AcqRel) {
            return;
        }
        self.state.work_stop();
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
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
        // SAFETY: power-on comes before the machine starts the board's
        // threads and the processors.
        let b = unsafe { &mut *self.state.board.get() };
        b.reset();
        // A power-on recording opens with the state the replay starts from.
        if self.state.rec.active() {
            if let Some(r) = self.state.rec.lock().as_mut() {
                r.mark(debug::memory_hash_bytes(b));
            }
        }
    }
}
