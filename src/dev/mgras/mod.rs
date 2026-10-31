//! IMPACT (MGRAS) graphics for the Indigo2.
//!
//! One board occupies the GIO graphics slot (`0x1F000000`) and decodes a 1 MB
//! window there. The host talks to one chip on it, the host interface,
//! which offers:
//!
//! | offset            | what                                              |
//! |-------------------|---------------------------------------------------|
//! | `0x00000`         | GIO ID                                            |
//! | `0x40000-0x45FFF` | command-processor microcode RAM (24-bit words)    |
//! | `0x50000-0x5FFFF` | privileged registers, privileged command FIFO     |
//! | `0x60000-0x67FFF` | display control bus devices (see `dcb`)           |
//! | `0x68000-0x6FFFF` | per-device bus protocol registers                 |
//! | `0x70000-0x7BFFF` | user registers: status, flags, command FIFO       |
//! | `0x7C000-0x7FFFF` | raster registers, direct access (see `rss`);      |
//! |                   | the alias at `+0x1000` also executes the IR       |
//!
//! Drawing arrives through the command FIFO as (command, data) pairs. Commands
//! at `0x1000` and up write raster registers directly, with bit `0x400`
//! meaning "execute"; lower numbers go to the command processor's microcode,
//! which this model interprets at the command/GE level rather than executing
//! the uploaded microcode. FIFO words are queued to the HQ3 thread, which
//! forwards raster work to the RSS thread; FIFO status reflects queued work.
//!
//! The model scans its framebuffer out through the colormap and DAC gamma into
//! the same window and status bar Newport uses (`GfxDisplay`).
//!
//! `IRIS_MGRAS_TRACE=<file>` logs every access to the board.

//!
//! Threads (clanker/mgras/DESIGN.md): the CPU thread owns the host-side
//! registers and the display control bus (`Front`). Command FIFO words and
//! direct raster writes go into `hq_fifo`; the HQ3 thread runs the frontend
//! and feeds raster writes into `rss_fifo`; the RSS thread draws. The display
//! thread scans the framebuffer out unlocked.

mod dcb;
mod fixed;
mod debug;
mod disp;
mod frame;
mod ge11;
mod gl;
mod hq3;
pub mod hq4;
pub mod microcode;
mod pixmem;
mod pixpipe;
mod plain;
mod record;
mod rss;
#[cfg(feature = "gr4-jit")]
mod rss_jit;
mod te1;

#[cfg(test)]
#[path = "mgras_tests.rs"]
mod mgras_tests;

#[cfg(all(test, feature = "gr4-jit"))]
#[path = "rss_jit_tests.rs"]
mod rss_jit_tests;

use parking_lot::Mutex;
use std::cell::UnsafeCell;
use std::collections::HashSet;
use std::io::Write as IoWrite;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::mem::MaybeUninit;
use std::ptr::addr_of_mut;
use std::sync::Arc;
use std::thread;

use crate::config::GraphicsBoard;
use crate::dev::ng1::rex3::Renderer;
use crate::gfifo::GFifo;
use crate::locks::SpinLock;
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_BUSY, BUS_OK, Device, Saveable};

use ge11::Ge11;
use hq3::{host, Hq3Engine, Hq3Regs, Hq3Sink};
pub use gl::Gl;
pub use hq3::Hq3Board;
use rss::Rss;

/// The graphics slot the board decodes.
pub const MGRAS_SLOT_GFX_BASE: u32 = 0x1F00_0000;
pub const MGRAS_SLOT_GFX_SIZE: u32 = 0x0040_0000;
/// The board's register window within the slot.
const MAP_SIZE: u32 = 0x10_0000;

/// GIO ID: product 0x10, 32-bit ID, revision 1, GIO64, no ROM, manufacturer 1.
pub const GIO_ID: u32 = 0x0005_0190;

const HQ_FIFO_DEPTH: usize = 65536;
/// Deep enough for most of a frame's texture loads (Quake 3 on High IMPACT
/// moves ~13 MB a frame, 1.6 M entries), so the HQ finishes a DMA, and the
/// CPU waiting on DMABUSY goes on, while the engines are still drawing.
/// At 64 K entries the HQ waited for room on every load.
const RSS_FIFO_DEPTH: usize = 1 << 20;
/// Command FIFO words FIFOSTATUS counts at the top of `hq_fifo`
/// (`cfifo_level`): room above the kernel's high-water mark of 32.
const CFIFO_WORDS: usize = 64;
/// Raster engine threads at most (`IRIS_MGRAS_ENGINES`).
const MAX_ENGINES: usize = 4;

/// FIFO entry tags. `hq_fifo`: command FIFO words carry their port (0 user,
/// 1 privileged); `RSS | entry` is a direct raster write for the HQ3 to
/// forward in order. `rss_fifo`: a raster write is `entry` itself (register
/// << 1 | execute), below `0x800`. Ops share numbers in both.
mod tag {
    pub const PORT_USER: u32 = 0;
    pub const PORT_PRIVILEGED: u32 = 1;
    pub const RSS: u32 = 0x0100_0000;
    /// Context switch requested: the incoming context follows as FIFO words.
    pub const CONTEXT_SWITCH: u32 = 0x8000_0001;
    /// The host read the high half of a PIO read doubleword: next one.
    pub const PIO_ADVANCE: u32 = 0x8000_0004;
    /// DMA pixel line into the armed transfer: value is line | bytes << 32,
    /// the bytes follow as payload, eight to an entry, first byte highest.
    pub const DMA_LINE: u32 = 0x8000_0010;
    /// A GL batch starts / ends: the RSS stashes the raster registers it
    /// loads, and puts them back after (see `rss::Rss::gl_enter`).
    pub const GL_ENTER: u32 = 0x8000_0020;
    pub const GL_LEAVE: u32 = 0x8000_0021;
    pub const EXIT: u32 = 0x8000_00FF;
}

/// `rss_fifo` raster write from the CPU's direct register window (for the
/// trace; the RSS treats both sources alike).
const RSS_SRC_CPU: u32 = 0x800;

/// A raster write as `rss_fifo` carries it.
fn rss_entry(r: u32, exec: bool) -> u32 {
    (r & 0x3FF) << 1 | exec as u32
}

/// CPU-side board state: host interface registers, the geometry engines'
/// diagnostic ports, the display control bus. Plain data, valid zeroed.
#[repr(C)]
struct Front {
    hq: Hq3Regs,
    ge: Ge11,
    dcb: dcb::Dcb,
    /// The GE revision a started engine reports; 0 for the Indigo2
    /// boards' 1.
    ge_revision: u32,
}

/// `Front` borrowed under its lock.
struct FrontGuard<'a> {
    _lock: crate::locks::SpinGuard<'a, ()>,
    front: &'a mut Front,
}

impl std::ops::Deref for FrontGuard<'_> {
    type Target = Front;
    fn deref(&self) -> &Front { self.front }
}

impl std::ops::DerefMut for FrontGuard<'_> {
    fn deref_mut(&mut self) -> &mut Front { self.front }
}

impl Front {
    fn init(&mut self, kind: GraphicsBoard) {
        // Board version bytes: [RB revision (6:4, 7 = no RB board), RB TRAMs
        // (3:2), RA TRAMs (1:0), as log2 of the TRAM count; product (6:5) +
        // GE count (1:0)]. High and Max Impact get 4 TRAMs (4 MB) per board.
        let bdvers = match kind {
            GraphicsBoard::SolidImpact => [0x70, 0x21],
            GraphicsBoard::HighImpact => [0x72, 0x01],
            GraphicsBoard::MaxImpact => [0x0A, 0x02],
            _ => [0, 0],
        };
        self.dcb.init(bdvers, [0xFB, 0xFB]);
    }

    /// Flags derived from state, on top of the stored ones.
    fn derived_flags(&self) -> u32 {
        if self.ge.out.is_empty() { 0 } else { host::FLAG_GE_DIAG }
    }

    fn read(&mut self, off: u32, bits: u32) -> u64 {
        match off {
            0 => GIO_ID as u64,
            host::UCODE..=0x45FFF => self.hq.ucode[((off - host::UCODE) / 4) as usize] as u64,
            0x60000..=0x67FFF => {
                let t = dcb::Txn::decode(off);
                let d = self.dcb.read(t);
                t.load_value(bits, d)
            }
            0x68000..=0x6803F => {
                let dev = ((off - 0x68000) >> 2) as usize;
                if bits == 64 {
                    ((self.dcb.read_dcbctrl(dev) as u64) << 32) | (self.dcb.read_dcbctrl(dev + 1) as u64)
                } else {
                    self.dcb.read_dcbctrl(dev) as u64
                }
            }
            host::FLAG_ENABLE_SET | host::FLAG_ENABLE_CLEAR => self.hq.flag_enable as u64,
            host::INTERRUPT_ENABLE_SET | host::INTERRUPT_ENABLE_CLEAR => self.hq.interrupt_enable as u64,
            host::GE_DIAG_READ => self.ge.out.pop().unwrap_or(0) as u64,
            host::GE_READBACK_HI => self.hq.ge_readback[0] as u64,
            host::GE_READBACK_LO => self.hq.ge_readback[1] as u64,
            _ => self.hq.regs.get(off) as u64,
        }
    }

    /// Returns true when what is displayed changed.
    fn write(&mut self, off: u32, bits: u32, val: u64, flags: &AtomicU32) -> bool {
        match off {
            host::UCODE..=0x45FFF => {
                let i = ((off - host::UCODE) / 4) as usize;
                self.hq.ucode[i] = val as u32 & 0xFF_FFFF;
                self.hq.ucode_hash.word(i, val as u32 & 0xFF_FFFF);
                false
            }
            0x60000..=0x67FFF => {
                let t = dcb::Txn::decode(off);
                self.dcb.write(t, t.store_data(bits, val));
                // Display-side state (colormaps, modes, the cursor) changes
                // what is shown without touching the framebuffer. Index
                // writes (select 1 and below) change nothing by themselves.
                t.crs >= 2 || t.dev == dcb::DEV_VC3
            }
            0x68000..=0x6803F => {
                let dev = ((off - 0x68000) >> 2) as usize;
                if bits == 64 {
                    self.dcb.write_dcbctrl(dev, (val >> 32) as u32);
                    self.dcb.write_dcbctrl(dev + 1, val as u32);
                } else {
                    self.dcb.write_dcbctrl(dev, val as u32);
                }
                false
            }
            off if Ge11::is_diag_port(off) => {
                if self.ge.write(off, val as u32) {
                    // Started: the version program's answer is waiting
                    // (revision 1 on the Indigo2 boards).
                    self.hq.ge_readback = [0, self.ge_revision.max(1)];
                    flags.fetch_or(host::FLAG_GE_DATA, Ordering::AcqRel);
                }
                false
            }
            host::FLAG_ENABLE_SET => { self.hq.flag_enable |= val as u32; false }
            host::FLAG_ENABLE_CLEAR => { self.hq.flag_enable &= !(val as u32); false }
            host::INTERRUPT_ENABLE_SET => { self.hq.interrupt_enable |= val as u32; false }
            host::INTERRUPT_ENABLE_CLEAR => { self.hq.interrupt_enable &= !(val as u32); false }
            _ => {
                self.hq.regs.insert(off, val as u32);
                false
            }
        }
    }
}

/// The board's three interrupt lines to the GIO slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    /// GIO interrupt 0: command FIFO high/low water.
    Fifo,
    /// GIO interrupt 1: the general graphics interrupt.
    General,
    /// GIO interrupt 2: vertical retrace.
    Retrace,
}

/// How long `bound_busy_status` lets STATUS read busy: well inside the
/// IP30 kernel's waits on its idle bits.
const BUSY_STATUS_BOUND: std::time::Duration = std::time::Duration::from_millis(500);

/// How long a STATUS or DMABUSY read waits for a busy board to go idle
/// before it answers (`settle`).
const POLL_SETTLE: std::time::Duration = std::time::Duration::from_micros(100);

pub struct Mgras {
    kind: GraphicsBoard,
    /// Perfetto edge trackers (zero-valid, so `new`'s zeroed allocation
    /// initialises them): a guest read waiting for idle, a store stalled on
    /// a full FIFO.
    pf_wait: crate::PfLevel,
    pf_fifo_full: crate::PfLevel,
    /// The board's three GIO interrupts, as the slot wires them: FIFO,
    /// general, retrace (see [`Line`]).
    irqs: [crate::irq::IrqLine; 3],
    /// Current level of the general interrupt line.
    general: AtomicBool,
    /// The FIFO interrupt's inputs, kept apart from `front` for the HQ3
    /// thread: the INTR_*FIFO_* enables (their own bits), the command
    /// FIFO's high-water mark in 6:0 and its low-water mark in 14:8.
    fifo_cfg: AtomicU32,
    /// Current level of the FIFO interrupt line, decided under the lock so
    /// the CPU and HQ3 threads cannot leave a stale level behind.
    fifo_line: SpinLock<bool>,
    /// `fifo_line`'s level, for checking without the lock.
    fifo_asserted: AtomicBool,
    /// When the refresh thread began the current frame (its retrace pulse),
    /// in ns since `epoch`: STATUS reads vertical blank for 1 ms after it.
    epoch: std::time::Instant,
    frame_start: AtomicU64,
    /// The GE11 programs' fingerprints as a front-end that keeps its own
    /// copy of the download (the Octane's HQ4) reports them; 0 when it
    /// does not, and the GE11 diagnostic ports' copy is used.
    ge_upload_fingerprint: [AtomicU64; 2],
    /// CPU-side state and its lock. The display thread reads it unlocked
    /// (plain data, tearing tolerated).
    front_lock: SpinLock<()>,
    front: UnsafeCell<Front>,
    /// Commands and registers not modelled yet, reported once each.
    unhandled: SpinLock<HashSet<String>>,
    /// HQ3 flags: the HQ3 thread raises them, the CPU sets and clears them.
    flags: AtomicU32,
    /// `mgras probe` slots: GE readbacks in a range answer a chosen word
    /// (bits 63:48 first address, 47:32 last + 1, 31:0 the answer; 0 is
    /// unused). For mapping how libGLcore decodes GE state words.
    probe: [AtomicU64; 8],
    /// Owned by the HQ3 thread (others only while the board is idle).
    eng: UnsafeCell<Hq3Engine>,
    /// Raster engine 0, owned by its RSS thread (others only while the
    /// board is idle; the display thread reads the framebuffer unlocked).
    /// The HQ3, the CPU and the monitor read raster state through it.
    rss: UnsafeCell<Rss>,
    /// Maximum Impact's second raster engine, on its own thread: both take
    /// every `rss_fifo` entry, and each draws only its own scanlines
    /// (engine 0 the even ones), as the board's two RE4s do.
    rss1: UnsafeCell<Rss>,
    /// Engines 2 and 3: more threads splitting the scanlines
    /// (`IRIS_MGRAS_ENGINES=4`), row y to thread y % 4.
    rss_more: [UnsafeCell<Rss>; MAX_ENGINES - 2],
    /// The pixel memory and clip-ID planes the engines draw into.
    planes: UnsafeCell<rss::Planes>,
    /// Raster engines running (2 on Maximum Impact, else 1).
    nrss: usize,
    /// PIO read advances engine 0 has done; engine 1 waits for each before
    /// it goes on drawing (see `rss_loop`).
    pio_advances: AtomicU64,
    /// System memory, for the board's DMA (it is a GIO bus master).
    mem: SpinLock<Option<Arc<dyn BusDevice>>>,
    hq_fifo: GFifo<HQ_FIFO_DEPTH>,
    rss_fifo: GFifo<RSS_FIFO_DEPTH, MAX_ENGINES>,
    hq_busy: AtomicBool,
    rss_busy: [AtomicBool; MAX_ENGINES],
    /// Held by every producer while it pushes, and by anyone who needs the
    /// engines to stay idle (reads of drawing state, checkpoints, monitor).
    /// Holds the recording, so records come out in FIFO order.
    submit: SpinLock<Option<record::RecHandle>>,
    /// The recording again, for DMA memory accesses on the HQ3 thread (which
    /// must not wait on `submit`: its holder may be waiting for idle).
    mem_rec: SpinLock<Option<record::RecHandle>>,
    engines: AtomicBool,
    /// `bound_busy_status`: STATUS reads idle once the board has been busy
    /// for `BUSY_STATUS_BOUND` without a break.
    busy_bound: AtomicBool,
    /// When the current busy stretch began, in ns since `epoch` plus one;
    /// zero while idle (or before anyone looked).
    busy_since: AtomicU64,
    threads: SpinLock<Vec<thread::JoinHandle<()>>>,
    hq_thread: SpinLock<Option<thread::Thread>>,
    rss_thread: [SpinLock<Option<thread::Thread>>; MAX_ENGINES],
    /// The engine threads parked (`sleep`): producers unpark one only then,
    /// so a push reads a line nobody writes instead of taking the thread's
    /// slot lock. Each on its own cache line.
    hq_asleep: crossbeam_utils::CachePadded<AtomicBool>,
    rss_asleep: [crossbeam_utils::CachePadded<AtomicBool>; MAX_ENGINES],
    dirty: AtomicBool,
    running: AtomicBool,
    refresh: SpinLock<Option<thread::JoinHandle<()>>>,
    renderer: Mutex<Option<Box<dyn Renderer>>>,
    /// The finished frame, handed to the renderer as a prebuilt picture
    /// (`Rex3Screen::prebuilt`), as GR2 does.
    screen: Mutex<crate::disp::Rex3Screen>,
    screenshot_pending: AtomicBool,
    heartbeat: Arc<AtomicU64>,
    fasttick: Arc<AtomicU64>,
    cycles: SpinLock<crate::cpu::mips_core::CyclesPtr>,
    /// The other CPUs, for the status bar.
    more_cpus: SpinLock<Vec<(crate::cpu::mips_core::CyclesPtr, Arc<AtomicU64>)>>,
    /// Annotated trace (`mgras trace`); `trace_mask` gates the hot paths.
    trace: SpinLock<debug::MgrasTrace>,
    trace_mask: AtomicU32,
    pf_cfifo_level: crate::PfValue,
    pf_fifo_intr: crate::PfLevel,
}

// SAFETY: `eng` is touched only by the HQ3 thread, `rss` only by the RSS
// thread, except by holders of `submit` while the board is idle (no producer
// can push, no engine is running). The display thread reads `rss`'s
// framebuffer unlocked; tearing is tolerated, as with REX3 and GR2.
unsafe impl Sync for Mgras {}
unsafe impl Send for Mgras {}

/// The HQ3 thread's view of the rest of the board.
struct HqSink<'a>(&'a Mgras);

impl Hq3Sink for HqSink<'_> {
    fn rss_write(&mut self, r: u32, val: u32, exec: bool) {
        self.0.rss_push(rss_entry(r, exec), val as u64);
    }
    fn transfer(&mut self) -> Option<(bool, (u32, u32))> {
        self.0.drain_rss();
        // SAFETY: the RSS is drained and idle; only the HQ3 feeds it now.
        let rss = unsafe { &*self.0.rss.get() };
        Some((rss.transfer_armed()?, rss.transfer_shape()?))
    }
    fn dma_write_line(&mut self, line: u32, bytes: &[u8]) {
        // Eight bytes to a FIFO word, the first in the high byte.
        let mut vals: Vec<u64> = Vec::with_capacity(bytes.len().div_ceil(8));
        let mut words = bytes.chunks_exact(8);
        vals.extend(words.by_ref().map(|c| u64::from_be_bytes(c.try_into().unwrap())));
        if !words.remainder().is_empty() {
            let mut last = [0u8; 8];
            last[..words.remainder().len()].copy_from_slice(words.remainder());
            vals.push(u64::from_be_bytes(last));
        }
        self.0.rss_fifo.push_batch(tag::DMA_LINE, line as u64 | (bytes.len() as u64) << 32, &vals);
        self.0.wake_rss();
    }
    fn dma_read_line(&mut self, line: u32) -> Vec<u8> {
        self.0.drain_rss();
        // SAFETY: as in `transfer`.
        unsafe { &*self.0.rss.get() }.dma_read_line(line)
    }
    fn gl_bracket(&mut self, enter: bool) {
        self.0.rss_push(if enter { tag::GL_ENTER } else { tag::GL_LEAVE }, 0);
    }
    fn finish_raster(&mut self) {
        if !self.0.rss_idle() {
            crate::pf_scope!(crate::perfetto::events::MGRAS_SWAP_WAIT, self.0.rss_fifo.len());
            self.0.drain_rss();
        }
    }
    fn set_flags(&mut self, bits: u32) {
        crate::pf_instant!(crate::perfetto::events::MGRAS_FLAG_SET, (self.0.rss_fifo.len() as u64) << 32 | bits as u64);
        self.0.flags.fetch_or(bits, Ordering::AcqRel);
        self.0.refresh_general();
    }
    fn take_flags(&mut self, mask: u32) -> u32 {
        let was = self.0.flags.fetch_and(!mask, Ordering::AcqRel) & mask;
        self.0.refresh_general();
        was
    }
    fn readback_probe(&self, addr: u32) -> Option<u32> {
        self.0.probe_answer(addr)
    }
    fn sys_mem(&mut self) -> Option<Arc<dyn BusDevice>> {
        let mem = self.0.mem.lock().clone()?;
        Some(match self.0.mem_rec.lock().clone() {
            Some(rec) => Arc::new(record::RecordingMem { inner: mem, rec }),
            None => mem,
        })
    }
    fn note(&mut self, what: String) {
        self.0.note(what);
    }
    fn wants_trace(&self) -> bool {
        self.0.tracing(debug::TRACE_HQ)
    }
    fn wants_trace_tex(&self) -> bool {
        self.0.tracing(debug::TRACE_TEX)
    }
    fn trace(&mut self, line: String) {
        self.0.trace_hq(&line);
    }
}

impl Mgras {
    /// The `mgras probe` answer for GE readback address `addr`, if any.
    pub(super) fn probe_answer(&self, addr: u32) -> Option<u32> {
        self.probe.iter().find_map(|p| {
            let v = p.load(Ordering::Relaxed);
            let (lo, end) = ((v >> 48) as u32, ((v >> 32) & 0xFFFF) as u32);
            (end != 0 && (lo..end).contains(&addr)).then_some(v as u32)
        })
    }

    /// Set (`Some((first, last, answer))`) or clear (`None`) probe slots;
    /// false when all are taken.
    pub(super) fn set_probe(&self, p: Option<(u32, u32, u32)>) -> bool {
        match p {
            None => {
                for s in &self.probe {
                    s.store(0, Ordering::Relaxed);
                }
                true
            }
            Some((lo, hi, v)) => {
                let w = (lo as u64 & 0xFFFF) << 48 | ((hi as u64 + 1) & 0xFFFF) << 32 | v as u64;
                self.probe.iter().any(|s| s.compare_exchange(0, w, Ordering::Relaxed, Ordering::Relaxed).is_ok())
            }
        }
    }

    /// Build the board in place on the heap: its multi-megabyte state is
    /// zero-initialised without touching the stack, and only the fields that
    /// are not valid zeroed are written.
    /// `irqs` are the FIFO, general and retrace interrupts, as the slot
    /// wires them.
    pub fn new(board: GraphicsBoard, irqs: [crate::irq::IrqLine; 3], heartbeat: Arc<AtomicU64>, fasttick: Arc<AtomicU64>) -> Arc<Self> {
        let mut a: Arc<MaybeUninit<Self>> = Arc::new_zeroed();
        let p = Arc::get_mut(&mut a).unwrap().as_mut_ptr();
        // SAFETY: `p` points at zeroed, exclusively owned memory. Every field
        // that is not valid zeroed is written below, before assume_init.
        let m = unsafe {
            addr_of_mut!((*p).kind).write(board);
            addr_of_mut!((*p).irqs).write(irqs);
            addr_of_mut!((*p).general).write(AtomicBool::new(false));
            addr_of_mut!((*p).fifo_cfg).write(AtomicU32::new(0));
            addr_of_mut!((*p).fifo_line).write(SpinLock::new(false));
            addr_of_mut!((*p).fifo_asserted).write(AtomicBool::new(false));
            addr_of_mut!((*p).epoch).write(std::time::Instant::now());
            addr_of_mut!((*p).front_lock).write(SpinLock::new(()));
            (*(*p).front.get()).init(board);
            addr_of_mut!((*p).unhandled).write(SpinLock::new(HashSet::new()));
            addr_of_mut!((*p).flags).write(AtomicU32::new(0));
            addr_of_mut!((*p).eng).write(UnsafeCell::new(Hq3Engine::new(if board == GraphicsBoard::MaxImpact { 2 } else { 1 })));
            let planes = (*p).planes.get();
            let nrss = Self::engines_for(board);
            addr_of_mut!((*p).nrss).write(nrss);
            for id in 0..MAX_ENGINES {
                let e = match id {
                    0 => (*p).rss.get(),
                    1 => (*p).rss1.get(),
                    n => (*p).rss_more[n - 2].get(),
                };
                Rss::init(e, planes);
                // Engine id draws the rows y with y % nrss == id.
                (*e).set_engine(id as u32, (nrss - 1) as u32);
            }
            (*p).rss_fifo.set_heads(nrss);
            addr_of_mut!((*p).mem).write(SpinLock::new(None));
            addr_of_mut!((*p).submit).write(SpinLock::new(None));
            addr_of_mut!((*p).mem_rec).write(SpinLock::new(None));
            addr_of_mut!((*p).threads).write(SpinLock::new(Vec::new()));
            addr_of_mut!((*p).hq_thread).write(SpinLock::new(None));
            addr_of_mut!((*p).rss_thread).write(std::array::from_fn(|_| SpinLock::new(None)));
            addr_of_mut!((*p).dirty).write(AtomicBool::new(true));
            addr_of_mut!((*p).refresh).write(SpinLock::new(None));
            addr_of_mut!((*p).renderer).write(Mutex::new(None));
            addr_of_mut!((*p).screen).write(Mutex::new(crate::disp::Rex3Screen::new()));
            addr_of_mut!((*p).heartbeat).write(heartbeat);
            addr_of_mut!((*p).fasttick).write(fasttick);
            addr_of_mut!((*p).cycles).write(SpinLock::new(crate::cpu::mips_core::CyclesPtr::dangling()));
            addr_of_mut!((*p).more_cpus).write(SpinLock::new(Vec::new()));
            addr_of_mut!((*p).trace).write(SpinLock::new(debug::MgrasTrace::new()));
            a.assume_init()
        };
        if let Some(path) = std::env::var_os("IRIS_MGRAS_REC") {
            let path = path.to_string_lossy();
            match m.set_recording(Some(&path)) {
                Ok(()) => eprintln!("mgras: recording to {path}"),
                Err(e) => eprintln!("mgras: cannot record to {path}: {e}"),
            }
        }
        if let Some(path) = std::env::var_os("IRIS_MGRAS_TRACE") {
            let path = path.to_string_lossy();
            match m.start_trace(&path, debug::TRACE_ALL) {
                Ok(()) => eprintln!("mgras: tracing to {path}"),
                Err(e) => eprintln!("mgras: cannot trace to {path}: {e}"),
            }
        }
        m
    }

    /// Put the board's locks in the `locks` monitor listing.
    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::{register_mutex_fn, register_spin};
        register_spin("mgras::submit", self, |m| &m.submit);
        register_spin("mgras::front", self, |m| &m.front_lock);
        register_spin("mgras::fifo_line", self, |m| &m.fifo_line);
        register_spin("mgras::mem", self, |m| &m.mem);
        register_spin("mgras::mem_rec", self, |m| &m.mem_rec);
        register_spin("mgras::unhandled", self, |m| &m.unhandled);
        register_spin("mgras::threads", self, |m| &m.threads);
        register_spin("mgras::hq_thread", self, |m| &m.hq_thread);
        if self.nrss == 1 {
            register_spin("mgras::rss_thread", self, |m| &m.rss_thread[0]);
        } else {
            let slots: [fn(&Self) -> &SpinLock<Option<thread::Thread>>; MAX_ENGINES] =
                [|m| &m.rss_thread[0], |m| &m.rss_thread[1], |m| &m.rss_thread[2], |m| &m.rss_thread[3]];
            for (id, get) in slots.into_iter().enumerate().take(self.nrss) {
                register_spin(format!("mgras::rss_thread[{id}]"), self, get);
            }
        }
        register_spin("mgras::refresh", self, |m| &m.refresh);
        register_spin("mgras::cycles", self, |m| &m.cycles);
        register_spin("mgras::trace", self, |m| &m.trace);
        register_mutex_fn("mgras::renderer", self, |m| m.renderer.is_locked());
        register_mutex_fn("mgras::screen", self, |m| m.screen.is_locked());
    }

    /// The CPU-side state, locked.
    #[track_caller]
    fn front(&self) -> FrontGuard<'_> {
        let lock = self.front_lock.lock();
        // SAFETY: mutable access only under `front_lock`; the display
        // thread's unlocked reads are of plain data (see `front_view`).
        FrontGuard { _lock: lock, front: unsafe { &mut *self.front.get() } }
    }

    /// The CPU-side state without the lock, for the display thread: plain
    /// data, so a concurrent write can only tear a value.
    fn front_view(&self) -> &Front {
        // SAFETY: see above.
        unsafe { &*self.front.get() }
    }

    /// Whether a "not modelled yet" report containing `what` was made.
    #[cfg(test)]
    pub(crate) fn noted(&self, what: &str) -> bool {
        self.unhandled.lock().iter().any(|n| n.contains(what))
    }

    fn note(&self, what: String) {
        let mut u = self.unhandled.lock();
        if u.len() < 256 && u.insert(what.clone()) {
            eprintln!("mgras: not modelled yet: {what}");
        }
    }

    /// After a push: unpark the consumer if it is asleep. The fence pairs
    /// with `sleep`'s: either this sees the thread asleep, or the thread
    /// sees the push before it parks.
    fn wake(asleep: &AtomicBool, slot: &SpinLock<Option<thread::Thread>>) {
        std::sync::atomic::fence(Ordering::SeqCst);
        if asleep.load(Ordering::Relaxed) {
            if let Some(t) = slot.lock().as_ref() {
                t.unpark();
            }
        }
    }

    /// An engine thread with nothing to do: park for up to 50 ms, unless
    /// `has_work` finds a push made before `asleep` was raised. The timeout
    /// is only a backstop for a wake the fence protocol should never lose,
    /// so it is long enough that an idle engine costs ~20 wakeups/s rather
    /// than 500; a hypothetical missed wake would cost one interval of
    /// latency, never correctness.
    fn sleep(asleep: &AtomicBool, has_work: impl FnOnce() -> bool) {
        asleep.store(true, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);
        if !has_work() {
            thread::park_timeout(std::time::Duration::from_millis(50));
        }
        asleep.store(false, Ordering::Relaxed);
    }

    /// Wake the HQ3 thread.
    fn wake_hq(&self) {
        Self::wake(&self.hq_asleep, &self.hq_thread);
    }

    /// Raster engine threads for a board: Maximum Impact's two, the others'
    /// one, unless `IRIS_MGRAS_ENGINES` (1, 2 or 4) says otherwise. More
    /// threads than the board has engines split its scanlines the same way
    /// (row y to thread y % n) without changing anything the guest sees:
    /// where a row lives in pixel memory follows DRBsize (`Buffer::locate`),
    /// not the thread count, and the threads share pixel memory and TRAM.
    fn engines_for(board: GraphicsBoard) -> usize {
        match std::env::var("IRIS_MGRAS_ENGINES").ok().as_deref().map(str::trim) {
            Some("1") => 1,
            Some("2") => 2,
            Some("4") => 4,
            _ if board == GraphicsBoard::MaxImpact => 2,
            _ => 1,
        }
    }

    /// True when the raster engines have run everything queued to them.
    fn rss_idle(&self) -> bool {
        self.rss_fifo.is_empty() && self.rss_busy[..self.nrss].iter().all(|b| !b.load(Ordering::Acquire))
    }

    /// True when neither FIFO holds work and no engine is mid-entry.
    /// The HQ3 is checked first: whatever it forwarded before going idle is
    /// already in `rss_fifo` when that is checked.
    fn idle(&self) -> bool {
        self.hq_fifo.is_empty() && !self.hq_busy.load(Ordering::Acquire) && self.rss_idle()
    }

    /// Raster engine `id`'s state.
    fn engine_cell(&self, id: usize) -> &UnsafeCell<Rss> {
        match id {
            0 => &self.rss,
            1 => &self.rss1,
            n => &self.rss_more[n - 2],
        }
    }

    /// Wake the raster engine threads.
    fn wake_rss(&self) {
        for id in 0..self.nrss {
            Self::wake(&self.rss_asleep[id], &self.rss_thread[id]);
        }
    }

    /// Wait until the board is idle. Holding `submit` keeps it idle.
    fn wait_idle(&self) {
        let backoff = crossbeam_utils::Backoff::new();
        while !self.idle() {
            backoff.snooze();
        }
    }

    /// A guest polling STATUS or DMABUSY for an idle board: give a busy
    /// board up to `POLL_SETTLE` to finish first. The drivers' idle waits
    /// count polls (a microsecond delay loop between them), not time, and
    /// the model draws far slower than the board: a context switch's wait
    /// (DMABUSY, 6 x 15000 polls 10 us apart, about 0.9 s) timed out
    /// behind long drawing and the kernel killed the X server ("hq idle
    /// wait timeout"). Under jitv2 the delay loop also runs about five
    /// times short (clanker/irix/delay-loop-calibrated-cold.md). Waiting
    /// here turns each poll into board time, so the budget is counted in
    /// work done; an idle board answers at once.
    fn settle(&self) {
        if self.idle() {
            return;
        }
        let t = std::time::Instant::now();
        let backoff = crossbeam_utils::Backoff::new();
        while !self.idle() && t.elapsed() < POLL_SETTLE {
            backoff.snooze();
        }
    }

    /// HQ3 thread: wait until the engines have run everything queued.
    fn drain_rss(&self) {
        let backoff = crossbeam_utils::Backoff::new();
        while !self.rss_idle() {
            backoff.snooze();
        }
    }

    /// HQ3 thread: queue one entry for the RSS.
    fn rss_push(&self, addr: u32, val: u64) {
        self.rss_fifo.push(addr, val);
        self.wake_rss();
    }

    /// Everything a replay must reproduce, as one byte string, little-endian
    /// u32s: the displayed main buffer's bottom-left 1280x1024 (row 0 at the
    /// bottom), the overlay's, colormap 0, the 32 main XMAP modes, the video
    /// timing chip's registers and SRAM, then, only if any other bit of
    /// pixel memory is set, the hash of the rest, then nonzero clipping IDs.
    /// (This is the byte string
    /// the old 1280x1024 plane model hashed, so older recordings keep their
    /// hashes.) Call with `submit` held and the board idle.
    fn state_bytes(&self) -> Vec<u8> {
        const RW: u32 = 1280;
        const RH: u32 = 1024;
        // SAFETY: idle, and the caller holds `submit` (see the Sync impl).
        let rss = unsafe { &*self.rss.get() };
        let f = self.front();
        let (main, overlay) = frame::scanout_buffers(rss, &f.dcb);
        let mut out = Vec::with_capacity((2 * (RW * RH) as usize + 8192 + 32 + 32 + 0x8000) * 4 + 32);
        let mut put = |v: u32| out.extend_from_slice(&v.to_le_bytes());
        for b in [Some(main), overlay] {
            for y in 0..RH {
                for x in 0..RW {
                    put(b.map_or(0, |b| rss.mem.get(&b, x, y) as u32));
                }
            }
        }
        for v in f.dcb.cmap[0].pal.iter() {
            put(*v);
        }
        for d in 0..32 {
            put(f.dcb.xmap.main_mode(d));
        }
        for v in f.dcb.vc3.regs.iter().chain(f.dcb.vc3.sram.iter()) {
            put(*v as u32);
        }
        // The rest: pixel memory with the bits hashed above cleared.
        let mut rest = plain::boxed_zeroed::<pixmem::PixMem>();
        rest.words.copy_from_slice(&rss.mem.words);
        for b in [Some(main), overlay].into_iter().flatten() {
            for y in 0..RH {
                for x in 0..RW {
                    rest.put(&b, x, y, 0);
                }
            }
        }
        if rest.words.iter().any(|&w| w != 0) {
            // The second raster engine's memory only when it holds anything,
            // so one-engine recordings keep their hashes.
            let (re0, re1) = rest.words.split_at(pixmem::RE_WORDS);
            let n = if re1.iter().any(|&w| w != 0) { rest.words.len() } else { re0.len() };
            let mut h = blake3::Hasher::new();
            for w in rest.words[..n].iter() {
                h.update(&w.to_le_bytes());
            }
            out.extend_from_slice(h.finalize().as_bytes());
        }
        // Clipping IDs affect future rendering even when the colour planes
        // are identical. Keep zero-CID recordings' historical hashes.
        if rss.cid.iter().any(|&c| c != 0) {
            out.extend_from_slice(b"CID\0");
            out.extend_from_slice(blake3::hash(&rss.cid[..]).as_bytes());
        }
        out
    }

    /// Hash of `state_bytes`, waiting for the board to be idle. Call with
    /// `submit` held.
    fn idle_state_hash(&self) -> [u8; 32] {
        self.wait_idle();
        *blake3::hash(&self.state_bytes()).as_bytes()
    }

    /// Start a recording at `path` (checkpointing the state it starts from),
    /// or stop the running one with `None` (checkpointing where it ends).
    fn set_recording(&self, path: Option<&str>) -> std::io::Result<()> {
        let mut sub = self.submit.lock();
        if let Some(rec) = sub.take() {
            *self.mem_rec.lock() = None;
            let h = self.idle_state_hash();
            let mut r = rec.lock();
            r.put(record::Rec::Hash(h));
            r.flush();
        }
        if let Some(p) = path {
            let rec = record::Recorder::create(p)?;
            rec.lock().put(record::Rec::Hash(self.idle_state_hash()));
            *self.mem_rec.lock() = Some(rec.clone());
            *sub = Some(rec);
        }
        Ok(())
    }

    /// Checkpoint the running recording now. Returns the hash and record
    /// count, or None when not recording.
    fn record_mark(&self) -> Option<([u8; 32], u64)> {
        let sub = self.submit.lock();
        let rec = sub.as_ref()?;
        let h = self.idle_state_hash();
        let mut r = rec.lock();
        r.put(record::Rec::Hash(h));
        r.flush();
        Some((h, r.records))
    }

    /// Composite a host GL frame at a screen position (see `disp::composite`).
    pub fn composite(&self, x: i32, y: i32, bgra: &[u8], stride: usize, w: usize, h: usize) -> bool {
        let sub = self.submit.lock();
        self.wait_idle();
        // SAFETY: idle with `submit` held.
        let rss = unsafe { &mut *self.rss.get() };
        let ok = disp::composite(rss, &self.front().dcb, x, y, bgra, stride, w, h);
        if ok {
            if let Some(rec) = sub.as_ref() {
                rec.lock().put(record::Rec::Composite);
            }
            self.dirty.store(true, Ordering::Release);
        }
        ok
    }

    /// A colour-index window's host GL frame, as indices (`disp::composite_index`).
    #[allow(clippy::too_many_arguments)]
    pub fn composite_index(&self, x: i32, y: i32, bgra: &[u8], stride: usize, w: usize, h: usize, overlay: bool) -> bool {
        let sub = self.submit.lock();
        self.wait_idle();
        // SAFETY: idle with `submit` held.
        let rss = unsafe { &mut *self.rss.get() };
        let ok = disp::composite_index(rss, &self.front().dcb, x, y, bgra, stride, w, h, overlay);
        if ok {
            if let Some(rec) = sub.as_ref() {
                rec.lock().put(record::Rec::Composite);
            }
            self.dirty.store(true, Ordering::Release);
        }
        ok
    }


    /// Framebuffer pixel (its 24 colour planes) at display (`x`, `y`), y = 0
    /// the top row, once the board has run everything queued (tests).
    #[cfg(test)]
    pub(crate) fn fb_pixel(&self, x: usize, y: usize) -> u32 {
        let _sub = self.submit.lock();
        self.wait_idle();
        // SAFETY: idle with `submit` held.
        let dcb = &self.front().dcb;
        let h = frame::display_size(dcb).1;
        let rss = unsafe { &*self.rss.get() };
        let (main, _) = frame::scanout_buffers(rss, dcb);
        // The colour planes (alpha, in the top byte, is not displayed).
        rss.mem.get(&main, x as u32, (h - 1 - y) as u32) as u32 & 0xFF_FFFF
    }

    /// The VC3's display size, from its timing tables and its DID table.
    #[cfg(test)]
    pub(crate) fn vc3_sizes(&self) -> (Option<(usize, usize)>, usize) {
        let f = self.front();
        (f.dcb.vc3.timing_size(), f.dcb.vc3.did_lines())
    }

    /// Hash of the state a replay must reproduce (`state_bytes`).
    #[cfg(test)]
    pub(crate) fn state_hash(&self) -> [u8; 32] {
        let _sub = self.submit.lock();
        self.idle_state_hash()
    }

    /// A frame snapshot of what is displayed now (monitor, screenshots).
    fn snapshot_frame(&self) -> Box<frame::Frame> {
        let mut f = plain::boxed_zeroed::<frame::Frame>();
        // SAFETY: a read-only view; tearing tolerated.
        let rss = unsafe { &*self.rss.get() };
        f.snapshot(rss, &self.front().dcb);
        f
    }

    /// Save the displayed frame as a PNG.
    pub(crate) fn save_shot(&self, path: &str) -> Result<(), String> {
        let mut out = vec![0u32; frame::OUT_STRIDE * frame::H];
        let f = self.snapshot_frame();
        f.compose(&mut out);
        disp::save_png(path, &out, f.width, f.height)
    }

    /// Identity for a board outside the Indigo2 (the Octane's XIO boards):
    /// the two board-version bytes, the colormaps' revision registers and
    /// the GE revision, which the readback registers hold from now until
    /// the first state query or engine start replaces it.
    pub fn set_identity(&self, bdvers: [u32; 2], cmap_rev: [u32; 2], ge_revision: u32) {
        let mut f = self.front();
        f.dcb.init(bdvers, cmap_rev);
        f.ge_revision = ge_revision;
        f.hq.ge_readback = [0, ge_revision];
    }

    /// Start board-to-host DMA in the Octane's HQ4 order (see
    /// `Hq3Engine::kick_starts_read`). Call before the engines start.
    pub fn start_reads_at_kick(&self) {
        assert!(!self.engines.load(Ordering::Acquire), "set before the engines start");
        // SAFETY: the HQ3 thread is not running.
        unsafe { &mut *self.eng.get() }.kick_starts_read = true;
    }

    /// Let STATUS's idle bits read set once the board has been busy for
    /// `BUSY_STATUS_BOUND` without a break. The model draws far slower than
    /// the board, and the IP30 kernel gives up waiting on those bits after
    /// about 2 to 3 s and kills the X server over its idle wait. A
    /// read that then goes ahead still waits for the work at its data
    /// register. An approximation: the board would be idle by then.
    pub fn bound_busy_status(&self) {
        self.busy_bound.store(true, Ordering::Relaxed);
    }

    /// Give a board outside the Indigo2 its hooks into the command
    /// handling. Call before the engines start.
    pub fn set_board(&self, board: Box<dyn Hq3Board>) {
        assert!(!self.engines.load(Ordering::Acquire), "set before the engines start");
        // SAFETY: the HQ3 thread is not running.
        unsafe { &mut *self.eng.get() }.board = Some(board);
    }

    /// Give the board its path to system memory, for DMA.
    pub fn set_phys(&self, mem: Arc<dyn BusDevice>) {
        *self.mem.lock() = Some(mem);
    }

    /// Drive one of the board's interrupt lines.
    fn set_line(&self, line: Line, active: bool) {
        let n = match line {
            Line::Fifo => 0,
            Line::General => 1,
            Line::Retrace => 2,
        };
        crate::pf_instant!(crate::perfetto::events::MGRAS_IRQ_LINE, n << 1 | active as usize);
        self.irqs[n].set(active);
    }

    /// Wire up the CPU cycle counter for the status bar's MIPS figure.
    pub fn set_cpu_cycles(&self, ptr: crate::cpu::mips_core::CyclesPtr) {
        *self.cycles.lock() = ptr;
    }

    /// One more CPU for the status bar.
    pub fn add_cpu_stats(&self, cycles: crate::cpu::mips_core::CyclesPtr, fasttick: Arc<AtomicU64>) {
        self.more_cpus.lock().push((cycles, fasttick));
    }

    fn more_cpu_stats(&self) -> [Option<(u64, u64)>; crate::disp::MORE_CPUS] {
        let more = self.more_cpus.lock();
        std::array::from_fn(|i| more.get(i).map(|(c, f)| (c.get(), f.load(Ordering::Relaxed))))
    }

    fn offset(&self, addr: u64) -> Option<u32> {
        let off = addr.wrapping_sub(MGRAS_SLOT_GFX_BASE as u64);
        (off < MAP_SIZE as u64).then_some(off as u32)
    }

    /// Flags as the host reads them.
    fn all_flags(&self, f: &Front) -> u32 {
        self.flags.load(Ordering::Acquire) | f.derived_flags()
    }

    /// Bring the general interrupt line (GIO line 1) in line with the flags:
    /// asserted while some enabled flag is set, a level held until the
    /// handler clears the flag or its enable. Decided and driven under the
    /// front lock, so two threads cannot leave a stale level behind.
    fn refresh_general(&self) {
        let f = self.front();
        let level = self.all_flags(&f) & f.hq.interrupt_enable & host::INTR_CAUSES != 0;
        if self.general.swap(level, Ordering::AcqRel) != level {
            self.set_line(Line::General, level);
        }
    }

    /// Reads whose answer depends on what the engines have done wait for
    /// them (bus busy): raster state for the raster engines too, HQ3 state
    /// only for the HQ3. Status registers answer at once (`status_read`):
    /// the drivers poll them with their own timeouts and sleep on the FIFO
    /// interrupt when the board stays busy, so making the read itself wait
    /// would hold the CPU in one load, often with interrupts masked.
    fn read_wait(&self, off: u32) -> bool {
        match off {
            host::RASTER..=0x7FFFF => !self.idle(),
            host::GE_READBACK_HI | host::GE_READBACK_LO
            | host::RASTER_IF_CONTEXT..=0x50228 | host::DMA_CONTEXT..=0x504FF => !self.hq_idle(),
            _ => false,
        }
    }

    /// With `bound_busy_status`, whether the board has been busy without a
    /// break for longer than `BUSY_STATUS_BOUND`, timed from the store that
    /// took it from idle to busy (or from the first STATUS read that saw it
    /// busy, if none did).
    fn busy_overdue(&self, idle: bool) -> bool {
        if !self.busy_bound.load(Ordering::Relaxed) {
            return false;
        }
        if idle {
            self.busy_since.store(0, Ordering::Relaxed);
            return false;
        }
        let now = self.epoch.elapsed().as_nanos() as u64 + 1;
        let since = match self.busy_since.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => now,
            Err(t) => t,
        };
        now.saturating_sub(since) > BUSY_STATUS_BOUND.as_nanos() as u64
    }

    /// The HQ3 has run every command it was given.
    fn hq_idle(&self) -> bool {
        self.hq_fifo.is_empty() && !self.hq_busy.load(Ordering::Acquire)
    }

    /// The command FIFO's level as FIFOSTATUS reports it, in words. The
    /// real CFIFO holds a few dozen; ours holds `HQ_FIFO_DEPTH`. As with
    /// REX3's GFIFO, anything queued reads 1 until the queue is within
    /// `CFIFO_WORDS` of really full, then the level climbs with it, so the
    /// drivers' high-water handling starts only when a writer is about to
    /// be held off for real.
    ///
    /// It reads 0 only once the whole board is idle, not when the HQ3 has
    /// drained: on the board an empty CFIFO means the shallow pipe behind
    /// it finishes within microseconds, and the kernel relies on that. It
    /// flushes to level 0 before a buffer swap (an immediate DCB write)
    /// and before a context switch, then polls DMABUSY for at most ~70 ms;
    /// with our deep `rss_fifo` still drawing, swaps showed half-drawn
    /// frames and the context switch failed ("FlushHQ (rss_sh) failed for
    /// ctxsw").
    fn cfifo_level(&self) -> u32 {
        if self.idle() {
            return 0;
        }
        let n = self.hq_fifo.len();
        (n.saturating_sub(HQ_FIFO_DEPTH - 1 - CFIFO_WORDS) as u32).clamp(1, CFIFO_WORDS as u32)
    }

    /// STATUS, FIFOSTATUS, GIOSTATUS and DMABUSY from the live state.
    fn status_read(&self, off: u32) -> u32 {
        let cfg = self.fifo_cfg.load(Ordering::Acquire);
        let level = self.cfifo_level();
        crate::pf_value!(self.pf_cfifo_level, crate::perfetto::events::MGRAS_CFIFO_LEVEL, level);
        match off {
            host::STATUS => {
                let mut v = host::STATUS_FIXED | self.vblank();
                let idle = self.idle();
                if self.hq_idle() {
                    v |= host::STATUS_HQ_IDLE;
                }
                if idle {
                    v |= host::STATUS_BOARD_IDLE;
                }
                if self.busy_overdue(idle) {
                    v |= host::STATUS_HQ_IDLE | host::STATUS_BOARD_IDLE;
                }
                if Self::cfifo_high(cfg, level) {
                    v |= host::STATUS_CFIFO_HIGH;
                }
                if Self::cfifo_low(cfg, level) {
                    v |= host::STATUS_CFIFO_LOW;
                }
                crate::pf_instant!(crate::perfetto::events::MGRAS_STATUS_READ, v);
                v
            }
            host::FIFOSTATUS => level | (cfg & 0x7F) << host::FIFOSTATUS_HIGH_SHIFT,
            host::DMABUSY => {
                let mut v = 0;
                if !self.hq_fifo.is_empty() {
                    v |= host::DMABUSY_CFIFO;
                }
                if self.hq_busy.load(Ordering::Acquire) {
                    v |= host::DMABUSY_HIF;
                }
                if !self.rss_idle() {
                    v |= host::DMABUSY_REIF;
                }
                v
            }
            _ => 0,
        }
    }

    /// Fingerprints of the microcode the kernel loaded (see `microcode`),
    /// read without the front lock: a download in progress gives a value
    /// that does not repeat.
    pub fn microcode_fingerprints(&self) -> microcode::Fingerprints {
        let f = self.front_view();
        let up = |ge: usize| self.ge_upload_fingerprint[ge].load(Ordering::Relaxed);
        microcode::Fingerprints {
            hq: f.hq.ucode_hash.get(),
            ge: [0, 1].map(|ge| match up(ge) {
                0 => f.ge.fingerprint(ge),
                fp => fp,
            }),
        }
    }

    /// A front-end's own copy of GE11 program `ge` has this fingerprint.
    pub fn set_ge_upload_fingerprint(&self, ge: usize, fp: u64) {
        self.ge_upload_fingerprint[ge & 1].store(fp, Ordering::Relaxed);
    }

    /// STATUS's vertical blank bit: the first 1 ms of the refresh thread's
    /// frame, the time of the retrace pulse.
    fn vblank(&self) -> u32 {
        let now = self.epoch.elapsed().as_nanos() as u64;
        if now.wrapping_sub(self.frame_start.load(Ordering::Relaxed)) < 1_000_000 { host::STATUS_VBLANK } else { 0 }
    }

    /// Mark the start of a displayed frame (`vblank`).
    fn begin_frame(&self) {
        self.frame_start.store(self.epoch.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// At or above the high-water mark (a mark of 0 is not programmed yet).
    fn cfifo_high(cfg: u32, level: u32) -> bool {
        let hw = cfg & 0x7F;
        hw != 0 && level >= hw
    }

    /// Below the low-water mark.
    fn cfifo_low(cfg: u32, level: u32) -> bool {
        level < (cfg >> 8) & 0x7F
    }

    /// Take the FIFO interrupt's enables and marks from the registers
    /// after the CPU wrote one of them.
    fn sync_fifo_cfg(&self, f: &Front) {
        const ENABLES: u32 = host::INTR_CFIFO_HIGH | host::INTR_CFIFO_LOW | host::INTR_DFIFO_HIGH | host::INTR_DFIFO_LOW;
        let cfg = (f.hq.interrupt_enable & ENABLES)
            | (f.hq.regs.get(host::CFIFO_HIGH_MARK) & 0x7F)
            | (f.hq.regs.get(host::CFIFO_LOW_MARK) & 0x7F) << 8;
        self.fifo_cfg.store(cfg, Ordering::Release);
    }

    /// Whether the FIFO line can change as the HQ3 drains the FIFO: a
    /// low-water interrupt is enabled, or the line is up.
    fn fifo_watched(&self) -> bool {
        self.fifo_cfg.load(Ordering::Relaxed) & host::INTR_CFIFO_LOW != 0 || self.fifo_asserted.load(Ordering::Relaxed)
    }

    /// Bring the FIFO interrupt line (GIO interrupt 0) in line with the
    /// command FIFO: asserted while an enabled water-mark condition holds.
    /// The data FIFO is not modelled; it never fills.
    fn refresh_fifo(&self) {
        let want = |cfg: u32, level: u32| {
            cfg & host::INTR_CFIFO_HIGH != 0 && Self::cfifo_high(cfg, level)
                || cfg & host::INTR_CFIFO_LOW != 0 && Self::cfifo_low(cfg, level)
        };
        // Most calls change nothing: check without the lock first.
        let level = self.cfifo_level();
        if want(self.fifo_cfg.load(Ordering::Acquire), level) == self.fifo_asserted.load(Ordering::Acquire) {
            return;
        }
        let mut line = self.fifo_line.lock();
        let level = want(self.fifo_cfg.load(Ordering::Acquire), self.cfifo_level());
        if *line != level {
            *line = level;
            self.fifo_asserted.store(level, Ordering::Release);
            crate::pf_level!(self.pf_fifo_intr, crate::perfetto::events::MGRAS_FIFO_INTR, level);
            self.set_line(Line::Fifo, level);
        }
    }

    /// One read, with `submit` held. None means "busy, retry".
    fn read_locked(&self, off: u32, bits: u32) -> Option<u64> {
        if self.read_wait(off) {
            return None;
        }
        // SAFETY (for the engines below): what `read_wait` waited for is
        // idle, with `submit` held.
        let v = match off {
            host::STATUS | host::FIFOSTATUS | host::GIOSTATUS | host::DMABUSY => self.status_read(off) as u64,
            host::SET_FLAGS | host::CLEAR_FLAGS | host::SET_FLAGS_PRIVILEGED | host::CLEAR_FLAGS_PRIVILEGED => {
                self.all_flags(&self.front()) as u64
            }
            host::RASTER_IF_CONTEXT..=0x50228 => {
                unsafe { &*self.eng.get() }.raster_if_regs[((off - host::RASTER_IF_CONTEXT) / 4) as usize] as u64
            }
            host::DMA_CONTEXT..=0x504FF => {
                unsafe { &*self.eng.get() }.dma_regs[((off - host::DMA_CONTEXT) / 4) as usize] as u64
            }
            host::PIO_READ_HI => {
                // Taking the high half moves the stream on; the RSS does
                // that, in order (the FIFO is empty, so this fits).
                let v = unsafe { &*self.rss.get() }.pio_peek_hi();
                self.hq_fifo.push(tag::PIO_ADVANCE, 0);
                self.wake_hq();
                v as u64
            }
            host::PIO_READ_LO => unsafe { &*self.rss.get() }.pio_read_lo() as u64,
            // A state readback's answer, once the HQ has given one; before
            // that, the diagnostic port's.
            host::GE_READBACK_HI | host::GE_READBACK_LO => match unsafe { &*self.eng.get() }.ge_return {
                Some(r) => r[(off == host::GE_READBACK_LO) as usize] as u64,
                None => self.front().read(off, bits),
            },
            host::RASTER..=0x7FFFF => {
                let rss = unsafe { &*self.rss.get() };
                let r = (off & 0xFFC) >> 2;
                if bits == 64 {
                    ((rss.read(r) as u64) << 32) | rss.read(r + 1) as u64
                } else {
                    rss.read(r) as u64
                }
            }
            _ => self.front().read(off, bits),
        };
        Some(v)
    }

    /// One write, with `submit` held. False means "busy, retry".
    fn write_locked(&self, off: u32, bits: u32, val: u64) -> bool {
        let (hi, lo) = ((val >> 32) as u32, val as u32);
        match off {
            host::CFIFO | 0x70084 | host::CFIFO_GL | host::CFIFO_PRIVILEGED | 0x50084 => {
                let port = if off >= host::STATUS { tag::PORT_USER } else { tag::PORT_PRIVILEGED };
                let ok = if bits == 64 {
                    // Both words or neither: a retried store must not push
                    // the first twice.
                    self.hq_fifo.try_push2(port, hi as u64, port, lo as u64)
                } else {
                    self.hq_fifo.try_push(port, lo as u64)
                };
                if !ok {
                    return false;
                }
                self.wake_hq();
            }
            host::RASTER..=0x7FFFF => {
                let r = (off & 0xFFC) >> 2;
                // Registers at 0x7C000 + 4r; the same register at +0x1000
                // also executes the primitive in the IR after the write.
                let exec = off & 0x1000 != 0;
                let ok = if bits == 64 {
                    self.hq_fifo.try_push2(tag::RSS | rss_entry(r, false), hi as u64, tag::RSS | rss_entry(r + 1, exec), lo as u64)
                } else {
                    self.hq_fifo.try_push(tag::RSS | rss_entry(r, exec), lo as u64)
                };
                if !ok {
                    return false;
                }
                self.wake_hq();
            }
            host::CONTEXT_SWITCH => {
                // The save phase completes at once; the incoming context
                // follows through the FIFO, after this marker.
                if !self.hq_fifo.try_push(tag::CONTEXT_SWITCH, 0) {
                    return false;
                }
                self.flags.fetch_or(host::FLAG_CONTEXT_SAVED, Ordering::AcqRel);
                self.wake_hq();
            }
            host::SET_FLAGS | host::SET_FLAGS_PRIVILEGED => {
                crate::pf_instant!(crate::perfetto::events::MGRAS_FLAG_SET, (self.rss_fifo.len() as u64) << 32 | lo as u64);
                self.flags.fetch_or(lo, Ordering::AcqRel);
            }
            host::CLEAR_FLAGS | host::CLEAR_FLAGS_PRIVILEGED => {
                crate::pf_instant!(crate::perfetto::events::MGRAS_FLAG_CLEAR, lo);
                self.flags.fetch_and(!lo, Ordering::AcqRel);
            }
            0x80000..=0xFFFFF => self.note(format!("fast-path command window write at {off:#x}")),
            _ => {
                let mut f = self.front();
                if f.write(off, bits, val, &self.flags) {
                    self.dirty.store(true, Ordering::Release);
                }
                if matches!(off, host::INTERRUPT_ENABLE_SET | host::INTERRUPT_ENABLE_CLEAR | host::CFIFO_HIGH_MARK | host::CFIFO_LOW_MARK) {
                    self.sync_fifo_cfg(&f);
                    drop(f);
                    self.refresh_fifo();
                }
                return true;
            }
        }
        // A FIFO push: the level may have crossed a water mark.
        self.refresh_fifo();
        true
    }

    /// Writes that go through `hq_fifo` (traced as HQ/RSS lines instead).
    fn is_fifo_write(off: u32) -> bool {
        matches!(off, host::CFIFO | 0x70084 | host::CFIFO_GL | host::CFIFO_PRIVILEGED | 0x50084 | host::RASTER..=0x7FFFF)
    }

    fn do_read(&self, addr: u64, bits: u32) -> Option<u64> {
        let Some(off) = self.offset(addr) else { return Some(0) };
        if matches!(off, host::STATUS | host::DMABUSY) {
            self.settle();
        }
        let sub = self.submit.lock();
        let v = self.read_locked(off, bits);
        crate::pf_level!(self.pf_wait, crate::perfetto::events::MGRAS_WAIT, v.is_none());
        let v = v?;
        if let Some(rec) = sub.as_ref() {
            rec.lock().put(record::Rec::Read { bits: bits as u8, off, val: v });
        }
        drop(sub);
        if self.tracing(debug::TRACE_CPU) {
            self.trace_cpu(false, off, v);
        }
        self.refresh_general();
        Some(v)
    }

    fn do_write(&self, addr: u64, bits: u32, val: u64) -> u32 {
        let Some(off) = self.offset(addr) else { return BUS_OK };
        let sub = self.submit.lock();
        // A store into an idle board may start a busy stretch: time the
        // bound from here, not from some earlier stretch no STATUS read saw end.
        if self.busy_bound.load(Ordering::Relaxed) && self.idle() {
            self.busy_since.store(self.epoch.elapsed().as_nanos() as u64 + 1, Ordering::Relaxed);
        }
        let ok = self.write_locked(off, bits, val);
        crate::pf_level!(self.pf_fifo_full, crate::perfetto::events::MGRAS_FIFO_FULL, !ok);
        if !ok {
            return BUS_BUSY;
        }
        if let Some(rec) = sub.as_ref() {
            let due = {
                let mut r = rec.lock();
                r.put(record::Rec::Write { bits: bits as u8, off, val });
                r.checkpoint_due()
            };
            if due {
                let h = self.idle_state_hash();
                rec.lock().put(record::Rec::Hash(h));
            }
        }
        drop(sub);
        if self.tracing(debug::TRACE_CPU) && !Self::is_fifo_write(off) {
            self.trace_cpu(true, off, val);
        }
        self.refresh_general();
        BUS_OK
    }

    // ── engine threads ───────────────────────────────────────────────────────

    fn hq_loop(&self) {
        *self.hq_thread.lock() = Some(thread::current());
        let mut sink = HqSink(self);
        let backoff = crossbeam_utils::Backoff::new();
        let pf_busy = crate::PfLevel::new();
        let mut taken = 0u32;
        // The busy flags share a cache line with the engines': store them
        // only when they change, not once an entry.
        let mut busy = false;
        loop {
            let Some((t, val)) = self.hq_fifo.peek() else {
                self.hq_fifo.flush_head();
                if busy {
                    busy = false;
                    self.hq_busy.store(false, Ordering::Release);
                    crate::pf_level!(pf_busy, crate::perfetto::events::MGRAS_HQ_BUSY, false);
                }
                // Drained: a low-water interrupt may be due (the kernel
                // sleeps on one until the FIFO empties).
                if self.fifo_watched() {
                    self.refresh_fifo();
                }
                if backoff.is_completed() {
                    Self::sleep(&self.hq_asleep, || self.hq_fifo.peek().is_some());
                } else {
                    backoff.snooze();
                }
                continue;
            };
            if !busy {
                busy = true;
                self.hq_busy.store(true, Ordering::Release);
                crate::pf_level!(pf_busy, crate::perfetto::events::MGRAS_HQ_BUSY, true);
            }
            // SAFETY: the HQ3 thread owns the engine.
            let eng = unsafe { &mut *self.eng.get() };
            match t {
                tag::EXIT => break,
                tag::PORT_USER | tag::PORT_PRIVILEGED => eng.push(t as usize, val as u32, &mut sink),
                tag::CONTEXT_SWITCH => eng.begin_context_switch(&mut sink),
                tag::PIO_ADVANCE => self.rss_push(t, 0),
                t if t & !0x7FF == tag::RSS => self.rss_push(t & 0x7FF | RSS_SRC_CPU, val),
                _ => {}
            }
            self.hq_fifo.consume();
            backoff.reset();
            // The head is published every 64 entries; follow the level as
            // it falls while the FIFO interrupt cares.
            taken = taken.wrapping_add(1);
            if taken % 64 == 0 && self.fifo_watched() {
                self.refresh_fifo();
            }
        }
        self.hq_fifo.consume();
        self.hq_fifo.flush_head();
        self.hq_busy.store(false, Ordering::Release);
    }

    /// Raster engine `id`'s thread: every `rss_fifo` entry, through head
    /// `id`. With two engines both see the same stream and keep the same
    /// register state; each draws only its own scanlines. Nothing they draw
    /// reads another scanline (blending, depth and stencil read the pixel
    /// being drawn; screen copies go through the host), so they never wait
    /// for each other, except at a PIO read advance: engine 0 reads the
    /// next doubleword from both engines' rows there, so engine 1 must not
    /// draw past it until engine 0 has. (Everything before it is drawn: the
    /// CPU reads only with the board idle.) Engine 0 does the counting,
    /// tracing and statistics the monitor shows.
    fn rss_loop(&self, id: usize) {
        *self.rss_thread[id].lock() = Some(thread::current());
        let backoff = crossbeam_utils::Backoff::new();
        let mut stale_seen = 0u32;
        let mut advances = 0u64;
        let mut payload: Vec<u64> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();
        let cell = self.engine_cell(id);
        let busy = &self.rss_busy[id];
        let pf_busy = crate::PfLevel::new();
        // Stored only when it changes (see `hq_loop`).
        let mut is_busy = false;
        loop {
            let Some((t, val)) = self.rss_fifo.peek_at(id) else {
                self.rss_fifo.flush_head_at(id);
                if is_busy {
                    is_busy = false;
                    busy.store(false, Ordering::Release);
                    crate::pf_level!(pf_busy, crate::perfetto::events::MGRAS_RSS_BUSY, false);
                }
                // The board may have just gone idle: the command FIFO's
                // level drops to 0 (`cfifo_level`), which a low-water
                // interrupt may be waiting for.
                if self.fifo_watched() {
                    self.refresh_fifo();
                }
                if backoff.is_completed() {
                    Self::sleep(&self.rss_asleep[id], || self.rss_fifo.peek_at(id).is_some());
                } else {
                    backoff.snooze();
                }
                continue;
            };
            if !is_busy {
                is_busy = true;
                busy.store(true, Ordering::Release);
                crate::pf_level!(pf_busy, crate::perfetto::events::MGRAS_RSS_BUSY, true);
            }
            // SAFETY: this engine's thread owns its raster state.
            let rss = unsafe { &mut *cell.get() };
            // DMA lines never read the texture engine; anything else may.
            if t != tag::DMA_LINE {
                rss.te_settle();
            }
            let changed = match t {
                tag::EXIT => break,
                t if t < 0x1000 => rss.write((t & 0x7FF) >> 1, val as u32, t & 1 != 0),
                tag::PIO_ADVANCE => {
                    advances += 1;
                    if id == 0 {
                        rss.pio_read_hi();
                        self.pio_advances.store(advances, Ordering::Release);
                    } else {
                        let wait = crossbeam_utils::Backoff::new();
                        while self.pio_advances.load(Ordering::Acquire) < advances {
                            wait.snooze();
                        }
                        rss.pio_read_hi();
                    }
                    false
                }
                tag::GL_ENTER => {
                    rss.gl_enter();
                    false
                }
                tag::GL_LEAVE => {
                    rss.gl_leave();
                    false
                }
                tag::DMA_LINE => {
                    let (line, len) = (val as u32, (val >> 32) as usize);
                    payload.resize(len.div_ceil(8), 0);
                    let n = payload.len();
                    self.rss_fifo.drain_payload_at(id, n, &mut payload[..]);
                    bytes.resize(8 * n, 0);
                    for (b, w) in bytes.chunks_exact_mut(8).zip(&payload) {
                        b.copy_from_slice(&w.to_be_bytes());
                    }
                    rss.dma_write_line(line, &bytes[..len]);
                    true
                }
                _ => false,
            };
            // The refresh thread clears it once a frame; until then a load
            // keeps the cache line shared instead of taking it each entry.
            if changed && !self.dirty.load(Ordering::Relaxed) {
                self.dirty.store(true, Ordering::Release);
            }
            if id == 0 {
                if self.tracing(debug::TRACE_RSS) {
                    self.trace_rss(t, val, rss, changed);
                }
                if rss.stale_events != stale_seen {
                    stale_seen = rss.stale_events;
                    if self.tracing(debug::TRACE_TEX) {
                        let [want, have] = rss.stale_last;
                        self.trace_hq(&format!("TEX stale sample #{stale_seen}: want {want:#x}, page holds {have:#x}"));
                    }
                }
            }
            self.rss_fifo.consume_at(id);
            backoff.reset();
        }
        self.rss_fifo.consume_at(id);
        self.rss_fifo.flush_head_at(id);
        busy.store(false, Ordering::Release);
    }

    /// Start the HQ3 and RSS threads.
    pub fn start_engines(self: &Arc<Self>) {
        if self.engines.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut threads = self.threads.lock();
        let me = Arc::clone(self);
        threads.push(thread::Builder::new().name("MGRAS-HQ3".into()).spawn(move || me.hq_loop()).expect("spawn MGRAS HQ3 thread"));
        self.pio_advances.store(0, Ordering::Release);
        // SAFETY: the engine threads are not running yet.
        unsafe {
            for id in 0..self.nrss {
                (*self.engine_cell(id).get()).share_te(self.nrss as u32);
            }
        }
        for id in 0..self.nrss {
            let me = Arc::clone(self);
            let name = if self.nrss == 1 { "MGRAS-RSS".to_string() } else { format!("MGRAS-RSS{id}") };
            threads.push(thread::Builder::new().name(name).spawn(move || me.rss_loop(id)).expect("spawn MGRAS RSS thread"));
        }
    }

    /// Stop the engine threads once they have run everything queued.
    pub fn stop_engines(&self) {
        if !self.engines.swap(false, Ordering::AcqRel) {
            return;
        }
        {
            let _sub = self.submit.lock();
            self.hq_fifo.push(tag::EXIT, 0);
            self.wake_hq();
        }
        let mut threads = self.threads.lock();
        if let Some(h) = (!threads.is_empty()).then(|| threads.remove(0)) {
            let _ = h.join();
        }
        self.rss_fifo.push(tag::EXIT, 0);
        self.wake_rss();
        for h in threads.drain(..) {
            let _ = h.join();
        }
        self.hq_fifo.reset();
        self.rss_fifo.reset();
    }
}

impl crate::gfx_display::GfxDisplay for Mgras {
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { &self.renderer }
    fn screen(&self) -> &Mutex<crate::disp::Rex3Screen> { &self.screen }
    fn request_screenshot(&self) { self.screenshot_pending.store(true, Ordering::Relaxed); }
    fn cycles(&self) -> crate::cpu::mips_core::CyclesPtr { *self.cycles.lock() }
    fn hw_cursor(&self) -> Option<crate::gfx_display::HwCursor> {
        // A racy read of the VC3, as the display thread's snapshot does.
        let vc3 = &self.front_view().dcb.vc3;
        let (x, y, size, glyph) = vc3.cursor()?;
        // The glyph layout `frame.rs` draws: two planes, MSB leftmost.
        let words_per_row = size / 16;
        let plane_words = size * words_per_row;
        let word = |plane: usize, row: usize, w: usize| vc3.sram[(glyph + plane * plane_words + row * words_per_row + w) & 0x7FFF] as u64;
        let mask = (0..size).map(|row| (0..words_per_row).fold(0u64, |m, w| {
            m | ((word(0, row, w) | word(1, row, w)).reverse_bits() >> 48) << (w * 16)
        })).collect();
        Some(crate::gfx_display::HwCursor { x, y, size, mask })
    }
}

impl Device for Mgras {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("mgras".into(), "IMPACT graphics: status|hq|gl|ge|ucode|vc3|xmap|cmap|dac|pix|stats|jit|fbdump|shot|dump|trace|rec (mgras help)".into()),
            ("rss".into(), "IMPACT raster subsystem: regs|pix".into()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn IoWrite + Send>) -> Result<(), String> {
        match cmd {
            "mgras" => self.cmd_mgras(args, &mut *w),
            "rss" => self.cmd_rss(args, &mut *w),
            _ => Err(format!("unknown command: {cmd}")),
        }
    }
}

impl Saveable for Mgras {
    // Bring-up: board state is not snapshotted yet.
    fn save_state(&self) -> toml::Value {
        toml::Value::Table(toml::map::Map::new())
    }

    fn load_state(&self, _v: &toml::Value) -> Result<(), String> {
        Ok(())
    }
}

fn read_status(v: Option<u64>) -> (u32, u64) {
    match v {
        Some(v) => (BUS_OK, v),
        None => (BUS_BUSY, 0),
    }
}

impl BusDevice for Mgras {
    fn read32(&self, addr: u64) -> BusRead32 {
        let (status, data) = read_status(self.do_read(addr, 32));
        BusRead32 { status, data: data as u32 }
    }
    fn write32(&self, addr: u64, val: u32) -> u32 { self.do_write(addr, 32, val as u64) }
    fn read8(&self, addr: u64) -> BusRead8 {
        let (status, data) = read_status(self.do_read(addr, 8));
        BusRead8 { status, data: data as u8 }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 { self.do_write(addr, 8, val as u64) }
    fn read16(&self, addr: u64) -> BusRead16 {
        let (status, data) = read_status(self.do_read(addr, 16));
        BusRead16 { status, data: data as u16 }
    }
    fn write16(&self, addr: u64, val: u16) -> u32 { self.do_write(addr, 16, val as u64) }
    fn read64(&self, addr: u64) -> BusRead64 {
        let (status, data) = read_status(self.do_read(addr, 64));
        BusRead64 { status, data }
    }
    fn write64(&self, addr: u64, val: u64) -> u32 { self.do_write(addr, 64, val) }
}

/// The board as the display host GL presents into: frames land in the
/// framebuffer under their window (see `disp::composite`). It cannot know the
/// guest X server's window ids, so only frames that say where their window is
/// are taken; the rest go back to the program to put up itself.
#[cfg(feature = "hostgl")]
pub struct ImpactScreen(pub Arc<Mgras>);

#[cfg(feature = "hostgl")]
impl iris_hostcall::Display for ImpactScreen {
    fn present(&self, _window: u32, _bgra: &[u8], _stride: usize, _width: usize, _height: usize) -> bool {
        false
    }

    fn present_at(&self, _window: u32, x: i32, y: i32, bgra: &[u8], stride: usize, width: usize, height: usize) -> bool {
        self.0.composite(x, y, bgra, stride, width, height)
    }

    fn present_index_at(&self, _window: u32, x: i32, y: i32, bgra: &[u8], stride: usize, width: usize, height: usize, overlay: bool) -> bool {
        self.0.composite_index(x, y, bgra, stride, width, height, overlay)
    }
}
