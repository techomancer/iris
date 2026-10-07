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
//! Threads (rules/mgras/DESIGN.md): the CPU thread owns the host-side
//! registers and the display control bus (`Front`). Command FIFO words and
//! direct raster writes go into `hq_fifo`; the HQ3 thread runs the frontend
//! and feeds raster writes into `rss_fifo`; the RSS thread draws. The display
//! thread scans the framebuffer out unlocked.

mod dcb;
mod debug;
mod disp;
mod frame;
mod ge11;
mod gl;
mod hq3;
mod pixmem;
mod plain;
mod record;
mod rss;
mod te1;

#[cfg(test)]
#[path = "mgras_tests.rs"]
mod mgras_tests;

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
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, BUS_BUSY, BUS_OK, Device, Saveable};

use ge11::Ge11;
use hq3::{host, Hq3Engine, Hq3Regs, Hq3Sink};
use rss::Rss;

/// The graphics slot the board decodes.
pub const MGRAS_SLOT_GFX_BASE: u32 = 0x1F00_0000;
pub const MGRAS_SLOT_GFX_SIZE: u32 = 0x0040_0000;
/// The board's register window within the slot.
const MAP_SIZE: u32 = 0x10_0000;

/// GIO ID: product 0x10, 32-bit ID, revision 1, GIO64, no ROM, manufacturer 1.
pub const GIO_ID: u32 = 0x0005_0190;

const HQ_FIFO_DEPTH: usize = 65536;
const RSS_FIFO_DEPTH: usize = 65536;

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
}

/// `Front` borrowed under its lock.
struct FrontGuard<'a> {
    _lock: parking_lot::MutexGuard<'a, ()>,
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

    fn status() -> u32 {
        // A vertical blank of about 1 ms in every 60 Hz frame.
        let us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64 % 16_667)
            .unwrap_or(0);
        host::STATUS_IDLE | if us < 1000 { host::STATUS_VBLANK } else { 0 }
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
                self.hq.ucode[((off - host::UCODE) / 4) as usize] = val as u32 & 0xFF_FFFF;
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
                    // (revision 1).
                    self.hq.ge_readback = [0, 1];
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

pub struct Mgras {
    kind: GraphicsBoard,
    ioc: crate::dev::ioc::Ioc,
    /// Current level of the general interrupt line.
    general: AtomicBool,
    /// CPU-side state and its lock. The display thread reads it unlocked
    /// (plain data, tearing tolerated).
    front_lock: Mutex<()>,
    front: UnsafeCell<Front>,
    /// Commands and registers not modelled yet, reported once each.
    unhandled: Mutex<HashSet<String>>,
    /// HQ3 flags: the HQ3 thread raises them, the CPU sets and clears them.
    flags: AtomicU32,
    /// Owned by the HQ3 thread (others only while the board is idle).
    eng: UnsafeCell<Hq3Engine>,
    /// Owned by the RSS thread (others only while the board is idle; the
    /// display thread reads the framebuffer unlocked).
    rss: UnsafeCell<Rss>,
    /// System memory, for the board's DMA (it is a GIO bus master).
    mem: Mutex<Option<Arc<dyn BusDevice>>>,
    hq_fifo: GFifo<HQ_FIFO_DEPTH>,
    rss_fifo: GFifo<RSS_FIFO_DEPTH>,
    hq_busy: AtomicBool,
    rss_busy: AtomicBool,
    /// Held by every producer while it pushes, and by anyone who needs the
    /// engines to stay idle (reads of drawing state, checkpoints, monitor).
    /// Holds the recording, so records come out in FIFO order.
    submit: Mutex<Option<record::RecHandle>>,
    /// The recording again, for DMA memory accesses on the HQ3 thread (which
    /// must not wait on `submit`: its holder may be waiting for idle).
    mem_rec: Mutex<Option<record::RecHandle>>,
    engines: AtomicBool,
    threads: Mutex<Vec<thread::JoinHandle<()>>>,
    hq_thread: Mutex<Option<thread::Thread>>,
    rss_thread: Mutex<Option<thread::Thread>>,
    dirty: AtomicBool,
    running: AtomicBool,
    refresh: Mutex<Option<thread::JoinHandle<()>>>,
    renderer: Mutex<Option<Box<dyn Renderer>>>,
    /// The finished frame, handed to the renderer as a prebuilt picture
    /// (`Rex3Screen::prebuilt`), as GR2 does.
    screen: Mutex<crate::disp::Rex3Screen>,
    screenshot_pending: AtomicBool,
    heartbeat: Arc<AtomicU64>,
    fasttick: Arc<AtomicU64>,
    cycles: Mutex<crate::cpu::mips_core::CyclesPtr>,
    /// Annotated trace (`mgras trace`); `trace_mask` gates the hot paths.
    trace: Mutex<debug::MgrasTrace>,
    trace_mask: AtomicU32,
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
        let vals: Vec<u64> = bytes
            .chunks(8)
            .map(|c| c.iter().enumerate().fold(0u64, |a, (i, &b)| a | (b as u64) << (56 - 8 * i)))
            .collect();
        self.0.rss_fifo.push_batch(tag::DMA_LINE, line as u64 | (bytes.len() as u64) << 32, &vals);
        Mgras::wake(&self.0.rss_thread);
    }
    fn dma_read_line(&mut self, line: u32) -> Vec<u8> {
        self.0.drain_rss();
        // SAFETY: as in `transfer`.
        unsafe { &*self.0.rss.get() }.dma_read_line(line)
    }
    fn gl_bracket(&mut self, enter: bool) {
        self.0.rss_push(if enter { tag::GL_ENTER } else { tag::GL_LEAVE }, 0);
    }
    fn set_flags(&mut self, bits: u32) {
        self.0.flags.fetch_or(bits, Ordering::AcqRel);
        self.0.refresh_general();
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
    fn tracing(&self) -> bool {
        self.0.tracing(debug::TRACE_HQ)
    }
    fn tracing_tex(&self) -> bool {
        self.0.tracing(debug::TRACE_TEX)
    }
    fn trace(&mut self, line: String) {
        self.0.trace_hq(&line);
    }
}

impl Mgras {
    /// Build the board in place on the heap: its multi-megabyte state is
    /// zero-initialised without touching the stack, and only the fields that
    /// are not valid zeroed are written.
    pub fn new(board: GraphicsBoard, ioc: crate::dev::ioc::Ioc, heartbeat: Arc<AtomicU64>, fasttick: Arc<AtomicU64>) -> Arc<Self> {
        let mut a: Arc<MaybeUninit<Self>> = Arc::new_zeroed();
        let p = Arc::get_mut(&mut a).unwrap().as_mut_ptr();
        // SAFETY: `p` points at zeroed, exclusively owned memory. Every field
        // that is not valid zeroed is written below, before assume_init.
        let m = unsafe {
            addr_of_mut!((*p).kind).write(board);
            addr_of_mut!((*p).ioc).write(ioc);
            addr_of_mut!((*p).general).write(AtomicBool::new(false));
            addr_of_mut!((*p).front_lock).write(Mutex::new(()));
            (*(*p).front.get()).init(board);
            addr_of_mut!((*p).unhandled).write(Mutex::new(HashSet::new()));
            addr_of_mut!((*p).flags).write(AtomicU32::new(0));
            addr_of_mut!((*p).eng).write(UnsafeCell::new(Hq3Engine::new()));
            Rss::init((*p).rss.get());
            addr_of_mut!((*p).mem).write(Mutex::new(None));
            addr_of_mut!((*p).submit).write(Mutex::new(None));
            addr_of_mut!((*p).mem_rec).write(Mutex::new(None));
            addr_of_mut!((*p).threads).write(Mutex::new(Vec::new()));
            addr_of_mut!((*p).hq_thread).write(Mutex::new(None));
            addr_of_mut!((*p).rss_thread).write(Mutex::new(None));
            addr_of_mut!((*p).dirty).write(AtomicBool::new(true));
            addr_of_mut!((*p).refresh).write(Mutex::new(None));
            addr_of_mut!((*p).renderer).write(Mutex::new(None));
            addr_of_mut!((*p).screen).write(Mutex::new(crate::disp::Rex3Screen::new()));
            addr_of_mut!((*p).heartbeat).write(heartbeat);
            addr_of_mut!((*p).fasttick).write(fasttick);
            addr_of_mut!((*p).cycles).write(Mutex::new(crate::cpu::mips_core::CyclesPtr::dangling()));
            addr_of_mut!((*p).trace).write(Mutex::new(debug::MgrasTrace::new()));
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

    /// The CPU-side state, locked.
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

    fn note(&self, what: String) {
        let mut u = self.unhandled.lock();
        if u.len() < 256 && u.insert(what.clone()) {
            eprintln!("mgras: not modelled yet: {what}");
        }
    }

    fn wake(slot: &Mutex<Option<thread::Thread>>) {
        if let Some(t) = slot.lock().as_ref() {
            t.unpark();
        }
    }

    /// True when neither FIFO holds work and neither engine is mid-entry.
    /// The HQ3 is checked first: whatever it forwarded before going idle is
    /// already in `rss_fifo` when that is checked.
    fn idle(&self) -> bool {
        self.hq_fifo.is_empty()
            && !self.hq_busy.load(Ordering::Acquire)
            && self.rss_fifo.is_empty()
            && !self.rss_busy.load(Ordering::Acquire)
    }

    /// Wait until the board is idle. Holding `submit` keeps it idle.
    fn wait_idle(&self) {
        let backoff = crossbeam_utils::Backoff::new();
        while !self.idle() {
            backoff.snooze();
        }
    }

    /// HQ3 thread: wait until the RSS has run everything queued to it.
    fn drain_rss(&self) {
        let backoff = crossbeam_utils::Backoff::new();
        while !self.rss_fifo.is_empty() || self.rss_busy.load(Ordering::Acquire) {
            backoff.snooze();
        }
    }

    /// HQ3 thread: queue one entry for the RSS.
    fn rss_push(&self, addr: u32, val: u64) {
        self.rss_fifo.push(addr, val);
        Self::wake(&self.rss_thread);
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
            let mut h = blake3::Hasher::new();
            for w in rest.words.iter() {
                h.update(&w.to_le_bytes());
            }
            out.extend_from_slice(h.finalize().as_bytes());
        }
        // Clipping IDs affect future rendering even when the colour planes
        // are identical. Keep zero-CID recordings' historical hashes.
        if rss.cid.iter().any(|&c| c != 0) {
            out.extend_from_slice(b"CID\0");
            out.extend_from_slice(blake3::hash(&rss.cid).as_bytes());
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

    /// Give the board its path to system memory, for DMA.
    pub fn set_phys(&self, mem: Arc<dyn BusDevice>) {
        *self.mem.lock() = Some(mem);
    }

    /// Drive one of the board's interrupt lines (graphics slot wiring).
    fn set_line(&self, line: Line, active: bool) {
        use crate::dev::ioc::IocInterrupt;
        let src = match line {
            Line::Fifo => IocInterrupt::GioSgFifo,
            Line::General => IocInterrupt::GioSgGraphics,
            Line::Retrace => IocInterrupt::GioSgRetrace,
        };
        self.ioc.set_interrupt(src, active);
    }

    /// Wire up the CPU cycle counter for the status bar's MIPS figure.
    pub fn set_cpu_cycles(&self, ptr: crate::cpu::mips_core::CyclesPtr) {
        *self.cycles.lock() = ptr;
    }

    fn offset(&self, addr: u32) -> Option<u32> {
        let off = addr.wrapping_sub(MGRAS_SLOT_GFX_BASE);
        (off < MAP_SIZE).then_some(off)
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
    /// them (bus busy) until the board is idle.
    fn read_needs_idle(off: u32) -> bool {
        matches!(off,
            host::STATUS | host::FIFOSTATUS | host::GIOSTATUS | host::DMABUSY
            | host::GE_READBACK_HI | host::GE_READBACK_LO
            | host::RASTER_IF_CONTEXT..=0x50228 | host::DMA_CONTEXT..=0x504FF
            | host::RASTER..=0x7FFFF)
    }

    /// One read, with `submit` held. None means "busy, retry".
    fn read_locked(&self, off: u32, bits: u32) -> Option<u64> {
        if Self::read_needs_idle(off) && !self.idle() {
            return None;
        }
        // SAFETY (for both engines below): idle with `submit` held.
        let v = match off {
            host::STATUS => Front::status() as u64,
            host::FIFOSTATUS | host::GIOSTATUS | host::DMABUSY => 0,
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
                Self::wake(&self.hq_thread);
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
                Self::wake(&self.hq_thread);
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
                Self::wake(&self.hq_thread);
            }
            host::CONTEXT_SWITCH => {
                // The save phase completes at once; the incoming context
                // follows through the FIFO, after this marker.
                if !self.hq_fifo.try_push(tag::CONTEXT_SWITCH, 0) {
                    return false;
                }
                self.flags.fetch_or(host::FLAG_CONTEXT_SAVED, Ordering::AcqRel);
                Self::wake(&self.hq_thread);
            }
            host::SET_FLAGS | host::SET_FLAGS_PRIVILEGED => {
                self.flags.fetch_or(lo, Ordering::AcqRel);
            }
            host::CLEAR_FLAGS | host::CLEAR_FLAGS_PRIVILEGED => {
                self.flags.fetch_and(!lo, Ordering::AcqRel);
            }
            0x80000..=0xFFFFF => self.note(format!("fast-path command window write at {off:#x}")),
            _ => {
                if self.front().write(off, bits, val, &self.flags) {
                    self.dirty.store(true, Ordering::Release);
                }
            }
        }
        true
    }

    /// Writes that go through `hq_fifo` (traced as HQ/RSS lines instead).
    fn is_fifo_write(off: u32) -> bool {
        matches!(off, host::CFIFO | 0x70084 | host::CFIFO_GL | host::CFIFO_PRIVILEGED | 0x50084 | host::RASTER..=0x7FFFF)
    }

    fn do_read(&self, addr: u32, bits: u32) -> Option<u64> {
        let Some(off) = self.offset(addr) else { return Some(0) };
        let sub = self.submit.lock();
        let v = self.read_locked(off, bits)?;
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

    fn do_write(&self, addr: u32, bits: u32, val: u64) -> u32 {
        let Some(off) = self.offset(addr) else { return BUS_OK };
        let sub = self.submit.lock();
        if !self.write_locked(off, bits, val) {
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
        loop {
            let Some((t, val)) = self.hq_fifo.peek() else {
                self.hq_fifo.flush_head();
                self.hq_busy.store(false, Ordering::Release);
                if backoff.is_completed() {
                    thread::park_timeout(std::time::Duration::from_millis(2));
                } else {
                    backoff.snooze();
                }
                continue;
            };
            self.hq_busy.store(true, Ordering::Release);
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
        }
        self.hq_fifo.consume();
        self.hq_fifo.flush_head();
        self.hq_busy.store(false, Ordering::Release);
    }

    fn rss_loop(&self) {
        *self.rss_thread.lock() = Some(thread::current());
        let backoff = crossbeam_utils::Backoff::new();
        let mut stale_seen = 0u32;
        let mut payload: Vec<u64> = Vec::new();
        loop {
            let Some((t, val)) = self.rss_fifo.peek() else {
                self.rss_fifo.flush_head();
                self.rss_busy.store(false, Ordering::Release);
                if backoff.is_completed() {
                    thread::park_timeout(std::time::Duration::from_millis(2));
                } else {
                    backoff.snooze();
                }
                continue;
            };
            self.rss_busy.store(true, Ordering::Release);
            // SAFETY: the RSS thread owns the raster subsystem.
            let rss = unsafe { &mut *self.rss.get() };
            let changed = match t {
                tag::EXIT => break,
                t if t < 0x1000 => rss.write((t & 0x7FF) >> 1, val as u32, t & 1 != 0),
                tag::PIO_ADVANCE => {
                    rss.pio_read_hi();
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
                    self.rss_fifo.drain_payload(n, &mut payload[..]);
                    let bytes: Vec<u8> = payload.iter().flat_map(|w| w.to_be_bytes()).take(len).collect();
                    rss.dma_write_line(line, &bytes);
                    true
                }
                _ => false,
            };
            if changed {
                self.dirty.store(true, Ordering::Release);
            }
            if self.tracing(debug::TRACE_RSS) {
                self.trace_rss(t, val, rss, changed);
            }
            if rss.te.stale_events != stale_seen {
                stale_seen = rss.te.stale_events;
                if self.tracing(debug::TRACE_TEX) {
                    let [want, have] = rss.te.stale_last;
                    self.trace_hq(&format!("TEX stale sample #{stale_seen}: want {want:#x}, page holds {have:#x}"));
                }
            }
            self.rss_fifo.consume();
            backoff.reset();
        }
        self.rss_fifo.consume();
        self.rss_fifo.flush_head();
        self.rss_busy.store(false, Ordering::Release);
    }

    /// Start the HQ3 and RSS threads.
    pub fn start_engines(self: &Arc<Self>) {
        if self.engines.swap(true, Ordering::AcqRel) {
            return;
        }
        let mut threads = self.threads.lock();
        let me = Arc::clone(self);
        threads.push(thread::Builder::new().name("MGRAS-HQ3".into()).spawn(move || me.hq_loop()).expect("spawn MGRAS HQ3 thread"));
        let me = Arc::clone(self);
        threads.push(thread::Builder::new().name("MGRAS-RSS".into()).spawn(move || me.rss_loop()).expect("spawn MGRAS RSS thread"));
    }

    /// Stop the engine threads once they have run everything queued.
    pub fn stop_engines(&self) {
        if !self.engines.swap(false, Ordering::AcqRel) {
            return;
        }
        {
            let _sub = self.submit.lock();
            self.hq_fifo.push(tag::EXIT, 0);
            Self::wake(&self.hq_thread);
        }
        let mut threads = self.threads.lock();
        if let Some(h) = (!threads.is_empty()).then(|| threads.remove(0)) {
            let _ = h.join();
        }
        self.rss_fifo.push(tag::EXIT, 0);
        Self::wake(&self.rss_thread);
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
}

impl Device for Mgras {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("mgras".into(), "IMPACT graphics: status|hq|gl|ge|ucode|vc3|xmap|cmap|dac|pix|stats|fbdump|shot|dump|trace|rec (mgras help)".into()),
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
    fn read32(&self, addr: u32) -> BusRead32 {
        let (status, data) = read_status(self.do_read(addr, 32));
        BusRead32 { status, data: data as u32 }
    }
    fn write32(&self, addr: u32, val: u32) -> u32 { self.do_write(addr, 32, val as u64) }
    fn read8(&self, addr: u32) -> BusRead8 {
        let (status, data) = read_status(self.do_read(addr, 8));
        BusRead8 { status, data: data as u8 }
    }
    fn write8(&self, addr: u32, val: u8) -> u32 { self.do_write(addr, 8, val as u64) }
    fn read16(&self, addr: u32) -> BusRead16 {
        let (status, data) = read_status(self.do_read(addr, 16));
        BusRead16 { status, data: data as u16 }
    }
    fn write16(&self, addr: u32, val: u16) -> u32 { self.do_write(addr, 16, val as u64) }
    fn read64(&self, addr: u32) -> BusRead64 {
        let (status, data) = read_status(self.do_read(addr, 64));
        BusRead64 { status, data }
    }
    fn write64(&self, addr: u32, val: u64) -> u32 { self.do_write(addr, 64, val) }
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
}
