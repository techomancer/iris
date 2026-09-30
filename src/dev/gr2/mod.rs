//! GR2 graphics (Indy XZ / Indigo2 XZ / Indigo2 Extreme).
//!
//! Block layout (rules/gr2/DESIGN.md):
//!   hq2.rs     HQ2 registers + HLE command interpreter (HQ2 thread)
//!   re3.rs     RE3 registers, VRAM, drawing (RE3 thread)
//!   ge7.rs     GE7 windows + microcode storage (inert)
//!   vc1.rs     VC1 registers + SRAM (inert)
//!   xmap5.rs   XMAP5 mode tables + CLUTs (inert)
//!   bt457.rs   Bt457 RAMDACs (inert)
//!   gr2disp.rs display thread: retrace + composition + present
//!   gr2comp.rs software compositor
//!
//! Data path: CPU FIFO writes go to `hq_fifo`, and the HQ2 thread turns them
//! into RE3 register writes on `re3_fifo`. CPU writes to the RE3 register
//! window go straight to `re3_fifo`, and the RE3 thread draws into VRAM. The
//! display thread reads VRAM and the inert blocks with no locking (tearing is
//! tolerated, as with REX3).
//!
//! The hardware reference for every value here is `ignore/gr2/*.h`.

pub mod bt457;
mod debug;
pub mod ge7;
pub mod gr2comp;
pub mod gr2disp;
pub mod hq2;
pub mod re3;
pub mod vc1;
pub mod xmap5;

#[cfg(test)]
#[path = "re3_tests.rs"]
mod re3_tests;
#[cfg(test)]
#[path = "hq2_tests.rs"]
mod hq2_tests;

use std::cell::{Cell, UnsafeCell};
use std::io::Write as IoWrite;
use std::mem::MaybeUninit;
use std::ptr::addr_of_mut;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use parking_lot::Mutex;

use crate::disp::Rex3Screen;
use crate::gfifo::GFifo;
use crate::mips_core::CyclesPtr;
use crate::rex3::Renderer;
use crate::snapshot::{get_field, hex_u32, load_u32_slice, load_u8_slice, toml_u32, u32_slice_to_toml, u8_slice_to_toml};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, Resettable, Saveable, BUS_BUSY, BUS_OK};

use bt457::Bt457;
use ge7::Ge7;
use hq2::{Hq2Engine, Hq2Regs, Re3Sink};
use re3::Re3;
use vc1::Vc1;
use xmap5::Xmap5;

/// GIO gfx slot base and size (the board decodes a 4 MB slot).
pub const GR2_BASE: u32 = 0x1F00_0000;
pub const GR2_SLOT_SIZE: u32 = 0x0040_0000;

// Region offsets from the board base (GR2.h).
const SHRAM: u32 = 0x00000;
const SHRAM_END: u32 = 0x20000;
const FIFO: u32 = 0x40000;
const FIFO_END: u32 = 0x60000;
const HQUCODE: u32 = 0x60000;
const HQUCODE_END: u32 = 0x68000;
const GEWIN: u32 = 0x68000;
const GEWIN_END: u32 = 0x6a000;
const HQREGS: u32 = 0x6a000;
const HQREGS_END: u32 = 0x6a104;
const BDVERS: u32 = 0x6c000;
const CLOCK: u32 = 0x6c020;
const VC1: u32 = 0x6c040;
const DAC: u32 = 0x6c0a0;
const XMAP: u32 = 0x6c100;
const XMAPALL: u32 = 0x6c1a0;
const AB1: u32 = 0x6c1c0;
const CC1: u32 = 0x6c1e0;
const RE3_REGS: u32 = 0x6c200;
const RE3_REGS_END: u32 = 0x6c300;
const RE3_32: u32 = 0x6c600;

pub const SHRAM_WORDS: usize = 32768;
/// shram word the PROM sets to 1 (UCODE_TP) once the textport microcode runs.
pub const TP_PROBE_ID: usize = 0x7fff;

pub const HQ_FIFO_DEPTH: usize = 65536;
pub const RE3_FIFO_DEPTH: usize = 65536;
/// Words the HQ2_GEDMA read port holds ahead of its reader (power of two).
pub const GEDMA_OUT_WORDS: usize = 16384;
/// How long the HQ2 waits for a reader when the read port is full.
const GEDMA_OUT_STALL_LIMIT_NS: u64 = 2_000_000_000;

/// Which board is installed. Same hardware family; only the probe answers differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gr2Variant {
    /// 2 GE7s. Indy XZ and Indigo2 XZ.
    Xz,
    /// 8 GE7s. Indigo2 Extreme.
    Extreme,
}

impl Gr2Variant {
    pub fn ges(self) -> u32 {
        match self { Self::Xz => 2, Self::Extreme => 8 }
    }
    /// Board revision in rd0 (GR2.h naming: 4 = GR3, 5..8 = GU1).
    pub fn board_rev(self) -> u32 {
        match self { Self::Xz => 4, Self::Extreme => 6 }
    }
    pub fn name(self) -> &'static str {
        match self { Self::Xz => "XZ", Self::Extreme => "Extreme" }
    }
}

/// Monitor ID reported in rd0[7:4]: a plain 60 Hz 1280x1024 CRT. Chosen to
/// avoid every ID with special handling (GR2.h "Monitor Types").
const MONITOR_ID: u32 = 0;

/// CPU-side register and memory state. Plain data, zero-valid.
#[repr(C)]
pub struct Gr2Regs {
    pub shram: [u32; SHRAM_WORDS],
    pub hq: Hq2Regs,
    pub ge: Ge7,
    /// Last configuration writes to rd0/rd1 (debug only; reads return IDs).
    pub bdvers_w: [u32; 4],
    pub vc1: Vc1,
    pub dac: [Bt457; 3],
    pub xmap: [Xmap5; 5],
    pub ab1: u32,
    /// ICS PLL bit-bang (clock.write): bit 0 data, LSB first per byte; bit 1
    /// set on the final bit (GR2.h "Clock"). Bits collected so far, count,
    /// and the last complete 7-byte programming.
    pub pll_bits: u64,
    pub pll_count: u32,
    pub pll_last: [u8; 8],
    pub pll_programs: u32,
}

/// Status-bar counters shared with the rest of the machine (same set REX3 gets).
pub struct Gr2Stats {
    pub heartbeat: Arc<AtomicU64>,
    pub fasttick: Arc<AtomicU64>,
}

pub struct Gr2 {
    variant: Gr2Variant,
    regs: UnsafeCell<Gr2Regs>,
    re3: UnsafeCell<Re3>,
    hq_engine: UnsafeCell<Hq2Engine>,
    /// Finish / 2D sync tokens (0x0A3, 0x155) the CPU has queued that the HQ2
    /// has not executed yet. FIN3 reads as clear while any is pending: with
    /// our deep FIFO an older Finish would otherwise raise FIN3 for a newer
    /// wait, and the pipeline would run a whole frame behind every swap
    /// (ideas flicker, IRIX 6.5.22). See rules/gr2/fin3-must-track-pending-finish.md.
    fin3_pending: AtomicU32,
    /// Set when the kernel acks FIN2 (write to 0x6A04C), which it does right
    /// before issuing a FIN2 command (pixel DMA 0x147, context save/restore)
    /// and polling version bit 1 in a counted loop (100,000 x us_delay(1) for
    /// pixel DMA). Our HQ2 may still be drawing a large pixel DMA when that
    /// budget runs out: IRIX 5.3 then logged "Gr2PixelDma: TIMEOUT", raised a
    /// graphics error and detached Xsgi. While set, a version read with FIN2
    /// clear stalls (bus busy) as long as the HQ2 still has work, so the
    /// kernel sees FIN2 as soon as the HQ2 gets there. See
    /// rules/gr2/fin2-wait-must-stall.md.
    fin2_wait: AtomicU32,
    /// Host time (fin2_clock ns) of the first stalled FIN2 poll since the
    /// wait was armed (0 = none yet). The stall gives up FIN2_STALL_LIMIT
    /// after that, so a pipeline that can never finish (RE3 held by a
    /// CPU-driven readback) cannot hang the machine. Counted from the first
    /// poll, not from the ack: the DMA between ack and poll may take long on
    /// a busy host (a 400-row pixel DMA on a GitHub CI runner did).
    fin2_armed_ns: AtomicU64,
    /// HQ2_GEDMA read port: HQ2 -> host words (context saves 0x1E1, pixel
    /// DMA reads 0x152 / 0x0AC). A ring the HQ2 thread fills and the reader
    /// (the kernel's VDMA) drains; head / tail count words since reset.
    /// A read with the ring empty waits (bus busy) while the HQ2 still has
    /// work, since the kernel starts the VDMA right after queuing the
    /// request. See rules/gr2/gedma-read-port.md.
    gedma_out: [AtomicU32; GEDMA_OUT_WORDS],
    gedma_out_head: AtomicU32,
    gedma_out_tail: AtomicU32,
    hq_fifo: GFifo<HQ_FIFO_DEPTH>,
    re3_fifo: GFifo<RE3_FIFO_DEPTH>,
    /// Pending half of a two-entry COPY op (RE3 thread only).
    copy_a: Cell<u64>,

    running: AtomicBool,
    hq_busy: AtomicBool,
    re3_busy: AtomicBool,
    /// Something visible changed since the last composed frame.
    dirty: AtomicBool,
    threads: Mutex<Vec<thread::JoinHandle<()>>>,
    hq_thread: Mutex<Option<thread::Thread>>,
    re3_thread: Mutex<Option<thread::Thread>>,

    pub screen: Mutex<Rex3Screen>,
    pub renderer: Mutex<Option<Box<dyn Renderer>>>,
    pub screenshot_pending: AtomicBool,
    screenshot_counter: AtomicU32,
    retrace_cb: Mutex<Option<Arc<dyn Fn(bool) + Send + Sync>>>,
    stats: Gr2Stats,
    cycles: Cell<CyclesPtr>,
    /// Annotated FIFO capture (`gr2 trace`); `trace_mask` gates the hot path.
    trace: Mutex<debug::Gr2Trace>,
    trace_mask: AtomicU32,
}

// SAFETY: same model as Rex3. `regs` is written only by the CPU thread (plus
// monitor/snapshot with the machine stopped). `re3` and `hq_engine` are owned
// by their threads, and the CPU only touches them after both FIFOs are
// drained. The display thread reads with tearing tolerated. `copy_a` is RE3
// thread only. `cycles` is set once during single-threaded setup.
unsafe impl Sync for Gr2 {}
unsafe impl Send for Gr2 {}

impl Gr2 {
    /// Build a GR2 in place on the heap. The multi-megabyte register/VRAM
    /// state is zero-initialised without ever touching the stack; only the
    /// runtime fields that are not all-zero-valid are written explicitly.
    pub fn new(variant: Gr2Variant, stats: Gr2Stats) -> Arc<Self> {
        let mut a: Arc<MaybeUninit<Self>> = Arc::new_zeroed();
        let p = Arc::get_mut(&mut a).unwrap().as_mut_ptr();
        // SAFETY: `p` points at zeroed, exclusively owned memory. Every field
        // that is not valid when zeroed is initialised below, before assume_init.
        unsafe {
            addr_of_mut!((*p).variant).write(variant);
            addr_of_mut!((*p).copy_a).write(Cell::new(0));
            addr_of_mut!((*p).running).write(AtomicBool::new(false));
            addr_of_mut!((*p).hq_busy).write(AtomicBool::new(false));
            addr_of_mut!((*p).re3_busy).write(AtomicBool::new(false));
            addr_of_mut!((*p).dirty).write(AtomicBool::new(true));
            addr_of_mut!((*p).threads).write(Mutex::new(Vec::new()));
            addr_of_mut!((*p).hq_thread).write(Mutex::new(None));
            addr_of_mut!((*p).re3_thread).write(Mutex::new(None));
            addr_of_mut!((*p).screen).write(Mutex::new(Rex3Screen::new()));
            addr_of_mut!((*p).renderer).write(Mutex::new(None));
            addr_of_mut!((*p).screenshot_pending).write(AtomicBool::new(false));
            addr_of_mut!((*p).screenshot_counter).write(AtomicU32::new(0));
            addr_of_mut!((*p).retrace_cb).write(Mutex::new(None));
            addr_of_mut!((*p).stats).write(stats);
            addr_of_mut!((*p).cycles).write(Cell::new(CyclesPtr::dangling()));
            addr_of_mut!((*p).trace).write(Mutex::new(debug::Gr2Trace::new()));
            addr_of_mut!((*p).trace_mask).write(AtomicU32::new(0));
            let a = a.assume_init();
            a.power_on();
            a
        }
    }

    pub fn variant(&self) -> Gr2Variant { self.variant }

    pub fn set_retrace_callback(&self, cb: Arc<dyn Fn(bool) + Send + Sync>) {
        *self.retrace_cb.lock() = Some(cb);
    }

    pub fn set_cpu_cycles(&self, ptr: CyclesPtr) {
        self.cycles.set(ptr);
    }

    #[allow(clippy::mut_from_ref)]
    fn regs(&self) -> &mut Gr2Regs {
        // SAFETY: see the Sync impl.
        unsafe { &mut *self.regs.get() }
    }

    /// VRAM, for the display thread and tests.
    pub fn vram(&self) -> &[u32] {
        // SAFETY: read-only view; tearing tolerated.
        unsafe { &(*self.re3.get()).vram }
    }

    /// RE3 Z control (tests; call when idle).
    #[cfg(test)]
    pub fn re3_zctl(&self) -> u32 {
        // SAFETY: read-only, used after wait_idle.
        unsafe { (*self.re3.get()).ctx.zctl }
    }

    /// RE3 pixel format (tests; call when idle).
    #[cfg(test)]
    pub fn re3_ctx_pixfmt(&self) -> u32 {
        // SAFETY: read-only, used after wait_idle.
        unsafe { (*self.re3.get()).ctx.pixfmt }
    }

    /// RE3 register value (tests; call when idle).
    #[cfg(test)]
    pub fn re3_reg(&self, reg: usize) -> u32 {
        // SAFETY: read-only, used after wait_idle.
        unsafe { (*self.re3.get()).ctx.reg[reg & 63] }
    }

    /// True when neither FIFO holds work and neither engine is mid-command.
    pub fn idle(&self) -> bool {
        self.hq_fifo.is_empty()
            && !self.hq_busy.load(Ordering::Acquire)
            && self.re3_fifo.is_empty()
            && !self.re3_busy.load(Ordering::Acquire)
    }

    /// A FIN2 poll may stall: true until FIN2_STALL_LIMIT after the first
    /// stalled poll of this wait (Gr2::fin2_armed_ns).
    fn fin2_stall_ok(&self) -> bool {
        let now = fin2_clock().max(1);
        let first = match self.fin2_armed_ns.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => now,
            Err(t) => t,
        };
        now.saturating_sub(first) < FIN2_STALL_LIMIT.as_nanos() as u64
    }

    /// Spin until both engines are idle (tests and snapshots).
    pub fn wait_idle(&self) {
        let backoff = crossbeam_utils::Backoff::new();
        while !self.idle() {
            backoff.snooze();
        }
    }

    /// HQ2 thread: a new HQ2_GEDMA read transfer starts. Words an earlier
    /// one left unread would shift this one, so drop them.
    fn gedma_begin(&self) {
        let head = self.gedma_out_head.load(Ordering::Relaxed);
        let tail = self.gedma_out_tail.load(Ordering::Acquire);
        if head != tail {
            crate::dlog_dev!(crate::devlog::LogModule::Gr2,
                "GR2: GEDMA: {} words of the previous transfer never read, dropped", head.wrapping_sub(tail));
            let _ = self.gedma_out_tail.compare_exchange(tail, head, Ordering::AcqRel, Ordering::Relaxed);
        }
    }

    /// HQ2 thread: queue one word on the HQ2_GEDMA read port, waiting while
    /// it is full. False if no reader drained it within the limit (or the
    /// engine stops): the rest of the transfer is dropped.
    fn gedma_push(&self, w: u32) -> bool {
        let head = self.gedma_out_head.load(Ordering::Relaxed);
        let mut since = 0u64;
        while head.wrapping_sub(self.gedma_out_tail.load(Ordering::Acquire)) as usize >= GEDMA_OUT_WORDS {
            if !self.running.load(Ordering::Relaxed) {
                return false;
            }
            let now = fin2_clock();
            if since == 0 {
                since = now;
            } else if now - since > GEDMA_OUT_STALL_LIMIT_NS {
                crate::dlog_dev!(crate::devlog::LogModule::Gr2, "GR2: GEDMA read port full, no reader: transfer dropped");
                return false;
            }
            thread::yield_now();
        }
        self.gedma_out[head as usize & (GEDMA_OUT_WORDS - 1)].store(w, Ordering::Relaxed);
        self.gedma_out_head.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Reader: the next HQ2_GEDMA word, if the HQ2 has produced one.
    fn gedma_pop(&self) -> Option<u32> {
        let mut tail = self.gedma_out_tail.load(Ordering::Acquire);
        loop {
            if tail == self.gedma_out_head.load(Ordering::Acquire) {
                return None;
            }
            let v = self.gedma_out[tail as usize & (GEDMA_OUT_WORDS - 1)].load(Ordering::Relaxed);
            // CAS: the CPU and the VDMA worker may both read the port.
            match self.gedma_out_tail.compare_exchange(tail, tail.wrapping_add(1), Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return Some(v),
                Err(t) => tail = t,
            }
        }
    }

    fn wake(slot: &Mutex<Option<thread::Thread>>) {
        if let Some(t) = slot.lock().as_ref() {
            t.unpark();
        }
    }

    // ── board ID registers ───────────────────────────────────────────────────

    fn bdvers_read(&self, idx: u32) -> u32 {
        match idx {
            // rd0: monitor ID in [7:4], ~rev in [3:0].
            0 => (MONITOR_ID << 4) | (!self.variant.board_rev() & 0xf),
            // rd1: Z (bit 5), 24bpp (bit 4), no corona (bits 3:2 = 11), VB rev 1.
            1 => 0x3c | (!1 & 3),
            // rd2/rd3 are only read on rev < 4 boards: report everything absent.
            _ => 0xff,
        }
    }

    // ── CPU register access (32-bit; narrower widths share the same decode) ──

    fn reg_read(&self, off: u32) -> BusRead32 {
        let res = self.reg_read_inner(off);
        if res.status == BUS_OK && !(FIFO..FIFO_END).contains(&off) && self.tracing(debug::TRACE_CPU) {
            self.trace_cpu(false, off, res.data);
        }
        res
    }

    fn reg_read_inner(&self, off: u32) -> BusRead32 {
        let r = self.regs();
        let v = match off {
            SHRAM..SHRAM_END => r.shram[(off >> 2) as usize],
            FIFO..FIFO_END => 0,
            HQUCODE..HQUCODE_END => r.hq.ucode[((off - HQUCODE) >> 2) as usize],
            GEWIN..GEWIN_END => r.ge.read(((off - GEWIN) >> 10) as usize, ((off >> 2) & 0xff) as usize),
            // HQ2_GEDMA read: the next word the HQ2 produced (context image,
            // pixel DMA read), read by the kernel's VDMA.
            0x6a068 => {
                // Sample "HQ2 has work" before looking at the port: a word
                // is pushed before the HQ2 goes idle, so an idle HQ2 seen
                // here means every word it produced is visible below.
                let working = !self.hq_fifo.is_empty() || self.hq_busy.load(Ordering::Acquire);
                match self.gedma_pop() {
                    Some(v) => v,
                    // Not produced yet: the request is queued or running.
                    None if working => return BusRead32::busy(),
                    None => {
                        crate::dlog_dev!(crate::devlog::LogModule::Gr2, "GR2: GEDMA read overrun (HQ2 idle, nothing to read)");
                        0
                    }
                }
            }
            0x6b000 => if self.fin3_pending.load(Ordering::Acquire) != 0 { 0 } else { r.hq.fin[hq2::FIN3].load(Ordering::Acquire) },
            HQREGS..HQREGS_END => {
                // Sample "HQ2 has work" before reading the register: the HQ2
                // raises FIN2 before it consumes the entry and drops
                // hq_busy, so if it was idle before the read, the read sees
                // every FIN2 it raised (sampling after would race).
                let hq_working = off - HQREGS == hq2::HQ_VERSION
                    && (!self.hq_fifo.is_empty() || self.hq_busy.load(Ordering::Acquire));
                let v = r.hq.read(off - HQREGS);
                if off - HQREGS == hq2::HQ_VERSION {
                    if self.fin2_wait.load(Ordering::Acquire) != 0 {
                        if v & 2 != 0 {
                            self.fin2_wait.store(0, Ordering::Release);
                        } else if hq_working && self.fin2_stall_ok() {
                            // FIN2 awaited and the HQ2 is still working: the
                            // read waits for it instead of spending the
                            // kernel's poll budget. With the HQ2 idle and no
                            // FIN2, the kernel times out as on hardware.
                            return BusRead32::busy();
                        }
                    }
                    if self.fin3_pending.load(Ordering::Acquire) != 0 {
                        return BusRead32::ok(v & !1);
                    }
                }
                v
            }
            0x6c000..0x6c010 => self.bdvers_read((off - BDVERS) >> 2),
            0x6c040..0x6c060 => r.vc1.read(off & 0x1c),
            0x6c0a0..0x6c100 => r.dac[((off - DAC) >> 5) as usize].read(off & 0x1c) as u32,
            0x6c100..0x6c1a0 => r.xmap[((off - XMAP) >> 5) as usize].read(off & 0x1c),
            0x6c1a0..0x6c1c0 => r.xmap[0].read(off & 0x1c),
            0x6c1c0..0x6c1e0 => r.ab1,
            RE3_REGS..RE3_REGS_END => return self.re3_read(((off - RE3_REGS) >> 2) as usize),
            0x6c600..0x6c604 => return self.re3_read(re3::REG_RWDATA),
            _ => 0, // CLOCK, CC1 (must not echo the flat-panel probe), unused
        };
        BusRead32::ok(v)
    }

    fn reg_write(&self, off: u32, val: u32) -> u32 {
        let st = self.reg_write_inner(off, val);
        if st == BUS_OK && !(FIFO..FIFO_END).contains(&off) && self.tracing(debug::TRACE_CPU) {
            self.trace_cpu(true, off, val);
        }
        st
    }

    /// Byte/halfword store. Registers see the value whatever the width; only
    /// the XMAP5 CLUT port distinguishes (a 32-bit store is a packed entry).
    fn reg_write_narrow(&self, off: u32, val: u32) -> u32 {
        if (0x6c100..0x6c1c0).contains(&off) && off & 0x1c == xmap5::XMAP_CLUT {
            let r = self.regs();
            if off >= XMAPALL {
                for x in r.xmap.iter_mut() {
                    x.write(xmap5::XMAP_CLUT, val);
                }
            } else {
                r.xmap[((off - XMAP) >> 5) as usize].write(xmap5::XMAP_CLUT, val);
            }
            self.dirty.store(true, Ordering::Relaxed);
            if self.tracing(debug::TRACE_CPU) {
                self.trace_cpu(true, off, val);
            }
            return BUS_OK;
        }
        self.reg_write(off, val)
    }

    fn reg_write_inner(&self, off: u32, val: u32) -> u32 {
        let r = self.regs();
        match off {
            SHRAM..SHRAM_END => r.shram[(off >> 2) as usize] = val,
            FIFO..FIFO_END => {
                let idx = (off - FIFO) >> 2;
                let fin = is_fin3_token(idx);
                if fin {
                    // Counted before the push so the HQ can never see the
                    // token before the counter includes it.
                    self.fin3_pending.fetch_add(1, Ordering::AcqRel);
                }
                if !self.hq_fifo.try_push(idx, val as u64) {
                    if fin {
                        self.fin3_pending.fetch_sub(1, Ordering::AcqRel);
                    }
                    return BUS_BUSY;
                }
                Self::wake(&self.hq_thread);
            }
            HQUCODE..HQUCODE_END => r.hq.ucode[((off - HQUCODE) >> 2) as usize] = val,
            GEWIN..GEWIN_END => r.ge.write(((off - GEWIN) >> 10) as usize, ((off >> 2) & 0xff) as usize, val),
            // HQ2_GEDMA: the DMA data port. Words join the command stream, so
            // a command waiting for DMA data (context restore, pixel writes)
            // receives them in order with its FIFO arguments.
            0x6a068 => {
                if !self.hq_fifo.try_push(hq2::HQ_TOKEN_GEDMA, val as u64) {
                    return BUS_BUSY;
                }
                Self::wake(&self.hq_thread);
            }
            // FIN3 write port (HQ2.h): Xsgi writes 0 after seeing FIN3;
            // the kernel restores a context's saved FIN3 here.
            0x6b000 => r.hq.fin[hq2::FIN3].store(val & 1, Ordering::Release),
            // unstall: restart the microcode. The marker keeps the restart
            // ordered with the FIFO words that follow it (start argument).
            0x6a078 => {
                if !self.hq_fifo.try_push(hq2::HQ_TOKEN_UNSTALL, 0) {
                    return BUS_BUSY;
                }
                r.hq.write(off - HQREGS, val, &mut r.ge);
                Self::wake(&self.hq_thread);
            }
            HQREGS..HQREGS_END => {
                if off - HQREGS == hq2::HQ_FIN2 {
                    self.fin2_armed_ns.store(0, Ordering::Release);
                    self.fin2_wait.store(1, Ordering::Release);
                }
                r.hq.write(off - HQREGS, val, &mut r.ge)
            }
            0x6c000..0x6c010 => r.bdvers_w[((off - BDVERS) >> 2) as usize] = val,
            0x6c040..0x6c060 => {
                r.vc1.write(off & 0x1c, val);
                self.dirty.store(true, Ordering::Relaxed);
            }
            0x6c0a0..0x6c100 => {
                r.dac[((off - DAC) >> 5) as usize].write(off & 0x1c, val as u8);
                self.dirty.store(true, Ordering::Relaxed);
            }
            0x6c100..0x6c1a0 => {
                let x = &mut r.xmap[((off - XMAP) >> 5) as usize];
                if off & 0x1c == xmap5::XMAP_CLUT { x.write_clut_packed(val) } else { x.write(off & 0x1c, val) }
                self.dirty.store(true, Ordering::Relaxed);
            }
            0x6c1a0..0x6c1c0 => {
                for x in r.xmap.iter_mut() {
                    if off & 0x1c == xmap5::XMAP_CLUT { x.write_clut_packed(val) } else { x.write(off & 0x1c, val) }
                }
                self.dirty.store(true, Ordering::Relaxed);
            }
            0x6c1c0..0x6c1e0 => r.ab1 = val,
            // ICS PLL bit-bang (GR2.h "Clock"): record the programmed bytes
            // (the dot clock itself isn't modelled).
            0x6c020..0x6c040 => {
                if r.pll_count < 64 {
                    r.pll_bits |= ((val & 1) as u64) << r.pll_count;
                }
                r.pll_count += 1;
                if val & 2 != 0 {
                    let bytes = r.pll_bits.to_le_bytes();
                    r.pll_last = bytes;
                    r.pll_last[7] = r.pll_count.min(255) as u8;
                    r.pll_programs += 1;
                    r.pll_bits = 0;
                    r.pll_count = 0;
                    if self.tracing(debug::TRACE_CPU) {
                        self.trace_note(&format!("clock PLL programmed: {}", debug::pll_describe(&r.pll_last)));
                    }
                }
            }
            // CC1 flat-panel probe writes 0x00/0x55/0xAA; reads never echo.
            0x6c1e0..0x6c200 => {}
            RE3_REGS..RE3_REGS_END => return self.re3_write(((off - RE3_REGS) >> 2) as usize, val),
            0x6c600..0x6c604 => return self.re3_write(re3::REG_RWDATA, val),
            _ => {
                crate::dlog_dev!(crate::devlog::LogModule::Gr2, "GR2: write {:#07x} = {:#010x} (unmapped)", off, val);
            }
        }
        BUS_OK
    }

    fn re3_write(&self, reg: usize, val: u32) -> u32 {
        if !self.re3_fifo.try_push(reg as u32, val as u64) {
            return BUS_BUSY;
        }
        Self::wake(&self.re3_thread);
        BUS_OK
    }

    /// RE3 register reads see the pipeline's result, so they wait (bus retry)
    /// until everything queued ahead of them has executed.
    fn re3_read(&self, reg: usize) -> BusRead32 {
        if !self.idle() {
            return BusRead32::busy();
        }
        // SAFETY: both FIFOs are drained and both engines idle, so the RE3
        // thread is not touching its state.
        let re3 = unsafe { &*self.re3.get() };
        let v = re3.ctx.reg[reg & 63];
        if reg == re3::REG_RWDATA && re3.ctx.stream == re3::STREAM_READ {
            // Consuming RWDATA advances the READBUF stream to the next pixel.
            if !self.re3_fifo.try_push(re3::RE3_OP_READ_ADVANCE, 0) {
                return BusRead32::busy();
            }
            Self::wake(&self.re3_thread);
        }
        BusRead32::ok(v)
    }

    // ── engine threads ───────────────────────────────────────────────────────

    fn hq_loop(&self) {
        struct Sink<'a>(&'a Gr2);
        impl Re3Sink for Sink<'_> {
            fn reg(&mut self, reg: usize, val: u32) {
                self.0.re3_fifo.push(reg as u32 | re3::RE3_SRC_HQ, val as u64);
                Gr2::wake(&self.0.re3_thread);
            }
            fn copy(&mut self, sx: i32, sy: i32, w: i32, h: i32, dx: i32, dy: i32) {
                let a = (sx as u16 as u64) | ((sy as u16 as u64) << 16) | ((w as u16 as u64) << 32) | ((h as u16 as u64) << 48);
                let b = (dx as u16 as u64) | ((dy as u16 as u64) << 16);
                while !self.0.re3_fifo.try_push2(re3::RE3_OP_COPY_A, a, re3::RE3_OP_COPY_B, b) {
                    std::hint::spin_loop();
                }
                Gr2::wake(&self.0.re3_thread);
            }
            fn finish(&mut self, flag: usize) {
                // The host polls the flag to learn the request is done, so it
                // must not be seen before the drawing it follows: drain RE3.
                while !self.0.re3_fifo.is_empty() || self.0.re3_busy.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                self.0.regs().hq.fin[flag].store(1, Ordering::Release);
            }
            fn shram(&mut self, word: usize, val: u32) {
                if let Some(w) = self.0.regs().shram.get_mut(word) {
                    *w = val;
                }
            }
            fn gedma_out(&mut self, words: &[u32]) {
                self.0.gedma_begin();
                for &w in words {
                    if !self.0.gedma_push(w) {
                        break;
                    }
                }
            }
            fn op(&mut self, op: u32, val: u64) {
                self.0.re3_fifo.push(op, val);
                Gr2::wake(&self.0.re3_thread);
            }
            fn read_image(&mut self, req: &hq2::ReadImage, dest: hq2::ReadDest) {
                while !self.0.re3_fifo.is_empty() || self.0.re3_busy.load(Ordering::Acquire) {
                    std::hint::spin_loop();
                }
                // SAFETY: RE3 is drained and idle; VRAM and Z are only read
                // here, as the display thread does.
                let re3 = unsafe { &*self.0.re3.get() };
                let plane: &[u32] = if req.decode == hq2::ReadDecode::Depth { &re3.zbuf } else { &re3.vram };
                let read = |x: i32, y: i32| {
                    if x < 0 || y < 0 || x as usize >= re3::FB_W || y as usize >= re3::FB_H {
                        0
                    } else {
                        plane[y as usize * re3::FB_W + x as usize]
                    }
                };
                match dest {
                    hq2::ReadDest::Shram => {
                        req.pack(read, &mut self.0.regs().shram[hq2::READ_IMAGE_SHRAM..]);
                        self.0.regs().hq.fin[hq2::FIN3].store(1, Ordering::Release);
                    }
                    hq2::ReadDest::Gedma => {
                        let mut words = vec![0u32; req.rows as usize * req.words_per_row as usize];
                        req.pack(read, &mut words);
                        self.gedma_out(&words);
                        // The kernel polls FIN2 once its VDMA has read them.
                        self.0.regs().hq.fin[hq2::FIN2].store(1, Ordering::Release);
                    }
                }
            }
        }
        *self.hq_thread.lock() = Some(thread::current());
        let mut sink = Sink(self);
        let backoff = crossbeam_utils::Backoff::new();
        while self.running.load(Ordering::Relaxed) {
            if let Some((index, val)) = self.hq_fifo.peek() {
                if index == u32::MAX {
                    break;
                }
                self.hq_busy.store(true, Ordering::Release);
                // SAFETY: the HQ2 thread owns the engine.
                let engine = unsafe { &mut *self.hq_engine.get() };
                if self.tracing(debug::TRACE_HQ) {
                    self.trace_hq_entry(index, val as u32);
                    let mut done = |d: String| self.trace_hq_exec(&d);
                    engine.push(index, val as u32, &mut sink, Some(&mut done));
                } else {
                    engine.push(index, val as u32, &mut sink, None);
                }
                if is_fin3_token(index) {
                    // Executed (FIN3 raised if it was a Finish): no longer
                    // pending. Saturating: tests and replays push directly.
                    let _ = self.fin3_pending.fetch_update(Ordering::AcqRel, Ordering::Acquire,
                        |n| Some(n.saturating_sub(1)));
                }
                self.hq_fifo.consume();
                backoff.reset();
            } else {
                self.hq_fifo.flush_head();
                self.hq_busy.store(false, Ordering::Release);
                if backoff.is_completed() {
                    thread::park_timeout(std::time::Duration::from_millis(2));
                } else {
                    backoff.snooze();
                }
            }
        }
        self.hq_busy.store(false, Ordering::Release);
    }

    fn re3_loop(&self) {
        *self.re3_thread.lock() = Some(thread::current());
        let backoff = crossbeam_utils::Backoff::new();
        while self.running.load(Ordering::Relaxed) {
            if let Some((addr, val)) = self.re3_fifo.peek() {
                if addr == re3::RE3_OP_EXIT {
                    break;
                }
                self.re3_busy.store(true, Ordering::Release);
                // SAFETY: the RE3 thread owns `re3` while it runs.
                let re3 = unsafe { &mut *self.re3.get() };
                if self.tracing(debug::TRACE_RE3) {
                    self.trace_re3(addr, val, &re3.ctx.reg);
                }
                let drew = match addr {
                    a if (a & !re3::RE3_SRC_HQ) < 64 => re3.write_reg((a & !re3::RE3_SRC_HQ) as usize, val as u32),
                    re3::RE3_OP_COPY_A => {
                        self.copy_a.set(val);
                        false
                    }
                    re3::RE3_OP_COPY_B => {
                        let a = self.copy_a.get();
                        let f = |v: u64, s: u32| (v >> s) as u16 as i16 as i32;
                        re3.copy_rect(f(a, 0), f(a, 16), f(a, 32), f(a, 48), f(val, 0), f(val, 16));
                        true
                    }
                    re3::RE3_OP_READ_ADVANCE => {
                        re3.read_buffer();
                        false
                    }
                    re3::RE3_OP_PIXFMT => {
                        re3.ctx.pixfmt = val as u32;
                        false
                    }
                    re3::RE3_OP_ZCTL => {
                        re3.ctx.zctl = val as u32;
                        false
                    }
                    re3::RE3_OP_STENCIL => {
                        re3.ctx.stencil = val;
                        false
                    }
                    re3::RE3_OP_BLEND => {
                        re3.ctx.blend = val as u32;
                        false
                    }
                    re3::RE3_OP_ALPHA => {
                        re3.ctx.a = (val as u32) as i64;
                        re3.ctx.da = (val >> 32) as u32 as i32 as i64;
                        false
                    }
                    re3::RE3_OP_ZFILL_A => {
                        re3.ctx.zfill_rect = val;
                        false
                    }
                    re3::RE3_OP_ZFILL_B => {
                        let rect = re3.ctx.zfill_rect;
                        re3.zfill(rect, val);
                        false
                    }
                    _ => false,
                };
                if drew {
                    self.dirty.store(true, Ordering::Relaxed);
                }
                self.re3_fifo.consume();
                backoff.reset();
            } else {
                self.re3_fifo.flush_head();
                self.re3_busy.store(false, Ordering::Release);
                if backoff.is_completed() {
                    thread::park_timeout(std::time::Duration::from_millis(2));
                } else {
                    backoff.snooze();
                }
            }
        }
        self.re3_busy.store(false, Ordering::Release);
    }
}

impl crate::gfx_display::GfxDisplay for Gr2 {
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { &self.renderer }
    fn screen(&self) -> &Mutex<Rex3Screen> { &self.screen }
    fn request_screenshot(&self) { self.screenshot_pending.store(true, Ordering::Relaxed); }
    fn cycles(&self) -> CyclesPtr { self.cycles.get() }
}

impl Resettable for Gr2 {
    fn power_on(&self) {
        let r = self.regs();
        r.ge.count = self.variant.ges();
        for d in r.dac.iter_mut() {
            *d = unsafe { std::mem::zeroed() };
        }
    }
}

impl BusDevice for Gr2 {
    fn read32(&self, addr: u32) -> BusRead32 {
        self.reg_read((addr - GR2_BASE) & (GR2_SLOT_SIZE - 4))
    }
    fn write32(&self, addr: u32, val: u32) -> u32 {
        self.reg_write((addr - GR2_BASE) & (GR2_SLOT_SIZE - 4), val)
    }
    // Narrow accesses hit 8/16-bit peripheral chips on word addresses: the
    // access carries the register value whatever its width or lane (GR2.h).
    fn read8(&self, addr: u32) -> BusRead8 {
        let r = self.read32(addr & !3);
        BusRead8 { status: r.status, data: r.data as u8 }
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        self.reg_write_narrow((addr - GR2_BASE) & (GR2_SLOT_SIZE - 4), val as u32)
    }
    fn read16(&self, addr: u32) -> BusRead16 {
        let r = self.read32(addr & !3);
        BusRead16 { status: r.status, data: r.data as u16 }
    }
    fn write16(&self, addr: u32, val: u16) -> u32 {
        self.reg_write_narrow((addr - GR2_BASE) & (GR2_SLOT_SIZE - 4), val as u32)
    }
    fn read64(&self, addr: u32) -> BusRead64 {
        let hi = self.read32(addr);
        if hi.status != BUS_OK {
            return BusRead64 { status: hi.status, data: 0 };
        }
        let lo = self.read32(addr + 4);
        if lo.status != BUS_OK {
            return BusRead64 { status: lo.status, data: 0 };
        }
        BusRead64::ok(((hi.data as u64) << 32) | lo.data as u64)
    }
    fn write64(&self, addr: u32, val: u64) -> u32 {
        let off = (addr - GR2_BASE) & (GR2_SLOT_SIZE - 8);
        let (hi, lo) = ((val >> 32) as u32, val as u32);
        if (FIFO..FIFO_END).contains(&off) {
            // Both words or neither: a retried store must not push the first twice.
            let idx = (off - FIFO) >> 2;
            let fins = is_fin3_token(idx) as u32 + is_fin3_token(idx + 1) as u32;
            if fins != 0 {
                self.fin3_pending.fetch_add(fins, Ordering::AcqRel);
            }
            if !self.hq_fifo.try_push2(idx, hi as u64, idx + 1, lo as u64) {
                if fins != 0 {
                    self.fin3_pending.fetch_sub(fins, Ordering::AcqRel);
                }
                return BUS_BUSY;
            }
            Self::wake(&self.hq_thread);
            return BUS_OK;
        }
        if (RE3_REGS..RE3_REGS_END).contains(&off) {
            let reg = ((off - RE3_REGS) >> 2) as u32;
            if !self.re3_fifo.try_push2(reg, hi as u64, reg + 1, lo as u64) {
                return BUS_BUSY;
            }
            Self::wake(&self.re3_thread);
            return BUS_OK;
        }
        let st = self.reg_write(off, hi);
        if st != BUS_OK {
            return st;
        }
        self.reg_write(off + 4, lo)
    }
}

impl Device for Gr2 {
    fn step(&self, _cycles: u64) {}

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        // SAFETY: Gr2 lives in the machine's Arc for the process lifetime; the
        // threads are joined in stop() before it could be dropped.
        let me: &'static Gr2 = unsafe { std::mem::transmute::<&Gr2, &'static Gr2>(self) };
        let mut threads = self.threads.lock();
        threads.push(thread::Builder::new().name("GR2-HQ2".into()).spawn(move || me.hq_loop()).unwrap());
        threads.push(thread::Builder::new().name("GR2-RE3".into()).spawn(move || me.re3_loop()).unwrap());
        threads.push(thread::Builder::new().name("GR2-Display".into()).spawn(move || me.display_loop()).unwrap());
    }

    fn stop(&self) {
        if !self.running.swap(false, Ordering::SeqCst) {
            return;
        }
        self.hq_fifo.push(u32::MAX, 0);
        self.re3_fifo.push(re3::RE3_OP_EXIT, 0);
        Self::wake(&self.hq_thread);
        Self::wake(&self.re3_thread);
        for t in self.threads.lock().drain(..) {
            let _ = t.join();
        }
        // Discard the exit sentinels (and anything behind them).
        self.hq_fifo.reset();
        self.fin3_pending.store(0, Ordering::Release);
        self.fin2_wait.store(0, Ordering::Release);
        let head = self.gedma_out_head.load(Ordering::Acquire);
        self.gedma_out_tail.store(head, Ordering::Release);
        self.re3_fifo.reset();
    }

    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    fn get_clock(&self) -> u64 {
        0
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("gr2".into(), "GR2 graphics: status|hq|ge|shram|ucode|vc1|xmap|clut|dac|pix|trace (gr2 help)".into()),
            ("re3".into(), "GR2 RE3 raster engine: regs|pix".into()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn IoWrite + Send>) -> Result<(), String> {
        match cmd {
            "gr2" => self.cmd_gr2(args, &mut *w),
            "re3" => self.cmd_re3(args, &mut *w),
            _ => return Err(format!("unknown gr2 command: {cmd}")),
        }
        .map_err(|e| e.to_string())
    }
}

impl Saveable for Gr2 {
    // Registers, microcode, SRAMs and colour tables. VRAM is not saved yet.
    fn save_state(&self) -> toml::Value {
        let r = self.regs();
        let mut t = toml::map::Map::new();
        t.insert("shram".into(), u32_slice_to_toml(&r.shram));
        t.insert("hq_ucode".into(), u32_slice_to_toml(&r.hq.ucode));
        t.insert("hq_attrjmp".into(), u32_slice_to_toml(&r.hq.attrjmp));
        t.insert("hq_gepc".into(), hex_u32(r.hq.gepc));
        t.insert("hq_numge".into(), hex_u32(r.hq.numge));
        t.insert("hq_running".into(), hex_u32(r.hq.running));
        t.insert("vc1_regs".into(), u8_slice_to_toml(&r.vc1.regs));
        t.insert("vc1_sram".into(), u8_slice_to_toml(&r.vc1.sram));
        t.insert("vc1_sysctl".into(), hex_u32(r.vc1.sysctl as u32));
        for (i, x) in r.xmap.iter().enumerate() {
            t.insert(format!("xmap{i}_mode"), u32_slice_to_toml(&x.mode));
            t.insert(format!("xmap{i}_clut"), u32_slice_to_toml(&x.clut));
            t.insert(format!("xmap{i}_misc"), u8_slice_to_toml(&x.misc));
        }
        for (i, d) in r.dac.iter().enumerate() {
            t.insert(format!("dac{i}_palette"), u8_slice_to_toml(&d.palette));
            t.insert(format!("dac{i}_ctrl"), u8_slice_to_toml(&[d.readmask, d.blinkmask, d.cmd, d.test]));
        }
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let r = self.regs();
        if let Some(x) = get_field(v, "shram") { load_u32_slice(x, &mut r.shram); }
        if let Some(x) = get_field(v, "hq_ucode") { load_u32_slice(x, &mut r.hq.ucode); }
        if let Some(x) = get_field(v, "hq_attrjmp") { load_u32_slice(x, &mut r.hq.attrjmp); }
        if let Some(x) = get_field(v, "hq_gepc") { r.hq.gepc = toml_u32(x).unwrap_or(0); }
        if let Some(x) = get_field(v, "hq_numge") { r.hq.numge = toml_u32(x).unwrap_or(0); }
        if let Some(x) = get_field(v, "hq_running") { r.hq.running = toml_u32(x).unwrap_or(0); }
        if let Some(x) = get_field(v, "vc1_regs") { load_u8_slice(x, &mut r.vc1.regs); }
        if let Some(x) = get_field(v, "vc1_sram") { load_u8_slice(x, &mut r.vc1.sram); }
        if let Some(x) = get_field(v, "vc1_sysctl") { r.vc1.sysctl = toml_u32(x).unwrap_or(0) as u8; }
        for (i, xm) in r.xmap.iter_mut().enumerate() {
            if let Some(x) = get_field(v, &format!("xmap{i}_mode")) { load_u32_slice(x, &mut xm.mode); }
            if let Some(x) = get_field(v, &format!("xmap{i}_clut")) { load_u32_slice(x, &mut xm.clut); }
            if let Some(x) = get_field(v, &format!("xmap{i}_misc")) { load_u8_slice(x, &mut xm.misc); }
        }
        for (i, d) in r.dac.iter_mut().enumerate() {
            if let Some(x) = get_field(v, &format!("dac{i}_palette")) { load_u8_slice(x, &mut d.palette); }
            if let Some(x) = get_field(v, &format!("dac{i}_ctrl")) {
                let mut c = [0u8; 4];
                load_u8_slice(x, &mut c);
                (d.readmask, d.blinkmask, d.cmd, d.test) = (c[0], c[1], c[2], c[3]);
            }
        }
        self.dirty.store(true, Ordering::Relaxed);
        Ok(())
    }
}

/// FIFO tokens that raise FIN3 when executed (GL Finish, 2D sync).
/// Longest a version read stalls for an awaited FIN2 (see Gr2::fin2_wait).
const FIN2_STALL_LIMIT: std::time::Duration = std::time::Duration::from_secs(2);

/// Monotonic host nanoseconds for the FIN2 stall limit.
fn fin2_clock() -> u64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_nanos() as u64
}

fn is_fin3_token(idx: u32) -> bool {
    idx == hq2::GL_FINISH || idx == hq2::HQ_GL_FIN3
}
