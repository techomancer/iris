//! The Hub's processor interface: pending-interrupt banks and masks, the
//! 1.25 MHz real-time counter and comparators, presence, enable, NMI, soft
//! reset, the cached-alias size, graphics flow control and the error record
//! of a failed processor read.
//!
//! Pending banks, masks, the counter origin and the comparators' pending and
//! enable bits are atomics, so registers a waiting processor polls take no
//! lock. The rest sits behind one mutex never held across an interrupt-line
//! call; a store the machine must hear of returns an [`Effect`].
//!
//! # Interrupt lines
//!
//! A processor's four lines are one function of the pending banks, masks and
//! comparators. Every writer of those inputs takes `lines`, changes its input
//! and writes the interrupt word under it, so every edge is a change of the
//! registers. `lines` is a leaf, taken inside the state mutex and never the
//! other way round; readers stay lock-free.
//!
//! Writing the word inside the hold is an exception to raising interrupts
//! outside a state lock: a descheduled writer would otherwise publish a stale
//! level after another writer's fall and rise. It holds only while nothing
//! calls down into the Hub with `lines` held (a parked processor's wake takes
//! only the parker's leaf lock).
//!
//! # What is approximated here
//!
//! - The reset values are the model's, except those Linux carries; the chip's are unknown.
//! - The counter derives from the machine's clock, which must move between two
//!   reads a few instructions apart, as Linux's comparator arming needs.
//! - A comparator written behind the counter, or one the counter is moved
//!   onto, waits for the counter to wrap; one written equal to it matches at
//!   once (IRIX 6.5.21f re-reads the counter after the store and would lose a
//!   deadline that became equal first). The chip's behaviour then is unknown. A match fires once.
//! - The profiling comparator matches the same way, over all 52 bits (the
//!   header gives no width). One match pends both processors, and each
//!   enable gates its own bit onto line 3, Cause IP5.
//! - PI_RT_LOCAL_CTRL stores bits 16 and 7:0; nothing there changes the counter's rate.
//! - The graphics flow-control registers store what is written and charge no
//!   credit, stall or interrupt; field widths follow the later chip's header and are inferred.
//! - A failed processor read records only the first doubleword of the error
//!   record (valid, overrun, address); the other fields stay zero.
//! - The Hub error interrupt, cross-calls and processor-bus checking are not modelled.
//! - PI_CPU_NUM names the reader by CPU thread index; a store is logged and ignored.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

use crate::cpu::mips_exec::SOFT_RESET_BIT;
use crate::cpu::smp::on_cpu_thread;
use crate::dev::heart::Clock;
use super::{Effect, PENDING_BANKS};
use crate::dev::pi_gfx::{PiGfx, PI_GFX_REGS};
use crate::devlog::LogModule;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::traits::Resettable;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use super::regs as reg;
/// The end of the graphics credit registers.
const GFX_END: u64 = reg::PI_GFX_WINDOW::END;

/// First offset that is not the processor interface's.
pub const END: u64 = 0x20_0000;

/// Interrupt-modify: bits 6:0 name the interrupt, bit 8 set means set.
pub const MODIFY_NUMBER: u64 = reg::PI_INT_PEND_MOD::LEVEL::MASK;
pub use crate::sn::MODIFY_SET;

/// The counter ticks at 1.25 MHz, one tick per 800 ns.
const TICK_NS: u64 = 800;
const RT_MASK: u64 = (1 << 52) - 1;
// The counter-rate fields (GCLK, GCLK_COUNT) are not stored.
const RT_LOCAL_CTRL_MASK: u64 = field_masks!(reg::PI_RT_LOCAL_CTRL; USE_INTERNAL, MAX_COUNT, GCLK_EN);
const RT_LOCAL_CTRL_RESET: u64 = 0x1_0000;
const CALIAS_MASK: u64 = 0xf;
pub(super) const CALIAS_RESET: u64 = reg::PI_CALIAS_SIZE::RESET;
const ERR_MASK: u64 = 0x01ff_ffff;
/// The bits of PI_ERR_INT_PEND a write of ones clears: 23:0 (Linux `hubpi.h`).
const ERR_PEND_CLEARABLE: u64 = 0x00ff_ffff;
/// PI_ERR_UNCAC_UNCORR_A and _B (Linux `hubpi.h`: bits 21 and 20).
const ERR_PEND_UNCACHED_LOAD: [u64; 2] = [1 << 21, 1 << 20];
const ERR_STACK_ADDR_MASK: u64 = 0xffff_fff8;
const ERR_STACK_SIZE_MASK: u64 = 0xf;
const STATUS0_VALID: u64 = 1 << 63;
const STATUS0_OVERRUN: u64 = 1 << 62;
const STATUS0_ADDR_SHIFT: u32 = 25;
const STATUS0_ADDR_MASK: u64 = 0x3fff_ffff_fe00_0000;
const PHYSICAL_BITS: u32 = 40;
/// Interrupt lines per processor: bank 0, bank 1, the processor's comparator,
/// the profiling comparator.
pub const LINES_PER_CPU: usize = 4;
/// The first Cause bit a processor's lines drive: line 0 is IP2 (bit 10), the
/// four lines run to IP5 (bit 13). The Hub touches no other bit.
const CAUSE_SHIFT: u32 = 10;
const LINE_MASK: u64 = (1 << LINES_PER_CPU) - 1;

pub use crate::sn::calias_bytes;

/// Index of the 8-byte register at `off` in the array that starts at `base`.
fn idx(off: u64, base: u64) -> usize {
    ((off - base) / 8) as usize
}

fn unimp(what: &str, off: u64, value: Option<u64>) {
    match value {
        Some(v) => crate::dlog!(LogModule::Hub, "Hub PI: write of {v:#x} to unimplemented {what} {off:#x}"),
        None => crate::dlog!(LogModule::Hub, "Hub PI: read of unimplemented {what} {off:#x}"),
    }
}

struct State {
    region_present: u64,
    calias: u64,
    cpu_enable: [bool; 2],
    rt_compare: [u64; 2],
    /// Clock reading at which each comparator matches, `None` once it has
    /// fired or while no processor is attached.
    due: [Option<u64>; 2],
    /// PI_PROFILE_COMPARE, one for the node, and the clock reading at which
    /// it matches, `None` once it has fired.
    prof_compare: u64,
    prof_due: Option<u64>,
    rt_local_ctrl: u64,
    /// PI_ERR_INT_PEND bits 23:0, latched and cleared by writing ones. Bit 24
    /// belongs to the memory-directory unit and is never set here.
    err_pend: u64,
    err_mask: [u64; 2],
    err_stack_addr: [u64; 2],
    err_stack_size: u64,
    err_status0: [u64; 2],
    gfx: PiGfx,
    running: bool,
}
impl State {
    fn new() -> Self {
        Self {
            region_present: reg::PI_REGION_PRESENT::RESET,
            calias: CALIAS_RESET,
            cpu_enable: [reg::PI_CPU_ENABLE_A::RESET != 0, reg::PI_CPU_ENABLE_B::RESET != 0],
            rt_compare: [reg::PI_RT_COMPARE_A::RESET, reg::PI_RT_COMPARE_B::RESET],
            due: [None; 2],
            prof_compare: reg::PI_PROFILE_COMPARE::RESET,
            prof_due: None,
            rt_local_ctrl: RT_LOCAL_CTRL_RESET,
            err_pend: 0,
            err_mask: [reg::PI_ERR_INT_MASK_A::RESET, reg::PI_ERR_INT_MASK_B::RESET],
            err_stack_addr: [reg::PI_ERR_STACK_ADDR_A::RESET, reg::PI_ERR_STACK_ADDR_B::RESET],
            err_stack_size: reg::PI_ERR_STACK_SIZE::RESET,
            err_status0: [0; 2],
            gfx: PiGfx::default(),
            running: false,
        }
    }
}

pub struct Pi {
    cpus: u8,
    first_cpu: usize,
    clock: Clock,
    /// Pending banks. Only `modify` changes them while the machine runs, under
    /// `lines`; other processors poll them as mailboxes without it.
    pend: [AtomicU64; PENDING_BANKS],
    mask: [AtomicU64; 4],
    /// The clock reading at which the counter read zero, modulo 2^64.
    rt_origin: AtomicU64,
    /// A match pends whether or not enabled; enable only gates the line.
    rt_pending: [AtomicBool; 2],
    rt_enable: [AtomicBool; 2],
    /// The profiling comparator's pending and enable bits, per processor. A
    /// match pends both; each enable gates its own processor's line.
    prof_pending: [AtomicBool; 2],
    prof_enable: [AtomicBool; 2],
    state: Mutex<State>,
    /// Each processor's interrupt word, where its four lines are published; null
    /// until the machine gives it. It must outlive every publisher; see `write_lines`.
    words: [AtomicPtr<AtomicU64>; 2],
    /// Taken by every writer of the inputs of the lines and by the write of the
    /// lines to the words: see the module comment.
    lines: Mutex<()>,
    /// Edges written to the words, per line, counted for the tests.
    #[cfg(test)]
    rises: [AtomicU64; 2 * LINES_PER_CPU],
    #[cfg(test)]
    falls: [AtomicU64; 2 * LINES_PER_CPU],
    /// Run once, in the middle of the next write of the lines to a word, for
    /// a test that stalls the writer there.
    #[cfg(test)]
    stall: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    /// Held weakly: the manager's callbacks reach the processor, whose bus
    /// reaches this Hub, so a strong handle here would keep all three alive.
    timers: OnceLock<Weak<TimerManager>>,
    timer: Mutex<Option<TimerId>>,
    weak: Weak<Self>,
}

impl Pi {
    /// `cpus` is the sockets populated (1 or 2); `first_cpu` the global index of
    /// slice A, which PI_CPU_NUM subtracts from the reader's.
    pub fn new(cpus: u8, first_cpu: usize, clock: Clock) -> Arc<Self> {
        assert!((1..=2).contains(&cpus));
        let now = clock();
        Arc::new_cyclic(|weak| Self {
            cpus,
            first_cpu,
            clock,
            pend: Default::default(),
            mask: Default::default(),
            rt_origin: AtomicU64::new(now.wrapping_sub(reg::PI_RT_COUNT::RESET * TICK_NS)),
            rt_pending: Default::default(),
            rt_enable: Default::default(),
            prof_pending: Default::default(),
            prof_enable: Default::default(),
            state: Mutex::new(State::new()),
            words: Default::default(),
            lines: Mutex::new(()),
            #[cfg(test)]
            rises: Default::default(),
            #[cfg(test)]
            falls: Default::default(),
            #[cfg(test)]
            stall: Mutex::new(None),
            timers: OnceLock::new(),
            timer: Mutex::new(None),
            weak: weak.clone(),
        })
    }

    /// The machine owns the manager; the interface only keeps a weak handle to
    /// arm the comparator timer.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timers.set(Arc::downgrade(&tm));
        self.schedule();
    }

    /// Where processor `slice`'s lines (bank 0, bank 1, its comparator, the profiling
    /// one) go. Stop publishers and call [`Self::clear_cpu_interrupts`] before freeing.
    pub fn set_cpu_interrupts(&self, slice: usize, word: *const AtomicU64) {
        let _lines = self.lines.lock();
        self.words[slice].store(word as *mut AtomicU64, Ordering::SeqCst);
        self.write_lines(slice);
    }

    #[cfg(test)]
    pub(crate) fn publishes_to(&self, slice: usize) -> bool {
        !self.words[slice].load(Ordering::SeqCst).is_null()
    }

    /// Let go of the processors' words. A publish already inside [`Self::write_lines`] holds
    /// `lines`; taking it here waits that out, so the words can be freed on return.
    pub fn clear_cpu_interrupts(&self) {
        let _lines = self.lines.lock();
        for word in &self.words {
            word.store(std::ptr::null_mut(), Ordering::SeqCst);
        }
    }

    /// The two pending banks, as a processor reads them.
    #[cfg(test)]
    pub fn pending(&self, bank: usize) -> u64 {
        self.pend[bank].load(Ordering::Acquire)
    }

    fn count_at(&self, now: u64) -> u64 {
        (now.wrapping_sub(self.rt_origin.load(Ordering::Relaxed)) / TICK_NS) & RT_MASK
    }

    /// Where the counter has got to, from the clock rather than stored state:
    /// firmware measures elapsed time by polling this register.
    pub fn rt_count(&self) -> u64 {
        self.count_at((self.clock)())
    }

    /// The clock reading at which the counter next equals `compare`: a value
    /// behind the counter waits a revolution; one `written` while it is held is due now.
    fn deadline(&self, compare: u64, now: u64, written: bool) -> u64 {
        let count = self.count_at(now);
        if written && compare == count {
            return now;
        }
        let ticks = (compare.wrapping_sub(count).wrapping_sub(1) & RT_MASK) + 1;
        let into_tick = now.wrapping_sub(self.rt_origin.load(Ordering::Relaxed)) % TICK_NS;
        now.saturating_add(ticks * TICK_NS).saturating_sub(into_tick)
    }

    fn arm(&self, s: &mut State, slot: usize, now: u64, written: bool) {
        // A socket with no processor never arms a timer.
        s.due[slot] = (slot < usize::from(self.cpus)).then(|| self.deadline(s.rt_compare[slot], now, written));
    }

    /// The profiling comparator's deadline. Unlike a processor's comparator
    /// it serves the node, so it arms whatever sockets are populated.
    fn arm_prof(&self, s: &mut State, now: u64, written: bool) {
        s.prof_due = Some(self.deadline(s.prof_compare, now, written));
    }

    /// Pend every comparator whose deadline has passed.
    fn sync(&self, now: u64) {
        let mut s = self.state.lock();
        for slot in 0..2 {
            if s.due[slot].is_some_and(|due| now >= due) {
                s.due[slot] = None;
                self.change(|| self.rt_pending[slot].store(true, Ordering::Release));
            }
        }
        if s.prof_due.is_some_and(|due| now >= due) {
            s.prof_due = None;
            self.change(|| {
                for pending in &self.prof_pending {
                    pending.store(true, Ordering::Release);
                }
            });
        }
    }

    /// Bring the comparators up to the clock and publish the lines. Tests and
    /// the machine's poll call it; the timer does the same on expiry.
    pub fn poll(&self) {
        self.sync((self.clock)());
        self.drive();
    }

    /// What slice `slot`'s four lines say: banks 0 and 1 through their masks,
    /// then its comparator and the profiling comparator when enabled. Called with `lines` held.
    fn outputs(&self, slot: usize) -> u64 {
        let (m0, m1) = (self.mask[slot * 2].load(Ordering::Acquire), self.mask[slot * 2 + 1].load(Ordering::Acquire));
        u64::from(self.pend[0].load(Ordering::Acquire) & m0 != 0)
            | u64::from(self.pend[1].load(Ordering::Acquire) & m1 != 0) << 1
            | u64::from(self.rt_pending[slot].load(Ordering::Acquire) && self.rt_enable[slot].load(Ordering::Acquire)) << 2
            | u64::from(self.prof_pending[slot].load(Ordering::Acquire) && self.prof_enable[slot].load(Ordering::Acquire)) << 3
    }

    /// Reset processor `slice`: raise bit 63 of its interrupt word, which makes it
    /// fetch the reset vector. An unwired slice resets nothing.
    fn soft_reset(&self, slice: usize) {
        let p = self.words[slice].load(Ordering::SeqCst);
        if p.is_null() {
            return;
        }
        // SAFETY: as in `write_lines`: a machine stops its threads before it
        // clears the words, so none reaches here after the word is freed.
        unsafe { &*p }.fetch_or(SOFT_RESET_BIT, Ordering::SeqCst);
        #[cfg(feature = "idle-pause")]
        crate::cpu::idle_park::wake();
    }

    /// Change an input of the lines and bring every slice's lines to what
    /// the registers say, as one step under `lines`.
    fn change(&self, input: impl FnOnce()) {
        let _lines = self.lines.lock();
        input();
        self.write_all();
    }

    /// Write the lines of slice `s` that differ from the registers, which
    /// are not moving: the caller holds `lines`.
    fn write_lines(&self, s: usize) {
        let p = self.words[s].load(Ordering::SeqCst);
        if p.is_null() {
            return;
        }
        // SAFETY: `set_cpu_interrupts`'s contract. The pointer was loaded under
        // `lines`, which `clear_cpu_interrupts` also takes, so the word outlives this write.
        let word = unsafe { &*p };
        let want = self.outputs(s);
        let have = (word.load(Ordering::SeqCst) >> CAUSE_SHIFT) & LINE_MASK;
        #[cfg(test)]
        {
            let stall = self.stall.lock().take();
            if let Some(stall) = stall {
                stall();
            }
        }
        if want & !have != 0 {
            word.fetch_or((want & !have) << CAUSE_SHIFT, Ordering::SeqCst);
            #[cfg(test)]
            for line in (0..LINES_PER_CPU).filter(|l| (want & !have) >> l & 1 != 0) {
                self.rises[s * LINES_PER_CPU + line].fetch_add(1, Ordering::SeqCst);
            }
            #[cfg(feature = "idle-pause")]
            crate::cpu::idle_park::wake();
        }
        if have & !want != 0 {
            word.fetch_and(!((have & !want) << CAUSE_SHIFT), Ordering::SeqCst);
            #[cfg(test)]
            for line in (0..LINES_PER_CPU).filter(|l| (have & !want) >> l & 1 != 0) {
                self.falls[s * LINES_PER_CPU + line].fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    /// [`Self::write_lines`] for every slice the node populates. The caller
    /// holds `lines`.
    fn write_all(&self) {
        for s in 0..usize::from(self.cpus) {
            self.write_lines(s);
        }
    }

    /// Bring the lines to the registers with no input changed.
    fn drive(&self) {
        let _lines = self.lines.lock();
        self.write_all();
    }

    fn schedule(&self) {
        let Some(tm) = self.timers.get().and_then(Weak::upgrade) else { return };
        let mut timer = self.timer.lock();
        if let Some(id) = timer.take() {
            tm.remove(id);
        }
        let s = self.state.lock();
        if !s.running {
            return;
        }
        let Some(next) = s.due.iter().chain([&s.prof_due]).flatten().min().copied() else { return };
        drop(s);
        let delay = next.saturating_sub((self.clock)()).clamp(1, 1_000_000_000);
        *timer = Some(tm.add_one_shot(Duration::from_nanos(delay), self.weak.clone(), |weak| {
            if let Some(pi) = weak.upgrade() {
                pi.poll();
                pi.schedule();
            }
            TimerReturn::Delete
        }));
    }

    /// Set or clear one pending interrupt, as a write to PI_INT_PEND_MOD
    /// does. Device interrupts arrive here from the Bridge as XIO writes.
    pub fn int_modify(&self, value: u64) {
        let number = value & MODIFY_NUMBER;
        let (bank, bit) = ((number / 64) as usize, 1u64 << (number % 64));
        self.change(|| {
            if value & MODIFY_SET != 0 {
                self.pend[bank].fetch_or(bit, Ordering::AcqRel);
            } else {
                self.pend[bank].fetch_and(!bit, Ordering::AcqRel);
            }
        });
    }

    /// A processor read failed: record it in that slice's error status 0; a
    /// later failure only sets overrun. A debugger or DMA does not come here.
    pub fn read_error(&self, slice: usize, physical: u64) {
        if slice >= usize::from(self.cpus) {
            return;
        }
        let address = physical & ((1 << PHYSICAL_BITS) - 1);
        let mut s = self.state.lock();
        // The failed uncached load is the only source for the pending bit (Linux's
        // "uncached uncorrectable"; PROM 6.156 clears it). Setting it on a bus error is inferred.
        s.err_pend |= ERR_PEND_UNCACHED_LOAD[slice];
        if s.err_status0[slice] & STATUS0_VALID != 0 {
            s.err_status0[slice] |= STATUS0_OVERRUN;
        } else {
            s.err_status0[slice] = STATUS0_VALID | (((address >> 3) << STATUS0_ADDR_SHIFT) & STATUS0_ADDR_MASK);
        }
    }

    /// Which of this node's slices the executing thread is, or `None` for a
    /// device, monitor or other node's processor thread (a debugger's access).
    pub fn reader_slice(&self) -> Option<usize> {
        on_cpu_thread()?.checked_sub(self.first_cpu).filter(|&i| i < usize::from(self.cpus))
    }

    fn slice(&self) -> u64 {
        self.reader_slice().unwrap_or(0) as u64
    }

    /// A stored register's value without side effects, for a 32-bit write's
    /// merge; `None` for a register that stores nothing a read-modify-write could return.
    pub fn peek(&self, off: u64) -> Option<u64> {
        let s = self.state.lock();
        Some(match off {
            reg::PI_REGION_PRESENT::OFFSET => s.region_present,
            reg::PI_CALIAS_SIZE::OFFSET => s.calias,
            reg::PI_CPU_ENABLE_A::OFFSET | reg::PI_CPU_ENABLE_B::OFFSET => u64::from(s.cpu_enable[idx(off, reg::PI_CPU_ENABLE_A::OFFSET)]),
            reg::PI_INT_MASK0_A::OFFSET..=reg::PI_INT_MASK1_B::OFFSET if off % 8 == 0 => {
                self.mask[idx(off, reg::PI_INT_MASK0_A::OFFSET)].load(Ordering::Acquire)
            }
            reg::PI_RT_COMPARE_A::OFFSET | reg::PI_RT_COMPARE_B::OFFSET => s.rt_compare[idx(off, reg::PI_RT_COMPARE_A::OFFSET)],
            reg::PI_RT_PEND_A::OFFSET | reg::PI_RT_PEND_B::OFFSET => {
                u64::from(self.rt_pending[idx(off, reg::PI_RT_PEND_A::OFFSET)].load(Ordering::Acquire))
            }
            reg::PI_RT_EN_A::OFFSET | reg::PI_RT_EN_B::OFFSET => {
                u64::from(self.rt_enable[idx(off, reg::PI_RT_EN_A::OFFSET)].load(Ordering::Acquire))
            }
            reg::PI_PROFILE_COMPARE::OFFSET => s.prof_compare,
            reg::PI_PROF_PEND_A::OFFSET | reg::PI_PROF_PEND_B::OFFSET => {
                u64::from(self.prof_pending[idx(off, reg::PI_PROF_PEND_A::OFFSET)].load(Ordering::Acquire))
            }
            reg::PI_PROF_EN_A::OFFSET | reg::PI_PROF_EN_B::OFFSET => {
                u64::from(self.prof_enable[idx(off, reg::PI_PROF_EN_A::OFFSET)].load(Ordering::Acquire))
            }
            reg::PI_RT_LOCAL_CTRL::OFFSET => s.rt_local_ctrl,
            reg::PI_GFX_PAGE_A::OFFSET..GFX_END if off % 8 == 0 => s.gfx.read(idx(off, reg::PI_GFX_PAGE_A::OFFSET) as u32),
            reg::PI_ERR_INT_PEND::OFFSET => s.err_pend,
            reg::PI_ERR_INT_MASK_A::OFFSET | reg::PI_ERR_INT_MASK_B::OFFSET => s.err_mask[idx(off, reg::PI_ERR_INT_MASK_A::OFFSET)],
            reg::PI_ERR_STACK_ADDR_A::OFFSET | reg::PI_ERR_STACK_ADDR_B::OFFSET => {
                s.err_stack_addr[idx(off, reg::PI_ERR_STACK_ADDR_A::OFFSET)]
            }
            reg::PI_ERR_STACK_SIZE::OFFSET => s.err_stack_size,
            reg::PI_ERR_STATUS0_A::OFFSET => s.err_status0[0],
            reg::PI_ERR_STATUS0_B::OFFSET => s.err_status0[1],
            reg::PI_RT_COUNT::OFFSET => self.rt_count(),
            _ => return None,
        })
    }

    /// Read the register at `off`. `clear` is false for a 32-bit half: only a
    /// full-width read clears a read-to-clear register (an approximation).
    pub fn read(&self, off: u64, clear: bool) -> u64 {
        match off {
            reg::PI_CPU_NUM::OFFSET => self.slice(),
            reg::PI_CPU_PRESENT_A::OFFSET => u64::from(self.cpus >= 1),
            reg::PI_CPU_PRESENT_B::OFFSET => u64::from(self.cpus >= 2),
            reg::PI_INT_PEND_MOD::OFFSET => {
                crate::dlog!(LogModule::Hub, "Hub PI: read of the write-only interrupt-modify register");
                0
            }
            // The registers a waiting processor polls take no lock.
            reg::PI_INT_PEND0::OFFSET => self.pend[0].load(Ordering::Acquire),
            reg::PI_INT_PEND1::OFFSET => self.pend[1].load(Ordering::Acquire),
            reg::PI_RT_PEND_A::OFFSET | reg::PI_RT_PEND_B::OFFSET | reg::PI_PROF_PEND_A::OFFSET | reg::PI_PROF_PEND_B::OFFSET => {
                self.sync((self.clock)());
                self.peek(off).unwrap_or(0)
            }
            reg::PI_ERR_STATUS1_A::OFFSET | reg::PI_ERR_STATUS1_A_RCLR::OFFSET | reg::PI_ERR_STATUS1_B::OFFSET | reg::PI_ERR_STATUS1_B_RCLR::OFFSET
            | reg::PI_CRB_TIMEOUT_A::OFFSET | reg::PI_CRB_TIMEOUT_B::OFFSET => 0,
            reg::PI_ERR_STATUS0_A_RCLR::OFFSET | reg::PI_ERR_STATUS0_B_RCLR::OFFSET => {
                let slice = usize::from(off == reg::PI_ERR_STATUS0_B_RCLR::OFFSET);
                let mut s = self.state.lock();
                let value = s.err_status0[slice];
                if clear {
                    s.err_status0[slice] = 0;
                }
                value
            }
            _ => match self.peek(off) {
                Some(v) => v,
                None => {
                    unimp("register", off, None);
                    0
                }
            },
        }
    }

    /// `size` is the access width: only a full-width zero store ends an error record through a
    /// clearing alias. The Hub acts on the returned effect after the unit's lock is released.
    pub fn write_sized(&self, off: u64, value: u64, size: u32) -> Effect {
        let now = || (self.clock)();
        match off {
            reg::PI_REGION_PRESENT::OFFSET => self.state.lock().region_present = value,
            reg::PI_CPU_NUM::OFFSET => {
                crate::dlog!(LogModule::Hub, "Hub PI: write of {value:#x} to the processor-number register (a logic-analyzer convention, not modelled)");
            }
            reg::PI_CALIAS_SIZE::OFFSET => {
                let code = value & CALIAS_MASK;
                let mut s = self.state.lock();
                s.calias = code;
                return Effect::CachedAlias(calias_bytes(code));
            }
            reg::PI_CPU_ENABLE_A::OFFSET | reg::PI_CPU_ENABLE_B::OFFSET => {
                let slice = idx(off, reg::PI_CPU_ENABLE_A::OFFSET);
                let on = value & 1 != 0;
                self.state.lock().cpu_enable[slice] = on;
                // What the store does to the processor is not established.
                if slice < usize::from(self.cpus) {
                    crate::dlog!(LogModule::Hub, "Hub PI: PI_CPU_ENABLE {slice} = {on} (no effect on the processor)");
                }
            }
            reg::PI_NMI_A::OFFSET | reg::PI_NMI_B::OFFSET => {
                let slice = idx(off, reg::PI_NMI_A::OFFSET);
                if slice < usize::from(self.cpus) {
                    crate::dlog!(LogModule::Hub, "Hub PI: NMI to slice {slice} (not delivered)");
                }
            }
            reg::PI_SOFTRESET::OFFSET => {
                let enabled = self.state.lock().cpu_enable;
                for slice in (0..usize::from(self.cpus)).filter(|&i| enabled[i]) {
                    self.soft_reset(slice);
                }
            }
            reg::PI_INT_PEND_MOD::OFFSET => self.int_modify(value),
            reg::PI_INT_PEND0::OFFSET | reg::PI_INT_PEND1::OFFSET => {
                crate::dlog!(LogModule::Hub, "Hub PI: write of {value:#x} to the read-only pending register {off:#x}");
            }
            reg::PI_INT_MASK0_A::OFFSET..=reg::PI_INT_MASK1_B::OFFSET if off % 8 == 0 => {
                self.change(|| self.mask[idx(off, reg::PI_INT_MASK0_A::OFFSET)].store(value, Ordering::Release));
            }
            reg::PI_RT_COMPARE_A::OFFSET | reg::PI_RT_COMPARE_B::OFFSET => {
                let slot = idx(off, reg::PI_RT_COMPARE_A::OFFSET);
                let mut s = self.state.lock();
                s.rt_compare[slot] = value & RT_MASK;
                self.arm(&mut s, slot, now(), true);
                drop(s);
                self.schedule();
            }
            reg::PI_RT_PEND_A::OFFSET | reg::PI_RT_PEND_B::OFFSET => {
                // The handler acknowledges by writing zero, which drops the line.
                self.change(|| self.rt_pending[idx(off, reg::PI_RT_PEND_A::OFFSET)].store(value & 1 != 0, Ordering::Release));
            }
            reg::PI_RT_EN_A::OFFSET | reg::PI_RT_EN_B::OFFSET => {
                self.change(|| self.rt_enable[idx(off, reg::PI_RT_EN_A::OFFSET)].store(value & 1 != 0, Ordering::Release));
            }
            reg::PI_PROFILE_COMPARE::OFFSET => {
                let mut s = self.state.lock();
                s.prof_compare = value & RT_MASK;
                self.arm_prof(&mut s, now(), true);
                drop(s);
                self.schedule();
            }
            reg::PI_PROF_PEND_A::OFFSET | reg::PI_PROF_PEND_B::OFFSET => {
                // Acknowledged by writing zero, as for the processors' comparators.
                self.change(|| self.prof_pending[idx(off, reg::PI_PROF_PEND_A::OFFSET)].store(value & 1 != 0, Ordering::Release));
            }
            reg::PI_PROF_EN_A::OFFSET | reg::PI_PROF_EN_B::OFFSET => {
                self.change(|| self.prof_enable[idx(off, reg::PI_PROF_EN_A::OFFSET)].store(value & 1 != 0, Ordering::Release));
            }
            reg::PI_RT_LOCAL_CTRL::OFFSET => self.state.lock().rt_local_ctrl = value & RT_LOCAL_CTRL_MASK,
            reg::PI_GFX_PAGE_A::OFFSET..GFX_END if off % 8 == 0 => {
                let n = idx(off, reg::PI_GFX_PAGE_A::OFFSET) as u32;
                debug_assert!(n < PI_GFX_REGS);
                self.state.lock().gfx.write(n, value);
            }
            reg::PI_ERR_INT_PEND::OFFSET => self.state.lock().err_pend &= !(value & ERR_PEND_CLEARABLE),
            reg::PI_ERR_INT_MASK_A::OFFSET | reg::PI_ERR_INT_MASK_B::OFFSET => {
                self.state.lock().err_mask[idx(off, reg::PI_ERR_INT_MASK_A::OFFSET)] = value & ERR_MASK;
            }
            reg::PI_ERR_STACK_ADDR_A::OFFSET | reg::PI_ERR_STACK_ADDR_B::OFFSET => {
                self.state.lock().err_stack_addr[idx(off, reg::PI_ERR_STACK_ADDR_A::OFFSET)] = value & ERR_STACK_ADDR_MASK;
            }
            reg::PI_ERR_STACK_SIZE::OFFSET => self.state.lock().err_stack_size = value & ERR_STACK_SIZE_MASK,
            // Either alias retires the whole record (inferred from software that
            // clears via status 1 and expects status 0 not valid); full-width zero only.
            reg::PI_ERR_STATUS0_A_RCLR::OFFSET | reg::PI_ERR_STATUS1_A_RCLR::OFFSET | reg::PI_ERR_STATUS0_B_RCLR::OFFSET | reg::PI_ERR_STATUS1_B_RCLR::OFFSET => {
                if size == 8 && value == 0 {
                    let slice = usize::from(off >= reg::PI_ERR_STATUS0_B::OFFSET);
                    self.state.lock().err_status0[slice] = 0;
                }
            }
            reg::PI_CRB_TIMEOUT_A::OFFSET | reg::PI_CRB_TIMEOUT_B::OFFSET => {
                unimp("timed-out request buffer", off, Some(value));
            }
            reg::PI_HARDRESET_BIT::OFFSET => {
                // Zero selects doubleword credits, the only mode there is.
                if value != 0 {
                    unimp("graphics credit mode", off, Some(value));
                }
            }
            reg::PI_RT_COUNT::OFFSET => {
                let (value, now) = (value & RT_MASK, now());
                self.rt_origin.store(now.wrapping_sub(value * TICK_NS), Ordering::Release);
                // Moving the counter moves every deadline standing against it.
                let mut s = self.state.lock();
                for slot in 0..2 {
                    self.arm(&mut s, slot, now, false);
                }
                self.arm_prof(&mut s, now, false);
                drop(s);
                self.schedule();
            }
            _ => unimp("register", off, Some(value)),
        }
        Effect::None
    }

    /// Whether the register state is locked now, for the `locks` monitor command.
    pub fn is_locked(&self) -> bool {
        self.state.is_locked() || self.lines.is_locked()
    }

    pub fn start(&self) {
        self.state.lock().running = true;
        self.schedule();
    }

    pub fn stop(&self) {
        self.poll();
        self.state.lock().running = false;
        if let Some(tm) = self.timers.get().and_then(Weak::upgrade) {
            if let Some(id) = self.timer.lock().take() {
                tm.remove(id);
            }
        }
    }

    pub fn is_running(&self) -> bool {
        self.state.lock().running
    }
}

impl Resettable for Pi {
    /// Everything returns to its reset value except the error stack addresses
    /// (firmware's warm-start marker); both enables clear, halting the processors.
    fn power_on(&self) {
        let now = (self.clock)();
        let mut s = self.state.lock();
        let (addr, running) = (s.err_stack_addr, s.running);
        *s = State::new();
        s.err_stack_addr = addr;
        s.running = running;
        drop(s);
        self.rt_origin.store(now.wrapping_sub(reg::PI_RT_COUNT::RESET * TICK_NS), Ordering::Release);
        self.change(|| {
            for bank in &self.pend {
                bank.store(0, Ordering::Release);
            }
            for m in &self.mask {
                m.store(0, Ordering::Release);
            }
            for slot in 0..2 {
                self.rt_pending[slot].store(false, Ordering::Release);
                self.rt_enable[slot].store(false, Ordering::Release);
                self.prof_pending[slot].store(false, Ordering::Release);
                self.prof_enable[slot].store(false, Ordering::Release);
            }
        });
        self.schedule();
    }
}

#[cfg(test)]
mod tests;
