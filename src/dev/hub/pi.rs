//! The Hub's processor interface: the node's pending-interrupt banks and
//! masks, the 1.25 MHz real-time counter and one comparator per processor,
//! processor presence, enable, NMI and soft reset, the cached-alias size, the
//! graphics flow-control registers, and the error record of a failed
//! processor read.
//!
//! Offsets are from the base of the Hub window. Pending banks, masks, the
//! counter origin and the comparator pending and enable bits are atomics, so
//! the registers a waiting processor polls take no lock. The remaining
//! register state sits behind one mutex that is never held across an
//! interrupt-line call. A store that the machine must hear of is returned as
//! an [`Effect`] for the Hub to act on.
//!
//! # Interrupt lines
//!
//! The three lines of each processor are one function of the pending banks,
//! the masks and the comparator. Every writer of those inputs takes `lines`,
//! changes its input, computes the lines and writes the processor's interrupt
//! word under it, and writes only the lines that changed. The lock orders the
//! edges, so every edge in a word is a change of the registers and no input
//! moves while the lines are computed. `lines` is a leaf: the word is an
//! atomic nobody calls out of, and the lock is taken inside the state mutex,
//! never the other way round, so nothing in the Hub is called while it is
//! held. Readers stay lock-free.
//!
//! The lines are written inside the hold. That is an exception to HACKING.md's
//! Locks rule 2, which raises interrupts outside a state lock: a writer that
//! sampled the inputs and was then descheduled would write a level the
//! registers no longer held, after another writer's fall and rise. The
//! Interrupts section's order (a controller drives the next one up under its
//! own lock) is the one kept, and it holds only while nothing calls down into
//! the Hub with `lines` held. (A parked processor's wake, with `idle-pause`,
//! takes only the parker's own leaf lock.)
//!
//! # What is approximated here
//!
//! - The reset values are the model's, except those Linux carries. The chip's
//!   power-on values are unknown.
//! - The counter derives from the clock the machine supplies. Linux arms a
//!   comparator by reading the counter and re-reading it to check the deadline
//!   has not passed, so the counter must move between two reads a few
//!   instructions apart: the clock must be fast enough to do so.
//! - A comparator written at or behind the counter does not match until the
//!   counter wraps round to it. A match fires once; only a write re-arms it.
//! - PI_RT_LOCAL_CTRL stores bits 16 and 7:0; nothing written there changes the
//!   counter's rate.
//! - The graphics flow-control registers keep the values written and count no
//!   stores: no credit is charged, no stall and no graphics interrupt follow,
//!   and the interrupt counter and compare are stored only. The field widths
//!   follow the later chip's header and are inferred for this one.
//! - A failed processor read records only the first doubleword of the error
//!   record (valid, overrun and the address); the type, request, supplemental
//!   and command fields stay zero.
//! - The profiling and Hub error interrupts, cross-calls and the
//!   processor-bus checking registers are not modelled.
//! - PI_CPU_NUM names the reader by the CPU thread's index. A store to it is
//!   logged and changes nothing.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

use crate::cpu::mips_exec::SOFT_RESET_BIT;
use crate::cpu::smp::on_cpu_thread;
use crate::dev::heart::Clock;
use super::{Effect, Routing, PENDING_BANKS};
use crate::dev::odyssey::{PiGfx, PI_GFX_REGS};
use crate::devlog::LogModule;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::traits::{BusDevice, BusRead64, Resettable, BUS_OK};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

pub mod reg {
    pub const REGION_PRESENT: u64 = 0x18;
    pub const CPU_NUM: u64 = 0x20;
    pub const CALIAS_SIZE: u64 = 0x28;
    pub const CPU_PRESENT_A: u64 = 0x40;
    pub const CPU_PRESENT_B: u64 = 0x48;
    pub const CPU_ENABLE_A: u64 = 0x50;
    pub const CPU_ENABLE_B: u64 = 0x58;
    pub const NMI_A: u64 = 0x70;
    pub const NMI_B: u64 = 0x78;
    pub const SOFTRESET: u64 = 0x80;
    pub const INT_PEND_MOD: u64 = 0x90;
    pub const INT_PEND0: u64 = 0x98;
    pub const INT_PEND1: u64 = 0xa0;
    pub const INT_MASK0_A: u64 = 0xa8;
    pub const INT_MASK1_A: u64 = 0xb0;
    pub const INT_MASK0_B: u64 = 0xb8;
    pub const INT_MASK1_B: u64 = 0xc0;
    pub const RT_COMPARE_A: u64 = 0x108;
    pub const RT_COMPARE_B: u64 = 0x110;
    pub const RT_PEND_A: u64 = 0x120;
    pub const RT_PEND_B: u64 = 0x128;
    pub const RT_EN_A: u64 = 0x140;
    pub const RT_EN_B: u64 = 0x148;
    pub const RT_LOCAL_CTRL: u64 = 0x160;
    pub const GFX_BASE: u64 = 0x300;
    pub const GFX_END: u64 = 0x350;
    pub const ERR_INT_PEND: u64 = 0x400;
    pub const ERR_INT_MASK_A: u64 = 0x408;
    pub const ERR_INT_MASK_B: u64 = 0x410;
    pub const ERR_STACK_ADDR_A: u64 = 0x418;
    pub const ERR_STACK_ADDR_B: u64 = 0x420;
    pub const ERR_STACK_SIZE: u64 = 0x428;
    pub const ERR_STATUS0_A: u64 = 0x430;
    pub const ERR_STATUS0_A_RCLR: u64 = 0x438;
    pub const ERR_STATUS1_A: u64 = 0x440;
    pub const ERR_STATUS1_A_RCLR: u64 = 0x448;
    pub const ERR_STATUS0_B: u64 = 0x450;
    pub const ERR_STATUS0_B_RCLR: u64 = 0x458;
    pub const ERR_STATUS1_B: u64 = 0x460;
    pub const ERR_STATUS1_B_RCLR: u64 = 0x468;
    pub const TIMEDOUT_A: u64 = 0x480;
    pub const TIMEDOUT_B: u64 = 0x488;
    pub const HARDRESET_BIT: u64 = 0x2_0068;
    pub const RT_COUNT: u64 = 0x3_0100;
}

/// First offset that is not the processor interface's.
pub const END: u64 = 0x20_0000;

/// Interrupt-modify: bits 6:0 name the interrupt, bit 8 set means set.
pub const MODIFY_NUMBER: u64 = 0x7f;
pub const MODIFY_SET: u64 = 0x100;

/// The counter ticks at 1.25 MHz, one tick per 800 ns.
const TICK_NS: u64 = 800;
const RT_MASK: u64 = (1 << 52) - 1;
const RT_LOCAL_CTRL_MASK: u64 = 0x1_00ff;
const RT_LOCAL_CTRL_RESET: u64 = 0x1_0000;
const CALIAS_MASK: u64 = 0xf;
pub(super) const CALIAS_RESET: u64 = 15;
const ERR_MASK: u64 = 0x01ff_ffff;
const ERR_STACK_ADDR_MASK: u64 = 0xffff_fff8;
const ERR_STACK_SIZE_MASK: u64 = 0xf;
const STATUS0_VALID: u64 = 1 << 63;
const STATUS0_OVERRUN: u64 = 1 << 62;
const STATUS0_ADDR_SHIFT: u32 = 25;
const STATUS0_ADDR_MASK: u64 = 0x3fff_ffff_fe00_0000;
const PHYSICAL_BITS: u32 = 40;
/// Interrupt lines per processor: bank 0, bank 1, the comparator.
pub const LINES_PER_CPU: usize = 3;
/// The first Cause bit a processor's lines drive: line 0 is hardware interrupt
/// 0 (IP2), Cause bit 10, and the three lines follow it. The Hub touches no
/// other bit of the word.
const CAUSE_SHIFT: u32 = 10;
const LINE_MASK: u64 = (1 << LINES_PER_CPU) - 1;

/// Bytes the cached alias spans for a PI_CALIAS_SIZE code: zero disables it,
/// one is 4 KiB and each further code doubles it, to 64 MiB at 15.
pub fn calias_bytes(code: u64) -> u64 {
    if code == 0 { 0 } else { (4 << 10) << (code - 1) }
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
    /// fired or while no processor is fitted.
    due: [Option<u64>; 2],
    rt_local_ctrl: u64,
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
            region_present: !0,
            calias: CALIAS_RESET,
            cpu_enable: [false; 2],
            rt_compare: [0; 2],
            due: [None; 2],
            rt_local_ctrl: RT_LOCAL_CTRL_RESET,
            err_mask: [0; 2],
            err_stack_addr: [0; 2],
            err_stack_size: 0,
            err_status0: [0; 2],
            gfx: PiGfx::default(),
            running: false,
        }
    }
}

pub struct Pi {
    cpus: u8,
    cpu_base: usize,
    clock: Clock,
    /// Where the cached-alias size is published, stored under the state lock
    /// as the register changes.
    routing: Arc<Routing>,
    /// Pending banks. Only `modify` changes them while the machine runs, under
    /// `lines`; other processors poll them as mailboxes without it.
    pend: [AtomicU64; PENDING_BANKS],
    mask: [AtomicU64; 4],
    /// The clock reading at which the counter read zero, modulo 2^64.
    rt_origin: AtomicU64,
    /// A match pends whether or not enabled; enable only gates the line.
    rt_pending: [AtomicBool; 2],
    rt_enable: [AtomicBool; 2],
    state: Mutex<State>,
    /// Each processor's interrupt word (`MipsCpu::interrupts_ptr`), where its
    /// three lines are published; null until the machine gives it. The word
    /// must outlive every thread that can publish; see `write_lines`.
    words: [AtomicPtr<AtomicU64>; 2],
    /// Taken by every writer of the inputs of the lines and by the write of the
    /// lines to the words: see the module comment.
    lines: Mutex<()>,
    /// Edges written to the words, per line, counted for the tests.
    #[cfg(test)]
    rises: [AtomicU64; 6],
    #[cfg(test)]
    falls: [AtomicU64; 6],
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
    /// `cpus` is how many processor sockets the node populates (1 or 2) and
    /// `cpu_base` the global index of its slice A, which PI_CPU_NUM subtracts
    /// from the reading processor's.
    pub fn new(cpus: u8, cpu_base: usize, clock: Clock, routing: Arc<Routing>) -> Arc<Self> {
        assert!((1..=2).contains(&cpus));
        let now = clock();
        Arc::new_cyclic(|weak| Self {
            cpus,
            cpu_base,
            clock,
            routing,
            pend: Default::default(),
            mask: Default::default(),
            rt_origin: AtomicU64::new(now),
            rt_pending: Default::default(),
            rt_enable: Default::default(),
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

    /// Where processor `slice`'s lines go: its interrupt word. Line 0 to 2 are
    /// bank 0, bank 1 and the comparator, the processor's hardware interrupts
    /// 0 to 2. The levels as they stand are published into the word now. The
    /// word must outlive every thread that can publish to it: stop them, call
    /// [`Self::clear_cpu_interrupts`], then free the word.
    pub fn set_cpu_interrupts(&self, slice: usize, word: *const AtomicU64) {
        let _lines = self.lines.lock();
        self.words[slice].store(word as *mut AtomicU64, Ordering::SeqCst);
        self.write_lines(slice);
    }

    #[cfg(test)]
    pub(crate) fn publishes_to(&self, slice: usize) -> bool {
        !self.words[slice].load(Ordering::SeqCst).is_null()
    }

    /// Let go of the processors' words: nothing is published to them again.
    pub fn clear_cpu_interrupts(&self) {
        for word in &self.words {
            word.store(std::ptr::null_mut(), Ordering::SeqCst);
        }
    }

    pub fn cpus(&self) -> u8 {
        self.cpus
    }

    /// Whether the node's PI_CPU_ENABLE bit of `slice` is set.
    pub fn cpu_enabled(&self, slice: usize) -> bool {
        slice < 2 && self.state.lock().cpu_enable[slice]
    }

    /// The cached-alias window size the register currently names.
    pub fn calias_bytes(&self) -> u64 {
        self.routing.calias()
    }

    /// The two pending banks, as a processor reads them.
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

    /// The clock reading at which the counter next equals `compare`. The
    /// counter reaches the compare value, never merely passes it: a value
    /// behind the counter waits a whole revolution, and one equal to it too.
    fn deadline(&self, compare: u64, now: u64) -> u64 {
        let count = self.count_at(now);
        let ticks = (compare.wrapping_sub(count).wrapping_sub(1) & RT_MASK) + 1;
        let into_tick = now.wrapping_sub(self.rt_origin.load(Ordering::Relaxed)) % TICK_NS;
        now.saturating_add(ticks * TICK_NS).saturating_sub(into_tick)
    }

    fn arm(&self, s: &mut State, slot: usize, now: u64) {
        // A socket with no processor never arms a timer.
        s.due[slot] = (slot < usize::from(self.cpus)).then(|| self.deadline(s.rt_compare[slot], now));
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
    }

    /// Bring the comparators up to the clock and publish the lines. Tests and
    /// the machine's poll call it; the timer does the same on expiry.
    pub fn poll(&self) {
        self.sync((self.clock)());
        self.drive();
    }

    /// What slice `slot`'s three lines say: bit 0 bank 0 through its mask, bit
    /// 1 bank 1 through its mask, bit 2 the comparator when enabled. Called
    /// with `lines` held, so no input moves while they are read.
    fn outputs(&self, slot: usize) -> u64 {
        let (m0, m1) = (self.mask[slot * 2].load(Ordering::Acquire), self.mask[slot * 2 + 1].load(Ordering::Acquire));
        u64::from(self.pend[0].load(Ordering::Acquire) & m0 != 0)
            | u64::from(self.pend[1].load(Ordering::Acquire) & m1 != 0) << 1
            | u64::from(self.rt_pending[slot].load(Ordering::Acquire) && self.rt_enable[slot].load(Ordering::Acquire)) << 2
    }

    /// Reset processor `slice`: raise bit 63 of its interrupt word, which the
    /// processor takes ahead of every interrupt and answers by fetching the
    /// reset vector. The word is the one the three lines are published into,
    /// so this needs no lock and no machine; a slice with no word wired
    /// resets nothing. What the store does beyond that is not established.
    fn soft_reset(&self, slice: usize) {
        let p = self.words[slice].load(Ordering::SeqCst);
        if p.is_null() {
            return;
        }
        // SAFETY: as in `write_lines`: sound while no thread can reach here
        // after the word is freed, which a machine ensures by stopping its
        // threads before it clears the words.
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
        // SAFETY: `set_cpu_interrupts`'s contract. The pointer was loaded just
        // above, and nothing here keeps `clear_cpu_interrupts` from running
        // before the write, so this is sound only while no thread can reach
        // here after the word is freed. A machine guarantees that by stopping
        // every thread that publishes (processors, devices, the Hub's timers)
        // before it clears the words and frees the processors; the clear then
        // guards a publisher that arrives later, it does not wait for one that
        // is running.
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
            for line in (0..3).filter(|l| (want & !have) >> l & 1 != 0) {
                self.rises[s * 3 + line].fetch_add(1, Ordering::SeqCst);
            }
            #[cfg(feature = "idle-pause")]
            crate::cpu::idle_park::wake();
        }
        if have & !want != 0 {
            word.fetch_and(!((have & !want) << CAUSE_SHIFT), Ordering::SeqCst);
            #[cfg(test)]
            for line in (0..3).filter(|l| (have & !want) >> l & 1 != 0) {
                self.falls[s * 3 + line].fetch_add(1, Ordering::SeqCst);
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
        let Some(next) = s.due.iter().flatten().min().copied() else { return };
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

    /// A processor read failed: record it in that slice's error status 0.
    /// The first failure is kept and a later one only sets overrun. A
    /// debugger's access or a device's DMA does not come here.
    pub fn read_error(&self, slice: usize, physical: u64) {
        if slice >= usize::from(self.cpus) {
            return;
        }
        let address = physical & ((1 << PHYSICAL_BITS) - 1);
        let mut s = self.state.lock();
        if s.err_status0[slice] & STATUS0_VALID != 0 {
            s.err_status0[slice] |= STATUS0_OVERRUN;
        } else {
            s.err_status0[slice] = STATUS0_VALID | (((address >> 3) << STATUS0_ADDR_SHIFT) & STATUS0_ADDR_MASK);
        }
    }

    /// Which of this node's slices the executing thread is, or `None` for a
    /// device or monitor thread, or a processor of another node. A debugger's
    /// access comes from such a thread.
    pub fn reader_slice(&self) -> Option<usize> {
        on_cpu_thread()?.checked_sub(self.cpu_base).filter(|&i| i < usize::from(self.cpus))
    }

    fn slice(&self) -> u64 {
        self.reader_slice().unwrap_or(0) as u64
    }

    /// The value of a register that stores one, without side effects: what a
    /// 32-bit write merges its half with. `None` for a register that stores
    /// nothing a read-modify-write could return.
    pub fn peek(&self, off: u64) -> Option<u64> {
        let s = self.state.lock();
        Some(match off {
            reg::REGION_PRESENT => s.region_present,
            reg::CALIAS_SIZE => s.calias,
            reg::CPU_ENABLE_A | reg::CPU_ENABLE_B => u64::from(s.cpu_enable[((off - reg::CPU_ENABLE_A) / 8) as usize]),
            reg::INT_MASK0_A..=reg::INT_MASK1_B if off % 8 == 0 => {
                self.mask[((off - reg::INT_MASK0_A) / 8) as usize].load(Ordering::Acquire)
            }
            reg::RT_COMPARE_A | reg::RT_COMPARE_B => s.rt_compare[((off - reg::RT_COMPARE_A) / 8) as usize],
            reg::RT_PEND_A | reg::RT_PEND_B => {
                u64::from(self.rt_pending[((off - reg::RT_PEND_A) / 8) as usize].load(Ordering::Acquire))
            }
            reg::RT_EN_A | reg::RT_EN_B => {
                u64::from(self.rt_enable[((off - reg::RT_EN_A) / 8) as usize].load(Ordering::Acquire))
            }
            reg::RT_LOCAL_CTRL => s.rt_local_ctrl,
            reg::GFX_BASE..reg::GFX_END if off % 8 == 0 => s.gfx.read(((off - reg::GFX_BASE) / 8) as u32),
            reg::ERR_INT_MASK_A | reg::ERR_INT_MASK_B => s.err_mask[((off - reg::ERR_INT_MASK_A) / 8) as usize],
            reg::ERR_STACK_ADDR_A | reg::ERR_STACK_ADDR_B => {
                s.err_stack_addr[((off - reg::ERR_STACK_ADDR_A) / 8) as usize]
            }
            reg::ERR_STACK_SIZE => s.err_stack_size,
            reg::ERR_STATUS0_A => s.err_status0[0],
            reg::ERR_STATUS0_B => s.err_status0[1],
            reg::RT_COUNT => self.rt_count(),
            _ => return None,
        })
    }

    /// Read the register at `off`. `clear` is false for a 32-bit half, which
    /// returns a read-to-clear register's half without clearing it: only a
    /// full-width read clears, an approximation that keeps a latch until
    /// software has read all of it.
    pub fn read(&self, off: u64, clear: bool) -> u64 {
        match off {
            reg::CPU_NUM => self.slice(),
            reg::CPU_PRESENT_A => u64::from(self.cpus >= 1),
            reg::CPU_PRESENT_B => u64::from(self.cpus >= 2),
            reg::INT_PEND_MOD => {
                crate::dlog!(LogModule::Hub, "Hub PI: read of the write-only interrupt-modify register");
                0
            }
            // The registers a waiting processor polls take no lock.
            reg::INT_PEND0 => self.pend[0].load(Ordering::Acquire),
            reg::INT_PEND1 => self.pend[1].load(Ordering::Acquire),
            reg::RT_COUNT => self.rt_count(),
            reg::RT_PEND_A | reg::RT_PEND_B => {
                self.sync((self.clock)());
                self.peek(off).unwrap_or(0)
            }
            reg::ERR_INT_PEND | reg::ERR_STATUS1_A | reg::ERR_STATUS1_A_RCLR | reg::ERR_STATUS1_B | reg::ERR_STATUS1_B_RCLR
            | reg::TIMEDOUT_A | reg::TIMEDOUT_B => 0,
            reg::ERR_STATUS0_A_RCLR | reg::ERR_STATUS0_B_RCLR => {
                let slice = usize::from(off == reg::ERR_STATUS0_B_RCLR);
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

    pub fn write(&self, off: u64, value: u64) -> Effect {
        self.write_sized(off, value, 8)
    }

    /// `size` is the guest's access width: only a full-width zero store ends
    /// a processor's error record through a clearing alias.
    /// What the store changed beyond the register is returned for the Hub to
    /// act on, after this unit's lock is released.
    pub fn write_sized(&self, off: u64, value: u64, size: u32) -> Effect {
        let now = || (self.clock)();
        match off {
            reg::REGION_PRESENT => self.state.lock().region_present = value,
            reg::CPU_NUM => {
                crate::dlog!(LogModule::Hub, "Hub PI: write of {value:#x} to the processor-number register (a logic-analyzer convention, not modelled)");
            }
            reg::CALIAS_SIZE => {
                let code = value & CALIAS_MASK;
                let mut s = self.state.lock();
                s.calias = code;
                self.routing.set_calias(calias_bytes(code));
                return Effect::CachedAlias;
            }
            reg::CPU_ENABLE_A | reg::CPU_ENABLE_B => {
                let slice = ((off - reg::CPU_ENABLE_A) / 8) as usize;
                let on = value & 1 != 0;
                self.state.lock().cpu_enable[slice] = on;
                // What the store does to the processor is not established.
                if slice < usize::from(self.cpus) {
                    crate::dlog!(LogModule::Hub, "Hub PI: PI_CPU_ENABLE {slice} = {on} (no effect on the processor)");
                }
            }
            reg::NMI_A | reg::NMI_B => {
                let slice = ((off - reg::NMI_A) / 8) as usize;
                if slice < usize::from(self.cpus) {
                    crate::dlog!(LogModule::Hub, "Hub PI: NMI to slice {slice} (not delivered)");
                }
            }
            reg::SOFTRESET => {
                let enabled = self.state.lock().cpu_enable;
                for slice in (0..usize::from(self.cpus)).filter(|&i| enabled[i]) {
                    self.soft_reset(slice);
                }
            }
            reg::INT_PEND_MOD => self.int_modify(value),
            reg::INT_PEND0 | reg::INT_PEND1 => {
                crate::dlog!(LogModule::Hub, "Hub PI: write of {value:#x} to the read-only pending register {off:#x}");
            }
            reg::INT_MASK0_A..=reg::INT_MASK1_B if off % 8 == 0 => {
                self.change(|| self.mask[((off - reg::INT_MASK0_A) / 8) as usize].store(value, Ordering::Release));
            }
            reg::RT_COMPARE_A | reg::RT_COMPARE_B => {
                let slot = ((off - reg::RT_COMPARE_A) / 8) as usize;
                let mut s = self.state.lock();
                s.rt_compare[slot] = value & RT_MASK;
                self.arm(&mut s, slot, now());
                drop(s);
                self.schedule();
            }
            reg::RT_PEND_A | reg::RT_PEND_B => {
                // The handler acknowledges by writing zero, which drops the line.
                self.change(|| self.rt_pending[((off - reg::RT_PEND_A) / 8) as usize].store(value & 1 != 0, Ordering::Release));
            }
            reg::RT_EN_A | reg::RT_EN_B => {
                self.change(|| self.rt_enable[((off - reg::RT_EN_A) / 8) as usize].store(value & 1 != 0, Ordering::Release));
            }
            reg::RT_LOCAL_CTRL => self.state.lock().rt_local_ctrl = value & RT_LOCAL_CTRL_MASK,
            reg::GFX_BASE..reg::GFX_END if off % 8 == 0 => {
                let n = ((off - reg::GFX_BASE) / 8) as u32;
                debug_assert!(n < PI_GFX_REGS);
                self.state.lock().gfx.write(n, value);
            }
            reg::ERR_INT_PEND => {} // write-one-to-clear; no source can set it
            reg::ERR_INT_MASK_A | reg::ERR_INT_MASK_B => {
                self.state.lock().err_mask[((off - reg::ERR_INT_MASK_A) / 8) as usize] = value & ERR_MASK;
            }
            reg::ERR_STACK_ADDR_A | reg::ERR_STACK_ADDR_B => {
                self.state.lock().err_stack_addr[((off - reg::ERR_STACK_ADDR_A) / 8) as usize] = value & ERR_STACK_ADDR_MASK;
            }
            reg::ERR_STACK_SIZE => self.state.lock().err_stack_size = value & ERR_STACK_SIZE_MASK,
            // Either clearing alias retires its processor's whole record
            // (inferred, from software that clears through the status-1 alias
            // and then expects status 0 to read not valid). Only a
            // full-width zero store does.
            reg::ERR_STATUS0_A_RCLR | reg::ERR_STATUS1_A_RCLR | reg::ERR_STATUS0_B_RCLR | reg::ERR_STATUS1_B_RCLR => {
                if size == 8 && value == 0 {
                    let slice = usize::from(off >= reg::ERR_STATUS0_B);
                    self.state.lock().err_status0[slice] = 0;
                }
            }
            reg::TIMEDOUT_A | reg::TIMEDOUT_B => {
                unimp("timed-out request buffer", off, Some(value));
            }
            reg::HARDRESET_BIT => {
                // Zero selects doubleword credits, the only mode there is.
                if value != 0 {
                    unimp("graphics credit mode", off, Some(value));
                }
            }
            reg::RT_COUNT => {
                let (value, now) = (value & RT_MASK, now());
                self.rt_origin.store(now.wrapping_sub(value * TICK_NS), Ordering::Release);
                // Moving the counter moves every deadline standing against it.
                let mut s = self.state.lock();
                for slot in 0..2 {
                    self.arm(&mut s, slot, now);
                }
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
    /// Everything returns to its reset value except the error stack
    /// addresses, which firmware uses as a warm-start marker. Both processor
    /// enables clear, so the node's processors halt until firmware releases
    /// them.
    fn power_on(&self) {
        let now = (self.clock)();
        let mut s = self.state.lock();
        let (addr, running) = (s.err_stack_addr, s.running);
        *s = State::new();
        self.routing.set_calias(calias_bytes(s.calias));
        s.err_stack_addr = addr;
        s.running = running;
        drop(s);
        self.rt_origin.store(now, Ordering::Release);
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
            }
        });
        self.schedule();
    }
}

impl BusDevice for Pi {
    fn read64(&self, addr: u64) -> BusRead64 {
        BusRead64::ok(self.read(addr, true))
    }
    fn write64(&self, addr: u64, value: u64) -> u32 {
        let _ = self.write(addr, value);
        BUS_OK
    }
}

#[cfg(test)]
mod tests;
