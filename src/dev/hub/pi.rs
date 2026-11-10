//! The Hub's processor interface: the node's pending-interrupt banks and
//! masks, the 1.25 MHz real-time counter and one comparator per processor,
//! processor presence, enable, NMI and soft reset, the cached-alias size, the
//! graphics flow-control registers, and the error record of a failed
//! processor read.
//!
//! Offsets are from the base of the Hub window. Pending banks, masks, the
//! counter origin and the comparator pending and enable bits are atomics, so
//! the registers a waiting processor polls take no lock. The remaining
//! register state sits behind one mutex that is never held across a hook or an
//! interrupt-line call.
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

use crate::cpu::smp::on_cpu_thread;
use crate::dev::heart::Clock;
use super::{HubHooks, HubObserver, Store, PENDING_BANKS};
use crate::dev::odyssey::{PiGfx, PI_GFX_REGS};
use crate::devlog::LogModule;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead64, Resettable, BUS_OK};
use parking_lot::{Mutex, ReentrantMutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
const CALIAS_RESET: u64 = 15;
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
    /// Pending banks. Only `modify` changes them while the machine runs;
    /// other processors poll them as mailboxes, so a bank is published with
    /// release order and read with acquire.
    pend: [AtomicU64; PENDING_BANKS],
    mask: [AtomicU64; 4],
    /// The clock reading at which the counter read zero, modulo 2^64.
    rt_origin: AtomicU64,
    /// A match pends whether or not enabled; enable only gates the line.
    rt_pending: [AtomicBool; 2],
    rt_enable: [AtomicBool; 2],
    state: Mutex<State>,
    hooks: OnceLock<Arc<dyn HubHooks>>,
    observer: OnceLock<Arc<dyn HubObserver>>,
    irq: Mutex<Option<Arc<dyn InterruptController>>>,
    /// Serialises line publication, so two writers cannot publish a stale
    /// level after a fresh one. Reentrant: a controller may read a register.
    dispatch: ReentrantMutex<()>,
    driven: Mutex<[Option<bool>; 2 * LINES_PER_CPU]>,
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
    pub fn new(cpus: u8, cpu_base: usize, clock: Clock) -> Arc<Self> {
        assert!((1..=2).contains(&cpus));
        let now = clock();
        Arc::new_cyclic(|weak| Self {
            cpus,
            cpu_base,
            clock,
            pend: Default::default(),
            mask: Default::default(),
            rt_origin: AtomicU64::new(now),
            rt_pending: Default::default(),
            rt_enable: Default::default(),
            state: Mutex::new(State::new()),
            hooks: OnceLock::new(),
            observer: OnceLock::new(),
            irq: Mutex::new(None),
            dispatch: ReentrantMutex::new(()),
            driven: Mutex::new([None; 2 * LINES_PER_CPU]),
            timers: OnceLock::new(),
            timer: Mutex::new(None),
            weak: weak.clone(),
        })
    }

    pub fn set_hooks(&self, hooks: Arc<dyn HubHooks>) {
        let _ = self.hooks.set(hooks);
    }

    /// Report the stores a processor rendezvous depends on (see [`Store`]).
    pub fn set_observer(&self, observer: Arc<dyn HubObserver>) {
        let _ = self.observer.set(observer);
    }

    fn observe(&self, store: Store) {
        if let Some(observer) = self.observer.get() {
            observer.stored(store);
        }
    }

    /// The machine owns the manager; the interface only keeps a weak handle to
    /// arm the comparator timer.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timers.set(Arc::downgrade(&tm));
        self.schedule();
    }

    /// Lines 0..2 are processor A's hardware interrupts 0 to 2 (bank 0,
    /// bank 1, the comparator) and lines 3..5 are processor B's. All are
    /// levels, recomputed on every change.
    pub fn set_interrupt_controller(&self, irq: Arc<dyn InterruptController>) {
        let _dispatch = self.dispatch.lock();
        *self.irq.lock() = Some(irq);
        *self.driven.lock() = [None; 2 * LINES_PER_CPU];
        self.drive();
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
        calias_bytes(self.state.lock().calias)
    }

    /// The two pending banks, as a processor reads them.
    pub fn pending(&self, bank: usize) -> u64 {
        self.pend[bank].load(Ordering::Acquire)
    }

    fn hooks(&self) -> Option<&Arc<dyn HubHooks>> {
        self.hooks.get()
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
                self.rt_pending[slot].store(true, Ordering::Release);
            }
        }
    }

    /// Bring the comparators up to the clock and publish the lines. Tests and
    /// the machine's poll call it; the timer does the same on expiry.
    pub fn poll(&self) {
        self.sync((self.clock)());
        self.drive();
    }

    fn lines(&self) -> [bool; 2 * LINES_PER_CPU] {
        let mut out = [false; 2 * LINES_PER_CPU];
        for slot in 0..usize::from(self.cpus) {
            let (m0, m1) = (self.mask[slot * 2].load(Ordering::Acquire), self.mask[slot * 2 + 1].load(Ordering::Acquire));
            out[slot * LINES_PER_CPU] = self.pend[0].load(Ordering::Acquire) & m0 != 0;
            out[slot * LINES_PER_CPU + 1] = self.pend[1].load(Ordering::Acquire) & m1 != 0;
            out[slot * LINES_PER_CPU + 2] =
                self.rt_pending[slot].load(Ordering::Acquire) && self.rt_enable[slot].load(Ordering::Acquire);
        }
        out
    }

    /// Publish each line that changed. The sample is taken under the dispatch
    /// lock, so the last publication is the freshest level.
    fn drive(&self) {
        let _dispatch = self.dispatch.lock();
        let Some(irq) = self.irq.lock().clone() else { return };
        let levels = self.lines();
        for (line, &level) in levels.iter().enumerate() {
            let changed = {
                let mut driven = self.driven.lock();
                let changed = driven[line] != Some(level);
                driven[line] = Some(level);
                changed
            };
            if changed {
                irq.set_line(line as u32, level);
            }
        }
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
        let before = if value & MODIFY_SET != 0 {
            self.pend[bank].fetch_or(bit, Ordering::AcqRel)
        } else {
            self.pend[bank].fetch_and(!bit, Ordering::AcqRel)
        };
        self.drive();
        self.observe(Store::IntPendMod { value, changed: (before & bit != 0) != (value & MODIFY_SET != 0) });
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

    pub fn write(&self, off: u64, value: u64) {
        self.write_sized(off, value, 8);
    }

    /// `size` is the guest's access width: only a full-width zero store ends
    /// a processor's error record through a clearing alias.
    pub fn write_sized(&self, off: u64, value: u64, size: u32) {
        let now = || (self.clock)();
        let hooks = self.hooks();
        match off {
            reg::REGION_PRESENT => self.state.lock().region_present = value,
            reg::CPU_NUM => {
                crate::dlog!(LogModule::Hub, "Hub PI: write of {value:#x} to the processor-number register (a logic-analyzer convention, not modelled)");
            }
            reg::CALIAS_SIZE => {
                let code = value & CALIAS_MASK;
                self.state.lock().calias = code;
                if let Some(h) = hooks {
                    h.cached_alias_changed(calias_bytes(code));
                }
            }
            reg::CPU_ENABLE_A | reg::CPU_ENABLE_B => {
                let slice = ((off - reg::CPU_ENABLE_A) / 8) as usize;
                let on = value & 1 != 0;
                self.state.lock().cpu_enable[slice] = on;
                if slice < usize::from(self.cpus) {
                    if let Some(h) = hooks {
                        h.cpu_enable(slice, on);
                    }
                }
                self.observe(Store::CpuEnable { slice, enabled: on });
            }
            reg::NMI_A | reg::NMI_B => {
                let slice = ((off - reg::NMI_A) / 8) as usize;
                if slice < usize::from(self.cpus) {
                    if let Some(h) = hooks {
                        h.nmi(slice);
                    }
                }
            }
            reg::SOFTRESET => {
                let enabled = self.state.lock().cpu_enable;
                for slice in (0..usize::from(self.cpus)).filter(|&i| enabled[i]) {
                    if let Some(h) = hooks {
                        h.soft_reset(slice);
                    }
                }
            }
            reg::INT_PEND_MOD => self.int_modify(value),
            reg::INT_PEND0 | reg::INT_PEND1 => {
                crate::dlog!(LogModule::Hub, "Hub PI: write of {value:#x} to the read-only pending register {off:#x}");
            }
            reg::INT_MASK0_A..=reg::INT_MASK1_B if off % 8 == 0 => {
                self.mask[((off - reg::INT_MASK0_A) / 8) as usize].store(value, Ordering::Release);
                self.drive();
            }
            reg::RT_COMPARE_A | reg::RT_COMPARE_B => {
                let slot = ((off - reg::RT_COMPARE_A) / 8) as usize;
                let mut s = self.state.lock();
                s.rt_compare[slot] = value & RT_MASK;
                self.arm(&mut s, slot, now());
                drop(s);
                self.schedule();
                self.observe(Store::RtCompare { slice: slot, value: value & RT_MASK });
            }
            reg::RT_PEND_A | reg::RT_PEND_B => {
                // The handler acknowledges by writing zero, which drops the line.
                self.rt_pending[((off - reg::RT_PEND_A) / 8) as usize].store(value & 1 != 0, Ordering::Release);
                self.drive();
            }
            reg::RT_EN_A | reg::RT_EN_B => {
                self.rt_enable[((off - reg::RT_EN_A) / 8) as usize].store(value & 1 != 0, Ordering::Release);
                self.drive();
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
    }

    /// Whether the register state is locked now, for the `locks` monitor command.
    pub fn is_locked(&self) -> bool {
        self.state.is_locked()
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
        s.err_stack_addr = addr;
        s.running = running;
        drop(s);
        self.rt_origin.store(now, Ordering::Release);
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
        self.drive();
        self.schedule();
        if let Some(h) = self.hooks() {
            for slice in 0..usize::from(self.cpus) {
                h.cpu_enable(slice, false);
            }
        }
    }
}

impl BusDevice for Pi {
    fn read64(&self, addr: u64) -> BusRead64 {
        BusRead64::ok(self.read(addr, true))
    }
    fn write64(&self, addr: u64, value: u64) -> u32 {
        self.write(addr, value);
        BUS_OK
    }
}

#[cfg(test)]
mod tests;
