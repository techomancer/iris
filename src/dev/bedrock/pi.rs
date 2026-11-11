//! A Bedrock processor interface: processor-slot control, pending
//! interrupts, the real-time counter and its comparators, the profile timer,
//! error records and the graphics flow-control registers. A node has two
//! PIs with two processor slots each, A and B. The Fuel fits one processor,
//! in PI0 slot A.
//!
//! Each slot drives IP2..IP6 of its processor: bank 0 under the slot's bank
//! 0 mask, bank 1 under its bank 1 mask, its RT comparator, the profile
//! timer and the error summary under its error cause. The inputs of those
//! five levels are atomics. A register store or an inbound packet changes
//! one and then publishes the slot's levels into Cause bits 10-14 of its
//! processor's interrupt word, without a lock; see [`Pi::publish`].
//!
//! # What is approximated here
//!
//! The processor runs from reset, with its CPU-enable bit set, whatever the
//! register later holds: no cold-reset value for the bit is known and the
//! register is not known to stop a processor. A write that clears its enable
//! is stored and logged, not carried out.
//!
//! For a slot with no processor nothing is delivered. A store that arms
//! such a slot is logged under the `bedrock` module instead of being dropped
//! silently: a CPU enable, an NMI, a nonzero interrupt mask (an
//! interprocessor interrupt needs one), a timer enable and an error cause.
//! The register itself is stored, as hardware holds it. No processor model
//! has an NMI entry, so every NMI is logged.
//!
//! The CPU-number register answers the reading processor's slice: its slot
//! in bit 0 and the interface in bit 1. Every read answers slot A, the one
//! processor's.
//!
//! The model accepts Linux's public `SGIrules` key at offset 0x8 for
//! compatibility. Linux documents the literal for Hub I/O, but the available
//! evidence does not establish that this PI offset maps to the same register.
//!
//! The graphics flow-control registers store what software writes and count
//! nothing: no uncached store into the graphics page takes a credit and no
//! board returns one.
//!
//! Ported from hw/misc/sgi-bedrock-pi.c in jamesbraid/qemu-sgi-origami.

use super::reg::IIO_PROTECT_OVRRD_KEY;
use crate::dev::odyssey::{PiGfx, PI_GFX_REGS};
use super::HostTimer;
use crate::hptimer::TimerId;
use parking_lot::Mutex;
use std::sync::atomic::{fence, AtomicPtr, AtomicU64, Ordering::{Acquire, Relaxed, Release, SeqCst}};

pub(super) const RT_MASK: u64 = (1 << 55) - 1;
pub(super) const ERROR_MASK: u64 = (1 << 35) - 1;
pub(super) const MEMORY_ERROR: u64 = 1 << 24;
/// Bank 1 bits that reflect another block's level; a pending-modify write
/// cannot change them.
pub(super) const REFLECT: u64 = (1 << 54) | (1 << 55) | (1 << 62) | (1 << 63);
/// Nanoseconds per real-time counter tick: the counter runs at 1.25 MHz.
pub(super) const TICK: u64 = 800;
/// Length of a PI's register aperture.
pub(super) const SIZE: u64 = 0x30108;
/// IP2 is Cause bit 10.
const CAUSE_IP2: u32 = 10;
/// The request region the requests and the protection key name.
const REQUEST_REGION: u64 = 0;

pub mod reg {
    pub const NODE_REGION: u64 = 0x0;
    pub const PROTECTION_KEY: u64 = 0x8;
    pub const OFFSET_0010: u64 = 0x10;
    pub const OFFSET_0018: u64 = 0x18;
    pub const INTERFACE_ID: u64 = 0x20;
    pub const CACHED_ALIAS: u64 = 0x28;
    pub const OFFSET_0030: u64 = 0x30;
    pub const OFFSET_0038: u64 = 0x38;
    pub const INTERFACE_SLOT_A: u64 = 0x40;
    pub const INTERFACE_SLOT_B: u64 = 0x48;
    pub const CPU_A_ENABLE: u64 = 0x50;
    pub const CPU_B_ENABLE: u64 = 0x58;
    pub const NMI_SLOT_A: u64 = 0x70;
    pub const NMI_SLOT_B: u64 = 0x78;
    pub const PENDING_MODIFY: u64 = 0x90;
    pub const PENDING_STATUS_A: u64 = 0x98;
    pub const PENDING_STATUS_B: u64 = 0xa0;
    pub const OFFSET_00A8: u64 = 0xa8;
    pub const OFFSET_00B0: u64 = 0xb0;
    pub const OFFSET_00B8: u64 = 0xb8;
    pub const OFFSET_00C0: u64 = 0xc0;
    pub const REQUEST_A: u64 = 0xc8;
    pub const REQUEST_B: u64 = 0xd0;
    pub const PENDING_A_CLEAR: u64 = 0xd8;
    pub const PENDING_B_CLEAR: u64 = 0xe0;
    pub const OFFSET_00E8: u64 = 0xe8;
    pub const OFFSET_00F0: u64 = 0xf0;
    pub const RT_COMPARE_A: u64 = 0x108;
    pub const RT_COMPARE_B: u64 = 0x110;
    pub const PROFILE_COMPARE: u64 = 0x118;
    pub const RT_STATUS_A: u64 = 0x120;
    pub const RT_STATUS_B: u64 = 0x128;
    pub const PROFILE_STATUS_A: u64 = 0x130;
    pub const PROFILE_STATUS_B: u64 = 0x138;
    pub const RT_ENABLE_A: u64 = 0x140;
    pub const RT_ENABLE_B: u64 = 0x148;
    pub const PROFILE_ENABLE_A: u64 = 0x150;
    pub const PROFILE_ENABLE_B: u64 = 0x158;
    pub const PENDING_MODIFY_B: u64 = 0x180;
    pub const ERROR_SUMMARY_ALIAS: u64 = 0x3f8;
    pub const ERROR_SUMMARY: u64 = 0x400;
    pub const ERROR_CAUSE_A: u64 = 0x408;
    pub const ERROR_CAUSE_B: u64 = 0x410;
    pub const ERROR_ADDRESS_A: u64 = 0x418;
    pub const ERROR_ADDRESS_B: u64 = 0x420;
    pub const ERROR_CONTROL: u64 = 0x428;
    pub const ERROR_CAPTURE_A: u64 = 0x438;
    pub const ERROR_CAPTURE_B: u64 = 0x448;
    pub const ERROR_CAPTURE_C: u64 = 0x458;
    pub const ERROR_CAPTURE_D: u64 = 0x468;
    pub const ERROR_SOURCE_A: u64 = 0x490;
    pub const ERROR_SOURCE_B: u64 = 0x4c0;
    pub const ERROR_DETAIL_A: u64 = 0x4a8;
    pub const ERROR_DETAIL_B: u64 = 0x4b0;
    pub const ERROR_DETAIL_C: u64 = 0x4b8;
    pub const OFFSET_0430: u64 = 0x430;
    pub const OFFSET_0440: u64 = 0x440;
    pub const OFFSET_0450: u64 = 0x450;
    pub const OFFSET_0460: u64 = 0x460;
    pub const OFFSET_04C8: u64 = 0x4c8;
    pub const OFFSET_04D0: u64 = 0x4d0;
    pub const CREDIT_MODE: u64 = 0x20068;
    pub const COUNTER: u64 = 0x30100;
    pub const GRAPHICS_BASE: u64 = 0x300;
    pub const GRAPHICS_END: u64 = 0x350;
}

/// Bank 0 bit 4 and bank 1 bits 54, 55 and 62: levels other blocks of the
/// chip drive, which both PIs reflect. Each writer sets its own bit under
/// its own lock, so two writers' levels land in register order.
#[derive(Default)]
pub(super) struct Levels([AtomicU64; 2]);
impl Levels {
    pub(super) const UART: (usize, u32) = (0, 4);
    pub(super) const XB_ERROR: (usize, u32) = (1, 54);
    pub(super) const LB_ERROR: (usize, u32) = (1, 55);
    pub(super) const NI_ERROR: (usize, u32) = (1, 62);
    /// Drive `(bank, bit)` to `level`. Whether the bit changed.
    pub(super) fn set(&self, (bank, bit): (usize, u32), level: bool) -> bool {
        let mask = 1u64 << bit;
        let before = if level {
            self.0[bank].fetch_or(mask, SeqCst)
        } else {
            self.0[bank].fetch_and(!mask, SeqCst)
        };
        (before & mask != 0) != level
    }
    fn bank(&self, bank: usize) -> u64 {
        self.0[bank].load(SeqCst)
    }
}

/// The real-time counter's origin, read without a lock. The counter is
/// `base + (time - epoch) / TICK`, `time` being the pause instant while the
/// machine is stopped. A writer holds the PI's timer lock and makes `seq`
/// odd while it changes the three words; a reader retries while `seq` is
/// odd or moved.
struct Counter {
    seq: AtomicU64,
    base: AtomicU64,
    epoch: AtomicU64,
    /// The pause instant, or `RUNNING`.
    paused: AtomicU64,
}
const RUNNING: u64 = u64::MAX;
#[derive(Clone, Copy)]
struct Origin {
    base: u64,
    epoch: u64,
    paused: u64,
}
impl Origin {
    fn time(&self, now: u64) -> u64 {
        if self.paused == RUNNING { now } else { self.paused }
    }
    fn count(&self, now: u64) -> u64 {
        self.base.wrapping_add(self.time(now).saturating_sub(self.epoch) / TICK) & RT_MASK
    }
    /// The start of the tick `time` falls in.
    fn tick_start(&self, time: u64) -> u64 {
        time - time.saturating_sub(self.epoch) % TICK
    }
}
impl Counter {
    fn new(now: u64) -> Self {
        Self { seq: AtomicU64::new(0), base: AtomicU64::new(0), epoch: AtomicU64::new(now), paused: AtomicU64::new(now) }
    }
    fn read(&self) -> Origin {
        loop {
            let seq = self.seq.load(Acquire);
            let origin = Origin { base: self.base.load(Relaxed), epoch: self.epoch.load(Relaxed), paused: self.paused.load(Relaxed) };
            fence(Acquire);
            if seq & 1 == 0 && self.seq.load(Relaxed) == seq {
                return origin;
            }
            std::hint::spin_loop();
        }
    }
    /// Publish `origin`. The caller holds the timer lock, so writers are one at a time.
    fn write(&self, origin: Origin) {
        let seq = self.seq.load(Relaxed);
        self.seq.store(seq + 1, Relaxed);
        fence(Release);
        self.base.store(origin.base, Relaxed);
        self.epoch.store(origin.epoch, Relaxed);
        self.paused.store(origin.paused, Relaxed);
        self.seq.store(seq + 2, Release);
    }
}

/// The comparators and their deadlines, in host nanoseconds, which change
/// together and with the counter's origin.
struct Timer {
    rt_compare: [u64; 2],
    profile_compare: u64,
    rt_due: [u128; 2],
    profile_due: u128,
    /// The host timer that fires at the earliest deadline.
    id: Option<TimerId>,
}

/// The registers that are not interrupt inputs: stored words and the
/// graphics registers.
struct State {
    w: [u64; 0x4d8 / 8],
    gfx: PiGfx,
    credit_mode: u64,
}
impl State {
    fn new() -> Self {
        let mut w = [0; 0x4d8 / 8];
        w[reg::NODE_REGION as usize / 8] = !0;
        w[reg::OFFSET_0018 as usize / 8] = !0;
        Self { w, gfx: PiGfx::default(), credit_mode: 0 }
    }
}

pub(super) struct Pi {
    interface: u8,
    pending: [AtomicU64; 2],
    /// Each slot's bank 0 and bank 1 masks: 0xa8 and 0xb0 for A, 0xb8 and
    /// 0xc0 for B.
    mask: [[AtomicU64; 2]; 2],
    /// Requests A and B and their enable, which make bank 0 bits 5 and 6.
    request: [AtomicU64; 2],
    request_enable: AtomicU64,
    /// The regions an inbound interrupt packet may come from (0x10).
    region: AtomicU64,
    rt_status: [AtomicU64; 2],
    rt_enable: [AtomicU64; 2],
    profile_status: [AtomicU64; 2],
    profile_enable: [AtomicU64; 2],
    error_summary: AtomicU64,
    error_cause: [AtomicU64; 2],
    /// Each slot's processor's interrupt word, null for an empty slot.
    words: [AtomicPtr<AtomicU64>; 2],
    counter: Counter,
    /// The earliest deadline in host nanoseconds; `u64::MAX` while stopped.
    next_due: AtomicU64,
    timer: Mutex<Timer>,
    state: Mutex<State>,
}

impl Pi {
    pub(super) fn new(interface: u8, now: u64) -> Self {
        let pi = Self {
            interface,
            pending: Default::default(),
            mask: Default::default(),
            request: Default::default(),
            request_enable: AtomicU64::new(0),
            region: AtomicU64::new(!0),
            rt_status: Default::default(),
            rt_enable: Default::default(),
            profile_status: Default::default(),
            profile_enable: Default::default(),
            error_summary: AtomicU64::new(0),
            error_cause: Default::default(),
            words: Default::default(),
            counter: Counter::new(now),
            next_due: AtomicU64::new(RUNNING),
            timer: Mutex::new(Timer { rt_compare: [0; 2], profile_compare: 0, rt_due: [0; 2], profile_due: 0, id: None }),
            state: Mutex::new(State::new()),
        };
        let mut s = pi.state.lock();
        for slot in 0..2 {
            s.w[reg::CPU_A_ENABLE as usize / 8 + slot] = u64::from(pi.started(slot));
        }
        drop(s);
        pi.arm_all(&mut pi.timer.lock(), now);
        pi
    }

    pub(super) fn state_locked(&self) -> bool {
        self.state.is_locked()
    }
    pub(super) fn timer_locked(&self) -> bool {
        self.timer.is_locked()
    }
    /// Where slot `slot`'s IP2..IP6 go.
    pub(super) fn set_word(&self, slot: usize, word: *const AtomicU64) {
        self.words[slot].store(word as *mut AtomicU64, SeqCst);
    }

    /// Whether a processor sits in `slot` of this interface: only PI0 slot A
    /// holds one.
    fn started(&self, slot: usize) -> bool {
        self.interface == 0 && slot == 0
    }

    /// Report an access to a register the model does not implement.
    pub(super) fn note(&self, off: u64, write: bool) {
        crate::dlog!(crate::devlog::LogModule::Bedrock,
            "Bedrock PI{} unsupported {} at {off:#x}", self.interface,
            if write { "write" } else { "read" });
    }

    /// Log a store to slot `slot` this interface does not carry out.
    fn refuse(&self, what: &str, slot: usize) {
        crate::dlog!(crate::devlog::LogModule::Bedrock,
            "Bedrock PI{} slot {}: not carried out: {what}", self.interface, if slot == 0 { 'A' } else { 'B' });
    }

    fn pending0(&self, levels: &Levels) -> u64 {
        let enable = self.request_enable.load(SeqCst);
        self.pending[0].load(SeqCst)
            | levels.bank(0)
            | u64::from(self.request[0].load(SeqCst) & enable != 0) << 5
            | u64::from(self.request[1].load(SeqCst) & enable != 0) << 6
    }
    fn pending1(&self, levels: &Levels) -> u64 {
        (self.pending[1].load(SeqCst) & !REFLECT) | levels.bank(1)
    }

    /// Slot `slot`'s IP2..IP6, bit 0 being IP2.
    fn outputs(&self, slot: usize, levels: &Levels) -> u64 {
        u64::from(self.pending0(levels) & self.mask[slot][0].load(SeqCst) != 0)
            | u64::from(self.pending1(levels) & self.mask[slot][1].load(SeqCst) != 0) << 1
            | (self.rt_status[slot].load(SeqCst) & self.rt_enable[slot].load(SeqCst)) << 2
            | (self.profile_status[slot].load(SeqCst) & self.profile_enable[slot].load(SeqCst)) << 3
            | u64::from(self.error_summary.load(SeqCst) & self.error_cause[slot].load(SeqCst) != 0) << 4
    }

    /// Bring each processor's IP2..IP6 to what the registers say. Every
    /// input change is a sequentially consistent store or read-modify-write
    /// followed by a publish, so no lock is needed: take the last write to a
    /// word. Its writer read the inputs again after it, and would have
    /// written again had any change before that read left the word stale. A
    /// change after that read is followed by its own publish, which would
    /// make a later write. So once writers stop, the word matches the
    /// registers. A publish that changes nothing reads the inputs and the
    /// word and stores nothing. Only the PI writes Cause bits 10-14.
    pub(super) fn publish(&self, levels: &Levels) {
        for slot in 0..2 {
            let p = self.words[slot].load(Acquire);
            if p.is_null() {
                continue;
            }
            // SAFETY: `Bedrock::set_cpu_interrupts`' contract: the word
            // outlives the chip.
            let word = unsafe { &*p };
            loop {
                let want = self.outputs(slot, levels);
                let have = (word.load(SeqCst) >> CAUSE_IP2) & 0x1f;
                if want == have {
                    break;
                }
                let (set, clear) = (want & !have, have & !want);
                if set != 0 {
                    word.fetch_or(set << CAUSE_IP2, SeqCst);
                    #[cfg(feature = "idle-pause")]
                    crate::cpu::idle_park::wake();
                }
                if clear != 0 {
                    word.fetch_and(!(clear << CAUSE_IP2), SeqCst);
                }
            }
        }
    }

    /// Set or clear one pending bit, as a pending-modify store does: bit 8
    /// sets, bit 6 picks the bank and bits 5:0 the bit. The reflected bits
    /// of bank 1 are left alone.
    fn modify(&self, value: u64) {
        let bank = ((value >> 6) & 1) as usize;
        let mask = 1u64 << (value & 63);
        if bank == 1 && mask & REFLECT != 0 {
            return;
        }
        if value & 0x100 != 0 {
            self.pending[bank].fetch_or(mask, SeqCst);
        } else {
            self.pending[bank].fetch_and(!mask, SeqCst);
        }
    }

    /// An interrupt packet from region `region` to the pending-modify
    /// register. Refused, and false, unless the region may send one.
    pub(super) fn io_modify(&self, value: u64, region: u8) -> bool {
        if region >= 64 || self.region.load(SeqCst) & (1 << region) == 0 {
            self.note(reg::PENDING_MODIFY, true);
            return false;
        }
        self.modify(value);
        true
    }

    /// The real-time counter at host time `now`. Lock-free.
    pub(super) fn count(&self, now: u64) -> u64 {
        self.counter.read().count(now)
    }

    fn arm_rt(&self, t: &mut Timer, o: &Origin, slot: usize, now: u64) {
        let time = o.time(now);
        let ticks = (t.rt_compare[slot].wrapping_sub(o.count(now)).wrapping_sub(1) & RT_MASK) + 1;
        t.rt_due[slot] = u128::from(o.tick_start(time)) + u128::from(ticks) * u128::from(TICK);
    }
    fn arm_profile(&self, t: &mut Timer, o: &Origin, now: u64) {
        let time = o.time(now);
        let ticks = (t.profile_compare as u32).wrapping_sub(o.count(now) as u32);
        let ticks = if ticks == 0 { 1u64 << 32 } else { u64::from(ticks) };
        t.profile_due = u128::from(o.tick_start(time)) + u128::from(ticks) * u128::from(TICK);
    }
    fn arm_all(&self, t: &mut Timer, now: u64) {
        let o = self.counter.read();
        self.arm_rt(t, &o, 0, now);
        self.arm_rt(t, &o, 1, now);
        self.arm_profile(t, &o, now);
        self.set_next_due(t, &o);
    }
    fn set_next_due(&self, t: &Timer, o: &Origin) {
        let due = if o.paused != RUNNING {
            RUNNING
        } else {
            t.rt_due.iter().copied().chain([t.profile_due]).min().unwrap().min(u128::from(RUNNING - 1)) as u64
        };
        self.next_due.store(due, SeqCst);
    }

    /// Latch the comparators that are due at `now`, under the timer lock.
    /// Whether one latched.
    fn latch(&self, t: &mut Timer, now: u64) -> bool {
        let o = self.counter.read();
        let time = o.time(now);
        let mut latched = false;
        for slot in 0..2 {
            if u128::from(time) >= t.rt_due[slot] {
                self.rt_status[slot].store(1, SeqCst);
                self.arm_rt(t, &o, slot, now);
                latched = true;
            }
        }
        if u128::from(time) >= t.profile_due {
            self.profile_status[0].store(1, SeqCst);
            self.profile_status[1].store(1, SeqCst);
            self.arm_profile(t, &o, now);
            latched = true;
        }
        self.set_next_due(t, &o);
        latched
    }

    /// Latch what is due at `now`: one load when nothing is. Whether a
    /// comparator latched, so the caller publishes.
    pub(super) fn sync(&self, now: u64) -> bool {
        if now < self.next_due.load(SeqCst) {
            return false;
        }
        self.latch(&mut self.timer.lock(), now)
    }

    /// The host timer fired: latch what is due and arm the next deadline.
    pub(super) fn expire(&self, host: &HostTimer, now: u64) -> bool {
        let mut t = self.timer.lock();
        let latched = self.latch(&mut t, now);
        self.arm_host(&mut t, host, now);
        latched
    }

    /// (Re)arm the host timer for the earliest deadline, under the timer
    /// lock. A deadline more than a second out is armed for a second.
    fn arm_host(&self, t: &mut Timer, host: &HostTimer, now: u64) {
        if let Some(id) = t.id.take() {
            host.tm.remove(id);
        }
        let due = self.next_due.load(SeqCst);
        if due == RUNNING {
            return;
        }
        let delay = due.saturating_sub(now).clamp(1, 1_000_000_000);
        t.id = Some(host.arm(self.interface, delay));
    }

    /// Re-arm after the comparators or the counter moved.
    pub(super) fn reschedule(&self, host: Option<&HostTimer>, now: u64) {
        if let Some(host) = host {
            self.arm_host(&mut self.timer.lock(), host, now);
        }
    }

    /// The machine started at `now`: the counter resumes from where it paused.
    pub(super) fn start(&self, host: Option<&HostTimer>, now: u64) {
        let mut t = self.timer.lock();
        let mut o = self.counter.read();
        if o.paused == RUNNING {
            return;
        }
        let delta = now.saturating_sub(o.paused);
        o.epoch = o.epoch.saturating_add(delta);
        o.paused = RUNNING;
        self.counter.write(o);
        for due in &mut t.rt_due {
            *due += u128::from(delta);
        }
        t.profile_due += u128::from(delta);
        self.set_next_due(&t, &o);
        if let Some(host) = host {
            self.arm_host(&mut t, host, now);
        }
    }

    /// The machine stopped at `now`: latch what is due, then freeze the counter.
    pub(super) fn stop(&self, host: Option<&HostTimer>, now: u64) -> bool {
        let mut t = self.timer.lock();
        let latched = self.latch(&mut t, now);
        let mut o = self.counter.read();
        if o.paused == RUNNING {
            o.paused = now;
            self.counter.write(o);
        }
        self.next_due.store(RUNNING, SeqCst);
        if let (Some(host), Some(id)) = (host, t.id.take()) {
            host.tm.remove(id);
        }
        latched
    }

    pub(super) fn is_running(&self) -> bool {
        self.counter.read().paused == RUNNING
    }

    /// The CPU-number register: the reading processor's slot in bit 0, A
    /// for the one processor, and this interface in bit 1.
    fn cpu_number(&self) -> u64 {
        u64::from(self.interface) << 1
    }

    /// A doubleword read at `off`, and whether a comparator latched on the
    /// way, so the caller publishes. `None` for a register the model does
    /// not have.
    pub(super) fn read(&self, off: u64, now: u64, levels: &Levels) -> (Option<u64>, bool) {
        let mut latched = false;
        let value = match off {
            reg::INTERFACE_ID => self.cpu_number(),
            reg::INTERFACE_SLOT_A => u64::from(self.started(0)),
            reg::INTERFACE_SLOT_B => u64::from(self.started(1)),
            reg::OFFSET_0010 => self.region.load(SeqCst),
            reg::PENDING_STATUS_A => self.pending0(levels),
            reg::PENDING_STATUS_B => self.pending1(levels),
            reg::OFFSET_00A8 | reg::OFFSET_00B0 | reg::OFFSET_00B8 | reg::OFFSET_00C0 => {
                let n = (off - reg::OFFSET_00A8) as usize / 8;
                self.mask[n / 2][n % 2].load(SeqCst)
            }
            reg::REQUEST_A | reg::PENDING_A_CLEAR => self.request[0].load(SeqCst),
            reg::REQUEST_B | reg::PENDING_B_CLEAR => self.request[1].load(SeqCst),
            reg::OFFSET_00E8 => self.request_enable.load(SeqCst),
            reg::RT_COMPARE_A | reg::RT_COMPARE_B => self.timer.lock().rt_compare[((off - reg::RT_COMPARE_A) / 8) as usize],
            reg::PROFILE_COMPARE => self.timer.lock().profile_compare,
            reg::RT_STATUS_A | reg::RT_STATUS_B => {
                latched = self.sync(now);
                self.rt_status[((off - reg::RT_STATUS_A) / 8) as usize].load(SeqCst)
            }
            reg::PROFILE_STATUS_A | reg::PROFILE_STATUS_B => {
                latched = self.sync(now);
                self.profile_status[((off - reg::PROFILE_STATUS_A) / 8) as usize].load(SeqCst)
            }
            reg::RT_ENABLE_A | reg::RT_ENABLE_B => self.rt_enable[((off - reg::RT_ENABLE_A) / 8) as usize].load(SeqCst),
            reg::PROFILE_ENABLE_A | reg::PROFILE_ENABLE_B => self.profile_enable[((off - reg::PROFILE_ENABLE_A) / 8) as usize].load(SeqCst),
            reg::ERROR_SUMMARY | reg::ERROR_SUMMARY_ALIAS => self.error_summary.load(SeqCst),
            reg::ERROR_CAUSE_A | reg::ERROR_CAUSE_B => self.error_cause[((off - reg::ERROR_CAUSE_A) / 8) as usize].load(SeqCst),
            reg::COUNTER => self.count(now),
            reg::GRAPHICS_BASE..reg::GRAPHICS_END if off % 8 == 0 => {
                let n = ((off - reg::GRAPHICS_BASE) / 8) as u32;
                debug_assert!(n < PI_GFX_REGS);
                self.state.lock().gfx.read(n)
            }
            reg::CREDIT_MODE => self.state.lock().credit_mode,
            reg::ERROR_CAPTURE_A | reg::ERROR_CAPTURE_B | reg::ERROR_CAPTURE_C | reg::ERROR_CAPTURE_D => {
                self.state.lock().w[(off - reg::PROTECTION_KEY) as usize / 8]
            }
            _ if stored(off) => self.state.lock().w[off as usize / 8],
            _ => {
                self.note(off, false);
                return (None, false);
            }
        };
        (Some(value), latched)
    }

    /// A doubleword store at `off`. Whether the comparators or the counter
    /// moved, so the caller re-arms the host timer. The caller publishes.
    pub(super) fn write(&self, off: u64, value: u64, now: u64) -> bool {
        let index = |base: u64| ((off - base) / 8) as usize;
        match off {
            reg::OFFSET_0010 => self.region.store(value, SeqCst),
            reg::OFFSET_00A8 | reg::OFFSET_00B0 | reg::OFFSET_00B8 | reg::OFFSET_00C0 => {
                let n = index(reg::OFFSET_00A8);
                self.mask[n / 2][n % 2].store(value, SeqCst);
                if value != 0 && !self.started(n / 2) {
                    self.refuse("interrupt mask for an empty slot", n / 2);
                }
            }
            reg::PENDING_MODIFY | reg::PENDING_MODIFY_B => self.modify(value),
            reg::REQUEST_A | reg::REQUEST_B => {
                self.request[index(reg::REQUEST_A)].fetch_or(1 << REQUEST_REGION, SeqCst);
            }
            reg::PENDING_A_CLEAR | reg::PENDING_B_CLEAR => {
                self.request[index(reg::PENDING_A_CLEAR)].fetch_and(!value, SeqCst);
            }
            reg::OFFSET_00E8 => self.request_enable.store(value, SeqCst),
            reg::RT_STATUS_A | reg::RT_STATUS_B => {
                // Latch a deadline that has passed first, so the clear takes it.
                self.sync(now);
                self.rt_status[index(reg::RT_STATUS_A)].store(0, SeqCst);
            }
            reg::PROFILE_STATUS_A | reg::PROFILE_STATUS_B => {
                self.sync(now);
                self.profile_status[index(reg::PROFILE_STATUS_A)].store(0, SeqCst);
            }
            reg::RT_ENABLE_A | reg::RT_ENABLE_B | reg::PROFILE_ENABLE_A | reg::PROFILE_ENABLE_B => {
                let rt = matches!(off, reg::RT_ENABLE_A | reg::RT_ENABLE_B);
                let slot = index(if rt { reg::RT_ENABLE_A } else { reg::PROFILE_ENABLE_A });
                if rt { &self.rt_enable[slot] } else { &self.profile_enable[slot] }.store(value & 1, SeqCst);
                if value & 1 != 0 && !self.started(slot) {
                    self.refuse("timer enable for an empty slot", slot);
                }
            }
            reg::ERROR_SUMMARY_ALIAS => {
                // Software sets every bit but the memory error, which only MD sets.
                let mut summary = self.error_summary.load(SeqCst);
                while let Err(now) = self.error_summary.compare_exchange_weak(
                    summary, (summary & MEMORY_ERROR) | (value & ERROR_MASK & !MEMORY_ERROR), SeqCst, SeqCst,
                ) {
                    summary = now;
                }
            }
            reg::ERROR_SUMMARY => {
                self.error_summary.fetch_and(!(value & ERROR_MASK), SeqCst);
            }
            reg::ERROR_CAUSE_A | reg::ERROR_CAUSE_B => {
                let slot = index(reg::ERROR_CAUSE_A);
                self.error_cause[slot].store(value & ERROR_MASK, SeqCst);
                if value & ERROR_MASK != 0 && !self.started(slot) {
                    self.refuse("error cause for an empty slot", slot);
                }
            }
            reg::RT_COMPARE_A | reg::RT_COMPARE_B | reg::PROFILE_COMPARE | reg::COUNTER => {
                let mut t = self.timer.lock();
                self.latch(&mut t, now);
                let mut o = self.counter.read();
                match off {
                    reg::RT_COMPARE_A | reg::RT_COMPARE_B => {
                        let slot = index(reg::RT_COMPARE_A);
                        t.rt_compare[slot] = value & RT_MASK;
                        self.arm_rt(&mut t, &o, slot, now);
                    }
                    reg::PROFILE_COMPARE => {
                        t.profile_compare = value & 0xffff_ffff;
                        self.arm_profile(&mut t, &o, now);
                    }
                    _ => {
                        o.base = value & RT_MASK;
                        o.epoch = o.time(now);
                        self.counter.write(o);
                        self.arm_rt(&mut t, &o, 0, now);
                        self.arm_rt(&mut t, &o, 1, now);
                        self.arm_profile(&mut t, &o, now);
                    }
                }
                self.set_next_due(&t, &o);
                return true;
            }
            reg::GRAPHICS_BASE..reg::GRAPHICS_END if off % 8 == 0 => {
                let n = ((off - reg::GRAPHICS_BASE) / 8) as u32;
                debug_assert!(n < PI_GFX_REGS);
                self.state.lock().gfx.write(n, value);
            }
            reg::CREDIT_MODE => self.state.lock().credit_mode = value & 1,
            _ => self.write_stored(off, value),
        }
        false
    }

    /// The stores that only change a stored word.
    fn write_stored(&self, off: u64, value: u64) {
        let mut s = self.state.lock();
        let w = |s: &mut State, at: u64, v: u64| s.w[at as usize / 8] = v;
        match off {
            reg::NODE_REGION | reg::OFFSET_0018 | reg::OFFSET_04C8 | reg::OFFSET_04D0 => w(&mut s, off, value),
            reg::PROTECTION_KEY => {
                if value == IIO_PROTECT_OVRRD_KEY {
                    s.w[0] |= 1 << REQUEST_REGION;
                }
            }
            reg::CACHED_ALIAS => w(&mut s, off, value & 15),
            reg::OFFSET_0030 => w(&mut s, off, value & 255),
            reg::OFFSET_0038 => w(&mut s, off, value & 0xffffff),
            reg::CPU_A_ENABLE | reg::CPU_B_ENABLE => {
                let slot = ((off - reg::CPU_A_ENABLE) / 8) as usize;
                w(&mut s, off, value & 1);
                drop(s);
                if value & 1 != 0 && !self.started(slot) {
                    self.refuse("CPU enable for an empty slot", slot);
                } else if value & 1 == 0 && self.started(slot) {
                    self.refuse("CPU disable; the processor keeps running", slot);
                }
            }
            reg::NMI_SLOT_A | reg::NMI_SLOT_B => {
                drop(s);
                if value & 1 != 0 {
                    self.refuse("NMI; no processor model takes one", ((off - reg::NMI_SLOT_A) / 8) as usize);
                } else {
                    self.note(off, true);
                }
            }
            reg::OFFSET_00F0 => {
                w(&mut s, off, value & 31);
                drop(s);
                if value & 31 != 0 {
                    self.note(off, true);
                }
            }
            reg::ERROR_ADDRESS_A | reg::ERROR_ADDRESS_B => w(&mut s, off, value & 0x1fffffff8),
            reg::ERROR_CONTROL => {
                w(&mut s, off, value & 15);
                drop(s);
                if value & 15 != 0 {
                    self.note(off, true);
                }
            }
            reg::ERROR_CAPTURE_A | reg::ERROR_CAPTURE_B | reg::ERROR_CAPTURE_C | reg::ERROR_CAPTURE_D => {
                let slot = usize::from(off >= 0x450);
                s.w[0x430 / 8 + slot * 4] = 0;
                s.w[0x440 / 8 + slot * 4] = 0;
            }
            reg::ERROR_SOURCE_A | reg::ERROR_SOURCE_B => w(&mut s, off, value & 63),
            reg::ERROR_DETAIL_A | reg::ERROR_DETAIL_B => w(&mut s, off, value & 0x1fffff),
            reg::ERROR_DETAIL_C => w(&mut s, off, value & 0xfffff),
            _ => {
                drop(s);
                self.note(off, true);
            }
        }
    }

    /// The PI block's local reset at `now`. The controls and captures
    /// hardware keeps across it survive; the interrupt inputs, timers,
    /// counter and graphics registers start over.
    pub(super) fn power_on(&self, now: u64) {
        let mut s = self.state.lock();
        let old = s.w;
        *s = State::new();
        for off in [0, 0x18, 0x28, 0x50, 0x58, 0x418, 0x420, 0x428, 0x430, 0x440, 0x450, 0x460, 0x4c8, 0x4d0] {
            s.w[off / 8] = old[off / 8];
        }
        drop(s);
        for bank in &self.pending {
            bank.store(0, SeqCst);
        }
        for a in self.mask.iter().flatten().chain(&self.request).chain([&self.request_enable])
            .chain(&self.rt_status).chain(&self.rt_enable).chain(&self.profile_status)
            .chain(&self.profile_enable).chain(&self.error_cause)
        {
            a.store(0, SeqCst);
        }
        let mut t = self.timer.lock();
        t.rt_compare = [0; 2];
        t.profile_compare = 0;
        let o = self.counter.read();
        let time = o.time(now);
        self.counter.write(Origin { base: 0, epoch: time, paused: o.paused });
        self.arm_all(&mut t, now);
    }
}

/// The doubleword registers that only hold what was stored.
fn stored(off: u64) -> bool {
    matches!(
        off,
        reg::NODE_REGION
            | reg::OFFSET_0018
            | reg::CACHED_ALIAS
            | reg::OFFSET_0030
            | reg::OFFSET_0038
            | reg::CPU_A_ENABLE
            | reg::CPU_B_ENABLE
            | reg::OFFSET_00F0
            | reg::ERROR_ADDRESS_A
            | reg::ERROR_ADDRESS_B
            | reg::ERROR_CONTROL
            | reg::OFFSET_0430
            | reg::OFFSET_0440
            | reg::OFFSET_0450
            | reg::OFFSET_0460
            | reg::ERROR_SOURCE_A
            | reg::ERROR_DETAIL_A
            | reg::ERROR_DETAIL_B
            | reg::ERROR_DETAIL_C
            | reg::ERROR_SOURCE_B
            | reg::OFFSET_04C8
            | reg::OFFSET_04D0
    )
}

#[cfg(test)]
mod tests;
