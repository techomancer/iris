//! Bedrock PI exposes processor-slot control, pending interrupts, real-time
//! counters, timers and shared graphics registers. Fuel uses PI0 slot A for
//! its processor; the other processor slot has no device attached. Node hooks
//! carry cache-alias changes, CPU enable and NMI events to the board.
//!
//! # What is approximated here
//!
//! The model accepts Linux's public `SGIrules` key at offset 0x8 for
//! compatibility. Linux documents the literal for Hub I/O, but the available
//! evidence does not establish that this PI offset maps to the same register.
//! Unconnected graphics registers read as zero and are recorded as unsupported.
//!
//! Ported from hw/misc/sgi-bedrock-pi.c in jamesbraid/qemu-sgi-origami.

use crate::dev::bedrock_io::IIO_PROTECT_OVRRD_KEY;
use crate::dev::heart::Clock;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead32, BusRead64, Device, Resettable, BUS_ERR, BUS_OK};
use parking_lot::{Mutex, ReentrantMutex};
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;
const RT_MASK: u64 = (1 << 55) - 1;
const ERROR_MASK: u64 = (1 << 35) - 1;
const MEMORY_ERROR: u64 = 1 << 24;
const REFLECT: u64 = (1 << 54) | (1 << 55) | (1 << 62) | (1 << 63);
const TICK: u64 = 800;
pub const SIZE: u64 = 0x30108;

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

#[derive(Clone, Copy, Debug, Default)]
pub struct BedrockPiConfig {
    pub interface: u8,
    pub request_region: u8,
}
/// Called outside the state lock. Graphics indices 0..9 are the shared PI
/// graphics registers at offsets 0x300..0x348. Disconnected access is observable.
pub trait NodeHooks: Send + Sync {
    fn cached_alias_changed(&self, _size: u8) {}
    fn rt_compare_written(&self, _interface: u8, _slot: usize) {}
    fn rt_counter_read(&self, _interface: u8) {}
    fn cpu_enable(&self, _interface: u8, _slot: usize, _enabled: bool) {}
    fn nmi(&self, _interface: u8, _slot: usize) {}
    fn gfx_read(&self, _index: usize) -> Option<u64> {
        None
    }
    fn gfx_write(&self, _index: usize, _value: u64) -> bool {
        false
    }
    fn gfx_reset(&self) {}
}
struct State {
    w: [u64; 160],
    pending: [u64; 2],
    reflected: u64,
    uart: bool,
    base: u64,
    epoch: u64,
    paused: Option<u64>,
    running: bool,
    rt_due: [u128; 2],
    profile_due: u128,
    levels: [u8; 256],
    rearm: [u64; 128],
    credit_mode: u64,
}
impl State {
    fn new(now: u64) -> Self {
        let mut s = Self {
            w: [0; 160],
            pending: [0; 2],
            reflected: 0,
            uart: false,
            base: 0,
            epoch: now,
            paused: Some(now),
            running: false,
            rt_due: [0; 2],
            profile_due: 0,
            levels: [0; 256],
            rearm: [0; 128],
            credit_mode: 0,
        };
        s.w[0] = !0;
        s.w[2] = !0;
        s.w[3] = !0;
        s.arm_all(now);
        s
    }
    fn time(&self, now: u64) -> u64 {
        self.paused.unwrap_or(now)
    }
    fn count(&self, now: u64) -> u64 {
        self.base
            .wrapping_add(self.time(now).saturating_sub(self.epoch) / TICK)
            & RT_MASK
    }
    fn arm_rt(&mut self, slot: usize, now: u64) {
        let now = self.time(now);
        let ticks = (self.w[0x108 / 8 + slot]
            .wrapping_sub(self.count(now))
            .wrapping_sub(1)
            & RT_MASK)
            + 1;
        self.rt_due[slot] = u128::from(now - now.saturating_sub(self.epoch) % TICK)
            + u128::from(ticks) * u128::from(TICK);
    }
    fn arm_profile(&mut self, now: u64) {
        let now = self.time(now);
        let ticks = (self.w[0x118 / 8] as u32).wrapping_sub(self.count(now) as u32);
        let ticks = if ticks == 0 {
            1u64 << 32
        } else {
            u64::from(ticks)
        };
        self.profile_due = u128::from(now - now.saturating_sub(self.epoch) % TICK)
            + u128::from(ticks) * u128::from(TICK);
    }
    fn arm_all(&mut self, now: u64) {
        self.arm_rt(0, now);
        self.arm_rt(1, now);
        self.arm_profile(now);
    }
    fn sync(&mut self, now: u64) {
        let now = self.time(now);
        for slot in 0..2 {
            if u128::from(now) >= self.rt_due[slot] {
                self.w[0x120 / 8 + slot] = 1;
                self.arm_rt(slot, now);
            }
        }
        if u128::from(now) >= self.profile_due {
            self.w[0x130 / 8] = 1;
            self.w[0x138 / 8] = 1;
            self.arm_profile(now);
        }
        for level in 7..128 {
            if self.rearm[level] != 0 && self.rearm[level] <= now {
                self.rearm[level] = 0;
                if self.levels.iter().any(|&v| usize::from(v) == level + 1) {
                    self.pending[level / 64] |= 1 << (level % 64);
                }
            }
        }
    }
    fn pending(&self, bank: usize) -> u64 {
        if bank == 1 {
            (self.pending[1] & !REFLECT) | self.reflected
        } else {
            self.pending[0]
                | if self.uart { 1 << 4 } else { 0 }
                | if self.w[0xc8 / 8] & self.w[0xe8 / 8] != 0 {
                    1 << 5
                } else {
                    0
                }
                | if self.w[0xd0 / 8] & self.w[0xe8 / 8] != 0 {
                    1 << 6
                } else {
                    0
                }
        }
    }
    fn outputs(&self) -> [bool; 10] {
        let mut out = [false; 10];
        for slot in 0..2 {
            out[slot * 5] = self.pending(0) & self.w[0xa8 / 8 + slot * 2] != 0;
            out[slot * 5 + 1] = self.pending(1) & self.w[0xb0 / 8 + slot * 2] != 0;
            out[slot * 5 + 2] = self.w[0x120 / 8 + slot] & self.w[0x140 / 8 + slot] != 0;
            out[slot * 5 + 3] = self.w[0x130 / 8 + slot] & self.w[0x150 / 8 + slot] != 0;
            out[slot * 5 + 4] = self.w[0x400 / 8] & self.w[0x408 / 8 + slot] != 0;
        }
        out
    }
    fn modify(&mut self, value: u64, reissue: bool, now: u64) {
        let bank = ((value >> 6) & 1) as usize;
        let bit = value & 63;
        let mask = 1 << bit;
        if bank == 1 && mask & REFLECT != 0 {
            return;
        }
        if value & 0x100 != 0 {
            self.pending[bank] |= mask;
        } else {
            self.pending[bank] &= !mask;
            let level = bank * 64 + bit as usize;
            if reissue && self.levels.iter().any(|&v| usize::from(v) == level + 1) {
                self.rearm[level] = now.saturating_add(1_000_000);
            }
        }
    }
}
pub struct BedrockPi {
    cfg: BedrockPiConfig,
    clock: Clock,
    state: Mutex<State>,
    hooks: Mutex<Option<Arc<dyn NodeHooks>>>,
    irq: Mutex<Option<Arc<dyn InterruptController>>>,
    timers: OnceLock<Arc<TimerManager>>,
    timer: Mutex<Option<TimerId>>,
    timer_pass: Mutex<()>,
    dispatch: ReentrantMutex<()>,
    driven: Mutex<[Option<bool>; 10]>,
    weak: Weak<Self>,
    unsupported: Mutex<Vec<(u64, bool)>>,
}
impl BedrockPi {
    pub fn new(cfg: BedrockPiConfig, clock: Clock) -> Arc<Self> {
        assert!(cfg.interface < 2 && cfg.request_region < 64);
        Arc::new_cyclic(|weak| Self {
            cfg,
            state: Mutex::new(State::new(clock())),
            clock,
            hooks: Mutex::new(None),
            irq: Mutex::new(None),
            timers: OnceLock::new(),
            timer: Mutex::new(None),
            timer_pass: Mutex::new(()),
            dispatch: ReentrantMutex::new(()),
            driven: Mutex::new([None; 10]),
            weak: weak.clone(),
            unsupported: Mutex::new(Vec::new()),
        })
    }
    pub fn set_node_hooks(&self, hooks: Arc<dyn NodeHooks>) {
        let _dispatch = self.dispatch.lock();
        *self.hooks.lock() = Some(hooks);
    }
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.timers.set(tm);
        self.schedule();
    }
    /// Lines 0..4 are slot A IP2..IP6; lines 5..9 are slot B IP2..IP6.
    pub fn set_interrupt_controller(&self, irq: Arc<dyn InterruptController>) {
        let _dispatch = self.dispatch.lock();
        *self.irq.lock() = Some(irq);
        *self.driven.lock() = [None; 10];
        self.drive();
    }
    fn hooks(&self) -> Option<Arc<dyn NodeHooks>> {
        self.hooks.lock().clone()
    }
    fn note(&self, off: u64, write: bool) {
        let mut u = self.unsupported.lock();
        if u.len() < 64 && !u.contains(&(off, write)) {
            u.push((off, write));
        }
    }
    pub fn unsupported(&self) -> Vec<(u64, bool)> {
        self.unsupported.lock().clone()
    }
    fn drive(&self) {
        // Serialize publication, including reentrant reads by a controller.
        // Publish one line at a time: a callback may change another line.
        let _dispatch = self.dispatch.lock();
        let irq = self.irq.lock().clone();
        if let Some(irq) = irq {
            for line in 0..10 {
                let level = self.state.lock().outputs()[line];
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
    }
    fn schedule(&self) {
        let _pass = self.timer_pass.lock();
        let Some(tm) = self.timers.get() else { return };
        let mut timer = self.timer.lock();
        if let Some(id) = timer.take() {
            tm.remove(id);
        }
        let s = self.state.lock();
        if !s.running {
            return;
        }
        let next = s
            .rt_due
            .iter()
            .copied()
            .chain([s.profile_due])
            .chain(s.rearm.iter().filter(|&&v| v != 0).map(|&v| u128::from(v)))
            .min()
            .unwrap();
        let delay = next
            .saturating_sub(u128::from((self.clock)()))
            .min(1_000_000_000)
            .max(1) as u64;
        drop(s);
        *timer = Some(
            tm.add_one_shot(Duration::from_nanos(delay), self.weak.clone(), |weak| {
                if let Some(pi) = weak.upgrade() {
                    {
                        let _pass = pi.timer_pass.lock();
                        let mut s = pi.state.lock();
                        if !s.running {
                            return TimerReturn::Delete;
                        }
                        s.sync((pi.clock)());
                    }
                    pi.drive();
                    pi.schedule();
                }
                TimerReturn::Delete
            }),
        );
    }
    pub fn poll(&self) {
        self.state.lock().sync((self.clock)());
        self.drive();
    }
    pub fn set_pending(&self, bank: usize, bit: u8) {
        if bank < 2 && bit < 64 {
            self.state.lock().pending[bank] |= 1 << bit;
            self.drive();
        }
    }
    fn reflected(&self, bit: u8, level: bool) {
        let mut s = self.state.lock();
        if level {
            s.reflected |= 1 << bit;
        } else {
            s.reflected &= !(1 << bit);
        }
        drop(s);
        self.drive();
    }
    pub fn set_lb_error(&self, level: bool) {
        self.reflected(55, level);
    }
    pub fn set_ni_error(&self, level: bool) {
        self.reflected(62, level);
    }
    pub fn set_xb_error(&self, level: bool) {
        self.reflected(54, level);
    }
    pub fn set_uart_irq(&self, level: bool) {
        self.state.lock().uart = level;
        self.drive();
    }
    pub fn set_memory_uncorrectable(&self) {
        self.state.lock().w[0x400 / 8] |= MEMORY_ERROR;
        self.drive();
    }
    pub fn set_level_source(&self, source: usize, level: u8, active: bool) {
        if source < 256 && (!active || (7..=127).contains(&level)) {
            self.state.lock().levels[source] = if active { level + 1 } else { 0 };
            self.drive();
        }
    }
    pub fn io_modify(&self, value: u64, region: u8) -> bool {
        let mut s = self.state.lock();
        if region >= 64 || s.w[2] & (1 << region) == 0 {
            drop(s);
            self.note(0x90, true);
            return false;
        }
        let now = s.time((self.clock)());
        s.modify(value, false, now);
        drop(s);
        self.drive();
        true
    }
    fn read(&self, off: u64) -> u64 {
        if (reg::GRAPHICS_BASE..reg::GRAPHICS_END).contains(&off) {
            if let Some(v) = self
                .hooks()
                .and_then(|h| h.gfx_read(((off - 0x300) / 8) as usize))
            {
                return v;
            }
            self.note(off, false);
            return 0;
        }
        let mut s = self.state.lock();
        s.sync((self.clock)());
        let value = match off {
            reg::INTERFACE_ID => u64::from(self.cfg.interface) << 1,
            reg::INTERFACE_SLOT_A => u64::from(self.cfg.interface == 0),
            reg::INTERFACE_SLOT_B => 0,
            reg::PENDING_STATUS_A => s.pending(0),
            reg::PENDING_STATUS_B => s.pending(1),
            reg::PENDING_A_CLEAR => s.w[reg::REQUEST_A as usize / 8],
            reg::PENDING_B_CLEAR => s.w[reg::REQUEST_B as usize / 8],
            reg::ERROR_SUMMARY_ALIAS => s.w[reg::ERROR_SUMMARY as usize / 8],
            reg::ERROR_CAPTURE_A | reg::ERROR_CAPTURE_B | reg::ERROR_CAPTURE_C | reg::ERROR_CAPTURE_D => s.w[(off - reg::PROTECTION_KEY) as usize / 8],
            reg::CREDIT_MODE => s.credit_mode,
            reg::COUNTER => s.count((self.clock)()),
            _ if readable(off) => s.w[off as usize / 8],
            _ => {
                self.note(off, false);
                0
            }
        };
        drop(s);
        self.drive();
        if off == reg::COUNTER {
            if let Some(h) = self.hooks() {
                h.rt_counter_read(self.cfg.interface);
            }
        }
        value
    }
    fn write(&self, off: u64, value: u64) {
        // Keep register changes and their node notifications in one order.
        // The state lock still drops before callbacks; dispatch is reentrant
        // so a node callback can read registers and recompute IRQ outputs.
        let _dispatch = self.dispatch.lock();
        if (reg::GRAPHICS_BASE..reg::GRAPHICS_END).contains(&off) {
            if !self
                .hooks()
                .is_some_and(|h| h.gfx_write(((off - 0x300) / 8) as usize, value))
            {
                self.note(off, true);
            }
            return;
        }
        let now = (self.clock)();
        let mut s = self.state.lock();
        s.sync(now);
        let mut alias = None;
        let mut compare = None;
        let mut enable = None;
        let mut nmi = None;
        match off {
            reg::NODE_REGION | reg::OFFSET_0010 | reg::OFFSET_0018 | reg::OFFSET_00A8 | reg::OFFSET_00B0 | reg::OFFSET_00B8 | reg::OFFSET_00C0 | reg::OFFSET_00E8 | reg::OFFSET_04C8 | reg::OFFSET_04D0 => {
                s.w[off as usize / 8] = value
            }
            reg::PROTECTION_KEY => {
                if value == IIO_PROTECT_OVRRD_KEY {
                    s.w[0] |= 1 << self.cfg.request_region;
                }
            }
            reg::CACHED_ALIAS => {
                if s.w[5] != value & 15 {
                    s.w[5] = value & 15;
                    alias = Some((value & 15) as u8);
                }
            }
            reg::OFFSET_0030 => s.w[6] = value & 255,
            reg::OFFSET_0038 => s.w[7] = value & 0xffffff,
            reg::CPU_A_ENABLE | reg::CPU_B_ENABLE => {
                let slot = ((off - 0x50) / 8) as usize;
                s.w[off as usize / 8] = value & 1;
                enable = Some((slot, value & 1 != 0));
            }
            reg::NMI_SLOT_A | reg::NMI_SLOT_B => {
                if value & 1 != 0 {
                    nmi = Some(((off - 0x70) / 8) as usize);
                } else {
                    self.note(off, true);
                }
            }
            reg::PENDING_MODIFY | reg::PENDING_MODIFY_B => {
                let now = s.time(now);
                s.modify(value, true, now);
            }
            reg::REQUEST_A | reg::REQUEST_B => s.w[off as usize / 8] |= 1 << self.cfg.request_region,
            reg::PENDING_A_CLEAR | reg::PENDING_B_CLEAR => s.w[(off - reg::OFFSET_0010) as usize / 8] &= !value,
            reg::OFFSET_00F0 => {
                s.w[0xf0 / 8] = value & 31;
                if value & 31 != 0 {
                    self.note(off, true);
                }
            }
            reg::RT_COMPARE_A | reg::RT_COMPARE_B => {
                let slot = ((off - 0x108) / 8) as usize;
                s.w[off as usize / 8] = value & RT_MASK;
                s.arm_rt(slot, now);
                compare = Some(slot);
            }
            reg::PROFILE_COMPARE => {
                s.w[0x118 / 8] = value & 0xffffffff;
                s.arm_profile(now);
            }
            reg::RT_STATUS_A | reg::RT_STATUS_B | reg::PROFILE_STATUS_A | reg::PROFILE_STATUS_B => s.w[off as usize / 8] = 0,
            reg::RT_ENABLE_A | reg::RT_ENABLE_B | reg::PROFILE_ENABLE_A | reg::PROFILE_ENABLE_B => s.w[off as usize / 8] = value & 1,
            reg::ERROR_SUMMARY_ALIAS => {
                s.w[0x400 / 8] =
                    (s.w[0x400 / 8] & MEMORY_ERROR) | (value & ERROR_MASK & !MEMORY_ERROR)
            }
            reg::ERROR_SUMMARY => s.w[reg::ERROR_SUMMARY as usize / 8] &= !(value & ERROR_MASK),
            reg::ERROR_CAUSE_A | reg::ERROR_CAUSE_B => s.w[off as usize / 8] = value & ERROR_MASK,
            reg::ERROR_ADDRESS_A | reg::ERROR_ADDRESS_B => s.w[off as usize / 8] = value & 0x1fffffff8,
            reg::ERROR_CONTROL => {
                s.w[0x428 / 8] = value & 15;
                if value & 15 != 0 {
                    self.note(off, true);
                }
            }
            reg::ERROR_CAPTURE_A | reg::ERROR_CAPTURE_B | reg::ERROR_CAPTURE_C | reg::ERROR_CAPTURE_D => {
                let slot = usize::from(off >= 0x450);
                s.w[0x430 / 8 + slot * 4] = 0;
                s.w[0x440 / 8 + slot * 4] = 0;
            }
            reg::ERROR_SOURCE_A | reg::ERROR_SOURCE_B => s.w[off as usize / 8] = value & 63,
            reg::ERROR_DETAIL_A | reg::ERROR_DETAIL_B => s.w[off as usize / 8] = value & 0x1fffff,
            reg::ERROR_DETAIL_C => s.w[off as usize / 8] = value & 0xfffff,
            reg::CREDIT_MODE => s.credit_mode = value & 1,
            reg::COUNTER => {
                s.base = value & RT_MASK;
                s.epoch = s.time(now);
                s.arm_all(now);
            }
            _ => self.note(off, true),
        }
        drop(s);
        if let Some(h) = self.hooks() {
            if let Some(size) = alias {
                h.cached_alias_changed(size);
            }
            if let Some(slot) = compare {
                h.rt_compare_written(self.cfg.interface, slot);
            }
            if let Some((slot, value)) = enable {
                h.cpu_enable(self.cfg.interface, slot, value);
            }
            if let Some(slot) = nmi {
                h.nmi(self.cfg.interface, slot);
            }
        }
        self.drive();
        self.schedule();
    }
}
fn readable(off: u64) -> bool {
    matches!(
        off,
        reg::NODE_REGION | reg::OFFSET_0010
            | reg::OFFSET_0018
            | reg::CACHED_ALIAS
            | reg::OFFSET_0030
            | reg::OFFSET_0038
            | reg::CPU_A_ENABLE
            | reg::CPU_B_ENABLE
            | reg::OFFSET_00A8
            | reg::OFFSET_00B0
            | reg::OFFSET_00B8
            | reg::OFFSET_00C0
            | reg::REQUEST_A
            | reg::REQUEST_B
            | reg::OFFSET_00E8
            | reg::OFFSET_00F0
            | reg::RT_COMPARE_A
            | reg::RT_COMPARE_B
            | reg::PROFILE_COMPARE
            | reg::RT_STATUS_A
            | reg::RT_STATUS_B
            | reg::PROFILE_STATUS_A
            | reg::PROFILE_STATUS_B
            | reg::RT_ENABLE_A
            | reg::RT_ENABLE_B
            | reg::PROFILE_ENABLE_A
            | reg::PROFILE_ENABLE_B
            | reg::ERROR_SUMMARY
            | reg::ERROR_CAUSE_A
            | reg::ERROR_CAUSE_B
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
impl BusDevice for BedrockPi {
    fn read64(&self, addr: u64) -> BusRead64 {
        if addr < SIZE && addr & 7 == 0 {
            BusRead64::ok(self.read(addr))
        } else {
            self.note(addr, false);
            BusRead64::err()
        }
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        if addr == reg::COUNTER || addr == reg::COUNTER + 4 {
            let value = self.read(reg::COUNTER);
            BusRead32::ok(if addr & 4 == 0 {
                (value >> 32) as u32
            } else {
                value as u32
            })
        } else {
            self.note(addr, false);
            BusRead32::err()
        }
    }
    fn write64(&self, addr: u64, value: u64) -> u32 {
        if addr < SIZE && addr & 7 == 0 {
            self.write(addr, value);
            BUS_OK
        } else {
            self.note(addr, true);
            BUS_ERR
        }
    }
}
impl Device for BedrockPi {
    fn start(&self) {
        let now = (self.clock)();
        let mut s = self.state.lock();
        if s.running {
            return;
        }
        if let Some(paused) = s.paused.take() {
            let delta = now.saturating_sub(paused);
            s.epoch = s.epoch.saturating_add(delta);
            for due in &mut s.rt_due {
                *due += u128::from(delta);
            }
            s.profile_due += u128::from(delta);
            for due in &mut s.rearm {
                if *due != 0 {
                    *due = due.saturating_add(delta);
                }
            }
        }
        s.running = true;
        drop(s);
        self.schedule();
    }
    fn stop(&self) {
        {
            let _pass = self.timer_pass.lock();
            let mut s = self.state.lock();
            s.sync((self.clock)());
            s.paused = Some(s.time((self.clock)()));
            s.running = false;
            if let Some(id) = self.timer.lock().take() {
                if let Some(tm) = self.timers.get() {
                    tm.remove(id);
                }
            }
        }
        self.drive();
    }
    fn is_running(&self) -> bool {
        self.state.lock().running
    }
    fn step(&self, _cycles: u64) {
        self.poll();
    }
    fn get_clock(&self) -> u64 {
        self.state.lock().count((self.clock)())
    }
}
impl Resettable for BedrockPi {
    fn power_on(&self) {
        let _dispatch = self.dispatch.lock();
        let now = (self.clock)();
        let mut s = self.state.lock();
        // These controls and captures survive the PI block's local reset.
        let old = s.w;
        let running = s.running;
        let reflected = s.reflected;
        let uart = s.uart;
        *s = State::new(now);
        for off in [
            0, 0x10, 0x18, 0x28, 0x50, 0x58, 0x400, 0x418, 0x420, 0x428, 0x430, 0x440, 0x450,
            0x460, 0x4c8, 0x4d0,
        ] {
            s.w[off / 8] = old[off / 8];
        }
        s.running = running;
        s.paused = if running { None } else { Some(now) };
        s.reflected = reflected;
        s.uart = uart;
        drop(s);
        if let Some(h) = self.hooks() {
            h.gfx_reset();
            for slot in 0..2 {
                let enabled = self.state.lock().w[0x50 / 8 + slot] != 0;
                h.cpu_enable(self.cfg.interface, slot, enabled);
            }
        }
        self.drive();
        self.schedule();
    }
}
#[cfg(test)]
mod tests;
