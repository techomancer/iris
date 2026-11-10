//! Bedrock's I/O block (the II) routes programmed PIO windows to crossbar
//! widgets, applies inbound permission checks to requests from widgets and
//! runs two block-transfer engines. Guests configure the widget masks before
//! touching devices; a BTE moves 128-byte lines and reports completion by a
//! memory write or an interrupt packet.
//!
//! The permission words and the translation entries every PIO and inbound
//! access checks are atomics, stored under the block's lock and read without
//! it.
//!
//! # What is approximated here
//!
//! BTE transfers and their completion run synchronously when a start is
//! written. Timing and unpublished register behavior are not modeled.
//!
//! The other register words sit in a sparse map keyed by offset, since the
//! 0x30010-byte block holds only a few dozen of them.
//!
//! Ported from hw/misc/sgi-bedrock-io.c in jamesbraid/qemu-sgi-origami.

use super::reg::IIO_PROTECT_OVRRD_KEY;
use super::Effect;
use crate::traits::{BusDevice, BusRead64, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = 0x30010;

pub mod reg {
    pub const DEVICE_ID: u64 = 0x0;
    pub const PROTECTION_KEY: u64 = 0x108;
    pub const PIO_PERMISSION: u64 = 0x100;
    pub const INBOUND_PERMISSION: u64 = 0x110;
    pub const INBOUND_DEVICE_PERMISSION: u64 = 0x120;
    pub const XIO_STATUS: u64 = 0x128;
    pub const LINK_LOG: u64 = 0x130;
    pub const LINK_STATUS: u64 = 0x138;
    pub const BTE1_BASE: u64 = 0x10000;
    pub const BTE_REGISTER_SIZE: u64 = 0x30;
    pub const READ_ONLY: u64 = 0x8;
    pub const BTE2_BASE: u64 = 0x20000;
    pub const WIDGET_STATUS: u64 = 0x20;
    pub const WIDGET_PERMISSION: u64 = 0x118;
    pub const REGISTER_0150: u64 = 0x150;
    pub const REGISTER_0158: u64 = 0x158;
    pub const ITTE_BASE: u64 = 0x160;
    pub const ITTE_LAST: u64 = 0x190;
    pub const REGISTER_0198_BASE: u64 = 0x198;
    pub const REGISTER_01D8_LAST: u64 = 0x1d8;
    pub const READ_ONLY_ALIAS: u64 = 0x1f8;
    pub const REGISTER_01E0: u64 = 0x1e0;
    pub const REGISTER_01E8: u64 = 0x1e8;
    pub const REGISTER_01F0: u64 = 0x1f0;
    pub const REGISTER_0200: u64 = 0x200;
    pub const REGISTER_03A8: u64 = 0x3a8;
    pub const REGISTER_03B0: u64 = 0x3b0;
    pub const REGISTER_03B8: u64 = 0x3b8;
    pub const REGISTER_03C0: u64 = 0x3c0;
    pub const RETAINED_WINDOW_BASE: u64 = 0x308;
    pub const RETAINED_WINDOW_LAST: u64 = 0x348;
    pub const RETAINED_CONFIG_BASE: u64 = 0x400;
    pub const RETAINED_CONFIG_LAST: u64 = 0x5d8;
    pub const GFX_MAP_BASE: u64 = 0x140;
}
pub mod bte_reg {
    pub const STATUS: u64 = 0x0;
    pub const SOURCE: u64 = 0x8;
    pub const DESTINATION: u64 = 0x10;
    pub const CONTROL: u64 = 0x18;
    pub const NOTIFICATION: u64 = 0x20;
    pub const INTERRUPT: u64 = 0x28;
}

const ID: u64 = (2 << 28) | (0xc110 << 12) | (0x36 << 1);
const ADDR_MASK: u64 = 0xffffffff80;
pub(super) const BUSY: u64 = 1 << 20;
pub(super) const ERROR: u64 = 1 << 16;
/// XIO status bit 13: a crossbar peer answers the link.
const PEER: u64 = 1 << 13;
const ITTES: usize = 7;

#[derive(Clone, Copy, Default)]
pub(super) struct Bte {
    status: u64,
    source: u64,
    destination: u64,
    control: u64,
    notification: u64,
    interrupt: u64,
}
struct Regs {
    words: BTreeMap<u64, u64>,
    bte: [Bte; 2],
}
impl Default for Regs {
    fn default() -> Self {
        let mut words = BTreeMap::new();
        for (a, v) in [(0x20, 0xa), (0x128, 0x3fc0fff6403 & !PEER), (0x1e8, 0x111)] {
            words.insert(a, v);
        }
        Self { words, bte: [Bte::default(); 2] }
    }
}

/// Where a finished BTE's interrupt packet goes: the PI it names, and the
/// pending-modify value.
pub(super) struct Completion {
    pub pi: usize,
    pub value: u64,
}

pub(super) struct Io {
    regs: Mutex<Regs>,
    /// PIO permission (0x100): bit 0 lets PIO through.
    pio_permission: AtomicU64,
    /// The widgets PIO may reach (0x110).
    pio_widgets: AtomicU64,
    /// The widgets whose requests are admitted (0x118).
    widget_permission: AtomicU64,
    /// Per-device admission for widgets 8..15, eight bits each (0x120).
    device_permission: AtomicU64,
    /// The large-window translation entries (0x160..0x190).
    itte: [AtomicU64; ITTES],
    /// Scratch register 0 (0x150), which the IP35 PROM uses to publish each
    /// interface's processors; an atomic so a board can read it without a
    /// lock.
    scratch0: AtomicU64,
    widgets: [OnceLock<Arc<dyn BusDevice>>; 16],
}

impl Io {
    pub(super) fn new() -> Self {
        let io = Self {
            regs: Mutex::new(Regs::default()),
            pio_permission: AtomicU64::new(0),
            pio_widgets: AtomicU64::new(0),
            widget_permission: AtomicU64::new(0),
            device_permission: AtomicU64::new(0),
            itte: Default::default(),
            scratch0: AtomicU64::new(0),
            widgets: Default::default(),
        };
        io.reset_permissions();
        io
    }
    fn reset_permissions(&self) {
        for word in [&self.pio_permission, &self.pio_widgets, &self.widget_permission, &self.device_permission] {
            word.store(!0, Relaxed);
        }
        for entry in &self.itte {
            entry.store(0, Relaxed);
        }
    }
    pub(super) fn locked(&self) -> bool {
        self.regs.is_locked()
    }
    /// Connect crossbar widget `widget`. Once per widget, before the machine runs.
    pub(super) fn connect_widget(&self, widget: u8, dev: Arc<dyn BusDevice>) {
        assert!(self.widgets[usize::from(widget)].set(dev).is_ok(), "Bedrock widget {widget} connected twice");
        *self.regs.lock().words.entry(reg::XIO_STATUS).or_default() |= PEER;
    }
    fn peer(&self) -> bool {
        self.widgets.iter().any(|w| w.get().is_some())
    }
    fn permitted(&self, widget: u8, device: u8) -> bool {
        widget < 16
            && device < 8
            && self.widget_permission.load(Relaxed) & (1 << widget) != 0
            && (widget < 8 || self.device_permission.load(Relaxed) & (1 << ((widget - 8) * 8 + device)) != 0)
    }
    /// Whether a request from `widget`'s device `device` is admitted;
    /// `None` asks for every master on the widget. Widget 0, the crossbar
    /// itself, is admitted by its widget bit alone.
    pub(super) fn admits(&self, widget: u8, device: Option<u8>) -> bool {
        let allowed = match device {
            Some(device) => self.permitted(widget, device),
            None => (0..8).all(|d| self.permitted(widget, d)),
        };
        if !allowed {
            crate::dlog!(crate::devlog::LogModule::Bedrock,
                "Bedrock IO inbound access refused from widget {widget} device {device:?}");
        }
        allowed
    }
    fn retained(a: u64) -> bool {
        matches!(
            a,
            0x208 | 0x210 | 0x218 | 0x220 | 0x228 | 0x300 | reg::GFX_MAP_BASE | 0x148
                | 0x388 | 0x390 | 0x398 | 0x3a0 | 0x600 | 0x608 | 0x610 | 0x30000 | 0x30008
        ) || (reg::RETAINED_WINDOW_BASE..reg::RETAINED_WINDOW_LAST).contains(&a)
            || (reg::RETAINED_CONFIG_BASE..=reg::RETAINED_CONFIG_LAST).contains(&a)
    }
    fn mask(a: u64) -> Option<u64> {
        Some(match a {
            reg::WIDGET_STATUS => 0x7fe01f,
            reg::PIO_PERMISSION | reg::INBOUND_PERMISSION | reg::WIDGET_PERMISSION | reg::INBOUND_DEVICE_PERMISSION | reg::REGISTER_0150 | reg::REGISTER_0158 => !0,
            reg::XIO_STATUS => 0x3ff0fff7d3f,
            reg::LINK_STATUS => 0x101ff7f,
            reg::ITTE_BASE..=reg::ITTE_LAST => 0x1f1f,
            reg::REGISTER_0198_BASE..=reg::REGISTER_01D8_LAST => !0,
            reg::REGISTER_01E0 => 0x3ffffff,
            reg::REGISTER_01E8 => 0x1111,
            reg::REGISTER_01F0 => 0x0fffffff03ffffff,
            reg::REGISTER_0200 => 0x10f,
            reg::REGISTER_03A8 => 0x13ffffffff30f3,
            reg::REGISTER_03B0 => 0x1ff0f,
            reg::REGISTER_03B8 => 0xff,
            reg::REGISTER_03C0 => 0xffffff,
            _ if Self::retained(a) => !0,
            _ => return None,
        })
    }
    /// The atomic that holds register `a`, for the words PIO and inbound
    /// accesses check.
    fn atomic(&self, a: u64) -> Option<&AtomicU64> {
        match a {
            reg::PIO_PERMISSION => Some(&self.pio_permission),
            reg::INBOUND_PERMISSION => Some(&self.pio_widgets),
            reg::WIDGET_PERMISSION => Some(&self.widget_permission),
            reg::INBOUND_DEVICE_PERMISSION => Some(&self.device_permission),
            reg::ITTE_BASE..=reg::ITTE_LAST if a % 8 == 0 => Some(&self.itte[((a - reg::ITTE_BASE) / 8) as usize]),
            reg::REGISTER_0150 => Some(&self.scratch0),
            _ => None,
        }
    }
    fn bte_at(a: u64) -> Option<(usize, u64)> {
        if (reg::BTE1_BASE..reg::BTE1_BASE + reg::BTE_REGISTER_SIZE).contains(&a) {
            Some((0, a - reg::BTE1_BASE))
        } else if (reg::BTE2_BASE..reg::BTE2_BASE + reg::BTE_REGISTER_SIZE).contains(&a) {
            Some((1, a - reg::BTE2_BASE))
        } else {
            None
        }
    }
    /// The widget and widget address a PIO offset below 4 GiB reaches: the
    /// small windows directly, the large ones through their translation entry.
    fn target(&self, offset: u64) -> Option<(u8, u64)> {
        if offset < 0x1000_0000 {
            Some(((offset >> 24) as u8, offset & 0xff_ffff))
        } else if offset >= 0x2000_0000 {
            let entry = ((offset >> 29) - 1) as usize;
            let itte = self.itte.get(entry)?.load(Relaxed);
            Some((
                ((itte >> 8) & 15) as u8,
                ((itte & 31) << 29) | (offset & 0x1fff_ffff) | if itte & (1 << 12) != 0 { 1 << 47 } else { 0 },
            ))
        } else {
            None
        }
    }
    /// A processor's uncached access to the I/O view below 4 GiB.
    pub(super) fn pio(&self, offset: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let Some((widget, target)) = self.target(offset) else {
            return BusRead64::err();
        };
        if self.pio_permission.load(Relaxed) & 1 == 0 || self.pio_widgets.load(Relaxed) & (1 << widget) == 0 {
            return BusRead64::err();
        }
        if !matches!(size, 1 | 2 | 4 | 8) {
            return BusRead64::err();
        }
        if widget == 1 && target == 0 {
            return BusRead64::ok(ID);
        }
        match self.widgets[usize::from(widget)].get() {
            Some(dev) => access(dev.as_ref(), target, size, value),
            None => BusRead64::err(),
        }
    }
    pub(super) fn read(&self, a: u64) -> u64 {
        if let Some(word) = self.atomic(a) {
            return word.load(Relaxed);
        }
        let mut r = self.regs.lock();
        if let Some((n, a)) = Self::bte_at(a) {
            let b = &mut r.bte[n];
            return match a {
                bte_reg::STATUS => b.status,
                bte_reg::SOURCE => b.source,
                bte_reg::DESTINATION => b.destination,
                bte_reg::CONTROL => {
                    b.status &= !BUSY;
                    b.control
                }
                bte_reg::NOTIFICATION => b.notification,
                bte_reg::INTERRUPT => b.interrupt,
                _ => 0,
            };
        }
        match a {
            reg::DEVICE_ID => ID,
            reg::READ_ONLY | reg::READ_ONLY_ALIAS => 0,
            reg::LINK_STATUS => r.words.get(&a).copied().unwrap_or(0) & !(1 << 28),
            reg::LINK_LOG => r.words.get(&a).copied().unwrap_or(0),
            _ if Self::mask(a).is_some() => r.words.get(&a).copied().unwrap_or(0),
            _ => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO unsupported read {a:#x}");
                0
            }
        }
    }
    /// A doubleword store at `a`: [`Effect::Bte`] when it starts an engine.
    pub(super) fn write(&self, a: u64, v: u64) -> Effect {
        let mut r = self.regs.lock();
        if let Some((n, offset)) = Self::bte_at(a) {
            let b = &mut r.bte[n];
            match offset {
                bte_reg::STATUS => b.status = v & (0xffff | BUSY),
                bte_reg::SOURCE => b.source = v & ADDR_MASK,
                bte_reg::DESTINATION => b.destination = v & ADDR_MASK,
                bte_reg::CONTROL => {
                    b.control = v & 0x111;
                    if b.control & 0x100 != 0 {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE poison unsupported");
                    } else if b.status & BUSY != 0 {
                        if b.status & 0xffff == 0 {
                            b.status = 0;
                            crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO zero-count BTE start");
                        } else {
                            return Effect::Bte(n);
                        }
                    }
                }
                bte_reg::NOTIFICATION => b.notification = v & ADDR_MASK,
                bte_reg::INTERRUPT => b.interrupt = v & 0x7f01ff,
                _ => {}
            }
            return Effect::None;
        }
        match a {
            reg::PROTECTION_KEY => {
                if v == IIO_PROTECT_OVRRD_KEY {
                    self.pio_permission.fetch_or(1, Relaxed);
                }
            }
            reg::INBOUND_DEVICE_PERMISSION => {
                self.device_permission.fetch_or(v, Relaxed);
            }
            _ if self.atomic(a).is_some() => self.atomic(a).unwrap().store(v & Self::mask(a).unwrap(), Relaxed),
            reg::REGISTER_01E8 => {
                *r.words.entry(a).or_default() |= v & Self::mask(a).unwrap();
            }
            reg::XIO_STATUS => {
                let v = v & Self::mask(a).unwrap() & if self.peer() { !0 } else { !PEER };
                r.words.insert(a, v);
            }
            reg::LINK_LOG => {
                if v == 0 {
                    r.words.insert(a, 0);
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO nonzero link-log write ignored");
                }
            }
            reg::READ_ONLY_ALIAS | reg::DEVICE_ID | reg::READ_ONLY => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO read-only or unsupported clear write {a:#x}");
            }
            _ => {
                if let Some(mask) = Self::mask(a) {
                    r.words.insert(a, v & mask);
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO unsupported write {a:#x} = {v:#x}");
                }
            }
        }
        Effect::None
    }
    /// Run engine `n` against `memory`, the node's DMA view, on a node
    /// numbered `node`. The completion interrupt, if the engine sends one to
    /// this node, is returned for the chip to deliver.
    pub(super) fn run_bte(&self, n: usize, memory: &dyn BusDevice, node: u8) -> Option<Completion> {
        let mut b = self.regs.lock().bte[n];
        let count = b.status & 0xffff;
        let bytes = count * 128;
        let overflow = (b.control & 1 == 0 && b.source > (1 << 40) - bytes) || b.destination > (1 << 40) - bytes;
        let mut remaining = count;
        if overflow {
            b.status = remaining | ERROR;
        } else {
            while remaining != 0 {
                let mut line = [0u64; 16];
                let mut status = BUS_OK;
                if b.control & 1 == 0 {
                    for (i, v) in line.iter_mut().enumerate() {
                        let r = memory.read64(b.source + i as u64 * 8);
                        status = r.status;
                        if status != BUS_OK {
                            break;
                        }
                        *v = r.data;
                    }
                }
                if status == BUS_OK {
                    for (i, v) in line.into_iter().enumerate() {
                        status = memory.write64(b.destination + i as u64 * 8, v);
                        if status != BUS_OK {
                            break;
                        }
                    }
                }
                if status != BUS_OK {
                    break;
                }
                remaining -= 1;
                b.source = (b.source + 128) & ADDR_MASK;
                b.destination = (b.destination + 128) & ADDR_MASK;
            }
            b.status = if remaining == 0 { 0 } else { remaining | ERROR };
        }
        self.regs.lock().bte[n] = b;
        if b.control & 0x10 != 0 {
            if memory.write64(b.notification, b.status) != BUS_OK {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE notification write failed");
            }
            return None;
        }
        if ((b.interrupt >> 1) & 255) as u8 != node {
            crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE interrupt reaches no node");
            return None;
        }
        Some(Completion { pi: (b.interrupt & 1) as usize, value: 0x100 | ((b.interrupt >> 16) & 127) })
    }
    pub(super) fn power_on(&self) {
        let mut r = self.regs.lock();
        *r = Regs::default();
        if self.peer() {
            *r.words.entry(reg::XIO_STATUS).or_default() |= PEER;
        }
        self.reset_permissions();
        self.scratch0.store(0, Relaxed);
    }
    /// Scratch register 0. One atomic load.
    pub(super) fn scratch0(&self) -> u64 {
        self.scratch0.load(Relaxed)
    }
}

/// An access of `size` bytes to `dev`: a write when `value` is given.
pub(super) fn access(dev: &dyn BusDevice, a: u64, size: u8, value: Option<u64>) -> BusRead64 {
    if let Some(v) = value {
        let status = match size {
            1 => dev.write8(a, v as u8),
            2 => dev.write16(a, v as u16),
            4 => dev.write32(a, v as u32),
            8 => dev.write64(a, v),
            _ => BUS_ERR,
        };
        return BusRead64 { status, data: 0 };
    }
    match size {
        1 => {
            let r = dev.read8(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        }
        2 => {
            let r = dev.read16(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        }
        4 => {
            let r = dev.read32(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        }
        8 => dev.read64(a),
        _ => BusRead64::err(),
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-io-test.c in
    // jamesbraid/qemu-sgi-origami: inbound-protection, bte and
    // bte-error-termination.
    use super::super::reg::{md, pi, IO_BASE, MD_BASE, PI1_BASE};
    use super::super::Rig;
    use super::*;
    use crate::traits::{BusRead64, BUS_BUSY};

    /// 64 MiB at bank 0, enabled.
    const BANK: u64 = 64 << 20;
    fn rig() -> Rig {
        let rig = Rig::new();
        rig.w(MD_BASE + md::CONFIG, 0x11);
        rig
    }
    #[derive(Default)]
    struct Memory(Mutex<BTreeMap<u64, u64>>);
    impl BusDevice for Memory {
        fn read64(&self, a: u64) -> BusRead64 {
            BusRead64::ok(self.0.lock().get(&a).copied().unwrap_or(0))
        }
        fn write64(&self, a: u64, v: u64) -> u32 {
            self.0.lock().insert(a, v);
            BUS_OK
        }
    }
    struct Busy;
    impl BusDevice for Busy {
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::busy()
        }
        fn write64(&self, _: u64, _: u64) -> u32 {
            BUS_BUSY
        }
    }
    fn launch(rig: &Rig, base: u64, count: u64, src: u64, dst: u64, notify: u64, control: u64) {
        rig.w(IO_BASE + base, count | BUSY);
        rig.w(IO_BASE + base + 8, src);
        rig.w(IO_BASE + base + 0x10, dst);
        rig.w(IO_BASE + base + 0x20, notify);
        rig.w(IO_BASE + base + 0x18, control);
    }

    /// The guest must enable widget and requester masks before PIO or inbound DMA succeeds.
    #[test]
    fn the_io_block_checks_pio_and_inbound_requester_permissions() {
        {
            let rig = rig();
            let pio = rig.chip.pio();
            let m = Arc::new(Memory::default());
            rig.chip.connect_widget(8, m.clone());
            rig.w(IO_BASE + reg::ITTE_BASE, 0x802);
            assert_eq!(pio.write64(0x20000080, 0xabcd), BUS_OK);
            assert_eq!(m.read64(0x40000080).data, 0xabcd);
            assert_eq!(pio.read64(0x20000080).data, 0xabcd);
            rig.w(IO_BASE + reg::INBOUND_PERMISSION, 0);
            assert_eq!(pio.read64(0x20000080).status, BUS_ERR);
            rig.w(IO_BASE + reg::INBOUND_PERMISSION, !0);
            rig.w(IO_BASE + reg::PIO_PERMISSION, 0);
            assert_eq!(pio.read64(0x20000080).status, BUS_ERR);
            rig.w(IO_BASE + reg::PROTECTION_KEY, IIO_PROTECT_OVRRD_KEY);
            assert_eq!(pio.read64(0x20000080).status, BUS_OK);
            assert_eq!(pio.read64(0x10000000).status, BUS_ERR);
            assert_ne!(rig.r(IO_BASE + reg::XIO_STATUS) & PEER, 0, "a widget answers the link");
            rig.chip.io.power_on();
            assert_eq!(rig.r(IO_BASE + reg::PIO_PERMISSION), !0);
            assert_eq!(rig.r(IO_BASE + reg::REGISTER_01E8), 0x111);
            assert_ne!(rig.r(IO_BASE + reg::XIO_STATUS) & PEER, 0, "the peer survives reset");
        }
        {
            let rig = rig();
            let chip = &rig.chip;
            assert_eq!(chip.inbound_write(14, Some(2), 0x80, 8, 42), BUS_OK);
            assert_eq!(chip.dma_view().read64(0x80).data, 42);
            chip.io.disable_device(14, 2);
            assert_eq!(chip.inbound_read(14, Some(2), 0x80, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(14, None, 0x80, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(14, Some(1), 0x80, 8).status, BUS_OK);
            // An interrupt packet to PI0's pending-modify register sets bank 0 bit 35.
            let a = (1 << 47) | 0x1800090;
            assert_eq!(chip.inbound_write(15, Some(0), a, 8, 0x123), BUS_OK);
            assert_eq!(rig.r(pi::PENDING_STATUS_A), 1 << 35);
            assert_eq!(chip.inbound_write(0, None, a + PI1_BASE, 8, 0x101), BUS_OK, "widget 0 is admitted");
            assert_eq!(rig.r(PI1_BASE + pi::PENDING_STATUS_A), 1 << 1);
            assert_eq!(chip.inbound_read(15, Some(0), a, 8).status, BUS_ERR, "only stores reach a PI");
            assert_eq!(chip.inbound_write(15, Some(0), a, 4, 0x123), BUS_ERR, "only doublewords");
            assert_eq!(chip.inbound_read(15, Some(0), a | (1 << 39), 8).status, BUS_ERR, "another node");
            assert_eq!(chip.inbound_read(15, Some(0), 1 << 48, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(15, Some(0), a + 0x800000, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(15, Some(0), BANK, 8).status, BUS_ERR, "beyond the bank");
        }
    }
    /// The guest observes BTE line copies and zero-fill through its completion registers.
    #[test]
    fn the_bte_copies_and_zeroes_lines_before_completion() {
        let rig = rig();
        let m = rig.chip.dma_view();
        for i in 0..32 {
            m.write64(0x1000 + i * 8, 0xfeed0000 | i);
        }
        m.write64(0x3000, 0xdead);
        launch(&rig, reg::BTE1_BASE, 2, 0x1000, 0x2000, 0x3000, 0x10);
        for i in 0..32 {
            assert_eq!(m.read64(0x2000 + i * 8).data, 0xfeed0000 | i);
        }
        assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::STATUS), 0);
        assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::SOURCE), 0x1100);
        assert_eq!(m.read64(0x3000).data, 0, "the notification holds the final status");
        rig.w(IO_BASE + reg::BTE2_BASE + bte_reg::INTERRUPT, 53 << 16);
        launch(&rig, reg::BTE2_BASE, 1, 0, 0x2000, 0, 1);
        for i in 0..16 {
            assert_eq!(m.read64(0x2000 + i * 8).data, 0);
        }
        assert_eq!(rig.r(pi::PENDING_STATUS_A), 1 << 53, "the completion interrupt");
    }
    /// A failed BTE line stops the transfer and reports the failing source and destination.
    #[test]
    fn the_bte_reports_transfer_errors_and_stops_after_a_failed_line() {
        // The second line runs off the end of the bank, on the read or the write side.
        for (src, dst) in [(BANK - 0x80, 0x2000), (0x1000, BANK - 0x80)] {
            let rig = rig();
            launch(&rig, reg::BTE1_BASE, 3, src, dst, 0x3000, 0x10);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::STATUS), 2 | ERROR);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::SOURCE), src + 0x80);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::DESTINATION), dst + 0x80);
            assert_eq!(rig.chip.dma_view().read64(0x3000).data, 2 | ERROR);
        }
        {
            let rig = rig();
            launch(&rig, reg::BTE1_BASE, 1, 0x1000, 0x2000, BANK, 0x10);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::STATUS), 0, "a failed notification still completes");
            launch(&rig, reg::BTE1_BASE, 2, 0xffffffff80, 0x2000, 0x3000, 0x10);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::STATUS), 2 | ERROR);
        }
        {
            let rig = rig();
            launch(&rig, reg::BTE1_BASE, 1, 0, 0, 0, 0x100);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::STATUS), 1 | BUSY);
            rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::CONTROL);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::STATUS), 1);
            launch(&rig, reg::BTE1_BASE, 0, 0, 0, 0, 0x10);
            assert_eq!(rig.r(IO_BASE + reg::BTE1_BASE + bte_reg::STATUS), 0);
        }
    }
    /// A widget that is busy answers busy through PIO, so the CPU retries.
    #[test]
    fn a_busy_widget_answers_busy_through_pio() {
        let rig = rig();
        rig.chip.connect_widget(8, Arc::new(Busy));
        let pio = rig.chip.pio();
        assert_eq!(pio.read64(0x8000000).status, BUS_BUSY);
        assert_eq!(pio.write64(0x8000000, 42), BUS_BUSY);
    }

    impl Io {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
        fn disable_device(&self, widget: u8, device: u8) {
            if (8..16).contains(&widget) && device < 8 {
                let _r = self.regs.lock();
                self.device_permission.fetch_and(!(1 << ((widget - 8) * 8 + device)), Relaxed);
            }
        }
    }
}
