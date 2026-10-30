//! Bedrock IO routes programmed PIO windows to PCI widgets, applies inbound
//! permission checks to DMA requests and exposes two block-transfer engines.
//! Guests configure the widget masks before touching devices; the BTE moves
//! 128-byte lines and writes completion state through the configured target.
//!
//! # What is approximated here
//!
//! BTE transfers and their completion callbacks run synchronously when a
//! start is written. Timing and unpublished register behavior are not modeled.
//!
//! The plain register words sit in a sparse map keyed by offset, since the
//! 0x30010-byte block holds only a few dozen of them.
//!
//! Ported from hw/misc/sgi-bedrock-io.c in jamesbraid/qemu-sgi-origami.

/// Length of the block-relative register aperture.
pub const SIZE: u64 = 0x30010;

/// The public Linux Hub I/O protection-override key literal.
///
/// Linux `arch/mips/include/asm/sn/sn0/hubio.h` at
/// `db2ddb87143519e20a95aa36c60b36107b736a58` defines
/// `IIO_PROTECT_OVRRD_KEY` as this value (lines 31, 47). This establishes
/// the literal's provenance only; it does not establish a register mapping.
pub const IIO_PROTECT_OVRRD_KEY: u64 = 0x53474972756c6573;

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

use crate::traits::{BusDevice, BusRead64, Resettable, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;
use std::thread::ThreadId;
const ID: u64 = (2 << 28) | (0xc110 << 12) | (0x36 << 1);
const ADDR_MASK: u64 = 0xffffffff80;
const BUSY: u64 = 1 << 20;
const ERROR: u64 = 1 << 16;
#[derive(Clone, Copy, Default)]
struct Bte {
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
    node: u8,
    peer: bool,
}
impl Default for Regs {
    fn default() -> Self {
        let mut words = BTreeMap::new();
        for (a, v) in [
            (0x20, 0xa),
            (0x100, !0),
            (0x110, !0),
            (0x118, !0),
            (0x120, !0),
            (0x128, 0x3fc0fff6403 & !(1 << 13)),
            (0x1e8, 0x111),
        ] {
            words.insert(a, v);
        }
        Self {
            words,
            bte: [Bte::default(); 2],
            node: 0,
            peer: false,
        }
    }
}
// Only a synchronous callback on the same thread is recursive. DMA workers
// on other threads may call the thread-safe downstream bus independently.
struct InboundAccess<'a> {
    active: &'a Mutex<HashSet<ThreadId>>,
    thread: ThreadId,
}
impl Drop for InboundAccess<'_> {
    fn drop(&mut self) {
        self.active.lock().remove(&self.thread);
    }
}

pub struct BedrockIo {
    inbound_threads: Mutex<HashSet<ThreadId>>,
    regs: Arc<Mutex<Regs>>,
    widgets: Mutex<[Option<Arc<dyn BusDevice>>; 16]>,
    memory: Mutex<Option<Arc<dyn BusDevice>>>,
    hub: Mutex<Option<Arc<dyn BusDevice>>>,
}
impl Default for BedrockIo {
    fn default() -> Self {
        Self::new()
    }
}
impl BedrockIo {
    pub fn new() -> Self {
        let regs = Arc::new(Mutex::new(Regs::default()));
        crate::locks::register_mutex("bedrock-io", &regs);
        Self {
            regs,
            inbound_threads: Mutex::new(HashSet::new()),
            widgets: Mutex::new(std::array::from_fn(|_| None)),
            memory: Mutex::new(None),
            hub: Mutex::new(None),
        }
    }
    pub fn connect_widget(&self, widget: u8, dev: Arc<dyn BusDevice>) {
        assert!(widget < 16);
        self.widgets.lock()[widget as usize] = Some(dev);
        self.set_xio_peer(true);
    }
    pub fn set_xio_peer(&self, peer: bool) {
        let mut r = self.regs.lock();
        r.peer = peer;
        if peer {
            *r.words.entry(reg::XIO_STATUS).or_default() |= 1 << 13;
        } else {
            *r.words.entry(reg::XIO_STATUS).or_default() &= !(1 << 13);
        }
    }
    pub fn connect_memory(&self, dev: Arc<dyn BusDevice>) {
        *self.memory.lock() = Some(dev);
    }
    pub fn connect_hub(&self, dev: Arc<dyn BusDevice>) {
        *self.hub.lock() = Some(dev);
    }
    pub fn set_node_number(&self, node: u8) {
        self.regs.lock().node = node;
    }
    pub fn gfx_map(&self, entry: usize) -> u64 {
        assert!(entry < 2);
        self.word(reg::GFX_MAP_BASE + entry as u64 * 8)
    }
    fn word(&self, a: u64) -> u64 {
        self.regs.lock().words.get(&a).copied().unwrap_or(0)
    }
    pub fn disable_device(&self, widget: u8, device: u8) {
        if (8..16).contains(&widget) && device < 8 {
            *self.regs.lock().words.entry(reg::INBOUND_DEVICE_PERMISSION).or_default() &=
                !(1 << ((widget - 8) * 8 + device));
        }
    }
    fn permitted(r: &Regs, widget: u8, device: u8) -> bool {
        widget < 16
            && device < 8
            && r.words.get(&reg::WIDGET_PERMISSION).copied().unwrap_or(0) & (1 << widget) != 0
            && (widget < 8
                || r.words.get(&reg::INBOUND_DEVICE_PERMISSION).copied().unwrap_or(0) & (1 << ((widget - 8) * 8 + device))
                    != 0)
    }
    fn retained(a: u64) -> bool {
        matches!(
            a,
            0x208
                | 0x210
                | 0x218
                | 0x220
                | 0x228
                | 0x300
                | reg::GFX_MAP_BASE
                | 0x148
                | 0x388
                | 0x390
                | 0x398
                | 0x3a0
                | 0x600
                | 0x608
                | 0x610
                | 0x30000
                | 0x30008
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
    fn bte_at(a: u64) -> Option<(usize, u64)> {
        if (reg::BTE1_BASE..reg::BTE1_BASE + reg::BTE_REGISTER_SIZE).contains(&a) {
            Some((0, a - reg::BTE1_BASE))
        } else if (reg::BTE2_BASE..reg::BTE2_BASE + reg::BTE_REGISTER_SIZE).contains(&a) {
            Some((1, a - reg::BTE2_BASE))
        } else {
            None
        }
    }
    fn target(&self, offset: u64) -> Option<(u8, u64)> {
        if offset < 0x10000000 {
            Some(((offset >> 24) as u8, offset & 0xffffff))
        } else if offset >= 0x20000000 {
            let entry = (offset >> 29) - 1;
            if entry >= 7 {
                return None;
            }
            let itte = self.word(reg::ITTE_BASE + entry * 8);
            Some((
                ((itte >> 8) & 15) as u8,
                ((itte & 31) << 29)
                    | (offset & 0x1fffffff)
                    | if itte & (1 << 12) != 0 { 1 << 47 } else { 0 },
            ))
        } else {
            None
        }
    }
    pub fn pio_read(&self, offset: u64, size: u8) -> BusRead64 {
        self.pio(offset, size, None)
    }
    pub fn pio_write(&self, offset: u64, size: u8, value: u64) -> u32 {
        self.pio(offset, size, Some(value)).status
    }
    fn pio(&self, offset: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let Some((widget, target)) = self.target(offset) else {
            return BusRead64::err();
        };
        {
            let r = self.regs.lock();
            if r.words.get(&0x100).copied().unwrap_or(0) & 1 == 0
                || r.words.get(&0x110).copied().unwrap_or(0) & (1 << widget) == 0
            {
                return BusRead64::err();
            }
        }
        if !matches!(size, 1 | 2 | 4 | 8) {
            return BusRead64::err();
        }
        if widget == 1 && target == 0 {
            return BusRead64::ok(ID);
        }
        let dev = self.widgets.lock()[widget as usize].clone();
        Self::access(dev, target, size, value)
    }
    /// Device None means every possible DMA master must be permitted.
    pub fn inbound_read(
        &self,
        widget: u8,
        device: Option<u8>,
        address: u64,
        size: u8,
    ) -> BusRead64 {
        self.inbound(widget, device, address, size, None)
    }
    pub fn inbound_write(
        &self,
        widget: u8,
        device: Option<u8>,
        address: u64,
        size: u8,
        value: u64,
    ) -> u32 {
        self.inbound(widget, device, address, size, Some(value))
            .status
    }
    fn inbound(
        &self,
        widget: u8,
        device: Option<u8>,
        address: u64,
        size: u8,
        value: Option<u64>,
    ) -> BusRead64 {
        if address >= (1 << 48) || size as u64 > (1 << 48) - address {
            return BusRead64::err();
        }
        {
            let r = self.regs.lock();
            let allowed = if let Some(device) = device {
                Self::permitted(&r, widget, device)
            } else {
                (0..8).all(|d| Self::permitted(&r, widget, d))
            };
            if !allowed {
                crate::dlog!(crate::devlog::LogModule::Bedrock,
                    "Bedrock IO inbound access refused from widget {widget} device {device:?}"
                );
                return BusRead64::err();
            }
        }
        let thread = std::thread::current().id();
        if !self.inbound_threads.lock().insert(thread) {
            crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO reentrant inbound packet refused");
            return BusRead64::err();
        }
        let _access = InboundAccess {
            active: &self.inbound_threads,
            thread,
        };
        if address & (1 << 47) != 0 {
            let node = ((address >> 39) & 255) as u8;
            let window = address & ((1 << 39) - 1);
            if node != self.regs.lock().node || window & !0x7fffff != 0x1800000 {
                return BusRead64::err();
            }
            let dev = self.hub.lock().clone();
            Self::access(dev, window & 0x7fffff, size, value)
        } else {
            let dev = self.memory.lock().clone();
            Self::access(dev, address, size, value)
        }
    }
    fn access(dev: Option<Arc<dyn BusDevice>>, a: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let Some(dev) = dev else {
            return BusRead64::err();
        };
        if let Some(v) = value {
            let status = match size {
                1 => dev.write8(a, v as u8),
                2 => dev.write16(a, v as u16),
                4 => dev.write32(a, v as u32),
                8 => dev.write64(a, v),
                _ => BUS_ERR,
            };
            BusRead64 { status, data: 0 }
        } else {
            match size {
                1 => {
                    let r = dev.read8(a);
                    BusRead64 {
                        status: r.status,
                        data: r.data as u64,
                    }
                }
                2 => {
                    let r = dev.read16(a);
                    BusRead64 {
                        status: r.status,
                        data: r.data as u64,
                    }
                }
                4 => {
                    let r = dev.read32(a);
                    BusRead64 {
                        status: r.status,
                        data: r.data as u64,
                    }
                }
                8 => dev.read64(a),
                _ => BusRead64::err(),
            }
        }
    }
    fn run_bte(&self, n: usize, mut b: Bte) {
        let memory = self.memory.lock().clone();
        let count = b.status & 0xffff;
        let bytes = count * 128;
        let overflow = (b.control & 1 == 0 && b.source > (1 << 40) - bytes)
            || b.destination > (1 << 40) - bytes;
        let mut remaining = count;
        if overflow {
            b.status = remaining | ERROR;
        } else {
            while remaining != 0 {
                let mut line = [0u64; 16];
                let mut status = BUS_OK;
                if b.control & 1 == 0 {
                    for (i, v) in line.iter_mut().enumerate() {
                        let r = Self::access(memory.clone(), b.source + i as u64 * 8, 8, None);
                        status = r.status;
                        if status != BUS_OK {
                            break;
                        }
                        *v = r.data;
                    }
                }
                if status == BUS_OK {
                    for (i, v) in line.into_iter().enumerate() {
                        status =
                            Self::access(memory.clone(), b.destination + i as u64 * 8, 8, Some(v))
                                .status;
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
            if Self::access(memory, b.notification, 8, Some(b.status)).status != BUS_OK {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE notification write failed");
            }
        } else {
            let node = ((b.interrupt >> 1) & 255) as u8;
            if node != self.regs.lock().node {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE interrupt reaches no node");
                return;
            }
            let hub = self.hub.lock().clone();
            let offset = if b.interrupt & 1 != 0 { 0x200090 } else { 0x90 };
            if Self::access(hub, offset, 8, Some(0x100 | ((b.interrupt >> 16) & 127))).status
                != BUS_OK
            {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE completion interrupt failed");
            }
        }
    }
}
impl Resettable for BedrockIo {
    fn power_on(&self) {
        let mut r = self.regs.lock();
        let peer = r.peer;
        let node = r.node;
        *r = Regs::default();
        r.node = node;
        r.peer = peer;
        if peer {
            *r.words.entry(reg::XIO_STATUS).or_default() |= 1 << 13;
        }
    }
}
impl BusDevice for BedrockIo {
    fn read64(&self, a: u64) -> BusRead64 {
        if a & 7 != 0 {
            return BusRead64::err();
        }
        let mut r = self.regs.lock();
        if let Some((n, a)) = Self::bte_at(a) {
            let b = &mut r.bte[n];
            return BusRead64::ok(match a {
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
            });
        }
        let v = match a {
            reg::DEVICE_ID => ID,
            reg::READ_ONLY | 0x1f8 => 0,
            reg::LINK_STATUS => r.words.get(&a).copied().unwrap_or(0) & !(1 << 28),
            reg::LINK_LOG => r.words.get(&a).copied().unwrap_or(0),
            _ if Self::mask(a).is_some() => r.words.get(&a).copied().unwrap_or(0),
            _ => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO unsupported read {a:#x}");
                0
            }
        };
        BusRead64::ok(v)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        if a & 7 != 0 {
            return BUS_ERR;
        }
        let mut r = self.regs.lock();
        if let Some((n, offset)) = Self::bte_at(a) {
            let b = &mut r.bte[n];
            let mut run = false;
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
                            run = true;
                        }
                    }
                }
                bte_reg::NOTIFICATION => b.notification = v & ADDR_MASK,
                bte_reg::INTERRUPT => b.interrupt = v & 0x7f01ff,
                _ => {}
            }
            let b = *b;
            drop(r);
            if run {
                self.run_bte(n, b);
            }
            return BUS_OK;
        }
        match a {
            reg::PROTECTION_KEY => {
                if v == IIO_PROTECT_OVRRD_KEY {
                    *r.words.entry(0x100).or_default() |= 1;
                }
            }
            reg::INBOUND_DEVICE_PERMISSION | reg::REGISTER_01E8 => {
                *r.words.entry(a).or_default() |= v & Self::mask(a).unwrap();
            }
            reg::XIO_STATUS => {
                let v = v & Self::mask(a).unwrap() & if r.peer { !0 } else { !(1 << 13) };
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
        BUS_OK
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-io-test.c in
    // jamesbraid/qemu-sgi-origami: inbound-protection, bte and
    // bte-error-termination.
    use super::*;
    #[derive(Default)]
    struct Memory {
        words: Mutex<BTreeMap<u64, u64>>,
        fail_read: Mutex<Option<u64>>,
        fail_write: Mutex<Option<u64>>,
    }
    impl BusDevice for Memory {
        fn read64(&self, a: u64) -> BusRead64 {
            if *self.fail_read.lock() == Some(a) {
                return BusRead64::err();
            }
            BusRead64::ok(self.words.lock().get(&a).copied().unwrap_or(0))
        }
        fn write64(&self, a: u64, v: u64) -> u32 {
            if *self.fail_write.lock() == Some(a) {
                return BUS_ERR;
            }
            self.words.lock().insert(a, v);
            BUS_OK
        }
    }
    fn launch(d: &BedrockIo, base: u64, count: u64, src: u64, dst: u64, notify: u64, control: u64) {
        d.write64(base, count | BUSY);
        d.write64(base + 8, src);
        d.write64(base + 0x10, dst);
        d.write64(base + 0x20, notify);
        d.write64(base + 0x18, control);
    }
    struct Reentrant(std::sync::Weak<BedrockIo>);
    impl BusDevice for Reentrant {
        fn read64(&self, a: u64) -> BusRead64 {
            self.0.upgrade().unwrap().inbound_read(15, Some(0), a, 8)
        }
    }
    struct Busy;
    impl BusDevice for Busy {
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::busy()
        }
        fn write64(&self, _: u64, _: u64) -> u32 {
            crate::traits::BUS_BUSY
        }
    }
    struct BlockingMemory {
        entered: std::sync::mpsc::Sender<()>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
        words: Mutex<BTreeMap<u64, u64>>,
    }
    impl BusDevice for BlockingMemory {
        fn write64(&self, address: u64, value: u64) -> u32 {
            if address == 0x1000 {
                self.entered.send(()).unwrap();
                self.release
                    .lock()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
            self.words.lock().insert(address, value);
            BUS_OK
        }
    }

    /// The guest must enable widget and requester masks before PIO or inbound DMA succeeds.
    #[test]
    fn the_io_block_checks_pio_and_inbound_requester_permissions() {
        {
            let d = BedrockIo::new();
            let m = Arc::new(Memory::default());
            d.connect_widget(8, m.clone());
            d.write64(reg::ITTE_BASE, 0x802);
            assert_eq!(d.pio_write(0x20000080, 8, 0xabcd), BUS_OK);
            assert_eq!(m.read64(0x40000080).data, 0xabcd);
            assert_eq!(d.pio_read(0x20000080, 8).data, 0xabcd);
            d.write64(reg::INBOUND_PERMISSION, 0);
            assert_eq!(d.pio_read(0x20000080, 8).status, BUS_ERR);
            d.write64(reg::INBOUND_PERMISSION, !0);
            d.write64(reg::PIO_PERMISSION, 0);
            assert_eq!(d.pio_read(0x20000080, 8).status, BUS_ERR);
            d.write64(reg::PROTECTION_KEY, IIO_PROTECT_OVRRD_KEY);
            assert_eq!(d.pio_read(0x20000080, 8).status, BUS_OK);
            assert_eq!(d.pio_read(0x10000000, 8).status, BUS_ERR);
            d.power_on();
            assert_eq!(d.read64(reg::PIO_PERMISSION).data, !0);
            assert_eq!(d.read64(reg::REGISTER_01E8).data, 0x111);

        }
        {
            let d = BedrockIo::new();
            let m = Arc::new(Memory::default());
            let h = Arc::new(Memory::default());
            d.connect_memory(m.clone());
            d.connect_hub(h.clone());
            assert_eq!(d.inbound_write(14, Some(2), 0x80, 8, 42), BUS_OK);
            assert_eq!(m.read64(0x80).data, 42);
            d.disable_device(14, 2);
            assert_eq!(d.inbound_read(14, Some(2), 0x80, 8).status, BUS_ERR);
            assert_eq!(d.inbound_read(14, None, 0x80, 8).status, BUS_ERR);
            assert_eq!(d.inbound_read(14, Some(1), 0x80, 8).status, BUS_OK);
            let a = (1 << 47) | 0x1800090;
            assert_eq!(d.inbound_write(15, Some(0), a, 8, 0x123), BUS_OK);
            assert_eq!(h.read64(0x90).data, 0x123);
            assert_eq!(
                d.inbound_read(15, Some(0), a | (1 << 39), 8).status,
                BUS_ERR
            );
            assert_eq!(d.inbound_read(15, Some(0), 1 << 48, 8).status, BUS_ERR);
            assert_eq!(d.inbound_read(15, Some(0), a + 0x800000, 8).status, BUS_ERR);
        }
    }
    /// The guest observes BTE line copies and zero-fill through its completion registers.
    #[test]
    fn the_bte_copies_and_zeroes_lines_before_completion() {
        let d = BedrockIo::new();
        let m = Arc::new(Memory::default());
        let h = Arc::new(Memory::default());
        d.connect_memory(m.clone());
        d.connect_hub(h.clone());
        for i in 0..32 {
            m.write64(0x1000 + i * 8, 0xfeed0000 | i);
        }
        launch(&d, reg::BTE1_BASE, 2, 0x1000, 0x2000, 0x3000, 0x10);
        for i in 0..32 {
            assert_eq!(m.read64(0x2000 + i * 8).data, 0xfeed0000 | i);
        }
        assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::STATUS).data, 0);
        assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::SOURCE).data, 0x1100);
        assert_eq!(m.read64(0x3000).data, 0);
        d.write64(reg::BTE2_BASE + bte_reg::INTERRUPT, 53 << 16);
        launch(&d, reg::BTE2_BASE, 1, 0, 0x2000, 0, 1);
        for i in 0..16 {
            assert_eq!(m.read64(0x2000 + i * 8).data, 0);
        }
        assert_eq!(h.read64(0x90).data, 0x135);
    }
    /// A failed BTE line stops the transfer and reports the failing source and destination.
    #[test]
    fn the_bte_reports_transfer_errors_and_stops_after_a_failed_line() {
        {
            for (read, fail) in [(true, 0x1080), (false, 0x2080)] {
                let d = BedrockIo::new();
                let m = Arc::new(Memory::default());
                d.connect_memory(m.clone());
                if read {
                    *m.fail_read.lock() = Some(fail);
                } else {
                    *m.fail_write.lock() = Some(fail);
                }
                launch(&d, reg::BTE1_BASE, 3, 0x1000, 0x2000, 0x3000, 0x10);
                assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::STATUS).data, 2 | ERROR);
                assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::SOURCE).data, 0x1080);
                assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::DESTINATION).data, 0x2080);
                assert_eq!(m.read64(0x3000).data, 2 | ERROR);
            }
            let d = BedrockIo::new();
            let m = Arc::new(Memory::default());
            d.connect_memory(m.clone());
            *m.fail_write.lock() = Some(0x3000);
            launch(&d, reg::BTE1_BASE, 1, 0x1000, 0x2000, 0x3000, 0x10);
            assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::STATUS).data, 0);
            launch(&d, reg::BTE1_BASE, 2, 0xffffffff80, 0x2000, 0x3000, 0x10);
            assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::STATUS).data, 2 | ERROR);
        }
        {
            let d = BedrockIo::new();
            launch(&d, reg::BTE1_BASE, 1, 0, 0, 0, 0x100);
            assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::STATUS).data, 1 | BUSY);
            d.read64(reg::BTE1_BASE + bte_reg::CONTROL);
            assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::STATUS).data, 1);
            launch(&d, reg::BTE1_BASE, 0, 0, 0, 0, 0x10);
            assert_eq!(d.read64(reg::BTE1_BASE + bte_reg::STATUS).data, 0);
        }
    }
    /// A downstream callback may reenter the IO block while an inbound transaction is active.
    #[test]
    fn reentrant_inbound_forwards_are_refused_and_can_be_retried() {
        let d = Arc::new(BedrockIo::new());
        d.connect_memory(Arc::new(Reentrant(Arc::downgrade(&d))));
        assert_eq!(d.inbound_read(15, Some(0), 0, 8).status, BUS_ERR);
        d.connect_memory(Arc::new(Busy));
        assert_eq!(
            d.inbound_read(15, Some(0), 0, 8).status,
            crate::traits::BUS_BUSY
        );
        d.connect_widget(8, Arc::new(Busy));
        assert_eq!(d.pio_read(0x8000000, 8).status, crate::traits::BUS_BUSY);
        assert_eq!(d.pio_write(0x8000000, 8, 42), crate::traits::BUS_BUSY);

    }
    /// The two guest BTEs can reach node memory without one worker holding the other off.
    #[test]
    fn independent_bte_workers_can_reach_memory_concurrently() {
        let d = Arc::new(BedrockIo::new());
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let memory = Arc::new(BlockingMemory {
            entered: entered_tx,
            release: Mutex::new(release_rx),
            words: Mutex::new(BTreeMap::new()),
        });
        d.connect_memory(memory.clone());
        let first_io = d.clone();
        let first =
            std::thread::spawn(move || first_io.inbound_write(14, Some(0), 0x1000, 8, 0x1234));
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let second_io = d.clone();
        let second =
            std::thread::spawn(move || second_io.inbound_write(15, Some(1), 0x2000, 8, 0x5678));
        let second_status = second.join().unwrap();
        release_tx.send(()).unwrap();
        let first_status = first.join().unwrap();
        assert_eq!(second_status, BUS_OK);
        assert_eq!(first_status, BUS_OK);
        assert_eq!(
            *memory.words.lock(),
            BTreeMap::from([(0x1000, 0x1234), (0x2000, 0x5678)])
        );
    }
}
