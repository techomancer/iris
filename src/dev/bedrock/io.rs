//! Bedrock's I/O block (the II): routes PIO windows to crossbar widgets, checks inbound
//! permissions and runs two block-transfer engines.
//!
//! # What is approximated here
//!
//! BTE transfers and their completion run synchronously when a start is written. Timing and
//! unpublished register behavior are not modeled.
//!
//! Ported from hw/misc/sgi-bedrock-io.c in jamesbraid/qemu-sgi-origami.

use super::regs::{io as reg, IIO_PROTECT_OVRRD_KEY};
use super::Effect;
use crate::dev::widths;
use crate::traits::{BusDevice, BusRead64, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = reg::PERFORMANCE_PROFILING::OFFSET + reg::PERFORMANCE_PROFILING::WIDTH as u64 / 8;

/// The widget status word, which reads zero, and each engine's span.
const READ_ONLY: u64 = reg::WIDGET_STATUS::OFFSET;
const BTE_REGISTER_SIZE: u64 = reg::BTE_REGISTER_SIZE;
const SCRATCH_1: u64 = reg::SCRATCH::END - reg::SCRATCH::STRIDE;
const ITTE_LAST: u64 = reg::TRANSLATION_TABLE_ENTRY::END - reg::TRANSLATION_TABLE_ENTRY::STRIDE;
const PIO_REQUEST_LAST: u64 = reg::PIO_REQUEST_BUFFER_W8::END - reg::PIO_REQUEST_BUFFER_W8::STRIDE;
const CRB_LAST: u64 = reg::CRB_D::END - reg::CRB_D::STRIDE;
const RETAINED_WINDOW_END: u64 = reg::PIO_READ_ADDRESS::END;
const BTE_STRIDE: u64 = reg::BTE_LENGTH_STATUS::STRIDE;
const GRAPHICS_MAP_1: u64 = reg::GRAPHICS_MAP::at(1);

const ID: u64 = (2 << 28) | (0xc110 << 12) | (0x36 << 1);
const ADDR_MASK: u64 = reg::BTE_SOURCE::ADDRESS::MASK;
pub(super) const BUSY: u64 = reg::BTE_LENGTH_STATUS::BUSY::MASK;
pub(super) const ERROR: u64 = reg::BTE_LENGTH_STATUS::ERROR::MASK;
/// The top bit of the XIO link status: a crossbar peer answers the link.
const PEER: u64 = 1 << (reg::LINK_CONTROL::LINK_STATUS::SHIFT + reg::LINK_CONTROL::LINK_STATUS::LENGTH - 1);
const ITTES: usize = reg::TRANSLATION_TABLE_ENTRY::COUNT;

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
        for (a, v) in [
            (reg::WIDGET_CONTROL::OFFSET, 0xa),
            (reg::LINK_CONTROL::OFFSET, 0x3fc0fff6403 & !PEER),
            (reg::MISC_ERROR_MASK::OFFSET, reg::MISC_ERROR_MASK::RESET),
        ] {
            words.insert(a, v);
        }
        Self { words, bte: [Bte::default(); 2] }
    }
}

/// Where a finished BTE's interrupt packet goes: the node it names, its PI
/// and the pending-modify value.
pub(super) struct Completion {
    pub node: crate::sn_nodes::NodeId,
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
    /// Scratch 0 (0x150), where the IP35 PROM publishes each interface's processors; an atomic, read lock-free.
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
        for word in [&self.pio_permission, &self.pio_widgets, &self.widget_permission] {
            word.store(!0, Relaxed);
        }
        self.device_permission.store(reg::INBOUND_DEVICE_ERROR_MASK::RESET, Relaxed);
        for entry in &self.itte {
            entry.store(reg::TRANSLATION_TABLE_ENTRY::RESET, Relaxed);
        }
    }
    pub(super) fn locked(&self) -> bool {
        self.regs.is_locked()
    }
    /// Connect crossbar widget `widget`. Once per widget, before the machine runs.
    pub(super) fn connect_widget(&self, widget: u8, dev: Arc<dyn BusDevice>) {
        assert!(self.widgets[usize::from(widget)].set(dev).is_ok(), "Bedrock widget {widget} connected twice");
        *self.regs.lock().words.entry(reg::LINK_CONTROL::OFFSET).or_default() |= PEER;
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
    /// Whether `widget`'s device `device` is admitted; `None` asks for every master. Widget 0 by its bit alone.
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
            0x208 | 0x210 | 0x218 | 0x220 | 0x228 | 0x300 | reg::GRAPHICS_MAP::OFFSET | GRAPHICS_MAP_1
                | 0x388 | 0x390 | 0x398 | 0x3a0 | 0x600 | 0x608 | 0x610 | 0x30000 | 0x30008
        ) || (reg::PIO_READ_ADDRESS::OFFSET..RETAINED_WINDOW_END).contains(&a)
            || (reg::CRB_A::OFFSET..=CRB_LAST).contains(&a)
    }
    fn mask(a: u64) -> Option<u64> {
        Some(match a {
            reg::WIDGET_CONTROL::OFFSET => !(reg::WIDGET_CONTROL::RO | reg::WIDGET_CONTROL::RSVD),
            reg::LOCAL_ACCESS_PROTECTION::OFFSET | reg::OUTBOUND_WIDGET_ACCESS::OFFSET | reg::INBOUND_WIDGET_ACCESS::OFFSET | reg::INBOUND_DEVICE_ERROR_MASK::OFFSET | reg::SCRATCH::OFFSET | SCRATCH_1 => !0,
            reg::LINK_CONTROL::OFFSET => field_masks!(
                reg::LINK_CONTROL; MAX_BURST, DATA_AVAILABLE_SELECT, MAX_RETRY, REMOTE_POWER, LINK_STATUS, EIGHT_BIT_MODE,
                LINK_ENABLE, WARM_RESET, NULL_TIMEOUT
            ),
            // SENT and the two FORWARDED_PI bits are status, not stored.
            reg::INTERRUPT_DESTINATION::OFFSET => field_masks!(reg::INTERRUPT_DESTINATION; ENABLE, NODE, PI_SELECT, LEVEL),
            reg::TRANSLATION_TABLE_ENTRY::OFFSET..=ITTE_LAST => !(reg::TRANSLATION_TABLE_ENTRY::RO | reg::TRANSLATION_TABLE_ENTRY::RSVD),
            reg::PIO_REQUEST_BUFFER_W0::OFFSET..=PIO_REQUEST_LAST => !0,
            reg::CREDIT_COUNT_TIMEOUT::OFFSET => !(reg::CREDIT_COUNT_TIMEOUT::RO | reg::CREDIT_COUNT_TIMEOUT::RSVD),
            reg::MISC_ERROR_MASK::OFFSET => {
                field_masks!(reg::MISC_ERROR_MASK; CLEAR_PRECISE, BTE_1_ENABLE, BTE_0_ENABLE, WIDGET_0_ENABLE)
            }
            reg::CROSSTALK_TIMEOUT_THRESHOLD::OFFSET => {
                !(reg::CROSSTALK_TIMEOUT_THRESHOLD::RO | reg::CROSSTALK_TIMEOUT_THRESHOLD::RSVD)
            }
            reg::BTE_GLOBAL_CONTROL::OFFSET => !(reg::BTE_GLOBAL_CONTROL::RO | reg::BTE_GLOBAL_CONTROL::RSVD),
            reg::REQUEST_BUFFER_MANAGEMENT::OFFSET => field_masks!(
                reg::REQUEST_BUFFER_MANAGEMENT; PRECISE, MARKS, VALID_BITS, FLOW_CONTROL_COUNT, CLEAR_PENDING_BIT_13,
                CLEAR_PENDING_BIT_12, COUNT, READ_HEADER, SPURIOUS_MESSAGE
            ),
            reg::REQUEST_BUFFER_CONTROL::OFFSET => field_masks!(reg::REQUEST_BUFFER_CONTROL; PENDING, COMMAND, ENTRY),
            reg::CRB_TIMEOUT::OFFSET => !(reg::CRB_TIMEOUT::RO | reg::CRB_TIMEOUT::RSVD),
            reg::CRB_TIMEOUT_PRESCALER::OFFSET => !(reg::CRB_TIMEOUT_PRESCALER::RO | reg::CRB_TIMEOUT_PRESCALER::RSVD),
            _ if Self::retained(a) => !0,
            _ => return None,
        })
    }
    /// The atomic that holds register `a`, for the words PIO and inbound
    /// accesses check.
    fn atomic(&self, a: u64) -> Option<&AtomicU64> {
        match a {
            reg::LOCAL_ACCESS_PROTECTION::OFFSET => Some(&self.pio_permission),
            reg::OUTBOUND_WIDGET_ACCESS::OFFSET => Some(&self.pio_widgets),
            reg::INBOUND_WIDGET_ACCESS::OFFSET => Some(&self.widget_permission),
            reg::INBOUND_DEVICE_ERROR_MASK::OFFSET => Some(&self.device_permission),
            reg::TRANSLATION_TABLE_ENTRY::OFFSET..=ITTE_LAST if a % 8 == 0 => Some(&self.itte[((a - reg::TRANSLATION_TABLE_ENTRY::OFFSET) / 8) as usize]),
            reg::SCRATCH::OFFSET => Some(&self.scratch0),
            _ => None,
        }
    }
    /// The engine a BTE register belongs to, and the register's offset in engine 0.
    fn bte_at(a: u64) -> Option<(usize, u64)> {
        let n = a.checked_sub(reg::BTE_LENGTH_STATUS::OFFSET)? / BTE_STRIDE;
        (n < reg::BTE_LENGTH_STATUS::COUNT as u64 && (a - n * BTE_STRIDE) < reg::BTE_LENGTH_STATUS::OFFSET + BTE_REGISTER_SIZE)
            .then_some((n as usize, a - n * BTE_STRIDE))
    }
    /// The widget and widget address a PIO offset below 4 GiB reaches: the
    /// small windows directly, the large ones through their translation entry.
    fn target(&self, offset: u64) -> Option<(u8, u64)> {
        use crate::sn_nodes::{big_window, BIG_WINDOW_SHIFT, IO_BIG_WINDOWS, SMALL_WINDOW_SHIFT};
        if offset < 16 << SMALL_WINDOW_SHIFT {
            Some(((offset >> SMALL_WINDOW_SHIFT) as u8, offset & ((1 << SMALL_WINDOW_SHIFT) - 1)))
        } else if offset >= IO_BIG_WINDOWS {
            let itte = self.itte.get(((offset >> BIG_WINDOW_SHIFT) - 1) as usize)?.load(Relaxed);
            Some(big_window(itte, offset))
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
            Some(dev) => widths::access(dev.as_ref(), target, usize::from(size), value),
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
                reg::BTE_LENGTH_STATUS::OFFSET => b.status,
                reg::BTE_SOURCE::OFFSET => b.source,
                reg::BTE_DESTINATION::OFFSET => b.destination,
                reg::BTE_CONTROL::OFFSET => {
                    b.status &= !BUSY;
                    b.control
                }
                reg::BTE_NOTIFICATION_ADDRESS::OFFSET => b.notification,
                reg::BTE_INTERRUPT_ADDRESS::OFFSET => b.interrupt,
                _ => {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE{n} unsupported read {a:#x}");
                    0
                }
            };
        }
        match a {
            reg::WIDGET_ID::OFFSET => ID,
            READ_ONLY | reg::ERROR_CLEAR::OFFSET => 0,
            reg::INTERRUPT_DESTINATION::OFFSET => r.words.get(&a).copied().unwrap_or(0) & !reg::INTERRUPT_DESTINATION::SENT::MASK,
            reg::LLP_LOG::OFFSET => r.words.get(&a).copied().unwrap_or(0),
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
                reg::BTE_LENGTH_STATUS::OFFSET => b.status = v & (reg::BTE_LENGTH_STATUS::LENGTH_LINES::MASK | BUSY),
                reg::BTE_SOURCE::OFFSET => b.source = v & ADDR_MASK,
                reg::BTE_DESTINATION::OFFSET => b.destination = v & ADDR_MASK,
                reg::BTE_CONTROL::OFFSET => {
                    b.control = v & field_masks!(reg::BTE_CONTROL; POISON, NOTIFY, ZERO_FILL);
                    if b.control & reg::BTE_CONTROL::POISON::MASK != 0 {
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
                reg::BTE_NOTIFICATION_ADDRESS::OFFSET => b.notification = v & ADDR_MASK,
                reg::BTE_INTERRUPT_ADDRESS::OFFSET => b.interrupt = v & field_masks!(reg::BTE_INTERRUPT_ADDRESS; LEVEL, NODE, PI_SELECT),
                _ => crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE{n} unsupported write {a:#x} = {v:#x}"),
            }
            return Effect::None;
        }
        match a {
            reg::PROTECTION_OVERRIDE::OFFSET => {
                if v == IIO_PROTECT_OVRRD_KEY {
                    self.pio_permission.fetch_or(1, Relaxed);
                }
            }
            reg::INBOUND_DEVICE_ERROR_MASK::OFFSET => {
                self.device_permission.fetch_or(v, Relaxed);
            }
            _ if self.atomic(a).is_some() => self.atomic(a).unwrap().store(v & Self::mask(a).unwrap(), Relaxed),
            reg::MISC_ERROR_MASK::OFFSET => {
                *r.words.entry(a).or_default() |= v & Self::mask(a).unwrap();
            }
            reg::LINK_CONTROL::OFFSET => {
                let v = v & Self::mask(a).unwrap() & if self.peer() { !0 } else { !PEER };
                r.words.insert(a, v);
            }
            reg::LLP_LOG::OFFSET => {
                if v == 0 {
                    r.words.insert(a, 0);
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO nonzero link-log write ignored");
                }
            }
            reg::ERROR_CLEAR::OFFSET | reg::WIDGET_ID::OFFSET | READ_ONLY => {
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
    /// Run engine `n` against `memory`; the completion interrupt, if any, is returned for the chip to deliver.
    pub(super) fn run_bte(&self, n: usize, memory: &dyn BusDevice) -> Option<Completion> {
        let mut b = self.regs.lock().bte[n];
        let count = b.status & 0xffff;
        let bytes = count * 128;
        let overflow = (b.control & 1 == 0 && b.source > (1 << 40) - bytes) || b.destination > (1 << 40) - bytes;
        let mut remaining = count;
        if overflow {
            b.status = remaining | ERROR;
        } else {
            while remaining != 0 {
                // A line is read whole, then written whole, in order, so a
                // destination that overlaps the source behaves as it always did.
                let mut line = [0u64; 16];
                let mut status = BUS_OK;
                if b.control & 1 == 0 {
                    status = memory.read_block(b.source, &mut line);
                }
                if status == BUS_OK {
                    status = memory.write_block(b.destination, &line);
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
        {
            use reg::BTE_INTERRUPT_ADDRESS as ia;
            Some(Completion {
                node: crate::sn_nodes::node_id((b.interrupt & ia::NODE::MASK) >> ia::NODE::SHIFT, crate::sn_nodes::SN1_NODE_SHIFT),
                pi: (b.interrupt & ia::PI_SELECT::MASK) as usize,
                value: crate::sn_nodes::MODIFY_SET | ((b.interrupt & ia::LEVEL::MASK) >> ia::LEVEL::SHIFT),
            })
        }
    }
    pub(super) fn power_on(&self) {
        let mut r = self.regs.lock();
        *r = Regs::default();
        if self.peer() {
            *r.words.entry(reg::LINK_CONTROL::OFFSET).or_default() |= PEER;
        }
        self.reset_permissions();
        self.scratch0.store(0, Relaxed);
    }
    /// Scratch register 0. One atomic load.
    pub(super) fn scratch0(&self) -> u64 {
        self.scratch0.load(Relaxed)
    }
}

#[cfg(test)]
pub(in crate::dev::bedrock) mod tests {
    // Cases follow tests/qtest/sn1-bedrock-io-test.c in jamesbraid/qemu-sgi-origami:
    // inbound-protection, bte and bte-error-termination.
    use super::super::regs::{md, pi, PI1_BASE};
    use super::super::Rig;
    use super::*;
    use crate::traits::{BusRead64, BUS_BUSY};

    /// 64 MiB at bank 0, enabled.
    const BANK: u64 = 64 << 20;
    fn rig() -> Rig {
        let rig = Rig::new();
        rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
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
    pub(in crate::dev::bedrock) fn launch(rig: &Rig, base: u64, count: u64, src: u64, dst: u64, notify: u64, control: u64) {
        rig.w(reg::BASE + base, count | BUSY);
        rig.w(reg::BASE + base + 8, src);
        rig.w(reg::BASE + base + 0x10, dst);
        rig.w(reg::BASE + base + 0x20, notify);
        rig.w(reg::BASE + base + 0x18, control);
    }

    /// The guest must enable widget and requester masks before PIO or inbound DMA succeeds.
    #[test]
    fn the_io_block_checks_pio_and_inbound_requester_permissions() {
        {
            let rig = rig();
            let pio = rig.chip.pio();
            let m = Arc::new(Memory::default());
            rig.chip.connect_widget(8, m.clone());
            rig.w(reg::BASE + reg::TRANSLATION_TABLE_ENTRY::OFFSET, 0x802);
            assert_eq!(pio.write64(0x20000080, 0xabcd), BUS_OK);
            assert_eq!(m.read64(0x40000080).data, 0xabcd);
            assert_eq!(pio.read64(0x20000080).data, 0xabcd);
            rig.w(reg::BASE + reg::OUTBOUND_WIDGET_ACCESS::OFFSET, 0);
            assert_eq!(pio.read64(0x20000080).status, BUS_ERR);
            rig.w(reg::BASE + reg::OUTBOUND_WIDGET_ACCESS::OFFSET, !0);
            rig.w(reg::BASE + reg::LOCAL_ACCESS_PROTECTION::OFFSET, 0);
            assert_eq!(pio.read64(0x20000080).status, BUS_ERR);
            rig.w(reg::BASE + reg::PROTECTION_OVERRIDE::OFFSET, IIO_PROTECT_OVRRD_KEY);
            assert_eq!(pio.read64(0x20000080).status, BUS_OK);
            assert_eq!(pio.read64(0x10000000).status, BUS_ERR);
            assert_ne!(rig.r(reg::BASE + reg::LINK_CONTROL::OFFSET) & PEER, 0, "a widget answers the link");
            rig.chip.io.power_on();
            assert_eq!(rig.r(reg::BASE + reg::LOCAL_ACCESS_PROTECTION::OFFSET), !0);
            assert_eq!(rig.r(reg::BASE + reg::MISC_ERROR_MASK::OFFSET), 0x111);
            assert_ne!(rig.r(reg::BASE + reg::LINK_CONTROL::OFFSET) & PEER, 0, "the peer survives reset");
        }
        {
            let rig = rig();
            let chip = &rig.chip;
            assert_eq!(chip.inbound_write(14, Some(2), 0x80, 8, 42), BUS_OK);
            assert_eq!(chip.decode.physical().read64(0x80).data, 42);
            chip.io.disable_device(14, 2);
            assert_eq!(chip.inbound_read(14, Some(2), 0x80, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(14, None, 0x80, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(14, Some(1), 0x80, 8).status, BUS_OK);
            // A block needs the same admission, and a refused one touches nothing.
            assert_eq!(chip.inbound_write_block(14, Some(2), 0x2000, &[1, 2]), BUS_ERR);
            assert_eq!(chip.inbound_write_block(14, None, 0x2000, &[1, 2]), BUS_ERR);
            assert_eq!(chip.inbound_read_block(14, Some(2), 0x80, &mut [0]), BUS_ERR);
            assert_eq!(chip.decode.physical().read64(0x2000).data, 0);
            assert_eq!(chip.inbound_write_block(14, Some(1), 0x2000, &[1, 2]), BUS_OK);
            assert_eq!(chip.decode.physical().read64(0x2008).data, 2);
            // An interrupt packet to PI0's pending-modify register sets bank 0 bit 35.
            let a = (1 << 47) | 0x1800090;
            assert_eq!(chip.inbound_write(15, Some(0), a, 8, 0x123), BUS_OK);
            assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 1 << 35);
            assert_eq!(chip.inbound_write(0, None, a + PI1_BASE, 8, 0x101), BUS_OK, "widget 0 is admitted");
            assert_eq!(rig.r(PI1_BASE + pi::INT_PEND0::OFFSET), 1 << 1);
            assert_eq!(chip.inbound_read(15, Some(0), a, 8).status, BUS_ERR, "only stores reach a PI");
            assert_eq!(chip.inbound_write(15, Some(0), a, 4, 0x123), BUS_ERR, "only doublewords");
            assert_eq!(chip.inbound_read(15, Some(0), a | (1 << 39), 8).status, BUS_ERR, "another node");
            assert_eq!(chip.inbound_read(15, Some(0), 1 << 48, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(15, Some(0), a + 0x800000, 8).status, BUS_ERR);
            assert_eq!(chip.inbound_read(15, Some(0), BANK, 8).status, BUS_ERR, "beyond the bank");
        }
        // A busy widget answers busy through PIO, so the CPU retries.
        let rig = rig();
        rig.chip.connect_widget(8, Arc::new(Busy));
        let pio = rig.chip.pio();
        assert_eq!(pio.read64(0x8000000).status, BUS_BUSY);
        assert_eq!(pio.write64(0x8000000, 42), BUS_BUSY);
    }
    /// The guest observes BTE line copies, zero-fill and overlapping copies through its completion registers.
    #[test]
    fn the_bte_copies_and_zeroes_lines_before_completion() {
        let rig = rig();
        let m = rig.chip.decode.physical();
        for i in 0..32 {
            m.write64(0x1000 + i * 8, 0xfeed0000 | i);
        }
        m.write64(0x3000, 0xdead);
        launch(&rig, reg::BTE_LENGTH_STATUS::OFFSET, 2, 0x1000, 0x2000, 0x3000, 0x10);
        for i in 0..32 {
            assert_eq!(m.read64(0x2000 + i * 8).data, 0xfeed0000 | i);
        }
        assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 0);
        assert_eq!(rig.r(reg::BASE + reg::BTE_SOURCE::OFFSET), 0x1100);
        assert_eq!(m.read64(0x3000).data, 0, "the notification holds the final status");
        rig.w(reg::BASE + BTE_STRIDE + reg::BTE_INTERRUPT_ADDRESS::OFFSET, 53 << 16);
        launch(&rig, reg::BTE_LENGTH_STATUS::OFFSET + BTE_STRIDE, 1, 0, 0x2000, 0, 1);
        for i in 0..16 {
            assert_eq!(m.read64(0x2000 + i * 8).data, 0);
        }
        assert_eq!(rig.r(pi::INT_PEND0::OFFSET), 1 << 53, "the completion interrupt");
        // The copy goes a line at a time, read whole then written whole: a destination one line above
        // its source repeats the first line down the range.
        let rig = &Rig::new();
        rig.w(md::BASE + md::MEMORY_CONFIG::OFFSET, 0x11);
        let m = rig.chip.decode.physical();
        let line = |n: u64| 0x1000 + n * 128;
        for i in 0..16 {
            m.write64(line(0) + i * 8, 0xa0 + i);
        }
        launch(rig, reg::BTE_LENGTH_STATUS::OFFSET, 4, line(0), line(1), 0, 0x10);
        for n in 1..5 {
            for i in 0..16 {
                assert_eq!(m.read64(line(n) + i * 8).data, 0xa0 + i, "line {n} word {i}");
            }
        }
        assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 0);
    }
    /// A failed BTE line stops the transfer and reports the failing source and destination.
    #[test]
    fn the_bte_reports_transfer_errors_and_stops_after_a_failed_line() {
        // The second line runs off the end of the bank, on the read or the write side.
        for (src, dst) in [(BANK - 0x80, 0x2000), (0x1000, BANK - 0x80)] {
            let rig = rig();
            launch(&rig, reg::BTE_LENGTH_STATUS::OFFSET, 3, src, dst, 0x3000, 0x10);
            assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 2 | ERROR);
            assert_eq!(rig.r(reg::BASE + reg::BTE_SOURCE::OFFSET), src + 0x80);
            assert_eq!(rig.r(reg::BASE + reg::BTE_DESTINATION::OFFSET), dst + 0x80);
            assert_eq!(rig.chip.decode.physical().read64(0x3000).data, 2 | ERROR);
        }
        {
            let rig = rig();
            launch(&rig, reg::BTE_LENGTH_STATUS::OFFSET, 1, 0x1000, 0x2000, BANK, 0x10);
            assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 0, "a failed notification still completes");
            launch(&rig, reg::BTE_LENGTH_STATUS::OFFSET, 2, 0xffffffff80, 0x2000, 0x3000, 0x10);
            assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 2 | ERROR);
        }
        {
            let rig = rig();
            launch(&rig, reg::BTE_LENGTH_STATUS::OFFSET, 1, 0, 0, 0, 0x100);
            assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 1 | BUSY);
            rig.r(reg::BASE + reg::BTE_CONTROL::OFFSET);
            assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 1);
            launch(&rig, reg::BTE_LENGTH_STATUS::OFFSET, 0, 0, 0, 0, 0x10);
            assert_eq!(rig.r(reg::BASE + reg::BTE_LENGTH_STATUS::OFFSET), 0);
        }
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
