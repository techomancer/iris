//! Fuel's XBridge revision 2 contains two independent PCI heads and their
//! shared crossbar. Each head has its own 1024-entry ATE, device windows and
//! interrupt state. Revision 2 places aliases differently from the Octane's
//! single-head Bridge, uses different PCI slot data-phase widths, and sends
//! host DMA through widget 0 with source-aware packets.
//!
//! This remains separate from `bridge.rs` because its two-head topology,
//! alias map and host packet path are externally visible parts of this chip.
//! Generic PCI BAR-run and lane helpers are shared; the ATE and interrupt
//! dispatch remain tied to the XBridge head layout. Sorted BAR runs decode PCI
//! PIO without a `Physical` table; the machine's RAM still uses one.
//!
//! # What is approximated here
//!
//! Slot widths, immediate link initialization, synchronous flushes and
//! revision-2 alias placement follow the project's BSD QEMU model, not a
//! silicon measurement. Timing, external SRAM, flash and diagnostic engines
//! are absent. Undefined accesses latch an error and remain observable.
//!
//! Ported from hw/pci-host/sgi-xbridge.c and hw/misc/sgi-xbridge-xbow.c
//! in jamesbraid/qemu-sgi-origami, revision
//! fe128635b6dfe2511f1bd451684cb30007686df1.

use crate::dev::w1;
use crate::irq::{InterruptController, IrqLine};
use crate::pci::{bar_runs, BarRun, PciBar, PciBusDevice, PciDma, PciDmaStatus, PciLanes, Space};
use crate::physical::{read_bytes, write_bytes};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_OK};
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::{atomic::{AtomicU64, Ordering}, Arc, Weak};

pub const HOST_WIDGET: u8 = 10;
pub const HEAD_E: u8 = 14;
pub const HEAD_F: u8 = 15;
pub const ATE_ENTRIES: usize = 1024;
pub const ATE_MAPPED: usize = 512;

/// Offsets of the PCI head register words, at eight-byte strides.
pub mod reg {
    pub const IDENTIFICATION: usize = 0x0;
    pub const STATUS: usize = 0x8;
    pub const ERR_UPPER: usize = 0x10;
    pub const ERR_LOWER: usize = 0x18;
    pub const CONTROL: usize = 0x20;
    pub const REQUEST_TIMEOUT: usize = 0x28;
    pub const INT_DEST_UPPER: usize = 0x30;
    pub const INT_DEST_LOWER: usize = 0x38;
    pub const LINK_CONFIG: usize = 0x48;
    pub const TEST_PIN_CONTROL: usize = 0x70;
    pub const DIRECT_MAP: usize = 0x80;
    pub const SSRAM_PARITY_ERR: usize = 0x90;
    pub const ARBITRATION: usize = 0xa0;
    pub const NIC: usize = 0xb0;
    pub const BUS_TIMEOUT: usize = 0xc0;
    pub const PCI_TYPE1_CONFIG: usize = 0xc8;
    pub const PCI_ERR_LOWER: usize = 0xd8;
    pub const INTERRUPT_STATUS: usize = 0x100;
    pub const INTERRUPT_ENABLE: usize = 0x108;
    pub const INTERRUPT_RESET: usize = 0x110;
    pub const INTERRUPT_MODE: usize = 0x118;
    pub const INTERRUPT_DEVICE: usize = 0x120;
    pub const INTERRUPT_ERROR_HOST: usize = 0x128;
    pub const INTERRUPT_HOST_BASE: usize = 0x130;
    pub const INTERRUPT_HOST_LAST: usize = 0x168;
    pub const INTERRUPT_MULTIPLE: usize = 0x170;
    pub const FORCE_INTERRUPT_BASE: usize = 0x180;
    pub const FORCE_PIN_BASE: usize = 0x1c0;
    pub const FORCE_PIN_LAST: usize = 0x1f8;
    pub const DEVICE_BASE: usize = 0x200;
    pub const DEVICE_LAST: usize = 0x238;
    pub const RRB_EVEN: usize = 0x280;
    pub const RRB_ODD: usize = 0x288;
    pub const RRB_CLEAR: usize = 0x298;
    pub const ATE_UPPER_BASE: u64 = 0x10000;
    pub const ATE_LOWER_BASE: u64 = 0x12000;
    pub const ATE_END: u64 = 0x14000;

}

/// Offsets of the crossbar's widget 0 registers, at eight-byte strides.
///
/// The crossbar has the Crossbow's widget 0 register map, so these take the
/// names the Octane's Crossbow model (`xbow.rs`) gives them from the Crossbow
/// specification. Whichever of the two lands second should make them one.
pub mod xbow_reg {
    pub const IDENTIFICATION: usize = 0x0;
    pub const STATUS: usize = 0x8;
    pub const ERR_UPPER: usize = 0x10;
    pub const ERR_LOWER: usize = 0x18;
    pub const CONTROL: usize = 0x20;
    pub const PACKET_TIMEOUT: usize = 0x28;
    /// Widget 0's interrupt destination: target widget and address 47:32.
    pub const INT_UPPER: usize = 0x30;
    /// Widget 0's interrupt destination: address 31:0.
    pub const INT_LOWER: usize = 0x38;
    /// The error command word; a write clears it and the error address.
    pub const ERR_CMD: usize = 0x40;
    pub const LLP_CONTROL: usize = 0x48;
    /// Reads status, then clears its error bits and pending interrupt.
    pub const STATUS_CLEAR: usize = 0x50;
    pub const ARB_RELOAD: usize = 0x58;
    pub const PERF_A: usize = 0x60;
    pub const PERF_B: usize = 0x68;
    pub const NIC: usize = 0x70;
    /// Widget 0's reset fence, then links 8 to 15's at a stride of 8.
    pub const FENCE_BASE: usize = 0x78;
    pub const FENCE_LAST: usize = 0xb8;
    /// The resource lock: a write takes it only while it reads zero.
    pub const LOCK: usize = 0xc0;
    /// Any write releases the lock; a read returns the holder.
    pub const LOCK_CLEAR: usize = 0xc8;
    pub const READ_ONLY_BASE: usize = 0xd0;
    pub const READ_ONLY_LAST: usize = 0xf8;
    pub const LINK_BASE: usize = 0x100;
    pub const LINK_STRIDE: usize = 0x40;
    pub const LINK_END: u64 = 0x300;
    // Offsets within a link's block.
    pub const LINK_FLUSH: usize = 0x0;
    pub const LINK_CONTROL: usize = 0x8;
    pub const LINK_STATUS: usize = 0x10;
    pub const LINK_ARB_UPPER: usize = 0x18;
    pub const LINK_ARB_LOWER: usize = 0x20;
    pub const LINK_STATUS_CLEAR: usize = 0x28;
    pub const LINK_RESET: usize = 0x30;
    pub const LINK_AUX_STATUS: usize = 0x38;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InterruptPacket {
    pub source_widget: u8,
    pub widget: u8,
    pub address: u64,
    pub vector: u8,
    pub level: bool,
}
pub type InterruptSender = Arc<dyn Fn(InterruptPacket) -> bool + Send + Sync>;

/// Source identity and placement hints retained for a host's inbound packet path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmaPacket {
    pub source_widget: u8,
    pub slot: u8,
    pub widget: u8,
    pub address: u64,
    pub write: bool,
    pub half_write_requested: bool,
    pub half_read_requested: bool,
}
/// Bytes are in XIO address order after bridge placement. A false return
/// refuses that access. The host applies the requested exchange of doubleword halves.
pub type DmaSender = Arc<dyn Fn(DmaPacket, &mut [u8]) -> bool + Send + Sync>;

struct Routes {
    targets: RwLock<[Option<Arc<dyn BusDevice>>; 16]>,
    interrupts: RwLock<Option<InterruptSender>>,
    dma: RwLock<Option<DmaSender>>,
}
impl Routes {
    fn interrupt(&self, p: InterruptPacket) -> bool {
        let sender = self.interrupts.read().clone();
        let accepted = if let Some(send) = sender {
            send(p)
        } else {
            let target = self.targets.read()[p.widget as usize].clone();
            target.is_some_and(|t| {
                t.write64(
                    p.address,
                    p.vector as u64 | if p.level { 1 << 8 } else { 0 },
                ) == BUS_OK
            })
        };
        if !accepted {
            crate::dlog!(crate::devlog::LogModule::Bedrock, "xbridge: refused interrupt {p:?}");
        }
        accepted
    }
}

/// One chip owns both heads and widget 0; views share routing and reset state.
pub struct XBridge {
    routes: Arc<Routes>,
    heads: [Arc<XBridgeHead>; 2],
    xbow: Arc<Crossbar>,
}
impl XBridge {
    pub fn new(memory: Arc<dyn BusDevice>) -> Arc<Self> {
        let routes = Arc::new(Routes {
            targets: RwLock::new(std::array::from_fn(|i| {
                if i == HOST_WIDGET as usize {
                    Some(memory.clone())
                } else {
                    None
                }
            })),
            interrupts: RwLock::new(None),
            dma: RwLock::new(None),
        });
        let heads = [
            XBridgeHead::new(HEAD_E, 1, routes.clone()),
            XBridgeHead::new(HEAD_F, 0, routes.clone()),
        ];
        let xbow = Arc::new(Crossbar {
            st: Mutex::new(XbowState::new()),
            order: Mutex::new(()),
            routes: routes.clone(),
            heads: heads.each_ref().map(Arc::downgrade),
            reset_output: RwLock::new(None),
        });
        Arc::new(Self {
            routes,
            heads,
            xbow,
        })
    }
    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        for head in &self.heads {
            for (what, probe) in [
                ("state", (|h: &XBridgeHead| h.st.is_locked()) as fn(&XBridgeHead) -> bool),
                ("irq_order", |h| h.order.is_locked()),
                ("microlan", |h| h.microlan.is_locked()),
                ("pio_map", |h| h.pio_map.is_locked()),
            ] {
                let h = Arc::downgrade(head);
                let name = format!("xbridge::head{:x}::{what}", head.widget);
                register_lock_fn(name, move || h.upgrade().is_some_and(|h| probe(&h)));
            }
        }
        for (what, probe) in [
            ("state", (|x: &Crossbar| x.st.is_locked()) as fn(&Crossbar) -> bool),
            ("irq_order", |x| x.order.is_locked()),
        ] {
            let x = Arc::downgrade(&self.xbow);
            register_lock_fn(format!("xbridge::crossbar::{what}"), move || {
                x.upgrade().is_some_and(|x| probe(&x))
            });
        }
    }
    fn pci_head(&self, widget: u8) -> &Arc<XBridgeHead> {
        assert!((HEAD_E..=HEAD_F).contains(&widget));
        &self.heads[(widget - HEAD_E) as usize]
    }
    pub fn head(&self, widget: u8) -> Arc<dyn BusDevice> {
        self.pci_head(widget).clone()
    }
    pub fn crossbar(&self) -> Arc<dyn BusDevice> {
        self.xbow.clone()
    }
    /// A reader-specific widget-0 view; the host reads port 10 by default.
    pub fn crossbar_for_port(&self, port: u8) -> Arc<dyn BusDevice> {
        assert!((8..=15).contains(&port));
        Arc::new(CrossbarView {
            xbow: self.xbow.clone(),
            port,
        })
    }
    /// Host-port processor projection uses the XBridge widget-0 alias.
    pub fn widget(&self, widget: u8) -> Option<Arc<dyn BusDevice>> {
        match widget {
            0 | HOST_WIDGET => Some(self.crossbar()),
            HEAD_E | HEAD_F => Some(self.head(widget)),
            _ => self
                .routes
                .targets
                .read()
                .get(widget as usize)
                .and_then(Clone::clone),
        }
    }
    pub fn attach(&self, widget: u8, slot: u8, device: Arc<dyn PciBusDevice>) {
        self.pci_head(widget).attach(slot, device);
    }
    pub fn dma(&self, widget: u8, slot: u8) -> Arc<dyn PciDma> {
        assert!(slot < 8);
        Arc::new(SlotDma {
            head: Arc::downgrade(self.pci_head(widget)),
            slot,
        })
    }
    pub fn pin_line(&self, widget: u8, pin: u8) -> IrqLine {
        assert!(pin < 8);
        IrqLine::new(
            Arc::new(Pin {
                head: Arc::downgrade(self.pci_head(widget)),
                pci: false,
            }),
            pin as u32,
        )
    }
    /// Raw PCI INTx is board wiring, independent of the eight head pins.
    pub fn set_pci_irq(&self, widget: u8, slot: u8, pin: u8, output: IrqLine) {
        assert!(slot < 8 && pin < 4);
        self.pci_head(widget).pci_outputs.write()[(slot * 4 + pin) as usize] = Some(output);
    }
    pub fn set_interrupt_sender(&self, send: InterruptSender) {
        *self.routes.interrupts.write() = Some(send);
    }
    pub fn set_dma_sender(&self, send: DmaSender) {
        *self.routes.dma.write() = Some(send);
    }
    pub fn set_dma_64bit(&self, widget: u8, slot: u8, wide: bool) {
        assert!(slot < 8);
        let head = self.pci_head(widget);
        {
            let mut st = head.st.lock();
            st.dma_64bit = (st.dma_64bit & !(1 << slot)) | ((wide as u8) << slot);
        }
        head.publish_bars();
    }
    pub fn set_wide_slot_mask(&self, widget: u8, mask: u8) {
        self.pci_head(widget).st.lock().wide_slots = mask;
    }
    pub fn set_microlan(&self, widget: u8, bus: w1::Bus) {
        *self.pci_head(widget).microlan.lock() = Some(bus);
    }
    /// An interrupt packet from the widget attached at `p.source_widget`,
    /// routed as the heads' packets are. False when nothing takes it.
    pub fn send_interrupt(&self, p: InterruptPacket) -> bool {
        assert!((8..14).contains(&p.source_widget) && p.source_widget != HOST_WIDGET);
        self.routes.interrupt(p)
    }
    pub fn attach_widget(&self, widget: u8, target: Arc<dyn BusDevice>) {
        assert!((8..14).contains(&widget) && widget != HOST_WIDGET);
        self.routes.targets.write()[widget as usize] = Some(target);
        self.set_link_present(widget, true);
    }
    pub fn set_link_present(&self, widget: u8, present: bool) {
        assert!((8..=15).contains(&widget));
        let mut st = self.xbow.st.lock();
        st.present = (st.present & !(1 << (widget - 8))) | ((present as u8) << (widget - 8));
    }
    pub fn set_link_8bit(&self, widget: u8, enabled: bool) {
        assert!((8..=15).contains(&widget));
        let mut st = self.xbow.st.lock();
        let bit = 1 << (widget - 8);
        st.eight_bit = (st.eight_bit & !bit) | if enabled { bit } else { 0 };
    }
    /// A pulse for each reset port, outside chip locks. A head's registers,
    /// routed PCI lines and function headers are already reset. The board
    /// must reset that port's attached engines through this callback.
    pub fn set_reset_output(&self, output: Arc<dyn Fn(u8) + Send + Sync>) {
        *self.xbow.reset_output.write() = Some(output);
    }
    /// Time is supplied by the machine's virtual clock, in nanoseconds.
    pub fn warm_reset(&self, source_port: u8, now_ns: u64) {
        self.xbow.warm_reset(source_port, now_ns);
    }
    pub fn reset(&self) {
        for h in &self.heads {
            h.reset();
        }
        let mut st = self.xbow.st.lock();
        st.reset();
        st.quiet_until = 0;
    }
}
impl Resettable for XBridge {
    fn power_on(&self) {
        self.reset();
    }
}

struct HeadState {
    reg: [u32; 0x300 / 8],
    ate: [u64; ATE_ENTRIES],
    pins: u8,
    dma_64bit: u8,
    wide_slots: u8,
}
impl HeadState {
    fn r(&self, off: usize) -> u32 {
        self.reg[off / 8]
    }
    fn set(&mut self, off: usize, value: u32) {
        self.reg[off / 8] = value;
    }
    fn reset(&mut self, speed: u8) {
        self.reg.fill(0);
        self.ate.fill(0);
        self.pins = 0;
        for (off, value) in [
            (0, 0x2d002049),
            (8, 0x20),
            (0x20, 0x0f00320f | ((speed as u32) << 4)),
            (0x28, 0xfffff),
            (reg::LINK_CONFIG, 0x03ff1810),
            (0x80, 0x20000),
            (0xa0, 0x20008),
            (0xc0, 0x10000),
        ] {
            self.set(off, value);
        }
        for slot in 0..8 {
            self.set(
                0x200 + slot * 8,
                0x18001000
                    | if slot < 2 {
                        slot as u32 * 2
                    } else {
                        slot as u32 + 2
                    },
            );
        }
    }
    fn packet(&self, pin: Option<usize>, level: bool) -> InterruptPacket {
        let upper = self.r(reg::INT_DEST_UPPER);
        let mut address = ((upper as u64 & 0xffff) << 32) | self.r(reg::INT_DEST_LOWER) as u64;
        let vector = if let Some(pin) = pin {
            let host = self.r(reg::INTERRUPT_HOST_BASE + pin * 8);
            address = (address & !(0x3ff << 38)) | (((host >> 8) as u64 & 0x3ff) << 38);
            host as u8
        } else {
            self.r(reg::INTERRUPT_ERROR_HOST) as u8
        };
        InterruptPacket {
            source_widget: (self.r(reg::CONTROL) & 15) as u8,
            widget: ((upper >> 16) & 15) as u8,
            address,
            vector,
            level,
        }
    }
}
// Lock order: order, then microlan (NIC only), then st.
struct XBridgeHead {
    widget: u8,
    speed: u8,
    st: Mutex<HeadState>,
    order: Mutex<()>,
    routes: Arc<Routes>,
    slots: RwLock<[Option<Arc<dyn PciBusDevice>>; 8]>,
    pci_outputs: RwLock<[Option<IrqLine>; 32]>,
    microlan: Mutex<Option<w1::Bus>>,
    pio: Arc<PioDecode>,
    pio_map: Mutex<PioMap>,
}

#[derive(Default)]
struct PioMap {
    entries: HashMap<(u8, usize, usize), MappedBar>,
    runs: Vec<(u64, u64)>,
}

struct MappedBar {
    function: Arc<dyn PciBusDevice>,
    lanes: Arc<Lanes>,
    device: Arc<dyn BusDevice>,
}

/// Whole BAR-page runs are few and carry no RAM or native fault-handler state.
/// Readers clone one endpoint and release the map lock before calling a device.
#[derive(Default)]
struct PioDecode {
    runs: RwLock<Vec<BarRun>>,
}
impl PioDecode {
    fn publish(&self, mut runs: Vec<BarRun>) -> Vec<BarRun> {
        runs.sort_unstable_by_key(|run| run.start);
        debug_assert!(runs.windows(2).all(|pair| pair[0].end <= pair[1].start));
        std::mem::replace(&mut *self.runs.write(), runs)
    }
    fn target(&self, address: u64, size: usize) -> Option<Arc<dyn BusDevice>> {
        let runs = self.runs.read();
        let i = runs.partition_point(|run| run.start <= address).checked_sub(1)?;
        let run = &runs[i];
        (address.checked_add(size as u64)? <= run.end).then(|| run.dev.clone())
    }
}

struct Lanes {
    // BAR-relative CPU base, data-phase width and stream/swap policy.
    policy: AtomicU64,
    alias: bool,
    size: u64,
}

const PIO_INVALID: u64 = 1 << 63;

impl Lanes {
    fn access(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, value: Option<u64>) -> u64 {
        let policy = self.policy.load(Ordering::Acquire);
        if policy & PIO_INVALID != 0 {
            return if value.is_none() { u64::MAX >> (64 - size * 8) } else { 0 };
        }
        self.cycle(f, bar, off.wrapping_sub(policy as u32), size, value, policy)
    }

    fn cycle(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, value: Option<u64>, policy: u64) -> u64 {
        let width = if policy & (1 << 32) != 0 { 8 } else { 4 };
        if size > width {
            let hi = self.cycle(f, bar, off, 4, value.map(|v| v >> 32), policy);
            let lo = self.cycle(f, bar, off + 4, 4, value.map(|v| v as u32 as u64), policy);
            return (hi << 32) | lo;
        }
        let selected = policy & (1 << 33) != 0;
        let reverse = if self.alias { !selected } else { selected };
        let placed = if selected { off } else { off ^ (width - size) as u32 };
        // A reader may have obtained the old table entry before relocation.
        // Its new policy must not send the stale address outside the BAR.
        if placed as u64 + size as u64 > self.size {
            return if value.is_none() { u64::MAX >> (64 - size * 8) } else { 0 };
        }
        let mut bytes = [0; 8];
        if let Some(v) = value {
            for (i, b) in bytes[..size].iter_mut().enumerate() {
                let shift = if self.alias { (size - 1 - i) * 8 } else { i * 8 };
                *b = (v >> shift) as u8;
            }
            if reverse { bytes[..size].reverse(); }
        }
        XBridgeHead::bar_cycle(f, bar, placed, &mut bytes[..size], value.is_some());
        if reverse && value.is_none() { bytes[..size].reverse(); }
        if self.alias {
            bytes[..size].iter().fold(0, |v, &b| (v << 8) | b as u64)
        } else {
            bytes[..size].iter().enumerate().fold(0, |v, (i, &b)| v | ((b as u64) << (i * 8)))
        }
    }
}

impl PciLanes for Lanes {
    fn read(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize) -> u64 {
        self.access(f, bar, off, size, None)
    }
    fn write(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, value: u64) {
        self.access(f, bar, off, size, Some(value));
    }
}
struct Pin {
    head: Weak<XBridgeHead>,
    pci: bool,
}
impl InterruptController for Pin {
    fn set_line(&self, line: u32, level: bool) {
        if let Some(head) = self.head.upgrade() {
            if self.pci {
                let output = head
                    .pci_outputs
                    .read()
                    .get(line as usize)
                    .and_then(Clone::clone);
                if let Some(output) = output {
                    output.set(level);
                } else if level {
                    crate::dlog!(crate::devlog::LogModule::Bedrock,
                        "xbridge: head {} PCI line {line} has no board wiring",
                        head.widget
                    );
                }
            } else {
                head.pin(line as usize, level);
            }
        }
    }
}
struct SlotDma {
    head: Weak<XBridgeHead>,
    slot: u8,
}
impl PciDma for SlotDma {
    fn dma_read(&self, address: u64, bytes: &mut [u8]) -> PciDmaStatus {
        if let Some(h) = self.head.upgrade() {
            h.dma(self.slot, address, bytes, false)
        } else {
            bytes.fill(0);
            PciDmaStatus::MasterAbort
        }
    }
    fn dma_write(&self, address: u64, bytes: &[u8]) -> PciDmaStatus {
        self.head.upgrade().map_or(PciDmaStatus::MasterAbort, |h| {
            h.dma(self.slot, address, &mut bytes.to_vec(), true)
        })
    }
}
impl XBridgeHead {
    fn new(widget: u8, speed: u8, routes: Arc<Routes>) -> Arc<Self> {
        let mut st = HeadState {
            reg: [0; 96],
            ate: [0; ATE_ENTRIES],
            pins: 0,
            dma_64bit: 0,
            wide_slots: 0,
        };
        st.reset(speed);
        Arc::new(Self {
            widget,
            speed,
            st: Mutex::new(st),
            order: Mutex::new(()),
            routes,
            slots: RwLock::new(std::array::from_fn(|_| None)),
            pci_outputs: RwLock::new(std::array::from_fn(|_| None)),
            microlan: Mutex::new(None),
            pio: Arc::new(PioDecode::default()),
            pio_map: Mutex::new(PioMap::default()),
        })
    }
    fn attach(self: &Arc<Self>, slot: u8, device: Arc<dyn PciBusDevice>) {
        assert!(slot < 8);
        self.slots.write()[slot as usize] = Some(device.clone());
        device.connect_dma(Arc::new(SlotDma {
            head: Arc::downgrade(self),
            slot,
        }));
        let pin = device.config().interrupt_pin();
        if (1..=4).contains(&pin) {
            device.connect_interrupt(IrqLine::new(
                Arc::new(Pin {
                    head: Arc::downgrade(self),
                    pci: true,
                }),
                slot as u32 * 4 + pin as u32 - 1,
            ));
        }
        self.publish_bars();
    }
    fn pin(&self, pin: usize, level: bool) {
        if pin >= 8 {
            return;
        }
        let _order = self.order.lock();
        let packet = {
            let mut st = self.st.lock();
            let bit = 1 << pin;
            let old = st.pins & bit != 0;
            st.pins = (st.pins & !bit) | if level { bit } else { 0 };
            let pins = st.pins as u32;
            let status = (st.r(reg::INTERRUPT_STATUS) & !255) | pins;
            st.set(reg::INTERRUPT_STATUS, status);
            (old != level
                && st.r(reg::INTERRUPT_ENABLE) & bit as u32 != 0
                && (level || st.r(reg::INTERRUPT_MODE) & bit as u32 != 0))
                .then(|| st.packet(Some(pin), level))
        };
        if let Some(packet) = packet {
            self.routes.interrupt(packet);
        }
    }
    fn reset(&self) {
        {
            let _order = self.order.lock();
            self.st.lock().reset(self.speed);
        }
        let outputs = self.pci_outputs.read().clone();
        for line in outputs.into_iter().flatten() {
            line.set(false);
        }
        let devices = self.slots.read().clone();
        for d in devices.into_iter().flatten() {
            d.config().reset();
        }
        self.publish_bars();
    }
    fn invalid(&self, address: u64) {
        let _order = self.order.lock();
        let packet = {
            let mut st = self.st.lock();
            let first = st.r(reg::INTERRUPT_STATUS) & (1 << 24) == 0;
            let status = st.r(reg::INTERRUPT_STATUS) | (1 << 24);
            st.set(reg::INTERRUPT_STATUS, status);
            let multi = st.r(reg::INTERRUPT_MULTIPLE) | (1 << 24);
            st.set(reg::INTERRUPT_MULTIPLE, multi);
            if first {
                st.set(reg::ERR_UPPER, (address >> 32) as u32);
                st.set(reg::ERR_LOWER, address as u32);
            }
            (first && st.r(reg::INTERRUPT_ENABLE) & (1 << 24) != 0).then(|| st.packet(None, true))
        };
        if let Some(p) = packet {
            self.routes.interrupt(p);
        }
        crate::dlog!(crate::devlog::LogModule::Bedrock,
            "xbridge: head {} invalid access at {address:#x}",
            self.widget
        );
    }
    fn defined(off: usize) -> bool {
        matches!(off, reg::IDENTIFICATION..=reg::TEST_PIN_CONTROL | reg::DIRECT_MAP | reg::SSRAM_PARITY_ERR | reg::ARBITRATION | reg::NIC | reg::BUS_TIMEOUT..=reg::PCI_ERR_LOWER | reg::INTERRUPT_STATUS..=reg::FORCE_PIN_LAST | reg::DEVICE_BASE..=reg::RRB_CLEAR)
    }
    fn deferred(address: u64) -> bool {
        (0x300..0x600).contains(&address)
            || (0x30000..0x30008).contains(&address)
            || (0xc00000..0x1000000).contains(&address)
    }
    fn config(&self, address: u64, size: usize, value: Option<u64>) -> u64 {
        if size == 8 {
            let hi = self.config(address, 4, value.map(|v| v >> 32));
            return (hi << 32) | self.config(address + 4, 4, value.map(|v| v & 0xffffffff));
        }
        let reg = ((address & 255) as usize) ^ (4 - size);
        let slot = if address < 0x28000 {
            ((address - 0x20000) >> 12) as usize
        } else {
            ((self.st.lock().r(reg::PCI_TYPE1_CONFIG) >> 11) & 31) as usize
        };
        let bus = if address < 0x28000 {
            0
        } else {
            (self.st.lock().r(reg::PCI_TYPE1_CONFIG) >> 16) & 255
        };
        let device = if bus == 0 && address & 0x700 == 0 {
            self.slots.read().get(slot).and_then(Clone::clone)
        } else {
            None
        };
        let mask = u32::MAX >> (32 - size * 8);
        let shift = (reg & 3) * 8;
        if let Some(dev) = device {
            if let Some(value) = value {
                dev.config().write_masked(
                    reg / 4,
                    (value as u32 & mask) << shift,
                    mask << shift,
                );
                self.publish_bars();
                0
            } else {
                ((dev.config().read(reg / 4) >> shift) & mask) as u64
            }
        } else {
            if value.is_none() {
                let mut st = self.st.lock();
                let multi = st.r(reg::INTERRUPT_MULTIPLE) | (1 << 11);
                st.set(reg::INTERRUPT_MULTIPLE, multi);
            }
            mask as u64
        }
    }
    fn reg_write(&self, address: u64, value: u32) {
        let off = address as usize & !7;
        let _order = self.order.lock();
        let mut packets = Vec::new();
        if off == 0xb0 {
            let pulse = (value >> 10) & 0x3ff;
            let sample = (value >> 2) & 255;
            let mut bus = self.microlan.lock();
            if let Some(bus) = bus.as_mut() {
                let level = w1::slot::slot(bus, pulse, sample);
                self.st.lock().set(reg::NIC, 2 | level as u32);
            } else {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "xbridge: head {} NIC has no board", self.widget);
            }
            return;
        }
        {
            let mut st = self.st.lock();
            let masked = match off {
                reg::CONTROL => (value & 0x0fe1f20f) | ((self.speed as u32) << 4),
                reg::REQUEST_TIMEOUT | reg::INT_DEST_UPPER => value & 0xfffff,
                reg::INT_DEST_LOWER | reg::RRB_EVEN | reg::RRB_ODD => value,
                reg::LINK_CONFIG => value & 0x03ffffff,
                reg::TEST_PIN_CONTROL => value & 0xffffff,
                reg::DIRECT_MAP => value & 0xf3ffff,
                reg::ARBITRATION => value & 0x3ff7f,
                reg::BUS_TIMEOUT => value & 0x1f03ff,
                reg::PCI_TYPE1_CONFIG => value & 0xfff800,
                reg::INTERRUPT_DEVICE => value & 0xffffff,
                reg::INTERRUPT_ERROR_HOST | reg::INTERRUPT_MODE => value & 255,
                reg::INTERRUPT_ENABLE => {
                    let value = value & 0x7fffffff;
                    let newly = value & !st.r(reg::INTERRUPT_ENABLE) & st.pins as u32;
                    st.set(off, value);
                    for pin in 0..8 {
                        if newly & (1 << pin) != 0 {
                            packets.push(st.packet(Some(pin), true));
                        }
                    }
                    value
                }
                reg::INTERRUPT_RESET => {
                    let clear = (if value & 2 != 0 { 1 << 30 } else { 0 })
                        | (if value & 8 != 0 { 1 << 24 } else { 0 });
                    let status = st.r(reg::INTERRUPT_STATUS) & !clear;
                    st.set(reg::INTERRUPT_STATUS, status);
                    let multi = if value & 64 != 0 {
                        0
                    } else {
                        st.r(reg::INTERRUPT_MULTIPLE) & !clear
                    };
                    st.set(reg::INTERRUPT_MULTIPLE, multi);
                    return;
                }
                reg::FORCE_INTERRUPT_BASE..=reg::FORCE_PIN_LAST => {
                    let pin = (off - reg::FORCE_INTERRUPT_BASE) / 8 & 7;
                    if st.r(reg::INTERRUPT_ENABLE) & (1 << pin) == 0 {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "xbridge: masked force semantics unresolved");
                    } else if off < reg::FORCE_PIN_BASE || st.pins & (1 << pin) != 0 {
                        packets.push(st.packet(Some(pin), true));
                    }
                    0
                }
                reg::INTERRUPT_HOST_BASE..=reg::INTERRUPT_HOST_LAST => value & 0x3ffff,
                reg::DEVICE_BASE..=reg::DEVICE_LAST => value & 0x1eefffff,
                _ => {
                    crate::dlog!(crate::devlog::LogModule::Bedrock,
                        "xbridge: head {} ignored register write {off:#x}",
                        self.widget
                    );
                    return;
                }
            };
            if !(reg::FORCE_INTERRUPT_BASE..=reg::FORCE_PIN_LAST).contains(&off) {
                st.set(off, masked);
            }
        }
        for p in packets {
            self.routes.interrupt(p);
        }
        drop(_order);
        if off == reg::CONTROL || (reg::DEVICE_BASE..=reg::DEVICE_LAST).contains(&off) {
            self.publish_bars();
        }
    }

    fn publish_bars(&self) {
        // Keep endpoint identities stable across BAR relocation. Publish the
        // complete small decoder after dropping state and attachment locks.
        let mut map = self.pio_map.lock();
        let (regs, control, wide) = {
            let st = self.st.lock();
            let regs: [u32; 8] = std::array::from_fn(|i| st.r(reg::DEVICE_BASE + i * 8));
            (regs, st.r(reg::CONTROL), st.dma_64bit)
        };
        let funcs: Vec<_> = self.slots.read().iter().enumerate()
            .filter_map(|(slot, f)| Some((slot, f.clone()?))).collect();
        let windows: Vec<_> = funcs.iter().flat_map(|(slot, f)| {
            f.config().windows().into_iter().map(move |(space, bar, base, size)| (*slot, bar, space, base as u64, base as u64 + size as u64))
        }).collect();
        map.runs.clear();
        let mut decoded = Vec::new();
        let mut retired = Vec::new();
        // Host replacement changes an endpoint's function. Guest configuration
        // writes only update policy and never allocate another table endpoint.
        map.entries.retain(|&(_, slot, _), entry| {
            let keep = funcs.iter().any(|(s, f)| *s == slot && Arc::ptr_eq(f, &entry.function));
            if !keep {
                entry.lanes.policy.fetch_or(PIO_INVALID, Ordering::AcqRel);
                retired.push(entry.device.clone());
            }
            keep
        });
        for kind in 0u8..11 {
            let (cpu, len, space, pci, width, selected) = if kind < 8 {
                let slot = kind as usize;
                let dev = regs[slot];
                let memory = dev & (1 << 12) != 0;
                let mut base = dev & if memory { 0x3ff } else { 0xfff };
                if slot < 2 { base &= !1; }
                let (cpu, len) = if slot < 2 {
                    (0x200000 + slot as u64 * 0x200000, 0x200000)
                } else {
                    (0x600000 + (slot as u64 - 2) * 0x100000, 0x100000)
                };
                (cpu, len, if memory { Space::Mem } else { Space::Io }, base << 20,
                    if dev & (1 << 22) != 0 { 8 } else { 4 }, dev & (1 << 13) != 0)
            } else {
                let (cpu, len, space, width) = match kind {
                    8 => (0x40000000, 0x40000000, Space::Mem, 4),
                    9 => (0x80000000, 0x40000000, Space::Mem, 8),
                    _ => (0x100000000, 0x100000000, Space::Io, 4),
                };
                (cpu, len, space, 0, width, control & (1 << if space == Space::Mem { 22 } else { 23 }) != 0)
            };
            let runs = bar_runs(&funcs, space, cpu, pci, len, |b| {
                let width = if kind == 9 && wide & (1 << b.slot) == 0 { 4 } else { width };
                let policy = b.cpu_base as u32 as u64 | ((width == 8) as u64) << 32 | (selected as u64) << 33;
                let entry = map.entries.entry((kind, b.slot, b.bar)).or_insert_with(|| {
                    let size = windows.iter().find(|&&(slot, bar, _, _, _)| slot == b.slot && bar == b.bar)
                        .map(|&(_, _, _, lo, hi)| hi - lo).unwrap();
                    let lanes = Arc::new(Lanes { policy: AtomicU64::new(policy), alias: kind >= 8, size });
                    let device = Arc::new(PciBar::new(lanes.clone(), b.dev.clone(), b.bar, u32::MAX as u64, 0));
                    MappedBar { function: b.dev.clone(), lanes, device }
                });
                entry.lanes.policy.store(policy, Ordering::Release);
                entry.device.clone()
            });
            for run in runs {
                let lo = run.start - cpu + pci as u64;
                let hi = run.end - cpu + pci as u64;
                // bar_runs rounds to whole pages. A subpage BAR can still
                // overlap one of those pages, so retain the slow decoder there.
                let claims = windows.iter().filter(|&&(_, _, s, l, h)| s == space && l < hi && lo < h).count();
                if claims != 1 { continue; }
                map.runs.push((run.start, run.end));
                decoded.push(run);
            }
        }
        let previous = self.pio.publish(decoded);
        drop(map);
        // Replaced functions may run shutdown callbacks from their destructor.
        drop(previous);
        drop(retired);
    }

    fn bar_cycle(dev: &dyn PciBusDevice, bar: usize, off: u32, bytes: &mut [u8], write: bool) {
        for (i, part) in bytes.chunks_mut(4).enumerate() {
            if write {
                let v = part.iter().enumerate().fold(0, |v, (j, &b)| v | ((b as u32) << (j * 8)));
                dev.bar_write(bar, off + i as u32 * 4, v, part.len());
            } else {
                let v = dev.bar_read(bar, off + i as u32 * 4, part.len());
                for (j, b) in part.iter_mut().enumerate() { *b = (v >> (j * 8)) as u8; }
            }
        }
    }
    /// PCI functions see little-endian values; bytes here remain in address order.
    fn pci_cycle(&self, space: Space, address: u32, bytes: &mut [u8], write: bool) -> bool {
        let claim = self.slots.read().iter().flatten().find_map(|dev| {
            let (bar, off) = dev.config().decode(space, address)?;
            let (end_bar, end) = dev
                .config()
                .decode(space, address.checked_add(bytes.len() as u32 - 1)?)?;
            (end_bar == bar && end == off + bytes.len() as u32 - 1).then(|| (dev.clone(), bar, off))
        });
        let Some((dev, bar, off)) = claim else {
            crate::dlog!(crate::devlog::LogModule::Bedrock,
                "xbridge: head {} no PCI BAR for {space:?} {address:#x}",
                self.widget
            );
            if !write {
                bytes.fill(0xff);
            }
            return false;
        };
        Self::bar_cycle(&*dev, bar, off, bytes, write);
        true
    }
    fn window_cycle(&self, slot: usize, off: u32, bytes: &mut [u8], write: bool) {
        let dev = self.st.lock().r(reg::DEVICE_BASE + slot * 8);
        let width = if dev & (1 << 22) != 0 { 8 } else { 4 };
        if bytes.len() > width {
            // Device windows split a doubleword high word first.
            let (low, high) = bytes.split_at_mut(4);
            self.window_cycle(slot, off, high, write);
            self.window_cycle(slot, off + 4, low, write);
            return;
        }
        let memory = dev & (1 << 12) != 0;
        let mut base = dev & if memory { 0x3ff } else { 0xfff };
        if slot < 2 {
            base &= !1;
        }
        let stream = dev & (1 << 13) != 0;
        let placed = if stream {
            off
        } else {
            off ^ (width - bytes.len()) as u32
        };
        if stream && write {
            bytes.reverse();
        }
        self.pci_cycle(
            if memory { Space::Mem } else { Space::Io },
            (base << 20).wrapping_add(placed),
            bytes,
            write,
        );
        if stream && !write {
            bytes.reverse();
        }
    }
    fn alias_cycle(&self, space: Space, address: u32, width: usize, bytes: &mut [u8], write: bool) {
        let width = if width == 8 {
            let slots = self.slots.read().clone();
            let access_len = bytes.len();
            let matches: Vec<_> = slots
                .iter()
                .enumerate()
                .flat_map(|(slot, dev)| {
                    dev.as_ref()
                        .map(|d| d.config().windows())
                        .unwrap_or_default()
                        .into_iter()
                        .filter_map(move |(s, _, base, size)| {
                            (s == space
                                && address >= base
                                && (address as u64 + access_len as u64)
                                    <= base as u64 + size as u64)
                                .then_some(slot)
                        })
                })
                .collect();
            if matches.len() == 1 && self.st.lock().dma_64bit & (1 << matches[0]) == 0 {
                4
            } else {
                8
            }
        } else {
            width
        };
        if bytes.len() > width {
            let (first, last) = bytes.split_at_mut(width);
            self.alias_cycle(space, address, width, first, write);
            self.alias_cycle(space, address + width as u32, width, last, write);
            return;
        }
        let swap = self.st.lock().r(reg::CONTROL) & (1 << if space == Space::Mem { 22 } else { 23 }) != 0;
        let address = if swap {
            address
        } else {
            address ^ (width - bytes.len()) as u32
        };
        if !swap && write {
            bytes.reverse();
        }
        self.pci_cycle(space, address, bytes, write);
        if !swap && !write {
            bytes.reverse();
        }
    }
    fn slow_pio(&self, address: u64, size: usize, value: Option<u64>) -> u64 {
        let write = value.is_some();
        let mut bytes = [0u8; 8];
        if (0x200000..0xc00000).contains(&address) {
            let (slot, off) = if address < 0x600000 {
                (((address - 0x200000) >> 21) as usize, address & 0x1fffff)
            } else {
                (2 + ((address - 0x600000) >> 20) as usize, address & 0xfffff)
            };
            if let Some(v) = value {
                for (i, b) in bytes[..size].iter_mut().enumerate() {
                    *b = (v >> (i * 8)) as u8;
                }
            }
            self.window_cycle(slot, off as u32, &mut bytes[..size], write);
            return bytes[..size]
                .iter()
                .enumerate()
                .fold(0, |v, (i, &b)| v | ((b as u64) << (i * 8)));
        }
        let alias = if (0x40000000..0x80000000).contains(&address) {
            Some((Space::Mem, address - 0x40000000, 4))
        } else if (0x80000000..0xc0000000).contains(&address) {
            Some((Space::Mem, address - 0x80000000, 8))
        } else if (0x100000000..0x200000000).contains(&address) {
            Some((Space::Io, address - 0x100000000, 4))
        } else {
            None
        };
        if let Some((space, off, width)) = alias {
            if let Some(v) = value {
                for (i, b) in bytes[..size].iter_mut().enumerate() {
                    *b = (v >> ((size - 1 - i) * 8)) as u8;
                }
            }
            self.alias_cycle(space, off as u32, width, &mut bytes[..size], write);
            return bytes[..size].iter().fold(0, |v, &b| (v << 8) | b as u64);
        }
        0
    }
    fn access(&self, address: u64, size: usize, value: Option<u64>) -> u64 {
        let write = value.is_some();
        if (0x20000..0x29000).contains(&address) {
            return self.config(address, size, value);
        }
        if (0x200000..0xc00000).contains(&address)
            || (0x40000000..0xc0000000).contains(&address)
            || (0x100000000..0x200000000).contains(&address)
        {
            // An unaligned access can straddle a BAR's final mapped page.
            if address & 0xfff > 0x1000 - size as u64 {
                return self.slow_pio(address, size, value);
            }
            let Some(endpoint) = self.pio.target(address, size) else {
                return self.slow_pio(address, size, value);
            };
            return if let Some(v) = value {
                match size {
                    1 => { endpoint.write8(address, v as u8); }
                    2 => { endpoint.write16(address, v as u16); }
                    4 => { endpoint.write32(address, v as u32); }
                    _ => { endpoint.write64(address, v); }
                }
                0
            } else {
                match size {
                    1 => endpoint.read8(address).data as u64,
                    2 => endpoint.read16(address).data as u64,
                    4 => endpoint.read32(address).data as u64,
                    _ => endpoint.read64(address).data,
                }
            };
        }
        if size == 8 && address & 7 == 0 && (reg::ATE_UPPER_BASE..reg::ATE_LOWER_BASE).contains(&address) && write {
            self.st.lock().ate[((address - reg::ATE_UPPER_BASE) / 8) as usize] =
                value.unwrap() & 0x0000ffffffffffff;
            return 0;
        }
        if size == 4 && address & 7 == 4 {
            if !write {
                if (reg::ATE_UPPER_BASE..reg::ATE_LOWER_BASE).contains(&address) {
                    return self.st.lock().ate[((address - reg::ATE_UPPER_BASE) / 8) as usize] >> 32;
                }
                if (reg::ATE_LOWER_BASE..reg::ATE_END).contains(&address) {
                    return self.st.lock().ate[((address - reg::ATE_LOWER_BASE) / 8) as usize] as u32 as u64;
                }
            }
            if address < 0x300 && Self::defined(address as usize & !7) {
                if let Some(value) = value {
                    self.reg_write(address, value as u32);
                    return 0;
                }
                return self.st.lock().r(address as usize & !7) as u64;
            }
        }
        if Self::deferred(address) {
            crate::dlog!(crate::devlog::LogModule::Bedrock,
                "xbridge: head {} deferred access at {address:#x}",
                self.widget
            );
        } else {
            self.invalid(address);
        }
        0
    }
    fn translate(
        &self,
        slot: u8,
        address: u64,
        write: bool,
        packets: bool,
    ) -> Option<(u8, u64, u64, bool)> {
        let mut st = self.st.lock();
        let dev = st.r(reg::DEVICE_BASE + slot as usize * 8);
        let wide = st.dma_64bit & (1 << slot) != 0;
        let device_swap = dev & (1 << 19) != 0;
        let (widget, target, swap, direct, page, wide64) = if (0x40000000..0x40800000)
            .contains(&address)
            || (0x60000000..0x60800000).contains(&address)
        {
            let shift = if st.r(reg::CONTROL) & (1 << 21) != 0 { 14 } else { 12 };
            let index = ((address & 0x1fffffff) >> shift) as usize;
            if index >= ATE_MAPPED || st.ate[index] & 1 == 0 {
                let v = st.r(reg::INTERRUPT_STATUS) | (1 << 30);
                st.set(reg::INTERRUPT_STATUS, v);
                let v = st.r(reg::INTERRUPT_MULTIPLE) | (1 << 30);
                st.set(reg::INTERRUPT_MULTIPLE, v);
                crate::dlog!(
                    crate::devlog::LogModule::Bedrock,
                    "xbridge: invalid page DMA {address:#x}, entry {index}"
                );
                return None;
            }
            let ate = st.ate[index];
            let mask = (1 << shift) - 1;
            (
                ((ate >> 8) & 15) as u8,
                (ate & 0x0000fffffffff000 & !mask) | (address & mask),
                ate & 32 != 0 || address & (1 << 29) != 0,
                false,
                true,
                false,
            )
        } else if address >= 0x1000000000000000 {
            (
                (address >> 60) as u8,
                address & 0x0000ffffffffffff,
                address & (1 << 55) != 0,
                false,
                false,
                true,
            )
        } else if (0x80000000..=0xffffffff).contains(&address) {
            let direct = st.r(reg::DIRECT_MAP);
            let mut target = ((direct as u64 & 0x1ffff) << 31) | (address & 0x7fffffff);
            if direct & 0x1ffff == 0 && direct & (1 << 17) != 0 {
                target += 0x20000000;
            }
            (
                ((direct >> 20) & 15) as u8,
                target,
                device_swap,
                true,
                false,
                false,
            )
        } else {
            crate::dlog!(crate::devlog::LogModule::Bedrock, "xbridge: unsupported DMA {address:#x}");
            return None;
        };
        let half = packets
            && swap
            && st.wide_slots & (1 << slot) != 0
            && if write {
                direct || page || (wide64 && device_swap)
            } else {
                direct || (wide64 && wide && device_swap) || (page && address & (1 << 29) != 0)
            };
        let mask = if swap || half {
            0
        } else if wide {
            7
        } else {
            3
        };
        Some((widget, target, mask, half))
    }
    // Retain the existing memory path's probe before issuing a chunk's writes.
    // The neutral helper alone can commit bytes before reporting a refusal.
    fn write_memory(mem: &dyn BusDevice, address: u64, data: &[u8]) -> bool {
        if data.is_empty() {
            return true;
        }
        let first = address & !7;
        let Some(end) = address
            .checked_add(data.len() as u64)
            .and_then(|a| a.checked_add(7))
        else {
            return false;
        };
        let mut probe = vec![0; ((end & !7) - first) as usize];
        read_bytes(mem, first, &mut probe) == BUS_OK && write_bytes(mem, address, data) == BUS_OK
    }
    fn dma(&self, slot: u8, address: u64, bytes: &mut [u8], write: bool) -> PciDmaStatus {
        let sender = self.routes.dma.read().clone();
        let mut good = true;
        let mut done = 0;
        while done < bytes.len() {
            let Some(at) = address.checked_add(done as u64) else {
                if !write {
                    bytes[done..].fill(0);
                }
                return PciDmaStatus::MasterAbort;
            };
            let n = (0x1000 - (at as usize & 0xfff)).min(bytes.len() - done);
            if (0x3fff0000..0x40000000).contains(&at) {
                if !write {
                    bytes[done..done + n].fill(0);
                }
                done += n;
                continue;
            }
            if !write {
                bytes[done..done + n].fill(0);
            }
            let translated = self.translate(slot, at, write, sender.is_some());
            let accepted = if let Some((widget, target, mask, half)) = translated {
                if let Some(send) = &sender {
                    // Packet legs use naturally aligned accesses, retaining source identity.
                    let mut offset = 0;
                    let mut ok = true;
                    while offset < n {
                        let p = target + offset as u64;
                        let mut size = 8usize;
                        while size > n - offset || p & (size as u64 - 1) != 0 {
                            size /= 2;
                        }
                        let mut data = [0u8; 8];
                        if write {
                            for i in 0..size {
                                data[i] = bytes[done
                                    + offset
                                    + if half {
                                        i
                                    } else {
                                        i ^ (mask as usize & (size - 1))
                                    }];
                            }
                        }
                        let placed = if half {
                            p
                        } else {
                            p ^ (mask & !(size as u64 - 1))
                        };
                        let packet = DmaPacket {
                            source_widget: (self.st.lock().r(reg::CONTROL) & 15) as u8,
                            slot,
                            widget,
                            address: placed,
                            write,
                            half_write_requested: half && write && size == 8,
                            half_read_requested: half && !write && size == 8,
                        };
                        let part = send(packet, &mut data[..size]);
                        ok &= part;
                        if !write {
                            for i in 0..size {
                                bytes[done + offset + i] = if part {
                                    data[if half {
                                        i
                                    } else {
                                        i ^ (mask as usize & (size - 1))
                                    }]
                                } else {
                                    0
                                };
                            }
                        }
                        offset += size;
                    }
                    ok
                } else {
                    let target_device = self.routes.targets.read()[widget as usize].clone();
                    target_device.is_some_and(|mem| {
                        let low = target & !mask;
                        let high = (target + n as u64 + mask) & !mask;
                        let mut data = vec![0; (high - low) as usize];
                        if write {
                            for i in 0..n {
                                data[(((target + i as u64) ^ mask) - low) as usize] =
                                    bytes[done + i];
                            }
                            if low == target && high == target + n as u64 {
                                Self::write_memory(&*mem, low, &data)
                            } else {
                                let mut ok = true;
                                for i in 0..n {
                                    let p = (target + i as u64) ^ mask;
                                    ok &= Self::write_memory(
                                        &*mem,
                                        p,
                                        &data[(p - low) as usize..(p - low) as usize + 1],
                                    );
                                }
                                ok
                            }
                        } else {
                            // Preserve the old helper's covering-doubleword probe
                            // and zero the whole failed physical chunk, even if
                            // the neutral helper returned part of its contents.
                            let first = low & !7;
                            let mut fetched = vec![0; (((high + 7) & !7) - first) as usize];
                            let ok = read_bytes(&*mem, first, &mut fetched) == BUS_OK;
                            for i in 0..n {
                                bytes[done + i] = if ok {
                                    fetched[(((target + i as u64) ^ mask) - first) as usize]
                                } else {
                                    0
                                };
                            }
                            ok
                        }
                    })
                }
            } else {
                false
            };
            if !accepted {
                good = false;
            }
            done += n;
        }
        if good {
            PciDmaStatus::Ok
        } else {
            PciDmaStatus::MasterAbort
        }
    }
}

struct XbowState {
    reg: [u32; 96],
    status: u32,
    present: u8,
    eight_bit: u8,
    quiet_until: u64,
}
impl XbowState {
    fn new() -> Self {
        let mut st = Self {
            reg: [0; 96],
            status: 0,
            present: (1 << 2) | (1 << 6) | (1 << 7),
            eight_bit: 0,
            quiet_until: 0,
        };
        st.reset();
        st
    }
    fn reset(&mut self) {
        self.reg.fill(0);
        self.status = 0;
        for (off, value) in [
            (xbow_reg::IDENTIFICATION, 0x2d000001),
            (xbow_reg::CONTROL, 0x80),
            (xbow_reg::PACKET_TIMEOUT, 0xfffff),
            (xbow_reg::INT_UPPER, 0x80000),
            (xbow_reg::LLP_CONTROL, 0x03ff1810),
            (xbow_reg::ARB_RELOAD, 2),
            (xbow_reg::NIC, 2),
            (xbow_reg::FENCE_BASE, 255),
        ] {
            self.reg[off / 8] = value;
        }
        for i in 0..8 {
            self.reg[(xbow_reg::FENCE_BASE + 8 + i * 8) / 8] = 255;
            self.reg[(xbow_reg::LINK_BASE + xbow_reg::LINK_CONTROL + i * xbow_reg::LINK_STRIDE) / 8] = 0x80000;
        }
    }
    fn packet(&self, level: bool) -> InterruptPacket {
        let upper = self.reg[xbow_reg::INT_UPPER / 8];
        InterruptPacket {
            source_widget: 0,
            widget: ((upper >> 16) & 15) as u8,
            address: ((upper as u64 & 0xffff) << 32) | self.reg[xbow_reg::INT_LOWER / 8] as u64,
            vector: (upper >> 24) as u8,
            level,
        }
    }
}
// Lock order: order, then st. Reset outputs run with both released.
struct Crossbar {
    st: Mutex<XbowState>,
    order: Mutex<()>,
    routes: Arc<Routes>,
    heads: [Weak<XBridgeHead>; 2],
    reset_output: RwLock<Option<Arc<dyn Fn(u8) + Send + Sync>>>,
}
impl Crossbar {
    fn reset_port(&self, port: u8) {
        if (14..=15).contains(&port) {
            if let Some(h) = self.heads[(port - 14) as usize].upgrade() {
                h.reset();
            }
        }
        let out = self.reset_output.read().clone();
        if let Some(out) = out {
            out(port);
        }
    }
    fn warm_reset(&self, source: u8, now: u64) {
        assert!((8..=15).contains(&source));
        let ports = {
            let mut st = self.st.lock();
            if now < st.quiet_until {
                return;
            }
            st.quiet_until = now.saturating_add(10240);
            if st.reg[xbow_reg::FENCE_BASE / 8] & (1 << (source - 8)) == 0 {
                return;
            }
            let ports: Vec<_> = (0..8)
                .filter(|&i| st.reg[(xbow_reg::FENCE_BASE + 8 + i * 8) / 8] & (1 << (source - 8)) != 0)
                .map(|i| i as u8 + 8)
                .collect();
            st.reset();
            ports
        };
        for port in ports {
            self.reset_port(port);
        }
    }
    fn access(&self, port: u8, address: u64, size: usize, value: Option<u64>) -> u64 {
        let _order = self.order.lock();
        let mut packet = None;
        let mut reset = None;
        let mut access_error = false;
        let result = {
            let mut st = self.st.lock();
            let off = address as usize & !7;
            if size != 4 || address & 7 != 4 || address >= xbow_reg::LINK_END || (xbow_reg::READ_ONLY_BASE..=xbow_reg::READ_ONLY_LAST).contains(&off) {
                st.status |= 32;
                access_error = true;
                if st.reg[xbow_reg::CONTROL / 8] & 32 != 0 && st.status & (1 << 23) == 0 {
                    st.status |= 1 << 23;
                    packet = Some(st.packet(true));
                }
                crate::dlog!(
                    crate::devlog::LogModule::Bedrock,
                    "xbridge-xbow: invalid access {address:#x} width {size}"
                );
                0
            } else if let Some(v) = value {
                let v = v as u32;
                if off >= xbow_reg::LINK_BASE {
                    match (off - xbow_reg::LINK_BASE) % xbow_reg::LINK_STRIDE {
                        xbow_reg::LINK_CONTROL => st.reg[off / 8] = v & 0xbf7f01ff,
                        xbow_reg::LINK_ARB_UPPER | xbow_reg::LINK_ARB_LOWER => st.reg[off / 8] = v,
                        xbow_reg::LINK_RESET => reset = Some((8 + (off - xbow_reg::LINK_BASE) / xbow_reg::LINK_STRIDE) as u8),
                        _ => {
                            st.status |= 32;
                            access_error = true;
                            crate::dlog!(
                                crate::devlog::LogModule::Bedrock,
                                "xbridge-xbow: read-only link register offset {off:#x}"
                            );
                        }
                    }
                } else {
                    match off {
                        xbow_reg::CONTROL => st.reg[off / 8] = v & 0xfe,
                        xbow_reg::PACKET_TIMEOUT => st.reg[off / 8] = v & 0xfffff,
                        xbow_reg::INT_UPPER => st.reg[off / 8] = (v & 0xff07ffff) | 0x80000,
                        xbow_reg::INT_LOWER => st.reg[off / 8] = v,
                        xbow_reg::ERR_CMD => {
                            st.reg[xbow_reg::ERR_CMD / 8] = 0;
                            st.reg[xbow_reg::ERR_UPPER / 8] = 0;
                            st.reg[xbow_reg::ERR_LOWER / 8] = 0;
                        }
                        xbow_reg::LLP_CONTROL => st.reg[off / 8] = v & 0x3ffffff,
                        xbow_reg::ARB_RELOAD => st.reg[off / 8] = v & 0x3f,
                        xbow_reg::PERF_A | xbow_reg::PERF_B => st.reg[off / 8] = v & 0x700000,
                        xbow_reg::NIC => {
                            st.reg[off / 8] = (v & 0xffffc) | 2;
                            if v & 0xffc00 != 0 {
                                crate::dlog!(
                                    crate::devlog::LogModule::Bedrock,
                                    "xbridge-xbow: one-wire pulse unimplemented at register offset 0x70"
                                );
                            }
                        }
                        xbow_reg::FENCE_BASE..=xbow_reg::FENCE_LAST => st.reg[off / 8] = v & 255,
                        xbow_reg::LOCK => {
                            if st.reg[off / 8] == 0 {
                                st.reg[off / 8] = v & 255;
                            }
                        }
                        xbow_reg::LOCK_CLEAR => st.reg[xbow_reg::LOCK / 8] = 0,
                        _ => {
                            st.status |= 32;
                            access_error = true;
                            crate::dlog!(
                                crate::devlog::LogModule::Bedrock,
                                "xbridge-xbow: read-only register offset {off:#x}"
                            );
                        }
                    }
                }
                0
            } else if off >= xbow_reg::LINK_BASE {
                let link = (off - xbow_reg::LINK_BASE) / xbow_reg::LINK_STRIDE;
                let present = st.present & (1 << link) != 0;
                match (off - xbow_reg::LINK_BASE) % xbow_reg::LINK_STRIDE {
                    xbow_reg::LINK_FLUSH => 0,
                    xbow_reg::LINK_STATUS | xbow_reg::LINK_STATUS_CLEAR => {
                        if present {
                            1 << 31
                        } else {
                            1 << 5
                        }
                    }
                    xbow_reg::LINK_AUX_STATUS => {
                        (if present { 1 << 5 } else { 1 << 6 })
                            | (if st.eight_bit & (1 << link) != 0 {
                                1 << 4
                            } else {
                                0
                            })
                    }
                    xbow_reg::LINK_RESET => {
                        st.status |= 32;
                        access_error = true;
                        0
                    }
                    _ => st.reg[off / 8] as u64,
                }
            } else {
                match off {
                    xbow_reg::STATUS => (st.status | ((port as u32) << 6)) as u64,
                    xbow_reg::STATUS_CLEAR => {
                        let v = st.status | ((port as u32) << 6);
                        if st.status & (1 << 23) != 0 {
                            packet = Some(st.packet(false));
                        }
                        st.status &= !((1 << 23) | 0x3f);
                        v as u64
                    }
                    xbow_reg::LOCK_CLEAR => st.reg[xbow_reg::LOCK / 8] as u64,
                    _ => st.reg[off / 8] as u64,
                }
            }
        };
        if access_error && packet.is_none() {
            let mut st = self.st.lock();
            if st.reg[xbow_reg::CONTROL / 8] & 32 != 0 && st.status & (1 << 23) == 0 {
                st.status |= 1 << 23;
                packet = Some(st.packet(true));
            }
        }
        if let Some(p) = packet {
            self.routes.interrupt(p);
        }
        // Reset outputs may call back into the chip, so release ordering too.
        drop(_order);
        if let Some(port) = reset {
            self.reset_port(port);
        }
        result
    }
}
struct CrossbarView {
    xbow: Arc<Crossbar>,
    port: u8,
}

macro_rules! bus_view {
    ($ty:ty, $read:expr) => {
        impl BusDevice for $ty {
            fn read8(&self, a: u64) -> BusRead8 {
                BusRead8::ok(($read)(self, a, 1, None) as u8)
            }
            fn read16(&self, a: u64) -> BusRead16 {
                BusRead16::ok(($read)(self, a, 2, None) as u16)
            }
            fn read32(&self, a: u64) -> BusRead32 {
                BusRead32::ok(($read)(self, a, 4, None) as u32)
            }
            fn read64(&self, a: u64) -> BusRead64 {
                BusRead64::ok(($read)(self, a, 8, None))
            }
            fn write8(&self, a: u64, v: u8) -> u32 {
                ($read)(self, a, 1, Some(v as u64));
                BUS_OK
            }
            fn write16(&self, a: u64, v: u16) -> u32 {
                ($read)(self, a, 2, Some(v as u64));
                BUS_OK
            }
            fn write32(&self, a: u64, v: u32) -> u32 {
                ($read)(self, a, 4, Some(v as u64));
                BUS_OK
            }
            fn write64(&self, a: u64, v: u64) -> u32 {
                ($read)(self, a, 8, Some(v));
                BUS_OK
            }
        }
    };
}
bus_view!(XBridgeHead, |s: &XBridgeHead, a, n, v| s.access(a, n, v));
bus_view!(Crossbar, |s: &Crossbar, a, n, v| s.access(
    HOST_WIDGET,
    a,
    n,
    v
));
bus_view!(CrossbarView, |s: &CrossbarView, a, n, v| s
    .xbow
    .access(s.port, a, n, v));

#[cfg(test)]
mod tests;
