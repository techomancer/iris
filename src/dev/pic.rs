//! The PIC, the PCI host bridge on the interface board of the Origin 350 and
//! Tezro (IP53 family). One chip sits in one XIO widget and carries two
//! independent bridge heads, at widget offset 0 and `0x800000`, each with its
//! own register block and PCI bus. Widget 15 holds the first PIC; the
//! two- and four-processor Tezro board adds a second at widget 11.
//!
//! A [`Pic`] is self-contained: the machine builds one per widget, hands the
//! processor's view of it to Bedrock IO ([`Pic::widget`]), gives it a
//! [`PicPeer`] for the packets it originates (interrupts and DMA), and
//! attaches PCI functions to the bus slots. Each head owns a shared [`PciBus`] (`src/pci/`) for its
//! functions, BAR decode and interrupt pins, and is that bus's [`PciHost`]:
//! it supplies the DMA a slot gets and takes the level of each of its eight
//! interrupt inputs. The bus slots are hooks only: IOC4 at device 1 (PIC
//! input 0), the ISP12160 at device 3 (input 2) and the BCM5701 at device 4
//! (input 3) are the board's choice, made by the machine; device `d`'s INTA
//! reaches input `d - 1`.
//!
//! Widget 0 of the board is the PXBOW of the PIC that carries IO9
//! ([`Pic::enable_pxbow`]); the other PIC has none.
//!
//! The register offsets are the ones the Bridge family shares ([`reg`]);
//! the PIC leaves out the ATE, the alias windows and the second and third
//! write-request banks' meaning.
//!
//! # What is approximated here
//!
//! The model is the compatibility profile QEMU selects for Tezro: revision 3
//! with identity bit 0 set, `DIR_MAP` bits 0 to 16 and 20 to 23, the control
//! and Device masks, and the reset values below are emulator choices that let
//! the PROM and IRIX run, not measured reset state. The CPU word view (a
//! 32-bit access to the low half of an eight-byte slot) and the processor's
//! byte-lane placement for PCI cycles are inferred from the public drivers.
//! Translated (ATE) DMA, PCI-X errors and the bridge's own PCI function are
//! unsupported and observable ([`Pic::unsupported`]). The packet face a peer
//! could read the heads through has no user and is not modeled. All
//! transfers are synchronous.
//!
//! An input's status bit is set while it is active, enabled or not, and a
//! head sends one set packet when an enabled input rises. Software asks for
//! another through the write-only force registers, which ignore the data:
//! force-always (0x180 + 8n) sends input n's packet whatever its level, and
//! force-with-pin (0x1c0 + 8n) only while it is active. The shipped drivers
//! write force-with-pin at the end of each interrupt dispatch, so an input a
//! handler leaves active is delivered again (established from the drivers);
//! the chip has no timer that sends a packet again, and neither has the
//! model. A packet the peer refuses is dropped and logged and the status
//! bit stays set; whether the chip retries one is unknown, and the model
//! leaves recovery to the next force write. Whether the interrupt enable
//! gates a forced packet is unknown; the model does not gate it.
//!
//! Direct64 DMA address bit 55 is the swap attribute and bit 19 of a Device
//! register the 32-bit direct window's (established bit names). A swap is
//! taken to exchange byte positions within each bus data phase, a 32-bit
//! word or, for a slot that drives REQ64#, a 64-bit doubleword (inferred
//! from the sibling XBridge). Each whole aligned unit reaches memory as one
//! write, so a word a device updates is never seen torn; a burst as a whole
//! is not atomic. A ragged end writes only its own bytes of the unit, as
//! one packet; the peer then applies them as naturally aligned pieces, so
//! an odd-sized end of three bytes is two writes (inferred; the chip's own
//! byte-enable handling is unknown).
//!
//! A configuration read that no function claims is a PCI master timeout:
//! the processor gets a bus error back through Bedrock, the error view
//! records bit 11 whatever the enable, and interrupt status records it only
//! while enable bit 11 is set (the gating is inferred). Status clears one
//! bit at a time; any reset-register write clears the view. Device 0, which
//! the PIC keeps for itself, reads as unclaimed: an approximation, since
//! what the chip answers there is unknown. Which bit a bus in PCI-X mode
//! sets (11 or 34) is unknown; the model sets 11. An unclaimed
//! configuration write goes nowhere and records nothing.
//!
//! Ported from hw/pci-host/sgi-pic.c in jamesbraid/qemu-sgi-origami.

use super::heart::Clock;
use super::pxbow::{AttachError, Pxbow};
use crate::devlog::LogModule;
use crate::irq::IrqLine;
use crate::pci::{
    BarDevice, BarMapping, BoardIrq, ByteOrder, CpuWindow, DevFn, IntxRouting, PciBus, PciDma, PciDmaStatus, PciFunction,
    PciHost, Space,
};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR, BUS_OK};
use parking_lot::{Mutex, RwLock};
use std::sync::{Arc, OnceLock, Weak};

/// Where the second head sits in the widget.
pub const HEAD_OFFSET: u64 = 0x80_0000;
/// The widget's address space.
pub const WIDGET_SIZE: u64 = 1 << 32;
/// Interrupt inputs per head.
pub const PINS: usize = 8;
/// PCI devices behind a head: configuration devices 1 to 4. Device 0 is the
/// bridge itself, and no PCI identity for it is established.
pub const DEVICES: u8 = 4;

const REVISION: u32 = 3;
const MANUFACTURER: u32 = 0x24;
const PARTS: [u32; 2] = [0xd102, 0xd112];

const REGISTER_BYTES: usize = 0x640;
const REQUEST_BANKS: usize = 3;
const REQUEST_ENTRIES: usize = 256;
const REQUEST_BASE: usize = 0x18000;
const REQUEST_END: usize = 0x19800;
const REQUEST_BANK_SIZE: usize = 0x800;
const CONFIG_BASE: u64 = 0x20000;
const CONFIG_TYPE1: u64 = 0x28000;
const CONFIG_END: u64 = 0x29000;
const PCI_WINDOW: [u64; 2] = [0x4000_0000, 0xc000_0000];
const PCI_WINDOW_SIZE: u64 = 1 << 30;
const SHORT_BASE: u64 = 0x20_0000;
/// The packet window of the 48-bit XIO address: bits 63:60 of a direct64 DMA
/// address name the destination widget, and a payload keeps 47 bits.
const PAYLOAD_BITS: u32 = 48;
const UNSUPPORTED_LIMIT: usize = 32;

/// The widget flush register (read to order earlier writes).
const FLUSH: usize = 0x50;
/// The link-side error address slot, whose low-word view the PROM writes.
const ERROR_ADDRESS_WORD: u64 = 0x7c;

const CONTROL_MASK: u64 = 0x0f00_f20f;
const CONTROL_RESET: u64 = 0x0f00_320f;
const REQUEST_TIMEOUT_RESET: u64 = 0xfffff;
const DIRECT_MAP_SUPPORTED: u64 = 0x1ffff | (0xf << 20);
const DEVICE_MASK: u64 = 0x1858_3fff;
const DEVICE_RESET: [u64; DEVICES as usize] = [0x1800_1000, 0x1800_1002, 0x1800_1004, 0x1800_1005];
const DEVICE_SWAP: u64 = 1 << 19;
const DEVICE_STREAM: u64 = 1 << 13;
const DEVICE_WIDE: u64 = 1 << 22;
const DEVICE_MEMORY: u64 = 1 << 12;
const INVALID_ADDRESS: u64 = 1 << 24;
/// PCI master timeout: a cycle no device claimed. The same bit in the
/// interrupt status, enable and error view registers.
const MASTER_TIMEOUT: u64 = 1 << 11;
/// The error view: the error bits, whether or not they are enabled.
/// Read-only; any write to the reset register clears it.
const ERROR_VIEW: usize = reg::INTERRUPT_MULTIPLE;
const SET: u64 = 0x100;
/// The force registers, one write-only slot per input: a write to
/// force-always sends the input's set packet whatever its level, and to
/// force-with-pin only while the input is active. The data is ignored.
const FORCE_ALWAYS: usize = 0x180;
const FORCE_PIN: usize = 0x1c0;
const FORCE_END: usize = 0x200;
const LEVELS: std::ops::RangeInclusive<u64> = 7..=127;

/// The head registers the PIC shares with the Bridge family, as byte offsets
/// into a head's register block.
pub mod reg {
    pub const IDENTIFICATION: usize = 0x0;
    pub const CONTROL: usize = 0x20;
    pub const REQUEST_TIMEOUT: usize = 0x28;
    pub const INT_DEST_UPPER: usize = 0x30;
    pub const INT_DEST_LOWER: usize = 0x38;
    pub const DIRECT_MAP: usize = 0x80;
    pub const PCI_TYPE1_CONFIG: usize = 0xc8;
    pub const INTERRUPT_STATUS: usize = 0x100;
    pub const INTERRUPT_ENABLE: usize = 0x108;
    pub const INTERRUPT_RESET: usize = 0x110;
    pub const INTERRUPT_MODE: usize = 0x118;
    pub const INTERRUPT_DEVICE: usize = 0x120;
    pub const INTERRUPT_ERROR_HOST: usize = 0x128;
    pub const INTERRUPT_HOST_BASE: usize = 0x130;
    pub const INTERRUPT_HOST_LAST: usize = 0x168;
    pub const INTERRUPT_MULTIPLE: usize = 0x170;
    pub const DEVICE_BASE: usize = 0x200;
}

/// A run of a head's address space that one BAR covers whole, and the bus
/// device that answers it.
struct PioRun {
    start: u64,
    end: u64,
    dev: Arc<dyn BusDevice>,
}

/// Whole BAR-page runs are few and carry no RAM or native fault-handler state.
/// Readers clone one endpoint and release the table lock before calling a device.
#[derive(Default)]
struct PioDecode {
    runs: RwLock<Vec<PioRun>>,
}
impl PioDecode {
    fn publish(&self, mut runs: Vec<PioRun>) -> Vec<PioRun> {
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

/// Widget 0's interrupt packets, which go to the peer as the PIC's do.
struct PxbowPackets(Arc<Wiring>);
impl crate::irq::InterruptController for PxbowPackets {
    fn set_line(&self, line: u32, level: bool) {
        self.packet(0, 0, line, level);
    }
    fn packet(&self, _source: u8, address: u64, line: u32, level: bool) -> bool {
        let payload = u64::from(line) | if level { SET } else { 0 };
        let taken = self.0.peer.get().is_some_and(|peer| peer.interrupt(0, address, payload).is_ok());
        if !taken {
            crate::dlog!(LogModule::Bedrock, "pic: PXBOW interrupt packet to {address:#x} refused");
        }
        taken
    }
}

/// The widget identity word of a head: revision, part and manufacturer
/// packed as the common widget header lays them out, with bit 0 set.
pub const fn identity(head: usize) -> u32 {
    (REVISION << 28) | (PARTS[head] << 12) | (MANUFACTURER << 1) | 1
}

/// A configuration cycle no device claimed: a PCI master timeout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Unclaimed;

/// What the machine's side of the XIO link refused: no target, a permission
/// check, or an address with nothing behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Refused;

/// The far end of the link: where the packets this chip originates go. On
/// Tezro that is Bedrock IO. Bytes are in address order, as the XIO packet
/// carries them.
pub trait PicPeer: Send + Sync {
    /// An interrupt packet: `payload` is the vector, with bit 8 for assert,
    /// written to the 48-bit destination `address`.
    fn interrupt(&self, source_widget: u8, address: u64, payload: u64) -> Result<(), Refused>;

    /// A DMA read of `data.len()` bytes at the 48-bit `address` of `widget`.
    fn dma_read(&self, source_widget: u8, widget: u8, address: u64, data: &mut [u8]) -> Result<(), Refused>;

    fn dma_write(&self, source_widget: u8, widget: u8, address: u64, data: &[u8]) -> Result<(), Refused>;
}

/// How the board uses a PCI function's slot.
#[derive(Clone, Copy, Debug, Default)]
pub struct Function {
    /// The function's registers keep the processor's byte sequence (IOC4),
    /// so the bridge leaves the address and value of a processor access
    /// alone instead of placing the bytes for a little-endian adapter.
    pub self_ordered_pio: bool,
    /// The slot drives REQ64# and the function takes 64-bit data phases:
    /// its swap-clear direct64 DMA reverses each doubleword, not each word.
    pub req64: bool,
}

/// The peer was already connected.
#[derive(Debug, PartialEq, Eq)]
pub struct AlreadyConnected;

/// Wiring set once before the machine runs.
struct Wiring {
    peer: OnceLock<Arc<dyn PicPeer>>,
}

struct State {
    reg: [u64; REGISTER_BYTES / 8],
    request: [[u64; REQUEST_ENTRIES]; REQUEST_BANKS],
    pins: u8,
    set_requests: [u32; PINS],
    clear_requests: [u32; PINS],
}

impl State {
    fn new() -> Self {
        let mut s = Self {
            reg: [0; REGISTER_BYTES / 8],
            request: [[0; REQUEST_ENTRIES]; REQUEST_BANKS],
            pins: 0,
            set_requests: [0; PINS],
            clear_requests: [0; PINS],
        };
        s.reset();
        s
    }

    fn r(&self, off: usize) -> u64 {
        self.reg[off / 8]
    }

    fn set(&mut self, off: usize, value: u64) {
        self.reg[off / 8] = value;
    }

    fn reset(&mut self) {
        *self = Self {
            reg: [0; REGISTER_BYTES / 8],
            request: [[0; REQUEST_ENTRIES]; REQUEST_BANKS],
            pins: 0,
            set_requests: [0; PINS],
            clear_requests: [0; PINS],
        };
        self.set(reg::REQUEST_TIMEOUT, REQUEST_TIMEOUT_RESET);
        self.set(reg::CONTROL, CONTROL_RESET);
        for (i, v) in DEVICE_RESET.iter().enumerate() {
            self.set(reg::DEVICE_BASE + i * 8, *v);
        }
    }
}

/// The head's PCI interrupt routing. Device `d`'s INTA reaches input `d - 1`;
/// INTB to INTD are not wired. The eight inputs are shared with the board's
/// own interrupts, which reach a head through [`Pic::pin_line`]. Device 0 is
/// the bridge itself and holds no function.
///
/// INTA of device `d` on input `d - 1` is what IRIX 6.5.30's pin rule
/// assigns (INTA and INTC to input `d - 1`, INTB and INTD to input
/// `(d - 1) ^ 4`), and an Origin 350's IO9 reports its IOC4, ISP12160 and
/// BCM5701 on inputs 0, 2 and 3 under OpenBSD: established as what software
/// expects. The board's own wiring is inferred from that, and INTB to INTD
/// are unknown; no function on the IO9 uses them.
static PIC_ROUTING: IntxRouting = IntxRouting {
    devices: DEVICES + 1,
    irqs: PINS as u8,
    intx: &[
        (1, [Some(0), None, None, None]),
        (2, [Some(1), None, None, None]),
        (3, [Some(2), None, None, None]),
        (4, [Some(3), None, None, None]),
    ],
    board_irqs: &[0, 1, 2, 3, 4, 5, 6, 7],
};

/// The head's CPU windows, by [`CpuWindow`] index: the four short device-I/O
/// windows, then the large PCI memory window. Addresses are the widget's.
const SHORT_WINDOWS: usize = 4;
const BIG_WINDOW: usize = 4;

/// The short windows' extents in the head's own space.
const SHORT: [(u64, u64); SHORT_WINDOWS] = [(0x20_0000, 0x20_0000), (0x40_0000, 0x20_0000), (0x60_0000, 0x10_0000), (0x70_0000, 0x10_0000)];

fn short_window(addr: u64) -> Option<(usize, u64, u64)> {
    match addr {
        0x20_0000..=0x3f_ffff => Some((0, addr - 0x20_0000, 0x20_0000)),
        0x40_0000..=0x5f_ffff => Some((1, addr - 0x40_0000, 0x20_0000)),
        0x60_0000..=0x6f_ffff => Some((2, addr - 0x60_0000, 0x10_0000)),
        0x70_0000..=0x7f_ffff => Some((3, addr - 0x70_0000, 0x10_0000)),
        _ => None,
    }
}

fn defined(off: usize) -> bool {
    matches!(
        off,
        reg::IDENTIFICATION
            | reg::CONTROL
            | reg::REQUEST_TIMEOUT
            | reg::INT_DEST_UPPER
            | reg::INT_DEST_LOWER
            | FLUSH
            | reg::DIRECT_MAP
            | reg::PCI_TYPE1_CONFIG
            | reg::INTERRUPT_STATUS
            | reg::INTERRUPT_ENABLE
            | reg::INTERRUPT_RESET
            | reg::INTERRUPT_MODE
            | reg::INTERRUPT_DEVICE
            | reg::INTERRUPT_ERROR_HOST
            | reg::INTERRUPT_HOST_BASE..=reg::INTERRUPT_HOST_LAST
            | ERROR_VIEW
            | reg::DEVICE_BASE..=0x218
    )
}

fn request_slot(off: usize) -> (usize, usize) {
    let rel = off - REQUEST_BASE;
    (rel / REQUEST_BANK_SIZE, (rel % REQUEST_BANK_SIZE) / 8)
}

/// The largest naturally aligned access of at most eight bytes that starts
/// at `address` and fits in `remaining`.
fn natural(address: u64, remaining: usize) -> usize {
    let mut n = 8;
    while n > remaining || !address.is_multiple_of(n as u64) {
        n /= 2;
    }
    n
}

fn le_bytes(size: usize, value: u64) -> [u8; 8] {
    let mut b = [0; 8];
    b[..size].copy_from_slice(&value.to_le_bytes()[..size]);
    b
}

fn be_bytes(size: usize, value: u64) -> [u8; 8] {
    let mut b = [0; 8];
    b[..size].copy_from_slice(&value.to_be_bytes()[8 - size..]);
    b
}

fn le_value(bytes: &[u8]) -> u64 {
    bytes.iter().rev().fold(0, |v, &b| (v << 8) | b as u64)
}

fn be_value(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0, |v, &b| (v << 8) | b as u64)
}

/// One bridge head: registers, one PCI bus, and its interrupt inputs.
// Lock order: `order`, then `st`. Packets go to the peer and the PIs with
// `st` dropped; `order` keeps two threads' packets in the order decided.
struct Head {
    index: usize,
    wiring: Arc<Wiring>,
    st: Mutex<State>,
    order: Mutex<()>,
    /// The head's PCI bus: its functions, BAR windows and interrupt pins.
    bus: PciBus,
    /// How the board uses each device's slot, by device number.
    slots: RwLock<[Option<Function>; DEVICES as usize + 1]>,
    /// Whole-page BAR mappings of the windows, which the processor reaches
    /// without a search of the bus.
    pio: PioDecode,
    unsupported: Mutex<Vec<(usize, bool, u64, usize)>>,
    /// This head, for the DMA a function is given: it must not keep the head alive.
    me: Weak<Head>,
}

struct SlotDma {
    head: Weak<Head>,
    device: u8,
}

impl PciDma for SlotDma {
    fn dma_read(&self, address: u64, buf: &mut [u8]) -> PciDmaStatus {
        match self.head.upgrade() {
            Some(head) => head.dma(self.device, address, buf, false),
            None => {
                buf.fill(0);
                PciDmaStatus::MasterAbort
            }
        }
    }

    fn dma_write(&self, address: u64, data: &[u8]) -> PciDmaStatus {
        match self.head.upgrade() {
            Some(head) => head.dma(self.device, address, &mut data.to_vec(), true),
            None => PciDmaStatus::MasterAbort,
        }
    }
}

/// What the head's bus asks of it: the DMA of a requester, and an input's
/// level, as the OR of every device line and board interrupt routed to it.
impl PciHost for Head {
    fn dma_for(&self, requester: u8) -> Arc<dyn PciDma> {
        Arc::new(SlotDma { head: self.me.clone(), device: requester })
    }

    fn set_irq(&self, pin: u8, level: bool) {
        self.pin(pin as usize, level);
    }
}

impl Head {
    fn new(index: usize, wiring: Arc<Wiring>) -> Arc<Self> {
        let this = Arc::new_cyclic(|me: &Weak<Head>| Self {
            index,
            wiring,
            st: Mutex::new(State::new()),
            order: Mutex::new(()),
            // iris's bus numbers count per bus tree; the heads are separate trees.
            bus: PciBus::new(index as u8 * 16, &PIC_ROUTING, me.clone()),
            slots: RwLock::new([None; DEVICES as usize + 1]),
            pio: PioDecode::default(),
            unsupported: Mutex::new(Vec::new()),
            me: me.clone(),
        });
        this.bus.set_host_windows(&|| this.windows());
        let head = Arc::downgrade(&this);
        this.bus.on_bar_mappings(Box::new(move |runs| {
            if let Some(head) = head.upgrade() {
                head.publish_mappings(runs);
            }
        }));
        this
    }

    fn note_unsupported(&self, addr: u64, size: usize, write: bool, value: u64) {
        let mut seen = self.unsupported.lock();
        if seen.len() >= UNSUPPORTED_LIMIT {
            return;
        }
        seen.push((self.index, write, addr, size));
        crate::dlog!(
            LogModule::Bedrock,
            "pic: head {} {} offset {addr:#x} size {size} value {value:#x} unsupported{}",
            self.index,
            if write { "write" } else { "read" },
            if seen.len() == UNSUPPORTED_LIMIT { "; further accesses suppressed" } else { "" }
        );
    }

    fn source_widget(&self) -> u8 {
        (self.st.lock().r(reg::CONTROL) & 0xf) as u8
    }

    // ---- interrupts ----

    /// Send pin `pin`'s set or clear packet. True when the peer took it.
    fn packet(&self, pin: usize, set: bool) -> bool {
        let (destination, source) = {
            let mut st = self.st.lock();
            if set {
                st.set_requests[pin] += 1;
            } else {
                st.clear_requests[pin] += 1;
            }
            (st.r(reg::INTERRUPT_HOST_BASE + pin * 8), (st.r(reg::CONTROL) & 0xf) as u8)
        };
        let level = destination >> 48;
        if !LEVELS.contains(&level) {
            crate::dlog!(LogModule::Bedrock, "pic: head {} pin {pin} rejected interrupt level {level:#x}", self.index);
            return false;
        }
        let address = destination & ((1 << PAYLOAD_BITS) - 1);
        let payload = level | if set { SET } else { 0 };
        let sent = self.send_interrupt(source, address, payload);
        if !sent {
            crate::dlog!(LogModule::Bedrock, "pic: head {} interrupt destination {destination:#x} refused", self.index);
        }
        sent
    }

    fn send_interrupt(&self, source: u8, address: u64, payload: u64) -> bool {
        match self.wiring.peer.get() {
            Some(peer) => peer.interrupt(source, address, payload).is_ok(),
            None => {
                crate::dlog!(LogModule::Bedrock, "pic: head {} interrupt has no peer", self.index);
                false
            }
        }
    }

    fn pin(&self, pin: usize, level: bool) {
        if pin >= PINS {
            return;
        }
        let _order = self.order.lock();
        let bit = 1u8 << pin;
        let (was_high, enabled, mode) = {
            let mut st = self.st.lock();
            let was_high = st.pins & bit != 0;
            st.pins = (st.pins & !bit) | if level { bit } else { 0 };
            if level && !was_high {
                let status = st.r(reg::INTERRUPT_STATUS) | bit as u64;
                st.set(reg::INTERRUPT_STATUS, status);
            }
            (was_high, st.r(reg::INTERRUPT_ENABLE) & bit as u64 != 0, st.r(reg::INTERRUPT_MODE) & bit as u64 != 0)
        };
        if level && !was_high && enabled {
            self.packet(pin, true);
        }
        if was_high && !level && mode {
            self.packet(pin, false);
        }
    }

    /// The PROM's write to the link-side error address slot latches the
    /// invalid-address condition and, when enabled, sends one packet.
    fn invalid_address(&self) {
        let _order = self.order.lock();
        let enabled = {
            let mut st = self.st.lock();
            let status = st.r(reg::INTERRUPT_STATUS) | INVALID_ADDRESS;
            st.set(reg::INTERRUPT_STATUS, status);
            st.r(reg::INTERRUPT_ENABLE) & INVALID_ADDRESS != 0
        };
        if enabled {
            self.error_interrupt();
        }
    }

    /// An unclaimed configuration read: the error view always records the
    /// master timeout, and interrupt status only while it is enabled, when
    /// it also sends the error interrupt.
    fn master_timeout(&self) {
        let _order = self.order.lock();
        let enabled = {
            let mut st = self.st.lock();
            let view = st.r(ERROR_VIEW) | MASTER_TIMEOUT;
            st.set(ERROR_VIEW, view);
            let enabled = st.r(reg::INTERRUPT_ENABLE) & MASTER_TIMEOUT != 0;
            if enabled {
                let status = st.r(reg::INTERRUPT_STATUS) | MASTER_TIMEOUT;
                st.set(reg::INTERRUPT_STATUS, status);
            }
            enabled
        };
        if enabled {
            self.error_interrupt();
        }
    }

    /// One set packet for the error interrupt, to the widget destination.
    /// With `order` held.
    fn error_interrupt(&self) {
        let number = self.st.lock().r(reg::INTERRUPT_ERROR_HOST);
        if number > 0x7f {
            crate::dlog!(LogModule::Bedrock, "pic: head {} refused error interrupt number {number:#x}", self.index);
            return;
        }
        let Some(address) = self.error_destination() else { return };
        let source = self.source_widget();
        if !self.send_interrupt(source, address, number | SET) {
            crate::dlog!(LogModule::Bedrock, "pic: head {} error destination {address:#x} refused", self.index);
        }
    }

    /// The error interrupt's address from the widget destination pair: a
    /// lower register that uses bits 47:32 is the address (the upper half
    /// may repeat them), otherwise the pair joins into 48 bits.
    fn error_destination(&self) -> Option<u64> {
        let (upper, lower) = {
            let st = self.st.lock();
            (st.r(reg::INT_DEST_UPPER), st.r(reg::INT_DEST_LOWER))
        };
        let upper_address = upper & 0xffff;
        let lower_high = lower >> 32;
        let refuse = |why: &str| {
            crate::dlog!(LogModule::Bedrock, "pic: head {} error destination {why} (upper {upper:#x}, lower {lower:#x})", self.index);
            None
        };
        if upper >> 32 != 0 {
            return refuse("has unsupported high bits");
        }
        if lower >> 48 != 0 {
            return refuse("exceeds 48 bits");
        }
        if upper_address != 0 && lower_high != 0 && upper_address != lower_high {
            return refuse("has conflicting address fields");
        }
        let address = if lower_high != 0 { lower } else { (upper_address << 32) | (lower & 0xffff_ffff) };
        if address == 0 {
            return refuse("is zero");
        }
        Some(address)
    }

    // ---- registers ----

    fn identity(&self) -> u64 {
        identity(self.index) as u64
    }

    /// The widget address of head address `addr`.
    fn widget_address(&self, addr: u64) -> u64 {
        self.index as u64 * HEAD_OFFSET + addr
    }

    /// `None` is a read the processor gets back as a bus error: only an
    /// unclaimed configuration read fails that way.
    fn read(&self, addr: u64, size: usize) -> Option<u64> {
        if short_window(addr).is_some() {
            return Some(self.short(self.widget_address(addr), size, None).unwrap_or(0));
        }
        if (CONFIG_BASE..CONFIG_END).contains(&addr) && size <= 4 {
            return self.config(addr, size, None).ok();
        }
        let off = addr as usize;
        if size == 4 && off % 8 == 4 {
            return Some(self.word_read(off));
        }
        if size == 8 && off == 0 {
            return Some(self.identity());
        }
        if size == 8 && off % 8 == 0 && defined(off) {
            return Some(self.st.lock().r(off));
        }
        if size == 8 && off % 8 == 0 && (REQUEST_BASE..REQUEST_END).contains(&off) {
            let (bank, index) = request_slot(off);
            return Some(self.st.lock().request[bank][index]);
        }
        self.note_unsupported(addr, size, false, 0);
        Some(0)
    }

    fn write(&self, addr: u64, size: usize, value: u64) {
        if short_window(addr).is_some() {
            self.short(self.widget_address(addr), size, Some(value));
            return;
        }
        if addr == ERROR_ADDRESS_WORD && size == 4 {
            self.invalid_address();
            return;
        }
        if (CONFIG_BASE..CONFIG_END).contains(&addr) && size <= 4 {
            let _ = self.config(addr, size, Some(value));
            return;
        }
        let off = addr as usize;
        if (FORCE_ALWAYS..FORCE_END).contains(&off) {
            self.force_write(off, size, value);
            return;
        }
        if size == 4 && off % 8 == 4 {
            self.word_write(off, value as u32);
            return;
        }
        self.register_write(off, size, value);
    }

    /// A write to a force register: a doubleword to its slot or a word to
    /// the slot's low half, as for the other registers. Other widths and
    /// every read are unsupported and observable.
    fn force_write(&self, off: usize, size: usize, value: u64) {
        let slot = off & !7;
        if !matches!((size, off - slot), (8, 0) | (4, 4)) {
            self.note_unsupported(off as u64, size, true, value);
            return;
        }
        let pin = (slot - FORCE_ALWAYS) / 8 % PINS;
        self.force(pin, slot < FORCE_PIN);
    }

    /// Send input `pin`'s set packet again on software's request: always,
    /// or only while the input is active. Software writes force-with-pin
    /// at the end of each interrupt dispatch, so an input that is still
    /// active after its handler ran is delivered again; the chip itself
    /// sends a packet only when an input rises. The status bit follows the
    /// input and is not changed. Whether the interrupt enable gates a
    /// forced packet is unknown; the model does not gate it.
    fn force(&self, pin: usize, always: bool) {
        let _order = self.order.lock();
        let active = self.st.lock().pins & (1 << pin) != 0;
        if always || active {
            self.packet(pin, true);
        }
    }

    /// A 32-bit access to the low half of an eight-byte slot.
    fn word_slot(off: usize) -> Option<usize> {
        let slot = off.checked_sub(4).filter(|slot| slot % 8 == 0)?;
        ((REQUEST_BASE..REQUEST_END).contains(&slot) || (defined(slot) && slot < REGISTER_BYTES)).then_some(slot)
    }

    fn word_read(&self, off: usize) -> u64 {
        match Self::word_slot(off) {
            Some(slot) if slot != FLUSH => {
                if slot == reg::IDENTIFICATION {
                    return self.identity();
                }
                let st = self.st.lock();
                let full = if slot >= REQUEST_BASE {
                    let (bank, index) = request_slot(slot);
                    st.request[bank][index]
                } else {
                    st.r(slot)
                };
                full as u32 as u64
            }
            _ => {
                self.note_unsupported(off as u64, 4, false, 0);
                0
            }
        }
    }

    fn word_write(&self, off: usize, value: u32) {
        let slot = match Self::word_slot(off) {
            Some(slot) if !matches!(slot, reg::IDENTIFICATION | FLUSH | reg::INTERRUPT_STATUS | reg::INTERRUPT_DEVICE | ERROR_VIEW) => slot,
            _ => {
                self.note_unsupported(off as u64, 4, true, value as u64);
                return;
            }
        };
        if slot >= REQUEST_BASE {
            let (bank, index) = request_slot(slot);
            let entry = &mut self.st.lock().request[bank][index];
            *entry = (*entry & 0xffff_ffff_0000_0000) | value as u64;
        } else if slot == reg::INTERRUPT_RESET {
            self.register_write(slot, 8, value as u64);
        } else {
            let merged = (self.st.lock().r(slot) & 0xffff_ffff_0000_0000) | value as u64;
            self.register_write(slot, 8, merged);
        }
    }

    fn register_write(&self, off: usize, size: usize, value: u64) {
        if size != 8 || off % 8 != 0 {
            self.note_unsupported(off as u64, size, true, value);
            return;
        }
        if (REQUEST_BASE..REQUEST_END).contains(&off) {
            let (bank, index) = request_slot(off);
            self.st.lock().request[bank][index] = value;
            return;
        }
        if !defined(off) || off >= REGISTER_BYTES {
            self.note_unsupported(off as u64, size, true, value);
            return;
        }
        match off {
            reg::IDENTIFICATION | FLUSH | reg::INTERRUPT_STATUS | ERROR_VIEW => {
                self.note_unsupported(off as u64, size, true, value)
            }
            reg::CONTROL => {
                if value & !CONTROL_MASK != 0 {
                    self.note_unsupported(off as u64, size, true, value);
                }
                self.st.lock().set(off, value & CONTROL_MASK);
            }
            reg::DIRECT_MAP => {
                if value & !DIRECT_MAP_SUPPORTED != 0 {
                    self.note_unsupported(off as u64, size, true, value);
                }
                self.st.lock().set(off, value & DIRECT_MAP_SUPPORTED);
            }
            reg::INTERRUPT_ENABLE => {
                let _order = self.order.lock();
                let (newly, pins, status) = {
                    let mut st = self.st.lock();
                    let newly = value & !st.r(off);
                    st.set(off, value);
                    (newly, st.pins as u64, st.r(reg::INTERRUPT_STATUS))
                };
                for pin in 0..PINS {
                    if newly & status & pins & (1 << pin) != 0 {
                        self.packet(pin, true);
                    }
                }
            }
            reg::INTERRUPT_RESET => {
                let _order = self.order.lock();
                let (pins, enable) = {
                    let mut st = self.st.lock();
                    // Each error bit clears on its own one; any write
                    // clears the view.
                    let status = (st.r(reg::INTERRUPT_STATUS) & !value) | st.pins as u64;
                    st.set(reg::INTERRUPT_STATUS, status);
                    st.set(ERROR_VIEW, 0);
                    (st.pins as u64, st.r(reg::INTERRUPT_ENABLE))
                };
                for pin in 0..PINS {
                    if value & pins & enable & (1 << pin) != 0 {
                        self.packet(pin, true);
                    }
                }
            }
            reg::INTERRUPT_DEVICE => {
                // The device-to-pin map has no effect here; the board wires
                // each device to its pin.
                self.note_unsupported(off as u64, size, true, value);
                self.st.lock().set(off, value);
            }
            _ => {
                let device = (reg::DEVICE_BASE..=0x218).contains(&off);
                let stored = if device {
                    if value & !DEVICE_MASK != 0 {
                        self.note_unsupported(off as u64, size, true, value);
                    }
                    value & DEVICE_MASK
                } else {
                    value
                };
                self.st.lock().set(off, stored);
                if device {
                    self.bus.set_host_windows(&|| self.windows());
                }
            }
        }
    }

    // ---- PCI configuration ----

    /// A processor configuration cycle. Bus 0 devices sit at `0x1000`
    /// strides from `0x20000`; the type-1 window addresses the bus and
    /// device `PCI_TYPE1_CONFIG` names. A read no function claims, device 0
    /// (the bridge's own) and every function but 0 included, is a master
    /// timeout and fails; a write nobody claims goes nowhere.
    fn config(&self, addr: u64, size: usize, value: Option<u64>) -> Result<u64, Unclaimed> {
        let mask = u32::MAX >> (32 - size * 8);
        let function = ((addr >> 8) & 7) as u8;
        let (bus, device) = if addr < CONFIG_TYPE1 {
            (Some(self.bus.clone()), ((addr - CONFIG_BASE) >> 12) as u8)
        } else {
            let t1 = self.st.lock().r(reg::PCI_TYPE1_CONFIG);
            (self.bus.find_bus(((t1 >> 16) & 0xff) as u8), ((t1 >> 11) & 0x1f) as u8)
        };
        let register = ((addr & 0xff) as usize) ^ (4 - size);
        let at = DevFn { device, function };
        let answered = match value {
            Some(value) => bus.is_some_and(|bus| bus.config_write(at, register, size, value as u32 & mask)),
            None => {
                let read = bus.and_then(|bus| bus.config_read(at, register, size));
                if let Some(read) = read {
                    return Ok(read as u64);
                }
                false
            }
        };
        if !answered {
            crate::dlog!(
                LogModule::Bedrock,
                "pic: head {} configuration {} miss at {addr:#x}",
                self.index,
                if value.is_some() { "write" } else { "read" }
            );
        }
        if value.is_some() {
            return Ok(0);
        }
        self.master_timeout();
        Err(Unclaimed)
    }

    // ---- PCI memory and I/O cycles ----

    /// The CPU windows, from the Device registers; see [`SHORT_WINDOWS`]. A
    /// short window whose Device register asks for a byte stream or a wide
    /// device is one the model refuses, and has no run.
    fn windows(&self) -> Vec<CpuWindow> {
        let st = self.st.lock();
        let base = self.index as u64 * HEAD_OFFSET;
        (0..=SHORT_WINDOWS)
            .map(|kind| {
                if kind < SHORT_WINDOWS {
                    let device = st.r(reg::DEVICE_BASE + kind * 8);
                    let (space, pci) = Self::device(device, kind);
                    let (start, len) = SHORT[kind];
                    let len = if device & (DEVICE_STREAM | DEVICE_WIDE) != 0 { 0 } else { len };
                    CpuWindow { space, cpu: base + start, pci, len, mask: u64::MAX }
                } else {
                    CpuWindow { space: Space::Mem, cpu: PCI_WINDOW[self.index], pci: 0, len: PCI_WINDOW_SIZE, mask: u64::MAX }
                }
            })
            .collect()
    }

    /// Device register `device` of short window `kind`: the space it opens
    /// onto and the PCI address of the window's first byte.
    fn device(device: u64, kind: usize) -> (Space, u32) {
        let memory = device & DEVICE_MEMORY != 0;
        let mut base = (device & if memory { 0x3ff } else { 0xfff }) as u32;
        if kind < 2 {
            base &= !1;
        }
        (if memory { Space::Mem } else { Space::Io }, base << 20)
    }

    /// The byte order of a processor access reaching `f`. A conventional
    /// adapter sees a numeric value at the address with its low two bits
    /// corrected (MACE's rule); a self-ordered function (IOC4) keeps the
    /// processor's byte sequence at the address as it is. The order of a
    /// 64-bit access to a conventional adapter is unknown.
    fn order(&self, f: &dyn PciFunction) -> ByteOrder {
        let same = |g: &Arc<dyn PciFunction>| std::ptr::addr_eq(Arc::as_ptr(g), f as *const dyn PciFunction);
        let device = (1..=DEVICES).find(|&device| self.bus.function(DevFn { device, function: 0 }).is_some_and(same));
        let native = device.is_some_and(|device| self.slots.read()[device as usize].is_some_and(|slot| slot.self_ordered_pio));
        if native { ByteOrder::Stream } else { ByteOrder::values(4) }
    }

    /// The bus reports the BAR mappings the head's windows hold: serve them
    /// from the head's own table.
    fn publish_mappings(&self, runs: Vec<BarMapping>) {
        let runs = runs
            .iter()
            .map(|r| {
                let base = r.cpu.wrapping_sub(r.off as u64);
                PioRun {
                    start: r.cpu,
                    end: r.cpu + r.len,
                    dev: Arc::new(BarDevice::new(r.function.clone(), r.bar, base, u64::MAX, self.order(&*r.function))),
                }
            })
            .collect();
        // Replaced runs may run shutdown callbacks from their destructor.
        drop(self.pio.publish(runs));
    }

    /// A processor access to the widget address `addr` of a short window or
    /// the PCI memory window: a whole BAR page answers it from the head's
    /// table, anything else through the bus's decode. Nothing claiming it
    /// reads zero and drops a write; the caller says where that is a
    /// refusal worth logging.
    fn pci_access(&self, addr: u64, size: usize, value: Option<u64>) -> Option<u64> {
        let crossing = addr & 0xfff > 0x1000 - size as u64;
        if !crossing {
            if let Some(endpoint) = self.pio.target(addr, size) {
                return Some(match value {
                    Some(v) => {
                        match size {
                            1 => endpoint.write8(addr, v as u8),
                            2 => endpoint.write16(addr, v as u16),
                            4 => endpoint.write32(addr, v as u32),
                            _ => endpoint.write64(addr, v),
                        };
                        0
                    }
                    None => match size {
                        1 => endpoint.read8(addr).data as u64,
                        2 => endpoint.read16(addr).data as u64,
                        4 => endpoint.read32(addr).data as u64,
                        _ => endpoint.read64(addr).data,
                    },
                });
            }
        }
        let (space, pci) = self.pci_address(addr)?;
        let (function, bar, offset) = self.bus.decode(space, pci)?;
        let order = self.order(function);
        Some(match value {
            None => order.read(function, bar, offset, size),
            Some(v) => {
                order.write(function, bar, offset, size, v);
                0
            }
        })
    }

    /// The space and PCI address the widget address `addr` reaches.
    fn pci_address(&self, addr: u64) -> Option<(Space, u32)> {
        let rel = addr.checked_sub(self.index as u64 * HEAD_OFFSET);
        if let Some((window, offset, _)) = rel.and_then(short_window) {
            let (space, base) = Self::device(self.st.lock().r(reg::DEVICE_BASE + window * 8), window);
            return Some((space, base.wrapping_add(offset as u32)));
        }
        let offset = addr.checked_sub(PCI_WINDOW[self.index])?;
        (offset < PCI_WINDOW_SIZE).then_some((Space::Mem, offset as u32))
    }

    /// The short device-I/O windows: the Device register of the window
    /// picks PCI memory or I/O space and the megabyte of it.
    fn short(&self, addr: u64, size: usize, value: Option<u64>) -> Option<u64> {
        let write = value.is_some();
        let (window, offset, window_size) = short_window(addr % HEAD_OFFSET)?;
        if size as u64 > window_size - offset {
            crate::dlog!(LogModule::Bedrock, "pic: head {} short window {addr:#x} rejects boundary-crossing {size}-byte access", self.index);
            return None;
        }
        let device = self.st.lock().r(reg::DEVICE_BASE + window * 8);
        let refused = device & (DEVICE_STREAM | DEVICE_WIDE);
        if refused != 0 {
            crate::dlog!(
                LogModule::Bedrock,
                "pic: head {} short window {addr:#x} refuses unsupported Device attributes {refused:#x}",
                self.index
            );
            return None;
        }
        let result = self.pci_access(addr, size, value);
        if result.is_none() {
            crate::dlog!(
                LogModule::Bedrock,
                "pic: head {} short window {addr:#x} PCI {} failed for {size} bytes",
                self.index,
                if write { "write" } else { "read" }
            );
        }
        result
    }

    /// The large PCI memory window of this head.
    fn pci_window(&self, addr: u64, size: usize, value: Option<u64>) -> u64 {
        let result = self.pci_access(addr, size, value);
        if result.is_none() {
            self.note_unsupported(addr, size, value.is_some(), value.unwrap_or(0));
        }
        result.unwrap_or(0)
    }

    // ---- DMA ----

    fn dma(&self, device: u8, address: u64, bytes: &mut [u8], write: bool) -> PciDmaStatus {
        if !(1..=DEVICES).contains(&device) {
            crate::dlog!(LogModule::Bedrock, "pic: head {} unsupported requester device {device}", self.index);
            if !write {
                bytes.fill(0);
            }
            return PciDmaStatus::MasterAbort;
        }
        let mut done = 0;
        while done < bytes.len() {
            let at = address.wrapping_add(done as u64);
            let n = (0x1000 - (at as usize & 0xfff)).min(bytes.len() - done);
            let ok = address.checked_add(done as u64).is_some() && self.dma_page(device, at, &mut bytes[done..done + n], write);
            if !ok {
                if !write {
                    bytes[done..].fill(0);
                }
                return PciDmaStatus::MasterAbort;
            }
            done += n;
        }
        PciDmaStatus::Ok
    }

    fn dma_page(&self, device: u8, at: u64, chunk: &mut [u8], write: bool) -> bool {
        if at > u32::MAX as u64 {
            return self.direct64(device, at, chunk, write);
        }
        if at < 0x8000_0000 {
            crate::dlog!(LogModule::Bedrock, "pic: head {} unsupported DMA {at:#x} from device {device}", self.index);
            return false;
        }
        self.direct32(device, at, chunk, write)
    }

    fn send_dma(&self, source: u8, widget: u8, address: u64, data: &mut [u8], write: bool) -> bool {
        let Some(peer) = self.wiring.peer.get() else {
            crate::dlog!(LogModule::Bedrock, "pic: head {} DMA has no peer", self.index);
            return false;
        };
        if write {
            peer.dma_write(source, widget, address, data).is_ok()
        } else {
            peer.dma_read(source, widget, address, data).is_ok()
        }
    }

    /// The direct map: PCI `0x80000000` and up reach the widget `DIR_MAP`
    /// names, at the address its high bits and the PCI offset make.
    fn direct32(&self, device: u8, at: u64, chunk: &mut [u8], write: bool) -> bool {
        let (direct, attributes, source) = {
            let st = self.st.lock();
            (st.r(reg::DIRECT_MAP), st.r(reg::DEVICE_BASE + (device as usize - 1) * 8), (st.r(reg::CONTROL) & 0xf) as u8)
        };
        let widget = ((direct >> 20) & 0xf) as u8;
        let target = ((direct & 0x1ffff) << 31) | (at & 0x7fff_ffff);
        if attributes & DEVICE_SWAP != 0 {
            // A swapped slot puts each byte at its address with the low two
            // bits flipped.
            return self.send_flipped(source, widget, target, chunk, write, 3, |_, _| true);
        }
        let mut done = 0;
        while done < chunk.len() {
            let n = natural(target + done as u64, chunk.len() - done);
            if !self.send_dma(source, widget, target + done as u64, &mut chunk[done..done + n], write) {
                return false;
            }
            done += n;
        }
        true
    }

    /// Why a direct64 request cannot be carried, if it cannot. It is
    /// checked in full before any packet leaves.
    fn direct64_refusal(address: u64, size: usize) -> Option<&'static str> {
        let payload = address & ((1 << PAYLOAD_BITS) - 1);
        if address >> 60 == 0 {
            Some("target-zero")
        } else if address & ((1 << 54) | (0x3f << 48)) != 0 {
            Some("reserved-bits")
        } else if payload & (1 << 47) != 0 {
            Some("register-space")
        } else if size as u64 > (1 << 47) - payload {
            Some("range")
        } else {
            None
        }
    }

    /// A PCI address above 32 bits carries the target widget in bits 63:60
    /// and the XIO payload below. Bit 55 is the swap attribute: set, the
    /// bytes go as an ordered byte stream; clear, they keep 32-bit numeric
    /// values (the byte address low bits flip by 3, or by 7 for a REQ64
    /// slot's 64-bit requester). Bits 56 to 59 (barrier, virtual channel,
    /// precise, prefetch) change nothing here.
    fn direct64(&self, device: u8, at: u64, chunk: &mut [u8], write: bool) -> bool {
        let wide = self.slots.read()[device as usize].is_some_and(|slot| slot.req64);
        let source = self.source_widget();
        let check = |address: u64, size: usize| match Self::direct64_refusal(address, size) {
            Some(why) => {
                crate::dlog!(LogModule::Bedrock, "pic: head {} direct64 {why} address {address:#018x} from device {device}", self.index);
                false
            }
            None => true,
        };
        if !check(at, 1) {
            return false;
        }
        let widget = (at >> 60) as u8;
        let payload = at & ((1 << PAYLOAD_BITS) - 1);
        if at & (1 << 55) == 0 {
            let mask = if wide { 7 } else { 3 };
            return self.send_flipped(source, widget, payload, chunk, write, mask, |offset, n| check(at + offset, n));
        }
        let mut done = 0;
        while done < chunk.len() {
            // Packets end on doubleword boundaries, so an aligned field
            // never straddles two of them.
            let n = (8 - ((payload + done as u64) & 7) as usize).min(chunk.len() - done);
            if !check(at + done as u64, n) || !self.send_dma(source, widget, payload + done as u64, &mut chunk[done..done + n], write) {
                return false;
            }
            done += n;
        }
        true
    }

    /// Carry a transfer whose bytes land at their addresses with the low
    /// bits in `mask` flipped, which reverses the bytes of each aligned
    /// group of `mask + 1`, the data phase. The flip moves bytes within a
    /// group, not the transfer apart: a run of whole groups goes as one
    /// packet, which the peer applies one aligned group at a time, so a
    /// processor reading a word the device is updating sees all of it old
    /// or all of it new; the run as a whole is not atomic, as a burst on
    /// the bus is not. A group the transfer covers in part writes only
    /// those bytes, still contiguous once flipped, as one packet.
    /// `fits` vets each packet by its offset in `chunk` and its length
    /// before it leaves.
    #[allow(clippy::too_many_arguments)]
    fn send_flipped(
        &self,
        source: u8,
        widget: u8,
        address: u64,
        chunk: &mut [u8],
        write: bool,
        mask: u64,
        fits: impl Fn(u64, usize) -> bool,
    ) -> bool {
        let group = mask as usize + 1;
        let mut done = 0;
        while done < chunk.len() {
            let here = address + done as u64;
            let whole = if here & mask == 0 { (chunk.len() - done) / group * group } else { 0 };
            // A partial group's bytes reverse into the bytes below its
            // flipped last address.
            let (n, at) = if whole == 0 {
                let n = (group - (here & mask) as usize).min(chunk.len() - done);
                (n, (here + n as u64 - 1) ^ mask)
            } else {
                (whole, here)
            };
            let part = &mut chunk[done..done + n];
            let mut flipped = part.to_vec();
            let n_group = if whole == 0 { n } else { group };
            if write {
                flipped.chunks_mut(n_group).for_each(<[u8]>::reverse);
            }
            if !fits(done as u64, n) || !self.send_dma(source, widget, at, &mut flipped, write) {
                return false;
            }
            if !write {
                flipped.chunks_mut(n_group).for_each(<[u8]>::reverse);
                part.copy_from_slice(&flipped);
            }
            done += n;
        }
        true
    }

    fn attach(&self, device: u8, function: Arc<dyn PciFunction>, options: Function) {
        assert!((1..=DEVICES).contains(&device), "PIC devices are 1 to {DEVICES}");
        self.slots.write()[device as usize] = Some(options);
        self.bus.attach(DevFn { device, function: 0 }, function);
    }

    fn reset(&self) {
        {
            let _order = self.order.lock();
            self.st.lock().reset();
        }
        // RST#: headers, then functions, which drop their own lines. The
        // Device registers are back at reset, so the windows move too.
        self.bus.reset();
        self.bus.set_host_windows(&|| self.windows());
        self.unsupported.lock().clear();
    }
}

/// The PCI host bridge in one XIO widget.
pub struct Pic {
    widget: u8,
    wiring: Arc<Wiring>,
    heads: [Arc<Head>; 2],
    pxbow: OnceLock<Arc<Pxbow>>,
}

impl Pic {
    /// A PIC that answers as XIO widget `widget`.
    pub fn new(widget: u8) -> Arc<Self> {
        let wiring = Arc::new(Wiring {
            peer: OnceLock::new(),
        });
        Arc::new(Self {
            widget,
            heads: [Head::new(0, wiring.clone()), Head::new(1, wiring.clone())],
            wiring,
            pxbow: OnceLock::new(),
        })
    }

    pub fn widget_number(&self) -> u8 {
        self.widget
    }

    /// The widget as the processor reaches it: both heads, the short
    /// device-I/O windows and the PCI memory windows. Bedrock IO takes this.
    pub fn widget(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(CpuWidget { heads: self.heads.clone() })
    }

    /// Give the chip its PXBOW, the widget-0 face of the board that owns
    /// it, reached from `host_widget`. On Tezro the PIC that carries IO9
    /// supplies widget 0; a second PIC gets none. The PIC's own widget is
    /// attached behind it, and widget 0's interrupt goes to the peer when it
    /// targets the host. Once, before the machine runs.
    pub fn enable_pxbow(self: &Arc<Self>, host_widget: u8, clock: Clock) -> Result<Arc<Pxbow>, AttachError> {
        let pxbow = Pxbow::new(host_widget, self.widget, clock)?;
        pxbow.attach_endpoint(self.widget, self.widget())?;
        pxbow.connect_host(Arc::new(PxbowPackets(self.wiring.clone())));
        self.pxbow.set(pxbow.clone()).map_err(|_| AttachError::Occupied(self.widget))?;
        Ok(pxbow)
    }

    /// The chip's PXBOW, if it has one.
    pub fn pxbow(&self) -> Option<&Arc<Pxbow>> {
        self.pxbow.get()
    }

    /// Where interrupts and DMA this chip originates go. Once, before the
    /// machine runs.
    pub fn connect_peer(&self, peer: Arc<dyn PicPeer>) -> Result<(), AlreadyConnected> {
        self.wiring.peer.set(peer).map_err(|_| AlreadyConnected)
    }

    /// Put a PCI function at configuration `device` (1 to 4) of `head`'s
    /// bus. It gets its DMA path and, if its header names an interrupt pin,
    /// the line of input `device - 1`.
    pub fn attach(&self, head: usize, device: u8, function: Arc<dyn PciFunction>, options: Function) {
        self.heads[head].attach(device, function, options);
    }

    /// The PCI bus of `head`.
    pub fn bus(&self, head: usize) -> &PciBus {
        &self.heads[head].bus
    }

    /// Interrupt input `pin` (0 to 7) of `head`, for a device whose
    /// interrupt the board wires to it directly.
    pub fn pin_line(&self, head: usize, pin: usize) -> IrqLine {
        assert!(pin < PINS);
        self.heads[head].bus.board_irq(BoardIrq(pin as u8))
    }

    /// Set interrupt input `pin` of `head`.
    pub fn set_pin(&self, head: usize, pin: usize, level: bool) {
        self.pin_line(head, pin).set(level);
    }

    /// The set and clear packets `head` has requested for `pin`, delivered
    /// or not.
    pub fn request_counts(&self, head: usize, pin: usize) -> (u32, u32) {
        let st = self.heads[head].st.lock();
        (st.set_requests[pin], st.clear_requests[pin])
    }

    /// Accesses the model refused or could not carry out, as (head, write,
    /// offset, size); the first 32 per head.
    pub fn unsupported(&self) -> Vec<(usize, bool, u64, usize)> {
        self.heads.iter().flat_map(|head| head.unsupported.lock().clone()).collect()
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        for head in &self.heads {
            for (what, probe) in [
                ("state", (|h: &Head| h.st.is_locked()) as fn(&Head) -> bool),
                ("irq_order", |h| h.order.is_locked()),
            ] {
                let h = Arc::downgrade(head);
                register_lock_fn(format!("pic{:x}::head{}::{what}", self.widget, head.index), move || {
                    h.upgrade().is_some_and(|h| probe(&h))
                });
            }
        }
        if let Some(pxbow) = self.pxbow.get() {
            pxbow.xbow().register_locks();
        }
    }
}

impl Resettable for Pic {
    fn power_on(&self) {
        if let Some(pxbow) = self.pxbow.get() {
            pxbow.xbow().power_on();
        }
        for head in &self.heads {
            head.reset();
        }
    }
}

/// The processor's view of the widget.
struct CpuWidget {
    heads: [Arc<Head>; 2],
}

impl CpuWidget {
    /// `None` is a bus error: a malformed access or an unclaimed
    /// configuration read. Accesses the PIC cannot make sense of that are
    /// not malformed (unsupported registers, refused windows) read zero and
    /// write nowhere without one.
    fn access(&self, addr: u64, size: usize, value: Option<u64>) -> Option<u64> {
        let write = value.is_some();
        let short = addr < 2 * HEAD_OFFSET && short_window(addr % HEAD_OFFSET).is_some();
        if !short && !addr.is_multiple_of(size as u64) {
            return None;
        }
        if addr >= 2 * HEAD_OFFSET {
            if addr < 2 * HEAD_OFFSET + 8 {
                crate::dlog!(LogModule::Bedrock, "pic: short-window guard {} at {addr:#x} refused for {size} bytes", if write { "write" } else { "read" });
                return Some(0);
            }
            let window = PCI_WINDOW.iter().position(|&base| (base..base + PCI_WINDOW_SIZE).contains(&addr))?;
            return Some(self.heads[window].pci_window(addr, size, value));
        }
        let head = &self.heads[(addr / HEAD_OFFSET) as usize];
        let rel = addr % HEAD_OFFSET;
        match value {
            Some(v) => {
                head.write(rel, size, v);
                Some(0)
            }
            None => head.read(rel, size),
        }
    }
}

macro_rules! cpu_access {
    ($r:ident, $w:ident, $t:ty, $out:ident, $n:expr) => {
        fn $r(&self, addr: u64) -> $out {
            self.access(addr, $n, None).map_or_else($out::err, |v| $out::ok(v as $t))
        }
        fn $w(&self, addr: u64, value: $t) -> u32 {
            if self.access(addr, $n, Some(value as u64)).is_some() { BUS_OK } else { BUS_ERR }
        }
    };
}

impl BusDevice for CpuWidget {
    cpu_access!(read8, write8, u8, BusRead8, 1);
    cpu_access!(read16, write16, u16, BusRead16, 2);
    cpu_access!(read32, write32, u32, BusRead32, 4);
    cpu_access!(read64, write64, u64, BusRead64, 8);
}

#[cfg(test)]
mod tests;
