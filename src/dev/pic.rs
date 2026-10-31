//! The PIC, the PCI host bridge on the interface board of the Origin 350 and
//! Tezro (IP53 family). One chip sits in one XIO widget and carries two
//! independent bridge heads, at widget offset 0 and `0x800000`, each with its
//! own register block and PCI bus. Widget 15 holds the first PIC; the
//! two- and four-processor Tezro board adds a second at widget 11.
//!
//! A [`Pic`] is self-contained: the machine builds one per widget, hands the
//! processor's view of it to Bedrock IO ([`Pic::widget`]), gives it a
//! [`PicPeer`] for the packets it originates (interrupts and DMA), connects
//! the Bedrock PIs it raises interrupts on, and attaches PCI functions to the
//! bus slots. The bus slots are hooks only: IOC4 at device 1 (PIC input 0),
//! the ISP12160 at device 3 (input 2) and the BCM5701 at device 4 (input 3)
//! are the board's choice, made by the machine.
//!
//! The register names are the ones the Bridge family shares
//! ([`xbridge::reg`](super::xbridge::reg)); the PIC leaves out the ATE, the
//! alias windows and the second and third write-request banks' meaning.
//!
//! # What is approximated here
//!
//! The model is the compatibility profile QEMU selects for Tezro: revision 3
//! with identity bit 0 set, `DIR_MAP` bits 0 to 16 and 20 to 23, the control
//! and Device masks, and the reset values below are emulator choices that let
//! the PROM and IRIX run, not measured reset state. The CPU word view (a
//! 32-bit access to the low half of an eight-byte slot) and the processor's
//! byte-lane placement for PCI cycles are inferred from the public drivers.
//! The one-millisecond reassertion of a still-high PCI input after the
//! processor clears its interrupt bit is the Bedrock PI's
//! ([`BedrockPi::set_level_source`](super::bedrock_pi::BedrockPi::set_level_source));
//! this chip tells it which inputs are active. Translated (ATE) DMA, PCI-X
//! errors and the bridge's own PCI function are unsupported and observable
//! ([`Pic::unsupported`]). The packet face a peer could read the heads
//! through has no user and is not modeled. All transfers are synchronous.
//!
//! Ported from hw/pci-host/sgi-pic.c in jamesbraid/qemu-sgi-origami.

use super::bedrock_pi::BedrockPi;
use super::xbridge::reg;
use crate::devlog::LogModule;
use crate::irq::{InterruptController, IrqLine};
use crate::pci::{PciBusDevice, PciDma, PciDmaStatus, Space};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR, BUS_OK};
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicUsize, Ordering};
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
const SET: u64 = 0x100;
const LEVELS: std::ops::RangeInclusive<u64> = 7..=127;

/// The widget identity word of a head: revision, part and manufacturer
/// packed as the common widget header lays them out, with bit 0 set.
pub const fn identity(head: usize) -> u32 {
    (REVISION << 28) | (PARTS[head] << 12) | (MANUFACTURER << 1) | 1
}

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
    pi: [OnceLock<Arc<BedrockPi>>; 2],
    /// The first PI level source this chip owns; head h, pin p is
    /// `source_base + h * PINS + p`.
    source_base: AtomicUsize,
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

struct Slot {
    function: Arc<dyn PciBusDevice>,
    options: Function,
}

/// A PCI cycle's responder: the slot and the offset within one of its BARs.
struct Claim {
    function: Arc<dyn PciBusDevice>,
    options: Function,
    bar: usize,
    offset: u32,
}

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
    slots: RwLock<[Option<Slot>; DEVICES as usize]>,
    pci_outputs: RwLock<[Option<IrqLine>; 32]>,
    unsupported: Mutex<Vec<(usize, bool, u64, usize)>>,
}

struct Pin {
    head: Weak<Head>,
    pci: bool,
}

impl InterruptController for Pin {
    fn set_line(&self, line: u32, level: bool) {
        let Some(head) = self.head.upgrade() else { return };
        if !self.pci {
            head.pin(line as usize, level);
            return;
        }
        let output = head.pci_outputs.read().get(line as usize).and_then(Clone::clone);
        match output {
            Some(output) => output.set(level),
            None if level => {
                crate::dlog!(LogModule::Bedrock, "pic: head {} PCI interrupt source {line} has no board wiring", head.index)
            }
            None => {}
        }
    }
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

impl Head {
    fn new(index: usize, wiring: Arc<Wiring>) -> Arc<Self> {
        Arc::new(Self {
            index,
            wiring,
            st: Mutex::new(State::new()),
            order: Mutex::new(()),
            slots: RwLock::new(std::array::from_fn(|_| None)),
            pci_outputs: RwLock::new(std::array::from_fn(|_| None)),
            unsupported: Mutex::new(Vec::new()),
        })
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

    /// Tell the Bedrock PI whether this pin's input is active at its
    /// destination's level, so it can reassert a level the processor
    /// cleared while the input stays high.
    fn update_level_source(&self, pin: usize, delivered: bool) {
        let (destination, active_pin) = {
            let st = self.st.lock();
            let bit = 1u64 << pin;
            (st.r(reg::INTERRUPT_HOST_BASE + pin * 8), st.pins as u64 & bit != 0 && st.r(reg::INTERRUPT_ENABLE) & bit != 0)
        };
        let level = destination >> 48;
        let target = usize::from(destination & (1 << 21) != 0);
        let active = delivered && active_pin && LEVELS.contains(&level);
        let source = self.wiring.source_base.load(Ordering::Acquire) + self.index * PINS + pin;
        for (interface, pi) in self.wiring.pi.iter().enumerate() {
            if let Some(pi) = pi.get() {
                pi.set_level_source(source, level as u8, active && interface == target);
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
        if !level {
            self.update_level_source(pin, false);
        }
        if level && !was_high && enabled {
            let delivered = self.packet(pin, true);
            self.update_level_source(pin, delivered);
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
        if !enabled {
            return;
        }
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

    fn read(&self, addr: u64, size: usize) -> u64 {
        if short_window(addr).is_some() {
            return self.short(addr, size, None).unwrap_or(0);
        }
        if (CONFIG_BASE..CONFIG_END).contains(&addr) && size <= 4 {
            return self.config(addr, size, None);
        }
        let off = addr as usize;
        if size == 4 && off % 8 == 4 {
            return self.word_read(off);
        }
        if size == 8 && off == 0 {
            return self.identity();
        }
        if size == 8 && off % 8 == 0 && defined(off) {
            return self.st.lock().r(off);
        }
        if size == 8 && off % 8 == 0 && (REQUEST_BASE..REQUEST_END).contains(&off) {
            let (bank, index) = request_slot(off);
            return self.st.lock().request[bank][index];
        }
        self.note_unsupported(addr, size, false, 0);
        0
    }

    fn write(&self, addr: u64, size: usize, value: u64) {
        if short_window(addr).is_some() {
            self.short(addr, size, Some(value));
            return;
        }
        if addr == ERROR_ADDRESS_WORD && size == 4 {
            self.invalid_address();
            return;
        }
        if (CONFIG_BASE..CONFIG_END).contains(&addr) && size <= 4 {
            self.config(addr, size, Some(value));
            return;
        }
        let off = addr as usize;
        if size == 4 && off % 8 == 4 {
            self.word_write(off, value as u32);
            return;
        }
        self.register_write(off, size, value);
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
            Some(slot) if !matches!(slot, reg::IDENTIFICATION | FLUSH | reg::INTERRUPT_STATUS | reg::INTERRUPT_DEVICE) => slot,
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
            reg::IDENTIFICATION | FLUSH | reg::INTERRUPT_STATUS => self.note_unsupported(off as u64, size, true, value),
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
                        let delivered = self.packet(pin, true);
                        self.update_level_source(pin, delivered);
                    } else if value & (1 << pin) == 0 {
                        self.update_level_source(pin, false);
                    }
                }
            }
            reg::INTERRUPT_RESET => {
                let _order = self.order.lock();
                let (pins, enable) = {
                    let mut st = self.st.lock();
                    let status = (st.r(reg::INTERRUPT_STATUS) & !value) | st.pins as u64;
                    st.set(reg::INTERRUPT_STATUS, status);
                    (st.pins as u64, st.r(reg::INTERRUPT_ENABLE))
                };
                for pin in 0..PINS {
                    if value & pins & enable & (1 << pin) != 0 {
                        let delivered = self.packet(pin, true);
                        self.update_level_source(pin, delivered);
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
                if (reg::INTERRUPT_HOST_BASE..=reg::INTERRUPT_HOST_LAST).contains(&off) {
                    let _order = self.order.lock();
                    self.update_level_source((off - reg::INTERRUPT_HOST_BASE) / 8, false);
                }
            }
        }
    }

    // ---- PCI configuration ----

    /// A processor configuration cycle. Bus 0 devices sit at `0x1000`
    /// strides from `0x20000`; the type-1 window addresses the bus and
    /// device `PCI_TYPE1_CONFIG` names. A miss reads all ones.
    fn config(&self, addr: u64, size: usize, value: Option<u64>) -> u64 {
        let mask = u32::MAX >> (32 - size * 8);
        let function = (addr >> 8) & 7;
        let (bus, device) = if addr < CONFIG_TYPE1 {
            (0, ((addr - CONFIG_BASE) >> 12) as usize)
        } else {
            let t1 = self.st.lock().r(reg::PCI_TYPE1_CONFIG);
            (((t1 >> 16) & 0xff) as usize, ((t1 >> 11) & 0x1f) as usize)
        };
        let register = ((addr & 0xff) as usize) ^ (4 - size);
        let found = (bus == 0 && function == 0 && (1..=DEVICES as usize).contains(&device))
            .then(|| self.slots.read()[device - 1].as_ref().map(|slot| slot.function.clone()))
            .flatten();
        let Some(function) = found else {
            crate::dlog!(
                LogModule::Bedrock,
                "pic: head {} configuration {} miss at {addr:#x}",
                self.index,
                if value.is_some() { "write" } else { "read" }
            );
            return mask as u64;
        };
        let shift = (register & 3) * 8;
        match value {
            Some(value) => {
                function.config().write_masked(register / 4, (value as u32 & mask) << shift, mask << shift);
                0
            }
            None => ((function.config().read(register / 4) >> shift) & mask) as u64,
        }
    }

    // ---- PCI memory and I/O cycles ----

    /// The function that decodes `len` bytes at `address` in `space`, whole
    /// within one BAR.
    fn claim(&self, space: Space, address: u64, len: usize) -> Option<Claim> {
        let first = u32::try_from(address).ok()?;
        let last = u32::try_from(address.checked_add(len as u64)? - 1).ok()?;
        self.slots.read().iter().flatten().find_map(|slot| {
            let (bar, offset) = slot.function.config().decode(space, first)?;
            let (end_bar, end) = slot.function.config().decode(space, last)?;
            (end_bar == bar && end == offset + (len as u32 - 1)).then(|| Claim {
                function: slot.function.clone(),
                options: slot.options,
                bar,
                offset,
            })
        })
    }

    /// Move `bytes` (little-endian PCI byte order, in address order) across
    /// a PCI cycle. False when no function decodes the whole run.
    fn pci_cycle(&self, space: Space, address: u64, bytes: &mut [u8], write: bool) -> bool {
        let Some(claim) = self.claim(space, address, bytes.len()) else {
            return false;
        };
        for (i, part) in bytes.chunks_mut(4).enumerate() {
            let offset = claim.offset + i as u32 * 4;
            if write {
                claim.function.bar_write(claim.bar, offset, le_value(part) as u32, part.len());
            } else {
                let v = claim.function.bar_read(claim.bar, offset, part.len());
                part.copy_from_slice(&v.to_le_bytes()[..part.len()]);
            }
        }
        true
    }

    /// A processor access to PCI space. Aligned bytes, halfwords and words
    /// reach a conventional adapter at the address with its low two bits
    /// corrected, carrying the numeric value; a self-ordered function gets
    /// the processor's byte sequence at the address as it is. Other widths
    /// pass through without placement.
    fn cpu_pci(&self, space: Space, mut address: u64, size: usize, value: Option<u64>) -> Option<u64> {
        let placed = matches!(size, 1 | 2 | 4) && address.is_multiple_of(size as u64);
        let native = placed
            && self.claim(space, address, size).is_some_and(|claim| claim.options.self_ordered_pio);
        if placed && !native && size < 4 {
            address ^= 4 - size as u64;
        }
        let mut bytes = [0u8; 8];
        let ok = if let Some(value) = value {
            bytes = if native { be_bytes(size, value) } else { le_bytes(size, value) };
            self.pci_cycle(space, address, &mut bytes[..size], true)
        } else {
            self.pci_cycle(space, address, &mut bytes[..size], false)
        };
        ok.then(|| if native { be_value(&bytes[..size]) } else { le_value(&bytes[..size]) })
    }

    /// The short device-I/O windows: the Device register of the window
    /// picks PCI memory or I/O space and the megabyte of it.
    fn short(&self, addr: u64, size: usize, value: Option<u64>) -> Option<u64> {
        let write = value.is_some();
        let (window, offset, window_size) = short_window(addr)?;
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
        let memory = device & DEVICE_MEMORY != 0;
        let mut base = device & if memory { 0x3ff } else { 0xfff };
        if window < 2 {
            base &= !1;
        }
        let result = self.cpu_pci(if memory { Space::Mem } else { Space::Io }, (base << 20) + offset, size, value);
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
    fn pci_window(&self, offset: u64, size: usize, value: Option<u64>) -> u64 {
        let result = self.cpu_pci(Space::Mem, offset, size, value);
        if result.is_none() {
            self.note_unsupported(PCI_WINDOW[0] + offset, size, value.is_some(), value.unwrap_or(0));
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
            // A swapped slot moves single bytes, each at its address with
            // the low two bits flipped.
            return chunk.iter_mut().enumerate().all(|(i, byte)| {
                self.send_dma(source, widget, (target + i as u64) ^ 3, std::slice::from_mut(byte), write)
            });
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
    /// and the XIO payload below. Bit 55 selects an ordered byte stream;
    /// clear, the bytes keep 32-bit numeric values (the byte address low
    /// bits flip by 3, or by 7 for a REQ64 slot's 64-bit requester).
    fn direct64(&self, device: u8, at: u64, chunk: &mut [u8], write: bool) -> bool {
        let wide = self.slots.read()[device as usize - 1].as_ref().is_some_and(|slot| slot.options.req64);
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
        let mask = if wide { 7 } else { 3 };
        let mut done = 0;
        while done < chunk.len() {
            let here = at + done as u64;
            if at & (1 << 55) != 0 {
                let n = 8.min(chunk.len() - done);
                if !check(here, n) || !self.send_dma(source, widget, payload + done as u64, &mut chunk[done..done + n], write) {
                    return false;
                }
                done += n;
            } else {
                if !check(here, 1) || !self.send_dma(source, widget, (payload + done as u64) ^ mask, &mut chunk[done..done + 1], write) {
                    return false;
                }
                done += 1;
            }
        }
        true
    }

    fn attach(self: &Arc<Self>, device: u8, function: Arc<dyn PciBusDevice>, options: Function) {
        assert!((1..=DEVICES).contains(&device), "PIC devices are 1 to {DEVICES}");
        self.slots.write()[device as usize - 1] = Some(Slot { function: function.clone(), options });
        function.connect_dma(Arc::new(SlotDma { head: Arc::downgrade(self), device }));
        let pin = function.config().interrupt_pin();
        if (1..=4).contains(&pin) {
            function.connect_interrupt(IrqLine::new(
                Arc::new(Pin { head: Arc::downgrade(self), pci: true }),
                device as u32 * 4 + pin as u32 - 1,
            ));
        }
    }

    fn reset(&self) {
        {
            let _order = self.order.lock();
            self.st.lock().reset();
        }
        for pin in 0..PINS {
            self.update_level_source(pin, false);
        }
        let outputs = self.pci_outputs.read().clone();
        for line in outputs.into_iter().flatten() {
            line.set(false);
        }
        for slot in self.slots.read().iter().flatten() {
            slot.function.config().reset();
        }
        self.unsupported.lock().clear();
    }
}

/// The PCI host bridge in one XIO widget.
pub struct Pic {
    widget: u8,
    wiring: Arc<Wiring>,
    heads: [Arc<Head>; 2],
}

impl Pic {
    /// A PIC that answers as XIO widget `widget`.
    pub fn new(widget: u8) -> Arc<Self> {
        let wiring = Arc::new(Wiring {
            peer: OnceLock::new(),
            pi: [OnceLock::new(), OnceLock::new()],
            source_base: AtomicUsize::new(0),
        });
        Arc::new(Self {
            widget,
            heads: [Head::new(0, wiring.clone()), Head::new(1, wiring.clone())],
            wiring,
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

    /// Where interrupts and DMA this chip originates go. Once, before the
    /// machine runs.
    pub fn connect_peer(&self, peer: Arc<dyn PicPeer>) -> Result<(), AlreadyConnected> {
        self.wiring.peer.set(peer).map_err(|_| AlreadyConnected)
    }

    /// Connect Bedrock PI `interface` (0 or 1) as a destination of this
    /// chip's interrupts. `source_base` is the first of the 16 level
    /// sources the chip owns on the PI; two PICs on one node need different
    /// bases.
    pub fn connect_pi(&self, interface: usize, pi: Arc<BedrockPi>, source_base: usize) {
        self.wiring.source_base.store(source_base, Ordering::Release);
        let _ = self.wiring.pi[interface].set(pi);
    }

    /// Put a PCI function at configuration `device` (1 to 4) of `head`'s
    /// bus. It gets its DMA path and its INTA to INTD lines; the board
    /// routes those with [`Pic::route_pci_interrupt`] or [`Pic::wire_inta`].
    pub fn attach(&self, head: usize, device: u8, function: Arc<dyn PciBusDevice>, options: Function) {
        self.heads[head].attach(device, function, options);
    }

    /// Interrupt input `pin` (0 to 7) of `head`, for a device whose
    /// interrupt the board wires to it directly.
    pub fn pin_line(&self, head: usize, pin: usize) -> IrqLine {
        assert!(pin < PINS);
        IrqLine::new(Arc::new(Pin { head: Arc::downgrade(&self.heads[head]), pci: false }), pin as u32)
    }

    /// Board wiring of a device's interrupt pin (`intx` 0 for INTA) to
    /// `output`.
    pub fn route_pci_interrupt(&self, head: usize, device: u8, intx: u8, output: IrqLine) {
        assert!((1..=DEVICES).contains(&device) && intx < 4);
        self.heads[head].pci_outputs.write()[(device * 4 + intx) as usize] = Some(output);
    }

    /// Route `device`'s INTA to the input numbered one less than the
    /// device, the convention the public drivers use.
    pub fn wire_inta(&self, head: usize, device: u8) {
        self.route_pci_interrupt(head, device, 0, self.pin_line(head, device as usize - 1));
    }

    /// Set interrupt input `pin` of `head`.
    pub fn set_pin(&self, head: usize, pin: usize, level: bool) {
        self.heads[head].pin(pin, level);
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
    }
}

impl Resettable for Pic {
    fn power_on(&self) {
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
    /// `None` is a bus error. Accesses the PIC cannot make sense of that are
    /// not malformed (unsupported registers, misses, refused windows) read
    /// zero and write nowhere without one.
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
            return Some(self.heads[window].pci_window(addr - PCI_WINDOW[window], size, value));
        }
        let head = &self.heads[(addr / HEAD_OFFSET) as usize];
        let rel = addr % HEAD_OFFSET;
        match value {
            Some(v) => {
                head.write(rel, size, v);
                Some(0)
            }
            None => Some(head.read(rel, size)),
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
