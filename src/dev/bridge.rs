//! The Octane's onboard Bridge: the XIO-to-PCI host bridge at widget 0xf.
//!
//! One side is an ordinary XIO widget whose 16 MB small window holds the
//! common widget registers, the translation table, PCI configuration space,
//! eight device windows onto PCI and the 4 MB boot-flash aperture. The other
//! side is a PCI bus with eight slots, an eight-pin interrupt controller that
//! sends its interrupts to HEART, and a translator for transfers a function
//! initiates.
//!
//! Window layout (offsets in the small window):
//!
//! | offset     | what                                              |
//! |------------|---------------------------------------------------|
//! | 0x000000   | widget and Bridge registers (64 KB decoded)       |
//! | 0x010000   | translation table, 128 eight-byte entries        |
//! | 0x011000   | the same entries' lower halves, read only         |
//! | 0x020000   | type 0 configuration space, 4 KB per device       |
//! | 0x080000   | external SSRAM (unpopulated)                      |
//! | 0x200000   | device windows: two of 2 MB, then six of 1 MB     |
//! | 0xc00000   | flash, two 2 MB chip selects on a 16-bit port     |
//!
//! Registers are 32 bits in the data (upper) half of an eight-byte slot.
//! The register, translation, byte-lane and interrupt behaviour follows the
//! published Bridge ASIC specification (rev 4.0, `docs/Octane/`). Where the
//! specification is silent the choice is marked as an approximation:
//!
//! * Arbitration, read response buffers, the SSRAM parity and request
//!   timer registers are storage. Nothing arbitrates, nothing times out.
//! * Error capture covers the PCI group (master time-out), the dropped DMA
//!   read and invalid register addresses. Parity, retry and target-abort
//!   errors are never raised.
//! * The translation table is the chip's internal 128 entries; no external
//!   SRAM extends it.
//! * Processor accesses to the 2 GB long window (the translated PIO window
//!   at 0x4000_0000) are not modelled.
//! * A function's DMA address at or above 4 GiB goes through the 64-bit
//!   direct window: bits 63:60 name the widget and the low 48 bits pass
//!   through. PCI-local space (below 0x4000_0000), which would reach
//!   another PCI device, is not carried anywhere and fails.
//! * Only function 0 of a slot exists.
//!
//! Interrupts leave the chip outside its state lock. A pin's level is
//! decided under the state lock and the packet to HEART goes out after it
//! is dropped, with a separate ordering lock held across both so two
//! threads' edges on one pin reach HEART in the order they were decided.
//! DMA goes to the memory every bus master shares (`Arc<dyn BusDevice>`).
//!
//! Ported from hw/pci-host/sgi-bridge.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, RwLock};

use crate::dev::w1;
use crate::irq::{InterruptController, IrqLine};
use crate::pci::{bar_runs, BarAt, BarRun, BarRunSink, PciBar, PciBusDevice, PciDma, PciDmaStatus, PciLanes, Space};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK};

/// The widgets this chip addresses by number on the Crossbow: HEART takes
/// its interrupt packets and memory transfers, widget 15 is its own port.
const WIDGET_HEART: u8 = 0x8;
const WIDGET_BRIDGE: u8 = 0xf;
/// A widget's 16 MB small window.
const WIDGET_SMALL_SIZE: u32 = 0x0100_0000;

/// Byte offsets of the registers (the slot's base; the data is at +4).
pub mod reg {
    pub const IDENTIFICATION: u32 = 0x000;
    pub const STATUS: u32 = 0x008;
    pub const ERR_UPPER: u32 = 0x010;
    pub const ERR_LOWER: u32 = 0x018;
    pub const CONTROL: u32 = 0x020;
    pub const REQUEST_TIMEOUT: u32 = 0x028;
    pub const INT_DEST_UPPER: u32 = 0x030;
    pub const INT_DEST_LOWER: u32 = 0x038;
    pub const ERR_CMD_WORD: u32 = 0x040;
    pub const LINK_CONFIG: u32 = 0x048;
    pub const TARGET_FLUSH: u32 = 0x050;
    pub const AUX_ERR_CMD_WORD: u32 = 0x058;
    pub const RESP_ERR_UPPER: u32 = 0x060;
    pub const RESP_ERR_LOWER: u32 = 0x068;
    pub const TEST_PIN_CONTROL: u32 = 0x070;
    pub const DIRECT_MAP: u32 = 0x080;
    pub const SSRAM_PARITY_ERR: u32 = 0x090;
    pub const ARBITRATION: u32 = 0x0a0;
    pub const NIC: u32 = 0x0b0;
    pub const BUS_TIMEOUT: u32 = 0x0c0;
    pub const PCI_TYPE1_CONFIG: u32 = 0x0c8;
    pub const PCI_ERR_UPPER: u32 = 0x0d0;
    pub const PCI_ERR_LOWER: u32 = 0x0d8;
    pub const INT_STATUS: u32 = 0x100;
    pub const INT_ENABLE: u32 = 0x108;
    pub const INT_RESET: u32 = 0x110;
    pub const INT_MODE: u32 = 0x118;
    pub const INT_DEVICE: u32 = 0x120;
    pub const HOST_ERR_FIELD: u32 = 0x128;
    pub const fn int_host(pin: u32) -> u32 {
        0x130 + 8 * pin
    }
    pub const fn device(slot: u32) -> u32 {
        0x200 + 8 * slot
    }
    pub const fn device_flush(slot: u32) -> u32 {
        0x240 + 8 * slot
    }
    pub const RRB_EVEN: u32 = 0x280;
    pub const RRB_ODD: u32 = 0x288;
    pub const RRB_STATUS: u32 = 0x290;
    pub const RRB_CLEAR: u32 = 0x298;
}

/// Window geometry, as offsets in the small window.
pub const ATE_BASE: u32 = 0x01_0000;
pub const ATE_LOW_BASE: u32 = 0x01_1000;
pub const CFG0_BASE: u32 = 0x02_0000;
pub const CFG0_SIZE: u32 = 8 * 0x1000;
pub const SSRAM_BASE: u32 = 0x08_0000;
pub const SSRAM_SIZE: u32 = 0x08_0000;
pub const DEVICE_WINDOW_BASE: u32 = 0x20_0000;
pub const FLASH_BASE: u32 = 0xc0_0000;
pub const FLASH_SELECT_SIZE: u32 = 0x20_0000;
pub const DEVICES: usize = 8;
pub const PINS: usize = 8;
pub const ATE_ENTRIES: usize = 128;

/// Offset of `slot`'s device window in the small window. The first two are
/// 2 MB, the rest 1 MB.
pub const fn device_window_offset(slot: u32) -> u32 {
    let large = if slot < 2 { slot } else { 2 };
    DEVICE_WINDOW_BASE + large * 0x20_0000 + (slot - large) * 0x10_0000
}

const fn device_window_size(slot: u32) -> u32 {
    if slot < 2 {
        0x20_0000
    } else {
        0x10_0000
    }
}

/// The slot whose device window holds small-window offset `off`.
fn window_slot(off: u32) -> Option<u32> {
    (0..DEVICES as u32).find(|&s| {
        let b = device_window_offset(s);
        off >= b && off - b < device_window_size(s)
    })
}

const SLOT_DATA: u32 = 4;
/// Where in HEART's widget space an interrupt packet lands.
const HEART_INT_PACKET: u64 = 0x80;
const REGS_DEFINED: u32 = 0x300;
const REG_SLOTS: usize = (REGS_DEFINED / 8) as usize;
/// The one reserved data half this chip answers specially.
const RESERVED_DATA_HALF: u32 = 0x19c;

const ID_REVISION_SHIFT: u32 = 28;
const ID_PART_SHIFT: u32 = 12;
const ID_MANUFACTURER_SHIFT: u32 = 1;
const REVISION: u32 = 1;
const PART: u32 = 0xc002;
const MANUFACTURER: u32 = 0x036;

const STATUS_FLASH_SELECT: u32 = 1 << 6;
const STATUS_PCI_MODE: u32 = 1 << 5;

pub const CONTROL_FLASH_WRITE_EN: u32 = 1 << 31;
pub const CONTROL_MEM_SWAP: u32 = 1 << 22;
pub const CONTROL_PAGE_16K: u32 = 1 << 21;
const CONTROL_RESET: u32 = (1 << 30) | (1 << 29) | (1 << 28) | (0xf << 24) | (3 << 12) | (1 << 9) | (0x1f << 4) | 0xf;
/// Two strobes that clear the link retry counters and always read zero.
const CONTROL_RETRY_CLEAR: u32 = 0x3 << 10;

const LINK_CONFIG_RESET: u32 = (0x3ff << 16) | (0x06 << 10) | 0x010;
const LINK_CONFIG_WRITABLE: u32 = (0x3ff << 16) | (0x3f << 10) | 0x3ff;

pub const DIRECT_WIDGET_SHIFT: u32 = 20;
const DIRECT_WIDGET: u32 = 0xf << DIRECT_WIDGET_SHIFT;
const DIRECT_REMOTE_MAP_MODE: u32 = 1 << 18;
pub const DIRECT_ADD_512MB: u32 = 1 << 17;
const DIRECT_OFFSET: u32 = 0x1ffff;
const DIRECT_MAP_WRITABLE: u32 = DIRECT_WIDGET | DIRECT_REMOTE_MAP_MODE | DIRECT_ADD_512MB | DIRECT_OFFSET;

const ARBITRATION_RESET: u32 = (2 << 16) | (1 << 3);
const ARBITRATION_WRITABLE: u32 = (3 << 16) | (0xff << 8) | (1 << 6) | (7 << 3) | 7;
const BUS_TIMEOUT_RESET: u32 = 1 << 16;
const BUS_TIMEOUT_WRITABLE: u32 = (0x1f << 16) | (1 << 12) | 0x3ff;
const TYPE1_WRITABLE: u32 = (0xff << 16) | (0x1f << 11);

// The one-wire master: writing the timing fields runs one bit slot.
const NIC_LEVEL: u32 = 1 << 0;
const NIC_DONE: u32 = 1 << 1;
const NIC_SAMPLE_SHIFT: u32 = 2;
const NIC_SAMPLE_MASK: u32 = 0xff << NIC_SAMPLE_SHIFT;
const NIC_PULSE_SHIFT: u32 = 10;
const NIC_PULSE_MASK: u32 = 0x3ff << NIC_PULSE_SHIFT;
const NIC_WRITABLE: u32 = NIC_SAMPLE_MASK | NIC_PULSE_MASK;

// Interrupt status and enable bits.
pub const INT_DEVICE_PINS: u32 = 0xff;
pub const INT_INVALID_ADDR: u32 = 1 << 24;
pub const INT_PCI_MASTER_TIMEOUT: u32 = 1 << 11;
const INT_PCI_GROUP: u32 = 0x3f << 10;
pub const INT_XREAD_REQ_TIMEOUT: u32 = 1 << 9;
const INT_RESP_BUF_GROUP: u32 = (1 << 28) | (1 << 26) | (1 << 9);
pub const INT_MULTI_ERR: u32 = 1 << 31;
const INT_ENABLE_WRITABLE: u32 = 0x7fff_ffff;
const INT_RESET_GROUPS: u32 = 0x7f;
const INT_RESET_PCI_GRP: u32 = 1 << 0;
const INT_RESET_REQ_DSP_GRP: u32 = 1 << 3;
const INT_RESET_RESP_BUF_GRP: u32 = 1 << 4;
const INT_RESET_MULTI_ERR: u32 = 1 << 6;
const INT_MODE_WRITABLE: u32 = 0xff;
const INT_DEVICE_WRITABLE: u32 = 0xff_ffff;
const INT_HOST_ADDRESS: u32 = 0x3ff << 8;
const INT_HOST_FIELD: u32 = 0xff;
const INT_HOST_WRITABLE: u32 = INT_HOST_ADDRESS | INT_HOST_FIELD;
const HOST_ERR_FIELD_WRITABLE: u32 = 0xff;

const PCI_ERR_DEVICE_SHIFT: u32 = 16;
const RESP_ERR_DEVICE_SHIFT: u32 = 20;

// Per-slot Device(x) register.
pub const DEV_ERROR_LOCK: u32 = 1 << 28;
pub const DEV_PAGE_CROSS_CHECK: u32 = 1 << 27;
pub const DEV_64BIT_DEVICE: u32 = 1 << 22;
pub const DEV_SWAP_TRANSLATED: u32 = 1 << 20;
pub const DEV_SWAP_DIRECT: u32 = 1 << 19;
pub const DEV_SWAP_PROGRAMMED: u32 = 1 << 13;
pub const DEV_MEMORY_SPACE: u32 = 1 << 12;
const DEV_WINDOW_BASE_MASK: u32 = 0xfff;
const DEVICE_WRITABLE: u32 = 0x1fff_ffff;

const fn device_window_reset(slot: u32) -> u32 {
    if slot < 2 {
        2 * slot
    } else {
        slot + 2
    }
}

const fn device_reset(slot: u32) -> u32 {
    DEV_ERROR_LOCK | DEV_PAGE_CROSS_CHECK | DEV_MEMORY_SPACE | device_window_reset(slot)
}

// Translation entries.
const ATE_ADDRESS: u64 = ((1u64 << 36) - 1) << 12;
const ATE_TARGET_WIDGET_SHIFT: u32 = 8;
const ATE_VALID: u64 = 1;

const PCI_TRANSLATED: u32 = 0x4000_0000;
const PCI_DIRECT: u32 = 0x8000_0000;
const DIRECT_512MB: u64 = 512 << 20;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Access {
    /// Writes are a guest error and are ignored.
    Ro,
    /// The writable mask selects the guest-settable bits.
    Rw,
    /// A strobe: the write is taken, a read gives the reset value.
    Wo,
    /// Read only and computed.
    Live,
}

#[derive(Clone, Copy)]
struct RegInfo {
    access: Access,
    reset: u32,
    writable: u32,
}

const fn info(access: Access, reset: u32, writable: u32) -> Option<RegInfo> {
    Some(RegInfo { access, reset, writable })
}

/// Decode, reset value and writable mask from one table, by slot base.
fn reg_info(slot_base: u32) -> Option<RegInfo> {
    use reg::*;
    use Access::*;
    match slot_base {
        IDENTIFICATION | ERR_UPPER | ERR_LOWER | ERR_CMD_WORD | TARGET_FLUSH | AUX_ERR_CMD_WORD | RESP_ERR_UPPER
        | RESP_ERR_LOWER | SSRAM_PARITY_ERR | PCI_ERR_UPPER | PCI_ERR_LOWER | RRB_STATUS => info(Ro, 0, 0),
        STATUS => info(Ro, STATUS_PCI_MODE, 0),
        CONTROL => info(Rw, CONTROL_RESET, !CONTROL_RETRY_CLEAR),
        REQUEST_TIMEOUT => info(Rw, 0xf_ffff, 0xf_ffff),
        INT_DEST_UPPER | INT_DEST_LOWER | TEST_PIN_CONTROL | RRB_EVEN | RRB_ODD => info(Rw, 0, 0xffff_ffff),
        LINK_CONFIG => info(Rw, LINK_CONFIG_RESET, LINK_CONFIG_WRITABLE),
        DIRECT_MAP => info(Rw, DIRECT_ADD_512MB, DIRECT_MAP_WRITABLE),
        ARBITRATION => info(Rw, ARBITRATION_RESET, ARBITRATION_WRITABLE),
        NIC => info(Rw, 0, NIC_WRITABLE),
        BUS_TIMEOUT => info(Rw, BUS_TIMEOUT_RESET, BUS_TIMEOUT_WRITABLE),
        PCI_TYPE1_CONFIG => info(Rw, 0, TYPE1_WRITABLE),
        INT_STATUS => info(Live, 0, 0),
        INT_ENABLE => info(Rw, 0, INT_ENABLE_WRITABLE),
        INT_RESET => info(Wo, 0, INT_RESET_GROUPS),
        RRB_CLEAR => info(Wo, 0, 0),
        INT_MODE => info(Rw, 0, INT_MODE_WRITABLE),
        INT_DEVICE => info(Rw, 0, INT_DEVICE_WRITABLE),
        HOST_ERR_FIELD => info(Rw, 0, HOST_ERR_FIELD_WRITABLE),
        0x130..=0x168 if slot_base % 8 == 0 => info(Rw, 0, INT_HOST_WRITABLE),
        0x200..=0x238 if slot_base % 8 == 0 => info(Rw, device_reset((slot_base - 0x200) / 8), DEVICE_WRITABLE),
        0x240..=0x278 if slot_base % 8 == 0 => info(Ro, 0, 0),
        _ => None,
    }
}

/// The decoded interrupt destination pair: widget, and the 48-bit address.
struct IntDest {
    widget: u8,
    address: u64,
}

fn int_dest(upper: u32, lower: u32) -> IntDest {
    IntDest { widget: ((upper >> 16) & 0xf) as u8, address: (((upper & 0xffff) as u64) << 32) | lower as u64 }
}

struct State {
    reg: [u32; REG_SLOTS],
    ate: [u64; ATE_ENTRIES],
    /// Latched error bits of the interrupt status register (31:8).
    int_error: u32,
    /// The eight pins' driven levels, which the status register reports.
    /// They follow the level and never latch.
    pin_level: u8,
    pci_pin_level: u8,
    gpio_pin_level: u8,
    /// Each slot's INTA..INTD, as `slot * 4 + pin - 1`.
    pci_lines: [bool; DEVICES * 4],
    /// Slots whose data phases are 64 bits wide.
    dma_64bit: u8,
    /// The transfer the last DMA translation served, for error capture.
    dma_pci_address: u64,
    dma_device: u32,
}

impl State {
    fn r(&self, off: u32) -> u32 {
        self.reg[(off / 8) as usize]
    }

    fn set_r(&mut self, off: u32, v: u32) {
        self.reg[(off / 8) as usize] = v;
    }

    /// The chip's own core: the register file, the table and the latched
    /// errors. The interrupt pins' levels belong to whatever drives them.
    fn reset_core(&mut self, flash_strap: bool) {
        for (i, slot) in self.reg.iter_mut().enumerate() {
            *slot = reg_info(i as u32 * 8).map_or(0, |r| r.reset);
        }
        self.set_r(
            reg::IDENTIFICATION,
            (REVISION << ID_REVISION_SHIFT) | (PART << ID_PART_SHIFT) | (MANUFACTURER << ID_MANUFACTURER_SHIFT) | 1,
        );
        if flash_strap {
            let s = self.r(reg::STATUS);
            self.set_r(reg::STATUS, s | STATUS_FLASH_SELECT);
        }
        self.int_error = 0;
        // Whole entries, not just valid bits: a stale page address behind a
        // clear valid bit is a working translation waiting for one write.
        self.ate = [0; ATE_ENTRIES];
    }
}

/// A flash chip select's part: a bus device addressed from the start of the
/// part, which answers 16-bit big-endian reads and writes.
struct FlashPart {
    dev: Arc<dyn BusDevice>,
    size: u32,
}

/// The packets a locked section decided to send to HEART: (vector, set).
type Packets = Vec<(u8, bool)>;

/// The onboard Bridge. Hand it out as the `BusDevice` of the widget's small
/// window: `Arc<Bridge>` coerces to `Arc<dyn BusDevice>`. As an
/// [`InterruptController`] its line is a pin, 0..8, driven as a level.
pub struct Bridge {
    st: Mutex<State>,
    /// Held across "decide a packet, send it", never with `st`.
    /// Nothing but HEART's `set_line` is called under it.
    irq_order: Mutex<()>,
    /// Memory as a bus master reaches it: physical addresses, fitted SDRAM
    /// only, a bus error for anything else.
    mem: Arc<dyn BusDevice>,
    /// HEART: a line is an interrupt bit, the vector a packet names.
    heart: Arc<dyn InterruptController>,
    slots: RwLock<[Option<Arc<dyn PciBusDevice>>; DEVICES]>,
    flash: RwLock<[Option<FlashPart>; 2]>,
    flash_strap: AtomicBool,
    /// The one-wire net behind the NIC register; none fitted reads as an
    /// empty bus.
    microlan: Mutex<Option<w1::Bus>>,
    /// Whether the Crossbow has a live link to a widget; default only HEART.
    link_alive: RwLock<Box<dyn Fn(u8) -> bool + Send + Sync>>,
    /// Where the runs of the device windows a BAR decodes whole go (the
    /// machine maps them over this widget's window), and the table entries
    /// made for them, so one configuration maps to one entry.
    bar_sink: RwLock<Option<BarRunSink>>,
    bar_entries: Mutex<std::collections::HashMap<(usize, usize, usize, bool, u64), Arc<dyn BusDevice>>>,
}

/// A function's INTA..INTD, as a level into the Bridge: the line is
/// `slot * 4 + pin - 1`.
struct PciPin {
    bridge: Weak<Bridge>,
}

impl InterruptController for PciPin {
    fn set_line(&self, line: u32, level: bool) {
        if let Some(b) = self.bridge.upgrade() {
            b.set_pci_line(line as usize, level);
        }
    }
}

/// One of the chip's eight pins driven directly (the GPIO inputs that
/// share them with PCI), holding the Bridge weakly so the device that
/// holds the line does not keep it alive.
struct WirePin {
    bridge: Weak<Bridge>,
}

impl InterruptController for WirePin {
    fn set_line(&self, line: u32, level: bool) {
        if let Some(b) = self.bridge.upgrade() {
            b.set_wire_irq(line, level);
        }
    }
}

/// A function's way to memory, tagged with its slot: the swap bits that
/// steer a transfer live in the requesting slot's own Device(x) register.
struct SlotDma {
    bridge: Weak<Bridge>,
    slot: usize,
}

impl PciDma for SlotDma {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        match self.bridge.upgrade() {
            Some(b) => b.dma(self.slot, addr, DmaBuf::Read(buf)),
            None => PciDmaStatus::MasterAbort,
        }
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        match self.bridge.upgrade() {
            Some(b) => b.dma(self.slot, addr, DmaBuf::Write(data)),
            None => PciDmaStatus::MasterAbort,
        }
    }
}

enum DmaBuf<'a> {
    Read(&'a mut [u8]),
    Write(&'a [u8]),
}

/// How a transfer's bytes sit in memory relative to PCI address order.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Placement {
    /// A byte stream: PCI byte k is host byte k.
    Identity,
    /// PCI byte k is host byte k ^ 3 within an aligned 32-bit word.
    Word,
    /// PCI byte k is host byte k ^ 7 within an aligned doubleword.
    Doubleword,
}

impl Placement {
    fn mask(self) -> u64 {
        match self {
            Placement::Identity => 0,
            Placement::Word => 3,
            Placement::Doubleword => 7,
        }
    }
}

impl Bridge {
    /// A Bridge that masters memory through `mem` and interrupts HEART
    /// through `heart` (line = vector). Registers start at their power-up
    /// values.
    pub fn new(mem: Arc<dyn BusDevice>, heart: Arc<dyn InterruptController>) -> Arc<Self> {
        let mut st = State {
            reg: [0; REG_SLOTS],
            ate: [0; ATE_ENTRIES],
            int_error: 0,
            pin_level: 0,
            pci_pin_level: 0,
            gpio_pin_level: 0,
            pci_lines: [false; DEVICES * 4],
            dma_64bit: 0,
            dma_pci_address: 0,
            dma_device: 0,
        };
        st.reset_core(false);
        Arc::new(Self {
            st: Mutex::new(st),
            irq_order: Mutex::new(()),
            mem,
            heart,
            slots: RwLock::new(Default::default()),
            flash: RwLock::new(Default::default()),
            flash_strap: AtomicBool::new(false),
            microlan: Mutex::new(None),
            link_alive: RwLock::new(Box::new(|w| w == WIDGET_HEART)),
            bar_sink: RwLock::new(None),
            bar_entries: Mutex::new(Default::default()),
        })
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        let me = self.clone();
        register_lock_fn("bridge::state", move || me.st.is_locked());
        let me = self.clone();
        register_lock_fn("bridge::irq_order", move || me.irq_order.is_locked());
        let me = self.clone();
        register_lock_fn("bridge::microlan", move || me.microlan.is_locked());
    }

    /// Run `f` on the state, then send the packets it decided on. The
    /// order lock spans both, so packets for one pin leave in the order
    /// they were decided; HEART is called with the state lock dropped.
    fn irq_locked<R>(&self, f: impl FnOnce(&mut State, &mut Packets) -> R) -> R {
        let _order = self.irq_order.lock();
        let mut packets = Packets::new();
        let ret = f(&mut self.st.lock(), &mut packets);
        for (vector, set) in packets {
            self.heart.set_line(vector as u32, set);
        }
        ret
    }

    /// One of the chip's pins as a line a device can hold: the level drives
    /// the pin directly, sharing it with whatever PCI functions are wired
    /// to it (IOC3's SuperIO interrupt is `pin_line(4)`). The line holds the
    /// Bridge weakly.
    pub fn pin_line(self: &Arc<Self>, pin: u8) -> IrqLine {
        IrqLine::new(Arc::new(WirePin { bridge: Arc::downgrade(self) }), pin as u32)
    }

    /// Put a function in `slot` (0..8), give it its way to memory and wire
    /// its interrupt pin (INTA..INTD per its header) to the Bridge's pins.
    pub fn attach(self: &Arc<Self>, slot: u8, dev: Arc<dyn PciBusDevice>) {
        let slot = slot as usize;
        assert!(slot < DEVICES, "Bridge has {DEVICES} slots");
        dev.connect_dma(Arc::new(SlotDma { bridge: Arc::downgrade(self), slot }));
        let pin = dev.config().interrupt_pin();
        if (1..=4).contains(&pin) {
            let router = Arc::new(PciPin { bridge: Arc::downgrade(self) });
            dev.connect_interrupt(IrqLine::new(router, (slot * 4 + pin as usize - 1) as u32));
        }
        self.slots.write()[slot] = Some(dev);
        self.publish_bars();
    }

    /// Send the runs of the device windows that one BAR decodes whole to
    /// `sink`, now and whenever a Device(x) register, a configuration
    /// header or a reset changes them. Each run is a [`PciBar`] offset in
    /// this widget's window: the machine maps it over the window, so the
    /// processor reaches the function with no search of the bus and no
    /// Bridge lock. Everything else in a window (pages a BAR covers in
    /// part, an unclaimed address and its master time-out) still comes here.
    pub fn on_bar_runs(&self, sink: BarRunSink) {
        *self.bar_sink.write() = Some(sink);
        self.publish_bars();
    }

    fn publish_bars(&self) {
        let sink = self.bar_sink.read();
        let Some(sink) = sink.as_ref() else { return };
        let regs: Vec<u32> = {
            let st = self.st.lock();
            (0..DEVICES as u32).map(|s| st.r(reg::device(s))).collect()
        };
        let funcs: Vec<(usize, Arc<dyn PciBusDevice>)> =
            self.slots.read().iter().enumerate().filter_map(|(i, d)| Some((i, d.clone()?))).collect();
        let mut runs: Vec<BarRun> = Vec::new();
        for (s, &d) in regs.iter().enumerate() {
            let base = (d & DEV_WINDOW_BASE_MASK) << 20;
            let width = if d & DEV_64BIT_DEVICE != 0 { 8 } else { 4 };
            let swap = d & DEV_SWAP_PROGRAMMED != 0;
            let space = if d & DEV_MEMORY_SPACE != 0 { Space::Mem } else { Space::Io };
            let (off, size) = (device_window_offset(s as u32) as u64, device_window_size(s as u32) as u64);
            runs.extend(bar_runs(&funcs, space, off, base, size, |b| self.bar_entry(b, width, swap)));
        }
        sink(runs);
    }

    fn bar_entry(&self, b: BarAt, width: usize, swap: bool) -> Arc<dyn BusDevice> {
        let key = (b.slot, b.bar, width, swap, b.cpu_base);
        self.bar_entries
            .lock()
            .entry(key)
            .or_insert_with(|| {
                let lanes = Arc::new(Lanes { width, swap });
                Arc::new(PciBar::new(lanes, b.dev.clone(), b.bar, WINDOW_MASK as u64, b.cpu_base))
            })
            .clone()
    }

    /// Fit a flash part on chip select 0. `part` is addressed from the
    /// start of the part (16-bit big-endian cycles, `read16`/`write16`) and
    /// `size` is a power of two of at most 2 MB. The part repeats across its
    /// chip select. Which chip select answers at the window base follows
    /// the flash-select strap ([`Bridge::set_flash_strap`]); the Octane's
    /// boot flash is at small-window offset 0xc0_0000, physical 0x1fc0_0000.
    pub fn attach_flash(&self, part: Arc<dyn BusDevice>, size: u32) {
        self.fit_flash(0, part, size);
    }

    /// As [`Bridge::attach_flash`], on chip select 1.
    pub fn attach_flash_select1(&self, part: Arc<dyn BusDevice>, size: u32) {
        self.fit_flash(1, part, size);
    }

    fn fit_flash(&self, cs: usize, part: Arc<dyn BusDevice>, size: u32) {
        assert!(size.is_power_of_two() && size <= FLASH_SELECT_SIZE, "flash part must be a power of two of at most 2 MB");
        self.flash.write()[cs] = Some(FlashPart { dev: part, size });
    }

    /// The board's flash-select strap: set, chip select 1 answers at the
    /// base of the flash window. Reads back in the status register.
    pub fn set_flash_strap(&self, select1: bool) {
        self.flash_strap.store(select1, Ordering::Release);
        let mut st = self.st.lock();
        let s = st.r(reg::STATUS);
        st.set_r(reg::STATUS, if select1 { s | STATUS_FLASH_SELECT } else { s & !STATUS_FLASH_SELECT });
    }

    /// Tell the Bridge which widgets the Crossbow has a live link to. A DMA
    /// request to any other widget is dropped by the Crossbow, which the
    /// Bridge's request timer reports. The default is HEART alone.
    pub fn set_link_alive(&self, alive: Box<dyn Fn(u8) -> bool + Send + Sync>) {
        *self.link_alive.write() = alive;
    }

    /// Install the one-wire net behind the NIC register: the board's
    /// serial-number part, and the Ethernet part when this Bridge is the
    /// one the MAC's NIC hangs off. Each write of the timing fields runs
    /// one bit slot on it. With none installed the bus is empty and reads
    /// as an undriven, pulled-up line.
    pub fn set_microlan(&self, bus: w1::Bus) {
        *self.microlan.lock() = Some(bus);
    }

    /// Mark `slot`'s data phases 64 bits wide (its card drove REQ64# at
    /// reset). Affects the byte placement of its DMA. No machine marks a card
    /// 64 bits wide yet, so only the tests call it.
    #[cfg(test)]
    pub fn set_dma_64bit(&self, slot: u8, wide: bool) {
        let mut st = self.st.lock();
        st.dma_64bit = (st.dma_64bit & !(1 << slot)) | ((wide as u8) << slot);
    }

    /// Drive one of the chip's directly wired interrupt inputs (the GPIO
    /// pins), which share the eight pins with PCI. `pin` is 0..8; anything
    /// else is not a pin.
    pub fn set_wire_irq(&self, pin: u32, level: bool) {
        if pin as usize >= PINS {
            return;
        }
        self.irq_locked(|st, p| {
            st.gpio_pin_level = (st.gpio_pin_level & !(1 << pin)) | ((level as u8) << pin);
            self.update_pin(st, p, pin as usize);
        });
    }

    /// Reset the Bridge's own state: the register file, the translation
    /// table and the latched errors return to power-up. The interrupt
    /// inputs keep their levels, since the functions holding them are not
    /// in this reset and would never raise another edge; the pins are
    /// re-evaluated against the cleared enables.
    pub fn reset(&self) {
        let strap = self.flash_strap.load(Ordering::Acquire);
        let mut st = self.st.lock();
        st.reset_core(strap);
        st.pin_level = st.pci_pin_level | st.gpio_pin_level;
        drop(st);
        self.publish_bars();
    }

    /// The machine's reset reaching the PCI side: RST# returns every
    /// function's configuration header to power-up, so none decodes until
    /// firmware assigns its BARs again. The functions' own state is theirs
    /// to reset.
    pub fn reset_pci(&self) {
        for d in self.slots.read().iter().flatten() {
            d.config().reset();
        }
        self.publish_bars();
    }

    /// The Crossbow's link reset arriving. IRIX saves the control and link
    /// configuration registers around it and rebuilds the rest, so the whole
    /// core is taken as lost: registers, translation entries and latched
    /// errors return to power-up. The interrupt inputs keep their levels and
    /// nothing reaches the PCI side. Approximation: the core is reset once
    /// and released, not held until the link is re-established; what a core
    /// held in reset answers is not established.
    pub fn link_reset(&self) {
        log::debug!("bridge: inbound link reset");
        self.reset();
    }

    /// The register at slot base `off` (such as [`reg::CONTROL`]), for
    /// inspection.
    pub fn register(&self, off: u32) -> u32 {
        let st = self.st.lock();
        match off {
            reg::INT_STATUS => (st.pin_level as u32 & INT_DEVICE_PINS) | st.int_error,
            _ => st.r(off),
        }
    }

    // ---- interrupts ----

    /// Pins A and C reach the pin numbered as the slot, B and D that
    /// number with bit two flipped.
    fn pin_of(slot: usize, pin0: usize) -> usize {
        if pin0 & 1 != 0 {
            slot ^ 4
        } else {
            slot
        }
    }

    fn set_pci_line(&self, line: usize, level: bool) {
        if line >= DEVICES * 4 {
            return;
        }
        self.irq_locked(|st, p| {
            st.pci_lines[line] = level;
            // Several functions can share a pin: the pin is the OR of its lines.
            let pin = Self::pin_of(line / 4, line % 4);
            let any = (0..DEVICES * 4).any(|l| st.pci_lines[l] && Self::pin_of(l / 4, l % 4) == pin);
            st.pci_pin_level = (st.pci_pin_level & !(1 << pin)) | ((any as u8) << pin);
            self.update_pin(st, p, pin);
        });
    }

    fn update_pin(&self, st: &mut State, p: &mut Packets, pin: usize) {
        let level = (st.pci_pin_level | st.gpio_pin_level) >> pin & 1 != 0;
        let was = st.pin_level >> pin & 1 != 0;
        let enabled = st.r(reg::INT_ENABLE) >> pin & 1 != 0;
        st.pin_level = (st.pin_level & !(1 << pin)) | ((level as u8) << pin);
        // The chip sends on a transition, not while a level is held.
        if level && !was && enabled {
            self.send_interrupt(st, p, pin, true);
        } else if !level && was && enabled && st.r(reg::INT_MODE) >> pin & 1 != 0 {
            // A falling edge clears HEART's bit only when the mode bit asks;
            // otherwise the guest clears it in its own handler.
            self.send_interrupt(st, p, pin, false);
        }
    }

    /// A pin's interrupt goes to the destination pair's widget with the
    /// vector from that pin's host register. HEART is the only widget that
    /// takes it; a pair naming any other has nowhere to go. The packet is
    /// queued for `irq_locked` to send once the state lock is dropped.
    fn send_interrupt(&self, st: &State, p: &mut Packets, pin: usize, set: bool) {
        let host = st.r(reg::int_host(pin as u32));
        if host == 0 {
            // An unprogrammed destination is an interrupt nobody routed.
            log::debug!("bridge: interrupt pin {pin} changed with no destination programmed");
            return;
        }
        let dest = int_dest(st.r(reg::INT_DEST_UPPER), st.r(reg::INT_DEST_LOWER));
        // The pin's own host-address field replaces bits 47:38 of the
        // destination address.
        let field = ((host & INT_HOST_ADDRESS) >> 8) as u64;
        let address = (dest.address & !(0x3ffu64 << 38)) | field << 38;
        if !Self::heart_accepts(&dest, address, (host & INT_HOST_FIELD) as u8) {
            log::debug!("bridge: interrupt packet to widget {} address {address:#x} is not HEART's; dropped", dest.widget);
            return;
        }
        p.push(((host & INT_HOST_FIELD) as u8, set));
    }

    /// HEART takes an interrupt packet only at its interrupt-set offset
    /// (0x80), naming one of its 64 bits.
    fn heart_accepts(dest: &IntDest, address: u64, vector: u8) -> bool {
        dest.widget == WIDGET_HEART && address == HEART_INT_PACKET && vector < 64
    }

    /// Chip errors go to the host-error bit at the destination pair.
    fn raise_host_error(&self, st: &State, p: &mut Packets) {
        let number = (st.r(reg::HOST_ERR_FIELD) & INT_HOST_FIELD) as u8;
        let dest = int_dest(st.r(reg::INT_DEST_UPPER), st.r(reg::INT_DEST_LOWER));
        // An unrouted error is one nobody asked for.
        if !Self::heart_accepts(&dest, dest.address, number) {
            log::debug!("bridge: error interrupt with no usable destination; dropped");
            return;
        }
        p.push((number, true));
    }

    /// The first enabled error in the PCI group owns its capture registers
    /// until software clears the group; a later one only adds its status bit
    /// and the multiple-error flag. A disabled error leaves no trace.
    fn pci_error(&self, st: &mut State, p: &mut Packets, bit: u32, device: u32, pci_address: u64) {
        if st.r(reg::INT_ENABLE) & bit == 0 {
            return;
        }
        if st.int_error & INT_PCI_GROUP != 0 {
            st.int_error |= INT_MULTI_ERR;
        } else {
            st.set_r(reg::PCI_ERR_UPPER, (device << PCI_ERR_DEVICE_SHIFT) | ((pci_address >> 32) & 0xffff) as u32);
            st.set_r(reg::PCI_ERR_LOWER, pci_address as u32);
        }
        st.int_error |= bit;
        self.raise_host_error(st, p);
    }

    /// A read this chip sent for a device and the crossbar dropped: the
    /// request timer expires at once. The capture is the response-buffer
    /// error pair (inferred from the specification, not stated by it);
    /// the buffer field is zero because no response buffers are held.
    fn read_timeout(&self, st: &mut State, p: &mut Packets) {
        if st.r(reg::INT_ENABLE) & INT_XREAD_REQ_TIMEOUT == 0 {
            return;
        }
        if st.int_error & INT_RESP_BUF_GROUP != 0 {
            st.int_error |= INT_MULTI_ERR;
        } else {
            let (dev, addr) = (st.dma_device, st.dma_pci_address);
            st.set_r(reg::RESP_ERR_UPPER, dev << RESP_ERR_DEVICE_SHIFT);
            st.set_r(reg::RESP_ERR_LOWER, addr as u32);
        }
        st.int_error |= INT_XREAD_REQ_TIMEOUT;
        self.raise_host_error(st, p);
    }

    // ---- the register block ----

    /// An undefined address in the block posts and flags rather than
    /// faulting: a firmware self-test provokes one and fails on any
    /// synchronous exception.
    fn invalid_address(&self, st: &mut State, p: &mut Packets, off: u32, write: bool) {
        log::debug!(
            "bridge: {} of undefined register {off:#x}; flagging an invalid-address error",
            if write { "write" } else { "read" }
        );
        st.int_error |= INT_INVALID_ADDR;
        if st.r(reg::INT_ENABLE) & INT_INVALID_ADDR != 0 {
            self.raise_host_error(st, p);
        }
    }

    fn reg_read(&self, off: u32, size: usize) -> Option<u64> {
        // Not implemented: no answer rather than a made-up one.
        if off == RESERVED_DATA_HALF || size != 4 {
            return None;
        }
        self.irq_locked(|st, p| {
            let Some(info) = (if off % 8 == SLOT_DATA { reg_info(off - SLOT_DATA) } else { None }) else {
                self.invalid_address(st, p, off, false);
                return Some(0xffff_ffff);
            };
            Some(match info.access {
                Access::Live => (st.pin_level as u32 & INT_DEVICE_PINS | st.int_error) as u64,
                Access::Wo => info.reset as u64,
                _ => st.r(off - SLOT_DATA) as u64,
            })
        })
    }

    fn reg_write(&self, off: u32, size: usize, val: u64) -> bool {
        if off == RESERVED_DATA_HALF {
            return size == 4 && val == 0;
        }
        if size != 4 {
            return false;
        }
        let val = val as u32;
        self.irq_locked(|st, p| {
            let Some(info) = (if off % 8 == SLOT_DATA { reg_info(off - SLOT_DATA) } else { None }) else {
                self.invalid_address(st, p, off, true);
                return true;
            };
            let base = off - SLOT_DATA;
            match info.access {
                Access::Ro | Access::Live => {
                    log::debug!("bridge: write of {val:#x} to read-only register {base:#x}");
                }
                Access::Wo => {
                    if base == reg::INT_RESET {
                        // A group's strobe clears that group's errors and no other's.
                        if val & INT_RESET_PCI_GRP != 0 {
                            st.int_error &= !INT_PCI_GROUP;
                        }
                        if val & INT_RESET_REQ_DSP_GRP != 0 {
                            st.int_error &= !INT_INVALID_ADDR;
                        }
                        if val & INT_RESET_RESP_BUF_GRP != 0 {
                            st.int_error &= !INT_RESP_BUF_GROUP;
                        }
                        if val & INT_RESET_MULTI_ERR != 0 {
                            st.int_error &= !INT_MULTI_ERR;
                        }
                    }
                }
                Access::Rw => {
                    let old = st.r(base);
                    let new = (old & !info.writable) | (val & info.writable);
                    st.set_r(base, new);
                    if base == reg::NIC {
                        self.onewire_slot(st, val);
                    } else if base == reg::INT_ENABLE {
                        // A pin already high that is newly enabled sends now.
                        let newly = (new & !old) & st.pin_level as u32 & INT_DEVICE_PINS;
                        for pin in 0..PINS {
                            if newly >> pin & 1 != 0 {
                                self.send_interrupt(st, p, pin, true);
                            }
                        }
                    }
                }
            }
            true
        })
    }

    /// One bit slot, reported finished inside the write the guest polls after.
    fn onewire_slot(&self, st: &mut State, val: u32) {
        let pulse = (val & NIC_PULSE_MASK) >> NIC_PULSE_SHIFT;
        let sample = (val & NIC_SAMPLE_MASK) >> NIC_SAMPLE_SHIFT;
        let level = match self.microlan.lock().as_mut() {
            Some(bus) => w1::slot::slot(bus, pulse, sample),
            // An empty bus: nothing answers a reset, so the line stays high;
            // a short pulse releases the wire (high) and one in between
            // holds it low.
            None => pulse >= w1::slot::RESET_PULSE || pulse < w1::slot::ONE_PULSE,
        };
        let held = st.r(reg::NIC) & !NIC_LEVEL;
        st.set_r(reg::NIC, held | NIC_DONE | if level { NIC_LEVEL } else { 0 });
    }

    // ---- the translation table ----

    /// Only the upper half of an entry reads back through the table itself.
    fn ate_read(&self, off: u32, size: usize) -> Option<u64> {
        let i = (off / 8) as usize;
        (size == 4 && off % 8 == SLOT_DATA && i < ATE_ENTRIES).then(|| self.st.lock().ate[i] >> 32)
    }

    /// The read-only view of the entries' lower halves.
    fn ate_low_read(&self, off: u32, size: usize) -> Option<u64> {
        let i = (off / 4) as usize;
        (size == 4 && i < ATE_ENTRIES).then(|| self.st.lock().ate[i] as u32 as u64)
    }

    /// Entries are written whole as 64-bit quantities; a 32-bit write would
    /// leave half an entry behind.
    fn ate_write(&self, off: u32, size: usize, val: u64) -> bool {
        let i = (off / 8) as usize;
        if size != 8 || off % 8 != 0 || i >= ATE_ENTRIES {
            return false;
        }
        self.st.lock().ate[i] = val;
        true
    }

    // ---- configuration space ----

    /// Undo the guest's byte-lane aim (offset xor (4 - width)): what makes
    /// a byte read of a vendor ID return the vendor's low byte.
    fn lane_correct(reg: u32, size: usize) -> u32 {
        (reg & !3) | ((reg & 3) ^ (4 - size as u32))
    }

    /// The function a type 0 configuration offset selects, and the corrected
    /// register byte offset within it.
    fn cfg_target(&self, off: u32, size: usize) -> (usize, u32, Option<Arc<dyn PciBusDevice>>) {
        let slot = ((off >> 12) & 7) as usize;
        let func = (off >> 8) & 7;
        let reg = Self::lane_correct(off & 0xff, size);
        let dev = if func == 0 { self.slots.read()[slot].clone() } else { None };
        (slot, reg, dev)
    }

    /// No device drove device select: a master time-out. The cycle's PCI
    /// address is the device's ID select, one line per device from line 24,
    /// then the register's word.
    fn config_master_timeout(&self, slot: usize, reg: u32) {
        self.irq_locked(|st, p| {
            self.pci_error(st, p, INT_PCI_MASTER_TIMEOUT, slot as u32, (1u64 << (24 + slot)) | (reg & 0x7fc) as u64)
        });
    }

    fn cfg_read(&self, off: u32, size: usize) -> Option<u64> {
        if size > 4 {
            return None;
        }
        let (slot, reg, dev) = self.cfg_target(off, size);
        let Some(dev) = dev else {
            self.config_master_timeout(slot, reg);
            // An unclaimed read fails back to the requester: guests read
            // empty slots only inside a bus-error guard.
            return None;
        };
        let dword = dev.config().read((reg >> 2) as usize) as u64;
        Some((dword >> (8 * (reg & 3))) & (u64::MAX >> (64 - 8 * size)))
    }

    /// A store posts: it has no response to carry a failure back.
    fn cfg_write(&self, off: u32, size: usize, val: u64) -> bool {
        if size > 4 {
            return false;
        }
        let (slot, reg, dev) = self.cfg_target(off, size);
        let Some(dev) = dev else {
            self.config_master_timeout(slot, reg);
            return true;
        };
        let cfg = dev.config();
        let i = (reg >> 2) as usize;
        let shift = 8 * (reg & 3);
        let mask = (u64::MAX >> (64 - 8 * size)) << shift;
        cfg.write_masked(i, (val << shift) as u32, mask as u32);
        true
    }

    // ---- device windows ----

    /// One PCI cycle of `host.len()` bytes (at most the bus width) at window
    /// offset `woff`. `host` is in CPU address order, most significant byte
    /// first. With the programmed swap set, host byte k is PCI byte k; clear,
    /// a 32-bit device sees it at k ^ 3 within its word and a 64-bit one at
    /// 7 - k within its doubleword (the specification's big-endian tables 65
    /// to 80), and an access narrower than the device moves to
    /// offset ^ (width - size).
    #[allow(clippy::too_many_arguments)]
    fn pio_cycle(&self, slot: usize, space: Space, base: u32, woff: u32, width: usize, swap: bool, host: &mut [u8], write: bool) -> bool {
        let n = host.len();
        let target = base.wrapping_add(if swap { woff } else { woff ^ (width - n) as u32 });
        let claimed = self.slots.read().iter().flatten().find_map(|d| {
            let (bar, boff) = d.config().decode(space, target)?;
            Some((d.clone(), bar, boff))
        });
        let Some((dev, bar, boff)) = claimed else {
            // Nobody drove device select: a master time-out.
            self.irq_locked(|st, p| self.pci_error(st, p, INT_PCI_MASTER_TIMEOUT, slot as u32, target as u64));
            return false;
        };
        Self::bar_cycle(&*dev, bar, boff, swap, host, write);
        true
    }

    /// The cycle `pio_cycle` found a target for: `host` (CPU order) to or
    /// from BAR `bar` of `dev` at `boff`, lanes as `swap` says.
    fn bar_cycle(dev: &dyn PciBusDevice, bar: usize, boff: u32, swap: bool, host: &mut [u8], write: bool) {
        let n = host.len();
        // PCI bytes in address order; a function sees little-endian values.
        let mut pci = [0u8; 8];
        if write {
            for i in 0..n {
                pci[i] = host[if swap { i } else { n - 1 - i }];
            }
        }
        let mut done = 0;
        while done < n {
            let chunk = (n - done).min(4);
            let at = boff + done as u32;
            if write {
                let v = pci[done..done + chunk].iter().rev().fold(0u32, |a, &b| a << 8 | b as u32);
                dev.bar_write(bar, at, v, chunk);
            } else {
                let v = dev.bar_read(bar, at, chunk);
                for i in 0..chunk {
                    pci[done + i] = (v >> (8 * i)) as u8;
                }
            }
            done += chunk;
        }
        if !write {
            for i in 0..n {
                host[i] = pci[if swap { i } else { n - 1 - i }];
            }
        }
    }

    /// A processor access of `size` bytes to slot `slot`'s window at window
    /// offset `woff`; `val` is `Some` for a write. A doubleword to a 32-bit
    /// device is two word cycles, lower address first, carrying host bytes 0
    /// to 3 and then 4 to 7. A read that fails on its first cycle ends
    /// there; a write posts, so both cycles go out and it never fails.
    fn window_access(&self, slot: usize, woff: u32, size: usize, val: Option<u64>) -> Option<u64> {
        let dev_reg = self.st.lock().r(reg::device(slot as u32));
        let base = (dev_reg & DEV_WINDOW_BASE_MASK) << 20;
        let width = if dev_reg & DEV_64BIT_DEVICE != 0 { 8 } else { 4 };
        let swap = dev_reg & DEV_SWAP_PROGRAMMED != 0;
        let space = if dev_reg & DEV_MEMORY_SPACE != 0 { Space::Mem } else { Space::Io };
        Self::cycles(width, woff, size, val, |woff, host, write| {
            self.pio_cycle(slot, space, base, woff, width, swap, host, write)
        })
    }

    /// A processor access of `size` bytes at window offset `woff` as the
    /// PCI cycles a `width`-byte device takes, each made by `cycle(offset,
    /// host bytes, write)`.
    fn cycles(width: usize, woff: u32, size: usize, val: Option<u64>, mut cycle: impl FnMut(u32, &mut [u8], bool) -> bool) -> Option<u64> {
        let write = val.is_some();
        let mut host = [0u8; 8];
        if let Some(v) = val {
            for (k, b) in host.iter_mut().enumerate().take(size) {
                *b = (v >> (8 * (size - 1 - k))) as u8;
            }
        }
        let ok = if size > width {
            let (a, b) = host.split_at_mut(4);
            let first = cycle(woff, a, write);
            if first || write {
                let second = cycle(woff + 4, &mut b[..4], write);
                first && second
            } else {
                false
            }
        } else {
            cycle(woff, &mut host[..size], write)
        };
        if write {
            return Some(0);
        }
        ok.then(|| host[..size].iter().fold(0u64, |a, &b| a << 8 | b as u64))
    }

    // ---- flash ----

    /// The part a window offset selects, and the offset within it. A chip
    /// select with nothing on it fails, as an unpopulated address does.
    fn flash_part(&self, woff: u32) -> Option<(Arc<dyn BusDevice>, u32)> {
        let strap = self.flash_strap.load(Ordering::Acquire);
        let select1 = (woff >= FLASH_SELECT_SIZE) ^ strap;
        let parts = self.flash.read();
        let p = parts[select1 as usize].as_ref()?;
        // A part repeats across its chip select above the address lines it decodes.
        Some((p.dev.clone(), woff % FLASH_SELECT_SIZE % p.size))
    }

    /// The port is 16 bits wide, so a read of any size is the part's words
    /// read one cycle at a time with the requested byte lanes kept.
    fn flash_read(&self, woff: u32, size: usize) -> Option<u64> {
        let end = woff + size as u32;
        let mut result = 0u64;
        let mut w = woff & !1;
        while w < end {
            let (dev, at) = self.flash_part(w)?;
            let word = dev.read16(at as u64);
            // A fitted part with floating outputs leaves pulled-up data.
            let word = if word.is_ok() { word.data } else { 0xffff };
            for (half, byte) in [(0u32, (word >> 8) as u8), (1, word as u8)] {
                if (woff..end).contains(&(w + half)) {
                    result = (result << 8) | byte as u64;
                }
            }
            w += 2;
        }
        Some(result)
    }

    /// The port takes 16-bit writes only, on account of the parts' command
    /// sequences, and completes none while the control register's flash
    /// write enable is clear. A refused write fails the transaction (the
    /// Octane board selects this; how the chip reports it is not
    /// established).
    fn flash_write(&self, woff: u32, size: usize, val: u64) -> bool {
        let Some((dev, at)) = self.flash_part(woff) else { return false };
        if self.st.lock().r(reg::CONTROL) & CONTROL_FLASH_WRITE_EN == 0 || size != 2 || woff & 1 != 0 {
            return false;
        }
        dev.write16(at as u64, val as u16) == BUS_OK
    }

    // ---- DMA ----

    /// Map a PCI address of slot `slot` to the XIO address it reaches and the
    /// placement of its bytes. `None`: not memory this Bridge reaches.
    fn translate(&self, st: &State, slot: usize, addr: u64) -> Option<(u64, Placement, u8)> {
        let dev = st.r(reg::device(slot as u32));
        let wide = st.dma_64bit >> slot & 1 != 0;
        let placement_for = |swap_bit: u32| match (dev & swap_bit != 0, wide) {
            (true, _) => Placement::Identity,
            (false, false) => Placement::Word,
            (false, true) => Placement::Doubleword,
        };
        // The 64-bit direct window, decoded ahead of the 31-bit one, which
        // would drop the widget at bits 63:60. The widget is named by the
        // address itself, its low 48 bits pass through, and the direct swap
        // bit steers the bytes.
        if addr >> 32 != 0 {
            let widget = ((addr >> 60) & 0xf) as u8;
            if widget == 0 {
                log::debug!("bridge: 64-bit DMA names widget 0, which the crossbar cannot answer");
                return None;
            }
            return Some((addr & ((1 << 48) - 1), placement_for(DEV_SWAP_DIRECT), widget));
        }
        let addr = addr as u32;
        // Local space never crosses the swapper.
        if addr < PCI_TRANSLATED {
            log::debug!("bridge: DMA to PCI-local address {addr:#x} is not carried anywhere");
            return None;
        }
        let page_shift = if st.r(reg::CONTROL) & CONTROL_PAGE_16K != 0 { 14 } else { 12 };
        let page_mask = (1u64 << page_shift) - 1;
        let placement = placement_for(if addr < PCI_DIRECT { DEV_SWAP_TRANSLATED } else { DEV_SWAP_DIRECT });
        if addr >= PCI_DIRECT {
            let direct = st.r(reg::DIRECT_MAP);
            let mut offset = (((direct & DIRECT_OFFSET) as u64) << 31) | (addr & 0x7fff_ffff) as u64;
            // Applies only with a zero offset field; the bit resets set.
            if direct & DIRECT_OFFSET == 0 && direct & DIRECT_ADD_512MB != 0 {
                offset += DIRECT_512MB;
            }
            let widget = ((direct & DIRECT_WIDGET) >> DIRECT_WIDGET_SHIFT) as u8;
            if widget == 0 {
                return None;
            }
            return Some((offset, placement, widget));
        }
        // Through the table; the index is the page number in the window.
        let index = ((addr - PCI_TRANSLATED) >> page_shift) as usize;
        if index >= ATE_ENTRIES {
            // External static RAM would supply the rest; this board has none.
            log::debug!("bridge: DMA to {addr:#x} needs translation entry {index}, past the {ATE_ENTRIES} held");
            return None;
        }
        let ate = st.ate[index];
        if ate & ATE_VALID == 0 {
            log::debug!("bridge: DMA to {addr:#x} uses translation entry {index}, which is not valid");
            return None;
        }
        let widget = ((ate >> ATE_TARGET_WIDGET_SHIFT) & 0xf) as u8;
        (widget != 0).then_some((((ate & ATE_ADDRESS) & !page_mask) | (addr as u64 & page_mask), placement, widget))
    }

    /// A function's transfer, split at 4 KB boundaries (the smallest
    /// translation page) so each chunk has one translation.
    fn dma(&self, slot: usize, addr: u64, mut buf: DmaBuf) -> PciDmaStatus {
        let len = match &buf {
            DmaBuf::Read(b) => b.len(),
            DmaBuf::Write(b) => b.len(),
        };
        let mut done = 0usize;
        let mut ok = true;
        while done < len {
            let a = addr.wrapping_add(done as u64);
            let step = (0x1000 - (a as usize & 0xfff)).min(len - done);
            let target = {
                let mut st = self.st.lock();
                st.dma_pci_address = a;
                st.dma_device = slot as u32;
                self.translate(&st, slot, a)
            };
            // Only HEART is memory. The Crossbow drops a request to a widget
            // it has no live link to (1 to 7 are illegal, 15 is this Bridge's
            // own port), and the Bridge's request timer reports a dropped
            // read. A refused translation, or a read that reaches HEART and
            // finds no memory, is a silent master abort.
            let mut dropped = false;
            let good = match target {
                None => false,
                Some((_, _, w)) if w != WIDGET_HEART => {
                    // Approximation: QEMU routes peer DMA to a fitted widget;
                    // no supported Octane guest is known to need it.
                    dropped = w == WIDGET_BRIDGE || !(self.link_alive.read())(w);
                    false
                }
                Some((xio, p, _)) => match &mut buf {
                    DmaBuf::Read(b) => self.placed_read(xio, p, &mut b[done..done + step]),
                    DmaBuf::Write(b) => self.placed_write(xio, p, &b[done..done + step]),
                },
            };
            if !good {
                ok = false;
                if let DmaBuf::Read(b) = &mut buf {
                    b[done..done + step].fill(0);
                    if dropped {
                        self.irq_locked(|st, p| self.read_timeout(st, p));
                    }
                }
            }
            done += step;
        }
        if ok { PciDmaStatus::Ok } else { PciDmaStatus::MasterAbort }
    }

    /// Memory bytes at `addr`, as every DMA master moves them
    /// (`physical::read_bytes`); a failed read leaves the buffer zero.
    fn mem_read(&self, addr: u64, out: &mut [u8]) -> bool {
        let ok = crate::physical::read_bytes(&*self.mem, addr, out) == crate::traits::BUS_OK;
        if !ok {
            out.fill(0);
        }
        ok
    }

    fn mem_write(&self, addr: u64, data: &[u8]) -> bool {
        crate::physical::write_bytes(&*self.mem, addr, data) == crate::traits::BUS_OK
    }

    /// Read `out.len()` bytes at PCI-order position `xio`, undoing the
    /// placement: the PCI byte at address p is memory byte p ^ mask.
    fn placed_read(&self, xio: u64, placement: Placement, out: &mut [u8]) -> bool {
        let mask = placement.mask();
        if mask == 0 {
            return self.mem_read(xio, out);
        }
        let unit = mask as usize + 1;
        if xio & mask == 0 && out.len() % unit == 0 {
            // Whole units: the bytes come straight into place and each unit
            // is reversed where it lies.
            let ok = self.mem_read(xio, out);
            out.chunks_exact_mut(unit).for_each(<[u8]>::reverse);
            return ok;
        }
        let lo = xio & !mask;
        let hi = (xio + out.len() as u64 + mask) & !mask;
        let mut span = vec![0u8; (hi - lo) as usize];
        let ok = self.mem_read(lo, &mut span);
        for (i, b) in out.iter_mut().enumerate() {
            *b = if ok { span[(((xio + i as u64) ^ mask) - lo) as usize] } else { 0 };
        }
        ok
    }

    fn placed_write(&self, xio: u64, placement: Placement, data: &[u8]) -> bool {
        let mask = placement.mask();
        if mask == 0 {
            return self.mem_write(xio, data);
        }
        let unit = mask as usize + 1;
        if xio & mask == 0 && data.len() % unit == 0 {
            // Whole units: reverse each into a copy and store it in one go.
            let mut stack = [0u8; 4096];
            let mut heap = Vec::new();
            let swapped: &mut [u8] = if data.len() <= stack.len() {
                &mut stack[..data.len()]
            } else {
                heap.resize(data.len(), 0);
                &mut heap
            };
            swapped.copy_from_slice(data);
            swapped.chunks_exact_mut(unit).for_each(<[u8]>::reverse);
            return self.mem_write(xio, swapped);
        }
        let lo = xio & !mask;
        let hi = (xio + data.len() as u64 + mask) & !mask;
        let mut span = vec![0u8; (hi - lo) as usize];
        for (i, &b) in data.iter().enumerate() {
            span[(((xio + i as u64) ^ mask) - lo) as usize] = b;
        }
        if xio == lo && xio + data.len() as u64 == hi {
            return self.mem_write(lo, &span);
        }
        // Partial units: write only the bytes covered, so a neighbour
        // another master is changing is not overwritten with a stale copy.
        let mut ok = true;
        for i in 0..data.len() as u64 {
            let p = (xio + i) ^ mask;
            ok &= self.mem_write(p, &span[(p - lo) as usize..(p - lo) as usize + 1]);
        }
        ok
    }

    // ---- the window ----

    /// A processor read of `size` bytes at `off` in the small window. `None`
    /// is a bus error.
    fn read(&self, off: u32, size: usize) -> Option<u64> {
        match off {
            0..=0xffff => self.reg_read(off, size),
            ATE_BASE..=0x10fff => self.ate_read(off - ATE_BASE, size),
            ATE_LOW_BASE..=0x11fff => self.ate_low_read(off - ATE_LOW_BASE, size),
            _ if (CFG0_BASE..CFG0_BASE + CFG0_SIZE).contains(&off) => self.cfg_read(off - CFG0_BASE, size),
            // Unpopulated SSRAM floats high: the PROM sizes it with an
            // unguarded probe, and all-ones probes as zero.
            _ if (SSRAM_BASE..SSRAM_BASE + SSRAM_SIZE).contains(&off) => Some(u64::MAX >> (64 - 8 * size)),
            _ if (DEVICE_WINDOW_BASE..FLASH_BASE).contains(&off) => {
                let slot = window_slot(off)?;
                self.window_access(slot as usize, off - device_window_offset(slot), size, None)
            }
            FLASH_BASE..=0xff_ffff => self.flash_read(off - FLASH_BASE, size),
            _ => None,
        }
    }

    /// As [`Bridge::read`]; the result is a bus status.
    fn write(&self, off: u32, size: usize, val: u64) -> u32 {
        let ok = match off {
            0..=0xffff => self.reg_write(off, size, val),
            ATE_BASE..=0x10fff => self.ate_write(off - ATE_BASE, size, val),
            _ if (CFG0_BASE..CFG0_BASE + CFG0_SIZE).contains(&off) => self.cfg_write(off - CFG0_BASE, size, val),
            _ if (SSRAM_BASE..SSRAM_BASE + SSRAM_SIZE).contains(&off) => true,
            _ if (DEVICE_WINDOW_BASE..FLASH_BASE).contains(&off) => match window_slot(off) {
                Some(s) => self.window_access(s as usize, off - device_window_offset(s), size, Some(val)).is_some(),
                None => false,
            },
            FLASH_BASE..=0xff_ffff => self.flash_write(off - FLASH_BASE, size, val),
            _ => false,
        };
        let device_reg = (reg::device(0)..reg::device(DEVICES as u32)).contains(&(off & !7));
        if device_reg || (CFG0_BASE..CFG0_BASE + CFG0_SIZE).contains(&off) {
            self.publish_bars();
        }
        if ok {
            BUS_OK
        } else {
            BUS_ERR
        }
    }
}

impl Bridge {
    /// The interrupt destination the guest programmed: (target widget,
    /// address), for the trace.
    pub fn interrupt_destination(&self) -> (u32, u64) {
        let st = self.st.lock();
        let d = int_dest(st.r(reg::INT_DEST_UPPER), st.r(reg::INT_DEST_LOWER));
        (d.widget as u32, d.address)
    }
}

/// A pin of the chip, 0..8, driven as a level (see [`Bridge::pin_line`],
/// which holds the Bridge weakly).
impl InterruptController for Bridge {
    fn set_line(&self, line: u32, level: bool) {
        self.set_wire_irq(line, level);
    }
}

impl crate::traits::Resettable for Bridge {
    /// Power-on: the chip's own state, as a reset. The functions behind it
    /// are not reset here; see [`Bridge::reset_pci`].
    fn power_on(&self) {
        self.reset();
    }
}

/// The widget window is decoded from the full physical address.
const WINDOW_MASK: u32 = WIDGET_SMALL_SIZE - 1;

/// A device window's lanes, as its Device(x) register sets them: the
/// device's width and the programmed swap. What [`PciBar`] entries drive
/// a function through, the same cycles `window_access` makes once it has
/// found the target.
struct Lanes {
    width: usize,
    swap: bool,
}

impl Lanes {
    fn access(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, val: Option<u64>) -> u64 {
        let (width, swap) = (self.width, self.swap);
        Bridge::cycles(width, off, size, val, |o, host, write| {
            let boff = if swap { o } else { o ^ (width - host.len()) as u32 };
            Bridge::bar_cycle(f, bar, boff, swap, host, write);
            true
        })
        .unwrap_or(0)
    }
}

impl PciLanes for Lanes {
    fn read(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize) -> u64 {
        self.access(f, bar, off, size, None)
    }
    fn write(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, val: u64) {
        self.access(f, bar, off, size, Some(val));
    }
}

impl BusDevice for Bridge {
    fn read8(&self, addr: u64) -> BusRead8 {
        self.read(addr as u32 & WINDOW_MASK, 1).map_or(BusRead8::err(), |v| BusRead8::ok(v as u8))
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write(addr as u32 & WINDOW_MASK, 1, val as u64)
    }

    fn read16(&self, addr: u64) -> BusRead16 {
        self.read(addr as u32 & WINDOW_MASK, 2).map_or(BusRead16::err(), |v| BusRead16::ok(v as u16))
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write(addr as u32 & WINDOW_MASK, 2, val as u64)
    }

    fn read32(&self, addr: u64) -> BusRead32 {
        self.read(addr as u32 & WINDOW_MASK, 4).map_or(BusRead32::err(), |v| BusRead32::ok(v as u32))
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.write(addr as u32 & WINDOW_MASK, 4, val as u64)
    }

    fn read64(&self, addr: u64) -> BusRead64 {
        self.read(addr as u32 & WINDOW_MASK, 8).map_or(BusRead64::err(), BusRead64::ok)
    }

    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write(addr as u32 & WINDOW_MASK, 8, val)
    }
}

/// A read-only flash image for [`Bridge::attach_flash`]. Reads are 16-bit
/// big-endian words of the image, padded with erased bytes (0xff) to
/// `size`.
///
/// Approximation: the part has no command interface. A write is dropped
/// without a fault (and counted in [`RawFlash::dropped_writes`]), so the
/// PROM's identification and programming sequences find no chip answering
/// and take their failure paths.
pub struct RawFlash {
    data: Vec<u8>,
    dropped: AtomicU64,
}

impl RawFlash {
    pub fn new(image: &[u8], size: usize) -> Self {
        assert!(size.is_power_of_two() && image.len() <= size, "image must fit a power-of-two part");
        let mut data = vec![0xff; size];
        data[..image.len()].copy_from_slice(image);
        Self { data, dropped: AtomicU64::new(0) }
    }

    pub fn size(&self) -> u32 {
        self.data.len() as u32
    }

    pub fn dropped_writes(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

impl BusDevice for RawFlash {
    fn read16(&self, addr: u64) -> BusRead16 {
        let a = addr as usize & !1;
        match self.data.get(a..a + 2) {
            Some(b) => BusRead16::ok(u16::from_be_bytes([b[0], b[1]])),
            None => BusRead16::err(),
        }
    }

    fn write16(&self, _addr: u64, _val: u16) -> u32 {
        self.dropped.fetch_add(1, Ordering::Relaxed);
        BUS_OK
    }
}

#[cfg(test)]
#[path = "bridge_tests.rs"]
mod tests;
