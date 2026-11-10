//! The Octane's onboard Bridge: the XIO-to-PCI host bridge at widget 0xf. The
//! same chip sits on the PCI option boards ([`super::xio_pci`]), each on a
//! port of its own.
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
//! | 0x028000   | type 1 configuration space, 4 KB                  |
//! | 0x080000   | external SSRAM (unpopulated)                      |
//! | 0x200000   | device windows: two of 2 MB, then six of 1 MB     |
//! | 0xc00000   | flash, two 2 MB chip selects on a 16-bit port     |
//!
//! Registers are 32 bits in the data (upper) half of an eight-byte slot.
//! The register, translation, byte-lane and interrupt behaviour follows the
//! published Bridge ASIC specification (rev 4.0, `docs/datasheets/octane/`). Where the
//! specification is silent the choice is marked as an approximation:
//!
//! * Arbitration, read response buffers, the SSRAM parity and request
//!   timer registers are storage. Nothing arbitrates, nothing times out.
//! * Error capture covers the PCI group (master time-out), the dropped DMA
//!   read and invalid register addresses. Parity, retry and target-abort
//!   errors are never raised.
//! * The translation table is the chip's internal 128 entries; no external
//!   SRAM extends it.
//! * Through the widget's 2 GB window the first 16 MB is the small window
//!   and 0x4000_0000 on is the 32-bit alias of PCI memory (specification
//!   p. 52); the rest of that window is a bus error. The 64-bit alias at
//!   0x8000_0000 and PCI I/O at 4 GB, which only the 64 GB window reaches,
//!   are not modelled. Which device a master time-out through the alias
//!   names in the error capture is not in the specification; iris names 0.
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
//! # The XBridge
//!
//! The IP35 boards' XBridge holds two PCI heads, each this chip's register
//! file, windows, interrupt pins and DMA windows at a crossbar port of its
//! own, built as [`Part::XBridge`]. Where the head differs the part decides,
//! in the slow paths only: identification, the Control straps and writable
//! masks, a 1024-entry translation table of which 512 entries map a window
//! with a byte-swap alias at bit 29, a swap bit at PCI bit 55 for 64-bit
//! DMA, the force-interrupt and multiple-error registers, the 64-bit and
//! I/O aliases at 2 GB and 4 GB, and what an unclaimed or undefined access
//! answers. A processor reaches a head through [`Bridge::head_window`],
//! whose decode is the head's own; the Bridge's own window decode is
//! untouched. The head's behaviour follows hw/pci-host/sgi-xbridge.c in
//! jamesbraid/qemu-sgi-origami, not a published specification; it has
//! no flash, SSRAM or diagnostic engines.
//!
//! Ported from hw/pci-host/sgi-bridge.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, RwLock};

use crate::dev::w1;
use crate::devlog::LogModule;
use crate::irq::InterruptController;
use crate::pci::BarMapping;
use crate::pci::{CpuWindow, DevFn, IntxRouting, PciBus, PciDma, PciDmaStatus, PciHost, Space};
use crate::pci::{BarDevices, ByteOrder};
use crate::snapshot::{field, field_u32, field_u32s, field_u64, field_u64_vec, field_u8, field_u8s, StateTable};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Saveable, BUS_ERR, BUS_OK};
use crate::xtalk::Upstream;

/// The widget number of this chip's own port on the Crossbow.
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
    /// The XBridge's multiple-error register: every error it latched since
    /// the groups were last cleared.
    pub const INT_MULTIPLE: u32 = 0x170;
    /// The XBridge's force registers: a write to the first eight sends
    /// pin n's packet, to the next eight only while pin n is high.
    pub const FORCE_ALWAYS: u32 = 0x180;
    pub const FORCE_PIN: u32 = 0x1c0;
    pub const FORCE_PIN_LAST: u32 = 0x1f8;
}

/// A register's name for the `bridge` log, from its slot base.
fn reg_name(base: u32) -> String {
    use reg::*;
    let name = match base {
        IDENTIFICATION => "identification",
        STATUS => "status",
        ERR_UPPER => "error upper",
        ERR_LOWER => "error lower",
        CONTROL => "control",
        REQUEST_TIMEOUT => "request timeout",
        INT_DEST_UPPER => "interrupt destination upper",
        INT_DEST_LOWER => "interrupt destination lower",
        ERR_CMD_WORD => "error command word",
        LINK_CONFIG => "link config",
        TARGET_FLUSH => "target flush",
        DIRECT_MAP => "direct map",
        ARBITRATION => "arbitration",
        NIC => "nic",
        BUS_TIMEOUT => "bus timeout",
        PCI_TYPE1_CONFIG => "type 1 config",
        PCI_ERR_UPPER => "pci error upper",
        PCI_ERR_LOWER => "pci error lower",
        INT_STATUS => "interrupt status",
        INT_ENABLE => "interrupt enable",
        INT_RESET => "interrupt reset",
        INT_MODE => "interrupt mode",
        INT_DEVICE => "interrupt device",
        HOST_ERR_FIELD => "host error field",
        RRB_EVEN => "rrb even",
        RRB_ODD => "rrb odd",
        RRB_STATUS => "rrb status",
        RRB_CLEAR => "rrb clear",
        0x130..=0x168 => return format!("interrupt host {}", (base - 0x130) / 8),
        0x200..=0x238 => return format!("device {}", (base - 0x200) / 8),
        0x240..=0x278 => return format!("device {} flush", (base - 0x240) / 8),
        _ => "",
    };
    format!("{base:#05x} {name}")
}

/// Window geometry, as offsets in the small window.
pub const ATE_BASE: u32 = 0x01_0000;
pub const ATE_LOW_BASE: u32 = 0x01_1000;
pub const CFG0_BASE: u32 = 0x02_0000;
pub const CFG0_SIZE: u32 = 8 * 0x1000;
pub const TYPE1_BASE: u32 = 0x02_8000;
pub const TYPE1_SIZE: u32 = 0x1000;
pub const SSRAM_BASE: u32 = 0x08_0000;
pub const SSRAM_SIZE: u32 = 0x08_0000;
pub const DEVICE_WINDOW_BASE: u32 = 0x20_0000;
pub const FLASH_BASE: u32 = 0xc0_0000;
pub const FLASH_SELECT_SIZE: u32 = 0x20_0000;
pub const DEVICES: usize = 8;
/// The 32-bit alias of PCI memory space, as an offset in the widget's XIO
/// space: PCI address 0 on (specification p. 52).
pub const PIO32_ALIAS: u32 = 0x4000_0000;
const PIO32_ALIAS_SIZE: u32 = 0x4000_0000;
/// The widget's 2 GB window, and the CPU window index the alias takes
/// after the eight device windows.
pub const LONG_WINDOW_SIZE: u64 = 0x8000_0000;
const LONG_MASK: u64 = LONG_WINDOW_SIZE - 1;
const ALIAS_WINDOW: usize = DEVICES;
pub const PINS: usize = 8;
pub const ATE_ENTRIES: usize = 128;
/// The XBridge head's table, and the entries of it its mapped window
/// reaches: 512 pages, at 0x4000_0000 and again with bytes swapped at
/// 0x6000_0000. Its lower halves read back at 0x12000, eight bytes apart.
pub const XBRIDGE_ATE_ENTRIES: usize = 1024;
const XBRIDGE_ATE_MAPPED: usize = 512;
const XBRIDGE_ATE_LOW_BASE: u32 = 0x01_2000;
const XBRIDGE_ATE_END: u32 = 0x01_4000;
const XBRIDGE_MAPPED_SPAN: u32 = 0x80_0000;
const XBRIDGE_MAPPED_SWAP: u32 = 1 << 29;
/// The address bits an XBridge table entry keeps.
const XBRIDGE_ATE_KEPT: u64 = (1 << 48) - 1;
/// An XBridge table entry's own swap bit.
const XBRIDGE_ATE_SWAP: u64 = 1 << 5;
/// The 64 KB below the mapped window the XBridge answers as its flush
/// target: a read finds zeros and a write is taken and dropped.
const XBRIDGE_FLUSH: std::ops::Range<u32> = 0x3fff_0000..0x4000_0000;
/// PCI address bit 55 asks the XBridge to swap a 64-bit direct transfer.
const XBRIDGE_DMA64_SWAP: u64 = 1 << 55;
/// The XBridge head's aliases of PCI space: 32-bit memory from 1 GB,
/// 64-bit memory from 2 GB and I/O from 4 GB.
const XBRIDGE_ALIASES: [(u64, u64, Space); 3] =
    [(0x4000_0000, 0x4000_0000, Space::Mem), (0x8000_0000, 0x4000_0000, Space::Mem), (0x1_0000_0000, 0x1_0000_0000, Space::Io)];

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
const REGS_DEFINED: u32 = 0x300;
const REG_SLOTS: usize = (REGS_DEFINED / 8) as usize;
/// The one reserved data half this chip answers specially.
const RESERVED_DATA_HALF: u32 = 0x19c;

const ID_REVISION_SHIFT: u32 = 28;
const ID_PART_SHIFT: u32 = 12;
const ID_MANUFACTURER_SHIFT: u32 = 1;
/// The silicon revision a Bridge reports until the board names its own.
const REVISION: u32 = 1;

/// Which member of the Bridge family a [`Bridge`] is: see the module
/// documentation for where they differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The Octane's and the IP27 boards' Bridge.
    Bridge,
    /// One PCI head of the IP35 boards' XBridge.
    XBridge,
}

impl Part {
    /// The part number and manufacturer in the identification word.
    const fn identity(self) -> (u32, u32) {
        match self {
            Part::Bridge => (0xc002, 0x036),
            Part::XBridge => (0xd002, 0x024),
        }
    }

    /// Translation entries the chip holds.
    pub const fn ate_entries(self) -> usize {
        match self {
            Part::Bridge => ATE_ENTRIES,
            Part::XBridge => XBRIDGE_ATE_ENTRIES,
        }
    }
}

const STATUS_FLASH_SELECT: u32 = 1 << 6;
const STATUS_PCI_MODE: u32 = 1 << 5;

pub const CONTROL_FLASH_WRITE_EN: u32 = 1 << 31;
pub const CONTROL_MEM_SWAP: u32 = 1 << 22;
pub const CONTROL_IO_SWAP: u32 = 1 << 23;
pub const CONTROL_PAGE_16K: u32 = 1 << 21;
/// Control 27:24, RST_PIN_N(3:0): the four PCI reset pins, active low, all
/// high (deasserted) at reset. Established: the bits, their reset value and
/// that a 1 drives the pin high (Bridge ASIC specification rev 4.0, Control
/// register; the pin list names four `BUS_RSTN` outputs, "PCI Bus reset
/// signals").
///
/// Which device each pin serves is not documented. The O2 Device Driver
/// Programming Guide (007-3443-002, pciio(D3), p. 550) describes
/// `pciio_reset()` as activating the reset line of one card "without
/// affecting other devices on the PCI bus", so a pin is taken to reset one
/// device. Pin `n` resets the functions at PCI device `n` and no others: the
/// pin-to-device wiring is inferred, from the pin count and the driver
/// interface alone. A write that takes a pin from 1 to 0 is logged. A core
/// reset and a link reset do not pulse the pins, because the specification
/// does not say that either does.
const CONTROL_RST_PINS: u32 = 0xf << 24;
const CONTROL_RESET: u32 = (1 << 30) | (1 << 29) | (1 << 28) | CONTROL_RST_PINS | (3 << 12) | (1 << 9) | (0x1f << 4) | 0xf;
/// The widget ID the chip puts on the packets it sends.
const CONTROL_WIDGET_ID: u32 = 0xf;
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
/// A transfer through a translation entry that is not valid, or past the
/// ones the chip has; the XBridge alone raises it.
pub const INT_PAGE_FAULT: u32 = 1 << 30;
const INT_SSRAM_GROUP: u32 = INT_PAGE_FAULT | (1 << 16);
const INT_ENABLE_WRITABLE: u32 = 0x7fff_ffff;
const INT_RESET_GROUPS: u32 = 0x7f;
const INT_RESET_PCI_GRP: u32 = 1 << 0;
const INT_RESET_SSRAM_GRP: u32 = 1 << 1;
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

/// The XBridge head's registers where they are not the Bridge's: Control's
/// PCI speed strap (bits 7:4) and the bits it lacks read only, narrower
/// fields elsewhere, and the multiple-error and force-interrupt registers.
/// `None` is the Bridge's register.
fn xbridge_reg_info(slot_base: u32) -> Option<Option<RegInfo>> {
    use reg::*;
    use Access::*;
    Some(match slot_base {
        CONTROL => info(Rw, 0x0f00_320f, 0x0fe1_f20f),
        DIRECT_MAP => info(Rw, DIRECT_ADD_512MB, 0xf3_ffff),
        BUS_TIMEOUT => info(Rw, BUS_TIMEOUT_RESET, 0x1f_03ff),
        INT_DEST_UPPER => info(Rw, 0, 0xf_ffff),
        TEST_PIN_CONTROL => info(Rw, 0, 0xff_ffff),
        INT_MULTIPLE => info(Live, 0, 0),
        0x178 => info(Ro, 0, 0),
        FORCE_ALWAYS..=FORCE_PIN_LAST if slot_base % 8 == 0 => info(Wo, 0, 0),
        0x200..=0x238 if slot_base % 8 == 0 => info(Rw, device_reset((slot_base - 0x200) / 8), 0x1eef_ffff),
        _ => return None,
    })
}

/// Decode, reset value and writable mask from one table, by slot base.
fn reg_info(part: Part, slot_base: u32) -> Option<RegInfo> {
    use reg::*;
    use Access::*;
    if part == Part::XBridge {
        if let Some(info) = xbridge_reg_info(slot_base) {
            return info;
        }
    }
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

/// The widget a packet says it came from: the ID the guest gave the chip in
/// Control bits 3:0.
fn source(st: &State) -> u8 {
    (st.r(reg::CONTROL) & CONTROL_WIDGET_ID) as u8
}

struct State {
    reg: [u32; REG_SLOTS],
    /// Sized for the largest part; a part uses its own count.
    ate: [u64; XBRIDGE_ATE_ENTRIES],
    /// Latched error bits of the interrupt status register (31:8).
    int_error: u32,
    /// The eight pins' driven levels, which the status register reports.
    /// They follow the level the bus combines for each pin and never latch.
    pin_level: u8,
    /// Slots whose data phases are 64 bits wide.
    dma_64bit: u8,
    /// The transfer the last DMA translation served, for error capture.
    dma_pci_address: u64,
    dma_device: u32,
    /// The XBridge's multiple-error register ([`reg::INT_MULTIPLE`]).
    multiple: u32,
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
    fn reset_core(&mut self, part: Part, flash_strap: bool, revision: u32, control_strap: u32) {
        for (i, slot) in self.reg.iter_mut().enumerate() {
            *slot = reg_info(part, i as u32 * 8).map_or(0, |r| r.reset);
        }
        let (number, manufacturer) = part.identity();
        self.set_r(
            reg::IDENTIFICATION,
            ((revision & 0xf) << ID_REVISION_SHIFT) | (number << ID_PART_SHIFT) | (manufacturer << ID_MANUFACTURER_SHIFT) | 1,
        );
        if flash_strap {
            let s = self.r(reg::STATUS);
            self.set_r(reg::STATUS, s | STATUS_FLASH_SELECT);
        }
        let c = self.r(reg::CONTROL);
        self.set_r(reg::CONTROL, c | control_strap);
        self.int_error = 0;
        self.multiple = 0;
        // Whole entries, not just valid bits: a stale page address behind a
        // clear valid bit is a working translation waiting for one write.
        self.ate.fill(0);
    }
}

/// A flash chip select's part: a bus device addressed from the start of the
/// part, which answers 16-bit big-endian reads and writes.
struct FlashPart {
    dev: Arc<dyn BusDevice>,
    size: u32,
}

/// An interrupt packet a locked section decided to send: from the widget
/// Control names, to `widget` at `address`, setting or clearing `vector`.
struct Packet {
    source: u8,
    widget: u8,
    address: u64,
    vector: u8,
    set: bool,
}

type Packets = Vec<Packet>;

/// The onboard Bridge. Hand it out as the `BusDevice` of the widget's small
/// window: `Arc<Bridge>` coerces to `Arc<dyn BusDevice>`. As an
/// [`InterruptController`] its line is a pin, 0..8, driven as a level.
pub struct Bridge {
    st: Mutex<State>,
    /// Held across "decide a packet, send it", never with `st`.
    /// Nothing but the host's interrupt input is called under it.
    irq_order: Mutex<()>,
    /// The host chips this Bridge's transfers and interrupt packets can go
    /// to, by the widget number the guest programs.
    up: Upstream,
    /// The functions on the Bridge's PCI bus, their BAR windows and their
    /// interrupt pins.
    bus: PciBus,
    /// This Bridge, for the DMA a function is given: it must not keep the
    /// Bridge alive.
    me: Weak<Bridge>,
    flash: RwLock<[Option<FlashPart>; 2]>,
    flash_strap: AtomicBool,
    /// The silicon revision in the identification word, bits 31:28.
    revision: AtomicU32,
    /// The devices made for BAR mappings, once each.
    bar_devices: BarDevices,
    /// The one-wire net behind the NIC register; none fitted reads as an
    /// empty bus.
    microlan: Mutex<Option<w1::Bus>>,
    /// Whether the Crossbow has a live link to a widget; default the hosts.
    link_alive: RwLock<Box<dyn Fn(u8) -> bool + Send + Sync>>,
    /// The Crossbow port this Bridge is on: 0xf for the onboard one.
    widget: AtomicU8,
    part: Part,
    /// Control bits the board straps, kept through a reset: the XBridge
    /// head's PCI bus speed in 7:4.
    control_strap: AtomicU32,
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

/// Where a DMA translation sends a transfer.
#[derive(Clone, Copy)]
enum Route {
    /// To the widget's XIO address, with the bytes placed so.
    Host(u64, Placement, u8),
    /// To the XBridge's flush target.
    Flush,
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
    /// A Bridge that masters memory and interrupts the host chips of `up`,
    /// each addressed as it states and chosen by the widget number the guest
    /// programs, its PCI bus wired as `wiring`. Registers start at their
    /// power-up values.
    pub fn new(up: Upstream, wiring: &'static IntxRouting) -> Arc<Self> {
        Self::with_part(Part::Bridge, up, wiring)
    }

    /// As [`Bridge::new`], for `part` of the family.
    pub fn with_part(part: Part, up: Upstream, wiring: &'static IntxRouting) -> Arc<Self> {
        let mut st = State {
            reg: [0; REG_SLOTS],
            ate: [0; XBRIDGE_ATE_ENTRIES],
            int_error: 0,
            pin_level: 0,
            dma_64bit: 0,
            dma_pci_address: 0,
            dma_device: 0,
            multiple: 0,
        };
        st.reset_core(part, false, REVISION, 0);
        let hosts = (0..16u8).filter(|&w| up.get(w).is_some()).fold(0u16, |m, w| m | 1 << w);
        let this = Arc::new_cyclic(|me: &Weak<Bridge>| Self {
            st: Mutex::new(st),
            irq_order: Mutex::new(()),
            up,
            bus: PciBus::new(0, wiring, me.clone()),
            me: me.clone(),
            flash: RwLock::new(Default::default()),
            flash_strap: AtomicBool::new(false),
            revision: AtomicU32::new(REVISION),
            microlan: Mutex::new(None),
            bar_devices: Default::default(),
            link_alive: RwLock::new(Box::new(move |w| hosts.checked_shr(w as u32).is_some_and(|m| m & 1 != 0))),
            widget: AtomicU8::new(WIDGET_BRIDGE),
            part,
            control_strap: AtomicU32::new(0),
        });
        this.bus.set_host_windows(&|| this.windows());
        this
    }

    /// The functions on the Bridge's PCI bus.
    pub fn bus(&self) -> &PciBus {
        &self.bus
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_mutex_fn;
        register_mutex_fn("bridge::state", self, |b| b.st.is_locked());
        register_mutex_fn("bridge::irq_order", self, |b| b.irq_order.is_locked());
        register_mutex_fn("bridge::microlan", self, |b| b.microlan.is_locked());
    }

    /// Run `f` on the state, then send the packets it decided on. The
    /// order lock spans both, so packets for one pin leave in the order
    /// they were decided; the host is called with the state lock dropped.
    fn irq_locked<R>(&self, f: impl FnOnce(&mut State, &mut Packets) -> R) -> R {
        let _order = self.irq_order.lock();
        let mut packets = Packets::new();
        let ret = f(&mut self.st.lock(), &mut packets);
        for Packet { source, widget, address, vector, set } in packets {
            match self.up.get(widget) {
                Some(h) if h.link_up() => {
                    if !h.irq.packet(source, address, vector as u32, set) {
                        log::debug!("bridge: widget {widget} refused interrupt packet {address:#x} vector {vector}");
                    }
                }
                _ => log::debug!("bridge: interrupt packet to widget {widget} dropped; no live link to a host"),
            }
        }
        ret
    }

    /// The eight device windows as the Device(x) registers have them: each
    /// opens onto PCI memory or I/O space at the base its register names.
    /// After them, the 32-bit alias of PCI memory in the 2 GB window.
    fn windows(&self) -> Vec<CpuWindow> {
        let st = self.st.lock();
        let aliases = match self.part {
            Part::Bridge => vec![CpuWindow { space: Space::Mem, cpu: PIO32_ALIAS as u64, pci: 0, len: PIO32_ALIAS_SIZE as u64, mask: LONG_MASK }],
            Part::XBridge => XBRIDGE_ALIASES.iter().map(|&(cpu, len, space)| CpuWindow { space, cpu, pci: 0, len, mask: u64::MAX }).collect(),
        };
        (0..DEVICES as u32)
            .map(|s| {
                let d = st.r(reg::device(s));
                CpuWindow {
                    space: if d & DEV_MEMORY_SPACE != 0 { Space::Mem } else { Space::Io },
                    cpu: device_window_offset(s) as u64,
                    pci: (d & DEV_WINDOW_BASE_MASK) << 20,
                    len: device_window_size(s) as u64,
                    mask: WINDOW_MASK as u64,
                }
            })
            .chain(aliases)
            .collect()
    }

    /// How a CPU access to window `window` reaches a function, from its
    /// Device(x) register: the programmed swap gives a byte stream, otherwise
    /// the device's width gives word values.
    /// Through the alias, Control's memory swap bit decides, as the
    /// programmed swap does for a window, and the device counts as 32-bit.
    fn order(&self, window: usize) -> ByteOrder {
        let st = self.st.lock();
        if window >= ALIAS_WINDOW {
            return Self::alias_order(st.r(reg::CONTROL), Self::alias_space(window));
        }
        Self::order_of(st.r(reg::device(window as u32)))
    }

    /// The space alias window `window` opens onto: the XBridge's third alias
    /// is I/O, every other memory.
    fn alias_space(window: usize) -> Space {
        if window == ALIAS_WINDOW + 2 { Space::Io } else { Space::Mem }
    }

    /// Control's memory swap bit steers the memory aliases, its I/O swap bit
    /// the I/O alias.
    fn alias_order(control: u32, space: Space) -> ByteOrder {
        let swap = if space == Space::Io { CONTROL_IO_SWAP } else { CONTROL_MEM_SWAP };
        if control & swap != 0 { ByteOrder::Stream } else { ByteOrder::values(4) }
    }

    fn order_of(device: u32) -> ByteOrder {
        if device & DEV_SWAP_PROGRAMMED != 0 {
            ByteOrder::Stream
        } else {
            ByteOrder::values(if device & DEV_64BIT_DEVICE != 0 { 8 } else { 4 })
        }
    }

    /// The bus device the machine maps over the part of the Bridge's window
    /// that `run` covers: the function's BAR through the order the window's
    /// Device(x) register sets, so the processor reaches the function with no
    /// search of the bus and no Bridge lock. The run is in the widget's small
    /// window, so a mapped device sees offsets in it. The same configuration
    /// gives back the same device, so updating the mappings does not make the
    /// physical bus keep a new one. A run in the alias is in the 2 GB window,
    /// and its device sees offsets in that.
    pub fn bar_device(&self, run: &BarMapping) -> Arc<dyn BusDevice> {
        let mask = if run.window as usize == ALIAS_WINDOW { LONG_MASK } else { WINDOW_MASK as u64 };
        self.bar_devices.get(run, mask, self.order(run.window as usize))
    }

    /// What the machine maps over the widget's 2 GB window.
    pub fn long_window(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(LongWindow(self.clone()))
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

    /// The silicon revision the identification word reports in bits 31:28
    /// (the board's chip: 3 on the 175 MHz Octane, 4 on a measured Octane2).
    /// It survives a reset, as the chip's mask does.
    pub fn set_revision(&self, revision: u32) {
        self.revision.store(revision, Ordering::Release);
        let mut st = self.st.lock();
        let id = st.r(reg::IDENTIFICATION);
        st.set_r(reg::IDENTIFICATION, (id & !(0xf << ID_REVISION_SHIFT)) | ((revision & 0xf) << ID_REVISION_SHIFT));
    }

    /// Strap Control bits the guest cannot write (the XBridge head's PCI
    /// bus speed, bits 7:4). They survive a reset, as the board's wiring does.
    pub fn set_control_strap(&self, bits: u32) {
        self.control_strap.store(bits, Ordering::Release);
        let mut st = self.st.lock();
        let c = st.r(reg::CONTROL);
        st.set_r(reg::CONTROL, c | bits);
    }

    /// Tell the Bridge which widgets the Crossbow has a live link to. A DMA
    /// request to any other widget is dropped by the Crossbow, which the
    /// Bridge's request timer reports. The default is the host chip alone.
    pub fn set_link_alive(&self, alive: Box<dyn Fn(u8) -> bool + Send + Sync>) {
        *self.link_alive.write() = alive;
    }

    /// The Crossbow port this Bridge is on, when it is not the onboard one
    /// at 0xf: a Bridge on an option board's own port. A DMA request that
    /// names this port is dropped by the Crossbow, as one naming the
    /// onboard Bridge's port is.
    pub fn set_widget(&self, widget: u8) {
        self.widget.store(widget, Ordering::Release);
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

    /// Reset the Bridge's own state: the register file, the translation
    /// table and the latched errors return to power-up. The interrupt
    /// inputs keep their levels, since the functions holding them are not
    /// in this reset and would never raise another edge.
    pub fn reset(&self) {
        let strap = self.flash_strap.load(Ordering::Acquire);
        let revision = self.revision.load(Ordering::Acquire);
        let control = self.control_strap.load(Ordering::Acquire);
        self.st.lock().reset_core(self.part, strap, revision, control);
        self.bus.set_host_windows(&|| self.windows());
    }

    /// The machine's reset reaching the PCI side: RST# on the bus. Every
    /// function's configuration header returns to power-up, so none decodes
    /// until firmware assigns its BARs again.
    pub fn reset_pci(&self) {
        self.bus.reset();
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

    /// Pin `pin` is now at `level`, as the bus combined every function and
    /// board wire routed to it. The chip sends on a transition, not while a
    /// level is held.
    fn update_pin(&self, st: &mut State, p: &mut Packets, pin: usize, level: bool) {
        let was = st.pin_level >> pin & 1 != 0;
        let enabled = st.r(reg::INT_ENABLE) >> pin & 1 != 0;
        if level != was {
            crate::dlog!(LogModule::Bridge, "bridge: pin {pin} {}{}", if level { "rose" } else { "fell" },
                if enabled { "" } else { " (not enabled)" });
        }
        st.pin_level = (st.pin_level & !(1 << pin)) | ((level as u8) << pin);
        if level && !was && enabled {
            self.send_interrupt(st, p, pin, true);
        } else if !level && was && enabled && st.r(reg::INT_MODE) >> pin & 1 != 0 {
            // A falling edge clears the host's bit only when the mode bit asks;
            // otherwise the guest clears it in its own handler.
            self.send_interrupt(st, p, pin, false);
        }
    }

    /// A pin's interrupt goes to the destination pair's widget with the
    /// vector from that pin's host register. The host chip is the only
    /// widget that takes it; a pair naming any other has nowhere to go. The packet is
    /// queued for `irq_locked` to send once the state lock is dropped.
    fn send_interrupt(&self, st: &State, p: &mut Packets, pin: usize, set: bool) {
        let host = st.r(reg::int_host(pin as u32));
        if host == 0 && self.part == Part::Bridge {
            // An unprogrammed destination is an interrupt nobody routed.
            log::debug!("bridge: interrupt pin {pin} changed with no destination programmed");
            return;
        }
        let dest = int_dest(st.r(reg::INT_DEST_UPPER), st.r(reg::INT_DEST_LOWER));
        // The pin's own host-address field replaces bits 47:38 of the
        // destination address.
        let field = ((host & INT_HOST_ADDRESS) >> 8) as u64;
        let address = (dest.address & !(0x3ffu64 << 38)) | field << 38;
        let vector = (host & INT_HOST_FIELD) as u8;
        match self.up.get(dest.widget) {
            Some(h) if h.host.accepts(dest.widget, address, vector) => {
                crate::dlog!(LogModule::Bridge, "bridge: pin {pin} {} vector {vector} of widget {}", if set { "sets" } else { "clears" }, dest.widget);
                p.push(Packet { source: source(st), widget: dest.widget, address, vector, set });
            }
            Some(_) => log::debug!("bridge: interrupt packet to widget {} address {address:#x} vector {vector} is not one its host takes; dropped", dest.widget),
            None => log::debug!("bridge: interrupt packet to widget {}, which is not a host; dropped", dest.widget),
        }
    }

    /// Chip errors go to the host-error bit at the destination pair.
    fn raise_host_error(&self, st: &State, p: &mut Packets) {
        let number = (st.r(reg::HOST_ERR_FIELD) & INT_HOST_FIELD) as u8;
        let dest = int_dest(st.r(reg::INT_DEST_UPPER), st.r(reg::INT_DEST_LOWER));
        // An unrouted error is one nobody asked for.
        match self.up.get(dest.widget) {
            Some(h) if h.host.accepts(dest.widget, dest.address, number) => {
                p.push(Packet { source: source(st), widget: dest.widget, address: dest.address, vector: number, set: true })
            }
            _ => log::debug!("bridge: error interrupt with no usable destination; dropped"),
        }
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
    fn invalid_address(&self, st: &mut State, p: &mut Packets, off: u64, write: bool) {
        log::debug!(
            "bridge: {} of undefined register {off:#x}; flagging an invalid-address error",
            if write { "write" } else { "read" }
        );
        let first = st.int_error & INT_INVALID_ADDR == 0;
        st.int_error |= INT_INVALID_ADDR;
        if self.part == Part::XBridge {
            // The XBridge captures the first such address and interrupts
            // for it alone.
            st.multiple |= INT_INVALID_ADDR;
            if !first {
                return;
            }
            st.set_r(reg::ERR_UPPER, (off >> 32) as u32);
            st.set_r(reg::ERR_LOWER, off as u32);
        }
        if st.r(reg::INT_ENABLE) & INT_INVALID_ADDR != 0 {
            self.raise_host_error(st, p);
        }
    }

    fn reg_read(&self, off: u32, size: usize) -> Option<u64> {
        // Not implemented: no answer rather than a made-up one.
        if (off == RESERVED_DATA_HALF && self.part == Part::Bridge) || size != 4 {
            return None;
        }
        self.irq_locked(|st, p| {
            let Some(info) = (if off % 8 == SLOT_DATA { reg_info(self.part, off - SLOT_DATA) } else { None }) else {
                self.invalid_address(st, p, off as u64, false);
                return Some(0xffff_ffff);
            };
            Some(match info.access {
                Access::Live if off - SLOT_DATA == reg::INT_MULTIPLE => st.multiple as u64,
                Access::Live => (st.pin_level as u32 & INT_DEVICE_PINS | st.int_error) as u64,
                Access::Wo => info.reset as u64,
                _ => st.r(off - SLOT_DATA) as u64,
            })
        })
    }

    fn reg_write(&self, off: u32, size: usize, val: u64) -> bool {
        if off == RESERVED_DATA_HALF && self.part == Part::Bridge {
            return size == 4 && val == 0;
        }
        if size != 4 {
            return false;
        }
        let val = val as u32;
        let mut asserted = 0u32;
        let ok = self.irq_locked(|st, p| {
            let Some(info) = (if off % 8 == SLOT_DATA { reg_info(self.part, off - SLOT_DATA) } else { None }) else {
                self.invalid_address(st, p, off as u64, true);
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
                        let mut clear = 0;
                        if val & INT_RESET_PCI_GRP != 0 {
                            clear |= INT_PCI_GROUP;
                        }
                        // Only the XBridge raises the group's bits.
                        if val & INT_RESET_SSRAM_GRP != 0 && self.part == Part::XBridge {
                            clear |= INT_SSRAM_GROUP;
                        }
                        if val & INT_RESET_REQ_DSP_GRP != 0 {
                            clear |= INT_INVALID_ADDR;
                        }
                        if val & INT_RESET_RESP_BUF_GRP != 0 {
                            clear |= INT_RESP_BUF_GROUP;
                        }
                        if val & INT_RESET_MULTI_ERR != 0 {
                            clear |= INT_MULTI_ERR;
                        }
                        st.int_error &= !clear;
                        st.multiple = if val & INT_RESET_MULTI_ERR != 0 { 0 } else { st.multiple & !clear };
                    } else if (reg::FORCE_ALWAYS..=reg::FORCE_PIN_LAST).contains(&base) {
                        let pin = ((base - reg::FORCE_ALWAYS) / 8 % PINS as u32) as usize;
                        if st.r(reg::INT_ENABLE) >> pin & 1 == 0 {
                            crate::dlog!(LogModule::Bridge, "bridge: force of masked pin {pin} ignored");
                        } else if base < reg::FORCE_PIN || st.pin_level >> pin & 1 != 0 {
                            self.send_interrupt(st, p, pin, true);
                        }
                    }
                }
                Access::Rw if base == reg::NIC && self.part == Part::XBridge && self.microlan.lock().is_none() => {
                    crate::dlog!(LogModule::Bridge, "bridge: XBridge NIC has no MicroLAN fitted; write of {val:#x} ignored");
                }
                Access::Rw => {
                    let old = st.r(base);
                    let new = (old & !info.writable) | (val & info.writable);
                    st.set_r(base, new);
                    // Control 27:24 going from 1 to 0 asserts RST# (see
                    // CONTROL_RST_PINS).
                    if base == reg::CONTROL && self.part == Part::Bridge {
                        asserted = (old & !new & CONTROL_RST_PINS) >> 24;
                    }
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
        });
        // With the state lock dropped: the functions reset themselves.
        for pin in (0..4).filter(|pin| asserted >> pin & 1 != 0) {
            log::warn!("bridge: Control drops reset pin {pin}: resetting PCI device {pin} (pin wiring is inferred)");
            self.bus.reset_device(pin);
        }
        ok
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
    fn cfg_target(off: u32, size: usize) -> (DevFn, u32) {
        let at = DevFn { device: ((off >> 12) & 7) as u8, function: ((off >> 8) & 7) as u8 };
        (at, Self::lane_correct(off & 0xff, size))
    }

    /// No device drove device select: a master time-out. The cycle's PCI
    /// address is the device's ID select, one line per device from line 24,
    /// then the register's word.
    fn config_master_timeout(&self, slot: u8, reg: u32) {
        self.config_error(slot as u32, (1u64 << (24 + slot)) | (reg & 0x7fc) as u64);
    }

    fn config_error(&self, device: u32, pci_address: u64) {
        self.irq_locked(|st, p| self.pci_error(st, p, INT_PCI_MASTER_TIMEOUT, device, pci_address));
    }

    /// The bus and function a type 1 configuration offset selects, and the
    /// corrected register byte offset. The PCI Type 1 Configuration register
    /// names the bus (bits 23:16) and device (bits 15:11); the aperture
    /// offset gives the function and register (specification tables 26 and
    /// 43). A type 1 cycle on the Bridge's bus is claimed only by a
    /// PCI-to-PCI bridge whose secondary to subordinate range holds the bus
    /// number, so the bus is None when none does. The last element is the
    /// cycle's PCI address (type 1 format, AD[1:0] = 01), for the error
    /// capture.
    fn type1_target(&self, off: u32, size: usize) -> (Option<PciBus>, DevFn, u32, u64) {
        let sel = self.st.lock().r(reg::PCI_TYPE1_CONFIG);
        let bus_number = (sel >> 16) as u8;
        let at = DevFn { device: ((sel >> 11) & 0x1f) as u8, function: ((off >> 8) & 7) as u8 };
        let reg = Self::lane_correct(off & 0xff, size);
        let address = (sel & TYPE1_WRITABLE) | (at.function as u32) << 8 | (reg & 0xfc) | 1;
        (self.bus.type1_target(bus_number), at, reg, address as u64)
    }

    fn type1_read(&self, off: u32, size: usize) -> Option<u64> {
        if size > 4 {
            return None;
        }
        let (bus, at, reg, address) = self.type1_target(off, size);
        let read = bus.and_then(|b| b.config_read(at, reg as usize, size));
        if read.is_none() {
            // Which device the capture names for a type 1 cycle is not in
            // the specification; the device number the cycle carried.
            self.config_error(at.device as u32, address);
        }
        read.map(u64::from)
    }

    fn type1_write(&self, off: u32, size: usize, val: u64) -> bool {
        if size > 4 {
            return false;
        }
        let (bus, at, reg, address) = self.type1_target(off, size);
        if !bus.is_some_and(|b| b.config_write(at, reg as usize, size, val as u32)) {
            self.config_error(at.device as u32, address);
        }
        true
    }

    fn type1_fault(&self, off: u32, size: usize) -> Option<String> {
        if size > 4 {
            return None;
        }
        let (bus, _, _, address) = self.type1_target(off, size);
        let n = address >> 16 & 0xff;
        match bus {
            None => Some(format!("PCI type 1 config master timeout: no bridge reaches bus {n}")),
            // Behind a bridge an unclaimed cycle master-aborts there and the
            // Bridge sees it complete.
            Some(_) => None,
        }
    }

    fn cfg_read(&self, off: u32, size: usize) -> Option<u64> {
        if size > 4 {
            return None;
        }
        let (at, reg) = Self::cfg_target(off, size);
        let read = self.bus.config_read(at, reg as usize, size);
        if read.is_none() {
            self.config_master_timeout(at.device, reg);
            // An unclaimed read fails back to the requester: guests read
            // empty slots only inside a bus-error guard.
        }
        read.map(u64::from)
    }

    /// Why a configuration access failed, when the cause is that nothing
    /// answered: the PROM and the kernel probe every slot and rely on the
    /// time-out.
    fn cfg_fault(&self, off: u32, size: usize) -> Option<String> {
        // A transfer wider than a word is refused before any slot is chosen.
        if size > 4 {
            return None;
        }
        let (at, _) = Self::cfg_target(off, size);
        if self.bus.function(at).is_some() {
            return None;
        }
        let DevFn { device: slot, function: func } = at;
        Some(if !self.bus.occupied(slot) {
            format!("PCI config master timeout: empty slot {slot} function {func}")
        } else {
            format!("PCI config master timeout: slot {slot} has no function {func}")
        })
    }

    /// A store posts: it has no response to carry a failure back.
    fn cfg_write(&self, off: u32, size: usize, val: u64) -> bool {
        if size > 4 {
            return false;
        }
        let (at, reg) = Self::cfg_target(off, size);
        if !self.bus.config_write(at, reg as usize, size, val as u32) {
            self.config_master_timeout(at.device, reg);
        }
        true
    }

    // ---- device windows ----

    /// A processor access of `size` bytes to slot `slot`'s window at window
    /// offset `woff`; `val` is `Some` for a write. The bus finds the BAR the
    /// address reaches and the window's byte order carries the bytes. Where
    /// nothing drives device select the cycle ends in a master time-out: a read
    /// fails, and a write posts and never does. A doubleword to a 32-bit device
    /// is two word cycles, and a read that fails on its first ends there while
    /// a write sends both.
    fn window_access(&self, slot: usize, woff: u32, size: usize, val: Option<u64>) -> Option<u64> {
        let dev_reg = self.st.lock().r(reg::device(slot as u32));
        let base = (dev_reg & DEV_WINDOW_BASE_MASK) << 20;
        let space = if dev_reg & DEV_MEMORY_SPACE != 0 { Space::Mem } else { Space::Io };
        let order = Self::order_of(dev_reg);
        if let Some((f, bar, off)) = self.bus.decode(space, base.wrapping_add(woff)) {
            return Some(match val {
                None => order.read(f, bar, off, size),
                Some(v) => {
                    order.write(f, bar, off, size, v);
                    0
                }
            });
        }
        if self.part == Part::XBridge {
            return Some(self.xbridge_unclaimed(space, base.wrapping_add(woff), size, val));
        }
        // The cycle's PCI address is where the byte order sends it: a byte stream as
        // given, word values with the access moved to the device's own byte position.
        let width = if dev_reg & DEV_64BIT_DEVICE != 0 { 8 } else { 4 };
        let aim = |woff: u32, n: usize| base.wrapping_add(if order == ByteOrder::Stream { woff } else { woff ^ (width - n) as u32 });
        let miss = |target: u32| self.irq_locked(|st, p| self.pci_error(st, p, INT_PCI_MASTER_TIMEOUT, slot as u32, target as u64));
        if size > width {
            miss(aim(woff, 4));
            if val.is_some() {
                miss(aim(woff + 4, 4));
            }
        } else {
            miss(aim(woff, size));
        }
        val.map(|_| 0)
    }

    /// A processor access to PCI memory address `pci` through the 32-bit
    /// alias. As through a window, nothing driving device select is a
    /// master time-out: a read fails and a write posts.
    fn alias_access(&self, space: Space, pci: u32, size: usize, val: Option<u64>) -> Option<u64> {
        let order = Self::alias_order(self.st.lock().r(reg::CONTROL), space);
        if let Some((f, bar, off)) = self.bus.decode(space, pci) {
            return Some(match val {
                None => order.read(f, bar, off, size),
                Some(v) => {
                    order.write(f, bar, off, size, v);
                    0
                }
            });
        }
        if self.part == Part::XBridge {
            return Some(self.xbridge_unclaimed(space, pci, size, val));
        }
        let aim = |a: u32, n: usize| if order == ByteOrder::Stream { a } else { a ^ (4 - n.min(4)) as u32 };
        self.config_error(0, aim(pci, size) as u64);
        val.map(|_| 0)
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
    /// read one cycle at a time with the requested byte order kept.
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
    fn translate(&self, st: &mut State, slot: usize, addr: u64) -> Option<Route> {
        match self.part {
            Part::Bridge => Self::translate_bridge(st, slot, addr).map(|(xio, p, w)| Route::Host(xio, p, w)),
            Part::XBridge => Self::translate_xbridge(st, slot, addr),
        }
    }

    /// The XBridge head's translation. Its table maps 512 pages at
    /// 0x4000_0000 and the same pages with bytes swapped at 0x6000_0000; an
    /// entry's own swap bit swaps too, and an entry that is not valid or not
    /// held latches a page fault. PCI address bit 55 swaps a 64-bit direct
    /// transfer, the Device(x) direct swap bit a 32-bit one. The 64 KB below
    /// the mapped window is the flush target.
    fn translate_xbridge(st: &mut State, slot: usize, addr: u64) -> Option<Route> {
        let dev = st.r(reg::device(slot as u32));
        let wide = st.dma_64bit >> slot & 1 != 0;
        let placement = |swap: bool| match (swap, wide) {
            (true, _) => Placement::Identity,
            (false, false) => Placement::Word,
            (false, true) => Placement::Doubleword,
        };
        if addr >> 32 != 0 {
            let widget = ((addr >> 60) & 0xf) as u8;
            if widget == 0 {
                log::debug!("bridge: 64-bit DMA names widget 0, which the crossbar cannot answer");
                return None;
            }
            return Some(Route::Host(addr & ((1 << 48) - 1), placement(addr & XBRIDGE_DMA64_SWAP != 0), widget));
        }
        let addr = addr as u32;
        if XBRIDGE_FLUSH.contains(&addr) {
            return Some(Route::Flush);
        }
        let window = addr & !XBRIDGE_MAPPED_SWAP;
        if (PCI_TRANSLATED..PCI_TRANSLATED + XBRIDGE_MAPPED_SPAN).contains(&window) {
            let page_shift = if st.r(reg::CONTROL) & CONTROL_PAGE_16K != 0 { 14 } else { 12 };
            let page_mask = (1u64 << page_shift) - 1;
            let index = ((window - PCI_TRANSLATED) >> page_shift) as usize;
            if index >= XBRIDGE_ATE_MAPPED || st.ate[index] & ATE_VALID == 0 {
                st.int_error |= INT_PAGE_FAULT;
                st.multiple |= INT_PAGE_FAULT;
                crate::dlog!(LogModule::Bridge, "bridge: XBridge DMA to {addr:#x} faults on translation entry {index}");
                return None;
            }
            let ate = st.ate[index];
            let widget = ((ate >> ATE_TARGET_WIDGET_SHIFT) & 0xf) as u8;
            let swap = ate & XBRIDGE_ATE_SWAP != 0 || addr & XBRIDGE_MAPPED_SWAP != 0;
            return Some(Route::Host(((ate & ATE_ADDRESS) & !page_mask) | (addr as u64 & page_mask), placement(swap), widget));
        }
        if addr >= PCI_DIRECT {
            let (xio, widget) = Self::direct(st, addr);
            return (widget != 0).then_some(Route::Host(xio, placement(dev & DEV_SWAP_DIRECT != 0), widget));
        }
        crate::dlog!(LogModule::Bridge, "bridge: XBridge DMA to {addr:#x} is in no window");
        None
    }

    /// The 31-bit direct window's XIO address and widget for `addr`.
    fn direct(st: &State, addr: u32) -> (u64, u8) {
        let direct = st.r(reg::DIRECT_MAP);
        let mut offset = (((direct & DIRECT_OFFSET) as u64) << 31) | (addr & 0x7fff_ffff) as u64;
        // Applies only with a zero offset field; the bit resets set.
        if direct & DIRECT_OFFSET == 0 && direct & DIRECT_ADD_512MB != 0 {
            offset += DIRECT_512MB;
        }
        (offset, ((direct & DIRECT_WIDGET) >> DIRECT_WIDGET_SHIFT) as u8)
    }

    fn translate_bridge(st: &State, slot: usize, addr: u64) -> Option<(u64, Placement, u8)> {
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
            let (offset, widget) = Self::direct(st, addr);
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
                self.translate(&mut st, slot, a)
            };
            // Only the host chip is memory. The Crossbow drops a request to a widget
            // it has no live link to (1 to 7 are illegal, 15 is this Bridge's
            // own port), and the Bridge's request timer reports a dropped
            // read. A refused translation, or a read that reaches the host and
            // finds no memory, is a silent master abort.
            let mut dropped = false;
            let good = match target {
                None => false,
                Some(Route::Flush) => {
                    if let DmaBuf::Read(b) = &mut buf {
                        b[done..done + step].fill(0);
                    }
                    true
                }
                Some(Route::Host(xio, p, w)) => match self.up.get(w) {
                    Some(h) if !h.link_up() => {
                        // The Crossbow drops it; a dropped read is timed out.
                        log::debug!("bridge: transfer to widget {w}, whose link is down; dropped");
                        dropped = true;
                        false
                    }
                    Some(h) => match &mut buf {
                        DmaBuf::Read(b) => self.placed_read(&*h.mem, xio, p, &mut b[done..done + step]),
                        DmaBuf::Write(b) => self.placed_write(&*h.mem, xio, p, &b[done..done + step]),
                    },
                    None => {
                        // Approximation: a transfer to a widget that is not a
                        // host is dropped, not routed to a peer; no supported
                        // guest is known to need peer DMA.
                        log::debug!("bridge: transfer to widget {w}, which is not a host; dropped");
                        dropped = w == self.widget.load(Ordering::Acquire) || !(self.link_alive.read())(w);
                        false
                    }
                },
            };
            if !good {
                crate::dlog!(LogModule::Bridge, "bridge: slot {slot} {} of {step} bytes at PCI {a:#x} reached no memory ({})",
                    if matches!(buf, DmaBuf::Read(_)) { "read" } else { "write" },
                    match target { Some(Route::Host(x, _, w)) => format!("widget {w:#x} address {x:#x}"), _ => "no translation".to_string() });
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
    fn mem_read(&self, mem: &dyn BusDevice, addr: u64, out: &mut [u8]) -> bool {
        let ok = crate::physical::read_bytes(mem, addr, out) == crate::traits::BUS_OK;
        if !ok {
            out.fill(0);
        }
        ok
    }

    fn mem_write(&self, mem: &dyn BusDevice, addr: u64, data: &[u8]) -> bool {
        crate::physical::write_bytes(mem, addr, data) == crate::traits::BUS_OK
    }

    /// Read `out.len()` bytes at PCI-order position `xio`, undoing the
    /// placement: the PCI byte at address p is memory byte p ^ mask.
    fn placed_read(&self, mem: &dyn BusDevice, xio: u64, placement: Placement, out: &mut [u8]) -> bool {
        let mask = placement.mask();
        if mask == 0 {
            return self.mem_read(mem, xio, out);
        }
        let unit = mask as usize + 1;
        if xio & mask == 0 && out.len() % unit == 0 {
            // Whole units: the bytes come straight into place and each unit
            // is reversed where it lies.
            let ok = self.mem_read(mem, xio, out);
            out.chunks_exact_mut(unit).for_each(<[u8]>::reverse);
            return ok;
        }
        let lo = xio & !mask;
        let hi = (xio + out.len() as u64 + mask) & !mask;
        let mut span = vec![0u8; (hi - lo) as usize];
        let ok = self.mem_read(mem, lo, &mut span);
        for (i, b) in out.iter_mut().enumerate() {
            *b = if ok { span[(((xio + i as u64) ^ mask) - lo) as usize] } else { 0 };
        }
        ok
    }

    fn placed_write(&self, mem: &dyn BusDevice, xio: u64, placement: Placement, data: &[u8]) -> bool {
        let mask = placement.mask();
        if mask == 0 {
            return self.mem_write(mem, xio, data);
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
            return self.mem_write(mem, xio, swapped);
        }
        let lo = xio & !mask;
        let hi = (xio + data.len() as u64 + mask) & !mask;
        let mut span = vec![0u8; (hi - lo) as usize];
        for (i, &b) in data.iter().enumerate() {
            span[(((xio + i as u64) ^ mask) - lo) as usize] = b;
        }
        if xio == lo && xio + data.len() as u64 == hi {
            return self.mem_write(mem, lo, &span);
        }
        // Partial units: write only the bytes covered, so a neighbour
        // another master is changing is not overwritten with a stale copy.
        let mut ok = true;
        for i in 0..data.len() as u64 {
            let p = (xio + i) ^ mask;
            ok &= self.mem_write(mem, p, &span[(p - lo) as usize..(p - lo) as usize + 1]);
        }
        ok
    }

    // ---- the window ----

    /// A processor read of `size` bytes at `off` in the small window. `None`
    /// is a bus error.
    fn read(&self, off: u32, size: usize) -> Option<u64> {
        match off {
            0..=0xffff => {
                let v = self.reg_read(off, size);
                crate::dlog!(LogModule::Bridge, "bridge: read {} /{size} -> {}", reg_name(off & !7),
                    v.map_or("bus error".to_string(), |v| format!("{v:#x}")));
                v
            }
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
            // Behind the device windows: probe traffic, not the PIO path.
            _ if (TYPE1_BASE..TYPE1_BASE + TYPE1_SIZE).contains(&off) => self.type1_read(off - TYPE1_BASE, size),
            FLASH_BASE..=0xff_ffff => self.flash_read(off - FLASH_BASE, size),
            _ => None,
        }
    }

    /// As [`Bridge::read`]; the result is a bus status.
    fn write(&self, off: u32, size: usize, val: u64) -> u32 {
        let swap_was = (off & !7 == reg::CONTROL).then(|| self.st.lock().r(reg::CONTROL) & CONTROL_MEM_SWAP);
        let ok = match off {
            0..=0xffff => {
                crate::dlog!(LogModule::Bridge, "bridge: write {} /{size} {val:#x}", reg_name(off & !7));
                self.reg_write(off, size, val)
            }
            ATE_BASE..=0x10fff => {
                crate::dlog!(LogModule::Bridge, "bridge: write translation entry {} /{size} {val:#x}", (off - ATE_BASE) / 8);
                self.ate_write(off - ATE_BASE, size, val)
            }
            _ if (CFG0_BASE..CFG0_BASE + CFG0_SIZE).contains(&off) => self.cfg_write(off - CFG0_BASE, size, val),
            _ if (SSRAM_BASE..SSRAM_BASE + SSRAM_SIZE).contains(&off) => true,
            _ if (DEVICE_WINDOW_BASE..FLASH_BASE).contains(&off) => match window_slot(off) {
                Some(s) => self.window_access(s as usize, off - device_window_offset(s), size, Some(val)).is_some(),
                None => false,
            },
            _ if (TYPE1_BASE..TYPE1_BASE + TYPE1_SIZE).contains(&off) => self.type1_write(off - TYPE1_BASE, size, val),
            FLASH_BASE..=0xff_ffff => self.flash_write(off - FLASH_BASE, size, val),
            _ => false,
        };
        // A Device(x) write moves or re-aims a window, and a change of
        // Control's memory swap bit re-aims the alias. Configuration writes
        // update the mappings inside the bus.
        let swap_changed = swap_was.is_some_and(|was| self.st.lock().r(reg::CONTROL) & CONTROL_MEM_SWAP != was);
        if (reg::device(0)..reg::device(DEVICES as u32)).contains(&(off & !7)) || swap_changed {
            self.bus.set_host_windows(&|| self.windows());
        }
        if ok {
            BUS_OK
        } else {
            BUS_ERR
        }
    }
}

/// What the XBridge head's own address space holds outside the Bridge's
/// register and window decode: registers it defers (they read zero and
/// take writes), with no flash, SSRAM or diagnostic engine behind them.
fn xbridge_deferred(off: u64) -> bool {
    (0x300..0x600).contains(&off) || (0x3_0000..0x3_0008).contains(&off) || (FLASH_BASE as u64..WIDGET_SMALL_SIZE as u64).contains(&off)
}

impl Bridge {
    /// What the machine connects at the XBridge head's crossbar port: the
    /// whole of the head's XIO space, the 16 MB small window and the three
    /// aliases of PCI space past it. An undefined address answers zero and
    /// latches an invalid-address error; nothing faults.
    pub fn head_window(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        assert_eq!(self.part, Part::XBridge, "only an XBridge head has a head window");
        Arc::new(HeadWindow(self.clone()))
    }

    /// An XBridge processor access nothing on the bus claims: a read finds
    /// ones and a write is dropped, with no error latched.
    fn xbridge_unclaimed(&self, space: Space, pci: u32, size: usize, val: Option<u64>) -> u64 {
        crate::dlog!(LogModule::Bridge, "bridge: XBridge widget {} has no PCI BAR at {space:?} {pci:#x}", self.widget.load(Ordering::Acquire));
        if val.is_some() { 0 } else { u64::MAX >> (64 - 8 * size) }
    }

    /// The XBridge head's configuration apertures. A doubleword is two
    /// configuration words, the lower address in the upper half. A type 1
    /// cycle for bus 0 reaches the head's own bus. A cycle nothing claims
    /// reads as ones and notes a master time-out in the multiple-error
    /// register.
    fn xbridge_config(&self, off: u32, size: usize, val: Option<u64>) -> u64 {
        if size == 8 {
            let hi = self.xbridge_config(off, 4, val.map(|v| v >> 32));
            return (hi << 32) | self.xbridge_config(off + 4, 4, val.map(|v| v & 0xffff_ffff));
        }
        let reg = Self::lane_correct(off & 0xff, size) as usize;
        let function = ((off >> 8) & 7) as u8;
        let (bus, device) = if off < TYPE1_BASE {
            (Some(self.bus.clone()), ((off - CFG0_BASE) >> 12) as u8)
        } else {
            let sel = self.st.lock().r(reg::PCI_TYPE1_CONFIG);
            (self.bus.find_bus((sel >> 16) as u8), ((sel >> 11) & 0x1f) as u8)
        };
        let at = DevFn { device, function };
        let mask = u32::MAX >> (32 - size * 8);
        if let Some(v) = val {
            if let Some(bus) = bus {
                bus.config_write(at, reg, size, v as u32 & mask);
            }
            return 0;
        }
        match bus.and_then(|b| b.config_read(at, reg, size)) {
            Some(v) => v as u64,
            None => {
                self.st.lock().multiple |= INT_PCI_MASTER_TIMEOUT;
                mask as u64
            }
        }
    }

    /// A processor access to the XBridge head's space at `off`; `val` is
    /// `Some` for a write. Registers are 32 bits in the data half of their
    /// slot, translation entries are written whole and read a half at a
    /// time, and anything else undefined latches an invalid-address error
    /// and reads zero.
    fn xbridge_access(&self, off: u64, size: usize, val: Option<u64>) -> u64 {
        let small = off < WIDGET_SMALL_SIZE as u64;
        let o = off as u32;
        if small && (CFG0_BASE..TYPE1_BASE + TYPE1_SIZE).contains(&o) {
            return self.xbridge_config(o, size, val);
        }
        if small && (DEVICE_WINDOW_BASE..FLASH_BASE).contains(&o) {
            if let Some(slot) = window_slot(o) {
                return self.window_access(slot as usize, o - device_window_offset(slot), size, val).unwrap_or(0);
            }
        }
        if let Some(i) = XBRIDGE_ALIASES.iter().position(|&(base, len, _)| (base..base + len).contains(&off)) {
            let (base, _, space) = XBRIDGE_ALIASES[i];
            return self.alias_access(space, (off - base) as u32, size, val).unwrap_or(0);
        }
        if small && size == 8 && o % 8 == 0 && (ATE_BASE..XBRIDGE_ATE_LOW_BASE).contains(&o) {
            if let Some(v) = val {
                crate::dlog!(LogModule::Bridge, "bridge: write translation entry {} {v:#x}", (o - ATE_BASE) / 8);
                self.st.lock().ate[((o - ATE_BASE) / 8) as usize] = v & XBRIDGE_ATE_KEPT;
                return 0;
            }
        }
        if small && size == 4 && o % 8 == SLOT_DATA {
            if val.is_none() && (ATE_BASE..XBRIDGE_ATE_END).contains(&o) {
                let st = self.st.lock();
                return if o < XBRIDGE_ATE_LOW_BASE {
                    st.ate[((o - ATE_BASE) / 8) as usize] >> 32
                } else {
                    st.ate[((o - XBRIDGE_ATE_LOW_BASE) / 8) as usize] & 0xffff_ffff
                };
            }
            if o < REGS_DEFINED && reg_info(Part::XBridge, o - SLOT_DATA).is_some() {
                crate::dlog!(LogModule::Bridge, "bridge: XBridge {} {} /{size}{}", if val.is_some() { "write" } else { "read" },
                    reg_name(o - SLOT_DATA), val.map_or(String::new(), |v| format!(" {v:#x}")));
                return match val {
                    Some(v) => {
                        self.reg_write(o, size, v);
                        0
                    }
                    None => self.reg_read(o, size).unwrap_or(0),
                };
            }
        }
        if xbridge_deferred(off) {
            crate::dlog!(LogModule::Bridge, "bridge: XBridge widget {} deferred access at {off:#x}", self.widget.load(Ordering::Acquire));
        } else {
            self.irq_locked(|st, p| self.invalid_address(st, p, off, val.is_some()));
        }
        0
    }
}

/// The XBridge head's XIO space as a processor reaches it: every access
/// completes.
struct HeadWindow(Arc<Bridge>);

macro_rules! head_access {
    ($r:ident, $w:ident, $t:ty, $out:ident, $n:expr) => {
        fn $r(&self, a: u64) -> $out {
            $out::ok(self.0.xbridge_access(a, $n, None) as $t)
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.0.xbridge_access(a, $n, Some(v as u64));
            BUS_OK
        }
    };
}

impl BusDevice for HeadWindow {
    head_access!(read8, write8, u8, BusRead8, 1);
    head_access!(read16, write16, u16, BusRead16, 2);
    head_access!(read32, write32, u32, BusRead32, 4);
    head_access!(read64, write64, u64, BusRead64, 8);
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

/// What the bus asks of its host: the DMA of a requester (the swap bits that
/// steer a transfer are its own slot's Device(x) register), and a pin's
/// level, as the OR of everything routed to it. The bus calls it with every
/// delivery to one pin serialised and in order, outside any lock of ours.
impl PciHost for Bridge {
    fn dma_for(&self, requester: u8) -> Arc<dyn PciDma> {
        Arc::new(SlotDma { bridge: self.me.clone(), slot: requester as usize })
    }

    fn set_irq(&self, pin: u8, level: bool) {
        self.irq_locked(|st, p| self.update_pin(st, p, pin as usize, level));
    }
}

impl crate::traits::Device for Bridge {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("pci".to_string(), "Bridge PCI bus: pci status | pci config <slot>".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "pci" {
            return Err("Command not found".to_string());
        }
        self.bus.command(args, &mut w)
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

/// The widget's 2 GB window: the small window's 16 MB, then nothing up to
/// the 32-bit alias of PCI memory. Runs a BAR decodes whole are mapped over
/// it; this answers the rest.
struct LongWindow(Arc<Bridge>);

impl LongWindow {
    fn read(&self, addr: u64, size: usize) -> Option<u64> {
        match (addr & LONG_MASK) as u32 {
            off @ 0..WIDGET_SMALL_SIZE => self.0.read(off, size),
            off @ PIO32_ALIAS.. => self.0.alias_access(Space::Mem, off - PIO32_ALIAS, size, None),
            // Approximation: the specification calls 16 MB to 1 GB reserved
            // (p. 52) and does not say what an access there does; a bus
            // error here.
            _ => None,
        }
    }

    fn write(&self, addr: u64, size: usize, val: u64) -> u32 {
        match (addr & LONG_MASK) as u32 {
            off @ 0..WIDGET_SMALL_SIZE => self.0.write(off, size, val),
            off @ PIO32_ALIAS.. => {
                self.0.alias_access(Space::Mem, off - PIO32_ALIAS, size, Some(val));
                BUS_OK
            }
            _ => BUS_ERR,
        }
    }
}

impl BusDevice for LongWindow {
    /// A configuration master time-out reached through the window's first
    /// 16 MB is explained as through the small window.
    fn describe_fault(&self, addr: u64, size: usize) -> Option<String> {
        let off = addr & LONG_MASK;
        if off < WIDGET_SMALL_SIZE as u64 { self.0.describe_fault(off, size) } else { None }
    }
    fn read8(&self, addr: u64) -> BusRead8 {
        self.read(addr, 1).map_or(BusRead8::err(), |v| BusRead8::ok(v as u8))
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        self.read(addr, 2).map_or(BusRead16::err(), |v| BusRead16::ok(v as u16))
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        self.read(addr, 4).map_or(BusRead32::err(), |v| BusRead32::ok(v as u32))
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        self.read(addr, 8).map_or(BusRead64::err(), BusRead64::ok)
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        self.write(addr, 1, val as u64)
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        self.write(addr, 2, val as u64)
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        self.write(addr, 4, val as u64)
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write(addr, 8, val)
    }
}

impl BusDevice for Bridge {
    fn describe_fault(&self, addr: u64, size: usize) -> Option<String> {
        let off = addr as u32 & WINDOW_MASK;
        if (CFG0_BASE..CFG0_BASE + CFG0_SIZE).contains(&off) {
            self.cfg_fault(off - CFG0_BASE, size)
        } else if (TYPE1_BASE..TYPE1_BASE + TYPE1_SIZE).contains(&off) {
            self.type1_fault(off - TYPE1_BASE, size)
        } else {
            None
        }
    }

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

// The Bridge's state for a snapshot: its registers, address translation
// entries, interrupt pin levels and latched errors, its MicroLAN bus, and
// the PCI bus (the configuration header of every function on it and the
// levels of the pins and board IRQs into the Bridge). Loading puts the headers
// back first and then publishes the BAR pages from them, so the native PIO
// decode is derived, not stored. The flash parts behind its chip selects
// save themselves.
impl Saveable for Bridge {
    fn save_state(&self) -> toml::Value {
        let microlan = self.microlan.lock().as_ref().map_or(toml::Value::Boolean(false), |bus| bus.save_state());
        let st = self.st.lock();
        let mut table = StateTable::new();
        table
            .u32s("reg", &st.reg)
            .u64s("ate", &st.ate[..self.part.ate_entries()])
            .u32("int_error", st.int_error)
            .u8s("pins", &[st.pin_level, st.dma_64bit])
            .u64("dma_pci_address", st.dma_pci_address)
            .u32("dma_device", st.dma_device)
            .value("bus", self.bus.save_state())
            .value("microlan", microlan)
            .u32("revision", self.revision.load(Ordering::Acquire));
        if self.part == Part::XBridge {
            table.u32("multiple", st.multiple);
        }
        table.build()
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        // The silicon revision is the board's, like the functions fitted.
        if field_u32(v, "revision")? != self.revision.load(Ordering::Acquire) {
            return Err("the saved Bridge is another revision".into());
        }
        let ate = field_u64_vec(v, "ate")?;
        let entries = self.part.ate_entries();
        if ate.len() != entries {
            return Err(format!("ate: {} entries, expected {entries}", ate.len()));
        }
        self.bus.load_state(field(v, "bus")?)?;
        let saved = field(v, "microlan")?;
        match (self.microlan.lock().as_mut(), saved.as_bool()) {
            (Some(bus), None) => bus.load_state(saved).map_err(|e| format!("MicroLAN: {e}"))?,
            (None, Some(false)) => {}
            _ => return Err("the saved Bridge has a different MicroLAN bus".into()),
        }
        {
            let [pin_level, dma_64bit] = field_u8s::<2>(v, "pins")?;
            let mut st = self.st.lock();
            st.reg = field_u32s(v, "reg")?;
            st.ate[..entries].copy_from_slice(&ate);
            if self.part == Part::XBridge {
                st.multiple = field_u32(v, "multiple")?;
            }
            st.int_error = field_u32(v, "int_error")?;
            st.pin_level = pin_level;
            st.dma_64bit = dma_64bit;
            st.dma_pci_address = field_u64(v, "dma_pci_address")?;
            st.dma_device = field_u32(v, "dma_device")?;
        }
        self.bus.set_host_windows(&|| self.windows());
        Ok(())
    }
}

#[cfg(test)]
#[path = "bridge_tests.rs"]
mod tests;
