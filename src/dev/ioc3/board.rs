//! What a system board fits around its IOC3, as data.
//!
//! This is the shape of Linux's `ioc3_infos` table in `drivers/mfd/ioc3.c`:
//! a row per board, keyed by the PCI subsystem id the board's IOC3 is given,
//! naming the children the board has. Each row also carries what Linux keeps
//! in the Bridge setup next to it: the Bridge input the SuperIO interrupt is
//! wired to. The chip's GPIO pins are the chip's; the row says what the board
//! wires to them (straps, and the lightbar a status-bar bit mirrors).
//!
//! The subsystem id is a model choice, not a register. The chip decodes only
//! 32 bytes of configuration space, so a guest never reads one; Linux's
//! Bridge driver synthesises it from the part number in the board's one-wire
//! record. The rows keep the values the machines used before the table.

use super::eth::Phy;
use super::gpio::PinWire;
use crate::dev::ng1::rex3::Rex3;
use crate::pci::PciConfigPolicy;

/// Which children a board fits, besides the Super I/O parts below.
#[derive(Clone, Copy, Debug)]
pub struct Cells {
    /// Both serial ports.
    pub serial: bool,
    /// The PS/2 keyboard and mouse controller.
    pub kbd: bool,
    /// The parallel port.
    pub parallel: bool,
    /// The Ethernet MAC behind `Ioc3Board::phy`.
    pub eth: bool,
    /// The station-address part on the MicroLAN wire. The master is the
    /// chip's: without a part its slots still finish and the wire reads
    /// absent.
    pub w1: bool,
    /// What sits on the byte bus.
    pub byte_bus: ByteBusPart,
}

/// The clock a board fits on the byte bus, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ByteBusPart {
    None,
    /// The M48T35-compatible Timekeeper on select 0, 32 KiB.
    M48t35,
    /// The DS1685 on selects 1 and 2, behind an index and a data port.
    Ds1685,
}

/// One row: a board's IOC3.
#[derive(Debug)]
pub struct Ioc3Board {
    pub name: &'static str,
    pub subsystem_id: u16,
    pub cells: Cells,
    /// The transceiver on the MAC's management bus, if `cells.eth`.
    pub phy: Phy,
    /// The Bridge input the SuperIO interrupt is wired to.
    pub superio_pin: u8,
    /// What the board wires to the chip's GPIO pins; a pin not listed is
    /// unwired.
    pub gpio: &'static [PinWire],
    /// Function-specific PCI header write behavior.
    pub pci_policy: PciConfigPolicy<'static>,
    /// Reset the PCI header along with the chip on power-on.
    pub reset_pci_on_power_on: bool,
}

const SERIAL_ETH: Cells = Cells {
    serial: true,
    kbd: false,
    parallel: false,
    eth: true,
    w1: true,
    byte_bus: ByteBusPart::M48t35,
};

/// The Octane and Octane2 system board ("IP30 System Board"). The board
/// drives GPIO inputs 2 and 3 high, read on an Octane2's PROM Command
/// Monitor (pin data 0x12d); an Octane is unmeasured and assumed the same.
/// Pins 0 and 1 drive the front lightbar: white power, shown as the status
/// bar's green LED, and red fault.
pub static IP30: Ioc3Board = Ioc3Board {
    name: "IP30 System Board",
    subsystem_id: 0xc304,
    cells: Cells { kbd: true, parallel: true, byte_bus: ByteBusPart::Ds1685, ..SERIAL_ETH },
    phy: Phy::Ics1890,
    superio_pin: 4,
    gpio: &[
        PinWire::status_bit(0, Rex3::HB_LED_GREEN),
        PinWire::status_bit(1, Rex3::HB_LED_RED),
        PinWire::strap(2, true),
        PinWire::strap(3, true),
    ],
    pci_policy: PciConfigPolicy::DEFAULT,
    reset_pci_on_power_on: false,
};

/// The Origin 200 system board ("IP29 System Board", Linux's BaseIO6G
/// setup).
pub static IP29: Ioc3Board = Ioc3Board {
    name: "IP29 System Board",
    subsystem_id: 0xc303,
    cells: Cells { kbd: true, parallel: true, ..SERIAL_ETH },
    phy: Phy::Dp83840,
    superio_pin: 3,
    gpio: &[],
    pci_policy: PciConfigPolicy::DEFAULT,
    reset_pci_on_power_on: false,
};

/// The Origin 2000 BaseIO: Ethernet, both serial ports, the one-wire part
/// and the Timekeeper, and no keyboard, mouse or parallel port.
pub static BASEIO: Ioc3Board = Ioc3Board {
    name: "IP27 BaseIO",
    subsystem_id: 0xc302,
    cells: SERIAL_ETH,
    phy: Phy::Dp83840,
    superio_pin: 4,
    gpio: &[],
    pci_policy: PciConfigPolicy::DEFAULT,
    reset_pci_on_power_on: false,
};

/// The Onyx2 BaseIO-G's own IOC3 (Linux's "IP27 BaseIO6G"): the BaseIO's
/// with a keyboard and mouse controller added. Linux maps the board's part
/// number, 030-0734, to subsystem id 0xc300; the id is the table's key and
/// no guest reads it, so the row takes Linux's.
pub static BASEIO6G: Ioc3Board = Ioc3Board {
    name: "IP27 BaseIO6G",
    subsystem_id: 0xc300,
    cells: Cells { kbd: true, ..SERIAL_ETH },
    phy: Phy::Dp83840,
    superio_pin: 4,
    gpio: &[],
    pci_policy: PciConfigPolicy::DEFAULT,
    reset_pci_on_power_on: false,
};

/// The Onyx2 BaseIO-G's MediaIO daughtercard (Linux's "IP27 MIO"): a second
/// IOC3 with both serial ports, a keyboard and mouse controller and the
/// parallel port, and no Ethernet, clock or one-wire part. Its SuperIO
/// interrupt reaches Bridge input 6.
pub static MIO: Ioc3Board = Ioc3Board {
    name: "IP27 MIO",
    subsystem_id: 0xc301,
    cells: Cells { serial: true, kbd: true, parallel: true, eth: false, w1: false, byte_bus: ByteBusPart::None },
    phy: Phy::Dp83840,
    superio_pin: 6,
    gpio: &[],
    pci_policy: PciConfigPolicy::DEFAULT,
    reset_pci_on_power_on: false,
};

const IP34_POLICY: PciConfigPolicy<'static> = PciConfigPolicy {
    status: 0x0280,
    status_w1c_mask: 0,
    command_mask: 0xffff,
    latency_mask: 0xff,
    cache_line_mask: 0,
    registers: &[],
};

/// The Fuel system board. Its id is in Linux's `ioc3.h` and Bridge setup
/// but not in `ioc3_infos`.
pub static IP34: Ioc3Board = Ioc3Board {
    name: "IP34 System Board",
    subsystem_id: 0xc30a,
    cells: Cells { kbd: true, parallel: true, ..SERIAL_ETH },
    phy: Phy::Dp83843,
    superio_pin: 0,
    gpio: &[],
    pci_policy: IP34_POLICY,
    reset_pci_on_power_on: true,
};

/// The Fuel's board with the PS/2 controller unfitted, for a machine that
/// takes no host keyboard and mouse.
pub static IP34_NO_PS2: Ioc3Board = Ioc3Board {
    name: "IP34 System Board",
    subsystem_id: 0xc30a,
    cells: Cells { parallel: true, ..SERIAL_ETH },
    phy: Phy::Dp83843,
    superio_pin: 0,
    gpio: &[],
    pci_policy: IP34_POLICY,
    reset_pci_on_power_on: true,
};
