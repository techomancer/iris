//! SGI IOC3 base I/O controller, as the Octane (IP30) fits it.
//!
//! One PCI function with one 1 MiB memory BAR. Behind it: two 16550 serial
//! ports with a ring-DMA engine each, a PS/2 keyboard and mouse controller,
//! a parallel port, a MicroLAN (one-wire) master, the GPIO pins, a DS1685
//! clock on the byte bus, the Ethernet buffer RAM and, in the register range
//! at 0xf0, the Ethernet MAC (which lives in `ioc3_eth.rs` behind
//! [`Ioc3EthPort`]).
//!
//! Behaviour follows the public Linux and OpenBSD drivers and measured guest
//! register traces; where the model approximates, the comment says so.
//!
//! # Two interrupt outputs
//!
//! The chip has two. The Ethernet block drives the PCI interrupt pin
//! ([`PciBusDevice::connect_interrupt`]); everything else (serial, PS/2,
//! parallel, the interrupt-output pulse) is gated by the status and enable
//! pair at 0x1c/0x20 and drives a second line the board wires to a Bridge
//! input ([`Ioc3::connect_superio_interrupt`]).
//!
//! # Byte order
//!
//! IOC3 owns the conversion between its register and DMA formats and PCI
//! bytes. Registers: a 32-bit access is the register value; a narrow access
//! to internal offset `a` arrives from the bridge at `a ^ (4 - width)`, which
//! [`PciBusDevice::bar_read`] undoes. DMA: an internal byte at address `a`
//! sits at bus address `a ^ 3`, so a 32-bit value is big-endian on the bus
//! with the bridge's swap clear ([`Ioc3Dma`]). Neither is a measured property
//! of the silicon; the chip's real placement is undocumented.
//!
//! # What is approximated here
//!
//! * Nothing is timed to a baud rate: serial transmit completes inside the
//!   write and a loopback character arrives at once (the part takes a
//!   character time).
//! * Serial ring DMA, the receive timer, transmit delays and the interrupt
//!   output are evaluated lazily on every register access, and an `hptimer`
//!   one-shot per deadline runs the same catch-up while the guest is idle.
//!   Host serial input is fed into the FIFOs by one thread per port.
//! * The DS1685 is the O2's DS1687 model behind an index and a data port; it
//!   has no alarm or periodic interrupts. Its RAM, including the 128-byte
//!   extended RAM, persists in a 512-byte file.
//!
//! Ported from hw/misc/sgi-ioc3.c and its serial, parallel and keyboard siblings
//! in jamesbraid/qemu-sgi-origami.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::dev::ds1687::MaceRtc;
use crate::dev::ioc3_port::Ioc3EthPort;
use crate::dev::ps2::Ps2Controller;
use crate::dev::w1;
use crate::dev::w1::sgi::{self, BoardRecord};
use crate::dev::ns16550::Ns16550;
use crate::dev::serial_host::SerialBackend;
use crate::irq::{InterruptController, IrqLine};
use crate::pci::{Bar, PciBusDevice, PciConfig, PciDma, PciDmaStatus, PCI_INTA};
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable};

pub const VENDOR_ID: u16 = 0x10a9;
pub const DEVICE_ID: u16 = 0x0003;
pub const REVISION: u8 = 0x01;
/// PCI subsystem id of the Octane's system board.
pub const SUBSYS_IP30_SYSBOARD: u16 = 0xc304;
pub const BAR_SIZE: u32 = 1 << 20;
/// The configuration dwords the chip decodes; the rest of the 16 repeat them.
const CONFIG_DECODED_DWORDS: usize = 8;
/// PCI status bit 7, fast back-to-back capable.
const PCI_STATUS_FAST_BACK: u32 = 0x0080;

/// Register offsets within the BAR.
pub mod reg {
    pub const INT_STATUS: u32 = 0x1c;
    pub const INT_ENABLE_SET: u32 = 0x20;
    pub const INT_ENABLE_CLR: u32 = 0x24;
    pub const CHIP_CONTROL: u32 = 0x28;
    pub const INT_OUTPUT: u32 = 0x2c;
    pub const MICROLAN: u32 = 0x30;
    /// The last dword of the PCI header that shows through the block.
    pub const HEADER_LAST: u32 = 0x10;
    pub const PIN_CONTROL_SET: u32 = 0x34;
    pub const PIN_CONTROL_CLR: u32 = 0x38;
    pub const PIN_DATA: u32 = 0x3c;
    pub const PIN_REG_FIRST: u32 = 0x40;
    pub const PIN_REG_LAST: u32 = 0x60;
    pub const PAR_DMA_BASE: u32 = 0x80;
    pub const KBC_BASE: u32 = 0x9c;
    pub const SDMA_RING_HI: u32 = 0xb0;
    pub const SDMA_RING_LO: u32 = 0xb4;
    pub const SDMA_PORT_A: u32 = 0xb8;
    pub const SDMA_PORT_B: u32 = 0xd4;
    /// First and last byte of the Ethernet MAC's registers.
    pub const ENET_FIRST: u32 = 0xf0;
    pub const ENET_LAST: u32 = 0x153;
    pub const ENET_MAC_CONTROL: u32 = 0xf0;
    pub const SDMA_ALIAS_A: u32 = 0x8000;
    pub const SDMA_ALIAS_B: u32 = 0xc000;
    pub const SDMA_ALIAS_SIZE: u32 = 0x4000;
    pub const SUPERIO: u32 = 0x20000;
    pub const SUPERIO_SIZE: u32 = 0x200;
    pub const SERIAL_A: u32 = 0x20178;
    pub const SERIAL_B: u32 = 0x20170;
    pub const PAR_ECP_BASE: u32 = 0x20150;
    pub const SSRAM: u32 = 0x40000;
    pub const SSRAM_SIZE: u32 = 0x40000;
    pub const BYTEBUS_DEV1: u32 = 0xa0000;
    pub const BYTEBUS_DEV2: u32 = 0xc0000;
    pub const BYTEBUS_SIZE: u32 = 0x20000;
}

/// Interrupt-status bits. Each serial port owns nine, A then B.
pub mod int {
    pub const PORT_STRIDE: u32 = 9;
    pub const TX_RING_EMPTY: u32 = 0;
    pub const RX_RING_FULL: u32 = 1;
    pub const RX_HIGH_WATER: u32 = 2;
    pub const RX_TIMER: u32 = 3;
    pub const PASSTHROUGH: u32 = 6;
    pub const TX_EXPLICIT: u32 = 7;
    pub const DMA_ERROR: u32 = 8;
    pub const fn serial(port: usize, src: u32) -> u32 {
        1 << (port as u32 * PORT_STRIDE + src)
    }
    pub const SERIAL_A: u32 = serial(0, PASSTHROUGH);
    pub const SERIAL_B: u32 = serial(1, PASSTHROUGH);
    pub const PARALLEL_SERVICE: u32 = 1 << 18;
    pub const PARALLEL_CONTEXT: u32 = 19;
    pub const PARALLEL_ERROR: u32 = 1 << 21;
    pub const KEYBOARD: u32 = 1 << 22;
    pub const RT_OUTPUT_PULSE: u32 = 1 << 27;
    /// Bits that follow their source's level; a write cannot clear them.
    pub const LEVELS: u32 = SERIAL_A
        | SERIAL_B
        | PARALLEL_SERVICE
        | serial(0, RX_HIGH_WATER)
        | serial(1, RX_HIGH_WATER)
        | serial(0, RX_RING_FULL)
        | serial(1, RX_RING_FULL)
        | serial(0, TX_RING_EMPTY)
        | serial(1, TX_RING_EMPTY);
}

// ---------------------------------------------------------------------------
// The 16550s
// ---------------------------------------------------------------------------

mod uart_reg {
    pub const RBR_THR: u32 = 0;
    pub const IIR_FCR: u32 = 2;
    pub const MCR: u32 = 4;
    pub const LSR: u32 = 5;
    pub const MSR: u32 = 6;
    pub const SCR: u32 = 7;
}
const LSR_DR: u8 = 0x01;
const LSR_OE: u8 = 0x02;
const LSR_PE: u8 = 0x04;
const LSR_FE: u8 = 0x08;
const LSR_BI: u8 = 0x10;
const MCR_DTR: u8 = 0x01;
const MCR_RTS: u8 = 0x02;
const MCR_OUT1: u8 = 0x04;
const MCR_OUT2: u8 = 0x08;
const MCR_LOOP: u8 = 0x10;
const MCR_AFE: u8 = 0x20;
const MSR_DCTS: u8 = 0x01;
const MSR_CTS: u8 = 0x10;
const MSR_DCD: u8 = 0x80;
const LSR_TEST_MASK: u8 = 0x7f;
const MSR_TEST_MASK: u8 = 0x0f;
/// The reference clock of the UART crystal: the pre-divider divides it, and
/// the divisor latch divides the result by 16 again.
const UART_CRYSTAL_HZ: u32 = 22_118_400;

/// A 16550 as IOC3 fits it: the shared core (`ns16550.rs`) with the chip's
/// own additions around it. Transmit is instantaneous (the byte goes to the
/// backend inside the write), so the transmitter always reads empty.
///
/// The additions: the auto-flow bit in modem control, which the core's modem
/// control masks away; factory-test writes to line and modem status; and the
/// scratch register doubling as the clock pre-divider while the divisor latch
/// is open. Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.
struct Uart {
    core: Ns16550,
    // IOC3 specifics, kept beside the 16550.
    predivider: u8,
    mcr_afe: u8,
    lsr_test: u8,
    msr_test: u8,
}

impl Uart {
    fn new() -> Uart {
        Uart { core: Ns16550::new(115200), predivider: 0, mcr_afe: 0, lsr_test: 0, msr_test: 0 }
    }

    fn reset(&mut self) {
        self.core.reset();
        self.mcr_afe = 0;
        self.lsr_test = 0;
        self.msr_test = 0;
        // The pre-divider keeps its value across a reset; no reset value is
        // specified for it.
    }

    fn loopback(&self) -> bool {
        self.core.loopback()
    }

    /// The interrupt line: low IIR bit clear.
    fn irq(&self) -> bool {
        self.core.irq()
    }

    /// Move whatever the host end has typed into the receive FIFO and let
    /// the receive timeout run out. Returns whether anything changed.
    fn pump(&mut self) -> bool {
        let moved = self.core.pump();
        self.core.check_timeout(Instant::now()) || moved
    }

    /// The 16550 proper: what the ring engine sees.
    fn read_core(&mut self, r: u32) -> u8 {
        self.core.read(r)
    }

    /// A guest's byte-wide read, with the chip's additions: the auto-flow
    /// bit in modem control, and a factory-test write to line or modem
    /// status showing in the read that follows it and no later one.
    fn read(&mut self, r: u32) -> u8 {
        let mut v = self.read_core(r);
        match r {
            uart_reg::MCR => v |= self.mcr_afe,
            uart_reg::LSR => {
                v |= self.lsr_test;
                self.lsr_test = 0;
            }
            uart_reg::MSR => {
                v |= self.msr_test;
                self.msr_test = 0;
            }
            _ => {}
        }
        v
    }

    fn write(&mut self, r: u32, v: u8) {
        match r {
            uart_reg::MCR => self.mcr_afe = v & MCR_AFE,
            uart_reg::LSR => self.lsr_test = v & LSR_TEST_MASK,
            uart_reg::MSR => self.msr_test = v & MSR_TEST_MASK,
            uart_reg::SCR => {
                // Scratch writes with the divisor latch up program the clock
                // pre-divider; twice the divide wanted. 0 and 1 mean nothing.
                if self.core.dlab() && v / 2 != 0 {
                    self.predivider = v;
                    self.core.set_baudbase(UART_CRYSTAL_HZ / (v / 2) as u32 / 16);
                }
            }
            _ => {}
        }
        self.core.write(r, v);
    }

    /// Modem status as the ring engine reads it with flow control looped back.
    fn flow_cts(&self, control: u32) -> bool {
        self.loopback() && control & sdma_ctl::FLOW_CONTROL != 0
    }
}

// ---------------------------------------------------------------------------
// Serial ring DMA
// ---------------------------------------------------------------------------

/// Port DMA register offsets (bytes, from the port's first register).
mod sdma_off {
    pub const CONTROL: u32 = 0x00;
    pub const TX_PRODUCE: u32 = 0x04;
    pub const TX_CONSUME: u32 = 0x08;
    pub const RX_PRODUCE: u32 = 0x0c;
    pub const RX_CONSUME: u32 = 0x10;
    pub const RX_TIMER: u32 = 0x14;
    pub const SHADOW: u32 = 0x18;
    pub const REGS: u32 = 7;
}
mod sdma_ctl {
    pub const HIGH_WATER: u32 = 0x1ff;
    pub const TX_TIMER_RUN: u32 = 1 << 16;
    pub const FLOW_CONTROL: u32 = 1 << 17;
    pub const RX_TO_MEMORY: u32 = 1 << 27;
    pub const ENABLE: u32 = 1 << 28;
    pub const PAUSE: u32 = 1 << 29;
    pub const PAUSED: u32 = 1 << 30;
    pub const RESET: u32 = 1 << 31;
    pub const UNSTORED: u32 = RESET | RX_TO_MEMORY | PAUSED | TX_TIMER_RUN;
}
const ENTRY: usize = 8;
const ENTRY_CHARS: usize = 4;
const ARM_RX_TIMER: u32 = 1 << 31;
const HELD_SHIFT: u32 = 24;
const TIMER_RELOAD: u32 = 0xfff;
const TIMER_COUNT_SHIFT: u32 = 16;
const TIMER_TICK_NS: u64 = 62_500; // 16 kHz
const TX_TYPE: u8 = 0xc0;
const TX_CHARACTER: u8 = 0x40;
const TX_MODEM: u8 = 0x80;
const TX_DELAY: u8 = 0xc0;
const TX_INTERRUPT: u8 = 0x20;
const RX_OVERRUN: u8 = 0x01;
const RX_PARITY: u8 = 0x02;
const RX_FRAMING: u8 = 0x04;
const RX_BREAK: u8 = 0x08;
const RX_CTS: u8 = 0x10;
const RX_CD: u8 = 0x20;
const RX_MODEM_OK: u8 = 0x40;
const RX_DATA_OK: u8 = 0x80;
/// Chip control: where each engine finds its 16550 (byte offset / 8).
const CHIP_CTRL_MASK: u32 = 0x0007_ffff;
const CHIP_CTRL_PORT_A_SHIFT: u32 = 1;
const CHIP_CTRL_PORT_B_SHIFT: u32 = 8;
const CHIP_CTRL_PORT_FIELD: u32 = 0x7f;
const CHIP_CTRL_ARB_IDLE: u32 = 1 << 22;

#[derive(Default)]
struct SdmaPort {
    reg: [u32; 7],
    packer: [u8; ENTRY],
    packed: usize,
    tx_entry: [u8; ENTRY],
    tx_fetched: bool,
    tx_slot: usize,
    armed: bool,
    drain_pending: bool,
    busy: bool,
    paused: bool,
    rx_deadline: Option<Instant>,
    rx_idle_count: u32,
    /// Time a frozen timer had left when a pause began.
    rx_frozen: Option<Duration>,
    tx_delay_deadline: Option<Instant>,
    tx_delay_frozen: Option<Duration>,
    tx_delay_irq: bool,
    /// An entry waits that the receive ring had no room for; drives
    /// receive-ring-full while ring DMA is on.
    post_refused: bool,
}

#[derive(Default)]
struct Sdma {
    base_hi: u32,
    base_lo: u32,
    port: [SdmaPort; 2],
}

// ---------------------------------------------------------------------------
// DMA as the chip masters the bus
// ---------------------------------------------------------------------------

/// The chip's view of bus-master memory. Internal byte `a` sits at bus
/// address `a ^ 3` (a word is big-endian on the bus with the bridge's swap
/// clear), so rings, descriptors and frames keep their layout whatever the
/// bridge then does to them. This is what the Ethernet block and the serial
/// and parallel engines are handed.
pub struct Ioc3Dma {
    raw: Arc<dyn PciDma>,
}

impl Ioc3Dma {
    pub fn new(raw: Arc<dyn PciDma>) -> Self {
        Self { raw }
    }
}

impl PciDma for Ioc3Dma {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        if buf.is_empty() {
            return PciDmaStatus::Ok;
        }
        let start = addr & !3;
        // A guest-chosen base can sit at the top of the address space.
        let Some(end) = addr.checked_add(buf.len() as u64 + 3).map(|e| e & !3) else {
            buf.fill(0);
            return PciDmaStatus::MasterAbort;
        };
        let mut tmp = vec![0u8; (end - start) as usize];
        let status = self.raw.dma_read(start, &mut tmp);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = tmp[((addr - start) as usize + i) ^ 3];
        }
        status
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        if data.is_empty() {
            return PciDmaStatus::Ok;
        }
        let start = addr & !3;
        let Some(end) = addr.checked_add(data.len() as u64 + 3).map(|e| e & !3) else {
            return PciDmaStatus::MasterAbort;
        };
        let mut tmp = vec![0u8; (end - start) as usize];
        if addr & 3 != 0 || data.len() & 3 != 0 {
            // Bytes outside the span keep their value: read the words the
            // span touches, overlay it, and write them back as one transfer.
            if !self.raw.dma_read(start, &mut tmp).is_ok() {
                return PciDmaStatus::MasterAbort;
            }
        }
        for (i, b) in data.iter().enumerate() {
            tmp[((addr - start) as usize + i) ^ 3] = *b;
        }
        self.raw.dma_write(start, &tmp)
    }
}

// ---------------------------------------------------------------------------
// PS/2 controller
// ---------------------------------------------------------------------------

const KBC_CONTROL: u32 = 0x9c;
const KBC_KBD_RECEIVE: u32 = 0xa0;
const KBC_AUX_RECEIVE: u32 = 0xa4;
const KBC_KBD_TRANSMIT: u32 = 0xa8;
const KBC_AUX_TRANSMIT: u32 = 0xac;
const KBC_RX_SLOTS: usize = 3;
/// Bits 16 and 17 (timeouts) read clear: no stalled line is simulated.
const KBC_CONTROL_MASK: u32 = 0x00fc_ffff;
const KBC_K_CLAMP_1: u32 = 0x0010_0000;
const KBC_M_CLAMP_1: u32 = 0x0020_0000;
const KBC_K_CLAMP_3: u32 = 0x0040_0000;
const KBC_M_CLAMP_3: u32 = 0x0080_0000;
const KBC_RX_SOURCE_AUX: u32 = 1 << 27;
const KBC_RX_OVERFLOW: u32 = 1 << 28;

#[derive(Default, Clone, Copy)]
struct KbcPort {
    data: [u8; KBC_RX_SLOTS],
    count: usize,
    overflow: bool,
}

#[derive(Default)]
struct Kbc {
    control: u32,
    port: [KbcPort; 2],
}

// ---------------------------------------------------------------------------
// Parallel port (TL16PIR552 population, 16-byte test FIFO)
// ---------------------------------------------------------------------------

const PAR_COUNT: u32 = 0x1_ffff;
const PAR_COUNT_ARM: u32 = 1 << 30;
const PAR_COUNT_LAST: u32 = 1 << 31;
const PAR_COUNT_STORED: u32 = PAR_COUNT | PAR_COUNT_ARM | PAR_COUNT_LAST;
const PAR_DMA_FROM_PERIPHERAL: u32 = 1 << 29;
const PAR_DMA_RESET: u32 = 1 << 30;
const PAR_DMA_ENABLE: u32 = 1 << 31;
const PAR_DMA_STORED: u32 = 0xe000_0000;
const PAR_ECR_FIFO_EMPTY: u8 = 0x01;
const PAR_ECR_FIFO_FULL: u8 = 0x02;
const PAR_ECR_DMA_ENABLE: u8 = 0x08;
const PAR_ECR_STORED: u8 = 0xfc;
const PAR_MODE_TEST: u8 = 6;
const PAR_MODE_CONFIG: u8 = 7;
const PAR_FIFO_DEPTH: usize = 16;
/// The part's fixed configuration: implementation id and reserved bits;
/// IRQ7 and DMA3.
const PAR_CONFIG_A: u8 = 0x1f;
const PAR_CONFIG_B: u8 = 0x0b;
// The status lines as the external SuperIO part reads them: bit 7 is the
// inverse of the Busy pin, bits 6 to 3 are nAck, PError, Select and nFault as
// the pins stand (nAck and nFault are active low). Bit 2, the part's PRINT
// indicator, is not modelled and reads clear.
const PAR_STATUS_NFAULT: u8 = 1 << 3;
const PAR_STATUS_SELECT: u8 = 1 << 4;
const PAR_STATUS_NACK: u8 = 1 << 6;
const PAR_STATUS_BUSY: u8 = 1 << 7;
/// Nothing on the connector: each of the five inputs sits at the level a
/// pull-up gives it, high, and Busy high reads as a clear bit 7. The pull
/// direction on the real boards is inferred; nothing measures an unplugged
/// port.
const PAR_STATUS_EMPTY: u8 = PAR_STATUS_NFAULT | PAR_STATUS_SELECT | 1 << 5 | PAR_STATUS_NACK;
/// A ready printer: Busy low (bit 7 set), nAck, Select and nFault high, no
/// paper error. Every host backend is taken as one that always accepts what
/// is sent, an approximation: a file or pty never holds Busy.
const PAR_STATUS_READY: u8 = PAR_STATUS_BUSY | PAR_STATUS_NACK | PAR_STATUS_SELECT | PAR_STATUS_NFAULT;

#[derive(Default)]
struct Parallel {
    addr_hi: [u32; 2],
    addr_lo: [u32; 2],
    count: [u32; 2],
    dma_control: u32,
    data: u8,
    control: u8,
    config_b: u8,
    extended: u8,
    fifo: VecDeque<u8>,
    backend: Option<Arc<dyn SerialBackend>>,
}

impl Parallel {
    fn mode(&self) -> u8 {
        self.extended >> 5
    }
}

// ---------------------------------------------------------------------------
// The DS1685 on the byte bus
// ---------------------------------------------------------------------------

/// How long the guest's NVRAM writes must pause before the file is
/// rewritten.
const NVRAM_QUIET: Duration = Duration::from_millis(500);
/// Registers 0x0e.. are storage rather than clock in the file.
const NVRAM_FIRST: usize = 0x0e;
/// The file is a 512-byte image: common registers 0x00..0x40,
/// bank-0 RAM 0x40..0x80, bank-1 registers 0x80..0xc0, extended RAM
/// 0xc0..0x140, zero padding after.
const NVRAM_IMAGE_LEN: usize = 0x200;
const NVRAM_OFF_NV: usize = 0x40;
const NVRAM_OFF_EXT: usize = 0x80;
const NVRAM_OFF_XRAM: usize = 0xc0;
/// Register A bit 4 selects bank 1 for registers 0x40..0x7f.
const RTC_A_DV0: u8 = 0x10;
/// Bank-1 extended RAM address and data ports.
const RTC_XRAM_ADDR: usize = 0x50;
const RTC_XRAM_DATA: usize = 0x53;
const RTC_XRAM_SIZE: usize = 128;

/// The clock, reached through an address port and a data port. The bank bit
/// switches only registers 0x40..0x7f: the four control registers and the
/// low RAM are the same cells in either bank. Bank 1 carries the extended
/// registers and, through the address and data ports at 0x50 and 0x53, 128
/// bytes of extended RAM, where the PROM keeps part of its environment.
///
/// The clock and the power-control and century registers are `MaceRtc`'s;
/// see the module notes for what the model leaves out (alarm and periodic
/// interrupts, the model number's side effects).
struct Rtc {
    dev: MaceRtc,
    index: u8,
    file: Option<std::path::PathBuf>,
    /// Bank-0 RAM at 0x40..0x7f.
    nv: [u8; 64],
    /// Bank-1 registers at 0x40..0x7f (0x48 and 0x4a live in `dev`).
    ext: [u8; 64],
    xram: [u8; RTC_XRAM_SIZE],
    /// When the RAM last changed, if the file has not caught up yet.
    dirty_since: Option<Instant>,
}

impl Rtc {
    fn new() -> Rtc {
        Rtc {
            dev: MaceRtc::new(),
            index: 0,
            file: None,
            nv: [0; 64],
            ext: [0; 64],
            xram: [0; RTC_XRAM_SIZE],
            dirty_since: None,
        }
    }

    fn bank1(&self) -> bool {
        self.dev.peek(0x0a) & RTC_A_DV0 != 0
    }

    /// Whether the selected register is one `MaceRtc` models itself.
    fn delegated(&self) -> bool {
        let i = self.index as usize;
        i < 0x40 || (self.bank1() && (i == 0x48 || i == 0x4a))
    }

    fn read(&self) -> u8 {
        let i = self.index as usize;
        if self.delegated() {
            self.dev.read8(((i as u64) << 8) | 7).data
        } else if !self.bank1() {
            self.nv[i - 0x40]
        } else if i == RTC_XRAM_DATA {
            self.xram[self.ext[RTC_XRAM_ADDR - 0x40] as usize & (RTC_XRAM_SIZE - 1)]
        } else {
            self.ext[i - 0x40]
        }
    }

    fn write(&mut self, v: u8) {
        let i = self.index as usize;
        if self.delegated() {
            self.dev.write8(((i as u64) << 8) | 7, v);
        } else if !self.bank1() {
            self.nv[i - 0x40] = v;
        } else if i == RTC_XRAM_DATA {
            let at = self.ext[RTC_XRAM_ADDR - 0x40] as usize & (RTC_XRAM_SIZE - 1);
            self.xram[at] = v;
        } else {
            self.ext[i - 0x40] = v;
        }
        // The file catches up once the writes stop: see `Ioc3::flush_nvram`.
        if i >= NVRAM_FIRST && self.file.is_some() {
            self.dirty_since = Some(Instant::now());
        }
    }

    fn image(&self) -> Vec<u8> {
        let mut img = vec![0u8; NVRAM_IMAGE_LEN];
        for (r, b) in img.iter_mut().enumerate().take(0x40) {
            *b = self.dev.peek(r);
        }
        img[NVRAM_OFF_NV..NVRAM_OFF_NV + 64].copy_from_slice(&self.nv);
        let mut ext = self.ext;
        ext[0x08] = self.dev.peek(0x48);
        ext[0x0a] = self.dev.peek(0x4a);
        img[NVRAM_OFF_EXT..NVRAM_OFF_EXT + 64].copy_from_slice(&ext);
        img[NVRAM_OFF_XRAM..NVRAM_OFF_XRAM + RTC_XRAM_SIZE].copy_from_slice(&self.xram);
        img
    }

    /// Load a 512-byte image. The clock registers (0x00..0x0d) are never
    /// loaded; time starts from the host. Returns false for any other length.
    fn load(&mut self, img: &[u8]) -> bool {
        if img.len() != NVRAM_IMAGE_LEN {
            return false;
        }
        for (r, &b) in img.iter().enumerate().take(0x40).skip(NVRAM_FIRST) {
            self.dev.poke(r, b);
        }
        self.nv.copy_from_slice(&img[NVRAM_OFF_NV..NVRAM_OFF_NV + 64]);
        self.ext.copy_from_slice(&img[NVRAM_OFF_EXT..NVRAM_OFF_EXT + 64]);
        self.xram.copy_from_slice(&img[NVRAM_OFF_XRAM..NVRAM_OFF_XRAM + RTC_XRAM_SIZE]);
        // Loading never powers the machine off.
        self.dev.poke(0x4a, self.ext[0x0a] & !0x08);
        true
    }

    /// The file and the image to put in it, if the RAM changed at least
    /// `quiet` ago; the RAM then counts as saved.
    fn take_dirty(&mut self, quiet: Duration) -> Option<(std::path::PathBuf, Vec<u8>)> {
        let since = self.dirty_since?;
        if since.elapsed() < quiet {
            return None;
        }
        self.dirty_since = None;
        Some((self.file.clone()?, self.image()))
    }
}

/// Say that an NVRAM save failed, once until a save succeeds again: the
/// guest's writes keep scheduling saves, and a full disk would otherwise
/// print on every one.
fn report_nvram_save(path: &std::path::Path, result: std::io::Result<()>) {
    static FAILING: AtomicBool = AtomicBool::new(false);
    match result {
        Ok(()) => FAILING.store(false, Ordering::Relaxed),
        Err(e) => {
            if !FAILING.swap(true, Ordering::Relaxed) {
                eprintln!("ioc3: NVRAM {}: {e}", path.display());
            }
        }
    }
}

/// Replace the NVRAM file in one step (a temporary file, synced and
/// renamed), so a kill mid-write leaves the previous image whole.
fn save_nvram(path: &std::path::Path, image: &[u8]) -> std::io::Result<()> {
    let name = path
        .to_str()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "NVRAM path is not UTF-8"))?;
    crate::nv_storage::save(name, image)
}

// ---------------------------------------------------------------------------
// Interrupt output
// ---------------------------------------------------------------------------

const IO_MASK: u32 = 0x4007_ffff;
const IO_COUNT: u32 = 0xffff;
const IO_MODE: u32 = 0x0007_0000;
const IO_LOW: u32 = 0;
const IO_HIGH: u32 = 0x0004_0000;
const IO_ONE_SHOT: u32 = 0x0005_0000;
const IO_PERIODIC: u32 = 0x0006_0000;
const IO_SQUARE: u32 = 0x0007_0000;
const IO_DIAG: u32 = 0x4000_0000;
const IO_STATE: u32 = 0x8000_0000;
const IO_TICK: Duration = Duration::from_nanos(7800);
const IO_PULSE_TICKS: u32 = 3;

#[derive(Default)]
struct IntOut {
    reg: u32,
    active: u32,
    state: bool,
    pulse_end: bool,
    square_started: bool,
    next: Option<Instant>,
}

fn int_out_valid(v: u32) -> bool {
    let mode = v & IO_MODE;
    let count = v & IO_COUNT;
    if v & IO_DIAG != 0 {
        return false;
    }
    if mode == IO_LOW || mode == IO_HIGH {
        return true;
    }
    // The internal strobe keeps count zero and still pulses.
    if mode == IO_ONE_SHOT && count == 0 {
        return true;
    }
    (2..=0xfffe).contains(&count)
        && (mode == IO_ONE_SHOT || (mode == IO_PERIODIC && count >= IO_PULSE_TICKS) || mode == IO_SQUARE)
}

// ---------------------------------------------------------------------------
// Chip state
// ---------------------------------------------------------------------------

/// Which sub-devices the board fits.
#[derive(Clone, Debug)]
pub struct Ioc3Config {
    pub subsystem_id: u16,
    /// The board's station address, which the one-wire part holds.
    pub mac: [u8; 6],
    pub serial: [bool; 2],
    pub parallel: bool,
    pub keyboard_mouse: bool,
    pub rtc: bool,
    /// GPIO pins that are board inputs, and their values.
    pub gpio_input_mask: u32,
    pub gpio_input_values: u32,
    /// Board-record parts on the MicroLAN, beside the Ethernet part:
    /// `(record text, ROM serial)`.
    pub records: Vec<(BoardRecordText, u64)>,
}

#[derive(Clone, Debug)]
pub struct BoardRecordText {
    pub barcode: String,
    pub part: String,
    pub revision: String,
    pub name: String,
}

impl BoardRecordText {
    fn new(barcode: &str, part: &str, revision: &str, name: &str) -> Self {
        Self { barcode: barcode.into(), part: part.into(), revision: revision.into(), name: name.into() }
    }
}

impl Ioc3Config {
    /// The Octane's IOC3 (`octane2` selects the Octane2's board records).
    /// The fictitious barcodes and ROM serials are the
    /// marked-synthetic ones (barcode `EMU...`).
    pub fn octane(mac: [u8; 6], octane2: bool) -> Self {
        let (fp_rev, ps_name, ps_part, ps_rev) = if octane2 {
            ("H", "PWR.SPPLY.ER", "060-0035-002", "A")
        } else {
            ("A", "PWR.SPPLY.SR", "060-0028-003", "A")
        };
        Self {
            subsystem_id: SUBSYS_IP30_SYSBOARD,
            mac,
            serial: [true, true],
            parallel: true,
            keyboard_mouse: true,
            rtc: true,
            // The board drives GPIO inputs 2 and 3 high: read on an Octane2's
            // PROM Command Monitor (pin data 0x12d). An Octane is unmeasured
            // and assumed the same; nothing shows a pin that tells them apart.
            gpio_input_mask: 1 << 2 | 1 << 3,
            gpio_input_values: 1 << 2 | 1 << 3,
            records: vec![
                (BoardRecordText::new("EMUF00", "030-0891-003", fp_rev, "FP1"), 0x5145_4d55_4600),
                (BoardRecordText::new("EMUS00", ps_part, ps_rev, ps_name), 0x5145_4d55_5300),
            ],
        }
    }
}

/// The wire the board carries: the station-address part and the board
/// records, the front plane and the power supply on the Octane.
fn default_wire(cfg: &Ioc3Config) -> w1::Bus {
    let mut bus = w1::Bus::new();
    bus.add(Box::new(sgi::ethernet_address_part(cfg.mac, sgi::ETHERNET_PART_SERIAL)));
    for (r, serial) in &cfg.records {
        let rec = BoardRecord { barcode: &r.barcode, part: &r.part, revision: &r.revision, name: &r.name };
        bus.add(Box::new(sgi::board_record_part(&rec, *serial)));
    }
    bus
}

struct St {
    cfg: Ioc3Config,
    int_status: u32,
    int_enable: u32,
    chip_control: u32,
    int_out: IntOut,
    microlan: u32,
    pin_control: u32,
    pin_data: u32,
    uart: [Uart; 2],
    sdma: Sdma,
    kbc: Kbc,
    par: Parallel,
    rtc: Rtc,
    ssram: Vec<u32>,
    /// The MicroLAN wire, built on first use (or handed in by the board).
    ow: Option<w1::Bus>,
    /// The DS1685 was told to power the machine off and the machine has not
    /// been told yet.
    power_off_pending: bool,
    dma: Option<Arc<Ioc3Dma>>,
}

const SSRAM_DATA: u32 = 0x0000_ffff;
const SSRAM_PARITY: u32 = 0x0001_0000;
const SSRAM_STORED: u32 = SSRAM_DATA | SSRAM_PARITY;
const SSRAM_PARITY_ERROR: u32 = 0x0002_0000;
/// MAC control bit 9: buffer RAM parity checking.
const ENET_MCR_RAM_PARITY: u32 = 1 << 9;

/// Say once that a narrow access to a 32-bit block was refused.
fn narrow_once(a: u32, w: usize) {
    static SAID: AtomicBool = AtomicBool::new(false);
    if !SAID.swap(true, Ordering::Relaxed) {
        log::warn!("ioc3: {w}-byte access at {a:#x} to a 32-bit register block refused (reads 0, writes dropped)");
    }
}

fn is_byte_wide(a: u32) -> bool {
    (reg::SUPERIO..reg::SUPERIO + reg::SUPERIO_SIZE).contains(&a) || a >= reg::BYTEBUS_DEV1 - 0x20000
}

/// What the chip needs from outside its own state during one access.
struct Ctx<'a> {
    ps2: &'a Ps2Controller,
    eth_parity: bool,
}

impl St {
    fn new(cfg: Ioc3Config) -> St {
        St {
            cfg,
            int_status: 0,
            int_enable: 0,
            chip_control: 0,
            int_out: IntOut::default(),
            microlan: 0,
            pin_control: 0,
            pin_data: 0,
            uart: [Uart::new(), Uart::new()],
            sdma: Sdma::default(),
            kbc: Kbc::default(),
            par: Parallel::default(),
            rtc: Rtc::new(),
            ssram: vec![0; (reg::SSRAM_SIZE / 4) as usize],
            ow: None,
            power_off_pending: false,
            dma: None,
        }
    }

    /// A PCI reset. The buffer RAM and the clock are memory, not state.
    fn reset(&mut self) {
        self.int_status = 0;
        self.int_enable = 0;
        self.sdma = Sdma::default();
        for u in &mut self.uart {
            u.reset();
        }
        self.microlan = 0;
        self.pin_control = 0;
        self.pin_data = 0;
        self.chip_control = 0;
        self.int_out = IntOut::default();
        self.kbc = Kbc::default();
        let backend = self.par.backend.take();
        self.par = Parallel::default();
        self.par.backend = backend;
    }

    fn set_bit(&mut self, mask: u32, level: bool) {
        if level {
            self.int_status |= mask;
        } else {
            self.int_status &= !mask;
        }
    }

    /// The host-time deadlines the chip holds, for the timers: the serial
    /// receive timers, transmit delays, the interrupt output while its status
    /// bit can reach the line, and the NVRAM file's debounce.
    fn deadlines(&self) -> Deadlines {
        let mut d = [None; KINDS];
        for (p, port) in self.sdma.port.iter().enumerate() {
            d[K_RX + p] = port.rx_deadline;
            // A paused port keeps its transmit delay frozen.
            d[K_TX_DELAY + p] = if port.paused { None } else { port.tx_delay_deadline };
        }
        if self.int_enable & int::RT_OUTPUT_PULSE != 0 {
            d[K_INT_OUT] = self.int_out.next;
        }
        d[K_NVRAM] = self.rtc.dirty_since.map(|t| t + NVRAM_QUIET);
        d
    }

    fn superio_level(&self) -> bool {
        self.int_status & self.int_enable != 0
    }

    // ----- the 16550 ports -----

    fn uart_changed(&mut self, p: usize) {
        let lvl = self.uart[p].irq();
        self.set_bit(int::serial(p, int::PASSTHROUGH), lvl);
        // On every level, not on assertion alone: the ring console receives
        // with the 16550's own interrupts disabled.
        if !self.sdma.port[p].busy {
            self.sdma_run(p);
        }
    }

    fn uart_read(&mut self, p: usize, r: u32) -> u8 {
        let mut v = self.uart[p].read(r);
        if r == uart_reg::MSR && self.uart[p].flow_cts(self.sdma.port[p].reg[0]) {
            v |= MSR_CTS;
        }
        self.uart_changed(p);
        v
    }

    fn uart_write(&mut self, p: usize, r: u32, v: u8) {
        self.uart[p].write(r, v);
        self.uart_changed(p);
    }

    // ----- serial ring DMA -----

    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.dma.as_ref().is_some_and(|d| d.dma_read(addr, buf).is_ok())
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> bool {
        self.dma.as_ref().is_some_and(|d| d.dma_write(addr, data).is_ok())
    }

    fn sdma_ctl(&self, p: usize) -> u32 {
        self.sdma.port[p].reg[0]
    }

    fn sdma_ring_size(&self) -> u32 {
        if self.sdma.base_lo & 1 != 0 { 4096 } else { 1024 }
    }

    fn sdma_mask(&self) -> u32 {
        (self.sdma_ring_size() - 1) & !(ENTRY as u32 - 1)
    }

    fn sdma_ring(&self, p: usize, dir: u32) -> u64 {
        let base = ((self.sdma.base_hi as u64) << 32) | (self.sdma.base_lo & 0xffff_f000) as u64;
        base.saturating_add(((p as u64 % 2) * 2 + dir as u64) * self.sdma_ring_size() as u64)
    }

    fn sdma_raise(&mut self, p: usize, src: u32) {
        self.int_status |= int::serial(p, src);
    }

    fn sdma_level(&mut self, p: usize, src: u32, level: bool) {
        self.set_bit(int::serial(p, src), level);
    }

    /// Chip control places each port's 16550 for its engine; aimed anywhere
    /// else, the engine does not reach the port.
    fn sdma_reachable(&self, p: usize) -> bool {
        if !self.cfg.serial[p] {
            return false;
        }
        let (shift, base) = if p == 1 {
            (CHIP_CTRL_PORT_B_SHIFT, reg::SERIAL_B)
        } else {
            (CHIP_CTRL_PORT_A_SHIFT, reg::SERIAL_A)
        };
        (self.chip_control >> shift) & CHIP_CTRL_PORT_FIELD == (base & 0xfff) / 8
    }

    fn sdma_running(&self, p: usize) -> bool {
        let c = self.sdma_ctl(p);
        c & sdma_ctl::ENABLE != 0 && c & sdma_ctl::PAUSE == 0
    }

    /// Transmit-ring-empty is a level: ring DMA on, nothing left in the ring
    /// or held by the engine.
    fn sdma_update_tx_empty(&mut self, p: usize) {
        let mask = self.sdma_mask();
        let port = &self.sdma.port[p];
        let empty = self.sdma_ctl(p) & sdma_ctl::ENABLE != 0
            && (port.reg[1] ^ port.reg[2]) & mask == 0
            && !port.tx_fetched
            && port.tx_delay_deadline.is_none();
        self.sdma_level(p, int::TX_RING_EMPTY, empty);
    }

    /// The receive high-water mark counts ring entries; a level. So is
    /// receive-ring-full: set while ring DMA is on and an entry waits that
    /// the ring had no room for, so an acknowledgement does not bring it back
    /// with every refused retry, and it falls once the consumer makes room,
    /// posted yet or not. A ring filled exactly, with nothing waiting, does
    /// not raise it. A model choice; what the part does is not known.
    fn sdma_update_high_water(&mut self, p: usize) {
        let mask = self.sdma_mask();
        let port = &self.sdma.port[p];
        let hw = self.sdma_ctl(p) & sdma_ctl::HIGH_WATER;
        let waiting = port.reg[3].wrapping_sub(port.reg[4]) & mask;
        let level = hw != 0 && waiting / ENTRY as u32 >= hw;
        if (port.reg[3] + ENTRY as u32) & mask != port.reg[4] & mask {
            self.sdma.port[p].post_refused = false;
        }
        let full = self.sdma_ctl(p) & sdma_ctl::ENABLE != 0 && self.sdma.port[p].post_refused;
        self.sdma_level(p, int::RX_HIGH_WATER, level);
        self.sdma_level(p, int::RX_RING_FULL, full);
    }

    /// Post the packer as one entry. Fails, keeping the characters, when the
    /// ring is full or the write is refused.
    fn sdma_post(&mut self, p: usize) -> bool {
        let mask = self.sdma_mask();
        let ring = self.sdma_ring(p, 1);
        let (produce, consume, packer) = {
            let port = &self.sdma.port[p];
            (port.reg[3], port.reg[4], port.packer)
        };
        let next = (produce + ENTRY as u32) & mask;
        if next == consume & mask {
            self.sdma.port[p].post_refused = true;
            return false;
        }
        if !self.dma_write(ring.saturating_add((produce & mask) as u64), &packer) {
            self.sdma_raise(p, int::DMA_ERROR);
            return false;
        }
        let port = &mut self.sdma.port[p];
        port.reg[3] = next;
        port.post_refused = false;
        port.packed = 0;
        port.packer = [0; ENTRY];
        true
    }

    fn sdma_rx_timer_start(&mut self, p: usize, since: Instant) {
        let port = &mut self.sdma.port[p];
        let reload = port.reg[5] & TIMER_RELOAD;
        port.rx_idle_count = 0;
        let dur = Duration::from_nanos(reload as u64 * TIMER_TICK_NS);
        if port.paused {
            // Started while paused, it waits frozen at its full count.
            port.rx_frozen = Some(dur);
        } else {
            port.rx_deadline = Some(since + dur);
        }
    }

    fn sdma_rx_timer_expired(&mut self, p: usize) {
        self.sdma.port[p].rx_idle_count = 0;
        self.sdma.port[p].rx_deadline = None;
        if self.sdma.port[p].packed != 0 {
            self.sdma_post(p);
        }
        // Raised at every expiry, even over an empty ring; the arm is spent.
        if self.sdma.port[p].armed {
            self.sdma_raise(p, int::RX_TIMER);
        }
        self.sdma.port[p].armed = false;
        self.sdma_update_high_water(p);
    }

    /// Expiries already due happen before an access.
    fn sdma_catch_up(&mut self, now: Instant) {
        for p in 0..2 {
            let delay = self.sdma.port[p].tx_delay_deadline;
            if let Some(d) = delay {
                if !self.sdma.port[p].paused && d <= now {
                    self.sdma_tx_delay_ended(p, d);
                }
            }
            if let Some(d) = self.sdma.port[p].rx_deadline {
                if d <= now {
                    self.sdma_rx_timer_expired(p);
                }
            }
        }
    }

    fn sdma_tx_delay_ended(&mut self, p: usize, when: Instant) {
        let irq = {
            let port = &mut self.sdma.port[p];
            port.tx_delay_deadline = None;
            std::mem::take(&mut port.tx_delay_irq)
        };
        if irq {
            self.sdma_raise(p, int::TX_EXPLICIT);
        }
        self.sdma_run_from(p, when);
    }

    fn sdma_receive(&mut self, p: usize, since: Instant) {
        while self.sdma_running(p) {
            if self.sdma.port[p].packed == ENTRY_CHARS && !self.sdma_post(p) {
                break;
            }
            // Line status before the character: reading it clears the latches.
            let lsr = self.uart[p].read_core(uart_reg::LSR);
            if lsr & LSR_DR == 0 {
                break;
            }
            let data = self.uart[p].read_core(uart_reg::RBR_THR);
            let mut msr = self.uart[p].read_core(uart_reg::MSR);
            if self.uart[p].flow_cts(self.sdma_ctl(p)) {
                msr |= MSR_CTS;
            }
            let mut status = RX_DATA_OK | RX_MODEM_OK;
            for (bit, flag) in [(LSR_OE, RX_OVERRUN), (LSR_PE, RX_PARITY), (LSR_FE, RX_FRAMING), (LSR_BI, RX_BREAK)] {
                if lsr & bit != 0 {
                    status |= flag;
                }
            }
            if msr & MSR_CTS != 0 {
                status |= RX_CTS;
            }
            if msr & MSR_DCD != 0 {
                status |= RX_CD;
            }
            let start_timer = {
                let port = &mut self.sdma.port[p];
                port.packer[port.packed] = data;
                port.packer[ENTRY_CHARS + port.packed] = status;
                port.packed += 1;
                port.armed && port.rx_deadline.is_none() && port.rx_frozen.is_none()
            };
            // Started by the first character after the arm; later ones leave
            // a running timer alone.
            if start_timer {
                self.sdma_rx_timer_start(p, since);
            }
            // A fourth character completes the entry; fewer wait for the
            // timer or a drain.
            if self.sdma.port[p].packed == ENTRY_CHARS {
                self.sdma_post(p);
            }
        }
        self.uart_pass_bit(p);
    }

    /// Keep the pass-through bit on the 16550's level without kicking the
    /// engine (it is already running).
    fn uart_pass_bit(&mut self, p: usize) {
        let lvl = self.uart[p].irq();
        self.set_bit(int::serial(p, int::PASSTHROUGH), lvl);
    }

    /// Each control byte types its data byte; the empty type ends the entry,
    /// wherever it sits.
    fn sdma_transmit(&mut self, p: usize, since: Instant) {
        let mask = self.sdma_mask();
        let ring = self.sdma_ring(p, 0);
        if self.sdma.port[p].tx_delay_deadline.is_some() {
            return;
        }
        loop {
            let (produce, consume) = {
                let port = &self.sdma.port[p];
                (port.reg[1], port.reg[2])
            };
            if !(self.sdma_running(p) && consume & mask != produce & mask) {
                break;
            }
            if !self.sdma.port[p].tx_fetched {
                let mut entry = [0u8; ENTRY];
                if !self.dma_read(ring.saturating_add((consume & mask) as u64), &mut entry) {
                    self.sdma_raise(p, int::DMA_ERROR);
                    break;
                }
                let port = &mut self.sdma.port[p];
                port.tx_entry = entry;
                port.tx_fetched = true;
            }
            let entry = self.sdma.port[p].tx_entry;
            let mut slot = self.sdma.port[p].tx_slot;
            while slot < ENTRY_CHARS {
                let control = entry[ENTRY_CHARS + slot];
                let ty = control & TX_TYPE;
                if ty == 0 {
                    break;
                }
                match ty {
                    TX_CHARACTER => {
                        let looped = self.uart[p].loopback();
                        // What has already arrived goes ahead of this one.
                        if looped {
                            self.sdma_receive(p, since);
                        }
                        self.uart[p].write(uart_reg::RBR_THR, entry[slot]);
                        // Approximation: a looped-back character arrives at
                        // once, where the part takes a character time.
                        self.sdma_receive(p, since);
                    }
                    TX_MODEM => self.uart[p].write(uart_reg::MCR, entry[slot]),
                    TX_DELAY => {
                        // The data byte is milliseconds of quiet.
                        if entry[slot] != 0 {
                            let port = &mut self.sdma.port[p];
                            port.tx_slot = slot + 1;
                            port.tx_delay_deadline = Some(since + Duration::from_millis(entry[slot] as u64));
                            port.tx_delay_irq = control & TX_INTERRUPT != 0;
                            return;
                        }
                    }
                    _ => unreachable!(),
                }
                slot += 1;
                self.sdma.port[p].tx_slot = slot;
                if control & TX_INTERRUPT != 0 {
                    self.sdma_raise(p, int::TX_EXPLICIT);
                }
            }
            let port = &mut self.sdma.port[p];
            port.tx_slot = 0;
            port.tx_fetched = false;
            port.reg[2] = (port.reg[2] + ENTRY as u32) & mask;
        }
        self.uart_pass_bit(p);
    }

    /// The drain posts what the packer holds and reads back set until it has;
    /// it waits while ring DMA is off or paused and while the ring is full.
    fn sdma_drain(&mut self, p: usize) {
        if !self.sdma.port[p].drain_pending || !self.sdma_running(p) {
            return;
        }
        if self.sdma.port[p].packed != 0 && !self.sdma_post(p) {
            return;
        }
        self.sdma.port[p].drain_pending = false;
    }

    fn sdma_run_from(&mut self, p: usize, since: Instant) {
        if !self.sdma_reachable(p) || self.sdma.port[p].busy {
            return;
        }
        self.sdma.port[p].busy = true;
        self.sdma_transmit(p, since);
        self.sdma_receive(p, since);
        self.sdma_drain(p);
        self.sdma_update_high_water(p);
        self.sdma_update_tx_empty(p);
        self.sdma.port[p].busy = false;
    }

    fn sdma_run(&mut self, p: usize) {
        self.sdma_run_from(p, Instant::now());
    }

    /// A passive mirror of the 16550's lines; no read side effects.
    fn sdma_shadow(&self, p: usize) -> u32 {
        let u = &self.uart[p];
        let lsr = u.core.lsr();
        let mcr = u.core.mcr();
        let mut msr = u.core.msr();
        if u.flow_cts(self.sdma_ctl(p)) {
            msr |= MSR_CTS;
        }
        let mut s = lsr as u32;
        for (m, bit) in [(MSR_DCTS, 16), (0x08, 19), (MSR_CTS, 20), (MSR_DCD, 23)] {
            if msr & m != 0 {
                s |= 1 << bit;
            }
        }
        for (m, bit) in [(MCR_DTR, 24), (MCR_RTS, 25), (MCR_OUT1, 26), (MCR_OUT2, 27), (MCR_LOOP, 28)] {
            if mcr & m != 0 {
                s |= 1 << bit;
            }
        }
        s
    }

    fn sdma_port_read(&mut self, p: usize, off: u32) -> u32 {
        let now = Instant::now();
        self.sdma_catch_up(now);
        if off == sdma_off::SHADOW {
            return self.sdma_shadow(p);
        }
        let port = &self.sdma.port[p];
        let mut v = port.reg[(off / 4) as usize];
        match off {
            sdma_off::CONTROL => {
                if v & sdma_ctl::PAUSE != 0 {
                    v |= sdma_ctl::PAUSED;
                }
                if port.tx_delay_deadline.is_some() {
                    v |= sdma_ctl::TX_TIMER_RUN;
                }
                if port.drain_pending {
                    v |= sdma_ctl::RX_TO_MEMORY;
                }
            }
            sdma_off::RX_CONSUME => {
                if port.armed {
                    v |= ARM_RX_TIMER;
                }
            }
            sdma_off::RX_PRODUCE => v |= (port.packed as u32) << HELD_SHIFT,
            sdma_off::RX_TIMER => {
                // Ticks left while running or frozen; idle, whatever a reload
                // write last loaded.
                let mut ticks = port.rx_idle_count;
                if let (true, Some(left)) = (port.paused, port.rx_frozen) {
                    ticks = (left.as_nanos() as u64 / TIMER_TICK_NS).min(TIMER_RELOAD as u64) as u32;
                } else if let Some(d) = port.rx_deadline {
                    let left = d.saturating_duration_since(now);
                    ticks = (left.as_nanos() as u64 / TIMER_TICK_NS).min(TIMER_RELOAD as u64) as u32;
                }
                v |= ticks << TIMER_COUNT_SHIFT;
            }
            _ => {}
        }
        v
    }

    fn sdma_port_write(&mut self, p: usize, off: u32, val: u32) {
        let now = Instant::now();
        self.sdma_catch_up(now);
        let mask = self.sdma_mask();
        match off {
            sdma_off::CONTROL => {
                self.sdma.port[p].reg[0] = val & !sdma_ctl::UNSTORED;
                if val & sdma_ctl::RESET != 0 {
                    // Resets the four indices and nothing else.
                    let port = &mut self.sdma.port[p];
                    port.reg[1..=4].fill(0);
                    port.armed = false;
                    port.drain_pending = false;
                    port.packed = 0;
                    port.post_refused = false;
                    port.packer = [0; ENTRY];
                    port.tx_slot = 0;
                    port.tx_fetched = false;
                    port.tx_delay_deadline = None;
                    port.tx_delay_irq = false;
                    port.rx_deadline = None;
                    port.rx_frozen = None;
                    port.rx_idle_count = 0;
                }
                if val & sdma_ctl::RX_TO_MEMORY != 0 {
                    self.sdma.port[p].drain_pending = true;
                }
                let port = &mut self.sdma.port[p];
                // A pause freezes the receive timer and a transmit delay;
                // resuming lets each finish the time it had left. A model
                // choice; no source says what the part does.
                if val & sdma_ctl::PAUSE != 0 && !port.paused {
                    port.paused = true;
                    port.rx_frozen = port.rx_deadline.take().map(|d| d.saturating_duration_since(now));
                    port.tx_delay_frozen = port.tx_delay_deadline.take().map(|d| d.saturating_duration_since(now));
                } else if val & sdma_ctl::PAUSE == 0 && port.paused {
                    port.paused = false;
                    port.rx_deadline = port.rx_frozen.take().map(|l| now + l);
                    port.tx_delay_deadline = port.tx_delay_frozen.take().map(|l| now + l);
                }
                if val & sdma_ctl::ENABLE == 0 {
                    port.rx_deadline = None;
                    port.rx_frozen = None;
                }
                self.sdma_run(p);
            }
            sdma_off::TX_PRODUCE | sdma_off::TX_CONSUME | sdma_off::RX_PRODUCE => {
                self.sdma.port[p].reg[(off / 4) as usize] = val & mask;
                self.sdma_run(p);
            }
            sdma_off::RX_CONSUME => {
                // Bit 31 sets the arm; a write with it clear leaves it.
                self.sdma.port[p].reg[4] = val & mask;
                if val & ARM_RX_TIMER != 0 {
                    self.sdma.port[p].armed = true;
                }
                self.sdma_run(p);
            }
            sdma_off::RX_TIMER => {
                let port = &mut self.sdma.port[p];
                port.reg[5] = val & TIMER_RELOAD;
                port.rx_idle_count = val & TIMER_RELOAD;
            }
            // The shadow discards writes silently.
            _ => {}
        }
        self.sdma_update_tx_empty(p);
        self.sdma_update_high_water(p);
    }
}

// ---------------------------------------------------------------------------
// Register decode
// ---------------------------------------------------------------------------

impl St {
    // ----- interrupt output -----

    fn io_drive(&mut self, level: bool) {
        if level && !self.int_out.state {
            self.int_status |= int::RT_OUTPUT_PULSE;
        }
        self.int_out.state = level;
    }

    fn io_hold_high(&mut self) {
        self.int_out.active = IO_HIGH;
        self.int_out.state = true;
        self.int_out.pulse_end = false;
        self.int_out.square_started = false;
        self.int_out.next = None;
    }

    /// One transition due at `at`; the next is scheduled from `at`, so a
    /// late one keeps the output's phase.
    fn io_step(&mut self, at: Instant) {
        let active = self.int_out.active;
        let count = active & IO_COUNT;
        match active & IO_MODE {
            IO_LOW => {
                self.io_drive(false);
                self.int_out.pulse_end = false;
            }
            IO_HIGH => {
                self.io_drive(true);
                self.int_out.pulse_end = false;
            }
            IO_ONE_SHOT => {
                if !self.int_out.pulse_end {
                    self.io_drive(true);
                    self.int_out.pulse_end = true;
                    self.int_out.next = Some(at + IO_TICK * IO_PULSE_TICKS);
                } else {
                    self.io_drive(false);
                    self.int_out.pulse_end = false;
                }
            }
            IO_PERIODIC => {
                if count < IO_PULSE_TICKS {
                    // No low phase: the output holds high.
                    self.io_drive(true);
                    self.io_hold_high();
                } else if !self.int_out.pulse_end {
                    self.io_drive(true);
                    self.int_out.pulse_end = true;
                    self.int_out.next = Some(at + IO_TICK * IO_PULSE_TICKS);
                } else {
                    self.io_drive(false);
                    self.int_out.pulse_end = false;
                    self.int_out.next = Some(at + IO_TICK * (count + 1 - IO_PULSE_TICKS));
                }
            }
            _ => {
                // IO_SQUARE
                if self.int_out.square_started {
                    let s = !self.int_out.state;
                    self.io_drive(s);
                } else {
                    self.int_out.square_started = true;
                }
                self.int_out.next = Some(at + IO_TICK * (count + 1));
            }
        }
    }

    /// Transitions already due happen before a read, not whenever the
    /// receive thread or a timer next runs.
    fn io_catch_up(&mut self, now: Instant) {
        for _ in 0..100_000 {
            match self.int_out.next {
                Some(due) if due <= now => {
                    self.int_out.next = None;
                    self.io_step(due);
                }
                _ => return,
            }
        }
        // Far behind (a stopped host): the sticky status bit is already set;
        // resume the phase from now.
        if self.int_out.next.is_some() {
            self.int_out.next = Some(now + IO_TICK);
        }
    }

    // ----- GPIO and MicroLAN -----

    fn pin_data_now(&self) -> u32 {
        (self.pin_data & !self.cfg.gpio_input_mask) | self.cfg.gpio_input_values
    }

    fn wire(&mut self) -> &mut w1::Bus {
        let cfg = &self.cfg;
        self.ow.get_or_insert_with(|| default_wire(cfg))
    }

    fn microlan_slot(&mut self, val: u32) {
        let level = if self.pin_control & (1 << 21) == 0 {
            // The engine is routed onto its pin by pin-control bit 21; with
            // it clear the bus is disconnected and reads high.
            true
        } else {
            let (pulse, sample) = w1::slot::fields(val as u64);
            w1::slot::slot(self.wire(), pulse, sample)
        };
        self.microlan = (val & !3) | 2 | level as u32;
    }

    // ----- main block -----

    fn main_read(&mut self, a: u32, c: &Ctx) -> u32 {
        if (reg::PIN_REG_FIRST..=reg::PIN_REG_LAST).contains(&a) && a & 3 == 0 {
            return (self.pin_data_now() >> ((a - reg::PIN_REG_FIRST) / 4)) & 1;
        }
        match a {
            reg::INT_STATUS => {
                self.sdma_catch_up(Instant::now());
                self.int_status
            }
            // The set alias reads back the live mask; the clear one does not.
            reg::INT_ENABLE_SET => self.int_enable,
            reg::CHIP_CONTROL => self.chip_control | CHIP_CTRL_ARB_IDLE,
            reg::INT_OUTPUT => {
                self.io_catch_up(Instant::now());
                self.int_out.reg | if self.int_out.state { IO_STATE } else { 0 }
            }
            reg::MICROLAN => self.microlan,
            // Both write aliases read the control word.
            reg::PIN_CONTROL_SET | reg::PIN_CONTROL_CLR => self.pin_control,
            reg::PIN_DATA => self.pin_data_now(),
            0x80..=0x98 => self.par_dma_read(a),
            0x9c | 0xa0 | 0xa4 => self.kbc_read(a, c.ps2),
            0xb0 => self.sdma.base_hi,
            0xb4 => self.sdma.base_lo,
            0xb8..=0xef => {
                let (p, off) = Self::sdma_port_of(a);
                self.sdma_port_read(p, off)
            }
            _ => {
                log::debug!("ioc3: read of unimplemented register {a:#x}");
                0
            }
        }
    }

    fn sdma_port_of(a: u32) -> (usize, u32) {
        if a >= reg::SDMA_PORT_B {
            (1, a - reg::SDMA_PORT_B)
        } else {
            (0, a - reg::SDMA_PORT_A)
        }
    }

    fn main_write(&mut self, a: u32, val: u32, c: &Ctx) {
        if (reg::PIN_REG_FIRST..=reg::PIN_REG_LAST).contains(&a) && a & 3 == 0 {
            let bit = 1 << ((a - reg::PIN_REG_FIRST) / 4);
            self.pin_data = (self.pin_data & !bit) | if val & 1 != 0 { bit } else { 0 };
            return;
        }
        match a {
            reg::INT_STATUS => {
                // An acknowledgement must not be undone by a late expiry.
                self.sdma_catch_up(Instant::now());
                // Write-one-to-clear; level-following bits ignore the write.
                self.int_status &= !(val & !int::LEVELS);
            }
            reg::INT_ENABLE_SET => self.int_enable |= val,
            reg::INT_ENABLE_CLR => self.int_enable &= !val,
            reg::CHIP_CONTROL => {
                self.chip_control = val & CHIP_CTRL_MASK;
                // The port offsets aim the engines; one aimed here may have
                // work waiting that no event will start.
                self.sdma_run(0);
                self.sdma_run(1);
            }
            reg::INT_OUTPUT => {
                let v = val & IO_MASK;
                // Stored as written; a command with no defined behaviour
                // leaves the running output alone.
                self.int_out.reg = v;
                if !int_out_valid(v) {
                    return;
                }
                let now = Instant::now();
                self.int_out.next = None;
                self.int_out.active = v & (IO_MODE | IO_COUNT);
                self.int_out.pulse_end = false;
                self.int_out.square_started = false;
                if v & IO_MODE == IO_ONE_SHOT {
                    self.int_out.state = false;
                }
                if v & IO_MODE == IO_LOW || v & IO_MODE == IO_HIGH {
                    // A fixed level is there for the next read.
                    self.io_step(now);
                } else {
                    self.int_out.next = Some(now + Duration::from_nanos(1));
                }
            }
            reg::MICROLAN => self.microlan_slot(val),
            reg::PIN_CONTROL_SET => self.pin_control |= val,
            reg::PIN_CONTROL_CLR => self.pin_control &= !val,
            reg::PIN_DATA => self.pin_data = val,
            0x80..=0x98 => self.par_dma_write(a, val),
            0x9c | 0xa8 | 0xac => self.kbc_write(a, val, c.ps2),
            0xb0 => self.sdma.base_hi = val,
            0xb4 => self.sdma.base_lo = val,
            0xb8..=0xef => {
                let (p, off) = Self::sdma_port_of(a);
                if self.cfg.serial[0] || self.cfg.serial[1] {
                    self.sdma_port_write(p, off, val);
                }
            }
            _ => log::debug!("ioc3: write of {val:#x} to unimplemented register {a:#x}"),
        }
    }

    // ----- PS/2 -----

    fn kbc_clamp(&self, port: usize) -> usize {
        let (one, three) = if port == 1 { (KBC_M_CLAMP_1, KBC_M_CLAMP_3) } else { (KBC_K_CLAMP_1, KBC_K_CLAMP_3) };
        if self.kbc.control & one != 0 {
            1
        } else if self.kbc.control & three != 0 {
            KBC_RX_SLOTS
        } else {
            0
        }
    }

    /// Move what the device has sent into this port's holding, up to what
    /// the clamp allows. A clamped port stops with its clock held low and
    /// the rest still in the device; an unclamped one loses the surplus and
    /// says so through the overflow flag. Every character that arrives
    /// latches the keyboard interrupt (a model choice: per character rather
    /// than a level over the holding).
    fn kbc_fill(&mut self, port: usize, ps2: &Ps2Controller) {
        if !self.cfg.keyboard_mouse {
            return;
        }
        let clamp = self.kbc_clamp(port);
        let mut arrived = false;
        loop {
            if clamp != 0 && self.kbc.port[port].count >= clamp {
                break;
            }
            let Some(byte) = ps2.pop_from(port == 1) else { break };
            let p = &mut self.kbc.port[port];
            if p.count >= KBC_RX_SLOTS {
                p.overflow = true;
                continue;
            }
            p.data[p.count] = byte;
            p.count += 1;
            arrived = true;
        }
        if arrived {
            self.int_status |= int::KEYBOARD;
        }
    }

    /// Report what the port holds, oldest character in the highest slot, and
    /// empty the holding; the clamp releases with it.
    fn kbc_receive(&mut self, port: usize, ps2: &Ps2Controller) -> u32 {
        let p = self.kbc.port[port];
        let mut val = 0u32;
        for (i, &b) in p.data.iter().enumerate().take(p.count) {
            let slot = (KBC_RX_SLOTS - 1) - i;
            val |= (b as u32) << (8 * slot);
            val |= 1 << (29 + slot);
        }
        if p.count != 0 && port == 1 {
            val |= KBC_RX_SOURCE_AUX;
        }
        if p.overflow {
            val |= KBC_RX_OVERFLOW;
        }
        self.kbc.port[port] = KbcPort::default();
        // The stopped device resumes inside this read; the part takes a
        // character time (an approximation).
        self.kbc_fill(port, ps2);
        val
    }

    fn kbc_read(&mut self, a: u32, ps2: &Ps2Controller) -> u32 {
        match a {
            KBC_CONTROL => self.kbc.control,
            KBC_KBD_RECEIVE => self.kbc_receive(0, ps2),
            KBC_AUX_RECEIVE => self.kbc_receive(1, ps2),
            _ => 0,
        }
    }

    fn kbc_write(&mut self, a: u32, val: u32, ps2: &Ps2Controller) {
        match a {
            KBC_CONTROL => {
                self.kbc.control = val & KBC_CONTROL_MASK;
                // Releasing a clamp lets a stopped device send again.
                self.kbc_fill(0, ps2);
                self.kbc_fill(1, ps2);
            }
            KBC_KBD_TRANSMIT | KBC_AUX_TRANSMIT => {
                if !self.cfg.keyboard_mouse {
                    return;
                }
                ps2.write_to(val as u8, a == KBC_AUX_TRANSMIT);
                self.kbc_fill(0, ps2);
                self.kbc_fill(1, ps2);
            }
            _ => {}
        }
    }

    // ----- parallel port -----

    fn par_status(&self) -> u32 {
        let mut s = 0;
        for (i, &count) in self.par.count.iter().enumerate() {
            let shift = i * 3;
            if count & PAR_COUNT_ARM != 0 {
                s |= 1 << shift;
            }
            if count & PAR_COUNT_LAST != 0 {
                s |= 2 << shift;
            }
        }
        s
    }

    fn par_dma_read(&self, a: u32) -> u32 {
        match a {
            0x80 => self.par.addr_hi[0],
            0x84 => self.par.addr_lo[0],
            0x88 => self.par.count[0],
            0x8c => self.par.dma_control | self.par_status(),
            0x90 => self.par.addr_hi[1],
            0x94 => self.par.addr_lo[1],
            0x98 => self.par.count[1],
            _ => 0,
        }
    }

    fn par_dma_write(&mut self, a: u32, val: u32) {
        match a {
            0x80 => self.par.addr_hi[0] = val,
            0x84 => self.par.addr_lo[0] = val,
            0x88 => {
                self.par.count[0] = val & PAR_COUNT_STORED;
                self.par_dma_run();
            }
            0x8c => {
                self.par.dma_control = val & PAR_DMA_STORED;
                self.par_dma_run();
            }
            0x90 => self.par.addr_hi[1] = val,
            0x94 => self.par.addr_lo[1] = val,
            0x98 => {
                self.par.count[1] = val & PAR_COUNT_STORED;
                self.par_dma_run();
            }
            _ => {}
        }
    }

    /// Wait for both enables, then drain context A, then B. Output goes to
    /// the backend synchronously; there is no peripheral to read from.
    fn par_dma_run(&mut self) {
        if self.par.dma_control & PAR_DMA_ENABLE == 0
            || self.par.dma_control & PAR_DMA_RESET != 0
            || self.par.extended & PAR_ECR_DMA_ENABLE == 0
        {
            return;
        }
        for i in 0..2 {
            if self.par.count[i] & PAR_COUNT_ARM == 0 {
                continue;
            }
            if self.par.dma_control & PAR_DMA_FROM_PERIPHERAL != 0 {
                log::debug!("ioc3: parallel context {i}: no peripheral to read from");
                return;
            }
            let addr = ((self.par.addr_hi[i] as u64) << 32) | self.par.addr_lo[i] as u64;
            let n = (self.par.count[i] & PAR_COUNT) as usize;
            let mut buf = vec![0u8; n];
            if n != 0 && !self.dma_read(addr, &mut buf) {
                // The context stays armed and the chain stops.
                self.int_status |= int::PARALLEL_ERROR;
                return;
            }
            if let Some(b) = &self.par.backend {
                for &x in &buf {
                    b.send_byte(x);
                }
            }
            self.par.count[i] &= !(PAR_COUNT | PAR_COUNT_ARM);
            // The context's own source latches; the driver clears it.
            self.int_status |= 1 << (int::PARALLEL_CONTEXT + i as u32);
            if self.par.count[i] & PAR_COUNT_LAST != 0 {
                self.int_status |= int::PARALLEL_SERVICE;
            }
        }
    }

    fn par_ecp_read(&mut self, r: u32) -> u8 {
        let p = &mut self.par;
        match r {
            0 => p.data,
            // The lines follow the cable: a peripheral, or pull-ups on nothing.
            1 if p.backend.is_some() => PAR_STATUS_READY,
            1 => PAR_STATUS_EMPTY,
            2 => p.control,
            8 => {
                if p.mode() == PAR_MODE_CONFIG {
                    PAR_CONFIG_A
                } else if p.mode() == PAR_MODE_TEST {
                    p.fifo.pop_front().unwrap_or(0)
                } else {
                    0
                }
            }
            9 => if p.mode() == PAR_MODE_CONFIG { PAR_CONFIG_B } else { p.config_b },
            0xa => {
                if p.mode() == PAR_MODE_TEST {
                    let mut s = p.extended;
                    if p.fifo.is_empty() {
                        s |= PAR_ECR_FIFO_EMPTY;
                    }
                    if p.fifo.len() >= PAR_FIFO_DEPTH {
                        s |= PAR_ECR_FIFO_FULL;
                    }
                    s
                } else {
                    p.extended | PAR_ECR_FIFO_EMPTY
                }
            }
            _ => 0,
        }
    }

    fn par_ecp_write(&mut self, r: u32, v: u8) {
        match r {
            0 => self.par.data = v,
            2 => self.par.control = v,
            8 => {
                let p = &mut self.par;
                match p.mode() {
                    // The fitted part's configuration is read-only.
                    PAR_MODE_CONFIG => {}
                    PAR_MODE_TEST => {
                        // The part's behaviour on overrun is unspecified;
                        // dropping the oldest byte is a model choice.
                        if p.fifo.len() >= PAR_FIFO_DEPTH {
                            p.fifo.pop_front();
                        }
                        p.fifo.push_back(v);
                    }
                    _ => {
                        if let Some(b) = &p.backend {
                            b.send_byte(v);
                        }
                    }
                }
            }
            9 => {
                if self.par.mode() != PAR_MODE_CONFIG {
                    self.par.config_b = v;
                }
            }
            0xa => {
                // Acknowledge service before starting DMA so a new
                // completion can assert it again.
                self.int_status &= !int::PARALLEL_SERVICE;
                self.par.extended = v & PAR_ECR_STORED;
                let m = self.par.mode();
                if m == 0 || m == 1 {
                    self.par.fifo.clear();
                }
                self.par_dma_run();
            }
            _ => {}
        }
    }

    // ----- byte-wide windows -----

    fn read_byte(&mut self, a: u32) -> u8 {
        if (reg::SUPERIO..reg::SUPERIO + reg::SUPERIO_SIZE).contains(&a) {
            for (p, base) in [(0, reg::SERIAL_A), (1, reg::SERIAL_B)] {
                if (base..base + 8).contains(&a) && self.cfg.serial[p] {
                    return self.uart_read(p, a - base);
                }
            }
            if (reg::PAR_ECP_BASE..reg::PAR_ECP_BASE + 0x0b).contains(&a) && self.cfg.parallel {
                return self.par_ecp_read(a - reg::PAR_ECP_BASE);
            }
            return 0;
        }
        if self.cfg.rtc {
            // Each select is a byte-wide port answering on any lane of its
            // first word: the guests land on different ones.
            if (reg::BYTEBUS_DEV2..reg::BYTEBUS_DEV2 + 4).contains(&a) {
                return self.rtc.read();
            }
            if (reg::BYTEBUS_DEV1..reg::BYTEBUS_DEV1 + 4).contains(&a) {
                // A write port on the part; the latch is what is there.
                return self.rtc.index;
            }
        }
        0
    }

    fn write_byte(&mut self, a: u32, v: u8) {
        if (reg::SUPERIO..reg::SUPERIO + reg::SUPERIO_SIZE).contains(&a) {
            for (p, base) in [(0, reg::SERIAL_A), (1, reg::SERIAL_B)] {
                if (base..base + 8).contains(&a) && self.cfg.serial[p] {
                    self.uart_write(p, a - base, v);
                    return;
                }
            }
            if (reg::PAR_ECP_BASE..reg::PAR_ECP_BASE + 0x0b).contains(&a) && self.cfg.parallel {
                self.par_ecp_write(a - reg::PAR_ECP_BASE, v);
            }
            return;
        }
        if self.cfg.rtc {
            if (reg::BYTEBUS_DEV2..reg::BYTEBUS_DEV2 + 4).contains(&a) {
                let was = self.rtc.dev.powered_off();
                self.rtc.write(v);
                if !was && self.rtc.dev.powered_off() {
                    self.power_off_pending = true;
                }
            } else if (reg::BYTEBUS_DEV1..reg::BYTEBUS_DEV1 + 4).contains(&a) {
                // Seven bits; the bank comes from register A.
                self.rtc.index = v & 0x7f;
            }
        }
    }

    // ----- anywhere in the BAR -----

    /// A read of `w` bytes at internal offset `a`.
    fn read(&mut self, a: u32, w: usize, c: &Ctx) -> u32 {
        if is_byte_wide(a) {
            // The windows are byte-wide: a wider access is several registers.
            let mut v = 0;
            for i in 0..w as u32 {
                v |= (self.read_byte(a + i) as u32) << (8 * i);
            }
            return v;
        }
        // The 32-bit blocks declare both their valid and implemented access
        // size as 4 in the QEMU model, so a narrower access is refused by the
        // bus there (not widened to a lane). There is no bus error to return
        // from `bar_read`: it reads zero, and the write below is dropped.
        if w != 4 {
            narrow_once(a, w);
            return 0;
        }
        if (reg::SSRAM..reg::SSRAM + reg::SSRAM_SIZE).contains(&a) {
            let mut word = self.ssram[((a - reg::SSRAM) / 4) as usize];
            if c.eth_parity && word.count_ones() & 1 != 0 {
                word |= SSRAM_PARITY_ERROR;
            }
            word
        } else if (reg::SDMA_ALIAS_A..reg::SDMA_ALIAS_B + reg::SDMA_ALIAS_SIZE).contains(&a) {
            let p = (a >= reg::SDMA_ALIAS_B) as usize;
            let off = a & (reg::SDMA_ALIAS_SIZE - 1) & !3;
            if off >= sdma_off::REGS * 4 || !self.cfg.serial[p] {
                0
            } else {
                self.sdma_port_read(p, off)
            }
        } else if a < 0x200 {
            self.main_read(a & !3, c)
        } else {
            0
        }
    }

    fn write(&mut self, a: u32, v: u32, w: usize, c: &Ctx) {
        if is_byte_wide(a) {
            for i in 0..w as u32 {
                self.write_byte(a + i, (v >> (8 * i)) as u8);
            }
            return;
        }
        if w != 4 || a & 3 != 0 {
            narrow_once(a, w);
            return;
        }
        if (reg::SSRAM..reg::SSRAM + reg::SSRAM_SIZE).contains(&a) {
            let mut word = v & SSRAM_STORED;
            // With checking enabled, bit 17 asks for generated stored parity.
            if c.eth_parity && v & SSRAM_PARITY_ERROR != 0 {
                word &= !SSRAM_PARITY;
                if (word & SSRAM_DATA).count_ones() & 1 != 0 {
                    word |= SSRAM_PARITY;
                }
            }
            self.ssram[((a - reg::SSRAM) / 4) as usize] = word;
        } else if (reg::SDMA_ALIAS_A..reg::SDMA_ALIAS_B + reg::SDMA_ALIAS_SIZE).contains(&a) {
            let p = (a >= reg::SDMA_ALIAS_B) as usize;
            let off = a & (reg::SDMA_ALIAS_SIZE - 1);
            // Three of the seven are read-only through a page: the two
            // indexes the engine advances and the 16550 shadow.
            if off >= sdma_off::REGS * 4
                || !self.cfg.serial[p]
                || matches!(off, sdma_off::TX_CONSUME | sdma_off::RX_PRODUCE | sdma_off::SHADOW)
            {
                return;
            }
            self.sdma_port_write(p, off, v);
        } else if a < 0x200 {
            self.main_write(a, v, c);
        }
    }
}

// ---------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------

/// A PS/2 device model's "data waiting" line, which tells the chip a
/// character has arrived.
struct KbcSink(Weak<Shared>);

impl InterruptController for KbcSink {
    fn set_line(&self, _line: u32, _level: bool) {
        if let Some(sh) = self.0.upgrade() {
            sh.kbc_poke();
        }
    }
}

/// How long a serial receive thread sleeps when its backend has nothing:
/// `SerialBackend` has no blocking read to wait on.
const RX_IDLE: Duration = Duration::from_millis(2);
/// The interrupt output can ask for a transition every few microseconds; a
/// timer never re-arms sooner than this, and the catch-up that follows applies
/// every transition due in the meantime.
const IO_REARM_MIN: Duration = Duration::from_micros(100);

/// The deadlines the chip keeps on host time, one timer slot each.
const K_RX: usize = 0;
const K_TX_DELAY: usize = 2;
const K_INT_OUT: usize = 4;
const K_NVRAM: usize = 5;
const KINDS: usize = 6;
type Deadlines = [Option<Instant>; KINDS];

/// A rewrite of the NVRAM file waiting for the writer thread.
type NvramJob = (std::path::PathBuf, Vec<u8>);

/// Lock order: `gate`, then `line_order`, then `st`; `armed`, the line and
/// Ethernet slots (`pin_line`, `superio_line`, `eth`, `raw_dma`) and the
/// NVRAM hand-off are leaves.
/// The serial and parallel engines move their rings with `st` held: the
/// DMA goes down through the Bridge and HEART, which never call back into
/// the chip, and an access to the Ethernet block goes to it with `st`
/// released.
struct Shared {
    pci: PciConfig,
    st: Mutex<St>,
    ps2: Arc<Ps2Controller>,
    eth: Mutex<Option<Arc<dyn Ioc3EthPort>>>,
    raw_dma: Mutex<Option<Arc<dyn PciDma>>>,
    /// PCI interrupt pin A: the Ethernet block.
    pin_line: Mutex<Option<IrqLine>>,
    /// The SuperIO line the board wires to a Bridge input.
    superio_line: Mutex<Option<IrqLine>>,
    /// The status bar's activity word, shared with the graphics board.
    heartbeat: OnceLock<Arc<AtomicU64>>,
    /// What each line was last driven to, and the Ethernet block's level.
    pin_level: AtomicBool,
    superio_level: AtomicBool,
    eth_level: AtomicBool,
    /// Serialises driving the two lines; see [`Shared::sync_superio`].
    line_order: Mutex<()>,
    /// A PS/2 character arrived while an access held the chip.
    kbc_dirty: AtomicBool,
    me: Weak<Shared>,
    timers: OnceLock<Arc<TimerManager>>,
    events: OnceLock<SyncSender<MachineEvent>>,
    /// Between `start` and `stop`.
    running: AtomicBool,
    /// Held by a timer callback while it runs, so `stop` can wait one out.
    gate: Mutex<()>,
    /// The timer pending for each deadline kind, and the deadline it serves.
    armed: Mutex<[Option<(TimerId, Instant)>; KINDS]>,
    /// The NVRAM image the writer thread has yet to put in the file.
    nvram_job: Mutex<Option<NvramJob>>,
    nvram_ready: Condvar,
}

impl Shared {
    /// Put the SuperIO line at the level the chip's state gives, then pick up
    /// any PS/2 character that arrived while the chip was held (a poke sets
    /// the flag before it tries the lock, so whoever holds the lock sees it
    /// here, after releasing).
    fn sync_superio(&self) {
        loop {
            self.drive_superio();
            if !self.kbc_dirty.swap(false, Ordering::SeqCst) {
                break;
            }
            let mut st = self.st.lock();
            st.kbc_fill(0, &self.ps2);
            st.kbc_fill(1, &self.ps2);
        }
    }

    /// The level is read after taking `line_order`, so whichever thread
    /// drives last has seen the latest state; two threads cannot leave the
    /// line stale. The order is always `line_order` then `st`, and `st` is
    /// never held when this is called.
    fn drive_superio(&self) {
        let _order = self.line_order.lock();
        let level = self.st.lock().superio_level();
        if self.superio_level.swap(level, Ordering::SeqCst) != level {
            if let Some(l) = self.superio_line.lock().as_ref() {
                l.set(level);
            }
        }
    }

    /// The same for the PCI pin, which follows the Ethernet block's level.
    fn sync_pin(&self) {
        let _order = self.line_order.lock();
        let level = self.eth_level.load(Ordering::SeqCst);
        if self.pin_level.swap(level, Ordering::SeqCst) != level {
            if let Some(l) = self.pin_line.lock().as_ref() {
                l.set(level);
            }
        }
    }

    /// Run `f` with the chip locked, then arm the timers for the deadlines it
    /// left and drive the lines from its state.
    fn with_st<R>(&self, parity: bool, f: impl FnOnce(&mut St, &Ctx) -> R) -> R {
        let (r, want, power_off) = {
            let mut st = self.st.lock();
            let ctx = Ctx { ps2: &self.ps2, eth_parity: parity };
            let pins = st.pin_data;
            let r = f(&mut st, &ctx);
            self.show_lightbar(&st, pins);
            (r, st.deadlines(), std::mem::take(&mut st.power_off_pending))
        };
        self.arm(want);
        self.sync_superio();
        if power_off {
            if let Some(tx) = self.events.get() {
                let _ = tx.try_send(MachineEvent::PowerOff);
            }
        }
        r
    }

    /// Mirror the system board's front lightbar into the status bar when
    /// pins 0 and 1 changed from `before`.
    fn show_lightbar(&self, st: &St, before: u32) {
        use crate::dev::ng1::rex3::Rex3;
        let Some(hb) = self.heartbeat.get() else { return };
        if st.cfg.subsystem_id != SUBSYS_IP30_SYSBOARD {
            return;
        }
        // Pin 0 is the white power LED, pin 1 the red fault LED (Linux
        // drivers/leds/leds-ip30.c writes 1 to light one). The status bar
        // shows white as its green LED. Set to false if the pins turn out
        // to be active low.
        const LIT_WHEN_HIGH: bool = true;
        for (pin, led) in [(0, Rex3::HB_LED_GREEN), (1, Rex3::HB_LED_RED)] {
            let (old, new) = ((before >> pin) & 1, (st.pin_data >> pin) & 1);
            if old == new {
                continue;
            }
            if (new == 1) == LIT_WHEN_HIGH {
                hb.fetch_or(led, Ordering::Relaxed);
            } else {
                hb.fetch_and(!led, Ordering::Relaxed);
            }
        }
    }

    /// Make the pending timers match `want`: one one-shot per deadline. A
    /// timer whose deadline moved is replaced; with none set or the chip not
    /// running, deadlines are only evaluated lazily, by the next access.
    fn arm(&self, want: Deadlines) {
        let Some(tm) = self.timers.get() else { return };
        let mut armed = self.armed.lock();
        // Checked under the lock `stop` clears the slots with: a pass that
        // sees `running` set has its timers removed by that stop.
        if !self.running.load(Ordering::SeqCst) {
            return;
        }
        for (k, w) in want.iter().enumerate() {
            if armed[k].map(|(_, d)| d) == *w {
                continue;
            }
            if let Some((id, _)) = armed[k].take() {
                tm.remove(id);
            }
            let Some(due) = *w else { continue };
            let mut delay = due.saturating_duration_since(Instant::now());
            if k == K_INT_OUT {
                delay = delay.max(IO_REARM_MIN);
            }
            let id = tm.add_one_shot(delay, self.me.clone(), move |me| Shared::fire(me, k, due));
            armed[k] = Some((id, due));
        }
    }

    /// A deadline passed with no guest access to notice it. Runs on the timer
    /// thread, shared with the CPU's compare: short, no waiting beyond the
    /// chip's own locks.
    fn fire(me: &Weak<Shared>, kind: usize, due: Instant) -> TimerReturn {
        let Some(sh) = me.upgrade() else { return TimerReturn::Delete };
        let _gate = sh.gate.lock();
        if !sh.running.load(Ordering::SeqCst) {
            return TimerReturn::Delete;
        }
        {
            let mut armed = sh.armed.lock();
            if armed[kind].map(|(_, d)| d) == Some(due) {
                armed[kind] = None;
            }
        }
        if kind == K_NVRAM {
            // The file is written by its own thread, not here.
            if let Some(job) = sh.st.lock().rtc.take_dirty(NVRAM_QUIET) {
                *sh.nvram_job.lock() = Some(job);
                sh.nvram_ready.notify_all();
            }
            sh.rearm();
        } else {
            sh.with_st(false, |st, _| {
                let now = Instant::now();
                st.sdma_catch_up(now);
                st.io_catch_up(now);
            });
        }
        TimerReturn::Delete
    }

    /// Arm for whatever deadlines the chip holds now.
    fn rearm(&self) {
        let want = self.st.lock().deadlines();
        self.arm(want);
    }

    /// Feed host input from serial backend `port` into its FIFO until the
    /// chip stops.
    fn rx_loop(me: Weak<Shared>, port: usize) {
        loop {
            {
                let Some(sh) = me.upgrade() else { return };
                if !sh.running.load(Ordering::SeqCst) {
                    return;
                }
                sh.with_st(false, |st, _| {
                    if st.uart[port].pump() {
                        st.uart_changed(port);
                    }
                });
            }
            std::thread::park_timeout(RX_IDLE);
        }
    }

    /// Write the NVRAM file when the timer hands over an image.
    fn nvram_loop(me: Weak<Shared>) {
        loop {
            let job = {
                let Some(sh) = me.upgrade() else { return };
                let mut slot = sh.nvram_job.lock();
                while slot.is_none() && sh.running.load(Ordering::SeqCst) {
                    sh.nvram_ready.wait(&mut slot);
                }
                match slot.take() {
                    Some(j) => j,
                    None => return,
                }
            };
            report_nvram_save(&job.0, save_nvram(&job.0, &job.1));
        }
    }

    /// A device raised its line. If an access holds the chip (possibly this
    /// very thread, from inside a transmit), that access picks the character
    /// up when it finishes: the flag is set before the lock is tried, so
    /// either this thread gets the lock or the holder sees the flag.
    fn kbc_poke(&self) {
        self.kbc_dirty.store(true, Ordering::SeqCst);
        if let Some(mut st) = self.st.try_lock() {
            if self.kbc_dirty.swap(false, Ordering::SeqCst) {
                st.kbc_fill(0, &self.ps2);
                st.kbc_fill(1, &self.ps2);
            }
            drop(st);
            self.sync_superio();
        }
    }

    /// Hand the Ethernet block its way to memory and its interrupt, once
    /// both it and the bridge's DMA are in hand.
    fn connect_eth(self: &Arc<Self>) {
        let eth = self.eth.lock().clone();
        let dma = self.raw_dma.lock().clone();
        if let (Some(e), Some(d)) = (eth, dma) {
            let weak = Arc::downgrade(self);
            e.connect(
                Arc::new(Ioc3Dma::new(d)),
                Box::new(move |level| {
                    if let Some(sh) = weak.upgrade() {
                        sh.eth_level.store(level, Ordering::SeqCst);
                        sh.sync_pin();
                    }
                }),
            );
        }
    }

    fn eth(&self) -> Option<Arc<dyn Ioc3EthPort>> {
        self.eth.lock().clone()
    }

    /// Whether buffer-RAM parity checking is on, for an access to the RAM.
    fn parity_for(&self, a: u32) -> bool {
        if !(reg::SSRAM..reg::SSRAM + reg::SSRAM_SIZE).contains(&a) {
            return false;
        }
        self.eth().is_some_and(|e| e.read32(reg::ENET_MAC_CONTROL) & ENET_MCR_RAM_PARITY != 0)
    }

    fn flush_nvram(&self, quiet: Duration) {
        let Some((path, image)) = self.st.lock().rtc.take_dirty(quiet) else { return };
        report_nvram_save(&path, save_nvram(&path, &image));
    }

    /// One synchronous pass of what the receive threads and timers do:
    /// host input into the FIFOs, expired receive timers and transmit
    /// delays, the interrupt output, and any PS/2 character that arrived.
    #[cfg(test)]
    fn service(&self) {
        self.with_st(false, |st, _| {
            let now = Instant::now();
            for p in 0..2 {
                if st.cfg.serial[p] {
                    st.uart[p].pump();
                    st.uart_changed(p);
                }
            }
            st.sdma_catch_up(now);
            for p in 0..2 {
                st.sdma_run(p);
            }
            st.io_catch_up(now);
        });
        self.kbc_poke();
    }
}

/// The IOC3 PCI function.
pub struct Ioc3 {
    sh: Arc<Shared>,
    /// The serial receive threads and the NVRAM writer, joined in `stop`.
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Ioc3 {
    pub fn new(cfg: Ioc3Config) -> Ioc3 {
        let subsystem = cfg.subsystem_id;
        let sh = Arc::new_cyclic(|weak: &Weak<Shared>| {
            let line = IrqLine::new(Arc::new(KbcSink(weak.clone())), 0);
            let ps2 = Arc::new(Ps2Controller::new(Some(line)));
            // Host key and mouse events are accepted from the start (this
            // only sets a flag); `stop` and `start` follow the chip's.
            Device::start(&*ps2);
            let pci = PciConfig::new(
                VENDOR_ID,
                DEVICE_ID,
                0xff_0000,
                REVISION,
                [Bar::Mem(BAR_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                PCI_INTA,
            );
            // The subsystem identity selects what a guest instantiates.
            pci.write(11, ((subsystem as u32) << 16) | VENDOR_ID as u32);
            Shared {
                pci,
                st: Mutex::new(St::new(cfg)),
                ps2,
                eth: Mutex::new(None),
                raw_dma: Mutex::new(None),
                pin_line: Mutex::new(None),
                superio_line: Mutex::new(None),
                heartbeat: OnceLock::new(),
                pin_level: AtomicBool::new(false),
                superio_level: AtomicBool::new(false),
                eth_level: AtomicBool::new(false),
                line_order: Mutex::new(()),
                kbc_dirty: AtomicBool::new(false),
                me: weak.clone(),
                timers: OnceLock::new(),
                events: OnceLock::new(),
                running: AtomicBool::new(false),
                gate: Mutex::new(()),
                armed: Mutex::new([None; KINDS]),
                nvram_job: Mutex::new(None),
                nvram_ready: Condvar::new(),
            }
        });
        Ioc3 { sh, threads: Mutex::new(Vec::new()) }
    }

    /// The Octane's (or Octane2's) onboard IOC3: both serial ports, PS/2,
    /// parallel, the DS1685 and the Ethernet-address and board-record parts.
    pub fn octane(mac: [u8; 6], octane2: bool) -> Ioc3 {
        Ioc3::new(Ioc3Config::octane(mac, octane2))
    }

    /// The timer manager the deadlines are armed on. Call before `start`.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        let _ = self.sh.timers.set(tm);
    }

    /// Where the DS1685's power-off goes: one `PowerOff` event when the guest
    /// writes it. Call before `start`.
    pub fn set_event_sender(&self, tx: SyncSender<MachineEvent>) {
        let _ = self.sh.events.set(tx);
    }

    /// Replace the MicroLAN wire the chip builds for itself (the station
    /// address and the board records).
    pub fn set_microlan(&self, bus: w1::Bus) {
        self.sh.st.lock().ow = Some(bus);
    }

    /// Wire the host end of serial port `port` (0 is the console): a TCP
    /// listener, a log tee, or nothing.
    pub fn set_serial_backend(&self, port: usize, backend: Arc<dyn SerialBackend>) {
        self.sh.st.lock().uart[port].core.set_backend(backend);
    }

    /// Where parallel-port output goes. Without one it is dropped. No machine
    /// wires one, so only the tests set it.
    #[cfg(test)]
    pub fn set_parallel_backend(&self, backend: Arc<dyn SerialBackend>) {
        self.sh.st.lock().par.backend = Some(backend);
    }

    /// Back the DS1685's RAM (user RAM, bank-1 registers and the 128-byte
    /// extended RAM) with a file. A 512-byte file (the layout the QEMU model
    /// uses) is loaded. The file is
    /// written as 512 bytes now and replaced whole once the guest's writes
    /// to the RAM pause (see [`Ioc3::flush_nvram`]). The clock starts from
    /// host time.
    pub fn set_nvram_file(&self, path: impl Into<std::path::PathBuf>) -> std::io::Result<()> {
        let path = path.into();
        let image = {
            let mut st = self.sh.st.lock();
            if let Ok(img) = std::fs::read(&path) {
                if !st.rtc.load(&img) {
                    log::warn!("ioc3: NVRAM file {} has {} bytes, not 512; ignored", path.display(), img.len());
                }
            }
            st.rtc.file = Some(path.clone());
            st.rtc.dirty_since = None;
            st.rtc.image()
        };
        save_nvram(&path, &image)
    }

    /// Write the NVRAM file if the RAM changed at least `quiet` ago. The
    /// PROM writes its environment a byte at a time, so a timer waits for a
    /// pause rather than rewriting the file per byte; stopping the machine
    /// and dropping the chip flush with no wait. The file is written outside
    /// the chip's lock.
    pub fn flush_nvram_after(&self, quiet: Duration) {
        self.sh.flush_nvram(quiet);
    }

    /// [`Ioc3::flush_nvram_after`] with no wait.
    pub fn flush_nvram(&self) {
        self.sh.flush_nvram(Duration::ZERO);
    }

    /// Show the system board's front lightbar in the status bar through
    /// `heartbeat`.
    pub fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        let _ = self.sh.heartbeat.set(heartbeat);
    }

    /// Attach the Ethernet MAC. It is told about DMA and the PCI interrupt
    /// pin as soon as the bridge provides them.
    pub fn attach_eth(&self, eth: Arc<dyn Ioc3EthPort>) {
        *self.sh.eth.lock() = Some(eth);
        self.sh.connect_eth();
    }

    /// The line the board wires the SuperIO interrupt (serial, PS/2,
    /// parallel, interrupt output) to. [`PciBusDevice::connect_interrupt`]
    /// takes the PCI pin, which only the Ethernet block drives.
    pub fn connect_superio_interrupt(&self, line: IrqLine) {
        {
            let _order = self.sh.line_order.lock();
            let level = self.sh.superio_level.load(Ordering::SeqCst);
            line.set(level);
            *self.sh.superio_line.lock() = Some(line);
        }
        self.sh.sync_superio();
    }

    /// The PS/2 keyboard and mouse model, for the host side to push keys and
    /// pointer motion into.
    pub fn ps2(&self) -> Arc<Ps2Controller> {
        self.sh.ps2.clone()
    }

    /// Whether the guest has asked the clock to power the machine off.
    pub fn powered_off(&self) -> bool {
        self.sh.st.lock().rtc.dev.powered_off()
    }

    /// A PCI reset of the chip (not of the clock or the buffer RAM). The
    /// Ethernet block is a separate device here and keeps its state: the
    /// machine resets it on its own.
    pub fn reset(&self) {
        self.sh.with_st(false, |st, _| st.reset());
    }

    /// One synchronous pass of the work the receive threads and the timers
    /// do between guest accesses. Tests use it to settle the chip without
    /// waiting on a thread.
    #[cfg(test)]
    pub fn service(&self) {
        self.sh.service();
    }

    /// Name the chip's long-lived locks in the lock registry, which holds
    /// only a weak reference to them.
    pub fn register_locks(&self) {
        for (what, probe) in [
            ("st", (|s: &Shared| s.st.is_locked()) as fn(&Shared) -> bool),
            ("line_order", |s| s.line_order.is_locked()),
        ] {
            let sh = Arc::downgrade(&self.sh);
            crate::locks::register_lock(format!("ioc3::{what}"), move || sh.upgrade().is_some_and(|s| probe(&s)));
        }
    }

    /// The SuperIO interrupt status register, for tests and the monitor.
    pub fn int_status(&self) -> u32 {
        self.sh.st.lock().int_status
    }
}

impl Device for Ioc3 {
    fn step(&self, _cycles: u64) {}

    /// Start the serial receive threads and the NVRAM writer and arm the
    /// timers for any deadline the chip already holds.
    fn start(&self) {
        if self.sh.running.swap(true, Ordering::SeqCst) {
            return;
        }
        Device::start(&*self.sh.ps2);
        let mut threads = self.threads.lock();
        let serial = self.sh.st.lock().cfg.serial;
        for (p, fitted) in serial.into_iter().enumerate() {
            if fitted {
                let me = self.sh.me.clone();
                threads.push(
                    std::thread::Builder::new()
                        .name(format!("IOC3-RX-{}", ["A", "B"][p]))
                        .spawn(move || Shared::rx_loop(me, p))
                        .expect("spawn ioc3 receive thread"),
                );
            }
        }
        let me = self.sh.me.clone();
        threads.push(
            std::thread::Builder::new()
                .name("IOC3-NVRAM".into())
                .spawn(move || Shared::nvram_loop(me))
                .expect("spawn ioc3 nvram thread"),
        );
        drop(threads);
        self.sh.rearm();
    }

    /// Remove the timers, wait out a callback in flight, join the threads and
    /// flush the NVRAM file. Nothing touches memory or a line afterwards,
    /// until `start`.
    fn stop(&self) {
        if !self.sh.running.swap(false, Ordering::SeqCst) {
            return;
        }
        {
            let mut armed = self.sh.armed.lock();
            for slot in armed.iter_mut() {
                if let (Some((id, _)), Some(tm)) = (slot.take(), self.sh.timers.get()) {
                    tm.remove(id);
                }
            }
        }
        drop(self.sh.gate.lock());
        // The writer waits on the job slot's lock: take it so the wake-up
        // cannot fall between its check of `running` and its wait.
        drop(self.sh.nvram_job.lock());
        self.sh.nvram_ready.notify_all();
        let mut threads = self.threads.lock();
        for t in threads.iter() {
            t.thread().unpark();
        }
        for t in threads.drain(..) {
            let _ = t.join();
        }
        drop(threads);
        Device::stop(&*self.sh.ps2);
        self.flush_nvram();
    }

    fn is_running(&self) -> bool {
        self.sh.running.load(Ordering::SeqCst)
    }

    fn get_clock(&self) -> u64 {
        0
    }
}

impl Resettable for Ioc3 {
    /// The chip as at power-on. The clock and the buffer RAM keep their
    /// contents, as they are memory.
    fn power_on(&self) {
        self.reset();
    }
}

impl Drop for Ioc3 {
    fn drop(&mut self) {
        Device::stop(&*self);
        self.flush_nvram();
    }
}

impl Ioc3 {
    /// The header as the chip answers it, for the configuration space and
    /// the first dwords of the register block alike. The status register
    /// reads fast back-to-back capable beside the medium decode timing.
    fn header_dword(&self, index: usize) -> u32 {
        let v = self.sh.pci.read(index);
        if index == 1 { v | PCI_STATUS_FAST_BACK << 16 } else { v }
    }
}

impl PciBusDevice for Ioc3 {
    fn config(&self) -> &PciConfig {
        &self.sh.pci
    }

    /// The chip decodes five configuration address bits, so dwords 0x20 to
    /// 0x3f read as 0x00 to 0x1f: no subsystem dword at 0x2c (the latency
    /// word shows there) and no interrupt pin at 0x3d. Only the guest's
    /// view repeats; the bridge still routes INTA from the stored pin, and
    /// the subsystem id stays the key that picks the board's population.
    fn config_dword(&self, index: usize) -> u32 {
        self.header_dword(if (CONFIG_DECODED_DWORDS..2 * CONFIG_DECODED_DWORDS).contains(&index) {
            index - CONFIG_DECODED_DWORDS
        } else {
            index
        })
    }

    /// `off` is where the bridge put the access on the bus; IOC3 undoes the
    /// bridge's narrow-access placement (`off ^ (4 - width)`), so a guest's
    /// byte access to register N reaches register N.
    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
        if bar != 0 || off >= BAR_SIZE {
            return 0;
        }
        let a = off ^ (4 - width.min(4) as u32);
        if width == 4 && a <= reg::HEADER_LAST && a & 3 == 0 {
            return self.header_dword((a / 4) as usize);
        }
        if (reg::ENET_FIRST..=reg::ENET_LAST).contains(&a) {
            // Only 32-bit accesses reach the Ethernet block.
            if width != 4 {
                return 0;
            }
            return self.sh.eth().map_or(0, |e| e.read32(a));
        }
        let parity = self.sh.parity_for(a);
        self.sh.with_st(parity, |st, c| st.read(a, width, c))
    }

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        if bar != 0 || off >= BAR_SIZE {
            return;
        }
        let a = off ^ (4 - width.min(4) as u32);
        if (reg::ENET_FIRST..=reg::ENET_LAST).contains(&a) {
            if width == 4 && a & 3 == 0 {
                if let Some(e) = self.sh.eth() {
                    e.write32(a, val);
                }
            }
            return;
        }
        let parity = self.sh.parity_for(a);
        self.sh.with_st(parity, |st, c| st.write(a, val, width, c));
    }

    /// The PCI interrupt pin: the Ethernet block's level.
    fn connect_interrupt(&self, line: IrqLine) {
        {
            let _order = self.sh.line_order.lock();
            line.set(self.sh.pin_level.load(Ordering::SeqCst));
            *self.sh.pin_line.lock() = Some(line);
        }
        self.sh.sync_pin();
    }

    fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        *self.sh.raw_dma.lock() = Some(dma.clone());
        self.sh.st.lock().dma = Some(Arc::new(Ioc3Dma::new(dma)));
        self.sh.connect_eth();
    }
}

#[cfg(test)]
#[path = "ioc3_tests.rs"]
mod tests;
