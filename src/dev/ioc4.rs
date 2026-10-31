//! SGI IOC4 base I/O controller, revision 0x53, as the IO9 board fits it.
//!
//! One PCI function with a 1 MiB memory BAR. Behind it: four 16550-compatible
//! serial ports with a ring-DMA engine each, a local interrupt controller for
//! them, the registers that capture a failed ring transfer, the UART clock
//! select, the ATA channel's DMA readback word the PROM's diagnostic probes,
//! and the timekeeping SRAM on byte-bus select 0 when the board fits one
//! ([`Ds1744`]), and the PS/2 keyboard and mouse controller the IO9 always
//! carries ([`Ioc3Kbc`], the block the IOC3 has too). The ATA channel, the
//! GPIO pins and the other byte-bus selects are not modelled; an access to
//! them reads zero or is dropped, and is logged.
//!
//! Behaviour follows the public Linux and OpenBSD drivers; where the model
//! approximates, the comment says so.
//!
//! # One interrupt output
//!
//! Everything the chip reports ORs into the PCI interrupt pin: the serial
//! status word (eight bits a port) and the other word (ATA, DMA errors,
//! keyboard), each behind its own enable. UART pass-through, transmit-ring-
//! empty, receive-ring-full and receive-high-water follow their source's level and ignore a
//! write-one-to-clear; the rest latch until acknowledged.
//!
//! # Byte order
//!
//! IOC4 orders its own bytes. Its slot is self-ordered: the bridge carries the
//! processor's bytes in address order and leaves the address alone, so a
//! byte access to UART register `r` of port `n` arrives at `0x380 + 8n + r`.
//! A PCI function's BAR value is its bytes read little-endian, and IOC4 reads
//! them big-endian, so the chip takes a 32-bit value byte-reversed: the
//! processor's value is the register's value, as QEMU's big-endian IOC4
//! region reads it. Control registers and the
//! keyboard block take 32-bit accesses only; a UART window or the clock takes
//! bytes only. Anything else is refused (reads zero, writes dropped). Which widths the silicon accepts is not
//! known.
//!
//! # What is approximated here
//!
//! * The reset values, the BAR size and the PCI class are the emulator's
//!   choices.
//! * The serial ring engine is the one the IOC3 uses (`ioc_serial_dma.rs`) and
//!   carries its approximations: timers run on host time and receive-ring-full
//!   is a level that follows a held entry. After a failed ring transfer the port
//!   halts until a channel reset.
//! * Register 0x20 reads the DMA-idle bit unless a transfer is under way.
//!   Transfers complete inside the access that starts them, so software
//!   always finds the engine idle.
//! * The keyboard block's status bits (write-pending, line control, line
//!   levels and idle flags, mask `0xc33f`) read clear and drop writes, as the
//!   QEMU adapter does. Its port order, write-pending timing, and whether
//!   the interrupt is a pulse per character or a level are not established,
//!   and the model follows QEMU: a latch per arriving character, which an
//!   acknowledge racing an arrival can lose. The IOC3-only receive flags are
//!   masked out of the receive registers. The ports are always fitted, as the
//!   IO9's are in QEMU.
//! * The ATA readback word at 0x148 is a plain 32-bit latch with no ATA
//!   behind it. Only boards that ask for it have it.
//! * The PCI status register reads 0x0220 (66 MHz capable, medium DEVSEL) on
//!   boards that ask for it: a value taken from another revision of the chip,
//!   not measured on this one.
//!
//! Names follow QEMU's: `sgi-ioc4.c` has the same registers under the same
//! names, its `sgi_ioc4_sdma_*` callbacks are the chip's `SerialDmaHost`, and
//! the ring engine is `ioc_serial_dma.rs` (QEMU's `sgi-ioc-serial-dma.c`).
//!
//! Ported from hw/misc/sgi-ioc4.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::dev::ds1744::Ds1744;
use crate::dev::ioc3_kbc::{self, Ioc3Kbc, KbcLine, KbcOwner};
use crate::dev::ps2::Ps2Controller;
use crate::dev::ioc_serial_dma::{self, off as sdma_off, SerialDmaUart, SerialDmaHost, SerialDmaParams, SerialDma, SerialDmaSource};
use crate::dev::ns16550::Ns16550;
use crate::dev::serial_host::SerialBackend;
use crate::dev::deadline_timers::DeadlineTimers;
use crate::hptimer::{TimerManager, TimerReturn};
use crate::irq::IrqLine;
use crate::pci::{Bar, PciConfig, PciConfigPolicy, PciDma, PciFunction, PciFunctionState, PCI_INTA};
use crate::traits::{BusDevice, Device, Resettable};

pub const VENDOR_ID: u16 = 0x10a9;
pub const DEVICE_ID: u16 = 0x100a;
pub const REVISION: u8 = 0x53;
/// PCI class, subclass and interface: "other", an emulator default.
const PCI_CLASS: u32 = 0x00_ff_00;
pub const BAR_SIZE: u32 = 1 << 20;
/// Status of a 66 MHz capable function with medium DEVSEL timing.
const PCI_STATUS_COMPAT: u16 = 0x0220;
pub const PORTS: usize = 4;

/// Register offsets within the BAR.
pub mod reg {
    /// Captured address of the first failed ring transfer, low and high.
    pub const ERROR_LOW: u32 = 0x00;
    pub const ERROR_HIGH: u32 = 0x04;
    pub const SERIAL_PENDING: u32 = 0x08;
    pub const OTHER_PENDING: u32 = 0x0c;
    pub const SERIAL_ENABLE_SET: u32 = 0x10;
    pub const OTHER_ENABLE_SET: u32 = 0x14;
    pub const SERIAL_ENABLE_CLR: u32 = 0x18;
    pub const OTHER_ENABLE_CLR: u32 = 0x1c;
    pub const SERIAL_CONTROL: u32 = 0x20;
    pub const MISC_CONTROL: u32 = 0x24;
    /// A word of the ATA channel's DMA block the PROM writes and reads back.
    pub const ATA_READBACK: u32 = 0x148;
    /// Ring bases, a low and a high word for each pair of ports.
    pub const SDMA_BASE: u32 = 0x300;
    pub const SDMA_PORT: u32 = 0x310;
    pub const SDMA_PORT_STRIDE: u32 = super::sdma_off::REGS * 4;
    /// The first 16550 window; eight bytes a port.
    pub const UART: u32 = 0x380;
    pub const UART_STRIDE: u32 = 8;
    /// Byte-bus select 0: the timekeeping SRAM, whose last eight bytes are
    /// the clock (at 0x87ff8).
    pub const CLOCK: u32 = 0x80000;
    pub const CLOCK_SIZE: u32 = crate::dev::ds1744::SIZE as u32;
    /// The PS/2 controller's five registers.
    pub const KBC: u32 = 0x200;
}

/// Bits of the serial status word, eight to a port.
pub mod int {
    pub const PORT_STRIDE: u32 = 8;
    pub const PASSTHROUGH: u32 = 6;
    pub const fn serial(port: usize, src: u32) -> u32 {
        1 << (port as u32 * PORT_STRIDE + src)
    }
    pub const fn pass(port: usize) -> u32 {
        serial(port, PASSTHROUGH)
    }
    /// The other word: a serial DMA error, one bit a port from bit 2.
    pub const fn dma_error(port: usize) -> u32 {
        1 << (port as u32 + 2)
    }
}

/// Bits of the captured error address.
mod err {
    pub const VALID: u32 = 1;
    pub const MULTIPLE: u32 = 1 << 5;
    pub const REQUESTER_SHIFT: u32 = 1;
    /// The low six address bits are not captured.
    pub const ADDRESS_MASK: u64 = 0xffff_ffc0;
}

/// Other word bit 6: a keyboard or mouse character arrived.
const OTHER_KEYBOARD: u32 = 0x40;
/// Keyboard control bits that report state on the controller: write-pending
/// (0, 1), the last write's line-control bit (2, 3), the data and clock line
/// levels (4, 5, 8, 9) and the state machines' idle flags (14, 15). The model
/// does not simulate the lines, so they all read clear and a write to them is
/// dropped.
const KBC_STATUS: u32 = 0x0000_c33f;
/// The receive registers carry no IOC3-only flags: bits 24 to 28 (per-slot
/// errors, source and overflow) read clear.
const KBC_RX_MASK: u32 = 0xe0ff_ffff;

/// Serial control bits the model acts on: the threshold, drain, enable,
/// pause and reset. The rest are stored and read back.
const SSCR_MODELLED: u32 = 0x1ff | (1 << 27) | (1 << 28) | (1 << 29) | (1 << 31);
/// The receive consume index also holds the timer arm in bit 31.
const SRCIR_TIMER_ARM: u32 = 1 << 31;
/// The ring size select in a pair base's low word.
const RING_4K: u32 = 1;

/// Register 0x20: the DMA engines are idle.
const SERIAL_CONTROL_IDLE: u32 = 0x80;

/// IOC4's engine halts a port on a failed transfer and does not define the
/// packer's held count.
const IOC4_SDMA_PARAMS: SerialDmaParams = SerialDmaParams { held_counts: false, halt_on_dma_error: true };

/// The UART input clock the board feeds the chip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ioc4Clock {
    Mhz33,
    Mhz66,
}

impl Ioc4Clock {
    pub fn hz(self) -> u32 {
        match self {
            Ioc4Clock::Mhz33 => 33_333_333,
            Ioc4Clock::Mhz66 => 66_666_667,
        }
    }
}

/// What the board makes of the chip.
#[derive(Clone, Copy, Debug)]
pub struct Ioc4Config {
    pub clock: Ioc4Clock,
    /// Report PCI status 0x0220.
    pub pci_status_compat: bool,
    /// Latch the ATA DMA readback word.
    pub ata_readback: bool,
}

impl Ioc4Config {
    /// The chip with nothing a board asks for beyond its clock.
    pub fn bare(clock: Ioc4Clock) -> Self {
        Self { clock, pci_status_compat: false, ata_readback: false }
    }

    /// The Tezro's IO9: a 66.67 MHz clock, and the status and ATA readback
    /// words its PROM expects.
    pub fn tezro() -> Self {
        Self { clock: Ioc4Clock::Mhz66, pci_status_compat: true, ata_readback: true }
    }
}

/// A 32-bit register in the ring-engine block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SdmaReg {
    Base { pair: usize, high: bool },
    Port { port: usize, off: u32 },
}

/// Ring bases sit at 0x300 (low then high word, a pair each), then seven
/// registers a port from 0x310.
fn sdma_decode(a: u32) -> Option<SdmaReg> {
    if a & 3 != 0 {
        return None;
    }
    if (reg::SDMA_BASE..reg::SDMA_PORT).contains(&a) {
        let n = a - reg::SDMA_BASE;
        return Some(SdmaReg::Base { pair: (n / 8) as usize, high: n & 4 != 0 });
    }
    let end = reg::SDMA_PORT + PORTS as u32 * reg::SDMA_PORT_STRIDE;
    if (reg::SDMA_PORT..end).contains(&a) {
        let n = a - reg::SDMA_PORT;
        return Some(SdmaReg::Port { port: (n / reg::SDMA_PORT_STRIDE) as usize, off: n % reg::SDMA_PORT_STRIDE });
    }
    None
}

/// Where a byte access to the UART windows lands.
fn uart_decode(a: u32) -> Option<(usize, u32)> {
    let end = reg::UART + PORTS as u32 * reg::UART_STRIDE;
    (reg::UART..end).contains(&a).then(|| (((a - reg::UART) / reg::UART_STRIDE) as usize, (a - reg::UART) % reg::UART_STRIDE))
}

fn in_kbc(a: u32) -> bool {
    (reg::KBC..reg::KBC + ioc3_kbc::reg::MMIO_SIZE).contains(&a)
}

fn in_clock(a: u32) -> bool {
    (reg::CLOCK..reg::CLOCK + reg::CLOCK_SIZE).contains(&a)
}

struct St {
    cfg: Ioc4Config,
    serial_pending: u32,
    serial_enable: u32,
    other_pending: u32,
    other_enable: u32,
    /// Sources that follow a level, in the serial word's layout.
    level: u32,
    error_low: u32,
    error_high: u32,
    ata_readback: u32,
    uart: [Ns16550; PORTS],
    sdma: SerialDma,
    kbc: Ioc3Kbc,
    ps2: Arc<Ps2Controller>,
    dma: Option<Arc<dyn PciDma>>,
}

impl St {
    fn new(cfg: Ioc4Config, ps2: Arc<Ps2Controller>) -> St {
        St {
            cfg,
            serial_pending: 0,
            serial_enable: 0,
            other_pending: 0,
            other_enable: 0,
            level: 0,
            error_low: 0,
            error_high: 0,
            ata_readback: 0,
            uart: std::array::from_fn(|_| Ns16550::new(cfg.clock.hz() / 16)),
            sdma: SerialDma::new(PORTS, IOC4_SDMA_PARAMS),
            // Every IO9 brings the keyboard and mouse out.
            kbc: Ioc3Kbc::new(true),
            ps2,
            dma: None,
        }
    }

    /// A PCI reset: the interrupt state, the ports and the engines.
    fn reset(&mut self) {
        self.serial_pending = 0;
        self.serial_enable = 0;
        self.other_pending = 0;
        self.other_enable = 0;
        self.level = 0;
        self.error_low = 0;
        self.error_high = 0;
        self.ata_readback = 0;
        for u in &mut self.uart {
            u.reset();
        }
        self.sdma.reset();
        self.kbc.reset();
        self.ps2.power_on();
    }

    /// Move what the host's PS/2 model has sent into the holdings, and latch
    /// the keyboard source if a character arrived.
    fn kbc_fill(&mut self) {
        let ps2 = self.ps2.clone();
        if self.kbc.fill_all(&ps2) {
            self.other_pending |= OTHER_KEYBOARD;
        }
    }

    /// A 32-bit read of the keyboard block at `off` in its window, as the
    /// adapter presents it: no write-pending or other status, and none of
    /// the IOC3-only receive flags.
    fn kbc_read(&mut self, off: u32) -> u32 {
        let ps2 = self.ps2.clone();
        let (raw, arrived) = self.kbc.read(off, &ps2);
        if arrived {
            self.other_pending |= OTHER_KEYBOARD;
        }
        if off == ioc3_kbc::reg::CONTROL { raw & !KBC_STATUS } else { raw & KBC_RX_MASK }
    }

    fn kbc_write(&mut self, off: u32, val: u32) {
        let mut val = val;
        if off == ioc3_kbc::reg::CONTROL {
            // The status bits read clear; a write that differs is a guest
            // error and is dropped.
            if val & KBC_STATUS != 0 {
                crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 write to read-only KBC status bits {:#x} ignored", val & KBC_STATUS);
            }
            val &= !KBC_STATUS;
        }
        let ps2 = self.ps2.clone();
        if self.kbc.write(off, val, &ps2) {
            self.other_pending |= OTHER_KEYBOARD;
        }
    }

    /// The PCI interrupt pin: any enabled source.
    fn line_level(&self) -> bool {
        (self.serial_pending | self.level) & self.serial_enable != 0 || self.other_pending & self.other_enable != 0
    }

    /// The host-time deadlines the engines hold, for the timers.
    fn deadlines(&self) -> Deadlines {
        let mut d = [None; KINDS];
        for p in 0..PORTS {
            d[K_RX + p] = self.sdma.rx_deadline(p);
            d[K_TX_DELAY + p] = self.sdma.tx_delay_deadline(p);
            d[K_LOOP + p] = self.sdma.loop_deadline(p);
        }
        d
    }

    fn sdma_engine_split(&mut self) -> (&mut SerialDma, Ioc4SerialDmaHost<'_>) {
        let host = Ioc4SerialDmaHost {
            uart: &mut self.uart,
            serial_pending: &mut self.serial_pending,
            level: &mut self.level,
            other_pending: &mut self.other_pending,
            error_low: &mut self.error_low,
            error_high: &mut self.error_high,
            dma: self.dma.as_deref(),
        };
        (&mut self.sdma, host)
    }

    fn sdma_run(&mut self, p: usize) {
        let (sdma, mut host) = self.sdma_engine_split();
        sdma.run(p, &mut host);
    }

    fn sdma_catch_up(&mut self, now: Instant) {
        let (sdma, mut host) = self.sdma_engine_split();
        sdma.catch_up(now, &mut host);
    }

    // ----- the 16550s -----

    /// Keep the pass-through source on the port's interrupt level, and let
    /// the engine move: a ring driver receives with the 16550's own
    /// interrupts disabled.
    fn uart_changed(&mut self, p: usize) {
        let level = self.uart[p].irq();
        set_level_bit(&mut self.level, int::pass(p), level);
        self.sdma_run(p);
    }

    /// Move host input into the port and let the receive timeout run out.
    /// Returns whether anything changed.
    fn uart_pump(&mut self, p: usize) -> bool {
        let u = &mut self.uart[p];
        let moved = u.pump();
        u.check_timeout(Instant::now()) || moved
    }

    fn uart_read(&mut self, p: usize, r: u32) -> u8 {
        let v = self.uart[p].read(r);
        self.uart_changed(p);
        v
    }

    fn uart_write(&mut self, p: usize, r: u32, v: u8) {
        if r == 0 && !self.uart[p].dlab() && self.sdma.control(p) & ioc_serial_dma::ctl::ENABLE != 0 {
            crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 PIO transmit on port {p} while DMA is enabled");
        }
        self.uart[p].write(r, v);
        self.uart_changed(p);
    }

    // ----- mode-switch diagnostics -----

    /// Report the DMA mode switches and the writes the model takes but does
    /// not act on. They are reported, not refused: the UART orders PIO and
    /// ring DMA itself, and the write goes ahead.
    fn check_port_write(&self, p: usize, off: u32, v: u32) {
        let log = |what: std::fmt::Arguments| crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 {what}");
        match off {
            sdma_off::CONTROL => {
                if v & !SSCR_MODELLED != 0 {
                    log(format_args!("serial control bits {:#010x} on port {p} are stored but not modelled", v & !SSCR_MODELLED));
                }
                self.control_check(p, v);
            }
            sdma_off::TX_PRODUCE | sdma_off::RX_CONSUME => {
                let size = if self.sdma.base_read(p / 2, false) & RING_4K != 0 { 0x1000 } else { 0x400 };
                let keep = (size - 8) | if off == sdma_off::RX_CONSUME { SRCIR_TIMER_ARM } else { 0 };
                if v & !keep != 0 {
                    let what = if off == sdma_off::TX_PRODUCE { "TX producer" } else { "RX consumer" };
                    log(format_args!("{what} {v:#010x} on port {p} has bits outside the pointer field"));
                }
            }
            sdma_off::RX_TIMER if v & !0xfff != 0 => {
                log(format_args!("unsupported RX timer bits {:#010x} on port {p}", v & !0xfff));
            }
            _ => {}
        }
    }

    /// Enabling DMA over a base misaligned for its rings, or over PIO data in
    /// the UART's FIFOs, and disabling it with ring work pending.
    fn control_check(&self, p: usize, v: u32) {
        use ioc_serial_dma::ctl;
        let old_on = self.sdma.control(p) & ctl::ENABLE != 0;
        let want_on = v & ctl::ENABLE != 0;
        if v & ctl::RESET != 0 || old_on == want_on {
            return;
        }
        let log = |what: std::fmt::Arguments| crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 {what}");
        let pair = p / 2;
        if want_on {
            let base = u64::from(self.sdma.base_read(pair, true)) << 32 | u64::from(self.sdma.base_read(pair, false) & !RING_4K);
            let size: u64 = if self.sdma.base_read(pair, false) & RING_4K != 0 { 0x1000 } else { 0x400 };
            if base & (size * 4 - 1) != 0 {
                log(format_args!("DMA enabled on port {p} with pair base {base:#x} misaligned for its rings"));
            }
            if self.uart[p].rx_pending() != 0 {
                log(format_args!("DMA enabled on port {p} with PIO data in the UART's FIFOs"));
            }
        } else if !self.sdma.port_quiescent(p) {
            log(format_args!("DMA disabled on port {p} with ring work pending"));
        }
    }

    // ----- control registers -----

    fn read32(&mut self, a: u32) -> u32 {
        self.sdma_catch_up(Instant::now());
        if let Some(r) = sdma_decode(a) {
            let (sdma, mut host) = self.sdma_engine_split();
            return match r {
                SdmaReg::Base { pair, high } => sdma.base_read(pair, high),
                SdmaReg::Port { port, off } => sdma.port_read(port, off, &mut host),
            };
        }
        match a {
            reg::ERROR_LOW => self.error_low,
            reg::ERROR_HIGH => self.error_high,
            reg::SERIAL_PENDING => self.serial_pending | self.level,
            reg::OTHER_PENDING => self.other_pending,
            reg::SERIAL_ENABLE_SET => self.serial_enable,
            reg::OTHER_ENABLE_SET => self.other_enable,
            reg::SERIAL_CONTROL => if self.sdma.idle() { SERIAL_CONTROL_IDLE } else { 0 },
            reg::MISC_CONTROL => u32::from(self.cfg.clock == Ioc4Clock::Mhz66),
            reg::ATA_READBACK if self.cfg.ata_readback => self.ata_readback,
            _ => {
                crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 read of unsupported register {a:#x}");
                0
            }
        }
    }

    fn write32(&mut self, a: u32, v: u32) {
        self.sdma_catch_up(Instant::now());
        if let Some(r) = sdma_decode(a) {
            match r {
                SdmaReg::Base { pair, high } => {
                    if !self.sdma.pair_quiescent(pair) {
                        crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 DMA base of pair {pair} changed while it is active");
                    }
                    self.sdma.base_write(pair, high, v);
                }
                SdmaReg::Port { port, off } => {
                    self.check_port_write(port, off, v);
                    let (sdma, mut host) = self.sdma_engine_split();
                    sdma.port_write(port, off, v, &mut host);
                }
            }
            return;
        }
        match a {
            // Acknowledge: write one to clear. Level sources ignore it.
            reg::SERIAL_PENDING => self.serial_pending &= !v,
            reg::OTHER_PENDING => self.other_pending &= !v,
            reg::SERIAL_ENABLE_SET => self.serial_enable |= v,
            reg::OTHER_ENABLE_SET => self.other_enable |= v,
            reg::SERIAL_ENABLE_CLR => self.serial_enable &= !v,
            reg::OTHER_ENABLE_CLR => self.other_enable &= !v,
            reg::ATA_READBACK if self.cfg.ata_readback => self.ata_readback = v,
            reg::ERROR_LOW | reg::ERROR_HIGH | reg::SERIAL_CONTROL | reg::MISC_CONTROL => {
                crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 write {v:#x} to read-only register {a:#x}");
            }
            _ => crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 write {v:#x} to unsupported register {a:#x}"),
        }
    }
}

fn set_level_bit(word: &mut u32, mask: u32, level: bool) {
    if level {
        *word |= mask;
    } else {
        *word &= !mask;
    }
}

/// IOC4 as the ring engine sees it: four 16550s, the serial status word,
/// the error capture registers and the bus master.
struct Ioc4SerialDmaHost<'a> {
    uart: &'a mut [Ns16550; PORTS],
    serial_pending: &'a mut u32,
    level: &'a mut u32,
    other_pending: &'a mut u32,
    error_low: &'a mut u32,
    error_high: &'a mut u32,
    dma: Option<&'a dyn PciDma>,
}

impl SerialDmaHost for Ioc4SerialDmaHost<'_> {
    fn uart(&mut self, port: usize) -> &mut dyn SerialDmaUart {
        &mut self.uart[port]
    }

    fn raise(&mut self, port: usize, src: SerialDmaSource) {
        *self.serial_pending |= int::serial(port, src as u32);
    }

    fn set_level(&mut self, port: usize, src: SerialDmaSource, level: bool) {
        set_level_bit(self.level, int::serial(port, src as u32), level);
    }

    /// The engines always reach their ports.
    fn reachable(&self, _port: usize) -> bool {
        true
    }

    fn uart_line_changed(&mut self, port: usize) {
        let level = self.uart[port].irq();
        set_level_bit(self.level, int::pass(port), level);
    }

    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.dma.is_some_and(|d| d.dma_read(addr, buf).is_ok())
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> bool {
        self.dma.is_some_and(|d| d.dma_write(addr, data).is_ok())
    }

    /// The first failure captures its address and its requester: twice the
    /// port for a transmit fetch, one more for a receive post. Later ones
    /// only set the multiple-error bit. The port's source is in the other
    /// word.
    fn dma_error(&mut self, port: usize, rx: bool, addr: u64) {
        if *self.error_low & err::VALID == 0 {
            let requester = (2 * port + usize::from(rx)) as u32;
            *self.error_low = (addr & err::ADDRESS_MASK) as u32 | requester << err::REQUESTER_SHIFT | err::VALID;
            *self.error_high = (addr >> 32) as u32;
        } else {
            *self.error_low |= err::MULTIPLE;
        }
        *self.other_pending |= int::dma_error(port);
        crate::dlog!(
            crate::devlog::LogModule::Ioc,
            "IOC4 serial DMA {} failed for port {port} at {addr:#x}",
            if rx { "write" } else { "read" }
        );
    }
}

// ---------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------

/// How long a serial receive thread sleeps when its backend has nothing:
/// `SerialBackend` has no blocking read to wait on.
const RX_IDLE: Duration = Duration::from_millis(2);

/// The deadlines the chip keeps on host time, one timer slot each.
const K_RX: usize = 0;
const K_TX_DELAY: usize = PORTS;
const K_LOOP: usize = 2 * PORTS;
const KINDS: usize = 3 * PORTS;
type Deadlines = [Option<Instant>; KINDS];

/// Lock order: `gate`, then `line_order`, then `st`; `armed` and `line` are
/// leaves. The engines move their rings with `st` held: the DMA goes down
/// through the bridge, which never calls back into the chip.
struct Shared {
    pci: PciFunctionState,
    st: Mutex<St>,
    /// PCI interrupt pin A.
    line: Mutex<Option<IrqLine>>,
    /// What the line was last driven to.
    line_level: AtomicBool,
    /// Serialises driving the line; see [`Shared::sync_line`].
    line_order: Mutex<()>,
    me: Weak<Shared>,
    timers: DeadlineTimers<KINDS>,
    /// A PS/2 character arrived while an access held the chip.
    kbc_dirty: AtomicBool,
    /// Between `start` and `stop`.
    running: AtomicBool,
    /// Held by a timer callback while it runs, so `stop` can wait one out.
    gate: Mutex<()>,
}

impl KbcOwner for Shared {
    /// A device raised its line. If an access holds the chip (possibly this
    /// very thread, from inside a transmit), that access picks the character
    /// up when it finishes: the flag is set before the lock is tried, so
    /// either this thread gets the lock or the holder sees the flag.
    fn kbc_poke(&self) {
        self.kbc_dirty.store(true, Ordering::SeqCst);
        if let Some(mut st) = self.st.try_lock() {
            if self.kbc_dirty.swap(false, Ordering::SeqCst) {
                st.kbc_fill();
            }
            drop(st);
            self.sync_line();
        }
    }
}

impl Shared {
    /// Put the interrupt line at the level the chip's state gives, then pick
    /// up any PS/2 character that arrived while the chip was held (a poke
    /// sets the flag before it tries the lock, so whoever holds the lock sees
    /// it here, after releasing).
    fn sync_line(&self) {
        loop {
            self.drive_line();
            if !self.kbc_dirty.swap(false, Ordering::SeqCst) {
                break;
            }
            self.st.lock().kbc_fill();
        }
    }

    /// The level is read after taking `line_order`, so whichever thread
    /// drives last has seen the latest state. `st` is never held when this
    /// is called.
    fn drive_line(&self) {
        let _order = self.line_order.lock();
        let level = self.st.lock().line_level();
        if self.line_level.swap(level, Ordering::SeqCst) != level {
            if let Some(l) = self.line.lock().as_ref() {
                l.set(level);
            }
        }
    }

    /// Run `f` with the chip locked, then arm the timers for the deadlines it
    /// left and drive the line from its state.
    fn with_st<R>(&self, f: impl FnOnce(&mut St) -> R) -> R {
        let (r, want) = {
            let mut st = self.st.lock();
            let r = f(&mut st);
            (r, st.deadlines())
        };
        self.arm(want);
        self.sync_line();
        r
    }

    /// Arm a timer for each deadline the chip holds.
    fn arm(&self, want: Deadlines) {
        self.timers.arm(&want, &self.running, &self.me, |_| Duration::ZERO, Shared::fire);
    }

    /// A deadline passed with no guest access to notice it. Runs on the timer
    /// thread: short, no waiting beyond the chip's own locks.
    fn fire(me: &Weak<Shared>, kind: usize, due: Instant) -> TimerReturn {
        let Some(sh) = me.upgrade() else { return TimerReturn::Delete };
        let _gate = sh.gate.lock();
        if !sh.running.load(Ordering::SeqCst) {
            return TimerReturn::Delete;
        }
        sh.timers.fired(kind, due);
        sh.with_st(|st| st.sdma_catch_up(Instant::now()));
        TimerReturn::Delete
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
                sh.with_st(|st| {
                    if st.uart_pump(port) {
                        st.uart_changed(port);
                    }
                });
            }
            std::thread::park_timeout(RX_IDLE);
        }
    }

    /// One synchronous pass of what the receive threads and timers do.
    #[cfg(test)]
    fn service(&self) {
        self.with_st(|st| {
            for p in 0..PORTS {
                st.uart_pump(p);
                st.uart_changed(p);
            }
            st.sdma_catch_up(Instant::now());
            for p in 0..PORTS {
                st.sdma_run(p);
            }
        });
    }
}

/// The IOC4 PCI function.
pub struct Ioc4 {
    sh: Arc<Shared>,
    /// The DS1744 on byte-bus select 0, when the board fits one.
    clock: OnceLock<Arc<Ds1744>>,
    /// The serial receive threads, joined in `stop`.
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Ioc4 {
    pub fn new(cfg: Ioc4Config) -> Ioc4 {
        let sh = Arc::new_cyclic(|weak: &Weak<Shared>| {
            let line = IrqLine::new(Arc::new(KbcLine(weak.clone())), 0);
            let ps2 = Arc::new(Ps2Controller::new(Some(line)));
            // Host key and mouse events are accepted from the start (this
            // only sets a flag); `stop` and `start` follow the chip's.
            Device::start(&*ps2);
            let bars = [Bar::Mem(BAR_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused];
            let policy = PciConfigPolicy {
                status: if cfg.pci_status_compat { PCI_STATUS_COMPAT } else { 0 },
                status_w1c_mask: 0,
                command_mask: 0xffff,
                ..Default::default()
            };
            Shared {
                pci: PciFunctionState::new(PciConfig::new_with_policy(VENDOR_ID, DEVICE_ID, PCI_CLASS, REVISION, bars, PCI_INTA, policy)),
                st: Mutex::new(St::new(cfg, ps2)),
                line: Mutex::new(None),
                line_level: AtomicBool::new(false),
                line_order: Mutex::new(()),
                kbc_dirty: AtomicBool::new(false),
                me: weak.clone(),
                timers: DeadlineTimers::new(),
                running: AtomicBool::new(false),
                gate: Mutex::new(()),
            }
        });
        Ioc4 { sh, clock: OnceLock::new(), threads: Mutex::new(Vec::new()) }
    }

    /// Fit the timekeeping SRAM on byte-bus select 0. Its 32 KiB fill the
    /// BAR from 0x80000 and its calendar sits in the last eight bytes. The
    /// chip's reset does not touch it: it is memory with a battery.
    pub fn attach_clock(&self, clock: Arc<Ds1744>) {
        assert!(self.clock.set(clock).is_ok(), "IOC4 clock attached more than once");
    }

    /// The timer manager for device deadlines. Set it before start.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        self.sh.timers.set_manager(tm);
    }

    /// Wire the host end of serial port `port` (0 is the console): a TCP
    /// listener, a log tee, or nothing.
    pub fn set_serial_backend(&self, port: usize, backend: Arc<dyn SerialBackend>) {
        self.sh.st.lock().uart[port].set_backend(backend);
    }

    /// The PS/2 keyboard and mouse model, for the host side to push keys and
    /// pointer motion into.
    pub fn ps2(&self) -> Arc<Ps2Controller> {
        self.sh.st.lock().ps2.clone()
    }

    /// A PCI reset of the chip.
    pub fn reset(&self) {
        self.sh.pci.config().reset();
        self.sh.with_st(St::reset);
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
            crate::locks::register_lock(format!("ioc4::{what}"), move || sh.upgrade().is_some_and(|s| probe(&s)));
        }
    }

    /// Say that an access the chip does not take was refused.
    fn refuse(what: &str, width: usize, off: u32) {
        crate::dlog!(crate::devlog::LogModule::Ioc, "IOC4 {what} of width {width} at {off:#x} refused");
    }
}

impl Device for Ioc4 {
    fn step(&self, _cycles: u64) {}

    /// Start the serial receive threads and arm the timers for any deadline
    /// the chip already holds.
    fn start(&self) {
        if self.sh.running.swap(true, Ordering::SeqCst) {
            return;
        }
        Device::start(&*self.sh.st.lock().ps2);
        let mut threads = self.threads.lock();
        for p in 0..PORTS {
            let me = self.sh.me.clone();
            threads.push(
                std::thread::Builder::new()
                    .name(format!("IOC4-RX-{p}"))
                    .spawn(move || Shared::rx_loop(me, p))
                    .expect("spawn ioc4 receive thread"),
            );
        }
        drop(threads);
        let want = self.sh.st.lock().deadlines();
        self.sh.arm(want);
    }

    /// Remove the timers, wait out a callback in flight and join the threads.
    /// Nothing touches memory or the line afterwards, until `start`.
    fn stop(&self) {
        if !self.sh.running.swap(false, Ordering::SeqCst) {
            return;
        }
        self.sh.timers.cancel_all();
        drop(self.sh.gate.lock());
        let mut threads = self.threads.lock();
        for t in threads.iter() {
            t.thread().unpark();
        }
        for t in threads.drain(..) {
            let _ = t.join();
        }
        drop(threads);
        Device::stop(&*self.sh.st.lock().ps2);
    }

    fn is_running(&self) -> bool {
        self.sh.running.load(Ordering::SeqCst)
    }

    fn get_clock(&self) -> u64 {
        0
    }
}

impl Resettable for Ioc4 {
    fn power_on(&self) {
        self.reset();
    }
}

impl Drop for Ioc4 {
    fn drop(&mut self) {
        Device::stop(&*self);
    }
}

impl PciFunction for Ioc4 {
    fn state(&self) -> &PciFunctionState {
        &self.sh.pci
    }

    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32 {
        if bar != 0 || off >= BAR_SIZE {
            return 0;
        }
        if let Some((p, r)) = uart_decode(off) {
            if width != 1 {
                Ioc4::refuse("UART read", width, off);
                return 0;
            }
            return u32::from(self.sh.with_st(|st| st.uart_read(p, r)));
        }
        if let Some(clock) = self.clock.get().filter(|_| in_clock(off)) {
            if width != 1 {
                Ioc4::refuse("clock read", width, off);
                return 0;
            }
            return u32::from(clock.read8(u64::from(off - reg::CLOCK)).data);
        }
        if in_kbc(off) {
            if width != 4 || off & 3 != 0 {
                Ioc4::refuse("KBC read", width, off);
                return 0;
            }
            return self.sh.with_st(|st| st.kbc_read(off - reg::KBC)).swap_bytes();
        }
        if width != 4 || off & 3 != 0 {
            Ioc4::refuse("read", width, off);
            return 0;
        }
        self.sh.with_st(|st| st.read32(off)).swap_bytes()
    }

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        if bar != 0 || off >= BAR_SIZE {
            return;
        }
        if let Some((p, r)) = uart_decode(off) {
            if width != 1 {
                Ioc4::refuse("UART write", width, off);
                return;
            }
            self.sh.with_st(|st| st.uart_write(p, r, val as u8));
            return;
        }
        if let Some(clock) = self.clock.get().filter(|_| in_clock(off)) {
            if width != 1 {
                Ioc4::refuse("clock write", width, off);
                return;
            }
            clock.write8(u64::from(off - reg::CLOCK), val as u8);
            return;
        }
        if in_kbc(off) {
            if width != 4 || off & 3 != 0 {
                Ioc4::refuse("KBC write", width, off);
                return;
            }
            self.sh.with_st(|st| st.kbc_write(off - reg::KBC, val.swap_bytes()));
            return;
        }
        if width != 4 || off & 3 != 0 {
            Ioc4::refuse("write", width, off);
            return;
        }
        self.sh.with_st(|st| st.write32(off, val.swap_bytes()));
    }

    /// The chip's DMA, and the PCI interrupt pin: any enabled source.
    fn attached(&self) {
        if let Some(dma) = self.sh.pci.dma() {
            self.sh.st.lock().dma = Some(dma.clone());
        }
        if let Some(line) = self.sh.pci.irq() {
            {
                let _order = self.sh.line_order.lock();
                line.set(self.sh.line_level.load(Ordering::SeqCst));
                *self.sh.line.lock() = Some(line.clone());
            }
            self.sh.sync_line();
        }
    }
}

#[cfg(test)]
#[path = "ioc4_tests.rs"]
mod tests;
