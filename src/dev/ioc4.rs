//! SGI IOC4 base I/O controller, revision 0x53, as the IO9 board fits it.
//!
//! One PCI function with a 1 MiB memory BAR. Behind it: four 16550-compatible
//! serial ports with a ring-DMA engine each, a local interrupt controller for
//! them, the registers that capture a failed ring transfer, the UART clock
//! select, the ATA channel's DMA readback word the PROM's diagnostic probes,
//! and the timekeeping SRAM on byte-bus select 0 when the board fits one
//! ([`Ds1744`]). The ATA channel, the keyboard and mouse controller, the GPIO
//! pins and the other byte-bus selects are not modelled; an access to them
//! reads zero or is dropped, and is logged.
//!
//! Behaviour follows the public Linux and OpenBSD drivers; where the model
//! approximates, the comment says so.
//!
//! # One interrupt output
//!
//! Everything the chip reports ORs into the PCI interrupt pin: the serial
//! status word (eight bits a port) and the other word (ATA, DMA errors,
//! keyboard), each behind its own enable. UART pass-through, transmit-ring-
//! empty and receive-high-water follow their source's level and ignore a
//! write-one-to-clear; the rest latch until acknowledged.
//!
//! # Byte order
//!
//! IOC4 orders its own bytes. The bridge hands it the register's address and
//! value as software wrote them, so a byte access to UART register `r` of
//! port `n` arrives at `0x380 + 8n + r`, and a 32-bit access is the register
//! value, with no lane swap on either side. Control registers take 32-bit
//! accesses only; a UART window or the clock takes bytes only. Anything else is refused
//! (reads zero, writes dropped). Which widths the silicon accepts is not
//! known.
//!
//! # What is approximated here
//!
//! * The reset values, the BAR size and the PCI class are the emulator's
//!   choices.
//! * The serial ring engine is the one the IOC3 uses (`ioc_serial_dma.rs`) and
//!   carries its approximations: a loopback character arrives inside the
//!   write that sent it, timers run on host time, and receive-ring-full
//!   latches on each refused post. After a failed ring transfer the port
//!   halts until a channel reset.
//! * Register 0x20 reads the DMA-idle bit set; transfers complete inside the
//!   access that starts them, so the engine is never busy when software looks.
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
use crate::dev::ioc_serial_dma::{off as sdma_off, SerialDmaUart, SerialDmaHost, SerialDmaParams, SerialDma, SerialDmaSource};
use crate::dev::ns16550::Ns16550;
use crate::dev::serial_host::SerialBackend;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::irq::IrqLine;
use crate::pci::{Bar, PciBusDevice, PciConfig, PciConfigPolicy, PciDma, PCI_INTA};
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
    dma: Option<Arc<dyn PciDma>>,
}

impl St {
    fn new(cfg: Ioc4Config) -> St {
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

    fn uart_read(&mut self, p: usize, r: u32) -> u8 {
        let v = self.uart[p].read(r);
        self.uart_changed(p);
        v
    }

    fn uart_write(&mut self, p: usize, r: u32, v: u8) {
        self.uart[p].write(r, v);
        self.uart_changed(p);
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
            reg::SERIAL_CONTROL => SERIAL_CONTROL_IDLE,
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
                SdmaReg::Base { pair, high } => self.sdma.base_write(pair, high, v),
                SdmaReg::Port { port, off } => {
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
const KINDS: usize = 2 * PORTS;
type Deadlines = [Option<Instant>; KINDS];

/// Lock order: `gate`, then `line_order`, then `st`; `armed` and `line` are
/// leaves. The engines move their rings with `st` held: the DMA goes down
/// through the bridge, which never calls back into the chip.
struct Shared {
    pci: PciConfig,
    st: Mutex<St>,
    /// PCI interrupt pin A.
    line: Mutex<Option<IrqLine>>,
    /// What the line was last driven to.
    line_level: AtomicBool,
    /// Serialises driving the line; see [`Shared::sync_line`].
    line_order: Mutex<()>,
    me: Weak<Shared>,
    timers: OnceLock<Arc<TimerManager>>,
    /// Between `start` and `stop`.
    running: AtomicBool,
    /// Held by a timer callback while it runs, so `stop` can wait one out.
    gate: Mutex<()>,
    /// The timer pending for each deadline kind, and the deadline it serves.
    armed: Mutex<[Option<(TimerId, Instant)>; KINDS]>,
}

impl Shared {
    /// Put the interrupt line at the level the chip's state gives. The level
    /// is read after taking `line_order`, so whichever thread drives last has
    /// seen the latest state. `st` is never held when this is called.
    fn sync_line(&self) {
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

    /// Make the pending timers match `want`: one one-shot per deadline. A
    /// timer whose deadline moved is replaced; with none set or the chip not
    /// running, deadlines are only evaluated lazily, by the next access.
    fn arm(&self, want: Deadlines) {
        let Some(tm) = self.timers.get() else { return };
        let mut armed = self.armed.lock();
        // Checked under the lock `stop` clears the slots with.
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
            let delay = due.saturating_duration_since(Instant::now());
            let id = tm.add_one_shot(delay, self.me.clone(), move |me| Shared::fire(me, k, due));
            armed[k] = Some((id, due));
        }
    }

    /// A deadline passed with no guest access to notice it. Runs on the timer
    /// thread: short, no waiting beyond the chip's own locks.
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
                    let moved = st.uart[port].pump();
                    if st.uart[port].check_timeout(Instant::now()) || moved {
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
                st.uart[p].pump();
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
            let bars = [Bar::Mem(BAR_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused];
            let policy = PciConfigPolicy {
                status: if cfg.pci_status_compat { PCI_STATUS_COMPAT } else { 0 },
                status_w1c_mask: 0,
                command_mask: 0xffff,
                ..Default::default()
            };
            Shared {
                pci: PciConfig::new_with_policy(VENDOR_ID, DEVICE_ID, PCI_CLASS, REVISION, bars, PCI_INTA, policy),
                st: Mutex::new(St::new(cfg)),
                line: Mutex::new(None),
                line_level: AtomicBool::new(false),
                line_order: Mutex::new(()),
                me: weak.clone(),
                timers: OnceLock::new(),
                running: AtomicBool::new(false),
                gate: Mutex::new(()),
                armed: Mutex::new([None; KINDS]),
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
        let _ = self.sh.timers.set(tm);
    }

    /// Wire the host end of serial port `port` (0 is the console): a TCP
    /// listener, a log tee, or nothing.
    pub fn set_serial_backend(&self, port: usize, backend: Arc<dyn SerialBackend>) {
        self.sh.st.lock().uart[port].set_backend(backend);
    }

    /// A PCI reset of the chip.
    pub fn reset(&self) {
        self.sh.pci.reset();
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
        {
            let mut armed = self.sh.armed.lock();
            for slot in armed.iter_mut() {
                if let (Some((id, _)), Some(tm)) = (slot.take(), self.sh.timers.get()) {
                    tm.remove(id);
                }
            }
        }
        drop(self.sh.gate.lock());
        let mut threads = self.threads.lock();
        for t in threads.iter() {
            t.thread().unpark();
        }
        for t in threads.drain(..) {
            let _ = t.join();
        }
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

impl PciBusDevice for Ioc4 {
    fn config(&self) -> &PciConfig {
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
        if width != 4 || off & 3 != 0 {
            Ioc4::refuse("read", width, off);
            return 0;
        }
        self.sh.with_st(|st| st.read32(off))
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
        if width != 4 || off & 3 != 0 {
            Ioc4::refuse("write", width, off);
            return;
        }
        self.sh.with_st(|st| st.write32(off, val));
    }

    /// The PCI interrupt pin: any enabled source.
    fn connect_interrupt(&self, line: IrqLine) {
        {
            let _order = self.sh.line_order.lock();
            line.set(self.sh.line_level.load(Ordering::SeqCst));
            *self.sh.line.lock() = Some(line);
        }
        self.sh.sync_line();
    }

    fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        self.sh.st.lock().dma = Some(dma);
    }
}

#[cfg(test)]
#[path = "ioc4_tests.rs"]
mod tests;
