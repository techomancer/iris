//! SGI IOC3 base I/O controller, as a parent and its children.
//!
//! One PCI function with one 1 MiB memory BAR, shaped as Linux's
//! `drivers/mfd/ioc3.c` drives it: this module owns the function, the
//! register window, the status and enable words of the interrupt output
//! and DMA, and dispatches each access by offset to the child that owns the
//! sub-range. Behind it:
//!
//! * `serial`: a ring-DMA engine for each of two serial ports (IOC3 logic);
//! * `superio`: the external Super I/O in the 0x20000 window, the two 16550
//!   UARTs and the parallel port's ECP registers;
//! * `kbd`: the PS/2 keyboard and mouse controller;
//! * `parallel`: the parallel port's DMA engine;
//! * `w1`: the MicroLAN (one-wire) master;
//! * `gpio`: the general-purpose I/O pins and what the board wires to them;
//! * `dma`: the chip's view of bus-master memory;
//! * the clock on the byte bus, the interrupt-output pulse
//!   generator and 256 KiB of Ethernet buffer RAM, in this module; and
//! * the Ethernet MAC (`eth`), whose registers sit at 0xf0.
//!
//! Which children a board fits is a row of the table in `board`. Behaviour follows the
//! public Linux and OpenBSD drivers and measured guest register traces; where
//! the model approximates, the comment says so.
//!
//! # Two interrupt outputs
//!
//! The chip has two. The Ethernet block drives the PCI interrupt pin (the
//! line in its [`PciFunctionState`]); everything else (serial, PS/2,
//! parallel, the interrupt-output pulse) is gated by the status and enable
//! pair at 0x1c/0x20 and drives a second line the board wires to a Bridge
//! input ([`Ioc3::connect_superio_interrupt`]). The pair and the level last
//! driven sit under one lock: a child changes its bit and the line is driven
//! inside it, and the Bridge is called only when the output changes.
//!
//! # Byte order
//!
//! IOC3 owns the conversion between its register and DMA formats and PCI
//! bytes. Registers: a 32-bit access is the register value; a narrow access
//! to internal offset `a` arrives from the bridge at `a ^ (4 - width)`, which
//! [`PciFunction::bar_read`] undoes. DMA: an internal byte at address `a`
//! sits at bus address `a ^ 3`, so a 32-bit value is big-endian on the bus
//! with the bridge's swap clear ([`dma::Ioc3Dma`]). Neither is a measured
//! property of the silicon; the chip's real placement is undocumented.
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
//!   extended RAM, persists in a 512-byte file, and once the guest sets the
//!   clock, the clock persists beside it (see `battery_clock`).
//!
//! Ported from hw/misc/sgi-ioc3.c and its serial, parallel and keyboard siblings
//! in jamesbraid/qemu-sgi-origami.

mod board;
mod dma;
mod gpio;
mod kbd;
mod parallel;
mod serial;
mod superio;
mod w1;

pub mod eth;

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::{Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::dev::battery_clock;
use crate::dev::ds1687::{image, MaceRtc};
pub use self::board::{ByteBusPart, Cells, Ioc3Board, BASEIO, BASEIO6G, IP29, IP30, IP34, IP34_NO_PS2};
pub use self::dma::Ioc3Dma;
use self::gpio::Gpio;
pub use self::gpio::{PinUse, PinWire};
use self::kbd::Kbd;
use self::parallel::Parallel;
use self::eth::{Ioc3Eth, Phy};
use self::serial::Serial;
use self::superio::SuperIo;
use self::w1::W1;
use crate::dev::ps2::Ps2Controller;
use crate::dev::w1 as onewire;
use crate::dev::w1::sgi;
use crate::dev::serial_host::SerialBackend;
use crate::irq::IrqLine;
use crate::pci::{Bar, BoardIrq, PciBus, PciConfig, PciDma, PciFunction, PciFunctionState, PCI_INTA};
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable, Saveable};
use crate::snapshot::{field, field_bool, field_u32, field_u64, field_u8, field_u8s, StateTable};

pub const VENDOR_ID: u16 = 0x10a9;
pub const DEVICE_ID: u16 = 0x0003;
pub const REVISION: u8 = 0x01;
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
    pub const BYTEBUS_DEV0: u32 = 0x80000;
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

/// Chip control: the bits the register stores, and the arbiter-idle bit it
/// always reads set.
const CHIP_CTRL_MASK: u32 = 0x0007_ffff;
const CHIP_CTRL_ARB_IDLE: u32 = 1 << 22;

// ---------------------------------------------------------------------------
// The DS1685 on the byte bus
// ---------------------------------------------------------------------------

/// How long the guest's NVRAM writes must pause before the file is
/// rewritten.
const NVRAM_QUIET: Duration = Duration::from_millis(500);
/// Registers 0x0e.. are storage rather than clock in the file.
const NVRAM_FIRST: usize = 0x0e;
/// The file is the DS1685's 512-byte image (see `ds1687::image`).
const NVRAM_IMAGE_LEN: usize = image::LEN;
const NVRAM_OFF_NV: usize = image::NV;
const NVRAM_OFF_EXT: usize = image::EXT;
const NVRAM_OFF_XRAM: usize = image::XRAM;
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
    /// Where the clock starts with no reference saved (`[rtc_offset]`).
    rtc_offset: crate::config::RtcOffset,
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
            rtc_offset: Default::default(),
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
        // A write to the clock registers can set the clock, which is saved
        // beside it. Register A's cannot, and the PROM flips its bank bit
        // around every extended-RAM access.
        if i != 0x0a && self.file.is_some() {
            self.dirty_since = Some(Instant::now());
        }
    }

    fn image(&self) -> Vec<u8> {
        let mut img = vec![0u8; NVRAM_IMAGE_LEN];
        let (regs, century) = self.dev.image_registers();
        img[..0x40].copy_from_slice(&regs);
        img[NVRAM_OFF_NV..NVRAM_OFF_NV + 64].copy_from_slice(&self.nv);
        let mut ext = self.ext;
        ext[0x08] = century;
        ext[0x0a] = self.dev.peek(0x4a);
        img[NVRAM_OFF_EXT..NVRAM_OFF_EXT + 64].copy_from_slice(&ext);
        img[NVRAM_OFF_XRAM..NVRAM_OFF_XRAM + RTC_XRAM_SIZE].copy_from_slice(&self.xram);
        img
    }

    /// Load a 512-byte image. The clock registers (0x00..0x0d) are never
    /// loaded: the clock comes from its reference. Returns false for any
    /// other length.
    fn load(&mut self, img: &[u8]) -> bool {
        if img.len() != NVRAM_IMAGE_LEN {
            return false;
        }
        self.dev.load_image_registers(img);
        self.nv.copy_from_slice(&img[NVRAM_OFF_NV..NVRAM_OFF_NV + 64]);
        self.ext.copy_from_slice(&img[NVRAM_OFF_EXT..NVRAM_OFF_EXT + 64]);
        self.xram.copy_from_slice(&img[NVRAM_OFF_XRAM..NVRAM_OFF_XRAM + RTC_XRAM_SIZE]);
        // Loading never powers the machine off.
        self.dev.poke(0x4a, self.ext[0x0a] & !0x08);
        true
    }

    /// The file, the image to put in it and the clock reference to put
    /// beside it, if the RAM changed at least `quiet` ago; the RAM then
    /// counts as saved.
    fn take_dirty(&mut self, quiet: Duration) -> Option<NvramJob> {
        let since = self.dirty_since?;
        if since.elapsed() < quiet {
            return None;
        }
        self.dirty_since = None;
        Some((self.file.clone()?, self.image(), self.dev.clock_reference()))
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
/// renamed), so a kill mid-write leaves the previous image whole, then the
/// clock reference beside it, if the guest has set the clock.
fn save_nvram((path, image, clock): &NvramJob) -> std::io::Result<()> {
    let name = path
        .to_str()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "NVRAM path is not UTF-8"))?;
    crate::nv_storage::save(name, image)?;
    match clock {
        Some(c) => battery_clock::write(&battery_clock::companion(path), c),
        None => Ok(()),
    }
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

/// The clock on the byte bus, built and owned by the machine because it
/// keeps its RAM in a file the machine names.
pub enum ByteBus {
    None,
    /// The M48T35-compatible Timekeeper, on select 0.
    M48t35(Arc<crate::dev::timekeeper::Timekeeper>),
    /// The DS1685 on selects 1 and 2. Its RAM and file are the chip's for now
    /// ([`Ioc3::set_nvram_file`]).
    Ds1685,
}

impl ByteBus {
    fn part(&self) -> ByteBusPart {
        match self {
            ByteBus::None => ByteBusPart::None,
            ByteBus::M48t35(_) => ByteBusPart::M48t35,
            ByteBus::Ds1685 => ByteBusPart::Ds1685,
        }
    }
}

/// The wire the board carries by default: the station-address part.
fn default_wire(mac: [u8; 6]) -> onewire::Bus {
    let mut bus = onewire::Bus::new();
    bus.add(Box::new(sgi::ethernet_address_part(mac, sgi::ETHERNET_PART_SERIAL)));
    bus
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
    (reg::SUPERIO..reg::SUPERIO + reg::SUPERIO_SIZE).contains(&a) || a >= reg::BYTEBUS_DEV0
}


impl Shared {
    fn io_drive(&self, o: &mut IntOut, level: bool) {
        if level && !o.state {
            self.core.set_bits(int::RT_OUTPUT_PULSE);
        }
        o.state = level;
    }

    fn io_hold_high(&self, o: &mut IntOut) {
        o.active = IO_HIGH;
        o.state = true;
        o.pulse_end = false;
        o.square_started = false;
        o.next = None;
    }

    /// One transition due at `at`; the next is scheduled from `at`, so a
    /// late one keeps the output's phase.
    fn io_step(&self, o: &mut IntOut, at: Instant) {
        let active = o.active;
        let count = active & IO_COUNT;
        match active & IO_MODE {
            IO_LOW => {
                self.io_drive(o, false);
                o.pulse_end = false;
            }
            IO_HIGH => {
                self.io_drive(o, true);
                o.pulse_end = false;
            }
            IO_ONE_SHOT => {
                if !o.pulse_end {
                    self.io_drive(o, true);
                    o.pulse_end = true;
                    o.next = Some(at + IO_TICK * IO_PULSE_TICKS);
                } else {
                    self.io_drive(o, false);
                    o.pulse_end = false;
                }
            }
            IO_PERIODIC => {
                if count < IO_PULSE_TICKS {
                    // No low phase: the output holds high.
                    self.io_drive(o, true);
                    self.io_hold_high(o);
                } else if !o.pulse_end {
                    self.io_drive(o, true);
                    o.pulse_end = true;
                    o.next = Some(at + IO_TICK * IO_PULSE_TICKS);
                } else {
                    self.io_drive(o, false);
                    o.pulse_end = false;
                    o.next = Some(at + IO_TICK * (count + 1 - IO_PULSE_TICKS));
                }
            }
            _ => {
                // IO_SQUARE
                if o.square_started {
                    let s = !o.state;
                    self.io_drive(o, s);
                } else {
                    o.square_started = true;
                }
                o.next = Some(at + IO_TICK * (count + 1));
            }
        }
    }

    /// Transitions already due happen before a read, not whenever the
    /// receive thread or a timer next runs.
    fn io_catch_up_locked(&self, o: &mut IntOut, now: Instant) {
        for _ in 0..100_000 {
            match o.next {
                Some(due) if due <= now => {
                    o.next = None;
                    self.io_step(o, due);
                }
                _ => return,
            }
        }
        // Far behind (a stopped host): the sticky status bit is already set;
        // resume the phase from now.
        if o.next.is_some() {
            o.next = Some(now + IO_TICK);
        }
    }
}

// ---------------------------------------------------------------------------
// What the parent shares with its children
// ---------------------------------------------------------------------------

/// The chip-wide state every child reaches: the interrupt status and enable
/// words, the registers more than one child reads (chip control, the ring
/// base), the chip's view of memory and the SuperIO line.
///
/// The status and enable words and the level last driven onto the SuperIO
/// line sit under one small lock. A change to the words and the drive of the
/// line happen inside it, and the parent is called only when the output
/// (status ANDed with enable) changes. The lock orders the edges, so the
/// Bridge, which turns a rising edge into a packet, sees exactly the edges
/// the registers made and in the order they made them: a rise only after a
/// change that really raised the output. The Bridge ignores a level it
/// already has and never calls back into the chip while it holds a lock of
/// its own, so the lock is taken child, then this one, then the Bridge, and
/// never the other way.
struct Core {
    sio: Mutex<Sio>,
    chip_control: AtomicU32,
    ring_hi: AtomicU32,
    ring_lo: AtomicU32,
    dma: OnceLock<Arc<dyn PciDma>>,
    superio_line: OnceLock<IrqLine>,
    /// PCI interrupt pin A: the Ethernet block, which drives it.
    inta: OnceLock<IrqLine>,
}

/// What `Core`'s lock guards.
#[derive(Default)]
struct Sio {
    status: u32,
    enable: u32,
    /// The level last driven onto the SuperIO line.
    out: bool,
}

impl Sio {
    fn output(&self) -> bool {
        self.status & self.enable != 0
    }
}

impl Core {
    fn new() -> Core {
        Core {
            sio: Mutex::new(Sio::default()),
            chip_control: AtomicU32::new(0),
            ring_hi: AtomicU32::new(0),
            ring_lo: AtomicU32::new(0),
            dma: OnceLock::new(),
            superio_line: OnceLock::new(),
            inta: OnceLock::new(),
        }
    }

    fn dma(&self) -> Option<&Arc<dyn PciDma>> {
        self.dma.get()
    }

    /// The interrupt status word.
    fn status(&self) -> u32 {
        self.sio.lock().status
    }

    /// The interrupt enable word.
    fn enabled(&self) -> u32 {
        self.sio.lock().enable
    }

    /// Change the words with `f` and, if that moved the output, drive the
    /// line before letting go.
    fn change(&self, f: impl FnOnce(&mut Sio)) {
        let mut s = self.sio.lock();
        f(&mut s);
        let want = s.output();
        if want != s.out {
            s.out = want;
            if let Some(line) = self.superio_line.get() {
                line.set(want);
            }
        }
    }

    /// Both words, stored whole (reset and snapshot load).
    fn set_words(&self, status: u32, enable: u32) {
        self.change(|s| {
            s.status = status;
            s.enable = enable;
        });
    }

    /// Raise the sources in `mask`.
    fn set_bits(&self, mask: u32) {
        self.change(|s| s.status |= mask);
    }

    /// Lower the sources in `mask`.
    fn clear_bits(&self, mask: u32) {
        self.change(|s| s.status &= !mask);
    }

    /// Put the sources in `mask` at `on`.
    fn level(&self, mask: u32, on: bool) {
        self.change(|s| if on { s.status |= mask } else { s.status &= !mask });
    }

    fn enable(&self, mask: u32) {
        self.change(|s| s.enable |= mask);
    }

    fn disable(&self, mask: u32) {
        self.change(|s| s.enable &= !mask);
    }

    /// Tell the line where the output is, changed or not: for when the line
    /// is first connected or the parent's state was just restored.
    fn republish(&self) {
        let s = self.sio.lock();
        if let Some(line) = self.superio_line.get() {
            line.set(s.out);
        }
    }
}

/// An optional deadline as one word: nanoseconds after the process epoch,
/// plus one; zero for none. A deadline can then be read without a lock.
fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

fn encode(t: Option<Instant>) -> u64 {
    t.map_or(0, |t| t.checked_duration_since(epoch()).map_or(1, |d| d.as_nanos() as u64 + 1))
}

fn decode(n: u64) -> Option<Instant> {
    (n != 0).then(|| epoch() + Duration::from_nanos(n - 1))
}

// ---------------------------------------------------------------------------
// The device
// ---------------------------------------------------------------------------

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
type NvramJob = (std::path::PathBuf, Vec<u8>, Option<Vec<u8>>);

/// Lock order: `gate`, then a child's lock (the serial engine's before the
/// Super I/O's it reaches through), then the interrupt words' lock in `Core`,
/// under which the SuperIO line is driven, then the Bridge's; `armed`, the
/// line slots (`pin_line`) and the NVRAM hand-off are leaves. The Bridge never
/// calls back into the chip while it holds a lock of its own, so a child's
/// lock may be held while the SuperIO line is driven.
/// The serial and parallel engines move their rings with their lock held: the
/// DMA goes down through the Bridge and HEART, which never call back into
/// the chip, and an access to the Ethernet block goes to it with no lock
/// held.
struct Shared {
    pci: PciFunctionState,
    core: Arc<Core>,
    board: &'static Ioc3Board,
    sio: Option<Arc<SuperIo>>,
    serial: [Option<Serial>; 2],
    kbd: Option<Arc<Kbd>>,
    /// The controller the host feeds keys to on a board with no keyboard
    /// port: nothing reads it.
    ps2_idle: Arc<Ps2Controller>,
    parallel: Option<Parallel>,
    w1: W1,
    gpio: Gpio,
    int_out: Mutex<IntOut>,
    /// The interrupt output's next transition, where the timers can see it.
    io_due: AtomicU64,
    rtc: Mutex<Rtc>,
    /// The NVRAM file's debounce deadline, likewise.
    nvram_due: AtomicU64,
    /// The DS1685 was told to power the machine off and the machine has not
    /// been told yet.
    power_off_pending: AtomicBool,
    ssram: Mutex<Vec<u32>>,
    eth: Option<Arc<Ioc3Eth>>,
    me: Weak<Shared>,
    timers: OnceLock<Arc<TimerManager>>,
    events: OnceLock<SyncSender<MachineEvent>>,
    /// Between `start` and `stop`.
    running: AtomicBool,
    /// Held by a timer callback while it runs, so `stop` can wait one out.
    gate: Mutex<()>,
    /// The timer pending for each deadline kind, and the deadline it serves.
    armed: Mutex<[Option<(TimerId, Instant)>; KINDS]>,
    /// The deadlines in `armed`, readable without its lock.
    armed_due: [AtomicU64; KINDS],
    /// The NVRAM image the writer thread has yet to put in the file.
    nvram_job: Mutex<Option<NvramJob>>,
    nvram_ready: Condvar,
}

impl Shared {
    /// What follows every access: arm the timers for the deadlines it left,
    /// pick up a PS/2 character that arrived meanwhile, and tell the machine
    /// of a power-off.
    fn after(&self) {
        self.arm();
        if let Some(k) = &self.kbd {
            k.service();
        }
        if self.power_off_pending.swap(false, Ordering::SeqCst) {
            if let Some(tx) = self.events.get() {
                let _ = tx.try_send(MachineEvent::PowerOff);
            }
        }
    }

    /// The host-time deadlines the chip holds, for the timers: the serial
    /// receive timers, transmit delays, the interrupt output while its status
    /// bit can reach the line, and the NVRAM file's debounce.
    fn deadlines(&self) -> Deadlines {
        let mut d = [None; KINDS];
        for (p, s) in self.serial.iter().enumerate() {
            if let Some(s) = s {
                let (rx, tx) = s.due();
                d[K_RX + p] = rx;
                d[K_TX_DELAY + p] = tx;
            }
        }
        if self.core.enabled() & int::RT_OUTPUT_PULSE != 0 {
            d[K_INT_OUT] = decode(self.io_due.load(Ordering::SeqCst));
        }
        d[K_NVRAM] = decode(self.nvram_due.load(Ordering::SeqCst)).map(|t| t + NVRAM_QUIET);
        d
    }

    /// Make the pending timers match the deadlines the chip holds: one
    /// one-shot per deadline. A timer whose deadline moved is replaced; with
    /// none set or the chip not running, deadlines are only evaluated lazily,
    /// by the next access.
    ///
    /// The deadlines are read again once `armed` is held, never before. Each
    /// child publishes its deadline under its own lock before its access
    /// returns to arm, so a pass that takes `armed` afterwards reads at least
    /// what that access left. A pass that read first and was held up (a
    /// receive thread's poll just before a register write) would otherwise
    /// arm from the older deadlines and remove the timer the write set.
    fn arm(&self) {
        let Some(tm) = self.timers.get() else { return };
        // Unlocked, only to skip the table when nothing moved; a read that
        // is stale here costs a pass the writer that made it stale also runs.
        if self.deadlines().iter().zip(&self.armed_due).all(|(w, a)| encode(*w) == a.load(Ordering::SeqCst)) {
            return;
        }
        let mut armed = self.armed.lock();
        // Checked under the lock `stop` clears the slots with: a pass that
        // sees `running` set has its timers removed by that stop.
        if !self.running.load(Ordering::SeqCst) {
            return;
        }
        let want = self.deadlines();
        for (k, w) in want.iter().enumerate() {
            if armed[k].map(|(_, d)| d) == *w {
                continue;
            }
            if let Some((id, _)) = armed[k].take() {
                tm.remove(id);
            }
            self.armed_due[k].store(0, Ordering::SeqCst);
            let Some(due) = *w else { continue };
            let mut delay = due.saturating_duration_since(Instant::now());
            if k == K_INT_OUT {
                delay = delay.max(IO_REARM_MIN);
            }
            let id = tm.add_one_shot(delay, self.me.clone(), move |me| Shared::fire(me, k, due));
            armed[k] = Some((id, due));
            self.armed_due[k].store(encode(Some(due)), Ordering::SeqCst);
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
                sh.armed_due[kind].store(0, Ordering::SeqCst);
            }
        }
        if kind == K_NVRAM {
            // The file is written by its own thread, not here.
            if let Some(job) = sh.take_nvram_dirty(NVRAM_QUIET) {
                *sh.nvram_job.lock() = Some(job);
                sh.nvram_ready.notify_all();
            }
            sh.rearm();
        } else {
            sh.catch_up(Instant::now());
            sh.after();
        }
        TimerReturn::Delete
    }

    /// The expiries already due, everywhere the chip keeps one.
    fn catch_up(&self, now: Instant) {
        for s in self.serial.iter().flatten() {
            s.catch_up(now);
        }
        self.io_catch_up(now);
    }

    fn io_catch_up(&self, now: Instant) {
        let mut o = self.int_out.lock();
        self.io_catch_up_locked(&mut o, now);
        self.io_due.store(encode(o.next), Ordering::SeqCst);
    }

    /// Arm for whatever deadlines the chip holds now.
    fn rearm(&self) {
        self.arm();
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
                if let Some(s) = &sh.serial[port] {
                    s.pump();
                }
                sh.after();
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
            report_nvram_save(&job.0, save_nvram(&job));
        }
    }

    /// The NVRAM image to write, if the RAM changed at least `quiet` ago.
    fn take_nvram_dirty(&self, quiet: Duration) -> Option<NvramJob> {
        let mut rtc = self.rtc.lock();
        let job = rtc.take_dirty(quiet);
        self.nvram_due.store(encode(rtc.dirty_since), Ordering::SeqCst);
        job
    }

    fn eth(&self) -> Option<&Arc<Ioc3Eth>> {
        self.eth.as_ref()
    }

    fn flush_nvram(&self, quiet: Duration) {
        let Some(job) = self.take_nvram_dirty(quiet) else { return };
        report_nvram_save(&job.0, save_nvram(&job));
    }

    /// One synchronous pass of what the receive threads and timers do:
    /// host input into the FIFOs, expired receive timers and transmit
    /// delays, the interrupt output, and any PS/2 character that arrived.
    #[cfg(test)]
    fn service(&self) {
        let now = Instant::now();
        for s in self.serial.iter().flatten() {
            s.service(now);
        }
        self.io_catch_up(now);
        if let Some(k) = &self.kbd {
            k.poke_for_test();
        }
        self.after();
    }

    // ----- register decode -----

    fn sdma_port_of(a: u32) -> (usize, u32) {
        if a >= reg::SDMA_PORT_B {
            (1, a - reg::SDMA_PORT_B)
        } else {
            (0, a - reg::SDMA_PORT_A)
        }
    }

    fn main_read(&self, a: u32) -> u32 {
        if Gpio::owns(a) {
            return self.gpio.read(a);
        }
        match a {
            reg::INT_STATUS => {
                let now = Instant::now();
                for s in self.serial.iter().flatten() {
                    s.catch_up(now);
                }
                self.core.status()
            }
            // The set alias reads back the live mask; the clear one does not.
            reg::INT_ENABLE_SET => self.core.enabled(),
            reg::CHIP_CONTROL => self.core.chip_control.load(Ordering::SeqCst) | CHIP_CTRL_ARB_IDLE,
            reg::INT_OUTPUT => {
                let mut o = self.int_out.lock();
                self.io_catch_up_locked(&mut o, Instant::now());
                self.io_due.store(encode(o.next), Ordering::SeqCst);
                o.reg | if o.state { IO_STATE } else { 0 }
            }
            reg::MICROLAN => self.w1.read(),
            0x80..=0x98 => self.parallel.as_ref().map_or(0, |p| p.read(a)),
            0x9c | 0xa0 | 0xa4 => self.kbd.as_ref().map_or(0, |k| k.read(a)),
            0xb0 => self.core.ring_hi.load(Ordering::SeqCst),
            0xb4 => self.core.ring_lo.load(Ordering::SeqCst),
            0xb8..=0xef => {
                let (p, off) = Self::sdma_port_of(a);
                self.serial[p].as_ref().map_or(0, |s| s.read(off))
            }
            _ => {
                log::debug!("ioc3: read of unimplemented register {a:#x}");
                0
            }
        }
    }

    fn main_write(&self, a: u32, val: u32) {
        if Gpio::owns(a) {
            self.gpio.write(a, val);
            return;
        }
        match a {
            reg::INT_STATUS => {
                // An acknowledgement must not be undone by a late expiry.
                let now = Instant::now();
                for s in self.serial.iter().flatten() {
                    s.catch_up(now);
                }
                // Write-one-to-clear; level-following bits ignore the write.
                self.core.clear_bits(val & !int::LEVELS);
            }
            reg::INT_ENABLE_SET => self.core.enable(val),
            reg::INT_ENABLE_CLR => self.core.disable(val),
            reg::CHIP_CONTROL => {
                self.core.chip_control.store(val & CHIP_CTRL_MASK, Ordering::SeqCst);
                // The port offsets aim the engines; one aimed here may have
                // work waiting that no event will start.
                for s in self.serial.iter().flatten() {
                    s.kick();
                }
            }
            reg::INT_OUTPUT => {
                let mut o = self.int_out.lock();
                self.io_write(&mut o, val);
                self.io_due.store(encode(o.next), Ordering::SeqCst);
            }
            reg::MICROLAN => self.w1.write(val, self.gpio.control()),
            0x80..=0x98 => {
                if let Some(p) = &self.parallel {
                    p.write(a, val);
                }
            }
            0x9c | 0xa8 | 0xac => {
                if let Some(k) = &self.kbd {
                    k.write(a, val);
                }
            }
            0xb0 => self.core.ring_hi.store(val, Ordering::SeqCst),
            0xb4 => self.core.ring_lo.store(val, Ordering::SeqCst),
            0xb8..=0xef => {
                let (p, off) = Self::sdma_port_of(a);
                if let Some(s) = &self.serial[p] {
                    s.write(off, val);
                }
            }
            _ => log::debug!("ioc3: write of {val:#x} to unimplemented register {a:#x}"),
        }
    }

    fn io_write(&self, o: &mut IntOut, val: u32) {
        let v = val & IO_MASK;
        // Stored as written; a command with no defined behaviour leaves the
        // running output alone.
        o.reg = v;
        if !int_out_valid(v) {
            return;
        }
        let now = Instant::now();
        o.next = None;
        o.active = v & (IO_MODE | IO_COUNT);
        o.pulse_end = false;
        o.square_started = false;
        if v & IO_MODE == IO_ONE_SHOT {
            o.state = false;
        }
        if v & IO_MODE == IO_LOW || v & IO_MODE == IO_HIGH {
            // A fixed level is there for the next read.
            self.io_step(o, now);
        } else {
            o.next = Some(now + Duration::from_nanos(1));
        }
    }

    // ----- byte-wide windows -----

    fn read_byte(&self, a: u32) -> u8 {
        if (reg::SUPERIO..reg::SUPERIO + reg::SUPERIO_SIZE).contains(&a) {
            for (p, base) in [(0, reg::SERIAL_A), (1, reg::SERIAL_B)] {
                if (base..base + 8).contains(&a) {
                    if let Some(s) = &self.serial[p] {
                        return s.pio_read(a - base);
                    }
                }
            }
            if (reg::PAR_ECP_BASE..reg::PAR_ECP_BASE + 0x0b).contains(&a) {
                if let Some(sio) = self.sio.as_ref().filter(|s| s.has_parallel()) {
                    return sio.ecp_read(a - reg::PAR_ECP_BASE);
                }
            }
            return 0;
        }
        if self.board.cells.byte_bus == ByteBusPart::Ds1685 {
            // Each select is a byte-wide port answering on any lane of its
            // first word: the guests land on different ones.
            if (reg::BYTEBUS_DEV2..reg::BYTEBUS_DEV2 + 4).contains(&a) {
                return self.rtc.lock().read();
            }
            if (reg::BYTEBUS_DEV1..reg::BYTEBUS_DEV1 + 4).contains(&a) {
                // A write port on the part; the latch is what is there.
                return self.rtc.lock().index;
            }
        }
        0
    }

    fn write_byte(&self, a: u32, v: u8) {
        if (reg::SUPERIO..reg::SUPERIO + reg::SUPERIO_SIZE).contains(&a) {
            for (p, base) in [(0, reg::SERIAL_A), (1, reg::SERIAL_B)] {
                if (base..base + 8).contains(&a) {
                    if let Some(s) = &self.serial[p] {
                        s.pio_write(a - base, v);
                        return;
                    }
                }
            }
            if (reg::PAR_ECP_BASE..reg::PAR_ECP_BASE + 0x0b).contains(&a) {
                if let (Some(sio), Some(par)) = (self.sio.as_ref().filter(|s| s.has_parallel()), &self.parallel) {
                    let ecr = a == reg::PAR_ECP_BASE + 0xa;
                    // Acknowledge service before starting DMA so a new
                    // completion can assert it again.
                    if ecr {
                        self.core.clear_bits(int::PARALLEL_SERVICE);
                    }
                    sio.ecp_write(a - reg::PAR_ECP_BASE, v);
                    if ecr {
                        par.run();
                    }
                }
            }
            return;
        }
        if self.board.cells.byte_bus == ByteBusPart::Ds1685 {
            if (reg::BYTEBUS_DEV2..reg::BYTEBUS_DEV2 + 4).contains(&a) {
                let mut rtc = self.rtc.lock();
                let was = rtc.dev.powered_off();
                rtc.write(v);
                if !was && rtc.dev.powered_off() {
                    self.power_off_pending.store(true, Ordering::SeqCst);
                }
                self.nvram_due.store(encode(rtc.dirty_since), Ordering::SeqCst);
            } else if (reg::BYTEBUS_DEV1..reg::BYTEBUS_DEV1 + 4).contains(&a) {
                // Seven bits; the bank comes from register A.
                self.rtc.lock().index = v & 0x7f;
            }
        }
    }

    // ----- anywhere in the BAR -----

    /// A read of `w` bytes at internal offset `a`.
    fn read(&self, a: u32, w: usize) -> u32 {
        if is_byte_wide(a) {
            // The windows are byte-wide: a wider access is several registers.
            let mut v = 0;
            for i in 0..w as u32 {
                v |= (self.read_byte(a + i) as u32) << (8 * i);
            }
            return v;
        }
        // The 32-bit blocks take 4-byte accesses only; a narrower one is
        // refused, not widened to a lane. There is no bus error to return
        // from `bar_read`: it reads zero, and the write below is dropped.
        if w != 4 {
            narrow_once(a, w);
            return 0;
        }
        if (reg::SSRAM..reg::SSRAM + reg::SSRAM_SIZE).contains(&a) {
            let mut word = self.ssram.lock()[((a - reg::SSRAM) / 4) as usize];
            if self.parity_checked() && word.count_ones() & 1 != 0 {
                word |= SSRAM_PARITY_ERROR;
            }
            word
        } else if (reg::SDMA_ALIAS_A..reg::SDMA_ALIAS_B + reg::SDMA_ALIAS_SIZE).contains(&a) {
            let p = (a >= reg::SDMA_ALIAS_B) as usize;
            let off = a & (reg::SDMA_ALIAS_SIZE - 1) & !3;
            match &self.serial[p] {
                Some(s) if off < serial::sdma_off::REGS * 4 => s.read(off),
                _ => 0,
            }
        } else if a < 0x200 {
            self.main_read(a & !3)
        } else {
            0
        }
    }

    fn write(&self, a: u32, v: u32, w: usize) {
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
            if self.parity_checked() && v & SSRAM_PARITY_ERROR != 0 {
                word &= !SSRAM_PARITY;
                if (word & SSRAM_DATA).count_ones() & 1 != 0 {
                    word |= SSRAM_PARITY;
                }
            }
            self.ssram.lock()[((a - reg::SSRAM) / 4) as usize] = word;
        } else if (reg::SDMA_ALIAS_A..reg::SDMA_ALIAS_B + reg::SDMA_ALIAS_SIZE).contains(&a) {
            let p = (a >= reg::SDMA_ALIAS_B) as usize;
            let off = a & (reg::SDMA_ALIAS_SIZE - 1);
            // Three of the seven are read-only through a page: the two
            // indexes the engine advances and the 16550 shadow.
            if let Some(s) = &self.serial[p] {
                if off < serial::sdma_off::REGS * 4 && !serial::alias_read_only(off) {
                    s.write(off, v);
                }
            }
        } else if a < 0x200 {
            self.main_write(a, v);
        }
    }

    /// Whether buffer-RAM parity checking is on.
    fn parity_checked(&self) -> bool {
        self.eth().is_some_and(|e| e.read32(reg::ENET_MAC_CONTROL) & ENET_MCR_RAM_PARITY != 0)
    }

    /// A PCI reset of the chip (not of the clock or the buffer RAM).
    fn reset(&self) {
        if self.board.reset_pci_on_power_on {
            self.pci.config().reset();
        }
        self.core.set_words(0, 0);
        self.core.chip_control.store(0, Ordering::SeqCst);
        self.core.ring_hi.store(0, Ordering::SeqCst);
        self.core.ring_lo.store(0, Ordering::SeqCst);
        self.gpio.reset();
        if let Some(sio) = &self.sio {
            sio.reset();
        }
        for s in self.serial.iter().flatten() {
            s.reset();
        }
        if let Some(k) = &self.kbd {
            k.reset();
        }
        if let Some(p) = &self.parallel {
            p.reset();
        }
        self.w1.reset();
        {
            let mut o = self.int_out.lock();
            *o = IntOut::default();
            self.io_due.store(0, Ordering::SeqCst);
        }
        self.after();
    }
}

/// The IOC3 PCI function.
pub struct Ioc3 {
    sh: Arc<Shared>,
    timekeeper: Option<Arc<crate::dev::timekeeper::Timekeeper>>,
    /// The serial receive threads and the NVRAM writer, joined in `stop`.
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl Ioc3 {
    /// The IOC3 of `board`, with the station address `mac` on its one-wire
    /// part and the clock the machine built for the byte bus.
    pub fn new(board: &'static Ioc3Board, mac: [u8; 6], byte_bus: ByteBus) -> Ioc3 {
        let cells = &board.cells;
        assert_eq!(byte_bus.part(), cells.byte_bus, "{}: the byte bus fits {:?}", board.name, cells.byte_bus);
        let core = Arc::new(Core::new());
        let sio = (cells.serial || cells.parallel)
            .then(|| Arc::new(SuperIo::new(core.clone(), [cells.serial; 2], cells.parallel)));
        let serial = [0, 1].map(|p| {
            cells.serial.then(|| Serial::new(p, core.clone(), sio.clone().expect("a serial port has a Super I/O")))
        });
        let kbd = cells.kbd.then(|| Kbd::new(core.clone()));
        let parallel = cells.parallel.then(|| Parallel::new(core.clone(), sio.clone().expect("a parallel port has a Super I/O")));
        let gpio = Gpio::new(board.gpio);
        let eth = cells.eth.then(|| Ioc3Eth::new(core.clone(), mac, board.phy));
        let timekeeper = match byte_bus {
            ByteBus::M48t35(clock) => Some(clock),
            _ => None,
        };
        let sh = Arc::new_cyclic(|weak: &Weak<Shared>| {
            let bars = [
                Bar::Mem(BAR_SIZE),
                Bar::Unused,
                Bar::Unused,
                Bar::Unused,
                Bar::Unused,
                Bar::Unused,
            ];
            let pci = PciConfig::new_with_policy(
                VENDOR_ID, DEVICE_ID, 0xff_0000, REVISION, bars, PCI_INTA, board.pci_policy,
            );
            // The subsystem identity selects what a guest instantiates.
            pci.set_subsystem_identity(((board.subsystem_id as u32) << 16) | VENDOR_ID as u32);
            let w1 = W1::new(default_wire(mac));
            Shared {
                pci: PciFunctionState::new(pci),
                core: core.clone(),
                sio,
                serial,
                kbd,
                ps2_idle: Arc::new(Ps2Controller::new(None)),
                parallel,
                w1,
                gpio,
                int_out: Mutex::new(IntOut::default()),
                io_due: AtomicU64::new(0),
                rtc: Mutex::new(Rtc::new()),
                nvram_due: AtomicU64::new(0),
                power_off_pending: AtomicBool::new(false),
                ssram: Mutex::new(vec![0; (reg::SSRAM_SIZE / 4) as usize]),
                eth,
                me: weak.clone(),
                timers: OnceLock::new(),
                events: OnceLock::new(),
                running: AtomicBool::new(false),
                gate: Mutex::new(()),
                armed: Mutex::new([None; KINDS]),
                armed_due: std::array::from_fn(|_| AtomicU64::new(0)),
                nvram_job: Mutex::new(None),
                nvram_ready: Condvar::new(),
                board,
            }
        });
        Ioc3 { sh, timekeeper, threads: Mutex::new(Vec::new()) }
    }

    /// The timer manager for device deadlines. Set it before start.
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
    pub fn set_microlan(&self, bus: onewire::Bus) {
        self.sh.w1.set_bus(bus);
    }

    /// Wire the host end of serial port `port` (0 is the console): a TCP
    /// listener, a log tee, or nothing.
    pub fn set_serial_backend(&self, port: usize, backend: Arc<dyn SerialBackend>) {
        if let Some(sio) = &self.sh.sio {
            sio.set_serial_backend(port, backend);
        }
    }

    /// Where parallel-port output goes. Without one it is dropped. No machine
    /// wires one, so only the tests set it.
    #[cfg(test)]
    pub fn set_parallel_backend(&self, backend: Arc<dyn SerialBackend>) {
        if let Some(sio) = &self.sh.sio {
            sio.set_parallel_backend(backend);
        }
    }

    /// Back the DS1685's RAM (user RAM, bank-1 registers and the 128-byte
    /// extended RAM) with a file. A 512-byte file (the layout the QEMU model
    /// uses) is loaded. The file is
    /// written as 512 bytes now and replaced whole once the guest's writes
    /// to the RAM pause (see [`Ioc3::flush_nvram`]). The clock runs from the
    /// reference beside the file (`<file>.clock`, the layout QEMU's model
    /// keeps) if there is one, else from where [`Ioc3::set_rtc_offset`] put
    /// it, host time by default.
    pub fn set_nvram_file(&self, path: impl Into<std::path::PathBuf>) -> std::io::Result<()> {
        let path = path.into();
        let clock = battery_clock::companion(&path);
        let job = {
            let mut rtc = self.sh.rtc.lock();
            if let Ok(img) = std::fs::read(&path) {
                if !rtc.load(&img) {
                    log::warn!("ioc3: NVRAM file {} has {} bytes, not 512; ignored", path.display(), img.len());
                }
            }
            if let Some(guest) = battery_clock::read_or_warn(&clock).and_then(|r| rtc.dev.restore_clock(&r)) {
                battery_clock::note_override(&rtc.rtc_offset, &clock, guest);
            }
            rtc.file = Some(path.clone());
            rtc.dirty_since = None;
            self.sh.nvram_due.store(0, Ordering::SeqCst);
            (path, rtc.image(), rtc.dev.clock_reference())
        };
        save_nvram(&job)
    }

    /// Start the DS1685's clock where `[rtc_offset]` puts it, for as long
    /// as the NVRAM file keeps no clock of the guest's.
    pub fn set_rtc_offset(&self, cfg: &crate::config::RtcOffset) {
        let mut rtc = self.sh.rtc.lock();
        rtc.rtc_offset = *cfg;
        rtc.dev.set_offset(battery_clock::seed(cfg));
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

    /// Show the system board's front lightbar and the Ethernet activity in
    /// the status bar through `heartbeat`.
    pub fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        if let Some(e) = &self.sh.eth {
            e.set_heartbeat(heartbeat.clone());
        }
        self.sh.gpio.set_heartbeat(heartbeat);
    }

    /// The Ethernet MAC, on a board that fits one.
    pub fn eth(&self) -> Option<&Arc<Ioc3Eth>> {
        self.sh.eth.as_ref()
    }

    /// The line the board wires the SuperIO interrupt (serial, PS/2, parallel,
    /// interrupt output) to. The line in its [`PciFunctionState`] is the PCI
    /// pin, which only the Ethernet block drives.
    pub fn connect_superio(&self, bus: &PciBus) {
        self.connect_superio_interrupt(bus.board_irq(BoardIrq(self.sh.board.superio_pin)));
    }

    /// [`Ioc3::connect_superio`] with the line given.
    pub fn connect_superio_interrupt(&self, line: IrqLine) {
        if self.sh.core.superio_line.set(line).is_ok() {
            self.sh.core.republish();
        }
    }

    /// The PS/2 keyboard and mouse model, for the host side to push keys and
    /// pointer motion into. On a board with no keyboard port it is a
    /// controller nothing reads.
    pub fn ps2(&self) -> Arc<Ps2Controller> {
        self.sh.kbd.as_ref().map_or_else(|| self.sh.ps2_idle.clone(), |k| k.ps2().clone())
    }

    /// Whether the guest has asked the clock to power the machine off.
    pub fn powered_off(&self) -> bool {
        self.sh.rtc.lock().dev.powered_off()
    }

    /// A PCI reset of the chip (not of the clock, the buffer RAM or the
    /// Ethernet block, which keeps its state: `power_on` resets that too).
    pub fn reset(&self) {
        self.sh.reset();
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
            ("int_out", (|s: &Shared| s.int_out.is_locked()) as fn(&Shared) -> bool),
            ("rtc", |s| s.rtc.is_locked()),
        ] {
            let sh = Arc::downgrade(&self.sh);
            crate::locks::register_lock(format!("ioc3::{what}"), move || sh.upgrade().is_some_and(|s| probe(&s)));
        }
        for (what, probe) in [
            ("serial_a", (|s: &Shared| s.serial[0].as_ref().is_some_and(|x| x.is_locked())) as fn(&Shared) -> bool),
            ("serial_b", |s| s.serial[1].as_ref().is_some_and(|x| x.is_locked())),
            ("superio", |s| s.sio.as_ref().is_some_and(|x| x.is_locked())),
            ("kbd", |s| s.kbd.as_ref().is_some_and(|x| x.is_locked())),
        ] {
            let sh = Arc::downgrade(&self.sh);
            crate::locks::register_lock(format!("ioc3::{what}"), move || sh.upgrade().is_some_and(|s| probe(&s)));
        }
        if let Some(e) = &self.sh.eth {
            e.register_locks();
        }
    }

    /// The SuperIO interrupt status register, for tests and the monitor.
    pub fn int_status(&self) -> u32 {
        self.sh.core.status()
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
        if let Some(k) = &self.sh.kbd {
            Device::start(&**k.ps2());
        }
        if let Some(e) = &self.sh.eth {
            Device::start(&**e);
        }
        let mut threads = self.threads.lock();
        for (p, s) in self.sh.serial.iter().enumerate() {
            if s.is_some() {
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
            for (k, slot) in armed.iter_mut().enumerate() {
                if let (Some((id, _)), Some(tm)) = (slot.take(), self.sh.timers.get()) {
                    tm.remove(id);
                }
                self.sh.armed_due[k].store(0, Ordering::SeqCst);
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
        if let Some(k) = &self.sh.kbd {
            Device::stop(&**k.ps2());
        }
        if let Some(e) = &self.sh.eth {
            Device::stop(&**e);
        }
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
    /// The chip as at power-on, MAC and transceiver included. The clock and
    /// the buffer RAM keep their contents, as they are memory.
    fn power_on(&self) {
        self.reset();
        if let Some(e) = &self.sh.eth {
            e.power_on();
        }
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
        let v = self.sh.pci.config().read(index);
        if index == 1 { v | PCI_STATUS_FAST_BACK << 16 } else { v }
    }
}

impl PciFunction for Ioc3 {
    fn state(&self) -> &PciFunctionState {
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
        if (reg::BYTEBUS_DEV0..reg::BYTEBUS_DEV0 + crate::dev::timekeeper::SIZE as u32).contains(&a) {
            if let Some(clock) = self.timekeeper.as_ref() {
                if width != 1 {
                    crate::dlog!(crate::devlog::LogModule::Ioc, "IOC3 Timekeeper {width}-byte read refused");
                    return 0;
                }
                return u32::from(clock.read8(u64::from(a - reg::BYTEBUS_DEV0)).data);
            }
        }

        if (reg::ENET_FIRST..=reg::ENET_LAST).contains(&a) {
            // Only 32-bit accesses reach the Ethernet block.
            if width != 4 {
                return 0;
            }
            return self.sh.eth().map_or(0, |e| e.read32(a));
        }
        let v = self.sh.read(a, width);
        self.sh.after();
        v
    }

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize) {
        if bar != 0 || off >= BAR_SIZE {
            return;
        }
        let a = off ^ (4 - width.min(4) as u32);
        if (reg::BYTEBUS_DEV0..reg::BYTEBUS_DEV0 + crate::dev::timekeeper::SIZE as u32).contains(&a) {
            if let Some(clock) = self.timekeeper.as_ref() {
                if width != 1 {
                    crate::dlog!(crate::devlog::LogModule::Ioc, "IOC3 Timekeeper {width}-byte write refused");
                    return;
                }
                clock.write8(u64::from(a - reg::BYTEBUS_DEV0), val as u8);
                return;
            }
        }

        if (reg::ENET_FIRST..=reg::ENET_LAST).contains(&a) {
            if width == 4 && a & 3 == 0 {
                if let Some(e) = self.sh.eth() {
                    e.write32(a, val);
                }
            }
            return;
        }
        self.sh.write(a, val, width);
        self.sh.after();
    }

    /// The chip's DMA goes through its own byte order, and the PCI
    /// interrupt pin is the Ethernet block's level.
    fn attached(&self) {
        if let Some(dma) = self.sh.pci.dma() {
            let _ = self.sh.core.dma.set(Arc::new(Ioc3Dma::new(dma.clone())));
        }
        if let Some(line) = self.sh.pci.irq() {
            if self.sh.core.inta.set(line.clone()).is_ok() {
                if let Some(e) = &self.sh.eth {
                    e.republish();
                }
            }
        }
    }
}

// The IOC3's state for a snapshot: the SuperIO block (two UARTs and their
// ring DMA, the PS/2 ports and their controller, the parallel port, the
// interrupt output, the clock and its RAM), the pins, the MicroLAN and the
// Ethernet buffer RAM. The Ethernet block is a device of its own. The keys
// are the ones the single-lock chip wrote, whichever child now holds the
// state.
//
// The serial DMA and the interrupt output time themselves on the host's
// monotonic clock. Each deadline is saved as the time it had left and set
// that far after the load, so a restored guest sees the same interval it
// was waiting out. Deadlines are encoded as nanoseconds left plus one,
// zero for none.

pub(crate) fn until(due: Option<Instant>, now: Instant) -> u64 {
    due.map_or(0, |d| d.saturating_duration_since(now).as_nanos() as u64 + 1)
}

pub(crate) fn due(left: u64, now: Instant) -> Option<Instant> {
    (left != 0).then(|| now + Duration::from_nanos(left - 1))
}

pub(crate) fn span(d: Option<Duration>) -> u64 {
    d.map_or(0, |d| d.as_nanos() as u64 + 1)
}

pub(crate) fn unspan(n: u64) -> Option<Duration> {
    (n != 0).then(|| Duration::from_nanos(n - 1))
}

pub(crate) fn usize_field(v: &toml::Value, key: &str, limit: usize) -> Result<usize, String> {
    let n = field_u64(v, key)? as usize;
    if n > limit {
        return Err(format!("{key}: {n} is more than {limit}"));
    }
    Ok(n)
}

impl Shared {
    fn save_state(&self, now: Instant) -> toml::Value {
        let c = &self.core;
        let mut t = StateTable::new();
        t.u32("int_status", c.status())
            .u32("int_enable", c.enabled())
            .u32("chip_control", c.chip_control.load(Ordering::SeqCst));
        {
            let o = self.int_out.lock();
            let mut int_out = StateTable::new();
            int_out
                .u32("reg", o.reg)
                .u32("active", o.active)
                .bool("state", o.state)
                .bool("pulse_end", o.pulse_end)
                .bool("square_started", o.square_started)
                .u64("next", until(o.next, now));
            t.value("int_out", int_out.build());
        }
        let (microlan, wire) = self.w1.save_state();
        let (pin_control, pin_data) = self.gpio.words();
        t.u32("microlan", microlan).u32("pin_control", pin_control).u32("pin_data", pin_data);
        if let Some(sio) = &self.sio {
            for p in 0..2 {
                if self.serial[p].is_some() {
                    t.value(["uart0", "uart1"][p], sio.save_state(p));
                }
            }
        }
        if self.serial.iter().any(|s| s.is_some()) {
            let mut sdma = StateTable::new();
            sdma.u32("base_hi", c.ring_hi.load(Ordering::SeqCst)).u32("base_lo", c.ring_lo.load(Ordering::SeqCst));
            for p in 0..2 {
                if let Some(s) = &self.serial[p] {
                    sdma.value(["port0", "port1"][p], s.save_state(now));
                }
            }
            t.value("sdma", sdma.build());
        }
        if let Some(k) = &self.kbd {
            t.value("kbc", k.save_state());
        }
        if let (Some(sio), Some(par)) = (&self.sio, &self.parallel) {
            let mut p = StateTable::new();
            par.save_state(&mut p);
            sio.save_ecp(&mut p);
            t.value("parallel", p.build());
        }
        {
            let r = self.rtc.lock();
            let mut rtc = StateTable::new();
            rtc.value("clock", r.dev.save_state())
                .u8("index", r.index)
                .u8s("nv", &r.nv)
                .u8s("ext", &r.ext)
                .u8s("xram", &r.xram)
                .bool("dirty", r.dirty_since.is_some());
            t.value("rtc", rtc.build());
        }
        t.u32s("ssram", &self.ssram.lock()).value("wire", wire).bool(
            "power_off_pending",
            self.power_off_pending.load(Ordering::SeqCst),
        );
        t.build()
    }

    fn load_state(&self, v: &toml::Value, now: Instant) -> Result<(), String> {
        let ssram = field(v, "ssram")?.as_array().ok_or("ssram: not a list")?;
        if ssram.len() != (reg::SSRAM_SIZE / 4) as usize {
            return Err(format!("ssram: {} words, expected {}", ssram.len(), reg::SSRAM_SIZE / 4));
        }
        let ssram: Vec<u32> = ssram.iter().map(crate::snapshot::toml_u32).collect::<Option<_>>()
            .ok_or("ssram: an entry is not a word")?;
        let o = field(v, "int_out")?;
        let int_out = IntOut {
            reg: field_u32(o, "reg")?,
            active: field_u32(o, "active")?,
            state: field_bool(o, "state")?,
            pulse_end: field_bool(o, "pulse_end")?,
            square_started: field_bool(o, "square_started")?,
            next: due(field_u64(o, "next")?, now),
        };
        self.w1.load_state(field_u32(v, "microlan")?, field(v, "wire")?)?;
        if let Some(sio) = &self.sio {
            for p in 0..2 {
                if self.serial[p].is_some() {
                    sio.load_state(p, field(v, ["uart0", "uart1"][p])?)?;
                }
            }
        }
        if self.serial.iter().any(|s| s.is_some()) {
            let sdma = field(v, "sdma")?;
            self.core.ring_hi.store(field_u32(sdma, "base_hi")?, Ordering::SeqCst);
            self.core.ring_lo.store(field_u32(sdma, "base_lo")?, Ordering::SeqCst);
            for p in 0..2 {
                if let Some(s) = &self.serial[p] {
                    s.load_state(field(sdma, ["port0", "port1"][p])?, now)?;
                }
            }
        }
        if let Some(k) = &self.kbd {
            k.load_state(field(v, "kbc")?)?;
        }
        if let (Some(sio), Some(par)) = (&self.sio, &self.parallel) {
            let p = field(v, "parallel")?;
            sio.load_ecp(p)?;
            par.load_state(p)?;
        }
        {
            let r = field(v, "rtc")?;
            let mut rtc = self.rtc.lock();
            rtc.dev.load_state(field(r, "clock")?)?;
            rtc.index = field_u8(r, "index")?;
            rtc.nv = field_u8s(r, "nv")?;
            rtc.ext = field_u8s(r, "ext")?;
            rtc.xram = field_u8s(r, "xram")?;
            // The NVRAM file catches up with the restored RAM in the usual way.
            rtc.dirty_since = (field_bool(r, "dirty")? || rtc.file.is_some()).then_some(now);
            self.nvram_due.store(encode(rtc.dirty_since), Ordering::SeqCst);
        }
        *self.int_out.lock() = int_out;
        self.io_due.store(encode(self.int_out.lock().next), Ordering::SeqCst);
        let c = &self.core;
        c.set_words(field_u32(v, "int_status")?, field_u32(v, "int_enable")?);
        c.chip_control.store(field_u32(v, "chip_control")?, Ordering::SeqCst);
        self.gpio.restore(field_u32(v, "pin_control")?, field_u32(v, "pin_data")?);
        *self.ssram.lock() = ssram;
        self.power_off_pending.store(field_bool(v, "power_off_pending")?, Ordering::SeqCst);
        Ok(())
    }
}

impl Saveable for Ioc3 {
    fn save_state(&self) -> toml::Value {
        let now = Instant::now();
        let st = self.sh.save_state(now);
        StateTable::new()
            .value("chip", st)
            .value("ps2", self.ps2().save_state())
            .build()
    }

    /// Called with the chip stopped; its timers are armed again by `start`.
    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        self.ps2().load_state(field(v, "ps2")?)?;
        let now = Instant::now();
        self.sh.load_state(field(v, "chip")?, now)?;
        self.sh.rearm();
        self.sh.core.republish();
        if let Some(e) = &self.sh.eth {
            e.republish();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
