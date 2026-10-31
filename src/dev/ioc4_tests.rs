//! IOC4 register-contract tests.
//!
//! Accesses go through `PciFunction` the way the bridge makes them: IOC4
//! orders its own bytes, so a register is at its address and a UART register
//! is a byte access at its window. Each test builds only the chip and a
//! block of bus memory for the rings.
//!
//! Ported from tests/qtest/sgi-ioc4-test.c in jamesbraid/qemu-sgi-origami.
//! That file drives the chip through an XBridge with chardev sockets and the
//! virtual clock; here the sockets are a `Wire` per port and time is the host
//! clock, so timers are given a few milliseconds and `service` plays the part
//! of the main loop.

use super::*;
use crate::dev::ds1744::{self, Ds1744, YearBase};
use crate::irq::LevelProbe;
use std::sync::atomic::{AtomicI64, Ordering as AtomicOrdering};
use crate::pci::PciDmaStatus;
use std::collections::VecDeque;

const MEM_SIZE: usize = 0x10000;
/// Where the rings sit in bus memory.
const RINGS: u64 = 0x4000;
/// An address no memory answers.
const BAD_BASE: u64 = 0x20_0000;
const TX_VALID: u8 = 0x40;
const TX_EXPLICIT: u8 = 0x20;
const TX_MCR: u8 = 0x80;
const TX_DELAY: u8 = 0xc0;
/// Received status from the line: data and modem valid, CTS and DCD up.
const RX_FROM_LINE: u8 = 0xf0;
/// Received status looped back: RTS returns as CTS, and OUT2 clear is no DCD.
const RX_LOOPBACK: u8 = 0xd0;
const ENABLE: u32 = 1 << 28;
const RESET: u32 = 1 << 31;
const THRESHOLD: u32 = 1;
const ARM: u32 = 1 << 31;
const DRAIN: u32 = 1 << 27;
const SHADOW_OUTPUTS: u32 = 0x1f00_0000;
const ERROR_VALID: u32 = 1;
const ERROR_MULTIPLE: u32 = 1 << 5;

/// A sink for a serial line: what the guest sent, and what the host types.
#[derive(Default)]
struct Wire {
    out: std::sync::Mutex<Vec<u8>>,
    inp: std::sync::Mutex<VecDeque<u8>>,
}

impl SerialBackend for Wire {
    fn send_byte(&self, b: u8) {
        self.out.lock().unwrap().push(b);
    }
    fn recv_byte(&self) -> std::io::Result<u8> {
        self.inp
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::WouldBlock, "empty"))
    }
}

/// Bus memory, as bytes at bus addresses. Addresses past its end abort.
struct TestMem(std::sync::Mutex<Vec<u8>>);

impl PciDma for TestMem {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        let a = addr as usize;
        if addr > u64::from(u32::MAX) || a + buf.len() > m.len() {
            buf.fill(0);
            return PciDmaStatus::MasterAbort;
        }
        buf.copy_from_slice(&m[a..a + buf.len()]);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        let a = addr as usize;
        if addr > u64::from(u32::MAX) || a + data.len() > m.len() {
            return PciDmaStatus::MasterAbort;
        }
        m[a..a + data.len()].copy_from_slice(data);
        PciDmaStatus::Ok
    }
}

struct Rig {
    c: Ioc4,
    mem: Arc<TestMem>,
    wire: [Arc<Wire>; PORTS],
    pin: Arc<LevelProbe>,
}

impl Rig {
    fn new() -> Rig {
        Rig::with(Ioc4Config::bare(Ioc4Clock::Mhz33))
    }

    fn with(cfg: Ioc4Config) -> Rig {
        let c = Ioc4::new(cfg);
        let mem = Arc::new(TestMem(std::sync::Mutex::new(vec![0; MEM_SIZE])));
        c.state().connect_dma(mem.clone());
        let wire: [Arc<Wire>; PORTS] = std::array::from_fn(|_| Arc::new(Wire::default()));
        for (p, w) in wire.iter().enumerate() {
            c.set_serial_backend(p, w.clone());
        }
        let (pin, line) = LevelProbe::line();
        c.state().connect_irq(line);
        c.attached();
        Rig { c, mem, wire, pin }
    }

    /// A 32-bit register as the processor sees it through the self-ordered
    /// slot: the BAR value is the address-order bytes read little-endian.
    fn r32(&self, a: u32) -> u32 {
        self.c.bar_read(0, a, 4).swap_bytes()
    }
    fn w32(&self, a: u32, v: u32) {
        self.c.bar_write(0, a, v.swap_bytes(), 4)
    }
    fn r8(&self, a: u32) -> u8 {
        self.c.bar_read(0, a, 1) as u8
    }
    fn w8(&self, a: u32, v: u8) {
        self.c.bar_write(0, a, u32::from(v), 1)
    }

    /// UART register `r` of port `p`.
    fn ur(&self, p: usize, r: u32) -> u8 {
        self.r8(reg::UART + reg::UART_STRIDE * p as u32 + r)
    }
    fn uw(&self, p: usize, r: u32, v: u8) {
        self.w8(reg::UART + reg::UART_STRIDE * p as u32 + r, v)
    }

    /// A port's DMA register.
    fn sr(&self, p: usize, off: u32) -> u32 {
        self.r32(reg::SDMA_PORT + reg::SDMA_PORT_STRIDE * p as u32 + off)
    }
    fn sw(&self, p: usize, off: u32, v: u32) {
        self.w32(reg::SDMA_PORT + reg::SDMA_PORT_STRIDE * p as u32 + off, v)
    }

    /// Point a pair's rings at `base` with `size` byte rings.
    fn pair_base(&self, pair: u32, base: u64, size: u32) {
        let lo = reg::SDMA_BASE + 8 * pair;
        self.w32(lo + 4, (base >> 32) as u32);
        self.w32(lo, base as u32 | u32::from(size == 0x1000));
    }

    /// Where port `p`'s ring for `dir` (0 transmit, 1 receive) is, with its
    /// pair based at `RINGS`.
    fn ring(&self, p: usize, dir: u64, size: u64) -> usize {
        (RINGS + ((p as u64 & 1) * 2 + dir) * size) as usize
    }

    fn put(&self, at: usize, bytes: &[u8]) {
        self.mem.0.lock().unwrap()[at..at + bytes.len()].copy_from_slice(bytes);
    }

    fn get(&self, at: usize, n: usize) -> Vec<u8> {
        self.mem.0.lock().unwrap()[at..at + n].to_vec()
    }

    /// One transmit entry in port `p`'s ring, `at` bytes in.
    fn tx_entry(&self, p: usize, size: u64, at: usize, data: [u8; 4], control: [u8; 4]) {
        let ring = self.ring(p, 0, size) + at;
        self.put(ring, &data);
        self.put(ring + 4, &control);
    }

    /// Wait out `ms` of host time, then play the timers.
    fn settle(&self, ms: u64) {
        std::thread::sleep(Duration::from_millis(ms));
        self.c.service();
    }

    /// Play the timers until `done`, a millisecond of host time apart.
    fn until(&self, what: &str, done: impl Fn(&Rig) -> bool) {
        for _ in 0..3000 {
            self.c.service();
            if done(self) {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        panic!("timed out waiting for {what}");
    }

    /// Divisor one: a character crosses the line in about five microseconds
    /// instead of a millisecond, so a long loopback run stays short.
    fn fast_line(&self, p: usize) {
        self.uw(p, 3, 0x80);
        self.uw(p, 0, 1);
        self.uw(p, 1, 0);
        self.uw(p, 3, 0x03);
    }

    fn sent(&self, p: usize) -> Vec<u8> {
        self.wire[p].out.lock().unwrap().clone()
    }

    fn type_in(&self, p: usize, bytes: &[u8]) {
        self.wire[p].inp.lock().unwrap().extend(bytes.iter().copied());
        self.c.service();
    }
}

// ---------------------------------------------------------------------------
// PCI function
// ---------------------------------------------------------------------------

/// Ports test_identity_and_reset: vendor, device, revision 0x53, the
/// emulator's class and interrupt pin, and a 1 MiB BAR.
#[test]
fn identity_and_bar() {
    let r = Rig::new();
    let cfg = r.c.state().config();
    assert_eq!(cfg.read(0), 0x100a_10a9);
    assert_eq!(cfg.read(2), 0x00ff_0053);
    assert_eq!(cfg.interrupt_pin(), PCI_INTA);
    cfg.write(4, 0xffff_ffff);
    assert_eq!(cfg.read(4), 0xfff0_0000, "one megabyte of memory space");
}

/// The IP53 status policy: 0x0220 survives every write and a reset, and an
/// instance without the policy keeps its own value.
#[test]
fn status_follows_the_board() {
    let r = Rig::with(Ioc4Config::tezro());
    let cfg = r.c.state().config();
    assert_eq!(cfg.read(1) >> 16, 0x0220);
    cfg.write(1, 0x0000_0006);
    assert_eq!(cfg.read(1), 0x0220_0006);
    cfg.write(1, 0xffff_0000);
    assert_eq!(cfg.read(1) >> 16, 0x0220, "a status write of ones leaves the read-only fields");
    r.c.reset();
    assert_eq!(cfg.read(1), 0x0220_0000);

    let bare = Rig::new();
    assert_eq!(bare.c.state().config().read(1) >> 16, 0);
}

/// Ports the reset half of test_identity_and_reset: the driver's
/// disable-and-acknowledge is not a reset, but a reset clears what it left.
#[test]
fn reset_clears_the_interrupt_state_and_the_ports() {
    let r = Rig::new();
    r.w32(reg::SERIAL_ENABLE_SET, int::pass(0));
    r.w32(reg::OTHER_ENABLE_SET, 0x40);
    r.uw(0, 3, 0x80);
    r.uw(0, 0, 0x34);
    r.uw(0, 3, 0x03);
    r.uw(0, 1, 0x02);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(0), int::pass(0));
    assert!(r.pin.level());

    r.c.reset();
    assert_eq!(r.r32(reg::SERIAL_ENABLE_SET), 0);
    assert_eq!(r.r32(reg::OTHER_ENABLE_SET), 0);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(0), 0);
    assert!(!r.pin.level());
    assert_eq!(r.ur(0, 3), 0, "line control");
    assert_eq!(r.ur(0, 1), 0, "interrupt enable");
    assert_eq!(r.ur(0, 2) & 0xc0, 0, "FIFOs off");
    // The driver's own initialisation: raw pending, separate from enabled.
    r.uw(0, 1, 2);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(0), int::pass(0));
    assert!(!r.pin.level(), "pending but not enabled");
}

/// The pin follows the enabled sources of either word. Ports the IRQ half
/// of test_identity_and_reset.
#[test]
fn the_pin_follows_enabled_sources() {
    let r = Rig::new();
    r.uw(0, 1, 0x02);
    assert!(!r.pin.level());
    r.w32(reg::SERIAL_ENABLE_SET, int::pass(0));
    assert!(r.pin.level());
    r.w32(reg::SERIAL_ENABLE_CLR, int::pass(0));
    assert!(!r.pin.level());
    r.w32(reg::OTHER_ENABLE_SET, int::dma_error(1));
    assert!(!r.pin.level(), "nothing pending in the other word");
}

// ---------------------------------------------------------------------------
// The four 16550s and the local interrupt controller
// ---------------------------------------------------------------------------

/// Ports test_uart_and_clock: the misc-control bit names the board's clock,
/// writes to it are ignored, and each port's input is the clock over 16.
#[test]
fn clock_select_reads_back_and_sets_the_baud_base() {
    for (clock, bit, hz) in [(Ioc4Clock::Mhz33, 0, 33_333_333u32), (Ioc4Clock::Mhz66, 1, 66_666_667)] {
        let r = Rig::with(Ioc4Config::bare(clock));
        assert_eq!(r.r32(reg::MISC_CONTROL) & 1, bit);
        r.w32(reg::MISC_CONTROL, u32::from(bit == 0));
        assert_eq!(r.r32(reg::MISC_CONTROL) & 1, bit, "the clock is the board's");
        for p in 0..PORTS {
            assert_eq!(r.c.sh.st.lock().uart[p].baudbase(), hz / 16);
        }
    }
}

/// Ports test_uart_and_clock's per-port register loop.
#[test]
fn every_port_has_its_own_registers() {
    let r = Rig::new();
    for n in 0..PORTS {
        let n8 = n as u8;
        assert_eq!(r.ur(n, 5) & 0x60, 0x60, "transmitter empty after reset");
        r.uw(n, 3, 0x80);
        r.uw(n, 0, 0x31 + n8);
        r.uw(n, 1, 0x12 + n8);
        assert_eq!(r.ur(n, 0), 0x31 + n8, "divisor low");
        assert_eq!(r.ur(n, 1), 0x12 + n8, "divisor high");
        r.uw(n, 3, 3);
        r.uw(n, 2, 1);
        assert_eq!(r.ur(n, 2) & 0xc0, 0xc0, "FIFOs enabled");
        r.uw(n, 2, 7);
    }
    for n in 0..PORTS {
        assert_eq!(r.ur(n, 3), 3, "no port inherited another's line control");
    }
}

/// Ports test_interrupts: the enable-set and enable-clear aliases, raw
/// pending against enabled, and one port's source isolated from the others'.
#[test]
fn serial_interrupts_mask_pending_and_acknowledge() {
    let r = Rig::new();
    r.w32(reg::SERIAL_ENABLE_CLR, u32::MAX);
    assert_eq!(r.r32(reg::SERIAL_ENABLE_SET), 0);
    r.w32(reg::OTHER_ENABLE_CLR, u32::MAX);
    assert_eq!(r.r32(reg::OTHER_ENABLE_SET), 0);
    r.w32(reg::SERIAL_PENDING, u32::MAX);
    r.w32(reg::OTHER_PENDING, u32::MAX);
    for n in 0..PORTS {
        r.uw(n, 1, 2);
        assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(n), int::pass(n));
        assert!(!r.pin.level(), "pending but masked");
        r.w32(reg::SERIAL_ENABLE_SET, int::pass(n));
        assert_eq!(r.r32(reg::SERIAL_ENABLE_SET), int::pass(n));
        assert!(r.pin.level());
        r.w32(reg::SERIAL_PENDING, int::pass(n));
        assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(n), int::pass(n), "a level ignores the acknowledge");
        assert!(r.pin.level());
        assert_eq!(r.ur(n, 2) & 0xf, 2, "transmit-empty cause");
        assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(n), 0, "the identification read quieted it");
        assert!(!r.pin.level());
        r.w32(reg::SERIAL_ENABLE_CLR, int::pass(n));
        assert_eq!(r.r32(reg::SERIAL_ENABLE_SET), 0);
        r.uw(n, 1, 0);
    }
    for n in 0..PORTS {
        r.uw(n, 1, 2);
        r.w32(reg::SERIAL_ENABLE_SET, int::pass(n));
    }
    let all = (0..PORTS).fold(0, |m, n| m | int::pass(n));
    assert_eq!(r.r32(reg::SERIAL_PENDING) & all, all);
    r.ur(0, 2);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & all, all & !int::pass(0), "only port 0 quieted");
    assert!(r.pin.level());
    assert_eq!(r.r32(reg::OTHER_PENDING), 0, "no ATA or error source");
    r.w32(reg::OTHER_ENABLE_SET, 0x40);
    assert_eq!(r.r32(reg::OTHER_ENABLE_SET), 0x40);
    r.w32(reg::OTHER_ENABLE_CLR, 0x40);
    assert_eq!(r.r32(reg::OTHER_ENABLE_SET), 0);
}

/// Ports test_uart_pass_through_is_a_level: a write cannot clear a UART's
/// source, only the UART can.
#[test]
fn uart_pass_through_follows_the_uart() {
    let r = Rig::new();
    r.w32(reg::SERIAL_PENDING, u32::MAX);
    r.uw(0, 1, 2);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(0), int::pass(0));
    r.w32(reg::SERIAL_PENDING, int::pass(0));
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(0), int::pass(0));
    r.uw(1, 1, 2);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(1), int::pass(1));
    assert_eq!(r.ur(1, 2) & 0xf, 2);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(1), 0, "quieted by the UART alone, it falls");
}

/// Ports test_pio_fifo: a loopback burst fills the sixteen-byte FIFO and
/// comes back in order, and a receive FIFO reset drops what is held.
#[test]
fn programmed_io_goes_through_the_fifo() {
    let r = Rig::new();
    for n in 0..PORTS {
        r.uw(n, 3, 3);
        r.uw(n, 2, 1);
        r.uw(n, 4, 0x10);
        for i in 0..16 {
            r.uw(n, 0, 0x40 + i);
        }
        assert_eq!(r.ur(n, 5) & 1, 1);
        for i in 0..16 {
            assert_eq!(r.ur(n, 0), 0x40 + i);
        }
        assert_eq!(r.ur(n, 5) & 1, 0);
        r.uw(n, 0, 0xaa);
        assert_eq!(r.ur(n, 5) & 1, 1);
        r.uw(n, 2, 3);
        assert_eq!(r.ur(n, 5) & 1, 0, "a receive FIFO reset drops it");
    }
}

/// Ports test_uart_rejects_aligned_wide_write: a wider access to a UART
/// window is refused whole and reaches no register.
#[test]
fn uart_windows_take_bytes_only() {
    let r = Rig::new();
    r.uw(0, 4, 0x10);
    assert_eq!(r.ur(0, 4), 0x10);
    r.c.bar_write(0, reg::UART, 0x5a5a_5a5a, 4);
    assert_eq!(r.ur(0, 5) & 1, 0, "no character arrived");
    r.c.bar_write(0, reg::UART, 0x5a5a, 2);
    assert_eq!(r.ur(0, 5) & 1, 0);
    assert_eq!(r.c.bar_read(0, reg::UART, 4), 0);
    r.uw(0, 0, 0x5a);
    assert_eq!(r.ur(0, 5) & 1, 1);
    assert_eq!(r.ur(0, 0), 0x5a);
}

/// Ports test_ata_readback_rejects_width: the control registers take 32-bit
/// accesses and nothing narrower, and the words around the latch are not it.
#[test]
fn control_registers_take_words_only() {
    let r = Rig::with(Ioc4Config::tezro());
    r.w32(reg::ATA_READBACK, 0x1234_5678);
    assert_eq!(r.c.bar_read(0, reg::ATA_READBACK, 1), 0);
    assert_eq!(r.c.bar_read(0, reg::ATA_READBACK, 2), 0);
    r.c.bar_write(0, reg::ATA_READBACK, 0xaa, 1);
    r.c.bar_write(0, reg::ATA_READBACK, 0xbbcc, 2);
    assert_eq!(r.r32(reg::ATA_READBACK), 0x1234_5678);
    assert_eq!(r.r32(reg::ATA_READBACK - 4), 0);
    assert_eq!(r.r32(reg::ATA_READBACK + 4), 0);
    r.w32(reg::ATA_READBACK - 4, 0xaabb_ccdd);
    r.w32(reg::ATA_READBACK + 4, 0x5566_7788);
    assert_eq!(r.r32(reg::ATA_READBACK), 0x1234_5678);
}

/// Ports test_ata_readback_latch and test_ata_readback_default_unsupported:
/// every one-hot value and all ones read back, a reset clears the word, and
/// a chip whose board did not ask for it reads zero.
#[test]
fn ata_readback_is_a_latch_where_the_board_asks_for_it() {
    let r = Rig::with(Ioc4Config::tezro());
    let before = r.get(RINGS as usize, 8);
    for bit in 0..32 {
        r.w32(reg::ATA_READBACK, 1 << bit);
        assert_eq!(r.r32(reg::ATA_READBACK), 1 << bit);
    }
    r.w32(reg::ATA_READBACK, u32::MAX);
    assert_eq!(r.r32(reg::ATA_READBACK), u32::MAX);
    assert_eq!(r.get(RINGS as usize, 8), before, "no DMA behind it");
    assert!(!r.pin.level());
    r.c.reset();
    assert_eq!(r.r32(reg::ATA_READBACK), 0);

    let bare = Rig::new();
    bare.w32(reg::ATA_READBACK, u32::MAX);
    assert_eq!(bare.r32(reg::ATA_READBACK), 0);
}

/// Ports test_serial_idle_status_default_pio: with every transfer finishing
/// inside the access that starts it, the DMA status reads idle, read-only.
#[test]
fn serial_control_reads_idle() {
    let r = Rig::new();
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);
    r.w32(reg::SERIAL_CONTROL, 0);
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);
}

// ---------------------------------------------------------------------------
// Serial ring DMA
// ---------------------------------------------------------------------------

/// Ports test_dma_tx_control_entries: a modem-control slot writes the 16550's
/// register, a delay slot holds the walk for its milliseconds, and an
/// explicit-interrupt flag raises the port's completion bit.
#[test]
fn tx_control_entries_drive_the_uart() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.tx_entry(0, 0x1000, 0, [0x1f, 3, 0, 0], [TX_MCR | TX_EXPLICIT, TX_DELAY | TX_EXPLICIT, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, SHADOW_OUTPUTS, "the modem control reached the UART");
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::serial(0, 7), int::serial(0, 7), "the first slot's completion");
    r.w32(reg::SERIAL_PENDING, int::serial(0, 7));
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0, "the delay holds the entry");
    assert_eq!(r.sr(0, sdma_off::CONTROL) & (1 << 16), 1 << 16, "and reads busy");
    r.settle(10);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 8);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::serial(0, 7), int::serial(0, 7), "the delay's completion");
}

/// Ports test_dma_tx_wrap_1k: the consumer wraps at the 1 KiB ring's last
/// entry, and the empty source comes back with it.
#[test]
fn tx_ring_wraps_at_one_kilobyte() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::TX_PRODUCE, 0x3f8);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0x3f8, "empty entries are skipped");
    r.tx_entry(0, 0x400, 0x3f8, [0x5e, 0, 0, 0], [TX_VALID, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 0);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0, "wrapped");
    assert_eq!(r.sent(0), [0x5e]);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::serial(0, 0), int::serial(0, 0), "transmit ring empty");
}

/// Ports test_dma_tx_empty_follows_the_ring: transmit-ring-empty is a level
/// that is clear while an entry is in the engine and deaf to a write.
#[test]
fn tx_empty_follows_the_ring() {
    let r = Rig::new();
    let empty = int::serial(0, 0);
    r.pair_base(0, RINGS, 0x1000);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & empty, 0, "off with ring DMA off");
    r.sw(0, sdma_off::CONTROL, ENABLE);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & empty, empty);
    r.w32(reg::SERIAL_PENDING, empty);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & empty, empty);
    r.tx_entry(0, 0x1000, 0, [3, 0, 0, 0], [TX_DELAY, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & empty, 0);
    r.settle(10);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & empty, empty);
}

/// Ports test_dma_rx_drain_and_threshold: characters wait in the packer
/// until a drain posts them, with the line's modem state, and the high-water
/// source follows the entries waiting.
#[test]
fn rx_drain_posts_the_packer_and_raises_the_threshold() {
    let r = Rig::new();
    let high = int::serial(0, 2);
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    // A long reload, so the drain and not the timer posts the entry.
    r.sw(0, sdma_off::RX_TIMER, 0xfff);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, b"abc");
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE) & 0xff8, 0);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD | DRAIN);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE) & 0xff8, 8);
    let entry = r.get(r.ring(0, 1, 0x1000), 8);
    assert_eq!(entry, [b'a', b'b', b'c', 0, RX_FROM_LINE, RX_FROM_LINE, RX_FROM_LINE, 0]);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & high, high);
    r.w32(reg::SERIAL_PENDING, high);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & high, high, "a level");
    r.sw(0, sdma_off::RX_CONSUME, 8);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & high, 0, "the consumer made room");
}

/// Ports test_dma_rx_timer_expiry: the timer posts a lone character when
/// its reload runs out and raises the expiry and threshold sources.
#[test]
fn rx_timer_posts_a_lone_character() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    r.sw(0, sdma_off::RX_TIMER, 2);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, &[0x6d]);
    r.settle(10);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE) & 0xff8, 8);
    let entry = r.get(r.ring(0, 1, 0x1000), 8);
    assert_eq!((entry[0], entry[4]), (0x6d, RX_FROM_LINE));
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::serial(0, 3), int::serial(0, 3), "timer expired");
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::serial(0, 2), int::serial(0, 2), "threshold");
}

/// Ports test_dma_threshold_is_a_level.
#[test]
fn threshold_is_a_level() {
    let r = Rig::new();
    let high = int::serial(0, 2);
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, THRESHOLD);
    r.sw(0, sdma_off::RX_PRODUCE, 8);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & high, high);
    r.w32(reg::SERIAL_PENDING, high);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & high, high);
    r.sw(0, sdma_off::RX_CONSUME, 8);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & high, 0);
}

/// Ports test_dma_channel_reset_keeps_settings: a channel reset zeroes the
/// four indices and keeps the timer reload, the UART's registers and what
/// the write that reset it stored.
#[test]
fn channel_reset_keeps_settings() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::RX_TIMER, 0x140);
    r.uw(0, 4, 0x03);
    r.sw(0, sdma_off::TX_PRODUCE, 0x40);
    r.sw(0, sdma_off::CONTROL, RESET | 5);
    assert_eq!(r.sr(0, sdma_off::TX_PRODUCE), 0);
    assert_eq!(r.sr(0, sdma_off::RX_TIMER) & 0xfff, 0x140);
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, 0x0300_0000);
    assert_eq!(r.sr(0, sdma_off::CONTROL), 5);
}

/// Ports test_dma_full_reset_clears_shadow: the shadow mirrors the UART and
/// ignores a write, and a chip reset puts it back to the UART's reset
/// modem control.
#[test]
fn shadow_mirrors_the_uart() {
    let r = Rig::new();
    r.uw(0, 4, 0x13);
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, 0x1300_0000);
    r.sw(0, sdma_off::SHADOW, 0x1f00_0000);
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, 0x1300_0000);
    r.c.reset();
    let mcr = u32::from(r.ur(0, 4));
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, mcr << 24);
}

/// Ports test_dma_loopback_tx_mcr in miniature: with the UART looped back
/// the engine's transmitted characters return through the receive ring with
/// the looped modem state (DTR and RTS set, so RTS comes back as CTS), and
/// none reach the line.
#[test]
fn loopback_returns_through_the_rings() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.fast_line(0);
    r.uw(0, 4, 0x13);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    r.tx_entry(0, 0x400, 0, *b"wxyz", [TX_VALID; 4]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    r.until("the receive entry", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 8);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 8);
    assert_eq!(r.get(r.ring(0, 1, 0x400), 8), [b'w', b'x', b'y', b'z', RX_LOOPBACK, RX_LOOPBACK, RX_LOOPBACK, RX_LOOPBACK]);
    assert!(r.sent(0).is_empty(), "nothing reached the line");
}

/// A looped-back character takes a character time to reach the receiver, and
/// the walk waits for it: at 9600 baud the second character of an entry is
/// not sent for a millisecond. Ports the "still on the line" case of
/// test_dma_loopback_tx_mcr.
#[test]
fn a_looped_back_character_takes_a_character_time() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.uw(0, 4, 0x13);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    r.tx_entry(0, 0x400, 0, *b"abcd", [TX_VALID; 4]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert!(r.c.sh.st.lock().sdma.loop_deadline(0).is_some(), "the first character is on the line");
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0, "the entry is not done");
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8, 0, "and nothing has been posted");
    r.until("the entry", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 8);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 8);
}

/// Ports test_dma_loopback_pio_mcr's loopback half: a full 127-entry transmit
/// ring crosses the line and fills the receive ring, entry for entry.
#[test]
fn loopback_fills_the_receive_ring() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.fast_line(0);
    r.uw(0, 4, 0x13);
    assert_eq!(r.ur(0, 4), 0x13);
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, 0x1300_0000);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    let expected = fill_loopback_ring(&r);
    r.until("the transmit ring", |r| r.sr(0, sdma_off::TX_CONSUME) & 0x3f8 == 0x3f8);
    r.until("the receive ring", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 0x3f8);
    assert_eq!((r.sr(0, sdma_off::TX_PRODUCE), r.sr(0, sdma_off::TX_CONSUME)), (0x3f8, 0x3f8));
    assert_eq!(r.sr(0, sdma_off::RX_CONSUME), 0);
    assert_eq!(r.get(r.ring(0, 1, 0x400), 0x3f8), expected);
    assert!(r.sent(0).is_empty());
}

/// Ports test_dma_loopback_tx_mcr: a modem-control slot ahead of the
/// characters writes the UART's own register, and its characters then loop
/// back, entry by entry, the first holding three.
#[test]
fn a_modem_control_slot_leads_a_looped_back_run() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.fast_line(0);
    r.uw(0, 4, 0);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    let mut expected = vec![0u8; 0x3f8];
    let mut count = 0usize;
    for i in 0..127usize {
        let (data, control): ([u8; 4], [u8; 4]) = if i == 0 {
            ([0x13, 0, 1, 2], [TX_MCR, TX_VALID, TX_VALID, TX_VALID])
        } else {
            (std::array::from_fn(|j| (i * 4 + j - 1) as u8), [TX_VALID; 4])
        };
        for &byte in &data[usize::from(i == 0)..] {
            expected[count / 4 * 8 + count % 4] = byte;
            expected[count / 4 * 8 + 4 + count % 4] = RX_LOOPBACK;
            count += 1;
        }
        r.tx_entry(0, 0x400, i * 8, data, control);
    }
    assert_eq!(count, 507);
    r.sw(0, sdma_off::TX_PRODUCE, 0x3f8);
    r.until("the transmit ring", |r| r.sr(0, sdma_off::TX_CONSUME) & 0x3f8 == 0x3f8);
    r.until("the last full entry", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 0x3f0);
    assert_eq!(r.ur(0, 4), 0x13, "the modem-control slot wrote the UART's register");
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, 0x1300_0000);
    // Three characters wait in the packer for a drain.
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD | DRAIN);
    r.until("the drain", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 0x3f8);
    assert_eq!(r.get(r.ring(0, 1, 0x400), 0x3f8), expected);
    assert!(r.sent(0).is_empty());
}

/// Ports test_dma_loopback_shadow_reset: a write to the shadow changes
/// nothing, and a channel reset leaves the outputs as they were.
#[test]
fn the_shadow_ignores_writes_and_a_channel_reset() {
    let r = Rig::new();
    r.uw(0, 4, 0);
    r.sw(0, sdma_off::SHADOW, SHADOW_OUTPUTS);
    assert_eq!(r.ur(0, 4), 0);
    r.uw(0, 4, 0x13);
    r.sw(0, sdma_off::SHADOW, 0x1300_0000);
    assert_eq!(r.ur(0, 4), 0x13);
    r.sw(0, sdma_off::CONTROL, RESET);
    assert_eq!(r.sr(0, sdma_off::SHADOW) & SHADOW_OUTPUTS, 0x1300_0000);
    assert_eq!(r.ur(0, 4), 0x13);
}

/// A full 127-entry transmit ring of valid characters; the receive ring it
/// should fill.
fn fill_loopback_ring(r: &Rig) -> Vec<u8> {
    let mut expected = vec![0u8; 0x3f8];
    for i in 0..127usize {
        let data: [u8; 4] = std::array::from_fn(|j| (i * 4 + j) as u8);
        expected[i * 8..i * 8 + 4].copy_from_slice(&data);
        expected[i * 8 + 4..i * 8 + 8].fill(RX_LOOPBACK);
        r.tx_entry(0, 0x400, i * 8, data, [TX_VALID; 4]);
    }
    r.sw(0, sdma_off::TX_PRODUCE, 0x3f8);
    expected
}

/// Ports test_dma_requester_ids' use of both pairs: ports 2 and 3 take their
/// rings from the second pair's base, apart from the first pair's.
#[test]
fn second_pair_has_its_own_base() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.pair_base(1, RINGS + 0x2000, 0x400);
    assert_eq!(r.r32(reg::SDMA_BASE), RINGS as u32);
    assert_eq!(r.r32(reg::SDMA_BASE + 8), (RINGS + 0x2000) as u32);
    for p in [0usize, 2, 3] {
        r.sw(p, sdma_off::CONTROL, ENABLE);
    }
    let at = |pair: u64, p: u64| (RINGS + pair * 0x2000 + (p & 1) * 2 * 0x400) as usize;
    r.put(at(1, 2), &[b'c', 0, 0, 0, TX_VALID, 0, 0, 0]);
    r.put(at(1, 3), &[b'd', 0, 0, 0, TX_VALID, 0, 0, 0]);
    r.put(at(0, 0), &[b'a', 0, 0, 0, TX_VALID, 0, 0, 0]);
    r.sw(3, sdma_off::TX_PRODUCE, 8);
    r.sw(2, sdma_off::TX_PRODUCE, 8);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!((r.sent(0), r.sent(1), r.sent(2), r.sent(3)), (b"a".to_vec(), vec![], b"c".to_vec(), b"d".to_vec()));
}

/// Ports test_dma_error_capture_and_reset: the first failed transfer
/// captures its address, a second sets the multiple bit, the port's source
/// is in the other word, and a channel reset recovers the port but leaves
/// the capture and the source for software to acknowledge.
#[test]
fn a_failed_ring_transfer_is_captured_and_halts_the_port() {
    let r = Rig::new();
    let bad = BAD_BASE + (1 << 32);
    let low = BAD_BASE as u32 | ERROR_VALID;
    r.pair_base(0, bad, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0, "the fetch failed");
    assert_eq!(r.r32(reg::ERROR_LOW), low);
    assert_eq!(r.r32(reg::ERROR_HIGH), 1);
    assert_eq!(r.r32(reg::OTHER_PENDING) & int::dma_error(0), int::dma_error(0));

    r.sw(0, sdma_off::CONTROL, RESET);
    assert_eq!(r.r32(reg::ERROR_LOW), low, "a channel reset leaves the capture");
    assert_eq!(r.r32(reg::OTHER_PENDING) & int::dma_error(0), int::dma_error(0));
    r.w32(reg::OTHER_PENDING, int::dma_error(0));
    assert_eq!(r.r32(reg::OTHER_PENDING) & int::dma_error(0), 0);
    r.w32(reg::ERROR_LOW, 0);
    assert_eq!(r.r32(reg::ERROR_LOW), low, "the capture is read-only");

    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.type_in(0, &[0x73]);
    r.sw(0, sdma_off::CONTROL, ENABLE | DRAIN);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE) & 0xff8, 0, "the post failed");
    assert_eq!(r.r32(reg::ERROR_LOW), low | ERROR_MULTIPLE);
    assert_eq!(r.r32(reg::OTHER_PENDING) & int::dma_error(0), int::dma_error(0));
    r.c.reset();
    assert_eq!((r.r32(reg::ERROR_LOW), r.r32(reg::ERROR_HIGH), r.r32(reg::OTHER_PENDING)), (0, 0, 0));
}

/// Ports test_dma_requester_ids: a transmit fetch names twice the port and a
/// receive post one more, on every port.
#[test]
fn the_error_capture_names_the_requester() {
    for p in 0..PORTS {
        let r = Rig::new();
        r.pair_base(p as u32 / 2, BAD_BASE, 0x400);
        r.sw(p, sdma_off::CONTROL, ENABLE);
        r.sw(p, sdma_off::TX_PRODUCE, 8);
        assert_eq!(r.r32(reg::OTHER_PENDING), int::dma_error(p), "only this port's bit");
        assert_eq!(r.r32(reg::ERROR_LOW) >> 1 & 0xf, 2 * p as u32);
        assert_eq!(r.r32(reg::ERROR_LOW) & ERROR_VALID, ERROR_VALID);

        r.c.reset();
        r.pair_base(p as u32 / 2, BAD_BASE, 0x400);
        r.sw(p, sdma_off::CONTROL, ENABLE);
        r.type_in(p, &[1, 2, 3, 4]);
        assert_eq!(r.r32(reg::ERROR_LOW) >> 1 & 0xf, 2 * p as u32 + 1);
        assert_eq!(r.sr(p, sdma_off::RX_PRODUCE), 0);
    }
}

/// Ports test_dma_port_fault_isolation_and_recovery: a fault on one port
/// leaves its pair's other port running, and the faulted port runs again
/// once reset.
#[test]
fn a_fault_halts_only_its_own_port() {
    let r = Rig::new();
    r.pair_base(0, BAD_BASE, 0x400);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(1, sdma_off::CONTROL, ENABLE);
    r.sw(1, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.r32(reg::ERROR_LOW) >> 1 & 0xf, 2);
    assert_eq!(r.r32(reg::OTHER_PENDING), int::dma_error(1));

    // The base is shared, so quiet both ports before changing it.
    r.sw(0, sdma_off::CONTROL, 0);
    r.sw(1, sdma_off::CONTROL, RESET);
    assert_eq!((r.sr(1, sdma_off::TX_PRODUCE), r.sr(1, sdma_off::TX_CONSUME)), (0, 0));
    r.pair_base(0, RINGS, 0x400);
    assert_eq!(r.r32(reg::SDMA_BASE), RINGS as u32);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.tx_entry(0, 0x400, 0, [b'p', 0, 0, 0], [TX_VALID, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sent(0), b"p");
    assert_eq!(r.sr(1, sdma_off::TX_CONSUME), 0);

    r.sw(1, sdma_off::CONTROL, ENABLE);
    r.tx_entry(1, 0x400, 0, [0x15, 0, 0, 0], [TX_MCR, 0, 0, 0]);
    r.sw(1, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sr(1, sdma_off::TX_CONSUME), 8, "the reset port runs");
    assert_eq!(r.sr(1, sdma_off::SHADOW) & SHADOW_OUTPUTS, 0x1500_0000);
}

// ---------------------------------------------------------------------------
// The clock on byte-bus select 0
// ---------------------------------------------------------------------------

const CLOCK_CONTROL: u32 = reg::CLOCK + ds1744::CONTROL as u32;

/// A chip with a clock on a stepped nanosecond counter, seeded on
/// 2024-01-01.
fn with_clock() -> (Rig, Arc<AtomicI64>) {
    let r = Rig::new();
    let ns = Arc::new(AtomicI64::new(0));
    let n = ns.clone();
    let seed = 19_723 * 86_400; // 2024-01-01
    let clock = Ds1744::with_clock(seed, YearBase::Gregorian, Arc::new(move || n.load(AtomicOrdering::SeqCst)));
    r.c.attach_clock(Arc::new(clock.unwrap()));
    (r, ns)
}

/// Set the calendar through the write bit: seconds, minutes, hours, date,
/// month and year, then the century that commits it.
fn set_time(r: &Rig, [sec, min, hour, date, month, year]: [u8; 6], century: u8) {
    r.w8(CLOCK_CONTROL, 0x80);
    for (i, v) in [sec, min, hour, 0x02, date, month, year].into_iter().enumerate() {
        r.w8(CLOCK_CONTROL + 1 + i as u32, v);
    }
    r.w8(CLOCK_CONTROL, century);
}

/// Ports test_rtc_absent: with no clock fitted the window reads zero and
/// drops a write.
#[test]
fn an_unfitted_clock_reads_zero() {
    let r = Rig::new();
    assert_eq!(r.r8(CLOCK_CONTROL), 0);
    r.w8(CLOCK_CONTROL, 0x55);
    assert_eq!(r.r8(CLOCK_CONTROL), 0);
    for a in [reg::CLOCK + 0x7ff7, reg::CLOCK + 0x8000, reg::CLOCK] {
        assert_eq!(r.r8(a), 0);
    }
}

/// Ports test_rtc_alias and test_nvram_window: the SRAM fills the BAR from
/// 0x80000 to the calendar at 0x87ff8, the frozen calendar reads through it,
/// and neither side of the window answers.
#[test]
fn the_clock_window_holds_ram_and_the_calendar() {
    let (r, ns) = with_clock();
    let frozen = [0x60u8, 0x59, 0x59, 0x23, 0x82, 0x31, 0x12, 0x24];
    r.w8(reg::CLOCK, 0xa5);
    r.w8(reg::CLOCK + 0x7ff7, 0x5a);
    for (i, a) in [0x0000e, 0x004bc, 0x07fcb, 0x07fd0].into_iter().enumerate() {
        r.w8(reg::CLOCK + a, 0x31 + i as u8);
    }
    assert_eq!((r.r8(reg::CLOCK), r.r8(reg::CLOCK + 0x7ff7)), (0xa5, 0x5a));
    for (i, a) in [0x0000e, 0x004bc, 0x07fcb, 0x07fd0].into_iter().enumerate() {
        assert_eq!(r.r8(reg::CLOCK + a), 0x31 + i as u8);
    }

    set_time(&r, [0x59, 0x59, 0x23, 0x31, 0x12, 0x24], 0x20);
    r.w8(CLOCK_CONTROL, 0x40);
    ns.fetch_add(2_000_000_000, AtomicOrdering::SeqCst);
    for (i, want) in frozen.into_iter().enumerate() {
        assert_eq!(r.r8(CLOCK_CONTROL + i as u32), want, "calendar byte {i}");
    }
    assert!(!r.pin.level());
    r.w8(CLOCK_CONTROL, 0);
    assert_eq!((r.r8(CLOCK_CONTROL + 1), r.r8(CLOCK_CONTROL + 5), r.r8(CLOCK_CONTROL + 6), r.r8(CLOCK_CONTROL + 7)), (1, 1, 1, 0x25));

    // Past the part on either side nothing answers.
    assert_eq!(r.r8(reg::CLOCK + 0x8000), 0);
    assert_eq!(r.r8(reg::CLOCK + 0x8008), 0);
    assert_eq!(r.r8(reg::CLOCK - 1), 0);

    // A chip reset leaves the SRAM and the calendar alone.
    r.w8(CLOCK_CONTROL, 0x40);
    r.c.reset();
    assert_eq!((r.r8(reg::CLOCK), r.r8(reg::CLOCK + 0x7ff7)), (0xa5, 0x5a));
    assert_eq!(r.r8(CLOCK_CONTROL), 0x60, "even a read in progress");
}

/// Ports test_rtc_rejects_wide_access: a halfword or word to the clock is
/// refused whole and reaches no register.
#[test]
fn the_clock_takes_bytes_only() {
    let (r, _) = with_clock();
    set_time(&r, [0x12, 0x34, 0x05, 0x15, 0x01, 0x24], 0x20);
    r.w8(CLOCK_CONTROL, 0x80);
    r.w8(CLOCK_CONTROL + 1, 0x11);
    r.w8(CLOCK_CONTROL + 2, 0x22);
    r.c.bar_write(0, CLOCK_CONTROL + 2, 0x4433, 2);
    r.c.bar_write(0, CLOCK_CONTROL, 0x2000_0000, 4);
    assert_eq!(r.c.bar_read(0, CLOCK_CONTROL + 2, 2), 0);
    assert_eq!(r.c.bar_read(0, CLOCK_CONTROL, 4), 0);
    assert_eq!((r.r8(CLOCK_CONTROL + 2), r.r8(CLOCK_CONTROL + 3)), (0x22, 0x05));
    r.w8(CLOCK_CONTROL, 0x20);
    assert_eq!((r.r8(CLOCK_CONTROL + 2), r.r8(CLOCK_CONTROL + 3)), (0x22, 0x05));
}

// ---------------------------------------------------------------------------
// A receive ring with no room
// ---------------------------------------------------------------------------

const RX_FULL: u32 = int::serial(0, 1);

/// Ports test_dma_rx_full_and_staged_recovery: a character the full ring
/// cannot take waits, receive-ring-full is a level that stays while an
/// entry waits for room and ignores an acknowledge, and the entries post as
/// the consumer makes room. Filled exactly with nothing waiting, it stays
/// down.
#[test]
fn receive_ring_full_is_a_level_while_an_entry_waits() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    for i in 0..127u32 {
        r.type_in(0, &[0x31, 0x32, 0x33, 0x34]);
        r.until("the entry", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == ((i + 1) * 8) & 0x3f8);
    }
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE), 0x3f8);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & RX_FULL, 0, "full, but nothing waits for room");

    r.sw(0, sdma_off::RX_TIMER, 1);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, &[0xa1, 0xa2, 0xa3, 0xa4, 0xb5]);
    r.until("the source", |r| r.r32(reg::SERIAL_PENDING) & RX_FULL != 0);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE), 0x3f8);
    r.w32(reg::SERIAL_PENDING, RX_FULL);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & RX_FULL, RX_FULL, "a level ignores the acknowledge");

    r.sw(0, sdma_off::RX_CONSUME, 16 | ARM);
    r.until("room", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 0);
    let entry = r.get(r.ring(0, 1, 0x400) + 0x3f8, 8);
    assert_eq!((&entry[..4], entry[4], entry[7]), (&[0xa1, 0xa2, 0xa3, 0xa4][..], RX_FROM_LINE, RX_FROM_LINE));
    assert_eq!(r.r32(reg::SERIAL_PENDING) & RX_FULL, 0, "it fell once the consumer made room");

    r.until("the last character", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 8);
    let entry = r.get(r.ring(0, 1, 0x400), 8);
    assert_eq!((entry[0], entry[1], entry[4], entry[5]), (0xb5, 0, RX_FROM_LINE, 0));
    assert_eq!(r.r32(reg::SERIAL_PENDING) & RX_FULL, 0, "full again, but nothing waits for room");
}

/// Ports test_dma_loopback_full_ring_recovery: a full receive ring does not
/// stall the transmit walk in loopback. Both entries go out and the eight
/// characters wait in the packer and the 16550's FIFO, then land as the
/// consumer frees the ring.
#[test]
fn a_full_ring_does_not_stall_a_loopback_walk() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.fast_line(0);
    r.uw(0, 2, 0x01);
    r.uw(0, 4, 0x13);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    let expected = fill_loopback_ring(&r);
    r.until("the transmit ring", |r| r.sr(0, sdma_off::TX_CONSUME) & 0x3f8 == 0x3f8);
    r.until("the receive ring", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 0x3f8);
    assert_eq!(r.get(r.ring(0, 1, 0x400), 0x3f8), expected);

    let looped = |c: [u8; 4]| [c[0], c[1], c[2], c[3], RX_LOOPBACK, RX_LOOPBACK, RX_LOOPBACK, RX_LOOPBACK];
    r.tx_entry(0, 0x400, 0x3f8, [0xa1, 0xa2, 0xa3, 0xa4], [TX_VALID; 4]);
    r.tx_entry(0, 0x400, 0, [0xb1, 0xb2, 0xb3, 0xb4], [TX_VALID; 4]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    r.until("both entries sent", |r| r.sr(0, sdma_off::TX_CONSUME) & 0x3f8 == 8);
    r.until("the source", |r| r.r32(reg::SERIAL_PENDING) & RX_FULL != 0);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE), 0x3f8);

    r.sw(0, sdma_off::RX_CONSUME, 8);
    r.until("the first", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 0);
    assert_eq!(r.get(r.ring(0, 1, 0x400) + 0x3f8, 8), looped([0xa1, 0xa2, 0xa3, 0xa4]));
    r.sw(0, sdma_off::RX_CONSUME, 16);
    r.until("the second", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8 == 8);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & RX_FULL, 0, "filled exactly with nothing waiting");
    assert_eq!(r.get(r.ring(0, 1, 0x400), 8), looped([0xb1, 0xb2, 0xb3, 0xb4]));
    assert!(r.sent(0).is_empty());
}

/// Ports test_dma_loopback_overrun_reaches_status: with the receive ring
/// out of room the characters that do not fit are lost to the 16550's
/// overrun, which shows in the status byte of the next character read.
#[test]
fn a_loopback_overrun_reaches_the_status_byte() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x400);
    r.fast_line(0);
    r.uw(0, 2, 0x01);
    r.uw(0, 4, 0x13);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    // Room for one entry before the ring is full.
    r.sw(0, sdma_off::RX_CONSUME, 16);
    for i in 0..8usize {
        r.tx_entry(0, 0x400, 8 * i, std::array::from_fn(|j| (4 * i + j) as u8), [TX_VALID; 4]);
    }
    r.sw(0, sdma_off::TX_PRODUCE, 64);
    r.until("the walk", |r| r.sr(0, sdma_off::TX_CONSUME) & 0x3f8 == 64);
    r.settle(5);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8, 8);

    r.sw(0, sdma_off::RX_CONSUME, 8);
    r.sw(0, sdma_off::CONTROL, ENABLE | DRAIN);
    r.until("the drain", |r| r.sr(0, sdma_off::CONTROL) & DRAIN == 0);
    let posted = (r.sr(0, sdma_off::RX_PRODUCE) & 0x3f8) / 8;
    assert!(posted > 1);
    let overrun = (1..posted as usize).any(|i| r.get(r.ring(0, 1, 0x400) + 8 * i + 4, 4).iter().any(|s| s & 1 != 0));
    assert!(overrun, "the loss is marked on a later character");
    assert!(posted * 4 < 32, "and not everything came back");
}

/// Ports test_serial_idle_status: the serial control word reads the idle bit
/// set after a write, a pause, a reset, PIO and a ring run, and the writes to
/// it and to the unsupported ATA window disturb no interrupt word.
#[test]
fn serial_idle_status_survives_writes_and_ring_runs() {
    let r = Rig::new();
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);
    r.w32(reg::SERIAL_CONTROL, 0x80);
    r.c.bar_read(0, reg::SERIAL_CONTROL, 1);
    r.c.bar_write(0, reg::SERIAL_CONTROL, 0x80, 1);
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);
    assert_eq!(r.r32(0x100), 0);
    r.w32(0x100, u32::MAX);
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);

    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE | (1 << 29));
    r.tx_entry(0, 0x1000, 0, [0x5a, 0, 0, 0], [TX_VALID, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_ne!(r.sr(0, sdma_off::CONTROL) & (1 << 30), 0, "paused");
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);

    r.c.reset();
    r.uw(0, 0, 0x5a);
    assert_eq!(r.sent(0), [0x5a]);
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);

    r.pair_base(0, RINGS, 0x1000);
    r.tx_entry(1, 0x1000, 0, [0x15, 0, 0, 0], [TX_MCR | TX_EXPLICIT, 0, 0, 0]);
    r.sw(1, sdma_off::CONTROL, ENABLE);
    r.sw(1, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sr(1, sdma_off::TX_CONSUME), 8);
    let (serial, other) = (r.r32(reg::SERIAL_PENDING), r.r32(reg::OTHER_PENDING));
    assert_ne!(serial, 0);
    r.w32(reg::SERIAL_CONTROL, 0x80);
    r.w32(0x100, u32::MAX);
    assert_eq!((r.r32(reg::SERIAL_PENDING), r.r32(reg::OTHER_PENDING)), (serial, other));
    assert_eq!(r.r32(reg::SERIAL_CONTROL), 0x80);
}

// ---------------------------------------------------------------------------
// The host end of a port
// ---------------------------------------------------------------------------

/// Ports test_dma_loopback_discards_external_input: in loopback the UART's
/// serial input is disconnected, as on a PC16550D. Input from outside is
/// discarded on arrival, not kept for later, and new input arrives as usual
/// once the loopback is off.
#[test]
fn loopback_discards_input_from_outside() {
    let r = Rig::new();
    let input = [0xa1, 0xa2, 0xa3, 0xa4];
    r.pair_base(0, RINGS, 0x400);
    r.uw(0, 4, 0x10);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD);
    r.type_in(0, &input);
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD | DRAIN);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE), 0);

    // Out of loopback nothing comes back: it was discarded, not held.
    r.uw(0, 4, 0);
    r.c.service();
    r.sw(0, sdma_off::CONTROL, ENABLE | THRESHOLD | DRAIN);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE), 0);

    r.type_in(0, &input);
    r.until("the entry", |r| r.sr(0, sdma_off::RX_PRODUCE) == 8);
    assert_eq!(&r.get(r.ring(0, 1, 0x400), 4), &input);
}

/// Ports test_dma_chardev_reconnect: a port's backend can be replaced while
/// the machine runs, and both PIO and the ring engine then use the new one.
#[test]
fn replacing_the_backend_keeps_the_engine_working() {
    let r = Rig::new();
    let new = Arc::new(Wire::default());
    r.c.set_serial_backend(0, new.clone());
    r.c.set_serial_backend(0, new.clone());
    r.uw(0, 0, b'p');
    assert_eq!(*new.out.lock().unwrap(), b"p");
    assert!(r.sent(0).is_empty(), "the old backend saw nothing");

    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.tx_entry(0, 0x1000, 0, [0x61, 0x62, 0, 0], [TX_VALID, TX_VALID, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 8);
    assert_eq!(*new.out.lock().unwrap(), b"pab");
    new.inp.lock().unwrap().push_back(0x7a);
    r.c.service();
    r.sw(0, sdma_off::CONTROL, ENABLE | DRAIN);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE) & 0xff8, 8, "and its input arrives");
    assert_eq!(r.get(r.ring(0, 1, 0x1000), 1), [0x7a]);
}

// ---------------------------------------------------------------------------
// The rest of the ring engine's register cases
// ---------------------------------------------------------------------------

const TIMER_EXPIRED: u32 = int::serial(0, 3);
const TX_COMPLETION: u32 = int::serial(0, 7);

/// Ports test_dma_tx_wrap_4k.
#[test]
fn tx_ring_wraps_at_four_kilobytes() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::TX_PRODUCE, 0xff8);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0xff8);
    r.tx_entry(0, 0x1000, 0xff8, [0x6e, 0, 0, 0], [TX_VALID, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 0);
    assert_eq!(r.sent(0), [0x6e]);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::serial(0, 0), int::serial(0, 0));
}

/// Ports test_dma_control_stores_its_bits: the control register keeps every
/// writable bit as written, including ones the model does not act on.
#[test]
fn control_stores_the_bits_it_does_not_act_on() {
    let r = Rig::new();
    let written = (1 << 17) | (1 << 20) | 0x1ff;
    r.sw(0, sdma_off::CONTROL, written);
    assert_eq!(r.sr(0, sdma_off::CONTROL), written);
}

/// Ports test_dma_engine_indices_take_writes and
/// test_dma_host_index_masks_its_pointer: the two indices the engine moves
/// still take a write, and every index keeps only its pointer field.
#[test]
fn indices_keep_their_pointer_field() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::TX_CONSUME, u32::MAX);
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 0xff8);
    r.sw(0, sdma_off::RX_PRODUCE, 0x18);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE), 0x18);
    r.sw(0, sdma_off::TX_PRODUCE, 0x10014);
    assert_eq!(r.sr(0, sdma_off::TX_PRODUCE), 0x10);
    r.sw(0, sdma_off::RX_CONSUME, 0x2c);
    assert_eq!(r.sr(0, sdma_off::RX_CONSUME) & 0xff8, 0x28);
}

/// Ports test_dma_empty_slot_ends_the_entry: an empty slot ends its entry
/// wherever it sits, so the slots behind it are not sent.
#[test]
fn an_empty_slot_ends_the_entry() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.tx_entry(0, 0x1000, 0, *b"abcd", [TX_VALID, 0, TX_VALID, TX_VALID]);
    r.tx_entry(0, 0x1000, 8, [b'e', 0, 0, 0], [TX_VALID, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 16);
    assert_eq!(r.sent(0), b"ae");
}

/// Ports test_dma_tx_control_decodes_type_and_bit5: a control byte is
/// decoded by its type field and bit 5, undefined low bits do not stop the
/// port, and a slot of type none raises nothing even with bit 5 set.
#[test]
fn a_control_byte_is_decoded_by_type_and_bit_five() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.tx_entry(0, 0x1000, 0, [b'a', b'b', b'c', 0], [TX_VALID | 0x10, TX_EXPLICIT, TX_VALID, 0]);
    r.tx_entry(0, 0x1000, 8, [b'd', 0, 0, 0], [TX_VALID, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 16);
    assert_eq!(r.sent(0), b"ad");
    assert_eq!(r.sr(0, sdma_off::TX_CONSUME), 16);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & TX_COMPLETION, 0);
}

/// Ports test_dma_tx_delay_sets_the_busy_bit: control bit 16 reads set while
/// a delay entry runs and clear once it has ended.
#[test]
fn a_delay_entry_sets_the_busy_bit() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.tx_entry(0, 0x1000, 0, [3, 0, 0, 0], [TX_DELAY, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sr(0, sdma_off::CONTROL) & (1 << 16), 1 << 16);
    r.settle(10);
    assert_eq!(r.sr(0, sdma_off::CONTROL) & (1 << 16), 0);
}

/// Ports test_dma_rx_timer_due_before_access: an expiry already due
/// happens before a register access, with no timer thread to run it.
#[test]
fn a_due_receive_timer_expires_before_an_access() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::RX_TIMER, 2);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, &[0x64]);
    std::thread::sleep(Duration::from_millis(5));
    assert_eq!(r.r32(reg::SERIAL_PENDING) & TIMER_EXPIRED, TIMER_EXPIRED);
}

/// Ports test_dma_rx_timer_count_reads_back: a reload write loads the live
/// count too and an idle timer holds it; a running one counts down.
#[test]
fn the_receive_timer_count_reads_back() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::RX_TIMER, 0xfff);
    assert_eq!(r.sr(0, sdma_off::RX_TIMER) & 0x0fff_0fff, 0x0fff_0fff);
    std::thread::sleep(Duration::from_millis(2));
    assert_eq!(r.sr(0, sdma_off::RX_TIMER) & 0x0fff_0fff, 0x0fff_0fff, "idle, it holds");
    r.sw(0, sdma_off::RX_TIMER, 0x140);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, &[0x63]);
    let count = r.sr(0, sdma_off::RX_TIMER) >> 16 & 0xfff;
    assert!((1..=0x140).contains(&count), "running: {count:#x} of 0x140 left");
}

/// Ports test_dma_consume_without_arm_keeps_it: a consume write with bit 31
/// clear leaves the arm, which reads back and still starts the timer.
#[test]
fn a_consume_write_without_the_arm_keeps_it() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::RX_TIMER, 2);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.sw(0, sdma_off::RX_CONSUME, 0);
    assert_eq!(r.sr(0, sdma_off::RX_CONSUME), ARM);
    r.type_in(0, &[0x6b]);
    r.until("the expiry", |r| r.r32(reg::SERIAL_PENDING) & TIMER_EXPIRED != 0);
    assert_eq!(r.sr(0, sdma_off::RX_CONSUME), 0, "and the expiry spent it");
}

/// Ports test_dma_zero_reload_expires_at_once.
#[test]
fn a_zero_reload_expires_at_once() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::RX_TIMER, 0);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, &[0x7a]);
    r.until("the expiry", |r| r.r32(reg::SERIAL_PENDING) & TIMER_EXPIRED != 0);
    assert_eq!(r.sr(0, sdma_off::RX_PRODUCE), 8);
}

/// Ports test_dma_rx_status_carries_modem_state: a character from the line
/// carries the live CTS and DCD levels, as the UART's modem status shows.
#[test]
fn a_received_character_carries_the_modem_state() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.type_in(0, &[0x6e]);
    r.sw(0, sdma_off::CONTROL, ENABLE | DRAIN);
    let entry = r.get(r.ring(0, 1, 0x1000), 8);
    assert_eq!((entry[0], entry[4]), (0x6e, RX_FROM_LINE));
    assert_eq!(r.ur(0, 6) & 0xb0, 0xb0);
}

/// Ports test_dma_rx_timer_expiry_after_full_entry: a full entry posts at
/// once, the timer still raises its source when it runs out, and once
/// acknowledged it does not come back without another arm.
#[test]
fn the_timer_still_expires_after_a_full_entry() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::RX_TIMER, 2);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, &[0x31, 0x32, 0x33, 0x34]);
    r.until("the entry", |r| r.sr(0, sdma_off::RX_PRODUCE) & 0xff8 == 8);
    let entry = r.get(r.ring(0, 1, 0x1000), 8);
    assert_eq!((&entry[..4], entry[4], entry[7]), (&[0x31, 0x32, 0x33, 0x34][..], RX_FROM_LINE, RX_FROM_LINE));
    r.until("the expiry", |r| r.r32(reg::SERIAL_PENDING) & TIMER_EXPIRED != 0);
    r.w32(reg::SERIAL_PENDING, TIMER_EXPIRED);
    r.settle(5);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & TIMER_EXPIRED, 0);
}

/// Ports test_dma_rx_disable_stops_timer_keeps_arm: clearing the enable stops
/// a running receive timer and discards its time, and the arm stays so the
/// first character after re-enabling starts it afresh.
#[test]
fn disabling_the_ring_stops_the_timer_and_keeps_the_arm() {
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.sw(0, sdma_off::RX_TIMER, 0x100);
    r.sw(0, sdma_off::RX_CONSUME, ARM);
    r.type_in(0, &[0x31]);
    assert!(r.c.sh.st.lock().sdma.rx_deadline(0).is_some(), "running");
    r.sw(0, sdma_off::CONTROL, 0);
    assert!(r.c.sh.st.lock().sdma.rx_deadline(0).is_none(), "stopped");
    r.settle(20);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & TIMER_EXPIRED, 0);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    assert_eq!(r.sr(0, sdma_off::RX_CONSUME) & ARM, ARM, "the arm stays");
    r.type_in(0, &[0x32]);
    assert!(r.c.sh.st.lock().sdma.rx_deadline(0).is_some(), "the next character starts it afresh");
}

// ---------------------------------------------------------------------------
// The PS/2 keyboard and mouse controller
// ---------------------------------------------------------------------------

const KBC_CONTROL: u32 = reg::KBC;
const KBC_KBD_RECEIVE: u32 = reg::KBC + 0x04;
const KBC_AUX_RECEIVE: u32 = reg::KBC + 0x08;
const KBC_KBD_TRANSMIT: u32 = reg::KBC + 0x0c;
const KBC_AUX_TRANSMIT: u32 = reg::KBC + 0x10;
const KBC_IRQ: u32 = 0x40;
const KBC_WRITE_PENDING: u32 = 0x03;

fn key(r: &Rig, code: winit::keyboard::KeyCode, down: bool) {
    r.c.ps2().push_kb(code, down);
}

/// Ports test_ps2_adapter (QEMU with the read-only status
/// bits): the five registers through the adapter, reply bytes in BE32
/// presentation, write-pending never set, the keyboard source in the other
/// word with its own enable and acknowledge, isolation from the serial
/// words, a mouse report, and reset.
#[test]
fn the_ps2_adapter_presents_the_controller() {
    use winit::keyboard::KeyCode::KeyA;
    let r = Rig::new();
    assert_eq!((r.r32(KBC_CONTROL), r.r32(KBC_KBD_RECEIVE), r.r32(KBC_AUX_RECEIVE)), (0, 0, 0));
    r.w32(KBC_CONTROL, u32::MAX);
    assert_eq!(r.r32(KBC_CONTROL), 0x00fc_3cc0, "the status bits read clear and drop the write");
    r.w32(KBC_CONTROL, 0);
    r.c.bar_read(0, KBC_CONTROL, 1);
    r.c.bar_write(0, KBC_KBD_TRANSMIT, 0xff, 2);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0, "a narrow write reached nothing");

    // 0xff000000 is a zero byte to the keyboard: it only acknowledges.
    r.w32(KBC_KBD_TRANSMIT, 0xff00_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x80fa_0000);
    r.w32(KBC_KBD_TRANSMIT, 0xff);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0xc0fa_aa00);
    assert_eq!(r.r32(KBC_CONTROL) & KBC_WRITE_PENDING, 0);
    // QEMU's keyboard scans again after the reset command; this core waits
    // for the driver's enable, as the O2's and Octane's PROMs give it.
    r.w32(KBC_KBD_TRANSMIT, 0xf4);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x80fa_0000);
    r.w32(reg::OTHER_PENDING, KBC_IRQ);

    key(&r, KeyA, true);
    assert_eq!(r.r32(reg::OTHER_PENDING) & KBC_IRQ, KBC_IRQ);
    assert!(!r.pin.level(), "pending but not enabled");
    r.w32(reg::OTHER_ENABLE_SET, KBC_IRQ);
    assert!(r.pin.level());
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x801c_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE) & 0x1f00_0000, 0);
    assert_eq!(r.r32(reg::OTHER_PENDING) & KBC_IRQ, KBC_IRQ, "a drain does not acknowledge");

    // A serial source holds the pin up when the keyboard's goes.
    r.uw(0, 3, 0x80);
    r.uw(0, 0, 0x34);
    r.uw(0, 1, 0x12);
    r.uw(0, 3, 0x03);
    r.uw(0, 2, 0x01);
    r.uw(0, 1, 0x02);
    r.w32(reg::SERIAL_ENABLE_SET, int::pass(0));
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(0), int::pass(0));
    assert!(r.pin.level());
    r.w32(reg::OTHER_ENABLE_CLR, KBC_IRQ);
    assert!(r.pin.level());
    r.w32(reg::OTHER_ENABLE_SET, KBC_IRQ);
    r.w32(reg::OTHER_PENDING, KBC_IRQ);
    assert_eq!(r.r32(reg::OTHER_PENDING) & KBC_IRQ, 0);
    assert_eq!(r.r32(reg::SERIAL_PENDING) & int::pass(0), int::pass(0), "the serial source is another word");
    assert!(r.pin.level());
    r.uw(0, 1, 0);
    assert!(!r.pin.level());

    r.w32(KBC_AUX_TRANSMIT, 0xff);
    assert_eq!(r.r32(KBC_AUX_RECEIVE), 0xe0fa_aa00);
    assert_eq!(r.r32(KBC_CONTROL) & KBC_WRITE_PENDING, 0);
    r.w32(reg::OTHER_PENDING, KBC_IRQ);
    r.w32(KBC_AUX_TRANSMIT, 0xf4);
    assert_eq!(r.r32(KBC_AUX_RECEIVE), 0x80fa_0000);
    r.w32(reg::OTHER_PENDING, KBC_IRQ);
    r.c.ps2().push_mouse_input(1, 4, 0, 0);
    assert_eq!(r.r32(KBC_AUX_RECEIVE), 0xe009_0400);

    r.c.reset();
    assert_eq!((r.r32(KBC_CONTROL), r.r32(KBC_KBD_RECEIVE), r.r32(KBC_AUX_RECEIVE)), (0, 0, 0));
    assert_eq!((r.r32(reg::OTHER_PENDING) & KBC_IRQ, r.r32(reg::OTHER_ENABLE_SET) & KBC_IRQ), (0, 0));
    r.w32(KBC_KBD_TRANSMIT, 0xff);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0xc0fa_aa00);
}

/// Ports test_ps2_ports_present: every IOC4 has both ports; there is no
/// switch to leave them unplugged.
#[test]
fn both_ps2_ports_are_always_fitted() {
    let r = Rig::with(Ioc4Config::bare(Ioc4Clock::Mhz33));
    r.w32(KBC_KBD_TRANSMIT, 0xff);
    r.w32(KBC_AUX_TRANSMIT, 0xff);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0xc0fa_aa00);
    assert_eq!(r.r32(KBC_AUX_RECEIVE), 0xe0fa_aa00);
}

/// Ports test_ps2_driver_sequence, the attach and interrupt sequence of
/// OpenBSD's public IOC4 keyboard driver: read-modify-write the control
/// register to clamp after one character, transmit after the write-pending
/// poll, then on each interrupt mask the source, read each receive register
/// until no valid flag is left, acknowledge and unmask. Also the
/// clamp-after-three delivery, and an unclamped port losing a fourth
/// character without showing the IOC3-only overflow flag.
#[test]
fn the_ps2_driver_sequence() {
    use winit::keyboard::KeyCode::{KeyA, KeyB, KeyC, KeyD};
    let r = Rig::new();
    let mut csr = r.r32(KBC_CONTROL);
    csr &= !0x00c0_0ccc;
    csr |= 0x0030_0000;
    r.w32(KBC_CONTROL, csr);
    assert_eq!(r.r32(KBC_CONTROL), 0x0030_0000);
    // Status bits are not storage: a write to them is dropped and logged.
    r.w32(KBC_CONTROL, 0x0030_0000 | 0xc33f);
    assert_eq!(r.r32(KBC_CONTROL), 0x0030_0000);
    r.w32(KBC_CONTROL, 0x0030_0000 | 0x3cc0);
    assert_eq!(r.r32(KBC_CONTROL), 0x0030_3cc0, "the writable bits among them stick");
    r.w32(KBC_CONTROL, 0x0030_0000);
    r.w32(reg::OTHER_ENABLE_SET, KBC_IRQ);

    // A reset reply is two characters, so the clamp holds the second back.
    assert_eq!(r.r32(KBC_CONTROL) & KBC_WRITE_PENDING, 0);
    r.w32(KBC_KBD_TRANSMIT, 0xff);
    assert!(r.pin.level());
    assert_eq!(r.r32(reg::OTHER_PENDING) & KBC_IRQ, KBC_IRQ);
    r.w32(reg::OTHER_ENABLE_CLR, KBC_IRQ);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x80fa_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x80aa_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0);
    assert_eq!(r.r32(KBC_AUX_RECEIVE), 0);
    // As in the adapter test: this keyboard model needs the driver's enable.
    r.w32(KBC_KBD_TRANSMIT, 0xf4);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x80fa_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0);
    r.w32(reg::OTHER_PENDING, KBC_IRQ);
    r.w32(reg::OTHER_ENABLE_SET, KBC_IRQ);
    assert!(!r.pin.level());

    // Press and release: three characters, one per read under the clamp.
    key(&r, KeyA, true);
    key(&r, KeyA, false);
    assert!(r.pin.level());
    r.w32(reg::OTHER_ENABLE_CLR, KBC_IRQ);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x801c_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x80f0_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0x801c_0000);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0);
    r.w32(reg::OTHER_PENDING, KBC_IRQ);
    r.w32(reg::OTHER_ENABLE_SET, KBC_IRQ);
    assert!(!r.pin.level());

    // Clamp after three: the same press and release arrive in one read.
    r.w32(KBC_CONTROL, 0x0040_0000);
    key(&r, KeyA, true);
    key(&r, KeyA, false);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0xe01c_f01c);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0);

    // No clamp: a fourth character is lost and the flag is not reported.
    r.w32(KBC_CONTROL, 0);
    for k in [KeyA, KeyB, KeyC, KeyD] {
        key(&r, k, true);
    }
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0xe01c_3221);
    assert_eq!(r.r32(KBC_KBD_RECEIVE), 0);
}

/// Ports the width rule of test_ps2_adapter: the keyboard block takes
/// aligned 32-bit accesses only, and a refused one reaches no register.
#[test]
fn the_ps2_block_takes_aligned_words_only() {
    let r = Rig::new();
    r.w32(KBC_CONTROL, 0x0030_0000);
    assert_eq!(r.c.bar_read(0, KBC_CONTROL, 1), 0);
    assert_eq!(r.c.bar_read(0, KBC_CONTROL, 2), 0);
    r.c.bar_write(0, KBC_CONTROL, 0, 2);
    r.c.bar_write(0, KBC_CONTROL + 1, 0, 1);
    assert_eq!(r.r32(KBC_CONTROL), 0x0030_0000);
    assert_eq!(r.c.bar_read(0, KBC_CONTROL + 2, 4), 0, "unaligned");
    // Beyond the five registers is not the controller.
    assert_eq!(r.r32(reg::KBC + 0x14), 0);
}

/// A key pressed on the host reaches the pin with no guest access, from the
/// host's thread, as the IOC3's does.
#[test]
fn a_host_key_raises_the_pin_without_a_guest_access() {
    let r = Rig::new();
    r.w32(reg::OTHER_ENABLE_SET, KBC_IRQ);
    r.w32(KBC_KBD_TRANSMIT, 0xf4);
    let _ = r.r32(KBC_KBD_RECEIVE);
    r.w32(reg::OTHER_PENDING, KBC_IRQ);
    assert!(!r.pin.level());
    key(&r, winit::keyboard::KeyCode::KeyA, true);
    assert!(r.pin.level());
}

// ---------------------------------------------------------------------------
// Reported, not refused
// ---------------------------------------------------------------------------

/// Everything the `ioc` log module has said in this process. Tests run side
/// by side and share it, so a check looks for its own message text.
fn device_log() -> String {
    use std::io::Write;
    struct Sink(Arc<std::sync::Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    static LOG: std::sync::OnceLock<Arc<std::sync::Mutex<Vec<u8>>>> = std::sync::OnceLock::new();
    let log = LOG.get_or_init(|| {
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let dl = crate::devlog::DEVLOG.get().cloned().unwrap_or_else(crate::devlog::init_devlog);
        dl.add_sink(Arc::new(parking_lot::Mutex::new(Sink(buf.clone()))));
        dl.enable(crate::devlog::LogModule::Ioc);
        buf
    });
    String::from_utf8_lossy(&log.lock().unwrap()).into_owned()
}

/// Ports test_dma_reports_bad_base_and_pointer: a misaligned pair base and a
/// base change on a busy pair are reported, not refused; the enable and the
/// new base both take effect, and an index keeps its pointer field while the
/// stray bits are reported.
#[test]
fn a_bad_base_and_a_stray_pointer_bit_are_reported_not_refused() {
    let _ = device_log();
    let r = Rig::new();
    r.pair_base(0, RINGS + 8, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    assert_eq!(r.sr(0, sdma_off::CONTROL) & ENABLE, ENABLE);
    r.sw(0, sdma_off::CONTROL, 0);
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.pair_base(0, RINGS + 0x4000, 0x1000);
    assert_eq!(r.r32(reg::SDMA_BASE), (RINGS + 0x4000) as u32 | 1);
    r.sw(0, sdma_off::CONTROL, RESET);
    r.sw(0, sdma_off::TX_PRODUCE, 4);
    assert_eq!(r.sr(0, sdma_off::TX_PRODUCE), 0);
    r.sw(0, sdma_off::RX_TIMER, 0x1_0002);
    r.sw(0, sdma_off::CONTROL, 1 << 20);
    let log = device_log();
    for what in [
        "misaligned for its rings",
        "changed while it is active",
        "TX producer 0x00000004 on port 0 has bits outside the pointer field",
        "unsupported RX timer bits 0x00010000 on port 0",
        "serial control bits 0x00100000 on port 0 are stored but not modelled",
    ] {
        assert!(log.contains(what), "no report of: {what}");
    }
}

/// Ports test_dma_mode_switch_over_pio_is_reported and
/// test_dma_tx_disable_reports_pending_delay: the UART orders PIO and ring
/// DMA itself, so enabling DMA over queued PIO data, a PIO transmit while DMA
/// is on, and clearing the enable over a running delay are reported and go
/// ahead.
#[test]
fn a_mode_switch_over_pio_is_reported_and_goes_ahead() {
    let _ = device_log();
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.uw(0, 2, 0x07);
    r.type_in(0, &[0x5a]);
    assert_eq!(r.ur(0, 5) & 1, 1);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    assert_eq!(r.sr(0, sdma_off::CONTROL) & ENABLE, ENABLE, "the enable took effect");
    r.uw(0, 0, 0xa5);
    assert_eq!(r.sent(0), [0xa5], "and so did the PIO byte");

    r.tx_entry(0, 0x1000, 0, [30, 0, 0, 0], [TX_DELAY, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    r.sw(0, sdma_off::CONTROL, 0);
    assert_eq!(r.sr(0, sdma_off::CONTROL) & ENABLE, 0);
    let log = device_log();
    assert!(log.contains("with PIO data in the UART's FIFOs"));
    assert!(log.contains("PIO transmit on port 0 while DMA is enabled"));
    assert!(log.contains("DMA disabled on port 0 with ring work pending"));
}

/// Ports the undefined-bits check of test_dma_tx_control_decodes_type_and_bit5.
#[test]
fn undefined_transmit_control_bits_are_reported() {
    let _ = device_log();
    let r = Rig::new();
    r.pair_base(0, RINGS, 0x1000);
    r.sw(0, sdma_off::CONTROL, ENABLE);
    r.tx_entry(0, 0x1000, 0, [b'q', 0, 0, 0], [TX_VALID | 0x10, 0, 0, 0]);
    r.sw(0, sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.sent(0), b"q", "the slot is obeyed by its type");
    assert!(device_log().contains("undefined TX control bits 0x10 in an entry on port 0"));
}
