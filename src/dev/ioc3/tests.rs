//! IOC3 register-contract tests.
//!
//! Accesses go through `PciFunction` the way a host bridge makes them: a
//! 32-bit access at its offset, a byte access at `offset ^ 3`.
//!
//! Ported from tests/qtest/octane-test.c in jamesbraid/qemu-sgi-origami.

use super::*;
use super::kbd::*;
use super::parallel::*;
use super::serial::*;
use super::superio::*;
use std::collections::VecDeque;
use std::sync::Mutex;
use crate::dev::ds1685::{Ds1685, NVRAM_OFF_NV};
use crate::pci::PciDmaStatus;
use crate::irq::LevelProbe;
use crate::traits::MachineEvent;
use crate::pci::PciFunction;
use crate::dev::ng1::rex3::Rex3;
use std::sync::atomic::{AtomicU64, Ordering};

const MAC: [u8; 6] = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];

impl Ioc3 {
    fn predivider(&self, p: usize) -> u8 {
        self.sh.sio.as_ref().unwrap().uart(p, |u| u.predivider)
    }

    fn baudbase(&self, p: usize) -> u32 {
        self.sh.sio.as_ref().unwrap().uart(p, |u| u.core.baudbase())
    }

    fn rtc(&self) -> &Arc<Ds1685> {
        self.sh.rtc.as_ref().unwrap()
    }

    fn rx_pending(&self, p: usize) -> usize {
        self.sh.sio.as_ref().unwrap().uart(p, |u| u.core.rx_pending())
    }
}

/// A sink for a serial line: what the guest sent, and what the host types.
#[derive(Default)]
struct Wire {
    out: Mutex<Vec<u8>>,
    inp: Mutex<VecDeque<u8>>,
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

/// Bus memory, as bytes at bus addresses.
struct TestMem(Mutex<Vec<u8>>);

impl TestMem {
    fn new() -> Arc<TestMem> {
        Arc::new(TestMem(Mutex::new(vec![0; 0x10000])))
    }
}

impl PciDma for TestMem {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        let a = addr as usize;
        if addr > u32::MAX as u64 || a + buf.len() > m.len() {
            buf.fill(0);
            return PciDmaStatus::MasterAbort;
        }
        buf.copy_from_slice(&m[a..a + buf.len()]);
        PciDmaStatus::Ok
    }
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        let a = addr as usize;
        if addr > u32::MAX as u64 || a + data.len() > m.len() {
            return PciDmaStatus::MasterAbort;
        }
        m[a..a + data.len()].copy_from_slice(data);
        PciDmaStatus::Ok
    }
}

/// The Octane's IOC3, as the machine builds it but for the records.
fn octane() -> Ioc3 {
    Ioc3::new(&IP30, MAC, ByteBus::Ds1685(Ds1685::new(None)))
}

struct Rig {
    c: Ioc3,
    mem: Arc<TestMem>,
    wire: [Arc<Wire>; 2],
    superio: Arc<LevelProbe>,
}

impl Rig {
    fn new() -> Rig {
        Rig::with(octane())
    }

    fn with(c: Ioc3) -> Rig {
        let mem = TestMem::new();
        c.state().connect_dma(mem.clone());
        c.attached();
        let wire = [Arc::new(Wire::default()), Arc::new(Wire::default())];
        c.set_serial_backend(0, wire[0].clone());
        c.set_serial_backend(1, wire[1].clone());
        let (superio, line) = LevelProbe::line();
        c.connect_superio_interrupt(line);
        Rig { c, mem, wire, superio }
    }

    fn r32(&self, a: u32) -> u32 {
        self.c.bar_read(0, a, 4)
    }
    fn w32(&self, a: u32, v: u32) {
        self.c.bar_write(0, a, v, 4)
    }
    /// A byte access as the bridge delivers it.
    fn r8(&self, a: u32) -> u8 {
        self.c.bar_read(0, a ^ 3, 1) as u8
    }
    fn w8(&self, a: u32, v: u8) {
        self.c.bar_write(0, a ^ 3, v as u32, 1)
    }

    /// The firmware's chip-control value: ports at 0x178/0x170, strobe 4.
    fn aim_engines(&self) {
        self.w32(reg::CHIP_CONTROL, 0x0002_2e5e);
    }

    /// DMA the way the chip sees memory: internal bytes.
    fn mem_view(&self) -> Ioc3Dma {
        Ioc3Dma::new(self.mem.clone())
    }
}

// ---------------------------------------------------------------------------
// PCI function
// ---------------------------------------------------------------------------

#[test]
fn identity_and_bar() {
    let r = Rig::new();
    let cfg = r.c.state().config();
    assert_eq!(cfg.read(0), 0x0003_10a9);
    assert_eq!(cfg.read(2) >> 8, 0xff_0000, "no defined class");
    assert_eq!(cfg.read(2) & 0xff, 1, "revision 1");
    assert_eq!(cfg.read(11), 0xc304_10a9, "the Octane system board");
    assert_eq!(cfg.interrupt_pin(), PCI_INTA);
    cfg.write(4, 0xffff_ffff);
    assert_eq!(cfg.read(4), 0xfff0_0000, "one megabyte of memory space");
}

/// A row of the board table builds a chip with the identity, the children
/// and the GPIO straps the row names, and nothing else.
#[test]
fn each_board_reads_back_as_its_row_says() {
    for board in [&IP30, &IP29, &BASEIO, &BASEIO6G, &IP34, &IP34_NO_PS2, &MIO] {
        let bus = match board.cells.byte_bus {
            ByteBusPart::None => ByteBus::None,
            ByteBusPart::Ds1685 => ByteBus::Ds1685(Ds1685::new(None)),
            ByteBusPart::M48t35 => {
                ByteBus::M48t35(Arc::new(crate::dev::timekeeper::Timekeeper::at(2026, 1, 1, 0, 0, 0).unwrap()))
            }
        };
        let r = Rig::with(Ioc3::new(board, MAC, bus));
        let name = board.name;
        assert_eq!(r.c.state().config().read(11), (board.subsystem_id as u32) << 16 | VENDOR_ID as u32, "{name}");
        // A child that is fitted keeps what is written to it; one that is
        // not reads zero.
        let keeps = |a: u32, v: u32, mask: u32| {
            r.w32(a, v);
            r.r32(a) & mask == v & mask
        };
        assert_eq!(r.c.sh.serial.iter().all(|s| s.is_some()), board.cells.serial, "{name}: serial");
        r.w8(reg::SERIAL_A + 7, 0x5a);
        assert_eq!(r.r8(reg::SERIAL_A + 7) == 0x5a, board.cells.serial, "{name}: UART");
        assert_eq!(keeps(0x9c, 0x1234, 0xffff), board.cells.kbd, "{name}: keyboard");
        assert_eq!(keeps(0x80, 0x55, u32::MAX), board.cells.parallel, "{name}: parallel");
        assert_eq!(keeps(0xf8, 0x3, 0x3), board.cells.eth, "{name}: Ethernet");
        assert_eq!(r.c.eth().is_some(), board.cells.eth);
        // The one-wire master is the chip's: a slot always finishes. Whether a
        // part answers the reset with presence is the board's.
        r.w32(reg::PIN_CONTROL_SET, 1 << 21);
        r.w32(reg::MICROLAN, (500 << 10) | (13 << 2));
        let v = r.r32(reg::MICROLAN);
        assert_eq!(v & 2, 2, "{name}: the slot finishes");
        assert_eq!(v & 1 == 0, board.cells.w1, "{name}: presence");
        let straps: u32 = board.gpio.iter().map(|w| matches!(w.to, PinUse::Strap(true)) as u32 * (1 << w.pin)).sum();
        assert_eq!(r.r32(reg::PIN_DATA), straps, "{name}: straps");
        let rtc = board.cells.byte_bus == ByteBusPart::Ds1685;
        r.w8(reg::BYTEBUS_DEV1, 0x40);
        r.w8(reg::BYTEBUS_DEV2, 0x55);
        assert_eq!(r.r8(reg::BYTEBUS_DEV2) == 0x55, rtc, "{name}: DS1685");
    }
}

/// The reference Octane2's PROM monitor reads the chip's type 0
/// configuration space as 32 bytes repeated: dwords 0x20-0x3f read 0x00-0x1f,
/// so the latency word shows at 0x2c where a subsystem id might sit and the
/// interrupt pin byte at 0x3d reads zero. The firmware's own programming is
/// replayed (BAR 0 0x500000, latency timer 0x28, command 0x146). The dword at
/// 0x1c reads the BAR 0 value on that machine for a reason no public source
/// gives, so it is not asserted.
///
/// QEMU: octane2/ioc3-config-repeats.
#[test]
fn configuration_space_repeats_its_first_32_bytes() {
    let r = Rig::new();
    let cfg = r.c.state().config();
    cfg.write(4, 0x0050_0000);
    cfg.write(3, 0x2800);
    cfg.write(1, 0x146);

    for (index, want) in [(0, 0x0003_10a9), (1, 0x0280_0146), (2, 0xff00_0001), (3, 0x0000_2800), (4, 0x0050_0000), (5, 0), (6, 0)] {
        assert_eq!(r.c.config_dword(index), want, "dword {index}");
    }
    for index in 8..16 {
        if index != 15 {
            assert_eq!(r.c.config_dword(index), r.c.config_dword(index - 8), "dword {index} repeats");
        }
    }
    assert_eq!(r.c.config_dword(11), 0x0000_2800, "the latency word, not a subsystem id");
    assert_eq!(r.c.config_dword(15) >> 8 & 0xff, 0, "no interrupt pin byte");
    // The stored header keeps both for the board's wiring.
    assert_eq!(cfg.read(11), 0xc304_10a9);
    assert_eq!(cfg.interrupt_pin(), PCI_INTA);
}

/// The first five dwords of the register block are the configuration header
/// the same monitor reads back as 0x310a9, 0x2800146, 0xff000001, 0x2800 and
/// 0x500000. Pin control clear reads the control word, as the monitor reads
/// 0x2000f3 at both 0x34 and 0x38.
///
/// QEMU: octane2/ioc3-register-header.
#[test]
fn the_register_block_starts_with_the_configuration_header() {
    let r = Rig::new();
    let cfg = r.c.state().config();
    cfg.write(4, 0x0050_0000);
    cfg.write(3, 0x2800);
    cfg.write(1, 0x146);

    for off in (0..=0x10).step_by(4) {
        assert_eq!(r.r32(off), r.c.config_dword(off as usize / 4), "offset {off:#x}");
    }
    assert_eq!(r.r32(0x04), 0x0280_0146);
    assert_eq!(r.r32(0x10), 0x0050_0000);

    r.w32(reg::PIN_CONTROL_SET, 0x0020_00f3);
    assert_eq!(r.r32(reg::PIN_CONTROL_SET), 0x0020_00f3);
    assert_eq!(r.r32(reg::PIN_CONTROL_CLR), 0x0020_00f3);
}

// ---------------------------------------------------------------------------
// SuperIO window: the two 16550s
// ---------------------------------------------------------------------------

#[test]
fn ports_are_distinct() {
    let r = Rig::new();
    r.w8(reg::SERIAL_A + 7, 0x5a);
    r.w8(reg::SERIAL_B + 7, 0xa5);
    assert_eq!(r.r8(reg::SERIAL_A + 7), 0x5a);
    assert_eq!(r.r8(reg::SERIAL_B + 7), 0xa5);
}

#[test]
fn registers_are_a_byte_apart() {
    let r = Rig::new();
    r.w8(reg::SERIAL_A + 3, 0x03);
    r.w8(reg::SERIAL_A + 4, 0x05);
    assert_eq!(r.r8(reg::SERIAL_A + 3), 0x03);
    assert_eq!(r.r8(reg::SERIAL_A + 4), 0x05);
}

/// A four-byte access to the window is four registers, one per lane.
#[test]
fn the_port_answers_a_byte_at_a_time() {
    let r = Rig::new();
    r.w8(reg::SERIAL_A + 4, 0x05);
    r.w8(reg::SERIAL_A + 7, 0x5a);
    let lane: Vec<u8> = (0..4).map(|i| r.r8(reg::SERIAL_A + 4 + i)).collect();
    assert_eq!(lane[0], 0x05);
    assert_eq!(lane[3], 0x5a);
    // Status registers between them were read above, clearing what a read
    // clears; the wide access reads them again.
    let wide = r.r32(reg::SERIAL_A + 4);
    assert_eq!(wide, u32::from_le_bytes([lane[0], lane[1], lane[2], lane[3]]));
}

#[test]
fn console_transmit_reaches_the_backend() {
    let r = Rig::new();
    for b in b"hi\r\n" {
        assert_ne!(r.r8(reg::SERIAL_A + 5) & 0x20, 0, "THRE");
        r.w8(reg::SERIAL_A, *b);
    }
    assert_eq!(&*r.wire[0].out.lock().unwrap(), b"hi\r\n");
    assert!(r.wire[1].out.lock().unwrap().is_empty());
}

#[test]
fn console_receive_polls_lsr() {
    let r = Rig::new();
    assert_eq!(r.r8(reg::SERIAL_A + 5) & 1, 0);
    r.wire[0].inp.lock().unwrap().extend(b"ab");
    assert_eq!(r.r8(reg::SERIAL_A + 5) & 1, 1);
    assert_eq!(r.r8(reg::SERIAL_A), b'a');
}

#[test]
fn the_scratch_register_programs_the_predivider_under_dlab() {
    let r = Rig::new();
    r.w8(reg::SERIAL_A + 3, 0x80);
    r.w8(reg::SERIAL_A + 7, 6);
    assert_eq!(r.r8(reg::SERIAL_A + 7), 6, "reads back what was written");
    assert_eq!(r.c.predivider(0), 6);
    assert_eq!(r.c.baudbase(0), 22_118_400 / 3 / 16, "the divisor's clock follows it");
    // Written 0 or 1 means nothing.
    r.w8(reg::SERIAL_A + 7, 1);
    assert_eq!(r.c.predivider(0), 6);
}

/// A serial pass-through bit follows the 16550's level and ignores an
/// acknowledgement.
#[test]
fn a_serial_bit_ignores_an_acknowledgement() {
    let r = Rig::new();
    r.w32(reg::INT_ENABLE_SET, int::SERIAL_A);
    r.w8(reg::SERIAL_A + 1, 0x01); // IER: receive data
    assert!(!r.superio.level());
    r.wire[0].inp.lock().unwrap().push_back(b'x');
    r.c.service();
    assert_ne!(r.r32(reg::INT_STATUS) & int::SERIAL_A, 0);
    assert!(r.superio.level(), "the SuperIO line is up");
    r.w32(reg::INT_STATUS, 0xffff_ffff);
    assert_ne!(r.r32(reg::INT_STATUS) & int::SERIAL_A, 0, "level bits ignore the write");
    assert_eq!(r.r8(reg::SERIAL_A), b'x');
    assert_eq!(r.r32(reg::INT_STATUS) & int::SERIAL_A, 0);
    assert!(!r.superio.level());
}

#[test]
fn loopback_returns_the_byte() {
    let r = Rig::new();
    r.w8(reg::SERIAL_A + 4, 0x13);
    r.w8(reg::SERIAL_A, 0x42);
    assert!(r.wire[0].out.lock().unwrap().is_empty(), "nothing leaves the chip");
    assert_eq!(r.r8(reg::SERIAL_A + 5) & 1, 1);
    assert_eq!(r.r8(reg::SERIAL_A), 0x42);
    // Modem outputs wire back to the inputs.
    assert_eq!(r.r8(reg::SERIAL_A + 6) & 0x30, 0x30, "RTS to CTS, DTR to DSR");
}

// ---------------------------------------------------------------------------
// Interrupt registers, chip control
// ---------------------------------------------------------------------------

#[test]
fn the_enable_mask_has_two_write_aliases() {
    let r = Rig::new();
    assert_eq!(r.r32(reg::INT_ENABLE_SET), 0);
    r.w32(reg::INT_ENABLE_SET, int::SERIAL_A | int::SERIAL_B);
    assert_eq!(r.r32(reg::INT_ENABLE_SET), int::SERIAL_A | int::SERIAL_B);
    r.w32(reg::INT_ENABLE_CLR, int::SERIAL_B);
    assert_eq!(r.r32(reg::INT_ENABLE_SET), int::SERIAL_A);
    r.w32(reg::INT_ENABLE_SET, int::KEYBOARD);
    assert_eq!(r.r32(reg::INT_ENABLE_SET), int::SERIAL_A | int::KEYBOARD);
}

#[test]
fn the_arbiter_reports_idle() {
    let r = Rig::new();
    assert_eq!(r.r32(reg::CHIP_CONTROL) & CHIP_CTRL_ARB_IDLE, CHIP_CTRL_ARB_IDLE);
    r.w32(reg::CHIP_CONTROL, 0x0007_8000);
    let cr = r.r32(reg::CHIP_CONTROL);
    assert_eq!(cr & CHIP_CTRL_ARB_IDLE, CHIP_CTRL_ARB_IDLE);
    assert_eq!(cr & 0x0007_8000, 0x0007_8000);
}

#[test]
fn the_interrupt_output_stores_its_configuration() {
    let r = Rig::new();
    r.w32(reg::INT_OUTPUT, IO_PERIODIC | 0x40);
    assert_eq!(r.r32(reg::INT_OUTPUT) & !IO_STATE, IO_PERIODIC | 0x40);
    // The pin readback bit is not stored.
    r.w32(reg::INT_OUTPUT, 0);
    r.w32(reg::INT_OUTPUT, IO_STATE);
    assert_eq!(r.r32(reg::INT_OUTPUT) & !IO_STATE, 0);
}

#[test]
fn a_fixed_level_is_visible_to_the_next_read() {
    let r = Rig::new();
    r.w32(reg::INT_OUTPUT, IO_HIGH);
    assert_eq!(r.r32(reg::INT_OUTPUT) & IO_STATE, IO_STATE);
    // A rising edge posts the output-pulse source, whatever the mode.
    assert_ne!(r.r32(reg::INT_STATUS) & int::RT_OUTPUT_PULSE, 0);
    r.w32(reg::INT_OUTPUT, IO_LOW);
    assert_eq!(r.r32(reg::INT_OUTPUT) & IO_STATE, 0);
}

#[test]
fn diagnostic_output_commands_keep_the_running_output() {
    let r = Rig::new();
    r.w32(reg::INT_OUTPUT, IO_PERIODIC | 100);
    r.w32(reg::INT_OUTPUT, IO_DIAG | IO_PERIODIC | 5);
    assert_eq!(r.c.sh.int_out.lock().active, IO_PERIODIC | 100);
    // So do counts outside 2..=0xfffe.
    r.w32(reg::INT_OUTPUT, IO_SQUARE | 1);
    assert_eq!(r.c.sh.int_out.lock().active, IO_PERIODIC | 100);
    r.w32(reg::INT_OUTPUT, IO_SQUARE | 0xffff);
    assert_eq!(r.c.sh.int_out.lock().active, IO_PERIODIC | 100);
}

#[test]
fn a_periodic_output_catches_up_to_the_clock() {
    let r = Rig::new();
    r.w32(reg::INT_OUTPUT, IO_PERIODIC | 100);
    std::thread::sleep(Duration::from_millis(3));
    // Many periods (782 us each) have passed; the read applies them all and
    // the pulse source is posted.
    let _ = r.r32(reg::INT_OUTPUT);
    assert_ne!(r.r32(reg::INT_STATUS) & int::RT_OUTPUT_PULSE, 0);
    r.w32(reg::INT_STATUS, int::RT_OUTPUT_PULSE);
    // A fixed command cancels the periodic one for good.
    r.w32(reg::INT_OUTPUT, IO_LOW);
    std::thread::sleep(Duration::from_millis(2));
    let _ = r.r32(reg::INT_OUTPUT);
    assert_eq!(r.r32(reg::INT_STATUS) & int::RT_OUTPUT_PULSE, 0);
}

#[test]
fn reset_cancels_the_interrupt_output() {
    let r = Rig::new();
    r.w32(reg::INT_OUTPUT, IO_SQUARE | 2);
    r.c.reset();
    std::thread::sleep(Duration::from_millis(1));
    assert_eq!(r.r32(reg::INT_OUTPUT), 0);
    assert_eq!(r.r32(reg::INT_STATUS), 0);
}

#[test]
fn the_pulse_source_raises_the_superio_line_when_enabled() {
    let r = Rig::new();
    r.w32(reg::INT_ENABLE_SET, int::RT_OUTPUT_PULSE);
    r.w32(reg::INT_OUTPUT, IO_HIGH);
    assert!(r.superio.level());
    r.w32(reg::INT_STATUS, int::RT_OUTPUT_PULSE);
    assert!(!r.superio.level());
}

// ---------------------------------------------------------------------------
// GPIO and MicroLAN
// ---------------------------------------------------------------------------

#[test]
fn pin_control_has_two_write_aliases() {
    let r = Rig::new();
    r.w32(reg::PIN_CONTROL_SET, 0x00a0_0003);
    assert_eq!(r.r32(reg::PIN_CONTROL_SET), 0x00a0_0003);
    r.w32(reg::PIN_CONTROL_CLR, 0x0000_0001);
    assert_eq!(r.r32(reg::PIN_CONTROL_SET), 0x00a0_0002);
}

#[test]
fn a_pin_register_reaches_the_shared_data_word() {
    let r = Rig::new();
    r.w32(reg::PIN_DATA, 0);
    r.w32(reg::PIN_REG_FIRST + 4 * 6, 1);
    assert_eq!(r.r32(reg::PIN_DATA) & (1 << 6), 1 << 6);
    assert_eq!(r.r32(reg::PIN_REG_FIRST + 4 * 6), 1);
    r.w32(reg::PIN_REG_FIRST + 4 * 6, 0);
    assert_eq!(r.r32(reg::PIN_DATA) & (1 << 6), 0);
}

/// Pins 2 and 3 are read-only inputs the board drives high, on the Octane
/// and the Octane2 alike (one row serves both). Writes and direction changes
/// cannot replace them.
#[test]
fn the_board_inputs_are_read_only() {
    {
        let r = Rig::with(octane());
        for pin in [2u32, 3] {
            let reg = reg::PIN_REG_FIRST + 4 * pin;
            assert_eq!(r.r32(reg), 1);
            assert_eq!(r.r32(reg::PIN_DATA) >> pin & 1, 1);
            r.w32(reg, 0);
            r.w32(reg::PIN_DATA, 0);
            r.w32(reg::PIN_CONTROL_SET, 1 << pin);
            r.w32(reg::PIN_CONTROL_CLR, 1 << pin);
            assert_eq!(r.r32(reg), 1);
            assert_eq!(r.r32(reg::PIN_DATA) >> pin & 1, 1);
            r.c.reset();
            assert_eq!(r.r32(reg), 1, "a reset keeps the board's value");
        }
        // An unrelated pin still writes.
        r.w32(reg::PIN_REG_FIRST + 4 * 6, 1);
        assert_eq!(r.r32(reg::PIN_REG_FIRST + 4 * 6), 1);
    }
}

/// Direction and data are separate words: control has set and clear aliases
/// and never touches data, data holds what is written to an output pin and
/// the per-pin register is one bit of it. Pins 2 and 3 read their strap.
#[test]
fn gpio_direction_data_and_inputs_are_separate_words() {
    let r = Rig::new();
    r.w32(reg::PIN_CONTROL_SET, 0x0000_00f0);
    assert_eq!(r.r32(reg::PIN_DATA), 0x0c, "direction leaves data, the straps read high");
    r.w32(reg::PIN_DATA, 0x0000_0050);
    assert_eq!(r.r32(reg::PIN_DATA), 0x5c);
    assert_eq!(r.r32(reg::PIN_REG_FIRST + 4 * 4), 1);
    assert_eq!(r.r32(reg::PIN_REG_FIRST + 4 * 5), 0);
    r.w32(reg::PIN_CONTROL_CLR, 0x0000_0030);
    assert_eq!(r.r32(reg::PIN_CONTROL_SET), 0x0000_00c0);
    assert_eq!(r.r32(reg::PIN_DATA), 0x5c, "direction does not change data");
    r.w32(reg::PIN_DATA, 0xffff_ffff);
    assert_eq!(r.r32(reg::PIN_DATA), 0xffff_ffff);
    r.w32(reg::PIN_DATA, 0);
    assert_eq!(r.r32(reg::PIN_DATA), 0x0c, "only the straps stay");
}

/// The Octane's pins 0 and 1 are the front lightbar: white power, shown as
/// the status bar's green, and red fault. Only a change of the pin's value
/// moves the bit, and other pins do not.
#[test]
fn the_octane_lightbar_follows_pins_0_and_1() {
    let r = Rig::new();
    let hb = Arc::new(AtomicU64::new(0));
    r.c.set_heartbeat(hb.clone());
    let lit = || hb.load(Ordering::Relaxed) & (Rex3::HB_LED_GREEN | Rex3::HB_LED_RED);
    r.w32(reg::PIN_REG_FIRST, 1);
    assert_eq!(lit(), Rex3::HB_LED_GREEN);
    r.w32(reg::PIN_REG_FIRST + 4, 1);
    assert_eq!(lit(), Rex3::HB_LED_GREEN | Rex3::HB_LED_RED);
    r.w32(reg::PIN_REG_FIRST, 0);
    assert_eq!(lit(), Rex3::HB_LED_RED);
    r.w32(reg::PIN_REG_FIRST + 4 * 6, 1);
    assert_eq!(lit(), Rex3::HB_LED_RED, "an unwired pin lights nothing");
    r.w32(reg::PIN_DATA, 0b11);
    assert_eq!(lit(), Rex3::HB_LED_GREEN | Rex3::HB_LED_RED);
    r.w32(reg::PIN_DATA, 0);
    assert_eq!(lit(), 0);
}

/// What each board's row says about the pins, written out per board: the
/// levels its straps hold, and the status-bar bits its outputs reach. A pin
/// the row does not list is unwired: it keeps what is written and reaches no
/// status-bar bit.
#[test]
fn each_row_wires_its_pins_as_listed() {
    // (board, pins held high, [(pin, status-bar bit)])
    let rows: [(&Ioc3Board, u32, &[(u32, u64)]); 6] = [
        (&IP30, 0b1100, &[(0, Rex3::HB_LED_GREEN), (1, Rex3::HB_LED_RED)]),
        (&IP29, 0, &[]),
        (&BASEIO, 0, &[]),
        (&BASEIO6G, 0, &[]),
        (&IP34, 0, &[]),
        (&MIO, 0, &[]),
    ];
    for (board, straps, bits) in rows {
        let bus = match board.cells.byte_bus {
            ByteBusPart::None => ByteBus::None,
            ByteBusPart::Ds1685 => ByteBus::Ds1685(Ds1685::new(None)),
            ByteBusPart::M48t35 => {
                ByteBus::M48t35(Arc::new(crate::dev::timekeeper::Timekeeper::at(2026, 1, 1, 0, 0, 0).unwrap()))
            }
        };
        let r = Rig::with(Ioc3::new(board, MAC, bus));
        let hb = Arc::new(AtomicU64::new(0));
        r.c.set_heartbeat(hb.clone());
        let name = board.name;
        for pin in 0..9u32 {
            let reg = reg::PIN_REG_FIRST + 4 * pin;
            let strapped = straps >> pin & 1 != 0;
            let bit = bits.iter().find(|b| b.0 == pin).map_or(0, |b| b.1);
            r.w32(reg, 1);
            let high = r.r32(reg) == 1;
            let leds = hb.load(Ordering::Relaxed) & (Rex3::HB_LED_GREEN | Rex3::HB_LED_RED);
            let listed = board.gpio.iter().any(|w| w.pin as u32 == pin && matches!(w.to, PinUse::Strap(_)));
            assert_eq!(high, strapped || !listed, "{name} pin {pin}: a written 1 reads back unless a strap holds it low");
            assert_eq!(leds, bit, "{name} pin {pin}: status-bar bit after a 1");
            r.w32(reg, 0);
            assert_eq!(r.r32(reg) == 1, strapped, "{name} pin {pin}: a written 0");
            assert_eq!(hb.load(Ordering::Relaxed) & (Rex3::HB_LED_GREEN | Rex3::HB_LED_RED), 0, "{name} pin {pin}: cleared");
        }
        assert_eq!(r.r32(reg::PIN_DATA), straps, "{name}: nothing but the straps left");
    }
}

// The MicroLAN protocol (search, match, page reads, CRCs) is tested with the
// wire in `dev::w1`; the register cell and the board's default wire are below.

// The register cell: one write is one slot, finished inside the write.
#[test]
fn the_microlan_register_finishes_a_slot_inside_the_write() {
    let r = Rig::new();
    // With the engine unrouted from its pin the wire reads high.
    r.w32(reg::MICROLAN, (500 << 10) | (13 << 2));
    let v = r.r32(reg::MICROLAN);
    assert_eq!(v & 3, 3, "done, and the wire reads high");
    assert_eq!(v & !3, (500 << 10) | (13 << 2), "the timing written reads back");
}

/// The wire the chip is built with: the station-address part. The board
/// records are the machine's.
#[test]
fn the_default_wire_carries_the_mac_part() {
    let r = Rig::new();
    r.w32(reg::PIN_CONTROL_SET, 1 << 21);
    r.w32(reg::MICROLAN, (500 << 10) | (13 << 2));
    assert_eq!(r.r32(reg::MICROLAN) & 1, 0, "a reset is answered with presence");
    let mut families: Vec<u8> = r.c.sh.w1.roms().iter().map(|rom| rom[0]).collect();
    families.sort();
    assert_eq!(families, [0x91]);
}

// ---------------------------------------------------------------------------
// PS/2
// ---------------------------------------------------------------------------

const K_RX: u32 = 0xa0;
const M_RX: u32 = 0xa4;
const K_TX: u32 = 0xa8;
const M_TX: u32 = 0xac;
const CTRL: u32 = 0x9c;
const VALID: [u32; 3] = [1 << 29, 1 << 30, 1 << 31];

#[test]
fn the_ps2_ports_read_empty() {
    let r = Rig::new();
    assert_eq!(r.r32(CTRL), 0);
    assert_eq!(r.r32(K_RX), 0);
    assert_eq!(r.r32(M_RX), 0);
    assert_eq!(r.r32(reg::INT_STATUS) & int::KEYBOARD, 0);
}

#[test]
fn the_ps2_keyboard_round_trips() {
    let r = Rig::new();
    r.w32(K_TX, 0xff);
    assert_ne!(r.r32(reg::INT_STATUS) & int::KEYBOARD, 0, "a reply latches the source");
    let v = r.r32(K_RX);
    // The oldest byte is in the highest slot: the ACK.
    assert_ne!(v & VALID[2], 0);
    assert_eq!((v >> 16) & 0xff, 0xfa);
    assert_eq!(v & (1 << 27), 0, "from the keyboard");
}

#[test]
fn the_ps2_mouse_round_trips() {
    let r = Rig::new();
    r.w32(M_TX, 0xf2);
    let v = r.r32(M_RX);
    assert_ne!(v & VALID[2], 0);
    assert_ne!(v & (1 << 27), 0, "from the mouse");
    assert_eq!((v >> 16) & 0xff, 0xfa);
    assert_eq!(r.r32(K_RX), 0, "and not the keyboard");
}

#[test]
fn the_ps2_receive_drains_on_read_but_the_interrupt_latches() {
    let r = Rig::new();
    r.w32(K_TX, 0xff);
    assert_ne!(r.r32(K_RX), 0);
    assert_eq!(r.r32(K_RX), 0, "a read drains the holding");
    assert_ne!(r.r32(reg::INT_STATUS) & int::KEYBOARD, 0, "the latch stays set");
    r.w32(reg::INT_STATUS, int::KEYBOARD);
    assert_eq!(r.r32(reg::INT_STATUS) & int::KEYBOARD, 0);
}

fn type_keys(r: &Rig) {
    let ps2 = r.c.ps2();
    for k in [winit::keyboard::KeyCode::KeyA, winit::keyboard::KeyCode::KeyB, winit::keyboard::KeyCode::KeyC] {
        ps2.push_kb(k, true);
        ps2.push_kb(k, false);
    }
}

/// More than three characters behind the clamp-after-three bits: three now,
/// the rest once the guest reads, and nothing lost.
#[test]
fn the_ps2_receive_holds_three_behind_the_clamp() {
    let r = Rig::new();
    r.w32(CTRL, KBC_K_CLAMP_3 | KBC_M_CLAMP_3);
    r.w32(K_TX, 0xf4); // enable scanning
    let _ = r.r32(K_RX);
    r.w32(reg::INT_STATUS, int::KEYBOARD);
    type_keys(&r);
    assert_ne!(r.r32(reg::INT_STATUS) & int::KEYBOARD, 0);
    let first = r.r32(K_RX);
    assert_eq!(first & (VALID[0] | VALID[1] | VALID[2]), VALID[0] | VALID[1] | VALID[2]);
    assert_eq!(first & KBC_RX_OVERFLOW, 0, "nothing lost");
    // The read released the clamp; the stopped keyboard resumed.
    let second = r.r32(K_RX);
    assert_ne!(second & VALID[2], 0, "the rest were held back, not dropped");
    assert_ne!((second >> 16) & 0xff, (first >> 16) & 0xff, "and are not a replay");
}

#[test]
fn the_ps2_receive_overflows_without_a_clamp() {
    let r = Rig::new();
    r.w32(K_TX, 0xf4);
    let _ = r.r32(K_RX);
    type_keys(&r);
    let v = r.r32(K_RX);
    assert_ne!(v & KBC_RX_OVERFLOW, 0, "a fourth character was lost");
}

#[test]
fn the_ps2_control_masks_writes() {
    let r = Rig::new();
    const K_TO_EN: u32 = 0x0004_0000;
    const M_TO_EN: u32 = 0x0008_0000;
    r.w32(CTRL, K_TO_EN);
    assert_eq!(r.r32(CTRL), K_TO_EN);
    r.w32(CTRL, K_TO_EN | M_TO_EN);
    assert_eq!(r.r32(CTRL), K_TO_EN | M_TO_EN);
    r.w32(CTRL, 0x0001_0000 | 0x0002_0000);
    assert_eq!(r.r32(CTRL), 0, "timeout status bits do not store");
    r.w32(CTRL, 0xffff_ffff);
    assert_eq!(
        r.r32(CTRL),
        0x0000_ffff | K_TO_EN | M_TO_EN | KBC_K_CLAMP_1 | KBC_M_CLAMP_1 | KBC_K_CLAMP_3 | KBC_M_CLAMP_3
    );
    r.c.reset();
    assert_eq!(r.r32(CTRL), 0);
}

#[test]
fn a_host_key_latches_the_interrupt_without_a_guest_access() {
    let r = Rig::new();
    r.w32(reg::INT_ENABLE_SET, int::KEYBOARD);
    r.w32(K_TX, 0xf4);
    let _ = r.r32(K_RX);
    r.w32(reg::INT_STATUS, int::KEYBOARD);
    assert!(!r.superio.level());
    r.c.ps2().push_kb(winit::keyboard::KeyCode::KeyA, true);
    assert!(r.superio.level(), "the SuperIO line rose from the host thread");
}

// ---------------------------------------------------------------------------
// Serial DMA
// ---------------------------------------------------------------------------

const A: u32 = reg::SDMA_PORT_A;
const B: u32 = reg::SDMA_PORT_B;

#[test]
fn the_serial_dma_indices_take_a_write() {
    let r = Rig::new();
    r.w32(A + sdma_off::TX_PRODUCE, 0x40);
    assert_eq!(r.r32(A + sdma_off::TX_PRODUCE), 0x40);
    r.w32(A + sdma_off::RX_CONSUME, 0x48);
    assert_eq!(r.r32(A + sdma_off::RX_CONSUME), 0x48);
    // The two the engine advances can still be placed by software.
    r.w32(A + sdma_off::TX_CONSUME, 0x50);
    assert_eq!(r.r32(A + sdma_off::TX_CONSUME), 0x50);
    r.w32(A + sdma_off::RX_PRODUCE, 0x58);
    assert_eq!(r.r32(A + sdma_off::RX_PRODUCE), 0x58);
    // Only the receive consume index keeps bit 31; the rest keep the pointer.
    r.w32(A + sdma_off::RX_CONSUME, 0xffff_ffff);
    assert_eq!(r.r32(A + sdma_off::RX_CONSUME), 0x8000_03f8);
    for o in [sdma_off::TX_PRODUCE, sdma_off::TX_CONSUME, sdma_off::RX_PRODUCE] {
        r.w32(A + o, 0xffff_ffff);
        assert_eq!(r.r32(A + o), 0x3f8);
    }
}

#[test]
fn the_serial_dma_ports_are_distinct() {
    let r = Rig::new();
    r.w32(A + sdma_off::TX_PRODUCE, 0x18);
    r.w32(B + sdma_off::TX_PRODUCE, 0x28);
    assert_eq!(r.r32(A + sdma_off::TX_PRODUCE), 0x18);
    assert_eq!(r.r32(B + sdma_off::TX_PRODUCE), 0x28);
    r.w32(A + sdma_off::TX_CONSUME, 0x20);
    r.w32(B + sdma_off::TX_CONSUME, 0x30);
    assert_eq!(r.r32(A + sdma_off::TX_CONSUME), 0x20);
    assert_eq!(r.r32(B + sdma_off::TX_CONSUME), 0x30);
    r.w32(reg::SDMA_RING_LO, 0x3000);
    assert_eq!(r.r32(reg::SDMA_RING_LO), 0x3000, "the ring base is shared");
}

#[test]
fn the_serial_dma_shadow_is_read_only() {
    let r = Rig::new();
    for base in [A, B] {
        let s = r.r32(base + sdma_off::SHADOW);
        r.w32(base + sdma_off::SHADOW, 0xffff_ffff);
        assert_eq!(r.r32(base + sdma_off::SHADOW), s);
        assert_ne!(s & (1 << 23), 0, "the port comes up with carrier asserted");
        assert_ne!(s & (1 << 20), 0, "and clear to send");
        assert_eq!(s & 0x60, 0x60, "transmitter empty");
    }
}

#[test]
fn the_alias_pages_expose_one_port_and_refuse_the_engine_indices() {
    let r = Rig::new();
    r.w32(A + sdma_off::TX_PRODUCE, 0x18);
    assert_eq!(r.r32(reg::SDMA_ALIAS_A + sdma_off::TX_PRODUCE), 0x18);
    assert_eq!(r.r32(reg::SDMA_ALIAS_B + sdma_off::TX_PRODUCE), 0);
    r.w32(reg::SDMA_ALIAS_A + sdma_off::TX_CONSUME, 0x30);
    assert_eq!(r.r32(A + sdma_off::TX_CONSUME), 0, "read-only through the page");
    r.w32(reg::SDMA_ALIAS_A + sdma_off::TX_PRODUCE, 0x20);
    assert_eq!(r.r32(A + sdma_off::TX_PRODUCE), 0x20, "the producer index is writable");
}

#[test]
fn the_receive_timer_reload_loads_the_count() {
    let r = Rig::new();
    r.w32(A + sdma_off::RX_TIMER, 0xfff);
    assert_eq!(r.r32(A + sdma_off::RX_TIMER) & 0x0fff_0fff, 0x0fff_0fff);
}

/// Write `data` as internal bytes at internal address `a`.
fn put(r: &Rig, a: u64, data: &[u8]) {
    assert!(r.mem_view().dma_write(a, data).is_ok());
}

fn get(r: &Rig, a: u64, n: usize) -> Vec<u8> {
    let mut b = vec![0; n];
    assert!(r.mem_view().dma_read(a, &mut b).is_ok());
    b
}

/// Ring DMA in loopback: four characters out of the transmit ring come back
/// through the receive ring as one entry, with their status bytes.
#[test]
fn ring_dma_loops_back_through_port_a() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1); // 4 KB rings
    r.w8(reg::SERIAL_A + 4, 0x11);
    put(&r, 0x2000, &[b'a', b'b', b'c', b'd', 0x40, 0x40, 0x40, 0x40]);
    r.w32(A, sdma_ctl::ENABLE);
    r.w32(A + sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.r32(A + sdma_off::TX_CONSUME), 8, "the engine took the entry");
    assert_eq!(r.r32(A + sdma_off::RX_PRODUCE) & 0xfff, 8, "and posted one back");
    let e = get(&r, 0x3000, 8);
    assert_eq!(&e[..4], b"abcd");
    for s in &e[4..] {
        assert_eq!(s & 0xc0, 0xc0, "data and modem fields valid");
        assert_eq!(s & 0x0f, 0, "no errors");
    }
    assert!(r.wire[0].out.lock().unwrap().is_empty());
}

#[test]
fn ring_dma_runs_on_port_b_too() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    r.w8(reg::SERIAL_B + 4, 0x11);
    put(&r, 0x4000, &[b'w', b'x', b'y', b'z', 0x40, 0x40, 0x40, 0x40]);
    r.w32(B, sdma_ctl::ENABLE);
    r.w32(B + sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.r32(B + sdma_off::TX_CONSUME), 8);
    assert_eq!(&get(&r, 0x5000, 4), b"wxyz");
}

/// Flow control in loopback reports clear-to-send even with RTS clear.
#[test]
fn flow_control_in_loopback_reports_cts() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    r.w8(reg::SERIAL_A + 4, 0x11);
    put(&r, 0x2000, &[b'a', b'b', b'c', b'd', 0x40, 0x40, 0x40, 0x40]);
    r.w32(A, sdma_ctl::ENABLE | sdma_ctl::FLOW_CONTROL | 32);
    r.w32(A + sdma_off::TX_PRODUCE, 8);
    let e = get(&r, 0x3000, 8);
    assert_eq!(e[4] & 0xdf, 0xd0, "data valid, modem valid, clear to send");
}

/// An empty slot ends its entry wherever it sits; the characters behind it
/// are not sent.
#[test]
fn an_empty_slot_ends_the_entry() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    r.w8(reg::SERIAL_A + 4, 0x11);
    put(&r, 0x2000, &[b'a', b'b', b'c', b'd', 0x40, 0x00, 0x40, 0x40]);
    put(&r, 0x2008, &[b'e', b'f', b'g', b'h', 0x40, 0x40, 0x40, 0x00]);
    r.w32(A, sdma_ctl::ENABLE);
    r.w32(A + sdma_off::TX_PRODUCE, 16);
    assert_eq!(r.r32(A + sdma_off::TX_CONSUME), 16);
    // a, then e f g: "aefg" in one receive entry.
    assert_eq!(&get(&r, 0x3000, 4), b"aefg");
}

#[test]
fn ring_dma_transmits_to_the_backend_and_raises_the_explicit_source() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    put(&r, 0x2000, &[b'o', b'k', b'\n', 0, 0x40, 0x40, 0x60, 0x00]);
    r.w32(A, sdma_ctl::ENABLE);
    r.w32(A + sdma_off::TX_PRODUCE, 8);
    assert_eq!(&*r.wire[0].out.lock().unwrap(), b"ok\n");
    let bit = int::serial(0, int::TX_EXPLICIT);
    assert_ne!(r.r32(reg::INT_STATUS) & bit, 0);
    r.w32(reg::INT_STATUS, bit);
    assert_eq!(r.r32(reg::INT_STATUS) & bit, 0, "latched, then cleared");
}

#[test]
fn transmit_ring_empty_is_a_level() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    let bit = int::serial(0, int::TX_RING_EMPTY);
    assert_eq!(r.r32(reg::INT_STATUS) & bit, 0, "zero while ring DMA is off");
    r.w32(A, sdma_ctl::ENABLE);
    assert_ne!(r.r32(reg::INT_STATUS) & bit, 0);
    r.w32(reg::INT_STATUS, bit);
    assert_ne!(r.r32(reg::INT_STATUS) & bit, 0, "a write cannot clear it");
}

/// Host input reaches the receive ring through the service pass, with the
/// 16550's own interrupts off, and the arm and timer post the partial entry.
#[test]
fn host_input_arrives_in_the_receive_ring_after_the_timer() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    // 0x200 ticks of 62.5 us: 32 ms.
    r.w32(A + sdma_off::RX_TIMER, 0x200);
    r.w32(A + sdma_off::RX_CONSUME, ARM_RX_TIMER);
    r.w32(A, sdma_ctl::ENABLE | 4);
    r.wire[0].inp.lock().unwrap().extend(b"hi");
    let before = Instant::now();
    r.c.service();
    let after = Instant::now();
    // Every register access first expires a deadline already past, so a
    // test thread descheduled past it would see the entry posted before it
    // looked. Hold the timer off while looking, then let it run out.
    let wait = Duration::from_nanos(0x200 * TIMER_TICK_NS);
    let due = {
        let serial = r.c.sh.serial[0].as_ref().unwrap();
        let due = serial.rx_deadline().expect("the first character started the timer");
        assert!(due >= before + wait && due <= after + wait, "the reload's 32 ms from the character");
        serial.set_rx_deadline(Some(Instant::now() + Duration::from_secs(3600)));
        due
    };
    assert_eq!(r.r32(A + sdma_off::RX_PRODUCE) >> HELD_SHIFT & 7, 2, "two held in the packer");
    assert_eq!(r.r32(A + sdma_off::RX_CONSUME) & ARM_RX_TIMER, ARM_RX_TIMER);
    r.c.sh.serial[0].as_ref().unwrap().set_rx_deadline(Some(due));
    std::thread::sleep(due.saturating_duration_since(Instant::now()));
    r.c.service();
    assert_eq!(r.r32(A + sdma_off::RX_PRODUCE), 8, "posted when the timer ran out");
    assert_eq!(&get(&r, 0x3000, 2), b"hi");
    assert_ne!(r.r32(reg::INT_STATUS) & int::serial(0, int::RX_TIMER), 0);
    assert_eq!(r.r32(A + sdma_off::RX_CONSUME) & ARM_RX_TIMER, 0, "the arm is spent");
}

/// Receive-ring-full is a level, a model choice: set while ring DMA is on
/// and an entry waits that the ring had no room for. An acknowledgement
/// leaves it, running the engine again with no new character does not raise
/// it afresh, and it falls when software makes room. A ring filled exactly,
/// with nothing waiting, does not raise it.
///
/// QEMU: sn0/ioc3-serial/receive-ring-full-is-a-level.
#[test]
fn receive_ring_full_is_a_level() {
    let r = Rig::new();
    let full = int::serial(0, int::RX_RING_FULL);
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);

    // Full to the last entry, but nothing waits for room.
    r.w32(A + sdma_off::RX_CONSUME, 8);
    r.w32(A, sdma_ctl::ENABLE);
    assert_eq!(r.r32(reg::INT_STATUS) & full, 0);

    // Room for one entry; the second waits.
    r.w32(A + sdma_off::RX_CONSUME, 16);
    r.wire[0].inp.lock().unwrap().extend(b"ringfull");
    r.c.service();
    assert_eq!(r.r32(A + sdma_off::RX_PRODUCE) & 0xff8, 8);
    assert_eq!(r.r32(reg::INT_STATUS) & full, full);

    // An acknowledgement does not clear it, and running the engine again
    // does not matter: it is the ring that is full.
    r.w32(reg::INT_STATUS, full);
    r.w32(A + sdma_off::TX_PRODUCE, 0);
    assert_eq!(r.r32(reg::INT_STATUS) & full, full);

    // Ring DMA off, the source with it; on again, it is back.
    r.w32(A, 0);
    assert_eq!(r.r32(reg::INT_STATUS) & full, 0);
    r.w32(A, sdma_ctl::ENABLE);
    assert_eq!(r.r32(reg::INT_STATUS) & full, full);

    // Room for the waiting entry: it posts and fills the ring exactly.
    r.w32(A + sdma_off::RX_CONSUME, 24);
    assert_eq!(r.r32(A + sdma_off::RX_PRODUCE) & 0xff8, 16);
    assert_eq!(r.r32(reg::INT_STATUS) & full, 0);
}

#[test]
fn an_engine_aimed_elsewhere_does_not_reach_its_port() {
    let r = Rig::new();
    r.w32(reg::CHIP_CONTROL, 0x0003_8000);
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    put(&r, 0x2000, &[b'a', 0, 0, 0, 0x40, 0, 0, 0]);
    r.w32(A, sdma_ctl::ENABLE);
    r.w32(A + sdma_off::TX_PRODUCE, 8);
    assert_eq!(r.r32(A + sdma_off::TX_CONSUME), 0);
    assert!(r.wire[0].out.lock().unwrap().is_empty());
}

#[test]
fn ioc3_dma_places_bytes_xor_three() {
    let r = Rig::new();
    let d = r.mem_view();
    // A big-endian word at an aligned address.
    assert!(d.dma_write(0x100, &[0x11, 0x22, 0x33, 0x44]).is_ok());
    assert_eq!(&r.mem.0.lock().unwrap()[0x100..0x104], &[0x44, 0x33, 0x22, 0x11]);
    // A single byte lands on its own lane.
    assert!(d.dma_write(0x201, &[0xab]).is_ok());
    assert_eq!(r.mem.0.lock().unwrap()[0x202], 0xab);
    // And a read of an unaligned span undoes the same placement.
    let mut b = [0u8; 3];
    assert!(d.dma_read(0x100, &mut b).is_ok());
    assert_eq!(b, [0x11, 0x22, 0x33]);
    // A failed access reports it, and the upper address bits pass through.
    let mut big = [0u8; 8];
    assert!(!d.dma_read(0xfffc, &mut big).is_ok());
    assert!(!d.dma_read(0x1_0000_0000, &mut big).is_ok());
}

// ---------------------------------------------------------------------------
// Buffer RAM, byte bus, parallel, Ethernet window
// ---------------------------------------------------------------------------

#[test]
fn the_buffer_ram_holds_seventeen_bits_and_does_not_alias() {
    let r = Rig::new();
    r.w32(reg::SSRAM, 0xffff_ffff);
    r.w32(reg::SSRAM + 0x20000, 0x1234);
    assert_eq!(r.r32(reg::SSRAM), 0x0001_ffff);
    assert_eq!(r.r32(reg::SSRAM + 0x20000), 0x1234, "a window aliasing at its midpoint fails the size probe");
    assert_eq!(r.r32(reg::SSRAM + 0x3fffc), 0);
}

#[test]
fn the_clock_is_reached_through_an_index_and_a_data_port() {
    let r = Rig::new();
    r.w8(reg::BYTEBUS_DEV1, 0x0b);
    r.w8(reg::BYTEBUS_DEV2, 0x06);
    assert_eq!(r.r8(reg::BYTEBUS_DEV2), 0x06);
    r.w8(reg::BYTEBUS_DEV1, 0x20);
    r.w8(reg::BYTEBUS_DEV2, 0xa5);
    r.w8(reg::BYTEBUS_DEV1, 0x21);
    r.w8(reg::BYTEBUS_DEV2, 0x5a);
    r.w8(reg::BYTEBUS_DEV1, 0x20);
    assert_eq!(r.r8(reg::BYTEBUS_DEV2), 0xa5);
    r.w8(reg::BYTEBUS_DEV1, 0x21);
    assert_eq!(r.r8(reg::BYTEBUS_DEV2), 0x5a);
    // The data port answers on any lane of its word.
    assert_eq!(r.c.bar_read(0, reg::BYTEBUS_DEV2 + 3, 1), 0x5a, "the lane for offset 0");
    assert_eq!(r.r8(reg::BYTEBUS_DEV2 + 3), 0x5a, "and offset 3");
    // Register D reads valid-RAM-and-time.
    r.w8(reg::BYTEBUS_DEV1, 0x0d);
    assert_eq!(r.r8(reg::BYTEBUS_DEV2) & 0x80, 0x80);
}

#[test]
fn the_clock_tells_the_time() {
    let r = Rig::new();
    r.w8(reg::BYTEBUS_DEV1, 0x09);
    let year = r.r8(reg::BYTEBUS_DEV2);
    assert!(year >= 0x26, "host time on power-up, in BCD: {year:#x}");
}

#[test]
fn nvram_survives_in_a_file() {
    let path = std::env::temp_dir().join(format!("ioc3-nvram-{}", std::process::id()));
    {
        let r = Rig::new();
        r.c.rtc().set_nvram_file(&path).unwrap();
        r.w8(reg::BYTEBUS_DEV1, 0x30);
        r.w8(reg::BYTEBUS_DEV2, 0xc3);
    }
    {
        let r = Rig::new();
        r.c.rtc().set_nvram_file(&path).unwrap();
        r.w8(reg::BYTEBUS_DEV1, 0x30);
        assert_eq!(r.r8(reg::BYTEBUS_DEV2), 0xc3);
    }
    let _ = std::fs::remove_file(&path);
}

/// The file is not rewritten per byte: it catches up once the writes
/// pause, replaced whole, and on a flush with no wait.
#[test]
fn nvram_writes_reach_the_file_after_a_pause() {
    let path = std::env::temp_dir().join(format!("ioc3-nvram-quiet-{}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let r = Rig::new();
    r.c.rtc().set_nvram_file(&path).unwrap();
    rtc_w(&r, 0x45, 0x66);
    let in_file = || std::fs::read(&path).unwrap()[NVRAM_OFF_NV + 5];
    assert_eq!(in_file(), 0, "not yet written");
    r.c.rtc().flush_after(Duration::from_secs(3600));
    assert_eq!(in_file(), 0, "the writes have not paused for long enough");
    r.c.rtc().flush_after(Duration::ZERO);
    assert_eq!(in_file(), 0x66);
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0x200);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn software_power_off_is_reported() {
    let r = Rig::new();
    assert!(!r.c.rtc().powered_off());
    r.w8(reg::BYTEBUS_DEV1, 0x0a);
    r.w8(reg::BYTEBUS_DEV2, 0x10); // bank 1
    r.w8(reg::BYTEBUS_DEV1, 0x4a);
    r.w8(reg::BYTEBUS_DEV2, 0x08);
    assert!(r.c.rtc().powered_off());
}

const ECP: u32 = reg::PAR_ECP_BASE;

#[test]
fn the_parallel_port_presents_the_probe_values() {
    let r = Rig::new();
    assert_eq!(r.r8(ECP + 0xa), PAR_ECR_FIFO_EMPTY, "empty, not full: the driver's presence check");
    assert_eq!(r.r8(ECP + 1) & 0x08, 0x08, "no fault");
}

/// The status register reads what is on the cable. A connector with nothing
/// on it sits at the five input pull-ups, 0x78, with bit 7 (the inverse of
/// Busy) clear; a host backend reads as a printer that can take data, 0xd8.
///
/// QEMU: sn0/ioc3-parallel (empty connector and file backend cases).
#[test]
fn the_parallel_status_follows_what_is_attached() {
    let r = Rig::new();
    assert_eq!(r.r8(ECP + 1), 0x78, "empty connector");
    r.c.set_parallel_backend(Arc::new(Wire::default()));
    assert_eq!(r.r8(ECP + 1), 0xd8, "a ready printer");
}

#[test]
fn the_parallel_test_fifo_stores_and_returns_bytes_in_order() {
    let r = Rig::new();
    r.w8(ECP + 0xa, 6 << 5);
    for i in 0..4 {
        r.w8(ECP + 8, i + 1);
    }
    assert_eq!(r.r8(ECP + 0xa) & 3, 0, "neither empty nor full");
    for i in 0..4 {
        assert_eq!(r.r8(ECP + 8), i + 1);
    }
    assert_ne!(r.r8(ECP + 0xa) & PAR_ECR_FIFO_EMPTY, 0);
    for i in 0..16 {
        r.w8(ECP + 8, i);
    }
    assert_ne!(r.r8(ECP + 0xa) & PAR_ECR_FIFO_FULL, 0, "sixteen bytes is full");
}

#[test]
fn the_parallel_configuration_is_fixed_in_config_mode() {
    let r = Rig::new();
    r.w8(ECP + 0xa, 7 << 5);
    assert_eq!(r.r8(ECP + 8), PAR_CONFIG_A);
    assert_eq!(r.r8(ECP + 9), PAR_CONFIG_B);
    r.w8(ECP + 8, 0);
    r.w8(ECP + 9, 0);
    assert_eq!(r.r8(ECP + 8), PAR_CONFIG_A, "read-only on the fitted part");
}

/// A DMA context drains to the backend and completion is the pass-through
/// service level, quieted by a write to the extended control register.
#[test]
fn parallel_dma_drains_a_context_and_raises_service() {
    let r = Rig::new();
    let out = Arc::new(Wire::default());
    r.c.set_parallel_backend(out.clone());
    put(&r, 0x3000, b"print me");
    r.w32(0x80, 0);
    r.w32(0x84, 0x3000);
    r.w8(ECP + 0xa, PAR_ECR_DMA_ENABLE | (2 << 5));
    r.w32(0x8c, PAR_DMA_ENABLE);
    r.w32(0x88, 8 | PAR_COUNT_ARM | PAR_COUNT_LAST);
    assert_eq!(&*out.out.lock().unwrap(), b"print me");
    let st = r.r32(reg::INT_STATUS);
    assert_ne!(st & (1 << int::PARALLEL_CONTEXT), 0, "context A's own source latched");
    assert_ne!(st & int::PARALLEL_SERVICE, 0);
    // The context disarmed itself.
    assert_eq!(r.r32(0x88) & (PAR_COUNT | PAR_COUNT_ARM), 0);
    r.w32(reg::INT_STATUS, 0xffff_ffff);
    let st = r.r32(reg::INT_STATUS);
    assert_eq!(st & (1 << int::PARALLEL_CONTEXT), 0, "latched bits clear on a write");
    assert_ne!(st & int::PARALLEL_SERVICE, 0, "the service level stays");
    r.w8(ECP + 0xa, PAR_ECR_DMA_ENABLE | (2 << 5));
    assert_eq!(r.r32(reg::INT_STATUS) & int::PARALLEL_SERVICE, 0, "quieted at the port");
}

#[test]
fn the_ethernet_registers_are_the_macs_and_it_gets_the_chips_dma() {
    let r = Rig::new();
    let eth = r.c.eth().unwrap().clone();
    r.w32(0xf8, 0x1234);
    r.w32(0x150, 0xbeef);
    assert_eq!(eth.read32(0xf8), 0x1234);
    assert_eq!(eth.read32(0x150), 0xbeef);
    assert_eq!(r.r32(0xf8), 0x1234);
    // Just before the block and just after it are not the block's.
    r.w32(0xec, 7);
    assert_eq!(eth.read32(0xec), 0);
    r.w32(0x154, 7);
    assert_eq!(r.r32(0x154), 0);
    // Only 32-bit accesses reach the block.
    assert_eq!(r.r8(0xf8), 0);
    // Its ring entries go through the chip's byte order: the first descriptor
    // word is big-endian on the bus, as `Ioc3Dma` places it.
    let dma = r.c.sh.core.dma().unwrap().clone();
    assert!(dma.dma_write(0x500, &[0xde, 0xad, 0xbe, 0xef]).is_ok());
    assert_eq!(&r.mem.0.lock().unwrap()[0x500..0x504], &[0xef, 0xbe, 0xad, 0xde]);
    let mut b = [0u8; 4];
    assert!(dma.dma_read(0x500, &mut b).is_ok());
    assert_eq!(b, [0xde, 0xad, 0xbe, 0xef]);
}

/// Ethernet drives the PCI pin; the SuperIO line is a different one.
#[test]
fn the_ethernet_level_drives_the_pci_pin_not_the_superio_line() {
    let c = octane();
    c.state().connect_dma(TestMem::new());
    let (pin, line) = LevelProbe::line();
    c.state().connect_irq(line);
    c.attached();
    let (sup, sline) = LevelProbe::line();
    c.connect_superio_interrupt(sline);
    let eth = c.eth().unwrap();
    eth.write32(0xf8, 1);
    eth.raise_for_test(1);
    assert!(pin.level());
    assert!(!sup.level());
    assert_eq!(c.int_status(), 0, "no SuperIO status bit for it");
    eth.write32(0xf4, 1);
    assert!(!pin.level());
}

#[test]
fn buffer_ram_parity_follows_the_mac_control_bit() {
    let r = Rig::new();
    let eth = r.c.eth().unwrap().clone();
    // Checking off: stored as written, 17 bits, no error flag.
    r.w32(reg::SSRAM, 0x0001_0001);
    assert_eq!(r.r32(reg::SSRAM), 0x0001_0001);
    // Checking on: an odd number of ones in the stored word reads as a
    // parity error.
    eth.write32(reg::ENET_MAC_CONTROL, ENET_MCR_RAM_PARITY);
    r.w32(reg::SSRAM + 4, 0x0000_0001);
    assert_eq!(r.r32(reg::SSRAM + 4), 0x0000_0001 | SSRAM_PARITY_ERROR);
    // Bit 17 asks for generated parity: data 0x0001 needs the parity bit set.
    r.w32(reg::SSRAM + 8, 0x0000_0001 | SSRAM_PARITY_ERROR);
    assert_eq!(r.r32(reg::SSRAM + 8), 0x0001_0001);
}

#[test]
fn reset_clears_chip_state_but_not_the_buffer_ram() {
    let r = Rig::new();
    r.w32(reg::INT_ENABLE_SET, 0xff);
    r.w32(reg::SSRAM, 0x55);
    r.w32(reg::PIN_CONTROL_SET, 1 << 21);
    r.c.reset();
    assert_eq!(r.r32(reg::INT_ENABLE_SET), 0);
    assert_eq!(r.r32(reg::PIN_CONTROL_SET), 0);
    assert_eq!(r.r32(reg::SSRAM), 0x55);
}

// ---------------------------------------------------------------------------
// Review fixes
// ---------------------------------------------------------------------------

/// Three threads change what drives the SuperIO line; once they stop, the
/// line must agree with the status and enable registers.
#[test]
fn the_superio_line_never_ends_up_stale_across_threads() {
    for _ in 0..20 {
        let r = Arc::new(Rig::new());
        r.w32(reg::INT_ENABLE_SET, int::KEYBOARD);
        r.w32(K_TX, 0xf4);
        let _ = r.r32(K_RX);
        let go = |f: Box<dyn Fn() + Send>| std::thread::spawn(f);
        let (a, b, c) = (r.clone(), r.clone(), r.clone());
        let t = [
            go(Box::new(move || {
                for _ in 0..200 {
                    a.c.ps2().push_kb(winit::keyboard::KeyCode::KeyA, true);
                    a.c.ps2().push_kb(winit::keyboard::KeyCode::KeyA, false);
                }
            })),
            go(Box::new(move || {
                for _ in 0..200 {
                    b.w32(reg::INT_STATUS, int::KEYBOARD);
                    let _ = b.r32(K_RX);
                }
            })),
            go(Box::new(move || {
                for _ in 0..200 {
                    c.c.service();
                }
            })),
        ];
        for t in t {
            t.join().unwrap();
        }
        r.c.service();
        let want = r.r32(reg::INT_STATUS) & r.r32(reg::INT_ENABLE_SET) != 0;
        assert_eq!(r.superio.level(), want);
    }
}

/// A line that records every call it gets. The Bridge turns a rise into a
/// packet, so each call must be a change of level: a repeat is an edge the
/// registers never made.
#[derive(Default)]
struct EdgeProbe {
    level: Mutex<bool>,
    rises: std::sync::atomic::AtomicUsize,
    repeats: std::sync::atomic::AtomicUsize,
}

impl crate::irq::InterruptController for EdgeProbe {
    fn set_line(&self, _line: u32, level: bool) {
        let mut now = self.level.lock().unwrap();
        if level == *now {
            self.repeats.fetch_add(1, Ordering::SeqCst);
        } else if level {
            self.rises.fetch_add(1, Ordering::SeqCst);
        }
        *now = level;
    }
}

/// Linux acknowledges a source and then unmasks it, as two writes. Status and
/// enable are one pair under one lock, so the Bridge, which latches a rising
/// edge, is not given one the words never made, whoever else reads or
/// republishes the line meanwhile. The source here is raised only while it is
/// masked, so the output is low for the whole run.
#[test]
fn the_superio_line_never_rises_while_status_and_enable_never_agree() {
    let c = Arc::new(octane());
    let probe = Arc::new(EdgeProbe::default());
    c.connect_superio_interrupt(crate::irq::IrqLine::new(probe.clone(), 0));
    let bit = int::KEYBOARD;
    let done = Arc::new(AtomicBool::new(false));
    let others: Vec<_> = (0..2)
        .map(|_| {
            let (c, done) = (c.clone(), done.clone());
            std::thread::spawn(move || {
                while !done.load(Ordering::SeqCst) {
                    c.sh.core.republish();
                    let _ = c.bar_read(0, reg::INT_STATUS, 4);
                }
            })
        })
        .collect();
    for _ in 0..100_000 {
        c.sh.core.set_bits(bit);
        c.bar_write(0, reg::INT_STATUS, bit, 4);
        c.bar_write(0, reg::INT_ENABLE_SET, bit, 4);
        c.bar_write(0, reg::INT_ENABLE_CLR, bit, 4);
    }
    done.store(true, Ordering::SeqCst);
    for t in others {
        t.join().unwrap();
    }
    assert_eq!(probe.rises.load(Ordering::SeqCst), 0, "a rising edge the words never made");
}

/// Four threads raise, lower, enable and disable one source. The line is
/// driven under the words' lock and only on a change, so it is never given a
/// repeated level (a stale or duplicate edge), rises and falls alternate, and
/// it ends where the words say.
#[test]
fn the_superio_line_is_driven_only_on_change_and_in_order_across_threads() {
    for _ in 0..20 {
        let c = Arc::new(octane());
        let probe = Arc::new(EdgeProbe::default());
        c.connect_superio_interrupt(crate::irq::IrqLine::new(probe.clone(), 0));
        // Connecting states the line's level once, changed or not.
        probe.repeats.store(0, Ordering::SeqCst);
        let bit = int::KEYBOARD;
        let ops: [fn(&Core, u32); 4] = [Core::set_bits, Core::clear_bits, Core::enable, Core::disable];
        let threads: Vec<_> = ops
            .into_iter()
            .map(|op| {
                let c = c.clone();
                std::thread::spawn(move || {
                    for _ in 0..5_000 {
                        op(&c.sh.core, bit);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(probe.repeats.load(Ordering::SeqCst), 0, "a level repeated to the parent");
        let want = c.sh.core.status() & c.sh.core.enabled() != 0;
        assert_eq!(*probe.level.lock().unwrap(), want);
    }
}

/// A host key and writes of the enable words race; once both finish, the
/// line agrees with status AND enable.
#[test]
fn the_superio_line_follows_enable_writes_racing_a_host_key() {
    for _ in 0..20 {
        let r = Arc::new(Rig::new());
        r.w32(K_TX, 0xf4);
        let _ = r.r32(K_RX);
        let (a, b) = (r.clone(), r.clone());
        let keys = std::thread::spawn(move || {
            for _ in 0..200 {
                a.c.ps2().push_kb(winit::keyboard::KeyCode::KeyA, true);
                a.c.ps2().push_kb(winit::keyboard::KeyCode::KeyA, false);
            }
        });
        let enables = std::thread::spawn(move || {
            for _ in 0..400 {
                b.w32(reg::INT_ENABLE_SET, int::KEYBOARD);
                b.w32(reg::INT_ENABLE_CLR, int::KEYBOARD);
            }
        });
        keys.join().unwrap();
        enables.join().unwrap();
        r.c.service();
        let want = r.r32(reg::INT_STATUS) & r.r32(reg::INT_ENABLE_SET) != 0;
        assert_eq!(r.superio.level(), want);
    }
}

#[test]
fn the_ethernet_pin_follows_the_last_level_given() {
    let c = octane();
    c.state().connect_dma(TestMem::new());
    let (pin, line) = LevelProbe::line();
    c.state().connect_irq(line);
    c.attached();
    let eth = c.eth().unwrap();
    eth.write32(0xf8, 1);
    for _ in 0..1000 {
        eth.raise_for_test(1);
        eth.write32(0xf4, 1);
    }
    eth.raise_for_test(1);
    assert!(pin.level());
}

#[test]
fn a_ring_at_the_top_of_the_address_space_is_a_dma_error_not_a_panic() {
    let r = Rig::new();
    r.aim_engines();
    r.w32(reg::SDMA_RING_HI, 0xffff_ffff);
    r.w32(reg::SDMA_RING_LO, 0xffff_f000 | 1);
    r.w32(A, sdma_ctl::ENABLE);
    r.w32(A + sdma_off::TX_PRODUCE, 8);
    assert_ne!(r.r32(reg::INT_STATUS) & int::serial(0, int::DMA_ERROR), 0);
    let d = r.mem_view();
    let mut b = [0u8; 8];
    assert!(!d.dma_read(u64::MAX - 2, &mut b).is_ok());
    assert!(!d.dma_write(u64::MAX - 2, &b).is_ok());
}

#[test]
fn an_unaligned_dma_write_keeps_its_neighbours_in_one_transfer() {
    let r = Rig::new();
    let d = r.mem_view();
    put(&r, 0x100, &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(d.dma_write(0x101, &[0xaa, 0xbb, 0xcc]).is_ok());
    assert_eq!(get(&r, 0x100, 8), [1, 0xaa, 0xbb, 0xcc, 5, 6, 7, 8]);
}

/// The 32-bit blocks refuse narrower accesses: they read zero and a write
/// does nothing.
#[test]
fn narrow_accesses_to_32_bit_registers_are_refused() {
    let r = Rig::new();
    r.w32(reg::INT_ENABLE_SET, 0x1234_5678);
    assert_eq!(r.r8(reg::INT_ENABLE_SET), 0);
    assert_eq!(r.c.bar_read(0, reg::INT_ENABLE_SET, 2), 0);
    r.w8(reg::INT_ENABLE_SET, 0xff);
    assert_eq!(r.r32(reg::INT_ENABLE_SET), 0x1234_5678);
    // The byte-wide windows are unaffected.
    r.w8(reg::SERIAL_A + 7, 0x77);
    assert_eq!(r.r8(reg::SERIAL_A + 7), 0x77);
}

// ---------------------------------------------------------------------------
// DS1685 extended RAM
// ---------------------------------------------------------------------------

fn rtc_w(r: &Rig, reg: u8, v: u8) {
    r.w8(reg::BYTEBUS_DEV1, reg);
    r.w8(reg::BYTEBUS_DEV2, v);
}

fn rtc_r(r: &Rig, reg: u8) -> u8 {
    r.w8(reg::BYTEBUS_DEV1, reg);
    r.r8(reg::BYTEBUS_DEV2)
}

/// The bank bit switches only 0x40..0x7f; the low registers and RAM are the
/// same in both banks.
#[test]
fn the_bank_bit_switches_only_the_upper_half() {
    let r = Rig::new();
    rtc_w(&r, 0x20, 0x11);
    rtc_w(&r, 0x60, 0x22); // bank 0 RAM
    rtc_w(&r, 0x0a, 0x10); // bank 1
    assert_eq!(rtc_r(&r, 0x20), 0x11, "low RAM is in both banks");
    assert_ne!(rtc_r(&r, 0x60), 0x22, "bank 1 has its own cell");
    rtc_w(&r, 0x60, 0x33);
    rtc_w(&r, 0x0a, 0x00);
    assert_eq!(rtc_r(&r, 0x60), 0x22, "bank 0 kept its value");
    rtc_w(&r, 0x0a, 0x10);
    assert_eq!(rtc_r(&r, 0x60), 0x33);
}

#[test]
fn the_extended_ram_is_reached_through_its_address_and_data_ports() {
    let r = Rig::new();
    rtc_w(&r, 0x0a, 0x10);
    for n in 0..128u8 {
        rtc_w(&r, 0x50, n);
        rtc_w(&r, 0x53, n ^ 0xa5);
    }
    for n in 0..128u8 {
        rtc_w(&r, 0x50, n);
        assert_eq!(rtc_r(&r, 0x53), n ^ 0xa5);
        assert_eq!(rtc_r(&r, 0x53), n ^ 0xa5, "the address does not advance");
    }
    // Seven-bit address: 0x80 wraps to cell 0.
    rtc_w(&r, 0x50, 0x80);
    assert_eq!(rtc_r(&r, 0x53), 0xa5);
    // In bank 0, register 0x53 is plain RAM, not the port.
    rtc_w(&r, 0x0a, 0x00);
    assert_eq!(rtc_r(&r, 0x53), 0);
}

#[test]
fn nvram_and_extended_ram_survive_in_a_512_byte_file() {
    let path = std::env::temp_dir().join(format!("ioc3-xram-{}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    {
        let r = Rig::new();
        r.c.rtc().set_nvram_file(&path).unwrap();
        rtc_w(&r, 0x0f, 0x5c);
        rtc_w(&r, 0x45, 0x66); // bank 0 RAM
        rtc_w(&r, 0x0a, 0x10);
        rtc_w(&r, 0x45, 0x77); // bank 1 register
        rtc_w(&r, 0x50, 0x09);
        rtc_w(&r, 0x53, b'1');
        rtc_w(&r, 0x50, 0x01);
        rtc_w(&r, 0x53, 0x02);
    }
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0x200);
    {
        let r = Rig::new();
        r.c.rtc().set_nvram_file(&path).unwrap();
        assert_eq!(rtc_r(&r, 0x0f), 0x5c);
        assert_eq!(rtc_r(&r, 0x45), 0x66);
        rtc_w(&r, 0x0a, 0x10);
        assert_eq!(rtc_r(&r, 0x45), 0x77);
        rtc_w(&r, 0x50, 0x09);
        assert_eq!(rtc_r(&r, 0x53), b'1');
        rtc_w(&r, 0x50, 0x01);
        assert_eq!(rtc_r(&r, 0x53), 0x02);
    }
    let _ = std::fs::remove_file(&path);
}

/// A full image in the QEMU layout loads, extended RAM included.
#[test]
fn a_qemu_layout_image_loads() {
    let path = std::env::temp_dir().join(format!("ioc3-qimg-{}", std::process::id()));
    let mut img = vec![0u8; 0x200];
    img[0x0f] = 0x42;
    img[0xc0 + 9] = b'1';
    img[0xc0 + 1] = 0x02;
    std::fs::write(&path, &img).unwrap();
    let r = Rig::new();
    r.c.rtc().set_nvram_file(&path).unwrap();
    assert_eq!(rtc_r(&r, 0x0f), 0x42);
    rtc_w(&r, 0x0a, 0x10);
    rtc_w(&r, 0x50, 9);
    assert_eq!(rtc_r(&r, 0x53), b'1');
    rtc_w(&r, 0x50, 1);
    assert_eq!(rtc_r(&r, 0x53), 0x02);
    // With no clock reference beside it, the clock starts at host time.
    rtc_w(&r, 0x0a, 0x20);
    assert!(rtc_r(&r, 0x09) >= 0x26, "host time, in BCD");
    let _ = std::fs::remove_file(&path);
}

/// Set 2004-06-01 19:00:30 UTC in IRIX's order: SET, the seconds through
/// the year, the century in bank 1, the weekday, SET clear. Binary, 24-hour.
fn set_2004(r: &Rig) {
    rtc_w(r, 0x0a, 0x30);
    rtc_w(r, 0x0b, 0x86);
    for (n, v) in [(0u8, 30u8), (2, 0), (4, 19), (7, 1), (8, 6), (9, 4), (0x48, 20), (6, 3)] {
        rtc_w(r, n, v);
    }
    rtc_w(r, 0x0b, 0x06);
    rtc_w(r, 0x0a, 0x20);
}

/// A date the guest sets survives a relaunch, still running: the clock
/// reference beside the file, in QEMU's layout, carries it.
#[test]
fn a_set_date_survives_a_relaunch() {
    let path = std::env::temp_dir().join(format!("ioc3-clock-{}", std::process::id()));
    let clock = crate::dev::battery_clock::companion(&path);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&clock);
    {
        let r = Rig::new();
        r.c.rtc().set_nvram_file(&path).unwrap();
        assert!(!clock.exists(), "no reference until the guest sets the clock");
        set_2004(&r);
    }
    assert_eq!(std::fs::metadata(&path).unwrap().len(), 0x200, "the image stays the chip's bytes");
    let reference = std::fs::read(&clock).unwrap();
    assert_eq!((reference.len(), &reference[..8]), (24, &b"DS168RF1"[..]));
    {
        let r = Rig::new();
        r.c.rtc().set_rtc_offset(&crate::config::RtcOffset { years: -5, ..Default::default() });
        r.c.rtc().set_nvram_file(&path).unwrap();
        // A new run: register B is the part's power-up BCD, 24-hour.
        rtc_w(&r, 0x0a, 0x30);
        assert_eq!(rtc_r(&r, 0x48), 0x20);
        assert_eq!(
            [0x09, 0x08, 0x07, 0x04, 0x02].map(|n| rtc_r(&r, n)),
            [0x04, 0x06, 0x01, 0x19, 0x00],
            "2004-06-01 19:00, not host time or the seed"
        );
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&clock);
}

/// A clock file that cannot be read is reported and passed over: the
/// NVRAM file still loads and the clock starts from host time.
#[test]
fn an_unreadable_clock_file_leaves_the_nvram_usable() {
    let path = std::env::temp_dir().join(format!("ioc3-clock-bad-{}", std::process::id()));
    let clock = crate::dev::battery_clock::companion(&path);
    let mut img = vec![0u8; 0x200];
    img[0x0f] = 0x42;
    std::fs::write(&path, &img).unwrap();
    let _ = std::fs::remove_dir_all(&clock);
    std::fs::create_dir(&clock).unwrap(); // reading a directory fails
    let r = Rig::new();
    r.c.rtc().set_nvram_file(&path).unwrap();
    assert_eq!(rtc_r(&r, 0x0f), 0x42);
    assert!(rtc_r(&r, 0x09) >= 0x26, "host time, in BCD");
    drop(r);
    let _ = std::fs::remove_dir_all(&clock);
    let _ = std::fs::remove_file(&path);
}

/// `[rtc_offset]` starts a clock the guest has not set, at every run, so an
/// edit to it takes effect.
#[test]
fn rtc_offset_starts_a_clock_with_no_reference() {
    let path = std::env::temp_dir().join(format!("ioc3-clock-seed-{}", std::process::id()));
    let clock = crate::dev::battery_clock::companion(&path);
    std::fs::write(&path, vec![0u8; 0x200]).unwrap();
    let _ = std::fs::remove_file(&clock);
    let year = |years| {
        let r = Rig::new();
        r.c.rtc().set_rtc_offset(&crate::config::RtcOffset { years, ..Default::default() });
        r.c.rtc().set_nvram_file(&path).unwrap();
        let y = rtc_r(&r, 0x09);
        drop(r);
        assert!(!clock.exists(), "a seed writes no reference");
        i32::from((y >> 4) * 10 + (y & 0x0f))
    };
    let (now, back10, back20) = (year(0), year(-10), year(-20));
    assert_eq!((now - back10).rem_euclid(100), 10);
    assert_eq!((back10 - back20).rem_euclid(100), 10, "the edited offset took effect");
    let _ = std::fs::remove_file(&path);
}

// ---------------------------------------------------------------------------
// Lifecycle: receive threads and timers with no guest access
// ---------------------------------------------------------------------------

/// A running chip with a timer manager, for the tests where nothing touches
/// a register while the chip works.
fn running_rig() -> (Rig, Arc<TimerManager>) {
    let r = Rig::new();
    let tm = Arc::new(TimerManager::new());
    r.c.set_timer_manager(tm.clone());
    (r, tm)
}

/// Whether a timer is pending for deadline `kind`, read while no timer
/// callback is running. A callback clears its slot on entry and arms the
/// next deadline before it returns, so between those a chip that holds a
/// deadline briefly shows none; the callback holds `gate` throughout.
fn timer_pending(r: &Rig, kind: usize) -> bool {
    let _idle = r.c.sh.gate.lock();
    r.c.sh.armed.lock()[kind].is_some()
}

/// Typed input reaches the ring and the receive timer posts it and raises its
/// line with no register access at all: the receive thread feeds the FIFO and
/// a timer runs the expiry. This guards the removal of the polling thread.
#[test]
fn the_receive_timer_fires_with_no_guest_access() {
    let (r, _tm) = running_rig();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    r.w32(A + sdma_off::RX_TIMER, 0x100);
    r.w32(A + sdma_off::RX_CONSUME, ARM_RX_TIMER);
    r.w32(A, sdma_ctl::ENABLE | 4);
    r.w32(reg::INT_ENABLE_SET, int::serial(0, int::RX_TIMER));
    r.c.start();
    r.wire[0].inp.lock().unwrap().extend(b"hi");
    assert!(eventually(|| r.superio.level()), "the expiry raised the line");
    // `int_status` is a plain read of the latch: it catches nothing up.
    assert_ne!(r.c.int_status() & int::serial(0, int::RX_TIMER), 0);
    assert_eq!(&get(&r, 0x3000, 2), b"hi");
    r.c.stop();
}

/// The interrupt output's pulse reaches the line from a timer, not from a
/// read.
#[test]
fn the_interrupt_output_pulses_with_no_guest_access() {
    let (r, _tm) = running_rig();
    r.c.start();
    r.w32(reg::INT_ENABLE_SET, int::RT_OUTPUT_PULSE);
    r.w32(reg::INT_OUTPUT, IO_PERIODIC | 100);
    assert!(eventually(|| r.superio.level()));
    assert_ne!(r.c.int_status() & int::RT_OUTPUT_PULSE, 0);
    r.c.stop();
}

/// A transmit delay in the ring ends, and its explicit interrupt rises, from
/// a timer.
#[test]
fn a_transmit_delay_ends_with_no_guest_access() {
    let (r, _tm) = running_rig();
    r.aim_engines();
    r.w32(reg::SDMA_RING_LO, 0x2000 | 1);
    r.c.start();
    // One entry: a 5 ms delay carrying the explicit interrupt flag.
    put(&r, 0x2000, &[5, 0, 0, 0, TX_DELAY | TX_INTERRUPT, 0, 0, 0]);
    r.w32(reg::INT_ENABLE_SET, int::serial(0, int::TX_EXPLICIT));
    r.w32(A, sdma_ctl::ENABLE);
    {
        // Holding the callback gate keeps the timer from ending the delay
        // while the test is preempted past 5 ms between write and read.
        let _idle = r.c.sh.gate.lock();
        r.w32(A + sdma_off::TX_PRODUCE, 8);
        assert!(!r.superio.level(), "the delay has not ended inside the write");
    }
    assert!(eventually(|| r.superio.level()));
    r.c.stop();
}

/// Poll `done` until it holds, giving up after a minute. The timer and the
/// NVRAM writer run on other threads and the suite runs many tests at once, so
/// a bound counted in milliseconds gives up on a busy host before either has
/// had a core; the deadline only has to catch a write that never comes.
fn eventually(mut done: impl FnMut() -> bool) -> bool {
    let end = Instant::now() + Duration::from_secs(60);
    while !done() {
        if Instant::now() >= end {
            return false;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    true
}

/// A guest write to the clock's RAM leaves the file alone: it only marks the
/// RAM changed, for the debounce to save later. The chip is not started, so
/// no timer or writer exists that could save it behind the test's back; with
/// them running, a read of the file "before the debounce" would race the
/// debounce itself on a slow host.
#[test]
fn an_nvram_write_marks_the_ram_dirty_and_leaves_the_file_alone() {
    let path = std::env::temp_dir().join(format!("ioc3-nvram-perbyte-{}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let (r, _tm) = running_rig();
    r.c.rtc().set_nvram_file(&path).unwrap();
    rtc_w(&r, 0x45, 0x66);
    assert_eq!(std::fs::read(&path).unwrap()[NVRAM_OFF_NV + 5], 0, "not written per byte");
    assert!(r.c.rtc().dirty_since().is_some(), "the RAM counts as unsaved");
    let _ = std::fs::remove_file(&path);
}

/// The file catches up once the guest's NVRAM writes have paused, written by
/// the chip's own writer with nobody calling a flush.
#[test]
fn nvram_reaches_the_file_after_the_debounce_with_no_flush() {
    let path = std::env::temp_dir().join(format!("ioc3-nvram-timer-{}", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let (r, _tm) = running_rig();
    r.c.rtc().set_nvram_file(&path).unwrap();
    r.c.start();
    rtc_w(&r, 0x45, 0x66);
    let in_file = || std::fs::read(&path).map(|f| f[NVRAM_OFF_NV + 5]).unwrap_or(0);
    assert!(eventually(|| in_file() == 0x66), "the writer saved the RAM");
    r.c.stop();
    let _ = std::fs::remove_file(&path);
}

/// `stop` leaves no thread, no timer and no host input taken; `start` brings
/// them back, and both can be repeated.
#[test]
fn stop_joins_the_threads_and_removes_the_timers() {
    let (r, _tm) = running_rig();
    assert!(!r.c.is_running());
    r.c.start();
    r.c.start();
    assert!(r.c.is_running());
    assert_eq!(r.c.threads.lock().len(), 2, "two receive threads");
    assert!(r.c.rtc().is_running(), "and the clock's NVRAM writer");
    r.w32(reg::INT_ENABLE_SET, int::RT_OUTPUT_PULSE);
    r.w32(reg::INT_OUTPUT, IO_PERIODIC | 1000);
    assert!(timer_pending(&r, K_INT_OUT));
    r.c.stop();
    r.c.stop();
    assert!(!r.c.is_running());
    assert!(r.c.threads.lock().is_empty());
    assert!(!r.c.rtc().is_running());
    assert!(r.c.sh.armed.lock().iter().all(|a| a.is_none()));
    // Typed input waits in the backend while stopped.
    r.wire[0].inp.lock().unwrap().extend(b"x");
    std::thread::sleep(Duration::from_millis(20));
    assert!(r.c.rx_pending(0) == 0);
    r.c.start();
    assert!(eventually(|| r.c.rx_pending(0) != 0));
    assert!(timer_pending(&r, K_INT_OUT), "start arms the deadlines the chip holds");
    r.c.stop();
}

/// A repeating output keeps a timer pending whenever no callback is in the
/// middle of re-arming it, however often the callbacks run.
#[test]
fn a_periodic_output_always_has_a_timer_pending_between_callbacks() {
    let (r, _tm) = running_rig();
    r.c.start();
    r.w32(reg::INT_ENABLE_SET, int::RT_OUTPUT_PULSE);
    r.w32(reg::INT_OUTPUT, IO_PERIODIC | 1000);
    for _ in 0..20_000 {
        assert!(timer_pending(&r, K_INT_OUT));
    }
    r.c.stop();
}

/// A pass that read the chip's deadlines before a register write must not arm
/// from them after it. A thread re-arms as the receive polls do while the
/// test rewrites the interrupt output, and after each write has returned the
/// timer slot must serve the chip's deadline. A stale pass would put the older
/// deadline back, or a timer for an output the write had just cleared.
#[test]
fn a_pass_that_read_the_chip_first_arms_first() {
    let (r, _tm) = running_rig();
    r.c.start();
    r.w32(reg::INT_ENABLE_SET, int::RT_OUTPUT_PULSE);
    let sh = &r.c.sh;
    // A callback re-arming on its own would blur what the slot should hold.
    let idle = sh.gate.lock();
    let stop = AtomicBool::new(false);
    std::thread::scope(|s| {
        let poller = s.spawn(|| {
            while !stop.load(Ordering::SeqCst) {
                sh.rearm();
                std::thread::yield_now();
            }
        });
        for i in 0..5_000u32 {
            r.w32(reg::INT_OUTPUT, if i % 2 == 0 { IO_PERIODIC | (0x1000 + i) } else { 0 });
            let held = sh.armed.lock()[K_INT_OUT].map(|(_, d)| d);
            let want = decode(sh.io_due.load(Ordering::SeqCst));
            if held != want {
                stop.store(true, Ordering::SeqCst);
                poller.join().unwrap();
                panic!("write {i} left the slot at {held:?}, the chip holds {want:?}");
            }
        }
        stop.store(true, Ordering::SeqCst);
        poller.join().unwrap();
    });
    // `stop` waits out a callback, so it cannot run while the gate is held.
    drop(idle);
    r.c.stop();
}

/// Dropping a running chip joins its threads; they hold it only weakly.
#[test]
fn dropping_a_running_chip_ends_its_threads() {
    let (r, _tm) = running_rig();
    r.c.start();
    let weak = Arc::downgrade(&r.c.sh);
    drop(r);
    assert!(weak.upgrade().is_none());
}

/// The clock's power-off bit tells the machine once, through its event sender.
#[test]
fn the_clocks_power_off_is_sent_to_the_machine_once() {
    let (tx, rx) = std::sync::mpsc::sync_channel(4);
    let r = Rig::with(Ioc3::new(&IP30, MAC, ByteBus::Ds1685(Ds1685::new(Some(tx)))));
    rtc_w(&r, 0x0a, 0x10); // bank 1
    assert!(rx.try_recv().is_err());
    rtc_w(&r, 0x4a, 0x08);
    assert!(matches!(rx.try_recv(), Ok(MachineEvent::PowerOff)));
    rtc_w(&r, 0x4a, 0x08);
    assert!(rx.try_recv().is_err(), "an edge, not a level");
}

// ---------------------------------------------------------------------------
// Register-level cases from the first Octane work, through the bridge's byte
// lanes
// ---------------------------------------------------------------------------

#[test]
fn rtc_nvram_and_extended_ram_hold_what_is_written() {
    let r = Rig::new();
    rtc_w(&r, 0x0e, 0x5a);
    assert_eq!(rtc_r(&r, 0x0e), 0x5a);
    assert_eq!(rtc_r(&r, 0x0d), 0x80, "valid RAM and time");
    rtc_w(&r, 0x0a, 0x30); // bank 1
    rtc_w(&r, 0x50, 0x07);
    rtc_w(&r, 0x53, 0xa5);
    rtc_w(&r, 0x50, 0x07);
    assert_eq!(rtc_r(&r, 0x53), 0xa5);
    assert!(rtc_r(&r, 0x00) < 0x60, "BCD seconds");
}

#[test]
fn the_uarts_are_in_the_superio_block() {
    let r = Rig::new();
    assert_eq!(r.r8(reg::SERIAL_A + 5) & 0x20, 0x20, "ready to transmit");
    r.w8(reg::SERIAL_A + 7, 0x5a);
    assert_eq!(r.r8(reg::SERIAL_A + 7), 0x5a);
    assert_eq!(r.r8(reg::SERIAL_B + 7), 0);
    assert_eq!(r.r32(reg::PIN_DATA) >> 2 & 1, 1, "an original Octane");
}

// ---------------------------------------------------------------------------
// Snapshots
// ---------------------------------------------------------------------------

/// The saved clock's seconds: the chip's calendar runs on in real time, so
/// two saves a host stall apart differ in it and in nothing else.
fn clock_seconds(v: &toml::Value) -> i64 {
    v["chip"]["rtc"]["clock"]["time"].as_integer().unwrap()
}

/// `v` with the running clock's seconds cleared, for comparing the rest.
fn without_clock(v: &toml::Value) -> toml::Value {
    let mut v = v.clone();
    v["chip"]["rtc"]["clock"].as_table_mut().unwrap().remove("time");
    v
}

#[test]
fn save_load_round_trip() {
    for build in [octane] {
        let src = Rig::with(build());
        src.w8(reg::SERIAL_A + 3, 0x80);
        src.w8(reg::SERIAL_A + 7, 6);
        src.w8(reg::SERIAL_A + 3, 0x03);
        src.w32(reg::SSRAM + 0x100, 0x1234);
        src.w32(reg::INT_ENABLE_SET, 0x40);
        let v1 = src.c.save_state();

        let dst = Rig::with(build());
        dst.c.load_state(&v1).expect("load_state");
        // Seconds may pass between the load and the save, as on a loaded host.
        dst.c.sh.rtc.as_ref().unwrap().advance_clock(2);
        let v2 = dst.c.save_state();
        assert_eq!(without_clock(&v1), without_clock(&v2));
        assert_eq!(clock_seconds(&v2), clock_seconds(&v1) + 2, "the restored clock runs on from the saved time");
        assert_eq!(dst.c.predivider(0), 6);
        assert_eq!(dst.c.sh.ssram.lock()[0x40] & SSRAM_DATA, 0x1234);
    }
}

/// A restore into a chip that has been in use equals one into a fresh chip.
#[test]
fn ioc3_load_into_a_dirtied_instance_equals_load_into_a_fresh_one() {
    let dirty = |r: &Rig, salt: u32| {
        r.w8(reg::SERIAL_A + 3, 0x80);
        r.w8(reg::SERIAL_A + 7, 4 + salt as u8);
        r.w8(reg::SERIAL_A + 3, 0x03);
        r.w32(reg::SSRAM + 0x100 * salt, 0x1000 + salt);
        r.w32(reg::INT_ENABLE_SET, 0x40 << salt);
    };
    let src = Rig::new();
    dirty(&src, 1);
    let v = src.c.save_state();
    let used = Rig::new();
    dirty(&used, 2);
    let fresh = Rig::new();
    used.c.load_state(&v).unwrap();
    fresh.c.load_state(&v).unwrap();
    // The loads and saves are not at one instant, so the clock may have moved.
    used.c.sh.rtc.as_ref().unwrap().advance_clock(1);
    assert_eq!(without_clock(&used.c.save_state()), without_clock(&fresh.c.save_state()));
    assert_eq!(without_clock(&used.c.save_state()), without_clock(&v));
}

/// A periodic interrupt output keeps the time its next pulse had left,
/// counted from the load rather than from the save.
#[test]
fn a_pending_output_keeps_its_time_left() {
    let src = octane();
    src.bar_write(0, reg::INT_OUTPUT, IO_PERIODIC | 100, 4);
    let at_save = Instant::now();
    let left = Duration::from_micros(20);
    let v = {
        src.sh.int_out.lock().next = Some(at_save + left);
        src.sh.save_state(at_save)
    };
    let dst = octane();
    let at_load = at_save + Duration::from_millis(5);
    dst.sh.load_state(&v, at_load).unwrap();
    assert_eq!(dst.sh.int_out.lock().next, Some(at_load + left));
}

#[test]
fn a_short_buffer_ram_is_refused() {
    let c = octane();
    let mut v = c.save_state();
    let chip = v.as_table_mut().unwrap().get_mut("chip").unwrap().as_table_mut().unwrap();
    chip.insert("ssram".into(), toml::Value::Array(Vec::new()));
    assert!(c.load_state(&v).unwrap_err().contains("ssram"));
}

/// The Octane keeps the IOC3's PCI header across a chip reset; only the
/// Fuel population resets it along with the chip.
#[test]
fn an_octane_chip_reset_keeps_the_pci_header() {
    let r = Rig::new();
    let cfg = r.c.state().config();
    cfg.write(1, 0xffff_0146);
    cfg.write(3, 0xffff_ffff);
    r.c.reset();
    assert_eq!(cfg.read(1), 0x0200_0146);
    assert_eq!(cfg.read(3), 0xffff);
}

/// Fuel's IOC3 has its own header write policy and resets the header with
/// the chip, reaches the flat Timekeeper on byte-bus device 0, leaves the
/// Dallas clock ports unfitted, and has no PHY reset on GPIO 5.
#[test]
fn the_fuel_population_keeps_its_clock_and_header_policy_across_reset() {
    let clock = Arc::new(crate::dev::timekeeper::Timekeeper::at(2026, 1, 1, 0, 0, 0).unwrap());
    let r = Rig::with(Ioc3::new(&IP34, MAC, ByteBus::M48t35(clock.clone())));
    let cfg = r.c.state().config();
    cfg.write(1, 0xffff_0146);
    cfg.write(3, 0xffff_ffff);
    assert_eq!(cfg.read(1), 0x0280_0146);
    assert_eq!(cfg.read(3), 0x0000_ff00);
    let base = reg::BYTEBUS_DEV0;
    r.w8(base + 0x7f8, 0xa5);
    r.c.reset();
    clock.power_on();
    assert_eq!(r.r8(base + 0x7fff), 0x58);
    assert_eq!(r.r8(base + 0x7f8), 0xa5);
    assert_eq!(cfg.read(1), 0x0280_0000);
    assert_eq!(cfg.read(3), 0);
    // Fuel selects the flat clock, leaving indexed Dallas ports unfitted.
    r.w8(reg::BYTEBUS_DEV1, 0x40);
    r.w8(reg::BYTEBUS_DEV2, 0x55);
    assert_eq!(r.r8(reg::BYTEBUS_DEV2), 0);
    use crate::dev::ioc3::eth::{bits::MII_READ, reg::{MII_CONTROL, MII_DATA_R, MII_DATA_W}};
    r.w32(MII_DATA_W, 7);
    r.w32(MII_CONTROL, (1 << 5) | 0x19);
    r.w32(reg::PIN_CONTROL_SET, 1 << 5);
    r.w32(reg::PIN_REG_FIRST + 5 * 4, 0);
    r.w32(reg::PIN_DATA, 1 << 5);
    r.w32(MII_CONTROL, MII_READ | (7 << 5) | 3);
    assert_eq!(r.r32(MII_DATA_R), 0x5c10, "Fuel GPIO5 has no PHY reset connection");
}

/// Fuel's IOC3 fits the keyboard and mouse controller, as the Octane's does
/// (OpenBSD configures it through the onboard IOC3's keyboard controller). A
/// key pushed from the host arrives in K_RX through the IOC3 registers and
/// latches the keyboard interrupt, and a mouse report arrives in M_RX; both
/// survive a chip reset.
#[test]
fn the_fuel_ioc3_delivers_keyboard_and_mouse_traffic_across_reset() {
    let clock = Arc::new(crate::dev::timekeeper::Timekeeper::at(2026, 1, 1, 0, 0, 0).unwrap());
    let r = Rig::with(Ioc3::new(&IP34, MAC, ByteBus::M48t35(clock)));
    for _ in 0..2 {
        r.c.reset();
        r.c.ps2().power_on();
        r.w32(K_TX, 0xff);
        let v = r.r32(K_RX);
        assert_ne!(v & VALID[2], 0, "the keyboard answers a reset");
        assert_eq!((v >> 16) & 0xff, 0xfa, "with an ACK");
        r.w32(M_TX, 0xf2);
        let v = r.r32(M_RX);
        assert_ne!(v & (1 << 27), 0, "the mouse answers from its own port");
        // Enable scanning, then inject a key from the host.
        r.w32(K_TX, 0xf4);
        while r.r32(K_RX) & VALID[2] != 0 {}
        r.w32(reg::INT_STATUS, int::KEYBOARD);
        r.c.ps2().push_kb(winit::keyboard::KeyCode::KeyA, true);
        assert_ne!(r.r32(reg::INT_STATUS) & int::KEYBOARD, 0, "a key latches the interrupt");
        let v = r.r32(K_RX);
        assert_ne!(v & VALID[2], 0, "and arrives in K_RX");
        assert_eq!(v & (1 << 27), 0, "from the keyboard");
    }
}
