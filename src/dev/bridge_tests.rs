//! Register-contract tests for the Bridge, against fake functions, memory
//! and an interrupt sink. They follow the reference model's register-contract tests: configuration
//! space with narrow accesses, an empty slot, one register reached through
//! both BARs, the programmed byte lanes of a device window, and a function's
//! DMA reaching memory through the direct map and the translation table.
//!
//! Ported from tests/qtest/sgi-bridge-bus-test.c in jamesbraid/qemu-sgi-origami.

use super::*;
use crate::ip30::{HEART_HOST, WIDGET_HEART};
use crate::irq::IrqLine;
use crate::pci::PciToPciBridge;
use crate::pci::{Bar, BoardIrq, DevFn, PciConfig, PciFunction, PciFunctionState, PCI_COMMAND_IO, PCI_COMMAND_MEMORY, PCI_INTA};
use crate::xtalk::{Upstream, XtalkHost};

/// HEART stand-in: the interrupt packets it received, in order. A line is
/// the ISR bit, so `set_line(vector, set)` is a packet.
#[derive(Default)]
struct Sink(Mutex<Vec<(u8, bool)>>);

impl InterruptController for Sink {
    fn set_line(&self, line: u32, level: bool) {
        self.0.lock().push((line as u8, level));
    }
}

impl Sink {
    fn take(&self) -> Vec<(u8, bool)> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// Memory at physical 0 and at 512 MB, 1 MB each; the rest is not memory.
/// Big-endian like the bus: `read32` at `a` is the bytes `a..a + 4`.
struct Mem(Mutex<[Vec<u8>; 2]>);

const HIGH: u64 = 0x2000_0000;

impl Mem {
    fn new() -> Self {
        Mem(Mutex::new([vec![0; 0x10_0000], vec![0; 0x10_0000]]))
    }

    /// The region and offset of `n` bytes at `a`, all in one region.
    fn locate(a: u64, n: u64) -> Option<(usize, usize)> {
        if a + n <= 0x10_0000 {
            Some((0, a as usize))
        } else if a >= HIGH && a + n <= HIGH + 0x10_0000 {
            Some((1, (a - HIGH) as usize))
        } else {
            None
        }
    }

    fn peek(&self, a: u64, n: usize) -> Vec<u8> {
        let m = self.0.lock();
        let (r, o) = Self::locate(a, n as u64).unwrap();
        m[r][o..o + n].to_vec()
    }

    fn poke(&self, a: u64, data: &[u8]) {
        let mut m = self.0.lock();
        let (r, o) = Self::locate(a, data.len() as u64).unwrap();
        m[r][o..o + data.len()].copy_from_slice(data);
    }
}

impl BusDevice for Mem {
    fn read8(&self, a: u64) -> BusRead8 {
        Self::locate(a, 1).map_or(BusRead8::err(), |(r, o)| BusRead8::ok(self.0.lock()[r][o]))
    }

    fn write8(&self, a: u64, v: u8) -> u32 {
        Self::locate(a, 1).map_or(BUS_ERR, |(r, o)| {
            self.0.lock()[r][o] = v;
            BUS_OK
        })
    }

    fn read32(&self, a: u64) -> BusRead32 {
        Self::locate(a, 4).map_or(BusRead32::err(), |(r, o)| {
            BusRead32::ok(u32::from_be_bytes(self.0.lock()[r][o..o + 4].try_into().unwrap()))
        })
    }

    fn write32(&self, a: u64, v: u32) -> u32 {
        Self::locate(a, 4).map_or(BUS_ERR, |(r, o)| {
            self.0.lock()[r][o..o + 4].copy_from_slice(&v.to_be_bytes());
            BUS_OK
        })
    }
}

const CARD_VENDOR: u32 = 0x1000;
const CARD_DEVICE: u32 = 0x0012;
const SCRATCH: u32 = 0x34;
const FIXED: u32 = 0x20;

/// A function with an I/O BAR and a memory BAR onto one 4 KB register file
/// that stores little-endian bytes, a fixed word at 0x20, an interrupt line
/// and the DMA it was given.
struct Card {
    pci: PciFunctionState,
    regs: Mutex<[u8; 0x1000]>,
}

impl Card {
    fn new(pin: u8) -> Arc<Self> {
        let mut regs = [0u8; 0x1000];
        regs[FIXED as usize..FIXED as usize + 4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        Arc::new(Self {
            pci: PciFunctionState::new(PciConfig::new(
                CARD_VENDOR as u16,
                CARD_DEVICE as u16,
                0x01_0000,
                1,
                [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                pin,
            )),
            regs: Mutex::new(regs),
        })
    }

    fn peek32(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.regs.lock()[off..off + 4].try_into().unwrap())
    }

    fn raise(&self, level: bool) {
        self.pci.irq().unwrap().set(level);
    }

    fn dma(&self) -> Arc<dyn PciDma> {
        self.pci.dma().unwrap().clone()
    }
}

impl PciFunction for Card {
    fn state(&self) -> &PciFunctionState {
        &self.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        let r = self.regs.lock();
        let mut b = [0u8; 4];
        b[..width].copy_from_slice(&r[off as usize..off as usize + width]);
        u32::from_le_bytes(b)
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.regs.lock()[off as usize..off as usize + width].copy_from_slice(&val.to_le_bytes()[..width]);
    }

}

struct Rig {
    bridge: Arc<Bridge>,
    sink: Arc<Sink>,
    mem: Arc<Mem>,
}

impl Rig {
    fn new() -> Self {
        Self::with_host(HEART_HOST)
    }

    fn with_host(host: XtalkHost) -> Self {
        let sink = Arc::new(Sink::default());
        let mem = Arc::new(Mem::new());
        Rig { bridge: Bridge::new(Upstream::single(host, mem.clone(), sink.clone()), &crate::ip30::PCI0), sink, mem }
    }

    /// Write register `off` (a slot base such as `reg::CONTROL`).
    fn set(&self, off: u32, val: u32) {
        assert_eq!(self.bridge.write32(u64::from(off + 4), val), BUS_OK);
    }

    fn reset(&self) {
        self.bridge.reset();
    }

    /// Fit `f` in `slot` on the Bridge's bus.
    fn fit(&self, slot: u8, f: Arc<dyn PciFunction>) {
        self.bridge.bus().attach(DevFn { device: slot, function: 0 }, f);
    }

    fn get(&self, off: u32) -> u32 {
        let r = self.bridge.read32(u64::from(off + 4));
        assert!(r.is_ok());
        r.data
    }

    /// The offset a CPU aims a `size`-byte access at to reach register or
    /// byte `x` of a little-endian device.
    fn aim(x: u32, size: u32) -> u32 {
        (x & !3) | ((x & 3) ^ (4 - size))
    }

    fn cfg(slot: u32, reg: u32) -> u32 {
        CFG0_BASE + slot * 0x1000 + reg
    }

    fn cfg_read(&self, slot: u32, reg: u32, size: u32) -> u32 {
        let a = Self::cfg(slot, Self::aim(reg, size));
        match size {
            1 => self.bridge.read8(u64::from(a)).data as u32,
            2 => self.bridge.read16(u64::from(a)).data as u32,
            _ => self.bridge.read32(u64::from(a)).data,
        }
    }

    fn cfg_write(&self, slot: u32, reg: u32, size: u32, val: u32) {
        let a = Self::cfg(slot, Self::aim(reg, size));
        let s = match size {
            1 => self.bridge.write8(u64::from(a), val as u8),
            2 => self.bridge.write16(u64::from(a), val as u16),
            _ => self.bridge.write32(u64::from(a), val),
        };
        assert_eq!(s, BUS_OK);
    }

    /// A card in `slot`, with its memory BAR at PCI 0x13000000, its I/O BAR
    /// at 0x2000 and both enabled, and the slot's window aimed at memory.
    fn card(&self, slot: u8, pin: u8) -> Arc<Card> {
        let c = Card::new(pin);
        self.fit(slot, c.clone());
        self.cfg_write(slot as u32, 0x10, 4, 0x2000);
        self.cfg_write(slot as u32, 0x14, 4, 0x1300_0000);
        self.cfg_write(slot as u32, 0x04, 2, PCI_COMMAND_IO | PCI_COMMAND_MEMORY);
        self.set(reg::device(slot as u32), DEV_MEMORY_SPACE | 0x130);
        c
    }

    fn win(slot: u32) -> u32 {
        device_window_offset(slot)
    }

    /// Route interrupts to HEART with vector `v` on `pin`, and enable it.
    fn route(&self, pin: u32, v: u32) {
        self.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
        self.set(reg::INT_DEST_LOWER, 0x80);
        self.set(reg::int_host(pin), v);
        let en = self.get(reg::INT_ENABLE);
        self.set(reg::INT_ENABLE, en | 1 << pin);
    }

    /// Point the direct map at HEART.
    fn direct_to_heart(&self, add_512mb: bool) {
        self.set(reg::DIRECT_MAP, (WIDGET_HEART as u32) << DIRECT_WIDGET_SHIFT | if add_512mb { DIRECT_ADD_512MB } else { 0 });
    }
}

// ---- registers ----

#[test]
fn power_up_registers() {
    let r = Rig::new();
    assert_eq!(r.get(reg::IDENTIFICATION), (1 << 28) | (0xc002 << 12) | (0x036 << 1) | 1);
    assert_eq!(r.get(reg::STATUS), 1 << 5, "strapped for PCI");
    assert_eq!(r.get(reg::CONTROL), CONTROL_RESET);
    assert_eq!(r.get(reg::CONTROL) & CONTROL_FLASH_WRITE_EN, 0, "flash writes start disabled");
    assert_eq!(r.get(reg::CONTROL) & 0xf, 0xf, "control widget field");
    assert_eq!(r.get(reg::DIRECT_MAP), DIRECT_ADD_512MB);
    assert_eq!(r.get(reg::device(0)), device_reset(0));
    assert_eq!(r.get(reg::device(1)) & 0xfff, 2);
    assert_eq!(r.get(reg::device(2)) & 0xfff, 4);
    assert_eq!(r.get(reg::device(7)) & 0xfff, 9);
    for s in 0..8 {
        let d = r.get(reg::device(s));
        assert_ne!(d & DEV_MEMORY_SPACE, 0);
        assert_ne!(d & DEV_ERROR_LOCK, 0);
    }
    assert_eq!(r.get(reg::REQUEST_TIMEOUT), 0xf_ffff);
    assert_eq!(r.get(reg::LINK_CONFIG), LINK_CONFIG_RESET);
}

/// The identification word's top nibble is the chip's silicon revision. The
/// 175 MHz Octane reads 3 (an OpenBSD boot prints "Bridge revision 3"); a
/// measured Octane2's PROM monitor read the word as 0x4c00206d, revision 4.
/// A reset keeps the board's revision.
///
/// QEMU: octane/bridge-identity.
#[test]
fn identification_carries_the_boards_revision() {
    let r = Rig::new();
    r.bridge.set_revision(3);
    assert_eq!(r.get(reg::IDENTIFICATION), 0x3c00_206d);
    r.reset();
    assert_eq!(r.get(reg::IDENTIFICATION), 0x3c00_206d);

    r.bridge.set_revision(4);
    assert_eq!(r.get(reg::IDENTIFICATION), 0x4c00_206d);
    r.reset();
    assert_eq!(r.get(reg::IDENTIFICATION), 0x4c00_206d);
}

#[test]
fn control_strobes_read_zero_and_read_only_registers_ignore_writes() {
    let r = Rig::new();
    r.set(reg::CONTROL, 0xffff_ffff);
    assert_eq!(r.get(reg::CONTROL) & CONTROL_RETRY_CLEAR, 0);
    assert_eq!(r.get(reg::CONTROL) & CONTROL_FLASH_WRITE_EN, CONTROL_FLASH_WRITE_EN);
    let id = r.get(reg::IDENTIFICATION);
    r.set(reg::IDENTIFICATION, 0);
    assert_eq!(r.get(reg::IDENTIFICATION), id);
    r.set(reg::INT_STATUS, 0xffff_ffff);
    assert_eq!(r.get(reg::INT_STATUS), 0);
}

#[test]
fn only_whole_words_reach_the_register_block() {
    let r = Rig::new();
    assert!(!r.bridge.read8(u64::from(reg::CONTROL + 4)).is_ok());
    assert!(!r.bridge.read16(u64::from(reg::CONTROL + 4)).is_ok());
    assert!(!r.bridge.read64(u64::from(reg::CONTROL)).is_ok());
    assert_eq!(r.bridge.write64(u64::from(reg::CONTROL), 0), BUS_ERR);
}

#[test]
fn an_undefined_register_posts_an_error_instead_of_faulting() {
    let r = Rig::new();
    // The empty half of a slot, and an offset past the defined block.
    assert_eq!(r.bridge.read32(u64::from(reg::CONTROL)).data, 0xffff_ffff);
    assert_ne!(r.get(reg::INT_STATUS) & INT_INVALID_ADDR, 0);
    r.set(reg::INT_RESET, INT_RESET_REQ_DSP_GRP);
    assert_eq!(r.get(reg::INT_STATUS) & INT_INVALID_ADDR, 0);
    assert_eq!(r.bridge.write32(0x1004, 1), BUS_OK);
    assert_ne!(r.get(reg::INT_STATUS) & INT_INVALID_ADDR, 0);
}

#[test]
fn the_reserved_data_half_takes_zero_and_answers_nothing() {
    let r = Rig::new();
    assert_eq!(r.bridge.write32(u64::from(RESERVED_DATA_HALF), 0), BUS_OK);
    assert_eq!(r.bridge.write32(u64::from(RESERVED_DATA_HALF), 1), BUS_ERR);
    assert!(!r.bridge.read32(u64::from(RESERVED_DATA_HALF)).is_ok());
}

#[test]
fn reset_returns_registers_and_the_table_to_power_up() {
    let r = Rig::new();
    r.set(reg::DIRECT_MAP, 0);
    assert_eq!(r.bridge.write64(u64::from(ATE_BASE), ATE_VALID | 0x5000), BUS_OK);
    r.reset();
    assert_eq!(r.get(reg::DIRECT_MAP), DIRECT_ADD_512MB);
    assert_eq!(r.bridge.read32(u64::from(ATE_BASE + 4)).data, 0);
    assert_eq!(r.bridge.read32(u64::from(ATE_LOW_BASE)).data, 0);
}

#[test]
fn ssram_floats_high_and_swallows_writes() {
    let r = Rig::new();
    assert_eq!(r.bridge.read32(u64::from(SSRAM_BASE)).data, 0xffff_ffff);
    assert_eq!(r.bridge.read8(u64::from(SSRAM_BASE + 3)).data, 0xff);
    assert_eq!(r.bridge.write32(u64::from(SSRAM_BASE), 0), BUS_OK);
}

#[test]
fn an_empty_one_wire_bus_answers_by_the_pulse_width() {
    let r = Rig::new();
    // Nothing answers a reset, so the line stays high; a short pulse
    // releases the wire; one between holds it low.
    r.set(reg::NIC, (520 << NIC_PULSE_SHIFT) | (65 << NIC_SAMPLE_SHIFT));
    assert_eq!(r.get(reg::NIC) & 3, NIC_DONE | NIC_LEVEL);
    r.set(reg::NIC, 10 << NIC_PULSE_SHIFT);
    assert_eq!(r.get(reg::NIC) & 3, NIC_DONE | NIC_LEVEL);
    r.set(reg::NIC, 100 << NIC_PULSE_SHIFT);
    assert_eq!(r.get(reg::NIC) & 3, NIC_DONE);
}

/// One bit slot per write to the NIC register at 0xb0 (its low word is
/// 0xb4): a reset pulse reads low when a part answers presence; then bits
/// go out and come back one slot at a time.
#[test]
fn the_one_wire_master_runs_a_bit_slot_per_write() {
    let r = Rig::new();
    let mut bus = w1::Bus::new();
    bus.add(Box::new(w1::Ds2502::ds2501(0x91, 1, &[0x0a])));
    r.bridge.set_microlan(bus);
    let slot = |pulse: u32| {
        r.set(reg::NIC, pulse << NIC_PULSE_SHIFT | 10 << NIC_SAMPLE_SHIFT);
        let v = r.get(reg::NIC);
        assert_ne!(v & NIC_DONE, 0, "a slot is over inside the write");
        v & NIC_LEVEL != 0
    };
    assert!(!slot(500), "presence pulls the wire low");
    // Read ROM (0x33), least significant bit first, then the 8 ROM bytes.
    for i in 0..8 {
        slot(if 0x33 >> i & 1 != 0 { 10 } else { 100 });
    }
    let mut rom = [0u8; 8];
    for b in rom.iter_mut() {
        for bit in 0..8 {
            *b |= (slot(10) as u8) << bit;
        }
    }
    assert_eq!(rom[0], 0x91, "the Ethernet-address part's family code");
    // NIC-register writes outside the timing fields are not slots.
    assert_eq!(r.get(reg::NIC) & !(NIC_LEVEL | NIC_DONE), (10 << NIC_PULSE_SHIFT) | (10 << NIC_SAMPLE_SHIFT));
}

// ---- configuration space ----

/// Identity as a word, halfwords and bytes, to catch narrow-access lane
/// errors hidden by a correct word read.
#[test]
fn configuration_space() {
    let r = Rig::new();
    r.fit(1, Card::new(PCI_INTA));
    assert_eq!(r.cfg_read(1, 0, 2), CARD_VENDOR);
    assert_eq!(r.cfg_read(1, 2, 2), CARD_DEVICE);
    assert_eq!(r.cfg_read(1, 0, 4), CARD_VENDOR | (CARD_DEVICE << 16));
    assert_eq!(r.cfg_read(1, 0, 1), CARD_VENDOR & 0xff);
    assert_eq!(r.cfg_read(1, 1, 1), CARD_VENDOR >> 8);
    assert_eq!(r.cfg_read(1, 3, 1), CARD_DEVICE >> 8);
}

#[test]
fn narrow_configuration_writes_merge_into_the_word() {
    let r = Rig::new();
    let c = Card::new(PCI_INTA);
    r.fit(1, c.clone());
    r.cfg_write(1, 0x3c, 4, 0x0000_0100);
    r.cfg_write(1, 0x3c, 1, 0x0b);
    assert_eq!(r.cfg_read(1, 0x3c, 4), 0x0000_010b, "pin kept, line written");
    // A BAR size probe: write ones, read which bits stuck.
    r.cfg_write(1, 0x14, 4, 0xffff_ffff);
    assert_eq!(r.cfg_read(1, 0x14, 4), 0xffff_f000);
}

#[test]
fn an_empty_slot_fails_the_read_and_posts_the_write() {
    let r = Rig::new();
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    // Disabled, the master time-out leaves no trace.
    assert!(!r.bridge.read32(u64::from(Rig::cfg(5, 0))).is_ok());
    assert_eq!(r.get(reg::INT_STATUS) & INT_PCI_MASTER_TIMEOUT, 0);
    r.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);
    assert!(!r.bridge.read32(u64::from(Rig::cfg(5, 0x10))).is_ok());
    assert_eq!(r.bridge.write32(u64::from(Rig::cfg(6, 0)), 0), BUS_OK);
    let s = r.get(reg::INT_STATUS);
    assert_ne!(s & INT_PCI_MASTER_TIMEOUT, 0);
    assert_ne!(s & INT_MULTI_ERR, 0, "the second error only flags");
    // The first one owns the capture: device 5, line 29 of the address.
    assert_eq!(r.get(reg::PCI_ERR_UPPER), 5 << 16);
    assert_eq!(r.get(reg::PCI_ERR_LOWER), (1 << 29) | 0x10);
    assert!(!r.sink.take().is_empty(), "the error interrupt was sent");
    r.set(reg::INT_RESET, INT_RESET_PCI_GRP | INT_RESET_MULTI_ERR);
    assert_eq!(r.get(reg::INT_STATUS), 0);
    // A function other than 0 is not there either.
    r.fit(2, Card::new(PCI_INTA));
    assert!(!r.bridge.read32(u64::from(Rig::cfg(2, 0x100))).is_ok());
}

// ---- device windows ----

/// One scratch register reached through the I/O and the memory BAR, written
/// whole and read back a halfword and a byte at a time.
#[test]
fn both_bars_reach_one_register() {
    let r = Rig::new();
    let c = r.card(1, PCI_INTA);
    let w = Rig::win(1);
    let pattern = 0x8b7a_6950u32;
    let (mem_at, io_at) = (w + SCRATCH, w + 0x2000 + SCRATCH);

    assert_eq!(r.bridge.write32(u64::from(mem_at), pattern), BUS_OK);
    assert_eq!(c.peek32(SCRATCH as usize), pattern);
    // The same register through the I/O window.
    r.set(reg::device(1), 0);
    assert_eq!(r.bridge.read32(u64::from(io_at)).data, pattern);
    assert_eq!(r.bridge.write32(u64::from(io_at), !pattern), BUS_OK);
    r.set(reg::device(1), DEV_MEMORY_SPACE | 0x130);
    assert_eq!(r.bridge.read32(u64::from(mem_at)).data, !pattern);

    let v = !pattern;
    assert_eq!(r.bridge.read16(u64::from(w + (SCRATCH ^ 2))).data as u32, v & 0xffff);
    assert_eq!(r.bridge.read16(u64::from(w + (SCRATCH + 2 ^ 2))).data as u32, v >> 16);
    for i in 0..4 {
        assert_eq!(r.bridge.read8(u64::from(w + ((SCRATCH + i) ^ 3))).data as u32, (v >> (8 * i)) & 0xff);
    }
}

/// With the programmed swap clear, a word keeps its value; set, the device
/// sees it byte-reversed. A doubleword to a 32-bit device is two cycles,
/// lower address first.
#[test]
fn device3_programmed_lanes() {
    let r = Rig::new();
    let c = r.card(3, PCI_INTA);
    let w = Rig::win(3);
    assert_eq!(w, 0x70_0000);

    r.set(reg::device(3), 0x1130);
    assert_eq!(r.bridge.read32(u64::from(w + 0x20)).data, 0x1234_5678);
    r.bridge.write32(u64::from(w + 0x24), 0x1234_5678);
    assert_eq!(c.peek32(0x24), 0x1234_5678);

    r.set(reg::device(3), 0x3130);
    assert_eq!(r.bridge.read32(u64::from(w + 0x20)).data, 0x7856_3412);
    r.bridge.write32(u64::from(w + 0x24), 0x1234_5678);
    assert_eq!(c.peek32(0x24), 0x7856_3412);

    // The second PCI-32 read must not sign-extend into the first word, and
    // the lower-address word is the more significant half.
    r.bridge.write32(u64::from(w + 0x24), 0xfedc_ba98);
    assert_eq!(r.bridge.read32(u64::from(w + 0x24)).data, 0xfedc_ba98);
    assert_eq!(r.bridge.read64(u64::from(w + 0x20)).data, 0x7856_3412_fedc_ba98);
    assert_eq!(r.bridge.write64(u64::from(w + 0x28), 0x0102_0304_0506_0708), BUS_OK);
    assert_eq!(c.peek32(0x28), 0x0403_0201);
    assert_eq!(c.peek32(0x2c), 0x0807_0605);
}

#[test]
fn window_sizes_are_two_then_one_megabyte() {
    let offs: Vec<u32> = (0..8).map(device_window_offset).collect();
    assert_eq!(offs, [0x20_0000, 0x40_0000, 0x60_0000, 0x70_0000, 0x80_0000, 0x90_0000, 0xa0_0000, 0xb0_0000]);
    assert_eq!(window_slot(0x5f_ffff), Some(1));
    assert_eq!(window_slot(0x60_0000), Some(2));
    assert_eq!(window_slot(0xbf_ffff), Some(7));
    assert_eq!(window_slot(0xc0_0000), None);
}

#[test]
fn a_window_cycle_nobody_claims_is_a_master_timeout() {
    let r = Rig::new();
    r.card(1, PCI_INTA);
    r.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    // Past the 4 KB memory BAR.
    assert!(!r.bridge.read32(u64::from(Rig::win(1) + 0x1000)).is_ok());
    assert_eq!(r.bridge.write32(u64::from(Rig::win(1) + 0x1000), 0), BUS_OK, "a write posts");
    assert_ne!(r.get(reg::INT_STATUS) & INT_PCI_MASTER_TIMEOUT, 0);
    assert_eq!(r.get(reg::PCI_ERR_LOWER), 0x1300_1000);
    assert_eq!(r.get(reg::PCI_ERR_UPPER), 1 << 16);
}

#[test]
fn an_empty_slot_config_read_names_the_master_timeout() {
    let r = Rig::new();
    r.card(1, PCI_INTA);
    let at = |slot: u32, reg: u32| u64::from(Rig::cfg(slot, reg));
    let empty = |slot: u32| Some(format!("PCI config master timeout: empty slot {slot} function 0"));

    // Empty slots, any register, any word-or-narrower width.
    for slot in [0, 2, 7] {
        assert_eq!(r.bridge.describe_fault(at(slot, 0), 4), empty(slot));
        assert_eq!(r.bridge.describe_fault(at(slot, Rig::aim(0x0e, 1)), 1), empty(slot));
    }
    // Function 1 of a populated slot has nothing behind it either.
    assert_eq!(
        r.bridge.describe_fault(at(1, 0x100), 4),
        Some("PCI config master timeout: slot 1 has no function 1".to_string())
    );
    // The description follows the population, and the read still fails.
    assert_eq!(r.bridge.describe_fault(at(1, 0), 4), None);
    assert!(!r.bridge.read32(at(0, 0)).is_ok());
    r.card(0, PCI_INTA);
    assert_eq!(r.bridge.describe_fault(at(0, 0), 4), None);
    // A doubleword config read fails for another reason.
    assert_eq!(r.bridge.describe_fault(at(2, 0), 8), None);
    // Other spaces: registers, device windows and the table.
    assert_eq!(r.bridge.describe_fault(0x10, 4), None);
    assert_eq!(r.bridge.describe_fault(u64::from(Rig::win(1) + 0x1000), 4), None);
    assert_eq!(r.bridge.describe_fault(u64::from(ATE_BASE), 4), None);
    // The Bridge decodes the full physical address; only the window offset counts.
    assert_eq!(r.bridge.describe_fault(0x1f00_0000 + at(3, 0), 4), empty(3));
}

// ---- type 1 configuration ----

impl Rig {
    /// A PCI-to-PCI bridge in `slot` with a card at device 0 behind it.
    fn bridge_with_card_behind(&self, slot: u8) -> Arc<Card> {
        let p2p = PciToPciBridge::new(self.bridge.bus(), DevFn { device: slot, function: 0 }, 0x1011, 0x0026);
        let c = Card::new(PCI_INTA);
        p2p.secondary().attach(DevFn { device: 0, function: 0 }, c.clone());
        self.fit(slot, p2p);
        c
    }

    /// Aim the type 1 aperture at `bus` and `device`.
    fn type1_target(&self, bus: u32, device: u32) {
        self.set(reg::PCI_TYPE1_CONFIG, bus << 16 | device << 11);
        assert_eq!(self.get(reg::PCI_TYPE1_CONFIG), bus << 16 | device << 11);
    }

    fn type1_read(&self, func: u32, reg: u32, size: u32) -> Option<u32> {
        let a = u64::from(TYPE1_BASE + (func << 8) + Self::aim(reg, size));
        match size {
            1 => Some(self.bridge.read8(a)).filter(|r| r.is_ok()).map(|r| r.data as u32),
            2 => Some(self.bridge.read16(a)).filter(|r| r.is_ok()).map(|r| r.data as u32),
            _ => Some(self.bridge.read32(a)).filter(|r| r.is_ok()).map(|r| r.data),
        }
    }

    fn type1_write(&self, func: u32, reg: u32, size: u32, val: u32) {
        let a = u64::from(TYPE1_BASE + (func << 8) + Self::aim(reg, size));
        let s = match size {
            1 => self.bridge.write8(a, val as u8),
            2 => self.bridge.write16(a, val as u16),
            _ => self.bridge.write32(a, val),
        };
        assert_eq!(s, BUS_OK);
    }
}

/// IRIX 6.5.30 on the Octane finds a bridge in slot 5, numbers its buses
/// with byte writes (primary 0, secondary and subordinate 0x10) and scans
/// all 32 devices of bus 0x10 through the type 1 aperture. The card behind
/// the bridge answers at device 0 and nothing else does: the bridge returns
/// all ones for the rest.
#[test]
fn the_type_1_aperture_reaches_the_bus_behind_a_bridge() {
    let r = Rig::new();
    let c = r.bridge_with_card_behind(5);
    assert_eq!(r.cfg_read(5, 0x08, 4) >> 8, 0x06_0400, "a PCI-to-PCI bridge");
    assert_eq!(r.cfg_read(5, 0x0e, 1) & 0x7f, 1, "type 1 header");
    r.cfg_write(5, 0x18, 1, 0x00);
    r.cfg_write(5, 0x19, 1, 0x10);
    r.cfg_write(5, 0x1a, 1, 0x10);

    let found: Vec<u32> = (0..32)
        .filter(|&d| {
            r.type1_target(0x10, d);
            r.type1_read(0, 0, 4) != Some(0xffff_ffff)
        })
        .collect();
    assert_eq!(found, [0]);

    r.type1_target(0x10, 0);
    assert_eq!(r.type1_read(0, 0, 4), Some(CARD_VENDOR | CARD_DEVICE << 16));
    assert_eq!(r.type1_read(0, 2, 2), Some(CARD_DEVICE), "the same byte order as type 0");
    assert_eq!(r.type1_read(0, 1, 1), Some(CARD_VENDOR >> 8));
    assert_eq!(r.type1_read(1, 0, 4), Some(0xffff_ffff), "the card has no function 1");

    // Size and place the card's memory BAR through the aperture.
    r.type1_write(0, 0x14, 4, 0xffff_ffff);
    assert_eq!(r.type1_read(0, 0x14, 4), Some(0xffff_f000));
    r.type1_write(0, 0x14, 4, 0x1300_0000);
    r.type1_write(0, 0x04, 2, PCI_COMMAND_MEMORY);
    assert_eq!(c.state().config().bar_base(1), 0x1300_0000);
    assert_eq!(r.type1_read(0, 0x04, 2), Some(PCI_COMMAND_MEMORY));

    // Open the bridge's memory window over it: slot 5's device window
    // reaches the card's registers.
    r.cfg_write(5, 0x20, 4, 0x1300_1300);
    r.cfg_write(5, 0x04, 2, PCI_COMMAND_MEMORY);
    r.set(reg::device(5), DEV_MEMORY_SPACE | 0x130);
    assert_eq!(r.bridge.read32(u64::from(Rig::win(5) + FIXED)).data, 0x1234_5678);
}

/// A type 1 cycle no bridge claims ends in a master time-out, like an empty
/// slot: bus 0 itself (type 0 functions ignore type 1 cycles) and a bus
/// number outside every bridge's range.
#[test]
fn a_type_1_cycle_nobody_claims_is_a_master_timeout() {
    let r = Rig::new();
    r.bridge_with_card_behind(5);
    r.cfg_write(5, 0x18, 4, 0x0001_0100);
    r.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);

    r.type1_target(0, 5);
    assert_eq!(r.type1_read(0, 0, 4), None, "the bridge itself is on bus 0");
    assert_eq!(r.bridge.describe_fault(u64::from(TYPE1_BASE), 4),
        Some("PCI type 1 config master timeout: no bridge reaches bus 0".to_string()));
    r.type1_target(2, 0);
    r.type1_write(0, 0x04, 2, 0);
    assert_eq!(r.type1_read(0, 0, 4), None);
    assert_ne!(r.get(reg::INT_STATUS) & INT_PCI_MASTER_TIMEOUT, 0);
    // The capture holds the first: a type 1 address (bus 0, device 5).
    assert_eq!(r.get(reg::PCI_ERR_LOWER), 5 << 11 | 1);

    r.type1_target(1, 0);
    assert!(r.type1_read(0, 0, 4).is_some());
    assert_eq!(r.bridge.describe_fault(u64::from(TYPE1_BASE), 4), None);
}

/// Behind a PCI-to-PCI bridge an unclaimed type 1 cycle ends in master abort
/// at the bridge, not at the Bridge: a read returns all ones, a write is
/// dropped, the bridge's secondary status records the abort, and no master
/// time-out is raised (PCI-to-PCI Bridge Architecture, master abort handling
/// with Master-Abort Mode clear).
#[test]
fn an_absent_device_behind_a_pci_bridge_master_aborts_at_the_bridge() {
    let r = Rig::new();
    r.bridge_with_card_behind(5);
    r.cfg_write(5, 0x18, 4, 0x0001_0100);
    r.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);
    assert_eq!(r.cfg_read(5, 0x1c, 4) >> 29 & 1, 0, "no abort yet");

    r.type1_target(1, 9);
    assert_eq!(r.type1_read(0, 0, 4), Some(0xffff_ffff));
    assert_eq!(r.type1_read(0, 0, 2), Some(0xffff));
    assert_eq!(r.type1_read(0, 3, 1), Some(0xff));
    r.type1_write(0, 0x04, 2, PCI_COMMAND_MEMORY);
    assert_eq!(r.bridge.describe_fault(u64::from(TYPE1_BASE), 4), None);
    assert_eq!(r.get(reg::INT_STATUS) & INT_PCI_MASTER_TIMEOUT, 0, "the Bridge saw every cycle complete");
    assert_eq!(r.cfg_read(5, 0x1c, 4) >> 29 & 1, 1, "received master abort in secondary status");

    // Write one to clear.
    r.cfg_write(5, 0x1c, 4, 1 << 29);
    assert_eq!(r.cfg_read(5, 0x1c, 4) >> 29 & 1, 0);

    // The Bridge's own bus is unchanged: nothing there is a master time-out.
    r.type1_target(0, 9);
    assert_eq!(r.type1_read(0, 0, 4), None);
    assert_ne!(r.get(reg::INT_STATUS) & INT_PCI_MASTER_TIMEOUT, 0);
}

// ---- the 2 GB window ----

/// Through the widget's 2 GB window the first 16 MB is the small window
/// again, and offset 0x4000_0000 on is PCI memory from address 0, with
/// the bytes placed by Control's memory swap bit (Bridge specification
/// pp. 52 and 55). IRIX 6.5.30 reaches a BAR this way when a bridge
/// window is too wide for the slot's device window.
#[test]
fn the_2gb_window_reaches_pci_memory_through_the_32_bit_alias() {
    let r = Rig::new();
    let c = r.card(1, PCI_INTA); // memory BAR at PCI 0x1300_0000
    let long = r.bridge.long_window();
    let alias = |pci: u32| u64::from(PIO32_ALIAS + pci) + 0xf_8000_0000;
    assert_eq!(long.read32(0xf_8000_0000 + u64::from(reg::IDENTIFICATION) + 4).data, r.get(reg::IDENTIFICATION));
    assert_eq!(long.read32(alias(0x1300_0000 + FIXED)).data, 0x1234_5678, "swap clear: word values");
    assert_eq!(long.write32(alias(0x1300_0000 + SCRATCH), 0x8b7a_6950), BUS_OK);
    assert_eq!(c.peek32(SCRATCH as usize), 0x8b7a_6950);
    assert_eq!(long.read8(alias(0x1300_0000 + (FIXED ^ 3))).data, 0x78, "a byte moves to its position");
    r.set(reg::CONTROL, r.get(reg::CONTROL) | CONTROL_MEM_SWAP);
    assert_eq!(long.read32(alias(0x1300_0000 + FIXED)).data, 0x7856_3412, "swap set: a byte stream");
    // Between the small window and the alias, and PCI memory nothing decodes.
    assert!(!long.read32(0xf_8000_0000 + 0x0100_0000).is_ok());
    r.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);
    assert!(!long.read32(alias(0x2000_0000)).is_ok());
    assert_ne!(r.get(reg::INT_STATUS) & INT_PCI_MASTER_TIMEOUT, 0);
    // Approximation: the capture names device 0, and the PCI address the
    // cycle carried.
    assert_eq!(r.get(reg::PCI_ERR_UPPER), 0);
    assert_eq!(r.get(reg::PCI_ERR_LOWER), 0x2000_0000);
}

/// A BAR the alias decodes whole is published as a run at its alias
/// offset, and the run answers as the 2 GB window does.
#[test]
fn a_bar_in_the_alias_is_a_run() {
    let r = Rig::new();
    let runs = Arc::new(Mutex::new(Vec::new()));
    let (sink, b) = (runs.clone(), Arc::downgrade(&r.bridge));
    r.bridge.bus().on_bar_mappings(Box::new(move |rs| {
        let b = b.upgrade().unwrap();
        *sink.lock() = rs.iter().map(|x| (x.cpu..x.cpu + x.len, b.bar_device(x))).collect::<Vec<_>>();
    }));
    r.card(1, PCI_INTA);
    let at = u64::from(PIO32_ALIAS + 0x1300_0000);
    let run = runs.lock().iter().find(|x| x.0.contains(&at)).cloned().expect("the alias carries a run");
    assert_eq!(run.0, at..at + 0x1000);
    let pa = 0xf_8000_0000 + at + u64::from(FIXED);
    assert_eq!(run.1.read32(pa).data, r.bridge.long_window().read32(pa).data);
    assert_eq!(run.1.read32(pa).data, 0x1234_5678);
}

#[test]
fn a_disabled_command_register_hides_the_bars() {
    let r = Rig::new();
    let c = r.card(1, PCI_INTA);
    r.cfg_write(1, 0x04, 2, 0);
    assert!(!r.bridge.read32(u64::from(Rig::win(1) + SCRATCH)).is_ok());
    assert_eq!(c.peek32(SCRATCH as usize), 0);
}

// ---- DMA ----

#[test]
fn dma_through_the_direct_map_places_bytes_by_the_swap_bit() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
    r.direct_to_heart(false);

    // Swap clear on a 32-bit slot: each aligned word is reversed.
    assert!(c.dma().dma_write(0x8000_1000, &data).is_ok());
    assert_eq!(r.mem.peek(0x1000, 8), [4, 3, 2, 1, 8, 7, 6, 5]);
    let mut back = [0u8; 8];
    assert!(c.dma().dma_read(0x8000_1000, &mut back).is_ok());
    assert_eq!(back, data);

    // Swap set: a byte stream.
    r.set(reg::device(2), r.get(reg::device(2)) | DEV_SWAP_DIRECT);
    assert!(c.dma().dma_write(0x8000_2000, &data).is_ok());
    assert_eq!(r.mem.peek(0x2000, 8), data);

    // A 64-bit slot reverses doublewords.
    r.set(reg::device(2), r.get(reg::device(2)) & !DEV_SWAP_DIRECT);
    r.bridge.set_dma_64bit(2, true);
    assert!(c.dma().dma_write(0x8000_3000, &data).is_ok());
    assert_eq!(r.mem.peek(0x3000, 8), [8, 7, 6, 5, 4, 3, 2, 1]);
}

#[test]
fn a_partial_word_lands_on_its_swapped_byte_and_spares_the_rest() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.direct_to_heart(false);
    r.mem.poke(0x1000, &[0x11, 0x22, 0x33, 0x44]);
    assert!(c.dma().dma_write(0x8000_1001, &[0xaa]).is_ok());
    assert_eq!(r.mem.peek(0x1000, 4), [0x11, 0x22, 0xaa, 0x44]);
    let mut b = [0u8; 2];
    assert!(c.dma().dma_read(0x8000_1002, &mut b).is_ok());
    assert_eq!(b, [0x22, 0x11], "PCI bytes 2 and 3 are memory bytes 1 and 0");
}

#[test]
fn the_direct_map_adds_512_mb_with_a_zero_offset_field() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::device(2), r.get(reg::device(2)) | DEV_SWAP_DIRECT);
    r.direct_to_heart(true);
    assert!(c.dma().dma_write(0x8000_0010, &[9, 8, 7]).is_ok());
    assert_eq!(r.mem.peek(HIGH + 0x10, 3), [9, 8, 7]);
    // A nonzero offset field supplies bits 31 and up and drops the add.
    r.set(reg::DIRECT_MAP, (WIDGET_HEART as u32) << DIRECT_WIDGET_SHIFT | DIRECT_ADD_512MB | 1);
    assert_eq!(c.dma().dma_write(0x8000_0010, &[1]), PciDmaStatus::MasterAbort, "physical 2 GB is past the memory the fake has");
}

#[test]
fn the_direct_map_reset_target_is_not_memory() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    // Widget 0 is the crossbar: nothing answers a data transfer there.
    assert_eq!(c.dma().dma_write(0x8000_0000, &[1]), PciDmaStatus::MasterAbort);
}

#[test]
fn dma_through_the_translation_table() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::device(2), r.get(reg::device(2)) | DEV_SWAP_TRANSLATED);
    let entry = ATE_VALID | ((WIDGET_HEART as u64) << ATE_TARGET_WIDGET_SHIFT) | 0x5000;
    assert_eq!(r.bridge.write64(u64::from(ATE_BASE + 8 * 5), entry), BUS_OK);
    assert!(c.dma().dma_write(0x4000_0000 + 5 * 0x1000 + 0x10, &[1, 2, 3, 4]).is_ok());
    assert_eq!(r.mem.peek(0x5010, 4), [1, 2, 3, 4]);

    // The entry reads back in halves; a 32-bit write is refused.
    assert_eq!(r.bridge.read32(u64::from(ATE_BASE + 8 * 5 + 4)).data as u64, entry >> 32);
    assert_eq!(r.bridge.read32(u64::from(ATE_LOW_BASE + 4 * 5)).data, entry as u32);
    assert_eq!(r.bridge.write32(u64::from(ATE_BASE + 8 * 5), 0), BUS_ERR);
    assert!(!r.bridge.read32(u64::from(ATE_BASE + 8 * 5)).is_ok(), "the empty half");
    assert_eq!(r.bridge.write32(u64::from(ATE_LOW_BASE), 0), BUS_ERR, "the lower-half view is read only");
}

#[test]
fn a_transfer_crossing_a_page_follows_both_entries() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::device(2), r.get(reg::device(2)) | DEV_SWAP_TRANSLATED);
    let e = |a: u64| ATE_VALID | ((WIDGET_HEART as u64) << ATE_TARGET_WIDGET_SHIFT) | a;
    r.bridge.write64(u64::from(ATE_BASE), e(0x10000));
    r.bridge.write64(u64::from(ATE_BASE + 8), e(0x30000));
    assert!(c.dma().dma_write(0x4000_0ffe, &[1, 2, 3, 4]).is_ok());
    assert_eq!(r.mem.peek(0x10ffe, 2), [1, 2]);
    assert_eq!(r.mem.peek(0x30000, 2), [3, 4]);
    let mut b = [0u8; 4];
    assert!(c.dma().dma_read(0x4000_0ffe, &mut b).is_ok());
    assert_eq!(b, [1, 2, 3, 4]);
}

#[test]
fn sixteen_kilobyte_pages_index_the_table_by_the_larger_page() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::device(2), r.get(reg::device(2)) | DEV_SWAP_TRANSLATED);
    r.set(reg::CONTROL, r.get(reg::CONTROL) | CONTROL_PAGE_16K);
    let e = ATE_VALID | ((WIDGET_HEART as u64) << ATE_TARGET_WIDGET_SHIFT) | 0x8000;
    r.bridge.write64(u64::from(ATE_BASE + 8), e);
    // Entry 1 covers 0x4000..0x8000 of the window; the page offset is 14 bits.
    assert!(c.dma().dma_write(0x4000_0000 + 0x4000 + 0x1234, &[0x5a]).is_ok());
    assert_eq!(r.mem.peek(0x8000 + 0x1234, 1), [0x5a]);
}

#[test]
fn translations_that_go_nowhere_fail_and_reads_return_zero() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    let mut b = [0xffu8; 4];
    // Invalid entry, past the table, PCI-local space.
    assert_eq!(c.dma().dma_read(0x4000_0000, &mut b), PciDmaStatus::MasterAbort);
    assert_eq!(b, [0; 4]);
    assert_eq!(c.dma().dma_write(0x4000_0000 + 128 * 0x1000, &[1]), PciDmaStatus::MasterAbort);
    assert_eq!(c.dma().dma_write(0x1000_0000, &[1]), PciDmaStatus::MasterAbort);
    // An entry aimed at another widget.
    r.bridge.write64(u64::from(ATE_BASE), ATE_VALID | (9 << ATE_TARGET_WIDGET_SHIFT));
    assert_eq!(c.dma().dma_write(0x4000_0000, &[1]), PciDmaStatus::MasterAbort);
}

#[test]
fn a_dropped_read_is_recorded_when_enabled() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    r.set(reg::INT_ENABLE, INT_XREAD_REQ_TIMEOUT);
    let mut b = [0u8; 4];
    // A translation that fails is refused up front and times nothing.
    assert_eq!(c.dma().dma_read(0x4000_3000, &mut b), PciDmaStatus::MasterAbort);
    assert_eq!(r.get(reg::INT_STATUS), 0);
    // A read that reaches HEART and finds no memory is a silent abort.
    r.direct_to_heart(false);
    assert_eq!(c.dma().dma_read(0x8000_0000 | 0x7000_0000, &mut b), PciDmaStatus::MasterAbort);
    assert_eq!(r.get(reg::INT_STATUS), 0);
    // The Crossbow drops a request to an illegal widget (1 to 7), to this
    // Bridge's own port, and to a widget with no live link.
    for w in [3u64, 15, 10] {
        r.set(reg::INT_RESET, 0x7f);
        let a = (w << 60) | 0x3000;
        assert_eq!(c.dma().dma_read(a, &mut b), PciDmaStatus::MasterAbort);
        assert_ne!(r.get(reg::INT_STATUS) & INT_XREAD_REQ_TIMEOUT, 0, "widget {w}");
        assert_eq!(r.get(reg::RESP_ERR_UPPER), 2 << 20);
        assert_eq!(r.get(reg::RESP_ERR_LOWER), a as u32);
    }
    // A widget with a live link is peer DMA, refused without a timeout.
    r.set(reg::INT_RESET, 0x7f);
    r.bridge.set_link_alive(Box::new(|w| w == WIDGET_HEART || w == 10));
    assert_eq!(c.dma().dma_read((10u64 << 60) | 0x3000, &mut b), PciDmaStatus::MasterAbort);
    assert_eq!(r.get(reg::INT_STATUS), 0);
    assert_eq!(c.dma().dma_read((11u64 << 60) | 0x3000, &mut b), PciDmaStatus::MasterAbort);
    assert_ne!(r.get(reg::INT_STATUS) & INT_XREAD_REQ_TIMEOUT, 0);
    r.set(reg::INT_RESET, INT_RESET_RESP_BUF_GRP);
    assert_eq!(r.get(reg::INT_STATUS), 0);
}

// ---- interrupts ----

#[test]
fn a_function_pin_sends_the_programmed_vector_to_heart() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.route(2, 17);
    c.raise(true);
    assert_eq!(r.sink.take(), [(17, true)]);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 2);
    // The level was already high: nothing more until it falls and rises.
    c.raise(true);
    assert!(r.sink.take().is_empty());
    // Without the mode bit the guest clears HEART's bit itself.
    c.raise(false);
    assert!(r.sink.take().is_empty());
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 0, "status follows the level");
    r.set(reg::INT_MODE, 1 << 2);
    c.raise(true);
    c.raise(false);
    assert_eq!(r.sink.take(), [(17, true), (17, false)]);
}

#[test]
fn enabling_a_pin_that_is_already_high_sends_at_once() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    r.set(reg::int_host(2), 5);
    c.raise(true);
    assert!(r.sink.take().is_empty());
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 2, "status shows it, enabled or not");
    r.set(reg::INT_ENABLE, 1 << 2);
    assert_eq!(r.sink.take(), [(5, true)]);
}

#[test]
fn an_unrouted_pin_sends_nothing() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::INT_ENABLE, 1 << 2);
    c.raise(true);
    assert!(r.sink.take().is_empty(), "no host register programmed");
    c.raise(false);
    // A destination naming another widget has nowhere to go.
    r.set(reg::INT_DEST_UPPER, 9 << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    r.set(reg::int_host(2), 3);
    c.raise(true);
    assert!(r.sink.take().is_empty());
}

/// Pins A and C reach the pin numbered as the slot, B and D that number with
/// bit two flipped; functions sharing a pin OR their levels.
#[test]
fn pins_follow_the_slot_wiring_and_share_by_or() {
    let r = Rig::new();
    let b1 = r.card(1, 2); // slot 1 INTB -> pin 5
    let a5 = r.card(5, PCI_INTA); // slot 5 INTA -> pin 5
    let c3 = r.card(3, 3); // slot 3 INTC -> pin 3
    r.route(5, 40);
    r.route(3, 41);
    b1.raise(true);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 5);
    a5.raise(true);
    assert_eq!(r.sink.take(), [(40, true)], "one edge for the shared pin");
    b1.raise(false);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 5, "the other function still holds it");
    a5.raise(false);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 0);
    c3.raise(true);
    assert_eq!(r.sink.take(), [(41, true)]);
}

#[test]
fn a_directly_wired_input_shares_the_pins() {
    let r = Rig::new();
    r.route(4, 33);
    r.bridge.bus().board_irq(BoardIrq(4)).set(true);
    assert_eq!(r.sink.take(), [(33, true)]);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 4);
    r.bridge.bus().board_irq(BoardIrq(4)).set(false);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 0);
}

// ---- the PCI reset pins, Control 27:24 ----

/// A recording function at `slot`, command register enabled, and the
/// log of the RST# deliveries it has seen.
fn watcher(r: &Rig, slot: u8) -> (Arc<std::sync::Mutex<Vec<String>>>, DevFn) {
    use crate::pci::testing::TestCard;
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    let at = DevFn { device: slot, function: 0 };
    let bars = [Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused];
    let name = ["w0", "w1", "w2", "w3", "w4", "w5", "w6", "w7"][slot as usize];
    r.fit(slot, TestCard::logged(name, bars, 0, &log));
    r.bridge.bus().config_write(at, 4, 2, PCI_COMMAND_MEMORY);
    (log, at)
}

const RST_PINS: u32 = 0xf << 24;

/// Control resets with every pin deasserted (1); writing it back with them
/// still deasserted resets nothing, so the PROM's first Control write
/// does not reset the functions.
#[test]
fn releasing_untouched_reset_pins_resets_nothing() {
    let r = Rig::new();
    let (log, at) = watcher(&r, 0);
    assert_eq!(r.get(reg::CONTROL) & RST_PINS, RST_PINS, "deasserted at reset");
    r.set(reg::CONTROL, r.get(reg::CONTROL));
    r.set(reg::CONTROL, r.get(reg::CONTROL) | DEV_ERROR_LOCK);
    assert!(log.lock().unwrap().is_empty());
    assert_eq!(r.bridge.bus().config_read(at, 4, 2), Some(PCI_COMMAND_MEMORY), "and the header is as it was");
}

/// Pin `n` going from 1 to 0 resets the function at device `n`, once, and no
/// other: not another pin's device and not a device with no pin. Raising the
/// pin again resets nothing. (The pin-to-device wiring is inferred.)
#[test]
fn a_one_to_zero_on_reset_pin_n_resets_device_n_only_once() {
    for pin in 0..4u8 {
        let r = Rig::new();
        let watchers: Vec<_> = [0u8, 1, 2, 3, 6].iter().map(|&slot| (slot, watcher(&r, slot))).collect();
        let ctl = r.get(reg::CONTROL);
        r.set(reg::CONTROL, ctl & !(1 << (24 + pin)));
        for (slot, (log, at)) in &watchers {
            if *slot == pin {
                assert_eq!(*log.lock().unwrap(), [format!("w{slot} cmd=0x0")], "pin {pin}: header reset first, then the function");
                assert_eq!(r.bridge.bus().config_read(*at, 4, 2), Some(0));
            } else {
                assert!(log.lock().unwrap().is_empty(), "pin {pin} left device {slot} alone");
                assert_eq!(r.bridge.bus().config_read(*at, 4, 2), Some(PCI_COMMAND_MEMORY));
            }
        }
        r.set(reg::CONTROL, ctl);
        assert_eq!(watchers[pin as usize].1 .0.lock().unwrap().len(), 1, "pin {pin}: releasing it resets nothing");
    }
}

/// On the Octane's own devices: pin 0 resets the first ISP1040 and leaves the
/// IOC3 (the console and Ethernet) and the other ISP alone.
#[test]
fn reset_pin_0_resets_device_0_and_leaves_the_ioc3_and_the_other_isp() {
    use crate::pci::testing::TestCard;
    let r = Rig::new();
    let bars = [Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused];
    let log = Arc::new(std::sync::Mutex::new(Vec::new()));
    // Devices 0 and 1 are the ISPs and 2 the IOC3 on the Octane.
    for (slot, name) in [(0u8, "isp0"), (1, "isp1"), (2, "ioc3")] {
        r.fit(slot, TestCard::logged(name, bars, 0, &log));
        r.bridge.bus().config_write(DevFn { device: slot, function: 0 }, 4, 2, PCI_COMMAND_MEMORY);
    }
    let ctl = r.get(reg::CONTROL);
    r.set(reg::CONTROL, ctl & !(1 << 24));
    assert_eq!(*log.lock().unwrap(), ["isp0 cmd=0x0"]);
    for slot in [1u8, 2] {
        assert_eq!(r.bridge.bus().config_read(DevFn { device: slot, function: 0 }, 4, 2), Some(PCI_COMMAND_MEMORY));
    }
}

#[test]
fn holding_a_pin_low_does_not_reset_again() {
    let r = Rig::new();
    let (log1, _) = watcher(&r, 1);
    let (log2, _) = watcher(&r, 2);
    let ctl = r.get(reg::CONTROL);
    r.set(reg::CONTROL, ctl & !(1 << 25));
    r.set(reg::CONTROL, ctl & !(1 << 25) | DEV_ERROR_LOCK);
    r.set(reg::CONTROL, ctl & !(0x3 << 25));
    assert_eq!(log1.lock().unwrap().len(), 1, "holding pin 1 low is not a new 1 to 0");
    assert_eq!(log2.lock().unwrap().len(), 1, "pin 2 is");
}

/// The specification does not say that a Bridge core reset or a Crossbow
/// link reset pulses the pins, so neither resets the bus; the functions
/// keep their lines.
#[test]
fn a_core_reset_and_a_link_reset_run_no_bus_reset_and_keep_pin_levels() {
    let r = Rig::new();
    let (log, at) = watcher(&r, 6);
    let c = r.card(2, PCI_INTA);
    r.route(2, 1);
    c.raise(true);
    r.reset();
    r.bridge.link_reset();
    assert!(log.lock().unwrap().is_empty());
    assert_eq!(r.bridge.bus().config_read(at, 4, 2), Some(PCI_COMMAND_MEMORY));
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 2, "the card still holds its line");
}

/// The wiring table is the rule the Bridge always had: pins A and C reach
/// the pin numbered as the slot, B and D that number with bit two flipped.
#[test]
fn octane_routes_equal_the_old_map_irq() {
    let old = |slot: usize, pin0: usize| if pin0 & 1 != 0 { slot ^ 4 } else { slot };
    for slot in 0..DEVICES {
        for pin in 1..=4usize {
            let r = Rig::new();
            let c = r.card(slot as u8, pin as u8);
            r.route(old(slot, pin - 1) as u32, 9);
            c.raise(true);
            assert_eq!(r.sink.take(), [(9, true)], "slot {slot} pin {pin}");
            assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << old(slot, pin - 1));
        }
    }
}

/// A pin that falls and rises again while a packet is still on its way to
/// HEART reaches the Bridge as a fall and a rise, so the second raise is a
/// new interrupt with its own packet, not a level that never dropped. (The
/// guest may have cleared HEART's bit in between, and only a packet sets
/// it again.)
#[test]
fn pin_fall_then_rise_gives_the_bridge_one_heart_packet_each() {
    use std::sync::mpsc::{channel, Receiver, Sender};
    /// HEART that holds its first packet until released.
    struct Slow {
        log: Mutex<Vec<(u32, bool)>>,
        stall: Mutex<Option<(Sender<()>, Receiver<()>)>>,
    }
    impl InterruptController for Slow {
        fn set_line(&self, line: u32, level: bool) {
            self.log.lock().push((line, level));
            if let Some((begun, go)) = self.stall.lock().take() {
                begun.send(()).unwrap();
                go.recv().unwrap();
            }
        }
    }
    let heart = Arc::new(Slow { log: Mutex::new(Vec::new()), stall: Mutex::new(None) });
    let bridge = Bridge::new(Upstream::single(HEART_HOST, Arc::new(Mem::new()), heart.clone()), &crate::ip30::PCI0);
    let r = Rig { bridge, sink: Arc::new(Sink::default()), mem: Arc::new(Mem::new()) };
    let c = r.card(2, PCI_INTA);
    r.route(2, 1);
    let (begun_tx, begun_rx) = channel();
    let (go_tx, go_rx) = channel();
    *heart.stall.lock() = Some((begun_tx, go_rx));
    let raiser = {
        let c = c.clone();
        std::thread::spawn(move || c.raise(true))
    };
    begun_rx.recv().unwrap();
    c.raise(false);
    c.raise(true);
    go_tx.send(()).unwrap();
    raiser.join().unwrap();
    assert_eq!(*heart.log.lock(), [(1, true), (1, true)]);
}

/// Slot 1 INTB and slot 5 INTA share pin 5: while the other line holds the
/// pin high, a fall and rise on one of them sends no packet.
#[test]
fn a_shared_pin_with_the_other_line_high_sends_no_packet() {
    let r = Rig::new();
    let b1 = r.card(1, 2);
    let a5 = r.card(5, PCI_INTA);
    r.route(5, 40);
    a5.raise(true);
    assert_eq!(r.sink.take(), [(40, true)]);
    b1.raise(true);
    b1.raise(false);
    b1.raise(true);
    assert!(r.sink.take().is_empty(), "the pin never fell");
}

/// A Device(x) write is the next access's window and byte order: re-aimed to
/// another PCI base the old BAR is out of reach, and switching the
/// programmed swap changes the bytes the function sees.
#[test]
fn a_device_x_write_reaches_the_next_access() {
    let r = Rig::new();
    let c = r.card(1, PCI_INTA);
    let at = Rig::win(1) + SCRATCH;
    assert_eq!(r.bridge.write32(u64::from(at), 0x0102_0304), BUS_OK);
    assert_eq!(c.peek32(SCRATCH as usize), 0x0102_0304, "word values keep the value");
    r.set(reg::device(1), DEV_MEMORY_SPACE | DEV_SWAP_PROGRAMMED | 0x130);
    assert_eq!(r.bridge.write32(u64::from(at), 0x0102_0304), BUS_OK);
    assert_eq!(c.peek32(SCRATCH as usize), 0x0403_0201, "the programmed swap is a byte stream");
    r.set(reg::device(1), DEV_MEMORY_SPACE | 0x140);
    assert!(!r.bridge.read32(u64::from(at)).is_ok(), "nothing decodes at the new base");
}

/// The IOC3's SuperIO interrupt is board wire 4, which shares the Bridge's
/// pin 4 with whatever PCI functions are wired to it.
#[test]
fn the_superio_wire_reaches_pin_4() {
    let r = Rig::new();
    let pci = r.card(4, PCI_INTA); // slot 4 INTA is pin 4 too
    r.route(4, 33);
    r.bridge.bus().board_irq(BoardIrq(4)).set(true);
    assert_eq!(r.sink.take(), [(33, true)]);
    pci.raise(true);
    r.bridge.bus().board_irq(BoardIrq(4)).set(false);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 4, "the card still holds it");
}

/// A Bridge-only reset leaves the functions' lines alone: a card still
/// holding INTA must show on the pin, and a later enable must send it.
#[test]
fn reset_keeps_the_input_levels() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.route(2, 1);
    c.raise(true);
    assert_eq!(r.sink.take(), [(1, true)]);
    r.reset();
    assert_eq!(r.get(reg::INT_STATUS), 1 << 2, "the card still holds its line");
    assert_eq!(r.get(reg::INT_ENABLE), 0);
    r.route(2, 1);
    assert_eq!(r.sink.take(), [(1, true)], "enabling a held pin sends");
    c.raise(false);
    c.raise(true);
    assert_eq!(r.sink.take(), [(1, true)]);
}

#[test]
fn a_link_reset_loses_the_core_but_not_the_lines() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.route(2, 1);
    c.raise(true);
    r.set(reg::DIRECT_MAP, 0);
    r.bridge.write64(u64::from(ATE_BASE), ATE_VALID | 0x5000);
    r.set(reg::INT_DEST_LOWER, 0x80);
    r.bridge.link_reset();
    assert_eq!(r.get(reg::DIRECT_MAP), DIRECT_ADD_512MB);
    assert_eq!(r.get(reg::INT_DEST_LOWER), 0);
    assert_eq!(r.bridge.read32(u64::from(ATE_BASE + 4)).data, 0);
    assert_eq!(r.get(reg::INT_STATUS), 1 << 2);
    // The card's configuration is not touched.
    assert_eq!(r.cfg_read(2, 0, 2), CARD_VENDOR);
}

/// The machine's reset clears every function's header. A BAR left from the
/// last boot would otherwise claim the address firmware hands another
/// function first: the stock PROM gives IOC3 the address the previous boot
/// gave the second SCSI controller, which then answered its diagnostic.
#[test]
fn a_pci_reset_forgets_the_bars_firmware_assigned() {
    let r = Rig::new();
    let stale = r.card(1, PCI_INTA);
    let c = r.card(2, PCI_INTA);
    r.reset();
    r.bridge.reset_pci();
    assert_eq!(r.cfg_read(1, 0x14, 4), 0, "BAR forgotten");
    assert_eq!(r.cfg_read(1, 0x04, 2), 0, "decoding off");
    assert_eq!(r.cfg_read(1, 0x10, 4), 1, "an I/O BAR keeps its type bit");
    assert_eq!(r.cfg_read(1, 0, 2), CARD_VENDOR);
    // Firmware assigns slot 2 alone the address both used to have.
    r.cfg_write(2, 0x14, 4, 0x1300_0000);
    r.cfg_write(2, 0x04, 2, PCI_COMMAND_MEMORY);
    r.set(reg::device(2), DEV_MEMORY_SPACE | 0x130);
    assert_eq!(r.bridge.write32(u64::from(Rig::win(2) + SCRATCH), 0x0102_0304), BUS_OK);
    assert_eq!(c.peek32(SCRATCH as usize), 0x0102_0304);
    assert_eq!(stale.peek32(SCRATCH as usize), 0);
}

/// HEART takes an interrupt packet only at its offset 0x80, for one of its
/// 64 bits; a packet anywhere else is dropped.
#[test]
fn interrupt_packets_must_land_on_hearts_interrupt_offset() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_ENABLE, 1 << 2);
    r.set(reg::int_host(2), 5);
    for lower in [0x88, 0x0, 0x7c] {
        r.set(reg::INT_DEST_LOWER, lower);
        c.raise(true);
        c.raise(false);
    }
    assert!(r.sink.take().is_empty());
    // Address bits 47:38 come from the pin's host field, not the pair.
    r.set(reg::INT_DEST_LOWER, 0x80);
    r.set(reg::INT_DEST_UPPER, ((WIDGET_HEART as u32) << 16) | 0xffc0);
    c.raise(true);
    c.raise(false);
    assert_eq!(r.sink.take(), [(5, true)]);
    r.set(reg::int_host(2), 1 << 8 | 5);
    c.raise(true);
    c.raise(false);
    assert!(r.sink.take().is_empty(), "the pin's own address bits are off 0x80");
    // Only 64 vectors exist.
    r.set(reg::int_host(2), 70);
    c.raise(true);
    assert!(r.sink.take().is_empty());
    // The same offset rule holds for the chip's own error interrupt.
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_DEST_LOWER, 0x88);
    r.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);
    let _ = r.bridge.read32(u64::from(Rig::cfg(6, 0)));
    assert!(r.sink.take().is_empty());
}

/// The host chip's widget number, packet offset and vector count are the
/// machine's to state: a packet follows the host it was built with.
#[test]
fn interrupt_packets_follow_the_host_the_bridge_was_built_with() {
    let host = XtalkHost { widget: 0xa, int_packet: 0x40, int_packet_mask: !0, vectors: 16 };
    let r = Rig::with_host(host);
    let c = r.card(2, PCI_INTA);
    r.set(reg::INT_ENABLE, 1 << 2);
    r.set(reg::int_host(2), 5);
    // HEART's address is nothing to this host.
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    c.raise(true);
    c.raise(false);
    assert!(r.sink.take().is_empty());
    r.set(reg::INT_DEST_UPPER, 0xa << 16);
    r.set(reg::INT_DEST_LOWER, 0x40);
    c.raise(true);
    c.raise(false);
    assert_eq!(r.sink.take(), [(5, true)]);
    r.set(reg::int_host(2), 16);
    c.raise(true);
    assert!(r.sink.take().is_empty(), "only 16 vectors exist");
}

/// A host with an empty packet mask takes a packet at any address, and its
/// vector count can pass 64.
#[test]
fn a_host_with_no_packet_address_takes_any_address_and_over_64_vectors() {
    let r = Rig::with_host(XtalkHost { widget: 0xa, int_packet: 0, int_packet_mask: 0, vectors: 128 });
    let c = r.card(2, PCI_INTA);
    r.set(reg::INT_ENABLE, 1 << 2);
    r.set(reg::INT_DEST_UPPER, 0xa << 16);
    r.set(reg::INT_DEST_LOWER, 0x1230);
    r.set(reg::int_host(2), 100);
    c.raise(true);
    assert_eq!(r.sink.take(), [(100, true)]);
}

/// A host that decodes the packet itself, as Bedrock's two processor
/// interfaces do, gets the whole packet: the widget ID the guest gave the
/// chip, and the destination address with the pin's host field in bits
/// 47:38. A host that refuses it costs the Bridge nothing.
#[test]
fn a_host_that_decodes_packets_sees_their_source_and_address() {
    #[derive(Default)]
    struct Decoder(Mutex<Vec<(u8, u64, u32, bool)>>);
    impl InterruptController for Decoder {
        fn set_line(&self, _: u32, _: bool) {
            unreachable!("a packet host takes packets");
        }
        fn packet(&self, source: u8, address: u64, line: u32, level: bool) -> bool {
            self.0.lock().push((source, address, line, level));
            false
        }
    }
    let host = XtalkHost { widget: 0xa, int_packet: 0, int_packet_mask: 0, vectors: 256 };
    let decoder = Arc::new(Decoder::default());
    let r = Rig { bridge: Bridge::new(Upstream::single(host, Arc::new(Mem::new()), decoder.clone()), &crate::ip30::PCI0),
        sink: Arc::new(Sink::default()), mem: Arc::new(Mem::new()) };
    let c = r.card(2, PCI_INTA);
    r.set(reg::CONTROL, (r.get(reg::CONTROL) & !0xf) | 0xd);
    r.set(reg::INT_DEST_UPPER, 0xa << 16 | 0x1);
    r.set(reg::INT_DEST_LOWER, 0x0180_0090);
    r.set(reg::int_host(2), 0x155 << 8 | 0x42);
    r.set(reg::INT_MODE, 1 << 2);
    r.set(reg::INT_ENABLE, 1 << 2);
    c.raise(true);
    c.raise(false);
    let address = (0x1_0180_0090u64 & !(0x3ff << 38)) | 0x155 << 38;
    assert_eq!(*decoder.0.lock(), [(0xd, address, 0x42, true), (0xd, address, 0x42, false)]);
}

/// Behind a Crossbow with two hosts, the widget the guest programs picks the
/// host: the direct map moves DMA and the interrupt destination moves
/// packets, and a host whose link is down hears neither.
#[test]
fn the_programmed_widget_picks_the_host_behind_a_crossbow() {
    use crate::dev::xbow::Xbow;
    use crate::xtalk::HostPort;
    let host_b = XtalkHost { widget: 10, int_packet: 0, int_packet_mask: 0, vectors: 128 };
    let build = |links: u16| {
        let xbow = Arc::new(Xbow::new(5, links));
        let (mem_a, sink_a) = (Arc::new(Mem::new()), Arc::new(Sink::default()));
        let (mem_b, sink_b) = (Arc::new(Mem::new()), Arc::new(Sink::default()));
        let up = xbow.upstream(vec![
            HostPort::new(HEART_HOST, mem_a.clone(), sink_a.clone()),
            HostPort::new(host_b, mem_b.clone(), sink_b.clone()),
        ]);
        let r = Rig { bridge: Bridge::new(up, &crate::ip30::PCI0), sink: sink_a.clone(), mem: mem_a };
        (r, mem_b, sink_b)
    };
    let (r, mem_b, sink_b) = build(1 << 8 | 1 << 10 | 1 << 15);
    let c = r.card(2, PCI_INTA);
    let dma_to = |widget: u32| r.set(reg::DIRECT_MAP, widget << DIRECT_WIDGET_SHIFT);

    dma_to(8);
    assert!(c.dma().dma_write(0x8000_1000, &[1, 2, 3, 4]).is_ok());
    assert_eq!(r.mem.peek(0x1000, 4), [4, 3, 2, 1]);
    assert_eq!(mem_b.peek(0x1000, 4), [0; 4]);
    dma_to(10);
    assert!(c.dma().dma_write(0x8000_2000, &[5, 6, 7, 8]).is_ok());
    assert_eq!(mem_b.peek(0x2000, 4), [8, 7, 6, 5]);
    assert_eq!(r.mem.peek(0x2000, 4), [0; 4]);
    let mut back = [0u8; 4];
    assert!(c.dma().dma_read(0x8000_2000, &mut back).is_ok());
    assert_eq!(back, [5, 6, 7, 8]);
    // A widget with no host is dropped, as an unreachable one always was.
    dma_to(9);
    assert_eq!(c.dma().dma_write(0x8000_3000, &[1]), PciDmaStatus::MasterAbort);

    r.set(reg::INT_ENABLE, 1 << 2);
    r.set(reg::int_host(2), 5);
    r.set(reg::INT_DEST_UPPER, 8 << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    c.raise(true);
    c.raise(false);
    assert_eq!(r.sink.take(), [(5, true)]);
    assert!(sink_b.take().is_empty());
    r.set(reg::INT_DEST_UPPER, 10 << 16);
    r.set(reg::INT_DEST_LOWER, 0);
    c.raise(true);
    c.raise(false);
    assert_eq!(sink_b.take(), [(5, true)]);
    assert!(r.sink.take().is_empty());

    // A translation entry names its own widget.
    r.set(reg::device(2), r.get(reg::device(2)) | DEV_SWAP_TRANSLATED);
    let entry = |w: u32, a: u64| ATE_VALID | ((w as u64) << ATE_TARGET_WIDGET_SHIFT) | a;
    r.bridge.write64(u64::from(ATE_BASE), entry(10, 0x5000));
    r.bridge.write64(u64::from(ATE_BASE + 8), entry(8, 0x6000));
    assert!(c.dma().dma_write(0x4000_0000, &[1, 2, 3, 4]).is_ok());
    assert!(c.dma().dma_write(0x4000_1000, &[5, 6, 7, 8]).is_ok());
    assert_eq!(mem_b.peek(0x5000, 4), [1, 2, 3, 4]);
    assert_eq!(r.mem.peek(0x5000, 4), [0; 4]);
    assert_eq!(r.mem.peek(0x6000, 4), [5, 6, 7, 8]);
    assert_eq!(mem_b.peek(0x6000, 4), [0; 4]);

    // The chip's own error interrupt goes to the destination widget too.
    r.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);
    r.set(reg::INT_DEST_UPPER, 10 << 16);
    assert!(!r.bridge.read32(u64::from(Rig::cfg(5, 0))).is_ok());
    assert_eq!(sink_b.take().len(), 1);
    assert!(r.sink.take().is_empty());
    r.set(reg::INT_RESET, INT_RESET_PCI_GRP | INT_RESET_MULTI_ERR);
    r.set(reg::INT_DEST_UPPER, 8 << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    assert!(!r.bridge.read32(u64::from(Rig::cfg(5, 0))).is_ok());
    assert_eq!(r.sink.take().len(), 1);
    assert!(sink_b.take().is_empty());

    // With no live link to host B, neither its DMA nor its interrupt arrives,
    // and a read the Crossbow drops is timed out.
    let (r, mem_b, sink_b) = build(1 << 8 | 1 << 15);
    let c = r.card(2, PCI_INTA);
    r.set(reg::DIRECT_MAP, 10 << DIRECT_WIDGET_SHIFT);
    assert_eq!(c.dma().dma_write(0x8000_1000, &[1]), PciDmaStatus::MasterAbort);
    assert_eq!(mem_b.peek(0x1000, 1), [0]);
    r.set(reg::INT_ENABLE, INT_XREAD_REQ_TIMEOUT);
    assert_eq!(c.dma().dma_read(0x8000_1000, &mut [0u8; 4]), PciDmaStatus::MasterAbort);
    assert_ne!(r.get(reg::INT_STATUS) & INT_XREAD_REQ_TIMEOUT, 0);
    r.set(reg::INT_ENABLE, 1 << 2);
    r.set(reg::int_host(2), 5);
    r.set(reg::INT_DEST_UPPER, 10 << 16);
    c.raise(true);
    assert!(sink_b.take().is_empty());
}

// ---- flash ----

fn image() -> Vec<u8> {
    (0..0x1000u32).map(|i| (i * 7 + 1) as u8).collect()
}

fn flash_rig() -> (Rig, Arc<RawFlash>, Vec<u8>) {
    let r = Rig::new();
    let img = image();
    let f = Arc::new(RawFlash::new(&img, 0x1000));
    r.bridge.attach_flash(f.clone(), 0x1000);
    (r, f, img)
}

#[test]
fn flash_reads_of_every_width_are_big_endian_image_bytes() {
    let (r, _, img) = flash_rig();
    let b = FLASH_BASE;
    assert_eq!(r.bridge.read8(u64::from(b + 5)).data, img[5]);
    assert_eq!(r.bridge.read16(u64::from(b + 4)).data, u16::from_be_bytes([img[4], img[5]]));
    assert_eq!(r.bridge.read32(u64::from(b + 8)).data, u32::from_be_bytes(img[8..12].try_into().unwrap()));
    assert_eq!(r.bridge.read64(u64::from(b + 16)).data, u64::from_be_bytes(img[16..24].try_into().unwrap()));
    // The part repeats across its chip select above the lines it decodes.
    assert_eq!(r.bridge.read32(u64::from(b + 0x1000 + 8)).data, r.bridge.read32(u64::from(b + 8)).data);
    assert_eq!(r.bridge.read32(u64::from(b + 0x1f_fffc)).data, r.bridge.read32(u64::from(b + 0xffc)).data);
}

#[test]
fn the_second_chip_select_is_empty_unless_fitted_or_strapped() {
    let (r, f, _) = flash_rig();
    assert!(!r.bridge.read32(u64::from(FLASH_BASE + FLASH_SELECT_SIZE)).is_ok());
    r.bridge.set_flash_strap(true);
    assert_eq!(r.get(reg::STATUS) & STATUS_FLASH_SELECT, STATUS_FLASH_SELECT);
    assert!(!r.bridge.read32(u64::from(FLASH_BASE)).is_ok(), "the strap swaps the selects");
    assert!(r.bridge.read32(u64::from(FLASH_BASE + FLASH_SELECT_SIZE)).is_ok());
    r.reset();
    assert_eq!(r.get(reg::STATUS) & STATUS_FLASH_SELECT, STATUS_FLASH_SELECT, "the strap survives reset");
    r.bridge.set_flash_strap(false);
    r.bridge.attach_flash_select1(f, 0x1000);
    assert!(r.bridge.read32(u64::from(FLASH_BASE + FLASH_SELECT_SIZE)).is_ok());
}

#[test]
fn flash_writes_need_the_enable_and_a_sixteen_bit_aligned_cycle() {
    let (r, f, _) = flash_rig();
    assert_eq!(r.bridge.write16(u64::from(FLASH_BASE), 0x9090), BUS_ERR, "write enable clear");
    r.set(reg::CONTROL, r.get(reg::CONTROL) | CONTROL_FLASH_WRITE_EN);
    assert_eq!(r.bridge.write8(u64::from(FLASH_BASE), 0x90), BUS_ERR);
    assert_eq!(r.bridge.write32(u64::from(FLASH_BASE), 0x9090_9090), BUS_ERR);
    assert_eq!(r.bridge.write16(u64::from(FLASH_BASE + 1), 0x9090), BUS_ERR);
    assert_eq!(f.dropped_writes(), 0);
    assert_eq!(r.bridge.write16(u64::from(FLASH_BASE), 0x9090), BUS_OK);
    assert_eq!(f.dropped_writes(), 1, "the raw image has no command interface");
}

#[test]
fn flash_sits_where_the_reset_vector_is() {
    // CPU physical 0x1fc0_0000 is widget 15's small window (0x1000_0000
    // plus 15 << 24) plus 0xc0_0000.
    assert_eq!(0x1000_0000 + ((WIDGET_BRIDGE as u32) << 24) + FLASH_BASE, 0x1fc0_0000);
}

#[test]
fn dma_above_4_gib_master_aborts_instead_of_wrapping() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.direct_to_heart(true);
    // The low 32 bits alone would be a valid direct-map address.
    assert_eq!(c.dma().dma_write(0x1_8000_0010, &[9]), PciDmaStatus::MasterAbort);
    let mut b = [0xffu8; 1];
    assert_eq!(c.dma().dma_read(0x1_8000_0010, &mut b), PciDmaStatus::MasterAbort);
    assert_eq!(b, [0]);
    assert_eq!(r.mem.peek(HIGH + 0x10, 1), [0]);
}

/// The IRIX ISP driver maps all its DMA through the 64-bit direct window:
/// the widget in bits 63:60, the XIO address in the low 48 bits.
#[test]
fn dma_through_the_64_bit_direct_window() {
    let r = Rig::new();
    let c = r.card(0, PCI_INTA);
    let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
    let at = |x: u64| ((WIDGET_HEART as u64) << 60) | x;

    // Swap clear on a 32-bit slot reverses words; the translated-window
    // swap bit does not steer this window.
    assert!(c.dma().dma_write(at(0x1000), &data).is_ok());
    assert_eq!(r.mem.peek(0x1000, 8), [4, 3, 2, 1, 8, 7, 6, 5]);
    let mut back = [0u8; 8];
    assert!(c.dma().dma_read(at(0x1000), &mut back).is_ok());
    assert_eq!(back, data);

    r.set(reg::device(0), r.get(reg::device(0)) | DEV_SWAP_TRANSLATED);
    assert!(c.dma().dma_write(at(0x2000), &data).is_ok());
    assert_eq!(r.mem.peek(0x2000, 8), [4, 3, 2, 1, 8, 7, 6, 5]);

    // The direct swap bit makes it a byte stream.
    r.set(reg::device(0), r.get(reg::device(0)) | DEV_SWAP_DIRECT);
    assert!(c.dma().dma_write(at(0x3000), &data).is_ok());
    assert_eq!(r.mem.peek(0x3000, 8), data);
    assert!(c.dma().dma_read(at(0x3000), &mut back).is_ok());
    assert_eq!(back, data);

    // Bits 47:0 pass through whole, unlike the 31-bit window's map.
    assert!(c.dma().dma_write(at(HIGH + 0x10), &[9]).is_ok());
    assert_eq!(r.mem.peek(HIGH + 0x10, 1), [9]);

    // A 64-bit slot reverses doublewords.
    r.set(reg::device(0), r.get(reg::device(0)) & !DEV_SWAP_DIRECT);
    r.bridge.set_dma_64bit(0, true);
    assert!(c.dma().dma_write(at(0x4000), &data).is_ok());
    assert_eq!(r.mem.peek(0x4000, 8), [8, 7, 6, 5, 4, 3, 2, 1]);
}

#[test]
fn the_64_bit_direct_window_refuses_widget_zero_and_other_widgets() {
    let r = Rig::new();
    let c = r.card(0, PCI_INTA);
    r.set(reg::INT_ENABLE, INT_XREAD_REQ_TIMEOUT);
    r.set(reg::INT_DEST_UPPER, (WIDGET_HEART as u32) << 16);
    r.set(reg::INT_DEST_LOWER, 0x80);
    assert_eq!(c.dma().dma_write(0x1000, &[1]), PciDmaStatus::MasterAbort);
    assert_eq!(c.dma().dma_write(0x1_0000_1000, &[1]), PciDmaStatus::MasterAbort, "widget 0");
    assert_eq!(c.dma().dma_write((9u64 << 60) | 0x1000, &[1]), PciDmaStatus::MasterAbort, "not HEART");
    r.bridge.set_link_alive(Box::new(|w| w == WIDGET_HEART || w == 9));
    let mut b = [0xffu8; 2];
    assert_eq!(c.dma().dma_read((9u64 << 60) | 0x1000, &mut b), PciDmaStatus::MasterAbort);
    assert_eq!(b, [0, 0]);
    assert_eq!(r.get(reg::INT_STATUS), 0, "refusals and peer DMA raise no timeout");
    assert_eq!(r.mem.peek(0x1000, 1), [0]);
}

// ---- lines into the Bridge, and the packets out of it ----

/// IOC3's SuperIO interrupt is wired to a pin directly; the line shares the
/// pin with PCI functions and does not keep the Bridge alive.
#[test]
fn a_board_irq_line_drives_its_pin_and_does_not_keep_the_bridge_alive() {
    let r = Rig::new();
    r.route(4, 33);
    let line = r.bridge.bus().board_irq(BoardIrq(4));
    line.set(true);
    assert_eq!(r.sink.take(), [(33, true)]);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 4);
    line.set(false);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 0);

    let weak = Arc::downgrade(&r.bridge);
    let Rig { bridge, .. } = r;
    drop(bridge);
    assert!(weak.upgrade().is_none(), "a line is not a reference");
    line.set(true);
}

/// HEART is driven with no Bridge lock held: it may look at the Bridge's
/// registers while handling the packet.
#[test]
fn a_packet_reaches_heart_outside_the_state_lock() {
    struct Prying(Mutex<Option<Arc<Bridge>>>, Mutex<Vec<u32>>);
    impl InterruptController for Prying {
        fn set_line(&self, _line: u32, _level: bool) {
            let b = self.0.lock().clone().unwrap();
            self.1.lock().push(b.register(reg::INT_STATUS));
        }
    }
    let heart = Arc::new(Prying(Mutex::new(None), Mutex::new(Vec::new())));
    let bridge = Bridge::new(Upstream::single(HEART_HOST, Arc::new(Mem::new()), heart.clone()), &crate::ip30::PCI0);
    *heart.0.lock() = Some(bridge.clone());
    let r = Rig { bridge, sink: Arc::new(Sink::default()), mem: Arc::new(Mem::new()) };
    r.route(4, 3);
    r.bridge.bus().board_irq(BoardIrq(4)).set(true);
    assert_eq!(*heart.1.lock(), [1 << 4], "the status already shows the level");
    // Errors leave the same way.
    r.set(reg::INT_ENABLE, 1 << 4 | INT_PCI_MASTER_TIMEOUT);
    assert!(!r.bridge.read32(u64::from(Rig::cfg(6, 0))).is_ok());
    assert_eq!(heart.1.lock().len(), 2);
    *heart.0.lock() = None;
}

/// Two threads toggling one pin: HEART sees the packets in the order the
/// levels were decided, so the last packet matches the final level.
#[test]
fn edges_from_two_threads_reach_heart_in_decision_order() {
    let r = Rig::new();
    r.route(4, 7);
    r.set(reg::INT_MODE, 1 << 4);
    let (b1, b2) = (r.bridge.bus().board_irq(BoardIrq(4)), r.bridge.bus().board_irq(BoardIrq(4)));
    let t1 = std::thread::spawn(move || (0..2000).for_each(|_| { b1.set(true); b1.set(false); }));
    let t2 = std::thread::spawn(move || (0..2000).for_each(|_| { b2.set(true); b2.set(false); }));
    t1.join().unwrap();
    t2.join().unwrap();
    let packets = r.sink.take();
    // Packets alternate set, clear: every clear is preceded by its set.
    for (i, p) in packets.iter().enumerate() {
        assert_eq!(*p, (7, i % 2 == 0), "packet {i}");
    }
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 0);
}

// ---- from the first Octane bring-up's register tests ----

/// As the stock PROM does: memory space on, a function's BAR0 at PCI
/// 0x400000, and window 2's Device register based there; then a byte
/// scratch register and a status bit read through it.
#[test]
fn window_2_reaches_a_bar_the_way_the_prom_points_it() {
    let r = Rig::new();
    let c = Card::new(PCI_INTA);
    r.fit(2, c.clone());
    // Nothing decodes before the setup.
    assert!(!r.bridge.read8(u64::from(Rig::win(2) + SCRATCH)).is_ok());
    r.cfg_write(2, 0x04, 4, PCI_COMMAND_MEMORY);
    r.cfg_write(2, 0x14, 4, 0x0040_0000);
    r.set(reg::device(2), DEV_MEMORY_SPACE | 4);
    assert_eq!(Rig::win(2), 0x60_0000);
    for i in 0..4u32 {
        assert_eq!(r.bridge.write8(u64::from(Rig::win(2) + ((SCRATCH + i) ^ 3)), 0x5a + i as u8), BUS_OK);
    }
    assert_eq!(c.peek32(SCRATCH as usize), 0x5d5c_5b5a);
    assert_eq!(r.bridge.read8(u64::from(Rig::win(2) + (SCRATCH ^ 3))).data, 0x5a);
}

/// The Bridge answers as widget part 0xc002; chip select 1 of the flash
/// window is a blank 2 MB part reading ones while chip select 0 holds the PROM.
#[test]
fn flash_is_in_the_external_select_and_the_other_reads_ones() {
    let r = Rig::new();
    let img = image();
    r.bridge.attach_flash(Arc::new(RawFlash::new(&img, 0x1000)), 0x1000);
    r.bridge.attach_flash_select1(Arc::new(RawFlash::new(&[], 2 << 20)), 2 << 20);
    assert_eq!(r.bridge.read32(u64::from(FLASH_BASE)).data, u32::from_be_bytes(img[..4].try_into().unwrap()));
    assert_eq!(r.bridge.read32(u64::from(FLASH_BASE + FLASH_SELECT_SIZE)).data, 0xffff_ffff);
    assert_eq!((r.get(reg::IDENTIFICATION) >> 12) & 0xffff, 0xc002);
}

/// The bus hands the window the full physical address; it decodes its own 16 MB.
#[test]
fn the_window_decodes_the_low_24_bits_of_a_physical_address() {
    let r = Rig::new();
    assert_eq!(r.bridge.read32(0x1f00_0000 + u64::from(reg::IDENTIFICATION + 4)).data, r.get(reg::IDENTIFICATION));
}

#[test]
fn long_and_short_transfers_place_every_byte_by_the_swap_rule() {
    let r = Rig::new();
    let c = r.card(2, PCI_INTA);
    r.direct_to_heart(false);
    // (swap bit, 64-bit slot, mask): the PCI byte at p is memory byte p ^ mask.
    for (swap, wide, mask) in [(true, false, 0usize), (false, false, 3), (false, true, 7)] {
        let d = r.get(reg::device(2)) & !DEV_SWAP_DIRECT;
        r.set(reg::device(2), if swap { d | DEV_SWAP_DIRECT } else { d });
        r.bridge.set_dma_64bit(2, wide);
        // Aligned and unaligned starts and lengths, around the 4 KB split
        // and the size a stack copy holds.
        for (off, len) in [(0usize, 8usize), (0, 4096), (0, 4104), (0, 8192), (8, 12288), (1, 4096), (0, 4095), (3, 9), (5, 8192)] {
            let data: Vec<u8> = (0..len).map(|i| (i * 31 + 7) as u8).collect();
            r.mem.poke(0x2000, &vec![0xee; 16384]);
            assert!(c.dma().dma_write(0x8000_2000 + off as u64, &data).is_ok());
            let lo = 0x2000 + off & !7;
            let hi = (0x2000 + off + len + 7) & !7;
            let mut want = vec![0xee; hi - lo];
            for (i, &b) in data.iter().enumerate() {
                let p = 0x2000 + off + i;
                want[(p ^ mask) - lo] = b;
            }
            assert_eq!(r.mem.peek(lo as u64, hi - lo), want, "write mask {mask} at {off} for {len}");
            let mut back = vec![0u8; len];
            assert!(c.dma().dma_read(0x8000_2000 + off as u64, &mut back).is_ok());
            assert_eq!(back, data, "read mask {mask} at {off} for {len}");
        }
    }
}

/// A saved Bridge loads into a fresh one with its registers, its table and
/// its functions' headers, and the BAR windows reach the function again; a
/// bus with no function in the slot refuses the load.
#[test]
fn save_load_round_trip_keeps_the_function_reachable() {
    use crate::traits::Saveable;
    let src = Rig::new();
    let _c = src.card(1, PCI_INTA);
    let v1 = src.bridge.save_state();
    let dst = Rig::new();
    let c = dst.card(1, PCI_INTA);
    // Unmapped before the load: the load must bring the window back.
    dst.set(reg::device(1), 0);
    dst.cfg_write(1, 0x04, 2, 0);
    dst.bridge.load_state(&v1).expect("load_state");
    assert_eq!(v1, dst.bridge.save_state());
    assert_eq!(dst.get(reg::device(1)), src.get(reg::device(1)));
    let w = Rig::win(1);
    assert_eq!(dst.bridge.write32(u64::from(w + SCRATCH), 0x1234_5678), BUS_OK);
    assert_eq!(c.peek32(SCRATCH as usize), 0x1234_5678);
    assert!(Rig::new().bridge.load_state(&v1).is_err());
    let other = Rig::new();
    let _c = other.card(1, PCI_INTA);
    other.bridge.set_revision(4);
    assert!(other.bridge.load_state(&v1).unwrap_err().contains("revision"));
}

/// CC and CSTS through slot 1's device window in each of its lane modes, by
/// word and by doubleword: CSTS.RDY is set by the store that sets CC.EN and
/// is there for the next load, whichever width either uses, and a store that
/// clears CC.EN clears it.
#[test]
fn nvme_ready_follows_enable_through_every_lane_mode_and_width() {
    #[derive(Clone, Copy, Debug)]
    enum Mode {
        ByteStream,
        Words4,
        Words8,
    }
    for mode in [Mode::ByteStream, Mode::Words4, Mode::Words8] {
        let r = Rig::new();
        let img = crate::dev::nvme::tests::temp_image("bridge-lanes");
        let card = crate::dev::nvme::Card::open(&crate::config::NvmeConfig {
            path: img.to_string_lossy().into_owned(),
            overlay: false,
            bridge: false,
            ..Default::default()
        })
        .unwrap();
        card.fit(r.bridge.bus(), 1);
        let bits = match mode {
            Mode::ByteStream => DEV_SWAP_PROGRAMMED,
            Mode::Words4 => 0,
            Mode::Words8 => DEV_64BIT_DEVICE,
        };
        r.set(reg::device(1), DEV_MEMORY_SPACE | bits | 0x130);
        r.cfg_write(1, 0x10, 4, 0x1300_0000);
        r.cfg_write(1, 0x14, 4, 0);
        r.cfg_write(1, 0x04, 2, PCI_COMMAND_MEMORY);
        let at = |off: u32| u64::from(Rig::win(1) + off);
        // The CPU's word store or load that reaches register `off`.
        let st32 = |off: u32, v: u32| {
            let s = match mode {
                Mode::ByteStream => r.bridge.write32(at(off), v.swap_bytes()),
                Mode::Words4 => r.bridge.write32(at(off), v),
                Mode::Words8 => r.bridge.write32(at(off ^ 4), v),
            };
            assert_eq!(s, BUS_OK);
        };
        let ld32 = |off: u32| match mode {
            Mode::ByteStream => r.bridge.read32(at(off)).data.swap_bytes(),
            Mode::Words4 => r.bridge.read32(at(off)).data,
            Mode::Words8 => r.bridge.read32(at(off ^ 4)).data,
        };
        // CC by a doubleword store to 0x10 (INTMC, which takes a zero) and
        // CSTS from a doubleword load at 0x18.
        let st64_cc = |cc: u32| {
            let v = match mode {
                Mode::ByteStream => u64::from(cc.swap_bytes()),
                Mode::Words4 => u64::from(cc),
                Mode::Words8 => u64::from(cc) << 32,
            };
            assert_eq!(r.bridge.write64(at(0x10), v), BUS_OK);
        };
        let ld64_csts = || {
            let d = r.bridge.read64(at(0x18)).data;
            match mode {
                Mode::ByteStream => (d as u32).swap_bytes(),
                Mode::Words4 => d as u32,
                Mode::Words8 => (d >> 32) as u32,
            }
        };
        assert_eq!(ld32(0x08), 0x0001_0400, "{mode:?}: VS");
        st32(0x24, 3 | 3 << 16);
        st32(0x28, 0x8000_1000);
        st32(0x30, 0x8000_2000);
        let cc = 1 | 6 << 16 | 4 << 20;
        st32(0x14, cc);
        assert_eq!((ld32(0x14), ld32(0x1c), ld64_csts()), (cc, 1, 1), "{mode:?}: word store, ready at once");
        st32(0x14, 0);
        assert_eq!((ld32(0x1c), ld64_csts()), (0, 0), "{mode:?}: disabled");
        st64_cc(cc);
        assert_eq!((ld32(0x14), ld32(0x1c), ld64_csts()), (cc, 1, 1), "{mode:?}: doubleword store, ready at once");
        st64_cc(0);
        assert_eq!(ld32(0x1c), 0, "{mode:?}: disabled by a doubleword store");
    }
}

// ---- the XBridge head ----
//
// Cases follow tests/qtest/sn1-xbridge-test.c (page-dma, interrupt-packets,
// access-errors) in jamesbraid/qemu-sgi-origami.

/// Bedrock's port as an XBridge head sees it: widget 10, packets taken at
/// any address.
const BEDROCK_HOST: XtalkHost = XtalkHost { widget: 0xa, int_packet: 0, int_packet_mask: 0, vectors: 256 };

/// One XBridge head at crossbar port 15, strapped for a 66 MHz bus, with
/// memory and an interrupt sink behind widget 10.
struct Head {
    bridge: Arc<Bridge>,
    win: Arc<dyn BusDevice>,
    sink: Arc<Sink>,
    mem: Arc<Mem>,
}

impl Head {
    fn new() -> Self {
        let sink = Arc::new(Sink::default());
        let mem = Arc::new(Mem::new());
        let bridge = Bridge::with_part(Part::XBridge, Upstream::single(BEDROCK_HOST, mem.clone(), sink.clone()), &crate::ip30::PCI0);
        bridge.set_revision(2);
        bridge.set_widget(15);
        bridge.set_control_strap(1 << 4);
        let win = bridge.head_window();
        Head { bridge, win, sink, mem }
    }

    fn set(&self, off: u32, val: u32) {
        assert_eq!(self.win.write32(u64::from(off + 4), val), BUS_OK);
    }

    fn get(&self, off: u32) -> u32 {
        self.win.read32(u64::from(off + 4)).data
    }

    /// A card in `slot` as [`Rig::card`] fits one: memory BAR at PCI
    /// 0x1300_0000, I/O BAR at 0x2000, the slot's window on memory.
    fn card(&self, slot: u8, pin: u8) -> Arc<Card> {
        let c = Card::new(pin);
        self.bridge.bus().attach(DevFn { device: slot, function: 0 }, c.clone());
        let cfg = |reg: u32, size: usize, v: u64| {
            let off = u64::from(Rig::cfg(slot as u32, reg) ^ (4 - size as u32) & 3);
            match size {
                2 => assert_eq!(self.win.write16(off, v as u16), BUS_OK),
                _ => assert_eq!(self.win.write32(off, v as u32), BUS_OK),
            }
        };
        cfg(0x10, 4, 0x2000);
        cfg(0x14, 4, 0x1300_0000);
        cfg(0x04, 2, u64::from(PCI_COMMAND_IO | PCI_COMMAND_MEMORY));
        self.set(reg::device(slot as u32), DEV_MEMORY_SPACE | 0x130);
        c
    }
}

#[test]
fn an_xbridge_head_identifies_itself_and_keeps_its_speed_strap() {
    let h = Head::new();
    assert_eq!(h.get(reg::IDENTIFICATION), 0x2d00_2049);
    assert_eq!(h.get(reg::CONTROL), 0x0f00_321f);
    h.set(reg::CONTROL, 0);
    assert_eq!(h.get(reg::CONTROL), 0x10, "the speed strap is not the guest's to write");
    h.bridge.reset();
    assert_eq!(h.get(reg::CONTROL), 0x0f00_321f);
    for (off, val, kept) in [
        (reg::DIRECT_MAP, u32::MAX, 0xf3_ffff),
        (reg::BUS_TIMEOUT, u32::MAX, 0x1f_03ff),
        (reg::INT_DEST_UPPER, u32::MAX, 0xf_ffff),
        (reg::TEST_PIN_CONTROL, u32::MAX, 0xff_ffff),
        (reg::device(2), u32::MAX, 0x1eef_ffff),
    ] {
        h.set(off, val);
        assert_eq!(h.get(off), kept, "{off:#x}");
    }
}

/// IRIX's driver treats all 1024 entries like the first 512 and zeroes them at
/// attach (established from the driver's code); the head keeps every one, so
/// the upper half stores and reads back like the lower, and a reset zeroes
/// the whole table.
#[test]
fn every_entry_of_an_xbridge_table_stores_and_a_reset_zeroes_them_all() {
    let h = Head::new();
    for i in [0u64, 511, 512, 1023] {
        h.win.write64(u64::from(ATE_BASE) + i * 8, 0x1234_0000_0000 | i);
        assert_eq!(h.win.read32(u64::from(XBRIDGE_ATE_LOW_BASE) + i * 8 + 4).data, i as u32, "entry {i}");
    }
    h.bridge.reset();
    for i in 0..1024u64 {
        assert_eq!(h.win.read32(u64::from(XBRIDGE_ATE_LOW_BASE) + i * 8 + 4).data, 0, "entry {i} low");
        assert_eq!(h.win.read32(u64::from(ATE_BASE) + i * 8 + 4).data, 0, "entry {i} high");
    }
}

/// The head's table holds 1024 entries, written whole and read a half at a
/// time; 512 of them map a window that repeats with bytes swapped at bit
/// 29, and an entry's own swap bit swaps too. A transfer through an entry
/// that is not valid, or past the mapped ones, latches a page fault, which
/// the SSRAM group's reset clears. The 64 KB below the window takes
/// transfers and drops them.
#[test]
fn an_xbridge_head_maps_dma_through_its_larger_table() {
    let h = Head::new();
    let c = h.card(2, PCI_INTA);
    h.win.write64(u64::from(ATE_BASE) + 800 * 8, 0xffff_1234_5678_9abc);
    assert_eq!(h.win.read32(u64::from(ATE_BASE) + 800 * 8 + 4).data, 0x1234, "the entry keeps 48 bits");
    assert_eq!(h.win.read32(u64::from(XBRIDGE_ATE_LOW_BASE) + 800 * 8 + 4).data, 0x5678_9abc);
    h.win.write64(u64::from(ATE_BASE), 0x2000 | 0xa << 8 | 1);
    h.win.write64(u64::from(ATE_BASE) + 8, 0x3000 | 0xa << 8 | 0x21);
    let data = [0, 1, 2, 3, 4, 5, 6, 7];
    for (pci, at, placed) in [
        (0x4000_0010u64, 0x2010, [3, 2, 1, 0, 7, 6, 5, 4]),
        (0x6000_0020, 0x2020, data),
        (0x4000_1030, 0x3030, data),
    ] {
        assert_eq!(c.dma().dma_write(pci, &data), PciDmaStatus::Ok, "{pci:#x}");
        assert_eq!(h.mem.peek(at, 8), placed, "{pci:#x}");
        let mut back = [0; 8];
        assert_eq!(c.dma().dma_read(pci, &mut back), PciDmaStatus::Ok);
        assert_eq!(back, data);
    }
    assert_eq!(h.get(reg::INT_STATUS) & INT_PAGE_FAULT, 0);
    for pci in [0x4000_2000, 0x4020_0000] {
        let mut bytes = [0xff; 4];
        assert_eq!(c.dma().dma_read(pci, &mut bytes), PciDmaStatus::MasterAbort, "{pci:#x}");
        assert_eq!(bytes, [0; 4]);
        assert_ne!(h.get(reg::INT_STATUS) & INT_PAGE_FAULT, 0, "{pci:#x}");
        assert_ne!(h.get(reg::INT_MULTIPLE) & INT_PAGE_FAULT, 0);
        h.set(reg::INT_RESET, 2);
        assert_eq!(h.get(reg::INT_STATUS) & INT_PAGE_FAULT, 0);
    }
    let mut bytes = [0xff; 4];
    assert_eq!(c.dma().dma_read(0x3fff_8000, &mut bytes), PciDmaStatus::Ok);
    assert_eq!(bytes, [0; 4]);
    assert_eq!(c.dma().dma_write(0x3fff_8000, &data), PciDmaStatus::Ok);
    assert_eq!(c.dma().dma_read(0x4080_0000, &mut bytes), PciDmaStatus::MasterAbort, "past the window");
}

/// A 64-bit transfer names its widget in bits 63:60 and asks for its bytes
/// unswapped with bit 55; the 31-bit direct window follows Device(x).
#[test]
fn an_xbridge_head_swaps_64_bit_dma_by_its_address() {
    let h = Head::new();
    let c = h.card(2, PCI_INTA);
    let data = [0, 1, 2, 3, 4, 5, 6, 7];
    assert_eq!(c.dma().dma_write(0xa << 60 | 1 << 55 | 0x2000, &data), PciDmaStatus::Ok);
    assert_eq!(h.mem.peek(0x2000, 8), data);
    assert_eq!(c.dma().dma_write(0xa << 60 | 0x2010, &data), PciDmaStatus::Ok);
    assert_eq!(h.mem.peek(0x2010, 8), [3, 2, 1, 0, 7, 6, 5, 4]);
    h.set(reg::DIRECT_MAP, 0xa << DIRECT_WIDGET_SHIFT);
    h.set(reg::device(2), DEV_MEMORY_SPACE | DEV_SWAP_DIRECT | 0x130);
    assert_eq!(c.dma().dma_write(0x8000_2020, &data), PciDmaStatus::Ok);
    assert_eq!(h.mem.peek(0x2020, 8), data);
}

/// The head reaches PCI memory through two aliases and I/O through a third,
/// each byte order from Control; nothing claiming an address reads ones,
/// and takes a write, with no error.
#[test]
fn an_xbridge_head_reaches_pci_space_through_its_three_aliases() {
    let h = Head::new();
    let _c = h.card(2, PCI_INTA);
    for at in [0x4000_0000u64 + 0x1300_0020, 0x8000_0000 + 0x1300_0020, 0x1_0000_0000 + 0x2020] {
        assert_eq!(h.win.read32(at).data, 0x1234_5678, "{at:#x}");
    }
    h.set(reg::CONTROL, CONTROL_IO_SWAP);
    assert_eq!(h.win.read32(0x1_0000_2020).data, 0x7856_3412, "the I/O alias swaps on its own bit");
    assert_eq!(h.win.read32(0x4000_0000 + 0x1300_0020).data, 0x1234_5678);
    h.set(reg::INT_ENABLE, INT_PCI_MASTER_TIMEOUT);
    assert_eq!(h.win.read32(0x4800_0000).data, u32::MAX);
    assert_eq!(h.win.write32(0x4800_0000, 0), BUS_OK);
    assert_eq!(h.win.read32(u64::from(device_window_offset(3))).data, u32::MAX);
    assert_eq!(h.get(reg::INT_STATUS) & !INT_DEVICE_PINS, 0);
}

/// An empty slot reads as ones and notes the master time-out in the
/// multiple-error register; a doubleword is two words, the lower address
/// in the upper half; a type 1 cycle for bus 0 reaches the head's own bus.
#[test]
fn an_xbridge_head_answers_configuration_cycles_as_its_bus_does() {
    let h = Head::new();
    let _c = h.card(2, PCI_INTA);
    assert_eq!(h.win.read32(u64::from(Rig::cfg(5, 0))).data, u32::MAX);
    assert_ne!(h.get(reg::INT_MULTIPLE) & INT_PCI_MASTER_TIMEOUT, 0);
    assert_eq!(h.get(reg::INT_STATUS) & INT_PCI_MASTER_TIMEOUT, 0);
    let id = u64::from(CARD_DEVICE << 16 | CARD_VENDOR);
    let command = u64::from(h.win.read32(u64::from(Rig::cfg(2, 4))).data);
    assert_eq!(h.win.read64(u64::from(Rig::cfg(2, 0))).data, id << 32 | command);
    h.set(reg::PCI_TYPE1_CONFIG, 2 << 11);
    assert_eq!(h.win.read32(u64::from(TYPE1_BASE)).data as u64, id);
}

/// An undefined address reads zero and latches an invalid-address error,
/// capturing the first address and interrupting for it once; the deferred
/// registers read zero with no error.
#[test]
fn an_xbridge_head_captures_its_first_invalid_address() {
    let h = Head::new();
    h.set(reg::INT_DEST_UPPER, 0xa << 16);
    h.set(reg::INT_DEST_LOWER, 0x80);
    h.set(reg::HOST_ERR_FIELD, 73);
    h.set(reg::INT_ENABLE, INT_INVALID_ADDR);
    assert_eq!(h.win.read32(0x300).data, 0, "deferred");
    assert!(h.sink.take().is_empty());
    assert_eq!(h.win.read32(0x100_0004).data, 0);
    assert_eq!(h.win.read32(0x200_0004).data, 0);
    assert_eq!(h.get(reg::ERR_LOWER), 0x100_0004);
    assert_eq!(h.sink.take(), [(73, true)]);
    assert_ne!(h.get(reg::INT_MULTIPLE) & INT_INVALID_ADDR, 0);
    h.set(reg::INT_RESET, 8);
    assert_eq!(h.get(reg::INT_STATUS) & INT_INVALID_ADDR, 0);
}

/// A pin's packet goes out even to an unprogrammed host field, and the
/// force registers send a pin's packet: always, or only while it is high.
#[test]
fn an_xbridge_head_forces_its_interrupts() {
    let h = Head::new();
    let c = h.card(2, PCI_INTA);
    h.set(reg::INT_DEST_UPPER, 0xa << 16);
    h.set(reg::INT_ENABLE, 1 << 2);
    c.raise(true);
    assert_eq!(h.sink.take(), [(0, true)]);
    h.set(reg::int_host(2), 0x42);
    h.set(reg::FORCE_ALWAYS + 2 * 8, 1);
    h.set(reg::FORCE_PIN + 2 * 8, 1);
    assert_eq!(h.sink.take(), [(0x42, true), (0x42, true)]);
    c.raise(false);
    h.set(reg::FORCE_PIN + 2 * 8, 1);
    h.set(reg::FORCE_ALWAYS + 3 * 8, 1);
    assert!(h.sink.take().is_empty(), "pin 2 is low and pin 3 is masked");
    assert_eq!(h.get(reg::FORCE_ALWAYS + 2 * 8), 0);
}

/// A snapshot keeps all 1024 entries and the multiple-error register.
#[test]
fn an_xbridge_head_snapshot_keeps_its_table_and_errors() {
    let src = Head::new();
    src.win.write64(u64::from(ATE_BASE) + 1000 * 8, 0x5000 | 0xa << 8 | 1);
    let _ = src.win.read32(u64::from(Rig::cfg(5, 0)));
    let v = src.bridge.save_state();
    let dst = Head::new();
    dst.bridge.load_state(&v).expect("load_state");
    assert_eq!(v, dst.bridge.save_state());
    assert_eq!(dst.win.read32(u64::from(XBRIDGE_ATE_LOW_BASE) + 1000 * 8 + 4).data, 0x5a01);
    assert!(Rig::new().bridge.load_state(&v).is_err(), "a Bridge holds 128 entries");
}
