//! Register-contract tests for the Bridge, against fake functions, memory
//! and an interrupt sink. They follow the reference model's register-contract tests: configuration
//! space with narrow accesses, an empty slot, one register reached through
//! both BARs, the programmed byte lanes of a device window, and a function's
//! DMA reaching memory through the direct map and the translation table.
//!
//! Ported from tests/qtest/sgi-bridge-bus-test.c in jamesbraid/qemu-sgi-origami.

use super::*;
use crate::irq::IrqLine;
use crate::pci::{Bar, PciConfig, PCI_COMMAND_IO, PCI_COMMAND_MEMORY, PCI_INTA};

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
    cfg: PciConfig,
    regs: Mutex<[u8; 0x1000]>,
    irq: Mutex<Option<IrqLine>>,
    dma: Mutex<Option<Arc<dyn PciDma>>>,
}

impl Card {
    fn new(pin: u8) -> Arc<Self> {
        let mut regs = [0u8; 0x1000];
        regs[FIXED as usize..FIXED as usize + 4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        Arc::new(Self {
            cfg: PciConfig::new(
                CARD_VENDOR as u16,
                CARD_DEVICE as u16,
                0x01_0000,
                1,
                [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                pin,
            ),
            regs: Mutex::new(regs),
            irq: Mutex::new(None),
            dma: Mutex::new(None),
        })
    }

    fn peek32(&self, off: usize) -> u32 {
        u32::from_le_bytes(self.regs.lock()[off..off + 4].try_into().unwrap())
    }

    fn raise(&self, level: bool) {
        self.irq.lock().as_ref().unwrap().set(level);
    }

    fn dma(&self) -> Arc<dyn PciDma> {
        self.dma.lock().clone().unwrap()
    }
}

impl PciBusDevice for Card {
    fn config(&self) -> &PciConfig {
        &self.cfg
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

    fn connect_interrupt(&self, line: IrqLine) {
        *self.irq.lock() = Some(line);
    }

    fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        *self.dma.lock() = Some(dma);
    }
}

struct Rig {
    bridge: Arc<Bridge>,
    sink: Arc<Sink>,
    mem: Arc<Mem>,
}

impl Rig {
    fn new() -> Self {
        let sink = Arc::new(Sink::default());
        let mem = Arc::new(Mem::new());
        Rig { bridge: Bridge::new(mem.clone(), sink.clone()), sink, mem }
    }

    /// Write register `off` (a slot base such as `reg::CONTROL`).
    fn set(&self, off: u32, val: u32) {
        assert_eq!(self.bridge.write32(u64::from(off + 4), val), BUS_OK);
    }

    fn reset(&self) {
        self.bridge.reset();
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
        self.bridge.attach(slot, c.clone());
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
    r.bridge.attach(1, Card::new(PCI_INTA));
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
    r.bridge.attach(1, c.clone());
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
    r.bridge.attach(2, Card::new(PCI_INTA));
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
    r.bridge.set_wire_irq(4, true);
    assert_eq!(r.sink.take(), [(33, true)]);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 4);
    r.bridge.set_wire_irq(4, false);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 0);
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
/// pin with PCI functions and holds the Bridge weakly.
#[test]
fn a_pin_line_drives_its_pin_and_does_not_keep_the_bridge_alive() {
    let r = Rig::new();
    r.route(4, 33);
    let line = r.bridge.pin_line(4);
    assert_eq!(line.line(), 4);
    line.set(true);
    assert_eq!(r.sink.take(), [(33, true)]);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 1 << 4);
    // The Bridge is itself a controller whose line is the pin.
    r.bridge.set_line(4, false);
    assert_eq!(r.get(reg::INT_STATUS) & 0xff, 0);
    // A line past the eight pins is dropped.
    r.bridge.set_line(8, true);
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
    let bridge = Bridge::new(Arc::new(Mem::new()), heart.clone());
    *heart.0.lock() = Some(bridge.clone());
    let r = Rig { bridge, sink: Arc::new(Sink::default()), mem: Arc::new(Mem::new()) };
    r.route(4, 3);
    r.bridge.set_wire_irq(4, true);
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
    let b1 = r.bridge.clone();
    let b2 = r.bridge.clone();
    let t1 = std::thread::spawn(move || (0..2000).for_each(|_| { b1.set_wire_irq(4, true); b1.set_wire_irq(4, false); }));
    let t2 = std::thread::spawn(move || (0..2000).for_each(|_| { b2.set_wire_irq(4, true); b2.set_wire_irq(4, false); }));
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
    r.bridge.attach(2, c.clone());
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
