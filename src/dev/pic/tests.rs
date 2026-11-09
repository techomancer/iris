// Cases follow tests/qtest/sgi-pic-test.c in jamesbraid/qemu-sgi-origami, for
// the Tezro runtime profile only: identity-fields, runtime-identity-bit,
// heads-and-widths, runtime-reset-compatibility-trial, widget-registers,
// cpu-word-view-registers, runtime-host-error-field,
// cpu-pio-overlapping-bars-use-active-function,
// runtime-dir-map-unsupported-fields, configuration, slot-bounds,
// runtime-profile-reserved-device-and-dma, type1-configuration,
// cpu-pio-conventional-lanes, cpu-pio-actual-ioc4,
// short-device-io-windows, short-device-io-attributes-boundaries,
// direct-dma, direct-dma-head-targets, direct64-card-write,
// direct64-requester-width, direct64-rejects-unsupported-addresses,
// direct64-range-boundary-is-split, direct64-target-error-preserves-data,
// pin-and-reset, remaining-pin-edges, bedrock-interrupt-level-bounds,
// runtime-invalid-address-error-trial and unwired-pci-interrupt. The qtest
// fixtures' packet face, translated DMA, migration streams and profile
// selection have no counterpart here, nor do its Bedrock reassertion cases:
// this model sends a packet only on a rising input, an enable, a reset
// write or a force write, and has no timer that sends one again.
use super::*;
use crate::dev::bedrock::{reg::{pi as pi_reg, PI1_BASE}, Bedrock, BedrockConfig};
use crate::irq::LevelProbe;
use crate::traits::Device;
use crate::pci::{Bar, PciConfig, PciFunctionState, PCI_INTA};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const HEAD1: u64 = HEAD_OFFSET;
const BAR_BASE: u32 = 0x0100_0000;
const MEMORY_AND_MASTER: u32 = 6;

/// A PCI function with one 4 KiB memory BAR that stores what it is given.
struct Card {
    state: PciFunctionState,
    bar: Mutex<[u8; 0x1000]>,
}

impl Card {
    fn new(device: u16) -> Arc<Self> {
        let bars = [Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused];
        Arc::new(Self {
            state: PciFunctionState::new(PciConfig::new(0x1234, device, 0x028000, 1, bars, PCI_INTA)),
            bar: Mutex::new([0; 0x1000]),
        })
    }

    fn dma(&self) -> &Arc<dyn PciDma> {
        self.state.dma().expect("attached to a bridge")
    }

    fn raise(&self, level: bool) {
        self.state.irq().expect("attached to a bridge").set(level);
    }
}

impl PciFunction for Card {
    fn state(&self) -> &PciFunctionState {
        &self.state
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        le_value(&self.bar.lock()[off as usize..off as usize + width]) as u32
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.bar.lock()[off as usize..off as usize + width].copy_from_slice(&val.to_le_bytes()[..width]);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dma {
    source: u8,
    widget: u8,
    address: u64,
    len: usize,
    write: bool,
}

/// The link's far end: sparse memory per widget, and a log of what arrived.
#[derive(Default)]
struct Peer {
    interrupts: Mutex<Vec<(u8, u64, u64)>>,
    dma: Mutex<Vec<Dma>>,
    memory: Mutex<HashMap<(u8, u64), u8>>,
    refuse: AtomicBool,
}

impl Peer {
    fn byte(&self, widget: u8, address: u64) -> u8 {
        self.memory.lock().get(&(widget, address)).copied().unwrap_or(0)
    }

    fn bytes(&self, widget: u8, address: u64, len: usize) -> Vec<u8> {
        (0..len as u64).map(|i| self.byte(widget, address + i)).collect()
    }

    fn put(&self, widget: u8, address: u64, data: &[u8]) {
        for (i, b) in data.iter().enumerate() {
            self.memory.lock().insert((widget, address + i as u64), *b);
        }
    }
}

impl PicPeer for Peer {
    fn interrupt(&self, source_widget: u8, address: u64, payload: u64) -> Result<(), Refused> {
        if self.refuse.load(Ordering::SeqCst) {
            return Err(Refused);
        }
        self.interrupts.lock().push((source_widget, address, payload));
        Ok(())
    }

    fn dma_read(&self, source: u8, widget: u8, address: u64, data: &mut [u8]) -> Result<(), Refused> {
        self.dma.lock().push(Dma { source, widget, address, len: data.len(), write: false });
        if self.refuse.load(Ordering::SeqCst) {
            return Err(Refused);
        }
        for (i, b) in data.iter_mut().enumerate() {
            *b = self.byte(widget, address + i as u64);
        }
        Ok(())
    }

    fn dma_write(&self, source: u8, widget: u8, address: u64, data: &[u8]) -> Result<(), Refused> {
        self.dma.lock().push(Dma { source, widget, address, len: data.len(), write: true });
        if self.refuse.load(Ordering::SeqCst) {
            return Err(Refused);
        }
        self.put(widget, address, data);
        Ok(())
    }
}

struct Rig {
    pic: Arc<Pic>,
    bus: Arc<dyn BusDevice>,
    peer: Arc<Peer>,
}

fn rig() -> Rig {
    let pic = Pic::new(15);
    let peer = Arc::new(Peer::default());
    pic.connect_peer(peer.clone()).unwrap();
    Rig { bus: pic.widget(), pic, peer }
}

impl Rig {
    fn r64(&self, addr: u64) -> u64 {
        let r = self.bus.read64(addr);
        assert!(r.is_ok(), "read64 {addr:#x}");
        r.data
    }

    fn r32(&self, addr: u64) -> u32 {
        let r = self.bus.read32(addr);
        assert!(r.is_ok(), "read32 {addr:#x}");
        r.data
    }

    fn w64(&self, addr: u64, v: u64) {
        assert_eq!(self.bus.write64(addr, v), BUS_OK, "write64 {addr:#x}");
    }

    fn w32(&self, addr: u64, v: u32) {
        assert_eq!(self.bus.write32(addr, v), BUS_OK, "write32 {addr:#x}");
    }

    /// Fit `card` at `device` of `head` and map its BAR at `BAR_BASE`.
    fn fit(&self, head: usize, device: u8, card: &Arc<Card>, options: Function) {
        self.pic.attach(head, device, card.clone(), options);
        let cfg = config(head, device as u64);
        self.w32(cfg + 0x10, BAR_BASE);
        self.w32(cfg + 4, MEMORY_AND_MASTER);
    }
}

fn head_base(head: usize) -> u64 {
    head as u64 * HEAD_OFFSET
}

fn config(head: usize, device: u64) -> u64 {
    head_base(head) + CONFIG_BASE + device * 0x1000
}

/// The processor address of byte `off` of a card's BAR through the large
/// PCI memory window.
fn window(head: usize, off: u64) -> u64 {
    PCI_WINDOW[head] + BAR_BASE as u64 + off
}

fn decode(word: u64, head: usize) {
    assert_eq!((word >> 28) & 0xf, 3, "revision");
    assert_eq!((word >> 12) & 0xffff, [0xd102, 0xd112][head], "part");
    assert_eq!((word & 0xffe) >> 1, 0x24, "manufacturer, packed one bit up");
    assert_eq!(word & 1, 1, "bit 0, which the PROM's identity check needs");
}

/// One of a Bedrock's PIs, through the chip's register window.
#[derive(Clone)]
struct Pi {
    chip: Arc<Bedrock>,
    regs: Arc<dyn BusDevice>,
    base: u64,
}
impl Pi {
    fn read64(&self, at: u64) -> BusRead64 {
        self.regs.read64(self.base + at)
    }
    fn write64(&self, at: u64, value: u64) -> u32 {
        self.regs.write64(self.base + at, value)
    }
    /// The chip's other PI.
    fn other(&self) -> Self {
        Self { base: PI1_BASE - self.base, ..self.clone() }
    }
}

/// A Bedrock's PI0, which the PIC raises its levels on, and the chip's clock.
fn pi_with_clock() -> (Pi, Arc<AtomicU64>) {
    let now = Arc::new(AtomicU64::new(0));
    let clock = now.clone();
    let chip = Bedrock::new(BedrockConfig {
        name: "test",
        clock: Arc::new(move || clock.load(Ordering::SeqCst)),
        processors: Default::default(),
        banks_mib: &[64],
        events: None,
    })
    .unwrap();
    chip.start();
    let regs = chip.registers();
    (Pi { chip, regs, base: 0 }, now)
}

/// The destination a pin's packet goes to: `level` in bits 63:48 and an
/// address in the PI's window.
fn destination(level: u64) -> u64 {
    (level << 48) | 0x1800090
}

/// The PIC's identity is the one thing the PROM checks before it trusts the
/// bridge, and each head has its own.
#[test]
fn each_head_has_its_own_identity_at_both_read_widths() {
    let r = rig();
    for head in 0..2 {
        decode(r.r64(head_base(head)), head);
        assert_eq!(r.r32(head_base(head) + 4) as u64, r.r64(head_base(head)));
        assert_eq!(identity(head) as u64, r.r64(head_base(head)));
    }
    assert_ne!(r.r64(0), r.r64(HEAD1));
    r.w64(0x20, 5);
    r.pic.power_on();
    decode(r.r64(0), 0);
    assert_eq!(r.r64(0x20), 0x0f00_320f, "reset restores control but not identity edits");
}

#[test]
fn reset_restores_the_selected_control_and_device_values() {
    let r = rig();
    for head in 0..2 {
        assert_eq!(r.r64(head_base(head) + 0x20), 0x0f00_320f);
        for (i, v) in [0x18001000, 0x18001002, 0x18001004, 0x18001005].into_iter().enumerate() {
            assert_eq!(r.r64(head_base(head) + 0x200 + i as u64 * 8), v);
        }
        assert_eq!(r.r64(head_base(head) + 0x28), 0xfffff);
    }
    // The low word of control merges into the register, per head.
    r.w32(0x24, 0x0f00_420f);
    assert_eq!(r.r32(0x24), 0x0f00_420f);
    assert_eq!(r.r32(HEAD1 + 0x24), 0x0f00_320f);
    r.w64(HEAD1 + 0x210, 0x1858_1007);
    assert_eq!(r.r64(HEAD1 + 0x210), 0x1858_1007);
    for head in 0..2 {
        for i in 0..4 {
            r.w64(head_base(head) + 0x200 + i * 8, 0xdead_0000 + head as u64 * 0x100 + i);
        }
    }
    r.pic.power_on();
    for head in 0..2 {
        assert_eq!(r.r64(head_base(head) + 0x20), 0x0f00_320f);
        assert_eq!(r.r64(head_base(head) + 0x200), 0x18001000);
        assert_eq!(r.r64(head_base(head) + 0x218), 0x18001005);
    }
}

/// The PROM writes the widget registers a word at a time through the low
/// half of each slot; a full doubleword write is the other form.
#[test]
fn widget_registers_take_full_and_low_word_writes_on_each_head() {
    let r = rig();
    for head in 0..2 {
        let base = head_base(head);
        for (n, seed) in [0xfffff_u64, 0, 0].into_iter().enumerate() {
            let slot = base + 0x28 + n as u64 * 8;
            let initial = (0x1234_5678_0000_0000 | seed) ^ ((head as u64) << 40);
            let low = if n == 1 && head == 1 { 0x5a5a_5a5a } else { 0xa5a5_a5a5 };
            assert_eq!(r.r64(slot), seed);
            r.w64(slot, initial);
            assert_eq!(r.r64(slot), initial);
            r.w32(slot + 4, low);
            assert_eq!(r.r32(slot + 4), low);
            assert_eq!(r.r64(slot), (initial & 0xffff_ffff_0000_0000) | low as u64);
            r.w64(slot, initial ^ 0x55aa_55aa);
            // Other widths are refused and change nothing.
            assert_eq!(r.bus.write8(slot, 0x7e), BUS_OK);
            assert_eq!(r.bus.write16(slot, 0x7e5a), BUS_OK);
            r.w32(slot, 0x7e5a_3c1d);
            assert_eq!(r.bus.read8(slot).data, 0);
            assert_eq!(r.bus.read16(slot).data, 0);
            assert_eq!(r.r32(slot), 0);
            assert_eq!(r.r64(slot), initial ^ 0x55aa_55aa);
        }
    }
    assert!(r.pic.unsupported().iter().any(|&(_, write, off, size)| write && off == 0x28 && size == 1));
    r.pic.power_on();
    for head in 0..2 {
        assert_eq!(r.r64(head_base(head) + 0x28), 0xfffff);
        assert_eq!(r.r64(head_base(head) + 0x30), 0);
        assert_eq!(r.r32(head_base(head) + 0x2c), 0xfffff);
    }
}

#[test]
fn the_host_error_field_has_a_low_word_view() {
    let r = rig();
    for head in 0..2u64 {
        let base = head * HEAD_OFFSET;
        assert_eq!(r.r64(base + 0x128), 0);
        r.w64(base + 0x128, 0x1234_5678_0000_0000 | (head + 1));
        assert_eq!(r.r32(base + 0x12c) as u64, head + 1);
        r.w32(base + 0x12c, 0x55 + head as u32);
        assert_eq!(r.r64(base + 0x128), 0x1234_5678_0000_0000 | (0x55 + head));
        assert_eq!(r.r64(base + 0x100), 0, "the write raised no status");
    }
    r.pic.power_on();
    assert_eq!(r.r64(0x128), 0);
}

#[test]
fn dir_map_keeps_only_the_implemented_bits() {
    let r = rig();
    for head in 0..2 {
        let at = head_base(head) + 0x80;
        r.w64(at, 0x1ffff | (0xf << 20));
        assert_eq!(r.r64(at), 0x1ffff | (0xf << 20));
        let before = r.pic.unsupported().len();
        r.w64(at, (1 << 17) | (1 << 40) | 0x1234);
        assert_eq!(r.r64(at), 0x1234, "unsupported bits clear and the rest stay");
        assert!(r.pic.unsupported().len() > before, "the refusal is observable");
    }
}

/// Device 0 is the bridge and answers nothing; the function at device `n`
/// is `n` windows up, and every miss reads all ones.
#[test]
fn configuration_cycles_reach_the_function_one_window_above_its_device() {
    let r = rig();
    let a = Card::new(0x5001);
    let b = Card::new(0x5002);
    r.pic.attach(0, 1, a.clone(), Function::default());
    r.pic.attach(0, 3, b.clone(), Function::default());
    assert_eq!(r.r32(config(0, 1)), 0x5001_1234);
    assert_eq!(r.r32(config(0, 3)), 0x5002_1234);
    assert!(!r.bus.read32(config(1, 1)).is_ok(), "head 1 has its own bus");
    for device in [0, 2, 4, 5, 7] {
        assert!(!r.bus.read32(config(0, device)).is_ok(), "device {device}");
        r.w32(config(0, device) + 4, 7);
    }
    // Bytes and halfwords of the big-endian processor land on the right
    // little-endian bytes of the register.
    assert_eq!(r.bus.read16(config(0, 1)).data, 0x5001, "offset 0 is the register's upper half");
    assert_eq!(r.bus.read16(config(0, 1) + 2).data, 0x1234);
    assert_eq!(r.bus.read8(config(0, 1)).data, 0x50);
    assert_eq!(r.bus.read8(config(0, 1) + 3).data, 0x34);
    // Only function 0 answers.
    assert!(!r.bus.read32(config(0, 1) + 0x100).is_ok());
    r.w32(config(0, 1) + 4, MEMORY_AND_MASTER);
    assert_eq!(r.r32(config(0, 1) + 4) & 0xffff, MEMORY_AND_MASTER);
    r.w32(config(0, 1) + 0x10, BAR_BASE);
    assert_eq!(r.r32(config(0, 1) + 0x10), BAR_BASE);
    assert_eq!(a.state.config().bar_base(0), BAR_BASE);
}

#[test]
fn type_one_configuration_selects_the_bus_and_device_through_the_register() {
    let r = rig();
    let a = Card::new(0x5001);
    r.pic.attach(0, 2, a, Function::default());
    r.w64(0xc8, 2 << 11);
    assert_eq!(r.r32(CONFIG_TYPE1), 0x5001_1234);
    r.w64(0xc8, (1 << 16) | (2 << 11));
    assert!(!r.bus.read32(CONFIG_TYPE1).is_ok(), "a subordinate bus has nothing");
    r.w64(0xc8, 3 << 11);
    assert!(!r.bus.read32(CONFIG_TYPE1).is_ok());
    r.w64(0xc8, 2 << 11);
    assert_eq!(r.r32(CONFIG_TYPE1 + 8), 0x0280_0001, "class and revision");
}

/// The processor's byte lanes: an adapter wired little-endian sees the
/// numeric value of a word, and a halfword at offset 2 reaches offset 0.
#[test]
fn a_conventional_function_sees_cpu_pio_with_its_address_corrected() {
    let r = rig();
    let card = Card::new(0x6001);
    r.fit(0, 1, &card, Function::default());
    r.w32(window(0, 0x10), 0x1234_5678);
    assert_eq!(&card.bar.lock()[0x10..0x14], [0x78, 0x56, 0x34, 0x12]);
    assert_eq!(r.r32(window(0, 0x10)), 0x1234_5678);
    assert_eq!(r.bus.write16(window(0, 0x22), 0x9abc), BUS_OK);
    assert_eq!(&card.bar.lock()[0x20..0x22], [0xbc, 0x9a], "offset 2 reaches offset 0");
    assert_eq!(r.bus.read16(window(0, 0x22)).data, 0x9abc);
    assert_eq!(r.bus.write8(window(0, 0x33), 0x5a), BUS_OK);
    assert_eq!(card.bar.lock()[0x30], 0x5a, "byte offset 3 reaches offset 0");
    assert_eq!(r.bus.read8(window(0, 0x33)).data, 0x5a);
    assert_eq!(r.bus.read8(window(0, 0x30)).data, 0, "offset 0 is a different byte");
    // The same card through the short window of head 0, memory space.
    r.w64(0x200, (1 << 12) | (BAR_BASE >> 20) as u64);
    r.w32(SHORT_BASE + 0x40, 0xcafe_f00d);
    assert_eq!(&card.bar.lock()[0x40..0x44], [0x0d, 0xf0, 0xfe, 0xca]);
    assert_eq!(r.r32(SHORT_BASE + 0x40), 0xcafe_f00d);
}

/// IOC4 keeps the processor's byte sequence and the address as written.
#[test]
fn a_self_ordered_function_keeps_the_processors_byte_sequence() {
    let r = rig();
    let card = Card::new(0x6002);
    r.fit(1, 1, &card, Function { self_ordered_pio: true, req64: false });
    r.w32(window(1, 0x10), 0x1234_5678);
    assert_eq!(&card.bar.lock()[0x10..0x14], [0x12, 0x34, 0x56, 0x78]);
    assert_eq!(r.r32(window(1, 0x10)), 0x1234_5678);
    assert_eq!(r.bus.write8(window(1, 0x21), 0xa5), BUS_OK);
    assert_eq!(card.bar.lock()[0x21], 0xa5, "no address correction");
    assert_eq!(r.bus.read8(window(1, 0x21)).data, 0xa5);
    assert_eq!(r.bus.write16(window(1, 0x22), 0x9abc), BUS_OK);
    assert_eq!(&card.bar.lock()[0x22..0x24], [0x9a, 0xbc]);
}

/// A short window follows its Device register: space, megabyte and size
/// limit, and it refuses the attributes it does not model.
#[test]
fn short_windows_follow_their_device_registers() {
    let r = rig();
    let card = Card::new(0x6003);
    r.fit(0, 1, &card, Function::default());
    // Window 0 and 1 are 2 MiB and ignore the low bit of the base; windows
    // 2 and 3 are 1 MiB.
    for (window_index, cpu) in [(0u64, 0x20_0000u64), (1, 0x40_0000), (2, 0x60_0000), (3, 0x70_0000)] {
        // The two big windows ignore the base's low bit; the small ones keep it.
        let base = (BAR_BASE >> 20) as u64 | if window_index < 2 { 1 } else { 0 };
        r.w64(0x200 + window_index * 8, (1 << 12) | base);
        card.bar.lock().fill(0);
        r.w32(cpu + 0x80, 0x1111_1111 * (window_index as u32 + 1));
        assert_eq!(le_value(&card.bar.lock()[0x80..0x84]), 0x1111_1111 * (window_index + 1), "window {window_index}");
    }
    // The last word of a window fits; a doubleword across its end does not.
    r.w32(0x70_0000 + 0xf_fffc, 7);
    let before = r.pic.unsupported().len();
    assert_eq!(r.bus.write64(0x70_0000 + 0xf_fffc, 7), BUS_OK);
    assert_eq!(r.r64(0x70_0000 + 0xf_fffc), 0, "a boundary-crossing access moves nothing");
    assert_eq!(r.pic.unsupported().len(), before, "it is logged, not listed as unsupported");
    // Swapped and wide programmed I/O are refused rather than guessed.
    r.w64(0x200, (1 << 12) | (BAR_BASE >> 20) as u64 | (1 << 13));
    card.bar.lock().fill(0);
    r.w32(SHORT_BASE + 0x40, 0xffff_ffff);
    assert_eq!(card.bar.lock()[0x40], 0);
    r.w64(0x200, (1 << 12) | (BAR_BASE >> 20) as u64 | (1 << 22));
    assert_eq!(r.r32(SHORT_BASE + 0x40), 0);
    assert_eq!(card.bar.lock()[0x40], 0);
}

#[test]
fn the_guard_after_the_second_head_reads_zero_and_the_gaps_fault() {
    let r = rig();
    assert_eq!(r.r64(2 * HEAD_OFFSET), 0);
    assert_eq!(r.bus.write64(2 * HEAD_OFFSET, 1), BUS_OK);
    assert!(!r.bus.read64(2 * HEAD_OFFSET + 8).is_ok());
    assert!(!r.bus.read64(0x8000_0000).is_ok(), "the 64-bit alias window is not this chip's");
    assert!(!r.bus.read32(0x1_0000_0000).is_ok());
    assert!(r.bus.read32(0x20_0001).is_ok(), "short windows take unaligned accesses");
    assert!(!r.bus.read32(0x1002).is_ok(), "registers need natural alignment");
}

/// Both heads send direct-map DMA to the widget and address their own
/// `DIR_MAP` names.
#[test]
fn direct_map_dma_goes_where_each_heads_direct_map_says() {
    let r = rig();
    let a = Card::new(0x7001);
    let b = Card::new(0x7002);
    r.fit(0, 1, &a, Function::default());
    r.fit(1, 1, &b, Function::default());
    r.w64(0x80, 10 << 20);
    r.w64(HEAD1 + 0x80, (11 << 20) | 1);
    let data: Vec<u8> = (0x11..0x19).collect();
    assert!(a.dma().dma_write(0x8000_1000, &data).is_ok());
    let other: Vec<u8> = (0x21..0x29).collect();
    assert!(b.dma().dma_write(0x8000_2000, &other).is_ok());
    assert_eq!(r.peer.bytes(10, 0x1000, 8), data);
    assert_eq!(r.peer.bytes(11, (1 << 31) | 0x2000, 8), other);
    assert_eq!(r.peer.bytes(10, 0x2000, 8), vec![0; 8], "the other head's target is untouched");
    let log = r.peer.dma.lock().clone();
    assert_eq!(log[0], Dma { source: 15, widget: 10, address: 0x1000, len: 8, write: true });
    // Reads come back through the same path.
    let mut back = [0u8; 8];
    assert!(a.dma().dma_read(0x8000_1000, &mut back).is_ok());
    assert_eq!(back.to_vec(), data);
    // A transfer at an odd address is cut into naturally aligned pieces, and
    // one that crosses a page is translated again at the page.
    r.peer.dma.lock().clear();
    assert!(a.dma().dma_write(0x8000_1ffd, &[1, 2, 3, 4, 5, 6]).is_ok());
    let sizes: Vec<_> = r.peer.dma.lock().iter().map(|p| (p.address, p.len)).collect();
    assert_eq!(sizes, [(0x1ffd, 1), (0x1ffe, 2), (0x2000, 2), (0x2002, 1)].map(|(a, l)| (a as u64, l)));
}

/// A slot whose Device register sets the swap bit puts each byte at its
/// address with the low two bits flipped, and still moves whole words as
/// one packet.
#[test]
fn a_swapping_slot_flips_the_byte_address_within_each_word() {
    let r = rig();
    let card = Card::new(0x7003);
    r.fit(0, 1, &card, Function::default());
    r.w64(0x80, 10 << 20);
    r.w64(0x200, 0x18001000 | (1 << 19));
    assert!(card.dma().dma_write(0x8000_0100, &[1, 2, 3, 4, 5, 6, 7, 8]).is_ok());
    assert_eq!(r.peer.bytes(10, 0x100, 8), [4, 3, 2, 1, 8, 7, 6, 5]);
    let mut back = [0u8; 8];
    assert!(card.dma().dma_read(0x8000_0100, &mut back).is_ok());
    assert_eq!(back, [1, 2, 3, 4, 5, 6, 7, 8]);
    let packets: Vec<_> = r.peer.dma.lock().iter().map(|p| (p.address, p.len)).collect();
    assert_eq!(packets, [(0x100, 8), (0x100, 8)]);
}

#[test]
fn dma_outside_the_direct_window_is_refused_and_a_read_comes_back_zero() {
    let r = rig();
    let card = Card::new(0x7004);
    r.fit(0, 1, &card, Function::default());
    r.peer.put(0, 0, &[0xaa; 16]);
    assert_eq!(card.dma().dma_write(0x7fff_f000, &[1; 8]), PciDmaStatus::MasterAbort, "below the window");
    let mut buf = [0xffu8; 8];
    assert_eq!(card.dma().dma_read(0x4000_0000, &mut buf), PciDmaStatus::MasterAbort, "translated DMA is unsupported");
    assert_eq!(buf, [0; 8]);
    assert!(r.peer.dma.lock().is_empty(), "nothing reached the peer");
    // A peer that refuses ends the transfer and zeroes the unread rest.
    r.peer.refuse.store(true, Ordering::SeqCst);
    let mut buf = [0xffu8; 8];
    assert_eq!(card.dma().dma_read(0x8000_0000, &mut buf), PciDmaStatus::MasterAbort);
    assert_eq!(buf, [0; 8]);
    // Device 0 and devices past 4 have no DMA path.
    r.peer.refuse.store(false, Ordering::SeqCst);
    assert_eq!(r.pic.heads[0].dma(0, 0x8000_0000, &mut [0; 4], false), PciDmaStatus::MasterAbort);
    assert_eq!(r.pic.heads[0].dma(5, 0x8000_0000, &mut [0; 4], false), PciDmaStatus::MasterAbort);
}

fn direct64(widget: u64, flags: u64, payload: u64) -> u64 {
    (widget << 60) | flags | payload
}

/// Bit 55 set is an ordered byte stream; clear, each 32-bit word's bytes
/// swap, or each doubleword's for a 64-bit requester.
#[test]
fn direct64_dma_keeps_a_stream_and_swaps_the_words_of_a_swap_clear_request() {
    let r = rig();
    let narrow = Card::new(0x7101);
    let wide = Card::new(0x7102);
    r.fit(0, 1, &narrow, Function::default());
    r.fit(0, 4, &wide, Function { self_ordered_pio: false, req64: true });
    let data = [0x12u8, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0];
    assert!(narrow.dma().dma_write(direct64(0xa, 1 << 55, 0x3000), &data).is_ok());
    assert_eq!(r.peer.bytes(0xa, 0x3000, 8), data, "ordered bytes unchanged");
    assert!(narrow.dma().dma_write(direct64(0xa, 0, 0x4000), &data).is_ok());
    assert_eq!(r.peer.bytes(0xa, 0x4000, 8), [0x78, 0x56, 0x34, 0x12, 0xf0, 0xde, 0xbc, 0x9a]);
    assert!(wide.dma().dma_write(direct64(0xa, 0, 0x5000), &data).is_ok());
    assert_eq!(r.peer.bytes(0xa, 0x5000, 8), [0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12]);
    assert!(wide.dma().dma_write(direct64(0xa, 1 << 55, 0x6000), &data).is_ok());
    assert_eq!(r.peer.bytes(0xa, 0x6000, 8), data, "a stream is a stream at either width");
    let mut back = [0u8; 8];
    assert!(narrow.dma().dma_read(direct64(0xa, 0, 0x4000), &mut back).is_ok());
    assert_eq!(back, data);
    // Prefetch, precise, virtual channel 1 and barrier are recognized and
    // ignored; DIR_MAP does not apply.
    r.w64(0x80, 3 << 20);
    for bits in [1 << 59, 1 << 58, 1 << 57, 1 << 56] {
        assert!(narrow.dma().dma_write(direct64(0xa, bits | (1 << 55), 0x7000), &data).is_ok());
    }
    assert!(r.peer.dma.lock().iter().all(|p| p.widget == 0xa));
}

/// The swap moves bytes within a word, not the transfer apart: whole words
/// leave as one packet, so a processor reading a word the card is updating
/// (a ring index in a status block) never sees half of it new. A ragged end
/// writes only its own bytes of the word, as one packet, so an aligned
/// halfword in it is not torn either.
#[test]
fn a_swap_clear_direct64_transfer_keeps_its_words_whole() {
    let r = rig();
    let narrow = Card::new(0x7101);
    let wide = Card::new(0x7102);
    r.fit(0, 1, &narrow, Function::default());
    r.fit(0, 4, &wide, Function { self_ordered_pio: false, req64: true });
    let data: Vec<u8> = (0x10..0x30).collect();
    for (card, mask, at) in [(&narrow, 3u64, 0x4000u64), (&wide, 7, 0x5000)] {
        r.peer.dma.lock().clear();
        assert!(card.dma().dma_write(direct64(0xa, 0, at), &data).is_ok());
        let packets: Vec<(u64, usize)> = r.peer.dma.lock().iter().map(|p| (p.address, p.len)).collect();
        assert_eq!(packets, [(at, data.len())], "mask {mask}: one packet");
        let want: Vec<u8> = (0..data.len() as u64).map(|i| data[(i ^ mask) as usize]).collect();
        assert_eq!(r.peer.bytes(0xa, at, data.len()), want, "mask {mask}: each word reversed");
        let mut back = vec![0u8; data.len()];
        assert!(card.dma().dma_read(direct64(0xa, 0, at), &mut back).is_ok());
        assert_eq!(back, data, "mask {mask}: read back");
    }
    // Two bytes before a word and one after it: the word still goes whole.
    r.peer.dma.lock().clear();
    assert!(narrow.dma().dma_write(direct64(0xa, 0, 0x6002), &[0xa1, 0xa2, 1, 2, 3, 4, 0xb1]).is_ok());
    let packets: Vec<(u64, usize)> = r.peer.dma.lock().iter().map(|p| (p.address, p.len)).collect();
    assert_eq!(packets, [(0x6000, 2), (0x6004, 4), (0x600b, 1)]);
    assert_eq!(r.peer.bytes(0xa, 0x6000, 4), [0xa2, 0xa1, 0, 0], "the word's other bytes are left alone");
    assert_eq!(r.peer.bytes(0xa, 0x6004, 4), [4, 3, 2, 1]);
    assert_eq!(r.peer.byte(0xa, 0x600b), 0xb1);
    // Three bytes from a doubleword's second: one packet, read back too.
    r.peer.dma.lock().clear();
    assert!(wide.dma().dma_write(direct64(0xa, 0, 0x7001), &[1, 2, 3]).is_ok());
    let packets: Vec<(u64, usize)> = r.peer.dma.lock().iter().map(|p| (p.address, p.len)).collect();
    assert_eq!(packets, [(0x7004, 3)]);
    assert_eq!(r.peer.bytes(0xa, 0x7004, 3), [3, 2, 1]);
    let mut back = [0u8; 3];
    assert!(wide.dma().dma_read(direct64(0xa, 0, 0x7001), &mut back).is_ok());
    assert_eq!(back, [1, 2, 3]);
}

/// However a transfer starts and ends, no naturally aligned field of a data
/// phase that it writes whole is split between two packets: a ragged start
/// or end is a packet of its own bytes, never cut inside an aligned
/// halfword, word or (where the phase is 64 bits, or the bytes are a
/// stream) doubleword. 11 bytes from 0x..23 cut at 0x..2b would split the
/// halfword at 0x..2a.
#[test]
fn no_aligned_field_is_split_between_packets() {
    let r = rig();
    let narrow = Card::new(0x7101);
    let wide = Card::new(0x7102);
    r.fit(0, 1, &narrow, Function::default());
    r.fit(0, 4, &wide, Function { self_ordered_pio: false, req64: true });
    for (card, flags, phase) in [(&narrow, 1u64 << 55, 8u64), (&narrow, 0, 4), (&wide, 0, 8), (&wide, 1 << 55, 8)] {
        for start in 0x20u64..0x28 {
            for len in 1..24usize {
                r.peer.dma.lock().clear();
                let data: Vec<u8> = (0..len as u8).collect();
                assert!(card.dma().dma_write(direct64(0xa, flags, start), &data).is_ok());
                let packets: Vec<(u64, u64)> =
                    r.peer.dma.lock().iter().map(|p| (p.address, p.address + p.len as u64)).collect();
                let covered = |a: u64| packets.iter().any(|&(lo, hi)| (lo..hi).contains(&a));
                for size in [2u64, 4, 8].into_iter().filter(|&n| n <= phase) {
                    for field in (0x18u64..0x48).step_by(size as usize) {
                        if (field..field + size).all(covered) {
                            assert!(
                                packets.iter().any(|&(lo, hi)| lo <= field && field + size <= hi),
                                "flags {flags:#x} start {start:#x} len {len}: {size}-byte field at {field:#x} split in {packets:x?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn direct64_refuses_what_it_cannot_carry_before_any_packet_leaves() {
    let r = rig();
    let card = Card::new(0x7103);
    r.fit(1, 2, &card, Function::default());
    let sentinel = [0xa5u8; 8];
    r.peer.put(0xa, 0x1000, &sentinel);
    for bit in 48..=54 {
        assert_eq!(card.dma().dma_write(direct64(0xa, 1 << bit, 0x1000), &[1; 4]), PciDmaStatus::MasterAbort, "bit {bit}");
    }
    assert_eq!(card.dma().dma_write(direct64(0xa, 0, (1 << 47) | 0x1000), &[1; 4]), PciDmaStatus::MasterAbort, "register space");
    assert_eq!(card.dma().dma_write(1 << 32, &[1; 4]), PciDmaStatus::MasterAbort, "target zero");
    assert!(r.peer.dma.lock().is_empty());
    assert_eq!(r.peer.bytes(0xa, 0x1000, 8), sentinel);
}

/// A transfer that runs into the register-space boundary delivers the part
/// below it and fails at the page that crosses.
#[test]
fn direct64_splits_at_the_register_space_boundary() {
    let r = rig();
    let card = Card::new(0x7104);
    r.fit(0, 1, &card, Function::default());
    let boundary = 1u64 << 47;
    let status = card.dma().dma_write(direct64(0xa, 1 << 55, boundary - 2), &[0x90, 0x91, 0x92, 0x93]);
    assert_eq!(status, PciDmaStatus::MasterAbort);
    assert_eq!(r.peer.bytes(0xa, boundary - 2, 2), [0x90, 0x91]);
    assert_eq!(r.peer.dma.lock().len(), 1);
}

#[test]
fn a_refused_direct64_write_leaves_the_target_as_it_was() {
    let r = rig();
    let card = Card::new(0x7105);
    r.fit(0, 1, &card, Function::default());
    r.peer.put(0xa, 0x3000, &[0xa5; 8]);
    r.peer.refuse.store(true, Ordering::SeqCst);
    for flags in [0, 1 << 55, 1 << 58, (1 << 56) | (1 << 55)] {
        assert_eq!(card.dma().dma_write(direct64(0xa, flags, 0x3000), &[1; 8]), PciDmaStatus::MasterAbort);
    }
    assert_eq!(r.peer.bytes(0xa, 0x3000, 8), [0xa5; 8]);
    assert!(r.peer.dma.lock().len() >= 4, "each attempt reached the peer");
}

/// An input that rises sets its status and, once enabled, sends the set
/// packet to its destination; a falling edge sends a clear if its mode bit
/// asks for one.
#[test]
fn a_rising_enabled_pin_sends_a_set_packet_and_a_falling_one_a_clear() {
    let r = rig();
    for pin in 0..PINS as u64 {
        r.w64(0x130 + pin * 8, destination(0x20 + pin));
    }
    r.pic.set_pin(0, 3, true);
    assert_eq!(r.r64(0x100), 1 << 3);
    assert!(r.peer.interrupts.lock().is_empty(), "disabled");
    r.w64(0x108, 1 << 3);
    assert_eq!(r.peer.interrupts.lock().as_slice(), [(15, 0x1800090, 0x123)], "enabling a pending input sends it");
    assert_eq!(r.pic.request_counts(0, 3), (1, 0));
    r.pic.set_pin(0, 3, false);
    assert_eq!(r.pic.request_counts(0, 3), (1, 0), "no clear without the mode bit");
    r.w64(0x118, 1 << 3);
    r.pic.set_pin(0, 3, true);
    r.pic.set_pin(0, 3, false);
    assert_eq!(r.pic.request_counts(0, 3), (2, 1));
    assert_eq!(r.peer.interrupts.lock().last().copied(), Some((15, 0x1800090, 0x23)));
    // The status bit stays until software clears it, and a pin that is
    // still high comes back.
    assert_eq!(r.r64(0x100), 1 << 3);
    r.w64(0x110, 1 << 3);
    assert_eq!(r.r64(0x100), 0);
    r.pic.set_pin(1, 0, true);
    assert_eq!(r.r64(HEAD1 + 0x100), 1, "the heads' pins are separate");
    assert_eq!(r.r64(0x100), 0);
    // Reset drops pins and status.
    r.pic.power_on();
    assert_eq!(r.r64(0x100), 0);
    r.pic.set_pin(0, 3, true);
    assert_eq!(r.pic.request_counts(0, 3), (0, 0), "counts restart and the enable is gone");
}

#[test]
fn a_clear_write_resends_a_pin_that_is_still_high_and_enabled() {
    let r = rig();
    r.w64(0x130, destination(9));
    r.w64(0x108, 1);
    r.pic.set_pin(0, 0, true);
    assert_eq!(r.pic.request_counts(0, 0), (1, 0));
    r.w64(0x110, 1);
    assert_eq!(r.pic.request_counts(0, 0), (2, 0), "the status bit comes back with the level, and so does the packet");
    assert_eq!(r.r64(0x100), 1);
    // A masked pin stays quiet on a clear write.
    r.w64(0x108, 0);
    r.w64(0x110, 1);
    assert_eq!(r.pic.request_counts(0, 0), (2, 0));
}

#[test]
fn only_levels_a_pi_can_take_leave_the_chip() {
    let r = rig();
    r.pic.set_pin(0, 0, true);
    for level in [0, 6, 128, 0x7fff] {
        r.w64(0x130, destination(level));
        r.w64(0x108, 0);
        r.w64(0x108, 1);
        assert!(r.peer.interrupts.lock().is_empty(), "level {level}");
    }
    assert_eq!(r.pic.request_counts(0, 0).0, 4, "each attempt was counted");
    for level in [7, 127] {
        r.w64(0x130, destination(level));
        r.w64(0x108, 0);
        r.w64(0x108, 1);
    }
    let sent: Vec<_> = r.peer.interrupts.lock().iter().map(|p| p.2).collect();
    assert_eq!(sent, [0x107, 0x17f]);
    // A peer that refuses is not an error the guest sees, but it is counted.
    r.peer.refuse.store(true, Ordering::SeqCst);
    r.w64(0x108, 0);
    r.w64(0x108, 1);
    assert_eq!(r.peer.interrupts.lock().len(), 2);
}

/// A PCI function's INTA reaches the input numbered one less than its
/// device, whichever functions are fitted, and nothing else raises it.
#[test]
fn pci_interrupts_reach_the_input_of_their_device() {
    let r = rig();
    let card = Card::new(0x7201);
    let other = Card::new(0x7202);
    r.pic.attach(0, 3, card.clone(), Function::default());
    r.pic.attach(0, 4, other.clone(), Function::default());
    r.w64(0x130 + 2 * 8, destination(0x31));
    r.w64(0x108, 1 << 2);
    card.raise(true);
    assert_eq!(r.r64(0x100), 1 << 2, "device 3 is input 2");
    assert_eq!(r.peer.interrupts.lock().as_slice(), [(15, 0x1800090, 0x131)]);
    card.raise(false);
    other.raise(true);
    assert_eq!(r.r64(0x100), (1 << 2) | (1 << 3), "device 4 is input 3, which the enable leaves unsent");
    assert_eq!(r.peer.interrupts.lock().len(), 1);
    // A board routes its own interrupt to an input through the bus.
    r.pic.pin_line(0, 6).set(true);
    assert_eq!(r.r64(0x100) & (1 << 6), 1 << 6);
}

/// Force-always (0x180 + 8n) sends input n's set packet on any write,
/// whether the input is active or not and whether it was sent before. The
/// status bit follows the input, not the force.
#[test]
fn force_always_sends_the_set_packet_whatever_the_input() {
    let r = rig();
    r.w64(0x130 + 3 * 8, destination(0x20));
    r.w64(0x180 + 3 * 8, 0);
    assert_eq!(r.peer.interrupts.lock().as_slice(), [(15, 0x1800090, 0x120)], "an inactive input");
    assert_eq!(r.r64(0x100), 0, "the force leaves status alone");
    r.w64(0x108, 1 << 3);
    r.pic.set_pin(0, 3, true);
    assert_eq!(r.peer.interrupts.lock().len(), 2, "the rising input");
    r.w64(0x180 + 3 * 8, 0);
    assert_eq!(r.peer.interrupts.lock().len(), 3, "an input already active and sent");
    assert_eq!(r.pic.request_counts(0, 3), (3, 0));
    assert_eq!(r.pic.request_counts(1, 3), (0, 0), "the other head is separate");
}

/// Force-with-pin (0x1c0 + 8n) sends input n's set packet only while the
/// input is active: the write software makes at the end of each interrupt
/// dispatch, which brings back an input its handler left active.
#[test]
fn force_with_pin_sends_only_while_the_input_is_active() {
    let r = rig();
    r.w64(0x130 + 3 * 8, destination(0x20));
    r.w64(0x108, 1 << 3);
    r.w64(0x1c0 + 3 * 8, 0);
    assert!(r.peer.interrupts.lock().is_empty(), "an inactive input sends nothing");
    r.pic.set_pin(0, 3, true);
    r.w64(0x1c0 + 3 * 8, 0);
    assert_eq!(r.peer.interrupts.lock().as_slice(), [(15, 0x1800090, 0x120); 2], "rising, then forced");
    r.pic.set_pin(0, 3, false);
    r.w64(0x1c0 + 3 * 8, 0);
    assert_eq!(r.peer.interrupts.lock().len(), 2, "the input dropped");
    r.w64(0x1c0 + 2 * 8, 0);
    assert_eq!(r.peer.interrupts.lock().len(), 2, "each input has its own register");
}

/// The force registers take a doubleword write or a word to the slot's
/// low half, as the other registers do, and ignore the data. Reads and
/// other widths are unsupported and recorded, and send nothing.
#[test]
fn the_force_registers_are_write_only_and_ignore_the_data() {
    let r = rig();
    r.w64(0x130 + 3 * 8, destination(0x20));
    for value in [0, 1, u64::MAX] {
        r.w64(0x180 + 3 * 8, value);
    }
    r.w32(0x180 + 3 * 8 + 4, 0);
    assert_eq!(r.peer.interrupts.lock().len(), 4);
    r.w32(0x180 + 3 * 8, 1);
    assert_eq!(r.bus.write16(0x180 + 3 * 8, 1), BUS_OK);
    assert_eq!(r.bus.write8(0x180 + 3 * 8, 1), BUS_OK);
    assert_eq!(r.r64(0x180 + 3 * 8), 0);
    assert_eq!(r.r32(0x1c0 + 3 * 8 + 4), 0);
    assert_eq!(r.peer.interrupts.lock().len(), 4, "nothing else sent");
    let seen = r.pic.unsupported();
    for (write, offset, size) in [(true, 0x198, 4), (true, 0x198, 2), (true, 0x198, 1), (false, 0x198, 8), (false, 0x1dc, 4)] {
        assert!(seen.contains(&(0, write, offset, size)), "{offset:#x} {size} recorded");
    }
}

/// A refused set packet is dropped and logged and nothing sends it again
/// by itself: the status bit stays set, and software's force write
/// delivers it once the peer takes packets.
#[test]
fn a_refused_packet_is_dropped_and_a_force_write_sends_it_again() {
    let r = rig();
    r.w64(0x130 + 3 * 8, destination(0x20));
    r.w64(0x108, 1 << 3);
    r.peer.refuse.store(true, Ordering::SeqCst);
    r.pic.set_pin(0, 3, true);
    assert!(r.peer.interrupts.lock().is_empty());
    assert_eq!(r.pic.request_counts(0, 3), (1, 0), "one attempt");
    assert_eq!(r.r64(0x100), 1 << 3, "the status bit stays set");
    r.peer.refuse.store(false, Ordering::SeqCst);
    assert!(r.peer.interrupts.lock().is_empty(), "no retry of its own");
    r.w64(0x1c0 + 3 * 8, 0);
    assert_eq!(r.peer.interrupts.lock().as_slice(), [(15, 0x1800090, 0x120)]);
}

/// Both PIs, told apart the way the node does: by address bit 21.
struct ToPi([Pi; 2]);
impl PicPeer for ToPi {
    fn interrupt(&self, _: u8, address: u64, payload: u64) -> Result<(), Refused> {
        self.0[(address >> 21) as usize & 1].write64(pi_reg::PENDING_MODIFY, payload);
        Ok(())
    }
    fn dma_read(&self, _: u8, _: u8, _: u64, _: &mut [u8]) -> Result<(), Refused> {
        Err(Refused)
    }
    fn dma_write(&self, _: u8, _: u8, _: u64, _: &[u8]) -> Result<(), Refused> {
        Err(Refused)
    }
}

/// The end of an interrupt dispatch: the processor cleared the PI level,
/// its handler left the input active, and its force-with-pin write brings
/// the level back at once.
#[test]
fn a_force_write_after_dispatch_brings_back_an_active_input() {
    let (pi, _now) = pi_with_clock();
    let pic = Pic::new(15);
    pic.connect_peer(Arc::new(ToPi([pi.clone(), pi.other()]))).unwrap();
    let bus = pic.widget();
    bus.write64(0x130 + 3 * 8, destination(12));
    bus.write64(0x108, 1 << 3);
    pic.set_pin(0, 3, true);
    assert_eq!(pi.read64(pi_reg::PENDING_STATUS_A).data, 1 << 12);
    pi.write64(pi_reg::PENDING_MODIFY, 12);
    assert_eq!(pi.read64(pi_reg::PENDING_STATUS_A).data, 0);
    bus.write64(0x1c0 + 3 * 8, 0);
    assert_eq!(pi.read64(pi_reg::PENDING_STATUS_A).data, 1 << 12, "the force write re-delivers");
    // With the input dropped, the same write leaves the level clear.
    pi.write64(pi_reg::PENDING_MODIFY, 12);
    pic.set_pin(0, 3, false);
    bus.write64(0x1c0 + 3 * 8, 0);
    assert_eq!(pi.read64(pi_reg::PENDING_STATUS_A).data, 0);
}

/// A processor on another thread that takes and clears the level the moment
/// the set packet lands, before the sending thread has gone on.
struct TakenAtOnce(Pi);
impl PicPeer for TakenAtOnce {
    fn interrupt(&self, _: u8, _: u64, payload: u64) -> Result<(), Refused> {
        self.0.write64(pi_reg::PENDING_MODIFY, payload);
        self.0.write64(pi_reg::PENDING_MODIFY, payload & !SET);
        Ok(())
    }
    fn dma_read(&self, _: u8, _: u8, _: u64, _: &mut [u8]) -> Result<(), Refused> {
        Err(Refused)
    }
    fn dma_write(&self, _: u8, _: u8, _: u64, _: &[u8]) -> Result<(), Refused> {
        Err(Refused)
    }
}

/// The chip sends a set packet when an input rises and on a force write,
/// and at no other time: a level the processor clears while the input
/// stays high, however soon after the packet lands, stays clear until the
/// dispatch's force write brings it back. Both ways a packet leaves on its
/// own are covered: a rising input and an enable.
#[test]
fn a_level_cleared_while_its_input_stays_high_waits_for_the_force_write() {
    for enable_last in [false, true] {
        let what = if enable_last { "enable" } else { "rising input" };
        let (pi, now) = pi_with_clock();
        let pic = Pic::new(15);
        pic.connect_peer(Arc::new(TakenAtOnce(pi.clone()))).unwrap();
        let bus = pic.widget();
        bus.write64(0x130, destination(12));
        if enable_last {
            pic.set_pin(0, 0, true);
            bus.write64(0x108, 1);
        } else {
            bus.write64(0x108, 1);
            pic.set_pin(0, 0, true);
        }
        assert_eq!(pic.request_counts(0, 0), (1, 0), "{what}: one packet");
        assert_eq!(pi.read64(pi_reg::PENDING_STATUS_A).data, 0, "{what}: the processor took it");
        now.store(1_000_000_000, Ordering::SeqCst);
        assert_eq!(pi.read64(pi_reg::PENDING_STATUS_A).data, 0, "{what}: nothing comes back by itself");
        assert_eq!(pic.request_counts(0, 0), (1, 0), "{what}: no packet of the chip's own");
        bus.write64(0x1c0, 0);
        assert_eq!(pic.request_counts(0, 0), (2, 0), "{what}: the force write sends it");
    }
}

/// The PROM writes the link-side error address slot's low word; the chip
/// latches the invalid-address status and, if enabled, sends one packet to
/// the destination the widget registers name.
#[test]
fn an_invalid_address_write_latches_status_and_sends_one_packet() {
    let r = rig();
    let invalid = 1u64 << 24;
    // Other widths and the doubleword form of the slot do nothing.
    r.bus.write8(0x7c, 0x11);
    r.bus.write16(0x7c, 0x2233);
    r.bus.write32(0x78, 0x4455_6677);
    r.w64(0x78, 0x8899_aabb_ccdd_eeff);
    assert_eq!(r.r64(0x100), 0);
    r.w32(0x7c, 0xdead_beef);
    assert_eq!(r.r32(0x104) as u64, invalid);
    assert_eq!(r.r32(HEAD1 + 0x104), 0, "per head");
    assert!(r.peer.interrupts.lock().is_empty(), "not enabled");
    r.w32(0x114, invalid as u32);
    assert_eq!(r.r32(0x104), 0, "written one to clear");
    let target = 0x1_0000_3000u64;
    r.w64(0x30, 0x3400_0000);
    r.w64(0x38, target);
    r.w32(0x10c, invalid as u32);
    for bad in [0x80u64, 0x100] {
        r.w64(0x128, bad);
        r.w32(0x7c, 0);
        assert!(r.peer.interrupts.lock().is_empty(), "interrupt number {bad:#x}");
    }
    r.w64(0x128, 0x65);
    r.w32(0x7c, 0);
    assert_eq!(r.peer.interrupts.lock().as_slice(), [(15, target, 0x165)]);
    r.w32(0x7c, 0);
    assert_eq!(r.peer.interrupts.lock().len(), 2, "every request sends while the status stays latched");
    // A repeat of the lower register's bits 47:32 in the upper register is
    // accepted; a different one, or a lone upper word, is not.
    r.w64(0x30, 1);
    r.w32(0x7c, 0);
    assert_eq!(r.peer.interrupts.lock().len(), 3);
    r.w64(0x30, 2);
    r.w32(0x7c, 0);
    r.w64(0x30, 0x3400_0000);
    r.w64(0x38, 0);
    r.w32(0x7c, 0);
    r.w64(0x30, 0x1_0000_0001);
    r.w64(0x38, target);
    r.w32(0x7c, 0);
    r.w64(0x30, 0x3400_0000);
    r.w64(0x38, 0x000f_0001_0000_3000);
    r.w32(0x7c, 0);
    assert_eq!(r.peer.interrupts.lock().len(), 3, "conflicts, zero and overflow are refused");
    // The PROM's two word writes join into the address.
    r.w64(0x30, 0);
    r.w64(0x38, 0);
    r.w32(0x34, 1);
    r.w32(0x3c, 0x3000);
    r.w32(0x7c, 0);
    assert_eq!(r.peer.interrupts.lock().last().map(|p| p.1), Some(target));
}

#[test]
fn two_pics_are_independent_devices() {
    let a = Pic::new(11);
    let b = Pic::new(15);
    let peer_a = Arc::new(Peer::default());
    let peer_b = Arc::new(Peer::default());
    a.connect_peer(peer_a.clone()).unwrap();
    b.connect_peer(peer_b.clone()).unwrap();
    assert_eq!(a.connect_peer(peer_b.clone()), Err(AlreadyConnected));
    assert_eq!((a.widget_number(), b.widget_number()), (11, 15));
    let (bus_a, bus_b) = (a.widget(), b.widget());
    bus_a.write64(0x20, 0x0f00_3205);
    assert_eq!(bus_b.read64(0x20).data, 0x0f00_320f, "state is per chip");
    bus_a.write64(0x130, destination(9));
    bus_a.write64(0x108, 1);
    a.set_pin(0, 0, true);
    assert_eq!(peer_a.interrupts.lock().as_slice(), [(5, 0x1800090, 0x109)], "the packet carries the control widget field");
    assert!(peer_b.interrupts.lock().is_empty());
    let card = Card::new(1);
    a.attach(0, 1, card.clone(), Function::default());
    assert!(!bus_b.read32(config(0, 1)).is_ok(), "the other chip's slot stays empty");
}

// An empty slot's configuration read fails as a master timeout. QEMU's
// sgi-pic.c returns all ones with no fault and has no counterpart case.

const MASTER_TIMEOUT_BIT: u64 = 1 << 11;
const STATUS: u64 = 0x100;
const ENABLE: u64 = 0x108;
const RESET: u64 = 0x110;
const VIEW: u64 = 0x170;

/// With every enable clear, as the PROM probes: an empty device fails on
/// either head, function 0 or 1, for every width the processor uses.
/// Error view bit 11 records it and status bit 11 stays clear.
#[test]
fn an_empty_slot_read_faults_and_only_the_error_view_records_it() {
    let r = rig();
    for head in [0, 1] {
        let base = head_base(head);
        for (device, function) in [(2u64, 0u64), (2, 1), (0, 0), (5, 1)] {
            let at = config(head, device) + function * 0x100;
            assert!(!r.bus.read32(at).is_ok(), "head {head} device {device}.{function}");
            assert!(!r.bus.read16(at + 2).is_ok());
            assert!(!r.bus.read8(at + 3).is_ok());
            assert_eq!(r.r64(base + VIEW) & MASTER_TIMEOUT_BIT, MASTER_TIMEOUT_BIT, "head {head}");
            assert_eq!(r.r32(base + VIEW + 4) as u64 & MASTER_TIMEOUT_BIT, MASTER_TIMEOUT_BIT, "low-word view");
            assert_eq!(r.r64(base + STATUS) & MASTER_TIMEOUT_BIT, 0, "status needs the enable");
            assert_eq!(r.r32(base + STATUS + 4) as u64 & MASTER_TIMEOUT_BIT, 0);
            // The PROM's low-word write of 0x41 clears the view.
            r.w32(base + RESET + 4, 0x41);
            assert_eq!(r.r64(base + VIEW), 0);
        }
        // The other head's view is its own.
        assert!(!r.bus.read32(config(head, 2)).is_ok());
        assert_eq!(r.r64(head_base(1 - head) + VIEW), 0);
        r.w64(base + RESET, 0x41);
        assert_eq!(r.r64(base + VIEW), 0, "an aligned write clears it too");
    }
    // The view is read-only and an unclaimed write records nothing.
    r.w64(VIEW, MASTER_TIMEOUT_BIT);
    r.w32(config(0, 2) + 4, 7);
    assert_eq!(r.r64(VIEW), 0);
    assert!(r.pic.unsupported().iter().any(|&(_, write, off, _)| write && off == VIEW));
    assert!(r.peer.interrupts.lock().is_empty());
}

/// With enable bit 11 set, status bit 11 sets too and the error interrupt
/// goes out. Status clears one bit at a time: the PROM's 0x41 leaves it set,
/// a one at bit 11 clears it.
#[test]
fn an_enabled_master_timeout_sets_status_until_bit_11_is_written() {
    let r = rig();
    r.w64(0x38, 0x1800090);
    r.w64(0x128, 9);
    r.w64(ENABLE, MASTER_TIMEOUT_BIT);
    assert!(!r.bus.read32(config(0, 2)).is_ok());
    assert_eq!(r.r64(STATUS) & MASTER_TIMEOUT_BIT, MASTER_TIMEOUT_BIT);
    assert_eq!(r.r64(VIEW) & MASTER_TIMEOUT_BIT, MASTER_TIMEOUT_BIT);
    assert_eq!(r.peer.interrupts.lock().len(), 1, "one error interrupt");
    r.w32(RESET + 4, 0x41);
    assert_eq!(r.r64(STATUS) & MASTER_TIMEOUT_BIT, MASTER_TIMEOUT_BIT, "0x41 leaves status bit 11");
    assert_eq!(r.r64(VIEW), 0, "but clears the view");
    r.w64(RESET, MASTER_TIMEOUT_BIT);
    assert_eq!(r.r64(STATUS) & MASTER_TIMEOUT_BIT, 0);
    assert_eq!(r.r64(HEAD1 + STATUS) & MASTER_TIMEOUT_BIT, 0, "head 1 untouched");
}

/// A populated device after an empty one still reads present once the
/// PROM's 0x41 has been written, as widget 15's second head lists device 3
/// after an empty device 2 on a real Tezro. A present single-function card
/// has header-type bit 7 clear, so the PROM never asks for its function 1,
/// which would fault.
#[test]
fn a_device_after_an_empty_one_is_still_found() {
    let r = rig();
    let card = Card::new(0x5003);
    r.pic.attach(1, 3, card, Function::default());
    let probe = |device: u64| {
        r.w32(HEAD1 + RESET + 4, 0x41);
        let read = r.bus.read32(config(1, device));
        let absent = !read.is_ok() || r.r32(HEAD1 + STATUS + 4) as u64 & MASTER_TIMEOUT_BIT != 0;
        (!absent).then_some(read.data)
    };
    assert_eq!(probe(1), None);
    assert_eq!(probe(2), None);
    assert_eq!(probe(3), Some(0x5003_1234));
    assert_eq!(probe(4), None);
    assert_eq!(r.r32(config(1, 3) + 0xc) >> 16 & 0x80, 0, "not multifunction");
    assert!(!r.bus.read32(config(1, 3) + 0x100).is_ok(), "its function 1 is unclaimed");
}

/// A reset clears the view with the rest of the head.
#[test]
fn reset_clears_the_error_view() {
    let r = rig();
    assert!(!r.bus.read32(config(0, 2)).is_ok());
    assert_ne!(r.r64(VIEW), 0);
    r.pic.power_on();
    assert_eq!(r.r64(VIEW), 0);
}

/// Two functions that decode the same range, as a probing PROM leaves them:
/// the shared bus answers from the lower device, and the other answers when
/// that one stops decoding. What real silicon does with two comparators
/// matching is not defined.
/// (cpu-pio-overlapping-bars-use-active-function)
#[test]
fn overlapping_bars_go_to_the_lower_device() {
    let r = rig();
    let first = Card::new(0x8001);
    let second = Card::new(0x8002);
    r.fit(0, 1, &first, Function::default());
    r.fit(0, 2, &second, Function::default());
    let at = window(0, 0x10);
    r.w32(at, 0x40);
    assert_eq!(le_value(&first.bar.lock()[0x10..0x14]), 0x40, "device 1 answers");
    assert_eq!(le_value(&second.bar.lock()[0x10..0x14]), 0);
    // Switching the winner off leaves the other one.
    r.w32(config(0, 1) + 4, 0);
    r.w32(at, 0x20);
    assert_eq!(le_value(&second.bar.lock()[0x10..0x14]), 0x20);
    assert_eq!(r.r32(at), 0x20);
    r.w32(config(0, 1) + 4, MEMORY_AND_MASTER);
    assert_eq!(r.r32(at), 0x40, "device 1 answers again");
}
