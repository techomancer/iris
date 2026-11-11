//! Register-contract tests for the PCI option boards: each board's Bridge as
//! the processor sees it through its widget window, a function in each slot
//! reached by configuration cycles, interrupts arriving at HEART as the
//! packets the guest programmed, and a function's DMA reaching memory.

use super::*;
use crate::dev::bridge::{device_window_offset, reg, CFG0_BASE, DEV_MEMORY_SPACE, DEV_SWAP_DIRECT};
use crate::irq::InterruptController;
use crate::pci::testing::TestCard;
use crate::pci::{Bar, PCI_COMMAND_MEMORY, PCI_INTA};
use crate::traits::{BusDevice, BusRead32, BusRead8, BUS_ERR, BUS_OK};
use parking_lot::Mutex;

/// HEART's interrupt side: the packets it received, `set_line(vector, set)`.
#[derive(Default)]
struct Heart(Mutex<Vec<(u8, bool)>>);

impl InterruptController for Heart {
    fn set_line(&self, line: u32, level: bool) {
        self.0.lock().push((line as u8, level));
    }
}

impl Heart {
    fn take(&self) -> Vec<(u8, bool)> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// 1 MB of memory at physical 0, big-endian like the bus.
struct Mem(Mutex<Vec<u8>>);

impl Mem {
    fn peek(&self, a: usize, n: usize) -> Vec<u8> {
        self.0.lock()[a..a + n].to_vec()
    }

    fn poke(&self, a: usize, data: &[u8]) {
        self.0.lock()[a..a + data.len()].copy_from_slice(data);
    }
}

impl BusDevice for Mem {
    fn read8(&self, a: u64) -> BusRead8 {
        self.0.lock().get(a as usize).map_or(BusRead8::err(), |&b| BusRead8::ok(b))
    }

    fn write8(&self, a: u64, v: u8) -> u32 {
        self.0.lock().get_mut(a as usize).map_or(BUS_ERR, |b| {
            *b = v;
            BUS_OK
        })
    }

    fn read32(&self, a: u64) -> BusRead32 {
        let m = self.0.lock();
        m.get(a as usize..a as usize + 4).map_or(BusRead32::err(), |b| BusRead32::ok(u32::from_be_bytes(b.try_into().unwrap())))
    }

    fn write32(&self, a: u64, v: u32) -> u32 {
        let mut m = self.0.lock();
        m.get_mut(a as usize..a as usize + 4).map_or(BUS_ERR, |b| {
            b.copy_from_slice(&v.to_be_bytes());
            BUS_OK
        })
    }
}

const HEART_WIDGET: u32 = 8;
/// A Bridge revision the machine would hand a PCI module, not the default.
const MACHINE_REVISION: u32 = 3;

struct Rig {
    board: XioPciBoard,
    heart: Arc<Heart>,
    mem: Arc<Mem>,
}

impl Rig {
    fn new(kind: Kind, widget: u8) -> Self {
        let heart = Arc::new(Heart::default());
        let mem = Arc::new(Mem(Mutex::new(vec![0; 0x10_0000])));
        let board = XioPciBoard::new(kind, widget, Upstream::single(crate::ip30::HEART_HOST, mem.clone(), heart.clone()), MACHINE_REVISION);
        Rig { board, heart, mem }
    }

    fn bridge(&self) -> &Bridge {
        self.board.bridge()
    }

    /// Register `off` (a slot base such as `reg::CONTROL`), data half.
    fn get(&self, off: u32) -> u32 {
        let r = self.bridge().read32(u64::from(off + 4));
        assert!(r.is_ok());
        r.data
    }

    fn set(&self, off: u32, val: u32) {
        assert_eq!(self.bridge().write32(u64::from(off + 4), val), BUS_OK);
    }

    /// A word of slot `device`'s configuration space, as a CPU reads it.
    fn config(&self, device: u8, off: u32) -> Option<u32> {
        let r = self.bridge().read32(u64::from(CFG0_BASE + device as u32 * 0x1000 + off));
        r.is_ok().then_some(r.data)
    }

    /// A card with a memory BAR and INT`pin` in `device`.
    fn card(&self, device: u8, pin: u8) -> Arc<TestCard> {
        let card = TestCard::new([Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], pin);
        self.board.attach(device, card.clone()).unwrap();
        card
    }

    /// Send interrupts on `pin` to HEART as `vector`, and enable the pin.
    fn route(&self, pin: u32, vector: u32) {
        self.set(reg::INT_DEST_UPPER, HEART_WIDGET << 16);
        self.set(reg::INT_DEST_LOWER, 0x80);
        self.set(reg::int_host(pin), vector);
        let enabled = self.get(reg::INT_ENABLE);
        self.set(reg::INT_ENABLE, enabled | 1 << pin);
    }
}

fn identity_word(rig: &Rig) -> u32 {
    rig.get(reg::IDENTIFICATION)
}

#[test]
fn a_shoehorn_is_a_bridge_with_revision_4_and_a_record() {
    let rig = Rig::new(Kind::Shoehorn, 12);
    let id = identity_word(&rig);
    assert_eq!(id >> 12 & 0xffff, 0xc002, "the Bridge's part number");
    assert_eq!(id >> 1 & 0x7ff, 0x036, "SGI's manufacturer code");
    assert_eq!(id >> 28, 4, "Bridge Rev D, whatever the machine's own Bridge is");
    assert_eq!(rig.board.widget(), 12);

    // The one-wire net holds one part, a DS2505 (family 0x0b).
    let slot = |pulse: u32| {
        rig.set(reg::NIC, pulse << 10 | 10 << 2);
        let v = rig.get(reg::NIC);
        assert_ne!(v & 2, 0, "a slot is over inside the write");
        v & 1 != 0
    };
    assert!(!slot(500), "a part answers the reset pulse");
    for i in 0..8 {
        slot(if 0x33 >> i & 1 != 0 { 10 } else { 100 });
    }
    let mut family = 0u8;
    for bit in 0..8 {
        family |= (slot(10) as u8) << bit;
    }
    assert_eq!(family, 0x0b);
}

#[test]
fn the_pci_module_reports_the_machines_bridge_revision_and_a_record() {
    let rig = Rig::new(Kind::Cage, crate::dev::xio_pci::CAGE_WIDGET);
    let id = identity_word(&rig);
    assert_eq!(id >> 12 & 0xffff, 0xc002);
    assert_eq!(id >> 28, MACHINE_REVISION);
    // A part answers the reset pulse by pulling the line low.
    rig.set(reg::NIC, 500 << 10 | 65 << 2);
    assert_eq!(rig.get(reg::NIC) & 3, 2);
}

#[test]
fn the_boards_have_the_slots_the_documents_number() {
    assert_eq!(Kind::Shoehorn.slots(), [1]);
    assert_eq!(Kind::Cage.slots(), [1, 2, 3]);
    assert_eq!(crate::dev::xio_pci::CAGE_WIDGET, 13);
}

/// A function in each slot answers configuration reads through the
/// widget's window; every other device position times out.
fn each_slot_answers(kind: Kind, widget: u8) {
    let rig = Rig::new(kind, widget);
    for device in 0..8u8 {
        assert_eq!(rig.config(device, 0), None, "device {device} is empty");
    }
    for &device in kind.slots() {
        rig.card(device, PCI_INTA);
        // Vendor in the low half, device in the high half, as the CPU reads
        // the big-endian window.
        assert_eq!(rig.config(device, 0), Some(0x5678_1234), "slot {device} answers");
    }
    for device in (0..8u8).filter(|d| !kind.slots().contains(d)) {
        assert_eq!(rig.config(device, 0), None, "device {device} has no slot");
    }
}

#[test]
fn the_shoehorns_slot_answers_configuration_reads() {
    each_slot_answers(Kind::Shoehorn, 12);
}

#[test]
fn each_slot_of_the_pci_module_answers_configuration_reads() {
    each_slot_answers(Kind::Cage, 13);
}

#[test]
fn the_flash_aperture_reads_all_ones_and_ignores_writes() {
    let rig = Rig::new(Kind::Shoehorn, 10);
    for off in [0xc0_0000u32, 0xc0_0004, 0xdf_fffc, 0xe0_0000, 0xff_fffc] {
        assert_eq!(rig.bridge().read32(u64::from(off)).data, 0xffff_ffff, "{off:#x}");
    }
    // Writes are refused while the control register's write enable is clear.
    assert_ne!(rig.bridge().write16(0xc0_0000, 0x1234), BUS_OK);
}

#[test]
fn a_function_goes_only_in_a_free_slot_the_board_has() {
    let rig = Rig::new(Kind::Cage, 13);
    let card = TestCard::new([Bar::Unused; 6], 0);
    assert!(rig.board.attach(0, card.clone()).is_err(), "device 0 is not a slot");
    assert!(rig.board.attach(4, card.clone()).is_err(), "nor is device 4");
    assert_eq!(rig.board.free_slot(), Some(1));
    rig.board.attach(1, card.clone()).unwrap();
    assert!(rig.board.attach(1, card.clone()).is_err(), "slot 1 is taken");
    assert_eq!(rig.board.free_slot(), Some(2));
    rig.board.attach(2, TestCard::new([Bar::Unused; 6], 0)).unwrap();
    rig.board.attach(3, TestCard::new([Bar::Unused; 6], 0)).unwrap();
    assert_eq!(rig.board.free_slot(), None);
}

/// INTA of a slot reaches the Bridge pin numbered as the device, INTB that
/// number with bit two flipped, and each is a packet to HEART naming the
/// vector the guest wrote for the pin.
fn interrupts(kind: Kind, widget: u8) {
    let rig = Rig::new(kind, widget);
    for (n, &device) in kind.slots().iter().enumerate() {
        let a = rig.card(device, PCI_INTA);
        let vector = 20 + n as u32;
        rig.route(device as u32, vector);
        a.state().irq().unwrap().set(true);
        assert_eq!(rig.heart.take(), [(vector as u8, true)], "INTA of slot {device}");
        a.state().irq().unwrap().set(false);
        rig.heart.take();
    }
    // INTB of a slot, on a function in the same slot of another board.
    let rig = Rig::new(kind, widget);
    let device = kind.slots()[0];
    let b = rig.card(device, PCI_INTA + 1);
    rig.route(device as u32 ^ 4, 40);
    b.state().irq().unwrap().set(true);
    assert_eq!(rig.heart.take(), [(40, true)], "INTB of slot {device} on pin {}", device ^ 4);
}

#[test]
fn a_shoehorn_cards_interrupts_reach_heart() {
    interrupts(Kind::Shoehorn, 10);
}

#[test]
fn the_pci_modules_interrupts_reach_heart() {
    interrupts(Kind::Cage, 13);
}

/// A function's DMA goes through the Bridge's direct map to the memory the
/// board was given. The default direct map names no widget that is memory.
fn dma(kind: Kind, widget: u8) {
    let rig = Rig::new(kind, widget);
    let card = rig.card(kind.slots()[0], PCI_INTA);
    let dma = card.state().dma().unwrap().clone();
    let mut buf = [0u8; 4];
    assert!(!dma.dma_read(0x8000_1000, &mut buf).is_ok(), "the direct map starts at nothing");

    rig.set(reg::DIRECT_MAP, HEART_WIDGET << 20);
    // Swap clear on a 32-bit slot: each aligned word is reversed.
    assert!(dma.dma_write(0x8000_1000, &[1, 2, 3, 4]).is_ok());
    assert_eq!(rig.mem.peek(0x1000, 4), [4, 3, 2, 1]);
    assert!(dma.dma_read(0x8000_1000, &mut buf).is_ok());
    assert_eq!(buf, [1, 2, 3, 4]);

    // A request that names the board's own port goes nowhere.
    rig.set(reg::DIRECT_MAP, (widget as u32) << 20);
    assert!(!dma.dma_write(0x8000_2000, &[9; 4]).is_ok());
    assert_eq!(rig.mem.peek(0x2000, 4), [0; 4]);
}

#[test]
fn a_shoehorn_cards_dma_reaches_memory() {
    dma(Kind::Shoehorn, 9);
}

#[test]
fn a_pci_module_cards_dma_reaches_memory() {
    dma(Kind::Cage, 13);
}

#[test]
fn two_boards_keep_their_own_interrupt_state() {
    let heart = Arc::new(Heart::default());
    let mem: Arc<dyn BusDevice> = Arc::new(Mem(Mutex::new(vec![0; 0x1000])));
    let up = || Upstream::single(crate::ip30::HEART_HOST, mem.clone(), heart.clone());
    let one = XioPciBoard::new(Kind::Shoehorn, 12, up(), 3);
    let two = XioPciBoard::new(Kind::Cage, 13, up(), 3);
    let a = TestCard::new([Bar::Unused; 6], PCI_INTA);
    let b = TestCard::new([Bar::Unused; 6], PCI_INTA);
    one.attach(1, a.clone()).unwrap();
    two.attach(1, b.clone()).unwrap();
    for (bridge, vector) in [(one.bridge(), 30u32), (two.bridge(), 31)] {
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        set(reg::INT_DEST_UPPER, HEART_WIDGET << 16);
        set(reg::INT_DEST_LOWER, 0x80);
        set(reg::int_host(1), vector);
        set(reg::INT_ENABLE, 1 << 1);
    }
    b.state().irq().unwrap().set(true);
    assert_eq!(heart.take(), [(31, true)], "only the second board's pin rose");
    a.state().irq().unwrap().set(true);
    assert_eq!(heart.take(), [(30, true)]);
}

/// An NVMe card on a board, driven through the slot's device window and the
/// direct map with the swap bit set (a byte stream, as a little-endian master
/// wants), the way the Octane's own Bridge drives one.
struct NvmeHost<'a>(&'a Rig, u8);

impl crate::dev::nvme::tests::HostIo for NvmeHost<'_> {
    fn w(&self, off: u32, v: u32) {
        assert_eq!(self.0.bridge().write32(u64::from(device_window_offset(self.1 as u32) + off), v), BUS_OK);
    }

    fn r(&self, off: u32) -> u32 {
        self.0.bridge().read32(u64::from(device_window_offset(self.1 as u32) + off)).data
    }

    fn put(&self, at: u64, data: &[u8]) {
        self.0.mem.poke(at as usize, data);
    }

    fn get(&self, at: u64, len: usize) -> Vec<u8> {
        self.0.mem.peek(at as usize, len)
    }

    fn bus(&self, at: u64) -> u64 {
        0x8000_0000 + at
    }
}

/// An NVMe card in `slot` of a board, directly or behind a PCI-to-PCI bridge,
/// is configured as firmware would, runs the controller through a queue pair
/// with data both ways, and its completions reach HEART on the slot's pin.
fn nvme_in_slot(kind: Kind, widget: u8, slot: u8, behind_bridge: bool) {
    let rig = Rig::new(kind, widget);
    let img = crate::dev::nvme::tests::temp_image(&format!("xio-{}-{slot}-{behind_bridge}", kind.name()));
    let card = crate::dev::nvme::Card::open(&crate::config::NvmeConfig {
        path: img.to_string_lossy().into_owned(),
        bridge: behind_bridge,
        ..Default::default()
    })
    .unwrap();
    rig.board.check_slot(slot).unwrap();
    card.fit(rig.board.bus(), slot);
    let cfg_write = |off: u32, size: u32, v: u32| {
        let (a, w) = (u64::from(CFG0_BASE + u32::from(slot) * 0x1000 + ((off & !3) | ((off & 3) ^ (4 - size)))), size);
        let s = match w {
            2 => rig.bridge().write16(a, v as u16),
            _ => rig.bridge().write32(a, v),
        };
        assert_eq!(s, BUS_OK);
    };
    rig.set(reg::DIRECT_MAP, HEART_WIDGET << 20);
    rig.set(reg::device(slot as u32), DEV_MEMORY_SPACE | DEV_SWAP_DIRECT | 0x130);
    rig.route(slot as u32, 0x21);
    if behind_bridge {
        // Bus 1 behind the bridge in the slot, memory 0x13000000 forwarded.
        cfg_write(0x18, 4, 0x0001_0100);
        cfg_write(0x20, 4, 0x130f_1300);
        cfg_write(0x04, 2, PCI_COMMAND_MEMORY);
        let sec = rig.board.bus().find_bus(1).expect("bus 1");
        let at = DevFn { device: 0, function: 0 };
        sec.config_write(at, 0x10, 4, 0x1300_0000);
        sec.config_write(at, 0x14, 4, 0);
        sec.config_write(at, 0x04, 2, PCI_COMMAND_MEMORY);
    } else {
        cfg_write(0x10, 4, 0x1300_0000);
        cfg_write(0x14, 4, 0);
        cfg_write(0x04, 2, PCI_COMMAND_MEMORY);
    }
    rig.heart.take();
    crate::dev::nvme::tests::smoke(&NvmeHost(&rig, slot));
    // The completion raised INTA of the slot, Bridge pin `slot`: a packet
    // for vector 0x21 reached HEART.
    assert!(rig.heart.take().contains(&(0x21, true)), "{} slot {slot}", kind.label());
}

#[test]
fn an_nvme_card_in_the_shoehorn_masters_ram_and_raises_its_pin() {
    nvme_in_slot(Kind::Shoehorn, 10, 1, false);
}

#[test]
fn an_nvme_card_behind_a_bridge_in_the_shoehorn_does_the_same() {
    nvme_in_slot(Kind::Shoehorn, 12, 1, true);
}

#[test]
fn an_nvme_card_in_each_slot_of_the_pci_module_masters_ram_and_raises_its_pin() {
    for slot in 1..=3 {
        nvme_in_slot(Kind::Cage, 13, slot, false);
    }
}

#[test]
fn an_nvme_card_behind_a_bridge_in_the_pci_module_does_the_same() {
    nvme_in_slot(Kind::Cage, 13, 2, true);
}
