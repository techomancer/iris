// Cases follow tests/qtest/sn1-xbridge-chip-test.c (head-dma-reaches-the-host)
// and sn1-xbridge-test.c (interrupt-packets, link-reset) in
// jamesbraid/qemu-sgi-origami. The heads' and the crossbar's register
// contracts are tested with their parts in bridge_tests.rs and xbow.rs.
use super::*;
use crate::pci::{Bar, BoardIrq, DevFn, PciConfig, PciConfigPolicy, PciDmaStatus, PciFunction, PciFunctionState};
use crate::traits::{BusRead32, BusRead8, BUS_ERR, BUS_OK};
use parking_lot::Mutex;

const BEDROCK: XtalkHost = XtalkHost { widget: HOST_WIDGET, int_packet: 0, int_packet_mask: 0, vectors: 256 };
const INT_DEST_UPPER: u64 = 0x30 + 4;
const INT_DEST_LOWER: u64 = 0x38 + 4;
const INT_ENABLE: u64 = 0x108 + 4;
const INT_MODE: u64 = 0x118 + 4;
const INT_STATUS: u64 = 0x100 + 4;
const INT_HOST_0: u64 = 0x130 + 4;
const CONTROL: u64 = 0x20 + 4;
const DIRECT_MAP: u64 = 0x80 + 4;
const PCI_TYPE1_CONFIG: u64 = 0xc8 + 4;
const XBOW_INT_UPPER: u64 = 0x30 + 4;
const XBOW_INT_LOWER: u64 = 0x38 + 4;
const XBOW_CONTROL: u64 = 0x20 + 4;
/// Port 15's (link 7's) reset register in the crossbar.
const LINK_RESET_15: u64 = 0x100 + 7 * 0x40 + 0x30 + 4;

fn at(device: u8) -> DevFn {
    DevFn { device, function: 0 }
}

/// Bytes in address order, as the host's memory holds them.
struct Memory(Mutex<Vec<u8>>);
impl BusDevice for Memory {
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
        self.0.lock().get_mut(a as usize..a as usize + 4).map_or(BUS_ERR, |b| {
            b.copy_from_slice(&v.to_be_bytes());
            BUS_OK
        })
    }
}

/// The host as a widget reaches it: one memory and the packets taken, with
/// the widget that asked for this port.
#[derive(Default)]
struct Packets(Mutex<Vec<(u8, u8, u64, u32, bool)>>);
struct Port {
    widget: u8,
    packets: Arc<Packets>,
}
impl InterruptController for Port {
    fn set_line(&self, _: u32, _: bool) {
        unreachable!("the host takes packets");
    }
    fn packet(&self, source: u8, address: u64, line: u32, level: bool) -> bool {
        self.packets.0.lock().push((self.widget, source, address, line, level));
        true
    }
}

fn chip() -> (Arc<XBridge>, Arc<Memory>, Arc<Packets>) {
    let mem = Arc::new(Memory(Mutex::new(vec![0; 0x8000])));
    let packets = Arc::new(Packets::default());
    let (m, p) = (mem.clone(), packets.clone());
    let chip = XBridge::new(BEDROCK, &|widget| {
        (m.clone() as Arc<dyn BusDevice>, Arc::new(Port { widget, packets: p.clone() }) as Arc<dyn InterruptController>)
    });
    (chip, mem, packets)
}

struct Card {
    state: PciFunctionState,
    bytes: Mutex<Vec<u8>>,
}
impl Card {
    fn new() -> Arc<Self> {
        Self::with_status_events(0)
    }
    fn with_status_events(mask: u16) -> Arc<Self> {
        Arc::new(Self {
            state: PciFunctionState::new(PciConfig::new_with_policy(
                0x1234,
                0x5678,
                0,
                0,
                [Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                1,
                PciConfigPolicy { status_w1c_mask: mask, ..Default::default() },
            )),
            bytes: Mutex::new((0..0x1000).map(|i| i as u8).collect()),
        })
    }
}
impl PciFunction for Card {
    fn state(&self) -> &PciFunctionState {
        &self.state
    }
    fn bar_read(&self, _: usize, off: u32, width: usize) -> u32 {
        let b = self.bytes.lock();
        (0..width).fold(0, |v, i| v | ((b[off as usize + i] as u32) << (i * 8)))
    }
    fn bar_write(&self, _: usize, off: u32, value: u32, width: usize) {
        let mut b = self.bytes.lock();
        for i in 0..width {
            b[off as usize + i] = (value >> (i * 8)) as u8;
        }
    }
    fn reset(&self) {
        if let Some(line) = self.state.irq() {
            line.set(false);
        }
    }
}

/// The machine lock registry must not extend the lifetime of a chip or attached function.
#[test]
fn lock_registry_does_not_keep_the_chip_or_its_devices_alive() {
    let (chip, mem, _) = chip();
    let memory = Arc::downgrade(&mem);
    drop(mem);
    let chip_ref = Arc::downgrade(&chip);
    let heads = chip.heads.each_ref().map(Arc::downgrade);
    let crossbar = Arc::downgrade(&chip.xbow);
    let card = Card::new();
    let card_ref = Arc::downgrade(&card);
    chip.bus(HEAD_F).attach(at(1), card.clone());
    chip.set_reset_output(Arc::new(move |_| card.state.config().reset()));
    chip.register_locks();
    assert!(heads.iter().all(|head| head.upgrade().is_some()));
    assert!(crossbar.upgrade().is_some() && card_ref.upgrade().is_some() && memory.upgrade().is_some());
    drop(chip);
    assert!(chip_ref.upgrade().is_none());
    assert!(heads.iter().all(|head| head.upgrade().is_none()));
    assert!(crossbar.upgrade().is_none());
    assert!(card_ref.upgrade().is_none());
    assert!(memory.upgrade().is_none());
}

/// Each head has a bus of its own: the same device number on the two heads
/// is two functions, configuration reaches only the head addressed, and a
/// function's INTA raises its own head's pin of that number alone.
#[test]
fn each_head_has_its_own_configuration_space_and_interrupt_pins() {
    let (c, _, _) = chip();
    let (first, second) = (Card::new(), Card::new());
    c.bus(HEAD_E).attach(at(3), first.clone());
    c.bus(HEAD_F).attach(at(3), second.clone());
    let (e, f) = (c.widget(HEAD_E).unwrap(), c.widget(HEAD_F).unwrap());
    for head in [&e, &f] {
        assert_eq!(head.read32(0x23000).data, 0x5678_1234);
    }
    assert_eq!(e.write32(0x2303c, 0x5a), BUS_OK);
    assert_eq!(first.state.config().read(15) & 0xff, 0x5a);
    assert_eq!(second.state.config().read(15) & 0xff, 0, "the other head's function is untouched");
    let pins = |head: &Arc<dyn BusDevice>| head.read32(INT_STATUS).data & 255;
    second.state.irq().unwrap().set(true);
    assert_eq!((pins(&e), pins(&f)), (0, 1 << 3));
    first.state.irq().unwrap().set(true);
    assert_eq!((pins(&e), pins(&f)), (1 << 3, 1 << 3));
    second.state.irq().unwrap().set(false);
    assert_eq!((pins(&e), pins(&f)), (1 << 3, 0));
    // The board's own wire shares pin 0.
    c.bus(HEAD_F).board_irq(BoardIrq(0)).set(true);
    assert_eq!(pins(&f), 1);
}

/// Each head sends to the host through the port of its own widget: DMA
/// lands in that port's memory, and a pin's packet carries the widget ID
/// the guest gave the head and the address it programmed. The crossbar's
/// packets come from widget 0.
#[test]
fn the_heads_and_the_crossbar_reach_the_host_through_their_own_ports() {
    let (c, mem, packets) = chip();
    let f = c.widget(HEAD_F).unwrap();
    f.write32(DIRECT_MAP, u32::from(HOST_WIDGET) << 20);
    assert_eq!(c.dma(HEAD_F, 0).dma_write(0x8000_2000, &[0, 1, 2, 3]), PciDmaStatus::Ok);
    assert_eq!(&mem.0.lock()[0x2000..0x2004], &[3, 2, 1, 0]);
    f.write32(CONTROL, (f.read32(CONTROL).data & !0xf) | u32::from(HEAD_F));
    f.write32(INT_DEST_UPPER, 0xa_0001);
    f.write32(INT_DEST_LOWER, 0x2345_0080);
    f.write32(INT_HOST_0, (0x155 << 8) | 0x42);
    f.write32(INT_ENABLE, 1);
    f.write32(INT_MODE, 1);
    let wire = c.bus(HEAD_F).board_irq(BoardIrq(0));
    wire.set(true);
    wire.set(false);
    let address = (0x1_2345_0080u64 & !(0x3ff << 38)) | (0x155 << 38);
    assert_eq!(*packets.0.lock(), [(HEAD_F, HEAD_F, address, 0x42, true), (HEAD_F, HEAD_F, address, 0x42, false)]);
    packets.0.lock().clear();
    let x = c.crossbar();
    x.write32(XBOW_INT_UPPER, 0x150a_0000);
    x.write32(XBOW_INT_LOWER, 0x80);
    x.write32(XBOW_CONTROL, 0x20);
    x.write32(4, 1);
    assert_eq!(*packets.0.lock(), [(0, 0, 0x80, 0x15, true)]);
}

/// A link reset of a head's port resets the head and runs RST# on its bus
/// before the board's output runs, so a function's BAR stops answering;
/// the output may reach the chip and replace itself.
#[test]
fn a_link_reset_resets_the_head_and_its_bus_before_the_board() {
    use crate::dev::isp1040::{mbox, reg as isp_reg, CTRL_INT_ENABLE, CTRL_RISC_INT_ENABLE, HOST_CMD_SHIFT};
    use crate::dev::isp12160::Isp12160;

    let (c, _, _) = chip();
    let isp = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus()));
    c.bus(HEAD_F).attach(at(1), isp.clone());
    let f = c.widget(HEAD_F).unwrap();
    let pin1 = |f: &Arc<dyn BusDevice>| f.read32(INT_STATUS).data & 2 != 0;
    // BAR 1 at PCI 0x0200_0000, memory and mastering on, through the head.
    assert_eq!(f.write32(0x21014, 0x0200_0000), BUS_OK);
    assert_eq!(f.write32(0x21004, 6), BUS_OK);
    isp.bar_write(1, isp_reg::BUS_CTRL, u32::from(CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE), 2);
    isp.bar_write(1, isp_reg::MAILBOX_BASE, u32::from(mbox::NOP), 2);
    isp.bar_write(1, isp_reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT, 2);
    assert!(pin1(&f));
    assert_eq!(f.read16(0x4200_0000 + u64::from(isp_reg::MAILBOX_BASE) ^ 2).data, mbox::COMPLETE);

    let weak = Arc::downgrade(&c);
    let reset_isp = isp.clone();
    c.set_reset_output(Arc::new(move |port| {
        assert_eq!(port, HEAD_F);
        let chip = weak.upgrade().unwrap();
        // The head is back at reset, its function's header too.
        assert_eq!(chip.widget(port).unwrap().read32(DIRECT_MAP).data, 0x20000);
        assert_eq!(reset_isp.state().config().command(), 0);
        reset_isp.power_on();
        chip.set_reset_output(Arc::new(|_| {}));
    }));
    f.write32(DIRECT_MAP, 0xa0_0000);
    c.crossbar().write32(LINK_RESET_15, 1);
    assert!(!pin1(&f));
    assert_eq!(isp.state().config().bar_base(1), 0);
    assert_eq!(f.read32(0x4200_0000).data, u32::MAX, "the BAR no longer answers");
    assert_eq!(isp.bar_read(1, isp_reg::BUS_CTRL, 2), 0);
    assert_eq!(isp.bar_read(1, isp_reg::MAILBOX_BASE + 2, 2), 0x4953);
    assert!(c.xbow.carries(HEAD_F), "the link comes back");
}

/// A board fitted at a port answers there while its link is up.
#[test]
fn an_attached_board_answers_at_its_port() {
    let (c, _, _) = chip();
    let board = Arc::new(Memory(Mutex::new(vec![0x5a; 8])));
    let port = c.attach_widget(13, board);
    assert_eq!(port.read8(3).data, 0x5a);
    assert_ne!(c.crossbar().read32(0x100 + 5 * 0x40 + 0x10 + 4).data & (1 << 31), 0, "link 13 is alive");
}

/// Narrow guest configuration writes preserve status bits outside the
/// written bytes, through either aperture.
#[test]
fn partial_configuration_writes_preserve_unwritten_status_events() {
    let (c, _, _) = chip();
    let f = c.widget(HEAD_F).unwrap();
    let card = Card::with_status_events(0xc000);
    c.bus(HEAD_F).attach(at(0), card.clone());
    for (bus, expected) in [(0, 0x5678_1234), (1, u32::MAX)] {
        f.write32(PCI_TYPE1_CONFIG, bus << 16);
        assert_eq!(f.read32(0x28000).data, expected);
    }
    card.state.config().set_status_bits(0xc000);
    assert_eq!(f.write16(0x20006, 0x0102), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0xc200_0102);
    assert_eq!(f.write8(0x20007, 3), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0xc200_0103);
    assert_eq!(f.write8(0x20006, 2), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0xc200_0203);
    assert_eq!(f.write8(0x20005, 0xff), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0xc200_0203);
    assert_eq!(f.write8(0x20004, 0x80), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0x4200_0203);
    assert_eq!(f.write16(0x20004, 0x4000), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0x0200_0203);
    card.state.config().set_status_bits(0xc000);
    assert_eq!(f.write32(0x20004, 0x8000_0002), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0x4200_0002);
}
