// Cases follow tests/qtest/sn1-xbridge-test.c (page-dma, interrupt-packets,
// access-errors), sn1-xbridge-chip-test.c (head-dma-reaches-the-host,
// warm-reset-reaches-the-heads) and sn1-xbridge-xbow-test.c (reset-values,
// lock, fences, link-reset-one-shot) in jamesbraid/qemu-sgi-origami.
use super::*;
use crate::traits::BUS_ERR;
struct Memory(Mutex<Vec<u8>>);
impl BusDevice for Memory {
    fn read32(&self, a: u64) -> BusRead32 {
        let mem = self.0.lock();
        if a as usize + 4 > mem.len() {
            return BusRead32::err();
        }
        BusRead32::ok(
            mem[a as usize..a as usize + 4]
                .iter()
                .fold(0, |v, &b| (v << 8) | b as u32),
        )
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        let mut mem = self.0.lock();
        if a as usize + 4 > mem.len() {
            return BUS_ERR;
        }
        for i in 0..4 {
            mem[a as usize + i] = (v >> (24 - i * 8)) as u8;
        }
        BUS_OK
    }

    fn read64(&self, a: u64) -> BusRead64 {
        let mem = self.0.lock();
        if a as usize + 8 > mem.len() {
            return BusRead64::err();
        }
        BusRead64::ok(
            mem[a as usize..a as usize + 8]
                .iter()
                .fold(0, |v, &b| (v << 8) | b as u64),
        )
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        let mut mem = self.0.lock();
        if a as usize + 8 > mem.len() {
            return BUS_ERR;
        }
        for i in 0..8 {
            mem[a as usize + i] = (v >> (56 - i * 8)) as u8;
        }
        BUS_OK
    }
    fn write64_masked(&self, a: u64, v: u64, mask: u64) -> u32 {
        let old = self.read64(a);
        if old.status != BUS_OK {
            return BUS_ERR;
        }
        self.write64(a, (old.data & !mask) | (v & mask))
    }

    fn read8(&self, a: u64) -> BusRead8 {
        match self.0.lock().get(a as usize) {
            Some(&value) => BusRead8 {
                data: value,
                status: BUS_OK,
            },
            None => BusRead8 {
                data: 0,
                status: BUS_ERR,
            },
        }
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        match self.0.lock().get_mut(a as usize) {
            Some(b) => {
                *b = v;
                BUS_OK
            }
            None => BUS_ERR,
        }
    }
}
fn chip() -> (Arc<XBridge>, Arc<Memory>) {
    let mem = Arc::new(Memory(Mutex::new(vec![0; 0x8000])));
    (XBridge::new(mem.clone()), mem)
}

/// The machine lock registry must not extend the lifetime of a chip or attached function.
#[test]
fn lock_registry_does_not_keep_the_chip_or_its_devices_alive() {
    let memory = Arc::new(Memory(Mutex::new(vec![0; 0x8000])));
    let memory_ref = Arc::downgrade(&memory);
    let chip = XBridge::new(memory);
    let chip_ref = Arc::downgrade(&chip);
    let heads = chip.heads.each_ref().map(Arc::downgrade);
    let pio = chip.heads.each_ref().map(|head| Arc::downgrade(&head.pio));
    let crossbar = Arc::downgrade(&chip.xbow);
    let card = Card::with_bar(crate::pci::Bar::Mem(0x1000), 0);
    let card_ref = Arc::downgrade(&card);
    chip.attach(15, 1, card.clone());
    chip.set_reset_output(Arc::new(move |_| card.config.reset()));
    chip.head(15).write32(0x21010, 0x02000000);
    chip.head(15).write32(0x21004, 2);
    assert!(!chip.pci_head(15).pio_map.lock().runs.is_empty());
    chip.register_locks();

    assert!(heads.iter().all(|head| head.upgrade().is_some()));
    assert!(crossbar.upgrade().is_some());
    assert!(card_ref.upgrade().is_some());
    assert!(memory_ref.upgrade().is_some());
    drop(chip);
    assert!(chip_ref.upgrade().is_none());
    assert!(heads.iter().all(|head| head.upgrade().is_none()));
    assert!(pio.iter().all(|view| view.upgrade().is_none()));
    assert!(crossbar.upgrade().is_none());
    assert!(card_ref.upgrade().is_none());
    assert!(memory_ref.upgrade().is_none());
}
/// A DMA packet reports failure without losing the valid prefix already transferred.
#[test]
fn page_dma_keeps_valid_prefix_and_refuses_invalid_suffix() {
    physical_chunk_refusal_does_not_publish_a_prefix();
    let (c, mem) = chip();
    let f = c.head(15);
    f.write64(reg::ATE_UPPER_BASE as u64, 0x2000 | (10 << 8) | 0x21);
    let dma = c.dma(15, 2);
    let bytes = [1, 2, 3, 4, 5, 6, 7, 8];
    assert_eq!(dma.dma_write(0x40000ffc, &bytes), PciDmaStatus::MasterAbort);
    assert_eq!(&mem.0.lock()[0x2ffc..0x3000], &[1, 2, 3, 4]);
    assert_eq!(&mem.0.lock()[0x3000..0x3004], &[0; 4]);
    let mut read = [0xff; 8];
    assert_eq!(
        dma.dma_read(0x40000ffc, &mut read),
        PciDmaStatus::MasterAbort
    );
    assert_eq!(read, [1, 2, 3, 4, 0, 0, 0, 0]);
    assert_ne!(f.read32((reg::INTERRUPT_STATUS + 4) as u64).data & (1 << 30), 0);
    f.write32((reg::INTERRUPT_RESET + 4) as u64, 2);
    assert_eq!(f.read32((reg::INTERRUPT_STATUS + 4) as u64).data & (1 << 30), 0);
    f.write64((reg::ATE_UPPER_BASE + 800 * 8) as u64, 0xffff123456789abc);
    assert_eq!(f.read32((reg::ATE_UPPER_BASE + 4 + 800 * 8) as u64).data, 0x1234);
    assert_eq!(f.read32((reg::ATE_LOWER_BASE + 4 + 800 * 8) as u64).data, 0x56789abc);
    f.write64((reg::ATE_UPPER_BASE + 512 * 8) as u64, 0x2a21);
    f.write32((reg::DIRECT_MAP + 4) as u64, 9 << 20);
    for (address, status) in [(0x80002000, PciDmaStatus::MasterAbort),
        (0x3fff8000, PciDmaStatus::Ok), (0x40800000, PciDmaStatus::MasterAbort),
        (0x40200000, PciDmaStatus::MasterAbort)] {
        let mut bytes = [0xff; 4];
        assert_eq!(dma.dma_read(address, &mut bytes), status);
        assert_eq!(bytes, [0; 4]);
    }
}
// A mapped memory range whose middle word refuses accesses.
struct FaultMemory(Arc<Memory>);
impl BusDevice for FaultMemory {
    fn read8(&self, a: u64) -> BusRead8 {
        if (0x2004..0x2008).contains(&a) {
            BusRead8::err()
        } else {
            self.0.read8(a)
        }
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        if (0x2004..0x2008).contains(&a) {
            BUS_ERR
        } else {
            self.0.write8(a, v)
        }
    }
    fn read32(&self, a: u64) -> BusRead32 {
        if (0x2004..0x2008).contains(&a) {
            BusRead32::err()
        } else {
            self.0.read32(a)
        }
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        if (0x2004..0x2008).contains(&a) {
            BUS_ERR
        } else {
            self.0.write32(a, v)
        }
    }
}
fn physical_chunk_refusal_does_not_publish_a_prefix() {
    let mem = Arc::new(Memory(Mutex::new(vec![0xaa; 0x8000])));
    let c = XBridge::new(Arc::new(FaultMemory(mem.clone())));
    let f = c.head(15);
    f.write32((reg::DIRECT_MAP + 4) as u64, 10 << 20);
    f.write32((reg::DEVICE_BASE + 4) as u64, 1 << 19);
    let dma = c.dma(15, 0);
    let mut bytes = [0xff; 8];
    assert_eq!(
        dma.dma_read(0x80002002, &mut bytes),
        PciDmaStatus::MasterAbort
    );
    assert_eq!(
        bytes, [0; 8],
        "a failed physical chunk zeroes its readable prefix and suffix"
    );
    assert_eq!(
        dma.dma_write(0x80002002, &[1; 8]),
        PciDmaStatus::MasterAbort
    );
    assert_eq!(
        &mem.0.lock()[0x2000..0x2010],
        &[0xaa; 16],
        "the probe prevents writes before or after the refused word"
    );
}
/// Enabling an asserted guest interrupt pin delivers its programmed destination.
#[test]
fn interrupt_keeps_destination_and_enabling_a_live_pin_asserts() {
    let (c, _) = chip();
    let f = c.head(15);
    let packets = Arc::new(Mutex::new(Vec::new()));
    let capture = packets.clone();
    c.set_interrupt_sender(Arc::new(move |p| {
        capture.lock().push(p);
        true
    }));
    f.write32((reg::INT_DEST_UPPER + 4) as u64, 0xa0001);
    f.write32((reg::INT_DEST_LOWER + 4) as u64, 0x23450080);
    f.write32((reg::INTERRUPT_HOST_BASE + 4) as u64, (0x155 << 8) | 0x42);
    c.pin_line(15, 0).set(true);
    assert!(packets.lock().is_empty());
    f.write32((reg::INTERRUPT_ENABLE + 4) as u64, 1);
    f.write32((reg::INTERRUPT_MODE + 4) as u64, 1);
    c.pin_line(15, 0).set(false);
    let p = packets.lock();
    assert_eq!(p.len(), 2);
    assert_eq!(p[0].source_widget, 15);
    assert_eq!(p[0].widget, 10);
    assert_eq!(p[0].vector, 0x42);
    assert_eq!(p[0].address, (0x123450080 & !(0x3ff << 38)) | (0x155 << 38));
    assert!(p[0].level);
    assert!(!p[1].level);
    drop(p);
    let card = Card::new();
    c.attach(15, 2, card.clone());
    let (probe, line) = crate::irq::LevelProbe::line();
    c.set_pci_irq(15, 2, 0, line);
    card.irq.lock().as_ref().unwrap().set(true);
    assert!(probe.level());
    assert_eq!(f.read32((reg::INTERRUPT_STATUS + 4) as u64).data & 255, 0);
    c.reset();
    assert!(!probe.level());
}

/// The guest receives host DMA packets with their original widget and address placement.
#[test]
fn host_dma_places_packets_and_rejects_invalid_addresses() {
    let (c, mem) = chip();
    let f = c.head(15);
    let d = c.dma(15, 0);
    f.write32((reg::DIRECT_MAP + 4) as u64, 10 << 20);
    f.write64(reg::ATE_UPPER_BASE as u64, 0x3000 | (10 << 8) | 1);
    for (wide, address, target, expected) in [
        (false, 0x80002000, 0x2000, [3, 2, 1, 0, 7, 6, 5, 4]),
        (true, 0x80002000, 0x2000, [7, 6, 5, 4, 3, 2, 1, 0]),
        (true, 0x60000000, 0x3000, [0, 1, 2, 3, 4, 5, 6, 7]),
        (true, (10 << 60) | (1 << 55) | 0x2000, 0x2000, [0, 1, 2, 3, 4, 5, 6, 7]),
    ] {
        c.set_dma_64bit(15, 0, wide);
        assert_eq!(d.dma_write(address, &[0, 1, 2, 3, 4, 5, 6, 7]), PciDmaStatus::Ok);
        assert_eq!(&mem.0.lock()[target..target + 8], &expected);
    }
    let records = Arc::new(Mutex::new(Vec::new()));
    let out = records.clone();
    c.set_dma_sender(Arc::new(move |p, b| {
        out.lock().push((p, b.to_vec()));
        true
    }));
    c.set_wide_slot_mask(15, 1 << 3);
    c.set_dma_64bit(15, 3, true);
    f.write32((reg::CONTROL + 4) as u64, 8);
    f.write32((reg::DIRECT_MAP + 4) as u64, 10 << 20);
    f.write32((reg::DEVICE_BASE + 3 * 8 + 4) as u64, 1 << 19);
    assert_eq!(
        c.dma(15, 3)
            .dma_write(0x80002000, &[0, 1, 2, 3, 4, 5, 6, 7]),
        PciDmaStatus::Ok
    );
    let r = records.lock();
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].0.source_widget, 8);
    assert_eq!(r[0].0.slot, 3);
    assert_eq!(r[0].0.widget, 10);
    assert_eq!(r[0].0.address, 0x2000);
    assert!(r[0].0.half_write_requested);
    assert_eq!(r[0].1, [0, 1, 2, 3, 4, 5, 6, 7]);
    drop(r);
    c.set_dma_sender(Arc::new(|p, b| {
        if p.address >= 0x2008 {
            return false;
        }
        b.fill(0x42);
        true
    }));
    let mut bytes = [0xff; 16];
    assert_eq!(
        c.dma(15, 0).dma_read(0x80002000, &mut bytes),
        PciDmaStatus::MasterAbort
    );
    assert_eq!(
        bytes,
        [0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0, 0, 0, 0, 0, 0, 0, 0]
    );
}
/// A PCI function may inspect XBridge from its link-reset callback.
#[test]
fn link_reset_callback_resets_a_function_and_can_reenter_the_chip() {
    use crate::dev::isp1040::{mbox, reg as isp_reg, CTRL_INT_ENABLE, CTRL_RISC_INT_ENABLE, HOST_CMD_SHIFT};
    use crate::dev::isp12160::Isp12160;

    let (c, _) = chip();
    let isp = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus()));
    c.attach(15, 1, isp.clone());
    let (probe, line) = crate::irq::LevelProbe::line();
    c.set_pci_irq(15, 1, 0, line);
    isp.config().write(1, 6);
    isp.config().write(5, 0x02000000);
    isp.bar_write(1, isp_reg::BUS_CTRL, u32::from(CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE), 2);
    isp.bar_write(1, isp_reg::MAILBOX_BASE, u32::from(mbox::NOP), 2);
    isp.bar_write(1, isp_reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT, 2);
    assert!(probe.level());
    assert_eq!(isp.bar_read(1, isp_reg::MAILBOX_BASE, 2), u32::from(mbox::COMPLETE));

    let weak = Arc::downgrade(&c);
    let reset_isp = isp.clone();
    c.set_reset_output(Arc::new(move |port| {
        assert_eq!(port, 15);
        let chip = weak.upgrade().unwrap();
        // Callbacks own attached engines and may inspect or rewire the chip.
        assert_eq!(chip.head(port).read32((reg::DIRECT_MAP + 4) as u64).data, 0x20000);
        assert_eq!(chip.crossbar().read32((xbow_reg::IDENTIFICATION + 4) as u64).data, 0x2d000001);
        assert_eq!(reset_isp.config().command(), 0);
        reset_isp.power_on();
        chip.set_reset_output(Arc::new(|_| {}));
    }));
    c.head(15).write32((reg::DIRECT_MAP + 4) as u64, 0xa00000);
    c.crossbar().write32((xbow_reg::LINK_BASE + 7 * xbow_reg::LINK_STRIDE + xbow_reg::LINK_RESET + 4) as u64, 1);
    assert!(!probe.level());
    assert_eq!(isp.config().bar_base(1), 0);
    assert_eq!(isp.bar_read(1, isp_reg::BUS_CTRL, 2), 0);
    assert_eq!(isp.bar_read(1, isp_reg::BUS_STATUS, 2), 0);
    assert_eq!(isp.bar_read(1, isp_reg::MAILBOX_BASE, 2), 0);
    assert_eq!(isp.bar_read(1, isp_reg::MAILBOX_BASE + 2, 2), 0x4953);
}

/// Only admitted guest ports receive warm-reset notifications after both heads reset.
#[test]
fn warm_reset_notifies_only_admitted_ports_after_head_state_is_reset() {
    let (c, _) = chip();
    let ports = Arc::new(Mutex::new(Vec::new()));
    let output = ports.clone();
    let weak = Arc::downgrade(&c);
    c.set_reset_output(Arc::new(move |port| {
        if port >= 14 {
            assert_eq!(weak.upgrade().unwrap().head(port).read32((reg::DIRECT_MAP + 4) as u64).data, 0x20000);
        }
        output.lock().push(port);
    }));
    let x = c.crossbar();
    x.write32((xbow_reg::FENCE_BASE + 8 + 6 * 8 + 4) as u64, 0xff & !(1 << 2));
    c.head(14).write32((reg::DIRECT_MAP + 4) as u64, 0xa00000);
    c.head(15).write32((reg::DIRECT_MAP + 4) as u64, 0xa00000);
    c.warm_reset(10, 0);
    assert_eq!(*ports.lock(), vec![8, 9, 10, 11, 12, 13, 15]);
    assert_eq!(c.head(14).read32((reg::DIRECT_MAP + 4) as u64).data, 0xa00000);
    ports.lock().clear();
    c.warm_reset(10, 10239);
    assert!(ports.lock().is_empty());
    c.warm_reset(10, 10240);
    assert_eq!(*ports.lock(), vec![8, 9, 10, 11, 12, 13, 14, 15]);
}
/// The crossbar reports the first error once and retains its programmed destination.
#[test]
fn crossbar_errors_deliver_once_then_clear_the_original_destination() {
    let (c, _) = chip();
    let x = c.crossbar();
    let packets = Arc::new(Mutex::new(Vec::new()));
    let out = packets.clone();
    c.set_interrupt_sender(Arc::new(move |p| {
        out.lock().push(p);
        true
    }));
    x.write32((xbow_reg::INT_UPPER + 4) as u64, 0x550a0000);
    x.write32((xbow_reg::INT_LOWER + 4) as u64, 0x80);
    x.write32((xbow_reg::CONTROL + 4) as u64, 0x20);
    x.read64(xbow_reg::IDENTIFICATION as u64);
    x.write32((xbow_reg::IDENTIFICATION + 4) as u64, 1);
    assert_eq!(packets.lock().len(), 1);
    assert_eq!(x.read32((xbow_reg::STATUS_CLEAR + 4) as u64).data & ((1 << 23) | 32), (1 << 23) | 32);
    let p = packets.lock();
    assert_eq!(p.len(), 2);
    assert_eq!(p[0].vector, 0x55);
    assert_eq!(p[0].widget, 10);
    assert_eq!(p[0].address, 0x80);
    assert!(p[0].level);
    assert!(!p[1].level);
}

struct Card {
    config: crate::pci::PciConfig,
    bytes: Mutex<Vec<u8>>,
    irq: Mutex<Option<IrqLine>>,
}
impl Card {
    fn new() -> Arc<Self> {
        Self::with_status_events(0)
    }
    fn with_status_events(mask: u16) -> Arc<Self> {
        Self::with_bar(crate::pci::Bar::Mem(32), mask)
    }
    fn with_bar(bar: crate::pci::Bar, mask: u16) -> Arc<Self> {
        use crate::pci::{Bar, PciConfig};
        let size = match bar { Bar::Mem(size) | Bar::Io(size) => size as usize, Bar::Unused => 0 };
        Arc::new(Self {
            config: PciConfig::new_with_policy(
                0x1234,
                0x5678,
                0,
                0,
                [
                    bar,
                    Bar::Unused,
                    Bar::Unused,
                    Bar::Unused,
                    Bar::Unused,
                    Bar::Unused,
                ],
                1,
                crate::pci::PciConfigPolicy { status_w1c_mask: mask, ..Default::default() },
            ),
            bytes: Mutex::new((0..size).map(|i| i as u8).collect()),
            irq: Mutex::new(None),
        })
    }
}
impl PciBusDevice for Card {
    fn config(&self) -> &crate::pci::PciConfig {
        &self.config
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
    fn connect_interrupt(&self, line: IrqLine) {
        *self.irq.lock() = Some(line);
    }
}
/// The guest can identify the first invalid head access and its error destination.
#[test]
fn invalid_register_captures_first_address_and_reports_error_destination() {
    let (c, _) = chip();
    let f = c.head(15);
    let packets = Arc::new(Mutex::new(Vec::new()));
    let out = packets.clone();
    c.set_interrupt_sender(Arc::new(move |p| {
        out.lock().push(p);
        true
    }));
    f.write32((reg::INT_DEST_UPPER + 4) as u64, 0xa0000);
    f.write32((reg::INT_DEST_LOWER + 4) as u64, 0x80);
    f.write32((reg::INTERRUPT_ERROR_HOST + 4) as u64, 73);
    f.write32((reg::INTERRUPT_ENABLE + 4) as u64, 1 << 24);
    f.read32(0x1000004);
    f.read32(0x2000004);
    assert_eq!(f.read32((reg::ERR_LOWER + 4) as u64).data, 0x1000004);
    assert_eq!(packets.lock().len(), 1);
    assert_eq!(packets.lock()[0].vector, 73);
    f.write32((reg::INTERRUPT_RESET + 4) as u64, 8);
    assert_eq!(f.read32((reg::INTERRUPT_STATUS + 4) as u64).data & (1 << 24), 0);
}
/// Narrow guest configuration writes preserve status bits outside the written lanes.
#[test]
fn partial_configuration_writes_preserve_unwritten_status_events() {
    let (c, _) = chip();
    let f = c.head(15);
    let card = Card::with_status_events(0xc000);
    c.attach(15, 0, card.clone());
    for (bus, expected) in [(0, 0x56781234), (1, u32::MAX)] {
        f.write32((reg::PCI_TYPE1_CONFIG + 4) as u64, bus << 16);
        assert_eq!(f.read32(0x28000).data, expected);
    }
    card.config.set_status_bits(0xc000);
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
    card.config.set_status_bits(0xc000);
    assert_eq!(f.write32(0x20004, 0x8000_0002), BUS_OK);
    assert_eq!(f.read32(0x20004).data, 0x4200_0002);
}

fn assert_pio_matches_slow(head: &XBridgeHead, card: &Card, address: u64, mask: usize) {
    for size in [1, 2, 4, 8] {
        for offset in (0..32).step_by(size) {
            let at = address + offset as u64;
            let before = card.bytes.lock().clone();
            let expected = (0..size).fold(0u64, |v, i| (v << 8) | before[(offset + i) ^ mask] as u64);
            assert_eq!(head.access(at, size, None), expected, "read {at:#x} width {size}");
            assert_eq!(head.slow_pio(at, size, None), expected);
            head.access(at, size, Some(0x0123456789abcdef));
            let fast = card.bytes.lock().clone();
            *card.bytes.lock() = before.clone();
            head.slow_pio(at, size, Some(0x0123456789abcdef));
            assert_eq!(*card.bytes.lock(), fast, "write {at:#x} width {size}");
            *card.bytes.lock() = before;
        }
    }
}

/// Native BAR dispatch preserves bridge data lanes at every supported access width.
#[test]
fn native_bar_pages_preserve_all_pio_lanes() {
    for memory in [false, true] {
        let (chip, _) = chip();
        let head = chip.pci_head(15);
        let bar = if memory { crate::pci::Bar::Mem(0x2000) } else { crate::pci::Bar::Io(0x2000) };
        let card = Card::with_bar(bar, 0);
        // Device window 0 reaches a function owned by slot 3.
        chip.attach(15, 3, card.clone());
        head.write32(0x23010, 0x02000000 | u32::from(!memory));
        head.write32(0x23004, if memory { 2 } else { 1 });
        for wide in [false, true] {
            chip.set_dma_64bit(15, 3, wide);
            for stream in [false, true] {
                head.write32((reg::DEVICE_BASE + 4) as u64, 0x20 | if memory { 1 << 12 } else { 0 }
                    | if wide { 1 << 22 } else { 0 } | if stream { 1 << 13 } else { 0 });
                assert!(head.pio_map.lock().runs.contains(&(0x200000, 0x202000)));
                assert_pio_matches_slow(head, &card, 0x200000, if stream { 0 } else if wide { 7 } else { 3 });
                for swap in [false, true] {
                    head.write32((reg::CONTROL + 4) as u64, if swap { 1 << if memory { 22 } else { 23 } } else { 0 });
                    let aliases = if memory {
                        vec![(0x42000000, false), (0x82000000, wide)]
                    } else { vec![(0x102000000, false)] };
                    for (address, wide) in aliases {
                        assert_pio_matches_slow(head, &card, address, if swap { 0 } else if wide { 7 } else { 3 });
                    }
                }
            }
        }
    }
}

/// A moved or reset BAR stops reaching its former endpoint.
#[test]
fn bar_moves_disable_and_reset_revoke_native_pages_without_new_endpoints() {
    let (chip, _) = chip();
    let head = chip.pci_head(15);
    let card = Card::with_bar(crate::pci::Bar::Mem(0x1000), 0);
    let old_ref = Arc::downgrade(&card);
    chip.attach(15, 0, card);
    head.write32(0x20010, 0x02000000);
    head.write32(0x20004, 2);
    head.write32((reg::DEVICE_BASE + 4) as u64, 0x1020);
    let original: Vec<_> = head.pio_map.lock().entries.iter()
        .map(|(key, entry)| (*key, entry.device.clone())).collect();
    let stale = head.pio_map.lock().entries[&(8, 0, 0)].device.clone();
    head.write32((reg::DEVICE_BASE + 4) as u64, 0x1022);
    assert_eq!(head.read32(0x200000).data, u32::MAX);
    head.write32((reg::DEVICE_BASE + 4) as u64, 0x1020);
    assert_eq!(head.read32(0x200000).data, 0x03020100);
    for i in 0..128 {
        let base = 0x02000000 + i * 0x1000;
        head.write32(0x20010, base);
        assert_eq!(head.read32(0x40000000 + base as u64).data, 0x03020100);
        if i != 0 {
            assert_eq!(head.read32(0x40000000 + base as u64 - 0x1000).data, u32::MAX);
        }
    }
    assert_eq!(stale.read32(0x42000000).data, u32::MAX);
    let entries = head.pio_map.lock();
    assert_eq!(entries.entries.len(), original.len());
    for (key, device) in &original { assert!(Arc::ptr_eq(device, &entries.entries[key].device)); }
    drop(entries);
    head.write8(0x20007, 0);
    assert!(head.pio_map.lock().runs.is_empty());
    assert_eq!(head.read32(0x4207f000).data, u32::MAX);
    head.write8(0x20007, 2);
    assert!(!head.pio_map.lock().runs.is_empty());
    chip.crossbar().write32((xbow_reg::LINK_BASE + 7 * xbow_reg::LINK_STRIDE + xbow_reg::LINK_RESET + 4) as u64, 1);
    assert!(head.pio_map.lock().runs.is_empty());
    assert_eq!(head.read32(0x4207f000).data, u32::MAX);
    head.write32(0x20010, 0x02000000);
    head.write32(0x20004, 2);
    let stale = head.pio_map.lock().entries[&(8, 0, 0)].device.clone();
    let replacement = Card::with_bar(crate::pci::Bar::Mem(0x1000), 0);
    replacement.bytes.lock().fill(0x5a);
    chip.attach(15, 0, replacement);
    assert_eq!(head.read32(0x42000000).data, u32::MAX);
    assert_eq!(stale.read32(0x42000000).data, u32::MAX);
    stale.write32(0x42000000, 0xdeadbeef);
    assert_eq!(old_ref.upgrade().unwrap().bytes.lock()[..4], [0, 1, 2, 3]);
    head.write32(0x20010, 0x02000000);
    head.write32(0x20004, 2);
    assert_eq!(head.read32(0x42000000).data, 0x5a5a5a5a);
}

/// A BAR page shared with another mapping must retain the slow decoder.
#[test]
fn partial_or_overlapping_bar_pages_use_the_slow_decoder() {
    let (chip, _) = chip();
    let head = chip.pci_head(15);
    let small = Card::new();
    let large = Card::with_bar(crate::pci::Bar::Mem(0x2000), 0);
    small.bytes.lock().fill(0xa5);
    chip.attach(15, 0, small);
    chip.attach(15, 1, large.clone());
    for slot in 0..2 {
        head.write32(0x20010 + slot * 0x1000, 0x02000000);
        head.write32(0x20004 + slot * 0x1000, 2);
    }
    head.write32((reg::DEVICE_BASE + 4) as u64, 0x1020);
    assert!(head.pio_map.lock().runs.is_empty());
    assert_eq!(head.read32(0x42000000).data, 0xa5a5a5a5);
    assert_eq!(head.read32(0x42000020).data, 0x23222120);
    head.write32(0x20004, 0);
    assert!(!head.pio_map.lock().runs.is_empty());
    assert_pio_matches_slow(head, &large, 0x42000000, 3);
    head.write32(0x21004, 0);
    head.write32(0x20004, 2);
    assert!(head.pio_map.lock().runs.is_empty());
    assert_eq!(head.read32(0x42000000).data, 0xa5a5a5a5);
}

/// BAR holes and cross-boundary cycles must use the original slow decoder,
/// including after the host publishes a replacement set of BAR runs.
#[test]
fn sparse_bar_lookup_rejects_holes_and_cross_boundary_cycles() {
    let decode = PioDecode::default();
    let first: Arc<dyn BusDevice> = Arc::new(Memory(Mutex::new(vec![0; 0x4000])));
    let second: Arc<dyn BusDevice> = Arc::new(Memory(Mutex::new(vec![0; 0x4000])));
    drop(decode.publish(vec![
        BarRun { start: 0x3000, end: 0x4000, dev: second.clone() },
        BarRun { start: 0x1000, end: 0x2000, dev: first.clone() },
    ]));
    assert!(Arc::ptr_eq(&decode.target(0x1000, 8).unwrap(), &first));
    assert!(Arc::ptr_eq(&decode.target(0x1ff8, 8).unwrap(), &first));
    assert!(decode.target(0x1ffc, 8).is_none());
    assert!(decode.target(0x2000, 8).is_none());
    assert!(decode.target(0x2ff8, 8).is_none());
    assert!(Arc::ptr_eq(&decode.target(0x3000, 8).unwrap(), &second));
    assert!(decode.target(0x4000, 8).is_none());
    assert!(decode.target(u64::MAX, 8).is_none());
    drop(decode.publish(vec![BarRun { start: 0x2000, end: 0x3000, dev: first.clone() }]));
    assert!(decode.target(0x1000, 8).is_none());
    assert!(Arc::ptr_eq(&decode.target(0x2000, 8).unwrap(), &first));
    assert!(decode.target(0x3000, 8).is_none());
}

fn xbow_read(x: &Arc<dyn BusDevice>, off: usize) -> u32 {
    x.read32((off + 4) as u64).data
}

/// Software probing the crossbar after reset sees these values. Each
/// link's control word resets to 0x80000; that value belongs in the control
/// word, not in the link reset register, which no read can see.
#[test]
fn the_crossbar_resets_to_its_documented_values() {
    let (c, _) = chip();
    let x = c.crossbar();
    for (off, value) in [
        (xbow_reg::ERR_UPPER, 0),
        (xbow_reg::ERR_LOWER, 0),
        (xbow_reg::CONTROL, 0x80),
        (xbow_reg::PACKET_TIMEOUT, 0xfffff),
        (xbow_reg::INT_UPPER, 0x80000),
        (xbow_reg::INT_LOWER, 0),
        (xbow_reg::ERR_CMD, 0),
        (xbow_reg::LLP_CONTROL, 0x03ff1810),
        (xbow_reg::ARB_RELOAD, 2),
        (xbow_reg::PERF_A, 0),
        (xbow_reg::PERF_B, 0),
        (xbow_reg::NIC, 2),
        (xbow_reg::FENCE_BASE, 0xff),
        (xbow_reg::LOCK, 0),
    ] {
        assert_eq!(xbow_read(&x, off), value, "{off:#x}");
    }
    for link in 0..8 {
        let base = xbow_reg::LINK_BASE + link * xbow_reg::LINK_STRIDE;
        assert_eq!(xbow_read(&x, xbow_reg::FENCE_BASE + 8 + link * 8), 0xff);
        assert_eq!(xbow_read(&x, base + xbow_reg::LINK_CONTROL), 0x80000, "link {}", link + 8);
        assert_eq!(xbow_read(&x, base + xbow_reg::LINK_ARB_UPPER), 0);
        assert_eq!(xbow_read(&x, base + xbow_reg::LINK_ARB_LOWER), 0);
    }
}

/// Software shares the crossbar between processors with the resource lock:
/// a write takes it only while it is free, so a second holder is refused
/// until the first writes the clear register.
#[test]
fn the_resource_lock_admits_one_holder_until_cleared() {
    let (c, _) = chip();
    let x = c.crossbar();
    let other = c.crossbar_for_port(HEAD_E);
    x.write32((xbow_reg::LOCK + 4) as u64, 0x5a);
    assert_eq!(xbow_read(&x, xbow_reg::LOCK), 0x5a);
    other.write32((xbow_reg::LOCK + 4) as u64, 0x3c);
    assert_eq!(xbow_read(&x, xbow_reg::LOCK), 0x5a);
    assert_eq!(xbow_read(&x, xbow_reg::LOCK_CLEAR), 0x5a);
    x.write32((xbow_reg::LOCK_CLEAR + 4) as u64, 0);
    assert_eq!(xbow_read(&x, xbow_reg::LOCK), 0);
    other.write32((xbow_reg::LOCK + 4) as u64, 0x3c);
    assert_eq!(xbow_read(&x, xbow_reg::LOCK), 0x3c);
}

/// Every reset fence bit is read and write, a link's own bit included, and
/// only the low eight bits exist.
#[test]
fn reset_fences_keep_their_low_eight_bits() {
    let (c, _) = chip();
    let x = c.crossbar();
    for off in (xbow_reg::FENCE_BASE..=xbow_reg::FENCE_LAST).step_by(8) {
        x.write32((off + 4) as u64, 0);
        assert_eq!(xbow_read(&x, off), 0);
        x.write32((off + 4) as u64, off as u32);
        assert_eq!(xbow_read(&x, off), off as u32 & 0xff);
        x.write32((off + 4) as u64, u32::MAX);
        assert_eq!(xbow_read(&x, off), 0xff);
    }
}
