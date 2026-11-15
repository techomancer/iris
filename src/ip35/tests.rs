use super::*;
use crate::ip35_machine::tests::{fuel_config, tezro_config, with_machine, Head};
use crate::ip35_machine::Ip35Machine;
use crate::test_temp::TempPath;
use crate::dev::bedrock::regs as bedrock_reg;
use crate::physical::Physical;
use crate::sn_nodes::UncachedBus;
use crate::traits::*;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// A bus whose attribute `n` space is `views[n]`, with nothing elsewhere.
fn bus_of(views: [SparseWindowMap; 4]) -> UncachedBus {
    UncachedBus::new(views.map(|map| Arc::new(map) as Arc<dyn BusDevice>))
}

/// The Fuel's processor on `bus`, its cache filling from `cached`, as the machine builds it.
fn fuel_cpu(
    bus: Arc<UncachedBus>,
    cached: Arc<dyn BusDevice>,
) -> Box<
    crate::cpu::mips_exec::MipsExecutor<
        crate::cpu::mips_tlb::MipsTlb,
        crate::cpu::mips_cache_shadow::R12000ShadowCache,
    >,
> {
    let extent = crate::dev::bedrock::RamExtent::new(&[256, 256]);
    let timers = Arc::new(crate::hptimer::TimerManager::new());
    crate::cpu::build::build_cpu(
        bus,
        cached,
        Arc::new(memory_map(&extent)),
        &crate::cpu::parts::R14000_500_SYSCLK_200,
        &timers,
        crate::cpu::parts::R14000_500_SYSCLK_200.part.count_hz(),
        0,
        false,
    )
}
/// [`fuel_cpu`] on the machine's node.
fn machine_cpu(
    m: &Ip35Machine,
) -> Box<
    crate::cpu::mips_exec::MipsExecutor<
        crate::cpu::mips_tlb::MipsTlb,
        crate::cpu::mips_cache_shadow::R12000ShadowCache,
    >,
> {
    fuel_cpu(m.nodes[0].bus.clone(), m.nodes[0].chip.decode.physical().clone())
}

/// A Bedrock unit's registers, at its base in the node's register window.
struct Unit(Arc<dyn BusDevice>, u64);
impl Unit {
    fn read64(&self, a: u64) -> BusRead64 {
        self.0.read64(self.1 + a)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.0.write64(self.1 + a, v)
    }
}
fn md_regs(m: &Ip35Machine) -> Unit {
    Unit(m.nodes[0].chip.registers(), bedrock_reg::md::BASE)
}
fn pi_regs(m: &Ip35Machine, interface: u64) -> Unit {
    Unit(m.nodes[0].chip.registers(), interface * bedrock_reg::PI1_BASE)
}

/// A processor access to widget `widget` through Bedrock's small window.
fn pio_widget(widget: u64, offset: u64) -> u64 {
    (widget << 24) | offset
}

struct Word(u64);
impl BusDevice for Word {
    fn read64(&self, _: u64) -> BusRead64 {
        BusRead64::ok(self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum NarrowWriteCall {
    Byte(u64, u8),
    Half(u64, u16),
    From(usize, u64, u64),
}
/// Records every narrow write it sees. With `sees_register` it overrides
/// `write_narrow_from` and keeps the whole source register, as Bedrock's ECC
/// view does; otherwise it takes the trait's default.
#[derive(Default)]
struct NarrowWriteSpy {
    sees_register: bool,
    calls: std::sync::Mutex<Vec<NarrowWriteCall>>,
}
impl NarrowWriteSpy {
    fn new(sees_register: bool) -> Arc<Self> {
        Arc::new(Self { sees_register, ..Default::default() })
    }
    fn calls(&self) -> Vec<NarrowWriteCall> {
        self.calls.lock().unwrap().clone()
    }
}
impl BusDevice for NarrowWriteSpy {
    fn write8(&self, address: u64, value: u8) -> u32 {
        self.calls.lock().unwrap().push(NarrowWriteCall::Byte(address, value));
        BUS_OK
    }
    fn write16(&self, address: u64, value: u16) -> u32 {
        self.calls.lock().unwrap().push(NarrowWriteCall::Half(address, value));
        BUS_OK
    }
    fn write_narrow_from(&self, address: u64, size: usize, gpr: u64) -> u32 {
        if !self.sees_register {
            return if size == 1 { self.write8(address, gpr as u8) } else { self.write16(address, gpr as u16) };
        }
        self.calls.lock().unwrap().push(NarrowWriteCall::From(size, address, gpr));
        BUS_OK
    }
}

/// An uncached SB or SH reaches a device with the whole source register, at the
/// device-local address; only a device that overrides the call sees more than
/// the low bits, and a window outside every map is a bus error.
#[test]
fn the_fuel_runs_the_r14000_at_revision_2_3() {
    use crate::cpu::mips_cache_shadow::R12000ShadowCache;
    use crate::cpu::mips_cache_v2::CpuModel as Model;
    use crate::cpu::parts;
    use crate::ip34_machine::CPUS;
    assert_eq!(CPUS, &[&parts::R14000_500]);
    let part = CPUS[0];
    assert_eq!((part.mhz, part.rev, part.l2_kb), (500, (2, 3), 2048));
    assert_eq!(part.prid(), 0x0f23);
    assert_eq!(part.count_hz(), 250_000_000);
    // `with_cpu!` builds the type `FuelCpu` names for this part.
    let built = crate::with_cpu!(part, true, false, C => std::any::type_name::<C>());
    assert_eq!(built, std::any::type_name::<R12000ShadowCache>());
    // Typed R12000, it takes the R10000 family's behaviour: reserved
    // secondary-tag bits read as zero, and a refused uncached store posts.
    assert!(<R12000ShadowCache as Model>::R10K_FAMILY);
    assert_ne!(<R12000ShadowCache as Model>::SECONDARY_TAG_MASK, u64::MAX);
    assert!(<R12000ShadowCache as Model>::POSTS_UNCACHED_STORES);
    assert!(<R12000ShadowCache as Model>::PARTIAL_LWR_KEEPS_UPPER);
    // The executor carries the chip's PRId.
    // The executor is large; the default test stack is not.
    crate::display_stop_tests::big_stack(|| {
        let cpu = fuel_cpu(Arc::new(bus_of(Default::default())), Physical::new(Vec::new()));
        assert_eq!(cpu.core.cp0_prid, 0x0f23);
    });
}

/// The Fuel's flash record and its processor module are two statements of one mode word,
/// which the PROM's early check compares; the processor resets with the module's Config.
#[test]
fn the_fuels_flash_mode_word_matches_its_cpu_module() {
    let dir = TempPath::dir("fuel-mode");
    let specs = crate::ip34_machine::specs(&fuel_config(&dir)).unwrap();
    let spec = &specs[0];
    assert_eq!(spec.firmware.cpu_mode_word, spec.module.mode_word);
    assert_eq!(*spec.module.part, *crate::ip34_machine::CPUS[0]);
    assert_eq!(u64::from(spec.firmware.cpu_clock_hz), spec.module.clock_hz());
}

/// The flash record has to agree with the board-id word's speed row; the PROM reports a mismatch
/// otherwise.
#[test]
fn the_fuels_board_word_is_class_4_with_the_500_mhz_row() {
    let dir = TempPath::dir("fuel-board-word");
    with_machine(fuel_config(&dir), |m| {
        let word = m.nodes[0].bus.read64(0x1400_0000);
        assert_eq!(word.status, BUS_OK);
        assert_eq!(word.data, 0x4004);
        assert_eq!(word.data, crate::ip34_machine::FUEL_BOARD_ID);
    });
}

#[test]
fn narrow_stores_carry_the_register_to_devices_that_ask_for_it() {
    let plain = NarrowWriteSpy::new(false);
    let ecc = NarrowWriteSpy::new(true);
    let mut views: [SparseWindowMap; 4] = Default::default();
    views[3].map(0x1000, 0x1000, plain.clone());
    views[3].map(0x2000, 0x1000, ecc.clone());
    let bus = bus_of(views);

    let source = 0x1122_3344_5566_77a5u64;
    let uc = 3u64 << 57;
    assert_eq!(bus.write_narrow_from(uc | 0x1003, 1, source), BUS_OK);
    assert_eq!(bus.write_narrow_from(uc | 0x1004, 2, source), BUS_OK);
    assert_eq!(
        plain.calls(),
        vec![NarrowWriteCall::Byte(3, source as u8), NarrowWriteCall::Half(4, source as u16)],
        "an ordinary window gets the architectural low bits",
    );

    assert_eq!(bus.write_narrow_from(uc | 0x2004, 2, source), BUS_OK);
    assert_eq!(ecc.calls(), vec![NarrowWriteCall::From(2, 4, source)]);

    assert_eq!(bus.write_narrow_from(uc | 0x8000, 2, source), BUS_ERR, "no window there");
}

/// Equal physical offsets in different XKPHYS access spaces reach different devices.
#[test]
fn uc_spaces_keep_equal_offsets_distinct() {
    let mut views: [SparseWindowMap; 4] = Default::default();
    for (space, view) in views.iter_mut().enumerate() {
        view.map(0x0100_0000, 0x1000, Arc::new(Word(0x100 + space as u64)));
    }
    let bus = bus_of(views);
    for space in 0..4 {
        let value = bus.read64(((space as u64) << 57) | 0x0100_0000);
        assert_eq!(value.status, BUS_OK);
        assert_eq!(value.data, 0x100 + space as u64);
    }
}

/// The processor must retain both XKPHYS access-space bits through translation.
#[test]
fn xkphys_uc_bits_survive_the_board_cpu_mask() {
    // MipsExecutor and the TLB's inline arrays exceed the test worker's stack.
    std::thread::Builder::new()
        .stack_size(4 * 1024 * 1024)
        .spawn(|| {
            let bus = Arc::new(bus_of(Default::default()));
            let mut cpu = fuel_cpu(bus, Physical::new(Vec::new()));
            cpu.core.write_cp0(
                12,
                (cpu.core.cp0_status | crate::cpu::mips_core::STATUS_KX) as u64,
            );
            for raw_uc in 0..4u64 {
                let va = 0x9000_0000_0100_0020 | (raw_uc << 57);
                for _ in 0..2 {
                    let translated = cpu.debug_translate(va);
                    assert_eq!(translated.status, crate::cpu::mips_exec::TR_UNCACHED);
                    assert_eq!(translated.phys, (raw_uc << 57) | 0x0100_0020);
                }
            }
        })
        .expect("spawn")
        .join()
        .expect("thread panicked");
}

/// RAM placed above the first GiB must remain shared by cached, uncached and DMA paths.
#[test]
fn cached_and_uncached_memory_keep_high_addresses() {
    let dir = TempPath::dir("fuel-high");
    with_machine(fuel_config(&dir), |m| {
        assert_eq!(md_regs(m).write64(8, 0x33), BUS_OK);
        let (chip, bus) = (&m.nodes[0].chip, &m.nodes[0].bus);
        let at = (1 << 30) + 0x1000;
        assert_eq!(chip.decode.physical().write64(at, 0x12345678), BUS_OK);
        assert_eq!((chip.decode.physical().clone() as Arc<dyn BusDevice>).read64(at).data, 0x12345678);
        assert_eq!(bus.read64((3 << 57) | at).data, 0x12345678);
        assert_eq!(bus.read64(at).status, BUS_ERR);
        assert_eq!(chip.decode.physical().read64(at).data, 0x12345678);
    });
}

/// A processor store through the cached view or an uncached one lands in
/// the bank the slot holds and advances that bank's page counter, whichever
/// bank sits there after a swap, and not the other bank's.
#[cfg(feature = "jitv2")]
#[test]
fn processor_stores_advance_the_counter_of_the_bank_they_reach() {
    let dir = TempPath::dir("fuel-counter");
    with_machine(fuel_config(&dir), |m| {
        let (chip, bus) = (&m.nodes[0].chip, &m.nodes[0].bus);
        let banks = chip.decode.physical().banks();
        let counter = |bank: usize| unsafe { (*banks[bank].gen_ptr(0x2000)).load(std::sync::atomic::Ordering::Relaxed) };
        let (zero, one) = (0x2000, (1 << 30) + 0x2000);
        // Bit 32 swaps the banks of a pair: slot 0 now reaches bank 1.
        assert_eq!(md_regs(m).write64(8, 0x33 | (1 << 32)), BUS_OK);
        let before = [counter(0), counter(1)];
        assert_eq!((chip.decode.physical().clone() as Arc<dyn BusDevice>).write64(zero, 0x44), BUS_OK);
        assert_eq!(banks[1].read64(0x2000).data, 0x44, "slot 0 reaches bank 1");
        assert_eq!(banks[0].read64(0x2000).data, 0);
        assert!(counter(1) > before[1], "a cached store to slot 0 counts in bank 1");
        assert_eq!(counter(0), before[0]);
        assert_eq!(bus.write64((3 << 57) | one, 0x55), BUS_OK);
        assert_eq!(banks[0].read64(0x2000).data, 0x55, "slot 1 reaches bank 0");
        assert!(counter(0) > before[0], "an uncached store to slot 1 counts in bank 0");
    });
}

/// Firmware resets the PCI link before reinitializing the attached storage controller.
#[test]
fn fuel_link_reset_resets_the_attached_scsi_engine() {
    use crate::dev::isp1040::{mbox, reg, CTRL_INT_ENABLE, CTRL_RISC_INT_ENABLE, HOST_CMD_SHIFT};
    let dir = TempPath::dir("fuel-link-reset");
    with_machine(fuel_config(&dir), |m| {
        let pio = m.nodes[0].chip.pio();
        let head = Head::new(pio.clone(), 15);
        // The ISP12160 is device 1 of head F: its memory BAR is the second.
        let bar = |head: &Head| head.map_bar(1, 0x14, 0x0200_0000);
        let win = bar(&head);
        win.write16(u64::from(reg::BUS_CTRL), CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
        win.write16(u64::from(reg::MAILBOX_BASE), mbox::NOP);
        win.write16(u64::from(reg::HOST_CMD), 0x5 << HOST_CMD_SHIFT);
        assert_eq!(win.read16(u64::from(reg::MAILBOX_BASE)), u32::from(mbox::COMPLETE));
        assert_ne!(win.read16(u64::from(reg::BUS_CTRL)), 0);
        // Link 15's reset register in widget 0's crossbar.
        assert_eq!(pio.write32(pio_widget(0, 0x2f4), 1), BUS_OK);
        assert_eq!(head.cfg_read(1, 4) & 0xffff, 0, "the command register is cleared");
        assert_eq!(head.cfg_read(1, 0x14), 0, "and so is the BAR");
        let win = bar(&head);
        assert_eq!(win.read16(u64::from(reg::BUS_CTRL)), 0);
        assert_eq!(win.read16(u64::from(reg::MAILBOX_BASE)), 0);
        assert_eq!(win.read16(u64::from(reg::MAILBOX_BASE) + 2), 0x4953);
    });
}

/// The PROM reads the station address from the L1; it must report the configured one.
#[test]
fn the_configured_station_address_reaches_the_l1() {
    let dir = TempPath::dir("fuel-mac");
    let mut cfg = fuel_config(&dir);
    cfg.network.mac = Some("08:00:69:21:43:65".into());
    with_machine(cfg, |m| assert_eq!(m.l1s[0].station_mac(), [0x08, 0x00, 0x69, 0x21, 0x43, 0x65]));
}

/// The L1's reported revision and brick serial keep their defaults unless the config replaces them.
#[test]
fn the_l1_identity_overrides_reach_the_controller() {
    let dir = TempPath::dir("fuel-l1-identity");
    let stock = with_machine(fuel_config(&dir), |m| m.l1s[0].config());
    assert_eq!(stock.revision, Some([1, 48, 1]));
    assert_eq!(stock.brick_serial.as_deref(), Some("EMUN00"));
    let mut cfg = fuel_config(&dir);
    cfg.fuel.l1_revision = Some("2.3.4".into());
    cfg.fuel.brick_serial = Some("EMUN09".into());
    let set = with_machine(cfg, |m| m.l1s[0].config());
    assert_eq!(set.revision, Some([2, 3, 4]));
    assert_eq!(set.brick_serial.as_deref(), Some("EMUN09"));
}

/// The Fuel's OHCI root ports each carry a low-speed device only when the
/// USB pair is configured, and then the host's input, which still reaches
/// the IOC3's PS/2 controller object, goes on to them.
#[test]
fn the_fuel_plugs_a_usb_keyboard_and_mouse_only_when_configured_on() {
    let dir = TempPath::dir("fuel-usb");
    // The OHCI controller is device 5 of head F; its root ports' status registers are at 0x54
    // and 0x58 of its first BAR, and a port is powered by writing 0x100 there.
    let ports = |usb_input| {
        let mut cfg = fuel_config(&dir);
        cfg.fuel.usb_input = usb_input;
        with_machine(cfg, |m| {
            let win = Head::new(m.nodes[0].chip.pio(), 15).map_bar(5, 0x10, 0x0300_0000);
            win.write32(0x54, 0x100);
            win.write32(0x58, 0x100);
            let status = [0x54, 0x58].map(|at| win.read32(at));
            let ps2 = m.io.ps2.clone();
            // The guest turns the PS/2 mouse on.
            ps2.write_to(0xf4, true);
            (status, ps2.mouse_enabled())
        })
    };
    let (status, mouse) = ports(true);
    for s in status {
        assert_eq!(s & 1, 1, "device connected");
        assert_eq!(s & 0x200, 0x200, "low speed");
    }
    assert!(!mouse, "the USB mouse answers for the host's pointer");
    let (status, mouse) = ports(false);
    assert_eq!(status.map(|s| s & 1), [0, 0], "nothing plugged in");
    assert!(mouse, "without the USB pair the PS/2 mouse does");
}

/// Run `body` on a Fuel with its first bank pair enabled and head F's widget number set, with
/// the XBridge whose DMA views stand in for a card's master, which the guest cannot reach.
fn on_fuel_dma(body: impl FnOnce(&Ip35Machine, &crate::dev::xbridge::XBridge) + Send + 'static) {
    let dir = TempPath::dir("fuel-dma");
    with_machine(fuel_config(&dir), |m| {
        assert_eq!(md_regs(m).write64(8, 0x33), BUS_OK);
        assert_eq!(m.nodes[0].chip.pio().write32(pio_widget(15, 0x24), 15), BUS_OK);
        body(m, m.io.xbridge.as_ref().expect("the Fuel's XBridge"));
    });
}

fn fuel_dma_address(address: u64) -> u64 {
    (10 << 60) | (1 << 55) | address
}

/// Storage DMA buffers can begin unaligned and cross a physical page boundary.
#[test]
fn dma_preserves_byte_order_and_crosses_ragged_pages() {
    use crate::pci::PciDmaStatus;
    on_fuel_dma(|m, xbridge| {
        let dma = xbridge.dma(15, 0);
        let memory = m.nodes[0].chip.decode.physical();
        for (address, length) in [(0x1030, 8), (0x1103, 27), (0x1ffb, 19)] {
            let data: Vec<u8> = (0..length).map(|i| 0x31 + i as u8).collect();
            assert_eq!(dma.dma_write(fuel_dma_address(address), &data), PciDmaStatus::Ok);
            for (i, &byte) in data.iter().enumerate() {
                assert_eq!(memory.read8(address + i as u64).data, byte);
            }
            let mut back = vec![0xff; length];
            assert_eq!(dma.dma_read(fuel_dma_address(address), &mut back), PciDmaStatus::Ok);
            assert_eq!(back, data);
        }
        // Without byte swapping enabled, the bridge reverses each 32-bit word.
        assert_eq!(dma.dma_write((10 << 60) | 0x3000, &[0, 1, 2, 3, 4, 5, 6, 7]), PciDmaStatus::Ok);
        assert_eq!(memory.read64(0x3000).data, 0x0302010007060504);
    });
}

/// A failed DMA read must clear only refused chunks and preserve completed data.
#[test]
fn dma_refusals_clear_failed_reads_without_losing_completed_chunks() {
    use crate::pci::PciDmaStatus;
    on_fuel_dma(|m, xbridge| {
        let dma = xbridge.dma(15, 0);
        let memory = m.nodes[0].chip.decode.physical();
        // Bank zero has 256 MiB fitted. The second qword is outside its backing.
        let last = 0x0fff_fff8;
        assert_eq!(memory.write64(last, 0x0102030405060708), BUS_OK);
        let mut bytes = [0xff; 16];
        assert_eq!(dma.dma_read(fuel_dma_address(last), &mut bytes), PciDmaStatus::MasterAbort);
        assert_eq!(bytes, [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(dma.dma_write(fuel_dma_address(last), &[0x42; 16]), PciDmaStatus::MasterAbort);
        assert_eq!(memory.read64(last).data, 0x4242424242424242);
    });
}

/// Inbound Hub writes must preserve access width and the widget's permissions.
#[test]
fn dma_hub_access_preserves_width_and_source_permissions() {
    use crate::pci::PciDmaStatus;
    on_fuel_dma(|m, xbridge| {
        let dma = xbridge.dma(15, 0);
        let hub = (1 << 47) | 0x1800000;
        for (interface, offset) in [(0, 0), (1, 0x200000)] {
            let address = fuel_dma_address(hub + offset + 0x90);
            assert_eq!(dma.dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::Ok);
            assert_ne!(pi_regs(m, interface as u64).read64(0x98).data & (1 << 12), 0);
            for width in [1, 2, 4] {
                assert_eq!(dma.dma_write(address, &[0; 8][..width]), PciDmaStatus::MasterAbort);
                assert_ne!(pi_regs(m, interface as u64).read64(0x98).data & (1 << 12), 0);
            }
            assert_eq!(dma.dma_write(address, &12u64.to_be_bytes()), PciDmaStatus::Ok);
            assert_eq!(pi_regs(m, interface as u64).read64(0x98).data & (1 << 12), 0);
        }
        let address = fuel_dma_address(hub + 0x90);
        // I/O modify must retain PI region protection and the widget's permissions.
        assert_eq!(pi_regs(m, 0).write64(0x10, 0), BUS_OK);
        assert_eq!(dma.dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::MasterAbort);
        assert_eq!(pi_regs(m, 0).read64(0x98).data & (1 << 12), 0);
        assert_eq!(pi_regs(m, 0).write64(0x10, 1), BUS_OK);
        assert_eq!(xbridge.dma(15, 1).dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::Ok);
        assert_ne!(pi_regs(m, 0).read64(0x98).data & (1 << 12), 0);
        let mut bytes = [0xff; 8];
        assert_eq!(xbridge.dma(15, 1).dma_read(fuel_dma_address(hub + 0x40), &mut bytes), PciDmaStatus::MasterAbort);
        assert_eq!(bytes, [0; 8]);
        assert_eq!(m.nodes[0].bus.read64((1 << 57) | 0x1800040).status, BUS_OK);
    });
}

/// Rounded MMIO pages must not expose padding or bypass cell permissions.
#[test]
fn exact_windows_reject_padding_and_revoked_permissions() {
    use std::sync::atomic::{AtomicBool, Ordering};
    struct Offset;
    impl BusDevice for Offset {
        fn read64(&self, offset: u64) -> BusRead64 { BusRead64::ok(offset) }
        fn write64(&self, _offset: u64, _value: u64) -> u32 { BUS_OK }
    }
    let permitted = Arc::new(AtomicBool::new(true));
    let permission = permitted.clone();
    let window = Window::new(0x1000, 0x88, Arc::new(Offset))
        .permitted(Arc::new(move || permission.load(Ordering::Acquire)));
    assert_eq!(window.read64(0x1080).data, 0x80);
    assert_eq!(window.read64(0x1088).status, BUS_ERR);
    assert_eq!(window.read64(0x1084).status, BUS_ERR);
    assert_eq!(window.read64(0x0ff8).status, BUS_ERR);
    assert_eq!(window.read64(u64::MAX).status, BUS_ERR);
    permitted.store(false, Ordering::Release);
    assert_eq!(window.read64(0x1000).status, BUS_ERR);
    assert_eq!(window.write64(0x1000, 1), BUS_ERR);
    permitted.store(true, Ordering::Release);
    assert_eq!(window.write64(0x1000, 1), BUS_OK);
}

/// Board default setup must preserve the direct RAM path and its shared DMA contents.
#[test]
fn direct_cached_banks_remain_visible_after_board_defaults_are_installed() {
    let dir = TempPath::dir("fuel-direct");
    with_machine(fuel_config(&dir), |m| {
        assert_eq!(md_regs(m).write64(8, 0x33), BUS_OK);
        let cached: Arc<dyn BusDevice> = m.nodes[0].chip.decode.physical().clone();
        assert!(cached.mem_ptr(0x2000).is_some());
        assert_eq!(cached.write64(0x2000, 0x123456789abcdef0), BUS_OK);
        assert_eq!(m.nodes[0].chip.decode.physical().read64(0x2000).data, 0x123456789abcdef0);
    });
}

/// A device overlay replaces its pages, including padding, while its neighbors
/// keep their original address translation.
#[test]
fn sparse_overlays_keep_neighbors_and_refuse_the_new_windows_padding() {
    let mut map = SparseWindowMap::default();
    map.map(0x1000, 0x4000, Arc::new(Word(1)));
    map.map(0x2040, 0x80, Arc::new(Word(2)));
    assert_eq!(map.read64(0x1000).data, 1);
    assert_eq!(map.read64(0x2040).data, 2);
    assert_eq!(map.read64(0x20b8).data, 2);
    assert_eq!(map.read64(0x20bc).status, BUS_ERR);
    assert_eq!(map.read64(0x2000).status, BUS_ERR);
    assert_eq!(map.read64(0x20c0).status, BUS_ERR);
    assert_eq!(map.read64(0x3000).data, 1);
    assert_eq!(map.read64(0x5000).status, BUS_ERR);
    map.map(0x1000, 0x4000, Arc::new(Word(3)));
    assert_eq!(map.ranges.len(), 1);
    assert_eq!(map.read64(0x2040).data, 3);
}

/// Sparse MMIO holes remain bus errors and a revoked cell permission does not
/// fall through to an older mapping beneath the protected window.
#[test]
fn sparse_holes_and_revoked_permissions_never_expose_older_devices() {
    use std::sync::atomic::{AtomicBool, Ordering};
    let permission = Arc::new(AtomicBool::new(true));
    let allowed = permission.clone();
    let mut map = SparseWindowMap::default();
    map.map(0, 0x3000, Arc::new(Word(1)));
    map.map(0x1000, 8, Arc::new(Window::new(0, 8, Arc::new(Word(2)))
        .permitted(Arc::new(move || allowed.load(Ordering::Acquire)))));
    assert_eq!(map.read64(0x1000).data, 2);
    permission.store(false, Ordering::Release);
    assert_eq!(map.read64(0x1000).status, BUS_ERR);
    assert_eq!(map.read64(0x1008).status, BUS_ERR);
    assert_eq!(map.read64(0x2000).data, 1);
    assert_eq!(map.read64(0x3000).status, BUS_ERR);
}

/// The JIT's page lookup and inline compile read code through the CPU's cached
/// bus, which a board whose uncached connection decodes plain addresses as
/// something else sets to the view the cache fills from.
#[test]
fn the_cpu_reads_code_through_the_cached_view_not_the_uncached_connection() {
    let dir = TempPath::dir("fuel-cached-bus");
    with_machine(fuel_config(&dir), |m| {
        let cpu = machine_cpu(m);
        let cached: Arc<dyn BusDevice> = m.nodes[0].chip.decode.physical().clone();
        assert!(Arc::ptr_eq(cpu.cached_bus(), &cached));
        let sysad: Arc<dyn BusDevice> = m.nodes[0].bus.clone();
        assert!(!Arc::ptr_eq(cpu.cached_bus(), &sysad), "sysad is the uncached connection");
    });
}

/// A bank configuration write remaps memory, and the processor's decoded
/// instruction pages are keyed on the map they were filled under: the key must
/// move with the map, or the cache keeps pages from the old one.
#[test]
fn a_bank_configuration_write_moves_the_processors_decode_cache_key() {
    let dir = TempPath::dir("fuel-decode-key");
    with_machine(fuel_config(&dir), |m| {
        let cpu: Arc<dyn crate::cpu::mips_exec::CpuDevice> = Arc::new(crate::cpu::mips_exec::MipsCpu::<_, crate::cpu::mips_cache_shadow::R12000ShadowCache>::new(
            *machine_cpu(m),
        ));
        node::connect_cpus("Fuel", &m.nodes[0].chip, &m.l1s[0], &std::sync::mpsc::channel().0, std::slice::from_ref(&cpu)).unwrap();
        let key = || unsafe { (*(cpu.tcache_key_ptr() as *const std::sync::atomic::AtomicU64)).load(std::sync::atomic::Ordering::Relaxed) };
        let before = key();
        assert_eq!(md_regs(m).write64(8, 0x33), BUS_OK);
        let one_bank = key();
        assert_ne!(one_bank, before, "the key stayed put across a remap");
        assert_eq!(md_regs(m).write64(8, 0x03), BUS_OK);
        assert_ne!(key(), one_bank, "the key stayed put across a second remap");
    });
}

/// An IP35 PROM stores a dirty primary tag for its stack line before MD is configured;
/// the node must give the cache that view of the window or the store bus-errors.
#[test]
fn the_proms_cached_stack_holds_its_page_before_memory_is_configured() {
    use crate::cpu::mips_isa::{CACH_PD, C_IST, OP_CACHE};
    const SP: u64 = 0x10_3ff0;
    // mtc0 t2, TagLo; cache Index_Store_Tag (primary data), 0(sp); spin.
    let words = [(0x10 << 26) | (4 << 21) | (10 << 16) | (28 << 11), (OP_CACHE << 26) | (29 << 21) | ((C_IST | CACH_PD) << 16), 0x1000_ffff, 0];
    let payload: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
    let dir = TempPath::dir("fuel-cached-stack");
    let mut cfg = fuel_config(&dir);
    std::fs::write(dir.join("prom.bin"), payload).unwrap();
    cfg.prom = dir.join("prom.bin").to_string_lossy().into_owned();
    with_machine(cfg, |m| {
        let mut exec = machine_cpu(m);
        exec.core.pc = 0xffff_ffff_bfc0_0000;
        exec.core.write_gpr(29, 0xa800_0000_0000_0000 | SP);
        // A DirtyExclusive primary tag for the stack's page.
        exec.core.write_gpr(10, (SP >> 12) << 8 | 3 << 6);
        let status = exec.core.cp0_status | crate::cpu::mips_core::STATUS_KX;
        exec.core.write_cp0(12, status as u64);
        let cpu: Arc<dyn crate::cpu::mips_exec::CpuDevice> = Arc::new(crate::cpu::mips_exec::MipsCpu::<_, crate::cpu::mips_cache_shadow::R12000ShadowCache>::new(*exec));
        node::connect_cpus("Fuel", &m.nodes[0].chip, &m.l1s[0], &std::sync::mpsc::channel().0, std::slice::from_ref(&cpu)).unwrap();
        cpu.step_n_inline(2).unwrap();
        let page = SP & !(crate::ppmem::granularity() as u64 - 1);
        // The machine's own processor holds view 0; this one is the second.
        assert_eq!(m.nodes[0].chip.decode.physical().ppmem_space().unwrap().held_line_pages(1), [page], "the cache holds the stack's page");
    });
}

/// Unmodeled Hub heads retain the PROM compatibility readback without
/// absorbing the adjacent, independently implemented L1 UART window.
#[test]
fn compatibility_hub_heads_do_not_hide_the_l1_uart_window() {
    let dir = TempPath::dir("fuel-hub-heads");
    with_machine(fuel_config(&dir), |m| {
        let (chip, bus) = (&m.nodes[0].chip, &m.nodes[0].bus);
        assert_ne!(chip.md.cell_permits() as u64, 0);
        for base in [0x1000_0000, 0x3000_0000] {
            for offset in [0, 8, 0x40, 0x78] {
                let address = base + offset;
                assert_eq!(bus.write8(address, 0xa5), BUS_OK);
                assert_eq!(bus.write16(address, 0xa5a5), BUS_OK);
                assert_eq!(bus.write32(address, 0xa5a5a5a5), BUS_OK);
                assert_eq!(bus.write64(address, 0x2000), BUS_OK);
                let value = bus.read8(address);
                assert_eq!(value.status, BUS_OK);
                assert_eq!(value.data, 0);
                let value = bus.read16(address);
                assert_eq!(value.status, BUS_OK);
                assert_eq!(value.data, 0);
                let value = bus.read32(address);
                assert_eq!(value.status, BUS_OK);
                assert_eq!(value.data, 0);
                let value = bus.read64(address);
                assert_eq!(value.status, BUS_OK);
                assert_eq!(value.data, 0);
            }
            assert_eq!(bus.read64(base + 0x80).status, BUS_OK);
        }
    });
}

/// A busy host port or unusable transcript path must not prevent a serial
/// machine from starting, including when any of its three ports serves CI.
#[test]
fn unavailable_serial_ports_and_logs_do_not_abort_the_machine() {
    use crate::config::Origin3200Console;
    let listeners: Vec<_> = (0..3).map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap()).collect();
    let ports: Vec<_> = listeners.iter().map(|socket| socket.local_addr().unwrap().port()).collect();
    let dir = TempPath::dir("fuel-busy-serial");
    let missing = dir.join("absent").join("serial.txt");
    assert!(!missing.parent().unwrap().exists(), "the log parent must be absent");
    for (ci, console) in [
        (true, Origin3200Console::L1),
        (true, Origin3200Console::Ioc3A),
        (true, Origin3200Console::Ioc3B),
        (false, Origin3200Console::L1),
    ] {
        let mut cfg = fuel_config(&dir);
        cfg.fuel.l1_port = Some(ports[0]);
        cfg.serial_port_a = Some(ports[1]);
        cfg.serial_port_b = Some(ports[2]);
        cfg.ci = ci;
        cfg.serial_log = Some(missing.to_string_lossy().into_owned());
        cfg.fuel.l1_log = cfg.serial_log.clone();
        cfg.fuel.console = console;
        with_machine(cfg, |_| ());
    }
}

/// A Fuel with `graphics` in its graphics slot.
fn fuel_with_graphics(dir: &TempPath, graphics: crate::config::GraphicsBoard) -> crate::config::MachineConfig {
    let mut cfg = fuel_config(dir);
    cfg.graphics.board = (graphics != crate::config::GraphicsBoard::None).then_some(graphics);
    cfg
}

/// The VPro's link status in the crossbar, in its slot's data half.
const VPRO_LINK_STATUS: u64 = 0x100 + 5 * 0x40 + 0x10 + 4;
/// A processor access to widget 13 through Bedrock's small window.
const VPRO_PIO: u64 = 13 << 24;

// The VPro cases follow tests/qtest/sn1-fuel-vpro-test.c in
// jamesbraid/qemu-sgi-origami: widget-identity, profile-identity-and-routing
// and headless-fuel-has-no-board.

// VPro host-interface registers the tests below touch, as byte offsets in
// the board's widget space. The board's own constants are private to it.
const ODSY_IDENTITY: u64 = 0x00;
const ODSY_CONTROL: u64 = 0x20;
const ODSY_CONTROL_WIDGET_MASK: u32 = 0xf;
const ODSY_INT_DEST_LOWER: u64 = 0x38;
const ODSY_INTR_CLEAR: u64 = 0x101c;
const ODSY_INTR_ENABLE_SET: u64 = 0x1034;
const ODSY_INTR_FIFO_LOW_WATER: u32 = 0x200;
const ODSY_ROUTE_DEST_UPPER: u64 = 0x1044;
const ODSY_ROUTE_DEST_LOWER: u64 = 0x104c;
const ODSY_ROUTE_VECTORS: u64 = 0x1054;
const ODSY_ROUTE_DEST_WIDGET_SHIFT: u32 = 25;

/// A Fuel IRIX inventory lists its VPro under xtalk/13: the processor must
/// reach the board there, learn its widget number from it, and find the link alive in the crossbar.
#[test]
fn the_vpro_answers_at_widget_13_with_its_link_alive() {
    let dir = TempPath::dir("fuel-vpro-id");
    for (graphics, revision) in [(crate::config::GraphicsBoard::V12, 0xa), (crate::config::GraphicsBoard::V10, 0xa)] {
        with_machine(fuel_with_graphics(&dir, graphics), move |m| {
            let pio = m.nodes[0].chip.pio();
            let identity = pio.read32(VPRO_PIO | (ODSY_IDENTITY + 4));
            assert_eq!(identity.status, BUS_OK);
            assert_eq!(identity.data as u32, (revision << 28) | 0x0c01_3047, "{graphics:?}");
            let control = pio.read32(VPRO_PIO | (ODSY_CONTROL + 4)).data as u32;
            assert_eq!(control & ODSY_CONTROL_WIDGET_MASK, 13);
            assert_ne!(pio.read32(pio_widget(0, VPRO_LINK_STATUS)).data & (1 << 31), 0);
        });
    }
}

/// The monitor's `vpro` commands resolve on a Fuel or a Tezro that has a VPro, and are unknown on
/// one that has none.
#[test]
fn the_monitor_resolves_vpro_commands_only_with_a_vpro() {
    let dir = TempPath::dir("vpro-monitor");
    for (what, plain) in [("Fuel", fuel_config(&dir)), ("Tezro", tezro_config(&dir, 1))] {
        let mut v12 = plain.clone();
        v12.graphics.board = Some(crate::config::GraphicsBoard::V12);
        for (cfg, vpro) in [(v12, true), (plain, false)] {
            let seen = with_machine(cfg, |m| {
                let has = |command: &str| m.io.monitor.iter().any(|d| d.register_commands().iter().any(|(c, _)| c == command));
                (has("vpro"), has("ps2"))
            });
            assert_eq!(seen, (vpro, true), "{what}: `vpro` only with a V12; the other board commands stay");
        }
    }
}

/// The board's interrupt packets cross the XBridge to Bedrock as the PCI
/// heads' do: a raised source sets its vector's pending bit in the
/// processor interface the destination names, and a cleared one clears it.
#[test]
fn vpro_interrupt_packets_set_and_clear_a_processor_interface_pending_bit() {
    let dir = TempPath::dir("fuel-vpro-irq");
    with_machine(fuel_with_graphics(&dir, crate::config::GraphicsBoard::V12), |m| {
        // Register space (bit 47), the node's hub window and PI0's pending
        // modify register, at the Bedrock's crossbar port 0xa.
        let packet = (1u64 << 47) | 0x180_0000 | 0x90;
        let vector = 5;
        let pio = m.nodes[0].chip.pio();
        let write = |offset: u64, value: u64| {
            assert_eq!(pio.write32(VPRO_PIO | offset, value as u32), BUS_OK, "{offset:#x}");
        };
        write(ODSY_ROUTE_DEST_UPPER, (0xa << ODSY_ROUTE_DEST_WIDGET_SHIFT) | (packet >> 32));
        write(ODSY_ROUTE_DEST_LOWER, packet & 0xffff_ffff);
        write(ODSY_ROUTE_VECTORS, vector);
        write(ODSY_INTR_ENABLE_SET, ODSY_INTR_FIFO_LOW_WATER as u64);
        let pending = || pi_regs(m, 0).read64(0x98).data & (1 << vector);
        assert_ne!(pending(), 0);
        write(ODSY_INTR_CLEAR, ODSY_INTR_FIFO_LOW_WATER as u64);
        assert_eq!(pending(), 0);
    });
}

/// A link reset of the graphics port resets the board behind it, as the
/// PCI port's does for the functions behind its head.
#[test]
fn a_graphics_link_reset_resets_the_vpro() {
    let dir = TempPath::dir("fuel-vpro-reset");
    with_machine(fuel_with_graphics(&dir, crate::config::GraphicsBoard::V12), |m| {
        let pio = m.nodes[0].chip.pio();
        let destination = VPRO_PIO | (ODSY_INT_DEST_LOWER + 4);
        assert_eq!(pio.write32(destination, 0x55), BUS_OK);
        assert_eq!(pio.read32(destination).data, 0x55);
        let link_reset = 0x100 + 5 * 0x40 + 0x30 + 4;
        assert_eq!(pio.write32(pio_widget(0, link_reset), 1), BUS_OK);
        assert_eq!(pio.read32(destination).data, 0);
    });
}

/// Without graphics nothing answers at widget 13 and its link is down, as on a headless Fuel.
#[test]
fn a_fuel_without_graphics_has_no_widget_13() {
    let dir = TempPath::dir("fuel-no-vpro");
    with_machine(fuel_with_graphics(&dir, crate::config::GraphicsBoard::None), |m| {
        let pio = m.nodes[0].chip.pio();
        assert!(m.io.display.is_none());
        assert_ne!(pio.read32(VPRO_PIO | 4).status, BUS_OK);
        assert_eq!(pio.read32(pio_widget(0, VPRO_LINK_STATUS)).data & (1 << 31), 0);
    });
}

/// IRIX's VPro driver writes each processor interface's graphics credit
/// counter at attach and tears the board down if it reads back zero; the
/// registers must keep what the guest writes, per interface.
#[test]
fn processor_interface_graphics_registers_keep_their_writes() {
    let dir = TempPath::dir("fuel-pi-graphics");
    with_machine(fuel_with_graphics(&dir, crate::config::GraphicsBoard::V12), |m| {
        // Slot A's credit counter and bias; PI1 keeps its own block.
        assert_eq!(pi_regs(m, 0).write64(0x308, 100), BUS_OK);
        assert_eq!(pi_regs(m, 0).write64(0x310, 0x1000 | 68), BUS_OK);
        assert_eq!(pi_regs(m, 0).read64(0x308).data, 100);
        assert_eq!(pi_regs(m, 0).read64(0x310).data, 68);
        assert_eq!(pi_regs(m, 1).read64(0x308).data, 0);
        m.nodes[0].chip.power_on();
        assert_eq!(pi_regs(m, 0).read64(0x308).data, 0);
    });
}

/// The JIT keys a code page by frame and whether it was fetched uncached: with RAM over
/// the reset window, cached and uncached fetches of 0x1fc00000 read different bytes.
#[cfg(feature = "jitv2")]
mod code_pages {
    use super::*;
    use crate::display_stop_tests::big_stack;
    use crate::cpu::jitv2::jitv2::CompileRequest;
    use crate::cpu::mips_isa::{JIT_REGION_BOUNDARY_SENTINEL, OP_ADDIU};

    struct Words(u32, AtomicU64);
    impl BusDevice for Words {
        fn read32(&self, _: u64) -> BusRead32 {
            BusRead32::ok(self.0)
        }
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::ok((self.0 as u64) << 32 | self.0 as u64)
        }
        fn gen_ptr(&self, _: u64) -> *const AtomicU64 {
            &self.1
        }
    }

    const RESET: u64 = 0x1fc0_0000;
    const ADDIU_R1: u32 = (OP_ADDIU << 26) | (1 << 16) | 1;
    type Cpu = crate::cpu::mips_exec::MipsExecutor<crate::cpu::mips_tlb::MipsTlb, crate::cpu::mips_cache_shadow::R16000ShadowCache>;

    /// The PROM (an ADDIU) in every uncached view of the reset frame and RAM
    /// (a region boundary, which compiles to nothing) in the cached one.
    fn rig() -> (Arc<Words>, Arc<Words>, Box<Cpu>) {
        use crate::cpu::mips_exec::{MipsCpuConfig, MipsExecutor};
        let rom = Arc::new(Words(ADDIU_R1, AtomicU64::new(0)));
        let ram = Arc::new(Words(JIT_REGION_BOUNDARY_SENTINEL, AtomicU64::new(0)));
        let physical = Physical::new(Vec::new());
        physical.set_default(Arc::new(crate::sn_nodes::UnfittedMemory(crate::devlog::LogModule::Bedrock)), "absent");
        map_device(&physical, RESET, 0x1000, ram.clone(), "RAM over the reset window");
        let mut views: [SparseWindowMap; 4] = Default::default();
        for view in &mut views {
            view.map(RESET, 0x1000, rom.clone());
        }
        views[0].map(0x4_0000_0000, 0x1000, rom.clone());
        let bus = bus_of(views);
        let cfg = MipsCpuConfig::for_model::<crate::cpu::mips_cache_shadow::R16000ShadowCache>(Arc::new(memory_map(&RamExtent::new(&[1024]))))
            .with_prid(crate::cpu::parts::R16000_700_EARLY.prid());
        let mut cpu = Box::new(MipsExecutor::new(Arc::new(bus) as Arc<dyn BusDevice>, crate::cpu::mips_tlb::MipsTlb::new(cfg.tlb_entries), &cfg));
        cpu.phys_mask = CPU_PHYS_MASK;
        cpu.core.cp0_config = (cpu.core.cp0_config & !7) | 3;
        cpu.set_cached_bus(physical);
        (rom, ram, cpu)
    }

    fn page_at(cpu: &mut Cpu, pc: u64) -> &'static mut crate::cpu::jitv2::PhysicalCodePage {
        cpu.core.pc = pc;
        let _ = cpu.step_int();
        let page = cpu.jitv2_current_page();
        assert!(!page.is_null(), "{pc:#x} lands on a code page");
        unsafe { &mut *page }
    }

    fn compile(cpu: &Cpu, page: &mut crate::cpu::jitv2::PhysicalCodePage) {
        page.mark_requested(0);
        let req = CompileRequest { page: page as *mut _, compiled_for_fr1: true };
        let mut analyzer = crate::cpu::jitv2::analyzer::Analyzer::new();
        let mut codegen = crate::cpu::jitv2::codegen::Codegen::new();
        #[cfg(feature = "developer")]
        crate::cpu::jitv2::comp::handle_request(&req, cpu.cached_bus(), &mut analyzer, &mut codegen, &crate::cpu::jitv2::JitStats::default());
        #[cfg(not(feature = "developer"))]
        crate::cpu::jitv2::comp::handle_request(&req, cpu.cached_bus(), &mut analyzer, &mut codegen);
    }

    #[test]
    fn a_cached_page_and_an_uncached_fetch_of_one_frame_compile_apart() {
        big_stack(|| {
            let (rom, ram, mut cpu) = rig();
            // The reset vector in KSEG1, then the same frame in KSEG0.
            let prom = page_at(&mut cpu, 0xffff_ffff_bfc0_0000);
            let cached = page_at(&mut cpu, 0xffff_ffff_9fc0_0000);
            assert_eq!((prom.pfn, cached.pfn), (0x1fc00, 0x1fc00));
            assert_ne!(prom as *mut _, cached as *mut _, "the two fetches key two pages");
            compile(&cpu, prom);
            compile(&cpu, cached);
            assert!(prom.is_runnable(0), "the uncached page compiles the PROM's ADDIU");
            assert!(!cached.is_runnable(0), "the cached page compiles RAM's boundary word to nothing");
            ram.1.fetch_add(1, std::sync::atomic::Ordering::Release);
            assert!(prom.is_runnable(0), "a write to RAM leaves the PROM's code");
            rom.1.fetch_add(1, std::sync::atomic::Ordering::Release);
            assert!(!prom.is_runnable(0), "a write to the PROM invalidates its code");
        });
    }

    fn wait_runnable(page: &crate::cpu::jitv2::PhysicalCodePage) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while std::time::Instant::now() < deadline {
            if page.is_runnable(0) { return true; }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    }

    /// The compile pool, restarted after a page-pool flush, reads a page
    /// fetched uncached through the processor's bus, not its own.
    #[test]
    fn a_pool_restart_after_a_flush_still_compiles_an_uncached_page() {
        big_stack(|| {
            let (_rom, _ram, mut cpu) = rig();
            {
                let jit = cpu.jitv2.clone();
                let mut g = jit.lock();
                let stats = g.stats.clone();
                g.compile_queue.set_thread_count(1);
                g.compile_queue.start(cpu.cached_bus().clone(), stats);
            }
            cpu.jitv2_flush_from_here(false);
            let page = page_at(&mut cpu, 0xffff_ffff_bfc0_0000);
            assert!(page.uncached.is_some());
            page.mark_requested(0);
            let jit = cpu.jitv2.clone();
            let g = jit.lock();
            let stats = g.stats.clone();
            assert!(g.compile_queue.send(CompileRequest { page: page as *mut _, compiled_for_fr1: true }, &stats));
            drop(g);
            assert!(wait_runnable(page), "the restarted pool must compile the PROM's page");
            jit.lock().compile_queue.stop();
        });
    }

    /// Uncached attributes 1 to 3 and anything past the end of memory stay in the interpreter.
    #[test]
    fn attributes_one_to_three_and_high_addresses_get_no_code_page() {
        big_stack(|| {
            for pc in [
                0x9000_0000_0000_0000u64 | RESET,           // attribute 0: a page
                0x9200_0000_0000_0000u64 | RESET,           // attribute 1
                0x9400_0000_0000_0000u64 | RESET,           // attribute 2
                0x9600_0000_0000_0000u64 | RESET,           // attribute 3
                0x9000_0004_0000_0000u64,                   // past the end of memory, attribute 0
            ] {
                // A fresh executor each time: a tracked page would survive a
                // fetch that wrongly kept the same frame.
                let (_rom, _ram, mut cpu) = rig();
                // KX, so XKPHYS addresses translate.
                let status = cpu.core.cp0_status | 0x80;
                cpu.set_cp0_status(status);
                cpu.core.pc = pc;
                let _ = cpu.step_int();
                let page = cpu.jitv2_current_page();
                if pc == 0x9000_0000_0000_0000u64 | RESET {
                    assert!(!page.is_null(), "attribute 0 below 4 GiB is compiled");
                } else {
                    assert!(page.is_null(), "{pc:#x} must not get a code page");
                }
            }
        });
    }
}
