use super::*;
use crate::dev::bedrock::reg as bedrock_reg;
use crate::traits::{BusDevice, BUS_OK};

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
fn md_regs(board: &crate::ip34_machine::FuelBoard) -> Unit {
    Unit(board.bedrock.registers(), bedrock_reg::MD_BASE)
}
fn pi_regs(board: &crate::ip34_machine::FuelBoard, interface: u64) -> Unit {
    Unit(board.bedrock.registers(), interface * bedrock_reg::PI1_BASE)
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
/// The Fuel's processor is the R14000 at revision 2.3 and 500 MHz, built as
/// the R12000 type with its PRId set over it; the Octane's own part is
/// revision 2.4.
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
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let cpu = crate::ip34_machine::build_cpu(Arc::new(Ip35Bus::new(Physical::new(Vec::new()))), crate::ip34_machine::CPUS[0]);
            assert_eq!(cpu.core.cp0_prid, 0x0f23);
        })
        .unwrap()
        .join()
        .expect("thread panicked");
}

/// The board-id word's low nibble selects the PROM's speed-table row, and the
/// flash record has to agree with that row's clock; the PROM reports a
/// mismatch otherwise. Read through the processor's bus at the local
/// `0x9000000014000000`.
#[test]
fn the_board_word_names_the_speed_row_of_the_fuels_processor() {
    use crate::cpu::parts;
    use crate::ip34_machine::{board_word, CPUS, FUEL_BOARD_ID, SPEED_ROWS};
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let word = board.bus.read64(0x1400_0000);
    assert_eq!(word.status, BUS_OK);
    assert_eq!(word.data, FUEL_BOARD_ID);
    for part in CPUS {
        assert_eq!(board_word(part), Some(word.data), "{}", part.name());
        let row = SPEED_ROWS.iter().find(|r| r.nibble == word.data & 0xf).expect("a row for the nibble");
        assert_eq!((row.mhz, row.l2_kb), (part.mhz, part.l2_kb));
        assert_eq!(u64::from(crate::dev::bedrock_flash::FirmwareRecord::FUEL.cpu_clock_hz), u64::from(part.mhz) * 1_000_000);
    }
    // Class 4 is the Fuel; the nibble 4 is the 500 MHz, 2 MB module.
    assert_eq!(word.data >> 12 & 0xf, 4);
    assert_eq!(word.data, 0x4004);
}

/// The PROM's table, row by row (read from the PROM's code), and the nibbles
/// with no row.
#[test]
fn the_speed_table_maps_each_established_row_and_refuses_the_rest() {
    use crate::cpu::parts::{Part, R14000_500};
    use crate::ip34_machine::{board_word, SPEED_ROWS};
    let part = |mhz, l2_kb| Part { mhz, l2_kb, ..R14000_500 };
    let rows = [(500, 2048, 0x4004), (600, 4096, 0x4007), (700, 4096, 0x4003), (800, 4096, 0x400c), (900, 8192, 0x4002)];
    for (mhz, l2_kb, word) in rows {
        assert_eq!(board_word(&part(mhz, l2_kb)), Some(word), "{mhz} MHz, {l2_kb} KB");
    }
    // A clock with the wrong cache, or a clock off the table, has no row.
    assert_eq!(board_word(&part(500, 4096)), None);
    assert_eq!(board_word(&part(600, 2048)), None);
    assert_eq!(board_word(&part(400, 2048)), None);
    // The nibbles the PROM reports as "Unknown PIMM_PSC type". Others are
    // simply not in the table here: the PROM's table was read, not these.
    for nibble in [0x0, 0x5, 0xf] {
        assert!(SPEED_ROWS.iter().all(|r| r.nibble != nibble), "{nibble:#x}");
    }
}

#[test]
fn narrow_stores_carry_the_register_to_devices_that_ask_for_it() {
    let mut bus = Ip35Bus::new(Physical::new(Vec::new()));
    let plain = NarrowWriteSpy::new(false);
    let ecc = NarrowWriteSpy::new(true);
    bus.map(3, 0x1000, 0x1000, plain.clone());
    bus.map(3, 0x2000, 0x1000, ecc.clone());

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
    let mut bus = Ip35Bus::new(Physical::new(Vec::new()));
    for space in 0..4 {
        bus.map(
            space,
            0x0100_0000,
            0x1000,
            Arc::new(Word(0x100 + space as u64)),
        );
    }
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
            let bus = Arc::new(Ip35Bus::new(Physical::new(Vec::new())));
            let mut cpu = crate::ip34_machine::build_cpu(bus, crate::ip34_machine::CPUS[0]);
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
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_eq!(md_regs(&board).write64(8, 0x33), BUS_OK);
    let at = (1 << 30) + 0x1000;
    assert_eq!(board.bedrock.dma_view().write64(at, 0x12345678), BUS_OK);
    assert_eq!(board.cached_bus().read64(at).data, 0x12345678);
    assert_eq!(board.bus.read64((3 << 57) | at).data, 0x12345678);
    assert_eq!(board.bus.read64(at).status, BUS_ERR);
    assert_eq!(board.bedrock.dma_view().read64(at).data, 0x12345678);
}

/// A processor store through the cached view or an uncached one lands in
/// the bank the slot holds and advances that slot's logical counter,
/// whichever bank sits there after a swap, and no other slot's.
#[test]
fn processor_stores_advance_the_counter_of_the_slot_they_reach() {
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let memory = board.bedrock.dma_view();
    let counter = |_at: u64| -> u64 {
        #[cfg(feature = "jitv2")]
        return unsafe { (*memory.gen_ptr(_at)).load(std::sync::atomic::Ordering::Relaxed) };
        #[cfg(not(feature = "jitv2"))]
        0
    };
    let advanced = |what: &str, before: u64, after: u64| {
        assert!(!cfg!(feature = "jitv2") || after > before, "{what}: {before} -> {after}");
    };
    let (zero, one) = (0x2000, (1 << 30) + 0x2000);
    // Bit 32 swaps the banks of a pair: slot 0 now reaches bank 1.
    assert_eq!(md_regs(&board).write64(8, 0x33 | (1 << 32)), BUS_OK);
    let banks = board.bedrock.physical().banks();
    let (before_zero, before_one) = (counter(zero), counter(one));
    assert_eq!(board.cached_bus().write64(zero, 0x44), BUS_OK);
    assert_eq!(banks[1].read64(0x2000).data, 0x44, "slot 0 reaches bank 1");
    assert_eq!(banks[0].read64(0x2000).data, 0);
    let after_zero = counter(zero);
    advanced("cached store to slot 0", before_zero, after_zero);
    assert_eq!(counter(one), before_one);
    assert_eq!(board.bus.write64((3 << 57) | one, 0x55), BUS_OK);
    assert_eq!(banks[0].read64(0x2000).data, 0x55, "slot 1 reaches bank 0");
    let after_one = counter(one);
    advanced("uncached store to slot 1", before_one, after_one);
    assert_eq!(board.bus.write8((3 << 57) | (one + 8), 0x77), BUS_OK);
    advanced("uncached byte store", after_one, counter(one));
    assert_eq!(counter(zero), after_zero);
}

/// Firmware resets the PCI link before reinitializing the attached storage controller.
#[test]
fn fuel_link_reset_resets_the_attached_scsi_engine() {
    use crate::dev::isp1040::{mbox, reg, CTRL_INT_ENABLE, CTRL_RISC_INT_ENABLE, HOST_CMD_SHIFT};
    use crate::pci::PciFunction;
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let isp = &board.scsi;
    isp.state().config().write(1, 6);
    isp.state().config().write(5, 0x02000000);
    isp.bar_write(1, reg::BUS_CTRL, u32::from(CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE), 2);
    isp.bar_write(1, reg::MAILBOX_BASE, u32::from(mbox::NOP), 2);
    isp.bar_write(1, reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT, 2);
    assert_eq!(isp.bar_read(1, reg::MAILBOX_BASE, 2), u32::from(mbox::COMPLETE));
    assert_ne!(isp.bar_read(1, reg::BUS_CTRL, 2), 0);
    assert_eq!(board.xbridge.widget(0).unwrap().write32(0x2f4, 1), BUS_OK);
    assert_eq!(isp.state().config().command(), 0);
    assert_eq!(isp.state().config().bar_base(1), 0);
    assert_eq!(isp.bar_read(1, reg::BUS_CTRL, 2), 0);
    assert_eq!(isp.bar_read(1, reg::MAILBOX_BASE, 2), 0);
    assert_eq!(isp.bar_read(1, reg::MAILBOX_BASE + 2, 2), 0x4953);
}

/// The PROM reads the station address from the L1 and IRIX from the
/// Ethernet; both must report the configured one.
#[test]
fn the_configured_station_address_reaches_the_l1_and_the_ethernet() {
    let mac = [0x08, 0x00, 0x69, 0x21, 0x43, 0x65];
    let board = crate::ip34_machine::FuelBoard::with_config(crate::ip34_machine::FuelBoardConfig {
        payload: &[], clock: Arc::new(|| 0), flash_path: None,
        station_mac: mac,
        timekeeper: Arc::new(crate::dev::timekeeper::Timekeeper::new().unwrap()),
        ps2_input: true,
        l1: Default::default(),
        graphics: crate::config::GraphicsBoard::None,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    }).unwrap();
    assert_eq!(board.l1.station_mac(), mac);
    assert_eq!(board.eth.mac(), mac);
}

/// The L1's reported revision and brick serial keep their defaults unless
/// the board is built with replacements.
#[test]
fn the_l1_identity_overrides_reach_the_controller() {
    use crate::ip34_machine::{FuelBoard, FuelBoardConfig, L1Identity};
    let build = |l1| FuelBoard::with_config(FuelBoardConfig {
        payload: &[], clock: Arc::new(|| 0), flash_path: None,
        station_mac: [8, 0, 0x69, 0x12, 0x34, 0x56],
        timekeeper: Arc::new(crate::dev::timekeeper::Timekeeper::new().unwrap()),
        ps2_input: true, l1,
        graphics: crate::config::GraphicsBoard::None,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    }).unwrap().l1.config();
    let stock = build(L1Identity::default());
    assert_eq!(stock.revision, Some([1, 48, 1]));
    assert_eq!(stock.brick_serial.as_deref(), Some("EMUN00"));
    let set = build(L1Identity { revision: Some([2, 3, 4]), brick_serial: Some("EMUN09".into()) });
    assert_eq!(set.revision, Some([2, 3, 4]));
    assert_eq!(set.brick_serial.as_deref(), Some("EMUN09"));
}

fn fuel_dma_board() -> crate::ip34_machine::FuelBoard {
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_eq!(md_regs(&board).write64(8, 0x33), BUS_OK);
    board.xbridge.widget(15).unwrap().write32(0x24, 15);
    board
}

fn fuel_dma_address(address: u64) -> u64 {
    (10 << 60) | (1 << 55) | address
}

/// Storage DMA buffers can begin unaligned and cross a physical page boundary.
#[test]
fn dma_preserves_byte_order_and_crosses_ragged_pages() {
    use crate::pci::PciDmaStatus;
    let board = fuel_dma_board();
    let dma = board.xbridge.dma(15, 0);
    let memory = board.bedrock.dma_view();
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
}

/// A failed DMA read must clear only refused chunks and preserve completed data.
#[test]
fn dma_refusals_clear_failed_reads_without_losing_completed_chunks() {
    use crate::pci::PciDmaStatus;
    let board = fuel_dma_board();
    let dma = board.xbridge.dma(15, 0);
    let memory = board.bedrock.dma_view();
    // Bank zero has 256 MiB fitted. The second qword is outside its backing.
    let last = 0x0fff_fff8;
    assert_eq!(memory.write64(last, 0x0102030405060708), BUS_OK);
    let mut bytes = [0xff; 16];
    assert_eq!(dma.dma_read(fuel_dma_address(last), &mut bytes), PciDmaStatus::MasterAbort);
    assert_eq!(bytes, [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(dma.dma_write(fuel_dma_address(last), &[0x42; 16]), PciDmaStatus::MasterAbort);
    assert_eq!(memory.read64(last).data, 0x4242424242424242);
}

/// Inbound Hub writes must preserve access width and the widget's permissions.
#[test]
fn dma_hub_access_preserves_width_and_source_permissions() {
    use crate::pci::PciDmaStatus;
    let board = fuel_dma_board();
    let dma = board.xbridge.dma(15, 0);
    let hub = (1 << 47) | 0x1800000;
    for (interface, offset) in [(0, 0), (1, 0x200000)] {
        let address = fuel_dma_address(hub + offset + 0x90);
        assert_eq!(dma.dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::Ok);
        assert_ne!(pi_regs(&board, interface as u64).read64(0x98).data & (1 << 12), 0);
        for width in [1, 2, 4] {
            assert_eq!(dma.dma_write(address, &[0; 8][..width]), PciDmaStatus::MasterAbort);
            assert_ne!(pi_regs(&board, interface as u64).read64(0x98).data & (1 << 12), 0);
        }
        assert_eq!(dma.dma_write(address, &12u64.to_be_bytes()), PciDmaStatus::Ok);
        assert_eq!(pi_regs(&board, interface as u64).read64(0x98).data & (1 << 12), 0);
    }
    let address = fuel_dma_address(hub + 0x90);
    // I/O modify must retain PI region protection and the widget's permissions.
    assert_eq!(pi_regs(&board, 0 as u64).write64(0x10, 0), BUS_OK);
    assert_eq!(dma.dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::MasterAbort);
    assert_eq!(pi_regs(&board, 0 as u64).read64(0x98).data & (1 << 12), 0);
    assert_eq!(pi_regs(&board, 0 as u64).write64(0x10, 1), BUS_OK);
    assert_eq!(board.xbridge.dma(15, 1).dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::Ok);
    assert_ne!(pi_regs(&board, 0 as u64).read64(0x98).data & (1 << 12), 0);
    let mut bytes = [0xff; 8];
    assert_eq!(board.xbridge.dma(15, 1).dma_read(
        fuel_dma_address(hub + 0x40), &mut bytes,
    ), PciDmaStatus::MasterAbort);
    assert_eq!(bytes, [0; 8]);
    assert_eq!(board.bus.read64((1 << 57) | 0x1800040).status, BUS_OK);
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
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_eq!(md_regs(&board).write64(8, 0x33), BUS_OK);
    let cached = board.cached_bus();
    assert!(cached.mem_ptr(0x2000).is_some());
    assert_eq!(cached.write64(0x2000, 0x123456789abcdef0), BUS_OK);
    assert_eq!(board.bedrock.dma_view().read64(0x2000).data, 0x123456789abcdef0);
    assert!(cached.mem_ptr(3 << 57).is_none(), "UC selectors cannot alias cached RAM");
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

/// Code-page lookup receives plain physical addresses and reads them through
/// the cached plane. Cached stores must also update the MD view.
#[cfg(feature = "jitv2")]
#[test]
fn code_generations_use_the_stable_ram_endpoint() {
    struct Generations {
        ram_only: bool,
        generation: AtomicU64,
        writes: std::sync::Mutex<Vec<(u64, u64)>>,
    }
    impl Generations {
        fn new(ram_only: bool) -> Arc<Self> {
            Arc::new(Self { ram_only, generation: AtomicU64::new(0), writes: Default::default() })
        }
    }
    impl BusDevice for Generations {
        fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
            if !self.ram_only || address < 0x1000 { &self.generation } else { std::ptr::null() }
        }
        fn write64(&self, address: u64, value: u64) -> u32 {
            self.writes.lock().unwrap().push((address, value));
            BUS_OK
        }
        fn write64_masked(&self, address: u64, value: u64, mask: u64) -> u32 {
            self.write64(address, value & mask)
        }
        fn write_block(&self, address: u64, values: &[u64]) -> u32 {
            for (index, value) in values.iter().enumerate() {
                self.write64(address + index as u64 * 8, *value);
            }
            BUS_OK
        }
    }
    let physical = Physical::new(Vec::new());
    let bus = Ip35Bus::new(physical.clone());
    let raw_ram = Generations::new(false);
    let stable_ram = Generations::new(true);
    let rom = Generations::new(false);
    map_device(&physical, 0, 0x1000, raw_ram.clone(), "raw RAM generation");
    map_device(&physical, 0x1fc00000, 0x1000, rom.clone(), "ROM generation");
    let cached = bus.cached_view(stable_ram.clone(), crate::dev::bedrock::RamExtent::new(&[256, 256]));
    assert_eq!(cached.gen_ptr(0x40), &stable_ram.generation as *const AtomicU64);
    assert_eq!(cached.gen_ptr(0x1fc00000), &rom.generation as *const AtomicU64);
    assert_eq!(cached.write64(0x40, 0x1234), BUS_OK);
    assert_eq!(cached.write_block(0x48, &[0x5678]), BUS_OK);
    assert_eq!(cached.write64_masked(0x50, 0xaabb, 0xff), BUS_OK);
    assert_eq!(*stable_ram.writes.lock().unwrap(), [(0x40, 0x1234), (0x48, 0x5678), (0x50, 0xbb)]);
    assert_eq!(cached.write64(0x1fc00000, 0x4321), BUS_OK);
    assert_eq!(*rom.writes.lock().unwrap(), [(0, 0x4321)], "the ROM window translates to a device-local offset");
    assert!(raw_ram.writes.lock().unwrap().is_empty());
}

/// The JIT's page lookup and inline compile read code through the CPU's cached
/// bus, which a board whose uncached connection decodes plain addresses as
/// something else sets to the view the cache fills from. The compile pool is
/// started on that same object.
#[test]
fn the_cpu_reads_code_through_the_cached_view_not_the_uncached_connection() {
    // An executor is large; a test thread's stack is not.
    std::thread::Builder::new().stack_size(64 << 20).spawn(cpu_cached_bus).unwrap().join().unwrap();
}

fn cpu_cached_bus() {
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let cpu = board.build_cpu(crate::ip34_machine::CPUS[0]);
    #[cfg(feature = "jitv2")]
    let cached = board.code_bus();
    #[cfg(not(feature = "jitv2"))]
    let cached = board.cached_bus();
    assert!(Arc::ptr_eq(cpu.cached_bus(), &cached));
    let sysad: Arc<dyn BusDevice> = board.bus.clone();
    assert!(!Arc::ptr_eq(cpu.cached_bus(), &sysad), "sysad is the uncached connection");
}

/// Unmodeled Hub heads retain the PROM compatibility readback without
/// absorbing the adjacent, independently implemented L1 UART window.
#[test]
fn compatibility_hub_heads_do_not_hide_the_l1_uart_window() {
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_ne!(board.bedrock.cell_permits() as u64, 0);
    for base in [0x1000_0000, 0x3000_0000] {
        for offset in [0, 8, 0x40, 0x78] {
            let address = base + offset;
            assert_eq!(board.bus.write8(address, 0xa5), BUS_OK);
            assert_eq!(board.bus.write16(address, 0xa5a5), BUS_OK);
            assert_eq!(board.bus.write32(address, 0xa5a5a5a5), BUS_OK);
            assert_eq!(board.bus.write64(address, 0x2000), BUS_OK);
            let value = board.bus.read8(address);
            assert_eq!(value.status, BUS_OK);
            assert_eq!(value.data, 0);
            let value = board.bus.read16(address);
            assert_eq!(value.status, BUS_OK);
            assert_eq!(value.data, 0);
            let value = board.bus.read32(address);
            assert_eq!(value.status, BUS_OK);
            assert_eq!(value.data, 0);
            let value = board.bus.read64(address);
            assert_eq!(value.status, BUS_OK);
            assert_eq!(value.data, 0);
        }
        assert_eq!(board.bus.read64(base + 0x80).status, BUS_OK);
    }
}

/// A busy host port or unusable transcript path must not prevent a serial
/// machine from starting, including when any of its three ports serves CI.
#[test]
fn unavailable_serial_ports_and_logs_do_not_abort_the_machine() {
    use crate::config::{FuelConsole, MachineConfig};
    let listeners: Vec<_> = (0..3).map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap()).collect();
    let ports: Vec<_> = listeners.iter().map(|socket| socket.local_addr().unwrap().port()).collect();
    let board = crate::ip34_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let mut config = MachineConfig::default();
    config.fuel.l1_port = Some(ports[0]);
    config.serial_port_a = Some(ports[1]);
    config.serial_port_b = Some(ports[2]);
    config.ci = true;
    let missing = std::env::temp_dir().join(format!("iris-fuel-missing-log-{}", std::process::id())).join("absent").join("serial.txt");
    assert!(!missing.parent().unwrap().exists(), "the log parent must be absent");
    config.serial_log = Some(missing.to_string_lossy().into_owned());
    config.fuel.l1_log = config.serial_log.clone();
    for (console, selection) in [FuelConsole::L1, FuelConsole::Ioc3A, FuelConsole::Ioc3B].into_iter().enumerate() {
        config.fuel.console = selection;
        crate::ip34_machine::FuelMachine::connect_serial_for_test(&config, &board, console).unwrap();
    }
    config.ci = false;
    crate::ip34_machine::FuelMachine::connect_serial_for_test(&config, &board, 0).unwrap();
}

fn fuel_with_graphics(graphics: crate::config::GraphicsBoard) -> crate::ip34_machine::FuelBoard {
    crate::ip34_machine::FuelBoard::with_config(crate::ip34_machine::FuelBoardConfig {
        payload: &[], clock: Arc::new(|| 0), flash_path: None,
        station_mac: [0x08, 0x00, 0x69, 0x21, 0x43, 0x65],
        timekeeper: Arc::new(crate::dev::timekeeper::Timekeeper::new().unwrap()),
        ps2_input: true,
        l1: Default::default(),
        graphics,
        graphics_pool: (2, 32),
        graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
    }).unwrap()
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
/// reach the board there, learn its widget number from it, and find the
/// link alive in the crossbar.
#[test]
fn the_vpro_answers_at_widget_13_with_its_link_alive() {
    for (graphics, revision) in [(crate::config::GraphicsBoard::V12, 0xa), (crate::config::GraphicsBoard::V10, 0xa)] {
        let board = fuel_with_graphics(graphics);
        let identity = board.bedrock.pio().read32(VPRO_PIO | (ODSY_IDENTITY + 4));
        assert_eq!(identity.status, BUS_OK);
        assert_eq!(identity.data as u32, (revision << 28) | 0x0c01_3047, "{graphics:?}");
        let control = board.bedrock.pio().read32(VPRO_PIO | (ODSY_CONTROL + 4)).data as u32;
        assert_eq!(control & ODSY_CONTROL_WIDGET_MASK, 13);
        assert_ne!(board.xbridge.crossbar().read32(VPRO_LINK_STATUS).data & (1 << 31), 0);
    }
}

/// The monitor's `vpro` commands resolve on a Fuel that has a VPro, and are
/// unknown on one that has none.
#[test]
fn the_monitor_resolves_vpro_commands_on_a_fuel_with_a_vpro() {
    let resolves = |board: &crate::ip34_machine::FuelBoard, command: &str| {
        board.monitor_devices().iter().any(|d| d.register_commands().iter().any(|(c, _)| c == command))
    };
    let with = fuel_with_graphics(crate::config::GraphicsBoard::V12);
    assert!(resolves(&with, "vpro"), "a Fuel with a V12 answers `vpro`");
    assert!(resolves(&with, "ps2"), "the other board commands stay");
    let without = fuel_with_graphics(crate::config::GraphicsBoard::None);
    assert!(!resolves(&without, "vpro"), "a Fuel with no VPro has no `vpro`");
    assert!(resolves(&without, "ps2"));
}

/// The board's interrupt packets cross the XBridge to Bedrock as the PCI
/// heads' do: a raised source sets its vector's pending bit in the
/// processor interface the destination names, and a cleared one clears it.
#[test]
fn vpro_interrupt_packets_set_and_clear_a_processor_interface_pending_bit() {
    let board = fuel_with_graphics(crate::config::GraphicsBoard::V12);
    // Register space (bit 47), the node's hub window and PI0's pending
    // modify register, at the Bedrock's crossbar port 0xa.
    let packet = (1u64 << 47) | 0x180_0000 | 0x90;
    let vector = 5;
    let write = |offset: u64, value: u64| {
        assert_eq!(board.bedrock.pio().write32(VPRO_PIO | offset, value as u32), BUS_OK, "{offset:#x}");
    };
    write(ODSY_ROUTE_DEST_UPPER, (0xa << ODSY_ROUTE_DEST_WIDGET_SHIFT) | (packet >> 32));
    write(ODSY_ROUTE_DEST_LOWER, packet & 0xffff_ffff);
    write(ODSY_ROUTE_VECTORS, vector);
    write(ODSY_INTR_ENABLE_SET, ODSY_INTR_FIFO_LOW_WATER as u64);
    let pending = || pi_regs(&board, 0 as u64).read64(0x98).data & (1 << vector);
    assert_ne!(pending(), 0);
    write(ODSY_INTR_CLEAR, ODSY_INTR_FIFO_LOW_WATER as u64);
    assert_eq!(pending(), 0);
}

/// A link reset of the graphics port resets the board behind it, as the
/// PCI port's does for the functions behind its head.
#[test]
fn a_graphics_link_reset_resets_the_vpro() {
    let board = fuel_with_graphics(crate::config::GraphicsBoard::V12);
    let destination = VPRO_PIO | (ODSY_INT_DEST_LOWER + 4);
    assert_eq!(board.bedrock.pio().write32(destination, 0x55 as u32), BUS_OK);
    assert_eq!(board.bedrock.pio().read32(destination).data, 0x55);
    let link_reset = 0x100 + 5 * 0x40 + 0x30 + 4;
    assert_eq!(board.xbridge.crossbar().write32(link_reset, 1), BUS_OK);
    assert_eq!(board.bedrock.pio().read32(destination).data, 0);
}

/// Without graphics nothing answers at widget 13 and its link is down, as
/// on a headless Fuel.
#[test]
fn a_fuel_without_graphics_has_no_widget_13() {
    let board = fuel_with_graphics(crate::config::GraphicsBoard::None);
    assert!(board.vpro.is_none());
    assert_ne!(board.bedrock.pio().read32(VPRO_PIO | 4).status, BUS_OK);
    assert_eq!(board.xbridge.crossbar().read32(VPRO_LINK_STATUS).data & (1 << 31), 0);
}

/// IRIX's VPro driver writes each processor interface's graphics credit
/// counter at attach and tears the board down if it reads back zero; the
/// registers must keep what the guest writes, per interface.
#[test]
fn processor_interface_graphics_registers_keep_their_writes() {
    let board = fuel_with_graphics(crate::config::GraphicsBoard::V12);
    // Slot A's credit counter and bias; PI1 keeps its own block.
    assert_eq!(pi_regs(&board, 0 as u64).write64(0x308, 100), BUS_OK);
    assert_eq!(pi_regs(&board, 0 as u64).write64(0x310, 0x1000 | 68), BUS_OK);
    assert_eq!(pi_regs(&board, 0 as u64).read64(0x308).data, 100);
    assert_eq!(pi_regs(&board, 0 as u64).read64(0x310).data, 68);
    assert_eq!(pi_regs(&board, 1 as u64).read64(0x308).data, 0);
    board.bedrock.power_on();
    assert_eq!(pi_regs(&board, 0 as u64).read64(0x308).data, 0);
}

/// The reset PROM fetches uncached at the same plain address that cached RAM
/// answers once memory covers its window. The board presents that fetch at the
/// PROM's own address (`uncached_fetch_alias`), so the JIT keys the page by
/// address and compiles the PROM's bytes; it used to read the cached view and
/// compile the RAM's bytes in place of the PROM's.
#[cfg(feature = "jitv2")]
#[test]
fn uncached_fetch_compiles_the_prom_not_the_ram_over_it() {
    // The executor and the code generator are large; the default test stack is not.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(uncached_fetch_compiles_the_prom_body)
        .unwrap()
        .join()
        .expect("thread panicked");
}

#[cfg(feature = "jitv2")]
fn uncached_fetch_compiles_the_prom_body() {
    use crate::cpu::jitv2::jitv2::CompileRequest;
    use crate::cpu::mips_isa::{JIT_REGION_BOUNDARY_SENTINEL, OP_ADDIU};

    /// Every word the same; a region-boundary word compiles to nothing.
    struct Words {
        word: u32,
        generation: AtomicU64,
    }
    impl Words {
        fn new(word: u32) -> Arc<Self> {
            Arc::new(Self { word, generation: AtomicU64::new(0) })
        }
    }
    impl BusDevice for Words {
        fn read32(&self, _: u64) -> BusRead32 {
            BusRead32::ok(self.word)
        }
        fn read64(&self, _: u64) -> BusRead64 {
            BusRead64::ok((self.word as u64) << 32 | self.word as u64)
        }
        fn gen_ptr(&self, _: u64) -> *const AtomicU64 {
            &self.generation
        }
    }

    let reset_vector = 0x1fc0_0000u64;
    let addiu_r1 = (OP_ADDIU << 26) | (1 << 16) | 1;
    let rom = Words::new(addiu_r1);
    let ram = Words::new(JIT_REGION_BOUNDARY_SENTINEL);
    let physical = Physical::new(Vec::new());
    let mut bus = Ip35Bus::new(physical.clone());
    map_device(&physical, reset_vector, 0x1000, ram.clone(), "RAM over the reset window");
    bus.map(0, reset_vector, 0x1000, rom.clone());
    bus.map(0, RESET_WINDOW_FETCH, 0x1000, rom.clone());
    let bus = Arc::new(bus);
    let code_bus = bus.code_view(bus.cached_view(ram.clone(), RamExtent::new(&[1024])));
    // The two views really disagree here.
    assert_eq!(bus.read32(reset_vector).data, addiu_r1);
    assert_eq!(bus.cached_view(ram.clone(), RamExtent::new(&[1024])).read32(reset_vector).data, JIT_REGION_BOUNDARY_SENTINEL);

    let mut cpu = alias_cpu(bus.clone());
    alias_the_reset_window(&mut cpu);
    cpu.set_cached_bus(code_bus.clone());
    // The reset vector, KSEG1: uncached, plain physical address 0x1fc00000.
    cpu.core.pc = 0xffff_ffff_bfc0_0000;
    let _ = cpu.step_int();
    let page = cpu.jitv2_current_page();
    assert!(!page.is_null(), "the uncached fetch must land on a code page");
    let page = unsafe { &mut *page };
    assert_eq!(page.pfn as u64, RESET_WINDOW_FETCH / 4096, "an uncached fetch of the reset window keys the PROM's own frame");

    page.mark_requested(0);
    let req = CompileRequest { page: page as *mut _, compiled_for_fr1: true };
    let mut analyzer = crate::cpu::jitv2::analyzer::Analyzer::new();
    let mut codegen = crate::cpu::jitv2::codegen::Codegen::new();
    #[cfg(feature = "developer")]
    crate::cpu::jitv2::comp::handle_request(&req, &cpu.cached_bus(), &mut analyzer, &mut codegen, &crate::cpu::jitv2::JitStats::default());
    #[cfg(not(feature = "developer"))]
    crate::cpu::jitv2::comp::handle_request(&req, &cpu.cached_bus(), &mut analyzer, &mut codegen);
    assert!(page.is_runnable(0), "the page must compile the PROM's ADDIU, not the RAM's boundary word");

    // The cached view of the same frame is another page, with RAM's counter and bytes.
    let jitv2 = cpu.jitv2.clone();
    let mut jit = jitv2.lock();
    let slot = jit.page_for(0x1fc00, 0x1fc0_0000, bus.as_ref(), false).unwrap();
    let cached = unsafe { &mut *jit.page_ptr(slot) };
    assert_ne!(cached as *mut _, page as *mut _);
    cached.mark_requested(0);
    let req = CompileRequest { page: cached as *mut _, compiled_for_fr1: true };
    #[cfg(feature = "developer")]
    crate::cpu::jitv2::comp::handle_request(&req, &code_bus, &mut analyzer, &mut codegen, &crate::cpu::jitv2::JitStats::default());
    #[cfg(not(feature = "developer"))]
    crate::cpu::jitv2::comp::handle_request(&req, &code_bus, &mut analyzer, &mut codegen);
    assert!(!cached.is_runnable(0), "the cached page compiles RAM's boundary word to nothing");

    // A write through one view invalidates only the pages that view's bytes feed.
    ram.generation.fetch_add(1, std::sync::atomic::Ordering::Release);
    assert!(page.is_runnable(0), "a cached-view write does not change the PROM's bytes");
    rom.generation.fetch_add(1, std::sync::atomic::Ordering::Release);
    assert!(!page.is_runnable(0), "a write to the PROM's bytes invalidates its code");
}

/// The R16000 is the part that takes an uncached fetch alias.
#[cfg(feature = "jitv2")]
type AliasCpu = crate::cpu::mips_exec::MipsExecutor<crate::cpu::mips_tlb::MipsTlb, crate::cpu::mips_cache_shadow::R16000ShadowCache>;

#[cfg(feature = "jitv2")]
fn alias_cpu(bus: Arc<Ip35Bus>) -> Box<AliasCpu> {
    use crate::cpu::mips_exec::{MipsCpuConfig, MipsExecutor};
    use crate::cpu::mips_tlb::MipsTlb;
    let cfg = MipsCpuConfig::for_model::<crate::cpu::mips_cache_shadow::R16000ShadowCache>(Arc::new(memory_map(&RamExtent::new(&[1024]))))
        .with_prid(crate::cpu::parts::R16000_700_EARLY.prid());
    let mut cpu = Box::new(MipsExecutor::new(bus as Arc<dyn BusDevice>, MipsTlb::new(cfg.tlb_entries), &cfg));
    cpu.phys_mask = CPU_PHYS_MASK;
    // KSEG0 cached, as the PROM's boot mode leaves it.
    cpu.core.cp0_config = (cpu.core.cp0_config & !7) | 3;
    cpu
}

#[cfg(feature = "jitv2")]
fn alias_the_reset_window(cpu: &mut AliasCpu) {
    cpu.set_uncached_fetch_alias(RESET_WINDOW, RESET_WINDOW_SIZE, RESET_WINDOW_FETCH);
}

#[cfg(feature = "jitv2")]
mod code_views {
    use super::*;
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

    /// A bus whose uncached attribute-0 view of the reset frame holds an ADDIU
    /// and whose cached view holds a region boundary, as RAM over the PROM does.
    fn bus_and_cpu(aliased: bool) -> (Arc<Ip35Bus>, Box<AliasCpu>) {
        let rom = Arc::new(Words(ADDIU_R1, AtomicU64::new(0)));
        let ram = Arc::new(Words(JIT_REGION_BOUNDARY_SENTINEL, AtomicU64::new(0)));
        let physical = Physical::new(Vec::new());
        let mut bus = Ip35Bus::new(physical.clone());
        map_device(&physical, RESET, 0x1000, ram.clone(), "RAM over the reset window");
        bus.map(0, RESET, 0x1000, rom.clone());
        // Code that is fetchable in the other views and past the end of memory, so only
        // the JIT's own range check keeps it out of a code page.
        for attribute in 1..4 {
            bus.map(attribute, RESET, 0x1000, rom.clone());
        }
        bus.map(0, RESET_WINDOW_FETCH, 0x1000, rom.clone());
        bus.map(0, 0x3_0000_0000, 0x1000, rom);
        let bus = Arc::new(bus);
        let mut cpu = alias_cpu(bus.clone());
        if aliased {
            let cached = bus.cached_view(ram, RamExtent::new(&[1024]));
            alias_the_reset_window(&mut cpu);
            cpu.set_cached_bus(bus.code_view(cached));
        }
        (bus, cpu)
    }

    /// The executor and the code generator are large; the default test stack is not.
    fn big_stack(body: fn()) {
        std::thread::Builder::new().stack_size(64 << 20).spawn(body).unwrap().join().expect("thread panicked");
    }

    fn wait_runnable(page: &crate::cpu::jitv2::PhysicalCodePage) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while std::time::Instant::now() < deadline {
            if page.is_runnable(0) { return true; }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        false
    }

    /// After a page-pool flush the restarted compile pool must still read the
    /// PROM's page through the compile bus. Restarted on the CPU's own bus it
    /// rejects the view address and the page never compiles.
    #[test]
    fn a_pool_restart_after_a_flush_keeps_the_compile_bus() {
        big_stack(|| {
            let (_bus, mut cpu) = bus_and_cpu(true);
            {
                let jit = cpu.jitv2.clone();
                let mut g = jit.lock();
                let stats = g.stats.clone();
                g.compile_queue.set_thread_count(1);
                g.compile_queue.start(cpu.cached_bus().clone(), stats);
            }
            cpu.jitv2_flush_from_here(false);
            cpu.core.pc = 0xffff_ffff_bfc0_0000;
            let _ = cpu.step_int();
            let page = unsafe { &mut *cpu.jitv2_current_page() };
            assert_eq!(page.pfn as u64, RESET_WINDOW_FETCH / 4096);
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

    /// With no alias an uncached fetch keys the frame it names, as on every
    /// other machine; and with it a cached fetch of the same window still does.
    #[test]
    fn a_cached_fetch_and_an_unaliased_executor_key_the_frame_they_name() {
        big_stack(|| {
            let (_bus, mut cpu) = bus_and_cpu(false);
            cpu.core.pc = 0xffff_ffff_bfc0_0000;
            let _ = cpu.step_int();
            let page = unsafe { &*cpu.jitv2_current_page() };
            assert_eq!(page.pfn, 0x1fc00);

            let (_bus, mut cpu) = bus_and_cpu(true);
            // KSEG0: cached, the same physical address.
            cpu.core.pc = 0xffff_ffff_9fc0_0000;
            let _ = cpu.step_int();
            let page = unsafe { &*cpu.jitv2_current_page() };
            assert_eq!(page.pfn, 0x1fc00, "a cached fetch is not presented at the PROM's frame");
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
                0x9000_0003_0000_0000u64,                   // past the end of memory, attribute 0
            ] {
                // A fresh executor each time: a tracked page would survive a
                // fetch that wrongly kept the same frame.
                let (_bus, mut cpu) = bus_and_cpu(true);
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
