use super::*;
use crate::traits::{BusDevice, BUS_OK};

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
            let mut cpu = crate::ip35_machine::build_cpu(bus);
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
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_eq!(board.md.md_bus().write64(8, 0x33), BUS_OK);
    let at = (1 << 30) + 0x1000;
    assert_eq!(board.md.memory_view().write64(at, 0x12345678), BUS_OK);
    assert_eq!(board.cached_bus().read64(at).data, 0x12345678);
    assert_eq!(board.bus.read64((3 << 57) | at).data, 0x12345678);
    assert_eq!(board.bus.read64(at).status, BUS_ERR);
    assert_eq!(board.md.dma_view().read64(at).data, 0x12345678);
}

/// Memory diagnostics must notify both processor interfaces and refuse bad uncached data.
#[test]
fn memory_errors_reach_both_interfaces_and_uncached_reads_fail() {
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let correctable = 1 << 61;
    board.pi[0].write64(0xb0, correctable);
    board.md.md_bus().write64(8, 0x33);
    for directory in [false, true] {
        if directory {
            board.md.md_bus().write64(0x40000, 2);
            board.md.directory_bus().write64(0, 1 << 16);
            board.md.md_bus().write64(0x40000, 0);
        } else {
            board.md.ecc_bus().write8(7, 1);
        }
        assert_eq!(board.bus.read64(0).status, BUS_OK);
        for pi in &board.pi {
            assert_eq!(pi.read64(0xa0).data & correctable, correctable);
        }
        assert_eq!(board.irq.bits() & (1 << 11), 1 << 11);
        for pi in &board.pi {
            pi.write64(0x90, 64 + 61);
            assert_eq!(pi.read64(0xa0).data & correctable, 0);
        }
        assert_eq!(board.irq.bits() & (1 << 11), 0);
    }

    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    board.md.md_bus().write64(8, 0x33);
    board.md.ecc_bus().write8(7, 3);
    for selector in [0, 3] {
        assert_eq!(board.bus.read64(selector << 57).status, BUS_ERR);
    }
    assert_eq!(board.cached_bus().read64(0).status, BUS_OK);
    assert_eq!(board.md.dma_view().read64(0).status, BUS_OK);
}

/// Firmware resets the PCI link before reinitializing the attached storage controller.
#[test]
fn fuel_link_reset_resets_the_attached_scsi_engine() {
    use crate::dev::isp1040::{mbox, reg, CTRL_INT_ENABLE, CTRL_RISC_INT_ENABLE, HOST_CMD_SHIFT};
    use crate::pci::PciBusDevice;
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let isp = &board.scsi;
    isp.config().write(1, 6);
    isp.config().write(5, 0x02000000);
    isp.bar_write(1, reg::BUS_CTRL, u32::from(CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE), 2);
    isp.bar_write(1, reg::MAILBOX_BASE, u32::from(mbox::NOP), 2);
    isp.bar_write(1, reg::HOST_CMD, 0x5 << HOST_CMD_SHIFT, 2);
    assert_eq!(isp.bar_read(1, reg::MAILBOX_BASE, 2), u32::from(mbox::COMPLETE));
    assert_ne!(isp.bar_read(1, reg::BUS_CTRL, 2), 0);
    assert_eq!(board.xbridge.widget(0).unwrap().write32(0x2f4, 1), BUS_OK);
    assert_eq!(isp.config().command(), 0);
    assert_eq!(isp.config().bar_base(1), 0);
    assert_eq!(isp.bar_read(1, reg::BUS_CTRL, 2), 0);
    assert_eq!(isp.bar_read(1, reg::MAILBOX_BASE, 2), 0);
    assert_eq!(isp.bar_read(1, reg::MAILBOX_BASE + 2, 2), 0x4953);
}

/// The PROM reads the station address from the L1 and IRIX from the
/// Ethernet; both must report the configured one.
#[test]
fn the_configured_station_address_reaches_the_l1_and_the_ethernet() {
    let mac = [0x08, 0x00, 0x69, 0x21, 0x43, 0x65];
    let board = crate::ip35_machine::FuelBoard::with_config(crate::ip35_machine::FuelBoardConfig {
        payload: &[], clock: Arc::new(|| 0), flash_path: None,
        station_mac: mac,
        timekeeper: Arc::new(crate::dev::timekeeper::Timekeeper::fuel().unwrap()),
        ps2_input: true, usb_input: true,
    }).unwrap();
    assert_eq!(board.l1.station_mac(), mac);
    assert_eq!(board.eth.mac(), mac);
}

/// The Fuel's cache reports primary data tag stores to the memory controller:
/// the PROM's installs over its window start the private page, and its sweep
/// ends it. Other caches' tag stores and non-installs do nothing.
#[test]
fn the_prom_tag_stores_reach_the_memory_controller_through_the_cpu_cache() {
    // An executor is large; a test thread's stack is not.
    std::thread::Builder::new().stack_size(64 << 20).spawn(prom_tag_stores).unwrap().join().unwrap();
}

fn prom_tag_stores() {
    use crate::cpu::mips_cache_v2::MipsCache;
    use crate::cpu::mips_isa::{CACH_PD, CACH_PI, C_IST};
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    let cpu = board.build_cpu();
    let window = |at: u64| board.md.physical().read64(at).status == BUS_OK;
    assert!(!window(0x10_0000));
    cpu.cache.cache_op(C_IST | CACH_PD, 0xa800_0000_0010_0000, 0xffff_ffcf);
    cpu.cache.cache_op(C_IST | CACH_PI, 0xa800_0000_0010_0000, 0x100c0);
    assert!(!window(0x10_0000), "a pattern and an instruction cache tag start nothing");
    cpu.cache.cache_op(C_IST | CACH_PD, 0xa800_0000_0010_0000, 0x100c0);
    assert!(window(0x10_0000));
    cpu.cache.cache_op(C_IST | CACH_PD, 0xa800_0000_0010_0000, 0);
    assert!(!window(0x10_0000));
}

fn fuel_dma_board() -> crate::ip35_machine::FuelBoard {
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_eq!(board.md.md_bus().write64(8, 0x33), BUS_OK);
    board.xbridge.head(15).write32(0x24, 15);
    board.xbridge.set_wide_slot_mask(15, 0);
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
    let memory = board.md.memory_view();
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
    let memory = board.md.memory_view();
    // Bank zero has 256 MiB fitted. The second qword is outside its backing.
    let last = 0x0fff_fff8;
    assert_eq!(memory.write64(last, 0x0102030405060708), BUS_OK);
    let mut bytes = [0xff; 16];
    assert_eq!(dma.dma_read(fuel_dma_address(last), &mut bytes), PciDmaStatus::MasterAbort);
    assert_eq!(bytes, [1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(dma.dma_write(fuel_dma_address(last), &[0x42; 16]), PciDmaStatus::MasterAbort);
    assert_eq!(memory.read64(last).data, 0x4242424242424242);
    board.io.disable_device(15, 0);
    bytes.fill(0xff);
    assert_eq!(dma.dma_read(fuel_dma_address(last), &mut bytes), PciDmaStatus::MasterAbort);
    assert_eq!(bytes, [0; 16]);
    assert_eq!(dma.dma_write(fuel_dma_address(last), &[0x99; 8]), PciDmaStatus::MasterAbort);
    assert_eq!(memory.read64(last).data, 0x4242424242424242);
}

/// A wide DMA leg must report an uncorrectable word once without publishing bad bytes.
#[test]
fn fuel_dma_checks_an_aligned_uncorrectable_word_once() {
    use crate::dev::bedrock_md::MemoryError;
    use crate::pci::PciDmaStatus;
    use std::sync::atomic::{AtomicU64, Ordering};
    let board = fuel_dma_board();
    assert_eq!(board.md.memory_view().write64(0, 0), BUS_OK);
    assert_eq!(board.md.ecc_bus().write8(7, 3), BUS_OK);
    let count = Arc::new(AtomicU64::new(0));
    let seen = count.clone();
    let md = Arc::downgrade(&board.md);
    board.md.on_error(Arc::new(move |kind| {
        assert_eq!(kind, MemoryError::Uncorrectable);
        assert_eq!(md.upgrade().unwrap().current_cell(), u64::MAX);
        seen.fetch_add(1, Ordering::Relaxed);
    }));
    // The former callback read the same persistent UCE once for each byte.
    for address in 0..8 {
        assert_eq!(board.io.inbound_read(15, Some(0), address, 1).status, BUS_OK);
    }
    assert_eq!(count.swap(0, Ordering::Relaxed), 8);
    assert_ne!(board.md.md_bus().read64(0x60018).data & (1 << 61), 0);
    let mut bytes = [0xff; 8];
    assert_eq!(board.xbridge.dma(15, 0).dma_read(fuel_dma_address(0), &mut bytes), PciDmaStatus::Ok);
    assert_eq!(bytes, [0; 8]);
    assert_eq!(count.load(Ordering::Relaxed), 1);
    let record = board.md.md_bus().read64(0x60010).data;
    assert_ne!(record & (1 << 60), 0);
    assert_eq!(record & (1 << 61), 0);
    assert_eq!(board.md.uncached_view().read64(0).status, BUS_ERR);
}

/// Inbound Hub writes must preserve access width and the requesting device identity.
#[test]
fn dma_hub_access_preserves_width_and_source_permissions() {
    use crate::pci::PciDmaStatus;
    let board = fuel_dma_board();
    let dma = board.xbridge.dma(15, 0);
    let hub = (1 << 47) | 0x1800000;
    for (interface, offset) in [(0, 0), (1, 0x200000)] {
        let address = fuel_dma_address(hub + offset + 0x90);
        assert_eq!(dma.dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::Ok);
        assert_ne!(board.pi[interface].read64(0x98).data & (1 << 12), 0);
        for width in [1, 2, 4] {
            assert_eq!(dma.dma_write(address, &[0; 8][..width]), PciDmaStatus::MasterAbort);
            assert_ne!(board.pi[interface].read64(0x98).data & (1 << 12), 0);
        }
        assert_eq!(dma.dma_write(address, &12u64.to_be_bytes()), PciDmaStatus::Ok);
        assert_eq!(board.pi[interface].read64(0x98).data & (1 << 12), 0);
    }
    let address = fuel_dma_address(hub + 0x90);
    // I/O modify must retain PI region protection and requesting-slot identity.
    assert_eq!(board.pi[0].write64(0x10, 0), BUS_OK);
    assert_eq!(dma.dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::MasterAbort);
    assert_eq!(board.pi[0].read64(0x98).data & (1 << 12), 0);
    assert_eq!(board.pi[0].write64(0x10, 1), BUS_OK);
    board.io.disable_device(15, 0);
    assert_eq!(dma.dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::MasterAbort);
    assert_eq!(board.pi[0].read64(0x98).data & (1 << 12), 0);
    assert_eq!(board.xbridge.dma(15, 1).dma_write(address, &0x10cu64.to_be_bytes()), PciDmaStatus::Ok);
    assert_ne!(board.pi[0].read64(0x98).data & (1 << 12), 0);
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
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_eq!(board.md.md_bus().write64(8, 0x33), BUS_OK);
    let cached = board.cached_bus();
    assert!(cached.mem_ptr(0x2000).is_some());
    assert_eq!(cached.write64(0x2000, 0x123456789abcdef0), BUS_OK);
    assert_eq!(board.md.dma_view().read64(0x2000).data, 0x123456789abcdef0);
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

/// Code-page lookup receives plain physical addresses, while tagged uncached
/// requests retain their view. Cached stores must also update the MD view.
#[cfg(feature = "jitv2")]
#[test]
fn code_generations_use_the_cached_plane_and_stable_ram_endpoint() {
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
    let mut bus = Ip35Bus::new(physical.clone());
    let raw_ram = Generations::new(false);
    let stable_ram = Generations::new(true);
    let rom = Generations::new(false);
    map_device(&physical, 0, 0x1000, raw_ram.clone(), "raw RAM generation");
    map_device(&physical, 0x1fc00000, 0x1000, rom.clone(), "ROM generation");
    bus.set_ram_generations(stable_ram.clone());
    bus.map(3, 0, 0x1000, stable_ram.clone());
    bus.map(3, 0x3000, 8, stable_ram.clone());
    let cached = bus.cached_view(stable_ram.clone());
    assert_eq!(bus.gen_ptr(0x40), &stable_ram.generation as *const AtomicU64);
    assert_eq!(cached.gen_ptr(0x40), &stable_ram.generation as *const AtomicU64);
    assert_eq!(bus.gen_ptr(0x1fc00000), &rom.generation as *const AtomicU64);
    assert_eq!(cached.gen_ptr(0x1fc00000), &rom.generation as *const AtomicU64);
    assert_eq!(bus.gen_ptr((3 << 57) | 0x40), &stable_ram.generation as *const AtomicU64);
    assert_eq!(bus.gen_ptr((3 << 57) | 0x3000), &stable_ram.generation as *const AtomicU64);
    assert!(bus.gen_ptr((3 << 57) | 0x3008).is_null());
    assert!(bus.gen_ptr((1 << 57) | 0x1fc00000).is_null());
    assert!(bus.gen_ptr(1 << 50).is_null());
    assert_eq!(cached.write64(0x40, 0x1234), BUS_OK);
    assert_eq!(cached.write_block(0x48, &[0x5678]), BUS_OK);
    assert_eq!(cached.write64_masked(0x50, 0xaabb, 0xff), BUS_OK);
    assert_eq!(*stable_ram.writes.lock().unwrap(), [(0x40, 0x1234), (0x48, 0x5678), (0x50, 0xbb)]);
    assert_eq!(cached.write64(0x1fc00000, 0x4321), BUS_OK);
    assert_eq!(*rom.writes.lock().unwrap(), [(0, 0x4321)], "the ROM window translates to a device-local offset");
    assert!(raw_ram.writes.lock().unwrap().is_empty());
}

/// Unmodeled Hub heads retain the PROM compatibility readback without
/// absorbing the adjacent, independently implemented L1 UART window.
#[test]
fn compatibility_hub_heads_do_not_hide_the_l1_uart_window() {
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
    assert_ne!(board.md.current_cell() & 1, 0);
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
    let board = crate::ip35_machine::FuelBoard::new(&[], Arc::new(|| 0)).unwrap();
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
        crate::ip35_machine::FuelMachine::connect_serial_for_test(&config, &board, console).unwrap();
    }
    config.ci = false;
    crate::ip35_machine::FuelMachine::connect_serial_for_test(&config, &board, 0).unwrap();
}

/// The Fuel's OHCI root ports each carry a low-speed device, and host
/// input reaches them through the IOC3's PS/2 controller object; the USB
/// pair is fitted only when asked for.
#[test]
fn the_fuel_plugs_a_usb_keyboard_and_mouse_only_when_configured_on() {
    use crate::pci::PciBusDevice;
    let ports = |usb_input| {
        let board = crate::ip35_machine::FuelBoard::with_config(crate::ip35_machine::FuelBoardConfig {
            payload: &[], clock: Arc::new(|| 0), flash_path: None,
            station_mac: [8, 0, 0x69, 0x12, 0x34, 0x56],
            timekeeper: Arc::new(crate::dev::timekeeper::Timekeeper::fuel().unwrap()),
            ps2_input: true, usb_input,
        }).unwrap();
        // Power each port: one with a device shows it connected.
        board.usb.bar_write(0, 0x54, 0x100, 4);
        board.usb.bar_write(0, 0x58, 0x100, 4);
        let status = [0x54, 0x58].map(|at| board.usb.bar_read(0, at, 4));
        (board, status)
    };
    let (board, status) = ports(true);
    for s in status {
        assert_eq!(s & 1, 1, "device connected");
        assert_eq!(s & 0x200, 0x200, "low speed");
    }
    // The mouse the USB side fits makes the shared controller report a mouse
    // only once the guest has configured it.
    assert!(!board.ioc3.ps2().mouse_enabled());
    let (_, status) = ports(false);
    assert_eq!(status.map(|s| s & 1), [0, 0], "nothing plugged in");
}
