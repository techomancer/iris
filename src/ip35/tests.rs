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
/// The Fuel's processor is the R14000 at revision 2.3, with the Octane
/// R14000's caches; the Octane's own part is revision 2.4.
#[test]
fn the_fuel_runs_the_r14000_at_revision_2_3() {
    use crate::cpu::mips_cache_shadow::{R14000FuelShadowCache, R14000_PRID};
    use crate::cpu::mips_cache_v2::{CpuModel, MipsCache};
    assert_eq!(<R14000FuelShadowCache as CpuModel>::PRID, 0x0f23);
    assert_eq!(R14000_PRID, 0x0f24);
    assert_eq!(<R14000FuelShadowCache as MipsCache>::L2_SIZE, 2 << 20);
    // Typed R12000, it takes the R10000 family's behaviour: reserved
    // secondary-tag bits read as zero, and a refused uncached store posts.
    assert!(<R14000FuelShadowCache as CpuModel>::R10K_FAMILY);
    assert_ne!(<R14000FuelShadowCache as CpuModel>::SECONDARY_TAG_MASK, u64::MAX);
    assert!(<R14000FuelShadowCache as CpuModel>::POSTS_UNCACHED_STORES);
    assert!(<R14000FuelShadowCache as CpuModel>::PARTIAL_LWR_KEEPS_UPPER);
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
        graphics: crate::config::GraphicsBoard::None,
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
            graphics: crate::config::GraphicsBoard::None,
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

/// Where the cached and uncached views of a frame differ, the JIT keys and
/// compiles the page by the view the CPU fetched it through. The reset PROM
/// fetches uncached at the same plain address that cached RAM answers once
/// memory covers its window; the compile pool used to read the cached view and
/// compile the RAM's bytes in place of the PROM's.
#[cfg(feature = "jitv2")]
#[test]
fn uncached_fetch_compiles_the_uncached_view_of_the_page() {
    // The executor and the code generator are large; the default test stack is not.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(uncached_fetch_compiles_the_uncached_view_body)
        .unwrap()
        .join()
        .expect("thread panicked");
}

#[cfg(feature = "jitv2")]
fn uncached_fetch_compiles_the_uncached_view_body() {
    use crate::cpu::jitv2::jitv2::{CompileRequest, UNCACHED_VIEW_PFN};
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
    bus.set_ram_generations(ram.clone());
    bus.map(0, reset_vector, 0x1000, rom.clone());
    let bus = Arc::new(bus);
    let code_bus = bus.code_view(bus.cached_view(ram.clone()));
    // The two views really disagree here.
    assert_eq!(bus.read32(reset_vector).data, addiu_r1);
    assert_eq!(bus.cached_view(ram.clone()).read32(reset_vector).data, JIT_REGION_BOUNDARY_SENTINEL);

    let mut cpu = crate::ip35_machine::build_cpu(bus.clone());
    cpu.enable_jitv2_uncached_views(code_bus.clone());
    // The reset vector, KSEG1: uncached, plain physical address 0x1fc00000.
    cpu.core.pc = 0xffff_ffff_bfc0_0000;
    let _ = cpu.step_int();
    let page = cpu.jitv2_current_page();
    assert!(!page.is_null(), "the uncached fetch must land on a code page");
    let page = unsafe { &mut *page };
    assert_eq!(page.pfn, 0x1fc00 | UNCACHED_VIEW_PFN, "an uncached fetch keys the uncached view");
    assert!(page.is_uncached_view());
    assert_eq!(page.phys_base(), 0x1fc0_0000);

    page.mark_requested(0);
    let req = CompileRequest { page: page as *mut _, compiled_for_fr1: true };
    let mut analyzer = crate::cpu::jitv2::analyzer::Analyzer::new();
    let mut codegen = crate::cpu::jitv2::codegen::Codegen::new();
    #[cfg(feature = "developer")]
    crate::cpu::jitv2::comp::handle_request(&req, &cpu.jitv2_compile_bus(), &mut analyzer, &mut codegen, &crate::cpu::jitv2::JitStats::default());
    #[cfg(not(feature = "developer"))]
    crate::cpu::jitv2::comp::handle_request(&req, &cpu.jitv2_compile_bus(), &mut analyzer, &mut codegen);
    assert!(page.is_runnable(0), "the page must compile the PROM's ADDIU, not the RAM's boundary word");

    // The cached view of the same frame is another page, with RAM's counter and bytes.
    let jitv2 = cpu.jitv2.clone();
    let mut jit = jitv2.lock();
    let slot = jit.page_for(0x1fc00, 0x1fc0_0000, bus.as_ref(), false).unwrap();
    let cached = unsafe { &mut *jit.page_ptr(slot) };
    assert!(!cached.is_uncached_view());
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

#[cfg(feature = "jitv2")]
mod code_views {
    use super::*;
    use crate::cpu::jitv2::jitv2::{CompileRequest, UNCACHED_VIEW_PFN};
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
    fn bus_and_cpu(opted_in: bool) -> (Arc<Ip35Bus>, Box<crate::ip35_machine::FuelCpu>) {
        let rom = Arc::new(Words(ADDIU_R1, AtomicU64::new(0)));
        let ram = Arc::new(Words(JIT_REGION_BOUNDARY_SENTINEL, AtomicU64::new(0)));
        let physical = Physical::new(Vec::new());
        let mut bus = Ip35Bus::new(physical.clone());
        map_device(&physical, RESET, 0x1000, ram.clone(), "RAM over the reset window");
        bus.set_ram_generations(ram.clone());
        bus.map(0, RESET, 0x1000, rom.clone());
        // Code that is fetchable in the other views and above 4 GiB, so only
        // the JIT's own range check keeps it out of a code page.
        for attribute in 1..4 {
            bus.map(attribute, RESET, 0x1000, rom.clone());
        }
        bus.map(0, 0x1_0000_0000, 0x1000, rom);
        let bus = Arc::new(bus);
        let mut cpu = crate::ip35_machine::build_cpu(bus.clone());
        if opted_in {
            let cached = bus.cached_view(ram);
            cpu.enable_jitv2_uncached_views(bus.code_view(cached));
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

    /// After a page-pool flush the restarted compile pool must still read
    /// uncached pages through the compile bus. Restarted on the CPU's own bus it
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
                g.compile_queue.start(cpu.jitv2_compile_bus(), stats);
            }
            cpu.jitv2_flush_from_here(false);
            cpu.core.pc = 0xffff_ffff_bfc0_0000;
            let _ = cpu.step_int();
            let page = unsafe { &mut *cpu.jitv2_current_page() };
            assert!(page.is_uncached_view());
            page.mark_requested(0);
            let jit = cpu.jitv2.clone();
            let g = jit.lock();
            let stats = g.stats.clone();
            assert!(g.compile_queue.send(CompileRequest { page: page as *mut _, compiled_for_fr1: true }, &stats));
            drop(g);
            assert!(wait_runnable(page), "the restarted pool must compile the uncached view");
            jit.lock().compile_queue.stop();
        });
    }

    /// A machine that has not opted in keeps one page per frame and the
    /// original-size frame map, whatever the cache attribute of the fetch.
    #[test]
    fn an_executor_that_has_not_opted_in_keeps_the_plain_key() {
        big_stack(|| {
            let (_bus, mut cpu) = bus_and_cpu(false);
            assert!(!cpu.jitv2.lock().uncached_views_enabled());
            cpu.core.pc = 0xffff_ffff_bfc0_0000;
            let _ = cpu.step_int();
            let page = unsafe { &*cpu.jitv2_current_page() };
            assert_eq!(page.pfn, 0x1fc00);
            assert_eq!(page.pfn & UNCACHED_VIEW_PFN, 0);
            assert!(!page.is_uncached_view());
        });
    }

    /// Uncached attributes 1 to 3 and anything over 4 GiB stay in the interpreter.
    #[test]
    fn attributes_one_to_three_and_high_addresses_get_no_code_page() {
        big_stack(|| {
            for pc in [
                0x9000_0000_0000_0000u64 | RESET,           // attribute 0: a page
                0x9200_0000_0000_0000u64 | RESET,           // attribute 1
                0x9400_0000_0000_0000u64 | RESET,           // attribute 2
                0x9600_0000_0000_0000u64 | RESET,           // attribute 3
                0x9000_0001_0000_0000u64,                   // 4 GiB, attribute 0
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

fn fuel_with_graphics(graphics: crate::config::GraphicsBoard) -> crate::ip35_machine::FuelBoard {
    crate::ip35_machine::FuelBoard::with_config(crate::ip35_machine::FuelBoardConfig {
        payload: &[], clock: Arc::new(|| 0), flash_path: None,
        station_mac: [0x08, 0x00, 0x69, 0x21, 0x43, 0x65],
        timekeeper: Arc::new(crate::dev::timekeeper::Timekeeper::fuel().unwrap()),
        ps2_input: true, usb_input: true,
        graphics,
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
        let identity = board.io.pio_read(VPRO_PIO | (ODSY_IDENTITY + 4), 4);
        assert_eq!(identity.status, BUS_OK);
        assert_eq!(identity.data as u32, (revision << 28) | 0x0c01_3047, "{graphics:?}");
        let control = board.io.pio_read(VPRO_PIO | (ODSY_CONTROL + 4), 4).data as u32;
        assert_eq!(control & ODSY_CONTROL_WIDGET_MASK, 13);
        assert_ne!(board.xbridge.crossbar().read32(VPRO_LINK_STATUS).data & (1 << 31), 0);
    }
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
        assert_eq!(board.io.pio_write(VPRO_PIO | offset, 4, value), BUS_OK, "{offset:#x}");
    };
    write(ODSY_ROUTE_DEST_UPPER, (0xa << ODSY_ROUTE_DEST_WIDGET_SHIFT) | (packet >> 32));
    write(ODSY_ROUTE_DEST_LOWER, packet & 0xffff_ffff);
    write(ODSY_ROUTE_VECTORS, vector);
    write(ODSY_INTR_ENABLE_SET, ODSY_INTR_FIFO_LOW_WATER as u64);
    let pending = || board.pi[0].read64(0x98).data & (1 << vector);
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
    assert_eq!(board.io.pio_write(destination, 4, 0x55), BUS_OK);
    assert_eq!(board.io.pio_read(destination, 4).data, 0x55);
    let link_reset = 0x100 + 5 * 0x40 + 0x30 + 4;
    assert_eq!(board.xbridge.crossbar().write32(link_reset, 1), BUS_OK);
    assert_eq!(board.io.pio_read(destination, 4).data, 0);
}

/// Without graphics nothing answers at widget 13 and its link is down, as
/// on a headless Fuel.
#[test]
fn a_fuel_without_graphics_has_no_widget_13() {
    let board = fuel_with_graphics(crate::config::GraphicsBoard::None);
    assert!(board.vpro.is_none());
    assert_ne!(board.io.pio_read(VPRO_PIO | 4, 4).status, BUS_OK);
    assert_eq!(board.xbridge.crossbar().read32(VPRO_LINK_STATUS).data & (1 << 31), 0);
}

/// IRIX's VPro driver writes each processor interface's graphics credit
/// counter at attach and tears the board down if it reads back zero; the
/// registers must keep what the guest writes, per interface.
#[test]
fn processor_interface_graphics_registers_keep_their_writes() {
    let board = fuel_with_graphics(crate::config::GraphicsBoard::V12);
    // Slot A's credit counter and bias; PI1 keeps its own block.
    assert_eq!(board.pi[0].write64(0x308, 100), BUS_OK);
    assert_eq!(board.pi[0].write64(0x310, 0x1000 | 68), BUS_OK);
    assert_eq!(board.pi[0].read64(0x308).data, 100);
    assert_eq!(board.pi[0].read64(0x310).data, 68);
    assert_eq!(board.pi[1].read64(0x308).data, 0);
    board.pi[0].power_on();
    assert_eq!(board.pi[0].read64(0x308).data, 0);
}
