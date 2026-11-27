use crate::window_map::Window;
use super::*;
use crate::ip35::test_support::{bus_of, Head, fuel_cpu, fuel_config, md_regs, pi_regs, pio_widget, with_machine};
use crate::test_temp::TempPath;
use crate::physical::Physical;
use crate::traits::*;
use std::sync::Arc;
/// [`fuel_cpu`] on the machine's node.
fn machine_cpu(
    m: &Machine<Bedrock, Flash>,
) -> Box<
    crate::cpu::mips_exec::MipsExecutor<
        crate::cpu::mips_tlb::MipsTlb,
        crate::cpu::mips_cache_shadow::R12000ShadowCache,
    >,
> {
    fuel_cpu(m.nodes[0].bus.clone(), m.nodes[0].chip.decode.physical().clone())
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
/// Run `body` on a Fuel with its first bank pair enabled, and an XBridge on its Bedrock whose DMA
/// views stand in for a card's master, which the guest cannot reach. It is built as `xbridge_on`
/// builds the board's, but not connected: the board's own holds those widgets.
fn on_fuel_dma(body: impl FnOnce(&Machine<Bedrock, Flash>, &crate::dev::xbridge::XBridge) + Send + 'static) {
    let dir = TempPath::dir("fuel-dma");
    with_machine(fuel_config(&dir), |m| {
        assert_eq!(md_regs(m).write64(8, 0x33), BUS_OK);
        let bedrock = &m.nodes[0].chip;
        let xbridge = crate::dev::xbridge::XBridge::new(BEDROCK_PORT, &|widget| {
            let port = bedrock.inbound(widget);
            (port.clone() as Arc<dyn BusDevice>, port as Arc<dyn crate::irq::InterruptController>)
        });
        assert_eq!(xbridge.widget(15).expect("head F").write32(0x24, 15), BUS_OK);
        body(m, &xbridge);
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
        // Bank zero has 256 MiB installed. The second qword is outside its backing.
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
        node::connect_cpus(&m.nodes[0].chip, std::slice::from_ref(&cpu)).unwrap();
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
        node::connect_cpus(&m.nodes[0].chip, std::slice::from_ref(&cpu)).unwrap();
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
    use crate::config::Ip35Console;
    let listeners: Vec<_> = (0..3).map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap()).collect();
    let ports: Vec<_> = listeners.iter().map(|socket| socket.local_addr().unwrap().port()).collect();
    let dir = TempPath::dir("fuel-busy-serial");
    let missing = dir.join("absent").join("serial.txt");
    assert!(!missing.parent().unwrap().exists(), "the log parent must be absent");
    for (ci, console) in [
        (true, Ip35Console::L1),
        (true, Ip35Console::Ioc3A),
        (true, Ip35Console::Ioc3B),
        (false, Ip35Console::L1),
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
/// The JIT keys a code page by frame and whether it was fetched uncached: with RAM over
/// the reset window, cached and uncached fetches of 0x1fc00000 read different bytes.
#[cfg(feature = "jitv2")]
mod code_pages {
    use super::*;
    use crate::window_map::{map_device, CPU_PHYS_MASK};
    use std::sync::atomic::AtomicU64;
    use crate::display_stop_tests::big_stack;
    use crate::cpu::jitv2::jitv2::CompileRequest;
    use crate::cpu::mips_isa::{JIT_REGION_BOUNDARY_SENTINEL, OP_ADDIU};

    struct Words(u32, AtomicU64, crate::traits::CodeBank);
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
        fn code_page(&self, _: u64) -> Option<crate::traits::CodePage> {
            Some(self.2.page(0, &self.1))
        }
    }

    const RESET: u64 = 0x1fc0_0000;
    const ADDIU_R1: u32 = (OP_ADDIU << 26) | (1 << 16) | 1;
    type Cpu = crate::cpu::mips_exec::MipsExecutor<crate::cpu::mips_tlb::MipsTlb, crate::cpu::mips_cache_shadow::R16000ShadowCache>;

    /// The PROM (an ADDIU) in every uncached view of the reset frame and RAM
    /// (a region boundary, which compiles to nothing) in the cached one.
    fn rig() -> (Arc<Words>, Arc<Words>, Box<Cpu>) {
        use crate::cpu::mips_exec::{MipsCpuConfig, MipsExecutor};
        let rom = Arc::new(Words(ADDIU_R1, AtomicU64::new(0), Default::default()));
        let ram = Arc::new(Words(JIT_REGION_BOUNDARY_SENTINEL, AtomicU64::new(0), Default::default()));
        let physical = Physical::new(Vec::new());
        physical.set_default(Arc::new(crate::sn::NoMemory(crate::devlog::LogModule::Bedrock)), "absent");
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
        let req = CompileRequest { page: page as *mut _, compiled_for_fr1: true, cpu: 0, tenancy: 0 };
        let mut analyzer = crate::cpu::jitv2::analyzer::Analyzer::new();
        let mut codegen = crate::cpu::jitv2::codegen::Codegen::new();
        let buses = cpu.jitv2.lock().buses().clone();
        #[cfg(feature = "developer")]
        crate::cpu::jitv2::comp::handle_request(&req, &buses, &mut analyzer, &mut codegen, &crate::cpu::jitv2::JitStats::default());
        #[cfg(not(feature = "developer"))]
        crate::cpu::jitv2::comp::handle_request(&req, &buses, &mut analyzer, &mut codegen);
    }

    #[test]
    fn a_cached_page_and_an_uncached_fetch_of_one_frame_compile_apart() {
        big_stack(|| {
            let (rom, ram, mut cpu) = rig();
            // The reset vector in KSEG1, then the same frame in KSEG0.
            let prom = page_at(&mut cpu, 0xffff_ffff_bfc0_0000);
            let cached = page_at(&mut cpu, 0xffff_ffff_9fc0_0000);
            assert_eq!((prom.addr_of(0), cached.addr_of(0)), (Some((0x1fc00, true)), Some((0x1fc00, false))));
            assert_ne!(prom.key, cached.key, "the frame decodes to the PROM uncached and to RAM cached");
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
            cpu.jitv2_flush_from_here();
            let page = page_at(&mut cpu, 0xffff_ffff_bfc0_0000);
            assert_eq!(page.addr_of(0), Some((0x1fc00, true)));
            page.mark_requested(0);
            let jit = cpu.jitv2.clone();
            let g = jit.lock();
            let stats = g.stats.clone();
            assert!(g.compile_queue.send(CompileRequest { page: page as *mut _, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));
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
/// `[jitv2] compile` reaches every product: `async` starts each node's compile pool and `sync`
/// leaves it stopped, for the processor to compile inline.
#[cfg(all(feature = "jitv2", not(feature = "jitv2_lockstep")))]
#[test]
fn the_jitv2_compile_setting_reaches_every_product_and_drop_joins_the_pool() {
    use crate::config::JitCompile;
    use crate::ip35::test_support::{config, tezro_config};
    let pools = |cfg: MachineConfig| with_machine(cfg, |m| m.cpus().map(|c| c.jitv2().lock().compile_queue.is_running()).collect::<Vec<_>>());
    for (mode, pool_runs) in [(JitCompile::Async, true), (JitCompile::Sync, false)] {
        let dir = TempPath::dir("ip35-jitmode");
        let (mut o3200, mut fuel, mut tezro) = (config(&dir, 1), fuel_config(&dir), tezro_config(&dir, 1));
        for cfg in [&mut o3200, &mut fuel, &mut tezro] {
            cfg.jitv2.compile = mode;
        }
        for (what, cfg) in [("the Origin 3200", o3200), ("the Fuel", fuel), ("the Tezro", tezro)] {
            assert_eq!(pools(cfg), [pool_runs], "{what}, {mode:?}");
        }
    }
}

/// A flash erase in flight across a stop and start waits out only the running time left, not
/// the machine's age.
#[test]
fn a_stop_and_start_keeps_a_flash_erase_in_flight() {
    use crate::workstation::Workstation;
    let dir = TempPath::dir("fuel-flash-clock");
    with_machine(fuel_config(&dir), move |m| {
        // An erase takes 512 ms of machine time: age the machine past it first.
        let clock = m.get_timer_manager().clock().clone();
        clock.restore(clock.now() + 600_000_000);
        let v = m.nodes[0].flash.boot_view();
        let unlock = || {
            assert_eq!(v.write64(8 * 0x555, 0xaa), BUS_OK);
            assert_eq!(v.write64(8 * 0x2aa, 0x55), BUS_OK);
        };
        unlock();
        assert_eq!(v.write64(8 * 0x555, 0x80), BUS_OK);
        unlock();
        assert_eq!(v.write64(4 * 0x200, 0x30), BUS_OK);
        m.stop();
        m.restart_peripherals();
        assert_eq!(v.read16(0x200).data & 0x80, 0, "the erase is still in flight");
    });
}

/// The I/O chip's first two ports serve `serial_socket_a/b` and keep `serial_log_a/b`, as the
/// other machines' do.
#[cfg(unix)]
#[test]
fn the_io_ports_honour_their_sockets_and_logs() {
    use crate::ip35::test_support::tezro_config;
    for (name, board) in [("fuel", fuel_config as fn(&std::path::Path) -> _), ("tezro", |d| tezro_config(d, 1))] {
        let dir = TempPath::dir(&format!("{name}-serial-lines"));
        let mut cfg = board(&dir);
        let path = |file: &str| dir.join(file).to_string_lossy().into_owned();
        cfg.serial_socket_a = Some(path("a.sock"));
        cfg.serial_socket_b = Some(path("b.sock"));
        cfg.serial_log_a = Some(path("a.log"));
        cfg.serial_log_b = Some(path("b.log"));
        let at = dir.to_path_buf();
        with_machine(cfg, move |_| {
            for file in ["a.sock", "b.sock", "a.log", "b.log"] {
                assert!(at.join(file).exists(), "{name}: {file}");
            }
        });
    }
}

/// A machine reset takes the Fuel's timekeeper out of a half-done clock write.
#[test]
fn a_reset_ends_the_fuels_clock_write() {
    use crate::dev::bridge::device_window_offset;
    use crate::dev::ioc3::reg::BYTEBUS_DEV0;
    use crate::dev::timekeeper::CONTROL;
    use crate::workstation::Workstation;
    let dir = TempPath::dir("fuel-clock-reset");
    with_machine(fuel_config(&dir), |m| {
        let pio = m.nodes[0].chip.pio();
        // The IOC3 is device 4 of head F; its byte-wide clock sits in the BAR.
        Head::new(pio.clone(), 15).map_bar(4, 0x10, 0x0200_0000);
        let control = pio_widget(15, u64::from(device_window_offset(4)) + u64::from(BYTEBUS_DEV0) + CONTROL as u64);
        assert_eq!(pio.write8(control, 0x80), BUS_OK);
        assert_eq!(pio.read8(control).data, 0x80, "the write latch is set");
        m.reset();
        // The reset cleared the BAR too.
        Head::new(pio.clone(), 15).map_bar(4, 0x10, 0x0200_0000);
        assert_eq!(pio.read8(control).data & 0x80, 0, "the write latch survived the reset");
    });
}

/// `headless` stores console=d in the Fuel's and the Origin 3200's base I/O store.
#[test]
fn headless_stores_console_d() {
    let dir = TempPath::dir("fuel-headless");
    with_machine(fuel_config(&dir), |_| ());
    assert!(crate::sn::tests::baseio_console_is_serial(&dir), "Fuel");
    let dir = TempPath::dir("origin3200-headless");
    with_machine(crate::ip35::test_support::config(&dir, 1), |_| ());
    assert!(crate::sn::tests::baseio_console_is_serial(&dir), "Origin 3200");
}
