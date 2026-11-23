use super::*;
use crate::cpu::parts;
use crate::config::MachineProfile;
use crate::dev::hub::regs as hub_regs;
use crate::irq::InterruptController;
use crate::test_temp::TempPath;
use test_support::{big_stack};
use crate::dev::hub::{installed_mappings, Hub, HubConfig, NODE_SPAN};
use crate::dev::node_flash::NodeFlash;
use crate::physical::BankMapping;
use crate::dev::hub::NODE_SHIFT;
use crate::sn::machine_memory;

fn flash(first: &[u8]) -> Arc<NodeFlash> {
    let mut image = vec![0xff; 0x10_0000];
    image[..first.len()].copy_from_slice(first);
    NodeFlash::new(image).unwrap()
}

/// `banks` on each of `count` nodes, their Hubs connected, with the windows of a board of
/// `window_nodes` nodes.
fn board(count: usize, banks: [u32; BANKS], window_nodes: usize, boot: &Arc<NodeFlash>) -> Vec<Arc<Hub>> {
    let config = |n: usize| HubConfig { first_cpu: 2 * n, banks_mb: banks, nodes: window_nodes, ..Default::default() };
    let hubs: Vec<_> = (0..count).map(|n| Hub::new(config(n), &crate::test_board::Stepped::new().tm, Some(boot.chip())).unwrap()).collect();
    if let [a, b] = &hubs[..] {
        a.connect_peer(b);
        b.connect_peer(a);
    }
    hubs
}

fn one(banks: [u32; BANKS], boot: &Arc<NodeFlash>) -> Arc<Hub> {
    board(1, banks, 1, boot).remove(0)
}

/// The Hub's node number and cached-alias stores, as the PROM makes them.
fn number(hub: &Hub, n: u64) {
    assert_eq!(hub.write64(crate::dev::hub::regs::NI_STATUS_REV_ID::OFFSET, n << 8), BUS_OK);
}
fn calias(hub: &Hub, bytes: u64) {
    let code = if bytes == 0 { 0 } else { (bytes >> 12).trailing_zeros() as u64 + 1 };
    assert_eq!(hub.write64(crate::dev::hub::regs::PI_CALIAS_SIZE::OFFSET, code), BUS_OK);
}

#[test]
fn a_small_bank_is_one_run_and_a_large_one_four() {
    assert_eq!(bank_mappings(0, 64), vec![(0, 0x400_0000)]);
    assert_eq!(bank_mappings(1, 128), vec![(0x2000_0000, 0x800_0000)]);
    assert_eq!(
        bank_mappings(2, 256),
        (0..4).map(|i| (0x4000_0000 + i * 0x800_0000, 0x400_0000)).collect::<Vec<_>>()
    );
    assert!(bank_mappings(3, 0).is_empty());
}

/// The layout skips empty banks, numbers the RAM of the installed ones in order,
/// and a 512 MiB bank is four runs of 128 MiB at 128 MiB intervals, each
/// starting where the last ended in the bank's own storage.
#[test]
fn the_layout_numbers_installed_banks_and_splits_a_large_one_in_four() {
    const MIB: u64 = 1 << 20;
    let runs = installed_mappings([64, 0, 512, 0]);
    assert_eq!(runs.len(), 5);
    assert_eq!(runs[0], BankMapping { base: 0, end: 64 * MIB, bank: 0, src: 0 });
    for (n, run) in runs[1..].iter().enumerate() {
        let n = n as u64;
        assert_eq!(*run, BankMapping { base: 1024 * MIB + n * 128 * MIB, end: 1024 * MIB + (n + 1) * 128 * MIB, bank: 1, src: n * 128 * MIB });
    }
    assert!(installed_mappings([0; BANKS]).is_empty());
}

#[test]
fn installed_memory_answers_where_the_runs_are_and_nowhere_else() {
    let hub = one([64, 0, 128, 0], &flash(&[]));
    let phys = hub.decode.physical();
    assert_eq!(phys.write32(0x1000, 0xdead_beef), BUS_OK);
    assert_eq!(phys.read32(0x1000).data, 0xdead_beef);
    assert_eq!(phys.write32(0x4000_0008, 7), BUS_OK);
    assert_eq!(phys.read32(0x4000_0008).data, 7);
}

/// A bank that is not installed, and the gap past a smaller one, fail every
/// access at every width: the firmware's sizing walk takes the bus error a
/// machine without that memory would give it.
#[test]
fn empty_memory_bus_errors_reads_and_writes() {
    let hub = one([64, 0, 128, 0], &flash(&[]));
    let phys = hub.decode.physical();
    for a in [0x400_0000, 0x2000_0000, 0x6000_0000] {
        assert_eq!(phys.read8(a).status, BUS_ERR, "{a:#x}");
        assert_eq!(phys.read16(a).status, BUS_ERR, "{a:#x}");
        assert_eq!(phys.read32(a).status, BUS_ERR, "{a:#x}");
        assert_eq!(phys.read64(a).status, BUS_ERR, "{a:#x}");
        assert_eq!(phys.write8(a, 1), BUS_ERR, "{a:#x}");
        assert_eq!(phys.write16(a, 1), BUS_ERR, "{a:#x}");
        assert_eq!(phys.write32(a, 1), BUS_ERR, "{a:#x}");
        assert_eq!(phys.write64(a, 1), BUS_ERR, "{a:#x}");
    }
    assert_eq!(phys.write32(0x3ff_fffc, 5), BUS_OK, "the last word of the installed bank still answers");
    assert_eq!(phys.read32(0x3ff_fffc).data, 5);
}

#[test]
fn the_reset_view_holds_the_flash_where_bank_zero_does_not_reach() {
    let hub = one([64, 0, 0, 0], &flash(&[0x12]));
    assert_eq!(hub.decode.physical().read8(0x1fc0_0000).data, 0x12);
    let map = memory_map([64, 0, 0, 0]);
    assert!(map.contains(0));
    assert!(map.contains(0x1fc0_0000));
    assert!(!map.contains(0x400_0000));
}

#[test]
fn the_bus_picks_its_space_from_the_attribute_bits() {
    struct Tag(u8);
    impl BusDevice for Tag {
        fn read8(&self, a: u64) -> BusRead8 {
            BusRead8::ok(self.0 ^ (a as u8))
        }
    }
    let bus = crate::sn::UncachedBus::new([Arc::new(Tag(0x10)), Arc::new(Tag(0x20)), Arc::new(Tag(0x30)), Arc::new(Tag(0x40))]);
    use crate::sn::UncachedSpace::*;
    for (attr, space, tag) in [(0u64, HubSpecial, 0x10u8), (1, Io, 0x20), (2, MemSpecial, 0x30), (3, Memory, 0x40)] {
        assert_eq!(crate::sn::UncachedSpace::of(attr << 57), space);
        assert_eq!(bus.read8((attr << 57) | 0x1_0000_0001).data, tag ^ 1, "attribute {attr}");
    }
    // A plain compatibility address has no attribute and reaches space 0.
    assert_eq!(bus.read8(0x1fc0_0001).data, 0x10 ^ 1);
}

#[test]
fn hub_special_repeats_memory_and_the_flash_at_the_named_node_offsets_too() {
    let flash = flash(&[0xff, 0xff, 0xff, 0x33]);
    let hub = one([64, 0, 0, 0], &flash);
    hub.decode.physical().write32(0x100, 0xcafe_f00d);
    let special = hub_special(hub.decode.own_memory(), &flash, hub.md.ecc_bus(), hub.md.directory_bus());
    for base in [0u64, 0x2000_0000] {
        assert_eq!(special.read32(base + 0x100).data, 0xcafe_f00d);
    }
    // The node's own memory whatever number it holds.
    number(&hub, 5);
    assert_eq!(special.read32(0x100).data, 0xcafe_f00d);
    for base in [0x1000_0000u64, 0x3000_0000] {
        assert_eq!(special.read8(base + 3).data, 0x33);
        assert_eq!(special.read8(base + 3 + 0x10_0000u64).data, 0x33, "the image repeats");
    }
    assert_eq!(special.read8(0x4000_0000).status, BUS_ERR, "nothing between the windows");
}

/// Two nodes' windows follow the numbers their Hubs publish, as the decode
/// table does: each shows its own memory and the other's at their numbers,
/// and a node that holds another number keeps its own memory below the
/// cached alias and shows node zero's above it.
#[test]
fn each_window_shows_both_nodes_memory_at_their_numbers() {
    use crate::ppmem::{bitmap_maps_at, MappedMemory, FINE_SHIFT};
    use std::sync::atomic::Ordering;
    // The windows of a four-node board, so node number two is within them.
    let hubs = board(2, [128, 0, 0, 0], 4, &flash(&[]));
    let update = |n: u64, bytes: u64| {
        calias(&hubs[1], bytes);
        number(&hubs[1], n);
    };
    // A jitv2 build on Windows maps nothing finer than 32 MiB.
    let small = crate::ppmem::map_granule().max(2 << 20);
    let (near, mid, far) = (0x1000, small + 0x1000, (96 << 20) + 0x1000);
    for (n, m) in hubs.iter().enumerate() {
        let n = n as u64;
        assert_eq!(m.decode.physical().banks()[0].write64(near, n << 4), BUS_OK);
        assert_eq!(m.decode.physical().banks()[0].write64(far, n << 4 | 1), BUS_OK);
        assert_eq!(m.decode.physical().banks()[0].write64(mid, n << 4 | 2), BUS_OK);
    }
    let seen = |n: usize, at: u64| {
        let sp = hubs[n].decode.physical().ppmem_space().unwrap();
        let (fine, regions) = sp.fine_table();
        let r = at >> FINE_SHIFT;
        let whole = r < regions && unsafe { (*fine.add((r / 64) as usize)).load(Ordering::Relaxed) } >> (r % 64) & 1 != 0;
        // The fine table holds everything the bitmap does.
        assert!(whole || !bitmap_maps_at(sp.mapped_bitmap(), sp.region_shift(), at));
        whole.then(|| unsafe { (*(sp.window_base().add(at as usize) as *const u64)).rotate_left(32) })
    };
    let alias = 64 << 20;
    let one = 1 << 32;
    // Both hold zero, the reset value: each window is its own memory.
    assert_eq!((seen(0, near), seen(1, near)), (Some(0x00), Some(0x10)));
    assert_eq!((seen(0, one + near), seen(1, one + near)), (None, None));
    // The second holds one.
    update(1, alias);
    assert_eq!(seen(0, near), Some(0x00));
    assert_eq!(seen(0, one + far), Some(0x11));
    assert_eq!(seen(1, near), Some(0x10));
    assert_eq!(seen(1, far), Some(0x01));
    assert_eq!(seen(1, one + near), Some(0x10));
    // A smaller alias shows more of node zero to the second node: 2 MiB,
    // or the host's `map_granule` where the window maps no finer.
    update(1, small);
    assert_eq!(seen(1, near), Some(0x10));
    assert_eq!(seen(1, mid), Some(0x02));
    // Two: one leaves both windows, and the second node's memory is at two.
    update(2, alias);
    assert_eq!((seen(0, one + near), seen(1, one + near)), (None, None));
    assert_eq!((seen(0, (2 << 32) + near), seen(1, (2 << 32) + far)), (Some(0x10), Some(0x11)));
    assert_eq!(seen(1, near), Some(0x10));
    // Back to zero.
    update(0, alias);
    assert_eq!((seen(0, near), seen(1, near), seen(1, far)), (Some(0x00), Some(0x10), Some(0x11)));
}

/// Every size bank 0 can be builds a memory map, and the boot flash's reset
/// address belongs to the flash unless a 512 MiB bank 0 reaches it, in which
/// case the bank holds it for cached references.
#[test]
fn every_bank_zero_size_builds_and_a_512_mib_one_keeps_the_reset_address() {
    for mb in [0, 64, 128, 256, 512] {
        let hub = one([mb, 0, 0, 0], &flash(&[0x12]));
        let map = memory_map([mb, 0, 0, 0]);
        assert!(map.contains(RESET_WINDOW), "{mb} MiB: the reset address is in the map");
        assert_eq!(map.is_ram(RESET_WINDOW), mb == 512, "{mb} MiB");
        let phys = hub.decode.physical();
        if mb == 512 {
            assert_eq!(phys.write32(RESET_WINDOW, 0xfeed_f00d), BUS_OK);
            assert_eq!(phys.read32(RESET_WINDOW).data, 0xfeed_f00d, "cached references reach the bank");
        } else {
            assert_eq!(phys.read8(RESET_WINDOW).data, 0x12, "{mb} MiB: the flash holds it");
        }
    }
}

/// The reset fetch is uncached, so it reaches the boot flash through the
/// Hub-special view and the processor's uncached bus whatever the size of
/// bank 0, a 512 MiB one that covers the address in cached space included.
#[test]
fn an_uncached_read_at_the_reset_address_reaches_the_flash_with_a_64_or_512_mib_bank_zero() {
    for mb in [64, 512] {
        let flash = flash(&0x3c1d_beefu32.to_be_bytes());
        let hub = one([mb, 0, 0, 0], &flash);
        let special = hub_special(hub.decode.own_memory(), &flash, hub.md.ecc_bus(), hub.md.directory_bus());
        assert_eq!(special.read32(RESET_WINDOW).data, 0x3c1d_beef, "{mb} MiB: the Hub-special view");
        let none: Arc<dyn BusDevice> = Arc::new(crate::sn::NoMemory(crate::devlog::LogModule::Hub));
        let bus = crate::sn::UncachedBus::new([special, none.clone(), none.clone(), none]);
        assert_eq!(bus.read32(RESET_WINDOW).data, 0x3c1d_beef, "{mb} MiB: the uncached bus");
    }
}

/// A renumbering on one processor while the other holds and releases a
/// line's page: the renumber maps the second node's window from the first
/// thread while the second reads the window's mapping list at a CACHE instruction.
#[test]
fn a_renumber_on_one_processor_does_not_disturb_the_others_line_pages() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Barrier;
    let hubs = board(2, [128, 0, 0, 0], 2, &flash(&[]));
    let sp = hubs[1].decode.physical().ppmem_space().unwrap();
    // The view has sinks like a processor's: plain words the space writes.
    let mut bitmap = Box::new(0u64);
    let mut key = Box::new(0u64);
    let v = sp.add_cached_view(&[&mut *bitmap as *mut u64]).unwrap();
    unsafe { sp.add_key_sink(v, &mut *key as *mut u64) };
    // One line per host granule (4 KB on Linux, 16 KB on Apple silicon,
    // 64 KB on Windows), so each hold takes a page no other line shares.
    let granule = crate::ppmem::granularity() as u64;
    let pages: Vec<u64> = (0..16).map(|n| 0x10_0000 + n * granule).collect();
    const RENUMBERS: u32 = 3000;
    let (start, done) = (Barrier::new(2), AtomicBool::new(false));
    let spurious = std::thread::scope(|s| {
        let other = s.spawn(|| {
            start.wait();
            let mut spurious = 0u32;
            let mut rounds = 0u32;
            while !done.load(Ordering::Acquire) || rounds < 1000 {
                // A CACHE instruction's worth.
                for &page in &pages {
                    spurious += u32::from(sp.hold_line_page(v, page) != Some(true));
                }
                for &page in &pages {
                    spurious += u32::from(!sp.release_line_page(v, page));
                }
                rounds += 1;
            }
            spurious
        });
        s.spawn(|| {
            start.wait();
            for i in 1..=RENUMBERS {
                // Node zero takes number one and gives it up: node one's
                // window gains and loses node zero's memory at 4 GiB.
                number(&hubs[0], u64::from(i & 1 == 0));
            }
            done.store(true, Ordering::Release);
        });
        other.join().unwrap()
    });
    assert_eq!(spurious, 0, "a line's page was found unmapped while another processor renumbered");
}

/// The generation counter the JIT watches for a page of the other node's memory is that node's
/// bank's own.
#[cfg(feature = "jitv2")]
#[test]
fn a_peer_page_hands_out_the_peer_banks_generation() {
    let hubs = board(2, [64, 0, 0, 0], 2, &flash(&[]));
    number(&hubs[1], 1);
    let bank = hubs[1].decode.physical().banks()[0].gen_ptr(0x2000);
    assert!(!bank.is_null());
    assert_eq!(hubs[0].decode.physical().gen_ptr((1 << 32) + 0x2000), bank);
}

/// A two-node board's processors reach memory at each node's number, so the map the JIT sizes its
/// page table from runs past 4 GiB. On the largest board (two nodes of four 512 MiB banks) it ends
/// at 6 GiB: a 12 MiB table.
#[test]
fn the_processors_memory_covers_every_node_number() {
    let (small, big) = (memory_map([64, 0, 0, 0]), memory_map([64, 0, 256, 0]));
    assert_eq!(machine_memory(std::slice::from_ref(&big), NODE_SHIFT), big);
    let two = machine_memory(&[small.clone(), big.clone()], NODE_SHIFT);
    assert_eq!(two.end(), NODE_SPAN + big.end());
    assert!(two.is_ram(NODE_SPAN + 0x1000));
    assert!(two.is_ram(NODE_SPAN + 0x4000_0000), "node 1 holds the wide layout whichever node it is");
    assert!(!two.contains(2 * NODE_SPAN));
    #[cfg(feature = "jitv2")]
    assert_eq!(crate::cpu::jitv2::jitv2::frames_for(&two), two.end() / 4096);
    let full = memory_map([512; BANKS]);
    let two = machine_memory(&[full.clone(), full], NODE_SHIFT);
    assert_eq!(two.end(), 6 << 30);
    #[cfg(feature = "jitv2")]
    crate::display_stop_tests::big_stack(move || {
        let empty = || Arc::new(crate::sn::NoMemory(crate::devlog::LogModule::Hub)) as Arc<dyn BusDevice>;
        let timers = Arc::new(crate::hptimer::TimerManager::new());
        let cpu = crate::cpu::build::build_cpu::<crate::cpu::mips_cache_shadow::R10000ShadowCache>(
            empty(), empty(), Arc::new(two), &crate::cpu::parts::R10000_180_SYSCLK_90, &timers, 1, 0, true,
        );
        assert_eq!(cpu.jitv2.lock().table_bytes(), 12 << 20);
    });
}

/// A Hub's node-number and cached-alias stores move memory under every
/// processor's view, so the remap parks the other processor whichever store it is.
#[test]
fn a_hubs_placing_stores_park_the_other_processor() {
    use crate::sn::tests::ParkProbe;
    let probe = ParkProbe::new();
    let hub = one([64, 0, 0, 0], &flash(&[]));
    hub.decode.set_world(probe.world());
    assert!(probe.parks_the_other(|| number(&hub, 1)), "the node number");
    assert!(probe.parks_the_other(|| calias(&hub, 16 << 10)), "the cached alias");
}

fn config(dir: &Path) -> MachineConfig {
    test_support::config(MachineProfile::Origin200Ip29, dir, 1)
}

/// A module's Config word, part table and built cache agree on the secondary cache (SS, bits 18:16,
/// 512 KB << SS).
#[test]
fn a_modules_config_word_part_and_cache_type_agree() {
    for (part, l2_kb) in [(&parts::R10000_180, 1024), (&parts::R12000_270, 4096)] {
        let config = parts::cpu_module(part).unwrap().config();
        assert_eq!(512u32 << ((config >> 16) & 7), l2_kb, "{}", part.name());
        assert_eq!(part.l2_kb, l2_kb, "{}", part.name());
        let model = crate::with_cpu!(part, true, false, C => <C as crate::cpu::mips_cache_v2::MipsCache>::L2_SIZE as u32 / 1024);
        assert_eq!(model, l2_kb, "{}", part.name());
    }
    assert_eq!(parts::R10000_180.count_hz(), 90_000_000);
    assert!(parts::cpu_module(&parts::R14000_600).is_none());
}

/// The Hub publishes straight into its processor's interrupt word, so the
/// machine lets go of the word before the processor goes: a cabled Hub, or any
/// other holder of this one, can still raise an interrupt afterwards.
#[test]
fn a_hub_that_outlives_its_machine_publishes_into_no_freed_processor() {
    big_stack(|| {
        let dir = TempPath::dir("ip27-o200-interrupt-word");
        let m = test_support::machine(&config(&dir)).unwrap();
        let hub = m.nodes[0].chip.clone();
        let cpu = Arc::downgrade(&m.node_cpus[0][0]);
        assert!(hub.pi.publishes_to(0), "the machine wired the processor's word");
        drop(m);
        assert_eq!(cpu.strong_count(), 0, "the processor is freed with the machine");
        assert!(!hub.pi.publishes_to(0), "and the Hub no longer holds its word");
        // Raising and unmasking an interrupt now publishes nowhere.
        assert_eq!(hub.write64(hub_regs::PI_INT_MASK0_A::OFFSET, !0), BUS_OK);
        hub.interrupt_target().set_line(3, true);
    });
}

/// The PROM of a board's second node keeps its stack in uncached Hub-special
/// memory above the 4 KiB cached alias it programs, and renumbers the node
/// between saving its return address there and reloading it. A store-conditional is a
/// compare-and-swap once processors share memory, so a cached reference by number reaches it.
#[test]
fn renumbering_a_lone_node_keeps_its_memory_and_lets_a_reference_by_number_compare_and_swap() {
    big_stack(|| {
        let dir = TempPath::dir("ip27-o200-renumber");
        let m = test_support::machine(&config(&dir)).unwrap();
        let hub = &m.nodes[0].chip;
        const SLOT: u64 = 0x01be_fe78;
        assert_eq!(hub.write64(crate::dev::hub::regs::PI_CALIAS_SIZE::OFFSET, 1), BUS_OK);
        assert_eq!(hub.cpu_view(crate::sn::UncachedSpace::HubSpecial).write64(SLOT, 0x5a5a), BUS_OK);
        assert_eq!(hub.write64(crate::dev::hub::regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
        assert_eq!(hub.cpu_view(crate::sn::UncachedSpace::HubSpecial).read64(SLOT).data, 0x5a5a, "the stack slot");
        let phys = m.nodes[0].chip.decode.physical();
        assert_eq!(phys.read64((1 << 32) + SLOT).data, 0x5a5a, "node 1 by number");
        assert_eq!(phys.read64(SLOT).data, 0x5a5a, "node zero");
        const WORD: u64 = (1 << 32) + 0x61_c000;
        assert_eq!(phys.write64(WORD, 0x99), BUS_OK);
        assert_eq!(phys.cas64(WORD, 0x99, 0x9a), Some(true));
        assert_eq!(phys.cas64(WORD, 0x99, 0x9b), Some(false));
        assert_eq!(phys.cas32(WORD + 4, 0x9a, 0x9c), Some(true));
        assert_eq!(hub.cpu_view(crate::sn::UncachedSpace::HubSpecial).read64(0x61_c000).data, 0x9c);
    });
}

/// A Hub node board of two processors, as the deskside's `nodes` node boards carry them.
fn two_slices(dir: &Path, profile: MachineProfile, nodes: u8) -> MachineConfig {
    let mut cfg = test_support::config(profile, dir, nodes);
    cfg.machine.cpus = Some(2);
    cfg
}

/// Each slice has its own pending mask and comparator, and each drives only
/// its own processor's lines: slice B's reach the node's second processor.
#[test]
fn slice_b_interrupts_and_comparator_reach_only_the_second_processor() {
    use std::sync::atomic::Ordering;
    big_stack(|| {
        let dir = TempPath::dir("ip27-slice-b-lines");
        let m = test_support::machine(&two_slices(&dir, MachineProfile::Origin2000Ip27, 2)).unwrap();
        // Cause IP2..IP5: bank 0, bank 1, the slice's comparator, profiling.
        let lines = |n: usize, s: usize| (unsafe { &*m.node_cpus[n][s].interrupts_ptr() }.load(Ordering::SeqCst) >> 10) & 0xf;
        for n in 0..2 {
            let hub = &m.nodes[n].chip;
            assert_eq!(m.node_cpus[n].len(), 2, "node {n}");
            assert_eq!(hub.read64(hub_regs::PI_CPU_PRESENT_B::OFFSET).data, 1, "node {n}");
            assert_eq!(hub.write64(hub_regs::PI_INT_MASK0_B::OFFSET, 1 << 5), BUS_OK);
            assert_eq!(hub.write64(hub_regs::PI_INT_PEND_MOD::OFFSET, crate::sn::MODIFY_SET | 5), BUS_OK);
            assert_eq!((lines(n, 0), lines(n, 1)), (0, 0b0001), "node {n}: bank 0 through B's mask");
            assert_eq!(hub.write64(hub_regs::PI_RT_EN_B::OFFSET, 1), BUS_OK);
            // A comparator written equal to the counter matches at once; the counter runs on
            // the host clock, so a store can land a tick late.
            for _ in 0..1000 {
                let now = hub.read64(hub_regs::PI_RT_COUNT::OFFSET).data;
                assert_eq!(hub.write64(hub_regs::PI_RT_COMPARE_B::OFFSET, now), BUS_OK);
                if hub.read64(hub_regs::PI_RT_PEND_B::OFFSET).data == 1 {
                    break;
                }
            }
            assert_eq!(hub.read64(hub_regs::PI_RT_PEND_A::OFFSET).data, 0, "node {n}");
            assert_eq!((lines(n, 0), lines(n, 1)), (0, 0b0101), "node {n}: B's comparator");
        }
    });
}

/// The PROM keeps its stack in primary data lines it tags itself, at the same
/// address on both processors of a node. Each processor's lines are its own:
/// slice B tagging the line after A stored there reads memory, not A's store.
#[test]
fn each_slice_keeps_its_own_cache_lines() {
    use crate::cpu::mips_isa::{CACH_PD, C_IST, OP_CACHE};
    const SP: u64 = 0x10_3ff0;
    const OUT: u64 = 0x2000;
    const TAG: u32 = ((SP >> 12) << 8 | 3 << 6) as u32;
    let (t0, t1, t2, t4, t5, t6) = (8, 9, 10, 12, 13, 14);
    let imm = |op: u32, rs: u32, rt: u32, v: u32| op << 26 | rs << 21 | rt << 16 | v;
    let dsll32 = |r: u32| r << 16 | r << 11 | 0x3c;
    let mtc0 = |rt: u32, rd: u32| 0x10 << 26 | 4 << 21 | rt << 16 | rd << 11;
    let words = [
        imm(0x0d, 0, t0, 0x80), // Status.KX
        mtc0(t0, 12),
        imm(0x0f, 0, t1, 0xa800), // t1 = SP in cached xkphys
        dsll32(t1),
        imm(0x0f, 0, t2, (SP >> 16) as u32),
        t1 << 21 | t2 << 16 | t1 << 11 | 0x25,
        imm(0x0d, t1, t1, (SP & 0xffff) as u32),
        imm(0x0f, 0, t2, TAG >> 16), // TagLo: the page, DirtyExclusive
        imm(0x0d, t2, t2, TAG & 0xffff),
        mtc0(t2, 28),
        imm(OP_CACHE, t1, C_IST | CACH_PD, 0),
        imm(0x37, t1, t4, 0), // ld t4, 0(t1)
        imm(0x0d, 0, t5, 0x5a5a),
        imm(0x3f, t1, t5, 0), // sd t5, 0(t1)
        imm(0x0f, 0, t6, 0x9600), // uncached memory
        dsll32(t6),
        imm(0x0d, t6, t6, OUT as u32),
        imm(0x3f, t6, t4, 0), // sd t4, OUT: what the line read
        0x1000_ffff,
        0,
    ];
    let payload: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
    big_stack(move || {
        let dir = TempPath::dir("ip27-slice-lines");
        let mut cfg = two_slices(&dir, MachineProfile::Origin200Ip29, 1);
        std::fs::write(dir.join("prom.bin"), &payload).unwrap();
        cfg.prom = dir.join("prom.bin").to_string_lossy().into_owned();
        let m = test_support::machine(&cfg).unwrap();
        let memory = m.nodes[0].chip.decode.physical();
        assert_eq!(memory.write64(OUT, 0xdead), BUS_OK);
        let [a, b] = &m.node_cpus[0][..] else { panic!("two processors") };
        a.step_n_inline(18).unwrap();
        b.step_n_inline(18).unwrap();
        assert_eq!(memory.read64(OUT).data, 0, "B's line holds memory, not A's store");
        assert_eq!(memory.read64(SP).data, 0, "neither store reached memory");
    });
}
