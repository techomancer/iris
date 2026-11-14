use super::*;
use crate::dev::hub::{fitted_mappings, Hub, HubConfig, NODE_SPAN};
use crate::dev::node_flash::NodeFlash;
use crate::physical::BankMapping;
use crate::sn_nodes::{machine_memory, SN0_NODE_SHIFT};

fn flash(first: &[u8]) -> Arc<NodeFlash> {
    let mut image = vec![0xff; 0x10_0000];
    image[..first.len()].copy_from_slice(first);
    NodeFlash::new(image).unwrap()
}

/// `banks` on each of `count` nodes, their Hubs connected, with the windows of
/// a board of `window_nodes` nodes.
fn board(count: usize, banks: [u32; BANKS], window_nodes: usize, boot: &Arc<NodeFlash>) -> Vec<Arc<Hub>> {
    let config = |n: usize| HubConfig { first_cpu: 2 * n, banks_mb: banks, nodes: window_nodes, ..Default::default() };
    let hubs: Vec<_> = (0..count).map(|n| Hub::new(config(n), Arc::new(|| 0), Some(boot.chip())).unwrap()).collect();
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

/// The layout skips empty banks, numbers the RAM of the fitted ones in order,
/// and a 512 MiB bank is four runs of 128 MiB at 128 MiB intervals, each
/// starting where the last ended in the bank's own storage.
#[test]
fn the_layout_numbers_fitted_banks_and_splits_a_large_one_in_four() {
    const MIB: u64 = 1 << 20;
    let runs = fitted_mappings([64, 0, 512, 0]);
    assert_eq!(runs.len(), 5);
    assert_eq!(runs[0], BankMapping { base: 0, end: 64 * MIB, bank: 0, src: 0 });
    for (n, run) in runs[1..].iter().enumerate() {
        let n = n as u64;
        assert_eq!(*run, BankMapping { base: 1024 * MIB + n * 128 * MIB, end: 1024 * MIB + (n + 1) * 128 * MIB, bank: 1, src: n * 128 * MIB });
    }
    assert!(fitted_mappings([0; BANKS]).is_empty());
}

#[test]
fn fitted_memory_answers_where_the_runs_are_and_nowhere_else() {
    let hub = one([64, 0, 128, 0], &flash(&[]));
    let phys = hub.decode.physical();
    assert_eq!(phys.write32(0x1000, 0xdead_beef), BUS_OK);
    assert_eq!(phys.read32(0x1000).data, 0xdead_beef);
    assert_eq!(phys.write32(0x4000_0008, 7), BUS_OK);
    assert_eq!(phys.read32(0x4000_0008).data, 7);
}

/// A bank that is not fitted, and the gap past a smaller one, fail every
/// access at every width: the firmware's sizing walk takes the bus error a
/// machine without that memory would give it.
#[test]
fn unfitted_memory_bus_errors_reads_and_writes() {
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
    assert_eq!(phys.write32(0x3ff_fffc, 5), BUS_OK, "the last word of the fitted bank still answers");
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
    let bus = crate::sn_nodes::UncachedBus::new([Arc::new(Tag(0x10)), Arc::new(Tag(0x20)), Arc::new(Tag(0x30)), Arc::new(Tag(0x40))]);
    use crate::sn_nodes::UncachedSpace::*;
    for (attr, space, tag) in [(0u64, HubSpecial, 0x10u8), (1, Io, 0x20), (2, MemSpecial, 0x30), (3, Memory, 0x40)] {
        assert_eq!(crate::sn_nodes::UncachedSpace::of(attr << 57), space);
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
    let special = hub_special(hub.decode.own_memory(), &flash, hub.ecc_bus(), hub.directory_bus());
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

/// Two nodes' tables follow the numbers their Hubs store: each shows its own
/// memory and the other's at its number; a node holding another number keeps its
/// own below the cached alias and shows node zero's above it.
#[test]
fn renumbering_maps_both_nodes_tables_through_the_hubs() {
    let hubs = board(2, [128, 0, 0, 0], 2, &flash(&[]));
    let (near, mid, far) = (0x1000, (32 << 20) + 0x1000, (64 << 20) + 0x1000);
    for (n, m) in hubs.iter().enumerate() {
        for (at, tag) in [(near, 0), (mid, 0x10), (far, 0x20)] {
            assert_eq!(m.decode.physical().banks()[0].write64(at, tag | n as u64), BUS_OK);
        }
    }
    let read = |n: usize, a: u64| {
        let r = hubs[n].decode.physical().read64(a);
        (r.status == BUS_OK).then_some(r.data)
    };
    // Both hold zero: each its own.
    assert_eq!((read(0, far), read(1, far)), (Some(0x20), Some(0x21)));
    assert_eq!(read(0, (1 << 32) + near), None);
    // Node 1 takes one: its own below the 64 MiB alias and at one, node
    // zero's above the alias; node 0 shows node 1's at one.
    number(&hubs[1], 1);
    assert_eq!((read(1, near), read(1, mid), read(1, far)), (Some(1), Some(0x11), Some(0x20)));
    assert_eq!((read(0, (1 << 32) + near), read(1, (1 << 32) + far)), (Some(1), Some(0x21)));
    // A smaller alias shows more of node zero's window.
    calias(&hubs[1], 32 << 20);
    assert_eq!((read(1, near), read(1, mid)), (Some(1), Some(0x10)));
    // Nobody holds zero: node zero's memory past the alias is nobody's.
    number(&hubs[0], 2);
    assert_eq!(read(1, mid), None);
    assert_eq!(read(1, (2 << 32) + near), Some(0));
    // Reset: both hold zero again.
    hubs[0].power_on();
    hubs[1].power_on();
    assert_eq!((read(1, mid), read(1, (2 << 32) + near)), (Some(0x11), None));
}

/// Two nodes' windows follow the numbers their Hubs publish, as the decode
/// table does: each shows its own memory and the other's at their numbers,
/// and a node that holds another number keeps its own memory below the
/// cached alias and shows node zero's above it. A number given up leaves
/// both windows. What the region bitmap does not hold whole, the second
/// number's memory here, the fine table does.
#[test]
fn each_window_shows_both_nodes_memory_at_their_numbers() {
    use crate::ppmem::{bitmap_maps_at, MappedMemory, FINE_SHIFT};
    use std::sync::atomic::Ordering;
    // The windows of a four-node board, so node number two fits.
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
        let special = hub_special(hub.decode.own_memory(), &flash, hub.ecc_bus(), hub.directory_bus());
        assert_eq!(special.read32(RESET_WINDOW).data, 0x3c1d_beef, "{mb} MiB: the Hub-special view");
        let none: Arc<dyn BusDevice> = Arc::new(crate::sn_nodes::UnfittedMemory(crate::devlog::LogModule::Hub));
        let bus = crate::sn_nodes::UncachedBus::new([special, none.clone(), none.clone(), none]);
        assert_eq!(bus.read32(RESET_WINDOW).data, 0x3c1d_beef, "{mb} MiB: the uncached bus");
    }
}

/// A renumbering on one processor while the other holds and releases a
/// line's page: the renumber maps the second node's window from the first
/// thread while the second reads the window's mapping list at a CACHE
/// instruction. Without the space's lock the list is rebuilt under the
/// reader and `hold_line_page` finds its page unmapped. (That failure is a
/// SIGILL that ends the whole test binary, not one failed assertion.)
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

/// The generation counter the JIT watches for a page of the other node's
/// memory is that node's bank's own.
#[cfg(feature = "jitv2")]
#[test]
fn a_peer_page_hands_out_the_peer_banks_generation() {
    let hubs = board(2, [64, 0, 0, 0], 2, &flash(&[]));
    number(&hubs[1], 1);
    let bank = hubs[1].decode.physical().banks()[0].gen_ptr(0x2000);
    assert!(!bank.is_null());
    assert_eq!(hubs[0].decode.physical().gen_ptr((1 << 32) + 0x2000), bank);
}

/// A two-node board's processors reach memory at each node's number, so the map
/// the JIT sizes its page table from has to run past 4 GiB.
#[test]
fn the_processors_memory_covers_every_node_number() {
    let (small, big) = (memory_map([64, 0, 0, 0]), memory_map([64, 0, 256, 0]));
    assert_eq!(machine_memory(&[big.clone()], SN0_NODE_SHIFT), big);
    let two = machine_memory(&[small.clone(), big.clone()], SN0_NODE_SHIFT);
    assert_eq!(two.end(), NODE_SPAN + big.end());
    assert!(two.is_ram(NODE_SPAN + 0x1000));
    assert!(two.is_ram(NODE_SPAN + 0x4000_0000), "node 1 holds the wide layout whichever node it is");
    assert!(!two.contains(2 * NODE_SPAN));
    #[cfg(feature = "jitv2")]
    assert_eq!(crate::cpu::jitv2::jitv2::frames_for(&two), two.end() / 4096);
}

/// The JIT's page table on an SN processor is 8 bytes a frame, sized from the end
/// of this map: on the largest board (two nodes of four 512 MiB banks) 6 GiB of
/// address space and a 12 MiB table. A change that grows it has to change this
/// number on purpose.
#[test]
fn the_processors_memory_on_the_largest_board_stays_at_six_gib() {
    let full = memory_map([512; BANKS]);
    let two = machine_memory(&[full.clone(), full], SN0_NODE_SHIFT);
    assert_eq!(two.end(), 6 << 30);
    #[cfg(feature = "jitv2")]
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            // The executor is large; the default test stack is not.
            let unfitted = || Arc::new(crate::sn_nodes::UnfittedMemory(crate::devlog::LogModule::Hub)) as Arc<dyn BusDevice>;
            let timers = Arc::new(crate::hptimer::TimerManager::new());
            let cpu = crate::cpu::build::build_cpu::<crate::cpu::mips_cache_shadow::R10000ShadowCache>(
                unfitted(), unfitted(), Arc::new(two), &crate::cpu::parts::R10000_180_SYSCLK_90, &timers, 1, 0, true,
            );
            assert_eq!(cpu.jitv2.lock().table_bytes(), 12 << 20);
        })
        .unwrap()
        .join()
        .expect("thread panicked");
}

/// A Hub's node-number and cached-alias stores move memory under every
/// processor's view, so the remap parks the other processor whichever store it is.
#[test]
fn a_hubs_placing_stores_park_the_other_processor() {
    use crate::sn_nodes::tests::ParkProbe;
    let probe = ParkProbe::new();
    let hub = one([64, 0, 0, 0], &flash(&[]));
    hub.decode.set_world(probe.world());
    assert!(probe.parks_the_other(|| number(&hub, 1)), "the node number");
    assert!(probe.parks_the_other(|| calias(&hub, 16 << 10)), "the cached alias");
}
