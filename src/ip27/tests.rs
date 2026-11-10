use super::*;
use crate::dev::node_flash::{self, NodeFlash};

#[test]
fn a_small_bank_is_one_run_and_a_large_one_four() {
    assert_eq!(bank_runs(0, 64), vec![(0, 0x400_0000)]);
    assert_eq!(bank_runs(1, 128), vec![(0x2000_0000, 0x800_0000)]);
    assert_eq!(
        bank_runs(2, 256),
        (0..4).map(|i| (0x4000_0000 + i * 0x800_0000, 0x400_0000)).collect::<Vec<_>>()
    );
    assert!(bank_runs(3, 0).is_empty());
}

/// The layout skips empty banks, numbers the RAM of the fitted ones in order,
/// and a 512 MiB bank is four runs of 128 MiB at 128 MiB intervals, each
/// starting where the last ended in the bank's own storage.
#[test]
fn the_layout_numbers_fitted_banks_and_splits_a_large_one_in_four() {
    const MIB: u64 = 1 << 20;
    let runs = layout([64, 0, 512, 0]);
    assert_eq!(runs.len(), 5);
    assert_eq!(runs[0], Run { bank: 0, index: 0, at: 0, len: 64 * MIB, src: 0 });
    for (n, run) in runs[1..].iter().enumerate() {
        let n = n as u64;
        assert_eq!(*run, Run { bank: 2, index: 1, at: 1024 * MIB + n * 128 * MIB, len: 128 * MIB, src: n * 128 * MIB });
    }
    assert!(layout([0; BANKS]).is_empty());
}

#[test]
fn fitted_memory_answers_where_the_runs_are_and_nowhere_else() {
    let mem = NodeMemory::new([64, 0, 128, 0]).unwrap();
    let phys = mem.physical();
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
    let mem = NodeMemory::new([64, 0, 128, 0]).unwrap();
    let phys = mem.physical();
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
    let mem = NodeMemory::new([64, 0, 0, 0]).unwrap();
    let mut image = vec![0xff; 0x10_0000];
    image[0] = 0x12;
    let flash = NodeFlash::new(image).unwrap();
    mem.map_reset_view(&flash);
    assert_eq!(mem.physical().read8(0x1fc0_0000).data, 0x12);
    let map = mem.memory_map();
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
    let bus = Ip27Bus::new(Arc::new(Tag(0x10)), Arc::new(Tag(0x20)), Arc::new(Tag(0x30)), Arc::new(Tag(0x40)));
    for (attr, tag) in [(0u64, 0x10u8), (1, 0x20), (2, 0x30), (3, 0x40)] {
        assert_eq!(bus.read8((attr << 57) | 0x1_0000_0001).data, tag ^ 1, "attribute {attr}");
    }
    // A plain compatibility address has no attribute and reaches space 0.
    assert_eq!(bus.read8(0x1fc0_0001).data, 0x10 ^ 1);
}

#[test]
fn a_back_door_stores_two_bytes_of_each_entry_and_folds_a_bank() {
    let door = Backdoor::new([64, 0, 0, 0]);
    door.write64(0, 0x1122_3344_5566_7788);
    assert_eq!(door.read64(0).data, 0x0000_0000_0000_7788);
    // 64 MiB of memory keeps 16 MiB of storage, repeated across the stride.
    assert_eq!(door.read64(0x100_0000).data, 0x7788);
    // A bank that is not fitted reads the pattern and drops writes.
    door.write64(0x800_0000, 5);
    assert_eq!(door.read64(0x800_0000).data, 0xa5a5_a5a5_a5a5_a5a5);
    assert_eq!(door.read8(0x800_0000).data, 0xa5);
}

#[test]
fn hub_special_repeats_memory_and_the_flash_at_the_named_node_offsets_too() {
    let mem = NodeMemory::new([64, 0, 0, 0]).unwrap();
    mem.physical().write32(0x100, 0xcafe_f00d);
    let mut image = vec![0xff; 0x10_0000];
    image[3] = 0x33;
    let flash = NodeFlash::new(image).unwrap();
    let special = hub_special(mem.physical().clone(), &flash, mem.back_door(), mem.back_door());
    for base in [0u64, 0x2000_0000] {
        assert_eq!(special.read32(base + 0x100).data, 0xcafe_f00d);
    }
    for base in [0x1000_0000u64, 0x3000_0000] {
        assert_eq!(special.read8(base + 3).data, 0x33);
        assert_eq!(special.read8(base + 3 + 0x10_0000u64).data, 0x33, "the image repeats");
    }
    assert_eq!(special.read8(0x4000_0000).status, BUS_ERR, "nothing between the windows");
}

/// A view that answers every read with its address, as the Hub's
/// node-qualified memory does for the Hub it names.
struct Echo;
impl BusDevice for Echo {
    fn read64(&self, a: u64) -> BusRead64 {
        BusRead64::ok(a)
    }
}

#[test]
fn cached_references_to_other_node_numbers_go_to_the_view() {
    let mem = NodeMemory::new([64, 0, 0, 0]).unwrap();
    mem.route_nodes(Arc::new(Echo));
    let phys = mem.physical();
    // Node zero's window is the node's own memory.
    assert_eq!(phys.write64(0x1000, 0x55), BUS_OK);
    assert_eq!(phys.read64(0x1000).data, 0x55);
    // Every other node number reaches the view with the number still in the
    // address.
    assert_eq!(phys.read64(0x1_0000_1000).data, 0x1_0000_1000);
    assert_eq!(phys.read64(0x7_0000_2000).data, 0x7_0000_2000);
    assert_eq!(phys.read64(0xff_0000_3000).data, 0xff_0000_3000);
}

#[test]
fn a_node_that_holds_another_number_sends_node_zero_beyond_the_cached_alias_to_the_view() {
    let mem = NodeMemory::new([128, 0, 0, 0]).unwrap();
    let phys = mem.physical();
    let echo: Arc<dyn BusDevice> = Arc::new(Echo);
    let zero = mem.node_zero(Routing::new(), &Arc::new(Windows::default()));
    mem.route_nodes(echo.clone());
    let alias = 64 << 20;
    assert_eq!(phys.write64(alias + 0x1000, 0x77), BUS_OK);
    // Holding zero changes nothing.
    zero.update(0, alias);
    assert_eq!(phys.read64(alias + 0x1000).data, 0x77);
    // Holding one: the alias stays local, memory beyond it reaches the view
    // with node zero in the address.
    zero.update(1, alias);
    assert_eq!(phys.write64(0x1000, 0x55), BUS_OK);
    assert_eq!(phys.read64(0x1000).data, 0x55);
    assert_eq!(phys.read64(alias + 0x1000).data, alias + 0x1000);
    // A larger alias keeps more of it local, a smaller one less.
    zero.update(1, 96 << 20);
    assert_eq!(phys.read64(alias + 0x1000).data, 0x77);
    assert_eq!(phys.read64((96 << 20) + 0x1000).data, (96 << 20) + 0x1000);
    // The number back to zero returns all of it.
    zero.update(0, alias);
    assert_eq!(phys.read64(alias + 0x1000).data, 0x77);
    assert_eq!(phys.read64(0x1000).data, 0x55);
}

/// Two nodes' windows follow the numbers their Hubs publish, as the decode
/// table does: each shows its own memory and the other's at their numbers,
/// and a node that holds another number keeps its own memory below the
/// cached alias and shows node zero's above it. A number given up leaves
/// both windows. What the region bitmap does not hold whole, the second
/// number's memory here, the fine table does.
#[test]
fn each_window_shows_both_nodes_memory_at_their_numbers() {
    use crate::ppmem::{bitmap_maps_at, FINE_SHIFT};
    use std::sync::atomic::Ordering;
    // The windows of a four-node board, so node number two fits.
    let (size, shift) = window(4);
    let windows = Arc::new(Windows::default());
    let echo: Arc<dyn BusDevice> = Arc::new(Echo);
    let mems: Vec<_> = (0..2).map(|_| NodeMemory::with_window([128, 0, 0, 0], size, shift).unwrap()).collect();
    let zeros: Vec<_> = mems
        .iter()
        .map(|m| {
            m.route_nodes(echo.clone());
            m.node_zero(Routing::new(), &windows)
        })
        .collect();
    let (near, mid, far) = (0x1000, (2 << 20) + 0x1000, (96 << 20) + 0x1000);
    for (n, m) in mems.iter().enumerate() {
        let n = n as u64;
        assert_eq!(m.physical().banks()[0].write64(near, n << 4), BUS_OK);
        assert_eq!(m.physical().banks()[0].write64(far, n << 4 | 1), BUS_OK);
        assert_eq!(m.physical().banks()[0].write64(mid, n << 4 | 2), BUS_OK);
    }
    let seen = |n: usize, at: u64| {
        let sp = mems[n].physical().ppmem_space().unwrap();
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
    zeros[1].update(1, alias);
    assert_eq!(seen(0, near), Some(0x00));
    assert_eq!(seen(0, one + far), Some(0x11));
    assert_eq!(seen(1, near), Some(0x10));
    assert_eq!(seen(1, far), Some(0x01));
    assert_eq!(seen(1, one + near), Some(0x10));
    // A smaller alias shows more of node zero to the second node. It is a
    // multiple of 2 MiB, which the jitv2 build maps at the finest.
    zeros[1].update(1, 2 << 20);
    assert_eq!(seen(1, near), Some(0x10));
    assert_eq!(seen(1, mid), Some(0x02));
    // Two: one leaves both windows, and the second node's memory is at two.
    zeros[1].update(2, alias);
    assert_eq!((seen(0, one + near), seen(1, one + near)), (None, None));
    assert_eq!((seen(0, (2 << 32) + near), seen(1, (2 << 32) + far)), (Some(0x10), Some(0x11)));
    assert_eq!(seen(1, near), Some(0x10));
    // Back to zero.
    zeros[1].update(0, alias);
    assert_eq!((seen(0, near), seen(1, near), seen(1, far)), (Some(0x00), Some(0x10), Some(0x11)));
}

/// Every size bank 0 can be builds a memory map, and the boot flash's reset
/// address belongs to the flash unless a 512 MiB bank 0 reaches it, in which
/// case the bank holds it for cached references.
#[test]
fn every_bank_zero_size_builds_and_a_512_mib_one_keeps_the_reset_address() {
    for mb in [0, 64, 128, 256, 512] {
        let mut image = vec![0xff; 0x10_0000];
        image[0] = 0x12;
        let flash = NodeFlash::new(image).unwrap();
        let mem = NodeMemory::new([mb, 0, 0, 0]).unwrap();
        mem.map_reset_view(&flash);
        let map = mem.memory_map();
        assert!(map.contains(RESET_VIEW), "{mb} MiB: the reset address is in the map");
        assert_eq!(map.is_ram(RESET_VIEW), mb == 512, "{mb} MiB");
        let phys = mem.physical();
        if mb == 512 {
            assert_eq!(phys.write32(RESET_VIEW, 0xfeed_f00d), BUS_OK);
            assert_eq!(phys.read32(RESET_VIEW).data, 0xfeed_f00d, "cached references reach the bank");
        } else {
            assert_eq!(phys.read8(RESET_VIEW).data, 0x12, "{mb} MiB: the flash holds it");
        }
    }
}

/// The reset fetch is uncached, so it reaches the boot flash through the
/// Hub-special view and the processor's uncached bus whatever the size of
/// bank 0, a 512 MiB one that covers the address in cached space included.
#[test]
fn an_uncached_read_at_the_reset_address_reaches_the_flash_with_a_64_or_512_mib_bank_zero() {
    for mb in [64, 512] {
        let mut image = vec![0xff; 0x10_0000];
        image[..4].copy_from_slice(&0x3c1d_beefu32.to_be_bytes());
        let flash = NodeFlash::new(image).unwrap();
        let mem = NodeMemory::new([mb, 0, 0, 0]).unwrap();
        mem.map_reset_view(&flash);
        let special = hub_special(mem.physical().clone(), &flash, mem.back_door(), mem.back_door());
        assert_eq!(special.read32(RESET_VIEW).data, 0x3c1d_beef, "{mb} MiB: the Hub-special view");
        let none: Arc<dyn BusDevice> = Arc::new(Echo);
        let bus = Ip27Bus::new(special, none.clone(), none.clone(), none);
        assert_eq!(bus.read32(RESET_VIEW).data, 0x3c1d_beef, "{mb} MiB: the uncached bus");
    }
}

/// A board reserves host address space for the node numbers it can hold, not
/// for all 256, and a number past the window leaves it for the bus path.
#[test]
fn the_window_spans_the_next_power_of_two_node_numbers() {
    assert_eq!(window(1), (crate::ppmem::WINDOW_SIZE, crate::ppmem::BITMAP_SHIFT));
    assert_eq!(window(2), (2 * NODE_SPAN, 27));
    assert_eq!(window(3), (4 * NODE_SPAN, 27));
    assert_eq!(window(4), (4 * NODE_SPAN, 27));
    assert_eq!(window(200), (256 * NODE_SPAN, 27));
}

/// Two nodes on a two-node board, each with a window: what the renumbering
/// test below drives.
fn two_nodes() -> (Arc<Windows>, Vec<NodeMemory>, Vec<Arc<NodeZero>>, Vec<Arc<Routing>>) {
    let (size, shift) = window(2);
    let windows = Arc::new(Windows::default());
    let echo: Arc<dyn BusDevice> = Arc::new(Echo);
    let mems: Vec<_> = (0..2).map(|_| NodeMemory::with_window([128, 0, 0, 0], size, shift).unwrap()).collect();
    let routing: Vec<_> = (0..2).map(|_| Routing::new()).collect();
    let zeros = mems
        .iter()
        .enumerate()
        .map(|(n, m)| {
            m.route_nodes(echo.clone());
            m.node_zero(routing[n].clone(), &windows)
        })
        .collect();
    (windows, mems, zeros, routing)
}

/// A renumbering on one processor while the other holds and releases a
/// line's page: the renumber lays the second node's window from the first
/// thread while the second reads the window's mapping list at a CACHE
/// instruction. Without the space's lock the list is rebuilt under the
/// reader and `hold_line_page` finds its page unmapped. (That failure is a
/// SIGILL that ends the whole test binary, not one failed assertion.)
#[test]
fn a_renumber_on_one_processor_does_not_disturb_the_others_line_pages() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Barrier;
    let (_windows, mems, zeros, routing) = two_nodes();
    let sp = mems[1].physical().ppmem_space().unwrap();
    // The view has sinks like a processor's: plain words the space writes.
    let mut bitmap = Box::new(0u64);
    let mut key = Box::new(0u64);
    let v = sp.add_cached_view(&[&mut *bitmap as *mut u64]).unwrap();
    unsafe { sp.add_key_sink(v, &mut *key as *mut u64) };
    let pages: Vec<u64> = (0..16).map(|n| 0x10_0000 + n * 0x1000).collect();
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
                routing[0].set(u16::from(i & 1 == 0), 64 << 20);
                zeros[0].refresh();
            }
            done.store(true, Ordering::Release);
        });
        other.join().unwrap()
    });
    assert_eq!(spurious, 0, "a line's page was found unmapped while another processor renumbered");
}
