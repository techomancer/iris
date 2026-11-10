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
    // Past the 64 MiB bank and in the empty bank 1: zero, writes dropped.
    assert_eq!(phys.write32(0x400_0000, 1), BUS_OK);
    assert_eq!(phys.read32(0x400_0000).data, 0);
    assert_eq!(phys.read32(0x2000_0000).data, 0);
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
    let zero = mem.node_zero();
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
