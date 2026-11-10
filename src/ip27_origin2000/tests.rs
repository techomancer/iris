use super::*;
use crate::ip27_machine::test_support::{self, big_stack, Scratch};
use crate::config::MachineProfile;
use crate::dev::hub::Space;
use crate::dev::{hub_md, hub_ni};
use crate::traits::{BusDevice, BUS_ERR, BUS_OK};
use std::path::Path;

fn config(dir: &Path, nodes: u8) -> MachineConfig {
    test_support::config(MachineProfile::Origin2000Ip27, dir, nodes)
}

fn built(name: &str, nodes: u8) -> (Scratch, Origin2000Machine) {
    let dir = Scratch::new(&format!("o2k-{name}"));
    let m = Origin2000Machine::new(&config(&dir.0, nodes)).unwrap();
    (dir, m)
}

#[test]
fn the_profile_is_chosen_by_its_snake_case_name() {
    assert_eq!(MachineProfile::from_name("origin2000_ip27"), Ok(MachineProfile::Origin2000Ip27));
    assert_eq!(MachineProfile::Origin2000Ip27.name(), "origin2000_ip27");
}

#[test]
fn the_deskside_has_one_or_two_node_boards() {
    let dir = Scratch::new("o2k-count");
    assert_eq!(config(&dir.0, 2).validate(), Ok(()));
    let err = Origin2000::specs(&config(&dir.0, 3)).err().unwrap();
    assert!(err.contains("one or two node boards"), "{err}");
    assert!(Origin2000::specs(&config(&dir.0, 0)).is_err());
}

#[test]
fn each_node_slot_drives_its_identity_into_the_hub() {
    big_stack(|| {
        let (_dir, m) = built("slots", 2);
        let slot = |n: usize| m.node_at(n).hub.read64(hub_md::reg::SLOTID_USTAT).data & 7;
        assert_eq!(slot(0), 7);
        assert_eq!(slot(1), 6);
    });
}

#[test]
fn the_flash_carries_no_machine_type() {
    big_stack(|| {
        let (_dir, m) = built("flash", 1);
        let word = |at: u64| m.node().hub.cpu_special().read32(0x1000_0000 + at).data;
        assert_eq!(word(0x90), 0);
        assert_eq!(word(0x8c), 15);
    });
}

#[test]
fn the_first_crossbow_carries_the_baseio_and_its_node() {
    big_stack(|| {
        let (_dir, m) = built("xbow", 1);
        let io = m.node().hub.claim(Space::Io);
        let id = |w: u64| io.read32((w << 24) + 4);
        // The Crossbow's identification: revision 2, bit 0 set.
        assert_eq!(id(0).data, 0x2000_0001);
        // The Bridge answers at widget 8, the Hub at its own port, 9.
        assert_eq!(id(8).status, BUS_OK);
        assert_ne!(id(8).data, id(9).data);
        // A port with nothing fitted fails.
        for w in [10, 11, 15] {
            assert_eq!(id(w).status, BUS_ERR, "widget {w}");
        }
        assert_eq!(io.write32((8 << 24) + crate::dev::bridge::reg::INT_ENABLE as u64 + 4, 0), BUS_OK);
    });
}

#[test]
fn the_second_node_is_the_host_of_a_crossbow_with_no_io() {
    big_stack(|| {
        let (_dir, m) = built("second", 2);
        let io = m.node_at(1).hub.claim(Space::Io);
        let id = |w: u64| io.read32((w << 24) + 4);
        assert_eq!(id(0).data, 0x2000_0001);
        assert_eq!(id(10).status, BUS_OK);
        assert_eq!(id(8).status, BUS_ERR, "the BaseIO is on the first node's Crossbow");
        assert_eq!(id(9).status, BUS_ERR);
    });
}

#[test]
fn the_cable_joins_the_nodes_once_they_are_numbered() {
    big_stack(|| {
        let (_dir, m) = built("cable", 2);
        let (a, b) = (&m.node_at(0).hub, &m.node_at(1).hub);
        // The PROM numbers the second node 1; the first keeps 0.
        assert_eq!(b.write64(hub_ni::reg::STATUS, 1 << 8), BUS_OK);
        let at = (1u64 << 32) + 0x1000;
        // A store through the first node's claim lands in the second's memory.
        assert_eq!(a.claim(Space::Memory).write64(at, 0x1122_3344_5566_7788), BUS_OK);
        assert_eq!(b.cpu_memory().read64(0x1000).data, 0x1122_3344_5566_7788);
        assert_eq!(a.cpu_memory().read64(0x1000).data, 0);
        // A cached reference from the first node to node 1 takes the same way.
        assert_eq!(m.node_at(0).memory.physical().read64(at).data, 0x1122_3344_5566_7788);
    });
}

#[test]
fn a_dropped_machine_frees_its_hubs_memory_and_timers() {
    big_stack(|| {
        let (_dir, m) = built("drop", 2);
        let hubs: Vec<_> = (0..2).map(|i| Arc::downgrade(&m.node_at(i).hub)).collect();
        let memory: Vec<_> = (0..2).map(|i| Arc::downgrade(m.node_at(i).memory.physical())).collect();
        let timers: Vec<_> = (0..2).map(|i| Arc::downgrade(&m.node_at(i).timers)).collect();
        let cpus: Vec<_> = (0..2).map(|i| Arc::downgrade(m.cpu_at(i))).collect();
        let bridge = Arc::downgrade(m.bridge());
        drop(m);
        for (i, w) in hubs.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "hub {i}");
        }
        for (i, w) in memory.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "memory map {i}");
        }
        for (i, w) in timers.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "timers {i}");
        }
        for (i, w) in cpus.iter().enumerate() {
            assert_eq!(w.strong_count(), 0, "cpu {i}");
        }
        assert_eq!(bridge.strong_count(), 0, "the Bridge");
    });
}

/// The BaseIO's Bridge sits behind the first node's Crossbow, so it reaches
/// that node's Hub as the host at widget 9, through the port's link. A
/// transfer or an interrupt packet naming widget 10, the second node's port
/// on its own Crossbow, has no host to land on and is dropped.
#[test]
fn the_bridge_reaches_its_nodes_hub_through_the_crossbow_port() {
    use crate::dev::bridge::{reg, DIRECT_WIDGET_SHIFT};
    big_stack(|| {
        let (_dir, m) = built("bridge", 2);
        let bridge = m.bridge().clone();
        let card = test_support::Card::new();
        bridge.bus().attach(crate::pci::DevFn { device: 7, function: 0 }, card.clone());
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        let dma = card.state.dma().unwrap().clone();
        let memory = |i: usize, off: u64| m.node_at(i).memory.physical().read32(off).data;
        let pending = |i: usize| m.node_at(i).hub.read64(crate::dev::hub_pi::reg::INT_PEND0).data;

        for (widget, lands) in [(9u32, true), (10, false)] {
            set(reg::DIRECT_MAP, widget << DIRECT_WIDGET_SHIFT);
            let at = 0x1000 * u64::from(widget);
            let _ = dma.dma_write(0x8000_0000 + at, &[1, 2, 3, 4]);
            assert_eq!(memory(0, at) != 0, lands, "widget {widget}: DMA");
            assert_eq!(memory(1, at), 0, "widget {widget}: the second node saw it");

            set(reg::INT_DEST_UPPER, widget << 16);
            set(reg::INT_DEST_LOWER, 0);
            set(reg::int_host(7), 5);
            set(reg::INT_ENABLE, 1 << 7);
            card.state.irq().unwrap().set(true);
            assert_eq!(pending(0) & 1 << 5 != 0, lands, "widget {widget}: interrupt");
            assert_eq!(pending(1) & 1 << 5, 0, "widget {widget}: the second node heard it");
            card.state.irq().unwrap().set(false);
            assert_eq!(m.node_at(0).hub.write64(crate::dev::hub_pi::reg::INT_PEND_MOD, 5), BUS_OK);
        }
    });
}
