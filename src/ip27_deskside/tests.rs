use super::*;
use crate::config::MachineProfile;
use crate::dev::hub::reg::ni as hub_ni;
use crate::dev::hub::Space;
use crate::ip27_machine::test_support::{self, big_stack, Scratch};
use crate::traits::{BusDevice, BUS_ERR, BUS_OK};
use std::path::Path;

const ORIGIN_2000_PROFILE: MachineProfile = MachineProfile::Origin2000Ip27;
const ONYX2_PROFILE: MachineProfile = MachineProfile::Onyx2Ip27;

fn config(profile: MachineProfile, dir: &Path, nodes: u8) -> MachineConfig {
    test_support::config(profile, dir, nodes)
}

fn built(profile: MachineProfile, name: &str, nodes: u8) -> (Scratch, DesksideMachine) {
    let dir = Scratch::new(&format!("{}-{name}", profile.name()));
    let m = DesksideMachine::new(&config(profile, &dir.0, nodes)).unwrap();
    (dir, m)
}

#[test]
fn the_profiles_are_chosen_by_their_snake_case_names() {
    assert_eq!(MachineProfile::from_name("origin2000_ip27"), Ok(MachineProfile::Origin2000Ip27));
    assert_eq!(MachineProfile::Origin2000Ip27.name(), "origin2000_ip27");
    assert_eq!(MachineProfile::from_name("onyx2_ip27"), Ok(MachineProfile::Onyx2Ip27));
    assert_eq!(MachineProfile::Onyx2Ip27.name(), "onyx2_ip27");
}

/// The layout table is consistent with the Crossbow: every widget a layout
/// names is a port (8 to 15), no two things share one, and only a shared
/// Crossbow has a graphics slot.
#[test]
fn a_layout_names_distinct_crossbow_ports() {
    for layout in [&ORIGIN_2000, &ONYX2] {
        let mut ports = vec![layout.baseio_widget];
        ports.extend(layout.hub_widgets);
        ports.extend(layout.graphics_widget);
        assert!(ports.iter().all(|w| (8..16).contains(w)), "{ports:?}");
        let mut unique = ports.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), ports.len(), "{ports:?}");
        assert_ne!(layout.slot_bits[0], layout.slot_bits[1]);
        assert!(layout.crossbows == Crossbows::Shared || layout.graphics_widget.is_none());
    }
}

#[test]
fn the_deskside_has_one_or_two_node_boards() {
    let dir = Scratch::new("o2k-count");
    assert_eq!(config(ORIGIN_2000_PROFILE, &dir.0, 2).validate(), Ok(()));
    let err = Deskside::specs(&config(ORIGIN_2000_PROFILE, &dir.0, 3)).err().unwrap();
    assert!(err.contains("one or two node boards"), "{err}");
    assert!(Deskside::specs(&config(ORIGIN_2000_PROFILE, &dir.0, 0)).is_err());
}

#[test]
fn each_node_slot_drives_its_identity_into_the_hub() {
    big_stack(|| {
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "slots", 2);
        let slot = |n: usize| m.node_at(n).hub.read64(crate::dev::hub::reg::md::SLOTID_USTAT).data & 7;
        assert_eq!(slot(0), 7);
        assert_eq!(slot(1), 6);
    });
}

#[test]
fn the_flash_carries_no_machine_type() {
    big_stack(|| {
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "flash", 1);
        let word = |at: u64| m.node().hub.cpu_special().read32(0x1000_0000 + at).data;
        assert_eq!(word(0x90), 0);
        assert_eq!(word(0x8c), 15);
    });
}

#[test]
fn the_first_crossbow_carries_the_baseio_and_its_node() {
    big_stack(|| {
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "xbow", 1);
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
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "second", 2);
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
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "cable", 2);
        let (a, b) = (&m.node_at(0).hub, &m.node_at(1).hub);
        // The PROM numbers the second node 1; the first keeps 0.
        assert_eq!(b.write64(hub_ni::STATUS, 1 << 8), BUS_OK);
        let at = (1u64 << 32) + 0x1000;
        // A store through the first node's claim lands in the second's memory.
        assert_eq!(a.claim(Space::Memory).write64(at, 0x1122_3344_5566_7788), BUS_OK);
        assert_eq!(b.cpu_memory().read64(0x1000).data, 0x1122_3344_5566_7788);
        assert_eq!(a.cpu_memory().read64(0x1000).data, 0);
        // A cached reference from the first node to node 1 takes the same way.
        assert_eq!(m.node_at(0).memory.physical().read64(at).data, 0x1122_3344_5566_7788);
    });
}

/// Each node's processor reads both nodes' memory through its own ppmem
/// window once the PROM has numbered the nodes, as the first node's processor
/// always read its own: a cached access is a load on either node, not a walk
/// through the Hubs, so the second node runs code as fast as the first. That
/// holds for every bank size a node takes (256 MiB on the R12000 module) and
/// for the numbers the stock PROM gives, 0 and 1. A number past the window
/// reaches the same memory over the bus path. The alias below 64 MiB, the
/// reset size, stays the second node's own.
#[test]
fn both_processors_reach_both_nodes_memory_through_their_windows() {
    use crate::ppmem::{bitmap_maps_at, MappedMemory, FINE_SHIFT};
    use std::sync::atomic::Ordering;
    for mb in [64, 128, 256] {
        big_stack(move || {
            let dir = Scratch::new(&format!("windows-{mb}"));
            let mut cfg = config(ORIGIN_2000_PROFILE, &dir.0, 2);
            if mb == 256 {
                cfg.set_cpu(&crate::cpu::parts::R12000_270);
            }
            cfg.banks[0] = mb;
            cfg.banks[1] = mb;
            let m = DesksideMachine::new(&cfg).unwrap();
            let (a, b) = (&m.node_at(0).hub, &m.node_at(1).hub);
            // Bank 0 lies below the alias, bank 1 above it.
            let (near, far) = (0x1000, (512 << 20) + 0x1000);
            // What node `n`'s processor reads at `at` through its window, as
            // the cache finds it there: the bitmap, or on a miss the fine table.
            let window = |n: usize, at: u64| {
                let sp = m.node_at(n).memory.physical().ppmem_space().unwrap();
                let bitmap = unsafe { *m.cpu_at(n).ppmem_bitmap_ptr() };
                let (fine, regions) = sp.fine_table();
                let r = at >> FINE_SHIFT;
                let whole = r < regions && unsafe { (*fine.add((r / 64) as usize)).load(Ordering::Relaxed) } >> (r % 64) & 1 != 0;
                assert!(
                    bitmap_maps_at(bitmap, sp.region_shift(), at) || whole,
                    "{mb} MiB banks: node {n}'s processor reaches {at:#x} by the bus"
                );
                unsafe { (*(sp.window_base().add(at as usize) as *const u64)).rotate_left(32) }
            };
            for number in [1u64, 5] {
                assert_eq!(b.write64(hub_ni::STATUS, number << 8), BUS_OK);
                let other = number << 32;
                // Through the Hubs: node 0's memory at its number, node 1's at its.
                for (at, v) in [(near, 0x0a), (far, 0x0b), (other + near, 0x1a), (other + far, 0x1b)] {
                    assert_eq!(a.claim(Space::Memory).write64(at, v + number), BUS_OK);
                }
                if number >= 2 {
                    // Past the two nodes' window, the bus path answers.
                    let phys = m.node_at(0).memory.physical();
                    assert_eq!(phys.read64(other + near).data, 0x1a + number);
                    assert_eq!(phys.read64(other + far).data, 0x1b + number);
                    continue;
                }
                // The first node: its own memory at zero, the second's at its number.
                assert_eq!(window(0, near), 0x0a + number);
                assert_eq!(window(0, far), 0x0b + number);
                assert_eq!(window(0, other + far), 0x1b + number);
                // The second: its own below the alias and at its number, node
                // zero's above the alias.
                assert_eq!(window(1, near), 0x1a + number);
                assert_eq!(window(1, far), 0x0b + number);
                assert_eq!(window(1, other + near), 0x1a + number);
            }
            // A store through a window lands in the memory the Hubs reach.
            unsafe {
                let sp = m.node_at(1).memory.physical().ppmem_space().unwrap();
                *(sp.window_base().add(far as usize) as *mut u64) = 0x2b_u64.rotate_left(32);
            }
            assert_eq!(a.claim(Space::Memory).read64(far).data, 0x2b);
        });
    }
}

/// The BaseIO's Bridge sits behind the first node's Crossbow, so it reaches
/// that node's Hub as the host at widget 9, through the port's link. A
/// transfer or an interrupt packet naming widget 10, the second node's port
/// on its own Crossbow, has no host to land on and is dropped.
#[test]
fn the_bridge_reaches_its_nodes_hub_through_the_crossbow_port() {
    use crate::dev::bridge::{reg, DIRECT_WIDGET_SHIFT};
    big_stack(|| {
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "bridge", 2);
        let bridge = m.bridge().clone();
        let card = test_support::Card::new();
        bridge.bus().attach(crate::pci::DevFn { device: 7, function: 0 }, card.clone());
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        let dma = card.state.dma().unwrap().clone();
        let memory = |i: usize, off: u64| m.node_at(i).memory.physical().read32(off).data;
        let pending = |i: usize| m.node_at(i).hub.read64(crate::dev::hub::reg::pi::INT_PEND0).data;

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
            assert_eq!(m.node_at(0).hub.write64(crate::dev::hub::reg::pi::INT_PEND_MOD, 5), BUS_OK);
        }
    });
}

#[test]
fn two_nodes_share_one_crossbow_and_both_reach_the_baseio() {
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "two", 2);
        let (n1, n2) = (&m.node_at(0).hub, &m.node_at(1).hub);
        // Slot identity 3 in the first node slot, 2 in the second.
        assert_eq!(n1.read64(crate::dev::hub::reg::md::SLOTID_USTAT).data & 7, 3);
        assert_eq!(n2.read64(crate::dev::hub::reg::md::SLOTID_USTAT).data & 7, 2);
        // The flash carries no machine type: the slot code selects the table.
        assert_eq!(n1.cpu_special().read32(0x1000_0090).data, 0);

        for hub in [n1, n2] {
            let io = hub.claim(Space::Io);
            let id = |w: u64| io.read32((w << 24) + 4);
            // The one Crossbow, revision 2.
            assert_eq!(id(0).data, 0x2000_0001);
            // Both Hubs' ports, and the test widget in the graphics slot.
            assert_eq!(id(9).status, BUS_OK);
            assert_eq!(id(10).status, BUS_OK);
            assert_eq!(id(8).data, TEST_WIDGET_ID);
            // The test widget refuses anything else.
            assert_eq!(io.write32((8 << 24) + 0x10, 1), BUS_ERR);
            assert_eq!(io.read32((8 << 24) + 0x10).status, BUS_ERR);
            // Nothing is fitted at 11 to 14.
            for w in 11..15 {
                assert_eq!(id(w).status, BUS_ERR, "widget {w}");
            }
        }
        // The BaseIO-G answers at widget 15 from either Hub.
        let id = |hub: &Arc<Hub>| hub.claim(Space::Io).read32((15 << 24) + 4);
        assert_eq!(id(n2).status, BUS_OK);
        assert_eq!(id(n1).status, BUS_OK);
        assert_eq!(id(n1).data, id(n2).data);

        // A big-window probe of the peer's port, as the PROM does it, reads
        // the peer's node number once the Hubs are numbered.
        assert_eq!(n2.write64(hub_ni::STATUS, 1 << 8), BUS_OK);
        let io = n1.cpu_io();
        let bwin = crate::dev::hub::IO_BIG_WINDOWS;
        assert_eq!(n1.write64(crate::dev::hub::reg::io::BWIN_TABLE, (9 << 8) | 0x1000), BUS_OK);
        let status = io.read64(bwin + 0x0160_0000);
        assert_eq!(status.status, BUS_OK);
        assert_eq!((status.data >> 8) & 0x1ff, 1);
    });
}


/// The BaseIO-G's Bridge sends DMA and interrupts to the Hub at the widget
/// the guest programs into it, not to a fixed one: the second node's Hub
/// (widget 9) gets both when the guest names it, and the first's (10) when
/// the guest names that.
#[test]
fn the_bridge_sends_dma_and_interrupts_to_the_hub_the_guest_names() {
    use crate::dev::bridge::{reg, DIRECT_WIDGET_SHIFT};
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "bridge", 2);
        let bridge = m.bridge().clone();
        let card = test_support::Card::new();
        bridge.bus().attach(crate::pci::DevFn { device: 4, function: 0 }, card.clone());
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        let dma = card.state.dma().unwrap().clone();
        let memory = |i: usize, off: u64| m.node_at(i).memory.physical().read32(off).data;
        let pending = |i: usize| m.node_at(i).hub.read64(crate::dev::hub::reg::pi::INT_PEND0).data;

        // Node slot 2 is widget 9 and node slot 1 is widget 10.
        for (widget, node, other) in [(9u32, 1usize, 0usize), (10, 0, 1)] {
            set(reg::DIRECT_MAP, widget << DIRECT_WIDGET_SHIFT);
            let at = 0x1000 * u64::from(widget);
            assert!(dma.dma_write(0x8000_0000 + at, &[1, 2, 3, 4]).is_ok(), "widget {widget}");
            assert_ne!(memory(node, at), 0, "widget {widget}: the named node did not get it");
            assert_eq!(memory(other, at), 0, "widget {widget}: the other node saw it");

            // The interrupt goes the same way, to the vector the pin names.
            set(reg::INT_DEST_UPPER, widget << 16);
            set(reg::INT_DEST_LOWER, 0);
            set(reg::int_host(4), 5);
            set(reg::INT_ENABLE, 1 << 4);
            card.state.irq().unwrap().set(true);
            assert_eq!(pending(node) & 1 << 5, 1 << 5, "widget {widget}");
            assert_eq!(pending(other) & 1 << 5, 0, "widget {widget}: the other node heard it");
            card.state.irq().unwrap().set(false);
            // The guest's handler clears the pending bit.
            assert_eq!(m.node_at(node).hub.write64(crate::dev::hub::reg::pi::INT_PEND_MOD, 5), BUS_OK);
        }
    });
}

const WIDGET_PIPE: u8 = 8;

fn onyx2_with(dir: &Path, nodes: u8, board: crate::config::GraphicsBoard) -> MachineConfig {
    let mut cfg = config(ONYX2_PROFILE, dir, nodes);
    cfg.graphics.board = Some(board);
    cfg
}

/// A store to the pipe's aperture through a Hub's I/O space, retried while
/// the pipe's ring is full.
fn pipe_store(hub: &Arc<Hub>, off: u64, v: u32) {
    let io = hub.claim(Space::Io);
    for _ in 0..1_000_000 {
        match io.write32((u64::from(WIDGET_PIPE) << 24) + off, v) {
            crate::traits::BUS_BUSY => std::thread::yield_now(),
            status => return assert_eq!(status, BUS_OK, "store of {v:#x} at {off:#x}"),
        }
    }
    panic!("the pipe stayed busy");
}

fn pipe_commands(hub: &Arc<Hub>, words: &[u32]) {
    for &w in words {
        pipe_store(hub, 0xc1_0000, w);
    }
}

#[test]
fn the_onyx2_layout_puts_the_pipe_where_the_test_widget_stood() {
    assert_eq!(ONYX2.graphics_widget, Some(WIDGET_PIPE));
}

#[test]
fn a_two_node_onyx2_builds_with_a_pipe_that_answers_its_widget_identity() {
    use crate::config::GraphicsBoard;
    big_stack(|| {
        for board in [GraphicsBoard::Ir2, GraphicsBoard::Ir2x2] {
            let dir = Scratch::new("pipe");
            let m = DesksideMachine::new(&onyx2_with(&dir.0, 2, board)).unwrap();
            assert!(m.pipe().is_some());
            assert!(crate::workstation::Workstation::get_display(&m).is_some());
            for node in 0..2 {
                let io = m.node_at(node).hub.claim(Space::Io);
                let id = io.read32((u64::from(WIDGET_PIPE) << 24) + 4);
                assert_eq!(id.status, BUS_OK, "{board:?} from node {node}");
                // Revision in the top nibble; part, manufacturer and the
                // presence bit under it.
                assert_eq!(id.data & 0x0fff_ffff, 0x0c10_2555, "{board:?} from node {node}");
                assert_ne!(id.data, TEST_WIDGET_ID);
            }
        }
    });
}

#[test]
fn without_a_board_the_graphics_slot_holds_the_test_widget() {
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "nopipe", 1);
        assert!(m.pipe().is_none());
        assert!(crate::workstation::Workstation::get_display(&m).is_none());
    });
}

#[test]
fn the_pipe_names_its_boards_by_the_graphics_key() {
    use crate::config::GraphicsBoard;
    for (name, board) in [("ir2", GraphicsBoard::Ir2), ("ir2:1rm", GraphicsBoard::Ir2), ("ir2:2rm", GraphicsBoard::Ir2x2)] {
        assert_eq!(name.parse::<GraphicsBoard>(), Ok(board), "{name}");
    }
    for board in [GraphicsBoard::Ir2, GraphicsBoard::Ir2x2] {
        assert!(board.supports(MachineProfile::Onyx2Ip27));
        for other in [MachineProfile::OctaneIp30, MachineProfile::Origin2000Ip27, MachineProfile::Origin200Ip29, MachineProfile::IndyIp24, MachineProfile::Indigo2Ip22] {
            assert!(!board.supports(other), "{board:?} on {other:?}");
        }
        assert_eq!(board.as_str().parse::<GraphicsBoard>(), Ok(board));
    }
    assert_eq!(GraphicsBoard::Ir2.rm_count(), Some(1));
    assert_eq!(GraphicsBoard::Ir2x2.rm_count(), Some(2));
}

/// The 2E fits the deskside under the names `ir2e` and `ir2e:2rm`.
#[test]
fn the_deskside_takes_the_2e() {
    use crate::config::{GraphicsBoard, MachineConfig};
    for (name, board, rms) in [("ir2e", GraphicsBoard::Ir2e, 1), ("ir2e:2rm", GraphicsBoard::Ir2ex2, 2)] {
        assert_eq!(name.parse::<GraphicsBoard>(), Ok(board), "{name}");
        assert_eq!(board.rm_count(), Some(rms));
        assert!(board.supports(MachineProfile::Onyx2Ip27), "{name}");
        assert_eq!(board.as_str(), name);
    }
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = MachineProfile::Onyx2Ip27;
    cfg.graphics.board = Some(GraphicsBoard::Ir2e);
    assert!(cfg.validate().is_ok());
}

/// The pipe's DMA and interrupt packets reach the Hub at the widget the
/// guest programs into it. Each packet sets its vector's pending bit; the
/// pipe never clears one, so a second packet after the handler cleared the
/// bit raises it again.
#[test]
fn the_pipe_sends_dma_and_interrupts_to_the_hub_the_guest_names() {
    use crate::dev::hub::reg::pi;
    use crate::traits::Device;
    const WIDGET_CONTROL: u64 = 0x24;
    const INT_DEST_UPPER: u64 = 0x34;
    const INT_DEST_LOWER: u64 = 0x3c;
    const INT_VECTORS: u64 = 0x5c;
    const INT_CAUSE: u64 = 0x80_0238;
    const GE_VECTOR: u32 = 0x11;
    const REASON_PAGE: u64 = 0x30_0000;
    big_stack(|| {
        let dir = Scratch::new("route");
        let m = DesksideMachine::new(&onyx2_with(&dir.0, 2, crate::config::GraphicsBoard::Ir2)).unwrap();
        let pipe = m.pipe().unwrap().clone();
        pipe.start();
        let memory = |i: usize| m.node_at(i).memory.physical().read32(REASON_PAGE).data;
        let pending = |i: usize| m.node_at(i).hub.read64(pi::INT_PEND0).data;
        let post = |hub: &Arc<Hub>| {
            pipe_commands(hub, &[0x1f8, 0x103]);
            pipe.quiesce();
            pipe.resume();
        };

        // Node slot 2 is widget 9 and node slot 1 is widget 10.
        for (widget, node, other) in [(9u32, 1usize, 0usize), (10, 0, 1)] {
            let from = &m.node_at(other).hub;
            pipe_store(from, WIDGET_CONTROL, widget << 22);
            pipe_store(from, INT_DEST_UPPER, widget << 16 | 0x8000);
            pipe_store(from, INT_DEST_LOWER, 0x180_0090);
            pipe_store(from, INT_VECTORS, 0x3e_2210);
            pipe_commands(from, &[0x1f1, 0x1000_0000 | (REASON_PAGE >> 12) as u32, (REASON_PAGE & 0xfff) as u32, 0x4000]);

            post(from);
            assert_eq!(memory(node), 0x103, "widget {widget}: the named node did not get the reason");
            assert_eq!(memory(other), 0, "widget {widget}: the other node saw it");
            assert_eq!(pending(node) & 1 << GE_VECTOR, 1 << GE_VECTOR, "widget {widget}");
            assert_eq!(pending(other) & 1 << GE_VECTOR, 0, "widget {widget}: the other node heard it");

            // The handler acknowledges the cause and clears the pending bit;
            // the next reason raises it again.
            pipe_store(from, INT_CAUSE, 0x20);
            assert_eq!(m.node_at(node).hub.write64(pi::INT_PEND_MOD, u64::from(GE_VECTOR)), BUS_OK);
            assert_eq!(pending(node) & 1 << GE_VECTOR, 0);
            // The pipe holds the next reason until the driver clears the page word.
            assert_eq!(m.node_at(node).memory.physical().write32(REASON_PAGE, 0), BUS_OK);
            post(from);
            assert_eq!(pending(node) & 1 << GE_VECTOR, 1 << GE_VECTOR, "widget {widget}: the second packet");
            assert_eq!(m.node_at(node).hub.write64(pi::INT_PEND_MOD, u64::from(GE_VECTOR)), BUS_OK);
            pipe_store(from, INT_CAUSE, 0x20);
            assert_eq!(m.node_at(node).memory.physical().write32(REASON_PAGE, 0), BUS_OK);
        }
        pipe.stop();
    });
}

#[test]
fn a_dropped_machine_frees_its_pipe() {
    big_stack(|| {
        let dir = Scratch::new("droppipe");
        let m = DesksideMachine::new(&onyx2_with(&dir.0, 2, crate::config::GraphicsBoard::Ir2)).unwrap();
        let pipe = Arc::downgrade(m.pipe().unwrap());
        let hub = Arc::downgrade(&m.node_at(0).hub);
        crate::traits::Device::start(&**m.pipe().unwrap());
        drop(m);
        assert_eq!(pipe.strong_count(), 0, "the pipe");
        assert_eq!(hub.strong_count(), 0, "the Hub");
    });
}

/// With an I/O PROM image named, the Bridge's flash window at widget offset
/// 0xc0_0000 shows it, and without one nothing answers there.
#[test]
fn the_io_prom_image_fills_the_baseio_flash() {
    big_stack(|| {
        let dir = Scratch::new("onyx2-ioprom");
        let (_none, bare) = built(ONYX2_PROFILE, "noioprom", 1);
        assert_eq!(bare.bridge().read32(0xc0_0000).status, BUS_ERR);
        let image = dir.0.join("io6prom.img");
        std::fs::write(&image, [0xde, 0xad, 0xbe, 0xef, 0x01]).unwrap();
        let mut cfg = config(ONYX2_PROFILE, &dir.0, 1);
        cfg.onyx2.io_prom = Some(image.to_string_lossy().into_owned());
        let m = DesksideMachine::new(&cfg).unwrap();
        assert_eq!(m.bridge().read32(0xc0_0000).data, 0xdead_beef);
        assert_eq!(m.bridge().read32(0xc0_0004).data, 0x01ff_ffff);
        cfg.onyx2.io_prom = Some(dir.0.join("missing").to_string_lossy().into_owned());
        assert!(DesksideMachine::new(&cfg).err().unwrap().contains("onyx2.io_prom"));
    });
}

#[test]
fn a_dropped_machine_frees_its_hubs_memory_and_timers() {
    big_stack(|| {
        for profile in [ORIGIN_2000_PROFILE, ONYX2_PROFILE] {
            let (_dir, m) = built(profile, "drop", 2);
            let hubs: Vec<_> = (0..2).map(|i| Arc::downgrade(&m.node_at(i).hub)).collect();
            let memory: Vec<_> = (0..2).map(|i| Arc::downgrade(m.node_at(i).memory.physical())).collect();
            let timers: Vec<_> = (0..2).map(|i| Arc::downgrade(&m.node_at(i).timers)).collect();
            let cpus: Vec<_> = (0..2).map(|i| Arc::downgrade(m.cpu_at(i))).collect();
            let bridge = Arc::downgrade(m.bridge());
            drop(m);
            for (i, w) in hubs.iter().enumerate() {
                assert_eq!(w.strong_count(), 0, "{profile:?}: hub {i}");
            }
            for (i, w) in memory.iter().enumerate() {
                assert_eq!(w.strong_count(), 0, "{profile:?}: memory map {i}");
            }
            for (i, w) in timers.iter().enumerate() {
                assert_eq!(w.strong_count(), 0, "{profile:?}: timers {i}");
            }
            for (i, w) in cpus.iter().enumerate() {
                assert_eq!(w.strong_count(), 0, "{profile:?}: cpu {i}");
            }
            assert_eq!(bridge.strong_count(), 0, "{profile:?}: the Bridge");
        }
    });
}

/// The BaseIO-G's Bridge line carries the base board's record and the
/// MediaIO's, as two parts with distinct ROM numbers, and the Origin 2000's
/// BaseIO carries its own alone.
#[test]
fn the_bridge_carries_the_baseio_and_mediaio_records() {
    let roms = Deskside::board_record(&ONYX2).roms();
    assert_eq!(roms.len(), 2);
    assert_ne!(roms[0], roms[1], "a search tells the boards apart");
    assert_eq!(Deskside::board_record(&ORIGIN_2000).roms().len(), 1);
}

/// The BaseIO-G's Bridge has two IOC3s, at devices 2 and 6, the SCSI
/// controllers at 0 and 1 and RAD1 at 7; nothing at 5. The Origin 2000's
/// BaseIO has the one IOC3 and no audio chip.
#[test]
fn the_baseio_g_has_a_second_ioc3_and_the_audio_chip() {
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "mio", 1);
        let bus = m.bridge().bus();
        let id = |device| bus.config_read(crate::pci::DevFn { device, function: 0 }, 0, 4);
        assert_eq!(id(0), Some(0x1020_1077), "an ISP1040");
        assert_eq!(id(2), Some(0x0003_10a9), "the board IOC3");
        assert_eq!(id(6), Some(0x0003_10a9), "the MediaIO IOC3");
        assert_eq!(id(7), Some(0x0005_10a9), "RAD1");
        assert_eq!(id(5), None, "nothing at 5");
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "no-mio", 1);
        let bus = m.bridge().bus();
        let id = |device| bus.config_read(crate::pci::DevFn { device, function: 0 }, 0, 4);
        assert_eq!(id(2), Some(0x0003_10a9));
        assert_eq!((id(6), id(7)), (None, None), "no daughtercard");
    });
}

/// A SuperIO source raised on the MediaIO's IOC3 reaches Bridge input 6, and
/// one raised on the board's IOC3 reaches input 4. The table row's pin is the
/// only thing that sets either, and no guest reads it back from the chip.
#[test]
fn each_ioc3_super_io_interrupt_reaches_its_own_bridge_input() {
    use crate::dev::bridge::reg::INT_STATUS as BRIDGE_INT_STATUS;
    use crate::dev::ioc3::{int, reg as ioc3_reg};
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "sio-irq", 1);
        let bridge = m.bridge().clone();
        let bus = bridge.bus();
        // The keyboard controller answers a command byte with an
        // acknowledge, which raises the keyboard source.
        let raise = |device: u8| {
            let f = bus.function(crate::pci::DevFn { device, function: 0 }).unwrap();
            f.bar_write(0, ioc3_reg::INT_ENABLE_SET, int::KEYBOARD, 4);
            f.bar_write(0, 0xa8, 0xf4, 4);
            assert_ne!(f.bar_read(0, ioc3_reg::INT_STATUS, 4) & int::KEYBOARD, 0, "device {device}: source raised");
        };
        let pending = || bridge.register(BRIDGE_INT_STATUS) & 0xff;
        assert_eq!(pending(), 0);
        raise(6);
        assert_eq!(pending(), 1 << 6, "the MediaIO's interrupt is Bridge input 6 and no other");
        raise(2);
        assert_eq!(pending(), 1 << 6 | 1 << 4, "the board IOC3's is input 4");
    });
}
