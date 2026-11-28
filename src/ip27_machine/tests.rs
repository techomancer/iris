use super::*;
use crate::test_temp::TempPath;
use crate::config::MachineProfile;
use crate::dev::hub::regs as hub_regs;
use crate::ip27::test_support::{self, big_stack, config};
use crate::dev::hub::Hub;
use crate::dev::node_flash::NodeFlash;
use crate::sn::Machine;
use crate::workstation::Workstation;
use crate::traits::{BusDevice, BUS_ERR, BUS_OK};
use std::path::Path;
const ORIGIN_2000_PROFILE: MachineProfile = MachineProfile::Origin2000Ip27;
const ONYX2_PROFILE: MachineProfile = MachineProfile::Onyx2Ip27;
fn built(profile: MachineProfile, name: &str, nodes: u8) -> (TempPath, Machine<Hub, NodeFlash>) {
    let dir = TempPath::dir(&format!("ip27-{}-{name}", profile.name()));
    let m = test_support::machine(&config(profile, &dir, nodes)).unwrap();
    (dir, m)
}
/// A Bridge hosted by `hosts` (node, Hub widget) on one Crossbow, built as `onyx2` and `origin2000`
/// build the BaseIO's. A test card goes on its bus, which the machine's own Bridge would not take.
fn bridge_on(cfg: &MachineConfig, m: &Machine<Hub, NodeFlash>, hosts: &[(usize, u8)], baseio: u8) -> Arc<Bridge> {
    let boards: Vec<_> = hosts.iter().map(|&(n, w)| (&m.nodes[n], w)).collect();
    let xbow = build_crossbow(&boards, 1 << baseio);
    let up = xbow.upstream(boards.iter().map(|(n, w)| n.chip.host_port(*w)).collect());
    let bridge = build_baseio(cfg, up, &[]).unwrap();
    xbow.serve_bridge(baseio, &bridge);
    bridge
}
/// A reset of a two-node Onyx2 puts both Hubs back as the PROM finds them: node number zero, the
/// processor enabled, the link to the other Hub up. The front ends see both processors.
#[test]
fn a_two_node_reset_returns_both_hubs_to_node_zero_with_the_processor_enabled_and_the_link_up() {
    big_stack(|| {
        let status = |m: &Machine<Hub, NodeFlash>, n: usize| m.nodes[n].chip.read64(hub_regs::NI_STATUS_REV_ID::OFFSET).data;
        let (_dir, mut m) = built(ONYX2_PROFILE, "reset", 2);
        assert_eq!(m.cpu_counters().len(), 2);
        assert_eq!(m.nodes[1].chip.write64(hub_regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
        assert_eq!((status(&m, 1) & hub_regs::NI_STATUS_REV_ID::NODE_ID::MASK) >> hub_regs::NI_STATUS_REV_ID::NODE_ID::SHIFT, 1);
        m.reset();
        for n in 0..2 {
            let chip = &m.nodes[n].chip;
            assert_eq!((status(&m, n) & hub_regs::NI_STATUS_REV_ID::NODE_ID::MASK) >> hub_regs::NI_STATUS_REV_ID::NODE_ID::SHIFT, 0, "node {n}");
            assert_eq!(chip.read64(hub_regs::PI_CPU_ENABLE_A::OFFSET).data, 1, "node {n}");
            assert_ne!(status(&m, n) & hub_regs::NI_STATUS_REV_ID::LINK_UP::MASK, 0, "node {n}");
        }
        m.stop();
    });
}
#[test]
fn each_node_slot_drives_its_identity_into_the_hub() {
    big_stack(|| {
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "slots", 2);
        let slot = |n: usize| m.nodes[n].chip.read64(crate::dev::hub::regs::MD_SLOTID_USTAT::OFFSET).data & 7;
        assert_eq!(slot(0), 7);
        assert_eq!(slot(1), 6);
    });
}
#[test]
fn the_flash_carries_no_machine_type() {
    big_stack(|| {
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "flash", 1);
        let word = |at: u64| m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::HubSpecial).read32(0x1000_0000 + at).data;
        assert_eq!(word(0x90), 0);
        assert_eq!(word(0x8c), 15);
    });
}
/// The Origin 2000's first Crossbow carries the BaseIO (widget 8) and its node (9); the second
/// node is the host of a Crossbow with no I/O (widget 10). A port with nothing attached fails.
#[test]
fn each_origin_2000_crossbow_answers_only_what_it_carries() {
    big_stack(|| {
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "xbow", 2);
        // (node, widgets that answer, widgets that fail)
        for (node, answers, fails) in [(0, &[8, 9][..], &[10, 11, 15][..]), (1, &[10], &[8, 9])] {
            let io = m.nodes[node].chip.cpu_view(crate::sn::UncachedSpace::Io);
            let id = |w: u64| io.read32((w << 24) + 4);
            // The Crossbow's identification: revision 2, bit 0 set.
            assert_eq!(id(0).data, 0x2000_0001);
            for w in answers {
                assert_eq!(id(*w).status, BUS_OK, "node {node} widget {w}");
            }
            for w in fails {
                assert_eq!(id(*w).status, BUS_ERR, "node {node} widget {w}");
            }
            if node == 0 {
                assert_ne!(id(8).data, id(9).data);
                assert_eq!(io.write32((8 << 24) + crate::dev::bridge::reg::INT_ENABLE as u64 + 4, 0), BUS_OK);
            }
        }
    });
}
/// Each node's processor reads both nodes' memory through its own ppmem
/// window once the PROM has numbered the nodes, as the first node's processor
/// always read its own: a cached access is a load on either node, not a walk
/// through the Hubs, so the second node runs code as fast as the first.
#[test]
fn both_processors_reach_both_nodes_memory_through_their_windows() {
    use crate::ppmem::{bitmap_maps_at, MappedMemory, FINE_SHIFT};
    use std::sync::atomic::Ordering;
    for mb in [64, 128, 256] {
        big_stack(move || {
            let dir = TempPath::dir(&format!("ip27-windows-{mb}"));
            let mut cfg = config(ORIGIN_2000_PROFILE, &dir, 2);
            if mb == 256 {
                cfg.set_cpu(&crate::cpu::parts::R12000_270);
            }
            cfg.banks[0] = mb;
            cfg.banks[1] = mb;
            let m = test_support::machine(&cfg).unwrap();
            let (a, b) = (&m.nodes[0].chip, &m.nodes[1].chip);
            // Bank 0 lies below the alias, bank 1 above it.
            let (near, far) = (0x1000, (512 << 20) + 0x1000);
            // What node `n`'s processor reads at `at` through its window, as
            // the cache finds it there: the bitmap, or on a miss the fine table.
            let window = |n: usize, at: u64| {
                let sp = m.nodes[n].chip.decode.physical().ppmem_space().unwrap();
                let bitmap = unsafe { *m.node_cpus[n][0].ppmem_bitmap_ptr() };
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
                assert_eq!(b.write64(hub_regs::NI_STATUS_REV_ID::OFFSET, number << 8), BUS_OK);
                let other = number << 32;
                // Through the Hubs: node 0's memory at its number, node 1's at its.
                for (at, v) in [(near, 0x0a), (far, 0x0b), (other + near, 0x1a), (other + far, 0x1b)] {
                    assert_eq!(a.inbound().write64(at, v + number), BUS_OK);
                }
                if number >= 2 {
                    // Past the two nodes' window, the bus path answers.
                    let phys = m.nodes[0].chip.decode.physical();
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
                let sp = m.nodes[1].chip.decode.physical().ppmem_space().unwrap();
                *(sp.window_base().add(far as usize) as *mut u64) = 0x2b_u64.rotate_left(32);
            }
            assert_eq!(a.inbound().read64(far).data, 0x2b);
        });
    }
}
/// The BaseIO's Bridge sits behind the first node's Crossbow, so it reaches
/// that node's Hub as the host at widget 9, through the port's link.
#[test]
fn the_bridge_reaches_its_nodes_hub_through_the_crossbow_port() {
    use crate::dev::bridge::{reg, DIRECT_WIDGET_SHIFT};
    big_stack(|| {
        let (dir, m) = built(ORIGIN_2000_PROFILE, "bridge", 2);
        let bridge = bridge_on(&config(ORIGIN_2000_PROFILE, &dir, 2), &m, &[(0, 9)], 8);
        let card = test_support::Card::new();
        bridge.bus().attach(crate::pci::DevFn { device: 7, function: 0 }, card.clone());
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        let dma = card.state.dma().unwrap().clone();
        let memory = |i: usize, off: u64| m.nodes[i].chip.decode.physical().read32(off).data;
        let pending = |i: usize| m.nodes[i].chip.read64(crate::dev::hub::regs::PI_INT_PEND0::OFFSET).data;

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
            assert_eq!(m.nodes[0].chip.write64(crate::dev::hub::regs::PI_INT_PEND_MOD::OFFSET, 5), BUS_OK);
        }
    });
}
#[test]
fn two_nodes_share_one_crossbow_and_both_reach_the_baseio() {
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "two", 2);
        let (n1, n2) = (&m.nodes[0].chip, &m.nodes[1].chip);
        // Slot identity 3 in the first node slot, 2 in the second.
        assert_eq!(n1.read64(crate::dev::hub::regs::MD_SLOTID_USTAT::OFFSET).data & 7, 3);
        assert_eq!(n2.read64(crate::dev::hub::regs::MD_SLOTID_USTAT::OFFSET).data & 7, 2);
        // The flash carries no machine type: the slot code selects the table.
        assert_eq!(n1.cpu_view(crate::sn::UncachedSpace::HubSpecial).read32(0x1000_0090).data, 0);

        for hub in [n1, n2] {
            let io = hub.cpu_view(crate::sn::UncachedSpace::Io);
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
            // Nothing is attached at 11 to 14.
            for w in 11..15 {
                assert_eq!(id(w).status, BUS_ERR, "widget {w}");
            }
        }
        // The BaseIO-G answers at widget 15 from either Hub.
        let id = |hub: &Arc<Hub>| hub.cpu_view(crate::sn::UncachedSpace::Io).read32((15 << 24) + 4);
        assert_eq!(id(n2).status, BUS_OK);
        assert_eq!(id(n1).status, BUS_OK);
        assert_eq!(id(n1).data, id(n2).data);

        // A big-window probe of the peer's port, as the PROM does it, reads
        // the peer's node number once the Hubs are numbered.
        assert_eq!(n2.write64(crate::dev::hub::regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
        let io = n1.cpu_view(crate::sn::UncachedSpace::Io);
        let bwin = crate::dev::hub::IO_BIG_WINDOWS;
        assert_eq!(n1.write64(crate::dev::hub::regs::IIO_ITTE::OFFSET, (9 << 8) | 0x1000), BUS_OK);
        let status = io.read64(bwin + 0x0160_0000);
        assert_eq!(status.status, BUS_OK);
        assert_eq!((status.data >> 8) & 0x1ff, 1);
    });
}
/// The BaseIO-G's Bridge sends DMA and interrupts to the Hub at the widget
/// the guest programs into it, not to a fixed one: the second node's Hub
/// (widget 9) gets both when the guest names it, and the first's (10) when the guest names that.
#[test]
fn the_bridge_sends_dma_and_interrupts_to_the_hub_the_guest_names() {
    use crate::dev::bridge::{reg, DIRECT_WIDGET_SHIFT};
    big_stack(|| {
        let (dir, m) = built(ONYX2_PROFILE, "bridge", 2);
        let bridge = bridge_on(&config(ONYX2_PROFILE, &dir, 2), &m, &[(0, 10), (1, 9)], 15);
        let card = test_support::Card::new();
        bridge.bus().attach(crate::pci::DevFn { device: 4, function: 0 }, card.clone());
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        let dma = card.state.dma().unwrap().clone();
        let memory = |i: usize, off: u64| m.nodes[i].chip.decode.physical().read32(off).data;
        let pending = |i: usize| m.nodes[i].chip.read64(crate::dev::hub::regs::PI_INT_PEND0::OFFSET).data;

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
            assert_eq!(m.nodes[node].chip.write64(crate::dev::hub::regs::PI_INT_PEND_MOD::OFFSET, 5), BUS_OK);
        }
    });
}
/// The packet's address, not the Crossbow port it leaves by, picks the node
/// that hears a device interrupt: a Bridge on the first node's port aimed at
/// the other node's number sets that node's pending bit.
#[test]
fn a_bridge_interrupt_aimed_at_the_other_nodes_number_reaches_that_node() {
    use crate::dev::bridge::reg;
    use crate::dev::hub::regs;
    big_stack(|| {
        let (dir, m) = built(ONYX2_PROFILE, "bridge-node", 2);
        let bridge = bridge_on(&config(ONYX2_PROFILE, &dir, 2), &m, &[(0, 10), (1, 9)], 15);
        let card = test_support::Card::new();
        bridge.bus().attach(crate::pci::DevFn { device: 4, function: 0 }, card.clone());
        let set = |off: u32, v: u32| assert_eq!(bridge.write32(u64::from(off + 4), v), BUS_OK);
        let pending = |i: usize| m.nodes[i].chip.read64(regs::PI_INT_PEND0::OFFSET).data;
        // As the PROM numbers an Onyx2: the second slot's node holds 0 and
        // the first slot's holds 1.
        assert_eq!(m.nodes[0].chip.write64(regs::NI_STATUS_REV_ID::OFFSET, 1 << 8), BUS_OK);
        assert_eq!(m.nodes[1].chip.write64(regs::NI_STATUS_REV_ID::OFFSET, 0), BUS_OK);

        // Widget 9 is the second slot's port. The pin's host-address field
        // replaces bits 47:38 of the destination: bit 47 for the I/O view and
        // number 1, which the first slot's Hub holds.
        set(reg::INT_DEST_UPPER, 9 << 16);
        set(reg::INT_DEST_LOWER, 0x0180_0090);
        set(reg::int_host(4), 5 | (1 << 9 | 1) << 8);
        set(reg::INT_ENABLE, 1 << 4);
        card.state.irq().unwrap().set(true);
        assert_eq!(pending(0) & 1 << 5, 1 << 5, "the node the address names did not hear it");
        assert_eq!(pending(1) & 1 << 5, 0, "the node whose port it left by heard it");
        card.state.irq().unwrap().set(false);
        // The guest's handler clears the pending bit on the node that heard it.
        assert_eq!(m.nodes[0].chip.write64(regs::PI_INT_PEND_MOD::OFFSET, 5), BUS_OK);
        assert_eq!(pending(0) & 1 << 5, 0);
    });
}
const WIDGET_PIPE: u8 = 8;
fn onyx2_with(dir: &Path, nodes: u8, board: crate::devices::BoardConfig) -> MachineConfig {
    let mut cfg = config(ONYX2_PROFILE, dir, nodes);
    cfg.set_graphics(Some(board));
    cfg
}
/// A store to the pipe's aperture through a Hub's I/O space, retried while the pipe's ring is full.
fn pipe_store(hub: &Arc<Hub>, off: u64, v: u32) {
    let io = hub.cpu_view(crate::sn::UncachedSpace::Io);
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
fn a_two_node_onyx2_builds_with_a_pipe_that_answers_its_widget_identity() {
    use crate::config::GraphicsBoard;
    big_stack(|| {
        for raster_managers in [1, 2] {
            let board = crate::devices::BoardConfig { model: GraphicsBoard::Ir2, raster_managers, ..Default::default() };
            let dir = TempPath::dir("ip27-pipe");
            let m = test_support::machine(&onyx2_with(&dir, 2, board)).unwrap();
            assert!(crate::workstation::Workstation::get_display(&m).is_some());
            for (node, board_node) in m.nodes.iter().enumerate() {
                let io = board_node.chip.cpu_view(crate::sn::UncachedSpace::Io);
                let id = io.read32((u64::from(WIDGET_PIPE) << 24) + 4);
                assert_eq!(id.status, BUS_OK, "{raster_managers} RMs from node {node}");
                // Revision in the top nibble; part, manufacturer and the
                // presence bit under it.
                assert_eq!(id.data & 0x0fff_ffff, 0x0c10_2555, "{raster_managers} RMs from node {node}");
                assert_ne!(id.data, TEST_WIDGET_ID);
            }
        }
    });
}
#[test]
fn without_a_board_the_graphics_slot_holds_the_test_widget() {
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "nopipe", 1);
        assert!(crate::workstation::Workstation::get_display(&m).is_none());
    });
}
/// The deskside's pipe is a `kona` entry: the model, with one or two raster managers, and only the Onyx2 has a slot for it.
#[test]
fn the_deskside_takes_an_ir2_or_ir2e_pipe_with_one_or_two_raster_managers() {
    use crate::config::{GraphicsBoard, MachineConfig};
    use crate::devices::{ConfiguredDevice, DeviceModel};
    for (arg, model, rms) in [("kona,model=ir2", GraphicsBoard::Ir2, 1), ("kona,model=ir2,raster_managers=2", GraphicsBoard::Ir2, 2), ("kona,model=ir2e,raster_managers=2", GraphicsBoard::Ir2e, 2)] {
        let DeviceModel::Board(b) = ConfiguredDevice::from_arg(arg).unwrap().device else { panic!("{arg}") };
        assert_eq!((b.model, b.raster_managers), (model, rms), "{arg}");
    }
    assert_eq!(MachineProfile::Onyx2Ip27.graphics_boards(), [GraphicsBoard::Ir2, GraphicsBoard::Ir2e]);
    for other in [MachineProfile::OctaneIp30, MachineProfile::Origin2000Ip27, MachineProfile::Origin200Ip29, MachineProfile::IndyIp24, MachineProfile::Indigo2Ip22] {
        assert!(other.graphics_boards().iter().all(|b| b.kona().is_none()), "{other:?}");
    }
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = MachineProfile::Onyx2Ip27;
    cfg.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::Ir2e)));
    assert!(cfg.validate().is_ok());
}
/// The driver's adaptor reset writes the Crossbow's link reset of the pipe's port, and the pipe
/// loses its Crosstalk interface registers; a link reset of another port leaves them.
#[test]
fn a_crossbow_link_reset_of_the_pipes_port_resets_its_interface_registers() {
    const WIDGET_CONTROL: u64 = 0x24;
    const INT_DEST_LOWER: u64 = 0x3c;
    const INT_VECTORS: u64 = 0x5c;
    // The reset register of link 0 (widget 8, the pipe's) and of link 7, low words, in widget 0.
    const LINK0_RESET: u64 = 0x100 + 0x30 + 4;
    const LINK7_RESET: u64 = 0x100 + 0x40 * 7 + 0x30 + 4;
    big_stack(|| {
        let dir = TempPath::dir("ip27-linkreset");
        let mut m = test_support::machine(&onyx2_with(&dir, 1, crate::devices::BoardConfig::new(crate::config::GraphicsBoard::Ir2))).unwrap();
        m.restart_peripherals();
        let hub = &m.nodes[0].chip;
        let io = hub.cpu_view(crate::sn::UncachedSpace::Io);
        // A read waits behind the stores the pipe's thread has not taken yet.
        let read = |off: u64| loop {
            let r = io.read32((u64::from(WIDGET_PIPE) << 24) + off);
            match r.status {
                crate::traits::BUS_BUSY => std::thread::yield_now(),
                status => {
                    assert_eq!(status, BUS_OK, "read of {off:#x}");
                    break r.data;
                }
            }
        };
        let reset_control = read(WIDGET_CONTROL);
        pipe_store(hub, WIDGET_CONTROL, 5 << 22);
        pipe_store(hub, INT_DEST_LOWER, 0x180_0090);
        pipe_store(hub, INT_VECTORS, 0x3e_2210);
        assert_eq!(io.write32(LINK7_RESET, 0), BUS_OK);
        assert_eq!((read(WIDGET_CONTROL), read(INT_DEST_LOWER), read(INT_VECTORS)), (5 << 22, 0x180_0090, 0x3e_2210), "another port's reset");
        assert_eq!(io.write32(LINK0_RESET, 0), BUS_OK);
        assert_eq!((read(WIDGET_CONTROL), read(INT_DEST_LOWER), read(INT_VECTORS)), (reset_control, 0, 0), "the pipe's port");
        m.stop();
    });
}
/// The pipe's DMA and interrupt packets reach the Hub at the widget the guest programs into it.
#[test]
fn the_pipe_sends_dma_and_interrupts_to_the_hub_the_guest_names() {
    use crate::dev::hub::regs;
    use crate::traits::Device;
    const WIDGET_CONTROL: u64 = 0x24;
    const INT_DEST_UPPER: u64 = 0x34;
    const INT_DEST_LOWER: u64 = 0x3c;
    const INT_VECTORS: u64 = 0x5c;
    const INT_CAUSE: u64 = 0x80_0238;
    const GE_VECTOR: u32 = 0x11;
    const REASON_PAGE: u64 = 0x30_0000;
    big_stack(|| {
        let dir = TempPath::dir("ip27-route");
        let mut m = test_support::machine(&onyx2_with(&dir, 2, crate::devices::BoardConfig::new(crate::config::GraphicsBoard::Ir2))).unwrap();
        m.restart_peripherals();
        let memory = |i: usize| m.nodes[i].chip.decode.physical().read32(REASON_PAGE).data;
        let pending = |i: usize| m.nodes[i].chip.read64(regs::PI_INT_PEND0::OFFSET).data;
        // The pipe's thread posts the reason: wait until the node's interrupt shows it.
        let post = |hub: &Arc<Hub>, node: usize| {
            pipe_commands(hub, &[0x1f8, 0x103]);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while pending(node) & 1 << GE_VECTOR == 0 {
                assert!(std::time::Instant::now() < deadline, "the pipe posted nothing");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };

        // Node slot 2 is widget 9 and node slot 1 is widget 10.
        for (widget, node, other) in [(9u32, 1usize, 0usize), (10, 0, 1)] {
            let from = &m.nodes[other].chip;
            pipe_store(from, WIDGET_CONTROL, widget << 22);
            pipe_store(from, INT_DEST_UPPER, widget << 16 | 0x8000);
            pipe_store(from, INT_DEST_LOWER, 0x180_0090);
            pipe_store(from, INT_VECTORS, 0x3e_2210);
            pipe_commands(from, &[0x1f1, 0x1000_0000 | (REASON_PAGE >> 12) as u32, (REASON_PAGE & 0xfff) as u32, 0x4000]);

            post(from, node);
            assert_eq!(memory(node), 0x103, "widget {widget}: the named node did not get the reason");
            assert_eq!(memory(other), 0, "widget {widget}: the other node saw it");
            assert_eq!(pending(node) & 1 << GE_VECTOR, 1 << GE_VECTOR, "widget {widget}");
            assert_eq!(pending(other) & 1 << GE_VECTOR, 0, "widget {widget}: the other node heard it");

            // The handler acknowledges the cause and clears the pending bit;
            // the next reason raises it again.
            pipe_store(from, INT_CAUSE, 0x20);
            assert_eq!(m.nodes[node].chip.write64(regs::PI_INT_PEND_MOD::OFFSET, u64::from(GE_VECTOR)), BUS_OK);
            assert_eq!(pending(node) & 1 << GE_VECTOR, 0);
            // The pipe holds the next reason until the driver clears the page word.
            assert_eq!(m.nodes[node].chip.decode.physical().write32(REASON_PAGE, 0), BUS_OK);
            post(from, node);
            assert_eq!(pending(node) & 1 << GE_VECTOR, 1 << GE_VECTOR, "widget {widget}: the second packet");
            assert_eq!(m.nodes[node].chip.write64(regs::PI_INT_PEND_MOD::OFFSET, u64::from(GE_VECTOR)), BUS_OK);
            pipe_store(from, INT_CAUSE, 0x20);
            assert_eq!(m.nodes[node].chip.decode.physical().write32(REASON_PAGE, 0), BUS_OK);
        }
        m.stop();
    });
}
/// With an I/O PROM image named, the Bridge's flash window at widget offset
/// 0xc0_0000 shows it; without one it shows the I/O PROM built into iris.
#[cfg(feature = "embedded-proms")]
#[test]
fn the_io_prom_image_fills_the_baseio_flash() {
    const BASEIO_G_WIDGET: u64 = 15;
    big_stack(|| {
        let flash = |m: &Machine<Hub, NodeFlash>, off: u64| {
            m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::Io).read32((BASEIO_G_WIDGET << 24) + 0xc0_0000 + off).data
        };
        let dir = TempPath::dir("ip27-onyx2-ioprom");
        let (_none, bare) = built(ONYX2_PROFILE, "noioprom", 1);
        let io6 = crate::dev::embedded_prom::get(crate::dev::embedded_prom::EmbeddedProm::Io6).unwrap();
        assert_eq!(flash(&bare, 0), u32::from_be_bytes(io6[..4].try_into().unwrap()));
        assert_eq!(flash(&bare, 4 * (io6.len() as u64 / 4)), u32::MAX, "erased after the image");
        let image = dir.join("io6prom.img");
        std::fs::write(&image, [0xde, 0xad, 0xbe, 0xef, 0x01]).unwrap();
        let mut cfg = config(ONYX2_PROFILE, &dir, 1);
        cfg.onyx2.io_prom = Some(image.to_string_lossy().into_owned());
        let m = test_support::machine(&cfg).unwrap();
        assert_eq!(flash(&m, 0), 0xdead_beef);
        assert_eq!(flash(&m, 4), 0x01ff_ffff);
        cfg.onyx2.io_prom = Some(dir.join("missing").to_string_lossy().into_owned());
        assert!(test_support::machine(&cfg).err().unwrap().contains("machine.io_prom"));
        // "(embedded)" loads the BaseIO PROM built into iris.
        cfg.onyx2.io_prom = Some("(embedded)".into());
        let m = test_support::machine(&cfg).unwrap();
        assert_eq!(flash(&m, 0), u32::from_be_bytes(io6[..4].try_into().unwrap()));
    });
}
/// The BaseIO-G's Bridge line carries the base board's record and the MediaIO's, as two parts with
/// distinct ROM numbers, and the Origin 2000's BaseIO carries its own alone.
#[test]
fn the_bridge_carries_the_baseio_and_mediaio_records() {
    let roms = board_records(&[(BASEIO_G_RECORD, BASEIO_G_SERIAL), (MIO_RECORD, MIO_SERIAL)]).roms();
    assert_eq!(roms.len(), 2);
    assert_ne!(roms[0], roms[1], "a search tells the boards apart");
    assert_eq!(board_records(&[(BASEIO_RECORD, BASEIO_SERIAL)]).roms().len(), 1);
}
/// The BaseIO-G's Bridge has two IOC3s, at devices 2 and 6, the SCSI
/// controllers at 0 and 1 and RAD1 at 7; nothing at 5.
#[test]
fn the_baseio_g_has_a_second_ioc3_and_the_audio_chip() {
    big_stack(|| {
        // Configuration register 0 of `device`, read through the first node's I/O space.
        let id = |m: &Machine<Hub, NodeFlash>, widget: u64, device: u64| {
            let r = m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::Io).read32((widget << 24) + u64::from(crate::dev::bridge::CFG0_BASE) + (device << 12));
            r.is_ok().then_some(r.data)
        };
        let (_dir, m) = built(ONYX2_PROFILE, "mio", 1);
        assert_eq!(id(&m, 15, 0), Some(0x1020_1077), "an ISP1040");
        assert_eq!(id(&m, 15, 2), Some(0x0003_10a9), "the board IOC3");
        assert_eq!(id(&m, 15, 6), Some(0x0003_10a9), "the MediaIO IOC3");
        assert_eq!(id(&m, 15, 7), Some(0x0005_10a9), "RAD1");
        assert_eq!(id(&m, 15, 5), None, "nothing at 5");
        let (_dir, m) = built(ORIGIN_2000_PROFILE, "no-mio", 1);
        assert_eq!(id(&m, 8, 2), Some(0x0003_10a9));
        assert_eq!((id(&m, 8, 6), id(&m, 8, 7)), (None, None), "no daughtercard");
    });
}
/// A SuperIO source raised on the MediaIO's IOC3 reaches Bridge input 6, and
/// one raised on the board's IOC3 reaches input 4.
#[test]
fn each_ioc3_super_io_interrupt_reaches_its_own_bridge_input() {
    use crate::dev::bridge::{device_window_offset, reg::INT_STATUS as BRIDGE_INT_STATUS, CFG0_BASE};
    use crate::dev::ioc3::{int, reg as ioc3_reg};
    const BASEIO_G_WIDGET: u64 = 15;
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "sio-irq", 1);
        let io = m.nodes[0].chip.cpu_view(crate::sn::UncachedSpace::Io);
        let base = BASEIO_G_WIDGET << 24;
        let read = |at: u64| {
            let r = io.read32(base + at);
            assert!(r.is_ok(), "read at {at:#x}");
            r.data
        };
        let write = |at: u64, v: u32| assert_eq!(io.write32(base + at, v), BUS_OK, "write at {at:#x}");
        // The keyboard controller answers a command byte with an
        // acknowledge, which raises the keyboard source. The IOC3's memory BAR goes where the
        // Bridge's window for its device already points.
        let raise = |device: u32| {
            let cfg = u64::from(CFG0_BASE + device * 0x1000);
            write(cfg + 0x10, (device + 2) << 20);
            write(cfg + 0x04, 2);
            let window = u64::from(device_window_offset(device));
            write(window + u64::from(ioc3_reg::INT_ENABLE_SET), int::KEYBOARD);
            write(window + 0xa8, 0xf4);
            assert_ne!(read(window + u64::from(ioc3_reg::INT_STATUS)) & int::KEYBOARD, 0, "device {device}: source raised");
        };
        let pending = || read(u64::from(BRIDGE_INT_STATUS) + 4) & 0xff;
        assert_eq!(pending(), 0);
        raise(6);
        assert_eq!(pending(), 1 << 6, "the MediaIO's interrupt is Bridge input 6 and no other");
        raise(2);
        assert_eq!(pending(), 1 << 6 | 1 << 4, "the board IOC3's is input 4");
    });
}
/// What `PipeHosts` needs of a pipe: the widget its DMA goes to, here a fixed
/// one.
fn pipe_hosts_for(m: &Machine<Hub, NodeFlash>, widget: u8) -> (PipeHosts, Arc<std::sync::atomic::AtomicU8>) {
    let handle = Arc::new(std::sync::atomic::AtomicU8::new(widget));
    let hosts = PipeHosts {
        widget: OnceLock::new(),
        up: Upstream::new()
            .with(m.nodes[0].chip.host_port(10))
            .with(m.nodes[1].chip.host_port(9)),
    };
    hosts.widget.set(handle.clone()).unwrap();
    (hosts, handle)
}
/// A block the pipe moves goes to the node whose Hub the widget names, and
/// follows the widget when the guest reprograms it.
#[test]
fn the_pipes_dma_blocks_follow_its_widget() {
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "pipeblock", 2);
        let (hosts, widget) = pipe_hosts_for(&m, 10);
        let at = 0x10_0000u64;
        let words = [0x0102_0304_0506_0708u64, 0x1112_1314_1516_1718];
        for (w, node, other) in [(10u8, 0usize, 1usize), (9, 1, 0)] {
            widget.store(w, std::sync::atomic::Ordering::Relaxed);
            for node in &m.nodes {
                assert_eq!(node.chip.decode.physical().write32(at, 0), BUS_OK);
            }
            assert_eq!(hosts.write_block(at, &words), BUS_OK, "widget {w}");
            assert_eq!(m.nodes[node].chip.decode.physical().read32(at).data, 0x0102_0304, "widget {w}");
            assert_eq!(m.nodes[other].chip.decode.physical().read32(at).data, 0, "widget {w}: the other node saw it");
            let mut back = [0u64; 2];
            assert_eq!(hosts.read_block(at, &mut back), BUS_OK, "widget {w}");
            assert_eq!(back, words, "widget {w}");
        }
        // A widget that is no Hub fails the block as it fails a word.
        widget.store(3, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(hosts.write_block(at, &words), BUS_ERR);
        assert_eq!(hosts.read_block(at, &mut [0u64; 2]), BUS_ERR);
    });
}
/// Pipe DMA words a second through `PipeHosts` into node memory, as a 64 KiB texture load moves
/// them.
#[test]
#[ignore]
fn pipe_dma_rate() {
    big_stack(|| {
        let (_dir, m) = built(ONYX2_PROFILE, "pipedmarate", 2);
        let (hosts, _widget) = pipe_hosts_for(&m, 9);
        let mut buf = vec![0u8; 64 * 1024];
        let at = 0x10_0000u64;
        let mut best = 0f64;
        for _ in 0..5 {
            let (start, mut loads) = (std::time::Instant::now(), 0u64);
            while start.elapsed() < std::time::Duration::from_millis(400) {
                assert_eq!(crate::physical::write_bytes(&hosts, at, &buf), BUS_OK);
                assert_eq!(crate::physical::read_bytes(&hosts, at, &mut buf), BUS_OK);
                loads += 1;
            }
            best = best.max(loads as f64 * 2.0 * 64.0 / 1024.0 / start.elapsed().as_secs_f64());
        }
        println!("pipe DMA through the Hub: {best:.1} MiB/s");
    });
}
