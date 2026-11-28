use super::*;
use crate::config::{GraphicsBoard, MachineConfig, MachineProfile, NvmeConfig, ScsiDeviceConfig};
use crate::devices::{DeviceModel, ConfiguredDevice};
use crate::dev::xbridge::{HEAD_E, HEAD_F};
use crate::ip35::test_support::{config, headless, widget, with_machine};
use crate::dev::bedrock::Bedrock;
use crate::dev::bedrock_flash::Flash;
use crate::sn::Machine;
use crate::test_temp::TempPath;
use crate::traits::{BusDevice, BUS_ERR, BUS_OK};
use std::path::Path;

/// The I-brick's slots read back as wired: the Fibre Channel adapter at device 1,
/// the IOC3 at 4 and the USB controller at 5 of head F, nothing anywhere on head E,
/// and nothing at the second host port.
#[test]
fn the_ibricks_slots_read_back_as_wired() {
    let dir = TempPath::dir("o3k-slots");
    with_machine(config(&dir, 1), |m| {
        let pio = m.nodes[0].chip.pio();
        let id = |head: u8, device: u64| pio.read32(widget(u64::from(head), 0x2_0000 + (device << 12)));
        for (device, want) in [(1, 0x2200_1077), (4, 0x0003_10a9), (5, 0x5802_11c1)] {
            assert_eq!(id(HEAD_F, device).data, want, "device {device}");
        }
        for device in 0..8 {
            let read = id(HEAD_E, device);
            assert!(read.status != BUS_OK || read.data == 0xffff_ffff, "head E device {device} is empty");
        }
        assert_eq!(pio.read32(widget(u64::from(SPARE_PORT), 0)).status, BUS_ERR, "port B is empty");
    });
}
/// An `ice1724` entry puts the ICE1724 in the slot it names, here device 2 of the I-brick's head F,
/// on either product, with the monitor's audio report; without one the slot is empty.
#[test]
fn the_audio_card_sits_in_the_ibrick_slot_it_is_placed_in() {
    let dir = TempPath::dir("o3k-audio");
    let slot = |cfg: MachineConfig| {
        with_machine(cfg, |m| {
            let read = m.nodes[0].chip.pio().read32(widget(u64::from(HEAD_F), 0x2_0000 + (2 << 12)));
            ((read.status == BUS_OK).then_some(read.data), m.io.monitor.iter().any(|d| d.audio_report().is_some()))
        })
    };
    let fitted = |mut cfg: MachineConfig| {
        cfg.no_audio = true;
        cfg.devices.push(crate::devices::ConfiguredDevice::new("xio15/pci0/2", crate::devices::DeviceModel::Ice1724));
        cfg
    };
    for cfg in [config(&dir, 1), onyx_config(&dir, GraphicsBoard::Ir3)] {
        let (id, report) = slot(cfg.clone());
        assert!(id.is_none_or(|id| id == u32::MAX) && !report, "{:?}: empty without the key", cfg.machine.profile);
        assert_eq!(slot(fitted(cfg.clone())), (Some(0x1724_1412), true), "{:?}", cfg.machine.profile);
    }
}

/// A processor's thread is a processor of the node it runs on and of no
/// other: the second node's chip numbers the machine's CPU 1 its own
/// processor 0, and takes the first node's thread for no processor.
#[test]
fn a_node_numbers_only_its_own_processors() {
    let dir = TempPath::dir("o3k-cpus");
    let specs = node_specs(&config(&dir, 2)).unwrap();
    let second = specs[1].processors.with_first_cpu(1);
    assert_eq!((second.local(0), second.local(1), second.local(2)), (None, Some(0), None));
    assert_eq!(second.slot_of_cpu(1), Some((0, 0)));
    assert_eq!(specs[0].processors.local(1), None);
}
fn refused(cfg: &MachineConfig) -> String {
    cfg.validate().unwrap_err()
}
/// The row's limits: one or two C-bricks, banks a C-brick takes in matched
/// pairs, disks on the loop and not on a SCSI bus, no option board keys, no graphics.
#[test]
fn the_configuration_refuses_what_the_row_does_not_hold() {
    let dir = TempPath::dir("o3k-config");
    config(&dir, 1).validate().unwrap();
    config(&dir, 2).validate().unwrap();
    assert!(refused(&config(&dir, 3)).contains("machine.nodes 3"));
    let mut cfg = config(&dir, 1);
    cfg.banks = [512, 512, 0, 0, 0, 0, 0, 0];
    cfg.banks_set = true;
    // A 512 MiB bank can sit among banks 0 to 3 and cover the PROM's window.
    cfg.validate().unwrap();
    cfg.banks = [256, 256, 256, 256, 512, 512, 0, 0];
    cfg.validate().unwrap();
    cfg.banks = [512, 256, 0, 0, 0, 0, 0, 0];
    assert!(refused(&cfg).contains("share one size"), "{}", refused(&cfg));
    cfg.banks = [256, 128, 0, 0, 0, 0, 0, 0];
    assert!(refused(&cfg).contains("256 or 512"));
    let loop_disk = |at: &str| -> MachineConfig {
        let mut cfg = config(&dir, 1);
        cfg.devices.push(ConfiguredDevice::new(at, DeviceModel::Scsi(ScsiDeviceConfig { path: "d.img".into(), ..Default::default() })));
        cfg
    };
    loop_disk("xio15/pci0/1/l1").validate().unwrap();
    assert!(refused(&loop_disk("xio15/pci0/1/l126")).contains("no slot xio15/pci0/1/l126"));
    let mut cfg = config(&dir, 1);
    let image = dir.join("n.img");
    std::fs::write(&image, vec![0u8; 1 << 20]).unwrap();
    cfg.devices.push(ConfiguredDevice::new("bus1-slot2", DeviceModel::Nvme(NvmeConfig { path: image.to_string_lossy().into_owned(), ..Default::default() })));
    cfg.resolve_slots().unwrap();
    let err = std::thread::Builder::new().stack_size(64 << 20).spawn(move || crate::ip35::test_support::machine(&cfg).err()).unwrap().join().unwrap();
    assert!(err.expect("the machine refuses a card in a slot").contains("takes no NVMe card"));
    let mut cfg = config(&dir, 1);
    cfg.set_disk(2, toml::from_str("path = \"d.raw\"").unwrap());
    assert!(refused(&cfg).contains("has no slot scsi0/t2"), "{}", refused(&cfg));
    let mut cfg = config(&dir, 1);
    cfg.set_graphics(Some(crate::devices::BoardConfig::new(crate::config::GraphicsBoard::V12)));
    assert!(refused(&cfg).contains("Origin 3200"));
}
/// `r12000` and `r12000_400` name the C-brick's processor, the 400 MHz part
/// with 8 MiB of secondary cache; the Octane's 2 MiB part is not offered.
#[test]
fn the_c_brick_takes_the_400_mhz_r12000_with_the_8_mib_cache() {
    for name in [None, Some("r12000"), Some("r12000_400")] {
        assert_eq!(MachineProfile::Origin3200Ip35.part(name).unwrap(), &parts::R12000_400_8M, "{name:?}");
    }
    assert!(MachineProfile::Origin3200Ip35.part(Some("r12000_360")).is_err());
}
/// With a 512 MiB bank 0 the memory covers the reset PROM's window.
#[test]
fn the_prom_still_fetches_under_a_512_mib_bank() {
    use crate::gdb_stub::StopReason;
    use gdbstub_arch::mips::reg::id::MipsRegId;
    let dir = TempPath::dir("o3k-bank512");
    let mut cfg = config(&dir, 1);
    cfg.banks = [512, 512, 0, 0, 0, 0, 0, 0];
    cfg.banks_set = true;
    with_machine(cfg, |m| {
        let node = &m.nodes[0];
        // Banks 0 and 1 enabled, size code 4 (512 MiB): slot 0 holds RAM past 0x1fc00000.
        assert_eq!(node.chip.registers().write64(crate::dev::bedrock::regs::md::BASE + 8, 0x43), BUS_OK);
        assert!(node.chip.decode.physical().read32(0x1fc0_0000).status == BUS_OK, "RAM answers the PROM's window in the cached view");
        assert_eq!(node.chip.decode.physical().read32(0x1fc0_0000).data, 0, "and it is not the PROM's word");
        let cpu = m.node_cpus[0][0].clone().debug_adapter();
        assert_eq!(cpu.step_one(), StopReason::DoneStep);
        let t0 = cpu.read_reg(MipsRegId::Gpr(8)).unwrap();
        assert_eq!(t0, 0xffff_ffff_a600_0000, "the first instruction was the PROM's lui, not RAM's zero word");
    });
}
/// The disks placed on the I-brick's loop are on the I-brick's loop at the loop IDs
/// they name, and a machine with none still has the adapter.
#[test]
fn the_loop_has_the_configured_disks() {
    let dir = TempPath::dir("o3k-loop");
    let image = dir.join("d.raw");
    std::fs::write(&image, vec![0u8; 1 << 20]).unwrap();
    let mut cfg = config(&dir, 1);
    cfg.devices.push(ConfiguredDevice::new("xio15/pci0/1/l3", DeviceModel::Scsi(ScsiDeviceConfig { path: image.to_string_lossy().into_owned(), overlay: true, ..Default::default() })));
    let attached = |m: &mut Machine<Bedrock, Flash>| {
        let mut attached = Vec::new();
        (m.io.disks)(&mut |buses| {
            for bus in buses {
                attached.push(bus.with_targets(|t| t.iter().enumerate().filter(|(_, d)| d.is_some()).map(|(i, _)| i).collect::<Vec<_>>()));
            }
        });
        attached
    };
    assert_eq!(with_machine(cfg, attached), [vec![3]]);
    assert_eq!(with_machine(config(&dir, 1), attached), [Vec::<usize>::new()], "the adapter is on the I-brick");
}

/// Put the board in the graphics slot, the I-brick's port: the only place a V12 goes with one C-brick.
fn at_graphics(cfg: &mut MachineConfig) {
    let p = cfg.machine.profile;
    cfg.devices.iter_mut().filter(|d| d.names_graphics(p)).for_each(|d| d.at = Some("graphics".into()));
}

fn onyx_config(dir: &Path, board: GraphicsBoard) -> MachineConfig {
    let mut cfg = headless(dir, MachineProfile::Onyx3000Ip35);
    cfg.set_graphics(Some(crate::devices::BoardConfig::new(board)));
    cfg.onyx3000.l1_port = Some(0);
    cfg
}

/// The pipe answers at widget 11 of the I-brick's crossbar and its identity
/// reads there, where the Origin 3200 has nothing.
#[test]
fn the_pipe_reads_back_at_widget_11() {
    let dir = TempPath::dir("o3k-ir-widget");
    with_machine(onyx_config(&dir, GraphicsBoard::Ir3), |m| {
        let pio = m.nodes[0].chip.pio();
        let id = pio.read32(widget(u64::from(PIPE_WIDGET), 0x04));
        assert_eq!((id.status, id.data), (BUS_OK, 0x0c10_2555), "widget identification");
        // The I-brick's other ports are as on the Origin 3200: the host port and
        // the heads answer, a port with nothing attached does not.
        for port in [HOST_WIDGET, HEAD_E, HEAD_F] {
            assert_eq!(pio.read32(widget(u64::from(port), 0x04)).status, BUS_OK, "port {port}");
        }
        assert_ne!(pio.read32(widget(9, 0x04)).status, BUS_OK, "port 9 is empty");
    });
}

/// The pipe's interrupt controller is the I-brick's inbound port for its widget: a packet reaches
/// the processor interface its address names, which is why the pipe hands over the address the
/// driver programmed (`dev::kona`'s test of the packet it sends).
#[test]
fn a_packet_from_the_pipes_widget_reaches_the_processor_interface_its_address_names() {
    use crate::dev::bedrock::regs::{pi, PI1_BASE};
    use crate::irq::InterruptController;
    let dir = TempPath::dir("o3k-ir-int");
    with_machine(onyx_config(&dir, GraphicsBoard::Ir3), |m| {
        let bedrock = &m.nodes[0].chip;
        let port = bedrock.inbound(PIPE_WIDGET);
        let pending = |base: u64| bedrock.registers().read64(base + 0x98).data & (1 << 12) != 0;
        assert!(!pending(PI1_BASE) && !pending(pi::BASE));
        assert!(port.packet(PIPE_WIDGET, 0x8000_0180_0090 + PI1_BASE, 12, true));
        assert!(pending(PI1_BASE), "the second processor interface took the packet");
        assert!(!pending(pi::BASE), "and the first did not");
    });
}

/// A Crossbow link reset of the pipe's port resets the widget's Crosstalk registers and leaves the
/// board's own: the driver's adaptor reset follows it with the board's reset pulse, and the board
/// state in between (the map RAM among it) must survive.
#[test]
fn a_link_reset_of_the_pipes_port_leaves_the_boards_own_registers() {
    const INT_VECTORS: u64 = 0x5c;
    const INT_ENABLE: u64 = 0x80_0230;
    // Link 3 (widget 11) of the crossbar's registers in widget 0: its reset register's low word.
    const LINK3_RESET: u64 = 0x100 + 0x40 * 3 + 0x30 + 4;
    let dir = TempPath::dir("o3k-ir-linkreset");
    with_machine(onyx_config(&dir, GraphicsBoard::Ir3), |m| {
        for d in &m.io.devices {
            d.start();
        }
        let pio = m.nodes[0].chip.pio();
        // The pipe applies its stores in order, and a read waits behind them.
        let read = |off: u64| loop {
            let r = pio.read32(widget(u64::from(PIPE_WIDGET), off));
            match r.status {
                crate::traits::BUS_BUSY => std::thread::yield_now(),
                status => {
                    assert_eq!(status, BUS_OK, "read of {off:#x}");
                    break r.data;
                }
            }
        };
        let write = |off: u64, value: u32| loop {
            match pio.write32(widget(u64::from(PIPE_WIDGET), off), value) {
                crate::traits::BUS_BUSY => std::thread::yield_now(),
                status => break assert_eq!(status, BUS_OK, "write of {off:#x}"),
            }
        };
        write(INT_VECTORS, 0x3e_2210);
        write(INT_ENABLE, 0xf7c);
        assert_eq!((read(INT_VECTORS), read(INT_ENABLE)), (0x3e_2210, 0xf7c));
        assert_eq!(pio.write32(widget(0, LINK3_RESET), 0), BUS_OK);
        assert_eq!((read(INT_VECTORS), read(INT_ENABLE)), (0, 0xf7c), "the widget's registers reset, the board's stay");
        for d in m.io.devices.iter().rev() {
            d.stop();
        }
    });
}

/// The profile's limits: the pipe is a product the table ships in an Onyx 3000; the machine has
/// one or two C-bricks with either board; it has graphics, so none is the Origin 3200.
#[test]
fn the_configuration_refuses_what_the_onyx_does_not_hold() {
    let dir = TempPath::dir("o3k-ir-config");
    onyx_config(&dir, GraphicsBoard::Ir3).validate().unwrap();
    for board in [GraphicsBoard::Ir2, GraphicsBoard::Ir2e, GraphicsBoard::V10, GraphicsBoard::V8] {
        assert!(refused(&onyx_config(&dir, board)).contains("no free slot for a graphics board"), "{}", board.as_str());
    }
    let mut none = onyx_config(&dir, GraphicsBoard::Ir3);
    none.set_graphics(None);
    assert!(refused(&none).contains("Onyx 3000 needs a graphics board"), "{}", refused(&none));
    // The C-bricks are `[onyx3000] c_bricks`, whichever the board: unset, one, or two with a V12.
    for board in [GraphicsBoard::Ir3, GraphicsBoard::V12] {
        let mut cfg = onyx_config(&dir, board);
        assert_eq!(cfg.c_bricks(), if board == GraphicsBoard::V12 { 2 } else { 1 });
        for count in [1u8, 2] {
            cfg.onyx3000.c_bricks = Some(count);
            if count == 1 {
                at_graphics(&mut cfg);
            }
            cfg.validate().unwrap();
            assert_eq!((cfg.c_bricks(), node_specs(&cfg).unwrap().len()), (usize::from(count), usize::from(count)));
        }
        cfg.onyx3000.c_bricks = Some(3);
        assert!(refused(&cfg).contains("machine.nodes 3"), "{}", refused(&cfg));
    }
    // The brick keys are read from `[onyx3000]`, and an error names it.
    let mut cfg = onyx_config(&dir, GraphicsBoard::Ir3);
    cfg.onyx3000.brick_serial = Some("X".repeat(38));
    assert!(refused(&cfg).starts_with("machine.brick_serial"), "{}", refused(&cfg));
    let mut cfg = onyx_config(&dir, GraphicsBoard::Ir3);
    cfg.origin3200.brick_serial = Some("X".repeat(38));
    cfg.validate().unwrap();
    cfg.onyx3000.l1_port = Some(8882);
    assert_eq!(cfg.ip35_section().l1_port, Some(8882));
    assert_eq!(cfg.ip35_section().brick_serial, None, "`[origin3200]` is not read");
    let mut cfg = onyx_config(&dir, GraphicsBoard::Ir3);
    let p = cfg.machine.profile;
    cfg.devices.retain(|d| !d.names_graphics(p));
    cfg.validate().unwrap();
    assert_eq!(cfg.graphics_board(), Some(GraphicsBoard::Ir3), "the default");
}

/// The V12 reads back at widget 8 of the second C-brick's crossbar; the first
/// C-brick has no such widget, and carries no pipe at widget 11.
#[test]
fn the_vpro_reads_back_at_widget_8_of_the_second_c_brick() {
    let dir = TempPath::dir("o3k-vpro-widget");
    with_machine(onyx_config(&dir, GraphicsBoard::V12), |m| {
        let second = m.nodes[1].chip.pio();
        let id = second.read32(widget(u64::from(VPRO_WIDGET), 0x04));
        assert_eq!(id.status, BUS_OK, "the V12 answers");
        assert_ne!(id.data, 0xffff_ffff);
        // The V-brick's crossbar is an XBridge at widget 0, with its host port.
        assert_eq!(second.read32(widget(0, 0)).status, BUS_OK);
        assert_eq!(second.read32(widget(10, 0)).status, BUS_OK);
        let first = m.nodes[0].chip.pio();
        assert_ne!(first.read32(widget(u64::from(VPRO_WIDGET), 0x04)).status, BUS_OK, "the I-brick has no widget 8");
        assert_ne!(first.read32(widget(u64::from(PIPE_WIDGET), 0x04)).status, BUS_OK, "and no pipe");
    });
}

/// Each C-brick's controller names the brick on its link: the I-brick on the
/// first, the V-brick on the second answering as an X-brick (approximation).
#[test]
fn each_c_bricks_controller_names_its_brick() {
    let dir = TempPath::dir("o3k-vpro-l1");
    let v = node_specs(&onyx_config(&dir, GraphicsBoard::V12)).unwrap();
    let attached: Vec<_> = v.iter().map(|s| s.l1.attached.map(|b| (b.type_code, b.graphics_bay))).collect();
    assert_eq!(attached, [Some((0x49, None)), Some((0x58, None))]);
    let g = node_specs(&onyx_config(&dir, GraphicsBoard::Ir3)).unwrap();
    assert_eq!(g.len(), 1);
    assert_eq!(g[0].l1.attached.map(|b| (b.type_code, b.graphics_bay)), Some((0x49, Some(G_BRICK_BAY))));
}

/// With one C-brick the V12 answers at widget 11 of the I-brick's crossbar, and the I-brick's
/// controller names a brick on that port; a pipe with two C-bricks stays on the first.
#[test]
fn either_board_takes_one_or_two_c_bricks() {
    let dir = TempPath::dir("o3k-free-bricks");
    let mut cfg = onyx_config(&dir, GraphicsBoard::V12);
    cfg.onyx3000.c_bricks = Some(1);
    at_graphics(&mut cfg);
    let specs = node_specs(&cfg).unwrap();
    assert_eq!(specs[0].l1.attached.map(|b| (b.type_code, b.graphics_bay)), Some((0x49, Some(G_BRICK_BAY))));
    with_machine(cfg, |m| {
        assert_eq!(m.nodes.len(), 1);
        let id = m.nodes[0].chip.pio().read32(widget(u64::from(PIPE_WIDGET), 0x04));
        assert_eq!(id.status, BUS_OK, "the V12 answers");
        assert_ne!(id.data, 0xffff_ffff);
    });
    let mut cfg = onyx_config(&dir, GraphicsBoard::Ir3);
    cfg.onyx3000.c_bricks = Some(2);
    let attached: Vec<_> = node_specs(&cfg).unwrap().iter().map(|s| s.l1.attached.map(|b| (b.type_code, b.graphics_bay))).collect();
    assert_eq!(attached, [Some((0x49, Some(G_BRICK_BAY))), None]);
    with_machine(cfg, |m| {
        assert_eq!(m.nodes.len(), 2);
        assert_eq!(m.nodes[0].chip.pio().read32(widget(u64::from(PIPE_WIDGET), 0x04)).status, BUS_OK, "the pipe answers");
        assert_ne!(m.nodes[1].chip.pio().read32(widget(0, 0x04)).status, BUS_OK, "the second C-brick has no I/O brick");
    });
}

/// A board's `at` moves it: a V12 in a brick of its own answers at the slot named,
/// and a pipe named onto the second C-brick answers there and not on the I-brick.
#[test]
fn a_boards_at_places_it() {
    let dir = TempPath::dir("o3k-placement");
    let at = |cfg: &mut MachineConfig, at: &str| {
        let p = cfg.machine.profile;
        cfg.devices.iter_mut().find(|d| d.names_graphics(p)).expect("a board").at = Some(at.into());
        // A board away from the graphics slot leaves the slot's default there; `none` drops it.
        if at != "xio11" {
            cfg.devices.push(ConfiguredDevice::new("graphics", DeviceModel::Empty));
        }
    };
    let mut cfg = onyx_config(&dir, GraphicsBoard::V12);
    at(&mut cfg, "node1/xio12");
    assert_eq!(graphics_port(&cfg), Ok(GraphicsPort { node: 1, widget: 12 }));
    with_machine(cfg, |m| {
        let second = m.nodes[1].chip.pio();
        assert_eq!(second.read32(widget(12, 0x04)).status, BUS_OK);
        assert_ne!(second.read32(widget(u64::from(VPRO_WIDGET), 0x04)).status, BUS_OK);
    });
    let mut cfg = onyx_config(&dir, GraphicsBoard::V12);
    at(&mut cfg, "xio11");
    assert_eq!(graphics_port(&cfg), Ok(GraphicsPort { node: 0, widget: PIPE_WIDGET }));
    assert_eq!(graphics_port(&onyx_config(&dir, GraphicsBoard::V12)), Ok(GraphicsPort { node: 1, widget: VPRO_WIDGET }), "a V12 with no slot takes the V-brick");
    let mut cfg = onyx_config(&dir, GraphicsBoard::Ir3);
    cfg.onyx3000.c_bricks = Some(2);
    at(&mut cfg, "node1/xio11");
    let attached: Vec<_> = node_specs(&cfg).unwrap().iter().map(|s| s.l1.attached.map(|b| (b.type_code, b.graphics_bay))).collect();
    assert_eq!(attached, [Some((0x49, None)), Some((0x58, Some(G_BRICK_BAY)))]);
    with_machine(cfg, |m| {
        assert_eq!(m.nodes[1].chip.pio().read32(widget(u64::from(PIPE_WIDGET), 0x04)).status, BUS_OK, "the pipe answers on the second");
        assert_ne!(m.nodes[0].chip.pio().read32(widget(u64::from(PIPE_WIDGET), 0x04)).status, BUS_OK, "and not on the I-brick");
    });
}

/// `[machine] cpus` fits one, two or four processors on each C-brick: the positions left empty
/// are the ones a fresh flash lists disabled, and the machine runs the bricks' processors
/// together, eight at most.
#[test]
fn each_c_brick_takes_one_two_or_four_processors() {
    let dir = TempPath::dir("o3k-populations");
    for (cpus, mask, absent, pimms) in [(1u8, 0b0001, "BCD", 0b01), (2, 0b0101, "BD", 0b11), (4, 0b1111, "", 0b11)] {
        let mut cfg = config(&dir, 2);
        cfg.machine.cpus = Some(cpus);
        cfg.validate().unwrap();
        assert_eq!(cfg.total_cpus(), 2 * usize::from(cpus));
        let specs = node_specs(&cfg).unwrap();
        for spec in &specs {
            assert_eq!((spec.processors.mask(), spec.absent), (mask, absent), "{cpus} per brick");
            assert_eq!(spec.l1.compute_node.as_ref().map(|n| n.pimm_mask), Some(pimms), "{cpus} per brick");
        }
    }
    let mut cfg = config(&dir, 2);
    for cpus in [0u8, 3, 5] {
        cfg.machine.cpus = Some(cpus);
        assert!(refused(&cfg).contains("1, 2 or 4 on each node board"), "{}", refused(&cfg));
    }
    // The V12 Onyx 3000 is two C-bricks; the G-brick one is one.
    let mut onyx = onyx_config(&dir, GraphicsBoard::V12);
    onyx.machine.cpus = Some(4);
    assert_eq!(onyx.total_cpus(), 8);
    let mut onyx = onyx_config(&dir, GraphicsBoard::Ir3);
    onyx.machine.cpus = Some(4);
    assert_eq!(onyx.total_cpus(), 4);
}

/// Two C-bricks of four build a machine of eight processors, numbered across the bricks in order.
#[test]
fn two_c_bricks_of_four_build_eight_processors() {
    let dir = TempPath::dir("o3k-eight");
    let mut cfg = config(&dir, 2);
    cfg.machine.cpus = Some(4);
    with_machine(cfg, |m| {
        assert_eq!(m.cpus().count(), 8);
        assert_eq!(m.nodes.iter().map(|n| n.first_cpu).collect::<Vec<_>>(), [0, 4]);
    });
}
