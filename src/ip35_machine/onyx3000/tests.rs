use super::super::{Chassis, Ip35Board, Onyx3000};
use super::*;
use crate::config::{GraphicsBoard, MachineProfile, NetMode};
use crate::test_temp::TempPath;
use crate::traits::{BusDevice, BUS_OK};
use std::path::Path;

fn prom() -> Vec<u8> {
    let mut p = vec![0u8; 0x1000];
    p[..4].copy_from_slice(&0x3c08_a600u32.to_be_bytes());
    p
}

fn config(dir: &Path, board: GraphicsBoard) -> MachineConfig {
    let mut cfg = MachineConfig::default();
    cfg.machine.profile = MachineProfile::Onyx3000Ip35;
    cfg.graphics.board = Some(board);
    cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
    cfg.headless = true;
    cfg.network.mode = NetMode::Off;
    cfg.serial_port_a = Some(0);
    cfg.serial_port_b = Some(0);
    cfg
}

/// A PIO address: the widget in bits 27:24.
fn widget(widget: u8, offset: u64) -> u64 {
    (u64::from(widget) << 24) | offset
}

/// The G-brick Onyx 3000 builds and is freed: nothing the pipe, its
/// interrupt path or its crossbar port hold keeps the C-brick's chip alive.
#[test]
fn the_g_brick_onyx_builds_and_is_freed() {
    let dir = TempPath::dir("o3k-ir-free");
    let chassis = Chassis::<Onyx3000>::build(&config(&dir, GraphicsBoard::Ir3), &prom(), Arc::new(|| 0)).unwrap();
    assert_eq!(chassis.nodes.len(), 1, "one C-brick");
    assert!(chassis.io.display.is_some() && chassis.io.status.is_some());
    let chip = Arc::downgrade(&chassis.nodes[0].bedrock);
    drop(chassis);
    assert!(chip.upgrade().is_none(), "the chip is held by the pipe's path");
}

/// The pipe answers at widget 11 of the I-brick's crossbar and its identity
/// reads there, where the Origin 3200 has nothing.
#[test]
fn the_pipe_reads_back_at_widget_11() {
    let dir = TempPath::dir("o3k-ir-widget");
    let chassis = Chassis::<Onyx3000>::build(&config(&dir, GraphicsBoard::Ir3), &prom(), Arc::new(|| 0)).unwrap();
    let pio = chassis.nodes[0].bedrock.pio();
    let id = pio.read32(widget(PIPE_WIDGET, 0x04));
    assert_eq!((id.status, id.data), (BUS_OK, 0x0c10_2555), "widget identification");
    // The I-brick's other ports are as on the Origin 3200: the host port and
    // the heads answer, a port with nothing fitted does not.
    for port in [IBRICK.host_port, IBRICK.heads[0], IBRICK.heads[1]] {
        assert_eq!(pio.read32(widget(port, 0x04)).status, BUS_OK, "port {port}");
    }
    assert_ne!(pio.read32(widget(9, 0x04)).status, BUS_OK, "port 9 is empty");
}

/// The pipe's interrupt controller is the I-brick's inbound port for its
/// widget: a packet reaches the processor interface its address names, which
/// is why the pipe hands over the address the driver programmed
/// (`dev::kona`'s test of the packet it sends).
#[test]
fn a_packet_from_the_pipes_widget_reaches_the_processor_interface_its_address_names() {
    use crate::dev::bedrock::reg::{PI0_BASE, PI1_BASE};
    use crate::irq::InterruptController;
    let dir = TempPath::dir("o3k-ir-int");
    let chassis = Chassis::<Onyx3000>::build(&config(&dir, GraphicsBoard::Ir3), &prom(), Arc::new(|| 0)).unwrap();
    let bedrock = &chassis.nodes[0].bedrock;
    let port = Inbound::port(bedrock, PIPE_WIDGET);
    let pending = |base: u64| bedrock.registers().read64(base + 0x98).data & (1 << 12) != 0;
    assert!(!pending(PI1_BASE) && !pending(PI0_BASE));
    assert!(port.packet(PIPE_WIDGET, 0x8000_0180_0090 + PI1_BASE, 12, true));
    assert!(pending(PI1_BASE), "the second processor interface took the packet");
    assert!(!pending(PI0_BASE), "and the first did not");
}

/// A Crossbow link reset of the pipe's port resets the widget's Crosstalk
/// registers and leaves the board's own: the driver's adaptor reset follows it
/// with the board's reset pulse, and the board state in between (the map RAM
/// among it) must survive. The interrupt vectors are the widget's; the
/// interrupt enable is the board's.
#[test]
fn a_link_reset_of_the_pipes_port_leaves_the_boards_own_registers() {
    const INT_VECTORS: u64 = 0x5c;
    const INT_ENABLE: u64 = 0x80_0230;
    // Link 3 (widget 11) of the crossbar's registers in widget 0: its reset register's low word.
    const LINK3_RESET: u64 = 0x100 + 0x40 * 3 + 0x30 + 4;
    let dir = TempPath::dir("o3k-ir-linkreset");
    let chassis = Chassis::<Onyx3000>::build(&config(&dir, GraphicsBoard::Ir3), &prom(), Arc::new(|| 0)).unwrap();
    (chassis.io.start)();
    let pio = chassis.nodes[0].bedrock.pio();
    // The pipe applies its stores in order, and a read waits behind them.
    let read = |off: u64| loop {
        let r = pio.read32(widget(PIPE_WIDGET, off));
        match r.status {
            crate::traits::BUS_BUSY => std::thread::yield_now(),
            status => {
                assert_eq!(status, BUS_OK, "read of {off:#x}");
                break r.data;
            }
        }
    };
    let write = |off: u64, value: u32| loop {
        match pio.write32(widget(PIPE_WIDGET, off), value) {
            crate::traits::BUS_BUSY => std::thread::yield_now(),
            status => break assert_eq!(status, BUS_OK, "write of {off:#x}"),
        }
    };
    write(INT_VECTORS, 0x3e_2210);
    write(INT_ENABLE, 0xf7c);
    assert_eq!((read(INT_VECTORS), read(INT_ENABLE)), (0x3e_2210, 0xf7c));
    assert_eq!(pio.write32(widget(0, LINK3_RESET), 0), BUS_OK);
    assert_eq!((read(INT_VECTORS), read(INT_ENABLE)), (0, 0xf7c), "the widget's registers reset, the board's stay");
    (chassis.io.stop)();
}

fn refused(cfg: &MachineConfig) -> String {
    cfg.validate().unwrap_err()
}

/// The profile's limits: the pipe is a product the table ships in an Onyx
/// 3000; the C-bricks are the graphics brick's, not a key; the machine has
/// graphics, so none is the Origin 3200.
#[test]
fn the_configuration_refuses_what_the_onyx_does_not_fit() {
    let dir = TempPath::dir("o3k-ir-config");
    config(&dir, GraphicsBoard::Ir3).validate().unwrap();
    for board in [GraphicsBoard::Ir2, GraphicsBoard::Ir2e, GraphicsBoard::None, GraphicsBoard::V10, GraphicsBoard::V8] {
        assert!(refused(&config(&dir, board)).contains("Onyx 3000"), "{}", board.as_str());
    }
    // Its graphics board fixes the C-bricks, so `[onyx3000]` has no count.
    assert!(toml::from_str::<MachineConfig>("[onyx3000]\nc_bricks = 2").is_err());
    // The brick keys are read from `[onyx3000]`, and an error names it.
    let mut cfg = config(&dir, GraphicsBoard::Ir3);
    cfg.onyx3000.brick_serial = Some("X".repeat(38));
    assert!(refused(&cfg).starts_with("onyx3000.brick_serial"), "{}", refused(&cfg));
    let mut cfg = config(&dir, GraphicsBoard::Ir3);
    cfg.origin3200.brick_serial = Some("X".repeat(38));
    cfg.validate().unwrap();
    cfg.onyx3000.l1_port = Some(8882);
    assert_eq!(cfg.ip35_section().l1_port, Some(8882));
    assert_eq!(cfg.ip35_section().brick_serial, None, "`[origin3200]` is not read");
    let mut cfg = config(&dir, GraphicsBoard::Ir3);
    cfg.graphics.board = None;
    cfg.validate().unwrap();
    assert_eq!(cfg.graphics_board(), GraphicsBoard::Ir3, "the default");
}

/// The V-brick Onyx 3000 is two C-bricks, the I-brick on the first and the
/// V-brick on the second, built and freed together.
#[test]
fn the_v_brick_onyx_builds_and_is_freed() {
    let dir = TempPath::dir("o3k-vpro-free");
    let chassis = Chassis::<Onyx3000>::build(&config(&dir, GraphicsBoard::V12), &prom(), Arc::new(|| 0)).unwrap();
    assert_eq!(chassis.nodes.len(), 2, "a C-brick for the I-brick and one for the V-brick");
    assert!(chassis.windows.is_some(), "the two C-bricks are cabled");
    assert!(chassis.io.display.is_some() && chassis.io.status.is_some());
    let chips: Vec<_> = chassis.nodes.iter().map(|n| Arc::downgrade(&n.bedrock)).collect();
    drop(chassis);
    assert!(chips.iter().all(|c| c.upgrade().is_none()), "a chip is held by the V-brick's path");
}

/// The V12 reads back at widget 8 of the second C-brick's crossbar; the first
/// C-brick has no such widget, and carries no pipe at widget 11.
#[test]
fn the_v12_reads_back_at_widget_8_of_the_second_c_brick() {
    let dir = TempPath::dir("o3k-vpro-widget");
    let chassis = Chassis::<Onyx3000>::build(&config(&dir, GraphicsBoard::V12), &prom(), Arc::new(|| 0)).unwrap();
    let second = chassis.nodes[1].bedrock.pio();
    let id = second.read32(widget(V12_WIDGET, 0x04));
    assert_eq!(id.status, BUS_OK, "the V12 answers");
    assert_ne!(id.data, 0xffff_ffff);
    // The V-brick's crossbar is an XBridge at widget 0, with its host port.
    assert_eq!(second.read32(widget(0, 0)).status, BUS_OK);
    assert_eq!(second.read32(widget(10, 0)).status, BUS_OK);
    let first = chassis.nodes[0].bedrock.pio();
    assert_ne!(first.read32(widget(V12_WIDGET, 0x04)).status, BUS_OK, "the I-brick has no widget 8");
    assert_ne!(first.read32(widget(PIPE_WIDGET, 0x04)).status, BUS_OK, "and no pipe");
}

/// Each C-brick's controller names the brick on its cable: the I-brick on the
/// first, the V-brick on the second answering as an X-brick (approximation).
#[test]
fn each_c_bricks_controller_names_its_brick() {
    let dir = TempPath::dir("o3k-vpro-l1");
    let specs = <Onyx3000 as Ip35Board>::specs(&config(&dir, GraphicsBoard::V12)).unwrap();
    let attached: Vec<_> = specs.iter().map(|s| s.l1.attached.map(|b| (b.type_code, b.graphics_bay))).collect();
    assert_eq!(attached, [Some((0x49, None)), Some((0x58, None))]);
    let g = <Onyx3000 as Ip35Board>::specs(&config(&dir, GraphicsBoard::Ir3)).unwrap();
    assert_eq!(g.len(), 1);
    assert_eq!(g[0].l1.attached.map(|b| (b.type_code, b.graphics_bay)), Some((0x49, Some(G_BRICK_BAY))));
}
