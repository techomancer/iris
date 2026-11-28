use crate::ip35::test_support::{bus_of, fuel_config, fuel_cpu, headless, pi_regs, pio_widget, tezro_config, with_machine, Head};
use crate::physical::Physical;
use crate::test_temp::TempPath;
use crate::traits::*;
use std::sync::Arc;

/// The VPro's link status in the crossbar, in its slot's data half.
const VPRO_LINK_STATUS: u64 = 0x100 + 5 * 0x40 + 0x10 + 4;
/// A processor access to widget 13 through Bedrock's small window.
const VPRO_PIO: u64 = 13 << 24;
// The VPro cases follow tests/qtest/sn1-fuel-vpro-test.c in
// jamesbraid/qemu-sgi-origami: widget-identity, profile-identity-and-routing
// and headless-fuel-has-no-board.

// VPro host-interface registers the tests below touch, as byte offsets in
// the board's widget space. The board's own constants are private to it.
const ODSY_IDENTITY: u64 = 0x00;
const ODSY_CONTROL: u64 = 0x20;
const ODSY_CONTROL_WIDGET_MASK: u32 = 0xf;
const ODSY_INT_DEST_LOWER: u64 = 0x38;
const ODSY_INTR_CLEAR: u64 = 0x101c;
const ODSY_INTR_ENABLE_SET: u64 = 0x1034;
const ODSY_INTR_FIFO_LOW_WATER: u32 = 0x200;
const ODSY_ROUTE_DEST_UPPER: u64 = 0x1044;
const ODSY_ROUTE_DEST_LOWER: u64 = 0x104c;
const ODSY_ROUTE_VECTORS: u64 = 0x1054;
const ODSY_ROUTE_DEST_WIDGET_SHIFT: u32 = 25;

/// An uncached SB or SH reaches a device with the whole source register, at the
/// device-local address; only a device that overrides the call sees more than
/// the low bits, and a window outside every map is a bus error.
#[test]
fn the_fuel_runs_the_r14000_at_revision_2_3() {
    use crate::cpu::mips_cache_shadow::R12000ShadowCache;
    use crate::cpu::mips_cache_v2::CpuModel as Model;
    use crate::cpu::parts;
    use crate::ip34_machine::CPUS;
    assert_eq!(CPUS, &[&parts::R14000_500]);
    let part = CPUS[0];
    assert_eq!((part.mhz, part.rev, part.l2_kb), (500, (2, 3), 2048));
    assert_eq!(part.prid(), 0x0f23);
    assert_eq!(part.count_hz(), 250_000_000);
    // `with_cpu!` builds the type `FuelCpu` names for this part.
    let built = crate::with_cpu!(part, true, false, C => std::any::type_name::<C>());
    assert_eq!(built, std::any::type_name::<R12000ShadowCache>());
    // Typed R12000, it takes the R10000 family's behaviour: reserved
    // secondary-tag bits read as zero, and a refused uncached store posts.
    const { assert!(<R12000ShadowCache as Model>::R10K_FAMILY) };
    assert_ne!(<R12000ShadowCache as Model>::SECONDARY_TAG_MASK, u64::MAX);
    const { assert!(<R12000ShadowCache as Model>::POSTS_UNCACHED_STORES) };
    const { assert!(<R12000ShadowCache as Model>::PARTIAL_LWR_KEEPS_UPPER) };
    // The executor carries the chip's PRId.
    // The executor is large; the default test stack is not.
    crate::display_stop_tests::big_stack(|| {
        let cpu = fuel_cpu(Arc::new(bus_of(Default::default())), Physical::new(Vec::new()));
        assert_eq!(cpu.core.cp0_prid, 0x0f23);
    });
}
/// The Fuel's flash record and its processor module are two statements of one mode word,
/// which the PROM's early check compares; the processor resets with the module's Config.
#[test]
fn the_fuels_flash_mode_word_matches_its_cpu_module() {
    let dir = TempPath::dir("fuel-mode");
    let specs = crate::ip34_machine::node_specs(&fuel_config(&dir)).unwrap();
    let spec = &specs[0];
    assert_eq!(spec.firmware.cpu_mode_word, spec.module.mode_word);
    assert_eq!(*spec.module.part, *crate::ip34_machine::CPUS[0]);
    assert_eq!(u64::from(spec.firmware.cpu_clock_hz), spec.module.clock_hz());
}
/// The flash record has to agree with the board-id word's speed row; the PROM reports a mismatch
/// otherwise.
#[test]
fn the_fuels_board_word_is_class_4_with_the_500_mhz_row() {
    let dir = TempPath::dir("fuel-board-word");
    with_machine(fuel_config(&dir), |m| {
        let word = m.nodes[0].bus.read64(0x1400_0000);
        assert_eq!(word.status, BUS_OK);
        assert_eq!(word.data, 0x4004);
        assert_eq!(word.data, crate::ip34_machine::FUEL_BOARD_ID);
    });
}
/// Firmware resets the PCI link before reinitializing the attached storage controller.
#[test]
fn fuel_link_reset_resets_the_attached_scsi_engine() {
    use crate::dev::isp1040::{mbox, reg, CTRL_INT_ENABLE, CTRL_RISC_INT_ENABLE, HOST_CMD_SHIFT};
    let dir = TempPath::dir("fuel-link-reset");
    with_machine(fuel_config(&dir), |m| {
        let pio = m.nodes[0].chip.pio();
        let head = Head::new(pio.clone(), 15);
        // The ISP12160 is device 1 of head F: its memory BAR is the second.
        let bar = |head: &Head| head.map_bar(1, 0x14, 0x0200_0000);
        let win = bar(&head);
        win.write16(u64::from(reg::BUS_CTRL), CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE);
        win.write16(u64::from(reg::MAILBOX_BASE), mbox::NOP);
        win.write16(u64::from(reg::HOST_CMD), 0x5 << HOST_CMD_SHIFT);
        assert_eq!(win.read16(u64::from(reg::MAILBOX_BASE)), u32::from(mbox::COMPLETE));
        assert_ne!(win.read16(u64::from(reg::BUS_CTRL)), 0);
        // Link 15's reset register in widget 0's crossbar.
        assert_eq!(pio.write32(pio_widget(0, 0x2f4), 1), BUS_OK);
        assert_eq!(head.cfg_read(1, 4) & 0xffff, 0, "the command register is cleared");
        assert_eq!(head.cfg_read(1, 0x14), 0, "and so is the BAR");
        let win = bar(&head);
        assert_eq!(win.read16(u64::from(reg::BUS_CTRL)), 0);
        assert_eq!(win.read16(u64::from(reg::MAILBOX_BASE)), 0);
        assert_eq!(win.read16(u64::from(reg::MAILBOX_BASE) + 2), 0x4953);
    });
}
/// The Fuel's OHCI root ports each carry a low-speed device only when the
/// USB pair is configured, and then the host's input, which still reaches
/// the IOC3's PS/2 controller object, goes on to them.
#[test]
fn the_fuel_plugs_a_usb_keyboard_and_mouse_only_when_configured_on() {
    let dir = TempPath::dir("fuel-usb");
    // The OHCI controller is device 5 of head F; its root ports' status registers are at 0x54
    // and 0x58 of its first BAR, and a port is powered by writing 0x100 there.
    let ports = |usb_input| {
        let mut cfg = fuel_config(&dir);
        cfg.fuel.usb_input = usb_input;
        with_machine(cfg, |m| {
            let win = Head::new(m.nodes[0].chip.pio(), 15).map_bar(5, 0x10, 0x0300_0000);
            win.write32(0x54, 0x100);
            win.write32(0x58, 0x100);
            let status = [0x54, 0x58].map(|at| win.read32(at));
            let ps2 = m.io.ps2.clone();
            // The guest turns the PS/2 mouse on.
            ps2.write_to(0xf4, true);
            (status, ps2.mouse_enabled())
        })
    };
    let (status, mouse) = ports(true);
    for s in status {
        assert_eq!(s & 1, 1, "device connected");
        assert_eq!(s & 0x200, 0x200, "low speed");
    }
    assert!(!mouse, "the USB mouse answers for the host's pointer");
    let (status, mouse) = ports(false);
    assert_eq!(status.map(|s| s & 1), [0, 0], "nothing plugged in");
    assert!(mouse, "without the USB pair the PS/2 mouse does");
}
/// An `ice1724` entry puts the ICE1724 in the slot it names, here device 2 of head F, its two I/O
/// BARs reachable once placed; without one the slot is empty.
#[test]
fn the_fuel_takes_the_audio_card_in_a_slot() {
    let dir = TempPath::dir("fuel-audio");
    let slot = |audio: bool| {
        let mut cfg = fuel_config(&dir);
        cfg.devices.extend(audio.then(|| crate::devices::ConfiguredDevice::new("xio15/pci0/2", crate::devices::DeviceModel::Ice1724)));
        cfg.no_audio = true;
        with_machine(cfg, |m| {
            let read = m.nodes[0].chip.pio().read32(pio_widget(15, 0x2_0000 + (2 << 12)));
            ((read.status == BUS_OK).then_some(read.data), m.io.monitor.iter().any(|d| d.audio_report().is_some()))
        })
    };
    assert_eq!(slot(true), (Some(0x1724_1412), true));
    let (id, report) = slot(false);
    assert!(id.is_none_or(|id| id == u32::MAX) && !report, "empty without the key");
}

/// The check agrees with the constructor: an NVMe card is refused before the machine is built.
#[test]
fn an_nvme_card_is_refused_by_the_check() {
    let dir = TempPath::dir("fuel-nvme");
    let mut cfg = fuel_config(&dir);
    cfg.devices.push(crate::devices::ConfiguredDevice::from_table(toml::from_str("type = \"nvme\"\npath = \"n.img\"").unwrap()).unwrap());
    assert!(cfg.validate().unwrap_err().contains("the Fuel takes no NVMe card yet"));
}

/// A Fibre Channel adapter in a slot of head F is built, with the disk placed on its loop,
/// and not dropped the way every machine but the Octane used to.
#[test]
fn a_fibre_channel_adapter_in_a_slot_is_built_with_its_loop() {
    let dir = TempPath::dir("fuel-fibre");
    let entry = |text: &str| crate::devices::ConfiguredDevice::from_table(toml::from_str(text).unwrap()).unwrap();
    let disk = TempPath::file("fuel-fibre-disk", &[0u8; 4096]);
    let mut cfg = fuel_config(&dir);
    cfg.devices.extend([
        entry("type = \"qla2200\"\nat = \"xio15/pci0/2\"\nname = \"fc\""),
        entry(&format!("type = \"hdd\"\nat = \"fc/l5\"\npath = {:?}", disk.to_string_lossy())),
    ]);
    with_machine(cfg, |m| {
        let id = |device: u64| m.nodes[0].chip.pio().read32(pio_widget(15, 0x2_0000 + (device << 12))).data;
        assert_eq!(id(2) & 0xffff, 0x1077, "the adapter answers in the slot");
    });
}

/// A Fuel with `graphics` in its graphics slot.
fn fuel_with_graphics(dir: &TempPath, graphics: Option<crate::config::GraphicsBoard>) -> crate::config::MachineConfig {
    let mut cfg = fuel_config(dir);
    cfg.set_graphics(graphics.map(crate::devices::BoardConfig::new));
    cfg
}
/// A Fuel IRIX inventory lists its VPro under xtalk/13: the processor must
/// reach the board there, learn its widget number from it, and find the link alive in the crossbar.
#[test]
fn the_vpro_answers_at_widget_13_with_its_link_alive() {
    let dir = TempPath::dir("fuel-vpro-id");
    for (graphics, revision) in [(crate::config::GraphicsBoard::V12, 0xa), (crate::config::GraphicsBoard::V10, 0xa)] {
        with_machine(fuel_with_graphics(&dir, Some(graphics)), move |m| {
            let pio = m.nodes[0].chip.pio();
            let identity = pio.read32(VPRO_PIO | (ODSY_IDENTITY + 4));
            assert_eq!(identity.status, BUS_OK);
            assert_eq!(identity.data, (revision << 28) | 0x0c01_3047, "{graphics:?}");
            let control = pio.read32(VPRO_PIO | (ODSY_CONTROL + 4)).data;
            assert_eq!(control & ODSY_CONTROL_WIDGET_MASK, 13);
            assert_ne!(pio.read32(pio_widget(0, VPRO_LINK_STATUS)).data & (1 << 31), 0);
        });
    }
}
/// The monitor's `vpro` commands resolve on a Fuel or a Tezro that has a VPro, and are unknown on
/// one that has none.
#[test]
fn the_monitor_resolves_vpro_commands_only_with_a_vpro() {
    let dir = TempPath::dir("vpro-monitor");
    for (what, plain) in [("Fuel", fuel_config(&dir)), ("Tezro", tezro_config(&dir, 1))] {
        let mut with_vpro = plain.clone();
        with_vpro.set_graphics(Some(crate::devices::BoardConfig::new(crate::config::GraphicsBoard::V12)));
        for (cfg, vpro) in [(with_vpro, true), (plain, false)] {
            let seen = with_machine(cfg, |m| {
                let has = |command: &str| m.io.monitor.iter().any(|d| d.register_commands().iter().any(|(c, _)| c == command));
                (has("vpro"), has("ps2"))
            });
            assert_eq!(seen, (vpro, true), "{what}: `vpro` only with a V12; the other board commands stay");
        }
    }
}
/// A VPro's interrupt packets reach its node's Bedrock as writes to the address the driver
/// programmed: a raised source sets the pending bit of the processor interface that address names
/// and a cleared one clears it. The Fuel (widget 13), the Tezro (12) and the Onyx 3000's V-brick
/// (8, on the second node) take the same path.
#[test]
fn a_vpro_interrupt_packet_sets_and_clears_the_pending_bit_its_address_names() {
    use crate::config::{GraphicsBoard, MachineProfile};
    let dir = TempPath::dir("vpro-irq");
    let mut fuel = fuel_config(&dir);
    fuel.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::V12)));
    let mut tezro = tezro_config(&dir, 1);
    tezro.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::V12)));
    let mut vbrick = headless(&dir, MachineProfile::Onyx3000Ip35);
    vbrick.set_graphics(Some(crate::devices::BoardConfig::new(GraphicsBoard::V12)));
    vbrick.onyx3000.l1_port = Some(0);
    for (what, cfg, node, widget) in [("Fuel", fuel, 0, 13), ("Tezro", tezro, 0, 12), ("V-brick", vbrick, 1, 8)] {
        with_machine(cfg, move |m| {
            // Register space (bit 47), the node's hub window and PI0's pending modify register, at
            // the Bedrock's crossbar port 0xa.
            let packet = (1u64 << 47) | 0x180_0000 | 0x90;
            let vector = 5;
            let pio = m.nodes[node].chip.pio();
            let write = |offset: u64, value: u64| {
                assert_eq!(pio.write32(pio_widget(widget, offset), value as u32), BUS_OK, "{what} {offset:#x}");
            };
            write(ODSY_ROUTE_DEST_UPPER, (0xa << ODSY_ROUTE_DEST_WIDGET_SHIFT) | (packet >> 32));
            write(ODSY_ROUTE_DEST_LOWER, packet & 0xffff_ffff);
            write(ODSY_ROUTE_VECTORS, vector);
            write(ODSY_INTR_ENABLE_SET, ODSY_INTR_FIFO_LOW_WATER as u64);
            let pending = || m.nodes[node].chip.registers().read64(0x98).data & (1 << vector);
            assert_ne!(pending(), 0, "{what}");
            write(ODSY_INTR_CLEAR, ODSY_INTR_FIFO_LOW_WATER as u64);
            assert_eq!(pending(), 0, "{what}");
        });
    }
}
/// A link reset of the graphics port resets the board behind it, as the
/// PCI port's does for the functions behind its head.
#[test]
fn a_graphics_link_reset_resets_the_vpro() {
    let dir = TempPath::dir("fuel-vpro-reset");
    with_machine(fuel_with_graphics(&dir, Some(crate::config::GraphicsBoard::V12)), |m| {
        let pio = m.nodes[0].chip.pio();
        let destination = VPRO_PIO | (ODSY_INT_DEST_LOWER + 4);
        assert_eq!(pio.write32(destination, 0x55), BUS_OK);
        assert_eq!(pio.read32(destination).data, 0x55);
        let link_reset = 0x100 + 5 * 0x40 + 0x30 + 4;
        assert_eq!(pio.write32(pio_widget(0, link_reset), 1), BUS_OK);
        assert_eq!(pio.read32(destination).data, 0);
    });
}
/// Without graphics nothing answers at widget 13 and its link is down, as on a headless Fuel.
#[test]
fn a_fuel_without_graphics_has_no_widget_13() {
    let dir = TempPath::dir("fuel-no-vpro");
    with_machine(fuel_with_graphics(&dir, None), |m| {
        let pio = m.nodes[0].chip.pio();
        assert!(m.io.display.is_none());
        assert_ne!(pio.read32(VPRO_PIO | 4).status, BUS_OK);
        assert_eq!(pio.read32(pio_widget(0, VPRO_LINK_STATUS)).data & (1 << 31), 0);
    });
}
/// IRIX's VPro driver writes each processor interface's graphics credit
/// counter at attach and tears the board down if it reads back zero; the
/// registers must keep what the guest writes, per interface.
#[test]
fn processor_interface_graphics_registers_keep_their_writes() {
    let dir = TempPath::dir("fuel-pi-graphics");
    with_machine(fuel_with_graphics(&dir, Some(crate::config::GraphicsBoard::V12)), |m| {
        // Slot A's credit counter and bias; PI1 keeps its own block.
        assert_eq!(pi_regs(m, 0).write64(0x308, 100), BUS_OK);
        assert_eq!(pi_regs(m, 0).write64(0x310, 0x1000 | 68), BUS_OK);
        assert_eq!(pi_regs(m, 0).read64(0x308).data, 100);
        assert_eq!(pi_regs(m, 0).read64(0x310).data, 68);
        assert_eq!(pi_regs(m, 1).read64(0x308).data, 0);
        m.nodes[0].chip.power_on();
        assert_eq!(pi_regs(m, 0).read64(0x308).data, 0);
    });
}
