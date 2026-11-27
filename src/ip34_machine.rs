//! The Fuel (IP34 board, IP35 software platform): an R14000 on Bedrock, XBridge and the base I/O,
//! selected with `[machine] profile = "fuel_ip34"`; `[fuel]` selects the console and flash. The
//! IOC3 carries PS/2 (default) and the OHCI controller can take a USB pair. `[graphics] board`
//! "v12" or "v10" puts a VPro at crossbar widget 13. The node's approximations are in `ip35::node`.
//!
//! # What is approximated here
//!
//! The graphics flow-control registers store what IRIX writes but count no stores, so the
//! credit-starvation interrupt never fires. The board's DMA reaches memory through Bedrock's
//! inbound path as widget 13.
//!
//! Board composition ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::{GraphicsBoard, MachineConfig};
use crate::cpu::parts::{self, Part};
use crate::dev::baseio_nvram;
use crate::dev::bedrock::{Bedrock, Population};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::clock::Clock;
use crate::dev::ioc3::{IP34, IP34_NO_PS2};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::{L1Config, L1};
use crate::dev::odyssey::Odyssey;
use crate::dev::battery_clock::YearBase;
use crate::dev::timekeeper::Timekeeper;
use crate::dev::xbridge::{XBridge, HEAD_E, HEAD_F, PCI_ROUTING};
use crate::hptimer::TimerManager;
use crate::ip35::{attach_ioc3, attach_usb, attach_vpro, xbridge_on};
use crate::ip35::node::NodeSpec;
use crate::ip35::serial::connect_serial_ports;
use crate::sn::BoardIo;
use crate::pci::DevFn;
use crate::topology::MachineSlotKind;
use crate::sn::NodeBoard;
use crate::traits::Resettable;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The Fuel's one node: its L1 answers with the configured station address,
/// and a configured revision or brick serial replaces the stock one.
pub(crate) fn node_specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
    let mut l1 = L1Config { mac: cfg.network().mac, ..Default::default() };
    if let Some(revision) = cfg.fuel.l1_revision.as_deref().map(crate::dev::l1::parse_revision).transpose()? {
        l1.revision = Some(revision);
    }
    if let Some(serial) = &cfg.fuel.brick_serial {
        l1.brick_serial = Some(serial.clone());
    }
    Ok(vec![NodeSpec {
        name: "Fuel",
        module: &parts::R14000_500_SYSCLK_200,
        firmware: FirmwareRecord::FUEL,
        board_id: FUEL_BOARD_ID,
        banks_mib: FUEL_BANKS_MIB.to_vec(),
        processors: Population::SINGLE,
        absent: "",
        l1,
        flash: cfg.fuel.flash.as_deref().map(PathBuf::from),
    }])
}

/// The Fuel's I/O around its node: the XBridge, the VPro in the graphics slot, and the base I/O.
pub(crate) fn build_io(cfg: &MachineConfig, nodes: &[NodeBoard<Bedrock, Flash>], l1s: &[Arc<L1>], timers: &Arc<TimerManager>, clock: &Clock) -> Result<BoardIo, String> {
    let bedrock = &nodes[0].chip;
    let xbridge = xbridge_on(bedrock);
    // The status bar's activity bits, shared with the base I/O's devices.
    let heartbeat = Arc::new(AtomicU64::new(0));
    let fasttick = Arc::new(AtomicU64::new(0));
    let status = crate::status::MachineStatus::new(heartbeat.clone(), fasttick.clone());
    status.set_scsi_wide(true);
    let graphics = cfg.graphics_board();
    let vpro = (graphics != GraphicsBoard::None)
        .then(|| attach_vpro(cfg, bedrock, WIDGET_VPRO, bedrock.interrupt_sender(WIDGET_VPRO, None), &status, |vpro| Ok(xbridge.attach_widget(WIDGET_VPRO, vpro))))
        .transpose()?;
    eprintln!("iris: {}", if vpro.is_some() { graphics.label() } else { "no graphics" });
    let mut io = build_baseio(cfg, &l1s[0], &xbridge, clock, timers, heartbeat, vpro.clone())?;
    io.status = Some(status);
    io.fasttick = Some(fasttick);
    if let Some(vpro) = vpro {
        io.display = Some(vpro.display());
        io.monitor.push(vpro.clone());
        io.devices.push(vpro.clone());
        let base_reset = std::mem::replace(&mut io.reset, Box::new(|| {}));
        io.reset = Box::new(move || {
            base_reset();
            vpro.power_on();
        });
    }
    Ok(io)
}

/// The ISP12160 on head F.
const SCSI_DEVICE: u8 = 1;

/// Head E's bus: the two slots of the 66 MHz bus. Which device numbers they
/// are is inferred: one owner-installed card was seen at device 2.
pub(crate) const HEAD_E_SLOTS: &[(u8, MachineSlotKind, Option<&str>)] = &[(1, MachineSlotKind::PCI_SLOT, None), (2, MachineSlotKind::PCI_SLOT, None)];

/// Head F's bus: the base I/O (ISP12160 at 1, IOC3 at 4, USB at 5), the two
/// slots of the 33 MHz bus at 2 and 3 (inferred, as head E's).
pub(crate) const HEAD_F_SLOTS: &[(u8, MachineSlotKind, Option<&str>)] = &[
        (SCSI_DEVICE, MachineSlotKind::Fixed("isp12160"), None),
        (2, MachineSlotKind::PCI_SLOT, None),
        (3, MachineSlotKind::PCI_SLOT, None),
        (4, MachineSlotKind::Fixed("ioc3"), None),
        (5, MachineSlotKind::Fixed("ohci"), None),
    ];

/// The Fuel's slots.
pub(crate) fn slots(s: &mut Vec<crate::topology::MachineSlot>) {
    crate::topology::open_slot(s, &format!("xio{WIDGET_VPRO}"), &[crate::topology::DeviceFamily::Gfx], &[]);
    crate::topology::pci_bus_slots(s, &format!("xio{HEAD_E}/pci0"), "Fuel XBridge head E", HEAD_E_SLOTS);
    crate::topology::pci_bus_slots(s, &format!("xio{HEAD_F}/pci0"), "Fuel XBridge head F", HEAD_F_SLOTS);
    // The ISP12160's two channels.
    for c in [0, 1] {
        crate::topology::scsi_bus_slots(s, &format!("xio{HEAD_F}/pci0/{SCSI_DEVICE}/c{c}"), "isp12160 channel", 15);
    }
}

/// The Fuel's base I/O on its XBridge's head F: the ISP12160, the IOC3 and the OHCI controller. A
/// link reset of head F resets them, and of the graphics port the `vpro` behind it.
fn build_baseio(
    cfg: &MachineConfig,
    l1: &Arc<L1>,
    xbridge: &Arc<XBridge>,
    clock: &Clock,
    timers: &Arc<TimerManager>,
    heartbeat: Arc<AtomicU64>,
    vpro: Option<Arc<Odyssey>>,
) -> Result<BoardIo, String> {
    // IRIX on the Fuel counts the Timekeeper's year cell from 1968.
    let timekeeper = Arc::new(Timekeeper::with_nvram_file(Path::new(&cfg.nvram), YearBase::From(1968), timers.clock().clone())?);
    let (serial, layout) = (timekeeper.with_ram(|r| baseio_nvram::console_is_serial(r)), baseio_nvram::Layout::Ip35 { graphics_default: true });
    crate::nv_storage::apply_headless_console(cfg.headless, "Fuel", &cfg.nvram, serial, || timekeeper.with_ram(|r| baseio_nvram::set_console_serial(r, layout)));
    let pci = xbridge.bus(HEAD_F);
    let board = if cfg.fuel.ps2_input { &IP34 } else { &IP34_NO_PS2 };
    let (ioc3, serial) = attach_ioc3(timers, cfg, pci, board, l1, &timekeeper, cfg.fuel.l1_port.unwrap_or(8882), "machine.l1_port", cfg.fuel.l1_log.clone());
    let eth = ioc3.eth().expect("the Fuel's IOC3 has Ethernet").clone();
    let scsi = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus(), timers));
    scsi.set_heartbeat(heartbeat.clone());
    ioc3.set_heartbeat(heartbeat);
    pci.attach(DevFn { device: SCSI_DEVICE, function: 0 }, scsi.clone());
    let usb = attach_usb(pci, &ioc3, timers, cfg.fuel.usb_input)?;
    let (reset_ioc3, reset_scsi, reset_usb) = (ioc3.clone(), scsi.clone(), usb.clone());
    let reset_vpro = vpro.as_ref().map(Arc::downgrade);
    xbridge.set_reset_output(Arc::new(move |port| {
        if port == HEAD_F {
            reset_ioc3.power_on();
            reset_scsi.power_on();
            reset_usb.power_on();
        }
        // A link reset of the graphics port resets the board behind it.
        if port == WIDGET_VPRO {
            if let Some(vpro) = reset_vpro.as_ref().and_then(|v| v.upgrade()) {
                vpro.power_on();
            }
        }
    }));
    let (table, product) = (cfg.machine.profile.table_name().unwrap_or_default(), cfg.machine.profile.product_name().unwrap_or_default());
    let buses = [0, 1].map(|c| scsi.bus_for_channel(c).expect("the ISP12160 has two channels"));
    let mut resolved = crate::topology::DevicePlacement::new(cfg)?;
    for head in [HEAD_E, HEAD_F] {
        resolved.bind_pci(&format!("xio{head}/pci0"), xbridge.bus(head).clone())?;
    }
    let scratch = crate::scsi::attach_without_daynaport(cfg, table, product, &resolved, &buses)?;
    let cards = Arc::new(resolved.build_cards(cfg, timers)?);
    let console_index = cfg.fuel.console.index();
    connect_serial_ports(cfg, "Fuel", &serial, console_index)?;
    let console = cfg.ci.then(|| serial[console_index].ci.clone());
    scsi.register_locks();
    ioc3.register_locks();
    xbridge.register_locks();
    let (reset_xbridge, reset_ioc3, reset_scsi, reset_usb, reset_clock) = (xbridge.clone(), ioc3.clone(), scsi.clone(), usb.clone(), timekeeper.clone());
    let disks = scsi.clone();
    let monitor: Vec<Arc<dyn crate::traits::Device>> = vec![scsi.clone(), ioc3.ps2(), crate::pci::pci_host_commands("XBridge", &[xbridge.bus(HEAD_E), pci], None)];
    let devices: Vec<Arc<dyn crate::traits::Device>> = vec![ioc3.clone(), scsi.clone(), usb];
    Ok(BoardIo {
        devices,
        reset: Box::new(move || {
            reset_xbridge.power_on();
            reset_ioc3.power_on();
            reset_scsi.power_on();
            reset_usb.power_on();
            reset_clock.power_on();
        }),
        at_batch: None,
        monitor,
        ps2: ioc3.ps2(),
        flush_nvram: Box::new(move || timekeeper.flush_nvram()),
        console,
        attach_net: Box::new(move |link| eth.attach(link)),
        disks: Box::new(move |f| f(&(0..2).filter_map(|channel| disks.bus_for_channel(channel)).collect::<Vec<_>>())),
        disc_changer: Some(scsi),
        scratch,
        display: None,
        status: None,
        fasttick: None,
        cpus: None,
    }
    .with_cards(&cards))
}

/// The processors a Fuel can be built with, the default first: the R14000A
/// the Fuel ships with.
pub const CPUS: &[&Part] = &[&parts::R14000_500];

/// The Fuel's board-id word: class 4 (bits 15:12; OpenBSD names it IP35_FUEL), nibble 4 (the
/// PROM 6.210 speed-table row of the 500 MHz, 2 MB module; established). Bits 11:4 and 63:16
/// of a real word are unknown and read zero.
pub(crate) const FUEL_BOARD_ID: u64 = 0x4004;
/// Two 256 MiB banks, locational banks 0 and 1.
const FUEL_BANKS_MIB: [u32; 2] = [256, 256];

/// The crossbar widget of the Fuel's graphics slot: a Fuel IRIX inventory
/// lists its VPro under xtalk/13, and OpenBSD finds Odyssey at widget 13.
pub const WIDGET_VPRO: u8 = 13;

#[cfg(test)]
mod tests;
