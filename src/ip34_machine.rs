//! The Fuel (IP34 board, IP35 software platform): an R14000 on Bedrock,
//! XBridge and the base I/O devices, with Iris's CPU thread, monitor and serial console.
//!
//! Selected with `[machine] profile = "fuel_ip34"`. Shared keys select the
//! PROM, NVRAM, memory and SCSI devices; `[fuel]` selects the serial console
//! and writable flash. `sn_nodes::UncachedBus` maps the processor views over the
//! same Physical memory banks. The IOC3 carries PS/2 keyboard and mouse ports
//! (default) and the OHCI controller can take a USB pair; without graphics the
//! machine runs on a serial console. With `[graphics] board = "v12"` (or
//! `"v10"`) a VPro board answers at crossbar widget 13, the Fuel's graphics
//! slot.
//!
//! The Bedrock node itself, and its approximations, are in `ip35::node`.
//!
//! # What is approximated here
//!
//! The processor interfaces' graphics flow-control registers store what
//! IRIX writes, which is what its VPro driver checks, but count no stores:
//! no uncached store into the graphics page takes a credit and the board
//! returns none, so the credit-starvation interrupt never fires. The board's
//! DMA reaches memory through Bedrock's inbound path as widget 13.
//!
//! Board composition ported from hw/mips/sgi/sn1-node.c
//! in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::{scsi_label, FuelConsole, GraphicsBoard, MachineConfig};
use crate::cpu::parts::{self, Part};
use crate::dev::bedrock::{Bedrock, Population};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::heart::host_clock;
use crate::dev::ioc3::{ByteBus, Ioc3, IP34, IP34_NO_PS2};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::{L1Config, L1};
use crate::dev::odyssey::{Odyssey, PacketSender};
use crate::dev::ohci::hid::{HidFunction, UsbHidInput};
use crate::dev::ohci::Ohci;
use crate::dev::serial_host::{CiSerialBackend, SerialBackend};
use crate::dev::timekeeper::Timekeeper;
use crate::dev::xbridge::XBridge;
use crate::hptimer::TimerManager;
use crate::ip35::BEDROCK_PORT;
use crate::ip35_machine::{BoardIo, NodeSpec, SerialPort};
use crate::pci::{BoardIrq, DevFn};
use crate::scsi::attach_from_config;
use crate::sn_nodes::NodeBoard;
use crate::traits::{BusDevice, Device, Resettable};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The Fuel's one node: its L1 answers with the configured station address,
/// and a configured revision or brick serial replaces the stock one.
pub(crate) fn specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
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

/// The Fuel's base I/O around its node: the XBridge, the IOC3, the ISP12160
/// and the OHCI controller on head F, and a VPro in the graphics slot.
pub(crate) fn wire(cfg: &MachineConfig, nodes: &[NodeBoard<Bedrock, Flash>], l1s: &[Arc<L1>], _: &Arc<TimerManager>) -> Result<BoardIo, String> {
    let (bedrock, l1) = (&nodes[0].chip, &l1s[0]);
    let graphics = cfg.graphics_board();
    let timekeeper = Arc::new(Timekeeper::with_nvram_file(Path::new(&cfg.nvram))?);
    let serial: [Arc<CiSerialBackend>; 3] = std::array::from_fn(|_| Arc::new(CiSerialBackend::new()));
    l1.set_serial_backend(serial[0].clone());
    let xbridge = XBridge::new(BEDROCK_PORT, &|widget| {
        let port = bedrock.inbound(widget);
        (port.clone() as Arc<dyn BusDevice>, port as Arc<dyn crate::irq::InterruptController>)
    });
    for widget in [0, crate::dev::xbridge::HOST_WIDGET, 14, 15] {
        bedrock.connect_widget(widget, xbridge.widget(widget).unwrap());
    }
    let mac = l1.station_mac();
    let board = if cfg.fuel.ps2_input { &IP34 } else { &IP34_NO_PS2 };
    let ioc3 = Arc::new(Ioc3::new(board, mac, ByteBus::M48t35(timekeeper.clone())));
    let eth = ioc3.eth().expect("the Fuel's IOC3 fits Ethernet").clone();
    ioc3.set_serial_backend(0, serial[1].clone());
    ioc3.set_serial_backend(1, serial[2].clone());
    let scsi = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus()));
    let heartbeat = Arc::new(AtomicU64::new(0));
    let fasttick = Arc::new(AtomicU64::new(0));
    scsi.set_heartbeat(heartbeat.clone());
    ioc3.set_heartbeat(heartbeat.clone());
    let status = crate::disp::StatusSource::new(heartbeat.clone(), fasttick.clone());
    let vpro = if graphics == GraphicsBoard::None {
        None
    } else {
        // Interrupt packets reach Bedrock as the heads' do; DMA enters
        // it as this widget's inbound traffic.
        let port = bedrock.inbound(WIDGET_VPRO);
        let dma: Arc<dyn BusDevice> = port.clone();
        let packets: PacketSender = Arc::new(move |widget, address, vector, level| {
            use crate::irq::InterruptController;
            widget == BEDROCK_PORT.widget && port.packet(WIDGET_VPRO, address, vector.into(), level)
        });
        let board = Odyssey::new_widget_relative(
            WIDGET_VPRO,
            graphics.as_str(),
            packets,
            dma,
            status.clone(),
            (cfg.graphics.threads, cfg.graphics.tile),
            (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()),
        )?;
        bedrock.connect_widget(WIDGET_VPRO, xbridge.attach_widget(WIDGET_VPRO, board.clone()));
        Some(board)
    };
    let pci = xbridge.bus(15);
    let slot = |device| DevFn { device, function: 0 };
    pci.attach(slot(1), scsi.clone());
    pci.attach(slot(4), ioc3.clone());
    let usb = Ohci::new();
    usb.set_clock(host_clock());
    if cfg.fuel.usb_input {
        // As on the Origin 3200's I-brick: the keyboard on root port 1,
        // the mouse on port 2, and the host's input, which the UI, VNC
        // and the monitor still hand to the PS/2 controller, goes to
        // them instead. Feeding both pairs would type every key twice
        // to a guest that opened both keyboards.
        let (keyboard, mouse) = (HidFunction::keyboard(), HidFunction::mouse());
        usb.attach(0, keyboard.clone())?;
        usb.attach(1, mouse.clone())?;
        ioc3.ps2().redirect_input(UsbHidInput::new(keyboard, mouse));
    }
    pci.attach(slot(5), usb.clone());
    ioc3.connect_superio_interrupt(pci.board_irq(BoardIrq(0)));
    let (reset_ioc3, reset_scsi, reset_usb) = (ioc3.clone(), scsi.clone(), usb.clone());
    let reset_vpro = vpro.as_ref().map(Arc::downgrade);
    xbridge.set_reset_output(Arc::new(move |port| {
        if port == 15 {
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
    let scratch = attach_scsi(cfg, &scsi)?;
    let console = match cfg.fuel.console {
        FuelConsole::L1 => 0,
        FuelConsole::Ioc3A => 1,
        FuelConsole::Ioc3B => 2,
    };
    eprintln!("iris: Fuel IP34, {}", if vpro.is_some() { graphics.label() } else { "no graphics" });
    let mut monitor: Vec<Arc<dyn Device>> = vec![scsi.clone(), ioc3.ps2()];
    if let Some(vpro) = &vpro {
        monitor.push(vpro.clone());
    }
    let (start_ioc3, start_scsi, start_usb, start_vpro) = (ioc3.clone(), scsi.clone(), usb.clone(), vpro.clone());
    let (stop_ioc3, stop_scsi, stop_usb, stop_vpro) = (ioc3.clone(), scsi.clone(), usb.clone(), vpro.clone());
    let (reset_xbridge, reset_ioc3, reset_scsi, reset_usb, reset_vpro) =
        (xbridge.clone(), ioc3.clone(), scsi.clone(), usb.clone(), vpro.clone());
    let poll_usb = usb.clone();
    let (lock_scsi, lock_ioc3, lock_xbridge) = (scsi.clone(), ioc3.clone(), xbridge.clone());
    let disks = scsi.clone();
    Ok(BoardIo {
        start: Box::new(move || {
            start_ioc3.start();
            start_scsi.start();
            start_usb.start();
            if let Some(vpro) = &start_vpro {
                vpro.start();
            }
        }),
        stop: Box::new(move || {
            if let Some(vpro) = &stop_vpro {
                vpro.stop();
            }
            stop_usb.stop();
            stop_scsi.stop();
            stop_ioc3.stop();
        }),
        reset: Box::new(move || {
                reset_xbridge.power_on();
            reset_ioc3.power_on();
            reset_scsi.power_on();
            reset_usb.power_on();
            if let Some(vpro) = &reset_vpro {
                vpro.power_on();
            }
        }),
        poll: Arc::new(move || poll_usb.sync_clock()),
        at_batch: None,
        register_locks: Box::new(move || {
            lock_scsi.register_locks();
            lock_ioc3.register_locks();
            lock_xbridge.register_locks();
        }),
        monitor,
        ps2: ioc3.ps2(),
        flush_nvram: Box::new(move || timekeeper.flush_nvram()),
        serial: serial_ports(cfg, l1, &ioc3, &serial),
        console,
        attach_net: Box::new(move |link| eth.attach(link)),
        disks: Box::new(move |f| f(&(0..2).filter_map(|channel| disks.bus_for_channel(channel)).collect::<Vec<_>>())),
        disc_changer: Some(scsi),
        scratch,
        display: vpro.as_ref().map(|vpro| vpro.display()),
        status: Some(status),
        fasttick: Some(fasttick),
        cpus: Box::new(|_| {}),
        #[cfg(test)]
        xbridge: Some(xbridge),
        #[cfg(test)]
        pics: Vec::new(),
    })
}

fn attach_scsi(cfg: &MachineConfig, scsi: &Isp12160) -> Result<Option<PathBuf>, String> {
    let attached = attach_from_config(cfg, Some("fuel"), &|channel| {
        scsi.bus_for_channel(channel)
            .ok_or_else(|| format!("controller {channel} (a Fuel has channels 0 and 1)"))
    })?;
    if let Some(device) = attached.daynaports.first() {
        return Err(format!("{}: kind = \"daynaport\" is not available on the Fuel",
            scsi_label(device.controller, device.id)));
    }
    Ok(attached.scratch)
}

/// The board's serial connections in console order: the L1, then the IOC3's
/// two ports.
fn serial_ports(cfg: &MachineConfig, l1: &Arc<L1>, ioc3: &Arc<Ioc3>, serial: &[Arc<CiSerialBackend>; 3]) -> Vec<SerialPort> {
    let [a, b] = cfg.serial_ports();
    let ioc3 = |port: usize| -> Box<dyn Fn(Arc<dyn SerialBackend>) + Send + Sync> {
        let ioc3 = ioc3.clone();
        Box::new(move |backend| ioc3.set_serial_backend(port, backend))
    };
    let l1 = l1.clone();
    vec![
        SerialPort {
            label: "L1",
            port: cfg.fuel.l1_port.unwrap_or(8882),
            setting: "fuel.l1_port",
            ci: serial[0].clone(),
            log: cfg.fuel.l1_log.clone(),
            connect: Box::new(move |backend| l1.set_serial_backend(backend)),
        },
        SerialPort { label: "IOC3 A", port: a, setting: "serial_port_a", ci: serial[1].clone(), log: None, connect: ioc3(0) },
        SerialPort { label: "IOC3 B", port: b, setting: "serial_port_b", ci: serial[2].clone(), log: None, connect: ioc3(1) },
    ]
}

/// The processors a Fuel can be built with, the default first: the R14000A
/// the Fuel ships with.
pub const CPUS: &[&Part] = &[&parts::R14000_500];


/// A row of the PROM's CPU speed table, found by the board-id word's low
/// nibble when its class (bits 15:12) is 4, the Fuel's. The PROM prints
/// "Unknown PIMM_PSC type" for a nibble with no row (0, 5, f) and falls back
/// to zeros. The hub clock is 200 MHz in every row (established, from the
/// PROM's code). That the nibble is the processor module's own code is
/// inferred; the rows say nothing of what else the word holds.
pub(crate) struct SpeedRow {
    pub(crate) nibble: u64,
    pub(crate) mhz: u32,
    pub(crate) l2_kb: u32,
}

/// Every row of the PROM 6.210 table (established). Two nibbles share 500 MHz
/// (4 and 6) and two 600 MHz (7 and 8); the first listed is the one used.
/// The 700 MHz row is the R16000.
pub(crate) const SPEED_ROWS: [SpeedRow; 7] = [
    SpeedRow { nibble: 0x4, mhz: 500, l2_kb: 2048 },
    SpeedRow { nibble: 0x6, mhz: 500, l2_kb: 2048 },
    SpeedRow { nibble: 0x7, mhz: 600, l2_kb: 4096 },
    SpeedRow { nibble: 0x8, mhz: 600, l2_kb: 4096 },
    SpeedRow { nibble: 0x3, mhz: 700, l2_kb: 4096 },
    SpeedRow { nibble: 0xc, mhz: 800, l2_kb: 4096 },
    SpeedRow { nibble: 0x2, mhz: 900, l2_kb: 8192 },
];

/// The class bits 15:12 of the board-id word for a Fuel. The PROM masks them
/// and compares 0x2000, 0x4000, 0x8000 and 0xc000; the kernel compares 4, 8
/// and c (established). OpenBSD names 4 IP35_FUEL.
const FUEL_BOARD_CLASS: u64 = 0x4000;

/// The board-id word of a Fuel built with `part`, or `None` when the PROM's
/// speed table has no row for its clock and secondary cache. The PROM checks
/// the flash record's mode word, clocks, timing bytes and machine type against
/// this row, so the word has to follow the processor.
///
/// Bits 11:4 and 63:16 of a real board's word are unknown; they read zero
/// here and the PROM does not look at them.
pub(crate) const fn board_word(part: &Part) -> Option<u64> {
    let mut i = 0;
    while i < SPEED_ROWS.len() {
        let row = &SPEED_ROWS[i];
        if row.mhz == part.mhz && row.l2_kb == part.l2_kb {
            return Some(FUEL_BOARD_CLASS | row.nibble);
        }
        i += 1;
    }
    None
}

/// The word for the Fuel's only processor, [`CPUS`]`[0]`.
pub(crate) const FUEL_BOARD_ID: u64 = match board_word(&parts::R14000_500) {
    Some(word) => word,
    None => panic!("the Fuel's processor has no row in the PROM's speed table"),
};
/// Two 256 MiB banks, locational banks 0 and 1.
const FUEL_BANKS_MIB: [u32; 2] = [256, 256];

/// The crossbar widget of the Fuel's graphics slot: a Fuel IRIX inventory
/// lists its VPro under xtalk/13, and OpenBSD finds Odyssey at widget 13.
pub const WIDGET_VPRO: u8 = 13;
