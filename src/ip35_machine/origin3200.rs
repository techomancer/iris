//! The Origin 3200: one or two C-bricks, each a Bedrock node with a 400 MHz
//! R12000, and an I-brick on the first C-brick's XIO connector carrying the
//! base I/O. Two C-bricks are joined by a direct NUMAlink cable with no
//! router.
//!
//! Selected with `[machine] profile = "origin3200_ip35"`. The shared keys select
//! the PROM, NVRAM, network and memory (`banks`, per C-brick: 256 + 256 MiB
//! unless set), and `[origin3200]` selects the number of C-bricks, the serial
//! console and writable flash. Each C-brick has its own L1; the first carries
//! the console.
//!
//! # The I-brick
//!
//! [`IBRICK`] is the base I/O brick as data: an XBridge whose heads E and F are
//! widgets 14 and 15 of the first C-brick, its host port A (widget 10)
//! cabled to the C-brick, and on head F's bus (the IO7) a Fibre Channel
//! adapter at device 1, the IOC3 at device 4 and the USB controller at
//! device 5. Port B (widget 11) is declared and empty. The Fibre Channel
//! adapter is always fitted, with the disks of `[fibre_channel]`; its slot runs
//! 64-bit data phases.
//!
//! # What is approximated here
//!
//! - A C-brick holds two or four processors; the model runs one a node, so
//!   positions B, C and D are listed absent in the flash and a one-PIMM L1
//!   record is answered. Which PIMM count a one-processor C-brick would
//!   report is not established.
//! - Which low nibble of the board identification word a C-brick presents
//!   is a choice: 2 is the row for the 400 MHz, 8 MiB part.
//! - The I-brick's own L1 is a stub: the first C-brick's L1 answers its bay
//!   and letter and serves every other request as its own.
//! - The I-brick's IEEE 1394 controller is not modelled, so the PROM's first
//!   diagnostic prints `No 1394 Firewire found on BASEIO` and carries on.
//! - The IO7's IOC3 is the Fuel's with the placeholder subsystem id `0xffff`.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use super::{BoardIo, Ip35Board, NodeSpec, SerialLine};
use crate::config::MachineConfig;
use crate::cpu::parts::{self, Part};
use crate::dev::bedrock::Population;
use crate::dev::bedrock_flash::{FirmwareRecord, Header};
use crate::dev::heart::host_clock;
use crate::dev::ioc3::{ByteBus, Ioc3, Ioc3Board, IO7};
use crate::dev::isp2200::card::Card;
use crate::dev::l1::{AttachedBrick, BoardIdentity, ComputeNode, L1Config};
use crate::dev::ohci::hid::{HidFunction, UsbHidInput};
use crate::dev::ohci::Ohci;
use crate::dev::timekeeper::Timekeeper;
use crate::dev::xbridge::{XBridge, HEAD_E, HEAD_F, HOST_WIDGET};
use crate::ip35::inbound::{Inbound, BEDROCK_PORT};
use crate::ip35::node::Node;
use crate::pci::{BoardIrq, DevFn};
use crate::traits::{BusDevice, Device, Resettable};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The processors an Origin 3200 C-brick is built with: the 400 MHz R12000
/// with an 8 MiB secondary cache.
pub const CPUS: &[&Part] = &[&parts::R12000_400_8M];

/// The C-brick's 400 MHz / 8 MiB PROM record selects these boot-mode inputs:
/// a mismatch with the processor's Config sends the PROM to its error routine
/// before it initialises GP. The mode word is the stock IP35 PROM's row for
/// that fit (Config 0x5414a785: SB set, SS 4); the machine type is 0, the
/// class the PROM derives from [`BOARD_ID`].
const FIRMWARE: FirmwareRecord = FirmwareRecord {
    cpu_mode_word: 0x5414_a785,
    cpu_clock_hz: 400_000_000,
    machine_type: 0,
    header: Header::Refreshed,
};

/// Bits 15:12 of the board identification word are the product code (0 is a
/// C-brick to the PROM) and the low nibble the PROM's configuration row; 2 is
/// the 400 MHz, 8 MiB row (a choice: which nibble a C-brick presents is not
/// established).
const BOARD_ID: u64 = 2;

/// The L1's brick type character for a C-brick.
const L1_TYPE: u32 = 0x43;

/// The I-brick of the Origin 3200 in module 001i14, as the first C-brick's
/// L1 reports it.
const I_BRICK: AttachedBrick = AttachedBrick { bay: 14, type_code: 0x49, graphics_bay: None };

/// A C-brick's node board and processor module as its L1 names them. None of
/// it is observed: the names are the product's, the part and revision the
/// marked placeholders the L1 model uses.
const NODE_BOARD: BoardIdentity = BoardIdentity { name: "NODE", part: "NOTOBSERVED0", revision: "00" };
const PIMM: BoardIdentity = BoardIdentity { name: "PIMM", part: "NOTOBSERVED0", revision: "00" };

/// Where each C-brick sits: the rack bay its L1 reports (module 001c04 is bay
/// 4) and the synthetic serial of its L1.
const BAYS: [(u32, &str); 2] = [(4, "EMUN00"), (7, "EMUN01")];

/// What sits on the I-brick's head F bus at a device.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Fitting {
    /// The base I/O's IOC3 of this board row.
    Ioc3(&'static Ioc3Board),
    /// The Lucent USB controller.
    Usb,
    /// The QLogic ISP2200 Fibre Channel adapter.
    FibreChannel,
}

/// The I-brick as data: which crossbar ports face what, and what sits on the
/// PCI bus behind head F.
#[derive(Debug)]
pub(crate) struct IBrickRow {
    pub name: &'static str,
    /// The crossbar ports of the two PCI heads, E then F.
    pub heads: [u8; 2],
    /// The Crosstown2 port that faces the first C-brick.
    pub host_port: u8,
    /// The second Crosstown2 port, declared with nothing cabled to it.
    pub spare_port: u8,
    /// The devices of head F's bus (the IO7), by PCI device number. Head E's
    /// bus has none.
    pub head_f: &'static [(u8, Fitting)],
}

/// The Origin 3200's I-brick. Established from published `hinv -mv` of an
/// Origin 3200, an Onyx 3000 and an Origin 3400: the QLogic 2200 at PCI slot
/// 1, the IOC3 at 4 and the USB controller at 5 of the bus at widget 15,
/// with the 1394 controller at 6 not modelled. That head F is widget 15 and
/// head E widget 14 is inferred from the port letters.
pub(crate) const IBRICK: IBrickRow = IBrickRow {
    name: "I-brick",
    heads: [HEAD_E, HEAD_F],
    host_port: HOST_WIDGET,
    spare_port: 11,
    head_f: &[
        (1, Fitting::FibreChannel),
        (4, Fitting::Ioc3(&IO7)),
        (5, Fitting::Usb),
    ],
};

pub(crate) struct Origin3200;

/// `path` for the C-brick `index`: the first keeps the name, the others add
/// `.index`.
fn flash_for(path: &str, index: usize) -> PathBuf {
    if index == 0 {
        PathBuf::from(path)
    } else {
        PathBuf::from(format!("{path}.{index}"))
    }
}

impl Ip35Board for Origin3200 {
    const NAME: &'static str = "Origin 3200";

    fn specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
        let section = &cfg.origin3200;
        let banks = cfg.origin3200_banks()?;
        let revision = section.l1_revision.as_deref().map(crate::dev::l1::parse_revision).transpose()?;
        (0..usize::from(section.c_bricks))
            .map(|index| {
                let (bay, serial) = BAYS[index];
                let mut mac = cfg.network().mac;
                mac[5] = mac[5].wrapping_add(index as u8);
                let mut l1 = L1Config {
                    bay,
                    type_code: L1_TYPE,
                    mac,
                    brick_serial: Some(serial.into()),
                    node_board: Some(NODE_BOARD),
                    attached: (index == 0).then_some(I_BRICK),
                    compute_node: Some(ComputeNode { index: index as u8, pimm_mask: 1, pimm: PIMM }),
                    ..Default::default()
                };
                if index == 0 {
                    if let Some(revision) = revision {
                        l1.revision = Some(revision);
                    }
                    if let Some(serial) = &section.brick_serial {
                        l1.brick_serial = Some(serial.clone());
                    }
                }
                Ok(NodeSpec {
                    name: "C-brick",
                    part: cfg.cpu(),
                    firmware: FIRMWARE,
                    board_id: BOARD_ID,
                    banks_mib: banks.clone(),
                    processors: Population::SINGLE,
                    // One processor runs: B, C and D hold none.
                    absent: "BCD",
                    l1,
                    flash: section.flash.as_deref().map(|path| flash_for(path, index)),
                })
            })
            .collect()
    }

    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String> {
        let first = nodes[0];
        let (bedrock, l1) = (&first.bedrock, &first.l1);
        let timekeeper = Arc::new(Timekeeper::with_nvram_file(Path::new(&cfg.nvram))?);
        let xbridge = XBridge::new(BEDROCK_PORT, &|widget| {
            let port = Inbound::port(bedrock, widget);
            (port.clone() as Arc<dyn BusDevice>, port as Arc<dyn crate::irq::InterruptController>)
        });
        // The crossbar answers at widget 0, the host port and the heads.
        for widget in [0, IBRICK.host_port, IBRICK.heads[0], IBRICK.heads[1]] {
            bedrock.connect_widget(widget, xbridge.widget(widget).expect("the I-brick's ports are the XBridge's"));
        }
        let serial = std::array::from_fn::<_, 3, _>(|_| Arc::new(crate::dev::serial_host::CiSerialBackend::new()));
        l1.set_serial_backend(serial[0].clone());
        let mac = l1.station_mac();
        let pci = xbridge.bus(IBRICK.heads[1]);
        let slot = |device| DevFn { device, function: 0 };
        let (mut ioc3, mut usb, mut fibre) = (None, None, None);
        for &(device, fitting) in IBRICK.head_f {
            match fitting {
                Fitting::Ioc3(board) => {
                    let chip = Arc::new(Ioc3::new(board, mac, ByteBus::M48t35(timekeeper.clone())));
                    pci.attach(slot(device), chip.clone());
                    ioc3 = Some(chip);
                }
                Fitting::Usb => {
                    let controller = Ohci::new();
                    controller.set_clock(host_clock());
                    pci.attach(slot(device), controller.clone());
                    usb = Some(controller);
                }
                // The on-board adapter, with the disks of `[fibre_channel]`.
                Fitting::FibreChannel => {
                    let card = Card::open(&cfg.fibre_channel.clone().unwrap_or_default(), cfg.ci)?;
                    card.fit(pci, device);
                    // The slot runs 64-bit data phases. That is a property of the
                    // I-brick's board wiring that no public document gives; it is
                    // inferred from the PROM's behaviour, whose control block and
                    // rings are right only if the bridge reverses each doubleword
                    // of the adapter's DMA as it does on such a slot.
                    xbridge.set_dma_64bit(IBRICK.heads[1], device, true);
                    eprintln!("iris: Fibre Channel on the I-brick, {} loop disk(s)", cfg.fibre_channel.as_ref().map_or(0, |f| f.disks.len()));
                    fibre = Some(card.fc);
                }
            }
        }
        let ioc3 = ioc3.ok_or("the I-brick's row has no IOC3")?;
        let usb = usb.ok_or("the I-brick's row has no USB controller")?;
        // The I-brick's keyboard and mouse are USB, not the IOC3's PS/2 ports.
        // Keyboard on root port 1 and mouse on port 2 is inferred.
        let (keyboard, mouse) = (HidFunction::keyboard(), HidFunction::mouse());
        usb.attach(0, keyboard.clone())?;
        usb.attach(1, mouse.clone())?;
        ioc3.ps2().redirect_input(UsbHidInput::new(keyboard, mouse));
        let eth = ioc3.eth().expect("the IO7's IOC3 fits Ethernet").clone();
        ioc3.set_serial_backend(0, serial[1].clone());
        ioc3.set_serial_backend(1, serial[2].clone());
        ioc3.connect_superio_interrupt(pci.board_irq(BoardIrq(0)));
        let (reset_ioc3, reset_usb) = (ioc3.clone(), usb.clone());
        let heads = IBRICK.heads;
        xbridge.set_reset_output(Arc::new(move |port| {
            if port == heads[1] {
                reset_ioc3.power_on();
                reset_usb.power_on();
            }
        }));
        let (l1_port, a, b) = {
            let [a, b] = cfg.serial_ports();
            (cfg.origin3200.l1_port.unwrap_or(8882), a, b)
        };
        let l1_for_serial = l1.clone();
        let (ioc3_a, ioc3_b) = (ioc3.clone(), ioc3.clone());
        let serial_lines = vec![
            SerialLine {
                label: "L1",
                port: l1_port,
                setting: "origin3200.l1_port",
                ci: serial[0].clone(),
                log: cfg.origin3200.l1_log.clone(),
                connect: Box::new(move |backend| l1_for_serial.set_serial_backend(backend)),
            },
            SerialLine {
                label: "IOC3 A",
                port: a,
                setting: "serial_port_a",
                ci: serial[1].clone(),
                log: None,
                connect: Box::new(move |backend| ioc3_a.set_serial_backend(0, backend)),
            },
            SerialLine {
                label: "IOC3 B",
                port: b,
                setting: "serial_port_b",
                ci: serial[2].clone(),
                log: None,
                connect: Box::new(move |backend| ioc3_b.set_serial_backend(1, backend)),
            },
        ];
        let (start_ioc3, start_usb, start_fibre) = (ioc3.clone(), usb.clone(), fibre.clone());
        let stop_fibre = fibre.clone();
        let (stop_ioc3, stop_usb) = (ioc3.clone(), usb.clone());
        let (reset_xbridge, reset_ioc3, reset_usb) = (xbridge.clone(), ioc3.clone(), usb.clone());
        let poll_usb = usb.clone();
        let lock_xbridge = xbridge.clone();
        let lock_ioc3 = ioc3.clone();
        Ok(BoardIo {
            start: Box::new(move || {
                start_ioc3.start();
                start_usb.start();
                if let Some(fibre) = &start_fibre {
                    fibre.start();
                }
            }),
            stop: Box::new(move || {
                if let Some(fibre) = &stop_fibre {
                    fibre.stop();
                }
                stop_usb.stop();
                stop_ioc3.stop();
            }),
            reset: Box::new(move || {
                reset_xbridge.power_on();
                reset_ioc3.power_on();
                reset_usb.power_on();
            }),
            poll: Arc::new(move || poll_usb.sync_clock()),
            register_locks: Box::new(move || {
                lock_xbridge.register_locks();
                lock_ioc3.register_locks();
            }),
            monitor: vec![ioc3.ps2() as Arc<dyn Device>],
            ps2: ioc3.ps2(),
            timekeeper,
            console: cfg.origin3200.console.index(),
            serial: serial_lines,
            eth,
            fibre,
        })
    }
}
