//! The Onyx2 deskside: one or two IP27 node boards on the 4P1G5 midplane,
//! both hosts of the one Crossbow, with the BaseIO-G at widget 15 and the
//! graphics slot at widget 8.
//!
//! Node slot 1 drives raw slot identity 3 into its Hub and takes widget 10;
//! slot 2 drives 2 and takes widget 9. The module controller, the Hubs' cable
//! and the node windows are the Origin 2000 deskside's
//! ([`crate::ip27_origin2000`]).
//!
//! # What is approximated here
//!
//! - The graphics slot holds a test widget that answers its identification
//!   and nothing else, not an InfiniteReality pipe.
//! - Either Hub's processor reaches the BaseIO-G, and the Bridge takes DMA and
//!   interrupts to the Hub whose widget the guest programs into it, so the
//!   board the stock PROM makes the global master receives them.
//! - The MediaIO's two serial ports have no host backend, and the host's keys
//!   and pointer go to the board IOC3's controller only; which of the two
//!   keyboard and mouse connectors reaches which IOC3 is not known.
//! - The rest is as the Origin 2000 deskside's list in
//!   [`crate::ip27_origin2000`].
//!
//! Ported from hw/mips/sgi/sn-machine.c in jamesbraid/qemu-sgi-origami.

use crate::config::MachineConfig;
use crate::dev::bridge::Bridge;
use crate::dev::hub::{Hub, HubConfig, Space};
use crate::dev::hub_md::SlotSignals;
use crate::dev::ioc3::{ByteBus, Ioc3};
use crate::dev::rad1::Rad1;
use crate::dev::node_flash::FlashValues;
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::dev::xbow::{widget_id, Xbow};
use crate::devlog::LogModule;
use crate::ip27_machine::{cpu_module, pci_devices, BoardIo, BoardSpec, Ip27Board, Ip27Machine, Node, NodeRecord};
use crate::pci::DevFn;
use crate::ip27_origin2000::{Origin2000, BRIDGE_REVISION, MAX_NODES, XBOW_REVISION};
use crate::traits::{BusDevice, BusRead32, BusRead64, Resettable, BUS_ERR};
use crate::xtalk::{HostPort, XtalkHost};
use std::sync::Arc;

pub type Onyx2Machine = Ip27Machine<Onyx2Deskside>;

/// The MediaIO daughtercard's IOC3, a PCI device of the BaseIO-G's Bridge.
const MEDIAIO_DEVICE: u8 = 6;

/// The MediaIO's RAD1 audio chip, the next device of the same Bridge.
const RAD1_DEVICE: u8 = 7;

/// The graphics slot's widget, and the BaseIO-G's.
const WIDGET_PIPE: u8 = 8;
const WIDGET_BASEIO: u8 = 15;
/// A node board's Hub port: 10 in node slot 1, 9 in slot 2.
const fn widget_hub(node: usize) -> u8 {
    10 - node as u8
}
/// What node slot `node` drives onto the Hub's slot identity: 3 and 2.
const fn slot_bits(node: usize) -> u8 {
    3 - node as u8
}
const fn hub_host(node: usize) -> XtalkHost {
    XtalkHost { widget: widget_hub(node), int_packet: 0, int_packet_mask: 0, vectors: 128 }
}
/// The test widget's identification: not a real part.
const TEST_WIDGET_ID: u32 = widget_id(0, 0x7e57, 0x36);

/// A widget that answers only its identification word.
struct TestWidget;
impl BusDevice for TestWidget {
    fn read32(&self, a: u64) -> BusRead32 {
        if a == 4 {
            return BusRead32::ok(TEST_WIDGET_ID);
        }
        crate::dlog!(LogModule::Hub, "Onyx2: read of {a:#x} in the graphics slot, which holds a test widget");
        BusRead32::err()
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        crate::dlog!(LogModule::Hub, "Onyx2: write of {v:#x} to {a:#x} in the graphics slot, which holds a test widget");
        BUS_ERR
    }
    fn read64(&self, a: u64) -> BusRead64 {
        crate::dlog!(LogModule::Hub, "Onyx2: 64-bit read of {a:#x} in the graphics slot, which holds a test widget");
        BusRead64::err()
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        crate::dlog!(LogModule::Hub, "Onyx2: 64-bit write of {v:#x} to {a:#x} in the graphics slot, which holds a test widget");
        BUS_ERR
    }
}

pub struct Onyx2Deskside;

impl Onyx2Deskside {
    /// The BaseIO-G's records on its Bridge's one-wire line: the base board
    /// and the MediaIO daughtercard's own, as one published `hinv -vm` lists
    /// them. The daughtercard's is a separate part, so its ROM serial is
    /// distinct from the base board's and a one-wire search tells them apart.
    /// Without it IRIX turns Ethernet on for the MediaIO's IOC3 too, which has
    /// none. Barcodes and serials are synthetic.
    fn board_record() -> w1::Bus {
        let mut bus = w1::Bus::new();
        for (record, serial) in [
            (BoardRecord { barcode: "EMUB01", part: "030-0734-002", revision: "N", name: "BASEIO" }, 0x5145_4d55_4201),
            (BoardRecord { barcode: "EMUD01", part: "030-0880-003", revision: "G", name: "MIO" }, 0x5145_4d55_4401),
        ] {
            bus.add(Box::new(board_record_part(&record, serial)));
        }
        bus
    }
}

impl Onyx2Deskside {
    /// The MediaIO daughtercard, on the BaseIO-G's Bridge. Its IOC3 is at PCI
    /// device 6: both serial ports, a keyboard and mouse controller and the
    /// parallel port, but no Ethernet, clock or one-wire part. Its ports have
    /// no host backend: they are peripheral lines, where the board IOC3's
    /// first is the console. The host's keys and pointer go to the board
    /// IOC3's controller.
    fn fit_mediaio(node: &Node, cfg: &MachineConfig, bridge: &Bridge, io: &mut BoardIo) {
        let pci = bridge.bus();
        let mio = Arc::new(Ioc3::new(&crate::dev::ioc3::MIO, [0; 6], ByteBus::None));
        pci.attach(DevFn { device: MEDIAIO_DEVICE, function: 0 }, mio.clone());
        mio.connect_superio(pci);
        mio.set_timer_manager(node.timers.clone());
        mio.register_locks();
        io.devices.push(mio.clone());
        io.monitor.push(mio.ps2());
        // The RAD1 audio chip, at device 7. It raises no interrupt: the audio
        // driver registers no data-path interrupt.
        let rad = Arc::new(Rad1::new());
        rad.set_timer_manager(node.timers.clone());
        if !cfg.no_audio {
            rad.enable_output(cfg.audio.clone());
        }
        pci.attach(DevFn { device: RAD1_DEVICE, function: 0 }, rad.clone());
        rad.register_locks();
        io.devices.push(rad.clone());
        let board = std::mem::replace(&mut io.reset, Box::new(|| {}));
        io.reset = Box::new(move || {
            board();
            mio.power_on();
            rad.reset();
        });
    }
}

impl Ip27Board for Onyx2Deskside {
    const NAME: &'static str = "Onyx2";

    fn specs(cfg: &MachineConfig) -> Result<Vec<BoardSpec>, String> {
        let module = *cpu_module(cfg.cpu_part()?).ok_or("an Onyx2 has an R10000 or an R12000")?;
        let nodes = usize::from(cfg.origin2000.nodes);
        if !(1..=MAX_NODES).contains(&nodes) {
            return Err(format!("origin2000.nodes {nodes}: the deskside has one or two node boards"));
        }
        Ok((0..nodes)
            .map(|node| BoardSpec {
                hub: HubConfig {
                    cpus: 1,
                    cpu_base: 2 * node,
                    ni_revision: 5,
                    slot: SlotSignals { slot: Some(slot_bits(node)), slave_module: false },
                    ..Default::default()
                },
                module,
                banks_mb: std::array::from_fn(|i| cfg.banks[i]),
                flash: FlashValues {
                    cpu_mode_bits: module.mode_word,
                    cpu_clock_hz: module.clock_hz(),
                    machine_type: 0,
                    ..FlashValues::ORIGIN200_R10000_180_1M
                },
                flash_file: Origin2000::per_node(&cfg.origin2000.flash, node),
                node_record: Some(NodeRecord::IP27),
            })
            .collect())
    }

    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String> {
        let controller = Origin2000::controller(nodes, cfg)?;
        let mut widgets = (1u16 << WIDGET_PIPE) | (1 << WIDGET_BASEIO);
        for i in 0..nodes.len() {
            widgets |= 1 << widget_hub(i);
        }
        let xbow = Arc::new(Xbow::new(XBOW_REVISION, widgets));
        // The Crossbow's own interrupts go to the last node's Hub, which the
        // stock PROM makes the master. Either Hub's processor reaches the
        // BaseIO through the shared Crossbow, and the Bridge sends DMA and
        // interrupts to the Hub at the widget the guest programs, so the
        // Crossbow gives it both, each behind its port's link.
        let owner_index = nodes.len() - 1;
        let owner = nodes[owner_index];
        xbow.connect_host(hub_host(owner_index), owner.hub.interrupt_target());
        xbow.register_locks();

        let hosts = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| HostPort::new(hub_host(i), n.hub.inbound(), n.hub.interrupt_target()))
            .collect();
        let bridge = Bridge::new(xbow.upstream(hosts), &crate::ip30::PCI0);
        bridge.set_revision(BRIDGE_REVISION);
        bridge.set_microlan(Self::board_record());
        let links = xbow.clone();
        bridge.set_link_alive(Box::new(move |w| links.carries(w)));
        let weak = Arc::downgrade(&bridge);
        xbow.on_link_reset(Box::new(move |w| {
            if w == WIDGET_BASEIO {
                if let Some(b) = weak.upgrade() {
                    b.link_reset();
                }
            }
        }));

        let peers: Vec<(u8, Arc<dyn BusDevice>)> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (widget_hub(i), n.hub.inbound()))
            .collect();
        for node in nodes {
            let io = node.hub.io();
            io.connect_widget(0, xbow.clone());
            io.connect_widget(WIDGET_PIPE, xbow.port(WIDGET_PIPE, Arc::new(TestWidget)));
            io.connect_widget(WIDGET_BASEIO, xbow.port(WIDGET_BASEIO, bridge.clone()));
            for (i, n) in nodes.iter().enumerate() {
                io.connect_widget(widget_hub(i), xbow.port(widget_hub(i), n.hub.widget_face()));
            }
            io.connect_packets(xbow.packets(peers.clone()));
        }
        if let [a, b] = nodes {
            Hub::cable(&a.hub, &b.hub);
            for node in nodes {
                node.memory.route_nodes(node.hub.claim(Space::Memory));
            }
        }
        let pci_bridge = bridge.clone();
        let mut io = pci_devices(owner, cfg, bridge, &crate::dev::ioc3::BASEIO6G, controller, "onyx2", Self::NAME)?;
        Self::fit_mediaio(owner, cfg, &pci_bridge, &mut io);
        Ok(io)
    }
}

#[cfg(test)]
mod tests;
