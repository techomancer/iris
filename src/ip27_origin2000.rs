//! The Origin 2000 deskside: one or two IP27 node boards on the module
//! midplane, each the host of its own Crossbow, the first Crossbow carrying
//! the BaseIO card, and with two nodes the Hubs joined by a direct CrayLink
//! cable, the Null Router.
//!
//! Each node board's slot identity is what the midplane drives into the Hub:
//! raw 7 in node slot 1 and 6 in slot 2. Node 1's Crossbow takes its node at
//! widget 9 and the BaseIO at widget 8; node 2's takes its node at widget 10
//! and nothing else.
//!
//! # What is approximated here
//!
//! - The module has one controller, reached through an I2C adapter per node.
//!   Each adapter completes a byte inside the register access that starts it,
//!   so the nodes cannot contend for the bus bit by bit
//!   ([`crate::dev::o2000_module`]).
//! - The BaseIO is a Bridge with the Origin 200's PCI population: two
//!   ISP1040s and an IOC3 with the DP83840A transceiver model. The card's
//!   boot flash is not fitted, so its flash window fails, and the BASEIO PROM
//!   loads from the copy in the node PROM's image.
//! - The module's midplane one-wire part is not fitted, and the Hub's line
//!   reaches only the node's own net.
//! - Node 2 has no I/O of its own: a deskside's second Crossbow carries the
//!   other six XIO slots, none of them fitted here.
//! - The Bridge reaches only the Hub that owns its Crossbow, node 1's. A
//!   Bridge that two Hubs can address is not modelled.
//! - Cached accesses to node numbers 1 and up route through the Hub, and a
//!   number no Hub holds fails. A node that holds a nonzero number sends node
//!   zero's window beyond its cached alias to the Hub that holds zero.
//!
//! Ported from hw/mips/sgi/sn-machine.c in jamesbraid/qemu-sgi-origami.

use crate::config::MachineConfig;
use crate::dev::bridge::Bridge;
use crate::dev::elsc::ElscEvent;
use crate::dev::elsc_nvram::ElscNvram;
use crate::dev::hub::{Hub, HubConfig, Space};
use crate::dev::hub_md::SlotSignals;
use crate::dev::node_flash::FlashValues;
use crate::dev::o2000_module::{ModuleConfig, ModuleEvent, Origin2000Module};
use crate::traits::Resettable;
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::dev::xbow::Xbow;
use crate::ip27_machine::{cpu_module, pci_devices, BoardIo, BoardSpec, Ip27Board, Ip27Machine, ModuleController, Node, NodeRecord};
use crate::xtalk::{Upstream, XtalkHost};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type Origin2000Machine = Ip27Machine<Origin2000>;

/// Node boards the deskside midplane has.
pub const MAX_NODES: usize = 2;
/// The Crossbow's widget for the BaseIO card, in XIO slot 1.
const WIDGET_BASEIO: u8 = 8;
/// The Crossbow's widget for a node board's Hub: 9 in node slot 1, 10 in 2.
const fn widget_hub(node: usize) -> u8 {
    9 + node as u8
}
/// What node slot `node` (0 or 1) drives onto the Hub's slot identity: 7 for
/// the first slot, 6 for the second.
const fn slot_bits(node: usize) -> u8 {
    7 - node as u8
}
/// The Crossbow revision: 1.1, the earliest the kernel accepts.
pub(crate) const XBOW_REVISION: u32 = 2;
/// The BaseIO's Bridge revision.
pub(crate) const BRIDGE_REVISION: u32 = 4;

/// The Hub as its Crossbow and the widgets below address it.
const fn hub_host(node: usize) -> XtalkHost {
    XtalkHost { widget: widget_hub(node), int_packet: 0, int_packet_mask: 0, vectors: 128 }
}

/// The module number the controller's store starts with.
const MODULE_NUMBER: u8 = 1;
/// The module's serial and its identity part's ROM serial: synthetic, so an
/// inventory from the model never reads as a real unit.
const MODULE_SERIAL: &str = "EMU0000001";
const MODULE_IDENTITY_SERIAL: u64 = 0x5145_4d55_4d00;

/// The module management as the machine drives it: the adapters to reset
/// with the module, and the outputs to drain.
pub(crate) struct ModuleManagement {
    module: Arc<Origin2000Module>,
    adapters: usize,
}

impl ModuleController for ModuleManagement {
    /// A module reset leaves the controller alone and returns the nodes'
    /// adapters to power-on.
    fn reset(&self) {
        for i in 0..self.adapters {
            self.module.adapter(i).power_on();
        }
    }
    fn take_events(&self) -> Vec<ElscEvent> {
        self.module
            .take_events()
            .into_iter()
            .filter_map(|e| match e {
                ModuleEvent::Power(on) => Some(ElscEvent::Power(on)),
                ModuleEvent::Reset(true) => Some(ElscEvent::Reset),
                ModuleEvent::Reset(false) => None,
                ModuleEvent::Nmi(level) => Some(ElscEvent::Nmi(level)),
            })
            .collect()
    }
}

pub struct Origin2000;

impl Origin2000 {
    /// The BaseIO's board record on its Bridge's one-wire line. The barcode
    /// and ROM serial are synthetic.
    fn board_record() -> w1::Bus {
        let mut bus = w1::Bus::new();
        let record = BoardRecord { barcode: "EMUB00", part: "030-1124-002", revision: "J", name: "BASEIO" };
        bus.add(Box::new(board_record_part(&record, 0x5145_4d55_4200)));
        bus
    }

    /// A file for node `node` beside the one the configuration names for the
    /// first.
    pub(crate) fn per_node(path: &Option<String>, node: usize) -> Option<PathBuf> {
        path.as_ref().map(|p| if node == 0 { PathBuf::from(p) } else { PathBuf::from(format!("{p}.{node}")) })
    }

    /// The module controller: one for the module, in front of every node's
    /// I2C controller.
    pub(crate) fn controller(nodes: &[&Node], cfg: &MachineConfig) -> Result<Arc<ModuleManagement>, String> {
        let store = ElscNvram::origin2000(MODULE_NUMBER, cfg.origin2000.controller_store.as_deref().map(Path::new))
            .map(Arc::new)
            .map_err(|e| format!("origin2000.controller_store: {e}"))?;
        let module = Origin2000Module::new(ModuleConfig { module_number: MODULE_NUMBER, ..Default::default() }, store, nodes[0].clock.clone());
        for (i, node) in nodes.iter().enumerate() {
            node.hub.attach_i2c(module.adapter(i));
        }
        module
            .attach_parts(MODULE_SERIAL, MODULE_IDENTITY_SERIAL, None)
            .map_err(|e| format!("module identity: {e}"))?;
        module.power_on();
        Ok(Arc::new(ModuleManagement { module, adapters: nodes.len() }))
    }
}

impl Ip27Board for Origin2000 {
    const NAME: &'static str = "Origin 2000";

    fn specs(cfg: &MachineConfig) -> Result<Vec<BoardSpec>, String> {
        let module = *cpu_module(cfg.cpu_part()?).ok_or("an Origin 2000 has an R10000 or an R12000")?;
        let nodes = usize::from(cfg.origin2000.nodes);
        if !(1..=MAX_NODES).contains(&nodes) {
            return Err(format!("origin2000.nodes {nodes}: the deskside has one or two node boards"));
        }
        Ok((0..nodes)
            .map(|node| BoardSpec {
                // Hub 2.3 on the Origin 2000 module.
                hub: HubConfig {
                    cpus: 1,
                    // Global processor numbers run two to a node slot.
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
                    // Zero in the stock image: the PROM chooses the module's
                    // Crossbow tables rather than the Origin 200's.
                    machine_type: 0,
                    ..FlashValues::ORIGIN200_R10000_180_1M
                },
                flash_file: Self::per_node(&cfg.origin2000.flash, node),
                node_record: Some(NodeRecord::IP27),
            })
            .collect())
    }

    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String> {
        let controller = Self::controller(nodes, cfg)?;
        let mut io = None;
        for (i, node) in nodes.iter().enumerate() {
            let host = hub_host(i);
            let mut widgets = 1u16 << host.widget;
            if i == 0 {
                widgets |= 1 << WIDGET_BASEIO;
            }
            let xbow = Arc::new(Xbow::new(XBOW_REVISION, widgets));
            xbow.connect_host(host, node.hub.interrupt_target());
            node.hub.io().connect_widget(0, xbow.clone());
            node.hub.io().connect_widget(host.widget, xbow.port(host.widget, node.hub.widget_face()));
            xbow.register_locks();
            if i == 0 {
                let bridge = Bridge::new(Upstream::single(host, node.hub.inbound(), node.hub.interrupt_target()), &crate::ip30::PCI0);
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
                node.hub.io().connect_widget(WIDGET_BASEIO, xbow.port(WIDGET_BASEIO, bridge.clone()));
                io = Some(pci_devices(node, cfg, bridge, &crate::dev::ioc3::BASEIO, controller.clone(), "origin2000", Self::NAME)?);
            }
        }
        if let [a, b] = nodes {
            Hub::cable(&a.hub, &b.hub);
            for node in nodes {
                node.memory.route_nodes(node.hub.claim(Space::Memory));
            }
        }
        io.ok_or_else(|| "an Origin 2000 has at least one node".to_string())
    }
}

#[cfg(test)]
mod tests;
