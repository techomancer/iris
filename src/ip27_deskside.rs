//! The Origin 2000 and Onyx2 desksides: one or two IP27 node boards on the
//! module midplane, the BaseIO card on a Crossbow, and with two nodes the Hubs
//! joined by a direct CrayLink cable, the Null Router.
//!
//! [`Deskside`] is the one board type. A product is a [`DesksideLayout`], data
//! read once when the machine is built: which widget each node board's Hub
//! takes on its Crossbow, the slot identity the midplane drives into it, where
//! the BaseIO and the graphics slot are, and whether the node boards share one
//! Crossbow. Nothing tests the product on an access.
//!
//! - The Origin 2000 gives each node board its own Crossbow. The first carries
//!   the BaseIO at widget 8 and its node at widget 9, and the second carries its
//!   node at widget 10 and nothing else. The slot identities are raw 7 in node
//!   slot 1 and 6 in slot 2.
//! - The Onyx2 (4P1G5 midplane) makes both node boards hosts of the one
//!   Crossbow, with the BaseIO-G at widget 15 and the graphics slot at widget
//!   8. Node slot 1 drives raw slot identity 3 and takes widget 10; slot 2
//!   drives 2 and takes widget 9.
//!
//! # What is approximated here
//!
//! - The module has one controller, reached through an I2C adapter per node.
//!   Each adapter completes a byte inside the register access that starts it,
//!   so the nodes cannot contend for the bus bit by bit
//!   ([`crate::dev::o2000_module`]).
//! - The BaseIO is a Bridge with the Origin 200's PCI population: two
//!   ISP1040s and an IOC3 with the DP83840A transceiver model. The card's
//!   boot flash is fitted only when `io_prom` names an image; without one its
//!   flash window fails, and the BASEIO PROM loads from the copy in the node
//!   PROM's image. The Onyx2's BaseIO-G also carries the MediaIO daughtercard,
//!   fitted as its layout names it: a second IOC3 at PCI device 6 and the RAD1
//!   audio chip at device 7, with the daughtercard's own board record. Its two
//!   serial ports have no host backend, and the host's keys and pointer go to
//!   the board IOC3's controller only; which of the two keyboard and mouse
//!   connectors reaches which IOC3 is not known.
//! - The module's midplane one-wire part is not fitted, and the Hub's line
//!   reaches only the node's own net.
//! - Origin 2000: node 2 has no I/O of its own, since a deskside's second
//!   Crossbow carries the other six XIO slots, none of them fitted here. The
//!   Bridge reaches only the Hub that owns its Crossbow, node 1's.
//! - Onyx2: the graphics slot holds an InfiniteReality2 or 2E pipe
//!   (`[graphics] board = "ir2"`, `"ir2:2rm"`, `"ir2e"` or `"ir2e:2rm"`), or
//!   with `none` a test widget that answers its identification and nothing
//!   else. The pipe's DMA and interrupt packets
//!   reach the Hub at the widget the guest programs into it, as the Bridge's
//!   do. Either Hub's processor reaches the BaseIO-G, and the Bridge takes DMA
//!   and interrupts to the Hub whose widget the guest programs into it, so the
//!   board the stock PROM makes the global master receives them.
//! - Cached accesses to node numbers 1 and up route through the Hub, and a
//!   number no Hub holds fails. A node that holds a nonzero number sends node
//!   zero's window beyond its cached alias to the Hub that holds zero.
//!
//! Ported from hw/mips/sgi/sn-machine.c in jamesbraid/qemu-sgi-origami.

use crate::config::{MachineConfig, MachineProfile};
use crate::dev::bridge::Bridge;
use crate::dev::elsc::ModuleOutput;
use crate::dev::elsc_nvram::ElscNvram;
use crate::devlog::LogModule;
use crate::dev::hub::{Hub, HubConfig, Space};
use crate::dev::hub::SlotSignals;
use crate::dev::ioc3::{ByteBus, Ioc3, Ioc3Board};
use crate::dev::kona::{Kona, KonaConfig};
use crate::dev::rad1::Rad1;
use crate::dev::node_flash::FlashValues;
use crate::dev::o2000_module::{ModuleConfig, Origin2000Module};
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::dev::xbow::{widget_id, Xbow};
use crate::pci::DevFn;
use crate::ip27_machine::{cpu_module, fit_io_prom, pci_devices, product, PciPopulation, section, table, BoardIo, BoardSpec, Ip27Board, Ip27Machine, ModuleController, Node, NodeRecord, Unmodelled};
use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR};
use crate::xtalk::Upstream;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

pub type DesksideMachine = Ip27Machine<Deskside>;

/// Node boards the deskside midplane has.
pub const MAX_NODES: usize = 2;
/// The Crossbow revision: 1.1, the earliest the kernel accepts.
const XBOW_REVISION: u32 = 2;
/// The BaseIO's Bridge revision.
const BRIDGE_REVISION: u32 = 4;
/// The module number the controller's store starts with.
const MODULE_NUMBER: u8 = 1;
/// The module's serial and its identity part's ROM serial: synthetic, so an
/// inventory from the model never reads as a real unit.
const MODULE_SERIAL: &str = "EMU0000001";
const MODULE_IDENTITY_SERIAL: u64 = 0x5145_4d55_4d00;

/// How a product's node boards reach their Crossbows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crossbows {
    /// Each node board hosts a Crossbow of its own, and the first carries the
    /// BaseIO.
    PerNode,
    /// Every node board is a host of the one Crossbow, which carries the
    /// BaseIO and the graphics slot.
    Shared,
}

/// A board record on a one-wire part of the BaseIO's Bridge line. The barcode
/// and ROM serial are synthetic.
#[derive(Clone, Copy, Debug)]
pub struct BaseioRecord {
    pub name: &'static str,
    pub barcode: &'static str,
    pub part: &'static str,
    pub revision: &'static str,
    pub rom_serial: u64,
}

/// The MediaIO daughtercard on a BaseIO-G: its IOC3 and the RAD1 audio chip
/// at PCI devices of the BaseIO's Bridge, and its own board record, a separate
/// part with its own ROM serial so a one-wire search tells it from the base
/// board. Without the record IRIX turns Ethernet on for the daughtercard's
/// IOC3 too, which has none.
#[derive(Clone, Copy, Debug)]
pub struct MediaIo {
    pub record: BaseioRecord,
    pub ioc3_device: u8,
    pub rad1_device: u8,
}

/// What differs between the desksides, as data read once when the machine is
/// built.
#[derive(Debug)]
pub struct DesksideLayout {
    pub crossbows: Crossbows,
    /// The Crossbow widget of the Hub in each node slot.
    pub hub_widgets: [u8; MAX_NODES],
    /// What each node slot drives onto its Hub's slot identity.
    pub slot_bits: [u8; MAX_NODES],
    pub baseio_widget: u8,
    pub baseio: BaseioRecord,
    /// The BaseIO's IOC3: its subsystem identification, the Bridge input its
    /// SuperIO interrupt takes and the ports it fits.
    pub ioc3: &'static Ioc3Board,
    /// The daughtercard the BaseIO carries, if any.
    pub mediaio: Option<MediaIo>,
    /// The graphics slot's widget, where the product has one on the shared
    /// Crossbow. The InfiniteReality pipe is fitted there.
    pub graphics_widget: Option<u8>,
}

/// The Origin 2000 deskside.
pub static ORIGIN_2000: DesksideLayout = DesksideLayout {
    crossbows: Crossbows::PerNode,
    hub_widgets: [9, 10],
    slot_bits: [7, 6],
    baseio_widget: 8,
    baseio: BaseioRecord { name: "BASEIO", barcode: "EMUB00", part: "030-1124-002", revision: "J", rom_serial: 0x5145_4d55_4200 },
    ioc3: &crate::dev::ioc3::BASEIO,
    mediaio: None,
    graphics_widget: None,
};

/// The Onyx2 deskside.
pub static ONYX2: DesksideLayout = DesksideLayout {
    crossbows: Crossbows::Shared,
    hub_widgets: [10, 9],
    slot_bits: [3, 2],
    baseio_widget: 15,
    baseio: BaseioRecord { name: "BASEIO", barcode: "EMUB01", part: "030-0734-002", revision: "N", rom_serial: 0x5145_4d55_4201 },
    ioc3: &crate::dev::ioc3::BASEIO6G,
    mediaio: Some(MediaIo {
        record: BaseioRecord { name: "MIO", barcode: "EMUD01", part: "030-0880-003", revision: "G", rom_serial: 0x5145_4d55_4401 },
        ioc3_device: 6,
        rad1_device: 7,
    }),
    graphics_widget: Some(8),
};

/// The test widget's identification: not a real part.
const TEST_WIDGET_ID: u32 = widget_id(0, 0x7e57, 0x36);

/// Where the pipe's interrupt packets and DMA go: the host at the widget the
/// guest has programmed into the pipe, behind that widget's Crossbow link.
/// Nothing is delivered while the widget names no host with a live link.
struct PipeHosts {
    pipe: OnceLock<Weak<Kona>>,
    up: Upstream,
}

impl PipeHosts {
    fn host(&self) -> Option<&crate::xtalk::HostPort> {
        let widget = self.pipe.get()?.upgrade()?.dma_widget() as u8;
        let host = self.up.get(widget).filter(|h| h.link_up());
        if host.is_none() {
            crate::dlog!(LogModule::Kona, "Onyx2: the pipe names widget {widget}, which is no Hub with a live link");
        }
        host
    }
}

impl InterruptController for PipeHosts {
    fn set_line(&self, line: u32, level: bool) {
        if let Some(h) = self.host() {
            h.irq.set_line(line, level);
        }
    }
}

macro_rules! pipe_dma {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            self.host().map_or($out::err(), |h| h.mem.$r(a))
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.host().map_or(BUS_ERR, |h| h.mem.$w(a, v))
        }
    };
}
impl BusDevice for PipeHosts {
    pipe_dma!(read8, write8, u8, BusRead8);
    pipe_dma!(read16, write16, u16, BusRead16);
    pipe_dma!(read32, write32, u32, BusRead32);
    pipe_dma!(read64, write64, u64, BusRead64);
}

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
    fn take_events(&self) -> Vec<ModuleOutput> {
        self.module.take_events()
    }
    fn next_due(&self) -> Option<Duration> {
        self.module.next_due()
    }
    fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        self.module.set_wake(wake);
    }
}

/// The deskside board: the Origin 2000 and the Onyx2 are layouts of it.
pub struct Deskside;

impl Deskside {
    /// The layout of the product the profile names.
    pub fn layout(profile: MachineProfile) -> Result<&'static DesksideLayout, String> {
        match profile {
            MachineProfile::Origin2000Ip27 => Ok(&ORIGIN_2000),
            MachineProfile::Onyx2Ip27 => Ok(&ONYX2),
            _ => Err(format!("{} is not a deskside", profile.name())),
        }
    }

    /// The BaseIO's board records on its Bridge's one-wire line: the base
    /// board's and, where the layout has one, the MediaIO's.
    fn board_record(layout: &DesksideLayout) -> w1::Bus {
        let mut bus = w1::Bus::new();
        for b in [Some(&layout.baseio), layout.mediaio.as_ref().map(|m| &m.record)].into_iter().flatten() {
            let record = BoardRecord { barcode: b.barcode, part: b.part, revision: b.revision, name: b.name };
            bus.add(Box::new(board_record_part(&record, b.rom_serial)));
        }
        bus
    }

    /// The MediaIO daughtercard, on the BaseIO-G's Bridge. Its IOC3 has both
    /// serial ports, a keyboard and mouse controller and the parallel port,
    /// but no Ethernet, clock or one-wire part. The RAD1 audio chip raises no
    /// interrupt: the audio driver registers no data-path interrupt.
    fn fit_mediaio(node: &Node, cfg: &MachineConfig, bridge: &Bridge, media: &MediaIo, io: &mut BoardIo) {
        let pci = bridge.bus();
        let mio = Arc::new(Ioc3::new(&crate::dev::ioc3::MIO, [0; 6], ByteBus::None));
        pci.attach(DevFn { device: media.ioc3_device, function: 0 }, mio.clone());
        mio.connect_superio(pci);
        mio.set_timer_manager(node.timers.clone());
        mio.register_locks();
        io.devices.push(mio.clone());
        io.monitor.push(mio.ps2());
        let rad = Arc::new(Rad1::new());
        rad.set_timer_manager(node.timers.clone());
        if !cfg.no_audio {
            rad.enable_output(cfg.audio.clone());
        }
        pci.attach(DevFn { device: media.rad1_device, function: 0 }, rad.clone());
        rad.register_locks();
        io.devices.push(rad.clone());
        let board = std::mem::replace(&mut io.reset, Box::new(|| {}));
        io.reset = Box::new(move || {
            board();
            mio.power_on();
            rad.reset();
        });
    }

    /// A file for node `node` beside the one the configuration names for the
    /// first.
    pub(crate) fn per_node(path: &Option<String>, node: usize) -> Option<PathBuf> {
        path.as_ref().map(|p| if node == 0 { PathBuf::from(p) } else { PathBuf::from(format!("{p}.{node}")) })
    }

    /// The module controller: one for the module, in front of every node's
    /// I2C controller.
    fn controller(nodes: &[&Node], cfg: &MachineConfig) -> Result<Arc<ModuleManagement>, String> {
        let store = ElscNvram::origin2000(MODULE_NUMBER, section(cfg).controller_store.as_deref().map(Path::new))
            .map(Arc::new)
            .map_err(|e| format!("{}.controller_store: {e}", table(cfg)))?;
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

    /// The BaseIO's Bridge, as `upstream` reaches its host: revision, board
    /// record and, when the machine names an I/O PROM, the boot flash.
    fn baseio_bridge(cfg: &MachineConfig, layout: &DesksideLayout, upstream: crate::xtalk::Upstream) -> Result<Arc<Bridge>, String> {
        let bridge = Bridge::new(upstream, &crate::ip30::PCI0);
        bridge.set_revision(BRIDGE_REVISION);
        bridge.set_microlan(Self::board_record(layout));
        fit_io_prom(&bridge, cfg)?;
        Ok(bridge)
    }

    /// Each node board hosts a Crossbow of its own; the first carries the
    /// BaseIO. Returns the Bridge and the node whose Hub it reaches.
    fn wire_per_node<'a>(nodes: &[&'a Node], cfg: &MachineConfig, layout: &DesksideLayout) -> Result<(Arc<Bridge>, &'a Node), String> {
        let mut bridge = None;
        for (i, node) in nodes.iter().enumerate() {
            let host = node.hub.host_port(layout.hub_widgets[i]);
            let widget = host.host.widget;
            let mut widgets = 1u16 << widget;
            if i == 0 {
                widgets |= 1 << layout.baseio_widget;
            }
            let xbow = Arc::new(Xbow::new(XBOW_REVISION, widgets));
            xbow.connect_host(host.host, node.hub.interrupt_target());
            node.hub.connect_widget(0, xbow.clone());
            node.hub.connect_widget(widget, xbow.port(widget, node.hub.widget_face()));
            xbow.register_locks();
            if i == 0 {
                let b = Self::baseio_bridge(cfg, layout, xbow.upstream(vec![host]))?;
                xbow.serve_bridge(layout.baseio_widget, &b);
                node.hub.connect_widget(layout.baseio_widget, xbow.port(layout.baseio_widget, b.clone()));
                bridge = Some(b);
            }
        }
        bridge.map(|b| (b, nodes[0])).ok_or_else(|| "a deskside has at least one node".to_string())
    }

    /// Every node board is a host of the one Crossbow, which carries the
    /// BaseIO and the graphics slot. Returns the Bridge and the node that owns
    /// the Crossbow.
    /// An InfiniteReality2 pipe in the graphics slot too, when the configuration
    /// names one: `status` is its status bar's source.
    fn wire_shared<'a>(
        nodes: &[&'a Node],
        cfg: &MachineConfig,
        layout: &DesksideLayout,
        status: &Arc<crate::disp::StatusSource>,
    ) -> Result<(Arc<Bridge>, &'a Node, Option<Arc<Kona>>), String> {
        let mut widgets = 1u16 << layout.baseio_widget;
        widgets |= layout.graphics_widget.map_or(0, |w| 1 << w);
        for i in 0..nodes.len() {
            widgets |= 1 << layout.hub_widgets[i];
        }
        let xbow = Arc::new(Xbow::new(XBOW_REVISION, widgets));
        // The Crossbow's own interrupts go to the last node's Hub, which the
        // stock PROM makes the master. Either Hub's processor reaches the
        // BaseIO through the shared Crossbow, and the Bridge sends DMA and
        // interrupts to the Hub at the widget the guest programs, so the
        // Crossbow gives it both, each behind its port's link.
        let owner_index = nodes.len() - 1;
        let owner = nodes[owner_index];
        let owner_port = owner.hub.host_port(layout.hub_widgets[owner_index]);
        xbow.connect_host(owner_port.host, owner.hub.interrupt_target());
        xbow.register_locks();

        let hosts = || nodes.iter().enumerate().map(|(i, n)| n.hub.host_port(layout.hub_widgets[i])).collect::<Vec<_>>();
        let bridge = Self::baseio_bridge(cfg, layout, xbow.upstream(hosts()))?;
        xbow.serve_bridge(layout.baseio_widget, &bridge);

        let pipe = match (cfg.graphics_board().kona(), layout.graphics_widget) {
            (Some((product, rm_count)), Some(_)) => {
                let pipe_hosts = Arc::new(PipeHosts { pipe: OnceLock::new(), up: xbow.upstream(hosts()) });
                let kona_cfg = KonaConfig::new(product, rm_count)?
                    .with_host_widget(owner_port.host.widget)?
                    .with_pool(cfg.graphics.threads, cfg.graphics.tile);
                let flash = cfg.graphics.flash.as_deref().map(|p| crate::dev::kona::flash_file(Path::new(p))).transpose()?;
                let pipe = Kona::new(kona_cfg, pipe_hosts.clone(), pipe_hosts.clone(), status.clone(), flash)?;
                let _ = pipe_hosts.pipe.set(Arc::downgrade(&pipe));
                Some(pipe)
            }
            _ => None,
        };

        let peers: Vec<(u8, Arc<dyn BusDevice>)> =
            nodes.iter().enumerate().map(|(i, n)| (layout.hub_widgets[i], n.hub.inbound())).collect();
        for node in nodes {
            let hub = &node.hub;
            hub.connect_widget(0, xbow.clone());
            if let Some(w) = layout.graphics_widget {
                let slot: Arc<dyn BusDevice> = match &pipe {
                    Some(p) => p.clone(),
                    None => Arc::new(Unmodelled::identified("graphics slot (a test widget)", TEST_WIDGET_ID)),
                };
                hub.connect_widget(w, xbow.port(w, slot));
            }
            hub.connect_widget(layout.baseio_widget, xbow.port(layout.baseio_widget, bridge.clone()));
            for (i, n) in nodes.iter().enumerate() {
                let w = layout.hub_widgets[i];
                hub.connect_widget(w, xbow.port(w, n.hub.widget_face()));
            }
            hub.connect_packets(xbow.packets(peers.clone()));
        }
        Ok((bridge, owner, pipe))
    }
}

impl Ip27Board for Deskside {
    fn specs(cfg: &MachineConfig) -> Result<Vec<BoardSpec>, String> {
        let product = product(cfg);
        let layout = Self::layout(cfg.machine.profile)?;
        let module = *cpu_module(cfg.cpu_part()?).ok_or_else(|| format!("an {product} has an R10000 or an R12000"))?;
        let nodes = usize::from(section(cfg).nodes);
        if !(1..=MAX_NODES).contains(&nodes) {
            return Err(format!("{}.nodes {nodes}: the deskside has one or two node boards", table(cfg)));
        }
        Ok((0..nodes)
            .map(|node| BoardSpec {
                // Hub 2.3 on the deskside node boards.
                hub: HubConfig {
                    cpus: 1,
                    // Global processor numbers run two to a node slot.
                    cpu_base: 2 * node,
                    ni_revision: 5,
                    slot: SlotSignals { slot: Some(layout.slot_bits[node]), slave_module: false },
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
                    ..FlashValues::STOCK_R10000_180
                },
                flash_file: Self::per_node(&section(cfg).flash, node),
                node_record: Some(NodeRecord::IP27),
            })
            .collect())
    }

    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String> {
        let layout = Self::layout(cfg.machine.profile)?;
        let controller = Self::controller(nodes, cfg)?;
        // The status bar's activity bits, shared with the Bridge's devices.
        let heartbeat = Arc::new(AtomicU64::new(0));
        let status = crate::disp::StatusSource::new(heartbeat.clone(), Arc::new(AtomicU64::new(0)));
        let (bridge, host, pipe) = match layout.crossbows {
            Crossbows::PerNode => {
                let (bridge, host) = Self::wire_per_node(nodes, cfg, layout)?;
                (bridge, host, None)
            }
            Crossbows::Shared => Self::wire_shared(nodes, cfg, layout, &status)?,
        };
        if let [a, b] = nodes {
            Hub::cable(&a.hub, &b.hub);
            for node in nodes {
                node.memory.route_nodes(node.hub.claim(Space::Memory));
            }
        }
        let pci_bridge = bridge.clone();
        let population = PciPopulation { bridge, ioc3: layout.ioc3, controller, heartbeat };
        let mut io = pci_devices(host, cfg, population)?;
        if let Some(media) = &layout.mediaio {
            Self::fit_mediaio(host, cfg, &pci_bridge, media, &mut io);
        }
        if let Some(p) = pipe {
            io.display = Some(p.display());
            io.status = Some(status);
            io.devices.push(p.clone());
            io.monitor.push(p.clone());
            let reset = std::mem::replace(&mut io.reset, Box::new(|| {}));
            let board = p.clone();
            io.reset = Box::new(move || {
                reset();
                board.power_on();
            });
            #[cfg(test)]
            {
                io.pipe = Some(p);
            }
        }
        Ok(io)
    }
}

#[cfg(test)]
mod tests;
