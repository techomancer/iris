//! The Origin 2000 and Onyx2 desksides: one or two IP27 node boards on the midplane, the BaseIO card
//! on a Crossbow, and with two nodes the Hubs joined by a direct CrayLink connection (the Null Router).
//!
//! [`node_specs`] and [`build_io`] serve both products; `origin2000` and `onyx2` build what differs: each node Hub's widget
//! on its Crossbow, its slot identity, the BaseIO and graphics slots, and whether the node boards
//! share one Crossbow. The Origin 2000 gives each its own (the first carries the
//! BaseIO at widget 8 and its node at 9; the second its node at 10; raw slot identities 7 and 6). The
//! Onyx2 (4P1G5 midplane) shares one, the BaseIO-G at widget 15 and graphics at 8 (slot 1: identity 3,
//! widget 10; slot 2: identity 2, widget 9).
//!
//! # What is approximated here
//!
//! - The module has one controller, reached through an I2C adapter per node; each adapter completes
//!   a byte inside the register access that starts it, so the nodes cannot contend bit by bit.
//! - The BaseIO is a Bridge with the Origin 200's PCI population (two ISP1040s, an IOC3 with the
//!   DP83840A model); its boot flash holds the built-in I/O PROM or the image `io_prom` names. The
//!   Onyx2's BaseIO-G adds the MediaIO daughtercard (a second IOC3 at device 6, RAD1 audio at 7,
//!   its own board record) with no serial backend; which keyboard and mouse connector reaches which
//!   IOC3 is not known, so the host's keys go to the board IOC3 only.
//! - The midplane one-wire part is not attached; the Hub's line reaches only the node's own net.
//! - Origin 2000: node 2 has no I/O (its Crossbow carries six XIO slots, none attached); the Bridge
//!   reaches only node 1's Hub.
//! - Onyx2: the graphics slot holds an InfiniteReality2 or 2E pipe (`[graphics] board` `ir2`,
//!   `ir2:2rm`, `ir2e`, `ir2e:2rm`) or, with `none`, a widget that answers only its identification.
//!   Its DMA and interrupts reach the Hub at the widget the guest programs, as the Bridge's do; either
//!   Hub's processor reaches the BaseIO-G, and the stock PROM's global master receives them.
//! - Cached accesses to node numbers 1 and up route through the Hub, and a number no Hub holds
//!   fails. A node holding a nonzero number sends node zero's window beyond its cached alias to the
//!   Hub that holds zero.
//!
//! Ported from hw/mips/sgi/sn-machine.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::parts;
use crate::clock::Clock;
use crate::dev::node_flash::NodeFlash;
use crate::hptimer::TimerManager;
use crate::sn::NodeBoard;
use crate::dev::bridge::{Bridge, PIN_WIRING};
use crate::dev::elsc::ModuleOutput;
use crate::dev::elsc_nvram::ElscNvram;
use crate::devlog::LogModule;
use crate::dev::hub::{Hub, HubConfig};
use crate::dev::hub::SlotSignals;
use crate::dev::ioc3::{ByteBus, Ioc3};
use crate::dev::kona::{Kona, KonaConfig};
use crate::dev::rad1::Rad1;
use crate::dev::node_flash::FlashValues;
use crate::dev::o2000_module::{ModuleConfig, Origin2000Module};
use crate::dev::widths::access_widths;
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::dev::xbow::{widget_id, Xbow};
use crate::pci::DevFn;
use crate::ip27::{attach_controller, load_io_prom, pci_devices, ModuleController, NodeSpec};
use crate::sn::BoardIo;
use crate::traits::MachineEvents;
use crate::irq::InterruptController;
use crate::traits::{BusDevice, Device, Resettable, BUS_ERR};
use crate::xtalk::Upstream;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};

/// The Crossbow revision: 1.1, the earliest the kernel accepts.
/// The Origin 2000 and Onyx2 node board.
pub const IP27_BOARD: BoardRecord<'static> = BoardRecord { barcode: "", part: "030-0733-003", revision: "G", name: "IP27" };

const XBOW_REVISION: u32 = 2;
/// The BaseIO's Bridge revision.
const BRIDGE_REVISION: u32 = 4;
/// The module number the controller's store starts with.
const MODULE_NUMBER: u8 = 1;
/// The module's serial and its identity part's ROM serial: synthetic, so an
/// inventory from the model never reads as a real unit.
const MODULE_SERIAL: &str = "EMU0000001";
const MODULE_IDENTITY_SERIAL: u64 = 0x5145_4d55_4d00;

/// The BaseIO and BaseIO-G board records on their Bridge's one-wire line, and the MediaIO
/// daughtercard's. The barcodes and ROM serials are synthetic.
const BASEIO_RECORD: BoardRecord<'static> = BoardRecord { barcode: "EMUB00", part: "030-1124-002", revision: "J", name: "BASEIO" };
const BASEIO_SERIAL: u64 = 0x5145_4d55_4200;
const BASEIO_G_RECORD: BoardRecord<'static> = BoardRecord { barcode: "EMUB01", part: "030-0734-002", revision: "N", name: "BASEIO" };
const BASEIO_G_SERIAL: u64 = 0x5145_4d55_4201;
const MIO_RECORD: BoardRecord<'static> = BoardRecord { barcode: "EMUD01", part: "030-0880-003", revision: "G", name: "MIO" };
const MIO_SERIAL: u64 = 0x5145_4d55_4401;
/// The MediaIO's PCI devices on the BaseIO-G's Bridge.
const MEDIAIO_IOC3_DEVICE: u8 = 6;
const MEDIAIO_RAD1_DEVICE: u8 = 7;

/// The test widget's identification: not a real part.
const TEST_WIDGET_ID: u32 = widget_id(0, 0x7e57, 0x36);

/// Where the pipe's interrupt packets and DMA go: the host at the widget the guest has programmed,
/// behind that widget's Crossbow link. Nothing is delivered while it names no host with a live
/// link.
struct PipeHosts {
    /// The pipe's DMA widget, read without taking the pipe's register lock.
    widget: OnceLock<Arc<AtomicU8>>,
    up: Upstream,
}

impl PipeHosts {
    fn host(&self) -> Option<&crate::xtalk::HostPort> {
        let widget = self.widget.get()?.load(Ordering::Relaxed);
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

impl PipeHosts {
    #[inline(always)]
    fn route(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
        Some((&*self.host()?.mem, a))
    }
}
impl BusDevice for PipeHosts {
    access_widths!(route);

    // One lookup of the host per block rather than one per word.
    fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 {
        self.host().map_or(BUS_ERR, |h| h.mem.read_block(a, buf))
    }
    fn write_block(&self, a: u64, buf: &[u64]) -> u32 {
        self.host().map_or(BUS_ERR, |h| h.mem.write_block(a, buf))
    }
}

/// A space or widget the model does not carry: every access fails, visibly in the log. A widget
/// slot can answer its identification word (the 4-byte read at offset 4) so the PROM finds it
/// attached.
struct Unmodelled {
    what: &'static str,
    id: Option<u32>,
}

impl Unmodelled {
    /// `what` answers `id` as its widget identification and nothing else.
    fn identified(what: &'static str, id: u32) -> Self {
        Self { what, id: Some(id) }
    }
}

impl BusDevice for Unmodelled {
    fn read32(&self, a: u64) -> crate::traits::BusRead32 {
        match self.id {
            Some(id) if a == 4 => crate::traits::BusRead32::ok(id),
            _ => {
                crate::dlog!(LogModule::Hub, "IP27: read of unmodelled {} {a:#x}", self.what);
                crate::traits::BusRead32::err()
            }
        }
    }
    fn read64(&self, a: u64) -> crate::traits::BusRead64 {
        crate::dlog!(LogModule::Hub, "IP27: read of unmodelled {} {a:#x}", self.what);
        crate::traits::BusRead64::err()
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        crate::dlog!(LogModule::Hub, "IP27: write of {v:#x} to unmodelled {} {a:#x}", self.what);
        BUS_ERR
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        crate::dlog!(LogModule::Hub, "IP27: write of {v:#x} to unmodelled {} {a:#x}", self.what);
        BUS_ERR
    }
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
    fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>) {
        self.module.set_wake(wake);
    }
    fn device(&self) -> Arc<dyn Device> {
        self.module.clone()
    }
}

/// The MediaIO daughtercard on the BaseIO-G's Bridge: an IOC3 with both serial ports, PS/2 and
/// parallel but no Ethernet, clock or one-wire part. The RAD1 raises no interrupt (its driver
/// registers none).
fn attach_mediaio(timers: &Arc<TimerManager>, cfg: &MachineConfig, bridge: &Bridge, io: &mut BoardIo) -> Result<(), String> {
    let pci = bridge.bus();
    let mio = Arc::new(Ioc3::new(timers, &crate::dev::ioc3::MIO, [0; 6], ByteBus::None));
    pci.attach(DevFn { device: MEDIAIO_IOC3_DEVICE, function: 0 }, mio.clone());
    mio.connect_superio(pci);
    let name = cfg.machine.profile.product_name().unwrap_or_default();
    let serial = |board, n| crate::dev::serial_dev::connect(cfg, name, board, n, None, None);
    mio.set_serial_backend(0, serial("MediaIO IOC3 A", 3)?);
    mio.set_serial_backend(1, serial("MediaIO IOC3 B", 4)?);
    mio.register_locks();
    io.devices.push(mio.clone());
    io.monitor.push(mio.ps2());
    let rad = Arc::new(Rad1::mediaio(timers));
    if !cfg.no_audio {
        rad.enable_output(cfg.audio.clone());
    }
    pci.attach(DevFn { device: MEDIAIO_RAD1_DEVICE, function: 0 }, rad.clone());
    rad.register_locks();
    io.devices.push(rad.clone());
    io.monitor.push(rad.clone());
    let board = std::mem::replace(&mut io.reset, Box::new(|| {}));
    io.reset = Box::new(move || {
        board();
        mio.power_on();
        rad.reset();
    });
    Ok(())
}

/// The module controller: one for the module, in front of every node's
/// I2C controller.
fn controller(nodes: &[NodeBoard<Hub, NodeFlash>], timers: &Arc<TimerManager>, cfg: &MachineConfig) -> Result<Arc<ModuleManagement>, String> {
    let store = ElscNvram::origin2000(MODULE_NUMBER, cfg.ip27_section().expect("an IP27 profile").controller_store.as_deref().map(Path::new))
        .map(Arc::new)
        .map_err(|e| format!("{}.controller_store: {e}", cfg.machine.profile.table_name().unwrap_or_default()))?;
    let module = Origin2000Module::new(ModuleConfig { module_number: MODULE_NUMBER, ..Default::default() }, store, timers);
    for (i, node) in nodes.iter().enumerate() {
        node.chip.attach_i2c(module.adapter(i));
    }
    module
        .attach_parts(MODULE_SERIAL, MODULE_IDENTITY_SERIAL, None)
        .map_err(|e| format!("module identity: {e}"))?;
    module.power_on();
    Ok(Arc::new(ModuleManagement { module, adapters: nodes.len() }))
}

/// The one-wire line carrying each of `records` with its ROM serial.
fn board_records(records: &[(BoardRecord<'static>, u64)]) -> w1::Bus {
    let mut line = w1::Bus::new();
    for (record, serial) in records {
        line.add(Box::new(board_record_part(record, *serial)));
    }
    line
}

/// The BaseIO's Bridge, as `upstream` reaches its host: revision, the board `records` on its
/// one-wire line and, when the machine names an I/O PROM, the boot flash.
fn build_baseio(cfg: &MachineConfig, upstream: crate::xtalk::Upstream, records: &[(BoardRecord<'static>, u64)]) -> Result<Arc<Bridge>, String> {
    let bridge = Bridge::new(upstream, &PIN_WIRING);
    bridge.set_revision(BRIDGE_REVISION);
    bridge.set_microlan(board_records(records));
    load_io_prom(&bridge, cfg)?;
    Ok(bridge)
}

/// A Crossbow hosted by `hosts` (node boards and their widgets) with the other `widgets` (a bit
/// each) beside them. Its interrupts go to the last host, the stock PROM's master.
fn build_crossbow(hosts: &[(&NodeBoard<Hub, NodeFlash>, u8)], widgets: u16) -> Arc<Xbow> {
    let xbow = Arc::new(Xbow::new(XBOW_REVISION, hosts.iter().fold(widgets, |m, (_, w)| m | 1 << w)));
    let (owner, widget) = hosts[hosts.len() - 1];
    xbow.connect_host(owner.chip.host_port(widget).host, owner.chip.interrupt_target());
    xbow.register_locks();
    xbow
}

pub(crate) fn node_specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
    let product = cfg.machine.profile.product_name().unwrap_or_default();
    let slot_bits = if cfg.machine.profile == MachineProfile::Onyx2Ip27 { [3, 2] } else { [7, 6] };
    let module = parts::cpu_module(cfg.cpu_part()?).ok_or_else(|| format!("an {product} has an R10000 or an R12000"))?;
    let nodes = usize::from(cfg.ip27_section().expect("an IP27 profile").nodes);
    Ok((0..nodes)
        .map(|node| NodeSpec {
            // Hub 2.3 on the deskside node boards.
            hub: HubConfig {
                cpus: cfg.machine.cpus(),
                // Global processor numbers run two to a node slot.
                first_cpu: 2 * node,
                ni_revision: 5,
                nodes,
                slot: SlotSignals { slot: Some(slot_bits[node]), slave_module: false },
                banks_mb: std::array::from_fn(|i| cfg.banks[i]),
                ..Default::default()
            },
            module,
            firmware: FlashValues {
                cpu_mode_bits: module.mode_word,
                cpu_clock_hz: module.clock_hz(),
                // Zero in the stock image: the PROM chooses the module's
                // Crossbow tables rather than the Origin 200's.
                machine_type: 0,
                ..FlashValues::STOCK_R10000_180
            },
            flash: crate::sn::per_node(&cfg.ip27_section().expect("an IP27 profile").flash, node),
            node_record: Some(IP27_BOARD),
        })
        .collect())
}

pub(crate) fn build_io(
    cfg: &MachineConfig,
    nodes: &[NodeBoard<Hub, NodeFlash>],
    clock: &Clock,
    timers: &Arc<TimerManager>,
    events: &MachineEvents,
) -> Result<BoardIo, String> {
    let controller = controller(nodes, timers, cfg)?;
    // The status bar's activity bits, shared with the Bridge's devices.
    let heartbeat = Arc::new(AtomicU64::new(0));
    let mut io = if cfg.machine.profile == MachineProfile::Onyx2Ip27 {
        onyx2(cfg, nodes, timers, controller.clone(), heartbeat.clone())?
    } else {
        origin2000(cfg, nodes, timers, controller.clone(), heartbeat.clone())?
    };
    // A machine with no graphics has no bar, but the monitor's `speed` and
    // `ticks` read the same status.
    if io.status.is_none() {
        let status = crate::status::MachineStatus::without_ticks(heartbeat);
        status.set_scsi_wide(true);
        io.status = Some(status);
    }
    attach_controller(&mut io, controller, events);
    Ok(io)
}

/// The Origin 2000: each node board hosts a Crossbow of its own, and the first carries the BaseIO
/// at widget 8.
fn origin2000(
    cfg: &MachineConfig,
    nodes: &[NodeBoard<Hub, NodeFlash>],
    timers: &Arc<TimerManager>,
    controller: Arc<ModuleManagement>,
    heartbeat: Arc<AtomicU64>,
) -> Result<BoardIo, String> {
    const ORIGIN2000_BASEIO_WIDGET: u8 = 8;
    let mut bridge = None;
    for (node, hub_widget) in nodes.iter().zip([9, 10]) {
        let first = bridge.is_none();
        let xbow = build_crossbow(&[(node, hub_widget)], if first { 1 << ORIGIN2000_BASEIO_WIDGET } else { 0 });
        node.chip.io.connect_widget(0, xbow.clone());
        node.chip.io.connect_widget(hub_widget, xbow.attach_widget(hub_widget, node.chip.widget_port()));
        if first {
            let b = build_baseio(cfg, xbow.upstream(vec![node.chip.host_port(hub_widget)]), &[(BASEIO_RECORD, BASEIO_SERIAL)])?;
            xbow.serve_bridge(ORIGIN2000_BASEIO_WIDGET, &b);
            node.chip.io.connect_widget(ORIGIN2000_BASEIO_WIDGET, xbow.attach_widget(ORIGIN2000_BASEIO_WIDGET, b.clone()));
            bridge = Some(b);
        }
    }
    let bridge = bridge.ok_or("a deskside has at least one node")?;
    pci_devices(timers, cfg, bridge, &crate::dev::ioc3::BASEIO, controller, heartbeat)
}

/// The Onyx2 (4P1G5 midplane): every node board is a host of the one Crossbow, which carries the
/// BaseIO-G at widget 15, the MediaIO behind it and the graphics slot at widget 8: an
/// InfiniteReality2 pipe when the configuration names one, else a test widget.
fn onyx2(
    cfg: &MachineConfig,
    nodes: &[NodeBoard<Hub, NodeFlash>],
    timers: &Arc<TimerManager>,
    controller: Arc<ModuleManagement>,
    heartbeat: Arc<AtomicU64>,
) -> Result<BoardIo, String> {
    const ONYX2_BASEIO_WIDGET: u8 = 15;
    const GRAPHICS_WIDGET: u8 = 8;
    let hubs: Vec<_> = nodes.iter().zip([10, 9]).collect();
    // Either Hub's processor reaches the BaseIO, and the Bridge's DMA and interrupts go to the Hub
    // at the guest's widget.
    let xbow = build_crossbow(&hubs.iter().map(|&(n, w)| (n, w)).collect::<Vec<_>>(), 1 << ONYX2_BASEIO_WIDGET | 1 << GRAPHICS_WIDGET);
    let owner_widget = hubs[hubs.len() - 1].1;
    let upstream = || xbow.upstream(hubs.iter().map(|(n, w)| n.chip.host_port(*w)).collect());
    let bridge = build_baseio(cfg, upstream(), &[(BASEIO_G_RECORD, BASEIO_G_SERIAL), (MIO_RECORD, MIO_SERIAL)])?;
    xbow.serve_bridge(ONYX2_BASEIO_WIDGET, &bridge);

    let status = crate::status::MachineStatus::without_ticks(heartbeat.clone());
    let pipe = match cfg.graphics_board().kona() {
        Some((product, rm_count)) => {
            let pipe_hosts = Arc::new(PipeHosts { widget: OnceLock::new(), up: upstream() });
            let kona_cfg = KonaConfig::new(product, rm_count)?
                .with_host_widget(owner_widget)?
                .with_widget(GRAPHICS_WIDGET)?
                .with_pool(cfg.graphics.threads, cfg.graphics.tile);
            let flash = cfg.graphics.flash.as_deref().map(|p| crate::dev::kona::flash_file(Path::new(p))).transpose()?;
            let pipe = Kona::new(kona_cfg, pipe_hosts.clone(), pipe_hosts.clone(), status.clone(), flash)?;
            let _ = pipe_hosts.widget.set(pipe.dma_widget_handle());
            // The driver's adaptor reset link-resets the pipe's widget before it pulses the board's
            // reset; the widget loses its Crosstalk interface registers.
            let weak = Arc::downgrade(&pipe);
            xbow.on_link_reset(Box::new(move |w| {
                if w == GRAPHICS_WIDGET {
                    if let Some(pipe) = weak.upgrade() {
                        pipe.link_reset();
                    }
                }
            }));
            Some(pipe)
        }
        None => None,
    };

    let peers: Vec<(u8, Arc<dyn BusDevice>)> = hubs.iter().map(|(n, w)| (*w, n.chip.inbound())).collect();
    for node in nodes {
        let hub = &node.chip;
        hub.io.connect_widget(0, xbow.clone());
        let slot: Arc<dyn BusDevice> = match &pipe {
            Some(p) => p.clone(),
            None => Arc::new(Unmodelled::identified("graphics slot (a test widget)", TEST_WIDGET_ID)),
        };
        hub.io.connect_widget(GRAPHICS_WIDGET, xbow.attach_widget(GRAPHICS_WIDGET, slot));
        hub.io.connect_widget(ONYX2_BASEIO_WIDGET, xbow.attach_widget(ONYX2_BASEIO_WIDGET, bridge.clone()));
        for &(n, w) in &hubs {
            hub.io.connect_widget(w, xbow.attach_widget(w, n.chip.widget_port()));
        }
        hub.io.connect_packets(xbow.packets(peers.clone()));
    }
    let pci_bridge = bridge.clone();
    let mut io = pci_devices(timers, cfg, bridge, &crate::dev::ioc3::BASEIO6G, controller, heartbeat)?;
    attach_mediaio(timers, cfg, &pci_bridge, &mut io)?;
    status.set_scsi_wide(true);
    io.status = Some(status);
    if let Some(p) = pipe {
        io.display = Some(p.display());
        io.devices.push(p.clone());
        io.monitor.push(p.clone());
        let reset = std::mem::replace(&mut io.reset, Box::new(|| {}));
        let board = p.clone();
        io.reset = Box::new(move || {
            reset();
            board.power_on();
        });
    }
    Ok(io)
}

#[cfg(test)]
mod tests;
