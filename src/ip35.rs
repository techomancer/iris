//! The IP35 node shared by the Fuel, the Tezro and the Origin 3200 and Onyx 3000 C-bricks: the
//! Bedrock chip with its memory and boot flash, the processor bus over its uncached spaces
//! ([`crate::sn::UncachedBus`]) and the reset PROM's window.
//!
//! # The machines
//!
//! [`Machine<Bedrock, Flash>`] is shared by every IP35 product: C-bricks, each a Bedrock node with its
//! processors, and the base I/O a product connects to them. A product is the facts about each node
//! ([`NodeSpec`]) and how its I/O is built on the first, as plain functions in its file: the
//! Origin 3200 and Onyx 3000 ([`crate::ip35_machine`]), the Fuel ([`crate::ip34_machine`]) and the
//! Tezro ([`crate::ip53_machine`]).
//!
//! Two C-bricks are joined by NUMAlink ([`Bedrock::connect_peer`]) and number their processors
//! across the machine. Each node's Physical shows every node's memory at its number, and one
//! [`World`] holds every processor still while a store moves memory.
//!
//! - A node's empty processor positions are listed disabled in a fresh flash, as the PROM would
//!   after three failed starts of each ([`NodeSpec::absent`]).
//! - A reset or power request from either node resets the machine; a node's reset alone is not modelled.
//! - Processors on several nodes share memory with no coherence protocol between the nodes.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::dev::bedrock::{Bedrock, RamExtent};
use crate::clock::Clock;
use crate::dev::ioc3::{ByteBus, Ioc3, Ioc3Board};
use crate::dev::odyssey::Odyssey;
use crate::dev::ohci::hid::{HidFunction, UsbHidInput};
use crate::dev::ohci::Ohci;
use crate::dev::timekeeper::Timekeeper;
use crate::dev::xbridge::{HEAD_E, HEAD_F, HOST_WIDGET};
use crate::pci::{BoardIrq, DevFn, PciBus};
use crate::config::MachineConfig;
use crate::cpu::build::build_cpu;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu};
use crate::cpu::smp::World;
use crate::dev::bedrock_flash::Flash;
use crate::dev::l1::L1;
use crate::dev::prom_image;
use crate::hptimer::TimerManager;
use crate::sn::{BoardIo, Machine, NodeBoard, MAX_NODES};
use crate::traits::Device;
use crate::workstation::Workstation;
use node::{connect_cpus, NodeSpec};
use std::path::PathBuf;
use crate::traits::MachineEvents;

use crate::dev::odyssey::PacketSender;
use crate::dev::xbridge::XBridge;
use crate::traits::BusDevice;
use crate::xtalk::XtalkHost;
use std::sync::Arc;

pub(crate) use crate::window_map::SparseWindowMap;

pub(crate) use crate::sn::RESET_WINDOW;
/// The reset PROM's compatibility window: the processor's reset vector and
/// the PROM's own code, fetched uncached.
pub(crate) use crate::dev::bedrock::RESET_WINDOW_SIZE;

/// Bedrock's crossbar port: widget 10, taking interrupt packets at any
/// address (the address picks the processor interface).
pub(crate) const BEDROCK_PORT: XtalkHost =
    XtalkHost { widget: crate::dev::xbridge::HOST_WIDGET, int_packet: 0, int_packet_mask: 0, vectors: 256 };

/// An XBridge whose DMA and interrupt packets enter `bedrock` as each widget's inbound traffic,
/// answering at widget 0, the host port and its two heads.
pub(crate) fn xbridge_on(bedrock: &Bedrock) -> Arc<XBridge> {
    let xbridge = XBridge::new(BEDROCK_PORT, &|widget| {
        let port = bedrock.inbound(widget);
        (port.clone() as Arc<dyn BusDevice>, port as Arc<dyn crate::irq::InterruptController>)
    });
    for widget in [0, HOST_WIDGET, HEAD_E, HEAD_F] {
        bedrock.io.connect_widget(widget, xbridge.widget(widget).expect("the XBridge's ports"));
    }
    xbridge
}

/// The IOC3 of `board` at device 4 of head `pci`, with its SuperIO interrupt.
pub(crate) fn attach_ioc3(timers: &Arc<TimerManager>, pci: &PciBus, board: &'static Ioc3Board, l1: &Arc<L1>, timekeeper: &Arc<Timekeeper>) -> Arc<Ioc3> {
    let ioc3 = Arc::new(Ioc3::new(timers, board, l1.station_mac(), ByteBus::M48t35(timekeeper.clone())));
    pci.attach(DevFn { device: 4, function: 0 }, ioc3.clone());
    ioc3.connect_superio_interrupt(pci.board_irq(BoardIrq(0)));
    ioc3
}

/// The OHCI controller at device 5 of head `pci`. With `input`, a USB keyboard and mouse sit on its
/// first two root ports and take the host's input instead of `ioc3`'s PS/2 pair, or a guest that
/// opened both would see every key twice.
pub(crate) fn attach_usb(pci: &PciBus, ioc3: &Ioc3, timers: &Arc<TimerManager>, input: bool) -> Result<Arc<Ohci>, String> {
    let usb = Ohci::new(timers);
    if input {
        let (keyboard, mouse) = (HidFunction::keyboard(), HidFunction::mouse());
        usb.attach(0, keyboard.clone())?;
        usb.attach(1, mouse.clone())?;
        ioc3.ps2().redirect_input(UsbHidInput::new(keyboard, mouse));
    }
    pci.attach(DevFn { device: 5, function: 0 }, usb.clone());
    Ok(usb)
}

/// A VPro at crossbar `widget` of `bedrock`'s node: its DMA enters Bedrock as that widget's inbound
/// traffic and its interrupt packets leave by `packets`. `attach` puts it behind the board's crossbar and returns what the
/// node's I/O space reaches it by.
pub(crate) fn attach_vpro(
    cfg: &MachineConfig,
    bedrock: &Bedrock,
    widget: u8,
    packets: PacketSender,
    status: &Arc<crate::status::MachineStatus>,
    attach: impl FnOnce(Arc<Odyssey>) -> Result<Arc<dyn BusDevice>, String>,
) -> Result<Arc<Odyssey>, String> {
    let vpro = Odyssey::new_widget_relative(
        widget,
        cfg.graphics_board().as_str(),
        packets,
        bedrock.inbound(widget),
        status.clone(),
        (cfg.graphics.threads, cfg.graphics.tile),
        (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()),
    )?;
    bedrock.io.connect_widget(widget, attach(vpro.clone())?);
    Ok(vpro)
}

/// Where a Bedrock node has memory: each 1 GiB slot `extent` says can hold RAM, and the reset
/// PROM. A slot 0 over 508 MiB reaches the PROM's range, so its RAM is listed on either side.
pub fn memory_map(extent: &RamExtent) -> crate::memory_map::MemoryMap {
    use crate::dev::bedrock::{BANK_SPAN, SLOTS};
    use crate::memory_map::{MemKind, MemRange, MemoryMap};
    const NAMES: [&str; SLOTS] =
        ["ram slot 0", "ram slot 1", "ram slot 2", "ram slot 3", "ram slot 4", "ram slot 5", "ram slot 6", "ram slot 7"];
    const PROM: (u64, u64) = (RESET_WINDOW, RESET_WINDOW + RESET_WINDOW_SIZE);
    let mut ranges = vec![MemRange::new("prom", MemKind::Rom, PROM.0, PROM.1)];
    for (slot, &name) in NAMES.iter().enumerate() {
        let (base, end) = (slot as u64 * BANK_SPAN, slot as u64 * BANK_SPAN + extent.prefix(slot));
        if base == end {
            continue;
        }
        if base < PROM.1 && end > PROM.0 {
            ranges.push(MemRange::new(name, MemKind::Ram, base, PROM.0));
            if end > PROM.1 {
                ranges.push(MemRange::new("ram slot 0 above the prom", MemKind::Ram, PROM.1, end));
            }
        } else {
            ranges.push(MemRange::new(name, MemKind::Ram, base, end));
        }
    }
    MemoryMap::new(ranges)
}

/// A machine of `cfg`'s product: `node_specs` gives each node's facts and `build_io` builds the
/// board's I/O around the nodes and their L1s.
#[allow(clippy::type_complexity)]
pub(crate) fn machine(
    cfg: &MachineConfig,
    node_specs: fn(&MachineConfig) -> Result<Vec<NodeSpec>, String>,
    build_io: fn(&MachineConfig, &[NodeBoard<Bedrock, Flash>], &[Arc<L1>], &Arc<TimerManager>, &Clock) -> Result<BoardIo, String>,
) -> Result<Machine<Bedrock, Flash>, String> {
    cfg.validate()?;
    let name = cfg.machine.profile.product_name().unwrap_or_default();
    let prom = prom_image::resolve(cfg.machine.profile, &cfg.prom, 8 * 1024 * 1024)
        .map_err(|e| format!("{name} {e}"))?
        .bytes;
    let timers = Arc::new(TimerManager::new());
    let clock = timers.clock().clone();
    let specs = node_specs(cfg)?;
    assert!((1..=MAX_NODES).contains(&specs.len()), "a machine has one or two nodes");
    // Any node's reset or power request reaches the machine's main thread: a
    // reset of one node alone is not modelled.
    let (events, receiver) = MachineEvents::channel(16);
    // The machine numbers its processors once, across its nodes in order.
    let (mut nodes, mut l1s, mut first_cpu) = (Vec::new(), Vec::new(), 0);
    for spec in &specs {
        let l1 = Arc::new(L1::new(&timers, spec.l1.clone())?);
        nodes.push(NodeBoard::<Bedrock, Flash>::build(cfg.machine.profile.table_name().unwrap_or_default(), spec, &prom, &timers, first_cpu, specs.len(), &events, &l1)?);
        l1s.push(l1);
        first_cpu += spec.processors.count();
    }
    let world = Arc::new(World::new());
    events.set_cpus(world.clone());
    if let [a, b] = &nodes[..] {
        a.chip.connect_peer(&b.chip);
        b.chip.connect_peer(&a.chip);
    }
    for n in &nodes {
        n.chip.decode.set_world(world.clone());
    }
    let io = build_io(cfg, &nodes, &l1s, &timers, &clock)?;
    let count = nodes.len();
    // Every processor reaches every node's memory at its number.
    let maps: Vec<_> = nodes.iter().map(|n| crate::ip35::memory_map(&n.chip.md.ram_extent())).collect();
    let memory = Arc::new(crate::sn::machine_memory(&maps, crate::dev::bedrock::NODE_SHIFT));
    let smp = specs.iter().map(|s| s.processors.count()).sum::<usize>() > 1;
    let mut node_cpus = Vec::new();
    for (index, (node, spec)) in nodes.iter().zip(&specs).enumerate() {
        let count_hz = cfg.clock.count_hz(spec.module.part).max(1);
        let several = spec.processors.count() > 1;
        let cpus: Vec<_> = (0..spec.processors.count())
            .map(|n| {
                // Every processor of a node of several joins the PROM's holds.
                let held = io.at_batch.clone().filter(|_| several);
                let hook: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                    if let Some(held) = &held {
                        held();
                    }
                });
                let fasttick = io.fasttick.clone().filter(|_| index == 0 && n == 0);
                crate::with_cpu!(spec.module.part, true, false, C => {
                    let cpu_id = node.first_cpu + n;
                    let bus = node.bus.clone();
                    let mut exec = build_cpu::<C>(bus, node.chip.decode.physical().clone(), memory.clone(), spec.module, &timers, count_hz, cpu_id, smp);
                    if let Some(fasttick) = fasttick {
                        exec.core.fasttick_count = fasttick;
                    }
                    exec.batch_hook = Some(Box::new(move |_| hook()));
                    Arc::new(MipsCpu::new(*exec)) as Arc<dyn CpuDevice>
                })
            })
            .collect();
        node_cpus.push(cpus);
    }
    let cpus: Vec<Arc<dyn CpuDevice>> = node_cpus.iter().flatten().cloned().collect();
    // One body of park slots for every processor of every node: a store that places
    // memory on either node holds the others still.
    world.set_slots(cpus.iter().map(|c| c.park_slot()).collect());
    for (node, cpus) in nodes.iter().zip(&node_cpus) {
        connect_cpus(&node.chip, cpus)?;
    }
    // One JIT for the machine: every processor of every node compiles through
    // one queue into one page pool, each node an address space of it.
    #[cfg(feature = "jitv2")]
    {
        let pool: Vec<_> = nodes
            .iter()
            .zip(&specs)
            .zip(&node_cpus)
            .map(|((node, spec), cpus)| crate::cpu::jitv2::jitv2::PoolNode { cpus, module: spec.module, bus: node.chip.decode.physical().clone() })
            .collect();
        crate::cpu::jitv2::jitv2::share_pool(&pool, cfg.jitv2_pool_size(), cfg.jitv2.compile, &world).map_err(|e| format!("{name}: {e}"))?;
    }
    if let Some(hook) = &io.cpus {
        hook(&cpus);
    }
    for c in &cpus {
        c.register_locks();
    }
    for node in &nodes {
        node.flash.register_locks();
        node.chip.register_locks();
    }
    eprintln!(
        "iris: {}, {count} node{} of {} MiB, {} x {}",
        name,
        if count == 1 { "" } else { "s" },
        specs[0].banks_mib.iter().sum::<u32>(),
        cpus.len(),
        specs[0].module.part.label(),
    );
    let memory = nodes.iter().map(|n| n.chip.decode.physical().clone() as Arc<dyn Device>).collect();
    Machine::new(cfg, nodes, l1s, node_cpus, io, memory, timers, (events, receiver), |bedrock| {
        // A timer or the other chip can still reach this one's processor.
        bedrock.clear_cpu_interrupts();
        bedrock.decode.physical().release_devices();
    })
}

pub(crate) mod node;
pub(crate) mod presence;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

