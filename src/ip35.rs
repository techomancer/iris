//! The IP35 node shared by the Fuel, the Tezro and the Origin 3200 and Onyx 3000 C-bricks: the
//! Bedrock chip with its memory and boot flash, the processor bus over its uncached spaces
//! ([`crate::sn::UncachedBus`]) and the reset PROM's window.
//!
//! # The machines
//!
//! [`Ip35Machine`] is shared by every IP35 product: C-bricks, each a Bedrock node with its
//! processors, and the base I/O a product connects to them. A product is the facts about each node
//! ([`NodeSpec`]) and how its I/O is built on the first, as plain functions in its file: the
//! Origin 3200 and Onyx 3000 ([`crate::ip35_machine`]), the Fuel ([`crate::ip34_machine`]) and the
//! Tezro ([`crate::ip53_machine`]).
//!
//! Two C-bricks are joined by NUMAlink ([`Bedrock::connect_peer`]) and number their processors
//! across the machine. Each node's Physical shows every node's memory at its number, and one
//! [`World`] holds every processor still while a store moves memory.
//!
//! - Every C-brick runs one processor; the others are listed disabled in the node's flash, as
//!   the PROM would after three failed starts of each ([`NodeSpec::absent`]).
//! - A reset or power request from either node resets the machine; a node's reset alone is not modelled.
//! - Processors on several nodes share memory with no coherence protocol between the nodes.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::dev::bedrock::{Bedrock, RamExtent};
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
use crate::dev::heart::host_clock;
use crate::dev::prom_image;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::CiSerialBackend;
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link, NatControl};
use crate::perf_monitor::PerfMonitor;
use crate::scsi::ScsiBus;
use crate::sn::{NodeBoard, Nat, MAX_NODES};
use crate::traits::{Device, MachineEvent, Resettable};
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use node::{connect_cpus, NodeSpec};
use parking_lot::Mutex;
use serial::{connect_serial_ports, SerialPort};
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::atomic::AtomicU64;

use crate::dev::odyssey::PacketSender;
use crate::dev::xbridge::XBridge;
use crate::traits::BusDevice;
use crate::xtalk::XtalkHost;
use std::sync::Arc;

pub(crate) use crate::window_map::{map_device, SparseWindowMap, Window, CPU_PHYS_MASK};

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

/// The IOC3 of `board` at device 4 of head `pci`, with its SuperIO interrupt, and the board's serial
/// connections in console order: the L1 (on `l1_port`, named by `l1_setting`, logging to `l1_log`),
/// then the IOC3's two ports.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attach_ioc3(
    cfg: &MachineConfig,
    pci: &PciBus,
    board: &'static Ioc3Board,
    l1: &Arc<L1>,
    timekeeper: &Arc<Timekeeper>,
    l1_port: u16,
    l1_setting: &'static str,
    l1_log: Option<String>,
) -> (Arc<Ioc3>, Vec<SerialPort>) {
    let ioc3 = Arc::new(Ioc3::new(board, l1.station_mac(), ByteBus::M48t35(timekeeper.clone())));
    pci.attach(DevFn { device: 4, function: 0 }, ioc3.clone());
    ioc3.connect_superio_interrupt(pci.board_irq(BoardIrq(0)));
    let serial: [Arc<CiSerialBackend>; 3] = std::array::from_fn(|_| Arc::new(CiSerialBackend::new()));
    l1.set_serial_backend(serial[0].clone());
    ioc3.set_serial_backend(0, serial[1].clone());
    ioc3.set_serial_backend(1, serial[2].clone());
    let [a, b] = cfg.serial_ports();
    let (l1, ioc3_a, ioc3_b) = (l1.clone(), ioc3.clone(), ioc3.clone());
    let ports = vec![
        SerialPort {
            label: "L1",
            port: l1_port,
            setting: l1_setting,
            ci: serial[0].clone(),
            log: l1_log,
            connect: Box::new(move |backend| l1.set_serial_backend(backend)),
        },
        SerialPort {
            label: "IOC3 A",
            port: a,
            setting: "serial_port_a",
            ci: serial[1].clone(),
            log: None,
            connect: Box::new(move |backend| ioc3_a.set_serial_backend(0, backend)),
        },
        SerialPort {
            label: "IOC3 B",
            port: b,
            setting: "serial_port_b",
            ci: serial[2].clone(),
            log: None,
            connect: Box::new(move |backend| ioc3_b.set_serial_backend(1, backend)),
        },
    ];
    (ioc3, ports)
}

/// The OHCI controller at device 5 of head `pci`. With `input`, a USB keyboard and mouse sit on its
/// first two root ports and take the host's input instead of `ioc3`'s PS/2 pair, or a guest that
/// opened both would see every key twice.
pub(crate) fn attach_usb(pci: &PciBus, ioc3: &Ioc3, input: bool) -> Result<Arc<Ohci>, String> {
    let usb = Ohci::new();
    usb.set_clock(host_clock());
    if input {
        let (keyboard, mouse) = (HidFunction::keyboard(), HidFunction::mouse());
        usb.attach(0, keyboard.clone())?;
        usb.attach(1, mouse.clone())?;
        ioc3.ps2().redirect_input(UsbHidInput::new(keyboard, mouse));
    }
    pci.attach(DevFn { device: 5, function: 0 }, usb.clone());
    Ok(usb)
}

/// A VPro at crossbar `widget` of `bedrock`'s node: its DMA and interrupt packets enter Bedrock as
/// that widget's inbound traffic. `attach` puts it behind the board's crossbar and returns what the
/// node's I/O space reaches it by.
pub(crate) fn attach_vpro(
    cfg: &MachineConfig,
    bedrock: &Bedrock,
    widget: u8,
    status: &Arc<crate::disp::StatusSource>,
    attach: impl FnOnce(Arc<Odyssey>) -> Result<Arc<dyn BusDevice>, String>,
) -> Result<Arc<Odyssey>, String> {
    let vpro = Odyssey::new_widget_relative(
        widget,
        cfg.graphics_board().as_str(),
        odyssey_packets(bedrock, widget),
        bedrock.inbound(widget),
        status.clone(),
        (cfg.graphics.threads, cfg.graphics.tile),
        (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()),
    )?;
    bedrock.io.connect_widget(widget, attach(vpro.clone())?);
    Ok(vpro)
}

/// An Odyssey board's interrupt packets at `widget`: only those for Bedrock's port enter it.
pub(crate) fn odyssey_packets(bedrock: &Bedrock, widget: u8) -> PacketSender {
    use crate::irq::InterruptController;
    let port = bedrock.inbound(widget);
    Arc::new(move |to, address, vector, level| to == BEDROCK_PORT.widget && port.packet(widget, address, vector.into(), level))
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

/// What a board's I/O hands the machine once it is wired to the first node.
pub(crate) struct BoardIo {
    /// The devices with threads or timers: started in order, stopped in reverse.
    pub devices: Vec<Arc<dyn Device>>,
    /// Put the board's devices back to power-on.
    pub reset: Box<dyn Fn() + Send + Sync>,
    /// Brings the board's clocked devices up to date, every batch.
    pub poll: Arc<dyn Fn() + Send + Sync>,
    /// Per-processor batch hook on a node of several, beside the poll.
    pub at_batch: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Devices the monitor gets commands from.
    pub monitor: Vec<Arc<dyn Device>>,
    pub ps2: Arc<Ps2Controller>,
    /// Write the NVRAM back to its file.
    pub flush_nvram: Box<dyn Fn() -> Result<(), String> + Send + Sync>,
    /// The connections in console order: the first node's L1, then the base
    /// I/O's ports.
    pub serial: Vec<SerialPort>,
    /// The console, as an index into `serial`.
    pub console: usize,
    pub attach_net: Box<dyn Fn(Link) + Send + Sync>,
    /// Run the closure over the SCSI buses that hold the machine's disks.
    pub disks: Box<dyn Fn(&mut dyn FnMut(&[&ScsiBus])) + Send + Sync>,
    /// The controller whose removable media the front ends change, if any.
    pub disc_changer: Option<Arc<dyn DiscChanger>>,
    pub scratch: Option<PathBuf>,
    /// The graphics board's window, when the product has one.
    pub display: Option<Arc<dyn GfxDisplay>>,
    /// What the graphics board's status bar shows.
    pub status: Option<Arc<crate::disp::StatusSource>>,
    /// The counter the status bar blinks on the first processor's Compare matches.
    pub fasttick: Option<Arc<AtomicU64>>,
    pub cpus: Box<dyn Fn(&[Arc<dyn CpuDevice>]) + Send + Sync>,
    /// The XBridge, whose DMA views a test drives as a device would: the guest has no path to them.
    #[cfg(test)]
    pub xbridge: Option<Arc<crate::dev::xbridge::XBridge>>,
    /// The Tezro's PICs, whose bus views a test reaches as a card on them would: the guest
    /// has no path to a PIC's own PCI buses.
    #[cfg(test)]
    pub pics: Vec<Arc<crate::dev::pic::Pic>>,
}


pub(crate) struct Ip35Machine {
    /// The product's name as a sentence has it.
    name: &'static str,
    pub(crate) nodes: Vec<NodeBoard<Bedrock, Flash>>,
    /// Each node's L1, in node order: one per brick.
    pub(crate) l1s: Vec<Arc<L1>>,
    /// Each node's processors, in node order, the node's first first.
    pub(crate) node_cpus: Vec<Vec<Arc<dyn CpuDevice>>>,
    pub(crate) io: BoardIo,
    _world: Arc<World>,
    timers: Arc<TimerManager>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci_serial: Option<Arc<CiSerialBackend>>,
    ci: bool,
    nvram: String,
    nat: Option<Nat>,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
}

impl Ip35Machine {
    /// A machine of `cfg`'s product: `node_specs` gives each node's facts and `build_io` builds the
    /// board's I/O around the nodes and their L1s.
    pub fn new(
        cfg: &MachineConfig,
        node_specs: fn(&MachineConfig) -> Result<Vec<NodeSpec>, String>,
        build_io: fn(&MachineConfig, &[NodeBoard<Bedrock, Flash>], &[Arc<L1>], &Arc<TimerManager>) -> Result<BoardIo, String>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let name = cfg.machine.profile.ip35_name().ok_or_else(|| format!("{} is not an IP35 machine", cfg.machine.profile.label()))?;
        let prom = prom_image::resolve(cfg.machine.profile, &cfg.prom, 8 * 1024 * 1024)
            .map_err(|e| format!("{name} {e}"))?
            .bytes;
        let timers = Arc::new(TimerManager::new());
        let clock = host_clock();
        let specs = node_specs(cfg)?;
        assert!((1..=MAX_NODES).contains(&specs.len()), "a machine has one or two nodes");
        // Any node's reset or power request reaches the machine's main thread: a
        // reset of one node alone is not modelled.
        let (events, receiver) = channel();
        // The machine numbers its processors once, across its nodes in order.
        let (mut nodes, mut l1s, mut first_cpu) = (Vec::new(), Vec::new(), 0);
        for spec in &specs {
            let l1 = Arc::new(L1::new(spec.l1.clone())?);
            nodes.push(NodeBoard::<Bedrock, Flash>::build(spec, &prom, &clock, first_cpu, specs.len(), &events, &l1)?);
            l1s.push(l1);
            first_cpu += spec.processors.count();
        }
        let world = Arc::new(World::new());
        if let [a, b] = &nodes[..] {
            a.chip.connect_peer(&b.chip);
            b.chip.connect_peer(&a.chip);
        }
        for n in &nodes {
            n.chip.decode.set_world(world.clone());
        }
        let io = build_io(cfg, &nodes, &l1s, &timers)?;
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
                    // A node's first processor brings its boot flash, and the first node's the
                    // board's devices, up to date; every processor of a node of several joins the
                    // PROM's holds.
                    let flash = (n == 0).then(|| (node.flash.clone(), clock.clone()));
                    let board = (index == 0 && n == 0).then(|| io.poll.clone());
                    let held = io.at_batch.clone().filter(|_| several);
                    let hook: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
                        if let Some((flash, clock)) = &flash {
                            flash.advance_to(clock());
                        }
                        if let Some(board) = &board {
                            board();
                        }
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
        for (index, (node, cpus)) in nodes.iter().zip(&node_cpus).enumerate() {
            connect_cpus(name, &node.chip, &l1s[index], &events, cpus)?;
            node.chip.set_timer_manager(&timers);
        }
        // One JIT for the processors of a node, which share its memory, its
        // pool parking them with the rest; one of its own for a lone one.
        #[cfg(feature = "jitv2")]
        for (node, cpus) in nodes.iter().zip(&node_cpus) {
            let several = cpus.len() > 1;
            if several {
                if cfg!(feature = "jitv2_lockstep") {
                    return Err(format!("{name}: a node of several processors cannot run with jitv2_lockstep"));
                }
                for c in cpus {
                    c.share_jitv2(cpus[0].jitv2(), world.clone());
                }
            }
            crate::cpu::jitv2::jitv2::attach_pool(
                &cpus[0],
                cfg.jitv2_pool_size(),
                cfg.jitv2.compile,
                node.chip.decode.physical().clone(),
                several.then(|| world.clone()),
            );
        }
        (io.cpus)(&cpus);
        connect_serial_ports(cfg, name, &io.serial, io.console)?;
        let ci_serial = cfg.ci.then(|| io.serial[io.console].ci.clone());
        let first = &cpus[0];
        if let Some(status) = &io.status {
            status.set_cpu_cycles(first.cycles_ptr());
            status.set_count_hz(first.count_hz_atomic());
            for c in &cpus[1..] {
                status.add_cpu_stats(c.cycles_ptr(), c.fasttick_count());
            }
        }
        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(cpus.clone())));
        monitor.register_device(PerfMonitor::with_cpus(first.running_flag(), cpus.iter().map(|c| c.cycles_ptr()).collect(), first.fasttick_count()));
        for node in &nodes {
            monitor.register_device(node.chip.decode.physical().clone());
        }
        monitor.register_device(l1s[0].clone());
        for device in &io.monitor {
            monitor.register_device(device.clone());
        }
        for c in &cpus {
            c.register_locks();
        }
        for node in &nodes {
            node.flash.register_locks();
            node.chip.register_locks();
        }
        let mut nat = None;
        let net = GatewayConfig::from_network(&cfg.network());
        if let Some(link) = Link::with_gateway(net.clone())? {
            nat = Some(Nat { control: link.nat(), expected: (net.client_ip, net.gateway_ip, net.netmask), frames: link.frames() });
            (io.attach_net)(link);
        }
        eprintln!(
            "iris: {}, {count} node{} of {} MiB, {} x {}, serial console {}",
            name,
            if count == 1 { "" } else { "s" },
            specs[0].banks_mib.iter().sum::<u32>(),
            cpus.len(),
            specs[0].module.part.label(),
            io.serial[io.console].label,
        );
        Ok(Self {
            name,
            nodes,
            l1s,
            node_cpus,
            io,
            _world: world,
            timers,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci_serial,
            ci: cfg.ci,
            nvram: cfg.nvram.clone(),
            nat,
            events: Some(receiver),
            slot: None,
            lifecycle: GuestState::Idle,
        })
    }

    /// CPU 0 of the first node, the one the front ends drive.
    fn cpu(&self) -> &Arc<dyn CpuDevice> {
        &self.node_cpus[0][0]
    }

    fn cpus(&self) -> impl Iterator<Item = &Arc<dyn CpuDevice>> {
        self.node_cpus.iter().flatten()
    }

    pub(crate) fn with_disks<R>(&self, f: impl FnOnce(&[&ScsiBus]) -> R) -> R {
        let mut f = Some(f);
        let mut out = None;
        (self.io.disks)(&mut |buses| out = f.take().map(|f| f(buses)));
        out.expect("the board visits its disks once")
    }
}

impl Workstation for Ip35Machine {
    fn start(&mut self) {
        self.lifecycle = GuestState::Starting;
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        if !self.ci {
            self.cpu_start();
        }
    }
    fn stop(&mut self) {
        for c in self.cpus() {
            c.clone().as_device().stop();
        }
        let now = host_clock()();
        for node in &self.nodes {
            node.flash.advance_to(now);
        }
        for d in self.io.devices.iter().rev() {
            d.stop();
        }
        for (node, l1) in self.nodes.iter().zip(&self.l1s) {
            l1.stop();
            node.chip.stop();
        }
        if let Err(e) = (self.io.flush_nvram)() {
            crate::dlog!(LogModule::Bedrock, "{} NVRAM: {e}", self.name);
        }
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        for c in self.cpus() {
            c.power_on();
        }
        for (node, l1) in self.nodes.iter().zip(&self.l1s) {
            node.chip.power_on();
            l1.power_on();
            node.flash.power_on();
        }
        (self.io.reset)();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        let now = host_clock()();
        for (node, l1) in self.nodes.iter().zip(&self.l1s) {
            node.flash.hold_clock(now);
            node.chip.start();
            l1.start();
        }
        for d in &self.io.devices {
            d.start();
        }
    }
    /// Every processor of every node starts at power-on; each node's PROM waits for the
    /// others' link and discovery.
    fn cpu_start(&self) {
        for c in self.cpus() {
            c.clone().as_device().start();
        }
    }
    fn cpu_is_running(&self) -> bool {
        self.cpus().any(|c| c.clone().as_device().is_running())
    }
    fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut Self as *mut dyn Workstation;
        self.slot.get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr)))).clone()
    }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self.events.take().expect("the system controller is registered once");
        register_system_controller(&self.monitor, slot, Some(events));
    }
    fn monitor_addr(&self) -> Option<SocketAddr> {
        self.monitor.bound_addr()
    }
    fn lifecycle(&self) -> GuestState {
        self.lifecycle
    }
    fn set_lifecycle(&mut self, state: GuestState) {
        self.lifecycle = state;
    }
    fn get_display(&self) -> Option<Arc<dyn GfxDisplay>> {
        self.io.display.clone()
    }
    // The chips and node memory keep no saved state, so a snapshot would not restore the machine.
    fn save_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the Bedrock chips and node memory have no saved state", self.name))
    }
    fn load_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the Bedrock chips and node memory have no saved state", self.name))
    }
    fn register_monitor_device(&self, device: Arc<dyn Device>) {
        self.monitor.register_device(device);
    }
    fn get_ps2(&self) -> Arc<Ps2Controller> {
        self.io.ps2.clone()
    }
    fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timers.clone()
    }
    fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> {
        self.cpu().clone().debug_adapter()
    }
    fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.cpu().fasttick_count()
    }
    fn cpu_counters(&self) -> Vec<(crate::cpu::mips_core::CyclesPtr, Arc<AtomicU64>)> {
        self.cpus().map(|c| (c.cycles_ptr(), c.fasttick_count())).collect()
    }
    fn get_ci_serial(&self) -> Option<Arc<CiSerialBackend>> {
        self.ci_serial.clone()
    }
    fn inject_serial_console(&self, bytes: &[u8]) {
        self.io.serial[self.io.console].ci.push_host(bytes);
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        if path.is_some_and(|p| p != self.nvram) {
            return Err(format!("{} NVRAM save uses the configured path", self.name));
        }
        (self.io.flush_nvram)()?;
        Ok(self.nvram.clone())
    }
    fn scratch_path(&self) -> Option<&Path> {
        self.io.scratch.as_deref()
    }
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        self.io.disc_changer.clone()
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.with_disks(|buses| buses.iter().map(|b| b.pending_chd_sync_count()).sum())
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.with_disks(|buses| buses.iter().try_for_each(|b| b.flush()))
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        self.with_disks(|buses| ScsiBus::sync_chd_disks(buses, only, progress, cancel))
    }
    fn net_frames(&self) -> Option<Arc<AtomicU64>> {
        self.nat.as_ref().map(|n| n.frames.clone())
    }
    fn nat_control(&self) -> Option<Arc<NatControl>> {
        self.nat.as_ref().map(|n| n.control.clone())
    }
    fn nat_expected(&self) -> Option<(Ipv4Addr, Ipv4Addr, Ipv4Addr)> {
        self.nat.as_ref().map(|n| n.expected)
    }
}

impl Drop for Ip35Machine {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        // Nothing publishes into a processor's interrupt word once the
        // processors go: a timer or the other chip can still reach this one.
        for node in &self.nodes {
            node.chip.clear_cpu_interrupts();
        }
        // The compile pool's workers read the node memory the machine frees.
        #[cfg(feature = "jitv2")]
        for cpus in &self.node_cpus {
            crate::cpu::jitv2::Jitv2::stop_for_drop(&cpus[0].jitv2(), || {
                for c in cpus {
                    c.clone().as_device().stop();
                }
            });
        }
        crate::workstation::flush_on_drop(self);
        self.monitor.shutdown();
        // Each node's table points into the other's banks, so every table
        // goes before either node does.
        for node in &self.nodes {
            node.chip.decode.physical().release_devices();
        }
    }
}

pub(crate) mod node;
pub(crate) mod presence;
pub(crate) mod serial;
#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;

