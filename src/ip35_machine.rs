//! The IP35 machines built from bricks: C-bricks, each a Bedrock node with its processors, and the
//! base I/O a product cables to them.
//!
//! [`Ip35Machine`] is shared by every such product. A product is the facts about each node
//! ([`NodeSpec`]) and how its I/O is wired to the first, as plain functions in its file: the
//! Origin 3200 ([`origin3200`]), the Onyx 3000, and the one-node Fuel and Tezro.
//!
//! Two C-bricks are joined by NUMAlink ([`Bedrock::connect_peer`]) and number their processors
//! across the machine. Each node's Physical shows every node's memory at its number, and one
//! [`World`] holds every processor still while a store moves memory.
//!
//! # What is approximated here
//!
//! - Every C-brick runs one processor; the others are listed disabled in the node's flash, as
//!   the PROM would after three failed starts of each (`NodeSpec::absent`).
//! - A reset or power request from either node resets the machine; a node's reset alone is not modelled.
//! - Processors on several nodes share memory with no coherence protocol between the nodes.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

mod onyx3000;
mod origin3200;

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::build::build_cpu;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu};
use crate::cpu::smp::World;
use crate::dev::bedrock::Bedrock;
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
use crate::ip27_machine::Nat;
use crate::ip35::node::connect_cpus;
pub(crate) use crate::ip35::node::NodeSpec;
pub(crate) use crate::ip35::serial::SerialPort;
use crate::ip35::serial::connect_serial_ports;
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link, NatControl};
use crate::perf_monitor::PerfMonitor;
use crate::scsi::ScsiBus;
use crate::sn_nodes::{NodeBoard, MAX_NODES};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable};
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use parking_lot::Mutex;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

pub use origin3200::CPUS;

/// What a board's I/O hands the machine once it is wired to the first node.
pub(crate) struct BoardIo {
    /// The board's devices come up (their threads and timers start).
    pub start: Box<dyn Fn() + Send + Sync>,
    pub stop: Box<dyn Fn() + Send + Sync>,
    /// Put the board's devices back to power-on.
    pub reset: Box<dyn Fn() + Send + Sync>,
    /// Brings the board's clocked devices up to date, every batch.
    pub poll: Arc<dyn Fn() + Send + Sync>,
    /// Per-processor batch hook on a node of several, beside the poll.
    pub at_batch: Option<Arc<dyn Fn() + Send + Sync>>,
    pub register_locks: Box<dyn Fn() + Send + Sync>,
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

/// The nodes the machine's profile names.
fn specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
    match cfg.machine.profile {
        MachineProfile::Origin3200Ip35 => origin3200::specs(cfg),
        MachineProfile::Onyx3000Ip35 => onyx3000::specs(cfg),
        MachineProfile::FuelIp34 => crate::ip34_machine::specs(cfg),
        MachineProfile::TezroIp53 => crate::ip53_machine::specs(cfg),
        other => Err(format!("{} is not an IP35 machine", other.label())),
    }
}

/// Wire the product's I/O to the first node and the machine's L1s, its clocked devices on `timers`.
fn wire(cfg: &MachineConfig, nodes: &[NodeBoard<Bedrock, Flash>], l1s: &[Arc<L1>], timers: &Arc<TimerManager>) -> Result<BoardIo, String> {
    match cfg.machine.profile {
        MachineProfile::Origin3200Ip35 => origin3200::wire(cfg, nodes, l1s, timers),
        MachineProfile::Onyx3000Ip35 => onyx3000::wire(cfg, nodes, l1s, timers),
        MachineProfile::FuelIp34 => crate::ip34_machine::wire(cfg, nodes, l1s, timers),
        MachineProfile::TezroIp53 => crate::ip53_machine::wire(cfg, nodes, l1s, timers),
        other => Err(format!("{} is not an IP35 machine", other.label())),
    }
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
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        cfg.validate()?;
        let name = cfg.machine.profile.ip35_name().ok_or_else(|| format!("{} is not an IP35 machine", cfg.machine.profile.label()))?;
        let prom = prom_image::resolve(cfg.machine.profile, &cfg.prom, 8 * 1024 * 1024)
            .map_err(|e| format!("{name} {e}"))?
            .bytes;
        let timers = Arc::new(TimerManager::new());
        let clock = host_clock();
        let specs = specs(cfg)?;
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
        let io = wire(cfg, &nodes, &l1s, &timers)?;
        let count = nodes.len();
        // Every processor reaches every node's memory at its number.
        let maps: Vec<_> = nodes.iter().map(|n| crate::ip35::memory_map(&n.chip.md.ram_extent())).collect();
        let memory = Arc::new(crate::sn_nodes::machine_memory(&maps, crate::sn_nodes::SN1_NODE_SHIFT));
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
        (io.register_locks)();
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
        (self.io.stop)();
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
        (self.io.start)();
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

#[cfg(test)]
pub(crate) mod tests;
