//! The IP35 machines built from bricks: C-bricks, each a Bedrock node with its
//! processors, and the base I/O a product cables to them.
//!
//! [`Ip35Machine`] is the machine every such product shares. A product is a
//! board ([`Ip35Board`]): the facts about each C-brick ([`NodeSpec`]: its
//! processor part, firmware row, board identification word, banks and L1
//! identity) and the one thing that differs in behaviour, how its I/O is
//! wired to the first C-brick ([`Ip35Board::wire`], which hands back a
//! [`BoardIo`]). One board is here, the Origin 3200
//! ([`origin3200`]), selected with `[machine] profile = "origin3200_ip35"`.
//! The Fuel and the Tezro are single boards of their own for now
//! ([`crate::ip34_machine`], [`crate::ip53_machine`]) and share the node
//! (`ip35::node`) but not this machine.
//!
//! # What is shared across the nodes
//!
//! Two C-bricks are joined by the NUMAlink cable ([`Bedrock::cable`]) and
//! number their processors across the machine, so a processor's thread says
//! which node it is on. The chip decodes node numbers
//! ([`crate::dev::bedrock`]); each node's ppmem window shows every node's
//! memory at its number ([`crate::node_windows::Windows`], node field at bit
//! 33), and one [`World`] holds every processor still while a bank
//! configuration write moves memory under their views, whichever node the
//! write lands on.
//!
//! # What is approximated here
//!
//! - Every C-brick runs one processor. A C-brick holds up to four; the others
//!   are listed disabled in the node's flash, as the PROM would after its
//!   third failed start of each ([`NodeSpec::absent`]).
//! - A reset or power request from either node resets the machine: a node's
//!   reset alone is not modelled.
//! - Processors on several nodes share memory with no coherence protocol
//!   between the nodes; both see the same host memory.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

mod onyx3000;
mod origin3200;

use crate::config::MachineConfig;
use crate::cpu::mips_cache_v2::{CpuModel as CacheCpuModel, MipsCache};
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::cpu::parts::{self, Part};
use crate::cpu::smp::World;
use crate::dev::bedrock::{Bedrock, Population};
use crate::dev::bedrock_flash::{disabled_positions_environment, FirmwareRecord};
use crate::dev::heart::{host_clock, Clock};
use crate::dev::ioc3::eth::Ioc3Eth;
use crate::dev::l1::L1Config;
use crate::dev::prom_image;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::{CiSerialBackend, SerialBackend};
use crate::dev::timekeeper::Timekeeper;
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::ip35::node::{connect_cpus, connect_placed_cpus, Node, NodeConfig, Placement};
use crate::ip35::serial::{connect_serial_ports, SerialPort};
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link, NatControl};
use crate::node_windows::{Windows, SN1_NODE_SHIFT};
use crate::perf_monitor::PerfMonitor;
use crate::traits::{BusDevice, Device, MachineEvent, Resettable};
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use parking_lot::Mutex;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

pub use origin3200::CPUS;
use onyx3000::Onyx3000;
use origin3200::Origin3200;

/// The Origin 3200: one or two C-bricks and an I-brick.
pub(crate) type Origin3200Machine = Ip35Machine<Origin3200>;

/// The Onyx 3000: the Origin 3200 with a graphics brick (`onyx3000`).
pub(crate) type Onyx3000Machine = Ip35Machine<Onyx3000>;

/// The most processors a machine numbers across its nodes: the CPU limit of
/// the machine's park slots.
const MAX_NODES: usize = 2;

/// What a product says about one of its C-bricks.
pub(crate) struct NodeSpec {
    /// The C-brick's name in logs and map names.
    pub name: &'static str,
    pub part: &'static Part,
    /// The boot flash's configuration record for this part.
    pub firmware: FirmwareRecord,
    /// The word the board reads back at the Hub-special board-id window.
    pub board_id: u64,
    /// The memory banks in MiB, locational bank 0 upward.
    pub banks_mib: Vec<u32>,
    /// The processor slots fitted.
    pub processors: Population,
    /// The processor positions (letters A to D) with nothing fitted. A fresh
    /// flash lists them disabled, which spares the PROM three power cycles for
    /// each.
    pub absent: &'static str,
    pub l1: L1Config,
    /// Persistent boot flash; absent means volatile.
    pub flash: Option<PathBuf>,
}

/// One of the board's serial connections, for the machine to connect to the
/// host: [`connect_serial_ports`] takes its fields.
pub(crate) struct SerialLine {
    pub label: &'static str,
    pub port: u16,
    /// The setting that names the port, for the message when it is taken.
    pub setting: &'static str,
    pub ci: Arc<CiSerialBackend>,
    pub log: Option<String>,
    pub connect: Box<dyn Fn(Arc<dyn SerialBackend>) + Send + Sync>,
}

/// What a board's I/O hands the machine once it is wired to the first
/// C-brick.
pub(crate) struct BoardIo {
    /// The board's devices come up (their threads and timers start).
    pub start: Box<dyn Fn() + Send + Sync>,
    pub stop: Box<dyn Fn() + Send + Sync>,
    /// Put the board's devices back to power-on.
    pub reset: Box<dyn Fn() + Send + Sync>,
    /// Brings the board's clocked devices up to date, from the first
    /// C-brick's processor, every batch.
    pub poll: Arc<dyn Fn() + Send + Sync>,
    pub register_locks: Box<dyn Fn() + Send + Sync>,
    /// Devices the monitor gets commands from.
    pub monitor: Vec<Arc<dyn Device>>,
    pub ps2: Arc<Ps2Controller>,
    pub timekeeper: Arc<Timekeeper>,
    /// The connections in console order: the first C-brick's L1, then the
    /// base I/O's ports.
    pub serial: Vec<SerialLine>,
    /// The console, as an index into `serial`.
    pub console: usize,
    pub eth: Arc<Ioc3Eth>,
    /// The Fibre Channel adapter whose loop holds the machine's disks.
    pub fibre: Option<Arc<crate::dev::isp2200::Isp2200>>,
    /// The graphics board's window, when the product has one.
    pub display: Option<Arc<dyn GfxDisplay>>,
    /// What the graphics board's status bar shows. The machine hands it the
    /// first processor's cycle counter.
    pub status: Option<Arc<crate::disp::StatusSource>>,
}

/// A product. Only wiring differs in behaviour between products;
/// everything else about one is the [`NodeSpec`]s.
pub(crate) trait Ip35Board: 'static {
    /// The product's name as a sentence has it.
    const NAME: &'static str;
    /// The C-bricks the product carries, in slot order.
    fn specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String>;
    /// Wire the board's I/O to the first C-brick.
    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String>;
}

/// The nodes of a machine, cabled, with the board's I/O wired to them and
/// no processors: what a test can build without a thread.
pub(crate) struct Chassis<B: Ip35Board> {
    pub nodes: Vec<Node>,
    /// What each node's processors run on.
    pub processors: Vec<Arc<dyn BusDevice>>,
    pub specs: Vec<NodeSpec>,
    pub io: BoardIo,
    /// The nodes' windows, when there is more than one node.
    pub windows: Option<Arc<Windows>>,
    /// Every node's resets and power events.
    pub events: Receiver<MachineEvent>,
    _board: std::marker::PhantomData<fn() -> B>,
}

impl<B: Ip35Board> Chassis<B> {
    pub(crate) fn build(cfg: &MachineConfig, prom: &[u8], clock: Clock) -> Result<Self, String> {
        let specs = B::specs(cfg)?;
        assert!((1..=MAX_NODES).contains(&specs.len()), "a machine has one or two nodes");
        let count = specs.len();
        let mut nodes = Vec::new();
        let mut processors = Vec::new();
        let (merged_tx, merged_rx) = channel();
        for (index, spec) in specs.iter().enumerate() {
            let placement = if count > 1 { Placement::Of { nodes: count, first_cpu: index } } else { Placement::Alone };
            let (mut node, processor) = Node::build_placed(
                NodeConfig {
                    name: spec.name,
                    payload: prom,
                    clock: clock.clone(),
                    flash_path: spec.flash.as_deref(),
                    banks_mib: &spec.banks_mib,
                    processors: spec.processors,
                    firmware: spec.firmware,
                    board_id: spec.board_id,
                    l1: spec.l1.clone(),
                },
                placement,
                &disabled_positions_environment(spec.absent),
            )?;
            // Any node's reset or power request reaches the machine's main
            // thread: a reset of one node alone is not modelled.
            let events = std::mem::replace(&mut node.receiver, channel().1);
            let tx = merged_tx.clone();
            std::thread::Builder::new()
                .name(format!("ip35-events-{index}"))
                .spawn(move || {
                    for event in events {
                        if tx.send(event).is_err() {
                            break;
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
            nodes.push(node);
            processors.push(processor);
        }
        drop(merged_tx);
        let windows = if let [a, b] = nodes.as_slice() {
            Bedrock::cable(&a.bedrock, &b.bedrock);
            let windows = Arc::new(Windows::new(SN1_NODE_SHIFT));
            for node in &nodes {
                windows.add_node(node.bedrock.physical(), node.bedrock.shape());
                node.bedrock.set_windows(&windows);
            }
            Some(windows)
        } else {
            None
        };
        let io = B::wire(&nodes.iter().collect::<Vec<_>>(), cfg)?;
        Ok(Self { nodes, processors, specs, io, windows, events: merged_rx, _board: std::marker::PhantomData })
    }
}

/// What a node's processor brings up to date every batch: the node's boot
/// flash clock, and on the first node the board's clocked devices.
struct NodePoll {
    clock: Clock,
    last: AtomicU64,
    running: AtomicBool,
    flash: Arc<crate::dev::bedrock_flash::Flash>,
    board: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl NodePoll {
    fn new(clock: Clock, flash: Arc<crate::dev::bedrock_flash::Flash>, board: Option<Arc<dyn Fn() + Send + Sync>>) -> Self {
        Self { last: AtomicU64::new(clock()), clock, running: AtomicBool::new(false), flash, board }
    }

    fn start(&self) {
        self.last.store((self.clock)(), Ordering::Release);
        self.running.store(true, Ordering::Release);
    }

    fn stop(&self) {
        self.poll();
        self.running.store(false, Ordering::Release);
    }

    fn poll(&self) {
        if self.running.load(Ordering::Acquire) {
            let now = (self.clock)();
            let previous = self.last.swap(now, Ordering::AcqRel);
            self.flash.advance_ns(now.saturating_sub(previous));
            if let Some(board) = &self.board {
                board();
            }
        }
    }
}

/// A C-brick with its processor.
struct Unit {
    node: Node,
    cpu: Arc<dyn CpuDevice>,
    poll: Arc<NodePoll>,
}

pub(crate) struct Ip35Machine<B: Ip35Board> {
    units: Vec<Unit>,
    io: BoardIo,
    /// The nodes' windows, which the chips hold weakly.
    _windows: Option<Arc<Windows>>,
    _world: Arc<World>,
    timers: Arc<TimerManager>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci_serial: Option<Arc<CiSerialBackend>>,
    ci: bool,
    scratch: Option<PathBuf>,
    nvram: String,
    frames: Option<Arc<AtomicU64>>,
    nat: Option<(Arc<NatControl>, (Ipv4Addr, Ipv4Addr, Ipv4Addr))>,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
    _board: std::marker::PhantomData<fn() -> B>,
}

impl<B: Ip35Board> Ip35Machine<B> {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        cfg.validate()?;
        let prom = prom_image::resolve(cfg.machine.profile, &cfg.prom, 8 * 1024 * 1024)
            .map_err(|e| format!("{} {e}", B::NAME))?
            .bytes;
        let Chassis { nodes, processors, specs, io, windows, events, .. } = Chassis::<B>::build(cfg, &prom, host_clock())?;
        let count = nodes.len();
        let timers = Arc::new(TimerManager::new());
        let board_poll = io.poll.clone();
        let mut units = Vec::new();
        for (index, ((node, spec), bus)) in nodes.into_iter().zip(&specs).zip(processors).enumerate() {
            let poll = Arc::new(NodePoll::new(host_clock(), node.flash.clone(), (index == 0).then(|| board_poll.clone())));
            let count_hz = cfg.clock.count_hz(spec.part).max(1);
            let cpu = crate::with_cpu!(spec.part, true, false, C => Self::start_cpu::<C>(&node, spec, bus, index, count, &timers, count_hz, poll.clone()));
            units.push(Unit { node, cpu, poll });
        }
        // One body of park slots for every processor of every node: a bank
        // configuration write on either node holds the others still, the
        // other node's processors included, whichever node's thread makes it.
        let world = Arc::new(World::new());
        world.set_slots(units.iter().map(|u| u.cpu.park_slot()).collect());
        for u in &units {
            u.node.bedrock.set_world(world.clone());
            let cpus = std::slice::from_ref(&u.cpu);
            if count > 1 {
                connect_placed_cpus(B::NAME, &u.node.bedrock, &u.node.l1, &u.node.events, cpus)?;
            } else {
                connect_cpus(B::NAME, &u.node.bedrock, &u.node.l1, &u.node.events, cpus)?;
            }
            u.node.bedrock.set_timer_manager(&timers);
        }
        #[cfg(feature = "jitv2")]
        for u in &units {
            let cpu_device: Arc<dyn Device> = u.cpu.clone().as_device();
            let jit = u.cpu.jitv2();
            let mut g = jit.lock();
            g.compile_queue.set_thread_count(cfg.jitv2.threads.max(1));
            g.compile_queue.set_cpu(Arc::downgrade(&cpu_device));
            g.compile_queue.set_owner(Arc::downgrade(&jit));
            // Not under lockstep, as on the other machines: lockstep compiles
            // inline on the CPU thread, and a running pool both owns the
            // Codegen inline needs and deadlocks the CPU thread's flush.
            #[cfg(not(feature = "jitv2_lockstep"))]
            {
                drop(g);
                crate::cpu::jitv2::jitv2::start_compiling(&*u.cpu, &jit, cfg.jitv2.compile, u.node.code.clone());
            }
            // The compile pool keeps only a weak handle on the CPU.
            std::mem::forget(cpu_device);
        }
        let scratch = None;
        let ports: Vec<_> = io.serial.iter().map(|line| SerialPort {
            label: line.label,
            port: line.port,
            setting: line.setting,
            ci: &line.ci,
            log: line.log.clone(),
            connect: Box::new(|backend| (line.connect)(backend)),
        }).collect();
        connect_serial_ports(cfg, B::NAME, &ports, io.console)?;
        drop(ports);
        let ci_serial = cfg.ci.then(|| io.serial[io.console].ci.clone());
        let first = &units[0].cpu;
        if let Some(status) = &io.status {
            status.set_cpu_cycles(first.cycles_ptr());
            status.set_count_hz(first.count_hz_atomic());
        }
        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(units.iter().map(|u| u.cpu.clone()).collect())));
        monitor.register_device(PerfMonitor::new(first.running_flag(), first.cycles_ptr(), first.fasttick_count(), None, None));
        for u in &units {
            monitor.register_device(u.node.bedrock.physical().clone());
        }
        monitor.register_device(units[0].node.l1.clone());
        for device in &io.monitor {
            monitor.register_device(device.clone());
        }
        // `scsi2` and `cow2`: the Fibre Channel loop's disks.
        if let Some(fibre) = &io.fibre {
            monitor.register_device(fibre.clone());
        }
        for u in &units {
            u.cpu.register_locks();
            u.node.flash.register_locks();
            u.node.bedrock.register_locks();
        }
        (io.register_locks)();
        if let Some(fibre) = &io.fibre {
            fibre.register_locks();
        }
        let (mut nat, mut frames) = (None, None);
        let net = GatewayConfig::from_network(&cfg.network());
        if let Some(link) = Link::with_gateway(net.clone())? {
            frames = Some(link.frames());
            nat = Some((link.nat(), (net.client_ip, net.gateway_ip, net.netmask)));
            io.eth.attach(link);
        }
        eprintln!(
            "iris: {}, {count} C-brick{} of {} MiB, {}, serial console {}",
            B::NAME,
            if count == 1 { "" } else { "s" },
            specs[0].banks_mib.iter().sum::<u32>(),
            specs[0].part.label(),
            io.serial[io.console].label,
        );
        Ok(Self {
            units,
            io,
            _windows: windows,
            _world: world,
            timers,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci_serial,
            ci: cfg.ci,
            scratch,
            nvram: cfg.nvram.clone(),
            nat,
            frames,
            events: Some(events),
            slot: None,
            lifecycle: GuestState::Idle,
            _board: std::marker::PhantomData,
        })
    }

    /// Processor `index` of `count`, of type `C`, on its node's bus: the
    /// part's identity and reset-mode inputs, the machine's number for it, and
    /// a batch hook that polls its node.
    #[allow(clippy::too_many_arguments)]
    fn start_cpu<C>(
        node: &Node,
        spec: &NodeSpec,
        bus: Arc<dyn BusDevice>,
        index: usize,
        count: usize,
        timers: &Arc<TimerManager>,
        count_hz: u64,
        poll: Arc<NodePoll>,
    ) -> Arc<dyn CpuDevice>
    where
        C: CacheCpuModel + MipsCache + From<Arc<dyn BusDevice>> + 'static,
        MipsCpu<MipsTlb, C>: CpuDevice,
    {
        let cfg = MipsCpuConfig::for_model::<C>(Arc::new(crate::ip35::memory_map(&node.bedrock.ram_extent())))
            .with_prid(spec.part.prid());
        let mut exec: MipsExecutor<MipsTlb, C> = MipsExecutor::new(bus, MipsTlb::new(cfg.tlb_entries), &cfg);
        exec.phys_mask = crate::ip35::CPU_PHYS_MASK;
        // The flash record's mode word selects these boot-mode inputs. A
        // mismatch enters the PROM's error routine before GP is initialized.
        exec.core.cp0_config = spec.firmware.reset_config0();
        exec.core.set_timer_manager(timers.clone());
        exec.core.set_count_hz(count_hz);
        exec.core.count_write_samples = true;
        exec.core.cpu_id = index as u8;
        // Processors on several nodes share memory: another's store cannot
        // clear this one's LLbit, so a store-conditional must be a host
        // compare-exchange.
        exec.core.smp = count > 1;
        exec.cache = C::from(node.cached.clone());
        // The PROM fetches its own code uncached. A 512 MiB bank among banks 0
        // to 3 can cover its window, where the cached plane answers RAM, so the
        // processor presents the fetch at the PROM's own address.
        if C::FETCH_ALIAS {
            exec.set_uncached_fetch_alias(
                crate::ip35::RESET_WINDOW,
                crate::ip35::RESET_WINDOW_SIZE,
                crate::ip35::RESET_WINDOW_FETCH,
            );
        }
        #[cfg(feature = "jitv2")]
        exec.set_cached_bus(node.code.clone());
        #[cfg(not(feature = "jitv2"))]
        exec.set_cached_bus(node.cached.clone());
        exec.batch_hook = Some(Box::new(move |_| poll.poll()));
        Arc::new(MipsCpu::new(exec))
    }

    /// The SCSI buses that hold disks: the Fibre Channel loop.
    fn disk_buses(&self) -> Vec<&crate::scsi::ScsiBus> {
        use crate::scsi_monitor::ScsiTargets;
        self.io.fibre.iter().map(|f| f.scsi_bus()).collect()
    }

    /// CPU 0 of the first node, the one the front ends drive.
    fn cpu(&self) -> &Arc<dyn CpuDevice> {
        &self.units[0].cpu
    }
}

impl<B: Ip35Board> Workstation for Ip35Machine<B> {
    fn start(&mut self) {
        self.lifecycle = GuestState::Starting;
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        if !self.ci {
            self.cpu_start();
        }
    }
    fn stop(&mut self) {
        for u in &self.units {
            u.cpu.clone().as_device().stop();
            u.poll.stop();
        }
        (self.io.stop)();
        for u in &self.units {
            u.node.l1.stop();
            u.node.bedrock.stop();
        }
        if let Err(e) = self.io.timekeeper.flush_nvram() {
            crate::dlog!(LogModule::Bedrock, "{} NVRAM: {e}", B::NAME);
        }
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        for u in &self.units {
            u.cpu.power_on();
            u.node.bedrock.power_on();
            u.node.l1.power_on();
            u.node.flash.power_on();
        }
        (self.io.reset)();
        self.io.timekeeper.power_on();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        for u in &self.units {
            u.poll.start();
            u.node.bedrock.start();
            u.node.l1.start();
        }
        (self.io.start)();
    }
    fn cpu_start(&self) {
        // Every node's PROM starts at power-on: each waits for the others'
        // link and discovery, and the first node gives up on a peer that
        // comes up after its second poll of the link.
        for u in &self.units {
            u.cpu.clone().as_device().start();
        }
    }
    fn cpu_is_running(&self) -> bool {
        self.cpu().clone().as_device().is_running()
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
    // The C-bricks' Bedrock chips, their node memory and the I-brick's devices
    // keep no saved state, so a snapshot would restore a machine that is not
    // the one saved. Say so rather than the generic "not supported here".
    fn save_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the Bedrock chips and node memory have no saved state", B::NAME))
    }
    fn load_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the Bedrock chips and node memory have no saved state", B::NAME))
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
            return Err(format!("{} NVRAM save uses the configured path", B::NAME));
        }
        self.io.timekeeper.flush_nvram()?;
        Ok(self.nvram.clone())
    }
    fn scratch_path(&self) -> Option<&Path> {
        self.scratch.as_deref()
    }
    // The loop's disks are plain disks: nothing to change.
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        None
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.disk_buses().iter().map(|b| b.pending_chd_sync_count()).sum()
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.disk_buses().iter().try_for_each(|b| b.flush())
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        crate::scsi::ScsiBus::sync_chd_disks(&self.disk_buses(), only, progress, cancel)
    }
    fn net_frames(&self) -> Option<Arc<AtomicU64>> {
        self.frames.clone()
    }
    fn nat_control(&self) -> Option<Arc<NatControl>> {
        self.nat.as_ref().map(|(n, _)| n.clone())
    }
    fn nat_expected(&self) -> Option<(Ipv4Addr, Ipv4Addr, Ipv4Addr)> {
        self.nat.as_ref().map(|(_, a)| *a)
    }
}

impl<B: Ip35Board> Drop for Ip35Machine<B> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        // The compile pool's workers read the node memory the machine frees.
        #[cfg(feature = "jitv2")]
        for u in &self.units {
            crate::cpu::jitv2::Jitv2::stop_for_drop(&u.cpu.jitv2(), || u.cpu.clone().as_device().stop());
        }
        crate::workstation::flush_on_drop(self);
        self.monitor.shutdown();
    }
}

#[cfg(test)]
mod tests;
