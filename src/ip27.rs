//! The IP27 processor bus: a node's memory, its uncached views and the Hub special windows.
//!
//! Two attribute bits beside an uncached address (XKPHYS bits 58:57) pick one of four spaces,
//! which [`crate::sn::UncachedBus`] indexes. Cached references, and uncached ones to memory
//! (attribute 3), use the node's `Physical` and its ppmem window; the Hub owns and remaps it.
//!
//! # What is approximated here
//!
//! - A node holding zero shows its memory at node-relative offsets. Once it holds another number it
//!   shows its memory there, keeps node zero's below the cached alias, and shows the memory of the
//!   node holding zero above it; on a two-node board each also shows the other's at its number.
//!   Each node reserves 4 GiB (`next_power_of_two(nodes) x 4 GiB` with two) of inaccessible host
//!   address space (`PROT_NONE` and `MAP_NORESERVE`, or a `VirtualAlloc2` placeholder), by reading
//!   the host code: only Linux was run.
//! - The low 128 KiB of the Hub-special memory view is not private to each processor of the pair.
//! - A 512 MiB bank 0 covers the boot flash's reset address: cached references reach the bank, the
//!   uncached reset fetch reaches the flash through the Hub-special view. That the hardware does
//!   the same is inferred (the PROM's memory test walks that address as RAM).
//! - A bank up to 128 MiB is one run at its 512 MiB stride; a larger one is four runs at 128 MiB.
//! - Empty memory fails every access with a logged bus error (a software expectation: the IRIX
//!   handler and SGI's event catalogue allow it, no capture shows it). The PROM's sizing reads the
//!   directory back door, which answers a fixed pattern for an absent bank, so boots log no access.
//!
//! # The machines
//!
//! One [`Ip27Machine`] serves every product: a Hub node with its memory, boot flash, module
//! controller and processors, and a board that supplies the I/O around it. A product is a
//! [`NodeSpec`] per node plus a `build_io` function handing back a [`BoardIo`]; the profile picks
//! both ([`crate::ip29_machine`] for the Origin 200, [`crate::ip27_machine`] for the desksides).
//!
//! - Every node holds one processor; each node's memory is mapped by the Hubs' remap.
//! - The Hub's per-slice enable and NMI stores only log; their effect on a processor is not established.
//! - The node flash and uncached Hub-special views carry no JIT page generation (`gen_ptr`), so a
//!   jitv2 build compiles no code from them and the PROM runs on the interpreter's path.
//!
//! Ported from hw/mips/sgi/sn-machine.c and hw/mips/sgi/sn0.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::{Ip27Section, MachineConfig, MachineProfile};
use crate::cpu::parts::{self, CpuModule};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::bridge::{Bridge, PIN_WIRING};
use crate::dev::elsc::{Elsc, ElscConfig, ModuleOutput};
use crate::dev::elsc_nvram::ElscNvram;
use crate::dev::heart::{host_clock, Clock};
use crate::dev::hub::{Hub, HubConfig};
use crate::sn::{NodeBoard, UncachedBus, UncachedSpace, RESET_WINDOW};
use crate::dev::i2c::{BusBusy, I2cBus};
use crate::dev::ioc3::{ByteBus, Ioc3, Ioc3Board};
use crate::dev::isp1040::Isp1040;
use crate::dev::node_flash::{self, FlashValues, NodeFlash};
use crate::dev::pcf8584::Pcf8584;
use crate::dev::prom_image;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::CiSerialBackend;
use crate::dev::timekeeper::Timekeeper;
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::cpu::smp::World;
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link, NatControl};
use crate::window_map::SparseWindowMap;
pub use crate::dev::hub::{bank_mappings, BANKS, BANK_STRIDE, SLOT_STRIDE};
use crate::scsi::ScsiBus;
use crate::traits::*;
use crate::xtalk::Upstream;
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use parking_lot::Mutex;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Duration;

/// The ranges the CPU model reports as memory for a node with `banks_mb`: the
/// installed runs, and the boot flash's reset address unless a 512 MiB bank 0 holds it.
pub fn memory_map(banks_mb: [u32; BANKS]) -> crate::memory_map::MemoryMap {
    use crate::memory_map::{MemKind, MemRange, MemoryMap};
    const NAMES: [&str; BANKS] = ["bank 0", "bank 1", "bank 2", "bank 3"];
    let end = RESET_WINDOW + crate::dev::node_flash::SIZE as u64;
    let runs = crate::dev::hub::installed_mappings(banks_mb);
    let mut ranges: Vec<_> = runs.iter().map(|r| MemRange::new(NAMES[(r.base / BANK_STRIDE) as usize], MemKind::Ram, r.base, r.end)).collect();
    if !runs.iter().any(|r| r.base < end && r.end > RESET_WINDOW) {
        ranges.push(MemRange::new("boot flash", MemKind::Rom, RESET_WINDOW, end));
    }
    MemoryMap::new(ranges)
}

/// The Hub-special space of a node: its own memory and boot flash and the
/// named node's, and the two back doors.
pub fn hub_special(memory: Arc<dyn BusDevice>, flash: &Arc<NodeFlash>, ecc: Arc<dyn BusDevice>, directory: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
    const WINDOW: u64 = 256 << 20;
    const BACK_DOOR: u64 = 1 << 30;
    let mut map = SparseWindowMap::default();
    for base in [0, 0x2000_0000] {
        map.map(base, WINDOW, memory.clone());
    }
    for base in [0x1000_0000, 0x3000_0000] {
        map.map(base, WINDOW, flash.window());
    }
    map.map(0x8000_0000, BACK_DOOR, ecc);
    map.map(0xc000_0000, BACK_DOOR, directory);
    Arc::new(map)
}

/// The profile's own `[origin200]`, `[origin2000]` or `[onyx2]` table. Only an
/// IP27 machine is built from it.
pub(crate) fn section(cfg: &MachineConfig) -> &Ip27Section {
    cfg.ip27_section().expect("an IP27 machine reads its product's table")
}

/// The product's name as a sentence has it: "Origin 200", "Origin 2000" or
/// "Onyx2".
pub(crate) fn product(cfg: &MachineConfig) -> &'static str {
    cfg.machine.profile.ip27_name().expect("an IP27 machine is one of three products")
}

/// The name of that table, for the prefix of an error about one of its keys.
pub(crate) fn table(cfg: &MachineConfig) -> &'static str {
    cfg.machine.profile.ip27_table().expect("an IP27 machine reads its product's table")
}

/// The PROM image tried when `prom` is left at the shared default, for all
/// three IP27 products.
pub const IP27_PROM_FILE: &str = "ip27prom.img";

/// The facts about a product the machine reads.
pub struct NodeSpec {
    pub hub: HubConfig,
    /// The processor module; its mode word is the flash block's.
    pub module: &'static CpuModule,
    pub banks_mb: [u32; BANKS],
    /// What the flash configuration block carries for this product and module.
    pub flash: FlashValues,
    /// The file that keeps the node's boot flash between runs; absent means
    /// volatile.
    pub flash_file: Option<PathBuf>,
    /// The record on the node's one-wire part, with the node's own barcode. None
    /// leaves the part unprogrammed.
    pub node_record: Option<BoardRecord<'static>>,
}

/// What the machine asks of the module controller a board wires.
pub trait ModuleController: Send + Sync {
    /// The module resets: the controller's bus side returns to power-on.
    fn reset(&self);
    /// The controller's outputs since the last call, with its time brought
    /// up to date.
    fn take_events(&self) -> Vec<ModuleOutput>;
    /// How long until the controller has an output or action due with no bus access to bring it
    /// up to date: zero if due now, `None` if nothing is pending.
    fn next_due(&self) -> Option<Duration>;
    /// `wake` runs, with no controller lock held, after any access that may
    /// have moved [`Self::next_due`]. Once, when the machine is built.
    fn set_wake(&self, wake: Box<dyn Fn() + Send + Sync>);
}

/// What a board's I/O hands the machine once it is wired to the node.
pub struct BoardIo {
    /// The devices with threads or timers, in start order.
    pub devices: Vec<Arc<dyn Device>>,
    /// Devices the monitor gets commands from.
    pub monitor: Vec<Arc<dyn Device>>,
    /// The SCSI controllers; the first takes disc changes.
    pub scsi: Vec<Arc<Isp1040>>,
    pub ps2: Arc<Ps2Controller>,
    pub timekeeper: Arc<Timekeeper>,
    /// The console, when it serves the CI socket.
    pub console: Option<Arc<CiSerialBackend>>,
    /// The `scratch = true` volume.
    pub scratch: Option<PathBuf>,
    pub nat: Option<crate::sn::Nat>,
    /// The module controller the nodes' I2C controllers reach.
    pub controller: Arc<dyn ModuleController>,
    /// Put the board's devices back to power-on.
    pub reset: Box<dyn Fn() + Send + Sync>,
    /// The window of the graphics board, when one is attached.
    pub display: Option<Arc<dyn GfxDisplay>>,
    /// The status bar's source, to learn the processors' clocks once they exist.
    pub status: Option<Arc<crate::disp::StatusSource>>,
    /// The InfiniteReality pipe, for a test to drive.
    #[cfg(test)]
    pub pipe: Option<Arc<crate::dev::kona::Kona>>,
    /// The Bridge, for a test to drive.
    #[cfg(test)]
    pub bridge: Arc<Bridge>,
}

pub struct Ip27Machine {
    io: BoardIo,
    nodes: Vec<NodeBoard<Hub, NodeFlash>>,
    /// The one timer manager of the machine, as on every other machine.
    timers: Arc<TimerManager>,
    /// Each node's processor, in node order.
    cpus: Vec<Arc<dyn CpuDevice>>,
    /// Brings the module controller up to date when it is next due.
    watch: Arc<ControllerWatch>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci: bool,
    nvram: String,
    /// The product's name, for messages.
    name: &'static str,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
}

/// Brings the module controller up to date when next due. It keeps its own time and a bus access
/// updates it; this serves a guest that leaves the bus idle across a countdown or pulse. One
/// one-shot timer is re-armed after each visit and access, so an idle machine wakes nothing.
struct ControllerWatch {
    controller: Arc<dyn ModuleController>,
    events: Sender<MachineEvent>,
    timers: Weak<TimerManager>,
    /// The armed timer, if any. Held across the arming so two arms cannot
    /// leave two timers.
    timer: Mutex<Option<TimerId>>,
    /// Whether the machine is running its peripherals.
    running: AtomicBool,
    me: Weak<ControllerWatch>,
}

impl ControllerWatch {
    fn new(controller: Arc<dyn ModuleController>, events: Sender<MachineEvent>, timers: &Arc<TimerManager>) -> Arc<Self> {
        let watch = Arc::new_cyclic(|me| Self {
            controller,
            events,
            timers: Arc::downgrade(timers),
            timer: Mutex::new(None),
            running: AtomicBool::new(false),
            me: me.clone(),
        });
        let me = Arc::downgrade(&watch);
        watch.controller.set_wake(Box::new(move || {
            if let Some(watch) = me.upgrade() {
                watch.arm();
            }
        }));
        watch
    }

    fn start(&self) {
        self.running.store(true, Ordering::Release);
        self.arm();
    }

    fn stop(&self) {
        self.running.store(false, Ordering::Release);
        self.arm();
    }

    /// Arm the timer for the controller's next due time, replacing the last.
    fn arm(&self) {
        let Some(timers) = self.timers.upgrade() else { return };
        let mut timer = self.timer.lock();
        if let Some(id) = timer.take() {
            timers.remove(id);
        }
        if !self.running.load(Ordering::Acquire) {
            return;
        }
        let Some(due) = self.controller.next_due() else { return };
        // Never zero: a controller that is due now and stays due after a
        // visit must not spin the timer thread.
        let due = due.max(Duration::from_nanos(1));
        let me = self.me.clone();
        *timer = Some(timers.add_one_shot(due, me, |me| {
            if let Some(watch) = me.upgrade() {
                watch.visit();
            }
            TimerReturn::Delete
        }));
    }

    /// Carry out what the controller has due, and arm for what is next.
    fn visit(&self) {
        for event in self.controller.take_events() {
            controller_output(event, &self.events);
        }
        self.arm();
    }

    #[cfg(test)]
    fn armed(&self) -> Option<TimerId> {
        *self.timer.lock()
    }
}

/// What the controller's outputs ask of the machine: power falling is power loss, rising or a
/// reset pulse a restart. The NMI output has no path to the processor yet and is only logged.
fn controller_output(event: ModuleOutput, machine: &Sender<MachineEvent>) {
    let _ = match event {
        ModuleOutput::Power(false) => machine.send(MachineEvent::PowerOff),
        ModuleOutput::Power(true) | ModuleOutput::Reset => machine.send(MachineEvent::HardReset),
        ModuleOutput::Nmi(level) => {
            crate::dlog!(LogModule::Hub, "IP27: module controller NMI {level} (not delivered)");
            Ok(())
        }
    };
}

impl NodeBoard<Hub, NodeFlash> {
    /// Node `index` of `nodes`: its flash from `prom` and `spec`, the Hub and the
    /// node board's one-wire record, and the uncached bus its processor uses.
    pub fn build(spec: &NodeSpec, index: usize, nodes: usize, prom: &[u8], clock: &Clock, timers: &Arc<TimerManager>) -> Result<Self, String> {
        let image = node_flash::initial_image(prom, &spec.flash)?;
        let flash = match &spec.flash_file {
            Some(path) => NodeFlash::with_file(image, path).map_err(|e| format!("node {index} flash {}: {e}", path.display()))?,
            None => NodeFlash::new(image)?,
        };
        let hub = Hub::new(HubConfig { banks_mb: spec.banks_mb, nodes, ..spec.hub }, clock.clone(), Some(flash.chip()))?;
        hub.pi.set_timer_manager(timers.clone());
        // The Hub's one-wire master reads the node board's record, whose serial is the node's
        // identity: the other node's PROM reads it as the remote NIC.
        let mut onewire = w1::Bus::new();
        let barcode = format!("EMUN{index:02}");
        let serial = 0x5145_4d55_4e00 | index as u64;
        match &spec.node_record {
            Some(r) => onewire.add(Box::new(board_record_part(&BoardRecord { barcode: &barcode, ..*r }, serial))),
            None => onewire.add(Box::new(w1::Ds2505::new(0x0b, serial, &[]))),
        }
        hub.md.set_onewire(onewire);
        hub.decode.set_space(
            UncachedSpace::HubSpecial,
            hub_special(hub.decode.own_memory(), &flash, hub.md.ecc_bus(), hub.md.directory_bus()),
        );
        hub.decode.set_space(UncachedSpace::Io, hub.io_space());
        // Uncached memory, attribute 3, is the node's decode, as cached is.
        let physical: Arc<dyn BusDevice> = hub.decode.physical().clone();
        let bus = Arc::new(UncachedBus::new([
            hub.cpu_view(UncachedSpace::HubSpecial),
            hub.processor_io(hub.cpu_view(UncachedSpace::Io)),
            Arc::new(crate::sn::FetchOp(hub.decode.physical().clone())),
            physical,
        ]));
        Ok(Self { chip: hub, flash, bus, first_cpu: spec.hub.first_cpu })
    }

    /// The populated processor's enable bit. A Hub resets with both enables clear and the stock
    /// PROM never sets its own, so the board does what the module's strap does.
    pub fn enable_cpu(&self) {
        self.chip.enable_cpu(0);
    }
}

impl Ip27Machine {
    /// A machine of `cfg`'s product: `node_specs` gives each node's facts and `build_io` builds the
    /// board's I/O around them.
    pub fn new(
        cfg: &MachineConfig,
        node_specs: fn(&MachineConfig) -> Result<Vec<NodeSpec>, String>,
        build_io: fn(&MachineConfig, &[NodeBoard<Hub, NodeFlash>], &Clock, &Arc<TimerManager>) -> Result<BoardIo, String>,
    ) -> Result<Self, String> {
        cfg.validate()?;
        let specs = node_specs(cfg)?;
        let prom = prom_image::resolve(cfg.machine.profile, &cfg.prom, node_flash::SIZE)
            .map_err(|e| format!("{} {e}", product(cfg)))?
            .bytes;
        let clock = host_clock();
        let timers = Arc::new(TimerManager::new());
        let mut nodes = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            let node = NodeBoard::<Hub, NodeFlash>::build(spec, i, specs.len(), &prom, &clock, &timers)?;
            node.enable_cpu();
            nodes.push(node);
        }
        let io = build_io(cfg, &nodes, &clock, &timers)?;
        let maps: Vec<_> = specs.iter().map(|s| memory_map(s.banks_mb)).collect();
        let memory = Arc::new(crate::sn::machine_memory(&maps, crate::dev::hub::NODE_SHIFT));

        // The Hubs are the memory controllers: a node number or alias store
        // remaps both nodes' memory with every other processor parked.
        if let [a, b] = &nodes[..] {
            a.chip.connect_peer(&b.chip);
            b.chip.connect_peer(&a.chip);
        }
        let world = Arc::new(World::new());
        for n in &nodes {
            n.chip.decode.set_world(world.clone());
        }
        let (tx, rx) = channel();
        let mut cpus = Vec::new();
        for (node, spec) in nodes.iter().zip(&specs) {
            let cpu = Self::build_cpu(cfg, spec, node, &memory, &timers, specs.len() > 1)?;
            // The Hub publishes its lines straight into the processor's
            // interrupt word, which `Drop` lets go of before the processor goes.
            node.chip.pi.set_cpu_interrupts(0, cpu.interrupts_ptr());
            node.chip.set_event_sender(tx.clone());
            node.chip.register_locks();
            cpu.register_locks();
            cpus.push(cpu);
        }
        world.set_slots(cpus.iter().map(|c| c.park_slot()).collect());
        let watch = ControllerWatch::new(io.controller.clone(), tx, &timers);
        if let Some(status) = &io.status {
            let cpu = &cpus[0];
            status.set_cpu_cycles(cpu.cycles_ptr());
            status.set_count_hz(cpu.count_hz_atomic());
            for c in &cpus[1..] {
                status.add_cpu_stats(c.cycles_ptr(), c.fasttick_count());
            }
        }

        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(cpus.clone())));
        for node in &nodes {
            monitor.register_device(node.chip.decode.physical().clone());
            monitor.register_device(node.chip.clone());
        }
        for d in &io.monitor {
            monitor.register_device(d.clone());
        }

        eprintln!(
            "iris: {}, {} node{}, {}, {} MB each",
            product(cfg),
            nodes.len(),
            if nodes.len() == 1 { "" } else { "s" },
            specs[0].module.part.label(),
            specs[0].banks_mb.iter().sum::<u32>()
        );
        Ok(Self {
            io,
            nodes,
            timers,
            cpus,
            watch,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci: cfg.ci,
            nvram: cfg.nvram.clone(),
            name: product(cfg),
            events: Some(rx),
            slot: None,
            lifecycle: GuestState::Idle,
        })
    }

    /// Node `n`'s processor, for tests.
    #[cfg(test)]
    pub(crate) fn cpu_at(&self, n: usize) -> &Arc<dyn CpuDevice> {
        &self.cpus[n]
    }

    /// The InfiniteReality pipe, for tests.
    #[cfg(test)]
    pub(crate) fn pipe(&self) -> Option<&Arc<crate::dev::kona::Kona>> {
        self.io.pipe.as_ref()
    }

    /// The board's Bridge, for tests.
    #[cfg(test)]
    pub(crate) fn bridge(&self) -> &Arc<Bridge> {
        &self.io.bridge
    }

    #[cfg(test)]
    pub(crate) fn node_at(&self, n: usize) -> &NodeBoard<Hub, NodeFlash> {
        &self.nodes[n]
    }

    /// `smp` when the board carries more than one node: the processors then
    /// share memory, and a store-conditional has to be a compare-and-swap.
    fn build_cpu(
        cfg: &MachineConfig,
        spec: &NodeSpec,
        node: &NodeBoard<Hub, NodeFlash>,
        memory: &Arc<crate::memory_map::MemoryMap>,
        timers: &Arc<TimerManager>,
        smp: bool,
    ) -> Result<Arc<dyn CpuDevice>, String> {
        let physical = node.chip.decode.physical();
        let cached: Arc<dyn BusDevice> = physical.clone();
        let (bus, memory) = (node.bus.clone(), memory.clone());
        let module = spec.module;
        // The R12000 module's 4 MB secondary cache has its own type; the shared table has only the
        // Octane's 2 MB, which the PROM's tag test refuses.
        let cpu = if module.part.l2_kb == 4096 {
            Self::cpu::<crate::cpu::mips_cache_shadow::R12000Module4MShadowCache>(cfg, module, bus, cached, memory, timers, node.first_cpu, smp)
        } else {
            crate::with_cpu!(module.part, true, false, C => Self::cpu::<C>(cfg, module, bus, cached, memory, timers, node.first_cpu, smp))
        };
        physical.connect_cpu(&*cpu, false)?;
        #[cfg(feature = "jitv2")]
        crate::cpu::jitv2::jitv2::attach_pool(&cpu, cfg.jitv2_pool_size(), cfg.jitv2.compile, physical.clone(), None);
        Ok(cpu)
    }

    #[allow(clippy::too_many_arguments)]
    fn cpu<C>(
        cfg: &MachineConfig,
        module: &CpuModule,
        bus: Arc<UncachedBus>,
        cached: Arc<dyn BusDevice>,
        memory: Arc<crate::memory_map::MemoryMap>,
        timers: &Arc<TimerManager>,
        cpu_id: usize,
        smp: bool,
    ) -> Arc<dyn CpuDevice>
    where
        C: crate::cpu::mips_cache_v2::CpuModel + MipsCache + From<Arc<dyn BusDevice>> + 'static,
        MipsCpu<MipsTlb, C>: CpuDevice,
    {
        // `[clock] fixed_mhz` overrides the module's Count rate.
        let count_hz = cfg.clock.count_hz(module.part).max(1);
        let exec = crate::cpu::build::build_cpu::<C>(bus, cached, memory, module, timers, count_hz, cpu_id, smp);
        Arc::new(MipsCpu::new(*exec))
    }
}

impl Workstation for Ip27Machine {
    fn start(&mut self) {
        self.lifecycle = GuestState::Starting;
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        if !self.ci {
            self.cpu_start();
        }
    }
    fn stop(&mut self) {
        for cpu in &self.cpus {
            cpu.clone().as_device().stop();
        }
        self.watch.stop();
        for d in self.io.devices.iter().rev() {
            d.stop();
        }
        for node in &self.nodes {
            node.chip.stop();
        }
        if let Err(e) = self.io.timekeeper.flush_nvram() {
            crate::dlog!(LogModule::Hub, "{} NVRAM: {e}", self.name);
        }
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        for (node, cpu) in self.nodes.iter().zip(&self.cpus) {
            cpu.power_on();
            node.chip.power_on();
            node.enable_cpu();
            node.flash.power_on();
        }
        self.io.controller.reset();
        (self.io.reset)();
        self.io.timekeeper.power_on();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        for node in &self.nodes {
            node.chip.start();
        }
        for d in &self.io.devices {
            d.start();
        }
        self.watch.start();
    }
    fn cpu_start(&self) {
        for cpu in &self.cpus {
            cpu.clone().as_device().start();
        }
    }
    fn cpu_is_running(&self) -> bool {
        self.cpus[0].clone().as_device().is_running()
    }
    fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut Self as *mut dyn Workstation;
        self.slot.get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr)))).clone()
    }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self.events.take().expect("system controller registered twice");
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
    fn save_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the Hubs and node memory have no saved state", self.name))
    }
    fn load_snapshot(&mut self, _name: &str) -> Result<(), String> {
        Err(format!("snapshots of the {} are not supported yet: the Hubs and node memory have no saved state", self.name))
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
        self.cpus[0].clone().debug_adapter()
    }
    fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.cpus[0].fasttick_count()
    }
    fn get_ci_serial(&self) -> Option<Arc<CiSerialBackend>> {
        self.io.console.clone()
    }
    fn inject_serial_console(&self, bytes: &[u8]) {
        if let Some(console) = &self.io.console {
            console.push_host(bytes);
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        if path.is_some_and(|p| p != self.nvram) {
            return Err(format!("{} NVRAM save uses the configured path", self.name));
        }
        self.io.timekeeper.flush_nvram()?;
        Ok(self.nvram.clone())
    }
    fn scratch_path(&self) -> Option<&Path> {
        self.io.scratch.as_deref()
    }
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        self.io.scsi.first().map(|c| c.clone() as Arc<dyn DiscChanger>)
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.io.scsi.iter().map(|c| c.bus().pending_chd_sync_count()).sum()
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.io.scsi.iter().try_for_each(|c| c.bus().flush())
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        let buses: Vec<&ScsiBus> = self.io.scsi.iter().map(|c| c.bus()).collect();
        ScsiBus::sync_chd_disks(&buses, only, progress, cancel)
    }
    fn net_frames(&self) -> Option<Arc<AtomicU64>> {
        self.io.nat.as_ref().map(|n| n.frames.clone())
    }
    fn nat_control(&self) -> Option<Arc<NatControl>> {
        self.io.nat.as_ref().map(|n| n.control.clone())
    }
    fn nat_expected(&self) -> Option<(Ipv4Addr, Ipv4Addr, Ipv4Addr)> {
        self.io.nat.as_ref().map(|n| n.expected)
    }
}

impl Drop for Ip27Machine {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        // Nothing publishes into a processor's interrupt word once the
        // processors go: another Hub can still reach this one's processor.
        for node in &self.nodes {
            node.chip.pi.clear_cpu_interrupts();
        }
        // The compile pool's requests point into the memory freed below.
        #[cfg(feature = "jitv2")]
        for cpu in &self.cpus {
            crate::cpu::jitv2::Jitv2::stop_for_drop(&cpu.jitv2(), || cpu.clone().as_device().stop());
        }
        crate::workstation::flush_on_drop(self);
        self.monitor.shutdown();
        // The bus holds the devices and the Hub and Bridge hold its memory, which owns the bus: let
        // go, or none is freed. Every table goes before any node, as each points into the other's
        // banks.
        for node in &self.nodes {
            node.chip.decode.physical().release_devices();
        }
    }
}

/// Load the BaseIO card's boot flash with the I/O PROM the machine's table names, else the
/// one built into iris (a build without `embedded-proms` leaves the flash empty).
pub(crate) fn load_io_prom(bridge: &Bridge, cfg: &MachineConfig) -> Result<(), String> {
    let table = table(cfg);
    let configured = section(cfg).io_prom.as_deref();
    // "(embedded)" and an unset `io_prom` both mean the built-in PROM; a named file must exist.
    let named = configured.filter(|p| !crate::dev::prom_image::is_embedded_keyword(p));
    let file = match named {
        Some(path) => std::fs::read(path).map_err(|e| format!("{table}.io_prom {path}: {e}"))?,
        // Unset, and not built in: leave the flash empty, as before.
        None => match crate::dev::embedded_prom::get(crate::dev::embedded_prom::EmbeddedProm::Io6) {
            Ok(built_in) => built_in,
            Err(_) if configured.is_none() => return Ok(()),
            Err(e) => return Err(format!("{table}.io_prom: {e}")),
        },
    };
    let flash = crate::dev::baseio_flash::BaseioFlash::new(&file)
        .map_err(|e| format!("{table}.io_prom {}: {e}", named.unwrap_or("(embedded)")))?;
    bridge.attach_flash(0, Arc::new(flash), crate::dev::baseio_flash::SIZE as u32);
    Ok(())
}

/// The PCI devices both IP27 I/O boards carry behind their Bridge: two ISP1040s and an IOC3.
/// `ioc3_board` gives its identification, ports and SuperIO interrupt input; the devices set the
/// status bar's activity bits in `heartbeat`.
pub(crate) fn pci_devices(
    timers: &Arc<TimerManager>,
    cfg: &MachineConfig,
    bridge: Arc<Bridge>,
    ioc3_board: &'static Ioc3Board,
    controller: Arc<dyn ModuleController>,
    heartbeat: Arc<AtomicU64>,
) -> Result<BoardIo, String> {
    let (scsi_name, name) = (table(cfg), product(cfg));
    let (bus_a, bus_b) = (Isp1040::new_bus(), Isp1040::new_bus());
    let scratch = crate::scsi::attach_without_daynaport(cfg, scsi_name, name, &|c| match c {
        0 => Ok(&bus_a),
        1 => Ok(&bus_b),
        _ => Err(format!("controller {c} ({name} has controllers 0 and 1)")),
    })?;
    let scsi = vec![Arc::new(Isp1040::new_at(0, bus_a)), Arc::new(Isp1040::new_at(1, bus_b))];
    for s in &scsi {
        s.set_heartbeat(heartbeat.clone());
    }
    let pci = bridge.bus();
    let slot = |device| crate::pci::DevFn { device, function: 0 };
    pci.attach(slot(0), scsi[0].clone());
    pci.attach(slot(1), scsi[1].clone());
    // The Fuel's M48T35 Timekeeper is the same part.
    let timekeeper = Arc::new(Timekeeper::with_nvram_file(Path::new(&cfg.nvram))?);
    let ioc3 = Arc::new(Ioc3::new(ioc3_board, cfg.network().mac, ByteBus::M48t35(timekeeper.clone())));
    ioc3.set_heartbeat(heartbeat);
    let eth = ioc3.eth().expect("an IP27 I/O board's IOC3 has Ethernet").clone();
    pci.attach(slot(2), ioc3.clone());
    ioc3.connect_superio(pci);
    ioc3.set_timer_manager(timers.clone());

    let console = cfg.ci.then(|| Arc::new(CiSerialBackend::new()));
    let (port_a, port_b) = crate::dev::serial_host::console_backends(cfg, name, console.as_ref());
    ioc3.set_serial_backend(0, port_a);
    if let Some(b) = port_b {
        ioc3.set_serial_backend(1, b);
    }

    let mut nat = None;
    let net = GatewayConfig::from_network(&cfg.network());
    if let Some(link) = Link::with_gateway(net.clone())? {
        nat = Some(crate::sn::Nat { control: link.nat(), expected: (net.client_ip, net.gateway_ip, net.netmask), frames: link.frames() });
        eth.attach(link);
    }
    bridge.register_locks();
    ioc3.register_locks();
    for s in &scsi {
        s.register_locks();
    }

    let mut monitor: Vec<Arc<dyn Device>> = scsi.iter().map(|s| s.clone() as Arc<dyn Device>).collect();
    monitor.push(ioc3.ps2());
    let devices: Vec<Arc<dyn Device>> = vec![ioc3.clone(), scsi[0].clone(), scsi[1].clone()];
    let ps2 = ioc3.ps2();
    #[cfg(test)]
    let test_bridge = bridge.clone();
    let reset_scsi = scsi.clone();
    let reset: Box<dyn Fn() + Send + Sync> = Box::new(move || {
        bridge.reset();
        bridge.reset_pci();
        ioc3.power_on();
        for s in &reset_scsi {
            s.reset();
        }
    });
    Ok(BoardIo {
        devices,
        monitor,
        scsi,
        ps2,
        timekeeper,
        console,
        scratch,
        nat,
        controller,
        reset,
        display: None,
        status: None,
        #[cfg(test)]
        pipe: None,
        #[cfg(test)]
        bridge: test_bridge,
    })
}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
