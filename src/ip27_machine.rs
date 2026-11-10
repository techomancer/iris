//! The SN0 machines (IP27 software platform): a Hub node with its memory,
//! boot flash, module controller and processors, and a board that supplies
//! the I/O around it.
//!
//! [`Ip27Machine`] is the machine every SN0 product shares. A product is
//! data, a [`BoardSpec`] of the facts the machine reads (the Hub, the
//! processor module, the memory and the flash values), plus the one thing
//! that differs in behaviour, how its I/O is wired to the node
//! ([`Ip27Board::wire`], which hands back a [`BoardIo`]). The one board here
//! is the Origin 200 (IP29), selected with
//! `[machine] profile = "origin200_ip29"`. One node and one CPU are built; the
//! PROM, the controller store and the timekeeper's NVRAM come from the shared
//! `prom` and `nvram` keys and `[origin200]`.
//!
//! # What is approximated here
//!
//! - One node, one processor, node number zero. Cached accesses ignore the
//!   node number (see [`crate::ip27`]).
//! - The Hub's per-slice enable, NMI and cached-alias hooks only log: what a
//!   store to those registers does to a processor is not established.
//! - The Bridge reaches the Hub through [`Hub::inbound`] for DMA and
//!   [`Hub::interrupt_target`] for interrupts, with the Hub addressed as
//!   widget 0xa. Both are the smallest adapter the Bridge needs and are meant
//!   to be replaced by the shared host interface.
//! - The module controller's configuration store comes from
//!   `ElscNvram::origin200`; the machine never looks inside it.
//! - The Bridge answers at fifteen widget numbers (every one but the Hub's
//!   own); which numbers real hardware answers is not established.
//! - The Origin 200 node keeps Hub revision 3 and the IP29 board record
//!   030-1244-001 revision H whatever the processor module. Real 225 and 270
//!   MHz machines report Hub revision 5 and board 030-1389-001.
//! - The node flash and the uncached Hub-special views carry no JIT page
//!   generation (`gen_ptr`), so a jitv2 build compiles no code from them; the
//!   PROM runs from them on the interpreter's path.
//!
//! Ported from hw/mips/sgi/sn-machine.c and hw/mips/sgi/sn0.c in
//! jamesbraid/qemu-sgi-origami.

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::parts::{self, Part};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::bridge::Bridge;
use crate::dev::elsc::{Elsc, ElscConfig, ElscEvent};
use crate::dev::elsc_nvram::ElscNvram;
use crate::dev::heart::{host_clock, Clock};
use crate::dev::hub::{Hub, HubConfig, HubHooks, Space};
use crate::dev::hub_pi;
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
use crate::ip27::{self, Ip27Bus, NodeMemory};
use crate::irq::InterruptController;
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link, NatControl};
use crate::scsi::ScsiBus;
use crate::traits::{BusDevice, Device, MachineEvent, Resettable, Signal, BUS_ERR};
use crate::xtalk::{Upstream, XtalkHost};
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use parking_lot::Mutex;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{atomic::AtomicU64, Arc, Weak};

/// The PROM image tried when `prom` is left at the shared default.
pub const ORIGIN200_PROM_FILE: &str = "ip27prom.img";

/// How often the module controller is brought up to date when the guest
/// leaves its bus idle. The controller keeps its own time, so a late visit
/// loses nothing.
const CONTROLLER_PERIOD: std::time::Duration = std::time::Duration::from_millis(10);

/// The Hub's own widget number: a small window there is not the Bridge's.
const WIDGET_HUB: u8 = 1;

/// A processor module the machine fits.
#[derive(Clone, Copy, Debug)]
pub struct CpuModule {
    pub part: &'static Part,
    /// The mode word the flash configuration block carries.
    pub mode_word: u32,
}

impl CpuModule {
    /// The pipeline clock in hertz.
    pub fn clock_hz(&self) -> u64 {
        u64::from(self.part.mhz) * 1_000_000
    }

    /// The PRId the PROM and IRIX read.
    pub fn prid(&self) -> u32 {
        self.part.prid()
    }

    /// CP0 Count runs at half the pipeline clock.
    pub fn count_hz(&self) -> u64 {
        self.part.count_hz()
    }

    /// Config at cold reset: the mode word's bits 24:0 under the hardwired
    /// 32 KiB primary-cache fields.
    pub fn config(&self) -> u32 {
        0x6c00_0000 | (self.mode_word & 0x01ff_ffff)
    }
}

/// The processors an Origin 200 takes, the default first: the 180 MHz R10000
/// and the 270 MHz R12000.
pub const CPUS: &[&Part] = &[&parts::R10000_180, &parts::R12000_270];

/// The processor modules an Origin 200 shipped with, by chip: the 180 MHz
/// R10000 (revision 2.6, 1 MiB secondary cache) and the 270 MHz R12000
/// (revision 2.3, 4 MiB).
///
/// The mode word is the module's: its Config word says 1 MiB and 64-byte
/// secondary lines for the R10000 and 4 MiB for the R12000. The R10000 module
/// runs on the Octane's R10000 cache type (1 MiB, 128-byte lines, which the
/// PROM accepts); the R12000 module has its own 4 MiB type, since the
/// Octane's 2 MiB fails the PROM's secondary-cache tag test. The line size of
/// both is the Octane part's, not the module's.
pub const MODULES: [CpuModule; 2] = [
    CpuModule { part: &parts::R10000_180, mode_word: 0x1211_a785 },
    CpuModule { part: &parts::R12000_270, mode_word: 0x1213_ab85 },
];

pub fn cpu_module(part: &Part) -> Option<&'static CpuModule> {
    MODULES.iter().find(|m| m.part == part)
}

/// The node every SN0 board plugs into.
pub struct Node {
    pub hub: Arc<Hub>,
    pub memory: NodeMemory,
    pub flash: Arc<NodeFlash>,
    pub clock: Clock,
    pub timers: Arc<TimerManager>,
}

impl Node {
    /// The populated processor's enable bit. A Hub comes out of reset with
    /// both enables clear, and the stock PROM never sets its own, so the
    /// board does what the processor module's strap does on the machine.
    fn enable_cpu(&self) {
        self.hub.write64(hub_pi::reg::CPU_ENABLE_A, 1);
    }
}

/// The facts about a product the machine reads.
pub struct BoardSpec {
    pub hub: HubConfig,
    pub module: CpuModule,
    pub banks_mb: [u32; ip27::BANKS],
    /// What the flash configuration block carries for this product and module.
    pub flash: FlashValues,
    /// The file that keeps the node's boot flash between runs; absent means
    /// volatile.
    pub flash_file: Option<PathBuf>,
    /// The record on the node's one-wire part. None leaves the part
    /// unprogrammed, which IRIX reports as missing page data.
    pub node_record: Option<NodeRecord>,
}

/// What a node board's one-wire part says about the board.
#[derive(Clone, Copy, Debug)]
pub struct NodeRecord {
    pub part: &'static str,
    pub revision: &'static str,
    pub name: &'static str,
}

impl NodeRecord {
    /// The 180 MHz one-processor module of an Origin 200, as a published sale
    /// listing shows it. No inventory gives the 270 MHz R12000 module's
    /// record, so that module has none.
    pub const PIMM_1XT5: Self = Self { part: "013-1895-001", revision: "C", name: "PIMM_1XT5_1MB" };
    /// The Origin 2000 and Onyx2 node board.
    pub const IP27: Self = Self { part: "030-0733-003", revision: "G", name: "IP27" };
}

/// What the machine asks of the module controller a board wires.
pub trait ModuleController: Send + Sync {
    /// The module resets: the controller's bus side returns to power-on.
    fn reset(&self);
    /// The controller's outputs since the last call, with its time brought
    /// up to date.
    fn take_events(&self) -> Vec<ElscEvent>;
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
    pub nat: Option<Nat>,
    /// The module controller the nodes' I2C controllers reach.
    pub controller: Arc<dyn ModuleController>,
    /// Put the board's devices back to power-on.
    pub reset: Box<dyn Fn() + Send + Sync>,
    /// The Bridge, for a test to drive.
    #[cfg(test)]
    pub bridge: Arc<Bridge>,
}

/// The Ethernet's NAT gateway, as the front ends read it.
pub struct Nat {
    pub control: Arc<NatControl>,
    pub expected: (Ipv4Addr, Ipv4Addr, Ipv4Addr),
    pub frames: Arc<AtomicU64>,
}

/// A product. Only wiring differs in behaviour between SN0 boards; everything
/// else about one is the [`BoardSpec`].
pub trait Ip27Board: 'static {
    const NAME: &'static str;
    /// The nodes the board carries, in slot order.
    fn specs(cfg: &MachineConfig) -> Result<Vec<BoardSpec>, String>;
    /// Wire the board's I/O to the nodes, and the nodes to each other.
    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String>;
}

pub type Origin200Machine = Ip27Machine<Origin200>;

/// A node with its processor and module controller.
struct Unit {
    node: Node,
    cpu: Arc<dyn CpuDevice>,
}

pub struct Ip27Machine<B: Ip27Board> {
    io: BoardIo,
    _board: std::marker::PhantomData<fn() -> B>,
    units: Vec<Unit>,
    controller_events: Sender<MachineEvent>,
    /// The recurring timer that brings the controller up to date.
    controller_timer: Option<TimerId>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci: bool,
    nvram: String,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
}

/// The CPU's three Hub lines are hardware interrupts 0 to 2, Cause IP2 to IP4.
struct CpuIrq(Weak<dyn CpuDevice>);
impl InterruptController for CpuIrq {
    fn set_line(&self, line: u32, level: bool) {
        if line < 3 {
            if let Some(cpu) = self.0.upgrade() {
                cpu.as_device().signal(Signal::Interrupt(line + 2, level));
            }
        }
    }
}

/// What the Hub asks of the machine around it.
struct Hooks {
    name: &'static str,
    cpu: Weak<dyn CpuDevice>,
    events: Sender<MachineEvent>,
    /// Where a cached reference naming node zero goes, as the node's number
    /// and cached alias set it.
    zero: Arc<ip27::NodeZero>,
    pi: Weak<crate::dev::hub_pi::HubPi>,
}
impl Hooks {
    fn route(&self, number: u16) {
        let calias = self.pi.upgrade().map_or(0, |pi| pi.calias_bytes());
        self.zero.update(number, calias);
    }
}
impl HubHooks for Hooks {
    fn cpu_enable(&self, slice: usize, enabled: bool) {
        crate::dlog!(LogModule::Hub, "{}: PI_CPU_ENABLE {slice} = {enabled} (no effect on the processor)", self.name);
    }
    fn nmi(&self, slice: usize) {
        crate::dlog!(LogModule::Hub, "{}: NMI to slice {slice} (not delivered)", self.name);
    }
    fn soft_reset(&self, _slice: usize) {
        if let Some(cpu) = self.cpu.upgrade() {
            cpu.as_device().signal(Signal::Reset(true));
        }
    }
    fn cached_alias_changed(&self, bytes: u64) {
        let number = self.zero.number();
        self.zero.update(number, bytes);
    }
    fn node_number_changed(&self, number: u16) {
        self.route(number);
    }
    fn machine_restart(&self) {
        let _ = self.events.send(MachineEvent::HardReset);
    }
}

/// What the module controller's outputs ask of the machine: its power output
/// falling is the module losing power, rising or a reset pulse a restart.
/// The NMI output has no path to the processor yet and is only logged.
fn controller_output(event: ElscEvent, machine: &Sender<MachineEvent>) {
    let _ = match event {
        ElscEvent::Power(false) => machine.send(MachineEvent::PowerOff),
        ElscEvent::Power(true) | ElscEvent::Reset => machine.send(MachineEvent::HardReset),
        ElscEvent::Nmi(level) => {
            crate::dlog!(LogModule::Hub, "IP27: module controller NMI {level} (not delivered)");
            Ok(())
        }
    };
}

/// Accesses to a space the model does not carry stay visible in the log.
struct Unmodelled(&'static str);
impl BusDevice for Unmodelled {
    fn read32(&self, a: u64) -> crate::traits::BusRead32 {
        crate::dlog!(LogModule::Hub, "IP27: read of unmodelled {} {a:#x}", self.0);
        crate::traits::BusRead32::err()
    }
    fn read64(&self, a: u64) -> crate::traits::BusRead64 {
        crate::dlog!(LogModule::Hub, "IP27: read of unmodelled {} {a:#x}", self.0);
        crate::traits::BusRead64::err()
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        crate::dlog!(LogModule::Hub, "IP27: write of {v:#x} to unmodelled {} {a:#x}", self.0);
        BUS_ERR
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        crate::dlog!(LogModule::Hub, "IP27: write of {v:#x} to unmodelled {} {a:#x}", self.0);
        BUS_ERR
    }
}

impl<B: Ip27Board> Ip27Machine<B> {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        cfg.validate()?;
        let specs = B::specs(cfg)?;
        let prom = prom_image::load(Path::new(&cfg.machine.profile.prom_path(&cfg.prom)), node_flash::SIZE)
            .map_err(|e| format!("{} {e}", B::NAME))?;
        let mut parts = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            let image = node_flash::initial_image(&prom, &spec.flash)?;
            let flash = match &spec.flash_file {
                Some(path) => NodeFlash::with_file(image, path).map_err(|e| format!("node {i} flash {}: {e}", path.display()))?,
                None => NodeFlash::new(image)?,
            };
            let clock = host_clock();
            let timers = Arc::new(TimerManager::new());
            let memory = NodeMemory::new(spec.banks_mb)?;
            memory.map_reset_view(&flash);
            let hub = Hub::new(spec.hub, clock.clone());
            hub.pi().set_timer_manager(timers.clone());
            // The Hub's one-wire master reads the node board's record, whose
            // serial is the node's identity: the other node's PROM reads it
            // as the remote NIC.
            let mut onewire = w1::Bus::new();
            let barcode = format!("EMUN{i:02}");
            let serial = 0x5145_4d55_4e00 | i as u64;
            match &spec.node_record {
                Some(r) => {
                    let record = BoardRecord { barcode: &barcode, part: r.part, revision: r.revision, name: r.name };
                    onewire.add(Box::new(board_record_part(&record, serial)));
                }
                None => onewire.add(Box::new(w1::Ds2505::new(0x0b, serial, &[]))),
            }
            hub.md().set_onewire(onewire);
            hub.set_space(Space::Memory, memory.home());
            hub.set_space(
                Space::HubSpecial,
                ip27::hub_special(memory.home(), &flash, memory.back_door(), memory.back_door()),
            );
            hub.set_space(Space::Io, hub.io_space());
            hub.set_space(Space::Special, Arc::new(Unmodelled("memory-special space")));
            let node = Node { hub, memory, flash, clock: clock.clone(), timers: timers.clone() };
            node.enable_cpu();

            parts.push(node);
        }

        let nodes: Vec<&Node> = parts.iter().collect();
        let io = B::wire(&nodes, cfg)?;

        let (tx, rx) = channel();
        let mut units = Vec::new();
        for (node, spec) in parts.into_iter().zip(&specs) {
            let cpu = Self::build_cpu(cfg, &spec.module, &node, spec.hub.cpu_base, specs.len() > 1)?;
            node.hub.pi().set_interrupt_controller(Arc::new(CpuIrq(Arc::downgrade(&cpu))));
            let zero = node.memory.node_zero();
            node.hub.set_hooks(Arc::new(Hooks {
                name: B::NAME,
                cpu: Arc::downgrade(&cpu),
                events: tx.clone(),
                zero,
                pi: Arc::downgrade(node.hub.pi()),
            }));
            node.hub.register_locks();
            cpu.register_locks();
            units.push(Unit { node, cpu });
        }

        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(units.iter().map(|u| u.cpu.clone()).collect())));
        for u in &units {
            monitor.register_device(u.node.memory.physical().clone());
            monitor.register_device(u.node.hub.clone());
        }
        for d in &io.monitor {
            monitor.register_device(d.clone());
        }

        eprintln!(
            "iris: {}, {} node{}, {}, {} MB each",
            B::NAME,
            units.len(),
            if units.len() == 1 { "" } else { "s" },
            specs[0].module.part.label(),
            units[0].node.memory.banks_mb().iter().sum::<u32>()
        );
        Ok(Self {
            io,
            _board: std::marker::PhantomData,
            units,
            controller_events: tx,
            controller_timer: None,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci: cfg.ci,
            nvram: cfg.nvram.clone(),
            events: Some(rx),
            slot: None,
            lifecycle: GuestState::Idle,
        })
    }

    /// The first node, the one the front ends and the monitor address.
    pub(crate) fn node(&self) -> &Node {
        self.node_at(0)
    }

    /// Node `n`'s processor, for tests.
    #[cfg(test)]
    pub(crate) fn cpu_at(&self, n: usize) -> &Arc<dyn CpuDevice> {
        &self.units[n].cpu
    }

    /// The board's Bridge, for tests.
    #[cfg(test)]
    pub(crate) fn bridge(&self) -> &Arc<Bridge> {
        &self.io.bridge
    }

    pub(crate) fn node_at(&self, n: usize) -> &Node {
        &self.units[n].node
    }

    /// `smp` when the board carries more than one node: the processors then
    /// share memory, and a store-conditional has to be a compare-and-swap.
    fn build_cpu(cfg: &MachineConfig, module: &CpuModule, node: &Node, cpu_id: usize, smp: bool) -> Result<Arc<dyn CpuDevice>, String> {
        let bus = Arc::new(Ip27Bus::new(
            node.hub.cpu_special(),
            node.hub.processor_io(node.hub.cpu_io()),
            node.hub.claim(Space::Special),
            node.hub.cpu_memory(),
        ));
        let phys: Arc<dyn BusDevice> = node.memory.physical().clone();
        let memory = Arc::new(node.memory.memory_map());
        let count_hz = cfg.clock.count_hz(module.part).max(1);
        // The R12000 module's 4 MB secondary cache has a type of its own: the
        // shared table has only the Octane's 2 MB, which the PROM's tag test
        // refuses.
        let cpu = if module.part.l2_kb == 4096 {
            Self::cpu::<crate::cpu::mips_cache_shadow::R12000Module4MShadowCache>(module, bus, phys, memory, &node.timers, count_hz, cpu_id, smp)
        } else {
            crate::with_cpu!(module.part, true, false, C => Self::cpu::<C>(module, bus, phys, memory, &node.timers, count_hz, cpu_id, smp))
        };
        // The shadow cache and the JIT read RAM through a cached view of the
        // ppmem window that is the CPU's own.
        if let Some(sp) = node.memory.physical().ppmem_space() {
            let v = sp
                .add_cached_view(&[cpu.ppmem_bitmap_ptr(), cpu.tcache_bitmap_ptr()])
                .map_err(|e| format!("ppmem: a CPU's cached view: {e}"))?;
            unsafe {
                cpu.set_tcache_window(sp.view_base(v));
                cpu.set_tcache_shift(crate::ppmem::window_shift(sp.window_size()));
                #[cfg(feature = "jitv2")]
                {
                    use crate::ppmem::MappedMemory;
                    cpu.set_tcache_gen_window(sp.gen_window_base());
                }
            }
        }
        #[cfg(feature = "jitv2")]
        {
            let cpu_device: Arc<dyn Device> = cpu.clone().as_device();
            let jit = cpu.jitv2();
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
                crate::cpu::jitv2::jitv2::start_compiling(&*cpu, &jit, cfg.jitv2.compile, node.memory.physical().clone() as Arc<dyn BusDevice>);
            }
        }
        Ok(cpu)
    }

    /// The controller's timers run when a bus access or this timer touches
    /// it. A guest that leaves the bus idle (an operator typing at the AUX
    /// port, a countdown expiring) still needs them, and its outputs, so a
    /// recurring timer brings it up to date. Nothing runs per instruction.
    fn start_controller_timer(&mut self) {
        self.stop_controller_timer();
        let (controller, events) = (self.io.controller.clone(), self.controller_events.clone());
        let id = self.units[0].node.timers.add_recurring(
            std::time::Instant::now() + CONTROLLER_PERIOD,
            CONTROLLER_PERIOD,
            (),
            move |_| {
                for event in controller.take_events() {
                    controller_output(event, &events);
                }
                TimerReturn::Continue
            },
        );
        self.controller_timer = Some(id);
    }

    fn stop_controller_timer(&mut self) {
        if let Some(id) = self.controller_timer.take() {
            self.units[0].node.timers.remove(id);
        }
    }

    fn cpu<C>(
        module: &CpuModule,
        bus: Arc<Ip27Bus>,
        cached: Arc<dyn BusDevice>,
        memory: Arc<crate::memory_map::MemoryMap>,
        timers: &Arc<TimerManager>,
        count_hz: u64,
        cpu_id: usize,
        smp: bool,
    ) -> Arc<dyn CpuDevice>
    where
        C: crate::cpu::mips_cache_v2::CpuModel + MipsCache + From<Arc<dyn BusDevice>> + 'static,
        MipsCpu<MipsTlb, C>: CpuDevice,
    {
        let cfg = MipsCpuConfig::for_model::<C>(memory).with_prid(module.prid());
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let mut exec: MipsExecutor<MipsTlb, C> = MipsExecutor::new(bus as Arc<dyn BusDevice>, tlb, &cfg);
        exec.cache = C::from(cached);
        exec.core.set_timer_manager(timers.clone());
        exec.core.set_count_hz(count_hz);
        exec.core.cp0_config = module.config();
        exec.core.count_write_samples = true;
        exec.phys_mask = ip27::CPU_PHYS_MASK;
        exec.core.cpu_id = cpu_id as u8;
        exec.core.smp = smp;
        Arc::new(MipsCpu::new(exec))
    }
}

impl<B: Ip27Board> Workstation for Ip27Machine<B> {
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
        }
        self.stop_controller_timer();
        for d in self.io.devices.iter().rev() {
            d.stop();
        }
        for u in &self.units {
            u.node.hub.stop();
        }
        if let Err(e) = self.io.timekeeper.flush_nvram() {
            crate::dlog!(LogModule::Hub, "{} NVRAM: {e}", B::NAME);
        }
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        for u in &self.units {
            u.cpu.power_on();
            u.node.hub.power_on();
            u.node.enable_cpu();
            u.node.flash.power_on();
        }
        self.io.controller.reset();
        (self.io.reset)();
        self.io.timekeeper.power_on();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        for u in &self.units {
            u.node.hub.start();
        }
        for d in &self.io.devices {
            d.start();
        }
        self.start_controller_timer();
    }
    fn cpu_start(&self) {
        for u in &self.units {
            u.cpu.clone().as_device().start();
        }
    }
    fn cpu_is_running(&self) -> bool {
        self.units[0].cpu.clone().as_device().is_running()
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
        None
    }
    fn register_monitor_device(&self, device: Arc<dyn Device>) {
        self.monitor.register_device(device);
    }
    fn get_ps2(&self) -> Arc<Ps2Controller> {
        self.io.ps2.clone()
    }
    fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.node().timers.clone()
    }
    fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> {
        self.units[0].cpu.clone().debug_adapter()
    }
    fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.units[0].cpu.fasttick_count()
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
            return Err(format!("{} NVRAM save uses the configured path", B::NAME));
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

impl<B: Ip27Board> Drop for Ip27Machine<B> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        // The compile pool's requests point into the memory freed below.
        #[cfg(feature = "jitv2")]
        for u in &self.units {
            crate::cpu::jitv2::Jitv2::stop_for_drop(&u.cpu.jitv2(), || u.cpu.clone().as_device().stop());
        }
        crate::workstation::flush_on_drop(self);
        self.monitor.shutdown();
        // The bus holds the devices, and the Hub and Bridge hold its memory,
        // which owns the bus: let go, or none of it is freed.
        for u in &self.units {
            u.node.memory.physical().release_devices();
        }
    }
}

/// The Hub as the Bridge addresses it: the master widget the PROM programs
/// for DMA and interrupts, taking an interrupt packet at any address and
/// naming one of 128 vectors.
const HUB_HOST: XtalkHost = XtalkHost { widget: 0xa, int_packet: 0, int_packet_mask: 0, vectors: 128 };
/// The IP29 System Board's Bridge revision.
const BRIDGE_REVISION: u32 = 4;

/// The Origin 200 (IP29 System Board): the Hub's XIO port straight to a
/// Bridge with two ISP1040s and an IOC3, no Crossbow.
pub struct Origin200;

impl Origin200 {
    /// The Bridge's one-wire line carries the System Board's record. The
    /// barcode and ROM serial are synthetic.
    fn board_record() -> w1::Bus {
        let mut bus = w1::Bus::new();
        let record = BoardRecord { barcode: "EMUN00", part: "030-1244-001", revision: "H", name: "IP29" };
        bus.add(Box::new(board_record_part(&record, 0x5145_4d55_4200)));
        bus
    }
}

/// The Origin 200's module controller, on its own node's I2C controller.
struct Origin200Controller {
    elsc: Arc<Elsc>,
    i2c: Arc<Pcf8584>,
}

impl Origin200Controller {
    fn new(node: &Node, store: Option<&Path>) -> Result<Arc<Self>, String> {
        let store = ElscNvram::origin200(store).map(Arc::new).map_err(|e| format!("origin200.controller_store: {e}"))?;
        let busy = BusBusy::default();
        let elsc = Arc::new(Elsc::new(ElscConfig::default(), store.clone(), node.clock.clone(), busy.clone()));
        let mut bus = I2cBus::new();
        bus.add(elsc.clone());
        bus.add(store);
        let i2c = Arc::new(Pcf8584::new(bus, busy));
        node.hub.attach_i2c(i2c.clone());
        Ok(Arc::new(Self { elsc, i2c }))
    }
}

impl ModuleController for Origin200Controller {
    fn reset(&self) {
        self.i2c.power_on();
        self.elsc.power_on();
    }
    fn take_events(&self) -> Vec<ElscEvent> {
        self.elsc.take_events()
    }
}

impl Ip27Board for Origin200 {
    const NAME: &'static str = "Origin 200";

    fn specs(cfg: &MachineConfig) -> Result<Vec<BoardSpec>, String> {
        let module = *cpu_module(cfg.cpu_part()?).ok_or("an Origin 200 has an R10000 or an R12000")?;
        Ok(vec![BoardSpec {
            // Hub 2.1 on the Origin 200; the node is not in a numbered midplane slot.
            hub: HubConfig { cpus: 1, ni_revision: 3, ..Default::default() },
            module,
            banks_mb: std::array::from_fn(|i| cfg.banks[i]),
            flash: FlashValues {
                cpu_mode_bits: module.mode_word,
                cpu_clock_hz: module.clock_hz(),
                ..FlashValues::ORIGIN200_R10000_180_1M
            },
            flash_file: cfg.origin200.flash.as_ref().map(PathBuf::from),
            node_record: (module.part == &parts::R10000_180).then_some(NodeRecord::PIMM_1XT5),
        }])
    }

    fn wire(nodes: &[&Node], cfg: &MachineConfig) -> Result<BoardIo, String> {
        let node = nodes[0];
        let controller = Origin200Controller::new(node, cfg.origin200.controller_store.as_deref().map(Path::new))?;
        let bridge = Bridge::new(Upstream::single(HUB_HOST, node.hub.inbound(), node.hub.interrupt_target()), &crate::ip30::PCI0);
        bridge.set_revision(BRIDGE_REVISION);
        bridge.set_microlan(Self::board_record());
        // The Bridge answers every small window but the Hub's own.
        for widget in (0..16).filter(|&w| w != WIDGET_HUB) {
            node.hub.io().connect_widget(widget, bridge.clone());
        }
        pci_devices(node, cfg, bridge, &crate::dev::ioc3::IP29, controller, "origin200", Self::NAME)
    }
}

/// The PCI devices both IP27 I/O boards carry behind their Bridge: two
/// ISP1040s and an IOC3 with its Ethernet, serial ports and Timekeeper.
pub(crate) fn pci_devices(
    node: &Node,
    cfg: &MachineConfig,
    bridge: Arc<Bridge>,
    ioc3_board: &'static Ioc3Board,
    controller: Arc<dyn ModuleController>,
    scsi_name: &str,
    name: &str,
) -> Result<BoardIo, String> {
    let heartbeat = Arc::new(AtomicU64::new(0));
    let (bus_a, bus_b) = (Isp1040::new_bus(), Isp1040::new_bus());
    let attached = crate::scsi::attach_from_config(cfg, Some(scsi_name), &|c| match c {
        0 => Ok(&bus_a),
        1 => Ok(&bus_b),
        _ => Err(format!("controller {c} ({name} has controllers 0 and 1)")),
    })?;
    if let Some(d) = attached.daynaports.first() {
        return Err(format!("{}: kind = \"daynaport\" is not available on the {name}", crate::config::scsi_label(d.controller, d.id)));
    }
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
    let eth = ioc3.eth().expect("an IP27 I/O board's IOC3 fits Ethernet").clone();
    pci.attach(slot(2), ioc3.clone());
    ioc3.connect_superio(pci);
    ioc3.set_timer_manager(node.timers.clone());

    let console = cfg.ci.then(|| Arc::new(CiSerialBackend::new()));
    let (port_a, port_b) = crate::dev::serial_host::console_backends(cfg, name, console.as_ref());
    ioc3.set_serial_backend(0, port_a);
    if let Some(b) = port_b {
        ioc3.set_serial_backend(1, b);
    }

    let mut nat = None;
    let net = GatewayConfig::from_network(&cfg.network());
    if let Some(link) = Link::with_gateway(net.clone())? {
        nat = Some(Nat { control: link.nat(), expected: (net.client_ip, net.gateway_ip, net.netmask), frames: link.frames() });
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
        scratch: attached.scratch,
        nat,
        controller,
        reset,
        #[cfg(test)]
        bridge: test_bridge,
    })
}

#[cfg(test)]
mod tests;
