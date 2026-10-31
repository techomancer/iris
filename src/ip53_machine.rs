//! The Tezro (IP53 board, IP35 software platform): R16000s on one Bedrock
//! node, a PIC host bridge and the IO9 board's IOC4, ISP12160 and BCM5701,
//! with Iris's CPU thread, monitor and serial console.
//!
//! Selected with `[machine] profile = "tezro_ip53"`. Shared keys select the PROM,
//! NVRAM, memory and SCSI devices; `[tezro]` selects the processor
//! population, the serial console and writable flash. The node itself is
//! the one the Fuel uses ([`Node`]); this board adds its widgets and its
//! inputs. Graphics is not attached.
//!
//! The board follows the QEMU Tezro compositions:
//!
//! - One PIC at XIO widget 15, whose PXBOW answers as widget 0 and forwards
//!   the processor's accesses to widget 15. Its head 0 carries the IO9 PCI
//!   bus: the IOC4 at device 1 (input 0), the ISP12160 at device 3 (input 2)
//!   and the BCM5701 at device 4 (input 3), with REQ64# wired on the
//!   BCM5701's slot. The PIC's interrupts and DMA go to Bedrock IO, the
//!   host widget 10.
//! - The IOC4 carries the DS1744-compatible clock and NVRAM in its BAR and
//!   the PS/2 keyboard and mouse; its four ports are serial streams beside
//!   the L1's, and the L1 is the console by default.
//! - Bedrock board word 0x2001, the Tezro firmware record (700 MHz, an
//!   authored configuration block), and an L1 that reports
//!   brick type 0x39, revision 1.48.1, the first port-speed integer 1 and
//!   the router `brick` reply.
//!
//! # What is approximated here
//!
//! - Every identity above (board word, L1 type and revision, PIC revision 3
//!   profile, PCI devices, revisions and interrupt routes, REQ64# on the
//!   BCM5701's slot) is a software input the QEMU profiles choose, not a
//!   value measured on a Tezro.
//! - The one-processor board has one PIC. The tower guide gives that board
//!   two PCI-X buses and one XIO connector; that it is one PIC at widget 15
//!   is inferred from them.
//! - PIC DMA aimed at a widget other than Bedrock IO is refused and logged,
//!   as the Fuel's XBridge path refuses it.
//! - No EEPROM image is fitted for the IO9, the node board or the BCM5701:
//!   no published source gives their contents. A BCM5701 EEPROM
//!   transfer does not complete.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use crate::config::{scsi_label, MachineConfig, TezroConsole};
use crate::cpu::mips_cache_shadow::R16000ShadowCache;
use crate::cpu::mips_cache_v2::MipsCache;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::cpu_monitor::CpuMonitor;
use crate::cpu::mips_tlb::MipsTlb;
use crate::cpu::smp::World;
use crate::dev::bcm5701::{Bcm5701, Bcm5701Config};
use crate::dev::bedrock_flash::FirmwareRecord;
use crate::dev::bedrock_io::BedrockIo;
use crate::dev::bedrock_mem::MemoryPopulation;
use crate::dev::bedrock_pi::Population;
use crate::dev::ds1744::{Ds1744, YearBase};
use crate::dev::heart::{host_clock, Clock};
use crate::dev::ioc4::{Ioc4, Ioc4Config};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::L1Config;
use crate::dev::pic::{Function, Pic, PicPeer, Refused};
use crate::dev::prom_image;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::CiSerialBackend;
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::ip35::CPU_PHYS_MASK;
use crate::ip35::node::{connect_cpu, connect_serial_ports, Node, NodeConfig, SerialPort};
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link, NatControl};
use crate::perf_monitor::PerfMonitor;
use crate::scsi::{attach_from_config, ScsiBus};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable, BUS_OK};
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use parking_lot::Mutex;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc, Weak};

/// The processor model, chosen here and nowhere else.
pub(crate) type TezroCache = R16000ShadowCache;
pub(crate) type TezroCpu = MipsExecutor<MipsTlb, TezroCache>;

/// The Bedrock board word the QEMU Tezro profiles select.
const BOARD_ID: u64 = 0x2001;
/// The IO9's PIC, and the widget Bedrock IO answers as.
const IO9_WIDGET: u8 = 15;
const HOST_WIDGET: u8 = 10;
/// IO9's PCI functions: configuration device on head 0.
const IOC4_DEVICE: u8 = 1;
const SCSI_DEVICE: u8 = 3;
const ETH_DEVICE: u8 = 4;
/// The count register runs at half the processor's 700 MHz.
const DEFAULT_COUNT_HZ: u64 = 350_000_000;
/// The serial connections: the L1, then IOC4 ports 0 to 3.
const SERIAL_PORTS: usize = 5;

pub struct TezroMachine {
    board: Arc<TezroBoard>,
    /// The running processors, CPU 0 first. Only CPU 0 runs today.
    cpus: Vec<Arc<dyn CpuDevice>>,
    timers: Arc<TimerManager>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci_serial: Option<Arc<CiSerialBackend>>,
    console: usize,
    ci: bool,
    scratch: Option<PathBuf>,
    nvram: String,
    frames: Option<Arc<AtomicU64>>,
    nat: Option<(Arc<NatControl>, (Ipv4Addr, Ipv4Addr, Ipv4Addr))>,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
}

impl TezroMachine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        cfg.validate()?;
        let prom = prom_image::load(Path::new(&cfg.prom), 8 * 1024 * 1024).map_err(|e| format!("Tezro {e}"))?;
        let clock = Arc::new(Ds1744::new(YearBase::From(1968))?);
        clock.set_nvram_file(Path::new(&cfg.nvram))?;
        let memory = cfg.tezro_memory()?;
        let board = Arc::new(TezroBoard::with_config(TezroBoardConfig {
            payload: &prom,
            clock: host_clock(),
            flash_path: cfg.tezro.flash.as_deref().map(Path::new),
            station_mac: cfg.network().mac,
            timekeeper: clock,
            memory: &memory,
            cpus: usize::from(cfg.tezro.cpus),
        })?);
        let timers = Arc::new(TimerManager::new());
        board.ioc4.set_timer_manager(timers.clone());
        let count_hz = cfg.clock.fixed_mhz.map_or(DEFAULT_COUNT_HZ, |mhz| (mhz * 1_000_000.0) as u64).max(1);
        let cpus = build_cpus(&board, &timers, count_hz);
        let cpu = cpus[0].clone();
        let node = &board.node;
        // A rebuild of the whole memory map runs with the other processors
        // parked; with one running there is nobody to park.
        let world = Arc::new(World::new());
        world.set_slots(cpus.iter().map(|c| c.park_slot()).collect());
        let remap_world = world.clone();
        node.md.set_remap_guard(Arc::new(move |work: &mut dyn FnMut()| remap_world.stop_others(work)));
        let rx = connect_cpu("Tezro", &node.md, &node.pi[0], &node.lb, &node.ni, &node.l1, &cpu);
        // After connect_cpu set the windows: each compile worker takes the
        // cache's geometry and the CPU model's policies once, as it starts.
        // Workers read plain physical code addresses, which the cached plane
        // serves; the uncached views would decode them as Hub registers.
        #[cfg(feature = "jitv2")]
        {
            let cpu_device: Arc<dyn Device> = cpu.clone().as_device();
            let jit = cpu.jitv2();
            let mut g = jit.lock();
            g.compile_queue.set_thread_count(cfg.jitv2.threads.max(1));
            g.compile_queue.set_cpu(Arc::downgrade(&cpu_device));
            g.compile_queue.set_owner(Arc::downgrade(&jit));
            #[cfg(not(feature = "jitv2_lockstep"))]
            {
                let stats = g.stats.clone();
                g.compile_queue.start(board.cached_bus(), stats);
            }
            // The compile pool keeps only a weak handle on the CPU.
            std::mem::forget(cpu_device);
        }
        let scratch = Self::attach_scsi(cfg, &board)?;
        let console = cfg.tezro.console.index();
        let ci_serial = cfg.ci.then(|| board.serial[console].clone());
        Self::connect_serial(cfg, &board, console)?;
        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(CpuMonitor::new(cpus.clone())));
        monitor.register_device(PerfMonitor::new(
            cpu.running_flag(), cpu.cycles_ptr(), cpu.fasttick_count(), None, None,
        ));
        monitor.register_device(node.md.physical().clone());
        monitor.register_device(node.l1.clone());
        monitor.register_device(board.scsi.clone());
        monitor.register_device(board.ioc4.ps2());
        cpu.register_locks();
        board.scsi.register_locks();
        board.ioc4.register_locks();
        board.eth.register_locks();
        for pic in &board.pics {
            pic.register_locks();
        }
        node.flash.register_locks();
        node.md.register_locks();
        let (mut nat, mut frames) = (None, None);
        let net = GatewayConfig::from_network(&cfg.network());
        if let Some(link) = Link::with_gateway(net.clone())? {
            frames = Some(link.frames());
            nat = Some((link.nat(), (net.client_ip, net.gateway_ip, net.netmask)));
            board.eth.attach(link);
        }
        eprintln!(
            "iris: Tezro IP53, {} R16000 700 MHz fitted, CPU 0 running, {} MiB, serial console {:?}",
            cfg.tezro.cpus,
            memory.total_mib(),
            cfg.tezro.console
        );
        Ok(Self {
            board,
            cpus,
            timers,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci_serial,
            console,
            ci: cfg.ci,
            scratch,
            nvram: cfg.nvram.clone(),
            nat,
            frames,
            events: Some(rx),
            slot: None,
            lifecycle: GuestState::Idle,
        })
    }

    fn attach_scsi(cfg: &MachineConfig, board: &TezroBoard) -> Result<Option<PathBuf>, String> {
        let attached = attach_from_config(cfg, Some("tezro"), &|channel| {
            board.scsi.bus_for_channel(channel)
                .ok_or_else(|| format!("controller {channel} (a Tezro has channels 0 and 1)"))
        })?;
        if let Some(device) = attached.daynaports.first() {
            return Err(format!("{}: kind = \"daynaport\" is not available on the Tezro",
                scsi_label(device.controller, device.id)));
        }
        Ok(attached.scratch)
    }

    fn connect_serial(cfg: &MachineConfig, board: &TezroBoard, console: usize) -> Result<(), String> {
        let [a, b] = cfg.serial_ports();
        let ioc4 = |port: usize| -> Box<dyn Fn(Arc<dyn crate::dev::serial_host::SerialBackend>) + '_> {
            Box::new(move |backend| board.ioc4.set_serial_backend(port, backend))
        };
        let ports = [
            SerialPort {
                label: "L1",
                port: cfg.tezro.l1_port.unwrap_or(8882),
                setting: "tezro.l1_port",
                ci: &board.serial[0],
                log: cfg.tezro.l1_log.clone(),
                connect: Box::new(|backend| board.node.l1.set_serial_backend(backend)),
            },
            SerialPort { label: "IOC4 0", port: a, setting: "serial_port_a", ci: &board.serial[1], log: None, connect: ioc4(0) },
            SerialPort { label: "IOC4 1", port: b, setting: "serial_port_b", ci: &board.serial[2], log: None, connect: ioc4(1) },
            SerialPort {
                label: "IOC4 2",
                port: cfg.tezro.serial_port_c.unwrap_or(8883),
                setting: "tezro.serial_port_c",
                ci: &board.serial[3],
                log: None,
                connect: ioc4(2),
            },
            SerialPort {
                label: "IOC4 3",
                port: cfg.tezro.serial_port_d.unwrap_or(8884),
                setting: "tezro.serial_port_d",
                ci: &board.serial[4],
                log: None,
                connect: ioc4(3),
            },
        ];
        connect_serial_ports(cfg, "Tezro", &ports, console)
    }
}

impl Workstation for TezroMachine {
    fn start(&mut self) {
        self.lifecycle = GuestState::Starting;
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        if !self.ci {
            self.cpu_start();
        }
    }
    fn stop(&mut self) {
        self.cpu().clone().as_device().stop();
        self.board.stop();
        self.board.timekeeper.flush();
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        self.cpu().power_on();
        self.board.power_on();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        self.board.start();
    }
    fn cpu_start(&self) {
        self.cpu().clone().as_device().start();
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
        let events = self.events.take().expect("Tezro system controller registered twice");
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
        self.board.ioc4.ps2()
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
        self.board.serial[self.console].push_host(bytes);
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        if path.is_some_and(|p| p != self.nvram) {
            return Err("Tezro NVRAM save uses the configured path".into());
        }
        self.board.timekeeper.flush();
        Ok(self.nvram.clone())
    }
    fn scratch_path(&self) -> Option<&Path> {
        self.scratch.as_deref()
    }
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        Some(self.board.scsi.clone() as Arc<dyn DiscChanger>)
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.scsi_buses().iter().map(|bus| bus.pending_chd_sync_count()).sum()
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.board.flush_disks()
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        ScsiBus::sync_chd_disks(&self.scsi_buses(), only, progress, cancel)
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

impl TezroMachine {
    /// CPU 0, the one the front ends drive.
    fn cpu(&self) -> &Arc<dyn CpuDevice> {
        &self.cpus[0]
    }

    fn scsi_buses(&self) -> Vec<&ScsiBus> {
        (0..2).filter_map(|channel| self.board.scsi.bus_for_channel(channel)).collect()
    }
}

impl Drop for TezroMachine {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        self.monitor.shutdown();
    }
}

/// The processors the machine runs, CPU 0 first. The board fits more, but
/// only CPU 0 is built: starting the others is not modelled, and the PIs
/// refuse and log what would reach them.
fn build_cpus(board: &Arc<TezroBoard>, timers: &Arc<TimerManager>, count_hz: u64) -> Vec<Arc<dyn CpuDevice>> {
    let mut exec = build_cpu(board);
    exec.core.set_timer_manager(timers.clone());
    exec.core.set_count_hz(count_hz);
    exec.core.count_write_samples = true;
    let poll_board = board.clone();
    exec.batch_hook = Some(Box::new(move |_| poll_board.poll()));
    vec![Arc::new(MipsCpu::<_, TezroCache>::new(*exec))]
}

/// The R16000 with the Tezro's reset-mode inputs, on the board's bus.
fn build_cpu(board: &TezroBoard) -> Box<TezroCpu> {
    let cfg = MipsCpuConfig::for_model::<TezroCache>(Arc::new(crate::ip35::memory_map(&board.node.md.ram_extent())));
    let mut cpu = Box::new(MipsExecutor::new(
        board.node.bus.clone() as Arc<dyn BusDevice>,
        MipsTlb::new(cfg.tlb_entries),
        &cfg,
    ));
    cpu.phys_mask = CPU_PHYS_MASK;
    // The flash record's mode word and the processor's reset mode must
    // agree, or the PROM enters its error routine before GP is set.
    cpu.core.cp0_config = FirmwareRecord::TEZRO.reset_config0();
    cpu.cache = TezroCache::new(board.cached_bus());
    // The PROM keeps its stack in cache lines it tags for a window with no
    // memory behind it yet; the controller answers that window while they stand.
    let memory = board.node.md.clone();
    cpu.cache.set_pd_tag_store_hook(Box::new(move |tag| memory.startup_tag_store(tag)));
    cpu
}

pub(crate) struct TezroBoardConfig<'a> {
    pub payload: &'a [u8],
    pub clock: Clock,
    pub flash_path: Option<&'a Path>,
    pub station_mac: [u8; 6],
    /// The DS1744-compatible clock and NVRAM in the IOC4's BAR.
    pub timekeeper: Arc<Ds1744>,
    pub memory: &'a MemoryPopulation,
    /// Processors fitted.
    pub cpus: usize,
}

/// The Tezro's devices around its Bedrock node.
pub(crate) struct TezroBoard {
    pub node: Node,
    /// The PICs, the IO9's at widget 15 first.
    pub pics: Vec<Arc<Pic>>,
    pub ioc4: Arc<Ioc4>,
    pub scsi: Arc<Isp12160>,
    pub eth: Arc<Bcm5701>,
    pub timekeeper: Arc<Ds1744>,
    pub serial: [Arc<CiSerialBackend>; SERIAL_PORTS],
    clock: Clock,
    last_clock: AtomicU64,
    running: AtomicBool,
}

impl TezroBoard {
    /// Flush the disks on both channels of the storage controller.
    pub fn flush_disks(&self) -> std::io::Result<()> {
        (0..2).try_for_each(|channel| self.scsi.bus_for_channel(channel).unwrap().flush())
    }

    #[cfg(test)]
    pub fn new(cpus: usize) -> Result<Self, String> {
        Self::with_config(TezroBoardConfig {
            payload: &[],
            clock: Arc::new(|| 0),
            flash_path: None,
            station_mac: L1Config::default().mac,
            timekeeper: Arc::new(Ds1744::new(YearBase::From(1968))?),
            memory: &MemoryPopulation::default(),
            cpus,
        })
    }

    pub fn with_config(config: TezroBoardConfig<'_>) -> Result<Self, String> {
        let TezroBoardConfig { payload, clock, flash_path, station_mac, timekeeper, memory, cpus } = config;
        if cpus != 1 {
            return Err(format!("{cpus} processors; this Tezro board fits one"));
        }
        memory.check_for_cpus(cpus)?;
        let node = Node::build(NodeConfig {
            name: "Tezro",
            payload,
            clock: clock.clone(),
            flash_path,
            banks_mib: &memory.banks_mib(),
            population: Population::fitted_cpus(cpus)?,
            firmware: FirmwareRecord::TEZRO,
            board_id: BOARD_ID,
            l1: L1Config {
                type_code: 0x39,
                revision: Some([1, 48, 1]),
                mac: station_mac,
                port_speed_first: 1,
                router: true,
                ..Default::default()
            },
        })?;
        let serial: [Arc<CiSerialBackend>; SERIAL_PORTS] = std::array::from_fn(|_| Arc::new(CiSerialBackend::new()));
        node.l1.set_serial_backend(serial[0].clone());

        let io9 = Pic::new(IO9_WIDGET);
        let pxbow = io9.enable_pxbow(HOST_WIDGET, clock.clone()).map_err(|e| format!("Tezro PXBOW: {e:?}"))?;
        node.io.connect_widget(0, pxbow.widget_zero());
        node.io.connect_widget(IO9_WIDGET, pxbow.route(IO9_WIDGET));
        let pics = vec![io9.clone()];
        for (index, pic) in pics.iter().enumerate() {
            pic.connect_peer(Arc::new(BedrockPeer(Arc::downgrade(&node.io))))
                .map_err(|_| "Tezro: PIC peer connected twice".to_string())?;
            // Each PIC owns its own 16 PI level sources, as QEMU numbers them.
            for (interface, pi) in node.pi.iter().enumerate() {
                pic.connect_pi(interface, pi.clone(), index * 2 * crate::dev::pic::PINS);
            }
        }

        let ioc4 = Arc::new(Ioc4::new(Ioc4Config::tezro()));
        ioc4.attach_clock(timekeeper.clone());
        for (port, backend) in serial[1..].iter().enumerate() {
            ioc4.set_serial_backend(port, backend.clone());
        }
        let scsi = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus()));
        let eth = Bcm5701::new(Bcm5701Config::new(node.l1.station_mac()))?;
        io9.attach(0, IOC4_DEVICE, ioc4.clone(), Function { self_ordered_pio: true, req64: false });
        io9.attach(0, SCSI_DEVICE, scsi.clone(), Function::default());
        io9.attach(0, ETH_DEVICE, eth.clone(), Function { self_ordered_pio: false, req64: true });
        for device in [IOC4_DEVICE, SCSI_DEVICE, ETH_DEVICE] {
            io9.wire_inta(0, device);
        }
        Ok(Self {
            node,
            pics,
            ioc4,
            scsi,
            eth,
            timekeeper,
            serial,
            last_clock: AtomicU64::new(clock()),
            clock,
            running: AtomicBool::new(false),
        })
    }

    pub fn cached_bus(&self) -> Arc<dyn BusDevice> {
        self.node.bus.cached_view(self.node.md.dma_view(), self.node.md.ram_extent())
    }

    /// Everything back to its power-on state. The functions reset before
    /// their PIC, so each samples its slot's REQ64# as the bridge's reset
    /// ends, as on the bus.
    pub fn power_on(&self) {
        let node = &self.node;
        for pi in &node.pi {
            pi.power_on();
        }
        node.md.power_on();
        node.lb.power_on();
        node.io.power_on();
        node.ni.power_on();
        node.xb.power_on();
        node.l1.power_on();
        node.flash.power_on();
        self.ioc4.power_on();
        self.scsi.power_on();
        self.eth.power_on();
        for pic in &self.pics {
            pic.power_on();
        }
    }

    pub fn start(&self) {
        self.last_clock.store((self.clock)(), Ordering::Release);
        self.node.md.start_clock();
        self.running.store(true, Ordering::Release);
        for pi in &self.node.pi {
            pi.start();
        }
        self.node.l1.start();
        self.ioc4.start();
        self.eth.start();
        self.scsi.start();
    }

    pub fn poll(&self) {
        if self.running.load(Ordering::Acquire) {
            let now = (self.clock)();
            let previous = self.last_clock.swap(now, Ordering::AcqRel);
            self.node.flash.advance_ns(now.saturating_sub(previous));
            for pi in &self.node.pi {
                pi.poll();
            }
        }
    }

    pub fn stop(&self) {
        self.poll();
        self.running.store(false, Ordering::Release);
        self.node.md.stop_clock();
        self.scsi.stop();
        self.eth.stop();
        self.ioc4.stop();
        self.node.l1.stop();
        for pi in &self.node.pi {
            pi.stop();
        }
    }
}

impl Drop for TezroBoard {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The far end of a PIC's link: Bedrock IO. Interrupt packets and DMA arrive
/// there as inbound packets from the PIC's widget, device 0, as QEMU's
/// Bedrock IO takes them by default. DMA aimed at another widget is refused:
/// peer-to-peer XIO is not modelled.
struct BedrockPeer(Weak<BedrockIo>);

const INBOUND_DEVICE: u8 = 0;

impl BedrockPeer {
    fn io(&self) -> Result<Arc<BedrockIo>, Refused> {
        self.0.upgrade().ok_or(Refused)
    }

    fn host(widget: u8) -> Result<(), Refused> {
        if widget == HOST_WIDGET {
            Ok(())
        } else {
            dlog!(LogModule::Bedrock, "Tezro: PIC DMA to widget {widget} refused; only Bedrock IO is reachable");
            Err(Refused)
        }
    }

    /// The legs of a transfer: each the largest naturally aligned size of
    /// eight bytes or fewer that fits, so word accesses keep their width.
    fn legs(address: u64, len: usize) -> impl Iterator<Item = (usize, usize)> {
        let mut offset = 0;
        std::iter::from_fn(move || {
            if offset == len {
                return None;
            }
            let at = address + offset as u64;
            let size = [8usize, 4, 2, 1]
                .into_iter()
                .find(|&n| at.is_multiple_of(n as u64) && n <= len - offset)
                .unwrap_or(1);
            let leg = (offset, size);
            offset += size;
            Some(leg)
        })
    }
}

impl PicPeer for BedrockPeer {
    fn interrupt(&self, source_widget: u8, address: u64, payload: u64) -> Result<(), Refused> {
        let io = self.io()?;
        (io.inbound_write(source_widget, Some(INBOUND_DEVICE), address, 8, payload) == BUS_OK)
            .then_some(())
            .ok_or(Refused)
    }

    fn dma_read(&self, source_widget: u8, widget: u8, address: u64, data: &mut [u8]) -> Result<(), Refused> {
        Self::host(widget)?;
        let io = self.io()?;
        for (offset, size) in Self::legs(address, data.len()) {
            let value = io.inbound_read(source_widget, Some(INBOUND_DEVICE), address + offset as u64, size as u8);
            if value.status != BUS_OK {
                return Err(Refused);
            }
            data[offset..offset + size].copy_from_slice(&value.data.to_be_bytes()[8 - size..]);
        }
        Ok(())
    }

    fn dma_write(&self, source_widget: u8, widget: u8, address: u64, data: &[u8]) -> Result<(), Refused> {
        Self::host(widget)?;
        let io = self.io()?;
        for (offset, size) in Self::legs(address, data.len()) {
            let value = data[offset..offset + size].iter().fold(0u64, |v, b| (v << 8) | u64::from(*b));
            if io.inbound_write(source_widget, Some(INBOUND_DEVICE), address + offset as u64, size as u8, value) != BUS_OK {
                return Err(Refused);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
