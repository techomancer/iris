//! The Tezro (IP53 board, IP35 software platform): R16000s on one Bedrock
//! node, a PIC host bridge and the IO9 board's IOC4, ISP12160 and BCM5701,
//! with Iris's CPU thread, monitor and serial console.
//!
//! Selected with `[machine] profile = "tezro_ip53"`. Shared keys select the PROM,
//! NVRAM, memory and SCSI devices; `[tezro]` selects the processor
//! population, the serial console and writable flash. The node itself is
//! the one the Fuel uses ([`Node`]); this board adds its widgets and its
//! inputs. With `[graphics] board = "v12"` a VPro V12 sits at widget 12
//! behind the PXBOW; otherwise there is no graphics.
//!
//! The board follows the QEMU Tezro compositions:
//!
//! - The one-processor board has one PIC at XIO widget 15; the two- and
//!   four-processor board adds a second at widget 11, with nothing on its
//!   buses, and a V12 slot at widget 12 that is left empty. The PIC at
//!   widget 15 is the IO9's: its PXBOW answers as widget 0 and forwards
//!   the processor's accesses to widget 15. On the larger board it carries
//!   widget 11 too, and shows that link present and alive (unlike QEMU's
//!   composition, below). Its head 0 carries the IO9 PCI bus: the IOC4 at
//!   device 1 (input 0), the ISP12160 at device 3 (input 2) and the
//!   BCM5701 at device 4 (input 3), with REQ64# wired on the BCM5701's
//!   slot. The PIC's interrupts and DMA go to Bedrock IO, the host widget
//!   10.
//! - The IOC4 carries the DS1744-compatible clock and NVRAM in its BAR and
//!   the PS/2 keyboard and mouse; its four ports are serial streams beside
//!   the L1's, and the L1 is the console by default.
//! - Bedrock board word 0x2001, the Tezro firmware record (the part's clock,
//!   an authored configuration block), and an L1 that reports
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
//! - Every fitted processor runs from reset, each on its own thread, as in
//!   QEMU's IP35 node; CPU enable is stored and starts or stops nothing. The
//!   PROM's boot rendezvous holds a processor that arrives early
//!   ([`Rendezvous`]), an approximation with no hardware counterpart.
//! - The second PIC is a port of the IO9 PIC's PXBOW, beside the host's
//!   and the IO9 PIC's links: the PROM and IRIX learn which widgets exist
//!   only from widget 0's link status and auxiliary status, so one
//!   crossbar per node carries both PICs. That is a modelling consequence
//!   of the discovery path, not a known route: whether the second PIC has
//!   a crossbar of its own in front of it, and its part (0xd102 or 0xd112)
//!   and link reset values beyond alive and present, are unknown. QEMU's
//!   composition still gives the second PIC its own PXBOW, whose widget 0
//!   nothing maps, and leaves link 11 absent on the IO9 PIC's. The second
//!   PIC's interrupts and DMA reach Bedrock IO directly, as the IO9 PIC's
//!   do, rather than through the crossbar.
//! - PIC DMA aimed at a widget other than Bedrock IO is refused and logged,
//!   as the Fuel's XBridge path refuses it.
//! - No EEPROM image is fitted for the IO9, the node board or the BCM5701:
//!   no published source gives their contents. A BCM5701 EEPROM
//!   transfer does not complete.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use crate::config::{scsi_label, CpuModel, GraphicsBoard, MachineConfig, TezroConsole};
use crate::cpu::mips_cache_shadow::{R16000ShadowCache, R16000_MHZ700_PRID};
use crate::cpu::mips_cache_v2::{CpuModel as CacheCpuModel, MipsCache};
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::cpu_monitor::CpuMonitor;
use crate::cpu::mips_tlb::MipsTlb;
use crate::cpu::smp::World;
use crate::dev::bcm5701::{Bcm5701, Bcm5701Config};
use crate::dev::bedrock_flash::FirmwareRecord;
use crate::dev::bedrock_io::{gfx_route, BedrockIo};
use crate::dev::bedrock_mem::MemoryPopulation;
use crate::dev::bedrock_pi::{BedrockPi, NodeHooks, Population};
use crate::dev::ds1744::{Ds1744, YearBase};
use crate::dev::heart::{host_clock, Clock};
use crate::dev::ioc4::{Ioc4, Ioc4Config};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::L1Config;
use crate::dev::odyssey::{
    Odyssey, PacketSender, PiGfx, PiGfxLink, PI_GFX_BIAS, PI_GFX_CREDIT, PI_GFX_CREDIT_MASK, PI_GFX_INT_COUNT,
    PI_GFX_REGS, PI_GFX_SLOTS, PI_GFX_SLOT_REGS,
};
use crate::dev::pic::{Function, Pic, PicPeer, Refused};
use crate::dev::pxbow::Pxbow;
use crate::dev::prom_image;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::CiSerialBackend;
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::ip35::CPU_PHYS_MASK;
use crate::ip35::node::{connect_cpus, connect_serial_ports, Node, NodeConfig, SerialPort, Store, StoreCounter, WidgetDma};
use crate::ip35::rendezvous::Rendezvous;
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

/// The Bedrock board word the QEMU Tezro profiles select.
const BOARD_ID: u64 = 0x2001;
/// The IO9's PIC, the second PIC of the two- and four-processor board, and
/// the widget Bedrock IO answers as. Widget 12 on that board is the V12's
/// XIO slot, which is left empty.
const IO9_WIDGET: u8 = 15;
const SECOND_PIC_WIDGET: u8 = 11;
const HOST_WIDGET: u8 = 10;
/// IO9's PCI functions: configuration device on head 0.
const IOC4_DEVICE: u8 = 1;
const SCSI_DEVICE: u8 = 3;
const ETH_DEVICE: u8 = 4;
/// The R16000 part a Tezro runs, from `[machine] cpu`. Both are the CPU
/// models' R16000 with a 4 MB secondary cache; they differ in PRId, the
/// Config reset word and the clock the firmware record gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TezroPart {
    /// Revision 2.1 at 700 MHz (`r16000_700`), the default.
    Mhz700,
    /// Revision 2.2 at 800 MHz (`r16000`).
    Mhz800,
}

impl TezroPart {
    /// The part `cpu` names. `r10000` names the family, as on the Fuel, and
    /// takes the default part.
    pub(crate) fn of(cpu: CpuModel) -> Result<Self, String> {
        match cpu {
            CpuModel::R16000Mhz700 | CpuModel::R10000 => Ok(Self::Mhz700),
            CpuModel::R16000 => Ok(Self::Mhz800),
            other => Err(format!("a Tezro has R16000s, not the {}", other.label())),
        }
    }

    /// The flash record the PROM checks this part against.
    pub(crate) fn firmware(self) -> FirmwareRecord {
        match self {
            Self::Mhz700 => FirmwareRecord::TEZRO,
            Self::Mhz800 => FirmwareRecord::TEZRO_800,
        }
    }

    /// The part's PRId, from the CPU models.
    pub(crate) fn prid(self) -> u32 {
        match self {
            Self::Mhz700 => R16000_MHZ700_PRID,
            Self::Mhz800 => <R16000ShadowCache as CacheCpuModel>::PRID,
        }
    }

    /// The processor clock in MHz.
    pub(crate) fn mhz(self) -> u32 {
        self.firmware().cpu_clock_hz / 1_000_000
    }

    /// The count register runs at half the processor clock.
    pub(crate) fn count_hz(self) -> u64 {
        u64::from(self.firmware().cpu_clock_hz) / 2
    }
}
/// How often the poll returns lifted checked pages to the direct window, in
/// board nanoseconds.
const RETIRE_INTERVAL_NS: u64 = 100_000_000;
/// The serial connections: the L1, then IOC4 ports 0 to 3.
const SERIAL_PORTS: usize = 5;

pub struct TezroMachine {
    board: Arc<TezroBoard>,
    /// The processors, CPU 0 first, in the board's slot order.
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
        #[cfg(feature = "jitv2")]
        check_jitv2_memory(&memory)?;
        let part = TezroPart::of(cfg.machine.cpu)?;
        let board = Arc::new(TezroBoard::with_config(TezroBoardConfig {
            payload: &prom,
            clock: host_clock(),
            flash_path: cfg.tezro.flash.as_deref().map(Path::new),
            station_mac: cfg.network().mac,
            timekeeper: clock,
            memory: &memory,
            cpus: usize::from(cfg.machine.cpus()),
            part,
            v12: cfg.graphics_board() == GraphicsBoard::V12,
            graphics_pool: (cfg.graphics.threads, cfg.graphics.tile),
        })?);
        let timers = Arc::new(TimerManager::new());
        board.ioc4.set_timer_manager(timers.clone());
        let count_hz = cfg.clock.fixed_mhz.map_or(part.count_hz(), |mhz| (mhz * 1_000_000.0) as u64).max(1);
        let cpus = build_cpus(&board, &timers, count_hz);
        let cpu = cpus[0].clone();
        board.status.set_cpu_cycles(cpu.cycles_ptr());
        board.status.set_count_hz(cpu.count_hz_atomic());
        for c in cpus.iter().skip(1) {
            board.status.add_cpu_stats(c.cycles_ptr(), c.fasttick_count());
        }
        let node = &board.node;
        // A rebuild of the whole memory map runs with the other processors
        // parked.
        let world = Arc::new(World::new());
        world.set_slots(cpus.iter().map(|c| c.park_slot()).collect());
        let remap_world = world.clone();
        node.md.set_remap_guard(Arc::new(move |work: &mut dyn FnMut()| remap_world.stop_others(work)));
        // One page pool and one body of compiled code, reaching each
        // processor's state through the core it is called with.
        #[cfg(feature = "jitv2")]
        if cpus.len() > 1 {
            if cfg!(feature = "jitv2_lockstep") {
                return Err("[machine] cpus above 1 cannot run with jitv2_lockstep".into());
            }
            let jit = cpu.jitv2();
            for c in &cpus {
                c.share_jitv2(jit.clone(), world.clone());
            }
        }
        board.rendezvous.set_cpus(&cpus);
        board.rendezvous.set_timer_manager(timers.clone());
        let rx = connect_cpus("Tezro", node, board.processors, &cpus);
        // After connect_cpu set the windows: each compile worker takes the
        // cache's geometry and the CPU model's policies once, as it starts.
        // Workers read plain physical code addresses through the view the CPU
        // fetched each page by: the cached plane for cached pages, the
        // uncached attribute-0 view for the reset PROM's uncached fetches.
        #[cfg(feature = "jitv2")]
        {
            let cpu_device: Arc<dyn Device> = cpu.clone().as_device();
            let jit = cpu.jitv2();
            let mut g = jit.lock();
            g.compile_queue.set_thread_count(cfg.jitv2.threads.max(1));
            g.compile_queue.set_cpu(Arc::downgrade(&cpu_device));
            g.compile_queue.set_owner(Arc::downgrade(&jit));
            if cpus.len() > 1 {
                g.compile_queue.set_world(world.clone());
            }
            #[cfg(not(feature = "jitv2_lockstep"))]
            {
                let stats = g.stats.clone();
                g.compile_queue.start(board.code_bus(), stats);
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
        for c in &cpus {
            c.register_locks();
        }
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
            "iris: Tezro IP53, {} R16000 {} MHz, {} MiB, {}, serial console {:?}",
            cpus.len(),
            part.mhz(),
            memory.total_mib(),
            if board.v12.is_some() { cfg.graphics_board().label() } else { "no graphics" },
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
        for c in &self.cpus {
            c.clone().as_device().stop();
        }
        self.board.stop();
        self.board.timekeeper.flush();
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        for c in &self.cpus {
            c.power_on();
        }
        self.board.power_on();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        self.board.start();
    }
    /// Every processor starts at once, as from the node's reset.
    fn cpu_start(&self) {
        for c in &self.cpus {
            c.clone().as_device().start();
        }
    }
    fn cpu_is_running(&self) -> bool {
        self.cpus.iter().any(|c| c.clone().as_device().is_running())
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
        self.board.v12.as_ref().map(|v12| v12.display())
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
    fn cpu_counters(&self) -> Vec<(crate::cpu::mips_core::CyclesPtr, Arc<AtomicU64>)> {
        self.cpus.iter().map(|c| (c.cycles_ptr(), c.fasttick_count())).collect()
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

/// The processors the machine runs, of the board's part: one for each
/// fitted slot, CPU 0 first, in the board's slot order.
fn build_cpus(board: &Arc<TezroBoard>, timers: &Arc<TimerManager>, count_hz: u64) -> Vec<Arc<dyn CpuDevice>> {
    let count = board.processors.count();
    (0..count).map(|id| start_cpu::<R16000ShadowCache>(board, timers, count_hz, id as u8, count)).collect()
}

/// Processor `id` of `count`, of type `C`, with its timers; CPU 0 also
/// runs the board's poll.
fn start_cpu<C>(board: &Arc<TezroBoard>, timers: &Arc<TimerManager>, count_hz: u64, id: u8, count: usize) -> Arc<dyn CpuDevice>
where
    C: CacheCpuModel + From<Arc<dyn BusDevice>> + 'static,
    MipsCpu<MipsTlb, C>: CpuDevice,
{
    let mut exec = build_cpu::<C>(board);
    // CPU 0's Compare matches blink the graphics status bar's LED.
    exec.core.fasttick_count = if id == 0 { board.fasttick.clone() } else { Arc::new(AtomicU64::new(0)) };
    exec.core.set_timer_manager(timers.clone());
    exec.core.set_count_hz(count_hz);
    exec.core.count_write_samples = true;
    exec.core.cpu_id = id;
    exec.core.smp = count > 1;
    if id == 0 {
        let poll_board = board.clone();
        exec.batch_hook = Some(Box::new(move |_| poll_board.poll()));
    }
    Arc::new(MipsCpu::new(*exec))
}

/// A jitv2 build keys the PROM's uncached code pages at frame 1 << 20 and
/// up, above every frame below 4 GiB (`Jitv2::enable_uncached_views`), so it
/// takes only memory that ends at or below 4 GiB: banks 0 to 3. The default
/// layout puts banks 4 and 5 above it. Refused here rather than left to the
/// JIT's assertion.
#[cfg(feature = "jitv2")]
fn check_jitv2_memory(memory: &MemoryPopulation) -> Result<(), String> {
    if memory.fits_direct_window() {
        return Ok(());
    }
    Err(format!(
        "a jitv2 build runs a Tezro only with memory below 4 GiB (at most four banks), since it \
         compiles the PROM's uncached code under page keys above it; banks = {:?} reach past 4 GiB: \
         set banks = [1024, 1024, 1024, 1024] or fewer, or use a build without jitv2",
        memory.banks_mib()
    ))
}

/// A processor of type `C` with the board part's PRId and reset-mode
/// inputs, on the board's bus.
fn build_cpu<C: CacheCpuModel + From<Arc<dyn BusDevice>>>(board: &TezroBoard) -> Box<MipsExecutor<MipsTlb, C>> {
    let cfg = MipsCpuConfig::for_model::<C>(Arc::new(crate::ip35::memory_map(&board.node.md.ram_extent())));
    let mut cpu = Box::new(MipsExecutor::new(
        board.node.bus.clone() as Arc<dyn BusDevice>,
        MipsTlb::new(cfg.tlb_entries),
        &cfg,
    ));
    cpu.phys_mask = CPU_PHYS_MASK;
    // The flash record's mode word and the processor's reset mode must
    // agree, or the PROM enters its error routine before GP is set.
    cpu.core.cp0_config = board.part.firmware().reset_config0();
    cpu.core.reset_prid = board.part.prid();
    cpu.core.cp0_prid = board.part.prid();
    cpu.cache = C::from(board.cached_bus());
    // The PROM fetches its own code uncached; with memory over its window
    // the cached and uncached views of it differ.
    #[cfg(feature = "jitv2")]
    cpu.enable_jitv2_uncached_views(board.code_bus());
    // The PROM keeps its stack in cache lines it tags for a window with no
    // memory behind it yet; the controller answers that window, each
    // processor's own, while they stand. The hook runs on the processor's
    // thread, which names the processor.
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
    /// The R16000 part fitted.
    pub part: TezroPart,
    /// A VPro V12 at widget 12.
    pub v12: bool,
    /// Rendering threads and tile size for the V12.
    pub graphics_pool: (usize, u32),
}

/// The Tezro's devices around its Bedrock node.
pub(crate) struct TezroBoard {
    pub node: Node,
    /// The slots whose processors run: every fitted one.
    pub processors: Population,
    /// The PROM's boot rendezvous across them.
    pub rendezvous: Arc<Rendezvous>,
    /// The PICs in QEMU's host order, which numbers their PI level sources:
    /// widget 11's first when the board has it, then the IO9's at widget 15.
    pub pics: Vec<Arc<Pic>>,
    pub ioc4: Arc<Ioc4>,
    pub scsi: Arc<Isp12160>,
    pub eth: Arc<Bcm5701>,
    pub timekeeper: Arc<Ds1744>,
    pub serial: [Arc<CiSerialBackend>; SERIAL_PORTS],
    /// The R16000 part fitted, whose record the flash holds.
    pub part: TezroPart,
    /// The VPro V12, when the board has one.
    pub v12: Option<Arc<Odyssey>>,
    /// Each PI's graphics flow-control registers.
    pub gfx: [Arc<Mutex<BedrockGfx>>; 2],
    /// The status bar's activity bits and the CPU's Compare matches, which
    /// the graphics board shows.
    pub heartbeat: Arc<AtomicU64>,
    pub fasttick: Arc<AtomicU64>,
    /// What the graphics board's status bar shows: the activity bits, the
    /// Compare matches and the CPU's cycle count and Count rate.
    pub status: Arc<crate::disp::StatusSource>,
    clock: Clock,
    last_clock: AtomicU64,
    /// When the poll last returned lifted checked pages to the direct window.
    last_retire: AtomicU64,
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
            part: TezroPart::Mhz700,
            v12: false,
            graphics_pool: (2, 32),
        })
    }

    pub fn with_config(config: TezroBoardConfig<'_>) -> Result<Self, String> {
        let TezroBoardConfig { payload, clock, flash_path, station_mac, timekeeper, memory, cpus, part, v12, graphics_pool } = config;
        if ![1, 2, 4].contains(&cpus) {
            return Err(format!("{cpus} processors; a Tezro board fits 1, 2 or 4"));
        }
        memory.check_for_cpus(cpus)?;
        let running = Population::fitted_cpus(cpus)?;
        let node = Node::build(NodeConfig {
            name: "Tezro",
            payload,
            clock: clock.clone(),
            flash_path,
            banks_mib: &memory.banks_mib(),
            population: running,
            running,
            firmware: part.firmware(),
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
        let gfx: [Arc<Mutex<BedrockGfx>>; 2] = std::array::from_fn(|_| Arc::new(Mutex::new(BedrockGfx::new(clock.clone()))));
        let rendezvous = Rendezvous::new(running, clock.clone(), &node.lb);
        node.lb.connect_scratch_hooks(rendezvous.clone());
        let publication = Arc::downgrade(&rendezvous);
        node.io.on_scratch0_write(Box::new(move || {
            if let Some(r) = publication.upgrade() {
                r.presence_published();
            }
        }));
        for (pi, gfx) in node.pi.iter().zip(&gfx) {
            pi.set_node_hooks(Arc::new(PiHooks { gfx: gfx.clone(), rendezvous: Arc::downgrade(&rendezvous) }));
        }
        let heartbeat = Arc::new(AtomicU64::new(0));
        let fasttick = Arc::new(AtomicU64::new(0));
        let status = crate::disp::StatusSource::new(heartbeat.clone(), fasttick.clone());
        let v12 = v12.then(|| fit_v12(&node, running, &pxbow, &gfx, &status, graphics_pool)).transpose()?;
        let pics = if cpus == 1 {
            vec![io9.clone()]
        } else {
            // The PROM finds widget 11 only as a live link of widget 0's
            // crossbar, so the second PIC is a port of the same switch.
            let second = Pic::new(SECOND_PIC_WIDGET);
            pxbow
                .attach_endpoint(SECOND_PIC_WIDGET, second.widget())
                .map_err(|e| format!("Tezro: second PIC on the PXBOW: {e:?}"))?;
            node.io.connect_widget(SECOND_PIC_WIDGET, pxbow.route(SECOND_PIC_WIDGET));
            vec![second, io9.clone()]
        };
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
        scsi.set_heartbeat(heartbeat.clone());
        let eth = Bcm5701::new(Bcm5701Config::new(node.l1.station_mac()))?;
        io9.attach(0, IOC4_DEVICE, ioc4.clone(), Function { self_ordered_pio: true, req64: false });
        io9.attach(0, SCSI_DEVICE, scsi.clone(), Function::default());
        io9.attach(0, ETH_DEVICE, eth.clone(), Function { self_ordered_pio: false, req64: true });
        for device in [IOC4_DEVICE, SCSI_DEVICE, ETH_DEVICE] {
            io9.wire_inta(0, device);
        }
        Ok(Self {
            node,
            processors: running,
            rendezvous,
            pics,
            ioc4,
            scsi,
            eth,
            timekeeper,
            serial,
            part,
            v12,
            gfx,
            heartbeat,
            fasttick,
            status,
            last_clock: AtomicU64::new(clock()),
            last_retire: AtomicU64::new(clock()),
            clock,
            running: AtomicBool::new(false),
        })
    }

    pub fn cached_bus(&self) -> Arc<dyn BusDevice> {
        self.node.bus.cached_view(self.node.md.dma_view(), self.node.md.ram_extent())
    }

    /// The bus the JIT compiles code through; see [`crate::ip35::Ip35Bus::code_view`].
    #[cfg(feature = "jitv2")]
    pub fn code_bus(&self) -> Arc<dyn BusDevice> {
        self.node.bus.code_view(self.cached_bus())
    }

    /// Everything back to its power-on state. The functions reset before
    /// their PIC, so each samples its slot's REQ64# as the bridge's reset
    /// ends, as on the bus.
    pub fn power_on(&self) {
        let node = &self.node;
        self.rendezvous.reset();
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
        for gfx in &self.gfx {
            gfx.lock().reset();
        }
        if let Some(v12) = &self.v12 {
            v12.power_on();
        }
    }

    pub fn start(&self) {
        if let Some(v12) = &self.v12 {
            v12.start();
        }
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
            // Pages whose checked marks were lifted below MD's batch limit
            // keep their region off the direct window until returned.
            let retired = self.last_retire.load(Ordering::Acquire);
            if now.saturating_sub(retired) >= RETIRE_INTERVAL_NS {
                self.last_retire.store(now, Ordering::Release);
                self.node.md.retire_lifted_marks();
            }
        }
    }

    pub fn stop(&self) {
        self.poll();
        if let Some(v12) = &self.v12 {
            v12.stop();
        }
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

/// Where the V12 sits: widget 12 of the crossbar behind the PIC that owns
/// widget 0. On the two- and four-processor board this is QEMU's graphics
/// slot. On the one-processor board the tower guide gives one XIO connector
/// for a VPro board but names neither its widget nor its route; widget 12
/// behind the same PXBOW is inferred, not established.
const V12_WIDGET: u8 = 12;

/// The longest a stopped slot holds a store before letting it go.
const STALL_LIMIT_NS: u64 = 1_000_000_000;

/// One Bedrock PI's graphics flow control: the register block, and the
/// Bedrock's stop below the bias. A counted store that leaves the counter
/// below the bias raises the slot's graphics interrupt and stops the slot:
/// its next counted store waits until the processor has taken the
/// interrupt, software has written the counter, the bias or the interrupt
/// count, or returned credits have lifted the counter above the bias (the
/// SN1 header's credit-counter comment, as QEMU's Bedrock models it). A
/// waiting store is answered busy, so the processor makes it again; it is
/// counted once, when it goes.
///
/// Approximations: whether the edge is below the bias or at it is unknown;
/// below, QEMU's reading. How long the chip holds a store is unknown: after
/// STALL_LIMIT_NS of host time the hold ends and is logged, QEMU's bound. A
/// held processor here retries the store and can take an interrupt between
/// attempts, where the chip holds the store itself.
pub(crate) struct BedrockGfx {
    regs: PiGfx,
    /// When each slot's waiting store was first held.
    held_since: [Option<u64>; PI_GFX_SLOTS],
    clock: Clock,
}

impl BedrockGfx {
    fn new(clock: Clock) -> Self {
        Self { regs: PiGfx::default(), held_since: [None; PI_GFX_SLOTS], clock }
    }

    fn reset(&mut self) {
        self.regs = PiGfx::default();
        self.held_since = [None; PI_GFX_SLOTS];
    }

    fn read(&self, n: u32) -> u64 {
        self.regs.read(n)
    }

    fn write(&mut self, n: u32, value: u64) {
        self.regs.write(n, value);
        let slot = (n / PI_GFX_SLOT_REGS) as usize;
        match n % PI_GFX_SLOT_REGS {
            PI_GFX_CREDIT | PI_GFX_BIAS => self.release_if_above(slot),
            // IRIX's graphics interrupt handler zeroes it.
            PI_GFX_INT_COUNT => self.release(slot),
            _ => {}
        }
    }

    /// A store `slot`'s processor makes at I/O offset `address`.
    fn store(&mut self, slot: usize, address: u64) -> Store {
        if !self.regs.covers(slot, address) {
            return Store::Uncounted;
        }
        if self.regs.stopped[slot] {
            let now = (self.clock)();
            let since = *self.held_since[slot].get_or_insert(now);
            if now.saturating_sub(since) < STALL_LIMIT_NS {
                return Store::Held;
            }
            dlog!(
                LogModule::Bedrock,
                "PI graphics slot {} held for {} ms with the counter at {:#x} and the interrupt not taken; letting the store go",
                slot_name(slot),
                STALL_LIMIT_NS / 1_000_000,
                self.regs.credit[slot]
            );
            self.release(slot);
        }
        self.regs.count_store(slot, address, "Tezro PI");
        let below_bias = self.regs.credit[slot] < self.regs.bias[slot];
        if below_bias {
            self.regs.stopped[slot] = true;
        }
        Store::Counted { below_bias }
    }

    /// The processor took the slot's graphics interrupt.
    fn taken(&mut self, slot: usize) {
        self.release(slot);
    }

    /// Put back exactly the credit a count took. Unlike a returned credit
    /// this never wraps the counter up, since it only undoes a count.
    fn refund(&mut self, slot: usize) {
        self.regs.credit[slot] = (self.regs.credit[slot] + 1) & PI_GFX_CREDIT_MASK;
        self.release_if_above(slot);
    }

    fn add_credits(&mut self, slot: usize, credits: u32) {
        self.regs.add_credits(slot, credits, "Tezro PI");
        self.release_if_above(slot);
    }

    fn release(&mut self, slot: usize) {
        self.regs.stopped[slot] = false;
        self.held_since[slot] = None;
    }

    fn release_if_above(&mut self, slot: usize) {
        if self.regs.credit[slot] > self.regs.bias[slot] {
            self.release(slot);
        }
    }
}

fn slot_name(slot: usize) -> char {
    (b'A' + slot as u8) as char
}

/// A PI's node hooks: its graphics flow-control registers, served from its
/// block, and its real-time registers' part in the boot rendezvous.
struct PiHooks {
    gfx: Arc<Mutex<BedrockGfx>>,
    rendezvous: Weak<Rendezvous>,
}
impl NodeHooks for PiHooks {
    fn rt_compare_written(&self, interface: u8, slot: usize) {
        if let Some(r) = self.rendezvous.upgrade() {
            r.rt_compare_written(interface, slot);
        }
    }
    fn rt_counter_read(&self, _interface: u8) {
        if let Some(r) = self.rendezvous.upgrade() {
            r.rt_counter_read();
        }
    }
    fn gfx_read(&self, index: usize) -> Option<u64> {
        (index < PI_GFX_REGS as usize).then(|| self.gfx.lock().read(index as u32))
    }
    fn gfx_write(&self, index: usize, value: u64) -> bool {
        if index >= PI_GFX_REGS as usize {
            return false;
        }
        self.gfx.lock().write(index as u32, value);
        true
    }
    fn gfx_reset(&self) {
        self.gfx.lock().reset();
    }
    fn gfx_interrupt_taken(&self, slot: usize) {
        self.gfx.lock().taken(slot);
    }
}

/// The node's end of the V12's credit return: a widget's credits go to the
/// interface and slot the node's graphics map names. Only node 0's own
/// interfaces are reachable.
///
/// A head reports a counted store it cannot take into its FIFO yet as owed,
/// and the node keeps a ledger of what each widget owes. A write to a map
/// entry first settles the debt of the widget the entry names, under the
/// entry as it stood, and drops the head's later repayments of it. IRIX's
/// detach zeroes the processor's entry and then saves the counter for the
/// next attach, so a credit arriving after the detach would be lost to every
/// later attach of that flow. This is the QEMU model's choice for a board
/// slower than the real one, not a known property of the I/O interface.
struct NodeCredits {
    io: Weak<BedrockIo>,
    gfx: [Arc<Mutex<BedrockGfx>>; 2],
    owed: Mutex<OwedCredits>,
}
impl NodeCredits {
    /// Credits from `widget`, to the slot `map` names.
    fn route(&self, map: [u64; 2], widget: u8, n: u32) {
        match gfx_route(map, widget) {
            Some(r) if r.node == 0 => self.gfx[r.interface].lock().add_credits(r.slot, n),
            other => dlog!(LogModule::Bedrock, "Tezro: {n} V12 credits for widget {widget} have no local route ({other:?})"),
        }
    }

    fn route_now(&self, widget: u8, n: u32) {
        if let Some(io) = self.io.upgrade() {
            self.route([io.gfx_map(0), io.gfx_map(1)], widget, n);
        }
    }

    /// A store replacing graphics map entry `entry`'s value `old` with `new`.
    fn map_written(&self, entry: usize, old: u64, new: u64) {
        let Some(io) = self.io.upgrade() else { return };
        if let Some((widget, credits)) = self.owed.lock().settle(old, new) {
            let mut map = [io.gfx_map(0), io.gfx_map(1)];
            map[entry] = old;
            self.route(map, widget, credits);
        }
    }
}
impl PiGfxLink for NodeCredits {
    fn return_credits(&self, widget: u32, n: u32) {
        self.route_now(widget as u8, n);
    }
    fn owe_credits(&self, widget: u32, n: i32) {
        self.owed.lock().owe(widget as u8, n);
    }
    fn repay_credits(&self, widget: u32, n: u32) {
        let due = self.owed.lock().repay(widget as u8, n);
        if due != 0 {
            self.route_now(widget as u8, due);
        }
    }
}

/// The credits each of a node's sixteen widgets owes.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct OwedCredits([u32; 16]);

impl OwedCredits {
    /// `n` more credits owed by `widget`; a negative count cancels owed
    /// credits a head reset threw away, never below zero.
    pub fn owe(&mut self, widget: u8, n: i32) {
        let owed = &mut self.0[usize::from(widget & 0xf)];
        *owed = if n >= 0 { owed.saturating_add(n as u32) } else { owed.saturating_sub(n.unsigned_abs()) };
    }

    /// `n` owed credits repaid: what is still owed of them, to route now. A
    /// repayment of credits already settled routes nothing.
    pub fn repay(&mut self, widget: u8, n: u32) -> u32 {
        let owed = &mut self.0[usize::from(widget & 0xf)];
        let repaid = n.min(*owed);
        *owed -= repaid;
        repaid
    }

    /// A store replacing graphics map entry `old` with `new`: the widget
    /// `old` names and the credits it owed, to route under `old` before the
    /// new value takes effect. Rewriting the same value settles nothing.
    pub fn settle(&mut self, old: u64, new: u64) -> Option<(u8, u32)> {
        let widget = (old & 0xf) as u8;
        if old == new || self.0[usize::from(widget)] == 0 {
            return None;
        }
        Some((widget, std::mem::take(&mut self.0[usize::from(widget)])))
    }
}

/// Fit the V12 behind the PXBOW: each processor's stores into its slot's
/// graphics page are counted, the board's returned credits go where the
/// node's graphics map sends them, its interrupt packets and DMA reach
/// Bedrock IO as the PIC's do.
fn fit_v12(
    node: &Node,
    running: Population,
    pxbow: &Arc<Pxbow>,
    gfx: &[Arc<Mutex<BedrockGfx>>; 2],
    status: &Arc<crate::disp::StatusSource>,
    graphics_pool: (usize, u32),
) -> Result<Arc<Odyssey>, String> {
    let io = Arc::downgrade(&node.io);
    let packets: PacketSender = Arc::new(move |widget, address, vector, set| {
        if widget != HOST_WIDGET {
            dlog!(LogModule::Bedrock, "Tezro: V12 interrupt to widget {widget} refused; only Bedrock IO is reachable");
            return false;
        }
        let Some(io) = io.upgrade() else { return false };
        let payload = u64::from(vector) | if set { 1 << 8 } else { 0 };
        io.inbound_write(V12_WIDGET, Some(INBOUND_DEVICE), address, 8, payload) == BUS_OK
    });
    let dma = Arc::new(WidgetDma { io: Arc::downgrade(&node.io), widget: V12_WIDGET });
    let v12 = Odyssey::new_widget_relative(
        V12_WIDGET,
        GraphicsBoard::V12.as_str(),
        packets,
        dma,
        status.clone(),
        graphics_pool,
    )?;
    let credits = Arc::new(NodeCredits { io: Arc::downgrade(&node.io), gfx: gfx.clone(), owed: Mutex::default() });
    v12.set_pi_link(credits.clone());
    let settle = Arc::downgrade(&credits);
    node.io.on_gfx_map_write(Box::new(move |entry, old, new| {
        if let Some(credits) = settle.upgrade() {
            credits.map_written(entry, old, new);
        }
    }));
    pxbow
        .attach_endpoint(V12_WIDGET, v12.clone() as Arc<dyn BusDevice>)
        .map_err(|e| format!("Tezro V12: {e:?}"))?;
    node.io.connect_widget(V12_WIDGET, pxbow.route(V12_WIDGET));
    node.count_io_stores(Box::new(SlotStores { pi: node.pi.clone(), gfx: gfx.clone(), running }));
    Ok(v12)
}

/// Each processor's uncached I/O stores, counted against its own slot, as
/// each store enters the processor's own interface. A thread that runs no
/// processor counts as CPU 0, in PI0 slot A.
struct SlotStores {
    pi: [Arc<BedrockPi>; 2],
    gfx: [Arc<Mutex<BedrockGfx>>; 2],
    running: Population,
}
impl SlotStores {
    fn slot(&self) -> (usize, usize) {
        let (interface, slot) = self.running.slot_of(crate::cpu::smp::current_cpu()).unwrap_or((0, 0));
        (usize::from(interface), slot)
    }
}
impl StoreCounter for SlotStores {
    fn count(&self, address: u64) -> Store {
        let (interface, slot) = self.slot();
        let store = self.gfx[interface].lock().store(slot, address);
        if store == (Store::Counted { below_bias: true }) {
            // The slot's graphics interrupt: bit 1 + slot of bank 0.
            self.pi[interface].set_pending(0, 1 + slot as u8);
        }
        store
    }
    fn refund(&self, _address: u64) {
        let (interface, slot) = self.slot();
        self.gfx[interface].lock().refund(slot);
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
