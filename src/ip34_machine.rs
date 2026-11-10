//! The Fuel (IP34 board, IP35 software platform): an R14000 on Bedrock,
//! XBridge and the base I/O devices, with Iris's CPU thread, monitor and serial console.
//!
//! Selected with `[machine] profile = "fuel_ip34"`. Shared keys select the
//! PROM, NVRAM, memory and SCSI devices; `[fuel]` selects the serial console
//! and writable flash. `ip35::Ip35Bus` maps the processor views over the
//! same Physical memory banks. The IOC3 carries PS/2 keyboard and mouse ports
//! (default) and the OHCI controller can take a USB pair; without graphics the
//! machine runs on a serial console. With `[graphics] board = "v12"` (or
//! `"v10"`) a VPro board answers at crossbar widget 13, the Fuel's graphics
//! slot.
//!
//! The Bedrock node itself, and its approximations, are in `ip35::node`.
//!
//! # What is approximated here
//!
//! The processor interfaces' graphics flow-control registers store what
//! IRIX writes, which is what its VPro driver checks, but count no stores:
//! no uncached store into the graphics page takes a credit and the board
//! returns none, so the credit-starvation interrupt never fires. The board's
//! DMA reaches memory through Bedrock's inbound path as widget 13.
//!
//! Board composition ported from hw/mips/sgi/sn1-node.c
//! in jamesbraid/qemu-sgi-origami.

use crate::config::{scsi_label, FuelConsole, GraphicsBoard, MachineConfig};
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::bedrock::{Bedrock, Population};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::heart::{host_clock, Clock};
use crate::dev::ioc3::{ByteBus, Ioc3, IP34, IP34_NO_PS2};
use crate::dev::odyssey::{Odyssey, PacketSender};
use crate::dev::ioc3::eth::Ioc3Eth;
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::{L1, L1Config};
use crate::dev::ohci::Ohci;
use crate::dev::prom_image;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::{null_serial_backend, open_log, tcp_serial_backend, CiSerialBackend, LogMode, SerialBackend, TeeBackend};
use crate::dev::timekeeper::Timekeeper;
use crate::dev::xbridge::XBridge;
use crate::pci::{BoardIrq, DevFn};
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::cpu::mips_cache_shadow::R12000ShadowCache;
use crate::cpu::parts::{self, Part};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::ip35::node::{connect_cpus, Node, NodeConfig};
use crate::ip35::{Ip35Bus, CPU_PHYS_MASK};
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
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc};

pub struct FuelMachine {
    board: Arc<FuelBoard>,
    cpu: Arc<dyn CpuDevice>,
    timers: Arc<TimerManager>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci_serial: Option<Arc<CiSerialBackend>>,
    console: usize,
    ci: bool,
    scratch: Option<PathBuf>,
    timekeeper: Arc<Timekeeper>,
    nvram: String,
    frames: Option<Arc<AtomicU64>>,
    nat: Option<(
        Arc<NatControl>,
        (Ipv4Addr, Ipv4Addr, Ipv4Addr),
    )>,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
}

impl FuelMachine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        cfg.validate()?;
        let prom = prom_image::load(Path::new(&cfg.prom), 8 * 1024 * 1024)
            .map_err(|e| format!("Fuel {e}"))?;
        let timekeeper = Arc::new(Timekeeper::with_nvram_file(
            Path::new(&cfg.nvram),
        )?);
        let board = Arc::new(FuelBoard::with_config(FuelBoardConfig {
            payload: &prom,
            clock: host_clock(),
            flash_path: cfg.fuel.flash.as_deref().map(Path::new),
            station_mac: cfg.network().mac,
            timekeeper: timekeeper.clone(),
            ps2_input: cfg.fuel.ps2_input,
            l1: L1Identity {
                revision: cfg.fuel.l1_revision.as_deref().map(crate::dev::l1::parse_revision).transpose()?,
                brick_serial: cfg.fuel.brick_serial.clone(),
            },
            graphics: cfg.graphics_board(),
            graphics_pool: (cfg.graphics.threads, cfg.graphics.tile),
            graphics_raster: (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()),
        })?);
        let timers = Arc::new(TimerManager::new());
        let part = cfg.cpu();
        let mut exec = board.build_cpu(part);
        // The CPU's Compare matches blink the graphics status bar's LED.
        exec.core.fasttick_count = board.fasttick.clone();
        exec.core.set_timer_manager(timers.clone());
        exec.core.set_count_hz(
            cfg.clock
                .fixed_mhz
                .map_or(part.count_hz(), |mhz| (mhz * 1_000_000.0) as u64)
                .max(1),
        );
        exec.core.count_write_samples = true;
        let poll_board = board.clone();
        exec.batch_hook = Some(Box::new(move |_| poll_board.poll()));
        let cpu: Arc<dyn CpuDevice> = Arc::new(MipsCpu::<_, R12000ShadowCache>::new(*exec));
        board.status.set_cpu_cycles(cpu.cycles_ptr());
        board.status.set_count_hz(cpu.count_hz_atomic());
        connect_cpus("Fuel", &board.bedrock, &board.l1, &board.events, std::slice::from_ref(&cpu))?;
        board.bedrock.set_timer_manager(&timers);
        // After the windows above: each compile worker takes the cache's
        // geometry and the CPU model's policies once, as it starts, and
        // without the windows the geometry has no inline load or store path.
        // Workers read plain physical code addresses through the CPU's code
        // bus: the cached bus, plus the reset PROM's uncached-fetch window.
        // The uncached views would decode plain addresses as Hub registers.
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
                let stats = g.stats.clone();
                g.compile_queue.start(board.code_bus(), stats);
            }
            // The compile pool keeps only a weak handle on the CPU.
            std::mem::forget(cpu_device);
        }
        let scratch = Self::attach_scsi(cfg, &board)?;
        let console = match cfg.fuel.console {
            FuelConsole::L1 => 0,
            FuelConsole::Ioc3A => 1,
            FuelConsole::Ioc3B => 2,
        };
        let ci_serial = cfg.ci.then(|| board.serial[console].clone());
        Self::connect_serial(cfg, &board, console)?;
        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(vec![cpu.clone()])));
        monitor.register_device(PerfMonitor::new(
            cpu.running_flag(), cpu.cycles_ptr(), cpu.fasttick_count(), None, None,
        ));
        for device in board.monitor_devices() {
            monitor.register_device(device);
        }
        cpu.register_locks();
        board.scsi.register_locks();
        board.ioc3.register_locks();
        board.xbridge.register_locks();
        board.flash.register_locks();
        board.bedrock.register_locks();
        let (mut nat, mut frames) = (None, None);
        let net = GatewayConfig::from_network(&cfg.network());
        if let Some(link) = Link::with_gateway(net.clone())? {
            frames = Some(link.frames());
            nat = Some((link.nat(), (net.client_ip, net.gateway_ip, net.netmask)));
            board.eth.attach(link);
        }
        let events = board.receiver.lock().take();
        eprintln!(
            "iris: Fuel IP34, {}, 512 MiB, {}, serial console {:?}",
            cfg.cpu().label(),
            if board.vpro.is_some() { cfg.graphics_board().label() } else { "no graphics" },
            cfg.fuel.console
        );
        Ok(Self {
            board,
            cpu,
            timers,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci_serial,
            console,
            ci: cfg.ci,
            scratch,
            timekeeper,
            nvram: cfg.nvram.clone(),
            nat,
            frames,
            events,
            slot: None,
            lifecycle: GuestState::Idle,
        })
    }

    fn attach_scsi(cfg: &MachineConfig, board: &FuelBoard) -> Result<Option<PathBuf>, String> {
        let attached = attach_from_config(cfg, Some("fuel"), &|channel| {
            board.scsi.bus_for_channel(channel)
                .ok_or_else(|| format!("controller {channel} (a Fuel has channels 0 and 1)"))
        })?;
        if let Some(device) = attached.daynaports.first() {
            return Err(format!("{}: kind = \"daynaport\" is not available on the Fuel",
                scsi_label(device.controller, device.id)));
        }
        Ok(attached.scratch)
    }

    fn connect_serial(
        cfg: &MachineConfig,
        board: &FuelBoard,
        console: usize,
    ) -> Result<(), String> {
        let [a, b] = cfg.serial_ports();
        let ports = [cfg.fuel.l1_port.unwrap_or(8882), a, b];
        if ports[0] == ports[1] || ports[0] == ports[2] || ports[1] == ports[2] {
            return Err("Fuel L1 and IOC3 serial ports must be distinct".into());
        }
        let labels = ["L1", "IOC3 A", "IOC3 B"];
        let settings = ["fuel.l1_port", "serial_port_a", "serial_port_b"];
        let mut log_config = cfg.clone();
        for (index, port) in ports.into_iter().enumerate() {
            log_config.serial_log = if index == console {
                cfg.serial_log.clone()
            } else if index == 0 {
                cfg.fuel.l1_log.clone()
            } else {
                None
            };
            let tag = format!("Fuel {}", labels[index]);
            let log = open_log(&log_config, &tag, LogMode::Fresh);
            let backend: Arc<dyn SerialBackend> = if cfg.ci && index == console {
                if let Some(log) = log {
                    board.serial[index].set_log(log);
                }
                board.serial[index].clone()
            } else {
                let mut backend = null_serial_backend();
                match tcp_serial_backend(port) {
                    Some((socket, actual_port)) => {
                        eprintln!("iris: {tag} on 127.0.0.1:{actual_port}");
                        backend = socket;
                    }
                    None => eprintln!(
                        "iris: {tag}: port {port} unavailable; is another iris using it? ({})",
                        settings[index]
                    ),
                }
                match log {
                    Some(log) => Arc::new(TeeBackend::new(backend, log)),
                    None => backend,
                }
            };
            if index == 0 {
                board.l1.set_serial_backend(backend);
            } else {
                board.ioc3.set_serial_backend(index - 1, backend);
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn connect_serial_for_test(cfg: &MachineConfig, board: &FuelBoard, console: usize) -> Result<(), String> {
        Self::connect_serial(cfg, board, console)
    }

}

impl Workstation for FuelMachine {
    fn start(&mut self) {
        self.lifecycle = GuestState::Starting;
        self.monitor
            .clone()
            .start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        if !self.ci {
            self.cpu_start();
        }
    }
    fn stop(&mut self) {
        self.cpu.clone().as_device().stop();
        self.board.stop();
        if let Err(e) = self.timekeeper.flush_nvram() {
            dlog!(LogModule::Bedrock, "Fuel NVRAM: {e}");
        }
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        self.cpu.power_on();
        self.board.bedrock.power_on();
        self.board.l1.power_on();
        self.board.flash.power_on();
        self.board.xbridge.power_on();
        self.board.ioc3.power_on();
        self.board.scsi.power_on();
        self.board.usb.power_on();
        if let Some(vpro) = &self.board.vpro {
            vpro.power_on();
        }
        self.timekeeper.power_on();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        self.board.start();
    }
    fn cpu_start(&self) {
        self.cpu.clone().as_device().start();
    }
    fn cpu_is_running(&self) -> bool {
        self.cpu.clone().as_device().is_running()
    }
    fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut Self as *mut dyn Workstation;
        self.slot
            .get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr))))
            .clone()
    }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self
            .events
            .take()
            .expect("Fuel system controller registered twice");
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
        self.board.vpro.as_ref().map(|vpro| vpro.display())
    }
    fn register_monitor_device(&self, device: Arc<dyn Device>) {
        self.monitor.register_device(device);
    }
    fn get_ps2(&self) -> Arc<Ps2Controller> {
        self.board.ioc3.ps2()
    }
    fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timers.clone()
    }
    fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> {
        self.cpu.clone().debug_adapter()
    }
    fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.cpu.fasttick_count()
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
            return Err("Fuel NVRAM save uses the configured path".into());
        }
        self.timekeeper.flush_nvram()?;
        Ok(self.nvram.clone())
    }
    fn scratch_path(&self) -> Option<&Path> {
        self.scratch.as_deref()
    }
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        Some(self.board.scsi.clone() as Arc<dyn DiscChanger>)
    }
    fn pending_chd_sync_count(&self) -> usize {
        (0..2).map(|channel| self.board.scsi.bus_for_channel(channel).unwrap().pending_chd_sync_count()).sum()
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
        let buses: Vec<_> = (0..2).map(|channel| self.board.scsi.bus_for_channel(channel).unwrap()).collect();
        ScsiBus::sync_chd_disks(&buses, only, progress, cancel)
    }
    fn net_frames(&self) -> Option<Arc<AtomicU64>> { self.frames.clone() }
    fn nat_control(&self) -> Option<Arc<NatControl>> {
        self.nat.as_ref().map(|(n, _)| n.clone())
    }
    fn nat_expected(&self) -> Option<(Ipv4Addr, Ipv4Addr, Ipv4Addr)> {
        self.nat.as_ref().map(|(_, a)| *a)
    }
}

impl Drop for FuelMachine {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        self.monitor.shutdown();
    }
}

/// The processors a Fuel can be built with, the default first: the R14000A
/// the Fuel ships with.
pub const CPUS: &[&Part] = &[&parts::R14000_500];

/// The cache type `with_cpu!` builds for every entry of [`CPUS`]; a test pins
/// it. The board's devices are not generic over the processor, so it names the
/// type once here.
pub(crate) type FuelCpu = MipsExecutor<MipsTlb, R12000ShadowCache>;

/// A row of the PROM's CPU speed table, found by the board-id word's low
/// nibble when its class (bits 15:12) is 4, the Fuel's. The PROM prints
/// "Unknown PIMM_PSC type" for a nibble with no row (0, 5, f) and falls back
/// to zeros. The hub clock is 200 MHz in every row (established, from the
/// PROM's code). That the nibble is the processor module's own code is
/// inferred; the rows say nothing of what else the word holds.
pub(crate) struct SpeedRow {
    pub(crate) nibble: u64,
    pub(crate) mhz: u32,
    pub(crate) l2_kb: u32,
}

/// Every row of the PROM 6.210 table (established). Two nibbles share 500 MHz
/// (4 and 6) and two 600 MHz (7 and 8); the first listed is the one used.
/// The 700 MHz row is the R16000.
pub(crate) const SPEED_ROWS: [SpeedRow; 7] = [
    SpeedRow { nibble: 0x4, mhz: 500, l2_kb: 2048 },
    SpeedRow { nibble: 0x6, mhz: 500, l2_kb: 2048 },
    SpeedRow { nibble: 0x7, mhz: 600, l2_kb: 4096 },
    SpeedRow { nibble: 0x8, mhz: 600, l2_kb: 4096 },
    SpeedRow { nibble: 0x3, mhz: 700, l2_kb: 4096 },
    SpeedRow { nibble: 0xc, mhz: 800, l2_kb: 4096 },
    SpeedRow { nibble: 0x2, mhz: 900, l2_kb: 8192 },
];

/// The class bits 15:12 of the board-id word for a Fuel. The PROM masks them
/// and compares 0x2000, 0x4000, 0x8000 and 0xc000; the kernel compares 4, 8
/// and c (established). OpenBSD names 4 IP35_FUEL.
const FUEL_BOARD_CLASS: u64 = 0x4000;

/// The board-id word of a Fuel built with `part`, or `None` when the PROM's
/// speed table has no row for its clock and secondary cache. The PROM checks
/// the flash record's mode word, clocks, timing bytes and machine type against
/// this row, so the word has to follow the processor.
///
/// Bits 11:4 and 63:16 of a real board's word are unknown; they read zero
/// here and the PROM does not look at them.
pub(crate) const fn board_word(part: &Part) -> Option<u64> {
    let mut i = 0;
    while i < SPEED_ROWS.len() {
        let row = &SPEED_ROWS[i];
        if row.mhz == part.mhz && row.l2_kb == part.l2_kb {
            return Some(FUEL_BOARD_CLASS | row.nibble);
        }
        i += 1;
    }
    None
}

/// The word for the Fuel's only processor, [`CPUS`]`[0]`.
pub(crate) const FUEL_BOARD_ID: u64 = match board_word(&parts::R14000_500) {
    Some(word) => word,
    None => panic!("the Fuel's processor has no row in the PROM's speed table"),
};
/// Two 256 MiB banks, locational banks 0 and 1.
const FUEL_BANKS_MIB: [u32; 2] = [256, 256];


/// Build the Fuel's processor, `part` being the one the config resolved, and
/// set the PROM boot-mode inputs.
pub(crate) fn build_cpu(bus: Arc<Ip35Bus>, part: &Part) -> Box<FuelCpu> {
    let extent = crate::dev::bedrock::RamExtent::new(&FUEL_BANKS_MIB);
    let cfg = MipsCpuConfig::for_model::<R12000ShadowCache>(Arc::new(crate::ip35::memory_map(&extent)))
        .with_prid(part.prid());
    let mut cpu = Box::new(MipsExecutor::new(
        bus as Arc<dyn BusDevice>,
        MipsTlb::new(cfg.tlb_entries),
        &cfg,
    ));
    cpu.phys_mask = CPU_PHYS_MASK;
    // The flash record's mode word selects these boot-mode inputs. A
    // mismatch enters the PROM's error routine before GP is initialized.
    cpu.core.cp0_config = FirmwareRecord::FUEL.reset_config0();
    cpu
}

/// Shared device composition for a single-processor Fuel.
/// Graphics is absent; the serial workstation shares these base I/O devices.
pub(crate) struct FuelBoard {
    pub bus: Arc<Ip35Bus>,
    /// Cacheable memory at plain physical addresses: what the cache fills
    /// from, and the one bus the JIT and its compile pool read code through.
    cached: Arc<dyn BusDevice>,
    /// What the JIT compiles code through: `cached`, and the reset PROM's
    /// uncached-fetch window ([`Ip35Bus::code_view`]).
    #[cfg(feature = "jitv2")]
    code: Arc<dyn BusDevice>,
    pub bedrock: Arc<Bedrock>,
    pub l1: Arc<L1>,
    pub flash: Arc<Flash>,
    /// The node's resets and power events, and their receiver until the
    /// system controller takes it.
    events: Sender<MachineEvent>,
    receiver: Mutex<Option<Receiver<MachineEvent>>>,
    pub xbridge: Arc<XBridge>,
    pub ioc3: Arc<Ioc3>,
    pub eth: Arc<Ioc3Eth>,
    pub scsi: Arc<Isp12160>,
    pub usb: Arc<Ohci>,
    /// The VPro board in the graphics slot, widget 13.
    pub vpro: Option<Arc<Odyssey>>,
    /// The status bar's activity bits and the CPU's Compare matches, which
    /// the graphics board shows.
    pub heartbeat: Arc<AtomicU64>,
    pub fasttick: Arc<AtomicU64>,
    /// What the graphics board's status bar shows: the activity bits, the
    /// Compare matches and the CPU's cycle count and Count rate.
    pub status: Arc<crate::disp::StatusSource>,
    pub serial: [Arc<CiSerialBackend>; 3],
    #[cfg(test)]
    pub irq: Arc<PendingIrq>,
    clock: Clock,
    last_clock: AtomicU64,
    running: AtomicBool,
}

pub(crate) struct FuelBoardConfig<'a> {
    pub payload: &'a [u8],
    pub clock: Clock,
    pub flash_path: Option<&'a Path>,
    pub station_mac: [u8; 6],
    pub timekeeper: Arc<Timekeeper>,
    /// The IOC3's PS/2 keyboard and mouse controller.
    pub ps2_input: bool,
    /// Replacements for the L1's reported identity; unset keeps its defaults.
    pub l1: L1Identity,
    /// `None`, `V12` or `V10`; validation refuses the others.
    pub graphics: GraphicsBoard,
    /// The VPro's fragment pipeline: worker count and tile size (`[graphics]`).
    pub graphics_pool: (usize, u32),
    /// Which unit draws and, for the GPU, the wgpu backend (`[graphics]`).
    pub graphics_raster: (crate::config::RasterUnit, &'a str),
}

/// What a Fuel config may replace in the L1's identity replies.
#[derive(Clone, Debug, Default)]
pub(crate) struct L1Identity {
    pub revision: Option<[u32; 3]>,
    pub brick_serial: Option<String>,
}

impl FuelBoard {
    /// Flush the disks on both channels of the storage controller.
    pub fn flush_disks(&self) -> std::io::Result<()> {
        (0..2).try_for_each(|channel| self.scsi.bus_for_channel(channel).unwrap().flush())
    }

    #[cfg(test)]
    pub fn new(payload: &[u8], clock: Clock) -> Result<Self, String> {
        Self::with_config(FuelBoardConfig {
            payload, clock, flash_path: None,
            station_mac: L1Config::default().mac,
            timekeeper: Arc::new(Timekeeper::new()?),
            ps2_input: true,
            l1: L1Identity::default(),
            graphics: GraphicsBoard::None,
            graphics_pool: (2, 32),
            graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
        })
    }

    pub fn with_config(config: FuelBoardConfig<'_>) -> Result<Self, String> {
        let FuelBoardConfig { payload, clock, flash_path, station_mac, timekeeper, ps2_input, l1: l1_identity, graphics, graphics_pool, graphics_raster } = config;
        let mut l1_config = L1Config { mac: station_mac, ..Default::default() };
        if let Some(revision) = l1_identity.revision {
            l1_config.revision = Some(revision);
        }
        if let Some(serial) = l1_identity.brick_serial {
            l1_config.brick_serial = Some(serial);
        }
        let node = Node::build(NodeConfig {
            name: "Fuel",
            payload,
            clock: clock.clone(),
            flash_path,
            banks_mib: &FUEL_BANKS_MIB,
            processors: Population::SINGLE,
            firmware: FirmwareRecord::FUEL,
            board_id: FUEL_BOARD_ID,
            l1: l1_config,
        })?;
        let Node {
            bus,
            bedrock,
            l1,
            flash,
            cached,
            #[cfg(feature = "jitv2")]
            code,
            events,
            receiver,
        } = node;
        #[cfg(test)]
        let irq = Arc::new(PendingIrq::default());
        #[cfg(test)]
        bedrock.set_cpu_interrupts(0, &irq.0);
        let serial = std::array::from_fn(|_| Arc::new(CiSerialBackend::new()));
        l1.set_serial_backend(serial[0].clone());
        let weak = Arc::downgrade(&bedrock);
        let xbridge = XBridge::new(BEDROCK_PORT, &|widget| {
            let port = Arc::new(Inbound { bedrock: weak.clone(), widget });
            (port.clone() as Arc<dyn BusDevice>, port as Arc<dyn crate::irq::InterruptController>)
        });
        for widget in [0, crate::dev::xbridge::HOST_WIDGET, 14, 15] {
            bedrock.connect_widget(widget, xbridge.widget(widget).unwrap());
        }
        let mac = l1.station_mac();
        let board = if ps2_input { &IP34 } else { &IP34_NO_PS2 };
        let ioc3 = Arc::new(Ioc3::new(board, mac, ByteBus::M48t35(timekeeper)));
        let eth = ioc3.eth().expect("the Fuel's IOC3 fits Ethernet").clone();
        ioc3.set_serial_backend(0, serial[1].clone());
        ioc3.set_serial_backend(1, serial[2].clone());
        let scsi = Arc::new(Isp12160::new(
            Isp12160::new_bus(),
            Isp12160::new_bus(),
        ));
        let heartbeat = Arc::new(AtomicU64::new(0));
        let fasttick = Arc::new(AtomicU64::new(0));
        scsi.set_heartbeat(heartbeat.clone());
        ioc3.set_heartbeat(heartbeat.clone());
        let status = crate::disp::StatusSource::new(heartbeat.clone(), fasttick.clone());
        let vpro = if graphics == GraphicsBoard::None {
            None
        } else {
            // Interrupt packets reach Bedrock as the heads' do; DMA enters
            // it as this widget's inbound traffic.
            let port = Inbound { bedrock: Arc::downgrade(&bedrock), widget: WIDGET_VPRO };
            let packets: PacketSender = Arc::new(move |widget, address, vector, level| {
                use crate::irq::InterruptController;
                widget == BEDROCK_PORT.widget && port.packet(WIDGET_VPRO, address, vector.into(), level)
            });
            let dma = bedrock.widget_dma(WIDGET_VPRO);
            let board = Odyssey::new_widget_relative(
                WIDGET_VPRO, graphics.as_str(), packets, dma, status.clone(), graphics_pool, graphics_raster,
            )?;
            bedrock.connect_widget(WIDGET_VPRO, xbridge.attach_widget(WIDGET_VPRO, board.clone()));
            Some(board)
        };
        let pci = xbridge.bus(15);
        let slot = |device| DevFn { device, function: 0 };
        pci.attach(slot(1), scsi.clone());
        pci.attach(slot(4), ioc3.clone());
        let usb = Ohci::new();
        usb.set_clock(clock.clone());
        pci.attach(slot(5), usb.clone());
        ioc3.connect_superio_interrupt(pci.board_irq(BoardIrq(0)));
        let reset_ioc3 = ioc3.clone();
        let reset_scsi = scsi.clone();
        let reset_usb = usb.clone();
        let reset_vpro = vpro.as_ref().map(Arc::downgrade);
        xbridge.set_reset_output(Arc::new(move |port| {
            if port == 15 {
                reset_ioc3.power_on();
                reset_scsi.power_on();
                reset_usb.power_on();
            }
            // A link reset of the graphics port resets the board behind it.
            if port == WIDGET_VPRO {
                if let Some(vpro) = reset_vpro.as_ref().and_then(|v| v.upgrade()) {
                    vpro.power_on();
                }
            }
        }));
        Ok(Self {
            bus,
            cached,
            #[cfg(feature = "jitv2")]
            code,
            bedrock,
            l1,
            flash,
            events,
            receiver: Mutex::new(Some(receiver)),
            xbridge,
            ioc3,
            eth,
            scsi,
            usb,
            vpro,
            heartbeat,
            fasttick,
            status,
            serial,
            #[cfg(test)]
            irq,
            last_clock: AtomicU64::new(clock()),
            clock,
            running: AtomicBool::new(false),
        })
    }

    pub fn cached_bus(&self) -> Arc<dyn BusDevice> {
        self.cached.clone()
    }

    /// The bus the JIT compiles code through; see [`Ip35Bus::code_view`].
    #[cfg(feature = "jitv2")]
    pub fn code_bus(&self) -> Arc<dyn BusDevice> {
        self.code.clone()
    }

    pub fn build_cpu(&self, part: &Part) -> Box<FuelCpu> {
        let mut cpu = build_cpu(self.bus.clone(), part);
        cpu.cache = R12000ShadowCache::new(self.cached_bus());
        #[cfg(feature = "jitv2")]
        cpu.set_cached_bus(self.code_bus());
        #[cfg(not(feature = "jitv2"))]
        cpu.set_cached_bus(self.cached_bus());
        cpu
    }

    pub fn start(&self) {
        self.last_clock
            .store((self.clock)(), Ordering::Release);
        self.running
            .store(true, Ordering::Release);
        self.bedrock.start();
        self.l1.start();
        self.ioc3.start();
        self.scsi.start();
        self.usb.start();
        if let Some(vpro) = &self.vpro {
            vpro.start();
        }
    }
    /// The board's devices that take monitor commands: the physical map, the
    /// L1, the SCSI controller, the keyboard and mouse (`ps2 type` and
    /// friends) and, when one is fitted, the VPro (`vpro …`).
    pub fn monitor_devices(&self) -> Vec<Arc<dyn Device>> {
        let mut devices: Vec<Arc<dyn Device>> = vec![
            self.bedrock.physical().clone(),
            self.l1.clone(),
            self.scsi.clone(),
            self.ioc3.ps2(),
        ];
        if let Some(vpro) = &self.vpro {
            devices.push(vpro.clone());
        }
        devices
    }

    pub fn poll(&self) {
        if self.running.load(Ordering::Acquire) {
            let now = (self.clock)();
            let previous = self
                .last_clock
                .swap(now, Ordering::AcqRel);
            self.flash.advance_ns(now.saturating_sub(previous));
            self.usb.sync_clock();
        }
    }
    pub fn stop(&self) {
        self.poll();
        self.running
            .store(false, Ordering::Release);
        if let Some(vpro) = &self.vpro {
            vpro.stop();
        }
        self.usb.stop();
        self.scsi.stop();
        self.ioc3.stop();
        self.l1.stop();
        self.bedrock.stop();
    }
}

/// The crossbar widget of the Fuel's graphics slot: a Fuel IRIX inventory
/// lists its VPro under xtalk/13, and OpenBSD finds Odyssey at widget 13.
pub const WIDGET_VPRO: u8 = 13;

impl Drop for FuelBoard {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Bedrock's crossbar port: widget 10, taking interrupt packets at any
/// address (the address picks the processor interface).
const BEDROCK_PORT: crate::xtalk::XtalkHost =
    crate::xtalk::XtalkHost { widget: crate::dev::xbridge::HOST_WIDGET, int_packet: 0, int_packet_mask: 0, vectors: 256 };

/// Bedrock as a widget sending through the XBridge reaches it: the
/// widget's DMA as inbound reads and writes, a doubleword at a time where
/// the transfer allows, and its interrupt packets as inbound doubleword
/// writes to the address they name. Bedrock checks the widget's
/// permissions on each, for every master on the widget.
struct Inbound {
    bedrock: std::sync::Weak<Bedrock>,
    widget: u8,
}

impl Inbound {
    fn access(&self, source: u8, a: u64, size: u8, value: Option<u64>) -> crate::traits::BusRead64 {
        let Some(bedrock) = self.bedrock.upgrade() else { return crate::traits::BusRead64::err() };
        match value {
            Some(v) => crate::traits::BusRead64 { status: bedrock.inbound_write(source, None, a, size, v), data: 0 },
            None => bedrock.inbound_read(source, None, a, size),
        }
    }
}

macro_rules! inbound_access {
    ($r:ident, $w:ident, $t:ty, $out:ident, $n:expr) => {
        fn $r(&self, a: u64) -> crate::traits::$out {
            let r = self.access(self.widget, a, $n, None);
            crate::traits::$out { data: r.data as $t, status: r.status }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.access(self.widget, a, $n, Some(v as u64)).status
        }
    };
}

impl BusDevice for Inbound {
    inbound_access!(read8, write8, u8, BusRead8, 1);
    inbound_access!(read16, write16, u16, BusRead16, 2);
    inbound_access!(read32, write32, u32, BusRead32, 4);
    inbound_access!(read64, write64, u64, BusRead64, 8);
    fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 {
        for (i, word) in buf.iter_mut().enumerate() {
            let r = self.read64(a + 8 * i as u64);
            if r.status != BUS_OK {
                return r.status;
            }
            *word = r.data;
        }
        BUS_OK
    }
    fn write_block(&self, a: u64, buf: &[u64]) -> u32 {
        for (i, &word) in buf.iter().enumerate() {
            let status = self.write64(a + 8 * i as u64, word);
            if status != BUS_OK {
                return status;
            }
        }
        BUS_OK
    }
}

impl crate::irq::InterruptController for Inbound {
    fn set_line(&self, line: u32, _: bool) {
        dlog!(LogModule::Bedrock, "Fuel: widget {} interrupt {line} with no packet address dropped", self.widget);
    }
    fn packet(&self, source: u8, address: u64, line: u32, level: bool) -> bool {
        self.access(source, address, 8, Some(u64::from(line) | if level { 1 << 8 } else { 0 })).status == BUS_OK
    }
}

/// CPU 0's interrupt word, for tests that build the board without a CPU.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct PendingIrq(pub(crate) AtomicU64);
#[cfg(test)]
impl PendingIrq {
    pub fn bits(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }
}
