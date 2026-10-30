//! The Fuel (IP34 board, IP35 software platform): an R14000 on Bedrock,
//! XBridge and the base I/O devices, with Iris's CPU thread, monitor and serial console.
//!
//! Selected with `[machine] profile = "fuel_ip34"`. Shared keys select the
//! PROM, NVRAM, memory and SCSI devices; `[fuel]` selects the serial console
//! and writable flash. `ip35::Ip35Bus` maps the processor views over the
//! same Physical memory banks. Graphics is not attached, and the IOC3 has
//! no keyboard or mouse ports.
//!
//! The Bedrock node itself, and its approximations, are in `ip35::node`.
//!
//! Board composition ported from hw/mips/sgi/sn1-node.c
//! in jamesbraid/qemu-sgi-origami.

use crate::config::{scsi_label, FuelConsole, MachineConfig};
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::bedrock::Bedrock;
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::heart::{host_clock, Clock};
use crate::dev::ioc3::{Ioc3, Ioc3Config};
use crate::dev::ioc3_eth::Ioc3Eth;
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
        let timekeeper = Arc::new(Timekeeper::fuel_with_nvram_file(
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
        })?);
        let timers = Arc::new(TimerManager::new());
        let part = cfg.cpu();
        let mut exec = board.build_cpu(part);
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
        connect_cpus("Fuel", &board.bedrock, &board.l1, &board.events, std::slice::from_ref(&cpu))?;
        board.bedrock.set_timer_manager(&timers);
        // After the windows above: each compile worker takes the cache's
        // geometry and the CPU model's policies once, as it starts, and
        // without the windows the geometry has no inline load or store path.
        // Workers read plain physical code addresses through the CPU's cached
        // bus, which serves them; the uncached views would decode them as Hub
        // registers.
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
                g.compile_queue.start(board.cached_bus(), stats);
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
        monitor.register_device(board.bedrock.physical().clone());
        monitor.register_device(board.l1.clone());
        monitor.register_device(board.scsi.clone());
        // `ps2 type` and friends: the keyboard and mouse from the monitor.
        monitor.register_device(board.ioc3.ps2());
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
            "iris: Fuel IP34, {}, 512 MiB, serial console {:?}",
            cfg.cpu().label(), cfg.fuel.console
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
        self.board.eth.power_on();
        self.board.scsi.power_on();
        self.board.usb.power_on();
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
        None
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

/// The board-id word. Bits 15:12 are the Fuel's product code, 4, as OpenBSD
/// reads them (IP35_FUEL, sys/arch/sgi/sgi/ip27.h). The low bits are chosen
/// to match what PROM 6.210 expects, not read from a board: with them and the
/// flash record's machine type 2 the PROM's CPU configuration check passes,
/// as on a real Fuel. A real board's bits 11:0 are unknown.
const FUEL_BOARD_ID: u64 = 0x4004;
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
            timekeeper: Arc::new(Timekeeper::fuel()?),
            ps2_input: true,
            l1: L1Identity::default(),
        })
    }

    pub fn with_config(config: FuelBoardConfig<'_>) -> Result<Self, String> {
        let FuelBoardConfig { payload, clock, flash_path, station_mac, timekeeper, ps2_input, l1: l1_identity } = config;
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
            firmware: FirmwareRecord::FUEL,
            board_id: FUEL_BOARD_ID,
            l1: l1_config,
        })?;
        let Node { bus, bedrock, l1, flash, cached, events, receiver } = node;
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
        let mut ioc3_config = Ioc3Config::fuel(mac);
        ioc3_config.keyboard_mouse = ps2_input;
        let ioc3 = Arc::new(Ioc3::new(ioc3_config));
        let eth = Ioc3Eth::new_fuel(mac);
        ioc3.attach_eth(eth.clone());
        ioc3.attach_timekeeper(timekeeper);
        ioc3.set_serial_backend(0, serial[1].clone());
        ioc3.set_serial_backend(1, serial[2].clone());
        let scsi = Arc::new(Isp12160::new(
            Isp12160::new_bus(),
            Isp12160::new_bus(),
        ));
        let pci = xbridge.bus(15);
        let slot = |device| DevFn { device, function: 0 };
        pci.attach(slot(1), scsi.clone());
        pci.attach(slot(4), ioc3.clone());
        let usb = Ohci::new();
        usb.set_clock(clock.clone());
        pci.attach(slot(5), usb.clone());
        ioc3.connect_superio_interrupt(pci.board_irq(BoardIrq(0)));
        let reset_ioc3 = ioc3.clone();
        let reset_eth = eth.clone();
        let reset_scsi = scsi.clone();
        let reset_usb = usb.clone();
        xbridge.set_reset_output(Arc::new(move |port| {
            if port == 15 {
                reset_ioc3.power_on();
                reset_eth.power_on();
                reset_scsi.power_on();
                reset_usb.power_on();
            }
        }));
        Ok(Self {
            bus,
            cached,
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

    pub fn build_cpu(&self, part: &Part) -> Box<FuelCpu> {
        let mut cpu = build_cpu(self.bus.clone(), part);
        cpu.cache = R12000ShadowCache::new(self.cached_bus());
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
        self.eth.start();
        self.scsi.start();
        self.usb.start();
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
        self.usb.stop();
        self.scsi.stop();
        self.eth.stop();
        self.ioc3.stop();
        self.l1.stop();
        self.bedrock.stop();
    }
}

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
