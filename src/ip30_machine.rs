//! The Octane (IP30) as a machine: iris's CPU thread, monitor and serial
//! console around the IP30 bus and devices (`ip30.rs`).
//!
//! Selected with `[machine] profile = "octane_ip30"` and configured by the
//! shared keys (`prom`, `nvram`, `banks`, the serial ports, `[clock]`) plus
//! `[octane]` (see [`crate::config::OctaneSection`]). Headless only for
//! now: the console is the IOC3's UART A on `serial_port_a`, and the
//! graphics widget is a register file. A service thread pumps the UARTs
//! and HEART's compare between instructions and writes the flash back
//! when the PROM changes its data store.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::config::{CpuModel, MachineConfig};
use crate::cpu::mips_cache_shadow::R10000ShadowCache;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::hptimer::TimerManager;
use crate::ip30::{Ip30Bus, Ip30Options};
use crate::traits::{BusDevice, Device, MachineEvent};
use crate::workstation::{MachineRef, MachineSlot, Workstation};

/// The PROM image tried when `prom` is left at the shared default, named
/// as IRIX ships it in /usr/cpu/firmware. The Octane has no embedded PROM.
pub const OCTANE_PROM_FILE: &str = "IP30prom.bin";

/// CP0 Count when `[clock] fixed_mhz` is unset: half of a 195 MHz R10000.
pub const OCTANE_COUNT_MHZ: f64 = 97.5;

pub struct OctaneMachine {
    pub bus: Arc<Ip30Bus>,
    cpu: Arc<dyn CpuDevice>,
    timer_manager: Arc<TimerManager>,
    running: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
    monitor: Arc<crate::monitor::Monitor>,
    monitor_port: u16,
    /// The console UART's in-process host end in CI mode (`--ci`).
    ci_serial: Option<Arc<crate::dev::z85c30::CiSerialBackend>>,
    events: std::sync::mpsc::SyncSender<MachineEvent>,
    event_rx: Option<std::sync::mpsc::Receiver<MachineEvent>>,
    controller_up: Arc<AtomicBool>,
    controller_slot: Option<MachineSlot>,
    /// Set while the machine is stopped: the service thread leaves the
    /// devices alone.
    paused: Arc<AtomicBool>,
    lifecycle: crate::machine_state::GuestState,
}

impl OctaneMachine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        if cfg.machine.cpu != CpuModel::R10000 {
            return Err("an Octane has an R10000 ([machine] cpu = \"r10000\")".into());
        }
        // `prom`, or IRIX's file name when `prom` is the shared default and
        // that file is not there: there is no embedded Octane PROM.
        let prom_path = if cfg.prom != crate::config::DEFAULT_PROM || std::path::Path::new(&cfg.prom).exists() {
            cfg.prom.clone()
        } else {
            OCTANE_PROM_FILE.to_string()
        };
        let prom = std::fs::read(&prom_path)
            .map_err(|e| format!("Octane PROM {prom_path}: {e} (set `prom` to an IP30 PROM image, the 1 MB flash contents)"))?;
        let banks = [cfg.banks[0], cfg.banks[1], cfg.banks[2], cfg.banks[3]];
        let opts = Ip30Options {
            flash_file: (!cfg.octane.flash.is_empty()).then(|| std::path::PathBuf::from(&cfg.octane.flash)),
            nvram_file: (!cfg.nvram.is_empty()).then(|| std::path::PathBuf::from(&cfg.nvram)),
            mac: cfg.network().mac,
        };
        let bus = Arc::new(Ip30Bus::new(banks, &prom, opts));
        eprintln!(
            "iris: Octane with {} MB ({}), serial console",
            banks.iter().sum::<u32>(),
            banks.iter().filter(|&&b| b != 0).map(|b| format!("{b}")).collect::<Vec<_>>().join("+"),
        );

        // The console UART's host end: in CI mode the in-process backend
        // the CI socket types into, otherwise TCP on `serial_port_a`, either
        // teed to `serial_log` when one is set. UART B serves `serial_port_b`.
        let [port_a, port_b] = cfg.serial_ports();
        let ci_serial = cfg.ci.then(|| Arc::new(crate::dev::z85c30::CiSerialBackend::new()));
        if let Some(ci) = &ci_serial {
            if let Some(path) = cfg.serial_log.as_deref() {
                match crate::dev::serial_log::SerialLog::create(std::path::Path::new(path), cfg.serial_log_timestamps) {
                    Ok(log) => {
                        ci.set_log(log);
                        eprintln!("iris: Octane serial console mirroring to {path}");
                    }
                    Err(e) => eprintln!("iris: Octane serial log {path}: {e}"),
                }
            }
            bus.ioc3.uart_a.set_backend(ci.clone());
        } else {
            let mut backend = crate::dev::z85c30::null_serial_backend();
            match crate::dev::z85c30::tcp_serial_backend(port_a) {
                Some((b, p)) => {
                    eprintln!("iris: Octane serial port 1 (the console) on 127.0.0.1:{p}");
                    backend = b;
                }
                None => eprintln!("iris: Octane serial port 1: port {port_a} unavailable; is another iris using it? (serial_port_a)"),
            }
            if let Some(path) = &cfg.serial_log {
                match crate::dev::serial_log::SerialLog::create(std::path::Path::new(path), cfg.serial_log_timestamps) {
                    Ok(log) => backend = Arc::new(crate::dev::z85c30::TeeBackend::new(backend.clone(), log)),
                    Err(e) => eprintln!("iris: Octane serial log {path}: {e}"),
                }
            }
            bus.ioc3.uart_a.set_backend(backend);
        }
        match crate::dev::z85c30::tcp_serial_backend(port_b) {
            Some((b, p)) => {
                eprintln!("iris: Octane serial port 2 on 127.0.0.1:{p}");
                bus.ioc3.uart_b.set_backend(b);
            }
            None => eprintln!("iris: Octane serial port 2: port {port_b} unavailable; is another iris using it? (serial_port_b)"),
        }

        let timer_manager = Arc::new(TimerManager::new());
        let sysad: Arc<dyn BusDevice> = bus.clone();
        // The R10000's Count ticks at half the pipeline clock; `[clock]
        // fixed_mhz` is Count.
        let count_hz = (cfg.clock.fixed_mhz.unwrap_or(OCTANE_COUNT_MHZ) * 1_000_000.0) as u64;
        let cpu: Arc<dyn CpuDevice> = {
            // The shadow cache model, as the IP28 uses: the Octane's PROM
            // does not keep anything in the cache that memory lacks.
            let cpucfg = MipsCpuConfig::for_model::<R10000ShadowCache>();
            let tlb = MipsTlb::new(cpucfg.tlb_entries);
            let mut exec: MipsExecutor<MipsTlb, R10000ShadowCache> = MipsExecutor::new(sysad, tlb, &cpucfg);
            exec.core.set_timer_manager(timer_manager.clone());
            exec.core.set_count_hz(count_hz.max(1));
            Arc::new(MipsCpu::new(exec))
        };

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
                g.compile_queue.start(bus.clone() as Arc<dyn BusDevice>, stats);
            }
            std::mem::forget(cpu_device);
        }

        // ppmem: the CPU's cache and the JIT reach RAM through the memory's
        // window, publishing the mapped bitmap into the CPU's own fields.
        // Remaps happen on the CPU thread (a MEMCFG write), which is the
        // window's contract.
        {
            use crate::ppmem::MappedMemory;
            let mut ram = bus.ram.lock();
            let sp = ram.space();
            unsafe {
                sp.set_bitmap_sink(cpu.ppmem_bitmap_ptr());
                bus.set_fast_ram(sp.window_base(), cpu.ppmem_bitmap_ptr());
                cpu.set_tcache_window(sp.window_base());
                let bitmap = cpu.tcache_bitmap_ptr();
                if !bitmap.is_null() {
                    sp.set_bitmap_sink2(bitmap);
                }
                #[cfg(feature = "jitv2")]
                cpu.set_tcache_gen_window(sp.gen_window_base());
            }
            // Publish the banks as they stand into the sinks just set.
            ram.remap();
        }

        // HEART's levels are the CPU's hardware interrupt pins.
        bus.connect_cpu_irq(cpu.interrupts_ptr());

        let mut monitor = crate::monitor::Monitor::new();
        monitor.register_device(crate::devlog::init_devlog());
        monitor.register_device(cpu.clone().as_device());
        monitor.register_device(bus.clone());
        let monitor_port = cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT);
        let (events, event_rx) = std::sync::mpsc::sync_channel(4);
        Ok(Self {
            bus,
            cpu,
            timer_manager,
            running: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
            monitor: Arc::new(monitor),
            monitor_port,
            ci_serial,
            events,
            event_rx: Some(event_rx),
            controller_up: Arc::new(AtomicBool::new(false)),
            controller_slot: None,
            paused: Arc::new(AtomicBool::new(false)),
            lifecycle: crate::machine_state::GuestState::Idle,
        })
    }

    pub fn start(&mut self) {
        if let Ok(starting) = self.lifecycle.transition(crate::machine_state::GuestState::Starting) {
            self.lifecycle = starting;
        }
        self.running.store(true, Ordering::SeqCst);
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        let bus = self.bus.clone();
        let running = self.running.clone();
        let paused = self.paused.clone();
        self.threads.push(
            std::thread::Builder::new()
                .name("IP30-Service".into())
                .spawn(move || Self::service_loop(bus, running, paused))
                .unwrap(),
        );
        // CI mode: the harness starts the CPU (`iris-ci start`).
        if self.ci_serial.is_none() {
            self.cpu.clone().as_device().start();
        }
    }

    /// Host input into the UARTs, HEART's compare, and the flash written
    /// back once the guest has stopped changing it.
    fn service_loop(bus: Arc<Ip30Bus>, running: Arc<AtomicBool>, paused: Arc<AtomicBool>) {
        let mut save_check = Instant::now();
        while running.load(Ordering::Relaxed) {
            if paused.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            bus.service();
            if save_check.elapsed() >= Duration::from_secs(2) {
                save_check = Instant::now();
                bus.save();
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    pub fn stop(&mut self) {
        self.cpu.clone().as_device().stop();
        self.paused.store(true, Ordering::SeqCst);
        self.bus.save();
        if self.lifecycle != crate::machine_state::GuestState::Idle {
            self.lifecycle = crate::machine_state::GuestState::Halted;
        }
    }

    pub fn restart_peripherals(&mut self) {
        self.paused.store(false, Ordering::SeqCst);
    }

    /// The hardware as at power-on, with the CPU stopped.
    pub fn reset(&mut self) {
        self.stop();
        crate::traits::Resettable::power_on(&*self.cpu);
        self.bus.power_on();
        self.restart_peripherals();
    }

    pub fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut OctaneMachine as *mut dyn Workstation;
        let slot = self.controller_slot.get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr))));
        debug_assert!(std::ptr::addr_eq(slot.lock().0, ptr), "OctaneMachine moved after its slot was handed out");
        slot.clone()
    }
}

impl Drop for OctaneMachine {
    fn drop(&mut self) {
        if let Some(slot) = self.controller_slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.running.store(false, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        self.monitor.shutdown();
    }
}

/// The Octane as the front ends see it: the essentials, no snapshots yet,
/// no display. The trait's defaults say so.
impl Workstation for OctaneMachine {
    fn start(&mut self) { OctaneMachine::start(self) }
    fn stop(&mut self) { OctaneMachine::stop(self) }
    fn reset(&mut self) { OctaneMachine::reset(self) }
    fn restart_peripherals(&mut self) { OctaneMachine::restart_peripherals(self) }
    fn cpu_start(&self) { self.cpu.clone().as_device().start() }
    fn cpu_is_running(&self) -> bool { self.cpu.clone().as_device().is_running() }
    fn machine_slot(&mut self) -> MachineSlot { OctaneMachine::machine_slot(self) }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self.event_rx.take();
        assert!(events.is_some(), "register_system_controller called twice");
        crate::workstation::register_system_controller(&self.monitor, slot, events);
        self.controller_up.store(true, Ordering::SeqCst);
        let _ = &self.events;
    }
    fn monitor_addr(&self) -> Option<std::net::SocketAddr> { self.monitor.bound_addr() }
    fn lifecycle(&self) -> crate::machine_state::GuestState { self.lifecycle }
    fn set_lifecycle(&mut self, state: crate::machine_state::GuestState) { self.lifecycle = state; }

    fn get_display(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> { None }
    /// Serial console: a keyboard nothing reads.
    fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> { Arc::new(crate::dev::ps2::Ps2Controller::new(None)) }
    fn get_timer_manager(&self) -> Arc<TimerManager> { self.timer_manager.clone() }
    fn get_cpu_debug(&self) -> Arc<dyn crate::gdb_stub::CpuDebug> { self.cpu.clone().debug_adapter() }
    fn fasttick_count(&self) -> Arc<std::sync::atomic::AtomicU64> { self.cpu.fasttick_count() }
    fn get_ci_serial(&self) -> Option<Arc<crate::dev::z85c30::CiSerialBackend>> { self.ci_serial.clone() }
    /// Typed at the console, UART A.
    fn inject_serial_console(&self, bytes: &[u8]) {
        match &self.ci_serial {
            Some(ci) => ci.push_host(bytes),
            None => self.bus.ioc3.uart_a.feed(bytes),
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    /// The DS1687's RAM, which holds the PROM's environment.
    fn save_nvram(&self, _path: Option<&str>) -> Result<String, String> {
        self.bus.ioc3.save_nvram().map(|p| p.display().to_string()).ok_or_else(|| "no nvram file configured".to_string())
    }
}
