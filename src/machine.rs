use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::AtomicU64;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;

use crate::config::MachineConfig;
use crate::traits::{BusDevice, Device, Resettable, Saveable, MachineEvent};
use crate::locks::LockMonitor;
use crate::eeprom_93c56::Eeprom93c56;
use crate::physical::Physical;

// Helper for passing *mut Physical into a Send+Sync closure (MEMCFG callback).
// Safety: Physical is Send+Sync, and the Arc keeps it alive for the callback's lifetime.
struct PhysPtr(*mut Physical);
unsafe impl Send for PhysPtr {}
unsafe impl Sync for PhysPtr {}
impl PhysPtr {
    fn get(&self) -> *mut Physical { self.0 }
}
use crate::mem::Memory;
use crate::prom::Prom;
use crate::mc::MemoryController;
use crate::mips_tlb::MipsTlb;
use crate::mips_exec::{MipsExecutor, MipsCpu, MipsCpuConfig};
use crate::mips_cache_v2::R4000Cache;
use crate::hpc3::Hpc3;
use crate::ioc::Ioc;
use crate::monitor::Monitor;
use crate::rex3::Rex3;
use crate::snapshot::Snapshot;
use crate::hptimer::TimerManager;


pub struct Machine {
    cpu: Arc<MipsCpu<MipsTlb, R4000Cache>>,
    _phys: Arc<Physical>, // Keep reference to Physical Bus
    mc: MemoryController,
    hpc3: Hpc3,
    pub interrupts: Arc<AtomicU64>,
    monitor: Arc<Monitor>,
    /// Sender for async machine events (HardReset, PowerOff) from devices.
    pub event_tx: mpsc::SyncSender<MachineEvent>,
    event_rx: Option<mpsc::Receiver<MachineEvent>>,
    timer_manager: Arc<TimerManager>,
}

impl Machine {
    pub fn new(cfg: MachineConfig) -> Self {
        // 0. Shared EEPROM
        let eeprom = Arc::new(Mutex::new(Eeprom93c56::new()));

        // 1. Create all devices first
        // Memory Controller
        let mc = MemoryController::new(eeprom.clone(), true, cfg.banks);

        // RAM banks sized per config. addr_mask is initialized to mem_size-1;
        // remap_banks() updates it via set_addr_mask() when MEMCFG0/1 are written during POST.
        let banks = [
            Memory::new(cfg.banks[0].max(1) as usize),
            Memory::new(cfg.banks[1].max(1) as usize),
            Memory::new(cfg.banks[2].max(1) as usize),
            Memory::new(cfg.banks[3].max(1) as usize),
        ];

        // PROM (1MB at 0x1FC00000)
        let prom = Prom::from_file_or_embedded(&cfg.prom);
        let prom_port = prom.get_port();

        // Shared atomics — created first so all devices and the display thread use the same Arc.
        let heartbeat     = Arc::new(AtomicU64::new(0)); // activity bits: see Rex3::HB_*
        let cycles        = Arc::new(AtomicU64::new(0)); // CPU cycle counter
        let fasttick_count = Arc::new(AtomicU64::new(0)); // CP0 Compare match counter
        let decoded_count = Arc::new(AtomicU64::new(0)); // pre-decoded instruction count
        let l1i_hit_count        = Arc::new(AtomicU64::new(0)); // L1-I hit counter
        let l1i_fetch_count      = Arc::new(AtomicU64::new(0)); // L1-I fetch counter
        let uncached_fetch_count = Arc::new(AtomicU64::new(0)); // uncached instruction fetches

        // HPC3 (512KB at 0x1FB80000)
        let ioc = Ioc::new(true);
        let timer_manager = Arc::new(TimerManager::new());
        ioc.set_timer_manager(timer_manager.clone());
        ioc.set_heartbeat(heartbeat.clone());
        let hpc3 = Hpc3::with_nfs(eeprom.clone(), ioc.clone(), true, heartbeat.clone(), cfg.nfs.clone(), cfg.port_forward.clone(), cfg.no_audio);
        hpc3.set_timer_manager(timer_manager.clone());

        // Attach SCSI devices from config (IDs 1–7).
        let mut scsi_ids: Vec<u8> = cfg.scsi.keys().copied().collect();
        scsi_ids.sort();
        for id in scsi_ids {
            let dev = &cfg.scsi[&id];
            // For CD-ROMs: build ordered disc list; first entry is mounted now.
            // For HDDs: disc list is unused (empty).
            let (path, discs) = if dev.cdrom {
                let mut list = dev.discs.clone();
                if list.is_empty() {
                    list.push(dev.path.clone());
                } else if list[0] != dev.path {
                    // Ensure path is front of list if explicitly set
                    list.insert(0, dev.path.clone());
                }
                (list[0].clone(), list)
            } else {
                (dev.path.clone(), vec![])
            };
            if let Err(e) = hpc3.add_scsi_device(id as usize, &path, dev.cdrom, discs, dev.overlay) {
                println!("Note: Could not attach {} to SCSI ID {}: {}", path, id, e);
            }
        }

        // REX3 Graphics — skipped in headless mode
        let rex3: Option<Arc<Rex3>> = if cfg.headless {
            None
        } else {
            let r = Arc::new(Rex3::new(heartbeat, cycles.clone(), fasttick_count.clone(), decoded_count.clone(), Arc::clone(&l1i_hit_count), Arc::clone(&l1i_fetch_count), Arc::clone(&uncached_fetch_count)));
            // Connect VBlank interrupt to IOC
            let ioc_clone = ioc.clone();
            r.set_vblank_callback(Arc::new(move |active| {
                ioc_clone.set_interrupt(crate::ioc::IocInterrupt::VerticalRetrace, active);
            }));
            Some(r)
        };

        // VINO (Video-In, No Out) — GIO64 at 0x1F080000
        let vino = crate::vino::Vino::new();
        {
            struct VinoIrqAdapter { ioc: crate::ioc::Ioc }
            impl crate::vino::VinoIrq for VinoIrqAdapter {
                fn set_interrupt(&self, active: bool) {
                    self.ioc.set_interrupt(crate::ioc::IocInterrupt::VideoVsync, active);
                }
            }
            vino.set_irq(Arc::new(VinoIrqAdapter { ioc: ioc.clone() }));
        }

        // 2. Create Physical Bus with devices
        let phys_raw = Physical::new(
            banks,
            rex3,
            vino,
            mc.clone(),
            hpc3.clone(),
            prom_port,
        );

        // Wrap Physical in Arc
        let phys = Arc::new(phys_raw);

        // Initialize device map now that Physical is in final location
        // SAFETY: We have exclusive access since Arc was just created and not shared yet
        unsafe {
            let phys_ptr = Arc::as_ptr(&phys) as *mut Physical;
            (*phys_ptr).init();
        }

        // Connect Physical to MC (for VDMA)
        mc.set_phys(phys.clone());
        mc.set_ioc(ioc.clone());

        // Wire MEMCFG callback: when MC writes MEMCFG0/1, remap banks in Physical.
        // SAFETY: Physical is pinned in Arc; remap_banks(&mut self) is only invoked
        // from the CPU thread (same thread that writes MEMCFG), never concurrently.
        {
            let phys_ptr = PhysPtr(Arc::as_ptr(&phys) as *mut Physical);
            mc.set_memcfg_callback(Box::new(move |addrs| {
                unsafe { (*phys_ptr.get()).remap_banks(addrs); }
            }));
        }

        // Fire initial remap using MC's boot-time MEMCFG values
        {
            let phys_ptr = Arc::as_ptr(&phys) as *mut Physical;
            let (memcfg0, memcfg1) = mc.get_memcfg();
            let addrs = mc.parse_memcfg(memcfg0, memcfg1);
            unsafe { (*phys_ptr).remap_banks(addrs); }
        }
        
        // Connect HPC3 to System Memory (via Physical)
        hpc3.set_phys(phys.clone());

        // Connect VINO to System Memory and start its DMA thread
        phys.vino.set_phys(phys.clone());
        phys.vino.start();

        // 5. CPU config + TLB + Executor
        let cfg = MipsCpuConfig::indy();
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let sysad: Arc<dyn BusDevice> = phys.clone();
        let mut executor: MipsExecutor<MipsTlb, R4000Cache> = MipsExecutor::new(sysad, tlb, &cfg);

        // Load default symbol maps if they exist
        {
            let mut symbols = executor.symbols.lock();
            if let Ok(count) = symbols.load("prom.map") {
                println!("Loaded {} symbols from prom.map", count);
            }
            if let Ok(count) = symbols.load("unix.map") {
                println!("Loaded {} symbols from unix.map", count);
            }
        }

        // Inject the shared cycles and fasttick_count Arcs into the executor core before wrapping in MipsCpu.
        executor.core.cycles = cycles;
        executor.core.fasttick_count = fasttick_count;
        executor.decoded_count       = decoded_count;
        executor.uncached_fetch_count = Arc::clone(&uncached_fetch_count);
        executor.cache.l1i_hit_count   = Arc::clone(&l1i_hit_count);
        executor.cache.l1i_fetch_count = Arc::clone(&l1i_fetch_count);
        // Re-sync raw pointers after Arc injection (the Arcs above replaced the ones captured in new()).
        executor.rebind_atomic_ptrs();

        // Share count_step_atomic from MipsCore with Rex3 so the refresh thread can display it.
        #[cfg(feature = "developer")]
        if let Some(rex3) = &phys.rex3 { rex3.set_count_step_atomic(Arc::clone(&executor.core.count_step_atomic)); }

        let cpu = Arc::new(MipsCpu::new(executor));
        let interrupts = cpu.interrupts.clone();

        // Connect CPU to MC and IOC for signaling
        let cpu_device: Arc<dyn Device> = cpu.clone();
        mc.set_cpu(Arc::downgrade(&cpu_device));
        ioc.set_interrupts(interrupts.clone());

        // Setup DevLog (must be before Monitor so log command is available)
        let devlog = crate::devlog::init_devlog();

        // Setup Monitor
        let mut monitor = Monitor::new();
        monitor.register_device(devlog.clone());
        monitor.register_device(cpu.clone());
        monitor.register_device(Arc::new(mc.clone()));
        monitor.register_device(Arc::new(hpc3.clone()));
        monitor.register_device(phys.clone());
        if let Some(rex3) = &phys.rex3 { monitor.register_device(rex3.clone()); }
        monitor.register_device(Arc::new(phys.vino.clone()));
        let monitor = Arc::new(monitor);

        // Register lock monitor device and all component locks
        {
            use crate::locks::register_lock_fn;
            let ep = eeprom.clone();
            register_lock_fn("mc::eeprom", move || ep.is_locked());
            mc.register_locks();
            hpc3.register_locks();
            if let Some(rex3) = &phys.rex3 { rex3.register_locks(); }
            cpu.register_locks();
        }
        {
            let monitor_ptr = Arc::as_ptr(&monitor) as *mut Monitor;
            unsafe { (*monitor_ptr).register_device(Arc::new(LockMonitor)); }
        }

        let (event_tx, event_rx) = mpsc::sync_channel::<MachineEvent>(4);

        // Give MC and IOC async event senders so they can request hard-reset / power-off.
        mc.set_event_sender(event_tx.clone());
        ioc.set_event_sender(event_tx.clone());

        Self {
            cpu,
            _phys: phys,
            mc,
            hpc3,
            interrupts,
            monitor,
            event_tx,
            event_rx: Some(event_rx),
            timer_manager,
        }
    }

    pub fn start(&mut self) {
        // Start peripherals
        self.mc.start();
        self.hpc3.start();
        if let Some(rex3) = &self._phys.rex3 { rex3.start(); }

        // Start monitor server on localhost:8888
        self.monitor.clone().start_server("127.0.0.1:8888".to_string());
        #[cfg(not(any(debug_assertions, feature = "developer")))]
        self.cpu.start();
    }

    /// Register a SystemController with the monitor so that `reset`, `save`,
    /// and `load` commands work. Must be called after `Machine::new()` while
    /// `self` is in its final stack location (i.e. before any moves).
    /// Also starts the machine event dispatch thread (HardReset, PowerOff).
    pub fn register_system_controller(&mut self) {
        // SAFETY: Machine lives for the entire process lifetime (stack in main).
        // SystemController stops all threads before mutating machine state.
        // The monitor serializes connections via its devices Mutex.
        let ptr = self as *const Machine as *mut Machine;
        let machine_arc = Arc::new(Mutex::new(ptr));
        let ctrl = Arc::new(SystemController {
            machine: machine_arc.clone(),
        });
        // We need interior mutability to register after construction.
        // Monitor::register_device takes &mut self, so we use unsafe to call it.
        // SAFETY: This is called once, before the monitor server thread starts,
        // while we have exclusive access to Machine.
        let monitor_ptr = Arc::as_ptr(&self.monitor) as *mut Monitor;
        unsafe {
            (*monitor_ptr).register_device(ctrl.clone());
        }

        // Spawn the event dispatch thread: receives MachineEvent from devices and
        // performs the requested system-level action.
        // Uses the same SystemController (which is Send+Sync via unsafe impls) so it
        // can stop all threads and mutate machine state safely.
        if let Some(rx) = self.event_rx.take() {
            thread::Builder::new().name("machine-events".to_string()).spawn(move || {
                while let Ok(event) = rx.recv() {
                    let _ = ctrl.with_machine(|machine| {
                        match event {
                            MachineEvent::HardReset => {
                                println!("Machine: SIN hard reset");
                                machine.reset();
                                machine.cpu.start();
                            }
                            MachineEvent::PowerOff => {
                                println!("Machine: soft power-off");
                                machine.stop();
                                #[cfg(not(feature = "developer"))]
                                std::process::exit(0);
                            }
                        }
                        Ok(())
                    });
                }
            }).unwrap();
        }
    }

    pub fn stop(&mut self) {
        self.cpu.stop();
        if let Some(rex3) = &self._phys.rex3 { rex3.stop(); }
        self.hpc3.stop();
        self.mc.stop();
    }

    pub fn run_console_client() {
        println!("IRIS: Irresponsible Rust Irix Simulator");
        println!("Connecting to monitor socket...");

        let mut stream = loop {
            match TcpStream::connect("127.0.0.1:8888") {
                Ok(s) => break s,
                Err(_) => {
                    thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
            }
        };

        let mut socket_reader = stream.try_clone().unwrap();
        thread::spawn(move || {
            let mut buf = [0u8; 1024];
            loop {
                match socket_reader.read(&mut buf) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        print!("{}", String::from_utf8_lossy(&buf[0..n]));
                        io::stdout().flush().unwrap();
                    }
                    Err(_) => break,
                }
            }
            std::process::exit(0);
        });

        let stdin = io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            if stdin.read_line(&mut line).is_err() {
                break;
            }
            if stream.write_all(line.as_bytes()).is_err() {
                break;
            }
        }
    }

    pub fn get_ps2(&self) -> Arc<crate::ps2::Ps2Controller> {
        self.hpc3.ioc().ps2()
    }

    pub fn get_rex3(&self) -> Option<Arc<crate::rex3::Rex3>> {
        self._phys.rex3.clone()
    }

    pub fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timer_manager.clone()
    }

    /// Restart peripherals (MC, HPC3, REX3) without restarting the monitor server.
    fn restart_peripherals(&mut self) {
        self.mc.start();
        self.hpc3.start();
        if let Some(rex3) = &self._phys.rex3 { rex3.start(); }
    }

    /// Helper to power-on reset all devices.
    /// Must be called with threads stopped.
    fn power_on_devices(&mut self) {
        self.cpu.power_on();
        self._phys.reset_memory();
        self.mc.power_on();
        self.hpc3.ioc().power_on();
        // SCC: clears channel regs; backend socket kept alive so console survives.
        self.hpc3.ioc().scc().power_on();
        // PIT: zeroes all channel registers.
        self.hpc3.ioc().pit().power_on();
        // PS2: reset state
        self.hpc3.ioc().ps2().power_on();
        // RTC: battery-backed, no-op.
        self.hpc3.rtc().power_on();
        // EEPROM: non-volatile, no-op.
        self.hpc3.eeprom().lock().power_on();
        // SCSI: execute hardware reset sequence.
        self.hpc3.scsi().power_on();
        // Seeq/Ethernet: reset regs + signal NAT flush.
        self.hpc3.seeq().power_on();
        // HAL2: reset all audio registers and channel state (timers already stopped).
        if let Some(hal2) = self.hpc3.hal2() { hal2.power_on(); }
        self.hpc3.power_on();
        if let Some(rex3) = &self._phys.rex3 { rex3.power_on(); }
    }

    /// Stop all threads, power-on reset every device in-place, restart peripherals.
    /// The CPU is left stopped — the monitor `run` command (or debugger) should start it.
    pub fn reset(&mut self) {
        self.stop();

        self.power_on_devices();

        // Restart peripherals (not monitor — it stays alive)
        self.restart_peripherals();
    }

    /// Save full machine snapshot to `saves/<name>/`.
    pub fn save_snapshot(&mut self, name: &str) -> Result<(), String> {
        self.stop();

        let dir = std::path::PathBuf::from("saves").join(name);
        let snap = Snapshot::new(&dir);
        snap.ensure_dir().map_err(|e| e.to_string())?;

        // CPU + TLB
        let cpu_toml = self.cpu.save_state();
        snap.write_toml("cpu.toml", &cpu_toml).map_err(|e| e.to_string())?;

        // Memory Controller
        let mc_toml = self.mc.save_state();
        snap.write_toml("mc.toml", &mc_toml).map_err(|e| e.to_string())?;

        // IOC
        let ioc_toml = self.hpc3.ioc().save_state();
        snap.write_toml("ioc.toml", &ioc_toml).map_err(|e| e.to_string())?;

        // SCC (Z85C30 serial)
        let scc_toml = self.hpc3.ioc().scc().save_state();
        snap.write_toml("scc.toml", &scc_toml).map_err(|e| e.to_string())?;

        // PIT (8254 timer)
        let pit_toml = self.hpc3.ioc().pit().save_state();
        snap.write_toml("pit.toml", &pit_toml).map_err(|e| e.to_string())?;

        // PS2
        let ps2_toml = self.hpc3.ioc().ps2().save_state();
        snap.write_toml("ps2.toml", &ps2_toml).map_err(|e| e.to_string())?;

        // RTC (DS1x86)
        let rtc_toml = self.hpc3.rtc().save_state();
        snap.write_toml("rtc.toml", &rtc_toml).map_err(|e| e.to_string())?;

        // EEPROM (93C56)
        let eeprom_toml = self.hpc3.eeprom().lock().save_state_owned();
        snap.write_toml("eeprom.toml", &eeprom_toml).map_err(|e| e.to_string())?;

        // SCSI (WD33C93A)
        let scsi_toml = self.hpc3.scsi().save_state();
        snap.write_toml("scsi.toml", &scsi_toml).map_err(|e| e.to_string())?;

        // Seeq8003 (Ethernet)
        let seeq_toml = self.hpc3.seeq().save_state();
        snap.write_toml("seeq.toml", &seeq_toml).map_err(|e| e.to_string())?;

        // HPC3
        let hpc3_toml = self.hpc3.save_state();
        snap.write_toml("hpc3.toml", &hpc3_toml).map_err(|e| e.to_string())?;

        // REX3
        if let Some(rex3) = &self._phys.rex3 {
            let rex3_toml = rex3.save_state();
            snap.write_toml("rex3.toml", &rex3_toml).map_err(|e| e.to_string())?;
            rex3.save_framebuffers(&snap.dir).map_err(|e| e.to_string())?;
        }

        // Bulk memory (raw binary, big-endian word layout) — 4 × 128MB banks
        for i in 0..4 {
            self._phys.save_bank(i, dir.join(format!("bank{}.bin", i))).map_err(|e| e.to_string())?;
        }

        self.restart_peripherals();
        println!("Snapshot saved to saves/{}", name);
        Ok(())
    }

    /// Restore full machine snapshot from `saves/<name>/`.
    pub fn load_snapshot(&mut self, name: &str) -> Result<(), String> {
        self.stop();

        // Reset to clean state before loading
        self.power_on_devices();

        let dir = std::path::PathBuf::from("saves").join(name);
        let snap = Snapshot::new(&dir);

        // CPU + TLB
        let cpu_toml = snap.read_toml("cpu.toml").map_err(|e| e.to_string())?;
        self.cpu.load_state(&cpu_toml)?;

        // Memory Controller
        let mc_toml = snap.read_toml("mc.toml").map_err(|e| e.to_string())?;
        self.mc.load_state(&mc_toml)?;

        // IOC
        let ioc_toml = snap.read_toml("ioc.toml").map_err(|e| e.to_string())?;
        self.hpc3.ioc().load_state(&ioc_toml)?;

        // SCC (Z85C30 serial)
        let scc_toml = snap.read_toml("scc.toml").map_err(|e| e.to_string())?;
        self.hpc3.ioc().scc().load_state(&scc_toml)?;

        // PIT (8254 timer)
        let pit_toml = snap.read_toml("pit.toml").map_err(|e| e.to_string())?;
        self.hpc3.ioc().pit().load_state(&pit_toml)?;

        // PS2
        let ps2_toml = snap.read_toml("ps2.toml").map_err(|e| e.to_string())?;
        self.hpc3.ioc().ps2().load_state(&ps2_toml)?;

        // RTC (DS1x86)
        let rtc_toml = snap.read_toml("rtc.toml").map_err(|e| e.to_string())?;
        self.hpc3.rtc().load_state(&rtc_toml)?;

        // EEPROM (93C56)
        let eeprom_toml = snap.read_toml("eeprom.toml").map_err(|e| e.to_string())?;
        self.hpc3.eeprom().lock().load_state_mut(&eeprom_toml)?;

        // SCSI (WD33C93A)
        let scsi_toml = snap.read_toml("scsi.toml").map_err(|e| e.to_string())?;
        self.hpc3.scsi().load_state(&scsi_toml)?;

        // Seeq8003 (Ethernet)
        let seeq_toml = snap.read_toml("seeq.toml").map_err(|e| e.to_string())?;
        self.hpc3.seeq().load_state(&seeq_toml)?;

        // HPC3
        let hpc3_toml = snap.read_toml("hpc3.toml").map_err(|e| e.to_string())?;
        self.hpc3.load_state(&hpc3_toml)?;

        // REX3
        if let Some(rex3) = &self._phys.rex3 {
            let rex3_toml = snap.read_toml("rex3.toml").map_err(|e| e.to_string())?;
            rex3.load_state(&rex3_toml)?;
            rex3.load_framebuffers(&snap.dir).map_err(|e| e.to_string())?;
        }

        // Bulk memory — 4 × 128MB banks
        for i in 0..4 {
            self._phys.load_bank(i, dir.join(format!("bank{}.bin", i))).map_err(|e| e.to_string())?;
        }

        self.restart_peripherals();
        println!("Snapshot loaded from saves/{}", name);
        Ok(())
    }
}

// ---- SystemController — registers reset/save/load with the monitor ----

/// A thin monitor device that wraps the machine behind a Mutex so the monitor
/// thread can issue system-level commands (reset, save, load).
pub struct SystemController {
    machine: Arc<Mutex<*mut Machine>>,
}

// SAFETY: Machine is only accessed from the monitor thread (one connection at
// a time, serialized) and all CPU/peripheral threads are stopped before any
// state mutation in reset/save/load.
unsafe impl Send for SystemController {}
unsafe impl Sync for SystemController {}

impl SystemController {
    fn with_machine<F: FnOnce(&mut Machine) -> Result<(), String>>(&self, f: F) -> Result<(), String> {
        let mut guard = self.machine.lock();
        let machine = unsafe { &mut **guard };
        f(machine)
    }
}

impl Device for SystemController {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { false }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("machine-stop".to_string(),  "Stop CPU and all peripherals".to_string()),
            ("machine-start".to_string(), "Start CPU and all peripherals".to_string()),
            ("reset".to_string(),         "Reset all hardware to power-on state".to_string()),
            ("save".to_string(),          "save <name> — Save snapshot to saves/<name>/".to_string()),
            ("load".to_string(),          "load <name> — Load snapshot from saves/<name>/".to_string()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        match cmd {
            "machine-stop" => {
                let _ = writeln!(writer, "Stopping machine...");
                self.with_machine(|m| { m.stop(); Ok(()) })
            }
            "machine-start" => {
                let _ = writeln!(writer, "Starting machine...");
                self.with_machine(|m| {
                    m.restart_peripherals();
                    m.cpu.start();
                    Ok(())
                })
            }
            "reset" => {
                let _ = writeln!(writer, "Resetting machine...");
                self.with_machine(|m| { m.reset(); Ok(()) })
            }
            "save" => {
                let name = args.first().ok_or_else(|| "Usage: save <name>".to_string())?;
                let _ = writeln!(writer, "Saving snapshot '{}'...", name);
                self.with_machine(|m| m.save_snapshot(name))
            }
            "load" => {
                let name = args.first().ok_or_else(|| "Usage: load <name>".to_string())?;
                let _ = writeln!(writer, "Loading snapshot '{}'...", name);
                self.with_machine(|m| m.load_snapshot(name))
            }
            _ => Err(format!("Unknown command: {}", cmd)),
        }
    }
}