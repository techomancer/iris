use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::AtomicU64;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;

use crate::devices::DeviceModel as Card;
use crate::config::{GraphicsBoard, MachineConfig, MachineProfile, NetworkConfig, ScsiDeviceConfig};
use crate::machine_state::GuestState;
use crate::traits::{BusDevice, Device, Resettable, Saveable, MachineEventQueue, MachineEvents, MachineFatal};
use crate::locks::LockMonitor;
use crate::dev::eeprom_93c56::Eeprom93c56;
use crate::physical::Physical;
use crate::ip22_bus::{Ip22Bus, Ip22Devices};

// Helper for passing *mut Physical into a Send+Sync closure (MEMCFG callback).
// Safety: Physical is Send+Sync, and the Arc keeps it alive for the callback's lifetime.
use crate::physical::RamBank;
use crate::dev::prom::Prom;
use crate::dev::mc::MemoryController;

/// The processors an Indy or `indigo2_ip22` shipped with, the default first.
/// The R10000 Indigo2 is the IP28 board.
pub const CPUS: &[&Part] = &[&parts::R4400_150, &parts::R5000_180_EARLY];

/// The IP28's.
pub const IP28_CPUS: &[&Part] = &[&parts::R10000_195_EARLY];
/// The IP28's SysClk divisor as R10000 Config.EC encodes it, and its value.
const IP28_SYSCLK_EC: u32 = 7;
const IP28_SYSCLK_DIV: u64 = 4;
use crate::cpu::mips_tlb::MipsTlb;
use crate::cpu::mips_exec::{MipsExecutor, MipsCpu, MipsCpuConfig, MipsCpuDebugAdapter};
use crate::gdb_stub::CpuDebug;
use crate::cpu::parts::{self, Part};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::dev::hpc3::Hpc3;
use crate::dev::ioc::{Ioc, GioSlot, GIO_SLOT_MAP, profile_idx};
use crate::monitor::Monitor;
use crate::dev::ng1::rex3::Rex3;
use crate::snapshot::{Snapshot, Manifest, SCHEMA_VERSION, ChunksManifest, DiskRef, NamedChunk, enabled_features};
use crate::chunk_store::{ChunkStore, get_chunks_as_words, put_words_as_chunks};
use crate::hptimer::TimerManager;
use crate::workstation::{MachineRef, MachineSlot, SystemController, Workstation};

/// The profile of the machine this process is running, set by
/// [`crate::workstation::build`] so [`emulator_name`] can pick a model word
/// that matches it. Stored as the `MachineProfile` discriminant; `u8::MAX`
/// means no machine has been built yet.
static EMULATOR_PROFILE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);

/// Record the profile of the machine being built (see [`emulator_name`]).
pub fn set_emulator_profile(profile: MachineProfile) {
    EMULATOR_PROFILE.store(profile as u8, std::sync::atomic::Ordering::Relaxed);
}

pub fn emulator_profile() -> Option<MachineProfile> {
    let raw = EMULATOR_PROFILE.load(std::sync::atomic::Ordering::Relaxed);
    MachineProfile::EVERY.into_iter().find(|p| *p as u8 == raw)
}

/// The words an `I` in the expansion may stand for, per machine. Each list
/// names the machine as the user configured it, so the title never calls an
/// Indigo2 an Indy or an O2 anything but IP32. Without a profile only the
/// model-neutral words remain.
fn emulator_name_thirds(profile: Option<MachineProfile>) -> &'static [&'static str] {
    match profile {
        Some(MachineProfile::IndyIp24) => &["IRIX", "Iris", "IPsomething", "Indy", "IP24"],
        Some(MachineProfile::Indigo2Ip22) => &["IRIX", "Iris", "IPsomething", "Indigo2", "IP22"],
        Some(MachineProfile::Indigo2Ip28) => &["IRIX", "Iris", "IPsomething", "Indigo2", "IP28"],
        Some(MachineProfile::O2Ip32) => &["IRIX", "Iris", "IPsomething", "IP32"],
        Some(MachineProfile::OctaneIp30) => &["IRIX", "Iris", "IPsomething", "IP30"],
        Some(MachineProfile::Origin200Ip29) => &["IRIX", "Iris", "IPsomething", "IP29"],
        Some(MachineProfile::Origin2000Ip27 | MachineProfile::Onyx2Ip27) => &["IRIX", "Iris", "IPsomething", "IP27"],
        Some(MachineProfile::FuelIp34) => &["IRIX", "Iris", "IPsomething", "IP34"],
        Some(MachineProfile::TezroIp53) => &["IRIX", "Iris", "IP53"],
        Some(MachineProfile::Origin3200Ip35 | MachineProfile::Onyx3000Ip35) => &["IRIX", "Iris", "IPsomething", "IP35"],
        None => &["IRIX", "Iris", "IPsomething"],
    }
}

/// The emulator's name for banners and window titles. Three runs in four it
/// is the canonical one; otherwise an expansion of IRIS rolled once per
/// process, whose third word is chosen to match the machine profile that
/// `set_emulator_profile` recorded (so a GUI that starts a different machine
/// later gets the same roll with the right model word).
pub fn emulator_name() -> String {
    /// `None` for the canonical name, else the roll (first, third, fourth).
    static ROLL: std::sync::OnceLock<Option<(usize, u64, usize)>> = std::sync::OnceLock::new();
    const FIRSTS: [&str; 9] = ["Irresponsible", "Incredible", "Insufferable", "Infuriating", "Inaccurate", "Incomplete", "Interactive", "Indomitable", "Irresistible"];
    const FOURTHS: [&str; 4] = ["Simulator", "System", "Substitute", "Sandbox"];

    let roll = ROLL.get_or_init(|| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        if now % 4 != 0 {
            return None;
        }
        Some((((now / 4) % FIRSTS.len() as u64) as usize, now / 64, ((now / 256) % FOURTHS.len() as u64) as usize))
    });

    match *roll {
        None => "Irresponsible Rust IRIX Simulator".to_string(),
        Some((first, third_roll, fourth)) => {
            let thirds = emulator_name_thirds(emulator_profile());
            let third = thirds[(third_roll % thirds.len() as u64) as usize];
            format!("{} Rust {} {}", FIRSTS[first], third, FOURTHS[fourth])
        }
    }
}

pub struct Machine {
    cpu: Arc<dyn crate::cpu::mips_exec::CpuDevice>,
    /// Where this machine has memory.
    memory: Arc<crate::memory_map::MemoryMap>,
    /// The physical bus and the Indy's or Indigo2's devices on it.
    board: Arc<Ip22Bus>,
    mc: MemoryController,
    hpc3: Hpc3,
    monitor: Arc<Monitor>,
    /// TCP port the monitor serves on 127.0.0.1 (`cfg.monitor_port`).
    monitor_port: u16,
    /// Sender for async machine events (HardReset, PowerOff) from devices.
    pub event_tx: MachineEvents,
    event_rx: Option<MachineEventQueue>,
    /// Reliable, unbounded terminal-device-fault path.
    fatal_rx: Option<mpsc::Receiver<MachineFatal>>,
    /// First terminal device failure. A failed machine cannot restart.
    fatal_fault: Option<MachineFatal>,
    timer_manager: Arc<TimerManager>,
    /// When `cfg.ci` is set, the channel-A backend is replaced by this
    /// in-process one so the CI control socket can drive the console.
    ci_serial: Option<Arc<crate::dev::serial_host::CiSerialBackend>>,
    /// The shared `--serial-log` sink, kept so `stop` can flush a final
    /// partial line (wall-timestamp mode holds one until its newline).
    serial_log: Option<Arc<crate::dev::serial_log::SerialLog>>,
    /// The last snapshot loaded, for `parent` and `rollback`.
    restored: crate::snapshot::Restored,
    /// Path of the configured scratch SCSI volume, if any. The CI socket reads
    /// and writes this file directly (with the machine briefly stopped) to
    /// inject/exfiltrate files without going through the network. None when no
    /// SCSI device has `scratch = true` set in the config.
    scratch_path: Option<std::path::PathBuf>,
    /// Disk provenance captured at construction (configured path + host file
    /// size per SCSI id). Written into snapshot manifests and validated on
    /// restore so captured state never lands on a different base disk.
    disks: Vec<DiskRef>,
    /// The GIO cards and where they sit, recorded in snapshot manifests.
    gio_slots: Vec<(String, crate::slot_path::MachineSlotPath)>,
    /// Configured nvram file path, recorded in snapshot manifests.
    nvram_path: String,
    /// Host-forced Newport resolution from `[graphics] resolution` at construction.
    display_resolution: crate::dev::ng1::vc2_timings::NewportResolution,
    /// Whether Newport is the selected graphics board.
    newport_active: bool,
    /// Indigo2 IP22 fullhouse layout (`!guinness`).
    fullhouse: bool,
    /// Pointer slot shared by every holder of a raw `*mut Machine` — the
    /// monitor's `SystemController`, the `machine-events` thread and the CI
    /// server. Created on first use by `machine_slot` and nulled in `Drop`, so
    /// a handle that outlives this Machine sees "machine is not running"
    /// instead of freed memory.
    controller_slot: Option<MachineSlot>,
    controller_registered: bool,
    /// Where the machine is in its start/stop lifecycle. Never `Running`:
    /// `status` derives that from the CPU's own running flag, so an
    /// execution-control command cannot leave it stale. See
    /// [`crate::machine_state`].
    lifecycle: GuestState,
}

#[cfg(feature = "wgpu")]
type ResidentCheckpoint = (toml::Value, (Vec<u32>, Vec<u32>));

/// One Newport head's monitor commands under numbered names: `rex1`,
/// `vc21`, `xmap1`, `cmap1`, `bt4451`, `dcb1`, `disp1` for the second head.
/// Both boards name their commands alike and the monitor gives a name to the
/// first device that has it, so the plain names reach the first head; it has
/// the `0` names as well.
struct HeadCommands { rex: Arc<Rex3>, head: u8 }

impl HeadCommands {
    /// `rex1` on head 1 is the board's `rex`.
    fn plain<'a>(&self, cmd: &'a str) -> Option<&'a str> {
        let digit = (b'0' + self.head) as char;
        cmd.strip_suffix(digit)
    }
}

impl Device for HeadCommands {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { self.rex.is_running() }

    fn register_commands(&self) -> Vec<(String, String)> {
        self.rex.register_commands().into_iter()
            .map(|(c, _)| (format!("{c}{}", self.head), format!("`{c}` on Newport head {}", self.head)))
            .collect()
    }

    fn execute_command(&self, cmd: &str, args: &[&str], writer: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        let plain = self.plain(cmd).ok_or("Command not found")?;
        self.rex.execute_command(plain, args, writer)
    }
}

impl Machine {
    /// Build the Indy or Indigo2 `cfg` describes. Panics if its PROM cannot
    /// be loaded; [`try_new`](Self::try_new) returns the error instead.
    pub fn new(cfg: MachineConfig) -> Self {
        Self::register_host_services();
        Self::new_with_testdev(cfg, None)
    }

    /// `new`, but construction errors are returned to the caller instead of panicking.
    pub fn try_new(cfg: MachineConfig) -> Result<Self, String> {
        Self::register_host_services();
        Self::try_new_with_testdev(cfg, None)
    }

    fn register_host_services() {
        // Host calls (iris-hostcall): the self-test service is always there
        // when the feature is built in, so a guest can prove the trap works.
        #[cfg(feature = "hostcall")]
        iris_hostcall::register(iris_hostcall::SELFTEST, Box::new(iris_hostcall::SelfTest));
        #[cfg(feature = "hostgl")]
        iris_hostgl::register();
    }

    /// `new`, with the bare-metal test device supplied by the caller rather
    /// than built from `cfg.test_device_dump`. Passing one enables it — the
    /// device *is* the request, so `cfg.test_device` need not also be set.
    ///
    /// Exists for hosts that link IRIS as a library and need the guest's
    /// console and exit code delivered in-process instead of to stdout and
    /// `process::exit` (`crate::bench_runner`, `TestDevice::new_embedded`).
    pub fn new_with_testdev(
        cfg: MachineConfig,
        testdev_override: Option<Arc<crate::dev::testdev::TestDevice>>,
    ) -> Self {
        Self::try_new_with_testdev(cfg, testdev_override)
            .unwrap_or_else(|error| panic!("machine construction failed: {error}"))
    }

    /// [`new_with_testdev`](Self::new_with_testdev), returning construction errors.
    pub fn try_new_with_testdev(
        cfg: MachineConfig,
        testdev_override: Option<Arc<crate::dev::testdev::TestDevice>>,
    ) -> Result<Self, String> {
        let prom = Prom::for_profile(cfg.machine.profile, &cfg.prom)?;
        Self::try_with_prom(cfg, testdev_override, prom)
    }

    fn try_with_prom(
        mut cfg: MachineConfig,
        testdev_override: Option<Arc<crate::dev::testdev::TestDevice>>,
        prom: Prom,
    ) -> Result<Self, String> {
        cfg.validate_newport_raster_gate()?;
        // The rig takes any part: `MachineConfig::cpu_part` looks at this.
        if testdev_override.is_some() && !cfg.has_device("test_device") {
            cfg.set_device("test_device", true);
        }
        let resolved = crate::topology::DevicePlacement::new(&cfg)?;
        let gio: Vec<(u8, Card)> = resolved.placed().filter_map(|(d, at)| Some((at.gio_slot()?, d.device.clone()))).collect();
        // Capture config flags that are needed after the local `cfg` binding
        // is shadowed later in this function.
        let ci_enabled = cfg.ci;
        let perf = cfg.perf.clone();
        #[cfg(feature = "jitv2")]
        let (jitv2_threads, jitv2_threads_reason) = cfg.jitv2_pool_size();
        #[cfg(feature = "jitv2")]
        let jitv2_compile = cfg.jitv2.compile;
        let display_resolution = cfg.graphics.resolution;
        let newport_active = cfg.graphics_board() == GraphicsBoard::Newport;
        let monitor_port = cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT);
        let cfg_cpu_model = cfg.cpu_part().expect("validate() accepted the cpu");

        let guinness = cfg.machine.profile.guinness();

        // 0. EEPROMs. Real IP22 hardware has two distinct 93-series serial
        // EEPROM chips, not one shared part: a CPU-daughtercard chip wired to
        // the MC (REG_EEROM @ 0x1fa00030 — CPU boot config, incl. CACHSZ_REG),
        // and a motherboard chip wired to HPC3 (MISC_EEPROM_DATA @
        // 0x1fbb0008 — NVRAM/env vars/MAC, see Eeprom93c56::backdoor_set_mac).
        let eeprom_mc = Arc::new(Mutex::new(Eeprom93c56::new()));
        let eeprom_hpc3 = Arc::new(Mutex::new(Eeprom93c56::with_path(crate::devlog::LogModule::Nveeprom, cfg.nveeprom.clone())));
        // CACHSZ_REG (word 0x11): secondary cache size in 4KB pages. The PROM reads
        // it when Config.SC reports no probed L2, which is exactly the case for a
        // model that has no secondary cache. The EEPROM powers up erased (0xFFFF),
        // so such a model MUST write 0 here — otherwise the PROM believes in a
        // 256 MB L2 and IRIX spends the boot flushing a cache that does not exist.
        // Keyed on the configured CPU, not on a cargo feature: the model is a
        // runtime choice, so a feature gate here silently stopped firing.
        let model_has_l2 = cfg_cpu_model.l2_kb > 0;
        if !model_has_l2 {
            eeprom_mc.lock().set_cachsz(0);
        }

        #[cfg(feature = "ci_clock")]
        let timer_manager = Arc::new(TimerManager::with_clock(crate::clock::Clock::counting()));
        #[cfg(not(feature = "ci_clock"))]
        let timer_manager = Arc::new(TimerManager::new());
        // 1. Create all devices first
        // Memory Controller
        let mc_banks: [u32; 4] = cfg.banks[..4].try_into().expect("the MC has four banks");
        let mc = MemoryController::new_for_profile(
            eeprom_mc.clone(), guinness, mc_banks, cfg.machine.profile.ip28(), timer_manager.clock().clone());
        // IP28: IRIX takes the CPU speed from the PROM's `cpufreq` (194, i.e.
        // a 195 MHz R10000) instead of measuring it, and times with it: UST
        // assumes Count at half that, and the RPSS cycle counter the MC clock
        // at the CPU clock over the Config.EC system-clock divisor and the
        // divider it programs. The generic 33 MHz Count and 50 MHz MC made
        // UST run at a third of real time (Quake in slow motion) and the
        // cycle counter at a quarter.
        //
        // The board runs SysAD, and so the MC, at a quarter of the CPU clock
        // (EC 7, 195 / 4 = 48.75 MHz). The PROM and IRIX also size the MC
        // refresh preload (CTRLD) from EC; with EC 0, which the PROM's
        // `system` command prints as "?", it came out four times too long and
        // IRIX warned of an old PROM on every boot.
        let count_hz = cfg.clock.count_hz(cfg_cpu_model);
        if cfg.machine.profile.ip28() {
            // The R10000's Count ticks at half the pipeline clock.
            mc.set_clock_hz(2 * count_hz / IP28_SYSCLK_DIV);
        }

        // RAM banks sized per config. addr_mask is initialized to mem_size-1;
        // remap_banks() updates it via set_addr_mask() when MEMCFG0/1 are written during POST.
        let banks = [
            RamBank::new(cfg.banks[0].max(1) as usize),
            RamBank::new(cfg.banks[1].max(1) as usize),
            RamBank::new(cfg.banks[2].max(1) as usize),
            RamBank::new(cfg.banks[3].max(1) as usize),
        ];

        // PROM (1MB at 0x1FC00000), loaded by the caller.
        let prom_port = prom.get_port();

        // Shared atomics — created first so all devices and the display thread use the same Arc.
        // (CPU cycle counter is no longer among these: it lives inline on
        // MipsCore.hot.cycles now, wired into devices via set_cpu_cycles
        // after the executor/CPU exist — see Hot::cycles's doc comment.)
        let heartbeat     = Arc::new(AtomicU64::new(0)); // activity bits: see crate::status::hb
        let fasttick_count = Arc::new(AtomicU64::new(0)); // CP0 Compare match counter
        // What every graphics board's status bar reads; the CPU feeds its
        // fetch counters and the cycle counter is wired in below.
        let status = crate::status::MachineStatus::new(heartbeat.clone(), fasttick_count.clone());
        let [decoded_count, l1i_hit_count, l1i_fetch_count, uncached_fetch_count] = status.fetch_counters();
        let (fatal_tx, fatal_rx) = mpsc::channel::<MachineFatal>();

        // HPC3 (512KB at 0x1FB80000). The SCC's channels start wired to
        // nothing; each gets its host end below.
        let ioc = Ioc::new_for_profile(&timer_manager, guinness, cfg.machine.profile.ip28());
        ioc.ps2().set_layout(cfg.keyboard);

        // One sink for both serial paths, so the CI and TCP logs are the same
        // bytes (timestamps included). Opened once, before any peripheral
        // `start()` clones the current backend Arc into the RX/TX threads.
        let tag = if guinness { "Indy" } else { "Indigo2" };
        let serial_log = crate::dev::serial_host::open_log(&cfg, tag);

        // Channel B (tty1) is the SGI serial console: in CI mode the
        // in-process backend the control socket drives directly, otherwise
        // TCP on `serial_port_b`, either teed to `serial_log`. Channel A
        // (tty2) serves TCP on `serial_port_a` outside CI mode, so several
        // `--ci` instances can run side by side.
        let [port_a, port_b] = cfg.serial_ports();
        let ci_serial = ci_enabled.then(|| Arc::new(crate::dev::serial_host::CiSerialBackend::new()));
        // A `[hostserial]`, `[spaceball]` or `[wacom]` section takes its port instead of the
        // socket: serial1 (ttyd1) is channel B, serial2 (ttyd2) channel A.
        let peripheral = |n: u8, log| crate::dev::serial_dev::peripheral(&cfg, n, log);
        let label = format!("{tag} serial channel B");
        let console = if let Some(b) = peripheral(1, serial_log.clone())? {
            log::warn!("{tag}: serial1 is the console port; the PROM and IRIX console now go to the peripheral");
            b
        } else {
            let (console, _) = crate::dev::serial_host::serial_line(
                &crate::dev::serial_host::SerialLine {
                    label: &label,
                    tcp_port: Some(port_b),
                    tcp_setting: "serial_port_b",
                    socket: cfg.serial_socket_b.as_deref(),
                    socket_setting: "serial_socket_b",
                },
                serial_log.clone(),
                ci_serial.as_ref(),
            );
            console
        };
        ioc.scc().set_backend_b(crate::dev::serial_map::mapped(tag, "SCC channel B", Some(1), true, &cfg, console));
        let label = format!("{tag} serial channel A");
        let log_a = crate::dev::serial_host::open_log_at(&cfg, cfg.serial_log_a.as_deref(), &label);
        let channel_a = if let Some(b) = peripheral(2, log_a.clone())? {
            b
        } else {
            let (channel_a, _) = crate::dev::serial_host::serial_line(
                &crate::dev::serial_host::SerialLine {
                    label: &label,
                    tcp_port: (!ci_enabled).then_some(port_a),
                    tcp_setting: "serial_port_a",
                    socket: cfg.serial_socket_a.as_deref(),
                    socket_setting: "serial_socket_a",
                },
                log_a,
                None,
            );
            channel_a
        };
        ioc.scc().set_backend_a(crate::dev::serial_map::mapped(tag, "SCC channel A", Some(2), false, &cfg, channel_a));
        ioc.set_heartbeat(heartbeat.clone());
        ioc.set_clock_ticks(fasttick_count.clone(), status.cpu(0).board_tick_rate());
        // Attach SCSI devices from config, each at its slot in the slot
        // list. The disks recorded here, and the scratch volume's path for the CI socket,
        // come back with it. Disk provenance for snapshot manifests is
        // captured while the MachineConfig `cfg` is still in scope (it is
        // shadowed by the CPU config below): the slot (controller * 16 + id,
        // see `config::scsi_slot`), the configured path and the host file
        // size.
        // The buses are filled before the controllers exist and handed to them.
        let scsi_buses = Hpc3::new_scsi_buses(guinness);
        let attached = match crate::scsi::attach_from_config(&cfg, None, &resolved, &scsi_buses.iter().collect::<Vec<_>>()) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("iris: fatal: {e}");
                std::process::exit(1);
            }
        };
        let hpc3 = Hpc3::with_net(eeprom_hpc3.clone(), ioc.clone(), guinness, heartbeat.clone(), cfg.network(), cfg.no_audio, cfg.audio.clone(), cfg.nvram.clone(), cfg.rtc_offset, cfg.scsi_deferred_int, scsi_buses, &timer_manager);

        // Backdoor-inject the configured (or default) Ethernet station
        // address before the CPU ever runs, since real hardware always has
        // one burned in and there is no user-facing way to set one. Both
        // NVRAM (Indy) and the motherboard EEPROM (Indigo2) are persisted to
        // disk (see cfg.nvram / cfg.nveeprom), so both only patch while the
        // `eaddr` slot is still blank — never clobbers a MAC the guest
        // already `setenv`'d (Indy) or that a prior run already saved
        // (Indigo2). A settings block that is still blank apart from that
        // address then gets the PROM's own defaults (`sgi_nvram`): the PROM
        // reads the volume for the boot tune before it repairs a bad block,
        // so a blank one would make the first boot silent.
        if guinness {
            hpc3.rtc().backdoor_set_mac_if_blank(cfg.network().mac);
            hpc3.rtc().initialize_nvram_if_erased(crate::dev::sgi_nvram::REV_IP22);
            let (rtc, rev) = (hpc3.rtc(), crate::dev::sgi_nvram::REV_IP22);
            crate::nv_storage::apply_headless_console(cfg.headless, "Indy", &cfg.nvram, rtc.console_is_serial(rev), || rtc.set_console_serial(rev));
        } else {
            let mut eeprom = eeprom_hpc3.lock();
            eeprom.backdoor_set_mac_if_blank(cfg.network().mac);
            let rev = if cfg.machine.profile.ip28() { crate::dev::sgi_nvram::REV_IP28 } else { crate::dev::sgi_nvram::REV_IP22 };
            eeprom.initialize_if_erased(rev);
            let serial = eeprom.console_is_serial(rev);
            crate::nv_storage::apply_headless_console(cfg.headless, "Indigo2", &cfg.nveeprom, serial, || eeprom.set_console_serial(rev));
        }

        // A DaynaPort is built where the gateway is, by Hpc3; a target that
        // will not attach ends the run (see `attach_from_config`).
        let daynaports = || -> Result<(), String> {
            for d in &attached.daynaports {
                let at = crate::config::scsi_label(d.controller, d.id);
                match hpc3.add_scsi_daynaport(d.controller, d.id as usize, d.params) {
                    Ok(()) => println!(
                        "iris: DaynaPort SCSI/Link at {at} — MAC {}, gateway {} (guest {}/{})",
                        crate::net::mac_str(&d.params.mac),
                        d.params.subnet.gateway_ip, d.params.subnet.client_ip, d.params.subnet.netmask),
                    Err(e) => crate::scsi::attach_failed(format!("could not attach DaynaPort to {at}: {e}"), &at)?,
                }
            }
            Ok(())
        };
        if let Err(e) = daynaports() {
            eprintln!("iris: fatal: {e}");
            std::process::exit(1);
        }
        let scratch_path = attached.scratch;
        // The Indy and Indigo2 have always listed CD-ROMs beside the disks.
        let disk_provenance = DiskRef::from_tables(&attached.tables, true);
        let gio_slots: Vec<(String, crate::slot_path::MachineSlotPath)> = gio.iter()
            .map(|(slot, c)| (c.spec().name.to_string(), crate::slot_path::MachineSlotPath::root().gio(*slot)))
            .collect();
        let nvram_provenance = cfg.nvram.clone();

        #[cfg(feature = "wgpu")]
        let resident_gpu = if newport_active
            && cfg.graphics.raster == crate::config::RasterUnit::Wgpu
        {
            let options = crate::wgpu::GpuOptions {
                backend: cfg.graphics.gpu_backend.clone(),
                ..Default::default()
            };
            let gpu = crate::wgpu::Gpu::new(&options)
                .map_err(|error| format!("create Newport resident GPU: {error}"))?;
            if !matches!(gpu.caps.backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12) {
                return Err(format!(
                    "Newport resident GPU resolved graphics.gpu_backend = {:?} to unsupported {:?}; \
                     use Vulkan or D3D12",
                    cfg.graphics.gpu_backend, gpu.caps.backend
                ));
            }
            gpu.check().map_err(|error| format!("initialize Newport resident GPU: {error}"))?;
            eprintln!("iris: Newport resident GPU enabled without CPU fallback: {}", gpu.caps);
            Some(gpu)
        } else {
            None
        };

        #[cfg(feature = "wgpu")]
        let resident_mode = match cfg.graphics.newport_wgpu_mode() {
            crate::config::NewportWgpuMode::Direct =>
                crate::dev::ng1::rex3::ResidentDisplayMode::DirectWgpu,
            crate::config::NewportWgpuMode::Readback =>
                crate::dev::ng1::rex3::ResidentDisplayMode::ReadbackCompatibility,
        };

        // REX3 Graphics — Newport only; skipped when another board, or none, is selected.
        let rex3: Option<Arc<Rex3>> = if cfg.graphics_board() != crate::config::GraphicsBoard::Newport {
            None
        } else {
            let r = Rex3::new(status.clone());
            #[cfg(feature = "wgpu")]
            let r = {
                let mut r = r;
                if let Some(gpu) = &resident_gpu {
                    r.activate_resident_gpu_for_runtime(Arc::clone(gpu), resident_mode)
                        .map_err(|error| format!("activate primary Newport resident GPU: {error}"))?;
                }
                r
            };
            let r = Arc::new(r);
            r.set_fatal_sender(fatal_tx.clone());
            let wiring = GIO_SLOT_MAP[profile_idx(guinness)][GioSlot::Gfx as usize];
            r.set_vblank_irq(ioc.irq_line(wiring.retrace));
            r.set_fifo_full_irq(ioc.irq_line(wiring.fifo_full));
            Some(r)
        };

        let rex3_head1: Option<Arc<Rex3>> = if !gio.iter().any(|c| matches!(c.1, Card::Newport)) {
            None
        } else {
            let r = Rex3::new(status.clone());
            #[cfg(feature = "wgpu")]
            let r = {
                let mut r = r;
                if let Some(gpu) = &resident_gpu {
                    r.activate_resident_gpu_for_runtime(Arc::clone(gpu), resident_mode)
                        .map_err(|error| format!("activate secondary Newport resident GPU: {error}"))?;
                }
                r
            };
            let r = Arc::new(r);
            r.set_fatal_sender(fatal_tx.clone());
            r.set_head_index(1);
            // An Indy baseboard older than revision 4 does not take a second
            // board: the PROM and IRIX both skip the probe below it.
            if guinness { ioc.set_board_rev(4); }
            // Expansion slot 0, the one place IRIX looks for a second board.
            // On an Indy the slot has one interrupt pin for both sources.
            let wiring = GIO_SLOT_MAP[profile_idx(guinness)][GioSlot::Exp0 as usize];
            if wiring.retrace == wiring.fifo_full {
                let [retrace, fifo] = crate::irq::IrqLine::wired_or(ioc.irq_line(wiring.retrace));
                r.set_vblank_irq(retrace);
                r.set_fifo_full_irq(fifo);
            } else {
                r.set_vblank_irq(ioc.irq_line(wiring.retrace));
                r.set_fifo_full_irq(ioc.irq_line(wiring.fifo_full));
            }
            Some(r)
        };

        // GR2 (XZ / Extreme) in the GIO gfx slot, when the board is selected.
        let gr2: Option<Arc<crate::dev::gr2::Gr2>> = match cfg.graphics_board() {
            crate::config::GraphicsBoard::Xz | crate::config::GraphicsBoard::Elan | crate::config::GraphicsBoard::Extreme => {
                use crate::config::GraphicsBoard;
                use crate::dev::gr2::{Gr2, Gr2Variant};
                // The Indy's XZ has 4 GE7s (libGLcore calls it GR3-Elan);
                // the Indigo2 sold XZ with 2 and Elan with 4.
                let variant = match cfg.graphics_board() {
                    GraphicsBoard::Extreme => Gr2Variant::Extreme,
                    GraphicsBoard::Elan => Gr2Variant::Elan,
                    _ if guinness => Gr2Variant::Elan,
                    _ => Gr2Variant::Xz,
                };
                let g = Gr2::new(variant, status.clone());
                // GR2 retrace is GIO interrupt 2 (LIO_GIO_2): on Indigo2 through
                // the EXTIO SG_RETRACE fan-out, on Indy the direct line. It is a
                // latch the guest clears (see Ioc::write8/int2_write8), so only
                // the rising edge is driven from here.
                let line = if guinness { crate::dev::ioc::IocInterrupt::VerticalRetrace } else { crate::dev::ioc::IocInterrupt::GioSgRetrace };
                if guinness { ioc.set_gen_cntl_retrace_ack(true); }
                let ioc_r = ioc.clone();
                // true = vertical blank starts, false = it ends. On Indigo2,
                // EXTIO SG_STAT_0 is the board's vertical status level (low in
                // blank); the PROM and the kernel retrace handler poll it.
                g.set_retrace_callback(Arc::new(move |vblank| {
                    if vblank { crate::pf_instant!(crate::perfetto::events::GR2_VBLANK); }
                    if vblank { ioc_r.set_interrupt(line, true); }
                    if !guinness { ioc_r.set_ext_io_level(crate::dev::ioc::ext_io_regs::SG_STAT_0, !vblank); }
                }));
                Some(g)
            }
            _ => None,
        };

        // IMPACT graphics in the GIO graphics slot.
        let mgras: Option<Arc<crate::dev::mgras::Mgras>> = match cfg.graphics_board() {
            crate::config::GraphicsBoard::SolidImpact
            | crate::config::GraphicsBoard::HighImpact
            | crate::config::GraphicsBoard::MaxImpact => {
                Some(crate::dev::mgras::Mgras::with_raster(cfg.graphics_board(), {
                    use crate::dev::ioc::IocInterrupt::{GioSgFifo, GioSgGraphics, GioSgRetrace};
                    [ioc.irq_line(GioSgFifo), ioc.irq_line(GioSgGraphics), ioc.irq_line(GioSgRetrace)]
                }, status.clone(), cfg.graphics.raster, &cfg.graphics.gpu_backend).unwrap_or_else(|e| {
                    eprintln!("mgras: {e}");
                    std::process::exit(1)
                }))
            }
            _ => None,
        };

        // N64 development board (Ultra64) — GIO slot 0 at 0x1F400000
        let ultra64: Option<Arc<crate::dev::ultra64::Ultra64>> = gio.iter().any(|c| matches!(c.1, Card::Ultra64)).then(|| {
            match crate::dev::ultra64::Ultra64::new(ioc.irq_line(crate::dev::ioc::IocInterrupt::GioExp0)) {
                Ok(dev) => Some(Arc::new(dev)),
                Err(e)  => {
                    eprintln!("ultra64: failed to initialize ({e}); running without N64 dev board");
                    None
                }
            }
        }).flatten();

        // VINO (Video-In, No Out) — GIO64 at 0x1F080000
        let vino = crate::dev::vino::Vino::new();
        vino.set_irq(ioc.irq_line(crate::dev::ioc::IocInterrupt::VideoVsync));

        // 2. Create Physical Bus with devices
        // `cfg` is shadowed by MipsCpuConfig further down; grab this first.
        let cheritest_hook = cfg.cheritest_dump_hook;

        // Bare-metal test device: default off.
        let testdev = if let Some(dev) = testdev_override {
            Some(dev)
        } else if let Some(Card::TestDevice(t)) = gio.iter().map(|c| &c.1).find(|c| matches!(c, Card::TestDevice(_))) {
            let path = t.dump.clone().unwrap_or_else(|| crate::dev::testdev::DEFAULT_DUMP_PATH.to_string());
            eprintln!("iris: test device enabled at {:#010x}, dumps to {}",
                      crate::dev::testdev::TEST_DEV_BASE, path);
            let run = t.run().expect("validate() checked the run config");
            Some(Arc::new(crate::dev::testdev::TestDevice::with_run_config(path, run)))
        } else {
            None
        };

        let phys = Physical::new(banks.into());

        // virtio devices: GIO expansion slot 0 or 1, on the line the IOC gives
        // that slot (the Indy's two direct pins, or the Indigo2's EXT_IO
        // fan-out). They master the bus through `phys`.
        let virtio_mmio = |name: &str, slot: u8, core| {
            use crate::dev::ioc::{GIO_SLOT_BASES, IocInterrupt};
            let slot = if slot == 0 { GioSlot::Exp0 } else { GioSlot::Exp1 };
            let source = match (guinness, slot) {
                (true, GioSlot::Exp0) => IocInterrupt::GioExp0,
                (true, _) => IocInterrupt::GioExp1,
                (false, s) => GIO_SLOT_MAP[profile_idx(false)][s as usize].retrace,
            };
            let base = GIO_SLOT_BASES[slot as usize];
            eprintln!("iris: {name} at {base:#010x}");
            let dev = crate::dev::virtio::mmio::VirtioMmio::new(
                core,
                ioc.irq_line(source),
                Arc::new(crate::dev::virtio::BusMemory(phys.clone())),
            );
            (Arc::new(dev), base)
        };
        let virtio = gio.iter().find_map(|(slot, c)| match c {
            Card::VirtioSerial(v) => Some(virtio_mmio("virtio-serial", *slot, crate::dev::virtio::console::from_config(v))),
            _ => None,
        });
        let virtio_9p = gio.iter().find_map(|(slot, c)| match c {
            Card::Virtio9p(v) => Some(virtio_mmio("virtio-9p", *slot, crate::dev::virtio::p9::from_config(v))),
            _ => None,
        });

        let board = Arc::new(Ip22Bus::new(phys.clone(), Ip22Devices {
            rex3: rex3.clone(),
            rex3_head1: rex3_head1.clone(),
            gr2: gr2.clone(),
            mgras: mgras.clone(),
            ultra64,
            testdev,
            virtio,
            virtio_9p,
            vino,
            eisa: (!guinness).then(|| Arc::new(crate::dev::eisa::Eisa::new())),
            mc: mc.clone(),
            hpc3: hpc3.clone(),
            prom: prom_port,
        }, cfg.machine.profile.ip28()));

        // Connect Physical to MC (for VDMA)
        mc.set_phys(phys.clone());
        mc.set_dma_irq(ioc.irq_line(crate::dev::ioc::IocInterrupt::McDma));

        // Wire MEMCFG callback: when MC writes MEMCFG0/1, remap the banks. Runs
        // on the one CPU's thread, so there is no other to park.
        let world = Arc::new(crate::cpu::smp::World::new());
        {
            let (board, world) = (Arc::downgrade(&board), world.clone());
            mc.set_memcfg_callback(Box::new(move |addrs| {
                if board.upgrade().is_some_and(|board| board.remap_banks(addrs)) {
                    world.layout_changed();
                }
            }));
        }

        // Fire initial remap using MC's boot-time MEMCFG values
        {
            let (memcfg0, memcfg1) = mc.get_memcfg();
            let addrs = mc.parse_memcfg(memcfg0, memcfg1);
            board.remap_banks(addrs);
        }
        
        // Connect HPC3 to System Memory (via Physical)
        hpc3.set_phys(phys.clone());

        // Connect VINO to System Memory, install a video source, start DMA.
        // Source kind + broadcast standard come from `[vino]` in iris.toml.
        board.vino.set_phys(phys.clone());
        if let Some(mgras) = &board.mgras { mgras.set_phys(phys.clone()); }
        // Host GL frames composite straight into the IMPACT framebuffer.
        #[cfg(feature = "hostgl")]
        if let Some(mgras) = &board.mgras {
            iris_hostcall::set_display(Arc::new(crate::dev::mgras::ImpactScreen(mgras.clone())));
        }
        let standard = match cfg.vino.standard {
            crate::config::VinoStandard::Ntsc => crate::video_source::VideoStandard::Ntsc,
            crate::config::VinoStandard::Pal  => crate::video_source::VideoStandard::Pal,
        };
        let source: Option<Arc<dyn crate::video_source::VideoSource>> = match cfg.vino.source {
            crate::config::VinoSource::Camera => {
                {
                    let idx = cfg.vino.camera_index;
                    match crate::dev::camera::CameraSource::new_with_index(standard, idx) {
                        Ok(c)  => Some(Arc::new(c)),
                        Err(e) => {
                            eprintln!("VINO: camera {} unavailable ({}); using black source", idx, e);
                            Some(Arc::new(crate::video_source::BlackSource::new(standard)))
                        }
                    }
                }
            }
            crate::config::VinoSource::TestPattern =>
                Some(Arc::new(crate::video_source::TestPatternSource::new(standard))),
            crate::config::VinoSource::Black =>
                Some(Arc::new(crate::video_source::BlackSource::new(standard))),
            // Video-In disabled: no source, no DMA thread. VINO stays mapped.
            crate::config::VinoSource::Off => None,
        };
        if let Some(source) = source {
            let adjusted: Arc<dyn crate::video_source::VideoSource> =
                Arc::new(crate::video_source::CdmcAdjustedSource::new(source, board.vino.clone()));
            board.vino.set_source(adjusted);
            board.vino.start();
        }

        // 5. CPU config + TLB + Executor. The model is a runtime choice, but each
        //    family monomorphises its own CPU (`crate::cpu::build`) — no per-model branch on the hot path.
        let sysad: Arc<dyn BusDevice> = phys.clone();
        let memory = Arc::new(crate::ip22_bus::memory_map(cfg.machine.profile.ip28()));
        let cfg_ip28 = cfg.machine.profile.ip28();
        macro_rules! build_cpu { ($cache:ty) => {{
        let cfg = MipsCpuConfig::for_model::<$cache>(memory.clone()).with_prid(cfg_cpu_model.prid());
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let mut executor: MipsExecutor<MipsTlb, $cache> = MipsExecutor::new(sysad.clone(), tlb, &cfg);
        // The MC decodes 32 physical address bits; see `MipsExecutor::phys_mask`.
        executor.phys_mask = u32::MAX as u64;
        if cfg_ip28 {
            // R10000 Config.EC (bits 12:9), the SysClk divisor the mode
            // word latches at reset: 7 is divide by 4.
            executor.core.cp0_config |= IP28_SYSCLK_EC << 9;
        }

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

        // CP0 Count runs at half the chip's clock unless the user overrode
        // it via `[clock] fixed_mhz` or the CLI. Must happen
        // before the core starts executing.
        executor.core.set_count_hz(count_hz);

        // Inject the shared fasttick_count Arc into the executor core before wrapping in MipsCpu.
        // (cycles has no equivalent here — it's inline on MipsCore.hot now.)
        executor.core.fasttick_count = fasttick_count;
        executor.decoded_count       = decoded_count;
        executor.uncached_fetch_count = Arc::clone(&uncached_fetch_count);
        executor.cache.set_l1i_counters(Arc::clone(&l1i_hit_count), Arc::clone(&l1i_fetch_count));
        // Re-sync raw pointers after Arc injection (the Arcs above replaced the ones captured in new()).
        executor.rebind_atomic_ptrs();

        // Give the core the machine's hptimer manager: CP0 Compare writes
        // arm a one-shot on it that raises IP7 from the timer thread. Safe
        // to hand over before the move into MipsCpu below — nothing gets
        // armed (no raw pointer captured) until the CPU actually executes a
        // Compare write, by which time the core sits at its final address
        // inside the executor's Arc<Mutex<..>>.
        executor.core.set_timer_manager(timer_manager.clone());

        Arc::new(MipsCpu::new(executor)) as Arc<dyn crate::cpu::mips_exec::CpuDevice>
        }}}

        let shadow = cfg.machine.cache.is_shadow(cfg_cpu_model);
        let cpu: Arc<dyn crate::cpu::mips_exec::CpuDevice> =
            crate::with_cpu!(cfg_cpu_model, shadow, false, C => build_cpu!(C));

        // The developer status bar shows the CPU's Count rate.
        status.set_count_hz(cpu.count_hz_atomic());

        world.set_slots(vec![cpu.park_slot()]);
        // The CPU's pool flushes through this world too: a CPU belongs to one.
        #[cfg(feature = "jitv2")]
        cpu.share_jitv2(cpu.jitv2(), world.clone(), 0, 0);
        // Connect CPU to MC and IOC for signaling
        let cpu_device: Arc<dyn Device> = cpu.clone().as_device();
        mc.set_cpu(Arc::downgrade(&cpu_device));
        ioc.set_interrupts(cpu.interrupts_ptr());

        // ppmem: the CPU's cache and the JIT read RAM through a cached view of
        // the window that is the CPU's own, so what the cache holds stays out
        // of every other master's sight (`PpMemSpace::add_cached_view`). Its
        // bitmap is published straight into the CPU's inline fields, the
        // executor's and the cache's, which are at their final addresses by
        // now; the only writer is this CPU thread's MEMCFG path. A cache with
        // no bitmap field (the full cache without tcache) returns null, which
        // the view skips. Physical's direct paths keep the shared window.
        {
            use crate::ppmem::MappedMemory;
            if let Some(sp) = phys.ppmem_space() {
                let v = sp
                    .add_cached_view(&[cpu.ppmem_bitmap_ptr(), cpu.tcache_bitmap_ptr()])
                    .expect("ppmem: the CPU's cached view");
                unsafe { cpu.set_tcache_window(sp.view_base(v)) };
                unsafe { cpu.set_cached_view(sp, v) };
                // The decode cache keys its window pointers on the maps.
                unsafe { sp.add_key_sink(v, cpu.tcache_key_ptr()) };
                // Window writes bypass BusDevice, so the cache needs the gen
                // window to bump page counters itself — otherwise jitv2 keeps
                // running compiled code for pages the guest has modified.
                #[cfg(feature = "jitv2")]
                unsafe { cpu.set_tcache_gen_window(sp.gen_window_base()) };
            }
        }

        // Wire up the CPU cycle counter for every device that reads it —
        // the boards and Wd33c93a::new run before the CPU exists, so this
        // can't be a constructor parameter (see Hot::cycles's doc comment).
        // Must happen before hpc3.scsi().start() (called later, from
        // Machine::start) actually spawns the worker thread that reads it.
        status.set_cpu_cycles(cpu.cycles_ptr());
        // Which timer is the clock depends on the kernel: IRIX on the Indy
        // takes it on CP0 Compare, on the Indigo2 and IP28 from the 8254.
        // Both count into the CPU's tick counter; each publishes the rate it
        // was programmed to, and the bar expects their sum.
        cpu.publish_tick_rate(status.cpu(0).tick_rate());
        if let Some(td) = &board.testdev { td.attach_core(cpu.core_ptr()); }
        if cheritest_hook {
            if board.testdev.is_none() {
                eprintln!("iris: --cheritest-dump-hook needs --test-device; ignoring");
            } else {
                cpu.set_cheritest_dump_hook(true);
            }
        }
        hpc3.scsi().set_cpu_cycles(cpu.cycles_ptr());
        #[cfg(feature = "ci_clock")]
        timer_manager.clock().set_cpu_cycles(cpu.cycles_ptr());

        // Inject the CPU/self handles jitv2's compile-thread worker needs to
        // pause the CPU around its own memory-growth flush (see
        // `Jitv2::codegen`/`CompileQueue`'s doc comments — the CPU doesn't
        // exist yet when `MipsExecutor::new` constructs its own `Jitv2`
        // default, so this can only happen here, after `Arc::new(MipsCpu::new(..))`,
        // same reasoning as `mc.set_cpu` right above).
        #[cfg(feature = "jitv2")]
        crate::cpu::jitv2::jitv2::attach_pool(&cpu, (jitv2_threads, jitv2_threads_reason), jitv2_compile, phys.clone(), world.clone());

        // Setup DevLog (must be before Monitor so log command is available)
        let devlog = crate::devlog::init_devlog();

        // Setup Monitor
        let monitor = Monitor::new();
        monitor.register_device(devlog.clone());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(vec![cpu.clone()])));
        monitor.register_device(Arc::new(mc.clone()));
        monitor.register_device(Arc::new(hpc3.clone()));
        monitor.register_device(phys.clone());
        if let Some(rex3) = &board.rex3 { monitor.register_device(rex3.clone()); }
        if let Some(rex3) = &board.rex3_head1 {
            // With two heads each has numbered commands; the plain ones stay head 0's.
            if let Some(head0) = &board.rex3 { monitor.register_device(Arc::new(HeadCommands { rex: head0.clone(), head: 0 })); }
            monitor.register_device(Arc::new(HeadCommands { rex: rex3.clone(), head: 1 }));
        }
        if let Some(td) = &board.testdev { monitor.register_device(td.clone()); }
        if let Some(gr2) = &board.gr2 { monitor.register_device(gr2.clone()); }
        if let Some(mgras) = &board.mgras { monitor.register_device(mgras.clone()); }
        if let Some(u64) = &board.ultra64 { monitor.register_device(u64.clone()); }
        if let Some(v) = &board.virtio { monitor.register_device(v.clone()); }
        if let Some(v) = &board.virtio_9p { monitor.register_device(v.clone()); }
        monitor.register_device(Arc::new(board.vino.clone()));
        #[cfg(feature = "perfetto")]
        {
            crate::perfetto::set_guest_cycles(Some(cpu.cycles_ptr()));
            monitor.register_device(Arc::new(crate::perfetto::PerfettoMonitor));
        }
        monitor.register_device(status.clone());
        monitor.register_device(crate::perf_monitor::PerfMonitor::new(
            cpu.running_flag(),
            cpu.cycles_ptr(),
            cpu.fasttick_count(),
            board.rex3.clone(),
            hpc3.hal2().cloned(),
        ));
        let monitor = Arc::new(monitor);

        // Register lock monitor device and all component locks
        {
            crate::locks::register_mutex("mc::eeprom", &eeprom_mc);
            mc.register_locks();
            hpc3.register_locks();
            if let Some(rex3) = &board.rex3 { rex3.register_locks(); }
            if let Some(rex3) = &board.rex3_head1 { rex3.register_locks(); }
            if let Some(mgras) = &board.mgras { mgras.register_locks(); }
            cpu.register_locks();
        }
        {
            let monitor_ptr = Arc::as_ptr(&monitor) as *mut Monitor;
            unsafe { (*monitor_ptr).register_device(Arc::new(LockMonitor)); }
        }

        let (event_tx, event_rx) = MachineEvents::channel(4);

        // Give MC and IOC async event senders so they can request hard-reset / power-off.
        event_tx.set_cpus(world.clone());
        mc.set_event_sender(event_tx.clone());
        ioc.set_event_sender(event_tx.clone());

        crate::thread_affinity::init(perf);

        Ok(Self {
            memory,            cpu,
            board,
            mc,
            hpc3,
            monitor,
            monitor_port,
            event_tx,
            event_rx: Some(event_rx),
            fatal_rx: Some(fatal_rx),
            fatal_fault: None,
            timer_manager,
            ci_serial,
            serial_log,
            restored: Default::default(),
            scratch_path,
            disks: disk_provenance,
            gio_slots,
            nvram_path: nvram_provenance,
            display_resolution,
            newport_active,
            fullhouse: !guinness,
            controller_slot: None,
            controller_registered: false,
            lifecycle: GuestState::Idle,
        })
    }

    fn apply_host_display_resolution(&self) {
        if !self.newport_active {
            return;
        }
        let mode = if self.display_resolution.is_guest() {
            // Fullhouse + guest-selected resolution: IRIX/PROM may never program VC2
            // (embedded Indy PROM, gfxinit mismatch). Bootstrap 1280×1024 so the GUI
            // refresh thread has non-zero dimensions; guest can reprogram VC2 later.
            if self.fullhouse {
                crate::dev::ng1::vc2_timings::NewportResolution::Res1280x1024
            } else {
                return;
            }
        } else {
            self.display_resolution
        };
        let apply = |rex3: &Rex3| {
            #[cfg(feature = "wgpu")]
            if rex3.resident_active() {
                rex3.apply_resident_display_resolution(mode, true)
                    .unwrap_or_else(|error| panic!("REX3 {error}"));
                return;
            }
            rex3.apply_display_resolution(mode);
        };
        if let Some(rex3) = &self.board.rex3 { apply(rex3); }
        if let Some(rex3) = &self.board.rex3_head1 { apply(rex3); }
    }

    /// Path of the configured scratch SCSI volume, if any. Used by the CI
    /// socket scratch-{write,read,clear,info} commands to act on the file
    /// directly while the machine is briefly stopped.
    pub fn scratch_path(&self) -> Option<&std::path::Path> {
        self.scratch_path.as_deref()
    }

    pub fn start(&mut self) {
        if let Some(fault) = &self.fatal_fault {
            eprintln!("iris: refusing to start machine after fatal device failure: {fault}");
            return;
        }
        // Publish "starting" before the monitor binds, so a caller that polls
        // readiness can tell "not started" (Idle) from "startup under way".
        // Halted -> Starting covers the GUI's Stop then Start; a stored
        // Running state is impossible (status derives it from the CPU).
        if let Ok(starting) = self.lifecycle.transition(GuestState::Starting) {
            self.lifecycle = starting;
        }
        // Start peripherals
        self.timer_manager.clock().start();
        self.mc.start();
        self.hpc3.start();
        // Program VC2 before the refresh thread runs so the first frame has size.
        self.apply_host_display_resolution();
        if let Some(rex3) = &self.board.rex3 { rex3.start(); }
        if let Some(mgras) = &self.board.mgras { mgras.start_display(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.start(); }
        if let Some(gr2) = &self.board.gr2 { gr2.start(); }
        if let Some(u64) = &self.board.ultra64 { u64.start(); }
        if let Some(v) = &self.board.virtio { v.start(); }
        if let Some(v) = &self.board.virtio_9p { v.start(); }

        // Monitor server on localhost (port 8888 unless `monitor_port` says
        // otherwise) — always start, even in CI mode, so debug helpers
        // (status/regs/bt/dis) stay reachable while iris-ci drives the serial
        // console.
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));

        // CI mode: the harness drives startup via `restore` / `start`. Don't
        // autostart the CPU so the first command finds a quiet machine.
        #[cfg(not(any(debug_assertions, feature = "developer")))]
        if self.ci_serial.is_none() {
            self.cpu.start();
        }
    }

    /// The shared, nullable pointer slot for this Machine, created on first
    /// use. Must be called while `self` is in its final location (boxed, or on
    /// `main`'s stack) — the slot records this address.
    pub(crate) fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut Machine as *mut dyn Workstation;
        let slot = self.controller_slot
            .get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr))));
        debug_assert!(std::ptr::addr_eq(slot.lock().0, ptr), "Machine moved after its slot was handed out");
        slot.clone()
    }

    /// Register a SystemController with the monitor so that `reset`, `save`,
    /// and `load` commands work. Must be called after `Machine::new()` while
    /// `self` is in its final stack location (i.e. before any moves).
    /// Also starts the machine event dispatch thread (HardReset, PowerOff).
    pub fn register_system_controller(&mut self) {
        // The controller and the event thread reach this Machine through the
        // shared slot, which `Drop` nulls: iris-gui drops a Machine on Stop
        // while the monitor thread (and a connected client) lives on.
        assert!(!self.controller_registered, "register_system_controller called twice");
        self.controller_registered = true;
        let slot = self.machine_slot();
        crate::workstation::register_system_controller(&self.monitor, slot.clone(), self.event_rx.take());
        crate::workstation::register_fatal_controller(slot, self.fatal_rx.take());
    }

    #[cfg(all(test, feature = "wgpu"))]
    pub(crate) fn take_fatal_receiver_for_test(&mut self) -> Option<mpsc::Receiver<MachineFatal>> {
        self.fatal_rx.take()
    }

    pub fn stop(&mut self) {
        self.cpu.stop();
        // Graphics last: a board's stop joins its display thread, which can
        // fail to end, and the disks must be stopped by then.
        self.hpc3.stop();
        self.mc.stop();
        if let Some(u64) = &self.board.ultra64 { u64.stop(); }
        if let Some(v) = &self.board.virtio { v.stop(); }
        if let Some(v) = &self.board.virtio_9p { v.stop(); }
        if let Some(rex3) = &self.board.rex3 { rex3.stop(); }
        if let Some(mgras) = &self.board.mgras { mgras.stop_display(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.stop(); }
        if let Some(gr2) = &self.board.gr2 { gr2.stop(); }
        self.timer_manager.clock().stop();
        // Idle -> Halted is not a lifecycle transition: a machine that was
        // never started has nothing to halt, and the monitor is not up to ask.
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
        // A graceful stop must not lose the final partial line.
        if let Some(log) = &self.serial_log { log.flush(); }
    }

    /// The address this Machine's own monitor is serving on, or None when it
    /// couldn't bind its port (another iris holding it) or has not started.
    pub fn memory_map(&self) -> &Arc<crate::memory_map::MemoryMap> {
        &self.memory
    }

    pub fn monitor_addr(&self) -> Option<std::net::SocketAddr> {
        self.monitor.bound_addr()
    }

    /// The terminal console: relays stdin to the monitor at `addr` and the
    /// monitor's output to stdout. At a terminal the lines come from a line
    /// editor (`console_edit`) that completes `monitor`'s command names. When the monitor closes the connection
    /// (`quit`) `on_eof` ends the process (through
    /// `workstation::exit_process`, which flushes the disks first), so `addr`
    /// must be this process's own monitor (`monitor_addr`): another iris's
    /// would take this one down with it when it quits.
    pub fn run_console_client(addr: std::net::SocketAddr, monitor: Arc<Monitor>, on_eof: impl FnOnce() + Send + 'static) {
        println!("IRIS: {}", emulator_name());
        println!("Connecting to monitor socket...");

        let mut stream = loop {
            match TcpStream::connect(addr) {
                Ok(s) => break s,
                Err(_) => {
                    thread::sleep(std::time::Duration::from_millis(10));
                    continue;
                }
            }
        };

        // At a terminal the lines come from the line editor, which also prints
        // the monitor's output; anywhere else (a pipe, /dev/null, a build
        // without `lineedit`) they are plain lines.
        #[cfg(feature = "lineedit")]
        let on_eof = {
            let once = crate::console_edit::Once::new(on_eof);
            if let Some(editor) = crate::console_edit::LineEditor::new(monitor) {
                editor.run(stream, once);
                return;
            }
            move || once.run()
        };
        Self::relay_monitor_output(stream.try_clone().unwrap(), on_eof);

        let stdin = io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            // Ok(0) is end of input (stdin from /dev/null, say): stop
            // relaying rather than spin; the monitor's output still prints.
            if !matches!(stdin.read_line(&mut line), Ok(n) if n > 0) {
                break;
            }
            if stream.write_all(line.as_bytes()).is_err() {
                break;
            }
        }
    }

    /// Print what the monitor sends until it closes the connection (`quit`, or
    /// iris going away), then run `on_eof`.
    pub(crate) fn relay_monitor_output(mut monitor: TcpStream, on_eof: impl FnOnce() + Send + 'static) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut buf = [0u8; 1024];
            loop {
                match monitor.read(&mut buf) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        print!("{}", String::from_utf8_lossy(&buf[0..n]));
                        io::stdout().flush().unwrap();
                    }
                    Err(_) => break,
                }
            }
            on_eof();
        })
    }

    pub fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> {
        self.hpc3.ioc().ps2()
    }

    /// Borrow the HPC3 controller — used by CI socket commands that touch
    /// RTC/SCSI directly (`rtc-save`, `cdrom-eject`).
    pub fn hpc3(&self) -> &Hpc3 {
        &self.hpc3
    }

    pub fn get_rex3(&self) -> Option<Arc<crate::dev::ng1::rex3::Rex3>> {
        self.board.rex3.clone()
    }

    pub fn get_rex3_head1(&self) -> Option<Arc<crate::dev::ng1::rex3::Rex3>> {
        self.board.rex3_head1.clone()
    }

    /// The graphics board driving the primary display, whichever it is.
    pub fn get_display(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> {
        if let Some(g) = &self.board.gr2 {
            return Some(g.clone() as Arc<dyn crate::gfx_display::GfxDisplay>);
        }
        if let Some(m) = &self.board.mgras {
            return Some(m.clone() as Arc<dyn crate::gfx_display::GfxDisplay>);
        }
        self.board.rex3.clone().map(|r| r as Arc<dyn crate::gfx_display::GfxDisplay>)
    }

    pub fn get_gr2(&self) -> Option<Arc<crate::dev::gr2::Gr2>> {
        self.board.gr2.clone()
    }

    /// The second head's display (Newport dual-head only).
    pub fn get_display_head1(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> {
        self.board.rex3_head1.clone().map(|r| r as Arc<dyn crate::gfx_display::GfxDisplay>)
    }

    pub fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timer_manager.clone()
    }

    /// Return a type-erased CpuDebug handle for the GDB stub.
    pub fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> {
        self.cpu.clone().debug_adapter()
    }

    /// Load a static ELF32 MSB binary into RAM and set PC to its entry point
    /// (`--load-elf`, and the monitor's `loadelf`). CPU must be stopped.
    pub fn load_elf(&self, path: &str) -> Result<String, String> {
        // Before POST every bank is invalid (MEMCFG0/1 = 0) and writes to RAM
        // addresses go to UnmappedRam, so map the banks as POST would first.
        let mapped = self.mc.post_map_banks();
        let out = self.cpu.load_elf(path)?;
        Ok(Self::note_banks(mapped, out))
    }

    /// `load_elf` for an image already in memory — see
    /// `MipsCpu::load_elf_bytes`. `name` only labels errors.
    pub fn load_elf_bytes(&self, bytes: &[u8], name: &str) -> Result<String, String> {
        let mapped = self.mc.post_map_banks();
        let out = self.cpu.load_elf_bytes(bytes, name)?;
        Ok(Self::note_banks(mapped, out))
    }

    fn note_banks(mapped: bool, out: String) -> String {
        if mapped { format!("  (mapped RAM banks; POST has not run)\n{}", out) } else { out }
    }

    /// The in-process serial backend used by `--ci` mode. `None` in
    /// interactive mode.
    pub fn get_ci_serial(&self) -> Option<Arc<crate::dev::serial_host::CiSerialBackend>> {
        self.ci_serial.clone()
    }

    /// Type bytes at the IRIX serial console (tty1) in-process, without any
    /// loopback TCP client. Used by the GUI to send `halt\n` for a clean
    /// shutdown so the feature doesn't depend on the serial server socket.
    pub fn inject_serial_console(&self, bytes: &[u8]) {
        self.hpc3.ioc().scc().inject_b(bytes);
    }

    /// Read (and consume) IRIX serial-console (tty1) output captured in-process
    /// since the last call. Pairs with `inject_serial_console` to drive a
    /// request/response probe over the console without a loopback TCP client.
    pub fn read_serial_console(&self) -> Vec<u8> {
        self.hpc3.ioc().scc().drain_console()
    }

    /// csh one-liner to (re)mount `/CDROM` for drive `16 * controller + id`
    /// (EFS s7, else iso9660 vol).
    pub fn irix_cdrom_remount_command(drive: u8) -> String {
        let (c, id) = (drive / 16, drive % 16);
        format!(
            "umount /CDROM >& /dev/null; \
             mount -t efs -o ro /dev/dsk/dks{c}d{id}s7 /CDROM || \
             mount -t iso9660 /dev/rdsk/dks{c}d{id}vol /CDROM\n"
        )
    }

    /// Best-effort `/CDROM` remount via the IRIX serial console (tty1).
    /// Works when a login shell or xterm on the console is active.
    pub fn remount_cdrom_guest(&self, scsi_id: u8) {
        let script = Self::irix_cdrom_remount_command(scsi_id);
        eprintln!(
            "IRIX: remount /CDROM for SCSI #{scsi_id} (console shell must be active)"
        );
        self.inject_serial_console(script.as_bytes());
    }

    /// CPU thread, started explicitly by the CI `start` command or by
    /// `ci_restore`. In `--ci` mode the CPU is not autostarted in `start()`
    /// — the harness drives startup via `restore`.
    pub fn cpu_start(&self) {
        if let Some(fault) = &self.fatal_fault {
            eprintln!("iris: refusing to start CPU after fatal device failure: {fault}");
            return;
        }
        self.cpu.start();
    }

    /// Whether the CPU thread is currently executing. Goes false when the CPU
    /// is stopped — including the soft power-off path (a guest `poweroff` makes
    /// the machine-events thread call `stop()`), so an embedder can tell the
    /// guest has shut down without subscribing to machine events.
    pub fn cpu_is_running(&self) -> bool {
        self.cpu.is_running()
    }

    /// The kernel's clock-tick counter — CP0 Compare matches, or the IOC's
    /// 8254 timer 0/1 interrupts when IRIX uses those instead (see
    /// `Ioc::set_clock_ticks`; the two never both run). Read a delta over wall
    /// time to get the guest's Hz, the same math the CLI status bar's
    /// `BarStats::fasttick` uses (`src/disp.rs`).
    pub fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.cpu.fasttick_count()
    }

    /// Where the GIO card of `ty` sits, if there is one.
    fn gio_at(&self, ty: &str) -> Option<&crate::slot_path::MachineSlotPath> {
        self.gio_slots.iter().find(|(t, _)| t == ty).map(|(_, at)| at)
    }

    /// Every SCSI bus on the board: the first controller's, and the second's
    /// on a Fullhouse Indigo2.
    pub(crate) fn scsi_buses(&self) -> Vec<&crate::scsi::ScsiBus> {
        use crate::scsi_monitor::ScsiTargets;
        let mut v = vec![self.hpc3.scsi().scsi_bus()];
        v.extend(self.hpc3.scsi1().map(|c| c.scsi_bus()));
        v
    }

    /// Number of attached CHD disks whose `.diff.chd` holds changes pending a
    /// fold-back into the base on a clean shutdown (the "Synchronizing disks"
    /// step). 0 means a clean exit needs no disk sync.
    pub fn pending_chd_sync_count(&self) -> usize {
        self.scsi_buses().iter().map(|b| b.pending_chd_sync_count()).sum()
    }

    /// Fold pending CHD diffs back into their bases, preserving compression.
    /// `only` limits it to one SCSI id; `None` means all.
    /// Call only after the guest has stopped (so disk I/O is quiesced).
    /// `progress(done, total, fraction)` drives a UI; `cancel()` aborts cleanly,
    /// leaving un-synced bases+diffs intact. Returns the count synced.
    pub fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        crate::scsi::ScsiBus::sync_chd_disks(&self.scsi_buses(), only, progress, cancel)
    }

    /// Step the CPU `n` instructions in-line on the calling thread, with all
    /// peripheral threads stopped so the CPU sees no external interrupts.
    /// Used by Phase 3.3 snapshot determinism validator.
    /// Caller must arrange `load_snapshot_paused` first.
    pub fn cpu_step_n_inline(&self, n: u64) -> Result<u64, String> {
        self.cpu.step_n_inline(n)
    }

    /// Snapshot the deterministic-from-state CPU registers.
    pub fn cpu_state_digest(&self) -> Result<crate::cpu::mips_exec::CpuStateDigest, String> {
        self.cpu.state_digest()
    }

    /// Step exactly one architectural instruction and return how many
    /// `step()` retired (usually 1; can be 2+ under real JIT dispatch — see
    /// `MipsCpu::step_one_inline_counting_instructions`'s doc comment).
    #[cfg(feature = "developer")]
    pub fn cpu_step_one_inline_counting_instructions(&self) -> Result<usize, String> {
        self.cpu.step_one_inline_counting_instructions()
    }

    /// Restore CPU registers from a `CpuStateDigest` — see
    /// `MipsCpu::restore_state_digest`'s doc comment.
    #[cfg(feature = "developer")]
    pub fn cpu_restore_state_digest(&self, digest: &crate::cpu::mips_exec::CpuStateDigest) -> Result<(), String> {
        self.cpu.restore_state_digest(digest)
    }

    /// Force `cp0_count`/`count_hz` to a reference digest's values — see
    /// `MipsCpu::fixup_cp0_count`'s doc comment.
    #[cfg(feature = "developer")]
    pub fn cpu_fixup_cp0_count(&self, digest: &crate::cpu::mips_exec::CpuStateDigest) -> Result<(), String> {
        self.cpu.fixup_cp0_count(digest)
    }

    /// Enable/disable real JIT dispatch at runtime; returns the previous
    /// value. See `MipsExecutor::jitv2_dispatch_enabled`'s doc comment.
    #[cfg(all(feature = "jitv2", feature = "developer"))]
    pub fn cpu_set_jitv2_dispatch_enabled(&self, enabled: bool) -> Result<bool, String> {
        self.cpu.set_jitv2_dispatch_enabled(enabled)
    }

    /// Arm/disarm `jitcheck`'s hardware-read fixup recording — see
    /// `MipsExecutor::hw_read_fixup_recording`'s doc comment.
    #[cfg(feature = "developer")]
    pub fn cpu_set_hw_read_fixup_recording(&self, recording: bool) -> Result<(), String> {
        self.cpu.set_hw_read_fixup_recording(recording)
    }

    /// Set/clear the recorded hardware-read values to substitute for the
    /// next replay-pass step — see
    /// `MipsExecutor::hw_read_fixup_replay`'s doc comment.
    #[cfg(feature = "developer")]
    pub fn cpu_set_hw_read_fixup_replay(&self, fixups: Option<Vec<(u64, u8, u64)>>) -> Result<(), String> {
        self.cpu.set_hw_read_fixup_replay(fixups)
    }

    #[cfg(feature = "wgpu")]
    fn capture_resident_checkpoint(rex: Option<&Rex3>) -> Result<Option<ResidentCheckpoint>, String> {
        rex.filter(|rex| rex.resident_active())
            .map(|rex| rex.snapshot_resident_checkpoint().map_err(|error| error.to_string()))
            .transpose()
    }

    /// The machine as it is, for `j2 trace` and `j2 replay` to rewind to: SCSI registers, not
    /// disk contents. Leaves it stopped, as the restore that follows wants.
    pub(crate) fn capture_live_checkpoint(&mut self) -> crate::snapshot::Image<'static> {
        self.stop();
        self.capture_image().unwrap_or_else(|error| panic!("{error}"))
    }

    /// Put back a checkpoint of [`Machine::capture_live_checkpoint`]. The peripherals run on and
    /// the CPU stays stopped, for the caller's inline single-stepping.
    pub(crate) fn restore_live_checkpoint(&mut self, cp: &crate::snapshot::Image) -> Result<(), String> {
        self.stop();
        self.apply_image(cp)?;
        self.restart_peripherals();
        Ok(())
    }

    /// Restart everything `stop` stopped, less the CPU, without restarting the
    /// monitor server.
    fn restart_peripherals(&mut self) {
        if let Some(fault) = &self.fatal_fault {
            eprintln!("iris: refusing to restart peripherals after fatal device failure: {fault}");
            return;
        }
        self.timer_manager.clock().start();
        self.mc.start();
        self.hpc3.start();
        // A board whose resident raster has faulted panics in `start`; a failed save restarts too.
        #[cfg(feature = "wgpu")]
        let faulted = |r: &Rex3| r.resident_fault().is_some();
        #[cfg(not(feature = "wgpu"))]
        let faulted = |_: &Rex3| false;
        if let Some(rex3) = self.board.rex3.as_ref().filter(|r| !faulted(r)) { rex3.start(); }
        if let Some(mgras) = &self.board.mgras { mgras.start_display(); }
        if let Some(rex3) = self.board.rex3_head1.as_ref().filter(|r| !faulted(r)) { rex3.start(); }
        if let Some(gr2) = &self.board.gr2 { gr2.start(); }
        if let Some(u64) = &self.board.ultra64 { u64.start(); }
        if let Some(v) = &self.board.virtio { v.start(); }
        if let Some(v) = &self.board.virtio_9p { v.start(); }
    }

    /// Helper to power-on reset all devices.
    /// Must be called with threads stopped.
    fn power_on_devices(&mut self) {
        self.event_tx.powered_on();
        self.cpu.power_on();
        self.board.phys.reset_memory();
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
        if let Some(dev) = self.hpc3.scsi1() { dev.power_on(); }
        // Seeq/Ethernet: reset regs + signal NAT flush.
        self.hpc3.seeq().power_on();
        // HAL2: reset all audio registers and channel state (timers already stopped).
        if let Some(hal2) = self.hpc3.hal2() { hal2.power_on(); }
        self.hpc3.power_on();
        if let Some(rex3) = &self.board.rex3 { rex3.power_on(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.power_on(); }
        if let Some(gr2) = &self.board.gr2 { gr2.power_on(); }
        if let Some(td) = &self.board.testdev { td.power_on(); }
        if let Some(v) = &self.board.virtio { v.power_on(); }
        if let Some(v) = &self.board.virtio_9p { v.power_on(); }
        self.apply_host_display_resolution();
    }

    /// Stop all threads, power-on reset every device in-place, restart peripherals.
    /// The CPU is left stopped — the monitor `run` command (or debugger) should start it.
    pub fn reset(&mut self) {
        self.stop();
        if let Some(fault) = &self.fatal_fault {
            eprintln!("iris: refusing to reset machine after fatal device failure: {fault}");
            return;
        }

        self.power_on_devices();

        // Restart peripherals (not monitor — it stays alive)
        self.restart_peripherals();
    }

    /// Load `saves/<name>/` and leave every thread stopped, for the determinism validator.
    pub fn load_snapshot_paused(&mut self, name: &str) -> Result<(), String> {
        let kept = crate::snapshot::read(self, std::path::Path::new("saves"), name)?;
        crate::snapshot::restore(self, &kept, false)?;
        self.hpc3.stop();
        self.mc.stop();
        if let Some(rex3) = &self.board.rex3 { rex3.stop(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.stop(); }
        if let Some(gr2) = &self.board.gr2 { gr2.stop(); }
        Ok(())
    }

    /// Call `f` with each device kept as one table, by file name, in the order they load.
    fn each_saved_device(&self, f: &mut dyn FnMut(&str, &dyn Saveable) -> Result<(), String>) -> Result<(), String> {
        let (ioc, hpc3) = (self.hpc3.ioc(), &self.hpc3);
        f("cpu", &*self.cpu)?;
        f("mc", &self.mc)?;
        f("ioc", ioc)?;
        f("scc", ioc.scc())?;
        f("pit", ioc.pit())?;
        f("ps2", &*ioc.ps2())?;
        f("rtc", &**hpc3.rtc())?;
        f("eeprom", &**hpc3.eeprom())?;
        f("scsi", &**hpc3.scsi())?;
        if let Some(scsi1) = hpc3.scsi1() {
            f("scsi1", &**scsi1)?;
        }
        f("seeq", &**hpc3.seeq())?;
        f("hpc3", hpc3)?;
        if let (Some(td), Some(at)) = (&self.board.testdev, self.gio_at("test_device")) {
            f(&at.to_string(), &**td)?;
        }
        if let (Some(v), Some(at)) = (&self.board.virtio, self.gio_at("virtio_serial")) {
            f(&at.to_string(), v.core())?;
        }
        Ok(())
    }

    /// A Newport head's registers and planes, whichever raster holds them.
    fn capture_rex3(rex3: &Rex3) -> Result<(toml::Value, (Vec<u32>, Vec<u32>)), String> {
        #[cfg(feature = "wgpu")]
        if let Some(resident) = Self::capture_resident_checkpoint(Some(rex3))? {
            rex3.resident_status().map_err(|e| e.to_string())?;
            return Ok(resident);
        }
        Ok((rex3.save_state(), rex3.snapshot_framebuffers_inmem()))
    }

    fn capture_image(&self) -> Result<crate::snapshot::Image<'static>, String> {
        use crate::snapshot::{Board, BoardList};
        let mut image = crate::snapshot::Image::default();
        self.each_saved_device(&mut |name, dev| Ok(image.states.push((name.into(), dev.save_state()))))?;
        if let Some(rex3) = &self.board.rex3 {
            let (state, planes) = Self::capture_rex3(rex3)?;
            image.states.push(("rex3".into(), state));
            image.framebuffer = Some(planes);
        }
        if let Some(rex3) = &self.board.rex3_head1 {
            let (state, (rgb, aux)) = Self::capture_rex3(rex3)?;
            image.states.push(("rex3_head1".into(), state));
            let be = |words: Vec<u32>| words.iter().flat_map(|w| w.to_be_bytes()).collect();
            image.files = vec![("rex3_head1_rgb.bin".into(), be(rgb)), ("rex3_head1_aux.bin".into(), be(aux))];
        }
        let mgras = self.board.mgras.as_ref().map(|m| m.snapshot_parts().map(|p| Board::new("mgras", BoardList::Impact, p)));
        let gr2 = self.board.gr2.as_ref().map(|g| g.snapshot_parts().map(|p| Board::new("gr2", BoardList::Gr2, p)));
        image.boards = mgras.into_iter().chain(gr2).collect::<Result<_, _>>()?;
        image.memory = (0..4).map(|i| (i, self.board.phys.snapshot_bank_inmem(i).into())).collect();
        Ok(image)
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        // Runs before any field is dropped. Taking the lock waits for an
        // in-flight monitor/event/CI command to finish, so none of them can see
        // a half-destroyed Machine. No controller command drops the Machine, so
        // this never waits on a lock its own thread holds.
        if let Some(slot) = self.controller_slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        // The guest clock points into this Machine's CPU core.
        #[cfg(feature = "perfetto")]
        crate::perfetto::set_guest_cycles(None);
        // The CPU, its compile pool and the device threads first: they reach
        // the core, the bus and the devices, which are freed once this returns.
        self.stop();
        #[cfg(feature = "jitv2")]
        crate::cpu::jitv2::Jitv2::stop_for_drop(&self.cpu.jitv2(), || self.cpu.clone().as_device().stop());
        crate::workstation::flush_on_drop(self);
        // Release the monitor port and this Machine's device handles, so the next
        // Machine in the same process (iris-gui Stop -> Start) gets the monitor.
        self.monitor.shutdown();
        // VINO's DMA thread runs from construction. It holds the bus.
        self.board.vino.stop();
        self.hpc3.release();
        // The bus holds the devices and they hold the bus (DMA): let go, or
        // none of it is freed.
        self.board.phys.release_devices();
    }
}

/// The Indy and Indigo2 family as the front ends see it: each method is the
/// inherent one of the same name.
impl Workstation for Machine {
    fn start(&mut self) { Machine::start(self) }
    fn stop(&mut self) { Machine::stop(self) }
    fn reset(&mut self) { Machine::reset(self) }
    fn restart_peripherals(&mut self) { Machine::restart_peripherals(self) }
    fn cpu_start(&self) { Machine::cpu_start(self) }
    fn cpu_is_running(&self) -> bool { Machine::cpu_is_running(self) }
    fn machine_slot(&mut self) -> MachineSlot { Machine::machine_slot(self) }
    fn register_system_controller(&mut self) { Machine::register_system_controller(self) }
    fn monitor(&self) -> &Arc<Monitor> { &self.monitor }
    fn lifecycle(&self) -> GuestState { self.lifecycle }
    fn set_lifecycle(&mut self, state: GuestState) { self.lifecycle = state; }
    fn fatal_fault(&self) -> Option<MachineFatal> { self.fatal_fault.clone() }
    fn latch_fatal_fault(&mut self, fault: MachineFatal) -> bool {
        if self.fatal_fault.is_some() {
            false
        } else {
            self.fatal_fault = Some(fault);
            true
        }
    }

    fn get_display(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> { Machine::get_display(self) }
    fn get_display_head1(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> { Machine::get_display_head1(self) }
    fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> { Machine::get_ps2(self) }
    fn register_monitor_device(&self, device: Arc<dyn crate::traits::Device>) { self.monitor.register_device(device); }
    fn get_timer_manager(&self) -> Arc<TimerManager> { Machine::get_timer_manager(self) }
    fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> { Machine::get_cpu_debug(self) }
    fn fasttick_count(&self) -> Arc<AtomicU64> { Machine::fasttick_count(self) }
    fn get_ci_serial(&self) -> Option<Arc<crate::dev::serial_host::CiSerialBackend>> { Machine::get_ci_serial(self) }
    fn inject_serial_console(&self, bytes: &[u8]) { Machine::inject_serial_console(self, bytes) }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn load_elf(&self, path: &str) -> Result<String, String> { Machine::load_elf(self, path) }
    fn snapshot_refusal(&self) -> Option<String> {
        self.board.virtio_9p.as_ref().map(|_| "a virtio-9p share keeps the guest's open files, which a snapshot does not hold".into())
    }
    fn snapshot_layout(&self) -> crate::snapshot::Layout {
        use crate::snapshot::BoardList;
        let mut states = Vec::new();
        let _ = self.each_saved_device(&mut |name, _| Ok(states.push(name.to_string())));
        states.extend(self.board.rex3.iter().map(|_| "rex3".to_string()));
        states.extend(self.board.rex3_head1.iter().map(|_| "rex3_head1".to_string()));
        let head1 = |f: &str| (f.to_string(), 0);
        let boards = [self.board.mgras.is_some().then_some(("mgras", BoardList::Impact)), self.board.gr2.is_some().then_some(("gr2", BoardList::Gr2))];
        crate::snapshot::Layout {
            disks: self.disks.clone(),
            slots: self.gio_slots.clone(),
            nvram: Some(self.nvram_path.clone()),
            cpu: (self.cpu.model_name().into(), self.cpu.cache_kind().into()),
            states,
            files: if self.board.rex3_head1.is_some() { vec![head1("rex3_head1_rgb.bin"), head1("rex3_head1_aux.bin")] } else { Vec::new() },
            memory: self.board.phys.banks().iter().map(|b| b.size() / 4).collect(),
            framebuffer: self.board.rex3.is_some(),
            boards: boards.into_iter().flatten().collect(),
            ..Default::default()
        }
    }
    fn capture(&self) -> Result<crate::snapshot::Image<'_>, String> { self.capture_image() }
    fn apply_image(&mut self, image: &crate::snapshot::Image) -> Result<(), String> {
        self.power_on_devices();
        self.each_saved_device(&mut |name, dev| dev.load_state(image.state(name)?).map_err(|e| format!("{name}: {e}")))?;
        if let Some(rex3) = &self.board.rex3 {
            rex3.load_state(image.state("rex3")?)?;
        }
        if let Some(rex3) = &self.board.rex3_head1 {
            rex3.load_state(image.state("rex3_head1")?)?;
            let planes = |f| image.file(f).map(|b| b.chunks_exact(4).map(|w| u32::from_be_bytes(w.try_into().unwrap())).collect::<Vec<u32>>());
            rex3.try_restore_framebuffers_inmem(&planes("rex3_head1_rgb.bin")?, &planes("rex3_head1_aux.bin")?)
                .map_err(|e| format!("restore REX3 head 1 framebuffers: {e}"))?;
        }
        for (bank, words) in &image.memory {
            self.board.phys.restore_bank_inmem(*bank, words);
        }
        if let (Some(rex3), Some((rgb, aux))) = (&self.board.rex3, &image.framebuffer) {
            rex3.try_restore_framebuffers_inmem(rgb, aux).map_err(|e| format!("restore REX3 framebuffers: {e}"))?;
        }
        if let Some(mgras) = &self.board.mgras {
            let b = image.board("mgras").ok_or("image has no state for the configured IMPACT board")?;
            mgras.restore_snapshot_parts(&b.metadata, &b.parts)?;
        }
        if let Some(gr2) = &self.board.gr2 {
            let b = image.board("gr2").ok_or("image has no state for the configured GR2 board")?;
            gr2.restore_snapshot_parts(&b.metadata, &b.parts)?;
        }
        Ok(())
    }
    fn with_disk_buses(&self, f: &mut dyn FnMut(&[crate::snapshot::DiskBus]) -> Result<(), String>) -> Result<(), String> {
        f(&crate::snapshot::disk_buses((0..).map(Hpc3::scsi_at), self.scsi_buses(), None))
    }
    fn snapshots(&mut self) -> Option<&mut crate::snapshot::Restored> { Some(&mut self.restored) }

    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        let rtc = self.hpc3.rtc();
        let path = path.unwrap_or_else(|| rtc.nvram_path()).to_string();
        rtc.save_nvram(&path).map(|_| path).map_err(|e| e.to_string())
    }
    fn scratch_path(&self) -> Option<&std::path::Path> { Machine::scratch_path(self) }
    fn disc_changer(&self) -> Option<Arc<dyn crate::workstation::DiscChanger>> {
        let ctls = std::iter::once(self.hpc3.scsi()).chain(self.hpc3.scsi1());
        Some(Arc::new(crate::workstation::DiscChangers(ctls.map(|c| c.clone() as _).collect())))
    }
    fn remount_cdrom_guest(&self, scsi_id: u8) { Machine::remount_cdrom_guest(self, scsi_id) }
    fn pending_chd_sync_count(&self) -> usize { Machine::pending_chd_sync_count(self) }
    fn disk_activity(&self) -> Vec<Vec<crate::scsi::DiskCounts>> {
        self.scsi_buses().iter().map(|b| b.activity()).collect()
    }
    fn read_disk(&self, bus: usize, id: u8, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.scsi_buses().get(bus).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no such disk bus"))?
            .read_media(id, offset, buf)
    }
    fn disk_size(&self, bus: usize, id: u8) -> Option<u64> {
        self.scsi_buses().get(bus).and_then(|b| b.media_size(id))
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.scsi_buses().iter().try_for_each(|b| b.flush())
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        Machine::sync_chd_disks(self, only, progress, cancel)
    }
    fn net_frames(&self) -> Option<Arc<std::sync::atomic::AtomicU64>> {
        self.hpc3.seeq().frames()
    }
    fn nat_control(&self) -> Option<Arc<crate::net::NatControl>> {
        Some(self.hpc3.seeq().nat_control())
    }
    fn nat_expected(&self) -> Option<(std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr)> {
        Some(self.hpc3.seeq().gateway_addrs())
    }
}

#[cfg(test)]
mod emulator_name_tests {
    use super::*;

    /// No profile's expansion may name another machine: an Indigo2 is never
    /// an "Indy", an Indy never "IP22", and the O2 is only ever "IP32".
    #[test]
    fn thirds_name_only_the_configured_machine() {
        let foreign = |p: Option<MachineProfile>| -> &'static [&'static str] {
            match p {
                Some(MachineProfile::IndyIp24) => &["Indigo2", "IP22", "IP28", "IP32", "O2"],
                Some(MachineProfile::Indigo2Ip22) => &["Indy", "IP24", "IP28", "IP32", "O2"],
                Some(MachineProfile::Indigo2Ip28) => &["Indy", "IP24", "IP22", "IP32", "O2"],
                Some(MachineProfile::O2Ip32) => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP30", "O2"],
                Some(MachineProfile::OctaneIp30) => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP32", "O2"],
                Some(MachineProfile::Origin200Ip29 | MachineProfile::Origin2000Ip27 | MachineProfile::Onyx2Ip27 | MachineProfile::FuelIp34) => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP30", "IP32", "O2"],
                Some(MachineProfile::TezroIp53) => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP30", "IP32", "IP34", "O2"],
                Some(MachineProfile::Origin3200Ip35 | MachineProfile::Onyx3000Ip35) => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP30", "IP32", "O2"],
                None => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP30", "IP32", "O2"],
            }
        };
        for p in std::iter::once(None).chain(MachineProfile::EVERY.into_iter().map(Some)) {
            let thirds = emulator_name_thirds(p);
            assert!(!thirds.is_empty());
            for w in thirds {
                assert!(w.starts_with('I'), "{p:?}: {w} breaks the acronym");
                assert!(!foreign(p).contains(w), "{p:?}: {w} names another machine");
            }
        }
    }

    #[test]
    fn profile_round_trips_through_the_global() {
        for p in MachineProfile::EVERY {
            set_emulator_profile(p);
            assert_eq!(emulator_profile(), Some(p));
        }
        let name = emulator_name();
        assert!(name.starts_with('I') && name.contains(" Rust "), "{name}");
    }
}

/// A dropped machine gives back its decode table: a process may have only 64,
/// and a test binary builds many machines.
#[cfg(test)]
mod drop_tests {
    use super::*;
    use crate::config::{GraphicsBoard, MachineProfile};
    use crate::display_stop_tests::{big_stack, config, scratch};

    #[test]
    fn a_dropped_machine_frees_its_bus_and_decode_table() {
        let dir = scratch("ip22-drop");
        for (profile, board) in [
            (MachineProfile::IndyIp24, GraphicsBoard::Newport),
            (MachineProfile::Indigo2Ip22, GraphicsBoard::Xz),
            (MachineProfile::Indigo2Ip28, GraphicsBoard::SolidImpact),
        ] {
            let cfg = config(&dir, profile, board);
            let machine = big_stack(move || Machine::new(cfg));
            let bus = Arc::downgrade(&machine.board.phys);
            let cpu = Arc::downgrade(&machine.cpu);
            drop(machine);
            assert_eq!(bus.strong_count(), 0, "{profile:?} with {board:?}: the bus outlives its machine");
            assert_eq!(cpu.strong_count(), 0, "{profile:?} with {board:?}: the CPU outlives its machine");
        }
    }
}

#[cfg(test)]
mod controller_lifetime_tests {
    use super::*;
    use crate::test_temp::{nvram_in, TempPath};

    /// `nv` holds the machine's NVRAM files, so a stop writes them there and
    /// not into the working directory.
    fn build_machine(nv: &TempPath) -> Box<Machine> {
        let nv = nv.to_path_buf();
        // Machine::new needs more stack than a test thread has (see main.rs).
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut cfg = crate::bench_runner::bench_config([16, 0, 0, 0], &parts::R4400_150);
                nvram_in(&mut cfg, &nv);
                Box::new(Machine::new(cfg))
            })
            .unwrap()
            .join()
            .unwrap()
    }

    /// A reset, or a restore of a live checkpoint, replaces the machine that
    /// asked for the hard resets still queued: the event thread drops them.
    #[test]
    fn a_reset_or_restore_makes_a_queued_reset_stale() {
        let nv = TempPath::dir("indy-stale-events");
        let mut machine = build_machine(&nv);
        let queue = machine.event_rx.take().unwrap();
        queue.listen();
        let send = |m: &Machine| {
            m.event_tx.try_send(crate::traits::MachineEvent::HardReset).unwrap();
            queue.recv().unwrap().0
        };
        let sent = send(&machine);
        assert!(queue.is_current(sent));
        assert_eq!(machine.cpu.park_slot().holds(), crate::cpu::smp::HOLD_RESET, "the CPU stops at the request");
        machine.reset();
        assert!(!queue.is_current(sent), "a reset");
        assert_eq!(machine.cpu.park_slot().holds(), 0, "and lets the CPU go");
        let sent = send(&machine);
        let live = machine.capture_live_checkpoint();
        machine.restore_live_checkpoint(&live).unwrap();
        assert!(!queue.is_current(sent), "a restore");
        machine.stop();
    }

    /// The Indy's console log adds to an existing file, and starts it over
    /// when `serial_log_overwrite` is set.
    #[test]
    fn the_console_log_appends_unless_told_to_overwrite() {
        let nv = TempPath::dir("indy-serial-log");
        for (overwrite, want) in [(false, &b"stale"[..]), (true, &b""[..])] {
            let log = nv.join("console.log");
            std::fs::write(&log, "stale").unwrap();
            let (dir, log) = (nv.to_path_buf(), log.to_string_lossy().into_owned());
            let _machine = std::thread::Builder::new()
                .stack_size(64 * 1024 * 1024)
                .spawn(move || {
                    let mut cfg = crate::bench_runner::bench_config([16, 0, 0, 0], &parts::R4400_150);
                    nvram_in(&mut cfg, &dir);
                    cfg.serial_log = Some(log);
                    cfg.serial_port_a = Some(0);
                    cfg.serial_port_b = Some(0);
                    cfg.serial_log_overwrite = overwrite;
                    Box::new(Machine::new(cfg))
                })
                .unwrap()
                .join()
                .unwrap();
            assert_eq!(std::fs::read(nv.join("console.log")).unwrap(), want, "serial_log_overwrite = {overwrite}");
        }
    }

    /// #102: `machine-start`'s acknowledgement names the transition it
    /// performed, from a never-started machine through `Starting` to
    /// `Running`. A second start is a no-op, and a start after `machine-stop`
    /// takes the same route from `Halted`. The process-level test in
    /// `tests/monitor_ready.rs` covers the monitor command and readiness.
    #[test]
    fn start_guest_acknowledges_the_lifecycle_transition() {
        let nv = TempPath::dir("machine-nv");
        let mut machine = build_machine(&nv);
        assert_eq!(machine.status().state, GuestState::Idle);
        assert!(!machine.status().monitor_ready(), "no monitor before Machine::start");

        let ack = machine.start_guest().expect("start from idle");
        assert_eq!(ack.text(), "idle -> starting -> running");
        assert_eq!(machine.status().state, GuestState::Running);
        assert!(machine.status().cpu_running);

        // Duplicate start: no second start and no error.
        let ack = machine.start_guest().expect("duplicate start");
        assert!(!ack.changed());
        assert_eq!(ack.text(), "running (no change)");

        let ack = machine.stop_guest();
        assert_eq!(ack.text(), "running -> halted");
        assert_eq!(machine.status().state, GuestState::Halted);

        let ack = machine.start_guest().expect("start after halt");
        assert_eq!(ack.text(), "halted -> starting -> running");
        machine.stop();
    }

    #[test]
    fn fatal_device_event_stops_machine_and_blocks_restart() {
        let nv = TempPath::dir("machine-fatal-nv");
        let mut machine = build_machine(&nv);
        machine.start();
        let slot = machine.machine_slot();
        let (tx, rx) = std::sync::mpsc::channel();
        crate::workstation::register_fatal_controller(slot.clone(), Some(rx));

        let first = MachineFatal::new("test GPU", "device lost");
        tx.send(first.clone()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let fault = crate::workstation::with_machine_slot(&slot, |machine| machine.fatal_fault()).unwrap();
            if fault.is_some() {
                assert_eq!(fault, Some(first.clone()));
                break;
            }
            assert!(std::time::Instant::now() < deadline, "fatal controller did not latch the failure");
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        assert!(machine.start_guest().unwrap_err().contains("device lost"));
        machine.start();
        machine.cpu_start();
        assert!(!machine.cpu_is_running());

        tx.send(MachineFatal::new("second", "ignored")).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let latched = crate::workstation::with_machine_slot(&slot, |machine| machine.fatal_fault()).unwrap();
        assert_eq!(latched, Some(first));
    }

    /// A guest power-off ends the process without dropping the machine (and so
    /// without the disks' Drop), so the power-off path has to flush the disks
    /// itself. Drives the event the IOC sends and looks for the overlay's
    /// sector record to be compacted: writes append to it, and only a flush
    /// rewrites its count.
    #[test]
    fn guest_power_off_flushes_overlay_disks() {
        use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};
        let nv = TempPath::dir("machine-nv");
        let nv_dir = nv.to_path_buf();
        let path = nv.join("disk");
        std::fs::write(&path, vec![0u8; 8 * 512]).unwrap();
        let sidecar = format!("{}.overlay.dirty", path.display());

        let path_str = path.to_str().unwrap().to_string();
        let mut machine = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut cfg = crate::bench_runner::bench_config([16, 0, 0, 0], &parts::R4400_150);
                nvram_in(&mut cfg, &nv_dir);
                cfg.set_disk(1, crate::config::ScsiDeviceConfig { path: path_str, overlay: true, ..Default::default() });
                Box::new(Machine::new(cfg))
            })
            .unwrap()
            .join()
            .unwrap();
        for lba in [3, 4] {
            let write = ScsiRequest {
                cdb: vec![scsi_cmd::WRITE_10, 0, 0, 0, 0, lba, 0, 0, 1, 0],
                data_len: ScsiDataLength::Unlimited,
                data_in: Some(vec![0xA5; 512]),
            };
            machine.scsi_buses()[0].request(1, &write).unwrap().unwrap();
        }
        // The record's count, which a flush rewrites.
        let counted = || {
            std::fs::read(&sidecar)
                .ok()
                .and_then(|b| b.get(..8).map(|c| u64::from_le_bytes(c.try_into().unwrap())))
        };
        assert_eq!(counted(), Some(1), "nothing has flushed yet: the second sector is only appended");

        // The handler would otherwise exit this test process.
        std::env::set_var("IRIS_NO_EXIT_ON_POWEROFF", "1");
        machine.register_system_controller();
        machine.event_tx.send(crate::traits::MachineEvent::PowerOff).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while counted() != Some(2) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert_eq!(counted(), Some(2), "power-off did not flush the overlay's record");
        let held = crate::cow_disk::load_dirty_sidecar(std::path::Path::new(&sidecar)).unwrap_or_default();
        assert!(held.contains(&3) && held.contains(&4), "power-off left no record of the written sectors");

        // The machine closes its disks as it drops, before `nv` goes.
        drop(machine);
    }

    /// iris-gui drops a Machine on Stop while the monitor keeps its
    /// SystemController. Every command through that controller afterwards used
    /// to dereference freed memory (SIGSEGV in `save_snapshot`).
    #[test]
    fn controller_outliving_its_machine_reports_not_running() {
        let nv = TempPath::dir("machine-nv");
        let mut machine = build_machine(&nv);
        machine.register_system_controller();
        let ctrl = SystemController { machine: machine.machine_slot() };
        drop(machine);
        for (cmd, args) in [("save", &["crashtest"][..]), ("reset", &[][..]), ("machine-stop", &[][..])] {
            let err = ctrl.execute_command(cmd, args, Box::new(std::io::sink())).unwrap_err();
            assert_eq!(err, "machine is not running", "{cmd}");
        }
    }
}

#[cfg(test)]
mod construction_error_tests {
    use super::*;
    use crate::display_stop_tests::big_stack;
    use crate::test_temp::TempPath;

    #[test]
    fn checked_constructors_return_explicit_prom_errors() {
        let path = TempPath::new("machine-prom-missing");
        let expected = path.display().to_string();
        let mut cfg = MachineConfig::default();
        cfg.prom = expected.clone();
        let with_testdev = cfg.clone();

        let error = big_stack(move || Machine::try_new(cfg).err().expect("missing PROM"));
        assert!(error.contains(&expected), "{error}");
        let error = big_stack(move || {
            Machine::try_new_with_testdev(with_testdev, None).err().expect("missing PROM")
        });
        assert!(error.contains(&expected), "{error}");
    }

    #[test]
    fn checked_constructor_returns_newport_configuration_errors() {
        let prom = TempPath::file("machine-prom", &[0, 0, 0, 0]);
        let mut cfg = MachineConfig::default();
        cfg.prom = prom.display().to_string();
        cfg.graphics.newport_mode = Some(crate::config::NewportWgpuMode::Readback);

        let error = big_stack(move || Machine::try_new(cfg).err().expect("invalid Newport mode"));
        assert!(error.contains("applies only"), "{error}");
    }
}

#[cfg(all(test, feature = "wgpu"))]
mod production_newport_wgpu_tests {
    use super::*;
    use crate::config::NewportWgpuMode;
    use crate::gfx_display::GfxDisplay;
    use crate::test_temp::{nvram_in, TempPath};

    fn production_config(nv_dir: &std::path::Path) -> MachineConfig {
        let mut cfg = MachineConfig::default();
        nvram_in(&mut cfg, nv_dir);
        cfg.devices.clear();
        cfg.no_audio = true;
        cfg.ci = true;
        cfg.monitor_port = Some(0);
        cfg.graphics.raster = crate::config::RasterUnit::Wgpu;
        cfg.graphics.gpu_backend = std::env::var("REX3_RESIDENT_BACKEND")
            .unwrap_or_else(|_| "vulkan".into());
        cfg
    }

    #[test]
    #[ignore = "requires a validated Vulkan or D3D12 adapter"]
    fn production_dual_head_wgpu_shares_device_and_restarts_idempotently() {
        let nv = TempPath::dir("newport-production-nv");
        let nv_dir = nv.to_path_buf();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut cfg = production_config(&nv_dir);
                cfg.set_device("newport", true);

                let mut machine = Machine::try_new(cfg).expect("construct production resident machine");
                let primary = machine.get_rex3().expect("primary Newport");
                let secondary = machine.get_rex3_head1().expect("secondary Newport");
                let primary_gpu = GfxDisplay::gpu(&*primary).expect("primary resident GPU");
                let secondary_gpu = GfxDisplay::gpu(&*secondary).expect("secondary resident GPU");
                assert!(Arc::ptr_eq(&primary_gpu, &secondary_gpu));
                assert!(GfxDisplay::requires_same_device_wgpu_presenter(&*primary));
                assert!(GfxDisplay::requires_same_device_wgpu_presenter(&*secondary));

                machine.start();
                machine.restart_peripherals();
                for rex in [&primary, &secondary] {
                    assert!(rex.processor_thread.lock().is_some());
                    assert!(rex.refresh_thread.lock().is_some());
                }
                machine.stop();
            })
            .expect("spawn production Newport test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }

    #[test]
    #[ignore = "requires a validated Vulkan or D3D12 adapter"]
    fn production_readback_mode_raises_frontend_capture_timeout() {
        let nv = TempPath::dir("newport-readback-nv");
        let nv_dir = nv.to_path_buf();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut cfg = production_config(&nv_dir);
                cfg.graphics.newport_mode = Some(NewportWgpuMode::Readback);
                let machine = Machine::try_new(cfg).expect("construct production resident machine");
                let rex = machine.get_rex3().expect("primary Newport");
                assert_eq!(
                    GfxDisplay::minimum_frame_request_timeout(&*rex),
                    std::time::Duration::from_secs(2)
                );
                assert!(!GfxDisplay::requires_same_device_wgpu_presenter(&*rex));
            })
            .expect("spawn production Newport test")
            .join()
            .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
    }
}

#[cfg(test)]
mod stop_order_tests {
    use super::*;
    use crate::display_stop_tests::{big_stack, Probe};
    use std::sync::atomic::Ordering::SeqCst;

    /// The graphics board can hang in its stop, so HPC3 must already be
    /// stopped when its renderer is torn down.
    #[test]
    fn stopping_the_disks_comes_before_stopping_the_graphics() {
        // A REX3 saves its shape corpus when it stops; keep it out of $HOME.
        let dir = crate::display_stop_tests::scratch("indy-order");
        let nv = dir.to_path_buf();
        let mut machine = big_stack(move || {
            let mut cfg = crate::bench_runner::bench_config([16, 0, 0, 0], &parts::R4400_150);
            crate::test_temp::nvram_in(&mut cfg, &nv);
            cfg.graphics.board = Some(GraphicsBoard::Newport);
            Box::new(Machine::new(cfg))
        });
        let hpc3 = machine.hpc3.clone();
        let (probe, overlap) = Probe::watching(move || hpc3.is_running());
        *machine.board.rex3.clone().expect("a Newport").head.renderer_slot().lock() = Some(Box::new(probe.clone()));
        machine.restart_peripherals();
        assert!(machine.hpc3.is_running(), "HPC3 is running");
        machine.stop();
        assert!(probe.stopped.load(SeqCst), "the display thread released its renderer");
        assert!(!overlap.load(SeqCst), "HPC3 was still running when the graphics stopped");
    }

    /// A CD-ROM on the Indigo2's second controller takes disc changes by its slot.
    #[test]
    fn second_controller_cdrom_takes_disc_changes() {
        use crate::config::{MachineProfile, ScsiDeviceConfig};
        use crate::workstation::Workstation;
        let dir = crate::display_stop_tests::scratch("indigo2-disc");
        let nv = dir.to_path_buf();
        let machine = big_stack(move || {
            let mut cfg = crate::bench_runner::bench_config([16, 0, 0, 0], &parts::R4400_150);
            crate::test_temp::nvram_in(&mut cfg, &nv);
            cfg.machine.profile = MachineProfile::Indigo2Ip22;
            cfg.set_disk(20, ScsiDeviceConfig { cdrom: true, ..Default::default() });
            Box::new(Machine::new(cfg))
        });
        let changer = Workstation::disc_changer(&*machine).expect("a disc changer");
        let id = crate::topology::drive_id(MachineProfile::Indigo2Ip22, "hpc3/c1/t4").unwrap();
        assert_eq!(changer.disc_status().iter().map(|d| d.0).collect::<Vec<_>>(), [id]);
        let iso = dir.join("a.iso");
        std::fs::write(&iso, vec![0u8; 2048]).unwrap();
        let path = iso.to_string_lossy().into_owned();
        assert_eq!(changer.load_disc(id, path.clone()), Ok(path));
        assert!(changer.load_disc(4, "x.iso".into()).is_err(), "controller 0 has no id 4");
    }

    /// The second controller is `dks1` to IRIX, so its drive's remount names it.
    #[test]
    fn remount_names_the_drives_controller() {
        let cmd = |drive| Machine::irix_cdrom_remount_command(drive);
        assert!(cmd(4).contains("dks0d4s7") && cmd(4).contains("dks0d4vol"));
        assert!(cmd(20).contains("dks1d4s7") && cmd(20).contains("dks1d4vol"));
    }
}

#[cfg(test)]
mod gr2_checkpoint_tests {
    use super::*;
    use crate::config::{GraphicsBoard, MachineProfile};
    use crate::dev::gr2::re3::{
        FB_W, IR_FLAT, REG_FUNC, REG_IR, REG_NUMPIX, REG_PIXMASK, REG_R, REG_XMAX, REG_YMAX,
        REG_YX, ROP_COPY,
    };
    use crate::dev::gr2::{Gr2, GR2_BASE};
    use crate::traits::{BusDevice, Resettable, BUS_BUSY};
    use std::path::Path;

    fn config(dir: &Path, board: GraphicsBoard) -> MachineConfig {
        let mut cfg = crate::display_stop_tests::config(dir, MachineProfile::Indigo2Ip22, board);
        cfg.headless = true;
        cfg
    }

    fn re3_reg(reg: usize) -> u32 {
        if reg < 0x20 { 0x6c200 + reg as u32 * 4 } else { 0x6c280 + (reg as u32 - 0x20) * 4 }
    }

    fn write(g: &Gr2, reg: usize, value: u32) {
        while g.write32(GR2_BASE + re3_reg(reg) as u64, value) == BUS_BUSY {
            std::hint::spin_loop();
        }
    }

    fn draw_test_pixel(g: &Gr2, value: u32) {
        g.start();
        for (reg, data) in [
            (REG_FUNC, ROP_COPY),
            (REG_PIXMASK, 0x00ff_ffff),
            (REG_XMAX, 1279),
            (REG_YMAX, 1023),
            (REG_R, value << 11),
            (REG_YX, (20 << 12) | 30),
            (REG_NUMPIX, 1),
            (REG_IR, IR_FLAT),
        ] {
            write(g, reg, data);
        }
        g.wait_idle();
        assert_eq!(g.vram()[20 * FB_W + 30], value);
        g.stop();
    }

    #[test]
    fn gr2_live_and_captured_checkpoints_restore_complete_state() {
        crate::display_stop_tests::big_stack(|| {
            let dir = crate::display_stop_tests::scratch("gr2-memory-checkpoints");
            let mut machine = Box::new(Machine::new(config(&dir, GraphicsBoard::Xz)));
            let gr2 = machine.get_gr2().unwrap();
            draw_test_pixel(&gr2, 0x31);

            let live = machine.capture_live_checkpoint();
            gr2.power_on();
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0);
            machine.restore_live_checkpoint(&live).unwrap();
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0x31);
            machine.stop();

            draw_test_pixel(&gr2, 0x52);
            machine.stop();
            let image = machine.capture_image().unwrap();
            gr2.power_on();
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0);
            machine.apply_image(&image).unwrap();
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0x52);
            machine.stop();

            let mut missing = machine.capture_live_checkpoint();
            missing.boards.clear();
            assert!(machine
                .restore_live_checkpoint(&missing)
                .unwrap_err()
                .contains("no state for the configured GR2 board"));
            machine.stop();
        });
    }

    /// A rollback puts back the image the restore read, and a snapshot of another machine
    /// is refused before the running machine is touched.
    #[test]
    fn rollback_keeps_the_read_image_and_a_refused_load_leaves_the_machine_alone() {
        crate::display_stop_tests::big_stack(|| {
            let dir = crate::display_stop_tests::scratch("indy-rollback");
            let saves = dir.join("saves");
            let mut machine = Box::new(Machine::new(config(&dir, GraphicsBoard::Xz)));
            let gr2 = machine.get_gr2().unwrap();
            draw_test_pixel(&gr2, 0x31);
            crate::snapshot::save_in(&mut *machine, &saves, "s").unwrap();
            crate::snapshot::ci_restore(&mut *machine, &saves, "s").unwrap();
            machine.stop();
            std::fs::remove_file(saves.join("s/snapshot.toml")).unwrap();
            draw_test_pixel(&gr2, 0x52);
            crate::snapshot::ci_rollback(&mut *machine, &saves).unwrap();
            machine.stop();
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0x31);

            crate::snapshot::save_in(&mut *machine, &saves, "other").unwrap();
            machine.stop();
            let path = saves.join("other/snapshot.toml");
            std::fs::write(&path, std::fs::read_to_string(&path).unwrap() + "machine = \"o2_ip32\"\n").unwrap();
            draw_test_pixel(&gr2, 0x66);
            let e = crate::snapshot::load_in(&mut *machine, &saves, "other").unwrap_err();
            assert!(e.contains("another machine"), "{e}");
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0x66, "the machine is as it was");
            machine.stop();
        });
    }

    #[test]
    fn gr2_disk_snapshot_round_trip_and_board_mismatches() {
        const CHILD: &str = "IRIS_GR2_M15_DISK_CHILD";
        if std::env::var_os(CHILD).is_none() {
            // Not under the working directory: a relocated target dir leaves no `target/` there.
            let path = crate::test_temp::TempPath::dir("gr2-m15-disk");
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .current_dir(&*path)
                .env(CHILD, &*path)
                .args([
                    "--exact",
                    "machine::gr2_checkpoint_tests::gr2_disk_snapshot_round_trip_and_board_mismatches",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "GR2 disk snapshot child failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        crate::display_stop_tests::big_stack(|| {
            let path = std::path::PathBuf::from(std::env::var_os(CHILD).unwrap());
            // The temp dir may sit behind a symlink (macOS /var), so compare resolved paths.
            assert_eq!(std::env::current_dir().unwrap().canonicalize().unwrap(), path.canonicalize().unwrap());

            let mut xz = Box::new(Machine::new(config(&path, GraphicsBoard::Xz)));
            let gr2 = xz.get_gr2().unwrap();
            draw_test_pixel(&gr2, 0x73);
            xz.save_snapshot("gr2-m15").unwrap();
            xz.stop();
            gr2.power_on();
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0);
            xz.load_snapshot_paused("gr2-m15").unwrap();
            assert_eq!(gr2.vram()[20 * FB_W + 30], 0x73);
            xz.stop();

            let mut extreme = Box::new(Machine::new(config(&path, GraphicsBoard::Extreme)));
            assert!(extreme
                .load_snapshot_paused("gr2-m15")
                .unwrap_err()
                .contains("running board"));
            extreme.stop();

            let mut bare = Box::new(Machine::new(config(&path, GraphicsBoard::None)));
            assert!(bare
                .load_snapshot_paused("gr2-m15")
                .unwrap_err()
                .contains("machine has no GR2 board"));
            bare.stop();
        });
    }
}

/// virtio-serial on the Indy and Indigo2: the device answers in its GIO
/// expansion slot, through the real physical map.
#[cfg(test)]
mod virtio_tests {
    use super::*;
    use crate::config::MachineProfile;
    use crate::dev::virtio::mmio::{MAGIC, REG_DEVICE_ID, REG_MAGIC};

    /// A `[[devices]]` entry from its TOML text.
    fn card(text: &str) -> crate::devices::ConfiguredDevice {
        crate::devices::ConfiguredDevice::from_table(toml::from_str(text).unwrap()).unwrap()
    }

    fn probe(profile: MachineProfile, slot: &'static str, base: u64) {
        let nv = crate::test_temp::TempPath::dir("virtio-nv");
        let nv_dir = nv.to_path_buf();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut cfg = MachineConfig::default();
                crate::test_temp::nvram_in(&mut cfg, &nv_dir);
                cfg.devices.clear();
                cfg.headless = true;
                cfg.machine.profile = profile;
                if profile == MachineProfile::Indigo2Ip28 {
                    cfg.machine.cpu = Some("r10000".into());
                    cfg.banks = [512, 512, 0, 0, 0, 0, 0, 0];
                    cfg.graphics.board = Some(crate::config::GraphicsBoard::SolidImpact);
                }
                // One unconnected port: no sockets in a unit test.
                cfg.devices.push(card(&format!("type = \"virtio_serial\"\nat = \"{slot}\"\n[[port]]")));
                let m = Machine::new(cfg);
                let phys = &m.board.phys;
                assert_eq!(phys.read32(base + REG_MAGIC).data, MAGIC, "{profile:?} {slot:?}");
                assert_eq!(phys.read32(base + REG_DEVICE_ID).data, 3);
                // The window is 64 KB: past it the slot is empty again.
                assert_ne!(phys.read32(base + 0x1_0000 + REG_MAGIC).data, MAGIC);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// The Indy keeps a virtio-serial device's state in a snapshot, and refuses one with a virtio-9p share.
    #[test]
    fn an_indy_snapshot_keeps_virtio_serial_and_refuses_9p() {
        crate::display_stop_tests::big_stack(|| {
            let dir = crate::display_stop_tests::scratch("indy-virtio");
            let saves = dir.join("saves");
            let cfg = |extra: String| {
                let mut cfg = crate::display_stop_tests::config(&dir, MachineProfile::IndyIp24, crate::config::GraphicsBoard::Newport);
                cfg.devices.push(card(&extra));
                cfg
            };
            let mut m = Box::new(Machine::new(cfg("type = \"virtio_serial\"\nat = \"exp1\"\n[[port]]".into())));
            crate::snapshot::save_in(&mut *m, &saves, "v").unwrap();
            assert!(saves.join("v/gio1.bin").exists(), "the device's state is in the snapshot");
            crate::snapshot::load_in(&mut *m, &saves, "v").unwrap();
            m.stop();
            drop(m);
            // virtio-9p is refused at config time on Windows, so it cannot be built there.
            #[cfg(unix)]
            {
                let share = format!("type = \"virtio_9p\"\nat = \"exp0\"\npath = {:?}\ntag = \"share\"", dir.to_str().unwrap());
                let nine_p = Box::new(Machine::new(cfg(share)));
                assert!(nine_p.snapshot_refusal().unwrap().contains("virtio-9p"));
            }
        });
    }

    /// IRIX looks for a second Newport only at 0x1F4F0000 (expansion slot 0)
    /// and, on an Indy, listens on that slot's one interrupt pin.
    #[test]
    fn the_second_head_is_where_irix_looks_for_it() {
        use crate::dev::ioc::{map_regs, IOC_BASE, IOC_INT3_MAP_STAT};
        let nv = crate::test_temp::TempPath::dir("head1-nv");
        let nv_dir = nv.to_path_buf();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut cfg = MachineConfig::default();
                crate::test_temp::nvram_in(&mut cfg, &nv_dir);
                cfg.devices.clear();
                cfg.machine.profile = MachineProfile::IndyIp24;
                cfg.set_device("newport", true);
                let m = Machine::new(cfg);
                let sys_id = m.board.phys.read32((IOC_BASE + crate::dev::ioc::IOC_SYS_ID) as u64).data;
                assert!((sys_id >> 1) & 0xF >= 4, "a baseboard revision that takes a second board");
                let phys = &m.board.phys;
                let pin = || phys.read32((IOC_BASE + IOC_INT3_MAP_STAT) as u64).data as u8;
                let head = m.get_rex3_head1().unwrap();
                let retrace = head.vblank_cb.lock().clone().unwrap();
                let fifo = head.fifo_full_cb.lock().clone().unwrap();
                assert_eq!(pin() & (map_regs::GIO_EXP0 | map_regs::GIO_EXP1), 0);
                retrace.set(true);
                fifo.set(true);
                retrace.set(false);
                assert_eq!(pin() & (map_regs::GIO_EXP0 | map_regs::GIO_EXP1), map_regs::GIO_EXP0,
                    "the FIFO still holds the slot's pin");
                fifo.set(false);
                assert_eq!(pin() & map_regs::GIO_EXP0, 0);

                // Reading STATUS at the address IRIX probes reaches this board:
                // it reports the retrace and lets go of the pin.
                use crate::dev::ng1::rex3::{REX3_STATUS, STATUS_VRINT};
                head.config.status.fetch_or(STATUS_VRINT, std::sync::atomic::Ordering::Relaxed);
                retrace.set(true);
                let status = phys.read32(0x1F4F_0000u64 + REX3_STATUS as u64).data;
                assert_ne!(status & STATUS_VRINT, 0);
                assert_eq!(pin() & map_regs::GIO_EXP0, 0);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn indy_slot_exp1() {
        probe(MachineProfile::IndyIp24, "gio1", 0x1F60_0000);
    }

    #[test]
    fn indy_slot_exp0() {
        probe(MachineProfile::IndyIp24, "gio0", 0x1F40_0000);
    }

    #[test]
    fn ip28_slot_exp1() {
        probe(MachineProfile::Indigo2Ip28, "gio1", 0x1F60_0000);
    }

    #[cfg(unix)]
    #[test]
    fn ip28_has_both_devices_in_their_slots() {
        let nv = crate::test_temp::TempPath::dir("virtio-both-nv");
        let nv_dir = nv.to_path_buf();
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let mut cfg = MachineConfig::default();
                crate::test_temp::nvram_in(&mut cfg, &nv_dir);
                cfg.devices.clear();
                cfg.headless = true;
                cfg.machine.profile = MachineProfile::Indigo2Ip28;
                cfg.machine.cpu = Some("r10000".into());
                cfg.banks = [512, 512, 0, 0, 0, 0, 0, 0];
                cfg.graphics.board = Some(crate::config::GraphicsBoard::SolidImpact);
                cfg.devices.push(card("type = \"virtio_serial\"\nat = \"gio1\"\n[[port]]"));
                cfg.devices.push(card(&format!("type = \"virtio_9p\"\nat = \"gio0\"\npath = {:?}", std::env::temp_dir().display().to_string())));
                assert_eq!(cfg.validate(), Ok(()));
                let m = Machine::new(cfg);
                let phys = &m.board.phys;
                assert_eq!(phys.read32(0x1F40_0000 + REG_DEVICE_ID).data, 9);
                assert_eq!(phys.read32(0x1F60_0000 + REG_DEVICE_ID).data, 3);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn one_slot_holds_one_device() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Indigo2Ip28;
        cfg.devices.push(card("type = \"virtio_serial\"\nat = \"gio1\""));
        cfg.devices.push(card("type = \"virtio_9p\"\nat = \"gio1\"\npath = \"/tmp\""));
        assert!(cfg.validate().unwrap_err().contains("gio1 is taken by the virtio_serial"));
    }

    /// On the Octane the virtio devices take the Bridge's extra slots, 7 then 5.
    #[test]
    fn octane_virtio_takes_the_bridge_extra_rows() {
        let places = |text: &str| {
            let mut cfg = MachineConfig::from_toml_str(&format!("devices = []\n[machine]\nprofile = \"octane_ip30\"\n{text}")).unwrap();
            cfg.resolve_slots()?;
            let p = crate::topology::DevicePlacement::new(&cfg)?;
            Ok::<_, String>((0..cfg.devices.len()).filter_map(|i| Some(p.at(i)?.to_string())).collect::<Vec<_>>())
        };
        let both = "[virtio_serial]\nenabled = true\n[virtio_9p]\nenabled = true\npath = \"/tmp\"\n";
        assert_eq!(places(both).unwrap(), ["xio15/pci0/7", "xio15/pci0/5"]);
    }
}

#[cfg(test)]
mod headless_tests {
    use super::*;
    use crate::display_stop_tests::{big_stack, scratch};

    fn build(dir: &std::path::Path, profile: MachineProfile, headless: bool) -> Box<Machine> {
        let nv = dir.to_path_buf();
        big_stack(move || {
            let mut cfg = crate::bench_runner::bench_config([16, 0, 0, 0], &parts::R4400_150);
            crate::test_temp::nvram_in(&mut cfg, &nv);
            cfg.machine.profile = profile;
            cfg.headless = headless;
            cfg.graphics.board = Some(GraphicsBoard::Newport);
            Box::new(Machine::new(cfg))
        })
    }

    #[test]
    fn headless_indy_keeps_its_newport_and_stores_console_d() {
        let dir = scratch("headless-indy");
        let m = build(&dir, MachineProfile::IndyIp24, true);
        assert!(m.board.rex3.is_some(), "headless does not drop the Newport");
        assert!(m.hpc3.rtc().console_is_serial(crate::dev::sgi_nvram::REV_IP22));
        let dir = scratch("windowed-indy");
        assert!(!build(&dir, MachineProfile::IndyIp24, false).hpc3.rtc().console_is_serial(crate::dev::sgi_nvram::REV_IP22));
    }

    #[test]
    fn headless_indigo2_keeps_its_newport_and_stores_console_d() {
        let dir = scratch("headless-indigo2");
        let m = build(&dir, MachineProfile::Indigo2Ip22, true);
        assert!(m.board.rex3.is_some());
        assert!(m.hpc3.eeprom().lock().console_is_serial(crate::dev::sgi_nvram::REV_IP22));
    }
}
