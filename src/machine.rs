use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::AtomicU64;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;

use crate::config::{GraphicsBoard, MachineConfig, MachineProfile, NetworkConfig, ScsiDeviceConfig};
use crate::machine_state::GuestState;
use crate::traits::{BusDevice, Device, Resettable, Saveable, MachineEvent};
use crate::locks::LockMonitor;
use crate::dev::eeprom_93c56::Eeprom93c56;
use crate::physical::Physical;
use crate::ip22_bus::{Ip22Bus, Ip22Devices};

// Helper for passing *mut Physical into a Send+Sync closure (MEMCFG callback).
// Safety: Physical is Send+Sync, and the Arc keeps it alive for the callback's lifetime.
use crate::physical::RamBank;
use crate::dev::prom::Prom;
use crate::dev::mc::MemoryController;

/// Count on the IP28's 195 MHz R10000: half the pipeline clock, and what
/// IRIX assumes there (see where the MC is created).
const IP28_COUNT_HZ: u64 = 97_500_000;
use crate::cpu::mips_tlb::MipsTlb;
use crate::cpu::mips_exec::{MipsExecutor, MipsCpu, MipsCpuConfig, MipsCpuDebugAdapter};
use crate::gdb_stub::CpuDebug;
use crate::cpu::mips_cache_v2::{MipsCache, R4400Cache, R5000Cache};
use crate::cpu::mips_cache_shadow::{R10000ShadowCache, R4400ShadowCache, R5000ShadowCache};
use crate::dev::hpc3::Hpc3;
use crate::dev::ioc::{Ioc, GioSlot, GIO_SLOT_MAP, profile_idx};
use crate::monitor::Monitor;
use crate::dev::ng1::rex3::Rex3;
use crate::snapshot::{Snapshot, Manifest, SCHEMA_VERSION, ChunksManifest, DiskRef, enabled_features};
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

fn emulator_profile() -> Option<MachineProfile> {
    let raw = EMULATOR_PROFILE.load(std::sync::atomic::Ordering::Relaxed);
    [MachineProfile::IndyIp24, MachineProfile::Indigo2Ip22, MachineProfile::Indigo2Ip28, MachineProfile::O2Ip32, MachineProfile::OctaneIp30, MachineProfile::FuelIp34]
        .into_iter().find(|p| *p as u8 == raw)
}

/// The words an `I` in the expansion may stand for, per machine. Each list
/// names the machine as the user configured it, so the title never calls an
/// Indigo2 an Indy or an O2 anything but IP32. Without a profile only the
/// model-neutral words remain.
fn emulator_name_thirds(profile: Option<MachineProfile>) -> &'static [&'static str] {
    match profile {
        Some(MachineProfile::IndyIp24) => &["IRIX", "Iris", "Indy", "IP24"],
        Some(MachineProfile::Indigo2Ip22) => &["IRIX", "Iris", "Indigo2", "IP22"],
        Some(MachineProfile::Indigo2Ip28) => &["IRIX", "Iris", "Indigo2", "IP28"],
        Some(MachineProfile::O2Ip32) => &["IRIX", "Iris", "IP32"],
        Some(MachineProfile::OctaneIp30) => &["IRIX", "Iris", "IP30"],
        Some(MachineProfile::FuelIp34) => &["IRIX", "Iris", "IP34"],
        None => &["IRIX", "Iris"],
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
    /// The physical bus and the Indy's or Indigo2's devices on it.
    board: Arc<Ip22Bus>,
    mc: MemoryController,
    hpc3: Hpc3,
    monitor: Arc<Monitor>,
    /// TCP port the monitor serves on 127.0.0.1 (`cfg.monitor_port`).
    monitor_port: u16,
    /// Sender for async machine events (HardReset, PowerOff) from devices.
    pub event_tx: mpsc::SyncSender<MachineEvent>,
    event_rx: Option<mpsc::Receiver<MachineEvent>>,
    timer_manager: Arc<TimerManager>,
    /// When `cfg.ci` is set, the channel-A backend is replaced by this
    /// in-process one so the CI control socket can drive the console.
    ci_serial: Option<Arc<crate::dev::serial_host::CiSerialBackend>>,
    /// The shared `--serial-log` sink, kept so `stop` can flush a final
    /// partial line (wall-timestamp mode holds one until its newline).
    serial_log: Option<Arc<crate::dev::serial_log::SerialLog>>,
    /// Most recent snapshot restored via `ci_restore`. `rollback` reuses this
    /// name as the fallback path if the in-memory checkpoint is absent.
    last_restore: Option<String>,
    /// In-memory copy of the just-loaded state, taken at the end of every
    /// successful `ci_restore`. Lets `ci_rollback` skip disk IO and TOML
    /// re-parsing — paste back the cached `toml::Value`s and `memcpy` the
    /// bank/framebuffer buffers. Cleared on any explicit `load_snapshot`
    /// outside the CI path.
    last_restore_checkpoint: Option<RollbackCheckpoint>,
    /// Path of the configured scratch SCSI volume, if any. The CI socket reads
    /// and writes this file directly (with the machine briefly stopped) to
    /// inject/exfiltrate files without going through the network. None when no
    /// SCSI device has `scratch = true` set in the config.
    scratch_path: Option<std::path::PathBuf>,
    /// Disk provenance captured at construction (configured path + host file
    /// size per SCSI id). Written into snapshot manifests and validated on
    /// restore so captured state never lands on a different base disk.
    disks: Vec<DiskRef>,
    /// Configured nvram file path, recorded in snapshot manifests.
    nvram_path: String,
    /// Host-forced Newport resolution from `[graphics] resolution` at construction.
    display_resolution: crate::dev::ng1::vc2_timings::NewportResolution,
    /// Whether Newport is active (not headless and Newport is selected).
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

/// In-memory snapshot of the just-restored guest state. Populated at the end
/// of `ci_restore`; consumed by `ci_rollback`. Trades ~270 MB of RSS for
/// disk-IO-free rollback.
struct RollbackCheckpoint {
    /// Snapshot directory (saves/<name>/) — re-used by rollback to reflink
    /// the COW overlays back into place.
    overlay_dir: std::path::PathBuf,
    /// Per-SCSI-id dirty sector lists from cow.toml at the time of restore.
    overlay_sets: Vec<(usize, Vec<u64>)>,
    /// Same, for the second (fullhouse) SCSI controller — kept in a
    /// separate list/subdirectory (see `capture_rollback_checkpoint`) since
    /// controller 0 and controller 1 can each have a device at the same id.
    overlay_sets1: Vec<(usize, Vec<u64>)>,

    /// Native-endian RAM bank words. `bank_words[i].len() ==
    /// banks[i].size_bytes / 4` for present banks; populated for all four.
    bank_words: [Vec<u32>; 4],

    /// Framebuffer contents (RGB, aux). `None` when running headless.
    framebuffers: Option<(Vec<u32>, Vec<u32>)>,

    /// Parsed device save_state TOMLs. Holding `toml::Value` directly skips
    /// the ~80 ms cpu.toml string-parse cost on every rollback.
    cpu: toml::Value,
    mc: toml::Value,
    ioc: toml::Value,
    scc: toml::Value,
    pit: toml::Value,
    ps2: toml::Value,
    rtc: toml::Value,
    eeprom: toml::Value,
    scsi: toml::Value,
    /// Second SCSI controller's state — fullhouse only. `None` on guinness.
    scsi1: Option<toml::Value>,
    seeq: toml::Value,
    hpc3: toml::Value,
    rex3: Option<toml::Value>,
    rex3_head1: Option<toml::Value>,
}

/// Parse a `cow.toml`/`cow1.toml`-shaped table (`{"scsi<id>": [dirty
/// sectors...]}`) into `(id, dirty_sectors)` pairs. Shared by rollback
/// capture and snapshot load so the two paths can't drift.
/// Inverse of `parse_cow_table`: build a `{"scsi<id>": [dirty sectors...]}`
/// table from `export_overlays`'s output.
pub(crate) fn build_cow_table(overlays: Vec<(usize, Vec<u64>)>) -> toml::Value {
    let mut tbl = toml::map::Map::new();
    for (id, dirty) in overlays {
        let arr: Vec<toml::Value> = dirty.into_iter().map(|v| toml::Value::Integer(v as i64)).collect();
        tbl.insert(format!("scsi{}", id), toml::Value::Array(arr));
    }
    toml::Value::Table(tbl)
}

pub(crate) fn parse_cow_table(cow_toml: &toml::Value) -> Vec<(usize, Vec<u64>)> {
    let mut sets = Vec::new();
    if let Some(tbl) = cow_toml.as_table() {
        for (k, v) in tbl {
            let Some(id_str) = k.strip_prefix("scsi") else { continue };
            let Ok(id) = id_str.parse::<usize>() else { continue };
            let Some(arr) = v.as_array() else { continue };
            let dirty: Vec<u64> = arr.iter()
                .filter_map(|x| x.as_integer().map(|i| i as u64))
                .collect();
            sets.push((id, dirty));
        }
    }
    sets
}

/// Full-machine in-memory checkpoint captured from *live* running state
/// (unlike `RollbackCheckpoint`, which is specifically "the state right
/// after the most recent `ci_restore` from a named snapshot on disk" and
/// re-reads that snapshot's `cow.toml` to reflink SCSI overlays back into
/// place on rollback). Used by the interactive monitor's `j2 trace <n>` /
/// `j2 replay` (`SystemController::execute_command`) to get a real,
/// byte-exact rewind point — including device state, not just CPU
/// registers — for comparing an interpreter-only reference run against a
/// real-JIT-dispatch run of the same instruction window.
///
/// Deliberately does **not** capture SCSI controller or COW-overlay dirty-
/// sector state — there is no "just loaded from snapshot X" name to
/// reference here (this captures live, mid-session state, not a disk
/// snapshot restore), and building a live-capture equivalent of
/// `RollbackCheckpoint`'s overlay handling is real, correctness-sensitive
/// work (get it wrong and it's disk corruption, not just a debugging
/// inconvenience) that early-boot PROM debugging — the actual motivating
/// use case — doesn't need at all (PROM hasn't touched SCSI yet). Restoring
/// a `LiveCheckpoint` taken after the guest has performed real disk I/O
/// will silently leave SCSI/disk state at whatever it drifted to during the
/// replayed window, not rewound — safe for CPU/JIT correctness comparisons
/// (nothing here reads disk contents), unsafe to treat as a general
/// save-state substitute once boot reaches real disk access.
pub(crate) struct LiveCheckpoint {
    bank_words: [Vec<u32>; 4],
    framebuffers: Option<(Vec<u32>, Vec<u32>)>,
    cpu: toml::Value,
    mc: toml::Value,
    ioc: toml::Value,
    scc: toml::Value,
    pit: toml::Value,
    ps2: toml::Value,
    rtc: toml::Value,
    eeprom: toml::Value,
    seeq: toml::Value,
    hpc3: toml::Value,
    rex3: Option<toml::Value>,
    rex3_head1: Option<toml::Value>,
}

/// The test device and the ultra64 dev board both decode GIO expansion slot 0,
/// so only one of them can exist. Panics rather than exiting: `Machine::new`
/// already panics on bad input, and a host that embeds IRIS catches that
/// (`iris-gui`'s worker wraps construction in `catch_unwind`) — killing the
/// application over a configuration mistake is not a choice a library gets to
/// make for its caller.
fn check_testdev_slot_free(ultra64_present: bool) {
    if ultra64_present {
        panic!("--test-device and the ultra64 dev board both claim GIO slot 0");
    }
}

impl Machine {
    pub fn new(cfg: MachineConfig) -> Self {
        // Host calls (iris-hostcall): the self-test service is always there
        // when the feature is built in, so a guest can prove the trap works.
        #[cfg(feature = "hostcall")]
        iris_hostcall::register(iris_hostcall::SELFTEST, Box::new(iris_hostcall::SelfTest));
        #[cfg(feature = "hostgl")]
        iris_hostgl::register();
        Self::new_with_testdev(cfg, None)
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
        // Capture config flags that are needed after the local `cfg` binding
        // is shadowed later in this function.
        let ci_enabled = cfg.ci;
        let perf = cfg.perf.clone();
        #[cfg(feature = "jitv2")]
        let jitv2_threads = cfg.jitv2.threads;
        let display_resolution = cfg.graphics.resolution;
        let newport_active = !cfg.headless && cfg.graphics_board() == GraphicsBoard::Newport;
        let monitor_port = cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT);
        let clock_fixed_mhz = cfg.clock.fixed_mhz;
        let cfg_cpu_model = cfg.machine.cpu;

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
        let model_has_l2 = match cfg_cpu_model {
            crate::config::CpuModel::R4400 => <R4400Cache as MipsCache>::L2_SIZE > 0,
            crate::config::CpuModel::R5000 => <R5000Cache as MipsCache>::L2_SIZE > 0,
            crate::config::CpuModel::R10000 => <R10000ShadowCache as MipsCache>::L2_SIZE > 0,
            crate::config::CpuModel::R12000 | crate::config::CpuModel::R14000 => unreachable!("an Indy or Indigo2 never has this part; MachineConfig::validate"),
        };
        if !model_has_l2 {
            eeprom_mc.lock().set_cachsz(0);
        }

        // 1. Create all devices first
        // Memory Controller
        let mc_banks: [u32; 4] = cfg.banks[..4].try_into().expect("the MC has four banks");
        let mc = MemoryController::new_for_profile(
            eeprom_mc.clone(), guinness, mc_banks, cfg.machine.profile.ip28());
        // IP28: IRIX takes the CPU speed from the PROM's `cpufreq` (194, i.e.
        // a 195 MHz R10000) instead of measuring it, and times with it: UST
        // assumes Count at half that, and the RPSS cycle counter the MC clock
        // at the CPU clock (Config.EC is 0, a 1:1 system-clock ratio) over the
        // divider it programs. The generic 33 MHz Count and 50 MHz MC made
        // UST run at a third of real time (Quake in slow motion) and the
        // cycle counter at a quarter.
        let ip28_count_hz: Option<u64> = if cfg.machine.profile.ip28() {
            Some(clock_fixed_mhz.map_or(IP28_COUNT_HZ, |mhz| (mhz * 1_000_000.0) as u64))
        } else {
            None
        };
        if let Some(count_hz) = ip28_count_hz {
            // The R10000's Count ticks at half the pipeline clock.
            mc.set_clock_hz(2 * count_hz);
        }

        // RAM banks sized per config. addr_mask is initialized to mem_size-1;
        // remap_banks() updates it via set_addr_mask() when MEMCFG0/1 are written during POST.
        let banks = [
            RamBank::new(cfg.banks[0].max(1) as usize),
            RamBank::new(cfg.banks[1].max(1) as usize),
            RamBank::new(cfg.banks[2].max(1) as usize),
            RamBank::new(cfg.banks[3].max(1) as usize),
        ];

        // PROM (1MB at 0x1FC00000). Each profile has its own embedded fallback.
        // Indigo2 also tries its profile's default filename after cfg.prom.
        let prom = if cfg.machine.profile.ip28() {
            Prom::from_file_or_embedded_ip28(&cfg.prom)
        } else if guinness {
            Prom::from_file_or_embedded(&cfg.prom)
        } else {
            Prom::from_file_or_embedded_ip22(&cfg.prom)
        };
        let prom_port = prom.get_port();

        // Shared atomics — created first so all devices and the display thread use the same Arc.
        // (CPU cycle counter is no longer among these: it lives inline on
        // MipsCore.hot.cycles now, wired into devices via set_cpu_cycles
        // after the executor/CPU exist — see Hot::cycles's doc comment.)
        let heartbeat     = Arc::new(AtomicU64::new(0)); // activity bits: see Rex3::HB_*
        let fasttick_count = Arc::new(AtomicU64::new(0)); // CP0 Compare match counter
        let decoded_count = Arc::new(AtomicU64::new(0)); // pre-decoded instruction count
        let l1i_hit_count        = Arc::new(AtomicU64::new(0)); // L1-I hit counter
        let l1i_fetch_count      = Arc::new(AtomicU64::new(0)); // L1-I fetch counter
        let uncached_fetch_count = Arc::new(AtomicU64::new(0)); // uncached instruction fetches

        // HPC3 (512KB at 0x1FB80000). CI mode skips the SCC TCP backend
        // bindings so multiple `--ci` instances can coexist.
        let ioc = Ioc::new_for_profile(guinness, ci_enabled, cfg.machine.profile.ip28(), cfg.serial_ports());
        ioc.ps2().set_layout(cfg.keyboard);

        // One sink for both serial paths, so the CI and TCP logs are the same
        // bytes (timestamps included). Opened once, before any peripheral
        // `start()` clones the current backend Arc into the RX/TX threads.
        let serial_log = crate::dev::serial_host::open_log(&cfg, "Indy", crate::dev::serial_host::LogMode::Append);

        // CI mode replaces the default TCP backend on channel B (tty1, the
        // SGI serial console) with an in-process backend the control socket
        // drives directly. Channel A (tty2) keeps its default TCP backend.
        let ci_serial = if ci_enabled {
            let b = Arc::new(crate::dev::serial_host::CiSerialBackend::new());
            if let Some(log) = &serial_log {
                b.set_log(log.clone());
            }
            ioc.scc().set_backend_b(b.clone());
            Some(b)
        } else {
            // Non-CI mode: channel B already has its TCP listener on
            // 127.0.0.1:8881 (or `serial_port_b`). Wrap it in a TeeBackend so
            // guest-emitted bytes also reach the file.
            if let Some(log) = &serial_log {
                let inner = ioc.scc().backend_b();
                ioc.scc().set_backend_b(Arc::new(crate::dev::serial_host::TeeBackend::new(inner, log.clone())));
            }
            None
        };
        let timer_manager = Arc::new(TimerManager::new());
        ioc.set_timer_manager(timer_manager.clone());
        ioc.set_heartbeat(heartbeat.clone());
        ioc.set_clock_ticks(fasttick_count.clone());
        // Attach SCSI devices from config: each `[scsi.KEY]` table puts a
        // target on a controller (`ScsiDeviceConfig::placement`). The disks
        // recorded here, and the scratch volume's path for the CI socket,
        // come back with it. Disk provenance for snapshot manifests is
        // captured while the MachineConfig `cfg` is still in scope (it is
        // shadowed by the CPU config below): the slot (controller * 16 + id,
        // see `config::scsi_slot`), the configured path and the host file
        // size.
        // The buses are filled before the controllers exist and handed to them.
        let scsi_buses = Hpc3::new_scsi_buses(guinness);
        let attached = match crate::scsi::attach_from_config(&cfg, None, &|c| Hpc3::route_scsi(&scsi_buses, c)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("iris: fatal: {e}");
                std::process::exit(1);
            }
        };
        let hpc3 = Hpc3::with_net(eeprom_hpc3.clone(), ioc.clone(), guinness, heartbeat.clone(), cfg.network(), cfg.no_audio, cfg.audio.clone(), cfg.nvram.clone(), cfg.rtc_offset, cfg.scsi_deferred_int, scsi_buses);
        hpc3.set_timer_manager(timer_manager.clone());

        // Backdoor-inject the configured (or default) Ethernet station
        // address before the CPU ever runs, since real hardware always has
        // one burned in and there is no user-facing way to set one. Both
        // NVRAM (Indy) and the motherboard EEPROM (Indigo2) are persisted to
        // disk (see cfg.nvram / cfg.nveeprom), so both only patch while the
        // `eaddr` slot is still blank — never clobbers a MAC the guest
        // already `setenv`'d (Indy) or that a prior run already saved
        // (Indigo2).
        if guinness {
            hpc3.rtc().backdoor_set_mac_if_blank(cfg.network().mac);
        } else {
            let mut eeprom = eeprom_hpc3.lock();
            let initialized = cfg.machine.profile.ip28() && eeprom.initialize_ip28_if_erased();
            eeprom.backdoor_set_mac_if_blank(cfg.network().mac);
            if initialized { eeprom.update_ip28_checksum(); }
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
        let disk_provenance: Vec<DiskRef> = attached.tables.iter()
            .map(|t| DiskRef { id: t.slot, path: t.path.clone(), size_bytes: t.size_bytes })
            .collect();
        let nvram_provenance = cfg.nvram.clone();

        // REX3 Graphics — Newport only; skipped in headless mode or when non-Newport board selected.
        let rex3: Option<Arc<Rex3>> = if cfg.headless || cfg.graphics_board() != crate::config::GraphicsBoard::Newport {
            None
        } else {
            let r = Arc::new(Rex3::new(heartbeat.clone(), fasttick_count.clone(), decoded_count.clone(), Arc::clone(&l1i_hit_count), Arc::clone(&l1i_fetch_count), Arc::clone(&uncached_fetch_count)));
            let wiring = GIO_SLOT_MAP[profile_idx(guinness)][GioSlot::Gfx as usize];
            r.set_vblank_irq(ioc.irq_line(wiring.retrace));
            r.set_fifo_full_irq(ioc.irq_line(wiring.fifo_full));
            Some(r)
        };

        let rex3_head1: Option<Arc<Rex3>> = if cfg.headless || cfg.graphics.heads < 2 {
            None
        } else {
            let r = Arc::new(Rex3::new(heartbeat.clone(), fasttick_count.clone(), decoded_count.clone(), Arc::clone(&l1i_hit_count), Arc::clone(&l1i_fetch_count), Arc::clone(&uncached_fetch_count)));
            let wiring = GIO_SLOT_MAP[profile_idx(guinness)][GioSlot::Exp0 as usize];
            r.set_vblank_irq(ioc.irq_line(wiring.retrace));
            r.set_fifo_full_irq(ioc.irq_line(wiring.fifo_full));
            Some(r)
        };

        // GR2 (XZ / Extreme) in the GIO gfx slot. Created even when headless:
        // selecting the board is explicit, and the PROM needs it to probe.
        let gr2: Option<Arc<crate::dev::gr2::Gr2>> = match cfg.graphics_board() {
            crate::config::GraphicsBoard::Xz | crate::config::GraphicsBoard::Extreme => {
                use crate::dev::gr2::{Gr2, Gr2Stats, Gr2Variant};
                let variant = if cfg.graphics_board() == crate::config::GraphicsBoard::Extreme { Gr2Variant::Extreme } else { Gr2Variant::Xz };
                let g = Gr2::new(variant, Gr2Stats { heartbeat: heartbeat.clone(), fasttick: fasttick_count.clone() });
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
                Some(crate::dev::mgras::Mgras::new(cfg.graphics_board(), {
                    use crate::dev::ioc::IocInterrupt::{GioSgFifo, GioSgGraphics, GioSgRetrace};
                    [ioc.irq_line(GioSgFifo), ioc.irq_line(GioSgGraphics), ioc.irq_line(GioSgRetrace)]
                }, heartbeat.clone(), fasttick_count.clone()))
            }
            _ => None,
        };

        // N64 development board (Ultra64) — GIO slot 0 at 0x1F400000
        let ultra64: Option<Arc<crate::dev::ultra64::Ultra64>> = if cfg.ultra64.enabled {
            match crate::dev::ultra64::Ultra64::new(ioc.irq_line(crate::dev::ioc::IocInterrupt::GioExp0)) {
                Ok(dev) => Some(Arc::new(dev)),
                Err(e)  => {
                    eprintln!("ultra64: failed to initialize ({e}); running without N64 dev board");
                    None
                }
            }
        } else {
            None
        };

        // VINO (Video-In, No Out) — GIO64 at 0x1F080000
        let vino = crate::dev::vino::Vino::new();
        vino.set_irq(ioc.irq_line(crate::dev::ioc::IocInterrupt::VideoVsync));

        // 2. Create Physical Bus with devices
        // `cfg` is shadowed by MipsCpuConfig further down; grab this first.
        let cheritest_hook = cfg.cheritest_dump_hook;

        // Bare-metal test device (--test-device): default off, and refused
        // alongside the ultra64 dev board, which claims the same GIO slot.
        let testdev = if let Some(dev) = testdev_override {
            check_testdev_slot_free(
                ultra64.is_some(),
            );
            Some(dev)
        } else if cfg.test_device {
            check_testdev_slot_free(
                ultra64.is_some(),
            );
            let path = cfg.test_device_dump.clone()
                .unwrap_or_else(|| crate::dev::testdev::DEFAULT_DUMP_PATH.to_string());
            eprintln!("iris: test device enabled at {:#010x}, dumps to {}",
                      crate::dev::testdev::TEST_DEV_BASE, path);
            Some(Arc::new(crate::dev::testdev::TestDevice::new(path)))
        } else {
            None
        };

        let phys = Physical::new(banks.into());
        let board = Arc::new(Ip22Bus::new(phys.clone(), Ip22Devices {
            rex3: rex3.clone(),
            rex3_head1: rex3_head1.clone(),
            gr2: gr2.clone(),
            mgras: mgras.clone(),
            ultra64,
            testdev,
            vino,
            mc: mc.clone(),
            hpc3: hpc3.clone(),
            prom: prom_port,
        }, cfg.machine.profile.ip28()));

        // Connect Physical to MC (for VDMA)
        mc.set_phys(phys.clone());
        mc.set_dma_irq(ioc.irq_line(crate::dev::ioc::IocInterrupt::McDma));

        // Wire MEMCFG callback: when MC writes MEMCFG0/1, remap the banks. Runs
        // on the CPU thread (the thread that writes MEMCFG). Weak: the MC is
        // one of the board's devices.
        {
            let board = Arc::downgrade(&board);
            mc.set_memcfg_callback(Box::new(move |addrs| {
                if let Some(board) = board.upgrade() {
                    board.remap_banks(addrs);
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
        //    arm below monomorphises its own CPU — no per-model branch on the hot path.
        let sysad: Arc<dyn BusDevice> = phys.clone();
        macro_rules! build_cpu { ($cache:ty) => {{
        let cfg = MipsCpuConfig::for_model::<$cache>();
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let mut executor: MipsExecutor<MipsTlb, $cache> = MipsExecutor::new(sysad.clone(), tlb, &cfg);
        // The MC decodes 32 physical address bits; see `MipsExecutor::phys_mask`.
        executor.phys_mask = u32::MAX as u64;

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

        // CP0 Count runs at a fixed frequency: DEFAULT_COUNT_HZ unless the
        // user overrode it via `[clock] fixed_mhz` or the CLI. Must happen
        // before the core starts executing.
        if let Some(hz) = ip28_count_hz {
            executor.core.set_count_hz(hz);
        } else if let Some(mhz) = clock_fixed_mhz {
            executor.core.set_count_hz((mhz * 1_000_000.0) as u64);
        }

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
        let cpu: Arc<dyn crate::cpu::mips_exec::CpuDevice> = match cfg_cpu_model {
            // R4400/R5000: the functional caches, or the shadow ones that
            // keep only what CACHE instructions observe — see CacheModel.
            crate::config::CpuModel::R4400 if shadow => build_cpu!(R4400ShadowCache),
            crate::config::CpuModel::R5000 if shadow => build_cpu!(R5000ShadowCache),
            crate::config::CpuModel::R4400 => build_cpu!(R4400Cache),
            crate::config::CpuModel::R5000 => build_cpu!(R5000Cache),
            // IP28 uses the shadow cache: out of the data path entirely, with
            // tag and data arrays that exist only to answer CACHE ops and the
            // PROM's diagnostics. See mips_cache_shadow.rs.
            crate::config::CpuModel::R10000 => build_cpu!(R10000ShadowCache),
            crate::config::CpuModel::R12000 | crate::config::CpuModel::R14000 => unreachable!("an Indy or Indigo2 never has this part; MachineConfig::validate"),
        };

        // Share count_hz_atomic from MipsCore with Rex3 so the refresh thread can display it.
        #[cfg(feature = "developer")]
        if let Some(rex3) = &board.rex3 { rex3.set_count_hz_atomic(cpu.count_hz_atomic()); }

        // Connect CPU to MC and IOC for signaling
        let cpu_device: Arc<dyn Device> = cpu.clone().as_device();
        mc.set_cpu(Arc::downgrade(&cpu_device));
        ioc.set_interrupts(cpu.interrupts_ptr());

        // ppmem: hand the CPU's inline bitmap field to the memory space, so a
        // MEMCFG remap publishes straight into the object the CPU executes out
        // of. Safe per `set_bitmap_sink`: the executor is at its final address
        // by now and the only writer is this same CPU thread's MEMCFG path.
        {
            use crate::ppmem::MappedMemory;
            if let Some(sp) = phys.ppmem_space() {
                unsafe { sp.set_bitmap_sink(cpu.ppmem_bitmap_ptr()) };
            }
            // Point Physical's cached bitmap at the CPU's field too, or its
            // direct-access fast path would keep reading the sink the space
            // just abandoned.
            phys.resync_ppmem_bitmap();
            // The cache reads RAM straight out of the same window: the full
            // cache under tcache, the shadow cache always. A cache that does
            // not (the full cache without tcache) has no bitmap field and
            // returns null. Must come after set_bitmap_sink, so it captures
            // the CPU's inline field rather than the sink the space just
            // abandoned.
            if let Some(sp) = phys.ppmem_space() {
                unsafe { cpu.set_tcache_window(sp.window_base()) };
                // The cache keeps the bitmap inline; ppmem publishes into it
                // directly on every remap, so the hot path needs no indirection.
                let bitmap = cpu.tcache_bitmap_ptr();
                if !bitmap.is_null() {
                    unsafe { sp.set_bitmap_sink2(bitmap) };
                }
                // Window writes bypass BusDevice, so the cache needs the gen
                // window to bump page counters itself — otherwise jitv2 keeps
                // running compiled code for pages the guest has modified.
                #[cfg(feature = "jitv2")]
                unsafe { cpu.set_tcache_gen_window(sp.gen_window_base()) };
            }
        }

        // Wire up the CPU cycle counter for every device that reads it —
        // Rex3::new/Wd33c93a::new run before the CPU exists, so this can't
        // be a constructor parameter (see Hot::cycles's doc comment). Must
        // happen before hpc3.scsi().start() (called later, from
        // Machine::start) actually spawns the worker thread that reads it.
        if let Some(rex3) = &board.rex3 { rex3.set_cpu_cycles(cpu.cycles_ptr()); }
        if let Some(mgras) = &board.mgras { mgras.set_cpu_cycles(cpu.cycles_ptr()); }
        if let Some(rex3) = &board.rex3_head1 { rex3.set_cpu_cycles(cpu.cycles_ptr()); }
        if let Some(gr2) = &board.gr2 { gr2.set_cpu_cycles(cpu.cycles_ptr()); }
        if let Some(td) = &board.testdev { td.attach_core(cpu.core_ptr()); }
        if cheritest_hook {
            if board.testdev.is_none() {
                eprintln!("iris: --cheritest-dump-hook needs --test-device; ignoring");
            } else {
                cpu.set_cheritest_dump_hook(true);
            }
        }
        hpc3.scsi().set_cpu_cycles(cpu.cycles_ptr());

        // Inject the CPU/self handles jitv2's compile-thread worker needs to
        // pause the CPU around its own memory-growth flush (see
        // `Jitv2::codegen`/`CompileQueue`'s doc comments — the CPU doesn't
        // exist yet when `MipsExecutor::new` constructs its own `Jitv2`
        // default, so this can only happen here, after `Arc::new(MipsCpu::new(..))`,
        // same reasoning as `mc.set_cpu` right above).
        #[cfg(feature = "jitv2")]
        {
            let jit = cpu.jitv2();
            // Fixed at startup, before the queue ever starts — see
            // CompileQueue::set_thread_count's own doc comment for why this
            // can't change at runtime.
            jit.lock().compile_queue.set_thread_count(jitv2_threads.max(1));
            jit.lock().compile_queue.set_cpu(Arc::downgrade(&cpu_device));
            jit.lock().compile_queue.set_owner(Arc::downgrade(&jit));
            // Threaded compile is the default (`MipsExecutor::jitv2_inline_compile`
            // starts `false`) — the queue must actually be running or every
            // compile request silently vanishes (send()s into a consumer
            // nothing is popping). `j2 inline on` reverses this later by
            // stopping the queue and reclaiming its Codegen for the inline
            // path (see that command's own handler).
            //
            // EXCEPT under jitv2_lockstep: the dispatch gate forces inline
            // compile there (see exec_decoded's `inline_compile`), and inline
            // compile takes the Codegen from `Jitv2::codegen` on every compile.
            // If the queue owned it instead, inline's `.take()` would get `None`
            // and NOTHING would ever compile — the whole run silently degrades
            // to pure interpreter with lockstep never firing (observed: a
            // lockstep boot showing `codegen owned by async compile thread`,
            // `compiles: 0`, running at interpreter speed). So leave the queue
            // stopped under lockstep and keep the Codegen in `Jitv2::codegen`
            // for inline to use.
            #[cfg(not(feature = "jitv2_lockstep"))]
            {
                let mut guard = jit.lock();
                // start() now builds its own Codegen internally (no more
                // shared-arena-across-modes handoff — see CompileQueue::start's
                // own doc comment) — the idle Jitv2::codegen slot built by
                // Jitv2::new() stays untouched here, reserved for inline
                // mode if `j2 inline on` ever switches to it.
                let stats = guard.stats.clone();
                let bus: Arc<dyn BusDevice> = phys.clone();
                guard.compile_queue.start(bus, stats);
            }
        }

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
        if let Some(rex3) = &board.rex3_head1 { monitor.register_device(rex3.clone()); }
        if let Some(td) = &board.testdev { monitor.register_device(td.clone()); }
        if let Some(gr2) = &board.gr2 { monitor.register_device(gr2.clone()); }
        if let Some(mgras) = &board.mgras { monitor.register_device(mgras.clone()); }
        if let Some(u64) = &board.ultra64 { monitor.register_device(u64.clone()); }
        monitor.register_device(Arc::new(board.vino.clone()));
        #[cfg(feature = "perfetto")]
        {
            crate::perfetto::set_guest_cycles(Some(cpu.cycles_ptr()));
            monitor.register_device(Arc::new(crate::perfetto::PerfettoMonitor));
        }
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
            use crate::locks::register_lock_fn;
            let ep = eeprom_mc.clone();
            register_lock_fn("mc::eeprom", move || ep.is_locked());
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

        let (event_tx, event_rx) = mpsc::sync_channel::<MachineEvent>(4);

        // Give MC and IOC async event senders so they can request hard-reset / power-off.
        mc.set_event_sender(event_tx.clone());
        ioc.set_event_sender(event_tx.clone());

        crate::thread_affinity::init(perf);

        Self {
            cpu,
            board,
            mc,
            hpc3,
            monitor,
            monitor_port,
            event_tx,
            event_rx: Some(event_rx),
            timer_manager,
            ci_serial,
            serial_log,
            last_restore: None,
            last_restore_checkpoint: None,
            scratch_path,
            disks: disk_provenance,
            nvram_path: nvram_provenance,
            display_resolution,
            newport_active,
            fullhouse: !guinness,
            controller_slot: None,
            controller_registered: false,
            lifecycle: GuestState::Idle,
        }
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
        if let Some(rex3) = &self.board.rex3 {
            rex3.apply_display_resolution(mode);
        }
        if let Some(rex3) = &self.board.rex3_head1 {
            rex3.apply_display_resolution(mode);
        }
    }

    /// Path of the configured scratch SCSI volume, if any. Used by the CI
    /// socket scratch-{write,read,clear,info} commands to act on the file
    /// directly while the machine is briefly stopped.
    pub fn scratch_path(&self) -> Option<&std::path::Path> {
        self.scratch_path.as_deref()
    }

    pub fn start(&mut self) {
        // Publish "starting" before the monitor binds, so a caller that polls
        // readiness can tell "not started" (Idle) from "startup under way".
        // Halted -> Starting covers the GUI's Stop then Start; a stored
        // Running state is impossible (status derives it from the CPU).
        if let Ok(starting) = self.lifecycle.transition(GuestState::Starting) {
            self.lifecycle = starting;
        }
        // Start peripherals
        self.mc.start();
        self.hpc3.start();
        // Program VC2 before the refresh thread runs so the first frame has size.
        self.apply_host_display_resolution();
        if let Some(rex3) = &self.board.rex3 { rex3.start(); }
        if let Some(mgras) = &self.board.mgras { mgras.start_display(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.start(); }
        if let Some(gr2) = &self.board.gr2 { gr2.start(); }
        if let Some(u64) = &self.board.ultra64 { u64.start(); }

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
        crate::workstation::register_system_controller(&self.monitor, slot, self.event_rx.take());
    }

    pub fn stop(&mut self) {
        self.cpu.stop();
        if let Some(rex3) = &self.board.rex3 { rex3.stop(); }
        if let Some(mgras) = &self.board.mgras { mgras.stop_display(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.stop(); }
        if let Some(gr2) = &self.board.gr2 { gr2.stop(); }
        self.hpc3.stop();
        self.mc.stop();
        if let Some(u64) = &self.board.ultra64 { u64.stop(); }
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
    pub fn monitor_addr(&self) -> Option<std::net::SocketAddr> {
        self.monitor.bound_addr()
    }

    /// The terminal console: relays stdin to the monitor at `addr` and the
    /// monitor's output to stdout. When the monitor closes the connection
    /// (`quit`) the process exits, so `addr` must be this process's own
    /// monitor (`monitor_addr`): another iris's would take this one down
    /// with it when it quits.
    pub fn run_console_client(addr: std::net::SocketAddr) {
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

    /// csh one-liner to (re)mount `/CDROM` for SCSI unit `id` (EFS s7, else iso9660 vol).
    pub fn irix_cdrom_remount_command(scsi_id: u8) -> String {
        format!(
            "umount /CDROM >& /dev/null; \
             mount -t efs -o ro /dev/dsk/dks0d{id}s7 /CDROM || \
             mount -t iso9660 /dev/rdsk/dks0d{id}vol /CDROM\n",
            id = scsi_id
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

    /// Every SCSI bus on the board: the first controller's, and the second's
    /// on a Fullhouse Indigo2.
    fn scsi_buses(&self) -> Vec<&crate::scsi::ScsiBus> {
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

    /// Full rewind: load the named snapshot, which now captures the COW
    /// overlay too so the filesystem state is deterministic per snapshot.
    /// The CPU resumes automatically (load_snapshot restarts it). After the
    /// load, an in-memory checkpoint of the just-restored state is taken so
    /// the next `ci_rollback` can run without touching disk.
    pub fn ci_restore(&mut self, name: &str) -> Result<(), String> {
        // Clear any leftover serial bytes from the previous run so the
        // next command doesn't see stale output.
        if let Some(ci) = &self.ci_serial {
            ci.reset();
        }

        self.load_snapshot(name)?;
        self.last_restore = Some(name.to_string());
        // Capture the rollback checkpoint. If this fails, the restore still
        // succeeded — rollback will fall back to the disk path.
        match self.capture_rollback_checkpoint(name) {
            Ok(cp) => self.last_restore_checkpoint = Some(cp),
            Err(e) => {
                eprintln!("ci_restore: rollback checkpoint capture failed: {} — rollback will use the disk path", e);
                self.last_restore_checkpoint = None;
            }
        }
        Ok(())
    }

    /// Roll back to the state captured at the last `ci_restore`. Uses the
    /// in-memory checkpoint when present; falls back to a disk reload if it's
    /// absent (legacy snapshot loaded outside CI, or capture failed).
    pub fn ci_rollback(&mut self) -> Result<(), String> {
        if let Some(ci) = &self.ci_serial {
            ci.reset();
        }

        // Take the checkpoint out so the apply path can hold &cp without
        // borrowing self at the same time. Restored after apply so repeated
        // rollbacks work.
        let cp = match self.last_restore_checkpoint.take() {
            Some(cp) => cp,
            None => {
                let name = self.last_restore.clone()
                    .ok_or_else(|| "no previous restore to roll back to".to_string())?;
                eprintln!("ci_rollback: no in-memory checkpoint — falling back to disk reload");
                return self.ci_restore(&name);
            }
        };
        let result = self.apply_rollback_checkpoint(&cp);
        self.last_restore_checkpoint = Some(cp);
        result
    }

    /// Capture in-memory state for fast rollback. Stops the CPU briefly.
    fn capture_rollback_checkpoint(&mut self, name: &str) -> Result<RollbackCheckpoint, String> {
        self.stop();

        let cpu = self.cpu.save_state();
        let mc = self.mc.save_state();
        let ioc = self.hpc3.ioc().save_state();
        let scc = self.hpc3.ioc().scc().save_state();
        let pit = self.hpc3.ioc().pit().save_state();
        let ps2 = self.hpc3.ioc().ps2().save_state();
        let rtc = self.hpc3.rtc().save_state();
        let eeprom = self.hpc3.eeprom().lock().save_state_owned();
        let scsi = self.hpc3.scsi().save_state();
        let scsi1 = self.hpc3.scsi1().map(|dev| dev.save_state());
        let seeq = self.hpc3.seeq().save_state();
        let hpc3 = self.hpc3.save_state();
        let rex3 = self.board.rex3.as_ref().map(|r| r.save_state());
        let rex3_head1 = self.board.rex3_head1.as_ref().map(|r| r.save_state());

        let bank_words: [Vec<u32>; 4] = [
            self.board.phys.snapshot_bank_inmem(0),
            self.board.phys.snapshot_bank_inmem(1),
            self.board.phys.snapshot_bank_inmem(2),
            self.board.phys.snapshot_bank_inmem(3),
        ];

        let framebuffers = self.board.rex3.as_ref()
            .map(|r| r.snapshot_framebuffers_inmem());

        // Re-read cow.toml (and cow1.toml, for controller 1) so rollback
        // knows which dirty sectors to import back. The files were just
        // consumed by load_snapshot but they're tiny and re-reading from
        // page cache is cheap (~µs).
        let overlay_dir = std::path::PathBuf::from("saves").join(name);
        let snap = Snapshot::new(&overlay_dir);
        let overlay_sets = snap.read_toml("cow.toml").map(|v| parse_cow_table(&v)).unwrap_or_default();
        let overlay_sets1 = snap.read_toml("cow1.toml").map(|v| parse_cow_table(&v)).unwrap_or_default();

        self.restart_peripherals();
        self.cpu.start();

        Ok(RollbackCheckpoint {
            overlay_dir,
            overlay_sets,
            overlay_sets1,
            bank_words,
            framebuffers,
            cpu, mc, ioc, scc, pit, ps2, rtc, eeprom, scsi, scsi1, seeq, hpc3, rex3, rex3_head1,
        })
    }

    /// Apply an in-memory checkpoint, restoring the guest to the state at
    /// the moment of capture. Skips disk IO and TOML string-parsing.
    fn apply_rollback_checkpoint(&mut self, cp: &RollbackCheckpoint) -> Result<(), String> {
        self.stop();
        self.power_on_devices();

        self.cpu.load_state(&cp.cpu)?;
        self.mc.load_state(&cp.mc)?;
        self.hpc3.ioc().load_state(&cp.ioc)?;
        self.hpc3.ioc().scc().load_state(&cp.scc)?;
        self.hpc3.ioc().pit().load_state(&cp.pit)?;
        self.hpc3.ioc().ps2().load_state(&cp.ps2)?;
        self.hpc3.rtc().load_state(&cp.rtc)?;
        self.hpc3.eeprom().lock().load_state_mut(&cp.eeprom)?;
        self.hpc3.scsi().load_state(&cp.scsi)?;
        if let (Some(dev), Some(scsi1_toml)) = (self.hpc3.scsi1(), &cp.scsi1) {
            dev.load_state(scsi1_toml)?;
        }
        self.hpc3.seeq().load_state(&cp.seeq)?;
        self.hpc3.load_state(&cp.hpc3)?;
        if let (Some(rex3), Some(rex3_toml)) = (&self.board.rex3, &cp.rex3) {
            rex3.load_state(rex3_toml)?;
        }
        if let (Some(rex3), Some(rex3_toml)) = (&self.board.rex3_head1, &cp.rex3_head1) {
            rex3.load_state(rex3_toml)?;
        }

        for (i, words) in cp.bank_words.iter().enumerate() {
            self.board.phys.restore_bank_inmem(i, words);
        }
        if let (Some(rex3), Some((rgb, aux))) = (&self.board.rex3, &cp.framebuffers) {
            rex3.restore_framebuffers_inmem(rgb, aux);
        }

        // Reflink the overlay back into place. saves/<name>/scsi*.overlay is
        // unchanged by guest writes (writes go to the live overlay), so this
        // can re-import directly.
        self.hpc3.scsi().import_overlays(&cp.overlay_dir, &cp.overlay_sets)
            .map_err(|e| format!("rollback: COW overlay import: {}", e))?;
        if let Some(dev) = self.hpc3.scsi1() {
            // Controller 1's overlays live in a "ctrl1" subdirectory so a
            // device at the same id on both controllers never collides on
            // the same scsi<id>.overlay filename.
            dev.import_overlays(&cp.overlay_dir.join("ctrl1"), &cp.overlay_sets1)
                .map_err(|e| format!("rollback: COW overlay import (controller 1): {}", e))?;
        }

        self.restart_peripherals();
        self.cpu.start();
        Ok(())
    }

    /// Capture a `LiveCheckpoint` from the machine's current running state —
    /// see that struct's doc comment for how this differs from
    /// `capture_rollback_checkpoint` (no SCSI/overlay capture; no on-disk
    /// snapshot name involved at all). Leaves the whole machine stopped on
    /// return (CPU and peripherals) — every real caller immediately follows
    /// this with a `restore_live_checkpoint` call anyway (which restarts
    /// peripherals itself, CPU staying parked for inline single-stepping —
    /// see its own doc comment), so there's no need to restart anything
    /// here just to have it stopped again a moment later.
    pub(crate) fn capture_live_checkpoint(&mut self) -> LiveCheckpoint {
        self.stop();

        let cpu = self.cpu.save_state();
        let mc = self.mc.save_state();
        let ioc = self.hpc3.ioc().save_state();
        let scc = self.hpc3.ioc().scc().save_state();
        let pit = self.hpc3.ioc().pit().save_state();
        let ps2 = self.hpc3.ioc().ps2().save_state();
        let rtc = self.hpc3.rtc().save_state();
        let eeprom = self.hpc3.eeprom().lock().save_state_owned();
        let seeq = self.hpc3.seeq().save_state();
        let hpc3 = self.hpc3.save_state();
        let rex3 = self.board.rex3.as_ref().map(|r| r.save_state());
        let rex3_head1 = self.board.rex3_head1.as_ref().map(|r| r.save_state());

        let bank_words: [Vec<u32>; 4] = [
            self.board.phys.snapshot_bank_inmem(0),
            self.board.phys.snapshot_bank_inmem(1),
            self.board.phys.snapshot_bank_inmem(2),
            self.board.phys.snapshot_bank_inmem(3),
        ];

        let framebuffers = self.board.rex3.as_ref()
            .map(|r| r.snapshot_framebuffers_inmem());

        // Deliberately NOT restart_peripherals()/cpu.start() here — see this
        // function's own doc comment.

        LiveCheckpoint {
            bank_words,
            framebuffers,
            cpu, mc, ioc, scc, pit, ps2, rtc, eeprom, seeq, hpc3, rex3, rex3_head1,
        }
    }

    /// Restore a `LiveCheckpoint` captured by `capture_live_checkpoint`.
    /// Does not touch SCSI controller state or disk contents at all — see
    /// `LiveCheckpoint`'s doc comment. Leaves the CPU thread stopped on
    /// return (so a caller doing its own inline single-stepping via
    /// `cpu_step_one_inline_counting_instructions` doesn't race a
    /// free-running CPU thread for the executor lock), but restarts
    /// MC/HPC3/REX3 before returning — the full `self.stop()` below is only
    /// needed transiently, to safely load device state without a live
    /// device thread mutating it mid-load; there's no reason to leave
    /// peripherals stopped after that's done, and every prior version of
    /// this function that did was a real, repeatedly-hit bug (the rest of
    /// the monitor console — symbol lookups, device register reads —
    /// stayed unresponsive after every `jitcheck` for no reason).
    pub(crate) fn restore_live_checkpoint(&mut self, cp: &LiveCheckpoint) -> Result<(), String> {
        self.stop();

        self.cpu.load_state(&cp.cpu)?;
        self.mc.load_state(&cp.mc)?;
        self.hpc3.ioc().load_state(&cp.ioc)?;
        self.hpc3.ioc().scc().load_state(&cp.scc)?;
        self.hpc3.ioc().pit().load_state(&cp.pit)?;
        self.hpc3.ioc().ps2().load_state(&cp.ps2)?;
        self.hpc3.rtc().load_state(&cp.rtc)?;
        self.hpc3.eeprom().lock().load_state_mut(&cp.eeprom)?;
        self.hpc3.seeq().load_state(&cp.seeq)?;
        self.hpc3.load_state(&cp.hpc3)?;
        if let (Some(rex3), Some(rex3_toml)) = (&self.board.rex3, &cp.rex3) {
            rex3.load_state(rex3_toml)?;
        }
        if let (Some(rex3), Some(rex3_toml)) = (&self.board.rex3_head1, &cp.rex3_head1) {
            rex3.load_state(rex3_toml)?;
        }

        for (i, words) in cp.bank_words.iter().enumerate() {
            self.board.phys.restore_bank_inmem(i, words);
        }
        if let (Some(rex3), Some((rgb, aux))) = (&self.board.rex3, &cp.framebuffers) {
            rex3.restore_framebuffers_inmem(rgb, aux);
        }

        // Peripherals back up; CPU deliberately left stopped — see this
        // function's own doc comment.
        self.restart_peripherals();
        Ok(())
    }

    /// Restart everything `stop` stopped, less the CPU, without restarting the
    /// monitor server.
    fn restart_peripherals(&mut self) {
        self.mc.start();
        self.hpc3.start();
        if let Some(rex3) = &self.board.rex3 { rex3.start(); }
        if let Some(mgras) = &self.board.mgras { mgras.start_display(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.start(); }
        if let Some(gr2) = &self.board.gr2 { gr2.start(); }
        if let Some(u64) = &self.board.ultra64 { u64.start(); }
    }

    /// Helper to power-on reset all devices.
    /// Must be called with threads stopped.
    fn power_on_devices(&mut self) {
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
        self.apply_host_display_resolution();
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

        // Write the manifest first so `read_manifest` succeeds even if a later
        // step crashes — the partial snapshot is at least diagnosable.
        let mut manifest = Manifest::for_current_save();
        manifest.parent = self.last_restore.clone();
        // Provenance: which disks + nvram this state was captured against, so a
        // later restore can refuse a mismatched base disk / build. (features are
        // filled by for_current_save.)
        manifest.disks = self.disks.clone();
        manifest.nvram = Some(self.nvram_path.clone());
        // The CPU is a runtime choice now, so it is state the restore must match.
        manifest.cpu_model = Some(self.cpu.model_name().to_string());
        manifest.cpu_cache = Some(self.cpu.cache_kind().to_string());
        snap.write_manifest(&manifest).map_err(|e| e.to_string())?;
        let sv = manifest.schema_version;

        // Device state — schema_version=2 writes *.bin (postcard-encoded
        // BinValue tree); legacy writes *.toml. write_state encapsulates the
        // choice so this orchestrator stays format-agnostic.
        snap.write_state("cpu",    &self.cpu.save_state(),                         sv).map_err(|e| e.to_string())?;
        snap.write_state("mc",     &self.mc.save_state(),                          sv).map_err(|e| e.to_string())?;
        snap.write_state("ioc",    &self.hpc3.ioc().save_state(),                  sv).map_err(|e| e.to_string())?;
        snap.write_state("scc",    &self.hpc3.ioc().scc().save_state(),            sv).map_err(|e| e.to_string())?;
        snap.write_state("pit",    &self.hpc3.ioc().pit().save_state(),            sv).map_err(|e| e.to_string())?;
        snap.write_state("ps2",    &self.hpc3.ioc().ps2().save_state(),            sv).map_err(|e| e.to_string())?;
        snap.write_state("rtc",    &self.hpc3.rtc().save_state(),                  sv).map_err(|e| e.to_string())?;
        snap.write_state("eeprom", &self.hpc3.eeprom().lock().save_state_owned(),  sv).map_err(|e| e.to_string())?;
        snap.write_state("scsi",   &self.hpc3.scsi().save_state(),                 sv).map_err(|e| e.to_string())?;
        if let Some(dev) = self.hpc3.scsi1() {
            snap.write_state("scsi1", &dev.save_state(), sv).map_err(|e| e.to_string())?;
        }
        snap.write_state("seeq",   &self.hpc3.seeq().save_state(),                 sv).map_err(|e| e.to_string())?;
        snap.write_state("hpc3",   &self.hpc3.save_state(),                        sv).map_err(|e| e.to_string())?;

        // REX3 (optional — absent in headless config). Framebuffers are
        // included in the chunks manifest below for v3+; v2 wrote them as
        // standalone .bin files.
        if let Some(rex3) = &self.board.rex3 {
            snap.write_state("rex3", &rex3.save_state(), sv).map_err(|e| e.to_string())?;
            if sv < 3 {
                rex3.save_framebuffers(&snap.dir).map_err(|e| e.to_string())?;
            }
        }
        if let Some(td) = &self.board.testdev {
            snap.write_state("testdev", &td.save_state(), sv).map_err(|e| e.to_string())?;
        }
        if let Some(rex3) = &self.board.rex3_head1 {
            snap.write_state("rex3_head1", &rex3.save_state(), sv).map_err(|e| e.to_string())?;
            if sv < 3 {
                let dir = &snap.dir;
                rex3.save_framebuffers_named(dir, "rex3_head1").map_err(|e| e.to_string())?;
            }
        }
        if let Some(gr2) = &self.board.gr2 {
            snap.write_state("gr2", &gr2.save_state(), sv).map_err(|e| e.to_string())?;
        }
        if let Some(mgras) = &self.board.mgras {
            snap.write_state("mgras", &mgras.save_state(), sv).map_err(|e| e.to_string())?;
        }

        // Bulk memory: v3+ goes to the content-addressable chunk store
        // shared across all snapshots in `saves/.cas/`. v2 (legacy) writes
        // raw bank{N}.bin files. Chunk hashes go in chunks.bin so load can
        // walk the right chunks back out.
        if sv >= 3 {
            let store = ChunkStore::new("saves");
            let mut chunks = ChunksManifest::default();
            for i in 0..4 {
                let words = self.board.phys.snapshot_bank_inmem(i);
                chunks.bank_chunks[i] = put_words_as_chunks(&store, &words)
                    .map_err(|e| format!("CAS bank{} put: {}", i, e))?;
            }
            if let Some(rex3) = &self.board.rex3 {
                let (rgb, aux) = rex3.snapshot_framebuffers_inmem();
                let rgb_chunks = put_words_as_chunks(&store, &rgb)
                    .map_err(|e| format!("CAS rex3 rgb put: {}", e))?;
                let aux_chunks = put_words_as_chunks(&store, &aux)
                    .map_err(|e| format!("CAS rex3 aux put: {}", e))?;
                chunks.framebuffer_chunks = Some((rgb_chunks, aux_chunks));
            }
            snap.write_chunks_manifest(&chunks).map_err(|e| e.to_string())?;
        } else {
            for i in 0..4 {
                self.board.phys.save_bank(i, dir.join(format!("bank{}.bin", i))).map_err(|e| e.to_string())?;
            }
        }

        // COW overlays per SCSI device, plus a `cow.toml` with the dirty
        // sector set for each one. Keeps the on-disk filesystem state
        // consistent with the captured RAM.
        let overlays = self.hpc3.scsi().export_overlays(&snap.dir)
            .map_err(|e| format!("COW overlay export: {}", e))?;
        snap.write_toml("cow.toml", &build_cow_table(overlays))
            .map_err(|e| e.to_string())?;
        if let Some(dev) = self.hpc3.scsi1() {
            // Own subdirectory + cow1.toml so controller 1 never collides
            // with controller 0's scsi<id>.overlay filenames.
            let ctrl1_dir = snap.dir.join("ctrl1");
            std::fs::create_dir_all(&ctrl1_dir).map_err(|e| e.to_string())?;
            let overlays1 = dev.export_overlays(&ctrl1_dir)
                .map_err(|e| format!("COW overlay export (controller 1): {}", e))?;
            snap.write_toml("cow1.toml", &build_cow_table(overlays1))
                .map_err(|e| e.to_string())?;
        }

        self.restart_peripherals();
        // Resume execution so the session feels like it never paused.
        // Without this the user sees JIT shutdown stats and a dead prompt
        // after `save` — the CPU would otherwise stay stopped.
        self.cpu.start();
        println!("Snapshot saved to saves/{}", name);
        Ok(())
    }

    /// Restore full machine snapshot from `saves/<name>/`. CPU is auto-started
    /// at the end so the guest resumes from the snapshotted PC.
    /// For determinism validation use `load_snapshot_paused` instead.
    pub fn load_snapshot(&mut self, name: &str) -> Result<(), String> {
        self.load_snapshot_inner(name)?;
        self.cpu.start();
        println!("Snapshot loaded from saves/{}", name);
        Ok(())
    }

    /// Same body as `load_snapshot` but leaves CPU and peripheral threads
    /// stopped on return. Used by the Phase 3.3 determinism validator which
    /// must prevent any thread from running between load and digest, since
    /// thread scheduling jitter would mask CPU determinism issues.
    pub fn load_snapshot_paused(&mut self, name: &str) -> Result<(), String> {
        self.load_snapshot_inner(name)?;
        // load_snapshot_inner restarted peripherals; stop them again.
        self.hpc3.stop();
        self.mc.stop();
        if let Some(rex3) = &self.board.rex3 { rex3.stop(); }
        if let Some(rex3) = &self.board.rex3_head1 { rex3.stop(); }
        if let Some(gr2) = &self.board.gr2 { gr2.stop(); }
        Ok(())
    }

    /// Restore full machine snapshot from `saves/<name>/`.
    ///
    /// JIT-cache invariant: `self.stop()` exits the CPU thread, which drops
    /// the `CodeCache` owned by `run_jit_dispatch`. Subsequent `cpu.start()`
    /// (in the public `load_snapshot` wrapper) builds a fresh cache. So no
    /// explicit invalidation is needed here as long as that ownership
    /// pattern holds. The persistent JIT profile uses content_hash to skip
    /// stale entries (see `profile_stale` in dispatch.rs).
    fn load_snapshot_inner(&mut self, name: &str) -> Result<(), String> {
        self.stop();

        // Any prior in-memory rollback checkpoint is now stale (it described
        // a different snapshot). ci_restore will recapture if reached via
        // that path; the monitor `load` command leaves it cleared.
        self.last_restore_checkpoint = None;

        // Reset to clean state before loading
        self.power_on_devices();

        let dir = std::path::PathBuf::from("saves").join(name);
        let snap = Snapshot::new(&dir);

        // Validate the manifest before reading anything else. Legacy snapshots
        // (no snapshot.toml) are accepted with a warning. Cross-arch loads are
        // refused — FPU bit-layout differs between aarch64 and x86_64 and we
        // don't have migration plumbing yet.
        let schema_version = match snap.read_manifest()? {
            Some(m) => {
                if let Some(other) = &m.machine {
                    return Err(format!("snapshot {name} is of another machine ({other})"));
                }
                if m.host_arch != std::env::consts::ARCH {
                    return Err(format!(
                        "snapshot host_arch '{}' does not match current host '{}'; cross-arch load is not supported",
                        m.host_arch, std::env::consts::ARCH
                    ));
                }
                if m.schema_version > SCHEMA_VERSION {
                    return Err(format!(
                        "snapshot schema_version {} is newer than this iris build supports ({})",
                        m.schema_version, SCHEMA_VERSION
                    ));
                }
                if let Some(rev) = &m.iris_git_rev {
                    if let Some(my_rev) = option_env!("IRIS_GIT_REV") {
                        if rev != my_rev {
                            eprintln!("load_snapshot: snapshot was captured at iris {} but current build is {}", rev, my_rev);
                        }
                    }
                }

                // Provenance validation: build features + disk presence/size = hard
                // error; disk path and nvram = warn. IRIS_SNAPSHOT_SKIP_CHECK=1
                // downgrades the hard errors to warnings.
                let skip_check = std::env::var("IRIS_SNAPSHOT_SKIP_CHECK")
                    .map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
                let mut fatal: Vec<String> = Vec::new();

                if m.features.is_empty() && m.disks.is_empty() {
                    eprintln!("load_snapshot: snapshot {} predates provenance recording — skipping disk/feature checks", name);
                } else {
                    // Build features must match exactly.
                    let cur_features = enabled_features();
                    let snap_features: Vec<String> = m.features.iter()
                        .filter(|f| !crate::snapshot::RETIRED_FEATURES.contains(&f.as_str()))
                        .cloned().collect();
                    if !m.features.is_empty() && snap_features != cur_features {
                        fatal.push(format!(
                            "build features differ: snapshot [{}] vs current [{}]",
                            m.features.join(","), cur_features.join(",")
                        ));
                    }
                    // The CPU model is guest-visible (PRId/FIR, cache geometry) and
                    // no longer a build flag, so a mismatch is as fatal as a feature one.
                    if let Some(e) = crate::snapshot::cpu_model_mismatch(
                        m.cpu_model.as_deref(), self.cpu.model_name()) { fatal.push(e); }
                    if let Some(e) = crate::snapshot::cache_model_mismatch(
                        m.cpu_cache.as_deref(), m.cpu_model.as_deref(), self.cpu.cache_kind()) { fatal.push(e); }
                    // Every recorded disk must still be configured at the same SCSI
                    // id with the same size. The host *path* is only where the file
                    // happens to live (it moves when disks are relocated, e.g. into a
                    // dist/ dir), so a path-only difference is a warning, not fatal —
                    // a disk's identity for restore is its id + size, not its path.
                    for d in &m.disks {
                        match self.disks.iter().find(|c| c.id == d.id) {
                            None => fatal.push(format!(
                                "snapshot disk SCSI {} ('{}') is not configured in this run", d.id, d.path)),
                            Some(cur) if cur.size_bytes != d.size_bytes => fatal.push(format!(
                                "SCSI {} ('{}') size differs: snapshot {} bytes vs current {} bytes",
                                d.id, d.path, d.size_bytes, cur.size_bytes)),
                            Some(cur) if cur.path != d.path => eprintln!(
                                "load_snapshot: SCSI {} path differs: snapshot '{}' vs current '{}' (same size — continuing)",
                                d.id, d.path, cur.path),
                            Some(_) => {}
                        }
                    }
                    // nvram mismatch is non-fatal (eeprom state is in the snapshot).
                    if let Some(nv) = &m.nvram {
                        if nv != &self.nvram_path {
                            eprintln!("load_snapshot: nvram differs: snapshot '{}' vs current '{}' (continuing)", nv, self.nvram_path);
                        }
                    }
                }

                if !fatal.is_empty() {
                    let msg = format!("snapshot provenance mismatch:\n  - {}", fatal.join("\n  - "));
                    if skip_check {
                        eprintln!("load_snapshot: {} [IRIS_SNAPSHOT_SKIP_CHECK set — continuing anyway]", msg);
                    } else {
                        return Err(format!("{}\n(set IRIS_SNAPSHOT_SKIP_CHECK=1 to override)", msg));
                    }
                }

                m.schema_version
            }
            None => {
                eprintln!("load_snapshot: no snapshot.toml in {} — treating as legacy v0 (no manifest)", dir.display());
                0
            }
        };

        // Device state — read_state picks <base>.bin (v2+) or <base>.toml
        // (legacy). v2 also falls back to .toml if .bin is absent.
        let cpu = snap.read_state("cpu", schema_version).map_err(|e| e.to_string())?;
        self.cpu.load_state(&cpu)?;

        let mc = snap.read_state("mc", schema_version).map_err(|e| e.to_string())?;
        self.mc.load_state(&mc)?;

        let ioc = snap.read_state("ioc", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.ioc().load_state(&ioc)?;

        let scc = snap.read_state("scc", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.ioc().scc().load_state(&scc)?;

        let pit = snap.read_state("pit", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.ioc().pit().load_state(&pit)?;

        let ps2 = snap.read_state("ps2", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.ioc().ps2().load_state(&ps2)?;

        let rtc = snap.read_state("rtc", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.rtc().load_state(&rtc)?;

        let eeprom = snap.read_state("eeprom", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.eeprom().lock().load_state_mut(&eeprom)?;

        let scsi = snap.read_state("scsi", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.scsi().load_state(&scsi)?;

        // scsi1.* is absent from snapshots saved before the second SCSI
        // controller existed, and from guinness machines — both are
        // legitimate "leave controller 1 untouched" cases, not errors.
        if let Some(dev) = self.hpc3.scsi1() {
            if let Ok(scsi1) = snap.read_state("scsi1", schema_version) {
                dev.load_state(&scsi1)?;
            }
        }

        let seeq = snap.read_state("seeq", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.seeq().load_state(&seeq)?;

        let hpc3 = snap.read_state("hpc3", schema_version).map_err(|e| e.to_string())?;
        self.hpc3.load_state(&hpc3)?;

        if let Some(rex3) = &self.board.rex3 {
            let rex3_v = snap.read_state("rex3", schema_version).map_err(|e| e.to_string())?;
            rex3.load_state(&rex3_v)?;
            if schema_version < 3 {
                rex3.load_framebuffers(&snap.dir).map_err(|e| e.to_string())?;
            }
        }
        if let Some(td) = &self.board.testdev {
            if let Ok(v) = snap.read_state("testdev", schema_version) {
                td.load_state(&v)?;
            }
        }
        if let Some(rex3) = &self.board.rex3_head1 {
            if let Ok(rex3_v) = snap.read_state("rex3_head1", schema_version) {
                rex3.load_state(&rex3_v)?;
                if schema_version < 3 {
                    rex3.load_framebuffers_named(&snap.dir, "rex3_head1").map_err(|e| e.to_string())?;
                }
            }
        }
        if let Some(gr2) = &self.board.gr2 {
            if let Ok(gr2_v) = snap.read_state("gr2", schema_version) {
                gr2.load_state(&gr2_v)?;
            }
        }
        if let Some(mgras) = &self.board.mgras {
            if let Ok(mgras_v) = snap.read_state("mgras", schema_version) {
                mgras.load_state(&mgras_v)?;
            }
        }

        // Bulk memory: v3+ comes from the content-addressable chunk store
        // shared across snapshots; v2 reads raw bank{N}.bin files.
        if schema_version >= 3 {
            let store = ChunkStore::new("saves");
            let chunks = snap.read_chunks_manifest()
                .map_err(|e| format!("read chunks.bin: {}", e))?;
            for (i, hashes) in chunks.bank_chunks.iter().enumerate() {
                if hashes.is_empty() { continue; }
                let words = get_chunks_as_words(&store, hashes)
                    .map_err(|e| format!("CAS bank{} get: {}", i, e))?;
                self.board.phys.restore_bank_inmem(i, &words);
            }
            if let (Some(rex3), Some((rgb_h, aux_h))) = (&self.board.rex3, &chunks.framebuffer_chunks) {
                let rgb = get_chunks_as_words(&store, rgb_h)
                    .map_err(|e| format!("CAS rex3 rgb get: {}", e))?;
                let aux = get_chunks_as_words(&store, aux_h)
                    .map_err(|e| format!("CAS rex3 aux get: {}", e))?;
                rex3.restore_framebuffers_inmem(&rgb, &aux);
            }
        } else {
            for i in 0..4 {
                self.board.phys.load_bank(i, dir.join(format!("bank{}.bin", i))).map_err(|e| e.to_string())?;
            }
        }

        // COW overlays — best-effort for backward compatibility with
        // snapshots saved before overlay capture was added.
        if let Ok(cow_toml) = snap.read_toml("cow.toml") {
            let sets = parse_cow_table(&cow_toml);
            self.hpc3.scsi().import_overlays(&snap.dir, &sets)
                .map_err(|e| format!("COW overlay import: {}", e))?;
            if let Some(dev) = self.hpc3.scsi1() {
                if let Ok(cow1_toml) = snap.read_toml("cow1.toml") {
                    let sets1 = parse_cow_table(&cow1_toml);
                    dev.import_overlays(&snap.dir.join("ctrl1"), &sets1)
                        .map_err(|e| format!("COW overlay import (controller 1): {}", e))?;
                }
            }
        } else {
            eprintln!("load_snapshot: no cow.toml in snapshot — overlays left unchanged");
        }

        self.restart_peripherals();
        Ok(())
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
        // Release the monitor port and this Machine's device handles, so the next
        // Machine in the same process (iris-gui Stop -> Start) gets the monitor.
        self.monitor.shutdown();
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
    fn monitor_addr(&self) -> Option<std::net::SocketAddr> { Machine::monitor_addr(self) }
    fn lifecycle(&self) -> GuestState { self.lifecycle }
    fn set_lifecycle(&mut self, state: GuestState) { self.lifecycle = state; }

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
    fn save_snapshot(&mut self, name: &str) -> Result<(), String> { Machine::save_snapshot(self, name) }
    fn load_snapshot(&mut self, name: &str) -> Result<(), String> { Machine::load_snapshot(self, name) }
    fn ci_restore(&mut self, name: &str) -> Result<(), String> { Machine::ci_restore(self, name) }
    fn ci_rollback(&mut self) -> Result<(), String> { Machine::ci_rollback(self) }

    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        let rtc = self.hpc3.rtc();
        let path = path.unwrap_or_else(|| rtc.nvram_path()).to_string();
        rtc.save_nvram(&path).map(|_| path).map_err(|e| e.to_string())
    }
    fn scratch_path(&self) -> Option<&std::path::Path> { Machine::scratch_path(self) }
    fn disc_changer(&self) -> Option<Arc<dyn crate::workstation::DiscChanger>> {
        Some(self.hpc3.scsi().clone())
    }
    fn remount_cdrom_guest(&self, scsi_id: u8) { Machine::remount_cdrom_guest(self, scsi_id) }
    fn pending_chd_sync_count(&self) -> usize { Machine::pending_chd_sync_count(self) }
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
                Some(MachineProfile::FuelIp34) => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP30", "IP32", "O2"],
                None => &["Indy", "Indigo2", "IP24", "IP22", "IP28", "IP30", "IP32", "O2"],
            }
        };
        for p in [None, Some(MachineProfile::IndyIp24), Some(MachineProfile::Indigo2Ip22),
                  Some(MachineProfile::Indigo2Ip28), Some(MachineProfile::O2Ip32), Some(MachineProfile::OctaneIp30), Some(MachineProfile::FuelIp34)] {
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
        for p in [MachineProfile::IndyIp24, MachineProfile::Indigo2Ip22, MachineProfile::Indigo2Ip28, MachineProfile::O2Ip32, MachineProfile::OctaneIp30, MachineProfile::FuelIp34] {
            set_emulator_profile(p);
            assert_eq!(emulator_profile(), Some(p));
        }
        let name = emulator_name();
        assert!(name.starts_with('I') && name.contains(" Rust "), "{name}");
    }
}

#[cfg(test)]
mod controller_lifetime_tests {
    use super::*;

    fn build_machine() -> Box<Machine> {
        // Machine::new needs more stack than a test thread has (see main.rs).
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| Box::new(Machine::new(crate::bench_runner::bench_config(
                [16, 0, 0, 0], crate::config::CpuModel::default()))))
            .unwrap()
            .join()
            .unwrap()
    }

    /// #102: `machine-start`'s acknowledgement names the transition it
    /// performed, from a never-started machine through `Starting` to
    /// `Running`. A second start is a no-op, and a start after `machine-stop`
    /// takes the same route from `Halted`. The process-level test in
    /// `tests/monitor_ready.rs` covers the monitor command and readiness.
    #[test]
    fn start_guest_acknowledges_the_lifecycle_transition() {
        let mut machine = build_machine();
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

    /// iris-gui drops a Machine on Stop while the monitor keeps its
    /// SystemController. Every command through that controller afterwards used
    /// to dereference freed memory (SIGSEGV in `save_snapshot`).
    #[test]
    fn controller_outliving_its_machine_reports_not_running() {
        let mut machine = build_machine();
        machine.register_system_controller();
        let ctrl = SystemController { machine: machine.machine_slot() };
        drop(machine);
        for (cmd, args) in [("save", &["crashtest"][..]), ("reset", &[][..]), ("machine-stop", &[][..])] {
            let err = ctrl.execute_command(cmd, args, Box::new(std::io::sink())).unwrap_err();
            assert_eq!(err, "machine is not running", "{cmd}");
        }
    }
}
