use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::AtomicU64;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;

use crate::config::{GraphicsBoard, MachineConfig, MachineProfile, NetworkConfig};
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
use crate::mips_exec::{MipsExecutor, MipsCpu, MipsCpuConfig, MipsCpuDebugAdapter};
use crate::gdb_stub::CpuDebug;
// Step 1 keeps the cargo feature as the selector; step 3 makes this a runtime choice.
#[cfg(not(feature = "r5k"))]
use crate::mips_cache_v2::R4400Cache as SelectedCache;
#[cfg(feature = "r5k")]
use crate::mips_cache_v2::R5000Cache as SelectedCache;
use crate::hpc3::Hpc3;
use crate::ioc::{Ioc, GioSlot, GIO_SLOT_MAP, profile_idx};
use crate::monitor::Monitor;
use crate::rex3::Rex3;
use crate::snapshot::{Snapshot, Manifest, SCHEMA_VERSION, ChunksManifest, DiskRef, enabled_features};
use crate::chunk_store::{ChunkStore, get_chunks_as_words, put_words_as_chunks};
use crate::hptimer::TimerManager;

pub fn emulator_name() -> &'static str {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        if now % 4 != 0 {
            return "Irresponsible Rust IRIX Simulator".to_string();
        }

        let firsts = ["Irresponsible", "Incredible", "Insufferable", "Infuriating", "Inaccurate", "Incomplete", "Interactive", "Indomitable", "Irresistible"];
        let thirds = ["IRIX", "Indy", "Iris"];
        let fourths = ["Simulator", "System", "Substitute", "Sandbox"];

        let first = firsts[((now / 4) % firsts.len() as u64) as usize];
        let third = thirds[((now / 64) % thirds.len() as u64) as usize];
        let fourth = fourths[((now / 256) % fourths.len() as u64) as usize];

        format!("{} Rust {} {}", first, third, fourth)
    }).as_str()
}

pub struct Machine {
    cpu: Arc<MipsCpu<MipsTlb, SelectedCache>>,
    _phys: Arc<Physical>, // Keep reference to Physical Bus
    mc: MemoryController,
    hpc3: Hpc3,
    monitor: Arc<Monitor>,
    /// Sender for async machine events (HardReset, PowerOff) from devices.
    pub event_tx: mpsc::SyncSender<MachineEvent>,
    event_rx: Option<mpsc::Receiver<MachineEvent>>,
    timer_manager: Arc<TimerManager>,
    /// When `cfg.ci` is set, the channel-A backend is replaced by this
    /// in-process one so the CI control socket can drive the console.
    ci_serial: Option<Arc<crate::z85c30::CiSerialBackend>>,
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
    display_resolution: crate::vc2_timings::NewportResolution,
    /// Whether Newport compositor is active (not headless / not XZ board).
    newport_active: bool,
    /// Indigo2 IP22 fullhouse layout (`!guinness`).
    fullhouse: bool,
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
fn build_cow_table(overlays: Vec<(usize, Vec<u64>)>) -> toml::Value {
    let mut tbl = toml::map::Map::new();
    for (id, dirty) in overlays {
        let arr: Vec<toml::Value> = dirty.into_iter().map(|v| toml::Value::Integer(v as i64)).collect();
        tbl.insert(format!("scsi{}", id), toml::Value::Array(arr));
    }
    toml::Value::Table(tbl)
}

fn parse_cow_table(cow_toml: &toml::Value) -> Vec<(usize, Vec<u64>)> {
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
fn check_testdev_slot_free(#[cfg(feature = "ultra64")] ultra64_present: bool) {
    #[cfg(feature = "ultra64")]
    if ultra64_present {
        panic!("--test-device and the ultra64 dev board both claim GIO slot 0");
    }
}

impl Machine {
    pub fn new(cfg: MachineConfig) -> Self {
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
        testdev_override: Option<Arc<crate::testdev::TestDevice>>,
    ) -> Self {
        // Capture config flags that are needed after the local `cfg` binding
        // is shadowed later in this function.
        let ci_enabled = cfg.ci;
        let perf = cfg.perf.clone();
        #[cfg(feature = "jitv2")]
        let jitv2_threads = cfg.jitv2.threads;
        let display_resolution = cfg.graphics.resolution;
        let newport_active = !cfg.headless && cfg.graphics.board == GraphicsBoard::Newport;
        let clock_fixed_mhz = cfg.clock.fixed_mhz;

        if !cfg.machine.profile.supported() {
            eprintln!(
                "iris: machine profile \"{}\" is not implemented; use {}",
                cfg.machine.profile.label(),
                MachineProfile::IndyIp24.label(),
            );
            std::process::exit(1);
        }
        let guinness = cfg.machine.profile.guinness();

        // 0. EEPROMs. Real IP22 hardware has two distinct 93-series serial
        // EEPROM chips, not one shared part: a CPU-daughtercard chip wired to
        // the MC (REG_EEROM @ 0x1fa00030 — CPU boot config, incl. CACHSZ_REG),
        // and a motherboard chip wired to HPC3 (MISC_EEPROM_DATA @
        // 0x1fbb0008 — NVRAM/env vars/MAC, see Eeprom93c56::backdoor_set_mac).
        let eeprom_mc = Arc::new(Mutex::new(Eeprom93c56::new()));
        let eeprom_hpc3 = Arc::new(Mutex::new(Eeprom93c56::with_path(crate::devlog::LogModule::Nveeprom, cfg.nveeprom.clone())));
        // CACHSZ_REG (word 0x11): secondary cache size in 4KB pages.
        // PROM reads this when SC=1 (size_2nd_cache probe returns 0) to determine L2 size.
        // r5ksc without r5ksc_triton: external SC sized via EEPROM. 256 = 1MB (256 × 4KB).
        // r5ksc_triton: Triton reports L2 size via CONFIG_TR_SS — EEPROM word left 0.
        // r5k without r5ksc: no L2 — leave 0 so PROM sees no secondary cache.
        #[cfg(all(feature = "r5ksc", not(feature = "r5ksc_triton")))]
        eeprom_mc.lock().set_cachsz((<SelectedCache as crate::mips_cache_v2::MipsCache>::L2_SIZE / 4096) as u16);
        #[cfg(all(feature = "r5k", not(feature = "r5ksc")))]
        eeprom_mc.lock().set_cachsz(0);

        // 1. Create all devices first
        // Memory Controller
        let mc = MemoryController::new(eeprom_mc.clone(), guinness, cfg.banks);

        // RAM banks sized per config. addr_mask is initialized to mem_size-1;
        // remap_banks() updates it via set_addr_mask() when MEMCFG0/1 are written during POST.
        let banks = [
            Memory::new(cfg.banks[0].max(1) as usize),
            Memory::new(cfg.banks[1].max(1) as usize),
            Memory::new(cfg.banks[2].max(1) as usize),
            Memory::new(cfg.banks[3].max(1) as usize),
        ];

        // PROM (1MB at 0x1FC00000). IP22 (Indigo2) uses a different PROM image
        // than Indy: try cfg.prom, then 070-1367-012.bin in cwd, then fall back
        // to the embedded PROM0701367012 (see prombini2.rs) rather than Indy's PROM.
        let prom = if guinness {
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
        let ioc = if ci_enabled { Ioc::new_ci(guinness) } else { Ioc::new(guinness) };

        // CI mode replaces the default TCP backend on channel B (tty1, the
        // SGI serial console) with an in-process backend the control socket
        // drives directly. Channel A (tty2) keeps its default TCP backend.
        // Must happen before any peripheral `start()` call (which clones the
        // current backend Arc into the RX/TX threads).
        let ci_serial = if ci_enabled {
            let b = Arc::new(crate::z85c30::CiSerialBackend::new());
            if let Some(path) = cfg.serial_log.as_deref() {
                if let Err(e) = b.set_log_file(path) {
                    eprintln!("iris: serial_log: failed to open {}: {}", path, e);
                } else {
                    eprintln!("iris: serial console mirroring to {}", path);
                }
            }
            ioc.scc().set_backend_b(b.clone());
            Some(b)
        } else {
            // Non-CI mode: channel B already has its TCP listener on
            // 127.0.0.1:8881.  If --serial-log was passed, wrap it in a
            // TeeBackend so guest-emitted bytes get mirrored to the file
            // in addition to whatever client is attached to the TCP socket.
            if let Some(path) = cfg.serial_log.as_deref() {
                let inner = ioc.scc().backend_b();
                match crate::z85c30::TeeBackend::new(inner, path) {
                    Ok(tee) => {
                        ioc.scc().set_backend_b(Arc::new(tee));
                        eprintln!("iris: serial console mirroring to {}", path);
                    }
                    Err(e) => {
                        eprintln!("iris: serial_log: failed to open {}: {}", path, e);
                    }
                }
            }
            None
        };
        let timer_manager = Arc::new(TimerManager::new());
        ioc.set_timer_manager(timer_manager.clone());
        ioc.set_heartbeat(heartbeat.clone());
        let hpc3 = Hpc3::with_net(eeprom_hpc3.clone(), ioc.clone(), guinness, heartbeat.clone(), cfg.network(), cfg.no_audio, cfg.audio.clone(), cfg.nvram.clone(), cfg.scsi_deferred_int);
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
            eeprom_hpc3.lock().backdoor_set_mac_if_blank(cfg.network().mac);
        }

        // Attach SCSI devices from config (IDs 1–7).
        let mut scsi_ids: Vec<u8> = cfg.scsi.keys().copied().collect();
        scsi_ids.sort();
        // CI mode: isolate each COW overlay under /tmp so an interactive
        // iris holding {base}.overlay can coexist with any number of `--ci`
        // processes. Files are kept for post-mortem inspection; cleanup
        // happens on machine drop below.
        let ci_pid = std::process::id();
        // Track the on-disk path of any scratch device so the CI socket can
        // read/write its bytes directly (Phase 2.4).
        let mut scratch_path: Option<std::path::PathBuf> = None;
        for id in scsi_ids {
            let dev = &cfg.scsi[&id];
            // DaynaPort SCSI/Link: a network adapter, not storage. None of the
            // scratch / changer / overlay handling below applies — it has no
            // image at all.
            if dev.is_daynaport() {
                let params = match dev.daynaport_params(id) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("iris: fatal: SCSI ID {}: {}", id, e);
                        std::process::exit(1);
                    }
                };
                match hpc3.add_scsi_daynaport(dev.controller, id as usize, params) {
                    Ok(()) => println!(
                        "iris: DaynaPort SCSI/Link at SCSI ID {} — MAC {}, gateway {} (guest {}/{})",
                        id, crate::net::mac_str(&params.mac),
                        params.subnet.gateway_ip, params.subnet.client_ip, params.subnet.netmask),
                    Err(e) => {
                        let msg = format!("could not attach DaynaPort to SCSI ID {id}: {e}");
                        if std::env::var_os("IRIS_NO_EXIT_ON_POWEROFF").is_some() {
                            eprintln!("iris: warning: {msg}; continuing without SCSI ID {id}");
                        } else {
                            eprintln!("iris: fatal: {msg}");
                            std::process::exit(1);
                        }
                    }
                }
                continue;
            }
            // Scratch volume: pre-create a raw file with a minimal SGI Volume
            // Header if it doesn't exist. Refuse cdrom/overlay combinations —
            // scratch must be a host-writable raw file. Default size 64 MB.
            //
            // The VH lays out partition 7 ("vol") spanning sectors 8..end and
            // partition 8 ("vh") spanning sectors 0..7 (the VH itself).
            // Without a VH, IRIX recognises the device but returns I/O error
            // on every read because /dev/rdsk/dks0dNvh and /dev/rdsk/dks0dNvol
            // both consult the partition table at sector 0.
            //
            // Convention: host writes payload via scratch-write at offset >=
            // SCRATCH_PAYLOAD_OFFSET (4096). Guest reads from offset 0 of
            // /dev/rdsk/dks0dNvol (which maps to sector 8 of the disk by
            // partition 7's first_block=8).
            if dev.scratch {
                if dev.cdrom || dev.overlay {
                    println!("Note: SCSI ID {}: scratch=true is incompatible with cdrom/overlay; ignoring scratch flag", id);
                } else {
                    let path = std::path::Path::new(&dev.path);
                    if !path.exists() {
                        let size_mb = dev.size_mb.unwrap_or(64) as u64;
                        let bytes = size_mb * 1024 * 1024;
                        match crate::sgi_vh::create_scratch_image(path, bytes) {
                            Ok(()) => println!("iris: created scratch volume {} ({} MB, with SGI VH)", dev.path, size_mb),
                            Err(e) => println!("Note: could not create scratch volume {}: {}", dev.path, e),
                        }
                    }
                    if scratch_path.is_some() {
                        println!("Note: multiple scratch SCSI devices configured; CI socket will use the lowest-id one");
                    } else {
                        scratch_path = Some(path.to_path_buf());
                    }
                }
            }
            let (path, discs) = if dev.cdrom {
                // Build the changer list, skipping empty paths (an empty path
                // means "drive present, tray empty" — a valid CD-ROM state
                // where media is loaded later at runtime).
                let mut list: Vec<String> = Vec::new();
                if !dev.path.is_empty() {
                    list.push(dev.path.clone());
                }
                for d in &dev.discs {
                    if !d.is_empty() && !list.contains(d) {
                        list.push(d.clone());
                    }
                }
                // Active disc is the first entry, or empty (no media) if none.
                let active = list.first().cloned().unwrap_or_default();
                (active, list)
            } else {
                (dev.path.clone(), vec![])
            };
            let result = if ci_enabled && dev.overlay && !dev.cdrom {
                let ci_overlay = format!("/tmp/iris-ci-{}-scsi{}.overlay", ci_pid, id);
                hpc3.add_scsi_device_with_overlay(dev.controller, id as usize, &path, dev.cdrom, discs, dev.overlay, &ci_overlay)
            } else {
                hpc3.add_scsi_device(dev.controller, id as usize, &path, dev.cdrom, discs, dev.overlay)
            };
            if let Err(e) = result {
                // A configured disk that won't attach (a CHD path when built
                // without --features chd, or a disk the macOS sandbox won't let
                // us read) can't host this device.
                //
                // Standalone CLI fails loudly and exits — booting on with a
                // silently-missing device only yields a confusing PROM "no such
                // device" later. But an embedder (the GUI sets
                // IRIS_NO_EXIT_ON_POWEROFF) must never be torn down by the
                // library: process::exit kills it outright, and panicking can't
                // be caught either because release builds use panic="abort"
                // (unwinding across the libchdman/JIT FFI would be UB). So we
                // skip just this device and boot without it; the GUI's Start
                // preflight already warns the user about an unreadable disk, and
                // they can stop, re-pick or detach it, and start again.
                let msg = format!("could not attach {path} to SCSI ID {id}: {e}");
                if std::env::var_os("IRIS_NO_EXIT_ON_POWEROFF").is_some() {
                    eprintln!("iris: warning: {msg}; continuing without SCSI ID {id}");
                    continue;
                }
                eprintln!("iris: fatal: {msg}");
                std::process::exit(1);
            }
        }

        // Disk + nvram provenance for snapshot manifests. Captured here while
        // the MachineConfig `cfg` is still in scope (it is shadowed by the CPU
        // config below). Identity is the configured path + host file size.
        let mut disk_provenance: Vec<DiskRef> = cfg.scsi.iter()
            .filter(|(_, dev)| !dev.is_daynaport()) // no image behind a DaynaPort
            .map(|(&id, dev)| {
                let size_bytes = std::fs::metadata(&dev.path).map(|m| m.len()).unwrap_or(0);
                DiskRef { id, path: dev.path.clone(), size_bytes }
            }).collect();
        disk_provenance.sort_by_key(|d| d.id);
        let nvram_provenance = cfg.nvram.clone();

        // REX3 Graphics — Newport only; skipped in headless mode or when XZ board selected
        let rex3: Option<Arc<Rex3>> = if cfg.headless || cfg.graphics.board != crate::config::GraphicsBoard::Newport {
            None
        } else {
            let r = Arc::new(Rex3::new(heartbeat.clone(), fasttick_count.clone(), decoded_count.clone(), Arc::clone(&l1i_hit_count), Arc::clone(&l1i_fetch_count), Arc::clone(&uncached_fetch_count)));
            let wiring = GIO_SLOT_MAP[profile_idx(guinness)][GioSlot::Gfx as usize];
            let ioc_clone = ioc.clone();
            r.set_vblank_callback(Arc::new(move |active| {
                ioc_clone.set_interrupt(wiring.retrace, active);
            }));
            let ioc_ff = ioc.clone();
            r.set_fifo_full_callback(Arc::new(move |active| {
                ioc_ff.set_interrupt(wiring.fifo_full, active);
            }));
            Some(r)
        };

        let rex3_head1: Option<Arc<Rex3>> = if cfg.headless || cfg.graphics.heads < 2 {
            None
        } else {
            let r = Arc::new(Rex3::new(heartbeat, fasttick_count.clone(), decoded_count.clone(), Arc::clone(&l1i_hit_count), Arc::clone(&l1i_fetch_count), Arc::clone(&uncached_fetch_count)));
            let wiring = GIO_SLOT_MAP[profile_idx(guinness)][GioSlot::Exp0 as usize];
            let ioc_clone = ioc.clone();
            r.set_vblank_callback(Arc::new(move |active| {
                ioc_clone.set_interrupt(wiring.retrace, active);
            }));
            let ioc_ff = ioc.clone();
            r.set_fifo_full_callback(Arc::new(move |active| {
                ioc_ff.set_interrupt(wiring.fifo_full, active);
            }));
            Some(r)
        };

        // Indy XZ/Elan preview stub — same GIO gfx slot as Newport, no compositor.
        let xz: Option<Arc<crate::xz::Xz>> = if guinness && cfg.graphics.board == crate::config::GraphicsBoard::Xz {
            Some(Arc::new(crate::xz::Xz::new()))
        } else {
            None
        };

        // Indigo2 IMPACT/MGRAS preview — multi-slot GIO stub.
        let mgras: Option<Arc<crate::mgras::Mgras>> = if !guinness && cfg.impact.any_enabled() {
            Some(Arc::new(crate::mgras::Mgras::new(&cfg.impact)))
        } else {
            None
        };

        // N64 development board (Ultra64) — GIO slot 0 at 0x1F400000
        #[cfg(feature = "ultra64")]
        let ultra64: Option<Arc<crate::ultra64::Ultra64>> = if cfg.ultra64.enabled {
            match crate::ultra64::Ultra64::new(ioc.clone()) {
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
        // `cfg` is shadowed by MipsCpuConfig further down; grab this first.
        let cheritest_hook = cfg.cheritest_dump_hook;

        // Bare-metal test device (--test-device): default off, and refused
        // alongside the ultra64 dev board, which claims the same GIO slot.
        let testdev = if let Some(dev) = testdev_override {
            check_testdev_slot_free(
                #[cfg(feature = "ultra64")]
                ultra64.is_some(),
            );
            Some(dev)
        } else if cfg.test_device {
            check_testdev_slot_free(
                #[cfg(feature = "ultra64")]
                ultra64.is_some(),
            );
            let path = cfg.test_device_dump.clone()
                .unwrap_or_else(|| crate::testdev::DEFAULT_DUMP_PATH.to_string());
            eprintln!("iris: test device enabled at {:#010x}, dumps to {}",
                      crate::testdev::TEST_DEV_BASE, path);
            Some(Arc::new(crate::testdev::TestDevice::new(path)))
        } else {
            None
        };

        let phys_raw = Physical::new(
            banks,
            rex3.clone(),
            rex3_head1.clone(),
            xz.clone(),
            mgras.clone(),
            #[cfg(feature = "ultra64")]
            ultra64,
            testdev,
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

        // Connect VINO to System Memory, install a video source, start DMA.
        // Source kind + broadcast standard come from `[vino]` in iris.toml.
        phys.vino.set_phys(phys.clone());
        let standard = match cfg.vino.standard {
            crate::config::VinoStandard::Ntsc => crate::video_source::VideoStandard::Ntsc,
            crate::config::VinoStandard::Pal  => crate::video_source::VideoStandard::Pal,
        };
        let source: Option<Arc<dyn crate::video_source::VideoSource>> = match cfg.vino.source {
            crate::config::VinoSource::Camera => {
                #[cfg(feature = "camera")]
                {
                    let idx = cfg.vino.camera_index;
                    match crate::camera::CameraSource::new_with_index(standard, idx) {
                        Ok(c)  => Some(Arc::new(c)),
                        Err(e) => {
                            eprintln!("VINO: camera {} unavailable ({}); using black source", idx, e);
                            Some(Arc::new(crate::video_source::BlackSource::new(standard)))
                        }
                    }
                }
                #[cfg(not(feature = "camera"))]
                {
                    eprintln!("VINO: source=\"camera\" set but iris was built without --features camera; using test pattern");
                    Some(Arc::new(crate::video_source::TestPatternSource::new(standard)))
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
                Arc::new(crate::video_source::CdmcAdjustedSource::new(source, phys.vino.clone()));
            phys.vino.set_source(adjusted);
            phys.vino.start();
        }

        // 5. CPU config + TLB + Executor
        let cfg = MipsCpuConfig::indy();
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let sysad: Arc<dyn BusDevice> = phys.clone();
        let mut executor: MipsExecutor<MipsTlb, SelectedCache> = MipsExecutor::new(sysad, tlb, &cfg);

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

        // Pin the CP0 Count frequency if the user asked for a fixed clock
        // (guests like Linux/MIPS whose periodic tick doesn't fit IRIX's
        // slow/fast two-bucket auto-inference model). Must happen before the
        // core starts executing.
        if let Some(mhz) = clock_fixed_mhz {
            executor.core.set_fixed_clock_hz((mhz * 1_000_000.0) as u64);
        }

        // Inject the shared fasttick_count Arc into the executor core before wrapping in MipsCpu.
        // (cycles has no equivalent here — it's inline on MipsCore.hot now.)
        executor.core.fasttick_count = fasttick_count;
        executor.decoded_count       = decoded_count;
        executor.uncached_fetch_count = Arc::clone(&uncached_fetch_count);
        executor.cache.l1i_hit_count   = Arc::clone(&l1i_hit_count);
        executor.cache.l1i_fetch_count = Arc::clone(&l1i_fetch_count);
        // Re-sync raw pointers after Arc injection (the Arcs above replaced the ones captured in new()).
        executor.rebind_atomic_ptrs();

        // Share count_hz_atomic from MipsCore with Rex3 so the refresh thread can display it.
        #[cfg(feature = "developer")]
        if let Some(rex3) = &phys.rex3 { rex3.set_count_hz_atomic(Arc::clone(&executor.core.count_hz_atomic)); }

        // Give the core the machine's hptimer manager: CP0 Compare writes
        // arm a one-shot on it that raises IP7 from the timer thread. Safe
        // to hand over before the move into MipsCpu below — nothing gets
        // armed (no raw pointer captured) until the CPU actually executes a
        // Compare write, by which time the core sits at its final address
        // inside the executor's Arc<Mutex<..>>.
        executor.core.set_timer_manager(timer_manager.clone());

        let cpu = Arc::new(MipsCpu::new(executor));

        // Connect CPU to MC and IOC for signaling
        let cpu_device: Arc<dyn Device> = cpu.clone();
        mc.set_cpu(Arc::downgrade(&cpu_device));
        ioc.set_interrupts(cpu.interrupts_ptr());

        // Wire up the CPU cycle counter for every device that reads it —
        // Rex3::new/Wd33c93a::new run before the CPU exists, so this can't
        // be a constructor parameter (see Hot::cycles's doc comment). Must
        // happen before hpc3.scsi().start() (called later, from
        // Machine::start) actually spawns the worker thread that reads it.
        if let Some(rex3) = &phys.rex3 { rex3.set_cpu_cycles(cpu.cycles_ptr()); }
        if let Some(rex3) = &phys.rex3_head1 { rex3.set_cpu_cycles(cpu.cycles_ptr()); }
        if let Some(td) = &phys.testdev { td.attach_core(cpu.core_ptr()); }
        if cheritest_hook {
            if phys.testdev.is_none() {
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
        let mut monitor = Monitor::new();
        monitor.register_device(devlog.clone());
        monitor.register_device(cpu.clone());
        monitor.register_device(Arc::new(mc.clone()));
        monitor.register_device(Arc::new(hpc3.clone()));
        monitor.register_device(phys.clone());
        if let Some(rex3) = &phys.rex3 { monitor.register_device(rex3.clone()); }
        if let Some(rex3) = &phys.rex3_head1 { monitor.register_device(rex3.clone()); }
        if let Some(td) = &phys.testdev { monitor.register_device(td.clone()); }
        if let Some(xz) = &phys.xz { monitor.register_device(xz.clone()); }
        if let Some(mgras) = &phys.mgras { monitor.register_device(mgras.clone()); }
        #[cfg(feature = "ultra64")]
        if let Some(u64) = &phys.ultra64 { monitor.register_device(u64.clone()); }
        monitor.register_device(Arc::new(phys.vino.clone()));
        monitor.register_device(crate::perf_monitor::PerfMonitor::new(
            cpu.running_flag(),
            cpu.cycles_ptr(),
            cpu.fasttick_count.clone(),
            phys.rex3.clone(),
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
            if let Some(rex3) = &phys.rex3 { rex3.register_locks(); }
            if let Some(rex3) = &phys.rex3_head1 { rex3.register_locks(); }
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
            _phys: phys,
            mc,
            hpc3,
            monitor,
            event_tx,
            event_rx: Some(event_rx),
            timer_manager,
            ci_serial,
            last_restore: None,
            last_restore_checkpoint: None,
            scratch_path,
            disks: disk_provenance,
            nvram_path: nvram_provenance,
            display_resolution,
            newport_active,
            fullhouse: !guinness,
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
                crate::vc2_timings::NewportResolution::Res1280x1024
            } else {
                return;
            }
        } else {
            self.display_resolution
        };
        if let Some(rex3) = &self._phys.rex3 {
            rex3.apply_display_resolution(mode);
        }
        if let Some(rex3) = &self._phys.rex3_head1 {
            rex3.apply_display_resolution(mode);
        }
    }

    /// Path of the configured scratch SCSI volume, if any. Used by the CI
    /// socket scratch-{write,read,clear,info} commands to act on the file
    /// directly while the machine is briefly stopped.
    pub fn scratch_path(&self) -> Option<&std::path::Path> {
        self.scratch_path.as_deref()
    }

    /// Briefly stop the machine, run `work`, then restart peripherals and the
    /// CPU only if it was running before. Used by the scratch-write/read/clear
    /// CI commands to mutate the scratch file without racing the SCSI device's
    /// in-flight reads. CPU stays stopped if the harness hasn't called `start`
    /// yet — a file injected before boot stays injected, the CPU doesn't get
    /// auto-started.
    pub fn with_paused<R>(&mut self, work: impl FnOnce() -> R) -> R {
        let was_running = self.cpu.is_running();
        self.stop();
        let r = work();
        self.restart_peripherals();
        if was_running {
            self.cpu.start();
        }
        r
    }

    pub fn start(&mut self) {
        // Start peripherals
        self.mc.start();
        self.hpc3.start();
        // Program VC2 before the refresh thread runs so the first frame has size.
        self.apply_host_display_resolution();
        if let Some(rex3) = &self._phys.rex3 { rex3.start(); }
        if let Some(rex3) = &self._phys.rex3_head1 { rex3.start(); }
        #[cfg(feature = "ultra64")]
        if let Some(u64) = &self._phys.ultra64 { u64.start(); }

        // Monitor server on localhost:8888 — always start, even in CI mode,
        // so debug helpers (status/regs/bt/dis) stay reachable while iris-ci
        // drives the serial console.
        self.monitor.clone().start_server("127.0.0.1:8888".to_string());

        // CI mode: the harness drives startup via `restore` / `start`. Don't
        // autostart the CPU so the first command finds a quiet machine.
        #[cfg(not(any(debug_assertions, feature = "developer")))]
        if self.ci_serial.is_none() {
            self.cpu.start();
        }
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
                                // Hosts that embed iris as a library (e.g. iris-gui)
                                // set IRIS_NO_EXIT_ON_POWEROFF=1 so a guest halt
                                // does not kill the host process.
                                #[cfg(not(feature = "developer"))]
                                if std::env::var_os("IRIS_NO_EXIT_ON_POWEROFF").is_none() {
                                    std::process::exit(0);
                                }
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
        if let Some(rex3) = &self._phys.rex3_head1 { rex3.stop(); }
        self.hpc3.stop();
        self.mc.stop();
        #[cfg(feature = "ultra64")]
        if let Some(u64) = &self._phys.ultra64 { u64.stop(); }
    }

    pub fn run_console_client() {
        println!("IRIS: {}", emulator_name());
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

    /// Borrow the HPC3 controller — used by CI socket commands that touch
    /// RTC/SCSI directly (`rtc-save`, `cdrom-eject`).
    pub fn hpc3(&self) -> &Hpc3 {
        &self.hpc3
    }

    pub fn get_rex3(&self) -> Option<Arc<crate::rex3::Rex3>> {
        self._phys.rex3.clone()
    }

    pub fn get_rex3_head1(&self) -> Option<Arc<crate::rex3::Rex3>> {
        self._phys.rex3_head1.clone()
    }

    pub fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timer_manager.clone()
    }

    /// Return a type-erased CpuDebug handle for the GDB stub.
    pub fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> {
        MipsCpuDebugAdapter::new(self.cpu.clone())
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
    pub fn get_ci_serial(&self) -> Option<Arc<crate::z85c30::CiSerialBackend>> {
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

    /// Number of attached CHD disks whose `.diff.chd` holds changes pending a
    /// fold-back into the base on a clean shutdown (the "Synchronizing disks"
    /// step). 0 means a clean exit needs no disk sync.
    pub fn pending_chd_sync_count(&self) -> usize {
        self.hpc3.scsi().pending_chd_sync_count()
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
        self.hpc3.scsi().sync_chd_disks(only, progress, cancel)
    }

    /// Cumulative count of guest-originated Ethernet frames the NAT engine has
    /// processed. Monotonic for the life of the machine; an embedder samples the
    /// delta to tell whether the guest's internal networking is alive.
    pub fn net_guest_frames(&self) -> u64 {
        self.hpc3.seeq().nat_control().guest_frames()
    }

    /// NAT addresses the emulator hands the guest: (ec0 client IP, gateway IP,
    /// netmask) — the source of truth for what the guest's ec0 should be.
    pub fn nat_expected(&self) -> (std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr) {
        self.hpc3.seeq().gateway_addrs()
    }

    /// The guest's own source IP as last seen on the wire (None if no frame has
    /// revealed one yet). Captured passively, so it works even when the guest's
    /// networking is misconfigured and nothing routes.
    pub fn net_observed_guest_ip(&self) -> Option<std::net::Ipv4Addr> {
        self.hpc3.seeq().nat_control().observed_guest_ip()
    }

    /// The guest's likely default gateway, inferred passively from the in-subnet
    /// address it keeps ARP-ing for but can't resolve. None if none seen.
    pub fn net_observed_gateway(&self) -> Option<std::net::Ipv4Addr> {
        self.hpc3.seeq().nat_control().observed_gateway()
    }

    /// Live status of the PCAP bridged-capture backend. `Inactive` unless this is
    /// a `--features pcap` build whose active machine uses `mode = "pcap"`; goes
    /// `PermissionDenied` when the raw capture can't be opened for lack of
    /// privilege, which the GUI turns into an "Enable packet capture" prompt.
    pub fn net_pcap_status(&self) -> crate::net::PcapStatus {
        self.hpc3.seeq().nat_control().pcap_status()
    }

    /// Move the running NAT onto a new subnet without a reboot: the NAT thread
    /// swaps its `(gateway, client, netmask)` and flushes connection state on
    /// its next loop. Typically gateway = network+1, client = network+2.
    pub fn set_nat_subnet(&self, gateway: std::net::Ipv4Addr, client: std::net::Ipv4Addr, netmask: std::net::Ipv4Addr) {
        self.hpc3.seeq().nat_control().request_subnet(gateway, client, netmask);
    }

    /// Tell the NAT engine the host's own IPv4 networks `(network, prefix)` so it
    /// won't adopt a guest subnet that overlaps them (which would shadow the
    /// host's real LAN). The embedder gathers these from the host interfaces.
    pub fn set_host_nets(&self, nets: Vec<(std::net::Ipv4Addr, u8)>) {
        self.hpc3.seeq().nat_control().set_host_nets(nets);
    }

    /// Replace the running NAT's inbound port-forward rules without a reboot;
    /// the NAT thread rebinds its host listeners on its next loop.
    pub fn set_port_forwards(&self, rules: Vec<crate::config::PortForwardConfig>) {
        self.hpc3.seeq().nat_control().set_port_forwards(rules);
    }

    /// Change the PCAP bridged host interface (`None` = auto-pick) on a running
    /// machine without rebooting the guest: the PcapEngine reopens its capture on
    /// the new NIC. No-op in NAT mode.
    pub fn set_pcap_interface(&self, iface: Option<String>) {
        self.hpc3.seeq().nat_control().request_pcap_interface(iface);
    }

    /// Step the CPU `n` instructions in-line on the calling thread, with all
    /// peripheral threads stopped so the CPU sees no external interrupts.
    /// Used by Phase 3.3 snapshot determinism validator.
    /// Caller must arrange `load_snapshot_paused` first.
    pub fn cpu_step_n_inline(&self, n: u64) -> Result<u64, String> {
        self.cpu.step_n_inline(n)
    }

    /// Snapshot the deterministic-from-state CPU registers.
    pub fn cpu_state_digest(&self) -> Result<crate::mips_exec::CpuStateDigest, String> {
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
    pub fn cpu_restore_state_digest(&self, digest: &crate::mips_exec::CpuStateDigest) -> Result<(), String> {
        self.cpu.restore_state_digest(digest)
    }

    /// Force `cp0_count`/`count_hz` to a reference digest's values — see
    /// `MipsCpu::fixup_cp0_count`'s doc comment.
    #[cfg(feature = "developer")]
    pub fn cpu_fixup_cp0_count(&self, digest: &crate::mips_exec::CpuStateDigest) -> Result<(), String> {
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
        let rex3 = self._phys.rex3.as_ref().map(|r| r.save_state());
        let rex3_head1 = self._phys.rex3_head1.as_ref().map(|r| r.save_state());

        let bank_words: [Vec<u32>; 4] = [
            self._phys.snapshot_bank_inmem(0),
            self._phys.snapshot_bank_inmem(1),
            self._phys.snapshot_bank_inmem(2),
            self._phys.snapshot_bank_inmem(3),
        ];

        let framebuffers = self._phys.rex3.as_ref()
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
        if let (Some(rex3), Some(rex3_toml)) = (&self._phys.rex3, &cp.rex3) {
            rex3.load_state(rex3_toml)?;
        }
        if let (Some(rex3), Some(rex3_toml)) = (&self._phys.rex3_head1, &cp.rex3_head1) {
            rex3.load_state(rex3_toml)?;
        }

        for (i, words) in cp.bank_words.iter().enumerate() {
            self._phys.restore_bank_inmem(i, words);
        }
        if let (Some(rex3), Some((rgb, aux))) = (&self._phys.rex3, &cp.framebuffers) {
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
        let rex3 = self._phys.rex3.as_ref().map(|r| r.save_state());
        let rex3_head1 = self._phys.rex3_head1.as_ref().map(|r| r.save_state());

        let bank_words: [Vec<u32>; 4] = [
            self._phys.snapshot_bank_inmem(0),
            self._phys.snapshot_bank_inmem(1),
            self._phys.snapshot_bank_inmem(2),
            self._phys.snapshot_bank_inmem(3),
        ];

        let framebuffers = self._phys.rex3.as_ref()
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
        if let (Some(rex3), Some(rex3_toml)) = (&self._phys.rex3, &cp.rex3) {
            rex3.load_state(rex3_toml)?;
        }
        if let (Some(rex3), Some(rex3_toml)) = (&self._phys.rex3_head1, &cp.rex3_head1) {
            rex3.load_state(rex3_toml)?;
        }

        for (i, words) in cp.bank_words.iter().enumerate() {
            self._phys.restore_bank_inmem(i, words);
        }
        if let (Some(rex3), Some((rgb, aux))) = (&self._phys.rex3, &cp.framebuffers) {
            rex3.restore_framebuffers_inmem(rgb, aux);
        }

        // Peripherals back up; CPU deliberately left stopped — see this
        // function's own doc comment.
        self.restart_peripherals();
        Ok(())
    }

    /// Restart peripherals (MC, HPC3, REX3) without restarting the monitor server.
    fn restart_peripherals(&mut self) {
        self.mc.start();
        self.hpc3.start();
        if let Some(rex3) = &self._phys.rex3 { rex3.start(); }
        if let Some(rex3) = &self._phys.rex3_head1 { rex3.start(); }
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
        if let Some(dev) = self.hpc3.scsi1() { dev.power_on(); }
        // Seeq/Ethernet: reset regs + signal NAT flush.
        self.hpc3.seeq().power_on();
        // HAL2: reset all audio registers and channel state (timers already stopped).
        if let Some(hal2) = self.hpc3.hal2() { hal2.power_on(); }
        self.hpc3.power_on();
        if let Some(rex3) = &self._phys.rex3 { rex3.power_on(); }
        if let Some(rex3) = &self._phys.rex3_head1 { rex3.power_on(); }
        if let Some(td) = &self._phys.testdev { td.power_on(); }
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
        if let Some(rex3) = &self._phys.rex3 {
            snap.write_state("rex3", &rex3.save_state(), sv).map_err(|e| e.to_string())?;
            if sv < 3 {
                rex3.save_framebuffers(&snap.dir).map_err(|e| e.to_string())?;
            }
        }
        if let Some(td) = &self._phys.testdev {
            snap.write_state("testdev", &td.save_state(), sv).map_err(|e| e.to_string())?;
        }
        if let Some(rex3) = &self._phys.rex3_head1 {
            snap.write_state("rex3_head1", &rex3.save_state(), sv).map_err(|e| e.to_string())?;
            if sv < 3 {
                let dir = &snap.dir;
                rex3.save_framebuffers_named(dir, "rex3_head1").map_err(|e| e.to_string())?;
            }
        }
        if let Some(xz) = &self._phys.xz {
            snap.write_state("xz", &xz.save_state(), sv).map_err(|e| e.to_string())?;
        }
        if let Some(mgras) = &self._phys.mgras {
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
                let words = self._phys.snapshot_bank_inmem(i);
                chunks.bank_chunks[i] = put_words_as_chunks(&store, &words)
                    .map_err(|e| format!("CAS bank{} put: {}", i, e))?;
            }
            if let Some(rex3) = &self._phys.rex3 {
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
                self._phys.save_bank(i, dir.join(format!("bank{}.bin", i))).map_err(|e| e.to_string())?;
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
        if let Some(rex3) = &self._phys.rex3 { rex3.stop(); }
        if let Some(rex3) = &self._phys.rex3_head1 { rex3.stop(); }
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
                    if !m.features.is_empty() && m.features != cur_features {
                        fatal.push(format!(
                            "build features differ: snapshot [{}] vs current [{}]",
                            m.features.join(","), cur_features.join(",")
                        ));
                    }
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

        if let Some(rex3) = &self._phys.rex3 {
            let rex3_v = snap.read_state("rex3", schema_version).map_err(|e| e.to_string())?;
            rex3.load_state(&rex3_v)?;
            if schema_version < 3 {
                rex3.load_framebuffers(&snap.dir).map_err(|e| e.to_string())?;
            }
        }
        if let Some(td) = &self._phys.testdev {
            if let Ok(v) = snap.read_state("testdev", schema_version) {
                td.load_state(&v)?;
            }
        }
        if let Some(rex3) = &self._phys.rex3_head1 {
            if let Ok(rex3_v) = snap.read_state("rex3_head1", schema_version) {
                rex3.load_state(&rex3_v)?;
                if schema_version < 3 {
                    rex3.load_framebuffers_named(&snap.dir, "rex3_head1").map_err(|e| e.to_string())?;
                }
            }
        }
        if let Some(xz) = &self._phys.xz {
            if let Ok(xz_v) = snap.read_state("xz", schema_version) {
                xz.load_state(&xz_v)?;
            }
        }
        if let Some(mgras) = &self._phys.mgras {
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
                self._phys.restore_bank_inmem(i, &words);
            }
            if let (Some(rex3), Some((rgb_h, aux_h))) = (&self._phys.rex3, &chunks.framebuffer_chunks) {
                let rgb = get_chunks_as_words(&store, rgb_h)
                    .map_err(|e| format!("CAS rex3 rgb get: {}", e))?;
                let aux = get_chunks_as_words(&store, aux_h)
                    .map_err(|e| format!("CAS rex3 aux get: {}", e))?;
                rex3.restore_framebuffers_inmem(&rgb, &aux);
            }
        } else {
            for i in 0..4 {
                self._phys.load_bank(i, dir.join(format!("bank{}.bin", i))).map_err(|e| e.to_string())?;
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
            #[cfg(all(feature = "jitv2", feature = "developer"))]
            ("jitcheck".to_string(), "jitcheck <n> [skip] — capture live state, run n instructions interpreter-only vs real JIT dispatch, reconverge past the first `skip` divergences and stop at the next one [DEV]".to_string()),
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
            #[cfg(all(feature = "jitv2", feature = "developer"))]
            "jitcheck" => {
                let n: u64 = args.first()
                    .and_then(|s| s.parse().ok())
                    .ok_or("Usage: jitcheck <n> [skip]".to_string())?;
                let skip: u64 = args.get(1)
                    .map(|s| s.parse().map_err(|_| "Usage: jitcheck <n> [skip]".to_string()))
                    .transpose()?
                    .unwrap_or(0);
                let _ = writeln!(writer, "jitcheck: capturing live state, running {} instructions interpreter-only then with real JIT dispatch (skipping first {} divergence(s))...", n, skip);
                self.with_machine(|m| {
                    let report = crate::validate::validate_jit_determinism(m, n, skip)
                        .map_err(|e| format!("jitcheck failed: {}", e))?;
                    for (i, d) in report.skipped.iter().enumerate() {
                        let _ = writeln!(writer, "jitcheck: SKIPPED divergence #{} at instruction {} of {} (pc={:#018x}), reconverged and continued", i + 1, d.instruction, n, d.replay_pc);
                        for (field, a, b) in &d.diffs {
                            let _ = writeln!(writer, "  {}: interp={} jit={}", field, a, b);
                        }
                    }
                    let next_divergence_number = report.skipped.len() + 1;
                    match &report.stopped_at {
                        None => {
                            let _ = writeln!(writer, "jitcheck: no further divergence across {} instructions", n);
                            let _ = writeln!(writer, "jitcheck: CPU left stopped (peripherals still running) with the replay (JIT-dispatch) pass's final state loaded — 'start' to resume the CPU");
                        }
                        Some(d) => {
                            let _ = writeln!(writer, "jitcheck: DIVERGED (divergence #{}) at instruction {} of {} (pc={:#018x})", next_divergence_number, d.instruction, n, d.replay_pc);
                            for (field, a, b) in &d.diffs {
                                let _ = writeln!(writer, "  {}: interp={} jit={}", field, a, b);
                            }
                            let _ = writeln!(writer, "jitcheck: rerun with skip={} to reconverge past this one too", next_divergence_number);
                            let _ = writeln!(writer, "jitcheck: stopped immediately at the divergence — CPU left stopped exactly as the JIT-dispatch pass left it (peripherals still running, console stays live) — 'start' to resume the CPU");
                        }
                    }
                    Ok(())
                })
            }
            _ => Err(format!("Unknown command: {}", cmd)),
        }
    }
}
