//! The O2 (IP32) as a machine: iris's CPU thread and window around the IP32
//! bus and devices (`ip32.rs`).
//!
//! Selected with `[machine] profile = "o2_ip32"` and configured by `[o2]`
//! (see [`crate::config::O2Section`]). Between instructions, two things run:
//!
//! - a service thread keeps the devices' interrupts current in CRIME and
//!   CRIME's in the CPU's IP2, runs the MACE clock from the host clock, and
//!   ends the run when the guest switches the machine off;
//! - a batch hook on the CPU thread hands the cache over with where the
//!   guest is running: the full model holds data until POST has finished
//!   (the PROM keeps its stack in it while it tests memory), the shadow model
//!   runs everything after, and POST starting over (a reboot) hands back.
//!   Both CPUs do this ([`HandoverCache`]).
//!
//! The screen is GBE scanned out of RAM by [`O2Display`], presented through
//! the same renderer the other machines use; keys from the window go to the
//! MACE PS/2 keyboard.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::config::{CpuModel, MachineConfig};
use crate::cpu::mips_core::CyclesPtr;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::ip32_handover::{HandoverCache, R10000Handover, R5000Handover};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::crime_mem::O2Memory;
use crate::dev::ng1::rex3::Renderer;
use crate::disp::Rex3Screen;
use crate::hptimer::TimerManager;
use crate::ip32::{Ip32Bus, Ip32Options, FIRMWARE_LOAD_VA, FIRMWARE_SPAN, NO_ECC_BASE, RAM_BASE, RAM_WINDOW_SIZE};
use crate::traits::{BusDevice, Device, MachineEvent, Signal};
use crate::workstation::{MachineRef, MachineSlot, Workstation};

const PROM_VA: std::ops::Range<u32> = 0xbfc0_0000..0xbfc8_0000;

/// Whether `pc` is in the firmware the PROM copies into RAM, at its KSEG0
/// address or its KSEG1 alias (0xa100_0000, uncached): it runs from both.
fn in_loaded_firmware(pc: u32) -> bool {
    let fw = FIRMWARE_LOAD_VA..FIRMWARE_LOAD_VA + FIRMWARE_SPAN;
    fw.contains(&pc) || fw.contains(&(pc.wrapping_sub(0x2000_0000)))
}

/// Whether `pc` is inside the PROM or its firmware copied into RAM.
fn in_firmware(pc: u32) -> bool {
    PROM_VA.contains(&pc) || in_loaded_firmware(pc)
}

/// The kernel takes over (`true`): hand over to the shadow model. The PROM
/// starts again (`false`): back to the full model. True if the model changed.
fn switch_model<F, S>(c: &HandoverCache<F, S>, kernel: bool) -> bool
where
    F: MipsCache,
    S: MipsCache,
{
    if kernel && !c.on_shadow() {
        c.hand_over();
        true
    } else if !kernel && c.on_shadow() {
        c.hand_back();
        true
    } else {
        false
    }
}

/// Where the batch hook is in the machine's boot, shared with the machine so
/// a snapshot carries it: whether POST has finished, whether the CPU is in
/// the firmware, and the FPU-enable bit the firmware was lent (`arcs_fpu`).
struct BootState {
    post_done: AtomicBool,
    inside: AtomicBool,
    cu1_saved: std::sync::atomic::AtomicU32,
}

impl BootState {
    fn new() -> Self {
        Self { post_done: AtomicBool::new(false), inside: AtomicBool::new(true), cu1_saved: Default::default() }
    }

    fn save(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("post_done".into(), toml::Value::Boolean(self.post_done.load(Ordering::Relaxed)));
        t.insert("inside".into(), toml::Value::Boolean(self.inside.load(Ordering::Relaxed)));
        t.insert("cu1_saved".into(), crate::snapshot::hex_u32(self.cu1_saved.load(Ordering::Relaxed)));
        toml::Value::Table(t)
    }

    fn load(&self, v: &toml::Value) {
        use crate::snapshot::{get_field, toml_bool, toml_u32};
        self.post_done.store(get_field(v, "post_done").and_then(toml_bool).unwrap_or(true), Ordering::Relaxed);
        self.inside.store(get_field(v, "inside").and_then(toml_bool).unwrap_or(false), Ordering::Relaxed);
        self.cu1_saved.store(get_field(v, "cu1_saved").and_then(toml_u32).unwrap_or(0), Ordering::Relaxed);
    }
}

/// The CPU's interrupt pins as a controller: line n is IPn.
struct CpuIrq(Arc<dyn crate::traits::Device>);

impl crate::irq::InterruptController for CpuIrq {
    fn set_line(&self, line: u32, level: bool) {
        self.0.signal(Signal::Interrupt(line, level));
    }
}

pub struct O2Machine {
    pub bus: Arc<Ip32Bus>,
    cpu: Arc<dyn CpuDevice>,
    timer_manager: Arc<TimerManager>,
    display: Arc<O2Display>,
    running: Arc<AtomicBool>,
    threads: Vec<std::thread::JoinHandle<()>>,
    /// The shared monitor (telnet, `monitor_port`): the CPU's commands and
    /// every O2 device's, through the bus.
    monitor: Arc<crate::monitor::Monitor>,
    monitor_port: u16,
    /// The console UART's in-process host end in CI mode (`--ci`), which the
    /// CI socket types into and reads.
    ci_serial: Option<Arc<crate::dev::z85c30::CiSerialBackend>>,
    /// The NAT's controls and the addresses it hands the guest (client,
    /// gateway, netmask), when the Ethernet is plugged in.
    nat: Option<(Arc<crate::net::NatControl>, (std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr))>,
    /// The guest switching the machine off, for the machine-events thread.
    events: std::sync::mpsc::SyncSender<MachineEvent>,
    event_rx: Option<std::sync::mpsc::Receiver<MachineEvent>>,
    /// Set once that thread runs; until then the service loop exits the
    /// process itself on a power-off, as it always has.
    controller_up: Arc<AtomicBool>,
    controller_slot: Option<MachineSlot>,
    /// The batch hook's place in the boot (see [`BootState`]).
    boot: Arc<BootState>,
    /// Set while the machine is stopped: the service thread leaves the
    /// devices alone, so nothing changes memory under a snapshot.
    paused: Arc<AtomicBool>,
    /// The disks as a snapshot records them, to refuse one taken on others.
    disks: Vec<crate::snapshot::DiskRef>,
    /// The flash file, recorded in a snapshot's manifest.
    flash_path: Option<String>,
    /// The `scratch = true` volume, for the CI socket's scratch commands.
    scratch_path: Option<std::path::PathBuf>,
    /// Where the machine is in its start/stop lifecycle (see
    /// `Workstation::lifecycle`); never `Running`, which `status` reads from
    /// the CPU.
    lifecycle: crate::machine_state::GuestState,
}

/// What an O2 snapshot's manifest names as its machine.
const SNAPSHOT_MACHINE: &str = "o2_ip32";

/// The PROM image tried when `prom` is left at the shared default: the last
/// release's, named as SGI shipped it. The O2 has no embedded PROM.
pub const O2_PROM_FILE: &str = "ip32prom.rev4.18.bin";

/// CP0 Count when `[clock] fixed_mhz` is unset: half of a 180 MHz CPU.
pub const O2_COUNT_MHZ: f64 = 90.0;

impl O2Machine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        let o2 = &cfg.o2;
        // `prom`, or the last release's file when `prom` is the shared default
        // and that file is not there: there is no embedded O2 PROM.
        let prom_path = if cfg.prom != crate::config::DEFAULT_PROM || std::path::Path::new(&cfg.prom).exists() {
            cfg.prom.clone()
        } else {
            O2_PROM_FILE.to_string()
        };
        let prom = std::fs::read(&prom_path)
            .map_err(|e| format!("O2 PROM {prom_path}: {e} (set `prom` to the O2 PROM image; 4.18 is the last release)"))?;
        // `banks`: the kit in each DIMM pair from A (validate checked them).
        let kits = cfg.o2_kits();
        let mem = O2Memory::from_kits(&kits)?;
        // The PROM's flash, and so its environment, lives in `nvram`.
        let opts = Ip32Options { scsi: true, flash_file: Some(cfg.nvram.clone().into()) };
        let mut bus = Ip32Bus::with_options(mem, prom, opts);
        // `[network] mac` is the address in the 1-Wire ID chip.
        if let Some(mac) = cfg.network.mac.as_deref() {
            let mac = crate::config::parse_mac(mac).map_err(|e| format!("network.mac: {e}"))?;
            bus.mace.onewire = crate::dev::onewire::OneWireId::from_mac(mac);
        }
        // MACE audio plays through the host unless audio is off.
        if !cfg.no_audio {
            bus.mace.audio.enable_output(cfg.audio.clone());
        }
        // `headless`: no keyboard, so the PROM and the OS take the serial
        // console, as on an O2 booted without one. The GBE stays: it is part
        // of every O2.
        if !cfg.headless {
            bus.attach_keyboard(o2.mouse);
        }
        let bus = Arc::new(bus);
        // The SCSI devices, from the same `[scsi.N]` tables as the other
        // machines; `controller` is the bus (0 internal, 1 external).
        let mut scratch_path: Option<std::path::PathBuf> = None;
        {
            let ctls = bus.scsi.lock().unwrap();
            let mut ids: Vec<u8> = cfg.scsi.keys().copied().collect();
            ids.sort();
            for key in ids {
                let dev = &cfg.scsi[&key];
                let (controller, id) = dev.placement(key);
                let at = crate::config::scsi_label(controller, id);
                if dev.is_daynaport() {
                    return Err(format!("{at}: kind = \"daynaport\" is not available on the O2"));
                }
                if let Some(path) = dev.prepare_scratch(key) {
                    if scratch_path.is_some() {
                        println!("Note: multiple scratch SCSI devices configured; CI socket will use the lowest-id one");
                    } else {
                        scratch_path = Some(path);
                    }
                }
                let ctl = ctls.get(controller as usize).ok_or_else(|| {
                    format!("{at}: controller {controller} (an O2 has controllers 0 and 1)")
                })?;
                let (path, discs) = dev.media();
                let cdrom = dev.is_cdrom();
                // CI mode keeps each copy-on-write overlay under /tmp, as the
                // Indy does, so an interactive iris holding PATH.overlay and
                // any number of CI runs can share the image.
                let ci_overlay = (cfg.ci && dev.overlay && !cdrom).then(|| crate::config::ScsiDeviceConfig::ci_overlay_path(key));
                match ctl.attach_device(id, &path, cdrom, discs, dev.overlay, ci_overlay.as_deref()) {
                    Ok(_) if path.is_empty() => eprintln!("ip32: {at}: CD-ROM, tray empty"),
                    Ok(blocks) => eprintln!("ip32: {at}: {path} ({blocks} blocks)"),
                    Err(e) => {
                        // As on the Indy: a configured disk that will not attach
                        // ends a command-line run, since booting on without it
                        // only yields a confusing PROM "no such device" later;
                        // an embedder (the GUI sets IRIS_NO_EXIT_ON_POWEROFF)
                        // boots without the device instead.
                        let msg = format!("could not attach {path} to {at}: {e}");
                        if std::env::var_os("IRIS_NO_EXIT_ON_POWEROFF").is_some() {
                            eprintln!("iris: warning: {msg}; continuing without {at}");
                        } else {
                            return Err(msg);
                        }
                    }
                }
            }
        }
        eprintln!(
            "iris: O2 with {} MB ({}), {}",
            kits.iter().sum::<u32>(),
            kits.iter().map(|k| format!("{k}")).collect::<Vec<_>>().join("+"),
            if cfg.headless { "serial console" } else { "graphics console" }
        );
        // The console UART's host end: in CI mode the in-process backend
        // the CI socket types into, otherwise TCP on `serial_port_a` (as the
        // Indy's channel A), either teed to `serial_log` when one is set. The
        // second port serves TCP on `serial_port_b`.
        let [port_a, port_b] = cfg.serial_ports();
        let ci_serial = cfg.ci.then(|| Arc::new(crate::dev::z85c30::CiSerialBackend::new()));
        if let Some(ci) = &ci_serial {
            if let Some(path) = cfg.serial_log.as_deref() {
                match crate::dev::serial_log::SerialLog::create(
                    std::path::Path::new(path),
                    cfg.serial_log_timestamps,
                ) {
                    Ok(log) => {
                        ci.set_log(log);
                        eprintln!("iris: O2 serial console mirroring to {path}");
                    }
                    Err(e) => eprintln!("iris: O2 serial log {path}: {e}"),
                }
            }
            bus.mace.com0.set_backend(ci.clone());
        } else {
            let mut backend = crate::dev::z85c30::null_serial_backend();
            match crate::dev::z85c30::tcp_serial_backend(port_a) {
                Some((b, p)) => {
                    eprintln!("iris: O2 serial port 1 on 127.0.0.1:{p}");
                    backend = b;
                }
                None => eprintln!("iris: O2 serial port 1: port {port_a} unavailable; is another iris using it? (serial_port_a)"),
            }
            if let Some(path) = &cfg.serial_log {
                // A fresh log each boot, as the O2 has always kept it: scripts
                // look at its tail for this boot's prompts.
                match crate::dev::serial_log::SerialLog::create(
                    std::path::Path::new(path),
                    cfg.serial_log_timestamps,
                ) {
                    Ok(log) => {
                        backend = Arc::new(crate::dev::z85c30::TeeBackend::new(backend.clone(), log));
                    }
                    Err(e) => eprintln!("iris: O2 serial log {path}: {e}"),
                }
            }
            bus.mace.com0.set_backend(backend);
        }
        match crate::dev::z85c30::tcp_serial_backend(port_b) {
            Some((b, p)) => {
                eprintln!("iris: O2 serial port 2 on 127.0.0.1:{p}");
                bus.mace.com1.set_backend(b);
            }
            None => eprintln!("iris: O2 serial port 2: port {port_b} unavailable; is another iris using it? (serial_port_b)"),
        }
        // The Ethernet on the NAT gateway `[network]` describes (`mode =
        // "off"` leaves it unplugged; validate refuses pcap here for now).
        let mut nat = None;
        if cfg.network.mode != crate::config::NetMode::Off {
            let net = crate::net::GatewayConfig::from_network(&cfg.network());
            eprintln!("iris: O2 Ethernet on NAT: gateway {}, guest {} (DHCP)", net.gateway_ip, net.client_ip);
            let addrs = (net.client_ip, net.gateway_ip, net.netmask);
            nat = Some((bus.mace.mec.start_nat(net), addrs));
        }

        let timer_manager = Arc::new(TimerManager::new());
        let sysad: Arc<dyn BusDevice> = bus.clone();
        let arcs_fpu = o2.arcs_fpu;
        // `[clock] fixed_mhz` is CP0 Count; the CPU clock the PROM measures
        // and reports is twice it.
        let count_mhz = cfg.clock.fixed_mhz.unwrap_or(O2_COUNT_MHZ);
        let cpu_mhz = (count_mhz * 2.0).round().max(1.0) as u32;
        let boot = Arc::new(BootState::new());
        let cpu: Arc<dyn CpuDevice> = match cfg.machine.cpu {
            // The full cache model through the PROM and the tagless shadow
            // model once the kernel is in charge: the JIT then reaches RAM
            // through the ppmem window with no tag probe, LRU or dirty bit on
            // every load and store (memory-bound code ran 2.5-4x slower than on
            // IP28 with the full model).
            CpuModel::R10000 => Self::build_cpu::<R10000Handover>(sysad, &timer_manager, arcs_fpu, cpu_mhz, boot.clone(), switch_model, HandoverCache::take_reset_seen),
            CpuModel::R5000 => Self::build_cpu::<R5000Handover>(sysad, &timer_manager, arcs_fpu, cpu_mhz, boot.clone(), switch_model, HandoverCache::take_reset_seen),
            CpuModel::R4400 => return Err("an O2 has an R5000 or an R10000 ([machine] cpu)".into()),
        };

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
            // Codegen inline needs and deadlocks the CPU thread's flush (its
            // join waits on workers parked on the Jitv2 lock it holds).
            #[cfg(not(feature = "jitv2_lockstep"))]
            {
                let stats = g.stats.clone();
                g.compile_queue.start(bus.clone() as Arc<dyn BusDevice>, stats);
            }
            // The compile pool keeps only a weak handle on the CPU.
            std::mem::forget(cpu_device);
        }

        // ppmem: the CPU's cache and the JIT reach RAM through the memory's
        // window, publishing the mapped bitmap into the CPU's own fields, as
        // the Indigo2 does. Remaps happen on the CPU thread (a CRIME bank
        // control write), which is the window's contract.
        {
            use crate::ppmem::MappedMemory;
            let ram = &*bus.ram;
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
        }

        // CRIME's output is the CPU's IP2, pushed the moment it changes.
        bus.mace.ust.use_host_clock();
        bus.connect_cpu_irq(crate::irq::IrqLine::new(Arc::new(CpuIrq(cpu.clone().as_device())), 2));

        bus.gbe.use_cpu_clock(cpu.cycles_ptr(), cpu_mhz);
        // CRIME's rendering engine draws on its own thread, fed through
        // its interface buffer.
        bus.crime.re.start(bus.clone(), |b: &Ip32Bus| &b.crime.re);
        let running = Arc::new(AtomicBool::new(false));
        let display = Arc::new(O2Display::new(bus.clone(), cpu.cycles_ptr(), cpu.fasttick_count(), running.clone()));
        if let Some(port) = o2.control_port {
            match std::net::TcpListener::bind(("127.0.0.1", port)) {
                Ok(l) => {
                    eprintln!("iris: O2 control on 127.0.0.1:{port} (the monitor's `o2` commands; `help o2` there lists them)");
                    let bus = bus.clone();
                    std::thread::Builder::new().name("O2-Control".into()).spawn(move || control_server(l, bus)).unwrap();
                }
                Err(e) => eprintln!("iris: O2 control port {port}: {e}"),
            }
        }
        // The shared monitor, as the other machines have it: the CPU's
        // commands (registers, breakpoints, memory) and the O2's devices'.
        let mut monitor = crate::monitor::Monitor::new();
        monitor.register_device(crate::devlog::init_devlog());
        monitor.register_device(cpu.clone().as_device());
        monitor.register_device(bus.clone());
        // The SCSI controllers' `scsi`/`scsi1` and `cow`/`cow1`.
        for ctl in bus.scsi.lock().unwrap().iter() {
            monitor.register_device(ctl.clone());
        }
        monitor.register_device(Arc::new(O2Control::new(bus.clone())));
        let monitor_port = cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT);

        let (events, event_rx) = std::sync::mpsc::sync_channel(4);
        // A snapshot's disks by slot (controller * 16 + id) and size.
        let mut disks: Vec<crate::snapshot::DiskRef> = cfg
            .scsi
            .iter()
            .filter(|(_, d)| !d.is_cdrom())
            .map(|(&key, d)| crate::snapshot::DiskRef {
                id: { let (c, id) = d.placement(key); crate::config::scsi_slot(c, id) },
                path: d.path.clone(),
                size_bytes: std::fs::metadata(&d.path).map(|m| m.len()).unwrap_or(0),
            })
            .collect();
        disks.sort_by_key(|d| d.id);
        Ok(Self {
            bus,
            cpu,
            timer_manager,
            display,
            running,
            threads: Vec::new(),
            monitor: Arc::new(monitor),
            monitor_port,
            ci_serial,
            nat,
            events,
            event_rx: Some(event_rx),
            controller_up: Arc::new(AtomicBool::new(false)),
            controller_slot: None,
            boot,
            paused: Arc::new(AtomicBool::new(false)),
            lifecycle: crate::machine_state::GuestState::Idle,
            disks,
            flash_path: Some(cfg.nvram.clone()),
            scratch_path,
        })
    }

    /// `kernel_mode(cache, in_kernel)` switches the cache model on leaving the
    /// firmware for the kernel (true) and on a PROM starting over (false),
    /// returning whether it changed, so the JIT's memory pointers are
    /// re-published. `reset_seen(cache)` says the PROM has started over.
    fn build_cpu<C>(sysad: Arc<dyn BusDevice>, tm: &Arc<TimerManager>, arcs_fpu: bool, mhz: u32, boot: Arc<BootState>,
                    kernel_mode: fn(&C, bool) -> bool, reset_seen: fn(&C) -> bool) -> Arc<dyn CpuDevice>
    where
        C: crate::cpu::mips_cache_v2::CpuModel + From<Arc<dyn BusDevice>> + 'static,
        MipsCpu<MipsTlb, C>: CpuDevice,
    {
        let cfg = MipsCpuConfig::for_model::<C>();
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let mut exec: MipsExecutor<MipsTlb, C> = MipsExecutor::new(sysad, tlb, &cfg);
        exec.core.set_timer_manager(tm.clone());
        // Count runs at half the pipeline clock on both CPUs.
        exec.core.set_count_hz(mhz.max(1) as u64 * 500_000);
        // The no-ECC alias of memory reaches memory, not the PCI window the
        // bus decodes at the same low 32 bits.
        exec.phys_alias = Some((NO_ECC_BASE, RAM_WINDOW_SIZE as u64, RAM_BASE as u64));
        // CRIME's decode, like the MC's, is modelled on the low 32 bits.
        exec.phys_mask = u32::MAX as u64;
        exec.batch_hook = Some(Box::new(move |exec: &mut MipsExecutor<MipsTlb, C>| {
            let pc = exec.core.pc as u32;
            LAST_PC.store(pc, Ordering::Relaxed);
            BATCHES.fetch_add(1, Ordering::Relaxed);
            for (slot, v) in LAST_REGS.iter().zip(exec.core.gpr.iter()) {
                slot.store(*v, Ordering::Relaxed);
            }
            LAST_REGS[32].store(exec.core.cp0_status as u64, Ordering::Relaxed);
            LAST_REGS[33].store(exec.core.cp0_cause as u64, Ordering::Relaxed);
            LAST_REGS[34].store(exec.core.cp0_epc, Ordering::Relaxed);
            LAST_REGS[35].store(exec.core.cp0_badvaddr, Ordering::Relaxed);
            // The timer, for the input stall: Count as last materialized,
            // Compare, the pending-interrupt word and the Compare-match count.
            LAST_REGS[36].store(exec.core.cp0_count, Ordering::Relaxed);
            LAST_REGS[37].store(exec.core.cp0_compare, Ordering::Relaxed);
            LAST_REGS[38].store(exec.core.hot.interrupts.load(Ordering::Relaxed), Ordering::Relaxed);
            LAST_REGS[39].store(exec.core.fasttick_count.load(Ordering::Relaxed), Ordering::Relaxed);
            // Instructions retired, for an idle-MIPS reading over the control port.
            LAST_REGS[40].store(exec.core.hot.cycles, Ordering::Relaxed);
            LAST_REGS[41].store(exec.core.cp0_config as u64, Ordering::Relaxed);
            if !boot.post_done.load(Ordering::Relaxed) {
                // POST is over once the firmware it loaded into RAM runs.
                if in_loaded_firmware(pc) {
                    boot.post_done.store(true, Ordering::Relaxed);
                } else {
                    return;
                }
            }
            // The boot PROM starting again (a reboot): the full model, so POST
            // has a cache that holds its stack. POST initialising the cache
            // tags says so (`take_reset_seen`), not the PC: `sash` and the
            // IRIX kernel call into the PROM's own code, which is no reboot,
            // and IRIX reboots by jumping into the PROM, which a batch
            // boundary need not catch at the reset vector.
            if reset_seen(&exec.cache) && kernel_mode(&exec.cache, false) {
                eprintln!("iris: O2 cache: full model again (PC {pc:#010x}, ra {:#010x})", exec.core.gpr[31] as u32);
                // Not back to waiting for POST: the PROM's cache routine
                // (`sash` calls it before starting a kernel) sets the tags too,
                // and the next code outside the firmware hands over again.
                boot.inside.store(true, Ordering::Relaxed);
                #[cfg(feature = "jitv2")]
                exec.install_jit_mem_ptrs();
                return;
            }
            let fw = in_firmware(pc);
            if fw != boot.inside.load(Ordering::Relaxed) {
                boot.inside.store(fw, Ordering::Relaxed);
                if !fw && kernel_mode(&exec.cache, true) {
                    eprintln!("iris: O2 cache: shadow model from PC {pc:#010x}");
                    #[cfg(feature = "jitv2")]
                    exec.install_jit_mem_ptrs();
                }
                if arcs_fpu {
                    const CU1: u32 = 1 << 29;
                    if fw {
                        boot.cu1_saved.store(exec.core.cp0_status & CU1, Ordering::Relaxed);
                        exec.core.cp0_status |= CU1;
                    } else {
                        exec.core.cp0_status = (exec.core.cp0_status & !CU1) | boot.cu1_saved.load(Ordering::Relaxed);
                    }
                    exec.resync_privilege_state();
                }
            }
        }));
        Arc::new(MipsCpu::new(exec))
    }

    pub fn start(&mut self) {
        // Publish "starting" before the monitor binds, so a caller polling
        // readiness can tell "not started" from "startup under way".
        if let Ok(starting) = self.lifecycle.transition(crate::machine_state::GuestState::Starting) {
            self.lifecycle = starting;
        }
        self.running.store(true, Ordering::SeqCst);
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        let d = self.display.clone();
        self.threads.push(std::thread::Builder::new().name("O2-Display".into()).spawn(move || d.display_loop()).unwrap());
        let bus = self.bus.clone();
        let cpu = self.cpu.clone();
        let running = self.running.clone();
        let events = (self.events.clone(), self.controller_up.clone());
        let paused = self.paused.clone();
        self.start_scsi();
        self.threads.push(
            std::thread::Builder::new()
                .name("O2-Service".into())
                .spawn(move || Self::service_loop(bus, cpu, running, paused, events))
                .unwrap(),
        );
        // CI mode: the harness starts the CPU (`iris-ci start`).
        if self.ci_serial.is_none() {
            self.cpu.clone().as_device().start();
        }
    }

    /// Device interrupts the guest did not cause by an access (network
    /// frames, keys), typing scripts, and the guest's power switch.
    fn service_loop(
        bus: Arc<Ip32Bus>,
        cpu: Arc<dyn CpuDevice>,
        running: Arc<AtomicBool>,
        paused: Arc<AtomicBool>,
        (events, controller_up): (std::sync::mpsc::SyncSender<MachineEvent>, Arc<AtomicBool>),
    ) {
        let dev = cpu.clone().as_device();
        // The PROM's environment lives in its flash: save it once the guest
        // has written there (setenv) and the writing has stopped.
        let mut flash_seen = bus.flash_writes();
        let mut flash_check = Instant::now();
        while running.load(Ordering::Relaxed) {
            if paused.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            if flash_check.elapsed() >= Duration::from_secs(2) {
                flash_check = Instant::now();
                let n = bus.flash_writes();
                if n != flash_seen {
                    flash_seen = n;
                    bus.save_flash();
                }
            }
            // Frames from the network, and any line that changed without a
            // register access (a disk finishing, a key from the window).
            bus.service();
            // A cold reset through CRIME (NetBSD's reboot): the CPU starts
            // again at the reset vector with every device as at power-on.
            if bus.crime.regs.take_reset_request() {
                eprintln!("iris: the O2 reset itself (CRIME cold reset)");
                dev.stop();
                crate::traits::Resettable::power_on(&*cpu);
                bus.power_on();
                dev.start();
            }
            if bus.mace.rtc.powered_off() {
                eprintln!("iris: the O2 switched itself off");
                dev.stop();
                bus.save_flash();
                // The machine-events thread stops the machine and exits,
                // unless the host embeds iris (IRIS_NO_EXIT_ON_POWEROFF).
                if controller_up.load(Ordering::SeqCst) && events.try_send(MachineEvent::PowerOff).is_ok() {
                    return;
                }
                running.store(false, Ordering::SeqCst);
                std::process::exit(0);
            }
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// Stop the CPU, hold the service thread off the devices, let the
    /// rendering engine finish what it was given, and save the PROM's flash.
    /// The display thread and the monitor run on until the machine is
    /// dropped; [`Self::restart_peripherals`] lets the service thread go.
    pub fn stop(&mut self) {
        self.cpu.clone().as_device().stop();
        self.paused.store(true, Ordering::SeqCst);
        // After the CPU, so nothing queues more; each finishes the command
        // it is running, so a snapshot finds none half done.
        for ctl in self.bus.scsi.lock().unwrap().iter() {
            ctl.stop();
        }
        if !self.bus.crime.re.drain(Duration::from_secs(2)) {
            eprintln!("iris: O2 stopped with rendering engine writes still queued");
        }
        self.bus.save_flash();
        // A machine that was never started has nothing to halt.
        if self.lifecycle != crate::machine_state::GuestState::Idle {
            self.lifecycle = crate::machine_state::GuestState::Halted;
        }
    }

    /// Let the service thread look after the devices again.
    pub fn restart_peripherals(&mut self) {
        self.paused.store(false, Ordering::SeqCst);
        self.start_scsi();
    }

    /// Give each SCSI controller its own thread, so disk transfers do not
    /// run on the CPU's.
    fn start_scsi(&self) {
        for ctl in self.bus.scsi.lock().unwrap().iter() {
            ctl.start();
        }
    }

    /// The hardware as at power-on, with the CPU stopped: what a CRIME cold
    /// reset does, less the restart.
    pub fn reset(&mut self) {
        self.stop();
        crate::traits::Resettable::power_on(&*self.cpu);
        self.bus.power_on();
        self.restart_peripherals();
    }

    /// Save the machine to `saves/<name>/`, then run on, as the Indy does.
    /// The format is the Indy's: device state, memory in the shared chunk
    /// store (all of it as bank 0), the copy-on-write disks' changes; plus
    /// the flash and the batch hook's place in the boot. Disks without
    /// `overlay` are not in it: restoring then pairs saved memory with
    /// whatever the disk holds by then.
    pub fn save_snapshot(&mut self, name: &str) -> Result<(), String> {
        self.stop();
        let saved = self.write_snapshot(name);
        self.restart_peripherals();
        self.cpu.clone().as_device().start();
        saved?;
        println!("Snapshot saved to saves/{name}");
        Ok(())
    }

    fn write_snapshot(&self, name: &str) -> Result<(), String> {
        use crate::chunk_store::{put_words_as_chunks, ChunkStore};
        use crate::snapshot::{ChunksManifest, Manifest, Snapshot};
        use crate::traits::Saveable;
        let snap = Snapshot::new(std::path::PathBuf::from("saves").join(name));
        snap.ensure_dir().map_err(|e| e.to_string())?;
        let mut manifest = Manifest::for_current_save();
        manifest.machine = Some(SNAPSHOT_MACHINE.into());
        manifest.disks = self.disks.clone();
        manifest.nvram = self.flash_path.clone();
        manifest.cpu_model = Some(self.cpu.model_name().to_string());
        manifest.cpu_cache = Some(self.cpu.cache_kind().to_string());
        snap.write_manifest(&manifest).map_err(|e| e.to_string())?;
        let sv = manifest.schema_version;
        snap.write_state("cpu", &self.cpu.save_state(), sv).map_err(|e| e.to_string())?;
        snap.write_state("o2", &self.bus.save_state(), sv).map_err(|e| e.to_string())?;
        snap.write_state("boot", &self.boot.save(), sv).map_err(|e| e.to_string())?;
        snap.write_bin("flash.bin", &self.bus.flash_bytes()).map_err(|e| e.to_string())?;
        let store = ChunkStore::new("saves");
        let mut chunks = ChunksManifest::default();
        chunks.bank_chunks[0] = put_words_as_chunks(&store, self.bus.ram.words())
            .map_err(|e| format!("CAS memory put: {e}"))?;
        snap.write_chunks_manifest(&chunks).map_err(|e| e.to_string())?;
        for (i, ctl) in self.bus.scsi.lock().unwrap().iter().enumerate() {
            let dir = snap.dir.join(format!("ctrl{i}"));
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let overlays = ctl.export_overlays(&dir).map_err(|e| format!("COW overlay export (controller {i}): {e}"))?;
            snap.write_toml(&format!("cow{i}.toml"), &crate::machine::build_cow_table(overlays))
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Restore the machine from `saves/<name>/` and run it. Everything is
    /// read and checked before the machine is touched, so a snapshot that
    /// does not fit (another machine, CPU, build, memory or disk) leaves it
    /// as it was.
    pub fn load_snapshot(&mut self, name: &str) -> Result<(), String> {
        let snap = self.read_snapshot(name)?;
        self.stop();
        let applied = self.apply_snapshot(snap);
        self.restart_peripherals();
        applied?;
        self.cpu.clone().as_device().start();
        println!("Snapshot loaded from saves/{name}");
        Ok(())
    }

    fn read_snapshot(&self, name: &str) -> Result<O2Snapshot, String> {
        use crate::chunk_store::{get_chunks_as_words, ChunkStore};
        use crate::snapshot::{Snapshot, SCHEMA_VERSION};
        let snap = Snapshot::new(std::path::PathBuf::from("saves").join(name));
        let m = snap.read_manifest()?.ok_or_else(|| format!("saves/{name}: no snapshot.toml"))?;
        if m.machine.as_deref() != Some(SNAPSHOT_MACHINE) {
            return Err(format!("snapshot {name} is not of an O2"));
        }
        if m.host_arch != std::env::consts::ARCH {
            return Err(format!("snapshot host_arch '{}' does not match current host '{}'", m.host_arch, std::env::consts::ARCH));
        }
        if m.schema_version > SCHEMA_VERSION {
            return Err(format!("snapshot schema_version {} is newer than this iris build supports ({SCHEMA_VERSION})", m.schema_version));
        }
        // As the Indy: build, CPU and disks must match;
        // IRIS_SNAPSHOT_SKIP_CHECK=1 makes those warnings.
        let mut fatal = Vec::new();
        let features: Vec<String> = m.features.iter()
            .filter(|f| !crate::snapshot::RETIRED_FEATURES.contains(&f.as_str())).cloned().collect();
        if features != crate::snapshot::enabled_features() {
            fatal.push(format!("build features differ: snapshot [{}] vs current [{}]",
                m.features.join(","), crate::snapshot::enabled_features().join(",")));
        }
        if let Some(e) = crate::snapshot::cpu_model_mismatch(m.cpu_model.as_deref(), self.cpu.model_name()) {
            fatal.push(e);
        }
        if let Some(e) = crate::snapshot::cache_model_mismatch(m.cpu_cache.as_deref(), m.cpu_model.as_deref(), self.cpu.cache_kind()) {
            fatal.push(e);
        }
        for d in &m.disks {
            match self.disks.iter().find(|c| c.id == d.id) {
                None => fatal.push(format!("snapshot disk {}:{} ('{}') is not configured in this run", d.id / 16, d.id % 16, d.path)),
                Some(c) if c.size_bytes != d.size_bytes => fatal.push(format!(
                    "disk {}:{} ('{}') size differs: snapshot {} bytes vs current {}", d.id / 16, d.id % 16, d.path, d.size_bytes, c.size_bytes)),
                Some(_) => {}
            }
        }
        if !fatal.is_empty() {
            let msg = format!("snapshot provenance mismatch:\n  - {}", fatal.join("\n  - "));
            if std::env::var("IRIS_SNAPSHOT_SKIP_CHECK").is_ok_and(|v| !v.is_empty() && v != "0") {
                eprintln!("load_snapshot: {msg} [IRIS_SNAPSHOT_SKIP_CHECK set; continuing]");
            } else {
                return Err(format!("{msg}\n(set IRIS_SNAPSHOT_SKIP_CHECK=1 to override)"));
            }
        }
        let sv = m.schema_version;
        let state = |base: &str| snap.read_state(base, sv).map_err(|e| format!("{base}: {e}"));
        let chunks = snap.read_chunks_manifest().map_err(|e| format!("chunks.bin: {e}"))?;
        let memory = get_chunks_as_words(&ChunkStore::new("saves"), &chunks.bank_chunks[0])
            .map_err(|e| format!("CAS memory get: {e}"))?;
        if memory.len() * 4 != self.bus.ram.len() {
            return Err(format!("snapshot memory is {} MB, this O2 has {} MB",
                memory.len() >> 18, self.bus.ram.len() >> 20));
        }
        let ctrls = self.bus.scsi.lock().unwrap().len();
        let overlays = (0..ctrls)
            .map(|i| snap.read_toml(&format!("cow{i}.toml")).ok().map(|t| crate::machine::parse_cow_table(&t)))
            .collect();
        Ok(O2Snapshot {
            cpu: state("cpu")?,
            bus: state("o2")?,
            boot: state("boot")?,
            flash: snap.read_bin("flash.bin").map_err(|e| format!("flash.bin: {e}"))?,
            memory,
            overlays,
            dir: snap.dir,
        })
    }

    /// Put a read snapshot into the stopped machine: the hardware as at
    /// power-on first, so nothing the snapshot does not hold is left over.
    fn apply_snapshot(&mut self, s: O2Snapshot) -> Result<(), String> {
        use crate::traits::{Resettable, Saveable};
        Resettable::power_on(&*self.cpu);
        self.bus.power_on();
        self.cpu.load_state(&s.cpu)?;
        self.bus.load_state(&s.bus)?;
        self.boot.load(&s.boot);
        self.bus.set_flash(&s.flash)?;
        self.bus.ram.restore_words(&s.memory)?;
        for (i, ctl) in self.bus.scsi.lock().unwrap().iter().enumerate() {
            if let Some(Some(sets)) = s.overlays.get(i) {
                ctl.import_overlays(&s.dir.join(format!("ctrl{i}")), sets)
                    .map_err(|e| format!("COW overlay import (controller {i}): {e}"))?;
            }
        }
        Ok(())
    }

    /// The shared slot (see [`MachineSlot`]) for this machine, created on
    /// first use. Only once the machine is boxed: the slot records this
    /// address.
    pub fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut O2Machine as *mut dyn Workstation;
        let slot = self.controller_slot.get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr))));
        debug_assert!(std::ptr::addr_eq(slot.lock().0, ptr), "O2Machine moved after its slot was handed out");
        slot.clone()
    }

    pub fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> {
        match self.bus.mace.ps2.devices() {
            Some(d) => d.clone(),
            // Serial console: a keyboard nothing reads.
            None => Arc::new(crate::dev::ps2::Ps2Controller::new(None)),
        }
    }

    pub fn get_display(&self) -> Arc<dyn crate::gfx_display::GfxDisplay> {
        self.display.clone()
    }

    pub fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timer_manager.clone()
    }
}

/// A snapshot read and checked, ready to put into the machine.
struct O2Snapshot {
    cpu: toml::Value,
    bus: toml::Value,
    boot: toml::Value,
    flash: Vec<u8>,
    memory: Vec<u32>,
    /// Each controller's copy-on-write sets, when the snapshot has them.
    overlays: Vec<Option<Vec<(usize, Vec<u64>)>>>,
    dir: std::path::PathBuf,
}

impl Drop for O2Machine {
    fn drop(&mut self) {
        // First wait out any monitor, event or CI command in flight, so none
        // sees a half-destroyed machine (see `Drop for Machine`).
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

/// The O2 as the front ends see it. The CI rollback and `--load-elf` are not
/// there yet; the trait's defaults say so.
impl Workstation for O2Machine {
    fn start(&mut self) { O2Machine::start(self) }
    fn stop(&mut self) { O2Machine::stop(self) }
    fn reset(&mut self) { O2Machine::reset(self) }
    fn restart_peripherals(&mut self) { O2Machine::restart_peripherals(self) }
    fn cpu_start(&self) { self.cpu.clone().as_device().start() }
    fn cpu_is_running(&self) -> bool { self.cpu.clone().as_device().is_running() }
    fn machine_slot(&mut self) -> MachineSlot { O2Machine::machine_slot(self) }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self.event_rx.take();
        assert!(events.is_some(), "register_system_controller called twice");
        crate::workstation::register_system_controller(&self.monitor, slot, events);
        self.controller_up.store(true, Ordering::SeqCst);
    }
    fn monitor_addr(&self) -> Option<std::net::SocketAddr> { self.monitor.bound_addr() }
    fn lifecycle(&self) -> crate::machine_state::GuestState { self.lifecycle }
    fn set_lifecycle(&mut self, state: crate::machine_state::GuestState) { self.lifecycle = state; }

    fn get_display(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> { Some(O2Machine::get_display(self)) }
    fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> { O2Machine::get_ps2(self) }
    fn get_timer_manager(&self) -> Arc<TimerManager> { O2Machine::get_timer_manager(self) }
    fn get_cpu_debug(&self) -> Arc<dyn crate::gdb_stub::CpuDebug> { self.cpu.clone().debug_adapter() }
    fn fasttick_count(&self) -> Arc<std::sync::atomic::AtomicU64> { self.cpu.fasttick_count() }
    fn get_ci_serial(&self) -> Option<Arc<crate::dev::z85c30::CiSerialBackend>> { self.ci_serial.clone() }
    /// Typed at serial port 1, the console with `graphics = false`.
    fn inject_serial_console(&self, bytes: &[u8]) {
        match &self.ci_serial {
            Some(ci) => ci.push_host(bytes),
            None => self.bus.mace.com0.feed(bytes),
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn save_snapshot(&mut self, name: &str) -> Result<(), String> { O2Machine::save_snapshot(self, name) }
    fn load_snapshot(&mut self, name: &str) -> Result<(), String> { O2Machine::load_snapshot(self, name) }

    fn scratch_path(&self) -> Option<&std::path::Path> { self.scratch_path.as_deref() }
    /// Disc changes on the internal bus (controller 0), as the Indigo2 offers
    /// its first controller.
    fn disc_changer(&self) -> Option<Arc<dyn crate::workstation::DiscChanger>> {
        self.bus.scsi.lock().unwrap().first().map(|c| c.clone() as Arc<dyn crate::workstation::DiscChanger>)
    }

    fn nat_control(&self) -> Option<Arc<crate::net::NatControl>> {
        self.nat.as_ref().map(|(n, _)| n.clone())
    }
    fn nat_expected(&self) -> Option<(std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr)> {
        self.nat.as_ref().map(|(_, a)| *a)
    }
}

/// GBE's picture, scanned out of RAM about sixty times a second and handed
/// to whatever renderer the front-end installs.
pub struct O2Display {
    bus: Arc<Ip32Bus>,
    renderer: Mutex<Option<Box<dyn Renderer>>>,
    screen: Mutex<Rex3Screen>,
    screenshot_pending: AtomicBool,
    cycles: CyclesPtr,
    /// The CPU's Compare-match count: IRIX's clock tick on the O2, which has
    /// no 8254. The status bar's Hz.
    ticks: Arc<std::sync::atomic::AtomicU64>,
    running: Arc<AtomicBool>,
}

impl O2Display {
    fn new(bus: Arc<Ip32Bus>, cycles: CyclesPtr, ticks: Arc<std::sync::atomic::AtomicU64>, running: Arc<AtomicBool>) -> Self {
        Self {
            bus,
            renderer: Mutex::new(None),
            screen: Mutex::new(Rex3Screen::new()),
            screenshot_pending: AtomicBool::new(false),
            cycles,
            ticks,
            running,
        }
    }

    fn display_loop(&self) {
        let frame_time = Duration::from_micros(16_667);
        let mut status_bar = crate::disp::StatusBar::new();
        let mut overlay = crate::debug_overlay::DebugOverlay::new();
        let mut sbtex = crate::disp::StatusBarTexture::new();
        let mut rgba = vec![0u8; 2048 * 2048 * 4];
        let mut size = (0usize, 0usize);
        let mut shots = 0u32;
        while self.running.load(Ordering::Relaxed) {
            let start = Instant::now();
            let shown = self.bus.render_screen(&mut rgba);
            // Off: keep the last size, black.
            let (w, h) = shown.map(|(w, h)| (w as usize, h as usize)).unwrap_or(if size.0 == 0 { (1280, 1024) } else { size });
            {
                let mut screen = self.screen.lock();
                let stride = crate::disp::FB_STRIDE;
                for y in 0..h {
                    for x in 0..w {
                        let v = if shown.is_some() {
                            let o = (y * w + x) * 4;
                            u32::from_le_bytes([rgba[o], rgba[o + 1], rgba[o + 2], 0xff])
                        } else {
                            0xff00_0000
                        };
                        screen.fb_rgb[y * stride + x] = v;
                        screen.rgba[y * stride + x] = v;
                    }
                }
                screen.prebuilt = true;
                screen.width = w;
                screen.height = h;
                screen.status_bar_only = false;
                let take_shot = self.screenshot_pending.swap(false, Ordering::Relaxed);
                let stats = crate::disp::BarStats {
                    now: start,
                    hb: 0,
                    cycles: self.cycles.get(),
                    fasttick: self.ticks.load(Ordering::Relaxed),
                    decoded_delta: 0,
                    l1i_hits: 0,
                    l1i_fetches: 0,
                    uncached: 0,
                    count_hz: 0,
                    gfifo_pending: 0,
                };
                if let Some(r) = self.renderer.lock().as_mut() {
                    if (w, h) != size {
                        r.resize(w, h);
                    }
                    r.present(&mut screen, &mut overlay, &mut status_bar, &mut sbtex, &stats, take_shot, None, None);
                }
                size = (w, h);
                if take_shot {
                    let pixels = screen.rgba.clone();
                    let path = format!("screenshot_{shots:04}.png");
                    shots += 1;
                    std::thread::spawn(move || match crate::disp::save_screenshot(&path, &pixels, w, h) {
                        Ok(()) => println!("iris: screenshot saved to {path}"),
                        Err(e) => println!("iris: screenshot failed: {e}"),
                    });
                }
            }
            let spent = start.elapsed();
            if spent < frame_time {
                std::thread::sleep(frame_time - spent);
            }
        }
        if let Some(r) = self.renderer.lock().as_mut() {
            r.stop();
        }
    }
}

impl crate::gfx_display::GfxDisplay for O2Display {
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> {
        &self.renderer
    }
    fn screen(&self) -> &Mutex<Rex3Screen> {
        &self.screen
    }
    fn request_screenshot(&self) {
        self.screenshot_pending.store(true, Ordering::Relaxed);
    }
    fn cycles(&self) -> CyclesPtr {
        self.cycles.clone()
    }
}

/// The control port: one command per line, one reply line each (`ok ...` or
/// `err ...`). For scripts and for an assistant driving the machine while a
/// person watches its window.
///
/// - `shot PATH`: the screen as a PNG.
/// - `type TEXT`: type it on the PS/2 keyboard (`\r` Enter, `\e` Escape,
///   `\b` Backspace, `\t` Tab).
/// - `key NAME`, `keydown NAME`, `keyup NAME`: a key by its winit name
///   (`Enter`, `KeyA`, `F1`, `ShiftLeft`, `ArrowUp`...).
/// - `mouse DX DY [BUTTONS]`: move by DX, DY (screen pixels, y down) with
///   BUTTONS held (1 left, 2 right, 4 middle).
/// - `click [BUTTON]`: press and release (default 1, left).
/// - `status`: what the display is showing, and the GBE registers behind it.
/// Where the CPU was at the end of its last batch, and how many batches it
/// has run: enough to tell a guest spinning in one place from one that is
/// still moving, without stopping it.
pub(crate) static LAST_PC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The general registers at the same moment, then Status, Cause, EPC and
/// BadVAddr: what a polling loop is polling.
static LAST_REGS: [std::sync::atomic::AtomicU64; 42] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 42];

/// The control commands on the shared monitor, as `o2 <command> [args]`:
/// the same commands, and replies, as the control port's.
struct O2Control {
    bus: Arc<Ip32Bus>,
    rgba: Mutex<Vec<u8>>,
}

impl O2Control {
    fn new(bus: Arc<Ip32Bus>) -> Self {
        Self { bus, rgba: Mutex::new(vec![0u8; 2048 * 2048 * 4]) }
    }
}

impl Device for O2Control {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("o2".to_string(), "O2 control: o2 shot PATH | status | pixel X Y | type TEXT | key|keydown|keyup NAME | \
            mouse DX DY [BUTTONS] | click [BUTTON] | pc | regs | peek ADDR LEN PATH | \
            retrace PATH | scsi PATH | unmapped | ps2log PATH | crimelog PATH | nictrace PATH".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "o2" {
            return Err("Command not found".to_string());
        }
        let Some((sub, rest)) = args.split_first() else {
            return Err("Usage: o2 <command> [args] (help o2)".to_string());
        };
        let reply = control_command(&self.bus, sub, &rest.join(" "), &mut self.rgba.lock());
        writeln!(w, "{reply}").map_err(|e| e.to_string())
    }
}

fn control_server(listener: std::net::TcpListener, bus: Arc<Ip32Bus>) {
    use std::io::{BufRead, BufReader, Write};
    for conn in listener.incoming().flatten() {
        let bus = bus.clone();
        let _ = std::thread::Builder::new().name("O2-Control-Client".into()).spawn(move || {
            let mut out = match conn.try_clone() {
                Ok(o) => o,
                Err(_) => return,
            };
            let mut rgba = vec![0u8; 2048 * 2048 * 4];
            for line in BufReader::new(conn).lines().map_while(Result::ok) {
                let (cmd, arg) = line.trim_end().split_once(' ').unwrap_or((line.trim_end(), ""));
                let reply = control_command(&bus, cmd, arg, &mut rgba);
                if writeln!(out, "{reply}").is_err() {
                    break;
                }
            }
        });
    }
}

fn control_command(bus: &Ip32Bus, cmd: &str, arg: &str, rgba: &mut [u8]) -> String {
    let kb = bus.mace.ps2.devices();
    match cmd {
        "shot" => match bus.render_screen(rgba) {
            Some((w, h)) => match crate::disp::save_png_rgba8(arg, &rgba[..(w * h * 4) as usize], w, h) {
                Ok(()) => format!("ok {w}x{h}"),
                Err(e) => format!("err {e}"),
            },
            None => "err display off".into(),
        },
        "pc" => format!(
            "ok pc 0x{:08x} after {} batches",
            LAST_PC.load(Ordering::Relaxed),
            BATCHES.load(Ordering::Relaxed)
        ),
        // `crimelog PATH`: the guest's recent CRIME interrupt-register
        // accesses, oldest first.
        "nictrace" => {
            let mut t = bus.mace.nic_trace.text();
            for l in bus.mace.onewire.protocol_log() {
                t.push_str(&format!("# {l}\n"));
            }
            match std::fs::write(arg.trim(), t) {
                Ok(()) => "ok".into(),
                Err(e) => format!("err {e}"),
            }
        }
        "crimelog" => match std::fs::write(arg.trim(), bus.crime.regs.irqlog_text()) {
            Ok(()) => "ok".into(),
            Err(e) => format!("err {e}"),
        },
        "regs" => {
            let r: Vec<u64> = LAST_REGS.iter().map(|v| v.load(Ordering::Relaxed)).collect();
            let mut s = format!("ok pc 0x{:08x}", LAST_PC.load(Ordering::Relaxed));
            for (i, v) in r[..32].iter().enumerate() {
                s.push_str(&format!(" r{i}=0x{v:x}"));
            }
            s.push_str(&format!(" sr=0x{:x} cause=0x{:x} epc=0x{:x} badva=0x{:x}", r[32], r[33], r[34], r[35]));
            s.push_str(&format!(" count=0x{:x} compare=0x{:x} irq=0x{:x} ticks={} cycles={} config=0x{:x}", r[36], r[37], r[38], r[39], r[40], r[41]));
            s
        }
        // `peek ADDR LEN PATH`: LEN bytes of the physical address space from
        // ADDR (hex), read as the CPU would, written to PATH as raw big-endian
        // bytes.
        "peek" => {
            let mut it = arg.split_whitespace();
            let addr = it.next().and_then(|a| u32::from_str_radix(a.trim_start_matches("0x"), 16).ok());
            let len = it.next().and_then(|l| l.parse::<u32>().ok());
            match (addr, len, it.next()) {
                (Some(a), Some(n), Some(path)) => {
                    let mut out = Vec::with_capacity(n as usize);
                    for i in (0..n.min(1 << 20)).step_by(4) {
                        out.extend_from_slice(&bus.read32(a.wrapping_add(i) as u64).data.to_be_bytes());
                    }
                    match std::fs::write(path, out) {
                        Ok(()) => "ok".into(),
                        Err(e) => format!("err {e}"),
                    }
                }
                _ => "err usage: peek ADDR LEN PATH".into(),
            }
        }
        // `ps2log PATH`: the PS/2 ports' recent traffic, `1>f4` a byte to
        // port 1's device, `1<fa` one from it.
        "ps2log" => match std::fs::write(arg, bus.mace.ps2.traffic_timed()) {
            Ok(()) => "ok".into(),
            Err(e) => format!("err {e}"),
        },
        "retrace" => match std::fs::write(arg, bus.crime.re.trace_text()) {
            Ok(()) => "ok".into(),
            Err(e) => format!("err {e}"),
        },
        "scsi" => {
            let mut text = String::new();
            for (n, c) in bus.scsi.lock().unwrap().iter().enumerate() {
                for line in c.notes() {
                    text.push_str(&format!("{n}: {line}\n"));
                }
            }
            match std::fs::write(arg, text) {
                Ok(()) => "ok".into(),
                Err(e) => format!("err {e}"),
            }
        }
        "unmapped" => {
            // Bus accesses that hit nothing (the first 64 distinct ones):
            // what a guest saw as a bus error, with the PC that made it.
            let u = bus.unmapped();
            let list: Vec<String> = u.iter().map(|(a, w, pc)| format!("{} {a:#010x} pc {pc:#010x}", if *w { "w" } else { "r" })).collect();
            format!("ok {} {}", u.len(), list.join("; "))
        }
        "pixel" => {
            // pixel X Y: what scan-out makes of one point (frame-buffer word,
            // DID, mode-table entry, colour map, overlay).
            let v: Vec<usize> = arg.split_whitespace().filter_map(|t| t.parse().ok()).collect();
            if v.len() != 2 {
                return "err pixel X Y".into();
            }
            *bus.gbe.probe.lock().unwrap() = (Some((v[0], v[1])), "not on screen".into());
            bus.render_screen(rgba);
            let mut p = bus.gbe.probe.lock().unwrap();
            p.0 = None;
            format!("ok {}", p.1)
        }
        "status" => {
            let shown = bus.render_screen(rgba);
            format!("ok display {shown:?}; {}", bus.gbe.summary())
        }
        _ if kb.is_none() => "err no keyboard or mouse (graphics = false)".into(),
        "type" => {
            let kb = kb.unwrap();
            let text = arg.replace("\\r", "\r").replace("\\e", "\x1b").replace("\\b", "\x08").replace("\\t", "\t");
            let mut n = 0;
            for c in text.chars() {
                if c == '\t' {
                    kb.push_kb(winit::keyboard::KeyCode::Tab, true);
                    kb.push_kb(winit::keyboard::KeyCode::Tab, false);
                    n += 1;
                } else if kb.type_char(c) {
                    n += 1;
                }
            }
            format!("ok {n}")
        }
        "key" | "keydown" | "keyup" => match crate::ip32_screen::keycode(arg.trim()) {
            Some(k) => {
                let kb = kb.unwrap();
                if cmd != "keyup" {
                    kb.push_kb(k, true);
                }
                if cmd != "keydown" {
                    kb.push_kb(k, false);
                }
                "ok".into()
            }
            None => format!("err no key {arg:?}"),
        },
        "mouse" => {
            let v: Vec<i32> = arg.split_whitespace().filter_map(|x| x.parse().ok()).collect();
            if v.len() < 2 {
                return "err mouse DX DY [BUTTONS]".into();
            }
            let buttons = v.get(2).copied().unwrap_or(0) as u8;
            let kb = kb.unwrap();
            // A PS/2 packet carries at most 255 either way: send big moves in steps.
            let (mut dx, mut dy) = (v[0], v[1]);
            loop {
                let (sx, sy) = (dx.clamp(-200, 200), dy.clamp(-200, 200));
                kb.push_mouse_input(buttons, sx, sy, 0);
                dx -= sx;
                dy -= sy;
                if dx == 0 && dy == 0 {
                    break;
                }
            }
            "ok".into()
        }
        "click" => {
            let b: u8 = arg.trim().parse().unwrap_or(1);
            let kb = kb.unwrap();
            kb.push_mouse_input(b, 0, 0, 0);
            kb.push_mouse_input(0, 0, 0, 0);
            "ok".into()
        }
        _ => format!("err unknown command {cmd:?}"),
    }
}

