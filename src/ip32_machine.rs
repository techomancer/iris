//! The O2 (IP32) as a machine: iris's CPU thread and window around the IP32
//! bus and devices (`ip32.rs`).
//!
//! Selected with `[machine] profile = "o2_ip32"`, the shared configuration
//! keys and O2-specific `[o2]` options ([`crate::config::O2Section`]). Between
//! instructions, two things run:
//!
//! - a service thread keeps the devices' interrupts current in CRIME and
//!   CRIME's in the CPU's IP2, runs the MACE clock from the host clock, and
//!   ends the run when the guest switches the machine off;
//! - a batch hook on the CPU thread lends the firmware the FPU, when
//!   `arcs_fpu` asks for it, with where the guest is running.
//!
//! R5000, R10000 and R12000 all run on the shadow cache from reset: the JIT reaches RAM through
//! the ppmem window with no tag probe, LRU or dirty bit on any load or store.
//! The stack POST keeps in cache tags is [`O2Memory`]'s business.
//!
//! The screen is GBE scanned out of RAM by [`O2Display`], presented through
//! the same renderer the other machines use; keys from the window go to the
//! MACE PS/2 keyboard.

use std::sync::atomic::{AtomicBool, Ordering};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::parts::{self, Part};
use crate::cpu::mips_core::CyclesPtr;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::crime_mem::O2Memory;
use crate::dev::prom_image;
use crate::dev::ng1::rex3::Renderer;
use crate::disp::Rex3Screen;
use crate::hptimer::TimerManager;
use crate::ip32::{Ip32Bus, Ip32Options, FIRMWARE_LOAD_VA, FIRMWARE_SPAN, NO_ECC_BASE, PCI_MEM_64_BASE, PCI_MEM_64_SIZE, PROM_SIZE, RAM_BASE, RAM_WINDOW_SIZE};
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
    /// Where this machine has memory.
    pub memory: Arc<crate::memory_map::MemoryMap>,
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
    ci_serial: Option<Arc<crate::dev::serial_host::CiSerialBackend>>,
    /// The NAT's controls and the addresses it hands the guest (client,
    /// gateway, netmask), when the Ethernet is plugged in.
    nat: Option<(Arc<crate::net::NatControl>, (std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr))>,
    /// Guest frames the network backend has processed.
    frames: Option<Arc<std::sync::atomic::AtomicU64>>,
    /// The guest switching the machine off, for the machine-events thread.
    events: crate::traits::MachineEvents,
    event_rx: Option<crate::traits::MachineEventQueue>,
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
    /// The last snapshot loaded, for `parent` and `rollback`.
    restored: crate::snapshot::Restored,
}

/// What an O2 snapshot's manifest names as its machine.
const SNAPSHOT_MACHINE: &str = "o2_ip32";

/// The PROM image tried when `prom` is left at the shared default: the last
/// release's, named as SGI shipped it. The 4.18 image is also built in.
pub const O2_PROM_FILE: &str = "ip32prom.rev4.18.bin";

/// The processors an O2 shipped with, the default first.
pub const CPUS: &[&Part] = &[&parts::R5000_180, &parts::R10000_250, &parts::R12000_400];

/// The O2's slots.
pub(crate) fn slots(s: &mut Vec<crate::topology::MachineSlot>) {
    crate::topology::pci_bus_slots(s, "pci0", "O2 system board", crate::ip32::PCI0_SLOTS);
}


impl O2Machine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        let part = MachineProfile::O2Ip32.part(cfg.machine.cpu.as_deref())?;
        let o2 = &cfg.o2;
        // `prom`, else the last release's file name, else the PROM built into
        // iris.
        let hint = "set `prom` to the O2 PROM image; 4.18 is the last release";
        let size = PROM_SIZE as usize;
        let loaded = prom_image::resolve(MachineProfile::O2Ip32, &cfg.prom, size).map_err(|e| format!("O2 {e} ({hint})"))?;
        let name = loaded.name();
        let mut prom = loaded.bytes;
        // A raw file is the whole flash. A shorter payload came out of the
        // installer's header, and the rest of the part reads erased.
        let file_len = loaded.source.as_ref().map_or(0, |p| std::fs::metadata(p).map_or(0, |m| m.len()));
        if prom.len() != size && prom.len() as u64 == file_len {
            return Err(format!("O2 PROM {}: {file_len} bytes is not a whole {size}-byte flash ({hint})", name));
        }
        prom.resize(size, 0xff);
        // `banks`: the kit in each DIMM pair from A (validate checked them).
        let kits = cfg.o2_kits();
        let mem = O2Memory::from_kits(&kits)?;
        // The PROM's flash, and so its environment, lives in `nvram`.
        // `[network] mac` is the address in the 1-Wire ID chip. `headless`:
        // no keyboard, so the PROM and the OS take the serial console, as on
        // an O2 booted without one. The GBE stays: it is part of every O2.
        let mac = match cfg.network.mac.as_deref() {
            Some(mac) => Some(crate::config::parse_mac(mac).map_err(|e| format!("network.mac: {e}"))?),
            None => None,
        };
        // The SCSI devices: scsi0 is the internal bus, scsi1 the external
        // one. The buses are filled before the controllers exist.
        let scsi = [crate::dev::aic7880::Aic7880::new_bus(), crate::dev::aic7880::Aic7880::new_bus()];
        let mut resolved = crate::topology::DevicePlacement::new(cfg)?;
        let attached = crate::scsi::attach_from_config(cfg, Some("ip32"), &resolved, &[&scsi[0], &scsi[1]])?;
        let opts = Ip32Options {
            scsi: Some(scsi),
            flash_file: Some(cfg.nvram.clone().into()),
            keyboard: (!cfg.headless).then_some(o2.mouse),
            mac,
            // Beside the flash, as the clock reference is beside its image.
            rtc_file: Some(format!("{}.rtc", cfg.nvram).into()),
            rtc_offset: cfg.rtc_offset,
        };
        let timer_manager = Arc::new(TimerManager::new());
        let bus = Ip32Bus::with_options(mem, prom, opts, &timer_manager);
        // MACE audio plays through the host unless audio is off.
        if !cfg.no_audio {
            bus.mace.audio.enable_output(cfg.audio.clone());
        }
        if let Some(d) = attached.daynaports.first() {
            return Err(format!("{}: kind = \"daynaport\" is not available on the O2", crate::config::scsi_label(d.controller, d.id)));
        }
        if cfg.has_device("qla2200") {
            return Err("[[devices]] qla2200: the O2 does not model a Fibre Channel adapter".into());
        }
        resolved.bind_pci("pci0", bus.mace.pci.bus().clone())?;
        let cards = resolved.build_cards(cfg, &timer_manager)?;
        let card_disks = cards.disk_refs();
        let _ = bus.cards.set(cards);
        let scratch_path = attached.scratch;
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
        let ci_serial = cfg.ci.then(|| Arc::new(crate::dev::serial_host::CiSerialBackend::new()));
        let (console, second) = crate::dev::serial_host::console_backends(cfg, "O2", ["MACE UART 0", "MACE UART 1"], ci_serial.as_ref())?;
        bus.mace.com0.set_backend(console);
        bus.mace.com1.set_backend(second);
        // The Ethernet on the NAT gateway `[network]` describes (`mode =
        // "off"` leaves it unplugged; validate refuses pcap here for now).
        let (mut nat, mut frames) = (None, None);
        let net = crate::net::GatewayConfig::from_network(&cfg.network());
        if let Some(link) = crate::net::Link::with_gateway(net.clone())? {
            if net.mode == crate::config::NetMode::Stream {
                eprintln!("iris: O2 Ethernet on {}", link.describe());
            } else {
                eprintln!("iris: O2 Ethernet on NAT: gateway {}, guest {} (DHCP)", net.gateway_ip, net.client_ip);
            }
            frames = Some(link.frames());
            nat = Some((link.nat(), (net.client_ip, net.gateway_ip, net.netmask)));
            bus.mace.mec.attach(link);
        }

        let sysad: Arc<dyn BusDevice> = bus.phys.clone();
        let arcs_fpu = o2.arcs_fpu;
        // The CPU clock the PROM measures and reports is twice Count.
        let count_hz = cfg.clock.count_hz(part);
        let cpu_mhz = (count_hz as f64 * 2.0 / 1e6).round().max(1.0) as u32;
        let boot = Arc::new(BootState::new());
        let memory = Arc::new(crate::ip32::memory_map());
        let cpu: Arc<dyn CpuDevice> = crate::with_cpu!(part, true, true, C =>
            Self::build_cpu::<C>(memory.clone(), sysad, &timer_manager, arcs_fpu, count_hz, boot.clone(), part.prid()));
        let world = Arc::new(crate::cpu::smp::World::new());
        world.set_slots(vec![cpu.park_slot()]);
        bus.ram.set_world(world.clone());
        // The CPU's pool flushes through this world too: a CPU belongs to one.
        #[cfg(feature = "jitv2")]
        cpu.share_jitv2(cpu.jitv2(), world.clone(), 0, 0);

        // ppmem: the CPU's cache and the JIT reach RAM through a cached view
        // of the memory's window, publishing its bitmap into the CPU's own
        // fields, as the Indigo2 does. Remaps happen on the CPU thread (a
        // CRIME bank control write), which is the window's contract.
        {
            let sp = bus.phys.ppmem_space().ok_or("ppmem: the O2 needs its memory window")?;
            let v = sp
                .add_cached_view(&[cpu.ppmem_bitmap_ptr(), cpu.tcache_bitmap_ptr()])
                .map_err(|e| format!("ppmem: the CPU's cached view: {e}"))?;
            unsafe {
                cpu.set_tcache_window(sp.view_base(v));
                cpu.set_cached_view(sp, v);
                sp.add_key_sink(v, cpu.tcache_key_ptr());
                #[cfg(feature = "jitv2")]
                {
                    use crate::ppmem::MappedMemory;
                    cpu.set_tcache_gen_window(sp.gen_window_base());
                }
            }
        }

        // After the windows above: each compile worker takes the cache's
        // geometry once, as it starts, and without the windows that geometry
        // has no inline load or store path.
        #[cfg(feature = "jitv2")]
        crate::cpu::jitv2::jitv2::attach_pool(&cpu, cfg.jitv2_pool_size(), cfg.jitv2.compile, bus.phys.clone(), world);

        // CRIME's output is the CPU's IP2, pushed the moment it changes.
        bus.connect_cpu_irq(crate::irq::IrqLine::new(Arc::new(CpuIrq(cpu.clone().as_device())), 2));

        // CRIME's rendering engine draws on its own thread, fed through
        // its interface buffer.
        bus.crime.re.start(bus.clone(), |b: &Ip32Bus| &b.crime.re);
        let running = Arc::new(AtomicBool::new(false));
        let status = crate::status::MachineStatus::new(bus.mace.id.heartbeat.clone(), cpu.fasttick_count());
        status.set_scsi_wide(true);
        status.set_cpu_cycles(cpu.cycles_ptr());
        status.set_count_hz(cpu.count_hz_atomic());
        // The O2's clock is CP0 Compare: publish the rate it programs.
        cpu.publish_tick_rate(status.cpu(0).tick_rate());
        let paused = Arc::new(AtomicBool::new(false));
        // GBE's timing interrupts come at their lines from the timer thread.
        status.refresh().set_timers(timer_manager.clone());
        let display = Arc::new(O2Display::new(bus.clone(), status.clone(), paused.clone()));
        let head: Arc<dyn crate::display::Head> = display.clone();
        display.head.attach(Arc::downgrade(&head));
        // The shared monitor, as the other machines have it: the CPU's
        // commands (registers, breakpoints, memory) and the O2's devices'.
        let monitor = crate::monitor::Monitor::new();
        monitor.register_device(Arc::new(crate::devlog::LogDevice));
        monitor.register_device(status.clone());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(vec![cpu.clone()])));
        monitor.register_device(bus.clone());
        let mace_pci = bus.mace.pci.clone();
        monitor.register_device(crate::pci::pci_host_commands("MACE", &[bus.mace.pci.bus()], Some(Box::new(move |w| mace_pci.write_status(w)))));
        // `phys`: the physical bus's decode table, memory and traces.
        monitor.register_device(bus.phys.clone());
        // The SCSI controllers' `scsi`/`scsi1` and `cow`/`cow1`.
        for ctl in bus.scsi.lock().unwrap().iter() {
            monitor.register_device(ctl.clone());
        }
        for c in bus.cards.get().into_iter().flat_map(|c| c.devices()) {
            monitor.register_device(c);
        }
        monitor.register_device(Arc::new(O2Control::new(bus.clone())));
        #[cfg(feature = "perfetto")]
        {
            crate::perfetto::set_guest_cycles(Some(cpu.cycles_ptr()));
            monitor.register_device(Arc::new(crate::perfetto::PerfettoMonitor));
        }
        let monitor_port = cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT);

        let (events, event_rx) = crate::traits::MachineEvents::channel(4);
        // A snapshot's disks by slot and size.
        let mut disks = crate::snapshot::DiskRef::from_tables(&attached.tables, false);
        disks.extend(card_disks);
        Ok(Self {
            memory,
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
            frames,
            events,
            event_rx: Some(event_rx),
            controller_up: Arc::new(AtomicBool::new(false)),
            controller_slot: None,
            boot,
            paused,
            lifecycle: crate::machine_state::GuestState::Idle,
            disks,
            flash_path: Some(cfg.nvram.clone()),
            scratch_path,
            restored: Default::default(),
        })
    }

    fn build_cpu<C>(memory: Arc<crate::memory_map::MemoryMap>, sysad: Arc<dyn BusDevice>, tm: &Arc<TimerManager>, arcs_fpu: bool, count_hz: u64, boot: Arc<BootState>,
                    prid: u32) -> Arc<dyn CpuDevice>
    where
        C: crate::cpu::mips_cache_v2::CpuModel + MipsCache + From<Arc<dyn BusDevice>> + 'static,
        MipsCpu<MipsTlb, C>: CpuDevice,
    {
        let cfg = MipsCpuConfig::for_model::<C>(memory).with_prid(prid);
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let mut exec: MipsExecutor<MipsTlb, C> = MipsExecutor::new(sysad, tlb, &cfg);
        exec.core.set_timer_manager(tm.clone());
        // Count runs at half the pipeline clock on both CPUs.
        exec.core.set_count_hz(count_hz.max(1));
        // The no-ECC alias of memory reaches memory, not the PCI window the
        // bus decodes at the same low 32 bits.
        exec.phys_alias = Some((NO_ECC_BASE, RAM_WINDOW_SIZE as u64, RAM_BASE as u64));
        // MACE's 64-bit PCI memory window is PCI, whatever its low 32 bits.
        exec.phys_alias_bypass = Some((PCI_MEM_64_BASE, PCI_MEM_64_SIZE));
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
            // Instructions retired, for an idle-MIPS reading from `o2 regs`.
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
            let fw = in_firmware(pc);
            if fw != boot.inside.load(Ordering::Relaxed) {
                boot.inside.store(fw, Ordering::Relaxed);
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
        self.timer_manager.clock().start();
        self.running.store(true, Ordering::SeqCst);
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.display.head.start();
        let bus = self.bus.clone();
        let cpu = self.cpu.clone();
        let running = self.running.clone();
        let events = (self.events.clone(), self.controller_up.clone());
        let paused = self.paused.clone();
        self.start_scsi();
        self.run_uarts(true);
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
        (events, controller_up): (crate::traits::MachineEvents, Arc<AtomicBool>),
    ) {
        let dev = cpu.clone().as_device();
        // The PROM's environment lives in its flash: save it once the guest
        // has written there (setenv) and the writing has stopped.
        let mut flash_seen = bus.flash_writes();
        let mut flash_check = Instant::now();
        // How long to sleep between service passes. While CRIME is
        // interrupting the machine is busy, so keep the tight 100 µs poll;
        // while it is quiet back off to 2 ms so an idle machine costs ~500
        // wakeups/s instead of 10 000. The backoff only adds latency to
        // asynchronous input (a typed key, a frame from the network) —
        // and while any of that is flowing, its interrupts keep the poll
        // at 100 µs anyway.
        const SERVICE_POLL: Duration = Duration::from_micros(100);
        const SERVICE_POLL_IDLE: Duration = Duration::from_millis(2);
        let mut poll = SERVICE_POLL;
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
            let interrupting = bus.service();
            poll = if interrupting { SERVICE_POLL } else {
                (poll * 2).min(SERVICE_POLL_IDLE)
            };
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
                bus.save_rtc();
                // The machine-events thread stops the machine and exits,
                // unless the host embeds iris (IRIS_NO_EXIT_ON_POWEROFF).
                if controller_up.load(Ordering::SeqCst) && events.try_send(MachineEvent::PowerOff).is_ok() {
                    return;
                }
                running.store(false, Ordering::SeqCst);
                crate::workstation::exit_process("power-off", 0, || {
                    crate::workstation::flushed(bus.flush_disks(), "power-off")
                });
            }
            std::thread::sleep(poll);
        }
    }

    /// Stop the CPU, hold the service thread off the devices, let the
    /// rendering engine finish what it was given, and save the PROM's flash.
    /// The display thread and the monitor run on until the machine is
    /// dropped; [`Self::restart_peripherals`] lets the service thread go.
    pub fn stop(&mut self) {
        self.cpu.clone().as_device().stop();
        self.paused.store(true, Ordering::SeqCst);
        // The guest clock points into this machine's CPU core.
        #[cfg(feature = "perfetto")]
        crate::perfetto::set_guest_cycles(None);
        // After the CPU, so nothing queues more; each finishes the command
        // it is running, so a snapshot finds none half done.
        for ctl in self.bus.scsi.lock().unwrap().iter() {
            ctl.stop();
        }
        for c in self.bus.cards.get().into_iter().flat_map(|c| c.devices()) {
            c.stop();
        }
        if !self.bus.crime.re.drain(Duration::from_secs(2)) {
            eprintln!("iris: O2 stopped with rendering engine writes still queued");
        }
        self.bus.save_flash();
        self.bus.save_rtc();
        // Not through the display: this stop leaves the guest's writes durable
        // whatever follows it.
        if let Err(e) = self.bus.flush_disks() {
            eprintln!("iris: disk flush at stop failed: {e} (guest writes may be lost)");
        }
        self.run_uarts(false);
        self.timer_manager.clock().stop();
        // A machine that was never started has nothing to halt.
        if self.lifecycle != crate::machine_state::GuestState::Idle {
            self.lifecycle = crate::machine_state::GuestState::Halted;
        }
    }

    /// Let the service thread look after the devices again.
    pub fn restart_peripherals(&mut self) {
        self.timer_manager.clock().start();
        self.paused.store(false, Ordering::SeqCst);
        self.start_scsi();
        self.run_uarts(true);
    }

    /// The console UARTs' receive-timeout timers follow the machine.
    fn run_uarts(&self, on: bool) {
        use crate::traits::Device;
        for com in [&self.bus.mace.com0, &self.bus.mace.com1] {
            if on { com.start() } else { com.stop() }
        }
    }

    /// Give each SCSI controller its own thread, so disk transfers do not
    /// run on the CPU's.
    fn start_scsi(&self) {
        for ctl in self.bus.scsi.lock().unwrap().iter() {
            ctl.start();
        }
        for c in self.bus.cards.get().into_iter().flat_map(|c| c.devices()) {
            c.start();
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

impl Drop for O2Machine {
    fn drop(&mut self) {
        // First wait out any monitor, event or CI command in flight, so none
        // sees a half-destroyed machine (see `Drop for Machine`).
        if let Some(slot) = self.controller_slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        // The CPU and its compile pool first: they reach the bus, which is
        // freed once this returns.
        self.cpu.clone().as_device().stop();
        #[cfg(feature = "jitv2")]
        crate::cpu::jitv2::Jitv2::stop_for_drop(&self.cpu.jitv2(), || self.cpu.clone().as_device().stop());
        // The refresh thread lets go of the renderer here (`DisplayStop`).
        self.display.head.stop();
        self.running.store(false, Ordering::SeqCst);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        crate::workstation::flush_on_drop(self);
        self.monitor.shutdown();
        // The rendering engine's thread runs from construction and holds the bus.
        self.bus.crime.re.stop();
        // A placed BAR's device holds the card, whose DMA memory owns the bus.
        self.bus.phys.release_devices();
    }
}

/// The O2 as the front ends see it. `--load-elf` is not there yet; the
/// trait's default says so.
impl O2Machine {
    /// Disk bus `bus` in the order of `disk_activity`: the AIC-7880s, then
    /// each NVMe card.
    fn with_disk_bus<R>(&self, bus: usize, f: impl FnOnce(&crate::scsi::ScsiBus) -> R) -> Option<R> {
        use crate::scsi_monitor::ScsiTargets;
        let controllers = self.bus.scsi.lock().unwrap().clone();
        if let Some(c) = controllers.get(bus) {
            return Some(f(c.bus()));
        }
        let cards = self.bus.cards.get()?;
        let card = cards.nvme.get(bus - controllers.len())?;
        Some(f(card.nvme.scsi_bus()))
    }
}

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
    fn monitor(&self) -> &Arc<crate::monitor::Monitor> { &self.monitor }
    /// The PROM's flash, where `setenv` keeps its settings, and the RTC's RAM,
    /// written out now. The flash goes only to the file `nvram` names.
    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        if path.is_some() {
            return Err("the O2 saves its flash only to the file `nvram` names".into());
        }
        let file = self.bus.flash_path().ok_or("no nvram file configured")?;
        self.bus.save_flash();
        self.bus.save_rtc();
        Ok(file.display().to_string())
    }
    fn lifecycle(&self) -> crate::machine_state::GuestState { self.lifecycle }
    fn set_lifecycle(&mut self, state: crate::machine_state::GuestState) { self.lifecycle = state; }

    fn get_display(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> { Some(O2Machine::get_display(self)) }
    fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> { O2Machine::get_ps2(self) }
    fn register_monitor_device(&self, device: Arc<dyn crate::traits::Device>) { self.monitor.register_device(device); }
    fn get_timer_manager(&self) -> Arc<TimerManager> { O2Machine::get_timer_manager(self) }
    fn get_cpu_debug(&self) -> Arc<dyn crate::gdb_stub::CpuDebug> { self.cpu.clone().debug_adapter() }
    fn fasttick_count(&self) -> Arc<std::sync::atomic::AtomicU64> { self.cpu.fasttick_count() }
    fn get_ci_serial(&self) -> Option<Arc<crate::dev::serial_host::CiSerialBackend>> { self.ci_serial.clone() }
    /// Typed at serial port 1, the console with `graphics = false`.
    fn inject_serial_console(&self, bytes: &[u8]) {
        match &self.ci_serial {
            Some(ci) => ci.push_host(bytes),
            None => self.bus.mace.com0.feed(bytes),
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }

    fn snapshot_refusal(&self) -> Option<String> {
        self.bus.cards.get().and_then(|c| c.snapshot_blocker())
    }
    fn snapshot_layout(&self) -> crate::snapshot::Layout {
        let cards = self.bus.cards.get();
        let mut states = vec!["cpu".to_string(), "o2".into(), "boot".into()];
        states.extend(cards.filter(|c| !c.is_empty()).map(|_| "cards".to_string()));
        crate::snapshot::Layout {
            machine: Some(SNAPSHOT_MACHINE),
            disks: self.disks.clone(),
            slots: cards.map(|c| c.manifest()).unwrap_or_default(),
            nvram: self.flash_path.clone(),
            cpu: (self.cpu.model_name().into(), self.cpu.cache_kind().into()),
            states,
            files: vec![("flash.bin".into(), self.bus.flash_bytes().len())],
            memory: vec![self.bus.ram.len() / 4],
            ..Default::default()
        }
    }
    fn capture(&self) -> Result<crate::snapshot::Image<'_>, String> {
        use crate::traits::Saveable;
        let mut states = vec![("cpu".to_string(), self.cpu.save_state()), ("o2".into(), self.bus.save_state()), ("boot".into(), self.boot.save())];
        states.extend(self.bus.cards.get().filter(|c| !c.is_empty()).map(|c| ("cards".to_string(), c.save_state())));
        Ok(crate::snapshot::Image {
            states,
            files: vec![("flash.bin".into(), self.bus.flash_bytes())],
            memory: self.bus.ram.bank_words().into_iter().map(|w| (0, w.into())).collect(),
            ..Default::default()
        })
    }
    fn apply_image(&mut self, image: &crate::snapshot::Image) -> Result<(), String> {
        use crate::traits::{Resettable, Saveable};
        Resettable::power_on(&*self.cpu);
        self.bus.power_on();
        self.cpu.load_state(image.state("cpu")?)?;
        self.bus.load_state(image.state("o2")?)?;
        if let Some(cards) = self.bus.cards.get() {
            cards.load_state(image.state("cards").unwrap_or(&toml::Value::Table(Default::default())))?;
        }
        self.boot.load(image.state("boot")?);
        self.bus.set_flash(image.file("flash.bin")?)?;
        self.bus.ram.restore_words(&image.memory[0].1)
    }
    fn with_disk_buses(&self, f: &mut dyn FnMut(&[crate::snapshot::DiskBus]) -> Result<(), String>) -> Result<(), String> {
        use crate::scsi_monitor::ScsiTargets;
        let controllers = self.bus.scsi.lock().unwrap().clone();
        let places = crate::topology::scsi_buses(&MachineProfile::O2Ip32.slots()).into_iter().map(|b| b.1);
        f(&crate::snapshot::disk_buses(places, controllers.iter().map(|c| c.scsi_bus()), self.bus.cards.get()))
    }
    fn snapshots(&mut self) -> Option<&mut crate::snapshot::Restored> {
        Some(&mut self.restored)
    }

    fn scratch_path(&self) -> Option<&std::path::Path> { self.scratch_path.as_deref() }
    fn disc_changer(&self) -> Option<Arc<dyn crate::workstation::DiscChanger>> {
        let ctls = self.bus.scsi.lock().unwrap().iter().map(|c| c.clone() as _).collect();
        Some(Arc::new(crate::workstation::DiscChangers(ctls)))
    }
    /// The same `/CDROM` remount the Indy types: controller 0 is `dks0` to
    /// IRIX on the O2 as well.
    fn remount_cdrom_guest(&self, scsi_id: u8) {
        let script = crate::machine::Machine::irix_cdrom_remount_command(scsi_id);
        eprintln!("IRIX: remount /CDROM for SCSI #{scsi_id} (console shell must be active)");
        self.inject_serial_console(script.as_bytes());
    }

    fn pending_chd_sync_count(&self) -> usize {
        use crate::scsi_monitor::ScsiTargets;
        self.bus.scsi.lock().unwrap().iter().map(|c| c.bus().pending_chd_sync_count()).sum::<usize>()
            + self.bus.cards.get().into_iter().flat_map(|c| &c.nvme).map(|c| c.nvme.scsi_bus().pending_chd_sync_count()).sum::<usize>()
    }
    fn disk_activity(&self) -> Vec<Vec<crate::scsi::DiskCounts>> {
        use crate::scsi_monitor::ScsiTargets;
        let controllers = self.bus.scsi.lock().unwrap().clone();
        controllers.iter().map(|c| c.bus().activity())
            .chain(self.bus.cards.get().into_iter().flat_map(|c| &c.nvme).map(|c| c.nvme.scsi_bus().activity()))
            .collect()
    }
    fn read_disk(&self, bus: usize, id: u8, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.with_disk_bus(bus, |b| b.read_media(id, offset, buf))
            .unwrap_or_else(|| Err(std::io::Error::new(std::io::ErrorKind::NotFound, "no such disk bus")))
    }
    fn disk_size(&self, bus: usize, id: u8) -> Option<u64> {
        self.with_disk_bus(bus, |b| b.media_size(id)).flatten()
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.bus.flush_disks()
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        use crate::scsi_monitor::ScsiTargets;
        let controllers = self.bus.scsi.lock().unwrap().clone();
        let nvme = self.bus.cards.get().into_iter().flat_map(|c| &c.nvme);
        let buses: Vec<_> = controllers.iter().map(|c| c.bus()).chain(nvme.map(|c| c.nvme.scsi_bus())).collect();
        crate::scsi::ScsiBus::sync_chd_disks(&buses, only, progress, cancel)
    }

    fn net_frames(&self) -> Option<Arc<std::sync::atomic::AtomicU64>> {
        self.frames.clone()
    }
    fn nat_control(&self) -> Option<Arc<crate::net::NatControl>> {
        self.nat.as_ref().map(|(n, _)| n.clone())
    }
    fn nat_expected(&self) -> Option<(std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr)> {
        self.nat.as_ref().map(|(_, a)| *a)
    }
}

/// GBE's picture, scanned out of RAM every frame by the display refresh
/// thread and handed to whatever renderer the front-end installs.
pub struct O2Display {
    bus: Arc<Ip32Bus>,
    /// The display channel: screen, renderer slot and frame requests
    /// (`crate::display`).
    head: Arc<crate::display::HeadHandle>,
    /// The scan-out's RGBA bytes, and the last size shown.
    scan: Mutex<Scan>,
    /// The status bar's inputs; the heartbeat is the MACE ID chip's, and the
    /// Hz is the CPU's Compare-match count (IRIX's clock tick on the O2,
    /// which has no 8254).
    status: Arc<crate::status::MachineStatus>,
    /// Where the beam meets each of GBE's four timing interrupts, in lines
    /// after the start of vertical blanking, as of the last retrace.
    irq_lines: Mutex<[Option<u32>; 4]>,
    /// The machine is stopped: the guest's state is left alone.
    paused: Arc<AtomicBool>,
}

/// The scan-out's side of the screen.
struct Scan {
    /// The reader's view of RAM.
    window: crate::dev::crime_mem::RamWindow,
    /// The last size shown.
    size: (usize, usize),
    /// What the screen's buffer holds of GBE's frame (`Gbe::render_direct`).
    memo: crate::dev::o2gfx::ScanMemo,
    /// The buffer holds the black of a display that is off.
    black: bool,
}

impl O2Display {
    fn new(bus: Arc<Ip32Bus>, status: Arc<crate::status::MachineStatus>, paused: Arc<AtomicBool>) -> Self {
        Self {
            irq_lines: Mutex::new([None; 4]),
            paused,
            head: status.refresh().head(status.clone()),
            scan: Mutex::new(Scan { window: bus.ram.window(), size: (0, 0), memo: Default::default(), black: false }),
            bus,
            status,
        }
    }
}

impl crate::display::Head for O2Display {
    /// GBE scans RAM the CPU and the rendering engine write with no mark
    /// the display could test, so every frame is looked at while someone
    /// watches: the scan-out finds what changed (`compose` answers
    /// `Unchanged` when nothing did). Headless, nothing would see it.
    fn changed(&self) -> bool {
        self.head.watched()
    }

    /// The frame rate is the guest's: GBE's frame size at its dot clock.
    /// The lines past the picture's height are the vertical blank. GBE's
    /// frame begins here (its shadows go live), and its timing interrupts,
    /// which are line compares, are asked for at their lines.
    fn vblank_start(&self, _field: u32, _t_ns: u64) {
        if !self.paused.load(Ordering::Relaxed) {
            let lines = self.bus.gbe.retrace();
            let mut at = [0u32; 4];
            let mut n = 0;
            for line in lines.into_iter().flatten() {
                at[n] = line;
                n += 1;
            }
            self.head.set_line_events(&at[..n]);
            *self.irq_lines.lock() = lines;
        }
        if let Some((h_total, v_total, pixel_clock_hz)) = self.bus.video_totals() {
            let height = self.scan.lock().size.1 as u32;
            let blank = v_total.saturating_sub(height);
            let vblank_lines = if height != 0 && blank != 0 { blank } else { (v_total / 25).max(1) };
            self.head.set_timing(crate::display::VideoTiming { h_total, v_total, pixel_clock_hz, vblank_lines, fields: 1 });
        }
    }

    fn line(&self, line: u32, _field: u32, _t_ns: u64) {
        if self.paused.load(Ordering::Relaxed) {
            return;
        }
        let lines = *self.irq_lines.lock();
        for (n, at) in lines.into_iter().enumerate() {
            if at == Some(line) {
                self.bus.gbe.timing_interrupt(n);
            }
        }
    }

    /// GBE's picture straight out of RAM into the screen
    /// (`Gbe::render_direct`); black at the last size while the display is
    /// off.
    fn compose(&self, screen: &mut Rex3Screen, _want: crate::display::Want) -> crate::display::Composed {
        let mut scan = self.scan.lock();
        let Scan { window, size, memo, black } = &mut *scan;
        let stride = crate::disp::FB_STRIDE;
        let (w, h, new) = match self.bus.gbe.render_direct(window, &mut screen.fb_rgb, stride, memo) {
            Some((w, h, wrote)) => {
                *black = false;
                (w as usize, h as usize, wrote)
            }
            None => {
                let (w, h) = if size.0 == 0 { (1280, 1024) } else { *size };
                let new = !*black;
                if new {
                    for y in 0..h {
                        screen.fb_rgb[y * stride..y * stride + w].fill(0xff00_0000);
                    }
                    memo.forget();
                    *black = true;
                }
                (w, h, new)
            }
        };
        let new = new || (w, h) != *size || !screen.prebuilt;
        screen.prebuilt = true;
        screen.width = w;
        screen.height = h;
        *size = (w, h);
        if new { crate::display::Composed::Frame } else { crate::display::Composed::Unchanged }
    }
}

impl crate::gfx_display::GfxDisplay for O2Display {
    fn renderer_slot(&self) -> &Mutex<Option<Box<dyn Renderer>>> { self.head.renderer_slot() }
    fn display_stop(&self) -> &crate::gfx_display::DisplayStop { self.head.display_stop() }
    fn screen(&self) -> &Mutex<Rex3Screen> { self.head.screen() }
    fn request_screenshot(&self) { self.head.request_screenshot(); }
    fn screenshot_pending(&self) -> bool { self.head.screenshot_pending() }
    fn request_frame(&self) { self.head.request_frame(); }
    fn frame_pending(&self) -> bool { self.head.frame_pending() }
    fn set_capture(&self, on: bool) { self.head.set_capture(on); }
    fn cycles(&self) -> CyclesPtr {
        self.status.cycles()
    }
    fn hw_cursor(&self) -> Option<crate::gfx_display::HwCursor> {
        let (x, y, mask) = self.bus.gbe.cursor()?;
        Some(crate::gfx_display::HwCursor { x, y, size: 32, mask })
    }
}

/// Where the CPU was at the end of its last batch, and how many batches it
/// has run: enough to tell a guest spinning in one place from one that is
/// still moving, without stopping it.
pub(crate) static LAST_PC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static BATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The general registers at the same moment, then Status, Cause, EPC and
/// BadVAddr: what a polling loop is polling.
static LAST_REGS: [std::sync::atomic::AtomicU64; 42] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 42];

/// The O2's own diagnostics on the shared monitor, as `o2 <command> [args]`.
/// Each replies with one line, `ok ...` or `err ...`.
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

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("o2".to_string(), "O2 diagnostics: o2 status | pixel X Y | pc | regs | peek ADDR LEN PATH | \
            scsi PATH | unmapped | ps2log PATH | crimelog PATH".to_string())]
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

fn control_command(bus: &Ip32Bus, cmd: &str, arg: &str, rgba: &mut [u8]) -> String {
    match cmd {
        "pc" => format!(
            "ok pc 0x{:08x} after {} batches",
            LAST_PC.load(Ordering::Relaxed),
            BATCHES.load(Ordering::Relaxed)
        ),
        // `crimelog PATH`: the guest's recent CRIME interrupt-register
        // accesses, oldest first.
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
        _ => format!("err unknown command {cmd:?}"),
    }
}


#[cfg(test)]
mod drop_tests {
    use super::*;
    use crate::config::{GraphicsBoard, MachineProfile};
    use crate::display_stop_tests::{big_stack, config, scratch};
    use crate::workstation::Workstation;

    /// A dropped O2 frees its bus and with it the decode table: a process may
    /// have only 64, and a test binary builds many machines.
    #[test]
    fn a_dropped_o2_frees_its_bus_and_decode_table() {
        let dir = scratch("o2-drop");
        let cfg = config(&dir, MachineProfile::O2Ip32, GraphicsBoard::Newport);
        let machine = big_stack(move || O2Machine::new(&cfg).expect("O2"));
        let phys = Arc::downgrade(&machine.bus.phys);
        drop(machine);
        assert_eq!(phys.strong_count(), 0, "the bus outlives its machine");
    }

    /// The installer's `ip32prom.image` is the flash behind a 256-byte
    /// header and is shorter than the part. Loaded as it was, the PROM ran
    /// 256 bytes off and parked the CPU without a word. The O2 strips the
    /// header and fills the rest of the flash erased; a raw file must still
    /// be the whole flash.
    #[test]
    fn the_installer_prom_image_fills_the_flash_and_a_short_raw_one_is_refused() {
        let dir = scratch("o2-installer-prom");
        let cfg = config(&dir, MachineProfile::O2Ip32, GraphicsBoard::Newport);
        let mut image = vec![0u8; 0x100];
        image[..4].copy_from_slice(b"PROM");
        image[12..16].copy_from_slice(&0x1000u32.to_be_bytes());
        image.extend(vec![0x5a; 0x1000]);
        std::fs::write(&cfg.prom, &image).unwrap();
        let wrapped = cfg.clone();
        let machine = big_stack(move || O2Machine::new(&wrapped).expect("O2"));
        let flash = machine.bus.flash_bytes();
        assert_eq!(flash.len(), PROM_SIZE as usize);
        assert!(flash[..0x1000].iter().all(|&b| b == 0x5a), "the payload starts the flash");
        assert!(flash[0x1000..].iter().all(|&b| b == 0xff), "the rest is erased");
        drop(machine);
        std::fs::write(&cfg.prom, vec![0x5a; 0x1000]).unwrap();
        let e = big_stack(move || O2Machine::new(&cfg).err()).expect("a short raw PROM is refused");
        assert!(e.contains("not a whole"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `iris-ci rtc-save` on an O2 writes the PROM's flash to the `nvram`
    /// file now, rather than failing as a machine with no NVRAM does.
    #[test]
    fn rtc_save_writes_the_flash_to_the_nvram_file() {
        let dir = scratch("o2-save-nvram");
        let cfg = config(&dir, MachineProfile::O2Ip32, GraphicsBoard::Newport);
        let nvram = std::path::PathBuf::from(&cfg.nvram);
        let machine = big_stack(move || O2Machine::new(&cfg).expect("O2"));
        assert!(!nvram.exists());
        assert_eq!(Workstation::save_nvram(&machine, None), Ok(nvram.display().to_string()));
        assert_eq!(std::fs::metadata(&nvram).unwrap().len(), PROM_SIZE as u64);
        assert!(Workstation::save_nvram(&machine, Some("elsewhere.bin")).is_err());
        drop(machine);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `iris-ci rollback` on an O2 returns memory and a copy-on-write disk
    /// to what the last `restore` loaded, again and again, without reading
    /// the snapshot's state a second time; with no restore it says so.
    /// Snapshots go under `saves/` in the working directory, so the test runs
    /// again as a child in a directory of its own.
    #[test]
    fn ci_rollback_returns_to_the_last_restore() {
        const CHILD: &str = "IRIS_O2_ROLLBACK_CHILD";
        let Some(dir) = std::env::var_os(CHILD) else {
            let dir = scratch("o2-rollback");
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .current_dir(&*dir)
                .env(CHILD, &*dir)
                .args(["--exact", "ip32_machine::drop_tests::ci_rollback_returns_to_the_last_restore",
                       "--nocapture", "--test-threads=1"])
                .output().unwrap();
            assert!(output.status.success(), "rollback child failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            return;
        };
        let dir = std::path::PathBuf::from(dir);
        let disk = dir.join("disk.raw");
        std::fs::write(&disk, vec![0u8; 1 << 20]).unwrap();
        let mut cfg = config(&dir, MachineProfile::O2Ip32, GraphicsBoard::Newport);
        cfg.set_disk(1, crate::config::ScsiDeviceConfig {
            path: disk.to_string_lossy().into_owned(), overlay: true, ..Default::default()
        });
        let mut machine = big_stack(move || Box::new(O2Machine::new(&cfg).expect("O2")));
        let at = crate::ip32::RAM_BASE as usize + 0x10_0000;
        let write = |m: &O2Machine, fill: u8| {
            m.bus.ram.write(at, &[fill; 4]);
            let req = crate::scsi::ScsiRequest {
                cdb: vec![0x2a, 0, 0, 0, 0, 2, 0, 0, 1, 0],
                data_len: crate::scsi::ScsiDataLength::Unlimited,
                data_in: Some(vec![fill; 512]),
            };
            let r = m.bus.scsi.lock().unwrap()[0].bus().request(1, &req).unwrap().unwrap();
            assert_eq!(r.status, 0, "the write reaches the disk");
        };
        let read = |m: &O2Machine| {
            let req = crate::scsi::ScsiRequest {
                cdb: vec![0x28, 0, 0, 0, 0, 2, 0, 0, 1, 0],
                data_len: crate::scsi::ScsiDataLength::Unlimited,
                data_in: None,
            };
            let r = m.bus.scsi.lock().unwrap()[0].bus().request(1, &req).unwrap().unwrap();
            (m.bus.ram.bytes(at, 4)[0], r.data[0])
        };

        assert!(machine.ci_rollback().is_err(), "nothing restored yet");
        machine.stop();
        write(&machine, 0x11);
        machine.save_snapshot("o2-rollback").unwrap();
        machine.stop();
        write(&machine, 0x22);
        machine.ci_restore("o2-rollback").unwrap();
        machine.stop();
        assert_eq!(read(&machine), (0x11, 0x11), "the restore puts back memory and disk");
        // The rollback keeps what the restore read; only the overlays are
        // read again.
        std::fs::remove_file(std::path::Path::new("saves/o2-rollback/snapshot.toml")).unwrap();
        for fill in [0x33, 0x44] {
            write(&machine, fill);
            machine.ci_rollback().unwrap();
            assert!(machine.cpu_is_running(), "the CPU runs on after a rollback");
            machine.stop();
            assert_eq!(read(&machine), (0x11, 0x11), "rolled back from {fill:#x}");
        }
        machine.save_snapshot("o2-child").unwrap();
        let child = crate::snapshot::Snapshot::new("saves/o2-child").read_manifest().unwrap().unwrap();
        assert_eq!(child.parent.as_deref(), Some("o2-rollback"), "a save names the restore it followed");
        drop(machine);
    }

    /// `[nvme]` puts a card in the "PCI option" slot and records its disk;
    /// the bus's disk flush reaches its namespace.
    #[test]
    fn an_nvme_card_is_attached_in_the_option_slot() {
        let dir = scratch("o2-nvme");
        let disk = dir.join("nvme.raw");
        std::fs::write(&disk, vec![0u8; 1 << 20]).unwrap();
        let mut cfg = config(&dir, MachineProfile::O2Ip32, GraphicsBoard::Newport);
        let card = crate::config::NvmeConfig { path: disk.to_string_lossy().into_owned(), overlay: true, ..Default::default() };
        cfg.devices.push(crate::devices::ConfiguredDevice::new("pci_slot", crate::devices::DeviceModel::Nvme(card)));
        let machine = big_stack(move || O2Machine::new(&cfg).expect("O2"));
        let at = crate::pci::DevFn { device: crate::dev::mace_pci::IP32_OPTION_SLOT, function: 0 };
        assert_eq!(machine.bus.mace.pci.bus().config_read(at, 8, 4).map(|d| d >> 8), Some(0x01_0802), "the card in the slot");
        let card = machine.bus.cards.get().and_then(|c| c.nvme.first()).expect("the card is attached");
        assert!(machine.disks.iter().any(|d| d.at == card.at && d.size_bytes == 1 << 20));
        machine.bus.flush_disks().unwrap();
        drop(machine);
    }

    /// An audio card with no slot takes the option slot.
    #[test]
    fn an_audio_card_takes_the_option_slot() {
        let dir = scratch("o2-audio");
        let mut cfg = config(&dir, MachineProfile::O2Ip32, GraphicsBoard::Newport);
        cfg.no_audio = true;
        cfg.devices.push(crate::devices::ConfiguredDevice::from_table(toml::from_str("type = \"ice1724\"").unwrap()).unwrap());
        let machine = big_stack(move || O2Machine::new(&cfg).expect("O2"));
        let at = crate::pci::DevFn { device: crate::dev::mace_pci::IP32_OPTION_SLOT, function: 0 };
        assert_eq!(machine.bus.mace.pci.bus().config_read(at, 0, 4), Some(0x1724_1412));
        assert_eq!(machine.bus.cards.get().unwrap().manifest().len(), 1);
        drop(machine);
    }

    /// A card whose BAR the guest has placed is mapped on the bus, and the
    /// card holds memory for its DMA, which owns the bus: the drop must
    /// still free it.
    #[test]
    fn a_dropped_o2_with_a_placed_pci_bar_frees_its_bus() {
        let dir = scratch("o2-bar-drop");
        let disk = dir.join("nvme.raw");
        std::fs::write(&disk, vec![0u8; 1 << 20]).unwrap();
        let mut cfg = config(&dir, MachineProfile::O2Ip32, GraphicsBoard::Newport);
        let card = crate::config::NvmeConfig { path: disk.to_string_lossy().into_owned(), overlay: true, ..Default::default() };
        cfg.devices.push(crate::devices::ConfiguredDevice::new("pci_slot", crate::devices::DeviceModel::Nvme(card)));
        let machine = big_stack(move || O2Machine::new(&cfg).expect("O2"));
        let pci = machine.bus.mace.pci.bus();
        let at = crate::pci::DevFn { device: crate::dev::mace_pci::IP32_OPTION_SLOT, function: 0 };
        let kept = machine.bus.phys.kept_len();
        pci.config_write(at, 0x10, 4, 0x8000_4000);
        pci.config_write(at, 0x14, 4, 0);
        pci.config_write(at, 0x04, 2, crate::pci::PCI_COMMAND_MEMORY);
        assert!(machine.bus.phys.kept_len() > kept, "the BAR is mapped on the bus");
        let phys = Arc::downgrade(&machine.bus.phys);
        drop(machine);
        assert_eq!(phys.strong_count(), 0, "the bus outlives its machine");
    }
}
