//! The Octane (IP30) as a machine: an R10000 on HEART, the Crossbow, and
//! the widgets behind it, with iris's CPU thread, monitor and front ends.
//!
//! Selected with `[machine] profile = "octane_ip30"`; the shared keys
//! configure it (`prom`, `nvram`, `banks`, `[scsi.N]`, `[network]`, the
//! serial ports, `[clock] fixed_mhz`) and `[octane]` holds what only an
//! Octane has (see [`crate::config::OctaneSection`]).
//!
//! Built as the Indy is: the devices first, then the bus over them
//! (`ip30::Ip30Bus`, which configures the machine's `Physical`), then `init`, then the CPU. On the Bridge's PCI
//! bus, as on the board: the two ISP1040 SCSI controllers in slots 0 and 1,
//! IOC3 (serial, keyboard and mouse, the clock, Ethernet) in slot 2, RAD1
//! audio in slot 3. IMPACT sits on a graphics port.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::config::{CpuModel, MachineConfig, MachineProfile};
use crate::cpu::mips_cache_shadow::R10000ShadowCache;
use crate::cpu::smp::World;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::bridge::{Bridge, RawFlash};
use crate::dev::heart::{host_clock, Heart, HeartConfig, PIU_BASE, PIU_SIZE};
use crate::dev::heart_mem::HeartRam;
use crate::dev::ioc3::Ioc3;
use crate::dev::ioc3_eth::Ioc3Eth;
use crate::dev::isp1040::Isp1040;
use crate::dev::mgras::hq4::ImpactXio;
use crate::dev::odyssey::Odyssey;
use crate::dev::rad1::Rad1;
use crate::dev::sflash::{CfiFlash, FLASH_SIZE};
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::dev::xbow::Xbow;
use crate::hptimer::TimerManager;
use crate::ip30::{
    widget_small_base, CachedFlashLines, Ip30Bus, LAST_PC, WIDGET_BRIDGE, WIDGET_HEART, WIDGET_SMALL_SIZE,
    WIDGET_XBOW,
};
use crate::chunk_store::{get_chunks_as_words, put_words_as_chunks, ChunkStore};
use crate::snapshot::{field, ChunksManifest, Manifest, Snapshot, StateTable, SCHEMA_VERSION};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable, Saveable};
use crate::workstation::{MachineRef, MachineSlot, Workstation};

/// The PROM image tried when `prom` is left at the shared default, named
/// as IRIX ships it in /usr/cpu/firmware. The Octane has no embedded PROM.
pub const OCTANE_PROM_FILE: &str = "IP30prom.bin";
/// The 195 MHz R10000 of SGI's standard 128 MB Octane configurations (the
/// OCTANE Technical Report): Count at half that.
pub const OCTANE_COUNT_HZ: u64 = 97_500_000;
/// R10000 revision 2.7, as an Octane's hinv reports it; the raw PRId is
/// inferred from the R10000 manual's encoding.
pub const OCTANE_PRID: u32 = 0x0927;
/// The Crossbow on the Octane reports revision code 4.
const XBOW_REVISION: u32 = 4;
/// The Octane2's Crossbow: revision code 5, which IRIX's inventory names
/// Crossbow 1.4 (a PROM-monitor read of 0x50000001 beside an inventory of
/// 1.4 on an Octane2). SGI's VPro upgrade kit (007-4210-003)
/// requires Crossbow 1.4 for the V10 and V12, so a machine with one of
/// those presents it.
const XBOW_REVISION_OCTANE2: u32 = 5;
/// The Bridge's silicon revision: 3 on the 175 MHz Octane (an OpenBSD boot
/// there prints "Bridge revision 3"), 4 on the reference Octane2, whose PROM
/// monitor read the identification word as 0x4c00206d.
const BRIDGE_REVISION: u32 = 3;
const BRIDGE_REVISION_OCTANE2: u32 = 4;

/// The PROM revision a VPro board needs (007-4210-003): 4.0 for the V6 and
/// V8, 4.5 for the V10 and V12.
fn vpro_prom_needed(board: crate::config::GraphicsBoard) -> (u32, u32) {
    match board {
        crate::config::GraphicsBoard::V10 | crate::config::GraphicsBoard::V12 => (4, 5),
        _ => (4, 0),
    }
}

/// The PROM's own revision, from its banner ("SGI Version 6.5 Rev 4.9
/// IP30 ..."), if the image carries one.
fn prom_revision(prom: &[u8]) -> Option<(u32, u32)> {
    let text = String::from_utf8_lossy(prom);
    let at = text.find(" Rev ")?;
    let rest = &text[at + 5..];
    let end = rest.find(" IP30")?;
    let (major, minor) = rest[..end].split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}
/// The graphics port the one IMPACT board sits in. A choice: it is where a
/// measured Octane2 has its graphics board, not a measured SI placement.
pub const WIDGET_GFX: u8 = 11;
/// The Bridge input IOC3's SuperIO block (serial, keyboard, clock) is
/// wired to; the Ethernet uses IOC3's PCI pin.
const SUPERIO_PIN: u8 = 4;
/// Bridge slot of the virtio device: 0 and 1 are SCSI, 2 IOC3, 3 the RAD.
const VIRTIO_SLOT: u8 = 4;
/// The blank part on the Bridge's second flash select: reads all ones and
/// keeps no writes, an approximation from one word read on a real Octane2.
/// The PROM looks for a part there too.
const FLASH_SELECT1_SIZE: usize = 2 << 20;

/// Bank sizes in bytes from `banks` (MB per HEART bank).
pub fn heart_banks(cfg: &MachineConfig) -> [u64; 4] {
    std::array::from_fn(|i| (cfg.banks[i] as u64) << 20)
}

/// A MicroLAN with one board-record part. The barcodes (EMU...) and ROM
/// serials are made up and marked so; the part numbers and names are an
/// Octane's.
fn record_bus(barcode: &str, part: &str, revision: &str, name: &str, serial: u64) -> w1::Bus {
    let mut bus = w1::Bus::new();
    bus.add(Box::new(board_record_part(&BoardRecord { barcode, part, revision, name }, serial)));
    bus
}

pub struct OctaneMachine {
    /// Where this machine has memory.
    pub memory: Arc<crate::memory_map::MemoryMap>,
    pub phys: Arc<Ip30Bus>,
    pub heart: Arc<Heart>,
    pub ram: Arc<HeartRam>,
    pub xbow: Arc<Xbow>,
    /// CPU 0 first. Whatever means "the CPU" to a front end is CPU 0.
    cpus: Vec<Arc<dyn CpuDevice>>,
    /// One per CPU: each caches the PROM's data privately.
    flash_lines: Vec<Arc<CachedFlashLines>>,
    timer_manager: Arc<TimerManager>,
    monitor: Arc<crate::monitor::Monitor>,
    monitor_port: u16,
    /// The console UART's in-process host end in CI mode (`--ci`).
    ci_serial: Option<Arc<crate::dev::serial_host::CiSerialBackend>>,
    pub bridge: Arc<Bridge>,
    pub ioc3: Arc<Ioc3>,
    pub scsi: [Arc<Isp1040>; 2],
    eth: Arc<Ioc3Eth>,
    rad: Arc<Rad1>,
    /// virtio-serial (`[virtio_serial]`), the next free Bridge slot.
    virtio: Option<Arc<crate::dev::virtio::pci::VirtioPci>>,
    flash: Arc<CfiFlash>,
    /// The IMPACT board, when `[graphics] board` names one.
    pub gfx: Option<Arc<ImpactXio>>,
    /// The VPro board, with `[graphics] board = "v6"`, `"v8"`, `"v10"` or `"v12"`.
    pub vpro: Option<Arc<Odyssey>>,
    nat: Option<(Arc<crate::net::NatControl>, (std::net::Ipv4Addr, std::net::Ipv4Addr, std::net::Ipv4Addr))>,
    /// Guest frames the network backend has processed.
    frames: Option<Arc<std::sync::atomic::AtomicU64>>,
    scratch_path: Option<std::path::PathBuf>,
    nvram_path: String,
    event_rx: Option<std::sync::mpsc::Receiver<MachineEvent>>,
    controller_slot: Option<MachineSlot>,
    lifecycle: crate::machine_state::GuestState,
    /// The disks as a snapshot records them, to refuse one taken on others.
    disks: Vec<crate::snapshot::DiskRef>,
    /// The snapshot the CI socket last restored, for `rollback`.
    last_restore: Option<String>,
    /// Where snapshots go: `saves` under the current directory, as on the
    /// Indy and the O2. A field rather than the literal so the tests, which
    /// share one process and so one current directory, can each use their own.
    saves: std::path::PathBuf,
}

impl OctaneMachine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        MachineProfile::OctaneIp30.accepts_cpu(cfg.machine.cpu)?;
        // `prom`, or IRIX's file name when `prom` is the shared default and
        // that file is not there: there is no embedded Octane PROM.
        let prom_path = MachineProfile::OctaneIp30.prom_path(&cfg.prom);
        let prom = std::fs::read(&prom_path)
            .map_err(|e| format!("Octane PROM {prom_path}: {e} (set `prom` to an IP30 PROM image, the 1 MB flash contents)"))?;
        if prom.len() > FLASH_SIZE {
            return Err(format!("Octane PROM {prom_path} is {} bytes; the flash holds {FLASH_SIZE}", prom.len()));
        }
        let cpu_count = cfg.machine.cpus;
        MachineProfile::OctaneIp30.accepts_cpus(cpu_count)?;
        let banks = heart_banks(cfg);
        let kind = cfg.graphics_board();
        let board = match kind {
            crate::config::GraphicsBoard::None => None,
            b if b.is_vpro() => None,
            b if b.is_impact() => Some(b),
            b => return Err(format!("graphics.board \"{}\" is not modelled on the Octane (valid: solidimpact, highimpact, maximpact, v6, v8, v10, v12, none)", b.as_str())),
        };
        let has_vpro = kind.is_vpro();
        if has_vpro {
            let needed = vpro_prom_needed(kind);
            if let Some(found) = prom_revision(&prom) {
                if found < needed {
                    eprintln!(
                        "iris: the {} needs PROM {}.{} or later; {prom_path} is {}.{}",
                        kind.label(),
                        needed.0,
                        needed.1,
                        found.0,
                        found.1
                    );
                }
            }
        }
        // The V10 and V12 shipped only in the Octane2, and SGI's upgrade kit
        // (007-4210-003) requires Crossbow 1.4 for them. Choosing one gives
        // the machine the Octane2's Crossbow (revision 5) and Bridge
        // (revision 4); there is no separate option for the machine.
        let octane2 = matches!(kind, crate::config::GraphicsBoard::V10 | crate::config::GraphicsBoard::V12);
        let xbow_revision = if octane2 { XBOW_REVISION_OCTANE2 } else { XBOW_REVISION };
        let (event_tx, event_rx) = std::sync::mpsc::sync_channel(4);
        let timer_manager = Arc::new(TimerManager::new());

        // Memory, and HEART, whose descriptors place it.
        let ram = Arc::new(HeartRam::try_new(std::array::from_fn(|i| cfg.banks[i]))?);
        let mem = ram.dma_view();
        let heart = Arc::new(Heart::new(HeartConfig { cpus: cpu_count, ..HeartConfig::octane(banks) }, host_clock()));
        // HEART's MicroLAN holds the processor module's record.
        // The PROM's diagnostics hold it to the processors HEART reports: a
        // PM10 has one and a PM20 two.
        let module = if cpu_count > 1 { "PM20" } else { "PM10" };
        heart.set_microlan(record_bus("EMUU00", "030-0888-004", "B", module, 0x5145_4d55_5500));

        let gfx_link = if board.is_some() || has_vpro { 1 << WIDGET_GFX } else { 0 };
        let xbow = Arc::new(Xbow::new(xbow_revision, (1 << WIDGET_HEART) | (1 << WIDGET_BRIDGE) | gfx_link));
        xbow.connect_heart(heart.clone());

        // The onboard Bridge: memory and interrupts through HEART, the boot
        // flash on its first chip select, the system board's record on its
        // MicroLAN.
        let bridge = Bridge::new(mem.clone(), heart.clone());
        bridge.set_revision(if octane2 { BRIDGE_REVISION_OCTANE2 } else { BRIDGE_REVISION });
        // The flash holds the PROM's own records as well as its code; with
        // `[octane] flash` they last from run to run.
        let flash = Arc::new(match &cfg.octane.flash {
            Some(path) => {
                CfiFlash::with_file(&prom, FLASH_SIZE, path).map_err(|e| format!("octane.flash {path}: {e}"))?
            }
            None => CfiFlash::new(&prom, FLASH_SIZE),
        });
        bridge.attach_flash(flash.clone(), FLASH_SIZE as u32);
        bridge.attach_flash_select1(Arc::new(RawFlash::new(&[], FLASH_SELECT1_SIZE)), FLASH_SELECT1_SIZE as u32);
        bridge.set_microlan(record_bus("EMUN00", "030-0887-003", "F", "IP30", 0x5145_4d55_4e00));
        // DMA reaches only widgets the Crossbow has a live link to.
        let links = xbow.clone();
        bridge.set_link_alive(Box::new(move |w| links.carries(w)));

        // The status bar's activity bits, shared with the graphics board.
        let heartbeat = Arc::new(AtomicU64::new(0));
        let mac = cfg.network().mac;
        // The buses are filled from `[scsi.N]` before the controllers exist.
        let [bus_a, bus_b] = [Isp1040::new_bus(), Isp1040::new_bus()];
        let (scratch_path, disks) = Self::attach_scsi(cfg, [&bus_a, &bus_b])?;
        let scsi = [Arc::new(Isp1040::new_at(0, bus_a)), Arc::new(Isp1040::new_at(1, bus_b))];
        for s in &scsi {
            s.set_heartbeat(heartbeat.clone());
        }
        bridge.attach(0, scsi[0].clone());
        bridge.attach(1, scsi[1].clone());
        let ioc3 = Arc::new(Ioc3::octane(mac, false));
        ioc3.set_heartbeat(heartbeat.clone());
        let eth = Ioc3Eth::new(mac);
        eth.set_heartbeat(heartbeat.clone());
        ioc3.attach_eth(eth.clone());
        bridge.attach(2, ioc3.clone());
        ioc3.connect_superio_interrupt(bridge.pin_line(SUPERIO_PIN));
        ioc3.set_timer_manager(timer_manager.clone());
        // The guest switching the machine off through the clock.
        ioc3.set_event_sender(event_tx.clone());
        // ... and resetting itself through HEART.
        heart.on_soft_reset(Box::new(move || {
            let _ = event_tx.try_send(MachineEvent::HardReset);
        }));
        if let Err(e) = ioc3.set_nvram_file(&cfg.nvram) {
            eprintln!("iris: Octane NVRAM {}: {e}", cfg.nvram);
        }
        let rad = Arc::new(Rad1::new());
        rad.set_timer_manager(timer_manager.clone());
        if !cfg.no_audio {
            rad.enable_output(cfg.audio.clone());
        }
        bridge.attach(3, rad.clone());
        let virtio = cfg.virtio_serial.enabled.then(|| {
            let dev = Arc::new(crate::dev::virtio::pci::VirtioPci::new(
                crate::dev::virtio::console::from_config(&cfg.virtio_serial),
            ));
            bridge.attach(VIRTIO_SLOT, dev.clone());
            eprintln!("iris: virtio-serial on Bridge slot {VIRTIO_SLOT}");
            dev
        });

        // The status bar's activity bits and the CPU's Compare matches,
        // shared with the graphics board before the CPU exists.
        let fasttick = Arc::new(AtomicU64::new(0));
        let status = crate::disp::StatusSource::new(heartbeat.clone(), fasttick.clone());
        let gfx = board.map(|b| ImpactXio::with_board(WIDGET_GFX, b, heart.clone(), mem.clone(), status.clone()));
        let vpro = if has_vpro {
            Some(Odyssey::new(WIDGET_GFX, kind.as_str(), heart.clone(), mem.clone(), status.clone())?)
        } else {
            None
        };

        // The bus over them, then the table.
        let phys = Ip30Bus::new(ram.clone());
        phys.map(PIU_BASE, PIU_BASE + PIU_SIZE, heart.clone(), "heart piu");
        phys.map(widget_small_base(WIDGET_XBOW), widget_small_base(WIDGET_XBOW) + WIDGET_SMALL_SIZE, xbow.clone(), "xbow");
        // The processor reaches its own HEART without a trip through the
        // switch.
        phys.map(widget_small_base(WIDGET_HEART), widget_small_base(WIDGET_HEART) + WIDGET_SMALL_SIZE, heart.clone(), "heart widget");
        phys.attach_widget(WIDGET_BRIDGE, bridge.clone(), false);
        if let Some(g) = &gfx {
            phys.attach_widget(WIDGET_GFX, g.clone(), false);
        }
        if let Some(v) = &vpro {
            phys.attach_widget(WIDGET_GFX, v.clone(), true);
        }
        let phys = Arc::new(phys);
        phys.init();
        Self::sync_links(&phys, &xbow);

        // Descriptor writes place memory; Crossbow link changes switch the
        // widgets' windows; a link reset of widget 15 resets the Bridge at
        // its far end. Weak, so none of these keeps the bus or the Bridge.
        // The window moves under no running CPU but the one that wrote.
        let world = Arc::new(World::new());
        let weak = Arc::downgrade(&phys);
        let remap_world = world.clone();
        heart.on_memcfg(Box::new(move |m| {
            if let Some(p) = weak.upgrade() {
                remap_world.stop_others(|| p.remap_banks(m));
            }
        }));
        let weak = Arc::downgrade(&phys);
        xbow.on_link_change(Box::new(move |w, alive| {
            if let Some(p) = weak.upgrade() {
                p.set_widget_link(w, alive);
            }
        }));
        // Device-window pages a PCI BAR decodes whole go straight to the
        // function.
        let weak = Arc::downgrade(&phys);
        bridge.on_bar_runs(Box::new(move |runs| {
            if let Some(p) = weak.upgrade() {
                p.set_widget_runs(WIDGET_BRIDGE, runs);
            }
        }));
        let weak = Arc::downgrade(&bridge);
        xbow.on_link_reset(Box::new(move |w| {
            if w == WIDGET_BRIDGE {
                if let Some(b) = weak.upgrade() {
                    b.link_reset();
                }
            }
        }));

        eprintln!(
            "iris: Octane with {} MB ({}), {}",
            banks.iter().sum::<u64>() >> 20,
            banks.iter().filter(|&&b| b != 0).map(|b| format!("{}", b >> 20)).collect::<Vec<_>>().join("+"),
            if has_vpro { kind.label() } else { board.map_or("no graphics", |b| b.label()) }
        );

        let count_hz = cfg.clock.fixed_mhz.map_or(OCTANE_COUNT_HZ, |mhz| (mhz * 1_000_000.0) as u64);
        let flash_lines: Vec<_> = (0..cpu_count)
            .map(|_| {
                let bus: Arc<dyn BusDevice> = phys.phys.clone();
                Arc::new(if cpu_count > 1 { CachedFlashLines::for_one_of_several_cpus(bus) } else { CachedFlashLines::new(bus) })
            })
            .collect();
        // CPU 0 keeps the counter the graphics board already reads.
        let mut ticks = vec![fasttick];
        ticks.extend((1..cpu_count).map(|_| Arc::new(AtomicU64::new(0))));
        let memory = Arc::new(crate::ip30::memory_map(&cfg.banks[..4]));
        let cpus: Vec<Arc<dyn CpuDevice>> = (0..cpu_count)
            .map(|id| Self::build_cpu(id, cpu_count, memory.clone(), phys.phys.clone(), flash_lines[id as usize].clone(), &timer_manager, count_hz.max(1), ticks[id as usize].clone()))
            .collect();
        let cpu = cpus[0].clone();
        world.set_slots(cpus.iter().map(|c| c.park_slot()).collect());
        // One page pool and one body of compiled code: it reaches each CPU's
        // state through the core pointer it is called with.
        #[cfg(feature = "jitv2")]
        if cpu_count > 1 {
            #[cfg(feature = "jitv2_lockstep")]
            return Err("[machine] cpus = 2 cannot run with jitv2_lockstep".into());
            let jit = cpu.jitv2();
            for c in &cpus {
                c.share_jitv2(jit.clone(), world.clone());
            }
        }
        status.set_cpu_cycles(cpu.cycles_ptr());
        status.set_count_hz(cpu.count_hz_atomic());
        for (c, tick) in cpus.iter().zip(&ticks).skip(1) {
            status.add_cpu_stats(c.cycles_ptr(), tick.clone());
        }
        if let Some(g) = &gfx {
            // Host GL frames composite straight into the IMPACT framebuffer.
            #[cfg(feature = "hostgl")]
            iris_hostcall::set_display(Arc::new(crate::dev::mgras::ImpactScreen(g.mgras().clone())));
        }

        // The shadow cache and the JIT read RAM straight from the ppmem
        // window; descriptor writes lay it out again on the CPU thread.
        if let Some(sp) = phys.phys.ppmem_space() {
            use crate::ppmem::MappedMemory;
            unsafe {
                sp.set_bitmap_sink(cpu.ppmem_bitmap_ptr());
                cpu.set_tcache_window(sp.window_base());
                cpu.set_tcache_shift(crate::ppmem::window_shift(sp.window_size()));
                let bitmap = cpu.tcache_bitmap_ptr();
                if !bitmap.is_null() {
                    sp.set_bitmap_sink2(bitmap);
                }
                #[cfg(feature = "jitv2")]
                cpu.set_tcache_gen_window(sp.gen_window_base());
                for c in cpus.iter().skip(1) {
                    sp.add_bitmap_sink(c.ppmem_bitmap_ptr());
                    c.set_tcache_window(sp.window_base());
                    c.set_tcache_shift(crate::ppmem::window_shift(sp.window_size()));
                    let bitmap = c.tcache_bitmap_ptr();
                    if !bitmap.is_null() {
                        sp.add_bitmap_sink(bitmap);
                    }
                    #[cfg(feature = "jitv2")]
                    c.set_tcache_gen_window(sp.gen_window_base());
                }
            }
            phys.phys.resync_ppmem_bitmap();
        }
        // Publish the descriptors as they stand into the sinks just set.
        ram.set_memcfg(heart.memcfg());

        // After the windows above: each compile worker takes the cache's
        // geometry once, as it starts, and without the windows that geometry
        // has no inline load or store path.
        #[cfg(feature = "jitv2")]
        {
            let cpu_device: Arc<dyn Device> = cpu.clone().as_device();
            let jit = cpu.jitv2();
            let mut g = jit.lock();
            g.compile_queue.set_thread_count(cfg.jitv2.threads.max(1));
            g.compile_queue.set_cpu(Arc::downgrade(&cpu_device));
            g.compile_queue.set_owner(Arc::downgrade(&jit));
            if cpu_count > 1 {
                g.compile_queue.set_world(world.clone());
            }
            #[cfg(not(feature = "jitv2_lockstep"))]
            {
                let stats = g.stats.clone();
                g.compile_queue.start(phys.phys.clone() as Arc<dyn BusDevice>, stats);
            }
        }

        // HEART's levels 0 to 4 are each CPU's IP2 to IP6.
        for (id, c) in cpus.iter().enumerate() {
            heart.set_cpu_interrupts(id, c.interrupts_ptr());
            heart.set_cpu_park(id, c.park_slot());
        }

        let monitor = crate::monitor::Monitor::new();
        monitor.register_device(crate::devlog::init_devlog());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(cpus.clone())));
        monitor.register_device(heart.clone());
        monitor.register_device(phys.phys.clone());
        monitor.register_device(Arc::new(Ip30Monitor {
            phys: phys.clone(),
            ram: ram.clone(),
            xbow: xbow.clone(),
            bridge: bridge.clone(),
            gfx: gfx.clone(),
        }));
        for c in &scsi {
            monitor.register_device(c.clone());
            c.register_locks();
        }
        // `ps2 type` and friends: the keyboard and mouse from the monitor.
        monitor.register_device(ioc3.ps2());
        #[cfg(feature = "perfetto")]
        {
            crate::perfetto::set_guest_cycles(Some(cpu.cycles_ptr()));
            monitor.register_device(Arc::new(crate::perfetto::PerfettoMonitor));
        }
        heart.register_locks();
        xbow.register_locks();
        bridge.register_locks();
        flash.register_locks();
        ioc3.register_locks();
        eth.register_locks();
        rad.register_locks();
        if let Some(g) = &gfx {
            monitor.register_device(g.mgras().clone());
            g.register_locks();
        }
        if let Some(v) = &vpro {
            monitor.register_device(v.clone());
        }
        let monitor_port = cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT);
        let ci_serial = cfg.ci.then(|| Arc::new(crate::dev::serial_host::CiSerialBackend::new()));
        Self::connect_serial(cfg, &ioc3, ci_serial.as_ref());

        // The Ethernet on the NAT gateway `[network]` describes.
        let (mut nat, mut frames) = (None, None);
        let net = crate::net::GatewayConfig::from_network(&cfg.network());
        if let Some(link) = crate::net::Link::with_gateway(net.clone())? {
            if net.mode == crate::config::NetMode::Stream {
                eprintln!("iris: Octane Ethernet on {}", link.describe());
            } else {
                eprintln!("iris: Octane Ethernet on NAT: gateway {}, guest {} (DHCP)", net.gateway_ip, net.client_ip);
            }
            frames = Some(link.frames());
            nat = Some((link.nat(), (net.client_ip, net.gateway_ip, net.netmask)));
            eth.attach(link);
        }
        Ok(Self {
            memory,
            phys,
            heart,
            ram,
            xbow,
            cpus,
            flash_lines,
            timer_manager,
            monitor: Arc::new(monitor),
            monitor_port,
            ci_serial,
            bridge,
            ioc3,
            scsi,
            eth,
            rad,
            virtio,
            flash,
            gfx,
            vpro,
            nat,
            frames,
            scratch_path,
            nvram_path: cfg.nvram.clone(),
            event_rx: Some(event_rx),
            controller_slot: None,
            lifecycle: crate::machine_state::GuestState::Idle,
            disks,
            last_restore: None,
            saves: std::path::PathBuf::from("saves"),
        })
    }

    /// Point every switched port's windows at its widget or at the bus
    /// error, as the Crossbow's links stand.
    fn sync_links(phys: &Ip30Bus, xbow: &Xbow) {
        for w in 9..16 {
            phys.set_widget_link(w, xbow.carries(w));
        }
    }

    /// The `[scsi.N]` devices: `controller` 0 is the internal bus, 1 the
    /// external one. Returns the `scratch = true` volume, if any, and the
    /// disks as a snapshot records them.
    fn attach_scsi(cfg: &MachineConfig, buses: [&crate::scsi::ScsiBus; 2]) -> Result<(Option<std::path::PathBuf>, Vec<crate::snapshot::DiskRef>), String> {
        let attached = crate::scsi::attach_from_config(cfg, Some("ip30"), &|c| {
            buses.get(c as usize)
                .copied()
                .ok_or_else(|| format!("controller {c} (an Octane has controllers 0 and 1)"))
        })?;
        if let Some(d) = attached.daynaports.first() {
            return Err(format!("{}: kind = \"daynaport\" is not available on the Octane", crate::config::scsi_label(d.controller, d.id)));
        }
        let disks = attached.tables.iter()
            .filter(|t| !t.is_cdrom)
            .map(|t| crate::snapshot::DiskRef { id: t.slot, path: t.path.clone(), size_bytes: t.size_bytes })
            .collect();
        Ok((attached.scratch, disks))
    }

    /// IOC3's first port is the console: the CI socket's backend in CI
    /// mode, otherwise TCP on `serial_port_a`, either teed to `serial_log`.
    /// The second port serves TCP on `serial_port_b`.
    fn connect_serial(cfg: &MachineConfig, ioc3: &Ioc3, ci: Option<&Arc<crate::dev::serial_host::CiSerialBackend>>) {
        let (console, second) = crate::dev::serial_host::console_backends(cfg, "Octane", ci);
        ioc3.set_serial_backend(0, console);
        if let Some(b) = second {
            ioc3.set_serial_backend(1, b);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_cpu(
        id: u8,
        cpu_count: u8,
        memory: Arc<crate::memory_map::MemoryMap>,
        sysad: Arc<dyn BusDevice>,
        cached: Arc<CachedFlashLines>,
        tm: &Arc<TimerManager>,
        count_hz: u64,
        fasttick: Arc<AtomicU64>,
    ) -> Arc<dyn CpuDevice> {
        let cfg = MipsCpuConfig::for_model::<R10000ShadowCache>(memory);
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let mut exec: MipsExecutor<MipsTlb, R10000ShadowCache> = MipsExecutor::new(sysad, tlb, &cfg);
        // Cached accesses see the PROM's dirty flash lines; see `CachedFlashLines`.
        exec.cache = R10000ShadowCache::from(cached as Arc<dyn BusDevice>);
        exec.core.set_timer_manager(tm.clone());
        exec.core.set_count_hz(count_hz);
        exec.core.fasttick_count = fasttick;
        exec.core.reset_prid = OCTANE_PRID;
        exec.core.cp0_prid = OCTANE_PRID;
        // IRIX zeroes Count and arms Compare a few instructions later; see
        // `MipsCore::count_write_samples`.
        exec.core.count_write_samples = true;
        // The R10000 drives 40 physical address bits; HEART decodes them
        // all, and the attribute bits above fold away.
        exec.phys_mask = (1 << 40) - 1;
        exec.core.cpu_id = id;
        exec.core.smp = cpu_count > 1;
        exec.batch_hook = Some(Box::new(move |exec: &mut MipsExecutor<MipsTlb, R10000ShadowCache>| {
            LAST_PC[id as usize].store(exec.core.pc, Ordering::Relaxed);
        }));
        Arc::new(MipsCpu::new(exec))
    }

    /// The devices with threads or time of their own, in start order; stop
    /// runs it backwards.
    fn devices(&self) -> Vec<Arc<dyn Device>> {
        let mut d: Vec<Arc<dyn Device>> = vec![
            self.heart.clone(),
            self.ioc3.clone(),
            self.eth.clone(),
            self.scsi[0].clone(),
            self.scsi[1].clone(),
            self.rad.clone(),
        ];
        if let Some(v) = &self.virtio {
            d.push(v.clone());
        }
        if let Some(g) = &self.gfx {
            d.push(g.clone());
        }
        if let Some(v) = &self.vpro {
            d.push(v.clone());
        }
        d
    }

    /// Reverse of start order, but graphics last: its stop joins a display
    /// thread, which can fail to end, and the disks must be stopped by then.
    fn stop_devices(&self) {
        // `devices` ends with the graphics board, if there is one.
        for d in self.devices().iter().rev().skip(self.gfx.iter().count()) {
            d.stop();
        }
        if let Some(g) = &self.gfx {
            g.stop();
        }
    }

    pub fn start(&mut self) {
        if let Ok(starting) = self.lifecycle.transition(crate::machine_state::GuestState::Starting) {
            self.lifecycle = starting;
        }
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        // CI mode: the harness starts the CPU (`iris-ci start`).
        if self.ci_serial.is_none() {
            self.start_cpus();
        }
    }

    /// Stop the CPU, then every device: their threads are joined and their
    /// timers removed, so nothing touches memory or an interrupt line until
    /// [`Self::restart_peripherals`]. HEART's counter holds still meanwhile
    /// and IOC3 writes its NVRAM out.
    pub fn stop(&mut self) {
        self.stop_cpus();
        self.stop_devices();
        // The guest clock points into this machine's CPU core.
        #[cfg(feature = "perfetto")]
        crate::perfetto::set_guest_cycles(None);
        if self.lifecycle != crate::machine_state::GuestState::Idle {
            self.lifecycle = crate::machine_state::GuestState::Halted;
        }
    }

    /// The hardware as at power-on, with the CPU stopped: memory and the
    /// flash keep their contents, the cached flash lines are lost, every
    /// PCI function's header returns to power-up (RST#).
    pub fn reset(&mut self) {
        self.stop();
        self.power_on_all();
        self.restart_peripherals();
    }

    /// Every device as at power-on, with the machine stopped.
    fn power_on_all(&self) {
        for c in &self.cpus {
            Resettable::power_on(&**c);
        }
        self.xbow.power_on();
        self.heart.power_on();
        for f in &self.flash_lines {
            f.clear();
        }
        self.bridge.reset();
        self.bridge.reset_pci();
        self.ioc3.power_on();
        self.eth.power_on();
        for c in &self.scsi {
            c.reset();
        }
        self.rad.reset();
        if let Some(v) = &self.virtio {
            Resettable::power_on(&**v);
        }
        self.flash.reset();
        if let Some(g) = &self.gfx {
            g.power_on();
        }
        if let Some(v) = &self.vpro {
            v.power_on();
        }
        Self::sync_links(&self.phys, &self.xbow);
    }

    /// Start what `stop` stopped, short of the CPU: HEART's counter, the
    /// device threads and timers, and the graphics engines and display.
    /// Without the last the PROM's first graphics readback after a reset
    /// waits forever on a pipeline nothing drains.
    pub fn restart_peripherals(&mut self) {
        for d in self.devices() {
            d.start();
        }
    }

    fn start_cpus(&self) {
        for c in &self.cpus {
            c.clone().as_device().start();
        }
    }

    fn stop_cpus(&self) {
        for c in &self.cpus {
            c.clone().as_device().stop();
        }
    }

    pub fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut OctaneMachine as *mut dyn Workstation;
        let slot = self.controller_slot.get_or_insert_with(|| Arc::new(parking_lot::Mutex::new(MachineRef(ptr))));
        debug_assert!(std::ptr::addr_eq(slot.lock().0, ptr), "OctaneMachine moved after its slot was handed out");
        slot.clone()
    }
}

// Snapshots of the Octane, in the O2's format: a manifest naming the
// machine, the CPU and the board's devices as postcard tables, all RAM as
// one chunk list in the shared store, the boot flash's bytes, and each SCSI
// controller's copy-on-write overlays. Written to `saves/<name>/` under the
// current directory, like the other machines' snapshots.
//
// The snapshot point is the machine's stop: the CPU threads have exited,
// and the devices' threads are joined and their timers removed. Host-clock
// state in the devices is saved relative to the stop, so the guest's clocks
// resume where they stopped.
//
// Not supported, and refused with the reason: an IMPACT or a VPro board
// (neither has saved state yet) and two processors (the stop
// does not yet bring both to a common instruction boundary that a snapshot
// can record). In a jitv2 build, putting memory back bumps every page's
// generation, so code compiled from the memory before the load is not run.

/// Why an Octane with an IMPACT board, a VPro board or `cpus` processors
/// cannot be saved, when it cannot.
fn snapshot_refusal_for(impact: bool, vpro: bool, cpus: usize) -> Option<String> {
    if impact {
        return Some("snapshots of an Octane with an IMPACT board are not supported yet: \
            its engines and display threads have no saved state".into());
    }
    if vpro {
        return Some("snapshots of an Octane with a VPro board are not supported yet: \
            the board has no saved state".into());
    }
    if cpus > 1 {
        return Some("snapshots of a two-processor Octane are not supported yet: \
            the stop does not bring both processors to one recorded instruction boundary".into());
    }
    None
}

/// The `machine` an Octane snapshot's manifest names.
pub(crate) const SNAPSHOT_MACHINE: &str = "octane_ip30";

/// One SCSI controller's copy-on-write sets: each target's dirty sectors.
type OverlaySets = Vec<(usize, Vec<u64>)>;

/// A snapshot read and checked, ready to put into the machine.
struct OctaneSnapshot {
    cpu: toml::Value,
    board: toml::Value,
    flash: Vec<u8>,
    memory: Vec<u32>,
    overlays: [Option<OverlaySets>; 2],
    dir: std::path::PathBuf,
}

impl OctaneMachine {
    /// Why this machine cannot be saved, when it cannot.
    fn snapshot_refusal(&self) -> Option<String> {
        snapshot_refusal_for(self.gfx.is_some(), self.vpro.is_some(), self.cpus.len())
    }

    /// Every device's state but the CPU's. Called with the machine stopped.
    fn save_board(&self) -> toml::Value {
        let lines = self.flash_lines.iter().map(|f| f.save_state()).collect();
        StateTable::new()
            .value("ioc3", self.ioc3.save_state())
            .value("eth", self.eth.save_state())
            .value("scsi0", self.scsi[0].save_state())
            .value("scsi1", self.scsi[1].save_state())
            .value("rad", self.rad.save_state())
            .value("flash", self.flash.save_state())
            .value("flash_lines", toml::Value::Array(lines))
            .value("bridge", self.bridge.save_state())
            .value("xbow", self.xbow.save_state())
            .value("heart", self.heart.save_state())
            .build()
    }

    /// Put back what [`Self::save_board`] gave, into a machine stopped and
    /// reset: leaf devices first and HEART last, so the pins it drives into
    /// the CPU, already loaded, come from its loaded registers, and memory
    /// is placed from its loaded descriptors.
    fn load_board(&self, v: &toml::Value) -> Result<(), String> {
        let load = |key: &str, dev: &dyn Saveable| dev.load_state(field(v, key)?).map_err(|e| format!("{key}: {e}"));
        load("ioc3", &*self.ioc3)?;
        load("eth", &*self.eth)?;
        load("scsi0", &*self.scsi[0])?;
        load("scsi1", &*self.scsi[1])?;
        load("rad", &*self.rad)?;
        load("flash", &*self.flash)?;
        let lines = field(v, "flash_lines")?.as_array().filter(|l| l.len() == self.flash_lines.len())
            .ok_or("flash_lines: not one entry per processor")?;
        for (f, saved) in self.flash_lines.iter().zip(lines) {
            f.load_state(saved)?;
        }
        load("bridge", &*self.bridge)?;
        load("xbow", &*self.xbow)?;
        Self::sync_links(&self.phys, &self.xbow);
        load("heart", &*self.heart)?;
        Ok(())
    }

    pub fn save_snapshot(&mut self, name: &str) -> Result<(), String> {
        if let Some(why) = self.snapshot_refusal() {
            return Err(why);
        }
        let lifecycle = self.lifecycle;
        let was_running = self.cpus[0].clone().as_device().is_running();
        self.stop();
        let saved = self.write_snapshot(name);
        if lifecycle != crate::machine_state::GuestState::Idle {
            self.restart_peripherals();
        }
        if was_running {
            self.start_cpus();
        }
        self.lifecycle = lifecycle;
        saved?;
        println!("Snapshot saved to {}/{name}", self.saves.display());
        Ok(())
    }

    fn write_snapshot(&self, name: &str) -> Result<(), String> {
        let cpu = &self.cpus[0];
        let snap = Snapshot::new(self.saves.join(name));
        snap.ensure_dir().map_err(|e| e.to_string())?;
        let mut manifest = Manifest::for_current_save();
        manifest.machine = Some(SNAPSHOT_MACHINE.into());
        manifest.parent = self.last_restore.clone();
        manifest.disks = self.disks.clone();
        manifest.nvram = Some(self.nvram_path.clone());
        manifest.cpu_model = Some(cpu.model_name().to_string());
        manifest.cpu_cache = Some(cpu.cache_kind().to_string());
        snap.write_manifest(&manifest).map_err(|e| e.to_string())?;
        let sv = manifest.schema_version;
        let io = |e: std::io::Error| e.to_string();
        snap.write_state("cpu", &cpu.save_state(), sv).map_err(io)?;
        snap.write_state("octane", &self.save_board(), sv).map_err(io)?;
        snap.write_bin("flash.bin", &self.flash.contents()).map_err(io)?;
        let store = ChunkStore::new(&self.saves);
        let mut chunks = ChunksManifest::default();
        for words in self.ram.bank_words() {
            chunks.bank_chunks[0].extend(put_words_as_chunks(&store, words).map_err(|e| format!("CAS memory put: {e}"))?);
        }
        snap.write_chunks_manifest(&chunks).map_err(io)?;
        for (i, ctl) in self.scsi.iter().enumerate() {
            let dir = snap.dir.join(format!("ctrl{i}"));
            std::fs::create_dir_all(&dir).map_err(io)?;
            let overlays = ctl.bus().export_overlays(&dir)
                .map_err(|e| format!("COW overlay export (controller {i}): {e}"))?;
            snap.write_toml(&format!("cow{i}.toml"), &crate::machine::build_cow_table(overlays)).map_err(io)?;
        }
        Ok(())
    }

    /// Restore from `saves/<name>/` and run. The snapshot is read and its
    /// machine, build, CPU, disks and memory size checked before the machine
    /// is touched; a mismatch found while the devices load (another function
    /// in a Bridge slot, other crossbow links, another MicroLAN part) leaves
    /// the machine reset and stopped.
    pub fn load_snapshot(&mut self, name: &str) -> Result<(), String> {
        if let Some(why) = self.snapshot_refusal() {
            return Err(why);
        }
        let snap = self.read_snapshot(name)?;
        self.stop();
        let applied = self.apply_snapshot(snap);
        self.restart_peripherals();
        applied?;
        self.start_cpus();
        self.lifecycle = crate::machine_state::GuestState::Running;
        println!("Snapshot loaded from {}/{name}", self.saves.display());
        Ok(())
    }

    fn read_snapshot(&self, name: &str) -> Result<OctaneSnapshot, String> {
        let cpu = &self.cpus[0];
        let snap = Snapshot::new(self.saves.join(name));
        let m = snap.read_manifest()?.ok_or_else(|| format!("{}/{name}: no snapshot.toml", self.saves.display()))?;
        if m.machine.as_deref() != Some(SNAPSHOT_MACHINE) {
            return Err(format!("snapshot {name} is not of an Octane"));
        }
        if m.host_arch != std::env::consts::ARCH {
            return Err(format!("snapshot host_arch '{}' does not match current host '{}'", m.host_arch, std::env::consts::ARCH));
        }
        if m.schema_version != SCHEMA_VERSION {
            return Err(format!("snapshot {name} is schema {}; this build reads Octane snapshots of schema {SCHEMA_VERSION}", m.schema_version));
        }
        // As the Indy and the O2: build, CPU and disks must match.
        crate::snapshot::check_provenance(&m, cpu.model_name(), cpu.cache_kind(), &self.disks)?;
        let sv = m.schema_version;
        let state = |base: &str| snap.read_state(base, sv).map_err(|e| format!("{base}: {e}"));
        let chunks = snap.read_chunks_manifest().map_err(|e| format!("chunks.bin: {e}"))?;
        let memory = get_chunks_as_words(&ChunkStore::new(&self.saves), &chunks.bank_chunks[0])
            .map_err(|e| format!("CAS memory get: {e}"))?;
        let fitted: usize = self.ram.bank_words().iter().map(|b| b.len()).sum();
        if memory.len() != fitted {
            return Err(format!("snapshot memory is {} MiB, this Octane has {} MiB", memory.len() >> 18, fitted >> 18));
        }
        let overlays = std::array::from_fn(|i| {
            snap.read_toml(&format!("cow{i}.toml")).ok().map(|t| crate::machine::parse_cow_table(&t))
        });
        // Checked here, before the machine is touched, rather than when the
        // part is loaded.
        let flash = snap.read_bin("flash.bin").map_err(|e| format!("flash.bin: {e}"))?;
        if flash.len() != self.flash.size() {
            return Err(format!("saved flash is {} bytes, this Octane's holds {}", flash.len(), self.flash.size()));
        }
        Ok(OctaneSnapshot {
            cpu: state("cpu")?,
            board: state("octane")?,
            flash,
            memory,
            overlays,
            dir: snap.dir,
        })
    }

    /// Put a read snapshot into the stopped machine: the hardware as at
    /// power-on first, so nothing the snapshot does not hold is left over.
    fn apply_snapshot(&mut self, s: OctaneSnapshot) -> Result<(), String> {
        self.power_on_all();
        self.cpus[0].load_state(&s.cpu).map_err(|e| format!("cpu: {e}"))?;
        self.flash.restore_contents(&s.flash)?;
        self.ram.restore_bank_words(&s.memory)?;
        // The overlays first: an import can take minutes, and the devices
        // count their deadlines from their load, so they load last, just
        // before the machine starts.
        for (i, ctl) in self.scsi.iter().enumerate() {
            if let Some(sets) = &s.overlays[i] {
                ctl.bus().import_overlays(&s.dir.join(format!("ctrl{i}")), sets)
                    .map_err(|e| format!("COW overlay import (controller {i}): {e}"))?;
            }
        }
        self.load_board(&s.board)
    }
}

impl Drop for OctaneMachine {
    fn drop(&mut self) {
        if let Some(slot) = self.controller_slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop_cpus();
        self.stop_devices();
        crate::workstation::flush_on_drop(self);
        self.monitor.shutdown();
        // The bus holds the devices, and the Bridge and IMPACT hold its
        // memory, which owns the bus: let go, or none of it is freed.
        self.phys.phys.release_devices();
    }
}

impl Workstation for OctaneMachine {
    fn start(&mut self) {
        OctaneMachine::start(self)
    }
    fn stop(&mut self) {
        OctaneMachine::stop(self)
    }
    fn reset(&mut self) {
        OctaneMachine::reset(self)
    }
    fn restart_peripherals(&mut self) {
        OctaneMachine::restart_peripherals(self)
    }
    fn cpu_start(&self) {
        self.start_cpus()
    }
    fn cpu_is_running(&self) -> bool {
        self.cpus.iter().any(|c| c.clone().as_device().is_running())
    }
    fn machine_slot(&mut self) -> MachineSlot {
        OctaneMachine::machine_slot(self)
    }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self.event_rx.take();
        assert!(events.is_some(), "register_system_controller called twice");
        crate::workstation::register_system_controller(&self.monitor, slot, events);
    }
    fn monitor_addr(&self) -> Option<std::net::SocketAddr> {
        self.monitor.bound_addr()
    }
    fn lifecycle(&self) -> crate::machine_state::GuestState {
        self.lifecycle
    }
    fn set_lifecycle(&mut self, state: crate::machine_state::GuestState) {
        self.lifecycle = state;
    }
    fn get_display(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> {
        match &self.vpro {
            Some(v) => Some(v.display()),
            None => self.gfx.as_ref().map(|g| g.display()),
        }
    }
    fn register_monitor_device(&self, device: Arc<dyn crate::traits::Device>) { self.monitor.register_device(device); }
    fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> {
        self.ioc3.ps2()
    }
    fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timer_manager.clone()
    }
    fn get_cpu_debug(&self) -> Arc<dyn crate::gdb_stub::CpuDebug> {
        self.cpus[0].clone().debug_adapter()
    }
    fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.cpus[0].fasttick_count()
    }
    fn cpu_counters(&self) -> Vec<(crate::cpu::mips_core::CyclesPtr, Arc<AtomicU64>)> {
        self.cpus.iter().map(|c| (c.cycles_ptr(), c.fasttick_count())).collect()
    }
    fn get_ci_serial(&self) -> Option<Arc<crate::dev::serial_host::CiSerialBackend>> {
        self.ci_serial.clone()
    }
    fn inject_serial_console(&self, bytes: &[u8]) {
        if let Some(ci) = &self.ci_serial {
            ci.push_host(bytes);
        }
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn save_snapshot(&mut self, name: &str) -> Result<(), String> {
        OctaneMachine::save_snapshot(self, name)
    }
    fn load_snapshot(&mut self, name: &str) -> Result<(), String> {
        OctaneMachine::load_snapshot(self, name)
    }
    fn ci_restore(&mut self, name: &str) -> Result<(), String> {
        // Clear what the guest printed before, so the next command does
        // not see stale output; the Indy does the same.
        if let Some(ci) = &self.ci_serial {
            ci.reset();
        }
        OctaneMachine::load_snapshot(self, name)?;
        self.last_restore = Some(name.to_string());
        Ok(())
    }
    /// Back to the last restored snapshot, read again from disk.
    fn ci_rollback(&mut self) -> Result<(), String> {
        let name = self.last_restore.clone().ok_or("rollback: nothing has been restored")?;
        if let Some(ci) = &self.ci_serial {
            ci.reset();
        }
        OctaneMachine::load_snapshot(self, &name)
    }
    /// The DS1685's RAM, which holds the PROM's environment, written out
    /// now. The flash writes itself through to its file.
    fn save_nvram(&self, _path: Option<&str>) -> Result<String, String> {
        if self.nvram_path.is_empty() {
            return Err("no nvram file configured".into());
        }
        self.ioc3.flush_nvram();
        Ok(self.nvram_path.clone())
    }
    fn scratch_path(&self) -> Option<&std::path::Path> {
        self.scratch_path.as_deref()
    }
    /// Disc changes on the internal bus.
    fn disc_changer(&self) -> Option<Arc<dyn crate::workstation::DiscChanger>> {
        Some(self.scsi[0].clone() as Arc<dyn crate::workstation::DiscChanger>)
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.scsi.iter().map(|c| c.bus().pending_chd_sync_count()).sum()
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.scsi.iter().try_for_each(|c| c.bus().flush())
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        let buses: Vec<_> = self.scsi.iter().map(|c| c.bus()).collect();
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

/// The monitor's `ip30` command: the board as a whole.
struct Ip30Monitor {
    phys: Arc<Ip30Bus>,
    ram: Arc<HeartRam>,
    xbow: Arc<Xbow>,
    bridge: Arc<Bridge>,
    gfx: Option<Arc<ImpactXio>>,
}

const IP30_SUBCOMMANDS: &str = "ip30 unmapped | memcfg | bridge | gfx | xbow";

impl Ip30Monitor {
    fn text(&self, sub: &str) -> Result<String, String> {
        use crate::dev::bridge::reg;
        Ok(match sub {
            "unmapped" => self
                .phys
                .unmapped()
                .iter()
                .map(|(a, wr, pc)| format!("{} {a:#x} pc {pc:#x}\n", if *wr { "w" } else { "r" }))
                .collect(),
            "memcfg" => {
                let cfg = self.ram.memcfg();
                let mut s = format!(
                    "descriptors {:08x} {:08x} {:08x} {:08x}; {} MB fitted\n",
                    cfg[0],
                    cfg[1],
                    cfg[2],
                    cfg[3],
                    self.ram.total() >> 20
                );
                for (base, span, real) in self.ram.layout() {
                    s.push_str(&format!("  {base:#011x} span {} MB, DIMMs {} MB\n", span >> 20, real >> 20));
                }
                s
            }
            "bridge" => {
                let (wid, addr) = self.bridge.interrupt_destination();
                let mut s = format!(
                    "interrupt destination widget {wid} address {addr:#x}; enable {:#04x} status {:#010x}\n",
                    self.bridge.register(reg::INT_ENABLE),
                    self.bridge.register(reg::INT_STATUS)
                );
                for n in 0..8 {
                    s.push_str(&format!(
                        "  device {n}: {:#010x}  pin {n} vector {:#04x}\n",
                        self.bridge.register(reg::device(n)),
                        self.bridge.register(reg::int_host(n))
                    ));
                }
                s
            }
            "gfx" => match &self.gfx {
                Some(g) => format!(
                    "IMPACT on widget {WIDGET_GFX}: link {}, identification {:#010x}\n",
                    if self.xbow.carries(WIDGET_GFX) { "up" } else { "down" },
                    g.read32(widget_small_base(WIDGET_GFX) + 4).data
                ),
                None => "no graphics board\n".into(),
            },
            "xbow" => self
                .xbow
                .unsupported()
                .iter()
                .map(|(o, s, wr)| format!("{} {s} bytes at {o:#x}\n", if *wr { "write" } else { "read" }))
                .collect(),
            _ => return Err(format!("Usage: {IP30_SUBCOMMANDS}")),
        })
    }
}

impl Device for Ip30Monitor {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool {
        true
    }
    fn get_clock(&self) -> u64 {
        0
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("ip30".into(), format!("Octane board: {IP30_SUBCOMMANDS}"))]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "ip30" {
            return Err("Command not found".into());
        }
        let text = self.text(args.first().copied().unwrap_or("unmapped"))?;
        w.write_all(text.as_bytes()).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The VPro fitment check reads the PROM's revision from its banner;
    /// a board that needs a newer PROM than the image warns.
    #[test]
    fn the_prom_banner_gives_its_revision() {
        let banner = b"\0\0SGI Version 6.5 Rev 4.9 IP30 Jan 20, 2004\0";
        assert_eq!(prom_revision(banner), Some((4, 9)));
        assert_eq!(prom_revision(b"no banner here"), None);
        assert!((4, 3) < vpro_prom_needed(crate::config::GraphicsBoard::V12));
        assert!((4, 3) >= vpro_prom_needed(crate::config::GraphicsBoard::V6));
    }

    /// A dropped Octane frees its bus and with it the decode table: a
    /// process may have only a few dozen tables, and a test binary builds
    /// many machines.
    #[test]
    fn a_dropped_octane_frees_its_decode_table() {
        use crate::config::{GraphicsBoard, MachineProfile};
        use crate::display_stop_tests::{big_stack, config, scratch};
        let dir = scratch("octane-drop");
        for board in [GraphicsBoard::None, GraphicsBoard::SolidImpact] {
            let cfg = config(&dir, MachineProfile::OctaneIp30, board);
            let machine = big_stack(move || OctaneMachine::new(&cfg).expect("Octane"));
            let phys = Arc::downgrade(&machine.phys.phys);
            drop(machine);
            assert_eq!(phys.strong_count(), 0, "{board:?}: the bus outlives its machine");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::cpu::mips_exec::CpuStateDigest;

    /// The stock PROM is not in the repository; without it the test that
    /// runs it says so and passes.
    fn prom() -> Option<String> {
        let path = std::env::var("IRIS_IP30_PROM").unwrap_or_else(|_| "/inputs/IP30prom.bin".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("skipped: no stock PROM at {path}");
            return None;
        }
        Some(path)
    }

    /// A test's own directory, removed when the test ends, failed or not.
    struct Scratch(std::path::PathBuf);

    impl std::ops::Deref for Scratch {
        type Target = std::path::Path;
        fn deref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("iris-octane-snap-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }

    /// A one-processor Octane with 256 MiB, no graphics, no network and a
    /// small overlay disk.
    fn machine(prom: &str, dir: &std::path::Path) -> OctaneMachine {
        machine_with(prom, dir, |_| {})
    }

    fn machine_with(prom: &str, dir: &std::path::Path, edit: impl FnOnce(&mut MachineConfig)) -> OctaneMachine {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = crate::config::MachineProfile::OctaneIp30;
        cfg.machine.cpu = CpuModel::R10000;
        cfg.banks = [256, 0, 0, 0, 0, 0, 0, 0];
        cfg.prom = prom.into();
        cfg.nvram = dir.join("nvram.bin").to_string_lossy().into_owned();
        cfg.headless = true;
        cfg.graphics.board = Some(crate::config::GraphicsBoard::None);
        // Not CI mode: its overlays share one path per process, and the
        // tests run in parallel; this way each sits beside its own disk.
        cfg.network.mode = crate::config::NetMode::Off;
        let held: Vec<_> = (0..3).map(|_| std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap()).collect();
        let port = |i: usize| Some(held[i].local_addr().unwrap().port());
        cfg.monitor_port = port(0);
        cfg.serial_port_a = port(1);
        cfg.serial_port_b = port(2);
        let disk = dir.join("disk.raw");
        if !disk.exists() {
            std::fs::write(&disk, vec![0u8; 4 << 20]).unwrap();
        }
        cfg.scsi.clear();
        cfg.scsi.insert(1, toml::from_str(&format!("path = {:?}\noverlay = true", disk.to_string_lossy())).unwrap());
        edit(&mut cfg);
        let mut m = OctaneMachine::new(&cfg).expect("an Octane");
        m.saves = dir.join("saves");
        m
    }

    fn digest(m: &OctaneMachine) -> (CpuStateDigest, blake3::Hash) {
        let mut h = blake3::Hasher::new();
        for words in m.ram.bank_words() {
            // SAFETY: a byte view of the same words.
            h.update(unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 4) });
        }
        (m.cpus[0].state_digest().unwrap(), h.finalize())
    }

    /// Run the PROM some way, save, restore into a fresh machine: the CPU
    /// and memory come back exactly, and both machines agree a million
    /// instructions on.
    #[test]
    fn a_restored_octane_continues_from_the_saved_state() {
        let Some(prom) = prom() else { return };
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || {
            let dir = scratch("continue");
            let mut a = machine(&prom, &dir);
            a.cpus[0].step_n_inline(2_000_000).unwrap();
            a.save_snapshot("at-n").unwrap();
            let saved = digest(&a);
            let mut b = machine(&prom, &dir);
            let snap = b.read_snapshot("at-n").unwrap();
            b.apply_snapshot(snap).unwrap();
            assert_eq!(digest(&b), saved, "the restore puts back the CPU and memory");
            assert_eq!(a.cpus[0].save_state(), b.cpus[0].save_state());
            a.cpus[0].step_n_inline(1_000_000).unwrap();
            b.cpus[0].step_n_inline(1_000_000).unwrap();
            assert_eq!(digest(&a), digest(&b));
        }).unwrap().join().unwrap();
    }

    /// An empty PROM in place of the stock one, so these run anywhere.
    fn empty_prom(dir: &std::path::Path) -> String {
        let path = dir.join("prom.bin");
        std::fs::write(&path, vec![0u8; 1 << 20]).unwrap();
        path.to_string_lossy().into_owned()
    }

    /// Without the stock PROM: a machine that has moved on takes back the
    /// CPU, memory and every device's state a save wrote, after the CPU had
    /// run and RAM had been written. (The same machine rather than a second
    /// one: every Octane a test builds holds one of the process's 64
    /// device-map tables until the process ends.)
    #[test]
    fn an_octane_on_an_empty_prom_takes_back_its_state() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("empty-prom");
            let prom = empty_prom(&dir);
            let mut a = machine(&prom, &dir);
            let pattern = |a: &OctaneMachine, salt: u32| {
                let mut words: Vec<u32> = a.ram.bank_words().concat();
                for (i, w) in words.iter_mut().enumerate().step_by(4099) {
                    *w = salt | i as u32;
                }
                a.ram.restore_bank_words(&words).unwrap();
            };
            // Move the CPU off reset (the empty PROM runs as no-ops) and put
            // a pattern in RAM, so a load that missed either shows.
            a.cpus[0].step_n_inline(1000).unwrap();
            pattern(&a, 0x5a5a_0000);
            a.save_snapshot("bare").unwrap();
            let (at_save, cpu_at_save) = (digest(&a), a.cpus[0].save_state());
            let saved = a.read_snapshot("bare").unwrap();
            let board = saved.board.clone();
            // Move on, then load what was saved.
            a.cpus[0].step_n_inline(1000).unwrap();
            pattern(&a, 0xa5a5_0000);
            assert_ne!(digest(&a), at_save);
            a.stop();
            a.apply_snapshot(saved).unwrap();
            assert_eq!(digest(&a), at_save);
            assert_eq!(a.cpus[0].save_state(), cpu_at_save);
            // Two clocks run in host time while the machine is stopped: the
            // PHY's negotiation countdown, whose time left can only have
            // shrunk since the load, and the RTC, which keeps the host's time
            // of day as on every machine. Everything else comes back as saved.
            let (mut board, mut got) = (board, a.save_board());
            let left = |t: &toml::Value| crate::snapshot::field_u64(&t["eth"]["phy"], "deadline").unwrap();
            assert!(left(&got) <= left(&board));
            for t in [&mut board, &mut got] {
                t["eth"]["phy"].as_table_mut().unwrap().remove("deadline");
                t["ioc3"]["chip"].as_table_mut().unwrap().remove("rtc");
            }
            assert_eq!(got, board);
        }).unwrap().join().unwrap();
    }

    /// Each case a save or load refuses gives its reason. (Decided without
    /// building a machine for each: every Octane a test builds holds one of
    /// the process's 64 device-map tables until the process ends.)
    #[test]
    fn each_refused_case_gives_its_reason() {
        assert_eq!(snapshot_refusal_for(false, false, 1), None);
        assert!(snapshot_refusal_for(true, false, 1).unwrap().contains("IMPACT"));
        assert!(snapshot_refusal_for(false, true, 1).unwrap().contains("VPro"));
        assert!(snapshot_refusal_for(false, false, 2).unwrap().contains("two-processor"));
    }

    /// A machine applies the refusal to both save and load.
    #[test]
    fn an_impact_board_is_refused_with_the_reason() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("impact");
            let prom = empty_prom(&dir);
            let mut m = machine_with(&prom, &dir, |cfg| cfg.graphics.board = Some(crate::config::GraphicsBoard::SolidImpact));
            let e = m.save_snapshot("x").unwrap_err();
            assert!(e.contains("IMPACT"), "{e}");
            let e = m.load_snapshot("x").unwrap_err();
            assert!(e.contains("IMPACT"), "{e}");
        }).unwrap().join().unwrap();
    }

    /// A snapshot of another machine, or with a boot flash of another size,
    /// is refused before the machine is touched: it carries on as it was.
    /// (One machine for both: every Octane a test builds holds one of the
    /// process's 64 device-map tables until the process ends.)
    #[test]
    fn a_snapshot_that_does_not_fit_is_refused_before_the_machine_is_touched() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("refuse");
            let prom = empty_prom(&dir);
            let mut a = machine(&prom, &dir);
            a.save_snapshot("other").unwrap();
            a.save_snapshot("flash").unwrap();
            let path = dir.join("saves/other/snapshot.toml");
            let manifest = std::fs::read_to_string(&path).unwrap().replace("octane_ip30", "o2_ip32");
            std::fs::write(&path, manifest).unwrap();
            std::fs::write(dir.join("saves/flash/flash.bin"), vec![0u8; 4096]).unwrap();
            a.cpus[0].step_n_inline(100).unwrap();
            let before = digest(&a);
            assert!(a.load_snapshot("other").unwrap_err().contains("not of an Octane"));
            let e = a.load_snapshot("flash").unwrap_err();
            assert!(e.contains("flash"), "{e}");
            assert_eq!(digest(&a), before, "the machine is as it was");
        }).unwrap().join().unwrap();
    }
}
