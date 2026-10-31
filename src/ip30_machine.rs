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

use crate::config::{CpuModel, MachineConfig};
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
use crate::traits::{BusDevice, Device, MachineEvent, Resettable};
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
/// 1.4 on an Octane2, recorded in jamesbraid/qemu-sgi-origami's
/// docs/system/mips/sgi/ip30.rst). SGI's VPro upgrade kit (007-4210-003)
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
        if prom.len() > FLASH_SIZE {
            return Err(format!("Octane PROM {prom_path} is {} bytes; the flash holds {FLASH_SIZE}", prom.len()));
        }
        let cpu_count = cfg.machine.cpus;
        if !(1..=2).contains(&cpu_count) {
            return Err(format!("[machine] cpus = {cpu_count}: IRIX runs an Octane with 1 or 2"));
        }
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
        // The Octane2 is the machine that takes a V10 or V12, as the VPro
        // upgrade kit requires its Crossbow revision.
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
        let scratch_path = Self::attach_scsi(cfg, [&bus_a, &bus_b])?;
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

        // The status bar's activity bits and the CPU's Compare matches,
        // shared with the graphics board before the CPU exists.
        let fasttick = Arc::new(AtomicU64::new(0));
        let gfx = board.map(|b| ImpactXio::with_board(WIDGET_GFX, b, heart.clone(), mem.clone(), heartbeat.clone(), fasttick.clone()));
        let vpro = if has_vpro {
            Some(Odyssey::new(WIDGET_GFX, kind.as_str(), heart.clone(), mem.clone(), heartbeat.clone(), fasttick.clone())?)
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
        let cpus: Vec<Arc<dyn CpuDevice>> = (0..cpu_count)
            .map(|id| Self::build_cpu(id, cpu_count, phys.phys.clone(), flash_lines[id as usize].clone(), &timer_manager, count_hz.max(1), ticks[id as usize].clone()))
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
        if let Some(g) = &gfx {
            g.mgras().set_cpu_cycles(cpu.cycles_ptr());
            for (c, tick) in cpus.iter().zip(&ticks).skip(1) {
                g.mgras().add_cpu_stats(c.cycles_ptr(), tick.clone());
            }
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
                let bitmap = cpu.tcache_bitmap_ptr();
                if !bitmap.is_null() {
                    sp.set_bitmap_sink2(bitmap);
                }
                #[cfg(feature = "jitv2")]
                cpu.set_tcache_gen_window(sp.gen_window_base());
                for c in cpus.iter().skip(1) {
                    sp.add_bitmap_sink(c.ppmem_bitmap_ptr());
                    c.set_tcache_window(sp.window_base());
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
            std::mem::forget(cpu_device);
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
            v.set_cpu_cycles(cpu.cycles_ptr());
            for (c, tick) in cpus.iter().zip(&ticks).skip(1) {
                v.add_cpu_stats(c.cycles_ptr(), tick.clone());
            }
            monitor.register_device(v.clone());
        }
        let monitor_port = cfg.monitor_port.unwrap_or(crate::monitor::DEFAULT_PORT);
        let ci_serial = cfg.ci.then(|| Arc::new(crate::dev::serial_host::CiSerialBackend::new()));
        Self::connect_serial(cfg, &ioc3, ci_serial.as_ref());

        // The Ethernet on the NAT gateway `[network]` describes.
        let (mut nat, mut frames) = (None, None);
        let net = crate::net::GatewayConfig::from_network(&cfg.network());
        if let Some(link) = crate::net::Link::with_gateway(net.clone())? {
            eprintln!("iris: Octane Ethernet on NAT: gateway {}, guest {} (DHCP)", net.gateway_ip, net.client_ip);
            frames = Some(link.frames());
            nat = Some((link.nat(), (net.client_ip, net.gateway_ip, net.netmask)));
            eth.attach(link);
        }
        Ok(Self {
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
    /// external one. Returns the `scratch = true` volume, if any.
    fn attach_scsi(cfg: &MachineConfig, buses: [&crate::scsi::ScsiBus; 2]) -> Result<Option<std::path::PathBuf>, String> {
        let attached = crate::scsi::attach_from_config(cfg, Some("ip30"), &|c| {
            buses.get(c as usize)
                .copied()
                .ok_or_else(|| format!("controller {c} (an Octane has controllers 0 and 1)"))
        })?;
        if let Some(d) = attached.daynaports.first() {
            return Err(format!("{}: kind = \"daynaport\" is not available on the Octane", crate::config::scsi_label(d.controller, d.id)));
        }
        Ok(attached.scratch)
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
        sysad: Arc<dyn BusDevice>,
        cached: Arc<CachedFlashLines>,
        tm: &Arc<TimerManager>,
        count_hz: u64,
        fasttick: Arc<AtomicU64>,
    ) -> Arc<dyn CpuDevice> {
        let cfg = MipsCpuConfig::for_model::<R10000ShadowCache>();
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
        if let Some(g) = &self.gfx {
            d.push(g.clone());
        }
        if let Some(v) = &self.vpro {
            d.push(v.clone());
        }
        d
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
        for d in self.devices().iter().rev() {
            d.stop();
        }
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
        self.flash.reset();
        if let Some(g) = &self.gfx {
            g.power_on();
        }
        if let Some(v) = &self.vpro {
            v.power_on();
        }
        Self::sync_links(&self.phys, &self.xbow);
        self.restart_peripherals();
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

impl Drop for OctaneMachine {
    fn drop(&mut self) {
        if let Some(slot) = self.controller_slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop_cpus();
        for d in self.devices().iter().rev() {
            d.stop();
        }
        self.monitor.shutdown();
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
}
