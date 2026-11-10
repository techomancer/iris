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

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::parts::{self, Part};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::cpu::smp::World;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::bridge::{Bridge, RawFlash};
use crate::dev::heart::{host_clock, Heart, HeartConfig, PIU_BASE, PIU_SIZE};
use crate::dev::heart_mem::HeartRam;
use crate::dev::ds1685::Ds1685;
use crate::dev::ioc3::{self, ByteBus, Ioc3};
use crate::dev::ioc3::eth::Ioc3Eth;
use crate::dev::isp1040::Isp1040;
use crate::dev::mgras::hq4::ImpactXio;
use crate::dev::odyssey::Odyssey;
use crate::dev::rad1::Rad1;
use crate::dev::sflash::{CfiFlash, FLASH_SIZE};
use crate::dev::w1;
use crate::dev::w1::sgi::{board_record_part, BoardRecord};
use crate::dev::xbow::Xbow;
use crate::dev::xio_pci::XioPciBoard;
use crate::hptimer::TimerManager;
use crate::ip30::{
    widget_small_base, CachedFlashLines, Ip30Bus, LAST_PC, WIDGET_BRIDGE, WIDGET_HEART, WIDGET_SMALL_SIZE,
    WIDGET_XBOW,
};
use crate::pci::{BoardIrq, DevFn};
use crate::chunk_store::{get_chunks_as_words, put_words_as_chunks, ChunkStore};
use crate::snapshot::{field, ChunksManifest, Manifest, Snapshot, StateTable, SCHEMA_VERSION};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable, Saveable};
use crate::workstation::{MachineRef, MachineSlot, Workstation};
use crate::xtalk::{HostPort, Upstream};

/// The PROM image tried when `prom` is left at the shared default, named
/// as IRIX ships it in /usr/cpu/firmware. The Octane has no embedded PROM.
pub const OCTANE_PROM_FILE: &str = "IP30prom.bin";
/// The processors an Octane shipped with, the default first.
pub const CPUS: &[&Part] = &[&parts::R10000_195, &parts::R12000_360, &parts::R14000_600];

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

/// One board record on a one-wire line: part, revision and name as the
/// product's inventory prints them. The barcode and the ROM serial are
/// made up and marked so (EMU...), so an inventory from the model never
/// reads as a real unit.
struct Record {
    barcode: &'static str,
    part: &'static str,
    revision: &'static str,
    name: &'static str,
    serial: u64,
}

impl Record {
    fn part(&self) -> w1::Ds2505 {
        board_record_part(
            &BoardRecord { barcode: self.barcode, part: self.part, revision: self.revision, name: self.name },
            self.serial,
        )
    }

    fn bus(&self) -> w1::Bus {
        let mut bus = w1::Bus::new();
        bus.add(Box::new(self.part()));
        bus
    }
}

/// The four identity records a product carries: the processor module on
/// HEART's one-wire master, the system board on the onboard Bridge's, and
/// the front plane and power supply on the IOC3's beside its
/// Ethernet-address part. Names, parts and revisions are the products'
/// inventories; the processor module's name depends on the module fitted as
/// well as the product.
struct Identity {
    processor: Record,
    system_board: Record,
    front_plane: Record,
    power_supply: Record,
}

impl Identity {
    /// The Octane. A PM10 has one processor and a PM20 two, and the PROM's
    /// diagnostics hold the record to the processors HEART reports.
    fn octane(cpus: u8) -> Identity {
        Identity {
            processor: Record {
                barcode: "EMUU00",
                part: "030-0888-004",
                revision: "B",
                name: if cpus > 1 { "PM20" } else { "PM10" },
                serial: 0x5145_4d55_5500,
            },
            system_board: Record {
                barcode: "EMUN00",
                part: "030-0887-003",
                revision: "F",
                name: "IP30",
                serial: 0x5145_4d55_4e00,
            },
            front_plane: Record {
                barcode: "EMUF00",
                part: "030-0891-003",
                revision: "A",
                name: "FP1",
                serial: 0x5145_4d55_4600,
            },
            power_supply: Record {
                barcode: "EMUS00",
                part: "060-0028-003",
                revision: "A",
                name: "PWR.SPPLY.SR",
                serial: 0x5145_4d55_5300,
            },
        }
    }

    /// The Octane2, from one reference unit's inventory. The processor
    /// module is that unit's single-processor 360 MHz one, for any processor
    /// count: no inventory of a dual-processor Octane2 is at hand
    /// (approximation).
    fn octane2() -> Identity {
        Identity {
            processor: Record {
                barcode: "EMUU00",
                part: "030-1590-002",
                revision: "B",
                name: "PM10360MHZ",
                serial: 0x5145_4d55_5500,
            },
            system_board: Record {
                barcode: "EMUN00",
                part: "030-1467-001",
                revision: "B",
                name: "IP30",
                serial: 0x5145_4d55_4e00,
            },
            front_plane: Record {
                barcode: "EMUF00",
                part: "030-0891-003",
                revision: "H",
                name: "FP1",
                serial: 0x5145_4d55_4600,
            },
            power_supply: Record {
                barcode: "EMUS00",
                part: "060-0035-002",
                revision: "A",
                name: "PWR.SPPLY.ER",
                serial: 0x5145_4d55_5300,
            },
        }
    }

    /// The IOC3's line: the station-address part, then the front plane and
    /// the power supply.
    fn ioc3_wire(&self, mac: [u8; 6]) -> w1::Bus {
        let mut bus = w1::Bus::new();
        bus.add(Box::new(w1::sgi::ethernet_address_part(mac, w1::sgi::ETHERNET_PART_SERIAL)));
        bus.add(Box::new(self.front_plane.part()));
        bus.add(Box::new(self.power_supply.part()));
        bus
    }
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
    /// The option boards in XIO slots (`[[xio_board]]`), each a Bridge of its
    /// own, in config order.
    pub xio_boards: Vec<XioPciBoard>,
    pub ioc3: Arc<Ioc3>,
    /// The clock on IOC3's byte bus, which IOC3 starts and stops.
    rtc: Arc<Ds1685>,
    pub scsi: [Arc<Isp1040>; 2],
    eth: Arc<Ioc3Eth>,
    rad: Arc<Rad1>,
    /// virtio-serial (`[virtio_serial]`), the next free Bridge slot.
    virtio: Option<Arc<crate::dev::virtio::pci::VirtioPci>>,
    /// The NVMe card (`[nvme]`), in a slot of an option board.
    nvme: Option<Arc<crate::dev::nvme::Nvme>>,
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
        let part = MachineProfile::OctaneIp30.part(cfg.machine.cpu.as_deref())?;
        // `prom`, or IRIX's file name when `prom` is the shared default and
        // that file is not there: there is no embedded Octane PROM.
        let prom_path = MachineProfile::OctaneIp30.prom_path(&cfg.prom);
        let prom = std::fs::read(&prom_path)
            .map_err(|e| format!("Octane PROM {prom_path}: {e} (set `prom` to an IP30 PROM image, the 1 MB flash contents)"))?;
        if prom.len() > FLASH_SIZE {
            return Err(format!("Octane PROM {prom_path} is {} bytes; the flash holds {FLASH_SIZE}", prom.len()));
        }
        let cpu_count = cfg.machine.cpus();
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
        let identity = if octane2 { Identity::octane2() } else { Identity::octane(cpu_count) };
        heart.set_microlan(identity.processor.bus());

        let gfx_link = if board.is_some() || has_vpro { 1 << WIDGET_GFX } else { 0 };
        let xio_widgets = cfg.xio_widgets()?;
        let xio_links = xio_widgets.iter().fold(0u16, |m, w| m | 1 << w);
        let xbow = Arc::new(Xbow::new(xbow_revision, (1 << WIDGET_HEART) | (1 << WIDGET_BRIDGE) | gfx_link | xio_links));
        xbow.connect_host(crate::ip30::HEART_HOST, heart.clone());

        // The onboard Bridge: memory and interrupts through HEART, the boot
        // flash on its first chip select, the system board's record on its
        // MicroLAN.
        let bridge = Bridge::new(Upstream::single(crate::ip30::HEART_HOST, mem.clone(), heart.clone()), &crate::ip30::PCI0);
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
        bridge.set_microlan(identity.system_board.bus());
        // DMA reaches only widgets the Crossbow has a live link to.
        let links = xbow.clone();
        bridge.set_link_alive(Box::new(move |w| links.carries(w)));

        // Option boards: each a Bridge on its own port, reaching memory and
        // HEART as the onboard one does, over the links the Crossbow has up.
        let bridge_revision = if octane2 { BRIDGE_REVISION_OCTANE2 } else { BRIDGE_REVISION };
        let xio_boards: Vec<XioPciBoard> = cfg.xio_board.iter().zip(&xio_widgets).map(|(b, &w)| {
            let up = xbow.upstream(vec![HostPort::new(crate::ip30::HEART_HOST, mem.clone(), heart.clone())]);
            let board = XioPciBoard::new(b.kind, w, up, bridge_revision);
            let links = xbow.clone();
            board.bridge().set_link_alive(Box::new(move |w| links.carries(w)));
            eprintln!("iris: {} in XIO slot {w}", b.kind.label());
            board
        }).collect();

        // The status bar's activity bits, shared with the graphics board.
        let heartbeat = Arc::new(AtomicU64::new(0));
        let mac = cfg.network().mac;
        // The buses are filled from `[scsi.N]` before the controllers exist.
        let [bus_a, bus_b] = [Isp1040::new_bus(), Isp1040::new_bus()];
        let (scratch_path, mut disks) = Self::attach_scsi(cfg, [&bus_a, &bus_b])?;
        let scsi = [Arc::new(Isp1040::new_at(0, bus_a)), Arc::new(Isp1040::new_at(1, bus_b))];
        for s in &scsi {
            s.set_heartbeat(heartbeat.clone());
        }
        let pci = bridge.bus();
        let slot = |device| DevFn { device, function: 0 };
        pci.attach(slot(0), scsi[0].clone());
        pci.attach(slot(1), scsi[1].clone());
        let nvme = (!cfg.nvme.path.is_empty())
            .then(|| -> Result<_, String> {
                // `validate` found the board; the slot may still be taken.
                let board = &xio_boards[cfg.nvme.board.unwrap_or(0)];
                let slot = cfg.nvme.slot.or_else(|| board.free_slot()).ok_or("[nvme]: the board has no free slot")?;
                board.check_slot(slot).map_err(|e| format!("[nvme]: {e}"))?;
                let card = crate::dev::nvme::Card::open(&cfg.nvme)?;
                card.fit(board.bus(), slot);
                eprintln!("iris: NVMe in the {} on XIO slot {}, slot {slot}", board.kind().label(), board.widget());
                Ok(card.nvme)
            })
            .transpose()?;
        disks.extend(nvme.as_ref().map(|n| crate::dev::nvme::disk_ref(&cfg.nvme, n)));
        // The DS1685 on IOC3's byte bus: the guest switching the machine off
        // through it, and its NVRAM in a file.
        let rtc = Ds1685::new(Some(event_tx.clone()));
        rtc.set_rtc_offset(&cfg.rtc_offset);
        if let Err(e) = rtc.set_nvram_file(&cfg.nvram) {
            eprintln!("iris: Octane NVRAM {}: {e}", cfg.nvram);
        }
        let ioc3 = Arc::new(Ioc3::new(&ioc3::IP30, mac, ByteBus::Ds1685(rtc.clone())));
        ioc3.set_microlan(identity.ioc3_wire(mac));
        ioc3.set_heartbeat(heartbeat.clone());
        let eth = ioc3.eth().expect("the Octane's IOC3 fits Ethernet").clone();
        pci.attach(slot(2), ioc3.clone());
        ioc3.connect_superio(pci);
        ioc3.set_timer_manager(timer_manager.clone());
        // ... and resetting itself through HEART.
        heart.on_soft_reset(Box::new(move || {
            let _ = event_tx.try_send(MachineEvent::HardReset);
        }));
        let rad = Arc::new(Rad1::new());
        rad.set_timer_manager(timer_manager.clone());
        if !cfg.no_audio {
            rad.enable_output(cfg.audio.clone());
        }
        pci.attach(slot(3), rad.clone());
        let virtio = cfg.virtio_serial.enabled.then(|| {
            let dev = Arc::new(crate::dev::virtio::pci::VirtioPci::new(
                crate::dev::virtio::console::from_config(&cfg.virtio_serial),
            ));
            pci.attach(slot(VIRTIO_SLOT), dev.clone());
            eprintln!("iris: virtio-serial on Bridge slot {VIRTIO_SLOT}");
            dev
        });

        // The status bar's activity bits and the CPU's Compare matches,
        // shared with the graphics board before the CPU exists.
        let fasttick = Arc::new(AtomicU64::new(0));
        let status = crate::disp::StatusSource::new(heartbeat.clone(), fasttick.clone());
        let gfx = board
            .map(|b| ImpactXio::with_raster(WIDGET_GFX, b, heart.clone(), mem.clone(), status.clone(), (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str())))
            .transpose()?;
        let vpro = if has_vpro {
            Some(Odyssey::new(WIDGET_GFX, kind.as_str(), heart.clone(), mem.clone(), status.clone(), (cfg.graphics.threads, cfg.graphics.tile), (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()))?)
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
        phys.attach_long_window(WIDGET_BRIDGE, bridge.long_window());
        for b in &xio_boards {
            phys.attach_widget(b.widget(), b.bridge().clone(), false);
            phys.attach_long_window(b.widget(), b.bridge().long_window());
        }
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
        let (weak, bridge_weak) = (Arc::downgrade(&phys), Arc::downgrade(&bridge));
        bridge.bus().on_bar_mappings(Box::new(move |runs| {
            if let (Some(p), Some(b)) = (weak.upgrade(), bridge_weak.upgrade()) {
                let runs = runs.iter().map(|r| (r.cpu..r.cpu + r.len, b.bar_device(r))).collect();
                p.set_widget_runs(WIDGET_BRIDGE, runs);
            }
        }));
        // Every option board's runs reach the processor over its own widget.
        for b in &xio_boards {
            let (weak, bridge_weak, w) = (Arc::downgrade(&phys), Arc::downgrade(b.bridge()), b.widget());
            b.bridge().bus().on_bar_mappings(Box::new(move |runs| {
                if let (Some(p), Some(b)) = (weak.upgrade(), bridge_weak.upgrade()) {
                    let runs = runs.iter().map(|r| (r.cpu..r.cpu + r.len, b.bar_device(r))).collect();
                    p.set_widget_runs(w, runs);
                }
            }));
        }
        let mut resettable: Vec<(u8, std::sync::Weak<Bridge>)> = vec![(WIDGET_BRIDGE, Arc::downgrade(&bridge))];
        resettable.extend(xio_boards.iter().map(|b| (b.widget(), Arc::downgrade(b.bridge()))));
        xbow.on_link_reset(Box::new(move |w| {
            if let Some((_, b)) = resettable.iter().find(|(p, _)| *p == w) {
                if let Some(b) = b.upgrade() {
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

        let count_hz = cfg.clock.count_hz(part);
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
            .map(|id| {
                let (sysad, cached, tm, fasttick) = (phys.phys.clone(), flash_lines[id as usize].clone(), &timer_manager, ticks[id as usize].clone());
                crate::with_cpu!(part, true, false, C => Self::build_cpu::<C>(id, cpu_count, memory.clone(), sysad, cached, tm, count_hz.max(1), fasttick, part.prid()))
            })
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

        // The shadow cache and the JIT read RAM through a cached view of the
        // ppmem window, one per CPU; descriptor writes lay the window and
        // every view out again on the CPU thread.
        if let Some(sp) = phys.phys.ppmem_space() {
            for c in cpus.iter() {
                let v = sp
                    .add_cached_view(&[c.ppmem_bitmap_ptr(), c.tcache_bitmap_ptr()])
                    .map_err(|e| format!("ppmem: a CPU's cached view: {e}"))?;
                unsafe {
                    c.set_tcache_window(sp.view_base(v));
                    c.set_cached_view(sp, v);
                    c.set_tcache_shift(crate::ppmem::window_shift(sp.window_size()));
                    #[cfg(feature = "jitv2")]
                    {
                        use crate::ppmem::MappedMemory;
                        c.set_tcache_gen_window(sp.gen_window_base());
                    }
                }
            }
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
                drop(g);
                crate::cpu::jitv2::jitv2::start_compiling(&*cpu, &jit, cfg.jitv2.compile, phys.phys.clone() as Arc<dyn BusDevice>);
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
        // `pci`: the Bridge's bus, as the O2's MACE shows its own.
        monitor.register_device(bridge.clone());
        // `nvme`: the card's registers, queues and copy-on-write disk.
        if let Some(n) = &nvme {
            monitor.register_device(n.clone());
        }
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
        for b in &xio_boards {
            b.bridge().register_locks();
        }
        flash.register_locks();
        ioc3.register_locks();
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
            xio_boards,
            ioc3,
            rtc,
            scsi,
            eth,
            rad,
            virtio,
            nvme,
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

    /// An Octane's processor `id` of `cpu_count` over cache type `C`.
    #[allow(clippy::too_many_arguments)]
    fn build_cpu<C>(
        id: u8,
        cpu_count: u8,
        memory: Arc<crate::memory_map::MemoryMap>,
        sysad: Arc<dyn BusDevice>,
        cached: Arc<CachedFlashLines>,
        tm: &Arc<TimerManager>,
        count_hz: u64,
        fasttick: Arc<AtomicU64>,
        prid: u32,
    ) -> Arc<dyn CpuDevice>
    where
        C: crate::cpu::mips_cache_v2::CpuModel + MipsCache + From<Arc<dyn BusDevice>> + 'static,
        MipsCpu<MipsTlb, C>: CpuDevice,
    {
        let cfg = MipsCpuConfig::for_model::<C>(memory).with_prid(prid);
        let tlb = MipsTlb::new(cfg.tlb_entries);
        let mut exec: MipsExecutor<MipsTlb, C> = MipsExecutor::new(sysad, tlb, &cfg);
        // Cached accesses see the PROM's dirty flash lines; see `CachedFlashLines`.
        exec.cache = C::from(cached as Arc<dyn BusDevice>);
        exec.core.set_timer_manager(tm.clone());
        exec.core.set_count_hz(count_hz);
        exec.core.fasttick_count = fasttick;
        // IRIX zeroes Count and arms Compare a few instructions later; see
        // `MipsCore::count_write_samples`.
        exec.core.count_write_samples = true;
        // The R10000 drives 40 physical address bits; HEART decodes them all.
        // EntryLo bits 33:32 are physical 39:38; a guest that keeps software
        // flags there clears them with FrameMask, so no narrowing is needed.
        exec.phys_mask = (1 << 40) - 1;
        exec.core.cpu_id = id;
        exec.core.smp = cpu_count > 1;
        exec.batch_hook = Some(Box::new(move |exec: &mut MipsExecutor<MipsTlb, C>| {
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
            self.scsi[0].clone(),
            self.scsi[1].clone(),
            self.rad.clone(),
        ];
        if let Some(v) = &self.virtio {
            d.push(v.clone());
        }
        if let Some(n) = &self.nvme {
            d.push(n.clone());
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
        for b in &self.xio_boards {
            b.bridge().reset();
            b.bridge().reset_pci();
        }
        self.ioc3.power_on();
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
    /// The NVMe disk's, when the snapshot has them.
    nvme_overlays: Option<OverlaySets>,
    dir: std::path::PathBuf,
}

impl OctaneMachine {
    /// The SCSI buses that hold disks: both controllers', then the NVMe
    /// card's namespace.
    fn disk_buses(&self) -> Vec<&crate::scsi::ScsiBus> {
        use crate::scsi_monitor::ScsiTargets;
        self.scsi.iter().map(|c| c.bus()).chain(self.nvme.iter().map(|n| n.scsi_bus())).collect()
    }

    /// Why this machine cannot be saved, when it cannot.
    fn snapshot_refusal(&self) -> Option<String> {
        snapshot_refusal_for(self.gfx.is_some(), self.vpro.is_some(), self.cpus.len())
    }

    /// Every device's state but the CPU's. Called with the machine stopped.
    fn save_board(&self) -> toml::Value {
        let lines = self.flash_lines.iter().map(|f| f.save_state()).collect();
        let mut t = StateTable::new();
        // Only with boards, so a machine without them saves what it always did.
        if !self.xio_boards.is_empty() {
            t.value("xio_boards", toml::Value::Array(self.xio_boards.iter().map(|b| b.bridge().save_state()).collect()));
        }
        if let Some(n) = &self.nvme {
            t.value("nvme", n.save_state());
        }
        t
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
        if let Some(n) = &self.nvme {
            load("nvme", &**n)?;
        }
        load("flash", &*self.flash)?;
        let lines = field(v, "flash_lines")?.as_array().filter(|l| l.len() == self.flash_lines.len())
            .ok_or("flash_lines: not one entry per processor")?;
        for (f, saved) in self.flash_lines.iter().zip(lines) {
            f.load_state(saved)?;
        }
        load("bridge", &*self.bridge)?;
        if !self.xio_boards.is_empty() {
            let boards = field(v, "xio_boards")?.as_array().filter(|l| l.len() == self.xio_boards.len())
                .ok_or("xio_boards: not one entry per option board")?;
            for (b, saved) in self.xio_boards.iter().zip(boards) {
                b.bridge().load_state(saved).map_err(|e| format!("xio_boards: {e}"))?;
            }
        }
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
        if let Some(n) = &self.nvme {
            let dir = snap.dir.join("nvme");
            std::fs::create_dir_all(&dir).map_err(io)?;
            let overlays = n.export_overlays(&dir).map_err(|e| format!("COW overlay export (NVMe): {e}"))?;
            snap.write_toml("cow_nvme.toml", &crate::machine::build_cow_table(overlays)).map_err(io)?;
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
        crate::snapshot::check_schema(name, m.schema_version)?;
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
        let nvme_overlays = snap.read_toml("cow_nvme.toml").ok().map(|t| crate::machine::parse_cow_table(&t));
        Ok(OctaneSnapshot {
            cpu: state("cpu")?,
            board: state("octane")?,
            flash,
            memory,
            overlays,
            nvme_overlays,
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
        if let (Some(n), Some(sets)) = (&self.nvme, &s.nvme_overlays) {
            n.import_overlays(&s.dir.join("nvme"), sets).map_err(|e| format!("COW overlay import (NVMe): {e}"))?;
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
        // The compile pool's requests point into the memory freed below.
        #[cfg(feature = "jitv2")]
        for c in &self.cpus {
            crate::cpu::jitv2::Jitv2::stop_for_drop(&c.jitv2(), || self.stop_cpus());
        }
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
        self.rtc.flush();
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
        self.disk_buses().iter().map(|b| b.pending_chd_sync_count()).sum()
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.disk_buses().iter().try_for_each(|b| b.flush())
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        crate::scsi::ScsiBus::sync_chd_disks(&self.disk_buses(), only, progress, cancel)
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

    /// The Octane2 carries its own four identity records, from one reference
    /// unit's inventory, and the Octane keeps the ones it had. The IOC3's
    /// wire holds the Ethernet-address part, the front plane and the power
    /// supply; HEART's and the Bridge's hold one record each.
    #[test]
    fn each_product_carries_its_own_identity_records() {
        let o = Identity::octane(1);
        let o2 = Identity::octane2();
        let key = |i: &Identity| {
            [&i.processor, &i.system_board, &i.front_plane, &i.power_supply].map(|r| (r.part, r.revision, r.name))
        };
        assert_eq!(
            key(&o),
            [
                ("030-0888-004", "B", "PM10"),
                ("030-0887-003", "F", "IP30"),
                ("030-0891-003", "A", "FP1"),
                ("060-0028-003", "A", "PWR.SPPLY.SR"),
            ]
        );
        assert_eq!(
            key(&o2),
            [
                ("030-1590-002", "B", "PM10360MHZ"),
                ("030-1467-001", "B", "IP30"),
                ("030-0891-003", "H", "FP1"),
                ("060-0035-002", "A", "PWR.SPPLY.ER"),
            ]
        );
        assert_eq!(Identity::octane(2).processor.name, "PM20", "a dual-processor Octane's module");
        let families = |bus: &w1::Bus| {
            let mut f: Vec<u8> = bus.roms().iter().map(|rom| rom[0]).collect();
            f.sort();
            f
        };
        assert_eq!(families(&o2.ioc3_wire([8, 0, 0x69, 1, 2, 3])), [0x0b, 0x0b, 0x91]);
        assert_eq!(families(&o2.processor.bus()), [0x0b]);
        assert_eq!(families(&o2.system_board.bus()), [0x0b]);
    }

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
        cfg.machine.cpu = Some("r10000".into());
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

    /// An Octane with a PCI module and shoehorns: each board's Crossbow link
    /// comes up and its Bridge answers on its own port, a function in each
    /// slot is reached by a configuration read through the processor's
    /// address, its interrupt arrives at HEART and its DMA lands in RAM. Ports
    /// with nothing fitted stay down, and so do the device numbers a board
    /// has no slot at.
    #[test]
    fn option_boards_are_fitted_on_their_ports() {
        use crate::config::{MachineProfile, XioBoardConfig};
        use crate::dev::bridge::{reg, CFG0_BASE, DIRECT_ADD_512MB};
        use crate::dev::xio_pci::Kind;
        use crate::pci::testing::TestCard;
        use crate::pci::{Bar, PciFunction, PCI_INTA};
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("xio-boards");
            let prom = empty_prom(&dir);
            let m = machine_with(&prom, &dir, |cfg| {
                assert_eq!(cfg.machine.profile, MachineProfile::OctaneIp30);
                cfg.xio_board = vec![
                    XioBoardConfig { kind: Kind::Shoehorn, widget: None },
                    XioBoardConfig { kind: Kind::Cage, widget: None },
                    XioBoardConfig { kind: Kind::Shoehorn, widget: Some(9) },
                ];
            });
            let ports: Vec<u8> = m.xio_boards.iter().map(|b| b.widget()).collect();
            assert_eq!(ports, [10, 13, 9], "a shoehorn takes 10, the module is in 13, one shoehorn asked for 9");
            for w in 9..=13u8 {
                assert_eq!(m.xbow.carries(w), ports.contains(&w), "link to port {w}");
            }
            let word = |w: u8, off: u32| {
                let r = m.phys.phys.read32(widget_small_base(w) + u64::from(off));
                r.is_ok().then_some(r.data)
            };
            for (&w, revision) in ports.iter().zip([4, BRIDGE_REVISION, 4]) {
                let id = word(w, 4).expect("the Bridge answers on its port");
                assert_eq!((id >> 12 & 0xffff, id >> 28), (0xc002, revision), "port {w}: a Bridge of the board's revision");
            }
            assert_eq!(word(11, 4), None, "nothing answers on port 11");
            assert_eq!(word(12, 4), None, "nor on 12");

            // One bank of memory at 0x2000_0000, as the PROM's descriptor
            // places it.
            m.ram.set_memcfg([0x8001_0000, 0, 0, 0]);
            for b in &m.xio_boards {
                let w = b.widget();
                let set = |off: u32, v: u32| assert_eq!(m.phys.phys.write32(widget_small_base(w) + u64::from(off + 4), v), crate::traits::BUS_OK);
                set(reg::INT_DEST_UPPER, u32::from(WIDGET_HEART) << 16);
                set(reg::INT_DEST_LOWER, 0x80);
                set(reg::DIRECT_MAP, u32::from(WIDGET_HEART) << 20 | DIRECT_ADD_512MB);
                for device in 0..8u8 {
                    let slot = b.kind().slots().contains(&device);
                    assert_eq!(word(w, CFG0_BASE + u32::from(device) * 0x1000), None, "port {w}: device {device} is empty");
                    if !slot {
                        continue;
                    }
                    let card = TestCard::new([Bar::Unused; 6], PCI_INTA);
                    b.attach(device, card.clone()).unwrap();
                    assert_eq!(word(w, CFG0_BASE + u32::from(device) * 0x1000), Some(0x5678_1234), "port {w}: slot {device} answers");

                    // Its INTA is Bridge pin `device`, sent to HEART as a
                    // vector of the test's choosing.
                    let vector = 16 + 8 * u32::from(w - 9) + u32::from(device);
                    set(reg::int_host(u32::from(device)), vector);
                    set(reg::INT_ENABLE, 1 << device);
                    assert_eq!(m.heart.pending() >> vector & 1, 0);
                    card.state().irq().unwrap().set(true);
                    assert_eq!(m.heart.pending() >> vector & 1, 1, "port {w} slot {device}: HEART's bit {vector}");

                    // Its DMA, through the direct map to HEART, into RAM.
                    let at = u64::from(vector) * 16;
                    assert!(card.state().dma().unwrap().dma_write(0x8000_0000 + at, &[1, 2, 3, 4]).is_ok());
                    assert_eq!(m.ram.dma_view().read32(0x2000_0000 + at).data, u32::from_be_bytes([4, 3, 2, 1]), "port {w} slot {device}: DMA reached memory");
                }
            }
        }).unwrap().join().unwrap();
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

    /// `[nvme]` puts a card in a slot of an option board, behind a
    /// PCI-to-PCI bridge when asked, and in no slot of the onboard Bridge. The
    /// machine starts and stops it with the other parts and records its disk.
    /// A snapshot takes back what the overlay held when it was saved, and the
    /// controller's state with it; a reset of the machine resets the
    /// controller to its power-on state.
    fn nvme_in_a_slot(kind: crate::dev::xio_pci::Kind, slot: Option<u8>, bridge: bool) {
        use crate::config::{NvmeConfig, XioBoardConfig};
        use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};
        use crate::scsi_monitor::ScsiTargets;
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || {
            let dir = scratch(&format!("nvme-{}-{}", kind.name(), slot.unwrap_or(0)));
            let prom = empty_prom(&dir);
            let disk = dir.join("nvme.raw");
            std::fs::write(&disk, vec![0u8; 1 << 20]).unwrap();
            let path = disk.to_string_lossy().into_owned();
            let mut m = machine_with(&prom, &dir, |cfg| {
                cfg.xio_board = vec![XioBoardConfig { kind, widget: None }];
                cfg.nvme = NvmeConfig { path, overlay: true, bridge, board: None, slot };
            });
            let board = &m.xio_boards[0];
            let slot = slot.unwrap_or(1);
            let at = DevFn { device: slot, function: 0 };
            let class = |bus: &crate::pci::PciBus, at| bus.config_read(at, 8, 4).map(|d| d >> 8);
            // Through the processor's address of the board's widget, as the
            // PROM reads it.
            let w = board.widget();
            let window = u64::from(crate::dev::bridge::CFG0_BASE + u32::from(slot) * 0x1000 + 8);
            let seen = m.phys.phys.read32(widget_small_base(w) + window).data >> 8;
            let pci = board.bus();
            if bridge {
                assert_eq!((seen, class(pci, at)), (0x06_0400, Some(0x06_0400)), "a PCI-to-PCI bridge in the slot");
                pci.config_write(at, 0x18, 4, 0x0001_0100);
                let sec = pci.find_bus(1).expect("the bus behind it");
                assert_eq!(class(&sec, DevFn { device: 0, function: 0 }), Some(0x01_0802), "the card behind it");
            } else {
                assert_eq!((seen, class(pci, at)), (0x01_0802, Some(0x01_0802)), "the card in the slot");
            }
            for (i, b) in m.xio_boards.iter().enumerate() {
                assert_eq!(b.bus().occupied(slot), true, "board {i}");
            }
            // The onboard Bridge keeps its own devices.
            let onboard: Vec<u8> = (0..8).filter(|&d| m.bridge.bus().occupied(d)).collect();
            assert_eq!(onboard, [0, 1, 2, 3], "SCSI, IOC3 and the RAD, and nothing else");
            assert!(m.disks.iter().any(|d| d.id == crate::dev::nvme::SNAPSHOT_SLOT && d.size_bytes == 1 << 20));

            let nvme = m.nvme.clone().unwrap();
            let io = |cdb: Vec<u8>, data: Option<Vec<u8>>| {
                let req = ScsiRequest { cdb, data_len: ScsiDataLength::Unlimited, data_in: data };
                nvme.scsi_bus().request(0, &req).unwrap().unwrap().data
            };
            let rw = |op: u8| vec![op, 0, 0, 0, 0, 3, 0, 0, 1, 0];
            io(rw(scsi_cmd::WRITE_10), Some(vec![0x11; 512]));
            m.save_snapshot("s").unwrap();
            assert!(m.saves.join("s/nvme/scsi0.overlay").exists(), "the overlay goes with the snapshot");
            io(rw(scsi_cmd::WRITE_10), Some(vec![0x22; 512]));
            assert_eq!(io(rw(scsi_cmd::READ_10), None), vec![0x22; 512]);
            m.load_snapshot("s").unwrap();
            assert_eq!(io(rw(scsi_cmd::READ_10), None), vec![0x11; 512], "the disk is as saved");

            // The machine's reset (HEART's software reset, the monitor's
            // reset) is RST#, which reaches the card on the option board.
            crate::dev::nvme::tests::dirty(&nvme);
            m.reset();
            m.stop();
            crate::dev::nvme::tests::assert_power_on(&nvme);
        }).unwrap().join().unwrap();
    }

    #[test]
    fn an_nvme_card_is_fitted_behind_a_bridge_in_a_cage_slot() {
        nvme_in_a_slot(crate::dev::xio_pci::Kind::Cage, Some(2), true);
    }

    #[test]
    fn an_nvme_card_is_fitted_in_the_cages_last_slot() {
        nvme_in_a_slot(crate::dev::xio_pci::Kind::Cage, Some(3), false);
    }

    #[test]
    fn an_nvme_card_is_fitted_in_a_shoehorn() {
        nvme_in_a_slot(crate::dev::xio_pci::Kind::Shoehorn, None, false);
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
