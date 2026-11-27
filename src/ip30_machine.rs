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

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::config::{MachineConfig, MachineProfile};
use crate::cpu::parts::{self, Part};
use crate::cpu::mips_cache_v2::MipsCache;
use crate::cpu::smp::World;
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::mips_tlb::MipsTlb;
use crate::dev::bridge::{Bridge, RawFlash, PIN_WIRING};
use crate::dev::heart::{Heart, HeartConfig, PIU_BASE, PIU_SIZE};
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
    widget_small_base, CachedFlashLines, Ip30Bus, LAST_PC, SWITCHED_WIDGETS, WIDGET_BRIDGE,
    WIDGET_HEART, WIDGET_SMALL_SIZE, WIDGET_XBOW,
};
use crate::pci::{BoardIrq, DevFn, PCI_INTA, PCI_INTB};
use crate::topology::MachineSlotKind;
use crate::snapshot::{field, StateTable};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable, Saveable};
use crate::workstation::{MachineRef, MachineSlot, Workstation};
use crate::xtalk::{HostPort, Upstream};

/// The PROM image tried when `prom` is left at the shared default, named
/// as IRIX ships it in /usr/cpu/firmware. The 6.5.30 image is also built in.
pub const OCTANE_PROM_FILE: &str = "IP30prom.bin";
/// The processors an Octane shipped with, the default first.
pub const CPUS: &[&Part] = &[&parts::R10000_195, &parts::R12000_360, &parts::R14000_600];

/// The two ISP1040s on the onboard Bridge, internal bus then external.
const SCSI_DEVICES: [u8; 2] = [0, 1];

/// The system board's Bridge bus: two ISP1040s, the IOC3, the RAD1 and spare slots 5 to 7 for virtio (7 first, on INTB).
/// Device 4 is none: its INTA is the IOC3's SuperIO pin.
pub(crate) const SYSTEM_SLOTS: &[(u8, MachineSlotKind, Option<&str>)] = &[
        (SCSI_DEVICES[0], MachineSlotKind::ScsiBus { part: "isp1040", ids: 15 }, None),
        (SCSI_DEVICES[1], MachineSlotKind::ScsiBus { part: "isp1040", ids: 15 }, None),
        (2, MachineSlotKind::Fixed("ioc3"), None),
        (3, MachineSlotKind::Fixed("rad1"), None),
        (7, MachineSlotKind::Extra { pin: PCI_INTB }, None),
        (5, MachineSlotKind::Extra { pin: PCI_INTA }, None),
        (6, MachineSlotKind::Extra { pin: PCI_INTA }, None),
    ];

/// The XIO slots as Crossbow ports, by letter A to D (007-3582-001 p. 33), in the order a board with no slot takes them;
/// real Octanes have a shoehorn in 10.
const XIO_SLOTS: [(u8, &str); 4] = [(10, "slot_d"), (12, "slot_a"), (9, "slot_c"), (11, "slot_b")];

/// The Octane's slots.
pub(crate) fn slots(s: &mut Vec<crate::topology::MachineSlot>) {
    use crate::topology::DeviceFamily;
    crate::topology::pci_bus_slots(s, &format!("xio{}/pci0", crate::ip30::WIDGET_BRIDGE), "Octane system board", SYSTEM_SLOTS);
    for (w, alias) in XIO_SLOTS {
        // The graphics board has the port it is fitted in whenever one is.
        crate::topology::open_slot(s, &format!("xio{w}"), if w == WIDGET_GFX { &[DeviceFamily::Xio, DeviceFamily::Gfx] } else { &[DeviceFamily::Xio] }, &[alias]);
    }
    crate::topology::open_slot(s, &format!("xio{}", crate::dev::xio_pci::CAGE_WIDGET), &[DeviceFamily::Cage], &["pci_module", "shoebox"]);
}

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
    name: std::borrow::Cow<'static, str>,
    serial: u64,
}

impl Record {
    fn part(&self) -> w1::Ds2505 {
        board_record_part(
            &BoardRecord { barcode: self.barcode, part: self.part, revision: self.revision, name: &self.name },
            self.serial,
        )
    }

    fn bus(&self) -> w1::Bus {
        let mut bus = w1::Bus::new();
        bus.add(Box::new(self.part()));
        bus
    }
}

/// Synthetic one-wire ROM serials, one per board kind.
const PROCESSOR_SERIAL: u64 = 0x5145_4d55_5500;
const SYSTEM_BOARD_SERIAL: u64 = 0x5145_4d55_4e00;
const FRONT_PLANE_SERIAL: u64 = 0x5145_4d55_4600;
const POWER_SUPPLY_SERIAL: u64 = 0x5145_4d55_5300;

/// The four identity records a product carries: the processor module on
/// HEART's one-wire master, the system board on the onboard Bridge's, and
/// the front plane and power supply on the IOC3's beside its
/// Ethernet-address part. Names, parts and revisions are the products'
/// inventories; the processor module's name depends on the module populated as
/// well as the product.
struct Identity {
    processor: Record,
    system_board: Record,
    front_plane: Record,
    power_supply: Record,
}

impl Identity {
    /// The processor module for the configured processor and processor
    /// count, whichever chassis it sits in. A PM10 carries one processor and a
    /// PM20 two; the PROM's power-on test holds the name's prefix to the
    /// processors HEART reports. The R10000 module is the original Octane's
    /// (PM10/PM20). Later modules carry their clock in the name, from two
    /// reference Octane2s: a single 360 MHz R12000 is PM10360MHZ,
    /// 030-1590-002 rev B; two 600 MHz R14000s are PM20600MHZ, 030-1778-001
    /// rev C. Other counts and clocks keep their family's part number
    /// (approximation: no inventory of them is at hand).
    fn processor_module(part: &Part, cpus: u8) -> Record {
        let count = if cpus > 1 { "PM20" } else { "PM10" };
        let (part_no, revision, name) = match part.family {
            parts::Family::R10000 => ("030-0888-004", "B", std::borrow::Cow::Borrowed(count)),
            parts::Family::R14000 => ("030-1778-001", "C", format!("{count}{}MHZ", part.mhz).into()),
            _ => ("030-1590-002", "B", format!("{count}{}MHZ", part.mhz).into()),
        };
        Record { barcode: "EMUU00", part: part_no, revision, name, serial: PROCESSOR_SERIAL }
    }

    /// The Octane's chassis records, from its inventory, around `processor`.
    fn octane(processor: Record) -> Identity {
        Identity {
            processor,
            system_board: Record {
                barcode: "EMUN00",
                part: "030-0887-003",
                revision: "F",
                name: "IP30".into(),
                serial: SYSTEM_BOARD_SERIAL,
            },
            front_plane: Record {
                barcode: "EMUF00",
                part: "030-0891-003",
                revision: "A",
                name: "FP1".into(),
                serial: FRONT_PLANE_SERIAL,
            },
            power_supply: Record {
                barcode: "EMUS00",
                part: "060-0028-003",
                revision: "A",
                name: "PWR.SPPLY.SR".into(),
                serial: POWER_SUPPLY_SERIAL,
            },
        }
    }

    /// The Octane2's chassis records, from one reference unit's inventory,
    /// around `processor`.
    fn octane2(processor: Record) -> Identity {
        Identity {
            processor,
            system_board: Record {
                barcode: "EMUN00",
                part: "030-1467-001",
                revision: "B",
                name: "IP30".into(),
                serial: SYSTEM_BOARD_SERIAL,
            },
            front_plane: Record {
                barcode: "EMUF00",
                part: "030-0891-003",
                revision: "H",
                name: "FP1".into(),
                serial: FRONT_PLANE_SERIAL,
            },
            power_supply: Record {
                barcode: "EMUS00",
                part: "060-0035-002",
                revision: "A",
                name: "PWR.SPPLY.ER".into(),
                serial: POWER_SUPPLY_SERIAL,
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
    /// The virtio and audio functions the entries placed on the buses.
    cards: crate::devices::AttachedCards,
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
    events: crate::traits::MachineEvents,
    event_rx: Option<crate::traits::MachineEventQueue>,
    controller_slot: Option<MachineSlot>,
    lifecycle: crate::machine_state::GuestState,
    /// The disks as a snapshot records them, to refuse one taken on others.
    disks: Vec<crate::snapshot::DiskRef>,
    /// The last snapshot loaded, for `parent` and `rollback`.
    restored: crate::snapshot::Restored,
}

impl OctaneMachine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        let part = MachineProfile::OctaneIp30.part(cfg.machine.cpu.as_deref())?;
        // `prom`, else IRIX's file name if it is here, else the PROM built into iris.
        let loaded = crate::dev::prom_image::resolve(MachineProfile::OctaneIp30, &cfg.prom, FLASH_SIZE)
            .map_err(|e| format!("Octane {e} (set `prom` to an IP30 PROM image, the 1 MB flash contents)"))?;
        let prom_path = loaded.name();
        let prom = loaded.bytes;
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
        let (event_tx, event_rx) = crate::traits::MachineEvents::channel(4);
        let events = event_tx.clone();
        let timer_manager = Arc::new(TimerManager::new());

        // Memory, and HEART, whose descriptors place it.
        let ram = Arc::new(HeartRam::try_new(std::array::from_fn(|i| cfg.banks[i]))?);
        let mem = ram.dma_view();
        let heart = Arc::new(Heart::new(HeartConfig { cpus: cpu_count, ..HeartConfig::octane(banks) }, timer_manager.clock().clone()));
        // HEART's MicroLAN holds the processor module's record. The module
        // follows the processor settings alone; the chassis records follow
        // the product.
        let processor = Identity::processor_module(part, cpu_count);
        let identity = if octane2 { Identity::octane2(processor) } else { Identity::octane(processor) };
        heart.set_microlan(identity.processor.bus());

        let gfx_link = if board.is_some() || has_vpro { 1 << WIDGET_GFX } else { 0 };
        // The option boards, in config order, each on the port it was placed at.
        let mut resolved = crate::topology::DevicePlacement::new(cfg)?;
        let xio_widgets: Vec<(crate::dev::xio_pci::Kind, u8)> = resolved.placed()
            .filter_map(|(d, at)| match d.device {
                crate::devices::DeviceModel::XioBoard(kind) => Some((kind, at.last_number()?)),
                _ => None,
            })
            .collect();
        let xio_links = xio_widgets.iter().fold(0u16, |m, (_, w)| m | 1 << w);
        let xbow = Arc::new(Xbow::new(xbow_revision, (1 << WIDGET_HEART) | (1 << WIDGET_BRIDGE) | gfx_link | xio_links));
        xbow.connect_host(crate::ip30::HEART_HOST, heart.clone());

        // The onboard Bridge: memory and interrupts through HEART, the boot
        // flash on its first chip select, the system board's record on its
        // MicroLAN.
        let bridge = Bridge::new(Upstream::single(crate::ip30::HEART_HOST, mem.clone(), heart.clone()), &PIN_WIRING);
        bridge.set_revision(if octane2 { BRIDGE_REVISION_OCTANE2 } else { BRIDGE_REVISION });
        // The flash holds the PROM's own records as well as its code; with
        // `[octane] flash` they last from run to run.
        let flash = Arc::new(match &cfg.octane.flash {
            Some(path) => {
                CfiFlash::with_file(&prom, FLASH_SIZE, path).map_err(|e| format!("machine.flash {path}: {e}"))?
            }
            None => CfiFlash::new(&prom, FLASH_SIZE),
        });
        bridge.attach_flash(0, flash.clone(), FLASH_SIZE as u32);
        bridge.attach_flash(1, Arc::new(RawFlash::new(&[], FLASH_SELECT1_SIZE)), FLASH_SELECT1_SIZE as u32);
        bridge.set_microlan(identity.system_board.bus());

        // Option boards: each a Bridge on its own port, reaching memory and
        // HEART as the onboard one does, over the links the Crossbow has up.
        let bridge_revision = if octane2 { BRIDGE_REVISION_OCTANE2 } else { BRIDGE_REVISION };
        let xio_boards: Vec<XioPciBoard> = xio_widgets.iter().map(|&(kind, w)| {
            let up = xbow.upstream(vec![HostPort::new(crate::ip30::HEART_HOST, mem.clone(), heart.clone())]);
            let board = XioPciBoard::new(kind, w, up, bridge_revision);
            xbow.serve_bridge(w, board.bridge());
            eprintln!("iris: {} in XIO slot {w}", kind.label());
            board
        }).collect();
        resolved.bind_pci(&format!("xio{WIDGET_BRIDGE}/pci0"), bridge.bus().clone())?;
        for b in &xio_boards {
            resolved.bind_pci(&format!("xio{}/pci0", b.widget()), b.bus().clone())?;
        }

        // The status bar's activity bits, shared with the graphics board.
        let heartbeat = Arc::new(AtomicU64::new(0));
        let mac = cfg.network().mac;
        // The buses are filled before the controllers exist.
        let [bus_a, bus_b] = [Isp1040::new_bus(), Isp1040::new_bus()];
        let cards = resolved.build_cards(cfg, &timer_manager)?;
        for f in &cards.fibre {
            f.fc.set_heartbeat(heartbeat.clone());
        }
        let (scratch_path, mut disks) = Self::attach_scsi(cfg, &resolved, [&bus_a, &bus_b])?;
        let scsi = [Arc::new(Isp1040::new_at(0, bus_a, &timer_manager)), Arc::new(Isp1040::new_at(1, bus_b, &timer_manager))];
        for s in &scsi {
            s.set_heartbeat(heartbeat.clone());
        }
        let pci = bridge.bus();
        let slot = |device| DevFn { device, function: 0 };
        pci.attach(slot(SCSI_DEVICES[0]), scsi[0].clone());
        pci.attach(slot(SCSI_DEVICES[1]), scsi[1].clone());
        disks.extend(cards.disk_refs());
        // The DS1685 on IOC3's byte bus: the guest switching the machine off
        // through it, and its NVRAM in a file.
        let rtc = Ds1685::new(Some(event_tx.clone()), timer_manager.clock().clone());
        rtc.set_rtc_offset(&cfg.rtc_offset);
        if let Err(e) = rtc.set_nvram_file(&cfg.nvram) {
            eprintln!("iris: Octane NVRAM {}: {e}", cfg.nvram);
        }
        let ioc3 = Arc::new(Ioc3::new(&timer_manager, &ioc3::IP30, mac, ByteBus::Ds1685(rtc.clone())));
        ioc3.set_microlan(identity.ioc3_wire(mac));
        ioc3.set_heartbeat(heartbeat.clone());
        let eth = ioc3.eth().expect("the Octane's IOC3 has Ethernet").clone();
        pci.attach(slot(2), ioc3.clone());
        ioc3.connect_superio(pci);
        // ... and resetting itself through HEART.
        heart.on_soft_reset(Box::new(move || {
            let _ = event_tx.try_send(MachineEvent::HardReset);
        }));
        let rad = Arc::new(Rad1::new(&timer_manager));
        if !cfg.no_audio {
            rad.enable_output(cfg.audio.clone());
        }
        pci.attach(slot(3), rad.clone());
        // The status bar's activity bits and the CPU's Compare matches,
        // shared with the graphics board before the CPU exists.
        let fasttick = Arc::new(AtomicU64::new(0));
        let status = crate::status::MachineStatus::new(heartbeat.clone(), fasttick.clone());
        status.set_scsi_wide(true);
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
        phys.attach_huge_window(WIDGET_BRIDGE, bridge.huge_window());
        for b in &xio_boards {
            phys.attach_widget(b.widget(), b.bridge().clone(), false);
            phys.attach_long_window(b.widget(), b.bridge().long_window());
            phys.attach_huge_window(b.widget(), b.bridge().huge_window());
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
        events.set_cpus(world.clone());
        let weak = Arc::downgrade(&phys);
        let remap_world = world.clone();
        heart.on_memcfg(Box::new(move |m| {
            if let Some(p) = weak.upgrade() {
                remap_world.stop_others(|| {
                    p.remap_banks(m);
                    remap_world.layout_changed();
                });
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
        // DMA reaches only widgets the Crossbow has a live link to, and a
        // reset of the Bridge's link resets it.
        xbow.serve_bridge(WIDGET_BRIDGE, &bridge);

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
        // state through the core pointer it is called with. A lone CPU shares
        // its own pool too, so its flushes go through this world: a CPU
        // belongs to one.
        #[cfg(feature = "jitv2")]
        {
            #[cfg(feature = "jitv2_lockstep")]
            if cpu_count > 1 {
                return Err("[machine] cpus = 2 cannot run with jitv2_lockstep".into());
            }
            let jit = cpu.jitv2();
            // Every CPU sees memory at the same addresses: one address space.
            for (n, c) in cpus.iter().enumerate() {
                c.share_jitv2(jit.clone(), world.clone(), n, 0);
            }
        }
        status.set_cpu_cycles(cpu.cycles_ptr());
        status.set_count_hz(cpu.count_hz_atomic());
        // Each processor's clock is its CP0 Compare (HEART's counter is
        // IRIX's time base, not its tick): publish the rate it programs.
        cpu.publish_tick_rate(status.cpu(0).tick_rate());
        for (c, tick) in cpus.iter().zip(&ticks).skip(1) {
            let n = status.add_cpu_stats(c.cycles_ptr(), tick.clone());
            c.publish_tick_rate(status.cpu(n).tick_rate());
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
                    sp.add_key_sink(v, c.tcache_key_ptr());
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
        crate::cpu::jitv2::jitv2::attach_pool(
            &cpu,
            cfg.jitv2_pool_size(),
            cfg.jitv2.compile,
            phys.phys.clone(),
            world.clone(),
        );

        // HEART's levels 0 to 4 are each CPU's IP2 to IP6.
        for (id, c) in cpus.iter().enumerate() {
            heart.set_cpu_interrupts(id, c.interrupts_ptr());
            heart.set_cpu_park(id, c.park_slot());
        }

        let monitor = crate::monitor::Monitor::new();
        monitor.register_device(crate::devlog::init_devlog());
        monitor.register_device(status.clone());
        monitor.register_device(Arc::new(crate::cpu::cpu_monitor::CpuMonitor::new(cpus.clone())));
        monitor.register_device(heart.clone());
        monitor.register_device(crate::pci::pci_host_commands("Bridge", &[bridge.bus()], None));
        for b in &xio_boards {
            monitor.register_device(crate::pci::pci_host_commands(b.kind().label(), &[b.bus()], None));
        }
        // `nvme`, and `scsi2` and `cow2` for the Fibre Channel loop's disks.
        for c in cards.devices() {
            monitor.register_device(c);
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
        monitor.register_device(rad.clone());
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
        Self::connect_serial(cfg, &ioc3, ci_serial.as_ref())?;

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
            cards,
            flash,
            gfx,
            vpro,
            nat,
            frames,
            scratch_path,
            nvram_path: cfg.nvram.clone(),
            events,
            event_rx: Some(event_rx),
            controller_slot: None,
            lifecycle: crate::machine_state::GuestState::Idle,
            disks,
            restored: Default::default(),
        })
    }

    /// Point every switched port's windows at its widget or at the bus
    /// error, as the Crossbow's links stand.
    fn sync_links(phys: &Ip30Bus, xbow: &Xbow) {
        for w in SWITCHED_WIDGETS {
            phys.set_widget_link(w, xbow.carries(w));
        }
    }

    /// The SCSI devices: scsi0 is the internal bus, scsi1 the external one.
    /// Returns the `scratch = true` volume, if any, and the disks as a
    /// snapshot records them.
    fn attach_scsi(cfg: &MachineConfig, resolved: &crate::topology::DevicePlacement, buses: [&crate::scsi::ScsiBus; 2]) -> Result<(Option<std::path::PathBuf>, Vec<crate::snapshot::DiskRef>), String> {
        let attached = crate::scsi::attach_from_config(cfg, Some("ip30"), resolved, &buses)?;
        if let Some(d) = attached.daynaports.first() {
            return Err(format!("{}: kind = \"daynaport\" is not available on the Octane", crate::config::scsi_label(d.controller, d.id)));
        }
        for w in Self::unprobed_internal_ids(&attached.tables) {
            eprintln!("ip30: warning: {w}");
        }
        Ok((attached.scratch, crate::snapshot::DiskRef::from_tables(&attached.tables, false)))
    }

    /// The PROM and IRIX look for devices on the internal bus (controller
    /// 0) only at ids 1-3, its three drive bays; ids 1-15 are probed on the
    /// external bus. One warning for each table attached where nothing will
    /// look for it.
    fn unprobed_internal_ids(tables: &[crate::scsi::TableRef]) -> Vec<String> {
        tables.iter()
            .filter(|t| t.slot < 16 && !(1..=3).contains(&t.slot))
            .map(|t| {
                let what = if t.is_cdrom { "CD-ROM" } else { "disk" };
                format!(
                    "{}: the Octane probes the internal bus only at ids 1-3, so the guest will not find this {what}; \
                     put it on the external bus with controller = 1",
                    crate::config::scsi_label(0, t.slot)
                )
            })
            .collect()
    }

    /// IOC3's first port is the console unless `machine.console` names the second.
    fn connect_serial(cfg: &MachineConfig, ioc3: &Ioc3, ci: Option<&Arc<crate::dev::serial_host::CiSerialBackend>>) -> Result<(), String> {
        let serial = |board, n| crate::dev::serial_dev::connect(cfg, "Octane", board, n, ci, None);
        ioc3.set_serial_backend(0, serial("IOC3 A", 1)?);
        ioc3.set_serial_backend(1, serial("IOC3 B", 2)?);
        Ok(())
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
        d.extend(self.cards.devices());
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
        self.timer_manager.clock().stop();
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
        self.events.powered_on();
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
        self.cards.power_on();
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
        self.timer_manager.clock().start();
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
// Not supported, and refused with the reason: two processors (the stop does
// not yet bring both to a common instruction boundary that a snapshot can
// record). In a jitv2 build, putting memory back bumps every page's generation,
// so code compiled from the memory before the load is not run.

/// Why an Octane with this configuration cannot be saved, when it cannot.
fn snapshot_refusal_for(_impact: bool, _vpro: bool, cpus: usize) -> Option<String> {
    if cpus > 1 {
        return Some("snapshots of a two-processor Octane are not supported yet: \
            the stop does not bring both processors to one recorded instruction boundary".into());
    }
    None
}

/// The `machine` an Octane snapshot's manifest names.
pub(crate) const SNAPSHOT_MACHINE: &str = "octane_ip30";

impl OctaneMachine {
    /// The SCSI buses that hold disks: both controllers', the NVMe cards'
    /// namespaces and the Fibre Channel loop.
    fn disk_buses(&self) -> Vec<&crate::scsi::ScsiBus> {
        use crate::scsi_monitor::ScsiTargets;
        self.scsi
            .iter()
            .map(|c| c.bus())
            .chain(self.cards.nvme.iter().map(|c| c.nvme.scsi_bus()))
            .chain(self.cards.fibre.iter().map(|f| f.fc.scsi_bus()))
            .collect()
    }

    /// Every device's state but the CPU's. Called with the machine stopped.
    fn save_board(&self) -> toml::Value {
        let lines = self.flash_lines.iter().map(|f| f.save_state()).collect();
        let mut t = StateTable::new();
        // Only with boards, so a machine without them saves what it always did.
        if !self.xio_boards.is_empty() {
            let boards = self.xio_boards.iter().map(|b| (format!("xio{}", b.widget()), b.bridge().save_state()));
            t.value("xio_boards", toml::Value::Table(boards.collect()));
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
            .u64("clock_ns", self.timer_manager.clock().now())
            .build()
    }

    /// Put back what [`Self::save_board`] gave, into a machine stopped and
    /// reset: leaf devices first and HEART last, so the pins it drives into
    /// the CPU, already loaded, come from its loaded registers, and memory
    /// is placed from its loaded descriptors.
    fn load_board(&self, v: &toml::Value, cards: &toml::Value) -> Result<(), String> {
        let load = |key: &str, dev: &dyn Saveable| dev.load_state(field(v, key)?).map_err(|e| format!("{key}: {e}"));
        // The machine is stopped, so the clock holds at the saved reading while the devices re-arm.
        self.timer_manager.clock().restore(crate::snapshot::field_u64(v, "clock_ns")?);
        load("ioc3", &*self.ioc3)?;
        load("eth", &*self.eth)?;
        load("scsi0", &*self.scsi[0])?;
        load("scsi1", &*self.scsi[1])?;
        load("rad", &*self.rad)?;
        self.cards.load_state(cards)?;
        load("flash", &*self.flash)?;
        let lines = field(v, "flash_lines")?.as_array().filter(|l| l.len() == self.flash_lines.len())
            .ok_or("flash_lines: not one entry per processor")?;
        for (f, saved) in self.flash_lines.iter().zip(lines) {
            f.load_state(saved)?;
        }
        load("bridge", &*self.bridge)?;
        if !self.xio_boards.is_empty() {
            let boards = field(v, "xio_boards")?;
            for b in &self.xio_boards {
                let at = format!("xio{}", b.widget());
                b.bridge().load_state(field(boards, &at).map_err(|_| format!("snapshot has no state for the board at {at}"))?)
                    .map_err(|e| format!("xio_boards: {e}"))?;
            }
        }
        load("xbow", &*self.xbow)?;
        Self::sync_links(&self.phys, &self.xbow);
        load("heart", &*self.heart)?;
        Ok(())
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
    fn monitor(&self) -> &Arc<crate::monitor::Monitor> {
        &self.monitor
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
    fn snapshot_refusal(&self) -> Option<String> {
        snapshot_refusal_for(self.gfx.is_some(), self.vpro.is_some(), self.cpus.len()).or_else(|| self.cards.snapshot_blocker())
    }
    fn snapshot_layout(&self) -> crate::snapshot::Layout {
        use crate::snapshot::BoardList;
        let cpu = &self.cpus[0];
        let mut states = vec!["cpu".to_string(), "octane".into()];
        states.extend((!self.cards.is_empty()).then(|| "cards".to_string()));
        let boards = [self.gfx.is_some().then_some(("impact", BoardList::Impact)), self.vpro.is_some().then_some(("vpro", BoardList::Vpro))];
        crate::snapshot::Layout {
            machine: Some(SNAPSHOT_MACHINE),
            disks: self.disks.clone(),
            slots: self.cards.manifest(),
            nvram: Some(self.nvram_path.clone()),
            cpu: (cpu.model_name().into(), cpu.cache_kind().into()),
            states,
            files: vec![("flash.bin".into(), self.flash.size())],
            memory: vec![self.ram.bank_words().iter().map(|b| b.len()).sum()],
            boards: boards.into_iter().flatten().collect(),
            ..Default::default()
        }
    }
    fn capture(&self) -> Result<crate::snapshot::Image<'_>, String> {
        use crate::snapshot::{Board, BoardList};
        let mut states = vec![("cpu".to_string(), self.cpus[0].save_state()), ("octane".into(), self.save_board())];
        if !self.cards.is_empty() {
            states.push(("cards".into(), self.cards.save_state()));
        }
        let impact = self.gfx.as_ref().map(|gfx| gfx.snapshot_parts().map(|p| Board::new("impact", BoardList::Impact, p))).transpose()?;
        let vpro = self.vpro.as_ref().map(|vpro| vpro.snapshot_parts().map(|p| Board::new("vpro", BoardList::Vpro, p))).transpose()?;
        Ok(crate::snapshot::Image {
            states,
            files: vec![("flash.bin".into(), self.flash.contents())],
            memory: self.ram.bank_words().into_iter().map(|w| (0, w.into())).collect(),
            boards: impact.into_iter().chain(vpro).collect(),
            ..Default::default()
        })
    }
    fn apply_image(&mut self, image: &crate::snapshot::Image) -> Result<(), String> {
        self.power_on_all();
        self.cpus[0].load_state(image.state("cpu")?).map_err(|e| format!("cpu: {e}"))?;
        self.flash.restore_contents(image.file("flash.bin")?)?;
        self.ram.restore_bank_words(&image.memory[0].1)?;
        let none = toml::Value::Table(Default::default());
        self.load_board(image.state("octane")?, image.state("cards").unwrap_or(&none))?;
        if let (Some(gfx), Some(b)) = (&self.gfx, image.board("impact")) {
            gfx.restore_snapshot_parts(&b.metadata, &b.parts)?;
        }
        if let (Some(vpro), Some(b)) = (&self.vpro, image.board("vpro")) {
            vpro.restore_snapshot_parts(&b.metadata, &b.parts)?;
        }
        Ok(())
    }
    fn with_disk_buses(&self, f: &mut dyn FnMut(&[crate::snapshot::DiskBus]) -> Result<(), String>) -> Result<(), String> {
        let places = crate::topology::scsi_buses(&MachineProfile::OctaneIp30.slots()).into_iter().map(|b| b.1);
        f(&crate::snapshot::disk_buses(places, self.scsi.iter().map(|c| c.bus()), Some(&self.cards)))
    }
    fn snapshots(&mut self) -> Option<&mut crate::snapshot::Restored> {
        Some(&mut self.restored)
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
    fn disc_changer(&self) -> Option<Arc<dyn crate::workstation::DiscChanger>> {
        Some(Arc::new(crate::workstation::DiscChangers(self.scsi.iter().map(|c| c.clone() as _).collect())))
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.disk_buses().iter().map(|b| b.pending_chd_sync_count()).sum()
    }
    fn disk_activity(&self) -> Vec<Vec<crate::scsi::DiskCounts>> {
        self.disk_buses().iter().map(|b| b.activity()).collect()
    }
    fn read_disk(&self, bus: usize, id: u8, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.disk_buses().get(bus).ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "no such disk bus"))?
            .read_media(id, offset, buf)
    }
    fn disk_size(&self, bus: usize, id: u8) -> Option<u64> {
        self.disk_buses().get(bus).and_then(|b| b.media_size(id))
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
                    "descriptors {:08x} {:08x} {:08x} {:08x}; {} MB installed\n",
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
                    "interrupt destination widget {wid} address {addr:#x}; enable {:#04x} status {:#010x}; \
                     DMA writes that reached no memory: {}\n",
                    self.bridge.register(reg::INT_ENABLE),
                    self.bridge.register(reg::INT_STATUS),
                    self.bridge.lost_writes()
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

    /// Only internal-bus ids outside the three drive bays draw a warning.
    #[test]
    fn internal_bus_tables_past_id_3_are_warned_about() {
        let table = |slot, is_cdrom| crate::scsi::TableRef { slot, at: crate::slot_path::MachineSlotPath::root(), is_cdrom, path: String::new(), size_bytes: 0 };
        let tables = [table(1, false), table(3, false), table(4, true), table(0, false), table(20, true), table(17, false)];
        let w = OctaneMachine::unprobed_internal_ids(&tables);
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(w[0].starts_with("SCSI ID 4:") && w[0].contains("CD-ROM"), "{}", w[0]);
        assert!(w[1].starts_with("SCSI ID 0:") && w[1].contains("disk"), "{}", w[1]);
    }

    /// The Octane2 carries its own four identity records, from one reference
    /// unit's inventory, and the Octane keeps the ones it had. The IOC3's
    /// wire holds the Ethernet-address part, the front plane and the power
    /// supply; HEART's and the Bridge's hold one record each.
    #[test]
    fn each_product_carries_its_own_identity_records() {
        let o = Identity::octane(Identity::processor_module(&parts::R10000_195, 1));
        let o2 = Identity::octane2(Identity::processor_module(&parts::R12000_360, 1));
        let key = |i: &Identity| {
            [&i.processor, &i.system_board, &i.front_plane, &i.power_supply].map(|r| (r.part, r.revision, r.name.to_string()))
        };
        assert_eq!(
            key(&o),
            [
                ("030-0888-004", "B", "PM10".into()),
                ("030-0887-003", "F", "IP30".into()),
                ("030-0891-003", "A", "FP1".into()),
                ("060-0028-003", "A", "PWR.SPPLY.SR".into()),
            ]
        );
        assert_eq!(
            key(&o2),
            [
                ("030-1590-002", "B", "PM10360MHZ".into()),
                ("030-1467-001", "B", "IP30".into()),
                ("030-0891-003", "H", "FP1".into()),
                ("060-0035-002", "A", "PWR.SPPLY.ER".into()),
            ]
        );
        // The module follows the processor settings, never the graphics board.
        let module = |p: &Part, n: u8| Identity::processor_module(p, n).name.to_string();
        assert_eq!(module(&parts::R10000_195, 2), "PM20", "two R10000s");
        assert_eq!(module(&parts::R12000_360, 2), "PM20360MHZ", "two 360 MHz R12000s");
        assert_eq!(module(&parts::R14000_600, 1), "PM10600MHZ", "one 600 MHz R14000");
        // octopus, a real dual 600 MHz R14000 Octane2.
        let r14k = Identity::processor_module(&parts::R14000_600, 2);
        assert_eq!((r14k.part, r14k.revision, &*r14k.name), ("030-1778-001", "C", "PM20600MHZ"));
        let families = |bus: &w1::Bus| {
            let mut f: Vec<u8> = bus.roms().iter().map(|rom| rom[0]).collect();
            f.sort();
            f
        };
        assert_eq!(families(&o2.ioc3_wire([8, 0, 0x69, 1, 2, 3])), [0x0b, 0x0b, 0x91]);
        assert_eq!(families(&o2.processor.bus()), [0x0b]);
        assert_eq!(families(&o2.system_board.bus()), [0x0b]);
    }

    /// The VPro compatibility check reads the PROM's revision from its banner;
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

    /// The Octane's PROM goes through the shared loader: a file longer than
    /// the 1 MiB flash is refused with the machine named, and a shorter raw
    /// one still runs from the start of the flash.
    #[test]
    fn the_octane_prom_is_bounded_by_its_flash() {
        use crate::config::{GraphicsBoard, MachineProfile};
        use crate::display_stop_tests::{big_stack, config, scratch};
        let dir = scratch("octane-prom-size");
        let cfg = config(&dir, MachineProfile::OctaneIp30, GraphicsBoard::None);
        std::fs::write(&cfg.prom, vec![0u8; FLASH_SIZE + 1]).unwrap();
        let big = cfg.clone();
        let e = big_stack(move || OctaneMachine::new(&big).err()).expect("an oversized PROM is refused");
        assert!(e.starts_with("Octane PROM image") && e.contains("exceeds flash capacity"), "{e}");
        std::fs::write(&cfg.prom, vec![0x5a; 0x1000]).unwrap();
        assert!(big_stack(move || OctaneMachine::new(&cfg).is_ok()), "a short raw PROM loads");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod snapshot_tests {
    use super::*;
    use crate::snapshot::Snapshot;
    use crate::workstation::Workstation;
    use crate::cpu::mips_exec::CpuStateDigest;

    /// The stock PROM: the file `IRIS_IP30_PROM` names (to try another release), else the
    /// PROM built into iris.
    fn prom() -> String {
        std::env::var("IRIS_IP30_PROM").unwrap_or_else(|_| "(embedded)".into())
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

    /// A `pci_shoehorn` or `pci_cage` entry, at `at` or anywhere.
    fn xio_board(kind: crate::dev::xio_pci::Kind, at: Option<&str>) -> crate::devices::ConfiguredDevice {
        crate::devices::ConfiguredDevice { at: at.map(Into::into), ..crate::devices::ConfiguredDevice::new("", crate::devices::DeviceModel::XioBoard(kind)) }
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
        let held = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        cfg.monitor_port = Some(held.local_addr().unwrap().port());
        let disk = dir.join("disk.raw");
        if !disk.exists() {
            std::fs::write(&disk, vec![0u8; 4 << 20]).unwrap();
        }
        cfg.devices.clear();
        crate::dev::serial_host::tests::free_ports(&mut cfg);
        cfg.set_disk(1, toml::from_str(&format!("path = {:?}\noverlay = true", disk.to_string_lossy())).unwrap());
        edit(&mut cfg);
        OctaneMachine::new(&cfg).expect("an Octane")
    }

    /// The monitor answers `audio status` on the Octane for RAD1, which the
    /// machine registers with it.
    #[test]
    fn the_monitor_answers_audio_status_with_the_rad1_card() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("audio-cmd");
            let prom = empty_prom(&dir);
            let m = machine_with(&prom, &dir, |cfg| cfg.no_audio = true);
            let text = m.monitor.audio_status();
            assert!(text.starts_with("audio: RAD1\n") && text.ends_with("host output: none\n"), "{text}");
        }).unwrap().join().unwrap();
    }

    /// An Octane with a PCI module and shoehorns: each board's Crossbow link
    /// comes up and its Bridge answers on its own port, a function in each
    /// slot is reached by a configuration read through the processor's
    /// address, its interrupt arrives at HEART and its DMA lands in RAM. Ports
    /// with nothing attached stay down, and so do the device numbers a board
    /// has no slot at.
    #[test]
    fn option_boards_are_attached_on_their_ports() {
        use crate::config::MachineProfile;
        use crate::dev::bridge::{reg, CFG0_BASE, DIRECT_ADD_512MB};
        use crate::dev::xio_pci::Kind;
        use crate::pci::testing::TestCard;
        use crate::pci::{Bar, PciFunction, PCI_INTA};
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("xio-boards");
            let prom = empty_prom(&dir);
            let m = machine_with(&prom, &dir, |cfg| {
                assert_eq!(cfg.machine.profile, MachineProfile::OctaneIp30);
                cfg.devices.extend([
                    xio_board(Kind::Shoehorn, None),
                    xio_board(Kind::Cage, None),
                    xio_board(Kind::Shoehorn, Some("xio9")),
                ]);
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
                    b.bus().attach(DevFn { device, function: 0 }, card.clone());
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
    #[cfg(feature = "embedded-proms")]
    #[test]
    fn a_restored_octane_continues_from_the_saved_state() {
        let prom = prom();
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || {
            let dir = scratch("continue");
            let mut a = machine(&prom, &dir);
            a.cpus[0].step_n_inline(2_000_000).unwrap();
            crate::snapshot::save_in(&mut a, &dir.join("saves"), "at-n").unwrap();
            let saved = digest(&a);
            let mut b = machine(&prom, &dir);
            let snap = crate::snapshot::read(&b, &dir.join("saves"), "at-n").unwrap();
            b.apply_image(&snap.image).unwrap();
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
        use crate::config::NvmeConfig;
        use crate::devices::{DeviceModel, ConfiguredDevice};
        use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};
        use crate::scsi_monitor::ScsiTargets;
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || {
            let dir = scratch(&format!("nvme-{}-{}", kind.name(), slot.unwrap_or(0)));
            let prom = empty_prom(&dir);
            let disk = dir.join("nvme.raw");
            std::fs::write(&disk, vec![0u8; 1 << 20]).unwrap();
            let path = disk.to_string_lossy().into_owned();
            let mut m = machine_with(&prom, &dir, |cfg| {
                let mut at = format!("board/pci0{}", slot.map_or(String::new(), |s| format!("/{s}")));
                cfg.devices.push(ConfiguredDevice { name: Some("board".into()), ..xio_board(kind, None) });
                if bridge {
                    cfg.devices.push(ConfiguredDevice { name: Some("br".into()), ..ConfiguredDevice::new(&at, DeviceModel::PciBridge) });
                    at = "br/pci0/0".into();
                }
                cfg.devices.push(ConfiguredDevice::new(&at, DeviceModel::Nvme(NvmeConfig { path, overlay: true, ..Default::default() })));
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
            let card_at = m.cards.nvme[0].at.clone();
            assert!(m.disks.iter().any(|d| d.at == card_at && d.size_bytes == 1 << 20), "the disk is recorded at the card");

            let nvme = m.cards.nvme[0].nvme.clone();
            let io = |cdb: Vec<u8>, data: Option<Vec<u8>>| {
                let req = ScsiRequest { cdb, data_len: ScsiDataLength::Unlimited, data_in: data };
                nvme.scsi_bus().request(0, &req).unwrap().unwrap().data
            };
            let rw = |op: u8| vec![op, 0, 0, 0, 0, 3, 0, 0, 1, 0];
            io(rw(scsi_cmd::WRITE_10), Some(vec![0x11; 512]));
            crate::snapshot::save_in(&mut m, &dir.join("saves"), "s").unwrap();
            assert!(dir.join("saves/s").join(card_at.to_string()).join("scsi0.overlay").exists(), "the overlay goes with the snapshot");
            io(rw(scsi_cmd::WRITE_10), Some(vec![0x22; 512]));
            assert_eq!(io(rw(scsi_cmd::READ_10), None), vec![0x22; 512]);
            crate::snapshot::load_in(&mut m, &dir.join("saves"), "s").unwrap();
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
    fn an_nvme_card_is_attached_behind_a_bridge_in_a_cage_slot() {
        nvme_in_a_slot(crate::dev::xio_pci::Kind::Cage, Some(2), true);
    }

    #[test]
    fn an_nvme_card_is_attached_in_the_cages_last_slot() {
        nvme_in_a_slot(crate::dev::xio_pci::Kind::Cage, Some(3), false);
    }

    #[test]
    fn an_nvme_card_is_attached_in_a_shoehorn() {
        nvme_in_a_slot(crate::dev::xio_pci::Kind::Shoehorn, None, false);
    }

    /// Two NVMe cards, a virtio-serial and an audio card in one Octane: a snapshot keeps
    /// each where it was placed, with its disk, and takes them all back.
    #[test]
    fn several_cards_round_trip_a_machine_snapshot() {
        use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};
        use crate::scsi_monitor::ScsiTargets;
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("cards");
            let prom = empty_prom(&dir);
            let nvme = |n: u8| {
                let disk = dir.join(format!("nvme{n}.raw"));
                std::fs::write(&disk, vec![0u8; 1 << 20]).unwrap();
                crate::devices::ConfiguredDevice::new(&format!("cage/pci0/{n}"), crate::devices::DeviceModel::Nvme(crate::config::NvmeConfig { path: disk.to_string_lossy().into_owned(), overlay: true, ..Default::default() }))
            };
            let mut m = machine_with(&prom, &dir, |cfg| {
                cfg.no_audio = true;
                cfg.devices.push(crate::devices::ConfiguredDevice { name: Some("cage".into()), ..xio_board(crate::dev::xio_pci::Kind::Cage, None) });
                cfg.devices.extend([nvme(1), nvme(2), crate::devices::ConfiguredDevice::new("cage/pci0/3", crate::devices::DeviceModel::Ice1724)]);
                cfg.devices.push(crate::devices::ConfiguredDevice::from_table(toml::from_str("type = \"virtio_serial\"").unwrap()).unwrap());
            });
            let at = |s: &str| s.parse::<crate::slot_path::MachineSlotPath>().unwrap();
            let want = [("nvme", "xio13/pci0/1"), ("nvme", "xio13/pci0/2"), ("virtio_serial", "xio15/pci0/7"), ("ice1724", "xio13/pci0/3")];
            assert_eq!(m.cards.manifest(), want.map(|(t, a)| (t.to_string(), at(a))));
            let cards: Vec<_> = m.cards.nvme.iter().map(|c| c.nvme.clone()).collect();
            let io = |n: usize, cdb: Vec<u8>, data: Option<Vec<u8>>| {
                let req = ScsiRequest { cdb, data_len: ScsiDataLength::Unlimited, data_in: data };
                cards[n].scsi_bus().request(0, &req).unwrap().unwrap().data
            };
            let rw = |op: u8| vec![op, 0, 0, 0, 0, 3, 0, 0, 1, 0];
            io(0, rw(scsi_cmd::WRITE_10), Some(vec![0x11; 512]));
            io(1, rw(scsi_cmd::WRITE_10), Some(vec![0x12; 512]));
            crate::snapshot::save_in(&mut m, &dir.join("saves"), "s").unwrap();
            let saved = Snapshot::new(dir.join("saves/s")).read_manifest().unwrap().unwrap();
            assert_eq!(saved.devices, m.cards.manifest());
            assert_eq!(saved.disks.iter().filter(|d| d.at.to_string().starts_with("xio13/pci0/")).count(), 2, "both cards' disks");
            for n in [1, 2] {
                assert!(dir.join(format!("saves/s/xio13/pci0/{n}/scsi0.overlay")).exists());
            }
            io(0, rw(scsi_cmd::WRITE_10), Some(vec![0x21; 512]));
            io(1, rw(scsi_cmd::WRITE_10), Some(vec![0x22; 512]));
            crate::snapshot::load_in(&mut m, &dir.join("saves"), "s").unwrap();
            assert_eq!(io(0, rw(scsi_cmd::READ_10), None), vec![0x11; 512], "the first card's disk is as saved");
            assert_eq!(io(1, rw(scsi_cmd::READ_10), None), vec![0x12; 512], "and the second's");
            m.stop();
        }).unwrap().join().unwrap();
    }

    /// A `qla2200` in an option board's slot, with its disks on the loop, started, stopped and recorded with the other parts.
    fn fibre_in_a_slot(kind: crate::dev::xio_pci::Kind, slot: Option<u8>) {
        use crate::scsi::{scsi_cmd, ScsiDataLength, ScsiRequest};
        use crate::scsi_monitor::ScsiTargets;
        std::thread::Builder::new().stack_size(64 << 20).spawn(move || {
            let dir = scratch(&format!("fc-{}-{}", kind.name(), slot.unwrap_or(0)));
            let prom = empty_prom(&dir);
            let slot = slot.unwrap_or(1);
            let mut loop_disks = Vec::new();
            for id in [3u8, 9] {
                let disk = dir.join(format!("loop{id}.raw"));
                std::fs::write(&disk, vec![0u8; 1 << 20]).unwrap();
                let table = format!("type = \"hdd\"\nat = \"cage/pci0/{slot}/loop{id}\"\npath = '{}'\noverlay = true", disk.display());
                loop_disks.push(crate::devices::ConfiguredDevice::from_table(toml::from_str(&table).unwrap()).unwrap());
            }
            let mut m = machine_with(&prom, &dir, |cfg| {
                cfg.devices.push(crate::devices::ConfiguredDevice { name: Some("cage".into()), ..xio_board(kind, None) });
                cfg.devices.push(crate::devices::ConfiguredDevice::new(&format!("cage/pci0/{slot}"), crate::devices::DeviceModel::Qla2200));
                cfg.devices.extend(loop_disks);
            });
            let at = DevFn { device: slot, function: 0 };
            let class = m.xio_boards[0].bus().config_read(at, 8, 4).map(|d| d >> 8);
            assert_eq!(class, Some(0x0c_0400), "the Fibre Channel card in the slot");
            let onboard: Vec<u8> = (0..8).filter(|&d| m.bridge.bus().occupied(d)).collect();
            assert_eq!(onboard, [0, 1, 2, 3], "the onboard Bridge keeps its own devices");
            for id in [3u8, 9] {
                let want = m.cards.fibre[0].at.loop_id(id);
                assert!(m.disks.iter().any(|d| d.at == want && d.size_bytes == 1 << 20), "disk {id} in the manifest");
            }

            let fc = m.cards.fibre[0].fc.clone();
            let io = |id: u8, cdb: Vec<u8>, data: Option<Vec<u8>>| {
                let req = ScsiRequest { cdb, data_len: ScsiDataLength::Unlimited, data_in: data };
                fc.scsi_bus().request(id, &req).unwrap().unwrap().data
            };
            let rw = |op: u8| vec![op, 0, 0, 0, 0, 3, 0, 0, 1, 0];
            io(3, rw(scsi_cmd::WRITE_10), Some(vec![0x11; 512]));
            io(9, rw(scsi_cmd::WRITE_10), Some(vec![0x99; 512]));
            crate::snapshot::save_in(&mut m, &dir.join("saves"), "s").unwrap();
            assert!(dir.join("saves/s").join(&m.cards.fibre[0].at.to_string()).is_dir(), "the overlays go with the snapshot");
            io(3, rw(scsi_cmd::WRITE_10), Some(vec![0x22; 512]));
            assert_eq!(io(3, rw(scsi_cmd::READ_10), None), vec![0x22; 512]);
            crate::snapshot::load_in(&mut m, &dir.join("saves"), "s").unwrap();
            assert_eq!(io(3, rw(scsi_cmd::READ_10), None), vec![0x11; 512], "disk 3 is as saved");
            assert_eq!(io(9, rw(scsi_cmd::READ_10), None), vec![0x99; 512], "and so is disk 9");
            m.stop();
        }).unwrap().join().unwrap();
    }

    #[test]
    fn a_fibre_channel_card_is_attached_in_a_cage_slot() {
        fibre_in_a_slot(crate::dev::xio_pci::Kind::Cage, Some(2));
    }

    /// HEART's software reset asked for before a restore belongs to the
    /// machine the restore replaced: the event thread must not act on it.
    #[test]
    fn a_restore_drops_a_reset_the_guest_asked_for_before_it() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("restore-reset");
            let prom = empty_prom(&dir);
            let mut m = machine(&prom, &dir);
            crate::snapshot::save_in(&mut m, &dir.join("saves"), "s").unwrap();
            let queue = m.event_rx.take().unwrap();
            queue.listen();
            assert!(m.heart.piu_write(0, 8, 0));
            assert!(m.heart.piu_write(0, 8, 1 << 22));
            let (sent, event) = queue.recv().unwrap();
            assert!(matches!(event, MachineEvent::HardReset) && queue.is_current(sent));
            assert_ne!(m.cpus[0].park_slot().holds() & crate::cpu::smp::HOLD_RESET, 0, "the CPU stops at the request");
            crate::snapshot::load_in(&mut m, &dir.join("saves"), "s").unwrap();
            assert!(!queue.is_current(sent), "the restore replaced the machine that asked");
            assert_eq!(m.cpus[0].park_slot().holds() & crate::cpu::smp::HOLD_RESET, 0, "and lets the CPU go");
            m.stop();
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
            crate::snapshot::save_in(&mut a, &dir.join("saves"), "bare").unwrap();
            let (at_save, cpu_at_save) = (digest(&a), a.cpus[0].save_state());
            let saved = crate::snapshot::read(&a, &dir.join("saves"), "bare").unwrap();
            let board = saved.image.state("octane").unwrap().clone();
            // Move on, then load what was saved.
            a.cpus[0].step_n_inline(1000).unwrap();
            pattern(&a, 0xa5a5_0000);
            assert_ne!(digest(&a), at_save);
            a.stop();
            a.apply_image(&saved.image).unwrap();
            assert_eq!(digest(&a), at_save);
            assert_eq!(a.cpus[0].save_state(), cpu_at_save);
            assert_eq!(a.timer_manager.clock().now(), crate::snapshot::field_u64(&board, "clock_ns").unwrap());
            // The RTC runs in host time while the machine is stopped, as on
            // every machine. Everything else comes back as saved.
            let (mut board, mut got) = (board, a.save_board());
            for t in [&mut board, &mut got] {
                t["ioc3"]["chip"].as_table_mut().unwrap().remove("rtc");
            }
            assert_eq!(got, board);
            // A snapshot from before the machine clock was saved is refused.
            let mut old = crate::snapshot::read(&a, &dir.join("saves"), "bare").unwrap();
            old.image.states.iter_mut().find(|s| s.0 == "octane").unwrap().1.as_table_mut().unwrap().remove("clock_ns");
            a.stop();
            assert!(a.apply_image(&old.image).unwrap_err().contains("clock_ns"));
        }).unwrap().join().unwrap();
    }

    /// The Octane, except that it cannot capture its state.
    struct NoCapture(OctaneMachine);

    impl Workstation for NoCapture {
        fn start(&mut self) { self.0.start() }
        fn stop(&mut self) { self.0.stop() }
        fn reset(&mut self) { self.0.reset() }
        fn restart_peripherals(&mut self) { self.0.restart_peripherals() }
        fn cpu_start(&self) { self.0.cpu_start() }
        fn cpu_is_running(&self) -> bool { self.0.cpu_is_running() }
        fn machine_slot(&mut self) -> crate::workstation::MachineSlot { self.0.machine_slot() }
        fn register_system_controller(&mut self) { self.0.register_system_controller() }
        fn monitor(&self) -> &Arc<crate::monitor::Monitor> { self.0.monitor() }
        fn lifecycle(&self) -> crate::machine_state::GuestState { self.0.lifecycle() }
        fn set_lifecycle(&mut self, state: crate::machine_state::GuestState) { self.0.set_lifecycle(state) }
        fn get_display(&self) -> Option<Arc<dyn crate::gfx_display::GfxDisplay>> { self.0.get_display() }
        fn get_ps2(&self) -> Arc<crate::dev::ps2::Ps2Controller> { self.0.get_ps2() }
        fn register_monitor_device(&self, device: Arc<dyn crate::traits::Device>) { self.0.register_monitor_device(device) }
        fn get_timer_manager(&self) -> Arc<TimerManager> { self.0.get_timer_manager() }
        fn get_cpu_debug(&self) -> Arc<dyn crate::gdb_stub::CpuDebug> { self.0.get_cpu_debug() }
        fn fasttick_count(&self) -> Arc<std::sync::atomic::AtomicU64> { self.0.fasttick_count() }
        fn get_ci_serial(&self) -> Option<Arc<crate::dev::serial_host::CiSerialBackend>> { self.0.get_ci_serial() }
        fn inject_serial_console(&self, bytes: &[u8]) { self.0.inject_serial_console(bytes) }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
        fn flush_disks(&self) -> std::io::Result<()> { self.0.flush_disks() }
        fn snapshot_refusal(&self) -> Option<String> { None }
        fn snapshot_layout(&self) -> crate::snapshot::Layout { self.0.snapshot_layout() }
        fn capture(&self) -> Result<crate::snapshot::Image<'_>, String> { Err("no capture".into()) }
    }

    /// A load that fails while applying leaves the machine stopped.
    #[test]
    fn a_failed_load_leaves_the_machine_stopped() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("failed-load");
            let (prom, saves) = (empty_prom(&dir), dir.join("saves"));
            let mut m = machine(&prom, &dir);
            crate::snapshot::save_in(&mut m, &saves, "s").unwrap();
            let good = crate::snapshot::read(&m, &saves, "s").unwrap();
            crate::snapshot::restore(&mut m, &good, true).unwrap();
            assert!(m.cpu_is_running());
            let mut bad = crate::snapshot::read(&m, &saves, "s").unwrap();
            bad.image.states.iter_mut().find(|s| s.0 == "octane").unwrap().1.as_table_mut().unwrap().remove("clock_ns");
            assert!(crate::snapshot::restore(&mut m, &bad, true).is_err());
            assert!(!m.cpu_is_running());
            let clock = m.timer_manager.clock();
            let at = clock.now();
            std::thread::sleep(std::time::Duration::from_millis(20));
            assert_eq!(clock.now(), at, "the peripherals were not restarted");
        }).unwrap().join().unwrap();
    }

    /// A save whose capture fails leaves the earlier snapshot of that name loadable.
    #[test]
    fn a_failed_capture_leaves_the_earlier_snapshot_loadable() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("failed-capture");
            let (prom, saves) = (empty_prom(&dir), dir.join("saves"));
            let mut m = machine(&prom, &dir);
            crate::snapshot::save_in(&mut m, &saves, "s").unwrap();
            let mut m = NoCapture(m);
            assert_eq!(crate::snapshot::save_in(&mut m, &saves, "s").unwrap_err(), "no capture");
            crate::snapshot::load_in(&mut m.0, &saves, "s").unwrap();
        }).unwrap().join().unwrap();
    }

    /// A rollback returns to the last restore's state, and a save after it names that restore its parent.
    #[test]
    fn a_rollback_returns_to_the_last_restore() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("rollback");
            let (prom, saves) = (empty_prom(&dir), dir.join("saves"));
            let mut m = machine(&prom, &dir);
            assert!(crate::snapshot::ci_rollback(&mut m, &saves).is_err(), "nothing restored yet");
            crate::snapshot::save_in(&mut m, &saves, "s").unwrap();
            let memory_at_save = digest(&m).1;
            crate::snapshot::ci_restore(&mut m, &saves, "s").unwrap();
            m.stop();
            let mut words: Vec<u32> = m.ram.bank_words().concat();
            words[4099] = 0x5a5a_5a5a;
            m.ram.restore_bank_words(&words).unwrap();
            assert_ne!(digest(&m).1, memory_at_save);
            crate::snapshot::ci_rollback(&mut m, &saves).unwrap();
            m.stop();
            assert_eq!(digest(&m).1, memory_at_save);
            crate::snapshot::save_in(&mut m, &saves, "child").unwrap();
            let child = Snapshot::new(saves.join("child")).read_manifest().unwrap().unwrap();
            assert_eq!(child.parent.as_deref(), Some("s"));
            // A plain load makes its snapshot the next save's parent and the rollback target.
            crate::snapshot::load_in(&mut m, &saves, "child").unwrap();
            m.stop();
            m.ram.restore_bank_words(&words).unwrap();
            crate::snapshot::ci_rollback(&mut m, &saves).unwrap();
            m.stop();
            assert_eq!(digest(&m).1, memory_at_save);
            crate::snapshot::save_in(&mut m, &saves, "grandchild").unwrap();
            let grandchild = Snapshot::new(saves.join("grandchild")).read_manifest().unwrap().unwrap();
            assert_eq!(grandchild.parent.as_deref(), Some("child"));
        }).unwrap().join().unwrap();
    }

    /// Each case a save or load refuses gives its reason. (Decided without
    /// building a machine for each: every Octane a test builds holds one of
    /// the process's 64 device-map tables until the process ends.)
    #[test]
    fn each_refused_case_gives_its_reason() {
        assert_eq!(snapshot_refusal_for(false, false, 1), None);
        assert_eq!(snapshot_refusal_for(true, false, 1), None);
        assert_eq!(snapshot_refusal_for(false, true, 1), None);
        assert!(snapshot_refusal_for(false, false, 2)
            .unwrap()
            .contains("two-processor"));
    }

    /// The Octane wrapper and its IMPACT pipeline return to the saved point.
    #[test]
    fn an_octane_impact_board_round_trips_through_a_machine_snapshot() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("impact");
            let prom = empty_prom(&dir);
            let mut m = machine_with(&prom, &dir, |cfg| cfg.graphics.board = Some(crate::config::GraphicsBoard::SolidImpact));
            let gfx = m.gfx.as_ref().unwrap().clone();
            let board = gfx.mgras().clone();
            board.start_engines();
            let rss = |reg: u32, value: u32, exec: bool| {
                let addr = crate::dev::mgras::MGRAS_SLOT_GFX_BASE as u64
                    + 0x7C000 + 4 * reg as u64 + if exec { 0x1000 } else { 0 };
                loop {
                    let status = board.write32(addr, value);
                    if status != crate::traits::BUS_BUSY {
                        assert_eq!(status, crate::traits::BUS_OK);
                        break;
                    }
                    std::thread::yield_now();
                }
            };
            rss(0x112, 0xCAC, false);
            rss(0x115, 1023 << 16, false);
            rss(0x16E, 0x31E, false);
            rss(0x16D, 0x240, false);
            rss(0x163, 0xFFF, false);
            rss(0x161, 0x0C00_4504, false);
            rss(0x110, 1 << 20, false);
            rss(0x176, 0x12, false);
            rss(0x045, 0x18, false);
            rss(0x046, 25 << 16 | 35, false);
            rss(0x047, 25 << 16 | 35, true);
            assert_eq!(board.fb_pixel(25, 35), 0x12);
            let base = crate::ip30::widget_small_base(crate::ip30_machine::WIDGET_GFX);
            assert_eq!(gfx.write32(base + 0x24, 0xC1D2_8354), crate::traits::BUS_OK);
            crate::snapshot::save_in(&mut m, &dir.join("saves"), "x").unwrap();

            board.start_engines();
            rss(0x176, 0x56, false);
            rss(0x045, 0x18, false);
            rss(0x046, 25 << 16 | 35, false);
            rss(0x047, 25 << 16 | 35, true);
            assert_eq!(board.fb_pixel(25, 35), 0x56);
            assert_eq!(gfx.write32(base + 0x24, 0), crate::traits::BUS_OK);
            crate::snapshot::load_in(&mut m, &dir.join("saves"), "x").unwrap();
            assert_eq!(board.fb_pixel(25, 35), 0x12);
            assert_eq!(gfx.read32(base + 0x24).data, 0xC1D2_8354);
            m.stop();
        }).unwrap().join().unwrap();
    }

    /// The Octane wrapper persists both named VPro parts and restores the
    /// host-visible board state after its power-on reset.
    #[test]
    fn an_octane_vpro_board_round_trips_through_a_machine_snapshot() {
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let dir = scratch("vpro");
                let prom = empty_prom(&dir);
                let mut m = machine_with(&prom, &dir, |cfg| {
                    cfg.graphics.board = Some(crate::config::GraphicsBoard::V6)
                });
                let vpro = m.vpro.as_ref().unwrap().clone();
                let sw_token = widget_small_base(WIDGET_GFX) + 0x107c;
                vpro.start();
                assert_eq!(vpro.write32(sw_token, 0x51c2_93a4), crate::traits::BUS_OK);
                assert_eq!(vpro.read32(sw_token).data, 0x51c2_93a4);
                crate::snapshot::save_in(&mut m, &dir.join("saves"), "x").unwrap();

                let snap = Snapshot::new(dir.join("saves/x"));
                let chunks = snap.read_chunks_manifest().unwrap();
                let mut names: Vec<_> = chunks
                    .odyssey_chunks
                    .iter()
                    .map(|part| part.name.as_str())
                    .collect();
                names.sort_unstable();
                assert_eq!(names, ["state", "vram"]);
                assert!(chunks
                    .odyssey_chunks
                    .iter()
                    .all(|part| part.words != 0 && !part.hashes.is_empty()));

                vpro.start();
                assert_eq!(vpro.write32(sw_token, 0xa51c_2738), crate::traits::BUS_OK);
                assert_eq!(vpro.read32(sw_token).data, 0xa51c_2738);
                crate::snapshot::load_in(&mut m, &dir.join("saves"), "x").unwrap();
                assert_eq!(vpro.read32(sw_token).data, 0x51c2_93a4);
                m.stop();
            })
            .unwrap()
            .join()
            .unwrap();
    }

    /// A snapshot of another machine, or with a boot flash of another size,
    /// is refused before the machine is touched: it carries on as it was.
    /// (One machine for both: every Octane a test builds holds one of the
    /// process's 64 device-map tables until the process ends.)
    #[test]
    fn a_snapshot_of_another_memory_size_is_refused_before_the_machine_is_touched() {
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("refuse");
            let prom = empty_prom(&dir);
            let mut a = machine(&prom, &dir);
            crate::snapshot::save_in(&mut a, &dir.join("saves"), "other").unwrap();
            crate::snapshot::save_in(&mut a, &dir.join("saves"), "flash").unwrap();
            let path = dir.join("saves/other/snapshot.toml");
            let manifest = std::fs::read_to_string(&path).unwrap().replace("octane_ip30", "o2_ip32");
            std::fs::write(&path, manifest).unwrap();
            std::fs::write(dir.join("saves/flash/flash.bin"), vec![0u8; 4096]).unwrap();
            a.cpus[0].step_n_inline(100).unwrap();
            let before = digest(&a);
            assert!(crate::snapshot::load_in(&mut a, &dir.join("saves"), "other").unwrap_err().contains("another machine"));
            let e = crate::snapshot::load_in(&mut a, &dir.join("saves"), "flash").unwrap_err();
            assert!(e.contains("flash"), "{e}");
            assert_eq!(digest(&a), before, "the machine is as it was");
        }).unwrap().join().unwrap();
    }

    /// After a restore, a CPU stopped by a breakpoint or the monitor must show
    /// as stopped, and `machine-start` must run it again.
    #[test]
    fn a_restored_octane_can_be_started_again() {
        use crate::machine_state::GuestState;
        use crate::workstation::Workstation;
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("restore-state");
            let prom = empty_prom(&dir);
            let mut m = machine(&prom, &dir);
            crate::snapshot::save_in(&mut m, &dir.join("saves"), "s").unwrap();
            crate::snapshot::load_in(&mut m, &dir.join("saves"), "s").unwrap();
            assert_eq!(m.status().state, GuestState::Running);
            m.stop_cpus();
            assert_ne!(m.status().state, GuestState::Running);
            m.start_guest().unwrap();
            assert!(m.cpu_is_running());
            m.stop();
        }).unwrap().join().unwrap();
    }

    /// A CD-ROM on the second ISP1040 takes disc changes by its slot.
    #[test]
    fn second_controller_cdrom_takes_disc_changes() {
        use crate::config::{MachineProfile, ScsiDeviceConfig};
        use crate::workstation::Workstation;
        std::thread::Builder::new().stack_size(64 << 20).spawn(|| {
            let dir = scratch("octane-disc");
            let prom = empty_prom(&dir);
            let mut m = machine_with(&prom, &dir, |cfg| cfg.set_disk(20, ScsiDeviceConfig { cdrom: true, ..Default::default() }));
            let iso = dir.join("a.iso");
            std::fs::write(&iso, vec![0u8; 2048]).unwrap();
            let path = iso.to_string_lossy().into_owned();
            let changer = m.disc_changer().expect("a disc changer");
            let id = crate::topology::drive_id(MachineProfile::OctaneIp30, "scsi1/t4").unwrap();
            assert_eq!(changer.disc_status().iter().map(|d| d.0).collect::<Vec<_>>(), [id]);
            assert_eq!(changer.load_disc(id, path.clone()), Ok(path));
            assert_eq!(changer.eject_to_empty(id), Ok(()));
            assert!(changer.load_disc(4, "x.iso".into()).is_err(), "controller 0 has no drive at 4");
            m.stop();
        }).unwrap().join().unwrap();
    }
}
