//! The Tezro (IP53 board, IP35 software platform): R16000s on one Bedrock node, a PIC host
//! bridge and the IO9's IOC4, ISP12160 and BCM5701, with Iris's CPU thread, monitor and console.
//!
//! Selected with `[machine] profile = "tezro_ip53"`; `[tezro]` selects the processors, console
//! and flash. The node is the Fuel's ([`NodeBoard`]). `[graphics] board = "v12"` puts a V12 at
//! widget 12 behind the PXBOW. The one-processor board has one PIC at widget 15 (the IO9's, its
//! PXBOW answering as widget 0); the larger board adds a second at widget 11 and a V12 slot at 12.
//! Head 0 of the IO9 PIC carries the IOC4 (device 1), a TI bridge (2), the ISP12160 (3) and the
//! BCM5701 (4, REQ64# wired). The IOC4 carries the clock, NVRAM and PS/2; the L1 is the default
//! console. Bedrock board word 0x2001; the L1 reports brick type 0x39, revision 1.48.1.
//!
//! # What is approximated here
//!
//! - Board word class 2 is inferred (banners rule out 4, 8 and 0xc; OpenBSD names 2); its low
//!   nibble, 1, is a software choice only the PROM's last-resort fallback reads. The L1 type and
//!   revision, PIC revision 3 and REQ64# on the BCM5701 are the QEMU profiles' choices, not measured.
//! - The PCI devices at 1 to 4 and the IOC4 (0x53), ISP12160 (6) and BCM5701 (0x15) revisions
//!   are as consoles print them (established). INTA and INTC reaching PIC input d - 1 is IRIX's
//!   pin rule; the board wiring, INTB and INTD are unknown.
//! - The audio controller (`[machine] audio_card`) is the ICE1724 with the DM8's converters, at device 2
//!   of the TI bridge's bus (`AUDIO_DEVICE`, inferred by exclusion); its converters, subsystem
//!   IDs and EEPROM are a model choice, not Tezro evidence. Its INTA reaches PIC input 1 via the
//!   bridge's rotation (wiring unknown).
//! - The one-processor board's single PIC at widget 15 is inferred from the tower guide.
//! - Every populated processor runs from reset on its own thread; CPU enable stores and starts
//!   nothing. The PROM's reset barrier and the presence publisher are held (`PresenceHold`);
//!   neither has a hardware counterpart.
//! - The second PIC sits on the IO9 PIC's PXBOW because discovery reads only widget 0's link
//!   status; whether it has its own crossbar, its part (0xd102 or 0xd112) and reset values are
//!   unknown. QEMU gives it its own PXBOW. Its interrupts and DMA reach Bedrock IO directly.
//! - PIC DMA aimed at a widget other than Bedrock IO is refused and logged.
//! - No EEPROM is attached for the IO9, node board or BCM5701; the L1's component 0 record stands
//!   in for the node board's (`node_board`) and an attached V12 is named in component 2
//!   (`VPRO_BOARD`) with an Octane2 V12's record. A BCM5701 EEPROM transfer never completes.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::{GraphicsBoard, MachineConfig};
use crate::cpu::parts::{self, CpuModule, Part};
use crate::dev::bcm5701::{Bcm5701, Bcm5701Config};
use crate::dev::bedrock::{Bedrock, Population, SLOTS};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::ds1744::{Ds1744, YearBase};
use crate::clock::Clock;
use crate::dev::ioc4::{Ioc4, Ioc4Config};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::{BoardIdentity, CpuContext, L1Config, L1};
use crate::dev::pic::{Function, Pic, PicPeer, Refused};
use crate::dev::serial_host::{CiSerialBackend, SerialBackend};
use crate::devlog::LogModule;
use crate::hptimer::TimerManager;
use crate::ip35::presence::PresenceHold;
use crate::ip35::{attach_vpro};
use crate::ip35::node::NodeSpec;
use crate::ip35::serial::{connect_serial_ports, SerialPort};
use crate::sn::BoardIo;
use crate::pci::{DevFn, PciToPciBridge};
use crate::sn::{NodeBoard};
use crate::traits::{Device, Resettable, BUS_OK};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

/// The Bedrock board word the QEMU Tezro profiles select.
const BOARD_ID: u64 = 0x2001;
/// The IO9's PIC, the second PIC of the larger board, and the widget Bedrock IO answers as.
/// Widget 12 there is the V12's XIO slot, left empty.
const IO9_WIDGET: u8 = 15;
const SECOND_PIC_WIDGET: u8 = 11;
const HOST_WIDGET: u8 = 10;
/// IO9's PCI functions: configuration device on head 0.
const IOC4_DEVICE: u8 = 1;
const SCSI_DEVICE: u8 = 3;
const ETH_DEVICE: u8 = 4;
/// The TI PCI2050 bridge at device 2 of widget 15 (`104c:ac28`, revision 1; established from Tezro
/// captures). Its secondary bus is empty unless audio is added; only the type 1 header is modelled.
const TI_BRIDGE_DEVICE: u8 = 2;
const TI_BRIDGE_VENDOR: u16 = 0x104c;
const TI_BRIDGE_ID: u16 = 0xac28;
const TI_BRIDGE_REVISION: u8 = 1;
/// The audio controller's device on the TI bridge's secondary bus: inferred from IRIX listings
/// printing `1412:1724` at "slot 2" beside the bridge.
const AUDIO_DEVICE: u8 = 2;
/// The IO9 PIC's bus 1, empty on the board, carries the virtio functions
/// (`[virtio_serial]`, `[virtio_9p]`): a software choice, as on the Octane.
const VIRTIO_BUS: usize = 1;
const VIRTIO_SERIAL_DEVICE: u8 = 1;
const VIRTIO_9P_DEVICE: u8 = 2;

/// The Tezro's default memory in MiB: 4 GiB as 1024, 1024 and four 512 MiB banks
/// (QEMU's layout, matching a captured Tezro). Banks 4 and 5 sit at 4 to 6 GiB.
pub const DEFAULT_BANKS_MIB: [u32; 6] = [1024, 1024, 512, 512, 512, 512];

/// Check a Tezro's banks in MiB, bank 0 upward: one to eight of 512 or 1024 (Origin 350 guide
/// 007-4566-001 Table 3-3; inferred for the Tezro). Banks 2k and 2k+1 share a size code, so a lane
/// at two sizes is refused. Approximated: a last lane with one bank is accepted.
pub fn check_banks(banks: &[u32]) -> Result<(), String> {
    if let Some(mib) = banks.iter().find(|&&mib| mib != 512 && mib != 1024) {
        return Err(format!("a Tezro bank is 512 or 1024 MiB, not {mib} MiB"));
    }
    if banks.is_empty() || banks.len() > SLOTS {
        return Err(format!("memory: {} banks; a Tezro holds 1 to {SLOTS}", banks.len()));
    }
    if let Some(lane) = banks.chunks(2).position(|pair| pair.len() == 2 && pair[0] != pair[1]) {
        return Err(format!(
            "memory: banks {} and {} are {} and {} MiB; the two banks of a lane share one size",
            2 * lane, 2 * lane + 1, banks[2 * lane], banks[2 * lane + 1]));
    }
    Ok(())
}

/// One processor needs at least 512 MB of memory; two or four need at least 1 GB
/// (the Tezro manuals, as the QEMU assembly layer records them).
pub fn check_banks_for_cpus(banks: &[u32], cpus: usize) -> Result<(), String> {
    let needed = match cpus {
        1 => 512,
        2..=4 => 1024,
        _ => return Err(format!("{cpus} processors; a Tezro holds 1 to 4")),
    };
    let total: u32 = banks.iter().sum();
    if total < needed {
        return Err(format!("{total} MiB of memory; {cpus} processor{} need at least {needed} MiB", if cpus == 1 { "" } else { "s" }));
    }
    Ok(())
}

/// The processors a Tezro takes, the default first: a bare `cpu = "r16000"` selects the 700 MHz
/// part. `with_cpu!` picks the cache type (4 MB at 700 and 800 MHz, 8 MB at 900, 16 MB at 1 GHz).
pub const CPUS: &[&Part] =
    &[&parts::R16000_700_EARLY, &parts::R16000_800, &parts::R16000_900, &parts::R16000_1000];

/// Processors populated when `[machine] cpus` is unset: the four-processor tower,
/// as in the Tezro inventories seen.
pub const DEFAULT_CPUS: u8 = 4;

/// What the board knows of a chip beyond its identity: the flash record the PROM checks it against
/// (which gives the low 25 Config bits), the processor counts it is populated with, and its cache
/// clock.
///
/// Established: PRId 0x0f50 at 1 GHz; Config words 0x009bad85, 0x009baf85 and 0x009cb185 (rows of
/// the stock IP35 PROM's speed table); the 800 MHz chip as four processors (both inventories); 1
/// GHz as two or four. Inferred: the other PRIds; that the R16000 lays Config out as the PROM's
/// editor does. Approximated: Config 0x00edb385 at 1 GHz (the table stops at 900 MHz); FIR 0x0900
/// (the one R14000 value on record, never read by the PROM); the 900 MHz chip as four processors
/// (no board at 900 MHz is captured). Unknown: Config bit 23 and the HSTL bits (bit 23 is copied).
/// The PROM has no processor count field, only IP53_nCPU names with no clock; the 700 MHz chip's
/// one, two or four follows the board's population. The cache clock (350, 400, 450 and 333 MHz) is
/// as inventories print it and not modelled. An 800 MHz part with an 8 MB cache (PROM row 21) is on
/// Onyx 350 inventories only and is not modelled.
pub(crate) struct TezroChip {
    /// The processor module: the part and the mode word its Config resets with.
    pub module: &'static CpuModule,
    pub firmware: FirmwareRecord,
    pub cpus: &'static [u8],
    /// The node board the L1 names for each processor count in `cpus`; see
    /// [`node_board`].
    pub boards: &'static [(u8, BoardIdentity)],
    /// Why `cpus` is what it is when that is not established: a clause for
    /// the refusal message.
    pub cpus_basis: &'static str,
}

pub(crate) const CHIPS: [TezroChip; 4] = [
    TezroChip {
        module: &parts::R16000_700_EARLY_SYSCLK_200,
        firmware: FirmwareRecord::TEZRO,
        cpus: &[1, 2, 4],
        boards: &[
            (1, board("IP53_1CPU", "", "")),
            (2, board("IP53_2CPU", "030-1869-001", "-C")),
            (4, board("IP53_4CPU", "030-1868-001", "-C")),
        ],
        cpus_basis: ", from the board's population and the PROM's assembly names, which pair with no clock (inferred)",
    },
    TezroChip {
        module: &parts::R16000_800_SYSCLK_200,
        firmware: FirmwareRecord::TEZRO_800,
        cpus: &[4],
        boards: &[(4, board("IP53_4CPU", "030-1956-002", "-C"))],
        cpus_basis: ", as on every captured board with it; the two-processor 800 MHz board carries revision 3.0 processors",
    },
    TezroChip {
        module: &parts::R16000_900_SYSCLK_200,
        firmware: FirmwareRecord::TEZRO_900,
        cpus: &[4],
        boards: &[(4, board("IP53_4CPU", "", ""))],
        cpus_basis: ", a model choice: no IP53 or IP59 board at 900 MHz is captured (approximated)",
    },
    TezroChip {
        module: &parts::R16000_1000_SYSCLK_200,
        firmware: FirmwareRecord::TEZRO_1000,
        cpus: &[2, 4],
        boards: &[(2, board("IP59_2CPU", "030-2059-002", "-A")), (4, board("IP59_4CPU", "030-1989-003", "-C"))],
        cpus_basis: ", as the captured IP59_4CPU and IP59_2CPU boards are",
    },
];

/// The board's data for `chip`. Panics for a part the Tezro does not take;
/// `MachineProfile::part` has refused one before a board is built.
pub(crate) fn chip(chip: &Part) -> &'static TezroChip {
    CHIPS.iter().find(|c| *c.module.part == *chip).expect("a Tezro takes only its own R16000 parts")
}

impl TezroChip {
    /// Whether the chip is populated as `cpus` processors, and the refusal if
    /// not.
    pub fn accepts_cpus(&self, cpus: u8) -> Result<(), String> {
        if self.cpus.contains(&cpus) {
            return Ok(());
        }
        let list: Vec<String> = self.cpus.iter().map(u8::to_string).collect();
        Err(format!(
            "cpus = {cpus}: the {} is populated with {} processors only{}",
            self.module.part.label(),
            list.join(", "),
            self.cpus_basis
        ))
    }
}

/// The CPU configuration the node board's L1 record carries: the flash record's mode word and
/// clock, and the 200 MHz Bedrock. Without it the PROM prints "CPU configuration values from board
/// EEPROM/PSC did not match". Approximated: a real board's record has not been read; this is what
/// QEMU's Tezro gives.
pub(crate) fn cpu_context(firmware: &FirmwareRecord) -> CpuContext {
    CpuContext { mode_word: firmware.cpu_mode_word, cpu_mhz: firmware.cpu_clock_hz / 1_000_000, hub_mhz: 200 }
}

/// The node board the L1's component 0 record names for `chip` populated as `cpus` processors: the
/// inventory's product name, part and revision. The 700 MHz four-processor board (`IP53_4CPU`
/// 030-1868-001 -C) and the 800 MHz one (030-1956-002 -C) are established; the 700 MHz two- and
/// one-processor names, and the 1 GHz boards (IP59, captured on Onyx 350 modules; -C is the
/// model's), are inferred; the 900 MHz board is approximated. The PROM tries only the sockets in
/// the mask of the matching product-name row (`IP53_4CPU` and `IP59_4CPU` 0xf, `IP53_2CPU` and
/// `IP59_2CPU` 0x5, `IP53_1CPU` 0x1) and power-cycles the module for each empty one; it reads no
/// part or revision.
fn node_board(chip: &TezroChip, cpus: usize) -> BoardIdentity {
    chip.boards
        .iter()
        .find(|(count, _)| usize::from(*count) == cpus)
        .map(|(_, board)| *board)
        .expect("a Tezro chip names a board for each processor count it takes")
}

const fn board(name: &'static str, part: &'static str, revision: &'static str) -> BoardIdentity {
    BoardIdentity { name, part, revision }
}

/// The V12's board record in component 2 of the node's L1. IRIX reads it at attach; without it the
/// boot logs two ALERT lines. Name, part and revision are an Octane2 `hinv -mv`'s (a Tezro prints
/// `ODY128B1_2`, which makes the kernel set up a digital output this model lacks); the serial is
/// synthetic.
const VPRO_BOARD: BoardIdentity = board("ODY128VERSIONB", "030-1611-001", "C");

/// The serial connections: the L1, then IOC4 ports 0 to 3.
const SERIAL_PORTS: usize = 5;

/// The Tezro's node of `cpus` processors of `chip`, with `banks_mib`, the L1's identity and the
/// boot flash in `flash` or volatile; an error for a population the board cannot hold.
fn spec(
    chip: &'static TezroChip,
    cpus: usize,
    banks_mib: &[u32],
    mac: [u8; 6],
    system_serial: Option<String>,
    vpro: bool,
    flash: Option<PathBuf>,
) -> Result<NodeSpec, String> {
    if ![1, 2, 4].contains(&cpus) {
        return Err(format!("{cpus} processors; a Tezro board holds 1, 2 or 4"));
    }
    chip.accepts_cpus(cpus as u8)?;
    check_banks_for_cpus(banks_mib, cpus)?;
    Ok(NodeSpec {
        name: "Tezro",
        module: chip.module,
        firmware: chip.firmware,
        board_id: BOARD_ID,
        banks_mib: banks_mib.to_vec(),
        processors: Population::populated_cpus(cpus)?,
        absent: "",
        l1: L1Config {
            type_code: 0x39,
            revision: Some([1, 48, 1]),
            mac,
            system_serial,
            port_speed_first: 1,
            router: true,
            cpu_context: Some(cpu_context(&chip.firmware)),
            node_board: Some(node_board(chip, cpus)),
            graphics_board: vpro.then_some(VPRO_BOARD),
            ..Default::default()
        },
        flash,
    })
}

/// The Tezro's node: its processors, memory, L1 identity and V12 as the configuration says.
pub(crate) fn node_specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
    let banks = cfg.tezro_memory()?;
    let cpus = usize::from(cfg.machine.cpus());
    let chip = chip(cfg.cpu());
    let vpro = cfg.graphics_board() == GraphicsBoard::V12;
    Ok(vec![spec(chip, cpus, &banks, cfg.network().mac, Some(cfg.tezro_system_serial()), vpro, cfg.tezro.flash.as_deref().map(PathBuf::from))?])
}

/// The IO9's devices around the node: the PXBOW and PICs, IOC4, ISP12160, BCM5701, the audio
/// controller when asked for, and the V12. The clocked devices run on `timers`.
pub(crate) fn build_io(cfg: &MachineConfig, nodes: &[NodeBoard<Bedrock, Flash>], l1s: &[Arc<L1>], timers: &Arc<TimerManager>, clock: &Clock) -> Result<BoardIo, String> {
    let (bedrock, l1) = (&nodes[0].chip, &l1s[0]);
    let cpus = usize::from(cfg.machine.cpus());
    let chip = chip(cfg.cpu());
    let running = Population::populated_cpus(cpus)?;
    let timekeeper = Arc::new(Ds1744::new(YearBase::From(1968), timers.clock().clone())?);
    timekeeper.set_nvram_file(Path::new(&cfg.nvram))?;
    let serial: [Arc<CiSerialBackend>; SERIAL_PORTS] = std::array::from_fn(|_| Arc::new(CiSerialBackend::new()));
    l1.set_serial_backend(serial[0].clone());

    let io9 = Pic::new(IO9_WIDGET);
    let pxbow = io9.enable_pxbow(HOST_WIDGET).map_err(|e| format!("Tezro PXBOW: {e:?}"))?;
    bedrock.io.connect_widget(0, pxbow.widget_zero());
    bedrock.io.connect_widget(IO9_WIDGET, pxbow.route(IO9_WIDGET));
    let presence = PresenceHold::new(running, clock.clone(), bedrock);
    let heartbeat = Arc::new(AtomicU64::new(0));
    let fasttick = Arc::new(AtomicU64::new(0));
    let status = crate::status::MachineStatus::new(heartbeat.clone(), fasttick.clone());
    status.set_scsi_wide(true);
    // The V12's interrupt packets and DMA reach Bedrock IO as the PIC's do. Nothing counts the
    // processors' stores into its graphics page or returns credits for them.
    let vpro = (cfg.graphics_board() == GraphicsBoard::V12)
        .then(|| {
            let widget = cfg.graphics.widget.unwrap_or(VPRO_WIDGET);
            check_vpro_widget(widget)?;
            attach_vpro(cfg, bedrock, widget, bedrock.interrupt_sender(widget, Some(INBOUND_DEVICE)), &status, |vpro| {
                pxbow.attach_widget(widget, vpro).map_err(|e| format!("Tezro V12: {e:?}"))?;
                Ok(pxbow.route(widget))
            })
        })
        .transpose()?;
    let pics = if cpus == 1 {
        vec![io9.clone()]
    } else {
        // The PROM finds widget 11 only as a live link of widget 0's
        // crossbar, so the second PIC is a port of the same switch.
        let second = Pic::new(SECOND_PIC_WIDGET);
        pxbow
            .attach_widget(SECOND_PIC_WIDGET, second.widget())
            .map_err(|e| format!("Tezro: second PIC on the PXBOW: {e:?}"))?;
        bedrock.io.connect_widget(SECOND_PIC_WIDGET, pxbow.route(SECOND_PIC_WIDGET));
        vec![second, io9.clone()]
    };
    for pic in &pics {
        pic.connect_peer(Arc::new(BedrockPeer(Arc::downgrade(bedrock))))
            .map_err(|_| "Tezro: PIC peer connected twice".to_string())?;
    }

    let ioc4 = Arc::new(Ioc4::new(timers, Ioc4Config::tezro()));
    ioc4.attach_clock(timekeeper.clone());
    for (port, backend) in serial[1..].iter().enumerate() {
        ioc4.set_serial_backend(port, backend.clone());
    }
    let scsi = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus(), timers));
    scsi.set_heartbeat(heartbeat.clone());
    // The Ethernet slot drives REQ64#, which the chip samples as reset ends.
    let eth = Bcm5701::new(timers, Bcm5701Config { req64: true, ..Bcm5701Config::new(l1.station_mac()) })?;
    io9.attach(0, IOC4_DEVICE, ioc4.clone(), Function { self_ordered_pio: true, req64: false });
    let ti = PciToPciBridge::with_wide_windows(
        io9.bus(0),
        DevFn { device: TI_BRIDGE_DEVICE, function: 0 },
        TI_BRIDGE_VENDOR,
        TI_BRIDGE_ID,
        TI_BRIDGE_REVISION,
    );
    io9.attach(0, TI_BRIDGE_DEVICE, ti.clone(), Function::default());
    let audio = cfg.tezro.audio.then(|| {
        let card = crate::ip35::audio_card(cfg, timers);
        ti.secondary().attach(DevFn { device: AUDIO_DEVICE, function: 0 }, card.clone());
        card
    });
    io9.attach(0, SCSI_DEVICE, scsi.clone(), Function::default());
    io9.attach(0, ETH_DEVICE, eth.clone(), Function { self_ordered_pio: false, req64: true });

    let (table, product) = (cfg.machine.profile.table_name().unwrap_or_default(), cfg.machine.profile.product_name().unwrap_or_default());
    let scratch = crate::scsi::attach_without_daynaport(cfg, table, product, &|channel| {
        scsi.bus_for_channel(channel).ok_or_else(|| format!("controller {channel} (a {product} has channels 0 and 1)"))
    })?;
    let virtio = Arc::new(attach_virtio(cfg, &io9));
    eprintln!("iris: {}", if vpro.is_some() { cfg.graphics_board().label() } else { "no graphics" });
    let console_index = cfg.tezro.console.index();
    connect_serial_ports(cfg, "Tezro", &serial_ports(cfg, l1, &ioc4, &serial), console_index)?;
    let console = cfg.ci.then(|| serial[console_index].clone());
    let mut monitor: Vec<Arc<dyn Device>> = vec![scsi.clone(), ioc4.ps2()];
    for pic in &pics {
        monitor.push(crate::pci::pci_host_commands(&format!("PIC at widget {}", pic.widget_number()), &[pic.bus(0), pic.bus(1)], None));
    }
    if let Some(vpro) = &vpro {
        monitor.push(vpro.clone());
    }
    if let Some(audio) = &audio {
        monitor.push(audio.clone());
    }
    let mut devices: Vec<Arc<dyn Device>> = Vec::new();
    devices.extend(vpro.iter().map(|v| v.clone() as Arc<dyn Device>));
    devices.extend([ioc4.clone() as Arc<dyn Device>, eth.clone(), scsi.clone()]);
    devices.extend(audio.iter().map(|a| a.clone() as Arc<dyn Device>));
    devices.extend(virtio.iter().map(|v| v.clone() as Arc<dyn Device>));
    let (reset_virtio, reset_presence, reset_ioc4, reset_scsi, reset_eth, reset_pics, reset_vpro) =
        (virtio, presence.clone(), ioc4.clone(), scsi.clone(), eth.clone(), pics.clone(), vpro.clone());
    let held = presence.clone();
    let cpu_presence = presence;
    scsi.register_locks();
    ioc4.register_locks();
    eth.register_locks();
    for pic in &pics {
        pic.register_locks();
    }
    if let Some(audio) = &audio {
        audio.register_locks();
    }
    let (cpu_eth, disks) = (eth.clone(), scsi.clone());
    Ok(BoardIo {
        devices,
        reset: Box::new(move || {
            reset_presence.reset();
            reset_ioc4.power_on();
            reset_scsi.power_on();
            reset_eth.power_on();
            for pic in &reset_pics {
                pic.power_on();
            }
            if let Some(vpro) = &reset_vpro {
                vpro.power_on();
            }
            for v in reset_virtio.iter() {
                Resettable::power_on(&**v);
            }
        }),
        at_batch: (running.count() > 1).then(|| Arc::new(move || held.at_batch()) as Arc<dyn Fn() + Send + Sync>),
        monitor,
        ps2: ioc4.ps2(),
        flush_nvram: Box::new(move || timekeeper.flush()),
        console,
        attach_net: Box::new(move |link| eth.attach(link)),
        disks: Box::new(move |f| f(&(0..2).filter_map(|channel| disks.bus_for_channel(channel)).collect::<Vec<_>>())),
        disc_changer: Some(scsi),
        scratch,
        display: vpro.as_ref().map(|vpro| vpro.display()),
        status: Some(status),
        fasttick: Some(fasttick),
        // The Ethernet holds a reply until the processor that sent the
        // request has run on past its doorbell.
        cpus: Some(Box::new(move |all| {
            cpu_presence.set_cpus(all);
            cpu_eth.set_cpu_cycles(all.iter().map(|c| c.cycles_ptr()).collect());
        })),
    })
}

/// The board's serial connections in console order: the L1, then the IOC4's
/// four ports.
fn serial_ports(cfg: &MachineConfig, l1: &Arc<L1>, ioc4: &Arc<Ioc4>, serial: &[Arc<CiSerialBackend>; SERIAL_PORTS]) -> Vec<SerialPort> {
    let [a, b] = cfg.serial_ports();
    let ioc4 = |port: usize| -> Box<dyn Fn(Arc<dyn SerialBackend>) + Send + Sync> {
        let ioc4 = ioc4.clone();
        Box::new(move |backend| ioc4.set_serial_backend(port, backend))
    };
    let l1 = l1.clone();
    vec![
        SerialPort {
            label: "L1",
            port: cfg.tezro.l1_port.unwrap_or(8882),
            setting: "machine.l1_port",
            ci: serial[0].clone(),
            log: cfg.tezro.l1_log.clone(),
            socket: None,
            connect: Box::new(move |backend| l1.set_serial_backend(backend)),
        },
        SerialPort { label: "IOC4 0", port: a, setting: "serial_port_a", ci: serial[1].clone(), log: cfg.serial_log_a.clone(), socket: cfg.serial_socket_a.clone().map(|p| (p, "serial_socket_a")), connect: ioc4(0) },
        SerialPort { label: "IOC4 1", port: b, setting: "serial_port_b", ci: serial[2].clone(), log: cfg.serial_log_b.clone(), socket: cfg.serial_socket_b.clone().map(|p| (p, "serial_socket_b")), connect: ioc4(1) },
        SerialPort {
            label: "IOC4 2",
            port: cfg.tezro.serial_port_c.unwrap_or(8883),
            setting: "machine.serial_port_c",
            ci: serial[3].clone(),
            log: None,
            socket: None,
            connect: ioc4(2),
        },
        SerialPort {
            label: "IOC4 3",
            port: cfg.tezro.serial_port_d.unwrap_or(8884),
            setting: "machine.serial_port_d",
            ci: serial[4].clone(),
            log: None,
            socket: None,
            connect: ioc4(3),
        },
    ]
}

/// The virtio functions the config enables, on the IO9 PIC's bus 1, each
/// with its INTA on input `device - 1`, as `Pic::attach` routes it.
fn attach_virtio(cfg: &MachineConfig, io9: &Arc<Pic>) -> Vec<Arc<crate::dev::virtio::pci::VirtioPci>> {
    use crate::dev::virtio::pci::VirtioPci;
    let mut attached = Vec::new();
    let mut attach = |device: u8, name: &str, dev: VirtioPci| {
        let dev = Arc::new(dev);
        io9.attach(VIRTIO_BUS, device, dev.clone(), Function::default());
        eprintln!("iris: {name} on the IO9 PIC, bus {VIRTIO_BUS} device {device}");
        attached.push(dev);
    };
    if cfg.virtio_serial.enabled {
        attach(VIRTIO_SERIAL_DEVICE, "virtio-serial",
            VirtioPci::new(crate::dev::virtio::console::from_config(&cfg.virtio_serial)));
    }
    if cfg.virtio_9p.enabled {
        attach(VIRTIO_9P_DEVICE, "virtio-9p", VirtioPci::new(crate::dev::virtio::p9::from_config(&cfg.virtio_9p)));
    }
    attached
}


/// Where the V12 sits: widget 12 behind the PIC that owns widget 0 (QEMU's graphics slot). On the
/// one-processor board the guide names no widget or route for it; this is inferred.
const VPRO_WIDGET: u8 = 12;

/// `[graphics] widget`: a port of the crossbar the V12 can take, which the host port, the second
/// PIC and the IO9 are not. Only 12 is known to work; the others are for experiment.
pub(crate) fn check_vpro_widget(widget: u8) -> Result<(), String> {
    if (8..=14).contains(&widget) && widget != HOST_WIDGET && widget != SECOND_PIC_WIDGET {
        Ok(())
    } else {
        Err(format!("graphics.widget {widget} is not a free port of the Tezro's crossbar (valid: 8, 9, 12, 13, 14)"))
    }
}

/// The far end of a PIC's link: Bedrock IO, where interrupt packets and DMA arrive as inbound
/// packets from the PIC's widget. DMA aimed at another widget is refused (no peer-to-peer XIO).
struct BedrockPeer(Weak<Bedrock>);

const INBOUND_DEVICE: u8 = 0;

impl BedrockPeer {
    fn io(&self) -> Result<Arc<Bedrock>, Refused> {
        self.0.upgrade().ok_or(Refused)
    }

    fn host(widget: u8) -> Result<(), Refused> {
        if widget == HOST_WIDGET {
            Ok(())
        } else {
            dlog!(LogModule::Bedrock, "Tezro: PIC DMA to widget {widget} refused; only Bedrock IO is reachable");
            Err(Refused)
        }
    }

    /// The legs of a transfer as (offset, bytes): whole doublewords from an 8-aligned address are
    /// one leg; the ragged bytes around them are each the largest naturally aligned size of four or
    /// fewer.
    fn legs(address: u64, len: usize) -> impl Iterator<Item = (usize, usize)> {
        let mut offset = 0;
        std::iter::from_fn(move || {
            if offset == len {
                return None;
            }
            let at = address + offset as u64;
            let left = len - offset;
            let size = if at.is_multiple_of(8) && left >= 8 {
                left & !7
            } else {
                [4usize, 2, 1]
                    .into_iter()
                    .find(|&n| at.is_multiple_of(n as u64) && n <= left)
                    .unwrap_or(1)
            };
            let leg = (offset, size);
            offset += size;
            Some(leg)
        })
    }
}

/// Doublewords moved per call to memory. The stack buffer is zeroed per transfer, so it is small:
/// a whole page cost a `rep stos` that outweighed the saving.
const BLOCK_WORDS: usize = 64;

impl PicPeer for BedrockPeer {
    fn interrupt(&self, source_widget: u8, address: u64, payload: u64) -> Result<(), Refused> {
        let io = self.io()?;
        (io.inbound_write(source_widget, Some(INBOUND_DEVICE), address, 8, payload) == BUS_OK)
            .then_some(())
            .ok_or(Refused)
    }

    fn dma_read(&self, source_widget: u8, widget: u8, address: u64, data: &mut [u8]) -> Result<(), Refused> {
        Self::host(widget)?;
        let io = self.io()?;
        for (offset, size) in Self::legs(address, data.len()) {
            let at = address + offset as u64;
            if size < 8 {
                let value = io.inbound_read(source_widget, Some(INBOUND_DEVICE), at, size as u8);
                if value.status != BUS_OK {
                    return Err(Refused);
                }
                data[offset..offset + size].copy_from_slice(&value.data.to_be_bytes()[8 - size..]);
                continue;
            }
            let mut words = [0u64; BLOCK_WORDS];
            for (n, part) in data[offset..offset + size].chunks_mut(8 * BLOCK_WORDS).enumerate() {
                let words = &mut words[..part.len() / 8];
                if io.inbound_read_block(source_widget, Some(INBOUND_DEVICE), at + (n * 8 * BLOCK_WORDS) as u64, words) != BUS_OK {
                    return Err(Refused);
                }
                for (bytes, word) in part.chunks_exact_mut(8).zip(words.iter()) {
                    bytes.copy_from_slice(&word.to_be_bytes());
                }
            }
        }
        Ok(())
    }

    fn dma_write(&self, source_widget: u8, widget: u8, address: u64, data: &[u8]) -> Result<(), Refused> {
        Self::host(widget)?;
        let io = self.io()?;
        for (offset, size) in Self::legs(address, data.len()) {
            let at = address + offset as u64;
            if size < 8 {
                let value = data[offset..offset + size].iter().fold(0u64, |v, b| (v << 8) | u64::from(*b));
                if io.inbound_write(source_widget, Some(INBOUND_DEVICE), at, size as u8, value) != BUS_OK {
                    return Err(Refused);
                }
                continue;
            }
            let mut words = [0u64; BLOCK_WORDS];
            for (n, part) in data[offset..offset + size].chunks(8 * BLOCK_WORDS).enumerate() {
                let words = &mut words[..part.len() / 8];
                for (word, bytes) in words.iter_mut().zip(part.chunks_exact(8)) {
                    *word = u64::from_be_bytes(bytes.try_into().unwrap());
                }
                if io.inbound_write_block(source_widget, Some(INBOUND_DEVICE), at + (n * 8 * BLOCK_WORDS) as u64, words) != BUS_OK {
                    return Err(Refused);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
