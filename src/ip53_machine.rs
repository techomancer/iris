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
//! - The audio controller (an `ice1724` entry) is the ICE1724 with the DM8's converters, at device 2
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
use crate::dev::battery_clock::YearBase;
use crate::dev::ds1744::Ds1744;
use crate::clock::Clock;
use crate::dev::ioc4::{Ioc4, Ioc4Config};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::{BoardIdentity, CpuContext, L1Config, DEFAULT_REVISION, L1};
use crate::dev::pic::{Function, Pic, PIC_ROUTING};
use crate::dev::serial_dev::connect_l1_board;
use crate::devlog::LogModule;
use crate::hptimer::TimerManager;
use crate::ip35::presence::PresenceHold;
use crate::ip35::{attach_vpro, BEDROCK_PORT};
use crate::xtalk::Upstream;
use crate::ip35::node::NodeSpec;
use crate::sn::BoardIo;
use crate::pci::{DevFn, PciToPciBridge, PCI_INTA};
use crate::topology::MachineSlotKind;
use crate::sn::{NodeBoard};
use crate::traits::{Device, Resettable};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The Bedrock board word the QEMU Tezro profiles select.
const BOARD_ID: u64 = 0x2001;
/// The IO9's PIC, the second PIC of the larger board, and the widget Bedrock IO answers as.
/// Widget 12 there is the V12's XIO slot, left empty.
const IO9_WIDGET: u8 = 15;
const SECOND_PIC_WIDGET: u8 = 11;
const HOST_WIDGET: u8 = 10;
/// The device the V12's packets name at Bedrock IO's inbound permission check.
const INBOUND_DEVICE: u8 = 0;

/// The L1's brick type for the Tezro.
const L1_TYPE: u32 = 0x39;

/// The IO9 PIC's bus 0: the IOC4, the TI bridge, the ISP12160 and the BCM5701.
pub(crate) const IO9_BUS0_SLOTS: &[(u8, MachineSlotKind, Option<&str>)] = &[
        (IOC4_DEVICE, MachineSlotKind::Fixed("ioc4"), None),
        (TI_BRIDGE_DEVICE, MachineSlotKind::Fixed("pci2050"), None),
        (SCSI_DEVICE, MachineSlotKind::Fixed("isp12160"), None),
        (ETH_DEVICE, MachineSlotKind::Fixed("bcm5701"), None),
    ];

/// Bus 1, empty on the board: two extra slots.
pub(crate) const IO9_BUS1_SLOTS: &[(u8, MachineSlotKind, Option<&str>)] = &[(1, MachineSlotKind::Extra { pin: PCI_INTA }, None), (2, MachineSlotKind::Extra { pin: PCI_INTA }, None)];

/// The Tezro's slots.
pub(crate) fn slots(s: &mut Vec<crate::topology::MachineSlot>) {
    // The V12's port first, the only one known to work; the others are for experiment.
    for w in [VPRO_WIDGET, 8, 9, 13, 14] {
        crate::topology::open_slot(s, &format!("xio{w}"), &[crate::topology::DeviceFamily::Gfx], &[]);
    }
    crate::topology::pci_bus_slots(s, &format!("xio{IO9_WIDGET}/pci0"), "Tezro IO9 PIC, bus 0", IO9_BUS0_SLOTS);
    // The ISP12160's two channels.
    for c in [0, 1] {
        crate::topology::scsi_bus_slots(s, &format!("xio{IO9_WIDGET}/pci0/{SCSI_DEVICE}/c{c}"), "isp12160 channel", 15);
    }
    crate::topology::pci_bus_slots(s, &format!("xio{IO9_WIDGET}/pci1"), "Tezro IO9 PIC, bus 1", IO9_BUS1_SLOTS);
    let behind_ti = format!("xio{IO9_WIDGET}/pci0/{TI_BRIDGE_DEVICE}/pci0");
    crate::topology::pci_bus_slots(s, &behind_ti, "TI PCI2050 secondary bus", &[(AUDIO_DEVICE, MachineSlotKind::PCI_SLOT, None)]);
}

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
/// The slot on the TI bridge's secondary bus, where an audio controller goes: inferred from IRIX
/// listings printing `1412:1724` at "slot 2" beside the bridge.
const AUDIO_DEVICE: u8 = 2;

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

/// The least memory in MiB: one processor, and two to four.
const MIN_MIB_ONE_CPU: u32 = 512;
const MIN_MIB_MULTI_CPU: u32 = 1024;

/// One processor needs at least 512 MB of memory; two or four need at least 1 GB
/// (the Tezro manuals, as the QEMU assembly layer records them).
pub fn check_banks_for_cpus(banks: &[u32], cpus: usize) -> Result<(), String> {
    let needed = match cpus {
        1 => MIN_MIB_ONE_CPU,
        2..=4 => MIN_MIB_MULTI_CPU,
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
            type_code: L1_TYPE,
            revision: Some(DEFAULT_REVISION),
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

/// The IO9's devices around the node: the PXBOW and PICs, IOC4, ISP12160, BCM5701, the cards the
/// config places, and the V12. The clocked devices run on `timers`.
pub(crate) fn build_io(cfg: &MachineConfig, nodes: &[NodeBoard<Bedrock, Flash>], l1s: &[Arc<L1>], timers: &Arc<TimerManager>, clock: &Clock) -> Result<BoardIo, String> {
    let (bedrock, l1) = (&nodes[0].chip, &l1s[0]);
    let cpus = usize::from(cfg.machine.cpus());
    let chip = chip(cfg.cpu());
    let running = Population::populated_cpus(cpus)?;
    let timekeeper = Arc::new(Ds1744::new(YearBase::From(1968), timers.clock().clone())?);
    timekeeper.set_nvram_file(Path::new(&cfg.nvram))?;

    let to_bedrock = |widget| {
        let port = bedrock.inbound(widget);
        Upstream::single(BEDROCK_PORT, port.clone(), port)
    };
    let io9 = Pic::new(IO9_WIDGET, HOST_WIDGET, to_bedrock(IO9_WIDGET));
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
    let mut resolved = crate::topology::DevicePlacement::new(cfg)?;
    let vpro = (cfg.graphics_board() == GraphicsBoard::V12)
        .then(|| {
            let widget = resolved.graphics().and_then(|at| at.xio_port()).map_or(VPRO_WIDGET, |p| p.1);
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
        let second = Pic::new(SECOND_PIC_WIDGET, HOST_WIDGET, to_bedrock(SECOND_PIC_WIDGET));
        pxbow
            .attach_widget(SECOND_PIC_WIDGET, second.widget())
            .map_err(|e| format!("Tezro: second PIC on the PXBOW: {e:?}"))?;
        bedrock.io.connect_widget(SECOND_PIC_WIDGET, pxbow.route(SECOND_PIC_WIDGET));
        vec![second, io9.clone()]
    };
    let ioc4 = Arc::new(Ioc4::new(timers, Ioc4Config::tezro()));
    ioc4.attach_clock(timekeeper.clone());
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
    io9.attach(0, SCSI_DEVICE, scsi.clone(), Function::default());
    io9.attach(0, ETH_DEVICE, eth.clone(), Function { self_ordered_pio: false, req64: true });

    let (table, product) = (cfg.machine.profile.table_name().unwrap_or_default(), cfg.machine.profile.product_name().unwrap_or_default());
    let buses = [0, 1].map(|c| scsi.bus_for_channel(c).expect("the ISP12160 has two channels"));
    for head in [0, 1] {
        resolved.bind_pci(&format!("xio{IO9_WIDGET}/pci{head}"), io9.bus(head).clone())?;
    }
    resolved.bind_pci(&format!("xio{IO9_WIDGET}/pci0/{TI_BRIDGE_DEVICE}/pci0"), ti.secondary().clone())?;
    let scratch = crate::scsi::attach_without_daynaport(cfg, table, product, &resolved, &buses)?;
    let cards = Arc::new(resolved.build_cards(cfg, timers)?);
    eprintln!("iris: {}", if vpro.is_some() { cfg.graphics_board().label() } else { "no graphics" });
    let console = connect_l1_board(
        cfg,
        "Tezro",
        ["IOC4 0", "IOC4 1"],
        (cfg.tezro.l1_port, cfg.tezro.l1_log.as_deref()),
        [&|b| l1.set_serial_backend(b), &|b| ioc4.set_serial_backend(0, b), &|b| ioc4.set_serial_backend(1, b)],
    )?;
    let mut monitor: Vec<Arc<dyn Device>> = vec![scsi.clone(), ioc4.ps2()];
    for pic in &pics {
        monitor.push(crate::pci::pci_host_commands(&format!("PIC at widget {}", pic.widget_number()), &[pic.bus(0), pic.bus(1)], None));
    }
    if let Some(vpro) = &vpro {
        monitor.push(vpro.clone());
    }
    let mut devices: Vec<Arc<dyn Device>> = Vec::new();
    devices.extend(vpro.iter().map(|v| v.clone() as Arc<dyn Device>));
    devices.extend([ioc4.clone() as Arc<dyn Device>, eth.clone(), scsi.clone()]);
    let (reset_presence, reset_ioc4, reset_scsi, reset_eth, reset_pics, reset_vpro) =
        (presence.clone(), ioc4.clone(), scsi.clone(), eth.clone(), pics.clone(), vpro.clone());
    let held = presence.clone();
    let cpu_presence = presence;
    scsi.register_locks();
    ioc4.register_locks();
    eth.register_locks();
    for pic in &pics {
        pic.register_locks();
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
    }
    .with_cards(&cards))
}

/// Where the V12 sits: widget 12 behind the PIC that owns widget 0 (QEMU's graphics slot). On the
/// one-processor board the guide names no widget or route for it; this is inferred.
const VPRO_WIDGET: u8 = 12;

#[cfg(test)]
mod tests;
