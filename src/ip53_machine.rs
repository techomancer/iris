//! The Tezro (IP53 board, IP35 software platform): R16000s on one Bedrock
//! node, a PIC host bridge and the IO9 board's IOC4, ISP12160 and BCM5701,
//! with Iris's CPU thread, monitor and serial console.
//!
//! Selected with `[machine] profile = "tezro_ip53"`. Shared keys select the PROM,
//! NVRAM, memory and SCSI devices; `[tezro]` selects the processor
//! population, the serial console and writable flash. The node itself is
//! the one the Fuel uses ([`NodeBoard`]); this board adds its widgets and its
//! inputs. With `[graphics] board = "v12"` a VPro V12 sits at widget 12
//! behind the PXBOW; otherwise there is no graphics.
//!
//! The board follows the QEMU Tezro compositions:
//!
//! - The one-processor board has one PIC at XIO widget 15; the two- and
//!   four-processor board adds a second at widget 11, with nothing on its
//!   buses, and a V12 slot at widget 12 that is left empty. The PIC at
//!   widget 15 is the IO9's: its PXBOW answers as widget 0 and forwards
//!   the processor's accesses to widget 15. On the larger board it carries
//!   widget 11 too, and shows that link present and alive (unlike QEMU's
//!   composition, below). Its head 0 carries the IO9 PCI bus: the IOC4 at
//!   device 1 (input 0), a TI PCI-to-PCI bridge with nothing behind it at
//!   device 2, the ISP12160 at device 3 (input 2) and the BCM5701 at device
//!   4 (input 3), with REQ64# wired on the BCM5701's slot. The PIC's interrupts and DMA go to Bedrock IO, the host widget
//!   10.
//! - The IOC4 carries the DS1744-compatible clock and NVRAM in its BAR and
//!   the PS/2 keyboard and mouse; its four ports are serial streams beside
//!   the L1's, and the L1 is the console by default.
//! - Bedrock board word 0x2001, the Tezro firmware record (the chip's clock,
//!   an authored configuration block), and an L1 that reports
//!   brick type 0x39, revision 1.48.1, the first port-speed integer 1, the
//!   router `brick` reply, the system serial `[machine] emu_id` gives
//!   (IRIX's host id is its fold) and, in the node board's record, the CPU
//!   configuration the flash block holds. The board serial in that record
//!   stays `EMUN00` whatever the emu_id: the PROM ties its NVRAM to it.
//!
//! # What is approximated here
//!
//! - The board word's class, 2, is inferred: Tezro PROM banners rule out
//!   classes 4, 8 and 0xc, OpenBSD names class 2 for this board family, and
//!   no raw read is on record. Its low nibble, 1, is a software choice: only
//!   the PROM's last-resort table fallback reads it, which a board whose L1
//!   record carries the CPU configuration never reaches. The L1 type and
//!   revision, the PIC revision 3 profile and REQ64# on the BCM5701's slot
//!   are software inputs the QEMU profiles choose, not values measured on a
//!   Tezro.
//! - The PCI devices at 1 to 4 and the revisions of the IOC4 (0x53), the
//!   ISP12160 (6) and the BCM5701 (0x15) are as Tezro and Origin 350
//!   consoles print them (established for those captures). Device d's INTA
//!   and INTC reaching PIC input d - 1 is what IRIX's pin rule assigns, and
//!   INTA is what an Origin 350 under OpenBSD reports (established as
//!   software expectation); the board wiring itself, and INTB and INTD, are
//!   unknown.
//! - The audio controller, off unless `[tezro] audio` adds it, is the
//!   ICE1724 with the DM8's converters and configuration EEPROM, at device 2
//!   of the TI bridge's secondary bus (`AUDIO_DEVICE`). Where it sits is
//!   inferred by exclusion from the IRIX hardware listings of Tezro towers,
//!   which print `1412:1724` at device 2 beside the bridge; what the
//!   Tezro's board carries for converters, subsystem IDs and EEPROM is unknown,
//!   so it takes the one board design with a documented recipe, a stated
//!   model choice and not Tezro evidence. Its INTA is
//!   rotated by the bridge to INTC of the bridge's slot and reaches PIC
//!   input 1; whether the board wires it so is unknown, and the controller
//!   masks every source at reset.
//! - The one-processor board has one PIC. The tower guide gives that board
//!   two PCI-X buses and one XIO connector; that it is one PIC at widget 15
//!   is inferred from them.
//! - Every fitted processor runs from reset, each on its own thread, as in
//!   QEMU's IP35 node; CPU enable is stored and starts or stops nothing. The
//!   PROM's reset barrier is ordered by Bedrock's LB; the first of a pair
//!   to post its liveness is held until its partner posts, and a presence
//!   publisher until the other interface publishes (`PresenceHold`):
//!   approximations with no hardware counterpart.
//! - The second PIC is a port of the IO9 PIC's PXBOW, beside the host's
//!   and the IO9 PIC's links: the PROM and IRIX learn which widgets exist
//!   only from widget 0's link status and auxiliary status, so one
//!   crossbar per node carries both PICs. That is a modelling consequence
//!   of the discovery path, not a known route: whether the second PIC has
//!   a crossbar of its own in front of it, and its part (0xd102 or 0xd112)
//!   and link reset values beyond alive and present, are unknown. QEMU's
//!   composition still gives the second PIC its own PXBOW, whose widget 0
//!   nothing maps, and leaves link 11 absent on the IO9 PIC's. The second
//!   PIC's interrupts and DMA reach Bedrock IO directly, as the IO9 PIC's
//!   do, rather than through the crossbar.
//! - PIC DMA aimed at a widget other than Bedrock IO is refused and logged,
//!   as the Fuel's XBridge path refuses it.
//! - No EEPROM image is fitted for the IO9, the node board or the BCM5701:
//!   no published source gives their contents. The L1's component 0 record
//!   stands in for the node board's: it names the board inventories print
//!   for the chip and processor count (`node_board`), and its
//!   CPU configuration is the flash record's, in the encoding QEMU gives (a
//!   software input the stock PROM accepts, not a read record). No I/O
//!   board record is given, as in QEMU. A BCM5701 EEPROM transfer does not
//!   complete. A fitted V12 is named in component 2 (`V12_BOARD`), with
//!   an Octane2 V12's record rather than a Tezro's.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

use crate::config::{scsi_label, AudioConfig, GraphicsBoard, MachineConfig, TezroConsole};
use crate::cpu::parts::{self, CpuModule, Part};
use crate::dev::bcm5701::{Bcm5701, Bcm5701Config};
use crate::dev::bedrock::{Bedrock, Population, SLOTS};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::ds1744::{Ds1744, YearBase};
use crate::dev::heart::{host_clock, Clock};
use crate::dev::ice1724::{Ice1724, Variant};
use crate::dev::ioc4::{Ioc4, Ioc4Config};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::{BoardIdentity, CpuContext, L1Config, L1};
use crate::dev::odyssey::{Odyssey, PacketSender};
use crate::dev::pic::{Function, Pic, PicPeer, Refused};
use crate::dev::pxbow::Pxbow;
use crate::dev::serial_host::{CiSerialBackend, SerialBackend};
use crate::devlog::LogModule;
use crate::hptimer::TimerManager;
use crate::ip35::presence::PresenceHold;
use crate::ip35_machine::{BoardIo, NodeSpec, SerialPort};
use crate::pci::{DevFn, PciToPciBridge};
use crate::scsi::attach_from_config;
use crate::sn_nodes::{NodeBoard, UncachedBus};
use crate::traits::{BusDevice, Device, Resettable, BUS_OK};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Weak};

/// The Bedrock board word the QEMU Tezro profiles select.
const BOARD_ID: u64 = 0x2001;
/// The IO9's PIC, the second PIC of the two- and four-processor board, and
/// the widget Bedrock IO answers as. Widget 12 on that board is the V12's
/// XIO slot, which is left empty.
const IO9_WIDGET: u8 = 15;
const SECOND_PIC_WIDGET: u8 = 11;
const HOST_WIDGET: u8 = 10;
/// IO9's PCI functions: configuration device on head 0.
const IOC4_DEVICE: u8 = 1;
const SCSI_DEVICE: u8 = 3;
const ETH_DEVICE: u8 = 4;
/// The TI PCI2050 PCI-to-PCI bridge every Tezro capture that prints widget
/// 15 shows at its device 2 (`104c:ac28`, revision 1; established for those
/// captures, and absent from every Origin 350 and Onyx 350 IO9 capture
/// found, for a reason not known). Its secondary bus is empty unless the
/// audio controller is added (see [`AUDIO_DEVICE`]). Only the type 1 header
/// is modelled, not the chip's own registers or capability list.
const TI_BRIDGE_DEVICE: u8 = 2;
const TI_BRIDGE_VENDOR: u16 = 0x104c;
const TI_BRIDGE_ID: u16 = 0xac28;
const TI_BRIDGE_REVISION: u8 = 1;
/// The audio controller's device number on the TI bridge's secondary bus:
/// inferred (the listings print `1412:1724` at "slot 2" beside the bridge,
/// and the slot of a function behind a bridge is its device number there).
const AUDIO_DEVICE: u8 = 2;
/// The IO9 PIC's bus 1, empty on the board, carries the virtio functions
/// (`[virtio_serial]`, `[virtio_9p]`): a software choice, as on the Octane.
const VIRTIO_BUS: usize = 1;
const VIRTIO_SERIAL_DEVICE: u8 = 1;
const VIRTIO_9P_DEVICE: u8 = 2;

/// A DIMM size the Tezro's IP53 board takes, which is also the size of each
/// locational bank it provides: each DIMM pair provides two banks of the
/// DIMM's size (Origin 350 guide 007-4566-001, Figure 3-4, p. 118).
///
/// 512 MB and 1 GB DIMMs only: established for the IP53 node board (Origin
/// 350 guide 007-4566-001, Table 3-3, p. 118); inferred for the Tezro, whose
/// guide (007-4564-001 pp. 27-28) names no size but whose inventories show
/// the same IP53 boards and only 512 and 1024 MB banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankSize {
    Mib512,
    Mib1024,
}
impl BankSize {
    pub const fn mib(self) -> u32 {
        match self {
            Self::Mib512 => 512,
            Self::Mib1024 => 1024,
        }
    }
    fn from_mib(mib: u32) -> Result<Self, String> {
        match mib {
            512 => Ok(Self::Mib512),
            1024 => Ok(Self::Mib1024),
            _ => Err(format!("a Tezro bank is 512 or 1024 MiB, not {mib} MiB")),
        }
    }
}

/// The banks a Tezro fits, locational bank 0 upward: up to eight banks of 512
/// or 1024 MiB with none left out between them. What the board needs in total
/// depends on its processors; see [`Self::check_for_cpus`].
///
/// The minimums come from the Tezro manuals as the QEMU assembly layer
/// records them. The default, the 4 GiB layout of 1024, 1024 and four
/// 512 MiB banks, follows QEMU and matches a captured Tezro.
///
/// Banks 2k and 2k+1 share one size code in Bedrock's memory configuration
/// word (established from the stock IP35 PROM's decode of it), so a
/// population with a lane's two banks at different sizes is refused.
/// Approximated: the board takes DIMMs in pairs (Origin 350 guide p. 116;
/// Tezro guide p. 28), so banks 2k and 2k+1 are present together; the model
/// also accepts a last lane with one bank, which the controller can enable
/// but the board cannot hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPopulation(Vec<BankSize>);
impl Default for MemoryPopulation {
    fn default() -> Self {
        Self::four_gib()
    }
}
impl MemoryPopulation {
    pub fn new(banks: &[BankSize]) -> Result<Self, String> {
        if banks.is_empty() || banks.len() > SLOTS {
            return Err(format!("memory: {} banks; a Tezro fits 1 to {SLOTS}", banks.len()));
        }
        if let Some(lane) = banks.chunks(2).position(|pair| pair.len() == 2 && pair[0] != pair[1]) {
            return Err(format!(
                "memory: banks {} and {} are {} and {} MiB; the two banks of a lane share one size",
                2 * lane, 2 * lane + 1, banks[2 * lane].mib(), banks[2 * lane + 1].mib()));
        }
        Ok(Self(banks.to_vec()))
    }
    /// From sizes in MiB, rejecting any a Tezro does not take.
    pub fn from_mib(banks: &[u32]) -> Result<Self, String> {
        let sizes = banks.iter().map(|&mib| BankSize::from_mib(mib)).collect::<Result<Vec<_>, _>>()?;
        Self::new(&sizes)
    }
    /// 1 GiB as two banks of 512 MiB. The manual's minimum of 1 GB for two or
    /// four processors permits it, but no real Tezro has been observed with
    /// less than 4 GiB, and nothing here claims the PROM or IRIX boot at it.
    pub fn one_gib() -> Self {
        Self(vec![BankSize::Mib512; 2])
    }
    /// 4 GiB: 1024, 1024 and four 512 MiB banks, the default. Banks 4 and 5
    /// sit at 4 to 6 GiB.
    pub fn four_gib() -> Self {
        let mut banks = vec![BankSize::Mib1024; 2];
        banks.extend([BankSize::Mib512; 4]);
        Self(banks)
    }
    pub fn banks(&self) -> &[BankSize] {
        &self.0
    }
    /// The sizes in MiB, as [`BedrockConfig::banks_mib`](crate::dev::bedrock::BedrockConfig) takes them.
    pub fn banks_mib(&self) -> Vec<u32> {
        self.0.iter().map(|bank| bank.mib()).collect()
    }
    pub fn total_mib(&self) -> u32 {
        self.0.iter().map(|bank| bank.mib()).sum()
    }
    /// One processor needs at least 512 MB; two or four need at least 1 GB.
    pub fn check_for_cpus(&self, cpus: usize) -> Result<(), String> {
        let needed = match cpus {
            1 => 512,
            2..=4 => 1024,
            _ => return Err(format!("{cpus} processors; a Tezro fits 1 to 4")),
        };
        if self.total_mib() < needed {
            return Err(format!(
                "{} MiB of memory; {cpus} processor{} need at least {needed} MiB",
                self.total_mib(), if cpus == 1 { "" } else { "s" }));
        }
        Ok(())
    }
}

/// The processors a Tezro takes, the default first: the 700 MHz R16000 a bare
/// `cpu = "r16000"` or `"r16000_700"` selects. Each part's clock, revision,
/// PRId and secondary-cache size are the `Part`'s; `with_cpu!` picks the
/// matching cache type (4 MB for 700 and 800 MHz, 8 MB for 900, 16 MB for 1 GHz).
pub const CPUS: &[&Part] =
    &[&parts::R16000_700_EARLY, &parts::R16000_800, &parts::R16000_900, &parts::R16000_1000];

/// Processors fitted when `[machine] cpus` is unset: the four-processor tower,
/// as in the Tezro inventories seen.
pub const DEFAULT_CPUS: u8 = 4;

/// What the Tezro board knows about one of its chips beyond the chip's own
/// identity: the flash record the PROM checks the processor against (which
/// also gives the low 25 bits of the Config word the processor resets with),
/// the processor counts the chip is fitted with, and the cache clock its
/// inventory prints.
///
/// What is established (E), inferred (I), approximated (A) or unknown (U),
/// by the kind of capture behind it:
///
/// - PRId 0x0f50 for the 1 GHz part: E (PROM banner `f50` and hinv 3.0 on
///   the same machine). 0x0f41 (700 MHz, rev 2.1), 0x0f42 (800 MHz, rev 2.2)
///   and the 900 MHz part's 0x0f50: I. The displayed revisions are printed
///   by inventories and the raw major follows the display rule, which Linux,
///   OpenBSD and PROM 6.210's printers apply (`Part::prid`). No raw read of a
///   2.x part is on record.
///
/// - Config words 0x009bad85 (700 MHz), 0x009baf85 (800 MHz) and 0x009cb185
///   (900 MHz): E. They are the low 25 bits of the mode words in rows of the
///   stock IP35 PROM's CPU speed table, which the PROM writes to Config at
///   reset and reads back (bits 15:0 and 24:16). That the R16000 lays Config
///   out as the PROM's editor does is E for the PROM and I for the chip: no
///   R16000 manual is retained. SB=1 (128-byte lines) is a model choice the
///   table agrees with. Which row a given machine's flash holds is inferred;
///   Tezro inventories' cache taps (0xc, 0xa, 0xb) agree with these rows.
/// - Config word 0x00edb385 (1 GHz): A. The table stops at 900 MHz and no
///   capture records the word; see `FirmwareRecord::TEZRO_1000`.
/// - FIR 0x0900, every chip: U. It is the one R14000 value on record; a model
///   choice. The PROM never reads or prints it and no R16000 FIR was found
///   in the retained evidence.
/// - Config bit 23 and the HSTL bits of the mode word: their meaning is U.
///   Bit 23 is set in every row of 500 MHz and up, and the model copies it.
/// - Processor counts: the PROM has no count field. It lists IP53_4CPU,
///   IP53_2CPU and IP53_1CPU assembly names with no clock paired to any (E
///   that the names exist), which neither forbids nor shows a two-processor
///   part at any clock. The counts follow the boards captured with each
///   processor (retained inventories of real IP53 and IP59 boards). The
///   700 MHz chip is fitted as one, two or four processors, from the board's
///   population and those names. The 800 MHz chip (revision 2.2, 4 MB) as
///   four only: E, both 800 MHz Tezro inventories found print four 2.2
///   processors on `IP53_4CPU` 030-1956-002; the two-processor 800 MHz board (`IP53_2CPU` 030-2029-001, on
///   an Origin 350) carries revision 3.0 processors, another PRId. The 900
///   MHz chip as four only: A, a model choice: no IP53 or IP59 board at 900
///   MHz is captured (the 900 MHz parts on record are on Fuel boards). The
///   1 GHz chip as two or four: E for the counts, from `IP59_4CPU`
///   (030-1989-003) and `IP59_2CPU` (030-2059-002, in a rack Tezro).
/// - The 900 MHz chip: the PROM's speed table has its row, but no IP53 board
///   at 900 MHz is captured, so that a Tezro was fitted with it is not
///   established.
/// - The 1 GHz chip: captured 1 GHz boards are IP59 node boards
///   (`IP59_4CPU`, `IP59_2CPU`), not IP53, and the L1 names them so
///   ([`node_board`]); the rest of the board is the IP53's, and the PROM's
///   socket masks for IP59 equal IP53's.
/// - Cache clock (350, 400, 450 and 333 MHz) is as the inventory prints it
///   and not modelled; the cache taps are only part of the flash records'
///   mode words.
///
/// An 800 MHz part with an 8 MB cache (PROM row 21, 0x549caf85) is on Onyx
/// 350 inventories; no Tezro capture found shows it, and it is not
/// modelled.
pub(crate) struct TezroChip {
    /// The processor module: the part and the mode word its Config resets with.
    pub module: &'static CpuModule,
    pub firmware: FirmwareRecord,
    /// Secondary-cache clock in MHz, as the chip is inventoried.
    #[allow(dead_code)]
    pub scache_mhz: u32,
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
        scache_mhz: 350,
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
        scache_mhz: 400,
        cpus: &[4],
        boards: &[(4, board("IP53_4CPU", "030-1956-002", "-C"))],
        cpus_basis: ", as on every captured board with it; the two-processor 800 MHz board carries revision 3.0 processors",
    },
    TezroChip {
        module: &parts::R16000_900_SYSCLK_200,
        firmware: FirmwareRecord::TEZRO_900,
        scache_mhz: 450,
        cpus: &[4],
        boards: &[(4, board("IP53_4CPU", "", ""))],
        cpus_basis: ", a model choice: no IP53 or IP59 board at 900 MHz is captured (approximated)",
    },
    TezroChip {
        module: &parts::R16000_1000_SYSCLK_200,
        firmware: FirmwareRecord::TEZRO_1000,
        scache_mhz: 333,
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
    /// Whether the chip is fitted as `cpus` processors, and the refusal if
    /// not.
    pub fn accepts_cpus(&self, cpus: u8) -> Result<(), String> {
        if self.cpus.contains(&cpus) {
            return Ok(());
        }
        let list: Vec<String> = self.cpus.iter().map(u8::to_string).collect();
        Err(format!(
            "cpus = {cpus}: the {} is fitted with {} processors only{}",
            self.module.part.label(),
            list.join(", "),
            self.cpus_basis
        ))
    }
}

/// The CPU configuration the node board's L1 record carries: the flash
/// record's mode word and processor clock and the 200 MHz Bedrock. The PROM
/// checks the flash block against it at every start; without it the PROM
/// checks against the table row it picks for board word 0x2001, a 400 MHz
/// part, and prints "CPU configuration values mismatch". A real board's
/// record has not been read, so this is approximated: what QEMU's Tezro gives.
pub(crate) fn cpu_context(firmware: &FirmwareRecord) -> CpuContext {
    CpuContext { mode_word: firmware.cpu_mode_word, cpu_mhz: firmware.cpu_clock_hz / 1_000_000, hub_mhz: 200 }
}

/// The node board the L1's component 0 record names for `chip` fitted as
/// `cpus` processors: the product name, part and revision that board's
/// inventory line prints. Tezro inventories key the board by processor count
/// and clock, and the 800 MHz and 1 GHz chips sit on other parts than the 700
/// MHz one, so the name follows the chip. What is behind each entry:
///
/// - 700 MHz, four processors: `IP53_4CPU` 030-1868-001 revision -C, with
///   2.1 processors at 4 MB and tap 0xc: established (Tezro and Origin 350
///   inventories; every board of that part prints the same tuple).
/// - 700 MHz, two processors: `IP53_2CPU` 030-1869-001 revision -C: inferred
///   from Tezro inventories outside the captures retained beside this
///   model.
/// - 700 MHz, one processor: `IP53_1CPU` with its part and revision empty.
///   The PROM's socket-mask table holds the name; no captured board prints
///   it (inferred for the name, unknown for the rest).
/// - 800 MHz, four processors: `IP53_4CPU` 030-1956-002 revision -C:
///   established for that part, four 2.2 processors at 4 MB and tap 0xa, by
///   both 800 MHz Tezro inventories found (one prints revision -B, one -C;
///   which letter a given board carries is unknown, and -C is the model's).
/// - 900 MHz: `IP53_4CPU` with its part and revision empty. No board at 900
///   MHz is captured on an IP53 or IP59 node (approximated).
/// - 1 GHz, four processors: `IP59_4CPU` 030-1989-003 revision -C:
///   inferred. Every captured board of that part carries four 1.0 GHz
///   revision 3.0 processors at 16 MB and tap 0x15, but on Onyx 350 modules;
///   no 1 GHz Tezro capture prints its node board. Revisions -A to -D are
///   captured; -C is the model's.
/// - 1 GHz, two processors: `IP59_2CPU` 030-2059-002 revision -A, the one
///   two-processor 1 GHz board on record (a rack Tezro): inferred.
///
/// QEMU's Tezro profiles name the 700 MHz boards the same way
/// (hw/mips/sgi/sn-machine.c and the Tezro product layers in
/// jamesbraid/qemu-sgi-origami). The record's other fields are the model's,
/// not read from a board.
///
/// The name matters to the PROM: it tries only the processor sockets in
/// the mask of the product-name row that matches (`IP53_4CPU` and
/// `IP59_4CPU` 0xf, `IP53_2CPU` and `IP59_2CPU` 0x5, `IP53_1CPU` 0x1) and all
/// four when no row matches, power-cycling the module three times for each
/// empty socket it tries (read from the PROM's per-socket bring-up code).
/// Nothing in the PROM or the L1 reads the part or revision.
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

/// The V12's board record, which the node's L1 gives in component 2's board
/// area. IRIX reads it at attach to name the board; without it the boot logs
/// two ALERT lines about the board's NIC name.
///
/// Name, part and revision are what a V12 prints in a published Octane2
/// `hinv -mv` (just.graphica.com.au, "What is inside that box", 2021); the
/// serial is synthetic. Tezro inventories print a different V12 record
/// (`ODY128B1_2`, `030-1884-xxx`). From a reading of the shipped kernel (not
/// run), that name makes it set up the board's digital output, which this
/// model does not implement, so the Octane2's record is used until it does.
const V12_BOARD: BoardIdentity = board("ODY128VERSIONB", "030-1611-001", "C");

/// The serial connections: the L1, then IOC4 ports 0 to 3.
const SERIAL_PORTS: usize = 5;

/// The Tezro's node of `cpus` processors of `chip`, with `memory`, the L1's identity and the
/// boot flash in `flash` or volatile; an error for a population the board cannot fit.
fn spec(
    chip: &'static TezroChip,
    cpus: usize,
    memory: &MemoryPopulation,
    mac: [u8; 6],
    system_serial: Option<String>,
    v12: bool,
    flash: Option<PathBuf>,
) -> Result<NodeSpec, String> {
    if ![1, 2, 4].contains(&cpus) {
        return Err(format!("{cpus} processors; a Tezro board fits 1, 2 or 4"));
    }
    chip.accepts_cpus(cpus as u8)?;
    memory.check_for_cpus(cpus)?;
    Ok(NodeSpec {
        name: "Tezro",
        module: chip.module,
        firmware: chip.firmware,
        board_id: BOARD_ID,
        banks_mib: memory.banks_mib(),
        processors: Population::fitted_cpus(cpus)?,
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
            graphics_board: v12.then_some(V12_BOARD),
            ..Default::default()
        },
        flash,
    })
}

/// The Tezro's node: its processors, memory, L1 identity and V12 as the configuration says.
pub(crate) fn specs(cfg: &MachineConfig) -> Result<Vec<NodeSpec>, String> {
    let memory = cfg.tezro_memory()?;
    let cpus = usize::from(cfg.machine.cpus());
    let chip = chip(cfg.cpu());
    let v12 = cfg.graphics_board() == GraphicsBoard::V12;
    Ok(vec![spec(chip, cpus, &memory, cfg.network().mac, Some(cfg.tezro_system_serial()), v12, cfg.tezro.flash.as_deref().map(PathBuf::from))?])
}

/// The IO9's devices around the node: the PXBOW and the PICs, the IOC4, the ISP12160 and
/// the BCM5701 on the IO9's bus, the audio controller behind the TI bridge when asked for,
/// and the V12 behind the PXBOW. The clocked devices run on `timers`.
pub(crate) fn wire(cfg: &MachineConfig, nodes: &[NodeBoard<Bedrock, Flash>], l1s: &[Arc<L1>], timers: &Arc<TimerManager>) -> Result<BoardIo, String> {
    let (bedrock, l1) = (&nodes[0].chip, &l1s[0]);
    let cpus = usize::from(cfg.machine.cpus());
    let chip = chip(cfg.cpu());
    let running = Population::fitted_cpus(cpus)?;
    let clock = host_clock();
    let timekeeper = Arc::new(Ds1744::new(YearBase::From(1968))?);
    timekeeper.set_nvram_file(Path::new(&cfg.nvram))?;
    let serial: [Arc<CiSerialBackend>; SERIAL_PORTS] = std::array::from_fn(|_| Arc::new(CiSerialBackend::new()));
    l1.set_serial_backend(serial[0].clone());

    let io9 = Pic::new(IO9_WIDGET);
    let pxbow = io9.enable_pxbow(HOST_WIDGET, clock.clone()).map_err(|e| format!("Tezro PXBOW: {e:?}"))?;
    bedrock.connect_widget(0, pxbow.widget_zero());
    bedrock.connect_widget(IO9_WIDGET, pxbow.route(IO9_WIDGET));
    let presence = PresenceHold::new(running, clock.clone(), bedrock);
    let heartbeat = Arc::new(AtomicU64::new(0));
    let fasttick = Arc::new(AtomicU64::new(0));
    let status = crate::disp::StatusSource::new(heartbeat.clone(), fasttick.clone());
    let v12 = (cfg.graphics_board() == GraphicsBoard::V12)
        .then(|| {
            fit_v12(bedrock, &pxbow, &status, (cfg.graphics.threads, cfg.graphics.tile), (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()))
        })
        .transpose()?;
    let pics = if cpus == 1 {
        vec![io9.clone()]
    } else {
        // The PROM finds widget 11 only as a live link of widget 0's
        // crossbar, so the second PIC is a port of the same switch.
        let second = Pic::new(SECOND_PIC_WIDGET);
        pxbow
            .attach_endpoint(SECOND_PIC_WIDGET, second.widget())
            .map_err(|e| format!("Tezro: second PIC on the PXBOW: {e:?}"))?;
        bedrock.connect_widget(SECOND_PIC_WIDGET, pxbow.route(SECOND_PIC_WIDGET));
        vec![second, io9.clone()]
    };
    for pic in &pics {
        pic.connect_peer(Arc::new(BedrockPeer(Arc::downgrade(bedrock))))
            .map_err(|_| "Tezro: PIC peer connected twice".to_string())?;
    }

    let ioc4 = Arc::new(Ioc4::new(Ioc4Config::tezro()));
    ioc4.attach_clock(timekeeper.clone());
    ioc4.set_timer_manager(timers.clone());
    for (port, backend) in serial[1..].iter().enumerate() {
        ioc4.set_serial_backend(port, backend.clone());
    }
    let scsi = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus()));
    scsi.set_heartbeat(heartbeat.clone());
    // The Ethernet slot drives REQ64#, which the chip samples as reset ends.
    let eth = Bcm5701::new(Bcm5701Config { req64: true, ..Bcm5701Config::new(l1.station_mac()) })?;
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
        let card = Arc::new(Ice1724::new(Variant::Dm8));
        if cfg.no_audio {
            eprintln!("iris: {AUDIO_WITH_NO_AUDIO}");
        } else {
            card.enable_output(cfg.audio.clone());
        }
        ti.secondary().attach(DevFn { device: AUDIO_DEVICE, function: 0 }, card.clone());
        card.set_timer_manager(timers.clone());
        card
    });
    io9.attach(0, SCSI_DEVICE, scsi.clone(), Function::default());
    io9.attach(0, ETH_DEVICE, eth.clone(), Function { self_ordered_pio: false, req64: true });

    let scratch = attach_scsi(cfg, &scsi)?;
    let virtio = Arc::new(attach_virtio(cfg, &io9));
    eprintln!(
        "iris: Tezro IP53, R16000 {} MHz, {}",
        chip.module.part.mhz,
        if v12.is_some() { cfg.graphics_board().label() } else { "no graphics" },
    );
    let mut monitor: Vec<Arc<dyn Device>> = vec![scsi.clone(), ioc4.ps2()];
    if let Some(v12) = &v12 {
        monitor.push(v12.clone());
    }
    if let Some(audio) = &audio {
        monitor.push(audio.clone());
    }
    let (start_v12, start_ioc4, start_eth, start_scsi, start_audio) =
        (v12.clone(), ioc4.clone(), eth.clone(), scsi.clone(), audio.clone());
    let (stop_v12, stop_ioc4, stop_eth, stop_scsi, stop_audio) =
        (v12.clone(), ioc4.clone(), eth.clone(), scsi.clone(), audio.clone());
    let (start_virtio, stop_virtio, reset_virtio) = (virtio.clone(), virtio.clone(), virtio);
    let (reset_presence, reset_ioc4, reset_scsi, reset_eth, reset_pics, reset_v12) =
        (presence.clone(), ioc4.clone(), scsi.clone(), eth.clone(), pics.clone(), v12.clone());
    let held = presence.clone();
    let cpu_presence = presence;
    let (lock_scsi, lock_ioc4, lock_eth, lock_pics, lock_audio) =
        (scsi.clone(), ioc4.clone(), eth.clone(), pics.clone(), audio.clone());
    let (cpu_eth, disks) = (eth.clone(), scsi.clone());
    Ok(BoardIo {
        start: Box::new(move || {
            if let Some(v12) = &start_v12 {
                v12.start();
            }
            start_ioc4.start();
            start_eth.start();
            start_scsi.start();
            if let Some(audio) = &start_audio {
                audio.start();
            }
            for v in start_virtio.iter() {
                v.start();
            }
        }),
        stop: Box::new(move || {
            for v in stop_virtio.iter() {
                v.stop();
            }
            if let Some(v12) = &stop_v12 {
                v12.stop();
            }
            if let Some(audio) = &stop_audio {
                audio.stop();
            }
            stop_scsi.stop();
            stop_eth.stop();
            stop_ioc4.stop();
        }),
        reset: Box::new(move || {
            reset_presence.reset();
            reset_ioc4.power_on();
            reset_scsi.power_on();
            reset_eth.power_on();
            for pic in &reset_pics {
                pic.power_on();
            }
            if let Some(v12) = &reset_v12 {
                v12.power_on();
            }
            for v in reset_virtio.iter() {
                Resettable::power_on(&**v);
            }
        }),
        poll: Arc::new(|| {}),
        at_batch: (running.count() > 1).then(|| Arc::new(move || held.at_batch()) as Arc<dyn Fn() + Send + Sync>),
        register_locks: Box::new(move || {
            lock_scsi.register_locks();
            lock_ioc4.register_locks();
            lock_eth.register_locks();
            for pic in &lock_pics {
                pic.register_locks();
            }
            if let Some(audio) = &lock_audio {
                audio.register_locks();
            }
        }),
        monitor,
        ps2: ioc4.ps2(),
        flush_nvram: Box::new(move || {
            timekeeper.flush();
            Ok(())
        }),
        serial: serial_ports(cfg, l1, &ioc4, &serial),
        console: cfg.tezro.console.index(),
        attach_net: Box::new(move |link| eth.attach(link)),
        disks: Box::new(move |f| f(&(0..2).filter_map(|channel| disks.bus_for_channel(channel)).collect::<Vec<_>>())),
        disc_changer: Some(scsi),
        scratch,
        display: v12.as_ref().map(|v12| v12.display()),
        status: Some(status),
        fasttick: Some(fasttick),
        // The Ethernet holds a reply until the processor that sent the
        // request has run on past its doorbell.
        cpus: Box::new(move |all| {
            cpu_presence.set_cpus(all);
            cpu_eth.set_cpu_cycles(all.iter().map(|c| c.cycles_ptr()).collect());
        }),
        #[cfg(test)]
        xbridge: None,
        #[cfg(test)]
        pics,
    })
}

fn attach_scsi(cfg: &MachineConfig, scsi: &Isp12160) -> Result<Option<PathBuf>, String> {
    let attached = attach_from_config(cfg, Some("tezro"), &|channel| {
        scsi.bus_for_channel(channel)
            .ok_or_else(|| format!("controller {channel} (a Tezro has channels 0 and 1)"))
    })?;
    if let Some(device) = attached.daynaports.first() {
        return Err(format!("{}: kind = \"daynaport\" is not available on the Tezro",
            scsi_label(device.controller, device.id)));
    }
    Ok(attached.scratch)
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
            setting: "tezro.l1_port",
            ci: serial[0].clone(),
            log: cfg.tezro.l1_log.clone(),
            connect: Box::new(move |backend| l1.set_serial_backend(backend)),
        },
        SerialPort { label: "IOC4 0", port: a, setting: "serial_port_a", ci: serial[1].clone(), log: None, connect: ioc4(0) },
        SerialPort { label: "IOC4 1", port: b, setting: "serial_port_b", ci: serial[2].clone(), log: None, connect: ioc4(1) },
        SerialPort {
            label: "IOC4 2",
            port: cfg.tezro.serial_port_c.unwrap_or(8883),
            setting: "tezro.serial_port_c",
            ci: serial[3].clone(),
            log: None,
            connect: ioc4(2),
        },
        SerialPort {
            label: "IOC4 3",
            port: cfg.tezro.serial_port_d.unwrap_or(8884),
            setting: "tezro.serial_port_d",
            ci: serial[4].clone(),
            log: None,
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

/// Said at startup and by `--validate` when the audio card is added with
/// `no_audio` set: the guest sees the card and nothing is played.
pub(crate) const AUDIO_WITH_NO_AUDIO: &str =
    "[tezro] audio is set but no_audio is too: the audio card is added and plays to nowhere";

/// Where the V12 sits: widget 12 of the crossbar behind the PIC that owns
/// widget 0. On the two- and four-processor board this is QEMU's graphics
/// slot. On the one-processor board the tower guide gives one XIO connector
/// for a VPro board but names neither its widget nor its route; widget 12
/// behind the same PXBOW is inferred, not established.
const V12_WIDGET: u8 = 12;

/// Fit the V12 behind the PXBOW: its interrupt packets and DMA reach
/// Bedrock IO as the PIC's do. Nothing counts the processors' stores into
/// its graphics page or returns credits for them.
fn fit_v12(
    bedrock: &Arc<Bedrock>,
    pxbow: &Arc<Pxbow>,
    status: &Arc<crate::disp::StatusSource>,
    graphics_pool: (usize, u32),
    graphics_raster: (crate::config::RasterUnit, &str),
) -> Result<Arc<Odyssey>, String> {
    let io = Arc::downgrade(bedrock);
    let packets: PacketSender = Arc::new(move |widget, address, vector, set| {
        if widget != HOST_WIDGET {
            dlog!(LogModule::Bedrock, "Tezro: V12 interrupt to widget {widget} refused; only Bedrock IO is reachable");
            return false;
        }
        let Some(io) = io.upgrade() else { return false };
        let payload = u64::from(vector) | if set { 1 << 8 } else { 0 };
        io.inbound_write(V12_WIDGET, Some(INBOUND_DEVICE), address, 8, payload) == BUS_OK
    });
    let dma: Arc<dyn BusDevice> = bedrock.inbound(V12_WIDGET);
    let v12 = Odyssey::new_widget_relative(
        V12_WIDGET,
        GraphicsBoard::V12.as_str(),
        packets,
        dma,
        status.clone(),
        graphics_pool,
        graphics_raster,
    )?;
    pxbow
        .attach_endpoint(V12_WIDGET, v12.clone() as Arc<dyn BusDevice>)
        .map_err(|e| format!("Tezro V12: {e:?}"))?;
    bedrock.connect_widget(V12_WIDGET, pxbow.route(V12_WIDGET));
    Ok(v12)
}

/// The far end of a PIC's link: Bedrock IO. Interrupt packets and DMA arrive
/// there as inbound packets from the PIC's widget, device 0, as QEMU's
/// Bedrock IO takes them by default. DMA aimed at another widget is refused:
/// peer-to-peer XIO is not modelled.
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

    /// The legs of a transfer, as (offset, bytes). Whole doublewords from an
    /// 8-aligned address go as one leg, a run for the memory to take in a
    /// block; the ragged bytes around it are each the largest naturally
    /// aligned size of four bytes or fewer that fits, so a word or halfword
    /// the device writes whole stays one access.
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

/// Doublewords moved per call to memory. A stack buffer of this size is
/// zeroed on every transfer, so it is small: 512 bytes cost a few stores,
/// a whole page a `rep stos` that outweighed the saving in the guest.
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
