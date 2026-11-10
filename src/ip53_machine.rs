//! The Tezro (IP53 board, IP35 software platform): R16000s on one Bedrock
//! node, a PIC host bridge and the IO9 board's IOC4, ISP12160 and BCM5701,
//! with Iris's CPU thread, monitor and serial console.
//!
//! Selected with `[machine] profile = "tezro_ip53"`. Shared keys select the PROM,
//! NVRAM, memory and SCSI devices; `[tezro]` selects the processor
//! population, the serial console and writable flash. The node itself is
//! the one the Fuel uses ([`Node`]); this board adds its widgets and its
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
//!   reaching PIC input d - 1 is what IRIX's pin rule assigns and what an
//!   Origin 350 under OpenBSD reports (established as software
//!   expectation); the board wiring itself, and INTB to INTD, are unknown.
//! - The one-processor board has one PIC. The tower guide gives that board
//!   two PCI-X buses and one XIO connector; that it is one PIC at widget 15
//!   is inferred from them.
//! - Every fitted processor runs from reset, each on its own thread, as in
//!   QEMU's IP35 node; CPU enable is stored and starts or stops nothing. The
//!   PROM's reset barrier is ordered by Bedrock's LB; the first of a pair
//!   to post its liveness is held until its partner posts, and a presence
//!   publisher until the other interface publishes ([`PresenceHold`]):
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
//!   for the chip and processor count ([`node_board`]), and its
//!   CPU configuration is the flash record's, in the encoding QEMU gives (a
//!   software input the stock PROM accepts, not a read record). No I/O
//!   board record is given, as in QEMU. A BCM5701 EEPROM transfer does not
//!   complete.
//!
//! Board composition ported from hw/mips/sgi/sn-machine.c and
//! hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use crate::config::{scsi_label, GraphicsBoard, MachineConfig, TezroConsole};
use crate::cpu::mips_cache_v2::{CpuModel as CacheCpuModel, MipsCache};
use crate::cpu::parts::{self, Part};
use crate::cpu::mips_exec::{CpuDevice, MipsCpu, MipsCpuConfig, MipsExecutor};
use crate::cpu::cpu_monitor::CpuMonitor;
use crate::cpu::mips_tlb::MipsTlb;
use crate::cpu::smp::World;
use crate::dev::bcm5701::{Bcm5701, Bcm5701Config};
use crate::dev::bedrock::{Bedrock, Population, SLOTS};
use crate::dev::bedrock_flash::{FirmwareRecord, Flash};
use crate::dev::ds1744::{Ds1744, YearBase};
use crate::dev::heart::{host_clock, Clock};
use crate::dev::ioc4::{Ioc4, Ioc4Config};
use crate::dev::isp12160::Isp12160;
use crate::dev::l1::{BoardIdentity, CpuContext, L1Config};
use crate::dev::odyssey::{Odyssey, PacketSender};
use crate::dev::pic::{Function, Pic, PicPeer, Refused};
use crate::dev::pxbow::Pxbow;
use crate::dev::prom_image;
use crate::dev::ps2::Ps2Controller;
use crate::dev::serial_host::{null_serial_backend, open_log, tcp_serial_backend, CiSerialBackend, LogMode, SerialBackend, TeeBackend};
use crate::dev::l1::L1;
use crate::devlog::{init_devlog, LogModule};
use crate::gdb_stub::CpuDebug;
use crate::gfx_display::GfxDisplay;
use crate::hptimer::TimerManager;
use crate::ip35::{Ip35Bus, CPU_PHYS_MASK};
use crate::ip35::node::{connect_cpus, Node, NodeConfig};
use crate::ip35::presence::PresenceHold;
use crate::machine_state::GuestState;
use crate::monitor::{Monitor, DEFAULT_PORT as DEFAULT_MONITOR_PORT};
use crate::net::{GatewayConfig, Link, NatControl};
use crate::perf_monitor::PerfMonitor;
use crate::pci::{DevFn, PciToPciBridge};
use crate::scsi::{attach_from_config, ScsiBus};
use crate::traits::{BusDevice, Device, MachineEvent, Resettable, BUS_OK};
use crate::workstation::{register_system_controller, DiscChanger, MachineRef, MachineSlot, Workstation};
use parking_lot::Mutex;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{atomic::{AtomicBool, AtomicU64, Ordering}, Arc, Weak};

/// The Config bits 24:0 a chip's reset word defines (K0, DN, CT, PE, PM, EC,
/// SB, SK, BE, SS, SC and the bits above them). The primary-cache sizes above
/// them are the family's, fixed by the cache type.
const CONFIG0_MODE_MASK: u32 = 0x01ff_ffff;

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
/// found, for a reason not known). No capture shows anything behind it, so
/// its secondary bus is empty (inferred). Only the type 1 header is
/// modelled, not the chip's own registers or capability list.
const TI_BRIDGE_DEVICE: u8 = 2;
const TI_BRIDGE_VENDOR: u16 = 0x104c;
const TI_BRIDGE_ID: u16 = 0xac28;
const TI_BRIDGE_REVISION: u8 = 1;

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
    pub chip: &'static Part,
    pub firmware: FirmwareRecord,
    /// Secondary-cache clock in MHz, as the chip is inventoried.
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
        chip: &parts::R16000_700_EARLY,
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
        chip: &parts::R16000_800,
        firmware: FirmwareRecord::TEZRO_800,
        scache_mhz: 400,
        cpus: &[4],
        boards: &[(4, board("IP53_4CPU", "030-1956-002", "-C"))],
        cpus_basis: ", as on every captured board with it; the two-processor 800 MHz board carries revision 3.0 processors",
    },
    TezroChip {
        chip: &parts::R16000_900,
        firmware: FirmwareRecord::TEZRO_900,
        scache_mhz: 450,
        cpus: &[4],
        boards: &[(4, board("IP53_4CPU", "", ""))],
        cpus_basis: ", a model choice: no IP53 or IP59 board at 900 MHz is captured (approximated)",
    },
    TezroChip {
        chip: &parts::R16000_1000,
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
    CHIPS.iter().find(|c| *c.chip == *chip).expect("a Tezro takes only its own R16000 parts")
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
            self.chip.label(),
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

/// The serial connections: the L1, then IOC4 ports 0 to 3.
const SERIAL_PORTS: usize = 5;

pub struct TezroMachine {
    board: Arc<TezroBoard>,
    /// The processors, CPU 0 first, in the board's slot order.
    cpus: Vec<Arc<dyn CpuDevice>>,
    timers: Arc<TimerManager>,
    monitor: Arc<Monitor>,
    monitor_port: u16,
    ci_serial: Option<Arc<CiSerialBackend>>,
    console: usize,
    ci: bool,
    scratch: Option<PathBuf>,
    nvram: String,
    frames: Option<Arc<AtomicU64>>,
    nat: Option<(Arc<NatControl>, (Ipv4Addr, Ipv4Addr, Ipv4Addr))>,
    events: Option<Receiver<MachineEvent>>,
    slot: Option<MachineSlot>,
    lifecycle: GuestState,
}

impl TezroMachine {
    pub fn new(cfg: &MachineConfig) -> Result<Self, String> {
        cfg.validate()?;
        let prom = prom_image::load(Path::new(&cfg.prom), 8 * 1024 * 1024).map_err(|e| format!("Tezro {e}"))?;
        let clock = Arc::new(Ds1744::new(YearBase::From(1968))?);
        clock.set_nvram_file(Path::new(&cfg.nvram))?;
        let memory = cfg.tezro_memory()?;
        let chip = chip(cfg.cpu());
        let board = Arc::new(TezroBoard::with_config(TezroBoardConfig {
            payload: &prom,
            clock: host_clock(),
            flash_path: cfg.tezro.flash.as_deref().map(Path::new),
            station_mac: cfg.network().mac,
            system_serial: Some(cfg.tezro_system_serial()),
            timekeeper: clock,
            memory: &memory,
            cpus: usize::from(cfg.machine.cpus()),
            chip,
            v12: cfg.graphics_board() == GraphicsBoard::V12,
            graphics_pool: (cfg.graphics.threads, cfg.graphics.tile),
            graphics_raster: (cfg.graphics.raster, cfg.graphics.gpu_backend.as_str()),
        })?);
        let timers = Arc::new(TimerManager::new());
        board.ioc4.set_timer_manager(timers.clone());
        let count_hz = cfg.clock.fixed_mhz.map_or(chip.chip.count_hz(), |mhz| (mhz * 1_000_000.0) as u64).max(1);
        let cpus = build_cpus(&board, &timers, count_hz);
        let cpu = cpus[0].clone();
        board.status.set_cpu_cycles(cpu.cycles_ptr());
        board.status.set_count_hz(cpu.count_hz_atomic());
        for c in cpus.iter().skip(1) {
            board.status.add_cpu_stats(c.cycles_ptr(), c.fasttick_count());
        }
        // The Ethernet holds a reply until the processor that sent the request
        // has run on past its doorbell.
        board.eth.set_cpu_cycles(cpus.iter().map(|c| c.cycles_ptr()).collect());
        // One page pool and one body of compiled code, reaching each
        // processor's state through the core it is called with.
        let world = Arc::new(World::new());
        world.set_slots(cpus.iter().map(|c| c.park_slot()).collect());
        // A bank configuration write parks the others while it moves memory.
        board.bedrock.set_world(world.clone());
        #[cfg(feature = "jitv2")]
        if cpus.len() > 1 {
            if cfg!(feature = "jitv2_lockstep") {
                return Err("[machine] cpus above 1 cannot run with jitv2_lockstep".into());
            }
            let jit = cpu.jitv2();
            for c in &cpus {
                c.share_jitv2(jit.clone(), world.clone());
            }
        }
        board.presence.set_cpus(&cpus);
        connect_cpus("Tezro", &board.bedrock, &board.l1, &board.events, &cpus)?;
        board.bedrock.set_timer_manager(&timers);
        let rx = board.receiver.lock().take().expect("the board's events are taken once");
        // After connect_cpu set the windows: each compile worker takes the
        // cache's geometry and the CPU model's policies once, as it starts.
        // Workers read plain physical code addresses through the view the CPU
        // fetched each page by: the cached plane for cached pages, the
        // uncached attribute-0 view for the reset PROM's uncached fetches.
        #[cfg(feature = "jitv2")]
        {
            let cpu_device: Arc<dyn Device> = cpu.clone().as_device();
            let jit = cpu.jitv2();
            let mut g = jit.lock();
            g.compile_queue.set_thread_count(cfg.jitv2.threads.max(1));
            g.compile_queue.set_cpu(Arc::downgrade(&cpu_device));
            g.compile_queue.set_owner(Arc::downgrade(&jit));
            if cpus.len() > 1 {
                g.compile_queue.set_world(world.clone());
            }
            #[cfg(not(feature = "jitv2_lockstep"))]
            {
                let stats = g.stats.clone();
                g.compile_queue.start(board.code_bus(), stats);
            }
            // The compile pool keeps only a weak handle on the CPU.
            std::mem::forget(cpu_device);
        }
        let scratch = Self::attach_scsi(cfg, &board)?;
        let console = cfg.tezro.console.index();
        let ci_serial = cfg.ci.then(|| board.serial[console].clone());
        Self::connect_serial(cfg, &board, console)?;
        let monitor = Monitor::new();
        monitor.register_device(init_devlog());
        monitor.register_device(Arc::new(CpuMonitor::new(cpus.clone())));
        monitor.register_device(PerfMonitor::new(
            cpu.running_flag(), cpu.cycles_ptr(), cpu.fasttick_count(), None, None,
        ));
        for device in board.monitor_devices() {
            monitor.register_device(device);
        }
        for c in &cpus {
            c.register_locks();
        }
        board.scsi.register_locks();
        board.ioc4.register_locks();
        board.eth.register_locks();
        for pic in &board.pics {
            pic.register_locks();
        }
        board.flash.register_locks();
        board.bedrock.register_locks();
        let (mut nat, mut frames) = (None, None);
        let net = GatewayConfig::from_network(&cfg.network());
        if let Some(link) = Link::with_gateway(net.clone())? {
            frames = Some(link.frames());
            nat = Some((link.nat(), (net.client_ip, net.gateway_ip, net.netmask)));
            board.eth.attach(link);
        }
        eprintln!(
            "iris: Tezro IP53, {} R16000 {} MHz, {} MiB, {}, serial console {:?}",
            cpus.len(),
            chip.chip.mhz,
            memory.total_mib(),
            if board.v12.is_some() { cfg.graphics_board().label() } else { "no graphics" },
            cfg.tezro.console
        );
        Ok(Self {
            board,
            cpus,
            timers,
            monitor: Arc::new(monitor),
            monitor_port: cfg.monitor_port.unwrap_or(DEFAULT_MONITOR_PORT),
            ci_serial,
            console,
            ci: cfg.ci,
            scratch,
            nvram: cfg.nvram.clone(),
            nat,
            frames,
            events: Some(rx),
            slot: None,
            lifecycle: GuestState::Idle,
        })
    }

    fn attach_scsi(cfg: &MachineConfig, board: &TezroBoard) -> Result<Option<PathBuf>, String> {
        let attached = attach_from_config(cfg, Some("tezro"), &|channel| {
            board.scsi.bus_for_channel(channel)
                .ok_or_else(|| format!("controller {channel} (a Tezro has channels 0 and 1)"))
        })?;
        if let Some(device) = attached.daynaports.first() {
            return Err(format!("{}: kind = \"daynaport\" is not available on the Tezro",
                scsi_label(device.controller, device.id)));
        }
        Ok(attached.scratch)
    }

    fn connect_serial(cfg: &MachineConfig, board: &TezroBoard, console: usize) -> Result<(), String> {
        let [a, b] = cfg.serial_ports();
        let ioc4 = |port: usize| -> Box<dyn Fn(Arc<dyn crate::dev::serial_host::SerialBackend>) + '_> {
            Box::new(move |backend| board.ioc4.set_serial_backend(port, backend))
        };
        let ports = [
            SerialPort {
                label: "L1",
                port: cfg.tezro.l1_port.unwrap_or(8882),
                setting: "tezro.l1_port",
                ci: &board.serial[0],
                log: cfg.tezro.l1_log.clone(),
                connect: Box::new(|backend| board.l1.set_serial_backend(backend)),
            },
            SerialPort { label: "IOC4 0", port: a, setting: "serial_port_a", ci: &board.serial[1], log: None, connect: ioc4(0) },
            SerialPort { label: "IOC4 1", port: b, setting: "serial_port_b", ci: &board.serial[2], log: None, connect: ioc4(1) },
            SerialPort {
                label: "IOC4 2",
                port: cfg.tezro.serial_port_c.unwrap_or(8883),
                setting: "tezro.serial_port_c",
                ci: &board.serial[3],
                log: None,
                connect: ioc4(2),
            },
            SerialPort {
                label: "IOC4 3",
                port: cfg.tezro.serial_port_d.unwrap_or(8884),
                setting: "tezro.serial_port_d",
                ci: &board.serial[4],
                log: None,
                connect: ioc4(3),
            },
        ];
        connect_serial_ports(cfg, "Tezro", &ports, console)
    }
}

/// One of the board's serial connections: what it is called, the TCP port
/// and the setting that names the port, the CI backend it uses when it is
/// the console, the log it keeps when it is not, and where the host end goes.
struct SerialPort<'a> {
    label: &'static str,
    port: u16,
    setting: &'static str,
    ci: &'a Arc<CiSerialBackend>,
    log: Option<String>,
    connect: Box<dyn Fn(Arc<dyn SerialBackend>) + 'a>,
}

/// Connect the board's serial ports to the host. The console takes
/// `serial_log` and, under CI, the in-process backend; the others listen on
/// their TCP ports, and a port that is taken leaves its stream unconnected
/// rather than stopping the machine.
fn connect_serial_ports(cfg: &MachineConfig, name: &str, ports: &[SerialPort<'_>], console: usize) -> Result<(), String> {
    for (i, a) in ports.iter().enumerate() {
        if let Some(b) = ports[i + 1..].iter().find(|b| b.port == a.port) {
            return Err(format!("{name} {} and {} serial ports must be distinct", a.label, b.label));
        }
    }
    let mut log_config = cfg.clone();
    for (index, port) in ports.iter().enumerate() {
        log_config.serial_log = if index == console { cfg.serial_log.clone() } else { port.log.clone() };
        let tag = format!("{name} {}", port.label);
        let log = open_log(&log_config, &tag, LogMode::Fresh);
        let backend: Arc<dyn SerialBackend> = if cfg.ci && index == console {
            if let Some(log) = log {
                port.ci.set_log(log);
            }
            port.ci.clone()
        } else {
            let mut backend = null_serial_backend();
            match tcp_serial_backend(port.port) {
                Some((socket, actual_port)) => {
                    eprintln!("iris: {tag} on 127.0.0.1:{actual_port}");
                    backend = socket;
                }
                None => eprintln!(
                    "iris: {tag}: port {} unavailable; is another iris using it? ({})",
                    port.port, port.setting
                ),
            }
            match log {
                Some(log) => Arc::new(TeeBackend::new(backend, log)),
                None => backend,
            }
        };
        (port.connect)(backend);
    }
    Ok(())
}

impl Workstation for TezroMachine {
    fn start(&mut self) {
        self.lifecycle = GuestState::Starting;
        self.monitor.clone().start_server(format!("127.0.0.1:{}", self.monitor_port));
        self.restart_peripherals();
        if !self.ci {
            self.cpu_start();
        }
    }
    fn stop(&mut self) {
        for c in &self.cpus {
            c.clone().as_device().stop();
        }
        self.board.stop();
        self.board.timekeeper.flush();
        if self.lifecycle != GuestState::Idle {
            self.lifecycle = GuestState::Halted;
        }
    }
    fn reset(&mut self) {
        self.stop();
        for c in &self.cpus {
            c.power_on();
        }
        self.board.power_on();
        self.restart_peripherals();
    }
    fn restart_peripherals(&mut self) {
        self.board.start();
    }
    /// Every processor starts at once, as from the node's reset.
    fn cpu_start(&self) {
        for c in &self.cpus {
            c.clone().as_device().start();
        }
    }
    fn cpu_is_running(&self) -> bool {
        self.cpus.iter().any(|c| c.clone().as_device().is_running())
    }
    fn machine_slot(&mut self) -> MachineSlot {
        let ptr = self as *mut Self as *mut dyn Workstation;
        self.slot.get_or_insert_with(|| Arc::new(Mutex::new(MachineRef(ptr)))).clone()
    }
    fn register_system_controller(&mut self) {
        let slot = self.machine_slot();
        let events = self.events.take().expect("Tezro system controller registered twice");
        register_system_controller(&self.monitor, slot, Some(events));
    }
    fn monitor_addr(&self) -> Option<SocketAddr> {
        self.monitor.bound_addr()
    }
    fn lifecycle(&self) -> GuestState {
        self.lifecycle
    }
    fn set_lifecycle(&mut self, state: GuestState) {
        self.lifecycle = state;
    }
    fn get_display(&self) -> Option<Arc<dyn GfxDisplay>> {
        self.board.v12.as_ref().map(|v12| v12.display())
    }
    fn register_monitor_device(&self, device: Arc<dyn Device>) {
        self.monitor.register_device(device);
    }
    fn get_ps2(&self) -> Arc<Ps2Controller> {
        self.board.ioc4.ps2()
    }
    fn get_timer_manager(&self) -> Arc<TimerManager> {
        self.timers.clone()
    }
    fn get_cpu_debug(&self) -> Arc<dyn CpuDebug> {
        self.cpu().clone().debug_adapter()
    }
    fn fasttick_count(&self) -> Arc<AtomicU64> {
        self.cpu().fasttick_count()
    }
    fn cpu_counters(&self) -> Vec<(crate::cpu::mips_core::CyclesPtr, Arc<AtomicU64>)> {
        self.cpus.iter().map(|c| (c.cycles_ptr(), c.fasttick_count())).collect()
    }
    fn get_ci_serial(&self) -> Option<Arc<CiSerialBackend>> {
        self.ci_serial.clone()
    }
    fn inject_serial_console(&self, bytes: &[u8]) {
        self.board.serial[self.console].push_host(bytes);
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn save_nvram(&self, path: Option<&str>) -> Result<String, String> {
        if path.is_some_and(|p| p != self.nvram) {
            return Err("Tezro NVRAM save uses the configured path".into());
        }
        self.board.timekeeper.flush();
        Ok(self.nvram.clone())
    }
    fn scratch_path(&self) -> Option<&Path> {
        self.scratch.as_deref()
    }
    fn disc_changer(&self) -> Option<Arc<dyn DiscChanger>> {
        Some(self.board.scsi.clone() as Arc<dyn DiscChanger>)
    }
    fn pending_chd_sync_count(&self) -> usize {
        self.scsi_buses().iter().map(|bus| bus.pending_chd_sync_count()).sum()
    }
    fn flush_disks(&self) -> std::io::Result<()> {
        self.board.flush_disks()
    }
    fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        ScsiBus::sync_chd_disks(&self.scsi_buses(), only, progress, cancel)
    }
    fn net_frames(&self) -> Option<Arc<AtomicU64>> {
        self.frames.clone()
    }
    fn nat_control(&self) -> Option<Arc<NatControl>> {
        self.nat.as_ref().map(|(n, _)| n.clone())
    }
    fn nat_expected(&self) -> Option<(Ipv4Addr, Ipv4Addr, Ipv4Addr)> {
        self.nat.as_ref().map(|(_, a)| *a)
    }
}

impl TezroMachine {
    /// CPU 0, the one the front ends drive.
    fn cpu(&self) -> &Arc<dyn CpuDevice> {
        &self.cpus[0]
    }

    fn scsi_buses(&self) -> Vec<&ScsiBus> {
        (0..2).filter_map(|channel| self.board.scsi.bus_for_channel(channel)).collect()
    }
}

impl Drop for TezroMachine {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            slot.lock().0 = MachineRef::null();
        }
        self.stop();
        self.monitor.shutdown();
    }
}

/// The processors the machine runs, of the board's chip: one for each
/// fitted slot, CPU 0 first, in the board's slot order.
fn build_cpus(board: &Arc<TezroBoard>, timers: &Arc<TimerManager>, count_hz: u64) -> Vec<Arc<dyn CpuDevice>> {
    let count = board.processors.count();
    crate::with_cpu!(board.chip.chip, true, false, C => {
        (0..count).map(|id| start_cpu::<C>(board, timers, count_hz, id as u8, count)).collect()
    })
}

/// Processor `id` of `count`, of type `C`, with its timers; CPU 0 also
/// runs the board's poll, and on a board of several processors every one
/// takes part in the PROM's holds at its batch boundary. A chip with a
/// secondary cache of another size is another type, instantiated here.
fn start_cpu<C>(board: &Arc<TezroBoard>, timers: &Arc<TimerManager>, count_hz: u64, id: u8, count: usize) -> Arc<dyn CpuDevice>
where
    C: CacheCpuModel + From<Arc<dyn BusDevice>> + 'static,
    MipsCpu<MipsTlb, C>: CpuDevice,
{
    let mut exec = build_cpu::<C>(board);
    // CPU 0's Compare matches blink the graphics status bar's LED.
    exec.core.fasttick_count = if id == 0 { board.fasttick.clone() } else { Arc::new(AtomicU64::new(0)) };
    exec.core.set_timer_manager(timers.clone());
    exec.core.set_count_hz(count_hz);
    exec.core.count_write_samples = true;
    exec.core.cpu_id = id;
    exec.core.smp = count > 1;
    let poll_board = board.clone();
    let cpu = usize::from(id);
    exec.batch_hook = match (id, count > 1) {
        (0, false) => Some(Box::new(move |_| poll_board.poll())),
        (0, true) => Some(Box::new(move |_| {
            poll_board.poll();
            poll_board.presence.at_batch(cpu);
        })),
        (_, true) => Some(Box::new(move |_| poll_board.presence.at_batch(cpu))),
        (_, false) => None,
    };
    Arc::new(MipsCpu::new(*exec))
}

/// A processor of type `C` with the board chip's identity and reset-mode
/// inputs, on the board's bus.
fn build_cpu<C: CacheCpuModel + From<Arc<dyn BusDevice>>>(board: &TezroBoard) -> Box<MipsExecutor<MipsTlb, C>> {
    let chip = board.chip.chip;
    let cfg = MipsCpuConfig::for_model::<C>(Arc::new(crate::ip35::memory_map(&board.bedrock.ram_extent())))
        .with_prid(chip.prid());
    let mut cpu = Box::new(MipsExecutor::new(
        board.bus.clone() as Arc<dyn BusDevice>,
        MipsTlb::new(cfg.tlb_entries),
        &cfg,
    ));
    cpu.phys_mask = CPU_PHYS_MASK;
    // The Config word of the chip's flash record over the low 25 bits the
    // cache type builds (FIR is the type's). The flash record's mode word and
    // the processor's reset mode must agree: on a mismatch the PROM's early
    // check calls a routine that clears the processor's presence and enable
    // bits and returns into the normal start-up, but it runs before GP is
    // set, so what it does on silicon is unknown; on the model it stops in
    // that routine.
    cpu.core.cp0_config = (cpu.core.cp0_config & !CONFIG0_MODE_MASK) | board.chip.firmware.reset_mode();
    cpu.cache = C::from(board.cached_bus());
    // The PROM fetches its own code uncached; with memory over its window
    // the cached plane answers RAM there, so the fetch is presented at the
    // PROM's own address and compiled from the view that holds it.
    cpu.set_uncached_fetch_alias(crate::ip35::RESET_WINDOW, crate::ip35::RESET_WINDOW_SIZE, crate::ip35::RESET_WINDOW_FETCH);
    #[cfg(feature = "jitv2")]
    cpu.set_cached_bus(board.code_bus());
    #[cfg(not(feature = "jitv2"))]
    cpu.set_cached_bus(board.cached_bus());
    cpu
}

pub(crate) struct TezroBoardConfig<'a> {
    pub payload: &'a [u8],
    pub clock: Clock,
    pub flash_path: Option<&'a Path>,
    pub station_mac: [u8; 6],
    /// The system serial every L1 answers with, the text IRIX folds into
    /// its host id. `None` answers with the station address in hex.
    pub system_serial: Option<String>,
    /// The DS1744-compatible clock and NVRAM in the IOC4's BAR.
    pub timekeeper: Arc<Ds1744>,
    pub memory: &'a MemoryPopulation,
    /// Processors fitted.
    pub cpus: usize,
    /// The R16000 chip fitted.
    pub chip: &'static TezroChip,
    /// A VPro V12 at widget 12.
    pub v12: bool,
    /// Rendering threads and tile size for the V12.
    pub graphics_pool: (usize, u32),
    /// Which unit draws and, for the GPU, the wgpu backend (`[graphics]`).
    pub graphics_raster: (crate::config::RasterUnit, &'a str),
}

/// The Tezro's devices around its Bedrock node.
pub(crate) struct TezroBoard {
    pub bus: Arc<Ip35Bus>,
    pub bedrock: Arc<Bedrock>,
    pub l1: Arc<L1>,
    pub flash: Arc<Flash>,
    /// Cacheable memory at plain physical addresses: what the caches fill
    /// from.
    cached: Arc<dyn BusDevice>,
    /// What the JIT compiles code through: `cached`, and the reset PROM's
    /// uncached-fetch window ([`Ip35Bus::code_view`]).
    #[cfg(feature = "jitv2")]
    code: Arc<dyn BusDevice>,
    /// The node's resets and power events, and their receiver until the
    /// system controller takes it.
    events: Sender<MachineEvent>,
    receiver: Mutex<Option<Receiver<MachineEvent>>>,
    /// The slots whose processors run: every fitted one.
    pub processors: Population,
    /// The holds on the PROM's pair liveness check and presence publisher
    /// across them.
    pub presence: Arc<PresenceHold>,
    /// The PICs in QEMU's host order, which numbers their PI level sources:
    /// widget 11's first when the board has it, then the IO9's at widget 15.
    pub pics: Vec<Arc<Pic>>,
    pub ioc4: Arc<Ioc4>,
    pub scsi: Arc<Isp12160>,
    pub eth: Arc<Bcm5701>,
    pub timekeeper: Arc<Ds1744>,
    pub serial: [Arc<CiSerialBackend>; SERIAL_PORTS],
    /// The R16000 chip fitted, whose flash record the node holds.
    pub chip: &'static TezroChip,
    /// The VPro V12, when the board has one.
    pub v12: Option<Arc<Odyssey>>,
    /// The status bar's activity bits and the CPU's Compare matches, which
    /// the graphics board shows.
    pub heartbeat: Arc<AtomicU64>,
    pub fasttick: Arc<AtomicU64>,
    /// What the graphics board's status bar shows: the activity bits, the
    /// Compare matches and the CPU's cycle count and Count rate.
    pub status: Arc<crate::disp::StatusSource>,
    clock: Clock,
    last_clock: AtomicU64,
    running: AtomicBool,
}

impl TezroBoard {
    /// The board's devices that take monitor commands: the physical map, the
    /// L1, the SCSI controller, the keyboard and mouse (`ps2 type` and
    /// friends) and, when one is fitted, the VPro (`vpro ...`).
    pub fn monitor_devices(&self) -> Vec<Arc<dyn Device>> {
        let mut devices: Vec<Arc<dyn Device>> = vec![
            self.bedrock.physical().clone(),
            self.l1.clone(),
            self.scsi.clone(),
            self.ioc4.ps2(),
        ];
        if let Some(v12) = &self.v12 {
            devices.push(v12.clone());
        }
        devices
    }

    /// Flush the disks on both channels of the storage controller.
    pub fn flush_disks(&self) -> std::io::Result<()> {
        (0..2).try_for_each(|channel| self.scsi.bus_for_channel(channel).unwrap().flush())
    }

    #[cfg(test)]
    pub fn new(cpus: usize) -> Result<Self, String> {
        Self::with_config(TezroBoardConfig {
            payload: &[],
            clock: Arc::new(|| 0),
            flash_path: None,
            station_mac: L1Config::default().mac,
            system_serial: None,
            timekeeper: Arc::new(Ds1744::new(YearBase::From(1968))?),
            memory: &MemoryPopulation::default(),
            cpus,
            chip: &CHIPS[0],
            v12: false,
            graphics_pool: (2, 32),
            graphics_raster: (crate::config::RasterUnit::Cpu, "auto"),
        })
    }

    pub fn with_config(config: TezroBoardConfig<'_>) -> Result<Self, String> {
        let TezroBoardConfig { payload, clock, flash_path, station_mac, system_serial, timekeeper, memory, cpus, chip, v12, graphics_pool, graphics_raster } = config;
        if ![1, 2, 4].contains(&cpus) {
            return Err(format!("{cpus} processors; a Tezro board fits 1, 2 or 4"));
        }
        chip.accepts_cpus(cpus as u8)?;
        memory.check_for_cpus(cpus)?;
        let running = Population::fitted_cpus(cpus)?;
        let node = Node::build(NodeConfig {
            name: "Tezro",
            payload,
            clock: clock.clone(),
            flash_path,
            banks_mib: &memory.banks_mib(),
            processors: running,
            firmware: chip.firmware,
            board_id: BOARD_ID,
            l1: L1Config {
                type_code: 0x39,
                revision: Some([1, 48, 1]),
                mac: station_mac,
                system_serial,
                port_speed_first: 1,
                router: true,
                cpu_context: Some(cpu_context(&chip.firmware)),
                node_board: Some(node_board(chip, cpus)),
                ..Default::default()
            },
        })?;
        let Node {
            bus,
            bedrock,
            l1,
            flash,
            cached,
            #[cfg(feature = "jitv2")]
            code,
            events,
            receiver,
        } = node;
        let serial: [Arc<CiSerialBackend>; SERIAL_PORTS] = std::array::from_fn(|_| Arc::new(CiSerialBackend::new()));
        l1.set_serial_backend(serial[0].clone());

        let io9 = Pic::new(IO9_WIDGET);
        let pxbow = io9.enable_pxbow(HOST_WIDGET, clock.clone()).map_err(|e| format!("Tezro PXBOW: {e:?}"))?;
        bedrock.connect_widget(0, pxbow.widget_zero());
        bedrock.connect_widget(IO9_WIDGET, pxbow.route(IO9_WIDGET));
        let presence = PresenceHold::new(running, clock.clone(), &bedrock);
        let heartbeat = Arc::new(AtomicU64::new(0));
        let fasttick = Arc::new(AtomicU64::new(0));
        let status = crate::disp::StatusSource::new(heartbeat.clone(), fasttick.clone());
        let v12 = v12.then(|| fit_v12(&bedrock, &pxbow, &status, graphics_pool, graphics_raster)).transpose()?;
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
            pic.connect_peer(Arc::new(BedrockPeer(Arc::downgrade(&bedrock))))
                .map_err(|_| "Tezro: PIC peer connected twice".to_string())?;
        }

        let ioc4 = Arc::new(Ioc4::new(Ioc4Config::tezro()));
        ioc4.attach_clock(timekeeper.clone());
        for (port, backend) in serial[1..].iter().enumerate() {
            ioc4.set_serial_backend(port, backend.clone());
        }
        let scsi = Arc::new(Isp12160::new(Isp12160::new_bus(), Isp12160::new_bus()));
        scsi.set_heartbeat(heartbeat.clone());
        // The Ethernet slot drives REQ64#, which the chip samples as reset ends.
        let eth = Bcm5701::new(Bcm5701Config { req64: true, ..Bcm5701Config::new(l1.station_mac()) })?;
        io9.attach(0, IOC4_DEVICE, ioc4.clone(), Function { self_ordered_pio: true, req64: false });
        let ti = PciToPciBridge::with_revision(
            io9.bus(0),
            DevFn { device: TI_BRIDGE_DEVICE, function: 0 },
            TI_BRIDGE_VENDOR,
            TI_BRIDGE_ID,
            TI_BRIDGE_REVISION,
        );
        io9.attach(0, TI_BRIDGE_DEVICE, ti, Function::default());
        io9.attach(0, SCSI_DEVICE, scsi.clone(), Function::default());
        io9.attach(0, ETH_DEVICE, eth.clone(), Function { self_ordered_pio: false, req64: true });
        Ok(Self {
            bus,
            bedrock,
            l1,
            flash,
            cached,
            #[cfg(feature = "jitv2")]
            code,
            events,
            receiver: Mutex::new(Some(receiver)),
            processors: running,
            presence,
            pics,
            ioc4,
            scsi,
            eth,
            timekeeper,
            serial,
            chip,
            v12,
            heartbeat,
            fasttick,
            status,
            last_clock: AtomicU64::new(clock()),
            clock,
            running: AtomicBool::new(false),
        })
    }

    pub fn cached_bus(&self) -> Arc<dyn BusDevice> {
        self.cached.clone()
    }

    /// The bus the JIT compiles code through; see [`Ip35Bus::code_view`].
    #[cfg(feature = "jitv2")]
    pub fn code_bus(&self) -> Arc<dyn BusDevice> {
        self.code.clone()
    }

    /// Everything back to its power-on state.
    pub fn power_on(&self) {
        self.presence.reset();
        self.bedrock.power_on();
        self.l1.power_on();
        self.flash.power_on();
        self.ioc4.power_on();
        self.scsi.power_on();
        self.eth.power_on();
        for pic in &self.pics {
            pic.power_on();
        }
        if let Some(v12) = &self.v12 {
            v12.power_on();
        }
    }

    pub fn start(&self) {
        if let Some(v12) = &self.v12 {
            v12.start();
        }
        self.last_clock.store((self.clock)(), Ordering::Release);
        self.running.store(true, Ordering::Release);
        self.bedrock.start();
        self.l1.start();
        self.ioc4.start();
        self.eth.start();
        self.scsi.start();
    }

    /// The boot flash's command timing, on CPU 0's batch boundary.
    pub fn poll(&self) {
        if self.running.load(Ordering::Acquire) {
            let now = (self.clock)();
            let previous = self.last_clock.swap(now, Ordering::AcqRel);
            self.flash.advance_ns(now.saturating_sub(previous));
        }
    }

    pub fn stop(&self) {
        self.poll();
        if let Some(v12) = &self.v12 {
            v12.stop();
        }
        self.running.store(false, Ordering::Release);
        self.scsi.stop();
        self.eth.stop();
        self.ioc4.stop();
        self.l1.stop();
        self.bedrock.stop();
    }
}

impl Drop for TezroBoard {
    fn drop(&mut self) {
        self.stop();
    }
}

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
    let dma = bedrock.widget_dma(V12_WIDGET);
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

    /// The legs of a transfer: each the largest naturally aligned size of
    /// eight bytes or fewer that fits, so word accesses keep their width.
    fn legs(address: u64, len: usize) -> impl Iterator<Item = (usize, usize)> {
        let mut offset = 0;
        std::iter::from_fn(move || {
            if offset == len {
                return None;
            }
            let at = address + offset as u64;
            let size = [8usize, 4, 2, 1]
                .into_iter()
                .find(|&n| at.is_multiple_of(n as u64) && n <= len - offset)
                .unwrap_or(1);
            let leg = (offset, size);
            offset += size;
            Some(leg)
        })
    }
}

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
            let value = io.inbound_read(source_widget, Some(INBOUND_DEVICE), address + offset as u64, size as u8);
            if value.status != BUS_OK {
                return Err(Refused);
            }
            data[offset..offset + size].copy_from_slice(&value.data.to_be_bytes()[8 - size..]);
        }
        Ok(())
    }

    fn dma_write(&self, source_widget: u8, widget: u8, address: u64, data: &[u8]) -> Result<(), Refused> {
        Self::host(widget)?;
        let io = self.io()?;
        for (offset, size) in Self::legs(address, data.len()) {
            let value = data[offset..offset + size].iter().fold(0u64, |v, b| (v << 8) | u64::from(*b));
            if io.inbound_write(source_widget, Some(INBOUND_DEVICE), address + offset as u64, size as u8, value) != BUS_OK {
                return Err(Refused);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
