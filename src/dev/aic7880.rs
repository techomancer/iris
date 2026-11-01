//! Adaptec AIC-7880, emulated above its sequencer.
//!
//! The chip has almost no fixed behaviour. A driver downloads a sequencer
//! program into it and starts it, and everything the controller then does is
//! that program running — so there is no register-level contract to implement
//! in the way there is for, say, a UART.
//!
//! This models the level *above* that: the driver fills a SCB, hands its index
//! to the queue-in FIFO, and expects the index to come back on the queue-out
//! FIFO with status in the SCB and `CMDCMPLT` raised. We do the SCSI work
//! ourselves and post the result, and the downloaded sequencer program is
//! stored and never executed.
//!
//! The trade is explicit: this is a contract with what drivers *observe*
//! rather than with the hardware, so it has to be checked against each one.
//! The SCB layout in particular is defined by whichever sequencer program was
//! downloaded, not by the silicon, so it is discovered by watching what the
//! driver writes before it queues — see `clanker/devices/ip32-o2-bringup-log.md`.
//!
//! Three contracts are known. SGI's PROM and the IRIX kernel both carry
//! Adaptec's "HIM" and its sequencer program, which hand the chip a tag
//! through `QINFIFO` and take completions from a byte array in host memory;
//! errors come back as pausing interrupts the HIM then sorts out itself.
//! NetBSD and OpenBSD ring a doorbell in scratch RAM instead. Which one is
//! in force is read from scratch RAM at each doorbell, never latched,
//! because one boot can hand the chip from one driver to the next.

use crate::locks::{SpinGuard, SpinLock};
use crate::scsi::{scsi_cmd, ScsiBus, ScsiDataLength, ScsiIdentities, ScsiRequest, CDROM_IDENTITY, DISK_IDENTITY};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::snapshot::{get_field, hex_u16, hex_u32, hex_u64, hex_u8, load_u8_slice, toml_u16, toml_u32, toml_u64, toml_u8, u8_slice_to_toml};

/// Registers, at the offsets the chip presents them. The bridge's byte-lane
/// swap is applied before we get here, so these are device offsets.
pub mod reg {
    pub const SCSISEQ: u32 = 0x00;
    /// The bus's control lines. Bit 1 is REQ.
    pub const SCSISIG: u32 = 0x03;
    pub const SCSIID: u32 = 0x05;
    /// Status that latches until cleared: reads are SSTAT0/SSTAT1, writes
    /// to the same addresses are CLRSINT0/CLRSINT1.
    pub const SSTAT0: u32 = 0x0b;
    pub const SSTAT1: u32 = 0x0c;
    pub const SIMODE0: u32 = 0x10;
    pub const SIMODE1: u32 = 0x11;
    pub const SELID: u32 = 0x19;
    pub const SBLKCTL: u32 = 0x1f;
    /// Sequencer scratch RAM, where the driver keeps its configuration.
    pub const SRAM_FIRST: u32 = 0x20;
    pub const SRAM_LAST: u32 = 0x5f;
    pub const SEQCTL: u32 = 0x60;
    pub const SEQRAM: u32 = 0x61;
    pub const SEQADDR0: u32 = 0x62;
    pub const SEQADDR1: u32 = 0x63;
    pub const HCNTRL: u32 = 0x87;
    pub const HADDR: u32 = 0x88;
    pub const HCNT: u32 = 0x8c;
    pub const SCBPTR: u32 = 0x90;
    pub const INTSTAT: u32 = 0x91;
    /// Reads as the sequencer error register, written to clear interrupts.
    pub const ERROR_CLRINT: u32 = 0x92;
    pub const DFCNTRL: u32 = 0x93;
    pub const DFSTATUS: u32 = 0x94;
    pub const SCBCNT: u32 = 0x9a;
    pub const QINFIFO: u32 = 0x9b;
    pub const QINCNT: u32 = 0x9c;
    pub const QOUTFIFO: u32 = 0x9d;
    pub const QOUTCNT: u32 = 0x9e;
    /// The window onto the selected SCB.
    pub const SCBARRAY_FIRST: u32 = 0xa0;
    pub const SCBARRAY_LAST: u32 = 0xbf;
}

pub mod hcntrl {
    /// Reset the chip. Reads back as an acknowledgement.
    pub const CHIPRST: u8 = 0x01;
    pub const INTEN: u8 = 0x02;
    /// Stop the sequencer. The driver sets this before touching shared state.
    pub const PAUSE: u8 = 0x04;
    pub const IRQMS: u8 = 0x08;
    pub const SWINT: u8 = 0x10;
    pub const POWRDN: u8 = 0x40;
}

pub mod intstat {
    pub const SEQINT: u8 = 0x01;
    /// A command finished and its index is on the queue-out FIFO.
    pub const CMDCMPLT: u8 = 0x02;
    pub const SCSIINT: u8 = 0x04;
    pub const BRKADRINT: u8 = 0x08;
    /// The interrupts that stop the sequencer until the host restarts it.
    /// A completion is the one that does not.
    pub const PAUSING: u8 = SEQINT | SCSIINT | BRKADRINT;
    /// What a `SEQINT` is about. Only meaningful while `SEQINT` is set.
    pub const INTCODE: u8 = 0xf0;
}

pub mod sstat1 {
    /// Selection timed out: nobody answered at that id.
    pub const SELTO: u8 = 0x80;
}

/// Bus control lines, as `SCSISIG` reads them.
pub mod scsisig {
    /// The target is asking for a byte. Nothing here ever does, and the HIM
    /// waits for this to drop before it carries on after an error.
    pub const REQ: u8 = 0x02;
}

pub mod seqctl {
    /// Writes to SEQRAM load the sequencer program.
    pub const LOADRAM: u8 = 0x01;
    pub const SEQRESET: u8 = 0x02;
    pub const STEP: u8 = 0x04;
}

/// Scratch RAM the downloaded program uses to find its work. These offsets
/// belong to that program, not to the chip, and were read out of the driver:
/// see `clanker/devices/ip32-o2-bringup-log.md`.
pub mod sram {
    /// Bus address of an array of SCB bus addresses, indexed by tag.
    pub const SCB_ARRAY: u32 = 0x3d;
    /// Bus address of the queue-out array in host memory. Under the HIM its
    /// low byte is also the sequencer's write cursor into that array.
    pub const QOUTFIFO: u32 = 0x56;
}

/// The HIM's side of scratch RAM, shared by SGI's PROM and the IRIX kernel.
pub mod him {
    /// Two placeholder pointers the HIM writes and neither of its sequencer
    /// programs ever reads, little end first. Nothing else puts these
    /// values there, so finding both is how we know the HIM is driving.
    pub const SIGNATURE: [(u32, u32); 2] = [(0x42, 0x0bad_0001), (0x5a, 0x0bad_0003)];
    /// The tag being selected, which is the one a selection timeout is for.
    pub const WAITING_SCB: u32 = 0x3a;
    /// The tag connected, which is the one a check condition is for.
    pub const ACTIVE_SCB: u32 = 0x3b;
    /// The SCSI status a check-condition interrupt hands over.
    pub const STATUS: u32 = 0x3c;
    /// `SEQINT` code: the target ended with a status other than GOOD.
    pub const BAD_STATUS: u8 = 0x50;
    /// Length of the PROM's build of the sequencer program. The kernel's is
    /// 1708 bytes. The PROM boots disks and CD-ROMs here with every error
    /// answered as success, so it keeps those answers, and the selection
    /// timeout and check-condition interrupts go only to other programs.
    /// That holds until a PROM boot has been seen to take them.
    pub const PROM_PROGRAM_LEN: usize = 1696;
}

/// SCSI block size. Everything SGI boots from uses this.
pub const BLOCK_SIZE: usize = 512;

/// A scatter-gather entry: a bus address and a length. The top byte of the
/// length carries flags, so it is masked off.
pub const SG_LEN_MASK: u32 = 0x00ff_ffff;
/// The one flag in that byte we use: this entry is the last.
pub const SG_LAST: u32 = 0x8000_0000;
/// How many entries to follow before deciding the list is not a list.
///
/// This was 32, which is a plausible-looking number and was wrong. A single
/// 264-block read scatters into 33 pages, so the walk stopped one entry short
/// and dropped the tail — silently, because the command still reported
/// success. The symptom was a 5456-byte hole in a loaded image, which
/// happened to contain the entry point: the PROM loaded `sashARCS`, printed
/// the right sizes and the right entry, jumped, and executed whatever had
/// been in that memory before.
///
/// The bound only exists so a corrupt list cannot spin forever. A transfer
/// ends when the data runs out or an entry is null, both of which come first
/// in any healthy list.
pub const SG_MAX: usize = 4096;


/// NetBSD's sequencer program uses a different arrangement from the PROM's,
/// and these names come from its own headers rather than from guessing.
///
/// Its host does not write a tag to `QINFIFO` at all. It puts the tag into a
/// 256-entry ring in host memory and then writes the *producer index* to a
/// scratch register; the sequencer consumes from the ring and fetches the SCB
/// from a flat array indexed by tag. Completion goes back through a second
/// ring beside the first.
pub mod nb {
    /// Scratch registers holding the two base addresses and the index.
    pub const HSCB_ADDR: u32 = 0x44;
    pub const SHARED_DATA_ADDR: u32 = 0x48;
    pub const KERNEL_QINPOS: u32 = 0x4c;

    /// `qoutfifo` sits at the shared-data base and `qinfifo` right after it,
    /// 256 bytes each.
    pub const FIFO_LEN: u32 = 256;
    /// Stride of the host's SCB array. The chip only looks at the first 32
    /// bytes; the rest is room for a long CDB.
    pub const HSCB_STRIDE: u32 = 64;

    /// Offsets inside one SCB.
    pub const CDB: u32 = 0;
    /// Status is returned in the same union the CDB arrived in, after two
    /// residual words.
    pub const STATUS: u32 = 8;
    pub const DATAPTR: u32 = 12;
    pub const DATACNT: u32 = 16;
    pub const SGPTR: u32 = 20;
    pub const CONTROL: u32 = 24;
    pub const SCSIID: u32 = 25;
    pub const LUN: u32 = 26;
    pub const CDB_LEN: u32 = 28;

    /// A scatter-gather entry's length carries flags in its top byte.
    pub const LAST_SEG: u32 = 0x8000_0000;
    pub const SG_PTR_MASK: u32 = 0xffff_fff8;
}

/// The host queue-out FIFO is 256 entries, filled with this until used.
pub const QOUTFIFO_LEN: u32 = 256;
pub const QOUTFIFO_EMPTY: u8 = 0xff;

/// Offsets within the 32-byte SCB, as host memory holds it. The HIM builds it
/// with 32-bit stores, so each word's bytes are the chip's in reverse.
pub mod scb {
    /// How many scatter-gather entries follow. Never zero: a command with
    /// no data still lists one empty entry.
    pub const SEG_COUNT: usize = 0;
    /// The length of the CDB, in bytes.
    ///
    /// This was first read as a target/lun byte, on the strength of the
    /// aic7xxx register named `SCB_TCL` sitting at the same offset. It is
    /// not: it holds `0x06` for INQUIRY, TEST UNIT READY, MODE SENSE and
    /// START STOP UNIT — every six-byte command — and `0x0a` for READ(10).
    /// Header names again, observed behaviour again, and the behaviour wins.
    pub const CDB_LEN: usize = 1;
    /// Control flags: disconnect allowed, tagged, and the like. Belongs to
    /// the driver; nothing here writes it.
    pub const CONTROL: usize = 2;
    /// Bus address of the scatter-gather list.
    pub const SG_PTR: usize = 4;
    /// Bus address of the CDB.
    pub const CMD_PTR: usize = 8;
    /// Target in bits 7..4, channel in bit 3, LUN in bits 2..0 -- the same
    /// shape as aic7xxx's `SCB_TCL`.
    ///
    /// Read as `tag << 4` at first, which fitted perfectly while the PROM was
    /// scanning targets 1, 2 and 3 with tags 1, 2 and 3. Two disks at
    /// different ids in one session separate them: asking for `dksc(0,1,8)`
    /// and then `dksc(0,4,8)` produces `0x10` and `0x40` with the same tag.
    pub const TCL: usize = 3;
    /// The target's SCSI status. The host writes zero and the HIM fills it
    /// in itself after a check-condition interrupt; the sequencer never
    /// writes the SCB back. Byte 2 was taken for this once, because the
    /// driver leaves 0x40 there, but that is "disconnect allowed".
    pub const TARGET_STATUS: usize = 15;
}

/// How many SCBs the chip has, and how big the window onto one is.
pub const SCB_COUNT: usize = 16;
pub const SCB_SIZE: usize = (reg::SCBARRAY_LAST - reg::SCBARRAY_FIRST + 1) as usize;

/// A command the sequencer has taken off a queue, with what it read from
/// the chip to run it.
enum Job {
    /// A tag from `QINFIFO` (the HIM, and drivers without a doorbell):
    /// the SCB array's address, whether errors go back as pausing
    /// interrupts, and the chip's own SCSI id.
    Tag { tag: u8, array: u32, errors: bool, own: u8 },
    /// A tag from NetBSD's queue-in ring.
    NetBsd { tag: u8, hscb_base: u32, shared: u32 },
}

/// What a finished command leaves for the sequencer to tell the host.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Done {
    /// Nobody answered at the id: `SCSIINT` with `SELTO`, the sequencer stopped.
    SelectionTimeout { tag: u8 },
    /// The target ended with this status: `SEQINT` for the HIM to collect.
    BadStatus { tag: u8, status: u8 },
    /// A tag for the queue-out, the HIM's way or the plain one.
    Complete { tag: u8, target: u8 },
    /// A tag for NetBSD's queue-out ring at `shared`.
    NetBsd { tag: u8, shared: u32 },
}

impl Done {
    fn to_toml(self) -> toml::Value {
        let (kind, tag, arg, ring) = match self {
            Done::SelectionTimeout { tag } => ("seltimeout", tag, 0, 0),
            Done::BadStatus { tag, status } => ("badstatus", tag, status, 0),
            Done::Complete { tag, target } => ("complete", tag, target, 0),
            Done::NetBsd { tag, shared } => ("netbsd", tag, 0, shared),
        };
        let mut t = toml::map::Map::new();
        t.insert("kind".into(), toml::Value::String(kind.into()));
        t.insert("tag".into(), hex_u8(tag));
        t.insert("arg".into(), hex_u8(arg));
        t.insert("ring".into(), hex_u32(ring));
        toml::Value::Table(t)
    }

    fn from_toml(v: &toml::Value) -> Option<Self> {
        let b = |k: &str| get_field(v, k).and_then(toml_u8).unwrap_or(0);
        let (tag, arg) = (b("tag"), b("arg"));
        Some(match get_field(v, "kind")?.as_str()? {
            "seltimeout" => Done::SelectionTimeout { tag },
            "badstatus" => Done::BadStatus { tag, status: arg },
            "complete" => Done::Complete { tag, target: arg },
            "netbsd" => Done::NetBsd { tag, shared: get_field(v, "ring").and_then(toml_u32).unwrap_or(0) },
            _ => return None,
        })
    }
}

/// What running one command produced, gathered without the state lock and
/// applied under it.
#[derive(Default)]
struct Outcome {
    notes: Vec<String>,
    cmds: u64,
    bytes_in: u64,
    bytes_out: u64,
    executed: Option<(u8, u32, Vec<u8>)>,
    done: Option<Done>,
}

/// Everything the guest can see, plus a record of what it did.
struct State {
    regs: [u8; 0x100],
    scb: [[u8; SCB_SIZE]; SCB_COUNT],
    seqram: Vec<u8>,
    seqaddr: u16,
    /// Tags waiting for the sequencer. It takes them only while running, so
    /// a tag written while paused stays here, where the host can still
    /// count, pop and rotate it.
    qin: VecDeque<u8>,
    qout: VecDeque<u8>,
    intstat: u8,
    /// SCSI status that latches until the host clears it, as SSTAT0 and
    /// SSTAT1 read. Kept apart from `regs` because the same addresses
    /// written are CLRSINT0 and CLRSINT1: storing the write there made a
    /// channel reset's "clear everything" read back as "everything
    /// happened", external bus reset included.
    sstat0: u8,
    sstat1: u8,
    /// SCBs handed to the queue-in FIFO, with their contents as queued. This
    /// is the evidence for what the SCB layout actually is.
    queued: Vec<(u8, [u8; SCB_SIZE])>,
    /// Commands actually fetched and run: tag, SCB address, CDB.
    executed: Vec<(u8, u32, Vec<u8>)>,
    /// Where the next completion goes in the host queue-out FIFO.
    qoutpos: u32,
    /// How far we have consumed NetBSD's queue-in ring.
    qinpos: u8,
    /// NetBSD has rung: its ring may hold tags past `qinpos`. Decided when
    /// `KERNEL_QINPOS` is written, acted on once the sequencer runs (the
    /// driver writes it with the sequencer paused).
    nb_doorbell: bool,
    /// Finished commands that found the sequencer paused, posted in order
    /// once it runs again.
    held: VecDeque<Done>,
    /// Bumped by a chip reset, so a command that was running across one is
    /// not posted afterwards.
    generation: u64,
    /// How many times the sequencer was paused or restarted.
    pauses: u64,
    /// Per SCB: filled in through the register window since last queued.
    scb_onchip: [bool; SCB_COUNT],
    /// Totals that survive a chip reset, because the drivers reset often and
    /// per-reset counts answer the wrong question. "Was the whole file
    /// transferred" needs a number for the whole session.
    total_cmds: u64,
    total_in: u64,
    total_out: u64,
    notes: Vec<String>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            regs: [0; 0x100],
            scb: [[0; SCB_SIZE]; SCB_COUNT],
            seqram: Vec::new(),
            seqaddr: 0,
            qin: VecDeque::new(),
            qout: VecDeque::new(),
            intstat: 0,
            sstat0: 0,
            sstat1: 0,
            queued: Vec::new(),
            executed: Vec::new(),
            qoutpos: 0,
            qinpos: 0,
            nb_doorbell: false,
            held: VecDeque::new(),
            generation: 0,
            pauses: 0,
            scb_onchip: [false; SCB_COUNT],
            total_cmds: 0,
            total_in: 0,
            total_out: 0,
            notes: Vec::new(),
        }
    }
}

pub struct Aic7880 {
    /// The chip's state. The CPU takes it for every register access and the
    /// worker only to take a command and to post its result, never across
    /// a transfer, so the sections are short and a spin lock fits.
    st: SpinLock<State>,
    /// Held by whoever is running a command, so a command run inline (no
    /// worker) can never overlap one on the worker while it is stopping.
    exec: Mutex<()>,
    /// The sequencer's thread, and whether it should be running. Commands
    /// run there, off the CPU's thread; see [`Self::service_one`].
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    running: AtomicBool,
    /// This controller, for the worker to hold.
    me: Weak<Self>,
    /// The targets on the bus, by SCSI id (the shared SCSI devices: disks,
    /// CHDs, overlays, CD-ROMs). An install needs at least two -- the media
    /// to read and the disk to write.
    bus: ScsiBus,
    /// Host memory, through the PCI bridge the chip masters. The controller
    /// fetches its work from there rather than being handed it a register at
    /// a time, so without this it can see the tag of a command and nothing
    /// else about it.
    dma: std::sync::OnceLock<std::sync::Arc<dyn crate::pci::PciDma>>,
    /// The SCSI id a disk attached without one answers at.
    target: u8,
    /// The status bar's activity bits, shared with the display thread.
    heartbeat: std::sync::OnceLock<Arc<AtomicU64>>,
    /// Which of the O2's two controllers this is (0 internal, 1 external):
    /// the monitor's `scsi`/`cow` address the first, `scsi1`/`cow1` the second.
    index: u8,
    /// The controller's own sense, as (key, ASC, ASCQ): a command for a
    /// logical unit the target does not have, handed out by the REQUEST
    /// SENSE that follows. A check condition followed by "no sense" tells a
    /// driver nothing went wrong. Everything else a target answers itself.
    sense: Mutex<Option<(u8, u8, u8)>>,
    /// Pages of DMA addresses already reported as not memory.
    dma_misses: Mutex<std::collections::BTreeSet<u32>>,
    /// The chip's interrupt pin, as the board routes it.
    irq: std::sync::OnceLock<crate::irq::IrqLine>,
    pci: crate::pci::PciConfig,
}

/// The chip's PCI identity: Adaptec, AIC-7880.
pub const PCI_VENDOR_ADAPTEC: u16 = 0x9004;
pub const PCI_DEVICE_AIC7880: u16 = 0x8078;
/// Mass storage controller, SCSI, no specific programming interface.
pub const PCI_CLASS_SCSI: u32 = 0x0100_00;

impl Aic7880 {
    /// An empty bus of the kind this chip drives: ids 0-15, and the
    /// emulated disk and CD-ROM. The machine fills it from `[scsi.N]` and
    /// hands it to [`Self::new_at`].
    pub fn new_bus() -> ScsiBus {
        ScsiBus::new(16, ScsiIdentities { disk: DISK_IDENTITY, cdrom: CDROM_IDENTITY })
    }

    /// The controller on an empty bus, for the tests.
    #[cfg(test)]
    pub fn new() -> Arc<Self> {
        Self::new_at(0, Self::new_bus())
    }

    /// The controller at `index` on `bus`: 0 on the internal bus, 1 on the
    /// external connector. Commands run inline until
    /// [`crate::traits::Device::start`] gives it its own thread.
    pub fn new_at(index: u8, bus: ScsiBus) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            index,
            st: SpinLock::new(State::default()),
            exec: Mutex::new(()),
            worker: Mutex::new(None),
            running: AtomicBool::new(false),
            me: me.clone(),
            bus,
            dma: std::sync::OnceLock::new(),
            sense: Mutex::new(None),
            dma_misses: Mutex::new(std::collections::BTreeSet::new()),
            irq: std::sync::OnceLock::new(),
            // 256 bytes of I/O and 4 KiB of memory, both onto the same
            // registers: what the part has.
            pci: {
                use crate::pci::{Bar, PciConfig, PCI_INTA};
                PciConfig::new(PCI_VENDOR_ADAPTEC, PCI_DEVICE_AIC7880, PCI_CLASS_SCSI, 0x01,
                    [Bar::Io(0x100), Bar::Mem(0x1000), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused], PCI_INTA)
            },
            target: 1,
            heartbeat: std::sync::OnceLock::new(),
        })
    }

    /// Light the status bar's SCSI activity bits in `heartbeat`.
    pub fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        let _ = self.heartbeat.set(heartbeat);
    }

    /// `n` bytes of host memory from bus address `addr`, through the bridge.
    /// Its two views reach the same bytes, and the driver uses both in the
    /// same breath: the SCB's own address is given as `0x410752fc` while the
    /// CDB pointer inside it is `0x0107532c`.
    fn dma_read_bytes(&self, addr: u32, n: usize) -> Vec<u8> {
        let mut b = vec![0u8; n];
        if !self.dma.get().is_some_and(|d| d.dma_read(addr as u64, &mut b).is_ok()) {
            self.dma_miss(addr, false);
        }
        b
    }

    fn dma_read8(&self, addr: u32) -> u8 {
        self.dma_read_bytes(addr, 1)[0]
    }

    fn dma_write_bytes(&self, addr: u32, data: &[u8]) {
        if !self.dma.get().is_some_and(|d| d.dma_write(addr as u64, data).is_ok()) {
            self.dma_miss(addr, true);
        }
    }

    /// A DMA address that is not memory: report each 4 KB page once (the
    /// first 64), since a driver handing the chip bus addresses we decode
    /// differently loses data silently otherwise.
    fn dma_miss(&self, addr: u32, write: bool) {
        let mut seen = self.dma_misses.lock().unwrap();
        if seen.len() < 64 && seen.insert(addr & !0xfff) {
            eprintln!("aic7880: DMA {} {addr:#010x} is not memory", if write { "write to" } else { "read from" });
        }
    }

    fn dma_write8(&self, addr: u32, val: u8) {
        self.dma_write_bytes(addr, &[val]);
    }


    /// A 32-bit word, in the order the host wrote it.
    ///
    /// The driver builds these with ordinary big-endian stores, and reading
    /// them any other way turns `0x410752fc` into `0xfc520741` and every
    /// pointer into nonsense. The chip's own scratch RAM is the opposite —
    /// written a register at a time, least significant first — so the two
    /// orders sit side by side and both are settled by observation.
    fn dma_read32(&self, addr: u32) -> u32 {
        let b = self.dma_read_bytes(addr, 4);
        u32::from_be_bytes([b[0], b[1], b[2], b[3]])
    }

    /// A 32-bit word the little-endian way.
    ///
    /// The chip is little-endian and NetBSD's driver writes its SCB fields
    /// accordingly (`ahc_htole32`), while the PROM's sequencer program was
    /// handed big-endian words by its own loader. Reading NetBSD's pointers
    /// the wrong way round turns 0x0005fe58 into 0x58fe0500 -- an address
    /// far outside memory, which lands nowhere and reports success.
    fn dma_read32_le(&self, addr: u32) -> u32 {
        let b = self.dma_read_bytes(addr, 4);
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    fn note(st: &mut State, what: String) {
        // The most recent stretch, not the first: a question about a
        // running guest is about what it is doing now.
        if st.notes.len() >= 8192 {
            st.notes.drain(..1024);
        }
        st.notes.push(what);
    }

    /// A running commentary on what the driver asked for.
    pub fn notes(&self) -> Vec<String> {
        self.st.lock().notes.clone()
    }

    /// Commands, bytes read and bytes written for the whole session.
    pub fn totals(&self) -> (u64, u64, u64) {
        let st = self.st.lock();
        (st.total_cmds, st.total_in, st.total_out)
    }

    /// True while the chip is asserting its interrupt line. Only the four
    /// interrupt bits count: the code bits above them say what a `SEQINT`
    /// is about and are not an interrupt of their own.
    pub fn interrupting(&self) -> bool {
        self.st.lock().intstat & 0x0f != 0
    }

    /// Wire the chip's interrupt pin to `line`.
    pub fn connect_irq(&self, line: crate::irq::IrqLine) {
        let _ = self.irq.set(line);
        self.update_irq();
    }

    /// Drive the interrupt pin to the chip's state.
    fn update_irq(&self) {
        self.drive_irq(&self.st.lock());
    }

    /// Drive the pin from `st`, with the state locked. The worker posts
    /// completions and the CPU clears them, so the pin is set under the
    /// same lock as the change; set after releasing it, a completion's
    /// "high" could land after the host's clear and stick.
    fn drive_irq(&self, st: &State) {
        if let Some(l) = self.irq.get() {
            l.set(st.intstat & 0x0f != 0);
        }
    }

    /// How many times the driver paused or restarted the sequencer.
    pub fn pauses(&self) -> u64 {
        self.st.lock().pauses
    }

    /// The sequencer program the driver downloaded. Never executed; its
    /// length is how we know the download happened at all.
    pub fn seqram_len(&self) -> usize {
        self.st.lock().seqram.len()
    }

    /// Every SCB handed to the queue-in FIFO, as it looked when queued.
    /// Commands the controller fetched and ran.
    pub fn executed(&self) -> Vec<(u8, u32, Vec<u8>)> {
        self.st.lock().executed.clone()
    }

    pub fn queued(&self) -> Vec<(u8, [u8; SCB_SIZE])> {
        self.st.lock().queued.clone()
    }

    /// A 32-bit value from the sequencer's scratch RAM, which the driver
    /// fills a register at a time, least significant byte first.
    fn scratch_le32(st: &State, off: u32) -> u32 {
        let b = |i: u32| st.regs[(off + i) as usize] as u32;
        b(0) | (b(1) << 8) | (b(2) << 16) | (b(3) << 24)
    }

    /// Whether the HIM is driving: its placeholder pointers are where it
    /// leaves them. Read fresh each time, because a NetBSD boot after the
    /// PROM rewrites the same scratch bytes and must fall out of it.
    fn him(st: &State) -> bool {
        him::SIGNATURE.iter().all(|(off, v)| Self::scratch_le32(st, *off) == *v)
    }

    /// Whether errors go back the way the HIM's own sequencer reports them:
    /// as pausing interrupts, with nothing on the queue-out. Everything but
    /// the PROM's build of the program; see `him::PROM_PROGRAM_LEN`.
    fn him_errors(st: &State) -> bool {
        Self::him(st) && st.seqram.len() != him::PROM_PROGRAM_LEN
    }

    /// Whether the sequencer is stopped. The host can stop it, and so does
    /// any interrupt that needs the host to look before it goes on.
    fn paused(st: &State) -> bool {
        st.regs[reg::HCNTRL as usize] & hcntrl::PAUSE != 0 || st.intstat & intstat::PAUSING != 0
    }

    /// Raise an interrupt that stops the sequencer. `HCNTRL` reads back as
    /// paused, as the real one does, since that is what the HIM polls.
    fn raise_pausing(st: &mut State, bits: u8) {
        st.intstat |= bits;
        st.regs[reg::HCNTRL as usize] |= hcntrl::PAUSE;
    }

    /// Whether an address is host memory. NetBSD's doorbell address lives in
    /// bytes the HIM also uses, so a doorbell whose rings are not memory is
    /// somebody else's bytes and not a doorbell.
    fn is_memory(&self, addr: u32) -> bool {
        self.dma.get().is_some_and(|d| d.dma_read(addr as u64, &mut [0u8]).is_ok())
    }

    /// A CDB's length follows from its opcode group. Taking it from the
    /// opcode rather than from a field in the SCB means one less piece of the
    /// layout has to be guessed, and it is how SCSI defines it anyway.
    fn cdb_len(op: u8) -> usize {
        match op >> 5 {
            0 => 6,
            1 | 2 => 10,
            4 => 16,
            5 => 12,
            _ => 6,
        }
    }

    /// The SCSI bus: attach targets to it and ask what is on it.
    pub fn bus(&self) -> &ScsiBus {
        &self.bus
    }

    fn has_target(&self, target: u8) -> bool {
        self.bus.has_target(target)
    }
}

/// The monitor's `scsi`/`cow` subcommands and the run-time disc changes
/// come from `scsi_monitor` through this.
impl crate::scsi_monitor::ScsiTargets for Aic7880 {
    fn scsi_bus(&self) -> &ScsiBus {
        &self.bus
    }
}

/// Controller 0 answers to `scsi`/`scsi0` and `cow`, controller 1 to `scsi1`
/// and `cow1`. Only `scsi regs` is this chip's own; the rest is
/// `scsi_monitor`'s, so the WD33C93A's `wdt`, `defer` and `debug` are not here.
impl crate::traits::Device for Aic7880 {
    fn step(&self, _cycles: u64) {}

    /// Stop the sequencer's thread. A command it is running finishes first,
    /// and is posted or held, so a snapshot taken after this sees no
    /// command half done. Commands then run inline until `start`.
    fn stop(&self) {
        let mut worker = self.worker.lock().unwrap();
        self.running.store(false, Ordering::Release);
        if let Some(w) = worker.take() {
            w.thread().unpark();
            let _ = w.join();
        }
    }

    /// Give the sequencer its own thread, so disk I/O no longer runs on the
    /// CPU's.
    fn start(&self) {
        let mut worker = self.worker.lock().unwrap();
        if self.running.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = self.me.clone();
        *worker = Some(
            std::thread::Builder::new()
                .name(format!("AIC-7880-{}", self.index))
                .spawn(move || Self::worker_loop(me))
                .expect("aic7880: worker thread"),
        );
    }

    fn is_running(&self) -> bool { self.running.load(Ordering::Acquire) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        let usage = crate::scsi_monitor::SCSI_USAGE;
        if self.index == 0 {
            vec![
                ("scsi".to_string(), format!("SCSI (Adaptec 7880, controller 0): {usage} | scsi regs")),
                ("scsi0".to_string(), "Alias for `scsi` (controller 0)".to_string()),
                ("cow".to_string(), "COW overlays on controller 0: cow status | cow commit [id] | cow reset [id]".to_string()),
            ]
        } else {
            vec![
                (format!("scsi{}", self.index), format!("SCSI controller {} (external): same subcommands as `scsi`", self.index)),
                (format!("cow{}", self.index), format!("COW overlays on controller {}: same subcommands as `cow`", self.index)),
            ]
        }
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        use std::io::Write as _;
        if cmd.starts_with("scsi") {
            if let Some(done) = crate::scsi_monitor::scsi_subcommand(self, args, &mut *w) {
                return done;
            }
            if args.first().copied() == Some("regs") {
                let st = self.st.lock();
                let _ = writeln!(w, "Adaptec AIC-7880, controller {}", self.index);
                let _ = writeln!(w, "  INTSTAT={:02x} SSTAT0={:02x} SSTAT1={:02x} SEQADDR={:#x}", st.intstat, st.sstat0, st.sstat1, st.seqaddr);
                let _ = writeln!(w, "  queue in: {} waiting (pos {})  queue out: {} waiting (pos {})", st.qin.len(), st.qinpos, st.qout.len(), st.qoutpos);
                let _ = writeln!(w, "  worker: {}  held completions: {}  NetBSD doorbell: {}",
                    if self.running.load(Ordering::Acquire) { "running" } else { "off (commands run inline)" },
                    st.held.len(), st.nb_doorbell);
                let _ = writeln!(w, "  sequencer pauses: {}  commands: {} ({} bytes in, {} out)", st.pauses, st.total_cmds, st.total_in, st.total_out);
                for n in &st.notes {
                    let _ = writeln!(w, "  note: {n}");
                }
                return Ok(());
            }
            return Err(format!("Usage: {} | scsi regs", crate::scsi_monitor::SCSI_USAGE));
        }
        if cmd.starts_with("cow") {
            return crate::scsi_monitor::cow_command(self, args, &mut *w);
        }
        Err("Command not found".to_string())
    }
}

impl Aic7880 {

    /// Whether this address is the one disk we have.
    ///
    /// A controller that answers for every target and LUN makes a driver
    /// attach sixty-four copies of the same disk, which is slow, confusing,
    /// and nothing like the machine. Where the target is known we answer only
    /// at our own address; where it is not — the PROM's SCBs do not carry it —
    /// we answer, because refusing would be a guess.
    /// Which disk a command is for, if any. A command whose target we cannot
    /// see -- the PROM's SCBs do not carry one -- goes to the default id,
    /// because refusing would be a guess.
    fn target_of(&self, who: Option<(u8, u8)>) -> Option<u8> {
        match who {
            Some((_, l)) if l != 0 => None,
            Some((t, _)) => Some(t),
            None => Some(self.target),
        }
    }

    /// Run a CDB against the attached disk. Returns the data to send back and
    /// a SCSI status byte.
    fn execute_for(&self, cdb: &[u8], who: Option<(u8, u8)>) -> (Vec<u8>, u8) {
        let present = self
            .target_of(who)
            .map(|t| self.has_target(t))
            .unwrap_or(false);
        if !present {
            // The way SCSI says "nothing here": an INQUIRY answered with
            // peripheral qualifier 011b and type 1Fh. Anything else is a
            // check condition.
            if cdb[0] == 0x12 {
                let mut d = vec![0u8; 36];
                d[0] = 0x7f;
                d[4] = 31;
                d.truncate((cdb[4] as usize).min(36));
                return (d, 0x00);
            }
            // A target that is there but has no such LUN says so, so the
            // REQUEST SENSE that follows explains the failure.
            if let Some((t, l)) = who {
                if l != 0 && self.has_target(t) {
                    // ILLEGAL REQUEST, logical unit not supported.
                    *self.sense.lock().unwrap() = Some((0x05, 0x25, 0x00));
                }
            }
            return (Vec::new(), 0x02);
        }
        self.execute_on(cdb, self.target_of(who).unwrap())
    }

    /// Run a CDB against the attached disk. Returns the data to send back and
    /// a SCSI status byte.
    fn execute(&self, cdb: &[u8]) -> (Vec<u8>, u8) {
        self.execute_on(cdb, self.target)
    }

    fn execute_on(&self, cdb: &[u8], target: u8) -> (Vec<u8>, u8) {
        self.request(target, cdb, None)
    }

    /// A command for the target at `target`, through the shared SCSI device:
    /// the data it returns and its status. A REQUEST SENSE first hands out
    /// the controller's own sense, if it holds one.
    fn request(&self, target: u8, cdb: &[u8], data_in: Option<&[u8]>) -> (Vec<u8>, u8) {
        const CHECK_CONDITION: u8 = 0x02;
        // The bar has lights for SCSI ids 0-5.
        if let (Some(hb), 0..=5) = (self.heartbeat.get(), target) {
            hb.fetch_or(1u64 << (crate::dev::ng1::rex3::Rex3::HB_SCSI_BASE as u64 + target as u64), Ordering::Relaxed);
        }
        if cdb[0] == scsi_cmd::REQUEST_SENSE {
            if let Some((key, asc, ascq)) = self.sense.lock().unwrap().take() {
                let mut d = vec![0u8; 18];
                d[0] = 0x70; // current error
                d[2] = key;
                d[7] = 10;
                d[12] = asc;
                d[13] = ascq;
                d.truncate(crate::scsi::request_sense_alloc_len(cdb).min(18));
                return (d, 0x00);
            }
        }
        let req = ScsiRequest { cdb: cdb.to_vec(), data_len: ScsiDataLength::Unlimited, data_in: data_in.map(|d| d.to_vec()) };
        match self.bus.request(target, &req) {
            None => (Vec::new(), CHECK_CONDITION),
            Some(Ok(r)) => (r.data, r.status),
            Some(Err(e)) => {
                eprintln!("aic7880: target {target}: {e}");
                (Vec::new(), CHECK_CONDITION)
            }
        }
    }

    /// Collect `want` bytes out of the buffers the driver listed, for a
    /// command that sends data to the device.
    /// Collect bytes out of an explicit list of buffers.
    fn gather_segs(&self, segs: &[(u32, u32)], want: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(want);
        for (a, l) in segs {
            if out.len() >= want {
                break;
            }
            let n = (*l as usize).min(want - out.len());
            out.extend(self.dma_read_bytes(*a, n));
        }
        out
    }

    /// Place bytes into an explicit list of buffers; returns how many landed.
    fn scatter_segs(&self, segs: &[(u32, u32)], data: &[u8]) -> usize {
        let mut done = 0usize;
        for (a, l) in segs {
            if done >= data.len() {
                break;
            }
            let n = (*l as usize).min(data.len() - done);
            self.dma_write_bytes(*a, &data[done..done + n]);
            done += n;
        }
        done
    }

    fn gather(&self, sg_ptr: u32, want: usize) -> (Vec<u8>, Vec<(u32, u32)>) {
        let mut entries = Vec::new();
        let mut out = Vec::with_capacity(want);
        for i in 0..SG_MAX {
            if out.len() >= want {
                break;
            }
            let a = self.dma_read32(sg_ptr + i as u32 * 8);
            let raw = self.dma_read32(sg_ptr + i as u32 * 8 + 4);
            let l = raw & SG_LEN_MASK;
            if a == 0 || l == 0 {
                break;
            }
            entries.push((a, l));
            let n = (l as usize).min(want - out.len());
            out.extend(self.dma_read_bytes(a, n));
            // Whatever follows the last entry is not part of the list.
            if raw & SG_LAST != 0 {
                break;
            }
        }
        (out, entries)
    }

    /// Copy data back into the buffers the driver listed. Returns how much
    /// was placed, and the list as read, for the trace.
    fn scatter(&self, sg_ptr: u32, data: &[u8]) -> (usize, Vec<(u32, u32)>) {
        let mut entries = Vec::new();
        let mut done = 0usize;
        for i in 0..SG_MAX {
            if done >= data.len() {
                break;
            }
            let a = self.dma_read32(sg_ptr + i as u32 * 8);
            let raw = self.dma_read32(sg_ptr + i as u32 * 8 + 4);
            let l = raw & SG_LEN_MASK;
            if a == 0 || l == 0 {
                break;
            }
            entries.push((a, l));
            let n = (l as usize).min(data.len() - done);
            self.dma_write_bytes(a, &data[done..done + n]);
            done += n;
            if raw & SG_LAST != 0 {
                break;
            }
        }
        (done, entries)
    }

    /// How many bytes a command sends *to* the device, if it is one that
    /// does. Everything else either receives data or carries none.
    fn data_out_len(cdb: &[u8]) -> Option<usize> {
        match cdb[0] {
            // WRITE(6): a zero count means 256 blocks, as with READ(6).
            0x0a => Some(if cdb[4] == 0 { 256 } else { cdb[4] as usize } * BLOCK_SIZE),
            // WRITE(10).
            0x2a => Some((((cdb[7] as usize) << 8) | cdb[8] as usize) * BLOCK_SIZE),
            // MODE SELECT(6) and (10): the parameter list, whose block
            // descriptor can change the block length.
            0x15 => Some(cdb[4] as usize),
            0x55 => Some(((cdb[7] as usize) << 8) | cdb[8] as usize),
            _ => None,
        }
    }

    /// Run a command that carries data to the disk.
    fn execute_write(&self, cdb: &[u8], data: &[u8]) -> u8 {
        self.execute_write_on(cdb, data, self.target)
    }

    fn execute_write_on(&self, cdb: &[u8], data: &[u8], target: u8) -> u8 {
        self.request(target, cdb, Some(data)).1
    }

    /// The next tag from NetBSD's queue-in ring, if the host has added one
    /// the sequencer has not taken. Clears the doorbell once it has caught up.
    fn plan_netbsd(&self, st: &mut State) -> Option<Job> {
        let want = st.regs[nb::KERNEL_QINPOS as usize];
        let shared = Self::scratch_le32(st, nb::SHARED_DATA_ADDR);
        let hscb_base = Self::scratch_le32(st, nb::HSCB_ADDR);
        if shared == 0 || hscb_base == 0 || st.qinpos == want {
            st.nb_doorbell = false;
            return None;
        }
        let tag = self.dma_read8(shared + nb::FIFO_LEN + st.qinpos as u32);
        st.qinpos = st.qinpos.wrapping_add(1);
        Some(Job::NetBsd { tag, hscb_base, shared })
    }

    /// The buffers one command transfers through: the SCB carries the first
    /// segment directly and points at the rest.
    ///
    /// The driver copies the first list entry into the SCB with its flags,
    /// and SGPTR points at the *second* entry. When the first is also the
    /// last, that second entry is whatever an earlier, longer command left
    /// there, so the list ends here. Following it put a device's surplus
    /// bytes (READ TOC answered 20 where 12 were asked for) at the start of
    /// a page the kernel had reused for file-system metadata.
    fn netbsd_segments(&self, hscb: u32) -> Vec<(u32, u32)> {
        let mut segs = Vec::new();
        let first = self.dma_read32_le(hscb + nb::DATACNT);
        let first_len = first & SG_LEN_MASK;
        if first_len != 0 {
            segs.push((self.dma_read32_le(hscb + nb::DATAPTR), first_len));
        }
        if first & nb::LAST_SEG != 0 {
            return segs;
        }
        let mut sg = self.dma_read32_le(hscb + nb::SGPTR) & nb::SG_PTR_MASK;
        for _ in 0..SG_MAX {
            if sg == 0 {
                break;
            }
            let a = self.dma_read32_le(sg);
            let raw = self.dma_read32_le(sg + 4);
            let l = raw & SG_LEN_MASK;
            if a == 0 && l == 0 {
                break;
            }
            segs.push((a, l));
            if raw & nb::LAST_SEG != 0 {
                break;
            }
            sg += 8;
        }
        segs
    }

    /// Run one command the NetBSD way. Called without the state lock.
    fn run_netbsd(&self, out: &mut Outcome, tag: u8, hscb_base: u32, shared: u32) {
        let hscb = hscb_base + tag as u32 * nb::HSCB_STRIDE;
        let len = self.dma_read8(hscb + nb::CDB_LEN) as usize;
        // A CDB longer than the twelve bytes that fit inline is kept
        // elsewhere and pointed at from the same field.
        let cdb = if len <= 12 {
            self.dma_read_bytes(hscb + nb::CDB, len.clamp(6, 12))
        } else {
            self.dma_read_bytes(self.dma_read32_le(hscb + nb::CDB), len.min(32))
        };
        let segs = self.netbsd_segments(hscb);
        // SCSIID holds the target in the high nibble and our own id in the
        // low one -- the driver's own card dump prints SCB_SCSIID[0x17] for
        // target 1 on an adapter at id 7.
        let scsiid = self.dma_read8(hscb + nb::SCSIID);
        let lun = self.dma_read8(hscb + nb::LUN) & 0x07;

        let (status, moved) = match Self::data_out_len(&cdb) {
            Some(want) => {
                let data = self.gather_segs(&segs, want);
                let n = data.len();
                let t = self.target_of(Some((scsiid >> 4, lun))).unwrap_or(self.target);
                out.bytes_out += n as u64;
                (self.execute_write_on(&cdb, &data, t), n)
            }
            None => {
                let (data, status) = self.execute_for(&cdb, Some((scsiid >> 4, lun)));
                let moved = self.scatter_segs(&segs, &data);
                // More data than the command's buffers hold: what does not
                // fit is dropped, as the chip would. Say so.
                if moved < data.len() {
                    out.notes.push(format!("nb overrun: CDB {:02x} returned {} bytes into {}",
                        cdb[0], data.len(), moved));
                }
                out.bytes_in += moved as u64;
                (status, moved)
            }
        };

        let hex: Vec<String> = cdb.iter().map(|b| format!("{b:02x}")).collect();
        let segtxt: Vec<String> = segs.iter()
            .map(|(a, l)| format!("0x{a:08x}+{l}")).collect();
        out.notes.push(format!(
            "nb tag {tag} id 0x{scsiid:02x} lun {lun}: CDB {} -> status {status}, {moved} bytes into [{}]",
            hex.join(" "), segtxt.join(" ")));

        // Status goes back in the SCB, and the residual words beside it are
        // cleared so the driver does not go looking for one. The SCB is the
        // command's own, so this can happen before the tag is posted.
        for i in 0..8 {
            self.dma_write8(hscb + i, 0);
        }
        self.dma_write8(hscb + nb::STATUS, status);
        for i in 0..4 {
            self.dma_write8(hscb + nb::SGPTR + i, 0);
        }
        out.cmds += 1;
        out.done = Some(Done::NetBsd { tag, shared });
    }

    /// What the sequencer needs from the chip to run a tag from `QINFIFO`,
    /// read while the state is locked.
    fn plan_tag(st: &mut State, tag: u8) -> Job {
        let i = (tag as usize) % SCB_COUNT;
        // The register window is used by both drivers, but only to *clear*
        // SCBs during initialisation -- a window write is not evidence that
        // the command lives on the chip. Report a window-filled SCB that
        // carries anything, so the day one does it is not a silent surprise,
        // and otherwise take the host-memory path both drivers actually use.
        if st.scb_onchip[i] {
            st.scb_onchip[i] = false;
            let scb = st.scb[i];
            if scb.iter().take(SCB_SIZE - 1).any(|b| *b != 0) {
                let hex: Vec<String> = scb.iter().map(|b| format!("{b:02x}")).collect();
                Self::note(st, format!("tag {tag}: on-chip SCB carries {}", hex.join(" ")));
            }
        }
        Job::Tag {
            tag,
            array: Self::scratch_le32(st, sram::SCB_ARRAY),
            errors: Self::him_errors(st),
            own: st.regs[reg::SCSIID as usize] & 0x0f,
        }
    }

    /// Fetch a queued command and run it. Called without the state lock:
    /// everything here is host memory and the disk.
    fn run_tag(&self, out: &mut Outcome, tag: u8, array: u32, errors: bool, own: u8) {
        if array == 0 {
            out.notes.push(format!("tag {tag} queued with no SCB array configured"));
            return;
        }
        let scb_addr = self.dma_read32(array + tag as u32 * 4);
        if scb_addr == 0 {
            out.notes.push(format!("tag {tag} has no SCB address in the array"));
            return;
        }
        let scb = self.dma_read_bytes(scb_addr, SCB_SIZE);
        let cmd_ptr = u32::from_be_bytes([
            scb[scb::CMD_PTR], scb[scb::CMD_PTR + 1],
            scb[scb::CMD_PTR + 2], scb[scb::CMD_PTR + 3],
        ]);
        // The SCB says how long the CDB is; fall back on the opcode group if
        // it says something impossible.
        let len = scb[scb::CDB_LEN] as usize;
        let len = if (6..=16).contains(&len) {
            len
        } else {
            Self::cdb_len(self.dma_read8(cmd_ptr))
        };
        let cdb = self.dma_read_bytes(cmd_ptr, len);

        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<Vec<_>>().join(" ");
        out.notes.push(format!(
            "tag {tag}: SCB at 0x{scb_addr:08x} control 0x{:02x} target {} lun {} cdblen {}",
            scb[scb::CONTROL], scb[scb::TCL] >> 4, scb[scb::TCL] & 0x07, len));
        out.notes.push(format!("   CDB {}", hex(&cdb)));
        out.notes.push(format!("   SCB {}", hex(&scb)));
        let sg_ptr = u32::from_be_bytes([
            scb[scb::SG_PTR], scb[scb::SG_PTR + 1],
            scb[scb::SG_PTR + 2], scb[scb::SG_PTR + 3],
        ]);
        let tcl = scb[scb::TCL];
        let target = tcl >> 4;
        let who = Some((target, tcl & 0x07));

        // Nobody at that id, or our own: the selection times out. The HIM
        // finds the tag in the waiting-SCB byte and gives up on the id at
        // once, where an INQUIRY answered "no device" with GOOD status made
        // it look for a device that never comes.
        if errors && (target == own || !self.has_target(target)) {
            out.notes.push(format!("   -> selection timeout at id {target}"));
            out.done = Some(Done::SelectionTimeout { tag });
            return;
        }
        let (status, moved, sg, dir) = match Self::data_out_len(&cdb) {
            Some(want) => {
                let (data, sg) = self.gather(sg_ptr, want);
                let n = data.len();
                let t = self.target_of(who).unwrap_or(self.target);
                (self.execute_write_on(&cdb, &data, t), n, sg, "out of")
            }
            None => {
                let (data, status) = self.execute_for(&cdb, who);
                let (sent, sg) = if data.is_empty() {
                    (0, Vec::new())
                } else {
                    self.scatter(sg_ptr, &data)
                };
                (status, sent, sg, "into")
            }
        };
        let sgtxt: Vec<String> = sg.iter()
            .map(|(a, l)| format!("0x{a:08x}+{l}")).collect();
        // Nothing is written back to the SCB. GOOD status means the zeros
        // the host left in it stand, and anything else is reported below.
        out.cmds += 1;
        if dir == "into" { out.bytes_in += moved as u64 } else { out.bytes_out += moved as u64 }
        out.notes.push(format!(
            "   -> status {status}, {moved} bytes {dir} [{}]", sgtxt.join(" ")));
        out.executed = Some((tag, scb_addr, cdb));

        // Any status but GOOD stops the sequencer with the status in
        // scratch RAM for the HIM to collect. It rewrites the same SCB as a
        // REQUEST SENSE and queues the tag again at the head, and that one
        // completes normally with the sense `self.sense` is holding.
        out.done = Some(if errors && status != 0x00 {
            Done::BadStatus { tag, status }
        } else {
            Done::Complete { tag, target }
        });
    }

    /// Hand a finished command back to the host. Called with the state
    /// locked and the sequencer running, so the host sees the completion
    /// byte, the cursor and `CMDCMPLT` change together.
    fn post(&self, st: &mut State, done: Done) {
        match done {
            Done::SelectionTimeout { tag } => {
                st.regs[him::WAITING_SCB as usize] = tag;
                st.sstat1 |= sstat1::SELTO;
                Self::raise_pausing(st, intstat::SCSIINT);
            }
            Done::BadStatus { tag, status } => {
                st.regs[him::ACTIVE_SCB as usize] = tag;
                st.regs[him::STATUS as usize] = status;
                st.regs[reg::SCBPTR as usize] = tag;
                st.intstat = (st.intstat & !intstat::INTCODE) | him::BAD_STATUS;
                Self::raise_pausing(st, intstat::SEQINT);
            }
            Done::Complete { tag, target } => {
                if Self::him(st) {
                    self.complete_him(st, tag, target);
                } else {
                    self.complete(st, tag);
                }
            }
            Done::NetBsd { tag, shared } => {
                // The tag goes back through the other ring.
                self.dma_write8(shared + st.qoutpos, tag);
                st.qoutpos = (st.qoutpos + 1) % nb::FIFO_LEN;
                st.intstat |= intstat::CMDCMPLT;
            }
        }
    }

    /// Post a finished tag the way the HIM's sequencer does: two single
    /// bytes into the queue-out array, with the cursor in scratch RAM.
    ///
    /// The HIM reads exactly one byte for each completion, at its own
    /// cursor (starting at 4, moving down by 4) with the low two bits
    /// flipped, because the array is in the view of memory that keeps
    /// 32-bit words whole and so turns single bytes around. The sequencer
    /// keeps its cursor four behind: it marks the slot the host will look
    /// at next as empty and puts the tag one slot ahead of that, which is
    /// where the host is looking now. Writing tags at 0, 1, 2, ... instead
    /// put every one where the kernel never looked, and each command timed
    /// out after ten seconds. The PROM scans the whole array, so this
    /// serves it too.
    fn complete_him(&self, st: &mut State, tag: u8, target: u8) {
        let ptr = Self::scratch_le32(st, sram::QOUTFIFO);
        let base = ptr & !0xff;
        let p = ptr & 0xff;
        // The low alias maps address zero to memory, so an array that was
        // never configured must not be written through it.
        if base != 0 {
            self.dma_write8(base + (p ^ 3), QOUTFIFO_EMPTY);
            self.dma_write8(base + (((p + 4) & 0xff) ^ 3), tag);
        }
        // Add 4 and then 0xf8, with no carry into the next byte: the
        // cursor wraps inside the array.
        st.regs[sram::QOUTFIFO as usize] = (p.wrapping_sub(4) & 0xff) as u8;
        st.regs[him::ACTIVE_SCB as usize] = 0xff;
        // The target is no longer busy. The busy table is byte 31 of the
        // on-chip SCB with the target's number.
        st.scb[target as usize % SCB_COUNT][SCB_SIZE - 1] = 0xff;
        st.intstat |= intstat::CMDCMPLT;
    }

    /// Post a finished tag where a driver without the HIM looks for it.
    fn complete(&self, st: &mut State, tag: u8) {
        let fifo = Self::scratch_le32(st, sram::QOUTFIFO);
        if fifo != 0 {
            self.dma_write8(fifo + st.qoutpos, tag);
            st.qoutpos = (st.qoutpos + 1) % QOUTFIFO_LEN;
        }
        st.qout.push_back(tag);
        st.intstat |= intstat::CMDCMPLT;
    }

    /// Whether the sequencer has anything to do now: it is running, and a
    /// completion is held, a tag is queued or NetBSD has rung.
    fn has_work(st: &State) -> bool {
        !Self::paused(st) && (!st.held.is_empty() || !st.qin.is_empty() || st.nb_doorbell)
    }

    /// One step of the sequencer: post a held completion, or take one
    /// command and run it. Returns false when there was nothing to do.
    ///
    /// The command's transfers and disk I/O run with the state unlocked, so
    /// the CPU keeps reading and writing registers meanwhile; only taking
    /// the work and posting the result hold the lock. A host that paused
    /// the sequencer while the command ran finds it held until it unpauses,
    /// as a stopped sequencer posts nothing. A chip reset in between drops it.
    fn service_one(&self) -> bool {
        let _one = self.exec.lock().unwrap();
        let mut st = self.st.lock();
        if Self::paused(&st) {
            return false;
        }
        if let Some(done) = st.held.pop_front() {
            self.post(&mut st, done);
            self.drive_irq(&st);
            return true;
        }
        let job = if let Some(tag) = st.qin.pop_front() {
            Self::plan_tag(&mut st, tag)
        } else if st.nb_doorbell {
            match self.plan_netbsd(&mut st) {
                Some(job) => job,
                None => return false,
            }
        } else {
            return false;
        };
        let generation = st.generation;
        drop(st);

        let mut out = Outcome::default();
        match job {
            Job::Tag { tag, array, errors, own } => self.run_tag(&mut out, tag, array, errors, own),
            Job::NetBsd { tag, hscb_base, shared } => self.run_netbsd(&mut out, tag, hscb_base, shared),
        }

        let mut st = self.st.lock();
        for n in out.notes {
            Self::note(&mut st, n);
        }
        st.total_cmds += out.cmds;
        st.total_in += out.bytes_in;
        st.total_out += out.bytes_out;
        if let Some(e) = out.executed {
            st.executed.push(e);
        }
        if let Some(done) = out.done.filter(|_| st.generation == generation) {
            if Self::paused(&st) {
                st.held.push_back(done);
            } else {
                self.post(&mut st, done);
            }
        }
        self.drive_irq(&st);
        true
    }

    /// Tell the sequencer there may be work: wake the worker, or, with no
    /// worker running (unit tests, the harness), do the work here.
    fn kick(&self) {
        if self.running.load(Ordering::Acquire) {
            if let Some(w) = self.worker.lock().unwrap().as_ref() {
                w.thread().unpark();
            }
            return;
        }
        while self.service_one() {}
    }

    /// The worker: run commands while there are any, sleep until kicked.
    /// It holds only a weak reference between commands, so a controller
    /// dropped without being stopped does not live on in its thread.
    fn worker_loop(me: std::sync::Weak<Self>) {
        loop {
            let Some(c) = me.upgrade() else { return };
            if !c.running.load(Ordering::Acquire) {
                return;
            }
            let worked = c.service_one();
            drop(c);
            if !worked {
                std::thread::park();
            }
        }
    }

    fn scb_index(st: &State) -> usize {
        (st.regs[reg::SCBPTR as usize] as usize) % SCB_COUNT
    }

    fn read8_locked(st: &mut State, off: u32) -> u8 {
        match off {
            reg::INTSTAT => st.intstat,
            reg::SSTAT0 => st.sstat0,
            reg::SSTAT1 => st.sstat1,
            // No target ever asks for a byte, and the HIM waits for REQ to
            // drop after an error, so it must never read as set.
            reg::SCSISIG => st.regs[off as usize] & !scsisig::REQ,
            // A real FIFO: reading takes the head. Only the low four bits
            // exist with no external SCB RAM, and an empty one reads zero.
            reg::QINFIFO => st.qin.pop_front().unwrap_or(0) & 0x0f,
            reg::QINCNT => st.qin.len() as u8,
            reg::QOUTCNT => st.qout.len() as u8,
            reg::QOUTFIFO => st.qout.pop_front().unwrap_or(0xff),
            reg::SEQRAM => {
                let a = st.seqaddr as usize;
                st.seqaddr = st.seqaddr.wrapping_add(1);
                st.seqram.get(a).copied().unwrap_or(0)
            }
            reg::SEQADDR0 => st.seqaddr as u8,
            reg::SEQADDR1 => (st.seqaddr >> 8) as u8,
            reg::SCBARRAY_FIRST..=reg::SCBARRAY_LAST => {
                let i = Self::scb_index(st);
                st.scb[i][(off - reg::SCBARRAY_FIRST) as usize]
            }
            _ => st.regs[off as usize],
        }
    }

    /// SCSIRSTO: the controller drives a SCSI bus reset, and every device
    /// returns to its power-on state. What that is for a CD-ROM depends on
    /// whose drive it is: an SGI drive comes up in 512-byte blocks, which
    /// SGI's own driver (the HIM, in the PROM and in IRIX) relies on --
    /// IRIX resets the bus when it attaches and never sends MODE SELECT
    /// again, then reads 512-byte sectors. The BSDs' driver expects the
    /// common 2048 after its reset (the PROM had left 512 behind).
    fn reset_targets(&self) {
        let sgi = Self::him(&self.st.lock());
        self.bus.for_each(|_, d| d.reset_block_size(sgi));
    }

    /// A register write with the state locked. Returns whether the
    /// sequencer may now have work, which the caller hands on once the
    /// lock is released.
    fn write8_locked(&self, st: &mut SpinGuard<'_, State>, off: u32, val: u8) -> bool {
        match off {
            reg::HCNTRL => {
                if val & hcntrl::CHIPRST != 0 {
                    // A chip reset clears everything except the program we
                    // were given, which the driver reloads anyway.
                    // A command still running is dropped when it finishes.
                    let notes = std::mem::take(&mut st.notes);
                    let (c, i, o) = (st.total_cmds, st.total_in, st.total_out);
                    let generation = st.generation + 1;
                    **st = State {
                        notes,
                        total_cmds: c,
                        total_in: i,
                        total_out: o,
                        generation,
                        ..Default::default()
                    };
                    Self::note(st, "chip reset".into());
                    // The reset bit reads back as an acknowledgement.
                    st.regs[reg::HCNTRL as usize] = hcntrl::CHIPRST;
                    return false;
                }
                let was = st.regs[reg::HCNTRL as usize];
                // An interrupt that stopped the sequencer keeps it stopped
                // until the host has cleared it; only then does a restart
                // take. The HIM checks for exactly this and retries.
                let val = if st.intstat & intstat::PAUSING != 0 { val | hcntrl::PAUSE } else { val };
                st.regs[reg::HCNTRL as usize] = val;
                // The driver pauses and restarts the sequencer around every
                // shared-state access, thousands of times. Counting them says
                // as much as logging them and leaves room for the commands.
                if was & hcntrl::PAUSE != val & hcntrl::PAUSE {
                    st.pauses += 1;
                }
                if val & hcntrl::PAUSE == 0 {
                    // The HIM's sequencer goes back to its idle loop, where
                    // nothing is being selected and nothing is connected.
                    if Self::him(st) {
                        st.regs[him::WAITING_SCB as usize] = 0xff;
                        st.regs[him::ACTIVE_SCB as usize] = 0xff;
                    }
                    // And picks up whatever was queued while it was stopped.
                    return Self::has_work(st);
                }
            }
            reg::SEQCTL => {
                let was = st.regs[reg::SEQCTL as usize];
                st.regs[off as usize] = val;
                if val & seqctl::SEQRESET != 0 {
                    st.seqaddr = 0;
                }
                // Only the rising edge starts a download. The driver writes
                // this register several times during one, and clearing the
                // program on each write throws the whole thing away.
                if val & seqctl::LOADRAM != 0 && was & seqctl::LOADRAM == 0 {
                    // LOADRAM only points SEQRAM accesses at the program
                    // store; it does not erase it. The driver sets it a
                    // second time to read the program back and check it, and
                    // clearing here would hand it zeros and fail the check.
                    st.seqaddr = 0;
                    Self::note(st, "sequencer RAM selected".into());
                }
                if was & seqctl::LOADRAM != 0 && val & seqctl::LOADRAM == 0 {
                    let n = st.seqram.len();
                    Self::note(st, format!("sequencer RAM deselected, {n} bytes held"));
                }
            }
            reg::SEQADDR0 => st.seqaddr = (st.seqaddr & 0xff00) | val as u16,
            reg::SEQADDR1 => st.seqaddr = (st.seqaddr & 0x00ff) | ((val as u16) << 8),
            reg::SEQRAM => {
                let a = st.seqaddr as usize;
                if st.seqram.len() <= a {
                    st.seqram.resize(a + 1, 0);
                }
                st.seqram[a] = val;
                st.seqaddr = st.seqaddr.wrapping_add(1);
            }
            nb::KERNEL_QINPOS => {
                st.regs[off as usize] = val;
                // The HIM's next-SCB table covers this byte, and filling it
                // at init used to ring the doorbell with its placeholder
                // pointer standing in for NetBSD's rings: 127 commands from
                // nowhere, and completions nobody had asked for. A doorbell
                // is only one when the HIM is not driving and both rings
                // are in memory. The sequencer takes the new tags when it
                // runs; the driver rings with it paused.
                let rings = [nb::HSCB_ADDR, nb::SHARED_DATA_ADDR]
                    .iter()
                    .all(|r| self.is_memory(Self::scratch_le32(st, *r)));
                if !Self::him(st) && rings {
                    st.nb_doorbell = true;
                    return Self::has_work(st);
                }
            }
            reg::ERROR_CLRINT => {
                // Write-one-to-clear, sharing an address with the sequencer
                // error register that reads back. Clearing SEQINT clears
                // its code with it.
                let clr = val & 0x0f;
                st.intstat &= !clr;
                if clr & intstat::SEQINT != 0 {
                    st.intstat &= !intstat::INTCODE;
                }
            }
            // CLRSINT0 and CLRSINT1: clear what is written, store nothing.
            reg::SSTAT0 => st.sstat0 &= !val,
            reg::SSTAT1 => st.sstat1 &= !val,
            reg::QINFIFO => {
                let tag = val & 0x0f;
                let scb = st.scb[tag as usize];
                st.queued.push((val, scb));
                // Sixteen deep; a write to a full FIFO is lost, as on the
                // chip.
                if st.qin.len() < SCB_COUNT {
                    st.qin.push_back(tag);
                }
                return Self::has_work(st);
            }
            reg::SCBARRAY_FIRST..=reg::SCBARRAY_LAST => {
                let i = Self::scb_index(st);
                st.scb[i][(off - reg::SCBARRAY_FIRST) as usize] = val;
                // Remember that this SCB was filled in on the chip. Two
                // drivers use two different submission paths: the PROM's
                // sequencer program fetches SCBs from host memory and is
                // handed only a tag, while NetBSD's writes the whole SCB
                // through this window first. Which one is in use is decided
                // by what the driver actually did, not by guessing.
                st.scb_onchip[i] = true;
            }
            _ => st.regs[off as usize] = val,
        }
        false
    }
}

impl Aic7880 {
    /// A register access of `width` bytes at `off`, as one bus cycle: the
    /// state is locked once for all of its bytes.
    pub fn read(&self, off: u32, width: usize) -> u32 {
        // Everything here is byte-wide; a wider access reads consecutive
        // registers, most significant first, the way the bus presents them.
        let mut st = self.st.lock();
        let mut v = 0u32;
        for i in 0..width as u32 {
            v = (v << 8) | Self::read8_locked(&mut st, off + i) as u32;
        }
        v
    }

    pub fn write(&self, off: u32, val: u32, width: usize) {
        let byte = |i: u32| (val >> (8 * (width as u32 - 1 - i))) as u8;
        if (0..width as u32).any(|i| off + i == reg::SCSISEQ && byte(i) & 0x01 != 0) {
            self.reset_targets();
        }
        let mut st = self.st.lock();
        let mut kick = false;
        for i in 0..width as u32 {
            kick |= self.write8_locked(&mut st, off + i, byte(i));
        }
        self.drive_irq(&st);
        drop(st);
        if kick {
            self.kick();
        }
    }
}

impl Aic7880 {
    /// Copy each target's copy-on-write overlay into `dir` as
    /// `scsi<id>.overlay`, and return the dirty sectors of each, as the
    /// Indigo2's and Indy's controller does for their snapshots.
    pub fn export_overlays(&self, dir: &std::path::Path) -> std::io::Result<Vec<(usize, Vec<u64>)>> {
        self.bus.export_overlays(dir)
    }

    /// Put back the overlays [`Self::export_overlays`] wrote.
    pub fn import_overlays(&self, dir: &std::path::Path, dirty_sets: &[(usize, Vec<u64>)]) -> std::io::Result<()> {
        self.bus.import_overlays(dir, dirty_sets)
    }
}

/// What the guest can see of the chip: registers, SCBs, the sequencer's
/// program and address, both queues, the latched status, the host-ring
/// positions, the controller's own sense, and finished commands held for a
/// paused sequencer; loading drives the interrupt again. Save with the
/// worker stopped, so no command is part way through. The disks' contents go with the snapshot's overlays
/// ([`Aic7880::export_overlays`]); the logs of queued and executed commands
/// are evidence, not state.
impl crate::traits::Saveable for Aic7880 {
    fn save_state(&self) -> toml::Value {
        let st = self.st.lock();
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u8_slice_to_toml(&st.regs));
        t.insert("scb".into(), toml::Value::Array(st.scb.iter().map(|b| u8_slice_to_toml(b)).collect()));
        t.insert("seqram".into(), u8_slice_to_toml(&st.seqram));
        t.insert("seqaddr".into(), hex_u16(st.seqaddr));
        let q = |q: &VecDeque<u8>| toml::Value::Array(q.iter().map(|&b| hex_u8(b)).collect());
        t.insert("qin".into(), q(&st.qin));
        t.insert("qout".into(), q(&st.qout));
        t.insert("intstat".into(), hex_u8(st.intstat));
        t.insert("sstat0".into(), hex_u8(st.sstat0));
        t.insert("sstat1".into(), hex_u8(st.sstat1));
        t.insert("qoutpos".into(), hex_u32(st.qoutpos));
        t.insert("qinpos".into(), hex_u8(st.qinpos));
        t.insert("nb_doorbell".into(), toml::Value::Boolean(st.nb_doorbell));
        t.insert("held".into(), toml::Value::Array(st.held.iter().map(|d| d.to_toml()).collect()));
        t.insert("scb_onchip".into(),
            toml::Value::Array(st.scb_onchip.iter().map(|&b| toml::Value::Boolean(b)).collect()));
        t.insert("total_cmds".into(), hex_u64(st.total_cmds));
        t.insert("total_in".into(), hex_u64(st.total_in));
        t.insert("total_out".into(), hex_u64(st.total_out));
        if let Some((key, asc, ascq)) = *self.sense.lock().unwrap() {
            t.insert("sense".into(), toml::Value::Array(vec![hex_u8(key), hex_u8(asc), hex_u8(ascq)]));
        }
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        {
            let mut st = self.st.lock();
            // Anything running when this was called belongs to the old state.
            let generation = st.generation + 1;
            *st = State { generation, ..Default::default() };
            let b = |k: &str| get_field(v, k).and_then(toml_u8).unwrap_or(0);
            if let Some(r) = get_field(v, "regs") {
                load_u8_slice(r, &mut st.regs);
            }
            if let Some(toml::Value::Array(a)) = get_field(v, "scb") {
                for (dst, src) in st.scb.iter_mut().zip(a) {
                    load_u8_slice(src, dst);
                }
            }
            if let Some(toml::Value::Array(a)) = get_field(v, "seqram") {
                st.seqram = a.iter().map(|x| toml_u8(x).unwrap_or(0)).collect();
            }
            st.seqaddr = get_field(v, "seqaddr").and_then(toml_u16).unwrap_or(0);
            let q = |k: &str| match get_field(v, k) {
                Some(toml::Value::Array(a)) => a.iter().map(|x| toml_u8(x).unwrap_or(0)).collect(),
                _ => VecDeque::new(),
            };
            st.qin = q("qin");
            st.qout = q("qout");
            st.intstat = b("intstat");
            st.sstat0 = b("sstat0");
            st.sstat1 = b("sstat1");
            st.qoutpos = get_field(v, "qoutpos").and_then(toml_u32).unwrap_or(0);
            st.qinpos = b("qinpos");
            st.nb_doorbell = get_field(v, "nb_doorbell").and_then(|x| x.as_bool()).unwrap_or(false);
            if let Some(toml::Value::Array(a)) = get_field(v, "held") {
                st.held = a.iter().filter_map(Done::from_toml).collect();
            }
            if let Some(toml::Value::Array(a)) = get_field(v, "scb_onchip") {
                for (o, x) in st.scb_onchip.iter_mut().zip(a) {
                    *o = x.as_bool().unwrap_or(false);
                }
            }
            let u = |k: &str| get_field(v, k).and_then(toml_u64).unwrap_or(0);
            st.total_cmds = u("total_cmds");
            st.total_in = u("total_in");
            st.total_out = u("total_out");
        }
        *self.sense.lock().unwrap() = match get_field(v, "sense").and_then(|s| s.as_array()) {
            Some(a) if a.len() == 3 => Some((toml_u8(&a[0]).unwrap_or(0), toml_u8(&a[1]).unwrap_or(0), toml_u8(&a[2]).unwrap_or(0))),
            _ => None,
        };
        self.update_irq();
        Ok(())
    }
}

/// Both BARs reach the same registers.
impl crate::pci::PciBusDevice for Aic7880 {
    fn config(&self) -> &crate::pci::PciConfig {
        &self.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        self.read(off, width)
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.write(off, val, width)
    }

    fn connect_interrupt(&self, line: crate::irq::IrqLine) {
        self.connect_irq(line);
    }

    fn connect_dma(&self, dma: std::sync::Arc<dyn crate::pci::PciDma>) {
        let _ = self.dma.set(dma);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A per-process disk image path. The suite runs in several processes at
    /// once (parallel cargo runs, nextest), all sharing one temp directory, so
    /// a name fixed per test lets another process rewrite the image under it.
    fn image_path(name: &str) -> TempImage {
        TempImage(std::env::temp_dir().join(format!("iris-aic7880-{name}-{}.img", std::process::id())))
    }

    /// A disk image path that is removed when the test is done with it. The
    /// bus keeps its own open handle, so the file can go while the chip lives.
    struct TempImage(std::path::PathBuf);

    impl std::ops::Deref for TempImage {
        type Target = std::path::Path;
        fn deref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl AsRef<std::path::Path> for TempImage {
        fn as_ref(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempImage {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn chip() -> Arc<Aic7880> {
        on(std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20)))
    }

    /// A chip that masters host memory `ram` through the bridge.
    fn on(ram: Ram) -> Arc<Aic7880> {
        let c = Aic7880::new();
        crate::pci::PciBusDevice::connect_dma(&*c, std::sync::Arc::new(crate::dev::mace_pci::HostMemory::new(ram)));
        c
    }

    #[test]
    fn save_load_round_trip() {
        use crate::traits::Saveable;
        let src = chip();
        {
            let mut st = src.st.lock();
            st.regs[0x20] = 0x5a;
            st.scb[3][0] = 0x12;
            st.scb[15][SCB_SIZE - 1] = 0xff;
            st.seqram = vec![1, 2, 3, 4];
            st.seqaddr = 0x1f;
            st.qin = [3u8, 7].into_iter().collect();
            st.qout = [1u8].into_iter().collect();
            st.intstat = 0x02;
            st.sstat0 = 0x40;
            st.sstat1 = 0x08;
            st.qoutpos = 9;
            st.qinpos = 4;
            st.scb_onchip[3] = true;
            st.nb_doorbell = true;
            st.held = [Done::Complete { tag: 2, target: 1 }, Done::NetBsd { tag: 5, shared: 0x1234_5600 },
                Done::BadStatus { tag: 3, status: 2 }, Done::SelectionTimeout { tag: 4 }].into_iter().collect();
            st.total_cmds = 100;
            st.total_in = 1 << 20;
            st.total_out = 4096;
        }
        *src.sense.lock().unwrap() = Some((5, 0x25, 0));
        let v1 = src.save_state();
        let dst = chip();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "Aic7880 save_state mismatch after load_state round-trip");
    }

    #[test]
    fn a_chip_reset_acknowledges_itself() {
        let c = chip();
        c.write(reg::HCNTRL, hcntrl::CHIPRST as u32, 1);
        assert_eq!(
            c.read(reg::HCNTRL, 1) as u8 & hcntrl::CHIPRST,
            hcntrl::CHIPRST,
            "the reset bit reads back as an acknowledgement"
        );
    }

    /// The driver selects an SCB and then reads and writes it through a fixed
    /// window. Getting this wrong silently corrupts every command.
    #[test]
    fn the_scb_window_follows_the_pointer() {
        let c = chip();
        c.write(reg::SCBPTR, 3, 1);
        c.write(reg::SCBARRAY_FIRST + 4, 0xab, 1);
        c.write(reg::SCBPTR, 7, 1);
        assert_eq!(c.read(reg::SCBARRAY_FIRST + 4, 1), 0, "a different SCB is a different buffer");
        c.write(reg::SCBPTR, 3, 1);
        assert_eq!(c.read(reg::SCBARRAY_FIRST + 4, 1), 0xab, "and the first one kept its contents");
    }

    /// The download is how the driver installs the behaviour we are standing
    /// in for, so it has to be accepted and read back.
    #[test]
    fn the_sequencer_program_loads_and_reads_back() {
        let c = chip();
        c.write(reg::SEQCTL, seqctl::LOADRAM as u32, 1);
        for (i, b) in [1u8, 2, 3, 4].iter().enumerate() {
            c.write(reg::SEQRAM, *b as u32, 1);
            assert_eq!(c.read(reg::SEQADDR0, 1), i as u32 + 1, "SEQRAM auto-increments");
        }
        assert_eq!(c.seqram_len(), 4);

        c.write(reg::SEQADDR0, 0, 1);
        c.write(reg::SEQADDR1, 0, 1);
        assert_eq!(c.read(reg::SEQRAM, 1), 1, "and reads back what was loaded");
    }

    /// Queueing captures the SCB as it stood, which is the evidence the
    /// layout is read off. The sequencer is paused so the tag stays queued
    /// long enough to be counted.
    #[test]
    fn queueing_an_scb_records_it() {
        let c = chip();
        c.write(reg::HCNTRL, hcntrl::PAUSE as u32, 1);
        c.write(reg::SCBPTR, 2, 1);
        c.write(reg::SCBARRAY_FIRST, 0x12, 1);
        c.write(reg::QINFIFO, 2, 1);

        assert_eq!(c.read(reg::QINCNT, 1), 1, "the queue reports its depth");
        let q = c.queued();
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].0, 2, "the index queued");
        assert_eq!(q[0].1[0], 0x12, "and the SCB as it was when queued");
    }

    /// The whole submission path, built the way the driver builds it: a SCB
    /// in host memory, its address in an array the chip was told about, and
    /// only a tag handed to the chip. Every step of this was read out of the
    /// driver rather than a datasheet, so it is worth pinning down.
    #[test]
    fn a_queued_tag_fetches_the_command_and_answers_it() {
        let ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20));
        let c = on(ram.clone());

        // A controller with nothing attached answers nothing, so the test
        // needs a disk as much as the machine does.
        let img = image_path("test");
        std::fs::write(&img, vec![0u8; 64 * BLOCK_SIZE]).expect("test image");
        c.bus().attach_image(1, &img, false).expect("attach");

        let base = crate::ip32::RAM_BASE;
        let (array, scb_addr, cdb_addr, sg_addr, buf, qout) =
            (0x1000u32, 0x2000u32, 0x2100u32, 0x2200u32, 0x3000u32, 0x4000u32);
        {
            let m = &*ram;
            let put32 = |m: &crate::dev::crime_mem::O2Memory, at: u32, v: u32| {
                m.write(at as usize, &v.to_be_bytes());
            };
            // array[1] = the SCB's bus address
            put32(m, array + 4, base + scb_addr);
            // SCB: pointers to the scatter-gather list and the CDB
            put32(m, scb_addr + scb::SG_PTR as u32, base + sg_addr);
            put32(m, scb_addr + scb::CMD_PTR as u32, base + cdb_addr);
            // The driver's control flags: disconnect allowed.
            m.write((scb_addr + scb::CONTROL as u32) as usize, &[0x40]);
            // Address the id the disk was attached at.
            m.write((scb_addr + scb::TCL as u32) as usize, &[1 << 4]);
            // INQUIRY, 64 bytes
            m.write(cdb_addr as usize, &[0x12]);
            m.write(cdb_addr as usize + 4, &[0x40]);
            // One scatter-gather entry
            put32(m, sg_addr, base + buf);
            put32(m, sg_addr + 4, 64);
        }
        // Tell the chip where the array and the completion FIFO are, the way
        // the driver does: a scratch register at a time, little end first.
        for (i, b) in (base + array).to_le_bytes().iter().enumerate() {
            c.write(sram::SCB_ARRAY + i as u32, *b as u32, 1);
        }
        for (i, b) in (base + qout).to_le_bytes().iter().enumerate() {
            c.write(sram::QOUTFIFO + i as u32, *b as u32, 1);
        }

        c.write(reg::QINFIFO, 1, 1);

        let m = &*ram;
        assert_eq!(m[buf as usize], 0x00, "a direct access device");
        assert_eq!(&m.bytes(buf as usize + 8, (buf as usize + 11) - (buf as usize + 8)), b"SGI",
                   "the inquiry data must land in the buffer the driver listed");
        assert_eq!(m[(scb_addr + scb::CONTROL as u32) as usize], 0x40,
                   "and the SCB is the driver's: nothing is written back into it");
        assert_eq!(m[qout as usize], 1, "the tag goes into the host completion FIFO");
        assert_eq!(c.read(reg::INTSTAT, 1) as u8 & intstat::CMDCMPLT, intstat::CMDCMPLT,
                   "and the driver is told to look");
    }

    /// A write goes the other way through the same scatter-gather list, and
    /// must reach the image rather than growing it.
    #[test]
    fn a_write_reaches_the_disk_and_reads_back() {
        let ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20));
        let c = on(ram.clone());
        let img = image_path("write");
        std::fs::write(&img, vec![0u8; 64 * BLOCK_SIZE]).expect("test image");
        c.bus().attach_image(1, &img, false).expect("attach");

        let base = crate::ip32::RAM_BASE;
        let (array, scb, cdb, sg, buf, qout) =
            (0x1000u32, 0x2000u32, 0x2100u32, 0x2200u32, 0x3000u32, 0x4000u32);
        let pattern: Vec<u8> = (0..BLOCK_SIZE).map(|i| (i % 251) as u8).collect();
        {
            let m = &*ram;
            let put32 = |m: &crate::dev::crime_mem::O2Memory, at: u32, v: u32| {
                m.write(at as usize, &v.to_be_bytes());
            };
            put32(m, array + 4, base + scb);
            put32(m, scb + scb::SG_PTR as u32, base + sg);
            put32(m, scb + scb::CMD_PTR as u32, base + cdb);
            m.write((scb + scb::CDB_LEN as u32) as usize, &[10]);
            m.write((scb + scb::TCL as u32) as usize, &[1 << 4]);
            // WRITE(10), one block at LBA 5.
            m.write(cdb as usize, &[0x2a]);
            m.write(cdb as usize + 5, &[5]);
            m.write(cdb as usize + 8, &[1]);
            put32(m, sg, base + buf);
            put32(m, sg + 4, BLOCK_SIZE as u32);
            m.write(buf as usize, &pattern);
        }
        for (i, b) in (base + array).to_le_bytes().iter().enumerate() {
            c.write(sram::SCB_ARRAY + i as u32, *b as u32, 1);
        }
        for (i, b) in (base + qout).to_le_bytes().iter().enumerate() {
            c.write(sram::QOUTFIFO + i as u32, *b as u32, 1);
        }
        sign(&c);
        c.write(reg::QINFIFO, 1, 1);

        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::CMDCMPLT,
                   "the write must report success, which is a plain completion");
        let on_disk = std::fs::read(&img).expect("read back");
        assert_eq!(&on_disk[5 * BLOCK_SIZE..6 * BLOCK_SIZE], &pattern[..],
                   "the block must land at the address the CDB named");
        assert!(on_disk[4 * BLOCK_SIZE..5 * BLOCK_SIZE].iter().all(|b| *b == 0),
                "and must not disturb its neighbour");
        assert_eq!(on_disk.len(), 64 * BLOCK_SIZE, "the image must not grow");
    }

    /// A transfer larger than any plausible fixed bound must still complete.
    /// A 32-entry cap looked reasonable and silently truncated a 33-page
    /// read, reporting success and leaving a hole in the middle of a loaded
    /// program.
    #[test]
    fn a_long_scatter_gather_list_is_followed_to_the_end() {
        let ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(4 << 20));
        let c = on(ram.clone());
        let img = image_path("sg");
        let pattern: Vec<u8> = (0..264 * BLOCK_SIZE).map(|i| (i % 253) as u8).collect();
        std::fs::write(&img, &pattern).expect("test image");
        c.bus().attach_image(1, &img, false).expect("attach");

        let base = crate::ip32::RAM_BASE;
        let (array, scb, cdb, sg, buf, qout) =
            (0x1000u32, 0x2000u32, 0x2100u32, 0x4000u32, 0x100000u32, 0x3000u32);
        // 264 blocks into 4 KiB pages: 33 entries, one past the old bound.
        let pages = 264 * BLOCK_SIZE / 4096;
        assert!(pages > 32, "the test must cross the bound it is checking");
        {
            let m = &*ram;
            let put32 = |m: &crate::dev::crime_mem::O2Memory, at: u32, v: u32| {
                m.write(at as usize, &v.to_be_bytes());
            };
            put32(m, array + 4, base + scb);
            put32(m, scb + scb::SG_PTR as u32, base + sg);
            put32(m, scb + scb::CMD_PTR as u32, base + cdb);
            m.write((scb + scb::CDB_LEN as u32) as usize, &[10]);
            m.write((scb + scb::TCL as u32) as usize, &[1 << 4]);
            // READ(10), 264 blocks from LBA 0.
            m.write(cdb as usize, &[0x28]);
            m.write(cdb as usize + 7, &[(264 >> 8) as u8]);
            m.write(cdb as usize + 8, &[(264 & 0xff) as u8]);
            for i in 0..pages as u32 {
                put32(m, sg + i * 8, base + buf + i * 4096);
                put32(m, sg + i * 8 + 4, 4096);
            }
        }
        for (i, b) in (base + array).to_le_bytes().iter().enumerate() {
            c.write(sram::SCB_ARRAY + i as u32, *b as u32, 1);
        }
        for (i, b) in (base + qout).to_le_bytes().iter().enumerate() {
            c.write(sram::QOUTFIFO + i as u32, *b as u32, 1);
        }
        c.write(reg::QINFIFO, 1, 1);

        let m = &*ram;
        let got = &m.bytes(buf as usize, (buf as usize + pattern.len()) - (buf as usize));
        assert_eq!(got, &pattern[..], "every page of the transfer must arrive");
    }

    /// Writing past the end is an error, not a bigger disk.
    #[test]
    fn a_write_past_the_end_fails_rather_than_growing_the_image() {
        let img = image_path("grow");
        std::fs::write(&img, vec![0u8; 8 * BLOCK_SIZE]).expect("test image");
        let c = chip();
        c.bus().attach_image(1, &img, false).expect("attach");
        // WRITE(10): four blocks at block 7 of eight.
        assert_eq!(c.execute_write_on(&[0x2a, 0, 0, 0, 0, 7, 0, 0, 4, 0], &vec![0xab; 4 * BLOCK_SIZE], 1), 0x02, "refused");
        assert_eq!(std::fs::metadata(&img).unwrap().len(), (8 * BLOCK_SIZE) as u64);
    }

    /// Memory answers both at its own base and in a low alias, and the driver
    /// uses both in one SCB. Resolving only one reads the SCB and then finds
    /// an all-zero command.
    #[test]
    fn dma_resolves_the_low_memory_alias() {
        let ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 16));
        ram.write(0x40, &[0xa5]);
        let c = on(ram);
        assert_eq!(c.dma_read8(0x40), 0xa5, "the low alias");
        assert_eq!(c.dma_read8(crate::ip32::RAM_BASE + 0x40), 0xa5, "and memory's own base");
    }

    #[test]
    fn interrupts_are_cleared_by_writing_their_bits() {
        let c = chip();
        {
            let mut st = c.st.lock();
            st.intstat = intstat::CMDCMPLT | intstat::SEQINT;
        }
        c.write(reg::ERROR_CLRINT, intstat::CMDCMPLT as u32, 1);
        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::SEQINT,
                   "clearing one interrupt must not clear the others");
    }

    type Ram = std::sync::Arc<crate::dev::crime_mem::O2Memory>;

    /// Where the HIM rig keeps things in host memory.
    const ARRAY: u32 = 0x1000;
    const QOUT: u32 = 0x4000;
    fn scb_at(tag: u8) -> u32 { 0x2000 + tag as u32 * 0x40 }
    fn cdb_at(tag: u8) -> u32 { 0x5000 + tag as u32 * 0x20 }
    fn sg_at(tag: u8) -> u32 { 0x6000 + tag as u32 * 0x10 }
    fn buf_at(tag: u8) -> u32 { 0x8000 + tag as u32 * 0x200 }

    fn put32(ram: &Ram, at: u32, v: u32) {
        ram.write(at as usize, &v.to_be_bytes());
    }

    /// The HIM's signature, as it writes it: a byte at a time, little end
    /// first.
    fn sign(c: &Aic7880) {
        for (off, v) in him::SIGNATURE {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                c.write(off + i as u32, *b as u32, 1);
            }
        }
    }

    /// A controller set up the way the kernel's HIM leaves it after init:
    /// both arrays configured, the signature in place, the next-SCB table
    /// full of 0x7f, the queue-out array full of 0xff, and its own id 0, as
    /// on an O2. One disk, at id 1.
    fn him_rig(name: &str) -> (Arc<Aic7880>, Ram) {
        let ram: Ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20));
        let c = on(ram.clone());
        let img = image_path(name);
        std::fs::write(&img, vec![0u8; 64 * BLOCK_SIZE]).expect("test image");
        c.bus().attach_image(1, &img, false).expect("attach");
        let base = crate::ip32::RAM_BASE;
        for (off, v) in [(sram::SCB_ARRAY, base + ARRAY), (sram::QOUTFIFO, base + QOUT)] {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                c.write(off + i as u32, *b as u32, 1);
            }
        }
        c.write(him::WAITING_SCB, 0xff, 1);
        c.write(him::ACTIVE_SCB, 0xff, 1);
        sign(&c);
        for off in 0x46..=0x55 {
            c.write(off, 0x7f, 1);
        }
        c.write(reg::SCSIID, 0, 1);
        ram.write(QOUT as usize, &vec![QOUTFIFO_EMPTY; (QOUT as usize + 256) - (QOUT as usize)]);
        (c, ram)
    }

    /// Build a command for `tag` in host memory, the way the HIM does: the
    /// SCB's address in the array, one scatter-gather entry marked last.
    fn put_cmd(ram: &Ram, tag: u8, target: u8, cdb: &[u8], len: u32) {
        let base = crate::ip32::RAM_BASE;
        let scb = scb_at(tag);
        put32(ram, ARRAY + tag as u32 * 4, base + scb);
        {
            let m = &*ram;
            m.write(scb as usize, &vec![0; (scb as usize + 32) - (scb as usize)]);
            m.write((scb + scb::SEG_COUNT as u32) as usize, &[1]);
            m.write((scb + scb::CDB_LEN as u32) as usize, &[cdb.len() as u8]);
            m.write((scb + scb::CONTROL as u32) as usize, &[0x40]);
            m.write((scb + scb::TCL as u32) as usize, &[target << 4]);
            m.write(cdb_at(tag) as usize, cdb);
        }
        put32(ram, scb + scb::SG_PTR as u32, base + sg_at(tag));
        // The CDB is in the low view, as the HIM points at it.
        put32(ram, scb + scb::CMD_PTR as u32, cdb_at(tag));
        put32(ram, sg_at(tag), buf_at(tag));
        put32(ram, sg_at(tag) + 4, len | SG_LAST);
    }

    fn qout(ram: &Ram) -> Vec<u8> {
        ram.bytes(QOUT as usize, (QOUT as usize + 256) - (QOUT as usize)).to_vec()
    }

    const TUR: [u8; 6] = [0x00, 0, 0, 0, 0, 0];

    /// The HIM reads one byte per completion, at its cursor (4, then down by
    /// 4) with the low bits flipped. Tags written anywhere else are never
    /// found, and every command times out.
    #[test]
    fn him_completions_land_where_the_kernel_looks() {
        let (c, ram) = him_rig("him-qout");
        put_cmd(&ram, 0, 1, &TUR, 0);
        put_cmd(&ram, 1, 1, &TUR, 0);
        c.write(reg::QINFIFO, 0, 1);
        c.write(reg::QINFIFO, 1, 1);

        let q = qout(&ram);
        assert_eq!(q[7], 0, "the first completion is at the HIM's cursor 4, flipped");
        assert_eq!(q[3], 1, "the second at cursor 0, flipped");
        assert_eq!(q[0xff], QOUTFIFO_EMPTY, "the slot after that is marked empty");
        assert!(q.iter().enumerate().all(|(i, b)| i == 3 || i == 7 || *b == QOUTFIFO_EMPTY),
                "and nothing else is touched");
        assert_eq!(c.read(sram::QOUTFIFO, 1), 0xf8, "the cursor moved down by 4 twice, wrapping");
        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::CMDCMPLT);
        assert_eq!(c.read(reg::QINCNT, 1), 0, "both were taken");
        assert_eq!(ram[(scb_at(0) + scb::CONTROL as u32) as usize], 0x40,
                   "and the SCB's control flags are left alone");
    }

    /// The HIM fills its next-SCB table with 0x7f, and one of those bytes
    /// is NetBSD's doorbell. It must not ring.
    #[test]
    fn the_him_filling_its_tables_rings_no_doorbell() {
        let c = chip();
        sign(&c);
        for off in 0x46..=0x55 {
            c.write(off, 0x7f, 1);
        }
        assert_eq!(c.read(reg::INTSTAT, 1), 0, "nothing completed");
        assert!(c.executed().is_empty());
        assert!(!c.notes().iter().any(|n| n.starts_with("nb tag")), "nothing ran");
        assert!(c.dma_misses.lock().unwrap().is_empty(), "and nothing was fetched");
    }

    /// NetBSD's doorbell, with its rings in memory and no HIM, still runs
    /// a command.
    #[test]
    fn the_netbsd_doorbell_still_runs_commands() {
        let ram: Ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20));
        let c = on(ram.clone());
        let img = image_path("nb");
        std::fs::write(&img, vec![0u8; 64 * BLOCK_SIZE]).expect("test image");
        c.bus().attach_image(1, &img, false).expect("attach");
        let (hscbs, shared) = (0x2000u32, 0x4000u32);
        let tag = 3u8;
        {
            let m = &*ram;
            m.write((shared + nb::FIFO_LEN) as usize, &[tag]);
            let h = (hscbs + tag as u32 * nb::HSCB_STRIDE) as usize;
            m.write(h + nb::CDB_LEN as usize, &[6]); // TEST UNIT READY
            m.write(h + nb::SCSIID as usize, &[0x17]);
            m.write(h + nb::STATUS as usize, &[0xaa]);
        }
        for (off, v) in [(nb::HSCB_ADDR, hscbs), (nb::SHARED_DATA_ADDR, shared)] {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                c.write(off + i as u32, *b as u32, 1);
            }
        }
        c.write(nb::KERNEL_QINPOS, 1, 1);
        let m = &*ram;
        assert_eq!(m[shared as usize], tag, "the tag comes back on the queue-out ring");
        assert_eq!(m[(hscbs + tag as u32 * nb::HSCB_STRIDE + nb::STATUS) as usize], 0, "with GOOD status");
        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::CMDCMPLT);
    }

    /// One NetBSD command, tag 3, through the doorbell: `cdb` with its first
    /// segment (`dataptr`, `datacnt` with flags) in the SCB and SGPTR at
    /// `sgptr`. Returns the chip; the caller inspects `ram`.
    fn netbsd_command(ram: &Ram, img_blocks: &[u8], cdb: &[u8], dataptr: u32, datacnt: u32, sgptr: u32) -> Arc<Aic7880> {
        let c = on(ram.clone());
        // One file per call: these tests run in parallel in one process.
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let img = TempImage(std::env::temp_dir().join(format!("iris-aic7880-nb-{}-{n}.img", std::process::id())));
        std::fs::write(&img, img_blocks).expect("test image");
        c.bus().attach_image(1, &img, false).expect("attach");
        let (hscbs, shared, tag) = (0x2000u32, 0x4000u32, 3u8);
        {
            let m = &*ram;
            m.write((shared + nb::FIFO_LEN) as usize, &[tag]);
            let h = (hscbs + tag as u32 * nb::HSCB_STRIDE) as usize;
            m.write(h + nb::CDB as usize, cdb);
            m.write(h + nb::CDB_LEN as usize, &[cdb.len() as u8]);
            m.write(h + nb::DATAPTR as usize, &dataptr.to_le_bytes());
            m.write(h + nb::DATACNT as usize, &datacnt.to_le_bytes());
            m.write(h + nb::SGPTR as usize, &sgptr.to_le_bytes());
            m.write(h + nb::SCSIID as usize, &[0x17]);
        }
        for (off, v) in [(nb::HSCB_ADDR, hscbs), (nb::SHARED_DATA_ADDR, shared)] {
            for (i, b) in v.to_le_bytes().iter().enumerate() {
                c.write(off + i as u32, *b as u32, 1);
            }
        }
        c.write(nb::KERNEL_QINPOS, 1, 1);
        c
    }

    /// The driver's first segment can be its whole list (last flag set in
    /// DATACNT); SGPTR then points at a list entry left over from an
    /// earlier command. A device answering more than fits must not spill
    /// into that entry's buffer: that put READ TOC's lead-out bytes into
    /// NetBSD's file-system metadata.
    #[test]
    fn a_one_segment_netbsd_command_ends_at_its_segment() {
        let ram: Ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20));
        {
            let m = &*ram;
            // The stale entry 1: a 256-byte buffer at 0x8000, holding 0x5a.
            m.write(0x7000, &0x8000u32.to_le_bytes());
            m.write(0x7004, &(0x100u32 | nb::LAST_SEG).to_le_bytes());
            m.write(0x8000, &[0x5a; 0x100]);
            m.write(0x6000, &[0x77; 16]);
        }
        // INQUIRY for 36 bytes into an 8-byte segment marked last.
        let c = netbsd_command(&ram, &vec![0u8; 64 * BLOCK_SIZE], &[0x12, 0, 0, 0, 36, 0],
            0x6000, 8 | nb::LAST_SEG, 0x7000);
        let m = &*ram;
        assert!(m.bytes(0x8000, 0x100).iter().all(|&b| b == 0x5a), "nothing lands in the stale entry's buffer");
        assert_eq!(m.bytes(0x6008, 8), [0x77; 8], "nothing past the segment either");
        assert_ne!(m.bytes(0x6000, 8), [0x77; 8], "the segment itself is filled");
        assert!(c.notes().iter().any(|n| n.starts_with("nb overrun")), "the overrun is reported");
    }

    /// A real two-entry list still fills both buffers in order.
    #[test]
    fn a_two_segment_netbsd_read_fills_both_buffers() {
        let ram: Ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::single(1 << 20));
        {
            let m = &*ram;
            m.write(0x7000, &0x9000u32.to_le_bytes());
            m.write(0x7004, &(BLOCK_SIZE as u32 | nb::LAST_SEG).to_le_bytes());
        }
        let mut img = vec![0u8; 64 * BLOCK_SIZE];
        img[..BLOCK_SIZE].fill(0x11);
        img[BLOCK_SIZE..2 * BLOCK_SIZE].fill(0x22);
        // READ(10) of blocks 0 and 1.
        netbsd_command(&ram, &img, &[0x28, 0, 0, 0, 0, 0, 0, 0, 2, 0], 0x6000, BLOCK_SIZE as u32, 0x7000);
        let m = &*ram;
        assert!(m.bytes(0x6000, BLOCK_SIZE).iter().all(|&b| b == 0x11));
        assert!(m.bytes(0x9000, BLOCK_SIZE).iter().all(|&b| b == 0x22));
    }

    /// While paused, the queue-in FIFO holds what it is given, and the host
    /// can pop and push to reorder it: that is how the HIM puts a REQUEST
    /// SENSE at the head. Restarting runs them in the new order.
    #[test]
    fn a_paused_queue_holds_and_rotates() {
        let (c, ram) = him_rig("him-rotate");
        put_cmd(&ram, 0, 1, &TUR, 0);
        put_cmd(&ram, 1, 1, &TUR, 0);
        c.write(reg::HCNTRL, hcntrl::PAUSE as u32, 1);
        c.write(reg::QINFIFO, 0, 1);
        c.write(reg::QINFIFO, 1, 1);
        assert_eq!(c.read(reg::QINCNT, 1), 2, "nothing runs while paused");
        assert!(c.executed().is_empty());

        let head = c.read(reg::QINFIFO, 1);
        assert_eq!(head, 0, "reading takes the head");
        assert_eq!(c.read(reg::QINCNT, 1), 1);
        c.write(reg::QINFIFO, head, 1);
        assert_eq!(c.read(reg::QINCNT, 1), 2, "and writing puts it at the back");

        c.write(reg::HCNTRL, hcntrl::INTEN as u32, 1);
        let order: Vec<u8> = c.executed().iter().map(|e| e.0).collect();
        assert_eq!(order, vec![1, 0], "restarting runs them in the new order");
        assert_eq!(c.read(reg::QINCNT, 1), 0);
    }

    /// Nobody at an id: the selection times out, the way the HIM expects,
    /// and nothing goes to the queue-out.
    #[test]
    fn an_empty_id_times_out_its_selection() {
        let (c, ram) = him_rig("him-selto");
        let (irq, line) = crate::irq::LevelProbe::line();
        c.connect_irq(line);
        let inquiry = [0x12, 0, 0, 0, 36, 0];
        put_cmd(&ram, 2, 3, &inquiry, 36);
        c.write(reg::QINFIFO, 2, 1);

        assert_eq!(c.read(reg::INTSTAT, 1), intstat::SCSIINT as u32);
        assert_ne!(c.read(reg::SSTAT1, 1) as u8 & sstat1::SELTO, 0, "SSTAT1 says selection timeout");
        assert_eq!(c.read(him::WAITING_SCB, 1), 2, "for the tag being selected");
        assert_ne!(c.read(reg::HCNTRL, 1) as u8 & hcntrl::PAUSE, 0, "and the sequencer has stopped");
        assert!(c.interrupting() && irq.level(), "the pin follows the chip");
        assert!(qout(&ram).iter().all(|b| *b == QOUTFIFO_EMPTY), "nothing completed");
        assert!(ram.bytes(buf_at(2) as usize, (buf_at(2) as usize + 36) - (buf_at(2) as usize)).iter().all(|b| *b == 0),
                "and no data moved");

        // A restart before the interrupt is cleared does not take.
        c.write(reg::HCNTRL, hcntrl::INTEN as u32, 1);
        assert_ne!(c.read(reg::HCNTRL, 1) as u8 & hcntrl::PAUSE, 0);

        c.write(reg::SSTAT1, 0x88, 1);
        c.write(reg::ERROR_CLRINT, intstat::SCSIINT as u32, 1);
        c.write(reg::HCNTRL, hcntrl::INTEN as u32, 1);
        assert_eq!(c.read(reg::INTSTAT, 1), 0);
        assert_eq!(c.read(reg::SSTAT1, 1), 0);
        assert_eq!(c.read(reg::HCNTRL, 1) as u8 & hcntrl::PAUSE, 0, "and it runs again");
        assert!(!c.interrupting() && !irq.level());
    }

    /// A failed command stops the sequencer with its status in scratch RAM.
    /// The HIM turns the same SCB into a REQUEST SENSE and queues it at the
    /// head; that one completes normally, carrying the reason.
    #[test]
    fn a_check_condition_goes_through_the_hims_auto_sense() {
        let (c, ram) = him_rig("him-sense");
        let tag = 5u8;
        put_cmd(&ram, tag, 1, &[0xc0, 0, 0, 0, 0, 0], 0);
        c.write(reg::QINFIFO, tag as u32, 1);

        assert_eq!(c.read(reg::INTSTAT, 1), 0x51, "SEQINT with the bad-status code");
        assert_eq!(c.read(him::ACTIVE_SCB, 1), tag as u32, "for the tag connected");
        assert_eq!(c.read(him::STATUS, 1), 0x02, "with CHECK CONDITION handed over");
        assert_ne!(c.read(reg::HCNTRL, 1) as u8 & hcntrl::PAUSE, 0);
        assert!(qout(&ram).iter().all(|b| *b == QOUTFIFO_EMPTY), "nothing completed");

        // The HIM rewrites the SCB as REQUEST SENSE, pointed at its own
        // sense list just after the SCB, and queues the tag again.
        let sense_list = scb_at(tag) + 32;
        let sense_buf = 0x9000u32;
        put32(&ram, sense_list, sense_buf);
        put32(&ram, sense_list + 4, 18 | SG_LAST);
        put32(&ram, scb_at(tag) + scb::SG_PTR as u32, crate::ip32::RAM_BASE + sense_list);
        ram.write(cdb_at(tag) as usize, &[0x03, 0, 0, 0, 18, 0]);
        c.write(reg::QINFIFO, tag as u32, 1);
        assert_eq!(c.read(reg::QINCNT, 1), 1, "held while the sequencer is stopped");

        c.write(reg::ERROR_CLRINT, intstat::SEQINT as u32, 1);
        assert_eq!(c.read(reg::INTSTAT, 1), 0, "clearing SEQINT clears its code too");
        c.write(reg::HCNTRL, hcntrl::INTEN as u32, 1);

        let m = &*ram;
        assert_eq!(m[sense_buf as usize + 2], 0x05, "ILLEGAL REQUEST");
        assert_eq!(m[sense_buf as usize + 12], 0x20, "invalid command operation code");
        assert_eq!(qout(&ram)[7], tag, "and the tag completes through the queue-out");
        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::CMDCMPLT);
        assert_eq!(c.read(reg::QINCNT, 1), 0);
    }

    /// The PROM's build of the program keeps the answers it boots with
    /// today: an empty id is "no device" with GOOD status, and completes.
    #[test]
    fn the_proms_program_keeps_its_old_answers() {
        let (c, ram) = him_rig("him-prom");
        c.write(reg::SEQCTL, seqctl::LOADRAM as u32, 1);
        for _ in 0..him::PROM_PROGRAM_LEN {
            c.write(reg::SEQRAM, 0, 1);
        }
        c.write(reg::SEQCTL, 0, 1);
        put_cmd(&ram, 1, 3, &[0x12, 0, 0, 0, 36, 0], 36);
        c.write(reg::QINFIFO, 1, 1);
        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::CMDCMPLT);
        assert_eq!(ram[buf_at(1) as usize], 0x7f, "nothing here");
        assert_eq!(qout(&ram)[7], 1, "completed where the HIM looks");
    }

    /// CLRSINT0 and CLRSINT1 share addresses with SSTAT0 and SSTAT1. A
    /// channel reset writes 0xff to both, and reading that back said an
    /// external bus reset had happened.
    #[test]
    fn clearing_scsi_status_leaves_it_clear() {
        let c = chip();
        c.write(reg::SSTAT1, 0xff, 1);
        assert_eq!(c.read(reg::SSTAT1, 1), 0);
        c.write(reg::SSTAT0, 0xff, 1);
        assert_eq!(c.read(reg::SSTAT0, 1), 0);
        {
            let mut st = c.st.lock();
            st.sstat0 = 0xff;
            st.sstat1 = 0xff;
        }
        c.write(reg::SSTAT1, 0xff, 1);
        assert_eq!(c.read(reg::SSTAT1, 1), 0, "a clear clears");
        c.write(reg::SSTAT0, 0xff, 1);
        assert_eq!(c.read(reg::SSTAT0, 1), 0);

        c.write(reg::SCSISIG, 0xff, 1);
        assert_eq!(c.read(reg::SCSISIG, 1) as u8 & scsisig::REQ, 0, "REQ never reads as set");
    }

    /// Failed writes say why, so the REQUEST SENSE after them is not "no
    /// sense".
    #[test]
    fn failed_writes_carry_sense() {
        let c = chip();
        let img = image_path("wsense");
        std::fs::write(&img, vec![0u8; 8 * BLOCK_SIZE]).expect("test image");
        c.bus().attach_image(1, &img, false).expect("attach");
        c.bus().attach_image(4, &img, true).expect("attach CD");
        let write10 = |lba: u8| [0x2a, 0, 0, 0, 0, lba, 0, 0, 1, 0];
        let block = vec![0u8; BLOCK_SIZE];

        let sense = |t: u8| {
            let (d, status) = c.execute_on(&[0x03, 0, 0, 0, 18, 0], t);
            assert_eq!(status, 0x00);
            (d[2] & 0x0f, d[12], d[13])
        };
        assert_eq!(c.execute_write_on(&write10(0), &block, 4), 0x02);
        assert_eq!(sense(4), (0x07, 0x27, 0x00), "a CD is write protected");
        assert_eq!(c.execute_write_on(&write10(8), &block, 1), 0x02);
        assert_eq!(sense(1), (0x05, 0x21, 0x00), "past the end is out of range");
    }

    /// The controller's own sense follows SCSI-2 like a device's: an
    /// allocation length of 0 asks for 4 bytes.
    #[test]
    fn controller_sense_honours_the_allocation_length() {
        let c = chip();
        *c.sense.lock().unwrap() = Some((0x05, 0x25, 0x00));
        let (d, status) = c.execute_on(&[0x03, 0, 0, 0, 0, 0], 1);
        assert_eq!(status, 0x00);
        assert_eq!(d.len(), 4, "SCSI-2: 0 means 4 bytes");
        assert_eq!(d[2], 0x05);
        *c.sense.lock().unwrap() = Some((0x05, 0x25, 0x00));
        assert_eq!(c.execute_on(&[0x03, 0, 0, 0, 14, 0], 1).0.len(), 14);
    }

    /// Wait up to five seconds for the worker to make `done` true.
    fn eventually(done: impl Fn() -> bool) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if done() {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        done()
    }

    /// With its own thread the controller runs what is queued without the
    /// writer waiting for it, and posts results the way it does inline.
    #[test]
    fn the_worker_runs_queued_commands() {
        use crate::traits::Device;
        let (c, ram) = him_rig("worker");
        let irq = std::sync::Arc::new(crate::irq::LevelProbe::default());
        c.connect_irq(crate::irq::IrqLine::new(irq.clone(), 0));
        put_cmd(&ram, 0, 1, &TUR, 0);
        put_cmd(&ram, 1, 1, &TUR, 0);
        c.start();
        assert!(c.is_running());
        c.write(reg::QINFIFO, 0, 1);
        c.write(reg::QINFIFO, 1, 1);
        assert!(eventually(|| c.totals().0 == 2 && qout(&ram)[3] == 1), "both commands complete");
        assert_eq!(qout(&ram)[7], 0, "in the order queued");
        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::CMDCMPLT);
        assert!(irq.level(), "the worker drives the pin");
        c.write(reg::ERROR_CLRINT, intstat::CMDCMPLT as u32, 1);
        assert!(!irq.level(), "and the host's clear drops it");
        c.stop();
        assert!(!c.is_running());
    }

    /// A paused sequencer takes nothing, worker or not; the host can count
    /// and rotate what it queued until it lets the sequencer go.
    #[test]
    fn the_worker_waits_while_the_host_holds_the_sequencer() {
        use crate::traits::Device;
        let (c, ram) = him_rig("worker-paused");
        put_cmd(&ram, 1, 1, &TUR, 0);
        c.start();
        c.write(reg::HCNTRL, hcntrl::PAUSE as u32, 1);
        c.write(reg::QINFIFO, 1, 1);
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(c.read(reg::QINCNT, 1), 1, "still queued");
        assert_eq!(c.totals().0, 0, "nothing ran");
        c.write(reg::HCNTRL, hcntrl::INTEN as u32, 1);
        assert!(eventually(|| qout(&ram)[7] == 1), "it runs once unpaused");
        c.stop();
    }

    /// A command that finishes while the host has the sequencer paused is
    /// not posted under the host's feet: it waits for the unpause.
    #[test]
    fn a_result_that_finds_the_sequencer_paused_is_posted_on_unpause() {
        let (c, ram) = him_rig("held");
        c.write(reg::HCNTRL, hcntrl::PAUSE as u32, 1);
        c.st.lock().held.push_back(Done::Complete { tag: 0, target: 1 });
        c.write(reg::SCBPTR, 0, 1);
        assert!(qout(&ram).iter().all(|b| *b == QOUTFIFO_EMPTY), "nothing posted while paused");
        assert_eq!(c.read(reg::INTSTAT, 1), 0);
        c.write(reg::HCNTRL, hcntrl::INTEN as u32, 1);
        assert_eq!(qout(&ram)[7], 0, "posted where the HIM looks");
        assert_eq!(c.read(reg::INTSTAT, 1) as u8, intstat::CMDCMPLT);
        assert!(c.st.lock().held.is_empty());
    }

    /// The machine fills the bus from `[scsi.N]` and gives it to the
    /// controller, which answers for what is on it and still takes a target
    /// attached later through the same bus.
    #[test]
    fn a_controller_built_on_a_filled_bus_answers_for_its_target() {
        let img = TempImage(std::env::temp_dir().join(format!("iris-aic7880-filled-{}.img", std::process::id())));
        std::fs::write(&img, vec![0u8; 64 * 512]).unwrap();
        let bus = Aic7880::new_bus();
        bus.attach_image(3, &img, false).unwrap();
        let c = Aic7880::new_at(1, bus);
        assert!(c.bus().has_target(3));
        assert!(!c.bus().has_target(4));
        let inquiry = crate::scsi::ScsiRequest {
            cdb: vec![crate::scsi::scsi_cmd::INQUIRY, 0, 0, 0, 36, 0],
            data_len: crate::scsi::ScsiDataLength::Unlimited,
            data_in: None,
        };
        let r = c.bus().request(3, &inquiry).expect("target answers").unwrap();
        assert_eq!(&r.data[8..16], b"SGI     ");
        assert_eq!(&r.data[16..32], b"IRIS EMUL DISK  ");
        c.bus().attach_image(4, &img, false).unwrap();
        assert!(c.bus().has_target(4), "a target attached after boot goes on the same bus");
    }
}
