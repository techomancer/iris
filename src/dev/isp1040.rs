//! QLogic ISP1040 SCSI host adapter, emulated above its RISC.
//!
//! The driver hands whole SCSI commands to a request ring in host memory and
//! collects completions from a response ring. Everything else is a mailbox
//! protocol of sixteen-bit registers. The microcode a driver downloads is
//! retained (so a checksum can be verified and read back) and never run: the
//! SCSI work is done here, against the shared target model, and the
//! completion is posted the way the firmware would.
//!
//! The register and ring contract follow the part's datasheet and the public
//! Linux and OpenBSD drivers. Where the model approximates, it says so:
//!
//! * Commands run one at a time, in ring order, on the adapter's own thread:
//!   the doorbell write returns at once and the completion and its
//!   interrupt follow when the target has answered, as the part's would.
//!   No abort/recovery mailbox commands exist (the ISP1040 does not carry
//!   them; they answer "invalid"). A soft reset cancels the command in
//!   flight without waiting for its target: no completion is posted, and
//!   what it had not yet stored in host memory is not stored. A queue
//!   initialisation does not cancel it: the command ran once, and its
//!   completion goes to the response ring in place when it finishes.
//! * A completion with no room in the response ring is held and posted when
//!   the driver acknowledges a slot; the request ring is not consumed while
//!   any completion is held.
//! * The target model has no bus reset, so a bus reset or soft reset does not
//!   arm a unit attention on the disk as QEMU's SCSI devices do. A bus reset
//!   retires a selection still waiting on an absent target at once, with the
//!   reset completion the public drivers handle for a command the reset
//!   destroyed. Inferred: the drivers establish only that they handle that
//!   completion, not that the firmware sends it for a command still in
//!   selection, and whether it also raises the bus-reset asynchronous event
//!   for a driver-issued reset is unknown (the ISP1040 datasheet's
//!   set-selection-timeout table grades both).
//! * A bus reset also starts the bus's hold-off: for mailbox 1 seconds the
//!   chip starts no SCSI activity on it (the datasheet's "Bus-reset
//!   hold-off"). Established only for the NVRAM setting of that name; that the
//!   mailbox value does the same is inferred. The rest are model choices:
//!   a command for the quiet bus waits and starts, oldest first, when the
//!   hold ends (refusing it would be as well supported); the other bus and
//!   mailbox commands carry on (inferred, not established); nothing is
//!   posted when the hold ends; a
//!   delay of 0 holds nothing and a later reset replaces the hold; the delay
//!   is not clamped to the vendor's 0-15 s; an absent target then waits its
//!   selection timeout from the end of the hold; a second reset retires the
//!   parked commands with the reset completion; a soft reset clears the hold
//!   and drops them, and a new request ring drops them with the hold kept.
//!   Request-out stays at the oldest parked entry, because the entry is read
//!   again when the hold ends: an approximation, a chip that had fetched it
//!   would plausibly move on. A parked entry the guest rewrites meanwhile is
//!   answered incomplete. A snapshot carries each bus's remaining hold and
//!   the parked positions. The ISP12160 would widen `BUSES` and decode the
//!   bus from mailbox 2 and its request entries; the per-bus state is there.
//! * Only logical unit 0 of a present target exists. A command for another
//!   unit is passed to the target with that unit number, which answers as
//!   SCSI says (INQUIRY: no device at this unit; anything else: check
//!   condition, logical unit not supported). An absent target times out:
//!   its completion is posted once the selection timeout the driver
//!   programmed has passed on the adapter's clock. The part is only known to
//!   wait at least that long; the model answers exactly at the value. A
//!   programmed value of zero is no wait, and a queue initialisation leaves
//!   a waiting selection alone: both are model choices, the datasheet being
//!   silent. That clock is host wall time less the time
//!   the machine was stopped; iris has no emulated-time source a device can
//!   read, so the wait does not follow the guest's instruction rate.
//!
//! Register windows: BAR 0 is I/O space and BAR 1 memory space, both 256
//! bytes onto the same sixteen-bit registers. Accesses of any other width
//! are not decoded: they read zero and writes are dropped, noted for the
//! first sixteen, where the QEMU model fails the access.
//!
//! Ported from hw/scsi/qlogic-isp.c in jamesbraid/qemu-sgi-origami.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::devlog::LogModule;
use crate::irq::IrqLine;
use crate::pci::{Bar, PciBusDevice, PciConfig, PciDma, PCI_INTA};
use crate::scsi::{scsi_cmd, ScsiBus, ScsiDataLength, ScsiIdentities, ScsiRequest, CDROM_IDENTITY, DISK_IDENTITY};
use crate::snapshot::{get_field, hex_u16, hex_u32, load_u16_slice, toml_u16, toml_u32, toml_u8, u16_slice_to_toml, u8_slice_to_toml};

pub const PCI_VENDOR_QLOGIC: u16 = 0x1077;
pub const PCI_DEVICE_ISP1040: u16 = 0x1020;
pub const PCI_REVISION_ISP1040: u8 = 0x05;
/// Mass storage controller, SCSI.
pub const PCI_CLASS_SCSI: u32 = 0x0001_0000;

pub const BLOCK_SIZE: usize = 512;

/// Register offsets.
pub mod reg {
    pub const ID_LOW: u32 = 0x00;
    pub const ID_HIGH: u32 = 0x02;
    pub const BUS_CFG0: u32 = 0x04;
    pub const BUS_CFG1: u32 = 0x06;
    pub const BUS_CTRL: u32 = 0x08;
    pub const BUS_STATUS: u32 = 0x0a;
    pub const SEMAPHORE: u32 = 0x0c;
    pub const NVRAM: u32 = 0x0e;
    pub const DMA_CMD_CFG: u32 = 0x20;
    pub const DMA_CMD_CTRL: u32 = 0x22;
    pub const DMA_DATA_CFG: u32 = 0x40;
    pub const DMA_DATA_CTRL: u32 = 0x42;
    pub const MAILBOX_BASE: u32 = 0x70;
    pub const REQUEST_QUEUE: u32 = MAILBOX_BASE + 0x08;
    pub const RESPONSE_QUEUE: u32 = MAILBOX_BASE + 0x0a;
    pub const BANK_BASE: u32 = 0x80;
    pub const HOST_CMD: u32 = 0xc0;
    pub const WINDOW_SIZE: u32 = 0x100;
}

const MAILBOX_COUNT: usize = 8;
const MAILBOX_END: u32 = reg::MAILBOX_BASE + MAILBOX_COUNT as u32 * 2;
const BANK_END: u32 = 0xc0;
const BANK_WORDS: usize = ((BANK_END - reg::BANK_BASE) / 2) as usize;
const MBOX_REQ: usize = 4;
const MBOX_RSP: usize = 5;

pub const CFG1_FIFO_MASK: u16 = 0x0070;
pub const CFG1_BANK_SELECT: u16 = 0x0008;
pub const CFG1_BURST_ENABLE: u16 = 0x0004;
const CFG1_WRITE_MASK: u16 = CFG1_FIFO_MASK | CFG1_BANK_SELECT | CFG1_BURST_ENABLE;
const DMA_CFG_WRITE_MASK: u16 = 0x0002;

pub const CTRL_SOFT_RESET: u16 = 0x0001;
pub const CTRL_INT_ENABLE: u16 = 0x0002;
pub const CTRL_RISC_INT_ENABLE: u16 = 0x0004;
const CTRL_WRITE_MASK: u16 = CTRL_INT_ENABLE | CTRL_RISC_INT_ENABLE;

pub const STATUS_INT_REQ: u16 = 0x0002;
pub const STATUS_RISC_INT_REQ: u16 = 0x0004;

pub const SEMAPHORE_LOCK: u16 = 0x0001;
const SEMAPHORE_MASK: u16 = 0x0003;

// Data-in is never asserted: every EEPROM read fails validation and the
// driver falls back to its defaults, as with an unprogrammed part.
const NVRAM_WRITE_MASK: u16 = 0x0007;

pub const SIG1: u16 = 0x4953;
pub const SIG2: u16 = 0x5020;
pub const SIG3: u16 = 0x2020;
pub const SIG4: u16 = 0x0001;

/// Mailbox command codes and completion codes.
pub mod mbox {
    pub const NOP: u16 = 0x0000;
    pub const LOAD_RAM: u16 = 0x0001;
    pub const EXEC_FIRMWARE: u16 = 0x0002;
    pub const DUMP_RAM: u16 = 0x0003;
    pub const WRITE_RAM_WORD: u16 = 0x0004;
    pub const READ_RAM_WORD: u16 = 0x0005;
    pub const REGISTER_TEST: u16 = 0x0006;
    pub const VERIFY_CHECKSUM: u16 = 0x0007;
    pub const ABOUT_FIRMWARE: u16 = 0x0008;
    pub const LOAD_RAM_A64: u16 = 0x0009;
    pub const INIT_REQ_QUEUE: u16 = 0x0010;
    pub const INIT_RSP_QUEUE: u16 = 0x0011;
    pub const ABORT_COMMAND: u16 = 0x0015;
    pub const BUS_RESET: u16 = 0x0018;
    pub const GET_FIRMWARE_STATUS: u16 = 0x001f;
    pub const GET_TARGET_PARAMETERS: u16 = 0x0028;
    pub const SET_INITIATOR_ID: u16 = 0x0030;
    pub const SET_SELECTION_TIMEOUT: u16 = 0x0031;
    pub const SET_RETRY_COUNT: u16 = 0x0032;
    pub const SET_TAG_AGE_LIMIT: u16 = 0x0033;
    pub const SET_CLOCK_RATE: u16 = 0x0034;
    pub const SET_ACTIVE_NEGATION: u16 = 0x0035;
    pub const SET_ASYNC_DATA_SETUP: u16 = 0x0036;
    pub const SET_PCI_CONTROL: u16 = 0x0037;
    pub const SET_TARGET_PARAMETERS: u16 = 0x0038;
    pub const SET_DEVICE_QUEUE: u16 = 0x0039;
    pub const SET_RESET_DELAY_PARAMS: u16 = 0x003a;
    pub const SET_SYSTEM_PARAMETER: u16 = 0x0045;
    pub const SET_FIRMWARE_FEATURES: u16 = 0x004a;
    pub const INIT_REQ_QUEUE_A64: u16 = 0x0052;
    pub const INIT_RSP_QUEUE_A64: u16 = 0x0053;
    pub const SET_DATA_OVERRUN_RECOVERY: u16 = 0x005a;

    pub const COMPLETE: u16 = 0x4000;
    pub const INVALID: u16 = 0x4001;
    pub const HOST_INTERFACE_ERROR: u16 = 0x4002;
    pub const TEST_FAILED: u16 = 0x4003;
    pub const PARAM_ERROR: u16 = 0x4006;
}

const RAM_WORDS: usize = 0x10000;
/// The command budget the adapter announces in its firmware status.
pub const MAX_REQUESTS: u16 = 512;
const FW_VERSION: [u16; 3] = [1, 0, 0];

pub const HOST_CMD_SHIFT: u32 = 12;
pub const RISC_FLAG_HOST_INT: u16 = 0x0080;
pub const RISC_FLAG_RESET: u16 = 0x0040;
pub const RISC_FLAG_PAUSED: u16 = 0x0020;

pub const ENTRY_SIZE: usize = 64;
pub const ENTRY_TYPE_REQUEST: u8 = 0x01;
pub const ENTRY_TYPE_STATUS: u8 = 0x03;
pub const ENTRY_TYPE_MARKER: u8 = 0x04;
pub const ENTRY_TYPE_COMMAND_A64: u8 = 0x09;

const REQ_OFF_HANDLE: u64 = 4;
const REQ_OFF_DEVICE: u64 = 8;
const REQ_OFF_CDBLEN: u64 = 10;
const REQ_OFF_SEGCOUNT: u64 = 18;
const REQ_OFF_CDB: u64 = 20;
const CDB_FIELD_LEN: usize = 12;
const REQ_OFF_SEGS: u64 = 32;
const REQ_SEGS_INLINE: usize = 4;
const SEG_SIZE: u64 = 8;
const CONT_OFF_SEGS: u64 = 8;
const CONT_SEGS: usize = 7;
const A64_OFF_SEGS: u64 = 40;
const A64_SEGS_INLINE: usize = 2;
const A64_SEG_SIZE: u64 = 12;
const A64_CONT_OFF_SEGS: u64 = 4;
const A64_CONT_SEGS: usize = 5;

const SENSE_MAX: usize = 32;
pub const SCSI_STATUS_SENSE_VALID: u16 = 0x0200;

pub const IOCB_COMPLETE: u16 = 0x0000;
pub const IOCB_INCOMPLETE: u16 = 0x0001;
pub const IOCB_RESET: u16 = 0x0004;
pub const IOCB_DATA_UNDERRUN: u16 = 0x0015;

pub const STATE_GOT_BUS: u16 = 0x0100;
pub const STATE_GOT_SENSE: u16 = 0x2000;
/// Every phase a completed command traverses; the ISP1040 reports this for
/// all of them, gated only on whether sense was collected.
pub const STATE_ALL_PHASES: u16 = 0x7f00;
pub const STATUS_FLAG_TIMEOUT: u16 = 0x0040;

/// What one unit of the selection timeout (mailbox command 0x0031, mailbox
/// 1) lasts. Milliseconds is inferred, not established: three public drivers
/// default the value to 250 and Sun's isp(7D) page says milliseconds, but no
/// vendor document is held.
const SELECTION_TIMEOUT_UNIT: Duration = Duration::from_millis(1);
/// The selection timeout in force until the driver programs one, in
/// [`SELECTION_TIMEOUT_UNIT`]s: qla1280's default. The chip's own power-on
/// default is unknown, so this is an approximation that only affects a
/// driver that never programs a value.
const DEFAULT_SELECTION_TIMEOUT: u16 = 250;

/// What one unit of the bus-reset delay (mailbox command 0x0018, mailbox 1)
/// lasts. Seconds is inferred from the vendor's NVRAM setting, which the
/// drivers pass through; no firmware interface document is held. The
/// datasheet puts the hold on the guest's virtual clock; iris has no such
/// source for a device, so it runs on the adapter's clock, host wall time
/// less the time the machine was stopped, like the selection timeout.
const BUS_RESET_DELAY_UNIT: Duration = Duration::from_secs(1);
/// The SCSI buses this part drives. The hold-off, the parked commands and the
/// selection are all kept per bus, so a two-bus part widens this and decodes
/// the bus from its own request entries and from mailbox 2.
const BUSES: usize = 1;

const CHECK_CONDITION: u8 = 0x02;
const MAX_TARGET: u8 = 15;
const MAX_LUN: u8 = 7;

/// A completion held for response-ring room.
#[derive(Clone, Default, PartialEq, Debug)]
struct Pending {
    handle: u32,
    scsi_status: u16,
    completion: u16,
    state_flags: u16,
    status_flags: u16,
    resid: u32,
    sense: Vec<u8>,
}

/// What the request-ring thread and the register side tell each other.
#[derive(Default)]
struct Ctl {
    /// Bumped by every reset: a command that finishes under another
    /// generation was cancelled with the chip, and posts nothing.
    gen: u64,
    /// Bumped when the request ring is replaced (and by a reset): a command
    /// read from the old ring does not advance the new one's consumer.
    rqst_epoch: u64,
    /// Bumped when the response ring is replaced (and by a reset): an entry
    /// written into the old ring is written again into the new one.
    resp_epoch: u64,
    /// The ring may have entries the thread has not looked at.
    work: bool,
    /// The thread is walking the ring.
    walking: bool,
    /// A command is running with the state unlocked.
    busy: bool,
    /// Holders that need the thread not to start another command
    /// (saving and loading state).
    hold: u32,
    /// The machine is stopped: the thread takes no further entries.
    paused: bool,
    shutdown: bool,
    /// The selection of an absent target the thread is waiting out. It
    /// survives the thread being parked meanwhile.
    selecting: Option<Selection>,
    /// When the adapter's clock last stopped with the machine, and how long
    /// it has stood still in all.
    stopped_at: Option<Instant>,
    stopped_ns: i64,
}

/// A selection of an absent target in progress.
#[derive(Clone, Debug)]
struct Selection {
    /// The request ring consumer position of its entry.
    cons: u32,
    /// When the adapter's clock says the selection times out.
    due: i64,
    /// The completion to post then.
    done: Pending,
    /// The bus it is selecting on: a bus reset retires only that bus's.
    bus: u8,
    /// The driver replaced the request ring after the entry was read: the
    /// command ran once all the same, but its entry is not in the new ring.
    orphan: bool,
}

/// The request ring as a step of the walk found it, for the part of the
/// step done with the state unlocked.
#[derive(Clone, Copy)]
struct Ring {
    base: u64,
    len: u32,
    cons: u32,
    epoch: u64,
    gen: u64,
}

/// One request entry, read and ready to run against its target.
struct Job {
    handle: u32,
    target: u8,
    cdb: Vec<u8>,
    xfer: usize,
    to_dev: bool,
    segs: Vec<(u64, u32)>,
    described: u64,
}

/// What carrying out a request entry came to.
enum Outcome {
    /// The thread must look at the ring again first.
    Again,
    /// The entry could not be read: the walk stops on it.
    Stop,
    /// Done, with the completion to post if any.
    Done(Option<Pending>),
}

/// What reading a request entry came to.
enum Prepared {
    /// The fabric refused the entry; the walk stops on it.
    Refused,
    /// Consumed with no answer (a marker).
    Skipped,
    /// Answered without reaching a target.
    Answered(Pending),
    /// Answered when selecting an absent target on `bus` times out.
    Absent(u8, Pending),
    /// Its bus is in its post-reset hold-off: the entry is parked, unread
    /// beyond its handle, until the hold ends.
    Quiet { bus: u8, handle: u32 },
    Run(Job),
}

/// A command waiting out its bus's post-reset hold-off. Only the ring
/// position is kept: the entry, its data segments included, is read again
/// from the guest's ring when the hold ends.
#[derive(Clone, Debug, PartialEq)]
struct Parked {
    /// Told apart from the others after the ring has changed under it.
    seq: u64,
    /// The request ring position of its first entry, and how many it spans.
    cons: u32,
    count: u8,
    bus: u8,
    /// The handle it was parked under: a rewritten entry is answered for it.
    handle: u32,
    /// The hold ended and the thread took it up. Such a command runs on
    /// whatever hold a later reset sets.
    started: bool,
}

struct State {
    ctl: Ctl,
    bus_cfg1: u16,
    bus_ctrl: u16,
    bus_status: u16,
    semaphore: u16,
    nvram: u16,
    risc_flags: u16,
    dma_cmd_cfg: u16,
    dma_data_cfg: u16,
    mailbox: [u16; MAILBOX_COUNT],
    risc_bank: [u16; BANK_WORDS],
    sxp_bank: [u16; BANK_WORDS],
    risc_ram: Vec<u16>,
    reset_delay: u16,
    /// Mailbox command 0x0031's value, in [`SELECTION_TIMEOUT_UNIT`]s.
    selection_timeout: u16,
    rqst_base: u64,
    resp_base: u64,
    rqst_len: u16,
    resp_len: u16,
    /// Where the ring thread has read up to. The position the guest reads
    /// is [`State::request_out`].
    rqst_cons: u16,
    resp_prod: u16,
    resp_cons: u16,
    pending: VecDeque<Pending>,
    /// Per bus, when the adapter's clock ends the hold-off a bus reset
    /// started. None: no hold.
    quiet_until: [Option<i64>; BUSES],
    /// Commands waiting out a hold-off, oldest first.
    parked: VecDeque<Parked>,
    park_seq: u64,
    total_cmds: u64,
}

impl State {
    fn new() -> Self {
        let mut s = State {
            ctl: Ctl::default(),
            bus_cfg1: 0,
            bus_ctrl: 0,
            bus_status: 0,
            semaphore: 0,
            nvram: 0,
            risc_flags: 0,
            dma_cmd_cfg: 0,
            dma_data_cfg: 0,
            mailbox: [0; MAILBOX_COUNT],
            risc_bank: [0; BANK_WORDS],
            sxp_bank: [0; BANK_WORDS],
            risc_ram: vec![0; RAM_WORDS],
            reset_delay: 0,
            selection_timeout: DEFAULT_SELECTION_TIMEOUT,
            rqst_base: 0,
            resp_base: 0,
            rqst_len: 0,
            resp_len: 0,
            rqst_cons: 0,
            resp_prod: 0,
            resp_cons: 0,
            pending: VecDeque::new(),
            quiet_until: [None; BUSES],
            parked: VecDeque::new(),
            park_seq: 0,
            total_cmds: 0,
        };
        s.reset();
        s
    }

    /// Power-on and guest soft reset land on one state. Mailbox 4's
    /// signature lives in the array so the register test's echo works before
    /// any queue exists.
    fn reset(&mut self) {
        self.ctl.gen += 1;
        self.ctl.rqst_epoch += 1;
        self.ctl.resp_epoch += 1;
        self.bus_cfg1 = 0;
        self.bus_ctrl = 0;
        self.bus_status = 0;
        self.semaphore = 0;
        self.nvram = 0;
        self.risc_flags = 0;
        self.dma_cmd_cfg = 0;
        self.dma_data_cfg = 0;
        self.mailbox = [0, SIG1, SIG2, SIG3, SIG4, 0, 0, 0];
        self.risc_bank = [0; BANK_WORDS];
        self.sxp_bank = [0; BANK_WORDS];
        // The driver re-downloads firmware after every reset.
        self.risc_ram.iter_mut().for_each(|w| *w = 0);
        self.reset_delay = 0;
        self.selection_timeout = DEFAULT_SELECTION_TIMEOUT;
        self.ctl.selecting = None;
        self.rqst_base = 0;
        self.resp_base = 0;
        self.rqst_len = 0;
        self.resp_len = 0;
        self.rqst_cons = 0;
        self.resp_prod = 0;
        self.resp_cons = 0;
        self.pending.clear();
        self.quiet_until = [None; BUSES];
        self.parked.clear();
    }

    /// The request-out position the guest reads. Approximation: it stays at
    /// the oldest parked command's slot until that command starts, because
    /// the model reads the entry again when the hold ends; a chip that had
    /// fetched the entry would plausibly advance it at once. The datasheet's
    /// bus-reset hold-off model choices record this.
    fn request_out(&self) -> u16 {
        self.parked.front().map_or(self.rqst_cons, |p| p.cons as u16)
    }

    /// Only the RISC request bit and its enable drive the line; they share a
    /// bit position.
    fn irq_level(&self) -> bool {
        self.bus_status & self.bus_ctrl & STATUS_RISC_INT_REQ != 0
    }

    fn resp_has_room(&self) -> bool {
        // One slot reserved so producer equal to consumer always means
        // empty; an uninitialised queue has no room.
        self.resp_len != 0 && (u32::from(self.resp_prod) + 1) % u32::from(self.resp_len) != u32::from(self.resp_cons)
    }

    fn active_bank(&mut self) -> &mut [u16; BANK_WORDS] {
        if self.bus_cfg1 & CFG1_BANK_SELECT != 0 { &mut self.sxp_bank } else { &mut self.risc_bank }
    }
}

/// The adapter. Its state lives in a [`Core`] shared with the thread that
/// runs the request ring.
pub struct Isp1040 {
    core: Arc<Core>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl std::ops::Deref for Isp1040 {
    type Target = Core;
    fn deref(&self) -> &Core { &self.core }
}

/// Lock order: `line_order`, then `st`. Nothing else is taken with `st`
/// held except the leaf locks (`notes`, `dma_misses`, `odd_accesses`): DMA,
/// the targets and `fence` are reached with it released. `fence` is held
/// across one guest-memory store at a time, and `targets` across one
/// target command; neither is held with the other.
pub struct Core {
    st: Mutex<State>,
    /// `st.ctl.gen`, readable without `st`: what a store into guest memory
    /// checks under `fence`.
    gen: AtomicU64,
    /// Held across each store the ring thread makes into guest memory. A
    /// reset moves the generation on and then passes through it, so once
    /// the reset returns no store of a cancelled command is under way or
    /// can start; the reset waits for one store, never for a disk.
    fence: Mutex<()>,
    /// Signals [`Ctl`] changes both ways.
    wake: Condvar,
    /// Serialises driving the interrupt line; see [`Core::sync_irq`].
    line_order: Mutex<()>,
    /// The targets on the bus, by SCSI id.
    bus: ScsiBus,
    dma: OnceLock<Arc<dyn PciDma>>,
    irq: OnceLock<IrqLine>,
    /// Which of the Octane's two controllers this is (0 internal, 1
    /// external): the monitor's `scsi`/`cow` address the first,
    /// `scsi1`/`cow1` the second.
    index: u8,
    pci: PciConfig,
    notes: Mutex<Vec<String>>,
    dma_misses: Mutex<std::collections::BTreeSet<u64>>,
    odd_accesses: Mutex<u32>,
    /// The status bar's activity bits, shared with the display thread.
    heartbeat: OnceLock<Arc<AtomicU64>>,
    /// Where the adapter's clock starts.
    start: Instant,
    /// When not `i64::MIN`, the clock in nanoseconds, for tests.
    fake_ns: AtomicI64,
}

fn le16(b: &[u8], off: usize) -> u16 { u16::from_le_bytes([b[off], b[off + 1]]) }

impl Isp1040 {
    /// An empty bus of the kind this chip drives: ids 0-15, and the
    /// emulated disk and CD-ROM. The machine fills it from `[scsi.N]` and
    /// hands it to [`Self::new_at`].
    pub fn new_bus() -> ScsiBus {
        ScsiBus::new(16, ScsiIdentities { disk: DISK_IDENTITY, cdrom: CDROM_IDENTITY })
    }

    /// The controller on an empty bus, for the tests.
    #[cfg(test)]
    pub fn new() -> Self {
        Self::new_at(0, Self::new_bus())
    }

    /// The controller at `index` (0 internal, 1 external) on `bus`.
    pub fn new_at(index: u8, bus: ScsiBus) -> Self {
        Self { core: Arc::new(Core::new_at(index, bus)), worker: Mutex::new(None) }
    }

    /// Start the ring thread (idempotent) and let it take whatever was
    /// published while the adapter was stopped.
    fn start_ring(&self) {
        let mut worker = self.worker.lock();
        if worker.is_some() {
            return;
        }
        self.st.lock().ctl.shutdown = false;
        let c = self.core.clone();
        *worker = Some(
            std::thread::Builder::new()
                .name(format!("ISP1040-{}", self.index))
                .spawn(move || c.run_ring())
                .expect("spawn isp1040 thread"),
        );
        drop(worker);
        self.resume();
    }

    /// Park the ring between entries, then end and join the thread. A
    /// command in flight finishes and posts first; afterwards nothing
    /// touches guest memory or the interrupt line.
    fn stop_ring(&self) {
        self.pause();
        let Some(h) = self.worker.lock().take() else { return };
        self.st.lock().ctl.shutdown = true;
        self.wake.notify_all();
        let _ = h.join();
    }
}

impl Isp1040 {
    /// Name the adapter's long-lived locks in the lock registry, which
    /// holds only a weak reference to them.
    pub fn register_locks(&self) {
        for (what, probe) in [
            ("st", (|c: &Core| c.st.is_locked()) as fn(&Core) -> bool),
            ("fence", |c| c.fence.is_locked()),
            ("targets", |c| c.bus.is_locked()),
        ] {
            let core = Arc::downgrade(&self.core);
            crate::locks::register_lock(format!("isp1040-{}::{what}", self.index), move || {
                core.upgrade().is_some_and(|c| probe(&c))
            });
        }
    }
}

impl Drop for Isp1040 {
    fn drop(&mut self) {
        // Backstop for a card dropped while started; the machine stops it.
        self.stop_ring();
    }
}

impl Core {
    fn new_at(index: u8, bus: ScsiBus) -> Self {
        let st = State::new();
        let gen = st.ctl.gen;
        Self {
            st: Mutex::new(st),
            gen: AtomicU64::new(gen),
            fence: Mutex::new(()),
            wake: Condvar::new(),
            line_order: Mutex::new(()),
            bus,
            dma: OnceLock::new(),
            irq: OnceLock::new(),
            index,
            // BAR 0 is the I/O window and BAR 1 its memory-space alias, one
            // register file.
            pci: PciConfig::new(
                PCI_VENDOR_QLOGIC, PCI_DEVICE_ISP1040, PCI_CLASS_SCSI, PCI_REVISION_ISP1040,
                [Bar::Io(reg::WINDOW_SIZE), Bar::Mem(reg::WINDOW_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused],
                PCI_INTA,
            ),
            notes: Mutex::new(Vec::new()),
            dma_misses: Mutex::new(std::collections::BTreeSet::new()),
            odd_accesses: Mutex::new(0),
            heartbeat: OnceLock::new(),
            start: Instant::now(),
            fake_ns: AtomicI64::new(i64::MIN),
        }
    }

    pub fn index(&self) -> u8 { self.index }

    /// A machine reset: the chip as at power-on (no firmware, no queues),
    /// its interrupt dropped. The disks stay attached. As the guest's soft
    /// reset, it cancels the command in flight without waiting for its
    /// target.
    pub fn reset(&self) {
        {
            let mut st = self.st.lock();
            st.reset();
            self.publish_gen(&st);
            self.wake.notify_all();
        }
        self.retire_stores();
        self.sync_irq();
    }

    fn publish_gen(&self, st: &State) {
        self.gen.store(st.ctl.gen, Ordering::Release);
    }

    /// Wait out a store into guest memory that a cancelled command began
    /// before the generation moved on; see `fence`.
    fn retire_stores(&self) {
        drop(self.fence.lock());
    }

    /// Run `store` for work of generation `gen`, unless a reset has
    /// cancelled that work.
    fn fenced<R>(&self, gen: u64, store: impl FnOnce() -> R) -> Option<R> {
        let _fence = self.fence.lock();
        (self.gen.load(Ordering::Acquire) == gen).then(store)
    }

    /// Light the status bar's SCSI activity bits in `heartbeat`.
    pub fn set_heartbeat(&self, heartbeat: Arc<AtomicU64>) {
        let _ = self.heartbeat.set(heartbeat);
    }

    /// Wire the adapter's interrupt pin to `line` (a level).
    pub fn connect_irq(&self, line: IrqLine) {
        let _ = self.irq.set(line);
        self.sync_irq();
    }

    /// True while the adapter asserts its interrupt line.
    pub fn interrupting(&self) -> bool {
        self.st.lock().irq_level()
    }

    /// Drive the line to the level the state gives. The level is read
    /// after taking `line_order`, so whichever of the CPU and the ring
    /// thread drives last has seen the latest state. `st` is never held
    /// when this is called.
    fn sync_irq(&self) {
        let _order = self.line_order.lock();
        let level = self.st.lock().irq_level();
        if let Some(l) = self.irq.get() {
            l.set(level);
        }
    }

    /// Block until the ring thread has run everything it was given and is
    /// waiting for more (or is held on a full response ring).
    /// A selection still timing out does not hold it.
    pub fn wait_idle(&self) {
        let mut st = self.st.lock();
        while (st.ctl.work && st.ctl.hold == 0 && !st.ctl.paused)
            || (st.ctl.hold == 0 && !st.ctl.paused && !st.ctl.walking && self.release_due(&st))
            || (st.ctl.walking && !st.ctl.selecting.as_ref().is_some_and(|sel| self.now_ns(&st) < sel.due))
        {
            self.wake.wait(&mut st);
        }
    }

    /// The adapter's clock in nanoseconds: host wall time less the time the
    /// machine was stopped.
    fn now_ns(&self, st: &State) -> i64 {
        match self.fake_ns.load(Ordering::Relaxed) {
            i64::MIN => {
                let at = st.ctl.stopped_at.unwrap_or_else(Instant::now);
                at.duration_since(self.start).as_nanos() as i64 - st.ctl.stopped_ns
            }
            t => t,
        }
    }

    /// Take the clock off the host's and set it to `ns` (tests). The
    /// thread reads the clock with `st` held and then waits on `wake`
    /// with no timeout, so the move is made with `st` held too: made
    /// outside it, the move and its wake-up can fall between that read and
    /// that wait, and the thread sleeps through the time it waits for.
    #[cfg(test)]
    fn set_test_clock(&self, ns: i64) {
        let _st = self.st.lock();
        self.fake_ns.store(ns, Ordering::Relaxed);
        self.wake.notify_all();
    }

    /// The buses in their post-reset hold-off now, as a bit mask.
    fn held_mask(&self, st: &State) -> u8 {
        let now = self.now_ns(st);
        (0..BUSES).filter(|&b| st.quiet_until[b].is_some_and(|until| now < until)).fold(0, |m, b| m | 1 << b)
    }

    /// The parked command to take up next: the oldest whose hold-off has
    /// ended, or that the thread had already taken up.
    fn releasable(&self, st: &State) -> Option<usize> {
        let held = self.held_mask(st);
        st.parked.iter().position(|p| p.started || held & (1 << p.bus) == 0)
    }

    /// A parked command could start now, and the thread is not stuck on a
    /// full response ring (which would leave it waiting for the driver).
    /// One the thread has already taken up is not counted: it is being
    /// waited out, or the thread is kicked to resume it.
    fn release_due(&self, st: &State) -> bool {
        let held = self.held_mask(st);
        (st.pending.is_empty() || st.resp_has_room())
            && st.parked.iter().any(|p| !p.started && held & (1 << p.bus) == 0)
    }

    /// How long until the first hold-off with a command parked behind it
    /// ends, when none has ended yet.
    fn next_release(&self, st: &State) -> Option<i64> {
        let now = self.now_ns(st);
        st.parked
            .iter()
            .filter_map(|p| st.quiet_until[usize::from(p.bus)])
            .map(|until| until - now)
            .filter(|&left| left > 0)
            .min()
    }

    /// Wait out the selection of an absent target on the ring entry at
    /// `cons`, with the state unlocked while the thread waits. True once the
    /// selection timeout has passed. False when the thread must look at the
    /// ring again first: it was parked, or a reset took the entry away.
    ///
    /// `start` is when the chip begins selecting: now, or for a command that
    /// waited out a hold-off, the moment the hold ended, so the wait is the
    /// hold plus the timeout however late the thread got to it.
    fn select_absent(&self, st: &mut MutexGuard<'_, State>, ring: &Ring, cons: u32, start: i64, bus: u8, done: Pending) -> bool {
        if !matches!(&st.ctl.selecting, Some(sel) if sel.cons == cons && !sel.orphan) {
            // A programmed zero is no wait: a model choice, the datasheet
            // being silent on zero.
            let wait = SELECTION_TIMEOUT_UNIT * u32::from(st.selection_timeout);
            let due = start.saturating_add(wait.as_nanos() as i64);
            st.ctl.selecting = Some(Selection { cons, due, done, bus, orphan: false });
        }
        // Whoever waits for the ring to settle may now see it waiting on a
        // selection, a resumed one included.
        self.wake.notify_all();
        self.wait_selection(st, ring.gen)
    }

    /// Wait for the selection in `ctl.selecting` to time out. False when
    /// the thread must look again first: parked, or a reset cancelled it.
    /// A new request ring does not cancel it: a model choice, the datasheet
    /// being silent on what the chip does to a selection in progress.
    fn wait_selection(&self, st: &mut MutexGuard<'_, State>, gen: u64) -> bool {
        loop {
            let Some(due) = st.ctl.selecting.as_ref().map(|sel| sel.due) else {
                return false;
            };
            if st.ctl.shutdown || st.ctl.hold > 0 || st.ctl.paused || st.ctl.gen != gen {
                return false;
            }
            let left = due - self.now_ns(st);
            if left <= 0 {
                return true;
            }
            // A test clock only moves when the test moves it.
            if self.fake_ns.load(Ordering::Relaxed) == i64::MIN {
                self.wake.wait_for(st, Duration::from_nanos(left as u64));
            } else {
                self.wake.wait(st);
            }
        }
    }

    /// Stop the ring thread between entries while the machine is stopped:
    /// once this returns the adapter touches neither guest memory nor its
    /// interrupt until [`Core::resume`]. A command in flight finishes and
    /// posts first.
    pub fn pause(&self) {
        let mut st = self.st.lock();
        st.ctl.paused = true;
        st.ctl.stopped_at.get_or_insert_with(Instant::now);
        // Parked mid-selection, the thread lets go of its entry.
        self.wake.notify_all();
        while st.ctl.walking {
            self.wake.wait(&mut st);
        }
    }

    /// Let the ring thread run again, taking up whatever was published
    /// while it was paused.
    pub fn resume(&self) {
        let mut st = self.st.lock();
        st.ctl.paused = false;
        if let Some(t) = st.ctl.stopped_at.take() {
            st.ctl.stopped_ns += t.elapsed().as_nanos() as i64;
        }
        self.kick(&mut st);
    }

    /// A save or a load is waiting for the ring thread (tests).
    #[cfg(test)]
    fn waiting_on_command(&self) -> bool {
        self.st.lock().ctl.hold != 0
    }

    /// A command is running against its target (tests).
    #[cfg(test)]
    fn command_running(&self) -> bool {
        self.st.lock().ctl.busy
    }

    /// Run `f` on the state with the ring thread stopped between entries
    /// and none allowed to start: the quiesce point for saving and loading.
    /// A command in flight posts its completion first.
    fn quiesced<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        let mut st = self.st.lock();
        st.ctl.hold += 1;
        self.wake.notify_all();
        while st.ctl.walking {
            self.wake.wait(&mut st);
        }
        let r = f(&mut st);
        st.ctl.hold -= 1;
        drop(st);
        self.wake.notify_all();
        r
    }

    /// Tell the ring thread the ring may have moved.
    fn kick(&self, st: &mut State) {
        st.ctl.work = true;
        self.wake.notify_all();
    }

    /// Commands run, for the whole session.
    pub fn total_commands(&self) -> u64 {
        self.st.lock().total_cmds
    }

    /// A running commentary on the odd things a driver did (accesses of a
    /// width the part does not decode, DMA that is not memory, commands not
    /// modelled).
    pub fn notes(&self) -> Vec<String> {
        self.notes.lock().clone()
    }

    fn note(&self, what: String) {
        let mut n = self.notes.lock();
        if n.len() < 256 {
            n.push(what);
        }
    }

    // ---- the bus ------------------------------------------------------

    /// The SCSI bus: attach targets to it and ask what is on it.
    pub fn bus(&self) -> &ScsiBus {
        &self.bus
    }

    fn has_target(&self, id: u8) -> bool {
        self.bus.has_target(id)
    }

    // ---- DMA ---------------------------------------------------------

    fn dma_ok(addr: u64, len: usize) -> bool {
        addr.checked_add(len as u64).is_some()
    }

    fn dma_miss(&self, addr: u64, write: bool) {
        let first = {
            let mut seen = self.dma_misses.lock();
            seen.len() < 64 && seen.insert(addr & !0xfff)
        };
        if first {
            let what = if write { "write to" } else { "read from" };
            crate::dlog!(LogModule::Scsi, "isp1040: DMA {what} {addr:#x} is not memory");
            self.note(format!("DMA {what} {addr:#x} is not memory"));
        }
    }

    /// `buf.len()` bytes from bus address `addr`; false (and zeros) if the
    /// fabric refuses any of them.
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let ok = Self::dma_ok(addr, buf.len()) && self.dma.get().is_some_and(|d| d.dma_read(addr, buf).is_ok());
        if !ok {
            buf.iter_mut().for_each(|b| *b = 0);
            self.dma_miss(addr, false);
        }
        ok
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> bool {
        let ok = Self::dma_ok(addr, data.len()) && self.dma.get().is_some_and(|d| d.dma_write(addr, data).is_ok());
        if !ok {
            self.dma_miss(addr, true);
        }
        ok
    }

    /// `base + off` on the bus, or None when the sum runs past the top of
    /// the 64-bit space: the guest chooses ring and buffer bases, and such
    /// an address is a refused transfer, as the QEMU model treats it.
    fn at(base: u64, off: u64) -> Option<u64> {
        base.checked_add(off)
    }

    /// Where entry `index` of a ring at `base` sits.
    fn entry_addr(base: u64, index: u32) -> Option<u64> {
        Self::at(base, u64::from(index) * ENTRY_SIZE as u64)
    }

    // A descriptor read the fabric refuses answers zero, deliberately: a
    // zeroed entry type takes the fail-visibly default and a zeroed segment
    // transfers nothing. An address past the top of the bus is refused.
    fn ld<const N: usize>(&self, addr: Option<u64>) -> [u8; N] {
        let mut b = [0u8; N];
        match addr {
            Some(a) => {
                self.dma_read(a, &mut b);
            }
            None => self.dma_miss(u64::MAX, false),
        }
        b
    }
    fn ld8(&self, addr: Option<u64>) -> u8 {
        self.ld::<1>(addr)[0]
    }
    fn ld16(&self, addr: Option<u64>) -> u16 {
        u16::from_le_bytes(self.ld(addr))
    }
    fn ld32(&self, addr: Option<u64>) -> u32 {
        u32::from_le_bytes(self.ld(addr))
    }

    // ---- RISC RAM ----------------------------------------------------

    fn mbox_host_addr(mb: &[u16; MAILBOX_COUNT]) -> u64 {
        ((mb[2] as u64) << 16) | mb[3] as u64
    }

    /// A RISC RAM load or dump of up to 128 KiB, with the state unlocked
    /// for the DMA. The whole transfer is read before a word of it is kept,
    /// so a refused DMA leaves the array as it was. Words travel
    /// little-endian. This part has no edge to its array: a transfer past
    /// the end folds to the start.
    fn ram_transfer(&self, st: &mut MutexGuard<'_, State>, load: bool) -> u16 {
        let base = st.mailbox[1];
        let count = st.mailbox[4] as usize;
        let host = Self::mbox_host_addr(&st.mailbox);
        if count == 0 {
            return mbox::COMPLETE;
        }
        if load {
            let mut bytes = vec![0u8; count * 2];
            if !MutexGuard::unlocked(st, || self.dma_read(host, &mut bytes)) {
                return mbox::HOST_INTERFACE_ERROR;
            }
            for i in 0..count {
                st.risc_ram[base.wrapping_add(i as u16) as usize] = le16(&bytes, i * 2);
            }
        } else {
            let bytes: Vec<u8> =
                (0..count).flat_map(|i| st.risc_ram[base.wrapping_add(i as u16) as usize].to_le_bytes()).collect();
            if !MutexGuard::unlocked(st, || self.dma_write(host, &bytes)) {
                return mbox::HOST_INTERFACE_ERROR;
            }
        }
        mbox::COMPLETE
    }

    // ---- mailbox -----------------------------------------------------

    /// Synchronous: the protocol has no way to leave a command pending.
    fn mailbox_command(&self, st: &mut MutexGuard<'_, State>) {
        let cmd = st.mailbox[0];
        match cmd {
            mbox::NOP => st.mailbox[0] = mbox::COMPLETE,
            mbox::WRITE_RAM_WORD => {
                let (at, word) = (st.mailbox[1], st.mailbox[2]);
                st.risc_ram[at as usize] = word;
                st.mailbox[0] = mbox::COMPLETE;
            }
            // Mailbox 3 is an output too; left alone it hands back stale data.
            mbox::READ_RAM_WORD => {
                st.mailbox[2] = st.risc_ram[st.mailbox[1] as usize];
                st.mailbox[3] = 0;
                st.mailbox[0] = mbox::COMPLETE;
            }
            mbox::LOAD_RAM => st.mailbox[0] = self.ram_transfer(st, true),
            mbox::DUMP_RAM => st.mailbox[0] = self.ram_transfer(st, false),
            // Mailboxes 1-7 are expected back unchanged; they already are.
            mbox::REGISTER_TEST => st.mailbox[0] = mbox::COMPLETE,
            // The word at base+3 is the image's length; a real image sums to
            // zero.
            mbox::VERIFY_CHECKSUM => {
                let start = st.mailbox[1];
                let len = st.risc_ram[start.wrapping_add(3) as usize];
                let mut sum = 0u16;
                for i in 0..len {
                    sum = sum.wrapping_add(st.risc_ram[start.wrapping_add(i) as usize]);
                }
                // The firmware reads the mailbox 2 copy.
                st.mailbox[1] = sum;
                st.mailbox[2] = sum;
                st.mailbox[0] = if sum == 0 { mbox::COMPLETE } else { mbox::TEST_FAILED };
            }
            // Accepted, never executed.
            mbox::EXEC_FIRMWARE => st.mailbox[0] = mbox::COMPLETE,
            mbox::ABOUT_FIRMWARE => {
                st.mailbox[1] = FW_VERSION[0];
                st.mailbox[2] = FW_VERSION[1];
                st.mailbox[3] = FW_VERSION[2];
                st.mailbox[6] = 0;
                st.mailbox[0] = mbox::COMPLETE;
            }
            // Sizes the driver's queues and command budget.
            mbox::GET_FIRMWARE_STATUS => {
                st.mailbox[1] = 0;
                st.mailbox[2] = MAX_REQUESTS;
                st.mailbox[3] = 0;
                st.mailbox[0] = mbox::COMPLETE;
            }
            // Asynchronous and narrow: the true state of an unnegotiated bus.
            mbox::GET_TARGET_PARAMETERS => {
                st.mailbox[2] = 0;
                st.mailbox[3] = 0;
                st.mailbox[0] = mbox::COMPLETE;
            }
            // Setters for state nothing reads back; answering invalid is a
            // fatal attach error, so they complete.
            mbox::SET_FIRMWARE_FEATURES
            | mbox::SET_CLOCK_RATE
            | mbox::SET_RETRY_COUNT
            | mbox::SET_ASYNC_DATA_SETUP
            | mbox::SET_ACTIVE_NEGATION
            | mbox::SET_TAG_AGE_LIMIT
            | mbox::SET_INITIATOR_ID
            | mbox::SET_TARGET_PARAMETERS
            | mbox::SET_DEVICE_QUEUE
            | mbox::SET_PCI_CONTROL
            | mbox::SET_SYSTEM_PARAMETER
            | mbox::SET_DATA_OVERRUN_RECOVERY => st.mailbox[0] = mbox::COMPLETE,
            mbox::SET_SELECTION_TIMEOUT => {
                st.selection_timeout = st.mailbox[1];
                st.mailbox[0] = mbox::COMPLETE;
            }
            mbox::SET_RESET_DELAY_PARAMS => {
                st.reset_delay = st.mailbox[1];
                st.mailbox[0] = mbox::COMPLETE;
            }
            mbox::INIT_REQ_QUEUE | mbox::INIT_REQ_QUEUE_A64 | mbox::INIT_RSP_QUEUE | mbox::INIT_RSP_QUEUE_A64 => {
                let a64 = cmd == mbox::INIT_REQ_QUEUE_A64 || cmd == mbox::INIT_RSP_QUEUE_A64;
                let mut addr = Self::mbox_host_addr(&st.mailbox);
                if a64 {
                    addr |= ((st.mailbox[7] as u64) << 32) | ((st.mailbox[6] as u64) << 48);
                }
                if addr == 0 || st.mailbox[1] == 0 {
                    st.mailbox[0] = mbox::PARAM_ERROR;
                } else if cmd == mbox::INIT_REQ_QUEUE || cmd == mbox::INIT_REQ_QUEUE_A64 {
                    st.ctl.rqst_epoch += 1;
                    if let Some(sel) = st.ctl.selecting.as_mut() {
                        sel.orphan = true;
                    }
                    // Their entries lived in the old ring. The hold-off
                    // stays: it belongs to the bus.
                    st.parked.clear();
                    st.rqst_base = addr;
                    st.rqst_len = st.mailbox[1];
                    st.rqst_cons = 0;
                    st.mailbox[MBOX_REQ] = 0;
                    st.mailbox[0] = mbox::COMPLETE;
                } else {
                    // A command in flight posts its completion here, as
                    // the QEMU model's does.
                    st.ctl.resp_epoch += 1;
                    st.resp_base = addr;
                    st.resp_len = st.mailbox[1];
                    st.resp_prod = 0;
                    st.resp_cons = 0;
                    // The backlog goes with the ring it was waiting for.
                    st.pending.clear();
                    st.mailbox[0] = mbox::COMPLETE;
                }
            }
            // Resets the bus, not the adapter: chip state survives. Mailbox 1
            // holds the configured reset delay, not a selector. The reset
            // destroys a selection still waiting on an absent target: it
            // ends now with the reset completion, and posts once. Inferred:
            // the drivers only establish that they handle that completion.
            //
            // It also starts the bus's hold-off: for mailbox 1 seconds the
            // chip starts no SCSI activity on that bus (datasheet,
            // "Bus-reset hold-off"). Commands for it park, to start when the
            // hold ends; the other bus and mailbox commands carry on. A
            // second reset retires what the first parked, with the same
            // reset completion, and replaces the hold; a delay of 0 holds
            // nothing. The delay is not clamped, and nothing is posted when
            // the hold ends: model choices, none established.
            mbox::BUS_RESET => {
                let now = self.now_ns(st);
                let bus = usize::from(st.mailbox[2]);
                if bus >= BUSES {
                    // A bus this part does not have: nothing to quiet or
                    // retire, and the datasheet's per-bus hold leaves the
                    // real bus alone. Noted so a driver that leaves mailbox 2
                    // unset on this single-bus part is not silently unheld.
                    self.note(format!("bus reset names bus {bus}, which this part does not have; ignored"));
                } else {
                    if let Some(sel) = st.ctl.selecting.as_mut().filter(|sel| usize::from(sel.bus) == bus) {
                        sel.due = now;
                        sel.done = Pending { handle: sel.done.handle, completion: IOCB_RESET, ..Default::default() };
                    }
                    // The thread posts these in order, with any other held
                    // completion. Without a response ring there is nowhere
                    // to post them, as for any completion.
                    let (retired, kept): (Vec<_>, Vec<_>) =
                        st.parked.drain(..).partition(|p| usize::from(p.bus) == bus && !p.started);
                    st.parked.extend(kept);
                    for p in retired {
                        if st.resp_len == 0 {
                            self.note(format!("reset completion for parked handle {:#010x} dropped: no response queue", p.handle));
                        } else {
                            st.pending.push_back(Pending { handle: p.handle, completion: IOCB_RESET, ..Default::default() });
                        }
                    }
                    let delay = BUS_RESET_DELAY_UNIT * u32::from(st.mailbox[1]);
                    st.quiet_until[bus] = (!delay.is_zero()).then(|| now.saturating_add(delay.as_nanos() as i64));
                    self.kick(st);
                }
                self.wake.notify_all();
                st.mailbox[0] = mbox::COMPLETE;
            }
            _ => {
                self.note(format!("mailbox command {cmd:#06x} not modelled"));
                st.mailbox[0] = mbox::INVALID;
            }
        }
        st.semaphore = SEMAPHORE_LOCK;
        st.bus_status |= STATUS_INT_REQ | STATUS_RISC_INT_REQ;
    }

    fn host_command(&self, st: &mut MutexGuard<'_, State>, val: u16) {
        match val >> HOST_CMD_SHIFT {
            0x0 => {}
            // A RISC reset replaces the whole flag set.
            0x1 => st.risc_flags = RISC_FLAG_RESET,
            0x2 => st.risc_flags |= RISC_FLAG_PAUSED,
            // Release clears reset too, or the flag would stand forever.
            0x3 => st.risc_flags &= !(RISC_FLAG_PAUSED | RISC_FLAG_RESET),
            // "A mailbox command is ready": the trigger to process one.
            0x5 => {
                st.risc_flags |= RISC_FLAG_HOST_INT;
                self.mailbox_command(st);
            }
            // Does nothing on real hardware; neither public driver issues it.
            0x6 => {}
            // The semaphore is released by the driver's separate write.
            0x7 => {
                st.risc_flags &= !RISC_FLAG_HOST_INT;
                st.bus_status = 0;
            }
            // Accepted with nothing behind them, so the attach proceeds.
            c @ (0x4 | 0x9 | 0xa | 0xe) => {
                self.note(format!("host command {c:#x} accepted, not modelled"));
            }
            c => self.note(format!("unknown host command {c:#x}")),
        }
    }

    // ---- completions -------------------------------------------------

    fn status_entry(p: &Pending) -> [u8; ENTRY_SIZE] {
        let mut entry = [0u8; ENTRY_SIZE];
        let sense_len = p.sense.len().min(SENSE_MAX);
        entry[0] = ENTRY_TYPE_STATUS;
        entry[1] = 1;
        entry[4..8].copy_from_slice(&p.handle.to_le_bytes());
        entry[8..10].copy_from_slice(&p.scsi_status.to_le_bytes());
        entry[10..12].copy_from_slice(&p.completion.to_le_bytes());
        entry[12..14].copy_from_slice(&p.state_flags.to_le_bytes());
        entry[14..16].copy_from_slice(&p.status_flags.to_le_bytes());
        entry[18..20].copy_from_slice(&(sense_len as u16).to_le_bytes());
        entry[20..24].copy_from_slice(&p.resid.to_le_bytes());
        entry[32..32 + sense_len].copy_from_slice(&p.sense[..sense_len]);
        entry
    }

    /// Post a completion to the response ring, or hold it for room: a
    /// model with no bus latency can outrun the driver. Only the ring
    /// thread posts. The entry is stored with the state unlocked; if the
    /// driver replaced the response ring meanwhile the entry goes into the
    /// new one, and a reset drops it.
    ///
    /// An I/O completion never takes the semaphore lock: the driver reads
    /// the semaphore first and falls through to bus status only when it is
    /// clear.
    fn post<'a>(&'a self, mut st: MutexGuard<'a, State>, p: Pending) -> MutexGuard<'a, State> {
        let entry = Self::status_entry(&p);
        loop {
            if st.resp_len == 0 {
                self.note("status posted before the response queue was initialised".into());
                return st;
            }
            if !st.resp_has_room() {
                self.note(format!("response queue full, holding handle {:#010x} until acknowledged", p.handle));
                st.pending.push_back(p);
                return st;
            }
            let (gen, epoch) = (st.ctl.gen, st.ctl.resp_epoch);
            let addr = Self::entry_addr(st.resp_base, u32::from(st.resp_prod));
            let stored =
                MutexGuard::unlocked(&mut st, || self.fenced(gen, || addr.is_some_and(|a| self.dma_write(a, &entry))));
            if st.ctl.gen != gen {
                return st;
            }
            if st.ctl.resp_epoch != epoch {
                continue;
            }
            // A completion the fabric refuses to store is dropped: advancing
            // the producer or raising the interrupt would point the driver
            // at a slot still holding a previous entry's bytes.
            if stored != Some(true) {
                self.note(format!("status entry write to {addr:#x?} refused; completion dropped"));
                return st;
            }
            st.resp_prod = ((u32::from(st.resp_prod) + 1) % u32::from(st.resp_len)) as u16;
            st.bus_status |= STATUS_INT_REQ | STATUS_RISC_INT_REQ;
            return st;
        }
    }

    fn incomplete(handle: u32, state_flags: u16, status_flags: u16) -> Pending {
        Pending { handle, completion: IOCB_INCOMPLETE, state_flags, status_flags, ..Default::default() }
    }

    // ---- requests ----------------------------------------------------

    /// How many bytes a command is declared to move, and whether they go to
    /// the device. `block` is the target's logical block size. An opcode
    /// with no rule here moves nothing; the target answers it as SCSI says.
    fn cdb_transfer(cdb: &[u8], block: usize) -> (usize, bool) {
        let at = |i: usize| cdb.get(i).copied().unwrap_or(0) as usize;
        let be16 = |i: usize| (at(i) << 8) | at(i + 1);
        let be24 = |i: usize| (at(i) << 16) | (at(i + 1) << 8) | at(i + 2);
        let six = || if at(4) == 0 { 256 } else { at(4) };
        match cdb[0] {
            scsi_cmd::READ_6 => (six() * block, false),
            scsi_cmd::WRITE_6 => (six() * block, true),
            scsi_cmd::READ_10 => (be16(7) * block, false),
            scsi_cmd::WRITE_10 => (be16(7) * block, true),
            scsi_cmd::INQUIRY => (be16(3), false),
            scsi_cmd::REQUEST_SENSE | scsi_cmd::MODE_SENSE_6 => (at(4), false),
            scsi_cmd::MODE_SELECT_6 => (at(4), true),
            scsi_cmd::MODE_SENSE_10
            | scsi_cmd::READ_TOC_PMA_ATIP
            | scsi_cmd::GET_CONFIGURATION
            | scsi_cmd::READ_SUB_CHANNEL
            | scsi_cmd::READ_DISC_INFORMATION => (be16(7), false),
            scsi_cmd::MODE_SELECT_10 => (be16(7), true),
            scsi_cmd::READ_CAPACITY_10 => (8, false),
            scsi_cmd::READ_BUFFER => (be24(6), false),
            scsi_cmd::WRITE_BUFFER => (be24(6), true),
            scsi_cmd::SEND_DIAGNOSTIC => (be16(3), true),
            scsi_cmd::RECEIVE_DIAGNOSTIC_RESULTS => (be16(3), false),
            _ => (0, false),
        }
    }

    /// One descriptor into the transfer. The descriptors name the host's
    /// buffer and the command names how much of it the target has to fill,
    /// so only that much is offered and a longer buffer keeps its tail. The
    /// whole named length is still counted: the residual is measured against
    /// it.
    fn add_segment(segs: &mut Vec<(u64, u32)>, described: &mut u64, max: u64, base: u64, len: u32) {
        let room = max.saturating_sub(*described);
        let placed = (len as u64).min(room);
        *described += len as u64;
        if placed > 0 {
            segs.push((base, placed as u32));
        }
    }

    fn read_segments(&self, ring: &Ring, entry_addr: u64, entry_count: u8, seg_count: u16, max: u64, a64: bool) -> (Vec<(u64, u32)>, u64) {
        let (inline_off, inline_n, seg_size, cont_off, cont_n) = if a64 {
            (A64_OFF_SEGS, A64_SEGS_INLINE, A64_SEG_SIZE, A64_CONT_OFF_SEGS, A64_CONT_SEGS)
        } else {
            (REQ_OFF_SEGS, REQ_SEGS_INLINE, SEG_SIZE, CONT_OFF_SEGS, CONT_SEGS)
        };
        let seg_at = |base: Option<u64>| -> (u64, u32) {
            let field = |off: u64| self.ld32(base.and_then(|b| Self::at(b, off)));
            if a64 {
                (u64::from(field(0)) | (u64::from(field(4)) << 32), field(8))
            } else {
                (u64::from(field(0)), field(4))
            }
        };
        let mut segs = Vec::new();
        let mut described = 0u64;
        let mut collected = 0u16;
        for i in 0..inline_n {
            if collected >= seg_count { break; }
            let (a, l) = seg_at(Self::at(entry_addr, inline_off + i as u64 * seg_size));
            Self::add_segment(&mut segs, &mut described, max, a, l);
            collected += 1;
        }
        let mut cont_index = ring.cons;
        let mut conts_left = entry_count as i32 - 1;
        while collected < seg_count && conts_left > 0 {
            cont_index = (cont_index + 1) % ring.len;
            let cont_addr = Self::entry_addr(ring.base, cont_index);
            for i in 0..cont_n {
                if collected >= seg_count { break; }
                let (a, l) = seg_at(cont_addr.and_then(|c| Self::at(c, cont_off + i as u64 * seg_size)));
                Self::add_segment(&mut segs, &mut described, max, a, l);
                collected += 1;
            }
            conts_left -= 1;
        }
        (segs, described)
    }

    /// Run a CDB against the target; `data` is what a write sends it. None
    /// when a reset cancelled the command while it waited for the target.
    fn scsi(&self, gen: u64, target: u8, cdb: &[u8], data: Option<Vec<u8>>) -> Option<(Vec<u8>, u8)> {
        let mut targets = self.bus.lock();
        if self.gen.load(Ordering::Acquire) != gen {
            return None;
        }
        let Some(dev) = targets.get_mut(target as usize).and_then(|t| t.as_mut()) else {
            return Some((Vec::new(), CHECK_CONDITION));
        };
        let req = ScsiRequest { cdb: cdb.to_vec(), data_len: ScsiDataLength::Unlimited, data_in: data };
        Some(match dev.request(&req) {
            Ok(r) => (r.data, r.status),
            Err(e) => {
                crate::dlog!(LogModule::Scsi, "isp1040: target {target}: {e}");
                self.note(format!("target {target}: {e}"));
                (Vec::new(), CHECK_CONDITION)
            }
        })
    }

    /// Read a request entry. Refused when the fabric refused the CDB read:
    /// the walk stops on the entry and nothing is posted.
    fn prepare_request(&self, ring: &Ring, addr: u64, entry_count: u8, a64: bool) -> Prepared {
        let handle = self.ld32(Self::at(addr, REQ_OFF_HANDLE));
        let device = self.ld16(Self::at(addr, REQ_OFF_DEVICE));
        let cdb_len = self.ld16(Self::at(addr, REQ_OFF_CDBLEN));
        let seg_count = self.ld16(Self::at(addr, REQ_OFF_SEGCOUNT));
        let mut cdb = [0u8; CDB_FIELD_LEN];
        if !Self::at(addr, REQ_OFF_CDB).is_some_and(|a| self.dma_read(a, &mut cdb)) {
            self.note(format!("CDB read for the request entry at {addr:#x} refused; request dropped"));
            return Prepared::Refused;
        }
        let channel = (device >> 15) & 1;
        let target = ((device >> 8) & 0x7f) as u8;
        let lun = (device & 0xff) as u8;

        // A bus this part does not have.
        if usize::from(channel) >= BUSES {
            return Prepared::Answered(Self::incomplete(handle, 0, 0));
        }
        // GOT_BUS alone is Linux's bad-target; the timeout bit beside it is
        // what lets the empty slot be treated as benign.
        if target > MAX_TARGET || !self.has_target(target) {
            return Prepared::Absent(channel as u8, Self::incomplete(handle, STATE_GOT_BUS, STATUS_FLAG_TIMEOUT));
        }
        if lun > MAX_LUN {
            return Prepared::Answered(Self::incomplete(handle, STATE_GOT_BUS, STATUS_FLAG_TIMEOUT));
        }

        // The target model reads the unit number from the CDB's second byte.
        cdb[1] = (cdb[1] & 0x1f) | (lun << 5);
        let cdb = &cdb[..(cdb_len as usize).clamp(1, CDB_FIELD_LEN)];
        let block = self
            .bus
            .lock()
            .get(target as usize)
            .and_then(|t| t.as_ref())
            .and_then(|t| t.block_size())
            .map(|b| b as usize)
            .filter(|&b| b != 0)
            .unwrap_or(BLOCK_SIZE);
        let (xfer, to_dev) = Self::cdb_transfer(cdb, block);
        let (segs, described) = self.read_segments(ring, addr, entry_count, seg_count, xfer as u64, a64);
        Prepared::Run(Job { handle, target, cdb: cdb.to_vec(), xfer, to_dev, segs, described })
    }

    /// Run a command against its target and build its completion. The
    /// state is not locked: this is the part that waits on the disk.
    ///
    /// A command whose payload transfer the fabric refuses stops there and
    /// builds no completion: a WRITE sends nothing to the target, a READ
    /// stores no more of what the target returned. The QEMU model leaves
    /// such a request outstanding, never completed, until the driver gives
    /// up on it; with no completion posted the driver sees the same here.
    ///
    /// None too when a reset cancelled the command (`gen` is the generation
    /// it was read under): what it had not yet stored in host memory is not
    /// stored, and no completion is built.
    fn run_job(&self, job: Job, gen: u64) -> Option<Pending> {
        let Job { handle, target, cdb, xfer, to_dev, segs, described } = job;
        // The bar has lights for SCSI ids 0-5.
        if let (Some(hb), 0..=5) = (self.heartbeat.get(), target) {
            hb.fetch_or(1u64 << (crate::dev::ng1::rex3::Rex3::HB_SCSI_BASE as u64 + target as u64), Ordering::Relaxed);
        }
        let (status, moved);
        if to_dev {
            let mut buf = Vec::new();
            for &(a, l) in &segs {
                let mut chunk = vec![0u8; l as usize];
                if !self.dma_read(a, &mut chunk) {
                    self.note(format!("payload DMA for handle {handle:#x} refused by the fabric; command stopped"));
                    return None;
                }
                buf.extend_from_slice(&chunk);
            }
            moved = buf.len() as u64;
            status = self.scsi(gen, target, &cdb, Some(buf))?.1;
        } else {
            let (data, s) = self.scsi(gen, target, &cdb, None)?;
            status = s;
            // Offer the target's bytes to the descriptors in order.
            let take = (data.len() as u64).min(described).min(xfer as u64) as usize;
            let mut off = 0usize;
            for &(a, l) in &segs {
                if off >= take { break; }
                let n = (l as usize).min(take - off);
                if !self.fenced(gen, || self.dma_write(a, &data[off..off + n]))? {
                    self.note(format!("payload DMA for handle {handle:#x} refused by the fabric; command stopped"));
                    return None;
                }
                off += n;
            }
            moved = off as u64;
        }

        // Auto-sense runs only after a CHECK CONDITION.
        let mut sense = Vec::new();
        if status == CHECK_CONDITION {
            sense = self.scsi(gen, target, &[scsi_cmd::REQUEST_SENSE, 0, 0, 0, SENSE_MAX as u8, 0], None)?.0;
            sense.truncate(SENSE_MAX);
        }
        let mut scsi_status = status as u16;
        let mut state_flags = STATE_ALL_PHASES & !STATE_GOT_SENSE;
        if !sense.is_empty() {
            scsi_status |= SCSI_STATUS_SENSE_VALID;
            state_flags |= STATE_GOT_SENSE;
        }
        let total = described.max(xfer as u64);
        let unused = (total - moved.min(total)) as u32;
        // Underrun is a short data transfer, not a refused command: a target
        // answering with a status of its own carries the failure there.
        let completion = if unused != 0 && status == 0 { IOCB_DATA_UNDERRUN } else { IOCB_COMPLETE };
        Some(Pending { handle, scsi_status, completion, state_flags, status_flags: 0, resid: unused, sense })
    }

    /// The ring thread: wait to be kicked, then walk the request ring.
    fn run_ring(&self) {
        let mut st = self.st.lock();
        loop {
            while !st.ctl.shutdown && !(st.ctl.hold == 0 && !st.ctl.paused && (st.ctl.work || self.release_due(&st))) {
                // Commands parked behind a hold-off start when it ends,
                // with nobody to wake the thread. A test clock only moves
                // when the test moves it.
                let ends = (st.ctl.hold == 0 && !st.ctl.paused && self.fake_ns.load(Ordering::Relaxed) == i64::MIN)
                    .then(|| self.next_release(&st))
                    .flatten();
                match ends {
                    Some(left) if left > 0 => {
                        self.wake.wait_for(&mut st, Duration::from_nanos(left as u64));
                    }
                    _ => self.wake.wait(&mut st),
                }
            }
            if st.ctl.shutdown {
                return;
            }
            st.ctl.work = false;
            st.ctl.walking = true;
            st = self.consume_requests(st);
            st.ctl.walking = false;
            self.wake.notify_all();
        }
    }

    /// Read the entry at the ring's consumer and what it asks for, with the
    /// state unlocked. Returns how many entries it spans with it.
    ///
    /// `held` is the buses in their hold-off: a request for one is parked,
    /// not read further. `expect` is the handle a parked command was
    /// parked under, when it is being read again to start: an entry that
    /// has been rewritten since is answered incomplete for that handle.
    fn read_entry(&self, ring: &Ring, budget: u32, held: u8, expect: Option<u32>) -> (u8, Prepared) {
        let Some(addr) = Self::entry_addr(ring.base, ring.cons) else {
            // Past the top of the bus: refused, and the walk stops.
            self.dma_miss(u64::MAX, false);
            return (1, Prepared::Refused);
        };
        let entry_type = self.ld8(Some(addr));
        let entry_count = self.ld8(Self::at(addr, 1)).max(1).min(budget.min(255) as u8);
        let is_request = matches!(entry_type, ENTRY_TYPE_REQUEST | ENTRY_TYPE_COMMAND_A64);
        if let Some(handle) = expect {
            if !is_request || self.ld32(Self::at(addr, REQ_OFF_HANDLE)) != handle {
                self.note(format!("parked request entry {} was rewritten; handle {handle:#010x} answered incomplete", ring.cons));
                return (entry_count, Prepared::Answered(Self::incomplete(handle, 0, 0)));
            }
        }
        let prepared = match entry_type {
            ENTRY_TYPE_REQUEST | ENTRY_TYPE_COMMAND_A64 => {
                let bus = (self.ld16(Self::at(addr, REQ_OFF_DEVICE)) >> 15) as u8 & 1;
                if usize::from(bus) < BUSES && held & (1 << bus) != 0 {
                    Prepared::Quiet { bus, handle: self.ld32(Self::at(addr, REQ_OFF_HANDLE)) }
                } else {
                    self.prepare_request(ring, addr, entry_count, entry_type == ENTRY_TYPE_COMMAND_A64)
                }
            }
            // Continuations are stepped past here; reading segments visits
            // them.
            ENTRY_TYPE_MARKER => Prepared::Skipped,
            // Answered incomplete rather than consumed silently.
            t => {
                self.note(format!("unexpected request entry type {t:#04x} at index {}", ring.cons));
                Prepared::Answered(Self::incomplete(self.ld32(Self::at(addr, REQ_OFF_HANDLE)), 0, 0))
            }
        };
        (entry_count, prepared)
    }

    /// Walk the request ring from the adapter's consumer position to the
    /// driver's producer, one entry at a time. Reading entries, running
    /// commands and storing completions all happen with the state
    /// unlocked, so the CPU's register accesses never wait on a disk or a
    /// DMA.
    fn consume_requests<'a>(&'a self, mut st: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        loop {
            if st.ctl.shutdown || st.ctl.hold > 0 || st.ctl.paused {
                // Resume once the holder lets go.
                st.ctl.work = true;
                return st;
            }
            // Completions held for room go first, in order, as the driver
            // frees it; while any is held the request ring waits.
            if !st.pending.is_empty() {
                if !st.resp_has_room() {
                    return st;
                }
                let p = st.pending.pop_front().unwrap();
                st = self.post(st, p);
                MutexGuard::unlocked(&mut st, || self.sync_irq());
                continue;
            }
            // A selection whose entry the driver replaced the ring under
            // still ends with its completion.
            if st.ctl.selecting.as_ref().is_some_and(|sel| sel.orphan) {
                let gen = st.ctl.gen;
                self.wake.notify_all();
                if self.wait_selection(&mut st, gen) {
                    if let Some(sel) = st.ctl.selecting.take() {
                        st = self.post(st, sel.done);
                    }
                    MutexGuard::unlocked(&mut st, || self.sync_irq());
                }
                continue;
            }
            // A command whose bus's hold-off has ended starts before anything
            // read from the ring after it, oldest first.
            if let Some(i) = self.releasable(&st) {
                st.parked[i].started = true;
                let parked = st.parked[i].clone();
                let now = self.now_ns(&st);
                // The chip would have begun selecting when the hold ended.
                let start = st.quiet_until[usize::from(parked.bus)].map_or(now, |until| until.min(now));
                let len = u32::from(st.rqst_len);
                let ring = Ring { base: st.rqst_base, len, cons: parked.cons, epoch: st.ctl.rqst_epoch, gen: st.ctl.gen };
                let (_, prepared) = MutexGuard::unlocked(&mut st, || {
                    self.read_entry(&ring, u32::from(parked.count), 0, Some(parked.handle))
                });
                if st.ctl.gen != ring.gen {
                    continue;
                }
                let (back, outcome) = self.run_prepared(st, &ring, prepared, start);
                st = back;
                match outcome {
                    Outcome::Again => continue,
                    Outcome::Stop => {
                        // Refused on this read: the command is dropped.
                        self.note(format!("parked handle {:#010x} dropped: its request entry could not be read", parked.handle));
                        st.parked.retain(|p| p.seq != parked.seq);
                        continue;
                    }
                    Outcome::Done(done) => {
                        st.parked.retain(|p| p.seq != parked.seq);
                        if let Some(p) = done {
                            st = self.post(st, p);
                        }
                        MutexGuard::unlocked(&mut st, || self.sync_irq());
                        continue;
                    }
                }
            }
            if st.rqst_len == 0 {
                return st;
            }
            // Ring arithmetic is done wide: the length is any sixteen-bit
            // value the driver chose, and indices wrap modulo it.
            let len = u32::from(st.rqst_len);
            let cons = u32::from(st.rqst_cons) % len;
            let prod = u32::from(st.mailbox[MBOX_REQ]) % len;
            // No exact-equality exit terminates for every driver-supplied
            // index and entry count, so each step is budgeted by the
            // published entries; every step consumes at least one.
            let budget = (prod + len - cons) % len;
            if budget == 0 {
                return st;
            }
            let ring = Ring { base: st.rqst_base, len, cons, epoch: st.ctl.rqst_epoch, gen: st.ctl.gen };
            let held = self.held_mask(&st);
            let (count, prepared) = MutexGuard::unlocked(&mut st, || self.read_entry(&ring, budget, held, None));
            // A reset took the ring away while the entry was read.
            if st.ctl.gen != ring.gen {
                continue;
            }
            // Behind a hold-off: set aside, and the walk goes on to what
            // follows it. The consumer the guest reads stays behind.
            if let Prepared::Quiet { bus, handle } = prepared {
                if st.ctl.rqst_epoch == ring.epoch {
                    st.park_seq += 1;
                    let seq = st.park_seq;
                    st.parked.push_back(Parked { seq, cons, count, bus, handle, started: false });
                    st.rqst_cons = ((cons + u32::from(count)) % len) as u16;
                }
                continue;
            }
            let now = self.now_ns(&st);
            let (back, outcome) = self.run_prepared(st, &ring, prepared, now);
            st = back;
            let done = match outcome {
                Outcome::Again => continue,
                Outcome::Stop => return st,
                Outcome::Done(done) => done,
            };
            // A new request ring starts at its own index 0; the command read
            // from the ring it replaced still ran once.
            if st.ctl.rqst_epoch == ring.epoch {
                st.rqst_cons = ((cons + u32::from(count)) % len) as u16;
            }
            if let Some(p) = done {
                st = self.post(st, p);
            }
            MutexGuard::unlocked(&mut st, || self.sync_irq());
        }
    }

    /// Carry out what reading the entry at `ring.cons` found, up to the
    /// completion to post. `start` is when selection of an absent target
    /// begins.
    fn run_prepared<'a>(
        &'a self,
        mut st: MutexGuard<'a, State>,
        ring: &Ring,
        prepared: Prepared,
        start: i64,
    ) -> (MutexGuard<'a, State>, Outcome) {
        let done = match prepared {
            Prepared::Refused => {
                if st.ctl.rqst_epoch != ring.epoch {
                    return (st, Outcome::Again);
                }
                return (st, Outcome::Stop);
            }
            Prepared::Skipped | Prepared::Quiet { .. } => None,
            Prepared::Answered(p) => Some(p),
            Prepared::Absent(bus, p) => {
                if !self.select_absent(&mut st, ring, ring.cons, start, bus, p) {
                    return (st, Outcome::Again);
                }
                // The wait may have outlived the ring; see `orphan`.
                st.ctl.selecting.take().map(|sel| sel.done)
            }
            Prepared::Run(job) => {
                st.total_cmds += 1;
                st.ctl.busy = true;
                let done = MutexGuard::unlocked(&mut st, || self.run_job(job, ring.gen));
                st.ctl.busy = false;
                self.wake.notify_all();
                // A reset cancelled the command.
                if st.ctl.gen != ring.gen {
                    return (st, Outcome::Again);
                }
                done
            }
        };
        (st, Outcome::Done(done))
    }

    // ---- registers ---------------------------------------------------

    fn reg_read(&self, off: u32) -> u16 {
        let mut st = self.st.lock();
        // The adapter's own producer count, never the driver's last write.
        if off == reg::RESPONSE_QUEUE {
            return if st.resp_len == 0 { st.mailbox[MBOX_RSP] } else { st.resp_prod };
        }
        // The adapter's consumer position: a driver computes its free ring
        // space from it, so echoing its own write breaks that.
        if off == reg::REQUEST_QUEUE {
            return if st.rqst_len == 0 { st.mailbox[MBOX_REQ] } else { st.request_out() };
        }
        if (reg::MAILBOX_BASE..MAILBOX_END).contains(&off) {
            return st.mailbox[((off - reg::MAILBOX_BASE) / 2) as usize];
        }
        if (reg::BANK_BASE..BANK_END).contains(&off) {
            return st.active_bank()[((off - reg::BANK_BASE) / 2) as usize];
        }
        match off {
            reg::ID_LOW => PCI_VENDOR_QLOGIC,
            reg::ID_HIGH => PCI_DEVICE_ISP1040,
            // The revision field runs one ahead of the PCI byte.
            reg::BUS_CFG0 => PCI_REVISION_ISP1040 as u16 + 1,
            reg::BUS_CFG1 => st.bus_cfg1,
            reg::BUS_CTRL => st.bus_ctrl,
            reg::BUS_STATUS => st.bus_status,
            reg::SEMAPHORE => st.semaphore,
            reg::NVRAM => st.nvram,
            reg::DMA_CMD_CFG => st.dma_cmd_cfg,
            reg::DMA_DATA_CFG => st.dma_data_cfg,
            reg::HOST_CMD => st.risc_flags,
            _ => {
                drop(st);
                self.note(format!("read of unimplemented register {off:#04x}"));
                0
            }
        }
    }

    fn reg_write(&self, off: u32, val: u16) {
        let mut st = self.st.lock();
        if off == reg::RESPONSE_QUEUE {
            // The consumer acknowledgement: recorded, never over the
            // adapter's own producer count.
            if st.resp_len == 0 {
                st.mailbox[MBOX_RSP] = val;
            } else {
                st.resp_cons = val % st.resp_len;
                // The ring thread posts what was held for this room.
                self.kick(&mut st);
            }
            return;
        }
        if (reg::MAILBOX_BASE..MAILBOX_END).contains(&off) {
            st.mailbox[((off - reg::MAILBOX_BASE) / 2) as usize] = val;
            // Writing the request-in pointer is itself the trigger.
            if off == reg::REQUEST_QUEUE {
                self.kick(&mut st);
            }
            return;
        }
        if (reg::BANK_BASE..BANK_END).contains(&off) {
            st.active_bank()[((off - reg::BANK_BASE) / 2) as usize] = val;
            return;
        }
        match off {
            reg::BUS_CFG1 => {
                if val & !CFG1_WRITE_MASK != 0 {
                    self.note(format!("bus configuration 1 bits {:#06x} are not modelled", val & !CFG1_WRITE_MASK));
                }
                st.bus_cfg1 = val & CFG1_WRITE_MASK;
            }
            // Applied after the enables are stored, so the reset wins.
            reg::BUS_CTRL => {
                st.bus_ctrl = val & CTRL_WRITE_MASK;
                // The command in flight is cancelled, not waited for.
                if val & CTRL_SOFT_RESET != 0 {
                    st.reset();
                    self.publish_gen(&st);
                    self.wake.notify_all();
                    drop(st);
                    self.retire_stores();
                }
            }
            reg::SEMAPHORE => st.semaphore = val & SEMAPHORE_MASK,
            reg::NVRAM => st.nvram = val & NVRAM_WRITE_MASK,
            reg::DMA_CMD_CFG => st.dma_cmd_cfg = val & DMA_CFG_WRITE_MASK,
            reg::DMA_DATA_CFG => st.dma_data_cfg = val & DMA_CFG_WRITE_MASK,
            // Nothing to quiesce: a command's transfers are its own.
            reg::DMA_CMD_CTRL | reg::DMA_DATA_CTRL => {}
            reg::HOST_CMD => self.host_command(&mut st, val),
            reg::ID_LOW | reg::ID_HIGH | reg::BUS_CFG0 | reg::BUS_STATUS => {
                drop(st);
                self.note(format!("write {val:#06x} to read-only register {off:#04x}"));
            }
            _ => {
                drop(st);
                self.note(format!("write {val:#06x} to unimplemented register {off:#04x}"));
            }
        }
    }

    /// A register access. Only sixteen-bit accesses are decoded.
    pub fn read(&self, off: u32, width: usize) -> u32 {
        if width != 2 || off & 1 != 0 {
            self.odd_access("read", off, width);
            return 0;
        }
        self.reg_read(off) as u32
    }

    pub fn write(&self, off: u32, val: u32, width: usize) {
        if width != 2 || off & 1 != 0 {
            self.odd_access("write", off, width);
            return;
        }
        self.reg_write(off, val as u16);
        self.sync_irq();
    }

    fn odd_access(&self, what: &str, off: u32, width: usize) {
        let first = {
            let mut n = self.odd_accesses.lock();
            *n += 1;
            *n <= 16
        };
        if first {
            self.note(format!("{width}-byte {what} at {off:#04x} is not decoded"));
        }
    }
}

/// The monitor's `scsi`/`cow` subcommands and the run-time disc changes come
/// from `scsi_monitor` through this.
/// The monitor's `scsi`/`cow` commands (`scsi1`/`cow1` for the external
/// bus), as the other SCSI controllers have them, plus `scsi notes`.
impl crate::traits::Device for Isp1040 {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) { self.stop_ring() }
    fn start(&self) { self.start_ring() }
    fn is_running(&self) -> bool { self.worker.lock().is_some() }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        let usage = crate::scsi_monitor::SCSI_USAGE;
        if self.index == 0 {
            vec![
                ("scsi".to_string(), format!("SCSI (QLogic ISP1040, controller 0): {usage} | scsi notes")),
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
            if args.first().copied() == Some("notes") {
                let _ = writeln!(w, "QLogic ISP1040, controller {}: {} commands", self.index, self.total_commands());
                for n in self.notes() {
                    let _ = writeln!(w, "  note: {n}");
                }
                return Ok(());
            }
            return Err(format!("Usage: {} | scsi notes", crate::scsi_monitor::SCSI_USAGE));
        }
        if cmd.starts_with("cow") {
            return crate::scsi_monitor::cow_command(self, args, &mut *w);
        }
        Err("Command not found".to_string())
    }
}

impl crate::scsi_monitor::ScsiTargets for Isp1040 {
    fn scsi_bus(&self) -> &ScsiBus {
        &self.bus
    }
}

/// Both BARs reach the same registers.
impl PciBusDevice for Isp1040 {
    fn config(&self) -> &PciConfig {
        &self.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        self.read(off, width)
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.write(off, val, width)
    }

    fn connect_interrupt(&self, line: IrqLine) {
        self.connect_irq(line);
    }

    fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        let _ = self.dma.set(dma);
    }
}

impl crate::traits::Saveable for Isp1040 {
    fn save_state(&self) -> toml::Value {
        self.quiesced(|st| {
            // Absolute deadlines mean nothing on another run's clock.
            let now = self.now_ns(st);
            let left = st.ctl.selecting.as_ref().map(|sel| sel.due - now);
            // A hold-off already over but not acted on still crosses: the
            // commands parked behind it start when the load re-arms it at 0.
            let quiet_left = st.quiet_until.map(|until| until.map(|u| u - now));
            Self::save_locked(st, left, quiet_left)
        })
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        self.quiesced(|st| {
            let now = self.now_ns(st);
            Self::load_locked(st, v, now);
            self.publish_gen(st);
        });
        self.sync_irq();
        Ok(())
    }
}

impl Isp1040 {
    fn save_locked(st: &State, selecting_left: Option<i64>, quiet_left: [Option<i64>; BUSES]) -> toml::Value {
        let mut t = toml::map::Map::new();
        for (k, v) in [
            ("bus_cfg1", st.bus_cfg1), ("bus_ctrl", st.bus_ctrl), ("bus_status", st.bus_status),
            ("semaphore", st.semaphore), ("nvram", st.nvram), ("risc_flags", st.risc_flags),
            ("dma_cmd_cfg", st.dma_cmd_cfg), ("dma_data_cfg", st.dma_data_cfg),
            ("rqst_len", st.rqst_len), ("resp_len", st.resp_len), ("rqst_cons", st.rqst_cons),
            ("resp_prod", st.resp_prod), ("resp_cons", st.resp_cons), ("reset_delay", st.reset_delay),
            ("selection_timeout", st.selection_timeout),
        ] {
            t.insert(k.into(), hex_u16(v));
        }
        t.insert("rqst_base".into(), hex_u32(st.rqst_base as u32));
        t.insert("rqst_base_hi".into(), hex_u32((st.rqst_base >> 32) as u32));
        t.insert("resp_base".into(), hex_u32(st.resp_base as u32));
        t.insert("resp_base_hi".into(), hex_u32((st.resp_base >> 32) as u32));
        t.insert("mailbox".into(), u16_slice_to_toml(&st.mailbox));
        t.insert("risc_bank".into(), u16_slice_to_toml(&st.risc_bank));
        t.insert("sxp_bank".into(), u16_slice_to_toml(&st.sxp_bank));
        // The downloaded microcode is large and usually all zero between a
        // reset and the driver's download.
        if st.risc_ram.iter().any(|&w| w != 0) {
            t.insert("risc_ram".into(), u16_slice_to_toml(&st.risc_ram));
        }
        let pending = st.pending.iter().map(|p| {
            let mut m = toml::map::Map::new();
            m.insert("handle".into(), hex_u32(p.handle));
            m.insert("scsi_status".into(), hex_u16(p.scsi_status));
            m.insert("completion".into(), hex_u16(p.completion));
            m.insert("state_flags".into(), hex_u16(p.state_flags));
            m.insert("status_flags".into(), hex_u16(p.status_flags));
            m.insert("resid".into(), hex_u32(p.resid));
            m.insert("sense".into(), u8_slice_to_toml(&p.sense));
            toml::Value::Table(m)
        });
        t.insert("pending".into(), toml::Value::Array(pending.collect()));
        // Each bus's hold-off after a bus reset, as the time left, and the
        // commands parked behind it by their ring positions: their entries
        // are in the guest's ring.
        let holds = quiet_left.iter().enumerate().filter_map(|(bus, left)| {
            let mut m = toml::map::Map::new();
            m.insert("bus".into(), toml::Value::Integer(bus as i64));
            m.insert("left_ns".into(), toml::Value::Integer((*left)?.max(0)));
            Some(toml::Value::Table(m))
        });
        t.insert("bus_hold".into(), toml::Value::Array(holds.collect()));
        let parked = st.parked.iter().map(|p| {
            let mut m = toml::map::Map::new();
            m.insert("cons".into(), toml::Value::Integer(i64::from(p.cons)));
            m.insert("count".into(), toml::Value::Integer(i64::from(p.count)));
            m.insert("bus".into(), toml::Value::Integer(i64::from(p.bus)));
            m.insert("handle".into(), hex_u32(p.handle));
            m.insert("started".into(), toml::Value::Boolean(p.started));
            toml::Value::Table(m)
        });
        t.insert("parked".into(), toml::Value::Array(parked.collect()));
        // A selection still timing out: what is left of it, and the
        // completion it ends in.
        if let (Some(sel), Some(left)) = (&st.ctl.selecting, selecting_left) {
            let mut m = toml::map::Map::new();
            m.insert("left_ns".into(), toml::Value::Integer(left.max(0)));
            m.insert("cons".into(), toml::Value::Integer(i64::from(sel.cons)));
            m.insert("orphan".into(), toml::Value::Boolean(sel.orphan));
            m.insert("bus".into(), toml::Value::Integer(i64::from(sel.bus)));
            m.insert("handle".into(), hex_u32(sel.done.handle));
            m.insert("scsi_status".into(), hex_u16(sel.done.scsi_status));
            m.insert("completion".into(), hex_u16(sel.done.completion));
            m.insert("state_flags".into(), hex_u16(sel.done.state_flags));
            m.insert("status_flags".into(), hex_u16(sel.done.status_flags));
            m.insert("resid".into(), hex_u32(sel.done.resid));
            m.insert("sense".into(), u8_slice_to_toml(&sel.done.sense));
            t.insert("selecting".into(), toml::Value::Table(m));
        }
        toml::Value::Table(t)
    }

    /// Unprocessed request entries survive: the ring thread is kicked and
    /// runs them after the load.
    fn load_locked(st: &mut State, v: &toml::Value, now: i64) {
        {
            let ctl = std::mem::take(&mut st.ctl);
            *st = State::new();
            st.ctl = ctl;
            st.ctl.gen += 1;
            st.ctl.rqst_epoch += 1;
            st.ctl.resp_epoch += 1;
            st.ctl.selecting = None;
            st.ctl.work = true;
            let u16f = |k: &str| get_field(v, k).and_then(toml_u16).unwrap_or(0);
            let u32f = |k: &str| get_field(v, k).and_then(toml_u32).unwrap_or(0);
            st.bus_cfg1 = u16f("bus_cfg1");
            st.bus_ctrl = u16f("bus_ctrl");
            st.bus_status = u16f("bus_status");
            st.semaphore = u16f("semaphore");
            st.nvram = u16f("nvram");
            st.risc_flags = u16f("risc_flags");
            st.dma_cmd_cfg = u16f("dma_cmd_cfg");
            st.dma_data_cfg = u16f("dma_data_cfg");
            st.rqst_len = u16f("rqst_len");
            st.resp_len = u16f("resp_len");
            st.rqst_cons = u16f("rqst_cons");
            st.resp_prod = u16f("resp_prod");
            st.resp_cons = u16f("resp_cons");
            st.reset_delay = u16f("reset_delay");
            st.selection_timeout = get_field(v, "selection_timeout").and_then(toml_u16).unwrap_or(DEFAULT_SELECTION_TIMEOUT);
            // Indices fold into their rings, whatever the file says.
            if st.rqst_len != 0 {
                st.rqst_cons %= st.rqst_len;
            }
            if st.resp_len != 0 {
                st.resp_prod %= st.resp_len;
                st.resp_cons %= st.resp_len;
            }
            st.rqst_base = u32f("rqst_base") as u64 | ((u32f("rqst_base_hi") as u64) << 32);
            st.resp_base = u32f("resp_base") as u64 | ((u32f("resp_base_hi") as u64) << 32);
            if let Some(m) = get_field(v, "mailbox") { load_u16_slice(m, &mut st.mailbox); }
            if let Some(m) = get_field(v, "risc_bank") { load_u16_slice(m, &mut st.risc_bank); }
            if let Some(m) = get_field(v, "sxp_bank") { load_u16_slice(m, &mut st.sxp_bank); }
            if let Some(m) = get_field(v, "risc_ram") { load_u16_slice(m, &mut st.risc_ram); }
            let load_pending = |p: &toml::Value| {
                let sense = match get_field(p, "sense") {
                    Some(toml::Value::Array(s)) => s.iter().take(SENSE_MAX).map(|x| toml_u8(x).unwrap_or(0)).collect(),
                    _ => Vec::new(),
                };
                Pending {
                    handle: get_field(p, "handle").and_then(toml_u32).unwrap_or(0),
                    scsi_status: get_field(p, "scsi_status").and_then(toml_u16).unwrap_or(0),
                    completion: get_field(p, "completion").and_then(toml_u16).unwrap_or(0),
                    state_flags: get_field(p, "state_flags").and_then(toml_u16).unwrap_or(0),
                    status_flags: get_field(p, "status_flags").and_then(toml_u16).unwrap_or(0),
                    resid: get_field(p, "resid").and_then(toml_u32).unwrap_or(0),
                    sense,
                }
            };
            if let Some(toml::Value::Array(a)) = get_field(v, "pending") {
                for p in a {
                    st.pending.push_back(load_pending(p));
                }
            }
            let int = |v: &toml::Value, k: &str| match get_field(v, k) {
                Some(toml::Value::Integer(n)) => Some(*n),
                _ => None,
            };
            if let Some(toml::Value::Array(a)) = get_field(v, "bus_hold") {
                for h in a {
                    if let (Some(bus), Some(left)) = (int(h, "bus"), int(h, "left_ns")) {
                        if let Some(slot) = usize::try_from(bus).ok().and_then(|b| st.quiet_until.get_mut(b)) {
                            *slot = Some(now.saturating_add(left.max(0)));
                        }
                    }
                }
            }
            if let Some(toml::Value::Array(a)) = get_field(v, "parked") {
                for p in a {
                    let bus = int(p, "bus").unwrap_or(0).clamp(0, BUSES as i64 - 1) as u8;
                    st.park_seq += 1;
                    let seq = st.park_seq;
                    st.parked.push_back(Parked {
                        seq,
                        // Folded into the ring, whatever the file says.
                        cons: if st.rqst_len == 0 { 0 } else { int(p, "cons").unwrap_or(0).max(0) as u32 % u32::from(st.rqst_len) },
                        count: int(p, "count").unwrap_or(1).clamp(1, 255) as u8,
                        bus,
                        handle: get_field(p, "handle").and_then(toml_u32).unwrap_or(0),
                        started: matches!(get_field(p, "started"), Some(toml::Value::Boolean(true))),
                    });
                }
            }
            // The wait goes on for what was left of it, from this clock.
            if let Some(sel) = get_field(v, "selecting") {
                let left = match get_field(sel, "left_ns") {
                    Some(toml::Value::Integer(n)) => (*n).max(0),
                    _ => 0,
                };
                let cons = match get_field(sel, "cons") {
                    Some(toml::Value::Integer(n)) => *n as u32,
                    _ => 0,
                };
                let orphan = matches!(get_field(sel, "orphan"), Some(toml::Value::Boolean(true)));
                let bus = int(sel, "bus").unwrap_or(0).clamp(0, BUSES as i64 - 1) as u8;
                st.ctl.selecting =
                    Some(Selection { cons, due: now.saturating_add(left), done: load_pending(sel), bus, orphan });
            }
        }
    }
}

#[cfg(test)]
mod tests;
