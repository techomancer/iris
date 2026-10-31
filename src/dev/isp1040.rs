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
//!   does not retire the command in flight either: it completes as it would
//!   have.
//! * Only logical unit 0 of a present target exists. A command for another
//!   unit is passed to the target with that unit number, which answers as
//!   SCSI says (INQUIRY: no device at this unit; anything else: check
//!   condition, logical unit not supported). An absent target times out.
//!
//! Register windows: BAR 0 is I/O space and BAR 1 memory space, both 256
//! bytes onto the same sixteen-bit registers. Accesses of any other width
//! are not decoded: they read zero and writes are dropped, noted for the
//! first sixteen, where the QEMU model fails the access.
//!
//! Ported from hw/scsi/qlogic-isp.c in jamesbraid/qemu-sgi-origami.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::{Condvar, Mutex, MutexGuard};

use crate::devlog::LogModule;
use crate::irq::IrqLine;
use crate::pci::{Bar, PciBusDevice, PciConfig, PciConfigPolicy, PciDma, PciRegisterPolicy, PCI_INTA};
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

pub(crate) const MAILBOX_COUNT: usize = 8;
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
    pub const ABORT_DEVICE: u16 = 0x0016;
    pub const ABORT_TARGET: u16 = 0x0017;
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
    pub const COMMAND_ERROR: u16 = 0x4005;
    pub const PARAM_ERROR: u16 = 0x4006;
}

const RAM_WORDS: usize = 0x10000;
/// The command budget the adapter announces in its firmware status.
pub const MAX_REQUESTS: u16 = 512;
pub(crate) const FW_VERSION: [u16; 3] = [1, 0, 0];

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
pub const IOCB_ABORTED: u16 = 0x0005;
pub const IOCB_DATA_UNDERRUN: u16 = 0x0015;

pub const STATE_GOT_BUS: u16 = 0x0100;
pub const STATE_GOT_SENSE: u16 = 0x2000;
/// Every phase a completed command traverses; the ISP1040 reports this for
/// all of them, gated only on whether sense was collected.
pub const STATE_ALL_PHASES: u16 = 0x7f00;
pub const STATUS_FLAG_TIMEOUT: u16 = 0x0040;

const CHECK_CONDITION: u8 = 0x02;
pub(crate) const MAX_TARGET: u8 = 15;
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

/// What reading a request entry came to.
enum Prepared {
    /// The fabric refused the entry; the walk stops on it.
    Refused,
    /// Consumed with no answer (a marker).
    Skipped,
    /// Answered without reaching a target.
    Answered(Pending),
    Run(Job),
}

/// Part-specific register and firmware behavior; the ring engine owns the state.
/// Defaults preserve the ISP1040 contract, including wrapping firmware RAM.
pub(crate) trait IspPart: Send + Sync {
    fn name(&self) -> &'static str;
    fn channels(&self) -> usize { 1 }
    fn pci_config(&self, bars: [Bar; 6]) -> PciConfig {
        PciConfig::new(PCI_VENDOR_QLOGIC, PCI_DEVICE_ISP1040, PCI_CLASS_SCSI, PCI_REVISION_ISP1040, bars, PCI_INTA)
    }
    fn firmware_version(&self) -> [u16; 3] { FW_VERSION }
    fn active_bank(&self, cfg1: u16) -> Bank {
        if cfg1 & CFG1_BANK_SELECT != 0 { Bank::Sxp } else { Bank::Risc }
    }
    fn checks_ram_bounds(&self) -> bool { false }
    fn has_abort_commands(&self) -> bool { false }
    fn target_on(&self, _channel: u8, target: u8) -> u8 { target }
    fn decode_target(&self, target: u8) -> (usize, u8) { (0, target) }
    fn id_high(&self) -> u16 { PCI_DEVICE_ISP1040 }
    /// The ISP1040 register revision runs one ahead of its PCI revision.
    fn bus_cfg0(&self) -> u16 { PCI_REVISION_ISP1040 as u16 + 1 }
    fn cfg1_write_mask(&self) -> u16 { CFG1_WRITE_MASK }
    fn unmodelled_dma_register(&self, _off: u32) -> bool { false }
    fn completion_phases(&self, _moved: u64, _total: u64) -> u16 { STATE_ALL_PHASES & !STATE_GOT_SENSE }
    fn mailbox_ready(&self, _semaphore: u16) -> bool { true }
    /// Extra target-parameter results, beyond the common asynchronous/narrow reply.
    fn clear_target_parameters(&self, _mailbox: &mut [u16; MAILBOX_COUNT]) {}
    fn set_reset_delay(&self, st: &mut State) { st.reset_delay = st.mailbox[1]; }
    /// Queue target notifications for the worker; the CPU never locks a bus.
    fn reset_targets(&self, _st: &mut State) -> bool { false }
    /// Handle extra mailbox commands, leaving result signalling to the engine.
    fn execute_mailbox(&self, _core: &Core, _st: &mut MutexGuard<'_, State>) -> bool { false }
    fn save_extra(&self, _st: &State, _t: &mut toml::map::Map<String, toml::Value>) {}
    fn load_extra(&self, _st: &mut State, _v: &toml::Value) {}
}

struct Isp1040Part;

impl IspPart for Isp1040Part {
    fn name(&self) -> &'static str { "ISP1040" }
}

/// The register window selected by BUS_CFG1, independent of its encoding.
pub(crate) enum Bank { Risc, Sxp, Sxp1, Dma }

pub(crate) struct State {
    pub(crate) reset_targets: bool,
    pub(crate) pending_target_resets: VecDeque<(u8, Option<u8>)>,
    pub(crate) active: Option<(u8, u8, u32)>,
    cancelled: bool,
    pub(crate) outstanding: Vec<(u8, u8, u32)>,
    pub(crate) sxp1_bank: [u16; BANK_WORDS],
    pub(crate) dma_bank: [u16; BANK_WORDS],
    pub(crate) reset_delay1: u16,
    ctl: Ctl,
    bus_cfg1: u16,
    bus_ctrl: u16,
    bus_status: u16,
    semaphore: u16,
    nvram: u16,
    risc_flags: u16,
    dma_cmd_cfg: u16,
    dma_data_cfg: u16,
    pub(crate) mailbox: [u16; MAILBOX_COUNT],
    risc_bank: [u16; BANK_WORDS],
    sxp_bank: [u16; BANK_WORDS],
    risc_ram: Vec<u16>,
    pub(crate) reset_delay: u16,
    rqst_base: u64,
    resp_base: u64,
    rqst_len: u16,
    resp_len: u16,
    rqst_cons: u16,
    resp_prod: u16,
    resp_cons: u16,
    pending: VecDeque<Pending>,
    total_cmds: u64,
}

impl State {
    fn new() -> Self {
        let mut s = State {
            reset_targets: false,
            pending_target_resets: VecDeque::new(),
            active: None,
            cancelled: false,
            outstanding: Vec::new(),
            sxp1_bank: [0; BANK_WORDS],
            dma_bank: [0; BANK_WORDS],
            reset_delay1: 0,
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
            rqst_base: 0,
            resp_base: 0,
            rqst_len: 0,
            resp_len: 0,
            rqst_cons: 0,
            resp_prod: 0,
            resp_cons: 0,
            pending: VecDeque::new(),
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
        self.sxp1_bank = [0; BANK_WORDS];
        self.dma_bank = [0; BANK_WORDS];
        self.reset_delay1 = 0;
        self.active = None;
        self.cancelled = false;
        self.outstanding.clear();
        // The driver re-downloads firmware after every reset.
        self.risc_ram.iter_mut().for_each(|w| *w = 0);
        self.reset_delay = 0;
        self.rqst_base = 0;
        self.resp_base = 0;
        self.rqst_len = 0;
        self.resp_len = 0;
        self.rqst_cons = 0;
        self.resp_prod = 0;
        self.resp_cons = 0;
        self.pending.clear();
        self.reset_targets = false;
        self.pending_target_resets.clear();
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

    fn active_bank(&mut self, bank: Bank) -> &mut [u16; BANK_WORDS] {
        match bank {
            Bank::Risc => &mut self.risc_bank,
            Bank::Sxp => &mut self.sxp_bank,
            Bank::Sxp1 => &mut self.sxp1_bank,
            Bank::Dma => &mut self.dma_bank,
        }
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
/// either bus and `fence` are reached with it released. `fence` is held
/// across one guest-memory store at a time, and either bus across one
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
    /// Generation for commands cancelled by aborts without resetting the chip.
    cancel: AtomicU64,
    /// Immutable part policy, shared by register, worker and snapshot paths.
    part: &'static dyn IspPart,
    /// The controller's one bus, or both channels of the ISP12160.
    buses: [ScsiBus; 2],
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
        Self { core: Arc::new(Core::new_at(index, &Isp1040Part, [bus, Self::new_bus()])), worker: Mutex::new(None) }
    }

    pub(crate) fn new_12160(bus0: ScsiBus, bus1: ScsiBus) -> Self {
        Self { core: Arc::new(Core::new_at(0, &crate::dev::isp12160::Isp12160Part, [bus0, bus1])), worker: Mutex::new(None) }
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
                .name(format!("{}-{}", self.core.model_name(), self.index))
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
        let mut locks: Vec<(&str, fn(&Core) -> bool)> = vec![
            ("st", (|c: &Core| c.st.is_locked()) as fn(&Core) -> bool),
            ("fence", |c| c.fence.is_locked()),
            ("targets", |c| c.buses[0].is_locked()),
        ];
        if self.core.part.channels() > 1 {
            locks.push(("targets1", |c| c.buses[1].is_locked()));
        }
        for (what, probe) in locks {
            let core = Arc::downgrade(&self.core);
            crate::locks::register_lock(format!("{}-{}::{what}", self.core.model_name().to_ascii_lowercase(), self.index), move || {
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
    fn new_at(index: u8, part: &'static dyn IspPart, buses: [ScsiBus; 2]) -> Self {
        let st = State::new();
        let gen = st.ctl.gen;
        let bars = [Bar::Io(reg::WINDOW_SIZE), Bar::Mem(reg::WINDOW_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused];
        let pci = part.pci_config(bars);
        Self {
            part,
            cancel: AtomicU64::new(0),
            st: Mutex::new(st),
            gen: AtomicU64::new(gen),
            fence: Mutex::new(()),
            wake: Condvar::new(),
            line_order: Mutex::new(()),
            buses,
            dma: OnceLock::new(),
            irq: OnceLock::new(),
            index,
            // BAR 0 is the I/O window and BAR 1 its memory-space alias, one
            // register file.
            pci,
            notes: Mutex::new(Vec::new()),
            dma_misses: Mutex::new(std::collections::BTreeSet::new()),
            odd_accesses: Mutex::new(0),
            heartbeat: OnceLock::new(),
        }
    }

    fn model_name(&self) -> &'static str { self.part.name() }

    pub fn index(&self) -> u8 { self.index }

    /// A machine reset: the chip as at power-on (no firmware, no queues),
    /// its interrupt dropped. The disks stay attached. As the guest's soft
    /// reset, it cancels the command in flight without waiting for its
    /// target.
    pub fn reset(&self) {
        {
            let mut st = self.st.lock();
            st.reset();
            if self.part.reset_targets(&mut st) { self.kick(&mut st); }
            self.publish_gen(&st);
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
    pub fn wait_idle(&self) {
        let mut st = self.st.lock();
        while (st.ctl.work && st.ctl.hold == 0 && !st.ctl.paused) || st.ctl.walking {
            self.wake.wait(&mut st);
        }
    }

    /// Stop the ring thread between entries while the machine is stopped:
    /// once this returns the adapter touches neither guest memory nor its
    /// interrupt until [`Core::resume`]. A command in flight finishes and
    /// posts first.
    pub fn pause(&self) {
        let mut st = self.st.lock();
        st.ctl.paused = true;
        while st.ctl.walking {
            self.wake.wait(&mut st);
        }
    }

    /// Let the ring thread run again, taking up whatever was published
    /// while it was paused.
    pub fn resume(&self) {
        let mut st = self.st.lock();
        st.ctl.paused = false;
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
    pub(crate) fn kick(&self, st: &mut State) {
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
        &self.buses[0]
    }

    pub(crate) fn bus_for_channel(&self, channel: u8) -> Option<&ScsiBus> {
        ((channel as usize) < self.part.channels()).then(|| &self.buses[channel as usize])
    }

    fn bus_for_target(&self, target: u8) -> (&ScsiBus, u8) {
        let (channel, id) = self.part.decode_target(target);
        (&self.buses[channel], id)
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
    /// little-endian. The ISP1040 wraps past the end; the ISP12160 rejects
    /// transfers that would cross it.
    pub(crate) fn ram_transfer(&self, st: &mut MutexGuard<'_, State>, load: bool) -> u16 {
        let base = st.mailbox[1];
        let count = st.mailbox[4] as usize;
        if self.part.checks_ram_bounds() && base as usize + count > RAM_WORDS {
            return mbox::PARAM_ERROR;
        }
        let mut host = Self::mbox_host_addr(&st.mailbox);
        if st.mailbox[0] == mbox::LOAD_RAM_A64 {
            host |= ((st.mailbox[7] as u64) << 32) | ((st.mailbox[6] as u64) << 48);
        }
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
        if self.part.has_abort_commands() && matches!(cmd, mbox::ABORT_COMMAND | mbox::ABORT_DEVICE | mbox::ABORT_TARGET | mbox::BUS_RESET) {
            self.with_dispatch_held(st, |st| self.execute_mailbox_command(st));
        } else {
            self.execute_mailbox_command(st);
        }
    }

    fn execute_mailbox_command(&self, st: &mut MutexGuard<'_, State>) {
        let cmd = st.mailbox[0];
        if self.part.execute_mailbox(self, st) {
            st.semaphore = SEMAPHORE_LOCK;
            st.bus_status |= STATUS_INT_REQ | STATUS_RISC_INT_REQ;
            return;
        }
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
                if self.part.checks_ram_bounds() && start as usize + 4 > RAM_WORDS {
                    st.mailbox[0] = mbox::PARAM_ERROR;
                    st.semaphore = SEMAPHORE_LOCK;
                    st.bus_status |= STATUS_INT_REQ | STATUS_RISC_INT_REQ;
                    return;
                }
                let len = st.risc_ram[start.wrapping_add(3) as usize];
                if self.part.checks_ram_bounds() && start as usize + len as usize > RAM_WORDS {
                    st.mailbox[0] = mbox::PARAM_ERROR;
                    st.semaphore = SEMAPHORE_LOCK;
                    st.bus_status |= STATUS_INT_REQ | STATUS_RISC_INT_REQ;
                    return;
                }
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
                st.mailbox[1..4].copy_from_slice(&self.part.firmware_version());
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
                self.part.clear_target_parameters(&mut st.mailbox);
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
            | mbox::SET_SELECTION_TIMEOUT
            | mbox::SET_INITIATOR_ID
            | mbox::SET_TARGET_PARAMETERS
            | mbox::SET_DEVICE_QUEUE
            | mbox::SET_PCI_CONTROL
            | mbox::SET_SYSTEM_PARAMETER
            | mbox::SET_DATA_OVERRUN_RECOVERY => st.mailbox[0] = mbox::COMPLETE,
            mbox::SET_RESET_DELAY_PARAMS => {
                self.part.set_reset_delay(st);
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
            mbox::BUS_RESET => st.mailbox[0] = mbox::COMPLETE,
            _ => {
                self.note(format!("mailbox command {cmd:#06x} not modelled"));
                st.mailbox[0] = mbox::INVALID;
            }
        }
        st.semaphore = SEMAPHORE_LOCK;
        st.bus_status |= STATUS_INT_REQ | STATUS_RISC_INT_REQ;
    }

    /// Prevent the worker from taking another entry while a mailbox drops `st`.
    fn with_dispatch_held<R>(&self, st: &mut MutexGuard<'_, State>, f: impl FnOnce(&mut MutexGuard<'_, State>) -> R) -> R {
        struct Hold<'a, 'b> {
            core: &'a Core,
            st: &'a mut MutexGuard<'b, State>,
        }
        impl Drop for Hold<'_, '_> {
            fn drop(&mut self) {
                self.st.ctl.hold -= 1;
                self.core.kick(self.st);
            }
        }
        st.ctl.hold += 1;
        let held = Hold { core: self, st };
        f(held.st)
    }

    /// Notify targets with `st` released; a blocked target must not stall register access.
    fn reset_targets(&self, st: &mut MutexGuard<'_, State>, channel: u8, target: Option<u8>) {
        let Some(bus) = self.bus_for_channel(channel) else { return };
        self.with_dispatch_held(st, |st| MutexGuard::unlocked(st, || bus.for_each(|id, dev| {
            if target.is_none_or(|target| target as usize == id) {
                dev.bus_reset();
            }
        })));
    }

    pub(crate) fn retire_outstanding(&self, st: &mut State, matches: impl Fn((u8, u8, u32)) -> bool, completion: u16) -> usize {
        let mut retired = 0;
        let mut keep = Vec::new();
        for item in st.outstanding.drain(..) {
            if matches(item) {
                st.pending.push_back(Pending { handle: item.2, completion, ..Pending::default() });
                retired += 1;
            } else {
                keep.push(item);
            }
        }
        st.outstanding = keep;
        if retired != 0 { self.kick(st); }
        retired
    }

    /// Queue one cancellation reply, then fence its remaining guest-memory stores.
    pub(crate) fn cancel_active(&self, st: &mut MutexGuard<'_, State>, completion: u16) {
        if st.cancelled { return; }
        if let Some((_, _, handle)) = st.active {
            self.cancel.fetch_add(1, Ordering::AcqRel);
            st.cancelled = true;
            st.pending.push_back(Pending { handle, completion, ..Pending::default() });
            self.kick(st);
            MutexGuard::unlocked(st, || self.retire_stores());
        }
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
                if !self.part.mailbox_ready(st.semaphore) {
                    self.note("mailbox submitted with an unacknowledged result".into());
                    return;
                }
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
    fn scsi(&self, gen: u64, cancel: u64, target: u8, cdb: &[u8], data: Option<Vec<u8>>) -> Option<(Vec<u8>, u8)> {
        let (bus, id) = self.bus_for_target(target);
        let mut targets = bus.lock();
        if self.gen.load(Ordering::Acquire) != gen || self.cancel.load(Ordering::Acquire) != cancel {
            return None;
        }
        let Some(dev) = targets.get_mut(id as usize).and_then(|t| t.as_mut()) else {
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
        if channel as usize >= self.part.channels() {
            return Prepared::Answered(Self::incomplete(handle, 0, 0));
        }
        // GOT_BUS alone is Linux's bad-target; the timeout bit beside it is
        // what lets the empty slot be treated as benign.
        let id = target;
        if target > MAX_TARGET || lun > MAX_LUN || !self.buses[channel as usize].has_target(id) {
            return Prepared::Answered(Self::incomplete(handle, STATE_GOT_BUS, STATUS_FLAG_TIMEOUT));
        }

        // The target model reads the unit number from the CDB's second byte.
        cdb[1] = (cdb[1] & 0x1f) | (lun << 5);
        let cdb = &cdb[..(cdb_len as usize).clamp(1, CDB_FIELD_LEN)];
        let block = self
            .buses[channel as usize]
            .lock()
            .get(id as usize)
            .and_then(|t| t.as_ref())
            .and_then(|t| t.block_size())
            .map(|b| b as usize)
            .filter(|&b| b != 0)
            .unwrap_or(BLOCK_SIZE);
        let (xfer, to_dev) = Self::cdb_transfer(cdb, block);
        let (segs, described) = self.read_segments(ring, addr, entry_count, seg_count, xfer as u64, a64);
        let target = self.part.target_on(channel as u8, target);
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
    fn run_job(&self, job: Job, gen: u64, cancel: u64) -> Option<Pending> {
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
            if self.cancel.load(Ordering::Acquire) != cancel { return None; }
            status = self.scsi(gen, cancel, target, &cdb, Some(buf))?.1;
        } else {
            let (data, s) = self.scsi(gen, cancel, target, &cdb, None)?;
            status = s;
            // Offer the target's bytes to the descriptors in order.
            let take = (data.len() as u64).min(described).min(xfer as u64) as usize;
            let mut off = 0usize;
            for &(a, l) in &segs {
                if off >= take { break; }
                let n = (l as usize).min(take - off);
                if !self.fenced(gen, || self.cancel.load(Ordering::Acquire) == cancel && self.dma_write(a, &data[off..off + n]))? {
                    self.note(format!("payload DMA for handle {handle:#x} refused by the fabric; command stopped"));
                    return None;
                }
                off += n;
            }
            moved = off as u64;
        }

        // Auto-sense runs only after a CHECK CONDITION.
        let mut sense = Vec::new();
        if self.cancel.load(Ordering::Acquire) != cancel { return None; }
        if status == CHECK_CONDITION {
            sense = self.scsi(gen, cancel, target, &[scsi_cmd::REQUEST_SENSE, 0, 0, 0, SENSE_MAX as u8, 0], None)?.0;
            sense.truncate(SENSE_MAX);
        }
        let mut scsi_status = status as u16;
        let mut state_flags = self.part.completion_phases(moved, described.max(xfer as u64));
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
            while !st.ctl.shutdown && !(st.ctl.work && st.ctl.hold == 0 && !st.ctl.paused) {
                self.wake.wait(&mut st);
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
    fn read_entry(&self, ring: &Ring, budget: u32) -> (u8, Prepared) {
        let Some(addr) = Self::entry_addr(ring.base, ring.cons) else {
            // Past the top of the bus: refused, and the walk stops.
            self.dma_miss(u64::MAX, false);
            return (1, Prepared::Refused);
        };
        let entry_type = self.ld8(Some(addr));
        let entry_count = self.ld8(Self::at(addr, 1)).max(1).min(budget.min(255) as u8);
        let prepared = match entry_type {
            ENTRY_TYPE_REQUEST | ENTRY_TYPE_COMMAND_A64 => {
                self.prepare_request(ring, addr, entry_count, entry_type == ENTRY_TYPE_COMMAND_A64)
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
            if std::mem::take(&mut st.reset_targets) {
                for channel in 0..self.part.channels() {
                    self.reset_targets(&mut st, channel as u8, None);
                }
                continue;
            }
            if let Some((channel, target)) = st.pending_target_resets.pop_front() {
                self.reset_targets(&mut st, channel, target);
                continue;
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
            let (count, prepared) = MutexGuard::unlocked(&mut st, || self.read_entry(&ring, budget));
            // A reset took the ring away while the entry was read.
            if st.ctl.gen != ring.gen {
                continue;
            }
            let done = match prepared {
                Prepared::Refused => {
                    if st.ctl.rqst_epoch != ring.epoch {
                        continue;
                    }
                    return st;
                }
                Prepared::Skipped => None,
                Prepared::Answered(p) => Some(p),
                Prepared::Run(job) => {
                    st.total_cmds += 1;
                    st.active = Some((job.target, job.cdb.get(1).copied().unwrap_or(0) >> 5, job.handle));
                    st.cancelled = false;
                    st.ctl.busy = true;
                    let cancel = self.cancel.load(Ordering::Acquire);
                    let mut done = MutexGuard::unlocked(&mut st, || self.run_job(job, ring.gen, cancel));
                    if st.cancelled { done = None; }
                    else if self.part.has_abort_commands() && done.is_none() && st.ctl.gen == ring.gen {
                        if let Some(active) = st.active { st.outstanding.push(active); }
                    }
                    st.active = None;
                    st.ctl.busy = false;
                    self.wake.notify_all();
                    // A reset cancelled the command.
                    if st.ctl.gen != ring.gen {
                        continue;
                    }
                    done
                }
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
            return if st.rqst_len == 0 { st.mailbox[MBOX_REQ] } else { st.rqst_cons };
        }
        if (reg::MAILBOX_BASE..MAILBOX_END).contains(&off) {
            return st.mailbox[((off - reg::MAILBOX_BASE) / 2) as usize];
        }
        if (reg::BANK_BASE..BANK_END).contains(&off) {
            let bank = self.part.active_bank(st.bus_cfg1);
            return st.active_bank(bank)[((off - reg::BANK_BASE) / 2) as usize];
        }
        if self.part.unmodelled_dma_register(off) {
            return 0;
        }
        match off {
            reg::ID_LOW => PCI_VENDOR_QLOGIC,
            reg::ID_HIGH => self.part.id_high(),
            reg::BUS_CFG0 => self.part.bus_cfg0(),
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
        if self.part.unmodelled_dma_register(off) {
            self.note(format!("write to ISP12160 DMA register {off:#04x} is not modelled"));
            return;
        }
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
            let bank = self.part.active_bank(st.bus_cfg1);
            st.active_bank(bank)[((off - reg::BANK_BASE) / 2) as usize] = val;
            return;
        }
        match off {
            reg::BUS_CFG1 => {
                let mask = self.part.cfg1_write_mask();
                if val & !mask != 0 {
                    self.note(format!("bus configuration 1 bits {:#06x} are not modelled", val & !mask));
                }
                st.bus_cfg1 = val & mask;
            }
            // Applied after the enables are stored, so the reset wins.
            reg::BUS_CTRL => {
                st.bus_ctrl = val & CTRL_WRITE_MASK;
                // The command in flight is cancelled, not waited for.
                if val & CTRL_SOFT_RESET != 0 {
                    st.reset();
                    if self.part.reset_targets(&mut st) { self.kick(&mut st); }
                    self.publish_gen(&st);
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
        &self.buses[0]
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
        self.quiesced(|st| self.save_locked(st))
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        self.quiesced(|st| {
            self.load_locked(st, v);
            self.cancel.store(0, Ordering::Release);
            self.publish_gen(st);
        });
        self.sync_irq();
        Ok(())
    }
}

impl Isp1040 {
    fn save_locked(&self, st: &State) -> toml::Value {
        let mut t = toml::map::Map::new();
        for (k, v) in [
            ("bus_cfg1", st.bus_cfg1), ("bus_ctrl", st.bus_ctrl), ("bus_status", st.bus_status),
            ("semaphore", st.semaphore), ("nvram", st.nvram), ("risc_flags", st.risc_flags),
            ("dma_cmd_cfg", st.dma_cmd_cfg), ("dma_data_cfg", st.dma_data_cfg),
            ("rqst_len", st.rqst_len), ("resp_len", st.resp_len), ("rqst_cons", st.rqst_cons),
            ("resp_prod", st.resp_prod), ("resp_cons", st.resp_cons), ("reset_delay", st.reset_delay),
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
        self.part.save_extra(st, &mut t);
        toml::Value::Table(t)
    }

    /// Unprocessed request entries survive: the ring thread is kicked and
    /// runs them after the load.
    fn load_locked(&self, st: &mut State, v: &toml::Value) {
        {
            let ctl = std::mem::take(&mut st.ctl);
            *st = State::new();
            st.ctl = ctl;
            self.part.load_extra(st, v);
            st.ctl.gen += 1;
            st.ctl.rqst_epoch += 1;
            st.ctl.resp_epoch += 1;
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
            if let Some(toml::Value::Array(a)) = get_field(v, "pending") {
                for p in a {
                    let sense = match get_field(p, "sense") {
                        Some(toml::Value::Array(s)) => s.iter().take(SENSE_MAX).map(|x| toml_u8(x).unwrap_or(0)).collect(),
                        _ => Vec::new(),
                    };
                    st.pending.push_back(Pending {
                        handle: get_field(p, "handle").and_then(toml_u32).unwrap_or(0),
                        scsi_status: get_field(p, "scsi_status").and_then(toml_u16).unwrap_or(0),
                        completion: get_field(p, "completion").and_then(toml_u16).unwrap_or(0),
                        state_flags: get_field(p, "state_flags").and_then(toml_u16).unwrap_or(0),
                        status_flags: get_field(p, "status_flags").and_then(toml_u16).unwrap_or(0),
                        resid: get_field(p, "resid").and_then(toml_u32).unwrap_or(0),
                        sense,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
