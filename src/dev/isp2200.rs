//! QLogic ISP2200 Fibre Channel host adapter, emulated above its RISC.
//!
//! The driver talks to the adapter's firmware through 24 sixteen-bit
//! mailboxes and two rings of 64-byte entries in host memory. As with the
//! SCSI parts ([`crate::dev::isp1040`]), the firmware a driver downloads is
//! retained, so a checksum can be verified and read back, and never run: the
//! mailbox commands and rings the firmware would provide are done here.
//!
//! No chip programming manual for the part was found. Every register,
//! mailbox and ring fact comes from the public Linux, OpenBSD and NetBSD
//! drivers. Where at least two of them agree the fact is settled; where they
//! are silent or disagree the model makes a choice, and a comment says so.
//!
//! * Registers take 16-bit accesses. A 32-bit access is answered as a
//!   16-bit one at its address; a byte access is not decoded (reads zero,
//!   writes dropped). A real bus would signal an error for the byte
//!   access; `PciFunction::bar_read` and `bar_write` have no error path to
//!   signal it with. Whether the chip takes either is unknown.
//! * Commands run one at a time, in ring order, on the adapter's own
//!   thread. No loop is behind the adapter yet, so every command completes
//!   "port unavailable" with no SCSI status.
//! * Every reset state, the command window that tells a ring pointer write
//!   from a mailbox input, the posting rules and the per-command error
//!   codes are model choices the drivers leave open.
//!
//! Ported from hw/scsi/qlogic-isp2200.c in jamesbraid/qemu-sgi-origami.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::MutexGuard;

use crate::dev::isp1040::{self, ENTRY_SIZE, HOST_CMD_SHIFT, MAX_REQUESTS, PCI_VENDOR_QLOGIC, RAM_WORDS, RISC_FLAG_HOST_INT, RISC_FLAG_PAUSED, RISC_FLAG_RESET, SEMAPHORE_LOCK, SIG1, SIG2, SIG3, ENTRY_TYPE_MARKER, ENTRY_TYPE_STATUS};
use crate::dev::qlogic_ring::{self, IrqLevel, Notes, RingCtl, RingState, RingThread, CONT_OFF_SEGS32, CONT_OFF_SEGS64, CONT_SEGS32, CONT_SEGS64, SEG64_SIZE};
use crate::pci::{Bar, PciConfig, PciFunction, PciFunctionState, PCI_INTA};
use crate::scsi::{ScsiBus, ScsiIdentities, CDROM_IDENTITY, DISK_IDENTITY};

pub const PCI_DEVICE_ISP2200: u16 = 0x2200;
/// Fibre Channel serial bus controller; no driver matches on it.
const PCI_CLASS_FIBRE_CHANNEL: u32 = 0x000c_0400;
/// The retail QLA2200 board's subsystem entry in the PCI ID list: no driver
/// reads it, so it is a model choice.
const PCI_SUBSYSTEM_ID: u32 = (0x0002 << 16) | PCI_VENDOR_QLOGIC as u32;

/// Register offsets.
pub mod reg {
    pub const FLASH_ADDR: u32 = 0x00;
    pub const FLASH_DATA: u32 = 0x02;
    pub const CSR: u32 = 0x06;
    pub const ICR: u32 = 0x08;
    pub const ISR: u32 = 0x0a;
    pub const SEMAPHORE: u32 = 0x0c;
    pub const NVRAM: u32 = 0x0e;
    pub const HCCR: u32 = 0xc0;
    pub const GPIO_DATA: u32 = 0xcc;
    pub const GPIO_ENABLE: u32 = 0xce;
    pub const WINDOW_SIZE: u32 = 0x100;

    const MBOX_LO: u32 = 0x10;
    const MBOX_HI: u32 = 0xe0;
    /// Mailboxes 0-7 sit after the bus interface, 8-23 at the window's end.
    pub const fn mbox(n: usize) -> u32 {
        if n < 8 { MBOX_LO + n as u32 * 2 } else { MBOX_HI + (n as u32 - 8) * 2 }
    }
    pub const REQUEST_QUEUE: u32 = mbox(4);
    pub const RESPONSE_QUEUE: u32 = mbox(5);
}

pub const MAILBOX_COUNT: usize = 24;
type Mailboxes = [u16; MAILBOX_COUNT];

/// The request and response queue pointers share these two mailboxes.
const MBOX_REQUEST: usize = 4;
const MBOX_RESPONSE: usize = 5;

pub const CSR_SOFT_RESET: u16 = 0x0001;
const CSR_BANK_MASK: u16 = 0x0030;

pub const ICR_GLOBAL: u16 = 0x8000;
pub const ICR_RISC: u16 = 0x0008;
const ICR_MASK: u16 = ICR_GLOBAL | ICR_RISC;

pub const ISR_PENDING: u16 = 0x8000;
pub const ISR_RISC: u16 = 0x0008;

/// The NVRAM register's bit-banged serial lines: clock, chip select and the
/// data line into the EEPROM are the host's; the data line out of it is the
/// part's.
const NVRAM_CLK: u16 = 0x0001;
const NVRAM_CS: u16 = 0x0002;
const NVRAM_DI: u16 = 0x0004;
const NVRAM_DO: u16 = 0x0008;
const NVRAM_WRITE_MASK: u16 = 0x0007;

/// A 256-byte image read as 128 little-endian words.
const NVRAM_SIZE: usize = 256;
const NVRAM_WORDS: usize = NVRAM_SIZE / 2;
/// Eleven command bits: start bit 1, read opcode 10, eight address bits.
const NVRAM_CMD_BITS: u8 = 11;
const NVRAM_OP_BITS: u8 = 3;
const NVRAM_OP_READ: u16 = 0x6;
const NVRAM_DATA_BITS: u8 = 16;

/// Name serial bytes are 24 bits. All ones is the "never programmed" value
/// the shipped drivers refuse, and zero is the unset serial.
pub const NVRAM_SERIAL_MAX: u32 = 0xff_fffe;

/// Image offsets of the fields the default image sets.
const NVRAM_OFF_VERSION: usize = 4;
const NVRAM_OFF_PARAM_VERSION: usize = 6;
const NVRAM_OFF_FW_OPTIONS: usize = 8;
const NVRAM_OFF_FRAME_SIZE: usize = 10;
const NVRAM_OFF_IOCB_COUNT: usize = 12;
const NVRAM_OFF_THROTTLE: usize = 14;
const NVRAM_OFF_RETRY_COUNT: usize = 16;
const NVRAM_OFF_RETRY_DELAY: usize = 17;
const NVRAM_OFF_PORT_NAME: usize = 18;
const NVRAM_OFF_NODE_NAME: usize = 30;
const NVRAM_OFF_CHECKSUM: usize = 255;

/// QLogic's company ID, in the middle of every name the part carries.
const COMPANY_ID: [u8; 3] = [0x00, 0xe0, 0x8b];

#[derive(Clone, Copy, PartialEq, Eq)]
enum NvramMode {
    /// Shifting in the eleven command bits.
    Command,
    /// Shifting a word out.
    Data,
    /// Not a read: nothing until chip deselect.
    Ignore,
}

pub const HCCR_HOST_INT: u16 = RISC_FLAG_HOST_INT;
pub const HCCR_RISC_RESET: u16 = RISC_FLAG_RESET;
pub const HCCR_RISC_PAUSED: u16 = RISC_FLAG_PAUSED;

/// What the mailboxes hold once the ROM code has booted.
const MBOX_BUSY: u16 = 0x0004;
/// The ROM revision the 2200A reports, and the only 2200/2200A tell.
const ROM_REVISION_2200A: u16 = 0x0004;

/// Mailbox command codes and completion codes.
pub mod mbox {
    pub use crate::dev::isp1040::mbox::{COMMAND_ERROR, COMPLETE, HOST_INTERFACE_ERROR, INVALID, PARAM_ERROR, TEST_FAILED};

    pub const NOP: u16 = 0x0000;
    /// Load RAM with a 32-bit host address: m2/m3 only, m4 words.
    pub const LOAD_RAM_32: u16 = 0x0001;
    pub const EXEC_FIRMWARE: u16 = 0x0002;
    pub const DUMP_RAM: u16 = 0x0003;
    pub const WRITE_RAM_WORD: u16 = 0x0004;
    pub const READ_RAM_WORD: u16 = 0x0005;
    pub const REGISTER_TEST: u16 = 0x0006;
    pub const VERIFY_CHECKSUM: u16 = 0x0007;
    pub const ABOUT_FIRMWARE: u16 = 0x0008;
    pub const LOAD_RAM: u16 = 0x0009;
    pub const LOAD_RAM_EXTENDED: u16 = 0x000b;
    pub const ABORT_COMMAND: u16 = 0x0015;
    pub const ABORT_TARGET: u16 = 0x0017;
    pub const FIRMWARE_STATUS: u16 = 0x001f;
    pub const GET_ADAPTER_ID: u16 = 0x0020;
    pub const GET_RETRY_COUNT: u16 = 0x0022;
    pub const GET_PORT_QUEUE: u16 = 0x0029;
    pub const SET_RETRY_COUNT: u16 = 0x0032;
    pub const SET_PORT_QUEUE: u16 = 0x0039;
    pub const RESET_LINK_STATUS: u16 = 0x0052;
    pub const INITIATE_LIP: u16 = 0x0062;
    pub const POSITION_MAP: u16 = 0x0063;
    pub const GET_PORT_DB: u16 = 0x0064;
    pub const GET_PORT_NAME: u16 = 0x006a;
    pub const GET_LINK_STATUS: u16 = 0x006b;
    pub const LIP_RESET: u16 = 0x006c;
    pub const FULL_LOGIN_LIP: u16 = 0x0072;
    pub const PORT_NAME_LIST: u16 = 0x0075;
    pub const GET_ID_LIST: u16 = 0x007c;
    pub const LUN_RESET: u16 = 0x007e;
    pub const GET_FW_OPTIONS: u16 = 0x0028;
    pub const SET_FW_OPTIONS: u16 = 0x0038;
    pub const INIT_FIRMWARE: u16 = 0x0060;
    pub const GET_FW_STATE: u16 = 0x0069;
}

/// Asynchronous event codes.
pub mod event {
    pub const REQUEST_DMA_ERROR: u16 = 0x8003;
    pub const RESPONSE_DMA_ERROR: u16 = 0x8004;
    pub const LIP_OCCURRED: u16 = 0x8010;
    pub const LOOP_UP: u16 = 0x8011;
    pub const LIP_RESET: u16 = 0x8013;
    pub const PORT_DB_CHANGED: u16 = 0x8014;
    /// Port-database changed m1: every port, not one that changed.
    pub const PORT_DB_GLOBAL: u16 = 0xffff;
}

const ENTRY_TYPE_CONTINUATION: u8 = 0x02;
const ENTRY_TYPE_CONTINUATION_A64: u8 = 0x0a;
const ENTRY_TYPE_COMMAND: u8 = 0x11;
const ENTRY_TYPE_COMMAND_LIST: u8 = 0x15;
const ENTRY_TYPE_COMMAND_A64: u8 = 0x19;

/// Entry status flags a response entry can carry.
const ENTRY_FLAG_BAD_TYPE: u8 = 0x04;
const ENTRY_FLAG_BAD_PARAM: u8 = 0x08;
const ENTRY_FLAG_BAD_COUNT: u8 = 0x10;
const ENTRY_FLAG_BAD_ORDER: u8 = 0x20;
const ENTRY_FLAG_DMA_ERROR: u8 = 0x40;

const ENTRY_OFF_TYPE: usize = 0;
const ENTRY_OFF_COUNT: usize = 1;
const ENTRY_OFF_STATUS: usize = 3;

const CMD_OFF_HANDLE: usize = 4;
/// High byte of the field at 8.
const CMD_OFF_TARGET: usize = 9;
const CMD_OFF_LUN: usize = 10;
const CMD_OFF_CONTROL: usize = 12;
const CMD_OFF_SEGCOUNT: usize = 18;
const CMD_OFF_CDB: usize = 20;
const CMD_CDB_LEN: usize = 16;
const CMD_OFF_BYTECOUNT: usize = 36;
const CMD_OFF_SEGS: usize = 40;
const CMD_OFF_LIST_TYPE: usize = 40;
const CMD_OFF_LIST_ADDR: usize = 46;

const CMD_SEGS32: usize = 3;
const CMD_SEGS64: usize = 2;

const STATUS_OFF_HANDLE: usize = 4;
const STATUS_OFF_SCSI_STATUS: usize = 8;
const STATUS_OFF_COMPLETION: usize = 10;
const STATUS_OFF_SENSE_LEN: usize = 18;
const STATUS_OFF_RESIDUAL: usize = 20;
const STATUS_OFF_SENSE: usize = 32;
const STATUS_SENSE_MAX: usize = 32;
const STATUS_OFF_SIGNATURE: usize = 60;
/// What Linux writes over the last word of each response it has read.
const RESPONSE_PROCESSED: u32 = 0xdead_dead;

const COMPLETION_PORT_UNAVAILABLE: u16 = 0x0028;

/// A model constant, deliberately not any vendor firmware release.
const FW_VERSION: [u16; 3] = [1, 0, 0];
/// Expanded LUN only: no fabric, no 2K logins.
const FW_ATTRIBUTES: u16 = 0x0002;

/// Get Firmware State: the only three the loop ever reports.
pub const FW_STATE_CONFIG_WAIT: u16 = 0;
pub const FW_STATE_WAIT_AL_PA: u16 = 1;
pub const FW_STATE_READY: u16 = 3;

/// Loop IDs 0..0x7d are loop ports; 0x7e and up are fabric and special.
pub const LOOP_ID_MAX: u8 = 0x7d;
const ICB_SIZE: usize = 96;
const ICB_VERSION: u8 = 1;
const ICB_OFF_OPTIONS: usize = 2;
const ICB_OFF_RQST_LEN: usize = 36;
const ICB_OFF_RESP_LEN: usize = 38;
const ICB_OFF_RQST_BASE: usize = 40;
const ICB_OFF_RESP_BASE: usize = 48;
const ICB_OPT_FAST_POST: u16 = 0x0008;
/// Events waiting for the host to release the result before them. No result
/// is ever dropped, so the queue is sized for every event that can wait: the
/// host interface raises two kinds itself, each queued at most once (see
/// [`State::raise_event`]), and the loop queues at most one bring-up's three.
/// That is five; the queue keeps a power of two.
const EVENTS_MAX: usize = 8;

fn le16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn le64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// Mapping D: m2 bits 31:16, m3 bits 15:0, m6 bits 63:48, m7 bits 47:32.
fn mbox_addr(mb: &Mailboxes) -> u64 {
    (u64::from(mb[6]) << 48) | (u64::from(mb[7]) << 32) | (u64::from(mb[2]) << 16) | u64::from(mb[3])
}

pub struct State {
    /// What the interrupt line was last driven to; see [`Core::drive_irq`].
    irq: IrqLevel,
    csr: u16,
    /// A soft reset in progress: it ends at the next register read, which
    /// still sees it (CSR bit 0 or HCCR bit 6), or at an HCCR write.
    csr_reset_visible: bool,
    icr: u16,
    semaphore: u16,
    /// The EEPROM port: the lines the host drives, and where a read has got to.
    nvram: u16,
    nvram_mode: NvramMode,
    nvram_bits: u8,
    nvram_shift: u16,
    nvram_dout: bool,
    gpio_data: u16,
    gpio_enable: u16,

    /// The RISC interrupt latch, ISR bit 3.
    risc_int: bool,
    /// HCCR bit 7: the doorbell, until cleared.
    host_int: bool,
    /// HCCR bit 6.
    risc_reset: bool,
    /// HCCR bit 5.
    risc_paused: bool,

    /// Two banks: commands read what the host last wrote, the host reads
    /// what the firmware last wrote. A host write always lands in the input
    /// bank, and an input the host did not rewrite keeps its last value. It
    /// reaches the output bank only while no result holds the semaphore, so
    /// a mailbox reads back then, and a posted result stays readable under
    /// the host's writes until its release.
    mbox_in: Mailboxes,
    mbox_out: Mailboxes,
    /// Open from a mailbox 0 write until the doorbell or a reset. Inside it,
    /// a mailbox 4 or 5 write is a command input, never a ring pointer.
    cmd_window: bool,
    /// Doorbell rung but the command not yet run: the RISC is paused, or a
    /// completion already waits in the slot. One latch, as HCCR bit 7 is, so
    /// a second doorbell before the command runs adds nothing.
    cmd_pending: bool,
    /// A command is executing with the state unlocked for its DMA.
    cmd_running: bool,

    fw_running: bool,
    fw_options: [u16; 3],
    risc_ram: Vec<u16>,

    /// The accepted initialisation control block.
    icb: [u8; ICB_SIZE],
    queues_init: bool,
    rqst_base: u64,
    resp_base: u64,
    rqst_len: u16,
    resp_len: u16,
    /// Host producer.
    rqst_in: u16,
    /// Adapter consumer.
    rqst_out: u16,
    /// Adapter producer.
    resp_in: u16,
    /// Host consumer.
    resp_out: u16,

    /// A mailbox completion waiting for the host to release the result
    /// before it. It goes ahead of every queued event.
    completion_held: bool,
    completion: Mailboxes,
    events: VecDeque<Mailboxes>,
    /// Response entries waiting for the host to clear the latch, release the
    /// semaphore or make ring room. Every request entry produces at most
    /// one, and consumption stops once the announced command budget is
    /// held, so the queue never needs more.
    held: VecDeque<[u8; ENTRY_SIZE]>,
    /// What Get Firmware State reports.
    loop_state: u16,
    /// A request-ring read was refused: the entry stays unconsumed until the
    /// next producer write.
    rqst_blocked: bool,
    ctl: Ctl,
}

/// What the register side and the ring thread tell each other: the shared
/// plumbing ([`RingCtl`], reached through `Deref`) and the one thing the ring walk here adds.
#[derive(Default)]
struct Ctl {
    ring: RingCtl,
    /// A thread is writing response entries with the state unlocked.
    posting: bool,
}

impl std::ops::Deref for Ctl {
    type Target = RingCtl;
    fn deref(&self) -> &RingCtl {
        &self.ring
    }
}

impl std::ops::DerefMut for Ctl {
    fn deref_mut(&mut self) -> &mut RingCtl {
        &mut self.ring
    }
}

impl RingState for State {
    fn ring_ctl(&self) -> &RingCtl {
        &self.ctl.ring
    }
    fn ring_ctl_mut(&mut self) -> &mut RingCtl {
        &mut self.ctl.ring
    }
}

/// One SCSI command as the host interface hands it to the Fibre Channel
/// service.
#[derive(Clone, Debug, Default)]
struct Command {
    handle: u32,
    loop_id: u8,
    lun: u16,
    control: u16,
    cdb: [u8; CMD_CDB_LEN],
    byte_count: u32,
    segs: Vec<(u64, u32)>,
}

impl State {
    fn new() -> Self {
        let mut st = State {
            irq: IrqLevel::default(),
            csr: 0,
            csr_reset_visible: false,
            icr: 0,
            semaphore: 0,
            nvram: 0,
            nvram_mode: NvramMode::Command,
            nvram_bits: 0,
            nvram_shift: 0,
            nvram_dout: false,
            gpio_data: 0,
            gpio_enable: 0,
            risc_int: false,
            host_int: false,
            risc_reset: false,
            risc_paused: false,
            mbox_in: [0; MAILBOX_COUNT],
            mbox_out: [0; MAILBOX_COUNT],
            cmd_window: false,
            cmd_pending: false,
            cmd_running: false,
            fw_running: false,
            fw_options: [0; 3],
            risc_ram: vec![0; RAM_WORDS],
            icb: [0; ICB_SIZE],
            queues_init: false,
            rqst_base: 0,
            resp_base: 0,
            rqst_len: 0,
            resp_len: 0,
            rqst_in: 0,
            rqst_out: 0,
            resp_in: 0,
            resp_out: 0,
            completion_held: false,
            completion: [0; MAILBOX_COUNT],
            events: VecDeque::new(),
            held: VecDeque::new(),
            loop_state: FW_STATE_CONFIG_WAIT,
            rqst_blocked: false,
            ctl: Ctl::default(),
        };
        st.power_on();
        st
    }

    /// Power-on and PCI reset leave the RISC running its ROM code, already
    /// showing the product-ID image: OpenBSD reads it at attach before it
    /// resets anything.
    fn power_on(&mut self) {
        self.soft_reset();
        self.csr_reset_visible = false;
        self.gpio_data = 0;
        self.gpio_enable = 0;
    }

    /// HCCR reset holds the RISC here until a release; the CSR soft reset
    /// passes through the same state and boots the ROM code. The firmware
    /// goes with it: every supported driver downloads again after a reset,
    /// and a guest that expects firmware to survive one is unsupported. The
    /// rings go too, and with them everything waiting to post.
    fn hold_risc(&mut self) {
        self.risc_reset = true;
        self.risc_paused = false;
        self.cmd_window = false;
        self.cmd_pending = false;

        self.mbox_in = [0; MAILBOX_COUNT];
        self.mbox_out = [0; MAILBOX_COUNT];
        self.mbox_out[0] = MBOX_BUSY;

        self.fw_running = false;
        self.fw_options = [0; 3];
        self.risc_ram.iter_mut().for_each(|w| *w = 0);

        self.clear_queues();
        self.drop_queued();
        self.loop_state = FW_STATE_CONFIG_WAIT;
        self.ctl.gen += 1;
    }

    /// The mailbox image the ROM code leaves once it has booted.
    fn boot_rom(&mut self) {
        self.mbox_out = [0; MAILBOX_COUNT];
        self.mbox_out[1..4].copy_from_slice(&[SIG1, SIG2, SIG3]);
        self.mbox_out[7] = ROM_REVISION_2200A;
    }

    /// The RISC comes out running ROM code, as from a device reset. The IRIX
    /// 6.5.30 driver polls HCCR bit 6 for the end of the reset with no
    /// release first; the public drivers reset and release the RISC
    /// themselves, so they see their own hold.
    fn soft_reset(&mut self) {
        self.hold_risc();
        self.risc_reset = false;
        self.boot_rom();
        self.csr = 0;
        self.csr_reset_visible = true;
        self.icr = 0;
        self.semaphore = 0;
        self.nvram = 0;
        self.nvram_deselect();
        self.risc_int = false;
        self.host_int = false;
    }

    fn drop_queued(&mut self) {
        self.completion_held = false;
        self.events.clear();
        self.held.clear();
    }

    fn clear_queues(&mut self) {
        self.queues_init = false;
        self.rqst_base = 0;
        self.resp_base = 0;
        self.rqst_len = 0;
        self.resp_len = 0;
        self.rqst_in = 0;
        self.rqst_out = 0;
        self.resp_in = 0;
        self.resp_out = 0;
        self.rqst_blocked = false;
        self.icb = [0; ICB_SIZE];
    }

    fn resp_space(&self) -> usize {
        let used = (usize::from(self.resp_in) + usize::from(self.resp_len) - usize::from(self.resp_out)) % usize::from(self.resp_len);
        // One slot stays empty so that in == out always means empty.
        usize::from(self.resp_len) - 1 - used
    }

    /// A status entry for the host. Every one of its 64 bytes is written,
    /// because Linux finds new entries by a signature it writes over the
    /// last four bytes of each one it has processed.
    fn complete(&mut self, handle: u32, scsi_status: u16, completion: u16, residual: u32, sense: &[u8]) {
        let mut entry = [0u8; ENTRY_SIZE];
        let mut sense_len = sense.len().min(STATUS_SENSE_MAX);
        entry[STATUS_OFF_SENSE..STATUS_OFF_SENSE + sense_len].copy_from_slice(&sense[..sense_len]);
        // Linux would take an entry ending in its own processed mark for one
        // it had already read, and never see it. Sense that ends that way
        // loses its last word instead.
        if u32::from_le_bytes(entry[STATUS_OFF_SIGNATURE..STATUS_OFF_SIGNATURE + 4].try_into().unwrap()) == RESPONSE_PROCESSED {
            sense_len = STATUS_OFF_SIGNATURE - STATUS_OFF_SENSE;
            entry[STATUS_OFF_SIGNATURE..STATUS_OFF_SIGNATURE + 4].fill(0);
        }
        entry[ENTRY_OFF_TYPE] = ENTRY_TYPE_STATUS;
        entry[ENTRY_OFF_COUNT] = 1;
        entry[STATUS_OFF_HANDLE..][..4].copy_from_slice(&handle.to_le_bytes());
        entry[STATUS_OFF_SCSI_STATUS..][..2].copy_from_slice(&scsi_status.to_le_bytes());
        entry[STATUS_OFF_COMPLETION..][..2].copy_from_slice(&completion.to_le_bytes());
        entry[STATUS_OFF_SENSE_LEN..][..2].copy_from_slice(&(sense_len as u16).to_le_bytes());
        entry[STATUS_OFF_RESIDUAL..][..4].copy_from_slice(&residual.to_le_bytes());
        self.held.push_back(entry);
    }

    /// A malformed request goes back as itself, its entry status flagged.
    fn reject_entry(&mut self, req: &[u8; ENTRY_SIZE], flag: u8) {
        let mut entry = *req;
        entry[ENTRY_OFF_STATUS] = flag;
        self.held.push_back(entry);
    }

    /// Whether response entries may be written now.
    fn can_post_entries(&self) -> bool {
        self.queues_init
            && self.can_post()
            && !self.completion_held
            && self.events.is_empty()
            && !self.held.is_empty()
            && self.resp_space() > 0
    }

    /// The line follows the latch when both enables are set.
    fn irq_level(&self) -> bool {
        self.risc_int && self.icr & ICR_MASK == ICR_MASK
    }

    // ---- NVRAM -------------------------------------------------------

    /// Chip deselect, and reset: the part forgets any command in progress.
    fn nvram_deselect(&mut self) {
        self.nvram_mode = NvramMode::Command;
        self.nvram_bits = 0;
        self.nvram_shift = 0;
        self.nvram_dout = false;
    }

    /// A register write to the serial EEPROM port. Only the read command is
    /// modelled, clocked the way the drivers clock it: eleven command bits,
    /// most significant first, each taken on a rising clock edge with chip
    /// select high, then sixteen data bits out, most significant first, each
    /// appearing after a rising edge.
    ///
    /// Approximation: the command format (start bit, two opcode bits, an
    /// eight-bit address) fits a 93C56- or 93C66-class microwire EEPROM in
    /// 16-bit mode; 128 words is a 93C56's size, and the part is not
    /// identified. The model skips zero bits before the start bit, as a
    /// microwire part waits for a one (also inferred), drives no dummy zero
    /// before the first data bit, and does not model write, erase or the
    /// sequential read a real part continues into the next word. A word
    /// address of 128 or more reads 0xffff, whatever a larger part would hold
    /// there. Whether the real part drives a dummy bit is unknown, and it
    /// only matters to a guest that sends a command shorter than eleven bits.
    fn nvram_write(&mut self, image: &[u8; NVRAM_SIZE], val: u16, notes: &mut Vec<String>) {
        let rising = val & NVRAM_CLK != 0 && self.nvram & NVRAM_CLK == 0;
        self.nvram = val & NVRAM_WRITE_MASK;
        if val & NVRAM_CS == 0 {
            self.nvram_deselect();
            return;
        }
        if !rising {
            return;
        }
        match self.nvram_mode {
            NvramMode::Command => {
                if self.nvram_bits == 0 && val & NVRAM_DI == 0 {
                    return;
                }
                self.nvram_shift = (self.nvram_shift << 1) | u16::from(val & NVRAM_DI != 0);
                self.nvram_bits += 1;
                if self.nvram_bits == NVRAM_OP_BITS && self.nvram_shift != NVRAM_OP_READ {
                    notes.push(format!("NVRAM command {:#x} is not a read; only reads are modelled", self.nvram_shift));
                    self.nvram_mode = NvramMode::Ignore;
                } else if self.nvram_bits == NVRAM_CMD_BITS {
                    let word = usize::from(self.nvram_shift & 0xff);
                    self.nvram_shift = if word < NVRAM_WORDS {
                        le16(image, word * 2)
                    } else {
                        notes.push(format!("NVRAM read of word {word} past the image's {NVRAM_WORDS} words"));
                        0xffff
                    };
                    self.nvram_mode = NvramMode::Data;
                    self.nvram_bits = 0;
                }
            }
            NvramMode::Data => {
                if self.nvram_bits < NVRAM_DATA_BITS {
                    self.nvram_dout = (self.nvram_shift >> (NVRAM_DATA_BITS - 1 - self.nvram_bits)) & 1 != 0;
                    self.nvram_bits += 1;
                } else {
                    notes.push("NVRAM clocked past one word; sequential read is not modelled".into());
                    self.nvram_dout = false;
                }
            }
            NvramMode::Ignore => {}
        }
    }

    fn nvram_read(&self) -> u16 {
        let dout = self.nvram_mode == NvramMode::Data && self.nvram_dout;
        self.nvram | if dout { NVRAM_DO } else { 0 }
    }

    // ---- posting -----------------------------------------------------

    /// The payload of the events the host interface raises itself is zero.
    /// Such an event that finds an identical one still queued adds nothing:
    /// the queued one reports the same failure, and the host reads it once it
    /// has released the result before. A guest that keeps retrying a refused
    /// ring therefore cannot grow the queue.
    fn raise_event(&mut self, code: u16) {
        let mut mb = [0; MAILBOX_COUNT];
        mb[0] = code;
        if !self.events.contains(&mb) {
            self.push_event(code, 0);
        }
    }

    fn push_event(&mut self, code: u16, m1: u16) {
        let mut mb = [0; MAILBOX_COUNT];
        mb[0] = code;
        mb[1] = m1;
        // Bounded by its sources; see EVENTS_MAX.
        debug_assert!(self.events.len() < EVENTS_MAX);
        self.events.push_back(mb);
    }

    /// A mailbox result or event writes the output bank, takes the
    /// semaphore lock and sets the RISC interrupt latch.
    fn post_mailbox(&mut self, mb: &Mailboxes) {
        self.mbox_out = *mb;
        self.semaphore |= SEMAPHORE_LOCK;
        self.risc_int = true;
    }

    /// Whether the RISC may write a result or a response entry now. Nothing
    /// is written while the latch is set or the semaphore is held: Linux
    /// clears the latch after walking the ring, so an entry landing in
    /// between would lose its interrupt.
    fn can_post(&self) -> bool {
        !(self.risc_reset || self.risc_paused || self.risc_int || self.semaphore & SEMAPHORE_LOCK != 0)
    }

    /// Post a waiting mailbox result, or event, alone. True if one posted.
    fn post_waiting_mailbox(&mut self) -> bool {
        if !self.can_post() {
            return false;
        }
        if self.completion_held {
            self.completion_held = false;
            let mb = self.completion;
            self.post_mailbox(&mb);
            return true;
        }
        if let Some(mb) = self.events.pop_front() {
            self.post_mailbox(&mb);
            return true;
        }
        false
    }

    fn mbox_index(off: u32) -> Option<usize> {
        match off {
            0x10..=0x1f => Some(((off - 0x10) / 2) as usize),
            0xe0..=0xff => Some(8 + ((off - 0xe0) / 2) as usize),
            _ => None,
        }
    }
}

/// The adapter. Its state lives in a [`Core`] shared with the thread that
/// runs the request ring.
pub struct Isp2200 {
    core: Arc<Core>,
}

impl std::ops::Deref for Isp2200 {
    type Target = Core;
    fn deref(&self) -> &Core {
        &self.core
    }
}

/// The state lock, the fence and the thread are the shared [`RingThread`],
/// reached through `Deref`.
///
/// Guest-memory transfers are made with `st` released, and the loop's bus is
/// never taken with `st` held. The interrupt line is driven under `st`; its
/// parent never calls back down.
/// `fence` is held across one guest-memory store at a time.
pub struct Core {
    ring: RingThread<State>,
    bus: ScsiBus,
    index: u8,
    pci: PciFunctionState,
    /// The board's EEPROM image, fixed when the adapter is built.
    nvram_image: [u8; NVRAM_SIZE],
    notes: Notes,
    odd_accesses: AtomicU64,
    wide_accesses: AtomicU64,
}

impl std::ops::Deref for Core {
    type Target = RingThread<State>;
    fn deref(&self) -> &RingThread<State> {
        &self.ring
    }
}

/// 32-bit FNV-1a over the string's bytes. Defined here, not borrowed from a
/// library, because the port name a guest sees must not change when a
/// library does.
fn fnv1a(s: &str) -> u32 {
    s.bytes().fold(0x811c_9dc5u32, |hash, byte| (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193))
}

impl Isp2200 {
    /// An empty loop: ports are SCSI ids 0 to 0x7d, and the emulated disk
    /// and CD-ROM answer INQUIRY as the other adapters' do.
    pub fn new_bus() -> ScsiBus {
        ScsiBus::new(usize::from(LOOP_ID_MAX) + 1, ScsiIdentities { disk: DISK_IDENTITY, cdrom: CDROM_IDENTITY })
    }

    /// The adapter on an empty loop, for the tests.
    #[cfg(test)]
    pub fn new() -> Self {
        Self::new_at(0, Self::new_bus())
    }

    /// The adapter at `index` on `bus`. Its port name's serial is a hash of
    /// the index, so two adapters in one machine differ and each keeps its
    /// name from run to run.
    pub fn new_at(index: u8, bus: ScsiBus) -> Self {
        let serial = fnv1a(&format!("isp2200-{index}")) % NVRAM_SERIAL_MAX + 1;
        Self { core: Arc::new(Core::new_at(index, bus, serial)) }
    }

    /// Start the ring thread (idempotent) and let it take whatever was
    /// published while the adapter was stopped.
    fn start_ring(&self) {
        self.start_thread(&self.core, format!("isp2200-{}", self.core.index), Core::run_ring);
    }

    /// Park the ring between entries, then end and join the thread. A
    /// command in flight finishes and posts first; afterwards nothing
    /// touches guest memory or the interrupt line.
    fn stop_ring(&self) {
        self.stop_thread();
    }
}

impl Drop for Isp2200 {
    fn drop(&mut self) {
        // Backstop for an adapter dropped while started; the machine stops it.
        self.stop_ring();
    }
}

impl Core {
    fn new_at(index: u8, bus: ScsiBus, serial: u32) -> Self {
        let bars = [Bar::Io(reg::WINDOW_SIZE), Bar::Mem(reg::WINDOW_SIZE), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused];
        let pci = PciConfig::new(PCI_VENDOR_QLOGIC, PCI_DEVICE_ISP2200, PCI_CLASS_FIBRE_CHANNEL, 0, bars, PCI_INTA);
        // The subsystem dword is the board's: a bus reset keeps what is written here.
        pci.write(11, PCI_SUBSYSTEM_ID);
        Self {
            ring: RingThread::new(State::new()),
            bus,
            index,
            pci: PciFunctionState::new(pci),
            nvram_image: Self::nvram_build(serial),
            notes: Notes::default(),
            odd_accesses: AtomicU64::new(0),
            wide_accesses: AtomicU64::new(0),
        }
    }

    pub fn index(&self) -> u8 {
        self.index
    }

    /// The loop: attach targets to it and ask what is on it.
    pub fn bus(&self) -> &ScsiBus {
        &self.bus
    }

    pub(crate) fn bus_for_channel(&self, channel: u8) -> Option<&ScsiBus> {
        (channel == 0).then_some(&self.bus)
    }

    /// The default image of the board's NVRAM.
    ///
    /// Approximation: a model choice, not a capture of any board. The fields
    /// the shipped drivers check are what the image must pass: the sum of its
    /// 256 bytes is zero modulo 256, the port name carries QLogic's company ID
    /// with a first byte of 0x20 or 0x21, and the serial bytes are neither all
    /// ones nor the whole image blank. The remaining fields are the values the
    /// public OpenBSD driver uses when it has no image, so a guest that copies
    /// them into its control block behaves as it does without one. The board's
    /// real contents are unknown.
    fn nvram_build(serial: u32) -> [u8; NVRAM_SIZE] {
        let mut img = [0u8; NVRAM_SIZE];
        img[..4].copy_from_slice(b"ISP ");
        img[NVRAM_OFF_VERSION] = 1;
        img[NVRAM_OFF_PARAM_VERSION] = 1;
        img[NVRAM_OFF_FW_OPTIONS..][..2].copy_from_slice(&0x0107u16.to_le_bytes());
        img[NVRAM_OFF_FRAME_SIZE..][..2].copy_from_slice(&1024u16.to_le_bytes());
        img[NVRAM_OFF_IOCB_COUNT..][..2].copy_from_slice(&256u16.to_le_bytes());
        img[NVRAM_OFF_THROTTLE..][..2].copy_from_slice(&16u16.to_le_bytes());
        img[NVRAM_OFF_RETRY_COUNT] = 3;
        img[NVRAM_OFF_RETRY_DELAY] = 5;
        for (offset, first) in [(NVRAM_OFF_PORT_NAME, 0x21), (NVRAM_OFF_NODE_NAME, 0x20)] {
            let name = &mut img[offset..offset + 8];
            name[0] = first;
            name[2..5].copy_from_slice(&COMPANY_ID);
            name[5..8].copy_from_slice(&serial.to_be_bytes()[1..]);
        }
        let sum = img[..NVRAM_OFF_CHECKSUM].iter().fold(0u8, |sum, byte| sum.wrapping_add(*byte));
        img[NVRAM_OFF_CHECKSUM] = sum.wrapping_neg();
        img
    }

    // ---- notes -------------------------------------------------------

    /// A running commentary on the odd things a driver did (accesses of a
    /// width the part does not decode, DMA that is not memory, commands not
    /// modelled).
    pub fn notes(&self) -> Vec<String> {
        self.notes.list()
    }

    fn note(&self, what: String) {
        self.notes.push(what);
    }

    fn note_all(&self, what: Vec<String>) {
        what.into_iter().for_each(|what| self.note(what));
    }

    // ---- interrupt ---------------------------------------------------

    /// True while the adapter asserts its interrupt line.
    pub fn interrupting(&self) -> bool {
        self.st.lock().irq_level()
    }

    /// Drive the line to the level the state gives, writing it only when it
    /// differs from what the line last held. Call it with `st` held, in the
    /// same hold that changed the state: the edges the parent sees are then
    /// the state's changes, in order, and a fall and a rise that follow each
    /// other are two edges and not a steady high.
    fn drive_irq(&self, st: &mut State) {
        let level = st.irq_level();
        st.irq.drive(level, self.pci.irq());
    }

    /// What the adapter checks before it lets go of `st` after a change: the
    /// line is where the state says. A change driven in a later hold, which
    /// would let another thread's fall and rise coalesce between the two,
    /// fails here. Compiled only into tests.
    #[cfg(test)]
    fn assert_driven(&self, st: &State) {
        if self.pci.irq().is_some() {
            assert_eq!(st.irq.last(), Some(st.irq_level()), "the state changed and the line was not driven in the same hold");
        }
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn assert_driven(&self, _st: &State) {}

    /// As [`Core::drive_irq`], writing the line whatever it last held.
    fn resync_irq(&self) {
        let mut st = self.st.lock();
        st.irq.forget();
        let level = st.irq_level();
        st.irq.drive(level, self.pci.irq());
    }

    // ---- DMA ---------------------------------------------------------

    /// Every adapter DMA depends on the host having enabled bus mastering:
    /// with it clear a transfer that would start is refused and noted.
    fn bus_master(&self) -> bool {
        self.pci.config().command() & 0x4 != 0
    }

    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let ok = qlogic_ring::dma_read(&self.pci, self.bus_master(), addr, buf);
        if !ok {
            self.note(format!("DMA read of {addr:#x} refused"));
        }
        ok
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> bool {
        let ok = qlogic_ring::dma_write(&self.pci, self.bus_master(), addr, data);
        if !ok {
            self.note(format!("DMA write of {addr:#x} refused"));
        }
        ok
    }

    // ---- mailbox commands --------------------------------------------

    /// The whole transfer is read before a word of it is kept, so a refused
    /// DMA leaves RAM as it was and a later checksum cannot blame the image
    /// for bytes the adapter never received.
    fn load_ram(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes) -> u16 {
        let base = usize::from(input[1]);
        let count = usize::from(input[4]);
        let mut host = mbox_addr(input);
        // The short form names no m6 or m7, whatever an earlier command left.
        if input[0] == mbox::LOAD_RAM_32 {
            host &= 0xffff_ffff;
        }
        if input[0] == mbox::LOAD_RAM_EXTENDED && input[8] != 0 {
            self.note(format!("extended RAM load above word 0xffff (m8 {:#06x}) refused", input[8]));
            return mbox::PARAM_ERROR;
        }
        if base + count > RAM_WORDS {
            return mbox::PARAM_ERROR;
        }
        if count == 0 {
            return mbox::COMPLETE;
        }
        let mut bytes = vec![0u8; count * 2];
        if !MutexGuard::unlocked(st, || self.dma_read(host, &mut bytes)) {
            return mbox::HOST_INTERFACE_ERROR;
        }
        for (i, word) in bytes.chunks_exact(2).enumerate() {
            st.risc_ram[base + i] = u16::from_le_bytes([word[0], word[1]]);
        }
        mbox::COMPLETE
    }

    /// The reverse of Load RAM: RAM words out to the host, little-endian.
    fn dump_ram(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes) -> u16 {
        let base = usize::from(input[1]);
        let count = usize::from(input[4]);
        let host = mbox_addr(input);
        if base + count > RAM_WORDS {
            return mbox::PARAM_ERROR;
        }
        if count == 0 {
            return mbox::COMPLETE;
        }
        let bytes: Vec<u8> = st.risc_ram[base..base + count].iter().flat_map(|word| word.to_le_bytes()).collect();
        if !MutexGuard::unlocked(st, || self.dma_write(host, &bytes)) {
            return mbox::HOST_INTERFACE_ERROR;
        }
        mbox::COMPLETE
    }

    /// Word 3 of an image is its length; a good image sums to zero. The real
    /// algorithm is unknown; this follows the SCSI parts.
    fn verify_checksum(st: &State, input: &Mailboxes, out: &mut Mailboxes) -> u16 {
        let start = usize::from(input[1]);
        out[1] = input[1];
        if start + 3 >= RAM_WORDS {
            return mbox::PARAM_ERROR;
        }
        let len = usize::from(st.risc_ram[start + 3]);
        if start + len > RAM_WORDS {
            return mbox::PARAM_ERROR;
        }
        let sum = st.risc_ram[start..start + len].iter().fold(0u16, |sum, word| sum.wrapping_add(*word));
        out[1] = sum;
        out[2] = sum;
        if sum == 0 { mbox::COMPLETE } else { mbox::TEST_FAILED }
    }

    /// INIT_FIRMWARE reads the 96-byte, version-1 block by DMA. The model
    /// takes the request and response ring lengths at offsets 36 and 38 and
    /// their 64-bit bases at 40 and 48, keeps the rest for the Fibre Channel
    /// service, sets all four ring pointers to 0 and marks the rings built.
    ///
    /// It refuses with `HOST_INTERFACE_ERROR` a block it cannot read, and with
    /// `PARAM_ERROR` a version other than 1, a zero length, a zero base or the
    /// fast-post option, which would need completion events this model does not
    /// post. A reduced-interrupt timer mode -- Linux asks for mode 4 on this
    /// part -- is accepted, noted and ignored: every completion is an ordinary
    /// status entry.
    fn init_firmware(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes) -> u16 {
        let addr = mbox_addr(input);
        let mut icb = [0u8; ICB_SIZE];
        if !MutexGuard::unlocked(st, || self.dma_read(addr, &mut icb)) {
            self.note(format!("initialisation control block at {addr:#x} unreadable"));
            return mbox::HOST_INTERFACE_ERROR;
        }
        let options = le16(&icb, ICB_OFF_OPTIONS);
        let rqst_len = le16(&icb, ICB_OFF_RQST_LEN);
        let resp_len = le16(&icb, ICB_OFF_RESP_LEN);
        let rqst_base = le64(&icb, ICB_OFF_RQST_BASE);
        let resp_base = le64(&icb, ICB_OFF_RESP_BASE);
        if icb[0] != ICB_VERSION || rqst_len == 0 || resp_len == 0 || rqst_base == 0 || resp_base == 0 {
            self.note(format!(
                "initialisation control block refused: version {}, request ring {rqst_base:#x}/{rqst_len}, response ring {resp_base:#x}/{resp_len}",
                icb[0]
            ));
            return mbox::PARAM_ERROR;
        }
        if options & ICB_OPT_FAST_POST != 0 {
            self.note("fast posting is not modelled".into());
            return mbox::PARAM_ERROR;
        }
        // A re-initialisation replaces the rings; completions held for the old
        // ones, or still to come from commands in flight, would land in the
        // new ones under stale handles.
        st.ctl.gen += 1;
        st.held.clear();
        st.icb = icb;
        st.queues_init = true;
        st.rqst_base = rqst_base;
        st.resp_base = resp_base;
        st.rqst_len = rqst_len;
        st.resp_len = resp_len;
        st.rqst_in = 0;
        st.rqst_out = 0;
        st.resp_in = 0;
        st.resp_out = 0;
        st.rqst_blocked = false;
        mbox::COMPLETE
    }

    /// The ROM answers these before any firmware is executed.
    fn rom_command(cmd: u16) -> bool {
        matches!(
            cmd,
            mbox::NOP
                | mbox::LOAD_RAM_32
                | mbox::DUMP_RAM
                | mbox::EXEC_FIRMWARE
                | mbox::WRITE_RAM_WORD
                | mbox::READ_RAM_WORD
                | mbox::REGISTER_TEST
                | mbox::VERIFY_CHECKSUM
                | mbox::ABOUT_FIRMWARE
                | mbox::LOAD_RAM
                | mbox::LOAD_RAM_EXTENDED
        )
    }

    /// Abort device, bus reset and target reset, which no supported driver
    /// issues; queue setup the SCSI parts use; resource counts.
    fn deferred_command(cmd: u16) -> bool {
        matches!(cmd, 0x0010 | 0x0011 | 0x0014 | 0x0016 | 0x0018 | 0x0042 | 0x0066)
    }

    /// One mailbox command, from the input bank into a fresh output bank.
    /// Output mailboxes a command does not name read zero.
    fn dispatch(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes, out: &mut Mailboxes) -> u16 {
        let cmd = input[0];
        if !st.fw_running && !Self::rom_command(cmd) {
            self.note(format!("mailbox command {cmd:#06x} before firmware is executed"));
            return mbox::INVALID;
        }
        match cmd {
            mbox::NOP => mbox::COMPLETE,
            mbox::REGISTER_TEST => {
                out[1..8].copy_from_slice(&input[1..8]);
                mbox::COMPLETE
            }
            mbox::WRITE_RAM_WORD => {
                st.risc_ram[usize::from(input[1])] = input[2];
                mbox::COMPLETE
            }
            mbox::READ_RAM_WORD => {
                out[2] = st.risc_ram[usize::from(input[1])];
                mbox::COMPLETE
            }
            mbox::LOAD_RAM | mbox::LOAD_RAM_32 | mbox::LOAD_RAM_EXTENDED => {
                out[1] = input[1];
                self.load_ram(st, input)
            }
            mbox::DUMP_RAM => self.dump_ram(st, input),
            mbox::VERIFY_CHECKSUM => Self::verify_checksum(st, input, out),
            // Nothing runs: the model is the firmware.
            mbox::EXEC_FIRMWARE => {
                st.fw_running = true;
                mbox::COMPLETE
            }
            mbox::ABOUT_FIRMWARE => {
                out[1..4].copy_from_slice(&FW_VERSION);
                out[6] = FW_ATTRIBUTES;
                mbox::COMPLETE
            }
            mbox::FIRMWARE_STATUS => {
                out[2] = MAX_REQUESTS;
                mbox::COMPLETE
            }
            mbox::INIT_FIRMWARE => self.init_firmware(st, input),
            mbox::GET_FW_STATE => {
                out[1] = st.loop_state;
                mbox::COMPLETE
            }
            // Stored and returned; what the options would change is not modelled.
            mbox::GET_FW_OPTIONS => {
                out[1..4].copy_from_slice(&st.fw_options);
                mbox::COMPLETE
            }
            mbox::SET_FW_OPTIONS => {
                st.fw_options.copy_from_slice(&input[1..4]);
                mbox::COMPLETE
            }
            _ => {
                if Self::deferred_command(cmd) {
                    self.note(format!("mailbox command {cmd:#06x} not modelled"));
                } else {
                    self.note(format!("unknown mailbox command {cmd:#06x}"));
                }
                mbox::INVALID
            }
        }
    }

    /// Runs only with the slot free, so its result always has a place.
    fn mailbox_command(&self, st: &mut MutexGuard<'_, State>) {
        debug_assert!(!st.completion_held);
        // The firmware takes the doorbell by clearing it. The stock PROM's
        // driver waits for bit 7 to drop and never writes the clear command.
        st.host_int = false;
        let input = st.mbox_in;
        let mut out = [0; MAILBOX_COUNT];
        out[0] = self.dispatch(st, &input, &mut out);
        st.completion = out;
        st.completion_held = true;
        self.post_waiting(st);
    }

    /// A latched doorbell runs once the RISC runs and the slot is free. The
    /// slot frees only when a result posts, which the host must then release,
    /// so the register write that follows a post always gets here.
    fn run_pending(&self, st: &mut MutexGuard<'_, State>) {
        if st.cmd_pending && !st.risc_reset && !st.risc_paused && !st.completion_held && !st.cmd_running {
            st.cmd_pending = false;
            st.cmd_running = true;
            self.mailbox_command(st);
            st.cmd_running = false;
        }
    }

    /// Post what is waiting for the host, on whichever thread made it
    /// postable. A recovery retires the command the ring thread is blocked
    /// on, and its entry must not wait for that disk. Only one thread writes
    /// entries at a time.
    fn post_waiting(&self, st: &mut MutexGuard<'_, State>) {
        while self.pump(st) {}
    }

    /// One post: a waiting mailbox result or event, alone, or a batch of
    /// response entries. True if one was made.
    fn pump(&self, st: &mut MutexGuard<'_, State>) -> bool {
        if st.post_waiting_mailbox() {
            return true;
        }
        if st.can_post_entries() && !st.ctl.posting {
            st.ctl.posting = true;
            self.post_entries(st);
            st.ctl.posting = false;
            self.wake.notify_all();
            return true;
        }
        false
    }

    // ---- the ring thread ---------------------------------------------

    /// Take the clock off the host's and set it to `ns` (tests). The thread
    /// reads the clock with `st` held and then waits on `wake` with no
    /// timeout, so the move is made with `st` held too.
    #[cfg(test)]
    pub(crate) fn set_test_clock(&self, ns: i64) {
        let mut st = self.st.lock();
        self.fake_ns.store(ns, Ordering::Relaxed);
        self.kick(&mut st);
    }

    /// Block until the ring thread has run everything it was given and is
    /// waiting for more.
    pub fn wait_idle(&self) {
        let mut st = self.st.lock();
        while (st.ctl.hold == 0 && !st.ctl.paused && st.ctl.work) || st.ctl.walking {
            self.wake.wait(&mut st);
        }
    }

    fn run_ring(&self) {
        let mut st = self.st.lock();
        loop {
            while !st.ctl.shutdown && !(st.ctl.hold == 0 && !st.ctl.paused && st.ctl.work) {
                self.wake.wait(&mut st);
            }
            if st.ctl.shutdown {
                return;
            }
            st.ctl.work = false;
            st.ctl.walking = true;
            st = self.walk(st);
            st.ctl.walking = false;
            self.wake.notify_all();
        }
    }

    /// Post what is waiting and take the request ring's entries until
    /// neither can go further.
    fn walk<'a>(&'a self, mut st: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        loop {
            if st.ctl.shutdown || st.ctl.hold > 0 || st.ctl.paused {
                // Resume once the holder lets go.
                st.ctl.work = true;
                return st;
            }
            if self.pump(&mut st) {
                self.drive_irq(&mut st);
                continue;
            }
            if !self.consume_one(&mut st) {
                return st;
            }
            self.drive_irq(&mut st);
        }
    }

    /// Writes held response entries into the ring, as many as fit, then
    /// advances response-in once and sets the latch once. A refused write
    /// drops its entry and queues an event in its place.
    fn post_entries(&self, st: &mut MutexGuard<'_, State>) {
        let take = st.resp_space().min(st.held.len());
        let gen = st.ctl.gen;
        let batch: Vec<(u64, [u8; ENTRY_SIZE])> = st
            .held
            .iter()
            .take(take)
            .enumerate()
            .map(|(i, entry)| {
                let slot = (usize::from(st.resp_in) + i) % usize::from(st.resp_len);
                (st.resp_base.wrapping_add(slot as u64 * ENTRY_SIZE as u64), *entry)
            })
            .collect();
        // Whole 64 bytes: Linux finds new entries by their last word.
        let (written, refused) = MutexGuard::unlocked(st, || {
            let mut written = 0;
            for (addr, entry) in &batch {
                match self.fenced(gen, || self.dma_write(*addr, entry)) {
                    Some(true) => written += 1,
                    Some(false) => {
                        self.note(format!("response entry write to {addr:#x} refused; entry dropped"));
                        return (written, true);
                    }
                    None => return (written, false),
                }
            }
            (written, false)
        });
        // A reset took the rings away while the entries were written.
        if st.ctl.gen != gen {
            return;
        }
        st.held.drain(..written + usize::from(refused));
        if refused {
            st.raise_event(event::RESPONSE_DMA_ERROR);
        }
        if written > 0 {
            st.resp_in = ((usize::from(st.resp_in) + written) % usize::from(st.resp_len)) as u16;
            st.risc_int = true;
        }
    }

    /// The adapter consumes the request ring as fast as the host fills it, so
    /// the ring is never seen full. It stops short only when the entry's
    /// continuations are not all published yet, when a read is refused, or
    /// when the announced command budget -- commands in flight plus
    /// completions waiting to post -- is used up. True if it did anything
    /// the walk should look at again.
    fn consume_one(&self, st: &mut MutexGuard<'_, State>) -> bool {
        if !st.queues_init || st.risc_reset || st.risc_paused || st.rqst_out == st.rqst_in || st.rqst_blocked {
            return false;
        }
        if st.held.len() >= usize::from(MAX_REQUESTS) {
            return false;
        }
        let gen = st.ctl.gen;
        let len = usize::from(st.rqst_len);
        let out = usize::from(st.rqst_out);
        let avail = (usize::from(st.rqst_in) + len - out) % len;
        let addr = st.rqst_base.wrapping_add(out as u64 * ENTRY_SIZE as u64);
        let mut req = [0u8; ENTRY_SIZE];
        let read = MutexGuard::unlocked(st, || self.dma_read(addr, &mut req));
        if st.ctl.gen != gen {
            return true;
        }
        // Not consumed: the next producer write tries the entry again.
        if !read {
            self.note(format!("request entry read at {addr:#x} refused"));
            st.rqst_blocked = true;
            st.raise_event(event::REQUEST_DMA_ERROR);
            return true;
        }
        let count = usize::from(req[ENTRY_OFF_COUNT]);
        let mut consumed = 1;
        match req[ENTRY_OFF_TYPE] {
            ENTRY_TYPE_COMMAND | ENTRY_TYPE_COMMAND_A64 | ENTRY_TYPE_COMMAND_LIST => {
                if count == 0 || count >= len {
                    st.reject_entry(&req, ENTRY_FLAG_BAD_COUNT);
                } else if count > avail {
                    // The host has not published the whole command yet.
                    return false;
                } else {
                    consumed = count;
                    self.command_entry(st, &req, count);
                }
            }
            // A marker resynchronises firmware state; nothing is posted.
            ENTRY_TYPE_MARKER => {}
            ENTRY_TYPE_CONTINUATION | ENTRY_TYPE_CONTINUATION_A64 => st.reject_entry(&req, ENTRY_FLAG_BAD_ORDER),
            _ => st.reject_entry(&req, ENTRY_FLAG_BAD_TYPE),
        }
        // A reset inside the dispatch has taken the ring with it.
        if st.ctl.gen != gen || !st.queues_init {
            return true;
        }
        st.rqst_out = ((out + consumed) % len) as u16;
        true
    }

    /// A command entry and the continuations its entry count claims. The
    /// count, not each entry's type, says how many slots the command takes:
    /// a 64-bit continuation has no handle, only segment data from byte 4.
    fn command_entry(&self, st: &mut MutexGuard<'_, State>, req: &[u8; ENTRY_SIZE], count: usize) {
        let kind = req[ENTRY_OFF_TYPE];
        let a64 = kind != ENTRY_TYPE_COMMAND;
        let segcount = usize::from(le16(req, CMD_OFF_SEGCOUNT));
        let mut cmd = Command {
            handle: u32::from_le_bytes(req[CMD_OFF_HANDLE..][..4].try_into().unwrap()),
            loop_id: req[CMD_OFF_TARGET],
            lun: le16(req, CMD_OFF_LUN),
            control: le16(req, CMD_OFF_CONTROL),
            byte_count: u32::from_le_bytes(req[CMD_OFF_BYTECOUNT..][..4].try_into().unwrap()),
            ..Default::default()
        };
        cmd.cdb.copy_from_slice(&req[CMD_OFF_CDB..CMD_OFF_CDB + CMD_CDB_LEN]);
        let gen = st.ctl.gen;

        if kind == ENTRY_TYPE_COMMAND_LIST {
            let list = le64(req, CMD_OFF_LIST_ADDR);
            let list_type = le16(req, CMD_OFF_LIST_TYPE);
            if list_type != 1 {
                self.note(format!("segment list type {list_type}, read as type 1"));
            }
            if segcount != 0 {
                let mut buf = vec![0u8; segcount * SEG64_SIZE];
                let read = MutexGuard::unlocked(st, || self.dma_read(list, &mut buf));
                if st.ctl.gen != gen {
                    return;
                }
                if !read {
                    st.reject_entry(req, ENTRY_FLAG_DMA_ERROR);
                    return;
                }
                qlogic_ring::parse_segments(&mut cmd.segs, &buf, segcount, true);
            }
            self.submit(st, cmd);
            return;
        }

        let (inline_segs, cont_segs, cont_off, cont_type) = if a64 {
            (CMD_SEGS64, CONT_SEGS64, CONT_OFF_SEGS64, ENTRY_TYPE_CONTINUATION_A64)
        } else {
            (CMD_SEGS32, CONT_SEGS32, CONT_OFF_SEGS32, ENTRY_TYPE_CONTINUATION)
        };
        if segcount > inline_segs + (count - 1) * cont_segs {
            st.reject_entry(req, ENTRY_FLAG_BAD_PARAM);
            return;
        }
        qlogic_ring::parse_segments(&mut cmd.segs, &req[CMD_OFF_SEGS..], segcount.min(inline_segs), a64);
        let len = usize::from(st.rqst_len);
        for i in 1..count {
            if cmd.segs.len() >= segcount {
                break;
            }
            let index = (usize::from(st.rqst_out) + i) % len;
            let addr = st.rqst_base.wrapping_add(index as u64 * ENTRY_SIZE as u64);
            let mut cont = [0u8; ENTRY_SIZE];
            let read = MutexGuard::unlocked(st, || self.dma_read(addr, &mut cont));
            if st.ctl.gen != gen {
                return;
            }
            if !read {
                st.reject_entry(req, ENTRY_FLAG_DMA_ERROR);
                return;
            }
            if cont[ENTRY_OFF_TYPE] != cont_type {
                st.reject_entry(req, ENTRY_FLAG_BAD_ORDER);
                return;
            }
            let want = (segcount - cmd.segs.len()).min(cont_segs);
            qlogic_ring::parse_segments(&mut cmd.segs, &cont[cont_off..], want, a64);
        }
        self.submit(st, cmd);
    }

    // ---- commands ------------------------------------------------------

    /// A command from the host interface. No loop is behind the adapter yet,
    /// so every command completes "port unavailable" with no SCSI status, as a
    /// disconnected adapter's do.
    fn submit(&self, st: &mut MutexGuard<'_, State>, cmd: Command) {
        crate::dlog!(
            crate::devlog::LogModule::Scsi,
            "isp2200: command {:#x} to loop ID {} LUN {} (cdb {:02x?}, control {:#06x}, {} bytes in {} segments) has no loop",
            cmd.handle,
            cmd.loop_id,
            cmd.lun,
            &cmd.cdb[..],
            cmd.control,
            cmd.byte_count,
            cmd.segs.len()
        );
        st.complete(cmd.handle, 0, COMPLETION_PORT_UNAVAILABLE, 0, &[]);
    }

    // ---- registers ---------------------------------------------------

    /// Bits 15:12 are the command; the rest of a command not modelled is
    /// ignored.
    fn host_command(&self, st: &mut MutexGuard<'_, State>, val: u16) {
        match val >> HOST_CMD_SHIFT {
            0x0 => {}
            0x1 => {
                st.hold_risc();
            }
            0x2 => {
                st.risc_paused = true;
                st.mbox_out[0] = MBOX_BUSY;
            }
            0x3 => self.release_risc(st),
            // The doorbell: the command is in the mailboxes.
            0x5 => {
                st.host_int = true;
                st.cmd_window = false;
                if st.risc_reset {
                    self.note(format!("mailbox command {:#06x} dropped: the RISC is held in reset", st.mbox_in[0]));
                    return;
                }
                if st.cmd_pending {
                    self.note("doorbell rung again before the last command ran; one command runs".into());
                }
                st.cmd_pending = true;
                self.run_pending(st);
            }
            0x6 => st.host_int = false,
            0x7 => {
                st.risc_int = false;
                self.post_waiting(st);
                self.kick(st);
            }
            // Parity masking or single step at 0x4000, parity pause at
            // 0x4001, and the diagnostic commands: accepted, and none changes
            // the RISC.
            0x4 | 0x8 | 0x9 | 0xa | 0xe | 0xf => {
                self.note(format!("host command {val:#06x} accepted, not modelled"));
            }
            _ => self.note(format!("unknown host command {val:#06x}")),
        }
    }

    fn release_risc(&self, st: &mut MutexGuard<'_, State>) {
        if st.risc_reset {
            st.risc_reset = false;
            st.risc_paused = false;
            st.boot_rom();
            return;
        }
        if !st.risc_paused {
            return;
        }
        st.risc_paused = false;
        st.mbox_out[0] = 0;
        self.run_pending(st);
        self.post_waiting(st);
        self.kick(st);
    }

    fn mailbox_write(&self, st: &mut State, idx: usize, val: u16) {
        if idx == 0 {
            st.cmd_window = true;
        }
        let pointers_live = st.queues_init && !st.cmd_window;
        if idx == MBOX_REQUEST && pointers_live {
            if val >= st.rqst_len {
                self.note(format!("request-in {val} past a ring of {} ignored", st.rqst_len));
                return;
            }
            st.rqst_in = val;
            st.rqst_blocked = false;
            self.kick(st);
            return;
        }
        if idx == MBOX_RESPONSE && pointers_live {
            if val >= st.resp_len {
                self.note(format!("response-out {val} past a ring of {} ignored", st.resp_len));
                return;
            }
            st.resp_out = val;
            self.kick(st);
            return;
        }
        st.mbox_in[idx] = val;
        // Linux and the IRIX 6.5.30 driver write a command's inputs without
        // draining a result posted just before; reading their own write back
        // would lose that result.
        if st.semaphore & SEMAPHORE_LOCK == 0 {
            st.mbox_out[idx] = val;
        }
    }

    fn mailbox_read(st: &State, idx: usize) -> u16 {
        // While a result holds the semaphore, mailboxes 4 and 5 are its data.
        if st.queues_init && st.semaphore & SEMAPHORE_LOCK == 0 {
            if idx == MBOX_REQUEST {
                return st.rqst_out;
            }
            if idx == MBOX_RESPONSE {
                return st.resp_in;
            }
        }
        st.mbox_out[idx]
    }

    fn reg_value(&self, st: &mut State, off: u32) -> u16 {
        let reset_visible = std::mem::take(&mut st.csr_reset_visible);
        if let Some(idx) = State::mbox_index(off) {
            return Self::mailbox_read(st, idx);
        }
        match off {
            reg::CSR => st.csr | if reset_visible { CSR_SOFT_RESET } else { 0 },
            reg::ICR => st.icr,
            reg::ISR => {
                let mut isr = if st.risc_int { ISR_RISC } else { 0 };
                if st.irq_level() {
                    isr |= ISR_PENDING;
                }
                isr
            }
            reg::SEMAPHORE => st.semaphore,
            reg::NVRAM => st.nvram_read(),
            reg::HCCR => {
                (if st.host_int { HCCR_HOST_INT } else { 0 })
                    | (if st.risc_reset || reset_visible { HCCR_RISC_RESET } else { 0 })
                    | (if st.risc_paused { HCCR_RISC_PAUSED } else { 0 })
            }
            reg::GPIO_DATA => st.gpio_data,
            reg::GPIO_ENABLE => st.gpio_enable,
            // Flash, DMA engine, banked and diagnostic registers.
            _ => {
                self.note(format!("read of unimplemented register {off:#04x}"));
                0
            }
        }
    }

    fn reg_write(&self, st: &mut MutexGuard<'_, State>, off: u32, val: u16) {
        if let Some(idx) = State::mbox_index(off) {
            self.mailbox_write(st, idx, val);
            return;
        }
        match off {
            reg::CSR => {
                if val & !(CSR_SOFT_RESET | CSR_BANK_MASK) != 0 {
                    self.note(format!("CSR flash bits {:#06x} not modelled", val & !(CSR_SOFT_RESET | CSR_BANK_MASK)));
                }
                if val & CSR_SOFT_RESET != 0 {
                    st.soft_reset();
                }
                st.csr = val & CSR_BANK_MASK;
            }
            reg::ICR => {
                if val & !ICR_MASK != 0 {
                    self.note(format!("interrupt control bits {:#06x} not modelled", val & !ICR_MASK));
                }
                st.icr = val & ICR_MASK;
            }
            reg::ISR => self.note(format!("write {val:#06x} to the read-only interrupt status register")),
            reg::SEMAPHORE => {
                if val != 0 {
                    self.note(format!("semaphore write {val:#06x} ignored; only a write of 0 releases it"));
                } else {
                    st.semaphore = 0;
                    self.post_waiting(st);
                    self.kick(st);
                }
            }
            reg::NVRAM => {
                let mut notes = Vec::new();
                st.nvram_write(&self.nvram_image, val, &mut notes);
                self.note_all(notes);
            }
            reg::HCCR => {
                // qlogicfc resets and releases before it reads anything.
                st.csr_reset_visible = false;
                self.host_command(st, val);
            }
            reg::GPIO_DATA => st.gpio_data = val,
            reg::GPIO_ENABLE => st.gpio_enable = val,
            _ => self.note(format!("write {val:#06x} to unimplemented register {off:#04x}")),
        }
        // A write that let a result post may have freed the slot.
        self.run_pending(st);
    }

    /// A register access. Only sixteen-bit accesses are decoded; a 32-bit one
    /// is answered as a 16-bit one at its address.
    pub fn read(&self, off: u32, width: usize) -> u32 {
        if !self.decoded("read", off, width) {
            return 0;
        }
        let val = {
            let mut st = self.st.lock();
            self.reg_value(&mut st, off)
        };
        u32::from(val)
    }

    pub fn write(&self, off: u32, val: u32, width: usize) {
        if !self.decoded("write", off, width) {
            return;
        }
        let reset = {
            let mut st = self.st.lock();
            let gen = st.ctl.gen;
            self.reg_write(&mut st, off, val as u16);
            let reset = st.ctl.gen != gen;
            if reset {
                self.publish_gen(&st);
                self.kick(&mut st);
            }
            // The write that changed the state drives the line, in the same hold.
            self.drive_irq(&mut st);
            self.assert_driven(&st);
            reset
        };
        if reset {
            self.retire_stores();
        }
    }

    fn decoded(&self, what: &str, off: u32, width: usize) -> bool {
        // Both notes are capped by a counter, so once a guest has used them up
        // a register access takes no lock and the notes list keeps room for
        // the ones that follow.
        if width == 4 {
            self.notes.push_first(&self.wide_accesses, 16, || format!("4-byte {what} at {off:#04x} answered as 16 bits"));
        }
        let ok = matches!(width, 2 | 4) && off & 1 == 0;
        if !ok {
            self.notes.push_first(&self.odd_accesses, 16, || format!("{width}-byte {what} at {off:#04x} is not decoded"));
        }
        ok
    }

    /// A machine reset: the chip as at power-on, its interrupt dropped. The
    /// disks stay attached.
    pub fn reset(&self) {
        {
            let mut st = self.st.lock();
            st.power_on();
            self.publish_gen(&st);
            self.kick(&mut st);
            self.drive_irq(&mut st);
            self.assert_driven(&st);
        }
        self.retire_stores();
    }
}

/// Both BARs reach the same registers.
impl PciFunction for Isp2200 {
    fn state(&self) -> &PciFunctionState {
        &self.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        self.read(off, width)
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.write(off, val, width)
    }

    /// Drive the pin the bus wired to the adapter's state.
    fn attached(&self) {
        // The controller's input may not be what the last write left.
        self.resync_irq();
    }

    fn reset(&self) {
        self.core.reset();
    }
}

impl Isp2200 {
    /// Name the adapter's long-lived locks in the lock registry, which holds
    /// only a weak reference to them.
    pub fn register_locks(&self) {
        let locks: [(&str, fn(&Core) -> bool); 3] =
            [("st", |c| c.st.is_locked()), ("fence", |c| c.fence.is_locked()), ("targets", |c| c.bus.is_locked())];
        for (what, probe) in locks {
            let core = Arc::downgrade(&self.core);
            crate::locks::register_lock(format!("isp2200-{}::{what}", self.core.index), move || {
                core.upgrade().is_some_and(|c| probe(&c))
            });
        }
    }
}

/// The monitor's `scsi` and `cow` commands over the loop's disks, as the
/// other SCSI controllers have them, plus `scsi notes`.
impl crate::traits::Device for Isp2200 {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {
        self.stop_ring()
    }
    fn start(&self) {
        self.start_ring()
    }
    fn is_running(&self) -> bool {
        self.worker.lock().is_some()
    }
    fn get_clock(&self) -> u64 {
        0
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        let usage = crate::scsi_monitor::SCSI_USAGE;
        if self.index == 0 {
            vec![
                ("scsi".to_string(), format!("SCSI (QLogic ISP2200 loop): {usage} | scsi notes")),
                ("scsi0".to_string(), "Alias for `scsi` (the loop of adapter 0)".to_string()),
                ("cow".to_string(), "COW overlays on the loop: cow status | cow commit [id] | cow reset [id]".to_string()),
            ]
        } else {
            vec![
                (format!("scsi{}", self.index), format!("SCSI loop of adapter {}: same subcommands as `scsi`", self.index)),
                (format!("cow{}", self.index), format!("COW overlays on the loop of adapter {}: same subcommands as `cow`", self.index)),
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
                let _ = writeln!(w, "QLogic ISP2200, adapter {}:", self.index);
                for note in self.notes() {
                    let _ = writeln!(w, "  note: {note}");
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

impl crate::scsi_monitor::ScsiTargets for Isp2200 {
    fn scsi_bus(&self) -> &ScsiBus {
        &self.bus
    }
}

#[cfg(test)]
mod tests;
