//! QLogic ISP2200 Fibre Channel host adapter, emulated above its RISC.
//!
//! The driver talks to the adapter's firmware through 24 sixteen-bit
//! mailboxes and two rings of 64-byte entries in host memory. As with the
//! SCSI parts ([`crate::dev::isp1040`]), the firmware a driver downloads is
//! retained, so a checksum can be verified and read back, and never run: the
//! mailbox commands and rings the firmware would provide are done here.
//!
//! Behind the host interface sits a private arbitrated loop kept inside the
//! device. Firmware hides Fibre Channel from the host, which sees loop IDs,
//! names and SCSI commands, so frames, logins and loop initialisation are
//! not modelled. Each target on the adapter's [`ScsiBus`] is the one port of
//! the loop ID equal to its SCSI id, named from that id.
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
//!   thread. An abort or reset matches the command running and nothing still
//!   queued in the request ring, where the real firmware has many in flight.
//! * The loop is read off the bus as the machine built it. A target attached
//!   or removed later is not announced (no LIP, no port database event) and
//!   is not seen before the next command; hotplug is not modelled.
//! * A target has one LUN, as the SCSI layer below it does. A command to
//!   another LUN up to 255 goes to the disk as for a LUN it lacks: INQUIRY
//!   reports no device (peripheral qualifier 3) and anything else is CHECK
//!   CONDITION, ILLEGAL REQUEST. REPORT LUNS is not answered, for LUN 0
//!   either.
//! * Every reset state, the command window that tells a ring pointer write
//!   from a mailbox input, the posting rules and the per-command error
//!   codes are model choices the drivers leave open.
//!
//! Ported from hw/scsi/qlogic-isp2200.c in jamesbraid/qemu-sgi-origami.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::{Mutex, MutexGuard};

use crate::dev::isp1040::{
    self, IOCB_ABORTED, IOCB_COMPLETE, IOCB_DATA_UNDERRUN, IOCB_RESET, SCSI_STATUS_SENSE_VALID, ENTRY_SIZE, HOST_CMD_SHIFT, MAX_REQUESTS, PCI_VENDOR_QLOGIC, RAM_WORDS, RISC_FLAG_HOST_INT, RISC_FLAG_PAUSED,
    RISC_FLAG_RESET, SEMAPHORE_LOCK, SIG1, SIG2, SIG3, ENTRY_TYPE_MARKER, ENTRY_TYPE_STATUS,
};
use crate::dev::qlogic_ring::{
    self, IrqLevel, Notes, RingCtl, RingState, RingThread, CONT_OFF_SEGS32, CONT_OFF_SEGS64, CONT_SEGS32, CONT_SEGS64, SEG32_SIZE, SEG64_SIZE,
};
use crate::pci::{Bar, PciConfig, PciFunction, PciFunctionState, PCI_INTA};
use crate::scsi::{get_cdb_length, scsi_cmd, ScsiBus, ScsiDataLength, ScsiIdentities, ScsiRequest, CDROM_IDENTITY, DISK_IDENTITY};
use crate::snapshot::{get_field, hex_u16, hex_u32, load_u16_slice, toml_u16, toml_u32, u16_slice_to_toml, u8_slice_to_toml};

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
/// LIP reset's m1 high byte for every loop ID.
const LIP_RESET_ALL: u8 = 0xff;

/// A model constant, deliberately not any vendor firmware release.
const FW_VERSION: [u16; 3] = [1, 0, 0];
/// Expanded LUN only: no fabric, no 2K logins.
const FW_ATTRIBUTES: u16 = 0x0002;

/// Get Firmware State: the only three the loop ever reports.
pub const FW_STATE_CONFIG_WAIT: u16 = 0;
pub const FW_STATE_WAIT_AL_PA: u16 = 1;
pub const FW_STATE_READY: u16 = 3;

/// How long a loop takes to come up once something starts it. A model
/// constant, long enough for a test to see the initialising state; how long
/// a real loop initialisation takes is unknown.
pub const BRINGUP: Duration = Duration::from_millis(100);

/// Loop IDs 0..0x7d are loop ports; 0x7e and up are fabric and special.
pub const LOOP_ID_MAX: u8 = 0x7d;
/// The LUNs an FC target addresses. The SCSI layer below the adapter reads
/// three bits of LUN from the CDB and has one LUN, so a LUN from 1 to 255 is
/// one the disk lacks and a command to a LUN past this is refused here.
const MAX_LUN: u16 = 255;
/// The LUN the CDB names for a LUN the disk lacks and the CDB cannot hold.
const ABSENT_CDB_LUN: u8 = 7;

/// Get Retry Count m3. No public source gives what the firmware returns and
/// the control block carries nothing for it. Linux halves it into its
/// R_A_TOV in tenths of a second, and 200 then gives the 10-second minimum
/// Linux already sets, leaving its state as it was.
const RETRY_TIMEOUT: u16 = 200;
/// Get Port Queue Parameters m2 and m3. Approximation: the firmware's values
/// are unknown; these are the ones IRIX substitutes itself when the command
/// fails, so the guest ends up with the parameters it already assumes.
const PORT_QUEUE_DEPTH: u16 = 256;
const PORT_QUEUE_THROTTLE: u16 = 255;
/// Get Link Status writes eight 32-bit counters, the length IRIX copies.
const LINK_STATUS_SIZE: usize = 32;
/// Get Adapter ID m1 while the loop is down: Linux's "loop in transition".
const ADAPTER_ID_NOT_READY: u16 = 7;
/// Bit 1 of the stored Set Firmware Options m1 enables LIP-reset events.
const FW_OPT_LIP_RESET_EVENTS: u16 = 0x0002;

const PDB_SIZE: usize = 128;
const PDB_OFF_MASTER: usize = 2;
const PDB_OFF_SLAVE: usize = 3;
const PDB_OFF_HARD_ADDR: usize = 6;
const PDB_OFF_PORT_ID: usize = 8;
const PDB_OFF_NODE_NAME: usize = 12;
const PDB_OFF_PORT_NAME: usize = 20;
const PDB_OFF_PRLI_SVC3: usize = 78;
const PDB_OFF_LOOP_ID: usize = 80;
const PDB_LOGGED_IN: u8 = 6;
const PDB_TARGET_ROLE: u8 = 0x10;
const POSITION_MAP_SIZE: usize = 128;
const ID_LIST_ENTRY: usize = 4;
const NAME_LIST_ENTRY: usize = 10;
/// Port Name List options OpenBSD sets: include initiators, all loop IDs.
const NAME_LIST_OPTIONS: u16 = 0x000a;

const SCSI_STATUS_RESID_UNDER: u16 = 0x0800;
const CHECK_CONDITION: u8 = 0x02;
const COMPLETION_DMA_ERROR: u16 = 0x0002;
const DIRECTION_DATA_IN: u16 = 0x0020;
const DIRECTION_DATA_OUT: u16 = 0x0040;

/// The standard table: loop ID n is AL_PA n, 0xef at 0 down to 0x01 at
/// 0x7d, then 0x00 for the fabric port. A private loop has domain and area
/// 0, so a port ID is its AL_PA alone.
const ALPA: [u8; LOOP_ID_MAX as usize + 2] = [
    0xef, 0xe8, 0xe4, 0xe2, 0xe1, 0xe0, 0xdc, 0xda, 0xd9, 0xd6, 0xd5, 0xd4, 0xd3, 0xd2, 0xd1, 0xce, 0xcd, 0xcc, 0xcb, 0xca, 0xc9,
    0xc7, 0xc6, 0xc5, 0xc3, 0xbc, 0xba, 0xb9, 0xb6, 0xb5, 0xb4, 0xb3, 0xb2, 0xb1, 0xae, 0xad, 0xac, 0xab, 0xaa, 0xa9, 0xa7, 0xa6,
    0xa5, 0xa3, 0x9f, 0x9e, 0x9d, 0x9b, 0x98, 0x97, 0x90, 0x8f, 0x88, 0x84, 0x82, 0x81, 0x80, 0x7c, 0x7a, 0x79, 0x76, 0x75, 0x74,
    0x73, 0x72, 0x71, 0x6e, 0x6d, 0x6c, 0x6b, 0x6a, 0x69, 0x67, 0x66, 0x65, 0x63, 0x5c, 0x5a, 0x59, 0x56, 0x55, 0x54, 0x53, 0x52,
    0x51, 0x4e, 0x4d, 0x4c, 0x4b, 0x4a, 0x49, 0x47, 0x46, 0x45, 0x43, 0x3c, 0x3a, 0x39, 0x36, 0x35, 0x34, 0x33, 0x32, 0x31, 0x2e,
    0x2d, 0x2c, 0x2b, 0x2a, 0x29, 0x27, 0x26, 0x25, 0x23, 0x1f, 0x1e, 0x1d, 0x1b, 0x18, 0x17, 0x10, 0x0f, 0x08, 0x04, 0x02, 0x01,
    0x00,
];

/// The node name of the port at `loop_id`. Whoever fits a disk to the loop
/// names it in the real fabric; the model names a port from its loop ID, so a
/// guest sees stable names that look synthetic.
pub const fn node_wwn(loop_id: u8) -> u64 {
    0x2000_0000_0000_0000 | loop_id as u64
}

/// The port name of the port at `loop_id`; see [`node_wwn`].
pub const fn port_wwn(loop_id: u8) -> u64 {
    0x2100_0000_0000_0000 | loop_id as u64
}

const ICB_SIZE: usize = 96;
const ICB_VERSION: u8 = 1;
const ICB_OFF_OPTIONS: usize = 2;
const ICB_OFF_RETRY_COUNT: usize = 10;
const ICB_OFF_RETRY_DELAY: usize = 11;
const ICB_OFF_HARD_ADDR: usize = 20;
const ICB_OFF_XOPTIONS: usize = 64;
const ICB_OFF_RQST_LEN: usize = 36;
const ICB_OFF_RESP_LEN: usize = 38;
const ICB_OFF_RQST_BASE: usize = 40;
const ICB_OFF_RESP_BASE: usize = 48;
const ICB_OPT_HARD_ADDR: u16 = 0x0001;
const ICB_OPT_FAST_POST: u16 = 0x0008;
const ICB_OPT_PORT_DB_EVENTS: u16 = 0x0100;
/// Extended options: bits 3:0 the reduced-completion mode, 6:4 topology.
const ICB_XOPT_TIMER_MASK: u16 = 0x000f;
const CONN_LOOP: u16 = 0;
const CONN_LOOP_PREFERRED: u16 = 2;
const CONN_P2P_PREFERRED: u16 = 3;

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
    /// Initialise-firmware accepted a loop connection mode; until then, or
    /// after a reset, nothing brings the loop up.
    loop_active: bool,
    /// What Get Firmware State reports.
    loop_state: u16,
    has_loop_id: bool,
    /// The adapter's own, chosen at initialisation.
    loop_id: u8,
    /// The running bring-up came from a LIP reset.
    bringup_lip_reset: bool,
    /// When the adapter's clock says the loop is up.
    bringup_due: Option<i64>,
    ports: Ports,
    /// The command running on the loop; at most one, the ring is walked in
    /// order.
    active: Option<Active>,
    /// Device resets for the ring thread to pass on, in order: a loop ID or
    /// every device.
    pending_resets: VecDeque<Option<u8>>,
    commands: u64,
    /// A request-ring read was refused: the entry stays unconsumed until the
    /// next producer write.
    rqst_blocked: bool,
    ctl: Ctl,
}

/// Which loop IDs have a target on the bus, as the adapter last saw it. The
/// mailbox commands answer from this: they run with the state locked, and
/// the bus is held across a disk command.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct Ports([u64; 2]);

impl Ports {
    fn has(&self, id: usize) -> bool {
        id <= usize::from(LOOP_ID_MAX) && self.0[id / 64] >> (id % 64) & 1 != 0
    }

    fn scan(targets: &[Option<Box<dyn crate::scsi::ScsiTarget>>]) -> Ports {
        let mut ports = Ports::default();
        for (id, target) in targets.iter().enumerate().take(usize::from(LOOP_ID_MAX) + 1) {
            if target.is_some() {
                ports.0[id / 64] |= 1 << (id % 64);
            }
        }
        ports
    }

    fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..=LOOP_ID_MAX).filter(|&id| self.has(usize::from(id)))
    }
}

/// The command on the loop's one device, running with the state unlocked.
#[derive(Clone, Copy, Debug)]
struct Active {
    loop_id: u8,
    lun: u16,
    handle: u32,
}

/// How a command ended, before it is a status entry.
struct Done {
    scsi_status: u16,
    completion: u16,
    resid: u32,
    sense: Vec<u8>,
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
            loop_active: false,
            loop_state: FW_STATE_CONFIG_WAIT,
            has_loop_id: false,
            loop_id: 0,
            bringup_lip_reset: false,
            bringup_due: None,
            ports: Ports::default(),
            active: None,
            pending_resets: VecDeque::new(),
            commands: 0,
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
        self.pending_resets.clear();

        self.mbox_in = [0; MAILBOX_COUNT];
        self.mbox_out = [0; MAILBOX_COUNT];
        self.mbox_out[0] = MBOX_BUSY;

        self.fw_running = false;
        self.fw_options = [0; 3];
        self.risc_ram.iter_mut().for_each(|w| *w = 0);

        self.clear_queues();
        self.drop_queued();
        // The loop goes with the firmware, and a command in flight posts
        // nothing: the rings it would post to are gone.
        self.loop_reset();
        self.active = None;
        self.ctl.gen += 1;
    }

    // ---- the private loop --------------------------------------------

    fn loop_reset(&mut self) {
        self.bringup_due = None;
        self.loop_active = false;
        self.loop_state = FW_STATE_CONFIG_WAIT;
        self.has_loop_id = false;
        self.loop_id = 0;
        self.bringup_lip_reset = false;
    }

    fn loop_ready(&self) -> bool {
        self.loop_state == FW_STATE_READY
    }

    /// A port other than the adapter, at a loop ID the loop has.
    fn target(&self, id: usize) -> bool {
        self.ports.has(id) && !(self.has_loop_id && id == usize::from(self.loop_id))
    }

    fn is_bringup_event(code: u16) -> bool {
        matches!(code, event::LIP_OCCURRED | event::LOOP_UP | event::LIP_RESET | event::PORT_DB_CHANGED)
    }

    /// Takes back the queued, not yet posted, events of earlier bring-ups.
    fn drop_bringup_events(&mut self) {
        self.events.retain(|mb| !Self::is_bringup_event(mb[0]));
    }

    /// Starts the loop coming up: state 1 until the clock reaches `due`. A
    /// new trigger restarts the wait and takes back whatever an earlier
    /// bring-up queued and has not posted, so the queue holds at most one
    /// bring-up's events and each bring-up posts a whole set. An event
    /// already posted stays until the host releases it.
    fn bringup(&mut self, lip_reset: bool, due: i64) {
        if !self.loop_active {
            return;
        }
        self.loop_state = FW_STATE_WAIT_AL_PA;
        self.bringup_lip_reset = lip_reset;
        self.drop_bringup_events();
        self.bringup_due = Some(due);
    }

    /// The loop is up. The events are a model choice conditioned on the two
    /// option bits the drivers set: LIP (or LIP reset), loop up, and port
    /// database changed for every port. Their payloads are zero but for that
    /// last m1, which Linux reads as a global update.
    fn bringup_done(&mut self) {
        self.bringup_due = None;
        let options = le16(&self.icb, ICB_OFF_OPTIONS);
        // With every loop ID taken the adapter never gets one, nor a loop.
        if !self.has_loop_id {
            return;
        }
        self.loop_state = FW_STATE_READY;
        if !self.bringup_lip_reset {
            self.push_event(event::LIP_OCCURRED, 0);
        } else if self.fw_options[0] & FW_OPT_LIP_RESET_EVENTS != 0 {
            self.push_event(event::LIP_RESET, 0);
        }
        self.push_event(event::LOOP_UP, 0);
        if options & ICB_OPT_PORT_DB_EVENTS != 0 {
            self.push_event(event::PORT_DB_CHANGED, event::PORT_DB_GLOBAL);
        }
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

    /// Retire the command running on the loop with the completion the
    /// recovery owes. True if one was running.
    fn cancel_active(&mut self, cancel: &AtomicU64, completion: u16) -> bool {
        let Some(active) = self.active.take() else { return false };
        cancel.fetch_add(1, Ordering::AcqRel);
        self.complete(active.handle, 0, completion, 0, &[]);
        true
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
    /// Bumped when a recovery cancels the running command, so a store the
    /// ring thread made for it is not made after the recovery returns.
    cancel: AtomicU64,
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
            cancel: AtomicU64::new(0),
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
        if st.active.take().is_some() {
            self.cancel.fetch_add(1, Ordering::AcqRel);
        }
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
        self.refresh_ports(st);
        self.loop_init(st);
        mbox::COMPLETE
    }

    /// Initialise-firmware starts the loop from its control block. The
    /// adapter takes the requested hard address when it asks for one, the
    /// address is a loop ID and no port holds it, and otherwise the lowest
    /// loop ID no port holds. Real loops settle this in loop initialisation,
    /// and which address a port then wins is unknown. The choice holds until
    /// the next initialisation or reset.
    ///
    /// Extended options bits 6:4 choose the connection. Loop only (0), loop
    /// preferred (2) and point-to-point preferred (3) bring up the loop; with
    /// no point-to-point partner mode 3 falls back to loop. Point-to-point
    /// only (1) and the undefined values leave state 0 and post nothing.
    fn loop_init(&self, st: &mut State) {
        let options = le16(&st.icb, ICB_OFF_OPTIONS);
        let hard = le16(&st.icb, ICB_OFF_HARD_ADDR);
        let xopts = le16(&st.icb, ICB_OFF_XOPTIONS);
        st.loop_reset();
        st.drop_bringup_events();
        if xopts & ICB_XOPT_TIMER_MASK != 0 {
            self.note(format!(
                "reduced-completion mode {} is not modelled; every command posts a full status entry",
                xopts & ICB_XOPT_TIMER_MASK
            ));
        }
        let conn = (xopts >> 4) & 7;
        if !matches!(conn, CONN_LOOP | CONN_LOOP_PREFERRED | CONN_P2P_PREFERRED) {
            self.note(format!("connection mode {conn} leaves the loop down: only a loop is modelled"));
            return;
        }
        if options & ICB_OPT_HARD_ADDR != 0 && hard <= u16::from(LOOP_ID_MAX) && !st.ports.has(usize::from(hard)) {
            st.has_loop_id = true;
            st.loop_id = hard as u8;
        } else if let Some(id) = (0..=LOOP_ID_MAX).find(|&id| !st.ports.has(usize::from(id))) {
            st.has_loop_id = true;
            st.loop_id = id;
        }
        st.loop_active = true;
        self.start_bringup(st, false);
    }

    fn start_bringup(&self, st: &mut State, lip_reset: bool) {
        let due = self.now_ns(st) + BRINGUP.as_nanos() as i64;
        st.bringup(lip_reset, due);
        self.kick(st);
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
        self.refresh_ports(st);
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
            _ if Self::fc_owned(cmd) => self.fc_mailbox(st, cmd, input, out),
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
    ///
    /// The loop comes up on the ring thread, which a command blocked on a
    /// device keeps from looking at the clock; the test's thread does it
    /// here, so a test can hold a command and still see the loop come up.
    #[cfg(test)]
    pub(crate) fn set_test_clock(&self, ns: i64) {
        let mut st = self.st.lock();
        self.fake_ns.store(ns, Ordering::Relaxed);
        if self.bringup_left(&st).is_some_and(|left| left <= 0) {
            st.bringup_done();
            self.post_waiting(&mut st);
        }
        self.kick(&mut st);
        self.drive_irq(&mut st);
    }

    /// How long until the loop is up, if it is coming up.
    fn bringup_left(&self, st: &State) -> Option<i64> {
        st.bringup_due.map(|due| due - self.now_ns(st))
    }

    /// Block until the ring thread has run everything it was given and is
    /// waiting for more.
    pub fn wait_idle(&self) {
        let mut st = self.st.lock();
        while (st.ctl.hold == 0 && !st.ctl.paused && (st.ctl.work || self.bringup_left(&st).is_some_and(|left| left <= 0)))
            || st.ctl.walking
        {
            self.wake.wait(&mut st);
        }
    }

    fn run_ring(&self) {
        let mut st = self.st.lock();
        loop {
            loop {
                if st.ctl.shutdown {
                    return;
                }
                let runnable = st.ctl.hold == 0 && !st.ctl.paused;
                let left = self.bringup_left(&st);
                if runnable && (st.ctl.work || left.is_some_and(|left| left <= 0)) {
                    break;
                }
                match left {
                    // A test clock only moves when the test moves it.
                    Some(left) if runnable && self.fake_ns.load(Ordering::Relaxed) == i64::MIN => {
                        self.wake.wait_for(&mut st, Duration::from_nanos(left as u64));
                    }
                    _ => {
                        self.wake.wait(&mut st);
                    }
                }
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
            if let Some(target) = st.pending_resets.pop_front() {
                MutexGuard::unlocked(&mut st, || self.reset_devices(target));
                continue;
            }
            if self.bringup_left(&st).is_some_and(|left| left <= 0) {
                st.bringup_done();
                self.drive_irq(&mut st);
                continue;
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
        if st.held.len() + usize::from(st.active.is_some()) >= usize::from(MAX_REQUESTS) {
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

    // ---- the loop commands -------------------------------------------

    fn fc_owned(cmd: u16) -> bool {
        matches!(
            cmd,
            0x0015 | 0x0017 | 0x0020 | 0x0022 | 0x0029 | 0x0032 | 0x0039 | 0x0040 | 0x0041 | 0x0047 | 0x0052
                | 0x0062 | 0x0063 | 0x0064 | 0x006a | 0x006b | 0x006c | 0x006e | 0x006f | 0x0071 | 0x0072
                | 0x0074 | 0x0075 | 0x007c | 0x007e
        )
    }

    /// Look at the bus without waiting for a disk command: the ports as of
    /// the last time the bus was free.
    fn refresh_ports(&self, st: &mut State) {
        if let Some(targets) = self.bus.try_lock() {
            st.ports = Ports::scan(&targets);
        }
    }

    fn dma_out(&self, st: &mut MutexGuard<'_, State>, addr: u64, data: &[u8]) -> bool {
        MutexGuard::unlocked(st, || self.dma_write(addr, data))
    }

    /// The ID list: four bytes per port, AL_PA, area 0, domain 0, loop ID.
    fn get_id_list(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes, out: &mut Mailboxes) -> u16 {
        // Its own split: m1 bits 31:16, m2 15:0, m3 63:48, m6 47:32.
        let addr = (u64::from(input[3]) << 48) | (u64::from(input[6]) << 32) | (u64::from(input[1]) << 16) | u64::from(input[2]);
        if !st.loop_ready() {
            return mbox::COMMAND_ERROR;
        }
        let list: Vec<u8> = (0..=LOOP_ID_MAX)
            .filter(|&id| st.target(usize::from(id)))
            .flat_map(|id| [ALPA[usize::from(id)], 0, 0, id])
            .collect();
        if !list.is_empty() && !self.dma_out(st, addr, &list) {
            return mbox::HOST_INTERFACE_ERROR;
        }
        out[1] = (list.len() / ID_LIST_ENTRY) as u16;
        mbox::COMPLETE
    }

    /// OpenBSD's port name list: ten bytes per port, the port name then the
    /// loop ID as a little-endian word. OpenBSD decodes this name as a
    /// little-endian integer but the port database's as big-endian, and
    /// compares the two; the name goes here with its wire-order bytes
    /// reversed so both decodes agree. The real firmware's order is unknown.
    fn port_name_list(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes, out: &mut Mailboxes) -> u16 {
        if !st.loop_ready() {
            return mbox::COMMAND_ERROR;
        }
        if input[1] & !NAME_LIST_OPTIONS != 0 {
            return mbox::PARAM_ERROR;
        }
        let list: Vec<u8> = (0..=LOOP_ID_MAX)
            .filter(|&id| st.target(usize::from(id)))
            .flat_map(|id| {
                let mut entry = [0u8; NAME_LIST_ENTRY];
                entry[..8].copy_from_slice(&port_wwn(id).to_le_bytes());
                entry[8..].copy_from_slice(&u16::from(id).to_le_bytes());
                entry
            })
            .collect();
        if !list.is_empty() && !self.dma_out(st, mbox_addr(input), &list) {
            return mbox::HOST_INTERFACE_ERROR;
        }
        out[1] = list.len() as u16;
        mbox::COMPLETE
    }

    /// One port's database entry. Every present port is logged in with the
    /// FCP target role and class 3 only; the bytes not set here are zero.
    fn get_port_db(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes) -> u16 {
        let id = (input[1] >> 8) as u8;
        if !st.loop_ready() || !st.target(usize::from(id)) {
            return mbox::COMMAND_ERROR;
        }
        let mut pd = [0u8; PDB_SIZE];
        pd[PDB_OFF_MASTER] = PDB_LOGGED_IN;
        pd[PDB_OFF_SLAVE] = PDB_LOGGED_IN;
        pd[PDB_OFF_HARD_ADDR] = ALPA[usize::from(id)];
        // Domain, a reserved byte, AL_PA, area.
        pd[PDB_OFF_PORT_ID + 2] = ALPA[usize::from(id)];
        pd[PDB_OFF_NODE_NAME..][..8].copy_from_slice(&node_wwn(id).to_be_bytes());
        pd[PDB_OFF_PORT_NAME..][..8].copy_from_slice(&port_wwn(id).to_be_bytes());
        // Both drivers find the target role in the low byte.
        pd[PDB_OFF_PRLI_SVC3] = PDB_TARGET_ROLE;
        pd[PDB_OFF_LOOP_ID..][..2].copy_from_slice(&u16::from(id).to_le_bytes());
        if !self.dma_out(st, mbox_addr(input), &pd) {
            return mbox::HOST_INTERFACE_ERROR;
        }
        mbox::COMPLETE
    }

    /// One port's name, by loop ID: option 0 the port name, 1 the node name,
    /// the names the port database also reports. Each of m2, m3, m6 and m7
    /// carries two wire-order bytes with the earlier in its low half. NetBSD
    /// decodes that order on every part before the 24xx and qlogicfc on this
    /// part; mainline Linux decodes the other way, but on this part it asks
    /// only for the fabric port, which is refused. The real firmware's order
    /// is unknown. Refusals follow the port database's, so the fabric check
    /// at 0x7e finds no fabric and a scan of every loop ID skips the adapter.
    fn get_port_name(st: &State, input: &Mailboxes, out: &mut Mailboxes) -> u16 {
        let option = input[1] & 0xff;
        if !st.loop_ready() {
            return mbox::COMMAND_ERROR;
        }
        if option > 1 {
            return mbox::PARAM_ERROR;
        }
        let id = (input[1] >> 8) as u8;
        if !st.target(usize::from(id)) {
            return mbox::COMMAND_ERROR;
        }
        let name = (if option == 1 { node_wwn(id) } else { port_wwn(id) }).to_be_bytes();
        for (slot, pair) in [2, 3, 6, 7].into_iter().zip(name.chunks_exact(2)) {
            out[slot] = u16::from_le_bytes([pair[0], pair[1]]);
        }
        mbox::COMPLETE
    }

    /// The position map: the count with the adapter, the adapter's AL_PA,
    /// then the ports' in loop ID order. A real map follows loop position;
    /// the order here is a model choice, and Linux reads only the count.
    fn position_map(&self, st: &mut MutexGuard<'_, State>, input: &Mailboxes) -> u16 {
        if !st.loop_ready() {
            return mbox::COMMAND_ERROR;
        }
        let mut map = [0u8; POSITION_MAP_SIZE];
        map[1] = ALPA[usize::from(st.loop_id)];
        let mut n = 1;
        for id in (0..=LOOP_ID_MAX).filter(|&id| st.target(usize::from(id))) {
            map[1 + n] = ALPA[usize::from(id)];
            n += 1;
        }
        map[0] = n as u8;
        if !self.dma_out(st, mbox_addr(input), &map) {
            return mbox::HOST_INTERFACE_ERROR;
        }
        mbox::COMPLETE
    }

    /// The recovery commands: abort one command, abort a target, reset one
    /// LUN, LIP reset. The completions they post are the public definitions
    /// and the reset semantics follow the SCSI parts'; what a real ISP2200
    /// does, including any frames it sends the target, is unknown.
    ///
    /// Approximation: commands run one at a time in ring order, so a
    /// recovery finds at most the one running on the loop; entries still
    /// waiting in the request ring are not yet commands and are not matched.
    /// A reset reaches the whole device at a loop ID: the SCSI layer below
    /// has one LUN per target.
    fn recovery(&self, st: &mut State, cmd: u16, input: &Mailboxes) -> u16 {
        let loop_id = (input[1] >> 8) as u8;
        let running = |st: &State, lun: Option<u16>| {
            st.active.is_some_and(|a| a.loop_id == loop_id && lun.is_none_or(|lun| a.lun == lun))
        };
        match cmd {
            mbox::ABORT_COMMAND => {
                let handle = (u32::from(input[3]) << 16) | u32::from(input[2]);
                if running(st, Some(input[6])) && st.active.is_some_and(|a| a.handle == handle) {
                    st.cancel_active(&self.cancel, IOCB_ABORTED);
                    mbox::COMPLETE
                } else {
                    mbox::COMMAND_ERROR
                }
            }
            mbox::ABORT_TARGET | mbox::LUN_RESET => {
                let lun = (cmd == mbox::LUN_RESET).then_some(input[2]);
                if running(st, lun) {
                    st.cancel_active(&self.cancel, IOCB_RESET);
                }
                st.pending_resets.push_back(Some(loop_id));
                self.kick(st);
                mbox::COMPLETE
            }
            mbox::LIP_RESET => {
                if !st.loop_active {
                    return mbox::COMMAND_ERROR;
                }
                let all = loop_id == LIP_RESET_ALL;
                if all || running(st, None) {
                    st.cancel_active(&self.cancel, IOCB_RESET);
                }
                st.pending_resets.push_back((!all).then_some(loop_id));
                self.start_bringup(st, true);
                mbox::COMPLETE
            }
            _ => unreachable!("not a recovery command"),
        }
    }

    /// The loop commands. Each refusal's code is a model choice: the public
    /// definitions name the codes but not which refusal returns which.
    fn fc_mailbox(&self, st: &mut MutexGuard<'_, State>, cmd: u16, input: &Mailboxes, out: &mut Mailboxes) -> u16 {
        match cmd {
            mbox::GET_ADAPTER_ID => {
                if !st.loop_ready() {
                    out[1] = ADAPTER_ID_NOT_READY;
                    return mbox::COMMAND_ERROR;
                }
                // m2 AL_PA and area 0, m3 domain 0, m6 topology 0: private loop.
                out[1] = u16::from(st.loop_id);
                out[2] = u16::from(ALPA[usize::from(st.loop_id)]);
                mbox::COMPLETE
            }
            mbox::GET_ID_LIST => self.get_id_list(st, input, out),
            mbox::PORT_NAME_LIST => self.port_name_list(st, input, out),
            mbox::GET_PORT_DB => self.get_port_db(st, input),
            mbox::GET_PORT_NAME => Self::get_port_name(st, input, out),
            mbox::POSITION_MAP => self.position_map(st, input),
            mbox::ABORT_COMMAND | mbox::ABORT_TARGET | mbox::LUN_RESET | mbox::LIP_RESET => self.recovery(st, cmd, input),
            mbox::INITIATE_LIP | mbox::FULL_LOGIN_LIP => {
                if !st.loop_active {
                    return mbox::COMMAND_ERROR;
                }
                self.start_bringup(st, false);
                mbox::COMPLETE
            }
            // The control block's retry count and delay, the order Linux and
            // NetBSD read them in; that the firmware echoes the bytes rather
            // than keeping values of its own is inferred. IRIX 6.5.30 scans
            // its loop at attach only when this completes. The pair comes
            // from the control block, not the loop, so it answers in any loop
            // state.
            mbox::GET_RETRY_COUNT => {
                if !st.queues_init {
                    return mbox::COMMAND_ERROR;
                }
                out[1] = u16::from(st.icb[ICB_OFF_RETRY_COUNT]);
                out[2] = u16::from(st.icb[ICB_OFF_RETRY_DELAY]);
                out[3] = RETRY_TIMEOUT;
                mbox::COMPLETE
            }
            // The five commands below come from IRIX alone and no public
            // source gives the firmware's replies. Each answer is an
            // approximation: it keeps the guest on the path it takes when
            // the command succeeds and invents no state.
            //
            // Not answered for the adapter's own loop ID: the driver asks
            // once per discovered port, which the adapter is not. A guest
            // that asks anyway gets the refusal the port database gives.
            mbox::GET_PORT_QUEUE => {
                if !st.loop_ready() || !st.target(usize::from(input[1] >> 8)) {
                    return mbox::COMMAND_ERROR;
                }
                self.approximate(cmd);
                out[2] = PORT_QUEUE_DEPTH;
                out[3] = PORT_QUEUE_THROTTLE;
                mbox::COMPLETE
            }
            // Accepted and forgotten: the retry pair stays the control block's.
            mbox::SET_RETRY_COUNT | mbox::RESET_LINK_STATUS => {
                self.approximate(cmd);
                mbox::COMPLETE
            }
            // IRIX sends only m0 and m1 and ignores the reply. The firmware's
            // status is unknown; 0x4000 is a model choice. Depth and throttle
            // are not kept, and no loop ID is checked: the guest's registers
            // come back as written.
            mbox::SET_PORT_QUEUE => {
                self.approximate(cmd);
                out[1..8].copy_from_slice(&input[1..8]);
                mbox::COMPLETE
            }
            // Every link counter reads zero. The driver's link-statistics
            // thread asks for the adapter's own loop ID as well as the
            // ports', so the adapter answers for itself, with the same zero
            // counters.
            mbox::GET_LINK_STATUS => {
                let id = (input[1] >> 8) as usize;
                if !st.loop_ready() || !(st.target(id) || (st.has_loop_id && id == usize::from(st.loop_id))) {
                    return mbox::COMMAND_ERROR;
                }
                self.approximate(cmd);
                if !self.dma_out(st, mbox_addr(input), &[0u8; LINK_STATUS_SIZE]) {
                    return mbox::HOST_INTERFACE_ERROR;
                }
                mbox::COMPLETE
            }
            _ => {
                self.note(format!("loop command {cmd:#06x} not modelled"));
                mbox::INVALID
            }
        }
    }

    /// Mark a reply that is an approximation, for the debug log.
    fn approximate(&self, cmd: u16) {
        crate::dlog!(crate::devlog::LogModule::Scsi, "isp2200: approximate reply to mailbox command {cmd:#06x}");
    }

    // ---- commands on the loop ----------------------------------------

    /// Device resets for the ring thread to pass on, with the bus unlocked
    /// from `st`: each device puts its mode parameters back and owes the
    /// host a unit attention on its next command.
    fn reset_devices(&self, target: Option<u8>) {
        self.bus.for_each(|id, device| {
            if target.is_none_or(|target| usize::from(target) == id) {
                device.reset_mode_parameters();
                device.bus_reset();
            }
        });
    }

    fn block_size(&self, id: u8) -> usize {
        self.bus
            .lock()
            .get(usize::from(id))
            .and_then(|target| target.as_ref())
            .and_then(|target| target.block_size())
            .map(|block| block as usize)
            .filter(|&block| block != 0)
            .unwrap_or(isp1040::BLOCK_SIZE)
    }

    /// Run a CDB against the target at `id`; `data` is what a write sends it.
    /// None when a reset or a recovery cancelled the command while it waited
    /// for the target.
    fn scsi(&self, gen: u64, cancel: u64, id: u8, cdb: &[u8], data: Option<Vec<u8>>) -> Option<(Vec<u8>, u8)> {
        let mut targets = self.bus.lock();
        if self.gen.load(Ordering::Acquire) != gen || self.cancel.load(Ordering::Acquire) != cancel {
            return None;
        }
        let Some(device) = targets.get_mut(usize::from(id)).and_then(|target| target.as_mut()) else {
            return Some((Vec::new(), CHECK_CONDITION));
        };
        let request = ScsiRequest { cdb: cdb.to_vec(), data_len: ScsiDataLength::Unlimited, data_in: data };
        Some(match device.request(&request) {
            Ok(response) => (response.data, response.status),
            Err(e) => {
                crate::dlog!(crate::devlog::LogModule::Scsi, "isp2200: target {id}: {e}");
                self.note(format!("target {id}: {e}"));
                (Vec::new(), CHECK_CONDITION)
            }
        })
    }

    /// Fixed-format sense, ILLEGAL REQUEST, LOGICAL UNIT NOT SUPPORTED.
    fn lun_not_supported() -> Vec<u8> {
        let mut sense = vec![0u8; 18];
        sense[0] = 0x70;
        sense[2] = 0x05;
        sense[7] = 10;
        sense[12] = 0x25;
        sense
    }

    /// Run a command against its device and build its completion, with the
    /// state unlocked: this is the part that waits on the disk.
    ///
    /// The segments name the host's buffer and the command names how much of
    /// it the target fills, so only that much is offered and the whole named
    /// length still counts toward the residual. A transfer the host bridge
    /// refuses completes the command with a DMA error and no SCSI status.
    fn run_command(&self, cmd: &Command, gen: u64, cancel: u64) -> Option<Done> {
        let id = cmd.loop_id;
        let mut cdb = cmd.cdb;
        // A LUN above 7 does not fit the CDB's three bits; the disk has no
        // LUN 7 either, so it answers as for any LUN it lacks.
        cdb[1] = (cdb[1] & 0x1f) | (cmd.lun.min(u16::from(ABSENT_CDB_LUN)) as u8) << 5;
        let cdb = &cdb[..get_cdb_length(cdb[0]).clamp(1, CMD_CDB_LEN)];
        let (mut xfer, mut to_device) = qlogic_ring::cdb_transfer(cdb, self.block_size(id));
        // An opcode with no rule here takes the direction and length the
        // driver put in the entry.
        if xfer == 0 && cmd.control & (DIRECTION_DATA_IN | DIRECTION_DATA_OUT) != 0 {
            xfer = cmd.byte_count as usize;
            to_device = cmd.control & DIRECTION_DATA_OUT != 0;
        }
        let mut segs = Vec::new();
        let mut described = 0u64;
        for &(addr, len) in &cmd.segs {
            qlogic_ring::add_segment(&mut segs, &mut described, xfer as u64, addr, len);
        }
        let dma_error = Done { scsi_status: 0, completion: COMPLETION_DMA_ERROR, resid: 0, sense: Vec::new() };

        let (status, moved);
        if to_device {
            let mut buf = Vec::new();
            for &(addr, len) in &segs {
                let mut chunk = vec![0u8; len as usize];
                if !self.dma_read(addr, &mut chunk) {
                    return Some(dma_error);
                }
                buf.extend_from_slice(&chunk);
            }
            moved = buf.len() as u64;
            status = self.scsi(gen, cancel, id, cdb, Some(buf))?.1;
        } else {
            let (data, s) = self.scsi(gen, cancel, id, cdb, None)?;
            status = s;
            let take = (data.len() as u64).min(described).min(xfer as u64) as usize;
            let mut off = 0usize;
            for &(addr, len) in &segs {
                if off >= take {
                    break;
                }
                let n = (len as usize).min(take - off);
                let stored =
                    self.fenced(gen, || self.cancel.load(Ordering::Acquire) == cancel && self.dma_write(addr, &data[off..off + n]))?;
                if !stored {
                    return Some(dma_error);
                }
                off += n;
            }
            moved = off as u64;
        }

        // Auto-sense runs only after a CHECK CONDITION.
        let mut sense = Vec::new();
        if status == CHECK_CONDITION {
            sense = self.scsi(gen, cancel, id, &[scsi_cmd::REQUEST_SENSE, 0, 0, 0, STATUS_SENSE_MAX as u8, 0], None)?.0;
            sense.truncate(STATUS_SENSE_MAX);
        }
        Some(Self::completion(status, moved, described.max(xfer as u64), sense))
    }

    /// Completion precedence, following what the two drivers decode rather
    /// than any firmware evidence: a SCSI status other than GOOD completes
    /// 0x0000 carrying the status and any sense, with residual-under when
    /// the transfer was short; GOOD and short is underrun 0x0015; GOOD and
    /// whole is 0x0000 with no flags.
    fn completion(status: u8, moved: u64, total: u64, sense: Vec<u8>) -> Done {
        let unused = total - moved.min(total);
        let mut scsi_status = 0;
        let mut completion = IOCB_COMPLETE;
        if status != 0 {
            scsi_status = u16::from(status);
            if !sense.is_empty() {
                scsi_status |= SCSI_STATUS_SENSE_VALID;
            }
        } else if unused != 0 {
            completion = IOCB_DATA_UNDERRUN;
        }
        if unused != 0 {
            scsi_status |= SCSI_STATUS_RESID_UNDER;
        }
        Done { scsi_status, completion, resid: unused.min(u64::from(u32::MAX)) as u32, sense }
    }

    /// A command from the host interface onto the loop. With no loop -- the
    /// connection mode left it down, or the adapter never took a loop ID --
    /// every command completes "port unavailable" with no SCSI status, as a
    /// disconnected adapter's do. So does one to a loop ID with no port, the
    /// adapter's own, or one past 0x7d; which code real firmware returns for
    /// those is unknown. Commands to present ports run while the loop comes
    /// up. A LUN past 255 gets CHECK CONDITION, ILLEGAL REQUEST without reaching
    /// a device.
    fn submit(&self, st: &mut MutexGuard<'_, State>, cmd: Command) {
        let seglen: u64 = cmd.segs.iter().map(|&(_, len)| u64::from(len)).sum();
        let gen = st.ctl.gen;
        let id = cmd.loop_id;
        let routed = st.loop_active && st.has_loop_id && id <= LOOP_ID_MAX && id != st.loop_id;
        let present = routed && MutexGuard::unlocked(st, || self.bus.has_target(id));
        if st.ctl.gen != gen {
            return;
        }
        if !present {
            st.complete(cmd.handle, 0, COMPLETION_PORT_UNAVAILABLE, 0, &[]);
            return;
        }
        if cmd.lun > MAX_LUN {
            let status = u16::from(CHECK_CONDITION) | SCSI_STATUS_SENSE_VALID | if seglen != 0 { SCSI_STATUS_RESID_UNDER } else { 0 };
            st.complete(cmd.handle, status, IOCB_COMPLETE, seglen.min(u64::from(u32::MAX)) as u32, &Self::lun_not_supported());
            return;
        }
        st.commands += 1;
        st.active = Some(Active { loop_id: id, lun: cmd.lun, handle: cmd.handle });
        let cancel = self.cancel.load(Ordering::Acquire);
        let done = MutexGuard::unlocked(st, || self.run_command(&cmd, gen, cancel));
        // A reset or a recovery retired the command while it ran: it owes
        // nothing more.
        if st.ctl.gen != gen || st.active.take().is_none() {
            return;
        }
        if let Some(done) = done {
            st.complete(cmd.handle, done.scsi_status, done.completion, done.resid, &done.sense);
        }
    }

    // ---- state ---------------------------------------------------------

    /// Commands run, for the whole session.
    pub fn total_commands(&self) -> u64 {
        self.st.lock().commands
    }

    fn save_locked(&self, st: &State, now: i64) -> toml::Value {
        let mut t = toml::map::Map::new();
        for (k, v) in [
            ("csr", st.csr),
            ("icr", st.icr),
            ("semaphore", st.semaphore),
            ("nvram", st.nvram),
            ("nvram_shift", st.nvram_shift),
            ("gpio_data", st.gpio_data),
            ("gpio_enable", st.gpio_enable),
            ("rqst_len", st.rqst_len),
            ("resp_len", st.resp_len),
            ("rqst_in", st.rqst_in),
            ("rqst_out", st.rqst_out),
            ("resp_in", st.resp_in),
            ("resp_out", st.resp_out),
            ("loop_state", st.loop_state),
        ] {
            t.insert(k.into(), hex_u16(v));
        }
        for (k, v) in [
            ("csr_reset_visible", st.csr_reset_visible),
            ("nvram_dout", st.nvram_dout),
            ("risc_int", st.risc_int),
            ("host_int", st.host_int),
            ("risc_reset", st.risc_reset),
            ("risc_paused", st.risc_paused),
            ("cmd_window", st.cmd_window),
            ("cmd_pending", st.cmd_pending),
            ("fw_running", st.fw_running),
            ("queues_init", st.queues_init),
            ("completion_held", st.completion_held),
            ("rqst_blocked", st.rqst_blocked),
            ("loop_active", st.loop_active),
            ("has_loop_id", st.has_loop_id),
            ("bringup_lip_reset", st.bringup_lip_reset),
        ] {
            t.insert(k.into(), toml::Value::Boolean(v));
        }
        t.insert(
            "nvram_mode".into(),
            toml::Value::Integer(match st.nvram_mode {
                NvramMode::Command => 0,
                NvramMode::Data => 1,
                NvramMode::Ignore => 2,
            }),
        );
        t.insert("nvram_bits".into(), toml::Value::Integer(i64::from(st.nvram_bits)));
        t.insert("loop_id".into(), toml::Value::Integer(i64::from(st.loop_id)));
        t.insert("commands".into(), toml::Value::Integer(st.commands as i64));
        t.insert("rqst_base".into(), hex_u32(st.rqst_base as u32));
        t.insert("rqst_base_hi".into(), hex_u32((st.rqst_base >> 32) as u32));
        t.insert("resp_base".into(), hex_u32(st.resp_base as u32));
        t.insert("resp_base_hi".into(), hex_u32((st.resp_base >> 32) as u32));
        t.insert("mbox_in".into(), u16_slice_to_toml(&st.mbox_in));
        t.insert("mbox_out".into(), u16_slice_to_toml(&st.mbox_out));
        t.insert("completion".into(), u16_slice_to_toml(&st.completion));
        t.insert("fw_options".into(), u16_slice_to_toml(&st.fw_options));
        t.insert("icb".into(), u8_slice_to_toml(&st.icb));
        // The downloaded microcode is large and usually all zero between a
        // reset and the driver's download.
        if st.risc_ram.iter().any(|&w| w != 0) {
            t.insert("risc_ram".into(), u16_slice_to_toml(&st.risc_ram));
        }
        t.insert("events".into(), toml::Value::Array(st.events.iter().map(|mb| u16_slice_to_toml(mb)).collect()));
        t.insert("held".into(), toml::Value::Array(st.held.iter().map(|entry| u8_slice_to_toml(entry)).collect()));
        // Absolute deadlines mean nothing on another run's clock.
        if let Some(due) = st.bringup_due {
            t.insert("bringup_left_ns".into(), toml::Value::Integer((due - now).max(0)));
        }
        toml::Value::Table(t)
    }

    /// Whatever the file says, indices fold into their rings and counts stay
    /// within what this build could have queued.
    fn load_locked(&self, st: &mut State, v: &toml::Value, now: i64) {
        let ctl = std::mem::take(&mut st.ctl);
        *st = State::new();
        st.ctl = ctl;
        st.ctl.gen += 1;
        let u16f = |k: &str| get_field(v, k).and_then(toml_u16).unwrap_or(0);
        let flag = |k: &str| matches!(get_field(v, k), Some(toml::Value::Boolean(true)));
        let int = |k: &str| match get_field(v, k) {
            Some(toml::Value::Integer(n)) => Some(*n),
            _ => None,
        };
        st.csr = u16f("csr") & CSR_BANK_MASK;
        st.icr = u16f("icr") & ICR_MASK;
        st.semaphore = u16f("semaphore") & SEMAPHORE_LOCK;
        st.nvram = u16f("nvram") & NVRAM_WRITE_MASK;
        st.nvram_shift = u16f("nvram_shift");
        st.nvram_bits = int("nvram_bits").unwrap_or(0).clamp(0, i64::from(NVRAM_CMD_BITS.max(NVRAM_DATA_BITS))) as u8;
        st.nvram_mode = match int("nvram_mode") {
            Some(1) => NvramMode::Data,
            Some(2) => NvramMode::Ignore,
            _ => NvramMode::Command,
        };
        st.nvram_dout = flag("nvram_dout");
        st.gpio_data = u16f("gpio_data");
        st.gpio_enable = u16f("gpio_enable");
        st.csr_reset_visible = flag("csr_reset_visible");
        st.risc_int = flag("risc_int");
        st.host_int = flag("host_int");
        st.risc_reset = flag("risc_reset");
        st.risc_paused = flag("risc_paused");
        st.cmd_window = flag("cmd_window");
        st.cmd_pending = flag("cmd_pending");
        st.fw_running = flag("fw_running");
        st.queues_init = flag("queues_init");
        st.completion_held = flag("completion_held");
        st.rqst_blocked = flag("rqst_blocked");
        st.loop_active = flag("loop_active");
        st.has_loop_id = flag("has_loop_id");
        st.bringup_lip_reset = flag("bringup_lip_reset");
        st.loop_state = match u16f("loop_state") {
            state @ (FW_STATE_CONFIG_WAIT | FW_STATE_WAIT_AL_PA | FW_STATE_READY) => state,
            _ => FW_STATE_CONFIG_WAIT,
        };
        st.loop_id = int("loop_id").unwrap_or(0).clamp(0, i64::from(LOOP_ID_MAX)) as u8;
        st.commands = int("commands").unwrap_or(0).max(0) as u64;
        let u32f = |k: &str| get_field(v, k).and_then(toml_u32).unwrap_or(0);
        st.rqst_base = u64::from(u32f("rqst_base")) | (u64::from(u32f("rqst_base_hi")) << 32);
        st.resp_base = u64::from(u32f("resp_base")) | (u64::from(u32f("resp_base_hi")) << 32);
        st.rqst_len = u16f("rqst_len");
        st.resp_len = u16f("resp_len");
        if st.queues_init && (st.rqst_len == 0 || st.resp_len == 0) {
            st.queues_init = false;
        }
        if st.queues_init {
            st.rqst_in = u16f("rqst_in") % st.rqst_len;
            st.rqst_out = u16f("rqst_out") % st.rqst_len;
            st.resp_in = u16f("resp_in") % st.resp_len;
            st.resp_out = u16f("resp_out") % st.resp_len;
        } else {
            st.clear_queues();
        }
        for (k, dst) in [("mbox_in", &mut st.mbox_in), ("mbox_out", &mut st.mbox_out), ("completion", &mut st.completion)] {
            if let Some(m) = get_field(v, k) {
                load_u16_slice(m, dst);
            }
        }
        if let Some(m) = get_field(v, "fw_options") {
            load_u16_slice(m, &mut st.fw_options);
        }
        if let Some(toml::Value::Array(icb)) = get_field(v, "icb") {
            for (dst, src) in st.icb.iter_mut().zip(icb) {
                *dst = toml_u16(src).unwrap_or(0) as u8;
            }
        }
        if let Some(m) = get_field(v, "risc_ram") {
            load_u16_slice(m, &mut st.risc_ram);
        }
        if let Some(toml::Value::Array(events)) = get_field(v, "events") {
            for e in events.iter().take(EVENTS_MAX) {
                let mut mb = [0; MAILBOX_COUNT];
                load_u16_slice(e, &mut mb);
                st.events.push_back(mb);
            }
        }
        if let Some(toml::Value::Array(held)) = get_field(v, "held") {
            for e in held.iter().take(usize::from(MAX_REQUESTS)) {
                let mut entry = [0u8; ENTRY_SIZE];
                if let toml::Value::Array(bytes) = e {
                    for (dst, src) in entry.iter_mut().zip(bytes) {
                        *dst = toml_u16(src).unwrap_or(0) as u8;
                    }
                }
                st.held.push_back(entry);
            }
        }
        // The wait goes on for what was left of it, from this clock.
        st.bringup_due = int("bringup_left_ns").filter(|_| st.loop_active).map(|left| now.saturating_add(left.max(0)));
        self.ports_unknown(st);
    }

    /// Ports are read off the bus again by the next command that needs them.
    fn ports_unknown(&self, st: &mut State) {
        self.refresh_ports(st);
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
                // The loop's devices see the reset too.
                if val & CSR_SOFT_RESET != 0 {
                    st.soft_reset();
                    st.pending_resets.push_back(None);
                    self.kick(st);
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
            let (gen, cancel) = (st.ctl.gen, self.cancel.load(Ordering::Acquire));
            self.reg_write(&mut st, off, val as u16);
            let reset = st.ctl.gen != gen || self.cancel.load(Ordering::Acquire) != cancel;
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
                let _ = writeln!(w, "QLogic ISP2200, adapter {}: {} commands", self.index, self.total_commands());
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

impl crate::traits::Saveable for Isp2200 {
    fn save_state(&self) -> toml::Value {
        self.quiesced(|st| {
            let now = self.now_ns(st);
            self.save_locked(st, now)
        })
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        self.quiesced(|st| {
            let now = self.now_ns(st);
            self.load_locked(st, v, now);
            self.publish_gen(st);
            st.ctl.work = true;
        });
        // The controller took its own state from the snapshot.
        self.resync_irq();
        Ok(())
    }
}

pub mod card;

#[cfg(test)]
pub(crate) mod tests;
