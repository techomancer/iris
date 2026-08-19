use std::fs::OpenOptions;
use std::cell::Cell;
use std::collections::VecDeque;
use std::thread;
use std::sync::Arc;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, Device, FifoDevice, DmaClient, DmaStatus, Resettable, Saveable};
use crate::devlog::{LogModule, devlog};
use crate::snapshot::{get_field, toml_u8, toml_bool, u8_slice_to_toml, load_u8_slice, hex_u8};
use crate::scsi::{self, ScsiDevice, scsi_cmd, ScsiRequest, ScsiDataLength};
use std::io::Write;

// Ring-buffer trace of WD33C93 register accesses with consecutive-dedup, so
// long polling loops compress to one "(x N)" line and the ring covers a much
// wider window of activity.
// Only compiled and active under --features developer.
#[cfg(feature = "developer")]
pub static WDT_SEQ: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "developer")]
pub struct WdtRing {
    entries: VecDeque<(u64, String, u64)>,   // (first_seq, message, repeat_count)
}
#[cfg(feature = "developer")]
pub static WDT_RING: parking_lot::Mutex<WdtRing> = parking_lot::Mutex::new(WdtRing { entries: VecDeque::new() });
#[cfg(feature = "developer")]
const WDT_RING_CAP: usize = 4000;

#[cfg(feature = "developer")]
pub fn wdt_log(args: std::fmt::Arguments<'_>) {
    let n = WDT_SEQ.fetch_add(1, Ordering::Relaxed);
    let msg = format!("{}", args);
    let mut ring = WDT_RING.lock();
    if let Some(last) = ring.entries.back_mut() {
        if last.1 == msg {
            last.2 += 1;
            return;
        }
    }
    if ring.entries.len() >= WDT_RING_CAP { ring.entries.pop_front(); }
    ring.entries.push_back((n, msg, 1));
}

#[cfg(feature = "developer")]
pub fn wdt_dump_tail() {
    let ring = WDT_RING.lock();
    for (seq, msg, cnt) in ring.entries.iter() {
        if *cnt == 1 {
            eprintln!("WDT {:8} {}", seq, msg);
        } else {
            eprintln!("WDT {:8} {}   (x{})", seq, msg, cnt);
        }
    }
    eprintln!("WDT_DUMP_END (total_seen={})", WDT_SEQ.load(Ordering::Relaxed));
}

/// Log a WDT entry — no-op when developer feature is disabled.
macro_rules! wdt {
    ($($arg:tt)*) => {
        #[cfg(feature = "developer")]
        wdt_log(format_args!($($arg)*));
    };
}

// Indirect Register Addresses (accessed via AR)
pub mod regs {
    pub const OWN_ID: u8 = 0x00;
    pub const CONTROL: u8 = 0x01;
    pub const TIMEOUT_PERIOD: u8 = 0x02;
    pub const CDB_1: u8 = 0x03;
    pub const CDB_2: u8 = 0x04;
    pub const CDB_3: u8 = 0x05;
    pub const CDB_4: u8 = 0x06;
    pub const CDB_5: u8 = 0x07;
    pub const CDB_6: u8 = 0x08;
    pub const CDB_7: u8 = 0x09;
    pub const CDB_8: u8 = 0x0A;
    pub const CDB_9: u8 = 0x0B;
    pub const CDB_10: u8 = 0x0C;
    pub const CDB_11: u8 = 0x0D;
    pub const CDB_12: u8 = 0x0E;
    pub const TARGET_LUN: u8 = 0x0F;
    pub const COMMAND_PHASE: u8 = 0x10;
    pub const SYNC_TRANSFER: u8 = 0x11;
    pub const TRANSFER_COUNT_MSB: u8 = 0x12;
    pub const TRANSFER_COUNT_2ND: u8 = 0x13;
    pub const TRANSFER_COUNT_LSB: u8 = 0x14;
    pub const DESTINATION_ID: u8 = 0x15;
    pub const SOURCE_ID: u8 = 0x16;
    pub const SCSI_STATUS: u8 = 0x17;
    pub const COMMAND: u8 = 0x18;
    pub const DATA: u8 = 0x19;
    pub const QUEUE_TAG: u8 = 0x1A;
    // 0x1B-0x1E are reserved
    pub const AUX_STATUS_DIRECT: u8 = 0x1F;
}

// Auxiliary Status Register (ASR) bits (read from A0=0)
pub mod asr {
    pub const DBR: u8 = 0x01;       // Data Buffer Ready
    pub const PE: u8 = 0x02;        // Parity Error
    pub const CIP: u8 = 0x10;       // Command in Progress
    pub const BSY: u8 = 0x20;       // Busy
    pub const LCI: u8 = 0x40;       // Last Command Ignored
    pub const INT: u8 = 0x80;       // Interrupt Pending
}

// SCSI Status Register (SSR) bits (indirect reg 0x17)
pub mod ssr {
    // Bits 7-3: SCSI Status Byte
    pub const INTERRUPT_STATE_MASK: u8 = 0x07; // Bits 2-0
}

// SCSI Status Register (SSR) values (from IRIX wd93.h)
pub mod scsi_status {
    // Group 0x00: reset
    pub const RESET: u8 = 0x00;                    // chip reset
    pub const RESET_EAF: u8 = 0x01;                // reset with advanced features (93A)

    // Group 0x10: selection / reselection complete
    pub const RESELECT_SUCCESS: u8 = 0x10;         // (T) reselection complete
    pub const SELECT_SUCCESS: u8 = 0x11;           // (I) selection complete — ST_SELECT

    // Group 0x13: command/transfer completion
    pub const COMMAND_SUCCESS: u8 = 0x13;          // (T) send/receive cmd done, no ATN
    pub const COMMAND_ATN_SUCCESS: u8 = 0x14;      // (T) send/receive cmd done, ATN asserted
    pub const TRANSLATE_SUCCESS: u8 = 0x15;        // translate address complete
    pub const SELECT_TRANSFER_SUCCESS: u8 = 0x16;  // (I) select-and-transfer complete — ST_SATOK
    pub const TRANSFER_DATA_OUT: u8 = 0x18;        // (I) xfer done, target requesting DATA OUT — ST_TR_DATAOUT
    pub const TRANSFER_DATA_IN: u8 = 0x19;         // (I) xfer done, target sending DATA IN — ST_TR_DATAIN
    pub const TRANSFER_STATUS_IN: u8 = 0x1B;       // (I) xfer done, target sending STATUS — ST_TR_STATIN
    pub const TRANSFER_MSG_IN: u8 = 0x1F;          // (I) xfer done, target sending MSG IN — ST_TR_MSGIN

    // Group 0x20: service required / pause
    pub const TRANSFER_PAUSE: u8 = 0x20;           // (I) transfer paused with ACK asserted — ST_TRANPAUSE
    pub const SAVE_DATA_POINTERS: u8 = 0x21;       // (I) SDP message received — ST_SAVEDP
    pub const SELECTION_ABORTED: u8 = 0x22;        // selection/reselection aborted
    pub const RECEIVE_SEND_ABORTED: u8 = 0x23;     // receive/send aborted (93A: was 0x13)
    pub const RECEIVE_SEND_ABORTED_ATN: u8 = 0x24; // receive/send aborted, ATN asserted
    pub const ABORT_DURING_SELECTION: u8 = 0x25;   // abort cmd issued during selection
    pub const RESELECTED_AFTER_DISC: u8 = 0x27;    // (I) lost selection, AM mode — ST_A_RESELECT (93A)
    pub const TRANSFER_ABORTED: u8 = 0x28;         // (I) transfer aborted due to phase mismatch — ST_MIS

    // Group 0x40: errors
    pub const INVALID_COMMAND: u8 = 0x40;          // command not valid in current state
    pub const UNEXPECTED_DISCONNECT: u8 = 0x41;    // (I) target disconnected unexpectedly — ST_UNEXPDISC
    pub const SELECTION_TIMEOUT: u8 = 0x42;        // (I) no response to selection — ST_TIMEOUT
    pub const PARITY_ERROR: u8 = 0x43;             // (I) SCSI parity error — ST_PARITY
    pub const PARITY_ERROR_ATN: u8 = 0x44;         // (I) parity error, ATN asserted — ST_PARITY_ATN
    pub const LOGICAL_ADDRESS_TOO_LARGE: u8 = 0x45;
    pub const RESELECTION_MISMATCH: u8 = 0x46;     // (I) reselected but ID didn't match
    pub const INCORRECT_STATUS_BYTE: u8 = 0x47;    // (I) bad status byte received — ST_INCORR_DATA
    pub const UNEXPECTED_RECV_DATA: u8 = 0x48;     // (I) unexpected DATA IN phase — ST_UNEX_RDATA
    pub const UNEXPECTED_SEND_DATA: u8 = 0x49;     // (I) unexpected DATA OUT phase — ST_UNEX_SDATA
    pub const UNEXPECTED_CMD_PHASE: u8 = 0x4A;     // (I) unexpected COMMAND phase — ST_UNEX_CMDPH
    pub const UNEXPECTED_SEND_STATUS: u8 = 0x4B;   // (I) unexpected STATUS phase — ST_UNEX_SSTATUS
    pub const UNEXPECTED_REQ_MSG_OUT: u8 = 0x4E;   // (I) unexpected MSG OUT phase — ST_UNEX_RMESGOUT
    pub const UNEXPECTED_SEND_MSG_IN: u8 = 0x4F;   // (I) unexpected MSG IN phase — ST_UNEX_SMESGIN

    // Group 0x80: target-mode / bus-free statuses
    pub const RESELECTED: u8 = 0x80;               // (I) reselected while idle, no AM — ST_RESELECT
    pub const RESELECTED_EAF: u8 = 0x81;           // (I) reselected while idle, AM mode — ST_93A_RESEL (93A)
    pub const SELECTED: u8 = 0x82;                 // (T) selected without ATN
    pub const SELECTED_ATN: u8 = 0x83;             // (T) selected with ATN
    pub const ATN: u8 = 0x84;                      // (T) ATN asserted
    pub const DISCONNECT: u8 = 0x85;               // (I) bus free after disconnect — ST_DISCONNECT
    pub const UNKNOWN_GROUP: u8 = 0x87;            // CDB1 specifies unknown group (SBIC_CSR_UNK_GROUP)

    // Group 0x88: bus-service REQ — low nibble encodes SCSI phase (SBIC_CSR_MIS_2)
    // REQ = 0x88 base; specific phase variants below
    pub const REQ_DATA_OUT:     u8 = 0x88;         // DATA OUT phase requested
    pub const REQ_DATA_IN:      u8 = 0x89;         // DATA IN phase requested
    pub const REQ_CMD_PHASE:    u8 = 0x8A;         // COMMAND phase requested — ST_NEEDCMD
    pub const REQ_STATUS:       u8 = 0x8B;         // STATUS phase requested
    pub const REQ_SEND_MSG_OUT: u8 = 0x8E;         // MSG OUT phase requested — ST_REQ_SMESGOUT
    pub const REQ_MSG_IN:       u8 = 0x8F;         // MSG IN phase requested — ST_REQ_SMESGIN
}

// Command Register (CMD) values (indirect reg 0x18)
pub mod cmd {
    pub const RESET: u8 = 0x00;
    pub const ABORT: u8 = 0x01;
    pub const ASSERT_ATN: u8 = 0x02;
    pub const NEGATE_ACK: u8 = 0x03;
    pub const DISCONNECT: u8 = 0x04;
    pub const RESELECT: u8 = 0x05;
    pub const SELECT_ATN: u8 = 0x06;
    pub const SELECT: u8 = 0x07;
    pub const SELECT_ATN_XFER: u8 = 0x08;
    pub const SELECT_XFER: u8 = 0x09;
    pub const RESELECT_RECEIVE: u8 = 0x0A;
    pub const RESELECT_SEND: u8 = 0x0B;
    pub const WAIT_SELECT_RECEIVE: u8 = 0x0C;
    pub const SEND_STATUS: u8 = 0x10;
    pub const SEND_DISCONNECT: u8 = 0x11;
    pub const SET_IDI: u8 = 0x12;
    pub const RECEIVE_COMMAND: u8 = 0x13;
    pub const RECEIVE_DATA: u8 = 0x14;
    pub const RECEIVE_MSG_OUT: u8 = 0x15;
    pub const RECEIVE_INFO: u8 = 0x16;
    pub const SEND_COMMAND: u8 = 0x17;
    pub const SEND_DATA: u8 = 0x18;
    pub const SEND_MSG_IN: u8 = 0x19;
    pub const SEND_INFO: u8 = 0x1A;
    pub const TRANSFER_INFO: u8 = 0x20;
}

// Command Phase Register (0x10) values
#[allow(dead_code)]
pub mod command_phase {
    pub const DISCONNECTED: u8 = 0x00;    // no device selected
    pub const SELECTED: u8 = 0x10;        // target selected
    pub const IDENTIFY_SENT: u8 = 0x20;   // identify message sent to target
    pub const COMMAND_START: u8 = 0x30;   // command phase started, 0 bytes sent
    // 0x31-0x3C: command phase, N bytes sent (N = low nibble + unit digit)
    pub const SAVE_DATA_POINTER: u8 = 0x41;  // SDP message received
    pub const DISCONNECT_MSG: u8 = 0x42;     // disconnect message received, bus not free
    pub const DISCONNECTED_OK: u8 = 0x43;    // target disconnected, bus free
    pub const RESELECTED: u8 = 0x44;         // reselected by original target
    pub const IDENTIFY_RECEIVED: u8 = 0x45;  // identify message received from target
    pub const TRANSFER_COUNT: u8 = 0x46;     // data transfer complete (TC = 0)
    pub const RECEIVE_STATUS: u8 = 0x47;     // target has begun status phase
    pub const STATUS_RECEIVED: u8 = 0x50;    // status byte received, stored in TARGET_LUN
    pub const COMPLETE_MSG: u8 = 0x60;       // command complete message received
}

struct Wd33c93aState {
    /// Which HPC3 SCSI controller this chip is (0 or 1) — Indigo2 fullhouse
    /// has two independent WD33C93A instances; embedded in every dlog! line
    /// (`"WD33C93A({}): ..."`) so their logs are distinguishable, matching
    /// Z85c30's Channel::name convention for its two channels.
    id: u8,
    // Indirectly accessed registers
    regs: [u8; 32],
    // Address Register (selects one of the 32 regs)
    ar: u8,
    // Auxiliary Status Register (read-only)
    asr: u8,
    // SCSI Devices (IDs 0-7)
    devices: [Option<ScsiDevice>; 8],
    fifo: VecDeque<u8>,
    // Data direction flag for computing DBR (true = data in from target to host)
    data_direction_in: bool,
    target_id: usize,
    // Shadow of the last LUN the host wrote to TARGET_LUN (reg 0x0F), updated
    // only on host writes (see the write() handler). regs[TARGET_LUN] itself
    // is dual-purpose on real WD33C93A hardware — it also holds the status
    // byte for host readback once a command concludes (STATUS_RECEIVED) — so
    // it can't be read back after the fact to recover the LUN the host set.
    target_lun: u8,
    pending_status: u8,
    pending_msg: u8,
    // Data staged from SCSI device response, waiting for driver to issue TRANSFER_INFO.
    // Used for the manual TRANSFER_INFO flow (non-auto_mode) where the driver sets up
    // DMA/PIO *after* seeing the TRANSFER_DATA_IN interrupt, so we can't push immediately.
    pending_data: Vec<u8>,
    advanced_mode: bool,
    pending_command: Option<u8>,
    // Mid-transfer pause state for 256KB chunk re-arm
    xfer_data: Vec<u8>,         // full data buffer for current SCSI command
    xfer_offset: usize,         // bytes already transferred
    xfer_direction_in: bool,    // true=send to host (READ cmd), false=receive from host (WRITE cmd)
    // IRQ status FIFO (mirrors MAME wd33c9x irq_fifo).
    // Statuses are pushed here; update_irq() pops the front into SCSI_STATUS and
    // sets ASR.INT. On SCSI_STATUS read (INT ack), update_irq() is called again —
    // if another entry is waiting it immediately re-asserts INT without toggling the
    // interrupt line low. This allows SELECT_ATN to push 0x11 then 0x8E as two
    // separate interrupts, matching real hardware behaviour.
    irq_fifo: VecDeque<(u8, Option<u8>, bool)>, // (status, phase, deferred) — deferred=true: worker drops lock before delivery
    callback: Option<Arc<dyn ScsiCallback>>,
    // Debug tracking: last values returned for register reads
    last_read_asr: Option<u8>,
    last_read_reg: Option<(u8, u8)>, // (register, value)
    last_cmd: u8,                    // last command written to COMMAND register
    /// When true, STATUS_IN interrupts are deferred: CIP cleared early, then
    /// INT asserted after a 1000-cycle spin so wd33c93_loop can exit first.
    /// Required for OpenBSD/NetBSD. Toggleable via `scsi defer on/off`.
    deferred_int: Arc<AtomicBool>,
}

pub trait ScsiCallback: Send + Sync {
    /// Set the SCSI interrupt line. Calling with false also clears any paired
    /// PDMA completion bit (implementors handle this internally).
    fn set_interrupt(&self, level: bool);
}

// Safety: `Cell<T>` is never `Sync` regardless of `T` (interior mutability
// without synchronization) — needed here because `cpu_cycles` (a `Cell`,
// deliberately: see that field's own doc comment) is set exactly once,
// during single-threaded `Machine::new` setup, strictly before `start()`
// spawns the only thread that ever reads it, so no *concurrent* access to
// the `Cell` is actually possible despite `Wd33c93a` being shared behind
// `Arc` across threads. Every other field is already genuinely `Sync` on
// its own.
unsafe impl Sync for Wd33c93a {}

pub struct Wd33c93a {
    state: Arc<Mutex<Wd33c93aState>>,
    cond: Arc<Condvar>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    running: Arc<AtomicBool>,
    dma: Option<Arc<dyn DmaClient>>,
    /// Activity heartbeat shared with the display thread.
    heartbeat: Arc<AtomicU64>,
    /// Pointer into the CPU's `MipsCore.hot.cycles` word (see
    /// `CyclesPtr`/`Hot::cycles`'s doc comments) — used to pace deferred
    /// interrupt delivery (the OpenBSD/NetBSD workaround below) without
    /// wall-clock sleeping. Set once via `set_cpu_cycles`, during
    /// single-threaded `Machine::new` setup — after the CPU exists (this
    /// device is constructed before it does, so this can't be a constructor
    /// parameter) but strictly before `start()` spawns the worker thread
    /// that actually reads it, so a plain `Cell` is enough.
    cpu_cycles: Cell<crate::mips_core::CyclesPtr>,
    /// Defer SCSI status interrupts: clear CIP, spin 1000 cycles, then assert INT.
    /// Required for OpenBSD/NetBSD so wd33c93_loop exits before INT fires.
    deferred_int: Arc<AtomicBool>,
}

impl Wd33c93a {
    pub fn new(dma: Option<Arc<dyn DmaClient>>, callback: Option<Arc<dyn ScsiCallback>>, heartbeat: Arc<AtomicU64>) -> Self {
        Self::new_with_config(dma, callback, heartbeat, true)
    }

    pub fn new_with_config(dma: Option<Arc<dyn DmaClient>>, callback: Option<Arc<dyn ScsiCallback>>, heartbeat: Arc<AtomicU64>, deferred_int: bool) -> Self {
        Self::new_with_id(dma, callback, heartbeat, deferred_int, 0)
    }

    /// Same as `new_with_config`, but tags every log line with `id` (the
    /// HPC3 SCSI controller number) — see `Wd33c93aState.id`'s doc comment.
    pub fn new_with_id(dma: Option<Arc<dyn DmaClient>>, callback: Option<Arc<dyn ScsiCallback>>, heartbeat: Arc<AtomicU64>, deferred_int: bool, id: u8) -> Self {
        let deferred_int_arc = Arc::new(AtomicBool::new(deferred_int));
        Self {
            state: Arc::new(Mutex::new(Wd33c93aState {
                id,
                regs: [0; 32],
                ar: 0,
                asr: 0, // Initially not busy, no interrupt
                devices: Default::default(),
                fifo: VecDeque::new(),
                data_direction_in: false,
                target_id: 0,
                target_lun: 0,
                pending_status: 0,
                pending_msg: 0,
                pending_data: Vec::new(),
                advanced_mode: false,
                pending_command: None,
                xfer_data: Vec::new(),
                xfer_offset: 0,
                xfer_direction_in: false,
                irq_fifo: VecDeque::new(),
                callback,
                last_read_asr: None,
                last_read_reg: None,
                last_cmd: 0,
                deferred_int: deferred_int_arc.clone(),
            })),
            cond: Arc::new(Condvar::new()),
            thread: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            dma,
            heartbeat,
            cpu_cycles: Cell::new(crate::mips_core::CyclesPtr::dangling()),
            deferred_int: deferred_int_arc,
        }
    }

    /// Wire up the CPU cycle counter — called from `Machine::new` once
    /// `MipsCpu` exists (this device is constructed before it does, so this
    /// can't be a constructor parameter; see `MipsCpu::cycles_ptr`). Must be
    /// called before `start()` — see `cpu_cycles`'s own doc comment.
    pub fn set_cpu_cycles(&self, ptr: crate::mips_core::CyclesPtr) {
        self.cpu_cycles.set(ptr);
    }

    /// Attach a SCSI device.
    /// For CD-ROMs, `discs` is the full ordered list of ISO paths; the first
    /// entry is mounted immediately.  For HDDs `discs` is ignored — only
    /// `path` is used.
    ///
    /// If `overlay_path_override` is `Some`, it specifies where the COW
    /// overlay file lives. This lets CI mode isolate its overlay from an
    /// interactive session sharing the same base image. Ignored when
    /// `overlay` is false.
    pub fn add_device(
        &self,
        id: usize,
        path: &str,
        is_cdrom: bool,
        discs: Vec<String>,
        overlay: bool,
        overlay_path_override: Option<&str>,
    ) -> std::io::Result<()> {
        use crate::cow_disk::CowDisk;
        use crate::scsi::DiskBackend;

        // Empty CD-ROM (drive present, tray empty). Stored backend=None so
        // TEST UNIT READY / READ CAPACITY / READ return MEDIUM NOT PRESENT,
        // while INQUIRY still advertises the drive. Use insert_disc() to
        // mount media later.
        if is_cdrom && path.is_empty() {
            let mut state = self.state.lock();
            if id < 8 {
                state.devices[id] = Some(crate::scsi::ScsiDevice::new_empty_cdrom());
            }
            return Ok(());
        }

        #[cfg(feature = "chd")]
        let is_chd_path = crate::chd_disk::is_chd(path);
        #[cfg(not(feature = "chd"))]
        let is_chd_path = {
            let p = path.to_ascii_lowercase();
            p.ends_with(".chd")
        };

        let (backend, size) = if is_chd_path {
            #[cfg(feature = "chd")]
            {
                use crate::chd_disk::{ChdCd, ChdHd};
                if is_cdrom {
                    let cd = ChdCd::open(path)?;
                    let sz = cd.size();
                    (DiskBackend::ChdCd(cd), sz)
                } else {
                    // HD CHD. The `overlay` flag is the per-disk copy-on-write
                    // toggle: COW on → always overlay (even an uncompressed base
                    // gets a diff) and never auto-fold on exit (commit/roll back
                    // manually); COW off → write in place (uncompressed) or a diff
                    // that auto-folds on a clean exit (compressed).
                    let hd = ChdHd::open(path, overlay)?;
                    let sz = hd.size();
                    (DiskBackend::ChdHd(hd), sz)
                }
            }
            #[cfg(not(feature = "chd"))]
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "CHD image support not compiled in (rebuild with --features chd)",
                ));
            }
        } else if overlay && !is_cdrom {
            let overlay_path = overlay_path_override
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{}.overlay", path));
            let cow = CowDisk::new(path, &overlay_path)?;
            let sz = cow.size();
            (DiskBackend::Cow(cow), sz)
        } else {
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(!is_cdrom)
                .open(path)?;
            let sz = file.metadata()?.len();
            (DiskBackend::Direct(file), sz)
        };

        let disc_list = if is_cdrom { discs } else { vec![] };

        let mut state = self.state.lock();
        if id < 8 {
            state.devices[id] = Some(ScsiDevice::new(backend, size, is_cdrom, path.to_string(), disc_list));
        }
        Ok(())
    }

    /// Attach a DaynaPort SCSI/Link target — a SCSI-attached Ethernet adapter.
    /// Unlike every other target it has no file backing at all, so none of the
    /// image/CHD/overlay path above runs. Its backend thread (NAT gateway, or
    /// PCAP bridge) is started here, before the device becomes visible on the
    /// bus, so the first INQUIRY already finds a live interface.
    #[cfg(feature = "daynaport")]
    pub fn add_daynaport(
        &self,
        id: usize,
        mac: [u8; 6],
        gateway: crate::net::GatewayConfig,
    ) -> std::io::Result<()> {
        if id >= 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput, "SCSI ID out of range"));
        }
        let mut dp = crate::daynaport::DaynaPort::new(id, mac, gateway, self.heartbeat.clone());
        dp.start();
        let mut state = self.state.lock();
        state.devices[id] = Some(ScsiDevice::new_daynaport(dp));
        Ok(())
    }

    #[cfg(not(feature = "daynaport"))]
    pub fn add_daynaport(
        &self,
        _id: usize,
        _mac: [u8; 6],
        _gateway: crate::net::GatewayConfig,
    ) -> std::io::Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "DaynaPort support not compiled in (rebuild with --features daynaport)",
        ))
    }

    /// Mount media on a CD-ROM device (newly inserts or swaps existing).
    /// Errors if the slot is empty, is not a CD-ROM, or the file can't open.
    pub fn insert_disc(&self, id: usize, path: &str) -> Result<(), String> {
        let mut state = self.state.lock();
        match state.devices.get_mut(id).and_then(|d| d.as_mut()) {
            None => Err(format!("No device at SCSI ID {}", id)),
            Some(dev) if !dev.is_cdrom() => Err(format!("SCSI ID {} is not a CD-ROM", id)),
            Some(dev) => dev.insert_media(path).map_err(|e| format!("insert_disc: {}", e)),
        }
    }

    /// Unload media from a CD-ROM (leave the drive present but empty).
    /// Use this when you want "drive present, no disc" rather than the
    /// changer-cycle behaviour of `eject_disc`.
    pub fn eject_to_empty(&self, id: usize) -> Result<(), String> {
        let mut state = self.state.lock();
        match state.devices.get_mut(id).and_then(|d| d.as_mut()) {
            None => Err(format!("No device at SCSI ID {}", id)),
            Some(dev) if !dev.is_cdrom() => Err(format!("SCSI ID {} is not a CD-ROM", id)),
            Some(dev) => { dev.unload_media(); Ok(()) }
        }
    }

    /// Eject the current disc on a CD-ROM device and advance to the next in
    /// the changer list.  Returns the new active path, or an error string.
    pub fn eject_disc(&self, id: usize) -> Result<String, String> {
        let mut state = self.state.lock();
        match state.devices.get_mut(id).and_then(|d| d.as_mut()) {
            None => Err(format!("No device at SCSI ID {}", id)),
            Some(dev) => dev.eject_next().ok_or_else(|| {
                if !dev.is_cdrom() {
                    format!("SCSI ID {} is not a CD-ROM", id)
                } else {
                    format!("SCSI ID {} has only one disc (nothing to eject to)", id)
                }
            }),
        }
    }

    /// Load an arbitrary image path into a CD-ROM device and make it the active
    /// disc immediately (medium change). Returns the loaded path.
    pub fn load_disc(&self, id: usize, path: String) -> Result<String, String> {
        let mut state = self.state.lock();
        match state.devices.get_mut(id).and_then(|d| d.as_mut()) {
            None => Err(format!("No device at SCSI ID {}", id)),
            Some(dev) => dev.load_disc(path),
        }
    }

    /// Add a disc path at position 1 (next-after-eject) for a CD-ROM device.
    pub fn add_disc(&self, id: usize, path: String) -> Result<(), String> {
        let mut state = self.state.lock();
        match state.devices.get_mut(id).and_then(|d| d.as_mut()) {
            None => Err(format!("No device at SCSI ID {}", id)),
            Some(dev) => dev.add_disc(path),
        }
    }

    /// Remove a disc by ordinal from a CD-ROM device's queue.
    pub fn remove_disc(&self, id: usize, ordinal: usize) -> Result<String, String> {
        let mut state = self.state.lock();
        match state.devices.get_mut(id).and_then(|d| d.as_mut()) {
            None => Err(format!("No device at SCSI ID {}", id)),
            Some(dev) => dev.remove_disc(ordinal),
        }
    }

    /// Move a disc by ordinal to position 1 (next-after-eject).
    pub fn move_disc_next(&self, id: usize, ordinal: usize) -> Result<(), String> {
        let mut state = self.state.lock();
        match state.devices.get_mut(id).and_then(|d| d.as_mut()) {
            None => Err(format!("No device at SCSI ID {}", id)),
            Some(dev) => dev.move_disc_next(ordinal),
        }
    }

    /// Return disc info for all attached CD-ROM devices.
    pub fn disc_status(&self) -> Vec<(usize, String, Vec<String>, u64, u64)> {
        let state = self.state.lock();
        state.devices.iter().enumerate()
            .filter_map(|(id, d)| {
                let dev = d.as_ref()?;
                if !dev.is_cdrom() { return None; }
                let (phys, logical) = dev.block_sizes();
                Some((id, dev.current_disc().to_string(), dev.disc_list().to_vec(), phys, logical))
            })
            .collect()
    }

    /// Reset the COW overlay on every attached device that's using COW.
    /// Direct-mode devices are left alone. Used by `Machine::ci_restore`.
    pub fn reset_all_overlays(&self) -> Vec<(usize, std::io::Result<()>)> {
        let mut state = self.state.lock();
        let mut results = Vec::new();
        for id in 0..8 {
            if let Some(dev) = &mut state.devices[id] {
                if dev.is_cow() {
                    results.push((id, dev.cow_reset()));
                }
            }
        }
        results
    }

    /// Number of attached CHD devices whose `.diff.chd` holds changes pending a
    /// fold-back into the base on a clean shutdown.
    pub fn pending_chd_sync_count(&self) -> usize {
        let state = self.state.lock();
        state.devices.iter().flatten().filter(|d| d.pending_chd_sync().is_some()).count()
    }

    /// Fold pending CHD diffs back into their bases ("sync"), preserving the
    /// base's compression. `only` limits it to one SCSI id; `None` means all.
    /// Releases each disk backend first (closing the CHD), then rebuilds
    /// outside the device lock since recompression is slow.
    /// `progress(done, total, fraction)` reports per-disk progress; `cancel()`
    /// stops before the next disk (the in-flight rebuild also honours it),
    /// leaving every un-synced base+diff intact. Returns the count synced.
    #[cfg(feature = "chd")]
    pub fn sync_chd_disks(
        &self,
        only: Option<usize>,
        progress: &mut dyn FnMut(usize, usize, f32),
        cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        // Collect pending (base, diff) pairs, releasing CHD handles under the
        // lock so the files can be atomically rebuilt. The rebuild itself runs
        // unlocked.
        let pending: Vec<(std::path::PathBuf, std::path::PathBuf)> = {
            let mut state = self.state.lock();
            let mut v = Vec::new();
            for id in 0..8 {
                if only.is_some_and(|o| o != id) { continue; }
                if let Some(dev) = &mut state.devices[id] {
                    if let Some(pair) = dev.take_pending_chd_sync() {
                        v.push(pair);
                    }
                }
            }
            v
        };
        let total = pending.len();
        let mut done = 0usize;
        for (base, diff) in pending {
            if cancel() {
                break;
            }
            progress(done, total, 0.0);
            crate::chd_disk::flatten_diff(&base, &diff, &mut |f| progress(done, total, f), cancel)?;
            done += 1;
            progress(done, total, 1.0);
        }
        Ok(done)
    }

    #[cfg(not(feature = "chd"))]
    pub fn sync_chd_disks(
        &self,
        _only: Option<usize>,
        _progress: &mut dyn FnMut(usize, usize, f32),
        _cancel: &dyn Fn() -> bool,
    ) -> std::io::Result<usize> {
        Ok(0)
    }

    /// Copy every COW overlay into `dir` as `scsi<id>.overlay`. Returns a
    /// list of `(id, dirty_sector_list)` entries so snapshot save can
    /// persist the dirty set alongside the raw overlay bytes.
    pub fn export_overlays(&self, dir: &std::path::Path) -> std::io::Result<Vec<(usize, Vec<u64>)>> {
        let mut state = self.state.lock();
        let mut out = Vec::new();
        for id in 0..8 {
            if let Some(dev) = &mut state.devices[id] {
                if dev.is_cow() {
                    let dest = dir.join(format!("scsi{}.overlay", id));
                    let dirty = dev.cow_export(&dest)?;
                    out.push((id, dirty));
                }
            }
        }
        Ok(out)
    }

    /// Replace each COW overlay with its saved counterpart in `dir` and
    /// adopt the matching dirty sector set. Devices with no corresponding
    /// entry in `dirty_sets` keep their current overlay untouched.
    pub fn import_overlays(
        &self,
        dir: &std::path::Path,
        dirty_sets: &[(usize, Vec<u64>)],
    ) -> std::io::Result<()> {
        let mut state = self.state.lock();
        for (id, dirty) in dirty_sets {
            if let Some(dev) = &mut state.devices[*id] {
                if dev.is_cow() {
                    let src = dir.join(format!("scsi{}.overlay", id));
                    dev.cow_import(&src, dirty.clone())?;
                }
            }
        }
        Ok(())
    }

    pub fn read_fifo(&self) -> u8 {
        let mut state = self.state.lock();
        state.fifo.pop_front().unwrap_or(0)
    }

    pub fn write_fifo(&self, val: u8, notify: bool) {
        let mut state = self.state.lock();
        state.fifo.push_back(val);
        if notify {
            self.cond.notify_one();
        }
    }

    pub fn read(&self, addr: u32) -> BusRead8 {
        let mut state = self.state.lock();

        if addr == 0 {
            // Read ASR (Auxiliary Status Register)
            return BusRead8::ok(state.read_asr());
        } else if addr == 1 {
            // Read register pointed to by AR
            let ar = state.ar & 0x1F;

            if ar == regs::DATA {
                let val = state.fifo.pop_front().unwrap_or(0);
                wdt!("R FIFO -> {:02x} (fifo_remaining={})",
                    val, state.fifo.len());
                dlog!(state.log_module(), "WD33C93A({}): Read FIFO -> {:02x}", state.id, val);

                // DBR-based PIO delivery: when DBR was set (data/status/msg byte ready),
                // each DATA read decrements TC and advances phase on fifo drain.
                // TRANSFER_COUNT = data-in bytes, RECEIVE_STATUS = status byte, STATUS_RECEIVED = msg byte.
                let cmd_phase = state.regs[regs::COMMAND_PHASE as usize];
                let mut fire_irq = false;
                let dbr_was_set = (state.asr & asr::DBR) != 0;
                if dbr_was_set {
                    state.decrement_transfer_count();
                    if !state.fifo.is_empty() {
                        // More bytes remain in this phase — keep DBR set for next byte.
                        // (No change needed; DBR stays as-is.)
                    } else {
                        state.update_asr(asr::DBR, 0);
                        // Fifo drained — advance to next phase via interrupt.
                        match cmd_phase {
                            command_phase::TRANSFER_COUNT => {
                                state.regs[regs::TARGET_LUN as usize] = state.pending_status;
                                wdt!("PIO_INT TRANSFER_COUNT→RECEIVE_STATUS phase={:02x}", cmd_phase);
                                state.queue_interrupt(Some(command_phase::RECEIVE_STATUS), scsi_status::TRANSFER_STATUS_IN);
                                fire_irq = true;
                            }
                            command_phase::RECEIVE_STATUS => {
                                wdt!("PIO_INT RECEIVE_STATUS→STATUS_RECEIVED phase={:02x}", cmd_phase);
                                state.queue_interrupt(Some(command_phase::STATUS_RECEIVED), scsi_status::TRANSFER_MSG_IN);
                                fire_irq = true;
                            }
                            command_phase::STATUS_RECEIVED => {
                                wdt!("PIO_INT STATUS_RECEIVED→COMPLETE_MSG phase={:02x}", cmd_phase);
                                state.queue_interrupt(Some(command_phase::COMPLETE_MSG), scsi_status::DISCONNECT);
                                fire_irq = true;
                            }
                            _ => {}
                        }
                    }
                }

                state.last_read_asr = None;
                state.last_read_reg = None;
                if fire_irq {
                    state.update_irq();
                }
                drop(state);
                return BusRead8::ok(val);
            }

            if ar == regs::AUX_STATUS_DIRECT {
                return BusRead8::ok(state.read_asr());
            }

            let val = state.regs[ar as usize];

            // Reading SCSI Status (0x17) clears the interrupt bit in ASR.
            // On the real WD33C93 + HPC3 pair, the SCSI ISR also implicitly
            // acks any pending PDMA SCSI0_DMA completion — both halves of the
            // SCSI line share the same INT3 source and the chip-INT ack settles
            // the line for both.  Without that coupling here, the PDMA bit
            // stays asserted forever once the chip-INT is cleared, IP2 storms,
            // and the IRIX 6.5 miniroot stalls.
            if ar == regs::SCSI_STATUS {
                // Ack the current interrupt and advance the IRQ FIFO.
                // If another entry is queued, update_irq() re-asserts INT immediately
                // without dropping the interrupt line — the handler sees a new status
                // without any line toggle. If the FIFO is empty, deassert the line.
                // LCI is cleared here — driver reads SCSI_STATUS to ack INT, then
                // re-issues the ignored command.
                state.asr &= !asr::LCI;
                let had_int = (state.asr & asr::INT) != 0;
                state.update_irq();
/*
                let status_name = match val {
                    0x00 => "RESET",
                    0x01 => "RESET_EAF",
                    0x10 => "RESELECT_SUCCESS",
                    0x11 => "SELECT_SUCCESS",
                    0x13 => "COMMAND_SUCCESS",
                    0x14 => "COMMAND_ATN_SUCCESS",
                    0x15 => "TRANSLATE_SUCCESS",
                    0x16 => "SELECT_TRANSFER_SUCCESS",
                    0x18 => "TRANSFER_DATA_OUT",
                    0x19 => "TRANSFER_DATA_IN",
                    0x1B => "TRANSFER_STATUS_IN",
                    0x1F => "TRANSFER_MSG_IN",
                    0x20 => "TRANSFER_PAUSE",
                    0x21 => "SAVE_DATA_POINTERS",
                    0x22 => "SELECTION_ABORTED",
                    0x23 => "RECEIVE_SEND_ABORTED",
                    0x24 => "RECEIVE_SEND_ABORTED_ATN",
                    0x25 => "ABORT_DURING_SELECTION",
                    0x27 => "RESELECTED_AFTER_DISC",
                    0x28 => "TRANSFER_ABORTED",
                    0x40 => "INVALID_COMMAND",
                    0x41 => "UNEXPECTED_DISCONNECT",
                    0x42 => "SELECTION_TIMEOUT",
                    0x43 => "PARITY_ERROR",
                    0x44 => "PARITY_ERROR_ATN",
                    0x45 => "LOGICAL_ADDRESS_TOO_LARGE",
                    0x46 => "RESELECTION_MISMATCH",
                    0x47 => "INCORRECT_STATUS_BYTE",
                    0x48 => "UNEXPECTED_RECV_DATA",
                    0x49 => "UNEXPECTED_SEND_DATA",
                    0x4A => "UNEXPECTED_CMD_PHASE",
                    0x4B => "UNEXPECTED_SEND_STATUS",
                    0x4E => "UNEXPECTED_REQ_MSG_OUT",
                    0x4F => "UNEXPECTED_SEND_MSG_IN",
                    0x80 => "RESELECTED",
                    0x81 => "RESELECTED_EAF",
                    0x82 => "SELECTED",
                    0x83 => "SELECTED_ATN",
                    0x84 => "ATN",
                    0x85 => "DISCONNECT",
                    0x87 => "UNKNOWN_GROUP",
                    0x88 => "REQ_DATA_OUT",
                    0x89 => "REQ_DATA_IN",
                    0x8A => "REQ_CMD_PHASE",
                    0x8B => "REQ_STATUS",
                    0x8E => "REQ_SEND_MSG_OUT",
                    0x8F => "REQ_MSG_IN",
                    _    => "?",
                };
                let cmd_name = match state.last_cmd & 0x7F {
                    0x00 => "RESET",
                    0x01 => "ABORT",
                    0x02 => "ASSERT_ATN",
                    0x03 => "NEGATE_ACK",
                    0x04 => "DISCONNECT",
                    0x05 => "RESELECT",
                    0x06 => "SELECT_ATN",
                    0x07 => "SELECT",
                    0x08 => "SELECT_ATN_XFER",
                    0x09 => "SELECT_XFER",
                    0x0A => "RESELECT_RECEIVE",
                    0x0B => "RESELECT_SEND",
                    0x0C => "WAIT_SELECT_RECEIVE",
                    0x10 => "SEND_STATUS",
                    0x11 => "SEND_DISCONNECT",
                    0x12 => "SET_IDI",
                    0x13 => "RECEIVE_COMMAND",
                    0x14 => "RECEIVE_DATA",
                    0x15 => "RECEIVE_MSG_OUT",
                    0x16 => "RECEIVE_INFO",
                    0x17 => "SEND_COMMAND",
                    0x18 => "SEND_DATA",
                    0x19 => "SEND_MSG_IN",
                    0x1A => "SEND_INFO",
                    0x20 => "TRANSFER_INFO",
                    _    => "?",
                };
                eprintln!("SCSI INT: status={:#04x} ({}) cmd={:#04x} ({}{})",
                    val, status_name,
                    state.last_cmd, cmd_name,
                    if state.last_cmd & 0x80 != 0 { "+SBT" } else { "" });
*/                    
            }

            wdt!("R REG[{:02x}] -> {:02x} (phase={:02x} stat={:02x} asr={:02x})",
                ar, val,
                state.regs[regs::COMMAND_PHASE as usize],
                state.regs[regs::SCSI_STATUS as usize],
                state.asr);

            // Auto-increment for registers except COMMAND (0x18), DATA (0x19), AUX_STATUS (0x1F)
            if ar != regs::COMMAND && ar != regs::AUX_STATUS_DIRECT {
                state.ar = (ar + 1) & 0x1F;
            }

            if ar == regs::TARGET_LUN || ar == regs::SCSI_STATUS {
                dlog!(state.log_module(), "WD33C93A({}): Read Reg {:02x} ({}) -> {:02x}", state.id,
                    ar,
                    if ar == regs::TARGET_LUN { "TARGET_LUN" } else { "SCSI_STATUS" },
                    val);
                state.last_read_reg = Some((ar, val));
                state.last_read_asr = None;
            } else {
                let should_print = match state.last_read_reg {
                    None => true,
                    Some((last_reg, last_val)) => last_reg != ar || last_val != val,
                };
                if should_print {
                    dlog!(state.log_module(), "WD33C93A({}): Read Reg {:02x} -> {:02x}", state.id, ar, val);
                    state.last_read_reg = Some((ar, val));
                }
                state.last_read_asr = None;
            }
            return BusRead8::ok(val);
        }
        BusRead8::err()
    }

    pub fn write(&self, addr: u32, val: u8) -> u32 {
        let mut state = self.state.lock();

        if addr == 0 {
            // Write AR (Address Register)
            state.ar = val & 0x1F;
            wdt!("W AR  <- {:02x}", val);
            dlog!(state.log_module(), "WD33C93A({}): Write AR <- {:02x}", state.id, val);
            state.last_read_asr = None;
            state.last_read_reg = None;
            return BUS_OK;
        } else if addr == 1 {
            // Write register pointed to by AR
            let ar = state.ar & 0x1F;

            {
                let label = match ar {
                    regs::DATA => "DATA",
                    regs::COMMAND => "COMMAND",
                    regs::AUX_STATUS_DIRECT => "ASR_DIR",
                    _ => "REG",
                };
                wdt!("W {}[{:02x}] <- {:02x} (phase={:02x})",
                    label, ar, val, state.regs[regs::COMMAND_PHASE as usize]);
            }
            dlog!(state.log_module(), "WD33C93A({}): Write Reg {:02x} <- {:02x}", state.id, ar, val);
            state.last_read_asr = None;
            state.last_read_reg = None;

            if ar == regs::DATA {
                // In data-in mode the host reads DATA; writes are abort-flush and must be
                // discarded — pushing them would re-fill the fifo and stall the drain loop.
                if !state.xfer_direction_in {
                    state.fifo.push_back(val);
                }
                if (state.asr & asr::DBR) != 0 {
                    state.decrement_transfer_count();
                    let tc = state.get_transfer_count();
                    let cmd_phase = state.regs[regs::COMMAND_PHASE as usize];
                    if tc == 0 {
                        if cmd_phase == command_phase::SELECTED
                            || cmd_phase == command_phase::IDENTIFY_SENT
                            || cmd_phase == command_phase::COMMAND_START
                        {
                            // All PIO outbound phases: fifo full, wake worker to process.
                            dlog!(state.log_module(), "WD33C93A({}): PIO xfer complete phase=0x{:02x}, waking worker ({} bytes)", state.id, cmd_phase, state.fifo.len());
                            state.update_asr(asr::DBR, 0);
                            state.pending_command = Some(state.regs[regs::COMMAND as usize]);
                            drop(state);
                            self.cond.notify_one();
                            return BUS_OK;
                        }
                    } else {
                        let sbt = (state.regs[regs::COMMAND as usize] & 0x80) != 0;
                        if !sbt {
                            state.update_asr(0, asr::DBR);
                        } else {
                            state.update_asr(asr::DBR, 0);
                        }
                    }
                }
                drop(state);
                return BUS_OK;
            }

            state.regs[ar as usize] = val;

            // TARGET_LUN (0x0F) is dual-purpose on real hardware: the host writes
            // it before SELECT to set the LUN for the auto-generated IDENTIFY
            // message, but the chip later overwrites the same register with the
            // status byte for host readback once a command concludes. Shadow the
            // host's write separately so the LUN survives past that point — see
            // target_lun's field comment and its use in process_scsi_command.
            if ar == regs::TARGET_LUN {
                state.target_lun = val & 0x7;
            }

            // QUEUE_TAG (0x1A) is not implemented on WD33C93A — the register
            // doesn't latch. OpenBSD probes by writing 0x55 and reading back;
            // returning 0 here makes it detect us as WD33C93A, not B.
            if ar == regs::QUEUE_TAG {
                state.regs[regs::QUEUE_TAG as usize] = 0;
            }

            if ar == regs::COMMAND {
                // New command cycle: clear stale state from previous cycle.
                state.clear_interrupt();
                state.irq_fifo.clear();
                state.last_cmd = val;
                state.update_asr(0, asr::CIP);

                // TRANSFER_INFO in PIO mode: defer entirely to the CPU/DBR path.
                // Arm DBR, let bytes accumulate into fifo, wake worker when TC=0.
                // Worker sees the phase and fifo contents and processes correctly.
                // Covers all three outbound phases: SELECTED (MESG_OUT), IDENTIFY_SENT
                // (CDB after 0x8a), and COMMAND_START (write CDB re-issue).
                // In DMA mode this block is skipped — TRANSFER_INFO goes to worker normally.
                let cmd = val & !0x80; // strip SBT
                let phase = state.regs[regs::COMMAND_PHASE as usize];
                if cmd == cmd::TRANSFER_INFO
                    && (phase == command_phase::SELECTED
                        || phase == command_phase::IDENTIFY_SENT
                        || phase == command_phase::COMMAND_START)
                    && !state.use_dma()
                {
                    let tc = state.get_transfer_count();
                    let tc = if tc == 0 { state.set_transfer_count(1); 1 } else { tc };
                    dlog!(state.log_module(), "WD33C93A({}): XFER_INFO PIO deferred phase=0x{:02x} tc={}", state.id, phase, tc);
                    state.fifo.clear();
                    state.update_asr(asr::CIP | asr::INT, asr::DBR);
                    return BUS_OK;
                }

                state.pending_command = Some(val);
                self.cond.notify_one();
                // COMMAND register does not auto-increment
            } else if ar != regs::AUX_STATUS_DIRECT {
                // Auto-increment for registers except COMMAND, DATA, AUX_STATUS
                state.ar = (ar + 1) & 0x1F;
            }

            return BUS_OK;
        }
        BUS_ERR
    }

    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_lock_fn;
        let me = self.clone(); register_lock_fn("scsi::state",  move || me.state.is_locked());
        let me = self.clone(); register_lock_fn("scsi::thread", move || me.thread.is_locked());
    }
}

impl FifoDevice for Wd33c93a {
    fn read_fifo(&self) -> u8 {
        self.read_fifo()
    }
    fn write_fifo(&self, val: u8, notify: bool) {
        self.write_fifo(val, notify)
    }
}

impl Device for Wd33c93a {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.cond.notify_all();
        if let Some(t) = self.thread.lock().take() {
            let _ = t.join();
        }
        // Any DaynaPort target owns a backend thread of its own; stop it too
        // (after the worker is joined, so nothing is mid-command).
        #[cfg(feature = "daynaport")]
        {
            let mut state = self.state.lock();
            for dev in state.devices.iter_mut().flatten() {
                if let Some(dp) = dev.daynaport_mut() { dp.stop(); }
            }
        }
    }
    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }
        // Re-arm any DaynaPort backend a previous stop() shut down. No-op on
        // first boot — add_daynaport() already started them.
        #[cfg(feature = "daynaport")]
        {
            let mut st = self.state.lock();
            for dev in st.devices.iter_mut().flatten() {
                if let Some(dp) = dev.daynaport_mut() { dp.start(); }
            }
        }
        let state = self.state.clone();
        let cond = self.cond.clone();
        let running = self.running.clone();
        let dma = self.dma.clone();
        // CyclesPtr is Copy + Send + Sync — extracted once here, before the
        // thread spawns (see that type's and cpu_cycles's own doc comments).
        let cpu_cycles = self.cpu_cycles.get();
        let heartbeat = self.heartbeat.clone();

        *self.thread.lock() = Some(thread::Builder::new().name("WD33C93A".to_string()).spawn(move || {
            let mut state_guard = state.lock();
            while running.load(Ordering::Relaxed) {
                cond.wait(&mut state_guard);
                if !running.load(Ordering::Relaxed) { break; }

                // Process command if one was queued.
                if let Some(cmd_reg) = state_guard.pending_command.take() {
                    dlog!(state_guard.log_module(), "WD33C93A({}): Processing Command {:02x}", state_guard.id, cmd_reg);
                    state_guard.process_wd_command(cmd_reg, dma.as_deref());

                    let tid = state_guard.target_id;
                    if tid < 7 {
                        heartbeat.fetch_or(1u64 << (crate::rex3::Rex3::HB_SCSI_BASE as u64 + tid as u64), Ordering::Relaxed);
                    }
                }

                // Deliver queued interrupt — update_irq sequences ASR transitions
                // and fires the interrupt line via set_asr when the front changes.
                if !state_guard.irq_fifo.is_empty() {
                    let deferred = state_guard.irq_fifo.front().map(|e| e.2).unwrap_or(false);
                    if deferred && state_guard.deferred_int.load(Ordering::Relaxed) {
                        // Clear CIP now so WAIT_CIP exits with INT=0, then spin 10000 cycles
                        // before asserting INT — giving wd33c93_loop time to exit via
                        // GET_SBIC_asr seeing INT=0 before we deliver the interrupt.
                        state_guard.update_asr(asr::CIP | asr::DBR, 0);
                        let deferred_int = state_guard.deferred_int.clone();
                        let start = cpu_cycles.get();
                        drop(state_guard);
                        loop {
                            let now = cpu_cycles.get();
                            if now.wrapping_sub(start) >= 10000 { break; }
                            // Re-check: if disabled at runtime, exit spin early
                            if !deferred_int.load(Ordering::Relaxed) { break; }
                        }
                        state_guard = state.lock();
                    }
                    state_guard.update_irq();
                }
            }
        }).unwrap());
    }
    fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("scsi".to_string(), "SCSI: scsi regs | scsi status | scsi dayna | scsi wdt [N] | scsi wdt file <path> | scsi eject <id> | scsi add <id> <path> | scsi list <id> | scsi del <id> <ord> | scsi next <id> <ord> | scsi debug <on|off> [DEV] | scsi defer <on|off>".to_string()),
            ("cow".to_string(), "COW overlay: cow status | cow commit [id] | cow reset [id]".to_string()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd == "scsi" || cmd == "scsi0" || cmd == "scsi1" {
            match args.first().copied() {
                Some("wdt") => {
                    #[cfg(feature = "developer")] {
                        // "scsi wdt [N]"          — dump last N entries to console
                        // "scsi wdt file <path>"  — dump entire ring to file
                        let file_mode = args.get(1).copied() == Some("file");
                        if file_mode {
                            let path = args.get(2).ok_or_else(|| "Usage: scsi wdt file <path>".to_string())?;
                            let mut f = std::fs::File::create(path)
                                .map_err(|e| format!("Cannot create {}: {}", path, e))?;
                            let ring = WDT_RING.lock();
                            for (seq, msg, cnt) in ring.entries.iter() {
                                if *cnt == 1 {
                                    writeln!(f, "WDT {:8} {}", seq, msg).unwrap();
                                } else {
                                    writeln!(f, "WDT {:8} {}   (x{})", seq, msg, cnt).unwrap();
                                }
                            }
                            writeln!(f, "WDT_DUMP_END (total_seen={})", WDT_SEQ.load(Ordering::Relaxed)).unwrap();
                            drop(ring);
                            writeln!(writer, "WDT ring written to {}", path).unwrap();
                        } else {
                            let n: Option<usize> = args.get(1).and_then(|s| s.parse().ok());
                            let ring = WDT_RING.lock();
                            let entries = &ring.entries;
                            let skip = n.map(|n| entries.len().saturating_sub(n)).unwrap_or(0);
                            for (seq, msg, cnt) in entries.iter().skip(skip) {
                                if *cnt == 1 {
                                    writeln!(writer, "WDT {:8} {}", seq, msg).unwrap();
                                } else {
                                    writeln!(writer, "WDT {:8} {}   (x{})", seq, msg, cnt).unwrap();
                                }
                            }
                            writeln!(writer, "(total_seen={})", WDT_SEQ.load(Ordering::Relaxed)).unwrap();
                        }
                    }
                    #[cfg(not(feature = "developer"))]
                    writeln!(writer, "WDT ring not available (build with --features developer)").unwrap();
                    return Ok(());
                }
                Some("regs") => {
                    let state = self.state.lock();
                    let r = &state.regs;
                    writeln!(writer, "WD33C93A Register Dump").unwrap();
                    writeln!(writer, "  AR  (addr reg) : {:02x}", state.ar).unwrap();
                    writeln!(writer, "  ASR (aux stat) : {:02x}  INT={} LCI={} BSY={} CIP={} PE={} DBR={}",
                        state.asr,
                        (state.asr & asr::INT) != 0,
                        (state.asr & asr::LCI) != 0,
                        (state.asr & asr::BSY) != 0,
                        (state.asr & asr::CIP) != 0,
                        (state.asr & asr::PE)  != 0,
                        (state.asr & asr::DBR) != 0,
                    ).unwrap();
                    writeln!(writer, "  00 OWN_ID       : {:02x}  id={} clk={} eaf={}",
                        r[0x00], r[0x00] & 0x07, (r[0x00] >> 6) & 0x03, (r[0x00] & 0x08) != 0).unwrap();
                    writeln!(writer, "  01 CONTROL      : {:02x}  hsp={} haf={} esr={} edbo={} polled={}",
                        r[0x01],
                        (r[0x01] & 0x01) != 0, (r[0x01] & 0x02) != 0,
                        (r[0x01] & 0x04) != 0, (r[0x01] & 0x08) != 0,
                        (r[0x01] & 0x80) != 0).unwrap();
                    writeln!(writer, "  02 TIMEOUT      : {:02x}", r[0x02]).unwrap();
                    writeln!(writer, "  03-0E CDB       : {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}  {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                        r[0x03], r[0x04], r[0x05], r[0x06], r[0x07], r[0x08],
                        r[0x09], r[0x0A], r[0x0B], r[0x0C], r[0x0D], r[0x0E]).unwrap();
                    writeln!(writer, "  0F TARGET_LUN   : {:02x}  tgt={} lun={}", r[0x0F], (r[0x0F] >> 3) & 0x07, r[0x0F] & 0x07).unwrap();
                    writeln!(writer, "  10 CMD_PHASE    : {:02x}", r[0x10]).unwrap();
                    writeln!(writer, "  11 SYNC_XFER    : {:02x}  tp={} offset={}", r[0x11], (r[0x11] >> 4) & 0x07, r[0x11] & 0x0F).unwrap();
                    let tc = ((r[0x12] as u32) << 16) | ((r[0x13] as u32) << 8) | (r[0x14] as u32);
                    writeln!(writer, "  12-14 XFER_CNT  : {:02x} {:02x} {:02x} = {}", r[0x12], r[0x13], r[0x14], tc).unwrap();
                    writeln!(writer, "  15 DEST_ID      : {:02x}  id={}", r[0x15], r[0x15] & 0x07).unwrap();
                    writeln!(writer, "  16 SOURCE_ID    : {:02x}", r[0x16]).unwrap();
                    writeln!(writer, "  17 SCSI_STATUS  : {:02x}", r[0x17]).unwrap();
                    writeln!(writer, "  18 COMMAND      : {:02x}", r[0x18]).unwrap();
                    writeln!(writer, "  19 DATA         : {:02x}", r[0x19]).unwrap();
                    writeln!(writer, "  1A QUEUE_TAG    : {:02x}", r[0x1A]).unwrap();
                    writeln!(writer, "  Internal state:").unwrap();
                    writeln!(writer, "    target_id={} adv_mode={} has_pending_cmd={}",
                        state.target_id, state.advanced_mode, state.pending_command.is_some()).unwrap();
                    writeln!(writer, "    pending_status={:02x} pending_msg={:02x}",
                        state.pending_status, state.pending_msg).unwrap();
                    writeln!(writer, "    xfer_offset={}/{} dir_in={} fifo_len={} tc={}",
                        state.xfer_offset, state.xfer_data.len(),
                        state.xfer_direction_in, state.fifo.len(),
                        state.get_transfer_count()).unwrap();
                    return Ok(());
                }
                Some("debug") => {
                    let val = match args.get(1).copied() {
                        Some("on")  => true,
                        Some("off") => false,
                        _ => return Err("Usage: scsi debug <on|off>".to_string()),
                    };
                    // Bare `scsi debug` toggles both controllers' logging;
                    // `scsi0 debug`/`scsi1 debug` toggle only their own —
                    // see LogModule::Scsi1's doc comment.
                    let modules: &[LogModule] = match cmd {
                        "scsi1" => &[LogModule::Scsi1],
                        "scsi0" => &[LogModule::Scsi],
                        _       => &[LogModule::Scsi, LogModule::Scsi1],
                    };
                    for m in modules {
                        if val { devlog().enable(*m); } else { devlog().disable(*m); }
                    }
                    writeln!(writer, "SCSI debug {} ({})", if val { "enabled" } else { "disabled" }, cmd).unwrap();
                    return Ok(());
                }
                Some("defer") => {
                    let val = match args.get(1).copied() {
                        Some("on")  => true,
                        Some("off") => false,
                        _ => return Err("Usage: scsi defer <on|off>".to_string()),
                    };
                    self.deferred_int.store(val, Ordering::Relaxed);
                    writeln!(writer, "SCSI deferred interrupts {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                Some("dayna") => {
                    #[cfg(feature = "daynaport")]
                    {
                        let state = self.state.lock();
                        let mut found = false;
                        for dev in state.devices.iter().flatten() {
                            if let Some(dp) = dev.daynaport() {
                                found = true;
                                for line in dp.status_lines() {
                                    writeln!(writer, "{}", line).unwrap();
                                }
                            }
                        }
                        if !found {
                            writeln!(writer, "No DaynaPort targets attached").unwrap();
                        }
                    }
                    #[cfg(not(feature = "daynaport"))]
                    writeln!(writer, "DaynaPort support not built in (rebuild with --features daynaport)").unwrap();
                    return Ok(());
                }
                Some("status") => {
                    #[cfg(feature = "daynaport")]
                    {
                        let state = self.state.lock();
                        for dev in state.devices.iter().flatten() {
                            if let Some(dp) = dev.daynaport() {
                                writeln!(writer, "{}", dp.status_lines()[0]).unwrap();
                            }
                        }
                    }
                    let discs = self.disc_status();
                    if discs.is_empty() {
                        writeln!(writer, "No CD-ROM devices attached").unwrap();
                    } else {
                        for (id, active, list, phys, logical) in discs {
                            writeln!(writer, "SCSI ID {}: {} ({} disc(s))  phys_block={} logical_block={}",
                                id, active, list.len(), phys, logical).unwrap();
                            for (i, d) in list.iter().enumerate() {
                                let tag = if i == 0 { " <active>" } else if i == 1 { " <next>" } else { "" };
                                writeln!(writer, "  [{}] {}{}", i, d, tag).unwrap();
                            }
                        }
                    }
                    return Ok(());
                }
                Some("eject") => {
                    let id: usize = args.get(1)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| "Usage: scsi eject <id>".to_string())?;
                    match self.eject_disc(id) {
                        Ok(path) => writeln!(writer, "SCSI ID {}: switched to {}", id, path).unwrap(),
                        Err(e)   => writeln!(writer, "Error: {}", e).unwrap(),
                    }
                    return Ok(());
                }
                Some("add") => {
                    let id: usize = args.get(1)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| "Usage: scsi add <id> <path>".to_string())?;
                    let path = args.get(2)
                        .ok_or_else(|| "Usage: scsi add <id> <path>".to_string())?
                        .to_string();
                    match self.add_disc(id, path.clone()) {
                        Ok(()) => writeln!(writer, "SCSI ID {}: added {} as next disc", id, path).unwrap(),
                        Err(e) => writeln!(writer, "Error: {}", e).unwrap(),
                    }
                    return Ok(());
                }
                Some("list") => {
                    let id: usize = args.get(1)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| "Usage: scsi list <id>".to_string())?;
                    let state = self.state.lock();
                    match state.devices.get(id).and_then(|d| d.as_ref()) {
                        None => writeln!(writer, "No device at SCSI ID {}", id).unwrap(),
                        Some(dev) if !dev.is_cdrom() => writeln!(writer, "SCSI ID {} is not a CD-ROM", id).unwrap(),
                        Some(dev) => {
                            let list = dev.disc_list();
                            writeln!(writer, "SCSI ID {} queue ({} disc(s)):", id, list.len()).unwrap();
                            for (i, d) in list.iter().enumerate() {
                                writeln!(writer, "  [{}] {}{}", i, d,
                                    if i == 0 { " <active>" } else if i == 1 { " <next>" } else { "" }).unwrap();
                            }
                        }
                    }
                    return Ok(());
                }
                Some("del") => {
                    let id: usize = args.get(1)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| "Usage: scsi del <id> <ordinal>".to_string())?;
                    let ordinal: usize = args.get(2)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| "Usage: scsi del <id> <ordinal>".to_string())?;
                    match self.remove_disc(id, ordinal) {
                        Ok(path) => writeln!(writer, "SCSI ID {}: removed [{}] {}", id, ordinal, path).unwrap(),
                        Err(e)   => writeln!(writer, "Error: {}", e).unwrap(),
                    }
                    return Ok(());
                }
                Some("next") => {
                    let id: usize = args.get(1)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| "Usage: scsi next <id> <ordinal>".to_string())?;
                    let ordinal: usize = args.get(2)
                        .and_then(|s| s.parse().ok())
                        .ok_or_else(|| "Usage: scsi next <id> <ordinal>".to_string())?;
                    match self.move_disc_next(id, ordinal) {
                        Ok(()) => writeln!(writer, "SCSI ID {}: [{}] moved to next position", id, ordinal).unwrap(),
                        Err(e) => writeln!(writer, "Error: {}", e).unwrap(),
                    }
                    return Ok(());
                }
                _ => return Err("Usage: scsi status | scsi dayna | scsi eject <id> | scsi add <id> <path> | scsi list <id> | scsi del <id> <ord> | scsi next <id> <ord> | scsi debug <on|off>".to_string()),
            }
        }
        if cmd == "cow" {
            let mut state = self.state.lock();
            match args.first().copied() {
                Some("status") => {
                    for (id, dev) in state.devices.iter().enumerate() {
                        if let Some(d) = dev {
                            if d.is_cow() {
                                writeln!(writer, "SCSI {}: COW overlay, {} dirty sectors", id, d.cow_dirty_count()).unwrap();
                            } else {
                                writeln!(writer, "SCSI {}: direct (no overlay)", id).unwrap();
                            }
                        }
                    }
                    return Ok(());
                }
                Some("commit") => {
                    let ids: Vec<usize> = if let Some(id_str) = args.get(1) {
                        vec![id_str.parse().map_err(|_| "invalid SCSI ID".to_string())?]
                    } else {
                        (0..8).filter(|&i| state.devices[i].as_ref().map(|d| d.is_cow()).unwrap_or(false)).collect()
                    };
                    for id in ids {
                        if let Some(dev) = &mut state.devices[id] {
                            match dev.cow_commit() {
                                Ok(n) if n > 0 => writeln!(writer, "SCSI {}: committed {} sectors to base image", id, n).unwrap(),
                                Ok(_) => writeln!(writer, "SCSI {}: nothing to commit", id).unwrap(),
                                Err(e) => writeln!(writer, "SCSI {}: commit failed: {}", id, e).unwrap(),
                            }
                        }
                    }
                    return Ok(());
                }
                Some("reset") => {
                    let ids: Vec<usize> = if let Some(id_str) = args.get(1) {
                        vec![id_str.parse().map_err(|_| "invalid SCSI ID".to_string())?]
                    } else {
                        (0..8).filter(|&i| state.devices[i].as_ref().map(|d| d.is_cow()).unwrap_or(false)).collect()
                    };
                    for id in ids {
                        if let Some(dev) = &mut state.devices[id] {
                            match dev.cow_reset() {
                                Ok(()) => writeln!(writer, "SCSI {}: overlay reset (all writes discarded)", id).unwrap(),
                                Err(e) => writeln!(writer, "SCSI {}: reset failed: {}", id, e).unwrap(),
                            }
                        }
                    }
                    return Ok(());
                }
                _ => return Err("Usage: cow status | cow commit [id] | cow reset [id]".to_string()),
            }
        }
        Err("Command not found".to_string())
    }
}

impl Default for Wd33c93a {
    fn default() -> Self {
        Self::new(None, None, Arc::new(AtomicU64::new(0)))
    }
}

// ============================================================================
// Resettable + Saveable for Wd33c93a
// ============================================================================

impl Resettable for Wd33c93a {
    /// Execute the WD33C93A RESET command in-place.
    /// Must be called with threads stopped.
    fn power_on(&self) {
        let mut state = self.state.lock();
        // Mirrors process_wd_command(cmd::RESET, ...) logic.
        state.fifo.clear();
        state.irq_fifo.clear();
        state.xfer_data.clear();
        state.xfer_offset = 0;
        state.regs[regs::COMMAND_PHASE as usize] = command_phase::DISCONNECTED;
        state.set_asr(asr::INT);
        state.target_id = 0;
        state.pending_status = 0;
        state.pending_msg = 0;
        state.advanced_mode = false;
        state.pending_command = None;
        state.last_read_asr = None;
        state.last_read_reg = None;
        // Hardware reset clears all registers including OWN_ID.
        // (Software CMD_RESET preserves OWN_ID and uses its EAF bit for RESET_EAF.)
        for i in 0x00..=0x16usize {
            state.regs[i] = 0;
        }
        state.regs[regs::COMMAND as usize] = 0;
        state.advanced_mode = false;
        state.regs[regs::SCSI_STATUS as usize] = scsi_status::RESET;
        // A DaynaPort comes up disabled with empty queues, and its NAT tables
        // are flushed on the backend thread's next loop — the same answer
        // Seeq8003::power_on gives for the onboard Ethernet.
        #[cfg(feature = "daynaport")]
        for dev in state.devices.iter_mut().flatten() {
            if let Some(dp) = dev.daynaport_mut() { dp.power_on(); }
        }
    }
}

impl Saveable for Wd33c93a {
    fn save_state(&self) -> toml::Value {
        let state = self.state.lock();
        let mut tbl = toml::map::Map::new();
        tbl.insert("regs".into(),              u8_slice_to_toml(&state.regs));
        tbl.insert("ar".into(),                hex_u8(state.ar));
        tbl.insert("asr".into(),               hex_u8(state.asr));
        tbl.insert("data_direction_in".into(), toml::Value::Boolean(state.data_direction_in));
        tbl.insert("target_id".into(),         hex_u8(state.target_id as u8));
        tbl.insert("target_lun".into(),        hex_u8(state.target_lun));
        tbl.insert("pending_status".into(),    hex_u8(state.pending_status));
        tbl.insert("pending_msg".into(),       hex_u8(state.pending_msg));
        tbl.insert("advanced_mode".into(),     toml::Value::Boolean(state.advanced_mode));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut state = self.state.lock();
        if let Some(r) = get_field(v, "regs") { load_u8_slice(r, &mut state.regs); }
        if let Some(x) = get_field(v, "ar")               { if let Some(n) = toml_u8(x)   { state.ar = n; } }
        if let Some(x) = get_field(v, "asr")              { if let Some(n) = toml_u8(x)   { state.set_asr(n); } }
        if let Some(x) = get_field(v, "data_direction_in"){ if let Some(b) = toml_bool(x) { state.data_direction_in = b; } }
        if let Some(x) = get_field(v, "target_id")        { if let Some(n) = toml_u8(x)   { state.target_id = n as usize; } }
        if let Some(x) = get_field(v, "target_lun")       { if let Some(n) = toml_u8(x)   { state.target_lun = n; } }
        if let Some(x) = get_field(v, "pending_status")   { if let Some(n) = toml_u8(x)   { state.pending_status = n; } }
        if let Some(x) = get_field(v, "pending_msg")      { if let Some(n) = toml_u8(x)   { state.pending_msg = n; } }
        if let Some(x) = get_field(v, "advanced_mode")    { if let Some(b) = toml_bool(x) { state.advanced_mode = b; } }
        // Transient state cleared on load.
        state.fifo.clear();
        state.xfer_data.clear();
        state.xfer_offset = 0;
        state.pending_command = None;
        state.last_read_asr = None;
        state.last_read_reg = None;
        Ok(())
    }
}

impl Wd33c93aState {
    /// LogModule::Scsi for controller 0, LogModule::Scsi1 for controller 1 —
    /// lets `scsi debug`/`scsi1 debug` toggle each chip's logging
    /// independently. See `id`'s doc comment.
    fn log_module(&self) -> LogModule {
        if self.id == 0 { LogModule::Scsi } else { LogModule::Scsi1 }
    }

    fn use_dma(&self) -> bool {
        let mode = (self.regs[regs::CONTROL as usize] >> 5) & 0x7;
        mode != 0
    }

    fn read_asr(&mut self) -> u8 {
        let val = self.asr;
        if self.last_read_asr.is_none() || self.last_read_asr.unwrap() != val {
            dlog!(self.log_module(), "WD33C93A({}): Read ASR -> {:02x}", self.id, val);
            self.last_read_asr = Some(val);
        }
        self.last_read_reg = None;
        wdt!("R ASR -> {:02x} (phase={:02x} stat={:02x})",
            val,
            self.regs[regs::COMMAND_PHASE as usize],
            self.regs[regs::SCSI_STATUS as usize]);
        val
    }

    // Apply a bit-clear/bit-set transition to ASR, firing the interrupt line if INT changes.
    fn update_asr(&mut self, clear: u8, set: u8) {
        let val = (self.asr & !clear) | set;
        self.set_asr(val);
    }

    // Write self.asr and fire the interrupt line if INT changed.
    fn set_asr(&mut self, val: u8) {
        let old_int = self.asr & asr::INT;
        self.asr = val;
        let new_int = self.asr & asr::INT;
        if new_int != old_int {
            wdt!("INT_LINE {} (asr={:02x})",
                if new_int != 0 { "ASSERT" } else { "DEASSERT" }, self.asr);
            if let Some(cb) = &self.callback { cb.set_interrupt(new_int != 0); }
        }
    }

    // Called on COMMAND write and ABORT: clear INT/DBR/LCI to start a new command cycle.
    // Driver writing a command implicitly acknowledges any pending interrupt.
    fn clear_interrupt(&mut self) {
        self.update_asr(asr::INT | asr::DBR | asr::LCI, 0);
    }

    /// Compute DBR (Data Buffer Ready) bit based on COMMAND_PHASE register

    /// Pop the front of the IRQ FIFO into SCSI_STATUS and assert ASR.INT.
    /// If the FIFO is empty, SCSI_STATUS is left unchanged and ASR.INT is cleared.
    /// Returns true if INT was asserted.
    fn update_irq(&mut self) -> bool {
        if let Some((status, phase_opt, deferred)) = self.irq_fifo.pop_front() {
            if let Some(phase) = phase_opt {
                self.regs[regs::COMMAND_PHASE as usize] = phase;
            }
            self.regs[regs::SCSI_STATUS as usize] = status;
            // Deferred: worker clears CIP and spins 10000 CPU cycles before calling
            // update_irq, giving wd33c93_loop time to exit before INT fires.
            self.update_asr(asr::CIP | asr::DBR, asr::INT);
            wdt!("QUEUE_IRQ phase={:02x} stat={:02x} deferred={} (fifo_remaining={})",
                self.regs[regs::COMMAND_PHASE as usize], status, deferred, self.irq_fifo.len());
            true
        } else {
            self.update_asr(asr::INT, 0);
            false
        }
    }

    /// Set COMMAND_PHASE and SCSI_STATUS registers, logging the transition.
    /// Does not touch ASR.INT — use raise_interrupt() to also queue an interrupt.
    fn set_status(&mut self, phase: u8, status: u8) {
        let old_phase = self.regs[regs::COMMAND_PHASE as usize];
        let old_status = self.regs[regs::SCSI_STATUS as usize];

        self.regs[regs::COMMAND_PHASE as usize] = phase;
        self.regs[regs::SCSI_STATUS as usize] = status;

        if old_phase != phase || old_status != status {
            if old_phase != phase && old_status != status {
                dlog!(self.log_module(), "WD33C93A({}): Phase {:02x}->{:02x} Status {:02x}->{:02x}", self.id,
                    old_phase, phase, old_status, status);
            } else if old_phase != phase {
                dlog!(self.log_module(), "WD33C93A({}): Phase {:02x}->{:02x} (Status={:02x})", self.id, old_phase, phase, status);
            } else {
                dlog!(self.log_module(), "WD33C93A({}): Status {:02x}->{:02x} (Phase={:02x})", self.id, old_status, status, phase);
            }
        }
    }

    /// Queue an interrupt for deferred delivery by the worker thread.
    /// Pushes (status, phase) into irq_fifo and clears CIP+DBR.
    /// Phase and status are written atomically by update_irq() when delivered.
    /// Pass None for phase when it should not change (e.g. multi-status SELECT_ATN).
    fn queue_interrupt(&mut self, phase: Option<u8>, status: u8) {
        self.queue_interrupt_ex(phase, status, false);
    }

    fn queue_interrupt_ex(&mut self, phase: Option<u8>, status: u8, deferred: bool) {
        dlog!(self.log_module(), "WD33C93A({}): queue_interrupt phase={:02x?} status={:02x} deferred={}", self.id, phase, status, deferred);
        self.irq_fifo.push_back((status, phase, deferred));
    }

    fn process_wd_command(&mut self, raw_cmd: u8, dma: Option<&dyn DmaClient>) {
        // Strip SBT (Single Byte Transfer) qualifier bit. SBT changes the host
        // polling protocol but not the logical operation — treat as the base command.
        let cmd = raw_cmd & !0x80;
        // Update target_id from DESTINATION_ID before logging so the log is accurate.
        match cmd {
            cmd::SELECT_ATN | cmd::SELECT_ATN_XFER | cmd::SELECT | cmd::SELECT_XFER => {
                self.target_id = (self.regs[regs::DESTINATION_ID as usize] & 0x7) as usize;
            }
            _ => {}
        }
        {
            let cmd_phase = self.regs[regs::COMMAND_PHASE as usize];
            dlog!(self.log_module(), "WD33C93A({}): Command {:02x} (CmdPhase: {:02x}, Tgt: {}, ASR: {:02x})", self.id,
                cmd, cmd_phase, self.target_id, self.asr);
            dlog!(self.log_module(), "WD33C93A({}):           Regs: CTRL={:02x} DST_ID={:02x} SRC_ID={:02x}", self.id,
                self.regs[regs::CONTROL as usize], self.regs[regs::DESTINATION_ID as usize], self.regs[regs::SOURCE_ID as usize]);
        }

        if cmd == cmd::RESET {
            self.fifo.clear();
            self.irq_fifo.clear();
            self.regs[regs::COMMAND_PHASE as usize] = command_phase::DISCONNECTED;
            self.set_asr(asr::INT);
            self.target_id = 0;
            self.target_lun = 0;
            self.pending_status = 0;
            self.pending_msg = 0;
            self.advanced_mode = false;
            self.pending_command = None;
            self.xfer_data.clear();
            self.xfer_offset = 0;

            // Registers 0x01 through 0x16 are reset to zero.
            for i in 0x01..=0x16 {
                self.regs[i] = 0;
            }
            // The Command register (0x18) is also reset to zero.
            self.regs[regs::COMMAND as usize] = 0;

            // The SCSI Status register is set as commanded by the EAF bit in the Own ID register.
            let own_id = self.regs[regs::OWN_ID as usize];
            let eaf = (own_id & 0x08) != 0;
            self.advanced_mode = eaf;
            self.regs[regs::SCSI_STATUS as usize] = if eaf { scsi_status::RESET_EAF } else { scsi_status::RESET };
            return;
        }

        // SEL_ATN_XFER with cmd_phase=0x46 and no pending xfer: NetBSD/OpenBSD wd33c93_xferdone
        // conclude path. Check before the xfer_data resume so stale xfer_data doesn't intercept.
        // (IRIX mid-transfer resume also has cmd_phase=0x46 but xfer_data is non-empty then.)
        if (cmd == cmd::SELECT_ATN_XFER || cmd == cmd::SELECT_XFER)
            && self.regs[regs::COMMAND_PHASE as usize] == command_phase::TRANSFER_COUNT
            && self.xfer_data.is_empty()
        {
            dlog!(self.log_module(), "WD33C93A({}): SELECT_ATN_XFER conclude (PH_DATA=0x46)", self.id);
            wdt!("CONCLUDE tgt={} status={:02x} pending_data={} xfer_data={} phase={:02x}", self.target_id, self.pending_status, self.pending_data.len(), self.xfer_data.len(), self.regs[regs::COMMAND_PHASE as usize]);
            self.regs[regs::TARGET_LUN as usize] = self.pending_status;
            self.queue_interrupt(Some(command_phase::COMPLETE_MSG), scsi_status::SELECT_TRANSFER_SUCCESS);
            return;
        }

        // Resume mid-transfer if SELECT_ATN_XFER arrives while a chunked transfer is paused
        if cmd == cmd::SELECT_ATN_XFER && !self.xfer_data.is_empty() {
            dlog!(self.log_module(), "WD33C93A({}): SELECT_ATN_XFER resume: dir_in={} offset=0x{:x}/0x{:x}", self.id,
                self.xfer_direction_in, self.xfer_offset, self.xfer_data.len());
            if self.xfer_direction_in {
                // Resuming a send (READ cmd): continue from xfer_offset
                let data = std::mem::take(&mut self.xfer_data);
                let offset = self.xfer_offset;
                self.xfer_offset = 0;
                if !self.send_data_chunked(data, offset, dma) {
                    return; // paused again at next chunk boundary
                }
            } else {
                // Resuming a receive (WRITE cmd): xfer_data holds bytes received so far,
                // xfer_offset holds the total expected length
                let total = self.xfer_offset;
                let partial = std::mem::take(&mut self.xfer_data);
                self.xfer_offset = 0;
                match self.receive_data_chunked_from(total, partial, dma) {
                    None => return, // paused again
                    Some(full_data) => {
                        // Build CDB from registers before borrowing device
                        let opcode = self.regs[regs::CDB_1 as usize];
                        let cdb_len = self.get_cdb_length(opcode);
                        let mut cdb = Vec::with_capacity(cdb_len);
                        for i in 0..cdb_len {
                            cdb.push(self.regs[(regs::CDB_1 as usize) + i]);
                        }
                        if cdb.len() > 1 && (cdb[1] >> 5) & 0x7 == 0 && self.target_lun != 0 {
                            cdb[1] |= self.target_lun << 5;
                        }
                        let request = ScsiRequest {
                            cdb,
                            data_len: ScsiDataLength::Unlimited,
                            data_in: Some(full_data),
                        };
                        let tid = self.target_id;
                        match self.devices[tid].as_mut().map(|d| d.request(&request)) {
                            Some(Ok(response)) => self.finish_command(response.status),
                            _ => self.finish_command(0x02),
                        }
                    }
                }
            }
            // Transfer complete — raise final completion interrupt
            wdt!("CONCLUDE tgt={} status={:02x} (resume path)", self.target_id, self.pending_status);
            self.regs[regs::TARGET_LUN as usize] = self.pending_status;
            self.queue_interrupt(Some(command_phase::COMPLETE_MSG), scsi_status::SELECT_TRANSFER_SUCCESS);
            return;
        }

        if self.devices[self.target_id].is_none() {
            dlog!(self.log_module(), "WD33C93A({}): No device at target {}, timing out", self.id, self.target_id);
            wdt!("SEL_TIMEO tgt={} cmd={:02x} asr={:02x}", self.target_id, cmd, self.asr);
            // First interrupt (0x42): consumed by selectbus's SBIC_WAIT(INT) poll.
            // Second interrupt (0x41 DISC): consumed by wd33c93_poll's wd33c93_loop
            // call, which handles DISC via nextstate → scsidone → ITSDONE set.
            self.queue_interrupt(Some(command_phase::DISCONNECTED), scsi_status::SELECTION_TIMEOUT);
            return;
        }

        match cmd {
            cmd::ABORT => {
                self.fifo.clear();
                self.irq_fifo.clear();
                self.clear_interrupt();
                self.xfer_data.clear();
                self.xfer_offset = 0;
                self.update_asr(asr::CIP | asr::BSY, 0);
                let status = if self.advanced_mode { scsi_status::RESET_EAF } else { scsi_status::RESET };
                self.queue_interrupt(Some(command_phase::DISCONNECTED), status);
            }
            cmd::ASSERT_ATN => {
                self.update_asr(asr::CIP, 0);
            }
            cmd::DISCONNECT => {
                self.queue_interrupt(Some(command_phase::DISCONNECTED), scsi_status::DISCONNECT);
                self.update_asr(asr::BSY, 0);
            }
            cmd::SELECT_ATN | cmd::SELECT_ATN_XFER | cmd::SELECT | cmd::SELECT_XFER => {
                let status = self.regs[regs::SCSI_STATUS as usize];
                self.set_status(command_phase::SELECTED, status);

                if cmd == cmd::SELECT_ATN_XFER || cmd == cmd::SELECT_XFER {
                    dlog!(self.log_module(), "WD33C93A({}): SELECT_XFER/SELECT_ATN_XFER", self.id);
                    // CDB is stored in registers starting at CDB_1 (0x03)
                    let opcode = self.regs[regs::CDB_1 as usize];
                    let len = self.get_cdb_length(opcode);
                    let mut cdb = Vec::with_capacity(len);
                    for i in 0..len {
                        cdb.push(self.regs[(regs::CDB_1 as usize) + i]);
                    }
                    self.process_scsi_command(&cdb, true, dma);
                } else {
                    // SELECT_ATN fires two interrupts:
                    //   1st: SELECT_SUCCESS (0x11) — selection complete
                    //   2nd: REQ_SEND_MSG_OUT (0x8E) — bus requesting MESG_OUT
                    // SELECT (no ATN) goes straight to CMD phase in one interrupt.
                    if cmd == cmd::SELECT_ATN {
                        dlog!(self.log_module(), "WD33C93A({}): SELECT_ATN → 0x11 then 0x8E", self.id);
                        wdt!("SEL_ATN tgt={}", self.target_id);
                        self.queue_interrupt(Some(command_phase::SELECTED), scsi_status::SELECT_SUCCESS);
                        self.queue_interrupt(None, scsi_status::REQ_SEND_MSG_OUT);
                    } else {
                        dlog!(self.log_module(), "WD33C93A({}): SELECT → 0x8A (CMD phase)", self.id);
                        self.queue_interrupt(Some(command_phase::SELECTED), scsi_status::REQ_CMD_PHASE);
                    }
                }
            }
            cmd::TRANSFER_INFO => {
                let scsi_st = self.regs[regs::SCSI_STATUS as usize];
                let cmd_phase = self.regs[regs::COMMAND_PHASE as usize];
                dlog!(self.log_module(), "WD33C93A({}): TRANSFER_INFO scsi_status={:02x} cmd_phase={:02x}", self.id,
                    scsi_st, cmd_phase);

                match cmd_phase {
                    command_phase::SELECTED => {
                        // MESG_OUT: receive IDENTIFY (+SDTR) bytes then fire 0x8a → CMD phase.
                        let count = if self.use_dma() {
                            let tc = self.get_transfer_count();
                            (if tc == 0 { self.set_transfer_count(1); 1 } else { tc }) as usize
                        } else {
                            self.fifo.len()
                        };
                        dlog!(self.log_module(), "WD33C93A({}): XFER_INFO MESG_OUT count={} dma={}", self.id, count, dma.is_some());
                        let _msg = self.receive_data(count, dma);
                        self.queue_interrupt(Some(command_phase::IDENTIFY_SENT), scsi_status::REQ_CMD_PHASE);
                    }
                    command_phase::IDENTIFY_SENT | command_phase::COMMAND_START => {
                        // IDENTIFY_SENT (0x20): MESG_OUT done, now in CMD phase.
                        // COMMAND_START (0x30): re-issued TRANSFER_INFO for write CDB.
                        // In both cases: read CDB bytes via receive_data() (DMA or PIO fifo),
                        // store into CDB registers, then execute.
                        // Write commands raise TRANSFER_DATA_OUT first; re-issue lands here
                        // with TC=0 and executes from registers (no receive_data needed).
                        // PIO: TC was decremented to 0 by DATA writes; use fifo.len().
                        // DMA: TC holds the byte count set by driver; use it.
                        let count = if self.use_dma() {
                            let tc = self.get_transfer_count();
                            (if tc == 0 { self.set_transfer_count(6); 6 } else { tc }) as usize
                        } else {
                            self.fifo.len()
                        };
                        self.regs[regs::COMMAND_PHASE as usize] = command_phase::COMMAND_START;
                        self.xfer_data.clear();
                        self.xfer_offset = 0;
                        let cdb_bytes = self.receive_data(count, dma);
                        let opcode = cdb_bytes.first().copied().unwrap_or(0);
                        dlog!(self.log_module(), "WD33C93A({}): CMD phase CDB 0x{:02x} count={} dma={}", self.id, opcode, count, dma.is_some());
                        let is_write = matches!(opcode,
                            scsi_cmd::WRITE_6 | scsi_cmd::WRITE_10 | scsi_cmd::WRITE_BUFFER |
                            scsi_cmd::MODE_SELECT_6 | scsi_cmd::FORMAT_UNIT | scsi_cmd::SEND_DIAGNOSTIC);
                        if is_write && count > 0 {
                            // First pass: driver needs to arm DMA/PIO for write data.
                            self.queue_interrupt(Some(command_phase::COMMAND_START), scsi_status::TRANSFER_DATA_OUT);
                        } else {
                            self.process_scsi_command(&cdb_bytes, false, dma);
                        }
                    }
                    command_phase::TRANSFER_COUNT => {
                        if !self.pending_data.is_empty() {
                            let data = std::mem::take(&mut self.pending_data);
                            wdt!("CONSUME tgt={} pending_data=0x{:x} bytes CTRL={:02x} use_dma={} phase={:02x}", self.target_id, data.len(), self.regs[regs::CONTROL as usize], self.use_dma(), self.regs[regs::COMMAND_PHASE as usize]);
                            //eprintln!("WD33C93A: TRANSFER_COUNT 0x{:x} bytes CTRL={:02x} use_dma={} dma={}",
                            //    data.len(), self.regs[regs::CONTROL as usize], self.use_dma(), dma.is_some());
                            if self.use_dma() {
                                // DMA path: CONTROL register now valid, driver has set up DMA.
                                dlog!(self.log_module(), "WD33C93A({}): XFER_INFO DATA_IN DMA, {} bytes", self.id, data.len());
                                if !self.send_data_chunked(data, 0, dma) {
                                    eprintln!("WD33C93A: send_data_chunked paused");
                                    return; // paused mid-chunk, interrupt already raised
                                }
                                self.regs[regs::TARGET_LUN as usize] = self.pending_status;
                                self.queue_interrupt_ex(Some(command_phase::RECEIVE_STATUS), scsi_status::TRANSFER_STATUS_IN, true);
                            } else {
                                // PIO path: load fifo, arm DBR for byte-by-byte delivery.
                                dlog!(self.log_module(), "WD33C93A({}): XFER_INFO DATA_IN PIO, {} bytes, arming DBR", self.id, data.len());
                                self.fifo.extend(data);
                                self.update_asr(asr::CIP | asr::INT, asr::DBR);
                            }
                        } else {
                            // No data (or fifo already drained): target now asserting STATUS phase.
                            dlog!(self.log_module(), "WD33C93A({}): XFER_INFO → STATUS phase", self.id);
                            self.regs[regs::TARGET_LUN as usize] = self.pending_status;
                            self.queue_interrupt_ex(Some(command_phase::RECEIVE_STATUS), scsi_status::TRANSFER_STATUS_IN, true);
                        }
                    }
                    command_phase::RECEIVE_STATUS => {
                        // Driver issued TRANSFER_INFO to read 1 status byte via DBR.
                        // Push status byte to fifo, arm DBR. DATA read handler fires TRANSFER_MSG_IN when drained.
                        self.regs[regs::TARGET_LUN as usize] = self.pending_status;
                        self.set_transfer_count(1);
                        self.fifo.push_back(self.pending_status);
                        dlog!(self.log_module(), "WD33C93A({}): XFER_INFO RECEIVE_STATUS → DBR status={:02x}", self.id, self.pending_status);
                        self.update_asr(asr::CIP | asr::INT, asr::DBR);
                    }
                    command_phase::STATUS_RECEIVED => {
                        // Driver issued TRANSFER_INFO to read 1 msg byte via DBR.
                        // Push msg byte to fifo, arm DBR. DATA read handler fires DISCONNECT when drained.
                        self.set_transfer_count(1);
                        self.fifo.push_back(self.pending_msg);
                        dlog!(self.log_module(), "WD33C93A({}): XFER_INFO STATUS_RECEIVED → DBR msg={:02x}", self.id, self.pending_msg);
                        self.update_asr(asr::CIP | asr::INT, asr::DBR);
                    }
                    _ => {
                        dlog!(self.log_module(), "WD33C93A({}): TRANSFER_INFO in unexpected state scsi_st={:02x} cmd_phase={:02x}", self.id,
                            scsi_st, cmd_phase);
                        self.update_asr(asr::CIP, 0);
                    }
                }
            }
            cmd::NEGATE_ACK => {
                // CLR_ACK after message-in byte(s): produce disconnect interrupt
                // so the driver's SBIC_WAIT loop gets the next CSR.
                let cmd_phase = self.regs[regs::COMMAND_PHASE as usize];
                if cmd_phase == command_phase::COMPLETE_MSG {
                    self.queue_interrupt(Some(command_phase::DISCONNECTED), scsi_status::DISCONNECT);
                } else {
                    self.update_asr(asr::CIP, 0);
                }
            }
            _ => {
                dlog!(self.log_module(), "WD33C93A({}): Unimplemented WD command {:02x}", self.id, cmd);
                self.update_asr(asr::CIP, 0);
            }
        }
    }


    fn get_cdb_length(&self, opcode: u8) -> usize {
        scsi::get_cdb_length(opcode)
    }

    fn process_scsi_command(&mut self, cdb: &[u8], auto_mode: bool, dma: Option<&dyn DmaClient>) {
        // SCSI-2 initiators (Linux's wd33c93 driver included) address the LUN via the
        // IDENTIFY message — i.e. TARGET_LUN — and leave the legacy CDB LUN field
        // (cdb[1] bits 5-7) at 0. Fold target_lun in here so scsi.rs, which only looks
        // at cdb[1], sees the real LUN. If firmware/PROM ever does put a nonzero LUN
        // directly in the CDB, that takes precedence.
        let mut cdb = cdb.to_vec();
        if !cdb.is_empty() && (cdb[1] >> 5) & 0x7 == 0 && self.target_lun != 0 {
            cdb[1] |= self.target_lun << 5;
        }
        let cdb = cdb.as_slice();
        if cdb.is_empty() {
            dlog!(self.log_module(), "WD33C93A({}): Empty CDB!", self.id);
            self.update_asr(0, asr::LCI);
            self.queue_interrupt(Some(command_phase::DISCONNECTED), scsi_status::INVALID_COMMAND);
            return;
        }

        {
            let cmd_name = match cdb[0] {
                scsi_cmd::TEST_UNIT_READY => "TEST_UNIT_READY",
                scsi_cmd::REQUEST_SENSE => "REQUEST_SENSE",
                scsi_cmd::FORMAT_UNIT => "FORMAT_UNIT",
                scsi_cmd::READ_6 => "READ_6",
                scsi_cmd::WRITE_6 => "WRITE_6",
                scsi_cmd::INQUIRY => "INQUIRY",
                scsi_cmd::MODE_SELECT_6 => "MODE_SELECT_6",
                scsi_cmd::MODE_SENSE_6 => "MODE_SENSE_6",
                scsi_cmd::MODE_SENSE_10 => "MODE_SENSE_10",
                scsi_cmd::START_STOP_UNIT => "START_STOP_UNIT",
                scsi_cmd::RECEIVE_DIAGNOSTIC_RESULTS => "RECEIVE_DIAGNOSTIC_RESULTS",
                scsi_cmd::SEND_DIAGNOSTIC => "SEND_DIAGNOSTIC",
                scsi_cmd::PREVENT_ALLOW_MEDIUM_REMOVAL => "PREVENT_ALLOW_MEDIUM_REMOVAL",
                scsi_cmd::READ_CAPACITY_10 => "READ_CAPACITY_10",
                scsi_cmd::READ_10 => "READ_10",
                scsi_cmd::WRITE_10 => "WRITE_10",
                scsi_cmd::VERIFY_10 => "VERIFY_10",
                scsi_cmd::SYNCHRONIZE_CACHE_10 => "SYNCHRONIZE_CACHE_10",
                scsi_cmd::WRITE_BUFFER => "WRITE_BUFFER",
                scsi_cmd::READ_BUFFER => "READ_BUFFER",
                scsi_cmd::READ_SUB_CHANNEL => "READ_SUB_CHANNEL",
                scsi_cmd::READ_TOC_PMA_ATIP => "READ_TOC_PMA_ATIP",
                scsi_cmd::PLAY_AUDIO_TRACK_INDEX => "PLAY_AUDIO_TRACK_INDEX",
                scsi_cmd::PAUSE_RESUME => "PAUSE_RESUME",
                scsi_cmd::READ_DISC_INFORMATION => "READ_DISC_INFORMATION",
                scsi_cmd::SGI_EJECT => "SGI_EJECT",
                scsi_cmd::SGI_HD2CDROM => "SGI_HD2CDROM",
                _ => "UNKNOWN",
            };

            let is_dayna = self.devices[self.target_id].as_ref()
                .map(|d| d.is_daynaport()).unwrap_or(false);
            let mut extra = String::new();
            match cdb[0] {
                // On a DaynaPort these two are packet RX/TX, not block I/O.
                scsi_cmd::READ_6 | scsi_cmd::WRITE_6 if is_dayna => {
                    let len = ((cdb[3] as usize) << 8) | cdb[4] as usize;
                    extra = format!(" [DaynaPort] Bytes=0x{:x} flags={:02x}", len, cdb[5]);
                }
                scsi_cmd::READ_6 | scsi_cmd::WRITE_6 => {
                    let lba = (((cdb[1] & 0x1F) as u64) << 16) | ((cdb[2] as u64) << 8) | (cdb[3] as u64);
                    let count = if cdb[4] == 0 { 256 } else { cdb[4] as usize };
                    extra = format!(" LBA=0x{:x} Blocks=0x{:x} Bytes=0x{:x}", lba, count, count * 512);
                }
                scsi_cmd::READ_10 | scsi_cmd::WRITE_10 => {
                    let lba = ((cdb[2] as u64) << 24) | ((cdb[3] as u64) << 16) | ((cdb[4] as u64) << 8) | (cdb[5] as u64);
                    let count = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                    extra = format!(" LBA=0x{:x} Blocks=0x{:x} Bytes=0x{:x}", lba, count, count * 512);
                }
                scsi_cmd::INQUIRY | scsi_cmd::REQUEST_SENSE | scsi_cmd::MODE_SENSE_6 => {
                    let len = cdb[4] as usize;
                    extra = format!(" Bytes=0x{:x}", len);
                }
                scsi_cmd::MODE_SENSE_10 => {
                    let len = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                    extra = format!(" Bytes=0x{:x}", len);
                }
                _ => {}
            }
            dlog!(self.log_module(), "WD33C93A({}): SCSI Command {:02x} ({}) Target {}{}", self.id, cdb[0], cmd_name, self.target_id, extra);
        }

        // Determine data_len based on command
        let data_len = match cdb[0] {
            scsi_cmd::INQUIRY | scsi_cmd::REQUEST_SENSE | scsi_cmd::MODE_SENSE_6 => {
                ScsiDataLength::Fixed(cdb[4] as usize)
            }
            scsi_cmd::READ_BUFFER => {
                // READ_BUFFER allocation length is in bytes 6-8 (24-bit)
                let len = ((cdb[6] as usize) << 16) | ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                ScsiDataLength::Fixed(len)
            }
            scsi_cmd::READ_TOC_PMA_ATIP | scsi_cmd::GET_CONFIGURATION | scsi_cmd::MODE_SENSE_10 => {
                // Allocation length in bytes 7-8
                let len = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                ScsiDataLength::Fixed(len)
            }
            _ => ScsiDataLength::Unlimited,
        };

        // A DaynaPort reuses WRITE(6) to transmit one Ethernet frame, so its
        // data-out length is a plain byte count in CDB 3..4 — not `blocks × 512`.
        let dayna = self.devices[self.target_id].as_ref()
            .map(|d| d.is_daynaport()).unwrap_or(false);

        // For WRITE commands, receive data first (data out from host to target)
        let data_in = match cdb[0] {
            scsi_cmd::WRITE_6 if dayna => {
                self.data_direction_in = false;
                let len = ((cdb[3] as usize) << 8) | cdb[4] as usize;
                if len > 0 {
                    match self.receive_data_chunked(len, dma) {
                        None => return, // paused; will resume on SELECT_ATN_XFER
                        data => data,
                    }
                } else {
                    None
                }
            }
            scsi_cmd::WRITE_6 => {
                self.data_direction_in = false;
                let count = if cdb[4] == 0 { 256 } else { cdb[4] as usize };
                match self.receive_data_chunked(count * 512, dma) {
                    None => return, // paused; will resume on SELECT_ATN_XFER
                    data => data,
                }
            }
            scsi_cmd::WRITE_10 => {
                self.data_direction_in = false;
                let count = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                match self.receive_data_chunked(count * 512, dma) {
                    None => return,
                    data => data,
                }
            }
            scsi_cmd::WRITE_BUFFER => {
                self.data_direction_in = false;
                let len = ((cdb[6] as usize) << 16) | ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                if len > 0 {
                    match self.receive_data_chunked(len, dma) {
                        None => return,
                        data => data,
                    }
                } else {
                    None
                }
            }
            scsi_cmd::SEND_DIAGNOSTIC => {
                self.data_direction_in = false;
                let len = ((cdb[3] as usize) << 8) | (cdb[4] as usize);
                if len > 0 {
                    match self.receive_data_chunked(len, dma) {
                        None => return,
                        data => data,
                    }
                } else {
                    None
                }
            }
            scsi_cmd::MODE_SELECT_6 => {
                self.data_direction_in = false;
                let len = cdb[4] as usize;
                if len > 0 {
                    match self.receive_data_chunked(len, dma) {
                        None => return,
                        data => data,
                    }
                } else {
                    None
                }
            }
            _ => {
                self.data_direction_in = true;
                None
            }
        };

        // Make SCSI request
        let device = &mut self.devices[self.target_id];
        if device.is_none() {
            self.finish_command(0x02);
            return;
        }

        let request = ScsiRequest {
            cdb: cdb.to_vec(),
            data_len,
            data_in,
        };

        match device.as_mut().unwrap().request(&request) {
            Ok(response) => {
                if !response.data.is_empty() {
                    if auto_mode {
                        // IRIX/PROM: SELECT_ATN_XFER with DMA pre-armed. Push data directly.
                        if self.use_dma() {
                            if !self.send_data_chunked(response.data, 0, dma) {
                                return; // paused mid-chunk; interrupt already raised
                            }
                        } else {
                            self.send_data(&response.data, dma);
                        }
                        self.finish_command(response.status);
                    } else {
                        // NetBSD/OpenBSD manual mode: driver sets CONTROL/DMA *after* seeing
                        // TRANSFER_DATA_IN. Stage data and let TRANSFER_INFO/TRANSFER_COUNT deliver it.
                        self.pending_status = response.status;
                        self.pending_msg = 0x00;
                        if !self.pending_data.is_empty() {
                            eprintln!("WD33C93A: WARNING pending_data not empty ({} bytes) when staging new response!", self.pending_data.len());
                        }
                        self.pending_data = response.data;
                        self.set_transfer_count(self.pending_data.len() as u32);
                        dlog!(self.log_module(), "WD33C93A({}): DATA_IN 0x{:x} bytes staged, raising TRANSFER_DATA_IN", self.id, self.pending_data.len());
                        wdt!("STAGE tgt={} pending_data=0x{:x} bytes tc=0x{:x}", self.target_id, self.pending_data.len(), self.get_transfer_count());
                        self.queue_interrupt(Some(command_phase::TRANSFER_COUNT), scsi_status::TRANSFER_DATA_IN);
                        return;
                    }
                } else {
                    self.finish_command(response.status);
                }
            }
            Err(_) => {
                self.finish_command(0x02); // Check Condition
            }
        }

        if auto_mode {
            // IRIX/PROM: auto mode expects S_XFERRED (0x16) directly after completion.
            wdt!("CONCLUDE tgt={} status={:02x} (auto) phase={:02x}", self.target_id, self.pending_status, self.regs[regs::COMMAND_PHASE as usize]);
            self.regs[regs::TARGET_LUN as usize] = self.pending_status;
            self.queue_interrupt(Some(command_phase::COMPLETE_MSG), scsi_status::SELECT_TRANSFER_SUCCESS);
        } else {
            // NetBSD/OpenBSD manual mode: signal STATUS phase; driver's nextstate STATUS case
            // calls xferdone() which then issues SELECT_ATN_XFER(cmd_phase=0x46) → our conclude.
            wdt!("STATUS_IN tgt={} status={:02x} phase={:02x}", self.target_id, self.pending_status, self.regs[regs::COMMAND_PHASE as usize]);
            self.regs[regs::TARGET_LUN as usize] = self.pending_status;
            // deferred=true: worker clears CIP and spins 10000 cycles before asserting INT,
            // giving wd33c93_loop time to exit before STATUS_IN fires.
            self.queue_interrupt_ex(Some(command_phase::RECEIVE_STATUS), scsi_status::TRANSFER_STATUS_IN, true);
        }
    }

    fn finish_command(&mut self, status: u8) {
        self.pending_status = status;
        self.pending_msg = 0x00; // Command Complete

        // Zero out Transfer Count registers to simulate completion
        self.regs[regs::TRANSFER_COUNT_MSB as usize] = 0;
        self.regs[regs::TRANSFER_COUNT_2ND as usize] = 0;
        self.regs[regs::TRANSFER_COUNT_LSB as usize] = 0;
    }

    /// Push `data[offset..]` to the host via DMA, pausing on chunk boundaries (XIE IRQ).
    /// Returns true if the full transfer completed, false if paused mid-transfer.
    /// On pause: stores remaining data in `self.xfer_data`/`self.xfer_offset` and raises
    /// UNEXPECTED_RECV_DATA interrupt so IRIX's unex_info() can re-arm for the next chunk.
    fn send_data_chunked(&mut self, data: Vec<u8>, offset: usize, dma: Option<&dyn DmaClient>) -> bool {
        dlog!(self.log_module(), "WD33C93A({}): Sending 0x{:x} bytes via DMA (offset=0x{:x})", self.id, data.len() - offset, offset);
        wdt!("DMA_OUT start: 0x{:x} bytes (offset=0x{:x})", data.len() - offset, offset);
        if let Some(dma_dev) = dma {
            let total = data.len();
            let last_idx = total.saturating_sub(1);
            let mut i = offset;
            while i < total {
                let is_last = i == last_idx;
                let (mut st, _) = dma_dev.write(data[i] as u32, is_last);
                // On first byte, if channel not yet active (driver calls wdsc_dmago after
                // issuing TRANSFER_INFO), spin up to 1ms for it to become ready.
                if i == 0 && st.not_active() {
                    // Driver calls wdsc_dmago *after* writing TRANSFER_INFO, so the channel
                    // may not be active yet. Spin up to 100ms to let the CPU thread arm it.
                    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
                    while st.not_active() && std::time::Instant::now() < deadline {
                        std::thread::yield_now();
                        (st, _) = dma_dev.write(data[i] as u32, is_last);
                    }
                    if st.not_active() {
                        dlog!(self.log_module(), "WD33C93A({}): DMA channel still not active after 1ms — pausing", self.id);
                        //eprintln!("WD33C93A: DMA channel still not active after 1ms — pausing");
                    }
                }
                // On refused: byte was not accepted, do not advance or decrement.
                // On eox/irq: byte was accepted, advance and decrement.
                if !st.refused() {
                    i += 1;
                    self.decrement_transfer_count();
                }
                // EOX mid-transfer: chain exhausted early (device sent less than allocated).
                // Pause so IRIX can re-arm via SELECT_ATN_XFER for the next chunk.
                // XIE (irq without eox): descriptor boundary, chain continues — keep writing.
                let pause = (st.eox() || st.refused()) && !is_last;
                if pause {
                    //eprintln!("WD33C93A: send_data_chunked pause: EOX={} XIE={} refused={} offset=0x{:x} remaining=0x{:x}",
                    //    st.eox(), st.irq(), st.refused(), i, total - i);
                    dlog!(self.log_module(), "WD33C93A({}): EOX={} XIE={} refused={} at offset=0x{:x}, remaining=0x{:x} — pausing", self.id,
                        st.eox(), st.irq(), st.refused(), i, total - i);
                    self.xfer_data = data;
                    self.xfer_offset = i;
                    self.xfer_direction_in = true;
                    wdt!("DMA_OUT pause: EOX={} refused={} at offset=0x{:x}", st.eox(), st.refused(), i);
                    self.queue_interrupt(Some(command_phase::TRANSFER_COUNT), scsi_status::UNEXPECTED_SEND_DATA);
                    return false;
                }
            }
        }
        wdt!("DMA_OUT done");
        true
    }

    /// Receive `len` bytes from host via DMA, pausing on chunk boundaries (XIE IRQ).
    /// Returns Some(data) when fully received, None if paused mid-transfer.
    /// On pause: stores received-so-far in `self.xfer_data`/`self.xfer_offset` and raises
    /// UNEXPECTED_SEND_DATA interrupt so IRIX's unex_info() can re-arm for the next chunk.
    fn receive_data_chunked(&mut self, len: usize, dma: Option<&dyn DmaClient>) -> Option<Vec<u8>> {
        self.receive_data_chunked_from(len, Vec::new(), dma)
    }

    fn receive_data_chunked_from(&mut self, total: usize, mut data: Vec<u8>, dma: Option<&dyn DmaClient>) -> Option<Vec<u8>> {
        dlog!(self.log_module(), "WD33C93A({}): Receiving 0x{:x} bytes via DMA (have=0x{:x})", self.id, total - data.len(), data.len());
        wdt!("DMA_IN start: 0x{:x} bytes (have=0x{:x})", total - data.len(), data.len());
        if let Some(dma_dev) = dma {
            while data.len() < total {
                match dma_dev.read() {
                    Some((val, st, _)) => {
                        // Byte accepted — decrement transfer count register to mirror real HW.
                        data.push(val as u8);
                        self.decrement_transfer_count();
                        // XIE without EOX: descriptor boundary, chain continues — keep reading.
                        // EOX mid-transfer: chain exhausted before all bytes received — pause for IRIX resume.
                        let pause = st.eox() && data.len() < total;
                        if pause {
                            dlog!(self.log_module(), "WD33C93A({}): EOX at offset=0x{:x}, remaining=0x{:x} — pausing", self.id, data.len(), total - data.len());
                            wdt!("DMA_IN pause(EOX): at offset=0x{:x} remaining=0x{:x}", data.len(), total - data.len());
                            self.xfer_data = data;
                            self.xfer_offset = total; // store total as sentinel; xfer_data.len() is progress
                            self.xfer_direction_in = false;
                            self.queue_interrupt(Some(command_phase::TRANSFER_COUNT), scsi_status::UNEXPECTED_RECV_DATA);
                            return None;
                        }
                    }
                    None => {
                        let remaining = total - data.len();
                        if remaining > 0 {
                            if data.is_empty() {
                                // Channel not yet active — driver issues TRANSFER_INFO before dmago
                                // (NetBSD: SET_SBIC_cmd then sc_dmago on same CPU thread).
                                // Spin up to 100ms for the CPU thread to call dmago.
                                wdt!("DMA_IN spin-wait: channel not yet active");
                                let deadline = std::time::Instant::now() + std::time::Duration::from_millis(100);
                                let mut got = false;
                                while std::time::Instant::now() < deadline {
                                    std::thread::yield_now();
                                    if let Some((val, st, _)) = dma_dev.read() {
                                        data.push(val as u8);
                                        self.decrement_transfer_count();
                                        let pause = st.eox() && data.len() < total;
                                        if pause {
                                            wdt!("DMA_IN pause(EOX after spin): offset=0x{:x}", data.len());
                                            self.xfer_data = data;
                                            self.xfer_offset = total;
                                            self.xfer_direction_in = false;
                                            self.queue_interrupt(Some(command_phase::TRANSFER_COUNT), scsi_status::UNEXPECTED_RECV_DATA);
                                            return None;
                                        }
                                        got = true;
                                        break;
                                    }
                                }
                                if !got {
                                    dlog!(self.log_module(), "WD33C93A({}): DMA channel still not active after 100ms — giving up", self.id);
                                    //eprintln!("WD33C93A: receive_data_chunked: DMA not active after 100ms");
                                    wdt!("DMA_IN spin-wait TIMEOUT: channel never became active");
                                    break;
                                }
                                // Successfully got first byte — continue outer loop
                            } else {
                                // Mid-transfer: chain exhausted early — pause for IRIX resume
                                dlog!(self.log_module(), "WD33C93A({}): EOX at offset=0x{:x}, remaining=0x{:x} — pausing", self.id, data.len(), remaining);
                                wdt!("DMA_IN pause(inactive mid-xfer): offset=0x{:x} remaining=0x{:x}", data.len(), remaining);
                                self.xfer_data = data;
                                self.xfer_offset = total;
                                self.xfer_direction_in = false;
                                self.queue_interrupt(Some(command_phase::TRANSFER_COUNT), scsi_status::UNEXPECTED_RECV_DATA);
                                return None;
                            }
                        } else {
                            break;
                        }
                    }
                }
            }
        } else {
            while data.len() < total {
                data.push(self.fifo.pop_front().unwrap_or(0));
            }
        }
        wdt!("DMA_IN done: 0x{:x} bytes", total);
        Some(data)
    }

    /// Push `data` to the host via DMA or PIO (FIFO), whichever is active.
    fn send_data(&mut self, data: &[u8], dma: Option<&dyn DmaClient>) {
        if !data.is_empty() {
            if self.use_dma() {
                dlog!(self.log_module(), "WD33C93A({}): Sending 0x{:x} bytes via DMA", self.id, data.len());
            } else {
                dlog!(self.log_module(), "WD33C93A({}): Pushing 0x{:x} bytes to FIFO", self.id, data.len());
            }
        }
        if self.use_dma() {
            if let Some(dma_dev) = dma {
                let last = data.len().saturating_sub(1);
                for (i, &b) in data.iter().enumerate() {
                    let _ = dma_dev.write(b as u32, i == last);
                }
            }
        } else {
            for &b in data {
                self.fifo.push_back(b);
            }
        }
    }

    /// Receive `data` from the host via DMA or PIO (FIFO), whichever is active.
    fn receive_data(&mut self, len: usize, dma: Option<&dyn DmaClient>) -> Vec<u8> {
        let mut data = vec![0u8; len];
        if self.use_dma() {
            if let Some(dma_dev) = dma {
                for i in 0..len {
                    if let Some((val, _, _)) = dma_dev.read() {
                        data[i] = val as u8;
                    }
                }
            }
        } else {
            for i in 0..len {
                data[i] = self.fifo.pop_front().unwrap_or(0);
            }
        }
        data
    }

    fn get_transfer_count(&self) -> u32 {
        let lo  = self.regs[regs::TRANSFER_COUNT_LSB as usize] as u32;
        let mid = self.regs[regs::TRANSFER_COUNT_2ND as usize] as u32;
        let hi  = self.regs[regs::TRANSFER_COUNT_MSB as usize] as u32;
        (hi << 16) | (mid << 8) | lo
    }

    fn set_transfer_count(&mut self, count: u32) {
        self.regs[regs::TRANSFER_COUNT_MSB as usize] = ((count >> 16) & 0xFF) as u8;
        self.regs[regs::TRANSFER_COUNT_2ND as usize] = ((count >> 8)  & 0xFF) as u8;
        self.regs[regs::TRANSFER_COUNT_LSB as usize] = (count         & 0xFF) as u8;
    }

    /// Decrement the 24-bit transfer count register by 1, mirroring real WD33C93A hardware
    /// which decrements on every byte transferred. save_datap() reads this to compute
    /// count_xferd = wd_xferlen - count_remain.
    fn decrement_transfer_count(&mut self) {
        let count = self.get_transfer_count().saturating_sub(1);
        self.set_transfer_count(count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scsi() -> Wd33c93a {
        Wd33c93a::new(None, None, Arc::new(AtomicU64::new(0)))
    }

    /// Phase 1.7 round-trip: a fresh SCSI controller loaded from a captured
    /// save_state must re-serialize byte-identically. Mutates regs and the
    /// scalar shadow fields (ar, asr, target_id, pending_*).
    #[test]
    fn save_load_round_trip() {
        let src = make_scsi();
        {
            let mut s = src.state.lock();
            s.regs[regs::CONTROL as usize]      = 0x60;
            s.regs[regs::SCSI_STATUS as usize]  = 0x10;
            s.regs[regs::COMMAND_PHASE as usize] = 0x46;
            s.regs[regs::OWN_ID as usize]       = 0x07;
            s.ar = 0x42;
            s.set_asr(0x10);
            s.data_direction_in = true;
            s.target_id = 4;
            s.pending_status = 0x02;
            s.pending_msg = 0x80;
            s.advanced_mode = true;
        }
        let v1 = src.save_state();

        let dst = make_scsi();
        dst.load_state(&v1).expect("load_state");
        let v2 = dst.save_state();

        assert_eq!(v1, v2, "Wd33c93a save_state mismatch after load_state round-trip");
    }
}