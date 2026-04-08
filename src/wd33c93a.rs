use std::fs::OpenOptions;
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
    pub const RESET: u8 = 0x00;
    pub const RESET_EAF: u8 = 0x01;

    pub const RESELECT_SUCCESS: u8 = 0x10;
    pub const SELECT_SUCCESS: u8 = 0x11;  // ST_SELECT
    pub const COMMAND_SUCCESS: u8 = 0x13;
    pub const COMMAND_ATN_SUCCESS: u8 = 0x14;
    pub const TRANSLATE_SUCCESS: u8 = 0x15;
    pub const SELECT_TRANSFER_SUCCESS: u8 = 0x16;  // ST_SATOK
    pub const TRANSFER_DATA_OUT: u8 = 0x18;  // ST_TR_DATAOUT - transfer cmd done, target requesting data
    pub const TRANSFER_DATA_IN: u8 = 0x19;   // ST_TR_DATAIN - transfer cmd done, target sending data
    pub const TRANSFER_STATUS_IN: u8 = 0x1B;  // ST_TR_STATIN - target sending status
    pub const TRANSFER_MSG_IN: u8 = 0x1F;  // ST_TR_MSGIN - transfer cmd done, target sending msg

    pub const TRANSFER_PAUSE: u8 = 0x20;  // ST_TRANPAUSE - transfer cmd paused with ACK
    pub const SAVE_DATA_POINTERS: u8 = 0x21;  // ST_SAVEDP
    pub const SELECTION_ABORTED: u8 = 0x22;
    pub const RECEIVE_SEND_ABORTED: u8 = 0x23;
    pub const RECEIVE_SEND_ABORTED_ATN: u8 = 0x24;
    pub const ABORT_DURING_SELECTION: u8 = 0x25;
    pub const RESELECTED_AFTER_DISC: u8 = 0x27;  // ST_A_RESELECT (93A)
    pub const TRANSFER_ABORTED: u8 = 0x28;

    pub const INVALID_COMMAND: u8 = 0x40;
    pub const UNEXPECTED_DISCONNECT: u8 = 0x41;  // ST_UNEXPDISC
    pub const SELECTION_TIMEOUT: u8 = 0x42;  // ST_TIMEOUT
    pub const PARITY_ERROR: u8 = 0x43;  // ST_PARITY
    pub const PARITY_ERROR_ATN: u8 = 0x44;  // ST_PARITY_ATN
    pub const LOGICAL_ADDRESS_TOO_LARGE: u8 = 0x45;
    pub const RESELECTION_MISMATCH: u8 = 0x46;
    pub const INCORRECT_STATUS_BYTE: u8 = 0x47;  // ST_INCORR_DATA
    pub const UNEXPECTED_PHASE: u8 = 0x48;
    pub const UNEXPECTED_RECV_DATA: u8 = 0x48;  // ST_UNEX_RDATA
    pub const UNEXPECTED_SEND_DATA: u8 = 0x49;  // ST_UNEX_SDATA
    pub const UNEXPECTED_CMD_PHASE: u8 = 0x4A;  // ST_UNEX_CMDPH
    pub const UNEXPECTED_SEND_STATUS: u8 = 0x4B;  // ST_UNEX_SSTATUS
    pub const UNEXPECTED_REQ_MSG_OUT: u8 = 0x4E;  // ST_UNEX_RMESGOUT
    pub const UNEXPECTED_SEND_MSG_IN: u8 = 0x4F;  // ST_UNEX_SMESGIN

    pub const RESELECTED: u8 = 0x80;  // ST_RESELECT (WD33C93)
    pub const RESELECTED_EAF: u8 = 0x81;  // ST_93A_RESEL (reselected while idle, 93A)
    pub const SELECTED: u8 = 0x82;
    pub const SELECTED_ATN: u8 = 0x83;
    pub const ATN: u8 = 0x84;
    pub const DISCONNECT: u8 = 0x85;  // ST_DISCONNECT
    pub const NEED_COMMAND_SIZE: u8 = 0x87;  // ST_NEEDCMD
    pub const REQ: u8 = 0x88;
    pub const REQ_SEND_MSG_OUT: u8 = 0x8E;  // ST_REQ_SMESGOUT
    pub const REQ_SEND_MSG_IN: u8 = 0x8F;  // ST_REQ_SMESGIN
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
    pub const DISCONNECTED: u8 = 0x00;
    pub const SELECTED: u8 = 0x10;
    pub const IDENTIFY_SENT: u8 = 0x20;
    pub const COMMAND_START: u8 = 0x30;
    pub const SAVE_DATA_POINTER: u8 = 0x41;
    pub const DISCONNECT_MSG: u8 = 0x42;
    pub const DISCONNECTED_OK: u8 = 0x43;
    pub const RESELECTED: u8 = 0x44;
    pub const IDENTIFY_RECEIVED: u8 = 0x45;
    pub const DATA_XFER_DONE: u8 = 0x46;
    pub const STATUS_PHASE: u8 = 0x47;
    pub const STATUS_RECEIVED: u8 = 0x50;
    pub const COMPLETE_MSG: u8 = 0x60;
}

struct Wd33c93aState {
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
    pending_status: u8,
    pending_msg: u8,
    advanced_mode: bool,
    has_pending_command: bool,
    // Mid-transfer pause state for 256KB chunk re-arm
    xfer_data: Vec<u8>,         // full data buffer for current SCSI command
    xfer_offset: usize,         // bytes already transferred
    xfer_direction_in: bool,    // true=send to host (READ cmd), false=receive from host (WRITE cmd)
    // Debug tracking: last values returned for register reads
    last_read_asr: Option<u8>,
    last_read_reg: Option<(u8, u8)>, // (register, value)
}

pub trait ScsiCallback: Send + Sync {
    fn set_interrupt(&self, level: bool);
}

pub struct Wd33c93a {
    state: Arc<Mutex<Wd33c93aState>>,
    cond: Arc<Condvar>,
    thread: Mutex<Option<thread::JoinHandle<()>>>,
    running: Arc<AtomicBool>,
    dma: Option<Arc<dyn DmaClient>>,
    callback: Option<Arc<dyn ScsiCallback>>,
    /// Activity heartbeat shared with the display thread.
    heartbeat: Arc<AtomicU64>,
}

impl Wd33c93a {
    pub fn new(dma: Option<Arc<dyn DmaClient>>, callback: Option<Arc<dyn ScsiCallback>>, heartbeat: Arc<AtomicU64>) -> Self {
        Self {
            state: Arc::new(Mutex::new(Wd33c93aState {
                regs: [0; 32],
                ar: 0,
                asr: 0, // Initially not busy, no interrupt
                devices: Default::default(),
                fifo: VecDeque::new(),
                data_direction_in: false,
                target_id: 0,
                pending_status: 0,
                pending_msg: 0,
                advanced_mode: false,
                has_pending_command: false,
                xfer_data: Vec::new(),
                xfer_offset: 0,
                xfer_direction_in: false,
                last_read_asr: None,
                last_read_reg: None,
            })),
            cond: Arc::new(Condvar::new()),
            thread: Mutex::new(None),
            running: Arc::new(AtomicBool::new(false)),
            dma,
            callback,
            heartbeat,
        }
    }

    /// Attach a SCSI device.
    /// For CD-ROMs, `discs` is the full ordered list of ISO paths; the first
    /// entry is mounted immediately.  For HDDs `discs` is ignored — only
    /// `path` is used.
    pub fn add_device(&self, id: usize, path: &str, is_cdrom: bool, discs: Vec<String>, overlay: bool) -> std::io::Result<()> {
        use crate::cow_disk::CowDisk;
        use crate::scsi::DiskBackend;

        let (backend, size) = if overlay && !is_cdrom {
            let overlay_path = format!("{}.overlay", path);
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

    /// Return disc info for all attached CD-ROM devices.
    pub fn disc_status(&self) -> Vec<(usize, String, Vec<String>)> {
        let state = self.state.lock();
        state.devices.iter().enumerate()
            .filter_map(|(id, d)| {
                let dev = d.as_ref()?;
                if !dev.is_cdrom() { return None; }
                Some((id, dev.current_disc().to_string(), dev.disc_list().to_vec()))
            })
            .collect()
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
            let mut val = state.asr;

            // Compute DBR bit based on COMMAND_PHASE and FIFO state
            if state.compute_dbr() {
                val |= asr::DBR;
            }

            if state.last_read_asr.is_none() || state.last_read_asr.unwrap() != val {
                dlog!(LogModule::Scsi, "WD33C93A: Read ASR -> {:02x}", val);
                state.last_read_asr = Some(val);
            }
            state.last_read_reg = None;
            return BusRead8::ok(val);
        } else if addr == 1 {
            // Read register pointed to by AR
            let ar = state.ar & 0x1F;
            
            if ar == regs::DATA {
                let val = state.fifo.pop_front().unwrap_or(0);
                dlog!(LogModule::Scsi, "WD33C93A: Read FIFO -> {:02x}", val);
                state.last_read_asr = None;
                state.last_read_reg = None;
                return BusRead8::ok(val);
            }

            if ar == regs::AUX_STATUS_DIRECT {
                let mut val = state.asr;
                if state.compute_dbr() {
                    val |= asr::DBR;
                }
                return BusRead8::ok(val);
            }

            let val = state.regs[ar as usize];

            // Reading SCSI Status (0x17) clears the interrupt bit in ASR
            if ar == regs::SCSI_STATUS {
                if (state.asr & asr::INT) != 0 {
                    state.asr &= !asr::INT;
                    if let Some(cb) = &self.callback {
                        cb.set_interrupt(false);
                    }
                }
            }

            // Auto-increment for registers except COMMAND (0x18), DATA (0x19), AUX_STATUS (0x1F)
            if ar != regs::COMMAND && ar != regs::AUX_STATUS_DIRECT {
                state.ar = (ar + 1) & 0x1F;
            }

            if ar == regs::TARGET_LUN || ar == regs::SCSI_STATUS {
                dlog!(LogModule::Scsi, "WD33C93A: Read Reg {:02x} ({}) -> {:02x}",
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
                    dlog!(LogModule::Scsi, "WD33C93A: Read Reg {:02x} -> {:02x}", ar, val);
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
            dlog!(LogModule::Scsi, "WD33C93A: Write AR <- {:02x}", val);
            state.last_read_asr = None;
            state.last_read_reg = None;
            return BUS_OK;
        } else if addr == 1 {
            // Write register pointed to by AR
            let ar = state.ar & 0x1F;

            dlog!(LogModule::Scsi, "WD33C93A: Write Reg {:02x} <- {:02x}", ar, val);
            state.last_read_asr = None;
            state.last_read_reg = None;

            if ar == regs::DATA {
                state.fifo.push_back(val);
                self.cond.notify_one();
                return BUS_OK;
            }

            state.regs[ar as usize] = val;

            if ar == regs::COMMAND {
                state.has_pending_command = true;
                state.asr |= asr::CIP;  // Set Command In Progress
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
    }
    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }
        let state = self.state.clone();
        let cond = self.cond.clone();
        let running = self.running.clone();
        let dma = self.dma.clone();
        let callback = self.callback.clone();
        let heartbeat = self.heartbeat.clone();

        *self.thread.lock() = Some(thread::Builder::new().name("WD33C93A".to_string()).spawn(move || {
            let mut state_guard = state.lock();
            while running.load(Ordering::Relaxed) {
                cond.wait(&mut state_guard);
                if !running.load(Ordering::Relaxed) { break; }

                // Check for Command Register write
                if state_guard.has_pending_command {
                    let cmd_reg = state_guard.regs[regs::COMMAND as usize];
                    state_guard.has_pending_command = false;
                    dlog!(LogModule::Scsi, "WD33C93A: Processing Command {:02x}", cmd_reg);
                    state_guard.regs[regs::COMMAND as usize] = 0; // Clear command register
                    drop(state_guard); // Drop lock before processing to allow re-entry if needed (though we need state access)

                    let mut state = state.lock();
                    state.process_wd_command(cmd_reg, dma.as_deref());

                    // Signal SCSI activity heartbeat (IDs 0-6; 7 is host, skip).
                    let tid = state.target_id;
                    if tid < 7 {
                        heartbeat.fetch_or(1u64 << (crate::rex3::Rex3::HB_SCSI_BASE as u64 + tid as u64), Ordering::Relaxed);
                    }

                    // Update interrupt state
                    let int_active = (state.asr & asr::INT) != 0;
                    if let Some(cb) = &callback {
                        cb.set_interrupt(int_active);
                    }

                    // Re-assign state_guard for next iteration
                    state_guard = state;
                }
            }
        }).unwrap());
    }
    fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("scsi".to_string(), "SCSI commands: scsi status | scsi eject <id> | scsi debug <on|off> [DEV]".to_string()),
            ("cow".to_string(), "COW overlay: cow status | cow commit [id] | cow reset [id]".to_string()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn Write + Send>) -> Result<(), String> {
        if cmd == "scsi" {
            match args.first().copied() {
                Some("debug") => {
                    let val = match args.get(1).copied() {
                        Some("on")  => true,
                        Some("off") => false,
                        _ => return Err("Usage: scsi debug <on|off>".to_string()),
                    };
                    if val { devlog().enable(LogModule::Scsi); } else { devlog().disable(LogModule::Scsi); }
                    writeln!(writer, "SCSI debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                Some("status") => {
                    let discs = self.disc_status();
                    if discs.is_empty() {
                        writeln!(writer, "No CD-ROM devices attached").unwrap();
                    } else {
                        for (id, active, list) in discs {
                            writeln!(writer, "SCSI ID {}: {} ({} disc(s))", id, active, list.len()).unwrap();
                            for (i, d) in list.iter().enumerate() {
                                writeln!(writer, "  [{}] {}{}", i, d, if i == 0 { " <active>" } else { "" }).unwrap();
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
                _ => return Err("Usage: scsi debug <on|off> | scsi status | scsi eject <id>".to_string()),
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
        state.xfer_data.clear();
        state.xfer_offset = 0;
        state.regs[regs::COMMAND_PHASE as usize] = command_phase::DISCONNECTED;
        state.asr = asr::INT;
        state.target_id = 0;
        state.pending_status = 0;
        state.pending_msg = 0;
        state.advanced_mode = false;
        state.has_pending_command = false;
        state.last_read_asr = None;
        state.last_read_reg = None;
        for i in 0x01..=0x16usize {
            state.regs[i] = 0;
        }
        state.regs[regs::COMMAND as usize] = 0;
        let own_id = state.regs[regs::OWN_ID as usize];
        let eaf = (own_id & 0x08) != 0;
        state.advanced_mode = eaf;
        state.regs[regs::SCSI_STATUS as usize] = if eaf { scsi_status::RESET_EAF } else { scsi_status::RESET };
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
        tbl.insert("pending_status".into(),    hex_u8(state.pending_status));
        tbl.insert("pending_msg".into(),       hex_u8(state.pending_msg));
        tbl.insert("advanced_mode".into(),     toml::Value::Boolean(state.advanced_mode));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut state = self.state.lock();
        if let Some(r) = get_field(v, "regs") { load_u8_slice(r, &mut state.regs); }
        if let Some(x) = get_field(v, "ar")               { if let Some(n) = toml_u8(x)   { state.ar = n; } }
        if let Some(x) = get_field(v, "asr")              { if let Some(n) = toml_u8(x)   { state.asr = n; } }
        if let Some(x) = get_field(v, "data_direction_in"){ if let Some(b) = toml_bool(x) { state.data_direction_in = b; } }
        if let Some(x) = get_field(v, "target_id")        { if let Some(n) = toml_u8(x)   { state.target_id = n as usize; } }
        if let Some(x) = get_field(v, "pending_status")   { if let Some(n) = toml_u8(x)   { state.pending_status = n; } }
        if let Some(x) = get_field(v, "pending_msg")      { if let Some(n) = toml_u8(x)   { state.pending_msg = n; } }
        if let Some(x) = get_field(v, "advanced_mode")    { if let Some(b) = toml_bool(x) { state.advanced_mode = b; } }
        // Transient state cleared on load.
        state.fifo.clear();
        state.xfer_data.clear();
        state.xfer_offset = 0;
        state.has_pending_command = false;
        state.last_read_asr = None;
        state.last_read_reg = None;
        Ok(())
    }
}

impl Wd33c93aState {
    fn use_dma(&self) -> bool {
        let mode = (self.regs[regs::CONTROL as usize] >> 5) & 0x7;
        mode != 0
    }

    /// Compute DBR (Data Buffer Ready) bit based on COMMAND_PHASE register
    fn compute_dbr(&self) -> bool {
        let cmd_phase = self.regs[regs::COMMAND_PHASE as usize];
        match cmd_phase & 0xF0 {
            0x00 | 0x10 | 0x20 => false,  // Disconnected, Selected, Identify - no data ready
            0x30 => true,                  // Command phase - ready to receive CDB bytes
            0x40 => {
                // Disconnect/reselect/data transfer phases
                match cmd_phase {
                    0x46 => self.data_direction_in && !self.fifo.is_empty(),  // Data transfer done, data in
                    0x47 | 0x50 => !self.fifo.is_empty(),  // Status phase - data ready when in FIFO
                    _ => false,
                }
            }
            0x50 => !self.fifo.is_empty(),  // Status received - message in FIFO
            0x60 => !self.fifo.is_empty(),  // Command complete message in FIFO
            _ => false,
        }
    }

    /// Set new phase and/or status, logging changes via LogModule::Scsi.
    fn set_phase_status(&mut self, phase: u8, status: u8) {
        let old_phase = self.regs[regs::COMMAND_PHASE as usize];
        let old_status = self.regs[regs::SCSI_STATUS as usize];

        self.regs[regs::COMMAND_PHASE as usize] = phase;
        self.regs[regs::SCSI_STATUS as usize] = status;

        if old_phase != phase || old_status != status {
            if old_phase != phase && old_status != status {
                dlog!(LogModule::Scsi, "WD33C93A: Phase {:02x}->{:02x} Status {:02x}->{:02x}",
                    old_phase, phase, old_status, status);
            } else if old_phase != phase {
                dlog!(LogModule::Scsi, "WD33C93A: Phase {:02x}->{:02x} (Status={:02x})", old_phase, phase, status);
            } else {
                dlog!(LogModule::Scsi, "WD33C93A: Status {:02x}->{:02x} (Phase={:02x})", old_status, status, phase);
            }
        }
    }

    fn process_wd_command(&mut self, cmd: u8, dma: Option<&dyn DmaClient>) {
        {
            let cmd_phase = self.regs[regs::COMMAND_PHASE as usize];
            dlog!(LogModule::Scsi, "WD33C93A: Command {:02x} (CmdPhase: {:02x}, Tgt: {}, ASR: {:02x})",
                cmd, cmd_phase, self.target_id, self.asr);
            dlog!(LogModule::Scsi, "          Regs: CTRL={:02x} DST_ID={:02x} SRC_ID={:02x}",
                self.regs[regs::CONTROL as usize], self.regs[regs::DESTINATION_ID as usize], self.regs[regs::SOURCE_ID as usize]);
        }

        if cmd == cmd::RESET {
            self.fifo.clear();
            self.regs[regs::COMMAND_PHASE as usize] = command_phase::DISCONNECTED;
            self.asr = asr::INT;
            self.target_id = 0;
            self.pending_status = 0;
            self.pending_msg = 0;
            self.advanced_mode = false;
            self.has_pending_command = false;
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

        // Resume mid-transfer if SELECT_ATN_XFER arrives while a chunked transfer is paused
        if cmd == cmd::SELECT_ATN_XFER && !self.xfer_data.is_empty() {
            dlog!(LogModule::Scsi, "WD33C93A: SELECT_ATN_XFER resume: dir_in={} offset={:x}/{:x}",
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
            self.set_phase_status(command_phase::COMPLETE_MSG, scsi_status::SELECT_TRANSFER_SUCCESS);
            self.regs[regs::TARGET_LUN as usize] = self.pending_status;
            self.asr |= asr::INT;
            self.asr &= !asr::CIP;
            return;
        }

        match cmd {
            cmd::SELECT_ATN | cmd::SELECT_ATN_XFER | cmd::SELECT | cmd::SELECT_XFER => {
                self.target_id = (self.regs[regs::DESTINATION_ID as usize] & 0x7) as usize;
            }
            _ => {}
        }

        if self.devices[self.target_id].is_none() {
            dlog!(LogModule::Scsi, "WD33C93A: No device at target {}, timing out", self.target_id);
            self.set_phase_status(command_phase::DISCONNECTED, scsi_status::SELECTION_TIMEOUT);
            self.asr |= asr::INT;
            self.asr &= !asr::CIP;
            return;
        }

        match cmd {
            cmd::ABORT => {
                self.fifo.clear();
                self.xfer_data.clear();
                self.xfer_offset = 0;
                self.asr = asr::INT;
                let status = if self.advanced_mode { scsi_status::RESET_EAF } else { scsi_status::RESET };
                self.set_phase_status(command_phase::DISCONNECTED, status);
            }
            cmd::ASSERT_ATN => {
                self.asr &= !asr::CIP;
            }
            cmd::DISCONNECT => {
                self.set_phase_status(command_phase::DISCONNECTED, scsi_status::DISCONNECT);
                self.asr |= asr::INT;
                self.asr &= !(asr::CIP | asr::BSY);
            }
            cmd::SELECT_ATN | cmd::SELECT_ATN_XFER | cmd::SELECT | cmd::SELECT_XFER => {
                let status = self.regs[regs::SCSI_STATUS as usize];
                self.set_phase_status(command_phase::SELECTED, status);

                if cmd == cmd::SELECT_ATN_XFER || cmd == cmd::SELECT_XFER {
                    dlog!(LogModule::Scsi, "WD33C93A: SELECT_XFER/SELECT_ATN_XFER");
                    // CDB is stored in registers starting at CDB_1 (0x03)
                    let opcode = self.regs[regs::CDB_1 as usize];
                    let len = self.get_cdb_length(opcode);
                    let mut cdb = Vec::with_capacity(len);
                    for i in 0..len {
                        cdb.push(self.regs[(regs::CDB_1 as usize) + i]);
                    }
                    self.process_scsi_command(&cdb, true, dma);
                } else {
                    dlog!(LogModule::Scsi, "WD33C93A: SELECT/SELECT_ATN");
                    // Manual SELECT: Just assert interrupt "Select Complete" (0x11)
                    // Driver will then write CDB and issue TRANSFER_INFO
                    let phase = self.regs[regs::COMMAND_PHASE as usize];
                    self.set_phase_status(phase, scsi_status::SELECT_SUCCESS);
                    self.asr |= asr::INT;
                    self.asr &= !asr::CIP;
                }
            }
            cmd::TRANSFER_INFO => {
                let cmd_phase = self.regs[regs::COMMAND_PHASE as usize];
                dlog!(LogModule::Scsi, "WD33C93A: TRANSFER_INFO CmdPhase={:02x}", cmd_phase);
                match cmd_phase & 0xF0 {
                    0x10 | 0x20 | 0x30 => {
                        // Selected, Identify sent, or Command phase - execute CDB
                        let mut cdb = Vec::new();
                        if !self.fifo.is_empty() {
                            let opcode = self.fifo.pop_front().unwrap_or(0);
                            let len = self.get_cdb_length(opcode);
                            cdb.reserve(len);
                            cdb.push(opcode);
                            for _ in 1..len {
                                cdb.push(self.fifo.pop_front().unwrap_or(0));
                            }
                        } else {
                            // Fallback to CDB registers
                            let opcode = self.regs[regs::CDB_1 as usize];
                            let len = self.get_cdb_length(opcode);
                            cdb.reserve(len);
                            for i in 0..len {
                                cdb.push(self.regs[(regs::CDB_1 as usize) + i]);
                            }
                        }

                        self.process_scsi_command(&cdb, false, dma);
                    }
                    0x40 if cmd_phase == command_phase::DATA_XFER_DONE => {
                        // Resume after data transfer - move to Status
                        self.regs[regs::TARGET_LUN as usize] = self.pending_status;
                        self.set_phase_status(command_phase::STATUS_PHASE, scsi_status::TRANSFER_STATUS_IN);
                        self.asr |= asr::INT;
                        self.asr &= !asr::CIP;
                    }
                    0x40 if cmd_phase == command_phase::STATUS_PHASE => {
                        // In status phase - move to Message In
                        self.regs[regs::TARGET_LUN as usize] = self.pending_status;
                        self.set_phase_status(command_phase::STATUS_RECEIVED, scsi_status::TRANSFER_MSG_IN);
                        self.asr |= asr::INT;
                        self.asr &= !asr::CIP;
                    }
                    0x50 => {
                        // Status received - send message byte (COMMAND COMPLETE)
                        self.fifo.push_back(self.pending_msg);
                        self.set_phase_status(command_phase::COMPLETE_MSG, scsi_status::DISCONNECT);
                        self.asr |= asr::INT;
                        self.asr &= !asr::CIP;
                    }
                    _ => {
                        dlog!(LogModule::Scsi, "WD33C93A: TRANSFER_INFO in unexpected phase {:02x}", cmd_phase);
                        self.asr &= !asr::CIP;
                    }
                }
            }
            cmd::NEGATE_ACK => {
                // Often used to complete a message transfer
                let cmd_phase = self.regs[regs::COMMAND_PHASE as usize];
                if cmd_phase == command_phase::DISCONNECTED || cmd_phase == command_phase::COMPLETE_MSG {
                     // Command Complete
                     self.set_phase_status(cmd_phase, scsi_status::SELECT_TRANSFER_SUCCESS);
                     // Or 0x85 (Disconnect)
                     // For now, just clear interrupt
                     self.asr &= !asr::INT;
                }
                self.asr &= !asr::CIP;
            }
            _ => {
                dlog!(LogModule::Scsi, "WD33C93A: Unimplemented WD command {:02x}", cmd);
                self.asr &= !asr::CIP;
            }
        }
    }


    fn get_cdb_length(&self, opcode: u8) -> usize {
        // In advanced mode, OWN_ID register can override the standard CDB length
        if self.advanced_mode {
            let own_id_len = self.regs[regs::OWN_ID as usize] as usize;
            if own_id_len > 0 {
                return own_id_len;
            }
        }
        // Use standard SCSI CDB length based on opcode group
        scsi::get_cdb_length(opcode)
    }

    fn process_scsi_command(&mut self, cdb: &[u8], auto_mode: bool, dma: Option<&dyn DmaClient>) {
        if cdb.is_empty() {
            dlog!(LogModule::Scsi, "WD33C93A: Empty CDB!");
            self.asr |= asr::LCI | asr::INT;
            self.asr &= !asr::CIP;
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

            let mut extra = String::new();
            match cdb[0] {
                scsi_cmd::READ_6 | scsi_cmd::WRITE_6 => {
                    let lba = (((cdb[1] & 0x1F) as u64) << 16) | ((cdb[2] as u64) << 8) | (cdb[3] as u64);
                    let count = if cdb[4] == 0 { 256 } else { cdb[4] as usize };
                    extra = format!(" LBA={:x} Blocks={:x} Bytes={:x}", lba, count, count * 512);
                }
                scsi_cmd::READ_10 | scsi_cmd::WRITE_10 => {
                    let lba = ((cdb[2] as u64) << 24) | ((cdb[3] as u64) << 16) | ((cdb[4] as u64) << 8) | (cdb[5] as u64);
                    let count = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                    extra = format!(" LBA={:x} Blocks={:x} Bytes={:x}", lba, count, count * 512);
                }
                scsi_cmd::INQUIRY | scsi_cmd::REQUEST_SENSE | scsi_cmd::MODE_SENSE_6 => {
                    let len = cdb[4] as usize;
                    extra = format!(" Bytes={:x}", len);
                }
                _ => {}
            }
            dlog!(LogModule::Scsi, "WD33C93A: SCSI Command {:02x} ({}) Target {}{}", cdb[0], cmd_name, self.target_id, extra);
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
            scsi_cmd::READ_TOC_PMA_ATIP | scsi_cmd::GET_CONFIGURATION => {
                // Allocation length in bytes 7-8
                let len = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
                ScsiDataLength::Fixed(len)
            }
            _ => ScsiDataLength::Unlimited,
        };

        // For WRITE commands, receive data first (data out from host to target)
        let data_in = match cdb[0] {
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
                    if self.use_dma() {
                        // send_data_chunked may pause mid-transfer and raise its own interrupt
                        if !self.send_data_chunked(response.data, 0, dma) {
                            return;
                        }
                    } else {
                        self.send_data(&response.data, dma);
                    }
                }
                self.finish_command(response.status);
            }
            Err(_) => {
                self.finish_command(0x02); // Check Condition
            }
        }

        if auto_mode {
            self.set_phase_status(command_phase::COMPLETE_MSG, scsi_status::SELECT_TRANSFER_SUCCESS);
            self.regs[regs::TARGET_LUN as usize] = self.pending_status;
            self.asr |= asr::INT;
            self.asr &= !asr::CIP;
        } else {
            self.set_phase_status(command_phase::DATA_XFER_DONE, scsi_status::TRANSFER_STATUS_IN);
            self.asr |= asr::INT;
            self.asr &= !asr::CIP;
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
        dlog!(LogModule::Scsi, "WD33C93A: Sending {:x} bytes via DMA (offset={:x})", data.len() - offset, offset);
        if let Some(dma_dev) = dma {
            let total = data.len();
            let last_idx = total.saturating_sub(1);
            let mut i = offset;
            while i < total {
                let is_last = i == last_idx;
                let (st, _) = dma_dev.write(data[i] as u32, is_last);
                // On refused: byte was not accepted, do not advance or decrement.
                // On eox/irq: byte was accepted, advance and decrement.
                if !st.refused() {
                    i += 1;
                    self.decrement_transfer_count();
                }
                let pause = (st.eox() || st.irq() || st.refused()) && !is_last;
                if pause {
                    dlog!(LogModule::Scsi, "WD33C93A: EOX/XIE/refused at offset={:x}, remaining={:x} — pausing", i, total - i);
                    self.xfer_data = data;
                    self.xfer_offset = i;
                    self.xfer_direction_in = true;
                    // ST_UNEX_SDATA (0x49): target is unexpectedly sending data (long READ)
                    self.set_phase_status(command_phase::DATA_XFER_DONE, scsi_status::UNEXPECTED_SEND_DATA);
                    self.asr |= asr::INT;
                    self.asr &= !asr::CIP;
                    return false;
                }
            }
        }
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
        dlog!(LogModule::Scsi, "WD33C93A: Receiving {:x} bytes via DMA (have={:x})", total - data.len(), data.len());
        if let Some(dma_dev) = dma {
            while data.len() < total {
                match dma_dev.read() {
                    Some((val, st, _)) => {
                        // Byte accepted — decrement transfer count register to mirror real HW.
                        data.push(val as u8);
                        self.decrement_transfer_count();
                        let pause = (st.eox() || st.irq()) && data.len() < total;
                        if pause {
                            dlog!(LogModule::Scsi, "WD33C93A: EOX/XIE at offset={:x}, remaining={:x} — pausing", data.len(), total - data.len());
                            self.xfer_data = data;
                            self.xfer_offset = total; // store total as sentinel; xfer_data.len() is progress
                            self.xfer_direction_in = false;
                            // ST_UNEX_RDATA (0x48): host is unexpectedly receiving data (long WRITE)
                            self.set_phase_status(command_phase::DATA_XFER_DONE, scsi_status::UNEXPECTED_RECV_DATA);
                            self.asr |= asr::INT;
                            self.asr &= !asr::CIP;
                            return None;
                        }
                    }
                    None => {
                        // Channel went inactive (EOX on empty terminator descriptor) mid-transfer
                        let remaining = total - data.len();
                        if remaining > 0 {
                            dlog!(LogModule::Scsi, "WD33C93A: DMA inactive at offset={:x}, remaining={:x} — pausing", data.len(), remaining);
                            self.xfer_data = data;
                            self.xfer_offset = total;
                            self.xfer_direction_in = false;
                            self.set_phase_status(command_phase::DATA_XFER_DONE, scsi_status::UNEXPECTED_RECV_DATA);
                            self.asr |= asr::INT;
                            self.asr &= !asr::CIP;
                            return None;
                        }
                        break;
                    }
                }
            }
        } else {
            while data.len() < total {
                data.push(self.fifo.pop_front().unwrap_or(0));
            }
        }
        Some(data)
    }

    /// Push `data` to the host via DMA or PIO (FIFO), whichever is active.
    fn send_data(&mut self, data: &[u8], dma: Option<&dyn DmaClient>) {
        if !data.is_empty() {
            if self.use_dma() {
                dlog!(LogModule::Scsi, "WD33C93A: Sending {:x} bytes via DMA", data.len());
            } else {
                dlog!(LogModule::Scsi, "WD33C93A: Pushing {:x} bytes to FIFO", data.len());
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

    fn set_transfer_count(&mut self, count: u32) {
        self.regs[regs::TRANSFER_COUNT_MSB as usize] = ((count >> 16) & 0xFF) as u8;
        self.regs[regs::TRANSFER_COUNT_2ND as usize] = ((count >> 8)  & 0xFF) as u8;
        self.regs[regs::TRANSFER_COUNT_LSB as usize] = (count         & 0xFF) as u8;
    }

    /// Decrement the 24-bit transfer count register by 1, mirroring real WD33C93A hardware
    /// which decrements on every byte transferred. save_datap() reads this to compute
    /// count_xferd = wd_xferlen - count_remain.
    fn decrement_transfer_count(&mut self) {
        let lo  = self.regs[regs::TRANSFER_COUNT_LSB as usize] as u32;
        let mid = self.regs[regs::TRANSFER_COUNT_2ND as usize] as u32;
        let hi  = self.regs[regs::TRANSFER_COUNT_MSB as usize] as u32;
        let count = (hi << 16) | (mid << 8) | lo;
        let count = count.saturating_sub(1);
        self.regs[regs::TRANSFER_COUNT_MSB as usize] = ((count >> 16) & 0xFF) as u8;
        self.regs[regs::TRANSFER_COUNT_2ND as usize] = ((count >> 8)  & 0xFF) as u8;
        self.regs[regs::TRANSFER_COUNT_LSB as usize] = (count         & 0xFF) as u8;
    }
}