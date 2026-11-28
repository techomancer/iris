use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU8, AtomicBool, Ordering};
use std::collections::VecDeque;
use std::io::{self, Write, Read};
use std::fs::OpenOptions;
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::net::{TcpListener, TcpStream};
use crate::hptimer::{Timer, TimerManager};
use crate::irq::IrqLine;
use crate::net::wake::{host_io, HostPump};
use std::io::Write as IoWrite;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, Device, Resettable, Saveable};
use crate::snapshot::{get_field, toml_u8, u8_slice_to_toml, load_u8_slice, hex_u8};
use crate::devlog::LogModule;
use crate::dev::serial_host::{LineConfig, ModemIn, ModemOut, NullBackend, Parity, SerialBackend};

pub mod scc_regs {
    pub const WR0: u8 = 0; // Command/Pointer
    pub const WR1: u8 = 1; // Interrupt/DMA
    pub const WR2: u8 = 2; // Interrupt Vector
    pub const WR3: u8 = 3; // Rx Parameters
    pub const WR4: u8 = 4; // Tx/Rx Miscellaneous
    pub const WR5: u8 = 5; // Tx Parameters
    pub const WR6: u8 = 6; // Sync Char 1
    pub const WR7: u8 = 7; // Sync Char 2 / Flag
    pub const WR8: u8 = 8; // Transmit Buffer
    pub const WR9: u8 = 9; // Master Interrupt Control
    pub const WR10: u8 = 10; // Extra Status
    pub const WR11: u8 = 11; // Clock Mode
    pub const WR12: u8 = 12; // Baud Rate Generator Time Constant Low
    pub const WR13: u8 = 13; // Baud Rate Generator Time Constant High
    pub const WR14: u8 = 14; // Misc Features (DPLL, BRG)
    pub const WR15: u8 = 15; // External/Status Interrupt Control

    pub const RR0: u8 = 0; // Status
    pub const RR1: u8 = 1; // Special Status
    pub const RR2: u8 = 2; // Interrupt Vector
    pub const RR3: u8 = 3; // Interrupt Pending
    pub const RR8: u8 = 8; // Receive Buffer
    pub const RR10: u8 = 10; // Misc Status
}

// Bit definitions for Register RR0 (Status)
pub mod rr0 {
    pub const RX_CHAR_AVAILABLE: u8 = 1 << 0;
    pub const ZERO_COUNT_DETECTED: u8 = 1 << 1;
    pub const TX_BUFFER_EMPTY: u8 = 1 << 2;
    pub const DCD_STATE: u8 = 1 << 3;
    pub const SYNC_HUNT: u8 = 1 << 4;
    pub const CTS_STATE: u8 = 1 << 5;
    pub const TX_UNDERRUN_EOM: u8 = 1 << 6;
    pub const BREAK_ABORT: u8 = 1 << 7;
}

// Bit definitions for Register WR1 (Interrupt)
pub mod wr1 {
    pub const EXT_INT_EN: u8 = 1 << 0;
    pub const TX_INT_EN: u8 = 1 << 1;
    pub const STATUS_AFFECTS_VECTOR: u8 = 1 << 2;
    pub const RX_INT_DIS: u8 = 0 << 3;
    pub const RX_INT_FIRST: u8 = 1 << 3;
    pub const RX_INT_ALL_PARA: u8 = 2 << 3;
    pub const RX_INT_ALL_SPECIAL: u8 = 3 << 3;
    pub const WAIT_ENABLE: u8 = 1 << 7;
}

// Bit definitions for Register WR3 (Rx Parameters)
pub mod wr3 {
    pub const RX_ENABLE: u8 = 1 << 0;
    pub const SYNC_CHAR_LOAD_INHIBIT: u8 = 1 << 1;
    pub const ADDRESS_SEARCH_MODE: u8 = 1 << 2;
    pub const RX_CRC_EN: u8 = 1 << 3;
    pub const HUNT_MODE: u8 = 1 << 4;
    pub const AUTO_ENABLE: u8 = 1 << 5;
    pub const RX_5BITS: u8 = 0 << 6;
    pub const RX_7BITS: u8 = 1 << 6;
    pub const RX_6BITS: u8 = 2 << 6;
    pub const RX_8BITS: u8 = 3 << 6;
}

// Bit definitions for Register WR4 (Protocol)
pub mod wr4 {
    pub const PARITY_EN: u8 = 1 << 0;
    pub const PARITY_EVEN: u8 = 1 << 1;
    pub const STOP1: u8 = 1 << 2;
    pub const STOP1_5: u8 = 2 << 2;
    pub const STOP2: u8 = 3 << 2;
    pub const SYNC_MODE_8BIT: u8 = 0 << 4;
    pub const SYNC_MODE_16BIT: u8 = 1 << 4;
    pub const SYNC_MODE_SDLC: u8 = 2 << 4;
    pub const ASYNC_MODE: u8 = 3 << 4; // x1, x16, x32, x64
    pub const CLOCK_MODE_SHIFT: u8 = 6;
}

// Bit definitions for Register WR5 (Tx Parameters)
pub mod wr5 {
    pub const TX_CRC_EN: u8 = 1 << 0;
    pub const RTS: u8 = 1 << 1;
    pub const CRC16: u8 = 1 << 2;
    pub const TX_ENABLE: u8 = 1 << 3;
    pub const SEND_BREAK: u8 = 1 << 4;
    pub const TX_5BITS: u8 = 0 << 5;
    pub const TX_7BITS: u8 = 1 << 5;
    pub const TX_6BITS: u8 = 2 << 5;
    pub const TX_8BITS: u8 = 3 << 5;
    pub const DTR: u8 = 1 << 7;
}

// Bit definitions for Register WR14 (Misc)
pub mod wr14 {
    pub const BRG_ENABLE: u8 = 1 << 0;
    pub const BRG_SOURCE_PCLK: u8 = 1 << 1;
    pub const AUTO_ECHO: u8 = 1 << 3;
    pub const LOCAL_LOOPBACK: u8 = 1 << 4;
}

// Bit definitions for Register WR11 (Clock Mode): the transmit clock source.
pub mod wr11 {
    pub const TX_CLOCK_SHIFT: u8 = 3;
    pub const CLOCK_RTXC: u8 = 0;
    pub const CLOCK_TRXC: u8 = 1;
    pub const CLOCK_BRG: u8 = 2;
}

// Bit definitions for Register WR15 (External/Status interrupt enables).
pub mod wr15 {
    pub const DCD_IE: u8 = 1 << 3;
    pub const CTS_IE: u8 = 1 << 5;
}

/// The SCC's bus clock (PCLK), which the baud-rate generator may count.
pub const PCLK_HZ: u32 = 10_000_000;
/// The clock on the RTxC pins (3.672 MHz on the Indy and Indigo2), which IRIX
/// feeds the baud-rate generator from.
pub const RTXC_HZ: u32 = 3_672_000;

pub struct Channel {
    pub regs: [u8; 16],
    pub reg_ptr: u8,
    pub status: u8, // RR0
    pub rx_queue: VecDeque<u8>,
    pub tx_queue: VecDeque<u8>,
    pub name: String,
    pub tx_delay: u64,
    pub ip_num: Arc<AtomicU8>,
    pub ip_other: Arc<AtomicU8>,
    pub irq: Option<IrqLine>,
    // Z85C30 Tx interrupt fires once when buffer empties, then arms again only
    // after RES_Tx_P.  Without this latch the IRQ line stays asserted forever.
    pub tx_int_pending: bool,
    /// The character in the shift register, until its time on the line is up.
    tx_shift: Option<u8>,
    /// When the last character finishes, in machine ns, so a burst is paced back to back.
    tx_clock: u64,
    /// The modem inputs as last seen, so a change can raise the
    /// external/status interrupt and a reset keeps showing the levels.
    modem_in: ModemIn,
    /// An external/status interrupt is latched (cleared by the Reset
    /// Ext/Status Interrupts command).
    ext_pending: bool,
    /// The line state the backend was last told.
    line_sent: Option<(LineConfig, ModemOut)>,
}

impl Channel {
    pub fn new(name: &str, ip_num: Arc<AtomicU8>, ip_other: Arc<AtomicU8>, irq: Option<IrqLine>) -> Self {
        let mut c = Self {
            regs: [0; 16],
            reg_ptr: 0,
            status: rr0::TX_BUFFER_EMPTY | rr0::DCD_STATE | rr0::CTS_STATE, // Tx Empty, DCD, CTS
            rx_queue: VecDeque::with_capacity(8),
            tx_queue: VecDeque::with_capacity(4),
            name: name.to_string(),
            tx_delay: 0,
            ip_num,
            ip_other,
            irq,
            tx_int_pending: false,
            tx_shift: None,
            tx_clock: 0,
            modem_in: ModemIn::IDLE,
            ext_pending: false,
            line_sent: None,
        };
        c.update_tx_delay();
        c
    }

    /// RR0's DCD and CTS bits for these input levels.
    fn modem_bits(m: ModemIn) -> u8 {
        (if m.dcd { rr0::DCD_STATE } else { 0 }) | (if m.cts { rr0::CTS_STATE } else { 0 })
    }

    /// The backend's modem inputs: shown in RR0, and a change in an input
    /// WR15 watches latches the external/status interrupt.
    pub fn apply_modem_in(&mut self, m: ModemIn) {
        if m == self.modem_in {
            return;
        }
        let changed = Self::modem_bits(m) ^ Self::modem_bits(self.modem_in);
        self.modem_in = m;
        self.status = (self.status & !(rr0::DCD_STATE | rr0::CTS_STATE)) | Self::modem_bits(m);
        let wr15 = self.regs[scc_regs::WR15 as usize];
        let watched = (if wr15 & wr15::DCD_IE != 0 { rr0::DCD_STATE } else { 0 }) | (if wr15 & wr15::CTS_IE != 0 { rr0::CTS_STATE } else { 0 });
        if changed & watched != 0 {
            self.ext_pending = true;
        }
        self.update_ip();
    }

    /// The line's rate and framing and the modem outputs, as the registers
    /// set them. The rate follows WR11's transmit clock: the baud-rate
    /// generator (fed from PCLK or RTxC per WR14), the RTxC pin directly, or
    /// the TRxC pin, an external clock whose rate is unknown here.
    pub fn line_state(&self) -> (LineConfig, ModemOut) {
        let r = &self.regs;
        let wr4 = r[scc_regs::WR4 as usize];
        let wr5 = r[scc_regs::WR5 as usize];
        let wr14 = r[scc_regs::WR14 as usize];
        let mult: u32 = match (wr4 >> wr4::CLOCK_MODE_SHIFT) & 3 {
            0 => 1,
            1 => 16,
            2 => 32,
            _ => 64,
        };
        let tc = ((r[scc_regs::WR13 as usize] as u32) << 8) | r[scc_regs::WR12 as usize] as u32;
        // WR4's stop-bit field at 0 is a synchronous mode (and the state after
        // reset): no asynchronous rate to give a host port.
        let async_mode = (wr4 >> 2) & 3 != 0;
        let baud = match (r[scc_regs::WR11 as usize] >> wr11::TX_CLOCK_SHIFT) & 3 {
            _ if !async_mode => None,
            wr11::CLOCK_BRG if wr14 & wr14::BRG_ENABLE != 0 => {
                let src = if wr14 & wr14::BRG_SOURCE_PCLK != 0 { PCLK_HZ } else { RTXC_HZ };
                Some(src / (2 * (tc + 2) * mult))
            }
            wr11::CLOCK_RTXC => Some(RTXC_HZ / mult),
            _ => None,
        };
        let data_bits = match (r[scc_regs::WR3 as usize] >> 6) & 3 {
            0 => 5,
            1 => 7,
            2 => 6,
            _ => 8,
        };
        let parity = match wr4 & 3 {
            1 => Parity::Odd,
            3 => Parity::Even,
            _ => Parity::None,
        };
        let stop_bits = if (wr4 >> 2) & 3 >= 2 { 2 } else { 1 };
        let out = ModemOut { dtr: wr5 & wr5::DTR != 0, rts: wr5 & wr5::RTS != 0, brk: wr5 & wr5::SEND_BREAK != 0 };
        (LineConfig { baud, data_bits, parity, stop_bits }, out)
    }

    /// The line state if it differs from what the backend was last told.
    fn line_change(&mut self) -> Option<(LineConfig, ModemOut)> {
        let now = self.line_state();
        if self.line_sent == Some(now) {
            return None;
        }
        self.line_sent = Some(now);
        Some(now)
    }

    fn update_tx_delay(&mut self) {
        let tc_lo = self.regs[scc_regs::WR12 as usize] as u64;
        let tc_hi = self.regs[scc_regs::WR13 as usize] as u64;
        let tc = (tc_hi << 8) | tc_lo;

        let wr4_val = self.regs[scc_regs::WR4 as usize];
        let clock_mode = (wr4_val >> wr4::CLOCK_MODE_SHIFT) & 0x3;
        let multiplier: u64 = match clock_mode {
            0 => 1,
            1 => 16,
            2 => 32,
            3 => 64,
            _ => 1,
        };
        
        // Calculate delay: Char time (10 bits) = 2 * Multiplier * (TC + 2) microseconds @ 10MHz
        self.tx_delay = 2 * multiplier * (tc + 2);
    }

    pub fn reset(&mut self) {
        self.tx_shift = None;
        self.reg_ptr = 0;
        // The modem inputs are pins, not register state: a reset keeps showing them.
        self.status = rr0::TX_BUFFER_EMPTY | Self::modem_bits(self.modem_in);
        self.regs.fill(0);
        self.tx_int_pending = false;
        self.ext_pending = false;
        self.update_ip();
        self.update_tx_delay();
    }

    pub fn read_control(&mut self) -> u8 {
        let reg = self.reg_ptr;
        let val = match self.reg_ptr {
            0 => self.status,
            1 => {
                // RR1: Special Status
                // Bit 0: All Sent (Async). Set when Tx Buffer and Shift Register are empty.
                // We approximate this by checking if tx_queue is empty.
                let mut val = 0;
                if self.tx_queue.is_empty() { val |= 0x01; }
                val
            },
            2 => self.regs[2], // RR2: Interrupt Vector
            3 => 0, // RR3: Interrupt Pending (Channel A only)
            8 => {
                // Reading RR8 is Receive Data (Alias)
                self.read_data()
            }
            10 => 0, // RR10: Misc Status
            12 => self.regs[12], // RR12: Time Constant Low
            13 => self.regs[13], // RR13: Time Constant High
            15 => self.regs[15], // RR15: Ext/Status Int Control
            _ => 0,
        };
        
        // Reset pointer after read
        self.reg_ptr = 0;
        if reg != 0 {
            crate::dlog_dev!(LogModule::Scc, "SCC({}): Read RR{} -> {:02x}", self.name, reg, val);
        }
        val
    }

    pub fn write_control(&mut self, val: u8) -> Option<u8> {
        let mut wr2_update = None;
        if self.reg_ptr != 0 {
            // Write to specific register
            let idx = self.reg_ptr as usize;
            if idx == 8 {
                // WR8 is alias for Data Register
                self.write_data(val);
            } else if idx < 16 {
                self.regs[idx] = val;
                if idx == 2 {
                    wr2_update = Some(val);
                }
                if idx == scc_regs::WR4 as usize || idx == scc_regs::WR12 as usize || idx == scc_regs::WR13 as usize {
                    self.update_tx_delay();
                }
                if idx == scc_regs::WR5 as usize {
                    crate::dlog_dev!(LogModule::Scc, "SCC({}): WR5 = {:02x} (TX_EN={})", self.name, val, (val & wr5::TX_ENABLE) != 0);
                }
            }
            self.reg_ptr = 0;
        } else {
            // Write to WR0
            self.regs[0] = val;
            let ptr = val & 0x7;
            let cmd = (val >> 3) & 0x7;

            if cmd == 1 {
                // Point High (Select registers 8-15)
                self.reg_ptr = ptr + 8;
            } else if ptr != 0 {
                self.reg_ptr = ptr;
            }

            // Command decoding
            match cmd {
                0 => {}, // Null
                1 => {}, // Point High (handled above)
                2 => { // Reset Ext/Status Interrupts
                    self.ext_pending = false;
                },
                3 => self.reset(), // Channel Reset
                4 => { // Enable Interrupt on Next Rx Character
                },
                5 => { // Reset Tx Interrupt Pending (RES_Tx_P)
                    // Re-arms the Tx interrupt latch so the next transmit
                    // opportunity will fire it again.
                    self.tx_int_pending = false;
                },
                6 => { // Error Reset (ERR_RES) — clears error bits in RR1
                },
                7 => { // Reset Highest IUS (RES_H_IUS)
                },
                _ => {},
            }
        }
        self.update_ip();
        wr2_update
    }

    pub fn get_ip(&self) -> u8 {
        let wr1 = self.regs[1];
        let mut ip = 0;

        // External/Status interrupt (bit 0 in the local IP group).
        if (wr1 & wr1::EXT_INT_EN) != 0 && self.ext_pending {
            ip |= 1 << 0;
        }

        // Rx Interrupt (Bit 2 in local IP group)
        // Check if Rx Interrupts enabled (WR1 bits 4,3 != 00)
        if (wr1 & 0x18) != 0 && !self.rx_queue.is_empty() {
            ip |= 1 << 2;
        }

        // Tx Interrupt (Bit 1 in local IP group)
        // Check if Tx Interrupt enabled (WR1 bit 1)
        // Real Z85C30: Tx IUS fires once on buffer-empty, must be reset by
        // RES_Tx_P before re-arming.  Gate on tx_int_pending so IRQ line
        // doesn't stay asserted continuously when Tx FIFO is idle.
        if (wr1 & wr1::TX_INT_EN) != 0 && self.tx_int_pending {
            ip |= 1 << 1;
        }

        ip
    }

    pub fn update_ip(&mut self) {
        let ip = self.get_ip();
        self.ip_num.store(ip, Ordering::SeqCst);
        
        if let Some(line) = &self.irq {
            let other = self.ip_other.load(Ordering::SeqCst);
            line.set((ip | other) != 0);
        }
    }

    pub fn read_data(&mut self) -> u8 {
        if !self.rx_queue.is_empty() {
            let val = self.rx_queue.pop_front().unwrap();
            if self.rx_queue.is_empty() {
                self.status &= !rr0::RX_CHAR_AVAILABLE;
                self.update_ip();
            }
            val
        } else {
            crate::dlog_dev!(LogModule::Scc, "SCC({}): Read Data EMPTY! (Queue len {})", self.name, self.rx_queue.len());
            0
        }
    }

    pub fn write_data(&mut self, val: u8) {
        crate::dlog_dev!(LogModule::Scc, "SCC({}): Write Data {:02x} (Queue len {})", self.name, val, self.tx_queue.len());
        // Push to TX queue (capacity 4)
        if self.tx_queue.len() >= 4 {
            self.tx_queue.pop_front();
        }
        self.tx_queue.push_back(val);

        // A new write re-arms the Tx interrupt latch so it fires again when
        // this char completes (TX_BUFFER_EMPTY goes high once the shift
        // register takes it and notify_tx_empty runs).
        self.tx_int_pending = false;

        // Update status to indicate Tx Buffer Full (clearing Empty bit) only if full
        if self.tx_queue.len() >= 4 {
            self.status &= !rr0::TX_BUFFER_EMPTY;
        }
        self.update_ip();
    }

    /// Called when a character has gone out.
    /// Sets the Tx-interrupt-pending latch so the driver is notified that the
    /// buffer is ready for the next character.
    pub fn notify_tx_empty(&mut self) {
        self.tx_int_pending = true;
        self.update_ip();
    }
}

#[derive(Clone)]
pub struct Z85c30 {
    pub channel_a: Arc<Mutex<Channel>>,
    pub channel_b: Arc<Mutex<Channel>>,
    /// Each channel's transmitter, paced on the machine clock.
    tx: [Arc<Tx>; 2],
    // Swappable so CI mode can replace the default TCP backend with a
    // `CiSerialBackend` before `start()` is called. Wrapped in `Arc<Mutex<_>>`
    // so `Z85c30` stays `Clone` and the swap is thread-safe.
    backend_a: Arc<Mutex<Arc<dyn SerialBackend>>>,
    backend_b: Arc<Mutex<Arc<dyn SerialBackend>>>,
    // Bounded mirror of channel-B (IRIX serial console) guest output. Lets an
    // in-process reader — the GUI's network probe — scrape the console without
    // opening the 8881 TCP socket. Oldest bytes drop once full.
    console_buf: Arc<Mutex<VecDeque<u8>>>,
    /// Each channel's receiver, paced on the machine clock.
    rx: [Arc<Rx>; 2],
    running: Arc<AtomicBool>,
    /// The host-IO thread's passes over each channel, while running.
    pumps: Arc<Mutex<Vec<HostPump>>>,
}

/// One channel's transmit path: the shift register's timer and where the
/// character goes when it is up.
struct Tx {
    name: &'static str,
    channel: Arc<Mutex<Channel>>,
    backend: Arc<Mutex<Arc<dyn SerialBackend>>>,
    /// Channel B (the IRIX serial console) mirrors its output here.
    tap: Option<Arc<Mutex<VecDeque<u8>>>>,
    timer: Timer,
}

impl Tx {
    /// Start the next queued character if the transmitter is on and free. From
    /// `finish` it follows on from `tx_clock`; otherwise the line was idle.
    fn kick(&self, ch: &mut Channel, follows: bool) {
        if ch.tx_shift.is_some() || ch.regs[scc_regs::WR5 as usize] & wr5::TX_ENABLE == 0 {
            return;
        }
        let Some(byte) = ch.tx_queue.pop_front() else { return };
        ch.tx_shift = Some(byte);
        // The holding register is free: the driver may queue another, but the
        // Tx interrupt waits for the character to finish (notify_tx_empty).
        if ch.tx_queue.len() < 4 {
            ch.status |= rr0::TX_BUFFER_EMPTY;
        }
        let start = if follows { ch.tx_clock } else { ch.tx_clock.max(self.timer.now()) };
        ch.tx_clock = start + ch.tx_delay * 1000;
        self.timer.set_due(Some(ch.tx_clock));
    }

    /// The character's time is up: send it, tell the driver, start the next.
    fn finish(&self) {
        let Some(byte) = self.channel.lock().tx_shift.take() else { return };
        crate::dlog_dev!(LogModule::Scc, "SCC: TX({}) '{}' ({:02x})", self.name, if byte.is_ascii_graphic() { byte as char } else { '.' }, byte);
        if let Some(tap) = &self.tap {
            let mut b = tap.lock();
            while b.len() >= CONSOLE_TAP_CAP { b.pop_front(); }
            b.push_back(byte);
        }
        self.backend.lock().clone().send_byte(byte);
        let mut ch = self.channel.lock();
        ch.notify_tx_empty();
        self.kick(&mut ch, true);
    }
}

/// One channel's receive path: what the host typed waits here, and reaches the FIFO a character
/// time apart on the machine clock.
struct Rx {
    channel: Arc<Mutex<Channel>>,
    backend: Arc<Mutex<Arc<dyn SerialBackend>>>,
    state: Mutex<RxState>,
    timer: Timer,
}

struct RxState {
    /// Taken from the host end, not yet in the FIFO; a full FIFO holds bytes here, not drops them.
    held: VecDeque<u8>,
    /// When the next character may arrive, in machine ns.
    clock: u64,
}

/// A long paste stays in the host end until the held bytes drain.
const RX_HOLD: usize = 1024;
/// How long the receiver waits for room in a full FIFO before looking again, in machine ns.
const RX_FULL_RETRY_NS: u64 = 1_000_000;

impl Rx {
    /// The host end's turn: write out its output, follow its modem lines, take what it typed.
    fn pump(&self) -> bool {
        let backend = self.backend.lock().clone();
        backend.flush();
        self.channel.lock().apply_modem_in(backend.modem_in());
        let mut st = self.state.lock();
        let before = st.held.len();
        while st.held.len() < RX_HOLD {
            let Ok(b) = backend.recv_byte() else { break };
            st.held.push_back(b);
        }
        self.deliver(&mut st) | (st.held.len() > before)
    }

    /// Move held bytes into the FIFO a character time apart, and arm the timer for the next one.
    fn deliver(&self, st: &mut RxState) -> bool {
        // A device's bytes go to the guest as they are; only a console gets the ^E conversion.
        let raw = self.backend.lock().raw();
        let now = self.timer.now();
        let (mut due, mut moved) = (None, false);
        while let Some(&byte) = st.held.front() {
            if st.clock > now {
                due = Some(st.clock);
                break;
            }
            let mut ch = self.channel.lock();
            let rx_on = ch.regs[scc_regs::WR3 as usize] & wr3::RX_ENABLE != 0;
            if rx_on && ch.rx_queue.len() >= 8 {
                due = Some(now + RX_FULL_RETRY_NS);
                break;
            }
            st.held.pop_front();
            // With RX off the byte is dropped, as on the chip.
            if !rx_on {
                continue;
            }
            let byte = if byte == 0x05 && !raw { 0x04 } else { byte };
            crate::dlog_dev!(LogModule::Scc, "SCC: RX({}) '{}' ({:02x})", ch.name, if byte.is_ascii_graphic() { byte as char } else { '.' }, byte);
            ch.rx_queue.push_back(byte);
            ch.status |= rr0::RX_CHAR_AVAILABLE;
            ch.update_ip();
            // A burst stays on its baud grid; a line idle for 100 ms starts afresh.
            if now.saturating_sub(st.clock) > 100_000_000 {
                st.clock = now;
            }
            st.clock += ch.tx_delay * 1000;
            moved = true;
        }
        self.timer.set_due(due);
        moved
    }
}

/// Cap on the channel-B console mirror (`console_buf`). 64 KiB is far more than
/// any single probe response; the buffer is drained on each read.
const CONSOLE_TAP_CAP: usize = 64 * 1024;

/// Loopback TCP ports of the serial channels unless the config says otherwise:
/// channel A (tty2) on 8880, channel B (tty1, the console) on 8881.
pub const DEFAULT_PORTS: [u16; 2] = [8880, 8881];

impl Z85c30 {
    /// A controller whose channels are wired to nothing: the machine installs
    /// each channel's host end with `set_backend_a` / `set_backend_b` before
    /// the first `start()`.
    pub fn new(irq: Option<IrqLine>, tm: &Arc<TimerManager>) -> Self {
        let ip_a = Arc::new(AtomicU8::new(0));
        let ip_b = Arc::new(AtomicU8::new(0));
        let (backend_a, backend_b): (Arc<dyn SerialBackend>, Arc<dyn SerialBackend>) =
            (Arc::new(NullBackend), Arc::new(NullBackend));

        let channel_a = Arc::new(Mutex::new(Channel::new("A", ip_a.clone(), ip_b.clone(), irq.clone())));
        // Note: Channel B gets ip_b as its 'num' and ip_a as 'other'
        let channel_b = Arc::new(Mutex::new(Channel::new("B", ip_b, ip_a, irq)));
        let backend_a = Arc::new(Mutex::new(backend_a));
        let backend_b = Arc::new(Mutex::new(backend_b));
        let console_buf = Arc::new(Mutex::new(VecDeque::new()));
        let tx = |name, channel: &Arc<Mutex<Channel>>, backend: &Arc<Mutex<Arc<dyn SerialBackend>>>, tap| {
            Arc::new_cyclic(|me| Tx {
                name,
                channel: channel.clone(),
                backend: backend.clone(),
                tap,
                timer: Timer::new(tm, me, |t: &Tx, _| t.finish()),
            })
        };
        let rx = |channel: &Arc<Mutex<Channel>>, backend: &Arc<Mutex<Arc<dyn SerialBackend>>>| {
            Arc::new_cyclic(|me| Rx {
                channel: channel.clone(),
                backend: backend.clone(),
                state: Mutex::new(RxState { held: VecDeque::new(), clock: 0 }),
                timer: Timer::new(tm, me, |r: &Rx, _| _ = r.deliver(&mut r.state.lock())),
            })
        };
        Self {
            rx: [rx(&channel_a, &backend_a), rx(&channel_b, &backend_b)],
            tx: [tx("A", &channel_a, &backend_a, None), tx("B", &channel_b, &backend_b, Some(console_buf.clone()))],
            channel_a,
            channel_b,
            backend_a,
            backend_b,
            console_buf,
            running: Arc::new(AtomicBool::new(false)),
            pumps: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Swap in an alternate backend for channel A (tty2 on Indy).
    /// Must be called before `start()`.
    pub fn set_backend_a(&self, backend: Arc<dyn SerialBackend>) {
        self.rx[0].timer.manager().inspect(|tm| backend.bind_timers(tm));
        *self.backend_a.lock() = backend;
    }

    /// Swap in an alternate backend for channel B (tty1, the PROM/IRIX
    /// serial console on Indy). Same constraint as `set_backend_a`.
    pub fn set_backend_b(&self, backend: Arc<dyn SerialBackend>) {
        self.rx[1].timer.manager().inspect(|tm| backend.bind_timers(tm));
        *self.backend_b.lock() = backend;
    }

    /// Read the currently-installed backend for channel B.  Used by
    /// non-CI `--serial-log` wiring to wrap the live TCP backend in a
    /// `TeeBackend` without replacing the listener.
    pub fn backend_b(&self) -> Arc<dyn SerialBackend> {
        self.backend_b.lock().clone()
    }

    /// Queue host bytes to be delivered to channel B (tty1, the IRIX serial
    /// console) as if typed at the console — entirely in-process, with no
    /// loopback TCP client. The bytes ride the same RX path as socket input
    /// (FIFO backpressure + baud pacing), so the guest sees them identically.
    /// Used by the GUI's "Send IRIX halt" for a clean shutdown that doesn't
    /// depend on the serial server socket.
    pub fn inject_b(&self, data: &[u8]) {
        self.rx[1].state.lock().held.extend(data);
        host_io().wake.notify();
    }

    /// Drain and return the channel-B (IRIX serial console) output captured
    /// since the last call. Empties the mirror buffer. Used by the GUI network
    /// probe to read the guest's response to an injected command in-process.
    pub fn drain_console(&self) -> Vec<u8> {
        self.console_buf.lock().drain(..).collect()
    }

    pub fn read_a_control(&self) -> u8 { 
        let mut a = self.channel_a.lock();
        if a.reg_ptr == 2 {
            // RR2 on channel A: the interrupt vector as WR2 wrote it. The
            // status-modified vector (WR9 VIS) belongs to channel B's RR2.
            let val = a.regs[2];
            a.reg_ptr = 0;
            return val;
        }
        if a.reg_ptr == 3 {
            // RR3: IP A (5,4,3) + IP B (2,1,0)
            let ip_a = a.get_ip();
            
            // Use atomic for B to avoid locking
            let ip_b = a.ip_other.load(Ordering::SeqCst);
            a.reg_ptr = 0;
            // Shift A's IP to bits 5,4,3
            return (ip_a << 3) | ip_b;
        }
        a.read_control() 
    }
    pub fn write_a_control(&self, val: u8) { 
        let change = {
            let mut a = self.channel_a.lock();
            let mut b = self.channel_b.lock();

            let update = a.write_control(val);
            self.tx[0].kick(&mut a, false);
            if let Some(wr2) = update {
                b.regs[2] = wr2;
            }
            a.line_change()
        };
        if let Some(line) = change {
            Self::send_line(&self.backend_a, "A", line);
        }
    }
    pub fn read_a_data(&self) -> u8 { self.channel_a.lock().read_data() }
    pub fn write_a_data(&self, val: u8) { 
        let mut lock = self.channel_a.lock();
        lock.write_data(val);
        self.tx[0].kick(&mut lock, false);
    }

    pub fn read_b_control(&self) -> u8 { 
        let mut b = self.channel_b.lock();
        if b.reg_ptr == 2 {
            // RR2 on channel B should be the vector modified by the
            // highest pending interrupt when WR9's VIS bit is set. Not
            // modelled: it returns the stored WR2 value unmodified.
            let val = b.regs[2];
            b.reg_ptr = 0;
            return val;
        }
        if b.reg_ptr == 3 {
            b.reg_ptr = 0;
            return 0;
        }
        b.read_control() 
    }
    pub fn write_b_control(&self, val: u8) { 
        let change = {
            let mut a = self.channel_a.lock();
            let mut b = self.channel_b.lock();

            let update = b.write_control(val);
            self.tx[1].kick(&mut b, false);
            if let Some(wr2) = update {
                a.regs[2] = wr2;
            }
            b.line_change()
        };
        if let Some(line) = change {
            Self::send_line(&self.backend_b, "B", line);
        }
    }

    /// Tell a channel's backend its line state, with no channel lock held:
    /// a host serial port reprograms itself, which is a system call.
    fn send_line(backend: &Mutex<Arc<dyn SerialBackend>>, name: &str, (line, out): (LineConfig, ModemOut)) {
        crate::dlog_dev!(LogModule::Scc, "SCC({}): line {} DTR={} RTS={} BRK={}", name, line, out.dtr, out.rts, out.brk);
        let b = backend.lock().clone();
        b.set_line(line);
        b.set_modem(out);
    }

    /// Tell both backends the current line state whatever they were told
    /// before: at start, and after a snapshot is loaded.
    fn resend_lines(&self) {
        for (ch, backend, name) in [(&self.channel_a, &self.backend_a, "A"), (&self.channel_b, &self.backend_b, "B")] {
            let line = {
                let mut c = ch.lock();
                let now = c.line_state();
                c.line_sent = Some(now);
                now
            };
            Self::send_line(backend, name, line);
        }
    }
    pub fn read_b_data(&self) -> u8 { self.channel_b.lock().read_data() }
    pub fn write_b_data(&self, val: u8) { 
        let mut lock = self.channel_b.lock();
        lock.write_data(val);
        self.tx[1].kick(&mut lock, false);
    }

    pub fn get_ip(&self) -> u8 {
        // Check both channels for pending interrupts
        // Lock A to get its status and read B's atomic (ip_other on A is B's ip)
        let a = self.channel_a.lock();
        let ip_a = a.get_ip();
        let ip_b = a.ip_other.load(Ordering::SeqCst);
        ip_a | ip_b
    }

    pub fn read(&self, addr: u32) -> BusRead8 {
        let val = match addr {
            0 => BusRead8::ok(self.read_b_control()),
            1 => BusRead8::ok(self.read_b_data()),
            2 => BusRead8::ok(self.read_a_control()),
            3 => BusRead8::ok(self.read_a_data()),
            _ => BusRead8::err(),
        };
        if val.is_ok() {
            crate::dlog_dev!(LogModule::Scc, "SCC({}): Read Reg {} -> {:02x}", if (addr & 2) != 0 { "A" } else { "B" }, addr, val.data);
        }
        val
    }

    pub fn write(&self, addr: u32, val: u8) -> u32 {
        crate::dlog_dev!(LogModule::Scc, "SCC({}): Write Reg {} = {:02x}", if (addr & 2) != 0 { "A" } else { "B" }, addr, val);
        match addr {
            0 => self.write_b_control(val),
            1 => self.write_b_data(val),
            2 => self.write_a_control(val),
            3 => self.write_a_data(val),
            _ => return BUS_ERR,
        }
        BUS_OK
    }
    pub fn register_locks(&self) {
        use crate::locks::{register_mutex, register_mutex_fn};
        register_mutex_fn("scc::channel_a", &self.channel_a, |c| c.is_locked());
        register_mutex_fn("scc::channel_b", &self.channel_b, |c| c.is_locked());
        register_mutex("scc::pumps", &self.pumps);
    }
}

impl Device for Z85c30 {
    fn step(&self, _cycles: u64) {}
    
    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        for t in &self.tx {
            t.timer.stop();
        }
        for r in &self.rx {
            r.timer.stop();
        }
        self.pumps.lock().clear();
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        for t in &self.tx {
            t.timer.start();
        }
        for r in &self.rx {
            r.timer.start();
        }
        self.resend_lines();
        *self.pumps.lock() = self.rx.iter().map(|r| {
            let r = r.clone();
            host_io().add(move || r.pump())
        }).collect();
    }

    fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("serial".to_string(), "serial status  — dump SCC channel A/B registers and FIFO state".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd == "serial" {
            if args.first().copied() == Some("status") || args.is_empty() {
                for (label, arc, backend) in [("A", &self.channel_a, &self.backend_a), ("B", &self.channel_b, &self.backend_b)] {
                    let dropped = backend.lock().dropped();
                    let ch = arc.lock();
                    writeln!(writer, "=== Channel {} ===", label).unwrap();
                    writeln!(writer, "  reg_ptr : {}", ch.reg_ptr).unwrap();
                    writeln!(writer, "  status  : {:02x}  (RX_AVAIL={} TX_EMPTY={} DCD={} CTS={})",
                        ch.status,
                        (ch.status & rr0::RX_CHAR_AVAILABLE) != 0,
                        (ch.status & rr0::TX_BUFFER_EMPTY) != 0,
                        (ch.status & rr0::DCD_STATE) != 0,
                        (ch.status & rr0::CTS_STATE) != 0,
                    ).unwrap();
                    writeln!(writer, "  rx_queue: {} bytes {:?}", ch.rx_queue.len(),
                        ch.rx_queue.iter().copied().collect::<Vec<_>>()).unwrap();
                    writeln!(writer, "  tx_queue: {} bytes {:?}", ch.tx_queue.len(),
                        ch.tx_queue.iter().copied().collect::<Vec<_>>()).unwrap();
                    writeln!(writer, "  tx_delay: {} µs", ch.tx_delay).unwrap();
                    writeln!(writer, "  dropped : {dropped} bytes for clients that did not read").unwrap();
                    for (i, v) in ch.regs.iter().enumerate() {
                        if *v != 0 {
                            writeln!(writer, "  WR{:<2}    : {:02x}", i, v).unwrap();
                        }
                    }
                }
                return Ok(());
            }
            return Err("Usage: serial status".to_string());
        }
        Err("Command not found".to_string())
    }
}

// ============================================================================
// Resettable + Saveable for Z85c30
// ============================================================================

impl Resettable for Z85c30 {
    /// Reset both SCC channels to power-on state.
    /// The backend TcpStream is NOT touched — the host console connection survives.
    /// The receive pumps keep running (they drop input while rx is off).
    fn power_on(&self) {
        for (ch, t) in [(&self.channel_a, &self.tx[0]), (&self.channel_b, &self.tx[1])] {
            ch.lock().reset();
            t.timer.set_due(None);
        }
    }
}

fn channel_to_toml(ch: &Channel) -> toml::Value {
    let mut t = toml::map::Map::new();
    t.insert("regs".into(),    u8_slice_to_toml(&ch.regs));
    t.insert("reg_ptr".into(), hex_u8(ch.reg_ptr));
    t.insert("status".into(),  hex_u8(ch.status));
    toml::Value::Table(t)
}

fn channel_from_toml(v: &toml::Value, ch: &mut Channel) {
    if let Some(r) = get_field(v, "regs")    { load_u8_slice(r, &mut ch.regs); }
    if let Some(x) = get_field(v, "reg_ptr") { if let Some(n) = toml_u8(x) { ch.reg_ptr = n; } }
    if let Some(x) = get_field(v, "status")  { if let Some(n) = toml_u8(x) { ch.status = n; } }
    ch.rx_queue.clear();
    ch.tx_queue.clear();
    ch.tx_shift = None;
    ch.tx_clock = 0;
    // A stale RX_CHAR_AVAILABLE has no setter left over an empty queue: read_data
    // returns 0 without touching it, so a guest polling RR0 spins forever.
    ch.status &= !rr0::RX_CHAR_AVAILABLE;
    ch.status |= rr0::TX_BUFFER_EMPTY;
    // DCD and CTS are the host end's pins now, not what the snapshot saw.
    ch.status = (ch.status & !(rr0::DCD_STATE | rr0::CTS_STATE)) | Channel::modem_bits(ch.modem_in);
    ch.ext_pending = false;
    // notify_tx_empty is the only setter and its character went out with tx_queue.
    // Gated on WR5 as well because the TX thread only ever latches while
    // TX_ENABLE is set, so a latch without it is a state the model cannot reach.
    let wr1_val = ch.regs[scc_regs::WR1 as usize];
    let wr5_val = ch.regs[scc_regs::WR5 as usize];
    ch.tx_int_pending = (wr1_val & wr1::TX_INT_EN) != 0 && (wr5_val & wr5::TX_ENABLE) != 0;
    ch.update_tx_delay();
    // Ioc::load_state restores map_stat wholesale after power_on_devices cleared
    // SERIAL, so the SCC redrives the line last.
    ch.update_ip();
}

impl Saveable for Z85c30 {
    fn save_state(&self) -> toml::Value {
        let mut tbl = toml::map::Map::new();
        tbl.insert("ch_a".into(), channel_to_toml(&self.channel_a.lock()));
        tbl.insert("ch_b".into(), channel_to_toml(&self.channel_b.lock()));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(ca) = get_field(v, "ch_a") {
            channel_from_toml(ca, &mut self.channel_a.lock());
            self.tx[0].timer.set_due(None);
        }
        if let Some(cb) = get_field(v, "ch_b") {
            channel_from_toml(cb, &mut self.channel_b.lock());
            self.tx[1].timer.set_due(None);
        }
        // The host end keeps its own state through a load: tell it the restored line.
        self.resend_lines();
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::dev::serial_host::{CiSerialBackend, TcpSocketBackend};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Two clients on one TCP serial console: both see every byte the guest
    /// sends, and what either types arrives, with telnet's own commands
    /// taken out on the way.
    #[test]
    fn a_tcp_console_takes_more_than_one_client() {
        use std::io::{Read, Write};
        let b = TcpSocketBackend::new(("127.0.0.1", 0)).expect("bind");
        let port = b.listener.local_addr().unwrap().port();
        let connect = || {
            let c = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
            c.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
            c
        };
        let (mut one, mut two) = (connect(), connect());
        // Accept both (recv_byte takes waiting clients), then read their handshakes.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let hs = crate::net::telnet::TelnetFilter::initial_handshake();
        let mut got = vec![0u8; hs.len()];
        let _ = b.recv_byte();
        one.read_exact(&mut got).expect("first handshake");
        while std::time::Instant::now() < until {
            let _ = b.recv_byte();
            two.set_nonblocking(true).unwrap();
            let mut probe = [0u8; 1];
            let ready = matches!(two.peek(&mut probe), Ok(1));
            two.set_nonblocking(false).unwrap();
            if ready {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        two.read_exact(&mut got).expect("second handshake");
        for &c in b"ok" {
            b.send_byte(c);
        }
        b.flush();
        let mut out = [0u8; 2];
        one.read_exact(&mut out).unwrap();
        assert_eq!(&out, b"ok");
        two.read_exact(&mut out).unwrap();
        assert_eq!(&out, b"ok");
        // Each types; IAC DO SGA in between is not input.
        one.write_all(&[b'a', 0xff, 0xfd, 0x03]).unwrap();
        two.write_all(b"b").unwrap();
        let mut typed = Vec::new();
        while typed.len() < 2 && std::time::Instant::now() < until {
            if let Ok(c) = b.recv_byte() {
                typed.push(c);
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
        typed.sort();
        assert_eq!(typed, b"ab");
    }

    /// Phase 1.7 round-trip: a fresh SCC loaded from a captured save_state must
    /// re-serialize byte-identically. Use new so the test doesn't bind any
    /// TCP ports.
    ///
    /// The `status` fixtures below are already normalized against empty FIFOs
    /// (`TX_BUFFER_EMPTY` set, `RX_CHAR_AVAILABLE` clear) and carry DCD and
    /// CTS as the idle host end shows them, because restore is deliberately
    /// not an identity on that field; do not "fix" them back.
    #[test]
    fn save_load_round_trip() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        {
            let mut ch = src.channel_a.lock();
            ch.regs[0]  = 0x44;
            ch.regs[1]  = 0x12;
            ch.regs[3]  = 0xc1;
            ch.regs[5]  = 0xea;
            ch.reg_ptr  = 7;
            ch.status   = 0x6c;
        }
        {
            let mut ch = src.channel_b.lock();
            ch.regs[0]  = 0x88;
            ch.regs[2]  = 0x10;
            ch.regs[15] = 0x05;
            ch.reg_ptr  = 3;
            ch.status   = 0xac;
        }
        let v1 = src.save_state();

        let dst = Z85c30::new(None, &Arc::new(TimerManager::new()));
        dst.load_state(&v1).expect("load_state");
        let v2 = dst.save_state();

        assert_eq!(v1, v2, "Z85c30 save_state mismatch after load_state round-trip");
    }

    /// Phase 3.5: a long single-line `serial-send` from the host must arrive
    /// at the guest's tty intact. Before the rx-thread fix, bytes 9..N of any
    /// burst were silently dropped when SCC's 8-byte rx_queue filled — a 53-
    /// char `dd if=/dev/rdsk/dks0d2s0 of=/tmp/r.bin bs=512 count=1\r` arrived
    /// at IRIX as `dd if=/d=512 count=1`, causing CI scripts to fabricate
    /// shell errors out of thin air. This test pushes that exact line through
    /// the loopback CiSerialBackend, drains the SCC rx_queue at the rate the
    /// IRIX kernel would (one byte at a time, polled), and asserts every
    /// byte arrives.
    #[test]
    fn long_input_round_trips_without_loss() {
        use std::sync::Arc;

        let tm = Arc::new(TimerManager::new());
        let scc = Z85c30::new(None, &tm);
        let backend = Arc::new(CiSerialBackend::new());
        scc.set_backend_a(backend.clone());

        // Enable RX on channel A so the receiver queues bytes. tx_delay is
        // tx-direction baud-rate emulation; set a small value so the test
        // doesn't pay 19.2 kbaud-per-char latency.
        {
            let mut ch = scc.channel_a.lock();
            ch.regs[scc_regs::WR3 as usize] |= wr3::RX_ENABLE;
            ch.tx_delay = 50; // 50 µs/byte
        }

        scc.start();

        let line = b"dd if=/dev/rdsk/dks0d2s0 of=/tmp/r.bin bs=512 count=1\r";
        backend.push_host(line);

        // Drain rx_queue at ~20 kHz so the receiver always has space to
        // push pending bytes. Mirrors how IRIX's tty driver consumes
        // RR0::RX_CHAR_AVAILABLE.
        let mut received = Vec::with_capacity(line.len());
        let deadline = Instant::now() + Duration::from_secs(5);
        while received.len() < line.len() && Instant::now() < deadline {
            let popped = {
                let mut ch = scc.channel_a.lock();
                ch.rx_queue.pop_front()
            };
            match popped {
                Some(b) => received.push(b),
                None    => std::thread::sleep(Duration::from_micros(50)),
            }
        }

        scc.stop();

        assert_eq!(received.len(), line.len(),
            "expected {} bytes, got {} (lossy rx_queue?)", line.len(), received.len());
        assert_eq!(&received, line, "byte content mismatch — bytes dropped or reordered");
    }

    #[test]
    fn restore_clears_rx_char_available() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        src.channel_a.lock().status |= rr0::RX_CHAR_AVAILABLE;
        let saved = src.save_state();

        let dst = Z85c30::new(None, &Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load_state");

        assert_eq!(dst.channel_a.lock().status & rr0::RX_CHAR_AVAILABLE, 0,
            "restore emptied rx_queue, and `read_data` leaves the bit alone on an \
             empty queue, so a guest polling RR0 would never see it drop");
    }

    #[test]
    fn restore_forces_tx_buffer_empty() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        src.channel_a.lock().status &= !rr0::TX_BUFFER_EMPTY;
        let saved = src.save_state();

        let dst = Z85c30::new(None, &Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load_state");

        assert_ne!(dst.channel_a.lock().status & rr0::TX_BUFFER_EMPTY, 0,
            "a guest that gates its write on TX_BUFFER_EMPTY would wait on a \
             transmitter whose queue restore already emptied");
    }

    #[test]
    fn restore_rearms_tx_int_pending() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        {
            let mut ch = src.channel_a.lock();
            ch.regs[scc_regs::WR1 as usize] = wr1::TX_INT_EN;
            ch.regs[scc_regs::WR5 as usize] = wr5::TX_ENABLE;
        }
        let saved = src.save_state();

        let dst = Z85c30::new(None, &Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load_state");

        assert!(dst.channel_a.lock().tx_int_pending,
            "clearing tx_queue destroyed the character whose completion would \
             have driven the only setter, `notify_tx_empty`");
    }

    /// Mutation guard. The assertion has to be on the field, not on RR3:
    /// `get_ip` gates the Tx bit on WR1 independently, so an RR3 assertion
    /// stays green even with the WR1 conjunct deleted from `channel_from_toml`.
    #[test]
    fn restore_leaves_latch_disarmed_when_tx_int_disabled() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        src.channel_a.lock().regs[scc_regs::WR5 as usize] = wr5::TX_ENABLE;
        let saved = src.save_state();

        let dst = Z85c30::new(None, &Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load_state");

        assert!(!dst.channel_a.lock().tx_int_pending,
            "WR1 has TX_INT_EN clear, so nothing should be latched");
    }

    /// The other half of the gate. `get_ip` ignores WR5, so this is unobservable
    /// through RR3 and has to assert the latch itself.
    #[test]
    fn restore_leaves_latch_disarmed_when_transmitter_disabled() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        src.channel_a.lock().regs[scc_regs::WR1 as usize] = wr1::TX_INT_EN;
        let saved = src.save_state();

        let dst = Z85c30::new(None, &Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load_state");

        assert!(!dst.channel_a.lock().tx_int_pending,
            "WR5 has TX_ENABLE clear, so the TX thread could never have latched");
    }

    /// `read_b_control` returns 0 for RR3, so B's interrupt reaches the guest
    /// only through A's RR3 via the shared `ip_other` atomic that `update_ip`
    /// publishes. `machine.rs` puts the CI console on B.
    #[test]
    fn restore_channel_b_tx_int_reaches_rr3_via_a() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        {
            let mut ch = src.channel_b.lock();
            ch.regs[scc_regs::WR1 as usize] = wr1::TX_INT_EN;
            ch.regs[scc_regs::WR5 as usize] = wr5::TX_ENABLE;
        }
        let saved = src.save_state();

        let dst = Z85c30::new(None, &Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load_state");

        dst.channel_a.lock().reg_ptr = scc_regs::RR3;
        assert_eq!(dst.read_a_control() & (1 << 1), 1 << 1,
            "B's Tx IP occupies bits 2..0 of A's RR3");
    }

    #[test]
    fn restore_redrives_irq_line() {
        let src = Z85c30::new(None, &Arc::new(TimerManager::new()));
        {
            let mut ch = src.channel_a.lock();
            ch.regs[scc_regs::WR1 as usize] = wr1::TX_INT_EN;
            ch.regs[scc_regs::WR5 as usize] = wr5::TX_ENABLE;
        }
        let saved = src.save_state();

        let (probe, line) = crate::irq::LevelProbe::line();
        let dst = Z85c30::new(Some(line), &Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load_state");

        assert!(probe.level(),
            "`power_on_devices` cleared map_stat SERIAL and `Ioc::load_state` \
             restored it wholesale, so only the SCC can put the line back up");
    }

    /// A character leaves the shift register when the machine clock reaches
    /// the end of its time on the line, paced back to back with the next; a
    /// stop keeps the deadline, and a load drops the character in flight.
    /// A host end that records the line state it is told and shows the
    /// modem inputs a test sets: a stand-in for a host serial port.
    struct LineProbe {
        lines: Mutex<Vec<LineConfig>>,
        outs: Mutex<Vec<ModemOut>>,
        input: Mutex<ModemIn>,
        rx: Mutex<VecDeque<u8>>,
    }

    impl LineProbe {
        fn new() -> Arc<Self> {
            Arc::new(Self { lines: Mutex::new(vec![]), outs: Mutex::new(vec![]), input: Mutex::new(ModemIn::IDLE), rx: Mutex::new(VecDeque::new()) })
        }
    }

    impl SerialBackend for LineProbe {
        fn send_byte(&self, _: u8) {}
        fn recv_byte(&self) -> io::Result<u8> {
            self.rx.lock().pop_front().ok_or_else(|| io::ErrorKind::WouldBlock.into())
        }
        fn set_line(&self, l: LineConfig) {
            self.lines.lock().push(l);
        }
        fn set_modem(&self, o: ModemOut) {
            self.outs.lock().push(o);
        }
        fn modem_in(&self) -> ModemIn {
            *self.input.lock()
        }
        fn raw(&self) -> bool {
            true
        }
    }

    fn wr_a(scc: &Z85c30, reg: u8, val: u8) {
        if reg >= 8 {
            scc.write_a_control(0x08 | (reg - 8));
        } else {
            scc.write_a_control(reg);
        }
        scc.write_a_control(val);
    }

    /// IRIX's 9600 8N1 setup: the baud-rate generator fed from the 3.672 MHz
    /// RTxC clock, x16, time constant 10. The rate it gives is 9562, which
    /// a host port rounds to 9600.
    #[test]
    fn line_state_follows_the_clock_irix_programs() {
        let clock = crate::test_board::Stepped::new();
        let scc = Z85c30::new(None, &clock.tm);
        for (reg, val) in [(4, 0x44), (3, 0xc1), (5, 0xea), (11, 0x50), (12, 10), (13, 0), (14, 0x01)] {
            wr_a(&scc, reg, val);
        }
        assert_eq!(Z85c30::new(None, &clock.tm).channel_a.lock().line_state().0.baud, None, "after reset: sync mode, no rate");
        let (line, out) = scc.channel_a.lock().line_state();
        assert_eq!(line, LineConfig { baud: Some(9562), data_bits: 8, parity: Parity::None, stop_bits: 1 });
        assert_eq!(out, ModemOut { dtr: true, rts: true, brk: false });
        // The generator off: no rate the emulator can name.
        wr_a(&scc, 14, 0x00);
        assert_eq!(scc.channel_a.lock().line_state().0.baud, None);
        // A Macintosh MIDI interface: the TRxC pin, x32, an external clock.
        wr_a(&scc, 11, 0x28);
        wr_a(&scc, 4, 0x84);
        assert_eq!(scc.channel_a.lock().line_state().0.baud, None);
        // The generator from PCLK: 10 MHz / (2 * 34 * 16) = 9191.
        for (reg, val) in [(11, 0x50), (4, 0x44), (12, 32), (14, 0x03)] {
            wr_a(&scc, reg, val);
        }
        assert_eq!(scc.channel_a.lock().line_state().0.baud, Some(9191));
        // Odd parity, two stop bits, 7 bits.
        wr_a(&scc, 4, 0x4d);
        wr_a(&scc, 3, 0x41);
        let line = scc.channel_a.lock().line_state().0;
        assert_eq!((line.data_bits, line.parity, line.stop_bits), (7, Parity::Odd, 2));
    }

    /// Register writes reach the backend, each change once, and start()
    /// tells it the state it begins in.
    #[test]
    fn the_backend_hears_the_guests_line_and_modem_lines() {
        let clock = crate::test_board::Stepped::new();
        let scc = Z85c30::new(None, &clock.tm);
        let probe = LineProbe::new();
        scc.set_backend_a(probe.clone());
        scc.start();
        assert_eq!(probe.outs.lock().last().copied(), Some(ModemOut::default()), "start: DTR and RTS low, as after reset");
        let told = probe.outs.lock().len();
        wr_a(&scc, 5, 0x82);
        assert_eq!(probe.outs.lock().last().copied(), Some(ModemOut { dtr: true, rts: true, brk: false }));
        assert_eq!(probe.outs.lock().len(), told + 1);
        wr_a(&scc, 5, 0x82);
        assert_eq!(probe.outs.lock().len(), told + 1, "an unchanged write tells nothing");
        wr_a(&scc, 5, 0x92);
        assert!(probe.outs.lock().last().unwrap().brk, "break");
        for (reg, val) in [(4, 0x44), (11, 0x50), (12, 10), (14, 0x01)] {
            wr_a(&scc, reg, val);
        }
        assert_eq!(probe.lines.lock().last().unwrap().baud, Some(9562));
        scc.stop();
    }

    /// CTS and DCD come from the backend: RR0 shows them, and a change WR15
    /// watches raises the external/status interrupt until the guest resets it.
    #[test]
    fn modem_inputs_show_in_rr0_and_interrupt() {
        let clock = crate::test_board::Stepped::new();
        let scc = Z85c30::new(None, &clock.tm);
        wr_a(&scc, 15, wr15::CTS_IE);
        wr_a(&scc, 1, wr1::EXT_INT_EN);
        let mut a = scc.channel_a.lock();
        assert_ne!(a.status & rr0::CTS_STATE, 0);
        a.apply_modem_in(ModemIn { cts: false, ..ModemIn::IDLE });
        assert_eq!(a.status & rr0::CTS_STATE, 0, "CTS dropped");
        assert_eq!(a.get_ip() & 1, 1, "external/status interrupt");
        a.write_control(0x10); // Reset Ext/Status Interrupts
        assert_eq!(a.get_ip() & 1, 0);
        a.apply_modem_in(ModemIn { cts: false, dcd: false, ..ModemIn::IDLE });
        assert_eq!(a.status & rr0::DCD_STATE, 0, "DCD shows");
        assert_eq!(a.get_ip() & 1, 0, "WR15 does not watch DCD");
        a.write_control(0x18); // Channel Reset
        assert_eq!(a.status & (rr0::CTS_STATE | rr0::DCD_STATE), 0, "a reset keeps showing the pins");
    }

    /// A device's bytes reach the guest unchanged: 0x05 is a console's ^E
    /// only on a line a person types at.
    #[test]
    fn a_raw_line_passes_0x05_through() {
        let clock = crate::test_board::Stepped::new();
        let scc = Z85c30::new(None, &clock.tm);
        let probe = LineProbe::new();
        probe.rx.lock().extend([0x05u8, 0x41]);
        scc.set_backend_a(probe.clone());
        wr_a(&scc, 3, 0xc1);
        scc.channel_a.lock().tx_delay = 50;
        scc.start();
        assert!(crate::test_board::until(|| scc.rx[0].timer.due().is_some()), "the first byte arrived");
        clock.step(50_000);
        scc.stop();
        assert_eq!(scc.channel_a.lock().rx_queue.iter().copied().collect::<Vec<_>>(), [0x05, 0x41]);
    }

    /// Host input reaches the FIFO a character time apart on the machine clock, not on host time.
    #[test]
    fn receive_is_paced_by_a_timer_on_the_machine_clock() {
        let clock = crate::test_board::Stepped::new();
        let scc = Z85c30::new(None, &clock.tm);
        let probe = LineProbe::new();
        probe.rx.lock().extend([1u8, 2]);
        scc.set_backend_a(probe.clone());
        wr_a(&scc, 3, 0xc1);
        scc.channel_a.lock().tx_delay = 50;
        scc.start();
        assert!(crate::test_board::until(|| scc.rx[0].timer.due().is_some()));
        assert_eq!(scc.channel_a.lock().rx_queue.len(), 1);
        clock.step(49_999);
        assert_eq!(scc.channel_a.lock().rx_queue.len(), 1, "not before its time is up");
        clock.step(1);
        assert_eq!(scc.channel_a.lock().rx_queue.len(), 2);
        scc.stop();
    }

    /// Bytes injected in-process reach channel B's FIFO with no host end attached.
    #[test]
    fn injected_bytes_reach_channel_b() {
        let tm = Arc::new(TimerManager::new());
        let scc = Z85c30::new(None, &tm);
        scc.write_b_control(scc_regs::WR3);
        scc.write_b_control(0xc1);
        scc.start();
        scc.inject_b(b"h");
        assert!(crate::test_board::until(|| scc.channel_b.lock().rx_queue.len() == 1));
        scc.stop();
    }

    #[test]
    fn transmit_is_paced_by_a_timer_on_the_machine_clock() {
        let clock = crate::test_board::Stepped::new();
        let scc = Z85c30::new(None, &clock.tm);
        let out = Arc::new(CiSerialBackend::new());
        scc.set_backend_b(out.clone());
        scc.write_b_control(scc_regs::WR5);
        scc.write_b_control(wr5::TX_ENABLE);
        scc.start();
        scc.write_b_data(b'a');
        let due = scc.tx[1].timer.due().expect("the character is on the line");
        clock.step(due - 1);
        assert!(out.drain_guest().is_empty(), "not before its time is up");
        clock.step(1);
        assert_eq!(out.drain_guest().len(), 1, "the timer sent it");
        scc.write_b_data(b'b');
        scc.stop();
        clock.step(due);
        assert!(out.drain_guest().is_empty() && scc.tx[1].timer.due().is_some(), "the SCC's stop stops its timer");
        scc.load_state(&scc.save_state()).unwrap();
        assert_eq!(scc.tx[1].timer.due(), None, "a load drops the character in flight");
        scc.stop();
    }

    /// A character the timer sends late does not push the next one back: a
    /// burst stays on the baud grid.
    #[test]
    fn a_late_timer_does_not_slow_a_burst() {
        let clock = crate::test_board::Stepped::new();
        let scc = Z85c30::new(None, &clock.tm);
        scc.set_backend_b(Arc::new(CiSerialBackend::new()));
        scc.write_b_control(scc_regs::WR5);
        scc.write_b_control(wr5::TX_ENABLE);
        scc.start();
        scc.write_b_data(b'a');
        scc.write_b_data(b'b');
        let first = scc.tx[1].timer.due().unwrap();
        let char_ns = scc.channel_b.lock().tx_delay * 1000;
        clock.step(first + char_ns / 2 - clock.now());
        assert_eq!(scc.tx[1].timer.due(), Some(first + char_ns),
            "the second character ends one character time after the first, not after the late fire");
        scc.stop();
    }

    struct Scc(Z85c30, #[allow(dead_code)] Arc<TimerManager>);

    impl crate::dev::serial_host::tests::Port for Scc {
        fn write(&mut self, b: u8) {
            self.0.write_b_data(b);
        }
        fn tx_ready(&mut self) -> bool {
            self.0.read_b_control() & rr0::TX_BUFFER_EMPTY != 0
        }
        fn poll(&mut self) {
            self.0.rx[1].pump();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    impl Drop for Scc {
        fn drop(&mut self) {
            self.0.stop();
        }
    }

    /// Channel B, the console, on `backend` with its transmitter on, for the
    /// shared serial contract.
    pub(crate) fn contract_port(backend: Arc<dyn SerialBackend>) -> Box<dyn crate::dev::serial_host::tests::Port> {
        let tm = Arc::new(TimerManager::new());
        let s = Z85c30::new(None, &tm);
        s.set_backend_b(backend);
        s.write_b_control(scc_regs::WR5);
        s.write_b_control(wr5::TX_ENABLE);
        s.start();
        Box::new(Scc(s, tm))
    }
}
