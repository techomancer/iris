use std::sync::Arc;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU8, AtomicBool, Ordering};
use std::collections::VecDeque;
use std::io::{self, Write, Read};
use std::fs::OpenOptions;
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};
use std::thread;
use std::io::Write as IoWrite;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, Device, Resettable, Saveable};
use crate::snapshot::{get_field, toml_u8, u8_slice_to_toml, load_u8_slice, hex_u8};
use crate::devlog::LogModule;

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
    pub const LOCAL_LOOPBACK: u8 = 1 << 3;
    pub const AUTO_ECHO: u8 = 1 << 4;
}

pub trait IrqCallback: Send + Sync {
    fn set_level(&self, level: bool);
}

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
    pub callback: Option<Arc<dyn IrqCallback>>,
    // Z85C30 Tx interrupt fires once when buffer empties, then arms again only
    // after RES_Tx_P.  Without this latch the IRQ line stays asserted forever.
    pub tx_int_pending: bool,
}

impl Channel {
    pub fn new(name: &str, ip_num: Arc<AtomicU8>, ip_other: Arc<AtomicU8>, callback: Option<Arc<dyn IrqCallback>>) -> Self {
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
            callback,
            tx_int_pending: false,
        };
        c.update_tx_delay();
        c
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
        self.reg_ptr = 0;
        self.status = rr0::TX_BUFFER_EMPTY;
        self.regs.fill(0);
        self.tx_int_pending = false;
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
        
        if let Some(cb) = &self.callback {
            let other = self.ip_other.load(Ordering::SeqCst);
            cb.set_level((ip | other) != 0);
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
        // this char completes (TX_BUFFER_EMPTY goes high after the TX thread
        // drains it and calls notify_tx_empty).
        self.tx_int_pending = false;

        // Update status to indicate Tx Buffer Full (clearing Empty bit) only if full
        if self.tx_queue.len() >= 4 {
            self.status &= !rr0::TX_BUFFER_EMPTY;
        }
        self.update_ip();
    }

    /// Called by the TX thread after sending a character and the queue drains.
    /// Sets the Tx-interrupt-pending latch so the driver is notified that the
    /// buffer is ready for the next character.
    pub fn notify_tx_empty(&mut self) {
        self.tx_int_pending = true;
        self.update_ip();
    }
}

pub trait SerialBackend: Send + Sync {
    fn send_byte(&self, byte: u8);
    fn recv_byte(&self) -> io::Result<u8>;
}

/// Drops TX bytes and never yields RX. Used as a placeholder when a channel
/// isn't wired to a host I/O source (e.g. CI mode unused channel).
struct NullBackend;

impl SerialBackend for NullBackend {
    fn send_byte(&self, _byte: u8) {}
    fn recv_byte(&self) -> io::Result<u8> {
        Err(io::Error::new(io::ErrorKind::WouldBlock, "null"))
    }
}

/// Tees guest-emitted bytes to a host file in addition to delegating to an
/// inner backend.  Used by `--serial-log <file>` to capture the serial
/// console transcript while the real TCP listener on port 8881 keeps
/// working unchanged.
pub struct TeeBackend {
    inner: Arc<dyn SerialBackend>,
    log: Mutex<std::fs::File>,
}

impl TeeBackend {
    pub fn new(inner: Arc<dyn SerialBackend>, path: &str) -> io::Result<Self> {
        let log = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { inner, log: Mutex::new(log) })
    }
}

impl SerialBackend for TeeBackend {
    fn send_byte(&self, byte: u8) {
        let mut f = self.log.lock();
        let _ = f.write_all(&[byte]);
        let _ = f.flush();
        drop(f);
        self.inner.send_byte(byte);
    }
    fn recv_byte(&self) -> io::Result<u8> {
        self.inner.recv_byte()
    }
}

#[cfg(unix)]
struct UnixSocketBackend {
    listener: UnixListener,
    stream: Mutex<Option<UnixStream>>,
}

#[cfg(unix)]
impl UnixSocketBackend {
    fn new<P: AsRef<Path>>(path: P) -> Self {
        let _ = std::fs::remove_file(&path);
        let listener = UnixListener::bind(path).expect("Failed to bind serial socket");
        listener.set_nonblocking(true).expect("Failed to set nonblocking");
        Self {
            listener,
            stream: Mutex::new(None),
        }
    }
}

#[cfg(unix)]
impl SerialBackend for UnixSocketBackend {
    fn send_byte(&self, byte: u8) {
        let mut stream_guard = self.stream.lock();
        if let Some(ref mut stream) = *stream_guard {
            if let Err(e) = stream.write_all(&[byte]) {
                if e.kind() != io::ErrorKind::WouldBlock {
                    *stream_guard = None;
                }
            }
        }
    }

    fn recv_byte(&self) -> io::Result<u8> {
        let mut guard = self.stream.lock();
        
        if guard.is_none() {
            match self.listener.accept() {
                Ok((socket, _)) => {
                    socket.set_nonblocking(true)?;
                    *guard = Some(socket);
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    return Err(io::Error::new(io::ErrorKind::WouldBlock, "No connection"));
                }
                Err(e) => return Err(e),
            }
        }

        if let Some(ref mut stream) = *guard {
            let mut buf = [0u8; 1];
            match stream.read(&mut buf) {
                Ok(1) => Ok(buf[0]),
                Ok(_) => { // EOF
                    *guard = None;
                    Err(io::Error::new(io::ErrorKind::NotConnected, "EOF"))
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => Err(io::Error::new(io::ErrorKind::WouldBlock, "WouldBlock")),
                Err(e) => {
                    *guard = None;
                    Err(e)
                }
            }
        } else {
            Err(io::Error::new(io::ErrorKind::WouldBlock, "No connection"))
        }
    }
}

struct TcpConn {
    stream: TcpStream,
    telnet: crate::telnet::TelnetFilter,
}

struct TcpSocketBackend {
    listener: TcpListener,
    conn: Mutex<Option<TcpConn>>,
}

impl TcpSocketBackend {
    /// Bind a TCP serial backend. Returns `None` (rather than panicking) if the
    /// port can't be bound — most commonly because a stale emulator process from
    /// an earlier crash is still holding it. Callers fall back to a NullBackend
    /// so the machine still boots; that serial channel is simply unavailable
    /// until the port frees.
    fn new<A: std::net::ToSocketAddrs>(addr: A) -> Option<Self> {
        let listener = match TcpListener::bind(addr) {
            Ok(l) => l,
            Err(e) => {
                log::warn!("serial TCP backend disabled: failed to bind socket: {e}");
                return None;
            }
        };
        if let Err(e) = listener.set_nonblocking(true) {
            log::warn!("serial TCP backend disabled: failed to set nonblocking: {e}");
            return None;
        }
        Some(Self {
            listener,
            conn: Mutex::new(None),
        })
    }
}

impl SerialBackend for TcpSocketBackend {
    fn send_byte(&self, byte: u8) {
        let mut guard = self.conn.lock();
        if let Some(ref mut c) = *guard {
            let mut buf = Vec::with_capacity(2);
            crate::telnet::escape_byte(byte, &mut buf);
            if let Err(e) = c.stream.write_all(&buf) {
                if e.kind() != io::ErrorKind::WouldBlock {
                    *guard = None;
                }
            }
        }
    }

    fn recv_byte(&self) -> io::Result<u8> {
        let mut guard = self.conn.lock();

        if guard.is_none() {
            match self.listener.accept() {
                Ok((socket, _)) => {
                    socket.set_nonblocking(true)?;
                    let mut c = TcpConn {
                        stream: socket,
                        telnet: crate::telnet::TelnetFilter::new(),
                    };
                    // Send the initial WILL/DO offers. If the write fails,
                    // drop the connection — the client can reconnect.
                    let hs = crate::telnet::TelnetFilter::initial_handshake();
                    if c.stream.write_all(&hs).is_err() {
                        return Err(io::Error::new(io::ErrorKind::WouldBlock, "handshake failed"));
                    }
                    *guard = Some(c);
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    return Err(io::Error::new(io::ErrorKind::WouldBlock, "No connection"));
                }
                Err(e) => return Err(e),
            }
        }

        // Pull bytes one at a time through the telnet filter until either a
        // real data byte falls out or the socket has nothing more to give.
        loop {
            let c = guard.as_mut().unwrap();
            let mut buf = [0u8; 1];
            match c.stream.read(&mut buf) {
                Ok(1) => {
                    let mut reply = Vec::new();
                    let data = c.telnet.feed(buf[0], &mut reply);
                    if !reply.is_empty() {
                        if let Err(e) = c.stream.write_all(&reply) {
                            if e.kind() != io::ErrorKind::WouldBlock {
                                *guard = None;
                                return Err(e);
                            }
                        }
                    }
                    if let Some(d) = data {
                        return Ok(d);
                    }
                    // Telnet command consumed; keep reading.
                }
                Ok(_) => {
                    *guard = None;
                    return Err(io::Error::new(io::ErrorKind::NotConnected, "EOF"));
                }
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                    return Err(io::Error::new(io::ErrorKind::WouldBlock, "WouldBlock"));
                }
                Err(e) => {
                    *guard = None;
                    return Err(e);
                }
            }
        }
    }
}

#[derive(Clone)]
pub struct Z85c30 {
    pub channel_a: Arc<(Mutex<Channel>, Condvar)>,
    pub channel_b: Arc<(Mutex<Channel>, Condvar)>,
    // Swappable so CI mode can replace the default TCP backend with a
    // `CiSerialBackend` before `start()` is called. Wrapped in `Arc<Mutex<_>>`
    // so `Z85c30` stays `Clone` and the swap is thread-safe.
    backend_a: Arc<Mutex<Arc<dyn SerialBackend>>>,
    backend_b: Arc<Mutex<Arc<dyn SerialBackend>>>,
    // In-process injection queue for channel B (tty1, the IRIX serial console).
    // Bytes queued via `inject_b` are delivered to the guest by the channel-B
    // RX thread ahead of any socket input, so a host action (e.g. the GUI's
    // "Send IRIX halt") can type at the console without opening a loopback TCP
    // client. Independent of whichever backend is installed.
    inject_b: Arc<Mutex<VecDeque<u8>>>,
    // Bounded mirror of channel-B (IRIX serial console) guest output. Lets an
    // in-process reader — the GUI's network probe — scrape the console without
    // opening the 8881 TCP socket. Oldest bytes drop once full.
    console_buf: Arc<Mutex<VecDeque<u8>>>,
    running: Arc<AtomicBool>,
    threads: Arc<Mutex<Vec<thread::JoinHandle<()>>>>,
}

/// Cap on the channel-B console mirror (`console_buf`). 64 KiB is far more than
/// any single probe response; the buffer is drained on each read.
const CONSOLE_TAP_CAP: usize = 64 * 1024;

impl Z85c30 {
    /// Default constructor: binds TCP serial backends on 127.0.0.1:8880
    /// (channel A / tty2) and 127.0.0.1:8881 (channel B / tty1).
    pub fn new(callback: Option<Arc<dyn IrqCallback>>) -> Self {
        Self::new_inner(callback, true)
    }

    /// CI-mode constructor: uses null backends instead of binding TCP. The
    /// caller is expected to install real backends via `set_backend_a` /
    /// `set_backend_b` before the first `start()`. Avoids port conflicts
    /// when multiple `--ci` instances run in parallel.
    pub fn new_null(callback: Option<Arc<dyn IrqCallback>>) -> Self {
        Self::new_inner(callback, false)
    }

    fn new_inner(callback: Option<Arc<dyn IrqCallback>>, bind_tcp: bool) -> Self {
        let ip_a = Arc::new(AtomicU8::new(0));
        let ip_b = Arc::new(AtomicU8::new(0));

        let (backend_a, backend_b): (Arc<dyn SerialBackend>, Arc<dyn SerialBackend>) = if bind_tcp {
            let tcp_or_null = |addr| -> Arc<dyn SerialBackend> {
                match TcpSocketBackend::new(addr) {
                    Some(b) => Arc::new(b),
                    None => Arc::new(NullBackend),
                }
            };
            (
                tcp_or_null("127.0.0.1:8880"),
                tcp_or_null("127.0.0.1:8881"),
            )
        } else {
            (Arc::new(NullBackend), Arc::new(NullBackend))
        };

        Self {
            channel_a: Arc::new((Mutex::new(Channel::new("A", ip_a.clone(), ip_b.clone(), callback.clone())), Condvar::new())),
            // Note: Channel B gets ip_b as its 'num' and ip_a as 'other'
            channel_b: Arc::new((Mutex::new(Channel::new("B", ip_b, ip_a, callback)), Condvar::new())),
            backend_a: Arc::new(Mutex::new(backend_a)),
            backend_b: Arc::new(Mutex::new(backend_b)),
            inject_b: Arc::new(Mutex::new(VecDeque::new())),
            console_buf: Arc::new(Mutex::new(VecDeque::new())),
            running: Arc::new(AtomicBool::new(false)),
            threads: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Swap in an alternate backend for channel A (tty2 on Indy).
    /// Must be called before `start()` — running RX/TX threads cache the
    /// backend Arc at spawn time and will not observe the new one until
    /// they are stopped and restarted.
    pub fn set_backend_a(&self, backend: Arc<dyn SerialBackend>) {
        *self.backend_a.lock() = backend;
    }

    /// Swap in an alternate backend for channel B (tty1, the PROM/IRIX
    /// serial console on Indy). Same constraint as `set_backend_a`.
    pub fn set_backend_b(&self, backend: Arc<dyn SerialBackend>) {
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
        self.inject_b.lock().extend(data.iter().copied());
    }

    /// Drain and return the channel-B (IRIX serial console) output captured
    /// since the last call. Empties the mirror buffer. Used by the GUI network
    /// probe to read the guest's response to an injected command in-process.
    pub fn drain_console(&self) -> Vec<u8> {
        self.console_buf.lock().drain(..).collect()
    }

    pub fn read_a_control(&self) -> u8 { 
        let mut a = self.channel_a.0.lock();
        if a.reg_ptr == 2 {
            // RR2: Interrupt Vector (modified in Channel B)
            // TODO: Implement Status Affects Vector modification
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
        let mut a = self.channel_a.0.lock();
        let mut b = self.channel_b.0.lock();
        
        let update = a.write_control(val);
        self.channel_a.1.notify_one();
        if let Some(wr2) = update {
            b.regs[2] = wr2;
        }
    }
    pub fn read_a_data(&self) -> u8 { self.channel_a.0.lock().read_data() }
    pub fn write_a_data(&self, val: u8) { 
        let mut lock = self.channel_a.0.lock();
        lock.write_data(val);
        self.channel_a.1.notify_one();
    }

    pub fn read_b_control(&self) -> u8 { 
        let mut b = self.channel_b.0.lock();
        if b.reg_ptr == 2 {
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
        let mut a = self.channel_a.0.lock();
        let mut b = self.channel_b.0.lock();
        
        let update = b.write_control(val);
        self.channel_b.1.notify_one();
        if let Some(wr2) = update {
            a.regs[2] = wr2;
        }
    }
    pub fn read_b_data(&self) -> u8 { self.channel_b.0.lock().read_data() }
    pub fn write_b_data(&self, val: u8) { 
        let mut lock = self.channel_b.0.lock();
        lock.write_data(val);
        self.channel_b.1.notify_one();
    }

    pub fn get_ip(&self) -> u8 {
        // Check both channels for pending interrupts
        // Lock A to get its status and read B's atomic (ip_other on A is B's ip)
        let a = self.channel_a.0.lock();
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
        use crate::locks::register_lock_fn;
        let ch = self.channel_a.clone();
        register_lock_fn("scc::channel_a", move || ch.0.is_locked());
        let ch = self.channel_b.clone();
        register_lock_fn("scc::channel_b", move || ch.0.is_locked());
        let th = self.threads.clone();
        register_lock_fn("scc::threads", move || th.is_locked());
    }
}

impl Device for Z85c30 {
    fn step(&self, _cycles: u64) {}
    
    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.channel_a.1.notify_all();
        self.channel_b.1.notify_all();
        let mut threads = self.threads.lock();
        for t in threads.drain(..) {
            let _ = t.join();
        }
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }

        let pairs = [
            (self.channel_a.clone(), self.backend_a.lock().clone()),
            (self.channel_b.clone(), self.backend_b.lock().clone()),
        ];

        let mut threads = self.threads.lock();

        for (i, (channel_arc, backend)) in pairs.iter().enumerate() {
            let ch_name = if i == 0 { "A" } else { "B" };
            // TX Thread
            let tx_channel = channel_arc.clone();
            let tx_backend = backend.clone();
            let running = self.running.clone();
            // Channel B (i == 1) is the IRIX serial console — mirror its output
            // into console_buf for in-process readers (the GUI network probe).
            let tx_tap = if i == 1 { Some(self.console_buf.clone()) } else { None };

            threads.push(thread::Builder::new().name(format!("SCC-TX-{}", ch_name)).spawn(move || {
                let mut last_tx_time = Instant::now();
                let channel_name = {
                    let (lock, _) = &*tx_channel;
                    lock.lock().name.clone()
                };

                while running.load(Ordering::Relaxed) {
                    let (lock, cvar) = &*tx_channel;
                    let mut channel = lock.lock();
                    
                    loop {
                        if !running.load(Ordering::Relaxed) { break; }

                        let wr5 = channel.regs[scc_regs::WR5 as usize];
                        let tx_enabled = (wr5 & wr5::TX_ENABLE) != 0;
                        if !channel.tx_queue.is_empty() && tx_enabled {
                            break;
                        }
                        
                        cvar.wait(&mut channel);
                    }

                    if !running.load(Ordering::Relaxed) { break; }

                    // Pop data to simulate moving to Shift Register
                    let val = channel.tx_queue.pop_front();

                    // Holding register is now empty (FIFO not full): update
                    // TX_BUFFER_EMPTY so the driver can queue another char,
                    // but do NOT fire the Tx interrupt yet — that happens only
                    // after transmission completes (notify_tx_empty below).
                    if channel.tx_queue.len() < 4 {
                        channel.status |= rr0::TX_BUFFER_EMPTY;
                    }

                    // Get pre-calculated delay
                    let delay_micros = channel.tx_delay;
                    let char_duration = Duration::from_micros(delay_micros);

                    // Release lock to sleep (simulate transmission time)
                    drop(channel);

                    if let Some(byte) = val {
                        let now = Instant::now();
                        if last_tx_time < now {
                            if now.duration_since(last_tx_time) > Duration::from_millis(100) {
                                last_tx_time = now;
                            }
                        }
                        last_tx_time += char_duration;
                        let wait = last_tx_time.saturating_duration_since(now);
                        if !wait.is_zero() {
                            thread::sleep(wait);
                        }

                        // Output character
                        crate::dlog_dev!(LogModule::Scc, "SCC: TX({}) '{}' ({:02x})", channel_name, if byte.is_ascii_graphic() { byte as char } else { '.' }, byte);
                        if let Some(tap) = &tx_tap {
                            let mut b = tap.lock();
                            while b.len() >= CONSOLE_TAP_CAP { b.pop_front(); }
                            b.push_back(byte);
                        }
                        tx_backend.send_byte(byte);

                        // Transmission complete: fire Tx interrupt (once) to
                        // tell the driver the buffer is ready for the next char.
                        let (lock, _) = &*tx_channel;
                        lock.lock().notify_tx_empty();
                    }
                }
            }).unwrap());

            // RX Thread
            let rx_channel = channel_arc.clone();
            let rx_backend = backend.clone();
            let running = self.running.clone();
            // Only channel B (the IRIX serial console) accepts in-process
            // injection; channel A has no queue.
            let rx_inject = if i == 1 { Some(self.inject_b.clone()) } else { None };

            threads.push(thread::Builder::new().name(format!("SCC-RX-{}", ch_name)).spawn(move || {
                let mut last_rx_time = Instant::now();
                // When the SCC's 8-byte rx_queue is full, hold the just-read
                // byte here and retry on the next iteration instead of
                // dropping it. This prevents loss when the host pushes a
                // long line into CiSerialBackend faster than IRIX's tty
                // driver clocks bytes off rx_queue. Without this hold, a
                // ~30-char `dd if=/dev/rdsk/dks0d2s0 bs=512` arrives at
                // the shell as `dd if=/d=512` (chars 9..24 dropped).
                let mut pending: Option<u8> = None;

                while running.load(Ordering::Relaxed) {
                    let mut byte = match pending.take() {
                        Some(b) => b,
                        // In-process injection (channel B) takes priority over
                        // socket input, so a queued "halt\n" is delivered even
                        // when no TCP client is attached.
                        None => match rx_inject.as_ref().and_then(|q| q.lock().pop_front()) {
                            Some(b) => b,
                            None => match rx_backend.recv_byte() {
                                Ok(b) => b,
                                Err(_) => {
                                    thread::sleep(Duration::from_millis(10));
                                    continue;
                                }
                            },
                        },
                    };
                    if byte == 0x05 {
                        crate::dlog_dev!(LogModule::Scc, "SCC: Converting ^E to ^D (BREAK)");
                        byte = 0x04;
                    }

                    let (lock, _cvar) = &*rx_channel;
                    let mut channel = lock.lock();

                    let wr3 = channel.regs[scc_regs::WR3 as usize];
                    let rx_enabled = (wr3 & wr3::RX_ENABLE) != 0;
                    let delay_micros = channel.tx_delay;
                    let char_duration = Duration::from_micros(delay_micros);

                    if !rx_enabled {
                        // RX disabled — drop the byte (matches real hw with
                        // RX off). Don't hold it in `pending` or we'd block
                        // forever waiting for re-enable.
                        drop(channel);
                        continue;
                    }

                    if channel.rx_queue.len() >= 8 {
                        // SCC FIFO full. Hold the byte and back off briefly
                        // so the guest's tty driver gets a chance to drain
                        // rx_queue. Don't drop — that's the bug this
                        // section fixes.
                        drop(channel);
                        pending = Some(byte);
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }

                    crate::dlog_dev!(LogModule::Scc, "SCC: RX({}) '{}' ({:02x})", channel.name, if byte.is_ascii_graphic() { byte as char } else { '.' }, byte);
                    channel.rx_queue.push_back(byte);
                    channel.status |= rr0::RX_CHAR_AVAILABLE;
                    channel.update_ip();
                    drop(channel);

                    // Pacing — simulate baud-rate inter-character spacing.
                    let now = Instant::now();
                    if last_rx_time < now {
                        if now.duration_since(last_rx_time) > Duration::from_millis(100) {
                            last_rx_time = now;
                        }
                    }
                    last_rx_time += char_duration;
                    let wait = last_rx_time.saturating_duration_since(now);
                    if !wait.is_zero() {
                        thread::sleep(wait);
                    }
                }
            }).unwrap());
        }
    }

    fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("serial".to_string(), "serial status  — dump SCC channel A/B registers and FIFO state".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd == "serial" {
            if args.first().copied() == Some("status") || args.is_empty() {
                for (label, arc) in [("A", &self.channel_a), ("B", &self.channel_b)] {
                    let ch = arc.0.lock();
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
    /// Threads keep running (they idle when tx_enabled / rx_enabled are off).
    fn power_on(&self) {
        let (lock_a, cvar_a) = &*self.channel_a;
        lock_a.lock().reset();
        cvar_a.notify_all();

        let (lock_b, cvar_b) = &*self.channel_b;
        lock_b.lock().reset();
        cvar_b.notify_all();
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
    // Both FIFOs are empty now and rr0 has to agree. This is the part that
    // actually unblocks the guest: IRIX polls RR0 on the console path, and a
    // restored status without TX_BUFFER_EMPTY reads as "still busy", after
    // which the driver never writes again. Neither bit heals itself, since
    // write_data only clears TX_BUFFER_EMPTY at a full FIFO and read_data only
    // clears RX_CHAR_AVAILABLE when it pops the last byte.
    ch.status |= rr0::TX_BUFFER_EMPTY;
    ch.status &= !rr0::RX_CHAR_AVAILABLE;
    // Clearing tx_queue destroyed the character whose completion would have
    // set the latch, and notify_tx_empty only fires when a transmit finishes,
    // so nothing else ever would.
    let wr1 = ch.regs[scc_regs::WR1 as usize];
    ch.tx_int_pending = (wr1 & wr1::TX_INT_EN) != 0;
    ch.update_tx_delay();
    // power_on_devices cleared map_stat SERIAL and Ioc::load_state then put it
    // back wholesale, so publish ip_num and redrive the line to keep the two
    // agreeing. Lock order (channel then IOC state) matches write_a_control,
    // which holds both channel locks across update_ip.
    ch.update_ip();
}

impl Saveable for Z85c30 {
    fn save_state(&self) -> toml::Value {
        let mut tbl = toml::map::Map::new();
        tbl.insert("ch_a".into(), channel_to_toml(&self.channel_a.0.lock()));
        tbl.insert("ch_b".into(), channel_to_toml(&self.channel_b.0.lock()));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(ca) = get_field(v, "ch_a") {
            channel_from_toml(ca, &mut self.channel_a.0.lock());
            self.channel_a.1.notify_all();
        }
        if let Some(cb) = get_field(v, "ch_b") {
            channel_from_toml(cb, &mut self.channel_b.0.lock());
            self.channel_b.1.notify_all();
        }
        Ok(())
    }
}

// ============================================================================
// CiSerialBackend — in-process serial backend used by --ci mode.
// ============================================================================

/// Serial backend that the CI control socket reads from and writes to.
/// `Machine::new` installs it on channel B, which is tty1, the PROM and IRIX
/// serial console. Host pushes bytes into `host_to_guest` via `push_host`; the
/// existing RX thread drains them into `channel_b.rx_queue`. Guest output
/// reaches `send_byte`, which pushes into `guest_to_host` and wakes anyone
/// waiting in `wait_for`.
pub struct CiSerialBackend {
    host_to_guest: Mutex<VecDeque<u8>>,
    guest_to_host: Mutex<Vec<u8>>,
    cv: Condvar,
    /// Optional file mirror: every byte that flows through ttyd1 in either
    /// direction is appended here. Host-injected bytes are echoed by IRIX
    /// anyway, so writing only guest output already captures the full
    /// transcript without duplication.
    log: Mutex<Option<std::fs::File>>,
}

impl CiSerialBackend {
    pub fn new() -> Self {
        Self {
            host_to_guest: Mutex::new(VecDeque::new()),
            guest_to_host: Mutex::new(Vec::new()),
            cv: Condvar::new(),
            log: Mutex::new(None),
        }
    }

    /// Open a log file that will receive every guest output byte. Append-only;
    /// existing contents are preserved. Errors are returned to the caller.
    pub fn set_log_file(&self, path: &str) -> io::Result<()> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        *self.log.lock() = Some(f);
        Ok(())
    }

    /// Inject bytes from host to guest (the harness typing on the console).
    pub fn push_host(&self, data: &[u8]) {
        let mut q = self.host_to_guest.lock();
        q.extend(data.iter().copied());
    }

    /// Drain everything the guest has produced since the last call. Empties
    /// the buffer; the returned Vec is the guest output as raw bytes.
    pub fn drain_guest(&self) -> Vec<u8> {
        let mut q = self.guest_to_host.lock();
        std::mem::take(&mut *q)
    }

    /// Block until `needle` is seen in guest output, or `timeout` expires.
    /// On success returns the consumed bytes up to and including the match;
    /// bytes that arrived after the match stay in the buffer for the next
    /// `serial-read`. On timeout returns `None` without consuming anything.
    pub fn wait_for(&self, needle: &[u8], timeout: Duration) -> Option<Vec<u8>> {
        if needle.is_empty() {
            return Some(Vec::new());
        }
        let deadline = Instant::now() + timeout;
        let mut q = self.guest_to_host.lock();
        loop {
            if let Some(pos) = find_subseq(&q, needle) {
                let end = pos + needle.len();
                let consumed: Vec<u8> = q.drain(..end).collect();
                return Some(consumed);
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            if self.cv.wait_until(&mut q, deadline).timed_out() {
                // One more scan in case bytes arrived between the last check
                // and the timeout.
                if let Some(pos) = find_subseq(&q, needle) {
                    let end = pos + needle.len();
                    let consumed: Vec<u8> = q.drain(..end).collect();
                    return Some(consumed);
                }
                return None;
            }
        }
    }

    /// Clear both queues. Called on `restore`/`rollback` so stale serial
    /// output from the previous run doesn't leak into the next test.
    pub fn reset(&self) {
        self.host_to_guest.lock().clear();
        self.guest_to_host.lock().clear();
    }
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

impl SerialBackend for CiSerialBackend {
    fn send_byte(&self, byte: u8) {
        self.guest_to_host.lock().push(byte);
        if let Some(f) = self.log.lock().as_mut() {
            let _ = f.write_all(&[byte]);
            let _ = f.flush();
        }
        self.cv.notify_all();
    }

    fn recv_byte(&self) -> io::Result<u8> {
        let mut q = self.host_to_guest.lock();
        match q.pop_front() {
            Some(b) => Ok(b),
            None => Err(io::Error::new(io::ErrorKind::WouldBlock, "empty")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 1.7 round-trip: a fresh SCC loaded from a captured save_state must
    /// re-serialize byte-identically. Use new_null so the test doesn't bind any
    /// TCP ports.
    #[test]
    fn save_load_round_trip() {
        let src = Z85c30::new_null(None);
        {
            let mut ch = src.channel_a.0.lock();
            ch.regs[0]  = 0x44;
            ch.regs[1]  = 0x12;
            ch.regs[3]  = 0xc1;
            ch.regs[5]  = 0xea;
            ch.reg_ptr  = 7;
            ch.status   = 0x40 | rr0::TX_BUFFER_EMPTY;
        }
        {
            let mut ch = src.channel_b.0.lock();
            ch.regs[0]  = 0x88;
            ch.regs[2]  = 0x10;
            ch.regs[15] = 0x05;
            ch.reg_ptr  = 3;
            ch.status   = 0x80 | rr0::TX_BUFFER_EMPTY;
        }
        let v1 = src.save_state();

        let dst = Z85c30::new_null(None);
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
        use std::time::{Duration, Instant};

        let scc = Z85c30::new_null(None);
        let backend = Arc::new(CiSerialBackend::new());
        scc.set_backend_a(backend.clone());

        // Enable RX on channel A so the rx thread queues bytes. tx_delay is
        // tx-direction baud-rate emulation; set a small value so the test
        // doesn't pay 19.2 kbaud-per-char latency.
        {
            let mut ch = scc.channel_a.0.lock();
            ch.regs[scc_regs::WR3 as usize] |= wr3::RX_ENABLE;
            ch.tx_delay = 50; // 50 µs/byte
        }

        scc.start();

        let line = b"dd if=/dev/rdsk/dks0d2s0 of=/tmp/r.bin bs=512 count=1\r";
        backend.push_host(line);

        // Drain rx_queue at ~20 kHz so the rx thread always has space to
        // push pending bytes. Mirrors how IRIX's tty driver consumes
        // RR0::RX_CHAR_AVAILABLE.
        let mut received = Vec::with_capacity(line.len());
        let deadline = Instant::now() + Duration::from_secs(5);
        while received.len() < line.len() && Instant::now() < deadline {
            let popped = {
                let mut ch = scc.channel_a.0.lock();
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

    /// Restore clears `tx_queue`, which destroys the character whose
    /// completion would have set the latch. Asserted through RR3, which is how
    /// a guest ISR sees it: `read_a_control` recomputes channel A's IP live.
    #[test]
    fn restore_rearms_tx_interrupt() {
        let src = Z85c30::new_null(None);
        {
            let mut ch = src.channel_a.0.lock();
            ch.regs[scc_regs::WR1 as usize] = wr1::TX_INT_EN;
        }
        let saved = src.save_state();

        let dst = Z85c30::new_null(None);
        dst.load_state(&saved).expect("load_state");
        dst.channel_a.0.lock().reg_ptr = 3;

        assert_eq!(dst.read_a_control(), (1 << 1) << 3,
            "RR3 must report a channel A TX interrupt after restore");
    }

    /// Guard on the re-arm above: a channel whose driver never enabled TX
    /// interrupts must not come back from a restore asserting one. Passes on
    /// the unfixed tree too, which forced the latch false unconditionally.
    #[test]
    fn restore_no_tx_int_when_disabled() {
        let src = Z85c30::new_null(None);
        {
            let mut ch = src.channel_a.0.lock();
            ch.regs[scc_regs::WR1 as usize] = 0;
        }
        let saved = src.save_state();

        let dst = Z85c30::new_null(None);
        dst.load_state(&saved).expect("load_state");
        dst.channel_a.0.lock().reg_ptr = 3;

        assert_eq!(dst.read_a_control(), 0,
            "no TX interrupt when WR1 TX_INT_EN is clear");
    }

    struct Recorder(AtomicBool);
    impl IrqCallback for Recorder {
        fn set_level(&self, level: bool) { self.0.store(level, Ordering::SeqCst); }
    }

    /// Channel B is the CI console (`machine.rs` installs `CiSerialBackend`
    /// with `set_backend_b`), and it is not symmetric with A: `read_b_control`
    /// returns 0 for RR3, so B's IP reaches the guest only through A's RR3 via
    /// the shared `ip_other` atomic.
    #[test]
    fn restore_rearms_tx_interrupt_channel_b() {
        let src = Z85c30::new_null(None);
        {
            let mut ch = src.channel_b.0.lock();
            ch.regs[scc_regs::WR1 as usize] = wr1::TX_INT_EN;
        }
        let saved = src.save_state();

        let irq = Arc::new(Recorder(AtomicBool::new(false)));
        let dst = Z85c30::new_null(Some(irq.clone()));
        dst.load_state(&saved).expect("load_state");
        dst.channel_a.0.lock().reg_ptr = 3;

        assert_eq!(dst.read_a_control(), 1 << 1,
            "RR3 must report a channel B TX interrupt in its low bits");
        assert!(irq.0.load(Ordering::SeqCst),
            "restore must raise the IRQ line for channel B");
    }

    /// The operative fix. A restored `status` without TX_BUFFER_EMPTY reads as
    /// "still busy" to a guest polling RR0, and `write_data` only ever clears
    /// that bit at a full FIFO, so the driver stops writing for good.
    #[test]
    fn restore_sets_tx_buffer_empty_status() {
        let src = Z85c30::new_null(None);
        {
            let mut ch = src.channel_a.0.lock();
            ch.status &= !rr0::TX_BUFFER_EMPTY;
        }
        let saved = src.save_state();

        let dst = Z85c30::new_null(None);
        dst.load_state(&saved).expect("load_state");

        let ch = dst.channel_a.0.lock();
        assert!(ch.tx_queue.is_empty(), "tx_queue must be empty after restore");
        assert!(ch.status & rr0::TX_BUFFER_EMPTY != 0,
            "TX_BUFFER_EMPTY must be set when tx_queue is empty");
    }

    /// Setting the latch is not delivery. `get_ip` is a pure read of the
    /// channel's own fields; the IRQ only reaches the IOC through `update_ip`,
    /// which stores `ip_num` and calls `set_level`.
    #[test]
    fn restore_drives_irq_line_to_parent() {
        let src = Z85c30::new_null(None);
        {
            let mut ch = src.channel_a.0.lock();
            ch.regs[scc_regs::WR1 as usize] = wr1::TX_INT_EN;
        }
        let saved = src.save_state();

        let irq = Arc::new(Recorder(AtomicBool::new(false)));
        let dst = Z85c30::new_null(Some(irq.clone()));
        dst.load_state(&saved).expect("load_state");

        // ip_num is channel B's ip_other, which B's update_ip ORs when it
        // computes the line level. RR3 for A is recomputed live, not read here.
        assert_eq!(dst.channel_a.0.lock().ip_num.load(Ordering::SeqCst), 1 << 1,
            "restore must publish the TX IP for the other channel to see");
        assert!(irq.0.load(Ordering::SeqCst),
            "restore must raise the IRQ line into the IOC");
    }

    /// A restored `status` claiming RX_CHAR_AVAILABLE over an empty `rx_queue`
    /// never recovers: `read_data` only drops the bit when it pops a byte, so
    /// the driver spins reading zeroes.
    #[test]
    fn restore_clears_rx_char_available_status() {
        let src = Z85c30::new_null(None);
        {
            let mut ch = src.channel_a.0.lock();
            ch.status |= rr0::RX_CHAR_AVAILABLE;
        }
        let saved = src.save_state();

        let dst = Z85c30::new_null(None);
        dst.load_state(&saved).expect("load_state");

        let ch = dst.channel_a.0.lock();
        assert!(ch.rx_queue.is_empty(), "rx_queue must be empty after restore");
        assert!(ch.status & rr0::RX_CHAR_AVAILABLE == 0,
            "RX_CHAR_AVAILABLE must be clear when rx_queue is empty");
    }
}
