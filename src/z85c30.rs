use std::sync::Arc;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicU8, AtomicBool, Ordering};
use std::collections::VecDeque;
use std::io::{self, Write, Read};
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
                3 => self.reset(), // Channel Reset
                _ => {}, // Other commands ignored for now
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
        // And Tx Buffer Empty (RR0 bit 2)
        if (wr1 & wr1::TX_INT_EN) != 0 && (self.status & rr0::TX_BUFFER_EMPTY) != 0 {
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

        // Update status to indicate Tx Buffer Full (clearing Empty bit) only if full
        if self.tx_queue.len() >= 4 {
            self.status &= !rr0::TX_BUFFER_EMPTY;
        }
        self.update_ip();
    }
}

pub trait SerialBackend: Send + Sync {
    fn send_byte(&self, byte: u8);
    fn recv_byte(&self) -> io::Result<u8>;
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

struct TcpSocketBackend {
    listener: TcpListener,
    stream: Mutex<Option<TcpStream>>,
}

impl TcpSocketBackend {
    fn new<A: std::net::ToSocketAddrs>(addr: A) -> Self {
        let listener = TcpListener::bind(addr).expect("Failed to bind serial TCP socket");
        listener.set_nonblocking(true).expect("Failed to set nonblocking");
        Self {
            listener,
            stream: Mutex::new(None),
        }
    }
}

impl SerialBackend for TcpSocketBackend {
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

#[derive(Clone)]
pub struct Z85c30 {
    pub channel_a: Arc<(Mutex<Channel>, Condvar)>,
    pub channel_b: Arc<(Mutex<Channel>, Condvar)>,
    backend_a: Arc<dyn SerialBackend>,
    backend_b: Arc<dyn SerialBackend>,
    running: Arc<AtomicBool>,
    threads: Arc<Mutex<Vec<thread::JoinHandle<()>>>>,
}

impl Z85c30 {
    pub fn new(callback: Option<Arc<dyn IrqCallback>>) -> Self {
        let ip_a = Arc::new(AtomicU8::new(0));
        let ip_b = Arc::new(AtomicU8::new(0));

        Self {
            channel_a: Arc::new((Mutex::new(Channel::new("A", ip_a.clone(), ip_b.clone(), callback.clone())), Condvar::new())),
            // Note: Channel B gets ip_b as its 'num' and ip_a as 'other'
            channel_b: Arc::new((Mutex::new(Channel::new("B", ip_b, ip_a, callback)), Condvar::new())),
            backend_a: Arc::new(TcpSocketBackend::new("127.0.0.1:8880")),
            backend_b: Arc::new(TcpSocketBackend::new("127.0.0.1:8881")),
            running: Arc::new(AtomicBool::new(false)),
            threads: Arc::new(Mutex::new(Vec::new())),
        }
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
            (self.channel_a.clone(), self.backend_a.clone()),
            (self.channel_b.clone(), self.backend_b.clone()),
        ];

        let mut threads = self.threads.lock();

        for (i, (channel_arc, backend)) in pairs.iter().enumerate() {
            let ch_name = if i == 0 { "A" } else { "B" };
            // TX Thread
            let tx_channel = channel_arc.clone();
            let tx_backend = backend.clone();
            let running = self.running.clone();
            
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
                    
                    // Holding register is now empty (FIFO not full)
                    if channel.tx_queue.len() < 4 {
                        channel.status |= rr0::TX_BUFFER_EMPTY;
                        channel.update_ip();
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
                        tx_backend.send_byte(byte);
                    }
                }
            }).unwrap());

            // RX Thread
            let rx_channel = channel_arc.clone();
            let rx_backend = backend.clone();
            let running = self.running.clone();

            threads.push(thread::Builder::new().name(format!("SCC-RX-{}", ch_name)).spawn(move || {
                let mut last_rx_time = Instant::now();

                while running.load(Ordering::Relaxed) {
                    if let Ok(mut byte) = rx_backend.recv_byte() {
                        if byte == 0x05 {
                            crate::dlog_dev!(LogModule::Scc, "SCC: Converting ^E to ^D (BREAK)");
                            byte = 0x04;
                        }
                        let (lock, _cvar) = &*rx_channel;
                        let mut channel = lock.lock();
                        
                        let wr3 = channel.regs[scc_regs::WR3 as usize];
                        let rx_enabled = (wr3 & wr3::RX_ENABLE) != 0;

                        // Get pre-calculated delay
                        let delay_micros = channel.tx_delay;
                        let char_duration = Duration::from_micros(delay_micros);

                        if rx_enabled && channel.rx_queue.len() < 8 {
                            crate::dlog_dev!(LogModule::Scc, "SCC: RX({}) '{}' ({:02x})", channel.name, if byte.is_ascii_graphic() { byte as char } else { '.' }, byte);
                            channel.rx_queue.push_back(byte);
                            channel.status |= rr0::RX_CHAR_AVAILABLE;
                            channel.update_ip();
                        }

                        drop(channel);

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
                    } else {
                        // Avoid busy loop on error
                        thread::sleep(Duration::from_millis(10));
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
    // Clear RX/TX queues — transient in-flight data is lost on restore.
    ch.rx_queue.clear();
    ch.tx_queue.clear();
    ch.update_tx_delay();
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
