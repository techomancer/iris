//! The O2's two 16550 UARTs in MACE's ISA space.

use std::sync::Mutex;

use std::io::Write;

use crate::dev::z85c30::SerialBackend;
use crate::snapshot::{get_field, load_u8_slice, toml_u8, u8_slice_to_toml};
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, Device, Resettable, Saveable, BUS_OK};
use crate::ip32::*;

/// A 16550 with just enough behaviour to be written to.
///
/// Register `n` lives at `base + (n << 8) + 7`: MACE spaces its ISA registers
/// 256 bytes apart and puts the byte in the last lane of the 64-bit slot
/// (`sgimips/bus.c`: `h + (o << 8) + 7`). Getting this wrong is silent — every
/// register reads back zero and the firmware simply never transmits.
///
/// Transmit only. The line status register always reports the holding register
/// empty, so the PROM never waits on us, and bytes are appended to a buffer
/// rather than going anywhere.
pub struct Com16550 {
    out: Mutex<Vec<u8>>,
    /// The host end of the serial line, as the Z85C30's channels have:
    /// a TCP console, a log tee, or nothing.
    backend: Mutex<std::sync::Arc<dyn SerialBackend>>,
    /// Characters waiting to be read by the guest. The PROM polls LSR for
    /// data-ready and then reads the receive register, so an empty queue simply
    /// means "nobody has typed anything".
    input: std::sync::Arc<Mutex<std::collections::VecDeque<u8>>>,
    regs: Mutex<[u8; 8]>,
    /// Which UART this is (`com0`, `com1`): its monitor command.
    name: &'static str,
    /// Its interrupt, while it is driven by PIO (MACE ISA, port 0's group).
    irq: std::sync::OnceLock<crate::irq::IrqLine>,
}

pub mod com_reg {
    /// Transmit holding / receive buffer (and divisor low when DLAB is set).
    pub const THR: u32 = 0;
    pub const IER: u32 = 1;
    pub const IIR_FCR: u32 = 2;
    pub const LCR: u32 = 3;
    pub const MCR: u32 = 4;
    pub const LSR: u32 = 5;
    pub const MSR: u32 = 6;
    pub const SCR: u32 = 7;

    /// IER: interrupt on received data, and on the transmitter draining.
    pub const IER_RX: u8 = 0x01;
    pub const IER_TX: u8 = 0x02;
    /// IIR: bit 0 clear means "something to service"; the cause is above it.
    pub const IIR_NONE: u8 = 0x01;
    pub const IIR_TX_EMPTY: u8 = 0x02;
    pub const IIR_RX_DATA: u8 = 0x04;
    /// The two high bits a 16550 sets once its FIFOs are enabled.
    pub const IIR_FIFO: u8 = 0xc0;

    /// LCR bit 7: the next accesses to 0 and 1 are the baud divisor.
    pub const LCR_DLAB: u8 = 0x80;
    /// LSR: a received character is waiting.
    pub const LSR_DR: u8 = 0x01;
    /// LSR: transmit holding register empty.
    pub const LSR_THRE: u8 = 0x20;
    /// LSR: transmitter completely empty.
    pub const LSR_TEMT: u8 = 0x40;

    /// MCR bit 4: loopback. The transmitter is wired back to the receiver and
    /// the modem control outputs to the modem status inputs. post1's serial
    /// self-test runs entirely in this mode, so a UART without it fails POST.
    pub const MCR_LOOP: u8 = 0x10;
    /// MCR outputs that loop back to MSR while `MCR_LOOP` is set.
    pub const MCR_DTR: u8 = 0x01;
    pub const MCR_RTS: u8 = 0x02;
    pub const MCR_OUT1: u8 = 0x04;
    pub const MCR_OUT2: u8 = 0x08;
    /// MSR inputs, in the order the loopback wires them.
    pub const MSR_CTS: u8 = 0x10;
    pub const MSR_DSR: u8 = 0x20;
    pub const MSR_RI: u8 = 0x40;
    pub const MSR_DCD: u8 = 0x80;
}

impl Default for Com16550 {
    fn default() -> Self { Self::new() }
}

impl Com16550 {
    pub fn new() -> Self {
        Self::named("com")
    }

    /// A UART answering monitor commands as `name`.
    pub fn named(name: &'static str) -> Self {
        Self {
            out: Mutex::new(Vec::new()),
            backend: Mutex::new(crate::dev::z85c30::null_serial_backend()),
            input: Default::default(),
            regs: Mutex::new([0u8; 8]),
            name,
            irq: std::sync::OnceLock::new(),
        }
    }

    /// Wire the host end of the line: a TCP console
    /// (`z85c30::tcp_serial_backend`), a `TeeBackend` logging to a file, or
    /// nothing.
    pub fn set_backend(&self, backend: std::sync::Arc<dyn SerialBackend>) {
        *self.backend.lock().unwrap() = backend;
    }

    /// Move whatever the host end has typed into the receive queue. The
    /// machine's service loop calls this, and so does a status read, so a
    /// driver polling for data sees it at once.
    pub fn pump(&self) {
        let backend = self.backend.lock().unwrap().clone();
        let mut got = false;
        while let Ok(b) = backend.recv_byte() {
            self.input.lock().unwrap().push_back(b);
            got = true;
        }
        if got {
            self.update_irq();
        }
    }

    /// Queue characters for the guest to read.
    pub fn feed(&self, bytes: &[u8]) {
        self.input.lock().unwrap().extend(bytes.iter().copied());
        self.update_irq();
    }

    /// Characters still unread by the guest.
    pub fn pending_input(&self) -> usize {
        self.input.lock().unwrap().len()
    }

    /// Everything written to the transmit register so far.
    pub fn output(&self) -> String {
        String::from_utf8_lossy(&self.out.lock().unwrap()).to_string()
    }

    pub fn bytes_out(&self) -> usize {
        self.out.lock().unwrap().len()
    }

    /// The last `n` bytes written, without copying the whole history.
    pub fn output_tail(&self, n: usize) -> String {
        let o = self.out.lock().unwrap();
        String::from_utf8_lossy(&o[o.len().saturating_sub(n)..]).to_string()
    }

    /// Decode a MACE ISA address into a 16550 register number, if it names one.
    fn reg_of(addr: u32) -> Option<u32> {
        // Only the +7 lane carries the byte.
        if addr & 7 != 7 {
            return None;
        }
        let n = (addr >> 8) & 7;
        Some(n)
    }

    fn dlab(&self) -> bool {
        self.regs.lock().unwrap()[com_reg::LCR as usize] & com_reg::LCR_DLAB != 0
    }

    /// True while MCR selects internal loopback.
    fn looped(&self) -> bool {
        self.regs.lock().unwrap()[com_reg::MCR as usize] & com_reg::MCR_LOOP != 0
    }

    /// What the chip wants serviced, in the form the driver reads it.
    ///
    /// A kernel does not poll: it fills the transmit FIFO and waits to be
    /// told it has drained. Reporting "nothing pending" here leaves it
    /// waiting forever, and the visible symptom is output that stops in the
    /// middle of a word.
    fn iir(&self) -> u8 {
        let ier = self.regs.lock().unwrap()[com_reg::IER as usize];
        if ier & com_reg::IER_RX != 0 && !self.input.lock().unwrap().is_empty() {
            return com_reg::IIR_RX_DATA;
        }
        if ier & com_reg::IER_TX != 0 {
            // The transmitter is always drained here; there is no wire.
            return com_reg::IIR_TX_EMPTY;
        }
        com_reg::IIR_NONE
    }

    /// True while the chip is asking for attention.
    /// Wire the UART's interrupt (the console's: MACE ISA port 0's group).
    pub fn connect_irq(&self, line: crate::irq::IrqLine) {
        let _ = self.irq.set(line);
    }

    /// Drive the interrupt from the UART's state.
    pub fn update_irq(&self) {
        if let Some(l) = self.irq.get() {
            l.set(self.interrupting());
        }
    }

    pub fn interrupting(&self) -> bool {
        self.iir() & com_reg::IIR_NONE == 0
    }

    pub fn read_reg(&self, addr: u32) -> u8 {
        let Some(n) = Self::reg_of(addr) else { return 0 };
        match n {
            // Reading IIR is how the driver asks what to do next, and on a
            // 16550 it also acknowledges a transmit interrupt.
            com_reg::IIR_FCR => {
                let fifo = if self.regs.lock().unwrap()[com_reg::IIR_FCR as usize] & 1 != 0 {
                    com_reg::IIR_FIFO
                } else {
                    0
                };
                self.iir() | fifo
            }
            // Always ready to take another byte, and data-ready whenever
            // something has been typed.
            com_reg::LSR => {
                self.pump();
                let mut v = com_reg::LSR_THRE | com_reg::LSR_TEMT;
                if !self.input.lock().unwrap().is_empty() {
                    v |= com_reg::LSR_DR;
                }
                v
            }
            // Register 0 reads the receive buffer (unless DLAB selects the
            // divisor, which nothing reads back).
            com_reg::THR if !self.dlab() => {
                self.input.lock().unwrap().pop_front().unwrap_or(0)
            }
            // In loopback the modem control outputs appear as the status
            // inputs. Nothing in POST depends on this, but a half-wired
            // loopback is the kind of thing that misleads the next reader.
            com_reg::MSR if self.looped() => {
                let mcr = self.regs.lock().unwrap()[com_reg::MCR as usize];
                let mut v = 0;
                for (out, inp) in [
                    (com_reg::MCR_RTS, com_reg::MSR_CTS),
                    (com_reg::MCR_DTR, com_reg::MSR_DSR),
                    (com_reg::MCR_OUT1, com_reg::MSR_RI),
                    (com_reg::MCR_OUT2, com_reg::MSR_DCD),
                ] {
                    if mcr & out != 0 {
                        v |= inp;
                    }
                }
                v
            }
            _ => self.regs.lock().unwrap()[n as usize],
        }
    }

    pub fn write_reg(&self, addr: u32, val: u8) {
        let Some(n) = Self::reg_of(addr) else { return };
        if n == com_reg::THR && !self.dlab() {
            self.transmit(val);
            return;
        }
        self.regs.lock().unwrap()[n as usize] = val;
    }

    /// Send one character, as a write of the transmit register does. MACE's
    /// serial DMA engine comes here too, so a byte sent from the ring reaches
    /// the log and the TCP console by exactly the same road as one the
    /// firmware writes by hand.
    pub fn transmit(&self, val: u8) {
        // Loopback: the byte never leaves the chip, it arrives back in
        // the receive register. post1 writes 0..254 this way and insists
        // on reading the same sequence back.
        if self.looped() {
            self.input.lock().unwrap().push_back(val);
            return;
        }
        self.out.lock().unwrap().push(val);
        let backend = self.backend.lock().unwrap().clone();
        backend.send_byte(val);
    }

    /// Set the modem control register, for a DMA pair that names it rather
    /// than the transmitter.
    pub fn write_mcr(&self, val: u8) {
        self.regs.lock().unwrap()[com_reg::MCR as usize] = val;
    }

    /// Hand up to `buf.len()` received characters to the DMA engine, oldest
    /// first, and say how many. They leave the queue, so the PIO path never
    /// sees them as well.
    pub fn take_input(&self, buf: &mut [u8]) -> usize {
        let mut q = self.input.lock().unwrap();
        let n = buf.len().min(q.len());
        for (b, c) in buf.iter_mut().zip(q.drain(..n)) {
            *b = c;
        }
        n
    }

    /// True when nothing has been received and not yet taken.
    pub(crate) fn input_empty(&self) -> bool {
        self.input.lock().unwrap().is_empty()
    }
}

impl Device for Com16550 {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(self.name.to_string(), format!("16550 UART: {0} status | {0} output [bytes] | {0} type <text>", self.name))]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != self.name {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let r = *self.regs.lock().unwrap();
                writeln!(w, "{}: IER {:#04x} LCR {:#04x} MCR {:#04x} SCR {:#04x}{}{}; {} bytes out, {} waiting in",
                    self.name, r[com_reg::IER as usize], r[com_reg::LCR as usize], r[4], r[7],
                    if self.looped() { ", loopback" } else { "" }, if self.interrupting() { ", interrupting" } else { "" },
                    self.bytes_out(), self.pending_input()).ok();
            }
            "output" => {
                let n = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(2000);
                write!(w, "{}", self.output_tail(n)).ok();
                writeln!(w).ok();
            }
            "type" => {
                let text = args[1..].join(" ").replace("\\r", "\r").replace("\\n", "\n");
                self.feed(text.as_bytes());
                writeln!(w, "{}: queued {} bytes", self.name, text.len()).ok();
            }
            _ => return Err(format!("Usage: {0} status | output [bytes] | type <text>", self.name)),
        }
        Ok(())
    }
}

/// The registers and what has been typed but not yet read. The transcript
/// is the host's record of the line, not the UART's state.
impl Saveable for Com16550 {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("regs".into(), u8_slice_to_toml(&*self.regs.lock().unwrap()));
        let input: Vec<u8> = self.input.lock().unwrap().iter().copied().collect();
        t.insert("input".into(), u8_slice_to_toml(&input));
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(r) = get_field(v, "regs") {
            load_u8_slice(r, &mut *self.regs.lock().unwrap());
        }
        if let Some(toml::Value::Array(a)) = get_field(v, "input") {
            *self.input.lock().unwrap() = a.iter().map(|b| toml_u8(b).unwrap_or(0)).collect();
        }
        self.update_irq();
        Ok(())
    }
}

impl Resettable for Com16550 {
    /// Registers to zero; the transcript and anything a console has typed
    /// stay, as they would on the far end of a real serial line.
    fn power_on(&self) {
        *self.regs.lock().unwrap() = [0u8; 8];
    }
}

impl Com16550 {
    fn access_read(&self, addr: u64) -> u8 {
        let v = self.read_reg((addr | 7) as u32);
        self.update_irq();
        v
    }

    fn access_write(&self, addr: u64, val: u8) -> u32 {
        self.write_reg((addr | 7) as u32, val);
        self.update_irq();
        BUS_OK
    }
}

impl BusDevice for Com16550 {
    // Every width must be answered. An unimplemented one falls through to the
    // trait default, which reports a bus error — and a DBE inside POST looks
    // like a missing device rather than a missing method.
    // The UART sits on MACE's 8-bit ISA bus, so its byte answers on any lane
    // of the doubleword: IRIX and the BSDs use +7, Linux's 8250 driver +0
    // (mapbase 0x1f390000, regshift 8).
    // Reading the data or IIR register and writing IER can each move the
    // interrupt, so every access drives it again: nothing else does on an
    // access (the service thread covers input arriving from the host).
    fn read8(&self, addr: u64) -> BusRead8 { BusRead8::ok(self.access_read(addr)) }
    fn read16(&self, addr: u64) -> BusRead16 { BusRead16::ok(self.access_read(addr) as u16) }
    fn write16(&self, addr: u64, val: u16) -> u32 { self.access_write(addr, val as u8) }
    fn write8(&self, addr: u64, val: u8) -> u32 { self.access_write(addr, val) }
    fn read32(&self, addr: u64) -> BusRead32 { BusRead32::ok(self.access_read(addr) as u32) }
    fn write32(&self, addr: u64, val: u32) -> u32 { self.access_write(addr, val as u8) }
    fn read64(&self, addr: u64) -> BusRead64 { BusRead64::ok(self.access_read(addr) as u64) }
    fn write64(&self, addr: u64, val: u64) -> u32 { self.access_write(addr, val as u8) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bus address of a byte in the first MACE serial port.
    fn com0(off: u32) -> u64 {
        MACE_COM0 as u64 + off as u64
    }

    /// The TCP console, through the shared serial backend: a client gets
    /// the telnet handshake, then every byte the guest transmits, and what
    /// it types becomes the guest's input -- with telnet's own commands
    /// taken out on the way.
    #[test]
    fn a_tcp_console_carries_both_directions() {
        use std::io::{Read, Write};
        let c = Com16550::new();
        let (backend, port) = crate::dev::z85c30::tcp_serial_backend(0).expect("bind");
        c.set_backend(backend);
        let mut client = std::net::TcpStream::connect(("127.0.0.1", port)).expect("connect");
        client.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
        // The backend takes the client on one of its polls, and greets it
        // then: keep polling until the greeting is all here.
        let until = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut hs = [0u8; 12];
        let mut got = 0;
        client.set_read_timeout(Some(std::time::Duration::from_millis(20))).unwrap();
        while got < hs.len() && std::time::Instant::now() < until {
            c.pump();
            match client.read(&mut hs[got..]) {
                Ok(n) => got += n,
                Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                Err(e) => panic!("handshake: {e}"),
            }
        }
        client.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
        assert_eq!(hs, crate::net::telnet::TelnetFilter::initial_handshake());
        // Typed, with a telnet command (IAC DO SGA, which it already offered) in between.
        client.write_all(&[b'l', 0xff, 0xfd, 0x03, b's', b'\r']).unwrap();
        while c.pending_input() < 3 && std::time::Instant::now() < until {
            c.pump();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(c.read8(com0(7)).data, b'l');
        assert_eq!(c.read8(com0(7)).data, b's');
        assert_eq!(c.read8(com0(7)).data, b'\r');
        assert_eq!(c.pending_input(), 0, "the telnet command is not input");
        // Transmitted: reaches the client.
        for &b in b"ok" {
            c.write8(com0(7), b);
        }
        let mut got = [0u8; 2];
        client.read_exact(&mut got).expect("output");
        assert_eq!(&got, b"ok");
        assert_eq!(c.output(), "ok", "the console buffer still has it too");
    }

    #[test]
    fn save_load_round_trip() {
        let src = Com16550::new();
        src.write8(com0((com_reg::IER << 8) + 7), com_reg::IER_RX);
        src.write8(com0((com_reg::LCR << 8) + 7), 0x03);
        src.write8(com0((com_reg::SCR << 8) + 7), 0x5a);
        src.feed(b"typed");
        let v1 = src.save_state();

        let dst = Com16550::new();
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "Com16550 save_state mismatch after load_state round-trip");
        assert_eq!(dst.read8(com0(7)).data, b't', "unread input survives");
    }
}
