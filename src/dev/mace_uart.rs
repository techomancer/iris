//! The O2's two 16550s in MACE's ISA space: the lane decode, the interrupt
//! line and the monitor around the shared [`Ns16550`] core.
//!
//! Register `n` lives at `base + (n << 8) + 7`: MACE spaces its ISA registers
//! 256 bytes apart and puts the byte in the last lane of the 64-bit slot
//! (`sgimips/bus.c`: `h + (o << 8) + 7`). Getting this wrong is silent: every
//! register reads back zero and the firmware simply never transmits.

use std::io::Write;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};

use crate::dev::ns16550::{com_reg, Ns16550};
use crate::dev::serial_host::SerialBackend;
use crate::hptimer::{Timer, TimerManager};
use crate::ip32::*;
use crate::irq::IrqLine;
use crate::net::wake::{host_io, HostPump};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, Resettable, Saveable, BUS_OK};

/// The TL16PIR552's reference over 16 until the guest programs the prescaler.
const BAUDBASE: u32 = 115200;
/// The clock MACE gives the TL16PIR552, which its prescaler divides: 22 MHz.
/// IRIX picks a divide of 3 (7.33 MHz) for the ordinary rates.
const UART_CLOCK_HZ: u32 = 22_000_000;

pub struct MaceUart {
    core: Mutex<Ns16550>,
    /// Which UART this is (`com0`, `com1`): its monitor command.
    name: &'static str,
    /// Its interrupt, while it is driven by PIO (MACE ISA, port 0's group).
    irq: OnceLock<IrqLine>,
    /// The receive timeout, on the machine clock.
    timer: Timer,
    me: Weak<Self>,
    /// The host-IO pump, while the machine runs.
    host: Mutex<Option<HostPump>>,
}

impl MaceUart {
    pub fn new(tm: &Arc<TimerManager>) -> Arc<Self> {
        Self::named("com", tm)
    }

    /// A UART answering monitor commands as `name`.
    pub fn named(name: &'static str, tm: &Arc<TimerManager>) -> Arc<Self> {
        Arc::new_cyclic(|me| {
            let mut core = Ns16550::new(BAUDBASE, tm.clock().clone());
            core.set_ti_prescaler(UART_CLOCK_HZ);
            Self { core: Mutex::new(core), name, irq: OnceLock::new(), timer: Timer::new(tm, me, |u: &Self, _| _ = u.pump()), me: me.clone(), host: Mutex::new(None) }
        })
    }

    /// The 16550 itself, for the serial DMA engine in front of it.
    pub fn core(&self) -> MutexGuard<'_, Ns16550> {
        self.core.lock().unwrap()
    }

    /// Wire the host end of the line: a TCP console, a log tee, or nothing.
    pub fn set_backend(&self, backend: Arc<dyn SerialBackend>) {
        self.timer.manager().inspect(|tm| backend.bind_timers(tm));
        self.core().set_backend(backend);
    }

    /// Move what the host end typed into the receive FIFO and let the receive timeout run
    /// out; whether any input arrived.
    pub fn pump(&self) -> bool {
        let (moved, changed) = {
            let mut c = self.core();
            let moved = c.pump();
            (moved, moved | c.check_timeout())
        };
        if changed {
            self.update_irq();
        }
        moved
    }

    /// Queue characters for the guest to read, as if they arrived on the line.
    pub fn feed(&self, bytes: &[u8]) {
        self.core().type_in(bytes);
        self.update_irq();
    }

    /// Hand up to `buf.len()` received characters to the DMA engine, oldest
    /// first, and say how many. They leave the UART, so the PIO path never
    /// sees them as well.
    pub fn take_input(&self, buf: &mut [u8]) -> usize {
        let mut c = self.core();
        let mut n = 0;
        while n < buf.len() {
            let Some((b, _)) = c.dma_read() else { break };
            buf[n] = b;
            n += 1;
        }
        n
    }

    /// True when nothing has been received and not yet taken.
    pub(crate) fn input_empty(&self) -> bool {
        let mut c = self.core();
        c.pump();
        c.rx_pending() == 0
    }

    /// Characters still unread by the guest.
    pub fn pending_input(&self) -> usize {
        self.core().rx_pending()
    }

    /// What was written to the transmit register, the last
    /// [`crate::dev::ns16550::OUT_KEEP`] bytes of it.
    pub fn output(&self) -> String {
        self.output_tail(crate::dev::ns16550::OUT_KEEP)
    }

    /// Bytes sent in all, including those no longer kept.
    pub fn bytes_out(&self) -> usize {
        self.core().bytes_out() as usize
    }

    /// The last `n` bytes written.
    pub fn output_tail(&self, n: usize) -> String {
        self.core().output_tail(n)
    }

    /// Wire the UART's interrupt (the console's: MACE ISA port 0's group).
    pub fn connect_irq(&self, line: IrqLine) {
        let _ = self.irq.set(line);
    }

    /// Drive the interrupt and the timeout timer from the UART's state.
    pub fn update_irq(&self) {
        // Under the lock, so an older deadline cannot land after a newer one.
        let level = {
            let c = self.core();
            self.timer.set_due(c.timeout_due());
            c.irq()
        };
        if let Some(l) = self.irq.get() {
            l.set(level);
        }
    }

    pub fn interrupting(&self) -> bool {
        self.core().irq()
    }

    /// Decode a MACE ISA address into a 16550 register number, if it names one.
    fn reg_of(addr: u32) -> Option<u32> {
        // Only the +7 lane carries the byte.
        if addr & 7 != 7 {
            return None;
        }
        Some((addr >> 8) & 7)
    }

    pub fn read_reg(&self, addr: u32) -> u8 {
        let Some(n) = Self::reg_of(addr) else { return 0 };
        self.core().read(n)
    }

    pub fn write_reg(&self, addr: u32, val: u8) {
        let Some(n) = Self::reg_of(addr) else { return };
        self.core().write(n, val);
    }

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

impl Device for MaceUart {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {
        self.timer.stop();
        self.host.lock().unwrap().take();
    }
    fn start(&self) {
        self.timer.start();
        let me = self.me.clone();
        self.host.lock().unwrap().get_or_insert_with(|| host_io().add(move || me.upgrade().is_some_and(|u| u.pump())));
    }
    fn is_running(&self) -> bool { true }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![(self.name.to_string(), format!("16550 UART: {0} status | {0} output [bytes] | {0} type <text>", self.name))]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != self.name {
            return Err("Command not found".to_string());
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                let c = self.core();
                writeln!(w, "{}: IER {:#04x} LCR {:#04x} MCR {:#04x} LSR {:#04x} SCR {:#04x}{}{}{}; {} bytes out, {} dropped for clients that did not read, {} waiting in",
                    self.name, c.ier(), c.lcr(), c.mcr(), c.lsr(), c.scr(),
                    if c.loopback() { ", loopback" } else { "" },
                    if c.fifos_enabled() { ", FIFOs" } else { "" },
                    if c.irq() { ", interrupting" } else { "" },
                    c.bytes_out(), c.host_dropped(), c.rx_pending()).ok();
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

/// The 16550's registers and what has been received but not yet read. The
/// transcript is the host's record of the line, not the UART's state.
impl Saveable for MaceUart {
    fn save_state(&self) -> toml::Value {
        self.core().save_state()
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        self.core().load_state(v);
        self.update_irq();
        Ok(())
    }
}

impl Resettable for MaceUart {
    /// Registers to their reset values; the transcript and anything a
    /// console has typed stay in the host's hands.
    fn power_on(&self) {
        self.core().reset();
        self.update_irq();
    }
}

impl BusDevice for MaceUart {
    // Every width must be answered. An unimplemented one falls through to the
    // trait default, which reports a bus error, and a DBE inside POST looks
    // like a missing device rather than a missing method.
    // The UART sits on MACE's 8-bit ISA bus, so its byte answers on any lane
    // of the doubleword: IRIX and the BSDs use +7, Linux's 8250 driver +0
    // (mapbase 0x1f390000, regshift 8).
    // Reading the data or IIR register and writing IER can each move the
    // interrupt, so every access drives it again: nothing else does on an
    // access (the host-IO thread covers input arriving from the host).
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
pub(crate) mod tests {
    use super::*;
    use crate::dev::ns16550::com_reg;
    use crate::irq::LevelProbe;
    use std::time::Duration;

    /// Register `n` is at `+ (n << 8) + 7`, the byte answers on any lane, and
    /// the port's interrupt reaches the line it was given.
    #[test]
    fn the_isa_register_spacing_maps_onto_the_core_and_drives_the_line() {
        let c = MaceUart::new(&Arc::new(TimerManager::new()));
        let (probe, line) = LevelProbe::line();
        c.connect_irq(line);
        let reg = |n: u32| MACE_COM0 as u64 + ((n as u64) << 8);
        c.write8(reg(com_reg::SCR) + 7, 0x5a);
        assert_eq!(c.read8(reg(com_reg::SCR) + 7).data, 0x5a);
        assert_eq!(c.read32(reg(com_reg::SCR)).data, 0x5a, "lane 0 reads the same byte");
        assert_eq!(c.read8(reg(com_reg::SCR) + 3).data, 0x5a, "so does any other");
        c.write8(reg(com_reg::IER) + 7, com_reg::IER_TX);
        assert!(probe.level(), "the transmit-empty cause is up");
        assert_eq!(c.read8(reg(com_reg::IIR_FCR) + 7).data & 0x0f, com_reg::IIR_TX_EMPTY);
        assert!(!probe.level(), "and reading the cause drops it");
    }

    #[test]
    fn a_saved_uart_loads_and_saves_the_same() {
        let src = MaceUart::new(&Arc::new(TimerManager::new()));
        src.write8(MACE_COM0 as u64 + ((com_reg::IIR_FCR as u64) << 8) + 7, 0x81);
        src.write8(MACE_COM0 as u64 + ((com_reg::IER as u64) << 8) + 7, com_reg::IER_RX);
        src.feed(b"typed");
        let saved = src.save_state();
        let dst = MaceUart::new(&Arc::new(TimerManager::new()));
        dst.load_state(&saved).expect("load");
        assert_eq!(saved, dst.save_state());
        assert_eq!(dst.read8(MACE_COM0 as u64 + 7).data, b't', "unread input survives");
    }

    /// The 16550's receive timeout reaches the line from the UART's own timer
    /// once the machine clock is four character times past the character; a
    /// stop keeps the deadline and a restore sets it again.
    #[test]
    fn the_receive_timeout_runs_on_a_timer_on_the_machine_clock() {
        use crate::clock::Clock;
        let clock = crate::test_board::Stepped::new();
        let reg = |n: u32| MACE_COM0 as u64 + ((n as u64) << 8) + 7;
        let armed = |u: &MaceUart| {
            u.write8(reg(com_reg::IIR_FCR), 0x81);
            u.write8(reg(com_reg::IER), com_reg::IER_RX);
            u.feed(b"x");
        };
        let u = MaceUart::new(&clock.tm);
        let (probe, line) = LevelProbe::line();
        u.connect_irq(line);
        armed(&u);
        u.start();
        let due = u.core().timeout_due().expect("the character started the timeout");
        assert_eq!(u.timer.due(), Some(due));
        clock.step(due - 1);
        assert!(!probe.level(), "not before the deadline");
        clock.step(1);
        assert!(probe.level(), "the timer raised the cause");

        let held = MaceUart::new(&clock.tm);
        let (probe, line) = LevelProbe::line();
        held.connect_irq(line);
        armed(&held);
        held.start();
        held.stop();
        clock.step(held.timer.due().unwrap() - held.timer.now());
        assert!(!probe.level() && held.timer.due().is_some(), "the UART's stop stops its timer");

        let (src, dst) = (MaceUart::new(&clock.tm), MaceUart::new(&clock.tm));
        armed(&src);
        dst.load_state(&src.save_state()).unwrap();
        assert_eq!(dst.timer.due(), dst.core().timeout_due());
        assert!(dst.timer.due().is_some(), "a restore sets the deadline again");
    }

    /// A byte the host types reaches the receive FIFO through the host-IO thread, with no pass from the service loop.
    #[test]
    fn host_input_reaches_the_fifo_through_the_host_io_thread() {
        let tm = Arc::new(TimerManager::new());
        let u = MaceUart::new(&tm);
        let host = Arc::new(crate::dev::serial_host::CiSerialBackend::new());
        u.set_backend(host.clone());
        u.start();
        host.push_host(b"x");
        assert!(crate::test_board::until(|| u.pending_input() == 1));
        u.stop();
    }

    struct Mace(Arc<MaceUart>);

    impl crate::dev::serial_host::tests::Port for Mace {
        fn write(&mut self, b: u8) {
            self.0.write8(MACE_COM0 as u64 + 7, b);
        }
        fn tx_ready(&mut self) -> bool {
            self.0.read8(MACE_COM0 as u64 + ((com_reg::LSR as u64) << 8) + 7).data & com_reg::LSR_THRE != 0
        }
        fn poll(&mut self) {
            self.0.pump();
        }
    }

    /// The O2's console UART on `backend`, for the shared serial contract.
    pub(crate) fn contract_port(backend: Arc<dyn SerialBackend>) -> Box<dyn crate::dev::serial_host::tests::Port> {
        let u = MaceUart::new(&Arc::new(TimerManager::new()));
        u.set_backend(backend);
        Box::new(Mace(u))
    }
}
