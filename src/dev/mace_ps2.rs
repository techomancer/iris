//! MACE's two PS/2 ports, around the shared `Ps2Controller`.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, Weak};

use std::io::Write;

use crate::ip32::*;
use crate::snapshot::{get_field, hex_u64, hex_u8, toml_u64, toml_u8};
use crate::traits::{BusDevice, BusRead8, BusRead16, BusRead32, BusRead64, Device, Resettable, Saveable, BUS_OK};

/// The pair of PS/2 ports, with nothing plugged into either.
///
/// We drive this machine on the serial console, so an empty port is the
/// honest model — but "empty" is not the same as "dead". A real controller
/// still drains its transmit register whether or not a keyboard is listening,
/// and the PROM waits for exactly that. Reporting zero makes every probe take
/// its full 20834-iteration timeout instead of returning at once.
pub struct MacePs2(Arc<Ps2Ports>);

impl std::ops::Deref for MacePs2 {
    type Target = Ps2Ports;
    fn deref(&self) -> &Ps2Ports {
        &self.0
    }
}

/// The ports themselves, shared with [`Ps2Wake`] so the device model's
/// "byte waiting" line can reach them.
pub struct Ps2Ports {
    ports: [Mutex<Ps2Port>; 2],
    /// The keyboard (port 0) and, when `mouse` is set, the mouse (port 1),
    /// as one PS/2 device model. None: nothing plugged in, which is how the
    /// serial-console machines run.
    dev: Option<std::sync::Arc<crate::dev::ps2::Ps2Controller>>,
    mouse: bool,
    /// The last bytes through either port, `(port, to the device?, byte)`,
    /// for working out a conversation that stalled.
    traffic: Mutex<std::collections::VecDeque<(u8, bool, u8, std::time::Instant)>>,
    /// Its four bits of MACE's ISA status (9..12): the ports' requests and
    /// poll flags.
    irq: std::sync::OnceLock<Vec<crate::irq::IrqLine>>,
}

struct Ps2Port {
    control: u64,
    /// The receive latch: it keeps the last byte, which a read returns again
    /// when nothing new has arrived (IRIX reads the ID twice that way).
    rx: u8,
    rx_full: bool,
    /// A byte written to TX that has not gone out yet (TxEN clear).
    tx: Option<u8>,
    /// Bytes the guest has sent, kept so a test can see the transmit path ran.
    sent: Vec<u8>,
}

impl Default for Ps2Port {
    fn default() -> Self {
        Self { control: ps2_ctl::CLKEN, rx: 0, rx_full: false, tx: None, sent: Vec::new() }
    }
}

/// CONTROL bits of a MACE PS/2 port.
pub mod ps2_ctl {
    pub const TX_ENABLE: u64 = 1 << 1;
    pub const TX_INT: u64 = 1 << 2;
    pub const RX_INT: u64 = 1 << 3;
    /// Clock enabled: while clear the device may not send.
    pub const CLKEN: u64 = 1 << 4;
    pub const RESET: u64 = 1 << 5;
}

thread_local! {
    /// Set while this thread holds a port and may take bytes from the device
    /// model, whose queue then drives its line again: [`Ps2Wake`] must not
    /// take the port a second time on the same thread, and needs not, since
    /// the holder drives the bits after.
    static IN_PORT: Cell<bool> = const { Cell::new(false) };
}

/// Marks this thread as inside a port until dropped.
struct InPort(bool);

impl InPort {
    fn enter() -> Self {
        InPort(IN_PORT.with(|c| c.replace(true)))
    }
}

impl Drop for InPort {
    fn drop(&mut self) {
        IN_PORT.with(|c| c.set(self.0));
    }
}

/// The device model's interrupt, "a byte is waiting", as the Indy's IOC
/// takes it: a key or mouse movement from the host latches into an empty
/// port and raises its bits at once, rather than when something next polls.
struct Ps2Wake(Weak<Ps2Ports>);

impl crate::irq::InterruptController for Ps2Wake {
    fn set_line(&self, _line: u32, level: bool) {
        if !level || IN_PORT.with(|c| c.get()) {
            return;
        }
        if let Some(ports) = self.0.upgrade() {
            ports.update_irq();
        }
    }
}

impl MacePs2 {
    pub fn new() -> Self {
        MacePs2(Arc::new(Ps2Ports::empty(None, false)))
    }

    /// With a keyboard on port 0 (and a mouse on port 1 if `mouse`).
    pub fn with_devices(mouse: bool) -> Self {
        MacePs2(Arc::new_cyclic(|me: &Weak<Ps2Ports>| {
            let wake = crate::irq::IrqLine::new(Arc::new(Ps2Wake(me.clone())), 0);
            let dev = Arc::new(crate::dev::ps2::Ps2Controller::new(Some(wake)));
            crate::traits::Device::start(&*dev);
            Ps2Ports::empty(Some(dev), mouse)
        }))
    }
}

impl Ps2Ports {
    fn empty(dev: Option<Arc<crate::dev::ps2::Ps2Controller>>, mouse: bool) -> Self {
        Self {
            ports: [Mutex::new(Ps2Port::default()), Mutex::new(Ps2Port::default())],
            dev,
            mouse,
            traffic: Mutex::new(std::collections::VecDeque::new()),
            irq: std::sync::OnceLock::new(),
        }
    }

    /// The device model, for the host side to push keys into.
    pub fn devices(&self) -> Option<&std::sync::Arc<crate::dev::ps2::Ps2Controller>> {
        self.dev.as_ref()
    }

    /// The device on port `p`, if one is plugged in.
    fn attached(&self, p: usize) -> Option<&std::sync::Arc<crate::dev::ps2::Ps2Controller>> {
        if p == 1 && !self.mouse { None } else { self.dev.as_ref() }
    }

    /// `(port, register)` for an address in the block, if it names one.
    fn decode(addr: u32) -> (usize, u32) {
        let off = addr - MACE_PS2;
        (((off >> 5) & 1) as usize, (off >> 3) & 3)
    }

    pub fn sent(&self, port: usize) -> Vec<u8> {
        self.ports[port].lock().unwrap().sent.clone()
    }

    /// Move the device's next byte into an empty latch, if the clock lets it.
    fn latch(&self, p: usize, port: &mut Ps2Port) {
        if port.rx_full || port.control & ps2_ctl::CLKEN == 0 || port.control & ps2_ctl::RESET != 0 {
            return;
        }
        if let Some(b) = self.attached(p).and_then(|d| d.pop_from(p == 1)) {
            port.rx = b;
            self.note(p, false, b);
            port.rx_full = true;
        }
    }

    /// Send the pending byte if the transmitter is enabled.
    fn transmit(&self, p: usize, port: &mut Ps2Port) {
        if port.control & ps2_ctl::TX_ENABLE == 0 || port.control & ps2_ctl::RESET != 0 {
            return;
        }
        if let Some(b) = port.tx.take() {
            self.note(p, true, b);
            if let Some(d) = self.attached(p) {
                d.write_to(b, p == 1);
            }
        }
    }

    pub fn read(&self, addr: u32) -> u64 {
        let (p, reg) = Self::decode(addr);
        let _in = InPort::enter();
        let mut port = self.ports[p].lock().unwrap();
        self.latch(p, &mut port);
        match reg {
            1 => {
                let v = port.rx as u64;
                port.rx_full = false;
                self.latch(p, &mut port);
                v
            }
            2 => port.control,
            3 => {
                let mut st = ps2_bit::CLOCK_SIGNAL;
                if port.control & ps2_ctl::CLKEN != 0 && port.control & ps2_ctl::RESET == 0 {
                    st |= ps2_bit::CLOCK_INHIBIT;
                }
                if port.tx.is_none() {
                    st |= ps2_bit::TX_EMPTY;
                }
                if port.rx_full {
                    st |= ps2_bit::RX_FULL;
                }
                st
            }
            _ => 0,
        }
    }

    pub fn write(&self, addr: u32, val: u64) {
        let (p, reg) = Self::decode(addr);
        let _in = InPort::enter();
        let mut port = self.ports[p].lock().unwrap();
        match reg {
            0 => {
                if port.sent.len() < 64 {
                    port.sent.push(val as u8);
                }
                if port.control & ps2_ctl::RESET == 0 {
                    port.tx = Some(val as u8);
                    self.transmit(p, &mut port);
                }
            }
            2 => {
                port.control = val & 0x3f;
                if port.control & ps2_ctl::RESET != 0 {
                    port.tx = None;
                    port.rx_full = false;
                }
                self.transmit(p, &mut port);
                self.latch(p, &mut port);
            }
            _ => {}
        }
    }

    fn note(&self, p: usize, out: bool, b: u8) {
        let mut t = self.traffic.lock().unwrap();
        if t.len() >= 4000 {
            t.pop_front();
        }
        t.push_back((p as u8, out, b, std::time::Instant::now()));
    }

    /// The recent traffic as text: `>` to a device, `<` from it, per port.
    pub fn traffic(&self) -> String {
        self.traffic
            .lock()
            .unwrap()
            .iter()
            .map(|&(p, out, b, _)| format!("{p}{}{b:02x}", if out { '>' } else { '<' }))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The recent traffic one byte per line, with host milliseconds since
    /// the first byte kept: to tell a periodic exchange from a reaction.
    pub fn traffic_timed(&self) -> String {
        let t = self.traffic.lock().unwrap();
        let Some(&(_, _, _, t0)) = t.front() else { return String::new() };
        t.iter()
            .map(|&(p, out, b, at)| {
                format!("{:9.1} {p}{}{b:02x}\n", at.duration_since(t0).as_secs_f64() * 1000.0, if out { '>' } else { '<' })
            })
            .collect()
    }

    /// The PS/2 bits of MACE's ISA interrupt status: 9 and 11 are the ports'
    /// interrupt requests (receive full with RxIEN, transmit empty with
    /// TxIEN), 10 and 12 their unmasked receive-full flags.
    /// Wire the ports' status bits into MACE's ISA interrupt controller.
    pub fn connect_irq(&self, isa: std::sync::Arc<dyn crate::irq::InterruptController>) {
        let _ = self.irq.set((9..=12).map(|b| crate::irq::IrqLine::new(isa.clone(), b)).collect());
    }

    /// Drive the ports' status bits. A level left high after the handler
    /// drained the port reads as a spurious interrupt to OpenBSD, so this
    /// follows every register access, and every byte the host sends
    /// ([`Ps2Wake`]).
    pub fn update_irq(&self) {
        let (Some(lines), Some(_)) = (self.irq.get(), self.dev.as_ref()) else { return };
        let bits = self.isa_bits();
        for l in lines {
            l.set(bits & (1 << l.line()) != 0);
        }
    }

    pub fn isa_bits(&self) -> u64 {
        let _in = InPort::enter();
        let mut bits = 0;
        for p in 0..2 {
            let mut port = self.ports[p].lock().unwrap();
            self.latch(p, &mut port);
            let irq = (port.control & ps2_ctl::RX_INT != 0 && port.rx_full)
                || (port.control & ps2_ctl::TX_INT != 0 && port.tx.is_none());
            if irq {
                bits |= 1 << (9 + 2 * p);
            }
            if port.rx_full {
                bits |= 1 << (10 + 2 * p);
            }
        }
        bits
    }
}

/// The bus side: every width reads the register's low bits and writes its
/// value, as MACE's 64-bit register slots do.
impl BusDevice for MacePs2 {
    fn read8(&self, addr: u64) -> BusRead8 { BusRead8::ok(self.read64(addr).data as u8) }
    fn read16(&self, addr: u64) -> BusRead16 { BusRead16::ok(self.read64(addr).data as u16) }
    fn read32(&self, addr: u64) -> BusRead32 { BusRead32::ok(self.read64(addr).data as u32) }
    fn read64(&self, addr: u64) -> BusRead64 {
        let v = self.read(addr as u32);
        self.update_irq();
        BusRead64::ok(v)
    }
    fn write8(&self, addr: u64, val: u8) -> u32 { self.write64(addr, val as u64) }
    fn write16(&self, addr: u64, val: u16) -> u32 { self.write64(addr, val as u64) }
    fn write32(&self, addr: u64, val: u32) -> u32 { self.write64(addr, val as u64) }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write(addr as u32, val);
        self.update_irq();
        BUS_OK
    }
}

/// `maceps2` shows the two ports; `ps2` goes to the shared controller
/// behind them (typing, the mouse), as the IOC passes its own through.
impl Device for MacePs2 {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        let mut c = vec![("maceps2".to_string(), "MACE PS/2 ports: maceps2 status | maceps2 traffic".to_string())];
        if let Some(d) = &self.dev {
            c.extend(d.register_commands());
        }
        c
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut w: Box<dyn std::io::Write + Send>) -> Result<(), String> {
        if cmd != "maceps2" {
            return match &self.dev {
                Some(d) => d.execute_command(cmd, args, w),
                None => Err("Command not found".to_string()),
            };
        }
        match args.first().copied().unwrap_or("status") {
            "status" => {
                for (p, name) in ["keyboard", "mouse"].iter().enumerate() {
                    let port = self.ports[p].lock().unwrap();
                    writeln!(w, "{name}: control {:#04x} rx {:#04x}{} tx {:?}, {}",
                        port.control, port.rx, if port.rx_full { " (full)" } else { "" }, port.tx,
                        if self.attached(p).is_some() { "attached" } else { "nothing plugged in" }).ok();
                }
            }
            "traffic" => {
                write!(w, "{}", self.traffic_timed()).ok();
            }
            _ => return Err("Usage: maceps2 status | traffic".to_string()),
        }
        Ok(())
    }
}

impl Resettable for MacePs2 {
    /// Both ports back to their power-on state; the devices behind them
    /// keep what they have queued.
    fn power_on(&self) {
        for p in &self.ports {
            *p.lock().unwrap() = Ps2Port::default();
        }
    }
}

/// Each port's latches, and the keyboard and mouse behind them (the shared
/// PS/2 model's own state). The traffic log and the sent bytes are for
/// debugging, not the ports' state.
impl Saveable for MacePs2 {
    fn save_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        let ports = self.ports.iter().map(|p| {
            let p = p.lock().unwrap();
            let mut e = toml::map::Map::new();
            e.insert("control".into(), hex_u64(p.control));
            e.insert("rx".into(), hex_u8(p.rx));
            e.insert("rx_full".into(), toml::Value::Boolean(p.rx_full));
            if let Some(b) = p.tx {
                e.insert("tx".into(), hex_u8(b));
            }
            toml::Value::Table(e)
        }).collect();
        t.insert("ports".into(), toml::Value::Array(ports));
        if let Some(d) = &self.dev {
            t.insert("devices".into(), d.save_state());
        }
        toml::Value::Table(t)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(toml::Value::Array(a)) = get_field(v, "ports") {
            for (port, e) in self.ports.iter().zip(a) {
                let mut p = port.lock().unwrap();
                p.control = get_field(e, "control").and_then(toml_u64).unwrap_or(ps2_ctl::CLKEN);
                p.rx = get_field(e, "rx").and_then(toml_u8).unwrap_or(0);
                p.rx_full = get_field(e, "rx_full").and_then(|b| b.as_bool()).unwrap_or(false);
                p.tx = get_field(e, "tx").and_then(toml_u8);
            }
        }
        if let (Some(d), Some(dv)) = (&self.dev, get_field(v, "devices")) {
            d.load_state(dv)?;
        }
        self.update_irq();
        Ok(())
    }
}

impl Default for MacePs2 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_load_round_trip() {
        let src = MacePs2::with_devices(true);
        {
            let mut p = src.ports[0].lock().unwrap();
            p.control = ps2_ctl::CLKEN | ps2_ctl::RX_INT | ps2_ctl::TX_ENABLE;
            p.rx = 0xfa;
            p.rx_full = true;
        }
        src.ports[1].lock().unwrap().tx = Some(0xf4);
        src.devices().unwrap().push_mouse_input(1, 3, -2, 0);
        let v1 = src.save_state();

        let dst = MacePs2::with_devices(true);
        dst.load_state(&v1).expect("load_state");
        assert_eq!(v1, dst.save_state(), "MacePs2 save_state mismatch after load_state round-trip");
    }
}
