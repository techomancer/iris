//! The serial ports of the running machines and what each is connected to,
//! for the monitor's `serial`.
//!
//! A machine wraps each port's host end with [`mapped`] as it wires it: the
//! port is in the map for as long as that backend lives, so it leaves with the
//! machine. The map keeps only a weak reference, and asks the backend to
//! describe itself when it is printed, so it shows who is connected and the
//! line a host serial port is set to now, not at start.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::config::MachineConfig;
use crate::dev::serial_host::{LineConfig, ModemIn, ModemOut, SerialBackend};

struct Entry {
    id: u64,
    machine: String,
    /// The part and port on the board: "SCC channel B", "IOC4 0", "L1".
    port: String,
    /// N of IRIX's /dev/ttydN, if IRIX names the port.
    ttyd: Option<u8>,
    console: bool,
    /// The config section that put a device on the port.
    section: Option<&'static str>,
    backend: Weak<dyn SerialBackend>,
}

static MAP: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
static NEXT: AtomicU64 = AtomicU64::new(1);

/// A port's host end that keeps the port in the map while it lives.
struct Mapped {
    inner: Arc<dyn SerialBackend>,
    id: u64,
}

impl Drop for Mapped {
    fn drop(&mut self) {
        MAP.lock().retain(|e| e.id != self.id);
    }
}

impl SerialBackend for Mapped {
    fn send_byte(&self, byte: u8) {
        self.inner.send_byte(byte)
    }
    fn set_line(&self, line: LineConfig) {
        self.inner.set_line(line)
    }
    fn set_modem(&self, out: ModemOut) {
        self.inner.set_modem(out)
    }
    fn modem_in(&self) -> ModemIn {
        self.inner.modem_in()
    }
    fn bind_timers(&self, tm: &std::sync::Arc<crate::hptimer::TimerManager>) {
        self.inner.bind_timers(tm)
    }
    fn raw(&self) -> bool {
        self.inner.raw()
    }
    fn describe(&self) -> String {
        self.inner.describe()
    }
    fn flush(&self) {
        self.inner.flush()
    }
    fn queued(&self) -> bool {
        self.inner.queued()
    }
    fn dropped(&self) -> u64 {
        self.inner.dropped()
    }
    fn recv_byte(&self) -> io::Result<u8> {
        self.inner.recv_byte()
    }
}

/// Put a port in the map and return its host end to wire: `machine` and
/// `port` name it ("Indy", "SCC channel B"), `ttyd` is IRIX's number for it,
/// `console` marks the console port.
pub fn mapped(machine: &str, port: &str, ttyd: Option<u8>, console: bool, cfg: &MachineConfig, backend: Arc<dyn SerialBackend>) -> Arc<dyn SerialBackend> {
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let section = ttyd.and_then(|n| crate::dev::serial_dev::section_at(cfg, n));
    MAP.lock().push(Entry {
        id,
        machine: machine.to_string(),
        port: port.to_string(),
        ttyd,
        console,
        section,
        backend: Arc::downgrade(&backend),
    });
    Arc::new(Mapped { inner: backend, id })
}

/// What `serial` prints: each machine's ports in the order they were wired.
pub fn report() -> String {
    let map = MAP.lock();
    if map.is_empty() {
        return "No serial ports.\n".into();
    }
    let pw = map.iter().map(|e| e.port.len()).max().unwrap_or(0);
    let mut out = String::new();
    let mut machine = None;
    for e in map.iter() {
        if machine != Some(&e.machine) {
            out += &format!("{}:\n", e.machine);
            machine = Some(&e.machine);
        }
        let name = e.ttyd.map_or_else(|| "-".to_string(), |n| format!("ttyd{n}"));
        let host = e.backend.upgrade().map_or_else(|| "gone".to_string(), |b| b.describe());
        let host = match e.section {
            Some(s) => format!("[{s}] {host}"),
            None => host,
        };
        out += &format!("  {name:6} {:pw$}  {:7}  {host}\n", e.port, if e.console { "console" } else { "" });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::serial_host::null_serial_backend;

    /// A port is listed while its backend lives, with its IRIX name and what
    /// the backend says it is, and leaves the list when the backend goes.
    #[test]
    fn a_port_is_listed_while_its_backend_lives() {
        let cfg = MachineConfig::default();
        let a = mapped("Test machine", "UART 0", Some(1), true, &cfg, null_serial_backend());
        let b = mapped("Test machine", "UART 2", None, false, &cfg, null_serial_backend());
        let text = report();
        assert!(text.contains("Test machine:"), "{text}");
        assert!(text.lines().any(|l| l.contains("ttyd1") && l.contains("UART 0") && l.contains("console") && l.ends_with("nothing")), "{text}");
        assert!(text.lines().any(|l| l.trim_start().starts_with("-") && l.contains("UART 2")), "{text}");
        drop(a);
        drop(b);
        assert!(!report().contains("Test machine"), "gone with the machine");
    }
}
