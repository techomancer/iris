//! The serial ports of the running machines and what each is connected to,
//! for the monitor's `devices`.
//!
//! A machine wraps each port's host end with [`mapped`] as it wires it: the
//! port is in the map for as long as that backend lives, so it leaves with the
//! machine. The map keeps only a weak reference, and asks the backend to
//! describe itself when it is asked, so it shows who is connected and the
//! line a host serial port is set to now, not at start.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::config::{MachineConfig, MachineProfile};
use crate::dev::serial_host::{LineConfig, ModemIn, ModemOut, SerialBackend};

struct Entry {
    id: u64,
    profile: MachineProfile,
    /// The part and port on the board: "SCC channel B", "IOC4 0", "L1".
    port: String,
    /// N of IRIX's /dev/ttydN, if IRIX names the port.
    ttyd: Option<u8>,
    console: bool,
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

/// Put a port in the map and return its host end to wire: `port` names it ("SCC channel B"), `ttyd` is IRIX's number for
/// it, `console` marks the console port.
pub fn mapped(port: &str, ttyd: Option<u8>, console: bool, cfg: &MachineConfig, backend: Arc<dyn SerialBackend>) -> Arc<dyn SerialBackend> {
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    MAP.lock().push(Entry { id, profile: cfg.machine.profile, port: port.to_string(), ttyd, console, backend: Arc::downgrade(&backend) });
    Arc::new(Mapped { inner: backend, id })
}

/// What the newest machine of `profile` has on `/dev/ttydN`: the board's port, the console mark and what it is connected to.
pub fn host_end(profile: MachineProfile, ttyd: u8) -> Option<String> {
    let map = MAP.lock();
    let e = map.iter().rev().find(|e| e.profile == profile && e.ttyd == Some(ttyd))?;
    let host = e.backend.upgrade().map_or_else(|| "gone".to_string(), |b| b.describe());
    Some(format!("{}{}: {host}", e.port, if e.console { " (console)" } else { "" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::serial_host::null_serial_backend;

    /// A port is found while its backend lives, with the board's port, the console mark and what the backend says it is, and
    /// leaves the map when the backend goes.
    #[test]
    fn a_port_is_found_while_its_backend_lives() {
        let mut cfg = MachineConfig::default();
        cfg.machine.profile = MachineProfile::Origin3200Ip35;
        let a = mapped("UART 0", Some(7), true, &cfg, null_serial_backend());
        assert_eq!(host_end(MachineProfile::Origin3200Ip35, 7).as_deref(), Some("UART 0 (console): nothing"));
        assert_eq!(host_end(MachineProfile::Onyx3000Ip35, 7), None);
        drop(a);
        assert_eq!(host_end(MachineProfile::Origin3200Ip35, 7), None, "gone with the machine");
    }
}
