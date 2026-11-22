//! The host sockets iris is listening on, for the monitor's `ports`.
//!
//! Whatever binds a host socket registers it here and keeps the returned
//! [`PortGuard`] next to the socket; dropping the guard (with the socket)
//! takes the entry out. The list is process-wide: the GUI can run more than
//! one machine's worth of sockets over its life, and `ports` shows them all.

use std::fmt::Display;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

/// One listening socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PortEntry {
    id: u64,
    /// `tcp`, `udp` or `unix`.
    pub proto: &'static str,
    /// Where it listens: `127.0.0.1:8888`, or a socket path.
    pub addr: String,
    /// What it is for.
    pub what: String,
}

static PORTS: Mutex<Vec<PortEntry>> = Mutex::new(Vec::new());
static NEXT: AtomicU64 = AtomicU64::new(1);

/// An entry in the list for as long as it lives.
#[must_use = "the port leaves the list when the guard drops"]
#[derive(Debug)]
pub struct PortGuard(u64);

impl Drop for PortGuard {
    fn drop(&mut self) {
        PORTS.lock().retain(|e| e.id != self.0);
    }
}

/// Add a listening socket to the list.
pub fn register(proto: &'static str, addr: impl Display, what: impl Into<String>) -> PortGuard {
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    PORTS.lock().push(PortEntry { id, proto, addr: addr.to_string(), what: what.into() });
    PortGuard(id)
}

/// Every socket listening now, in the order they were opened.
pub fn list() -> Vec<PortEntry> {
    PORTS.lock().clone()
}

/// What `ports` prints.
pub fn report() -> String {
    let ports = list();
    if ports.is_empty() {
        return "No listening sockets.\n".into();
    }
    let w = ports.iter().map(|p| p.addr.len()).max().unwrap_or(0);
    ports.iter().map(|p| format!("{:5} {:w$}  {}\n", p.proto, p.addr, p.what)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_port_is_listed_while_its_guard_lives() {
        let g = register("tcp", "127.0.0.1:65530", "test listener");
        assert!(list().iter().any(|p| p.addr == "127.0.0.1:65530" && p.what == "test listener"));
        assert!(report().contains("tcp   127.0.0.1:65530"));
        drop(g);
        assert!(!list().iter().any(|p| p.addr == "127.0.0.1:65530"));
    }
}
