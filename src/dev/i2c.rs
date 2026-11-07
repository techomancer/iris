//! A small untimed I2C bus: the targets on one master's wire.
//!
//! A transfer is not clocked bit by bit. The master starts at an address, the
//! bus finds the target that claims it (every target that takes a general
//! call, for address 0) and the bytes then move one call at a time. A repeated
//! start to the address already selected does not rescan, as on the wire.
//!
//! The bus holds no lock of its own; the controller that owns it serialises
//! access. Targets lock their own state, so the owner's lock is held across
//! target calls and a target must never call back into the controller.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Whether a transfer is open on a bus, readable without the controller's
/// lock. The controller stores it after every access; a target that decides
/// something by "is the bus free" (the system controller's grant ring and its
/// operator command line) loads it.
pub type BusBusy = Arc<AtomicBool>;

/// What a target asks of its master when serviced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Service {
    /// An address byte (seven-bit address and direction) to put on the wire as
    /// an address-only transaction nobody acknowledges.
    pub grant: Option<u8>,
    /// End the open transfer, as a target does that resets or powers down
    /// something while still selected.
    pub release_bus: bool,
}

/// One device on the bus.
pub trait I2cTarget: Send + Sync {
    /// Whether the target answers a directed start at `addr7`.
    fn claims(&self, addr7: u8) -> bool;

    /// Whether the target takes part in a general call (address 0).
    fn answers_general_call(&self) -> bool {
        true
    }

    /// A start or repeated start. Answering false refuses the address.
    fn start(&self, _addr7: u8, _recv: bool) -> bool {
        true
    }

    /// A byte from the master; false is a NAK.
    fn send(&self, byte: u8) -> bool;

    /// The next byte for the master.
    fn recv(&self) -> u8;

    /// The master NAKed the last byte it read.
    fn nack(&self) {}

    /// The transfer ended, with a stop or a start to another address.
    fn finish(&self) {}

    /// Let time pass for the target and collect what it wants of its master.
    /// Called around every register access of the controller.
    fn service(&self) -> Service {
        Service::default()
    }
}

#[derive(Default)]
pub struct I2cBus {
    targets: Vec<Arc<dyn I2cTarget>>,
    current: Vec<Arc<dyn I2cTarget>>,
    broadcast: bool,
}

impl I2cBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, target: Arc<dyn I2cTarget>) {
        self.targets.push(target);
    }

    pub fn targets(&self) -> &[Arc<dyn I2cTarget>] {
        &self.targets
    }

    /// A target is selected.
    pub fn busy(&self) -> bool {
        !self.current.is_empty()
    }

    pub fn is_broadcast(&self) -> bool {
        self.broadcast
    }

    /// Select the targets for a new transfer. Does nothing while a transfer
    /// is open, since a repeated start keeps the selection.
    pub fn scan(&mut self, addr7: u8) {
        if !self.current.is_empty() {
            return;
        }
        self.broadcast = addr7 == 0;
        if self.broadcast {
            self.current = self
                .targets
                .iter()
                .filter(|t| t.answers_general_call())
                .cloned()
                .collect();
        } else if let Some(t) = self.targets.iter().find(|t| t.claims(addr7)) {
            self.current.push(t.clone());
        }
    }

    /// Deliver the start to the selected targets. A refusal ends the transfer
    /// when `fresh` (this start opened it); a repeated start that is refused
    /// leaves the earlier transfer for the caller to end. A general call is
    /// never refused.
    pub fn start_selected(&mut self, addr7: u8, recv: bool, fresh: bool) -> bool {
        let mut refused = false;
        for t in &self.current {
            if !t.start(addr7, recv) && !self.broadcast {
                refused = true;
                break;
            }
        }
        if refused && fresh {
            self.end();
        }
        !refused
    }

    /// Send a byte to every selected target; false if any NAKed.
    pub fn send(&self, byte: u8) -> bool {
        let mut ok = true;
        for t in &self.current {
            ok = t.send(byte) && ok;
        }
        ok
    }

    /// A byte from the first selected target; a general call reads 0xff.
    pub fn recv(&self) -> u8 {
        match self.current.first() {
            Some(t) if !self.broadcast => t.recv(),
            _ => 0xff,
        }
    }

    pub fn nack(&self) {
        for t in &self.current {
            t.nack();
        }
    }

    /// End the transfer.
    pub fn end(&mut self) {
        for t in self.current.drain(..) {
            t.finish();
        }
        self.broadcast = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct Probe {
        addr: u8,
        general: bool,
        log: Mutex<Vec<String>>,
    }

    impl I2cTarget for Probe {
        fn claims(&self, addr7: u8) -> bool {
            addr7 == self.addr
        }
        fn answers_general_call(&self) -> bool {
            self.general
        }
        fn start(&self, addr7: u8, recv: bool) -> bool {
            self.log.lock().push(format!("start {addr7:#x} {recv}"));
            true
        }
        fn send(&self, byte: u8) -> bool {
            self.log.lock().push(format!("send {byte:#x}"));
            true
        }
        fn recv(&self) -> u8 {
            0x5a
        }
        fn finish(&self) {
            self.log.lock().push("finish".into());
        }
    }

    fn probe(addr: u8, general: bool) -> Arc<Probe> {
        Arc::new(Probe { addr, general, ..Default::default() })
    }

    #[test]
    fn directed_start_selects_the_claiming_target_only() {
        let (a, b) = (probe(0x08, true), probe(0x50, false));
        let mut bus = I2cBus::new();
        bus.add(a.clone());
        bus.add(b.clone());
        bus.scan(0x50);
        assert!(bus.busy());
        assert!(bus.start_selected(0x50, false, true));
        assert!(bus.send(1));
        bus.end();
        assert!(!bus.busy());
        assert!(a.log.lock().is_empty());
        assert_eq!(*b.log.lock(), ["start 0x50 false", "send 0x1", "finish"]);
    }

    #[test]
    fn general_call_reaches_only_targets_that_answer_it() {
        let (a, b) = (probe(0x08, true), probe(0x50, false));
        let mut bus = I2cBus::new();
        bus.add(a.clone());
        bus.add(b.clone());
        bus.scan(0);
        assert!(bus.is_broadcast());
        assert!(bus.send(7));
        assert_eq!(bus.recv(), 0xff);
        assert_eq!(*a.log.lock(), ["send 0x7"]);
        assert!(b.log.lock().is_empty());
    }

    #[test]
    fn unclaimed_address_selects_nothing() {
        let mut bus = I2cBus::new();
        bus.add(probe(0x08, true));
        bus.scan(0x2b);
        assert!(!bus.busy());
    }

    #[test]
    fn rescan_is_skipped_while_a_transfer_is_open() {
        let (a, b) = (probe(0x08, true), probe(0x09, true));
        let mut bus = I2cBus::new();
        bus.add(a);
        bus.add(b.clone());
        bus.scan(0x08);
        bus.scan(0x09);
        assert!(b.log.lock().is_empty());
        assert!(!bus.targets()[1].claims(0x08));
        assert_eq!(bus.recv(), 0x5a);
    }
}
