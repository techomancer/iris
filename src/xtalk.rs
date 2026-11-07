//! The chip that sits at the top of a widget bus and takes what the widgets
//! send upstream: interrupt packets and memory transfers.
//!
//! Its interrupt inputs are an [`InterruptController`](crate::irq::InterruptController)
//! like any other. What the widgets also need to know is how it is
//! addressed: the widget number their requests name, where in that
//! widget's space an interrupt packet is accepted, and how many
//! vectors a packet can name. A machine states these once when it builds the
//! Bridge and the Crossbow, and they compare against them where they used to
//! compare against constants.

use std::sync::Arc;

use crate::irq::InterruptController;
use crate::traits::BusDevice;

/// How a widget bus's host chip is addressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XtalkHost {
    /// The widget number requests for the host name.
    pub widget: u8,
    /// Where in the host's widget space an interrupt packet is taken: the
    /// address bits in `int_packet_mask` must equal those of `int_packet`.
    /// A host that takes a packet at any address has a zero mask.
    pub int_packet: u64,
    pub int_packet_mask: u64,
    /// Vectors a packet can name: 0..vectors.
    pub vectors: u16,
}

impl XtalkHost {
    /// Whether an interrupt packet to `widget` at `address` naming `vector`
    /// is one the host takes.
    #[inline]
    pub fn accepts(&self, widget: u8, address: u64, vector: u8) -> bool {
        widget == self.widget && address & self.int_packet_mask == self.int_packet && (vector as u16) < self.vectors
    }
}

/// A host chip as the widgets upstream of it reach it: how it is addressed,
/// the memory a transfer to it lands in (physical addresses, a bus error for
/// anything not fitted) and the interrupt inputs a packet to it sets (line =
/// vector).
pub struct HostPort {
    pub host: XtalkHost,
    pub mem: Arc<dyn BusDevice>,
    pub irq: Arc<dyn InterruptController>,
    /// Whether the link to the host is up, for a host behind a Crossbow.
    /// None is a host with no link to lose.
    link: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

impl HostPort {
    pub fn new(host: XtalkHost, mem: Arc<dyn BusDevice>, irq: Arc<dyn InterruptController>) -> Self {
        Self { host, mem, irq, link: None }
    }

    /// The same host behind a link that is up while `up` says so. The Bridge
    /// asks once per transfer chunk and once per interrupt packet, never per
    /// bus word.
    pub fn with_link(mut self, up: Arc<dyn Fn() -> bool + Send + Sync>) -> Self {
        self.link = Some(up);
        self
    }

    /// Whether a request to the host gets through.
    #[inline]
    pub fn link_up(&self) -> bool {
        self.link.as_ref().is_none_or(|up| up())
    }
}

/// The hosts a widget can send to, by widget number. A machine with one host
/// (HEART, a Hub with no Crossbow) has one entry. A widget behind a Crossbow
/// has one per host chip on its ports, each with that port's link
/// (`Xbow::upstream`), so a request to a host with no live link is dropped
/// where the Crossbow would drop it.
///
/// The table is fixed when the machine is built and indexed by the widget
/// number the guest programmed, so a transfer finds its host with one load
/// and no lock.
pub struct Upstream {
    hosts: [Option<HostPort>; 16],
}

impl Upstream {
    pub fn new() -> Self {
        Self { hosts: [const { None }; 16] }
    }

    /// An upstream with the one host `port`.
    pub fn single(host: XtalkHost, mem: Arc<dyn BusDevice>, irq: Arc<dyn InterruptController>) -> Self {
        Self::new().with(HostPort::new(host, mem, irq))
    }

    /// Add a host at its widget number.
    pub fn with(mut self, port: HostPort) -> Self {
        let w = port.host.widget as usize;
        assert!(w < 16, "widget {w} is not a widget number");
        assert!(self.hosts[w].is_none(), "two hosts at widget {w}");
        self.hosts[w] = Some(port);
        self
    }

    /// The host at `widget`, if there is one.
    #[inline]
    pub fn get(&self, widget: u8) -> Option<&HostPort> {
        self.hosts.get(widget as usize)?.as_ref()
    }
}

impl Default for Upstream {
    fn default() -> Self {
        Self::new()
    }
}
