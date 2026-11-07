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
