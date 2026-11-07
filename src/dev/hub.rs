//! The Hub, the node controller of the SN0 machines (Origin 200, Origin 2000,
//! Onyx2): one per node, joining the node's processors, memory and directory,
//! one XIO port and one CrayLink port. The register window is split by
//! section into `hub_pi`, `hub_md`, `hub_io` and `hub_ni`, as the Bedrock
//! files are.
//!
//! This module holds what the sections share: the machine-facing hooks and
//! the observer of the stores a processor rendezvous depends on.
//!
//! # What is approximated here
//!
//! The model is a functional profile for the stock IP27 PROM and the guests
//! that boot on these machines. It is not a revision-matched description of the
//! chip, and no Hub specification is public: almost every register behaviour
//! is a software expectation or an approximation. The sections name theirs.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

/// Length of the register window: 8 MiB.
pub const WINDOW_SIZE: u64 = 8 << 20;

/// Pending interrupts are two banks of 64 bits.
pub const PENDING_BANKS: usize = 2;

/// A store the stock PROM's processor rendezvous depends on, reported after
/// it has taken effect. These are the registers the processors of a node, and
/// the nodes of a machine, use as mailboxes: a mechanism that holds a
/// processor until a partner has run needs to see them, and the Hub changes
/// no register value on its account.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Store {
    /// PI_RT_COMPARE of `slice`, as stored (52 bits).
    RtCompare { slice: usize, value: u64 },
    /// PI_CPU_ENABLE of `slice`, whether or not the node populates it.
    CpuEnable { slice: usize, enabled: bool },
    /// PI_INT_PEND_MOD, as written; `changed` says whether it flipped a
    /// pending bit. Interrupts from a Bridge or an engine arrive here too.
    IntPendMod { value: u64, changed: bool },
    /// IIO_SCRATCH_REG0 or 1, as stored (36 bits).
    IoScratch { index: usize, value: u64 },
    /// NI_SCRATCH_REG0 or 1, as stored. `vector` is set when a vector PIO
    /// from another node wrote it rather than a load or store on the bus.
    NiScratch { index: usize, value: u64, vector: bool },
    /// NI_PORT_RESET, as written.
    PortReset { value: u64 },
}

/// An optional observer of those stores, connected by the board. Nothing in
/// the Hub waits for it: it is called outside every section lock, on the
/// thread that did the store, and when none is connected a store costs one
/// atomic load.
pub trait HubObserver: Send + Sync {
    fn stored(&self, store: Store);
}

/// What the Hub asks of the machine around it. Called outside every section
/// lock, on the thread that wrote the register. All defaults do nothing, so a
/// test or a partial machine implements only what it wires.
pub trait HubHooks: Send + Sync {
    /// PI_CPU_ENABLE of `slice` was stored, or cleared by a reset. The stock
    /// PROM stores zero to its partner's enable after a failed liveness
    /// handshake and never stores one; what the store does to the processor is
    /// unknown. A machine that halts the processor on a cleared bit and
    /// releases it on a set one is making that choice. Slices the node does not
    /// populate are not reported.
    fn cpu_enable(&self, _slice: usize, _enabled: bool) {}
    /// A store to PI_NMI of `slice`: the processor takes a non-maskable
    /// interrupt.
    fn nmi(&self, _slice: usize) {}
    /// A store to PI_SOFTRESET reached the enabled processor `slice`.
    fn soft_reset(&self, _slice: usize) {}
    /// PI_CALIAS_SIZE was written: the cached-alias window of every processor
    /// view now spans `bytes` (0 disables it).
    fn cached_alias_changed(&self, _bytes: u64) {}
    /// The node number this Hub answers to changed, by a store to
    /// NI_STATUS_REV_ID or a reset.
    fn node_number_changed(&self, _number: u16) {}
    /// NI_PORT_RESET bit 0: restart the whole machine.
    fn machine_restart(&self) {}
}
