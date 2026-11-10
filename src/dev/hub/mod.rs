//! The Hub, the node controller of the SN0 machines (Origin 200, Origin 2000,
//! Onyx2): one per node, joining the node's processors, memory and directory,
//! one XIO port and one CrayLink port. The register window is split by
//! section into the processor interface (`pi`), memory and directory (`md`),
//! I/O (`io`) and network (`ni`) units, private to this module; the module is
//! the window itself and what the units share. Machines reach the Hub only
//! through its methods.
//!
//! Address decode is data, not a flag tested per access. Each Hub keeps a
//! table from node number to the Hub that answers it, rebuilt when a node
//! number or the cable changes (the PROM numbers the nodes after reset);
//! the big windows are decoded by their translation entries and the cached
//! alias by its size. A machine places the views [`Hub::claim`],
//! [`Hub::cpu_memory`] and [`Hub::io_space`] in its address map.
//!
//! A Bridge reaches the Hub as an XIO host ([`Hub::host_port`]): DMA lands in
//! node memory and an interrupt packet sets one of 128 interrupt numbers.
//!
//! # What is approximated here
//!
//! The model is a functional profile for the stock IP27 PROM and the guests
//! that boot on these machines. It is not a revision-matched description of the
//! chip, and no Hub specification is public: almost every register behaviour
//! is a software expectation or an approximation. The sections name theirs.
//!
//! - A Hub whose module is powered down is the machine's to express: it
//!   unmaps the register window. The network port reads disabled at the peer.
//! - Failing, rather than timing out, an access to a node number no reachable
//!   Hub holds is an approximation.
//! - A processor rendezvous hold is not modelled.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

mod io;
mod md;
mod ni;
mod pi;

pub use self::md::SlotSignals;

/// Register offsets in the Hub's window, by unit. Machines and their tests
/// name registers by these; the units themselves are private.
pub mod reg {
    pub mod pi {
        pub use super::super::pi::reg::*;
    }
    pub mod md {
        pub use super::super::md::reg::*;
    }
    pub mod io {
        pub use super::super::io::reg::*;
    }
    pub mod ni {
        pub use super::super::ni::reg::*;
    }
}

/// Length of the register window: 8 MiB.
pub const WINDOW_SIZE: u64 = 8 << 20;

/// Pending interrupts are two banks of 64 bits.
pub const PENDING_BANKS: usize = 2;

/// The numbers the rest of the node decodes by, published by the unit that
/// owns each: the network unit stores the node number and the processor
/// interface the cached-alias size, each under its own lock as the register
/// changes, and any reader loads them without one. The Hub's address decode
/// reads them on every access, and the machine's node-zero routing reads them
/// when a store has changed one.
///
/// The node number is 9 bits in NI_STATUS_REV_ID (a store of 0x1ff reads
/// back whole), so it is a `u16`, not a byte.
pub struct Routing {
    number: AtomicU16,
    calias: AtomicU64,
}

impl Routing {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { number: AtomicU16::new(0), calias: AtomicU64::new(pi::calias_bytes(pi::CALIAS_RESET)) })
    }

    /// The node number the Hub answers to.
    pub fn number(&self) -> u16 {
        self.number.load(Ordering::Acquire)
    }

    /// Bytes the cached alias spans: the processor view's node-zero memory
    /// below it stays on this node.
    pub fn calias(&self) -> u64 {
        self.calias.load(Ordering::Acquire)
    }

    fn set_number(&self, number: u16) {
        self.number.store(number, Ordering::Release);
    }

    fn set_calias(&self, bytes: u64) {
        self.calias.store(bytes, Ordering::Release);
    }

    /// Publish a node number and alias directly, for a test of whatever
    /// reads them.
    #[cfg(test)]
    pub(crate) fn set(&self, number: u16, calias: u64) {
        self.set_number(number);
        self.set_calias(calias);
    }
}

use crate::dev::heart::Clock;
use crate::ip27::NodeZero;
use self::io::{Io, NodeFabric};
use self::md::Md;
use self::ni::{NetPeer, Ni};
use self::pi::Pi;
use crate::devlog::LogModule;
use crate::dev::w1;
use crate::irq::InterruptController;
use crate::hptimer::TimerManager;
use crate::physical::{read_bytes, write_bytes};
use crate::xtalk::{HostPort, XtalkHost};
use crate::traits::{
    BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, Resettable, BUS_BUSY, BUS_ERR, BUS_OK,
};
use crate::traits::MachineEvent;
use std::sync::atomic::{AtomicU16, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock, Weak};

/// What differs between Hubs: the processors the node populates, where its
/// slice A sits in the machine's numbering, the preprogrammed node number and
/// the revision fields. The widget identification and network status carry
/// separate revisions, and the model sets them separately.
#[derive(Clone, Copy, Debug)]
pub struct HubConfig {
    /// Populated processor sockets: 1 or 2.
    pub cpus: u8,
    /// Global index of this node's slice A, which PI_CPU_NUM subtracts.
    pub cpu_base: usize,
    /// NI_STATUS_REV_ID bits 7:4. The model reports 3 (Hub 2.1) on the
    /// Origin 200 and 5 (Hub 2.3) on the Origin 2000 rack and deskside and the
    /// Onyx2.
    pub ni_revision: u8,
    /// Widget identification bits 31:28. No capture of the field is known, so
    /// 1 is an approximation.
    pub widget_revision: u8,
    /// The 11-bit XIO manufacturer: 0x36.
    pub manufacturer: u16,
    /// What the board drives on the slot-identity signals.
    pub slot: SlotSignals,
}
impl Default for HubConfig {
    fn default() -> Self {
        Self { cpus: 1, cpu_base: 0, ni_revision: 3, widget_revision: 1, manufacturer: 0x36, slot: SlotSignals::default() }
    }
}

/// The four uncached address spaces a node-qualified processor address
/// selects, in the order of their attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Node memory, cached or uncached.
    Memory = 0,
    /// The reader's own Hub-special resources.
    HubSpecial = 1,
    /// Widgets and Hub registers.
    Io = 2,
    /// Fetch-and-op memory.
    Special = 3,
}
const SPACES: usize = 4;

/// Node numbers are bits 39:32 of a physical address in M mode, so each node
/// owns a 4 GiB window.
pub const NODE_SHIFT: u32 = 32;
const NODE_MASK: u64 = 0xff;
const NODE_SPAN: u64 = 1 << NODE_SHIFT;

/// Offsets in the I/O space of a node, from the base of its address range.
pub const IO_REGISTERS_LOCAL: u64 = 0x0100_0000;
pub const IO_REGISTERS_NAMED: u64 = 0x0180_0000;
pub const IO_BIG_WINDOWS: u64 = 0x2000_0000;
/// A doubleword read here fails at the issuing Hub whatever node it names.
pub const IO_BUS_ERROR_PROBE: u64 = 0x0240_0208;

/// Where in the XIO address a packet is taken as an interrupt-modify write:
/// bit 47 selects the I/O view, the node number is bits 45:38.
const XIO_IO_VIEW: u64 = 1 << 47;
const XIO_NODE_SHIFT: u32 = 38;
const XIO_MEMORY_SIZE: u64 = 1 << 40;
const XIO_INT_MODIFY: u64 = IO_REGISTERS_NAMED + pi::reg::INT_PEND_MOD;
const XIO_IDENTITY: u64 = 0x0160_0000;

/// What a register write changed that the rest of the chip or the machine must
/// act on. A unit returns it from its write, as the AMD flash engine returns
/// its program effect, and the Hub acts on it once the unit's lock is
/// released: a unit holds no reference to a sibling and no closure. The new
/// values are in [`Routing`], not in the effect, so two writers that race
/// leave the routing at the last value written, not the last effect applied.
#[derive(Clone, Copy, Debug, Default)]
pub enum Effect {
    #[default]
    None,
    /// NI_STATUS_REV_ID changed the node number this Hub answers to.
    NodeNumber,
    /// A vector request from this Hub's port changed the neighbour's node
    /// number.
    PeerNodeNumber,
    /// PI_CALIAS_SIZE changed the cached-alias window.
    CachedAlias,
    /// NI_PORT_RESET asked to restart the whole machine.
    Restart,
    /// A control write launched a block-transfer engine.
    Bte(usize, io::Bte),
}

/// What another Hub needs of this node: its spaces and its interrupt
/// banks. It holds no reference back to the Hub, so a cable can share it.
struct NodeView {
    spaces: [OnceLock<Arc<dyn BusDevice>>; SPACES],
    pi: Arc<Pi>,
    /// The machine's node-zero routing, if it has one: a neighbour that
    /// renumbers this node over the cable has it re-read the number.
    zero: OnceLock<Arc<NodeZero>>,
}

/// The register window's sections. Kept apart from the Hub's address decode
/// so the node's I/O space can hold the registers without holding the Hub:
/// the Hub owns the spaces, and a space that held the Hub would be a cycle
/// that frees neither.
pub struct Regs {
    pi: Arc<Pi>,
    md: Md,
    io: Io,
    ni: Arc<Ni>,
    /// The node's I2C controller, mapped over the MD block.
    i2c: OnceLock<Arc<dyn BusDevice>>,
}

impl Regs {
    fn section_read(&self, off: u64, clear: bool) -> u64 {
        match off {
            ..pi::END => self.pi.read(off, clear),
            md::START..md::END => self.md.read(off),
            io::START..io::END => self.io.read(off),
            ni::START..ni::END => self.ni.read(off, clear),
            _ => {
                crate::dlog!(LogModule::Hub, "Hub: read of unimplemented register {off:#x}");
                0
            }
        }
    }

    fn section_peek(&self, off: u64) -> Option<u64> {
        match off {
            ..pi::END => self.pi.peek(off),
            md::START..md::END => self.md.peek(off),
            io::START..io::END => self.io.peek(off),
            ni::START..ni::END => self.ni.peek(off),
            _ => None,
        }
    }

    fn section_write(&self, off: u64, value: u64, size: u32) -> Effect {
        match off {
            ..pi::END => return self.pi.write_sized(off, value, size),
            md::START..md::END => self.md.write(off, value),
            io::START..io::END => return self.io.write(off, value),
            ni::START..ni::END => return self.ni.write(off, value),
            _ => crate::dlog!(LogModule::Hub, "Hub: write of {value:#x} to unimplemented register {off:#x}"),
        }
        Effect::None
    }

    /// The narrow device over an offset, for the I2C controller mapped over
    /// the MD block.
    fn overlay(&self, off: u64) -> Option<(&dyn BusDevice, u64)> {
        const I2C: std::ops::Range<u64> = md::reg::I2C_BASE..md::reg::I2C_BASE + 0x10;
        if I2C.contains(&off) {
            self.i2c.get().map(|d| (&**d, off - I2C.start))
        } else {
            None
        }
    }
}

pub struct Hub {
    regs: Arc<Regs>,
    /// What answers each space of this node, addressed from the node's base.
    view: Arc<NodeView>,
    /// The Hub across the CrayLink cable, as its spaces and interrupts. Written
    /// once when the cable is fitted, so an access to it takes no lock and
    /// touches no reference count.
    peer: OnceLock<Arc<NodeView>>,
    /// The node number and cached-alias size, as the units that own them
    /// publish them.
    routing: Arc<Routing>,
    /// Where NI_PORT_RESET bit 0 restarts the machine: the receiver goes to
    /// the machine's system controller, as the Indy IOC's does.
    events: OnceLock<Sender<MachineEvent>>,
    weak: Weak<Hub>,
}

impl std::ops::Deref for Hub {
    type Target = Regs;
    fn deref(&self) -> &Regs {
        &self.regs
    }
}

impl Hub {
    pub fn new(cfg: HubConfig, clock: Clock) -> Arc<Self> {
        let hub = Arc::new_cyclic(|weak: &Weak<Hub>| {
            let routing = Routing::new();
            let pi = Pi::new(cfg.cpus, cfg.cpu_base, clock.clone(), routing.clone());
            let view = Arc::new(NodeView { spaces: Default::default(), pi: pi.clone(), zero: OnceLock::new() });
            let ni = Arc::new(Ni::new(cfg.ni_revision, routing.clone()));
            let io = Io::new(cfg.widget_revision, cfg.manufacturer);
            let regs = Arc::new(Regs {
                pi,
                md: Md::new(clock, cfg.slot),
                io,
                ni,
                i2c: OnceLock::new(),
            });
            Self {
                regs,
                view,
                peer: OnceLock::new(),
                routing,
                events: OnceLock::new(),
                weak: weak.clone(),
            }
        });
        hub
    }

    fn pi(&self) -> &Arc<Pi> {
        &self.pi
    }
    fn md(&self) -> &Md {
        &self.md
    }
    fn io(&self) -> &Io {
        &self.io
    }
    fn ni(&self) -> &Arc<Ni> {
        &self.ni
    }

    /// Answer the widget's small window with `dev`.
    pub fn connect_widget(&self, widget: u8, dev: Arc<dyn BusDevice>) {
        self.io.connect_widget(widget, dev);
    }

    /// Where a peer's packets to this Hub's big windows land; see
    /// [`crate::dev::xbow::Xbow::packets`].
    pub fn connect_packets(&self, dev: Arc<dyn BusDevice>) {
        self.io.connect_packets(dev);
    }

    /// The machine's timer manager, for the real-time comparator's timer.
    pub fn set_timer_manager(&self, tm: Arc<TimerManager>) {
        self.pi.set_timer_manager(tm);
    }

    /// The one-wire net the memory section's master drives.
    pub fn set_onewire(&self, bus: w1::Bus) {
        self.md.set_onewire(bus);
    }

    /// Set the node's PI_CPU_ENABLE bit of `slice`. A Hub comes out of reset
    /// with both enables clear, and the stock PROM never sets its own, so the
    /// board does what the processor module's strap does on the machine.
    pub fn enable_cpu(&self, slice: usize) {
        let reg = pi::reg::CPU_ENABLE_A + 8 * slice as u64;
        assert!(reg <= pi::reg::CPU_ENABLE_B, "a node has two processor slices");
        self.write64(reg, 1);
    }

    /// The cached-alias window size PI_CALIAS_SIZE currently names.
    pub fn calias_bytes(&self) -> u64 {
        self.routing.calias()
    }

    /// The node number and cached-alias size as the units publish them.
    pub fn routing(&self) -> &Arc<Routing> {
        &self.routing
    }

    /// Where processor `slice`'s three interrupt lines go: its interrupt word
    /// (`MipsCpu::interrupts_ptr`), Cause bits 10 to 12. The Hub touches no
    /// other bit. The word must outlive the Hub, or be let go of with
    /// [`Self::clear_cpu_interrupts`] first; a machine clears it before it
    /// frees its processors.
    pub fn set_cpu_interrupts(&self, slice: usize, word: *const std::sync::atomic::AtomicU64) {
        self.pi.set_cpu_interrupts(slice, word);
    }

    /// Stop publishing to the processors' words.
    pub fn clear_cpu_interrupts(&self) {
        self.pi.clear_cpu_interrupts();
    }

    /// Whether slice `slice` has a word to publish to.
    #[cfg(test)]
    pub(crate) fn publishes_to_cpu(&self, slice: usize) -> bool {
        self.pi.publishes_to(slice)
    }
    /// Where NI_PORT_RESET bit 0 sends [`MachineEvent::HardReset`]. Wired
    /// once; a Hub with no sender logs the restart and carries on.
    pub fn set_event_sender(&self, tx: Sender<MachineEvent>) {
        let _ = self.events.set(tx);
    }

    /// The machine's node-zero routing for this node. It is told to re-read
    /// [`Routing`] whenever a store changes the node number or the cached
    /// alias, here or from the neighbour's vector PIO, and after reset.
    pub fn set_node_zero(&self, zero: Arc<NodeZero>) {
        let _ = self.view.zero.set(zero);
    }

    /// Act on what a register write changed, now that the unit that made it
    /// has let go of its lock.
    fn apply(&self, effect: Effect) {
        match effect {
            Effect::None => {}
            Effect::NodeNumber | Effect::CachedAlias => {
                if let Some(zero) = self.view.zero.get() {
                    zero.refresh();
                }
            }
            Effect::PeerNodeNumber => {
                if let Some(zero) = self.peer.get().and_then(|peer| peer.zero.get()) {
                    zero.refresh();
                }
            }
            Effect::Restart => match self.events.get() {
                Some(tx) => {
                    let _ = tx.send(MachineEvent::HardReset);
                }
                None => crate::dlog!(LogModule::Hub, "Hub NI: restart requested; no machine is listening"),
            },
            Effect::Bte(engine, launch) => self.io.run_bte(engine, launch, self),
        }
    }

    /// Register the sections' state locks with the `locks` monitor command.
    pub fn register_locks(self: &Arc<Self>) {
        crate::locks::register_mutex_fn("hub-pi", self, |h| h.pi.is_locked());
        crate::locks::register_mutex_fn("hub-md", self, |h| h.md.is_locked());
        crate::locks::register_mutex_fn("hub-io", self, |h| h.io.is_locked());
        crate::locks::register_mutex_fn("hub-ni", self, |h| h.ni.is_locked());
    }

    /// Where this node's four spaces answer, for the node-qualified claims of
    /// every Hub that reaches this one and for its own engines.
    pub fn set_space(&self, space: Space, dev: Arc<dyn BusDevice>) {
        self.view.spaces[space as usize].set(dev).unwrap_or_else(|_| panic!("{space:?} is wired twice"));
    }

    /// The node's I2C controller, answering at 0x22_0000 of the window.
    pub fn attach_i2c(&self, dev: Arc<dyn BusDevice>) {
        self.i2c.set(dev).unwrap_or_else(|_| panic!("the I2C controller is wired twice"));
    }

    /// Join the Hubs across a direct CrayLink cable, as between the two node
    /// boards of an Origin 2000 or Onyx2 deskside.
    pub fn cable(a: &Arc<Hub>, b: &Arc<Hub>) {
        assert!(a.peer.set(b.view.clone()).is_ok() && b.peer.set(a.view.clone()).is_ok(), "a Hub is cabled once");
        a.ni.connect_peer(Some(Arc::downgrade(&b.ni) as Weak<dyn NetPeer>));
        b.ni.connect_peer(Some(Arc::downgrade(&a.ni) as Weak<dyn NetPeer>));
    }

    /// Run `f` on the Hub the node number names. This Hub always answers for
    /// its own first: while two Hubs both hold the reset value zero, each
    /// keeps zero for itself. The peer is looked up only across the cable, and
    /// a number no reachable Hub holds fails. Both numbers are the network
    /// unit's own mirrors, so the decode takes no lock.
    fn with_node<R>(&self, number: u8, f: impl FnOnce(&NodeView) -> R) -> Option<R> {
        let number = u16::from(number);
        if number == self.routing.number() {
            Some(f(&self.view))
        } else if self.ni.neighbour_number() == Some(number) {
            self.peer.get().map(|peer| f(peer))
        } else {
            None
        }
    }

    /// A node-qualified address of `space`, claimed by the Hub holding its
    /// number. Addresses are the 40-bit physical address with the node in bits
    /// 39:32.
    pub fn claim(self: &Arc<Self>, space: Space) -> Arc<dyn BusDevice> {
        Arc::new(View { hub: self.clone(), routing: self.routing.clone(), space, local: Local::Never })
    }

    /// The processor's view of memory: a node-zero address below the cached
    /// alias size stays on this node and any other goes to the claim map.
    pub fn cpu_memory(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(View { hub: self.clone(), routing: self.routing.clone(), space: Space::Memory, local: Local::Calias })
    }

    /// The processor's view of the Hub-special space: always this node's own.
    pub fn cpu_special(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(View { hub: self.clone(), routing: self.routing.clone(), space: Space::HubSpecial, local: Local::Always })
    }

    /// The processor's view of the I/O space: this node's registers at the
    /// reader's own alias (node field zero), whatever number the node holds,
    /// and any other address claimed by the node number it carries.
    pub fn cpu_io(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(View { hub: self.clone(), routing: self.routing.clone(), space: Space::Io, local: Local::IoAlias })
    }

    /// This node's I/O space, addressed from the node's base: the crossbar
    /// widgets' small windows, the Hub's own registers at both offsets, and
    /// the seven big windows.
    pub fn io_space(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(IoSpace { hub: self.weak.clone(), regs: self.regs.clone(), small: self.io.small_windows(), big: self.io.big_windows() })
    }

    /// This Hub as the XIO host a Bridge sends to, addressed as `widget`: DMA
    /// goes to node memory, an interrupt packet to any address of the widget
    /// sets or clears one of its 128 interrupt numbers. Behind a Crossbow,
    /// pass it to [`crate::dev::xbow::Xbow::upstream`] so the port's link
    /// gates it.
    pub fn host_port(&self, widget: u8) -> HostPort {
        HostPort::new(
            XtalkHost { widget, int_packet: 0, int_packet_mask: 0, vectors: 128 },
            self.inbound(),
            self.interrupt_target(),
        )
    }

    /// The Hub as a bus master's target for interrupts: a line is the
    /// interrupt number 0 to 127, set and cleared through interrupt-modify. A
    /// Bridge that reaches the Hub by an XIO write is equivalent.
    pub fn interrupt_target(&self) -> Arc<dyn InterruptController> {
        Arc::new(InterruptTarget(self.pi.clone()))
    }

    /// The XIO port's packet face: what a widget's bus-master requests reach,
    /// a memory address below 2^40, an interrupt-modify write or the
    /// identity probe.
    pub fn inbound(&self) -> Arc<dyn BusDevice> {
        Arc::new(Inbound(self.weak.clone()))
    }

    /// A peer's programmed I/O to the Hub's own widget number.
    pub fn widget_face(&self) -> Arc<dyn BusDevice> {
        Arc::new(WidgetFace(self.weak.clone()))
    }

    /// Wrap the downstream I/O space a processor reaches through this Hub, so
    /// a failed read records in that slice's error status.
    pub fn processor_io(self: &Arc<Self>, downstream: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
        Arc::new(ProcessorIo { hub: self.clone(), downstream })
    }

    /// Record a failed processor read in slice `slice`'s error status.
    pub fn read_error(&self, slice: usize, physical: u64) {
        self.pi.read_error(slice, physical);
    }


}

/// How a view treats a node-zero address.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Local {
    /// Every address is claimed by its node number.
    Never,
    /// Node-zero addresses below the cached-alias size stay on this node.
    Calias,
    /// Node-zero addresses are always this node's own.
    Always,
    /// Node-zero addresses in the registers' own alias are this node's
    /// whatever number the node holds: the alias is the one I/O region the
    /// reader's Hub answers without asking which node holds zero.
    IoAlias,
}

/// A node-qualified address space as a processor, an engine or a widget sees
/// it: the node number picks the Hub, and the rest the address within it.
struct View {
    hub: Arc<Hub>,
    /// The hub's published numbers, held here so a decode loads them through
    /// one pointer.
    routing: Arc<Routing>,
    space: Space,
    local: Local,
}
impl View {
    fn run<R>(&self, a: u64, f: impl FnOnce(&dyn BusDevice, u64) -> R) -> Option<R> {
        let offset = a & (NODE_SPAN - 1);
        let number = ((a >> NODE_SHIFT) & NODE_MASK) as u8;
        let stays = match self.local {
            Local::Never => false,
            Local::Calias => number == 0 && a < self.routing.calias(),
            Local::Always => number == 0,
            Local::IoAlias => number == 0 && (IO_REGISTERS_LOCAL..IO_REGISTERS_NAMED).contains(&offset),
        };
        if stays {
            return self.hub.view.spaces[self.space as usize].get().map(|d| f(&**d, offset));
        }
        self.hub.with_node(number, |node| node.spaces[self.space as usize].get().map(|d| f(&**d, offset))).flatten()
    }
    fn unclaimed(&self, a: u64, write: bool) {
        crate::dlog!(LogModule::Hub, "Hub: {} of {:?} {a:#x}: no Hub holds node {} (or its space is unwired)",
            if write { "write" } else { "read" }, self.space, (a >> NODE_SHIFT) & NODE_MASK);
    }
}
macro_rules! view_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            self.run(a, |d, at| d.$r(at)).unwrap_or_else(|| {
                self.unclaimed(a, false);
                $out::err()
            })
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.run(a, |d, at| d.$w(at, v)).unwrap_or_else(|| {
                self.unclaimed(a, true);
                BUS_ERR
            })
        }
    };
}
impl BusDevice for View {
    // A store-conditional to another node's memory has to be as atomic as
    // one to the processor's own.
    fn cas32(&self, a: u64, old: u32, new: u32) -> Option<bool> {
        self.run(a, |d, at| d.cas32(at, old, new)).flatten()
    }
    fn cas64(&self, a: u64, old: u64, new: u64) -> Option<bool> {
        self.run(a, |d, at| d.cas64(at, old, new)).flatten()
    }
    view_access!(read8, write8, u8, BusRead8);
    view_access!(read16, write16, u16, BusRead16);
    view_access!(read32, write32, u32, BusRead32);
    view_access!(read64, write64, u64, BusRead64);
}

/// The node's I/O space.
struct IoSpace {
    /// The Hub, for the effect of a register write. Weak: the Hub owns its
    /// spaces.
    hub: Weak<Hub>,
    regs: Arc<Regs>,
    small: Arc<dyn BusDevice>,
    big: Arc<dyn BusDevice>,
}
impl IoSpace {
    /// The register window offset a Hub-register address names.
    fn registers(a: u64) -> Option<u64> {
        let off = a.checked_sub(IO_REGISTERS_LOCAL)? & !(1 << 23);
        (a - IO_REGISTERS_LOCAL < 2 * WINDOW_SIZE).then_some(off & (WINDOW_SIZE - 1))
    }
}
macro_rules! io_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            if let Some(off) = Self::registers(a) {
                self.regs.$r(off)
            } else if a >= IO_BIG_WINDOWS {
                self.big.$r(a - IO_BIG_WINDOWS)
            } else {
                self.small.$r(a)
            }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            if let Some(off) = Self::registers(a) {
                // Through the Hub, which acts on what the write changed.
                match self.hub.upgrade() {
                    Some(hub) => hub.$w(off, v),
                    None => BUS_ERR,
                }
            } else if a >= IO_BIG_WINDOWS {
                self.big.$w(a - IO_BIG_WINDOWS, v)
            } else {
                self.small.$w(a, v)
            }
        }
    };
}
impl BusDevice for IoSpace {
    io_access!(read8, write8, u8, BusRead8);
    io_access!(read16, write16, u16, BusRead16);
    io_access!(read32, write32, u32, BusRead32);
    io_access!(read64, write64, u64, BusRead64);
}

/// Every I/O-space request a processor makes passes through its Hub. A read
/// that fails there, or downstream, is recorded in the issuing slice's error
/// status 0; a debugger's access or a device's DMA does not come this way.
struct ProcessorIo {
    hub: Arc<Hub>,
    downstream: Arc<dyn BusDevice>,
}
impl ProcessorIo {
    /// Record a failed read for the issuing slice. A debugger's access, or
    /// any thread that is not one of the node's processors, records nothing.
    fn failed(&self, a: u64) {
        if let Some(slice) = self.hub.pi.reader_slice() {
            self.hub.read_error(slice, a);
        }
    }
}
macro_rules! processor_read {
    ($r:ident, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            let r = self.downstream.$r(a);
            // A busy answer is a retry, not a failed read.
            if r.status != BUS_OK && r.status != BUS_BUSY {
                self.failed(a);
            }
            r
        }
    };
}
impl BusDevice for ProcessorIo {
    processor_read!(read8, BusRead8);
    processor_read!(read16, BusRead16);
    processor_read!(read32, BusRead32);
    fn read64(&self, a: u64) -> BusRead64 {
        // The Hub fails one address itself: a doubleword read at node offset
        // 0x0240_0208 (IIO_IIBUSERR, "reads here cause a bus error"), whatever
        // node it names. Nothing goes out on the XIO link.
        let r = if a & (NODE_SPAN - 1) == IO_BUS_ERROR_PROBE { BusRead64::err() } else { self.downstream.read64(a) };
        if r.status != BUS_OK && r.status != BUS_BUSY {
            self.failed(a);
        }
        r
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        self.downstream.write8(a, v)
    }
    fn write16(&self, a: u64, v: u16) -> u32 {
        self.downstream.write16(a, v)
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        self.downstream.write32(a, v)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.downstream.write64(a, v)
    }
}

/// A line is an interrupt number; a rising level sets its pending bit and a
/// falling one clears it.
struct InterruptTarget(Arc<Pi>);
impl InterruptController for InterruptTarget {
    fn set_line(&self, line: u32, level: bool) {
        if line < 128 {
            self.0.int_modify(u64::from(line) | if level { pi::MODIFY_SET } else { 0 });
        }
    }
}

/// The XIO packet face: a 48-bit address, bit 47 selecting the I/O view.
struct Inbound(Weak<Hub>);
impl Inbound {
    fn memory(hub: &Hub, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        if a >= XIO_MEMORY_SIZE || size as u64 > XIO_MEMORY_SIZE - a {
            return BusRead64::err();
        }
        let mut bytes = [0u8; 8];
        match value {
            Some(v) => {
                bytes[..size].copy_from_slice(&v.to_be_bytes()[8 - size..]);
                BusRead64 { status: if hub.write(a, &bytes[..size]) { BUS_OK } else { BUS_ERR }, data: 0 }
            }
            None => {
                if !hub.read(a, &mut bytes[..size]) {
                    return BusRead64::err();
                }
                let mut word = [0u8; 8];
                word[8 - size..].copy_from_slice(&bytes[..size]);
                BusRead64::ok(u64::from_be_bytes(word))
            }
        }
    }
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let Some(hub) = self.0.upgrade() else { return BusRead64::err() };
        if a & XIO_IO_VIEW == 0 {
            return Self::memory(&hub, a, size, value);
        }
        let number = ((a >> XIO_NODE_SHIFT) & NODE_MASK) as u16;
        let offset = a & ((1 << XIO_NODE_SHIFT) - 1);
        match value {
            // A peer's identity probe: node field zero, the status register.
            None if size == 8 && a == (XIO_IO_VIEW | XIO_IDENTITY) => BusRead64::ok(hub.ni.read(ni::reg::STATUS, false)),
            // A device interrupt: a doubleword write to a node's
            // interrupt-modify register lands on that node's Hub.
            Some(v) if size == 8 && offset == XIO_INT_MODIFY => {
                let reached = NodeFabric::interrupt(&*hub, number, v);
                BusRead64 { status: if reached { BUS_OK } else { BUS_ERR }, data: 0 }
            }
            _ => {
                crate::dlog!(LogModule::Hub, "Hub: unsupported inbound XIO {} at {a:#x} size {size}",
                    if value.is_some() { "write" } else { "read" });
                BusRead64::err()
            }
        }
    }
}
macro_rules! inbound_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            let r = self.access(a, std::mem::size_of::<$t>(), None);
            $out { status: r.status, data: r.data as $t }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.access(a, std::mem::size_of::<$t>(), Some(u64::from(v))).status
        }
    };
}
impl BusDevice for Inbound {
    inbound_access!(read8, write8, u8, BusRead8);
    inbound_access!(read16, write16, u16, BusRead16);
    inbound_access!(read32, write32, u32, BusRead32);
    inbound_access!(read64, write64, u64, BusRead64);
}

/// A peer's programmed I/O to this Hub's widget number answers only the
/// 4-byte read of the identification word at offset 4.
struct WidgetFace(Weak<Hub>);
impl BusDevice for WidgetFace {
    fn read32(&self, a: u64) -> BusRead32 {
        match self.0.upgrade().and_then(|hub| hub.io.widget_face_read32(a)) {
            Some(v) => BusRead32::ok(v),
            None => {
                crate::dlog!(LogModule::Hub, "Hub: unsupported peer read at {a:#x} on the Hub's widget face");
                BusRead32::err()
            }
        }
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        crate::dlog!(LogModule::Hub, "Hub: unsupported peer write of {v:#x} at {a:#x} on the Hub's widget face");
        BUS_ERR
    }
}

impl NodeFabric for Hub {
    fn read(&self, address: u64, buf: &mut [u8]) -> bool {
        let number = ((address >> NODE_SHIFT) & NODE_MASK) as u8;
        let offset = address & (NODE_SPAN - 1);
        self.with_node(number, |node| {
            node.spaces[Space::Memory as usize].get().is_some_and(|mem| read_bytes(&**mem, offset, buf) == BUS_OK)
        })
        .unwrap_or(false)
    }
    fn write(&self, address: u64, data: &[u8]) -> bool {
        let number = ((address >> NODE_SHIFT) & NODE_MASK) as u8;
        let offset = address & (NODE_SPAN - 1);
        self.with_node(number, |node| {
            node.spaces[Space::Memory as usize].get().is_some_and(|mem| write_bytes(&**mem, offset, data) == BUS_OK)
        })
        .unwrap_or(false)
    }
    fn interrupt(&self, number: u16, value: u64) -> bool {
        number < 256
            && self
                .with_node(number as u8, |node| node.pi.int_modify(value))
                .is_some()
    }
}

/// The register window. The I2C controller takes every
/// width, one register access each: the stock PROM polls the controller's
/// status with 8-byte loads. Elsewhere it takes 4- and 8-byte accesses. A 32-bit
/// access names half of the aligned doubleword, offset 0 carrying bits 63:32
/// as on a big-endian bus. A 32-bit write replaces its half and takes the
/// ordinary write path; for the interrupt-modify register, the port error
/// register and any register that stores nothing the other half is zero. A
/// 32-bit read of a read-to-clear register returns its half without clearing
/// it: what hardware does for the narrower access is unknown, and this keeps
/// a latch until software has read all of it.
///
/// A write to a register returns what it changed ([`Effect`]) beside the bus
/// status, for the Hub to act on once the unit's lock is released; the
/// narrow writes reach only the I2C controller and change nothing else.
impl Regs {
    fn store64(&self, a: u64, v: u64) -> (u32, Effect) {
        if a & 7 != 0 || a >= WINDOW_SIZE {
            return (BUS_ERR, Effect::None);
        }
        if let Some((dev, at)) = self.overlay(a) {
            return (dev.write64(at, v), Effect::None);
        }
        (BUS_OK, self.section_write(a, v, 8))
    }

    fn store32(&self, a: u64, v: u32) -> (u32, Effect) {
        if a & 3 != 0 || a >= WINDOW_SIZE {
            return (BUS_ERR, Effect::None);
        }
        if let Some((dev, at)) = self.overlay(a & !7) {
            return (dev.write32(at + (a & 7), v), Effect::None);
        }
        let aligned = a & !7;
        let mut merged = if matches!(aligned, pi::reg::INT_PEND_MOD | ni::reg::PORT_ERROR) {
            0
        } else {
            self.section_peek(aligned).unwrap_or(0)
        };
        merged = if a & 4 != 0 {
            (merged & !0xffff_ffff) | u64::from(v)
        } else {
            (merged & 0xffff_ffff) | (u64::from(v) << 32)
        };
        (BUS_OK, self.section_write(aligned, merged, 4))
    }
}

impl BusDevice for Regs {
    fn read64(&self, a: u64) -> BusRead64 {
        if a & 7 != 0 || a >= WINDOW_SIZE {
            return BusRead64::err();
        }
        if let Some((dev, at)) = self.overlay(a) {
            return dev.read64(at);
        }
        BusRead64::ok(self.section_read(a, true))
    }
    fn read32(&self, a: u64) -> BusRead32 {
        if a & 3 != 0 || a >= WINDOW_SIZE {
            return BusRead32::err();
        }
        if let Some((dev, at)) = self.overlay(a & !7) {
            return dev.read32(at + (a & 7));
        }
        let value = self.section_read(a & !7, false);
        BusRead32::ok(if a & 4 != 0 { value as u32 } else { (value >> 32) as u32 })
    }
    fn read8(&self, a: u64) -> BusRead8 {
        match self.overlay(a & !7) {
            Some((dev, at)) => dev.read8(at + (a & 7)),
            None => BusRead8::err(),
        }
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        match self.overlay(a & !7) {
            Some((dev, at)) => dev.write8(at + (a & 7), v),
            None => BUS_ERR,
        }
    }
    fn read16(&self, a: u64) -> BusRead16 {
        match self.overlay(a & !7) {
            Some((dev, at)) => dev.read16(at + (a & 7)),
            None => BusRead16::err(),
        }
    }
    fn write16(&self, a: u64, v: u16) -> u32 {
        match self.overlay(a & !7) {
            Some((dev, at)) => dev.write16(at + (a & 7), v),
            None => BUS_ERR,
        }
    }
}

macro_rules! delegate {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            self.regs.$r(a)
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.regs.$w(a, v)
        }
    };
}
macro_rules! delegate_store {
    ($r:ident, $w:ident, $store:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            self.regs.$r(a)
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            let (status, effect) = self.regs.$store(a, v);
            self.apply(effect);
            status
        }
    };
}
/// The Hub as the register window a machine maps.
impl BusDevice for Hub {
    delegate!(read8, write8, u8, BusRead8);
    delegate!(read16, write16, u16, BusRead16);
    delegate_store!(read32, write32, store32, u32, BusRead32);
    delegate_store!(read64, write64, store64, u64, BusRead64);
}

impl Resettable for Hub {
    /// Every register returns to its reset value, the node holds number zero
    /// again and the machine is told so. The error stack addresses, the memory
    /// behind the node and the one-wire part's contents survive.
    fn power_on(&self) {
        self.pi.power_on();
        self.md.power_on();
        self.io.power_on();
        self.ni.power_on();
        self.apply(Effect::NodeNumber);
    }
}

impl Device for Hub {
    fn step(&self, _cycles: u64) {
        self.pi.poll();
    }
    fn stop(&self) {
        self.pi.stop();
    }
    fn start(&self) {
        self.pi.start();
    }
    fn is_running(&self) -> bool {
        self.pi.is_running()
    }
    fn get_clock(&self) -> u64 {
        self.pi.rt_count()
    }
}

#[cfg(test)]
mod tests;
