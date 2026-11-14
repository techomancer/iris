//! The Hub, the node controller of the SN0 machines (Origin 200, Origin 2000,
//! Onyx2): one per node, joining the node's processors, memory and directory,
//! one XIO port and one CrayLink port. The register window is split by
//! section into the processor interface (`pi`), memory and directory (`md`),
//! I/O (`io`) and network (`ni`) units, private to this module; the module is
//! the window itself and what the units share. Machines reach the Hub only
//! through its methods.
//!
//! The Hub is the node's memory controller: a store to the node number or the
//! cached-alias size places the node, and the Hub remaps both nodes' decode
//! tables ([`crate::sn_nodes::NodeDecode::remap`]).
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

#![warn(dead_code)]

mod io;
mod md;
mod ni;
mod pi;
// Generated: rows the model does not use yet stay in the table.
#[allow(dead_code)]
pub mod regs;

pub use self::md::{bank_mappings, fitted_mappings, SlotSignals, BANKS, BANK_STRIDE, SLOT_STRIDE};

/// Length of the register window: 8 MiB.
pub const WINDOW_SIZE: u64 = 8 << 20;

/// Pending interrupts are two banks of 64 bits.
pub const PENDING_BANKS: usize = 2;

use crate::dev::heart::Clock;
use crate::dev::widths::{self, access_widths};
use self::io::Io;
use self::md::Md;
use self::ni::Ni;
use self::pi::Pi;
use crate::devlog::LogModule;
use crate::dev::w1;
use crate::irq::InterruptController;
use crate::hptimer::TimerManager;
use crate::sn_nodes::{calias_bytes, node_id, Node, NodeDecode, NodeId, UncachedSpace, XIO_IO_VIEW};
use crate::xtalk::{HostPort, XtalkHost};
use crate::traits::{
    BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Device, Resettable, BUS_BUSY, BUS_ERR, BUS_OK,
};
use crate::traits::MachineEvent;
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock, Weak};

/// What differs between Hubs: the processors the node populates, where its
/// slice A sits in the machine's numbering, the memory fitted, the revision
/// fields and the nodes of the machine. The widget identification and network
/// status carry
/// separate revisions, and the model sets them separately.
#[derive(Clone, Copy, Debug)]
pub struct HubConfig {
    /// Populated processor sockets: 1 or 2.
    pub cpus: u8,
    /// Global index of this node's slice A, which PI_CPU_NUM subtracts.
    pub first_cpu: usize,
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
    /// The memory banks in MiB, bank 0 first; zero for an empty bank.
    pub banks_mb: [u32; BANKS],
    /// The nodes of the machine, one or two, for the node's ppmem window.
    pub nodes: usize,
}
impl Default for HubConfig {
    fn default() -> Self {
        Self {
            cpus: 1,
            first_cpu: 0,
            ni_revision: 3,
            widget_revision: 1,
            manufacturer: 0x36,
            slot: SlotSignals::default(),
            banks_mb: [0; BANKS],
            nodes: 1,
        }
    }
}

/// Node numbers are bits 39:32 of a physical address in M mode, so each node
/// owns a 4 GiB window.
pub const NODE_SHIFT: u32 = crate::sn_nodes::SN0_NODE_SHIFT;
pub const NODE_SPAN: u64 = 1 << NODE_SHIFT;

/// Offsets in the I/O space of a node, from the base of its address range.
pub use crate::sn_nodes::{IO_REGISTERS_LOCAL, IO_REGISTERS_NAMED};
pub use crate::sn_nodes::IO_BIG_WINDOWS;
/// A doubleword read here fails at the issuing Hub whatever node it names.
pub const IO_BUS_ERROR_PROBE: u64 = 0x0240_0208;

/// Where in the XIO address a packet is taken as an interrupt-modify write:
/// bit 47 selects the I/O view, the node number is bits 45:38.
const XIO_NODE_SHIFT: u32 = 38;
/// The node an XIO I/O-view address names.
fn xio_node(a: u64) -> NodeId {
    node_id(a >> XIO_NODE_SHIFT, NODE_SHIFT)
}
const XIO_MEMORY_SIZE: u64 = 1 << 40;
const XIO_INT_MODIFY: u64 = IO_REGISTERS_NAMED + regs::PI_INT_PEND_MOD::OFFSET;
const XIO_IDENTITY: u64 = regs::PI_LOCAL_WINDOW_BASE + regs::NI_STATUS_REV_ID::OFFSET;

/// What a register write changed that the rest of the chip or the machine must
/// act on. A unit returns it from its write, as the AMD flash engine returns
/// its program effect, and the Hub acts on it once the unit's lock is
/// released: a unit holds no reference to a sibling and no closure.
#[derive(Clone, Copy, Debug, Default)]
pub enum Effect {
    #[default]
    None,
    /// NI_STATUS_REV_ID changed the node number this Hub answers to. The
    /// register holds nine bits; the number is the eight an address carries.
    NodeNumber(NodeId),
    /// PI_CALIAS_SIZE changed the cached-alias window, now this many bytes.
    CachedAlias(u64),
    /// NI_PORT_RESET asked to restart the whole machine.
    Restart,
    /// A control write launched a block-transfer engine.
    Bte(usize, io::Bte),
    /// A vector request from this Hub's port: perform it at the far end of
    /// the port and latch the reply.
    Vector(ni::VectorLaunch),
    /// NI_PORT_RESET reset the link: both ends settle the width again.
    LinkReset,
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
            _ => self.ni.read(off, clear),
        }
    }

    fn section_peek(&self, off: u64) -> Option<u64> {
        match off {
            ..pi::END => self.pi.peek(off),
            md::START..md::END => self.md.peek(off),
            io::START..io::END => self.io.peek(off),
            _ => self.ni.peek(off),
        }
    }

    fn section_write(&self, off: u64, value: u64, size: u32) -> Effect {
        match off {
            ..pi::END => return self.pi.write_sized(off, value, size),
            md::START..md::END => self.md.write(off, value),
            io::START..io::END => return self.io.write(off, value),
            _ => return self.ni.write(off, value),
        }
        Effect::None
    }

    /// The narrow device over an offset, for the I2C controller mapped over
    /// the MD block.
    fn overlay(&self, off: u64) -> Option<(&dyn BusDevice, u64)> {
        const I2C: std::ops::Range<u64> = regs::MD_UREG0::at(0)..regs::MD_UREG0::at(2);
        if I2C.contains(&off) {
            self.i2c.get().map(|d| (&**d, off - I2C.start))
        } else {
            None
        }
    }
}

pub struct Hub {
    regs: Arc<Regs>,
    /// This node and the Hub at the far end of its NI's port, by the numbers they
    /// publish, and the node's memory, which the Hub remaps.
    pub(crate) decode: Arc<NodeDecode<Hub>>,
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
    /// A Hub with the memory `cfg` fits and the node's boot `flash`, which the Hub
    /// maps at the reset address wherever no bank covers it.
    pub fn new(cfg: HubConfig, clock: Clock, flash: Option<Arc<dyn BusDevice>>) -> Result<Arc<Self>, String> {
        let memory = md::memory(cfg.banks_mb, cfg.nodes)?;
        let boot = flash.map(|f| (crate::sn_nodes::RESET_WINDOW, crate::dev::node_flash::SIZE as u64, f));
        let runs = md::fitted_mappings(cfg.banks_mb);
        let hub = Arc::new_cyclic(|weak: &Weak<Hub>| {
            let node = Node::new(weak.clone(), memory, runs, boot, false);
            let regs = Arc::new(Regs {
                pi: Pi::new(cfg.cpus, cfg.first_cpu, clock.clone()),
                md: Md::new(clock, cfg.slot, cfg.banks_mb),
                io: Io::new(cfg.widget_revision, cfg.manufacturer),
                ni: Arc::new(Ni::new(cfg.ni_revision)),
                i2c: OnceLock::new(),
            });
            Self {
                regs,
                // The Hub-special space at node field zero is the reader's
                // whole node span.
                decode: NodeDecode::new(NODE_SHIFT, NODE_SPAN, node, LogModule::Hub),
                events: OnceLock::new(),
                weak: weak.clone(),
            }
        });
        hub.decode.set_alias(0, calias_bytes(pi::CALIAS_RESET));
        Ok(hub)
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
        let reg = regs::PI_CPU_ENABLE_A::OFFSET + (regs::PI_CPU_ENABLE_B::OFFSET - regs::PI_CPU_ENABLE_A::OFFSET) * slice as u64;
        assert!(reg <= regs::PI_CPU_ENABLE_B::OFFSET, "a node has two processor slices");
        self.write64(reg, 1);
    }

    /// Where processor `slice`'s four interrupt lines go: its interrupt word
    /// (`MipsCpu::interrupts_ptr`), Cause bits 10 to 13. The Hub touches no
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

    /// Act on what a register write changed, now that the unit that made it
    /// has let go of its lock.
    fn apply(&self, effect: Effect) {
        match effect {
            Effect::None => {}
            Effect::NodeNumber(n) => self.decode.remap(&|| self.decode.set_number(n)),
            Effect::CachedAlias(bytes) => self.decode.remap(&|| self.decode.set_alias(0, bytes)),
            Effect::Restart => match self.events.get() {
                Some(tx) => {
                    let _ = tx.send(MachineEvent::HardReset);
                }
                None => crate::dlog!(LogModule::Hub, "Hub NI: restart requested; no machine is listening"),
            },
            Effect::Bte(engine, launch) => {
                if let Some(io::Completion { node, value }) = self.io.run_bte(engine, launch, &*self.decode) {
                    if !self.interrupt(node, value) {
                        crate::dlog!(LogModule::Hub, "Hub IO: BTE{engine} completion interrupt aimed at node {node}, which no reachable Hub holds");
                    }
                }
            }
            Effect::Vector(launch) => {
                // A neighbour with no Router behind it answers a routed access with an
                // address error, not the timeout an empty port gives.
                let reply = self.decode.peer().map(|peer| match launch.route {
                    0 => peer.vector_target(&launch.request()),
                    _ => ni::VectorReply { data: None, node: peer.ni.number() },
                });
                self.ni.vector_done(&launch, reply);
            }
            Effect::LinkReset => {
                let far = self.decode.peer();
                self.ni.link_reset(far.as_ref().map(|far| far.ni.forces_eight_bit()));
                if let Some(far) = far {
                    far.ni.link_reset_from_far(self.ni.forces_eight_bit());
                }
            }
        }
    }

    /// The target's half of a vector request from the neighbour's port: the answer,
    /// or `None` for an address error, and this node's number.
    fn vector_target(&self, request: &ni::VectorRequest) -> ni::VectorReply {
        let (data, effect) = self.ni.vector_access(request);
        self.apply(effect);
        ni::VectorReply { data, node: self.ni.number() }
    }

    /// Write `value` to the interrupt-modify register of the Hub holding
    /// `number`; whether one does.
    fn interrupt(&self, number: NodeId, value: u64) -> bool {
        self.decode.on_holder(self, number, |hub| hub.pi.int_modify(value)).is_some()
    }

    /// Register the sections' state locks with the `locks` monitor command.
    pub fn register_locks(self: &Arc<Self>) {
        crate::locks::register_mutex_fn("hub-pi", self, |h| h.pi.is_locked());
        crate::locks::register_mutex_fn("hub-md", self, |h| h.md.is_locked());
        crate::locks::register_mutex_fn("hub-io", self, |h| h.io.is_locked());
        crate::locks::register_mutex_fn("hub-ni", self, |h| h.ni.is_locked());
    }

    /// The node's I2C controller, answering at 0x22_0000 of the window.
    pub fn attach_i2c(&self, dev: Arc<dyn BusDevice>) {
        self.i2c.set(dev).unwrap_or_else(|_| panic!("the I2C controller is wired twice"));
    }

    /// Connect this Hub's NI port to the Hub at the far end of a direct CrayLink,
    /// as between the two node boards of an Origin 2000 or Onyx2 deskside; the
    /// machine connects each end to the other.
    pub fn connect_peer(&self, far: &Hub) {
        self.decode.connect_peer(far.decode.node());
        self.ni.link_up(far.ni.forces_eight_bit());
    }

    /// The back door onto the node's ECC bytes.
    pub fn ecc_bus(&self) -> Arc<dyn BusDevice> {
        self.md.ecc_bus()
    }

    /// The back door onto the node's directory entries.
    pub fn directory_bus(&self) -> Arc<dyn BusDevice> {
        self.md.directory_bus()
    }

    /// The processor's view of the Hub-special or I/O space: node field zero is this
    /// node's own whatever number it holds; any other address goes to the Hub
    /// holding that number.
    pub fn cpu_view(&self, space: UncachedSpace) -> Arc<dyn BusDevice> {
        // The view holds this node's I/O space with the Hub: the Hub never
        // holds the view.
        let own = (space == UncachedSpace::Io).then(|| Arc::new(self.io_space_held(self.weak.upgrade().expect("a live Hub"))) as Arc<dyn BusDevice>);
        self.decode.cpu_view(space, own)
    }

    /// This node's I/O space, addressed from the node's base: the crossbar
    /// widgets' small windows, the Hub's own registers at both offsets, and
    /// the seven big windows.
    pub fn io_space(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(self.io_space_held(self.weak.clone()))
    }

    fn io_space_held<H>(&self, hub: H) -> IoSpace<H> {
        IoSpace { hub, regs: self.regs.clone(), small: self.io.small_windows(), big: self.io.big_windows() }
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
        Arc::new(Inbound(self.weak.clone()))
    }

    /// The XIO port's inbound packets: what a widget's bus-master requests reach,
    /// a memory address below 2^40, an interrupt-modify write or the
    /// identity probe.
    pub fn inbound(&self) -> Arc<dyn BusDevice> {
        Arc::new(Inbound(self.weak.clone()))
    }

    /// The Hub's own widget port: a peer's programmed I/O to the Hub's widget
    /// number.
    pub fn widget_port(&self) -> Arc<dyn BusDevice> {
        Arc::new(WidgetPort(self.weak.clone()))
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

/// The node's I/O space, held with the Hub as `Weak` by the space the Hub owns
/// and as `Arc` by a processor's view of its own node, which skips an upgrade.
/// A register write stores through `regs` and needs the Hub only when it
/// changed something the Hub acts on, so a peer's plain store takes no upgrade.
struct IoSpace<H> {
    /// For the effect of a register write.
    hub: H,
    regs: Arc<Regs>,
    small: Arc<dyn BusDevice>,
    big: Arc<dyn BusDevice>,
}
impl<H> IoSpace<H> {
    /// The register window offset a Hub-register address names.
    fn registers(a: u64) -> Option<u64> {
        let off = a.checked_sub(IO_REGISTERS_LOCAL)? & !(IO_REGISTERS_NAMED - IO_REGISTERS_LOCAL);
        (a - IO_REGISTERS_LOCAL < 2 * WINDOW_SIZE).then_some(off & (WINDOW_SIZE - 1))
    }

    /// A register write's effect goes to `act`, which reports whether a Hub
    /// was there to act on it.
    #[inline(always)]
    fn route(&self, a: u64, size: usize, value: Option<u64>, act: impl FnOnce(Effect) -> bool) -> BusRead64 {
        match (Self::registers(a), value) {
            (Some(off), Some(v)) => {
                let (status, effect) = self.regs.store(off, size, v);
                if matches!(effect, Effect::None) || act(effect) { BusRead64 { status, data: 0 } } else { BusRead64::err() }
            }
            (Some(off), None) => widths::access(&*self.regs, off, size, None),
            (None, _) if a >= IO_BIG_WINDOWS => widths::access(&*self.big, a - IO_BIG_WINDOWS, size, value),
            (None, _) => widths::access(&*self.small, a, size, value),
        }
    }
}
impl IoSpace<Arc<Hub>> {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        self.route(a, size, value, |effect| {
            self.hub.apply(effect);
            true
        })
    }
}
impl IoSpace<Weak<Hub>> {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        self.route(a, size, value, |effect| self.hub.upgrade().map(|hub| hub.apply(effect)).is_some())
    }
}
impl BusDevice for IoSpace<Arc<Hub>> {
    access_widths!();
}
impl BusDevice for IoSpace<Weak<Hub>> {
    access_widths!();
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

    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        // The Hub fails one address itself: a doubleword read at node offset
        // 0x0240_0208 (IIO_IIBUSERR, "reads here cause a bus error"), whatever
        // node it names. Nothing goes out on the XIO link.
        let probe = value.is_none() && size == 8 && a & (NODE_SPAN - 1) == IO_BUS_ERROR_PROBE;
        let r = if probe { BusRead64::err() } else { widths::access(&*self.downstream, a, size, value) };
        // A busy answer is a retry, not a failed read.
        if value.is_none() && r.status != BUS_OK && r.status != BUS_BUSY {
            self.failed(a);
        }
        r
    }
}
impl BusDevice for ProcessorIo {
    access_widths!();
}

/// The XIO port's inbound packets: a 48-bit address, bit 47 selecting the I/O view. As an
/// interrupt target a rising level sets a pending bit and a falling one clears it.
struct Inbound(Weak<Hub>);
impl Inbound {
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let Some(hub) = self.0.upgrade() else { return BusRead64::err() };
        if a & XIO_IO_VIEW == 0 {
            if a >= XIO_MEMORY_SIZE || size as u64 > XIO_MEMORY_SIZE - a {
                return BusRead64::err();
            }
            return widths::access(&*hub.decode, a, size, value);
        }
        let offset = a & ((1 << XIO_NODE_SHIFT) - 1);
        match value {
            // A peer's identity probe: node field zero, the status register.
            None if size == 8 && a == (XIO_IO_VIEW | XIO_IDENTITY) => BusRead64::ok(hub.ni.read(regs::NI_STATUS_REV_ID::OFFSET, false)),
            // A device interrupt: a doubleword write to a node's
            // interrupt-modify register lands on that node's Hub.
            Some(v) if size == 8 && offset == XIO_INT_MODIFY => {
                let reached = hub.interrupt(xio_node(a), v);
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
impl Inbound {
    /// A block of doublewords in node memory, each page from the node its number
    /// names; refused beyond the 40-bit space or in the I/O view.
    fn block<R>(&self, a: u64, words: usize, f: impl FnOnce(&dyn BusDevice) -> R) -> Option<R> {
        let hub = self.0.upgrade()?;
        let bytes = (words as u64).checked_mul(8)?;
        if a & XIO_IO_VIEW != 0 {
            crate::dlog!(LogModule::Hub, "Hub: unsupported inbound XIO block at {a:#x} of {words} doublewords");
            return None;
        }
        if a >= XIO_MEMORY_SIZE || bytes > XIO_MEMORY_SIZE - a {
            return None;
        }
        Some(f(&*hub.decode))
    }
}
impl InterruptController for Inbound {
    fn set_line(&self, line: u32, level: bool) {
        if line >= 128 {
            crate::dlog!(LogModule::Hub, "Hub: interrupt line {line} past the 128 the Hub has, dropped");
        } else if let Some(hub) = self.0.upgrade() {
            hub.pi.int_modify(u64::from(line) | if level { pi::MODIFY_SET } else { 0 });
        }
    }

    /// The packet's address names the node whose Hub takes it, as an
    /// interrupt-modify write does: a Bridge here can interrupt the other node.
    fn packet(&self, _source: u8, address: u64, line: u32, level: bool) -> bool {
        let Some(hub) = self.0.upgrade() else { return false };
        if line >= 128 {
            crate::dlog!(LogModule::Hub, "Hub: interrupt packet for line {line} past the 128 the Hub has, refused");
            return false;
        }
        hub.interrupt(xio_node(address), u64::from(line) | if level { pi::MODIFY_SET } else { 0 })
    }
}
impl BusDevice for Inbound {
    fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 {
        self.block(a, buf.len(), |mem| mem.read_block(a, buf)).unwrap_or(BUS_ERR)
    }
    fn write_block(&self, a: u64, buf: &[u64]) -> u32 {
        self.block(a, buf.len(), |mem| mem.write_block(a, buf)).unwrap_or(BUS_ERR)
    }
    access_widths!();
}

/// A peer's programmed I/O to this Hub's widget number answers only the
/// 4-byte read of the identification word at offset 4.
struct WidgetPort(Weak<Hub>);
impl BusDevice for WidgetPort {
    fn read32(&self, a: u64) -> BusRead32 {
        match self.0.upgrade().and_then(|hub| hub.io.widget_port_read32(a)) {
            Some(v) => BusRead32::ok(v),
            None => {
                crate::dlog!(LogModule::Hub, "Hub: unsupported peer read at {a:#x} on the Hub's widget port");
                BusRead32::err()
            }
        }
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        crate::dlog!(LogModule::Hub, "Hub: unsupported peer write of {v:#x} at {a:#x} on the Hub's widget port");
        BUS_ERR
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
    /// A store of `size` bytes; only the 32- and 64-bit stores change anything.
    #[inline(always)]
    fn store(&self, a: u64, size: usize, v: u64) -> (u32, Effect) {
        match size {
            4 => self.store32(a, v as u32),
            8 => self.store64(a, v),
            _ => (widths::access(self, a, size, Some(v)).status, Effect::None),
        }
    }

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
        let mut merged = if matches!(aligned, regs::PI_INT_PEND_MOD::OFFSET | regs::NI_PORT_ERROR::OFFSET) {
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

/// The Hub as the register window a machine maps. A 32- or 64-bit store
/// acts on what it changed; a narrower one only stores.
impl Hub {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let Some(v) = value else { return widths::access(&*self.regs, a, size, None) };
        let (status, effect) = self.regs.store(a, size, v);
        self.apply(effect);
        BusRead64 { status, data: 0 }
    }
}
impl BusDevice for Hub {
    access_widths!();
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
        // The link comes out of reset now, its width settled from both ends' force bits.
        if let Some(far) = self.decode.peer() {
            self.ni.link_up(far.ni.forces_eight_bit());
        }
        self.decode.remap(&|| {
            self.decode.set_number(0);
            self.decode.set_alias(0, calias_bytes(pi::CALIAS_RESET));
        });
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
