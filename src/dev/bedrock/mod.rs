//! Bedrock, the IP35 node controller, as one device: two processor interfaces (PI), memory and directory (MD),
//! the I/O block (II), the local block (LB), the network interface (NI) and the crossbar block (XB).
//!
//! Units are plain fields that lock only their own state and hold no reference to another; what an access changes
//! beyond its unit returns as an `Effect`, which the chip acts on after the unit's lock is released. The chip
//! calls nothing in the board: a node reset leaves as a [`MachineEvent::HardReset`]. LB's region store, a PI's
//! cached-alias store and MD's configuration store place memory; the chip remaps both nodes' Physicals
//! ([`crate::sn::NodeDecode::remap`]).
//!
//! [`Bedrock::connect_peer`] joins two chips once each way over a NUMAlink; a vector request or warm reset crosses
//! as an `Effect` after the originating lock is released, so both ends may launch at once.
//!
//! # What is approximated here
//!
//! A remote uncached reference reaches whatever the peer's processors reach, not the hardware's narrower list.
//! An access to a number no reachable chip holds fails rather than timing out. The reader-relative forms at node
//! field zero are inferred from firmware.
//!
//! # Locks
//!
//! - PI: interrupt inputs are atomics (no lock for mask, pending-modify or counter access); `state` holds the
//!   stored words and graphics registers, `timer` the comparators and host timer.
//! - MD: the cell register is an atomic and memory takes no lock. II: permission words and translation entries
//!   are atomics. LB, NI and XB: `regs` each, under which a block drives its error level.
//!
//! Ported from hw/misc/sgi-bedrock*.c in jamesbraid/qemu-sgi-origami.

#![warn(dead_code)]

pub(crate) mod io;
pub(crate) mod lb;
pub(crate) mod md;
mod mem;
mod ni;
mod pi;
pub mod regs;
mod xb;
#[cfg(test)]
mod tests;
#[cfg(test)]
use tests::{connected_pair, test_config, Rig};

pub use mem::{check_banks, RamExtent, BANK_SPAN, SLOTS};

pub use pi::Population;

use crate::dev::heart::Clock;
use crate::dev::widths::access_widths;
use crate::hptimer::{TimerId, TimerManager, TimerReturn};
use crate::irq::{InterruptController, IrqLine};
use crate::traits::{BusDevice, BusRead64, Device, MachineEvent, Resettable, BUS_ERR, BUS_OK};
use crate::sn::{
    calias_bytes, node_id, Node, NodeDecode, NodeId, UncachedSpace, IO_REGISTERS_LOCAL, IO_REGISTERS_NAMED, MODIFY_SET,
    RESET_WINDOW, XIO_IO_VIEW,
};
use io::Io;
use md::{Md, View};
use pi::{Levels, Pi};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

/// The node number's bit in a physical address: bits 39:33 name the node.
pub const NODE_SHIFT: u32 = regs::extra::NODE_SHIFT as u32;

/// What a board gives its Bedrock. Fixed for the chip's life.
pub struct BedrockConfig<'a> {
    /// The board's name in log lines.
    pub name: &'static str,
    /// Host nanoseconds: the PIs' counter and timers run on it, and the
    /// other units' clocked work.
    pub clock: Clock,
    /// The processor slots the board populates. CPU n sits in
    /// `processors.slot_of(n)`; a populated slot starts with its enable bit set.
    pub processors: Population,
    /// The memory banks in MiB, locational bank 0 upward.
    pub banks_mib: &'a [u32],
    /// The nodes of the board; more than one makes room in the memory window for every node at its
    /// number.
    pub nodes: usize,
    /// Where a node reset goes: LB's soft reset and NI's local reset each
    /// send [`MachineEvent::HardReset`].
    pub events: Option<Sender<MachineEvent>>,
    /// The reset PROM, addressed from zero.
    pub prom: Option<Arc<dyn BusDevice>>,
}

/// The reset PROM's window at [`crate::sn::RESET_WINDOW`].
pub const RESET_WINDOW_SIZE: u64 = 0x40_0000;

/// What asked for a node reset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeReset {
    /// LB's soft-reset bit.
    LocalBlock,
    /// NI's local-reset bit.
    Network,
    /// A warm reset arrived over the link and the reset enable took it.
    Link,
}

/// What a unit's access changed outside the unit, returned with the unit's
/// lock released for [`Bedrock::apply`].
#[must_use]
enum Effect {
    None,
    /// A bit of the chip's levels changed: both PIs publish.
    Reflect,
    /// LB's region write gave the node this number (seven bits).
    NodeNumber(NodeId),
    /// PI `.0`'s cached-alias store: the alias now spans `.1` bytes.
    CachedAlias(usize, u64),
    /// MD's configuration store moved the banks.
    Remap,
    /// MD's self-test zero-fills `.1` bytes of bank `.0`.
    Zero(usize, u64),
    /// Run block-transfer engine `n` now.
    Bte(usize),
    /// The unit has reset itself; the node resets.
    Reset(NodeReset),
    /// NI's port-reset bit: warm reset over the link, and with `and_local` this node too (firmware
    /// writes both).
    LinkReset { and_local: bool },
    /// LB launched a vector request: perform it at the far end of the port
    /// and latch the reply.
    Vector(lb::Launch),
}

/// Who makes a register access.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Who {
    /// A processor, a debugger, a test or a widget: the thread's CPU number,
    /// if any, is the requester, and MD's cell register gates the window.
    Cpu,
    /// The peer's vector engine. It is no processor of this node, and a
    /// remote request applies no permission check.
    Peer,
}
impl Who {
    /// The requesting processor, by its number in the node, if the thread is
    /// one of `processors`.
    fn cpu(self, processors: Population) -> Option<usize> {
        if self == Who::Cpu { crate::cpu::smp::on_cpu_thread().and_then(|cpu| processors.local(cpu)) } else { None }
    }
}

/// A one-shot `delay` nanoseconds out that latches PI `interface`'s due
/// comparators and arms the next.
fn arm_host_timer(tm: &TimerManager, chip: &Weak<Bedrock>, interface: u8, delay: u64) -> TimerId {
    tm.add_one_shot(Duration::from_nanos(delay), (chip.clone(), interface), |(chip, interface)| {
        if let Some(chip) = chip.upgrade() {
            chip.expire(usize::from(*interface));
        }
        TimerReturn::Delete
    })
}

pub struct Bedrock {
    name: &'static str,
    clock: Clock,
    processors: Population,
    pi: [Pi; 2],
    pub(crate) md: Md,
    pub(crate) io: Io,
    pub(crate) lb: lb::Lb,
    ni: ni::Ni,
    xb: xb::Xb,
    levels: Levels,
    /// This node and the chip at the far end of its NI's port, by the numbers they
    /// publish, and the node's memory, which the chip remaps.
    pub(crate) decode: Arc<NodeDecode<Bedrock>>,
    timers: OnceLock<Weak<TimerManager>>,
    events: Option<Sender<MachineEvent>>,
    weak: Weak<Self>,
}

/// Where an XIO address's I/O view names the node.
const XIO_NODE_SHIFT: u32 = 39;
impl Bedrock {
    pub fn new(config: BedrockConfig<'_>) -> Result<Arc<Self>, String> {
        let BedrockConfig { name, clock, processors, banks_mib, nodes, events, prom } = config;
        let memory = mem::Memory::new(banks_mib, nodes)?;
        let now = clock();
        let chip = Arc::new_cyclic(|weak: &Weak<Self>| {
            let md = Md::new(memory, clock.clone());
            // The Hub-special space's reader-relative forms end at 512 MiB.
            let node = Node::new(weak.clone(), md.physical().clone(), md.mappings(), prom.map(|p| (RESET_WINDOW, RESET_WINDOW_SIZE, p)), true);
            let decode = NodeDecode::new(NODE_SHIFT, 0x2000_0000, node, crate::devlog::LogModule::Bedrock);
            Self {
                name,
                pi: [Pi::new(0, processors, now), Pi::new(1, processors, now)],
                md,
                io: Io::new(),
                lb: lb::Lb::new(),
                ni: ni::Ni::new(),
                xb: xb::Xb::new(),
                levels: Levels::default(),
                decode,
                timers: OnceLock::new(),
                events,
                clock,
                processors,
                weak: weak.clone(),
            }
        });
        Ok(chip)
    }

    /// Connect this chip's NI port to the far chip of a NUMAlink, once, before the machine runs;
    /// the link then stays up.
    pub fn connect_peer(&self, far: &Bedrock) {
        assert!(!std::ptr::eq(self, far), "a Bedrock cannot be connected to itself");
        self.decode.connect_peer(far.decode.node());
    }
    pub fn register_locks(self: &Arc<Self>) {
        use crate::locks::register_mutex_fn;
        register_mutex_fn("bedrock-pi0::state", self, |c| c.pi[0].state_locked());
        register_mutex_fn("bedrock-pi1::state", self, |c| c.pi[1].state_locked());
        register_mutex_fn("bedrock-pi0::timer", self, |c| c.pi[0].timer_locked());
        register_mutex_fn("bedrock-pi1::timer", self, |c| c.pi[1].timer_locked());
        register_mutex_fn("bedrock-pi0::out", self, |c| c.pi[0].out_locked());
        register_mutex_fn("bedrock-pi1::out", self, |c| c.pi[1].out_locked());
        register_mutex_fn("bedrock-md::state", self, |c| c.md.state_locked());
        register_mutex_fn("bedrock-io", self, |c| c.io.locked());
        register_mutex_fn("bedrock-lb", self, |c| c.lb.locked());
        register_mutex_fn("bedrock-ni", self, |c| c.ni.locked());
        register_mutex_fn("bedrock-xb", self, |c| c.xb.locked());
    }

    /// The 8 MiB register window in its node XIO form, as the guest reaches it. It
    /// holds the chip weakly: the chip holds the spaces.
    #[cfg(test)]
    pub fn registers(&self) -> Arc<dyn BusDevice> {
        Arc::new(Registers(self.weak.clone()))
    }
    /// Whether the register window's local XIO form selects the requester's PI, so PI1's
    /// registers appear there for a processor on PI1. Approximated pending a hardware read.
    fn local_form(&self) -> bool {
        (0..2).any(|slot| self.processors.populated(1, slot))
    }
    /// The II's PIO window below 4 GiB, reached through the node's I/O space.
    #[cfg(test)]
    pub fn pio(&self) -> Arc<dyn BusDevice> {
        self.io_space()
    }
    /// The node's I/O space for the peer's processors: the register window and the PIO
    /// window below 4 GiB.
    pub fn io_space(&self) -> Arc<dyn BusDevice> {
        Arc::new(IoSpace { chip: self.weak.clone(), local: self.local_form() })
    }
    pub fn directory_bus(&self) -> Arc<dyn BusDevice> {
        self.md.view(self.weak.clone(), View::Directory)
    }
    pub fn ecc_bus(&self) -> Arc<dyn BusDevice> {
        self.md.view(self.weak.clone(), View::Ecc)
    }
    pub fn ecc_scrub_bus(&self) -> Arc<dyn BusDevice> {
        self.md.view(self.weak.clone(), View::Scrub)
    }
    /// A processor's view of `space`: node field zero stays on this node in the
    /// reader-relative forms, any other address goes to the node holding its number.
    pub fn cpu_view(&self, space: UncachedSpace) -> Arc<dyn BusDevice> {
        // The view holds this node's I/O space with the chip; the chip never holds the view.
        let own = (space == UncachedSpace::Io)
            .then(|| Arc::new(IoSpace { chip: self.weak.upgrade().expect("a live Bedrock"), local: self.local_form() }) as Arc<dyn BusDevice>);
        self.decode.cpu_view(space, own)
    }
    // Widgets.

    /// A read arriving from widget `source`'s device `device` (`None`: every
    /// master on the widget must be admitted): memory below bit 47.
    pub fn inbound_read(&self, source: u8, device: Option<u8>, address: u64, size: u8) -> BusRead64 {
        self.inbound_access(source, device, address, size, None)
    }
    /// A write from widget `source`: memory below bit 47, or an interrupt packet to a PI's
    /// pending-modify register.
    pub fn inbound_write(&self, source: u8, device: Option<u8>, address: u64, size: u8, value: u64) -> u32 {
        self.inbound_access(source, device, address, size, Some(value)).status
    }
    fn inbound_access(&self, widget: u8, device: Option<u8>, address: u64, size: u8, value: Option<u64>) -> BusRead64 {
        if address >= (1 << 48) || u64::from(size) > (1 << 48) - address || !self.io.admits(widget, device) {
            return BusRead64::err();
        }
        if address & XIO_IO_VIEW == 0 {
            return crate::dev::widths::access(&*self.decode, address, usize::from(size), value);
        }
        // The I/O view names a node in bits 46:39.
        let node = node_id(address >> XIO_NODE_SHIFT, NODE_SHIFT);
        let window = address & ((1 << XIO_NODE_SHIFT) - 1);
        if window & !(regs::WINDOW_SIZE - 1) != IO_REGISTERS_NAMED {
            return BusRead64::err();
        }
        // XIO register packets reach only the two PIs' pending-modify registers.
        const PI0_MODIFY: u64 = regs::pi::BASE + regs::pi::INT_PEND_MOD::OFFSET;
        const PI1_MODIFY: u64 = regs::PI1_BASE + regs::pi::INT_PEND_MOD::OFFSET;
        match (window & (regs::WINDOW_SIZE - 1), size, value) {
            (offset @ (PI0_MODIFY | PI1_MODIFY), 8, Some(value)) if self.deliver(node, usize::from(offset >= regs::PI1_BASE), value) => {
                BusRead64 { status: BUS_OK, data: 0 }
            }
            _ => BusRead64::err(),
        }
    }
    /// An interrupt packet to PI `pi` of node `node`, here or at the peer; whether the PI took it.
    fn deliver(&self, node: NodeId, pi: usize, value: u64) -> bool {
        self.decode.on_holder(self, node, |chip| chip.interrupt(pi, value)) == Some(true)
    }
    /// A block of doublewords from widget `source`'s device `device` into memory below bit 47,
    /// admitted and range-checked once. One starting in the register-packet space is single-word
    /// packets; one running into it or past 48 bits is refused.
    pub fn inbound_read_block(&self, source: u8, device: Option<u8>, address: u64, buf: &mut [u64]) -> u32 {
        if address & XIO_IO_VIEW != 0 {
            return self.inbound_packets(address, buf.len(), |i, at| {
                let r = self.inbound_access(source, device, at, 8, None);
                if r.status == BUS_OK {
                    buf[i] = r.data;
                }
                r.status
            });
        }
        self.inbound_block(source, device, address, buf.len(), |dma| dma.read_block(address, buf)).unwrap_or(BUS_ERR)
    }
    /// The write twin of [`Self::inbound_read_block`].
    pub fn inbound_write_block(&self, source: u8, device: Option<u8>, address: u64, buf: &[u64]) -> u32 {
        if address & XIO_IO_VIEW != 0 {
            return self.inbound_packets(address, buf.len(), |i, at| self.inbound_access(source, device, at, 8, Some(buf[i])).status);
        }
        self.inbound_block(source, device, address, buf.len(), |dma| dma.write_block(address, buf)).unwrap_or(BUS_ERR)
    }
    /// `words` single-word packets from `address`, stopping at the first refusal.
    fn inbound_packets(&self, address: u64, words: usize, mut each: impl FnMut(usize, u64) -> u32) -> u32 {
        (0..words).map(|i| each(i, address.wrapping_add(8 * i as u64))).find(|&s| s != BUS_OK).unwrap_or(BUS_OK)
    }
    fn inbound_block<R>(&self, widget: u8, device: Option<u8>, address: u64, words: usize, f: impl FnOnce(&dyn BusDevice) -> R) -> Option<R> {
        let bytes = (words as u64).checked_mul(8)?;
        if address >= (1 << 48) || bytes > (1 << 48) - address || !self.io.admits(widget, device) {
            return None;
        }
        if bytes > XIO_IO_VIEW - address {
            return None;
        }
        Some(f(&*self.decode))
    }
    /// An interrupt packet to PI `pi`'s pending-modify register; whether the
    /// PI took it.
    fn interrupt(&self, pi: usize, value: u64) -> bool {
        self.pi[pi].io_modify(value, 0, &self.levels)
    }
    /// `widget`'s interrupt packets, as a sender for its board: from `device` of the PIC there, or
    /// from the widget itself when `device` is `None`. Only packets for the host widget enter.
    pub fn interrupt_sender(&self, widget: u8, device: Option<u8>) -> crate::dev::odyssey::PacketSender {
        let chip = self.weak.clone();
        Arc::new(move |to, address, vector, set| {
            let Some(chip) = chip.upgrade().filter(|_| to == crate::dev::xbridge::HOST_WIDGET) else { return false };
            chip.inbound_write(widget, device, address, 8, u64::from(vector) | if set { MODIFY_SET } else { 0 }) == BUS_OK
        })
    }
    /// Widget `widget`'s port: its DMA and interrupt packets as inbound
    /// traffic, with the widget's inbound permissions applied.
    pub fn inbound(&self, widget: u8) -> Arc<Inbound> {
        Arc::new(Inbound { chip: self.weak.clone(), widget })
    }

    // Processors.

    /// Where CPU `cpu`'s IP2..IP6 go: Cause bits 10-14 of its interrupt word, from its slot. The
    /// word must outlive every publishing thread: stop them, call [`Self::clear_cpu_interrupts`],
    /// then free it.
    pub fn set_cpu_interrupts(&self, cpu: usize, word: *const AtomicU64) {
        if let Some((interface, slot)) = self.processors.slot_of(cpu) {
            let pi = &self.pi[usize::from(interface)];
            pi.set_word(slot, word);
            pi.publish(&self.levels);
        }
    }
    /// Let go of the processors' words. Takes each PI's `out`, so a publish already
    /// writing a word finishes before this returns.
    pub fn clear_cpu_interrupts(&self) {
        for pi in &self.pi {
            for slot in 0..2 {
                pi.set_word(slot, std::ptr::null());
            }
        }
    }
    /// The PIs' comparators and profile timers fire on `tm`. Held weakly.
    pub fn set_timer_manager(&self, tm: &Arc<TimerManager>) {
        let _ = self.timers.set(Arc::downgrade(tm));
        let now = (self.clock)();
        for pi in &self.pi {
            pi.reschedule(Some(tm), &self.weak, now);
        }
    }
    /// The L1 UART's line: bank 0 bit 4 of both PIs.
    pub fn uart_irq(self: &Arc<Self>) -> IrqLine {
        IrqLine::new(Arc::new(Uart(self.weak.clone())), 0)
    }
    fn timer_manager(&self) -> Option<Arc<TimerManager>> {
        self.timers.get().and_then(Weak::upgrade)
    }
    /// PI `interface`'s host timer fired.
    fn expire(&self, interface: usize) {
        let Some(tm) = self.timer_manager() else { return };
        self.pi[interface].expire(&tm, &self.weak, (self.clock)(), &self.levels);
    }
    fn publish_all(&self) {
        for pi in &self.pi {
            pi.publish(&self.levels);
        }
    }
    fn apply(&self, effect: Effect) {
        match effect {
            Effect::None => {}
            Effect::Reflect => self.publish_all(),
            Effect::Bte(n) => {
                if let Some(done) = self.io.run_bte(n, &*self.decode) {
                    if !self.deliver(done.node, done.pi, done.value) {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock IO BTE completion interrupt reaches no node");
                    }
                }
            }
            Effect::NodeNumber(n) => self.decode.remap(&|| self.decode.set_number(n)),
            Effect::CachedAlias(pi, bytes) => self.decode.remap(&|| self.decode.set_alias(pi, bytes)),
            // Read the runs inside the stop, under MD's state lock, so the last remap to run
            // maps the newest configuration; no parked processor holds that lock.
            Effect::Remap => self.decode.remap(&|| self.decode.set_mappings(self.md.mappings())),
            Effect::Zero(bank, size) => self.decode.remap(&|| self.md.memory().zero(bank, size)),
            Effect::Reset(cause) => {
                // LB's reset returns the node to number 0.
                self.decode.remap(&|| self.decode.set_number(0));
                crate::dlog!(crate::devlog::LogModule::Bedrock, "{}: {cause:?} reset from CPU {:?}; restarting the node",
                    self.name, crate::cpu::smp::on_cpu_thread());
                if let Some(events) = &self.events {
                    let _ = events.send(MachineEvent::HardReset);
                }
            }
            Effect::LinkReset { and_local } => {
                match self.decode.peer() {
                    Some(peer) => peer.link_reset_in(),
                    None => crate::dlog!(crate::devlog::LogModule::Bedrock, "{}: port reset with no link reaches nothing", self.name),
                }
                if and_local {
                    self.apply(Effect::Reset(NodeReset::Network));
                }
            }
            Effect::Vector(launch) => self.vector(&launch),
        }
    }
    /// A warm reset arrived over the link.
    fn link_reset_in(&self) {
        let effect = self.ni.link_reset(&self.levels);
        self.apply(effect);
    }
    /// Perform `launch` at the far end of the port and latch the reply; anything unanswerable
    /// latches nothing, so the firmware's poll times out. Called with no unit lock held: the peer
    /// may be launching at this chip.
    fn vector(&self, launch: &lb::Launch) {
        let Some(peer) = self.decode.peer().filter(|_| launch.route() == 0) else {
            crate::dlog!(crate::devlog::LogModule::Bedrock, "{}: vector launch on route {:#x} reaches no peer; no reply latched",
                self.name, launch.route());
            return;
        };
        let reply = peer.access(Who::Peer, 0, launch.offset(), 8, launch.write());
        if reply.status != BUS_OK {
            crate::dlog!(crate::devlog::LogModule::Bedrock, "{}: vector {} of {:#x} refused by the peer",
                self.name, if launch.write().is_some() { "write" } else { "read" }, launch.offset());
            return;
        }
        self.lb.vector_reply(launch, launch.write().is_none().then_some(reply.data));
    }

    /// The PI a local-form access at offset 0 reaches: the requester's own.
    /// A reader that is no processor reaches PI0.
    fn own_pi(&self) -> usize {
        crate::cpu::smp::on_cpu_thread()
            .and_then(|cpu| self.processors.slot_of_cpu(cpu))
            .map_or(0, |(interface, _)| usize::from(interface))
    }

    /// An access of `size` bytes at `offset` in the register window; PI0's range reaches PI `pi0`.
    /// Anything no unit answers is a bus error.
    fn access(&self, who: Who, pi0: usize, offset: u64, size: u8, value: Option<u64>) -> BusRead64 {
        if who == Who::Cpu && !self.md.cell_permits() {
            return BusRead64::err();
        }
        match offset {
            regs::pi::BASE..regs::PI1_BASE => self.pi_access(pi0, offset, size, value),
            regs::PI1_BASE..regs::io::BASE => self.pi_access(1, offset - regs::PI1_BASE, size, value),
            regs::io::BASE..regs::md::BASE => self.block_access(who, offset, size, value),
            regs::md::BASE..regs::WINDOW_SIZE => {
                let (result, effect) = self.md.access(View::Registers, offset - regs::md::BASE, u64::from(size), value);
                self.apply(effect);
                result.map_or(BusRead64::err(), BusRead64::ok)
            }
            _ => BusRead64::err(),
        }
    }
    fn pi_access(&self, index: usize, a: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let pi = &self.pi[index];
        let now = (self.clock)();
        match (size, value) {
            (8, None) if a < pi::SIZE && a & 7 == 0 => {
                pi.read(a, now, &self.levels).map_or(BusRead64::err(), BusRead64::ok)
            }
            (8, Some(v)) if a < pi::SIZE && a & 7 == 0 => {
                if a == regs::pi::RT_COMPARE_A::OFFSET || a == regs::pi::RT_COMPARE_B::OFFSET {
                    // The comparator's own processor, by slot: only it stores there while the
                    // PROM's liveness post is awaited.
                    let slot = (index as u8, usize::from(a == regs::pi::RT_COMPARE_B::OFFSET));
                    if let Some(cpu) = (0..self.processors.count()).find(|&n| self.processors.slot_of(n) == Some(slot)) {
                        self.lb.note_rt_compare(cpu);
                    }
                }
                if pi.write(a, v, now, &self.levels) {
                    let tm = self.timer_manager();
                    pi.reschedule(tm.as_deref(), &self.weak, now);
                }
                if a == regs::pi::CALIAS_SIZE::OFFSET {
                    self.apply(Effect::CachedAlias(index, calias_bytes(pi.cached_alias())));
                }
                BusRead64 { status: BUS_OK, data: 0 }
            }
            // The counter's two halves, high word first.
            (4, None) if a & !4 == regs::pi::RT_COUNTER::OFFSET => {
                BusRead64::ok(pi.count(now) >> (32 - (a & 4) * 8) & 0xffff_ffff)
            }
            _ => {
                pi.note(a, value.is_some());
                BusRead64::err()
            }
        }
    }
    /// An access to the II, LB, NI or XB at `offset` in the window: aligned
    /// doublewords only.
    fn block_access(&self, who: Who, offset: u64, size: u8, value: Option<u64>) -> BusRead64 {
        let (base, end) = match offset {
            regs::io::BASE..regs::lb::BASE => (regs::io::BASE, io::SIZE),
            regs::lb::BASE..regs::ni::BASE => (regs::lb::BASE, lb::SIZE),
            regs::ni::BASE..regs::xb::BASE => (regs::ni::BASE, ni::SIZE),
            _ => (regs::xb::BASE, xb::SIZE),
        };
        let a = offset - base;
        if size != 8 || a & 7 != 0 || a >= end {
            return BusRead64::err();
        }
        let now = (self.clock)();
        let (data, effect) = match (base, value) {
            (regs::io::BASE, None) => (self.io.read(a), Effect::None),
            (regs::io::BASE, Some(v)) => (0, self.io.write(a, v)),
            (regs::lb::BASE, None) => (self.lb.read(a, now, who.cpu(self.processors)), Effect::None),
            (regs::lb::BASE, Some(v)) => (0, self.lb.write(a, v, now, who.cpu(self.processors), &self.levels)),
            (regs::ni::BASE, None) => self.ni.read(a, &self.levels, self.decode.connected()),
            (regs::ni::BASE, Some(v)) => (0, self.ni.write(a, v, &self.levels)),
            (_, None) => (self.xb.read(a), Effect::None),
            (_, Some(v)) => (0, self.xb.write(a, v, &self.levels)),
        };
        self.apply(effect);
        BusRead64::ok(data)
    }
}

impl Device for Bedrock {
    fn step(&self, _cycles: u64) {}
    /// The counters run and the PIs' timers arm.
    fn start(&self) {
        let now = (self.clock)();
        self.md.start_clock();
        let tm = self.timer_manager();
        for pi in &self.pi {
            pi.start(tm.as_deref(), &self.weak, now);
        }
        self.publish_all();
    }
    /// The counters pause where they are and the PIs' timers stop.
    fn stop(&self) {
        let now = (self.clock)();
        let tm = self.timer_manager();
        for pi in &self.pi {
            pi.stop(tm.as_deref(), now, &self.levels);
        }
        self.md.stop_clock();
    }
    fn is_running(&self) -> bool {
        self.pi[0].is_running()
    }
    fn get_clock(&self) -> u64 {
        self.pi[0].count((self.clock)())
    }
}

impl Resettable for Bedrock {
    /// Every unit's reset. Memory keeps its contents.
    fn power_on(&self) {
        let now = (self.clock)();
        let tm = self.timer_manager();
        for pi in &self.pi {
            pi.power_on(now, &self.levels);
            pi.reschedule(tm.as_deref(), &self.weak, now);
        }
        self.md.power_on();
        self.lb.power_on();
        self.io.power_on();
        let _ = self.ni.power_on(&self.levels);
        let _ = self.xb.power_on(&self.levels);
        self.publish_all();
        // LB resets to node zero and MD to its reset configuration, read inside the stop as
        // in `apply`; each PI keeps its cached-alias register.
        let alias = self.pi.each_ref().map(|pi| calias_bytes(pi.cached_alias()));
        self.decode.remap(&|| {
            self.decode.set_number(0);
            self.decode.set_mappings(self.md.mappings());
            for (pi, &bytes) in alias.iter().enumerate() {
                self.decode.set_alias(pi, bytes);
            }
        });
    }
}

/// The register window as a processor reaches it: PI0's range answers PI0. Held
/// weakly, as the memory directory's views hold it: what the chip holds may hold it.
#[cfg(test)]
struct Registers(Weak<Bedrock>);
#[cfg(test)]
impl Registers {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        self.0.upgrade().map_or_else(BusRead64::err, |chip| chip.access(Who::Cpu, 0, a, size as u8, value))
    }
}
#[cfg(test)]
impl BusDevice for Registers {
    access_widths!();
}

/// Where the PIO window ends.
const PIO_SPAN: u64 = 1 << 32;

/// The node's I/O space: the register window at its local and node offsets and the PIO window
/// around them; an access past the window's end is a bus error. A processor's view holds the chip
/// as `Arc`, the owned space as `Weak`.
struct IoSpace<H> {
    chip: H,
    /// Whether the local register form selects the requester's PI.
    local: bool,
}
impl<H> IoSpace<H> {
    #[inline(always)]
    fn reach(&self, chip: &Bedrock, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        // The two register forms are one window apart, the local form first.
        const _: () = assert!(IO_REGISTERS_NAMED == IO_REGISTERS_LOCAL + regs::WINDOW_SIZE);
        let size = size as u64;
        if a & !(2 * regs::WINDOW_SIZE - 1) == IO_REGISTERS_LOCAL {
            let at = a & (regs::WINDOW_SIZE - 1);
            let pi = if self.local && a < IO_REGISTERS_NAMED { chip.own_pi() } else { 0 };
            match at.checked_add(size) {
                Some(end) if end <= regs::WINDOW_SIZE => chip.access(Who::Cpu, pi, at, size as u8, value),
                _ => BusRead64::err(),
            }
        } else {
            match a.checked_add(size) {
                Some(end) if end <= PIO_SPAN => chip.io.pio(a, size as u8, value),
                _ => BusRead64::err(),
            }
        }
    }
}
impl IoSpace<Arc<Bedrock>> {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        self.reach(&self.chip, a, size, value)
    }
}
impl IoSpace<Weak<Bedrock>> {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        self.chip.upgrade().map_or_else(BusRead64::err, |chip| self.reach(&chip, a, size, value))
    }
}
impl BusDevice for IoSpace<Arc<Bedrock>> {
    access_widths!();
}
impl BusDevice for IoSpace<Weak<Bedrock>> {
    access_widths!();
}

/// The L1 UART's interrupt line.
struct Uart(Weak<Bedrock>);
impl InterruptController for Uart {
    fn set_line(&self, _: u32, level: bool) {
        if let Some(chip) = self.0.upgrade() {
            if chip.levels.set(Levels::UART, level) {
                chip.publish_all();
            }
        }
    }
}

/// A crossbar widget's port into the chip: DMA as inbound reads, writes and blocks, interrupt
/// packets as inbound doubleword writes. Weak: the widget's device may hold it.
pub struct Inbound {
    chip: Weak<Bedrock>,
    widget: u8,
}
impl Inbound {
    fn from_source(&self, source: u8, a: u64, size: u8, value: Option<u64>) -> BusRead64 {
        match self.chip.upgrade() {
            Some(chip) => chip.inbound_access(source, None, a, size, value),
            None => BusRead64::err(),
        }
    }
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        self.from_source(self.widget, a, size as u8, value)
    }
}
impl BusDevice for Inbound {
    fn read_block(&self, a: u64, buf: &mut [u64]) -> u32 {
        self.chip.upgrade().map_or(BUS_ERR, |chip| chip.inbound_read_block(self.widget, None, a, buf))
    }
    fn write_block(&self, a: u64, buf: &[u64]) -> u32 {
        self.chip.upgrade().map_or(BUS_ERR, |chip| chip.inbound_write_block(self.widget, None, a, buf))
    }
    access_widths!();
}
impl InterruptController for Inbound {
    fn set_line(&self, line: u32, _: bool) {
        crate::dlog!(crate::devlog::LogModule::Bedrock, "widget {} interrupt {line} with no packet address dropped", self.widget);
    }
    fn packet(&self, source: u8, address: u64, line: u32, level: bool) -> bool {
        self.from_source(source, address, 8, Some(u64::from(line) | if level { MODIFY_SET } else { 0 })).status == BUS_OK
    }
}
