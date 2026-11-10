//! The Hub's network interface: the node number and mode, the link to the
//! neighbour across the CrayLink cable, vector PIO, the routing tables, the
//! scratch and port registers, and the resets.
//!
//! Offsets are from the base of the Hub window. Two Hubs joined by a cable
//! are the only fabric here: the neighbour is reached by a direct vector, and
//! a nonzero vector needs a Router the cable does not have.
//!
//! # What is approximated here
//!
//! - The node number, mode and region size reset to zero, M mode and set (or
//!   to the node number the machine preprograms). The link-up and
//!   eight-bit bits follow the cable; a Hub with no powered peer reads down
//!   with the never-came-out-of-reset reason set, which the guest kernel
//!   takes to mean no CrayLink rather than a lost one. The model stays in M mode.
//! - Failing, rather than timing out, an access to a node number no reachable
//!   Hub holds is an approximation (the Hub's node decode makes it from the numbers
//!   this unit mirrors).
//! - A vector launch that reaches nothing leaves the status untouched, so the
//!   requester's poll times out. Overrun never sets. A target answers from its
//!   own register store with no local side effects; a register outside the set
//!   a vector access decodes completes with the address error type.
//! - The routing tables are stored; an entry not written since reset routes
//!   nothing. They drive vector routing and the claim map only.
//! - A link reset completes when it is written. Both ends settle the width
//!   from their force-eight-bit bits (`NI_DIAG_PARMS` bit 12) and latch
//!   `NI_PORT_ERROR` bit 37. How long a reset takes, and the latch on a Hub
//!   with no peer, are unknown.
//! - Protection, I/O protection, ageing and the diagnostic bits other than
//!   0 and 12 are stored and never consulted: enforcing protection a firmware
//!   forgot to open would leave a dead machine with no diagnostic.
//! - `NI_PORT_RESET` bit 0 restarts the whole machine, not one node, and
//!   bit 7 alone is logged and does nothing; its effect alone is unknown.
//! - A Hub whose module is powered down is disabled: its port reads
//!   disabled and its peer sees the link fail.
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

use super::{Effect, Routing};
use crate::devlog::LogModule;
use crate::traits::{BusDevice, BusRead64, Resettable, BUS_OK};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};
use std::sync::{Arc, Weak};

pub mod reg {
    pub const STATUS: u64 = 0x60_0000;
    pub const PORT_RESET: u64 = 0x60_0008;
    pub const PROTECTION: u64 = 0x60_0010;
    pub const SCRATCH_0: u64 = 0x60_0100;
    pub const SCRATCH_1: u64 = 0x60_0108;
    pub const DIAG_PARMS: u64 = 0x60_0110;
    pub const VECTOR_PARMS: u64 = 0x60_0200;
    pub const VECTOR: u64 = 0x60_0208;
    pub const VECTOR_DATA: u64 = 0x60_0210;
    pub const VECTOR_STATUS: u64 = 0x60_0300;
    pub const RETURN_VECTOR: u64 = 0x60_0308;
    pub const VECTOR_READ_DATA: u64 = 0x60_0310;
    pub const VECTOR_CLEAR: u64 = 0x60_0380;
    pub const IO_PROTECT: u64 = 0x60_0400;
    pub const IO_PROT_OVRRD: u64 = 0x60_0408;
    pub const AGE_BASE: u64 = 0x60_0500;
    pub const PORT_PARMS: u64 = 0x60_8000;
    pub const PORT_ERROR: u64 = 0x60_8008;
    pub const PORT_ERROR_CLEAR: u64 = 0x60_8088;
    pub const META_TABLE: u64 = 0x63_8000;
    pub const LOCAL_TABLE: u64 = 0x63_8100;
}

pub const START: u64 = 0x60_0000;
pub const END: u64 = 0x80_0000;
pub const META_ROUTES: usize = 32;
pub const LOCAL_ROUTES: usize = 16;
/// Node numbers the routing tables name.
pub const NUMBERS: u16 = 256;

/// Bit 30 eight-bit link, 29 link up, 28 down reason (0 failed, 1 never came
/// out of reset), 18 more nodes (M mode when clear), 17 region size, 16:8 the
/// node number, 7:4 the chip revision.
pub const STATUS_EIGHT_BIT: u64 = 1 << 30;
pub const STATUS_LINK_UP: u64 = 1 << 29;
pub const STATUS_NEVER_OUT_OF_RESET: u64 = 1 << 28;
const STATUS_MORE_NODES: u64 = 1 << 18;
const STATUS_REGION_SIZE: u64 = 1 << 17;
const STATUS_NODE_MASK: u64 = 0x1ff << 8;
const STATUS_RESET: u64 = STATUS_REGION_SIZE | STATUS_NEVER_OUT_OF_RESET;
const STATUS_WRITABLE: u64 = STATUS_MORE_NODES | STATUS_REGION_SIZE | STATUS_NODE_MASK;

const PORT_RESET_LOCAL: u64 = 1 << 0;
const PORT_RESET_LINK: u64 = 1 << 1;
const PORT_RESET_WARM: u64 = 1 << 7;
const PROTECTION_MASK: u64 = 1;
const DIAG_MASK: u64 = 0x7_ffff;
const DIAG_SEND_DATA_ERROR: u64 = 1 << 0;
const DIAG_EIGHT_BIT: u64 = 1 << 12;
const AGEING_MASK: u64 = 0xfff;
const PORT_PARMS_MASK: u64 = 0xffff;
const PORT_PARMS_RESET: u64 = (6 << 10) | 0x3f0;
const PORT_ERROR_MASK: u64 = 0x3f_ffff_ffff;
const PORT_ERROR_LINK_RESET: u64 = 1 << 37;
const PORT_CHECK_ERROR_SHIFT: u32 = 8;
const PROT_OVRRD_PASSWORD: u64 = 0x5347_4972_756c_6573;
const ROUTE_MASK: u8 = 0xf;

const VECTOR_PIOID_MASK: u64 = 0x3ff << 40;
const VECTOR_WRITEID_MASK: u64 = 0xff << 32;
const VECTOR_ADDRESS_MASK: u64 = 0xf_fff8;
const VECTOR_TYPE_MASK: u64 = 3;
const VECTOR_VALID: u64 = 1 << 63;
const VECTOR_TARGET_SHIFT: u32 = 51;
const VECTOR_STATUS_ADDR_MASK: u64 = 0xffff_fff8;
pub const VECTOR_READ: u64 = 0;
pub const VECTOR_WRITE: u64 = 1;
pub const VECTOR_EXCHANGE: u64 = 3;
pub const VECTOR_ADDRESS_ERROR: u64 = 4;

/// One vector PIO request, as it arrives at the far end.
#[derive(Clone, Copy, Debug)]
pub struct VectorRequest {
    pub op: u64,
    /// Bits 19:3 of the register's address in the target's network block.
    pub address: u64,
    pub data: u64,
    /// What is left of the route; a Hub target never forwards.
    pub vector: u64,
}

/// The far end of the cable, as this Hub's port sees it.
pub trait NetPeer: Send + Sync {
    /// Perform a vector request at the target: its answer, or `None` for an
    /// address error, the answering node number, and what the request changed
    /// that the target's Hub must act on.
    fn receive(&self, request: &VectorRequest) -> (Option<u64>, u16, Effect);
    /// The far end's programmed node number.
    fn number(&self) -> u16;
    /// The far end is powered and its port enabled.
    fn enabled(&self) -> bool;
    /// The far end forces the link to eight bits.
    fn forces_eight_bit(&self) -> bool;
    /// The far end reset the link: settle the width again and latch the event.
    fn link_reset(&self);
    /// The far end's node number or routes changed.
    fn changed(&self);
}

fn unimp(what: &str, off: u64, value: Option<u64>) {
    match value {
        Some(v) => crate::dlog!(LogModule::Hub, "Hub NI: write of {v:#x} to {what} {off:#x}"),
        None => crate::dlog!(LogModule::Hub, "Hub NI: read of {what} {off:#x}"),
    }
}

struct State {
    status: u64,
    scratch: [u64; 2],
    vector: u64,
    vector_data: u64,
    vector_status: u64,
    return_vector: u64,
    vector_read_data: u64,
    port_parms: u64,
    port_error: u64,
    ageing: [u64; 8],
    io_protect: u64,
    protection: u64,
    diag_parms: u64,
    meta_route: [u8; META_ROUTES],
    local_route: [u8; LOCAL_ROUTES],
    meta_written: u32,
    local_written: u16,
}
impl State {
    fn new(revision: u8) -> Self {
        Self {
            status: STATUS_RESET | (u64::from(revision) << 4),
            scratch: [0; 2],
            vector: 0,
            vector_data: 0,
            vector_status: 0,
            return_vector: 0,
            vector_read_data: 0,
            port_parms: PORT_PARMS_RESET,
            port_error: 0,
            ageing: [0; 8],
            io_protect: !0,
            protection: 1,
            diag_parms: 0,
            meta_route: [0; META_ROUTES],
            local_route: [0; LOCAL_ROUTES],
            meta_written: 0,
            local_written: 0,
        }
    }
    fn number(&self) -> u16 {
        ((self.status & STATUS_NODE_MASK) >> 8) as u16
    }
}

pub struct Ni {
    revision: u8,
    state: Mutex<State>,
    peer: Mutex<Option<Weak<dyn NetPeer>>>,
    /// The module has power. A disabled port reads down at its peer.
    enabled: AtomicBool,
    /// The node number and the neighbour's, mirrored from the state and the
    /// cable whenever either changes, so the Hub's node decode reads them
    /// without a lock. `NO_NEIGHBOUR` while no powered neighbour is across.
    routing: Arc<Routing>,
    neighbour: AtomicU32,
}

const NO_NEIGHBOUR: u32 = u32::MAX;

impl Ni {
    /// `revision` is the chip revision code in NI_STATUS_REV_ID bits 7:4 (3 is
    /// Hub 2.1, 5 is Hub 2.3).
    pub fn new(revision: u8, routing: Arc<Routing>) -> Self {
        assert!(revision < 16);
        Self {
            revision,
            state: Mutex::new(State::new(revision)),
            peer: Mutex::new(None),
            enabled: AtomicBool::new(true),
            routing,
            neighbour: AtomicU32::new(NO_NEIGHBOUR),
        }
    }

    /// Join this port to the far end of a cable, or take the cable off.
    pub fn connect_peer(&self, peer: Option<Weak<dyn NetPeer>>) {
        *self.peer.lock() = peer;
        self.announce();
    }

    /// Power to the module comes or goes. A disabled port reads down at the
    /// peer, with the reason left at "failed". No machine takes a module's
    /// power from its Hub, so only the tests do this.
    #[cfg(test)]
    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Release);
        if let Some(peer) = self.peer() {
            peer.changed();
        }
    }

    fn peer(&self) -> Option<Arc<dyn NetPeer>> {
        self.peer.lock().as_ref().and_then(Weak::upgrade)
    }

    /// The neighbour, if the cable has one that is up.
    fn neighbour(&self) -> Option<Arc<dyn NetPeer>> {
        self.peer().filter(|peer| peer.enabled())
    }

    /// The node number this Hub answers to.
    pub fn number(&self) -> u16 {
        self.routing.number()
    }

    /// The direction the first Router takes for a packet to `destination`, or
    /// `None` while the entry has not been written since reset.
    pub fn first_turn(&self, destination: u16) -> Option<u8> {
        if destination >= NUMBERS {
            return None;
        }
        let s = self.state.lock();
        if destination >> 4 == s.number() >> 4 {
            let i = usize::from(destination & 0xf);
            (s.local_written & (1 << i) != 0).then(|| s.local_route[i])
        } else {
            let i = usize::from(destination >> 4);
            (s.meta_written & (1 << i) != 0).then(|| s.meta_route[i])
        }
    }

    /// Whether the diagnostic bit that makes this Hub's vector writes count a
    /// check-bit error at the first Router is set.
    pub fn sends_data_error(&self) -> bool {
        self.state.lock().diag_parms & DIAG_SEND_DATA_ERROR != 0
    }

    /// A Router counted a check-bit error against this Hub: bits 15:8 of the
    /// port error count them, saturating at 255.
    pub fn record_check_error(&self) {
        let mut s = self.state.lock();
        let count = (s.port_error >> PORT_CHECK_ERROR_SHIFT) & 0xff;
        if count != 0xff {
            s.port_error = (s.port_error & !(0xff << PORT_CHECK_ERROR_SHIFT)) | ((count + 1) << PORT_CHECK_ERROR_SHIFT);
        }
    }

    pub fn latch_port_error(&self, errors: u64) {
        self.state.lock().port_error |= errors & PORT_ERROR_MASK;
    }

    /// Re-derive the link bits and the neighbour's number from the cable.
    pub fn refresh(&self) {
        self.derive();
    }

    /// This port's own state changed (the node number, a route, the cable):
    /// re-derive it and tell the neighbour, which re-derives its own and does
    /// not answer back.
    fn announce(&self) {
        self.refresh();
        if let Some(peer) = self.peer() {
            peer.changed();
        }
    }

    /// Whether a powered neighbour is across the cable, and its node number,
    /// as the cable and the neighbour last announced them. Lock-free.
    pub fn neighbour_number(&self) -> Option<u16> {
        match self.neighbour.load(Ordering::Acquire) {
            NO_NEIGHBOUR => None,
            number => Some(number as u16),
        }
    }

    /// The link bits: up when a powered neighbour is across the cable. The
    /// width is what the link settled on when it came up or was last reset,
    /// so a link that stays up keeps it; one that is down has none.
    fn derive(&self) {
        // Ask the far end before taking this state's lock: the two ends
        // never hold their locks across a call to each other.
        let neighbour = self.neighbour();
        let forces = neighbour.as_ref().map(|peer| peer.forces_eight_bit());
        self.neighbour.store(neighbour.as_ref().map_or(NO_NEIGHBOUR, |peer| u32::from(peer.number())), Ordering::Release);
        let mut s = self.state.lock();
        match forces {
            None => s.status &= !(STATUS_LINK_UP | STATUS_EIGHT_BIT),
            Some(forces) => {
                if s.status & STATUS_LINK_UP == 0 {
                    Self::link_up(&mut s, forces);
                }
            }
        }
    }

    /// Bring this end up at the width the two ends settle on now. The link
    /// coming up clears the never-out-of-reset reason until the next reset.
    fn link_up(s: &mut State, peer_forces: bool) {
        s.status = (s.status & !(STATUS_NEVER_OUT_OF_RESET | STATUS_EIGHT_BIT)) | STATUS_LINK_UP;
        if peer_forces || s.diag_parms & DIAG_EIGHT_BIT != 0 {
            s.status |= STATUS_EIGHT_BIT;
        }
    }

    /// A link reset, completed as it is asked for. Node number, mode, routing,
    /// vector and diagnostic state keep their values, which is what lets
    /// software reset the link and keep running.
    fn link_reset(&self) {
        let neighbour = self.neighbour();
        let forces = neighbour.as_ref().map(|peer| peer.forces_eight_bit());
        {
            let mut s = self.state.lock();
            s.port_error |= PORT_ERROR_LINK_RESET;
            match forces {
                None => {
                    s.status = (s.status & !(STATUS_LINK_UP | STATUS_EIGHT_BIT)) | STATUS_NEVER_OUT_OF_RESET;
                }
                Some(forces) => Self::link_up(&mut s, forces),
            }
        }
        if let Some(peer) = neighbour {
            peer.link_reset();
        }
    }

    fn route_entry(off: u64) -> Option<(bool, usize)> {
        if off % 8 != 0 {
            return None;
        }
        if let Some(rel) = off.checked_sub(reg::META_TABLE).filter(|rel| rel / 8 < META_ROUTES as u64) {
            return Some((true, (rel / 8) as usize));
        }
        let rel = off.checked_sub(reg::LOCAL_TABLE).filter(|rel| rel / 8 < LOCAL_ROUTES as u64)?;
        Some((false, (rel / 8) as usize))
    }

    fn ageing(off: u64) -> Option<usize> {
        let rel = off.checked_sub(reg::AGE_BASE)?;
        (rel % 8 == 0 && rel / 8 < 8).then_some((rel / 8) as usize)
    }

    /// A register's value without side effects, for the registers that store.
    pub fn peek(&self, off: u64) -> Option<u64> {
        let s = self.state.lock();
        if let Some((meta, i)) = Self::route_entry(off) {
            return Some(u64::from(if meta { s.meta_route[i] } else { s.local_route[i] }));
        }
        if let Some(i) = Self::ageing(off) {
            return Some(s.ageing[i]);
        }
        Some(match off {
            reg::STATUS => s.status,
            reg::PROTECTION => s.protection,
            reg::SCRATCH_0 | reg::SCRATCH_1 => s.scratch[((off - reg::SCRATCH_0) / 8) as usize],
            reg::DIAG_PARMS => s.diag_parms,
            reg::VECTOR => s.vector,
            reg::VECTOR_DATA => s.vector_data,
            reg::VECTOR_STATUS | reg::VECTOR_CLEAR => s.vector_status,
            reg::RETURN_VECTOR => s.return_vector,
            reg::VECTOR_READ_DATA => s.vector_read_data,
            reg::IO_PROTECT => s.io_protect,
            reg::PORT_PARMS => s.port_parms,
            reg::PORT_ERROR | reg::PORT_ERROR_CLEAR => s.port_error,
            _ => return None,
        })
    }

    /// Read the register at `off`. `clear` is false for a 32-bit half, which
    /// reads a read-to-clear register without clearing it.
    pub fn read(&self, off: u64, clear: bool) -> u64 {
        match off {
            reg::VECTOR_CLEAR => {
                let mut s = self.state.lock();
                if clear {
                    s.vector_status = 0;
                }
                // Reads zero and clears the status.
                0
            }
            reg::PORT_ERROR_CLEAR => {
                let mut s = self.state.lock();
                let value = s.port_error;
                if clear {
                    s.port_error = 0;
                }
                value
            }
            reg::PORT_RESET | reg::VECTOR_PARMS | reg::IO_PROT_OVRRD => {
                unimp("write-only register", off, None);
                0
            }
            _ => match self.peek(off) {
                Some(v) => v,
                None => {
                    unimp("unimplemented register", off, None);
                    0
                }
            },
        }
    }

    /// A store from the bus. What it changed beyond the register is returned
    /// for the Hub to act on, after this unit's lock is released.
    pub fn write(&self, off: u64, value: u64) -> Effect {
        self.store(off, value)
    }

    /// A store to the register at `off`, from the bus or from another node's
    /// vector PIO.
    fn store(&self, off: u64, value: u64) -> Effect {
        if let Some((meta, i)) = Self::route_entry(off) {
            self.write_route(meta, i, value);
            return Effect::None;
        }
        let mut changed = false;
        let mut effect = Effect::None;
        {
            let mut s = self.state.lock();
            if let Some(i) = Self::ageing(off) {
                s.ageing[i] = value & AGEING_MASK;
                return Effect::None;
            }
            match off {
                reg::STATUS => {
                    let before = s.number();
                    s.status = (s.status & !STATUS_WRITABLE) | (value & STATUS_WRITABLE);
                    changed = true;
                    let after = s.number();
                    self.routing.set_number(after);
                    if after != before {
                        effect = Effect::NodeNumber;
                    }
                }
                reg::PORT_RESET => {
                    drop(s);
                    effect = self.port_reset(value);
                }
                reg::PROTECTION => s.protection = value & PROTECTION_MASK,
                reg::SCRATCH_0 | reg::SCRATCH_1 => {
                    let index = ((off - reg::SCRATCH_0) / 8) as usize;
                    s.scratch[index] = value;
                }
                reg::DIAG_PARMS => s.diag_parms = value & DIAG_MASK,
                reg::VECTOR_PARMS => {
                    let request = (s.vector, s.vector_data);
                    drop(s);
                    effect = self.vector_launch(value, request);
                }
                reg::VECTOR => s.vector = value,
                reg::VECTOR_DATA => s.vector_data = value,
                reg::VECTOR_STATUS => {
                    if value == 0 {
                        s.vector_status = 0;
                    }
                }
                reg::RETURN_VECTOR => s.return_vector = value,
                reg::VECTOR_READ_DATA => s.vector_read_data = value,
                reg::VECTOR_CLEAR => s.vector_status = 0,
                reg::IO_PROTECT => s.io_protect = value,
                reg::IO_PROT_OVRRD => Self::note_override(value),
                reg::PORT_PARMS => s.port_parms = value & PORT_PARMS_MASK,
                reg::PORT_ERROR => s.port_error &= !(value & PORT_ERROR_MASK),
                _ => unimp("unimplemented or read-only register", off, Some(value)),
            }
        }
        if changed {
            self.announce();
        }
        effect
    }

    fn note_override(value: u64) {
        // The password unlock is a no-op on a model that never enforces
        // protection; a non-password write's effect is unstated.
        if value != PROT_OVRRD_PASSWORD {
            crate::dlog!(LogModule::Hub, "Hub NI: write of {value:#x} to the protection override (not the password; effect unstated)");
        }
    }

    fn write_route(&self, meta: bool, i: usize, value: u64) {
        {
            let mut s = self.state.lock();
            let entry = value as u8 & ROUTE_MASK;
            // An undefined reset entry does not establish a programmed route.
            if meta {
                s.meta_route[i] = entry;
                s.meta_written |= 1 << i;
            } else {
                s.local_route[i] = entry;
                s.local_written |= 1 << i;
            }
        }
        self.announce();
    }

    fn port_reset(&self, value: u64) -> Effect {
        if value & PORT_RESET_WARM != 0 && value & PORT_RESET_LOCAL == 0 {
            crate::dlog!(LogModule::Hub, "Hub NI: port reset bit 7 without bit 0 is not implemented");
        }
        if value & PORT_RESET_LOCAL != 0 {
            // Software that resets the machine writes bits 7 and 0 together.
            return Effect::Restart;
        }
        if value & PORT_RESET_LINK != 0 {
            self.link_reset();
        }
        Effect::None
    }

    /// Vector PIO: launch a request from this Hub's own port and perform the
    /// access on whatever the route ends at.
    fn vector_launch(&self, parms: u64, (vector, data): (u64, u64)) -> Effect {
        let address = parms & VECTOR_ADDRESS_MASK;
        let op = parms & VECTOR_TYPE_MASK;
        let Some(peer) = self.neighbour() else {
            // A route nothing answers latches nothing: the poll times out.
            crate::dlog!(LogModule::Hub, "Hub NI: vector launch (parms {parms:#x}) reached nothing that answers");
            return Effect::None;
        };
        let request = VectorRequest { op, address, data, vector };
        // The neighbour is on the wire and answers, but a route past it needs
        // a Router the cable does not have: the address-error response, not
        // the timeout a cable with nothing on it leaves.
        let (answer, source, effect) =
            if vector != 0 { (None, peer.number(), Effect::None) } else { peer.receive(&request) };
        let mut s = self.state.lock();
        let mut response = op;
        match answer {
            Some(value) => {
                if op != VECTOR_WRITE {
                    s.vector_read_data = value;
                }
            }
            None => response = VECTOR_ADDRESS_ERROR,
        }
        s.vector_status = VECTOR_VALID
            | (u64::from(source) << VECTOR_TARGET_SHIFT)
            | (parms & (VECTOR_PIOID_MASK | VECTOR_WRITEID_MASK))
            | (address & VECTOR_STATUS_ADDR_MASK)
            | response;
        // A request that renumbered the neighbour tells that Hub's machine.
        match effect {
            Effect::NodeNumber => Effect::PeerNodeNumber,
            _ => Effect::None,
        }
    }

    /// The target's half of a vector operation. A read or write reaches the
    /// registers a vector access decodes: status,
    /// protection, scratch, diagnostics, I/O protection, ageing, the port
    /// registers and both routing tables. An exchange writes only over a zero.
    fn vector_access(&self, request: &VectorRequest) -> (Option<u64>, Effect) {
        let off = reg::STATUS + request.address;
        let mut effect = Effect::None;
        let answer = match request.op {
            VECTOR_READ => self.vector_read(off),
            VECTOR_WRITE => self.vector_write(off, request.data, &mut effect).then_some(0),
            VECTOR_EXCHANGE => match self.vector_read(off) {
                Some(old) if old != 0 || self.vector_write(off, request.data, &mut effect) => Some(old),
                _ => None,
            },
            _ => None,
        };
        (answer, effect)
    }

    fn vector_read(&self, off: u64) -> Option<u64> {
        match off {
            reg::PORT_ERROR_CLEAR => Some(self.read(off, true)),
            reg::STATUS | reg::PROTECTION | reg::SCRATCH_0 | reg::SCRATCH_1 | reg::DIAG_PARMS | reg::IO_PROTECT
            | reg::PORT_PARMS | reg::PORT_ERROR => self.peek(off),
            _ if Self::ageing(off).is_some() || Self::route_entry(off).is_some() => self.peek(off),
            _ => None,
        }
    }

    fn vector_write(&self, off: u64, value: u64, effect: &mut Effect) -> bool {
        match off {
            reg::STATUS | reg::PROTECTION | reg::SCRATCH_0 | reg::SCRATCH_1 | reg::DIAG_PARMS | reg::IO_PROTECT
            | reg::PORT_PARMS | reg::PORT_ERROR => *effect = self.store(off, value),
            reg::IO_PROT_OVRRD => Self::note_override(value),
            _ if Self::ageing(off).is_some() || Self::route_entry(off).is_some() => *effect = self.store(off, value),
            _ => return false,
        }
        true
    }
}

impl NetPeer for Ni {
    fn receive(&self, request: &VectorRequest) -> (Option<u64>, u16, Effect) {
        let (answer, effect) =
            if self.enabled.load(Ordering::Acquire) { self.vector_access(request) } else { (None, Effect::None) };
        (answer, self.number(), effect)
    }
    fn number(&self) -> u16 {
        Ni::number(self)
    }
    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }
    fn forces_eight_bit(&self) -> bool {
        self.state.lock().diag_parms & DIAG_EIGHT_BIT != 0
    }
    fn link_reset(&self) {
        // The peer reset the link: settle the width again and latch the event.
        let forces = self.neighbour().map(|peer| peer.forces_eight_bit());
        let mut s = self.state.lock();
        s.port_error |= PORT_ERROR_LINK_RESET;
        if let Some(forces) = forces {
            Self::link_up(&mut s, forces);
        }
    }
    fn changed(&self) {
        self.refresh();
    }
}

impl Ni {
    /// Whether the register state is locked now, for the `locks` monitor command.
    pub fn is_locked(&self) -> bool {
        self.state.is_locked()
    }
}

impl Resettable for Ni {
    fn power_on(&self) {
        {
            let mut s = self.state.lock();
            *s = State::new(self.revision);
        }
        self.routing.set_number(0);
        // The link comes out of reset now: its width is settled from both
        // ends' force bits as they stand.
        self.announce();
    }
}

impl BusDevice for Ni {
    fn read64(&self, addr: u64) -> BusRead64 {
        BusRead64::ok(self.read(addr, true))
    }
    fn write64(&self, addr: u64, value: u64) -> u32 {
        let _ = self.write(addr, value);
        BUS_OK
    }
}

#[cfg(test)]
mod tests;
