//! The Hub's network interface: node number and mode, the link to the
//! neighbour across the CrayLink, vector PIO, routing tables, scratch and port
//! registers and resets. Two Hubs joined by a CrayLink are the whole network:
//! the neighbour is reached by a direct vector; a nonzero vector needs a Router.
//!
//! # What is approximated here
//!
//! - The node number, mode and region size reset to zero, M mode and set (or
//!   the machine's preprogrammed number). Link-up and eight-bit follow the
//!   port's connection, which no machine takes down; a Hub with no peer reads
//!   down with the never-came-out-of-reset reason. The model stays in M mode.
//! - A vector launch reaching nothing leaves the status untouched, so the
//!   requester's poll times out. Overrun never sets. A target answers from its
//!   own registers; a vector write of the node number renumbers it; an
//!   undecoded register completes with the address error type.
//! - The routing tables are stored and drive nothing: the one path is the direct link.
//! - A link reset completes when written. Both ends settle the width from
//!   their force-eight-bit bits (`NI_DIAG_PARMS` bit 12) and latch
//!   `NI_PORT_ERROR` bit 37; the duration and the latch with no peer are unknown.
//! - Protection, I/O protection, ageing and diagnostic bits other than 0 and
//!   12 are stored and never consulted: enforcing protection firmware forgot
//!   to open would leave a dead machine with no diagnostic.
//! - `NI_PORT_RESET` bit 0 restarts the whole machine, not one node; bit 7
//!   alone is logged and does nothing (its effect alone is unknown).
//!
//! Ported from hw/misc/sgi-hub.c in jamesbraid/qemu-sgi-origami.

use super::Effect;
use crate::devlog::LogModule;
use crate::sn::{node_id, NodeId};
use crate::traits::Resettable;
use parking_lot::Mutex;

use super::regs as reg;

pub const META_ROUTES: usize = reg::NI_META_TABLE::COUNT;
pub const LOCAL_ROUTES: usize = reg::NI_LOCAL_TABLE::COUNT;

/// Bits: 30 eight-bit link, 29 link up, 28 down reason (1 never came out of
/// reset), 18 more nodes, 17 region size, 16:8 node number, 7:4 revision.
pub const STATUS_EIGHT_BIT: u64 = reg::NI_STATUS_REV_ID::EIGHTBIT_MODE::MASK;
pub const STATUS_LINK_UP: u64 = reg::NI_STATUS_REV_ID::LINK_UP::MASK;
pub const STATUS_NEVER_OUT_OF_RESET: u64 = reg::NI_STATUS_REV_ID::DOWN_REASON::MASK;
const STATUS_MORE_NODES: u64 = reg::NI_STATUS_REV_ID::MORE_NODES::MASK;
const STATUS_REGION_SIZE: u64 = reg::NI_STATUS_REV_ID::REGION_SIZE::MASK;
const STATUS_NODE_MASK: u64 = reg::NI_STATUS_REV_ID::NODE_ID::MASK;
const STATUS_RESET: u64 = STATUS_REGION_SIZE | STATUS_NEVER_OUT_OF_RESET;
const STATUS_WRITABLE: u64 = STATUS_MORE_NODES | STATUS_REGION_SIZE | STATUS_NODE_MASK;

const PORT_RESET_LOCAL: u64 = reg::NI_PORT_RESET::LOCAL_RESET::MASK;
const PORT_RESET_LINK: u64 = reg::NI_PORT_RESET::LINK_RESET::MASK;
const PORT_RESET_WARM: u64 = reg::NI_PORT_RESET::WARM_RESET::MASK;
const PROTECTION_MASK: u64 = reg::NI_PROTECTION::RESET_OK::MASK;
const DIAG_MASK: u64 = 0x7_ffff;
const DIAG_EIGHT_BIT: u64 = reg::NI_DIAG_PARMS::LLP_8BIT_MODE::MASK;
const AGEING_MASK: u64 = field_masks!(reg::NI_AGE_CPU0_MEMORY; VCHANNEL, CONGESTION, AGE);
const PORT_PARMS_MASK: u64 = field_masks!(reg::NI_PORT_PARMS; NULL_TIMEOUT, MAX_BURST);
const PORT_PARMS_RESET: u64 = reg::NI_PORT_PARMS::RESET;
const PORT_ERROR_MASK: u64 = 0x3f_ffff_ffff;
const PORT_ERROR_LINK_RESET: u64 = reg::NI_PORT_ERROR::LINK_RESET::MASK;
const ROUTE_MASK: u8 = reg::NI_META_TABLE::EXIT_PORT::MASK as u8;

const VECTOR_PIOID_MASK: u64 = reg::NI_VECTOR_PARMS::PIO_ID::MASK;
const VECTOR_WRITEID_MASK: u64 = reg::NI_VECTOR_PARMS::WRITE_ID::MASK;
const VECTOR_ADDRESS_MASK: u64 = reg::NI_VECTOR_PARMS::ADDRESS::MASK;
const VECTOR_TYPE_MASK: u64 = reg::NI_VECTOR_PARMS::PIO_TYPE::MASK;
const VECTOR_VALID: u64 = reg::NI_VECTOR_STATUS::VALID::MASK;
const VECTOR_TARGET_SHIFT: u32 = reg::NI_VECTOR_STATUS::TARGET::SHIFT;
const VECTOR_STATUS_ADDR_MASK: u64 = reg::NI_VECTOR_STATUS::ADDRESS::MASK;
pub const VECTOR_READ: u64 = reg::NI_VECTOR_PARMS::PIO_TYPE::READ;
pub const VECTOR_WRITE: u64 = reg::NI_VECTOR_PARMS::PIO_TYPE::WRITE;
pub const VECTOR_EXCHANGE: u64 = reg::NI_VECTOR_PARMS::PIO_TYPE::EXCHANGE;
pub const VECTOR_ADDRESS_ERROR: u64 = reg::NI_VECTOR_STATUS::PIO_TYPE::ADDRESS_ERROR;

/// One vector PIO request, as it arrives at the far end.
#[derive(Clone, Copy, Debug)]
pub struct VectorRequest {
    pub op: u64,
    /// Bits 19:3 of the register's address in the target's network block.
    pub address: u64,
    pub data: u64,
}

/// What the target of a vector request answers: `data` is `None` for an address
/// error, and `node` is the answering node's number.
pub struct VectorReply {
    pub data: Option<u64>,
    pub node: NodeId,
}

/// A vector request this port launched; the Hub performs it at the far end
/// outside this lock and hands the reply to [`Ni::vector_done`].
#[derive(Clone, Copy, Debug)]
pub struct VectorLaunch {
    parms: u64,
    /// The vector route: 0 is the Hub at the far end of the port.
    pub route: u64,
    data: u64,
}

impl VectorLaunch {
    /// The request as the target receives it.
    pub fn request(&self) -> VectorRequest {
        VectorRequest { op: self.parms & VECTOR_TYPE_MASK, address: self.parms & VECTOR_ADDRESS_MASK, data: self.data }
    }
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
    ageing: [u64; reg::NI_AGE::COUNT],
    io_protect: u64,
    protection: u64,
    diag_parms: u64,
    meta_route: [u8; META_ROUTES],
    local_route: [u8; LOCAL_ROUTES],
}
impl State {
    fn new(revision: u8) -> Self {
        Self {
            status: STATUS_RESET | (u64::from(revision) << reg::NI_STATUS_REV_ID::CHIP_REV::SHIFT),
            scratch: [reg::NI_SCRATCH_REG0::RESET, reg::NI_SCRATCH_REG1::RESET],
            vector: 0,
            vector_data: 0,
            vector_status: 0,
            return_vector: 0,
            vector_read_data: 0,
            port_parms: PORT_PARMS_RESET,
            port_error: 0,
            ageing: [reg::NI_AGE_CPU0_MEMORY::RESET; reg::NI_AGE::COUNT],
            io_protect: reg::NI_IO_PROTECT::RESET,
            protection: reg::NI_PROTECTION::RESET,
            diag_parms: 0,
            meta_route: [0; META_ROUTES],
            local_route: [0; LOCAL_ROUTES],
        }
    }
    fn number(&self) -> u16 {
        ((self.status & STATUS_NODE_MASK) >> reg::NI_STATUS_REV_ID::NODE_ID::SHIFT) as u16
    }
}

pub struct Ni {
    revision: u8,
    state: Mutex<State>,
}

impl Ni {
    /// `revision` is the chip revision code in NI_STATUS_REV_ID bits 7:4 (3 is
    /// Hub 2.1, 5 is Hub 2.3).
    pub fn new(revision: u8) -> Self {
        assert!(revision < 16);
        Self { revision, state: Mutex::new(State::new(revision)) }
    }

    /// The node number NI_STATUS_REV_ID holds, which a vector request
    /// reports as the answering node.
    pub fn number(&self) -> NodeId {
        self.state.lock().number()
    }

    /// Bring the link up at the width this end and the far end, which forces
    /// eight bits when `far_forces`, settle on now.
    pub(super) fn link_up(&self, far_forces: bool) {
        Self::up(&mut self.state.lock(), far_forces);
    }

    /// The link coming up clears the never-out-of-reset reason until the next reset.
    fn up(s: &mut State, far_forces: bool) {
        s.status = (s.status & !(STATUS_NEVER_OUT_OF_RESET | STATUS_EIGHT_BIT)) | STATUS_LINK_UP;
        if far_forces || s.diag_parms & DIAG_EIGHT_BIT != 0 {
            s.status |= STATUS_EIGHT_BIT;
        }
    }

    /// A link reset, completed as asked: the far end's width when there is
    /// one, else the link stays down. Other state keeps its values.
    pub(super) fn link_reset(&self, far_forces: Option<bool>) {
        let mut s = self.state.lock();
        s.port_error |= PORT_ERROR_LINK_RESET;
        match far_forces {
            None => s.status = (s.status & !(STATUS_LINK_UP | STATUS_EIGHT_BIT)) | STATUS_NEVER_OUT_OF_RESET,
            Some(far_forces) => Self::up(&mut s, far_forces),
        }
    }

    /// The far end reset the link: settle the width again and latch the event.
    pub(super) fn link_reset_from_far(&self, far_forces: bool) {
        let mut s = self.state.lock();
        s.port_error |= PORT_ERROR_LINK_RESET;
        Self::up(&mut s, far_forces);
    }

    /// Whether this end forces the link to eight bits.
    pub(super) fn forces_eight_bit(&self) -> bool {
        self.state.lock().diag_parms & DIAG_EIGHT_BIT != 0
    }

    fn route_entry(off: u64) -> Option<(bool, usize)> {
        if off % 8 != 0 {
            return None;
        }
        if let Some(i) = reg::NI_META_TABLE::index(off) {
            return Some((true, i));
        }
        reg::NI_LOCAL_TABLE::index(off).map(|i| (false, i))
    }

    fn ageing(off: u64) -> Option<usize> {
        reg::NI_AGE::index(off).filter(|&i| off == reg::NI_AGE::at(i))
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
            reg::NI_STATUS_REV_ID::OFFSET => s.status,
            reg::NI_PROTECTION::OFFSET => s.protection,
            reg::NI_SCRATCH_REG0::OFFSET | reg::NI_SCRATCH_REG1::OFFSET => s.scratch[((off - reg::NI_SCRATCH_REG0::OFFSET) / 8) as usize],
            reg::NI_DIAG_PARMS::OFFSET => s.diag_parms,
            reg::NI_VECTOR::OFFSET => s.vector,
            reg::NI_VECTOR_DATA::OFFSET => s.vector_data,
            reg::NI_VECTOR_STATUS::OFFSET | reg::NI_VECTOR_CLEAR::OFFSET => s.vector_status,
            reg::NI_RETURN_VECTOR::OFFSET => s.return_vector,
            reg::NI_VECTOR_READ_DATA::OFFSET => s.vector_read_data,
            reg::NI_IO_PROTECT::OFFSET => s.io_protect,
            reg::NI_PORT_PARMS::OFFSET => s.port_parms,
            reg::NI_PORT_ERROR::OFFSET | reg::NI_PORT_ERROR_CLEAR::OFFSET => s.port_error,
            _ => return None,
        })
    }

    /// Read the register at `off`. `clear` is false for a 32-bit half, which
    /// reads a read-to-clear register without clearing it.
    pub fn read(&self, off: u64, clear: bool) -> u64 {
        match off {
            reg::NI_VECTOR_CLEAR::OFFSET => {
                let mut s = self.state.lock();
                if clear {
                    s.vector_status = 0;
                }
                // Reads zero and clears the status.
                0
            }
            reg::NI_PORT_ERROR_CLEAR::OFFSET => {
                let mut s = self.state.lock();
                let value = s.port_error;
                if clear {
                    s.port_error = 0;
                }
                value
            }
            reg::NI_PORT_RESET::OFFSET | reg::NI_VECTOR_PARMS::OFFSET | reg::NI_IO_PROT_OVRRD::OFFSET => {
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
        let mut effect = Effect::None;
        {
            let mut s = self.state.lock();
            if let Some(i) = Self::ageing(off) {
                s.ageing[i] = value & AGEING_MASK;
                return Effect::None;
            }
            match off {
                reg::NI_STATUS_REV_ID::OFFSET => {
                    let before = s.number();
                    s.status = (s.status & !STATUS_WRITABLE) | (value & STATUS_WRITABLE);
                    let after = s.number();
                    if after != before {
                        effect = Effect::NodeNumber(node_id(u64::from(after), super::NODE_SHIFT));
                    }
                }
                reg::NI_PORT_RESET::OFFSET => {
                    drop(s);
                    effect = self.port_reset(value);
                }
                reg::NI_PROTECTION::OFFSET => s.protection = value & PROTECTION_MASK,
                reg::NI_SCRATCH_REG0::OFFSET | reg::NI_SCRATCH_REG1::OFFSET => {
                    let index = ((off - reg::NI_SCRATCH_REG0::OFFSET) / 8) as usize;
                    s.scratch[index] = value;
                }
                reg::NI_DIAG_PARMS::OFFSET => s.diag_parms = value & DIAG_MASK,
                reg::NI_VECTOR_PARMS::OFFSET => effect = Effect::Vector(VectorLaunch { parms: value, route: s.vector, data: s.vector_data }),
                reg::NI_VECTOR::OFFSET => s.vector = value,
                reg::NI_VECTOR_DATA::OFFSET => s.vector_data = value,
                reg::NI_VECTOR_STATUS::OFFSET => {
                    if value == 0 {
                        s.vector_status = 0;
                    }
                }
                reg::NI_RETURN_VECTOR::OFFSET => s.return_vector = value,
                reg::NI_VECTOR_READ_DATA::OFFSET => s.vector_read_data = value,
                reg::NI_VECTOR_CLEAR::OFFSET => s.vector_status = 0,
                reg::NI_IO_PROTECT::OFFSET => s.io_protect = value,
                reg::NI_IO_PROT_OVRRD::OFFSET => Self::note_override(value),
                reg::NI_PORT_PARMS::OFFSET => s.port_parms = value & PORT_PARMS_MASK,
                reg::NI_PORT_ERROR::OFFSET => s.port_error &= !(value & PORT_ERROR_MASK),
                _ => unimp("unimplemented or read-only register", off, Some(value)),
            }
        }
        effect
    }

    fn note_override(value: u64) {
        // The password unlock is a no-op on a model that never enforces
        // protection; a non-password write's effect is unstated.
        if value != reg::IIO_PROTECT_OVRRD_KEY {
            crate::dlog!(LogModule::Hub, "Hub NI: write of {value:#x} to the protection override (not the password; effect unstated)");
        }
    }

    fn write_route(&self, meta: bool, i: usize, value: u64) {
        let mut s = self.state.lock();
        let entry = value as u8 & ROUTE_MASK;
        if meta {
            s.meta_route[i] = entry;
        } else {
            s.local_route[i] = entry;
        }
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
            return Effect::LinkReset;
        }
        Effect::None
    }

    /// Latch the completion of `launch` from the target's reply.
    pub fn vector_done(&self, launch: &VectorLaunch, reply: Option<VectorReply>) {
        let Some(VectorReply { data: answer, node: source }) = reply else {
            crate::dlog!(LogModule::Hub, "Hub NI: vector launch (parms {:#x}) reached nothing that answers", launch.parms);
            return;
        };
        let (op, address) = (launch.parms & VECTOR_TYPE_MASK, launch.parms & VECTOR_ADDRESS_MASK);
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
            | (launch.parms & (VECTOR_PIOID_MASK | VECTOR_WRITEID_MASK))
            | (address & VECTOR_STATUS_ADDR_MASK)
            | response;
    }

    /// The target's half of a vector operation: a read or write reaches the
    /// decoded registers (status to routing tables); an exchange writes only over a zero.
    pub fn vector_access(&self, request: &VectorRequest) -> (Option<u64>, Effect) {
        let off = reg::NI_STATUS_REV_ID::OFFSET + request.address;
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
            reg::NI_PORT_ERROR_CLEAR::OFFSET => Some(self.read(off, true)),
            reg::NI_STATUS_REV_ID::OFFSET | reg::NI_PROTECTION::OFFSET | reg::NI_SCRATCH_REG0::OFFSET | reg::NI_SCRATCH_REG1::OFFSET | reg::NI_DIAG_PARMS::OFFSET | reg::NI_IO_PROTECT::OFFSET
            | reg::NI_PORT_PARMS::OFFSET | reg::NI_PORT_ERROR::OFFSET => self.peek(off),
            _ if Self::ageing(off).is_some() || Self::route_entry(off).is_some() => self.peek(off),
            _ => None,
        }
    }

    fn vector_write(&self, off: u64, value: u64, effect: &mut Effect) -> bool {
        match off {
            reg::NI_STATUS_REV_ID::OFFSET | reg::NI_PROTECTION::OFFSET | reg::NI_SCRATCH_REG0::OFFSET | reg::NI_SCRATCH_REG1::OFFSET | reg::NI_DIAG_PARMS::OFFSET | reg::NI_IO_PROTECT::OFFSET
            | reg::NI_PORT_PARMS::OFFSET | reg::NI_PORT_ERROR::OFFSET => *effect = self.store(off, value),
            reg::NI_IO_PROT_OVRRD::OFFSET => Self::note_override(value),
            _ if Self::ageing(off).is_some() || Self::route_entry(off).is_some() => *effect = self.store(off, value),
            _ => return false,
        }
        true
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
        *self.state.lock() = State::new(self.revision);
    }
}

#[cfg(test)]
mod tests;
