//! Bedrock's network interface (NI) exposes node routing, sideband and error
//! registers. The Fuel runs as one node without a NUMAlink peer, but the
//! guest still reads and programs these local control words during machine
//! setup. A chip whose port is connected ([`super::Bedrock::connect_peer`]) reads its
//! port status from the link and can send its peer a warm reset.
//!
//! # What is approximated here
//!
//! Routing and sideband writes are retained; the link protocol and packet
//! delivery are not simulated, and the link comes up when the port is connected, with no
//! training sequence. Unspecified power-on fields start at zero, and the
//! sideband word survives reset as the current compatibility behavior.
//!
//! Ported from hw/misc/sgi-bedrock-ni.c and hw/misc/sgi-bedrock-ni-regs.c
//! in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::{Effect, NodeReset};
use parking_lot::Mutex;

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = reg::GLOBAL_TABLE::OFFSET + reg::GLOBAL_TABLE::WIDTH as u64 / 8;

/// Port status of a node with no peer.
///
/// Established from the PROM's and IRIX's code: only bit 1 is read, as
/// link-up, and nothing reads bit 0 alone. With no peer the PROM polls bit 1,
/// prints "Local hub NUMAlink is down", records code 0x9c and carries on, so
/// bit 1 is clear here. Real Fuels' consoles show the same line
/// (comp.sys.sgi.admin, Atro Tossavainen 2005-04-13, Tristram Scott
/// 2007-07-18).
///
/// Unknown: bits 0 and 2 and the word as a whole on a real Fuel with no peer.
/// Bit 0 is our choice, kept because the IRIX 6.5.30 bus-error handler prints
/// "Detected a link reset due to other partition reset" on every faulting
/// widget probe when this word and the reset-enable register are both zero,
/// and the 2005 Fuel's IRIX took a PIO read error with no such line. The
/// choice only has to keep the word nonzero with bit 1 clear. A read of this
/// register on a node with no peer would settle the real value.
const PORT_STATUS_LINK_DOWN: u64 = reg::PORT_STATUS::STATUS_BIT0::MASK;

/// Port status bit 1: the link is up.
///
/// Established from the PROM's and IRIX's code, as for the no-peer word: bit
/// 1 is the only bit the PROM tests, polling it for a second before it
/// reports the local link down and boots alone. Unknown: bits 0 and 2 on a
/// live link. They read 0 here, an approximation.
const PORT_STATUS_LINK_UP: u64 = reg::PORT_STATUS::LINK_UP::MASK;

/// Port error bit 52, "link reset in": a warm reset arrived and the reset
/// enable refused it. The bit's name is a software expectation; that a
/// refused reset latches it is inferred (it also raises the error line).
const LINK_RESET_IN: u64 = reg::PORT_ERRORS::LINK_RESET_IN::MASK;

use super::regs::ni as reg;
/// The end of the local routing table.
const ROUTES_END: u64 = reg::LOCAL_TABLE::END;
/// The next channel's low bit and the exit port, in a routing-table entry.
const ROUTE_MASK: u64 = field_masks!(reg::LOCAL_TABLE; NEXT_VCH_LSB, NEXT_EXIT_PORT);
const ERRORS_MASK: u64 = field_masks!(
    reg::PORT_ERRORS; LINK_RESET_IN, INTERNAL_LONG, INTERNAL_SHORT, EXTERNAL_BAD_HEADER, EXTERNAL_LONG, EXTERNAL_SHORT,
    FIFO_OVERFLOW, TAIL_TIMEOUT, RETRY_COUNT, CB_ERROR_COUNT, SN_ERROR_COUNT
);
const SIDEBAND_MASK: u64 = field_masks!(
    reg::PORT_SIDEBAND; VCH_ACTIVE, SEND_ERROR, CB_STATUS, SN_STATUS, SQUASH, BAD_MESSAGE, RCV_ERROR, BAD_PREXSEL,
    BAD_DEST, SIDEBAND
);

#[derive(Clone)]
struct Regs {
    control: [u64; 8],
    routes: [u64; reg::LOCAL_TABLE::COUNT],
    global: u64,
    errors: u64,
    sideband: u64,
}
impl Default for Regs {
    fn default() -> Self {
        Self { control: [0; 8], routes: [0; reg::LOCAL_TABLE::COUNT], global: 0, errors: 0, sideband: 0 }
    }
}
impl Regs {
    fn error_level(&self) -> bool {
        // The counters in bits 23:0 raise no line.
        self.errors
            & field_masks!(
                reg::PORT_ERRORS; LINK_RESET_IN, INTERNAL_LONG, INTERNAL_SHORT, EXTERNAL_BAD_HEADER, EXTERNAL_LONG,
                EXTERNAL_SHORT, FIFO_OVERFLOW, TAIL_TIMEOUT
            )
            != 0
    }
}

pub(super) struct Ni {
    regs: Mutex<Regs>,
}

impl Ni {
    pub(super) fn new() -> Self {
        Self { regs: Mutex::new(Regs::default()) }
    }
    pub(super) fn locked(&self) -> bool {
        self.regs.is_locked()
    }
    fn mask(a: u64) -> u64 {
        match a {
            reg::RESET_ENABLE::OFFSET => reg::RESET_ENABLE::RESET_OK::MASK,
            reg::DIAG_PARMS::OFFSET => field_masks!(reg::DIAG_PARMS; SEND_ERR_OFF, PORT_DISABLE, SEND_DATA_ERROR),
            reg::CHANNEL_CONTROL::OFFSET => field_masks!(
                reg::CHANNEL_CONTROL; SEND_VCH_SEL, VCH_VECTOR_REPLY, VCH_NINE_REPLY, VCH_TWO_REPLY, VCH_ONE_REPLY,
                VCH_VECTOR_REQUEST, VCH_NINE_REQUEST, VCH_TWO_REQUEST, VCH_ONE_REQUEST
            ),
            reg::PORT_PARMS::OFFSET => field_masks!(
                reg::PORT_PARMS; VCH_ERR_ENABLE, SQUASH_ERR_ENABLE, FIRST_ERR_ENABLE, D_AVAIL_SEL, MAX_RETRY, NULL_TIMEOUT,
                MAX_BURST
            ),
            reg::CHANNEL_AGE::OFFSET => field_masks!(reg::CHANNEL_AGE; REPLY_INJECT_AGE, REQUEST_INJECT_AGE),
            _ => 0,
        }
    }
    fn reflect(r: &Regs, levels: &Levels) -> Effect {
        if levels.set(Levels::NI_ERROR, r.error_level()) { Effect::Reflect } else { Effect::None }
    }
    /// A warm reset arrived over the link. Reset enable (bit 0 of control 0)
    /// lets it reset the node; otherwise the node runs on and latches "link
    /// reset in", which raises the error line. The local request is not gated.
    pub(super) fn link_reset(&self, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        if r.control[reg::RESET_ENABLE::OFFSET as usize / 8] & reg::RESET_ENABLE::RESET_OK::MASK != 0 {
            return Effect::Reset(NodeReset::Link);
        }
        r.errors |= LINK_RESET_IN;
        Self::reflect(&r, levels)
    }
    pub(super) fn power_on(&self, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        let sideband = r.sideband;
        *r = Regs::default();
        r.sideband = sideband;
        Self::reflect(&r, levels)
    }
    /// A doubleword read. Reading the error-clear word clears the errors, so
    /// the error level reaches `levels` under the block's lock.
    /// A connected port carries a live link: nothing takes one down.
    pub(super) fn read(&self, a: u64, levels: &Levels, link: bool) -> (u64, Effect) {
        let mut r = self.regs.lock();
        let v = if Self::mask(a) != 0 {
            r.control[a as usize / 8]
        } else if (reg::LOCAL_TABLE::OFFSET..ROUTES_END).contains(&a) {
            r.routes[(a - reg::LOCAL_TABLE::OFFSET) as usize / 8]
        } else {
            match a {
                reg::PORT_STATUS::OFFSET => if link { PORT_STATUS_LINK_UP } else { PORT_STATUS_LINK_DOWN },
                reg::PORT_ERRORS::OFFSET => r.errors,
                reg::PORT_HEADER_A::OFFSET | reg::PORT_HEADER_B::OFFSET => 0,
                reg::PORT_SIDEBAND::OFFSET => r.sideband,
                reg::PORT_ERROR_CLEAR::OFFSET => {
                    let v = r.errors;
                    r.errors = 0;
                    return (v, Self::reflect(&r, levels));
                }
                reg::GLOBAL_TABLE::OFFSET => r.global,
                _ => {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock NI unsupported read {a:#x}");
                    0
                }
            }
        };
        (v, Effect::None)
    }
    /// A doubleword store. The local-reset bit (bit 2 of the reset word)
    /// asks for the node's reset, and the port-reset bit (bit 1) for a warm
    /// reset over the link; the block's own registers are left to them.
    pub(super) fn write(&self, a: u64, v: u64, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        if Self::mask(a) != 0 {
            r.control[a as usize / 8] = v & Self::mask(a);
        } else if (reg::LOCAL_TABLE::OFFSET..ROUTES_END).contains(&a) {
            r.routes[(a - reg::LOCAL_TABLE::OFFSET) as usize / 8] = v & ROUTE_MASK;
        } else {
            match a {
                reg::PORT_RESET::OFFSET => {
                    if v & reg::PORT_RESET::PORT_RESET::MASK != 0 {
                        return Effect::LinkReset { and_local: v & reg::PORT_RESET::LOCAL_RESET::MASK != 0 };
                    }
                    if v & reg::PORT_RESET::LOCAL_RESET::MASK != 0 {
                        return Effect::Reset(NodeReset::Network);
                    }
                }
                reg::PORT_ERRORS::OFFSET => r.errors = v & ERRORS_MASK,
                reg::PORT_SIDEBAND::OFFSET => r.sideband = v & SIDEBAND_MASK,
                reg::GLOBAL_TABLE::OFFSET => r.global = v & ROUTE_MASK,
                _ => crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock NI unsupported or read-only write {a:#x} = {v:#x}"),
            }
        }
        Self::reflect(&r, levels)
    }
}

#[cfg(test)]
mod tests {
    // Cases ported from tests/qtest/sn1-bedrock-test.c in jamesbraid/qemu-sgi-origami.
    use super::super::regs::{pi, PI1_BASE};
    use super::super::{connected_pair, Rig};
    use super::*;
    use crate::traits::{BusDevice, BUS_ERR};

    fn error_level(rig: &Rig) -> bool {
        let levels = [0, PI1_BASE].map(|base| rig.r(base + pi::INT_PEND1::OFFSET) & (1 << 62) != 0);
        assert_eq!(levels[0], levels[1], "both interfaces reflect bit 62");
        levels[0]
    }
    fn resets(rig: &Rig) -> usize {
        rig.events.lock().try_iter().count()
    }

    /// The guest programs route fields and reads the error register to acknowledge faults.
    #[test]
    fn route_registers_store_fields_and_error_reads_clear_pending_errors() {
        let rig = Rig::new();
        for (a, mask) in [
            (reg::RESET_ENABLE::OFFSET, 1),
            (reg::DIAG_PARMS::OFFSET, 7),
            (reg::CHANNEL_CONTROL::OFFSET, 0x1ff),
            (reg::PORT_PARMS::OFFSET, 0x7efffffff),
            (reg::CHANNEL_AGE::OFFSET, 0xffff),
        ] {
            rig.w(reg::BASE + a, !0);
            assert_eq!(rig.r(reg::BASE + a), mask);
        }
        for a in [reg::LOCAL_TABLE::OFFSET, ROUTES_END - 8] {
            rig.w(reg::BASE + a, !0);
            assert_eq!(rig.r(reg::BASE + a), 31);
        }
        rig.w(reg::BASE + reg::GLOBAL_TABLE::OFFSET, !0);
        assert_eq!(rig.r(reg::BASE + reg::GLOBAL_TABLE::OFFSET), 31);
        rig.w(reg::BASE + reg::RESET_ENABLE::OFFSET, 0);
        rig.w(reg::BASE + reg::PORT_RESET::OFFSET, 4);
        assert_eq!(resets(&rig), 1);
        rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, 0xffffff);
        assert!(!error_level(&rig));
        rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, 1 << 24);
        assert!(error_level(&rig));
        rig.r(reg::BASE + reg::PORT_ERROR_CLEAR::OFFSET);
        assert!(!error_level(&rig));
        rig.w(reg::BASE + reg::PORT_SIDEBAND::OFFSET, !0);
        let _ = rig.chip.ni.power_on(&rig.chip.levels);
        assert_eq!(rig.r(reg::BASE + reg::PORT_SIDEBAND::OFFSET), 0xfffff);
        assert_eq!(rig.r(reg::BASE + reg::PORT_PARMS::OFFSET), 0);
    }

    /// The PROM's one power-on pass writes a sign-extended port-parameter constant and zero
    /// to the error and sideband words; each must read back through its defined bits only.
    #[test]
    fn power_on_programming_pass_reads_back_through_the_defined_bits() {
        let rig = Rig::new();
        rig.w(reg::BASE + reg::RESET_ENABLE::OFFSET, 1);
        rig.w(reg::BASE + reg::DIAG_PARMS::OFFSET, 0);
        rig.w(reg::BASE + reg::CHANNEL_CONTROL::OFFSET, 0x166);
        rig.w(reg::BASE + reg::PORT_PARMS::OFFSET, 0xffff_ffff_a3ff_1990);
        rig.w(reg::BASE + reg::CHANNEL_AGE::OFFSET, 0xc800);
        rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, 0);
        rig.r(reg::BASE + reg::PORT_ERROR_CLEAR::OFFSET);
        rig.w(reg::BASE + reg::PORT_SIDEBAND::OFFSET, 0);
        for (a, want) in [
            (reg::RESET_ENABLE::OFFSET, 1),
            (reg::DIAG_PARMS::OFFSET, 0),
            (reg::CHANNEL_CONTROL::OFFSET, 0x166),
            (reg::PORT_PARMS::OFFSET, 0x7_a3ff_1990),
            (reg::CHANNEL_AGE::OFFSET, 0xc800),
            (reg::PORT_ERRORS::OFFSET, 0),
            (reg::PORT_SIDEBAND::OFFSET, 0),
            (reg::PORT_ERROR_CLEAR::OFFSET, 0),
        ] {
            assert_eq!(rig.r(reg::BASE + a), want, "register {a:#x}");
        }
    }

    /// Only the local-reset bit of the reset word reaches the node; the other bits must
    /// not restart it, and the interface's own registers are left to the board's reset.
    #[test]
    fn only_the_local_reset_bit_requests_a_node_reset() {
        let rig = Rig::new();
        for enable in 0..2 {
            rig.w(reg::BASE + reg::RESET_ENABLE::OFFSET, enable);
            for command in 0..8u64 {
                rig.w(reg::BASE + reg::PORT_RESET::OFFSET, command);
                assert_eq!(resets(&rig), usize::from(command & 4 != 0), "command {command:#x}");
            }
        }
        // The request does not clear this block.
        assert_eq!(rig.r(reg::BASE + reg::RESET_ENABLE::OFFSET), 1);
    }

    /// A latched port error raises the interrupt line through its interrupting bits only;
    /// drivers rely on the line dropping when the status is rewritten or cleared.
    #[test]
    fn stored_error_status_drives_the_error_line_through_the_interrupting_bits() {
        let rig = Rig::new();
        for (value, high) in [(1u64 << 30, true), (0xff, false), (1 << 52, true), (0, false)] {
            rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, value);
            assert_eq!(error_level(&rig), high, "status {value:#x}");
        }
        // Bit 53 is beyond the register and must not be retained.
        rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, 1 << 53);
        assert_eq!(rig.r(reg::BASE + reg::PORT_ERRORS::OFFSET), 0);
    }

    /// Reset values of the programmable words are unknown on hardware, so the model clears them;
    /// a guest that re-runs its setup after a reset must see zeroes, not stale routes.
    #[test]
    fn reset_clears_every_programmable_register_and_misaligned_accesses_fault() {
        let rig = Rig::new();
        let offsets = [
            reg::RESET_ENABLE::OFFSET, reg::DIAG_PARMS::OFFSET, reg::CHANNEL_CONTROL::OFFSET, reg::PORT_PARMS::OFFSET, reg::CHANNEL_AGE::OFFSET,
            reg::LOCAL_TABLE::OFFSET, ROUTES_END - 8, reg::GLOBAL_TABLE::OFFSET,
        ];
        for a in offsets {
            rig.w(reg::BASE + a, !0);
        }
        // A misaligned write must not land in the neighbouring register.
        assert_eq!(rig.regs.write64(reg::BASE + reg::CHANNEL_AGE::OFFSET + 1, 0), BUS_ERR);
        assert_eq!(rig.regs.read64(reg::BASE + reg::CHANNEL_AGE::OFFSET + 1).status, BUS_ERR);
        assert_eq!(rig.r(reg::BASE + reg::CHANNEL_AGE::OFFSET), 0xffff);
        let _ = rig.chip.ni.power_on(&rig.chip.levels);
        for a in offsets {
            assert_eq!(rig.r(reg::BASE + a), 0, "register {a:#x}");
        }
    }

    /// A one-node port with no peer reads status bits 1:0 = 1: link down, bit 0
    /// set. Reading 0 there makes the guest's bus-error handler print a
    /// link-reset notice. Bit 18 stays clear (a one-node machine) and a
    /// write does not change the value.
    #[test]
    fn a_port_with_no_peer_reads_link_down_and_ignores_writes() {
        let rig = Rig::new();
        assert_eq!(rig.r(reg::BASE + reg::PORT_STATUS::OFFSET), 1);
        rig.w(reg::BASE + reg::PORT_STATUS::OFFSET, !0);
        assert_eq!(rig.r(reg::BASE + reg::PORT_STATUS::OFFSET), 1);
        let _ = rig.chip.ni.power_on(&rig.chip.levels);
        assert_eq!(rig.r(reg::BASE + reg::PORT_STATUS::OFFSET), 1);
    }

    /// The firmware polls bit 1 of the port status: it follows the link
    /// from reset with nothing written, a port never connected reads bit 0
    /// alone, and a write leaves the status alone.
    #[test]
    fn port_status_follows_the_link() {
        let (a, b) = connected_pair();
        assert_eq!(a.r(reg::BASE + reg::PORT_STATUS::OFFSET), 2);
        assert_eq!(b.r(reg::BASE + reg::PORT_STATUS::OFFSET), 2);
        a.w(reg::BASE + reg::PORT_STATUS::OFFSET, !0);
        assert_eq!(a.r(reg::BASE + reg::PORT_STATUS::OFFSET), 2);
        let _ = a.chip.ni.power_on(&a.chip.levels);
        assert_eq!(a.r(reg::BASE + reg::PORT_STATUS::OFFSET), 2);
        assert_eq!(Rig::new().r(reg::BASE + reg::PORT_STATUS::OFFSET), 1);
    }

    /// Bit 1 of the reset word sends a warm reset over a live link and leaves
    /// the sender alone; bit 2 resets the sender; the far end takes a warm
    /// reset only with its reset enable set.
    #[test]
    fn a_port_reset_arrives_gated_by_the_far_enable() {
        let (a, b) = connected_pair();
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 2);
        assert_eq!((resets(&a), resets(&b)), (0, 0));
        b.w(reg::BASE + reg::RESET_ENABLE::OFFSET, 1);
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 2);
        assert_eq!((resets(&a), resets(&b)), (0, 1));
        // The firmware's own write: port and local reset together.
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 6);
        assert_eq!((resets(&a), resets(&b)), (1, 1));
        // The local reset bit is not gated by the enable.
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 4);
        assert_eq!((resets(&a), resets(&b)), (1, 0));
        // Bit 0, the link reset, reaches nothing.
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 1);
        assert_eq!((resets(&a), resets(&b)), (0, 0));
    }

    /// A warm reset the far end refuses runs on and latches "link reset in",
    /// which raises its network error line; nothing else is latched.
    #[test]
    fn a_refused_port_reset_latches_link_reset_in() {
        let (a, b) = connected_pair();
        assert!(!error_level(&b));
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 2);
        assert!(error_level(&b));
        assert_eq!(b.r(reg::BASE + reg::PORT_ERRORS::OFFSET), 1 << 52);
        assert!(!error_level(&a));
        b.r(reg::BASE + reg::PORT_ERROR_CLEAR::OFFSET);
        assert!(!error_level(&b));
        // The peer that takes the reset latches nothing.
        b.w(reg::BASE + reg::RESET_ENABLE::OFFSET, 1);
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 2);
        assert!(!error_level(&b));
        assert_eq!(b.r(reg::BASE + reg::PORT_ERRORS::OFFSET), 0);
    }

    impl Ni {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
    }
}
