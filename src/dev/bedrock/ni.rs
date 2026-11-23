//! Bedrock's network interface (NI): node routing, sideband and error registers. A chip whose port is connected
//! ([`super::Bedrock::connect_peer`]) reads port status from the link and can send its peer a warm reset.
//!
//! # What is approximated here
//!
//! Routing and sideband writes are retained; the link protocol and packet delivery are not simulated and the link
//! comes up when connected, with no training. Unspecified power-on fields start at zero; the sideband word survives reset.
//!
//! Ported from hw/misc/sgi-bedrock-ni.c and hw/misc/sgi-bedrock-ni-regs.c
//! in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::{Effect, NodeReset};
use parking_lot::Mutex;

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = reg::GLOBAL_TABLE::OFFSET + reg::GLOBAL_TABLE::WIDTH as u64 / 8;

/// Port status of a node with no peer: bit 1 (link up) clear, as the PROM and IRIX read only that
/// bit (established; the PROM prints "Local hub NUMAlink is down"). Bit 0 is our choice, unknown on
/// real hardware: it keeps IRIX 6.5.30's bus-error handler from printing a link-reset notice on
/// every widget probe. A read on a real peerless node would settle it.
const PORT_STATUS_LINK_DOWN: u64 = reg::PORT_STATUS::STATUS_BIT0::MASK;

/// Port status bit 1: the link is up; the only bit the PROM tests (established). Bits 0 and 2 read
/// 0, an approximation.
const PORT_STATUS_LINK_UP: u64 = reg::PORT_STATUS::LINK_UP::MASK;

/// Port error bit 52, "link reset in": a refused warm reset latches it and raises the error line
/// (inferred).
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
    /// A warm reset arrived over the link: reset enable (bit 0 of control 0) lets it reset the
    /// node, otherwise "link reset in" latches and raises the error line. The local request is not
    /// gated.
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
    /// A doubleword read; reading the error-clear word clears the errors. A connected port's link
    /// never goes down.
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
    /// A doubleword store; bit 2 of the reset word resets the node, bit 1 sends a warm reset over
    /// the link.
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
    use super::super::{connected_pair, Rig};
    use super::*;
    use crate::traits::{BusDevice, BUS_ERR};

    fn resets(rig: &Rig) -> usize {
        std::iter::from_fn(|| rig.events.lock().try_recv().ok()).count()
    }

    /// The guest programs route fields and reads the error register to acknowledge faults.
    #[test]
    fn route_registers_store_fields_and_error_reads_clear_pending_errors() {
        let rig = Rig::new();
        // All ones read back through the defined bits, then the PROM's sign-extended power-on pass.
        for (a, value, want) in [
            (reg::RESET_ENABLE::OFFSET, !0, 1),
            (reg::DIAG_PARMS::OFFSET, !0, 7),
            (reg::CHANNEL_CONTROL::OFFSET, !0, 0x1ff),
            (reg::PORT_PARMS::OFFSET, !0, 0x7efffffff),
            (reg::CHANNEL_AGE::OFFSET, !0, 0xffff),
            (reg::RESET_ENABLE::OFFSET, 1, 1),
            (reg::DIAG_PARMS::OFFSET, 0, 0),
            (reg::CHANNEL_CONTROL::OFFSET, 0x166, 0x166),
            (reg::PORT_PARMS::OFFSET, 0xffff_ffff_a3ff_1990, 0x7_a3ff_1990),
            (reg::CHANNEL_AGE::OFFSET, 0xc800, 0xc800),
        ] {
            rig.w(reg::BASE + a, value);
            assert_eq!(rig.r(reg::BASE + a), want, "register {a:#x} written {value:#x}");
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
        assert!(!rig.reflected(62));
        rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, 1 << 24);
        assert!(rig.reflected(62));
        rig.r(reg::BASE + reg::PORT_ERROR_CLEAR::OFFSET);
        assert!(!rig.reflected(62));
        rig.w(reg::BASE + reg::PORT_SIDEBAND::OFFSET, !0);
        let _ = rig.chip.ni.power_on(&rig.chip.levels);
        assert_eq!(rig.r(reg::BASE + reg::PORT_SIDEBAND::OFFSET), 0xfffff);
        assert_eq!(rig.r(reg::BASE + reg::PORT_PARMS::OFFSET), 0);
        // The pass writes zero to the error and sideband words; they read back zero.
        rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, 0);
        rig.w(reg::BASE + reg::PORT_SIDEBAND::OFFSET, 0);
        for a in [reg::PORT_ERRORS::OFFSET, reg::PORT_SIDEBAND::OFFSET, reg::PORT_ERROR_CLEAR::OFFSET] {
            assert_eq!(rig.r(reg::BASE + a), 0, "register {a:#x}");
        }
    }

    /// Only the local-reset bit of the reset word reaches the node; the block's own registers are
    /// left alone.
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

    /// A latched port error raises the interrupt line through its interrupting bits only, and drops
    /// when cleared.
    #[test]
    fn stored_error_status_drives_the_error_line_through_the_interrupting_bits() {
        let rig = Rig::new();
        for (value, high) in [(1u64 << 30, true), (0xff, false), (1 << 52, true), (0, false)] {
            rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, value);
            assert_eq!(rig.reflected(62), high, "status {value:#x}");
        }
        // Bit 53 is beyond the register and must not be retained.
        rig.w(reg::BASE + reg::PORT_ERRORS::OFFSET, 1 << 53);
        assert_eq!(rig.r(reg::BASE + reg::PORT_ERRORS::OFFSET), 0);
    }

    /// The model clears the programmable words on reset (hardware values unknown), so a re-run
    /// setup sees zeroes.
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

    /// Port status bit 1 follows the link; a peerless port reads 1 (bit 0 set, or the bus-error
    /// handler prints a notice).
    #[test]
    fn port_status_follows_the_link() {
        let (a, b) = connected_pair();
        for (rig, status) in [(&Rig::new(), 1), (&a, 2), (&b, 2)] {
            assert_eq!(rig.r(reg::BASE + reg::PORT_STATUS::OFFSET), status);
            rig.w(reg::BASE + reg::PORT_STATUS::OFFSET, !0);
            let _ = rig.chip.ni.power_on(&rig.chip.levels);
            assert_eq!(rig.r(reg::BASE + reg::PORT_STATUS::OFFSET), status);
        }
    }

    /// Bit 1 of the reset word warm-resets a live link's far end only with its reset enable set;
    /// bit 2 resets the sender.
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

    /// A warm reset the far end refuses latches "link reset in" and raises its error line; nothing
    /// else.
    #[test]
    fn a_refused_port_reset_latches_link_reset_in() {
        let (a, b) = connected_pair();
        assert!(!b.reflected(62));
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 2);
        assert!(b.reflected(62));
        assert_eq!(b.r(reg::BASE + reg::PORT_ERRORS::OFFSET), 1 << 52);
        assert!(!a.reflected(62));
        b.r(reg::BASE + reg::PORT_ERROR_CLEAR::OFFSET);
        assert!(!b.reflected(62));
        // The peer that takes the reset latches nothing.
        b.w(reg::BASE + reg::RESET_ENABLE::OFFSET, 1);
        a.w(reg::BASE + reg::PORT_RESET::OFFSET, 2);
        assert!(!b.reflected(62));
        assert_eq!(b.r(reg::BASE + reg::PORT_ERRORS::OFFSET), 0);
    }

    impl Ni {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
    }
}
