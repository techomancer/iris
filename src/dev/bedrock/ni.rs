//! Bedrock's network interface (NI) exposes node routing, sideband and error
//! registers. The Fuel runs as one node without a NUMAlink peer, but the
//! guest still reads and programs these local control words during machine
//! setup. A chip joined to another by [`super::Bedrock::cable`] reads its
//! port status from the cable and can send its peer a warm reset.
//!
//! # What is approximated here
//!
//! Routing and sideband writes are retained; the link protocol and packet
//! delivery are not simulated, and the link comes up with the cable, with no
//! training sequence. Unspecified power-on fields start at zero, and the
//! sideband word survives reset as the current compatibility behavior.
//!
//! Ported from hw/misc/sgi-bedrock-ni.c and hw/misc/sgi-bedrock-ni-regs.c
//! in jamesbraid/qemu-sgi-origami.

use super::pi::Levels;
use super::{Effect, NodeReset};
use parking_lot::Mutex;

/// Length of the block-relative register aperture.
pub(super) const SIZE: u64 = 0x2008;

/// Port status of a node with no cable.
///
/// Established from the PROM's and IRIX's code: only bit 1 is read, as
/// link-up, and nothing reads bit 0 alone. With no peer the PROM polls bit 1,
/// prints "Local hub NUMAlink is down", records code 0x9c and carries on, so
/// bit 1 is clear here. Real Fuels' consoles show the same line
/// (comp.sys.sgi.admin, Atro Tossavainen 2005-04-13, Tristram Scott
/// 2007-07-18).
///
/// Unknown: bits 0 and 2 and the word as a whole on a real uncabled Fuel.
/// Bit 0 is our choice, kept because the IRIX 6.5.30 bus-error handler prints
/// "Detected a link reset due to other partition reset" on every faulting
/// widget probe when this word and the reset-enable register are both zero,
/// and the 2005 Fuel's IRIX took a PIO read error with no such line. The
/// choice only has to keep the word nonzero with bit 1 clear. A read of this
/// register on an uncabled node would settle the real value.
const PORT_STATUS_UNCABLED: u64 = 1;

/// Port status bit 1: the link is up.
///
/// Established from the PROM's and IRIX's code, as for the uncabled word: bit
/// 1 is the only bit the PROM tests, polling it for a second before it
/// reports the local link down and boots alone. Unknown: bits 0 and 2 on a
/// live link. They read 0 here, an approximation.
const PORT_STATUS_LINK_UP: u64 = 2;

/// The state of the cable on the port, as the chip reads it from the cable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Link {
    /// Nothing is cabled to the port.
    Uncabled,
    /// A cable is fitted and carries a live link.
    Up,
    /// A cable is fitted and marked down: the word reads 0, which is the
    /// uncabled word's bit 0 and bit 1 both clear. Inferred from the PROM's
    /// test of bit 1 alone; unknown on hardware.
    Down,
}

/// Port error bit 52, "link reset in": a warm reset arrived and the reset
/// enable refused it. The bit's name is a software expectation; that a
/// refused reset latches it is inferred (it also raises the error line).
const LINK_RESET_IN: u64 = 1 << 52;

pub mod reg {
    /// Port status, hub-relative `0x680000` (the first word of the block).
    pub const PORT_STATUS: u64 = 0x0;
    pub const LOCAL_RESET: u64 = 0x8;
    pub const CONTROL_0: u64 = 0x10;
    pub const CONTROL_1: u64 = 0x18;
    pub const CONTROL_2: u64 = 0x20;
    pub const CONTROL_3: u64 = 0x30;
    pub const CONTROL_4: u64 = 0x38;
    pub const ERROR_STATUS: u64 = 0x100;
    pub const ERROR_STATUS_UPPER: u64 = 0x108;
    pub const ERROR_STATUS_LOWER: u64 = 0x110;
    pub const SIDE_BAND: u64 = 0x118;
    pub const ERROR_CLEAR: u64 = 0x120;
    pub const ROUTES: u64 = 0x1000;
    pub const ROUTES_END: u64 = 0x1400;
    pub const GLOBAL_CONTROL: u64 = 0x2000;
}

#[derive(Clone)]
struct Regs {
    control: [u64; 8],
    routes: [u64; 128],
    global: u64,
    errors: u64,
    sideband: u64,
}
impl Default for Regs {
    fn default() -> Self {
        Self { control: [0; 8], routes: [0; 128], global: 0, errors: 0, sideband: 0 }
    }
}
impl Regs {
    fn error_level(&self) -> bool {
        self.errors & 0x1fffffff000000 != 0
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
            reg::CONTROL_0 => 1,
            reg::CONTROL_1 => 7,
            reg::CONTROL_2 => 0x1ff,
            reg::CONTROL_3 => 0x7efffffff,
            reg::CONTROL_4 => 0xffff,
            _ => 0,
        }
    }
    fn reflect(r: &Regs, levels: &Levels) -> Effect {
        if levels.set(Levels::NI_ERROR, r.error_level()) { Effect::Reflect } else { Effect::None }
    }
    /// A warm reset arrived over the cable. Reset enable (bit 0 of control 0)
    /// lets it reset the node; otherwise the node runs on and latches "link
    /// reset in", which raises the error line. The local request is not gated.
    pub(super) fn link_reset(&self, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        if r.control[reg::CONTROL_0 as usize / 8] & 1 != 0 {
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
    pub(super) fn read(&self, a: u64, levels: &Levels, link: Link) -> (u64, Effect) {
        let mut r = self.regs.lock();
        let v = if Self::mask(a) != 0 {
            r.control[a as usize / 8]
        } else if (reg::ROUTES..reg::ROUTES_END).contains(&a) {
            r.routes[(a - reg::ROUTES) as usize / 8]
        } else {
            match a {
                reg::PORT_STATUS => match link {
                    Link::Uncabled => PORT_STATUS_UNCABLED,
                    Link::Up => PORT_STATUS_LINK_UP,
                    Link::Down => 0,
                },
                reg::ERROR_STATUS => r.errors,
                reg::ERROR_STATUS_UPPER | reg::ERROR_STATUS_LOWER => 0,
                reg::SIDE_BAND => r.sideband,
                reg::ERROR_CLEAR => {
                    let v = r.errors;
                    r.errors = 0;
                    return (v, Self::reflect(&r, levels));
                }
                reg::GLOBAL_CONTROL => r.global,
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
    /// reset over the cable; the block's own registers are left to them.
    pub(super) fn write(&self, a: u64, v: u64, levels: &Levels) -> Effect {
        let mut r = self.regs.lock();
        if Self::mask(a) != 0 {
            r.control[a as usize / 8] = v & Self::mask(a);
        } else if (reg::ROUTES..reg::ROUTES_END).contains(&a) {
            r.routes[(a - reg::ROUTES) as usize / 8] = v & 31;
        } else {
            match a {
                reg::LOCAL_RESET => {
                    if v & 2 != 0 {
                        return Effect::LinkReset { and_local: v & 4 != 0 };
                    }
                    if v & 4 != 0 {
                        return Effect::Reset(NodeReset::Network);
                    }
                }
                reg::ERROR_STATUS => r.errors = v & ((1 << 53) - 1),
                reg::SIDE_BAND => r.sideband = v & 0xfffff,
                reg::GLOBAL_CONTROL => r.global = v & 31,
                _ => crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock NI unsupported or read-only write {a:#x} = {v:#x}"),
            }
        }
        Self::reflect(&r, levels)
    }
}

#[cfg(test)]
mod tests {
    // Cases ported from tests/qtest/sn1-bedrock-test.c in jamesbraid/qemu-sgi-origami.
    use super::super::reg::{pi, NI_BASE, PI1_BASE};
    use super::super::{cabled_pair, Rig};
    use super::*;
    use crate::traits::{BusDevice, BUS_ERR};

    fn error_level(rig: &Rig) -> bool {
        let levels = [0, PI1_BASE].map(|base| rig.r(base + pi::PENDING_STATUS_B) & (1 << 62) != 0);
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
            (reg::CONTROL_0, 1),
            (reg::CONTROL_1, 7),
            (reg::CONTROL_2, 0x1ff),
            (reg::CONTROL_3, 0x7efffffff),
            (reg::CONTROL_4, 0xffff),
        ] {
            rig.w(NI_BASE + a, !0);
            assert_eq!(rig.r(NI_BASE + a), mask);
        }
        for a in [reg::ROUTES, reg::ROUTES_END - 8] {
            rig.w(NI_BASE + a, !0);
            assert_eq!(rig.r(NI_BASE + a), 31);
        }
        rig.w(NI_BASE + reg::GLOBAL_CONTROL, !0);
        assert_eq!(rig.r(NI_BASE + reg::GLOBAL_CONTROL), 31);
        rig.w(NI_BASE + reg::CONTROL_0, 0);
        rig.w(NI_BASE + reg::LOCAL_RESET, 4);
        assert_eq!(resets(&rig), 1);
        rig.w(NI_BASE + reg::ERROR_STATUS, 0xffffff);
        assert!(!error_level(&rig));
        rig.w(NI_BASE + reg::ERROR_STATUS, 1 << 24);
        assert!(error_level(&rig));
        rig.r(NI_BASE + reg::ERROR_CLEAR);
        assert!(!error_level(&rig));
        rig.w(NI_BASE + reg::SIDE_BAND, !0);
        let _ = rig.chip.ni.power_on(&rig.chip.levels);
        assert_eq!(rig.r(NI_BASE + reg::SIDE_BAND), 0xfffff);
        assert_eq!(rig.r(NI_BASE + reg::CONTROL_3), 0);
    }

    /// The PROM's one power-on pass writes a sign-extended port-parameter constant and zero
    /// to the error and sideband words; each must read back through its defined bits only.
    #[test]
    fn power_on_programming_pass_reads_back_through_the_defined_bits() {
        let rig = Rig::new();
        rig.w(NI_BASE + reg::CONTROL_0, 1);
        rig.w(NI_BASE + reg::CONTROL_1, 0);
        rig.w(NI_BASE + reg::CONTROL_2, 0x166);
        rig.w(NI_BASE + reg::CONTROL_3, 0xffff_ffff_a3ff_1990);
        rig.w(NI_BASE + reg::CONTROL_4, 0xc800);
        rig.w(NI_BASE + reg::ERROR_STATUS, 0);
        rig.r(NI_BASE + reg::ERROR_CLEAR);
        rig.w(NI_BASE + reg::SIDE_BAND, 0);
        for (a, want) in [
            (reg::CONTROL_0, 1),
            (reg::CONTROL_1, 0),
            (reg::CONTROL_2, 0x166),
            (reg::CONTROL_3, 0x7_a3ff_1990),
            (reg::CONTROL_4, 0xc800),
            (reg::ERROR_STATUS, 0),
            (reg::SIDE_BAND, 0),
            (reg::ERROR_CLEAR, 0),
        ] {
            assert_eq!(rig.r(NI_BASE + a), want, "register {a:#x}");
        }
    }

    /// Only the local-reset bit of the reset word reaches the node; the other bits must
    /// not restart it, and the interface's own registers are left to the board's reset.
    #[test]
    fn only_the_local_reset_bit_requests_a_node_reset() {
        let rig = Rig::new();
        for enable in 0..2 {
            rig.w(NI_BASE + reg::CONTROL_0, enable);
            for command in 0..8u64 {
                rig.w(NI_BASE + reg::LOCAL_RESET, command);
                assert_eq!(resets(&rig), usize::from(command & 4 != 0), "command {command:#x}");
            }
        }
        // The request does not clear this block.
        assert_eq!(rig.r(NI_BASE + reg::CONTROL_0), 1);
    }

    /// A latched port error raises the interrupt line through its interrupting bits only;
    /// drivers rely on the line dropping when the status is rewritten or cleared.
    #[test]
    fn stored_error_status_drives_the_error_line_through_the_interrupting_bits() {
        let rig = Rig::new();
        for (value, high) in [(1u64 << 30, true), (0xff, false), (1 << 52, true), (0, false)] {
            rig.w(NI_BASE + reg::ERROR_STATUS, value);
            assert_eq!(error_level(&rig), high, "status {value:#x}");
        }
        // Bit 53 is beyond the register and must not be retained.
        rig.w(NI_BASE + reg::ERROR_STATUS, 1 << 53);
        assert_eq!(rig.r(NI_BASE + reg::ERROR_STATUS), 0);
    }

    /// Reset values of the programmable words are unknown on hardware, so the model clears them;
    /// a guest that re-runs its setup after a reset must see zeroes, not stale routes.
    #[test]
    fn reset_clears_every_programmable_register_and_misaligned_accesses_fault() {
        let rig = Rig::new();
        let offsets = [
            reg::CONTROL_0, reg::CONTROL_1, reg::CONTROL_2, reg::CONTROL_3, reg::CONTROL_4,
            reg::ROUTES, reg::ROUTES_END - 8, reg::GLOBAL_CONTROL,
        ];
        for a in offsets {
            rig.w(NI_BASE + a, !0);
        }
        // A misaligned write must not land in the neighbouring register.
        assert_eq!(rig.regs.write64(NI_BASE + reg::CONTROL_4 + 1, 0), BUS_ERR);
        assert_eq!(rig.regs.read64(NI_BASE + reg::CONTROL_4 + 1).status, BUS_ERR);
        assert_eq!(rig.r(NI_BASE + reg::CONTROL_4), 0xffff);
        let _ = rig.chip.ni.power_on(&rig.chip.levels);
        for a in offsets {
            assert_eq!(rig.r(NI_BASE + a), 0, "register {a:#x}");
        }
    }

    /// An uncabled one-node port reads status bits 1:0 = 1: link down, bit 0
    /// set. Reading 0 there makes the guest's bus-error handler print a
    /// link-reset notice. Bit 18 stays clear (a one-node machine) and a
    /// write does not change the value.
    #[test]
    fn uncabled_port_status_reads_one_and_ignores_writes() {
        let rig = Rig::new();
        assert_eq!(rig.r(NI_BASE + reg::PORT_STATUS), 1);
        rig.w(NI_BASE + reg::PORT_STATUS, !0);
        assert_eq!(rig.r(NI_BASE + reg::PORT_STATUS), 1);
        let _ = rig.chip.ni.power_on(&rig.chip.levels);
        assert_eq!(rig.r(NI_BASE + reg::PORT_STATUS), 1);
    }

    /// The firmware polls bit 1 of the port status: it follows the cable
    /// from reset with nothing written, a port never cabled reads bit 0
    /// alone, and a write leaves the status alone.
    #[test]
    fn port_status_follows_the_cable() {
        let (a, b) = cabled_pair();
        assert_eq!(a.r(NI_BASE + reg::PORT_STATUS), 2);
        assert_eq!(b.r(NI_BASE + reg::PORT_STATUS), 2);
        a.w(NI_BASE + reg::PORT_STATUS, !0);
        assert_eq!(a.r(NI_BASE + reg::PORT_STATUS), 2);
        a.chip.set_link(false);
        assert_eq!(a.r(NI_BASE + reg::PORT_STATUS), 0);
        assert_eq!(b.r(NI_BASE + reg::PORT_STATUS), 0, "both ends see the link down");
        b.chip.set_link(true);
        let _ = a.chip.ni.power_on(&a.chip.levels);
        assert_eq!(a.r(NI_BASE + reg::PORT_STATUS), 2);
        assert_eq!(Rig::new().r(NI_BASE + reg::PORT_STATUS), 1);
    }

    /// Bit 1 of the reset word sends a warm reset over a live link and leaves
    /// the sender alone; bit 2 resets the sender; the far end takes a warm
    /// reset only with its reset enable set.
    #[test]
    fn a_port_reset_arrives_gated_by_the_far_enable() {
        let (a, b) = cabled_pair();
        a.w(NI_BASE + reg::LOCAL_RESET, 2);
        assert_eq!((resets(&a), resets(&b)), (0, 0));
        b.w(NI_BASE + reg::CONTROL_0, 1);
        a.w(NI_BASE + reg::LOCAL_RESET, 2);
        assert_eq!((resets(&a), resets(&b)), (0, 1));
        // The firmware's own write: port and local reset together.
        a.w(NI_BASE + reg::LOCAL_RESET, 6);
        assert_eq!((resets(&a), resets(&b)), (1, 1));
        // The local reset bit is not gated by the enable.
        a.w(NI_BASE + reg::LOCAL_RESET, 4);
        assert_eq!((resets(&a), resets(&b)), (1, 0));
        a.chip.set_link(false);
        a.w(NI_BASE + reg::LOCAL_RESET, 2);
        assert_eq!((resets(&a), resets(&b)), (0, 0), "a downed link carries nothing");
        // Bit 0, the link reset, reaches nothing.
        a.chip.set_link(true);
        a.w(NI_BASE + reg::LOCAL_RESET, 1);
        assert_eq!((resets(&a), resets(&b)), (0, 0));
    }

    /// A warm reset the far end refuses runs on and latches "link reset in",
    /// which raises its network error line; nothing else is latched.
    #[test]
    fn a_refused_port_reset_latches_link_reset_in() {
        let (a, b) = cabled_pair();
        assert!(!error_level(&b));
        a.w(NI_BASE + reg::LOCAL_RESET, 2);
        assert!(error_level(&b));
        assert_eq!(b.r(NI_BASE + reg::ERROR_STATUS), 1 << 52);
        assert!(!error_level(&a));
        b.r(NI_BASE + reg::ERROR_CLEAR);
        assert!(!error_level(&b));
        // The peer that takes the reset latches nothing.
        b.w(NI_BASE + reg::CONTROL_0, 1);
        a.w(NI_BASE + reg::LOCAL_RESET, 2);
        assert!(!error_level(&b));
        assert_eq!(b.r(NI_BASE + reg::ERROR_STATUS), 0);
    }

    impl Ni {
        /// Hold the block's lock, for tests of the paths that must not take it.
        pub(in crate::dev::bedrock) fn hold(&self) -> impl Sized + '_ {
            self.regs.lock()
        }
    }
}
