//! The BCM570x's integrated copper PHY, reached through the MAC's MI_COMM
//! register.
//!
//! The PHY has no bus address of its own: software writes a register number
//! into MI_COMM and reads a word back. The contract is the ordinary MII/GMII
//! one the public brgphy driver drives. Link is modelled without a cable: out
//! of reset the PHY negotiates, and autonegotiation completes a fixed settle
//! later, which restarts when the guest restarts it. Reported link is also
//! gated on the backend's cable state.
//!
//! The identity decodes to Broadcom's organisational identifier and the
//! BCM5701 model, the pair brgphy matches to bind.
//!
//! Ported from hw/net/bcm570x_phy.c in jamesbraid/qemu-sgi-origami.

use std::time::{Duration, Instant};

pub mod reg {
    pub const BMCR: u8 = 0x00;
    pub const BMSR: u8 = 0x01;
    pub const IDR1: u8 = 0x02;
    pub const IDR2: u8 = 0x03;
    pub const ANAR: u8 = 0x04;
    pub const ANLPAR: u8 = 0x05;
    pub const GTCR: u8 = 0x09;
    pub const GTSR: u8 = 0x0a;
    pub const EXTSR: u8 = 0x0f;
    /// Broadcom auxiliary status summary.
    pub const AUXSTS: u8 = 0x19;
}

pub const IDR1_VALUE: u16 = 0x0020;
pub const IDR2_BCM5701: u16 = 0x6110;

pub mod bmcr {
    pub const RESET: u16 = 0x8000;
    pub const AUTOEN: u16 = 0x1000;
    pub const ISOLATE: u16 = 0x0400;
    pub const STARTNEG: u16 = 0x0200;
    pub const FDX: u16 = 0x0100;
    pub const SPEED1000: u16 = 0x0040;
}
mod bmsr {
    pub const TX100_FDX: u16 = 0x4000;
    pub const TX100_HDX: u16 = 0x2000;
    pub const T10_FDX: u16 = 0x1000;
    pub const T10_HDX: u16 = 0x0800;
    pub const EXTSTAT: u16 = 0x0100;
    pub const ANEGDONE: u16 = 0x0020;
    pub const CANAUTONEG: u16 = 0x0008;
    pub const LINK: u16 = 0x0004;
    pub const EXTCAP: u16 = 0x0001;
}
mod anar {
    pub const TX100_FDX: u16 = 0x0100;
    pub const TX100_HDX: u16 = 0x0080;
    pub const T10_FDX: u16 = 0x0040;
    pub const T10_HDX: u16 = 0x0020;
    pub const CSMA: u16 = 0x0001;
    pub const ACK: u16 = 0x4000;
}
const GTCR_ADV_1000TFDX: u16 = 0x0200;
const GTSR_LINKED: u16 = 0x4000 | 0x2000 | 0x1000 | 0x0800;
const EXTSR_1000: u16 = 0x2000 | 0x1000;
const AUXSTS_UP: u16 = 0x8000 | 0x0700;

const BMCR_RESET_VALUE: u16 = bmcr::AUTOEN | bmcr::FDX | bmcr::SPEED1000;
const ANAR_RESET_VALUE: u16 =
    anar::CSMA | anar::T10_HDX | anar::T10_FDX | anar::TX100_HDX | anar::TX100_FDX;

/// How long autonegotiation takes to settle. It has to finish inside the
/// driver's own patience: the IRIX driver gives up its link wait well before
/// a second.
pub const SETTLE: Duration = Duration::from_millis(100);

/// What the PHY could not carry out, for the owner to count and log.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Refusal {
    /// A read of a register the model does not implement; it reads 0, never
    /// the all-ones an absent PHY returns.
    ReadUnimplemented(u8),
    WriteUnimplemented(u8),
    /// A write to a read-only register.
    WriteReadOnly(u8),
}

pub struct Phy {
    bmcr: u16,
    anar: u16,
    gtcr: u16,
    autoneg_done: bool,
    link_down: bool,
    settle_armed: bool,
    settle: Option<Instant>,
    pub refused: Vec<Refusal>,
}

impl Phy {
    pub fn new(now: Instant) -> Self {
        let mut phy = Self {
            bmcr: 0,
            anar: 0,
            gtcr: 0,
            autoneg_done: false,
            link_down: false,
            settle_armed: false,
            settle: None,
            refused: Vec::new(),
        };
        phy.reset(now);
        phy
    }

    /// The deadline at which the settle completes.
    pub fn deadline(&self) -> Option<Instant> {
        self.settle
    }

    /// Complete the settle if its time has come.
    pub fn sync(&mut self, now: Instant) {
        if self.settle.is_some_and(|d| d <= now) {
            self.settle = None;
            self.autoneg_done = true;
        }
    }

    pub fn autoneg_done(&self) -> bool {
        self.autoneg_done
    }

    pub fn link_up(&self) -> bool {
        self.autoneg_done && !self.link_down
    }

    /// The far end's cable state; not a register, so a PHY reset leaves it.
    pub fn set_link_down(&mut self, down: bool) {
        self.link_down = down;
    }

    pub fn link_down(&self) -> bool {
        self.link_down
    }

    fn status(&self) -> u16 {
        let mut status = bmsr::EXTCAP
            | bmsr::CANAUTONEG
            | bmsr::EXTSTAT
            | bmsr::TX100_FDX
            | bmsr::TX100_HDX
            | bmsr::T10_FDX
            | bmsr::T10_HDX;
        if self.link_up() {
            status |= bmsr::LINK | bmsr::ANEGDONE;
        }
        status
    }

    /// A real PHY negotiates as soon as autonegotiation is enabled, so the
    /// settle runs from reset and from the write that restarts it, not from
    /// the driver's first look at link status.
    fn arm_settle(&mut self, now: Instant) {
        if !self.settle_armed && !self.autoneg_done && self.bmcr & bmcr::AUTOEN != 0 {
            self.settle_armed = true;
            self.settle = Some(now + SETTLE);
        }
    }

    fn refuse(&mut self, what: Refusal) {
        if self.refused.len() < 64 {
            self.refused.push(what);
        }
    }

    pub fn read(&mut self, register: u8, now: Instant) -> u16 {
        self.sync(now);
        let up = self.link_up();
        match register {
            reg::BMCR => self.bmcr,
            reg::BMSR => {
                self.arm_settle(now);
                self.status()
            }
            reg::IDR1 => IDR1_VALUE,
            reg::IDR2 => IDR2_BCM5701,
            reg::ANAR => self.anar,
            // The far end advertises exactly what this end does, which puts
            // the driver on the route it trusts.
            reg::ANLPAR => {
                if up {
                    self.anar | anar::ACK
                } else {
                    0
                }
            }
            reg::GTCR => self.gtcr,
            reg::GTSR => {
                if up {
                    GTSR_LINKED
                } else {
                    0
                }
            }
            reg::EXTSR => EXTSR_1000,
            // The register the driver gates its link-up path on, so a poll
            // that reads only this one still brings the link up.
            reg::AUXSTS => {
                self.arm_settle(now);
                if self.link_up() {
                    AUXSTS_UP
                } else {
                    0
                }
            }
            other => {
                self.refuse(Refusal::ReadUnimplemented(other));
                0
            }
        }
    }

    pub fn write(&mut self, register: u8, value: u16, now: Instant) {
        self.sync(now);
        match register {
            reg::BMCR => {
                // Reset and restart are self-clearing.
                if value & bmcr::RESET != 0 {
                    self.reset(now);
                    return;
                }
                // Only a restart (or isolating the part) restarts
                // negotiation: a driver rewrites this register for other
                // reasons, and dropping a working link each time would show
                // as the carrier coming back again and again.
                if value & (bmcr::ISOLATE | bmcr::STARTNEG) != 0 {
                    self.autoneg_done = false;
                    self.settle_armed = false;
                    self.settle = None;
                }
                self.bmcr = value & !(bmcr::RESET | bmcr::STARTNEG);
                if value & bmcr::ISOLATE == 0 {
                    self.arm_settle(now);
                }
            }
            reg::ANAR => self.anar = value,
            reg::GTCR => self.gtcr = value,
            reg::BMSR | reg::IDR1 | reg::IDR2 | reg::ANLPAR | reg::GTSR | reg::EXTSR => {
                self.refuse(Refusal::WriteReadOnly(register));
            }
            other => self.refuse(Refusal::WriteUnimplemented(other)),
        }
    }

    pub fn reset(&mut self, now: Instant) {
        self.bmcr = BMCR_RESET_VALUE;
        self.anar = ANAR_RESET_VALUE;
        self.gtcr = GTCR_ADV_1000TFDX;
        self.autoneg_done = false;
        self.settle_armed = false;
        self.settle = None;
        // Out of reset autonegotiation is enabled, so the part starts it.
        self.arm_settle(now);
    }
}
