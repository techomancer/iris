//! The DP83840A MII model: the Origin 200 System Board's Ethernet
//! transceiver, reached through the IOC3's management registers.
//!
//! Out of reset the link is down. The first status read arms a settle, after
//! which negotiation reads complete; a write to the control register links at
//! once, since there is no cable to negotiate against. The DP83843 has its own
//! model, [`crate::dev::dp83843`].
//!
//! # What is approximated here
//!
//! - The synthetic partner offers every mode and the settle is two seconds,
//!   inside the part's 1.5 to 2.3 s range, long enough that SGI's diagnostic
//!   sweep still finds the link settling.
//! - Registers 0x17, 0x18 and 0x1c are storage with their reset values; the
//!   pin-latched bits of 0x18 are unobserved and modelled as normal
//!   operation. Every other register reads zero and is logged.
//!
//! Ported from hw/net/dp83840.c in jamesbraid/qemu-sgi-origami.

use std::time::{Duration, Instant};

use crate::dev::mii::MiiPhy;
use crate::snapshot::{field_bool, field_u16, field_u64, field_u8, StateTable};

pub mod reg {
    pub const CONTROL: u8 = 0;
    pub const STATUS: u8 = 1;
    pub const ID1: u8 = 2;
    pub const ID2: u8 = 3;
    pub const ADVERTISE: u8 = 4;
    pub const PARTNER: u8 = 5;
    pub const EXPANSION: u8 = 6;
    pub const PCS_CONFIG: u8 = 0x17;
    pub const LOOPBACK_BYPASS: u8 = 0x18;
    pub const ADDRESS_STATUS: u8 = 0x19;
    pub const TENBT_CONFIG: u8 = 0x1c;
}

const ID1: u16 = 0x2000;
const ID2: u16 = 0x5c01;
const CONTROL_RESET: u16 = 0x3100;
const PCS_RESET: u16 = 0x8040;
const TENBT_RESET: u16 = 0x80b9;
const LOOPBACK_RESET: u16 = 0x8020;
/// Both ends offer every mode; what a real machine negotiates is unrecorded.
const ADVERTISE_RESET: u16 = 0x1e1;
const SETTLE: Duration = Duration::from_secs(2);

const BMCR_RESET: u16 = 1 << 15;
const BMCR_LOOPBACK: u16 = 1 << 14;
const BMCR_AUTONEG: u16 = 1 << 12;
const BMCR_ISOLATE: u16 = 1 << 10;
const BMCR_RESTART: u16 = 1 << 9;
const BMSR_CAPS: u16 = 0x7809; // 100TX/10T both duplexes, autoneg and ext-cap
const BMSR_LINK: u16 = 1 << 2;
const BMSR_NEGOTIATED: u16 = 1 << 5;
const LBREMR_LOOPBACK: u16 = 0x0300 | 0x0800;

pub struct Dp83840 {
    addr: u8,
    control: u16,
    advertise: u16,
    pcs_config: u16,
    loopback_bypass: u16,
    tenbt_config: u16,
    negotiated: bool,
    armed: bool,
    deadline: Option<Instant>,
    forced: bool,
    backend_up: bool,
}

impl Dp83840 {
    pub fn new(addr: u8) -> Self {
        let mut p = Self {
            addr: addr & 31,
            control: 0,
            advertise: 0,
            pcs_config: 0,
            loopback_bypass: 0,
            tenbt_config: 0,
            negotiated: false,
            armed: false,
            deadline: None,
            forced: false,
            backend_up: true,
        };
        p.reset();
        p
    }

    pub fn reset(&mut self) {
        self.control = CONTROL_RESET;
        self.advertise = ADVERTISE_RESET;
        self.pcs_config = PCS_RESET;
        self.loopback_bypass = LOOPBACK_RESET;
        self.tenbt_config = TENBT_RESET;
        self.negotiated = false;
        self.armed = false;
        self.deadline = None;
        self.forced = false;
    }

    fn link(&self) -> bool {
        self.backend_up && (self.negotiated || self.forced)
    }

    fn status(&self) -> u16 {
        let mut v = BMSR_CAPS;
        if self.link() {
            v |= BMSR_LINK;
            if self.negotiated {
                v |= BMSR_NEGOTIATED;
            }
        }
        v
    }

    fn unimplemented(&self, r: u8, write: Option<u16>) {
        match write {
            Some(v) => crate::dlog!(crate::devlog::LogModule::Ioc, "dp83840: write of {v:#06x} to unimplemented register {r:#x}"),
            None => crate::dlog!(crate::devlog::LogModule::Ioc, "dp83840: read of unimplemented register {r:#x}"),
        }
    }
}

impl MiiPhy for Dp83840 {
    fn address(&self) -> u8 {
        self.addr
    }

    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// The model does not gate frames on the link.
    fn wire_enabled(&self) -> bool {
        true
    }

    /// Three switches select loopback; the diagnostic uses more than one.
    fn in_loopback(&self) -> bool {
        self.control & BMCR_LOOPBACK != 0 || self.loopback_bypass & LBREMR_LOOPBACK != 0
    }

    fn sync(&mut self, now: Instant) {
        if self.deadline.is_some_and(|d| now >= d) {
            self.deadline = None;
            self.negotiated = true;
        }
    }

    fn read(&mut self, register: u8, now: Instant) -> u16 {
        self.sync(now);
        match register {
            reg::CONTROL => self.control,
            reg::STATUS => {
                // The settle arms on the first status read: armed at reset it
                // would elapse before SGI's diagnostic probes, which must find
                // the link down.
                if !self.armed && !self.negotiated && self.control & BMCR_AUTONEG != 0 {
                    self.armed = true;
                    self.deadline = Some(now + SETTLE);
                }
                self.status()
            }
            reg::ID1 => ID1,
            reg::ID2 => ID2,
            reg::ADVERTISE => self.advertise,
            reg::PARTNER => {
                if self.negotiated {
                    ADVERTISE_RESET
                } else {
                    0
                }
            }
            reg::EXPANSION => u16::from(self.negotiated),
            reg::PCS_CONFIG => self.pcs_config,
            reg::LOOPBACK_BYPASS => self.loopback_bypass,
            reg::TENBT_CONFIG => self.tenbt_config,
            reg::ADDRESS_STATUS => u16::from(self.addr),
            r => {
                // Zero: all ones reads as an absent transceiver.
                self.unimplemented(r, None);
                0
            }
        }
    }

    fn write(&mut self, register: u8, value: u16, now: Instant) {
        self.sync(now);
        match register {
            reg::CONTROL => {
                if value & BMCR_RESET != 0 {
                    self.reset();
                    return;
                }
                // No cable to negotiate against: a configuring write links at
                // once, unless it isolates the part.
                self.forced = false;
                self.deadline = None;
                self.armed = true;
                self.negotiated = value & BMCR_ISOLATE == 0;
                self.control = value & !(BMCR_RESET | BMCR_RESTART);
            }
            reg::ADVERTISE => self.advertise = value,
            reg::PCS_CONFIG => self.pcs_config = value,
            reg::LOOPBACK_BYPASS => self.loopback_bypass = value,
            reg::TENBT_CONFIG => self.tenbt_config = value,
            reg::STATUS | reg::ID1 | reg::ID2 | reg::PARTNER | reg::EXPANSION | reg::ADDRESS_STATUS => {
                crate::dlog!(crate::devlog::LogModule::Ioc, "dp83840: write of {value:#06x} to read-only register {register:#x}");
            }
            r => self.unimplemented(r, Some(value)),
        }
    }

    fn set_backend_link(&mut self, up: bool, now: Instant) {
        self.sync(now);
        self.backend_up = up;
    }

    fn power_on(&mut self) {
        self.reset();
    }

    fn save_state(&self, now: Instant) -> toml::Value {
        StateTable::new()
            .u8("addr", self.addr)
            .u16("control", self.control)
            .u16("advertise", self.advertise)
            .u16("pcs_config", self.pcs_config)
            .u16("loopback_bypass", self.loopback_bypass)
            .u16("tenbt_config", self.tenbt_config)
            .bool("negotiated", self.negotiated)
            .bool("armed", self.armed)
            .u64("deadline", self.deadline.map_or(0, |d| d.saturating_duration_since(now).as_nanos() as u64 + 1))
            .bool("forced", self.forced)
            .bool("backend_up", self.backend_up)
            .build()
    }

    fn load_state(&mut self, v: &toml::Value, now: Instant) -> Result<(), String> {
        let left = field_u64(v, "deadline")?;
        *self = Self {
            addr: field_u8(v, "addr")?,
            control: field_u16(v, "control")?,
            advertise: field_u16(v, "advertise")?,
            pcs_config: field_u16(v, "pcs_config")?,
            loopback_bypass: field_u16(v, "loopback_bypass")?,
            tenbt_config: field_u16(v, "tenbt_config")?,
            negotiated: field_bool(v, "negotiated")?,
            armed: field_bool(v, "armed")?,
            deadline: (left != 0).then(|| now + Duration::from_nanos(left - 1)),
            forced: field_bool(v, "forced")?,
            backend_up: field_bool(v, "backend_up")?,
        };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sgi-ioc3-eth-test.c in jamesbraid/qemu-sgi-origami.
    use super::*;

    #[test]
    fn the_reset_values_are_the_ones_sgis_diagnostic_compares() {
        let now = Instant::now();
        let mut p = Dp83840::new(31);
        assert_eq!(p.address(), 31);
        assert_eq!(p.read(reg::CONTROL, now), 0x3100);
        assert_eq!(p.read(reg::ID1, now), 0x2000);
        assert_eq!(p.read(reg::ID2, now) & 0xfffe, 0x5c00);
        assert_eq!(p.read(reg::EXPANSION, now), 0);
        assert_eq!(p.read(reg::PCS_CONFIG, now), 0x8040);
        assert_eq!(p.read(reg::TENBT_CONFIG, now) & 0x3d, 0x39);
        assert_eq!(p.read(reg::ADDRESS_STATUS, now), 31);
    }

    #[test]
    fn the_link_is_down_until_the_settle_after_the_first_status_read() {
        let now = Instant::now();
        let mut p = Dp83840::new(31);
        assert_eq!(p.read(reg::STATUS, now) & 0x4, 0);
        assert_eq!(p.read(reg::STATUS, now + Duration::from_millis(1999)) & 0x4, 0);
        let up = p.read(reg::STATUS, now + Duration::from_secs(2));
        assert_eq!(up & (0x24), 0x24);
        p.set_backend_link(false, now + Duration::from_secs(3));
        assert_eq!(p.read(reg::STATUS, now + Duration::from_secs(3)) & 0x4, 0);
    }

    #[test]
    fn a_control_write_links_at_once_and_isolate_does_not() {
        let now = Instant::now();
        let mut p = Dp83840::new(31);
        p.write(reg::CONTROL, 0x1000, now);
        assert_eq!(p.read(reg::STATUS, now) & 0x4, 0x4);
        p.write(reg::CONTROL, 0x1400, now);
        assert_eq!(p.read(reg::STATUS, now) & 0x4, 0);
        p.write(reg::CONTROL, 0x8000, now);
        assert_eq!(p.read(reg::CONTROL, now), 0x3100);
    }

    #[test]
    fn any_of_three_switches_selects_loopback() {
        let now = Instant::now();
        let mut p = Dp83840::new(31);
        assert!(!p.in_loopback());
        p.write(reg::CONTROL, 0x5000, now);
        assert!(p.in_loopback());
        p.write(reg::CONTROL, 0x1000, now);
        p.write(reg::LOOPBACK_BYPASS, 0x8020 | 0x0100, now);
        assert!(p.in_loopback());
        p.write(reg::LOOPBACK_BYPASS, 0x8020 | 0x0800, now);
        assert!(p.in_loopback());
        p.write(reg::LOOPBACK_BYPASS, 0x8020, now);
        assert!(!p.in_loopback());
    }

    #[test]
    fn read_only_and_unknown_registers_keep_their_value_and_read_zero() {
        let now = Instant::now();
        let mut p = Dp83840::new(31);
        p.write(reg::ID2, 0, now);
        assert_eq!(p.read(reg::ID2, now), 0x5c01);
        p.write(0x12, 5, now);
        assert_eq!(p.read(0x12, now), 0);
    }

    #[test]
    fn a_snapshot_keeps_the_settle_it_had_left() {
        let now = Instant::now();
        let mut p = Dp83840::new(31);
        p.read(reg::STATUS, now);
        let saved = p.save_state(now + Duration::from_millis(500));
        let mut q = Dp83840::new(31);
        q.load_state(&saved, now).unwrap();
        assert_eq!(q.deadline(), Some(now + Duration::from_millis(1500)));
    }
}
