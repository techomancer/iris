//! The DP83843 MII model exposes the control, status and auto-negotiation
//! registers used by the Fuel IOC3 Ethernet port. A guest starts negotiation
//! through MII; the backend link level determines the simulated partner result.
//!
//! # What is approximated here
//!
//! Link settling and partner responses are software models. BMSR reports the
//! current link level rather than the hardware's latched-low status.
//!
//! Ported from hw/net/dp83840.c in jamesbraid/qemu-sgi-origami.

use std::time::{Duration, Instant};
pub mod reg {
    pub const BASIC_CONTROL: u8 = 0x0;
    pub const BASIC_STATUS: u8 = 0x1;
    pub const PHY_ID1: u8 = 0x2;
    pub const PHY_ID2: u8 = 0x3;
    pub const AUTO_NEG_ADVERTISE: u8 = 0x4;
    pub const AUTO_NEG_PARTNER: u8 = 0x5;
    pub const AUTO_NEG_EXPANSION: u8 = 0x6;
    pub const NEXT_PAGE_TRANSMIT: u8 = 0x7;
    pub const PHY_STATUS: u8 = 0x10;
    pub const INTERRUPT_CONTROL: u8 = 0x11;
    pub const INTERRUPT_STATUS: u8 = 0x12;
    pub const DISCONNECT_COUNTER: u8 = 0x13;
    pub const FALSE_CARRIER_COUNTER: u8 = 0x14;
    pub const RECEIVE_ERROR_COUNTER: u8 = 0x15;
    pub const PCS_CONFIG: u8 = 0x16;
    pub const LOOPBACK: u8 = 0x17;
    pub const TEN_BASE_T_STATUS: u8 = 0x18;
    pub const PHY_CONTROL: u8 = 0x19;
}
const SETTLE: Duration = Duration::from_secs(2);
const BREAK: Duration = Duration::from_millis(1500);

/// Management state for one transceiver. No board GPIO reset is connected.
pub struct Dp83843 {
    configured_addr: u8,
    regs: [u16; 32],
    backend_up: bool,
    done: bool,
    forced: bool,
    resolved: u8,
    armed: bool,
    deadline: Option<Instant>,
    complete: Option<Instant>,
    pending: bool,
    restarting: bool,
    pub unimplemented_accesses: u32,
}
impl Dp83843 {
    pub fn new(addr: u8) -> Self {
        let mut p = Self {
            configured_addr: addr & 31,
            regs: [0; 32],
            backend_up: true,
            done: false,
            forced: false,
            resolved: 0,
            armed: false,
            deadline: None,
            complete: None,
            pending: false,
            restarting: false,
            unimplemented_accesses: 0,
        };
        p.reset();
        p
    }
    pub fn reset(&mut self) {
        self.regs = [0; 32];
        self.regs[reg::BASIC_CONTROL as usize] = 0x3100 | if self.configured_addr == 0 { 0x400 } else { 0 };
        self.regs[reg::AUTO_NEG_ADVERTISE as usize] = 0x1e1;
        self.regs[reg::NEXT_PAGE_TRANSMIT as usize] = 0x2001;
        self.regs[reg::PCS_CONFIG as usize] = 0x20;
        self.regs[reg::PHY_CONTROL as usize] = u16::from(self.configured_addr);
        self.done = false;
        self.forced = false;
        self.resolved = 0;
        self.armed = false;
        self.deadline = None;
        self.complete = None;
        self.pending = false;
        self.restarting = false;
    }
    pub fn address(&self) -> u8 {
        (self.regs[reg::PHY_CONTROL as usize] & 31) as u8
    }
    pub fn backend_up(&self) -> bool {
        self.backend_up
    }
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }
    fn autoneg(&self) -> bool {
        self.regs[reg::BASIC_CONTROL as usize] & 0x1000 != 0
    }
    fn supported(&self) -> bool {
        self.regs[reg::PHY_CONTROL as usize] & 0x220 == 0
            && self.regs[reg::TEN_BASE_T_STATUS as usize] & 0x1a20 == 0
            && self.regs[reg::LOOPBACK as usize] & 0x3f80 == 0
    }
    fn link(&self) -> bool {
        self.backend_up
            && self.supported()
            && self.regs[reg::BASIC_CONTROL as usize] & 0x800 == 0
            && if self.autoneg() {
                self.done && self.resolved != 0
            } else {
                self.forced
            }
    }
    pub fn data_path(&self) -> bool {
        self.supported() && self.regs[reg::BASIC_CONTROL as usize] & 0xc00 == 0
    }
    pub fn wire_enabled(&self) -> bool {
        self.data_path() && self.link()
    }
    fn loopback_selected(&self) -> bool {
        self.regs[reg::BASIC_CONTROL as usize] & 0x4000 != 0 || self.regs[reg::LOOPBACK as usize] & 0x30 != 0
    }
    pub fn in_loopback(&self) -> bool {
        self.data_path() && self.loopback_selected()
    }
    fn status(&self) -> u16 {
        let mut v = if self.autoneg() { 0x200 } else { 0 };
        if self.loopback_selected() {
            v |= 8;
        }
        if !self.link() {
            return v;
        }
        v |= 1;
        let (speed, full) = if self.autoneg() {
            v |= 0x20;
            (self.resolved <= 2, self.resolved == 1 || self.resolved == 3)
        } else {
            (self.regs[reg::BASIC_CONTROL as usize] & 0x2000 != 0, self.regs[reg::BASIC_CONTROL as usize] & 0x100 != 0)
        };
        if !speed {
            v |= 2;
        }
        if full {
            v |= 4;
        }
        if self.regs[reg::BASIC_CONTROL as usize] & 0x400 == 0 {
            v |= 0x800;
        }
        v
    }
    fn event(&mut self, before: u16) {
        let after = self.status();
        if self.regs[reg::INTERRUPT_CONTROL as usize] & 2 != 0 && ((before ^ after) & 1 != 0 || !before & after & 0x20 != 0) {
            self.pending = true;
        }
    }
    fn resolve(&mut self) {
        self.resolved = 0;
        if self.autoneg() && self.supported() {
            for (i, bit) in [0x100, 0x80, 0x40, 0x20].iter().enumerate() {
                if self.regs[reg::AUTO_NEG_ADVERTISE as usize] & bit != 0 {
                    self.resolved = i as u8 + 1;
                    break;
                }
            }
        }
        if self.resolved == 0 {
            crate::dlog!(crate::devlog::LogModule::Ioc, "dp83843: no common supported advertised mode");
        }
    }
    pub fn sync(&mut self, now: Instant) {
        if self.deadline.is_some_and(|d| now >= d) {
            let before = self.status();
            if let Some(end) = self.complete.take() {
                self.regs[reg::BASIC_CONTROL as usize] &= !0x200;
                self.deadline = Some(end);
                if now < end {
                    return;
                }
            }
            self.deadline = None;
            self.restarting = false;
            self.done = true;
            self.resolve();
            self.event(before);
        }
    }
    pub fn set_backend_link(&mut self, up: bool, now: Instant) {
        self.sync(now);
        let before = self.status();
        self.backend_up = up;
        self.event(before);
    }
    fn unimplemented(&mut self, r: u8) {
        self.unimplemented_accesses += 1;
        crate::dlog!(crate::devlog::LogModule::Ioc, "dp83843: unsupported register or control at {r:#x}");
    }
    pub fn read(&mut self, r: u8, now: Instant) -> u16 {
        self.sync(now);
        let v = match r {
            reg::BASIC_CONTROL => self.regs[reg::BASIC_CONTROL as usize],
            reg::BASIC_STATUS => {
                if !self.armed && !self.done && self.autoneg() {
                    self.armed = true;
                    self.deadline = Some(now + SETTLE);
                }
                let status = self.status();
                0x7849 | if status & 1 != 0 { 4 } else { 0 } | (status & 0x20)
            }
            reg::PHY_ID1 => 0x2000,
            reg::PHY_ID2 => 0x5c10,
            reg::AUTO_NEG_ADVERTISE | reg::NEXT_PAGE_TRANSMIT => self.regs[r as usize],
            reg::AUTO_NEG_PARTNER => {
                if self.done {
                    0x1e1
                } else {
                    0
                }
            }
            reg::AUTO_NEG_EXPANSION => 4 | u16::from(self.done),
            reg::PHY_STATUS => {
                let v = self.status() | if self.pending { 0x100 } else { 0 };
                self.pending = false;
                v
            }
            reg::INTERRUPT_CONTROL => self.regs[reg::INTERRUPT_CONTROL as usize],
            reg::INTERRUPT_STATUS => {
                let v = if self.pending { 0x8000 } else { 0 };
                self.pending = false;
                v
            }
            reg::DISCONNECT_COUNTER..=reg::RECEIVE_ERROR_COUNTER => {
                let v = self.regs[r as usize];
                self.regs[r as usize] = 0;
                v
            }
            reg::PCS_CONFIG => {
                let v = self.regs[reg::PCS_CONFIG as usize];
                if self.autoneg() {
                    v & !0x4000
                } else if self.regs[reg::PHY_CONTROL as usize] & 0x20 != 0 {
                    v | 0x4000
                } else {
                    v
                }
            }
            reg::LOOPBACK => {
                self.regs[reg::LOOPBACK as usize]
                    | if self.regs[reg::PHY_CONTROL as usize] & 0x20 != 0 {
                        0x1000
                    } else {
                        0
                    }
            }
            reg::TEN_BASE_T_STATUS => {
                self.regs[reg::TEN_BASE_T_STATUS as usize]
                    | if self.status() & 4 != 0 || self.regs[reg::PHY_CONTROL as usize] & 0x200 != 0 {
                        0x80
                    } else {
                        0
                    }
            }
            reg::PHY_CONTROL => self.regs[reg::PHY_CONTROL as usize],
            _ => {
                self.unimplemented(r);
                0
            }
        };
        if self.regs[reg::INTERRUPT_CONTROL as usize] & 1 != 0 {
            self.pending = true;
        }
        v
    }
    pub fn write(&mut self, r: u8, val: u16, now: Instant) {
        self.sync(now);
        let before = self.status();
        match r {
            reg::BASIC_CONTROL => {
                let val = val & 0xff80;
                if val & 0x8000 != 0 {
                    self.reset();
                    return;
                }
                if val & 0x80 != 0 {
                    self.unimplemented(r);
                }
                self.regs[reg::LOOPBACK as usize] &= !0x380;
                if val & 0x4000 != 0 {
                    self.regs[reg::LOOPBACK as usize] |= 0x30;
                }
                if val & 0x400 == 0 && val & 0x1200 == 0x1200 {
                    self.done = false;
                    self.resolved = 0;
                    self.forced = false;
                    self.armed = true;
                    self.restarting = true;
                    self.deadline = Some(now + BREAK);
                    self.complete = Some(now + BREAK + SETTLE);
                    self.regs[reg::BASIC_CONTROL as usize] = val;
                } else if val & 0x400 == 0 && (self.restarting || self.forced && !self.autoneg()) {
                    if val & 0x1000 == 0 {
                        self.restarting = false;
                        self.complete = None;
                        self.deadline = None;
                        self.forced = val & 0x800 == 0;
                    }
                    self.regs[reg::BASIC_CONTROL as usize] = val & !0x200;
                } else {
                    self.restarting = false;
                    self.complete = None;
                    self.deadline = None;
                    self.forced = false;
                    self.done = val & 0x400 == 0;
                    self.armed = self.done;
                    self.regs[reg::BASIC_CONTROL as usize] = val & !0x8200;
                }
                if !self.autoneg() {
                    self.resolved = 0;
                    self.done = false;
                    self.forced = self.supported() && self.regs[reg::BASIC_CONTROL as usize] & 0xc00 == 0;
                } else if self.done && self.resolved == 0 {
                    self.resolve();
                }
            }
            reg::BASIC_STATUS..=reg::PHY_ID2 | reg::AUTO_NEG_PARTNER | reg::AUTO_NEG_EXPANSION | reg::PHY_STATUS | reg::INTERRUPT_STATUS => {}
            reg::AUTO_NEG_ADVERTISE => self.regs[reg::AUTO_NEG_ADVERTISE as usize] = val & 0xa5ff,
            reg::NEXT_PAGE_TRANSMIT => self.regs[reg::NEXT_PAGE_TRANSMIT as usize] = (val & 0xb7ff) | (self.regs[reg::NEXT_PAGE_TRANSMIT as usize] & 0x800),
            reg::INTERRUPT_CONTROL => self.regs[reg::INTERRUPT_CONTROL as usize] = val & 3,
            reg::DISCONNECT_COUNTER..=reg::RECEIVE_ERROR_COUNTER => self.regs[r as usize] = val,
            reg::PCS_CONFIG => {
                if val & !0x20 != 0 {
                    self.unimplemented(r);
                }
                let mask = if self.autoneg() || self.regs[reg::PHY_CONTROL as usize] & 0x20 != 0 {
                    0xbfef
                } else {
                    0xffef
                };
                self.regs[reg::PCS_CONFIG as usize] = (self.regs[reg::PCS_CONFIG as usize] & !mask) | (val & mask);
            }
            reg::LOOPBACK => {
                if val & 0x4000 != 0 {
                    self.unimplemented(r);
                }
                self.regs[reg::LOOPBACK as usize] = val & 0x7fb0;
            }
            reg::TEN_BASE_T_STATUS => {
                if val & 0x55c != 0 {
                    self.unimplemented(r);
                }
                self.regs[reg::TEN_BASE_T_STATUS as usize] = val & 0x1ffc;
            }
            reg::PHY_CONTROL => {
                if val & !31 != 0 {
                    self.unimplemented(r);
                }
                self.regs[reg::PHY_CONTROL as usize] = val & 0x3bff;
            }
            _ => self.unimplemented(r),
        }
        if !self.supported() {
            self.resolved = 0;
            self.forced = false;
            self.unimplemented(r);
        }
        self.event(before);
        if self.regs[reg::INTERRUPT_CONTROL as usize] & 1 != 0 {
            self.pending = true;
        }
    }
}
impl crate::dev::ioc3_eth::MiiPhy for Dp83843 {
    fn address(&self) -> u8 { Dp83843::address(self) }
    fn deadline(&self) -> Option<Instant> { Dp83843::deadline(self) }
    fn wire_enabled(&self) -> bool { Dp83843::wire_enabled(self) }
    fn in_loopback(&self) -> bool { Dp83843::in_loopback(self) }
    fn sync(&mut self, now: Instant) { Dp83843::sync(self, now) }
    fn read(&mut self, register: u8, now: Instant) -> u16 { Dp83843::read(self, register, now) }
    fn write(&mut self, register: u8, value: u16, now: Instant) { Dp83843::write(self, register, value, now) }
    fn set_backend_link(&mut self, up: bool, now: Instant) { Dp83843::set_backend_link(self, up, now) }
    fn power_on(&mut self) { Dp83843::reset(self) }
}

#[cfg(test)]
mod tests {
    // Cases ported from tests/qtest/sgi-ioc3-eth-test.c in jamesbraid/qemu-sgi-origami.
    use super::*;

    /// The guest observes link and negotiation changes through MII status registers.
    #[test]
    fn the_phy_restarts_negotiation_and_resets_the_link() {
        let now = Instant::now();
        let mut p = Dp83843::new(1);
        assert_eq!(p.read(reg::BASIC_STATUS, now) & 0x24, 0);
        assert!(!p.wire_enabled());
        assert_eq!(p.read(reg::BASIC_STATUS, now + Duration::from_secs(2)) & 0x24, 0x24);
        p.write(reg::BASIC_CONTROL, 0x3300, now + Duration::from_secs(3));
        for (millis, restart, link) in [(4499, true, false), (4500, false, false), (6500, false, true)] {
            let at = now + Duration::from_millis(millis);
            assert_eq!(p.read(reg::BASIC_CONTROL, at) & 0x200 != 0, restart);
            assert_eq!(p.read(reg::BASIC_STATUS, at) & 4 != 0, link);
            assert_eq!(p.wire_enabled(), link);
        }
        let at = now + Duration::from_secs(7);
        p.set_backend_link(false, at);
        p.write(reg::PHY_CONTROL, 7, at);
        p.write(reg::BASIC_CONTROL, 0x8000, at);
        assert_eq!(p.address(), 1);
        assert_eq!(p.read(reg::BASIC_STATUS, at) & 4, 0);
        assert_eq!(p.read(reg::BASIC_STATUS, at + Duration::from_secs(2)) & 4, 0);
        p.set_backend_link(true, at + Duration::from_secs(2));
        assert!(p.wire_enabled());
    }

    /// A PROM probing the PHY reads the next-page register's reset value and
    /// writes it back; the read-only toggle bit must survive every write.
    #[test]
    fn the_next_page_register_keeps_its_toggle_bit_and_resets_to_0x2001() {
        let now = Instant::now();
        let mut p = Dp83843::new(1);
        assert_eq!(p.read(reg::NEXT_PAGE_TRANSMIT, now), 0x2001);
        for bit in 0..16 {
            let value = 1u16 << bit;
            p.write(reg::NEXT_PAGE_TRANSMIT, value, now);
            assert_eq!(p.read(reg::NEXT_PAGE_TRANSMIT, now), value & 0xb7ff, "bit {bit}");
        }
        p.write(reg::NEXT_PAGE_TRANSMIT, 0xffff, now);
        assert_eq!(p.read(reg::NEXT_PAGE_TRANSMIT, now), 0xb7ff);
        p.write(reg::BASIC_CONTROL, 0x8000, now);
        assert_eq!(p.read(reg::NEXT_PAGE_TRANSMIT, now), 0x2001);
    }

    /// The PROM's PHY probe expects zero loopback and masked 10BASE-T status
    /// registers, plus the standard BMSR and ANER capability bits, both at
    /// power-on and after a BMCR reset.
    #[test]
    fn the_phy_reads_the_probe_values_the_prom_expects_before_and_after_reset() {
        let now = Instant::now();
        let mut p = Dp83843::new(1);
        for round in 0..2 {
            assert_eq!(p.read(reg::BASIC_STATUS, now) & 0x7849, 0x7849, "round {round}");
            assert_eq!(p.read(reg::AUTO_NEG_EXPANSION, now) & 4, 4);
            assert_eq!(p.read(reg::LOOPBACK, now), 0);
            assert_eq!(p.read(reg::TEN_BASE_T_STATUS, now) & 0xc77f, 0);
            p.write(reg::BASIC_CONTROL, 0x2100, now);
            p.write(reg::LOOPBACK, 0x20, now);
            p.write(reg::BASIC_CONTROL, 0x8000, now);
        }
        assert_eq!(p.unimplemented_accesses, 0);
    }

    /// Setting BMCR loopback makes the PHY report loopback in the LBR bits the
    /// MAC consults, and leaving it must drop the stale bypass bits.
    #[test]
    fn bmcr_loopback_selects_the_loopback_register_and_clears_the_bypass_bits() {
        let now = Instant::now();
        let mut p = Dp83843::new(1);
        p.write(reg::LOOPBACK, 0x0180, now);
        p.write(reg::BASIC_CONTROL, 0x7300, now);
        assert_eq!(p.read(reg::LOOPBACK, now) & 0x03b0, 0x0030);
        assert!(p.in_loopback());
        let lbr = p.read(reg::LOOPBACK, now);
        p.write(reg::LOOPBACK, lbr | 0x0180, now);
        p.write(reg::BASIC_CONTROL, 0x3100, now);
        assert_eq!(p.read(reg::LOOPBACK, now) & 0x0380, 0);
        p.write(reg::BASIC_CONTROL, 0x8000, now);
        assert_eq!(p.read(reg::LOOPBACK, now), 0);
        assert!(!p.in_loopback());
    }

    /// Drivers poll the error counters and expect each read to return the count
    /// since the previous read, not a running total.
    #[test]
    fn the_error_counters_clear_when_read() {
        let now = Instant::now();
        let mut p = Dp83843::new(1);
        for r in [reg::DISCONNECT_COUNTER, reg::FALSE_CARRIER_COUNTER, reg::RECEIVE_ERROR_COUNTER] {
            p.write(r, 0xbeef, now);
            assert_eq!(p.read(r, now), 0xbeef);
            assert_eq!(p.read(r, now), 0);
        }
    }

    /// A guest that restarts negotiation twice in a row must see the break
    /// interval from the second write, not a shortened one from the first.
    #[test]
    fn a_second_restart_request_rearms_the_break_interval() {
        let now = Instant::now();
        let ms = |m: u64| now + Duration::from_millis(m);
        let mut p = Dp83843::new(1);
        p.write(reg::BASIC_CONTROL, 0x3200, ms(0));
        p.write(reg::BASIC_CONTROL, 0x3200, ms(750));
        assert_ne!(p.read(reg::BASIC_CONTROL, ms(2249)) & 0x200, 0);
        assert_eq!(p.read(reg::BASIC_CONTROL, ms(2250)) & 0x200, 0);
        assert_eq!(p.read(reg::BASIC_STATUS, ms(4249)) & 0x20, 0);
        assert_eq!(p.read(reg::BASIC_STATUS, ms(4250)) & 0x20, 0x20);
    }
}
