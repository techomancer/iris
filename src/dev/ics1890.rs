//! The ICS1890 transceiver fitted to Octane's IOC3.
//!
//! This module is shared by the Octane IOC3 Ethernet model. It was moved here
//! from `ioc3/eth.rs`; its management-register behavior follows the existing
//! Octane behavior and Integrated Circuit Systems, "ICS1890 10Base-T/100Base-TX
//! Integrated PHYceiver", document ICS1890RevG, 21 October 1997.
//!
//! Ported from hw/net/ics1890.c in jamesbraid/qemu-sgi-origami.

use std::time::{Duration, Instant};

use crate::dev::mii::MiiPhy;
use crate::snapshot::{field_bool, field_u16, field_u32, field_u64, field_u8, StateTable};

/// Ordinary and command-override writable masks, by register.
fn phy_masks(r: usize) -> (u16, u16) {
    match r {
        0 => (0xff80, 0),
        1 => (0, 0x7f80),
        2 | 3 => (0, 0xffff),
        4 => (0x21e0, 0x1c1f),
        6 => (0, 0xffe0),
        0x10 => (0x802d, 0),
        0x12 => (0x002f, 0),
        0x13 => (0x0011, 0),
        _ => (0, 0),
    }
}

/// The ICS1890 as the Octane carries it: address 1, node mode, software
/// priority, revision 3.
pub struct Ics1890 {
    addr: u8,
    bmcr: u16,
    bmsr_caps: u16,
    id1: u16,
    id2: u16,
    anar: u16,
    partner: u16,
    exp: u16,
    ext_ctl: u16,
    tenbt: u16,
    ext2: u16,
    override_armed: bool,
    page_received: bool,
    link_low_latch: bool,
    last_link: bool,
    backend_up: bool,
    nego_done: bool,
    deadline: Option<Instant>,
    delay: Duration,
    /// Accesses to registers the part does not implement.
    pub unimplemented_accesses: u32,
}

impl Ics1890 {
    pub fn new(addr: u8, delay: Duration) -> Self {
        let mut p = Self {
            addr,
            bmcr: 0,
            bmsr_caps: 0,
            id1: 0,
            id2: 0,
            anar: 0,
            partner: 0,
            exp: 0,
            ext_ctl: 0,
            tenbt: 0,
            ext2: 0,
            override_armed: false,
            page_received: false,
            link_low_latch: false,
            last_link: false,
            backend_up: true,
            nego_done: false,
            deadline: None,
            delay,
            unimplemented_accesses: 0,
        };
        p.reset(Instant::now());
        p
    }



    /// Hardware and software reset: defaults, no override, no latches, and a
    /// fresh negotiation if enabled.
    pub fn reset(&mut self, now: Instant) {
        self.bmcr = 0x3000;
        self.bmsr_caps = 0x7800;
        self.id1 = 0x0015;
        self.id2 = 0xf423;
        self.anar = 0x01e1;
        self.partner = 0;
        self.exp = 0;
        self.ext_ctl = 0x0008;
        self.tenbt = 0;
        self.ext2 = 0x0001;
        self.override_armed = false;
        self.page_received = false;
        self.link_low_latch = false;
        self.nego_done = false;
        self.deadline = None;
        self.last_link = false;
        self.start_negotiation(now);
    }

    fn autoneg(&self) -> bool {
        self.bmcr & 0x1000 != 0
    }
    fn power_down(&self) -> bool {
        self.bmcr & 0x0800 != 0
    }
    fn isolated(&self) -> bool {
        self.bmcr & 0x0400 != 0
    }

    /// The live link: down in power-down, down without a backend, and with
    /// negotiation enabled up only once it has completed.
    fn link(&self) -> bool {
        !self.power_down() && self.backend_up && (!self.autoneg() || self.nego_done)
    }

    fn start_negotiation(&mut self, now: Instant) {
        self.nego_done = false;
        self.partner = 0;
        self.page_received = false;
        self.deadline = None;
        if self.autoneg() && self.backend_up && !self.power_down() {
            // The synthetic peer advertises 0x01e1: IEEE 802.3, no fault.
            let common = self.anar & 0x01e0;
            if self.anar & 0x1f == 1 && common != 0 && self.anar & 0x2000 == 0 {
                self.deadline = Some(now + self.delay);
            }
        }
    }









    /// Frames may cross the MII at all.
    pub fn data_path(&self) -> bool {
        !self.power_down() && !self.isolated()
    }



    fn resolved(&self) -> (bool, bool) {
        // (100 Mb/s, full duplex)
        if self.autoneg() {
            let common = self.anar & self.partner & 0x01e0;
            if common & 0x100 != 0 {
                (true, true)
            } else if common & 0x80 != 0 {
                (true, false)
            } else if common & 0x40 != 0 {
                (false, true)
            } else {
                (false, false)
            }
        } else {
            (self.bmcr & 0x2000 != 0, self.bmcr & 0x0100 != 0)
        }
    }




}

// The transceiver's registers and auto-negotiation progress for a
// snapshot. Its one deadline is saved as the time it had left.
impl Ics1890 {



}


impl MiiPhy for Ics1890 {
    fn address(&self) -> u8 {
        self.addr
    }

    /// When a negotiation in progress completes, if one is.
    fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Frames may go to or come from the packet backend.
    fn wire_enabled(&self) -> bool {
        self.data_path() && self.link()
    }

    /// BMCR loopback turns frames round inside the part. Isolate and
    /// power-down take precedence.
    fn in_loopback(&self) -> bool {
        self.bmcr & 0x4000 != 0 && !self.power_down() && !self.isolated()
    }

    /// Advance the negotiation deadline and note link transitions. Reads never
    /// start or postpone negotiation; this only observes the clock.
    fn sync(&mut self, now: Instant) {
        if let Some(d) = self.deadline {
            if now >= d && !self.power_down() && self.backend_up {
                self.deadline = None;
                self.nego_done = true;
                self.partner = 0x01e1;
                self.page_received = true;
            }
        }
        let link = self.link();
        if self.last_link && !link {
            self.link_low_latch = true;
        }
        self.last_link = link;
    }

    fn read(&mut self, r: u8, now: Instant) -> u16 {
        self.sync(now);
        match r {
            0 => self.bmcr,
            1 => {
                let mut v = 0x0009 | self.bmsr_caps;
                if self.autoneg() && self.nego_done {
                    v |= 0x0020;
                }
                if self.link() && !self.link_low_latch {
                    v |= 0x0004;
                }
                self.link_low_latch = false;
                v
            }
            2 => self.id1,
            3 => self.id2,
            4 => self.anar,
            5 => self.partner,
            6 => {
                let v = (self.exp & 0xffe0)
                    | if self.nego_done { 1 } else { 0 }
                    | if self.page_received { 2 } else { 0 };
                self.page_received = false;
                v
            }
            0x10 => {
                // Bit 4 is a fixed reserved read choice (stock firmware
                // expects it set); bit 15 reads the override state.
                (self.ext_ctl & 0x002d)
                    | (u16::from(self.addr & 0x1f) << 6)
                    | 0x0010
                    | if self.override_armed { 0x8000 } else { 0 }
            }
            0x11 => {
                let link = self.link();
                let (speed, full) = self.resolved();
                let mut v = 0;
                if link {
                    v |= 1;
                    if speed {
                        v |= 0x8000;
                    }
                    if full {
                        v |= 0x4000;
                    }
                }
                if self.nego_done {
                    v |= 0x0010 | (7 << 11);
                }
                v
            }
            0x12 => (self.tenbt & 0x002f) | 0x0010,
            // Node mode strap in bit 14, software bit 0.
            0x13 => (self.ext2 & 0x0011) | 0x4000,
            _ => {
                self.unimplemented_accesses += 1;
                0
            }
        }
    }

    fn write(&mut self, r: u8, val: u16, now: Instant) {
        self.sync(now);
        let r = usize::from(r);
        let armed = self.override_armed;
        match r {
            0 => {
                if val & 0x8000 != 0 {
                    self.reset(now);
                    self.sync(now);
                    return;
                }
                let old = self.bmcr;
                // Reset and restart are self-clearing.
                self.bmcr = val & 0x7d80;
                let restart = val & 0x0200 != 0;
                let pd_now = self.power_down();
                if pd_now && old & 0x0800 == 0 {
                    // Entering power-down clears the latching status.
                    self.link_low_latch = false;
                    self.page_received = false;
                    self.deadline = None;
                } else if !pd_now && old & 0x0800 != 0 {
                    // Release: enabled negotiation starts afresh.
                    self.start_negotiation(now);
                } else if self.autoneg() && (restart || old & 0x1000 == 0) {
                    self.start_negotiation(now);
                } else if !self.autoneg() {
                    self.nego_done = false;
                    self.deadline = None;
                }
                if armed {
                    self.override_armed = false;
                }
            }
            1 | 2 | 3 | 4 | 6 | 0x10 | 0x12 | 0x13 => {
                let (ord, ovr) = phy_masks(r);
                let mask = ord | if armed { ovr } else { 0 };
                let cell = match r {
                    1 => &mut self.bmsr_caps,
                    2 => &mut self.id1,
                    3 => &mut self.id2,
                    4 => &mut self.anar,
                    6 => &mut self.exp,
                    0x10 => &mut self.ext_ctl,
                    0x12 => &mut self.tenbt,
                    _ => &mut self.ext2,
                };
                let before = *cell;
                *cell = (before & !mask) | (val & mask);
                // The status caps live in bits 14..7 only.
                if r == 1 {
                    self.bmsr_caps &= 0x7f80;
                }
                if r == 4 && self.anar != before && self.autoneg() {
                    self.start_negotiation(now);
                }
                if armed {
                    self.override_armed = false;
                } else if r == 0x10 && val & 0x8000 != 0 {
                    self.override_armed = true;
                }
            }
            _ => {
                // Partner ability, Quick Poll and unimplemented registers
                // ignore writes. The part is addressed, so a write still
                // consumes an armed override.
                if r != 5 && r != 0x11 {
                    self.unimplemented_accesses += 1;
                }
                if armed {
                    self.override_armed = false;
                }
            }
        }
        self.sync(now);
    }

    /// The packet backend's cable state.
    fn set_backend_link(&mut self, up: bool, now: Instant) {
        self.sync(now);
        if self.backend_up == up {
            return;
        }
        self.backend_up = up;
        if up {
            // Recovery starts a fresh negotiation.
            self.start_negotiation(now);
        } else {
            self.deadline = None;
            self.nego_done = false;
        }
        self.sync(now);
    }

    fn save_state(&self, now: Instant) -> toml::Value {
        StateTable::new()
            .u8("addr", self.addr)
            .u16("bmcr", self.bmcr)
            .u16("bmsr_caps", self.bmsr_caps)
            .u16("id1", self.id1)
            .u16("id2", self.id2)
            .u16("anar", self.anar)
            .u16("partner", self.partner)
            .u16("exp", self.exp)
            .u16("ext_ctl", self.ext_ctl)
            .u16("tenbt", self.tenbt)
            .u16("ext2", self.ext2)
            .bool("override_armed", self.override_armed)
            .bool("page_received", self.page_received)
            .bool("link_low_latch", self.link_low_latch)
            .bool("last_link", self.last_link)
            .bool("backend_up", self.backend_up)
            .bool("nego_done", self.nego_done)
            // Nanoseconds left plus one; zero for none.
            .u64("deadline", self.deadline.map_or(0, |d| d.saturating_duration_since(now).as_nanos() as u64 + 1))
            .u64("delay", self.delay.as_nanos() as u64)
            .u32("unimplemented_accesses", self.unimplemented_accesses)
            .build()
    }

    fn load_state(&mut self, v: &toml::Value, now: Instant) -> Result<(), String> {
        let deadline = field_u64(v, "deadline")?;
        *self = Self {
            addr: field_u8(v, "addr")?,
            bmcr: field_u16(v, "bmcr")?,
            bmsr_caps: field_u16(v, "bmsr_caps")?,
            id1: field_u16(v, "id1")?,
            id2: field_u16(v, "id2")?,
            anar: field_u16(v, "anar")?,
            partner: field_u16(v, "partner")?,
            exp: field_u16(v, "exp")?,
            ext_ctl: field_u16(v, "ext_ctl")?,
            tenbt: field_u16(v, "tenbt")?,
            ext2: field_u16(v, "ext2")?,
            override_armed: field_bool(v, "override_armed")?,
            page_received: field_bool(v, "page_received")?,
            link_low_latch: field_bool(v, "link_low_latch")?,
            last_link: field_bool(v, "last_link")?,
            backend_up: field_bool(v, "backend_up")?,
            nego_done: field_bool(v, "nego_done")?,
            deadline: (deadline != 0).then(|| now + Duration::from_nanos(deadline - 1)),
            delay: Duration::from_nanos(field_u64(v, "delay")?),
            unimplemented_accesses: field_u32(v, "unimplemented_accesses")?,
        };
        Ok(())
    }
    fn power_on(&mut self) {
        Ics1890::reset(self, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phy() -> (Ics1890, Instant) {
        let now = Instant::now();
        (Ics1890::new(1, Duration::from_millis(100)), now)
    }

    /// The identity and advertisement a driver's probe compares.
    #[test]
    fn the_reset_values_are_what_a_probe_reads() {
        let (mut p, now) = phy();
        assert_eq!(p.read(2, now), 0x0015);
        assert_eq!(p.read(3, now), 0xf423);
        assert_eq!(p.read(4, now), 0x01e1);
        assert_eq!(p.read(0, now) & 0x1000, 0x1000, "negotiation is enabled");
        assert_eq!(p.read(0x10, now) >> 6 & 0x1f, 1, "the address strap shows in the extended control");
        assert_eq!(p.read(0x13, now) & 0x4000, 0x4000, "node mode");
    }

    /// An ordinary write reaches only the writable bits of a register, and
    /// the identity registers need the command override.
    #[test]
    fn writes_stop_at_the_writable_mask() {
        let (mut p, now) = phy();
        p.write(4, 0xffff, now);
        assert_eq!(p.read(4, now), 0x01e1 & !0x21e0 | 0x21e0);
        p.write(2, 0x1234, now);
        assert_eq!(p.read(2, now), 0x0015, "the identity is read-only without the override");
        p.write(0x10, 0x8000, now);
        p.write(2, 0x1234, now);
        assert_eq!(p.read(2, now), 0x1234, "armed by a write to extended control bit 15");
        p.write(2, 0x4321, now);
        assert_eq!(p.read(2, now), 0x1234, "the override is spent by one write");
    }

    #[test]
    fn a_register_the_part_lacks_reads_zero_and_counts() {
        let (mut p, now) = phy();
        assert_eq!(p.read(0x1f, now), 0);
        assert_eq!(p.unimplemented_accesses, 1);
    }
}
