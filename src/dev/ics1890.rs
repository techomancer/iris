//! The ICS1890 transceiver fitted to Octane's IOC3.
//!
//! This module is shared by the Octane IOC3 Ethernet model. It was moved here
//! from `ioc3_eth.rs`; its management-register behavior follows the public
//! datasheet and the existing Octane behavior.
//!

use std::time::{Duration, Instant};

use crate::dev::ioc3_eth::MiiPhy;

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

    pub fn address(&self) -> u8 {
        self.addr
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

    /// When a negotiation in progress completes, if one is.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Advance the negotiation deadline and note link transitions. Reads never
    /// start or postpone negotiation; this only observes the clock.
    pub fn sync(&mut self, now: Instant) {
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

    /// The packet backend's cable state.
    pub fn set_backend_link(&mut self, up: bool, now: Instant) {
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

    /// BMCR loopback turns frames round inside the part. Isolate and
    /// power-down take precedence.
    pub fn in_loopback(&self) -> bool {
        self.bmcr & 0x4000 != 0 && !self.power_down() && !self.isolated()
    }

    /// Frames may cross the MII at all.
    pub fn data_path(&self) -> bool {
        !self.power_down() && !self.isolated()
    }

    /// Frames may go to or come from the packet backend.
    pub fn wire_enabled(&self) -> bool {
        self.data_path() && self.link()
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

    pub fn read(&mut self, r: u8, now: Instant) -> u16 {
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

    pub fn write(&mut self, r: u8, val: u16, now: Instant) {
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
}

impl MiiPhy for Ics1890 {
    fn address(&self) -> u8 { Ics1890::address(self) }
    fn deadline(&self) -> Option<Instant> { Ics1890::deadline(self) }
    fn wire_enabled(&self) -> bool { Ics1890::wire_enabled(self) }
    fn in_loopback(&self) -> bool { Ics1890::in_loopback(self) }
    fn sync(&mut self, now: Instant) { Ics1890::sync(self, now) }
    fn read(&mut self, register: u8, now: Instant) -> u16 { Ics1890::read(self, register, now) }
    fn write(&mut self, register: u8, value: u16, now: Instant) { Ics1890::write(self, register, value, now) }
    fn set_backend_link(&mut self, up: bool, now: Instant) { Ics1890::set_backend_link(self, up, now) }
    fn power_on(&mut self) { Ics1890::reset(self, Instant::now()) }
}
