//! The management interface of an Ethernet transceiver (PHY) as a MAC sees
//! it over MII, and what the MAC needs from the chip behind it.

use std::time::Instant;

/// The management and packet-path behavior that varies between IOC3 PHYs.
pub trait MiiPhy: Send {
    fn address(&self) -> u8;
    fn deadline(&self) -> Option<Instant>;
    fn wire_enabled(&self) -> bool;
    fn in_loopback(&self) -> bool;
    fn sync(&mut self, now: Instant);
    fn read(&mut self, register: u8, now: Instant) -> u16;
    fn write(&mut self, register: u8, value: u16, now: Instant);
    fn set_backend_link(&mut self, up: bool, now: Instant);
    fn power_on(&mut self);
    /// The transceiver's registers and negotiation progress for a snapshot;
    /// its deadline is saved as the time it had left.
    fn save_state(&self, now: Instant) -> toml::Value;
    fn load_state(&mut self, v: &toml::Value, now: Instant) -> Result<(), String>;
}
