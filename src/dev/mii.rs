//! The management interface of an Ethernet transceiver (PHY) as a MAC sees
//! it over MII, and what the MAC needs from the chip behind it.

/// The management and packet-path behavior that varies between IOC3 PHYs.
/// Times are machine nanoseconds.
pub trait MiiPhy: Send {
    fn address(&self) -> u8;
    fn deadline(&self) -> Option<u64>;
    fn wire_enabled(&self) -> bool;
    fn in_loopback(&self) -> bool;
    fn sync(&mut self, now: u64);
    fn read(&mut self, register: u8, now: u64) -> u16;
    fn write(&mut self, register: u8, value: u16, now: u64);
    fn set_backend_link(&mut self, up: bool, now: u64);
    fn power_on(&mut self, now: u64);
    /// The transceiver's registers and negotiation progress for a snapshot;
    /// its deadline is saved as an absolute machine time.
    fn save_state(&self) -> toml::Value;
    fn load_state(&mut self, v: &toml::Value) -> Result<(), String>;
}
