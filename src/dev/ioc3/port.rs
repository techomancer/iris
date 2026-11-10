//! How IOC3 reaches its Ethernet, which lives in its own file.

use std::sync::Arc;
use crate::pci::PciDma;

/// The Ethernet half of IOC3, as the rest of the chip sees it. Offsets are
/// within IOC3's register BAR, so they match the published register map.
pub trait Ioc3EthPort: Send + Sync {
    fn read32(&self, off: u32) -> u32;
    fn write32(&self, off: u32, val: u32);
    /// IOC3 gives the Ethernet its way to memory and a level for the
    /// Ethernet's bit in IOC3's interrupt status.
    fn connect(&self, dma: Arc<dyn PciDma>, irq: Box<dyn Fn(bool) + Send + Sync>);
}
