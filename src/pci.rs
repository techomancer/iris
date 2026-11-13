//! PCI: the configuration header every function has, and the trait a
//! function implements to sit behind a host bridge.
//!
//! The host bridge owns everything machine-specific: how configuration
//! cycles are generated, where the CPU sees PCI memory, byte order, and
//! which interrupt input each slot's pins are wired to. A function only
//! sees configuration registers, accesses to its BARs, and an interrupt
//! line for its pin.

mod bus;
mod config;
mod decode;
mod function;
mod host;
mod intx;
mod pci_to_pci_bridge;
#[cfg(test)]
pub(crate) mod testing;

pub use bus::PciBus;
pub use config::{Bar, PciConfig, PciConfigPolicy, PciRegisterPolicy, Space, PCI_COMMAND_IO, PCI_COMMAND_MEMORY, PCI_INTA};
pub use decode::{BarMapping, BarMappingSink, CpuWindow};
pub use function::{DevFn, PciDma, PciDmaStatus, PciFunction, PciFunctionState};
pub use host::{BarDevice, BarDevices, ByteOrder, PciHost};
pub use intx::{BoardIrq, IntxRouting};
pub use pci_to_pci_bridge::PciToPciBridge;
