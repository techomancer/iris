//! PCI: the configuration header every function has, and the trait a
//! function implements to sit behind a host bridge.
//!
//! The host bridge owns everything machine-specific: how configuration
//! cycles are generated, where the CPU sees PCI memory, byte order, and
//! which interrupt input each slot's pins are wired to. A function only
//! sees configuration registers, accesses to its BARs, and an interrupt
//! line for its pin.

mod config;
mod function;

pub use config::{
    PCI_COMMAND_IO, PCI_COMMAND_MEMORY, PCI_INTA, Bar, BarAt, BarRun, BarRunSink, PciBar, PciConfig, PciConfigPolicy,
    PciLanes, PciRegisterPolicy, Space, bar_runs,
};
pub use function::{DevFn, PciDma, PciDmaStatus, PciFunction, PciFunctionState};
