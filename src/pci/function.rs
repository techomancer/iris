//! A function as the bus sees it: the trait a function implements, the
//! state the bus fills in when it attaches one, where it sits, and how a
//! bus master reaches memory.

use std::sync::{Arc, OnceLock};

use crate::irq::IrqLine;
use super::{PciBus, PciConfig};

/// What the bus sees of one function: its configuration header, and the
/// interrupt line and the way to memory the bus gives it when it is
/// attached, each set once. A function that raises no interrupt (RAD1)
/// never has a line.
pub struct PciFunctionState {
    config: PciConfig,
    irq: OnceLock<IrqLine>,
    dma: OnceLock<Arc<dyn PciDma>>,
}

impl PciFunctionState {
    pub fn new(config: PciConfig) -> Self {
        Self { config, irq: OnceLock::new(), dma: OnceLock::new() }
    }

    pub fn config(&self) -> &PciConfig {
        &self.config
    }

    pub fn irq(&self) -> Option<&IrqLine> {
        self.irq.get()
    }

    pub fn dma(&self) -> Option<&Arc<dyn PciDma>> {
        self.dma.get()
    }

    /// The bus wires the function's interrupt pin to `line`. Only the first
    /// line counts.
    pub fn connect_irq(&self, line: IrqLine) {
        let _ = self.irq.set(line);
    }

    /// The bus gives a function that masters the bus its way to memory.
    /// Only the first counts.
    pub fn connect_dma(&self, dma: Arc<dyn PciDma>) {
        let _ = self.dma.set(dma);
    }
}

/// A PCI function, as a host bridge sees it.
///
/// BAR accesses arrive with the offset and value the device sees: whatever
/// the bridge does to byte order it has already done.
pub trait PciFunction: Send + Sync {
    /// The function's header, and the interrupt line and DMA the bus gave it.
    fn state(&self) -> &PciFunctionState;

    /// What a configuration read of dword `index` returns to the guest. The
    /// header as stored, unless the chip decodes fewer address lines than
    /// the bus gives it (IOC3 repeats its first 32 bytes).
    fn config_dword(&self, index: usize) -> u32 {
        self.state().config().read(index)
    }

    /// A read of `width` bytes (1, 2 or 4) at `off` within BAR `bar`.
    fn bar_read(&self, bar: usize, off: u32, width: usize) -> u32;

    fn bar_write(&self, bar: usize, off: u32, val: u32, width: usize);

    /// The bus has attached the function and filled in its state's
    /// interrupt line and DMA. A function that builds on either does it
    /// here: IOC3 wraps its DMA and drives its pin, virtio hands both to its
    /// shared core, and the SCSI controllers drive their pins.
    fn attached(&self) {}

    /// A configuration write of the whole dword `reg`, offered before the
    /// header sees it. A function whose chip computes the register (the
    /// indirect windows of a NIC) takes it here, returns true, and the header
    /// stores nothing; every other function leaves it to the header. Narrower
    /// writes always take the header's path. Called with no header lock held.
    fn config_write(&self, _reg: usize, _val: u32) -> bool {
        false
    }

    /// The bus wrote configuration dword `reg` of this function's header and
    /// it changed from `old` to `new`. A PCI-to-PCI bridge acts on bridge
    /// control here; every other function has nothing to do.
    fn config_written(&self, _reg: usize, _old: u32, _new: u32) {}

    /// The bus behind this function, if it is a PCI-to-PCI bridge.
    fn secondary(&self) -> Option<&PciBus> {
        None
    }

    /// RST# reaches the function after the bus has reset its header: drop
    /// the interrupt line and return to power-on state. A function never
    /// resets its own header, so the bus's decode table cannot go stale.
    fn reset(&self) {}
}

/// Where a function sits on a bus: its device number and its function number.
/// Orders by device, then function, the order RST# visits them in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DevFn {
    pub device: u8,
    pub function: u8,
}

/// How a bus master's access ended.
#[must_use]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PciDmaStatus {
    Ok,
    /// Some of the bytes had no target: reads return zero for them and
    /// writes drop them. A bridge may also log it and raise its error
    /// interrupt.
    MasterAbort,
}

impl PciDmaStatus {
    pub fn is_ok(self) -> bool {
        self == PciDmaStatus::Ok
    }
}

/// Memory as a bus master reaches it. The bridge decides what a PCI address
/// is and in which order its bytes arrive; the function sees bytes at bus
/// addresses, as it would on the bus.
///
/// A function calls only the object the bus left in its state and never
/// learns whether it is shared. The O2's MACE hands every function one
/// object (its translation is fixed); a bridge whose translation depends on the
/// mastering device, as the Octane's does, hands each its own.
pub trait PciDma: Send + Sync {
    /// `buf.len()` bytes from bus address `addr`, 64 bits wide for masters
    /// that use dual address cycles.
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus;

    /// `data` to bus address `addr`.
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::{Bar, PCI_INTA};

    struct Func(PciFunctionState);
    impl PciFunction for Func {
        fn state(&self) -> &PciFunctionState { &self.0 }
        fn bar_read(&self, _: usize, off: u32, _: usize) -> u32 { off }
        fn bar_write(&self, _: usize, _: u32, _: u32, _: usize) {}
    }

    fn func() -> Func {
        Func(PciFunctionState::new(PciConfig::new(0x9004, 0x8078, 0x0100_00, 1, [Bar::Unused; 6], PCI_INTA)))
    }

    #[test]
    fn a_state_takes_its_line_and_dma_once() {
        struct Dma;
        impl PciDma for Dma {
            fn dma_read(&self, _: u64, _: &mut [u8]) -> PciDmaStatus { PciDmaStatus::Ok }
            fn dma_write(&self, _: u64, _: &[u8]) -> PciDmaStatus { PciDmaStatus::Ok }
        }
        let (a, b) = (crate::irq::LevelProbe::line(), crate::irq::LevelProbe::line());
        let f = func();
        assert!(f.state().irq().is_none() && f.state().dma().is_none());
        f.state().connect_irq(a.1);
        f.state().connect_irq(b.1);
        f.state().connect_dma(Arc::new(Dma));
        f.state().irq().unwrap().set(true);
        assert!(a.0.level() && !b.0.level(), "the second line is ignored");
        assert!(f.state().dma().is_some());
        assert!(func().state().irq().is_none(), "a function with no interrupt never has a line");
    }

    #[test]
    fn devfn_orders_by_device_then_function() {
        let a = |device, function| DevFn { device, function };
        let mut v = [a(3, 0), a(1, 1), a(1, 0)];
        v.sort();
        assert_eq!(v, [a(1, 0), a(1, 1), a(3, 0)]);
    }
}
