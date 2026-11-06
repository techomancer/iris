//! What a PCI bus asks of its host bridge. Everything the bridges have in
//! common lives in the bus; what differs stays in the bridge: how a CPU
//! access becomes a configuration cycle, what an absent function looks
//! like, byte lanes, DMA translation and where an interrupt output goes.

use std::sync::Arc;

use super::{PciDma, PciLanes, Space};

/// A run of a host's CPU window onto a PCI space, and how it moves bytes.
pub struct CpuWindow {
    pub space: Space,
    /// CPU-side address of the window's first byte, as `bar_runs` takes it.
    pub cpu: u64,
    /// The PCI address it opens onto.
    pub pci: u32,
    pub len: u64,
    /// The CPU address bits the host decodes (`PciBar`'s mask).
    pub mask: u64,
    pub lanes: Arc<dyn PciLanes>,
    /// Equal keys mean equal lanes, so one configuration is one table entry.
    pub lanes_key: u32,
}

pub trait PciHost: Send + Sync {
    /// The way to memory for the function at `device`.
    fn dma_for(&self, device: u8) -> Arc<dyn PciDma>;

    /// Host interrupt output `output` is now `level`: the OR of every pin
    /// and wire routed to it.
    fn set_output(&self, output: u8, level: bool);

    /// The CPU windows onto PCI space, as the host is configured now.
    fn windows(&self) -> Vec<CpuWindow>;
}

#[cfg(test)]
pub(crate) use test_host::{TestDma, TestHost};

#[cfg(test)]
mod test_host {
    use std::sync::{Arc, Mutex};

    use super::{CpuWindow, PciHost};
    use crate::pci::{PciBusDevice, PciDma, PciDmaStatus, PciLanes};

    /// A host bridge for the bus's own tests: it records what the bus
    /// drives into it and hands out the windows a test sets.
    pub struct TestHost {
        outputs: Mutex<Vec<(u8, bool)>>,
        windows: Mutex<Vec<CpuWindow>>,
        lanes: Arc<TestLanes>,
        dma: Mutex<Vec<(u8, Arc<TestDma>)>>,
    }

    /// Lanes that move nothing and record every access they carried.
    pub struct TestLanes {
        log: Mutex<Vec<(usize, u32, usize)>>,
    }

    impl PciLanes for TestLanes {
        fn read(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize) -> u64 {
            self.log.lock().unwrap().push((bar, off, size));
            f.bar_read(bar, off, size) as u64
        }

        fn write(&self, f: &dyn PciBusDevice, bar: usize, off: u32, size: usize, val: u64) {
            self.log.lock().unwrap().push((bar, off, size));
            f.bar_write(bar, off, val as u32, size);
        }
    }

    /// The way to memory the test host gives one device: it reaches nothing.
    pub struct TestDma {
        pub device: u8,
    }

    impl PciDma for TestDma {
        fn dma_read(&self, _addr: u64, buf: &mut [u8]) -> PciDmaStatus {
            buf.fill(0);
            PciDmaStatus::MasterAbort
        }

        fn dma_write(&self, _addr: u64, _data: &[u8]) -> PciDmaStatus {
            PciDmaStatus::MasterAbort
        }
    }

    impl TestHost {
        pub fn new() -> Arc<Self> {
            Arc::new(Self {
                outputs: Mutex::new(Vec::new()),
                windows: Mutex::new(Vec::new()),
                lanes: Arc::new(TestLanes { log: Mutex::new(Vec::new()) }),
                dma: Mutex::new(Vec::new()),
            })
        }

        /// Every `set_output` call, in order.
        pub fn outputs(&self) -> Vec<(u8, bool)> {
            self.outputs.lock().unwrap().clone()
        }

        pub fn set_windows(&self, w: Vec<CpuWindow>) {
            *self.windows.lock().unwrap() = w;
        }

        /// The lanes this host's windows should carry, so a test sees what they moved.
        pub fn lanes(&self) -> Arc<dyn PciLanes> {
            self.lanes.clone()
        }

        /// The `(bar, offset, size)` of every access the host's lanes carried.
        pub fn lanes_log(&self) -> Vec<(usize, u32, usize)> {
            self.lanes.log.lock().unwrap().clone()
        }

        /// The DMA object handed to `device`, if one was asked for.
        pub fn dma_of(&self, device: u8) -> Option<Arc<TestDma>> {
            self.dma.lock().unwrap().iter().find(|(d, _)| *d == device).map(|(_, d)| d.clone())
        }
    }

    impl PciHost for TestHost {
        fn dma_for(&self, device: u8) -> Arc<dyn PciDma> {
            let d = Arc::new(TestDma { device });
            self.dma.lock().unwrap().push((device, d.clone()));
            d
        }

        fn set_output(&self, output: u8, level: bool) {
            self.outputs.lock().unwrap().push((output, level));
        }

        fn windows(&self) -> Vec<CpuWindow> {
            self.windows
                .lock()
                .unwrap()
                .iter()
                .map(|w| CpuWindow { lanes: w.lanes.clone(), ..*w })
                .collect()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_host_records_outputs_in_order() {
            let h = TestHost::new();
            h.set_output(3, true);
            h.set_output(3, false);
            h.set_output(1, true);
            assert_eq!(h.outputs(), vec![(3, true), (3, false), (1, true)]);
        }
    }
}
