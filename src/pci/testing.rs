//! Test doubles for the bus: a function that records what it is asked, and
//! a host that records what it is told.

use std::sync::{Arc, Mutex};

use super::{Bar, PciConfig, PciDma, PciDmaStatus, PciFunction, PciFunctionState, PciHost};

/// A function with the given BARs whose BAR 0 is a little-endian register
/// file, and which remembers the BAR calls it was given and, in a shared
/// log, when it was reset. It does not touch its interrupt line on reset, so
/// a test can see that the bus leaves the line alone.
pub(crate) struct TestCard {
    pci: PciFunctionState,
    mem: Mutex<[u8; 0x1000]>,
    calls: Mutex<Vec<(u32, usize)>>,
    name: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

impl TestCard {
    pub fn new(bars: [Bar; 6], pin: u8) -> Arc<Self> {
        Self::logged("card", bars, pin, &Arc::default())
    }

    /// A card that writes `"<name> cmd=<command register>"` to `log` when it
    /// is reset: the command shows whether its header was reset first.
    pub fn logged(name: &'static str, bars: [Bar; 6], pin: u8, log: &Arc<Mutex<Vec<String>>>) -> Arc<Self> {
        Arc::new(Self {
            pci: PciFunctionState::new(PciConfig::new(0x1234, 0x5678, 0, 1, bars, pin)),
            mem: Mutex::new([0; 0x1000]),
            calls: Mutex::new(Vec::new()),
            name,
            log: log.clone(),
        })
    }

    /// The register file.
    pub fn mem(&self) -> Vec<u8> {
        self.mem.lock().unwrap().to_vec()
    }

    /// The (offset, width) of each BAR call so far.
    pub fn calls(&self) -> Vec<(u32, usize)> {
        self.calls.lock().unwrap().clone()
    }
}

impl PciFunction for TestCard {
    fn state(&self) -> &PciFunctionState {
        &self.pci
    }

    fn bar_read(&self, _bar: usize, off: u32, width: usize) -> u32 {
        self.calls.lock().unwrap().push((off, width));
        let m = self.mem.lock().unwrap();
        m[off as usize..off as usize + width].iter().rev().fold(0, |a, &b| a << 8 | b as u32)
    }

    fn bar_write(&self, _bar: usize, off: u32, val: u32, width: usize) {
        self.calls.lock().unwrap().push((off, width));
        self.mem.lock().unwrap()[off as usize..off as usize + width].copy_from_slice(&val.to_le_bytes()[..width]);
    }


    fn reset(&self) {
        self.log.lock().unwrap().push(format!("{} cmd={:#x}", self.name, self.pci.config().command()));
    }
}

/// A DMA identity whose reads fill the buffer with the requester's number.
pub(crate) struct TestDma(pub u8);

impl PciDma for TestDma {
    fn dma_read(&self, _addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        buf.fill(self.0);
        PciDmaStatus::Ok
    }

    fn dma_write(&self, _addr: u64, _data: &[u8]) -> PciDmaStatus {
        PciDmaStatus::Ok
    }
}

/// A host that logs each host IRQ change.
#[derive(Default)]
pub(crate) struct TestHost {
    irqs: Mutex<Vec<(u8, bool)>>,
}

impl TestHost {
    pub fn irqs(&self) -> Vec<(u8, bool)> {
        self.irqs.lock().unwrap().clone()
    }
}

impl PciHost for TestHost {
    fn dma_for(&self, requester: u8) -> Arc<dyn PciDma> {
        Arc::new(TestDma(requester))
    }

    fn set_irq(&self, irq: u8, level: bool) {
        self.irqs.lock().unwrap().push((irq, level));
    }
}
