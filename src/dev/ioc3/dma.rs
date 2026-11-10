//! The chip's view of bus-master memory, shared by every child that masters
//! the bus.

use std::sync::Arc;

use crate::pci::{PciDma, PciDmaStatus};

/// The chip's view of bus-master memory. Internal byte `a` sits at bus
/// address `a ^ 3` (a word is big-endian on the bus with the bridge's swap
/// clear), so rings, descriptors and frames keep their layout whatever the
/// bridge then does to them. This is what the Ethernet block and the serial
/// and parallel engines are handed.
pub struct Ioc3Dma {
    raw: Arc<dyn PciDma>,
}

impl Ioc3Dma {
    pub fn new(raw: Arc<dyn PciDma>) -> Self {
        Self { raw }
    }
}

impl PciDma for Ioc3Dma {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        if buf.is_empty() {
            return PciDmaStatus::Ok;
        }
        let start = addr & !3;
        // A guest-chosen base can sit at the top of the address space.
        let Some(end) = addr.checked_add(buf.len() as u64 + 3).map(|e| e & !3) else {
            buf.fill(0);
            return PciDmaStatus::MasterAbort;
        };
        let mut tmp = vec![0u8; (end - start) as usize];
        let status = self.raw.dma_read(start, &mut tmp);
        for (i, b) in buf.iter_mut().enumerate() {
            *b = tmp[((addr - start) as usize + i) ^ 3];
        }
        status
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        if data.is_empty() {
            return PciDmaStatus::Ok;
        }
        let start = addr & !3;
        let Some(end) = addr.checked_add(data.len() as u64 + 3).map(|e| e & !3) else {
            return PciDmaStatus::MasterAbort;
        };
        let mut tmp = vec![0u8; (end - start) as usize];
        if addr & 3 != 0 || data.len() & 3 != 0 {
            // Bytes outside the span keep their value: read the words the
            // span touches, overlay it, and write them back as one transfer.
            if !self.raw.dma_read(start, &mut tmp).is_ok() {
                return PciDmaStatus::MasterAbort;
            }
        }
        for (i, b) in data.iter().enumerate() {
            tmp[((addr - start) as usize + i) ^ 3] = *b;
        }
        self.raw.dma_write(start, &tmp)
    }
}
