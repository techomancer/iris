//! The chip's view of bus-master memory, shared by every child that masters
//! the bus.

use std::sync::Arc;

use crate::pci::{PciDma, PciDmaStatus};

/// The chip's view of bus-master memory. Internal byte `a` sits at bus
/// address `a ^ 3` (a word is big-endian on the bus with the bridge's swap
/// clear), so rings, descriptors and frames keep their layout whatever the
/// bridge then does to them. This is what the IOC3's Ethernet block and
/// serial and parallel engines are handed, and the IOC4's serial engine.
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

    /// Writes only the bytes of `data`. A word the span covers in part goes
    /// to the bus as a write of just those bytes, so another master's bytes
    /// in the same word are never read and written back; the whole words in
    /// between go as one transfer.
    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        if data.is_empty() {
            return PciDmaStatus::Ok;
        }
        if addr.checked_add(data.len() as u64).is_none() {
            return PciDmaStatus::MasterAbort;
        }
        let (mut at, mut rest) = (addr, data);
        if at & 3 != 0 || rest.len() < 4 {
            let n = rest.len().min(4 - (at & 3) as usize);
            let status = self.write_partial(at, &rest[..n]);
            if !status.is_ok() {
                return status;
            }
            at += n as u64;
            rest = &rest[n..];
        }
        let body = rest.len() & !3;
        if body != 0 {
            let mut tmp = vec![0u8; body];
            for (t, d) in tmp.chunks_exact_mut(4).zip(rest.chunks_exact(4)) {
                t.copy_from_slice(&[d[3], d[2], d[1], d[0]]);
            }
            let status = self.raw.dma_write(at, &tmp);
            if !status.is_ok() {
                return status;
            }
            at += body as u64;
            rest = &rest[body..];
        }
        if rest.is_empty() {
            PciDmaStatus::Ok
        } else {
            self.write_partial(at, rest)
        }
    }
}

impl Ioc3Dma {
    /// `data`, which lies inside one word from internal byte `addr`, as one
    /// bus write of the bytes it covers: they sit in reverse order, from bus
    /// offset 4 - end of the span in the word.
    fn write_partial(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let first = (addr & 3) as usize;
        debug_assert!(first + data.len() <= 4);
        let mut bytes = [0u8; 4];
        for (k, b) in data.iter().rev().enumerate() {
            bytes[k] = *b;
        }
        self.raw.dma_write((addr & !3) + (4 - first - data.len()) as u64, &bytes[..data.len()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    /// Bus memory that remembers the extent of every write and refuses reads.
    struct Logged(Mutex<Vec<u8>>, Mutex<Vec<(u64, usize)>>);

    impl PciDma for Logged {
        fn dma_read(&self, _addr: u64, _buf: &mut [u8]) -> PciDmaStatus {
            panic!("a write reads no word of the bus");
        }
        fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
            self.1.lock().push((addr, data.len()));
            self.0.lock()[addr as usize..addr as usize + data.len()].copy_from_slice(data);
            PciDmaStatus::Ok
        }
    }

    /// A write puts on the bus exactly the bytes it carries, with no read first, so
    /// a neighbour's byte in the same word is never overwritten, whatever the
    /// alignment and length.
    #[test]
    fn a_write_touches_no_byte_outside_its_span() {
        for len in [1, 2, 3, 4, 5, 6, 7, 8, 9, 129, 200] {
            for lead in 0..4u64 {
                let bus = Arc::new(Logged(Mutex::new(vec![0; 512]), Mutex::new(Vec::new())));
                let dma = Ioc3Dma::new(bus.clone());
                let addr = 64 + lead;
                let data: Vec<u8> = (0..len).map(|i| 1 + i as u8).collect();
                assert!(dma.dma_write(addr, &data).is_ok());
                let mut covered = vec![0u8; 512];
                for &(a, n) in bus.1.lock().iter() {
                    for b in a..a + n as u64 {
                        covered[b as usize] += 1;
                    }
                }
                // Each byte lands at its place in the bus word, not just inside it.
                let bytes = bus.0.lock();
                for (i, d) in data.iter().enumerate() {
                    assert_eq!(bytes[(addr as usize + i) ^ 3], *d, "len {len} lead {lead}: byte {i}");
                }
                for at in 0..512usize {
                    let inside = (0..len).any(|i| (addr as usize + i) ^ 3 == at);
                    assert_eq!(covered[at], inside as u8, "len {len} lead {lead}: bus byte {at}");
                }
            }
        }
    }
}
