//! Test doubles shared by the Bridge, XIO PCI board and XBridge tests: the
//! host's memory and its interrupt side.

use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead32, BusRead8, BUS_ERR, BUS_OK};
use parking_lot::Mutex;

/// HEART stand-in: the interrupt packets it received, in order. A line is
/// the ISR bit, so `set_line(vector, set)` is a packet.
#[derive(Default)]
pub(crate) struct Heart(Mutex<Vec<(u8, bool)>>);

impl InterruptController for Heart {
    fn set_line(&self, line: u32, level: bool) {
        self.0.lock().push((line as u8, level));
    }
}

impl Heart {
    pub fn take(&self) -> Vec<(u8, bool)> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// Host memory as a set of (base, bytes) regions; the rest of the address
/// space is not memory and errors. Big-endian like the bus: `read32` at `a`
/// is the bytes `a..a + 4`.
pub(crate) struct Memory(Mutex<Vec<(u64, Vec<u8>)>>);

impl Memory {
    /// Zeroed regions of `(base, len)`.
    pub fn new(regions: &[(u64, usize)]) -> Self {
        Memory(Mutex::new(regions.iter().map(|&(base, len)| (base, vec![0; len])).collect()))
    }

    /// One region at 0, every byte `fill`.
    pub fn filled(len: usize, fill: u8) -> Self {
        Memory(Mutex::new(vec![(0, vec![fill; len])]))
    }

    /// The bytes `a..a + n`, which must lie in one region.
    pub fn peek(&self, a: u64, n: usize) -> Vec<u8> {
        self.with(a, n, |b| Some(b.to_vec())).unwrap()
    }

    pub fn poke(&self, a: u64, data: &[u8]) {
        self.with(a, data.len(), |b| Some(b.copy_from_slice(data))).unwrap()
    }

    fn with<T>(&self, a: u64, n: usize, f: impl FnOnce(&mut [u8]) -> Option<T>) -> Option<T> {
        let mut regions = self.0.lock();
        let (base, bytes) = regions.iter_mut().find(|(base, bytes)| a >= *base && a - *base + n as u64 <= bytes.len() as u64)?;
        let off = (a - *base) as usize;
        f(&mut bytes[off..off + n])
    }
}

impl BusDevice for Memory {
    fn read8(&self, a: u64) -> BusRead8 {
        self.with(a, 1, |b| Some(b[0])).map_or(BusRead8::err(), BusRead8::ok)
    }

    fn write8(&self, a: u64, v: u8) -> u32 {
        self.with(a, 1, |b| Some(b[0] = v)).map_or(BUS_ERR, |()| BUS_OK)
    }

    fn read32(&self, a: u64) -> BusRead32 {
        self.with(a, 4, |b| Some(u32::from_be_bytes((&*b).try_into().unwrap()))).map_or(BusRead32::err(), BusRead32::ok)
    }

    fn write32(&self, a: u64, v: u32) -> u32 {
        self.with(a, 4, |b| Some(b.copy_from_slice(&v.to_be_bytes()))).map_or(BUS_ERR, |()| BUS_OK)
    }
}
