use std::fs;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};

use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device};

const PROM_BASE: u32 = 0x1FC00000;
const PROM_SIZE: u32 = 1024 * 1024; // 1MB
const PROM_FILENAME: &str = "070-9101-011.bin";

struct PromInner {
    data: Vec<u32>,
    clock: AtomicU64,
    running: AtomicBool,
}

pub struct PromPort {
    inner: Arc<PromInner>,
}

pub struct Prom {
    inner: Arc<PromInner>,
}

impl Prom {
    pub fn new() -> Self {
        let path = Path::new(PROM_FILENAME);
        let bytes = fs::read(path).unwrap_or_else(|_| {
            eprintln!("Warning: Could not read PROM file: {}", PROM_FILENAME);
            Vec::new()
        });

        Self::from_bytes(&bytes)
    }

    /// Load PROM from `path`; if that fails, fall back to the embedded binary.
    pub fn from_file_or_embedded(path: &str) -> Self {
        match fs::read(path) {
            Ok(bytes) => {
                println!("Loaded PROM from {}", path);
                Self::from_bytes(&bytes)
            }
            Err(e) => {
                eprintln!("Warning: Could not read PROM file '{}': {} — using embedded PROM", path, e);
                Self::from_bytes(&crate::prombin::PROM0709101011)
            }
        }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        // Pack bytes into u32 (Big Endian)
        let mut data = Vec::with_capacity((bytes.len() + 3) / 4);
        for chunk in bytes.chunks(4) {
            let mut buf = [0u8; 4];
            for (i, &b) in chunk.iter().enumerate() {
                buf[i] = b;
            }
            data.push(u32::from_be_bytes(buf));
        }

        Prom {
            inner: Arc::new(PromInner {
                data,
                clock: AtomicU64::new(0),
                running: AtomicBool::new(false),
            }),
        }
    }

    pub fn get_port(&self) -> PromPort {
        PromPort {
            inner: self.inner.clone(),
        }
    }
}

impl Device for Prom {
    fn step(&self, _cycles: u64) {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);
    }

    fn stop(&self) {
        self.inner.running.store(false, Ordering::SeqCst);
    }

    fn start(&self) {
        self.inner.running.store(true, Ordering::SeqCst);
    }

    fn is_running(&self) -> bool {
        self.inner.running.load(Ordering::SeqCst)
    }

    fn get_clock(&self) -> u64 {
        self.inner.clock.load(Ordering::Relaxed)
    }
}

impl BusDevice for PromPort {
    fn read8(&self, addr: u32) -> BusRead8 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);

        if addr < PROM_BASE || addr >= PROM_BASE + PROM_SIZE {
            return BusRead8::ok(0xFF);
        }

        let offset = (addr - PROM_BASE) as usize;
        let word_index = offset / 4;
        let byte_offset = offset % 4;

        if word_index < self.inner.data.len() {
            let word = self.inner.data[word_index];
            let byte = ((word >> (24 - byte_offset * 8)) & 0xFF) as u8;
            BusRead8::ok(byte)
        } else {
            BusRead8::ok(0xFF)
        }
    }

    fn write8(&self, _addr: u32, _val: u8) -> u32 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);
        BUS_OK
    }

    fn read16(&self, addr: u32) -> BusRead16 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);

        if addr < PROM_BASE || addr >= PROM_BASE + PROM_SIZE {
            return BusRead16::ok(0xFFFF);
        }

        let offset = (addr - PROM_BASE) as usize;
        let word_index = offset / 4;
        let byte_offset = offset % 4;

        if word_index < self.inner.data.len() {
            let word = self.inner.data[word_index];
            let halfword = ((word >> (16 - byte_offset * 8)) & 0xFFFF) as u16;
            BusRead16::ok(halfword)
        } else {
            BusRead16::ok(0xFFFF)
        }
    }

    fn write16(&self, _addr: u32, _val: u16) -> u32 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);
        BUS_OK
    }

    fn read32(&self, addr: u32) -> BusRead32 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);

        if addr < PROM_BASE || addr >= PROM_BASE + PROM_SIZE {
            return BusRead32::ok(0xFFFFFFFF);
        }

        let offset = (addr - PROM_BASE) as usize;
        let index = offset / 4;

        if index < self.inner.data.len() {
            BusRead32::ok(self.inner.data[index])
        } else {
            BusRead32::ok(0xFFFFFFFF)
        }
    }

    fn write32(&self, _addr: u32, _val: u32) -> u32 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);
        BUS_OK
    }

    fn read64(&self, addr: u32) -> BusRead64 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);

        // Read two consecutive 32-bit words
        let high = { let _r = self.read32(addr); if _r.is_ok() { _r.data as u64 } else { 0xFFFFFFFF } };
        let low = { let _r = self.read32(addr + 4); if _r.is_ok() { _r.data as u64 } else { 0xFFFFFFFF } };

        BusRead64::ok((high << 32) | low)
    }

    fn write64(&self, _addr: u32, _val: u64) -> u32 {
        self.inner.clock.fetch_add(1, Ordering::Relaxed);
        BUS_OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prom_behavior() {
        // Mock data: F0 0B F0 00 (Big Endian u32: 0xF00BF000)
        let data = vec![0xF0, 0x0B, 0xF0, 0x00];
        let prom = Prom::from_bytes(&data);
        let port = prom.get_port();

        // 1. Check if [0] is F0 0B F0 00
        { let _r = port.read32(PROM_BASE); assert!(_r.is_ok(), "Expected ok read"); assert_eq!(_r.data, 0xF00BF000, "First word mismatch"); }

        // 2. Check reads outside range return 0xFFFFFFFF
        // Inside mapped range but outside data length
        { let _r = port.read32(PROM_BASE + 4); assert!(_r.is_ok(), "Expected ok read"); assert_eq!(_r.data, 0xFFFFFFFF, "Read past data end mismatch"); }
        // Outside mapped range
        { let _r = port.read32(PROM_BASE - 4); assert!(_r.is_ok(), "Expected ok read"); assert_eq!(_r.data, 0xFFFFFFFF, "Read before base mismatch"); }

        // 3. Check writes do nothing
        assert_eq!(port.write32(PROM_BASE, 0xDEADBEEF), BUS_OK);
        // Verify data is unchanged
        { let _r = port.read32(PROM_BASE); assert!(_r.is_ok(), "Expected ok read"); assert_eq!(_r.data, 0xF00BF000, "Write modified ROM data"); }
    }

    #[test]
    fn test_prom_disassembly() {
        use crate::mips_dis;

        let prom = Prom::new();
        let port = prom.get_port();

        println!("\nDisassembling first 256 words from PROM:\n");

        // PROM is at physical address 0x1FC00000
        // But CPU starts at 0xBFC00000 (KSEG1 uncached mapping)
        const KSEG1_OFFSET: u32 = 0xA0000000;

        for i in 0..256 {
            let phys_addr = PROM_BASE + (i * 4);
            let kseg1_addr = phys_addr + KSEG1_OFFSET; // 0xBFC00000 + offset

            let r = port.read32(phys_addr);
            if r.is_ok() {
                let disasm = mips_dis::disassemble(r.data, kseg1_addr as u64, None);
                println!("0x{:08x}: 0x{:08x}: {}", kseg1_addr, r.data, disasm);
            } else {
                println!("0x{:08x}: ERROR reading", kseg1_addr);
            }
        }
    }
}