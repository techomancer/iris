//! The Octane's PROM flash: the part behind the Bridge's external flash
//! select, at physical 0x1fc00000.
//!
//! One 1 MB part, 16 bits wide, in sixteen 64 KB segments: the recovery
//! PROM in 0-2, the main PROM in 3-14 and the PROM's data store (PDS,
//! where `setenv` settings and logs go) in 15. Reads of any width return
//! the array, or the ID or status after the matching command; a program
//! command ANDs a halfword in, an erase sets a segment to ones. Changes
//! are written back to the flash file, if there is one, so the PDS
//! survives a restart.
//!
//! Offsets are within the part; the bus subtracts the window's base.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_OK};

pub const FLASH_SIZE: u32 = 0x10_0000;
const FLASH_SEG: usize = 64 * 1024;
const SFLASH_MFG_ID: u16 = 0x00b0;
const SFLASH_DEV_ID: u16 = 0x66a8;
const SFLASH_STATUS_READY: u16 = 0x80;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlashMode {
    Read,
    Id,
    Status,
    Program,
    Erase,
}

struct FlashState {
    data: Vec<u8>,
    mode: FlashMode,
    /// Changed since the last save.
    dirty: bool,
}

pub struct Flash {
    s: Mutex<FlashState>,
    path: Option<std::path::PathBuf>,
    /// jitv2's code-generation counter for the whole part.
    gen: AtomicU64,
}

impl Flash {
    /// From `path` if it holds a flash image, else from `image` (and
    /// `path`, when given, gets a copy).
    pub fn new(image: &[u8], path: Option<std::path::PathBuf>) -> Self {
        let saved = path.as_ref().and_then(|p| std::fs::read(p).ok())
            .filter(|d| d.len() == FLASH_SIZE as usize);
        let from_file = saved.is_some();
        let mut data = saved.unwrap_or_else(|| image.to_vec());
        data.resize(FLASH_SIZE as usize, 0xff);
        let f = Flash {
            s: Mutex::new(FlashState { data, mode: FlashMode::Read, dirty: !from_file }),
            path,
            gen: AtomicU64::new(0),
        };
        f.save();
        f
    }

    /// Write the part to its file if it changed since the last save.
    pub fn save(&self) {
        let Some(p) = &self.path else { return };
        let mut s = self.s.lock();
        if !s.dirty {
            return;
        }
        match std::fs::write(p, &s.data) {
            Ok(()) => s.dirty = false,
            Err(e) => eprintln!("iris: flash: could not save {}: {e}", p.display()),
        }
    }

    /// The part's contents.
    pub fn bytes(&self) -> Vec<u8> {
        self.s.lock().data.clone()
    }

    fn get16(&self, off: usize) -> u16 {
        let s = self.s.lock();
        match s.mode {
            FlashMode::Read => u16::from_be_bytes([s.data[off], s.data[off + 1]]),
            FlashMode::Id => if off & 2 == 0 { SFLASH_MFG_ID } else { SFLASH_DEV_ID },
            _ => SFLASH_STATUS_READY,
        }
    }

    fn put16(&self, off: usize, val: u16) {
        let mut s = self.s.lock();
        match s.mode {
            FlashMode::Program => {
                let old = u16::from_be_bytes([s.data[off], s.data[off + 1]]);
                let new = (old & val).to_be_bytes();
                s.data[off..off + 2].copy_from_slice(&new);
                s.mode = FlashMode::Status;
                s.dirty = true;
                self.gen.fetch_add(1, Ordering::Release);
                return;
            }
            FlashMode::Erase if val & 0xff == 0xd0 => {
                let seg = off / FLASH_SEG * FLASH_SEG;
                s.data[seg..seg + FLASH_SEG].fill(0xff);
                s.mode = FlashMode::Status;
                s.dirty = true;
                self.gen.fetch_add(1, Ordering::Release);
                return;
            }
            _ => {}
        }
        s.mode = match val & 0xff {
            0xff => FlashMode::Read,
            0x90 => FlashMode::Id,
            0x70 | 0x71 | 0x50 => FlashMode::Status,
            0x40 | 0x10 => FlashMode::Program,
            0x20 => FlashMode::Erase,
            _ => s.mode,
        };
        self.gen.fetch_add(1, Ordering::Release);
    }

    fn off(addr: u64) -> Option<usize> {
        (addr < FLASH_SIZE as u64).then_some(addr as usize)
    }
}

impl BusDevice for Flash {
    fn read8(&self, addr: u64) -> BusRead8 {
        let Some(o) = Self::off(addr) else { return BusRead8::ok(0xff) };
        BusRead8::ok((self.get16(o & !1) >> (8 * (1 - (o & 1)))) as u8)
    }
    fn read16(&self, addr: u64) -> BusRead16 {
        Self::off(addr).map_or(BusRead16::ok(0xffff), |o| BusRead16::ok(self.get16(o & !1)))
    }
    fn read32(&self, addr: u64) -> BusRead32 {
        let Some(o) = Self::off(addr & !3) else { return BusRead32::ok(!0) };
        BusRead32::ok((self.get16(o) as u32) << 16 | self.get16(o + 2) as u32)
    }
    fn read64(&self, addr: u64) -> BusRead64 {
        let hi = self.read32(addr & !7).data as u64;
        BusRead64::ok(hi << 32 | self.read32((addr & !7) + 4).data as u64)
    }
    fn write16(&self, addr: u64, val: u16) -> u32 {
        if let Some(o) = Self::off(addr) {
            self.put16(o & !1, val);
        }
        BUS_OK
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        if let Some(o) = Self::off(addr & !3) {
            self.put16(o, (val >> 16) as u16);
            if self.s.lock().mode == FlashMode::Read {
                self.put16(o + 2, val as u16);
            }
        }
        BUS_OK
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        BusDevice::write16(self, addr & !1, val as u16 * 0x101)
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        self.write32(addr & !7, (val >> 32) as u32)
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, _addr: u64) -> *const AtomicU64 {
        &self.gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_answers_id_status_program_and_erase() {
        let f = Flash::new(&[0x12, 0x34, 0x56, 0x78], None);
        assert_eq!(f.read32(0).data, 0x1234_5678);
        f.write16(0, 0x9090);
        assert_eq!(f.read16(0).data, 0x00b0);
        assert_eq!(f.read16(2).data, 0x66a8);
        f.write16(0, 0x4040);
        f.write16(2, 0x0f0f); // program: AND into 0x5678
        assert_eq!(f.read16(0).data & 0x80, 0x80);
        f.write16(0, 0xffff);
        assert_eq!(f.read16(2).data, 0x0608);
        f.write16(0, 0x2020);
        f.write16(0, 0xd0d0);
        f.write16(0, 0xffff);
        assert_eq!(f.read32(0).data, 0xffff_ffff);
        assert_eq!(f.read8(FLASH_SIZE as u64 + 4).data, 0xff); // past the part
    }
}
