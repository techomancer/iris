//! Bedrock's boot flash: two independent 8 MiB arrays, selected 32 MiB
//! apart, with the AMD command set and CFI query data.
//!
//! Boot aliases and executable views share the same array bytes. Each array
//! has its own command state; instruction fetches read stored bytes even
//! while register reads return command status.
//!
//! Ported from hw/misc/sgi-bedrock-flash.c and hw/mips/sgi/initial-flash.c
//! in jamesbraid/qemu-sgi-origami.



use super::sflash::amd::{AmdCore, Effect, SECTOR_SIZE};
use super::sflash::storage::FlashBacking;
use crate::traits::{
    BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR, BUS_OK,
};
use parking_lot::Mutex;
use std::io;
use std::path::Path;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

pub const PART_SIZE: usize = 8 << 20;
pub const FLASH_SIZE: usize = 2 * PART_SIZE;
pub const SECOND_POSITION: u64 = 32 << 20;

fn sum(bytes: &[u8]) -> u8 {
    bytes.iter().fold(0u8, |s, b| s.wrapping_add(*b))
}

/// Apply Fuel's firmware record fields while retaining the configuration sum.
/// This policy also applies when reopening an existing persistent store.
fn configure(data: &mut [u8]) {
    let config = &mut data[0x60..0xb0];
    let old = sum(config);
    config[0x34..0x38].fill(0);
    config[4..8].copy_from_slice(&0x549aa985u32.to_be_bytes());
    config[0x14..0x18].copy_from_slice(&500_000_000u32.to_be_bytes());
    config[0x30..0x34].copy_from_slice(&2u32.to_be_bytes());
    config[0x38..0x3c].copy_from_slice(&1u32.to_be_bytes());
    let adjustment = old.wrapping_sub(sum(config));
    config[0x34..0x38].copy_from_slice(&(adjustment as u32).to_be_bytes());
}

struct State {
    data: Vec<u8>,
    parts: [AmdCore; 2],
    backing: Option<FlashBacking>,
}

impl State {
    fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            parts: std::array::from_fn(|_| AmdCore::default()),
            backing: None,
        }
    }
    // Keep the in-memory array unchanged until each backing write succeeds.
    // If a later range fails, restore the old bytes for all ranges already tried.
    fn persist(&mut self, ranges: &[(usize, &[u8])]) -> io::Result<()> {
        let Some(backing) = &mut self.backing else {
            return Ok(());
        };
        for (index, &(offset, bytes)) in ranges.iter().enumerate() {
            if let Err(error) = backing.write_range(offset, bytes) {
                for &(restore_offset, restore_bytes) in &ranges[..=index] {
                    let old = &self.data[restore_offset..restore_offset + restore_bytes.len()];
                    if let Err(rollback) = backing.write_range(restore_offset, old) {
                        log::error!(
                            "bedrock-flash: backing rollback at {restore_offset:#x} failed: {rollback}"
                        );
                    }
                }
                return Err(error);
            }
        }
        Ok(())
    }
    fn replace(&mut self, offset: usize, bytes: &[u8]) -> io::Result<bool> {
        if self.data[offset..offset + bytes.len()] == *bytes {
            return Ok(false);
        }
        self.persist(&[(offset, bytes)])?;
        self.data[offset..offset + bytes.len()].copy_from_slice(bytes);
        Ok(true)
    }
    fn erase(&mut self, part: usize, sectors: &[usize]) -> io::Result<bool> {
        let erased = vec![0xff; SECTOR_SIZE];
        let ranges: Vec<_> = sectors
            .iter()
            .map(|sector| (part * PART_SIZE + sector * SECTOR_SIZE, erased.as_slice()))
            .collect();
        self.persist(&ranges)?;
        let mut changed = false;
        for (offset, bytes) in ranges {
            let data = &mut self.data[offset..offset + bytes.len()];
            changed |= data.iter().any(|byte| *byte != 0xff);
            data.fill(0xff);
        }
        Ok(changed)
    }
}

pub struct Flash {
    st: Mutex<State>,
    gen: AtomicU64,
}

impl Flash {
    /// Format a fresh two-position store from a decoded payload.
    pub fn new(payload: &[u8]) -> io::Result<Self> {
        if payload.len() > PART_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Fuel PROM payload exceeds 8 MiB",
            ));
        }
        let mut data = vec![0xff; FLASH_SIZE];
        data[..payload.len()].copy_from_slice(payload);
        let label = PART_SIZE + 0x1e0000;
        data[label + 0x10..label + 0x14].copy_from_slice(&0x504c4f49u32.to_be_bytes());
        data[label + 0x14..label + 0x18].copy_from_slice(&1u32.to_be_bytes());
        data[label + 0x18..label + 0x1c].copy_from_slice(&1u32.to_be_bytes());
        configure(&mut data);
        Ok(Self {
            st: Mutex::new(State::new(data)),
            gen: AtomicU64::new(0),
        })
    }

    /// Load an exact 16 MiB store, or seed a missing/empty one. Wrong sizes are
    /// refused without writes. Reopening reapplies Fuel's configuration policy.
    /// Completed guest programming and erasure write through to this file.
    pub fn with_file(payload: &[u8], path: impl AsRef<Path>) -> io::Result<Self> {
        let mut data = Self::new(payload)?.st.into_inner().data;
        let mut backing = FlashBacking::open(path, &mut data).map_err(|error| {
            if error.kind() == io::ErrorKind::InvalidData {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Fuel flash store must be exactly 16 MiB (or empty)",
                )
            } else {
                error
            }
        })?;
        configure(&mut data);
        // Opening an existing store loads it before Fuel's configuration
        // fields are refreshed; an empty store was seeded from this same data.
        backing.write_range(0, &data)?;
        let flash = Self {
            st: Mutex::new(State {
                data,
                parts: std::array::from_fn(|_| AmdCore::default()),
                backing: Some(backing),
            }),
            gen: AtomicU64::new(0),
        };
        Ok(flash)
    }

    pub fn contents(&self) -> Vec<u8> {
        self.st.lock().data.clone()
    }
    pub fn generation(&self) -> u64 {
        self.gen.load(Ordering::Acquire)
    }
    pub fn reset(&self) {
        let mut st = self.st.lock();
        for part in &mut st.parts {
            part.reset();
        }
        self.gen.fetch_add(1, Ordering::Release);
    }
    /// Advance the model clock independently of flash reads.
    pub fn advance_ns(&self, ns: u64) {
        let mut st = self.st.lock();
        for part in 0..2 {
            let selecting = st.parts[part].selecting();
            let complete = st.parts[part].advance_ns(ns);
            if selecting && !st.parts[part].selecting() {
                self.gen.fetch_add(1, Ordering::Release);
            }
            if !complete {
                continue;
            }
            let sectors: Vec<_> = st.parts[part].selected_sectors().collect();
            let success = match st.erase(part, &sectors) {
                Ok(_) => true,
                Err(e) => {
                    log::error!("bedrock-flash: position {part} erase backing write failed: {e}");
                    false
                }
            };
            st.parts[part].finish_erase(success);
            self.gen.fetch_add(1, Ordering::Release);
        }
    }
    pub fn register_locks(self: &Arc<Self>) {
        let me = self.clone();
        crate::locks::register_lock_fn("bedrock_flash::data", move || me.st.is_locked());
    }
    /// Use the same shared flash for local and remote boot spaces.
    pub fn boot_view(self: &Arc<Self>) -> FlashView {
        FlashView {
            flash: self.clone(),
            range: None,
        }
    }
    /// A contiguous executable alias. Offsets refer to the 16 MiB backing.
    pub fn array_view(self: &Arc<Self>, offset: usize, len: usize) -> io::Result<FlashView> {
        if offset.checked_add(len).is_none_or(|end| end > FLASH_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Fuel flash alias exceeds backing",
            ));
        }
        Ok(FlashView {
            flash: self.clone(),
            range: Some((offset, len)),
        })
    }
}
impl Resettable for Flash {
    fn power_on(&self) {
        self.reset();
    }
}

pub struct FlashView {
    flash: Arc<Flash>,
    range: Option<(usize, usize)>,
}
impl FlashView {
    fn offset(&self, addr: u64, width: usize) -> Option<usize> {
        if let Some((base, len)) = self.range {
            let off = usize::try_from(addr).ok()?;
            return off
                .checked_add(width)
                .filter(|end| *end <= len)
                .map(|_| base + off);
        }
        let (base, within) = if addr < PART_SIZE as u64 {
            (0, addr)
        } else if addr >= SECOND_POSITION {
            (PART_SIZE, addr - SECOND_POSITION)
        } else {
            return None;
        };
        let within = usize::try_from(within).ok()?;
        within
            .checked_add(width)
            .filter(|end| *end <= PART_SIZE)
            .map(|_| base + within)
    }
    fn read(&self, addr: u64, width: usize) -> Option<u64> {
        let off = self.offset(addr, width)?;
        let mut st = self.flash.st.lock();
        if self.range.is_none() {
            let part = off / PART_SIZE;
            if !st.parts[part].array_reads() && (width != 2 || off & 1 != 0) {
                return None;
            }
            if width == 2 && off & 1 == 0 {
                if let Some(value) = st.parts[part].read16(off % PART_SIZE) {
                    return Some(value as u64);
                }
            }
        }
        Some(
            st.data[off..off + width]
                .iter()
                .fold(0, |v, b| (v << 8) | *b as u64),
        )
    }
    fn refuse(&self, addr: u64, width: usize, value: u64) -> u32 {
        crate::dlog!(
            crate::devlog::LogModule::Bedrock,
            "bedrock-flash: unsupported {width}-byte store {value:#x} at {addr:#x}; invalid bus cycle or executable view"
        );
        BUS_ERR
    }
}
impl BusDevice for FlashView {
    fn read8(&self, a: u64) -> BusRead8 {
        self.read(a, 1)
            .map_or_else(BusRead8::err, |v| BusRead8::ok(v as u8))
    }
    fn read16(&self, a: u64) -> BusRead16 {
        self.read(a, 2)
            .map_or_else(BusRead16::err, |v| BusRead16::ok(v as u16))
    }
    fn read32(&self, a: u64) -> BusRead32 {
        self.read(a, 4)
            .map_or_else(BusRead32::err, |v| BusRead32::ok(v as u32))
    }
    fn read64(&self, a: u64) -> BusRead64 {
        self.read(a, 8).map_or_else(BusRead64::err, BusRead64::ok)
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        self.refuse(a, 1, v as u64)
    }
    fn write16(&self, a: u64, v: u16) -> u32 {
        self.refuse(a, 2, v as u64)
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        self.refuse(a, 4, v as u64)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        if self.range.is_some() || a & 7 != 0 || v > u16::MAX as u64 {
            return self.refuse(a, 8, v);
        }
        let (part, within) = if a < SECOND_POSITION {
            (0, a)
        } else {
            (1, a - SECOND_POSITION)
        };
        let Some(end) = within.checked_add(8) else {
            return self.refuse(a, 8, v);
        };
        if end > SECOND_POSITION {
            return self.refuse(a, 8, v);
        }
        let offset = (within / 4) as usize;
        let mut st = self.flash.st.lock();
        let effect = match st.parts[part].write16(offset, v as u16) {
            Ok(effect) => effect,
            Err(()) => {
                self.flash.gen.fetch_add(1, Ordering::Release);
                return BUS_ERR;
            }
        };
        if let Effect::Program { offset, value } = effect {
            let backing = part * PART_SIZE + offset;
            let old = u16::from_be_bytes([st.data[backing], st.data[backing + 1]]);
            match st.replace(backing, &(old & value).to_be_bytes()) {
                Ok(_) => {}
                Err(e) => {
                    log::error!("bedrock-flash: program backing write at {backing:#x} failed: {e}");
                    self.flash.gen.fetch_add(1, Ordering::Release);
                    return BUS_ERR;
                }
            }
        }
        self.flash.gen.fetch_add(1, Ordering::Release);
        BUS_OK
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, _addr: u64) -> *const AtomicU64 {
        &self.flash.gen
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-flash-test.c in
    // jamesbraid/qemu-sgi-origami: program, sector-erase, erase-polling,
    // reset-states and backing-file.
    use super::*;
    fn unlock(v: &FlashView, part: u64) {
        assert_eq!(v.write64(part + 8 * 0x555, 0xaa), crate::traits::BUS_OK);
        assert_eq!(v.write64(part + 8 * 0x2aa, 0x55), crate::traits::BUS_OK);
    }
    fn program(v: &FlashView, part: u64, off: u64, value: u16) {
        unlock(v, part);
        assert_eq!(v.write64(part + 8 * 0x555, 0xa0), crate::traits::BUS_OK);
        assert_eq!(
            v.write64(part + 4 * off, value as u64),
            crate::traits::BUS_OK
        );
    }
    fn erase_select(v: &FlashView, part: u64, offset: u64) {
        unlock(v, part);
        assert_eq!(v.write64(part + 8 * 0x555, 0x80), BUS_OK);
        unlock(v, part);
        assert_eq!(v.write64(part + 4 * offset, 0x30), BUS_OK);
    }

    /// The PROM reads CPU configuration and the volume label from its boot flash.
    #[test]
    fn the_seed_exposes_fuel_configuration_and_volume_label() {
        let seed = vec![0x21; 0xb0];
        let f = Flash::new(&seed).unwrap();
        let bytes = f.contents();
        assert_eq!(sum(&bytes[0x60..0xb0]), sum(&seed[0x60..0xb0]));
        assert_eq!(&bytes[0x64..0x68], &0x549aa985u32.to_be_bytes());
        assert_eq!(&bytes[0x74..0x78], &500_000_000u32.to_be_bytes());
        assert_eq!(&bytes[0x90..0x94], &2u32.to_be_bytes());
        let label = PART_SIZE + 0x1e0000;
        assert_eq!(&bytes[label..label + 16], &[0xff; 16]);
        assert_eq!(
            &bytes[label + 16..label + 28],
            &[0x50, 0x4c, 0x4f, 0x49, 0, 0, 0, 1, 0, 0, 0, 1]
        );
        assert!(bytes[label + 28..].iter().all(|b| *b == 0xff));
        assert!(Flash::new(&vec![0; PART_SIZE + 1]).is_err());
    }
    /// The PROM programs through one view while executable aliases must
    /// keep reading array bytes.
    #[test]
    fn programming_changes_shared_bytes_without_exposing_status_to_instruction_fetches() {
        {
            let f = Arc::new(Flash::new(&[]).unwrap());
            let local = f.boot_view();
            let remote = f.boot_view();
            let execute = f.array_view(0, PART_SIZE).unwrap();
            assert_eq!(local.write64(SECOND_POSITION, 0xf0), crate::traits::BUS_OK);
            for part in [0, SECOND_POSITION] {
                assert_eq!(local.write64(part + 8 * 0x55, 0x98), crate::traits::BUS_OK);
                assert_eq!(remote.read16(part + 0x20).data, b'Q' as u16);
                assert_eq!(remote.read16((SECOND_POSITION - part) + 0x20).data, 0xffff);
                assert_eq!(execute.read16(0x20).data, 0xffff);
                assert_eq!(remote.write64(part, 0xf0), crate::traits::BUS_OK);
                assert_eq!(local.write64(part + 8 * 0x555, 0xaa), crate::traits::BUS_OK);
                assert_eq!(
                    remote.write64(part + 8 * 0x2aa, 0x55),
                    crate::traits::BUS_OK
                );
                assert_eq!(local.write64(part + 8 * 0x555, 0xa0), crate::traits::BUS_OK);
                assert_eq!(
                    remote.write64(part + 4 * 0x200, 0x1234),
                    crate::traits::BUS_OK
                );
                assert_eq!(local.read16(part + 0x200).data, 0x1234);
            }
            assert_eq!(execute.read16(0x200).data, 0x1234);
            assert!(!local.read8(PART_SIZE as u64).is_ok());
        }
        {
            let f = Arc::new(Flash::new(&[]).unwrap());
            let v = f.boot_view();
            program(&v, 0, 0x200, 0x1234);
            program(&v, 0, 0x200, 0xf0ff);
            assert_eq!(v.read16(0x200).data, 0x1034);
            unlock(&v, 0);
            assert_eq!(v.write64(8 * 0x555, 0xa0), crate::traits::BUS_OK);
            let gen = f.generation();
            for (addr, value) in [
                (0x800, 0x10000),
                (0x804, 0),
                (64 << 20, 0),
            ] {
                assert_eq!(v.write64(addr, value), BUS_ERR);
            }
            assert_eq!(v.write16(0x800, 0), BUS_ERR);
            assert_eq!(f.generation(), gen);
            assert_eq!(v.write64(0x800, 0x00ff), crate::traits::BUS_OK);
            assert_eq!(v.read16(0x200).data, 0x0034);
            let execute = f.array_view(0, PART_SIZE).unwrap();
            unlock(&v, 0);
            assert_eq!(execute.write64(8 * 0x555, 0xa0), BUS_ERR);
            assert_eq!(v.write64(8 * 0x555, 0xa0), crate::traits::BUS_OK);
            assert_eq!(v.write64(0x808, 0x5555), crate::traits::BUS_OK);
            assert_eq!(execute.read16(0x202).data, 0x5555);
        }
        {
            let f = Arc::new(Flash::new(&[]).unwrap());
            let v = f.boot_view();
            assert_eq!(v.write64(0x550, 0x98), BUS_ERR);
            assert_eq!(v.read16(0x20).data, 0xffff);
            assert_eq!(v.write64(8 * 0x55, 0x98), BUS_OK);
            let expected = [
                0x51, 0x52, 0x59, 2, 0, 0, 0, 0, 0, 0, 0, 0x27, 0x36, 0, 0, 3, 0, 9, 0, 4, 0, 4, 0, 23,
                1, 0, 0, 0, 1, 0x7f, 0, 0, 1,
            ];
            for (index, value) in expected.into_iter().enumerate() {
                assert_eq!(v.read16(((index + 0x10) * 2) as u64).data, value);
            }
            assert_eq!(v.read16(0x100).data, 0);
            assert!(!v.read8(0x20).is_ok());
            assert!(!v.read16(0x21).is_ok());
            f.reset();
            assert_eq!(v.read16(0x20).data, 0xffff);
        }
    }
    /// The PROM polls erase completion and can reset the chip before
    /// selected sectors change.
    #[test]
    fn erase_completes_only_for_selected_sectors_and_reset_cancels_it() {
        let f = Arc::new(Flash::new(&[]).unwrap());
        let v = f.boot_view();
        program(&v, 0, 0x200, 0);
        program(&v, 0, 0x10000, 0);
        erase_select(&v, 0, 0x200);
        assert!(!v.read8(0x200).is_ok());
        let a = v.read16(0x200).data;
        let b = v.read16(0x200).data;
        assert_eq!((a ^ b) & 0x44, 0x44);
        let a = v.read16(0x10000).data;
        let b = v.read16(0x10000).data;
        assert_eq!((a ^ b) & 0x44, 0x40);
        f.advance_ns(40_000);
        assert_eq!(v.write64(4 * 0x10000, 0x30), BUS_OK);
        f.advance_ns(40_000);
        assert_eq!(v.read16(0x10000).data & 8, 0);
        assert_eq!(v.write64(SECOND_POSITION + 8 * 0x55, 0x98), BUS_OK);
        assert_eq!(v.read16(SECOND_POSITION + 0x20).data, b'Q' as u16);
        f.advance_ns(10_000);
        assert_eq!(v.read16(0x200).data & 8, 8);
        assert_eq!(v.write64(0, 0xf0), BUS_OK);
        assert_eq!(v.read16(0x200).data & 0x80, 0);
        f.reset();
        f.advance_ns(u64::MAX);
        assert_eq!(v.read16(0x200).data, 0);
        assert_eq!(v.read16(0x10000).data, 0);
        assert_eq!(v.read16(SECOND_POSITION + 0x20).data, 0xffff);
        erase_select(&v, 0, 0x200);
        assert_eq!(v.write64(0, 0xf0), BUS_OK);
        f.advance_ns(u64::MAX);
        assert_eq!(v.read16(0x200).data, 0);
        erase_select(&v, 0, 0x200);
        assert_eq!(v.write64(4 * 0x10000, 0x30), BUS_OK);
        f.advance_ns(50_000 + 512_000_000);
        assert_eq!(v.read16(0x200).data, 0xffff);
        assert_eq!(v.read16(0x10000).data, 0xffff);
        unlock(&v, 0);
        assert_eq!(v.write64(8 * 0x555, 0xa0), BUS_OK);
        f.reset();
        assert_eq!(v.write64(4 * 0x200, 0x1234), BUS_ERR);
        assert_eq!(v.read16(0x200).data, 0xffff);
        program(&v, 0, 0x200, 0);
        program(&v, 0, 0x10000, 0x3456);
        erase_select(&v, 0, 0x200);
        f.advance_ns(50_000 + 512_000_000);
        assert_eq!(v.read16(0x200).data, 0xffff);
        assert_eq!(v.read16(0x10000).data, 0x3456);
    }
    /// Firmware updates must persist across emulator restarts without
    /// overwriting a supplied image.
    #[test]
    fn completed_programming_and_erases_survive_reopening_the_backing_file() {
        {
            let path = std::env::temp_dir().join(format!("iris-bedrock-store-{}", std::process::id()));
            let mut bytes = vec![0x5a; FLASH_SIZE];
            bytes[PART_SIZE] = 0xc7;
            std::fs::write(&path, &bytes).unwrap();
            let f = Flash::with_file(&[0x11], &path).unwrap();
            assert_eq!(f.contents()[0], 0x5a);
            assert_eq!(f.contents()[PART_SIZE], 0xc7);
            assert_eq!(std::fs::read(&path).unwrap(), f.contents());
            drop(f);
            std::fs::write(&path, [1, 2, 3]).unwrap();
            assert!(Flash::with_file(&[], &path).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), [1, 2, 3]);
            std::fs::write(&path, []).unwrap();
            let g = Flash::with_file(&[0x34], &path).unwrap();
            assert_eq!(g.contents()[0], 0x34);
            assert_eq!(std::fs::metadata(&path).unwrap().len(), FLASH_SIZE as u64);
            drop(g);
            std::fs::remove_file(path).unwrap();
        }
        {
            let path = std::env::temp_dir().join(format!("iris-amd-persist-{}", std::process::id()));
            std::fs::write(&path, []).unwrap();
            let f = Arc::new(Flash::with_file(&[], &path).unwrap());
            let v = f.boot_view();
            program(&v, 0, 0x200, 0x1234);
            program(&v, SECOND_POSITION, (PART_SIZE - 2) as u64, 0x5678);
            drop(v);
            drop(f);
            let f = Arc::new(Flash::with_file(&[], &path).unwrap());
            let v = f.boot_view();
            assert_eq!(v.read16(0x200).data, 0x1234);
            assert_eq!(
                v.read16(SECOND_POSITION + PART_SIZE as u64 - 2).data,
                0x5678
            );
            unlock(&v, SECOND_POSITION);
            assert_eq!(v.write64(SECOND_POSITION + 8 * 0x555, 0x80), BUS_OK);
            unlock(&v, SECOND_POSITION);
            assert_eq!(
                v.write64(SECOND_POSITION + 4 * (PART_SIZE as u64 - 2), 0x30),
                BUS_OK
            );
            f.advance_ns(50_000 + 512_000_000);
            drop(v);
            drop(f);
            let f = Arc::new(Flash::with_file(&[], &path).unwrap());
            let v = f.boot_view();
            assert_eq!(v.read16(0x200).data, 0x1234);
            assert_eq!(
                v.read16(SECOND_POSITION + PART_SIZE as u64 - 2).data,
                0xffff
            );
            std::fs::remove_file(path).unwrap();
        }
    }

    /// A host write failure must preserve flash contents and remain visible
    /// to the guest.
    #[test]
    fn failed_backing_writes_leave_array_bytes_unchanged() {
        {
            let f = Arc::new(Flash::new(&[]).unwrap());
            // A read-only descriptor makes writes fail without a special backend.
            let path = std::env::temp_dir().join(format!("iris-amd-read-only-{}", std::process::id()));
            std::fs::write(&path, []).unwrap();
            f.st.lock().backing = Some(FlashBacking::from_file(
                std::fs::File::open(&path).unwrap(),
                FLASH_SIZE,
            ));
            let v = f.boot_view();
            unlock(&v, 0);
            assert_eq!(v.write64(8 * 0x555, 0xa0), BUS_OK);
            assert_eq!(v.write64(4 * 0x200, 0x1234), BUS_ERR);
            assert_eq!(v.read16(0x200).data, 0xffff);
            std::fs::remove_file(path).unwrap();
        }
        {
            let f = Arc::new(Flash::new(&[]).unwrap());
            let v = f.boot_view();
            program(&v, 0, 0x200, 0);
            let path =
                std::env::temp_dir().join(format!("iris-amd-erase-failure-{}", std::process::id()));
            std::fs::write(&path, []).unwrap();
            f.st.lock().backing = Some(FlashBacking::from_file(
                std::fs::File::open(&path).unwrap(),
                FLASH_SIZE,
            ));
            erase_select(&v, 0, 0x200);
            f.advance_ns(50_000 + 512_000_000);
            assert_eq!(v.read16(0x200).data & 0x20, 0x20);
            assert_eq!(f.contents()[0x200], 0);
            assert_eq!(v.write64(0, 0xf0), BUS_OK);
            assert_eq!(v.read16(0x200).data, 0);
            std::fs::remove_file(path).unwrap();
        }
    }

}
