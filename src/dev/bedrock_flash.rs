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
use crate::cpu::mips_cache_shadow::R16000_800_CONFIG0;
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

/// The fields of the PROM's configuration block that depend on the board.
///
/// The PROM reads them from flash before it touches the processor, so they
/// must agree with the CPU the machine builds: a mismatch in the boot-mode
/// word sends it to its error routine before it initialises GP. The values
/// are inputs the QEMU models choose for each board; none is a measured
/// flash dump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirmwareRecord {
    /// The CPU boot-mode word the PROM compares with the processor's mode.
    pub cpu_mode_word: u32,
    /// The mode bits the processor latches into CP0 Config at reset: Config
    /// bits 24:0 (the R10000 family's reset-mode inputs).
    pub reset_mode: u32,
    /// The processor clock the PROM reports and times against, in Hz.
    pub cpu_clock_hz: u32,
    /// The record's machine type field.
    pub machine_type: u32,
    /// How the record reaches the configuration block.
    pub header: Header,
}

/// How a board's record is put into the configuration block at 0x60.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Header {
    /// Rewrite the board's fields over the payload's own block, keeping its
    /// byte sum, whenever a store is formatted or opened. QEMU configures a
    /// Fuel's flash this way.
    Refreshed,
    /// Replace the whole block with the one authored for the board's flash
    /// seed, once, when a store is formatted. An opened store keeps the block
    /// it holds, as QEMU leaves the header of a supplied IP53 flash image
    /// alone, so a flash update the PROM wrote survives.
    Authored { checksum_adjustment: u8 },
}

impl FirmwareRecord {
    /// The Fuel's 500 MHz R14000 with a 2 MiB secondary cache.
    pub const FUEL: Self = Self {
        cpu_mode_word: 0x549a_a985,
        reset_mode: 0x009a_a985,
        cpu_clock_hz: 500_000_000,
        machine_type: 2,
        header: Header::Refreshed,
    };
    /// The Tezro's 700 MHz R16000 with a 4 MiB secondary cache, as the
    /// Tezro flash profile for PROM 6.210 records it: the stock IP35 PROM's
    /// word for 700 MHz and 4 MiB with cache tap 0xc, and the check
    /// adjustment that profile writes.
    pub const TEZRO: Self = Self {
        cpu_mode_word: 0x589b_ad85,
        reset_mode: 0x009b_ad85,
        cpu_clock_hz: 700_000_000,
        machine_type: 0,
        header: Header::Authored { checksum_adjustment: 0xda },
    };
    /// The Tezro's 800 MHz R16000 with a 4 MiB secondary cache: the QEMU
    /// Tezro offering's word for this part (cache tap 0xa), whose low 25
    /// bits are the part's Config reset word in the CPU models. No flash
    /// profile writes this block; it is the 700 MHz one with this word and
    /// the clock, and a check adjustment chosen to keep its byte sum.
    pub const TEZRO_800: Self = Self {
        cpu_mode_word: 0x549b_af85,
        reset_mode: R16000_800_CONFIG0,
        cpu_clock_hz: 800_000_000,
        machine_type: 0,
        header: Header::Authored { checksum_adjustment: 0xff },
    };
    /// CP0 Config at reset: the fixed 32 KiB primary cache sizes (fields 3
    /// in bits 31:29 and 28:26) with the board's mode bits.
    pub const fn reset_config0(&self) -> u32 {
        (3 << 29) | (3 << 26) | (self.reset_mode & 0x01ff_ffff)
    }

    /// The configuration block written into an IP53 seed: the format word,
    /// the board's mode word, the `ip27conf` magic, the processor and
    /// 200 MHz controller clocks as 64-bit words, the fixed 1,250,000 word,
    /// 16-cycle flash read and write timing, the machine type, the check
    /// adjustment, a program count of one and the PROM's 6.210 version.
    /// Only that profile defines these fields; their PROM meaning beyond that
    /// is not established here.
    fn authored_block(&self, checksum_adjustment: u8) -> [u8; CONFIG_LEN] {
        let mut block = [0u8; CONFIG_LEN];
        let mut put = |offset: usize, bytes: &[u8]| block[offset..offset + bytes.len()].copy_from_slice(bytes);
        put(0x00, &21u32.to_be_bytes());
        put(0x04, &self.cpu_mode_word.to_be_bytes());
        put(0x08, b"ip27conf");
        put(0x10, &u64::from(self.cpu_clock_hz).to_be_bytes());
        put(0x18, &200_000_000u64.to_be_bytes());
        put(0x20, &1_250_000u64.to_be_bytes());
        for (offset, value) in [
            (0x28, 1),
            (0x2c, 16),
            (0x30, self.machine_type),
            (0x34, u32::from(checksum_adjustment)),
            (0x38, 1),
            (0x3c, 16),
            (0x40, 6),
            (0x44, 210),
            (0x48, 0),
        ] {
            put(offset, &u32::to_be_bytes(value));
        }
        block
    }
}

/// Where the PROM's configuration block sits in the payload, and its size.
const CONFIG_BASE: usize = 0x60;
const CONFIG_LEN: usize = 0x50;

/// Put the board's record into a store being formatted.
fn format_header(data: &mut [u8], record: &FirmwareRecord) {
    match record.header {
        Header::Refreshed => configure(data, record),
        Header::Authored { checksum_adjustment } => {
            data[CONFIG_BASE..CONFIG_BASE + CONFIG_LEN].copy_from_slice(&record.authored_block(checksum_adjustment));
        }
    }
}

/// Put the board's record into a store that was opened rather than formatted.
fn reopen_header(data: &mut [u8], record: &FirmwareRecord) {
    match record.header {
        Header::Refreshed => configure(data, record),
        Header::Authored { .. } => {}
    }
}

/// Apply the board's firmware record while retaining the configuration sum.
fn configure(data: &mut [u8], record: &FirmwareRecord) {
    let config = &mut data[CONFIG_BASE..CONFIG_BASE + CONFIG_LEN];
    let old = sum(config);
    config[0x34..0x38].fill(0);
    config[4..8].copy_from_slice(&record.cpu_mode_word.to_be_bytes());
    config[0x14..0x18].copy_from_slice(&record.cpu_clock_hz.to_be_bytes());
    config[0x30..0x34].copy_from_slice(&record.machine_type.to_be_bytes());
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
    /// Format a fresh two-position store from a decoded payload, with the
    /// board's firmware record.
    pub fn new(payload: &[u8], record: FirmwareRecord) -> io::Result<Self> {
        if payload.len() > PART_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PROM payload exceeds 8 MiB",
            ));
        }
        let mut data = vec![0xff; FLASH_SIZE];
        data[..payload.len()].copy_from_slice(payload);
        let label = PART_SIZE + 0x1e0000;
        data[label + 0x10..label + 0x14].copy_from_slice(&0x504c4f49u32.to_be_bytes());
        data[label + 0x14..label + 0x18].copy_from_slice(&1u32.to_be_bytes());
        data[label + 0x18..label + 0x1c].copy_from_slice(&1u32.to_be_bytes());
        format_header(&mut data, &record);
        Ok(Self {
            st: Mutex::new(State::new(data)),
            gen: AtomicU64::new(0),
        })
    }

    /// Load an exact 16 MiB store, or seed a missing/empty one. Wrong sizes are
    /// refused without writes. Reopening reapplies a refreshed firmware record
    /// and leaves an authored one as the store holds it.
    /// Completed guest programming and erasure write through to this file.
    pub fn with_file(payload: &[u8], path: impl AsRef<Path>, record: FirmwareRecord) -> io::Result<Self> {
        let mut data = Self::new(payload, record)?.st.into_inner().data;
        let mut backing = FlashBacking::open(path, &mut data).map_err(|error| {
            if error.kind() == io::ErrorKind::InvalidData {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "flash store must be exactly 16 MiB (or empty)",
                )
            } else {
                error
            }
        })?;
        reopen_header(&mut data, &record);
        // Opening an existing store loads it before a refreshed record's
        // fields are applied; an empty store was seeded from this same data.
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
        let f = Flash::new(&seed, FirmwareRecord::FUEL).unwrap();
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
        assert!(Flash::new(&vec![0; PART_SIZE + 1], FirmwareRecord::FUEL).is_err());
    }
    /// The stock image's factory block: the 400 MHz mode word, machine type 0
    /// and a zero check adjustment, before any board configures it.
    fn factory_block() -> Vec<u8> {
        let mut image = vec![0x21; 0xb0];
        image[0x60..0xb0].fill(0);
        image[0x64..0x68].copy_from_slice(&0x5219a785u32.to_be_bytes());
        image[0x70..0x78].copy_from_slice(&400_000_000u64.to_be_bytes());
        image
    }

    /// A refreshed record leaves the block's byte sum unchanged: the check
    /// adjustment absorbs the difference. Follows test_ip35_configure in
    /// tests/unit/test-sgi-initial-flash.c, which pins 0x9b for the Fuel.
    #[test]
    fn a_refreshed_record_configures_the_block_and_keeps_its_byte_sum() {
        let record = FirmwareRecord::FUEL;
        let image = factory_block();
        let bytes = Flash::new(&image, record).unwrap().contents();
        let block = &bytes[0x60..0xb0];
        assert_eq!(&block[4..8], &record.cpu_mode_word.to_be_bytes());
        assert_eq!(&block[0x14..0x18], &record.cpu_clock_hz.to_be_bytes());
        assert_eq!(&block[0x30..0x34], &record.machine_type.to_be_bytes());
        assert_eq!(&block[0x38..0x3c], &1u32.to_be_bytes(), "flash program count");
        assert_eq!(&block[0x34..0x38], &[0, 0, 0, 0x9b]);
        assert_eq!(sum(block), sum(&image[0x60..0xb0]));
    }

    /// The Tezro's block is the one the Tezro seed profile writes, byte for
    /// byte, whatever block the payload carried.
    #[test]
    fn the_tezro_seed_writes_its_authored_block() {
        const AUTHORED_BLOCK: &str = concat!(
            "00000015589bad8569703237636f6e66",
            "0000000029b92700000000000bebc200",
            "00000000001312d00000000100000010",
            "00000000000000da0000000100000010",
            "00000006000000d20000000000000000",
        );
        let expected: Vec<u8> = (0..AUTHORED_BLOCK.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&AUTHORED_BLOCK[i..i + 2], 16).unwrap())
            .collect();
        for payload in [factory_block(), vec![0x5a; 0x200]] {
            let bytes = Flash::new(&payload, FirmwareRecord::TEZRO).unwrap().contents();
            assert_eq!(&bytes[0x60..0xb0], expected.as_slice());
            assert_eq!(&bytes[..0x60], &payload[..0x60], "the payload around the block is untouched");
            assert_eq!(&bytes[0xb0..payload.len()], &payload[0xb0..]);
        }
    }

    /// The Tezro's record carries the QEMU Tezro profile's firmware
    /// configuration, and the Config the CPU resets to carries the same mode
    /// bits, so the PROM's check of one against the other passes.
    #[test]
    fn the_tezro_record_carries_its_700_mhz_mode_and_the_config_it_implies() {
        let bytes = Flash::new(&factory_block(), FirmwareRecord::TEZRO).unwrap().contents();
        assert_eq!(&bytes[0x64..0x68], &0x589bad85u32.to_be_bytes());
        assert_eq!(&bytes[0x70..0x78], &700_000_000u64.to_be_bytes());
        assert_eq!(&bytes[0x90..0x94], &0u32.to_be_bytes(), "machine type 0");
        assert_eq!(FirmwareRecord::TEZRO.reset_config0(), 0x6c9b_ad85);
        assert_eq!(FirmwareRecord::FUEL.reset_config0(), 0x6c9a_a985, "the Fuel's value is unchanged");
        // Config bits 18:16 hold the secondary cache size: 3 is 4 MiB here
        // against the Fuel's 2 for 2 MiB.
        assert_eq!((FirmwareRecord::TEZRO.reset_config0() >> 16) & 7, 3);
        assert_eq!((FirmwareRecord::FUEL.reset_config0() >> 16) & 7, 2);
    }

    /// The 800 MHz record differs from the 700 MHz one in the mode bits the
    /// CPU models give the part and in its clock, and keeps the authored
    /// block's byte sum. Each record's reset Config is its part's.
    #[test]
    fn the_tezro_800_mhz_record_takes_the_parts_config_and_keeps_the_byte_sum() {
        use crate::cpu::mips_cache_shadow::{CONFIG0_RESET_MASK, R16000_MHZ700_CONFIG0};
        assert_eq!(FirmwareRecord::TEZRO.reset_mode, R16000_MHZ700_CONFIG0);
        assert_eq!(FirmwareRecord::TEZRO_800.reset_mode, R16000_800_CONFIG0);
        assert_eq!(FirmwareRecord::TEZRO_800.cpu_mode_word & CONFIG0_RESET_MASK, R16000_800_CONFIG0);
        assert_eq!(FirmwareRecord::TEZRO_800.reset_config0(), 0x6c9b_af85);
        let slow = Flash::new(&factory_block(), FirmwareRecord::TEZRO).unwrap().contents();
        let fast = Flash::new(&factory_block(), FirmwareRecord::TEZRO_800).unwrap().contents();
        assert_eq!(&fast[0x64..0x68], &0x549b_af85u32.to_be_bytes());
        assert_eq!(&fast[0x70..0x78], &800_000_000u64.to_be_bytes());
        assert_eq!(sum(&fast[0x60..0xb0]), sum(&slow[0x60..0xb0]));
        let differ: Vec<usize> = (0x60..0xb0).filter(|&i| fast[i] != slow[i]).collect();
        assert_eq!(differ, [0x64, 0x66, 0x74, 0x75, 0x76, 0x97], "mode word, clock and check adjustment only");
    }

    /// Reopening a store reapplies a refreshed record, so a Fuel follows the
    /// machine that opens it, and leaves an authored block as stored, so a
    /// block the PROM rewrote on a Tezro survives the next start.
    #[test]
    fn reopening_refreshes_the_fuels_record_and_keeps_a_tezros_stored_block() {
        let dir = std::env::temp_dir().join(format!("bedrock-flash-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fuel_path = dir.join("fuel.bin");
        drop(Flash::with_file(&factory_block(), &fuel_path, FirmwareRecord::TEZRO).unwrap());
        let fuel = Flash::with_file(&[], &fuel_path, FirmwareRecord::FUEL).unwrap().contents();
        assert_eq!(&fuel[0x64..0x68], &0x549aa985u32.to_be_bytes());
        assert_eq!(&fuel[0x90..0x94], &2u32.to_be_bytes());

        let tezro_path = dir.join("tezro.bin");
        let seeded = Flash::with_file(&factory_block(), &tezro_path, FirmwareRecord::TEZRO).unwrap().contents();
        assert_eq!(std::fs::read(&tezro_path).unwrap(), seeded, "the seed is what the store holds");
        let mut stored = seeded.clone();
        stored[0x64..0x68].copy_from_slice(&0x1234_5678u32.to_be_bytes());
        std::fs::write(&tezro_path, &stored).unwrap();
        let reopened = Flash::with_file(&factory_block(), &tezro_path, FirmwareRecord::TEZRO).unwrap().contents();
        assert_eq!(reopened, stored);
        std::fs::remove_dir_all(&dir).unwrap();
    }
    /// The PROM programs through one view while executable aliases must
    /// keep reading array bytes.
    #[test]
    fn programming_changes_shared_bytes_without_exposing_status_to_instruction_fetches() {
        {
            let f = Arc::new(Flash::new(&[], FirmwareRecord::FUEL).unwrap());
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
            let f = Arc::new(Flash::new(&[], FirmwareRecord::FUEL).unwrap());
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
            let f = Arc::new(Flash::new(&[], FirmwareRecord::FUEL).unwrap());
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
        let f = Arc::new(Flash::new(&[], FirmwareRecord::FUEL).unwrap());
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
            let f = Flash::with_file(&[0x11], &path, FirmwareRecord::FUEL).unwrap();
            assert_eq!(f.contents()[0], 0x5a);
            assert_eq!(f.contents()[PART_SIZE], 0xc7);
            assert_eq!(std::fs::read(&path).unwrap(), f.contents());
            drop(f);
            std::fs::write(&path, [1, 2, 3]).unwrap();
            assert!(Flash::with_file(&[], &path, FirmwareRecord::FUEL).is_err());
            assert_eq!(std::fs::read(&path).unwrap(), [1, 2, 3]);
            std::fs::write(&path, []).unwrap();
            let g = Flash::with_file(&[0x34], &path, FirmwareRecord::FUEL).unwrap();
            assert_eq!(g.contents()[0], 0x34);
            assert_eq!(std::fs::metadata(&path).unwrap().len(), FLASH_SIZE as u64);
            drop(g);
            std::fs::remove_file(path).unwrap();
        }
        {
            let path = std::env::temp_dir().join(format!("iris-amd-persist-{}", std::process::id()));
            std::fs::write(&path, []).unwrap();
            let f = Arc::new(Flash::with_file(&[], &path, FirmwareRecord::FUEL).unwrap());
            let v = f.boot_view();
            program(&v, 0, 0x200, 0x1234);
            program(&v, SECOND_POSITION, (PART_SIZE - 2) as u64, 0x5678);
            drop(v);
            drop(f);
            let f = Arc::new(Flash::with_file(&[], &path, FirmwareRecord::FUEL).unwrap());
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
            let f = Arc::new(Flash::with_file(&[], &path, FirmwareRecord::FUEL).unwrap());
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
            let f = Arc::new(Flash::new(&[], FirmwareRecord::FUEL).unwrap());
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
            let f = Arc::new(Flash::new(&[], FirmwareRecord::FUEL).unwrap());
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
