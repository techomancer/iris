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

/// The fields of the PROM's configuration block that depend on the board.
///
/// The 80-byte record at flash 0x60 holds, in order: +0x00 0x15, +0x04 the
/// CPU mode word, +0x08 "ip27conf", +0x10 CPU Hz, +0x18 bus Hz, +0x20
/// 0x1312d0, +0x28 1, +0x2c flash timing 0x10, +0x30 machine type, +0x34 0,
/// +0x3c write timing 0x10 (established from the PROM). "CPU configuration
/// values mismatch" compares +0x00, 0x04, 0x08, 0x10, 0x18, 0x28, 0x2c, 0x30
/// and 0x3c against a reference record the PROM builds from the board
/// EEPROM's record, or, when it has none, from the speed-table row the
/// board-id word selects. Either way the machine type in the reference is the
/// one the board word's class gives. This struct sets the mode word, CPU Hz
/// and machine type; the PROM image carries the rest, the 200 MHz hub clock
/// among them (the same in every speed row).
///
/// The PROM reads the record before it touches the processor, so it must
/// agree with the CPU the machine builds: a mismatch in the boot-mode word
/// sends it to its error routine before it initialises GP. The values are
/// not a measured flash dump: each field's doc says how far it is known. The
/// Fuel's word selects its row by class 0x4000 and the processor's nibble.
/// The Tezro's records reuse the mode words of speed-table rows with the
/// same clock, cache size and tap, but a Tezro's board word is outside that
/// class, so the PROM checks them against the node board's record and the
/// class's machine type, not against a row. A real 800 MHz Fuel's mode word
/// was 0x549baf85.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FirmwareRecord {
    /// The CPU boot-mode word the PROM compares with the processor's mode.
    /// Its bits 24:0 are the mode inputs the processor latches into CP0
    /// Config at reset (the R10000 family's reset-mode inputs). The 500 MHz
    /// value is chosen: this PROM accepts it with board nibble 4, but it was
    /// not read from a 500 MHz Fuel. The 800 MHz row's real value, 0x549baf85,
    /// differs from it in bits 16 and 10:9.
    pub cpu_mode_word: u32,
    /// The processor clock the PROM reports and times against, in Hz. On the
    /// Fuel it is the clock of the speed-table row the board-id word selects;
    /// on the Tezro, the clock of the fitted part.
    pub cpu_clock_hz: u32,
    /// The record's word at +0x30, which the PROM's start-up check compares
    /// with a machine class of its own. When the node board's L1 record
    /// carries a CPU configuration, the PROM derives that class from the
    /// board word's top nibble (0x4 gives 2, 0x8 gives 1, any other 0).
    /// Established from the stock PROM 6.210's code: the class is 2 for every
    /// row of the class-0x4000 speed table, so the Fuel's record is 2. With 1,
    /// PROM 6.210 prints "CPU configuration values mismatch", which a real
    /// Fuel's console does not. The name SN11 for type 2 is inferred from a
    /// public flash transcript only: a Fuel's PROM flash session takes
    /// 2 from "(0)SN1 (1)SN10 (2)SN11 (3)SN12" and its IRIX logs "Selecting
    /// SN11" (just.graphica.com.au/tips/sgi-o350-chimera-notes).
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
    /// seed, once, when a store is formatted, with the compensation byte the
    /// PROM's editor would choose (the block's byte sum is the replaced
    /// block's, so the image still sums to zero). An opened store keeps the
    /// block it holds, so a flash update the PROM wrote survives.
    Authored,
}

impl FirmwareRecord {
    /// The Fuel's 500 MHz R14000 with a 2 MiB secondary cache.
    pub const FUEL: Self = Self {
        cpu_mode_word: 0x549a_a985,
        cpu_clock_hz: 500_000_000,
        machine_type: 2,
        header: Header::Refreshed,
    };
    /// The Tezro's 700 MHz R16000 with a 4 MiB secondary cache. The mode word
    /// is the row of the stock IP35 PROM's speed table for 700 MHz, 4 MiB and
    /// cache tap 0xc (established), and its low 25 bits are the Config word the
    /// Tezro board resets this chip with. The machine type is 0, the class the
    /// PROM derives from the Tezro's board word 0x2001, so that the block
    /// agrees with the CPU configuration the node board's L1 record carries
    /// (see `ip53_machine`). The speed-table row itself carries 2, the class
    /// of a board word in 0x4000, as the Fuel's is: copying it made the PROM
    /// print "CPU configuration values mismatch" at every start. The PROM
    /// derives the machine type from the board word's class: 2 for class 4,
    /// 1 for class 8, 0 for any other (established for PROM 6.210 and 6.211).
    /// A real IP53 or IP59 board's class is not 4, 8 or 0xc (established:
    /// Tezro and Origin 350 PROM banners print neither the IP34 nor the IP45
    /// mode line and do print the brick-type banner, which the PROM reaches
    /// only for other classes), so its flash must hold 0; that a real Tezro
    /// flash holds 0 is inferred (no mismatch notice in two real captures; no
    /// flash has been read). The speed-table rows that carry 2 are reached
    /// only through class 4: they are the Fuel's R16000 options, whose tuples
    /// match Tezro boards because the same tuple means the same mode bits.
    pub const TEZRO: Self = Self {
        cpu_mode_word: 0x589b_ad85,
        cpu_clock_hz: 700_000_000,
        machine_type: 0,
        header: Header::Authored,
    };
    /// The Tezro's 800 MHz R16000 with a 4 MiB secondary cache: the PROM speed
    /// table's row for 800 MHz, 4 MiB and tap 0xa (established; the table has a
    /// second row with tap 0xb and the same low 25 bits, and the 0xa one is
    /// what 800 MHz Tezro inventories print).
    pub const TEZRO_800: Self = Self {
        cpu_mode_word: 0x549b_af85,
        cpu_clock_hz: 800_000_000,
        machine_type: 0,
        header: Header::Authored,
    };
    /// The Tezro's 900 MHz R16000 with an 8 MiB secondary cache: the PROM
    /// speed table's row for 900 MHz (established; it is the table's only 900
    /// MHz row, tap 0xb, which a 900 MHz inventory prints). That a Tezro was
    /// ever sold or flashed with this part is not established.
    pub const TEZRO_900: Self = Self {
        cpu_mode_word: 0x569c_b185,
        cpu_clock_hz: 900_000_000,
        machine_type: 0,
        header: Header::Authored,
    };
    /// The Tezro's 1 GHz R16000 with a 16 MiB secondary cache. The PROM's
    /// speed table, whose top clock is 900 MHz, has no row for it, and no
    /// capture records the word: it is computed by formula, so the record is
    /// approximated. The clock-dependent fields follow the formulas the
    /// table's rows decode by, and bit 23 is set as in every row of 500 MHz
    /// and up.
    /// EC code 9 is 2 * 1000 / 200 - 1, SC code 5 gives the printed 333 MHz
    /// cache clock (2 * 1000 / (SC + 1)), SS code 5 is 16 MiB, and the printed
    /// tap 0x15 is 0x5 in bits 28:25 with the fifth bit at bit 22.
    pub const TEZRO_1000: Self = Self {
        cpu_mode_word: 0x4aed_b385,
        cpu_clock_hz: 1_000_000_000,
        machine_type: 0,
        header: Header::Authored,
    };

    /// CP0 Config at reset: the fixed 32 KiB primary cache sizes (fields 3
    /// in bits 31:29 and 28:26) with the board's mode bits.
    pub const fn reset_config0(&self) -> u32 {
        (3 << 29) | (3 << 26) | self.reset_mode()
    }
    /// The mode bits the processor latches into Config at reset: the mode
    /// word's bits 24:0.
    pub const fn reset_mode(&self) -> u32 {
        self.cpu_mode_word & 0x01ff_ffff
    }

    /// The configuration block written into an IP53 seed: the format word,
    /// the board's mode word, the `ip27conf` magic, the processor and
    /// 200 MHz controller clocks as 64-bit words, the real-time clock's
    /// 1,250,000 Hz, the machine type, a zero compensation byte (the caller sets
    /// it), an edit counter of one, and the PROM's 6.210 version fields.
    /// The layout is the PROM editor's.
    fn authored_block(&self) -> [u8; CONFIG_LEN] {
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
            (0x34, 0),
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
        Header::Authored => {
            // The compensation byte is the PROM editor's rule: the new block
            // sums, byte for byte, to what the block it replaces did, so the
            // image still sums to zero (the check the PROM runs on itself).
            let old = sum(&data[CONFIG_BASE..CONFIG_BASE + CONFIG_LEN]);
            let mut block = record.authored_block();
            block[0x37] = old.wrapping_sub(sum(&block));
            data[CONFIG_BASE..CONFIG_BASE + CONFIG_LEN].copy_from_slice(&block);
        }
    }
}

/// Put the board's record into a store that was opened rather than formatted.
fn reopen_header(data: &mut [u8], record: &FirmwareRecord) {
    match record.header {
        Header::Refreshed => configure(data, record),
        Header::Authored => {}
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
    // Keep the in-memory array unchanged until the backing write succeeds.
    fn persist(&mut self, ranges: &[(usize, &[u8])]) -> io::Result<()> {
        let Some(backing) = &mut self.backing else {
            return Ok(());
        };
        for &(offset, bytes) in ranges {
            backing.write_range(offset, bytes)?;
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
        assert_eq!(&bytes[0x90..0x94], &2u32.to_be_bytes(), "machine type: SN11");
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
    /// adjustment absorbs the difference.
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

    /// The compensation keeps the image's byte sum whatever block the
    /// payload carried, and changing any one byte of a seeded block off its
    /// compensated value breaks the sum.
    #[test]
    fn the_compensation_byte_keeps_the_images_sum_and_any_change_breaks_it() {
        // An image that sums to zero, as the PROM's own does, still sums to
        // zero once the record's fields are seeded into it.
        let mut zero = vec![0x21; 0x200];
        zero[0x60..0xb0].copy_from_slice(&factory_block()[0x60..0xb0]);
        zero[0x1ff] = 0u8.wrapping_sub(sum(&zero[..0x1ff]));
        assert_eq!(sum(&zero), 0);
        assert_eq!(sum(&Flash::new(&zero, FirmwareRecord::FUEL).unwrap().contents()[..0x200]), 0);
        for payload in [zero.clone(), vec![0x5a; 0x200], factory_block()] {
            let end = 0x200.min(payload.len());
            let before = sum(&payload[..end]);
            let bytes = Flash::new(&payload, FirmwareRecord::FUEL).unwrap().contents();
            assert_eq!(sum(&bytes[..end]), before, "the image's byte sum");
            // A record that differs in one field is compensated to the same sum.
            let other = FirmwareRecord { cpu_mode_word: FirmwareRecord::FUEL.cpu_mode_word ^ 0x0100_0000, ..FirmwareRecord::FUEL };
            assert_eq!(sum(&Flash::new(&payload, other).unwrap().contents()[..end]), before, "another mode word");
            for at in 0x60..0xb0 {
                let mut broken = bytes[..end].to_vec();
                broken[at] ^= 1;
                assert_ne!(sum(&broken), before, "byte {at:#x}");
            }
        }
    }

    /// The Config the CPU resets to carries the mode word's low 25 bits under
    /// the fixed primary cache sizes, so the PROM's check of one against the
    /// other passes: the Fuel's 500 MHz record gives 0x6c9aa985.
    #[test]
    fn the_reset_config_carries_the_mode_words_low_bits() {
        assert_eq!(FirmwareRecord::FUEL.reset_mode(), 0x009a_a985);
        assert_eq!(FirmwareRecord::FUEL.reset_config0(), 0x6c9a_a985);
        let other = FirmwareRecord { cpu_mode_word: 0xfe00_0000 | 0x0012_3456, ..FirmwareRecord::FUEL };
        assert_eq!(other.reset_mode(), 0x0012_3456);
    }

    /// Reopening a store reapplies the record, so a store follows the
    /// machine that opens it.
    #[test]
    fn reopening_a_store_reapplies_the_record() {
        let dir = std::env::temp_dir().join(format!("bedrock-flash-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fuel.bin");
        let other = FirmwareRecord { cpu_mode_word: 0x1234_5678, machine_type: 0, ..FirmwareRecord::FUEL };
        drop(Flash::with_file(&factory_block(), &path, other).unwrap());
        assert_eq!(&std::fs::read(&path).unwrap()[0x64..0x68], &0x1234_5678u32.to_be_bytes());
        let fuel = Flash::with_file(&[], &path, FirmwareRecord::FUEL).unwrap().contents();
        assert_eq!(&fuel[0x64..0x68], &0x549aa985u32.to_be_bytes());
        assert_eq!(&fuel[0x90..0x94], &2u32.to_be_bytes(), "machine type: SN11");
        assert_eq!(std::fs::read(&path).unwrap(), fuel, "the store holds what the machine sees");
        std::fs::remove_dir_all(&dir).unwrap();
    }
    const TEZRO_RECORDS: [(&str, FirmwareRecord); 4] = [
        ("700", FirmwareRecord::TEZRO),
        ("800", FirmwareRecord::TEZRO_800),
        ("900", FirmwareRecord::TEZRO_900),
        ("1000", FirmwareRecord::TEZRO_1000),
    ];

    /// The stock block the PROM ships: the 400 MHz R10000 and R12000 row,
    /// machine type 0, no compensation, edit counter 0. Built field by field, not
    /// copied; its byte sum is 0xac, as the PROM's own checksum rule sees it.
    fn stock_block() -> [u8; CONFIG_LEN] {
        let mut block = FirmwareRecord { cpu_mode_word: 0x5219_a785, cpu_clock_hz: 400_000_000, machine_type: 0, ..FirmwareRecord::TEZRO }
            .authored_block();
        block[0x38..0x3c].fill(0);
        assert_eq!(sum(&block), 0xac, "the stock block's byte sum");
        block
    }

    /// An image whose configuration block is the stock one, as the PROM's own
    /// image has it.
    fn stock_image() -> Vec<u8> {
        let mut image = vec![0x21; 0x200];
        image[0x60..0xb0].copy_from_slice(&stock_block());
        image
    }

    /// The seed's block follows the PROM editor's layout and rule: the mode
    /// word, the clock, the Bedrock and real-time clocks, machine type 0 (the
    /// class the PROM derives from the Tezro's board word) and an edit
    /// counter of one, and a compensation byte that makes the block sum to
    /// what the stock block did, 0xac, so the image keeps summing to zero:
    /// 0xac less the sum of the other bytes. The PROM's editor, copying a
    /// speed-table row's class 2, computes 0xd8, 0xfd (the 800 MHz row) and
    /// 0x1c for one edit of a stock-seeded image; class 0 makes each two more.
    #[test]
    fn the_tezro_seed_writes_the_blocks_the_proms_editor_would() {
        let image = stock_image();
        for ((name, record), compensation) in TEZRO_RECORDS.iter().zip([0xdau8, 0xff, 0x1e, 0xfa]) {
            let bytes = Flash::new(&image, *record).unwrap().contents();
            let block = &bytes[0x60..0xb0];
            assert_eq!(&block[0..4], &21u32.to_be_bytes(), "{name}");
            assert_eq!(&block[4..8], &record.cpu_mode_word.to_be_bytes(), "{name}");
            assert_eq!(&block[8..16], b"ip27conf", "{name}");
            assert_eq!(&block[0x10..0x18], &u64::from(record.cpu_clock_hz).to_be_bytes(), "{name}");
            assert_eq!(&block[0x18..0x20], &200_000_000u64.to_be_bytes(), "{name}: Bedrock clock");
            assert_eq!(&block[0x20..0x28], &1_250_000u64.to_be_bytes(), "{name}: real-time clock");
            assert_eq!(&block[0x30..0x34], &0u32.to_be_bytes(), "{name}: the board word's machine class");
            assert_eq!(&block[0x38..0x3c], &1u32.to_be_bytes(), "{name}: one edit");
            assert_eq!(sum(block), 0xac, "{name}: the stock block's sum");
            assert_eq!(block[0x37], compensation, "{name}: the compensation the PROM editor computes");
            assert_eq!(&bytes[..0x60], &image[..0x60], "{name}: the payload around the block is untouched");
            assert_eq!(&bytes[0xb0..image.len()], &image[0xb0..]);
        }
    }

    /// The compensation follows whatever block the payload carried (the whole
    /// image keeps its sum), and changing any one byte of a seeded block off
    /// its compensated value breaks the sum.
    #[test]
    fn a_tezro_compensation_byte_keeps_the_images_sum_and_any_change_breaks_it() {
        // An image that sums to zero, as the PROM's own does, still sums to
        // zero once a record's block is seeded into it.
        let mut zero = stock_image();
        zero[0x1ff] = 0u8.wrapping_sub(sum(&zero[..0x1ff]));
        assert_eq!(sum(&zero), 0);
        for (name, record) in TEZRO_RECORDS {
            assert_eq!(sum(&Flash::new(&zero, record).unwrap().contents()[..0x200]), 0, "{name}: the seeded image sums to zero");
        }
        for payload in [stock_image(), vec![0x5a; 0x200], factory_block()] {
            let before = sum(&payload[..0x200.min(payload.len())]);
            for (name, record) in TEZRO_RECORDS {
                let bytes = Flash::new(&payload, record).unwrap().contents();
                let end = 0x200.min(payload.len());
                assert_eq!(sum(&bytes[..end]), before, "{name}: the image's byte sum");
                // A record that differs in one field is compensated to the same sum.
                let other = FirmwareRecord { cpu_mode_word: record.cpu_mode_word ^ 0x0100_0000, ..record };
                assert_eq!(sum(&Flash::new(&payload, other).unwrap().contents()[..end]), before, "{name}: another mode word");
                for at in 0x60..0xb0 {
                    let mut broken = bytes[..end].to_vec();
                    broken[at] ^= 1;
                    assert_ne!(sum(&broken), before, "{name}: byte {at:#x}");
                }
            }
        }
    }

    /// The Tezro records carry the PROM table's mode words and the Config the
    /// CPU resets to carries the same low 25 bits, so the PROM's check of one
    /// against the other passes; the mode words of the three chips the PROM's
    /// speed table lists are its rows, and the 1 GHz one decodes by the
    /// editor's formulas.
    #[test]
    fn the_tezro_records_carry_their_mode_words_and_the_configs_they_imply() {
        // Config bits 24:0 are the mode inputs; the cache sizes above are fixed.
        const CONFIG0_MODE_MASK: u32 = 0x01ff_ffff;
        // (mode word, clock MHz, Bedrock MHz = 200): the PROM table's rows.
        let rows = [(0x589bad85u32, 700u32, 0x6c9b_ad85u32), (0x549baf85, 800, 0x6c9b_af85), (0x569cb185, 900, 0x6c9c_b185)];
        for ((_, record), (word, mhz, config)) in TEZRO_RECORDS.iter().zip(rows) {
            assert_eq!(record.cpu_mode_word, word);
            assert_eq!(record.cpu_clock_hz, mhz * 1_000_000);
            assert_eq!(record.reset_config0(), config);
        }
        let field = |word: u32, lo: u32, bits: u32| (word >> lo) & ((1 << bits) - 1);
        let decoded = [(350, 4u32 << 20), (400, 4 << 20), (450, 8 << 20), (333, 16 << 20)];
        for ((name, r), (cache_mhz, cache_bytes)) in TEZRO_RECORDS.iter().zip(decoded) {
            let (w, mhz) = (r.cpu_mode_word, r.cpu_clock_hz / 1_000_000);
            assert_eq!(w & CONFIG0_MODE_MASK, r.reset_mode(), "{name}");
            assert_eq!((field(w, 0, 3), field(w, 7, 2), field(w, 13, 1), field(w, 15, 1)), (5, 3, 1, 1), "{name}: K0 PM SB BE");
            assert_eq!(field(w, 9, 4), 2 * mhz / 200 - 1, "{name}: EC = 2 * CPU / Bedrock - 1");
            assert_eq!(2 * mhz / (field(w, 19, 3) + 1), cache_mhz, "{name}: cache clock = 2 * CPU / (SC + 1)");
            assert_eq!(1u32 << (field(w, 16, 3) + 19), cache_bytes, "{name}: SS is 512 KiB << SS");
        }
        // The tap the PROM prints: tap[3:0] in bits 28:25, tap[4] at bit 22.
        let tap = |word: u32| field(word, 25, 4) | (field(word, 22, 1) << 4);
        assert_eq!(TEZRO_RECORDS.map(|(_, r)| tap(r.cpu_mode_word)), [0xc, 0xa, 0xb, 0x15]);
        assert_eq!(FirmwareRecord::TEZRO_1000.cpu_mode_word, 0x4aed_b385);
        assert_eq!(FirmwareRecord::TEZRO_1000.reset_config0(), 0x6ced_b385);
    }

    /// Reopening a store reapplies a refreshed record, so a Fuel follows the
    /// machine that opens it, and leaves an authored block as stored, so a
    /// block the PROM rewrote on a Tezro survives the next start.
    #[test]
    fn reopening_keeps_a_tezros_stored_block_and_refreshes_a_fuels() {
        let dir = std::env::temp_dir().join(format!("bedrock-flash-tezro-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fuel_path = dir.join("fuel.bin");
        drop(Flash::with_file(&factory_block(), &fuel_path, FirmwareRecord::TEZRO).unwrap());
        let fuel = Flash::with_file(&[], &fuel_path, FirmwareRecord::FUEL).unwrap().contents();
        assert_eq!(&fuel[0x64..0x68], &0x549aa985u32.to_be_bytes());
        assert_eq!(&fuel[0x90..0x94], &2u32.to_be_bytes(), "machine type: SN11");

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
