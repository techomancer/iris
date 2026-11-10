//! The boot flash of an SN0 node: one byte-wide 1 MiB AMD-command part in
//! sixteen 64 KiB sectors, behind the Hub's flash controller.
//!
//! The part is seen two ways. [`NodeFlash::chip`] is the chip itself, byte
//! addressed, as the reset view at node offset 0x1fc0_0000 shows it. The Hub's
//! controller window ([`NodeFlash::window`]) decodes the chip onto the 64-bit
//! junk bus: a read repeats the image across the window, and a command is a
//! 64-bit write with chip byte N at bus offset 8N. Both reach the same bytes
//! and the same command state.
//!
//! The command set is the shared AMD engine ([`super::sflash::amd`]) at this
//! part's [`GEOMETRY`]. The part is not identified. The geometry, the
//! manufacturer and device codes (0x01, 0xd5) and the unlock addresses
//! (0x5555, 0x2aaa) are the ones the node PROM's flash commands need. Approximations:
//!
//! * Program and sector erase complete inside the write that starts them, so
//!   a status poll finds the finished data and no toggle ever shows.
//! * Programming only clears bits, as on the part. A program of an address in
//!   a protected sector is not modelled (there is no protection).
//! * Chip erase and the CFI query set are not implemented; either is logged
//!   and ignored.
//!
//! Ported from hw/mips/sgi/sn0.c and hw/mips/sgi/initial-flash.c in
//! jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;
use crate::dlog;
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_ERR, BUS_OK};
use parking_lot::Mutex;
use std::io;
use std::path::Path;
use std::sync::Arc;

use super::sflash::amd::{AmdCore, Effect, Geometry, Identity};
use super::sflash::storage::FlashBacking;

pub const SIZE: usize = 1 << 20;
pub const SECTOR_SIZE: usize = 0x1_0000;
const UNLOCK_1: usize = 0x5555;
const UNLOCK_2: usize = 0x2aaa;
pub const MANUFACTURER: u8 = 0x01;
pub const DEVICE: u8 = 0xd5;

/// The configuration block at 0x60 and the values a node's product and
/// processor module put in it.
pub const CONFIG_OFFSET: usize = 0x60;
const CONFIG_SIZE: usize = 0x50;
const CONFIG_MAGIC: u64 = 0x6970_3237_636f_6e66; // "ip27conf"
/// The environment log's sector and header.
pub const LOG_SECTOR: usize = 14;
const LOG_MAGIC: u32 = 0x504c_4f47; // "PLOG"

/// The Hub's two fixed clocks, not configuration.
const HUB_CLOCK_HZ: u64 = 0x055d_4a80;
const HUB_RTC_CLOCK_HZ: u64 = 0x0013_12d0;

/// What a product and its processor module patch into the configuration block.
#[derive(Clone, Copy, Debug)]
pub struct FlashValues {
    pub machine_type: u32,
    pub read_cycles: u32,
    pub write_cycles: u32,
    pub cpu_mode_bits: u32,
    pub cpu_clock_hz: u64,
}

impl FlashValues {
    /// The stock PROM image's own values: the Origin 200 with a 180 MHz R10000
    /// and 1 MiB of secondary cache. Every product starts from them and
    /// replaces what differs for it and its processor module.
    pub const STOCK_R10000_180: Self = Self {
        machine_type: 1,
        read_cycles: 15,
        write_cycles: 4,
        cpu_mode_bits: 0x1211_a785,
        cpu_clock_hz: 180_000_000,
    };
}

fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_be_bytes());
}
fn put64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_be_bytes());
}
fn byte_sum(b: &[u8]) -> u8 {
    b.iter().fold(0u8, |s, x| s.wrapping_add(*x))
}

/// The template a flash starts from: entry zero spins, the other nine jump-table
/// entries return, the configuration block is filled in and the environment
/// log has its header. A payload overlays the front of it.
fn template(values: &FlashValues) -> Vec<u8> {
    let mut f = vec![0xff; SIZE];
    for i in 0..10 {
        put32(&mut f, i * 8, if i == 0 { 0x1000_ffff } else { 0x03e0_0008 });
        put32(&mut f, i * 8 + 4, 0);
    }
    let c = &mut f[CONFIG_OFFSET..CONFIG_OFFSET + CONFIG_SIZE];
    put32(c, 0x00, 0xb);
    put32(c, 0x04, values.cpu_mode_bits);
    put64(c, 0x08, CONFIG_MAGIC);
    put64(c, 0x10, values.cpu_clock_hz);
    put64(c, 0x18, HUB_CLOCK_HZ);
    put64(c, 0x20, HUB_RTC_CLOCK_HZ);
    put32(c, 0x28, 1);
    put32(c, 0x2c, values.read_cycles);
    put32(c, 0x30, values.machine_type);
    put32(c, 0x34, 0);
    put32(c, 0x38, 0);
    put32(c, 0x3c, values.write_cycles);
    put32(c, 0x40, 6);
    put32(c, 0x44, 156);
    put32(c, 0x48, 0);
    let log = LOG_SECTOR * SECTOR_SIZE;
    put32(&mut f, log + 0x10, LOG_MAGIC);
    put32(&mut f, log + 0x14, 1);
    put32(&mut f, log + 0x18, 1);
    f
}

/// Patch the product and module values into the configuration block, keeping
/// the block's byte sum: the checksum adjustment word absorbs the change. A
/// block without the magic is left alone.
fn configure(f: &mut [u8], values: &FlashValues) {
    let c = &mut f[CONFIG_OFFSET..CONFIG_OFFSET + CONFIG_SIZE];
    if u64::from_be_bytes(c[0x08..0x10].try_into().unwrap()) != CONFIG_MAGIC {
        return;
    }
    let old = byte_sum(c);
    put32(c, 0x34, 0);
    put32(c, 0x04, values.cpu_mode_bits);
    put64(c, 0x10, values.cpu_clock_hz);
    put32(c, 0x2c, values.read_cycles);
    put32(c, 0x30, values.machine_type);
    put32(c, 0x3c, values.write_cycles);
    let adjust = old.wrapping_sub(byte_sum(c));
    put32(c, 0x34, u32::from(adjust));
}

/// A fresh flash image: the template, the PROM payload over its front, and the
/// product and module values patched into whichever configuration block is
/// there.
pub fn initial_image(payload: &[u8], values: &FlashValues) -> Result<Vec<u8>, String> {
    if payload.len() > SIZE {
        return Err(format!("the PROM payload is {} bytes; the node flash holds {SIZE}", payload.len()));
    }
    let mut f = template(values);
    f[..payload.len()].copy_from_slice(payload);
    configure(&mut f, values);
    Ok(f)
}

/// The part, to the command engine: 1 MiB, byte-wide, sixteen sectors.
const GEOMETRY: Geometry = Geometry {
    x8: true,
    sector_size: SECTOR_SIZE,
    sectors: SIZE / SECTOR_SIZE,
    unlock: [UNLOCK_1, UNLOCK_2],
    address_mask: usize::MAX,
    // Manufacturer, device and "sector not protected", then erased bytes.
    identity: Identity { codes: [MANUFACTURER as u16, DEVICE as u16, 0], other: 0xff },
    cfi: false,
    read_only: false,
    // Program and sector erase complete inside the write that starts them.
    timed_erase: false,
    unlock_from_identify: true,
    log: LogModule::Hub,
};

struct State {
    data: Vec<u8>,
    core: AmdCore,
    backing: Option<FlashBacking>,
}

impl State {
    fn new(data: Vec<u8>, backing: Option<FlashBacking>) -> Self {
        Self { data, core: AmdCore::new(GEOMETRY), backing }
    }

    fn read(&mut self, off: usize) -> u8 {
        match self.core.read16(off) {
            Some(code) => code as u8,
            None => self.data[off],
        }
    }

    fn persist(&mut self, off: usize, len: usize) {
        if let Some(b) = &mut self.backing {
            if let Err(e) = b.write_range(off, &self.data[off..off + len]) {
                log::error!("node-flash: backing write at {off:#x} failed: {e}");
            }
        }
    }

    fn write(&mut self, off: usize, v: u8) {
        // An unsupported cycle is logged by the engine and leaves the part in
        // array mode; the bus answers either way.
        if let Ok(Effect::Program { offset, value }) = self.core.write16(off, u16::from(v)) {
            self.data[offset] &= value as u8;
            self.persist(offset, 1);
        }
        // An erase is selected and done at once.
        let erase = self.core.selected_sectors().next();
        if let Some(sector) = erase {
            let start = sector * SECTOR_SIZE;
            self.data[start..start + SECTOR_SIZE].fill(0xff);
            self.persist(start, SECTOR_SIZE);
            self.core.finish_erase(true);
        }
    }
}

pub struct NodeFlash {
    st: Mutex<State>,
}

impl NodeFlash {
    pub fn new(image: Vec<u8>) -> Result<Arc<Self>, String> {
        if image.len() != SIZE {
            return Err(format!("the node flash is {} bytes, not {SIZE}", image.len()));
        }
        Ok(Arc::new(Self {
            st: Mutex::new(State::new(image, None)),
        }))
    }

    /// A flash backed by `path`: an existing file of exactly 1 MiB replaces
    /// `image` (completed programs and erases are written through), an empty
    /// or missing one is seeded from it.
    pub fn with_file(image: Vec<u8>, path: impl AsRef<Path>) -> io::Result<Arc<Self>> {
        let mut data = image;
        let backing = FlashBacking::open(path, &mut data)?;
        Ok(Arc::new(Self {
            st: Mutex::new(State::new(data, Some(backing))),
        }))
    }

    pub fn contents(&self) -> Vec<u8> {
        self.st.lock().data.clone()
    }

    /// The chip, byte addressed from its first byte.
    pub fn chip(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(Chip(self.clone()))
    }

    /// The Hub's controller window over the chip.
    pub fn window(self: &Arc<Self>) -> Arc<dyn BusDevice> {
        Arc::new(Window(self.clone()))
    }
}

impl Resettable for NodeFlash {
    /// The chip returns to array reads; its contents survive.
    fn power_on(&self) {
        self.st.lock().core.reset();
    }
}

struct Chip(Arc<NodeFlash>);

impl Chip {
    fn bytes(&self, a: u64, n: usize) -> Option<u64> {
        let a = usize::try_from(a).ok().filter(|a| a.checked_add(n).is_some_and(|e| e <= SIZE))?;
        let mut st = self.0.st.lock();
        Some((0..n).fold(0u64, |v, i| (v << 8) | u64::from(st.read(a + i))))
    }
}

macro_rules! chip_read {
    ($r:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            self.bytes(a, std::mem::size_of::<$t>()).map_or_else($out::err, |v| $out::ok(v as $t))
        }
    };
}
impl BusDevice for Chip {
    chip_read!(read8, u8, BusRead8);
    chip_read!(read16, u16, BusRead16);
    chip_read!(read32, u32, BusRead32);
    chip_read!(read64, u64, BusRead64);
    fn write8(&self, a: u64, v: u8) -> u32 {
        match usize::try_from(a) {
            Ok(a) if a < SIZE => {
                self.0.st.lock().write(a, v);
                BUS_OK
            }
            _ => BUS_ERR,
        }
    }
}

/// The controller window: reads repeat the image, only 64-bit writes reach
/// the command port.
struct Window(Arc<NodeFlash>);

macro_rules! window_read {
    ($r:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            let mut st = self.0.st.lock();
            let n = std::mem::size_of::<$t>();
            let v = (0..n).fold(0u64, |v, i| (v << 8) | u64::from(st.read((a as usize + i) % SIZE)));
            $out::ok(v as $t)
        }
    };
}
impl BusDevice for Window {
    window_read!(read8, u8, BusRead8);
    window_read!(read16, u16, BusRead16);
    window_read!(read32, u32, BusRead32);
    window_read!(read64, u64, BusRead64);
    fn write64(&self, a: u64, v: u64) -> u32 {
        self.0.st.lock().write((a as usize / 8) % SIZE, v as u8);
        BUS_OK
    }
    fn write8(&self, a: u64, v: u8) -> u32 {
        dlog!(LogModule::Hub, "node flash: ignored {v:#x} byte write at window offset {a:#x}");
        BUS_OK
    }
    fn write16(&self, a: u64, v: u16) -> u32 {
        dlog!(LogModule::Hub, "node flash: ignored {v:#x} halfword write at window offset {a:#x}");
        BUS_OK
    }
    fn write32(&self, a: u64, v: u32) -> u32 {
        dlog!(LogModule::Hub, "node flash: ignored {v:#x} word write at window offset {a:#x}");
        BUS_OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: FlashValues = FlashValues {
        machine_type: 1,
        read_cycles: 15,
        write_cycles: 4,
        cpu_mode_bits: 0x1211_a785,
        cpu_clock_hz: 180_000_000,
    };

    fn payload() -> Vec<u8> {
        // A payload whose configuration block is the stock shape: magic in
        // place, product fields as a stock image holds them.
        let mut p = vec![0u8; 0x1000];
        let c = &mut p[0x60..0xb0];
        put32(c, 0x04, 0x1211_a785);
        put64(c, 0x08, 0x6970_3237_636f_6e66u64);
        put64(c, 0x10, 180_000_000);
        put32(c, 0x2c, 8);
        put32(c, 0x30, 0);
        put32(c, 0x3c, 1);
        p
    }

    #[test]
    fn the_four_product_words_are_patched_and_the_block_sum_kept() {
        let p = payload();
        let before = byte_sum(&p[0x60..0xb0]);
        let f = initial_image(&p, &V).unwrap();
        let c = &f[0x60..0xb0];
        assert_eq!(u32::from_be_bytes(c[0x2c..0x30].try_into().unwrap()), 15);
        assert_eq!(u32::from_be_bytes(c[0x30..0x34].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(c[0x3c..0x40].try_into().unwrap()), 4);
        assert_ne!(c[0x34..0x38], [0; 4], "the adjustment word absorbs the change");
        assert_eq!(byte_sum(c), before);
    }

    /// A stock block (the shipped PROM's values) patched for the default
    /// module moves exactly the cycle counts and machine type, and the
    /// adjustment word that keeps the block's sum is 0xf5.
    #[test]
    fn the_adjustment_byte_for_the_stock_block_is_0xf5() {
        let mut p = vec![0u8; 0x1000];
        let c = &mut p[0x60..0xb0];
        put32(c, 0x00, 0xb);
        put32(c, 0x04, 0x1211_a785);
        put64(c, 0x08, 0x6970_3237_636f_6e66);
        put64(c, 0x10, 180_000_000);
        put64(c, 0x18, 90_000_000);
        put64(c, 0x20, 1_250_000);
        put32(c, 0x28, 1);
        put32(c, 0x2c, 8);
        put32(c, 0x3c, 1);
        put32(c, 0x40, 6);
        put32(c, 0x44, 156);
        let f = initial_image(&p, &V).unwrap();
        assert_eq!(&f[0x94..0x98], &0xf5u32.to_be_bytes());
        assert_eq!(&f[0x8c..0x90], &15u32.to_be_bytes());
        assert_eq!(&f[0x90..0x94], &1u32.to_be_bytes());
        assert_eq!(&f[0x9c..0xa0], &4u32.to_be_bytes());
    }

    #[test]
    fn a_payload_without_the_magic_keeps_its_block() {
        let mut p = vec![0u8; 0x1000];
        p[0x60 + 0x30 + 3] = 7;
        let f = initial_image(&p, &V).unwrap();
        assert_eq!(f[0x60 + 0x33], 7);
    }

    #[test]
    fn the_template_fills_what_the_payload_leaves() {
        let f = initial_image(&[], &V).unwrap();
        assert_eq!(&f[0..8], &[0x10, 0, 0xff, 0xff, 0, 0, 0, 0], "entry zero spins");
        assert_eq!(&f[8..12], &0x03e0_0008u32.to_be_bytes());
        let log = 14 * 0x1_0000;
        assert_eq!(&f[log + 0x10..log + 0x14], &0x504c_4f47u32.to_be_bytes());
        assert_eq!(&f[log + 0x14..log + 0x18], &1u32.to_be_bytes());
        assert_eq!(&f[log + 0x18..log + 0x1c], &1u32.to_be_bytes());
        assert_eq!(&f[0x60 + 0x44..0x60 + 0x48], &156u32.to_be_bytes(), "the firmware revision the banner shows");
        assert!(f[log + 0x1c..log + 0x1_0000].iter().all(|b| *b == 0xff));
        assert_eq!(f[0x10_0000 - 1], 0xff);
    }

    #[test]
    fn a_payload_larger_than_the_part_is_refused() {
        assert!(initial_image(&vec![0; 0x10_0001], &V).is_err());
    }

    fn command(f: &Arc<NodeFlash>, off: usize, v: u8) {
        assert_eq!(f.window().write64(off as u64 * 8, u64::from(v)), BUS_OK);
    }

    #[test]
    fn program_clears_bits_and_erase_restores_a_sector() {
        let f = NodeFlash::new(vec![0xff; 0x10_0000]).unwrap();
        let chip = f.chip();
        for (off, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0xa0), (0x1234, 0x5a)] {
            command(&f, off, v);
        }
        assert_eq!(chip.read8(0x1234).data, 0x5a);
        // Programming cannot set a bit.
        for (off, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0xa0), (0x1234, 0xa5)] {
            command(&f, off, v);
        }
        assert_eq!(chip.read8(0x1234).data, 0x00);
        for (off, v) in [
            (0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0x80),
            (0x5555, 0xaa), (0x2aaa, 0x55), (0x1000, 0x30),
        ] {
            command(&f, off, v);
        }
        assert_eq!(chip.read8(0x1234).data, 0xff);
    }

    /// The data cycle of a program takes every bit, so a byte of 0xf0 is
    /// programmed rather than read as a reset.
    #[test]
    fn a_program_of_0xf0_is_data_not_a_reset() {
        let f = NodeFlash::new(vec![0xff; 0x10_0000]).unwrap();
        for (off, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0xa0), (0x77, 0xf0)] {
            command(&f, off, v);
        }
        assert_eq!(f.chip().read8(0x77).data, 0xf0);
    }

    #[test]
    fn identify_mode_reads_the_codes_until_reset() {
        let f = NodeFlash::new(vec![0x11; 0x10_0000]).unwrap();
        for (off, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0x90)] {
            command(&f, off, v);
        }
        assert_eq!(f.chip().read8(0).data, 0x01);
        assert_eq!(f.chip().read8(1).data, 0xd5);
        command(&f, 0, 0xf0);
        assert_eq!(f.chip().read8(0).data, 0x11);
    }

    #[test]
    fn the_window_repeats_the_image_and_takes_commands_on_64_bit_writes_only() {
        let mut image = vec![0xff; 0x10_0000];
        image[5] = 0x42;
        let f = NodeFlash::new(image).unwrap();
        let w = f.window();
        assert_eq!(w.read8(5).data, 0x42);
        assert_eq!(w.read8(5 + 0x10_0000u64 * 3).data, 0x42, "repeats across the window");
        w.write32(0x5555 as u64 * 8, 0xaa);
        w.write8(0x5555 as u64 * 8, 0xaa);
        for (off, v) in [(0x2aaa, 0x55), (0x5555, 0xa0), (5, 0)] {
            command(&f, off, v);
        }
        assert_eq!(w.read8(5).data, 0x42, "the narrow writes started no command");
    }

    #[test]
    fn a_backing_file_keeps_programmed_bytes() {
        let dir = std::env::temp_dir().join(format!("iris-node-flash-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("flash.raw");
        let _ = std::fs::remove_file(&path);
        let f = NodeFlash::with_file(vec![0xff; 0x10_0000], &path).unwrap();
        for (off, v) in [(0x5555, 0xaa), (0x2aaa, 0x55), (0x5555, 0xa0), (9, 0x12)] {
            command(&f, off, v);
        }
        drop(f);
        let again = NodeFlash::with_file(vec![0xff; 0x10_0000], &path).unwrap();
        assert_eq!(again.chip().read8(9).data, 0x12);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
