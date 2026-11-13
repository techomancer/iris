//! Configuration NVRAM of a module system controller: one 2048-byte store on
//! the module's management I2C bus.
//!
//! Address `A` answers at seven-bit I2C address `0x50 + (A >> 8)` with the
//! one-byte word address `A & 0xff`, as a 24C16-class EEPROM does. The
//! controller owns the top block, `0x700`-`0x7ff`; the rest is the board
//! inventory log the PROM keeps.
//!
//! Model choices, not measured behaviour: the store is one device answering
//! eight addresses; the address counter wraps inside its 256-byte block
//! (a real 24C16 wraps a write inside a 16-byte page); writes complete at once
//! and are never refused as busy; the part does not answer a general call; a
//! fresh store holds what a master controller writes at its first power-up.
//!
//! The backing file is replaced whole through `nv_storage` when a transfer
//! ends with changed bytes, not on every byte.
//!
//! Ported from hw/nvram/sgi-elsc-nvram.c in jamesbraid/qemu-sgi-origami.

use crate::dev::i2c::I2cTarget;
use crate::devlog::LogModule;
use crate::dlog;
use crate::nv_storage;
use crate::traits::Resettable;
use parking_lot::Mutex;
use std::io;
use std::path::{Path, PathBuf};

/// Seven-bit address of block 0.
pub const FIRST_ADDR: u8 = 0x50;
pub const BLOCKS: usize = 8;
pub const BLOCK_BYTES: usize = 256;
pub const BYTES: usize = BLOCKS * BLOCK_BYTES;

pub const MARKER: usize = 0x700;
pub const PASSWORD: usize = 0x701;
pub const DEBUG: usize = 0x705;
pub const CONFIG: usize = 0x707;
pub const MODULE: usize = 0x708;
pub const CLOCK: usize = 0x70b;
pub const MARKER_VALUE: u8 = 0x37;

/// Configuration byte: bit 0 fans high, bit 1 automatic power-up.
pub const CONFIG_FAN_HIGH: u8 = 0x01;
pub const CONFIG_AUTO_POWER: u8 = 0x02;

/// Clock-select default a controller writes on first power-up for the
/// full-clock midplane type, which the Origin 200 build always assumes.
pub const CLOCK_FULL: u8 = 0x78;

/// The state a master controller leaves after its first power-up: zeroes,
/// then at 0x700 the validity marker, the password "none", zero debug-switch
/// values, `config`, `module`, zero self-test flags and partition, and `clock`.
pub fn fresh_image(config: u8, module: u8, clock: u8) -> [u8; BYTES] {
    let mut image = [0u8; BYTES];
    image[MARKER] = MARKER_VALUE;
    image[PASSWORD..PASSWORD + 4].copy_from_slice(b"none");
    image[CONFIG] = config;
    image[MODULE] = module;
    image[CLOCK] = clock;
    image
}

/// The controller store an Origin 200 machine supplies at power-on. The PROM
/// refuses the store unless `0x700` holds `0x37`; the password `none` at
/// `0x701`, `0x02` at `0x707` and `0x78` at `0x70b` are copied as they are
/// without the PROM's use of them being established, and every other byte is
/// zero. The machine calls this (or [`ElscNvram::origin200`]) and does not
/// build the contents itself.
pub fn origin200_image() -> [u8; BYTES] {
    fresh_image(CONFIG_AUTO_POWER, 0, CLOCK_FULL)
}

/// The controller store an Origin 2000 module starts from. The controller
/// writes this after its first power-up: the marker, the password `none`,
/// configuration byte 0 (no automatic power-up bit), the module number the
/// machine selects and clock-select byte 0, which an Origin 2000 PROM reads as
/// the domain. Every other byte is zero.
///
/// Byte `0x70b` is an approximation. A controller writes its clock-select
/// default there, `0x78` on the full-clock midplane type and 0 on the others,
/// and that this module is the full-clock type is only inferred, so it stays 0.
/// The Origin 200 image differs in the configuration byte (`0x02`) and this
/// byte (`0x78`).
pub fn origin2000_image(module: u8) -> [u8; BYTES] {
    fresh_image(0, module, 0)
}

struct State {
    mem: [u8; BYTES],
    /// Block the current transfer addressed.
    block: usize,
    /// Eleven-bit address the next data byte reads or writes.
    cur: usize,
    /// A write transfer whose word address has not arrived.
    want_word: bool,
    /// Bytes changed since the backing file was last replaced; a failed
    /// replacement leaves it set so the next transfer's end retries.
    dirty: bool,
    path: Option<PathBuf>,
}

impl State {
    fn advance(&mut self) {
        self.cur = (self.cur & !0xff) | ((self.cur + 1) & 0xff);
    }

    fn store(&mut self, data: u8) {
        if self.mem[self.cur] != data {
            self.mem[self.cur] = data;
            self.dirty = self.path.is_some();
        }
    }

    /// Replace the backing file if bytes changed. A failure is reported and
    /// the bytes stay in the store.
    fn flush(&mut self) {
        let Some(path) = self.path.as_ref().filter(|_| self.dirty) else {
            return;
        };
        match nv_storage::save(&path.to_string_lossy(), &self.mem) {
            Ok(()) => self.dirty = false,
            Err(err) => dlog!(LogModule::Module, "elsc nvram: write to {} failed: {err}", path.display()),
        }
    }

    fn start(&mut self, addr7: u8, recv: bool) {
        self.block = (addr7 - FIRST_ADDR) as usize;
        if recv {
            // A read continues from the counter inside the block this
            // transfer addressed.
            self.cur = (self.block << 8) | (self.cur & 0xff);
            self.want_word = false;
        } else {
            self.want_word = true;
        }
    }

    fn send(&mut self, byte: u8) {
        if self.want_word {
            self.cur = (self.block << 8) | byte as usize;
            self.want_word = false;
            return;
        }
        self.store(byte);
        self.advance();
    }

    fn recv(&mut self) -> u8 {
        let byte = self.mem[self.cur];
        self.advance();
        byte
    }
}

pub struct ElscNvram {
    state: Mutex<State>,
}

impl ElscNvram {
    fn from_image(mem: [u8; BYTES], path: Option<PathBuf>) -> Self {
        Self {
            state: Mutex::new(State {
                mem,
                block: 0,
                cur: 0,
                want_word: false,
                dirty: false,
                path,
            }),
        }
    }

    /// The Origin 200 machine's store: [`origin200_image`], held in memory
    /// when `path` is `None`, or backed by that file as [`Self::open`] does.
    pub fn origin200(path: Option<&Path>) -> io::Result<Self> {
        match path {
            None => Ok(Self::from_image(origin200_image(), None)),
            Some(p) => Self::open(p, CONFIG_AUTO_POWER, 0, CLOCK_FULL),
        }
    }

    /// The Origin 2000 module's store for module number `module`: see
    /// [`origin2000_image`]. In memory when `path` is `None`, otherwise
    /// backed by that file as [`Self::open`] does, its contents winning.
    pub fn origin2000(module: u8, path: Option<&Path>) -> io::Result<Self> {
        match path {
            None => Ok(Self::from_image(origin2000_image(module), None)),
            Some(p) => Self::open(p, 0, module, 0),
        }
    }

    /// A store with no backing file, holding a fresh image.
    pub fn new(config: u8, module: u8, clock: u8) -> Self {
        Self::from_image(fresh_image(config, module, clock), None)
    }

    /// A store backed by `path`. A file that exists must be exactly
    /// [`BYTES`] long and its contents win over the fresh-image values;
    /// a missing file is created holding the fresh image.
    pub fn open(path: &Path, config: u8, module: u8, clock: u8) -> io::Result<Self> {
        let mem = if path.exists() {
            let bytes = std::fs::read(path)?;
            <[u8; BYTES]>::try_from(bytes.as_slice()).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{}: the controller NVRAM file is {} bytes, not {BYTES}",
                        path.display(),
                        bytes.len()
                    ),
                )
            })?
        } else {
            let mem = fresh_image(config, module, clock);
            nv_storage::save(&path.to_string_lossy(), &mem)?;
            mem
        };
        Ok(Self::from_image(mem, Some(path.to_path_buf())))
    }

    /// One byte of the store, without moving the address counter.
    pub fn peek(&self, addr: usize) -> u8 {
        self.state.lock().mem[addr % BYTES]
    }

    pub fn contents(&self) -> [u8; BYTES] {
        self.state.lock().mem
    }

    /// One complete transfer as the controller makes it to a part of its own:
    /// the word address written, then a byte written or read. Answers `None`
    /// when no block answers `addr7`. The address counter ends where a real
    /// access leaves it.
    pub fn access(&self, addr7: u8, offset: u8, write: Option<u8>) -> Option<u8> {
        if !self.claims(addr7) {
            return None;
        }
        let mut st = self.state.lock();
        st.start(addr7, false);
        st.send(offset);
        let value = match write {
            Some(byte) => {
                st.send(byte);
                byte
            }
            None => {
                st.start(addr7, true);
                st.recv()
            }
        };
        st.want_word = false;
        st.flush();
        Some(value)
    }
}

impl I2cTarget for ElscNvram {
    fn claims(&self, addr7: u8) -> bool {
        (FIRST_ADDR..FIRST_ADDR + BLOCKS as u8).contains(&addr7)
    }

    fn answers_general_call(&self) -> bool {
        false
    }

    fn start(&self, addr7: u8, recv: bool) -> bool {
        self.state.lock().start(addr7, recv);
        true
    }

    fn send(&self, byte: u8) -> bool {
        self.state.lock().send(byte);
        true
    }

    fn recv(&self) -> u8 {
        self.state.lock().recv()
    }

    fn finish(&self) {
        let mut st = self.state.lock();
        st.want_word = false;
        st.flush();
    }
}

impl Resettable for ElscNvram {
    /// Only the transfer state resets; the store is non-volatile.
    fn power_on(&self) {
        let mut st = self.state.lock();
        st.block = 0;
        st.cur = 0;
        st.want_word = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::i2c::I2cBus;
    use std::sync::Arc;

    /// A bus with the store and a master that does what the PROM does.
    struct Rig {
        bus: I2cBus,
        nvram: Arc<ElscNvram>,
    }

    impl Rig {
        fn new(nvram: ElscNvram) -> Self {
            let nvram = Arc::new(nvram);
            let mut bus = I2cBus::new();
            bus.add(nvram.clone());
            Self { bus, nvram }
        }

        fn begin_write(&mut self, addr: usize, word: u8) {
            let a7 = FIRST_ADDR + (addr >> 8) as u8;
            self.bus.scan(a7);
            assert!(self.bus.start_selected(a7, false, true));
            assert!(self.bus.send(word));
        }

        fn write(&mut self, addr: usize, data: &[u8]) {
            self.begin_write(addr, addr as u8);
            for b in data {
                assert!(self.bus.send(*b));
            }
            self.bus.end();
        }

        fn read(&mut self, addr: usize) -> u8 {
            let a7 = FIRST_ADDR + (addr >> 8) as u8;
            self.begin_write(addr, addr as u8);
            assert!(self.bus.start_selected(a7, true, false));
            let v = self.bus.recv();
            self.bus.nack();
            self.bus.end();
            v
        }
    }

    fn fresh() -> Rig {
        Rig::new(ElscNvram::new(CONFIG_AUTO_POWER, 0, CLOCK_FULL))
    }

    #[test]
    fn new_store_defaults() {
        let mut rig = fresh();
        assert_eq!(rig.read(MARKER), 0x37);
        for (i, ch) in b"none".iter().enumerate() {
            assert_eq!(rig.read(PASSWORD + i), *ch);
        }
        assert_eq!(rig.read(0x705), 0);
        assert_eq!(rig.read(0x706), 0);
        assert_eq!(rig.read(CONFIG), 0x02);
        assert_eq!(rig.read(MODULE), 0);
        assert_eq!(rig.read(0x709), 0);
        assert_eq!(rig.read(0x70a), 0);
        assert_eq!(rig.read(CLOCK), 0x78);
        assert_eq!(rig.read(0x70c), 0);
        assert_eq!(rig.read(0x70d), 0);
        // Everything outside the controller's block is zero.
        assert_eq!(rig.read(0x000), 0);
        assert_eq!(rig.read(0x6ff), 0);
        assert_eq!(rig.read(0x70e), 0);
    }

    #[test]
    fn the_origin_200_image_is_the_marker_and_the_copied_bytes_only() {
        let image = origin200_image();
        let mut want = [0u8; BYTES];
        want[0x700] = 0x37;
        want[0x701..0x705].copy_from_slice(b"none");
        want[0x707] = 0x02;
        want[0x70b] = 0x78;
        assert_eq!(image, want);
        assert_eq!(ElscNvram::origin200(None).unwrap().contents(), want);
    }

    #[test]
    fn the_origin_2000_image_has_no_auto_power_bit_and_the_selected_module() {
        let image = origin2000_image(3);
        let mut want = [0u8; BYTES];
        want[0x700] = 0x37;
        want[0x701..0x705].copy_from_slice(b"none");
        want[0x708] = 3;
        assert_eq!(image, want);
        assert_eq!(ElscNvram::origin2000(3, None).unwrap().contents(), want);
    }

    #[test]
    fn blocks_are_addressed_by_the_i2c_address() {
        let mut rig = fresh();
        for block in 0..BLOCKS {
            rig.write(block * 0x100 + 0x10, &[0xa0 + block as u8]);
        }
        for block in 0..BLOCKS {
            assert_eq!(rig.read(block * 0x100 + 0x10), 0xa0 + block as u8);
            // The same word address in the neighbouring block is untouched.
            assert_ne!(rig.read(block * 0x100 + 0x11), 0xa0 + block as u8);
        }
        assert!(!rig.nvram.claims(0x4f));
        assert!(!rig.nvram.claims(0x58));
    }

    #[test]
    fn twelve_byte_block_write_is_one_transaction() {
        let mut rig = fresh();
        let data: Vec<u8> = (1..=12).collect();
        rig.write(MARKER, &data);
        for (i, b) in data.iter().enumerate() {
            assert_eq!(rig.read(MARKER + i), *b);
        }
    }

    #[test]
    fn sequential_access_wraps_inside_the_block() {
        let mut rig = fresh();
        rig.write(0x1fe, &[0xaa, 0xbb, 0xcc]);
        assert_eq!(rig.read(0x1fe), 0xaa);
        assert_eq!(rig.read(0x1ff), 0xbb);
        // The third byte wrapped to the first of the same block.
        assert_eq!(rig.read(0x100), 0xcc);
        assert_eq!(rig.read(0x200), 0);
    }

    #[test]
    fn survives_a_reset_and_does_not_answer_a_general_call() {
        let mut rig = fresh();
        rig.write(0x123, &[0x42]);
        rig.nvram.power_on();
        assert_eq!(rig.read(0x123), 0x42);
        assert_eq!(rig.read(MARKER), 0x37);
        rig.bus.scan(0);
        assert!(!rig.bus.busy());
    }

    fn scratch_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("iris-elsc-nvram-{tag}-{}.raw", std::process::id()))
    }

    #[test]
    fn caller_image_wins_over_the_defaults() {
        let path = scratch_path("wins");
        let mut image = [0u8; BYTES];
        image[CONFIG] = 0x01;
        image[MODULE] = 9;
        std::fs::write(&path, image).unwrap();
        let nv = ElscNvram::open(&path, CONFIG_AUTO_POWER, 0, CLOCK_FULL).unwrap();
        // The file's zeros stand: no marker, no password, no clock byte.
        assert_eq!(nv.peek(MARKER), 0);
        assert_eq!(nv.peek(CONFIG), 0x01);
        assert_eq!(nv.peek(MODULE), 9);
        assert_eq!(nv.peek(CLOCK), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_missing_file_is_created_fresh_and_a_wrong_size_is_refused() {
        let path = scratch_path("size");
        let _ = std::fs::remove_file(&path);
        let nv = ElscNvram::open(&path, CONFIG_AUTO_POWER, 3, CLOCK_FULL).unwrap();
        assert_eq!(nv.peek(MODULE), 3);
        assert_eq!(std::fs::read(&path).unwrap(), nv.contents().to_vec());
        drop(nv);
        std::fs::write(&path, [0u8; 100]).unwrap();
        let err = ElscNvram::open(&path, 0, 0, 0).err().unwrap();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn persists_across_instances() {
        let path = scratch_path("persist");
        let _ = std::fs::remove_file(&path);
        {
            let mut rig = Rig::new(ElscNvram::open(&path, CONFIG_AUTO_POWER, 0, CLOCK_FULL).unwrap());
            rig.write(0x42, &[0x99]);
            rig.write(MODULE, &[5]);
        }
        let nv = ElscNvram::open(&path, 0, 0, 0).unwrap();
        assert_eq!(nv.peek(0x42), 0x99);
        assert_eq!(nv.peek(MODULE), 5);
        // The fresh-store values it was not given this time do not apply.
        assert_eq!(nv.peek(CLOCK), CLOCK_FULL);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_controllers_own_access_moves_the_counter_like_a_transfer() {
        let nv = ElscNvram::new(CONFIG_AUTO_POWER, 0, CLOCK_FULL);
        assert_eq!(nv.access(0x57, 0x07, None), Some(0x02));
        assert_eq!(nv.access(0x57, 0x07, Some(0x03)), Some(0x03));
        assert_eq!(nv.peek(CONFIG), 0x03);
        assert_eq!(nv.access(0x08, 0, None), None);
    }
}
