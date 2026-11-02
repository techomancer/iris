// Derived from MAME's intelfsh.cpp (the 16-bit Intel/Sharp command path),
// BSD-3-Clause, copyright Aaron Giles.

//! The Octane's boot flash: a 16-bit, big-endian part with the Intel/Sharp
//! command set, for [`Bridge::attach_flash`], at physical 0x1fc0_0000.
//!
//! The fitted chip is not identified. This is a compatibility part chosen to
//! satisfy the stock PROM: a 1 MiB array of sixteen 64 KiB erase blocks, manufacturer
//! `0x00b0` (Sharp) and device `0x66a8` as 16-bit identifier words, and the
//! command subset the stock PROM drives (identify, read status, clear status,
//! word program, block erase). The state machine is a port of the Intel/Sharp
//! path of MAME's `intelfsh`; `0x66a8` is an added part.
//!
//! Approximations, each deliberate:
//! - Program and erase finish instantly: the status register is ready the moment
//!   the data or confirming cycle lands.
//! - Programming ANDs the word in, as NOR cells do: a program cycle can only clear
//!   bits, and only an erase sets them again. (MAME overwrites the word.)
//! - The enhanced command set (page buffers, queued operations, extended and
//!   per-block status, block locking, suspend) is absent. Such a command is
//!   logged and ignored, so a guest finds it unsupported rather than
//!   silently successful.
//! - Status reads repeat the status byte on both data lanes.
//!
//! Persistence: [`CfiFlash::with_file`] backs the array with a file. An existing
//! file of the right size replaces the seed image, a missing or empty file is
//! created from it, and a file of any other size is refused rather than
//! overwritten. Completed program and erase cycles are written through to the file.
//!
//! [`Bridge::attach_flash`]: crate::dev::bridge::Bridge::attach_flash

use std::cell::Cell;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::Mutex;

use crate::traits::{BusDevice, BusRead16, BUS_ERR, BUS_OK};

pub(crate) mod storage;
pub(crate) mod amd;
use self::storage::FlashBacking;

/// The Octane's part: 1 MiB.
pub const FLASH_SIZE: usize = 1 << 20;
/// Erase block size.
pub const BLOCK_SIZE: usize = 64 * 1024;
/// Identifier words, read at part byte offsets 0 and 2 in identifier mode.
pub const MANUFACTURER_ID: u16 = 0x00b0;
pub const DEVICE_ID: u16 = 0x66a8;

// Command bytes, taken from DQ7:0 of a write cycle.
const CMD_PROGRAM: u8 = 0x40;
const CMD_PROGRAM_ALT: u8 = 0x10;
const CMD_ERASE: u8 = 0x20;
const CMD_CONFIRM: u8 = 0xd0;
const CMD_READ_STATUS: u8 = 0x70;
const CMD_CLEAR_STATUS: u8 = 0x50;
const CMD_READ_ID: u8 = 0x90;
const CMD_READ_ARRAY: u8 = 0xff;

// Status register bits.
const SR_READY: u8 = 0x80;
const SR_ERASE_ERR: u8 = 0x20;
const SR_PROGRAM_ERR: u8 = 0x10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Array,
    Id,
    Status,
    /// The next write cycle is the data word to program.
    ProgramData,
    /// An erase setup was seen; the next write must be the confirm.
    EraseConfirm,
}

struct State {
    data: Vec<u8>,
    mode: Mode,
    status: u8,
    backing: Option<FlashBacking>,
}

pub struct CfiFlash {
    st: Mutex<State>,
    /// jitv2's code-generation counter for the whole part, bumped by every
    /// write cycle: a command changes what a read returns.
    gen: AtomicU64,
}

/// The name this part had when it was only a PROM store.
pub type Flash = CfiFlash;

thread_local! {
    static FILLING_A_CACHE_LINE: Cell<bool> = const { Cell::new(false) };
}

/// Run `f`, whose flash reads see the array whatever command the part is in.
/// For a CPU's cached view of the flash, which on the hardware holds lines
/// from before the command: a second CPU keeps running out of its cache while
/// the first has the part in status mode.
pub fn reading_the_array<R>(f: impl FnOnce() -> R) -> R {
    FILLING_A_CACHE_LINE.with(|c| c.set(true));
    let r = f();
    FILLING_A_CACHE_LINE.with(|c| c.set(false));
    r
}

impl CfiFlash {
    /// A part of `size` bytes (a multiple of the block size, a power of two)
    /// holding `image` from offset 0 and erased (0xff) above it.
    pub fn new(image: &[u8], size: usize) -> Self {
        assert!(size.is_power_of_two() && size >= BLOCK_SIZE, "flash size must be a power of two of at least one block");
        assert!(image.len() <= size, "image must fit the part");
        let mut data = vec![0xff; size];
        data[..image.len()].copy_from_slice(image);
        Self { st: Mutex::new(State { data, mode: Mode::Array, status: SR_READY, backing: None }), gen: AtomicU64::new(0) }
    }

    /// A reset: back to read-array mode with the status clear. The array
    /// keeps what was programmed.
    pub fn reset(&self) {
        let mut st = self.st.lock();
        st.mode = Mode::Array;
        st.status = SR_READY;
    }

    /// As [`CfiFlash::new`], backed by `path`. A file of exactly `size` bytes is
    /// the array's contents (a previous run's programming survives, and the
    /// seed image is ignored). A missing or empty file is created from the seeded
    /// array. A file of any other size is refused with `InvalidData` and left
    /// untouched, since it is probably some other file.
    pub fn with_file(image: &[u8], size: usize, path: impl AsRef<Path>) -> io::Result<Self> {
        let flash = Self::new(image, size);
        let mut st = flash.st.lock();
        let backing = FlashBacking::open(path, &mut st.data)?;
        st.backing = Some(backing);
        drop(st);
        Ok(flash)
    }

    pub fn size(&self) -> usize {
        self.st.lock().data.len()
    }

    pub fn register_locks(self: &Arc<Self>) {
        let me = self.clone();
        crate::locks::register_lock_fn("sflash::state", move || me.st.is_locked());
    }

    /// A copy of the array.
    pub fn contents(&self) -> Vec<u8> {
        self.st.lock().data.clone()
    }
}

impl State {
    fn sync(&mut self, off: usize, len: usize) {
        if let Some(backing) = self.backing.as_mut() {
            let res = backing.write_range(off, &self.data[off..off + len]);
            if let Err(e) = res {
                // The guest sees a completed cycle; the file is only the backing store.
                log::warn!("flash: backing file write at {off:#x} failed: {e}");
            }
        }
    }

    fn word(&self, off: usize) -> u16 {
        u16::from_be_bytes([self.data[off], self.data[off + 1]])
    }

    fn read(&self, off: usize) -> u16 {
        if self.mode != Mode::Array && FILLING_A_CACHE_LINE.with(Cell::get) {
            return self.word(off);
        }
        match self.mode {
            Mode::Array => self.word(off),
            Mode::Id => match off {
                0 => MANUFACTURER_ID,
                2 => DEVICE_ID,
                _ => 0,
            },
            // Any read in status mode returns the status register, on both lanes.
            Mode::Status | Mode::ProgramData | Mode::EraseConfirm => u16::from_be_bytes([self.status; 2]),
        }
    }

    fn write(&mut self, off: usize, val: u16) {
        let cmd = val as u8;
        match self.mode {
            Mode::ProgramData => {
                let new = (self.word(off) & val).to_be_bytes();
                self.data[off..off + 2].copy_from_slice(&new);
                self.sync(off, 2);
                self.mode = Mode::Status;
            }
            Mode::EraseConfirm => {
                if cmd == CMD_CONFIRM {
                    let base = off & !(BLOCK_SIZE - 1);
                    self.data[base..base + BLOCK_SIZE].fill(0xff);
                    self.sync(base, BLOCK_SIZE);
                } else {
                    // Erase setup not followed by the confirm: both error bits.
                    self.status |= SR_ERASE_ERR | SR_PROGRAM_ERR;
                }
                self.mode = Mode::Status;
            }
            _ => match cmd {
                CMD_READ_ARRAY => self.mode = Mode::Array,
                CMD_READ_ID => self.mode = Mode::Id,
                CMD_READ_STATUS => self.mode = Mode::Status,
                CMD_CLEAR_STATUS => {
                    self.status &= !(SR_ERASE_ERR | SR_PROGRAM_ERR | 0x08);
                    self.mode = Mode::Status;
                }
                CMD_PROGRAM | CMD_PROGRAM_ALT => self.mode = Mode::ProgramData,
                CMD_ERASE => self.mode = Mode::EraseConfirm,
                other => {
                    log::warn!("flash: unsupported command {other:#04x} at {off:#x} ignored");
                }
            },
        }
    }
}

impl BusDevice for CfiFlash {
    fn read16(&self, addr: u64) -> BusRead16 {
        let st = self.st.lock();
        let off = addr as usize & !1;
        if off >= st.data.len() {
            return BusRead16::err();
        }
        BusRead16::ok(st.read(off))
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        let mut st = self.st.lock();
        let off = addr as usize & !1;
        if off >= st.data.len() {
            return BUS_ERR;
        }
        st.write(off, val);
        self.gen.fetch_add(1, Ordering::Release);
        BUS_OK
    }

    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, _addr: u64) -> *const AtomicU64 {
        &self.gen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: usize = 1 << 20;

    fn part() -> CfiFlash {
        CfiFlash::new(&[0x12, 0x34, 0x56, 0x78], SIZE)
    }

    fn rd(f: &CfiFlash, a: u64) -> u16 {
        let r = f.read16(a);
        assert!(r.is_ok());
        r.data
    }

    fn wr(f: &CfiFlash, a: u64, v: u16) {
        assert_eq!(f.write16(a, v), BUS_OK);
    }

    fn program(f: &CfiFlash, a: u64, v: u16) -> u16 {
        wr(f, a, 0x4040);
        wr(f, a, v);
        wr(f, a, 0x7070);
        rd(f, a)
    }

    fn erase(f: &CfiFlash, a: u64) -> u16 {
        wr(f, a, 0x2020);
        wr(f, a, 0xd0d0);
        wr(f, a, 0x7070);
        rd(f, a)
    }

    #[test]
    fn array_reads_and_erased_fill() {
        let f = part();
        assert_eq!(rd(&f, 0), 0x1234);
        assert_eq!(rd(&f, 2), 0x5678);
        assert_eq!(rd(&f, 4), 0xffff);
        assert_eq!(f.size(), SIZE);
        assert!(!f.read16(SIZE as u64).is_ok());
    }

    #[test]
    fn identifier_words_and_return_to_array() {
        let f = part();
        wr(&f, 0, 0x9090);
        assert_eq!(rd(&f, 0), 0x00b0);
        assert_eq!(rd(&f, 2), 0x66a8);
        wr(&f, 0, 0xffff);
        assert_eq!(rd(&f, 0), 0x1234);
    }

    #[test]
    fn program_sets_ready_and_stores() {
        let f = part();
        let sr = program(&f, 0x1000, 0xbeef);
        assert_eq!(sr & 0xff, 0x80);
        wr(&f, 0, 0xffff);
        assert_eq!(rd(&f, 0x1000), 0xbeef);
        // The data word may equal a command opcode, including the array command.
        assert_eq!(program(&f, 0x1002, 0xffff) & 0xff, 0x80);
        assert_eq!(program(&f, 0x1004, 0x2020) & 0xff, 0x80);
        wr(&f, 0, 0xffff);
        assert_eq!([rd(&f, 0x1002), rd(&f, 0x1004)], [0xffff, 0x2020]);
    }

    /// Programming clears bits and cannot set them: 0x5678 with 0x0f0f
    /// written is 0x0608, and only an erase brings ones back.
    #[test]
    fn program_ands_into_the_word() {
        let f = part();
        assert_eq!(program(&f, 2, 0x0f0f) & 0xff, 0x80);
        wr(&f, 0, 0xffff);
        assert_eq!(rd(&f, 2), 0x0608);
        program(&f, 2, 0xf0f0);
        wr(&f, 0, 0xffff);
        assert_eq!(rd(&f, 2), 0x0000);
        erase(&f, 2);
        wr(&f, 0, 0xffff);
        assert_eq!(rd(&f, 2), 0xffff);
    }

    #[test]
    fn clear_status_drops_errors() {
        let f = part();
        wr(&f, 0x20000, 0x2020);
        wr(&f, 0x20000, 0x0000); // not a confirm
        wr(&f, 0x20000, 0x7070);
        assert_eq!(rd(&f, 0) & 0xff, 0xb0);
        wr(&f, 0, 0x5050);
        assert_eq!(rd(&f, 0) & 0xff, 0x80);
    }

    #[test]
    fn erase_one_block_only() {
        let f = part();
        program(&f, 0x10000, 0x0000);
        program(&f, 0x20000, 0x0000);
        program(&f, 0x2fffe, 0x0000);
        let sr = erase(&f, 0x20004);
        assert_eq!(sr & 0xff, 0x80);
        wr(&f, 0, 0xffff);
        assert_eq!(rd(&f, 0x10000), 0);
        assert_eq!(rd(&f, 0x20000), 0xffff);
        assert_eq!(rd(&f, 0x2fffe), 0xffff);
        // Reprogram after the erase works.
        assert_eq!(program(&f, 0x20000, 0xa5a5) & 0xff, 0x80);
    }

    #[test]
    fn erase_without_confirm_is_a_command_error() {
        let f = part();
        wr(&f, 0x20000, 0x2020);
        wr(&f, 0x20000, 0x1234);
        wr(&f, 0x20000, 0x7070);
        assert_eq!(rd(&f, 0) & 0xff, 0xb0); // ready, erase and program error
        wr(&f, 0, 0x5050);
        wr(&f, 0, 0x7070);
        assert_eq!(rd(&f, 0) & 0xff, 0x80);
    }

    #[test]
    fn status_is_returned_for_any_address() {
        let f = part();
        wr(&f, 0, 0x7070);
        assert_eq!(rd(&f, 0), 0x8080);
        assert_eq!(rd(&f, 0x8000), 0x8080);
    }

    #[test]
    fn unsupported_command_is_ignored() {
        let f = part();
        wr(&f, 0, 0x7070);
        wr(&f, 0, 0x7171);
        assert_eq!(rd(&f, 0), 0x8080); // still in status mode
        // A CPU's cached view is not fooled by it.
        assert_ne!(reading_the_array(|| rd(&f, 0)), 0x8080);
    }

    #[test]
    fn file_backing_seeds_then_reloads() {
        let path = std::env::temp_dir().join(format!("iris-flash-test-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        {
            let f = CfiFlash::with_file(&[0xaa, 0xbb], SIZE, &path).unwrap();
            assert_eq!(rd(&f, 0), 0xaabb);
            assert_eq!(program(&f, 0x40, 0x1357) & 0xff, 0x80);
            assert_eq!(erase(&f, 0xf0000) & 0xff, 0x80);
            program(&f, 0xf0002, 0x2468);
        }
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(on_disk.len(), SIZE);
        assert_eq!(&on_disk[..2], &[0xaa, 0xbb]);
        assert_eq!(&on_disk[0x40..0x42], &[0x13, 0x57]);
        assert_eq!(&on_disk[0xf0002..0xf0004], &[0x24, 0x68]);
        // A reload takes the file, not the seed image.
        let g = CfiFlash::with_file(&[0x00, 0x00], SIZE, &path).unwrap();
        assert_eq!(rd(&g, 0), 0xaabb);
        assert_eq!(rd(&g, 0x40), 0x1357);
        assert_eq!(rd(&g, 0xf0002), 0x2468);
        drop(g);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn a_wrong_sized_file_is_refused_and_left_alone() {
        let path = std::env::temp_dir().join(format!("iris-flash-size-{}", std::process::id()));
        std::fs::write(&path, [1, 2, 3]).unwrap();
        let err = CfiFlash::with_file(&[0xcc, 0xdd], SIZE, &path).err().expect("refused");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(std::fs::read(&path).unwrap(), [1, 2, 3]);
        // An empty file is a new one and gets the seed.
        std::fs::write(&path, []).unwrap();
        let h = CfiFlash::with_file(&[0xcc, 0xdd], SIZE, &path).unwrap();
        assert_eq!(rd(&h, 0), 0xccdd);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), SIZE as u64);
        std::fs::remove_file(&path).unwrap();
    }
}
