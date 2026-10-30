//! Unit tests for the VPro (Odyssey) board model.
//!
//! The cases follow the register-contract tests in QEMU's maintained
//! `sgi-origami` branch (tests/qtest/octane-vpro-test.c and
//! tests/qtest/sn1-odyssey-test.c), which are grounded in captures from
//! real boards and the reviewed datasheet. A doc comment names the QEMU
//! case a test comes from. Cases that are about the Fuel's or Origin's
//! fabric, QOM properties, migration streams or the rasteriser's pixel
//! maths are not repeated here.
//!
//! The tests reach the board through `Board` and `Odyssey`, the same
//! surface the machine uses: memory-mapped reads and writes, and the
//! command stream pushed through the command-FIFO port. With no board
//! thread running, a command store runs in the store, which is how a
//! test sees its effect at once.

use super::*;
use crate::irq::InterruptController;
use crate::traits::BusDevice;
use parking_lot::Mutex;
use std::sync::Arc;

mod catalogue;
mod display;
mod dma;
mod fifo;
mod interrupts;
mod raster;
mod registers;

pub(super) const VPRO_WIDGET: u32 = 11;
const HEART_WIDGET: u64 = 8;
const HEART_PACKET: u64 = 0x80;
/// HEART vectors the tests route the board's four interrupt sources to.
pub(super) const FLAG_VECTOR: u32 = 18;
pub(super) const DMA_VECTOR: u32 = 19;
pub(super) const RETRACE_VECTOR: u32 = 20;
pub(super) const SOFTSYNC_VECTOR: u32 = 21;

// Aperture offsets the older tests name directly.
const IDENTITY: u64 = 0x0004;
const MEMORY_CONFIG: u64 = 0x106c;
const CONTROL: u64 = 0x0024;
const ROUTE_DEST_UPPER: u64 = 0x1044;
const ROUTE_DEST_LOWER: u64 = 0x104c;
const ROUTE_VECTORS: u64 = 0x1054;
const INTR_CLEAR: u64 = 0x101c;
const INTR_ENABLE_SET: u64 = 0x1034;
const INTR_FIFO_LOW_WATER: u64 = 0x200;
const TEST_VECTOR: u32 = 18;
const IDENTITY_DEFAULT: u32 = 0xac013047;

/// Records every (line, level) the board drives.
#[derive(Default)]
pub(super) struct Recorder(Mutex<Vec<(u32, bool)>>);

impl InterruptController for Recorder {
    fn set_line(&self, line: u32, level: bool) {
        self.0.lock().push((line, level));
    }
}

impl Recorder {
    /// The level of `line` after the last event on it.
    pub(super) fn level(&self, line: u32) -> bool {
        self.0.lock().iter().rev().find(|e| e.0 == line).map(|e| e.1).unwrap_or(false)
    }

    /// Every event on `line`, oldest first.
    pub(super) fn events(&self, line: u32) -> Vec<bool> {
        self.0.lock().iter().filter(|e| e.0 == line).map(|e| e.1).collect()
    }

    /// Whether the board has sent no packet at all.
    pub(super) fn silent(&self) -> bool {
        self.0.lock().is_empty()
    }
}

pub(super) fn board(product: &str) -> (Box<Board>, Arc<Recorder>) {
    let rec = Arc::new(Recorder::default());
    let s = Board::new(product, VPRO_WIDGET).expect("board");
    s.front.lock().irq = Some(rec.clone());
    (s, rec)
}

/// Guest RAM for the board's DMA: a flat big-endian byte array that
/// starts at address zero.
pub(super) struct Ram(Mutex<Vec<u8>>);

impl Ram {
    pub(super) fn new(len: usize) -> Arc<Self> {
        Arc::new(Ram(Mutex::new(vec![0; len])))
    }

    pub(super) fn put(&self, addr: u64, bytes: &[u8]) {
        self.0.lock()[addr as usize..][..bytes.len()].copy_from_slice(bytes);
    }

    pub(super) fn get(&self, addr: u64, len: usize) -> Vec<u8> {
        self.0.lock()[addr as usize..][..len].to_vec()
    }

    pub(super) fn half(&self, addr: u64) -> u16 {
        u16::from_be_bytes(self.get(addr, 2).try_into().unwrap())
    }
}

impl BusDevice for Ram {
    fn read8(&self, addr: u64) -> crate::traits::BusRead8 {
        match self.0.lock().get(addr as usize) {
            Some(b) => crate::traits::BusRead8 { status: crate::traits::BUS_OK, data: *b },
            None => crate::traits::BusRead8::err(),
        }
    }
    fn write8(&self, addr: u64, val: u8) -> u32 {
        match self.0.lock().get_mut(addr as usize) {
            Some(b) => {
                *b = val;
                crate::traits::BUS_OK
            }
            None => crate::traits::BUS_ERR,
        }
    }
    fn read32(&self, addr: u64) -> crate::traits::BusRead32 {
        match self.0.lock().get(addr as usize..addr as usize + 4) {
            Some(b) => crate::traits::BusRead32 {
                status: crate::traits::BUS_OK,
                data: u32::from_be_bytes(b.try_into().unwrap()),
            },
            None => crate::traits::BusRead32::err(),
        }
    }
    fn write32(&self, addr: u64, val: u32) -> u32 {
        match self.0.lock().get_mut(addr as usize..addr as usize + 4) {
            Some(b) => {
                b.copy_from_slice(&val.to_be_bytes());
                crate::traits::BUS_OK
            }
            None => crate::traits::BUS_ERR,
        }
    }
    fn read64(&self, addr: u64) -> crate::traits::BusRead64 {
        match self.0.lock().get(addr as usize..addr as usize + 8) {
            Some(b) => crate::traits::BusRead64 {
                status: crate::traits::BUS_OK,
                data: u64::from_be_bytes(b.try_into().unwrap()),
            },
            None => crate::traits::BusRead64::err(),
        }
    }
    fn write64(&self, addr: u64, val: u64) -> u32 {
        match self.0.lock().get_mut(addr as usize..addr as usize + 8) {
            Some(b) => {
                b.copy_from_slice(&val.to_be_bytes());
                crate::traits::BUS_OK
            }
            None => crate::traits::BUS_ERR,
        }
    }
}

/// A board whose DMA reaches 40 MiB of RAM.
pub(super) fn board_with_ram() -> (Box<Board>, Arc<Recorder>, Arc<Ram>) {
    let (mut s, rec) = board("v12");
    let ram = Ram::new(40 << 20);
    s.guest = Some(ram.clone());
    (s, rec, ram)
}

pub(super) fn reg_read(s: &mut Board, off: u64) -> u32 {
    s.mmio_read(off, 4) as u32
}

pub(super) fn reg_write(s: &mut Board, off: u64, value: u32) {
    s.mmio_write(off, value as u64, 4);
}

/// One command-FIFO word.
pub(super) fn cmd(s: &mut Board, word: u32) {
    s.mmio_write(ODSY_CMD_FIFO, word as u64, 4);
}

/// Two command-FIFO words in one 64-bit store, upper half first.
pub(super) fn cmd64(s: &mut Board, first: u32, second: u32) {
    s.mmio_write(ODSY_CMD_FIFO, (first as u64) << 32 | second as u64, 8);
}

/// A register-select word for `reg`, as the driver sends it.
pub(super) fn select(reg: u32) -> u32 {
    ODSY_CMD_REGISTER_TAG | reg
}

/// Program the routed destination to HEART's packet register, with the
/// four sources on their own vectors.
pub(super) fn route_to_heart(s: &mut Board) {
    reg_write(s, ODSY_ROUTE_DEST_UPPER, (HEART_WIDGET << 25) as u32);
    reg_write(s, ODSY_ROUTE_DEST_LOWER, HEART_PACKET as u32);
    reg_write(
        s,
        ODSY_ROUTE_VECTORS,
        SOFTSYNC_VECTOR << 24 | RETRACE_VECTOR << 16 | DMA_VECTOR << 8 | FLAG_VECTOR,
    );
}

/// The pixel at (x, y) of colour buffer `buffer`, as the plane's 0xRRGGBBAA.
pub(super) fn pixel(s: &Board, buffer: u32, x: i32, y: i32) -> u32 {
    let (mem, plane) = if buffer == 0 { (&s.mem, s.colour_plane) } else { (&s.buffer1_mem, s.buffer1_plane) };
    let p = mem.plane(plane).expect("the colour plane");
    mem.plane_load(&p, x, y)
}

/// Store a pixel into colour buffer 0, as a draw would, and mark its
/// display line for the scanout to resolve again.
pub(super) fn pixel_write(s: &mut Board, x: i32, y: i32, value: u32) {
    let p = s.mem.plane(s.colour_plane).expect("the colour plane");
    s.mem.plane_store(&p, x, y, value);
    s.scanout.invalidate(y, 1);
}
