//! The BaseIO card's boot flash: a 1 MiB, 512K x 16 top-boot part on the
//! Bridge's flash port, holding the I/O PROM the IP27 PROM copies into memory
//! at "Loading BASEIO prom". Without it the IP27 PROM falls back to its own
//! reduced copy, which has no graphics console.
//!
//! The part answers the AMD command set far enough for the PROM's loader to
//! accept it: reset (0xf0) and the autoselect sequence, with manufacturer
//! 0x01 and device 0x22da, one of the three identities the loader accepts.
//! Which part SGI fitted is not established. The array is read-only here:
//! program and erase cycles are logged and refused. The unlock addresses are
//! x16 word addresses, matched on their low eleven bits as the part does.
//!
//! Ported from hw/mips/sgi/baseio-flash.c in jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;
use crate::traits::{BusDevice, BusRead16, BUS_ERR, BUS_OK};
use parking_lot::Mutex;

pub const SIZE: usize = 1 << 20;
pub const MANUFACTURER: u16 = 0x01;
pub const DEVICE: u16 = 0x22da;
const UNLOCK_0: usize = 0x555;
const UNLOCK_1: usize = 0x2aa;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Array,
    Unlock1,
    Unlock2,
    Autoselect,
}

pub struct BaseioFlash {
    image: Box<[u8]>,
    mode: Mutex<Mode>,
}

impl BaseioFlash {
    /// The part holding `file` as supplied, with erased bytes (0xff) after it.
    pub fn new(file: &[u8]) -> Result<Self, String> {
        if file.len() > SIZE {
            return Err(format!("the I/O PROM image is {} bytes; the flash holds {SIZE}", file.len()));
        }
        let mut image = vec![0xff; SIZE].into_boxed_slice();
        image[..file.len()].copy_from_slice(file);
        Ok(Self { image, mode: Mutex::new(Mode::Array) })
    }
}

impl BusDevice for BaseioFlash {
    fn read16(&self, addr: u64) -> BusRead16 {
        let a = addr as usize & !1;
        if *self.mode.lock() == Mode::Autoselect {
            // The identification words sit at the first two word addresses.
            return BusRead16::ok(match a / 2 & 0xff {
                0 => MANUFACTURER,
                1 => DEVICE,
                _ => 0,
            });
        }
        match self.image.get(a..a + 2) {
            Some(b) => BusRead16::ok(u16::from_be_bytes([b[0], b[1]])),
            None => BusRead16::err(),
        }
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        let (word, cmd) = (addr as usize / 2 & 0x7ff, val as u8);
        let mut mode = self.mode.lock();
        if cmd == 0xf0 {
            *mode = Mode::Array;
            return BUS_OK;
        }
        *mode = match (*mode, word, cmd) {
            (Mode::Array, UNLOCK_0, 0xaa) => Mode::Unlock1,
            (Mode::Unlock1, UNLOCK_1, 0x55) => Mode::Unlock2,
            (Mode::Unlock2, UNLOCK_0, 0x90) => Mode::Autoselect,
            (m, _, _) => {
                crate::dlog!(LogModule::Hub, "BaseIO flash: unsupported cycle {val:#06x} at byte offset {addr:#x} in {m:?}");
                *mode = Mode::Array;
                return BUS_ERR;
            }
        };
        BUS_OK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cycles(f: &BaseioFlash, list: &[(u64, u16)]) {
        for &(a, v) in list {
            assert_eq!(f.write16(a, v), BUS_OK, "{a:#x} {v:#x}");
        }
    }

    #[test]
    fn the_array_holds_the_file_then_erased_bytes() {
        let f = BaseioFlash::new(&[0x12, 0x34, 0x56]).unwrap();
        assert_eq!(f.read16(0).data, 0x1234);
        assert_eq!(f.read16(2).data, 0x56ff);
        assert_eq!(f.read16(4).data, 0xffff);
        assert_eq!(f.read16((SIZE - 2) as u64).data, 0xffff);
        assert!(f.read16(SIZE as u64).status != BUS_OK);
        assert!(BaseioFlash::new(&vec![0; SIZE + 1]).is_err());
    }

    /// The PROM's loader writes the unlock cycles at byte offsets 0xaaaa and
    /// 0x5554, and reads the codes at the first two words.
    #[test]
    fn autoselect_answers_the_codes_and_reset_returns_the_array() {
        let f = BaseioFlash::new(&[0xab, 0xcd, 0xef, 0x01]).unwrap();
        cycles(&f, &[(0xaaaa, 0xaa), (0x5554, 0x55), (0xaaaa, 0x90)]);
        assert_eq!(f.read16(0).data, MANUFACTURER);
        assert_eq!(f.read16(2).data, DEVICE);
        cycles(&f, &[(0, 0xf0)]);
        assert_eq!(f.read16(0).data, 0xabcd);
        assert_eq!(f.read16(2).data, 0xef01);
    }

    #[test]
    fn program_and_erase_are_refused_and_leave_the_array_alone() {
        let f = BaseioFlash::new(&[0xab, 0xcd]).unwrap();
        cycles(&f, &[(0xaaaa, 0xaa), (0x5554, 0x55)]);
        assert_eq!(f.write16(0xaaaa, 0xa0), BUS_ERR);
        assert_eq!(f.read16(0).data, 0xabcd);
        cycles(&f, &[(0xaaaa, 0xaa), (0x5554, 0x55)]);
        assert_eq!(f.write16(0xaaaa, 0x80), BUS_ERR);
        // A stray cycle in the array mode is refused too, and nothing sticks.
        assert_eq!(f.write16(0x40, 0x55), BUS_ERR);
        assert_eq!(f.read16(0).data, 0xabcd);
    }
}
