//! The BaseIO card's boot flash: a 1 MiB, 512K x 16 top-boot part on the
//! Bridge's flash port, holding the I/O PROM the IP27 PROM copies into memory
//! at "Loading BASEIO prom". Without it the IP27 PROM falls back to its own
//! reduced copy, which has no graphics console.
//!
//! The part answers the AMD command set ([`crate::dev::sflash::amd`]) far
//! enough for the PROM's loader to accept it: reset (0xf0) and the autoselect
//! sequence, with manufacturer 0x01 and device 0x22da, one of the three
//! identities the loader accepts. Which part SGI fitted is not established.
//! The array is read-only here: program and erase cycles are logged and
//! refused. The unlock addresses are x16 word addresses, matched on their low
//! eleven bits as the part does.
//!
//! Ported from hw/mips/sgi/baseio-flash.c in jamesbraid/qemu-sgi-origami.

use crate::dev::sflash::amd::{AmdCore, Geometry, Identity};
use crate::devlog::LogModule;
use crate::traits::{BusDevice, BusRead16, BUS_ERR, BUS_OK};
use parking_lot::Mutex;

pub const SIZE: usize = 1 << 20;
pub const MANUFACTURER: u16 = 0x01;
pub const DEVICE: u16 = 0x22da;

/// The part, to the command engine: 512K x 16, unlock addresses matched on
/// their low eleven bits, the identification words at the first two word
/// addresses, and program and erase refused.
const GEOMETRY: Geometry = Geometry {
    x8: false,
    sector_size: 0x1_0000,
    sectors: SIZE / 0x1_0000,
    unlock: [0x555, 0x2aa],
    address_mask: 0x7ff,
    identity: Identity { codes: [MANUFACTURER, DEVICE, 0], other: 0 },
    cfi: false,
    read_only: true,
    timed_erase: false,
    unlock_from_identify: false,
    log: LogModule::Hub,
};

pub struct BaseioFlash {
    image: Box<[u8]>,
    core: Mutex<AmdCore>,
}

impl BaseioFlash {
    /// The part holding `file` as supplied, with erased bytes (0xff) after it.
    pub fn new(file: &[u8]) -> Result<Self, String> {
        if file.len() > SIZE {
            return Err(format!("the I/O PROM image is {} bytes; the flash holds {SIZE}", file.len()));
        }
        let mut image = vec![0xff; SIZE].into_boxed_slice();
        image[..file.len()].copy_from_slice(file);
        Ok(Self { image, core: Mutex::new(AmdCore::new(GEOMETRY)) })
    }
}

impl BusDevice for BaseioFlash {
    fn read16(&self, addr: u64) -> BusRead16 {
        let a = addr as usize & !1;
        if let Some(code) = self.core.lock().read16(a) {
            return BusRead16::ok(code);
        }
        match self.image.get(a..a + 2) {
            Some(b) => BusRead16::ok(u16::from_be_bytes([b[0], b[1]])),
            None => BusRead16::err(),
        }
    }

    fn write16(&self, addr: u64, val: u16) -> u32 {
        // The engine logs a cycle it does not take, including the program and
        // erase this part refuses.
        match self.core.lock().write16(addr as usize, val) {
            Ok(_) => BUS_OK,
            Err(()) => BUS_ERR,
        }
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
