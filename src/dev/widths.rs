//! The width adapter the node controllers' bus devices share: [`access`]
//! splits one sized access, and [`access_widths!`] writes a device's eight
//! sized `BusDevice` reads and writes from an inherent method.

use crate::traits::{BusDevice, BusRead64, BUS_ERR};

/// An access of `size` bytes to `device`, a write when `value` is given.
#[inline(always)]
pub fn access<D: BusDevice + ?Sized>(device: &D, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
    if let Some(v) = value {
        let status = match size {
            1 => device.write8(a, v as u8),
            2 => device.write16(a, v as u16),
            4 => device.write32(a, v as u32),
            8 => device.write64(a, v),
            _ => BUS_ERR,
        };
        return BusRead64 { status, data: 0 };
    }
    match size {
        1 => {
            let r = device.read8(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        }
        2 => {
            let r = device.read16(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        }
        4 => {
            let r = device.read32(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        }
        8 => device.read64(a),
        _ => BusRead64::err(),
    }
}

/// The eight sized `BusDevice` reads and writes of a device that implements
/// inherent methods; the arm says which.
macro_rules! access_widths {
    // The device answers every width through its inherent
    //   fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64
    () => {
        fn read8(&self, a: u64) -> $crate::traits::BusRead8 {
            let r = self.access(a, 1, None);
            $crate::traits::BusRead8 { status: r.status, data: r.data as u8 }
        }
        fn read16(&self, a: u64) -> $crate::traits::BusRead16 {
            let r = self.access(a, 2, None);
            $crate::traits::BusRead16 { status: r.status, data: r.data as u16 }
        }
        fn read32(&self, a: u64) -> $crate::traits::BusRead32 {
            let r = self.access(a, 4, None);
            $crate::traits::BusRead32 { status: r.status, data: r.data as u32 }
        }
        fn read64(&self, a: u64) -> $crate::traits::BusRead64 {
            self.access(a, 8, None)
        }
        fn write8(&self, a: u64, v: u8) -> u32 {
            self.access(a, 1, Some(u64::from(v))).status
        }
        fn write16(&self, a: u64, v: u16) -> u32 {
            self.access(a, 2, Some(u64::from(v))).status
        }
        fn write32(&self, a: u64, v: u32) -> u32 {
            self.access(a, 4, Some(u64::from(v))).status
        }
        fn write64(&self, a: u64, v: u64) -> u32 {
            self.access(a, 8, Some(v)).status
        }
    };
    // A bus that passes each access on unchanged, through its inherent
    //   fn route(&self, a: u64) -> Option<(&dyn BusDevice, u64)>
    // A bus error where it returns None, with no log.
    (route) => {
        fn read8(&self, a: u64) -> $crate::traits::BusRead8 {
            self.route(a).map_or($crate::traits::BusRead8::err(), |(d, at)| d.read8(at))
        }
        fn read16(&self, a: u64) -> $crate::traits::BusRead16 {
            self.route(a).map_or($crate::traits::BusRead16::err(), |(d, at)| d.read16(at))
        }
        fn read32(&self, a: u64) -> $crate::traits::BusRead32 {
            self.route(a).map_or($crate::traits::BusRead32::err(), |(d, at)| d.read32(at))
        }
        fn read64(&self, a: u64) -> $crate::traits::BusRead64 {
            self.route(a).map_or($crate::traits::BusRead64::err(), |(d, at)| d.read64(at))
        }
        fn write8(&self, a: u64, v: u8) -> u32 {
            self.route(a).map_or($crate::traits::BUS_ERR, |(d, at)| d.write8(at, v))
        }
        fn write16(&self, a: u64, v: u16) -> u32 {
            self.route(a).map_or($crate::traits::BUS_ERR, |(d, at)| d.write16(at, v))
        }
        fn write32(&self, a: u64, v: u32) -> u32 {
            self.route(a).map_or($crate::traits::BUS_ERR, |(d, at)| d.write32(at, v))
        }
        fn write64(&self, a: u64, v: u64) -> u32 {
            self.route(a).map_or($crate::traits::BUS_ERR, |(d, at)| d.write64(at, v))
        }
    };
    // As (route), and forwards gen_ptr (jitv2), write64_masked and
    // write_narrow_from to the same target.
    (route, forward) => {
        $crate::dev::widths::access_widths!(route);
        fn write64_masked(&self, a: u64, v: u64, mask: u64) -> u32 {
            self.route(a).map_or($crate::traits::BUS_ERR, |(d, at)| d.write64_masked(at, v, mask))
        }
        fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
            self.route(a).map_or($crate::traits::BUS_ERR, |(d, at)| d.write_narrow_from(at, size, gpr))
        }
        #[cfg(feature = "jitv2")]
        fn gen_ptr(&self, a: u64) -> *const std::sync::atomic::AtomicU64 {
            self.route(a).map_or(std::ptr::null(), |(d, at)| d.gen_ptr(at))
        }
        #[cfg(feature = "jitv2")]
        fn code_page(&self, a: u64) -> Option<$crate::traits::CodePage> {
            self.route(a).and_then(|(d, at)| d.code_page(at))
        }
    };
}
pub(crate) use access_widths;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BusRead8, BUS_OK};
    use parking_lot::Mutex;

    /// Sixteen bytes that answer bytes and doublewords only, and note how
    /// a narrow store reached them.
    #[derive(Default)]
    struct Words(Mutex<[u8; 16]>, Mutex<Vec<String>>);
    impl BusDevice for Words {
        fn read8(&self, a: u64) -> BusRead8 {
            BusRead8::ok(self.0.lock()[a as usize])
        }
        fn write8(&self, a: u64, v: u8) -> u32 {
            self.0.lock()[a as usize] = v;
            BUS_OK
        }
        fn read64(&self, a: u64) -> BusRead64 {
            BusRead64::ok(u64::from_be_bytes(self.0.lock()[a as usize..a as usize + 8].try_into().unwrap()))
        }
        fn write64_masked(&self, a: u64, v: u64, mask: u64) -> u32 {
            self.1.lock().push(format!("masked {a:#x} {v:#x} {mask:#x}"));
            BUS_OK
        }
        fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
            self.1.lock().push(format!("narrow {a:#x} {size} {gpr:#x}"));
            BUS_OK
        }
        #[cfg(feature = "jitv2")]
        fn gen_ptr(&self, _: u64) -> *const std::sync::atomic::AtomicU64 {
            static GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            &GEN
        }
    }

    /// Answers 0x100..0x110 itself and records each access it sees.
    #[derive(Default)]
    struct Direct(Words, Mutex<Vec<String>>);
    impl Direct {
        fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
            self.1.lock().push(format!("{a:#x} {size} {value:?}"));
            if (0x100..0x110).contains(&a) { access(&self.0, a - 0x100, size, value) } else { BusRead64::err() }
        }
    }
    impl BusDevice for Direct {
        access_widths!();
    }

    #[test]
    fn every_width_reaches_access_with_its_size_and_value() {
        let d = Direct::default();
        assert_eq!(d.write8(0x101, 0x5a), BUS_OK);
        assert_eq!(d.read8(0x101).data, 0x5a);
        assert_eq!(d.read64(0x100).data, 0x005a << 48);
        // The target answers a width it does not have with its own bus error.
        assert_eq!(d.write16(0x102, 1), BUS_ERR);
        assert_eq!(d.read32(0x104).status, BUS_ERR);
        // Nothing answers past the words.
        assert_eq!(d.write32(0x200, 7), BUS_ERR);
        assert_eq!(d.read64(0x200).status, BUS_ERR);
        assert_eq!(
            *d.1.lock(),
            [
                "0x101 1 Some(90)", "0x101 1 None", "0x100 8 None", "0x102 2 Some(1)", "0x104 4 None",
                "0x200 4 Some(7)", "0x200 8 None",
            ]
        );
        assert_eq!(access(&d.0, 0, 3, None).status, BUS_ERR, "no such width");
    }

    /// A bus that passes accesses on to the words at 0x100..0x110.
    struct Routed(Words);
    impl Routed {
        fn route(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
            (0x100..0x110).contains(&a).then(|| (&self.0 as &dyn BusDevice, a - 0x100))
        }
    }
    impl BusDevice for Routed {
        access_widths!(route);
    }

    #[test]
    fn a_route_passes_each_width_on_and_fails_a_miss() {
        let d = Routed(Words::default());
        assert_eq!(d.write8(0x101, 0x5a), BUS_OK);
        assert_eq!(d.read8(0x101).data, 0x5a);
        assert_eq!(d.read64(0x100).data, 0x005a << 48);
        assert_eq!(d.write16(0x102, 1), BUS_ERR);
        assert_eq!(d.read32(0x104).status, BUS_ERR);
        assert_eq!(d.read16(0x200).status, BUS_ERR);
        assert_eq!(d.write64(0x200, 1), BUS_ERR);
        // Not forwarded: the trait's default splits a masked store into bytes.
        assert_eq!(d.write64_masked(0x100, 0xff00_0000_0000_0000, 0xff00_0000_0000_0000), BUS_OK);
        assert_eq!(d.0 .0.lock()[0], 0xff);
        assert!(d.0 .1.lock().is_empty());
    }

    struct Forwarded(Words);
    impl Forwarded {
        fn route(&self, a: u64) -> Option<(&dyn BusDevice, u64)> {
            (0x100..0x110).contains(&a).then(|| (&self.0 as &dyn BusDevice, a - 0x100))
        }
    }
    impl BusDevice for Forwarded {
        access_widths!(route, forward);
    }

    #[test]
    fn a_forwarding_route_passes_the_special_stores_and_the_jit_pointer_on() {
        let d = Forwarded(Words::default());
        assert_eq!(d.read8(0x103).status, BUS_OK);
        assert_eq!(d.write64_masked(0x108, 1, 0xff), BUS_OK);
        assert_eq!(d.write_narrow_from(0x109, 2, 0xabcd), BUS_OK);
        assert_eq!(*d.0 .1.lock(), ["masked 0x8 0x1 0xff", "narrow 0x9 2 0xabcd"]);
        assert_eq!(d.write64_masked(0x300, 1, 0xff), BUS_ERR);
        assert_eq!(d.write_narrow_from(0x300, 1, 0), BUS_ERR);
        #[cfg(feature = "jitv2")]
        {
            assert!(!d.gen_ptr(0x100).is_null());
            assert!(d.gen_ptr(0x300).is_null(), "no page behind a miss");
        }
    }
}
