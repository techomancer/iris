//! Windows over small MMIO apertures, for the SN machines' address maps.
//!
//! Maps replace whole 4 KiB pages as Physical does, while each endpoint still
//! rejects padding and enforces its cell permission. The sparse map places
//! several windows in one device without a host fault-handler entry for each.
//!
//! Ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use crate::physical::Physical;
use crate::traits::*;
#[cfg(feature = "jitv2")]
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// The 40-bit physical address of an SN processor.
pub(crate) const PA_MASK: u64 = (1u64 << 40) - 1;
/// XKPHYS bits 58:57 ride along with the physical address into the bus.
pub(crate) const CPU_PHYS_MASK: u64 = PA_MASK | (3u64 << 57);

struct Unclaimed;
impl BusDevice for Unclaimed {}

/// An exact-bounds chip-relative window, optionally restricted by a cell permission.
/// Physical rounds maps to pages, so the adapter also refuses their padding.
pub(crate) struct Window {
    base: u64,
    size: u64,
    device: Arc<dyn BusDevice>,
    permission: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}
impl Window {
    pub(crate) fn new(base: u64, size: u64, device: Arc<dyn BusDevice>) -> Self {
        Self { base, size, device, permission: None }
    }
    pub(crate) fn permitted(mut self, permission: Arc<dyn Fn() -> bool + Send + Sync>) -> Self {
        self.permission = Some(permission);
        self
    }
    pub(crate) fn contains(&self, a: u64, n: u64) -> bool { self.local(a, n).is_some() }
    fn local(&self, a: u64, n: u64) -> Option<u64> {
        let local = a.checked_sub(self.base)?;
        (local.checked_add(n)? <= self.size
            && self.permission.as_ref().is_none_or(|allowed| allowed()))
            .then_some(local)
    }
    fn write_masked_bytes(&self, a: u64, val: u64, mask: u64) -> u32 {
        for i in 0..8usize {
            let shift = (7 - i) * 8;
            if (mask >> shift) & 0xff == 0 {
                continue;
            }
            let Some(local) = self.local(a + i as u64, 1) else {
                return BUS_ERR;
            };
            let status = self.device.write8(local, (val >> shift) as u8);
            if status != BUS_OK {
                return status;
            }
        }
        BUS_OK
    }
}
macro_rules! window_access {
    ($r:ident,$w:ident,$t:ty,$out:ident,$n:expr) => {
        fn $r(&self, a: u64) -> $out {
            self.local(a, $n).map(|at| self.device.$r(at)).unwrap_or_else($out::err)
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.local(a, $n).map(|at| self.device.$w(at, v)).unwrap_or(BUS_ERR)
        }
    };
}
impl BusDevice for Window {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
        self.local(address, 1).map(|local| self.device.gen_ptr(local)).unwrap_or(std::ptr::null())
    }
    fn write64_masked(&self, a: u64, val: u64, mask: u64) -> u32 {
        if a & 7 != 0 {
            return BUS_ERR;
        }
        if mask == 0 {
            return BUS_OK;
        }
        if let Some(local) = self.local(a, 8) {
            self.device.write64_masked(local, val, mask)
        } else {
            self.write_masked_bytes(a, val, mask)
        }
    }
    fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
        self.local(a, size as u64)
            .map(|local| self.device.write_narrow_from(local, size, gpr))
            .unwrap_or(BUS_ERR)
    }
    window_access!(read8, write8, u8, BusRead8, 1);
    window_access!(read16, write16, u16, BusRead16, 2);
    window_access!(read32, write32, u32, BusRead32, 4);
    window_access!(read64, write64, u64, BusRead64, 8);
}
pub(crate) fn map_device(
    view: &Physical,
    base: u64,
    size: u64,
    device: Arc<dyn BusDevice>,
    name: &str,
) {
    assert!(size > 0 && base.checked_add(size).is_some_and(|end| end <= PA_MASK + 1));
    view.map(base, base + size, Arc::new(Window::new(base, size, device)), name);
}

pub(crate) struct DecodeRange {
    start: u64,
    end: u64,
    window: Arc<Window>,
}

/// Small MMIO maps have no RAM, direct pointers or host fault-handler entries.
/// Last map wins over whole pages, including padding rejected by its Window.
#[derive(Default)]
pub(crate) struct SparseWindowMap {
    pub(crate) ranges: Vec<DecodeRange>,
}
impl SparseWindowMap {
    pub(crate) fn map(&mut self, base: u64, size: u64, device: Arc<dyn BusDevice>) {
        const PAGE: u64 = 1 << crate::ppmem::devmap::SLOT_SHIFT;
        let end = base.checked_add(size).filter(|&end| size != 0 && end <= PA_MASK + 1)
            .expect("a window exceeds the physical address space");
        let start = base & !(PAGE - 1);
        let end = end.next_multiple_of(PAGE);
        let window = Arc::new(Window::new(base, size, device));
        let mut ranges = Vec::with_capacity(self.ranges.len() + 2);
        for old in self.ranges.drain(..) {
            if old.end <= start || old.start >= end {
                ranges.push(old);
            } else {
                if old.start < start {
                    ranges.push(DecodeRange { start: old.start, end: start, window: old.window.clone() });
                }
                if old.end > end {
                    ranges.push(DecodeRange { start: end, end: old.end, window: old.window });
                }
            }
        }
        ranges.push(DecodeRange { start, end, window });
        ranges.sort_unstable_by_key(|range| range.start);
        self.ranges = ranges;
    }
    fn target(&self, address: u64) -> Option<&Window> {
        let i = self.ranges.partition_point(|range| range.start <= address).checked_sub(1)?;
        let range = &self.ranges[i];
        (address < range.end).then_some(range.window.as_ref())
    }
}
macro_rules! sparse_access {
    ($read:ident,$write:ident,$ty:ty,$result:ident) => {
        fn $read(&self, address: u64) -> $result {
            self.target(address).map(|window| window.$read(address)).unwrap_or_else($result::err)
        }
        fn $write(&self, address: u64, value: $ty) -> u32 {
            self.target(address).map(|window| window.$write(address, value)).unwrap_or(BUS_ERR)
        }
    };
}
impl BusDevice for SparseWindowMap {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
        self.target(address).map(|window| window.gen_ptr(address)).unwrap_or(std::ptr::null())
    }
    fn write64_masked(&self, address: u64, value: u64, mask: u64) -> u32 {
        self.target(address).map(|window| window.write64_masked(address, value, mask))
            .unwrap_or_else(|| Unclaimed.write64_masked(address, value, mask))
    }
    fn write_narrow_from(&self, address: u64, size: usize, gpr: u64) -> u32 {
        self.target(address).map(|window| window.write_narrow_from(address, size, gpr))
            .unwrap_or(BUS_ERR)
    }
    sparse_access!(read8, write8, u8, BusRead8);
    sparse_access!(read16, write16, u16, BusRead16);
    sparse_access!(read32, write32, u32, BusRead32);
    sparse_access!(read64, write64, u64, BusRead64);
}


#[cfg(test)]
mod tests {
    use super::*;

    struct Tag(u8);
    impl BusDevice for Tag {
        fn read8(&self, a: u64) -> BusRead8 {
            BusRead8::ok(self.0 ^ a as u8)
        }
        fn write8(&self, _: u64, _: u8) -> u32 {
            BUS_OK
        }
    }

    /// A window answers exactly its bounds and the page around it is padding;
    /// a later map over the same page replaces the earlier one.
    #[test]
    fn a_window_refuses_its_padding_and_the_last_map_wins() {
        let mut map = SparseWindowMap::default();
        map.map(0x1000, 0x100, Arc::new(Tag(0x10)));
        map.map(0x8000, 0x2000, Arc::new(Tag(0x20)));
        assert_eq!(map.read8(0x1004).data, 0x10 ^ 4, "the device sees an offset from its base");
        assert_eq!(map.read8(0x1100).status, BUS_ERR, "padding in the page is refused");
        assert_eq!(map.read8(0x3000).status, BUS_ERR, "and so is a page nobody mapped");
        assert_eq!(map.read8(0x9000).data, 0x20, "a later page of a long window");
        map.map(0x1080, 0x40, Arc::new(Tag(0x30)));
        assert_eq!(map.read8(0x1084).data, 0x30 ^ 4, "the last map over the page wins");
        assert_eq!(map.read8(0x1004).status, BUS_ERR, "and takes the page's other bytes with it");
    }

    #[test]
    fn the_masks_keep_the_attribute_bits_beside_the_address() {
        assert_eq!(PA_MASK, 0xff_ffff_ffff);
        assert_eq!(CPU_PHYS_MASK, PA_MASK | 3 << 57);
    }
}
