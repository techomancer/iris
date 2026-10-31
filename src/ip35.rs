//! The IP35 processor bus: Bedrock's four XKPHYS access-space views.
//!
//! Raw address bits 58:57 select Hub-special, I/O, fetch operations or uncached
//! RAM. Each view has a small fixed set of MMIO apertures, decoded by sorted
//! windows rather than a full physical-address table. Maps replace whole 4 KiB
//! pages as Physical does, while each endpoint still rejects padding and
//! enforces its cell permission.
//!
//! Cached RAM alone uses Physical, including its direct bank pointers, ppmem
//! and generation mappings. The MMIO views and register sub-bus need none of
//! those RAM facilities. Giving each its own full table would consume several
//! entries in the process-wide fault-handler registry per machine.
//!
//! Ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use crate::dev::bedrock_mem::{BANK_SPAN, FITTED_BANK};
use crate::physical::Physical;
use crate::traits::*;
#[cfg(feature = "jitv2")]
use std::sync::{atomic::AtomicU64, OnceLock};
use std::sync::Arc;

pub(crate) const CPU_PHYS_MASK: u64 = ((1u64 << 40) - 1) | (3u64 << 57);
const PA_MASK: u64 = (1u64 << 40) - 1;

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

struct DecodeRange {
    start: u64,
    end: u64,
    window: Arc<Window>,
}

/// Small MMIO maps have no RAM, direct pointers or host fault-handler entries.
/// Last map wins over whole pages, including padding rejected by its Window.
#[derive(Default)]
pub(crate) struct SparseWindowMap {
    ranges: Vec<DecodeRange>,
}
impl SparseWindowMap {
    pub(crate) fn map(&mut self, base: u64, size: u64, device: Arc<dyn BusDevice>) {
        const PAGE: u64 = 1 << crate::ppmem::devmap::SLOT_SHIFT;
        let end = base.checked_add(size).filter(|&end| size != 0 && end <= PA_MASK + 1)
            .expect("IP35 window exceeds physical address space");
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

/// The four uncached processor views decode MMIO separately from cached RAM.
pub struct Ip35Bus {
    pub phys: Arc<Physical>,
    views: [SparseWindowMap; 4],
    #[cfg(feature = "jitv2")]
    ram_generations: OnceLock<Arc<dyn BusDevice>>,
}
impl Ip35Bus {
    pub fn new(phys: Arc<Physical>) -> Self {
        phys.set_default(Arc::new(Unclaimed), "IP35 unclaimed cached address");
        Self {
            phys,
            views: std::array::from_fn(|_| SparseWindowMap::default()),
            #[cfg(feature = "jitv2")]
            ram_generations: OnceLock::new(),
        }
    }
    /// Code pages use stable logical RAM generations across bank placement and
    /// diagnostic remaps; non-RAM pages keep their own device generations.
    #[cfg(feature = "jitv2")]
    pub(crate) fn set_ram_generations(&self, memory: Arc<dyn BusDevice>) {
        assert!(self.ram_generations.set(memory).is_ok(), "IP35 RAM generations already wired");
    }
    /// Raw XKPHYS bits 58:57 select Hub-special, I/O, fetchop, or uncached RAM.
    pub fn map(&mut self, raw_uc: u8, base: u64, size: u64, device: Arc<dyn BusDevice>) {
        assert!(raw_uc < 4);
        self.views[raw_uc as usize].map(base, size, device);
    }
    pub fn set_fallback(&mut self, raw_uc: u8, end: u64, device: Arc<dyn BusDevice>) {
        self.map(raw_uc, 0, end, device);
    }
    fn target(&self, a: u64, n: u64) -> Option<(&SparseWindowMap, u64)> {
        let pa = plain_address(a, n)?;
        Some((&self.views[(a >> 57) as usize], pa))
    }
    pub fn cached_view(&self, plain_memory: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
        Arc::new(CachedPhysical {
            phys: self.phys.clone(),
            plain_memory,
        })
    }
}
fn plain_address(a: u64, n: u64) -> Option<u64> {
    if a & !CPU_PHYS_MASK != 0 {
        return None;
    }
    let pa = a & PA_MASK;
    (pa.checked_add(n)? <= PA_MASK + 1).then_some(pa)
}
struct CachedPhysical {
    phys: Arc<Physical>,
    plain_memory: Arc<dyn BusDevice>,
}
impl CachedPhysical {
    fn memory_target(&self, a: u64, n: u64) -> Option<(&dyn BusDevice, u64)> {
        self.target(a, n).map(|(physical, pa)| {
            // MD owns every guest write to a possible fitted-RAM prefix, so
            // all paths update its stable logical generation counters. PROM
            // and other devices retain the cached Physical connection.
            let memory: &dyn BusDevice = if pa / BANK_SPAN < 4 && pa % BANK_SPAN < FITTED_BANK {
                self.plain_memory.as_ref()
            } else { physical };
            (memory, pa)
        })
    }
    fn target(&self, a: u64, n: u64) -> Option<(&Physical, u64)> {
        // UC selectors belong to the uncached SysAd connection. Reject them
        // here rather than turning a malformed cached request into an alias.
        if a & !PA_MASK != 0 || a & (n - 1) != 0 {
            return None;
        }
        Some((self.phys.as_ref(), plain_address(a, n)?))
    }
}
macro_rules! processor_access {
    ($r:ident,$w:ident,$t:ty,$out:ident,$n:expr,$writes:ident) => {
        fn $r(&self, a: u64) -> $out {
            self
                .target(a, $n)
                .map(|(v, pa)| v.$r(pa))
                .unwrap_or_else($out::err)
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self
                .$writes(a, $n)
                .map(|(view, pa)| view.$w(pa, v))
                .unwrap_or(BUS_ERR)
        }
    };
}
impl BusDevice for Ip35Bus {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, address: u64) -> *const AtomicU64 {
        if address & !PA_MASK == 0 {
            // JIT page lookup passes plain physical code addresses, without
            // a cache attribute. Raw UC0 therefore cannot select the MMIO map
            // here: high cached RAM and the reset PROM use the cached plane.
            if let Some(memory) = self.ram_generations.get() {
                let generation = memory.gen_ptr(address);
                if !generation.is_null() { return generation; }
            }
            return self.phys.gen_ptr(address);
        }
        self.target(address, 1).map(|(view, pa)| view.gen_ptr(pa)).unwrap_or(std::ptr::null())
    }
    fn write64_masked(&self, a: u64, val: u64, mask: u64) -> u32 {
        if a & 7 != 0 {
            return BUS_ERR;
        }
        self.target(a, 8)
            .map(|(view, pa)| view.write64_masked(pa, val, mask))
            .unwrap_or(BUS_ERR)
    }
    fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
        self.target(a, size as u64)
            .map(|(view, pa)| view.write_narrow_from(pa, size, gpr))
            .unwrap_or(BUS_ERR)
    }
    processor_access!(read8, write8, u8, BusRead8, 1, target);
    processor_access!(read16, write16, u16, BusRead16, 2, target);
    processor_access!(read32, write32, u32, BusRead32, 4, target);
    processor_access!(read64, write64, u64, BusRead64, 8, target);
}
impl BusDevice for CachedPhysical {
    fn write64_masked(&self, address: u64, value: u64, mask: u64) -> u32 {
        self.memory_target(address, 8).map(|(view, pa)| view.write64_masked(pa, value, mask))
            .unwrap_or(BUS_ERR)
    }
    fn mem_ptr(&self, a: u64) -> Option<*const u64> {
        self.target(a, 8).and_then(|(view, pa)| view.mem_ptr(pa))
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
        self.target(a, 1).map(|(view, pa)| {
            let generation = self.plain_memory.gen_ptr(pa);
            if generation.is_null() { view.gen_ptr(pa) } else { generation }
        }).unwrap_or(std::ptr::null())
    }
    fn read_block(&self, a: u64, out: &mut [u64]) -> u32 {
        self.target(a, 8)
            .map(|(view, pa)| view.read_block(pa, out)).unwrap_or(BUS_ERR)
    }
    fn write_block(&self, a: u64, data: &[u64]) -> u32 {
        self.memory_target(a, 8)
            .map(|(view, pa)| view.write_block(pa, data)).unwrap_or(BUS_ERR)
    }
    processor_access!(read8, write8, u8, BusRead8, 1, memory_target);
    processor_access!(read16, write16, u16, BusRead16, 2, memory_target);
    processor_access!(read32, write32, u32, BusRead32, 4, memory_target);
    processor_access!(read64, write64, u64, BusRead64, 8, memory_target);
    fn dma_read64(&self, a: u64) -> BusRead64 {
        plain_address(a, 8)
            .map(|pa| self.plain_memory.dma_read64(pa))
            .unwrap_or_else(BusRead64::err)
    }
    fn dma_write64(&self, a: u64, v: u64) -> u32 {
        plain_address(a, 8)
            .map(|pa| self.plain_memory.dma_write64(pa, v))
            .unwrap_or(BUS_ERR)
    }
}

#[cfg(test)]
mod tests;
