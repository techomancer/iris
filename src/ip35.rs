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

use crate::dev::bedrock::RamExtent;
use crate::dev::widths::{self, access_widths};
use crate::physical::Physical;
use crate::traits::*;
#[cfg(feature = "jitv2")]
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

pub(crate) use crate::window_map::{map_device, SparseWindowMap, Window, CPU_PHYS_MASK, PA_MASK};

/// The reset PROM's compatibility window: the processor's reset vector and
/// the PROM's own code are at 0x1fc00000, fetched uncached.
pub(crate) const RESET_WINDOW: u64 = 0x1fc0_0000;
pub(crate) const RESET_WINDOW_SIZE: u64 = 0x40_0000;

/// Where a Bedrock node has memory: each 1 GiB software slot `extent` says
/// can hold RAM, as far as its largest bank reaches, and the reset PROM.
/// Which bank lands in which slot of a group of four depends on how the PROM
/// programs the memory controller; the Fuel's two 256 MiB banks can be in any
/// of the first four, and a Tezro's banks 4 and 5 in any of the next four.  A slot 0 larger than 508 MiB reaches the PROM's range, which the
/// node maps over it, so its RAM is listed on either side.
pub fn memory_map(extent: &RamExtent) -> crate::memory_map::MemoryMap {
    use crate::dev::bedrock::{BANK_SPAN, SLOTS};
    use crate::memory_map::{MemKind, MemRange, MemoryMap};
    const NAMES: [&str; SLOTS] =
        ["ram slot 0", "ram slot 1", "ram slot 2", "ram slot 3", "ram slot 4", "ram slot 5", "ram slot 6", "ram slot 7"];
    const PROM: (u64, u64) = (RESET_WINDOW, RESET_WINDOW + RESET_WINDOW_SIZE);
    let mut ranges = vec![MemRange::new("prom", MemKind::Rom, PROM.0, PROM.1)];
    for (slot, &name) in NAMES.iter().enumerate() {
        let (base, end) = (slot as u64 * BANK_SPAN, slot as u64 * BANK_SPAN + extent.prefix(slot));
        if base == end {
            continue;
        }
        if base < PROM.1 && end > PROM.0 {
            ranges.push(MemRange::new(name, MemKind::Ram, base, PROM.0));
            if end > PROM.1 {
                ranges.push(MemRange::new("ram slot 0 above the prom", MemKind::Ram, PROM.1, end));
            }
        } else {
            ranges.push(MemRange::new(name, MemKind::Ram, base, end));
        }
    }
    MemoryMap::new(ranges)
}

struct Unclaimed;
impl BusDevice for Unclaimed {}

/// The four uncached processor views decode MMIO separately from cached RAM.
pub struct Ip35Bus {
    pub phys: Arc<Physical>,
    views: [SparseWindowMap; 4],
}
impl Ip35Bus {
    pub fn new(phys: Arc<Physical>) -> Self {
        phys.set_default(Arc::new(Unclaimed), "IP35 unclaimed cached address");
        Self {
            phys,
            views: std::array::from_fn(|_| SparseWindowMap::default()),
        }
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
    /// Cached RAM, with `plain_memory` answering wherever `extent` says
    /// fitted RAM can be.
    pub fn cached_view(&self, plain_memory: Arc<dyn BusDevice>, extent: RamExtent) -> Arc<dyn BusDevice> {
        Arc::new(CachedPhysical {
            phys: self.phys.clone(),
            plain_memory,
            extent,
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
    extent: RamExtent,
}
impl CachedPhysical {
    fn memory_target(&self, a: u64, n: u64) -> Option<(&dyn BusDevice, u64)> {
        self.target(a, n).map(|(physical, pa)| {
            // MD owns every guest write to a possible fitted-RAM prefix, so
            // all paths update its stable logical generation counters. PROM
            // and other devices retain the cached Physical connection.
            let memory: &dyn BusDevice = if self.extent.contains(pa) {
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
impl Ip35Bus {
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let Some((v, pa)) = self.target(a, size as u64) else { return BusRead64::err() };
        widths::access(v, pa, size, value)
    }
}
impl BusDevice for Ip35Bus {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
        // A page fetched uncached counts the writes of the device that answers it.
        self.target(a, 1).map_or(std::ptr::null(), |(view, pa)| view.gen_ptr(pa))
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
    access_widths!();
}
impl CachedPhysical {
    /// A write goes where the memory decode sends it.
    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        if value.is_some() {
            let Some((v, pa)) = self.memory_target(a, size as u64) else { return BusRead64::err() };
            return widths::access(v, pa, size, value);
        }
        let Some((v, pa)) = self.target(a, size as u64) else { return BusRead64::err() };
        widths::access(v, pa, size, value)
    }
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
            // MD's logical counters belong to a view that names its RAM. A
            // node of several leaves the extent empty: its stores reach the
            // banks, which bump their own counters, so a compiled page must
            // watch those. Handing it a logical counter that only DMA and
            // uncached stores bump leaves stale code running.
            let generation = if self.extent.is_empty() { std::ptr::null() } else { self.plain_memory.gen_ptr(pa) };
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
    // A processor's SC to RAM the shadow cache cannot reach directly, such
    // as a bank smaller than a direct-window region: the swap the store makes atomic,
    // which matters once a node runs more than one processor.
    fn cas32(&self, a: u64, old: u32, new: u32) -> Option<bool> {
        self.memory_target(a, 4).and_then(|(view, pa)| view.cas32(pa, old, new))
    }
    fn cas64(&self, a: u64, old: u64, new: u64) -> Option<bool> {
        self.memory_target(a, 8).and_then(|(view, pa)| view.cas64(pa, old, new))
    }
    access_widths!();
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

pub(crate) mod inbound;
pub(crate) mod node;
pub(crate) mod presence;
pub(crate) mod routed;
pub(crate) mod serial;
#[cfg(test)]
mod tests;
