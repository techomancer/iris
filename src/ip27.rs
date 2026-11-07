//! The SN0 processor bus: a node's memory, its uncached views and the Hub
//! special windows.
//!
//! The processor drives two attribute bits beside an uncached address (XKPHYS
//! bits 58:57) that pick one of four spaces: 0 Hub special, 1 I/O, 2 memory
//! special and 3 uncached memory. [`Ip27Bus`] keeps one device per space and
//! indexes them with those bits, so an access takes no decision beyond the
//! index. A plain address, as the compatibility segments produce, carries no
//! attribute and lands in view 0, the Hub-special space, as it does on the
//! Fuel. That is an approximation: on the machine a compatibility uncached
//! reference is attribute 3, memory. The two agree wherever this model
//! answers, because the Hub-special space repeats node memory in its first
//! 256 MiB and the boot flash in the next, which is where the reset vector
//! and the PROM's uncached references land.
//!
//! Cached references use the node's [`Physical`] directly: RAM at its
//! node-relative offsets, with the ppmem window the cache and the JIT read.
//!
//! # What is approximated here
//!
//! - Cached accesses ignore the node number in address bits 39:32 and use
//!   node-relative offsets, so they are right for a node numbered zero, which
//!   is the one-node Origin 200.
//! - The low 128 KiB of the Hub-special memory view is not private to each
//!   processor of the pair.
//! - A bank no larger than 128 MiB is one run at its 512 MiB stride; a larger
//!   one is four runs at 128 MiB intervals.
//! - Memory that is not fitted reads zero and drops writes, logged under
//!   `phys hole on`.
//! - The ECC and directory back doors keep only the low two bytes of each
//!   eight-byte entry. An unfitted bank's back door answers a fixed pattern
//!   to reads and drops writes.
//!
//! Ported from hw/mips/sgi/sn0.c in jamesbraid/qemu-sgi-origami.

use crate::dev::node_flash::NodeFlash;
use crate::ip35::SparseWindowMap;
use crate::physical::{Physical, RamBank};
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::ppmem::MappedMemory;
use crate::traits::*;
use parking_lot::Mutex;
use std::sync::Arc;

pub(crate) const PA_MASK: u64 = (1u64 << 40) - 1;
/// XKPHYS bits 58:57 ride along with the physical address into the bus.
pub(crate) const CPU_PHYS_MASK: u64 = PA_MASK | (3u64 << 57);

/// The boot flash's reset view, and the node memory a bank stride spans.
pub const RESET_VIEW: u64 = 0x1fc0_0000;
pub const BANK_STRIDE: u64 = 512 << 20;
pub const SLOT_STRIDE: u64 = 128 << 20;
/// Banks the Origin 200's board wires.
pub const BANKS: usize = 4;

const MIB: u64 = 1 << 20;

/// Where bank `bank` of `mb` MiB lies: `(node offset, length)` runs, in the
/// order they take the bank's storage.
pub fn bank_runs(bank: usize, mb: u32) -> Vec<(u64, u64)> {
    let size = u64::from(mb) * MIB;
    if size == 0 {
        return Vec::new();
    }
    let base = bank as u64 * BANK_STRIDE;
    if size <= SLOT_STRIDE {
        vec![(base, size)]
    } else {
        let run = size / 4;
        (0..4).map(|i| (base + i * SLOT_STRIDE, run)).collect()
    }
}

/// The node's memory: the banks on a decode table, placed at their stride.
pub struct NodeMemory {
    phys: Arc<Physical>,
    banks_mb: [u32; BANKS],
}

impl NodeMemory {
    pub fn new(banks_mb: [u32; BANKS]) -> Result<Self, String> {
        let mut banks = Vec::new();
        for (i, &mb) in banks_mb.iter().enumerate() {
            if mb != 0 {
                banks.push(RamBank::try_new(mb as usize).map_err(|e| format!("bank {i}: {e}"))?);
            }
        }
        let phys = Physical::new(banks);
        phys.set_default(Arc::new(Absent), "unfitted memory");
        let mut index = 0;
        for (bank, &mb) in banks_mb.iter().enumerate() {
            if mb == 0 {
                continue;
            }
            let mut src = 0u64;
            for (at, len) in bank_runs(bank, mb) {
                phys.fill_slots(at >> SLOT_SHIFT, len >> SLOT_SHIFT, phys.bank_view(index, at.wrapping_sub(src)));
                if let Some(sp) = phys.ppmem_space() {
                    if at + len <= sp.window_size() {
                        sp.map_range(index, src, at, len).map_err(|e| format!("bank {bank}: window map: {e}"))?;
                    }
                }
                src += len;
            }
            index += 1;
        }
        Ok(Self { phys, banks_mb })
    }

    pub fn physical(&self) -> &Arc<Physical> {
        &self.phys
    }

    pub fn banks_mb(&self) -> [u32; BANKS] {
        self.banks_mb
    }

    /// Place the boot flash's reset view in node memory.
    pub fn map_reset_view(&self, flash: &Arc<NodeFlash>) {
        crate::ip35::map_device(&self.phys, RESET_VIEW, crate::dev::node_flash::SIZE as u64, flash.chip(), "boot flash");
    }

    /// The ranges the CPU model reports as memory.
    pub fn memory_map(&self) -> crate::memory_map::MemoryMap {
        use crate::memory_map::{MemKind, MemRange, MemoryMap};
        const NAMES: [&str; 16] = [
            "bank 0 run 0", "bank 0 run 1", "bank 0 run 2", "bank 0 run 3",
            "bank 1 run 0", "bank 1 run 1", "bank 1 run 2", "bank 1 run 3",
            "bank 2 run 0", "bank 2 run 1", "bank 2 run 2", "bank 2 run 3",
            "bank 3 run 0", "bank 3 run 1", "bank 3 run 2", "bank 3 run 3",
        ];
        let mut ranges = Vec::new();
        for (bank, &mb) in self.banks_mb.iter().enumerate() {
            for (run, (at, len)) in bank_runs(bank, mb).into_iter().enumerate() {
                ranges.push(MemRange::new(NAMES[bank * 4 + run], MemKind::Ram, at, at + len));
            }
        }
        ranges.push(MemRange::new(
            "boot flash",
            MemKind::Rom,
            RESET_VIEW,
            RESET_VIEW + crate::dev::node_flash::SIZE as u64,
        ));
        MemoryMap::new(ranges)
    }

    /// The back door onto the ECC bytes or the directory entries.
    pub fn back_door(&self) -> Arc<Backdoor> {
        Backdoor::new(self.banks_mb)
    }
}

/// Memory that is not fitted: reads zero, drops writes. `phys hole on` logs it.
struct Absent;
impl BusDevice for Absent {
    fn read8(&self, _: u64) -> BusRead8 { BusRead8::ok(0) }
    fn read16(&self, _: u64) -> BusRead16 { BusRead16::ok(0) }
    fn read32(&self, _: u64) -> BusRead32 { BusRead32::ok(0) }
    fn read64(&self, _: u64) -> BusRead64 { BusRead64::ok(0) }
    fn write8(&self, _: u64, _: u8) -> u32 { BUS_OK }
    fn write16(&self, _: u64, _: u16) -> u32 { BUS_OK }
    fn write32(&self, _: u64, _: u32) -> u32 { BUS_OK }
    fn write64(&self, _: u64, _: u64) -> u32 { BUS_OK }
}

const BACKDOOR_ENTRY: usize = 8;
const BACKDOOR_LIVE: usize = 2;
const BACKDOOR_BANK_STRIDE: u64 = 128 * MIB;
/// What an unfitted bank's back door reads: firmware's presence probe calls a
/// bank absent when neither a pattern nor zero survives a write.
const BACKDOOR_ABSENT: u64 = 0xa5a5_a5a5_a5a5_a5a5;

/// One back door window (1 GiB, eight banks of 128 MiB): per fitted bank, a
/// quarter of the bank's size of storage folded across the bank's span.
pub struct Backdoor {
    spans: [u64; 8],
    offsets: [u64; 8],
    data: Mutex<Vec<u8>>,
}

impl Backdoor {
    fn new(banks_mb: [u32; BANKS]) -> Arc<Self> {
        let mut spans = [0u64; 8];
        let mut offsets = [0u64; 8];
        let mut total = 0;
        for (i, &mb) in banks_mb.iter().enumerate() {
            spans[i] = u64::from(mb) * MIB / 4;
            offsets[i] = total;
            total += spans[i];
        }
        Arc::new(Self { spans, offsets, data: Mutex::new(vec![0; total as usize]) })
    }

    /// The storage index of byte `a`, or None for a bank that is not fitted.
    fn index(&self, a: u64) -> Option<usize> {
        let bank = (a / BACKDOOR_BANK_STRIDE) as usize;
        let span = *self.spans.get(bank)?;
        (span != 0).then(|| (self.offsets[bank] + (a % BACKDOOR_BANK_STRIDE) % span) as usize)
    }

    fn read(&self, a: u64, n: usize) -> u64 {
        let Some(at) = self.index(a) else {
            return BACKDOOR_ABSENT & (u64::MAX >> (64 - 8 * n as u32));
        };
        let data = self.data.lock();
        (0..n).fold(0u64, |v, i| (v << 8) | u64::from(data[at + i]))
    }

    fn write(&self, a: u64, n: usize, v: u64) {
        let Some(at) = self.index(a) else { return };
        let mut data = self.data.lock();
        for i in 0..n {
            // Only the low two bytes of an eight-byte entry are stored.
            if (a as usize + i) % BACKDOOR_ENTRY >= BACKDOOR_ENTRY - BACKDOOR_LIVE {
                data[at + i] = (v >> (8 * (n - 1 - i))) as u8;
            }
        }
    }
}

macro_rules! backdoor_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            $out::ok(self.read(a, std::mem::size_of::<$t>()) as $t)
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.write(a, std::mem::size_of::<$t>(), u64::from(v));
            BUS_OK
        }
    };
}
impl BusDevice for Backdoor {
    backdoor_access!(read8, write8, u8, BusRead8);
    backdoor_access!(read16, write16, u16, BusRead16);
    backdoor_access!(read32, write32, u32, BusRead32);
    backdoor_access!(read64, write64, u64, BusRead64);
}

/// The Hub-special space of a node: its own memory and boot flash and the
/// named node's, and the two back doors.
pub fn hub_special(memory: Arc<dyn BusDevice>, flash: &Arc<NodeFlash>, ecc: Arc<Backdoor>, directory: Arc<Backdoor>) -> Arc<dyn BusDevice> {
    const WINDOW: u64 = 256 * MIB;
    const BACK_DOOR: u64 = 1 << 30;
    let mut map = SparseWindowMap::default();
    for base in [0, 0x2000_0000] {
        map.map(base, WINDOW, memory.clone());
    }
    for base in [0x1000_0000, 0x3000_0000] {
        map.map(base, WINDOW, flash.window());
    }
    map.map(0x8000_0000, BACK_DOOR, ecc);
    map.map(0xc000_0000, BACK_DOOR, directory);
    Arc::new(map)
}

/// The processor's uncached bus: one device per attribute, indexed by
/// address bits 58:57. See the module notes for plain addresses.
pub struct Ip27Bus {
    views: [Arc<dyn BusDevice>; 4],
}

impl Ip27Bus {
    /// `special`, `io`, `fetchop` and `memory` are the devices attributes 0
    /// to 3 select.
    pub fn new(special: Arc<dyn BusDevice>, io: Arc<dyn BusDevice>, fetchop: Arc<dyn BusDevice>, memory: Arc<dyn BusDevice>) -> Self {
        Self { views: [special, io, fetchop, memory] }
    }

    #[inline(always)]
    fn target(&self, a: u64) -> (&dyn BusDevice, u64) {
        (&*self.views[((a >> 57) & 3) as usize], a & PA_MASK)
    }
}

macro_rules! bus_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            let (v, pa) = self.target(a);
            v.$r(pa)
        }
        fn $w(&self, a: u64, d: $t) -> u32 {
            let (v, pa) = self.target(a);
            v.$w(pa, d)
        }
    };
}
impl BusDevice for Ip27Bus {
    fn write64_masked(&self, a: u64, val: u64, mask: u64) -> u32 {
        let (v, pa) = self.target(a);
        v.write64_masked(pa, val, mask)
    }
    fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
        let (v, pa) = self.target(a);
        v.write_narrow_from(pa, size, gpr)
    }
    bus_access!(read8, write8, u8, BusRead8);
    bus_access!(read16, write16, u16, BusRead16);
    bus_access!(read32, write32, u32, BusRead32);
    bus_access!(read64, write64, u64, BusRead64);
}

#[cfg(test)]
mod tests;
