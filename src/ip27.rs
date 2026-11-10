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
//! - Cached accesses to node zero's window use node-relative offsets while
//!   the node holds zero, which is the one-node Origin 200 and the first node
//!   of a deskside. A board with several nodes routes the windows of the
//!   other node numbers through the Hub ([`NodeMemory::route_nodes`]), and
//!   sends node zero's, beyond the cached alias, to the Hub that holds zero
//!   once the node holds another number ([`NodeZero`]).
//! - The low 128 KiB of the Hub-special memory view is not private to each
//!   processor of the pair.
//! - A bank no larger than 128 MiB is one run at its 512 MiB stride; a larger
//!   one is four runs at 128 MiB intervals.
//! - Memory that is not fitted fails every access with a bus error, logged
//!   under the `hub` log category: the IRIX kernel's handler for an uncached
//!   bus error considers memory that does not exist, and SGI's event catalogue
//!   lists an access to non-existent memory. No capture of a physical machine
//!   shows it, so it is a software expectation. The PROM's memory sizing reads
//!   each bank through the directory back door, which answers a fixed pattern
//!   for an absent bank, so it does not reach unfitted memory: boots of one,
//!   two and 256 MiB banks log no access.
//! - The ECC and directory back doors keep only the low two bytes of each
//!   eight-byte entry. An unfitted bank's back door answers a fixed pattern
//!   to reads and drops writes.
//!
//! Ported from hw/mips/sgi/sn0.c in jamesbraid/qemu-sgi-origami.

use crate::dev::hub::Routing;
use crate::dev::node_flash::NodeFlash;
use crate::devlog::LogModule;
use crate::ip35::SparseWindowMap;
use crate::physical::{Physical, RamBank};
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::ppmem::MappedMemory;
use crate::traits::*;
use parking_lot::Mutex;
use std::sync::{Arc, Weak};

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
/// Node numbers, whose cached windows [`NodeMemory::route_nodes`] answers.
const NODE_WINDOWS: u64 = 256;

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

/// One fitted run of node memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    /// The bank whose stride the run lies in.
    pub bank: usize,
    /// The run's RAM bank among the fitted ones, in the order they were built.
    pub index: usize,
    /// Where the run lies in the node's offsets.
    pub at: u64,
    pub len: u64,
    /// Where in its RAM bank the run starts.
    pub src: u64,
}

/// Every fitted run of `banks_mb`, in the order they take their storage. The
/// one place that skips empty banks and numbers the RAM they are built from;
/// the decode table, the node-zero reroute, the Hub's direct lookup and the
/// memory map all read it.
fn layout(banks_mb: [u32; BANKS]) -> Arc<[Run]> {
    let mut runs = Vec::new();
    let mut index = 0;
    for (bank, &mb) in banks_mb.iter().enumerate() {
        if mb == 0 {
            continue;
        }
        let mut src = 0;
        for (at, len) in bank_runs(bank, mb) {
            runs.push(Run { bank, index, at, len, src });
            src += len;
        }
        index += 1;
    }
    runs.into()
}

/// The node's memory: the banks on a decode table, placed at their stride.
pub struct NodeMemory {
    phys: Arc<Physical>,
    banks_mb: [u32; BANKS],
    layout: Arc<[Run]>,
    /// The view [`Self::route_nodes`] sent the other node numbers to, held
    /// weakly: it reaches the Hub, which holds this memory's users.
    claim: Arc<Mutex<Option<Weak<dyn BusDevice>>>>,
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
        let layout = layout(banks_mb);
        for &Run { bank, index, at, len, src } in layout.iter() {
            phys.fill_slots(at >> SLOT_SHIFT, len >> SLOT_SHIFT, phys.bank_view(index, at.wrapping_sub(src)));
            if let Some(sp) = phys.ppmem_space() {
                if at + len <= sp.window_size() {
                    sp.map_range(index, src, at, len).map_err(|e| format!("bank {bank}: window map: {e}"))?;
                }
            }
        }
        Ok(Self { phys, banks_mb, layout, claim: Arc::default() })
    }

    pub fn physical(&self) -> &Arc<Physical> {
        &self.phys
    }

    pub fn banks_mb(&self) -> [u32; BANKS] {
        self.banks_mb
    }

    /// Answer cached references to node numbers 1 to 255 through `view`, the
    /// Hub's node-qualified memory, so a reference to memory another Hub holds
    /// reaches it across the cable and one that no Hub holds fails, logged by
    /// the Hub. The windows are 4 GiB apart at address bit 32, above
    /// everything the node's own banks occupy. Node zero's window is the
    /// node's own memory and is not routed.
    pub fn route_nodes(&self, view: Arc<dyn BusDevice>) {
        *self.claim.lock() = Some(Arc::downgrade(&view));
        for node in 1..NODE_WINDOWS {
            self.phys.map(node << 32, (node + 1) << 32, view.clone(), "node-qualified memory");
        }
    }

    /// The node's answer to a node-zero cached reference once the node holds
    /// another number, for [`NodeZero::update`] to apply. It sends the
    /// reference to the view [`Self::route_nodes`] was given, so a board with
    /// one node, which routes nothing, never sends it anywhere.
    pub fn node_zero(&self, routing: Arc<Routing>) -> Arc<NodeZero> {
        Arc::new(NodeZero {
            routing,
            phys: Arc::downgrade(&self.phys),
            layout: self.layout.clone(),
            claim: self.claim.clone(),
            state: Mutex::new(NodeZeroState { number: 0, calias: 0, away: false }),
        })
    }

    /// The node's memory at its node-relative offsets whatever number the
    /// node holds, for the Hub's spaces: the Hub-special space, and every
    /// reference another Hub or an engine makes by the node's number.
    /// [`NodeZero`] reroutes only the processor's own cached references.
    pub fn home(&self) -> Arc<dyn BusDevice> {
        Arc::new(Home::new(self.phys.clone(), &self.layout))
    }

    /// Place the boot flash's reset view in node memory.
    pub fn map_reset_view(&self, flash: &Arc<NodeFlash>) {
        crate::ip35::map_device(&self.phys, RESET_VIEW, crate::dev::node_flash::SIZE as u64, flash.chip(), "boot flash");
    }

    /// The ranges the CPU model reports as memory.
    pub fn memory_map(&self) -> crate::memory_map::MemoryMap {
        use crate::memory_map::{MemKind, MemRange, MemoryMap};
        const NAMES: [&str; BANKS] = ["bank 0", "bank 1", "bank 2", "bank 3"];
        let mut ranges: Vec<_> = self.layout.iter().map(|r| MemRange::new(NAMES[r.bank], MemKind::Ram, r.at, r.at + r.len)).collect();
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

/// [`NodeMemory::home`]: an offset inside a fitted run reaches its bank
/// directly, past whatever [`NodeZero`] put in the decode table there; any
/// other offset is the decode table's.
struct Home {
    phys: Arc<Physical>,
    /// Each bank's fitted runs, laid out by stride: a node offset names a
    /// bank by its stride, and a bank has at most four runs.
    runs: [[Option<Run>; 4]; BANKS],
}

impl Home {
    fn new(phys: Arc<Physical>, layout: &[Run]) -> Self {
        let mut runs = [[None; 4]; BANKS];
        for run in layout {
            let slot = runs[run.bank].iter().position(Option::is_none).expect("a bank has at most four runs");
            runs[run.bank][slot] = Some(*run);
        }
        Self { phys, runs }
    }

    /// The bank index and the offset within it that node offset `a` names,
    /// if a fitted run holds it. No allocation: this is on every access.
    #[inline]
    fn bank(&self, a: u64) -> Option<(usize, u64)> {
        let runs = self.runs.get((a / BANK_STRIDE) as usize)?;
        runs.iter().flatten().find(|r| (r.at..r.at + r.len).contains(&a)).map(|r| (r.index, r.src + a - r.at))
    }
}

macro_rules! home_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            match self.bank(a) {
                Some((i, at)) => self.phys.banks()[i].$r(at),
                None => self.phys.$r(a),
            }
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            match self.bank(a) {
                Some((i, at)) => self.phys.banks()[i].$w(at, v),
                None => self.phys.$w(a, v),
            }
        }
    };
}
impl BusDevice for Home {
    fn cas32(&self, a: u64, old: u32, new: u32) -> Option<bool> {
        match self.bank(a) {
            Some((i, at)) => self.phys.banks()[i].cas32(at, old, new),
            None => self.phys.cas32(a, old, new),
        }
    }
    fn cas64(&self, a: u64, old: u64, new: u64) -> Option<bool> {
        match self.bank(a) {
            Some((i, at)) => self.phys.banks()[i].cas64(at, old, new),
            None => self.phys.cas64(a, old, new),
        }
    }
    home_access!(read8, write8, u8, BusRead8);
    home_access!(read16, write16, u16, BusRead16);
    home_access!(read32, write32, u32, BusRead32);
    home_access!(read64, write64, u64, BusRead64);
}

/// Memory that is not fitted: every access is a bus error, logged under
/// `hub`. Only accesses outside the fitted runs reach it, so fitted memory
/// pays nothing for it.
struct Absent;
macro_rules! absent_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            crate::dlog!(LogModule::Hub, "Hub: read of unfitted memory {a:#x}");
            $out::err()
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            crate::dlog!(LogModule::Hub, "Hub: write of {v:#x} to unfitted memory {a:#x}");
            BUS_ERR
        }
    };
}
impl BusDevice for Absent {
    absent_access!(read8, write8, u8, BusRead8);
    absent_access!(read16, write16, u16, BusRead16);
    absent_access!(read32, write32, u32, BusRead32);
    absent_access!(read64, write64, u64, BusRead64);
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

/// Where a node that holds a nonzero number sends a cached reference that
/// names node zero. Within the cached alias the reference stays on the node.
/// Beyond it the node-qualified memory answers by number, and the reference
/// reaches whichever Hub holds zero, across the cable. The node's own memory
/// is then reached by its own number.
///
/// Node zero's window is the node's own memory, and its RAM is mapped into
/// the ppmem window the cache and the JIT read directly. Sending a reference
/// elsewhere takes the mapping away, so every cached reference of the node
/// takes the bus path while the number is nonzero. That costs the second node
/// speed and nothing else, since its software addresses its memory by its
/// number.
pub struct NodeZero {
    /// Where the Hub publishes the number and the alias.
    routing: Arc<Routing>,
    /// Weak: the Hub holds this, and the node owns the memory.
    phys: Weak<Physical>,
    layout: Arc<[Run]>,
    claim: Arc<Mutex<Option<Weak<dyn BusDevice>>>>,
    state: Mutex<NodeZeroState>,
}

struct NodeZeroState {
    number: u16,
    /// The cached alias size in bytes.
    calias: u64,
    /// Whether the RAM beyond the alias is sent through the claim.
    away: bool,
}

impl NodeZero {
    /// Re-read the node's number and cached alias from [`Routing`] and route
    /// to match. The read is under this lock, so of two stores that race the
    /// later refresh to take it applies the later value. Called after a store
    /// changed either, never on an access.
    pub fn refresh(&self) {
        let mut st = self.state.lock();
        st.number = self.routing.number();
        st.calias = self.routing.calias();
        self.apply(&mut st);
    }

    /// The node holds `number` and its cached alias spans `calias` bytes.
    #[cfg(test)]
    pub fn update(&self, number: u16, calias: u64) {
        self.routing.set(number, calias);
        self.refresh();
    }

    fn apply(&self, st: &mut NodeZeroState) {
        let Some(phys) = self.phys.upgrade() else { return };
        let claim = self.claim.lock().as_ref().and_then(Weak::upgrade);
        let away = st.number != 0 && claim.is_some();
        if !away && !st.away {
            return;
        }
        let ppmem = phys.ppmem_space();
        if let Some(sp) = ppmem {
            sp.clear_mappings();
        }
        for &Run { index, at, len, src, .. } in self.layout.iter() {
            let end = at + len;
            // The part of the run past the alias goes through the claim; the
            // rest, and every run when the number is zero, is the bank.
            let split = if away { st.calias.clamp(at, end) } else { end };
            if split > at {
                let first = at >> SLOT_SHIFT;
                phys.fill_slots(first, (split - at) >> SLOT_SHIFT, phys.bank_view(index, at.wrapping_sub(src)));
            }
            if split < end {
                if let Some(claim) = &claim {
                    phys.map(split, end, claim.clone(), "node zero beyond the cached alias");
                }
            }
            if !away {
                if let Some(sp) = ppmem {
                    if end <= sp.window_size() {
                        let _ = sp.map_range(index, src, at, len);
                    }
                }
            }
        }
        st.away = away;
    }
}

#[cfg(test)]
mod tests;
