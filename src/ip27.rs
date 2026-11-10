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
//!   once the node holds another number ([`NodeZero`]). The ppmem window
//!   shows the same memory at the same addresses ([`Windows`]). Each node
//!   reserves host address space for it, 4 GiB for one node and
//!   `next_power_of_two(nodes) x 4 GiB` for a board with a cable, as an
//!   inaccessible mapping that commits nothing (`mmap` with `PROT_NONE` and
//!   `MAP_NORESERVE` on Unix, a `VirtualAlloc2` placeholder on Windows). That
//!   is by reading the host code: only Linux was run, macOS and Windows were
//!   not.
//! - The low 128 KiB of the Hub-special memory view is not private to each
//!   processor of the pair.
//! - A 512 MiB bank 0 covers the boot flash's reset address. Cached references
//!   reach the bank and the reset fetch, which is uncached, reaches the flash
//!   through the Hub-special view; the Hub-special view, not the cached
//!   decode, is where the PROM sits. That the hardware does the same is
//!   *inferred*: the PROM's memory test walks that address through the cached
//!   space as RAM, and a reset fetch of the first instruction cannot come
//!   from anywhere but the flash.
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
use crate::window_map::{SparseWindowMap, CPU_PHYS_MASK, PA_MASK};
use crate::physical::{Physical, RamBank};
use crate::ppmem::devmap::SLOT_SHIFT;
use crate::ppmem::MappedMemory;
use crate::traits::*;
use parking_lot::Mutex;
use std::sync::{Arc, Weak};

/// The boot flash's reset view, and the node memory a bank stride spans.
pub const RESET_VIEW: u64 = 0x1fc0_0000;
pub const BANK_STRIDE: u64 = 512 << 20;
pub const SLOT_STRIDE: u64 = 128 << 20;
/// Banks an SN0 node board wires.
pub const BANKS: usize = 4;

const MIB: u64 = 1 << 20;
/// Node numbers, whose cached windows [`NodeMemory::route_nodes`] answers.
const NODE_WINDOWS: u64 = 256;
/// The memory a node number spans in a physical address.
const NODE_SPAN: u64 = 1 << 32;

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
        Self::with_window(banks_mb, crate::ppmem::WINDOW_SIZE, crate::ppmem::BITMAP_SHIFT)
    }

    /// [`Self::new`] with a ppmem window of `window` bytes whose region bitmap
    /// counts `1 << shift` byte regions: a board of several nodes reaches each
    /// node's memory at its number ([`window`]).
    pub fn with_window(banks_mb: [u32; BANKS], window: u64, shift: u32) -> Result<Self, String> {
        let mut banks = Vec::new();
        for (i, &mb) in banks_mb.iter().enumerate() {
            if mb != 0 {
                banks.push(RamBank::try_new(mb as usize).map_err(|e| format!("bank {i}: {e}"))?);
            }
        }
        let phys = Physical::with_window_regions(banks, window, shift);
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
    /// one node, which routes nothing, never sends it anywhere. The node's
    /// ppmem window joins `windows`, which lays it as the numbers route.
    pub fn node_zero(&self, routing: Arc<Routing>, windows: &Arc<Windows>) -> Arc<NodeZero> {
        windows.add(self, routing.clone());
        Arc::new(NodeZero {
            routing,
            windows: windows.clone(),
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

    /// Whether a fitted run holds the boot flash's reset view: a 512 MiB bank
    /// 0 spans the whole stride, flash address included.
    fn ram_holds_reset_view(&self) -> bool {
        let end = RESET_VIEW + crate::dev::node_flash::SIZE as u64;
        self.layout.iter().any(|r| r.at < end && r.at + r.len > RESET_VIEW)
    }

    /// Place the boot flash's reset view in node memory. Where a 512 MiB bank
    /// 0 holds that address the bank keeps it for cached references: the
    /// PROM's memory test walks it as RAM, and the reset fetch, which is
    /// uncached, reaches the flash through the Hub-special view
    /// ([`hub_special`]) instead.
    pub fn map_reset_view(&self, flash: &Arc<NodeFlash>) {
        if self.ram_holds_reset_view() {
            return;
        }
        crate::window_map::map_device(&self.phys, RESET_VIEW, crate::dev::node_flash::SIZE as u64, flash.chip(), "boot flash");
    }

    /// The ranges the CPU model reports as memory.
    pub fn memory_map(&self) -> crate::memory_map::MemoryMap {
        use crate::memory_map::{MemKind, MemRange, MemoryMap};
        const NAMES: [&str; BANKS] = ["bank 0", "bank 1", "bank 2", "bank 3"];
        let mut ranges: Vec<_> = self.layout.iter().map(|r| MemRange::new(NAMES[r.bank], MemKind::Ram, r.at, r.at + r.len)).collect();
        if !self.ram_holds_reset_view() {
            ranges.push(MemRange::new(
                "boot flash",
                MemKind::Rom,
                RESET_VIEW,
                RESET_VIEW + crate::dev::node_flash::SIZE as u64,
            ));
        }
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
/// This routes the decode table. The ppmem window the cache and the JIT read
/// follows the same numbers through [`Windows`].
pub struct NodeZero {
    /// Where the Hub publishes the number and the alias.
    routing: Arc<Routing>,
    windows: Arc<Windows>,
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
        drop(st);
        self.windows.refresh();
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
        }
        st.away = away;
    }
}

/// The ppmem window for a board of `nodes` nodes, and its region bitmap's
/// region shift. One node keeps the 4 GiB window and its 64 MiB regions. A
/// board with a cable spans node numbers up to the next power of two above the
/// node count, node `n`'s memory starting at `n << 32`, so two nodes reserve
/// 8 GiB of host address space and four reserve 16 GiB. The bitmap's 64
/// regions of 128 MiB cover numbers 0 and 1, which the stock PROM gives the
/// two nodes; memory the bitmap does not hold whole, a 64 or 256 MiB bank's or
/// a higher number's, is reached through the window's fine table, at 64 MiB.
///
/// The window only speeds cached references up. A node that a guest gives a
/// number past the window still works: its memory stays on the bus path, which
/// answers every number up to 255.
pub fn window(nodes: usize) -> (u64, u32) {
    if nodes > 1 {
        (nodes.next_power_of_two().min(NODE_WINDOWS as usize) as u64 * NODE_SPAN, 27)
    } else {
        (crate::ppmem::WINDOW_SIZE, crate::ppmem::BITMAP_SHIFT)
    }
}

/// What every node's ppmem window shows, which is what its processor's cache
/// and the JIT read without the bus. It follows the decode table exactly, as
/// the Hubs route it: node zero's window is the node's own memory while it
/// holds zero, its own memory below the cached alias and the memory of the
/// node holding zero above it once it holds another number, and on a board
/// with a cable each node's memory sits at its number. So a processor reaches
/// both its own memory and the other node's at the cost of a load, whichever
/// node it is on.
///
/// A number or alias change lays only the difference, and one host mapping
/// call moves a page that stays mapped from one node's memory to the other's.
/// A range that goes leaves the region bitmap before its pages do. The other
/// processor can be running meanwhile: a renumbering never clears the window
/// it is executing from.
///
/// A store to any Hub's number or alias, from whichever processor made it,
/// publishes the numbers ([`Routing`]) and lays every node's window at once
/// from that thread, as the hardware takes the new number at once. The other
/// node's processor can be running or issuing CACHE instructions meanwhile.
/// [`crate::ppmem::PpMemSpace`] serializes the layout against those with its
/// own lock, taken by a remap or a held-line change and never by an access
/// that hits. Windows' lock orders two renumberings; the space's lock is
/// taken inside it, never the other way round.
#[derive(Default)]
pub struct Windows {
    nodes: Mutex<Vec<Shown>>,
}

struct Shown {
    phys: Weak<Physical>,
    layout: Arc<[Run]>,
    claim: Arc<Mutex<Option<Weak<dyn BusDevice>>>>,
    routing: Arc<Routing>,
    /// What the window holds now.
    laid: Vec<Piece>,
}

/// A stretch of one node's bank laid in a window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Piece {
    /// The node whose memory it is, and the RAM bank among its fitted ones.
    owner: usize,
    index: usize,
    /// Where in the bank, and where in the window.
    src: u64,
    at: u64,
    len: u64,
}

impl Windows {
    fn add(&self, mem: &NodeMemory, routing: Arc<Routing>) {
        let mut nodes = self.nodes.lock();
        // What `NodeMemory::with_window` laid: its own runs at their offsets.
        let laid = match mem.phys.ppmem_space() {
            Some(sp) => mem
                .layout
                .iter()
                .filter(|r| r.at + r.len <= sp.window_size())
                .map(|r| Piece { owner: nodes.len(), index: r.index, src: r.src, at: r.at, len: r.len })
                .collect(),
            None => Vec::new(),
        };
        nodes.push(Shown {
            phys: Arc::downgrade(&mem.phys),
            layout: mem.layout.clone(),
            claim: mem.claim.clone(),
            routing,
            laid,
        });
    }

    /// The numbers or aliases a Hub publishes moved: one node's number moves
    /// its memory in the other's window too. Every window is laid now, from
    /// the calling thread, to the numbers as they stand under the lock; of two
    /// stores that race, the later refresh to take it lays the later values.
    fn refresh(&self) {
        let mut nodes = self.nodes.lock();
        for i in 0..nodes.len() {
            Self::lay(&mut nodes, i);
        }
    }

    /// Lay node `i`'s window to the numbers and aliases the Hubs publish now.
    fn lay(nodes: &mut [Shown], i: usize) {
        let numbers: Vec<_> = nodes.iter().map(|n| (n.routing.number(), n.routing.calias())).collect();
        let phys: Vec<_> = nodes.iter().map(|n| n.phys.upgrade()).collect();
        let Some(sp) = phys[i].as_ref().and_then(|p| p.ppmem_space()) else { return };
        let want = Self::wanted(nodes, &numbers, i, sp);
        // A piece the host could not map is not recorded as laid, so the
        // next call tries it again.
        let mut failed = Vec::new();
        for p in want.iter().filter(|p| !nodes[i].laid.contains(p)) {
            let Some(owner) = &phys[p.owner] else { continue };
            if let Err(e) = sp.map_range_of(&owner.banks()[p.index], p.src, p.at, p.len) {
                crate::dlog!(LogModule::Hub, "Hub: node {i} window at {:#x}: {e}", p.at);
                failed.push(*p);
            }
        }
        for old in &nodes[i].laid {
            for (at, len) in uncovered(old.at, old.len, &want) {
                sp.unmap_range(at, len);
            }
        }
        nodes[i].laid = want.into_iter().filter(|p| !failed.contains(p)).collect();
    }

    /// What node `i`'s window should hold, as its Hub routes a cached
    /// reference (`Hub::claim`): its own number first, then the neighbour's.
    fn wanted(nodes: &[Shown], numbers: &[(u16, u64)], i: usize, sp: &crate::ppmem::PpMemSpace) -> Vec<Piece> {
        let (number, calias) = numbers[i];
        let routed = nodes[i].claim.lock().as_ref().and_then(Weak::upgrade).is_some();
        let mut want = Vec::new();
        let mut show = |owner: usize, base: u64, from: u64, to: u64| {
            for r in nodes[owner].layout.iter() {
                let (lo, hi) = (r.at.max(from), (r.at + r.len).min(to));
                let piece = Piece { owner, index: r.index, src: r.src + (lo.wrapping_sub(r.at)), at: base + lo, len: hi.wrapping_sub(lo) };
                if lo < hi && sp.can_map(piece.src, piece.at, piece.len) {
                    want.push(piece);
                }
            }
        };
        if !routed || number == 0 {
            show(i, 0, 0, u64::MAX);
        } else {
            show(i, 0, 0, calias);
            if let Some(zero) = (0..nodes.len()).find(|&k| k != i && numbers[k].0 == 0) {
                show(zero, 0, calias, u64::MAX);
            }
        }
        if routed {
            for (k, &(n, _)) in numbers.iter().enumerate() {
                if n != 0 && (k == i || n != number) {
                    show(k, u64::from(n) * NODE_SPAN, 0, u64::MAX);
                }
            }
        }
        want
    }
}

/// The parts of `[at, at + len)` that no piece of `want` covers.
fn uncovered(at: u64, len: u64, want: &[Piece]) -> Vec<(u64, u64)> {
    let mut gaps = vec![(at, at + len)];
    for p in want {
        let (pa, pe) = (p.at, p.at + p.len);
        gaps = gaps
            .into_iter()
            .flat_map(|(a, e)| {
                if pe <= a || pa >= e {
                    vec![(a, e)]
                } else {
                    [(a, pa.max(a)), (pe.min(e), e)].into_iter().filter(|(x, y)| x < y).collect()
                }
            })
            .collect();
    }
    gaps.into_iter().map(|(a, e)| (a, e - a)).collect()
}

#[cfg(test)]
mod tests;
