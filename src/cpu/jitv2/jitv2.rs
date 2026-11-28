//! JIT v2 core data structures.
//!
//! See `clanker/jitv2/jit-v2-design.md` for the full design. This module holds the
//! per-page metadata the rest of the engine builds on (§2.4):
//!
//! - [`PhysicalCodePage`]: the mips executor's view of "what's compiled here" for
//!   the physical page it's currently executing out of.
//! - Generation counters live in the owning `BusDevice` (one per page for RAM,
//!   a single shared never-bumped counter for ROM — `BusDevice::gen_ptr`,
//!   `src/dev/mem.rs`, `src/dev/prom.rs`) and are read through a raw pointer here so the
//!   hot path avoids an indirect call through the device trait object.
//!
//! Threading model: the mips exec thread owns `PhysicalCodePage` management
//! (arrival, promotion — §6.1) and pushes compile requests onto the
//! [`CompileQueue`] (§6.4), a bounded queue plus a small hot lane that every
//! worker of the compile pool pops from (one worker per guest CPU unless
//! `[jitv2] threads` says otherwise). A worker publishes finished artifacts
//! back into the page's `entry_table`/`entry_bits` (§6.1.3), which is why
//! requests carry a mutable pointer. `[jitv2] compile = "sync"` instead
//! compiles on the CPU thread at the dispatch gate, with the pool stopped.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;

use parking_lot::Mutex;

use crate::cpu::mips_core::MipsCore;
use crate::cpu::mips_exec::ExecStatus;
use crate::cpu::jitv2::arena_ring::{ArenaRing, Switch};
use crate::cpu::jitv2::pool_barrier::PoolBarrier;
use crate::traits::{BusDevice, CodeKey, Device};

/// A compile request pushed from the mips exec thread to the compile pool
/// (§6.4). Carries the page pointer rather than a
/// snapshotted generation: the compile thread reads `gen` itself at snapshot
/// time (§6.5 step 2, `gen_snap = gen`) and re-reads it at publish time — the
/// generation at queue time is never consulted, only current-at-compile and
/// current-at-publish. The pointer is mutable because publish (§6.1.3) writes
/// into the page's `entry_table`/`entry_bits`/`artifact_list`.
/// The compile thread snapshots the page's whole `requested` bitmap
/// fresh at dequeue time, picking up whatever accumulated by then,
/// not just whichever offset happened to trigger this send.
///
/// # Safety
/// `page` must outlive the request — pages live for the lifetime of their
/// owning device (see [`PhysicalCodePage`]'s Send/Sync safety note).
#[derive(Debug, Clone, Copy)]
pub struct CompileRequest {
    pub page: *mut PhysicalCodePage,
    /// Live `STATUS_FR` bit at enqueue time, threaded through because the
    /// compile thread has no `MipsCore` to read it from itself — codegen's
    /// FPR-access emitters are FR-mode-specific and must match whatever mode
    /// the executor will actually be in when it calls the compiled function
    /// (same value `exec_decoded` used to run the interpreter fallback for
    /// this same arrival).
    pub compiled_for_fr1: bool,
    /// The CPU that asked, by its number in the pool: the page's bytes are
    /// read at that CPU's address for it, through that CPU's bus
    /// ([`CodeBuses::locate`]).
    pub cpu: u8,
    /// The page slot's tenancy when the CPU asked
    /// ([`PhysicalCodePage::tenancy`]). A request can outlive its page: the
    /// hot lane's copy is still queued when the first copy's compile ends,
    /// and the pool may then give the slot to other memory. The worker
    /// checks this before it reads the page
    /// ([`PhysicalCodePage::pin_for_compile`]).
    pub tenancy: u32,
}

unsafe impl Send for CompileRequest {}

/// Most CPUs one pool serves: a page keeps an address per CPU.
pub const POOL_CPUS: usize = crate::cpu::smp::MAX_CPUS;

/// "This CPU has not reached the page", in [`PhysicalCodePage`]'s `addr`.
const NO_ADDR: u32 = u32::MAX;

/// The two buses a CPU fetches through.
#[derive(Clone)]
pub struct CpuBuses {
    /// The CPU's address space in the pool.
    pub space: usize,
    /// Decodes a cached fetch.
    pub cached: Arc<dyn BusDevice>,
    /// Decodes an uncached fetch (the CPU's own bus).
    pub uncached: Arc<dyn BusDevice>,
}

/// The buses of every CPU that shares a pool, by its number in the pool. A
/// page is keyed by the memory behind it, which each CPU may reach at its own
/// address, so a compile reads the bytes at one CPU's address through that
/// CPU's bus.
#[derive(Default)]
pub struct CodeBuses(parking_lot::RwLock<[Option<CpuBuses>; POOL_CPUS]>);

impl CodeBuses {
    /// One CPU, number 0, that fetches everything through `bus`.
    pub fn single(bus: Arc<dyn BusDevice>) -> Self {
        let buses = Self::default();
        buses.set(0, 0, bus.clone(), bus);
        buses
    }

    pub fn set(&self, cpu: usize, space: usize, cached: Arc<dyn BusDevice>, uncached: Arc<dyn BusDevice>) {
        self.0.write()[cpu] = Some(CpuBuses { space, cached, uncached });
    }

    /// `bus` for CPU 0 if nothing named its buses (a pool started on its own).
    fn set_default(&self, bus: Arc<dyn BusDevice>) {
        let mut g = self.0.write();
        if g[0].is_none() {
            g[0] = Some(CpuBuses { space: 0, cached: bus.clone(), uncached: bus });
        }
    }

    /// Let go of every bus (the machine is going away).
    pub fn clear(&self) {
        *self.0.write() = Default::default();
    }

    /// A bus and the address on it where `page`'s bytes are right now:
    /// `first`'s if it has one, else any CPU's. An address a remap has since
    /// moved other memory under no longer decodes to the page's key and is
    /// passed over.
    pub fn locate(&self, page: &PhysicalCodePage, first: usize) -> Option<(Arc<dyn BusDevice>, u64)> {
        let g = self.0.read();
        std::iter::once(first).chain(0..POOL_CPUS).find_map(|cpu| {
            let (pfn, uncached) = page.addr_of(cpu)?;
            let buses = g.get(cpu)?.as_ref()?;
            let bus = if uncached { &buses.uncached } else { &buses.cached };
            let at = pfn as u64 * PAGE_SIZE as u64;
            // A page keyed by its address is its own space's alone.
            let mine = !page.key.is_address() || page.key == CodeKey::at_address(buses.space, pfn, uncached);
            (mine && page_is_at(page, &**bus, at)).then(|| (bus.clone(), at))
        })
    }
}

/// Whether `bus` still decodes `at` to `page`: to its bank's page, or, for a
/// page no bank names, to no bank still.
pub fn page_is_at(page: &PhysicalCodePage, bus: &dyn BusDevice, at: u64) -> bool {
    match bus.code_page(at) {
        Some(code) => code.key == page.key,
        None => page.key.is_address(),
    }
}

// ============================================================================
// Tunables & initial settings. Single source of truth: every knob that shapes
// JIT v2's steady-state memory footprint and flush/queue behavior lives here.
// See each constant's doc comment for the reasoning behind its specific value.
// ============================================================================

/// Number of 4 KiB physical frames in the largest ppmem window: the `PfnMap`
/// of a `Jitv2` built with no machine to size it from.
pub const PFN_MAP_ENTRIES: usize = crate::ppmem::LARGE_WINDOW_SIZE as usize / PAGE_SIZE as usize;

/// How many frames a machine's `PfnMap` needs: up to the end of its memory.
pub fn frames_for(memory: &crate::memory_map::MemoryMap) -> u64 {
    memory.end().div_ceil(PAGE_SIZE as u64)
}

/// The JIT frame for a physical address, or `None` above the `frames` that
/// `PfnMap` covers. Translation yields up to 40 bits. Code up there is not
/// compiled; it runs in the interpreter.
#[inline(always)]
pub fn pfn_of(phys_addr: u64, frames: u64) -> Option<Pfn> {
    let pfn = phys_addr / PAGE_SIZE as u64;
    (pfn < frames).then_some(pfn as Pfn)
}

/// Where a code page sits in the `PfnMap`: its frame, plus on a `split` machine
/// (an IP27 or IP35 node, whose uncached fetch can read other bytes) whether it was
/// fetched uncached.
#[inline(always)]
pub fn page_key(pfn: Pfn, uncached: bool, split: bool) -> u32 {
    if split { pfn << 1 | uncached as u32 } else { pfn }
}

/// Direct-mapped `pfn -> PageSlot` lookup for one address space: which pool
/// slot holds the page a CPU of that space reaches at a frame. A cache of the
/// answer the bus ([`BusDevice::code_page`]) and the pool's key index give,
/// so a page switch to a frame already reached takes two loads and no lock.
///
/// One `u32` per physical frame up to the end of the machine's memory
/// ([`frames_for`]), two on an IP27 or IP35 machine ([`page_key`]). Allocated once, at its
/// final size, by the pool's first claim. Never resized. `PFN_MAP_EMPTY` means
/// "no pool slot for this key".
///
/// Why not a hash map: `page_for` is on the page-crossing path (see
/// `jitv2_track_pcp`), and a profile of a real IRIX boot put it at ~4.5% of
/// total runtime with `alloc::alloc::Global`/`RawTable` frames directly above
/// it — the map's own incremental rehash-and-grow, since the pool claims a
/// fresh pfn on every page switch it hasn't seen. A direct index has no hash,
/// no probe sequence, no allocation ever, and no `pages[slot]` bounds-check
/// dance beyond the one the caller already does.
///
/// Boxed rather than inline so `Jitv2` itself stays small enough to move
/// cheaply (it is constructed into an `Arc<Mutex<..>>`).
pub struct PfnMap(Box<[AtomicU32]>);

/// A frame table entry is the pool slot in its low [`SLOT_BITS`] bits and the
/// low bits of the slot's tenancy above them (`PhysicalCodePage::tenancy`). A
/// slot is recycled for other memory without anyone visiting the frame
/// tables: the tenancy moves, and every entry made for the old tenant stops
/// matching ([`frame_entry_page`]).
pub const SLOT_BITS: u32 = 16;
/// Most slots a pool can have: `u16::MAX`, so no entry reads as
/// [`PFN_MAP_EMPTY`].
pub const MAX_POOL_SLOTS: usize = (1 << SLOT_BITS) - 1;

/// The frame table entry for `slot` in its tenancy `tenancy`.
#[inline(always)]
pub fn frame_entry(slot: PageSlot, tenancy: u32) -> u32 {
    tenancy << SLOT_BITS | slot
}

/// The slot a frame table entry names.
#[inline(always)]
pub fn frame_entry_slot(entry: u32) -> PageSlot {
    entry & ((1 << SLOT_BITS) - 1)
}

/// Whether `entry` was made for the tenant now in its slot, whose tenancy is
/// `tenancy`.
#[inline(always)]
pub fn frame_entry_current(entry: u32, tenancy: u32) -> bool {
    entry >> SLOT_BITS == tenancy & ((1 << (32 - SLOT_BITS)) - 1)
}

/// The slot each CPU of a pool is on, or `NO_SLOT`: one word per CPU, each on
/// a cache line of its own. A CPU publishes the slot before it trusts it
/// (store, then check the slot's tenancy again), and the pool never recycles a
/// published slot (it moves the tenancy, then reads these). One of the two
/// always sees the other, as long as both sides use `SeqCst`.
#[repr(align(128))]
pub struct CpuSlot(pub AtomicU32);

/// `PfnMap`'s own "no slot here" sentinel. Numerically identical to each
/// impl module's private `NO_SLOT` (both `u32::MAX`), spelled separately
/// because `PfnMap` is shared code that predates either module in this file
/// and cannot name a `mod`-private constant.
pub const PFN_MAP_EMPTY: u32 = u32::MAX;

impl PfnMap {
    /// One allocation, every frame empty.
    pub fn new(frames: usize) -> Self {
        // A zeroed allocation and one memset. Filling it element by element
        // is slow in a debug build and runs once per executor.
        let mut slots: Box<[u32]> = vec![0u32; frames].into_boxed_slice();
        // SAFETY: the pointer and length describe `slots`, and every byte
        // pattern is a valid `u32`.
        unsafe { std::ptr::write_bytes(slots.as_mut_ptr(), 0xFF, slots.len()) };
        const { assert!(PFN_MAP_EMPTY == u32::MAX) };
        const { assert!(size_of::<AtomicU32>() == size_of::<u32>() && align_of::<AtomicU32>() == align_of::<u32>()) };
        let raw = Box::into_raw(slots) as *mut [AtomicU32];
        // SAFETY: `AtomicU32` has the size and bit validity of `u32`, and the
        // assertion above pins its alignment, so the allocation layout matches.
        PfnMap(unsafe { Box::from_raw(raw) })
    }

    /// The slot for `pfn`, or `None` when unmapped.
    #[inline(always)]
    pub fn get(&self, pfn: u32) -> Option<u32> {
        let slot = self.0[pfn as usize].load(Ordering::Acquire);
        if slot == PFN_MAP_EMPTY { None } else { Some(slot) }
    }

    /// After the slot is claimed: another CPU's lock-free lookup that sees this
    /// sees the claim too. A claimed page's `pfn` and `gen` change only in a
    /// flush, which every CPU sits out, so they need no atomics of their own.
    #[inline(always)]
    pub fn insert(&mut self, pfn: u32, slot: u32) {
        self.0[pfn as usize].store(slot, Ordering::Release);
    }

    #[inline(always)]
    pub fn remove(&mut self, pfn: u32) {
        self.0[pfn as usize].store(PFN_MAP_EMPTY, Ordering::Release);
    }

    /// Drop every mapping (`mega_flush`). A memset of the table — flushes are rare
    /// and already far more expensive than this, so no generation-tag
    /// indirection is worth paying for on every lookup to avoid it.
    pub fn clear(&mut self) {
        for e in self.0.iter() {
            e.store(PFN_MAP_EMPTY, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const AtomicU32 {
        self.0.as_ptr()
    }
}


impl Default for PfnMap {
    fn default() -> Self { Self::new(PFN_MAP_ENTRIES) }
}

/// Upfront reservation for the shared `Codegen`'s Cranelift `ArenaMemoryProvider`
/// (`Codegen::new_module`'s own doc comment for why this exists at all). A
/// `PROT_NONE` virtual-address-space reservation (cheap — nothing is actually
/// committed/faulted-in until code is written there), not a RAM commitment, so
/// there's no real memory cost to reserving more than the steady-state working
/// set needs. Must stay in lockstep with `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`
/// below.
pub const ARENA_RESERVE_SIZE: usize = 512 * 1024 * 1024;

/// Flush threshold for the shared `Codegen`'s Cranelift arena, in bytes
/// actually reserved (`Codegen::packing_stats()`'s `reserved` — real
/// host-page-rounded arena footprint, not a function-count proxy).
/// `cranelift_jit::Memory` never frees on drop/replace (`Codegen::reset`'s own
/// doc comment), so nothing else bounds arena growth — a long-enough-running
/// compile (real IRIX boot, not just PROM) will otherwise exhaust the whole
/// `ARENA_RESERVE_SIZE` reservation. 16MiB of headroom covers the batch that
/// happens to be in flight when this trips (a batch isn't finalized/counted
/// until it flushes, so the real reservation can run slightly ahead of this
/// threshold between checks) while still flushing well before the arena's own
/// exhaustion error could ever fire — that error path (`comp::handle_request`'s
/// exhaustion match arm) stays as a belt-and-suspenders backstop, not the
/// primary trigger.
pub const CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES: u64 = ARENA_RESERVE_SIZE as u64 - 16 * 1024 * 1024;

/// [`CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES`], or `IRIS_JIT_ARENA_FLUSH_MB`: a low
/// value makes the compile workers' own flush happen often enough to test.
pub fn arena_flush_threshold() -> u64 {
    static T: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("IRIS_JIT_ARENA_FLUSH_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES, |mb| (mb << 20).min(CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES))
    })
}

/// How often a worker with a full arena and no empty one to move to looks
/// again. The wait ends when every CPU has passed a batch boundary, which is
/// well under a millisecond for a running CPU.
const ARENA_STALL_POLL: std::time::Duration = std::time::Duration::from_millis(1);

/// Default page-pool capacity for `Jitv2::new()` as embedded in `MipsExecutor`.
/// Sizing is a Phase 0 measurement per the design doc (§9, "Max live entries per
/// epoch"); `mega_flush` absorbs it being wrong in either direction. Now that
/// the whole pool is a single array allocated once at this capacity, a larger
/// capacity is a real memory cost, not just reserved address space — 4096 is a
/// deliberately modest working-set size rather than a generous upper bound;
/// `mega_flush`'s cost of getting this wrong low (a pool-exhaustion flush) is
/// cheap relative to permanently carrying a much larger array.
pub const JITV2_INITIAL_PAGE_CAPACITY: usize = 4096;

/// [`JITV2_INITIAL_PAGE_CAPACITY`], or `IRIS_JIT_PAGES`: a low value makes
/// the pool recycle page slots often enough to test.
pub fn page_pool_capacity() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("IRIS_JIT_PAGES")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map_or(JITV2_INITIAL_PAGE_CAPACITY, |n| n.clamp(16, MAX_POOL_SLOTS))
    })
}

/// Depth of the compile-request SPSC ring (§6.4 "bounded queue; drop on full —
/// hot pages re-trigger"). A starting guess, doubled from 1024 after a live
/// `j2 status` reading showed the compile thread genuinely falling behind at
/// that size (20.9% of dispatches dropped for a full queue, average depth at
/// dispatch 248.6/1024, out of 1,166,218 total dispatches during one session)
/// rather than the queue mostly sitting near-empty.
pub const COMPILE_QUEUE_CAPACITY: usize = 2048;

/// Arrivals at a queued-but-uncompiled page before it jumps the queue: code
/// being executed this often right now is worth compiling before older,
/// colder requests (a program started just after a large compile otherwise
/// waits seconds behind that compile's backlog, measured with jitcov on IP28).
pub const HOT_PROMOTE_ARRIVALS: u32 = 256;

/// How many arrivals the dispatch gate folds into a page's shared hot-lane
/// counter at a time.
///
/// The counter lives in the page, and the page is shared with the compile
/// worker: writing it on every arrival put a shared atomic write on the CPU
/// thread's hot path, where it ping-ponged with the worker's copy of the same
/// cache line. The CPU counts arrivals in a field of its own instead and
/// folds the total in batches, so every arrival is still counted — the
/// promotion threshold is unchanged — and the shared line is written once
/// per batch.
///
/// One bounded exception: a batch that spans a wait boundary (the page's
/// request completes while the CPU still holds part of a batch) carries at
/// most `HOT_ARRIVAL_BATCH - 1` arrivals into the *next* wait, so the page
/// can be promoted a few arrivals early out of `HOT_PROMOTE_ARRIVALS`. The
/// threshold is a heuristic — how hard a page is being executed — so a few
/// arrivals either way do not change what it is for.
pub const HOT_ARRIVAL_BATCH: u32 = 16;

/// Requests the hot lane holds (see `CompileQueue::hot`), shared by every CPU
/// and worker of the machine. A tunable: few, so a page that turns hot waits
/// behind few compiles, not pages that were hot a backlog ago; a refused
/// promotion is retried by its page. Measured on a one-CPU Octane: 8, 16, 32
/// and 256 left hot user code interpreted 0.1-0.4, 0.7-1.5, 1.2 and 3.7-12 s.
const HOT_QUEUE_CAPACITY: usize = 8;

/// Wake-up channel between the compile-request producers and the compile
/// pool's workers.
///
/// The workers used to poll the request queue on a 200µs timer. That cost a
/// measurable slice of a host core per worker whenever the guest had nothing
/// to compile (measured 2026-11-16 on an idle guest: 1.4% of a core for one
/// worker) and kept a small host from settling into deep C-states — the
/// same failure the device threads had, and the same fix
/// (`clanker/perf/idle-poll-backoffs-for-small-hosts.md`): size a wait for
/// idle cost, and let the producer wake the waiter.
///
/// The producers now unpark the workers directly. `park`/`unpark` tokens are
/// sticky — an `unpark` that arrives while the worker is busy is remembered,
/// so the next `park` returns immediately instead of sleeping through the
/// request — so a wakeup cannot be lost. That is what lets the worker's
/// backstop be long: it only has to cover the periodic pending-publish
/// sweep, not request latency.
///
/// The producer runs on the CPU thread, so it only pays for a wake when a
/// worker is actually asleep: `sleepers` counts the workers inside
/// [`Self::park_idle`], and [`Self::wake_if_sleeping`] reads it before
/// touching the registry. While the pool is busy that is one load per
/// accepted request.
pub struct CompileWake {
    /// The worker threads, registered when a worker starts and cleared when
    /// the pool stops. A mutex, not an atomic swap: the registry is written
    /// only at pool start/stop, and the producer reads it only when a worker
    /// is asleep.
    workers: Mutex<Vec<std::thread::Thread>>,
    /// Workers between deciding to park and waking up (see
    /// [`Self::park_idle`]). Zero means every worker is running and will
    /// look at the queues again on its own.
    sleepers: AtomicUsize,
}

impl CompileWake {
    fn new() -> Self {
        Self { workers: Mutex::new(Vec::new()), sleepers: AtomicUsize::new(0) }
    }

    /// Park the calling worker until a producer wakes it or `backstop`
    /// expires, unless a request is already waiting in `queue` or `hot`.
    ///
    /// The count goes up before the queues are checked again, and the
    /// producer pushes before it reads the count, with a `SeqCst` fence on
    /// each side between the write and the read. So either this worker sees
    /// the request and does not park, or the producer sees the sleeper and
    /// unparks it (and the token is sticky, so an unpark that lands before
    /// the `park_timeout` is not lost).
    fn park_idle(&self, queue: &CompileLane, hot: &CompileLane, backstop: std::time::Duration) {
        self.sleepers.fetch_add(1, Ordering::SeqCst);
        std::sync::atomic::fence(Ordering::SeqCst);
        if queue.is_empty() && hot.is_empty() {
            std::thread::park_timeout(backstop);
        }
        self.sleepers.fetch_sub(1, Ordering::SeqCst);
    }

    /// The producer's wake, after a push was accepted: unpark the workers
    /// only if one of them is asleep (see [`Self::park_idle`] for why the
    /// fence makes skipping safe).
    #[inline]
    fn wake_if_sleeping(&self) {
        std::sync::atomic::fence(Ordering::SeqCst);
        if self.sleepers.load(Ordering::SeqCst) != 0 {
            self.notify_all();
        }
    }

    /// Publish this worker's thread handle so a producer can wake it.
    fn register(&self, thread: std::thread::Thread) {
        self.workers.lock().push(thread);
    }

    /// Wake every worker unconditionally. The producer goes through
    /// [`Self::wake_if_sleeping`]; this is for the rare callers (the flush
    /// and seal leaders, `stop`) that must reach every worker whatever it is
    /// doing. A token left on a running worker costs one extra loop
    /// iteration.
    fn notify_all(&self) {
        for thread in self.workers.lock().iter() {
            thread.unpark();
        }
    }

    /// Forget the workers of a stopped pool, so a later `start` does not
    /// accumulate dead thread handles.
    fn clear(&self) {
        self.workers.lock().clear();
    }
}

/// One compile-request lane: the bounded queue plus the pool's wake-up
/// channel, so a producer that pushes a request can wake the workers
/// without going through `Jitv2`'s mutex — the dispatch gate must not take
/// that lock (see `MipsExecutor::jitv2_compile_queue_handle`).
///
/// Derefs to the queue, so every existing use — `push`, `pop`, `len`,
/// `is_full` — keeps working unchanged.
pub struct CompileLane {
    queue: crossbeam_queue::ArrayQueue<CompileRequest>,
    wake: Arc<CompileWake>,
}

impl CompileLane {
    fn new(capacity: usize, wake: Arc<CompileWake>) -> Self {
        Self { queue: crossbeam_queue::ArrayQueue::new(capacity), wake }
    }
}

impl std::ops::Deref for CompileLane {
    type Target = crossbeam_queue::ArrayQueue<CompileRequest>;
    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}

/// Put a request on the hot lane, saying whether it took it. On a refusal the
/// caller re-arms the page: its FIFO copy, when it has one, is still queued,
/// but a page the full FIFO refused has no other request. An accepted push
/// wakes a sleeping worker, which must see the request now, not after its
/// backstop expires; a refused one does not, since a full lane means the
/// workers have work.
pub fn push_hot_request(hot: &Arc<CompileLane>, req: CompileRequest) -> bool {
    let accepted = hot.push(req).is_ok();
    if accepted {
        hot.wake.wake_if_sleeping();
    }
    accepted
}


/// Page size for JIT v2 (§2.4) — matches the MIPS TLB/cache page granularity
/// used throughout the codebase, and the granularity of the per-page
/// generation counters. An architectural property of the machine: JIT v2
/// keys pages by PFN at this size, and `mem.rs` re-exports it as
/// `JITV2_PAGE_SIZE` for its own generation-counter indexing. Canonical home
/// for this constant.
pub const PAGE_SIZE: u32 = 4096;

/// Carried by a page's generation counter once for each CPU whose cache
/// holds lines there (`PpMemSpace::hold_line_page`): that CPU runs the page
/// from its own view, so it is not compiled (`comp::snapshot_page`). Each
/// comes off with an ordinary bump, below the held page's last publish, so
/// `publish` takes any change of generation as new bytes.
pub const GEN_HELD: u64 = 1 << 48;

/// Number of possible entry offsets per page: one per 4-byte-aligned word
/// (MIPS instructions are always word-aligned), i.e. `PAGE_SIZE / 4` (§2.4:
/// `entry_bits` 16×u64 = 1024 bits, `entry_table` 1024 entries).
pub const ENTRIES_PER_PAGE: usize = (PAGE_SIZE / 4) as usize;

/// u64 words needed for a 1-bit-per-entry bitmap over `ENTRIES_PER_PAGE` offsets.
pub const BITMAP_WORDS: usize = ENTRIES_PER_PAGE / 64;

// ============================================================================
// Local Tunables & Sentinels
// ============================================================================

/// How many of the most-recently-used pages survive a flush with their
/// `requested`/`denied` bitmaps (and pfn/gen/hash entry) intact, to be


/// Force-seal trigger for a continuously busy batching worker — see
/// `worker_loop`'s own comment at its call site. `handle_request_deferred`
/// finalizes every compile immediately, but a non-forced finalize only
/// seals a page the bump cursor has already moved past, so `pending` can
/// otherwise grow without bound while the queue never goes empty long
/// enough to reach the idle-timeout sweep. Small and arbitrary — this only
/// needs to be "small enough that pending never grows unbounded," not tuned
/// to any particular page size (unlike the old page-cross trigger it
/// replaces, this one has no reason to line up with `ENTRIES_PER_PAGE`).
const PENDING_FORCE_SEAL_THRESHOLD: usize = 64;

/// Queue-drain fallback: how long the compile-request queue must stay
/// continuously empty (wall-clock, tracked across repeated empty polls, not
/// "first empty poll") before `worker_loop` force-seals whatever's left
/// rather than waiting for a page to fill on its own — see `worker_loop`'s
/// own comment at its call site for the two-phase (non-forced then forced)
/// sealing design this backstops.
const IDLE_FORCE_SEAL_THRESHOLD: std::time::Duration = std::time::Duration::from_millis(100);

/// Backstop between empty-queue polls in `worker_loop`. The pool is woken by
/// the producer (`CompileWake`), so this only has to cover the periodic
/// pending-publish sweep and the "a worker that missed its wakeup" case;
/// `park`/`unpark` tokens make a missed wakeup impossible.
///
/// Sized for idle cost, not for latency: the old 200µs poll cost a
/// measurable slice of a host core per worker whenever the guest had nothing
/// to compile (measured 2026-11-16: 1.4% of a core for one worker) and kept
/// a small host from settling. Long enough that `IDLE_FORCE_SEAL_THRESHOLD`'s
/// wall-clock idle window still resolves in a handful of ticks.
const WORKER_IDLE_PARK_BACKSTOP: std::time::Duration = std::time::Duration::from_millis(20);

/// Backstop for a worker that still has a publish outstanding (its own
/// range is gap-blocked behind another worker's, so the code is compiled but
/// not yet sealed and published). The retry has to be prompt — the guest is
/// running that page interpreted until it lands — so this stays at the old
/// poll interval. It only applies while something is pending, which the
/// pending threshold and the idle-timeout force-seal both bound.
const WORKER_IDLE_PENDING_BACKOFF: std::time::Duration = std::time::Duration::from_micros(200);

/// How often a flush leader with several CPUs retries `World::try_lead`
/// while a CPU leads a flush of its own.
const BARRIER_FOLLOWER_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

/// Legacy arrival-count setting, exposed by `j2 min-calls [N]`.
/// The setter stores the value and the monitor reads it back, but whole-page
/// dispatch does not consult it: eligible offsets request compilation on
/// their first arrival. Defaults to 0.
static MIN_CALLS_BEFORE_COMPILE: AtomicU64 = AtomicU64::new(0);

pub fn set_min_calls_before_compile(n: u64) {
    MIN_CALLS_BEFORE_COMPILE.store(n, Ordering::Relaxed);
}

pub fn min_calls_before_compile() -> u64 {
    MIN_CALLS_BEFORE_COMPILE.load(Ordering::Relaxed)
}

// ============================================================================
// End tunables
// ============================================================================

/// Physical frame number. Physical addresses are keyed by PFN, never by VA (§2.1).
pub type Pfn = u32;

/// Sentinel `pfn` for an unclaimed slot (`PhysicalCodePage::new`/
/// `reset_to_unclaimed`) — deliberately NOT `0`: pfn 0 is a real, legitimate
/// physical page (low RAM), so zeroing `pfn` on reset let any stale reference
/// to a freed slot (a dangling raw pointer held past a `mega_flush` eviction,
/// or a frame-table entry that wasn't removed before the reset) silently
/// read back as "page 0's real data" instead of something obviously wrong —
/// undetectable by inspection (`j2 pcp` on a stale pointer would show a
/// plausible-looking pfn/gen/entry_gen for the wrong page, not a crash or an
/// obvious sentinel). `Pfn::MAX` can never collide with a real pfn (would
/// require 16TB+ of physical address space at `PAGE_SIZE` granularity).
pub const UNCLAIMED_PFN: Pfn = Pfn::MAX;

/// No FR re-pin outstanding — see `PhysicalCodePage::fr_repin`. Zero so a
/// freshly-zeroed or reset slot means "nothing demanded", the safe default.
pub const FR_REPIN_NONE: u8 = 0;
/// An FR-guard bail observed live FR=0 and demands the page be re-pinned to
/// FR0 at its next compile.
pub const FR_REPIN_FR0: u8 = 1;
/// An FR-guard bail observed live FR=1 and demands the page be re-pinned to
/// FR1 at its next compile.
pub const FR_REPIN_FR1: u8 = 2;

/// Compiled-function ABI (§6.1.2's "handler ABI", simplified for this storage
/// pass — no `DecodedInstr`/state-struct plumbing yet, just direct MipsCore
/// access): takes a pointer to the executor's `MipsCore` and returns the same
/// `ExecStatus` every interpreter handler returns. `vbase` derivation, the
/// two mirrored checks (§3.2), and exit-stub materialization all live inside
/// the compiled body once codegen exists — this signature is just the call
/// boundary.
pub type JitFn = unsafe extern "C" fn(*mut MipsCore) -> ExecStatus;

/// What `func` holds when a page has no code: callable, and sends the CPU to the interpreter.
pub unsafe extern "C" fn jit_no_code(_core: *mut MipsCore) -> ExecStatus { crate::cpu::mips_exec::EXEC_FALLBACK }

/// The `func` value that means "no code", as the pointer a page stores.
pub const NO_CODE: JitFn = jit_no_code;

/// A page-level bitmap over `ENTRIES_PER_PAGE` offsets — one bit per
/// 4-byte-aligned word (§13.1). `BITMAP_WORDS` (=16) `AtomicU64`s.
pub type EntryBitmap = [AtomicU64; BITMAP_WORDS];

fn new_bitmap() -> EntryBitmap {
    [const { AtomicU64::new(0) }; BITMAP_WORDS]
}

/// All-ones bitmap — the fresh/reset state for `denied` (§13.1: inverted
/// sense, 1 = still eligible to compile), so a freshly-claimed or
/// just-reset page means "everything eligible" without a separate memset
/// pass anywhere that constructs one.
fn new_bitmap_all_set() -> EntryBitmap {
    [const { AtomicU64::new(u64::MAX) }; BITMAP_WORDS]
}

#[inline]
fn bitmap_test(bm: &EntryBitmap, offset: usize, order: Ordering) -> bool {
    bm[offset >> 6].load(order) & (1u64 << (offset & 63)) != 0
}

/// Test-and-set a single bit; returns `true` iff this call is the one that
/// set it (i.e. it was previously clear) — same "first caller wins" contract
/// as the old per-entry `ENTRY_SCHEDULED` flag's `try_schedule`.
#[inline]
fn bitmap_test_and_set(bm: &EntryBitmap, offset: usize, order: Ordering) -> bool {
    let prev = bm[offset >> 6].fetch_or(1u64 << (offset & 63), order);
    prev & (1u64 << (offset & 63)) == 0
}

/// Snapshot every word of a bitmap into a plain array (relaxed loads — see
/// §13.3 step 3: this need not be seqlock'd against anything, under- or
/// over-collecting relative to a single instant is harmless here).
fn bitmap_snapshot(bm: &EntryBitmap) -> [u64; BITMAP_WORDS] {
    std::array::from_fn(|i| bm[i].load(Ordering::Relaxed))
}

/// Whether every set bit in `subset` is also set in `bm` (relaxed loads).
fn bitmap_is_subset_of(subset: &[u64; BITMAP_WORDS], bm: &EntryBitmap) -> bool {
    (0..BITMAP_WORDS).all(|i| subset[i] & !bm[i].load(Ordering::Relaxed) == 0)
}

/// [`bitmap_is_subset_of`] for two plain arrays — the churn-avoidance
/// snapshot's bitmaps (`CompileSnapshot`) are ordinary `[u64; _]` behind a
/// `Mutex`, not atomics, since nothing reads them lock-free.
fn bitmap_is_subset_of_plain(subset: &[u64; BITMAP_WORDS], of: &[u64; BITMAP_WORDS]) -> bool {
    (0..BITMAP_WORDS).all(|i| subset[i] & !of[i] == 0)
}

/// OR `bits` into `bm`, word by word, at the given ordering — used to publish
/// a compile's newly-covered entry points into `compiled` (§13.3 step 6).
fn bitmap_union_from(bm: &EntryBitmap, bits: &[u64; BITMAP_WORDS], order: Ordering) {
    for i in 0..BITMAP_WORDS {
        if bits[i] != 0 {
            bm[i].fetch_or(bits[i], order);
        }
    }
}

/// Whether a bitmap is entirely zero (relaxed loads) — used by the
/// pre-compile/pre-publish subsumption checks (§13.3 steps 4/6) to skip the
/// "is X a subset of Y" walk when there's nothing requested at all.
fn bitmap_is_empty(bm: &[u64; BITMAP_WORDS]) -> bool {
    bm.iter().all(|&w| w == 0)
}

/// Process-wide JIT event counters, displayed by `j2 status` only under the
/// `developer` feature — plain `AtomicU64`s rather than per-`Jitv2` state
/// protected by a lock, since these are touched from both the compile
/// thread (`comp::handle_request`, on every request, inline or threaded)
/// and the CPU thread (`jit_kill_entry`, on every FR-mismatch bail —
/// `mips_exec.rs`) and none of them need to be read together atomically
/// with anything else; a lock here would just be hot-path contention for no
/// correctness benefit. The struct and its `Arc<JitStats>` field on `Jitv2`
/// exist unconditionally (not `#[cfg(feature = "developer")]`) so nothing
/// on the path from `Jitv2::new`/`CompileQueue::start` down to
/// `handle_request` needs a second, feature-gated copy of itself just to
/// thread one more `Arc` through — only the actual `.fetch_add()` call
/// sites and the `j2 status` display are gated. Survives `mega_flush`
/// deliberately (counts are lifetime totals, not "since last reset" —
/// `Codegen::function_count`'s own since-reset counter already covers that
/// angle).
/// Why a compile request was declined — `failed_compiles`'s single counter
/// doesn't distinguish these, but they're different enough situations to
/// want separately (`j2 status`'s "rejections by reason" breakdown):
/// a codegen gap you could go implement an emitter for reads very
/// differently from a Cranelift verifier bug, which reads differently again
/// from "the analyzer and codegen's emitter tables disagree" (should be
/// structurally impossible after `opcode_support::has_emitter` unified them —
/// a nonzero count here means they've drifted again).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "developer")]
pub enum RejectReason {
    /// `walk_bounded` reported the entry offset itself unreachable —
    /// architecturally excluded (COP0, CACHE, LL/SC, ...) or a codegen gap
    /// at the very first instruction, `analyzer::classify` returned
    /// `Excluded` either way (`comp::handle_request`'s `!non_empty` arm).
    EntryExcluded,
    /// `compile_region`'s upfront rejection loop found a visited instruction
    /// with no emitter in any of the four lookup tables. Per
    /// `opcode_support::has_emitter` being the single source of truth both
    /// `analyzer::classify` and this loop consult, this should be
    /// unreachable in practice — every instruction the analyzer walked past
    /// as `Sequential`/`Branch`/`Jump`/`RegJump` already passed the same
    /// check. Tracked anyway as a canary: a nonzero count here means the
    /// tables have drifted apart again, the exact bug class
    /// `opcode_support.rs` was built to close structurally.
    AnalyzerCodegenDisagreement,
    /// Cranelift's `define_function` returned `ModuleError::Compilation` —
    /// this module emitted IR the verifier rejected, a real bug in some
    /// emitter, not an unsupported-instruction-shape decline (those are all
    /// caught by the upfront loop before Cranelift is ever invoked).
    CraneliftVerifierError,
    /// The walked region's instruction count fell below
    /// `comp::min_instrs_to_compile()` (`j2 min-instrs`) — not a codegen gap
    /// at all, just not judged worth the fixed per-compile cost. See
    /// `comp::MIN_INSTRS_TO_COMPILE`'s own doc comment.
    TooShort,
}
#[cfg(feature = "developer")]
pub const REJECT_REASON_COUNT: usize = 4;
#[cfg(feature = "developer")]
impl RejectReason {
    /// Every variant, in display order. The single source of truth for
    /// anything that enumerates reasons (`j2 stats`'s breakdown) — a new
    /// variant missing from here would silently never be reported. `index()`'s
    /// exhaustive `match` is what forces this to be updated: adding a variant
    /// without extending it fails to compile, and the length assertion below
    /// catches a variant added to the match but not to this list.
    pub const ALL: [RejectReason; REJECT_REASON_COUNT] = [
        RejectReason::EntryExcluded,
        RejectReason::AnalyzerCodegenDisagreement,
        RejectReason::CraneliftVerifierError,
        RejectReason::TooShort,
    ];

    pub fn index(self) -> usize {
        match self {
            RejectReason::EntryExcluded => 0,
            RejectReason::AnalyzerCodegenDisagreement => 1,
            RejectReason::CraneliftVerifierError => 2,
            RejectReason::TooShort => 3,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            RejectReason::EntryExcluded => "entry excluded (unsupported first instruction / architectural exclusion)",
            RejectReason::AnalyzerCodegenDisagreement => "analyzer/codegen disagreement (should be unreachable — see doc comment)",
            RejectReason::CraneliftVerifierError => "Cranelift verifier error (real emitter bug)",
            RejectReason::TooShort => "region too short (below j2 min-instrs threshold)",
        }
    }
}

#[derive(Default)]
pub struct JitStats {
    /// Times a CPU forgot its addresses because a memory controller moved
    /// memory (`Jitv2::forget_addresses`). No code is dropped for one.
    pub remaps: AtomicU64,
    /// Code arenas the compile pool has retired, and the pages that had code
    /// in one when it went (they compile again when next entered).
    pub arena_retires: AtomicU64,
    pub arena_retired_pages: AtomicU64,
    /// Page slots taken back from memory nobody was running and given to
    /// other memory (`Jitv2::recycle_slot`).
    pub pages_recycled: AtomicU64,
    /// Successful `compile_region` calls that got published
    /// (`comp::handle_request`'s `Some(jit_fn)` arm).
    pub compiles: AtomicU64,
    /// `compile_region` declines or `walk_bounded` exclusions
    /// (`comp::handle_request`'s two `page.denylist(offset)` sites) — codegen
    /// gaps and analyzer-excluded entry offsets alike, not distinguished
    /// further (both end up sticky-denylisted the same way). See
    /// `reject_reasons` for the breakdown by cause.
    pub failed_compiles: AtomicU64,
    /// Per-`RejectReason` breakdown of `failed_compiles`, indexed by
    /// `RejectReason::index()`. Excludes the arena-out-of-memory outcome
    /// (`comp::handle_request`'s `last_compile_ran_out_of_memory` arm) —
    /// that one isn't a rejection of the *region*, it retries on its own,
    /// so it doesn't belong in a "why did this instruction never compile"
    /// breakdown; it's counted in `failed_compiles` but not here.
    #[cfg(feature = "developer")]
    pub reject_reasons: [AtomicU64; REJECT_REASON_COUNT],
    /// `jit_kill_entry` calls — a compiled unit's FR-mode-mismatch guard
    /// bailing and un-publishing its own entry (`emit_fr_mode_guard`'s FR
    /// mismatch design, `MipsCore::kill_entry_fn`'s doc comment).
    pub kill_entry_calls: AtomicU64,
    /// Total `CompileQueue::send` calls (both accepted and dropped) —
    /// `j2 status`'s FIFO-fullness section denominator. `developer`-only:
    /// unlike `compiles`/`failed_compiles`/`kill_entry_calls` (rare events,
    /// off the hot path), `send` runs on every dispatch-gate arrival that
    /// misses the JIT cache — real per-instruction-adjacent traffic, so the
    /// extra atomic touch these three fields cost is worth avoiding outside
    /// a diagnostics build.
    #[cfg(feature = "developer")]
    pub compile_queue_dispatches: AtomicU64,
    /// `CompileQueue::send` calls that dropped the request because the ring
    /// buffer (`COMPILE_QUEUE_CAPACITY`) was already full — see `send`'s own
    /// doc comment for why a drop here is a normal, non-fatal outcome (the
    /// hot page just re-triggers the request on a later arrival), not an
    /// error; this counter exists to tell whether that's happening often
    /// enough to be a real bottleneck, not a rare edge case.
    #[cfg(feature = "developer")]
    pub compile_queue_full: AtomicU64,
    /// Running sum of `Producer::slots()`-derived occupancy
    /// (`capacity - free_slots`) sampled at every `send` call — divide by
    /// `compile_queue_dispatches` for the mean queue depth the compile
    /// thread is actually running at. A cumulative sum rather than a
    /// separate min/max/histogram: cheap (one more atomic add on an
    /// already-atomic-touching path), and mean depth is the number that
    /// actually answers "is the queue usually near-empty (compile thread
    /// keeping up) or usually near-full (compile thread falling behind)."
    #[cfg(feature = "developer")]
    pub compile_queue_depth_sum: AtomicU64,
    /// `try_force_seal` calls triggered by `pending` crossing
    /// `PENDING_FORCE_SEAL_THRESHOLD` on a continuously busy worker — `j2
    /// stats`'s batching section, split from `batch_flushes_queue_drain` so
    /// it's possible to tell whether a worker is getting throttled by
    /// sustained load (this counter) or is mostly idling between bursts
    /// (that one).
    #[cfg(feature = "developer")]
    pub batch_flushes_pending_threshold: AtomicU64,
    /// `try_force_seal` calls triggered by the compile queue draining empty
    /// (`worker_loop`'s `None` arm's idle-timeout sweep) — see
    /// `batch_flushes_pending_threshold`.
    #[cfg(feature = "developer")]
    pub batch_flushes_queue_drain: AtomicU64,
    /// High-water mark of `pending` just before any force-seal sweep
    /// actually published something (i.e. sampled right before the sweep
    /// runs, not after) — `j2 stats`'s "biggest backlog we ever let build
    /// up" figure. A `Relaxed` compare-and-swap loop, not a simple
    /// `fetch_max` (stabilized in std but this codebase's MSRV predates it
    /// for `AtomicU64` — matches the CAS-loop pattern already used elsewhere
    /// in this codebase for high-water-mark tracking).
    #[cfg(feature = "developer")]
    pub batch_max_pending: AtomicU64,
    /// Interpreter-fallback words compiled into published regions
    /// (`comp::handle_request`, incremented once per `is_fallback` word). The
    /// direct answer to "did the fallback path actually run this session?" — a
    /// clean boot with `j2 fallback on` means nothing if this is 0 (the flag was
    /// flipped after the boot-critical regions already compiled without it).
    /// Also `fallback_regions`: published regions that contained at least one
    /// fallback word, so you can tell a few big fallback loops from many tiny
    /// ones.
    #[cfg(feature = "developer")]
    pub fallback_words: AtomicU64,
    #[cfg(feature = "developer")]
    pub fallback_regions: AtomicU64,
}

/// Why a force-seal sweep ran — see `JitStats::batch_flushes_pending_threshold`/
/// `batch_flushes_queue_drain`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "developer")]
pub enum BatchFlushReason {
    PendingThreshold,
    QueueDrain,
}

#[cfg(feature = "developer")]
impl JitStats {
    /// Bump both `failed_compiles` and the per-reason breakdown together —
    /// the two call sites (`comp::handle_request`'s two denylist arms,
    /// `Codegen::compile_region`'s two `None` returns) always want to record
    /// both, and a shared helper means they can't drift out of sync (one
    /// incremented without the other).
    pub fn record_reject(&self, reason: RejectReason) {
        self.failed_compiles.fetch_add(1, Ordering::Relaxed);
        self.reject_reasons[reason.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Record one force-seal sweep: bump the trigger-specific counter and
    /// update the high-water mark from `pending_len` (the backlog size just
    /// before this sweep drained it).
    pub fn record_batch_flush(&self, pending_len: usize, reason: BatchFlushReason) {
        match reason {
            BatchFlushReason::PendingThreshold => self.batch_flushes_pending_threshold.fetch_add(1, Ordering::Relaxed),
            BatchFlushReason::QueueDrain => self.batch_flushes_queue_drain.fetch_add(1, Ordering::Relaxed),
        };
        let pending_len = pending_len as u64;
        let mut cur = self.batch_max_pending.load(Ordering::Relaxed);
        while pending_len > cur {
            match self.batch_max_pending.compare_exchange_weak(cur, pending_len, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(actual) => cur = actual,
            }
        }
    }
}

/// Fallback generation counter for a page whose backing `BusDevice` doesn't
/// implement `gen_ptr` (MMIO, etc — the trait's default returns null). A
/// single `static`, shared by every such page rather than one dummy per page:
/// it never advances (same as a ROM's shared, never-bumped counter — see
/// `PhysicalCodePage::gen`'s doc comment), so there's no reason for each page
/// to have its own copy. Pointing `gen` here instead of leaving it null means
/// `current_gen()`/every dispatch-path deref is unconditionally valid — no
/// null check needed anywhere, ever (`claim()` is the only place that decides
/// whether a page's real `gen_ptr` or this fallback gets used). If code
/// somehow executes out of true MMIO, this makes the JIT treat it like an
/// immutable ROM page instead — the JIT is not in the business of making
/// that scenario correct or fast, just of not crashing on it.
static NEVER_COMPILABLE_GEN: AtomicU64 = AtomicU64::new(0);

/// Per-physical-page code cache metadata (§13.1 — one compiled function per
/// page, internal dispatch by entry offset, superseding the old one-function-
/// per-entry `JitEntry` table). One instance per physical RAM/ROM page that
/// has ever been a JIT compilation target; the executor holds a pointer to
/// the page it is currently executing out of.
pub struct PhysicalCodePage {
    /// The memory this page is: its bank and the page within it
    /// ([`CodeKey`]), the same for every CPU and through every remap.
    /// [`CodeKey::NONE`] while the slot is unclaimed.
    pub key: CodeKey,
    /// Where each CPU that shares the pool last reached this page, by the
    /// CPU's number in the pool: `frame << 1 | uncached`, or `NO_ADDR`. A
    /// compile reads the bytes there through that CPU's bus
    /// ([`CodeBuses::locate`]). A remap can leave one stale; nothing trusts it
    /// without asking the bus for the key there again.
    addr: [AtomicU32; POOL_CPUS],
    /// The page's 1024 words in its bank's own storage
    /// ([`crate::traits::CodePage::words`]), where a compile reads them, or
    /// null for a bank that hands none out: then the compile reads through a
    /// CPU's bus at `addr`.
    words: *const u32,
    /// Free list pointer: singly-linked free list threaded through
    /// `Jitv2::free_head`. `NO_SLOT` when claimed or at tail of free list.
    next: u32,
    /// Pointer to this page's generation counter, obtained from the owning
    /// `BusDevice` via `gen_ptr` (§2.4, §7). RAM devices return one counter
    /// per page; ROM devices point every page at a single counter that is
    /// initialized to 0 and never bumped, since ROM content is immutable.
    /// Never null — a device with no real gen tracking (MMIO, etc — `gen_ptr`
    /// returns null) gets pointed at the shared [`NEVER_COMPILABLE_GEN`]
    /// fallback instead, by `claim()`.
    gen: *const AtomicU64,
    /// Offsets some dispatch has asked for an entry point at (§13.2) — a
    /// superset of `compiled` in general (a bit can be requested long before
    /// any compile picks it up, and stays requested across however many
    /// compiles it takes to eventually get covered).
    requested: EntryBitmap,
    /// Inverted sticky-rejection bitmap: **1 = still eligible to compile, 0 =
    /// permanently denied** for the page's current generation (§13.1, §13.6).
    /// Inverted specifically so it composes with `bitmap_is_subset_of`/
    /// dispatch's other bitmaps the same way — but that means a fresh/reset
    /// page must be explicitly initialized all-ones (`new_bitmap_all_set`,
    /// not the plain zeroed `new_bitmap` every other bitmap here uses), or
    /// every offset would read as denylisted before anything ever ran.
    /// Reset to all-ones again whenever a publish actually advances
    /// `entry_gen` (§13.6) — a pure entry-coverage publish against unchanged
    /// bytes leaves it alone.
    denied: EntryBitmap,
    /// Offsets that are live entry points into `func` right now (§13.2).
    /// Authoritative for dispatch together with `entry_gen` — see
    /// `is_runnable`.
    compiled: EntryBitmap,
    /// The page's one compiled function, or `jit_no_code` if nothing has
    /// published yet. Never null. Validity is owned by `compiled`'s bits together with `entry_gen`
    /// matching `current_gen()` — see `is_runnable`.
    func: std::sync::atomic::AtomicPtr<()>,
    /// The FR mode `func` was compiled for. Distinct from the pin (`fr1`),
    /// which is the mode the *next* compile targets: the two disagree while a
    /// page used under both modes waits for its recompile. `publish` needs
    /// it, because a compile for the other mode is never redundant with the
    /// installed code, whatever entries the two share. Written only under
    /// `publish_lock`.
    func_fr1: std::sync::atomic::AtomicBool,
    /// How many times this slot has been given to other memory. It moves,
    /// under `publish_lock`, when the pool takes the slot back: frame table
    /// entries and publishes made for the old tenant then no longer match.
    tenancy: AtomicU32,
    /// A CPU entered the page since the pool's recycling hand last passed.
    /// The hand clears it and takes the first page it finds clear.
    referenced: std::sync::atomic::AtomicBool,
    /// The arena `func` lives in, as that arena's epoch
    /// (`PagedArenaState::epoch`), or 0 when `func` is the stub. When the
    /// compile pool retires an arena it takes the code off every page that
    /// names it ([`Self::retire_code`]). Written only under `publish_lock`.
    code_epoch: AtomicU64,
    /// Generation this `func`/`compiled` pair was last published against
    /// (§13.3 step 6). Only ever written to a value `>=` its current value;
    /// numerically advances only when the publish that wrote it was actually
    /// anchored to newer page bytes than what was live before (the
    /// gen-bump case) — a publish that only adds entry-point coverage against
    /// unchanged bytes rewrites the same value. Readers never need to
    /// distinguish the two cases: `entry_gen == current_gen()` is the only
    /// fact dispatch needs (§13.2).
    entry_gen: AtomicU64,
    /// Page-level "a `CompileRequest` for this page is already in flight"
    /// flag (§13.2). Distinct from `requested`'s per-offset bits: those track
    /// *what* has been asked for and stay set until a compile actually covers
    /// them (§13.3's coalescing — many discoveries share one in-flight
    /// request), while this is purely a dedup gate on *sending* — cleared
    /// once `handle_request` has decided the page one way or another (some
    /// coverage published, or nothing new to offer), mirroring the old
    /// per-offset `ENTRY_SCHEDULED` flag's contract but page-scoped instead
    /// of offset-scoped, since one request now serves however many offsets
    /// have accumulated in `requested` by the time it's dequeued.
    page_scheduled: std::sync::atomic::AtomicBool,
    /// Arrivals at this page while its compile request was queued or refused
    /// by a full FIFO (`try_schedule_page` lost, or `drop_request`). Once it
    /// crosses [`HOT_PROMOTE_ARRIVALS`] the page is hot *now* and gets one
    /// request on the hot lane (`promoted` stops a second). Both reset whenever
    /// a compile for the page completes (`clear_scheduled`).
    hot_wait: std::sync::atomic::AtomicU32,
    promoted: std::sync::atomic::AtomicBool,
    /// Pinned FR mode for this page, set at page-claim time from live
    /// `STATUS_FR` and held until the next `mega_flush`/`reset_to_unclaimed`,
    /// or until a real invalidation or an FR-guard bail re-pins it (see
    /// "Re-pinned" below). `true` = FR1.
    ///
    /// **Why pinned, not per-compile** (as it was under the old one-function-
    /// per-entry design, where `CompileRequest::compiled_for_fr1` could
    /// legitimately differ compile to compile): §13's one-function-per-page
    /// model means every entry point on a page shares the SAME compiled
    /// function, and FR mode is baked into every FPR-access emitter in that
    /// function at compile time (§4.2.1) — there is nowhere to put a second
    /// version for a second FR mode. If two entries on one page genuinely
    /// wanted different FR modes, only one can ever be served; pinning at
    /// claim time picks whichever mode was live first and commits to it.
    ///
    /// **No dispatch-side check**: unlike an earlier draft of this field,
    /// `exec_decoded`'s gate does NOT compare live `STATUS_FR` against this
    /// value before calling `func` — that would tax every single dispatch
    /// through the hot gate, forever, on every page. Instead the existing
    /// in-function FR-mode guard (`emit_fr_mode_guard`) does the job,
    /// unchanged in its existing gating: only emitted when the compiled
    /// region actually contains a CP1 instruction (`has_fpu`,
    /// `compile_region_uncommitted`) — a page with no FPU usage anywhere in
    /// it has no FR-mode-dependent code to protect, so it pays nothing at
    /// all, dispatch-side or in-function. Where the guard IS present, a
    /// mismatch kills just the triggering entry (`jit_kill_entry`/
    /// `page.kill`) — a page whose workload keeps arriving in the wrong mode
    /// simply has its FPU-touching entries killed one at a time as each is
    /// dispatched, self-limiting to at most (published entry count) guard
    /// failures rather than one check per dispatch forever.
    ///
    /// **Re-pinned, as of the fix below.** The original TODO here said `fr1`
    /// was never re-pinned, and justified it as "measured at ~2 occurrences
    /// per real IRIX boot (rare)". That measurement was of a *different*
    /// scenario — a page whose entries all get killed — and said nothing
    /// about **page recycling**, which is the damaging case: a page freed by
    /// an o32 (FR0) process and reallocated to an n32 (FR1) one kept the old
    /// pin even though every byte on it was new. Real symptom: 4Dwm/fm/
    /// background core-dumping at session start (a burst of exactly that
    /// recycling), cleared by `j2 flush`. Two resets now cover it, and
    /// neither subsumes the other:
    ///
    /// 1. [`Self::publish`] clears the pin on a real invalidation (the guest
    ///    wrote new bytes), so the next compile re-derives from the live
    ///    mode. This sits beside the existing `compiled`/`denied` resets,
    ///    which already reason "the bytes changed, so what was decided
    ///    against the OLD bytes deserves a fresh chance" — `fr1` had
    ///    identical justification and was the one field left behind.
    /// 2. [`Self::request_fr_repin`] records the *demanded* mode when the
    ///    in-function FR guard bails, for the case (1) cannot see: the same
    ///    bytes arriving in a new mode (the shared `checkfp.s` page).
    ///
    /// Why the guard alone was never sufficient: `emit_fr_mode_guard`
    /// computes `band(cu1_set, ...)`, deliberately suppressing the FR check
    /// when CU1 is clear so it can't pre-empt the real per-instruction CU1
    /// fault. So a stale pin is *not* always caught, and wrong FR packing on
    /// `ldc1`/`lwc1` puts an FP value in the wrong register half — silent
    /// corruption, not a fault.
    ///
    /// Entries already published against the old mode are retired **lazily**:
    /// the per-entry guard kills them one at a time and `mega_flush` recycles
    /// the page eventually. A page-wide kill on re-pin would discard correct
    /// code — non-FP entries on the same page are mode-independent — to save
    /// a bounded number of guard failures.
    fr1: std::sync::atomic::AtomicBool,
    /// FR mode demanded by the most recent in-function FR-guard bail, or
    /// `None` if no bail is outstanding. Read and cleared by the next compile
    /// for this page, which re-pins [`Self::fr1`] to it.
    ///
    /// Stores the **demanded mode**, not a "flip me" bit, deliberately.
    /// Several bails normally land before the next compile — the guard kills
    /// a page's FPU-touching entries one at a time as each is dispatched, and
    /// each kill records a demand — so a flip bit would toggle once per
    /// killed entry and an even number of kills would settle back on the
    /// wrong mode. Recording the observed live FR is idempotent: N bails all
    /// demanding FR1 say "FR1" N times, and the recompile reads a value
    /// rather than applying a delta.
    ///
    /// Encoded as a `u8` rather than `Option<bool>` so it can live in an
    /// atomic: [`FR_REPIN_NONE`], [`FR_REPIN_FR0`], [`FR_REPIN_FR1`].
    fr_repin: std::sync::atomic::AtomicU8,
    /// Set by [`Self::publish`] on a real invalidation: the page's bytes were
    /// replaced, so the FR mode pinned against the OLD bytes carries no
    /// information about what the new ones want. The next compile re-derives
    /// [`Self::fr1`] from the live mode rather than keeping the stale pin.
    ///
    /// Separate from [`Self::fr_repin`] because the two answer different
    /// questions — `fr_repin` says "a bail proved the pin wrong, use THIS
    /// mode", while this says "the pin is merely unjustified, use whatever is
    /// live at the next compile". A bail's demand is the stronger signal and
    /// wins when both are set.
    fr_unpinned: std::sync::atomic::AtomicBool,
    /// Serializes the publish step (§13.3 step 6/§13.5) — the compare/union/
    /// swap of `func`+`compiled`+`entry_gen` together needs to happen as one
    /// step even though individual dispatch reads of those fields remain
    /// independently orderable. Does **not** guard the expensive work
    /// (snapshot/analyze/codegen, §13.3 steps 1-5) — compiles proceed
    /// unlocked and in parallel, including two compiles of the same page;
    /// only the cheap publish tail is serialized here.
    publish_lock: Mutex<()>,
    /// Page-granular compile diagnostics for `j2 pcp`/`j2 status` (§13.1 —
    /// page-granular now that there's one function; the old per-entry
    /// `instr_count`/`code_size`/`call_count`/`block_include_count` don't
    /// have a meaningful per-offset home anymore since overlap between
    /// entries is structurally impossible under this design).
    ///
    /// **`instr_count`/`code_size` are NOT `developer`-gated**, unlike the
    /// counters below. They are the emitted-code-size metric
    /// (`code_size_by_instr_count`, `j2 status`' bytes/instruction), and
    /// gating them made that metric available *only* in the build that
    /// invalidates it: `developer` forces `opt_level=none` **and** emits
    /// `emit_dev_trace_bp` — a 4-argument `call_indirect` per instruction,
    /// which is both large and an opaque clobber that kills the GPR
    /// forwarding the metric is usually being read to evaluate. Measured
    /// live, that build reports ~380 code bytes per guest instruction where
    /// the same pages compile to ~92 without `developer`: a 4x error, and
    /// the whole point of the number is to compare codegen changes.
    ///
    /// Cost of keeping them always-on: 8 bytes per pooled page (two u32),
    /// i.e. 32 KiB at the default 4096-page capacity, written once per
    /// successful `publish`. Nothing reads them on any hot path. See
    /// `clanker/jitv2/block-fragmentation-blocks-cse.md`'s "measure without
    /// `developer`" rule, which this makes possible to actually follow.
    pub instr_count: std::sync::atomic::AtomicU32,
    pub code_size: std::sync::atomic::AtomicU32,
    #[cfg(feature = "developer")]
    pub call_count: AtomicU64,
    /// How many times `publish()` has actually committed a fresh compile for
    /// this page — incremented once per successful `publish()` call (the two
    /// early `return false`s, stale-generation and subsumed-by-existing-
    /// coverage, don't count: nothing new was compiled in). `j2 html`'s
    /// per-page stats use this to show recompile churn (a page compiled 40
    /// times is either very hot or thrashing on a denylist/invalidation
    /// loop — both worth seeing at a glance).
    #[cfg(feature = "developer")]
    pub compile_count: std::sync::atomic::AtomicU32,
    /// Same trigger as `compile_count` (bumped once per successful
    /// `publish()`), but with the opposite lifetime and always present, not
    /// `developer`-gated: `compile_count` is a lifetime-total churn counter
    /// deliberately preserved across a page's reset (see that field's own
    /// doc comment), whereas this one is cleared by `reset_to_unclaimed` —
    /// so it answers
    /// "has this page compiled at all since its last flush/claim" directly,
    /// without needing a `developer` build or reasoning through generation
    /// numbers to rule out "never even tried" vs "tried and is legitimately
    /// stuck." `j2 pcp` prints it unconditionally for exactly that reason —
    /// a page reporting `gen` in the thousands with `compiles_since_flush=0`
    /// is a real red flag (scheduling/dispatch never reached it, or every
    /// attempt bailed before `publish`), not just "hasn't gotten hot yet."
    pub compiles_since_flush: std::sync::atomic::AtomicU32,
    /// How many times `prepare_multi_entry_compile`/`handle_request(_deferred)`
    /// (`comp.rs`) has REJECTED a compile attempt for this page — denylisted
    /// (per-entry codegen exclusion, or a whole-region decline), too-short,
    /// or out-of-memory-retry — as opposed to a successful `publish()`
    /// (tracked by `compiles_since_flush`/`compile_count` instead). Always
    /// present, not `developer`-gated, and NOT cleared by any reset path
    /// (lifetime total, same reasoning as `compile_count`'s own doc comment
    /// on why churn history outlives a flush the page survived) — the point
    /// is distinguishing "this page has been attempted repeatedly and keeps
    /// getting rejected" from "this page has never been attempted at all"
    /// (`compiles_since_flush=0` alone can't tell those apart), directly in
    /// `j2 pcp` without needing a `developer` build's `j2 status` rejection
    /// breakdown.
    pub rejected_compiles: std::sync::atomic::AtomicU32,
    /// The `rejected_compiles` subset decided during the analyze stage
    /// (`prepare_multi_entry_compile`, `comp.rs`): a candidate excluded at
    /// its own entry during the walk (`EntryExcluded`), or the whole merged
    /// region coming in too short (`TooShort`). Distinct from
    /// `codegen_rejected` — this page's request never even reached
    /// Cranelift. Always present, not `developer`-gated, lifetime total.
    pub analyze_rejected: std::sync::atomic::AtomicU32,
    /// The `rejected_compiles` subset decided during the codegen stage
    /// (`handle_request`/`handle_request_deferred`'s non-OOM `None =>` arm,
    /// `comp.rs`): Cranelift itself declined — no emitter for some visited
    /// instruction, or a real verifier rejection. Always present, not
    /// `developer`-gated, lifetime total.
    pub codegen_rejected: std::sync::atomic::AtomicU32,
    /// How many times a dispatch has REACHED the `try_schedule_page()`
    /// test-and-set for this page (`exec_decoded`'s dispatch gate,
    /// `mips_exec.rs`) — every such dispatch, win or lose, not just the one
    /// that actually goes on to send (that subset is `sends_attempted`
    /// instead). `schedule_attempts == 0` on a page with real `requested`
    /// bits means dispatch never even reached this branch for it at all — a
    /// different, earlier problem than anything downstream of the
    /// test-and-set. `schedule_attempts` vs `sends_attempted` shows the
    /// attempt/win ratio directly (a hot page loses the race often, a cold
    /// one wins almost every time). Lifetime total, not cleared by any
    /// reset. `developer`-gated: this fires on EVERY dispatch that reaches
    /// the trigger gate, win or lose — by far the hottest of the three
    /// dispatch-side counters, not worth an atomic increment on the CPU's
    /// hot path in a normal/`lightning` build (the root cause this whole
    /// family of counters was built to chase turned out to be in the arena
    /// allocator, not dispatch scheduling — see `codegen_oom_bounced`'s
    /// sibling comment history).
    #[cfg(feature = "developer")]
    pub schedule_attempts: std::sync::atomic::AtomicU32,
    /// How many times a `CompileRequest` for this page was actually handed to
    /// `push_compile_request` (`mips_exec.rs`'s dispatch gate) — i.e. this
    /// dispatch won `try_schedule_page()` AND the send call was reached.
    /// Should track `schedule_attempts` 1:1 today (nothing currently sits
    /// between winning the flag and calling send) — kept as its own counter
    /// rather than assumed equal, so a future divergence between the two is
    /// itself a diagnostic signal. Lifetime total. `developer`-gated — see
    /// `schedule_attempts`'s own doc comment for why the whole dispatch-side
    /// family moved behind it.
    #[cfg(feature = "developer")]
    pub sends_attempted: std::sync::atomic::AtomicU32,
    /// How many times `push_compile_request` returned `false` for this page
    /// — the bounded compile queue was full at that exact moment, so the
    /// request was dropped before ever reaching a worker thread (never
    /// `handle_request`/`handle_request_deferred`, so neither `compiles_since_flush`
    /// nor `rejected_compiles` ever sees it). The dispatch gate clears
    /// `page_scheduled` itself when this happens (see that call site's own
    /// comment) so the page isn't permanently starved, but the compile
    /// itself is still lost — this counter is what makes that loss visible
    /// in `j2 pcp` instead of silently returning to zero. Lifetime total.
    /// `developer`-gated — see `schedule_attempts`'s own doc comment for why
    /// the whole dispatch-side family moved behind it.
    #[cfg(feature = "developer")]
    pub sends_dropped_queue_full: std::sync::atomic::AtomicU32,
    /// How many times `prepare_multi_entry_compile` (`comp.rs`) bailed out
    /// via one of its three silent, non-denylisting early returns for this
    /// page: a `bus.read32` failure mid-snapshot, the 8-retry seqlock never
    /// stabilizing (`current_gen()` kept changing under it), or the
    /// subsumption check finding nothing new to offer. None of these touch
    /// `rejected_compiles` (nothing was actually rejected — no offset was
    /// denylisted, no codegen ran) yet they still consume the one
    /// `CompileRequest` this dispatch's `try_schedule_page()` win produced,
    /// same as any other outcome — so a page could show `sends_attempted`
    /// far above `rejected_compiles + compiles_since_flush` with this being
    /// the reason why: NOT a queue drop, but every attempt individually
    /// losing the seqlock race (or hitting a transient bus-read failure)
    /// before ever reaching the actual compiler. A sufficiently hot
    /// self-modifying/shared-DMA page bumping `gen` faster than 8 retries
    /// can outrun is the realistic way this happens repeatedly, not a one-off.
    /// Always present, not `developer`-gated, lifetime total (not cleared by
    /// any reset — same reasoning as `rejected_compiles`).
    pub prepare_bounced: std::sync::atomic::AtomicU32,
    /// How many times `Codegen::compile_region_uncommitted` returned `None`
    /// specifically because `last_compile_ran_out_of_memory()` was true for
    /// this page's compile — the shared Cranelift arena was full at that
    /// exact moment, NOT a real decline (no emitter gap, no verifier
    /// rejection). Distinct from `rejected_compiles`: the caller
    /// deliberately does NOT denylist anything here (see that call site's
    /// own comment — "leave everything un-denylisted so it retries
    /// naturally on a later arrival") and instead triggers a pool-wide arena
    /// flush. If this page's compiled region is large/expensive enough that
    /// its turn in the queue keeps landing exactly when the arena is
    /// already near its flush threshold, EVERY attempt can bounce here,
    /// forever, with nothing else this page shows (`rejected_compiles`,
    /// `prepare_bounced`, `sends_dropped_queue_full` all staying 0) hinting
    /// at why — this counter is what finally makes that visible. Always
    /// present, not `developer`-gated, lifetime total (not cleared by any
    /// reset).
    pub codegen_oom_bounced: std::sync::atomic::AtomicU32,
    /// How many times this page's compile fully succeeded in codegen
    /// (`compile_region_uncommitted` returned a real `FuncId`) but
    /// `finalize_batch_nonforced` came back empty — gap-blocked behind an
    /// earlier, still-unpatched entry in the shared arena's strictly-ordered
    /// seal queue (`paged_memory.rs`'s bump allocator only ever returns a
    /// contiguous prefix). Not a loss by itself — the idle-timeout/
    /// pending-threshold sweep is supposed to eventually force-seal it — but
    /// a page whose `codegen_oom_bounced`/this counter both climb while
    /// `compiles_since_flush` stays 0 is a real, visible red flag that
    /// something downstream of a successful compile keeps consuming this
    /// page's request without ever actually publishing it. Always present,
    /// not `developer`-gated, lifetime total (not cleared by any reset).
    pub seal_gap_blocked: std::sync::atomic::AtomicU32,
    /// How many times this page's `finalize_batch_nonforced` call came back
    /// empty because `module.finalize_definitions()` itself errored — a
    /// real Cranelift/JIT-backend failure, NOT a normal gap-blocked seal
    /// (that's `seal_gap_blocked`, which self-resolves once the earlier
    /// entry closes the gap). This one never will: confirmed as a real bug
    /// — `handle_request_deferred` couldn't tell the two apart before
    /// `Codegen::last_finalize_failed` existed, so a page hitting this kept
    /// getting bumped into `pending` and waiting on a sweep to "unblock" an
    /// entry that was never blocked, just permanently failed. Always
    /// present, not `developer`-gated, lifetime total (not cleared by any
    /// reset).
    pub finalize_failed: std::sync::atomic::AtomicU32,
    /// How many compile requests for this page were answered entirely by the
    /// churn-avoidance snapshot — no walk, no codegen, no finalize, just an
    /// `entry_gen` re-validation of the code already installed. See
    /// [`Self::try_skip_redundant_compile`]; `Jitv2::redundant_compile_totals`
    /// sums these across the pool for the process-wide ratio.
    ///
    /// Lives per-page rather than on `JitStats` because a page is the only
    /// thing the compile path has in hand at the point of the decision
    /// (`prepare_multi_entry_compile` has no `JitStats` outside a
    /// `developer` build), and because the per-page breakdown is the more
    /// useful reading anyway — "which pages churn" rather than just "how
    /// much".
    ///
    /// A page with a high count here is one whose generation churns for
    /// reasons unrelated to its code — data sharing the page, or DMA into
    /// it — and is exactly the case this cache was built for. Always
    /// present, not `developer`-gated.
    ///
    /// **Per-flush-epoch, not a lifetime total** — cleared by
    /// `reset_to_unclaimed`, unlike
    /// `rejected_compiles`/`prepare_bounced` next door. A boot's exec storm
    /// racks up hundreds of thousands of skips (measured: ~286k, 98% of
    /// everything checked), and as a lifetime total that figure would sit in
    /// `j2 status` forever, drowning out whatever the system is doing now.
    /// The flush is the honest epoch boundary anyway: it invalidates the very
    /// snapshots these counted, so the old tallies describe a state that no
    /// longer exists.
    pub redundant_skipped: std::sync::atomic::AtomicU32,
    /// The complement of [`Self::redundant_skipped`] for this page: requests
    /// that had a snapshot to compare against and still had to compile. A
    /// page where this dominates is genuinely being rewritten (or flipping
    /// FR mode), not merely churning generations.
    pub redundant_rejected: std::sync::atomic::AtomicU32,
    /// Churn-avoidance cache: what the last successful compile of this page
    /// actually looked at, kept so a later request whose only difference is
    /// a bumped generation can be answered without recompiling at all. See
    /// [`CompileSnapshot`] for the contents and
    /// [`Self::try_skip_redundant_compile`] for the comparison rule.
    ///
    /// Inline rather than `Option<Box<..>>` deliberately, matching the same
    /// call made for `entries` under the old design: the pool array is
    /// allocated once up front, so a boxed payload would buy nothing but a
    /// pointer-chase and an allocation on every page's first compile, while
    /// the flat version's cost is bounded, predictable, and mostly untouched
    /// zero pages for slots that never compile.
    ///
    /// Set by [`Self::kill`], cleared by [`Self::stage_compile_snapshot`]:
    /// "some entry was killed since the snapshot below was taken."
    ///
    /// A kill means "stop dispatching into `func` here", so a skip must not
    /// re-advertise the killed offset. `compiled` cannot answer that question
    /// on its own, because a narrowing skip clears bits there for an entirely
    /// unrelated reason — hence a separate record. A counter rather than a
    /// flag: each snapshot notes the count when it was staged, and refuses a
    /// skip once any kill has happened since, so a snapshot kept past a newer
    /// compile's staging (see `snapshots`) still sees every kill after its
    /// own. Page-granular because the skip is all-or-nothing anyway.
    kills: std::sync::atomic::AtomicU32,
    /// The churn-avoidance record: `committed` describes the `func` that is
    /// installed right now, `staged` the compile in flight, if any.
    ///
    /// Two slots because one lost the record whenever it was needed most. A
    /// compile staged over the only slot (invalidating it) at compile time
    /// and re-validated it at publish time, but on the deferred path the
    /// publish waits in the seal queue after `page_scheduled` is released, and
    /// a publish can also fail outright (stale generation, subsumed). In both
    /// cases the page kept working code that no record vouched for, so every
    /// request in between recompiled identical bytes: measured on an IP28
    /// session as 1,338 of 6,500 compiles (`jitv2::hashstats`).
    ///
    /// A mutex because that same window lets a commit (in `publish_all`, on
    /// any worker) run while another worker stages or skips for this page.
    /// Only the compile path takes it; dispatch never does.
    snapshots: Mutex<Snapshots>,
    /// Diagnostics only (`jitv2::hashstats`): why the last
    /// `try_skip_redundant_compile` refused. 0 = it did not, 1 = no valid
    /// snapshot or FR mismatch, 2 = no installed func, 3 = entries not in the
    /// last compile's set, 4 = a used word differs, 5 = gen moved, 6 =
    /// entry_gen ahead, 7 = killed since the snapshot.
    pub last_skip_reject: std::sync::atomic::AtomicU8,
}

/// What a successful compile of a page looked at, recorded so the next
/// compile request for the same page can prove it would produce identical
/// code and skip the work (§ churn reduction).
///
/// The whole 4KB is stored, not a hash: the comparison must be exact (a hash
/// collision here would publish code for bytes that no longer exist), and at
/// this size a `memcmp` of the used words is cheaper than re-reading and
/// re-walking the page anyway. The `used` mask is what makes the comparison
/// *useful* rather than merely correct — a page where the guest rewrote a
/// data constant, or any word the walk never reached, still matches, because
/// only words the compile actually consumed are compared.
struct Snapshots {
    committed: CompileSnapshot,
    staged: Option<Box<CompileSnapshot>>,
}

struct CompileSnapshot {
    /// Whether the fields below describe a real, still-installed compile.
    /// False on a fresh page and after any reset that drops `func` — a
    /// snapshot without the function it describes must never authorize a
    /// skip, since there would be nothing left to re-publish.
    ///
    /// A record starts life with this `false` ([`PhysicalCodePage::stage_compile_snapshot`])
    /// and only becomes `true` once the publish it describes actually lands
    /// ([`PhysicalCodePage::commit_compile_snapshot`]). The deferred compile
    /// path (`comp.rs`'s `handle_request_deferred`) needs that split: it
    /// hands its compile to the shared arena's seal queue and the matching
    /// publish can happen later, on another worker's `finalize` call, with
    /// only a `PublishInfo` in hand — far too late and too small to carry a
    /// 4KB page snapshot through. Staging at compile time and committing at
    /// publish time keeps the 4KB on the page it belongs to while still
    /// guaranteeing a snapshot is never consultable before the code it
    /// describes is installed.
    valid: bool,
    /// The generation the staged record was compiled against, used by
    /// [`PhysicalCodePage::commit_compile_snapshot`] to confirm the publish
    /// it is being asked to vouch for is the compile that staged this record.
    /// The case this guards is a reset landing under an in-flight compile:
    /// [`PhysicalCodePage::invalidate_compile_snapshot`] poisons this to
    /// `u64::MAX`, which no real generation can equal, so the compile's later
    /// commit finds no match and the record stays dead instead of vouching
    /// for a function the reset already discarded.
    staged_gen: u64,
    /// The page bytes as the recorded compile saw them (its seqlock
    /// snapshot, `prepare_multi_entry_compile` step 2). Only the words
    /// selected by `used` are ever compared.
    words: [u32; ENTRIES_PER_PAGE],
    /// Words the recorded compile actually decoded — every instruction the
    /// merged multi-entry walk visited (`instrs_linear`), including delay
    /// slots. A word outside this set had no influence on the emitted code,
    /// so a change to it cannot change what a recompile would produce.
    used: [u64; BITMAP_WORDS],
    /// Entry points the recorded compile published (`new_entries` as handed
    /// to `publish`). A new request asking for an entry point outside this
    /// set needs a real compile: the published `func`'s dispatch switch has
    /// no case for it, so no amount of byte equality helps.
    entries: [u64; BITMAP_WORDS],
    /// FR mode the recorded compile was specialized for
    /// (`CompileRequest::compiled_for_fr1`). Compared alongside the bytes,
    /// and NOT optional: FR mode is baked into every FPR-access emitter at
    /// compile time (§4.2.1) and is not visible anywhere in the page bytes,
    /// so identical bytes compiled for the other mode are genuinely
    /// different code. Skipping on a mismatch would leave the old mode's
    /// `func` installed while the requester expects the new one — and
    /// because `emit_fr_mode_guard` suppresses itself when CU1 is clear (see
    /// [`PhysicalCodePage::fr1`]'s doc comment), the in-function guard is not
    /// a reliable backstop: the failure mode is `ldc1`/`lwc1` packing an FP
    /// value into the wrong register half, which is silent corruption rather
    /// than a fault. A page whose FR pin legitimately flips therefore always
    /// takes a real recompile, exactly as it did before this cache existed.
    fr1: bool,
    /// `PhysicalCodePage::kills` when this record was staged. A skip is
    /// refused once the count has moved.
    kills_at_stage: u32,
}

impl CompileSnapshot {
    fn new() -> Self {
        Self {
            valid: false,
            staged_gen: u64::MAX,
            words: [0u32; ENTRIES_PER_PAGE],
            used: [0u64; BITMAP_WORDS],
            entries: [0u64; BITMAP_WORDS],
            fr1: false,
            kills_at_stage: 0,
        }
    }
}

// Safety: `gen` points into the owning BusDevice's storage, which outlives
// every PhysicalCodePage derived from it (devices are held in Arcs/statics
// for the lifetime of the machine). The pointee is only ever read or
// atomically incremented, never moved or freed while a PhysicalCodePage
// referencing it exists. `func`, when non-null, points to finalized
// JIT-compiled code owned by the compile-thread arena, which outlives every
// PhysicalCodePage referencing it until the next mega_flush.
//
// `compile_snapshot`'s UnsafeCell is what makes the Sync impl load-bearing
// rather than merely a formality for the raw `gen` pointer: it is genuinely
// unsynchronized data behind `&self`. Sound because a page is never compiled
// twice at once and nothing but a compile (or a reset, mutually exclusive
// with one) ever touches it — see that field's own doc comment for the
// `page_scheduled` argument.
unsafe impl Send for PhysicalCodePage {}
unsafe impl Sync for PhysicalCodePage {}

impl PhysicalCodePage {
    /// Construct an as-yet-unclaimed page descriptor: `pfn = 0`, every
    /// bitmap zeroed, no compiled function. Used both to build `Jitv2::pages`'
    /// full-capacity array up front (every slot starts unclaimed) and,
    /// functionally identically, by [`Self::claim`]/[`Self::reset_to_unclaimed`]
    /// to return a slot to this same state in place without reallocating
    /// anything.
    ///
    /// `gen` null is accepted here (callers pass it for an as-yet-unclaimed
    /// slot, or when the real device has no gen tracking) and normalized to
    /// `NEVER_COMPILABLE_GEN` — `self.gen` itself is never null, so nothing
    /// downstream ever needs to check.
    ///
    /// A `pfn` other than [`UNCLAIMED_PFN`] makes a page on its own, outside
    /// a pool (tests): frame `pfn` of a flat bus, reached there by CPU 0.
    pub fn new(pfn: Pfn, gen: *const AtomicU64) -> Self {
        let claimed = pfn != UNCLAIMED_PFN;
        Self {
            key: if claimed { CodeKey::flat(pfn as u64 * PAGE_SIZE as u64) } else { CodeKey::NONE },
            addr: std::array::from_fn(|cpu| AtomicU32::new(if claimed && cpu == 0 { pfn << 1 } else { NO_ADDR })),
            words: std::ptr::null(),
            next: NO_SLOT,
            gen: if gen.is_null() { &NEVER_COMPILABLE_GEN } else { gen },
            requested: new_bitmap(),
            denied: new_bitmap_all_set(),
            compiled: new_bitmap(),
            func: std::sync::atomic::AtomicPtr::new(NO_CODE as *mut ()),
            func_fr1: std::sync::atomic::AtomicBool::new(false),
            tenancy: AtomicU32::new(0),
            referenced: std::sync::atomic::AtomicBool::new(false),
            code_epoch: AtomicU64::new(0),
            entry_gen: AtomicU64::new(0),
            page_scheduled: std::sync::atomic::AtomicBool::new(false),
            hot_wait: std::sync::atomic::AtomicU32::new(0),
            promoted: std::sync::atomic::AtomicBool::new(false),
            fr1: std::sync::atomic::AtomicBool::new(false),
            fr_repin: std::sync::atomic::AtomicU8::new(FR_REPIN_NONE),
            fr_unpinned: std::sync::atomic::AtomicBool::new(false),
            publish_lock: Mutex::new(()),
            redundant_skipped: std::sync::atomic::AtomicU32::new(0),
            redundant_rejected: std::sync::atomic::AtomicU32::new(0),
            kills: std::sync::atomic::AtomicU32::new(0),
            snapshots: Mutex::new(Snapshots { committed: CompileSnapshot::new(), staged: None }),
            last_skip_reject: std::sync::atomic::AtomicU8::new(0),
            instr_count: std::sync::atomic::AtomicU32::new(0),
            code_size: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            call_count: AtomicU64::new(0),
            #[cfg(feature = "developer")]
            compile_count: std::sync::atomic::AtomicU32::new(0),
            compiles_since_flush: std::sync::atomic::AtomicU32::new(0),
            rejected_compiles: std::sync::atomic::AtomicU32::new(0),
            analyze_rejected: std::sync::atomic::AtomicU32::new(0),
            codegen_rejected: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            schedule_attempts: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            sends_attempted: std::sync::atomic::AtomicU32::new(0),
            #[cfg(feature = "developer")]
            sends_dropped_queue_full: std::sync::atomic::AtomicU32::new(0),
            prepare_bounced: std::sync::atomic::AtomicU32::new(0),
            codegen_oom_bounced: std::sync::atomic::AtomicU32::new(0),
            seal_gap_blocked: std::sync::atomic::AtomicU32::new(0),
            finalize_failed: std::sync::atomic::AtomicU32::new(0),
        }
    }

    /// Zero every bitmap and drop the compiled function, in place — no
    /// reallocation. Called only from [`Self::reset_to_unclaimed`]
    /// (`Jitv2::mega_flush`'s per-slot reset) — a fresh, never-claimed slot
    /// is already zeroed by `PhysicalCodePage::new` and doesn't need this.
    fn reset_entries_and_bitmaps(&mut self) {
        for bm in [&self.requested, &self.compiled] {
            for word in bm.iter() { word.store(0, Ordering::Relaxed); }
        }
        // `denied` resets to all-ones, not zero — see its own field doc
        // (inverted sense: 0 means "still eligible" would make every reset
        // page permanently denylisted everywhere).
        for word in self.denied.iter() { word.store(u64::MAX, Ordering::Relaxed); }
        self.func.store(NO_CODE as *mut (), Ordering::Relaxed);
        self.func_fr1.store(false, Ordering::Relaxed);
        self.code_epoch.store(0, Ordering::Relaxed);
        // The snapshot describes the `func` just dropped — see
        // `invalidate_compile_snapshot`.
        self.invalidate_compile_snapshot();
        self.entry_gen.store(0, Ordering::Relaxed);
        self.page_scheduled.store(false, Ordering::Relaxed);
        self.hot_wait.store(0, Ordering::Relaxed);
        self.promoted.store(false, Ordering::Relaxed);
        self.compiles_since_flush.store(0, Ordering::Relaxed);
        // Churn-avoidance tallies are per-flush-epoch (see the
        // `redundant_skipped` field), and this slot is additionally about to
        // be handed to a different physical page entirely — carrying the
        // previous tenant's counts into it would misattribute them.
        self.redundant_skipped.store(0, Ordering::Relaxed);
        self.redundant_rejected.store(0, Ordering::Relaxed);
        self.instr_count.store(0, Ordering::Relaxed);
        self.code_size.store(0, Ordering::Relaxed);
        #[cfg(feature = "developer")]
        {
            self.call_count.store(0, Ordering::Relaxed);
            self.compile_count.store(0, Ordering::Relaxed);
        }
    }

    /// Bump-allocate this (already-constructed, already-clean) slot into
    /// service for a newly-arrived physical page — the in-place counterpart
    /// to what used to be a fresh `PhysicalCodePage::new(pfn, gen)` +
    /// `Vec::push`. Deliberately does **not** reset bitmaps itself — see
    /// `reset_entries_and_bitmaps`'s call sites for why a slot only ever
    /// reaches `claim` already clean.
    ///
    /// `fr1`: live `STATUS_FR` at the moment of first arrival. Pinned, but no
    /// longer for the page's whole lifetime — it is re-derived on a real
    /// invalidation and on an FR-guard bail (see `Self::fr1`'s doc comment).
    #[inline]
    pub fn is_claimed(&self) -> bool {
        self.key != CodeKey::NONE
    }

    /// Where CPU `cpu` of the pool reaches this page: its frame, and whether
    /// through an uncached fetch. `None` if it has not, or not since a remap.
    #[inline]
    pub fn addr_of(&self, cpu: usize) -> Option<(Pfn, bool)> {
        match self.addr.get(cpu)?.load(Ordering::Relaxed) {
            NO_ADDR => None,
            a => Some((a >> 1, a & 1 != 0)),
        }
    }

    /// CPU `cpu` reached this page at frame `pfn`.
    #[inline]
    pub fn set_addr(&self, cpu: usize, pfn: Pfn, uncached: bool) {
        self.addr[cpu].store(pfn << 1 | uncached as u32, Ordering::Relaxed);
    }

    /// CPU `cpu`'s address for this page no longer holds (a remap).
    #[inline]
    pub fn forget_addr(&self, cpu: usize) {
        self.addr[cpu].store(NO_ADDR, Ordering::Relaxed);
    }

    /// A frame some CPU reaches this page at, for display: the lowest-numbered
    /// CPU's that has one, else [`UNCLAIMED_PFN`]. CPUs of different nodes may
    /// each have their own.
    pub fn pfn(&self) -> Pfn {
        (0..POOL_CPUS).find_map(|cpu| self.addr_of(cpu)).map_or(UNCLAIMED_PFN, |(pfn, _)| pfn)
    }

    /// The page's words in its bank's storage, or null: see the field.
    #[inline]
    pub fn words(&self) -> *const u32 {
        self.words
    }

    /// [`Self::claim`] for the page a bus answered with.
    pub fn claim_code(&mut self, code: crate::traits::CodePage, fr1: bool) {
        self.claim(code.key, code.gen, fr1);
        self.words = code.words;
    }

    pub fn claim(&mut self, key: CodeKey, gen: *const AtomicU64, fr1: bool) {
        debug_assert!(std::ptr::eq(self.gen, &NEVER_COMPILABLE_GEN) && !self.is_claimed(),
            "claim() called on a slot that wasn't clean (key={:?}) — every path that reuses a slot must reset it first (see reset_to_unclaimed)",
            self.key);
        debug_assert!(!self.has_code(),
            "claim() called on a slot with a still-published function — mega_flush must reset_to_unclaimed before this slot can be reused");
        self.key = key;
        self.gen = if gen.is_null() { &NEVER_COMPILABLE_GEN } else { gen };
        self.fr1.store(fr1, Ordering::Relaxed);
        // A freshly-claimed slot has no bail outstanding: `fr1` was just set
        // from the live mode, so there is nothing to correct.
        self.fr_repin.store(FR_REPIN_NONE, Ordering::Relaxed);
        self.fr_unpinned.store(false, Ordering::Relaxed);
    }

    /// This page's pinned FR mode (§13, see `Self::fr1`'s own doc comment).
    #[inline]
    pub fn is_fr1(&self) -> bool {
        self.fr1.load(Ordering::Relaxed)
    }

    /// Record that an in-function FR-guard bail observed live FR = `fr1` and
    /// that this page's pin should be corrected to it at the next compile.
    /// Called from `jit_kill_entry` (mips_exec.rs), which already holds both
    /// the executor (for live `cp0_status`) and this page.
    ///
    /// Idempotent by construction — it stores the demanded mode, so repeated
    /// bails demanding the same mode are indistinguishable from one. Last
    /// writer wins, which is correct: the most recent observation of the live
    /// mode is the best available estimate of what the next compile should
    /// target.
    #[inline]
    pub fn request_fr_repin(&self, fr1: bool) {
        self.fr_repin.store(if fr1 { FR_REPIN_FR1 } else { FR_REPIN_FR0 }, Ordering::Relaxed);
    }

    /// Consume any outstanding FR re-pin request, applying it to `Self::fr1`
    /// and returning the mode the next compile should target.
    ///
    /// Takes `live_fr1` so a page with no request outstanding keeps its
    /// existing pin rather than tracking whatever mode happens to be live at
    /// compile time — a page is normally dispatched in its pinned mode, and
    /// re-deriving unconditionally would let an unrelated dispatch re-pin a
    /// page that was never mismatched.
    ///
    /// The read-only fast path is the steady state: nothing is outstanding
    /// and the page is pinned, so the mode is simply the pinned one. Both
    /// swaps below wrote shared state on every arrival, which ping-ponged
    /// their cache lines with the compile worker; a re-pin requested between
    /// the fast path's load and its return is not lost — it stays outstanding
    /// and the next arrival consumes it.
    #[inline]
    pub fn take_fr_repin(&self, live_fr1: bool) -> bool {
        if self.fr_repin.load(Ordering::Relaxed) == FR_REPIN_NONE
            && !self.fr_unpinned.load(Ordering::Relaxed)
        {
            return self.fr1.load(Ordering::Relaxed);
        }
        match self.fr_repin.swap(FR_REPIN_NONE, Ordering::Relaxed) {
            FR_REPIN_FR0 => { self.fr1.store(false, Ordering::Relaxed); false }
            FR_REPIN_FR1 => { self.fr1.store(true, Ordering::Relaxed); true }
            // No bail outstanding. If `publish` cleared the pin on a real
            // invalidation the page is unpinned and takes the live mode;
            // otherwise it keeps what it has.
            _ => {
                if self.fr_unpinned.swap(false, Ordering::Relaxed) {
                    self.fr1.store(live_fr1, Ordering::Relaxed);
                    live_fr1
                } else {
                    self.fr1.load(Ordering::Relaxed)
                }
            }
        }
    }

    /// Return this slot to the fully-unclaimed state (`pfn = UNCLAIMED_PFN`,
    /// `gen` pointed at the shared `NEVER_COMPILABLE_GEN` fallback) —
    /// `Jitv2::mega_flush`'s in-place counterpart to what used to be
    /// `Vec::clear()` dropping every `PhysicalCodePage` outright. `pfn` is
    /// poisoned to [`UNCLAIMED_PFN`], not `0` (see that constant's own doc
    /// comment) — a dangling raw pointer or leftover frame-table entry that
    /// reads this slot after it's been evicted and reused reads back an
    /// obviously-bogus pfn instead of silently aliasing page 0's real data.
    pub fn reset_to_unclaimed(&mut self) {
        self.reset_entries_and_bitmaps();
        self.key = CodeKey::NONE;
        self.words = std::ptr::null();
        for a in &self.addr {
            a.store(NO_ADDR, Ordering::Relaxed);
        }
        self.gen = &NEVER_COMPILABLE_GEN;
    }

    /// Reset `requested`/`denied`/`compiled` bitmaps, the compiled `func`
    /// pointer, and every diagnostic counter back to a freshly-claimed
    /// page's defaults — but, unlike [`Self::reset_to_unclaimed`], leaves
    /// `pfn`/`gen` untouched: this page stays claimed under the same
    /// physical frame, it just looks like nothing has ever been compiled for
    /// it yet. `requested`/`denied` are cleared too — this is a deliberate
    /// "forget everything you learned about this page" for
    /// `j2 clear <paddr>`.
    pub fn reset_compiled_state(&mut self) {
        self.reset_entries_and_bitmaps();
    }

    /// Current generation count for this page.    /// Current generation count for this page. `self.gen` is never null (see
    /// its own doc comment) — a page whose backing device has no real gen
    /// tracking (MMIO, etc) reads the shared, never-bumped
    /// `NEVER_COMPILABLE_GEN` fallback here instead, so this is
    /// unconditionally safe to call on any claimed or unclaimed page, no
    /// branch needed on the caller's part.
    #[inline]
    pub fn current_gen(&self) -> u64 {
        // Relaxed: publish-time (§13.3) and mutation-time (§7) orderings are
        // established by the fetch_or/re-read pair at those call sites, not here.
        unsafe { (*self.gen).load(Ordering::Relaxed) }
    }

    /// Relaxed read of `entry_gen` — the generation `func`/`compiled` were
    /// last published against. Used both by `j2 pcp` diagnostics and by
    /// `comp.rs`'s pre-compile subsumption check (§13.3 step 4 — comparing
    /// against this is what makes that check gate on "compiled for THIS
    /// generation" rather than blindly re-reading `compiled`'s bits, which
    /// can hold stale-generation leftovers, see `publish`'s own doc
    /// comment). Not part of the dispatch-hot path — `is_runnable` does
    /// its own Acquire load inline rather than calling this.
    #[inline]
    pub fn entry_gen(&self) -> u64 {
        self.entry_gen.load(Ordering::Relaxed)
    }

    /// How many times `publish()` has succeeded since this page's last
    /// `reset_to_unclaimed` — see the field's own
    /// doc comment for why this exists alongside `compile_count` (opposite
    /// lifetime, always present). Not on the hot dispatch path — only read
    /// by `j2 pcp`.
    #[inline]
    pub fn compiles_since_flush(&self) -> u32 {
        self.compiles_since_flush.load(Ordering::Relaxed)
    }

    /// Record one analyze-stage rejected compile attempt for this page (a
    /// candidate excluded at its own entry, or the merged region too short)
    /// — bumps both the stage-specific `analyze_rejected` and the umbrella
    /// `rejected_compiles` total. See both fields' own doc comments. Called
    /// from `comp.rs`'s `prepare_multi_entry_compile`.
    #[inline]
    pub fn mark_analyze_rejected(&self) {
        self.analyze_rejected.fetch_add(1, Ordering::Relaxed);
        self.rejected_compiles.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn analyze_rejected(&self) -> u32 {
        self.analyze_rejected.load(Ordering::Relaxed)
    }

    /// Record one codegen-stage rejected compile attempt for this page
    /// (Cranelift declined: no emitter, or a real verifier rejection) —
    /// bumps both the stage-specific `codegen_rejected` and the umbrella
    /// `rejected_compiles` total. See both fields' own doc comments. Called
    /// from `comp.rs`'s `handle_request`/`handle_request_deferred`'s non-OOM
    /// `None =>` arm.
    #[inline]
    pub fn mark_codegen_rejected(&self) {
        self.codegen_rejected.fetch_add(1, Ordering::Relaxed);
        self.rejected_compiles.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn codegen_rejected(&self) -> u32 {
        self.codegen_rejected.load(Ordering::Relaxed)
    }

    /// Lifetime total of every rejected compile attempt for this page,
    /// across both the analyze and codegen stages — see
    /// `rejected_compiles`'s own field doc comment. Not on the hot dispatch
    /// path — only read by `j2 pcp`.
    #[inline]
    pub fn rejected_compiles(&self) -> u32 {
        self.rejected_compiles.load(Ordering::Relaxed)
    }

    /// Record that a dispatch just won `try_schedule_page()` for this page
    /// — see `schedule_attempts`'s own field doc comment.
    #[cfg(feature = "developer")]
    #[inline]
    pub fn mark_schedule_attempt(&self) {
        self.schedule_attempts.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "developer")]
    #[inline]
    pub fn schedule_attempts(&self) -> u32 {
        self.schedule_attempts.load(Ordering::Relaxed)
    }

    /// Record that a `CompileRequest` for this page was actually handed to
    /// `push_compile_request` — see `sends_attempted`'s own field doc comment.
    #[cfg(feature = "developer")]
    #[inline]
    pub fn mark_send_attempted(&self) {
        self.sends_attempted.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "developer")]
    #[inline]
    pub fn sends_attempted(&self) -> u32 {
        self.sends_attempted.load(Ordering::Relaxed)
    }

    /// Record that a send for this page was dropped because the compile
    /// queue was full — see `sends_dropped_queue_full`'s own field doc
    /// comment.
    #[cfg(feature = "developer")]
    #[inline]
    pub fn mark_send_dropped_queue_full(&self) {
        self.sends_dropped_queue_full.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(feature = "developer")]
    #[inline]
    pub fn sends_dropped_queue_full(&self) -> u32 {
        self.sends_dropped_queue_full.load(Ordering::Relaxed)
    }

    /// Record one of `prepare_multi_entry_compile`'s three silent early-bounce
    /// outcomes for this page — see `prepare_bounced`'s own field doc comment.
    #[inline]
    pub fn mark_prepare_bounced(&self) {
        self.prepare_bounced.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn prepare_bounced(&self) -> u32 {
        self.prepare_bounced.load(Ordering::Relaxed)
    }

    /// Record that this page's compile bounced on arena-out-of-memory in
    /// codegen — see `codegen_oom_bounced`'s own field doc comment.
    #[inline]
    pub fn mark_codegen_oom_bounced(&self) {
        self.codegen_oom_bounced.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn codegen_oom_bounced(&self) -> u32 {
        self.codegen_oom_bounced.load(Ordering::Relaxed)
    }

    /// Record that this page's successful compile came back gap-blocked
    /// (unsealed) from `finalize_batch_nonforced` — see `seal_gap_blocked`'s
    /// own field doc comment.
    #[inline]
    pub fn mark_seal_gap_blocked(&self) {
        self.seal_gap_blocked.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn seal_gap_blocked(&self) -> u32 {
        self.seal_gap_blocked.load(Ordering::Relaxed)
    }

    /// Record that this page's `finalize_batch_nonforced` call failed
    /// outright (`Codegen::last_finalize_failed`) rather than merely
    /// gap-blocking — see `finalize_failed`'s own field doc comment.
    #[inline]
    pub fn mark_finalize_failed(&self) {
        self.finalize_failed.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn finalize_failed(&self) -> u32 {
        self.finalize_failed.load(Ordering::Relaxed)
    }

    /// Whether `offset_word` is a live entry point into `func` right now,
    /// without regard to whether `func` is still fresh against the page's
    /// current gen. Callers that need "is this dispatchable right now" want
    /// [`Self::is_runnable`]; this is for the dispatch-trigger gate
    /// (§13.2), which probes coverage first and only then decides
    /// exec-vs-recompile from gen.
    #[inline]
    pub fn is_published(&self, offset_word: usize) -> bool {
        bitmap_test(&self.compiled, offset_word, Ordering::Acquire)
    }

    /// Whether `offset_word` is both a live entry (`compiled` bit set) and
    /// `func` is still fresh (the page's `entry_gen` matches its current
    /// gen, §13.3). A set compiled bit whose gen has drifted is stale — the
    /// caller should treat it as unpublished (downgrade to interpreter,
    /// §13.2) rather than dispatch it.
    ///
    /// Generation first (Acquire), then the bit, so a bit is never paired with a
    /// newer generation. A later held publish can still swap in `jit_no_code`.
    #[inline]
    pub fn is_runnable(&self, offset_word: usize) -> bool {
        self.entry_gen.load(Ordering::Acquire) == self.current_gen()
            && bitmap_test(&self.compiled, offset_word, Ordering::Acquire)
    }

    /// Whether `func` is real code rather than the `jit_no_code` stub.
    #[inline]
    pub fn has_code(&self) -> bool {
        self.func.load(Ordering::Relaxed) != NO_CODE as *mut ()
    }

    /// The page's compiled function, or `jit_no_code` if unpublished. Callers must
    /// check [`Self::is_runnable`] first (in that order — see its doc
    /// comment on why the `entry_gen` Acquire load must happen-before this
    /// read) before trusting the returned pointer.
    #[inline]
    pub fn func(&self) -> *const () {
        self.func.load(Ordering::Relaxed) as *const ()
    }

    /// Whether `offset_word` has been sticky-rejected by the compiler for the
    /// page's current generation (§13.6). `denied` is inverted-sense: a clear
    /// bit means denied.
    #[inline]
    pub fn is_denylisted(&self, offset_word: usize) -> bool {
        !bitmap_test(&self.denied, offset_word, Ordering::Relaxed)
    }

    /// Sticky-reject `offset_word` for the page's current generation (§13.6).
    #[inline]
    pub fn denylist(&self, offset_word: usize) {
        // `denied` is inverted (1 = allowed) — "deny" clears the bit, so this
        // can't be a plain fetch_or the way the old ENTRY_DENYLISTED flag was;
        // AND with everything except this bit instead.
        self.denied[offset_word >> 6].fetch_and(!(1u64 << (offset_word & 63)), Ordering::Relaxed);
    }

    /// Sticky-reject every offset on this page at once. Not on any hot path
    /// (a page-wide, one-time operation) — used to make a whole page
    /// permanently non-dispatchable, e.g. tests that need a deterministic
    /// "this page never compiles/dispatches, everything runs on the
    /// interpreter" guarantee.
    pub fn denylist_all(&self) {
        for word in self.denied.iter() { word.store(0, Ordering::Relaxed); }
    }

    /// Undo [`Self::denylist_all`].
    pub fn allow_all(&self) {
        for word in self.denied.iter() { word.store(u64::MAX, Ordering::Relaxed); }
    }

    /// Clear `offset_word`'s `compiled` bit only — NOT `denied`. Two
    /// distinct callers:
    ///
    /// - `emit_fr_mode_guard`'s FR-mismatch case (`jit_kill_entry`): the
    ///   artifact itself isn't wrong in general, just stale for *this*
    ///   dispatch (compiled for the wrong FR mode). A later dispatch is
    ///   expected and welcome to recompile — most likely for the *other* FR
    ///   mode — so `denied` is left alone.
    /// - `comp.rs`'s `prepare_multi_entry_compile`, paired with `denylist`:
    ///   a re-walk that excludes a previously-`compiled` offset (§13's
    ///   one-function-per-page model folds prior coverage into every
    ///   compile's candidates) leaves that offset with no case in the fresh
    ///   `func`'s dispatch switch at all — `compiled`'s bit must not survive
    ///   or `is_runnable` keeps reporting it dispatchable into a function
    ///   that doesn't recognize it, turning every future hit into a silent
    ///   JIT-entry-then-EXEC_FALLBACK round trip forever. Here `denied` is
    ///   *not* left alone (the paired `denylist` call sticky-rejects it, on
    ///   the theory that a walk which just excluded it once is likely to
    ///   again) — the two calls are deliberately paired at that call site.
    ///
    /// `func` itself is deliberately left in place either way; nothing
    /// between clearing this bit and the next `publish` ever reads `func`
    /// without first re-checking `is_runnable`.
    #[inline]
    pub fn kill(&self, offset_word: usize) {
        self.compiled[offset_word >> 6].fetch_and(!(1u64 << (offset_word & 63)), Ordering::Release);
        // Block the churn-avoidance skip until a real compile re-derives this
        // page's coverage — see `kills`.
        self.kills.fetch_add(1, Ordering::Release);
    }

    /// `kill` for an offset the compile path is denylisting in the same
    /// breath: clears its `compiled` bit but does not bump `kills`.
    ///
    /// The kill count exists to stop a skip re-advertising an offset whose
    /// dispatch should end. A denied offset cannot be re-advertised by a
    /// skip at all: denials are masked out of every candidate set a skip is
    /// asked to cover. Once a later generation clears `denied` and the offset
    /// comes back, a skip still only re-advertises it if the installed `func`
    /// has a case for it (`snap.entries`), for identical decoded bytes, in
    /// the same FR mode, which is exactly when dispatching into it is right.
    ///
    /// Bumping here cost real work: every compile attempt that denied an
    /// entry (excluded at its own entry, or a region too short) poisoned the
    /// page's snapshot until a full recompile landed, and on an IP28 session
    /// that refused 1,068 of 6,500 compiles' worth of skips.
    #[inline]
    pub fn kill_denylisted(&self, offset_word: usize) {
        self.compiled[offset_word >> 6].fetch_and(!(1u64 << (offset_word & 63)), Ordering::Release);
    }

    /// Mark `offset_word` as a requested entry point (§13.2) — the bit
    /// accumulates and stays set across however many compile passes it takes
    /// to eventually get covered (§13.3's coalescing); it is not claimed by a
    /// single "first caller" and cleared per-offset once decided. Idempotent
    /// — setting an already-set bit is a no-op.
    ///
    /// Test-then-set, not an unconditional `fetch_or`: the steady state is
    /// an arrival at an offset that is *already* requested (the compile that
    /// would cover it has not run yet), and the unconditional read-modify-
    /// write put a shared cache-line write on the CPU thread's hot path
    /// there. That write invalidates the compile worker's copy of the same
    /// line — the worker clears these very bits when it publishes — and the
    /// two then ping-pong the line for every arrival. The load below is
    /// enough to skip the write; the `fetch_or` still does the actual set,
    /// so two CPUs racing on a fresh offset both end up with it requested.
    #[inline]
    pub fn mark_requested(&self, offset_word: usize) {
        let word = offset_word >> 6;
        let bit = 1u64 << (offset_word & 63);
        if self.requested[word].load(Ordering::Relaxed) & bit != 0 {
            return;
        }
        self.requested[word].fetch_or(bit, Ordering::Relaxed);
    }

    /// Whether `offset_word` currently has a live `requested` bit — i.e. some
    /// dispatch has asked for it and no compile has covered/denylisted it
    /// since. Diagnostic use (`j2 pcp`); the compile-request path reads the
    /// whole bitmap at once via `snapshot_compile_candidates` instead.
    #[inline]
    pub fn is_requested(&self, offset_word: usize) -> bool {
        bitmap_test(&self.requested, offset_word, Ordering::Relaxed)
    }

    /// Test-and-set the page-level "a `CompileRequest` is already in flight
    /// for this page" flag: returns `true` (and sets it) only if it was
    /// previously clear, i.e. only the first caller should actually build and
    /// send a `CompileRequest` for this page right now — every other
    /// concurrent/subsequent caller sees `false` and skips it, since whatever
    /// gets dequeued will snapshot `requested` fresh and pick up every bit
    /// accumulated by then anyway (§13.2/§13.3), not just the one that
    /// triggered this particular call. Distinct from `mark_requested`: this
    /// gate is purely about not flooding the queue with redundant page-level
    /// sends, not about tracking which offsets are wanted.
    ///
    /// The plain load first is an optimization, not the decision: the steady
    /// state is "a request is already in flight", and the unconditional
    /// `swap` wrote the shared flag on every arrival (ping-ponging its cache
    /// line with the compile worker that clears it). The `swap` below is
    /// still the test-and-set that picks the one caller that sends.
    #[inline]
    pub fn try_schedule_page(&self) -> bool {
        if self.page_scheduled.load(Ordering::Relaxed) {
            return false;
        }
        !self.page_scheduled.swap(true, Ordering::Relaxed)
    }

    /// Read-only peek at the page-level in-flight flag — diagnostic use
    /// (`j2 pcp`) only. Never use this to decide whether to send a
    /// `CompileRequest`: that's `try_schedule_page`'s test-and-set job, and a
    /// plain load-then-branch here would race it (two callers could both
    /// observe `false` and both send).
    #[inline]
    pub fn is_scheduled(&self) -> bool {
        self.page_scheduled.load(Ordering::Relaxed)
    }

    /// Clear the page-level in-flight flag — called once `handle_request`
    /// (`comp.rs`) has decided this page's request one way or another (some
    /// coverage published, or nothing new to offer), so a later legitimate
    /// re-request (e.g. a fresh discovery that arrived after this compile's
    /// snapshot was already taken) isn't permanently blocked by a stale flag
    /// from a request that already finished.
    #[inline]
    pub fn clear_scheduled(&self) {
        self.page_scheduled.store(false, Ordering::Relaxed);
        self.hot_wait.store(0, Ordering::Relaxed);
        self.promoted.store(false, Ordering::Relaxed);
    }

    /// The hot lane was full: count afresh, so the page is promoted again if
    /// it is still hot.
    pub fn rearm_promotion(&self) {
        self.hot_wait.store(0, Ordering::Relaxed);
        self.promoted.store(false, Ordering::Relaxed);
    }

    /// The FIFO was full: the page asks again at its next arrival, and the
    /// arrivals it has counted still bring it to the hot lane.
    pub fn drop_request(&self) {
        self.page_scheduled.store(false, Ordering::Relaxed);
    }

    /// Count one arrival at this page while its request is already queued.
    /// Returns `true` exactly once per wait, when the count reaches
    /// [`HOT_PROMOTE_ARRIVALS`]: the caller then puts a copy of the request on
    /// the hot lane. The FIFO copy, when it is eventually dequeued, is skipped
    /// as redundant by the ordinary subsumption checks.
    #[inline]
    pub fn note_waiting_arrival(&self) -> bool {
        self.hot_wait.fetch_add(1, Ordering::Relaxed) + 1 >= HOT_PROMOTE_ARRIVALS
            && !self.promoted.swap(true, Ordering::Relaxed)
    }

    /// Count `n` arrivals at this page while its request is already queued.
    ///
    /// The dispatch gate counts arrivals here so a page that is executed
    /// hard while its request sits in a full FIFO gets promoted to the hot
    /// lane. Counting them one at a time put a shared atomic write on the
    /// CPU thread's hot path, where it ping-ponged with the compile worker's
    /// copy of the same cache line. The caller counts per CPU instead (a
    /// field of its own, not shared) and folds the total in here, so the
    /// total counted is the same — the promotion threshold is unchanged —
    /// but the shared line is written once per batch instead of once per
    /// arrival.
    ///
    /// The one bounded exception is a batch that spans a wait boundary: if
    /// the page's request completes while the CPU still holds part of a
    /// batch, those arrivals are folded into the *next* wait, so the page
    /// can be promoted a few arrivals early out of `HOT_PROMOTE_ARRIVALS`.
    ///
    /// This only counts. The promotion is a separate step
    /// ([`Self::take_hot_promotion`]): the caller that counts the arrivals
    /// is the one that can put the request on the hot lane, and it must be
    /// the page it is on.
    #[inline]
    pub fn note_waiting_arrivals(&self, n: u32) {
        self.hot_wait.fetch_add(n, Ordering::Relaxed);
    }

    /// Whether this page is due for the hot lane: its arrival count has
    /// reached [`HOT_PROMOTE_ARRIVALS`] and it has not been promoted yet.
    ///
    /// Returns `true` exactly once per wait — the caller then puts a copy of
    /// the request on the hot lane.
    #[inline]
    pub fn take_hot_promotion(&self) -> bool {
        // `promoted` first: it stays set for the rest of the wait, so the
        // steady state is one load and no write. The count is read second,
        // and the swap below only happens when the page is actually due.
        !self.promoted.load(Ordering::Relaxed)
            && self.hot_wait.load(Ordering::Relaxed) >= HOT_PROMOTE_ARRIVALS
            && !self.promoted.swap(true, Ordering::Relaxed)
    }

    /// Clear every `requested` bit this compile's snapshot covered, after a
    /// successful publish — so a later re-request for the same offset (e.g.
    /// after a future gen bump) isn't shadowed by a stale bit this compile
    /// already acted on. Bits set by a *newer* discovery (arrived after this
    /// compile's snapshot, not part of `covered`) are left untouched — they
    /// still need a future compile to pick them up.
    #[inline]
    pub fn clear_requested_bits(&self, covered: &[u64; BITMAP_WORDS]) {
        for i in 0..BITMAP_WORDS {
            if covered[i] != 0 {
                self.requested[i].fetch_and(!covered[i], Ordering::Relaxed);
            }
        }
    }

    /// Snapshot the `requested` bitmap (§13.3 step 3).
    #[inline]
    pub fn snapshot_requested(&self) -> [u64; BITMAP_WORDS] {
        bitmap_snapshot(&self.requested)
    }

    /// Snapshot the `compiled` bitmap — every offset that's a live entry
    /// point into `func` right now (before any `entry_gen` staleness check;
    /// see `is_runnable`). Diagnostic use only (`j2 dump-pcp`) — the hot
    /// dispatch path never needs the whole bitmap at once, only
    /// `is_published`'s single-bit test, and `comp.rs`'s own use of
    /// `compiled` alongside `requested` goes through
    /// [`Self::snapshot_compile_candidates`] instead.
    #[inline]
    pub fn snapshot_compiled(&self) -> [u64; BITMAP_WORDS] {
        bitmap_snapshot(&self.compiled)
    }

    /// Snapshot the `denied` bitmap, in its raw (inverted: 1 = allowed)
    /// sense — see the field's own doc comment. Diagnostic use only.
    #[inline]
    pub fn snapshot_denied_raw(&self) -> [u64; BITMAP_WORDS] {
        bitmap_snapshot(&self.denied)
    }

    /// `comp.rs`'s `prepare_multi_entry_compile` candidate set, computed in
    /// one pass: `(requested | compiled) & denied` per word (`denied`'s raw,
    /// inverted sense — a set bit there means "still eligible", so ANDing
    /// with it is exactly "not denylisted").
    ///
    /// `include_compiled`: whether to fold `compiled` in at all — the caller
    /// must only pass `true` when `entry_gen == gen_snap == current_gen()`
    /// (a `compiled` bit from a superseded generation doesn't describe the
    /// bytes this compile is about to snapshot; `publish`'s own
    /// `is_real_invalidation` path already resets the bitmap for that case,
    /// so this method doesn't re-check gen itself — it has no `gen_snap` of
    /// its own to compare against).
    ///
    /// Folding `compiled` in (when eligible) is what makes every recompile
    /// re-cover previously-published entries, not just this round's fresh
    /// `requested` bits: §13's one-function-per-page model replaces `func`
    /// wholesale on every compile, and its dispatch switch only recognizes
    /// the entries this exact walk covers (codegen.rs's `compile_region`
    /// builds the switch from nothing else) — an old entry left out of a
    /// merged walk has no case in the new `func` at all, not just stale
    /// coverage, so every future dispatch into it would silently fall to
    /// EXEC_FALLBACK forever.
    #[inline]
    pub fn snapshot_compile_candidates(&self, include_compiled: bool) -> [u64; BITMAP_WORDS] {
        let mut out = [0u64; BITMAP_WORDS];
        for i in 0..BITMAP_WORDS {
            let mut bits = self.requested[i].load(Ordering::Relaxed);
            if include_compiled {
                bits |= self.compiled[i].load(Ordering::Acquire);
            }
            out[i] = bits & self.denied[i].load(Ordering::Relaxed);
        }
        out
    }

    /// Whether every bit set in `bits` is already covered by `compiled`
    /// (§13.3 steps 4/6's subsumption check).
    #[inline]
    pub fn requested_subsumed_by_compiled(&self, bits: &[u64; BITMAP_WORDS]) -> bool {
        bitmap_is_subset_of(bits, &self.compiled)
    }

    /// Publish a freshly compiled function covering `new_entries` (§13.3 step
    /// 6): re-check page gen and subsumption, then union `new_entries` into
    /// `compiled`, install `func`, and write `entry_gen` — all under
    /// `publish_lock` so the three-field update happens as one step relative
    /// to any concurrent publisher for this same page (§13.5). Returns `true`
    /// iff this call actually published (i.e. wasn't stale/subsumed).
    ///
    /// `snap_gen` is the page generation the compile snapshot was confirmed
    /// against (§13.3 step 2's seqlock). Two independent reasons abort:
    ///
    /// - `current_gen() > snap_gen`: the page mutated after this compile's
    ///   snapshot was taken — the compiled bytes are stale regardless of
    ///   entry coverage.
    /// - `new_entries ⊆ compiled` (current, re-read under the lock) **for
    ///   code of the same FR mode** (`fr1`, the mode this compile targeted):
    ///   some other compile already published a superset of what this
    ///   compile covers — redundant.
    ///
    /// The FR mode has to be part of that second test. Kernel pages run
    /// under whichever mode the interrupted process uses, so one generation
    /// can be compiled for FR1 and then, after the pin moves, for FR0. Were
    /// the FR0 compile refused as "already covered", the page would be left
    /// pinned FR0 with FR1 code and an FR1 churn snapshot: every later
    /// request fails the skip on the mode, recompiles, and is refused again,
    /// until the bytes change or the arena is flushed. That loop was seen
    /// running thousands of rounds on a kernel page.
    ///
    /// Stores `func`, `compiled`, then `entry_gen` (Release); `is_runnable` reads
    /// them back in reverse. `entry_gen` is written unconditionally on every actual
    /// publish, but only **changes** when `snap_gen != entry_gen` (a real
    /// page-bump-triggered recompile, or the first after `GEN_HELD` comes
    /// off, §13.6/§13.3's
    /// discussion) — a pure entry-coverage publish against unchanged bytes
    /// (`snap_gen == entry_gen` already) rewrites the same value. Readers
    /// never need to distinguish the two: `entry_gen == current_gen()` is
    /// the only fact `is_runnable` needs.
    ///
    /// When `snap_gen` genuinely advances `entry_gen`, `denied` resets to
    /// all-allowed (§13.6): a page-gen bump means the bytes are provably
    /// different from whatever got denied last time.
    ///
    /// For code that lives in no rotating arena (the stub, a test's
    /// function); compiled code goes through [`Self::publish_from`].
    pub fn publish(
        &self,
        new_entries: &[u64; BITMAP_WORDS],
        func: *const (),
        snap_gen: u64,
        instr_count: usize,
        code_size: u32,
        fr1: bool,
    ) -> bool {
        self.publish_from(&crate::cpu::jitv2::paged_memory::PagedArenaState::UNTRACKED, self.tenancy(), new_entries, func, snap_gen, instr_count, code_size, fr1)
    }

    /// [`Self::publish`] for a function in `arena`. Refused once the arena is
    /// retired: its code is being taken off the pages, and the test is made
    /// under `publish_lock`, which [`Self::retire_code`] takes too, so this
    /// publish either lands first and is swept, or does not land.
    ///
    /// `tenancy` is the slot's tenancy when the compile read the page
    /// ([`Self::tenancy`]). A publish waits in the seal queue after its
    /// compile is over, and the pool may give the slot to other memory
    /// meanwhile: the code is then for the wrong page, and is refused.
    pub fn publish_from(
        &self,
        arena: &crate::cpu::jitv2::paged_memory::PagedArenaState,
        tenancy: u32,
        new_entries: &[u64; BITMAP_WORDS],
        func: *const (),
        snap_gen: u64,
        instr_count: usize,
        code_size: u32,
        fr1: bool,
    ) -> bool {
        let _guard = self.publish_lock.lock();

        if arena.is_retired() || self.tenancy.load(Ordering::SeqCst) != tenancy {
            return false;
        }
        if self.current_gen() > snap_gen {
            return false; // page mutated past this compile's snapshot; discard rather than publish stale code
        }
        // The subsumption check only means anything when comparing against
        // `compiled` bits published for the SAME generation this compile
        // targets: a bit set under an older, now-superseded generation
        // doesn't actually cover this offset for the bytes this compile
        // just analyzed — it's leftover coverage from before the page
        // changed, about to be wiped by the entry_gen advance below. Only
        // skip-as-redundant when entry_gen already matches snap_gen (a pure
        // entry-coverage publish racing another one for the same bytes).
        let same_gen = self.entry_gen.load(Ordering::Relaxed) == snap_gen;
        let mode_switch = self.has_code()
            && self.func_fr1.load(Ordering::Relaxed) != fr1;
        if same_gen && !mode_switch && bitmap_is_subset_of(new_entries, &self.compiled) {
            return false; // some other compile already published everything this one covers, for this same generation and mode
        }

        // Safety: `func` is a raw pointer write behind `&self` — sound
        // because no concurrent reader trusts it without first Acquire-
        // loading `entry_gen` (via `is_runnable`) and observing it equal
        // to current_gen(); `publish_lock` excludes any concurrent writer of
        // these same three fields.
        self.func.store(func as *mut (), Ordering::Release);
        self.func_fr1.store(fr1, Ordering::Relaxed);
        self.code_epoch.store(if func == NO_CODE as *const () { 0 } else { arena.epoch() }, Ordering::Relaxed);
        self.compiles_since_flush.fetch_add(1, Ordering::Relaxed);
        self.instr_count.store(instr_count as u32, Ordering::Relaxed);
        self.code_size.store(code_size, Ordering::Relaxed);
        #[cfg(feature = "developer")]
        {
            self.compile_count.fetch_add(1, Ordering::Relaxed);
        }

        let prev_entry_gen = self.entry_gen.load(Ordering::Relaxed);
        // Not `>`: a page's generation drops when `GEN_HELD` comes off.
        let is_real_invalidation = snap_gen != prev_entry_gen;
        // Code for the other FR mode replaces the old function outright, so
        // the old function's entries must go with it: its dispatch switch is
        // gone, and the new one only has cases for `new_entries`.
        if is_real_invalidation || mode_switch {
            // The old `compiled` bits refer to a now-superseded generation —
            // offsets compiled against bytes that no longer exist. Replace,
            // don't union: this compile's `new_entries` becomes the entire
            // coverage set for the new generation, not an addition to stale
            // leftovers (which would let a reader dispatch into `func` for
            // an offset this fresh compile never actually analyzed).
            for i in 0..BITMAP_WORDS {
                self.compiled[i].store(new_entries[i], Ordering::Release);
            }
        } else {
            bitmap_union_from(&self.compiled, new_entries, Ordering::Release);
        }

        self.entry_gen.store(snap_gen, Ordering::Release);
        if is_real_invalidation {
            // Real invalidation, not just added coverage: the bytes changed,
            // so whatever got sticky-denied against the OLD bytes deserves a
            // fresh chance against the new ones (§13.6).
            for word in self.denied.iter() { word.store(u64::MAX, Ordering::Relaxed); }
            // `fr1` has exactly the same justification and used to be the one
            // field left behind here. A physical page freed by an o32 (FR0)
            // process and reallocated to an n32 (FR1) one arrived with new
            // bytes but kept the previous tenant's pin, so the next compile
            // specialized fresh code for the wrong FR mode — and because
            // `emit_fr_mode_guard` suppresses itself when CU1 is clear, that
            // was not reliably caught. Don't pin a mode here (the compiling
            // thread's notion of "live" is the one that matters, and it is
            // read at compile-request time); just drop the stale claim so the
            // next compile re-derives it. See `fr1`'s own doc comment.
            self.fr_unpinned.store(true, Ordering::Relaxed);
        }
        true
    }

    /// See the `tenancy` field.
    #[inline(always)]
    pub fn tenancy(&self) -> u32 {
        self.tenancy.load(Ordering::SeqCst)
    }

    /// A CPU is entering the page: the recycling hand passes it over once.
    #[inline(always)]
    pub fn touch(&self) {
        // Read first: the line is shared with every CPU on the page.
        if !self.referenced.load(Ordering::Relaxed) {
            self.referenced.store(true, Ordering::Relaxed);
        }
    }

    /// A worker is about to compile this page for a request made in tenancy
    /// `tenancy`: hold the slot for the compile, or say the request is stale.
    ///
    /// The worker reads the page's key, bytes and generation pointer, which
    /// the pool rewrites when it recycles the slot, so the two must not
    /// overlap. The worker sets the in-flight flag and then reads the
    /// tenancy; the pool moves the tenancy and then reads the flag
    /// ([`Jitv2::recycle_slot`]); both `SeqCst`. So either the pool sees the
    /// flag and leaves the slot, or the worker sees the tenancy moved and
    /// drops the request. `handle_request`'s guard clears the flag when the
    /// compile is over.
    pub fn pin_for_compile(&self, tenancy: u32) -> bool {
        let was_set = self.page_scheduled.swap(true, Ordering::SeqCst);
        if self.tenancy.load(Ordering::SeqCst) == tenancy {
            return true;
        }
        // Not this request's page any more. Leave the flag as it was: set,
        // it is the new tenant's own request.
        if !was_set {
            self.page_scheduled.store(false, Ordering::SeqCst);
        }
        false
    }

    /// The recycling hand: was the page entered since the last pass?
    fn take_referenced(&self) -> bool {
        self.referenced.swap(false, Ordering::Relaxed)
    }

    /// The pool is taking this slot back: frame table entries and publishes
    /// made for its tenant stop matching. Returns the new tenancy.
    fn end_tenancy(&self) -> u32 {
        let _guard = self.publish_lock.lock();
        self.tenancy.fetch_add(1, Ordering::SeqCst).wrapping_add(1)
    }

    /// The epoch of the arena `func` lives in, 0 for the stub.
    #[inline]
    pub fn code_epoch(&self) -> u64 {
        self.code_epoch.load(Ordering::Relaxed)
    }

    /// The compile pool is retiring the arena of epoch `epoch`: if this
    /// page's code lives there, take it away. The page keeps its key, its FR
    /// pin and its denials, and compiles again when it is next entered.
    ///
    /// `requested` goes with the code. A page that lives through many
    /// rotations would otherwise keep every entry point it ever had, and
    /// compile all of them again each time
    /// (clanker/jitv2/flush-preserved-entry-point-inflation.md).
    ///
    /// A CPU may be between `is_runnable` and `func()` on this page: it then
    /// calls either the old function, whose arena stays mapped until every
    /// CPU has left compiled code after this sweep, or the stub.
    ///
    /// Returns whether the page had code there.
    pub fn retire_code(&self, epoch: u64) -> bool {
        // Snapshot lock first, as `try_skip_redundant_compile` takes them.
        let mut snapshots = self.snapshots.lock();
        let _guard = self.publish_lock.lock();
        if epoch == 0 || self.code_epoch.load(Ordering::Relaxed) != epoch {
            return false;
        }
        for word in self.compiled.iter() { word.store(0, Ordering::Release); }
        self.func.store(NO_CODE as *mut (), Ordering::Release);
        self.func_fr1.store(false, Ordering::Relaxed);
        self.code_epoch.store(0, Ordering::Relaxed);
        self.instr_count.store(0, Ordering::Relaxed);
        self.code_size.store(0, Ordering::Relaxed);
        for word in self.requested.iter() { word.store(0, Ordering::Relaxed); }
        // The record describes the function just dropped, and a compile in
        // flight must not come back and vouch for it.
        snapshots.committed.valid = false;
        snapshots.staged = None;
        true
    }

    /// Record what a compile looked at, **provisionally** — the record is
    /// stored but left invalid, so it cannot authorize a skip until
    /// [`Self::commit_compile_snapshot`] vouches for it with the publish
    /// that actually installed the code.
    ///
    /// Called right after codegen succeeds, before the publish (which, on
    /// the deferred path, may happen later and on another thread). `used`
    /// must be every word the walk decoded, including delay slots; `entries`
    /// exactly the bitmap that will be handed to `publish`.
    ///
    /// Overwrites any earlier staged-but-uncommitted record — which can only
    /// be one left behind by a compile whose publish never landed (it was
    /// stale, subsumed, or the page was reset under it), since this page
    /// cannot be compiled twice at once. `staged_gen` is what lets
    /// [`Self::commit_compile_snapshot`] tell its own record from such a
    /// leftover.
    pub fn stage_compile_snapshot(
        &self,
        words: &[u32; ENTRIES_PER_PAGE],
        used: &[u64; BITMAP_WORDS],
        entries: &[u64; BITMAP_WORDS],
        gen_snap: u64,
        fr1: bool,
    ) {
        // Into the staging slot only: `committed` keeps describing the
        // installed `func` until this compile's publish actually lands.
        let mut rec = Box::new(CompileSnapshot::new());
        rec.staged_gen = gen_snap;
        rec.words.copy_from_slice(words);
        rec.used.copy_from_slice(used);
        rec.entries.copy_from_slice(entries);
        rec.fr1 = fr1;
        rec.kills_at_stage = self.kills.load(Ordering::Acquire);
        self.snapshots.lock().staged = Some(rec);
    }

    /// Promote a staged record to usable, now that the publish it describes
    /// has actually installed its code.
    ///
    /// `gen_snap` and `entries` identify the publish doing the vouching; the
    /// record is only committed if they match what was staged. The mismatch
    /// case that matters is a reset landing between stage and publish
    /// (`invalidate_compile_snapshot` poisons `staged_gen` precisely so this
    /// cannot re-validate a snapshot for a function the reset discarded) —
    /// the record is dropped rather than committed, costing a future
    /// recompile instead of risking one that skips against the wrong bytes.
    pub fn commit_compile_snapshot(&self, gen_snap: u64, entries: &[u64; BITMAP_WORDS], fr1: bool) {
        let mut g = self.snapshots.lock();
        // FR mode too: two compiles of one page at one generation can differ
        // only in mode, and the record must describe the mode installed.
        let matches = matches!(&g.staged, Some(r)
            if r.staged_gen == gen_snap && r.entries == *entries && r.fr1 == fr1);
        if matches {
            let mut rec = g.staged.take().unwrap();
            rec.valid = true;
            g.committed = *rec;
        } else {
            // A publish landed that the staged record does not describe (a
            // reset took it, or a newer compile replaced it). Whatever is
            // installed now, `committed` no longer describes it.
            g.committed.valid = false;
        }
    }

    /// Whether this page has a recorded compile snapshot to compare a new
    /// request against. `false` on a page that has never compiled, or whose
    /// snapshot was invalidated by a reset — those requests are not
    /// candidates for a skip at all, and the caller counts neither outcome
    /// for them (see [`Self::redundant_skipped`]).
    #[inline]
    pub fn has_compile_snapshot(&self) -> bool {
        self.snapshots.lock().committed.valid
    }

    /// Count one compile request answered by the snapshot — see
    /// [`Self::redundant_skipped`].
    #[inline]
    pub fn mark_redundant_compile_skipped(&self) {
        self.redundant_skipped.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one compile request that had a snapshot but still had to
    /// compile — see [`Self::redundant_rejected`].
    #[inline]
    pub fn mark_redundant_compile_rejected(&self) {
        self.redundant_rejected.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn redundant_skipped(&self) -> u32 {
        self.redundant_skipped.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn redundant_rejected(&self) -> u32 {
        self.redundant_rejected.load(Ordering::Relaxed)
    }

    /// Drop the recorded compile snapshot — the `func` it describes is gone
    /// or no longer trustworthy, so it must not authorize a skip.
    ///
    /// Every reset path that clears `func` calls this. There is no path that
    /// legitimately keeps a snapshot past its function: a skip works by
    /// re-validating the *installed* code, so a stale snapshot would
    /// advance `entry_gen` over a null or superseded `func` and send
    /// dispatch into nothing.
    #[inline]
    pub fn invalidate_compile_snapshot(&self) {
        let mut g = self.snapshots.lock();
        g.committed.valid = false;
        // Drop the staging slot too: a compile in flight across this reset
        // would otherwise come back and commit its record, re-validating a
        // snapshot for a `func` this reset just threw away.
        g.staged = None;
    }

    /// Churn-avoidance fast path (§13.3 step 2.5): decide whether the compile
    /// this request is asking for would reproduce, byte for byte, the code
    /// already installed — and if so, re-validate that code against the new
    /// generation and skip the compile entirely.
    ///
    /// The caller has already taken its seqlock byte snapshot (`words`,
    /// coherent with `gen_snap`) and computed the entry points it wants
    /// (`wanted_entries`). Three things must all hold to skip:
    ///
    /// 1. **The wanted entries are a subset of what's published.** The
    ///    installed `func`'s dispatch switch only has cases for the entries
    ///    its own compile covered, so an entry outside that set cannot be
    ///    served no matter how identical the bytes are.
    /// 2. **Every word the recorded compile decoded is unchanged.** Words
    ///    outside `used` are irrelevant by construction — the walk never
    ///    read them, so they had no influence on the emitted code. This is
    ///    what makes the check pay off on a page whose generation bumps
    ///    because the guest wrote *data* sharing the page with code.
    /// 3. **The FR mode matches.** See `CompileSnapshot::fr1`.
    ///
    /// On success `entry_gen` is advanced to `gen_snap` under
    /// `publish_lock`, which is exactly what makes `is_runnable` start
    /// returning true again — the same single-field write `publish` would
    /// have performed, minus the `func`/`compiled` stores (unchanged) and
    /// minus the `denied` reset, which would be actively wrong here: the
    /// bytes are provably identical, so an offset denied against them is
    /// still correctly denied, and clearing it would re-walk and re-reject
    /// it on every generation bump — the very churn this exists to remove.
    ///
    /// Returns `true` iff the compile was skipped.
    pub fn try_skip_redundant_compile(
        &self,
        words: &[u32; ENTRIES_PER_PAGE],
        wanted_entries: &[u64; BITMAP_WORDS],
        gen_snap: u64,
        fr1: bool,
    ) -> bool {
        // Held for the whole check, so a commit on another worker cannot
        // swap the record out from under the comparison.
        let g = self.snapshots.lock();
        let snap = &g.committed;
        if !snap.valid || snap.fr1 != fr1 {
            { self.last_skip_reject.store(1, Ordering::Relaxed); return false; }
        }
        // A published `func` is what the snapshot describes; without one
        // there is nothing to re-validate. (Belt and braces — every path
        // that nulls `func` also invalidates the snapshot.)
        if !self.has_code() {
            { self.last_skip_reject.store(2, Ordering::Relaxed); return false; }
        }
        if !bitmap_is_subset_of_plain(wanted_entries, &snap.entries) {
            { self.last_skip_reject.store(3, Ordering::Relaxed); return false; }
        }
        for i in 0..BITMAP_WORDS {
            let mut bits = snap.used[i];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let w = i * 64 + bit;
                if words[w] != snap.words[w] {
                    { self.last_skip_reject.store(4, Ordering::Relaxed); return false; }
                }
            }
        }
        // Re-validate the installed code against the new generation. Taken
        // under `publish_lock` for the same reason `publish` does: this page
        // is never compiled twice at once, but a dispatch-side `kill` can
        // still land concurrently, and the gen write must not interleave with
        // the coverage check below.
        let _guard = self.publish_lock.lock();
        // Re-check under the lock, exactly as `publish` does — a mutation
        // may have landed between the byte snapshot and here, in which case
        // the bytes we just compared are already history.
        if self.current_gen() > gen_snap {
            { self.last_skip_reject.store(5, Ordering::Relaxed); return false; }
        }
        // Never move `entry_gen` backwards: a concurrent publish may have
        // already installed newer coverage at a higher generation, and
        // rewinding it would invalidate that publish's own code.
        if self.entry_gen.load(Ordering::Relaxed) > gen_snap {
            { self.last_skip_reject.store(6, Ordering::Relaxed); return false; }
        }
        // Deliberately NOT `wanted_entries ⊆ compiled`. That is the wrong
        // question: `compiled` is what is currently *advertised*, which a
        // previous skip may legitimately have narrowed (see the coverage
        // write below), whereas what matters is whether the installed `func`
        // has a case for these offsets — `snap.entries`, checked above, which
        // does not change while that same `func` stays installed. Requiring
        // the stricter condition refuses to re-cover an entry an earlier skip
        // dropped even though the very same function can still serve it,
        // forcing a pointless recompile: a second churn source with the same
        // "skips a lot, runs slower" signature as the first.
        //
        // What DOES have to bail is a `kill` — it means "stop dispatching
        // into `func` here", so a skip must never re-advertise a killed
        // offset. `kills` moving since the snapshot is the exact condition;
        // `compiled` cannot express it, because a narrowing skip clears bits
        // there for an unrelated reason. In production every `kill` is
        // already paired with either an FR repin (caught by `snap.fr1 !=
        // fr1`) or a `denylist` (caught by `denied`, which the caller folds
        // into `wanted_entries`), so this is defence in depth rather than the
        // only guard — but it is the one that states the actual invariant.
        if self.kills.load(Ordering::Acquire) != snap.kills_at_stage {
            { self.last_skip_reject.store(7, Ordering::Relaxed); return false; }
        }

        // Leave exactly the coverage a real `publish` of this same candidate
        // set would have left, because that is what the installed `func` can
        // actually serve.
        //
        // This is NOT a formality — getting it wrong is a throughput
        // regression that masquerades as the optimization working. After a
        // gen bump the caller's candidate set is `requested & denied`
        // (`snapshot_compile_candidates(same_gen=false)` deliberately drops
        // `compiled`), and `publish` treats `snap_gen > entry_gen` as a real
        // invalidation and REPLACES `compiled` wholesale with that set. A
        // skip that merely bumped `entry_gen` would leave the older, wider
        // bitmap published: `is_runnable` then returns true for offsets this
        // check never verified, dispatch calls `func` for one, the switch
        // `compile_region` built has no case for it, and it returns
        // EXEC_FALLBACK — on every arrival, forever, each one re-requesting a
        // compile. Measured as high skip counts with *lower* throughput,
        // worst on R4K.
        //
        // Mirrors `publish`'s own is_real_invalidation branch, and for the
        // same reason: replace on a generation advance (old bits describe
        // bytes that no longer exist), union when the generation is
        // unchanged (pure added coverage against the same bytes).
        if gen_snap > self.entry_gen.load(Ordering::Relaxed) {
            for i in 0..BITMAP_WORDS {
                let dropped = self.compiled[i].swap(wanted_entries[i], Ordering::Release)
                    & !wanted_entries[i];
                // Entries this narrowing drops are still legitimately wanted:
                // unlike `publish`'s equivalent branch, the bytes here are
                // provably UNCHANGED, so an offset that was worth compiling a
                // moment ago is worth compiling still — it just is not in
                // this round's candidate set (it was covered, so nothing
                // re-`requested` it, and `snapshot_compile_candidates` drops
                // `compiled` once the generation moves). Folding it back into
                // `requested` is what gives it a path to a real recompile;
                // without it the entry is silently stranded on the
                // interpreter until something happens to fetch it again,
                // which is the same "no automatic path back" failure a
                // dropped-but-still-wanted entry always risks, guarded
                // against the same way.
                if dropped != 0 { self.requested[i].fetch_or(dropped, Ordering::Relaxed); }
            }
        } else {
            bitmap_union_from(&self.compiled, wanted_entries, Ordering::Release);
        }
        self.entry_gen.store(gen_snap, Ordering::Release);
        true
    }

    // ---- `pcp_dump` capture accessors (`j2 dumppcp` / `j2 corpus`) ----
    //
    // Read-only accessors for whole-page bitmaps, generation, and FPU mode.
    // PcpDump uses these to capture the live compiler state for offline tools.
    pub fn dump_requested(&self) -> [u64; BITMAP_WORDS] { self.snapshot_requested() }
    pub fn dump_compiled(&self) -> [u64; BITMAP_WORDS] { self.snapshot_compiled() }
    pub fn dump_denied_raw(&self) -> [u64; BITMAP_WORDS] { self.snapshot_denied_raw() }
    pub fn dump_entry_gen(&self) -> u64 { self.entry_gen() }
    pub fn dump_fr1(&self) -> bool { self.is_fr1() }

    /// Dispatches into this page's compiled code. Page-level here (§13's one
    /// function per page); 0 when the counter is compiled out, which is the
    /// dump format's documented "unknown".
    pub fn dump_call_count(&self) -> u64 {
        #[cfg(feature = "developer")]
        { self.call_count.load(Ordering::Relaxed) }
        #[cfg(not(feature = "developer"))]
        { 0 }
    }
}

/// Slot index into `Jitv2::pages`.
pub type PageSlot = u32;

/// Sentinel for [`PhysicalCodePage::prev`]/[`PhysicalCodePage::next`]: "no
/// link" (list head's `prev`, list tail's `next`, or a not-yet-linked slot).
/// `pages.len()` can never reach `u32::MAX` in practice (would be 16TB+ of
/// `PhysicalCodePage`s), so this can't collide with a real slot index.
const NO_SLOT: u32 = u32::MAX;
const _: () = assert!(NO_SLOT == PFN_MAP_EMPTY);

/// JIT v2 engine state embedded in the mips executor.
///
/// Owns the [`PhysicalCodePage`] pool (§2.4): a single array, allocated once
/// at full `capacity` in `Jitv2::new` (every slot pre-built as unclaimed —
/// see `PhysicalCodePage::new`'s doc comment), never resized or reallocated
/// afterward. This is stronger than the old `Vec::push`-based growth: that
/// also never reallocated once capacity was reserved, but each `push` still
/// moved a freshly-constructed `PhysicalCodePage` value into place — with
/// `entries` now inline (not boxed), that move would copy the whole
/// 1024-entry table. Claiming a pre-existing slot in place avoids that
/// entirely: nothing is ever copied after `Jitv2::new` builds the array.
///
/// Slots are handed out via `Self::free_head`, a singly-linked free list
/// threaded through every unclaimed slot's `PhysicalCodePage::next`. A slot
/// is claimed iff its `pfn != UNCLAIMED_PFN` (see
/// [`PhysicalCodePage::is_claimed`]) — there is no separate recency list:
/// `mega_flush` resets the whole pool to empty and rebuilds the free list
/// from slot 0, so after a flush "next unclaimed slot" is once again the
/// whole array. (An earlier design preserved the most-recently-used slots
/// across a flush, threaded on an MRU list; that was removed — preserved
/// entry points could be stale, and keeping them was at best neutral.)
///
/// A page is keyed by the memory it holds ([`CodeKey`]: bank and page within
/// it), found through `by_key`. Lookup from a CPU's `pfn` to pool slot goes
/// through its address space's frame table (`spaces`), which caches what the
/// bus and `by_key` say — the design
/// doc's dense pfn-indexed alternative (§2.4), now built: page-switch lookup
/// *did* show up hot in profiling (~4.5% of a real IRIX boot, with the
/// HashMap's own `RawTable` grow-and-rehash allocation directly above it),
/// which is exactly the trigger this comment used to name. See [`PfnMap`].
pub struct Jitv2 {
    /// The full-capacity page pool, allocated once — see this struct's own
    /// doc comment. Indices are stable for the pool's entire lifetime,
    /// including across `mega_flush` (slots are reset/relinked in place, the
    /// array itself never shrinks or reallocates), so pointers into it —
    /// e.g. the executor's current-PCP pointer, `CompileRequest::page` —
    /// stay valid for the process's whole lifetime, not just "until the next
    /// flush" as before. A `CompileRequest`/PCP pointer surviving a flush it
    /// raced against and landing back in a slot that's since been reclaimed
    /// for a different physical page is still a real hazard (unchanged from
    /// before — `worker_loop`'s `drain_pending` and the various
    /// gen-mismatch/`is_runnable` checks are what actually guard against
    /// stale content, not pointer validity) — array-lifetime stability alone
    /// doesn't imply content freshness. Every request still in flight at
    /// flush time is drained before any slot is reclaimed (`mega_flush`'s own
    /// doc comment), so an evicted slot handed to a different pfn is never
    /// racing a stale in-flight reference to begin with.
    pages: Box<[PhysicalCodePage]>,
    /// Head of the free-slot singly-linked list, threaded through
    /// `PhysicalCodePage::next`. `NO_SLOT` when the pool is fully claimed.
    /// Every slot starts here (`Jitv2::new` links the whole array into this
    /// list up front); `page_for` pops from the head on a fresh claim,
    /// `mega_flush` resets the pool and rebuilds the free list.
    free_head: u32,
    /// The pool's index: which slot holds the page for a key. Consulted when
    /// a CPU reaches a frame its space's table has no slot for, which is once
    /// per frame per space between flushes and remaps; sized for the whole
    /// pool up front, so it never grows or rehashes.
    by_key: std::collections::HashMap<CodeKey, PageSlot>,
    /// One frame table per address space: see [`Space`]. A machine whose CPUs
    /// all see memory at the same addresses has one; a machine of several
    /// nodes has one per node.
    spaces: Vec<Space>,
    /// Pool capacity, fixed at construction (== `pages.len()`). With the free
    /// list exhausted, a claim recycles a slot nobody is using
    /// ([`Self::recycle_slot`]).
    capacity: usize,
    /// The slot each CPU is on: see [`CpuSlot`]. Boxed, so the executors'
    /// pointers into it stay good when the `Jitv2` moves.
    cpu_slots: Box<[CpuSlot]>,
    /// Where the recycling hand stands: the next slot it looks at.
    hand: usize,
    /// Compile-request fifo + worker thread (§6.4/§9). Constructed alongside
    /// the pool; `start()`/`stop()` are separate calls so the executor
    /// controls the compile thread's lifetime independently (mirrors
    /// `MipsCpu`'s own start/stop, e.g. no compile thread while paused).
    pub compile_queue: CompileQueue,
    /// The one `Codegen` (and therefore its one Cranelift `JITModule`/memory
    /// arena) shared between `jitv2_inline_compile` and the async compile
    /// thread — the two modes are mutually exclusive at any given moment (a
    /// monitor command, `j2 inline on|off`, is the only way to switch), so
    /// there is no reason for each to carry its own separate arena that then
    /// each needs its own independent flush bookkeeping. `None` exactly when
    /// the compile-thread worker currently owns it (moved out by
    /// `CompileQueue::start`, handed back by `stop`); `Some` otherwise —
    /// whenever inline mode is what's actually running, or the queue simply
    /// isn't started. `flush` operates on whichever `Some` it finds; callers
    /// on the inline path (`exec_decoded`'s dispatch gate,
    /// `jitv2_track_pcp`'s pool-exhaustion handler) take it via
    /// `codegen.lock()` for the duration of one compile/reset, same
    /// exclusion discipline the compile thread already has by construction
    /// (only it ever touches its own moved-out copy).
    pub codegen: Mutex<Option<crate::cpu::jitv2::codegen::Codegen>>,

    /// Event counters, read only under `j2 status` (dev-only display — see
    /// `JitStats`'s own doc comment for why the *fields themselves* still
    /// exist and get threaded through unconditionally: it's cheaper to carry
    /// one always-present `Arc` clone than to duplicate every function on
    /// its path behind `#[cfg(feature = "developer")]`). `Arc`, not embedded
    /// by value, so `CompileQueue::start` can clone a handle into the worker
    /// thread once (same pattern as `function_count`/`cpu`/`jitv2` below)
    /// instead of every `handle_request` call locking `Jitv2` just to touch
    /// a counter.
    pub stats: Arc<JitStats>,
}

/// Per-instruction-count bucket of a `Jitv2::code_size_by_instr_count()`
/// scan: how many published entries landed at this instruction count, and
/// the raw (not host-page-rounded — that rounding only matters for the
/// arena-usage estimate, `code_bytes_used`; here we want the real
/// per-function code size Cranelift actually emitted) `code_size` bytes
/// across them, as count/sum/min/max — enough to report both an average and
/// a spread without keeping every individual size around.
/// Not `developer`-gated: `code_size_by_instr_count` is the
/// emitted-code-size metric and has to be readable in a production-shaped
/// build — see `PhysicalCodePage::instr_count`'s field doc for why measuring
/// it under `developer` gives a ~4x wrong answer.
#[derive(Debug, Clone, Copy)]
pub struct CodeSizeBucket {
    pub count: u32,
    pub sum_bytes: u64,
    pub min_bytes: u32,
    pub max_bytes: u32,
}

/// One address space's view of the pool: `frame -> slot` for the frames its
/// CPUs have reached, the lock-free path a page switch takes. It only caches
/// what the bus and `by_key` say, so a remap empties it and nothing else.
struct Space {
    /// Frames of memory the table covers. Zero until a CPU sets it.
    frames: usize,
    /// Whether an uncached fetch of a frame has its own entry ([`page_key`]):
    /// on a machine where it can decode to other memory than a cached one.
    split: bool,
    /// Empty until the first claim in this space.
    map: PfnMap,
}

impl Default for Space {
    fn default() -> Self {
        Space { frames: 0, split: false, map: PfnMap(Box::new([])) }
    }
}

impl Space {
    fn entries(&self) -> usize {
        if self.split { 2 * self.frames } else { self.frames }
    }
}

/// Where a CPU fetched from: what [`Jitv2::claim_page`] finds a page for.
#[derive(Clone, Copy, Debug)]
pub struct PageAt {
    /// The CPU's number in the pool.
    pub cpu: usize,
    /// Its address space.
    pub space: usize,
    pub pfn: Pfn,
    /// Fetched uncached, on a machine that keeps that apart.
    pub uncached: bool,
}

/// What [`Jitv2::claim_page`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claim {
    Page(PageSlot),
    /// Every slot is in use (a CPU is on it, or its compile is in flight):
    /// run the page in the interpreter and ask again at the next fetch.
    Full,
}

/// The workers read `pages` through raw pointers, and the fields drop in
/// declaration order, so `pages` would be freed ahead of `compile_queue`'s
/// own drop. Stop and join the workers first.
impl Drop for Jitv2 {
    fn drop(&mut self) {
        self.compile_queue.stop();
    }
}

impl Jitv2 {
    /// Stop the CPUs that run on `jit` and its compile pool for good, before
    /// the machine frees the memory and devices their requests point into.
    /// A queued request holds its page's generation pointer, into the
    /// machine's RAM, and the pool outlives the machine's own drop.
    ///
    /// Not `CompileQueue::stop` under the lock: a worker leading a flush
    /// takes that lock, and restarts the CPUs when the flush is done, so
    /// stop both until neither comes back.
    pub fn stop_for_drop(jit: &Mutex<Jitv2>, stop_cpus: impl Fn()) {
        loop {
            stop_cpus();
            let workers = {
                let mut g = jit.lock();
                g.compile_queue.barrier.stop();
                std::mem::take(&mut g.compile_queue.threads)
            };
            if workers.is_empty() {
                break;
            }
            for w in workers {
                let _ = w.join();
            }
        }
        let mut g = jit.lock();
        g.compile_queue.drain_pending_queue();
        g.buses().clear();
    }

    /// Stop the compile pool from a thread that runs no CPU (the monitor's
    /// `j2 inline on`) and hand back the workers' `Codegen`s, as
    /// `CompileQueue::stop` does, but join them with `jit` unlocked: a worker
    /// leading a flush takes that lock, and would wait on it for ever.
    pub fn stop_pool(jit: &Mutex<Jitv2>) -> Vec<crate::cpu::jitv2::codegen::Codegen> {
        let workers = {
            let mut g = jit.lock();
            g.compile_queue.barrier.stop();
            std::mem::take(&mut g.compile_queue.threads)
        };
        let stopped = workers.into_iter().filter_map(|w| w.join().ok()).collect();
        jit.lock().compile_queue.drain_pending_queue();
        crate::cpu::jit_feedback::JIT_FEEDBACK.set_queue_fill(0, COMPILE_QUEUE_CAPACITY);
        stopped
    }

    /// Allocate the full page pool up front, `capacity` slots, every one
    /// starting unclaimed (`PhysicalCodePage::new(UNCLAIMED_PFN, null)`) —
    /// see `Jitv2`'s own doc comment for why this is a single one-shot
    /// allocation rather than lazy `push`-based growth. Sizing is a Phase 0
    /// measurement per the design doc (§9); start with whatever the caller
    /// passes and let `mega_flush` absorb a too-small guess rather than
    /// trying to size this perfectly up front — getting it wrong now just
    /// means an earlier flush, not a correctness problem. Does not start the
    /// compile thread — call `compile_queue.start()`.
    pub fn new(capacity: usize) -> Self {
        Self::with_frames(capacity, PFN_MAP_ENTRIES, false)
    }

    /// [`Self::new`] with a page table that covers a machine's memory.
    pub fn for_memory(capacity: usize, memory: &crate::memory_map::MemoryMap) -> Self {
        Self::with_frames(capacity, frames_for(memory) as usize, false)
    }

    /// The buses of the CPUs that share this pool: see [`CodeBuses`].
    pub fn buses(&self) -> &Arc<CodeBuses> {
        &self.compile_queue.buses
    }

    /// Describe address space `space`: the frames of memory its table covers,
    /// and whether an uncached fetch of a frame is kept apart from a cached
    /// one. Each CPU says so for its own space, before the space holds a page.
    pub fn set_space(&mut self, space: usize, frames: usize, split: bool) {
        if self.spaces.len() <= space {
            self.spaces.resize_with(space + 1, Space::default);
        }
        let sp = &mut self.spaces[space];
        if sp.frames == frames && sp.split == split {
            return;
        }
        assert!(sp.map.0.is_empty(), "an address space is laid out before it holds a page");
        sp.frames = frames;
        sp.split = split;
    }

    /// Keep an uncached page of a frame apart from its cached page, in space 0.
    /// Before any page is claimed.
    #[cfg(test)]
    pub fn split_uncached_code(&mut self) {
        let frames = self.spaces[0].frames;
        self.set_space(0, frames, true);
    }

    fn with_frames(capacity: usize, frames: usize, split: bool) -> Self {
        let mut pages: Box<[PhysicalCodePage]> = (0..capacity).map(|_| PhysicalCodePage::new(UNCLAIMED_PFN, std::ptr::null())).collect();
        // Thread every slot onto the free list up front, index order —
        // `page_for`'s claim path only ever pops from the head, so initial
        // order just decides which physical slot backs the first `capacity`
        // claims, which doesn't matter.
        for i in 0..capacity {
            pages[i].next = if i + 1 < capacity { (i + 1) as u32 } else { NO_SLOT };
        }
        assert!(capacity <= MAX_POOL_SLOTS, "a JIT page pool holds at most {MAX_POOL_SLOTS} pages");
        let mut compile_queue = CompileQueue::new();
        compile_queue.set_pages(&pages);
        Self {
            cpu_slots: (0..POOL_CPUS).map(|_| CpuSlot(AtomicU32::new(NO_SLOT))).collect(),
            hand: 0,
            pages,
            free_head: if capacity > 0 { 0 } else { NO_SLOT },
            by_key: std::collections::HashMap::with_capacity(capacity),
            spaces: vec![Space { frames, split, map: PfnMap(Box::new([])) }],
            capacity,
            compile_queue,
            codegen: Mutex::new(Some(crate::cpu::jitv2::codegen::Codegen::new())),
            stats: Arc::new(JitStats::default()),
        }
    }

    /// The pool slot for frame `pfn` of a flat bus, as CPU 0 in space 0
    /// reaches it cached: [`Self::claim_page`] for a pool with one CPU.
    /// `None` if the pool is exhausted.
    pub fn page_for(&mut self, pfn: Pfn, _phys_addr: u64, bus: &dyn BusDevice, fr1: bool) -> Option<PageSlot> {
        self.page_at(PageAt { cpu: 0, space: 0, pfn, uncached: false }, bus, fr1)
    }

    /// [`Self::claim_page`] for a caller with nothing to do about a full pool
    /// (the monitor): the slot, or `None`.
    pub fn page_at(&mut self, at: PageAt, bus: &dyn BusDevice, fr1: bool) -> Option<PageSlot> {
        match self.claim_page(at, bus, fr1) {
            Claim::Page(slot) => Some(slot),
            Claim::Full => None,
        }
    }

    /// The pool slot for the page a CPU fetched from at `at`, through `bus`
    /// (the bus that decodes that fetch: its cached one, or its own for an
    /// uncached fetch on a machine that keeps those apart). The bus names the
    /// memory there ([`BusDevice::code_page`]); the page for that memory is
    /// claimed if no CPU has reached it yet, at whatever address. Where no
    /// bank answers, the page is keyed by the address, within this space
    /// ([`CodeKey::at_address`]), and counted as the bus counts it. Records
    /// where this CPU reaches it and enters it in the space's frame table.
    ///
    /// [`Claim::Full`] if the pool is exhausted — the caller (mips exec
    /// thread) is responsible for running `mega_flush` and retrying.
    pub fn claim_page(&mut self, at: PageAt, bus: &dyn BusDevice, fr1: bool) -> Claim {
        let space = &mut self.spaces[at.space];
        let uncached = at.uncached && space.split;
        let entry = page_key(at.pfn, uncached, space.split);
        if space.map.0.is_empty() {
            space.map = PfnMap::new(space.entries());
        }
        // An entry made for a tenant the slot no longer has is as good as none.
        if let Some(e) = space.map.get(entry) {
            let slot = frame_entry_slot(e);
            let page = &self.pages[slot as usize];
            if frame_entry_current(e, page.tenancy()) {
                self.cpu_slots[at.cpu].0.store(slot, Ordering::SeqCst);
                page.set_addr(at.cpu, at.pfn, uncached);
                page.touch();
                return Claim::Page(slot);
            }
        }
        let addr = at.pfn as u64 * PAGE_SIZE as u64;
        let code = bus.code_page(addr).unwrap_or_else(|| crate::traits::CodePage {
            key: CodeKey::at_address(at.space, at.pfn, uncached),
            gen: bus.gen_ptr(addr),
            words: std::ptr::null(),
        });
        let slot = match self.by_key.get(&code.key) {
            Some(&slot) => slot,
            None => {
                let slot = if self.free_head != NO_SLOT {
                    let slot = self.free_head;
                    self.free_head = self.pages[slot as usize].next;
                    slot
                } else {
                    match self.recycle_slot() {
                        Some(slot) => slot,
                        None => return Claim::Full,
                    }
                };
                self.pages[slot as usize].claim_code(code, fr1);
                self.pages[slot as usize].next = NO_SLOT;
                self.by_key.insert(code.key, slot);
                slot
            }
        };
        debug_assert_eq!(self.pages[slot as usize].key, code.key, "the key index and slot {slot} disagree");
        // Published under the pool's lock, which the recycler holds: no slot
        // can be taken between this claim and the CPU using it.
        self.cpu_slots[at.cpu].0.store(slot, Ordering::SeqCst);
        let page = &self.pages[slot as usize];
        page.set_addr(at.cpu, at.pfn, uncached);
        page.touch();
        let tenancy = page.tenancy();
        self.spaces[at.space].map.insert(entry, frame_entry(slot, tenancy));
        Claim::Page(slot)
    }

    /// Take a slot back from memory nobody is running, for a claim that
    /// found the free list empty. `None` if every slot is in use.
    ///
    /// A clock: the hand passes over a page entered since its last pass
    /// (clearing the mark), a page some CPU is on, and a page whose compile
    /// request is queued or running (the worker reads the page's key, bytes
    /// and generation pointer through it; see
    /// [`PhysicalCodePage::pin_for_compile`] for the worker's half). The first page left is emptied
    /// and handed out.
    ///
    /// Order matters. The tenancy moves first, which stops every lock-free
    /// lookup from matching the slot; only then are the CPUs' slot words
    /// read. A CPU that had already matched has published the slot and is
    /// seen; one that publishes later sees the new tenancy and comes to the
    /// pool's lock, which the caller holds. A CPU can only schedule a compile
    /// for a page it is on, so once none is on it the flag is final too.
    fn recycle_slot(&mut self) -> Option<PageSlot> {
        let n = self.pages.len();
        // Two passes: the first may only clear marks.
        for _ in 0..2 * n {
            let slot = self.hand;
            self.hand = (self.hand + 1) % n;
            let page = &self.pages[slot];
            if page.take_referenced() || page.is_scheduled() || self.cpu_on(slot as PageSlot) {
                continue;
            }
            let tenancy = page.end_tenancy();
            if self.cpu_on(slot as PageSlot) || self.pages[slot].page_scheduled.load(Ordering::SeqCst) {
                // Lost the race to a CPU. The page stays, under its new
                // tenancy: its frame table entries are made again on demand.
                continue;
            }
            if tenancy & ((1 << (32 - SLOT_BITS)) - 1) == 0 {
                // The tenancy bits of a frame table entry have wrapped: an
                // entry made 65536 tenants ago would match again. None may
                // survive that long.
                for space in &mut self.spaces {
                    space.map.clear();
                }
            }
            let key = self.pages[slot].key;
            self.by_key.remove(&key);
            self.pages[slot].reset_to_unclaimed();
            self.stats.pages_recycled.fetch_add(1, Ordering::Relaxed);
            return Some(slot as PageSlot);
        }
        None
    }

    /// Whether some CPU has published `slot` as the one it is on.
    fn cpu_on(&self, slot: PageSlot) -> bool {
        self.cpu_slots.iter().any(|c| c.0.load(Ordering::SeqCst) == slot)
    }

    /// CPU `cpu`'s word in the pool's table of current slots: see [`CpuSlot`].
    pub fn cpu_slot_ptr(&self, cpu: usize) -> *const AtomicU32 {
        &self.cpu_slots[cpu].0
    }

    /// Memory moved under the addresses of `space`: forget which frame holds
    /// which page there, and where CPU `cpu` reached each page. The pages and
    /// their code stay: they are keyed by the memory, not by its address, so
    /// the next fetch finds them again wherever the memory went. Each CPU of
    /// the space calls this for itself before it runs guest code again.
    pub fn forget_addresses(&mut self, space: usize, cpu: usize) {
        self.spaces[space].map.clear();
        for page in self.pages.iter().filter(|p| p.is_claimed()) {
            page.forget_addr(cpu);
        }
        self.stats.remaps.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn table_bytes(&self) -> usize {
        self.spaces.iter().map(Space::entries).sum::<usize>() * size_of::<AtomicU32>()
    }

    /// Pointers for lock-free `frame -> *mut PhysicalCodePage` fast lookup in
    /// `space`.
    pub fn fast_lookup_ptrs(&mut self, space: usize) -> (*const AtomicU32, *mut PhysicalCodePage) {
        (self.pfn_map_ptr(space), self.pages.as_mut_ptr())
    }

    /// `space`'s frame table pointer, null before its first claim.
    pub fn pfn_map_ptr(&self, space: usize) -> *const AtomicU32 {
        let map = &self.spaces[space].map;
        if map.0.is_empty() { std::ptr::null() } else { map.as_ptr() }
    }

    /// Raw pointer to the page at `slot`. Valid for the process's entire
    /// lifetime (see the `pages` field doc — the array itself never moves or
    /// resizes, including across `mega_flush`) — used to set the executor's
    /// current-PCP pointer without holding a borrow of `self`.
    #[inline]
    pub fn page_ptr(&mut self, slot: PageSlot) -> *mut PhysicalCodePage {
        &mut self.pages[slot as usize] as *mut PhysicalCodePage
    }

    /// Number of pool slots currently claimed (since construction or the
    /// last `mega_flush`). Exit-time diagnostic — see `MipsCpu::stop`.
    ///
    /// Scans `pages` rather than consulting the pfn map: [`PfnMap`] is a
    /// direct-mapped array with no live-entry count of its own, while
    /// `pfn != UNCLAIMED_PFN` is the authoritative claimed test. O(capacity),
    /// and this is a diagnostic, not a hot path.
    ///
    /// Reads 0 immediately after a `mega_flush`, which resets every slot —
    /// so this is "claimed right now", not a cumulative total. It is NOT
    /// comparable against `Codegen::function_count`, which resets on a
    /// Cranelift arena flush instead; the two count over different windows.
    #[inline]
    pub fn pages_used(&self) -> usize {
        self.pages.iter().filter(|p| p.is_claimed()).count()
    }

    /// Pool-wide churn-avoidance totals as `(skipped, rejected)`: compile
    /// requests answered by re-validating already-installed code, vs. those
    /// that had a snapshot to compare against and had to compile anyway.
    pub fn redundant_compile_totals(&self) -> (u64, u64) {
        let (mut skipped, mut rejected) = (0u64, 0u64);
        for page in self.pages.iter().filter(|p| p.is_claimed()) {
            skipped += page.redundant_skipped() as u64;
            rejected += page.redundant_rejected() as u64;
        }
        (skipped, rejected)
    }

    /// Pages more than one CPU of the pool has reached, and how many of those
    /// some CPU reached at another frame than the rest (another node's address
    /// for the same memory): the sharing a key by memory makes possible.
    pub fn shared_pages(&self) -> (usize, usize) {
        let (mut shared, mut apart) = (0, 0);
        for page in self.pages.iter().filter(|p| p.is_claimed()) {
            let mut frames = (0..POOL_CPUS).filter_map(|cpu| page.addr_of(cpu)).map(|(pfn, _)| pfn);
            let Some(first) = frames.next() else { continue };
            let rest: Vec<Pfn> = frames.collect();
            if !rest.is_empty() {
                shared += 1;
                apart += usize::from(rest.iter().any(|&pfn| pfn != first));
            }
        }
        (shared, apart)
    }

    /// Pool capacity, as passed to `new()`.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Sum, across every pooled page with a published function, of that
    /// page's `code_size` — diagnostic for `j2 status`.
    ///
    /// **Not rounded.** The page-rounding this used to do modelled
    /// `ArenaMemoryProvider`, which gave every function its own
    /// page-rounded segment; that provider no longer exists. The live
    /// `PagedArenaMemoryProvider` packs allocations back to back, so a
    /// function costs about its own `code_size`, and on a 16 KiB-page host
    /// the old rounding over-reported a ~215-byte function by about 76x.
    ///
    /// NOT `#[cfg(feature = "developer")]`, though the commit this came
    /// from added that: `j2 status` is read on `lightning` builds, and
    /// gating a statistic to `developer` is how `last_code_size` came to
    /// report 0 bytes in the only build worth measuring (see
    /// clanker/build/the-three-builds-we-actually-use.md).
    pub fn code_bytes_used(&self) -> u64 {
        self.pages.iter()
            .filter(|page| page.has_code())
            .map(|page| page.code_size.load(Ordering::Relaxed) as u64)
            .sum()
    }

    /// Histogram of `PhysicalCodePage::instr_count` across every pooled page
    /// with a published function, paired with each bucket's code-size
    /// distribution — dev-only diagnostic (`j2 status`). Indexed by
    /// instruction count directly (`result[n]` = stats for published pages
    /// whose most recent compile covered exactly `n` instructions); `n == 0`
    /// is always absent. §13: page-granular now, not per-offset — see
    /// `PhysicalCodePage::instr_count`'s own field doc for why per-offset
    /// overlap accounting no longer applies (overlap between entries is
    /// structurally impossible under the one-function-per-page design).
    pub fn code_size_by_instr_count(&self) -> Vec<Option<CodeSizeBucket>> {
        let mut hist: Vec<Option<CodeSizeBucket>> = Vec::new();
        for page in self.pages.iter() {
            if !page.has_code() { continue; }
            let n = page.instr_count.load(Ordering::Relaxed) as usize;
            let size = page.code_size.load(Ordering::Relaxed);
            if n >= hist.len() { hist.resize(n + 1, None); }
            match &mut hist[n] {
                Some(bucket) => {
                    bucket.count += 1;
                    bucket.sum_bytes += size as u64;
                    bucket.min_bytes = bucket.min_bytes.min(size);
                    bucket.max_bytes = bucket.max_bytes.max(size);
                }
                slot @ None => {
                    *slot = Some(CodeSizeBucket { count: 1, sum_bytes: size as u64, min_bytes: size, max_bytes: size });
                }
            }
        }
        hist
    }

    /// Every claimed page in the pool, exposed as a borrowing iterator
    /// rather than folded into an aggregate, for callers (`j2 html`) that
    /// need the raw per-page detail instead of a summary statistic. Not
    /// necessarily a contiguous `pages[..n]` prefix, hence the full scan
    /// filtered on
    /// [`PhysicalCodePage::is_claimed`]. `pages` itself stays private (index
    /// stability, see the field's own doc comment, is an invariant only this
    /// module should rely on).
    pub fn claimed_pages(&self) -> impl Iterator<Item = &PhysicalCodePage> {
        self.pages.iter().filter(|p| p.is_claimed())
    }

    /// Reset the whole page pool to empty — the MAME-style "flush the world"
    /// response to running out of any bump-allocated JIT resource (§6.3's
    /// `flush_all()`: arena-full, `restore`, `rollback` all route through
    /// this one routine).
    ///
    /// Every slot gets [`PhysicalCodePage::reset_to_unclaimed`] (which clears
    /// `pfn` back to `UNCLAIMED_PFN`, so `pages_used()` reads 0 afterwards),
    /// every frame table and `by_key` are cleared, and the free list is relinked across the
    /// entire array from slot 0. Slot *indices* stay stable — the array is
    /// reset in place and never reallocates (see the `pages` field's own doc
    /// comment) — but a slot's *contents* do not survive, so any surviving
    /// pointer to one must re-check pfn/gen before trusting it.
    ///
    /// An earlier design preserved the `pfn`/`requested`/`denied` of the N
    /// most-recently-used pages across a flush and immediately requeued them
    /// for recompilation, to avoid re-warming the page cache from nothing.
    /// That was removed: the preserved entry points could be stale, and
    /// accumulating bad ones was measured as at best neutral and slightly
    /// harmful. The MRU list it depended on is gone with it.
    ///
    /// Does not yet demote promoted decode-entry handlers or null
    /// entry_table slots (§6.1.3, §6.3) — there are none to demote until
    /// the dispatcher/compiler land. Once they exist, this is where that
    /// walk goes, on the executor thread, before the reset loop below.
    ///
    /// Private: real callers want [`Self::flush_from_cpu_thread`]/
    /// [`Self::flush_from_jit_thread`], which wrap this with the compile-queue
    /// pause/restart and requeue every caller of this used to have to
    /// remember (see those methods' doc comments for why bundling it there,
    /// rather than leaving it the caller's responsibility, is what makes the
    /// whole operation self-contained now that `Jitv2` owns its own
    /// compile-queue lifecycle independently of `MipsCpu::stop()`/`start()`).
    fn mega_flush(&mut self) {
        crate::cpu::jitv2::hashstats::FLUSH_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for space in &mut self.spaces {
            space.map.clear();
        }
        self.by_key.clear();
        let cap = self.pages.len();
        for (i, page) in self.pages.iter_mut().enumerate() {
            page.reset_to_unclaimed();
            page.next = (i + 1) as u32;
        }
        if cap > 0 {
            self.pages[cap - 1].next = NO_SLOT;
            self.free_head = 0;
        } else {
            self.free_head = NO_SLOT;
        }
        // No CPU is on a page: each drops its own pointer when it resumes.
        for c in self.cpu_slots.iter() {
            c.0.store(NO_SLOT, Ordering::SeqCst);
        }
        self.hand = 0;
        crate::cpu::jit_feedback::JIT_FEEDBACK.set_arena_fill(0, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
        crate::cpu::jit_feedback::JIT_FEEDBACK.record_flush();
    }

    /// Self-contained page-pool + compiled-code-arena flush, called FROM the
    /// CPU thread (`jitv2_track_pcp`'s pool-exhaustion handler).
    pub unsafe fn flush_from_cpu_thread(&mut self, bus: Arc<dyn BusDevice>) {
        let stopped = self.compile_queue.stop();
        let was_threaded = !stopped.is_empty();
        self.compile_queue.drain_pending_queue();
        // Inline mode compiles through the idle `Codegen`, which sits on one
        // of the ring's arenas once the pool has ever run (`j2 inline on`
        // takes the active one), or on an arena of its own otherwise.
        let inline_on_ring = !was_threaded
            && self.codegen.get_mut().as_ref().is_some_and(|c| self.compile_queue.owns_arena(&c.shared_arena().0));
        let function_count = if was_threaded {
            stopped.iter().map(|c| c.function_count()).sum()
        } else {
            self.codegen.get_mut().as_ref().map_or(0, |c| c.function_count())
        };
        eprintln!(
            "jitv2: mega_flush (from cpu thread) — {} / {} pages used, {} functions compiled",
            self.pages_used(), self.capacity(), function_count,
        );
        self.mega_flush();
        // Every page that could reach compiled code was just cleared, the
        // workers are joined, and this runs on the CPU thread outside
        // compiled code with the other CPUs parked: every arena can go.
        drop(stopped);
        if was_threaded || inline_on_ring {
            unsafe { self.compile_queue.reset_ring(); }
        }
        if inline_on_ring {
            if let (Some(codegen), Some((arena, state))) = (self.codegen.get_mut().as_mut(), self.compile_queue.active_arena()) {
                unsafe { codegen.reset_with_shared_arena(arena, state); }
            }
        } else if !was_threaded {
            if let Some(codegen) = self.codegen.get_mut().as_mut() {
                unsafe { codegen.reset(); }
            }
        }
        if was_threaded {
            let stats = self.stats.clone();
            self.compile_queue.start(bus, stats);
        }
    }

}

/// One of the compile pool's code arenas, and its state.
type PoolArena = (
    Arc<parking_lot::Mutex<crate::cpu::jitv2::paged_memory::SharedArena>>,
    Arc<crate::cpu::jitv2::paged_memory::PagedArenaState>,
);

/// How many code arenas the compile pool rotates through: the one being
/// compiled into, the one before it, whose code still runs, and one empty.
/// See `arena_ring`.
pub const ARENA_COUNT: usize = 3;

/// Build the memory helpers into `arena`, which must be empty and have no
/// worker on it: they go at its lowest addresses and are sealed at once (see
/// `Codegen::emit_mem_helpers`). Every `Codegen` built on the arena
/// afterwards calls them.
fn build_arena_helpers(
    arena: &PoolArena,
    dc_geometry: crate::cpu::mips_cache_v2::JitDcGeometry,
    jit_consts: crate::cpu::jitv2::codegen::JitConsts,
) {
    let mut codegen = crate::cpu::jitv2::codegen::Codegen::new_with_shared_arena(arena.0.clone(), arena.1.clone());
    codegen.dc_geometry = dc_geometry;
    codegen.jit_consts = jit_consts;
    codegen.emit_mem_helpers();
}

/// The page pool's slots, for the worker that retires an arena. The array is
/// allocated once and never moves (`Jitv2::pages`), and the sweep touches
/// only what a page guards with its own locks and atomics.
#[derive(Clone, Copy)]
struct PageSlots {
    base: *const PhysicalCodePage,
    len: usize,
}

// SAFETY: see the type; `PhysicalCodePage` is `Sync`.
unsafe impl Send for PageSlots {}
unsafe impl Sync for PageSlots {}

impl PageSlots {
    const NONE: PageSlots = PageSlots { base: std::ptr::null(), len: 0 };

    /// Take the code of the arena of epoch `epoch` off every page. Returns
    /// how many pages had some.
    fn retire_code(&self, epoch: u64) -> usize {
        (0..self.len).filter(|&i| unsafe { &*self.base.add(i) }.retire_code(epoch)).count()
    }
}

/// The compile thread pool and its inbound MPMC queue (§6.4/§9 Phase 1:
/// "mips exec thread pushes work via fifo, jit thread(s) compile").
///
/// One shared `crossbeam_queue::ArrayQueue` (`queue`) — the CPU thread
/// pushes, every worker thread pops from the same instance, no
/// producer/consumer split needed. The queue is wrapped in a [`CompileLane`],
/// which carries the pool's wake-up channel so a push also wakes the
/// workers. Restartable: `stop()` joins every worker
/// thread and collects their `Codegen`s, but the queue itself (an `Arc`) is
/// never taken apart — it just sits idle (still holding whatever's left
/// un-popped) between a `stop()` and the next `start()`. This is what lets
/// `mega_flush` stop-drain-restart the pool around a page-pool reset instead
/// of the queue being single-use (see `mega_flush`'s call site in
/// `mips_exec.rs`, `jitv2_track_pcp`, for why that's required: every
/// `CompileRequest::page` in flight or still queued points into the `Vec`
/// `mega_flush` is about to clear, so every worker must be fully joined,
/// and the queue drained, before that clear happens).
///
/// Shared body of `CompileQueue::send`: push `req` onto the raw queue,
/// record `stats` counters under `developer`, and wake the pool. Free
/// function (not a `CompileQueue` method) so `MipsExecutor` can call it
/// directly through a bare `Arc<CompileLane>` (`CompileQueue::queue_handle`)
/// without going through `Jitv2`'s outer mutex at all — the hot dispatch-gate
/// path only ever needs this one push, never the pool/codegen state that
/// mutex actually protects.
pub(crate) fn push_compile_request(
    queue: &Arc<CompileLane>,
    req: CompileRequest,
    #[allow(unused_variables)] stats: &JitStats,
) -> bool {
    #[cfg(feature = "developer")]
    {
        // len() before push(): occupancy right now, not after this one
        // lands — matches "depth the compile pool is running at" rather
        // than counting this dispatch's own contribution to it.
        let occupancy = queue.len();
        stats.compile_queue_dispatches.fetch_add(1, Ordering::Relaxed);
        stats.compile_queue_depth_sum.fetch_add(occupancy as u64, Ordering::Relaxed);
    }
    let accepted = queue.push(req).is_ok();
    // Wake a sleeping worker, or it only notices the request on its next
    // backstop tick, which is exactly the latency the park replaces. Not on
    // a dropped push: the queue is full, so no worker is waiting for work,
    // and a full queue is when this runs most often — every dispatch of an
    // uncompiled page retries the push, on the CPU thread.
    if accepted {
        queue.wake.wake_if_sleeping();
    }
    #[cfg(feature = "developer")]
    if !accepted {
        stats.compile_queue_full.fetch_add(1, Ordering::Relaxed);
    }
    accepted
}

/// Start compiling for `cpu` the way `mode` asks. `Async` starts the pool.
/// `Sync` leaves it stopped and has the CPU compile at its dispatch gate with
/// the `Codegen` that `Jitv2` keeps idle for that, as the monitor's `j2 inline
/// on` does at run time. Call with no lock on `jit` held.
#[cfg(not(feature = "jitv2_lockstep"))]
pub fn start_compiling(
    cpu: &dyn crate::cpu::mips_exec::CpuDevice,
    jit: &Mutex<Jitv2>,
    mode: crate::config::JitCompile,
    bus: Arc<dyn BusDevice>,
) {
    match mode {
        crate::config::JitCompile::Async => {
            let mut g = jit.lock();
            let stats = g.stats.clone();
            g.compile_queue.start(bus, stats);
        }
        crate::config::JitCompile::Sync => cpu.compile_on_cpu_thread(),
    }
}

/// Give `cpu`'s JIT its compile pool and start compiling as `mode` asks,
/// reading code through `bus`. The pool's flushes park the CPUs through
/// `world`, the machine's, which every CPU on the pool shares
/// (`CpuDevice::share_jitv2`). Under lockstep no pool starts.
pub fn attach_pool(
    cpu: &Arc<dyn crate::cpu::mips_exec::CpuDevice>,
    (threads, threads_reason): (usize, String),
    mode: crate::config::JitCompile,
    bus: Arc<dyn BusDevice>,
    world: Arc<crate::cpu::smp::World>,
) {
    // Weak handles only: a strong one would keep the board alive after the machine.
    let device: Arc<dyn Device> = cpu.clone().as_device();
    let jit = cpu.jitv2();
    {
        let mut g = jit.lock();
        g.compile_queue.set_thread_count_with_reason(threads.max(1), &threads_reason);
        g.compile_queue.set_cpu(Arc::downgrade(&device));
        g.compile_queue.set_owner(Arc::downgrade(&jit));
        g.compile_queue.set_world(world);
    }
    #[cfg(not(feature = "jitv2_lockstep"))]
    start_compiling(&**cpu, &jit, mode, bus);
    #[cfg(feature = "jitv2_lockstep")]
    let _ = (mode, bus);
}

/// What [`share_pool`] needs of one node: its processors, the module they
/// are, and the bus they read cached code through.
pub struct PoolNode<'a> {
    pub cpus: &'a [Arc<dyn crate::cpu::mips_exec::CpuDevice>],
    pub module: &'a crate::cpu::parts::CpuModule,
    pub bus: Arc<dyn BusDevice>,
}

/// Give a machine of several nodes its JIT: one page pool, one compile queue
/// and one body of compiled code for every processor, whichever node it is
/// on. A page is keyed by the memory it was compiled from, so the node whose
/// processor reaches that memory at another address runs the same code. Each
/// node is an address space of the pool, with its own frame table.
///
/// Compiled code is specific to the processor module (its instruction set,
/// its cache geometry), so nodes with different modules get a pool each.
pub fn share_pool(
    nodes: &[PoolNode],
    pool_size: (usize, String),
    mode: crate::config::JitCompile,
    world: &Arc<crate::cpu::smp::World>,
) -> Result<(), String> {
    let mut done = vec![false; nodes.len()];
    for first in 0..nodes.len() {
        if done[first] {
            continue;
        }
        let members: Vec<usize> = (first..nodes.len()).filter(|&n| !done[n] && nodes[n].module == nodes[first].module).collect();
        members.iter().for_each(|&n| done[n] = true);
        let group: Vec<&PoolNode> = members.iter().map(|&n| &nodes[n]).collect();
        let Some(lead) = group.iter().find_map(|n| n.cpus.first()) else {
            continue;
        };
        let several = group.iter().map(|n| n.cpus.len()).sum::<usize>() > 1;
        if several && cfg!(feature = "jitv2_lockstep") {
            return Err("several processors cannot run with jitv2_lockstep".into());
        }
        // A lone processor shares its own pool too, so the pool's flushes go
        // through the machine's world: a CPU belongs to one.
        let jit = lead.jitv2();
        let mut number = 0;
        for (space, node) in group.iter().enumerate() {
            for cpu in node.cpus {
                cpu.share_jitv2(jit.clone(), world.clone(), number, space);
                number += 1;
            }
        }
        attach_pool(lead, pool_size.clone(), mode, group[0].bus.clone(), world.clone());
    }
    Ok(())
}

pub struct CompileQueue {
    /// The ISA level every worker's `Analyzer` walks at — the `CpuModel`'s
    /// `MIPS4` const, set by `MipsExecutor::new` through [`set_isa`] before
    /// the pool starts. Defaults to the process-wide value so tools and unit
    /// tests that never publish one still behave as the build was configured.
    ///
    /// [`set_isa`]: CompileQueue::set_isa
    mips4: bool,

    /// L1-D geometry for the inline load/store fast path, published by the
    /// CPU thread (`MipsExecutor::install_jit_mem_ptrs`) and copied into each
    /// worker's `Codegen` as the worker is spawned. Held here rather than
    /// written straight into `Codegen` because the worker owns that by value
    /// while compiling, and read at spawn so a worker never waits for the
    /// pool: a CPU that flushes again before the workers it just started
    /// have run holds the pool while it joins them.
    pub dc_geometry: Mutex<crate::cpu::mips_cache_v2::JitDcGeometry>,
    /// Compile-time constants for the one core these workers compile for —
    /// see `codegen::JitConsts`. Same publication route as `dc_geometry`.
    pub jit_consts: Mutex<crate::cpu::jitv2::codegen::JitConsts>,

    /// The one shared bounded MPMC ring: the CPU thread pushes
    /// (`CompileQueue::send`), every worker thread pops from the same
    /// `Arc` — no producer/consumer handoff needed (unlike the old
    /// `rtrb`-based SPSC design, `ArrayQueue::push`/`pop` both take `&self`,
    /// so this can just be cloned into each worker thread and left there
    /// permanently, alive across `stop()`/`start()` cycles). Bounded, with
    /// `push` returning `Err` on full rather than blocking — preserves
    /// §6.4's "drop on full, hot pages re-trigger" semantics exactly. The
    /// lane also carries the pool's wake-up channel, so a push wakes the
    /// workers instead of leaving them to poll.
    queue: Arc<CompileLane>,
    /// The hot lane: a small second queue the workers drain before `queue`,
    /// for pages executed hard while they wait. One per pool, as `queue` is:
    /// a request holds raw pointers into its own JIT's pages and its own
    /// machine's memory, so another machine's worker must never take it.
    hot: Arc<CompileLane>,
    /// The buses the workers read page bytes through, one pair per CPU that
    /// shares the pool: see [`CodeBuses`].
    pub buses: Arc<CodeBuses>,
    /// The pool's wake-up channel, shared by both lanes — the same workers
    /// drain both. See [`CompileWake`].
    wake: Arc<CompileWake>,
    /// One `JoinHandle` per worker thread — empty when not running. Each
    /// thread hands back its own `Codegen` on exit (see `worker_loop`'s
    /// return type); `stop()` collects all of them.
    threads: Vec<JoinHandle<crate::cpu::jitv2::codegen::Codegen>>,
    /// Weak handle to the CPU device, so the worker can stop/start it itself
    /// when a memory-growth flush is needed (`Codegen::function_count()`
    /// crossing `CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES` — see `worker_loop`). Set
    /// once, via `set_cpu`, right after `Arc<MipsCpu<T,C>>` is constructed in
    /// `Machine::new` (mirrors `Mc::set_cpu`'s own `Arc::downgrade`
    /// injection — the CPU doesn't exist yet when `Jitv2`/`CompileQueue` are
    /// first built, so this can't be a constructor parameter). `Weak`, not
    /// `Arc`: a strong reference here would be a real cycle (MipsCpu owns the
    /// executor, which owns `Jitv2`, which would own this) that nothing
    /// would ever break — `Weak::upgrade` failing just means the machine is
    /// mid-teardown and there's nothing to stop/start anymore, which is fine
    /// to skip.
    cpu: Mutex<Option<Weak<dyn Device>>>,
    /// How a flush stops the CPUs this pool compiles for: it parks them, and
    /// takes turns with their own flushes (`MipsExecutor::jitv2_flush_from_here`).
    /// Every `MipsCpu` sets one, a world of one CPU when it shares its pool
    /// with no other. `cpu` is not used then: stopping a CPU joins its thread,
    /// which may itself be joining this pool for a flush of its own, and
    /// restarting it would start a CPU the monitor has stopped. Without a
    /// world (a bare `CompileQueue` in a test) the leader stops `cpu`.
    world: Mutex<Option<Arc<crate::cpu::smp::World>>>,
    /// Weak handle back to the owning `Jitv2` (behind whatever `Arc<Mutex<..>>`
    /// the caller shares it in) — the worker needs this to clear the page
    /// pool during its own flush (`worker_loop`'s doc comment explains why
    /// that's safe despite §6.1.3's usual CPU-thread-only contract: the CPU
    /// is provably stopped for the whole operation). Set via `set_owner`,
    /// same injection-after-construction reasoning as `cpu`.
    jitv2: Mutex<Option<Weak<Mutex<Jitv2>>>>,
    /// Per-worker mirror of each worker's own `codegen.function_count()`,
    /// one slot per pool thread, updated after every compile that worker
    /// performs — exists purely so `j2 stats` can read a total function
    /// count while each worker owns its own real `Codegen` by value (not
    /// behind any lock a status command could take without contending the
    /// hot compile path). `function_count()` sums across all slots. Built
    /// fresh (length `thread_count`) by `start_inner` on every start, since
    /// the slot count itself can only change between runs, never while a
    /// pool is live. Each worker's own slot is reset to 0 alongside every
    /// real flush its `Codegen` goes through (`run_leader_flush`/
    /// `PoolBarrier::park`'s resume path) so it never drifts from what
    /// `Codegen::function_count()` would report if you could ask that
    /// worker's real `Codegen` directly.
    function_counts: Vec<Arc<AtomicU32>>,
    /// Fixed pool size — read once by `start()`, never changed at runtime
    /// (no "j2 threads N" mutation; see `set_thread_count`'s own doc
    /// comment). Defaults to 1, so a queue built without a machine (tests,
    /// `equiv_test.rs`) runs one worker unless it asks for more.
    thread_count: usize,
    /// Why the pool is sized as it is — the configuration's auto-sizing
    /// explanation ("auto: one worker per guest CPU ...") or "set
    /// explicitly". `None` for a queue built without a machine (tests,
    /// `equiv_test.rs`).
    thread_reason: Option<String>,
    /// The pool's quiesce barrier and run flag, cloned into every worker:
    /// see `pool_barrier`.
    barrier: Arc<PoolBarrier<()>>,
    /// The code arenas, reserved at the first start and kept for the queue's
    /// life: see `arena_ring`.
    ring: Option<Arc<ArenaRing<PoolArena>>>,
    /// The page pool this queue compiles for, for an arena's retirement.
    pages: PageSlots,
}

/// A queue dropped on its own (not inside a `Jitv2`) still must not leave
/// its workers running: they hold the bus they compile from.
impl Drop for CompileQueue {
    fn drop(&mut self) {
        self.stop();
        // The arenas go with the queue: the CPUs that ran their code are
        // gone by the time their JIT is dropped (`Jitv2::stop_for_drop`).
        if let Some(ring) = self.ring.take() {
            for arena in ring.arenas() {
                unsafe { arena.0.lock().release_ring_owned(); }
            }
        }
    }
}

impl CompileQueue {
    /// Construct the queue without starting any worker threads. Call
    /// [`Self::start`] to spawn the pool.
    pub fn new() -> Self {
        let wake = Arc::new(CompileWake::new());
        Self {
            mips4: crate::cpu::jitv2::isa::mips4_enabled(),
            dc_geometry: Mutex::new(crate::cpu::mips_cache_v2::JitDcGeometry::unsupported()),
            jit_consts: Mutex::new(crate::cpu::jitv2::codegen::JitConsts::default()),
            queue: Arc::new(CompileLane::new(COMPILE_QUEUE_CAPACITY, wake.clone())),
            hot: Arc::new(CompileLane::new(HOT_QUEUE_CAPACITY, wake.clone())),
            buses: Arc::new(CodeBuses::default()),
            wake,
            threads: Vec::new(),
            cpu: Mutex::new(None),
            world: Mutex::new(None),
            jitv2: Mutex::new(None),
            function_counts: Vec::new(),
            thread_count: 1,
            thread_reason: None,
            barrier: Arc::new(PoolBarrier::new()),
            ring: None,
            pages: PageSlots::NONE,
        }
    }

    /// The page pool whose pages this queue's code is published to. An
    /// arena's retirement sweeps it.
    fn set_pages(&mut self, pages: &[PhysicalCodePage]) {
        self.pages = PageSlots { base: pages.as_ptr(), len: pages.len() };
    }

    /// The arena the pool compiles into now. `None` before the first start.
    pub fn active_arena(&self) -> Option<PoolArena> {
        self.ring.as_ref().map(|ring| ring.active().0)
    }

    /// Whether `arena` is one of this queue's ring.
    fn owns_arena(&self, arena: &Arc<parking_lot::Mutex<crate::cpu::jitv2::paged_memory::SharedArena>>) -> bool {
        self.ring.as_ref().is_some_and(|ring| ring.arenas().iter().any(|a| Arc::ptr_eq(&a.0, arena)))
    }

    /// Which arena is doing what, for `j2 status`. `None` before the first start.
    pub fn ring_status(&self) -> Option<crate::cpu::jitv2::arena_ring::RingStatus> {
        self.ring.as_ref().map(|ring| ring.status())
    }

    /// Bytes handed out of each arena, in ring order.
    pub fn arena_fill(&self) -> Vec<u64> {
        self.ring.as_ref().map_or(Vec::new(), |ring| ring.arenas().iter().map(|a| a.1.packing_stats().0).collect())
    }

    /// Empty every arena and start again on the first: for a flush of the
    /// whole page pool.
    ///
    /// # Safety
    /// No worker runs, no page holds code from the ring, and every CPU is
    /// outside compiled code.
    unsafe fn reset_ring(&mut self) {
        let Some(ring) = &self.ring else { return };
        ring.reset(|arena| unsafe { arena.0.lock().recycle() });
        let (geometry, consts) = (*self.dc_geometry.lock(), *self.jit_consts.lock());
        for arena in ring.arenas() {
            build_arena_helpers(arena, geometry, consts);
        }
    }

    /// Set the pool's fixed worker-thread count — must be called before the
    /// queue is ever `start()`-ed (later calls are a no-op, guarded by
    /// `debug_assert!`, since the whole point is "fixed at startup, no
    /// runtime mutation"). Each machine's constructor calls it (`Machine::new`
    /// and the `ip*_machine` builders) with `[jitv2] threads` /
    /// `--jitv2-threads`, which when unset is one thread per guest CPU, at
    /// most `AUTO_JITV2_THREADS_MAX`, and capped by the host's usable CPUs
    /// with one held back for the device threads
    /// (`MachineConfig::settle_jitv2_threads`).
    /// `n.max(1)`: a misconfigured 0 would
    /// otherwise silently degrade to "no compile threads at all."
    pub fn set_thread_count(&mut self, n: usize) {
        debug_assert!(self.threads.is_empty(), "set_thread_count must be called before the pool ever starts");
        self.thread_count = n.max(1);
    }

    /// Set the pool's fixed worker-thread count and why it was chosen — the
    /// machine builders' way in, so `j2 threads` can say where the number
    /// came from (auto-sized for the guest and the host, or set explicitly).
    pub fn set_thread_count_with_reason(&mut self, n: usize, reason: &str) {
        self.set_thread_count(n);
        self.thread_reason = Some(reason.to_string());
    }

    /// Configured pool size — `j2 threads`'s read-only report.
    #[inline]
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// Why the pool is sized as it is — `j2 threads`'s read-only report.
    /// `None` for a queue built without a machine.
    #[inline]
    pub fn thread_count_reason(&self) -> Option<&str> {
        self.thread_reason.as_deref()
    }

    /// Whether the worker pool is currently spawned — the real signal for
    /// "does the async compile thread own compilation right now," since
    /// `Jitv2::codegen` being `Some`/`None` no longer tracks that (the pool
    /// leaves `Jitv2::codegen`'s idle slot alone the whole time it runs,
    /// reserved for `j2 inline on` to take later — see `Machine::new`'s own
    /// comment on `compile_queue.start()`). `j2 status`/`j2 stats` should
    /// check this, not `jit.codegen.lock().is_some()`, to decide whether to
    /// report from the idle slot or from `function_count()`/pool-wide state.
    #[inline]
    pub fn is_running(&self) -> bool {
        !self.threads.is_empty()
    }

    /// Read-only snapshot of the seal queue's front — `j2 seal-queue`'s only
    /// caller. `None` if the pool has never started (no arena has ever been
    /// built yet). Briefly locks the shared `Mutex<SharedArena>` — not on
    /// any hot path, safe for a monitor command to call at any time
    /// (contends real compiles for the duration of one queue-front peek,
    /// same as any other `try_seal_ready` call already does).
    pub fn seal_queue_snapshot(&self) -> Option<crate::cpu::jitv2::paged_memory::SealQueueSnapshot> {
        let (arena, _) = self.active_arena()?;
        let snapshot = arena.lock().seal_queue_snapshot();
        Some(snapshot)
    }

    /// Every seal-queue entry, front to back — `j2 seal-queue list`'s only
    /// caller. `None` if the pool has never started. See
    /// `SharedArena::seal_queue_entries`'s own doc comment; not bounded, the
    /// caller decides how much to print.
    pub fn seal_queue_entries(&self) -> Option<Vec<(usize, usize, bool, std::thread::ThreadId, *mut crate::cpu::jitv2::PhysicalCodePage)>> {
        let (arena, _) = self.active_arena()?;
        let entries = arena.lock().seal_queue_entries();
        Some(entries)
    }


    /// Sum, across every worker's own slot, of `Codegen::function_count()`
    /// — see `function_counts`' field doc comment for why this is per-worker
    /// rather than one shared mirror. Meaningful only while the pool is
    /// actually running (`j2 stats`/`j2 status` callers should prefer
    /// `Jitv2::codegen.lock()` when it's `Some`, and fall back to this only
    /// when it's `None`, i.e. the pool currently owns every `Codegen`).
    #[inline]
    pub fn function_count(&self) -> u32 {
        self.function_counts.iter().map(|c| c.load(Ordering::Relaxed)).sum()
    }

    /// Current occupancy of the compile-request queue — readable regardless
    /// of whether any worker is running (the queue is a plain shared `Arc`,
    /// not handed off to a consumer). `j2 status`'s live "how full is it
    /// right now" reading, distinct from `JitStats::compile_queue_depth_sum`'s
    /// historical average.
    #[inline]
    pub fn queue_occupancy(&self) -> usize {
        self.queue.len()
    }

    /// Clone of the underlying `Arc<CompileLane>` — lets a caller that only
    /// ever needs to `push` (the CPU thread's dispatch-gate `send`) hold a
    /// handle directly, bypassing `Jitv2`'s outer mutex entirely on the hot
    /// path. Safe to hand out freely: the lane is already the shared,
    /// `&self`-only primitive every worker thread pops from (see the `queue`
    /// field's own doc comment), so an extra clone here is exactly as sound
    /// as the ones `start_inner` already makes per worker.
    pub fn queue_handle(&self) -> Arc<CompileLane> {
        self.queue.clone()
    }

    /// The hot lane, as [`Self::queue_handle`] gives the FIFO.
    pub fn hot_handle(&self) -> Arc<CompileLane> {
        self.hot.clone()
    }

    /// Inject the CPU device handle — see the `cpu` field's doc comment for
    /// why this is a separate setter rather than a constructor parameter.
    /// Safe to call at any time (even while the worker is running); takes
    /// effect on that worker's next flush-threshold check, or immediately
    /// for a not-yet-started queue.
    pub fn set_cpu(&self, cpu: Weak<dyn Device>) {
        *self.cpu.lock() = Some(cpu);
    }

    /// How a flush parks the CPUs on this pool: see the `world` field.
    pub fn set_world(&self, world: Arc<crate::cpu::smp::World>) {
        *self.world.lock() = Some(world);
    }

    /// Inject the weak handle back to the owning `Jitv2` — see the `jitv2`
    /// field's doc comment. Same call-after-construction reasoning as
    /// `set_cpu` (the `Arc<Mutex<Jitv2>>` doesn't exist until after this
    /// `CompileQueue`, which lives inside it, has already been constructed).
    pub fn set_owner(&self, jitv2: Weak<Mutex<Jitv2>>) {
        *self.jitv2.lock() = Some(jitv2);
    }

    /// Push a compile request. Non-blocking: per §6.4, a full queue drops the
    /// request rather than backing up the exec thread — the page that wanted
    /// it stays hot and will re-trigger the request on a later arrival.
    /// Returns `false` if the request was dropped. `stats` is `Jitv2::stats`
    /// — under `developer`, records this dispatch, whether it was dropped
    /// for a full queue, and the queue's occupancy at this exact moment
    /// (`JitStats`'s own doc comments on the three `compile_queue_*` fields)
    /// for `j2 status`'s FIFO-fullness section. Accepted unconditionally
    /// (not `#[cfg]`-gated itself) so this signature doesn't change across
    /// feature combinations; the instrumentation work inside is what's
    /// gated, to keep the extra `slots()`/atomic touches off this
    /// per-dispatch-gate-miss hot path outside a diagnostics build.
    ///
    /// Takes `&self`, not `&mut self`: `self.queue` is an `Arc<ArrayQueue>`
    /// whose own `push`/`len` only need `&self`, and every counter this
    /// touches is an atomic — nothing here actually needs exclusive access.
    /// This lets a caller send through a bare `&CompileQueue`/`queue_handle`
    /// obtained once at construction instead of relocking the whole `Jitv2`
    /// mutex on every dispatch just to reach this one `Arc`-backed field
    /// (see `MipsExecutor::jitv2_compile_queue`/`jitv2_stats`).
    pub fn send(&self, req: CompileRequest, stats: &JitStats) -> bool {
        push_compile_request(&self.queue, req, stats)
    }

    /// Discard every `CompileRequest` currently sitting in the shared queue
    /// without processing it. Every `CompileRequest::page` is a raw pointer
    /// into `Jitv2::pages` (`Vec<PhysicalCodePage>`) — a request enqueued
    /// before a `mega_flush` still holds a pointer into the just-cleared
    /// `Vec` after it, and `comp::handle_request` dereferencing it is a real
    /// use-after-free (confirmed live: `PhysicalCodePage::publish` segfault,
    /// `jitv2-compile` thread, immediately after a `flush_from_jit_thread`
    /// that never drained the queue first). Must run with no worker thread
    /// concurrently popping — true both when called from the flush leader
    /// itself right after `cpu.stop()` has fully joined the CPU and every
    /// other worker has parked (nothing pops anymore), and when called by
    /// `flush_from_cpu_thread`'s caller right after `stop()` has joined
    /// every worker thread (provably not popping).
    ///
    /// Clears each dropped request's page-level `page_scheduled` flag before
    /// discarding it — this specific request is never reaching
    /// `handle_request_deferred`, which is the only place that would
    /// otherwise clear it (its own `ClearScheduledOnDrop` guard). Today
    /// `mega_flush`, called by both callers right after this, independently
    /// resets `page_scheduled` for every page still in the pool anyway (both
    /// its preserve and evict branches), so this is currently redundant in
    /// practice — but making it explicit here, at the actual drop site,
    /// means the invariant ("every request this function discards leaves its
    /// page immediately re-schedulable") holds on its own rather than by
    /// coincidence of what a *different* function happens to do right
    /// after. `req.page` is still safely dereferenceable here — the pool
    /// hasn't been cleared yet at either call site (see this function's own
    /// doc comment above for exactly why that ordering is guaranteed).
    fn drain_pending(queue: &CompileLane, hot: &CompileLane) {
        while let Some(req) = queue.pop() {
            unsafe { (*req.page).clear_scheduled(); }
        }
        while let Some(req) = hot.pop() {
            unsafe { (*req.page).clear_scheduled(); }
        }
    }

    /// Public entry point for `Self::drain_pending` when the caller only
    /// has `&CompileQueue`, not direct access to the shared queue — see
    /// `drain_pending`'s own doc comment for why this must run and when
    /// it's safe to call (no worker concurrently popping).
    pub fn drain_pending_queue(&mut self) {
        Self::drain_pending(&self.queue, &self.hot);
    }

    /// Spawn the compile-thread pool with one freshly-reserved shared arena
    /// — today's normal entry point (`Machine::new`'s startup path). See
    /// `start_inner`'s own doc comment for the full contract.
    /// Publish the CPU model's ISA level to the pool. Call before `start`.
    ///
    /// This replaces a process-global that jitv2 used to read from deep
    /// inside `has_emitter`: the level now travels as data, from the model,
    /// through the worker's `Analyzer`, into `classify`.
    pub fn set_isa(&mut self, mips4: bool) {
        self.mips4 = mips4;
    }

    pub fn start(&mut self, bus: Arc<dyn BusDevice>, stats: Arc<JitStats>) {
        self.start_inner(bus, stats);
    }

    /// No-op if already running. `bus` is the executor's `sysad` — every
    /// worker reads the page snapshot off it at compile time (§6.5 step 2).
    /// `stats` is `Jitv2::stats`, cloned in by the caller (mirrors
    /// `cpu`/`jitv2` below — see `JitStats`'s own doc comment for why this
    /// is an `Arc` handed in rather than reached via a lock on every
    /// `handle_request` call). Threaded through unconditionally (cheap: one
    /// more `Arc` clone per worker) rather than duplicating this whole
    /// function behind `#[cfg(feature = "developer")]` — the actual counter
    /// increments are what's gated, in `comp::handle_request`.
    ///
    /// Spawns exactly `self.thread_count` `jitv2-compile-N` OS threads, each
    /// with its own `Codegen` built via `Codegen::new_with_shared_arena` on
    /// top of the ONE `shared`/`state` pair passed in — one arena
    /// reservation for the whole pool, N independent `JITModule`s sharing
    /// it (see `paged_memory`'s module doc comment for why that's safe:
    /// `allocate()` is mutex-serialized inside `SharedArena`, and sealing is
    /// worker-agnostic by construction). Every worker pops from the same
    /// `self.queue` (`crossbeam_queue::ArrayQueue`) — no per-worker ring, no
    /// dispatch routing needed on the send side.
    fn start_inner(
        &mut self,
        bus: Arc<dyn BusDevice>,
        stats: Arc<JitStats>,
    ) {
        if !self.threads.is_empty() {
            return;
        }
        // The arenas are reserved once, here, and outlive every stop: code
        // published from them keeps running while the pool is stopped.
        let thread_count = self.thread_count;
        let ring = self.ring.get_or_insert_with(|| {
            let arenas = (0..ARENA_COUNT).map(|_| {
                let state = Arc::new(crate::cpu::jitv2::paged_memory::PagedArenaState::default());
                let shared = crate::cpu::jitv2::paged_memory::PagedArenaMemoryProvider::new_shared(
                    ARENA_RESERVE_SIZE, state.clone(),
                ).expect("CompileQueue::start: failed to reserve a jitv2 arena");
                shared.lock().set_ring_owned();
                (shared, state)
            }).collect();
            Arc::new(ArenaRing::new(arenas, thread_count))
        }).clone();
        // Helpers go into an arena while it is empty and no worker is on it.
        // An arena that already holds code keeps what it has.
        let (geometry, consts) = (*self.dc_geometry.lock(), *self.jit_consts.lock());
        for arena in ring.arenas() {
            if arena.1.packing_stats().0 == 0 && arena.1.helpers().iter().all(|&h| h == 0) {
                build_arena_helpers(arena, geometry, consts);
            }
        }
        ring.workers_start();
        let ((shared, state), at) = ring.active();
        self.buses.set_default(bus);
        // Captured before the spawn loop: each worker builds its own
        // `Analyzer` and needs the level by value, not through `&self`.
        let worker_mips4 = self.mips4;
        self.barrier.start(());
        self.function_counts = (0..self.thread_count).map(|_| Arc::new(AtomicU32::new(0))).collect();
        for i in 0..self.thread_count {
            let mut codegen = crate::cpu::jitv2::codegen::Codegen::new_with_shared_arena(shared.clone(), state.clone());
            codegen.dc_geometry = *self.dc_geometry.lock();
            codegen.jit_consts = *self.jit_consts.lock();
            let queue = self.queue.clone();
            let hot = self.hot.clone();
            let bus = self.buses.clone();
            let world = self.world.lock().clone();
            let ring = ring.clone();
            let pages = self.pages;
            let function_count = self.function_counts[i].clone();
            let stats = stats.clone();
            let barrier = self.barrier.clone();
            let thread_count = self.thread_count;
            let wake = self.wake.clone();
            self.threads.push(
                std::thread::Builder::new()
                    .name(format!("jitv2-compile-{i}"))
                    .spawn(move || Self::worker_loop(queue, hot, bus, codegen, world, ring, pages, i, at, function_count, stats, barrier, thread_count, worker_mips4, wake))
                    .expect("jitv2-compile spawn"),
            );
        }
    }

    /// Stop the worker thread(s) and join them, reclaiming the consumer(s)
    /// and every worker's own `Codegen` so a later `start()` can resume
    /// (with fresh `Codegen`s of its own — see `start()`'s own doc comment
    /// on why these returned ones are no longer handed back in). Empty
    /// `Vec` if not running; otherwise one `Codegen` per worker thread, in
    /// no particular order — every caller (`flush_from_cpu_thread`, the `j2
    /// inline` handler) already treats this as "however many were running,"
    /// not "exactly one."
    ///
    /// Drains every request still sitting in the shared queue (the CPU
    /// thread can keep enqueueing right up until the moment `running` flips
    /// false, so this is never guaranteed empty just because every worker
    /// joined) and zeroes the status-bar queue-fill gauge — without this, a
    /// request dropped here silently vanishes (harmless on its own, same as
    /// any other §6.4 drop-on-full/drop-on-flush case; the offset just
    /// re-triggers on its next arrival) but the status bar's occupancy
    /// reading was confirmed live to otherwise stay stuck at whatever it
    /// last was before the stop, forever, since nothing calls
    /// `set_queue_fill` again once no worker is running.
    pub fn stop(&mut self) -> Vec<crate::cpu::jitv2::codegen::Codegen> {
        self.barrier.stop();
        // Wake any worker parked on its idle backstop so it sees the stop
        // now, instead of sleeping out the backstop first (and making this
        // join wait for it).
        self.wake.notify_all();
        // A worker can hold the last reference for a moment (a flush upgrades
        // its weak handles) and so run this from `Drop`; it is already on
        // its way out and cannot join itself.
        let me = std::thread::current().id();
        let result: Vec<_> = self.threads.drain(..).filter(|h| h.thread().id() != me).filter_map(|h| h.join().ok()).collect();
        self.drain_pending_queue();
        // The workers are gone: drop their thread handles, so a later
        // `start` does not accumulate dead ones.
        self.wake.clear();
        crate::cpu::jit_feedback::JIT_FEEDBACK.set_queue_fill(0, COMPILE_QUEUE_CAPACITY);
        result
    }

    /// Worker body: pop requests until stopped and hand each to
    /// `comp::handle_request` (reachability walk + codegen + publish, §6.5 —
    /// no filesystem access of any kind; corpus capture is now an on-demand
    /// `j2 corpus` walk of this pool, see `jitv2/pcp_dump.rs`). Owns the
    /// `Analyzer` scratch state for the thread's whole lifetime (meant to be
    /// reused across jobs, not rebuilt per request); `codegen` is the shared
    /// one moved in by `start`. Parks when the queue is empty rather than
    /// busy-spinning or polling on a timer: the producer wakes the worker
    /// directly (`CompileWake`), and the park's timeout is only a backstop
    /// for the periodic pending-publish sweep — compile requests are bursty
    /// (arrival threshold crossings), not latency-critical, but they do not
    /// need a wakeup either.
    ///
    /// Every compile routes through `comp::handle_request_deferred`, never
    /// `comp::handle_request` — `handle_request`'s forced-seal contract
    /// (mprotect a whole host page to RX immediately, unconditionally) is
    /// only safe with a single caller, since it assumes nothing else could
    /// possibly still be bump-allocating or `copy_nonoverlapping`-ing into
    /// that same page; under a real N>1 pool, another worker routinely can
    /// be — confirmed live as a SIGSEGV inside `CompiledBlob::new`'s own
    /// write the first time this pool ran with `thread_count > 1` outside
    /// tests. `handle_request` stays in use elsewhere (inline compile,
    /// `mips_exec.rs`'s `jitv2_inline_compile` — single-threaded on the CPU
    /// thread, where the forced-seal assumption still holds), just never
    /// here. `handle_request_deferred` still finalizes every successful
    /// compile immediately (`Codegen::finalize_batch_nonforced`,
    /// called with just that one `FuncId` — cheap, and this is what patches
    /// relocations; see that function's own doc comment for why deferring
    /// the finalize call itself, not just the seal, used to be a real bug
    /// under multi-worker concurrency). What's deferred is only the *seal*
    /// (mprotect) step: `finalize_batch_nonforced` pushes the just-finalized
    /// range onto the shared arena's seal queue and opportunistically seals
    /// whatever's now a contiguous prefix from the watermark — the common
    /// case publishes right away, same call. If this range is gap-blocked
    /// behind an earlier, not-yet-finalized range from another worker, it
    /// comes back unsealed and `handle_request_deferred` pushes exactly one
    /// `PendingPublish` onto `pending` for a later retry — see
    /// `paged_memory`'s module doc comment for why sealing (not finalizing)
    /// is what actually lets functions pack tightly instead of each getting
    /// its own page. `pending` is drained two ways: (1) on every queue-empty
    /// poll (the `None` arm), a *non-forced* `comp::publish_ready_nonforced`
    /// attempt runs first (safe/cheap — publishes anything already sealable
    /// without forcing a page closed early), and only once the queue has
    /// been continuously empty for a real ~100ms idle threshold does
    /// `comp::force_publish_pending` force-seal whatever's still stuck — so
    /// a lone straggler that never grows into a full page still gets
    /// published eventually, at the cost of at most one prematurely-closed
    /// partial page per idle burst (see `Codegen::force_seal_pending`'s own
    /// doc comment). `pending` is normally at most one or two entries per
    /// worker (genuinely gap-blocked ranges), not an accumulated batch.
    /// `pending` is discarded (never flushed) immediately before `do_flush` runs:
    /// `do_flush` resets `codegen` (freeing the whole arena — every
    /// not-yet-finalized `FuncId` in `pending` would dangle) and clears the
    /// page pool (every `PendingPublish::page` would dangle too), so there's
    /// nothing safe left to publish by that point — same reasoning as
    /// `drain_pending` already applies to in-flight `CompileRequest`s for
    /// the identical reset.
    ///
    /// After every successful compile, checks `codegen.function_count()`
    /// against the flush threshold (`Codegen::function_count`'s doc comment)
    /// — if crossed, the first worker to detect it becomes leader
    /// (`PoolBarrier::try_lead`) and runs `run_leader_flush`:
    /// stops the CPU (via `cpu`, upgraded from `Weak`; skipped entirely if
    /// unset or already gone — nothing to flush for if the machine has no
    /// CPU to pause), waits for every other worker to park
    /// (`PoolBarrier::park`, called by any worker whose own loop-top check —
    /// or its own trigger detection — sees a cycle already in progress
    /// set), flushes the page pool (`Jitv2::flush_from_jit_thread` — NOT
    /// `Jitv2::flush`, which would try to stop this same compile queue from
    /// within itself), builds one fresh shared arena, rebuilds every
    /// worker's own `Codegen` on top of it
    /// (`Codegen::new_with_shared_arena`/`reset_with_shared_arena`),
    /// publishes it into the barrier, and restarts the CPU. With one worker
    /// (`thread_count == 1`) there are no followers to wait for — the
    /// leader's own barrier wait is immediately satisfied — so this is
    /// observably identical to the pre-pool `do_flush` sequence; the
    /// election/park machinery itself is tested with real concurrent threads,
    /// and modelled under loom, in `pool_barrier`'s tests.
    /// Returns `codegen` on exit so `stop()` can collect it — the shared
    /// `queue` itself is never handed back (it's an `Arc`, cloned into every
    /// worker up front and simply left running/idle after they all exit).
    fn worker_loop(
        queue: Arc<CompileLane>,
        hot: Arc<CompileLane>,
        bus: Arc<CodeBuses>,
        mut codegen: crate::cpu::jitv2::codegen::Codegen,
        world: Option<Arc<crate::cpu::smp::World>>,
        ring: Arc<ArenaRing<PoolArena>>,
        pages: PageSlots,
        // This worker's number in the pool, and the ring switch its
        // `codegen` was built for.
        worker: usize,
        mut my_switch: u64,
        function_count: Arc<AtomicU32>,
        stats: Arc<JitStats>,
        barrier: Arc<PoolBarrier<()>>,
        thread_count: usize,
        // The CPU model’s ISA level, for this worker’s `Analyzer`.
        mips4: bool,
        // The pool's wake-up channel: this worker registers its thread
        // handle here, so a producer can unpark it instead of the worker
        // polling the queue.
        wake: Arc<CompileWake>,
    ) -> crate::cpu::jitv2::codegen::Codegen {
        // Publish this worker's thread handle so the producer can wake it
        // out of the idle park below.
        wake.register(std::thread::current());
        // The memory helpers are already in the arena: `start` and a reclaim
        // build them while the arena is empty and no worker is on it.

        let mut analyzer = crate::cpu::jitv2::analyzer::Analyzer::with_isa(mips4);
        let mut pending: crate::cpu::jitv2::comp::PendingCount = 0;
        // Wall-clock start of the current unbroken stretch of "queue empty
        // AND pending still non-empty after a non-forced publish attempt" —
        // see the `Err(_)` arm below (queue-drain fallback) for the
        // idle-timeout scheme this drives. `None` whenever there's nothing
        // idle to time (queue non-empty, or pending fully drained).
        let mut idle_since: Option<std::time::Instant> = None;
        // Wait for every other worker to park. `false` if the pool stops
        // first: a follower's own top-of-loop check can see the stop and
        // leave `worker_loop` instead of parking, and the leader would wait
        // for it forever (seen live: `stop()`'s join on the leader never
        // returned). The caller must then abandon the cycle and touch
        // nothing shared.
        let followers = thread_count.saturating_sub(1);
        // Release the barrier without publishing anything (no arena, no
        // seal result) — the shared cleanup for every way a leader sequence
        // can end without actually doing the flush/seal work: `cpu`/`jitv2`
        // failed to upgrade, or the pool stopped while waiting for
        // followers (`wait_for_followers` returned `false`).
        // Bumping `generation` here still matters even with nothing to
        // publish — any follower already parked (or about to park) must
        // still wake up and see `was_flush: false`, not hang forever behind
        // a leader that's giving up.
        let abandon_quiesce = |pending: &mut crate::cpu::jitv2::comp::PendingCount| {
            *pending = 0;
            barrier.release(None);
        };
        // Take an arena away: refuse its publishes, sweep its code off every
        // page, then ask every CPU to pass a batch boundary. In that order: a
        // CPU that answers before the sweep is done could still load a
        // function from the arena afterwards. The arena is emptied later, by
        // whichever worker finds the CPUs have all answered (`try_reclaim`
        // at the top of the loop).
        let retire = |arena: &PoolArena| {
            crate::pf_scope!(crate::perfetto::events::JIT_CPU_FLUSH);
            arena.1.retire();
            let pages_swept = pages.retire_code(arena.1.epoch());
            if let Some(w) = &world {
                w.retire_code();
            }
            ring.kicked();
            crate::cpu::jit_feedback::JIT_FEEDBACK.record_flush();
            stats.arena_retires.fetch_add(1, Ordering::Relaxed);
            stats.arena_retired_pages.fetch_add(pages_swept as u64, Ordering::Relaxed);
        };
        // The active arena is full: move the pool to a spare one. `true` if
        // there is none yet, and this worker must wait for a reclaim.
        let rotate = |my_switch: u64| -> bool {
            match ring.switch(my_switch) {
                Switch::Switched { retire: old } => {
                    // The others may be parked on their idle backstop, and a
                    // reclaim waits for each of them to follow.
                    wake.notify_all();
                    if let Some(arena) = old {
                        retire(&arena);
                    }
                    false
                }
                Switch::Behind => false,
                Switch::NoSpare => true,
            }
        };
        // The active arena is full and no arena is empty yet.
        let mut stalled = false;
        // Runs the full leader sequence for a forced-seal-only quiesce — no
        // arena replacement, no CPU stop, just "wait for everyone else to
        // park (guaranteeing none of them is mid-compile — no lower-`start`
        // range can still arrive), force-seal whatever's queued under that
        // guarantee, then resume everyone in place." Called only after this
        // thread has already won leadership via the SAME
        // `PoolBarrier::try_lead` a flush leader would use — see
        // `pool_barrier` for why this and `run_leader_flush` share one
        // gate/barrier instead of two.
        let run_leader_seal = |codegen: &mut crate::cpu::jitv2::codegen::Codegen,
                                pending: &mut crate::cpu::jitv2::comp::PendingCount| {
            // Bails out cleanly (no seal performed) if the pool stops
            // first — see `followers` above for the `CompileQueue::stop()`
            // race this guards.
            if !barrier.wait_for_followers(followers) {
                abandon_quiesce(pending);
                return;
            }
            crate::cpu::jitv2::comp::force_publish_pending(codegen, pending);
            barrier.release(None);
        };
        // Leader election for a forced seal: the first worker whose compile
        // needs one wins `try_lead` and runs it; every other worker that
        // sees the cycle parks at the barrier until it is over.
        let try_force_seal = |codegen: &mut crate::cpu::jitv2::codegen::Codegen,
                               pending: &mut crate::cpu::jitv2::comp::PendingCount| {
            if barrier.try_lead() {
                // Same as `try_flush`: wake the followers off their idle
                // backstop so they reach the barrier now.
                wake.notify_all();
                run_leader_seal(codegen, pending);
            } else {
                *pending = 0;
                barrier.park();
            }
        };
        while barrier.running() {
            // A worker idling between compiles (about to pop its next
            // request) that sees another worker already leading a quiesce
            // cycle (flush OR seal-only — one shared gate, see
            // `pool_barrier`) must park immediately
            // rather than pop/compile anything new — its own next compile
            // would target a page pool/arena that's about to be cleared out
            // from under it, or would race the leader's forced seal. With
            // one worker the leader can only be this same thread, which
            // never reaches back here until it's done (the flag is cleared
            // before the leader function returns), so it is a no-op there.
            if barrier.in_progress() {
                pending = 0;
                barrier.park();
                continue;
            }
            // An arena is on its way out: empty it once every CPU has left
            // its code and every worker has left it.
            if ring.retire_pending() {
                let (geometry, consts) = (codegen.dc_geometry, codegen.jit_consts);
                ring.try_reclaim(
                    || world.as_ref().is_none_or(|w| w.code_retired()),
                    |arena| {
                        // SAFETY: the sweep took this arena's code off every
                        // page before the CPUs were asked; each has since
                        // passed a batch boundary, and no worker is on it.
                        unsafe { arena.0.lock().recycle(); }
                        build_arena_helpers(arena, geometry, consts);
                    },
                );
            }
            // The pool switched arenas: follow. What this worker still has
            // unsealed in the arena it leaves is dropped, and those pages ask
            // again; what is ready publishes, since that arena's code still
            // runs.
            if ring.switches() != my_switch {
                crate::cpu::jitv2::comp::publish_ready_nonforced(&mut codegen, &mut pending);
                pending = 0;
                let ((arena, state), at) = ring.active();
                // SAFETY: the arena this `Codegen` leaves belongs to the
                // ring, which keeps it mapped until its code is retired.
                unsafe { codegen.reset_with_shared_arena(arena, state); }
                my_switch = at;
                ring.reached(worker, at);
                function_count.store(codegen.function_count(), Ordering::Relaxed);
                stalled = false;
            }
            #[cfg(all(test, not(loom)))]
            if barrier.take_forced_flush() {
                stalled = rotate(my_switch);
                continue;
            }
            if stalled {
                // Nothing compiles until an arena is empty. Requests stay
                // queued or are dropped as on a full queue; the CPUs run what
                // is published and interpret the rest.
                stalled = rotate(my_switch);
                if stalled {
                    std::thread::park_timeout(ARENA_STALL_POLL);
                }
                continue;
            }
            // The hot lane first: pages being executed right now that are
            // still waiting (see `CompileQueue::hot`).
            match hot.pop().or_else(|| queue.pop()) {
                Some(req) => {
                    // Always the deferred/non-forced path — never
                    // handle_request's forced-seal one. Forced sealing
                    // mprotects a WHOLE host page to RX immediately,
                    // unconditionally safe only when nothing else can be
                    // mid-allocate/mid-copy on that same page — true for
                    // inline compile (single-threaded, its only other
                    // caller) but not here: a second worker can easily
                    // still be bump-allocating into (or, worse,
                    // mid-`copy_nonoverlapping` writing compiled bytes
                    // into) the very page this worker's own forced seal
                    // would just mprotect out from under it. Confirmed
                    // live as a real SIGSEGV (copy_nonoverlapping crashing
                    // inside CompiledBlob::new) the first time this pool
                    // actually ran with thread_count > 1 outside tests —
                    // `j2 batch off` (the `developer`-build default) used
                    // to route here into handle_request instead, which is
                    // what triggered it.
                    let ran_out_of_memory = {
                        #[cfg(feature = "developer")]
                        { crate::cpu::jitv2::comp::handle_request_deferred(&req, &bus, &mut analyzer, &mut codegen, &mut pending, &stats) }
                        #[cfg(not(feature = "developer"))]
                        { crate::cpu::jitv2::comp::handle_request_deferred(&req, &bus, &mut analyzer, &mut codegen, &mut pending) }
                    };
                    // Keep CompileQueue::function_count's mirror in sync —
                    // see that field's doc comment for why it exists
                    // (`j2 stats` can't read the real codegen.function_count()
                    // directly while this thread owns `codegen` by value).
                    function_count.store(codegen.function_count(), Ordering::Relaxed);
                    // Status-bar feedback (disp.rs's StatusBar), updated
                    // from this side (the compile thread, once per actual
                    // compile) rather than from CompileQueue::send on the
                    // CPU/exec thread — send() runs on the hot dispatch-gate
                    // path on every compile trigger, and an extra atomic
                    // store there would tax exactly the thread this whole
                    // async-queue design exists to keep off the compile
                    // work. Reusing packing_stats() here means this doesn't
                    // even add a second call — just reads the tuple this
                    // iteration was already computing for the threshold
                    // check below.
                    let reserved_bytes = codegen.packing_stats().1;
                    crate::cpu::jit_feedback::JIT_FEEDBACK.set_arena_fill(reserved_bytes, CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES);
                    crate::cpu::jit_feedback::JIT_FEEDBACK.set_queue_fill(queue.len(), COMPILE_QUEUE_CAPACITY);
                    if ran_out_of_memory {
                        // The compile that just ran couldn't get memory —
                        // flush immediately, regardless of the byte
                        // threshold below (the arena is provably already
                        // full, so there's nothing to gain from checking).
                        // The request that just failed is gone
                        // (handle_request already returned), so it'll only
                        // be retried on this offset's next real arrival —
                        // same as any other "retry later" outcome.
                        stalled = rotate(my_switch);
                    } else if reserved_bytes > arena_flush_threshold() {
                        // Real bytes reserved in the arena (not a
                        // function-count proxy — see
                        // CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES's own doc
                        // comment) crossing the threshold means this
                        // thread's own Cranelift memory arena has grown
                        // unboundedly (nothing else ever frees it — see
                        // Codegen::reset's doc comment) and needs flushing
                        // pre-emptively, before it actually runs out.
                        stalled = rotate(my_switch);
                    } else if pending >= PENDING_FORCE_SEAL_THRESHOLD {
                        // A continuously busy worker (queue never goes
                        // empty for it) never reaches the `None` arm's
                        // idle-timeout sweep below — every compile finalizes
                        // immediately (see handle_request_deferred's own doc
                        // comment) but a non-forced finalize only seals a
                        // page the bump cursor has already moved past, so a
                        // steady stream of small compiles can accumulate in
                        // `pending` indefinitely with nothing to force a
                        // seal. Once `pending` grows past a small bound,
                        // force-seal it now rather than wait for a queue-
                        // empty tick that may never come — bounds `pending`
                        // under sustained load at the cost of at most one
                        // prematurely-closed partial page per threshold
                        // crossing, same trade-off as the idle-timeout sweep.
                        // Goes through try_force_seal (the shared quiesce
                        // barrier), not a direct force_publish_pending call
                        // — see `pool_barrier` for why a
                        // forced seal is only safe once every other worker
                        // is provably parked (not mid-compile).
                        #[cfg(feature = "developer")]
                        stats.record_batch_flush(pending, crate::cpu::jitv2::BatchFlushReason::PendingThreshold);
                        try_force_seal(&mut codegen, &mut pending);
                        function_count.store(codegen.function_count(), Ordering::Relaxed);
                    }
                }
                None => {
                    // Queue-drain fallback: even under batching, a lone
                    // compile followed by a quiet period must not sit
                    // unpublished indefinitely waiting for a page to fill.
                    //
                    // Two-phase, matching the non-forced sealing design
                    // (`paged_memory`'s module doc comment,
                    // `Codegen::finalize_batch_nonforced`/`force_seal_pending`):
                    // on every empty poll, try a non-forced publish attempt
                    // first (`publish_ready_nonforced` — cheap, harmless,
                    // and at today's N=1 worker count it's what actually
                    // does the real work here, converging with the page-cross
                    // trigger above). Only once the queue has been
                    // continuously empty for the real idle threshold
                    // (~100ms, tracked in wall-clock time across repeated
                    // empty polls, not "first empty poll") does this
                    // force-seal whatever's left — so a lone straggler that
                    // never grows into a full page still gets published
                    // eventually, but a genuinely busy queue that's just
                    // between two back-to-back bursts doesn't pay for a
                    // premature forced seal on every backstop tick
                    // (`WORKER_IDLE_PARK_BACKSTOP`).
                    // At N=1 this is behaviorally a no-op relative to
                    // today (nothing here can ever actually be blocked on
                    // another worker), but it's real, reachable machinery —
                    // this is where the same idle-sweep this worker will
                    // need once a real multi-worker pool shares one arena
                    // gets exercised and proven first.
                    if pending > 0 {
                        crate::cpu::jitv2::comp::publish_ready_nonforced(&mut codegen, &mut pending);
                        if pending > 0 {
                            if idle_since.is_none() {
                                idle_since = Some(std::time::Instant::now());
                            }
                            if idle_since.is_some_and(|t| t.elapsed() >= IDLE_FORCE_SEAL_THRESHOLD) {
                                #[cfg(feature = "developer")]
                                stats.record_batch_flush(pending, crate::cpu::jitv2::BatchFlushReason::QueueDrain);
                                // Same shared quiesce barrier as the pending-
                                // threshold trigger above — see
                                // `pool_barrier`.
                                try_force_seal(&mut codegen, &mut pending);
                                idle_since = None;
                            }
                        } else {
                            idle_since = None;
                        }
                        function_count.store(codegen.function_count(), Ordering::Relaxed);
                    } else {
                        idle_since = None;
                    }
                    // Park until a producer has work, or the backstop
                    // expires. `park_idle` counts this worker as asleep
                    // and checks the queues again before parking, so a
                    // request that arrives after the check above is not
                    // lost: either the recheck sees it, or the producer
                    // sees the sleeper and unparks it. Sized for idle cost,
                    // not for latency — the producer wakes this worker
                    // directly (see `CompileWake`).
                    //
                    // Two backstops: with a publish still outstanding, the
                    // worker must come back soon to retry it (the compiled
                    // code reaches the guest on that retry), which is the
                    // old 200µs poll. With nothing pending the worker is
                    // idle, and the long backstop is what keeps it off the
                    // host's cores.
                    let backstop = if pending > 0 {
                        WORKER_IDLE_PENDING_BACKOFF
                    } else {
                        WORKER_IDLE_PARK_BACKSTOP
                    };
                    wake.park_idle(&queue, &hot, backstop);
                }
            }
        }
        codegen
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64};

    /// A claimed, otherwise-untouched page slot for the FR-pin tests below.
    /// `claim` debug-asserts the slot is clean, so it must start unclaimed.
    /// `retire_code` takes exactly the named arena's code, with the entry
    /// requests and the churn snapshot that went with it, and a publish from
    /// a retired arena is refused.
    #[test]
    fn retiring_an_arena_takes_its_code_off_the_page_and_refuses_its_publishes() {
        use crate::cpu::jitv2::paged_memory::{PagedArenaMemoryProvider, PagedArenaState};
        let arena = |size| {
            let state = Arc::new(PagedArenaState::default());
            let _keep = PagedArenaMemoryProvider::new_shared(size, state.clone()).unwrap();
            state
        };
        let (a, b) = (arena(1 << 16), arena(1 << 16));
        assert_ne!(a.epoch(), b.epoch());
        let page = claimed_page(false);
        let gen = page.current_gen();
        let mut bits = [0u64; BITMAP_WORDS];
        bits[0] = 0b101;
        let words = [0u32; ENTRIES_PER_PAGE];
        page.mark_requested(0);
        page.mark_requested(7);
        page.stage_compile_snapshot(&words, &bits, &bits, gen, false);
        assert!(page.publish_from(&a, page.tenancy(), &bits, 0x1000 as *const (), gen, 3, 64, false));
        page.commit_compile_snapshot(gen, &bits, false);
        assert!(page.has_compile_snapshot() && page.is_runnable(2));
        assert_eq!(page.code_epoch(), a.epoch());

        assert!(!page.retire_code(b.epoch()), "another arena's retirement leaves the page alone");
        assert!(page.is_runnable(2));

        assert!(page.retire_code(a.epoch()));
        assert!(!page.is_runnable(0) && !page.is_runnable(2) && !page.has_code());
        assert_eq!(page.func(), NO_CODE as *const (), "never null: a CPU may be about to call it");
        assert!(!page.is_requested(7), "entry requests go with the code");
        assert!(!page.has_compile_snapshot(), "nothing may re-validate the function that went");
        assert!(!page.retire_code(a.epoch()), "nothing left to take");

        // A compile that was in flight comes back to publish.
        a.retire();
        assert!(!page.publish_from(&a, page.tenancy(), &bits, 0x1000 as *const (), gen, 3, 64, false));
        assert!(!page.has_code());
        // The same code from an arena that is still in use lands.
        assert!(page.publish_from(&b, page.tenancy(), &bits, 0x2000 as *const (), gen, 3, 64, false));
        assert_eq!(page.code_epoch(), b.epoch());
    }

    fn claimed_page(fr1: bool) -> Box<PhysicalCodePage> {
        let mut page = Box::new(PhysicalCodePage::new(UNCLAIMED_PFN, std::ptr::null()));
        page.claim(CodeKey::flat(0x1234 << 12), std::ptr::null(), fr1);
        page
    }

    /// A page recycled between processes of different ABIs (o32 FR=0 ->
    /// n32 FR=1) must not keep the previous tenant's FR pin. `publish` already
    /// replaces `compiled` and resets `denied` on a real invalidation,
    /// reasoning that decisions made against the OLD bytes are worthless once
    /// the bytes change; `fr1` has identical justification and used to be the
    /// one field left behind.
    ///
    /// This is the 4Dwm/fm session-start core-dump shape: a burst of process
    /// creation recycles pages, the stale pin makes the next compile
    /// specialize for the wrong FR mode, and because `emit_fr_mode_guard`
    /// suppresses itself when CU1 is clear that is not reliably caught —
    /// wrong FPR packing on `ldc1`/`lwc1` silently corrupts FP data.
    /// A page compiled for FR1 and then re-pinned to FR0 at the same
    /// generation (kernel code runs in whichever mode the interrupted process
    /// uses) must accept the FR0 compile. Refusing it as "already covered"
    /// left FR1 code, an FR1 churn snapshot and an FR0 pin, and every request
    /// after that recompiled and was refused again: a loop seen running
    /// thousands of rounds on one kernel page.
    #[test]
    fn same_gen_compile_for_the_other_fr_mode_publishes() {
        let page = claimed_page(true);
        let words = [0u32; ENTRIES_PER_PAGE];
        let mut used = [0u64; BITMAP_WORDS];
        used[0] = 0b111;
        let mut e = [0u64; BITMAP_WORDS];
        e[0] = 0b101;
        // An FR1 compile at a new generation publishes (which unpins FR)
        // and commits its snapshot.
        page.stage_compile_snapshot(&words, &used, &e, 1, true);
        assert!(page.publish(&e, 0x1000 as *const (), 1, 3, 0, true));
        page.commit_compile_snapshot(1, &e, true);
        // The next request comes from an FR0 context and takes the live mode;
        // the churn skip rightly refuses FR1 code for it.
        assert!(!page.take_fr_repin(false));
        assert!(!page.try_skip_redundant_compile(&words, &e, 1, false));
        // The FR0 compile of the same bytes and entries must install.
        page.stage_compile_snapshot(&words, &used, &e, 1, false);
        assert!(page.publish(&e, 0x2000 as *const (), 1, 3, 0, false), "a compile for the other FR mode is not redundant");
        page.commit_compile_snapshot(1, &e, false);
        assert_eq!(page.func(), 0x2000 as *const ());
        // Now consistent: the next FR0 request is answered by the skip, and
        // a duplicate FR0 compile is still refused as redundant.
        assert!(!page.take_fr_repin(false));
        assert!(page.try_skip_redundant_compile(&words, &e, 1, false), "the FR0 snapshot now serves FR0 requests");
        assert!(!page.publish(&e, 0x3000 as *const (), 1, 3, 0, false), "same mode, same entries: still redundant");
    }

    /// Installing code for the other FR mode replaces the advertised entries
    /// rather than adding to them: the old function and its dispatch switch
    /// are gone, so an entry only it had must stop being dispatchable. The
    /// sticky denials stay, since the bytes did not change.
    #[test]
    fn mode_switch_replaces_coverage_and_keeps_denials() {
        let page = claimed_page(true);
        let mut wide = [0u64; BITMAP_WORDS];
        wide[0] = 0b1111;
        let mut narrow = [0u64; BITMAP_WORDS];
        narrow[0] = 0b0011;
        assert!(page.publish(&wide, 0x1000 as *const (), 1, 3, 0, true));
        page.denylist(40);
        assert!(page.publish(&narrow, 0x2000 as *const (), 1, 3, 0, false));
        assert!(page.is_published(0) && page.is_published(1));
        assert!(!page.is_published(2) && !page.is_published(3), "entries only the FR1 function had must not stay advertised");
        assert!(page.is_denylisted(40), "same bytes: denials stand");
    }

    #[test]
    fn real_invalidation_drops_the_stale_fr_pin() {
        let page = claimed_page(false); // o32 tenant: FR0
        assert!(!page.is_fr1());

        // A real invalidation: the guest replaced this page's bytes, so the
        // publish carries a generation strictly newer than `entry_gen`.
        let entries = [0u64; BITMAP_WORDS];
        let published = page.publish(&entries, std::ptr::null(), 1, 1, 0, false);
        assert!(published, "setup: a fresh generation must actually publish");

        // The next compile arrives from an n32 process (live FR=1). Before
        // the fix this returned the stale `false` and compiled FR0 code for
        // FR1 bytes.
        assert!(page.take_fr_repin(true), "a page whose bytes were replaced must re-derive FR from the live mode, not keep the old tenant's pin");
    }

    /// Coverage added to an EXISTING generation is not an invalidation — the
    /// bytes are unchanged, so the pin still describes them and an unrelated
    /// dispatch in another mode must not silently re-specialize the page.
    /// Guards the obvious over-correction of the test above.
    #[test]
    fn coverage_only_publish_keeps_the_fr_pin() {
        let page = claimed_page(false);
        let mut entries = [0u64; BITMAP_WORDS];
        entries[0] = 1;
        assert!(page.publish(&entries, std::ptr::null(), 0, 1, 0, false));

        assert!(!page.take_fr_repin(true), "a coverage-only publish leaves the bytes alone, so the pin still describes them");
    }

    /// The FR guard's bail records the mode the dispatch actually demanded,
    /// and the next compile adopts it. Without this a page whose live mode
    /// flipped without its bytes changing (the shared `checkfp.s` page)
    /// rebuilds the same wrong-mode code and guard-fails again, forever.
    #[test]
    fn fr_guard_bail_repins_to_the_demanded_mode() {
        let page = claimed_page(false);
        page.request_fr_repin(true); // guard observed live FR=1

        assert!(page.take_fr_repin(false), "the compile must adopt the mode the bail demanded, not the stale pin");
        assert!(page.is_fr1(), "the demand must be committed to the pin, not just returned once");
        // Consumed: a second compile with no new bail keeps the new pin.
        assert!(page.take_fr_repin(false), "the request is one-shot; the re-pinned mode persists");
    }

    /// The re-pin request stores the DEMANDED mode, not a "flip me" bit.
    /// Multiple bails before one compile is the NORMAL case, not a race: the
    /// guard kills a page's FPU-touching entries one at a time as each is
    /// dispatched, so a page with two such entries records two demands. A
    /// flip-bit encoding would toggle twice and land back on the wrong mode.
    #[test]
    fn repeated_repin_requests_are_idempotent() {
        let page = claimed_page(false);
        page.request_fr_repin(true);
        page.request_fr_repin(true);
        assert!(page.take_fr_repin(false), "two bails demanding FR1 must still mean FR1, not a double flip back to FR0");
    }

    /// With nothing outstanding, an ordinary dispatch must not re-pin a page
    /// that was never mismatched — a page is normally dispatched in its
    /// pinned mode, and re-deriving unconditionally would let any passing
    /// dispatch re-specialize it.
    #[test]
    fn take_fr_repin_is_inert_without_a_request() {
        let page = claimed_page(true);
        assert!(page.take_fr_repin(false), "no request and no invalidation: keep the existing pin");
        assert!(page.is_fr1());
    }

    /// Pins `comp::MAX_INSTRS_PER_COMPILE` to an explicit value for the
    /// lifetime of the guard, restoring the prior value on drop (including
    /// on panic/unwind) — tests must never rely on whatever the process-wide
    /// default happens to be, since it's a single global shared with every
    /// other test in this binary (potentially running concurrently) and with
    /// production code's own default.
    struct MaxInstrsGuard(usize);
    impl MaxInstrsGuard {
        fn set(n: usize) -> Self {
            let prev = crate::cpu::jitv2::comp::max_instrs_per_compile();
            crate::cpu::jitv2::comp::set_max_instrs_per_compile(n);
            Self(prev)
        }
    }
    impl Drop for MaxInstrsGuard {
        fn drop(&mut self) {
            crate::cpu::jitv2::comp::set_max_instrs_per_compile(self.0);
        }
    }

    #[test]
    fn null_gen_ptr_reads_the_never_compilable_fallback_without_panicking() {
        // A device with no real gen tracking (gen_ptr returns null, e.g.
        // most MMIO) must never leave PhysicalCodePage::gen null itself —
        // current_gen() has to be unconditionally safe to call on any page,
        // claimed or not, real gen or not (see NEVER_COMPILABLE_GEN's doc
        // comment) — so this must read 0, not panic/deref-null.
        let page = PhysicalCodePage::new(0, std::ptr::null());
        assert_eq!(page.current_gen(), 0);
    }

    #[test]
    fn physical_code_page_size_is_within_expected_bounds() {
        // Guardrail, not a strict contract: §13 replaced the old 1024-entry
        // AoS table with a handful of fixed-size bitmaps + one func pointer,
        // so this should be dramatically smaller than before — fails loudly
        // if PhysicalCodePage ever grows past a sanity ceiling rather than
        // silently ballooning Jitv2::new's one-shot allocation.
        //
        // The churn-avoidance `CompileSnapshot` (see its own doc comment)
        // deliberately put ~4.25KB of that back: a 4KB copy of the page
        // bytes plus two bitmaps. That is the dominant term now — roughly
        // 20MB across the default 4096-slot pool, traded knowingly against
        // the analyze+codegen+finalize passes it avoids. Slots that never
        // compile never touch their copy, so it stays untouched zero pages
        // rather than resident memory. If this assert fires, check whether
        // something OTHER than that snapshot grew before raising the
        // ceiling.
        let page_size = std::mem::size_of::<PhysicalCodePage>();
        println!("size_of::<PhysicalCodePage>() = {page_size} bytes");
        assert!(page_size < 128 * 1024, "PhysicalCodePage grew unexpectedly large: {page_size} bytes — check for accidental field bloat before raising this ceiling");
    }

    #[test]
    fn reads_through_gen_pointer() {
        let counter = AtomicU64::new(42);
        let page = PhysicalCodePage::new(7, &counter as *const AtomicU64);
        assert_eq!(page.current_gen(), 42);
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(page.current_gen(), 43);
    }

    #[test]
    fn entry_starts_unpublished_and_undenylisted() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(!page.is_runnable(4));
        assert!(!page.is_denylisted(4));
        assert!(!page.has_code());
    }

    /// `try_schedule` must be a genuine test-and-set: the first caller for a
    /// given offset wins (returns `true`); every subsequent caller for the
    /// same offset, before `clear_scheduled` runs, must lose (returns
    /// `false`) — this is what stops `exec_decoded`'s dispatch gate from
    /// sending a duplicate `CompileRequest` for the same PAGE every time a
    /// hot PC re-satisfies the gate's trigger conditions while the first
    /// request is still in flight (§13.2 — page-scoped now, not per-offset;
    /// see `try_schedule_page`'s own doc comment).
    #[test]
    fn try_schedule_page_is_test_and_set() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule_page(), "first caller for a fresh page must win");
        assert!(!page.try_schedule_page(), "second caller before clear_scheduled must lose");
        assert!(!page.try_schedule_page(), "still losing on a third call");
    }

    #[test]
    fn a_waiting_page_is_promoted_once_at_the_threshold_and_rearms_after_a_compile() {
        let page = PhysicalCodePage::new(1, std::ptr::null());
        assert!(page.try_schedule_page(), "first request queues");
        for _ in 1..HOT_PROMOTE_ARRIVALS {
            assert!(!page.note_waiting_arrival(), "not hot yet");
        }
        assert!(page.note_waiting_arrival(), "promoted at the threshold");
        assert!(!page.note_waiting_arrival(), "and only once per wait");
        page.clear_scheduled(); // a compile for the page completed
        for _ in 1..HOT_PROMOTE_ARRIVALS {
            assert!(!page.note_waiting_arrival());
        }
        assert!(page.note_waiting_arrival(), "re-arms after the compile");
    }

    /// The batched counter must reach the same threshold as the one-at-a-time
    /// one, and promote exactly once per wait: the dispatch gate counts
    /// arrivals per CPU and folds them in batches, so the shared counter
    /// moves in steps and the promotion lands on the first check after the
    /// threshold is reached.
    #[test]
    fn a_batch_of_arrivals_reaches_the_same_promotion_threshold() {
        let page = PhysicalCodePage::new(1, std::ptr::null());
        assert!(page.try_schedule_page(), "first request queues");
        // Every batch but the last stays short of the threshold.
        let mut counted = 0u32;
        while counted + HOT_ARRIVAL_BATCH < HOT_PROMOTE_ARRIVALS {
            page.note_waiting_arrivals(HOT_ARRIVAL_BATCH);
            counted += HOT_ARRIVAL_BATCH;
            assert!(!page.take_hot_promotion(), "not hot yet");
        }
        // The last batch crosses it.
        page.note_waiting_arrivals(HOT_PROMOTE_ARRIVALS - counted);
        assert!(page.take_hot_promotion(), "promoted at the threshold");
        assert!(!page.take_hot_promotion(), "and only once per wait");
        page.clear_scheduled(); // a compile for the page completed
        page.note_waiting_arrivals(HOT_PROMOTE_ARRIVALS);
        assert!(page.take_hot_promotion(), "re-arms after the compile");
    }

    /// A batch that does not cross the threshold must not promote, and the
    /// count it folds in must be the count the one-at-a-time path would
    /// have reached — the threshold is on arrivals, not on calls.
    #[test]
    fn a_batch_below_the_threshold_does_not_promote() {
        let page = PhysicalCodePage::new(1, std::ptr::null());
        assert!(page.try_schedule_page());
        page.note_waiting_arrivals(HOT_PROMOTE_ARRIVALS - 1);
        assert!(!page.take_hot_promotion());
        // The next arrival reaches the threshold exactly.
        page.note_waiting_arrivals(1);
        assert!(page.take_hot_promotion(), "the threshold counts arrivals");
    }

    #[test]
    fn a_promotion_the_hot_lane_refused_can_be_made_again() {
        let page = PhysicalCodePage::new(0, std::ptr::null());
        assert!(page.try_schedule_page());
        for _ in 1..HOT_PROMOTE_ARRIVALS {
            page.note_waiting_arrival();
        }
        assert!(page.note_waiting_arrival());
        page.rearm_promotion();
        for _ in 1..HOT_PROMOTE_ARRIVALS {
            assert!(!page.note_waiting_arrival());
        }
        assert!(page.note_waiting_arrival(), "promoted again");
        assert!(page.is_scheduled(), "still queued in the FIFO");
    }

    #[test]
    fn clear_scheduled_allows_a_fresh_try_schedule_page() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        assert!(page.try_schedule_page());
        assert!(!page.try_schedule_page());

        page.clear_scheduled();
        assert!(page.try_schedule_page(), "after clear_scheduled, a fresh request for the same page must be allowed again");
    }

    #[test]
    fn clear_scheduled_on_an_unset_page_is_a_harmless_no_op() {
        // The jitv2_inline_compile path calls handle_request (and therefore
        // clear_scheduled, via its scope guard) without ever having called
        // try_schedule_page first — must not panic.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.clear_scheduled();
        assert!(page.try_schedule_page(), "page must still be schedulable after a no-op clear");
    }

    #[test]
    fn is_scheduled_reflects_try_schedule_page_and_clear_scheduled_without_mutating() {
        // `is_scheduled` (j2 pcp's "scheduled(in-flight)" diagnostic column)
        // must be a pure read — checking it repeatedly must never itself
        // flip the flag the way try_schedule_page's test-and-set does.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(!page.is_scheduled());
        assert!(!page.is_scheduled(), "a plain read must not consume/flip the flag");

        assert!(page.try_schedule_page());
        assert!(page.is_scheduled());
        assert!(page.is_scheduled(), "still in flight on a repeated read");

        page.clear_scheduled();
        assert!(!page.is_scheduled());
    }

    #[test]
    fn is_requested_reflects_mark_requested_per_offset() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert!(!page.is_requested(4));

        page.mark_requested(4);
        assert!(page.is_requested(4));
        assert!(!page.is_requested(5), "a different offset must not be marked");
    }

    #[test]
    fn mark_requested_is_idempotent_and_per_offset_independent() {
        // §13.2: unlike the old per-offset ENTRY_SCHEDULED test-and-set,
        // mark_requested is unconditional/idempotent — it just accumulates
        // bits for whatever a future compile's walk should cover, with no
        // "first caller wins" contract at the offset level (that dedup now
        // lives at the page level, try_schedule_page).
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        page.mark_requested(4);
        let snap = page.snapshot_requested();
        assert_eq!(snap[0] & (1 << 4), 1 << 4, "offset 4 must be marked requested");
        assert_eq!(snap[0] & (1 << 5), 0, "a different offset must not be marked");

        page.mark_requested(4); // idempotent
        page.mark_requested(5);
        let snap = page.snapshot_requested();
        assert_eq!(snap[0] & ((1 << 4) | (1 << 5)), (1 << 4) | (1 << 5), "both offsets stay marked");
    }

    /// `code_bytes_used` sums raw `code_size`, with no page rounding.
    ///
    /// It used to round every published entry up to a hardcoded 4096, which
    /// described `ArenaMemoryProvider` — one page-rounded segment per
    /// function, a type that no longer exists. The live
    /// `PagedArenaMemoryProvider` packs, so a function costs about its own
    /// `code_size`.
    ///
    /// The constant was also wrong wherever the host page is not 4 KiB, which
    /// includes every Apple Silicon Mac (16 KiB) and aarch64 Linux kernels
    /// built for 16 or 64 KiB. That is why nothing here hardcodes a page size
    /// and the failure message asks `region::page::size()` at runtime: the
    /// page size is not a property of the target triple.
    #[test]
    fn code_bytes_used_is_not_page_rounded() {
        let counter = AtomicU64::new(0);
        let sizes = [215u32, 300, 48];
        // One function per page: `code_bytes_used` sums one `code_size` per
        // *page*, so three entries means three pages. Publishing three times
        // into one page just overwrites its `code_size`.
        let mut jit = Jitv2::new(sizes.len());

        for (i, sz) in sizes.iter().enumerate() {
            jit.pages[i].claim(CodeKey::flat((i as u64 + 1) << 12), &counter as *const AtomicU64, false);
            let mut bits = [0u64; BITMAP_WORDS];
            bits[0] |= 1u64;
            assert!(jit.pages[i].publish(&bits, 0x1000 as *const (), 0, 1, *sz, false));
        }

        let expected: u64 = sizes.iter().map(|s| *s as u64).sum();
        assert_eq!(
            jit.code_bytes_used(), expected,
            "must sum raw code_size; page-rounding would report {} instead",
            sizes.len() as u64 * region::page::size() as u64,
        );
    }

    #[test]
    fn snapshot_compile_candidates_unions_requested_and_compiled_masked_by_denied() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        // offset 4: requested only.
        page.mark_requested(4);
        // offset 8: published (compiled) only, no fresh request.
        let mut bits = [0u64; BITMAP_WORDS];
        bits[0] |= 1u64 << 8;
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0, false));
        // offset 12: requested, but denylisted — must never appear regardless
        // of include_compiled.
        page.mark_requested(12);
        page.denylist(12);

        // include_compiled=false: only offset 4 (the fresh request); offset
        // 8's compiled-only coverage is invisible, offset 12 stays excluded.
        let without_compiled = page.snapshot_compile_candidates(false);
        assert_eq!(without_compiled[0] & (1 << 4), 1 << 4, "requested offset must appear");
        assert_eq!(without_compiled[0] & (1 << 8), 0, "compiled-only offset must not appear when include_compiled is false");
        assert_eq!(without_compiled[0] & (1 << 12), 0, "denylisted offset must never appear");

        // include_compiled=true: offset 8 joins offset 4; offset 12 still excluded.
        let with_compiled = page.snapshot_compile_candidates(true);
        assert_eq!(with_compiled[0] & (1 << 4), 1 << 4, "requested offset must still appear");
        assert_eq!(with_compiled[0] & (1 << 8), 1 << 8, "compiled offset must appear when include_compiled is true");
        assert_eq!(with_compiled[0] & (1 << 12), 0, "denylisted offset must never appear even when include_compiled is true");
    }

    #[test]
    fn entry_valid_only_when_bit_set_and_gen_matches() {
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;

        // Bit not set at all: invalid regardless of gen.
        assert!(!page.is_runnable(offset));

        // Publish: set the bit -> now valid.
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset >> 6] |= 1u64 << (offset & 63);
        assert!(page.publish(&bits, 0x1000 as *const (), 5, 1, 0, false));
        assert!(page.is_runnable(offset));

        // Page mutates (gen bumps past what the entry was compiled against):
        // bit is still set, but the entry must read as stale.
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_runnable(offset), "stale entry (gen mismatch) must not be reported valid");
    }

    #[test]
    fn kill_clears_valid_bit_but_not_denylist() {
        // emit_fr_mode_guard's FR-mismatch arm (jit_kill_entry) uses this
        // to un-publish a compiled-for-the-wrong-FR-mode entry: the JIT gate
        // must stop dispatching it, but a later visit is expected and
        // welcome to recompile the same offset fresh — unlike denylist,
        // which is permanent (§6.4 sticky rejection).
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 4usize;
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset >> 6] |= 1u64 << (offset & 63);
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0, false));
        assert!(page.is_runnable(offset));

        page.kill(offset);

        assert!(!page.is_published(offset), "kill must clear the valid bit");
        assert!(!page.is_runnable(offset));
        assert!(!page.is_denylisted(offset), "kill must not sticky-reject the offset — a fresh compile is expected to follow");

        // A later re-publish (simulating the next visit's fresh compile)
        // must work normally — kill leaves the offset fully recompilable.
        // Same gen (0): a pure entry-coverage republish, not a real
        // invalidation — publish() still succeeds because `kill` cleared the
        // `compiled` bit, so the previous "subsumed" check no longer blocks it.
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0, false));
        assert!(page.is_runnable(offset), "offset must be re-publishable after kill");
    }

    #[test]
    fn publish_recompiling_a_stale_entry_updates_func_and_gen_together() {
        // Regression test for the recompile-ordering race: an entry whose
        // gen has drifted stale (page mutated, ENTRY_VALID still set) gets
        // recompiled in place by handle_request (comp.rs) — the ONE case
        // where publish() is called on an offset whose flag is already set.
        // ENTRY_VALID's Release/Acquire pairing gives no fresh synchronization
        // on that path (the flag's value doesn't change), so `gen` itself
        // must be the ordering point: func must be visible before gen ever
        // reads as matching current_gen(). This test can't force a true
        // concurrent interleaving, but it does verify the sequential
        // contract publish() must uphold for that ordering argument to hold
        // at all: func actually gets updated, and is_runnable only
        // reports true once gen matches (i.e. after publish() completes).
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 100usize;
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset >> 6] |= 1u64 << (offset & 63);

        let old_fn = 0x1000usize as *const ();
        assert!(page.publish(&bits, old_fn, 5, 1, 0, false));
        assert!(page.is_runnable(offset));
        assert_eq!(page.func(), old_fn);

        // Page mutates: entry goes stale (bit stays 1, gen no longer matches).
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_runnable(offset), "must read stale immediately after the page mutates");

        // Recompile in place (comp.rs's handle_request path for a
        // stale-but-still-published entry) — gen_snap=6 was captured before
        // this second compile started, matching the page's now-current gen.
        let new_fn = 0x2000usize as *const ();
        assert!(page.publish(&bits, new_fn, 6, 1, 0, false));
        assert!(page.is_runnable(offset), "recompiled entry must read valid once publish completes");
        assert_eq!(page.func(), new_fn, "func must be the NEW function, not the stale one, once gen reads as current");
    }

    /// Build the page-byte/used/entries triple for the churn-avoidance
    /// tests: `entries` bits set for each offset in `entry_offsets`, `used`
    /// bits set for each word in `used_words`, and `words` filled with a
    /// recognisable pattern.
    fn churn_fixture(
        entry_offsets: &[usize],
        used_words: &[usize],
    ) -> ([u32; ENTRIES_PER_PAGE], [u64; BITMAP_WORDS], [u64; BITMAP_WORDS]) {
        let words: [u32; ENTRIES_PER_PAGE] = std::array::from_fn(|i| 0x1000_0000 + i as u32);
        let mut used = [0u64; BITMAP_WORDS];
        for &w in used_words { used[w >> 6] |= 1u64 << (w & 63); }
        let mut entries = [0u64; BITMAP_WORDS];
        for &o in entry_offsets { entries[o >> 6] |= 1u64 << (o & 63); }
        (words, used, entries)
    }

    /// Stage + commit a compile snapshot against a real publish, the way
    /// `comp.rs` does — the churn tests all start from a page in this state.
    fn publish_with_snapshot(
        page: &PhysicalCodePage,
        words: &[u32; ENTRIES_PER_PAGE],
        used: &[u64; BITMAP_WORDS],
        entries: &[u64; BITMAP_WORDS],
        gen: u64,
        fr1: bool,
        func: *const (),
    ) {
        page.stage_compile_snapshot(words, used, entries, gen, fr1);
        assert!(page.publish(entries, func, gen, 1, 0, false), "fixture publish must succeed");
        page.commit_compile_snapshot(gen, entries, fr1);
        assert!(page.has_compile_snapshot(), "fixture must leave a usable snapshot");
    }

    #[test]
    fn a_bumped_gen_with_unchanged_code_words_skips_the_compile() {
        // The case the whole mechanism exists for: something wrote to the
        // page (gen bumped) but not to any word the compile actually
        // decoded, so recompiling would emit byte-identical code. The skip
        // must re-validate the installed function against the new gen —
        // leaving it dispatchable — without touching func or compiled.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11, 12]);
        let func = 0x1000usize as *const ();
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, func);

        // A write lands somewhere on the page that is NOT one of the decoded
        // words — data sharing the page, DMA, a neighbouring structure.
        let mut changed = words;
        changed[900] = 0xdead_beef;
        counter.store(6, Ordering::Relaxed);
        assert!(!page.is_runnable(10), "gen bump must make the entry stale first");

        assert!(page.try_skip_redundant_compile(&changed, &entries, 6, false),
            "unchanged decoded words must skip the compile");
        assert!(page.is_runnable(10), "the skip must re-validate the installed code against the new gen");
        assert_eq!(page.func(), func, "a skip must leave the installed function alone");
    }

    #[test]
    fn a_changed_decoded_word_forces_a_real_compile() {
        // The complement: the write DID land on a word the compile decoded,
        // so the installed code no longer describes the page and a real
        // recompile is mandatory.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11, 12]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        let mut changed = words;
        changed[11] = 0xdead_beef; // a decoded word
        counter.store(6, Ordering::Relaxed);

        assert!(!page.try_skip_redundant_compile(&changed, &entries, 6, false),
            "a changed decoded word must NOT be skippable");
        assert!(!page.is_runnable(10), "a rejected skip must leave the entry stale, not re-validate it");
    }

    #[test]
    fn an_fr_mode_flip_forces_a_real_compile_despite_identical_bytes() {
        // FR mode is baked into every FPR-access emitter and is invisible in
        // the page bytes, so identical bytes compiled for the other mode are
        // genuinely different code. Skipping here would leave the wrong
        // mode's function installed — silent FP register-half corruption,
        // since emit_fr_mode_guard suppresses itself when CU1 is clear.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        counter.store(6, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&words, &entries, 6, true),
            "identical bytes in the OTHER FR mode must never be skipped");
        // Same request in the mode it was compiled for still skips, proving
        // the rejection above was the FR check and nothing else.
        assert!(page.try_skip_redundant_compile(&words, &entries, 6, false),
            "the same request in the compiled-for mode must still skip");
    }

    #[test]
    fn a_wanted_entry_point_outside_published_coverage_forces_a_compile() {
        // The installed function's dispatch switch only has cases for the
        // entries its own compile covered. An entry outside that set cannot
        // be served no matter how identical the bytes are — skipping would
        // leave every dispatch into it falling through to the interpreter
        // forever while `entry_gen` claimed it was runnable.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        let mut wants_more = entries;
        wants_more[0] |= 1u64 << 20; // offset 20 was never compiled
        counter.store(6, Ordering::Relaxed);

        assert!(!page.try_skip_redundant_compile(&words, &wants_more, 6, false),
            "asking for an uncompiled entry point must force a real compile");
    }

    #[test]
    fn a_killed_entry_forces_a_compile_even_with_identical_bytes() {
        // `kill` clears a `compiled` bit without dropping `func` (an
        // FR-guard bail, or a denylist landing on a previously-published
        // offset). The skip path re-checks coverage against the LIVE
        // `compiled` bitmap under the lock precisely so a killed entry
        // cannot be silently resurrected by a byte comparison.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        page.kill(10);
        counter.store(6, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&words, &entries, 6, false),
            "an entry killed since publish must not be revived by a skip");
    }

    #[test]
    fn a_snapshot_is_unusable_until_its_publish_commits_it() {
        // Staging alone must never authorize a skip: on the deferred path
        // the matching publish can be gap-blocked in the seal queue for a
        // while, and a snapshot consulted in that window would re-validate
        // a `func` that was never installed.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);

        page.stage_compile_snapshot(&words, &used, &entries, 5, false);
        assert!(!page.has_compile_snapshot(), "a staged-but-uncommitted record must not be usable");
        assert!(!page.try_skip_redundant_compile(&words, &entries, 5, false),
            "a staged-but-uncommitted record must never authorize a skip");

        assert!(page.publish(&entries, 0x1000usize as *const (), 5, 1, 0, false));
        page.commit_compile_snapshot(5, &entries, false);
        assert!(page.has_compile_snapshot(), "the commit must make it usable");
    }

    #[test]
    fn staging_a_recompile_keeps_the_installed_codes_snapshot_until_it_publishes() {
        // The measured churn source (jitv2::hashstats on IP28): staging used to
        // overwrite the only record, so while the new compile waited in the
        // seal queue, or after its publish failed, the page kept code that no
        // record vouched for and every request recompiled identical bytes.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        // A recompile is staged (its publish is not in yet).
        counter.store(6, Ordering::Relaxed);
        page.stage_compile_snapshot(&words, &used, &entries, 6, false);
        assert!(page.has_compile_snapshot(),
            "staging must not invalidate the record of the code still installed");
        assert!(page.try_skip_redundant_compile(&words, &entries, 6, false),
            "the installed code still matches the bytes, so a skip is still right");

        // Its publish fails (the skip above already took generation 6).
        assert!(!page.publish(&entries, 0x2000usize as *const (), 6, 1, 0, false));
        assert!(page.has_compile_snapshot(),
            "a failed publish leaves the old code, and its record, in place");
    }

    #[test]
    fn a_publish_the_staged_record_does_not_describe_invalidates_the_committed_one() {
        // A publish landed but is not the compile that was staged: the
        // installed code is now something no record describes.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        counter.store(6, Ordering::Relaxed);
        page.stage_compile_snapshot(&words, &used, &entries, 6, false);
        let mut other_entries = [0u64; BITMAP_WORDS];
        other_entries[0] |= 1u64 << 42;
        assert!(page.publish(&other_entries, 0x3000usize as *const (), 6, 1, 0, false));
        page.commit_compile_snapshot(6, &other_entries, false);
        assert!(!page.has_compile_snapshot(),
            "the committed record described the replaced code and must go");
    }

    #[test]
    fn a_commit_for_a_different_compile_drops_the_staged_record() {
        // A commit must only ever validate its OWN record. A page is never
        // compiled twice at once (`page_scheduled` is held for the whole
        // compile), so the reachable way the staging slot stops matching is a
        // reset landing under an in-flight compile — covered separately by
        // `a_late_commit_cannot_revive_a_snapshot_invalidated_by_a_reset`.
        // This pins the matching rule itself, which is what makes that
        // protection work: a commit whose identity doesn't match what was
        // staged drops the record rather than validating it. Costs one
        // future recompile; committing it would let a later skip compare
        // against the wrong bytes.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);

        page.stage_compile_snapshot(&words, &used, &entries, 5, false);
        assert!(page.publish(&entries, 0x1000usize as *const (), 5, 1, 0, false));
        // A different compile's publish tries to vouch for it.
        let mut other_entries = [0u64; BITMAP_WORDS];
        other_entries[0] |= 1u64 << 42;
        page.commit_compile_snapshot(5, &other_entries, false);
        assert!(!page.has_compile_snapshot(),
            "a commit whose entries don't match the staged record must drop it, not validate it");
    }

    #[test]
    fn dropping_the_compiled_function_invalidates_the_snapshot() {
        // A snapshot outliving the `func` it describes would let a skip
        // advance `entry_gen` over a null function and send dispatch into
        // nothing. Every reset that clears `func` must clear the snapshot.
        let counter = AtomicU64::new(5);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        page.reset_to_unclaimed();
        assert!(!page.has_compile_snapshot(),
            "reset_to_unclaimed drops func, so the snapshot must go with it");
    }

    #[test]
    fn a_late_commit_cannot_revive_a_snapshot_invalidated_by_a_reset() {
        // A compile in flight across a flush comes back and tries to commit.
        // The reset poisoned `staged_gen`, so the commit must find no match
        // and leave the snapshot invalid rather than re-validating a record
        // for a function the flush threw away.
        let counter = AtomicU64::new(5);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);

        page.stage_compile_snapshot(&words, &used, &entries, 5, false);
        page.reset_to_unclaimed();
        page.commit_compile_snapshot(5, &entries, false);
        assert!(!page.has_compile_snapshot(),
            "a commit arriving after a reset must not revive the snapshot");
    }

    #[test]
    fn a_skip_never_moves_entry_gen_backwards() {
        // A concurrent publish may have already installed newer coverage at
        // a higher generation. Rewinding `entry_gen` to this compile's older
        // snapshot would invalidate that publish's code while leaving it
        // installed.
        let counter = AtomicU64::new(9);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 9, false, 0x1000usize as *const ());

        // A stale request (snapshot taken at gen 7) arrives after the page
        // has already published at gen 9.
        assert!(!page.try_skip_redundant_compile(&words, &entries, 7, false),
            "a skip must refuse to rewind entry_gen");
        assert_eq!(page.entry_gen(), 9, "entry_gen must be left where the newer publish put it");
    }

    #[test]
    fn a_skip_requeues_the_entries_its_narrowing_drops() {
        // Companion to `a_skip_must_not_leave_coverage_the_check_never_verified`:
        // narrowing `compiled` to the verified set is necessary, but it must
        // not silently strand the entries it drops. Unlike `publish`'s
        // equivalent branch, the bytes here are provably UNCHANGED, so a
        // dropped entry is still legitimately wanted — nothing re-requested
        // it only because it was already covered. Folding it back into
        // `requested` is what gives it a path to a real recompile, the same
        // guard `publish` applies for a dropped-but-still-wanted entry.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10, 40], &[10, 11, 40, 41]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        // Both were covered, so neither is outstanding as a request.
        page.clear_requested_bits(&entries);
        assert!(!page.is_requested(40));

        counter.store(6, Ordering::Relaxed);
        let mut asks_for_one = [0u64; BITMAP_WORDS];
        asks_for_one[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &asks_for_one, 6, false));

        assert!(!page.is_runnable(40), "the dropped entry must not stay published");
        assert!(page.is_requested(40),
            "the dropped entry must be re-requested, or it is stranded on the interpreter \
             with no automatic path back to compiled dispatch");
    }

    #[test]
    fn churn_counters_reset_on_flush_but_rejected_compiles_does_not() {
        // The churn tallies are per-flush-epoch on purpose: a boot storm
        // racks up ~286k skips, and as a lifetime total that would sit in
        // `j2 status` forever instead of describing current behaviour. This
        // pins that, and pins the contrast with `rejected_compiles`, which
        // deliberately DOES survive a flush (pure churn history) — the two
        // sit next to each other and the difference is easy to "tidy up" by
        // mistake later.
        let counter = AtomicU64::new(5);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        page.mark_redundant_compile_skipped();
        page.mark_redundant_compile_rejected();
        page.mark_analyze_rejected(); // bumps rejected_compiles
        assert_eq!((page.redundant_skipped(), page.redundant_rejected()), (1, 1));

        page.reset_to_unclaimed();
        assert_eq!((page.redundant_skipped(), page.redundant_rejected()), (0, 0),
            "reset_to_unclaimed must clear churn counters");
        assert_eq!(page.rejected_compiles(), 1,
            "rejected_compiles is lifetime history and must NOT be reset by a flush");

        // An evicted-and-recycled slot must not carry the previous tenant's
        // counts into a different physical page either.
        page.mark_redundant_compile_skipped();
        page.reset_to_unclaimed();
        assert_eq!(page.redundant_skipped(), 0,
            "a recycled slot must not inherit the previous page's churn counts");
    }

    #[test]
    fn a_skip_must_not_leave_coverage_the_check_never_verified() {
        // THE CHURN BUG. After a gen bump `same_gen` is false, so the
        // caller's candidate set is `requested & denied` — it deliberately
        // EXCLUDES `compiled`. A real compile then walks exactly that set and
        // `publish` REPLACES `compiled` with it wholesale (is_real_invalidation).
        //
        // A skip that only bumps `entry_gen` leaves the OLD, wider `compiled`
        // in place, so `is_runnable` starts reporting true for entries this
        // check never verified. Dispatch calls `func` for one, the installed
        // function's switch has no case for it, and it returns EXEC_FALLBACK
        // — forever, on every arrival, plus a re-request each time. That is
        // strictly worse than having recompiled: high skip counts, lower
        // throughput.
        //
        // So a skip must end with the same coverage a real publish of the
        // same candidate set would have produced.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        // Page was compiled with two entry points.
        let (words, used, entries) = churn_fixture(&[10, 40], &[10, 11, 40, 41]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        // Gen bumps; only offset 10 is freshly requested this round.
        counter.store(6, Ordering::Relaxed);
        let mut asks_for_one = [0u64; BITMAP_WORDS];
        asks_for_one[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &asks_for_one, 6, false));

        assert!(page.is_runnable(10), "the verified entry must be runnable");
        assert!(!page.is_runnable(40),
            "offset 40 was NOT in this round's candidate set, so a real compile would have \
             dropped it from `compiled` — a skip that leaves it published sends every \
             dispatch into EXEC_FALLBACK forever");
    }

    #[test]
    fn a_re_requested_dropped_entry_can_still_skip_against_the_same_function() {
        // The snapshot's `entries` records what the installed `func` CAN
        // serve (its switch cases), which does not change while that `func`
        // stays installed. `compiled` records what is currently advertised,
        // which a skip narrows to the set it verified. Keeping those two
        // separate is what lets an entry dropped by one skip come back and be
        // served by a later one WITHOUT a recompile — narrowing the snapshot
        // to match `compiled` would forfeit exactly the reuse this whole
        // mechanism exists for.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10, 40], &[10, 11, 40, 41]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        // First skip verifies only offset 10; 40 is dropped and re-requested.
        counter.store(6, Ordering::Relaxed);
        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &only_10, 6, false));
        assert!(!page.is_runnable(40));

        // 40 comes back around. The same `func` still has a case for it, so
        // this must skip too rather than forcing a recompile.
        counter.store(7, Ordering::Relaxed);
        let mut both = [0u64; BITMAP_WORDS];
        both[0] |= (1u64 << 10) | (1u64 << 40);
        assert!(page.try_skip_redundant_compile(&words, &both, 7, false),
            "an entry the installed function can still serve must be re-coverable without a recompile");
        assert!(page.is_runnable(10));
        assert!(page.is_runnable(40), "both entries must be advertised again");
    }

    #[test]
    fn a_stale_denial_carried_by_a_skip_self_heals_on_the_next_real_compile() {
        // A skip deliberately does NOT reset `denied`, unlike `publish` on a
        // gen bump. `publish`'s reset is justified as "the bytes changed, so
        // old denials deserve a fresh chance" — and on the skip path the
        // DECODED bytes provably did not change, so the denials that matter
        // (ones earned while walking those bytes) are still valid.
        //
        // The gap: a denial can be earned against bytes outside `used`, which
        // a skip never compares. Carrying it forward is conservative in the
        // SAFE direction — an offset stays on the interpreter that might have
        // compiled. It is lost opportunity, never wrong code, and it is
        // bounded because it self-heals the moment a decoded word changes and
        // a real compile runs.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        counter.store(6, Ordering::Relaxed);
        page.denylist(40);

        // Skips keep the denial while the decoded words hold still.
        counter.store(7, Ordering::Relaxed);
        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;
        assert!(page.try_skip_redundant_compile(&words, &only_10, 7, false));
        assert!(page.is_denylisted(40), "a skip must not resurrect denials it did not re-derive");

        // A decoded word finally changes: no skip, real compile, and
        // `publish`'s own is_real_invalidation reset clears the denial.
        let mut changed = words;
        changed[11] = 0xdead_beef;
        counter.store(8, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&changed, &only_10, 8, false),
            "a changed decoded word must force a real compile");
        assert!(page.publish(&only_10, 0x2000usize as *const (), 8, 1, 0, false));
        assert!(!page.is_denylisted(40),
            "the real compile's publish must reset denied, healing any stale denial a skip carried");
    }

    #[test]
    fn a_rewritten_excluded_entry_word_forces_a_real_compile() {
        // Every production denylist is a COMPILER verdict, with three
        // reasons: (1) the entry word itself was excluded, (2) the merged
        // region was too short, (3) Cranelift declined. For (2) and (3) the
        // offending code is a COVERED offset, so its word is in `used` by
        // construction and a rewrite fails the byte compare on its own.
        //
        // (1) is the odd one out: `visit` returns false, so
        // `walk_multi_entry` never marks that word visited and
        // `instrs_linear` never reports it. `prepare_multi_entry_compile`
        // therefore folds the excluded entry's own word into `used`
        // explicitly — without it, the guest could rewrite that instruction
        // into something compilable and the sticky denial would survive
        // every skip forever, when a real compile would have reset `denied`.
        //
        // This test drives the page API directly with a `used` mask built the
        // way that fold builds it, i.e. covered words PLUS the excluded
        // entry's word.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        // Covered {10} (words 10,11); offset 40 was a candidate the walk
        // excluded — so 40 is denied AND 40 is folded into `used`.
        let (words, used, entries) = churn_fixture(&[10], &[10, 11, 40]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        page.denylist(40);

        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;

        // Unchanged: the skip is still available (the denial is still valid
        // against these exact bytes, so re-deriving it would be pure churn).
        counter.store(6, Ordering::Relaxed);
        assert!(page.try_skip_redundant_compile(&words, &only_10, 6, false),
            "an unchanged excluded word must not cost a recompile");

        // Rewritten: the excluded word is in `used`, so the compare fails and
        // a real compile runs — which resets `denied` and retries offset 40.
        let mut rewritten = words;
        rewritten[40] = 0x2021_0000;
        counter.store(7, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&rewritten, &only_10, 7, false),
            "rewriting an excluded+denied entry word must force a real compile, or the \
             sticky denial outlives the instruction that earned it");
    }

    #[test]
    fn a_denied_offsets_word_stays_compared_across_a_later_same_gen_compile() {
        // `denied` is sticky page state that outlives the compile that set
        // it, so "this compile's denials are covered by this compile's
        // `used`" is NOT sufficient. A later same-generation compile (pure
        // added coverage — it does not reset `denied`) re-stages the
        // snapshot, and its own walk has no reason to touch a word denied
        // long ago: `denied` masks that offset out of the candidate set, so
        // nothing walks it. The denial would stay live while the evidence for
        // it dropped out of the snapshot, and from then on no skip could ever
        // notice the guest rewriting that instruction into something
        // compilable — the denial outliving the code that earned it,
        // permanently.
        //
        // `prepare_multi_entry_compile` therefore folds the WHOLE `denied`
        // set into `used` when it stages, not just the offsets it denied
        // itself. This test models that second compile's staging directly.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);

        // Compile A covers {10}, denies 40.
        let (words, used_a, entries_a) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used_a, &entries_a, 5, false, 0x1000usize as *const ());
        page.denylist(40);

        // Compile B, SAME generation: adds entry 20. Its walk covers only
        // {10,11,20,21} — nothing walks word 40. Staging applies the fold,
        // so word 40 is still carried into `used`.
        let (_w, mut used_b, entries_b) = churn_fixture(&[10, 20], &[10, 11, 20, 21]);
        for (i, w) in page.snapshot_denied_raw().iter().enumerate() {
            used_b[i] |= !*w;
        }
        page.stage_compile_snapshot(&words, &used_b, &entries_b, 5, false);
        assert!(page.publish(&entries_b, 0x2000usize as *const (), 5, 1, 0, false));
        page.commit_compile_snapshot(5, &entries_b, false);
        assert!(page.is_denylisted(40), "a same-gen publish must not reset denied");

        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;

        // Unchanged bytes still skip — the fold must not cost churn.
        counter.store(6, Ordering::Relaxed);
        assert!(page.try_skip_redundant_compile(&words, &only_10, 6, false),
            "folding denied words in must not break the ordinary skip");

        // Rewriting the denied word is now noticed, even though the compile
        // that staged this snapshot never walked it.
        let mut rewritten = words;
        rewritten[40] = 0x2021_0000;
        counter.store(7, Ordering::Relaxed);
        assert!(!page.try_skip_redundant_compile(&rewritten, &only_10, 7, false),
            "a denied offset's word must stay compared across later compiles, or the denial \
             outlives the instruction that earned it with no way back");
    }

    #[test]
    fn every_cpu_thread_denial_path_also_kills_which_blocks_the_skip() {
        // Why the churn-avoidance `denied` fold does not need to worry about
        // denials originating on the CPU thread.
        //
        // Structurally, it could not matter anyway: the CPU thread only ever
        // reaches an offset it was actually EXECUTING, which means that
        // offset was published and dispatchable, which means it was in
        // `covered` at its last compile — so its word is already in `used` by
        // construction and its bytes are already compared.
        //
        // Mechanically there is also a backstop, which is what this test
        // pins: every CPU-thread path that touches denial state (`j2 deny`
        // in the monitor, and the FR-guard's `jit_kill_entry`) also calls
        // `kill()`, and `kill()` bumps `kills`, which refuses
        // the next skip outright so a real compile re-derives everything.
        //
        // If a future CPU-thread path ever denylists WITHOUT killing, this
        // test won't catch it directly — but the invariant it records is the
        // thing to re-check at that point.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;
        counter.store(6, Ordering::Relaxed);

        // The `j2 deny` / FR-guard shape: denylist + kill together.
        page.denylist(40);
        page.kill(40);

        assert!(!page.try_skip_redundant_compile(&words, &only_10, 6, false),
            "a kill from the CPU thread must block the next skip, so a real compile re-derives \
             coverage and denial state rather than a skip carrying stale assumptions forward");
    }

    #[test]
    fn a_compile_path_denial_does_not_block_a_skip_of_the_installed_code() {
        // The compile path denies and un-publishes an entry together
        // (`kill_denylisted`). That must not poison the page's snapshot: the
        // denial already keeps the offset out of every skip's candidate set.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());

        page.denylist(40);
        page.kill_denylisted(40);
        let mut only_10 = [0u64; BITMAP_WORDS];
        only_10[0] |= 1u64 << 10;
        counter.store(6, Ordering::Relaxed);
        assert!(page.try_skip_redundant_compile(&words, &only_10, 6, false),
            "a compile-path denial must leave the installed code's snapshot usable");
        assert!(!page.snapshot_compiled().iter().enumerate()
            .any(|(i, w)| i == 0 && w & (1u64 << 40) != 0),
            "the denied offset must not be re-advertised");
    }

    #[test]
    fn a_skip_does_not_reset_the_denylist() {
        // A gen bump normally clears `denied` (§13.6: new bytes deserve a
        // fresh chance). A skip proves the opposite — the decoded bytes are
        // IDENTICAL — so an offset denied against them is still correctly
        // denied. Clearing it would re-walk and re-reject it on every gen
        // bump, which is exactly the churn this exists to remove.
        let counter = AtomicU64::new(5);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let (words, used, entries) = churn_fixture(&[10], &[10, 11]);
        publish_with_snapshot(&page, &words, &used, &entries, 5, false, 0x1000usize as *const ());
        page.denylist(200);
        assert!(page.is_denylisted(200));

        counter.store(6, Ordering::Relaxed);
        assert!(page.try_skip_redundant_compile(&words, &entries, 6, false));
        assert!(page.is_denylisted(200),
            "a skip must leave the denylist alone — the bytes are provably unchanged");
    }

    #[test]
    fn denylist_bit_is_independent_of_valid_bit() {
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 7usize;
        page.denylist(offset);
        assert!(page.is_denylisted(offset));
        assert!(!page.is_runnable(offset), "denylisting must not itself mark an entry valid");
    }

    #[test]
    fn denylist_alone_does_not_clear_an_already_published_entry() {
        // The real reason `prepare_multi_entry_compile` (comp.rs) must pair
        // every `denylist` call on a previously-`compiled` offset with an
        // explicit `kill()`: dispatch (`exec_decoded`'s gate, mips_exec.rs)
        // only ever consults `is_published`/`is_runnable`, never
        // `is_denylisted` — denylisting purely gates whether a FUTURE
        // compile is offered this offset as a candidate again, it does
        // nothing to stop dispatch from jumping into whatever `func`
        // currently claims to cover it. An offset that was published, then
        // denylisted without also being killed, would still read
        // `is_runnable() == true` forever, dispatchable into a `func`
        // that may no longer actually have a case for it.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        let offset = 7usize;
        let mut bits = [0u64; BITMAP_WORDS];
        bits[offset / 64] |= 1u64 << (offset % 64);
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0, false));
        assert!(page.is_runnable(offset));

        page.denylist(offset);
        assert!(page.is_denylisted(offset));
        assert!(page.is_runnable(offset),
            "denylist alone must not clear a previously-published entry's valid bit — \
             proving kill() is the mechanism that actually has to do that");

        page.kill(offset);
        assert!(!page.is_runnable(offset), "kill() is what actually un-publishes the entry");
    }

    #[test]
    fn compiles_since_flush_counts_successful_publishes_and_is_cleared_by_both_reset_paths() {
        // Always-present, opposite-lifetime counterpart to the developer-only
        // compile_count: answers "has this page compiled at all since its
        // last flush/claim" without needing a developer build.
        let counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert_eq!(page.compiles_since_flush(), 0);

        let mut bits = [0u64; BITMAP_WORDS];
        bits[0] |= 1u64 << 4;
        assert!(page.publish(&bits, 0x1000 as *const (), 0, 1, 0, false));
        assert_eq!(page.compiles_since_flush(), 1);

        // A redundant publish (nothing new, same generation) must not count.
        assert!(!page.publish(&bits, 0x1000 as *const (), 0, 1, 0, false));
        assert_eq!(page.compiles_since_flush(), 1, "a subsumed/no-op publish must not increment the counter");

        let mut bits2 = [0u64; BITMAP_WORDS];
        bits2[0] |= 1u64 << 8;
        assert!(page.publish(&bits2, 0x1000 as *const (), 0, 1, 0, false));
        assert_eq!(page.compiles_since_flush(), 2);

        page.reset_to_unclaimed();
        assert_eq!(page.compiles_since_flush(), 0, "reset_to_unclaimed must clear it");

        assert!(page.publish(&bits, 0x2000 as *const (), 0, 1, 0, false));
        assert_eq!(page.compiles_since_flush(), 1);
        page.reset_to_unclaimed();
        assert_eq!(page.compiles_since_flush(), 0, "reset_to_unclaimed must clear it too");
    }

    #[test]
    fn rejected_compiles_accumulates_and_survives_both_reset_paths() {
        // Distinguishes "this page has been attempted repeatedly and keeps
        // getting rejected" from "never attempted at all" — compiles_since_flush=0
        // alone can't tell those apart. Lifetime total, same reasoning as
        // compile_count's own doc comment: unlike compiles_since_flush, this
        // is NOT cleared by either reset path.
        let counter = AtomicU64::new(0);
        let page = PhysicalCodePage::new(0, &counter as *const AtomicU64);
        assert_eq!(page.rejected_compiles(), 0);

        page.mark_analyze_rejected();
        assert_eq!(page.rejected_compiles(), 1);
        page.mark_analyze_rejected();
        page.mark_codegen_rejected();
        assert_eq!(page.rejected_compiles(), 3, "the umbrella total must sum both stage-specific counters");
    }



    /// Minimal BusDevice whose gen_ptr always returns the same fixed counter,
    /// standing in for a real RAM/ROM device in these pool-only tests.
    struct FakeDevice(AtomicU64);
    impl BusDevice for FakeDevice {
        fn read8(&self, _addr: u64) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u64, _val: u8) -> u32 { crate::traits::BUS_ERR }
        fn read16(&self, _addr: u64) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u64, _val: u16) -> u32 { crate::traits::BUS_ERR }
        fn read32(&self, _addr: u64) -> BusRead32 { BusRead32::err() }
        fn write32(&self, _addr: u64, _val: u32) -> u32 { crate::traits::BUS_ERR }
        fn read64(&self, _addr: u64) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u64, _val: u64) -> u32 { crate::traits::BUS_ERR }
        fn gen_ptr(&self, _addr: u64) -> *const AtomicU64 { &self.0 as *const AtomicU64 }
        fn code_page(&self, addr: u64) -> Option<crate::traits::CodePage> { Some(crate::traits::CodeKey::flat_page(addr, self.gen_ptr(addr))) }
    }

    fn at(cpu: usize, pfn: Pfn) -> PageAt {
        PageAt { cpu, space: 0, pfn, uncached: false }
    }

    /// What the executor's lock-free lookup does with a frame table entry:
    /// the page it names, if the entry is for the slot's present tenant.
    fn fast_lookup(jit: &mut Jitv2, pfn: Pfn) -> Option<PageSlot> {
        let (map, pages) = jit.fast_lookup_ptrs(0);
        let entry = unsafe { (*map.add(page_key(pfn, false, false) as usize)).load(Ordering::Acquire) };
        if entry == PFN_MAP_EMPTY {
            return None;
        }
        let slot = frame_entry_slot(entry);
        frame_entry_current(entry, unsafe { &*pages.add(slot as usize) }.tenancy()).then_some(slot)
    }

    /// A full pool gives a new page the slot of one nobody is using, and the
    /// frame that led to the old tenant no longer leads to the slot.
    #[test]
    fn a_full_pool_recycles_a_slot_nobody_is_on() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(3);
        let slots: Vec<PageSlot> = (0..3).map(|pfn| jit.page_at(at(0, pfn), &dev, false).unwrap()).collect();
        assert_eq!(jit.pages_used(), 3);
        // The CPU is on the last page it claimed. Every page is marked as
        // entered, so the hand's first pass only clears marks.
        let fourth = jit.page_at(at(0, 3), &dev, false).expect("a slot nobody is on is recycled");
        assert_eq!(fourth, slots[0], "the hand takes the first slot that is not in use");
        assert_eq!(jit.pages[fourth as usize].key, CodeKey::flat(3 << 12));
        assert_eq!(jit.pages_used(), 3);
        assert_eq!(jit.stats.pages_recycled.load(Ordering::Relaxed), 1);
        assert_eq!(fast_lookup(&mut jit, 3), Some(fourth));
        assert_eq!(fast_lookup(&mut jit, 0), None, "the old tenant's frame must not reach the new tenant");
        assert_eq!(fast_lookup(&mut jit, 1), Some(slots[1]), "the pages that stayed are still found without the lock");
        // The evicted memory comes back as a page of its own.
        let again = jit.page_at(at(0, 0), &dev, false).unwrap();
        assert_eq!(jit.pages[again as usize].key, CodeKey::flat(0));
        assert_ne!(again, fourth, "the CPU was on that slot when it asked");
    }

    /// The page a CPU is on, and a page whose compile is in flight, are
    /// never taken: the CPU holds a pointer to the one, a worker to the other.
    #[test]
    fn a_slot_in_use_is_not_recycled() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(3);
        let on = jit.page_at(at(1, 0), &dev, false).unwrap();
        let queued = jit.page_at(at(0, 1), &dev, false).unwrap();
        let idle = jit.page_at(at(0, 2), &dev, false).unwrap();
        assert!(jit.pages[queued as usize].try_schedule_page());
        // CPU 0 moves on, twice; CPU 1 stays where it is.
        let next = jit.page_at(at(0, 3), &dev, false);
        assert_eq!(next, None, "CPU 0 is on the only slot nothing else holds");
        // Once CPU 0 is off its page that slot can go.
        jit.cpu_slots[0].0.store(NO_SLOT, Ordering::SeqCst);
        let next = jit.page_at(at(0, 3), &dev, false).unwrap();
        assert_eq!(next, idle);
        assert_eq!(jit.pages[on as usize].key, CodeKey::flat(0), "CPU 1's page was left alone");
        assert_eq!(jit.pages[queued as usize].key, CodeKey::flat(1 << 12), "the queued page was left alone");
        // The compile ends and CPU 1 leaves: both slots can go.
        jit.pages[queued as usize].clear_scheduled();
        jit.cpu_slots[1].0.store(NO_SLOT, Ordering::SeqCst);
        assert!(jit.page_at(at(0, 4), &dev, false).is_some());
        assert!(jit.page_at(at(0, 5), &dev, false).is_some());
    }

    /// A page entered since the hand last passed is spared once.
    #[test]
    fn the_hand_spares_a_page_entered_since_its_last_pass() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(3);
        let slots: Vec<PageSlot> = (0..3).map(|pfn| jit.page_at(at(0, pfn), &dev, false).unwrap()).collect();
        // First recycle clears every mark and takes slot 0.
        assert_eq!(jit.page_at(at(0, 3), &dev, false), Some(slots[0]));
        // Page 1 is entered again (by another CPU, which then leaves).
        assert_eq!(jit.page_at(at(1, 1), &dev, false), Some(slots[1]));
        jit.cpu_slots[1].0.store(NO_SLOT, Ordering::SeqCst);
        // The hand stands at slot 1: it is marked, so slot 2 goes instead.
        assert_eq!(jit.page_at(at(0, 4), &dev, false), Some(slots[2]));
        assert_eq!(jit.pages[slots[1] as usize].key, CodeKey::flat(1 << 12));
    }

    /// Code compiled for a slot's old tenant must not be installed on the new
    /// one: its publish waited in the seal queue while the slot was recycled.
    #[test]
    fn a_publish_for_a_recycled_slot_is_refused() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(2);
        let slot = jit.page_at(at(0, 0), &dev, false).unwrap();
        let tenancy = jit.pages[slot as usize].tenancy();
        jit.page_at(at(0, 1), &dev, false).unwrap();
        // Page 0's slot goes to page 2.
        assert_eq!(jit.page_at(at(0, 2), &dev, false), Some(slot));
        let page = &jit.pages[slot as usize];
        assert_ne!(page.tenancy(), tenancy);
        let mut bits = [0u64; BITMAP_WORDS];
        bits[0] = 1;
        let arena = crate::cpu::jitv2::paged_memory::PagedArenaState::UNTRACKED;
        assert!(!page.publish_from(&arena, tenancy, &bits, 0x1000 as *const (), page.current_gen(), 1, 16, false));
        assert!(!page.has_code());
        assert!(page.publish_from(&arena, page.tenancy(), &bits, 0x1000 as *const (), page.current_gen(), 1, 16, false));
    }

    /// A request can outlive its page (the hot lane's copy). The worker must
    /// not read a slot that has gone to other memory, and must hold a slot it
    /// does read.
    #[test]
    fn a_worker_pins_the_page_it_compiles_and_drops_a_request_for_a_recycled_slot() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(2);
        let slot = jit.page_at(at(0, 0), &dev, false).unwrap();
        let asked = jit.pages[slot as usize].tenancy();
        // A worker takes the request: the slot is held although no CPU
        // scheduled it (the hot lane's copy, after the first compile ended).
        assert!(jit.pages[slot as usize].pin_for_compile(asked));
        let other = jit.page_at(at(0, 1), &dev, false).unwrap();
        assert_eq!(jit.page_at(at(0, 2), &dev, false), None, "the only other slot is being compiled");
        jit.pages[slot as usize].clear_scheduled();
        // Now it can go, and a request made for the old tenant is stale.
        jit.cpu_slots[0].0.store(other, Ordering::SeqCst);
        assert_eq!(jit.page_at(at(0, 2), &dev, false), Some(slot));
        assert!(!jit.pages[slot as usize].pin_for_compile(asked));
        assert!(!jit.pages[slot as usize].is_scheduled(), "a stale request leaves no flag behind");
        // A flag the new tenant set itself survives a stale request.
        assert!(jit.pages[slot as usize].try_schedule_page());
        assert!(!jit.pages[slot as usize].pin_for_compile(asked));
        assert!(jit.pages[slot as usize].is_scheduled());
    }

    /// A recycled slot starts clean: no code, no entry requests, no denials
    /// from its last tenant.
    #[test]
    fn a_recycled_slot_starts_clean() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(2);
        let slot = jit.page_at(at(0, 0), &dev, false).unwrap();
        {
            let page = &jit.pages[slot as usize];
            let mut bits = [0u64; BITMAP_WORDS];
            bits[0] = 1;
            assert!(page.publish(&bits, 0x1000 as *const (), page.current_gen(), 1, 16, false));
            page.mark_requested(5);
            page.denylist(9);
        }
        jit.page_at(at(0, 1), &dev, false).unwrap();
        assert_eq!(jit.page_at(at(0, 2), &dev, false), Some(slot));
        let page = &jit.pages[slot as usize];
        assert!(!page.has_code() && !page.is_runnable(0) && !page.is_requested(5) && !page.is_denylisted(9));
        assert_eq!(page.func(), NO_CODE as *const ());
    }

    #[test]
    fn pfn_of_stops_at_the_pfn_map() {
        let octane = frames_for(&crate::ip30::memory_map(&[2048, 2048, 2048, 2048]));
        assert_eq!(pfn_of(0x1fc0_0004, octane), Some(0x1fc00));
        assert_eq!(pfn_of(0xFFFF_F000, octane), Some(0xFFFFF));
        assert_eq!(pfn_of(0x2_1FFF_F000, octane), Some(0x21FFFF));
        assert_eq!(pfn_of(0x2_2000_0000, octane), None);
        assert_eq!(pfn_of(0xFF_FFFF_F000, octane), None);
        let indy = frames_for(&crate::ip22_bus::memory_map(false));
        assert_eq!(pfn_of(0x2FFF_F000, indy), Some(0x2FFFF));
        assert_eq!(pfn_of(0x3000_0000, indy), None);
    }

    #[test]
    fn page_for_allocates_once_and_caches_by_pfn() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(4);
        let slot_a = jit.page_for(3, (3 * PAGE_SIZE) as u64, &dev, false).unwrap();
        let slot_b = jit.page_for(3, (3 * PAGE_SIZE) as u64, &dev, false).unwrap();
        assert_eq!(slot_a, slot_b, "second arrival at the same pfn must reuse the slot");
        let slot_c = jit.page_for(4, (4 * PAGE_SIZE) as u64, &dev, false).unwrap();
        assert_ne!(slot_a, slot_c);
    }

    #[test]
    fn page_for_returns_none_when_pool_exhausted() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(1);
        assert!(jit.page_for(0, (0) as u64, &dev, false).is_some());
        assert!(jit.page_for(1, (PAGE_SIZE) as u64, &dev, false).is_none(), "pool of 1 must reject a second distinct pfn");
    }

    #[test]
    fn mega_flush_resets_all_pages_and_frees_all_slots() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(4);
        for i in 0..4 as Pfn {
            jit.page_for(i, (i * PAGE_SIZE) as u64, &dev, false).unwrap();
        }
        assert_eq!(jit.pages_used(), 4);
        // A full pool recycles a slot rather than refuse.
        assert!(jit.page_for(4, (4 * PAGE_SIZE) as u64, &dev, false).is_some());
        assert_eq!(jit.pages_used(), 4);

        jit.mega_flush();
        assert_eq!(jit.pages_used(), 0);

        for i in 0..4 as Pfn {
            assert!(jit.page_for(i + 10, ((i + 10) * PAGE_SIZE) as u64, &dev, false).is_some());
        }
        assert_eq!(jit.pages_used(), 4);
    }

    #[test]
    fn a_page_table_doubles_only_when_split() {
        use crate::memory_map::{MemKind, MemRange, MemoryMap};
        let dev = FakeDevice(AtomicU64::new(0));
        let map = MemoryMap::new(vec![MemRange::new("ram", MemKind::Ram, 0, 64 << 20)]);
        for (split, entries) in [(false, 1), (true, 2)] {
            let mut jit = Jitv2::for_memory(1, &map);
            if split {
                jit.split_uncached_code();
            }
            assert!(jit.fast_lookup_ptrs(0).0.is_null());
            jit.page_for(1, PAGE_SIZE as u64, &dev, false).unwrap();
            assert_eq!(jit.table_bytes(), (64 << 20) / PAGE_SIZE as usize * 4 * entries);
            assert_eq!(size_of_val(&*jit.spaces[0].map.0), jit.table_bytes());
        }
    }

    /// One RAM bank answering `base` bytes up a node's bus.
    struct NodeBus {
        ram: Arc<crate::dev::mem::Memory>,
        base: u64,
    }
    impl BusDevice for NodeBus {
        fn read32(&self, a: u64) -> BusRead32 {
            a.checked_sub(self.base).map_or_else(BusRead32::err, |a| self.ram.read32(a))
        }
        fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
            a.checked_sub(self.base).map_or(std::ptr::null(), |a| self.ram.gen_ptr(a))
        }
        fn code_page(&self, a: u64) -> Option<crate::traits::CodePage> {
            self.ram.code_page(a.checked_sub(self.base)?)
        }
    }

    /// Two nodes that reach one bank at different addresses share its page,
    /// each through its own frame table, and a compile reads it through
    /// whichever CPU's address still decodes to it.
    #[test]
    fn two_address_spaces_share_the_page_of_one_bank() {
        let ram = Arc::new(crate::dev::mem::Memory::new(1));
        let near: Arc<dyn BusDevice> = Arc::new(NodeBus { ram: ram.clone(), base: 0 });
        let far: Arc<dyn BusDevice> = Arc::new(NodeBus { ram: ram.clone(), base: 0x40 * PAGE_SIZE as u64 });
        let mut jit = Jitv2::new(4);
        jit.set_space(1, PFN_MAP_ENTRIES, false);
        jit.buses().set(0, 0, near.clone(), near.clone());
        jit.buses().set(1, 1, far.clone(), far.clone());
        let a = jit.claim_page(PageAt { cpu: 0, space: 0, pfn: 3, uncached: false }, &*near, false);
        let b = jit.claim_page(PageAt { cpu: 1, space: 1, pfn: 0x43, uncached: false }, &*far, false);
        let Claim::Page(slot) = a else { panic!("{a:?}") };
        assert_eq!(b, Claim::Page(slot), "the same bank page from the other node is the same slot");
        // The other node's frame 3 is not this memory, nor any bank: a page
        // of its own, keyed by that node's address.
        let Claim::Page(hole) = jit.claim_page(PageAt { cpu: 1, space: 1, pfn: 3, uncached: false }, &*far, false) else { panic!() };
        assert_ne!(hole, slot);
        assert_eq!(unsafe { &*jit.page_ptr(hole) }.key, CodeKey::at_address(1, 3, false));
        let page = unsafe { &*jit.page_ptr(slot) };
        assert_eq!(page.key, ram.code_page(3 * PAGE_SIZE as u64).unwrap().key);
        assert_eq!((page.addr_of(0), page.addr_of(1)), (Some((3, false)), Some((0x43, false))));
        assert!(std::ptr::eq(page.gen, ram.gen_ptr(3 * PAGE_SIZE as u64)), "counted by the bank's own counter");

        let (_, at) = jit.buses().locate(page, 0).unwrap();
        assert_eq!(at, 3 * PAGE_SIZE as u64);
        // CPU 0's bus stops decoding that address to the bank: its address
        // for the page is stale, and the page is read through CPU 1.
        jit.buses().set(0, 0, far.clone(), far.clone());
        let (_, at) = jit.buses().locate(page, 0).unwrap();
        assert_eq!(at, 0x43 * PAGE_SIZE as u64);

        // A compile needs neither: RAM hands out its storage, so the page is
        // read by its key alone, whatever the buses decode.
        ram.write32(3 * PAGE_SIZE as u64 + 8, 0x1234_5678);
        let mut words = [0u32; crate::cpu::jitv2::ENTRIES_PER_PAGE];
        assert!(!page.words().is_null());
        assert!(matches!(crate::cpu::jitv2::comp::snapshot_page(page, None, &mut words), crate::cpu::jitv2::comp::Snapshot::Taken(_)));
        assert_eq!(words[2], 0x1234_5678);

        // A remap in space 0 forgets its frames and CPU 0's addresses only.
        jit.forget_addresses(0, 0);
        assert_eq!((page.addr_of(0), page.addr_of(1)), (None, Some((0x43, false))));
        assert_eq!(jit.pages_used(), 2, "and drops no page");
        assert_eq!(jit.claim_page(PageAt { cpu: 0, space: 0, pfn: 0x43, uncached: false }, &*far, false), Claim::Page(slot));
        jit.mega_flush();
        assert_eq!(jit.pages_used(), 0);
        assert!(matches!(jit.claim_page(PageAt { cpu: 1, space: 1, pfn: 0x43, uncached: false }, &*far, false), Claim::Page(_)));
    }

    #[test]
    fn fast_lookup_ptrs_lookup_hit() {
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(4);
        let slot = jit.page_for(7, (7 * PAGE_SIZE) as u64, &dev, false).unwrap();
        let (pfn_map_ptr, pages_base) = jit.fast_lookup_ptrs(0);
        unsafe {
            let mapped_slot = (*pfn_map_ptr.add(page_key(7, false, false) as usize)).load(Ordering::Acquire);
            assert_eq!(mapped_slot, slot);
            let page = &*pages_base.add(mapped_slot as usize);
            assert_eq!(page.pfn(), 7);
            assert_eq!(page.key, CodeKey::flat(7 << 12));
        }
    }

    #[test]
    #[cfg(feature = "developer")]
    fn code_size_by_instr_count_buckets_by_instr_count_and_tracks_min_max_sum() {
        // §13: instr_count/code_size are page-level now (one function per
        // page), so distinct buckets need distinct pages, not distinct
        // offsets on the same page.
        let dev = FakeDevice(AtomicU64::new(0));
        let mut jit = Jitv2::new(3);

        let bits0: [u64; BITMAP_WORDS] = { let mut b = [0u64; BITMAP_WORDS]; b[0] |= 1; b };

        // Two pages at instr_count=3 (sizes 100, 300), one at instr_count=5 (size 50).
        let slot_a = jit.page_for(0, (0) as u64, &dev, false).unwrap() as usize;
        assert!(jit.pages[slot_a].publish(&bits0, 0x1000 as *const (), 0, 3, 100, false));
        let slot_b = jit.page_for(1, (PAGE_SIZE) as u64, &dev, false).unwrap() as usize;
        assert!(jit.pages[slot_b].publish(&bits0, 0x2000 as *const (), 0, 3, 300, false));
        let slot_c = jit.page_for(2, (2 * PAGE_SIZE) as u64, &dev, false).unwrap() as usize;
        assert!(jit.pages[slot_c].publish(&bits0, 0x3000 as *const (), 0, 5, 50, false));

        let hist = jit.code_size_by_instr_count();
        let bucket3 = hist[3].expect("instr_count=3 bucket must be present");
        assert_eq!(bucket3.count, 2);
        assert_eq!(bucket3.sum_bytes, 400);
        assert_eq!(bucket3.min_bytes, 100);
        assert_eq!(bucket3.max_bytes, 300);

        let bucket5 = hist[5].expect("instr_count=5 bucket must be present");
        assert_eq!(bucket5.count, 1);
        assert_eq!(bucket5.sum_bytes, 50);
        assert_eq!(bucket5.min_bytes, 50);
        assert_eq!(bucket5.max_bytes, 50);

        assert!(hist[4].is_none(), "instr_count=4 has no published entries");
    }

    #[test]
    #[cfg(feature = "developer")]
    fn record_reject_increments_both_failed_compiles_and_the_reason_bucket() {
        let stats = JitStats::default();
        stats.record_reject(RejectReason::EntryExcluded);
        stats.record_reject(RejectReason::EntryExcluded);
        stats.record_reject(RejectReason::CraneliftVerifierError);

        assert_eq!(stats.failed_compiles.load(Ordering::Relaxed), 3);
        assert_eq!(stats.reject_reasons[RejectReason::EntryExcluded.index()].load(Ordering::Relaxed), 2);
        assert_eq!(stats.reject_reasons[RejectReason::CraneliftVerifierError.index()].load(Ordering::Relaxed), 1);
        assert_eq!(stats.reject_reasons[RejectReason::AnalyzerCodegenDisagreement.index()].load(Ordering::Relaxed), 0);
    }

    #[test]
    fn compile_queue_start_stop_drains_without_hanging() {
        // FakeDevice's read32 always errors, so handle_request bails before
        // any filesystem I/O — no cwd/tempdir isolation needed here.
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, std::sync::Arc::new(JitStats::default()));
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        for i in 0..8u16 {
            let _ = i;
            assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));
        }
        // stop() joins the worker; must return promptly even with requests in flight.
        q.stop();
    }

    /// Dropping a `Jitv2` joins its workers before the page array goes: they
    /// read it, and hold the bus they compile from.
    #[test]
    fn dropping_a_jitv2_joins_its_workers() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut jit = Jitv2::new(4);
        let stats = jit.stats.clone();
        jit.compile_queue.start(dev.clone(), stats);
        assert!(Arc::strong_count(&dev) > 1, "the workers hold the bus");
        drop(jit);
        assert_eq!(Arc::strong_count(&dev), 1, "a worker outlived its Jitv2");
    }

    /// A CPU flushing the pool joins the workers while it holds the pool, so
    /// no worker may wait for the pool. Holding the guard across two flushes
    /// is the case a two-CPU Origin 3200 hung on in its PROM: the second
    /// flush took the pool before the host had run the workers the first
    /// flush started. The pool starts with no owner, so its first workers
    /// have nothing to reach back to and only the restarted ones could block.
    /// Run on a spawned thread so a regression fails here instead of hanging
    /// the test binary.
    #[test]
    fn back_to_back_cpu_flushes_never_join_a_worker_waiting_for_the_pool() {
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
            let jit = Arc::new(Mutex::new(Jitv2::new(4)));
            let stamped = {
                let mut g = jit.lock();
                // A value no fresh `Codegen` has, to show the restarted
                // workers still get what the CPU published.
                g.compile_queue.jit_consts.lock().lwr_keeps_upper = true;
                let stats = g.stats.clone();
                g.compile_queue.start(dev.clone(), stats);
                g.compile_queue.set_owner(Arc::downgrade(&jit));
                unsafe {
                    g.flush_from_cpu_thread(dev.clone());
                    g.flush_from_cpu_thread(dev);
                }
                g.compile_queue.stop().iter().map(|c| c.jit_consts.lwr_keeps_upper).collect::<Vec<_>>()
            };
            let _ = done_tx.send(stamped);
        });
        let stamped = done_rx.recv_timeout(std::time::Duration::from_secs(10))
            .expect("the second flush is still joining a worker that waits for the pool");
        assert!(!stamped.is_empty() && stamped.iter().all(|&s| s), "a restarted worker missed the published constants: {stamped:?}");
    }

    /// A CPU thread as a compile pool sees one: it parks while held, and
    /// stops the way `MipsCpu::halt` does, with the park bit and a join. The
    /// first time it is held it flushes the pool itself instead, as a CPU whose
    /// page pool runs out just as a worker leads an arena flush would. Not
    /// under loom: the forced flush it relies on is a plain-test hook.
    #[cfg(not(loom))]
    struct FlushingCpu {
        word: Arc<AtomicU64>,
        slot: Arc<crate::cpu::smp::ParkSlot>,
        running: Arc<AtomicBool>,
        flushes: Arc<AtomicU32>,
        thread: Mutex<Option<JoinHandle<()>>>,
    }

    #[cfg(not(loom))]
    impl FlushingCpu {
        fn start(jit: Arc<Mutex<Jitv2>>, world: Option<Arc<crate::cpu::smp::World>>, bus: Arc<dyn BusDevice>, word: Arc<AtomicU64>, slot: Arc<crate::cpu::smp::ParkSlot>) -> Arc<Self> {
            use crate::cpu::smp::HOST_PARK_BIT;
            let cpu = Arc::new(Self { word: word.clone(), slot: slot.clone(), running: Arc::new(AtomicBool::new(true)), flushes: Arc::new(AtomicU32::new(0)), thread: Mutex::new(None) });
            let (running, flushes) = (cpu.running.clone(), cpu.flushes.clone());
            slot.set_active(true);
            *cpu.thread.lock() = Some(std::thread::spawn(move || {
                crate::cpu::smp::set_current_cpu(0);
                while running.load(Ordering::SeqCst) {
                    if word.load(Ordering::SeqCst) & HOST_PARK_BIT == 0 {
                        std::thread::yield_now();
                    } else if flushes.load(Ordering::SeqCst) == 0 {
                        // `MipsExecutor::jitv2_flush_from_here`.
                        let flush = || unsafe { jit.lock().flush_from_cpu_thread(bus.clone()) };
                        match &world {
                            Some(w) => w.stop_others(flush),
                            None => flush(),
                        }
                        flushes.fetch_add(1, Ordering::SeqCst);
                    } else {
                        slot.park(&running);
                    }
                }
                slot.set_active(false);
            }));
            cpu
        }
    }

    #[cfg(not(loom))]
    impl Device for FlushingCpu {
        fn step(&self, _cycles: u64) {}
        fn stop(&self) {
            self.running.store(false, Ordering::SeqCst);
            self.word.fetch_or(crate::cpu::smp::HOST_PARK_BIT, Ordering::SeqCst);
            if let Some(t) = self.thread.lock().take() {
                let _ = t.join();
            }
            self.word.fetch_and(!crate::cpu::smp::HOST_PARK_BIT, Ordering::SeqCst);
        }
        fn start(&self) {}
        fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
    }

    /// A worker retires an arena and asks the CPU to pass a batch boundary;
    /// the CPU, at that boundary, flushes the pool itself and joins that
    /// worker (the shape of issue #306, where the worker stopped the CPU
    /// instead of asking). The worker never waits for the CPU, so the join
    /// returns, and the flush leaves the ring whole. Run on a spawned thread
    /// so a regression fails here instead of hanging the test binary.
    #[cfg(not(loom))]
    #[test]
    fn a_cpu_flush_answering_a_retire_request_joins_the_worker_that_asked() {
        let epoch = crate::cpu::jitv2::hashstats::FLUSH_EPOCH.load(Ordering::SeqCst);
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
            let jit = Arc::new(Mutex::new(Jitv2::new(4)));
            let word = Arc::new(AtomicU64::new(0));
            let slot = Arc::new(crate::cpu::smp::ParkSlot::new(0, Arc::as_ptr(&word)));
            let world = Arc::new(crate::cpu::smp::World::solo(slot.clone()));
            let cpu = FlushingCpu::start(jit.clone(), Some(world.clone()), dev.clone(), word.clone(), slot);
            let cpu_device: Arc<dyn Device> = cpu.clone();
            {
                let mut g = jit.lock();
                g.compile_queue.set_cpu(Arc::downgrade(&cpu_device));
                g.compile_queue.set_owner(Arc::downgrade(&jit));
                g.compile_queue.set_world(world);
                let stats = g.stats.clone();
                g.compile_queue.start(dev, stats);
            }
            // The switch that uses the last spare arena retires the oldest.
            for switch in 1..ARENA_COUNT as u64 {
                jit.lock().compile_queue.barrier.force_flush();
                while cpu.flushes.load(Ordering::SeqCst) == 0
                    && jit.lock().compile_queue.ring_status().unwrap().switches < switch
                {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            while cpu.flushes.load(Ordering::SeqCst) == 0 {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            cpu_device.stop();
            let status = jit.lock().compile_queue.ring_status().unwrap();
            let restarted = !Jitv2::stop_pool(&jit).is_empty();
            let _ = done_tx.send((restarted, status));
            drop(word);
        });
        let (restarted, status) = done_rx.recv_timeout(std::time::Duration::from_secs(20))
            .expect("the CPU's flush and the worker that retired an arena are waiting for each other");
        assert!(restarted, "the CPU's flush did not restart the pool");
        let flushes = crate::cpu::jitv2::hashstats::FLUSH_EPOCH.load(Ordering::SeqCst) - epoch;
        assert!(flushes >= 1, "the CPU did not flush");
        assert_eq!((status.active, status.live.len(), status.spare.len(), status.retiring), (0, 0, ARENA_COUNT - 1, None),
            "a flush of the pool leaves every arena empty");
    }

    /// A pool with no CPU to ask: a forced rotation.
    #[cfg(not(loom))]
    fn rotate_and_wait(jit: &Mutex<Jitv2>, switches: u64) {
        jit.lock().compile_queue.barrier.force_flush();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while jit.lock().compile_queue.ring_status().unwrap().switches < switches {
            assert!(std::time::Instant::now() < deadline, "the pool did not switch arenas");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// The ring goes round with nothing stopped: the switch that uses the
    /// last spare retires the oldest arena, whose code leaves its page; the
    /// arena before the active one keeps its code; and the retired arena
    /// comes back empty, under a new epoch.
    #[cfg(not(loom))]
    #[test]
    fn the_pool_retires_its_oldest_arena_and_keeps_the_one_before() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        // Two pages with code, one in the first arena and one in the second.
        // They stand in for the pool's slots, which the sweep walks, and
        // outlive the pool (declared first, dropped last).
        let counter = AtomicU64::new(0);
        let pages: Box<[PhysicalCodePage]> = Box::new([PhysicalCodePage::new(1, &counter), PhysicalCodePage::new(2, &counter)]);
        let (old, kept) = (&pages[0], &pages[1]);
        let jit = Mutex::new(Jitv2::new(4));
        {
            let mut g = jit.lock();
            let stats = g.stats.clone();
            g.compile_queue.set_thread_count(2);
            g.compile_queue.set_pages(&pages);
            g.compile_queue.start(dev, stats);
        }
        let arenas: Vec<PoolArena> = jit.lock().compile_queue.ring.as_ref().unwrap().arenas().to_vec();
        let epochs: Vec<u64> = arenas.iter().map(|a| a.1.epoch()).collect();
        let mut bits = [0u64; BITMAP_WORDS];
        bits[0] = 1;
        let func = 0x1000 as *const ();
        assert!(old.publish_from(&arenas[0].1, old.tenancy(), &bits, func, 0, 1, 16, false));
        assert!(kept.publish_from(&arenas[1].1, kept.tenancy(), &bits, func, 0, 1, 16, false));
        assert_eq!((old.code_epoch(), kept.code_epoch()), (epochs[0], epochs[1]));

        rotate_and_wait(&jit, 1);
        let status = jit.lock().compile_queue.ring_status().unwrap();
        assert_eq!((status.active, status.live.clone(), status.retiring), (1, vec![0], None), "the first switch retires nothing");
        assert!(!arenas[0].1.is_retired());
        assert!(old.is_runnable(0) && kept.is_runnable(0), "a switch takes no code away");

        rotate_and_wait(&jit, 2);
        // With no CPU to wait for, the reclaim follows as soon as both
        // workers have moved.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while jit.lock().compile_queue.ring_status().unwrap().retiring.is_some() {
            assert!(std::time::Instant::now() < deadline, "the retired arena was never reclaimed");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let status = jit.lock().compile_queue.ring_status().unwrap();
        assert_eq!((status.active, status.live, status.spare), (2, vec![1], vec![0]));
        assert_ne!(arenas[0].1.epoch(), epochs[0], "the reclaimed arena is a new epoch");
        assert_eq!(arenas[1].1.epoch(), epochs[1], "the arena before the active one was not touched");
        assert_eq!(jit.lock().stats.arena_retires.load(Ordering::Relaxed), 1);
        assert_eq!(jit.lock().stats.arena_retired_pages.load(Ordering::Relaxed), 1);
        assert!(!old.is_runnable(0) && !old.has_code(), "the retired arena's code is off its page");
        assert_eq!(old.code_epoch(), 0);
        assert!(kept.is_runnable(0) && kept.has_code());
        jit.lock().compile_queue.stop();
    }

    /// Minimal `Device` (not `BusDevice`) stand-in for `CompileQueue::set_cpu` —
    /// tests that need `run_leader_flush`'s real flush path (not its
    /// no-cpu-wired recovery path) to actually run just need something a
    /// `Weak<dyn Device>` can upgrade to and call `stop()`/`start()` on; no
    /// test here cares whether cycles are actually stepped.
    struct NoopDevice;
    impl Device for NoopDevice {
        fn step(&self, _cycles: u64) {}
        fn stop(&self) {}
        fn start(&self) {}
        fn is_running(&self) -> bool { true }
    }

    /// Every word decodes as `ADDIU r1, r0, 1` — a real, compilable
    /// instruction, unlike `FakeDevice` (which always errors so
    /// `handle_request` bails before ever reaching codegen). Needed for a
    /// genuine end-to-end batching test: `handle_request_deferred` must
    /// actually produce a `FuncId` for there to be anything to batch.
    /// Two real `ADDIU r1, r0, 1` words (offsets 0 and 1) at the start of
    /// every physical page, followed by the JIT region-boundary sentinel at
    /// every other word — a genuine short, terminated region, matching how
    /// real code actually looks (a handful of instructions, then a branch/
    /// boundary — never "every word is a walkable instruction forever").
    /// Two instructions, not one: `comp::min_instrs_to_compile()` defaults
    /// to 2 outside `developer` builds (the build these tests actually run
    /// under), and a region below that floor is sticky-denylisted rather
    /// than compiled — confirmed live: an earlier one-instruction version of
    /// this device made every request get silently denylisted, 0/N ever
    /// published. Returning ADDIU unconditionally for every address (no
    /// sentinel at all) was tried before that and confirmed a different
    /// wrong shape: with `MAX_INSTRS_PER_COMPILE == usize::MAX` (a
    /// deliberate production setting, not a bug — see that constant's own
    /// doc comment in `comp.rs`), an all-ADDIU device makes every walk
    /// consume the entire 1024-word page, compiling a ~1000-instruction
    /// function instead of the couple of instructions each test actually
    /// intends — still correct, but ~300x slower per request than
    /// intended, confirmed live via direct instrumentation during the
    /// compile-pool work.
    struct AddiuDevice(AtomicU64);
    impl BusDevice for AddiuDevice {
        fn read8(&self, _addr: u64) -> BusRead8 { BusRead8::err() }
        fn write8(&self, _addr: u64, _val: u8) -> u32 { crate::traits::BUS_ERR }
        fn read16(&self, _addr: u64) -> BusRead16 { BusRead16::err() }
        fn write16(&self, _addr: u64, _val: u16) -> u32 { crate::traits::BUS_ERR }
        fn read32(&self, addr: u64) -> BusRead32 {
            let offset_word = (addr % PAGE_SIZE as u64) / 4;
            if offset_word < 2 {
                BusRead32::ok((crate::cpu::mips_isa::OP_ADDIU << 26) | (1 << 16) | 1)
            } else {
                BusRead32::ok(crate::cpu::mips_isa::JIT_REGION_BOUNDARY_SENTINEL)
            }
        }
        fn write32(&self, _addr: u64, _val: u32) -> u32 { crate::traits::BUS_ERR }
        fn read64(&self, _addr: u64) -> BusRead64 { BusRead64::err() }
        fn write64(&self, _addr: u64, _val: u64) -> u32 { crate::traits::BUS_ERR }
        fn gen_ptr(&self, _addr: u64) -> *const AtomicU64 { &self.0 as *const AtomicU64 }
        fn code_page(&self, addr: u64) -> Option<crate::traits::CodePage> { Some(crate::traits::CodeKey::flat_page(addr, self.gen_ptr(addr))) }
    }

    #[test]
    fn batching_eventually_publishes_via_queue_drain_fallback() {
        // End-to-end: a real worker thread with batching on, one compile
        // request, no second request to ever trigger a page-crossing flush.
        // Without the queue-drain fallback this entry would sit in `pending`
        // forever (nothing else would ever flush it) — this is the specific
        // regression the fallback trigger exists to prevent, exercised
        // through the real threaded path rather than just the synchronous
        // handle_request_deferred/flush_pending_batch unit tests.
        //
        // The polling loop below runs inside catch_unwind specifically so
        // q.stop() always executes afterward, even if the deadline assert!
        // fires: without that, a timeout panic would unwind straight out of
        // this function with the worker thread still running, holding a raw
        // pointer (CompileRequest::page) into `page` below — which is about
        // to be dropped as this function unwinds. That's a genuine
        // use-after-free from the still-live orphaned thread, not a benign
        // leaked-thread annoyance — confirmed live as the actual cause of an
        // intermittent SIGSEGV in unrelated, later-running tests during a
        // full-workspace parallel run (this test's deadline is only ever at
        // real risk of firing under heavy system load, which isolated/
        // single-test runs never reproduce — hence why this wasn't caught
        // immediately).
        let _stable = crate::cpu::jitv2::opcode_support::table_stable();
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, std::sync::Arc::new(JitStats::default()));

        // A real (non-null) gen counter: page.publish(, false) calls current_gen(),
        // which is always safe now (reads the shared NEVER_COMPILABLE_GEN
        // fallback for a null gen), but this test wants a real, independent
        // counter behind it since it's exercising a real compile+publish,
        // not just checking current_gen() doesn't crash.
        let gen_counter = AtomicU64::new(0);
        let mut page = PhysicalCodePage::new(0, &gen_counter as *const AtomicU64);
        // offset 0, matching every other AddiuDevice-based test in this
        // file: AddiuDevice's own read32 only decodes a real ADDIU for word
        // offsets < 2, sentinel otherwise (see its own doc comment) — offset
        // 4 (word index 4) would land straight on the sentinel with nothing
        // to compile at all. §13.2: mark_requested BEFORE send — a request
        // carries no offset, the compile thread reads `requested` fresh at
        // dequeue time.
        page.mark_requested(0);
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));

        // 30s, not the original 5s: this test passes in ~0.1s in isolation,
        // but the compile-worker thread genuinely needs real CPU time to get
        // scheduled — 5s proved too tight under a full-workspace parallel
        // test run (confirmed live: this deadline firing, not any actual
        // correctness bug, was tripping under heavy contention from every
        // other test's threads competing for the same cores).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !page.is_runnable(0) {
                assert!(std::time::Instant::now() < deadline, "entry never published — queue-drain fallback did not fire");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn batching_page_cross_trigger_publishes_without_waiting_for_queue_drain() {
        // Send enough compile requests, each targeting entry offset 0 of its
        // own distinct physical page, to force at least one page-crossing
        // flush (in the Cranelift arena's host-page-segment sense — not to
        // be confused with the distinct MIPS *physical* pages these requests
        // target, a coincidental naming overlap) before the queue ever
        // drains. Deliberately one request per PhysicalCodePage, all at
        // offset 0: a single walk_bounded from offset 0 with
        // MAX_INSTRS_PER_COMPILE=usize::MAX would otherwise treat many
        // requests against *different offsets of the same page* as one
        // giant sequential region (every word here decodes as the same
        // branch-free ADDIU), which isn't 300 independent compiles at
        // all — this was tried first and confirmed the wrong shape for this
        // test; distinct pages avoids that entirely and matches how real
        // page-cross-worthy traffic (many distinct hot pages arriving close
        // together) actually looks. Pinned explicitly rather than relying on
        // the ambient default (which no longer happens to be usize::MAX in
        // production) so this reasoning stays true regardless of future
        // default changes.
        let _max_instrs_guard = MaxInstrsGuard::set(usize::MAX);
        let _stable = crate::cpu::jitv2::opcode_support::table_stable();
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, std::sync::Arc::new(JitStats::default()));

        const N: usize = 300;
        let gen_counters: Vec<AtomicU64> = (0..N).map(|_| AtomicU64::new(0)).collect();
        let mut pages: Vec<PhysicalCodePage> = gen_counters.iter()
            .enumerate()
            .map(|(i, counter)| PhysicalCodePage::new(i as Pfn, counter as *const AtomicU64))
            .collect();
        let stats = JitStats::default();
        for page in pages.iter_mut() {
            // §13.2: same as the queue-drain-fallback test above.
            page.mark_requested(0);
            assert!(q.send(CompileRequest { page: page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));
        }

        // Poll for the *first* page specifically, well before the full
        // queue could possibly have drained (300 requests, 200µs backoff
        // only between empty-queue polls — draining that many keeps the
        // worker continuously busy, not backed off): if only the page-cross
        // trigger (not queue-drain) is what eventually publishes it, this
        // still succeeds this early. `stop()` itself does NOT guarantee a
        // full drain (it just flips `running` and joins whatever the loop's
        // current iteration is, same as any other "stop now" contract) — so
        // this must observe the page-cross trigger firing before stopping,
        // not rely on stop() to finish the queue first.
        //
        // catch_unwind for the same reason as the queue-drain-fallback test
        // above: without it, a deadline timeout here would unwind out of
        // this function with the worker thread still running against
        // `pages`/`gen_counters`, about to be dropped — a genuine
        // use-after-free from the orphaned thread, not a benign leaked
        // thread. This was the confirmed root cause of an intermittent
        // SIGSEGV surfacing in unrelated, later-running tests under
        // full-workspace parallel load.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            while !pages[0].is_runnable(0) {
                assert!(std::time::Instant::now() < deadline, "entry never published within the timeout");
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn multi_worker_pool_compiles_and_publishes_every_request() {
        // Real end-to-end proof that N>1 worker threads actually compile
        // concurrently and correctly: set_thread_count(4), send far more
        // distinct-page requests than any single worker could plausibly
        // have handled alone within the deadline if the "pool" were secretly
        // still just one thread, and assert every single one gets published
        // — not just "doesn't deadlock" or "doesn't crash," but genuinely
        // produces correct output for all of them, exercising the shared
        // arena (SharedArena::allocate serialized across 4 concurrent
        // Codegens), the shared queue (crossbeam_queue::ArrayQueue popped
        // concurrently by 4 threads), and — if the arena ever fills —
        // real leader-election/barrier contention among genuinely live
        // worker threads, not the single-worker degenerate case every other
        // test in this file exercises.
        let _stable = crate::cpu::jitv2::opcode_support::table_stable();
        let dev: Arc<dyn BusDevice> = Arc::new(AddiuDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.set_thread_count(4);
        assert_eq!(q.thread_count(), 4);
        // Wire real cpu/jitv2 stand-ins before starting: 2000 requests
        // across 4 workers sharing one arena is expected to cross
        // CODEGEN_ARENA_FLUSH_THRESHOLD_BYTES and exercise a real
        // leader-election flush — run_leader_flush needs both Weaks to
        // upgrade to do anything (see its own doc comment for what happens
        // if they can't).
        let cpu_stub: Arc<dyn Device> = Arc::new(NoopDevice);
        q.set_cpu(Arc::downgrade(&cpu_stub));
        let jit_owner = Arc::new(Mutex::new(Jitv2::new(JITV2_INITIAL_PAGE_CAPACITY)));
        q.set_owner(Arc::downgrade(&jit_owner));
        q.start(dev, std::sync::Arc::new(JitStats::default()));

        const N: usize = 2000;
        let gen_counters: Vec<AtomicU64> = (0..N).map(|_| AtomicU64::new(0)).collect();
        let mut pages: Vec<PhysicalCodePage> = gen_counters.iter()
            .enumerate()
            .map(|(i, counter)| PhysicalCodePage::new(i as Pfn, counter as *const AtomicU64))
            .collect();
        let stats = JitStats::default();
        for page in pages.iter_mut() {
            // §13.2: a request carries no offset anymore — the compile
            // thread snapshots `requested` fresh at dequeue time, so the
            // offset must be marked BEFORE sending, or handle_request_deferred
            // finds zero candidates and silently no-ops.
            page.mark_requested(0);
            assert!(q.send(CompileRequest { page: page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));
        }

        // catch_unwind for the same reason as the other threaded tests in
        // this file: a deadline timeout must not unwind out of this
        // function with worker threads still live against `pages` — see
        // the page-cross test's own comment for the confirmed
        // use-after-free this guards.
        // Well under a second alone; with a test on every core the four
        // workers can wait a long time for one.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            loop {
                let published = pages.iter().filter(|p| p.is_runnable(0)).count();
                if published == N {
                    break;
                }
                assert!(std::time::Instant::now() < deadline, "only {published}/{N} entries published within the timeout — pool may be stuck");
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }));
        q.stop();
        result.unwrap();
    }

    #[test]
    fn compile_queue_send_before_start_still_delivered_after_start() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));
        q.start(dev, std::sync::Arc::new(JitStats::default()));
        q.stop();
    }

    #[test]
    fn compile_queue_send_drops_when_full() {
        // Declared before the queue so it outlives it: dropping the queue
        // drains the requests, and each clears its page's scheduled flag.
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        let q = CompileQueue::new();
        // Don't start the worker: nothing drains, so capacity fills exactly.
        let stats = JitStats::default();
        let mut accepted = 0;
        for i in 0..(COMPILE_QUEUE_CAPACITY + 10) {
            if q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats) {
                accepted += 1;
            }
        }
        assert_eq!(accepted, COMPILE_QUEUE_CAPACITY, "queue must drop pushes past capacity, not block or panic");
    }

    #[test]
    fn drain_pending_queue_clears_scheduled_on_every_discarded_request() {
        // A request sitting in the queue when a flush drains it never
        // reaches handle_request_deferred, the only other place that clears
        // page_scheduled (via its ClearScheduledOnDrop guard) — drain_pending
        // must do it itself, or (absent mega_flush's own independent reset
        // covering the same page) that page would be permanently blocked
        // from ever being scheduled again.
        let mut q = CompileQueue::new();
        // Don't start the worker: nothing pops on its own, so the request is
        // still sitting there for drain_pending_queue to discard directly.
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        assert!(page.try_schedule_page());
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));
        assert!(page.is_scheduled(), "sanity: the send above left the page marked in-flight");

        q.drain_pending_queue();
        assert!(!page.is_scheduled(),
            "drain_pending must clear_scheduled() every request it discards, or the page is stuck \
             thinking a compile is still in flight for it forever");
    }

    #[test]
    fn a_hot_request_stays_with_its_own_pool() {
        // A request points into its own JIT's pages: another machine's
        // pool, which may outlive that JIT, must never see it.
        let mut page = PhysicalCodePage::new(0, std::ptr::null());
        assert!(page.try_schedule_page());
        let mut ours = CompileQueue::new();
        let mut theirs = CompileQueue::new();
        push_hot_request(&ours.hot_handle(), CompileRequest { page: &mut page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 });
        theirs.drain_pending_queue();
        assert!(page.is_scheduled(), "another pool's flush discarded this pool's hot request");
        assert_eq!(theirs.hot_handle().len(), 0);
        ours.drain_pending_queue();
        assert!(!page.is_scheduled());
    }

    #[test]
    fn compile_queue_start_is_idempotent() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev.clone(), std::sync::Arc::new(JitStats::default()));
        q.start(dev, std::sync::Arc::new(JitStats::default())); // must not panic or spawn a second thread
        q.stop();
    }

    #[test]
    fn compile_queue_stop_without_start_is_a_noop() {
        let mut q = CompileQueue::new();
        q.stop(); // must not panic
    }

    /// The pool's whole idle-cost fix rests on this: a parked worker must
    /// wake when the producer pushes a request. `park`/`unpark` tokens are
    /// sticky, so the wakeup cannot be lost even when it arrives before the
    /// worker actually parks — which is the race the producer and the
    /// worker are in all the time.
    #[test]
    fn a_parked_worker_is_woken_by_the_producer() {
        let wake = Arc::new(CompileWake::new());
        let (tx, rx) = std::sync::mpsc::channel();
        let worker_wake = wake.clone();
        let worker = std::thread::spawn(move || {
            worker_wake.register(std::thread::current());
            tx.send(()).expect("the test's channel is open");
            // Parks with no timeout: only the producer's unpark can end this.
            std::thread::park();
            tx.send(()).expect("the test's channel is open");
        });
        // The worker has registered itself: from here on, notify_all reaches it.
        rx.recv_timeout(std::time::Duration::from_secs(30)).expect("the worker did not start");
        wake.notify_all();
        rx.recv_timeout(std::time::Duration::from_secs(30))
            .expect("the parked worker was not woken by the producer");
        worker.join().expect("the worker thread panicked");
    }

    /// The producer skips the wake while no worker is asleep, so the
    /// sleeper count has to be visible before a worker parks: a worker in
    /// `park_idle` must be woken by an accepted push, long before its
    /// backstop.
    #[test]
    fn an_accepted_push_wakes_a_counted_sleeper() {
        let wake = Arc::new(CompileWake::new());
        let queue = Arc::new(CompileLane::new(1, wake.clone()));
        let hot = Arc::new(CompileLane::new(1, wake.clone()));
        let (w, q, h) = (wake.clone(), queue.clone(), hot.clone());
        let worker = std::thread::spawn(move || {
            w.register(std::thread::current());
            let t = std::time::Instant::now();
            // The backstop is far longer than the test allows: only the
            // producer's unpark can end this park in time.
            while q.is_empty() {
                w.park_idle(&q, &h, std::time::Duration::from_secs(60));
            }
            t.elapsed()
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while wake.sleepers.load(Ordering::SeqCst) == 0 && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(wake.sleepers.load(Ordering::SeqCst), 1, "the worker never parked");
        let stats = JitStats::default();
        let req = CompileRequest { page: std::ptr::null_mut(), compiled_for_fr1: false, cpu: 0, tenancy: 0 };
        assert!(push_compile_request(&queue, req, &stats));
        let waited = worker.join().expect("the worker thread panicked");
        assert!(waited < std::time::Duration::from_secs(30), "the sleeper waited out its backstop: {waited:?}");
        assert_eq!(wake.sleepers.load(Ordering::SeqCst), 0);
        // A dropped push (the lane holds one request already) must not wake
        // anyone, and must not fail for want of a sleeper.
        let req = CompileRequest { page: std::ptr::null_mut(), compiled_for_fr1: false, cpu: 0, tenancy: 0 };
        assert!(!push_compile_request(&queue, req, &stats));
    }

    /// A request pushed to a running pool must reach a worker that is
    /// parked on its idle backstop — the end-to-end shape of the wake-up:
    /// push, and the worker takes it.
    #[test]
    fn a_pushed_request_reaches_a_parked_worker() {
        let dev: Arc<dyn BusDevice> = Arc::new(FakeDevice(AtomicU64::new(0)));
        let mut q = CompileQueue::new();
        q.start(dev, std::sync::Arc::new(JitStats::default()));
        // Boxed: the request carries a raw pointer to this page, and the
        // worker dereferences it. The box outlives the pool, which is
        // stopped before the test returns.
        let mut page = Box::new(PhysicalCodePage::new(0, std::ptr::null()));
        assert!(page.try_schedule_page());
        let stats = JitStats::default();
        assert!(q.send(CompileRequest { page: &mut *page as *mut PhysicalCodePage, compiled_for_fr1: true, cpu: 0, tenancy: 0 }, &stats));
        // The worker pops the request and drops it (the page has no code to
        // compile), which clears the in-flight flag. Waiting for that proves
        // the request was taken, not left in the queue.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while page.is_scheduled() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!page.is_scheduled(), "the worker never took the request off the queue");
        q.stop();
    }
}
