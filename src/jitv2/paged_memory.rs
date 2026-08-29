//! Custom `cranelift_jit::JITMemoryProvider` for lazy, page-batched
//! finalization (see `rules/jitv2/jit-v2-design.md`'s memory-packing notes).
//!
//! `cranelift_jit::ArenaMemoryProvider` (the stock implementation) already
//! does everything we need *except* tell the caller when a new host-page
//! segment was just started, and it seals (RW->RX) synchronously inside its
//! own `finalize()` call. Neither of those fits a design where several
//! `cranelift_jit::JITModule`s (eventually, one per compile-pool worker)
//! share a single arena reservation: `finalize()` there is `&mut self` on
//! `JITMemoryProvider`, so it can't be shared directly (see `SharedArena`'s
//! own doc comment) — and even with sharing solved, one worker's
//! `finalize()` call must never touch bytes another worker might still be
//! writing.
//!
//! The design here: `allocate()` is a real, immediate bump-allocation (it
//! has to be — the caller needs a genuinely writable pointer back right
//! away to copy compiled code into). `finalize()` is *inert* as far as the
//! underlying memory is concerned — it never calls `mprotect` itself.
//! Instead it pushes the just-relocated byte range onto a shared, ordered
//! **seal queue** and immediately tries to advance a **sealed watermark**
//! through the longest contiguous prefix of that queue, `mprotect`-ing
//! whatever became newly sealable in one shot. This is safe for *any*
//! caller to do to *any* entry in the queue, regardless of who pushed it,
//! because by the time an entry is pushed, `cranelift_jit`'s own
//! `finalize_definitions()` has already finished patching relocations into
//! it (see `SharedArena::finalize`'s doc comment) — nothing will ever write
//! into that range again, so sealing it is just a page-protection flip with
//! no ordering dependency on which thread performs it.
//!
//! With a single, single-threaded owner (today's only caller, and this
//! module's `new_with_size` compatibility constructor), the seal queue
//! never has more than one entry in it at a time — every `finalize()` call
//! immediately seals its own just-pushed range, so this degenerates to
//! exactly the old segment-per-batch behavior with no observable
//! difference. The queue/watermark machinery only does real work once
//! there's more than one concurrent writer sharing one `SharedArena`.

use std::collections::VecDeque;
use std::io;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use cranelift_jit::{BranchProtection, JITMemoryKind, JITMemoryProvider};
use cranelift_module::ModuleResult;
use parking_lot::Mutex;

/// Shared state a `PagedArenaMemoryProvider` publishes and `Codegen` polls —
/// split out from the provider itself because `cranelift_jit::JITModule`
/// takes ownership of the provider as an opaque `Box<dyn JITMemoryProvider +
/// Send>` and never gives any of it back (no downcast, no accessor). Atomics
/// so `j2 status`/`Codegen::provider_crossed_page`/`packing_stats` can read
/// them without contending `SharedArena`'s own lock — the lock is still what
/// provides real write-side exclusion; these are a lock-free read mirror of
/// state that's always written while that lock is held.
#[derive(Default)]
pub struct PagedArenaState {
    /// Set true whenever an `allocate()` call started a brand-new segment
    /// (as opposed to packing into, or growing, the existing unfinalized
    /// one). `Codegen::provider_crossed_page` reads-and-clears this after
    /// every compile.
    crossed_page: AtomicBool,
    /// Sum of every segment's bump-allocator cursor (`Segment::position`) —
    /// the real code+relocation bytes written, across every segment ever
    /// allocated (finalized or not). See `PagedArenaMemoryProvider::packing_stats`.
    used_bytes: AtomicU64,
    /// Sum of every segment's reserved length (`Segment::len`), always a
    /// multiple of the host page size.
    reserved_bytes: AtomicU64,
    /// The arena's real (hugepage-aligned, on Linux) base address and total
    /// reserved length — written once, by `new_with_size`, before the
    /// provider is ever handed to `JITModule`; read-only from then on (no
    /// `Ordering` subtlety needed beyond Relaxed — this is set-once-then-frozen,
    /// not a value that changes concurrently with reads). `j2 hugepages`
    /// (`developer` only) uses this to scope its `/proc/self/smaps`
    /// `AnonHugePages` query to exactly this arena's address range.
    arena_ptr: AtomicU64,
    arena_len: AtomicU64,
}

impl PagedArenaState {
    pub fn crossed_page(&self) -> bool {
        self.crossed_page.swap(false, Ordering::Relaxed)
    }

    /// `(bytes_actually_used, bytes_reserved)` — see `PagedArenaMemoryProvider::packing_stats`.
    pub fn packing_stats(&self) -> (u64, u64) {
        (self.used_bytes.load(Ordering::Relaxed), self.reserved_bytes.load(Ordering::Relaxed))
    }

    /// `(base_address, len)` of the arena's real (possibly hugepage-aligned)
    /// reservation — `(0, 0)` before `new_with_size` has run. Dev diagnostic
    /// only (`j2 hugepages`); nothing on the hot compile path reads this.
    #[cfg(feature = "developer")]
    pub fn arena_range(&self) -> (u64, u64) {
        (self.arena_ptr.load(Ordering::Relaxed), self.arena_len.load(Ordering::Relaxed))
    }
}

fn align_up(addr: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (addr + align - 1) & !(align - 1)
}

/// Transparent-hugepage size this arena aligns/collapses against — 2MiB, the
/// standard THP size on Linux (the only OS any of this hugepage machinery
/// runs on at all — see this constant's `#[cfg]`: Windows has no madvise or
/// THP concept, large pages there require `MEM_LARGE_PAGES` +
/// `SeLockMemoryPrivilege` at allocation time, a different mechanism
/// entirely; macOS has no public THP/superpage API on Apple Silicon, and the
/// old x86 `VM_FLAGS_SUPERPAGE_SIZE_2MB` is dead). Not queried at runtime
/// (the real value lives in
/// `/sys/kernel/mm/transparent_hugepage/hpage_pmd_size`, a file read neither
/// `region` nor `libc` wrap) — 2MiB is correct for both mainstream Linux
/// hugepage-capable architectures (x86_64, aarch64) this project ships on;
/// every operation below is advisory (`madvise` failures are logged, never
/// fatal) and alignment only wastes a little reserved-but-unused address
/// space if that ever stops being true, never incorrect behavior.
#[cfg(target_os = "linux")]
const HUGE_PAGE_SIZE: usize = 2 * 1024 * 1024;

/// Best-effort `madvise(MADV_HUGEPAGE)` over the whole reservation, called
/// once right after `region::alloc` — asks the kernel to prefer backing this
/// VMA with transparent hugepages as it gets populated (khugepaged can also
/// promote it later without this, but async promotion can take a scan cycle
/// or more to kick in; asking upfront means a freshly-compiled hot function
/// doesn't have to wait for that). Purely advisory: failure is logged once
/// and otherwise ignored — this is a throughput optimization, never a
/// correctness requirement, and the arena works identically (just without
/// the TLB-pressure benefit) if the kernel declines or the platform lacks
/// `MADV_HUGEPAGE` entirely (see the `#[cfg]` gate on this whole family of
/// functions).
#[cfg(target_os = "linux")]
fn madvise_hugepage(ptr: *mut u8, len: usize) {
    let rc = unsafe { libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_HUGEPAGE) };
    if rc != 0 {
        eprintln!("jitv2: madvise(MADV_HUGEPAGE) failed on paged arena reservation: {}", std::io::Error::last_os_error());
    }
}
#[cfg(not(target_os = "linux"))]
fn madvise_hugepage(_ptr: *mut u8, _len: usize) {}

/// Best-effort `madvise(MADV_COLLAPSE)` over `[ptr, ptr+len)`, which must
/// already be hugepage-aligned and hugepage-sized (caller's responsibility —
/// see `SharedArena::advance_hugepage_collapse_watermark`, the only caller).
/// Unlike `MADV_HUGEPAGE` (a standing preference the kernel acts on lazily,
/// via khugepaged's background scanner), `MADV_COLLAPSE` (Linux 6.1+)
/// synchronously compacts the region into a hugepage right now — worth
/// paying for once a region is fully written and sealed (RX) rather than
/// waiting for khugepaged's own schedule, which is tuned for general
/// workloads, not "this code just got hot, please collapse it immediately."
/// Best-effort: a kernel without `MADV_COLLAPSE` support (pre-6.1) returns
/// `ENOSYS`/`EINVAL`, silently ignored — no version probe needed, the
/// syscall's own failure path is the probe.
#[cfg(target_os = "linux")]
fn madvise_collapse(ptr: *mut u8, len: usize) {
    let _ = unsafe { libc::madvise(ptr as *mut libc::c_void, len, libc::MADV_COLLAPSE) };
}
#[cfg(not(target_os = "linux"))]
fn madvise_collapse(_ptr: *mut u8, _len: usize) {}

/// Dev diagnostic (`j2 hugepages`): sum the `AnonHugePages:` field of every
/// `/proc/self/smaps` VMA entry that falls (even partially) within
/// `[range_start, range_start+range_len)`. Lets a live run confirm THP is
/// actually landing on the JIT arena, rather than assuming `MADV_HUGEPAGE`/
/// `MADV_COLLAPSE` succeeded from their own (best-effort, silently-ignored)
/// return codes alone — a host with `transparent_hugepage=never` in sysfs,
/// or a container/cgroup that disables THP, makes every `madvise` call here
/// a silent no-op, and this is the only way short of external tooling
/// (`grep AnonHugePages /proc/<pid>/smaps`) to notice that's happening
/// before it shows up as an unexplained iTLB-miss regression.
///
/// smaps format per VMA: a header line (`<start>-<end> <perms> ...`)
/// followed by indented `Key:    N kB` lines until the next header. Only
/// `AnonHugePages` is summed; every other field is skipped. Returns `None`
/// if `/proc/self/smaps` can't be opened/read at all (containerized/
/// sandboxed environments sometimes restrict it) — the caller reports that
/// as "unavailable," not as "0 hugepages," since those mean different things.
#[cfg(all(target_os = "linux", feature = "developer"))]
pub fn anon_hugepages_in_range(range_start: u64, range_len: u64) -> Option<u64> {
    let range_end = range_start + range_len;
    let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let mut total_kb: u64 = 0;
    let mut in_range = false;
    for line in smaps.lines() {
        if let Some((addrs, _rest)) = line.split_once(' ') {
            if let Some((start_hex, end_hex)) = addrs.split_once('-') {
                if let (Ok(start), Ok(end)) = (u64::from_str_radix(start_hex, 16), u64::from_str_radix(end_hex, 16)) {
                    // A real smaps header line always parses both halves as
                    // hex; anything that doesn't (an indented "Key: N kB"
                    // line) falls through to the `else` below instead.
                    in_range = start < range_end && end > range_start;
                    continue;
                }
            }
        }
        if in_range {
            if let Some(rest) = line.strip_prefix("AnonHugePages:") {
                if let Some(kb) = rest.trim().strip_suffix(" kB").and_then(|n| n.trim().parse::<u64>().ok()) {
                    total_kb += kb;
                }
            }
        }
    }
    Some(total_kb * 1024)
}
#[cfg(not(all(target_os = "linux", feature = "developer")))]
pub fn anon_hugepages_in_range(_range_start: u64, _range_len: u64) -> Option<u64> { None }

/// Port of `cranelift_jit::memory::set_readable_and_executable` — `pub(crate)`
/// in that crate, so not reachable from here; duplicated verbatim rather than
/// reimplemented from scratch, to stay exactly in sync with the icache/BTI
/// handling the stock `ArenaMemoryProvider` relies on. Clears the icache for
/// the newly-written code before flipping the mapping to RX (some CPUs have
/// errata around doing this after, per the original's own comment), then
/// applies ARM BTI protection if requested and supported. Does *not* flush
/// the instruction pipeline (`wasmtime_jit_icache_coherence::pipeline_flush_mt`)
/// — same as the original, that's the caller's job once a whole seal-sweep
/// batch has been sealed (`SharedArena::try_seal_ready`).
fn set_readable_and_executable(ptr: *mut u8, len: usize, branch_protection: BranchProtection) {
    unsafe {
        wasmtime_jit_icache_coherence::clear_cache(ptr as *const std::ffi::c_void, len)
            .expect("Failed cache clear")
    };

    unsafe {
        region::protect(ptr, len, region::Protection::READ_EXECUTE)
            .expect("unable to make jitv2 paged arena segment readable+executable");
    }

    if branch_protection == BranchProtection::BTI {
        #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
        if std::arch::is_aarch64_feature_detected!("bti") {
            let prot = libc::PROT_EXEC | libc::PROT_READ | /* PROT_BTI */ 0x10;
            unsafe {
                assert!(
                    libc::mprotect(ptr as *mut libc::c_void, len, prot) >= 0,
                    "unable to make jitv2 paged arena segment readable+executable with BTI: {}",
                    std::io::Error::last_os_error()
                );
            }
        }
    }
}

/// Everything a sealed range's `page.publish()` call needs — carried
/// end-to-end from the compiling worker (who has all of this in hand right
/// after its own `compile_region_uncommitted`/`finalize_definitions()` call)
/// through the seal queue to whichever worker's sweep eventually seals it.
/// See this module's doc comment for why *any* worker's sweep can publish
/// *any* entry here, not just its own: by construction, everything needed
/// is already sitting on the entry itself, with no `Codegen`/`FuncId`/module
/// access required to resolve it further. `jit_fn` is resolved via
/// `get_finalized_function` right after `finalize_definitions()` succeeds —
/// safe to *read* at that point even though the range isn't callable until
/// sealed (relocations are already patched; only the page protection is
/// still pending — see `SharedArena::finalize`'s own doc comment).
#[derive(Clone, Copy, Debug)]
pub struct PublishInfo {
    pub page: *mut crate::jitv2::PhysicalCodePage,
    /// Single entry-point offset — the default (`not(feature = "j2wp")`)
    /// path's one-function-per-entry-point model, consumed by
    /// `PhysicalCodePage::publish(offset_word, ...)`. Unused (left `0`) by
    /// the `j2wp` path, which uses `new_entries` instead.
    #[cfg(not(feature = "j2wp"))]
    pub offset: usize,
    /// §13.4: every entry offset this compile covers, not a single `offset`
    /// — a `j2wp` compile can publish coverage for several entry points at
    /// once (§13.2/§13.3's coalescing). Consumed by
    /// `PhysicalCodePage::publish(new_entries, ...)`. Unused (left zeroed)
    /// by the default path, which uses `offset` instead.
    #[cfg(feature = "j2wp")]
    pub new_entries: [u64; crate::jitv2::BITMAP_WORDS],
    pub gen_snap: u64,
    pub instr_count: usize,
    pub code_size: u32,
    /// Dev diagnostic only (`JitEntry::compiled_for_fr1`, `j2 pcp`): the
    /// `STATUS_FR` value this region was compiled against. Carried through the
    /// seal queue so the eventual `page.publish` can stamp it on the entry.
    pub compiled_for_fr1: bool,
    /// `None` only for a `finalize()` call reached without going through
    /// `Codegen` (a test driving the `JITMemoryProvider` trait directly,
    /// with no publish info to give — see this call site's own comment).
    /// Every real caller always sets this.
    pub jit_fn: Option<crate::jitv2::JitFn>,
}

unsafe impl Send for PublishInfo {}

impl PublishInfo {
    /// A blank placeholder — `page` null, no coverage, `jit_fn: None` — for
    /// `push_placeholder`'s bare reservation (patched later by
    /// `patch_pending_publish`) and tests. Whichever of `offset`/
    /// `new_entries` this build carries is left at its zero value.
    pub(crate) fn blank() -> Self {
        Self {
            page: std::ptr::null_mut(),
            #[cfg(not(feature = "j2wp"))]
            offset: 0,
            #[cfg(feature = "j2wp")]
            new_entries: [0u64; crate::jitv2::BITMAP_WORDS],
            gen_snap: 0,
            instr_count: 0,
            code_size: 0,
            compiled_for_fr1: false,
            jit_fn: None,
        }
    }
}

/// One `finalize()` call's worth of already-relocated, write-complete bytes
/// waiting for their page(s) to be sealed (RW->RX) — see this module's doc
/// comment for why it's safe for *any* thread to seal any entry here,
/// regardless of who pushed it. `start`/`end` are byte offsets from the
/// arena base; entries are pushed in increasing `start` order (a caller's
/// own allocations only ever grow the bump cursor, so one caller's batch is
/// always contiguous, and batches from different callers interleave in
/// whatever order their `finalize()` calls happen to land in — never out of
/// order relative to the arena's own `position`, since `start` is fixed at
/// `allocate()` time).
#[derive(Clone, Copy, Debug)]
struct SealEntry {
    start: usize,
    end: usize,
    /// Which OS thread's `push_placeholder` call created this entry —
    /// stamped at allocate time (before we know whether finalize will ever
    /// actually run for it), unlike `publish.page`/`publish.jit_fn`, which
    /// stay null/`None` until `patch_pending_publish` fills them in. This is
    /// what lets `j2 seal-queue` name the stuck thread directly instead of
    /// just reporting "some placeholder, somewhere" — every worker thread
    /// looks identical from the outside otherwise.
    thread_id: std::thread::ThreadId,
    /// The physical page this compile was for, stamped at the same
    /// allocate-time point as `thread_id` — independent of `publish.page`
    /// (which stays null on a bare placeholder). Always non-null: every real
    /// `push_placeholder` caller (`Codegen::compile_region_uncommitted`)
    /// has a real page by construction.
    page: *mut crate::jitv2::PhysicalCodePage,
    publish: PublishInfo,
}

unsafe impl Send for SealEntry {}

/// Diagnostic snapshot of `SharedArena`'s seal queue front — see
/// `SharedArena::seal_queue_snapshot`'s own doc comment for what this is
/// for and how to read it (`j2 seal-queue`).
#[derive(Clone, Copy, Debug)]
pub struct SealQueueSnapshot {
    pub queue_len: usize,
    pub position: usize,
    pub sealed_up_to: usize,
    pub front_start: Option<usize>,
    pub front_end: Option<usize>,
    pub front_is_unpatched_placeholder: Option<bool>,
    /// Which OS thread's `push_placeholder` call created the front entry —
    /// `ThreadId`'s own `Debug` output (`ThreadId::as_u64()` is still
    /// nightly-only as of this writing), since `ThreadId` isn't `Display`.
    /// Names the stuck thread directly instead of "some placeholder,
    /// somewhere."
    pub front_thread_id: Option<std::thread::ThreadId>,
    /// The physical page the front entry's compile was for, stamped at the
    /// same allocate-time point as `front_thread_id` (independent of
    /// `PublishInfo::page`, which stays null until patched). Raw pointer —
    /// the caller (`j2 seal-queue`) is responsible for treating it as an
    /// opaque identity/debug value only, never dereferencing it without the
    /// same care any other `*mut PhysicalCodePage` needs (the pool could
    /// have reset this exact slot since the entry was pushed).
    pub front_page: Option<*mut crate::jitv2::PhysicalCodePage>,
    /// Index (0-based, from the front) of the entry where the contiguity
    /// scan actually stops — `None` only when the queue is empty. This is
    /// almost always the entry worth investigating, NOT the front: a queue
    /// can have thousands of already-patched, already-contiguous entries
    /// ahead of the real gap (confirmed live — a patched front entry with a
    /// `queue_len` in the thousands and nothing draining). `0` means the
    /// front entry itself is the gap, matching what `front_*` already
    /// reports.
    pub first_gap_index: Option<usize>,
    pub first_gap_start: Option<usize>,
    pub first_gap_end: Option<usize>,
    /// Why the scan stopped here: `true` if this entry is still unpatched
    /// (the real, actionable case — see this struct's own doc comment for
    /// what that means); `false` if it's patched but its `start` doesn't
    /// connect to the running end (a hole in `start` ordering — should be
    /// structurally impossible per `push_placeholder`'s own insertion
    /// contract, so seeing `false` here at all would itself be a new,
    /// different bug worth chasing).
    pub first_gap_is_unpatched_placeholder: Option<bool>,
    pub first_gap_thread_id: Option<std::thread::ThreadId>,
    pub first_gap_page: Option<*mut crate::jitv2::PhysicalCodePage>,
}

unsafe impl Send for SealQueueSnapshot {}

/// The one real allocator + seal queue behind however many
/// `PagedArenaMemoryProvider` handles share it. See this module's doc
/// comment for the overall design. Reached only through
/// `Arc<parking_lot::Mutex<SharedArena>>` — every method here assumes `&mut
/// self`, i.e. the caller already holds that lock.
pub struct SharedArena {
    alloc: ManuallyDrop<Option<region::Allocation>>,
    ptr: *mut u8,
    size: usize,
    /// Bump cursor — the arena's committed-and-handed-out extent. Monotonic;
    /// never rewound. Every `allocate()` call, and only `allocate()`, moves
    /// this forward.
    position: usize,
    /// Page-rounded watermark: `[ptr, ptr+sealed_up_to)` is RX right now.
    /// Only `try_seal_ready` moves this forward.
    sealed_up_to: usize,
    /// Entries pushed by `finalize()`, popped by `try_seal_ready` once
    /// they're covered by a page-rounded, contiguous-from-`sealed_up_to`
    /// prefix. Ordered by `start`.
    seal_queue: VecDeque<SealEntry>,
    /// How much of `[ptr, ptr+size)` has already been `MADV_COLLAPSE`d, as a
    /// byte offset from `ptr` — always a multiple of `HUGE_PAGE_SIZE` (or 0
    /// on a platform without hugepage support, where it never advances).
    /// Monotonic: `try_seal_ready` is the only writer, advancing it by whole
    /// hugepage-sized steps as `sealed_up_to` clears each boundary.
    collapsed_up_to: usize,
    /// The real `BranchProtection` cranelift computed and passed into the
    /// most recent real `finalize()` call (`JITModule::finalize_definitions()`
    /// derives this from the live ISA — BTI on aarch64 when the target
    /// supports it, `None` otherwise; see that function's own source).
    /// Cached here so a sweep that seals memory *without* going through a
    /// real `finalize()` call (`try_seal_ready_forced`, the idle-timeout
    /// path — see its own doc comment) can still apply the correct
    /// protection instead of guessing/hardcoding `None`. `None` (the
    /// `Option`, not `BranchProtection::None`) until the first real
    /// `finalize()` call has ever run; a sweep that fires before that has
    /// nothing to seal yet anyway (nothing gets pushed to `seal_queue`
    /// except through `finalize()`), so this is never actually read while
    /// still unset.
    last_branch_protection: Option<BranchProtection>,
    /// Ranges `try_seal_ready` most recently reported as newly sealed —
    /// written by `push_and_seal`/`try_seal_ready` on every call (even an
    /// empty result overwrites this with an empty `Vec`, so a caller reading
    /// it right after its own `finalize()` call never sees a stale result
    /// from some earlier, unrelated call). Lives here rather than on any one
    /// `PagedArenaMemoryProvider` handle for the same reason
    /// `force_seal_on_finalize` does — see that field's own doc comment.
    last_sealed: Vec<PublishInfo>,
    /// Shared with whoever constructed this arena (`Codegen`, or the
    /// compile-pool builder) — see `PagedArenaState`'s own doc comment.
    state: Arc<PagedArenaState>,
}

unsafe impl Send for SharedArena {}

impl SharedArena {
    fn new_with_size(reserve_size: usize, state: Arc<PagedArenaState>) -> Result<Self, region::Error> {
        let size = align_up(reserve_size, region::page::size());
        // Over-reserve by one extra hugepage and use the hugepage-aligned
        // address within that larger mapping as the arena's real base — the
        // padding before/after is virtual address space only (still
        // PROT_NONE / never touched), never physically backed, so this
        // costs nothing but address space. `region::alloc` gives no
        // alignment control of its own (a plain `mmap(NULL, size, ...)`
        // under the hood — kernel picks a page-aligned but not necessarily
        // hugepage-aligned base), so this is the standard way to get a
        // hugepage-aligned region through it. `self.alloc` keeps the
        // *original*, untrimmed `Allocation` for correct freeing (its
        // `Drop` frees by its own stored base/size, unaffected by what
        // `self.ptr` below points at) — only `self.ptr`/`self.size` (what
        // gets bump-allocated from) are the aligned sub-region.
        // On a platform with no hugepage support (HUGE_PAGE_SIZE undefined),
        // this degenerates to exactly the old unaligned behavior.
        #[cfg(target_os = "linux")]
        let (mut alloc, ptr, size) = {
            let over_reserved = size + HUGE_PAGE_SIZE;
            let mut alloc = region::alloc(over_reserved, region::Protection::NONE)?;
            let base = alloc.as_mut_ptr::<u8>() as usize;
            let aligned = align_up(base, HUGE_PAGE_SIZE);
            (alloc, aligned as *mut u8, size)
        };
        #[cfg(not(target_os = "linux"))]
        let (mut alloc, ptr, size) = {
            let alloc = region::alloc(size, region::Protection::NONE)?;
            let ptr = alloc.as_ptr::<u8>() as *mut u8;
            (alloc, ptr, size)
        };
        let _ = &mut alloc; // silence unused_mut when the aligned-address branch doesn't need it
        madvise_hugepage(ptr, size);
        #[cfg(feature = "developer")]
        {
            state.arena_ptr.store(ptr as u64, Ordering::Relaxed);
            state.arena_len.store(size as u64, Ordering::Relaxed);
        }
        Ok(Self {
            alloc: ManuallyDrop::new(Some(alloc)),
            ptr,
            size,
            position: 0,
            sealed_up_to: 0,
            seal_queue: VecDeque::new(),
            collapsed_up_to: 0,
            last_branch_protection: None,
            last_sealed: Vec::new(),
            state,
        })
    }

    /// Diagnostic snapshot of the seal queue's front — `j2 seal-queue`'s only
    /// caller. Answers "is there a permanent gap, and where" without needing
    /// per-worker-thread introspection — every `SealEntry` now carries the
    /// `thread_id`/`page` that created it, stamped at `push_placeholder`
    /// time (allocate time), independent of `publish`'s own fields (which
    /// stay null/`None` until `patch_pending_publish` runs). If the front
    /// entry's `start` hasn't advanced across two calls spaced any real time
    /// apart while `queue_len` stays nonzero, that placeholder is the
    /// permanent gap: the named thread's compile for the named page reserved
    /// this range and its matching `finalize_batch_nonforced` call never ran.
    ///
    /// The front entry alone isn't always where the queue is actually
    /// stuck: `try_seal_ready`'s own contiguity scan (this same function's
    /// logic, mirrored here) can advance past several already-patched,
    /// already-contiguous entries before hitting the real gap further back
    /// in the queue — a `queue_len` in the thousands with a *patched* front
    /// entry is exactly that shape. `first_gap_index` names where the scan
    /// actually stops, and `first_gap_*` describes that entry specifically —
    /// which is almost always the one worth investigating, not the front.
    pub fn seal_queue_snapshot(&self) -> SealQueueSnapshot {
        let front = self.seal_queue.front();
        // Mirrors try_seal_ready's own two-phase scan exactly: entries
        // already covered by `sealed_up_to` would already have been popped
        // by a real try_seal_ready call, so front.start > sealed_up_to is
        // the common case here (this is read-only — it never pops anything
        // itself). Then walk forward while each entry is patched
        // (jit_fn.is_some()) AND contiguous with the running end (or
        // straddles it, for the very first entry) — the first entry that
        // fails either check is the real gap.
        let mut expected_start = self.sealed_up_to;
        let mut gap_index = None;
        for (i, entry) in self.seal_queue.iter().enumerate() {
            let contiguous = if i == 0 { entry.start <= expected_start } else { entry.start == expected_start };
            if entry.publish.jit_fn.is_none() || !contiguous {
                gap_index = Some(i);
                break;
            }
            expected_start = entry.end.max(expected_start);
        }
        let gap = gap_index.and_then(|i| self.seal_queue.get(i));
        SealQueueSnapshot {
            queue_len: self.seal_queue.len(),
            position: self.position,
            sealed_up_to: self.sealed_up_to,
            front_start: front.map(|e| e.start),
            front_end: front.map(|e| e.end),
            front_is_unpatched_placeholder: front.map(|e| e.publish.jit_fn.is_none()),
            front_thread_id: front.map(|e| e.thread_id),
            front_page: front.map(|e| e.page),
            first_gap_index: gap_index,
            first_gap_start: gap.map(|e| e.start),
            first_gap_end: gap.map(|e| e.end),
            first_gap_is_unpatched_placeholder: gap.map(|e| e.publish.jit_fn.is_none()),
            first_gap_thread_id: gap.map(|e| e.thread_id),
            first_gap_page: gap.map(|e| e.page),
        }
    }

    /// Every entry in the seal queue, front to back, as
    /// `(start, end, is_unpatched_placeholder, thread_id, page)` — `j2
    /// seal-queue`'s only caller. Not bounded/paginated: the caller is
    /// expected to cap how much it prints.
    pub fn seal_queue_entries(&self) -> Vec<(usize, usize, bool, std::thread::ThreadId, *mut crate::jitv2::PhysicalCodePage)> {
        self.seal_queue.iter()
            .map(|e| (e.start, e.end, e.publish.jit_fn.is_none(), e.thread_id, e.page))
            .collect()
    }

    fn record_packing(&self) {
        // `used_bytes`: real bytes actually handed out (the bump cursor
        // itself — never page-rounded). `reserved_bytes`: how much of the
        // reservation is currently committed RW, i.e. `position` rounded up
        // to the containing whole page(s) — the direct analogue of the old
        // "sum of segment lengths" (`Segment::len` was always page-rounded
        // too).
        self.state.used_bytes.store(self.position as u64, Ordering::Relaxed);
        let page = region::page::size();
        self.state.reserved_bytes.store(align_up(self.position, page) as u64, Ordering::Relaxed);
    }

    /// Real bump-allocation: extend `position` by `size` (page-rounded, RW
    /// from the start — there's no separate "grow the segment" step
    /// anymore, since there are no segments; every allocation just claims
    /// the next `size` bytes and the whole arena is RW up to `position`
    /// until `try_seal_ready` catches up). Returns the raw pointer the
    /// caller (`CompiledBlob::new`, via `define_function`) writes compiled
    /// code into, outside this lock.
    ///
    /// If `position` sits inside a range `try_seal_ready` already sealed to
    /// RX (possible after an idle-timeout/queue-drain sweep force-seals a
    /// still-partial page — see `try_seal_ready`'s `force` parameter — and a
    /// later allocation then arrives wanting to pack into what's now
    /// read-only memory), skip forward to `sealed_up_to` first: that memory
    /// is permanently off-limits to future allocation once sealed, so
    /// packing into it is never safe again, forced-seal or not.
    /// Returns `(unaligned_start, end, ptr)` — `unaligned_start` is
    /// `position` *before* this call's `align_up`, i.e. the byte offset
    /// where this call's own alignment padding begins, and `end` is the
    /// TRUE end of this allocation's real range, `align_up(unaligned_start,
    /// align) + size`. Callers that track "this compile's range" for later
    /// sealing (`PagedArenaMemoryProvider::allocate`) must use
    /// `unaligned_start` for the start and this `end` directly — NOT
    /// recompute the end themselves as `unaligned_start + size`. Confirmed
    /// live as a real bug when the caller used to do exactly that: whenever
    /// `align_up` actually padded (`unaligned_start` wasn't already
    /// aligned), `unaligned_start + size` under-reported the consumed range
    /// by exactly the padding amount — a permanent, unclaimed gap between
    /// consecutive `SealEntry`s that `try_seal_ready`'s contiguity scan can
    /// never close short of a full arena flush (thousands of genuinely
    /// fully-patched entries with nothing draining — a tiny gap between two
    /// adjacent entries' real `end`/`start` was the actual cause, not a
    /// stuck compile thread).
    fn allocate(&mut self, size: usize, align: u64, _kind: JITMemoryKind) -> io::Result<(usize, usize, *mut u8)> {
        let align = usize::try_from(align).expect("alignment too big");
        assert!(align <= region::page::size(), "alignment over page size is not supported");

        if self.position < self.sealed_up_to {
            self.position = self.sealed_up_to;
        }

        let unaligned_start = self.position;
        let start = align_up(self.position, align);
        let end = start + size;
        let page = region::page::size();
        let committed_end = align_up(self.position.max(end), page);
        // committed_end only differs from self.position's own page-rounding
        // when this allocation extends past whatever was already committed
        // RW — i.e. the "new page(s) needed" case. Track whether that
        // happened for the crossed_page signal (today's semantics: true iff
        // this call reached into fresh, previously-uncommitted address
        // space — still meaningful pool-wide even though there's no
        // per-caller segment identity anymore, see `PagedArenaState`'s own
        // note on this becoming a pool-wide signal once shared).
        let prev_committed = align_up(self.position, page);
        let crosses = committed_end > prev_committed;

        if end > self.size {
            return Err(io::Error::new(io::ErrorKind::Other, "pre-allocated jit memory region exhausted"));
        }
        if committed_end > prev_committed {
            unsafe {
                region::protect(self.ptr.add(prev_committed), committed_end - prev_committed, region::Protection::READ_WRITE)
                    .expect("unable to change memory protection for jitv2 paged arena segment");
            }
        }
        self.position = end;
        self.state.crossed_page.store(crosses, Ordering::Relaxed);
        self.record_packing();
        Ok((unaligned_start, end, unsafe { self.ptr.add(start) }))
    }

    /// Scan `seal_queue` from the front for the longest prefix that is
    /// contiguous starting at `sealed_up_to`, `mprotect` whatever whole
    /// pages that prefix — and only that prefix — fully covers, advance
    /// `sealed_up_to`, and pop+return every entry whose range is now fully
    /// inside the sealed watermark, as `(start, end)` (in order), for the
    /// caller to match back against its own pending publish-info list. A gap
    /// (the next entry's `start` doesn't match the running end) stops the
    /// scan — that range hasn't been pushed yet by whoever owns it.
    ///
    /// The front entry is allowed to *straddle* `sealed_up_to` (`start <=
    /// sealed_up_to < end`) rather than start exactly at it: a previous call
    /// can have sealed only part of an entry's range (page-rounding stopped
    /// short of its `end` — see below), in which case that entry stays in
    /// the queue, unpopped and unreported, for a later call to finish
    /// sealing once the rest of its page is dealt with.
    ///
    /// `force` controls how far into the *still-open* page (the one
    /// `self.position` currently sits in) this call is allowed to seal:
    /// - `force: false` (the queue-drain/idle-timeout sweep path — see
    ///   `PagedArenaMemoryProvider`'s pool-mode callers) only ever seals a
    ///   page the bump cursor has fully moved past (`page_round_down(ready_end)`,
    ///   capped by `page_round_down(self.position)`) — the currently-open
    ///   page is left alone, since another allocation may still legitimately
    ///   pack into its remainder.
    /// - `force: true` (every `finalize()` call — see its own doc comment)
    ///   seals up through `page_round_up(ready_end)` regardless of where
    ///   `self.position` sits, on the theory that the caller invoking
    ///   `finalize()` is explicitly declaring "I need my pointer usable
    ///   right now" — matching `finalize_definitions()`'s own contract that
    ///   its caller gets back an immediately-callable function. A later
    ///   `allocate()` call whose `position` would otherwise have landed in
    ///   the now-sealed remainder of that page is bumped forward past it
    ///   instead (`allocate`'s own `position < sealed_up_to` check) — a
    ///   forced seal costs a little packing density (the sealed page's
    ///   unused tail is wasted), never correctness.
    ///
    /// Uses `self.last_branch_protection` (the real value cranelift computed
    /// on the most recent actual `finalize()` call — see that field's own
    /// doc comment) rather than taking one as a parameter, so a sweep that
    /// doesn't go through `finalize()` at all (`try_seal_ready_forced`, the
    /// idle-timeout path) still applies the correct protection instead of
    /// guessing. Falls back to `BranchProtection::None` if nothing has ever
    /// been finalized yet (nothing could be queued to seal in that case
    /// anyway).
    /// Also writes `self.last_sealed` with the same result before returning
    /// (on every path, including the two early "nothing to do" returns) —
    /// see that field's own doc comment for why: a caller reaching this
    /// through a *different* `PagedArenaMemoryProvider` handle than the one
    /// whose `finalize()` call triggered it (`Codegen::seal_handle`) has no
    /// other way to learn the result, since `JITMemoryProvider::finalize`'s
    /// own trait signature returns nothing useful.
    fn try_seal_ready(&mut self, force: bool) -> Vec<PublishInfo> {
        let branch_protection = self.last_branch_protection.unwrap_or(BranchProtection::None);
        // A straggler can arrive after the watermark has already passed it:
        // worker A's own finalize() call runs late (relative to wall clock,
        // not to the arena's own ordering — its range's `start` can still
        // be earlier than sealed_up_to) while some other, faster worker's
        // contiguous chain already sealed straight past A's range without
        // ever seeing A's entry (it wasn't in the queue yet). When A's entry
        // finally gets pushed here, its bytes are already RX — nothing left
        // to seal — but it must still be drained from the front of
        // `seal_queue` and still reported as sealed (its caller still needs
        // to learn this so it can publish), or it permanently masquerades as
        // "the next gap to fill," blocking every real, still-open range
        // behind it forever (confirmed live: a stuck compile-pool test with
        // a `seal_queue` gap sitting exactly at a page boundary, caused by
        // exactly this). Must run before the "did anything change" early
        // returns below, and must be included in every return path (not
        // just the "something new sealed" one), since a stale-only push
        // (nothing NEW to seal) would otherwise hit one of those early
        // returns and report nothing for a range that genuinely is sealed.
        // An entry pushed at allocate() time (jit_fn: None — see
        // `SharedArena::allocate`'s own doc comment) is not yet safe to
        // seal or report: its bytes/relocations may not even be fully
        // written yet (that only happens once the compiling worker's own
        // finalize_definitions() call returns and patches jit_fn in via
        // `patch_pending_jit_fn`). Treat it exactly like a gap — never pop
        // it, never let the contiguity scan walk past it — everything below
        // this point in the queue stays blocked until it's patched, same as
        // a genuinely-not-yet-pushed range would block them.
        let mut sealed = Vec::new();
        while let Some(front) = self.seal_queue.front() {
            if front.publish.jit_fn.is_none() || front.end > self.sealed_up_to {
                break;
            }
            let entry = self.seal_queue.pop_front().unwrap();
            sealed.push(entry.publish);
        }
        let mut ready_end = self.sealed_up_to;
        for (i, entry) in self.seal_queue.iter().enumerate() {
            if entry.publish.jit_fn.is_none() {
                break;
            }
            let starts_here = if i == 0 { entry.start <= ready_end } else { entry.start == ready_end };
            if !starts_here {
                break;
            }
            ready_end = entry.end.max(ready_end);
        }
        if ready_end == self.sealed_up_to {
            self.last_sealed.clone_from(&sealed);
            return sealed;
        }

        let page = region::page::size();
        let page_rounded_end = if force {
            align_up(ready_end, page)
        } else {
            // Only a page the bump cursor has fully moved past is safe to
            // seal without forcing — see this function's own doc comment.
            let vacated_end = (self.position / page) * page;
            (ready_end.min(vacated_end) / page) * page
        };
        if page_rounded_end <= self.sealed_up_to {
            self.last_sealed.clone_from(&sealed);
            return sealed; // ready, but not yet a whole sealable page beyond what's already sealed
        }

        let seal_ptr = unsafe { self.ptr.add(self.sealed_up_to) };
        let seal_len = page_rounded_end - self.sealed_up_to;
        set_readable_and_executable(seal_ptr, seal_len, branch_protection);
        self.sealed_up_to = page_rounded_end;

        while let Some(front) = self.seal_queue.front() {
            if front.publish.jit_fn.is_none() || front.end > self.sealed_up_to {
                break;
            }
            let entry = self.seal_queue.pop_front().unwrap();
            sealed.push(entry.publish);
        }

        self.advance_hugepage_collapse_watermark();
        wasmtime_jit_icache_coherence::pipeline_flush_mt().expect("Failed pipeline flush");
        self.last_sealed.clone_from(&sealed);
        sealed
    }

    /// `MADV_COLLAPSE` every hugepage-sized region that has now become
    /// fully sealed since the last call, advancing `collapsed_up_to`. Never
    /// touches memory past `sealed_up_to` — collapsing memory that might
    /// still be RW (allocated but not yet sealed) would be both wasted work
    /// and pointless (the whole point is compacting *finished*, stable
    /// code).
    #[cfg(target_os = "linux")]
    fn advance_hugepage_collapse_watermark(&mut self) {
        let target = (self.sealed_up_to / HUGE_PAGE_SIZE) * HUGE_PAGE_SIZE; // round DOWN — only whole, fully-sealed hugepages
        if target > self.collapsed_up_to {
            let len = target - self.collapsed_up_to;
            let region_ptr = unsafe { self.ptr.add(self.collapsed_up_to) };
            madvise_collapse(region_ptr, len);
            self.collapsed_up_to = target;
        }
    }
    #[cfg(not(target_os = "linux"))]
    fn advance_hugepage_collapse_watermark(&mut self) {}

    unsafe fn free_memory(&mut self) {
        if self.ptr == ptr::null_mut() {
            return;
        }
        self.seal_queue.clear();
        let _: Option<region::Allocation> = self.alloc.take();
        self.ptr = ptr::null_mut();
    }
}

impl Drop for SharedArena {
    fn drop(&mut self) {
        if self.ptr == ptr::null_mut() {
            return;
        }
        let is_live = self.sealed_up_to > 0;
        if !is_live {
            unsafe { self.free_memory() };
        }
    }
}

/// Thin per-`JITModule` handle over a `SharedArena` — see this module's doc
/// comment. `cranelift_jit::JITModule` owns its memory provider as a plain
/// `Box<dyn JITMemoryProvider + Send>` (verified against `cranelift-jit`
/// 0.134.3's `backend.rs`), so the literal same provider instance can't be
/// handed to more than one `JITModule`; this handle is what lets several
/// `JITModule`s (eventually, one per compile-pool worker) all forward into
/// one real `SharedArena`.
///
/// `JITMemoryProvider::finalize`'s own signature carries no `FuncId`/range,
/// and — more fundamentally — no way to give it a resolved `JitFn` (only
/// valid *after* `finalize_definitions()` as a whole returns, i.e. strictly
/// after this trait method has already run). So this handle does not use
/// `finalize()` to push seal-queue entries at all: `Codegen` pushes a
/// placeholder itself at *allocate* time (`push_placeholder`, called right
/// after `define_function` succeeds, using the exact range `last_allocation`
/// reports) and patches in the real `JitFn` once `finalize_definitions()`
/// returns (`patch_pending_jit_fn`) — see both methods' own doc comments.
/// `finalize()` here is therefore a true no-op as far as the seal queue is
/// concerned; it exists only because `JITMemoryProvider` requires it.
pub struct PagedArenaMemoryProvider {
    inner: Arc<Mutex<SharedArena>>,
    /// Optional side-channel to a *sibling* handle over the same arena
    /// (`Codegen::seal_handle`, sharing this `Arc` — see
    /// `Codegen::new_module`'s construction of both) that wants to observe
    /// each individual `allocate()` call's own exact `[start, end)`, one
    /// call at a time — unlike `pending` above (which accumulates across a
    /// whole batch until the next `finalize()`), this is overwritten fresh
    /// on every single `allocate()` call. Exists because `Codegen` needs
    /// the *per-`FuncId`* byte range (to build `func_ranges`) immediately
    /// after each `compile_region_uncommitted` call, before any batching or
    /// finalize has happened — reading it through `seal_handle` (which
    /// never allocates anything itself) is race-free because only this
    /// handle (the one actually inside `JITModule`) ever writes to it, and
    /// `Codegen` only ever reads it synchronously, immediately after the
    /// one `define_function` call that produced it — never confused with a
    /// *different* worker's concurrent allocation, unlike reading the
    /// shared arena's own live `position` field would be (see this field's
    /// introduction: an earlier version bracketed `seal_handle.position()`
    /// before/after `define_function` instead, which raced with other
    /// workers' concurrent `allocate()` calls moving `position` in between
    /// — confirmed live as the cause of a real stuck-forever compile-pool
    /// bug under genuine multi-thread contention).
    last_allocation: Option<Arc<Mutex<Option<(usize, usize)>>>>,
}

unsafe impl Send for PagedArenaMemoryProvider {}

impl PagedArenaMemoryProvider {
    /// Build a fresh `SharedArena` and one handle over it. The compatibility
    /// entry point for every single-owner caller today (`Codegen::new()`'s
    /// standalone/inline-mode path, and this module's own tests) — with
    /// exactly one handle over the arena, every `finalize()` call's pushed
    /// range is always exactly at the current watermark (nothing else can
    /// ever be queued ahead of it), so `try_seal_ready` always seals it
    /// immediately: this degenerates to the old one-segment-per-batch
    /// behavior with no observable difference.
    pub fn new_with_size(reserve_size: usize, state: Arc<PagedArenaState>) -> Result<Self, region::Error> {
        let shared = Self::new_shared(reserve_size, state)?;
        Ok(Self::from_shared(shared))
    }

    /// Build a fresh `SharedArena` and return the `Arc` directly, for a
    /// caller that intends to build several `from_shared` handles over it
    /// (the eventual compile-pool case).
    pub fn new_shared(reserve_size: usize, state: Arc<PagedArenaState>) -> Result<Arc<Mutex<SharedArena>>, region::Error> {
        Ok(Arc::new(Mutex::new(SharedArena::new_with_size(reserve_size, state)?)))
    }

    /// Build one handle over an already-existing shared arena.
    pub fn from_shared(inner: Arc<Mutex<SharedArena>>) -> Self {
        Self { inner, last_allocation: None }
    }

    /// Build one handle wired to report every individual `allocate()` call's
    /// own exact range through `mailbox`, for a sibling handle (constructed
    /// separately, over the same `inner`) to read via `take_last_allocation`.
    /// See `last_allocation`'s own field doc comment for why this exists.
    pub fn from_shared_with_mailbox(inner: Arc<Mutex<SharedArena>>, mailbox: Arc<Mutex<Option<(usize, usize)>>>) -> Self {
        Self { inner, last_allocation: Some(mailbox) }
    }

    /// Read-and-clear whatever the most recent `allocate()` call (through
    /// a *sibling* handle sharing this one's mailbox — see
    /// `from_shared_with_mailbox`) reported as its own exact range. `None`
    /// if this handle has no mailbox wired (the common case — only
    /// `Codegen::seal_handle` ever calls this) or nothing has been
    /// allocated since the last read.
    pub fn take_last_allocation(&self) -> Option<(usize, usize)> {
        self.last_allocation.as_ref().and_then(|m| m.lock().take())
    }

    /// Clone out the `Arc<Mutex<SharedArena>>` this handle points at, so a
    /// caller can build sibling handles over the exact same arena (e.g.
    /// `Codegen::new_with_shared_arena` when rebuilding a `Codegen` pool on
    /// top of an already-reserved arena, without a fresh mmap reservation —
    /// see that constructor's own doc comment).
    pub fn shared(&self) -> Arc<Mutex<SharedArena>> {
        self.inner.clone()
    }

    /// Try to advance the sealed watermark without pushing anything new —
    /// for the seal-quiesce/idle-timeout sweep: re-attempts sealing whatever
    /// is already queued (any worker's own still-pending, patched entries).
    /// Non-forced — safe to call speculatively at any time. Returns the
    /// newly-sealed entries directly (unlike the trait's `finalize()`, there
    /// is no `last_sealed`/`take_last_sealed` indirection needed here since
    /// this isn't reached through that fixed-signature trait method).
    pub fn try_seal_ready(&mut self) -> Vec<PublishInfo> {
        self.inner.lock().try_seal_ready(false)
    }

    /// Forced counterpart to `try_seal_ready` — only ever called from under
    /// the seal-quiesce barrier (`CompileQueue::run_seal_leader`, once every
    /// worker is provably parked/not mid-compile — see `SealBarrierState`'s
    /// own doc comment for why forcing needs that guarantee): seals
    /// everything currently queued and *finalized* (a `jit_fn: None`
    /// placeholder still blocks the scan regardless of `force` — see
    /// `try_seal_ready`'s own doc comment) up through a full page boundary,
    /// regardless of whether the bump cursor has moved past it.
    pub fn try_seal_ready_forced(&mut self) -> Vec<PublishInfo> {
        self.inner.lock().try_seal_ready(true)
    }

    /// Reserve this `[start, end)` range's slot in the seal queue at
    /// *allocate* time, with `jit_fn: None` — before this compile's bytes
    /// are even fully written, let alone relocated. Every real caller
    /// (`Codegen::compile_region_uncommitted`, right after `define_function`
    /// succeeds) follows this with a later `patch_pending_publish` call once
    /// `finalize_definitions()` has actually run — see that method's own
    /// doc comment for why a placeholder is pushed now rather than waiting:
    /// `JITModule` owns its provider handle opaquely (never gives it back),
    /// so there is no *other* reliable hook to push through once `Codegen`
    /// resolves the real `JitFn`, and pushing now (at a point `Codegen` DOES
    /// still control directly) is what avoids relying on the
    /// `JITMemoryProvider::finalize` trait callback's automatic push at all
    /// for this path. Does not attempt to seal — a `jit_fn: None` entry
    /// blocks `try_seal_ready`'s contiguity scan by construction (see that
    /// method's own doc comment), so there would be nothing to gain by
    /// trying yet regardless.
    ///
    /// Insertion-sorted, not a plain back-push: `allocate()` and this call
    /// are two separate lock acquisitions (`compile_region_uncommitted`
    /// calls this only after `define_function` has already returned), so a
    /// *different*, faster worker's own allocate-then-push can land a
    /// higher-`start` entry in between this worker's own `allocate()` and
    /// this call — one worker's own entries arrive back-to-back (its bump
    /// position only ever advances between its own calls), but the queue as
    /// a whole, across every worker sharing this arena, is not strictly
    /// append-only.
    pub fn push_placeholder(&mut self, start: usize, end: usize, page: *mut crate::jitv2::PhysicalCodePage) {
        let mut inner = self.inner.lock();
        let idx = inner.seal_queue.iter().rposition(|e| e.start <= start).map_or(0, |i| i + 1);
        let thread_id = std::thread::current().id();
        inner.seal_queue.insert(idx, SealEntry { start, end, thread_id, page, publish: PublishInfo::blank() });
    }

    /// Fill in the real `PublishInfo` (resolved `jit_fn` included) for the
    /// placeholder `push_placeholder` reserved at `[start, end)`, then
    /// immediately attempt to seal. Called once, right after
    /// `module.finalize_definitions()` returns (relocations patched,
    /// `get_finalized_function` now valid — see `Codegen::finalize_batch_nonforced`).
    /// The entry is guaranteed to still be exactly where it was pushed: `start`
    /// is unique to this one compile (the bump cursor never reuses a range),
    /// and a `jit_fn: None` entry can never be popped by any concurrent
    /// `try_seal_ready` call (blocks the scan by construction) — so there is
    /// no race to resolve here, only a lookup. `force` matches the caller's
    /// own contract (`false` for a normal per-compile finalize, `true` only
    /// ever reached from the seal-quiesce barrier's leader). Uses whatever
    /// `BranchProtection` the trait's own `finalize()` call already recorded
    /// during this same `finalize_definitions()` call — no need to pass it
    /// again.
    pub fn patch_pending_publish(&mut self, start: usize, end: usize, publish: PublishInfo, force: bool) -> Vec<PublishInfo> {
        debug_assert!(publish.jit_fn.is_some(), "patch_pending_publish's entry must carry a resolved JitFn");
        let mut inner = self.inner.lock();
        let entry = inner.seal_queue.iter_mut().find(|e| e.start == start && e.end == end)
            .expect("patch_pending_publish: no placeholder found for this range — push_placeholder must run first");
        debug_assert!(entry.publish.jit_fn.is_none(), "patch_pending_publish: this range was already patched once");
        entry.publish = publish;
        inner.try_seal_ready(force)
    }

    /// Ranges (byte offsets from the arena base) that became newly sealed on
    /// the shared arena's most recent seal attempt, whichever handle
    /// (`finalize()` call or explicit sweep) actually triggered it —
    /// `Codegen` reads this via `seal_handle` right after calling
    /// `module.finalize_definitions()` through the module's own (different,
    /// unreachable) handle, to know which (if any) of its own
    /// just-finalized `FuncId`s are actually safe to call
    /// `get_finalized_function` on and publish. Empty if that attempt's own
    /// range is still blocked behind an earlier gap, or if nothing has been
    /// finalized yet. See `SharedArena::last_sealed`'s own doc comment for
    /// why this lives on the shared arena rather than any one handle.
    pub fn take_last_sealed(&mut self) -> Vec<PublishInfo> {
        std::mem::take(&mut self.inner.lock().last_sealed)
    }

    /// See `SharedArena::seal_queue_snapshot`'s own doc comment.
    pub fn seal_queue_snapshot(&self) -> SealQueueSnapshot {
        self.inner.lock().seal_queue_snapshot()
    }

    /// See `SharedArena::seal_queue_entries`'s own doc comment.
    pub fn seal_queue_entries(&self) -> Vec<(usize, usize, bool, std::thread::ThreadId, *mut crate::jitv2::PhysicalCodePage)> {
        self.inner.lock().seal_queue_entries()
    }

    /// The arena's base address — `PublishInfo`/`Codegen` need this to turn
    /// a raw pointer (from `allocate`'s return, or `sealed` ranges) into a
    /// byte offset and back, since the seal queue and `try_seal_ready` speak
    /// in offsets, not pointers (offsets stay stable across the `Arc` clone
    /// boundary; a raw pointer captured before a hypothetical future arena
    /// swap would not).
    pub fn arena_base(&self) -> *mut u8 {
        self.inner.lock().ptr
    }


    pub(crate) unsafe fn free_memory(&mut self) {
        unsafe { self.inner.lock().free_memory() }
    }
}

impl JITMemoryProvider for PagedArenaMemoryProvider {
    fn allocate(&mut self, size: usize, align: u64, kind: JITMemoryKind) -> io::Result<*mut u8> {
        let mut inner = self.inner.lock();
        let (start, end, ptr) = inner.allocate(size, align, kind)?;
        drop(inner);
        // `start` is the pre-alignment offset (`SharedArena::allocate`'s
        // `unaligned_start`), not `ptr`'s own (post-alignment) offset — see
        // that method's own doc comment for why the range this handle
        // tracks/reports must include its own leading alignment pad: that
        // padding is never anyone else's to claim, so folding it into this
        // call's own range is what keeps seal_queue's ranges gap-free. `end`
        // is `SharedArena::allocate`'s own real end (`align_up(start, align)
        // + size`), NOT recomputed here as `start + size` — that used to be
        // a real, confirmed-live bug: whenever alignment actually padded,
        // `start + size` silently undershot the true end by the pad amount,
        // leaving a permanent gap in the seal queue no flush-free recovery
        // could ever close (see `SharedArena::allocate`'s own doc comment).
        // Report this exact call's own range to a sibling handle, if one is
        // listening — see `last_allocation`'s own field doc comment. The
        // real caller (`Codegen::compile_region_uncommitted`) reads this
        // right after `define_function` returns and calls `push_placeholder`
        // itself — the seal queue entry is reserved here, at allocate time,
        // not deferred to `finalize()` (see this struct's own doc comment).
        if let Some(mailbox) = &self.last_allocation {
            *mailbox.lock() = Some((start, end));
        }
        Ok(ptr)
    }

    unsafe fn free_memory(&mut self) {
        unsafe { self.free_memory() }
    }

    fn finalize(&mut self, branch_protection: BranchProtection) -> ModuleResult<()> {
        // True no-op as far as the seal queue is concerned — see this
        // struct's own doc comment for why: every real caller pushes at
        // allocate time (`push_placeholder`) and patches in the resolved
        // `JitFn` explicitly (`patch_pending_jit_fn`), bypassing this trait
        // method entirely. Still record the branch protection cranelift
        // computed for this call, though — `patch_pending_jit_fn`'s own
        // seal attempt needs a real value, not a stale/default one, and
        // this is the only place cranelift ever hands it to us.
        self.inner.lock().last_branch_protection = Some(branch_protection);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_arena(size: usize) -> (PagedArenaMemoryProvider, Arc<PagedArenaState>) {
        let state = Arc::new(PagedArenaState::default());
        let arena = PagedArenaMemoryProvider::new_with_size(size, state.clone()).unwrap();
        (arena, state)
    }

    #[test]
    fn first_allocation_starts_a_segment_and_reports_crossed_page() {
        let (mut arena, state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page(), "the very first allocation must commit a fresh page");
    }

    #[test]
    fn crossed_page_is_edge_triggered_and_clears_on_read() {
        let (mut arena, state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page());
        assert!(!state.crossed_page(), "a second read without an intervening allocation must report false");
    }

    #[test]
    fn packing_into_the_same_committed_range_does_not_cross_a_page() {
        let (mut arena, state) = new_arena(1 << 20);
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page());
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(!state.crossed_page(), "a second small allocation must pack into the already-committed page, not commit a new one");
    }

    #[test]
    fn allocation_bigger_than_a_page_commits_multiple_pages() {
        let (mut arena, state) = new_arena(16 << 20);
        let page = region::page::size();
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page());
        arena.allocate(page * 3, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page(), "an allocation extending past the already-committed page must report a crossing");
    }

    fn dummy_publish() -> PublishInfo {
        PublishInfo::blank()
    }

    fn resolved_publish() -> PublishInfo {
        PublishInfo { jit_fn: Some(unsafe { std::mem::transmute::<usize, crate::jitv2::JitFn>(1) }), ..dummy_publish() }
    }

    #[test]
    fn finalizing_seals_and_forces_the_next_allocation_to_a_fresh_page() {
        // The real Codegen protocol: push_placeholder at allocate time,
        // patch_pending_publish (force=true) once "finalize_definitions()"
        // would have returned — see both methods' own doc comments. Not
        // going through the (now-inert, see `finalize`'s own trait-impl doc
        // comment) `JITMemoryProvider::finalize` call at all.
        let (mut arena, state) = new_arena(1 << 20);
        let ptr = arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        state.crossed_page(); // clear
        let base = arena.arena_base() as usize;
        let start = ptr as usize - base;
        arena.push_placeholder(start, start + 64, std::ptr::null_mut());
        let sealed = arena.patch_pending_publish(start, start + 64, resolved_publish(), true);
        assert_eq!(sealed.len(), 1);
        assert!(sealed[0].jit_fn.is_some());

        // A forced patch seals the whole page this allocation lived on —
        // the next allocation must be bumped past the now-sealed page,
        // reporting a fresh crossing.
        arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(state.crossed_page(), "a sealed page must never accept another allocation");
    }

    #[test]
    fn non_forced_patch_lets_a_later_allocation_pack_into_the_same_page() {
        // The whole point of this redesign: a non-forced patch_pending_publish
        // call must NOT seal the still-open page its own range lives in, so
        // a later allocation — even from the same handle — can still pack
        // into the remainder of that page before anything actually mprotects.
        let (mut arena, state) = new_arena(1 << 20);

        let ptr1 = arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        let base = arena.arena_base() as usize;
        let start1 = ptr1 as usize - base;
        arena.push_placeholder(start1, start1 + 64, std::ptr::null_mut());
        let sealed = arena.patch_pending_publish(start1, start1 + 64, resolved_publish(), false);
        assert!(sealed.is_empty(), "a non-forced patch must not seal the still-open page");
        state.crossed_page(); // clear

        let ptr2 = arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        assert!(!state.crossed_page(), "a later allocation must still be able to pack into the page a non-forced patch left open");
        let start2 = ptr2 as usize - base;
        arena.push_placeholder(start2, start2 + 64, std::ptr::null_mut());
        let sealed = arena.patch_pending_publish(start2, start2 + 64, resolved_publish(), false);
        assert_eq!(sealed.len(), 0, "still not a whole vacated page's worth — position (128) is nowhere near the next page boundary");

        // Forcing (a later, real Codegen::finalize_batch-style call) finally
        // seals everything queued so far.
        let ptr3 = arena.allocate(64, 16, JITMemoryKind::Executable).unwrap();
        let start3 = ptr3 as usize - base;
        arena.push_placeholder(start3, start3 + 64, std::ptr::null_mut());
        let sealed = arena.patch_pending_publish(start3, start3 + 64, resolved_publish(), true);
        assert_eq!(sealed.len(), 3,
            "a forced patch must sweep up every still-queued range, not just its own most recent one");
    }

    #[test]
    fn packing_stats_track_used_vs_reserved_bytes() {
        let (mut arena, state) = new_arena(1 << 20);
        let (used0, reserved0) = state.packing_stats();
        assert_eq!(used0, 0);
        assert_eq!(reserved0, 0);

        arena.allocate(96, 16, JITMemoryKind::Executable).unwrap();
        let (used1, reserved1) = state.packing_stats();
        assert_eq!(used1, 96);
        assert_eq!(reserved1, region::page::size() as u64);

        arena.allocate(208, 16, JITMemoryKind::Executable).unwrap();
        let (used2, reserved2) = state.packing_stats();
        assert_eq!(used2, 304, "used bytes must sum across packed allocations");
        assert_eq!(reserved2, region::page::size() as u64, "reserved must stay one page while packing continues to fit");
    }

    #[test]
    fn over_capacity_returns_err_not_panic() {
        let (mut arena, _state) = new_arena(1 << 20);
        arena.allocate(900_000, 1, JITMemoryKind::Executable).unwrap();
        assert!(arena.allocate(200_000, 1, JITMemoryKind::Executable).is_err());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn arena_base_is_hugepage_aligned() {
        let (arena, _state) = new_arena(1 << 20);
        assert_eq!(arena.arena_base() as usize % HUGE_PAGE_SIZE, 0, "arena base must be hugepage-aligned");
    }

    /// Real Codegen protocol, driven by hand: allocate, push_placeholder at
    /// the returned range, then patch_pending_publish once "finalize" would
    /// have run — see both methods' own doc comments. `finalize()` itself
    /// (the `JITMemoryProvider` trait method) no longer does any of this.
    fn allocate_push_and_patch(arena: &mut PagedArenaMemoryProvider, size: usize, force: bool) -> (usize, usize, Vec<PublishInfo>) {
        let ptr = arena.allocate(size, 16, JITMemoryKind::Executable).unwrap();
        let base = arena.arena_base() as usize;
        let start = ptr as usize - base;
        let end = start + size;
        arena.push_placeholder(start, end, std::ptr::null_mut());
        let sealed = arena.patch_pending_publish(start, end, resolved_publish(), force);
        (start, end, sealed)
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn collapse_watermark_advances_only_past_fully_sealed_hugepages() {
        let (mut arena, _state) = new_arena(HUGE_PAGE_SIZE * 3);

        let chunk = region::page::size();
        let mut bytes_allocated = 0usize;
        while bytes_allocated < HUGE_PAGE_SIZE + chunk {
            allocate_push_and_patch(&mut arena, chunk, true);
            bytes_allocated += align_up(chunk, region::page::size());
        }

        assert_eq!(arena.inner.lock().collapsed_up_to, HUGE_PAGE_SIZE,
            "watermark must advance exactly one whole hugepage once that much has been sealed, not the partial second hugepage too");
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn collapse_watermark_does_not_advance_below_one_hugepage() {
        let (mut arena, _state) = new_arena(1 << 20);
        allocate_push_and_patch(&mut arena, 64, true);
        assert_eq!(arena.inner.lock().collapsed_up_to, 0, "a single small sealed range must not advance the watermark -- it's nowhere near a full hugepage");
    }

    #[test]
    fn out_of_order_finalize_blocks_until_contiguous() {
        // Two allocations from two different handles, B finalized before A:
        // try_seal_ready must not seal B's range until A's has also been
        // pushed, even though B's finalize() call happened first.
        let state = Arc::new(PagedArenaState::default());
        let shared = PagedArenaMemoryProvider::new_shared(1 << 20, state).unwrap();
        let mut a = PagedArenaMemoryProvider::from_shared(shared.clone());
        let mut b = PagedArenaMemoryProvider::from_shared(shared.clone());

        let (a_start, a_end) = {
            let ptr = a.allocate(64, 16, JITMemoryKind::Executable).unwrap();
            let base = a.arena_base() as usize;
            (ptr as usize - base, ptr as usize - base + 64)
        };
        let (b_start, b_end) = {
            let ptr = b.allocate(64, 16, JITMemoryKind::Executable).unwrap();
            let base = b.arena_base() as usize;
            (ptr as usize - base, ptr as usize - base + 64)
        };
        a.push_placeholder(a_start, a_end, std::ptr::null_mut());
        b.push_placeholder(b_start, b_end, std::ptr::null_mut());

        let sealed = b.patch_pending_publish(b_start, b_end, resolved_publish(), true);
        assert!(sealed.is_empty(), "B's range must not seal while A's (earlier) range is still outstanding");

        let sealed = a.patch_pending_publish(a_start, a_end, resolved_publish(), true);
        assert_eq!(sealed.len(), 2, "once A's range arrives, both A's and B's contiguous ranges must seal together");
    }

    #[test]
    fn a_handles_own_finalize_can_seal_another_handles_range() {
        // Symmetric case to the one above: once B's earlier gap is filled by
        // A's patch, B itself never needs to patch again to learn its own
        // range sealed — but if it does, it should see nothing new (already
        // sealed by A's call). This locks in "any handle's seal attempt can
        // surface any pending range as sealed," not just its own — the core
        // simplification this design relies on.
        let state = Arc::new(PagedArenaState::default());
        let shared = PagedArenaMemoryProvider::new_shared(1 << 20, state).unwrap();
        let mut a = PagedArenaMemoryProvider::from_shared(shared.clone());
        let mut b = PagedArenaMemoryProvider::from_shared(shared.clone());

        let (a_start, a_end) = {
            let ptr = a.allocate(64, 16, JITMemoryKind::Executable).unwrap();
            let base = a.arena_base() as usize;
            (ptr as usize - base, ptr as usize - base + 64)
        };
        let (b_start, b_end) = {
            let ptr = b.allocate(64, 16, JITMemoryKind::Executable).unwrap();
            let base = b.arena_base() as usize;
            (ptr as usize - base, ptr as usize - base + 64)
        };
        a.push_placeholder(a_start, a_end, std::ptr::null_mut());
        b.push_placeholder(b_start, b_end, std::ptr::null_mut());

        let sealed = b.patch_pending_publish(b_start, b_end, resolved_publish(), true);
        assert!(sealed.is_empty());

        let sealed = a.patch_pending_publish(a_start, a_end, resolved_publish(), true);
        assert_eq!(sealed.len(), 2, "A's patch call sealed both its own and B's earlier-queued range");
    }

    #[test]
    fn mailbox_reported_end_includes_alignment_padding_not_just_unaligned_start_plus_size() {
        // Real, confirmed-live bug: the mailbox used to report `end` as
        // `unaligned_start + size`, silently skipping whatever padding
        // align_up added — a permanent, uncounted gap between this
        // allocation and the next one's `start`, which `push_placeholder`
        // then baked into two SealEntrys that never actually touch,
        // permanently blocking `try_seal_ready`'s contiguity scan. The fix:
        // SharedArena::allocate itself computes and returns the true end
        // (`align_up(unaligned_start, align) + size`); the wrapper must use
        // that value directly, never recompute it from `unaligned_start`.
        let state = Arc::new(PagedArenaState::default());
        let shared = PagedArenaMemoryProvider::new_shared(1 << 20, state.clone()).unwrap();
        let mailbox = Arc::new(parking_lot::Mutex::new(None));
        let mut arena = PagedArenaMemoryProvider::from_shared_with_mailbox(shared, mailbox.clone());

        // First allocation: 3 bytes, unaligned, leaves position at an
        // odd (non-8-aligned) offset so the SECOND allocation's 8-byte
        // alignment request is guaranteed to need real padding.
        arena.allocate(3, 1, JITMemoryKind::Executable).unwrap();
        let (first_start, first_end) = mailbox.lock().take().unwrap();
        assert_eq!(first_end - first_start, 3);

        arena.allocate(64, 8, JITMemoryKind::Executable).unwrap();
        let (second_start, second_end) = mailbox.lock().take().unwrap();

        assert_eq!(second_start, first_end,
            "the second allocation's reported start must be exactly where the first one's reported range \
             ended — this is what push_placeholder relies on to keep the seal queue gap-free");
        // second_start (== first_end == 3) isn't itself 8-aligned, so this
        // allocation's real range starts at align_up(3, 8) = 8 and really
        // ends at 8 + 64 = 72 — a real 5-byte pad the reported range must
        // fold in. The confirmed-live bug reported end as
        // second_start + 64 = 67 instead, silently dropping that pad.
        let real_aligned_start = second_start.div_ceil(8) * 8;
        assert_eq!(second_end, real_aligned_start + 64,
            "the reported end must be the TRUE end (aligned start + size), not unaligned_start + size — \
             the confirmed-live bug undershot by exactly the alignment pad, leaving those bytes as a \
             permanent, unclaimed gap between this entry and the next one's start");
    }
}
