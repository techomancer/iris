//! Host-MMU-backed physical memory banks — a drop-in alternative to
//! [`crate::mem::Memory`].
//!
//! See `docs/ppmem-design.md`. In short: a bank's storage is a shared-memory
//! object rather than a `Vec<u32>`, which lets the same bank be mapped into
//! the guest's physical address space at several addresses at once. SIMM
//! mirroring (a bank smaller than its configured slot repeating within it) and
//! the low-512KB alias then become *mappings* instead of `addr & addr_mask` on
//! every single access.
//!
//! # Storage layout
//!
//! Byte-for-byte identical to `Memory`: u32 words in native host order, with
//! 64-bit accesses stored `rotate_left(32)`. This is not incidental — it is
//! what lets `PpMemory` substitute for `Memory` without touching snapshots,
//! `save_bin`/`load_bin`, or any of the sub-word access paths. Endianness
//! still lives only at The Edge (HACKING.md); nothing here byte-swaps.
//!
//! # What is and isn't here
//!
//! [`PpMemory`] is one bank: it implements `BusDevice` + `Resettable` and all
//! the inherent methods `physical.rs`/`machine.rs` call on `Memory`, so it
//! plugs into the existing bus with no changes to the callers.
//!
//! [`PpMemSpace`] is the 4GB window that banks get mapped into, and the
//! `MappedMemory` trait is the extra capability (clear/map/bitmap). The window
//! is what a future fast path would read through; the bus abstraction keeps
//! working either way.

use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use std::cell::UnsafeCell;

use super::map::{granularity, AddrSpace, Prot, SharedMem};
use crate::traits::{
    BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, Resettable, BUS_OK,
};

#[cfg(feature = "jitv2")]
const JITV2_PAGE_SIZE: usize = crate::jitv2::jitv2::PAGE_SIZE as usize;

/// Size of the emulated physical address space the window covers: 4GB.
pub const WINDOW_SIZE: u64 = 1 << 32;

/// Granularity of [`MappedMemory::mapped_bitmap`]: 4GB / 64 bits = 64MB/bit.
pub const BITMAP_SHIFT: u32 = 26;

/// Bytes of RAM per generation counter: one `AtomicU64` per 4KB page.
///
/// `4096 / 8 == 512`, so a region's gen block is always its data size >> 9 and
/// sits at its data offset >> 9 — which is what keeps `gen_ptr` a pure shift
/// off the physical address.
#[cfg(feature = "jitv2")]
pub const GEN_RATIO: u64 = (JITV2_PAGE_SIZE as u64) / 8;

/// Size of the generation window: one `AtomicU64` per 4KB page of the 4GB
/// physical space = 8MB.
#[cfg(feature = "jitv2")]
pub const GEN_WINDOW_SIZE: u64 = WINDOW_SIZE / GEN_RATIO;

/// Extra capability beyond what `Memory` offers — the reason ppmem exists.
///
/// Implemented by [`PpMemSpace`].
pub trait MappedMemory {
    /// Drop every mapping in the window, reverting it to inaccessible
    /// reservation. The window's address-space claim itself is retained.
    fn clear_mappings(&self);

    /// Map `bank` into the window at `[offset, offset+size)`, mirroring every
    /// `period` bytes.
    ///
    /// `period` is the SIMM's mirror period — `addr_mask + 1` as decoded by
    /// `MemoryController::memcfg_bank_info`. It is **not** necessarily the
    /// bank's size, and `size` is **not** necessarily a multiple of it:
    ///
    /// | SIMM | `addr_mask+1` (period) | `limit` (slot) |
    /// |---|---|---|
    /// | 8MB | 8MB | 4MB |
    /// | 32MB | 32MB | 16MB |
    /// | 128MB | 128MB | 64MB |
    ///
    /// For dual-rank SIMMs the configured slot is *half* the physical bank,
    /// because the two ranks are placed separately — so a region can be
    /// smaller than one period (map a prefix, no repeat), exactly one period,
    /// or several periods (repeat to fill).
    fn map_bank(&self, bank: usize, offset: u64, size: u64, period: u64) -> io::Result<()>;

    /// Map the first `size` bytes of `bank` at `offset`, without repeating.
    /// Used for the low-512KB alias, which is a partial view of bank 0.
    fn map_alias(&self, bank: usize, offset: u64, size: u64) -> io::Result<()>;

    /// Base of the window. Guest physical `p` lives at `base + p`.
    fn window_base(&self) -> *mut u8;

    /// Base of the generation window. The counter for guest physical `p` is at
    /// `gen_window_base() + (p >> 12)` — a pure shift off the physical
    /// address, no bank lookup, no masking, because banks' gen objects are
    /// mapped here in lockstep with their data.
    ///
    /// Stable for the process lifetime: the window itself never moves, only
    /// what is mapped inside it. A remap changes which physical counters back
    /// an address, which is exactly the "this page changed" signal the
    /// generation scheme exists to deliver — so a previously handed-out
    /// pointer stays correct rather than becoming stale.
    #[cfg(feature = "jitv2")]
    fn gen_window_base(&self) -> *mut AtomicU64;

    /// Bitmap of which 64MB regions are wholly backed by a direct mapping,
    /// one bit per region — 4GB / 64 = 64MB per bit, so the whole space fits
    /// in a single `u64`.
    ///
    /// A CPU fast path can then decide RAM-vs-bus in one shift and test:
    /// ```text
    /// if mask & (1u64 << (phys >> BITMAP_SHIFT)) != 0 { direct } else { bus }
    /// ```
    ///
    /// A bit is set only when the *entire* 64MB is mapped RAM, so partial
    /// regions (the low-512KB alias, anything sharing a 64MB span with MMIO)
    /// correctly read as unmapped and fall back to the bus. On the Indy map
    /// that means LOMEM and HIMEM — both 256MB-aligned and 256MB long — get
    /// four clean bits each, while the alias and the MC/HPC3/PROM cluster do
    /// not. See the design doc §3.1 for why this granularity was chosen over
    /// a finer, multi-word bitmap.
    fn mapped_bitmap(&self) -> u64;

    /// Redirect bitmap publication into the CPU's inline `ppmem_bitmap` field.
    ///
    /// Pass `MipsExecutor::ppmem_bitmap_ptr()`, queried once the executor is at
    /// its final address. From then on every mapping change writes straight
    /// through this pointer, so the CPU's RAM-vs-bus test is a load from the
    /// object it is already executing out of — no indirection through
    /// `PpMemSpace`, no atomic, no lock.
    ///
    /// Optional: until this is called the space publishes into a `u64` it owns
    /// itself, so the sink is always valid and publication never branches.
    ///
    /// # Safety
    ///
    /// `ptr` must point to a live `u64` that outlives this `PpMemSpace`, and
    /// must only ever be written from the CPU thread. Both hold for the CPU's
    /// inline field: the executor never moves once inside its `Arc`, and the
    /// only writer is the MEMCFG path on that same thread.
    unsafe fn set_bitmap_sink(&self, ptr: *mut u64);
}

/// One physical memory bank.
///
/// The ppmem counterpart of [`crate::mem::Memory`], with the same storage
/// layout and the same interface. Cheap to clone-share via `Arc`; the backing
/// object is refcounted so the bank stays alive as long as either the bank
/// handle or any mapping of it does.
pub struct PpMemory {
    mem: Arc<SharedMem>,
    /// Direct pointer to the bank's own private mapping of its storage.
    ///
    /// Banks are always reachable here regardless of whether they are mapped
    /// into the window, which is what keeps `BusDevice` working before
    /// `remap_banks` has run and for banks the MC leaves unmapped.
    base: *mut u8,
    size_bytes: usize,
    /// Kept for interface compatibility with `Memory::set_addr_mask`. Under
    /// ppmem the *mapping* enforces mirroring for window accesses, but this
    /// still masks accesses arriving through the bus, so the two paths agree.
    addr_mask: AtomicU64,
    /// Its own private mapping, kept alive for `base`'s lifetime.
    _own: AddrSpace,
    /// JIT v2: this bank's generation counters, one `AtomicU64` per 4KB page.
    ///
    /// A shared object exactly like the data storage, and mapped into the gen
    /// window in lockstep with it — so a mirrored data page and its counter
    /// are the *same physical memory* at every mirror, structurally rather
    /// than by folding an index. See [`PpMemSpace::map_bank`].
    #[cfg(feature = "jitv2")]
    gen_mem: Arc<SharedMem>,
    /// Direct pointer to this bank's private mapping of `gen_mem`, for
    /// `gen_ptr` on a bank that is not (or not yet) in a window.
    #[cfg(feature = "jitv2")]
    gen_base: *mut AtomicU64,
    #[cfg(feature = "jitv2")]
    _gen_own: AddrSpace,
}
 
// Safety: same contract as `Memory` — access is serialized through the bus.
unsafe impl Send for PpMemory {}
unsafe impl Sync for PpMemory {}

impl PpMemory {
    /// Create a bank of `size_mb` megabytes.
    ///
    /// Panics on allocation failure, matching `Memory::new`'s infallible
    /// signature — a machine that cannot allocate its RAM cannot boot, and
    /// every caller in `machine.rs` treats this as infallible.
    pub fn new(size_mb: usize) -> Self {
        Self::try_new(size_mb).expect("ppmem: failed to allocate RAM bank")
    }

    /// Fallible form of [`PpMemory::new`], for tests and for callers that want
    /// to fall back to `Memory`.
    pub fn try_new(size_mb: usize) -> io::Result<Self> {
        let size_bytes = size_mb * 1024 * 1024;
        assert!(size_bytes > 0, "ppmem: zero-sized bank");
        assert!(
            size_bytes % granularity() == 0,
            "ppmem: bank size {size_bytes:#x} is not a multiple of host granularity {:#x}",
            granularity()
        );

        let mem = Arc::new(SharedMem::new(size_bytes)?);
        // Give the bank its own mapping so it is directly addressable whether
        // or not the MC has mapped it into the window.
        let own = AddrSpace::reserve(size_bytes)?;
        unsafe { own.map(0, size_bytes, &mem, 0, Prot::ReadWrite)? };
        let base = own.base();

        // Generation counters get the same treatment: a shared object, sized
        // at the fixed 512:1 ratio, plus a private mapping of their own. The
        // ratio is what lets a window map data and gen in lockstep.
        #[cfg(feature = "jitv2")]
        let (gen_mem, gen_base, gen_own) = {
            let gen_bytes = super::map::align_up(size_bytes / GEN_RATIO as usize);
            let gm = Arc::new(SharedMem::new(gen_bytes)?);
            let go = AddrSpace::reserve(gen_bytes)?;
            unsafe { go.map(0, gen_bytes, &gm, 0, Prot::ReadWrite)? };
            let gb = go.base() as *mut AtomicU64;
            (gm, gb, go)
        };

        Ok(Self {
            mem,
            base,
            size_bytes,
            addr_mask: AtomicU64::new((size_bytes - 1) as u64),
            _own: own,
            #[cfg(feature = "jitv2")]
            gen_mem,
            #[cfg(feature = "jitv2")]
            gen_base,
            #[cfg(feature = "jitv2")]
            _gen_own: gen_own,
        })
    }

    /// Size in bytes.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.size_bytes
    }

    /// The shared object behind this bank, for mapping it into a window.
    pub fn shared(&self) -> &Arc<SharedMem> {
        &self.mem
    }

    #[inline(always)]
    fn mask(&self) -> usize {
        self.addr_mask.load(Ordering::Relaxed) as usize
    }

    /// Byte offset for `addr`, wrapped into the bank exactly as `Memory` does.
    #[inline(always)]
    fn off(&self, addr: u32) -> usize {
        (addr as usize) & self.mask()
    }

    /// Set a new address mask. AND-ed with `size_bytes-1` so it can never
    /// address outside the bank — same contract as `Memory::set_addr_mask`.
    pub fn set_addr_mask(&self, mask: u32) {
        let size_mask = (self.size_bytes - 1) as u64;
        self.addr_mask.store((mask as u64) & size_mask, Ordering::Relaxed);
    }

    /// JIT v2: the shared object holding this bank's generation counters, for
    /// mapping it into a window alongside the data.
    #[cfg(feature = "jitv2")]
    pub fn gen_shared(&self) -> &Arc<SharedMem> {
        &self.gen_mem
    }

    /// Number of generation counters this bank has (one per 4KB page).
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn gen_count(&self) -> usize {
        self.size_bytes / JITV2_PAGE_SIZE
    }

    /// JIT v2: bump the generation counter for the page containing `addr`.
    #[cfg(feature = "jitv2")]
    #[inline(always)]
    fn bump_gen(&self, addr: u32) {
        let page = self.off(addr) / JITV2_PAGE_SIZE;
        // Relaxed for the same reason `Memory::bump_gen` is: the publish-side
        // re-check provides the ordering (jit-v2-design.md §6.5).
        unsafe { (*self.gen_base.add(page)).fetch_add(1, Ordering::Relaxed) };
    }

    /// JIT v2: raw pointer to the generation counter for `addr`'s page.
    ///
    /// Valid for the lifetime of `&self`. Note the window offers the same
    /// counter at `gen_window_base + (phys >> 12) * 8` — the two resolve to the
    /// same physical memory whenever this bank is mapped, because
    /// [`PpMemSpace::map_bank`] maps data and gen from these same two objects
    /// in lockstep.
    #[cfg(feature = "jitv2")]
    #[inline]
    pub fn gen_ptr(&self, addr: u32) -> *const AtomicU64 {
        let page = self.off(addr) / JITV2_PAGE_SIZE;
        unsafe { self.gen_base.add(page) as *const AtomicU64 }
    }

    #[cfg(feature = "jitv2")]
    fn bump_gen_all(&self) {
        for i in 0..self.gen_count() {
            unsafe { (*self.gen_base.add(i)).fetch_add(1, Ordering::Relaxed) };
        }
    }

    /// Save bank contents to a raw binary file, big-endian words — byte-for-byte
    /// the same format `Memory::save_bin` writes.
    pub fn save_bin(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let words = self.as_words();
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for &w in words {
            bytes.extend_from_slice(&w.to_be_bytes());
        }
        std::fs::write(path, &bytes)
    }

    /// Load bank contents from a file written by [`PpMemory::save_bin`].
    pub fn load_bin(&self, path: impl AsRef<std::path::Path>) -> io::Result<()> {
        let bytes = std::fs::read(path)?;
        let words = self.as_words_mut();
        let n = (bytes.len() / 4).min(words.len());
        for i in 0..n {
            let b = &bytes[i * 4..(i + 1) * 4];
            words[i] = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen_all();
        Ok(())
    }

    /// Clone the bank's words in native endian, for the in-memory rollback
    /// checkpoint. Caller should quiesce other threads to avoid a torn read.
    pub fn snapshot_words(&self) -> Vec<u32> {
        self.as_words().to_vec()
    }

    /// Overwrite the bank's words from `src`, clamped to the bank's length.
    pub fn restore_words(&self, src: &[u32]) {
        let words = self.as_words_mut();
        let n = src.len().min(words.len());
        words[..n].copy_from_slice(&src[..n]);
        // Snapshot restore mutates RAM under any compiled artifact regardless
        // of content equality (jit-v2-design.md §7.1 channel 4, §7.6).
        #[cfg(feature = "jitv2")]
        self.bump_gen_all();
    }

    #[inline(always)]
    fn as_words(&self) -> &[u32] {
        unsafe { std::slice::from_raw_parts(self.base as *const u32, self.size_bytes / 4) }
    }

    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    fn as_words_mut(&self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.base as *mut u32, self.size_bytes / 4) }
    }
}

impl Resettable for PpMemory {
    fn power_on(&self) {
        // Hole-punching frees the physical pages and zeroes every mapping of
        // the bank in one syscall. Where that isn't available (macOS, Windows)
        // fall back to writing zeroes.
        if !self.mem.discard() {
            self.as_words_mut().fill(0);
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen_all();
    }
}

// ---------------------------------------------------------------------------
// BusDevice — byte-for-byte the same semantics as `Memory`'s implementation.
// ---------------------------------------------------------------------------

impl BusDevice for PpMemory {
    #[inline(always)]
    fn read8(&self, addr: u32) -> BusRead8 {
        unsafe { BusRead8::ok(*self.base.add(self.off(addr) ^ 3)) }
    }

    #[inline(always)]
    fn write8(&self, addr: u32, val: u8) -> u32 {
        unsafe { *self.base.add(self.off(addr) ^ 3) = val };
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline(always)]
    fn read16(&self, addr: u32) -> BusRead16 {
        unsafe {
            let p = self.base as *const u16;
            BusRead16::ok(*p.add((self.off(addr) >> 1) ^ 1))
        }
    }

    #[inline(always)]
    fn write16(&self, addr: u32, val: u16) -> u32 {
        unsafe {
            let p = self.base as *mut u16;
            *p.add((self.off(addr) >> 1) ^ 1) = val;
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline(always)]
    fn read32(&self, addr: u32) -> BusRead32 {
        unsafe {
            let p = self.base as *const u32;
            BusRead32::ok(*p.add(self.off(addr) >> 2))
        }
    }

    #[inline(always)]
    fn write32(&self, addr: u32, val: u32) -> u32 {
        unsafe {
            let p = self.base as *mut u32;
            *p.add(self.off(addr) >> 2) = val;
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline(always)]
    fn read64(&self, addr: u32) -> BusRead64 {
        unsafe {
            let p = self.base as *const u64;
            BusRead64::ok((*p.add(self.off(addr) >> 3)).rotate_left(32))
        }
    }

    #[inline(always)]
    fn write64(&self, addr: u32, val: u64) -> u32 {
        unsafe {
            let p = self.base as *mut u64;
            *p.add(self.off(addr) >> 3) = val.rotate_left(32);
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[inline]
    fn mem_ptr(&self, addr: u32) -> Option<*const u64> {
        unsafe {
            let p = self.base as *const u64;
            Some(p.add(self.off(addr) >> 3))
        }
    }

    #[inline]
    fn read_block(&self, addr: u32, buf: &mut [u64]) -> u32 {
        unsafe {
            let p = self.base as *const u64;
            let off = self.off(addr) >> 3;
            for (i, slot) in buf.iter_mut().enumerate() {
                *slot = (*p.add(off + i)).rotate_left(32);
            }
        }
        BUS_OK
    }

    #[inline]
    fn write_block(&self, addr: u32, buf: &[u64]) -> u32 {
        unsafe {
            let p = self.base as *mut u64;
            let off = self.off(addr) >> 3;
            for (i, &val) in buf.iter().enumerate() {
                *p.add(off + i) = val.rotate_left(32);
            }
        }
        // Per-page write cursor: bump once per page touched, not per qword
        // (jit-v2-design.md §7.2).
        #[cfg(feature = "jitv2")]
        {
            let start = self.off(addr) / JITV2_PAGE_SIZE;
            let last = addr.wrapping_add(((buf.len().max(1) - 1) as u32) * 8);
            let end = self.off(last) / JITV2_PAGE_SIZE;
            let bump = |page: usize| unsafe {
                (*self.gen_base.add(page)).fetch_add(1, Ordering::Relaxed);
            };
            if end >= start {
                for page in start..=end {
                    bump(page);
                }
            } else {
                // Wrapped around the mask: bump the two remaining spans.
                for page in start..self.gen_count() {
                    bump(page);
                }
                for page in 0..=end {
                    bump(page);
                }
            }
        }
        BUS_OK
    }

    #[inline(always)]
    fn write64_masked(&self, addr: u32, val: u64, mask: u64) -> u32 {
        unsafe {
            let p = (self.base as *mut u64).add(self.off(addr) >> 3);
            // Storage keeps qwords rotate_left(32); rotate val/mask to match.
            let old = *p;
            let v = val.rotate_left(32);
            let m = mask.rotate_left(32);
            *p = (old & !m) | (v & m);
        }
        #[cfg(feature = "jitv2")]
        self.bump_gen(addr);
        BUS_OK
    }

    #[cfg(feature = "jitv2")]
    #[inline]
    fn gen_ptr(&self, addr: u32) -> *const AtomicU64 {
        PpMemory::gen_ptr(self, addr)
    }
}

// Mirrors `impl BusDevice for Arc<Memory>` so Arc-wrapped banks work directly.
impl BusDevice for Arc<PpMemory> {
    #[inline(always)]
    fn read8(&self, addr: u32) -> BusRead8 { (**self).read8(addr) }
    #[inline(always)]
    fn write8(&self, addr: u32, val: u8) -> u32 { (**self).write8(addr, val) }
    #[inline(always)]
    fn read16(&self, addr: u32) -> BusRead16 { (**self).read16(addr) }
    #[inline(always)]
    fn write16(&self, addr: u32, val: u16) -> u32 { (**self).write16(addr, val) }
    #[inline(always)]
    fn read32(&self, addr: u32) -> BusRead32 { (**self).read32(addr) }
    #[inline(always)]
    fn write32(&self, addr: u32, val: u32) -> u32 { (**self).write32(addr, val) }
    #[inline(always)]
    fn read64(&self, addr: u32) -> BusRead64 { (**self).read64(addr) }
    #[inline(always)]
    fn write64(&self, addr: u32, val: u64) -> u32 { (**self).write64(addr, val) }
    #[inline(always)]
    fn write64_masked(&self, addr: u32, val: u64, mask: u64) -> u32 {
        (**self).write64_masked(addr, val, mask)
    }
    #[inline]
    fn mem_ptr(&self, addr: u32) -> Option<*const u64> { (**self).mem_ptr(addr) }
    #[inline]
    fn read_block(&self, addr: u32, buf: &mut [u64]) -> u32 { (**self).read_block(addr, buf) }
    #[inline]
    fn write_block(&self, addr: u32, buf: &[u64]) -> u32 { (**self).write_block(addr, buf) }
    #[cfg(feature = "jitv2")]
    #[inline]
    fn gen_ptr(&self, addr: u32) -> *const AtomicU64 { (**self).gen_ptr(addr) }
}

// ---------------------------------------------------------------------------
// The window
// ---------------------------------------------------------------------------

/// One live mapping inside the window, tracked so `clear_mappings` can undo it
/// and `mapped_bitmap` can be recomputed.
#[derive(Clone, Copy, Debug)]
struct Mapping {
    at: u64,
    len: u64,
}

/// The 4GB window that banks are mapped into, plus the banks themselves.
///
/// Holding the banks here is what lets `map_bank` take an index — matching the
/// brief's "accepts a vec of base,size memory bank pairs" — while
/// `physical.rs` keeps its own `Arc<PpMemory>` handles for bus access.
pub struct PpMemSpace {
    space: AddrSpace,
    /// The banks' backing objects, in bank order. Deliberately the *shared
    /// objects* rather than the `PpMemory` structs: `Physical` owns its banks
    /// by value, so the window cannot hold the banks themselves. Mapping only
    /// ever needs the object anyway, and the object is refcounted, so a bank
    /// and its mappings keep each other's storage alive.
    banks: Vec<Arc<SharedMem>>,
    /// The generation window: 8MB of `AtomicU64`, one per 4KB page of the 4GB
    /// physical space, reserved once and never moved.
    ///
    /// Mapped in lockstep with `space` — see [`PpMemSpace::map_bank`] — so a
    /// mirrored data page and its counter are the same physical memory. That
    /// makes `gen_ptr(phys) == gen_window_base + (phys >> 12) * 8` a pure
    /// function of the physical address, with no bank lookup and no masking.
    #[cfg(feature = "jitv2")]
    gen_space: AddrSpace,
    /// Per-bank generation objects, parallel to `banks`.
    #[cfg(feature = "jitv2")]
    gen_banks: Vec<Arc<SharedMem>>,
    /// Default publication target — see [`OwnBitmap`]. Read through
    /// `mapped_bitmap()`; written only via `state.sink`.
    own_bitmap: OwnBitmap,
    /// No lock: every path that touches this — `map_bank`, `map_alias`,
    /// `clear_mappings`, `set_bitmap_sink` — runs on the CPU thread, reached
    /// as CPU store → MC register write → `remap_banks`. No device thread ever
    /// enters this type. (`unsafe impl Sync` below records that contract.)
    state: UnsafeCell<SpaceState>,
}

struct SpaceState {
    mappings: Vec<Mapping>,
    /// Where the bitmap is published. Points at `own_bitmap` until the CPU
    /// hands over its inline field via [`MappedMemory::set_bitmap_sink`], so
    /// it is never null and publication is an unconditional store.
    sink: *mut u64,
}

/// The bitmap `PpMemSpace` publishes into before a CPU claims it.
///
/// Boxed so its address is stable: `PpMemSpace` is moved into place after
/// construction, which would invalidate a pointer to an inline field.
struct OwnBitmap(Box<u64>);

// Safety: `PpMemSpace` lives inside `Physical`, which is shared across threads
// via `Arc`, so it must be `Sync` to compile. The `UnsafeCell<SpaceState>` it
// contains is **not** synchronised, so this is a real contract rather than a
// formality:
//
//   * Every mutating entry point — `map_bank`, `map_alias`, `clear_mappings`,
//     `set_bitmap_sink` — is reached only from `Physical::remap_banks`, which
//     the MC invokes synchronously inside the CPU's store to MEMCFG. That is
//     always the CPU thread.
//   * Device threads (MC-DMA, seeq, VINO) reach RAM through `BusDevice` on the
//     individual `PpMemory` banks. They never touch `PpMemSpace`.
//   * `mapped_bitmap()` reads the sink and is likewise CPU-thread-only; the
//     hot-path reader is the CPU's own inline `MipsCore::ppmem_bitmap` field,
//     which is a plain `u64` and never goes through here.
//
// If a device thread ever needs to remap, this must become a real lock.
unsafe impl Send for PpMemSpace {}
unsafe impl Sync for PpMemSpace {}

impl PpMemSpace {
    /// Reserve a window over `banks`, borrowing each one's backing objects.
    ///
    /// Under `jitv2` this also reserves the parallel generation window and
    /// takes each bank's gen object, so the two can be mapped in lockstep.
    pub fn over(banks: &[PpMemory]) -> io::Result<Self> {
        let space = AddrSpace::reserve(WINDOW_SIZE as usize)?;
        let own_bitmap = OwnBitmap(Box::new(0u64));
        let sink = &*own_bitmap.0 as *const u64 as *mut u64;
        Ok(Self {
            space,
            banks: banks.iter().map(|b| b.shared().clone()).collect(),
            #[cfg(feature = "jitv2")]
            gen_space: AddrSpace::reserve(GEN_WINDOW_SIZE as usize)?,
            #[cfg(feature = "jitv2")]
            gen_banks: banks.iter().map(|b| b.gen_shared().clone()).collect(),
            own_bitmap,
            state: UnsafeCell::new(SpaceState {
                mappings: Vec::new(),
                sink,
            }),
        })
    }

    /// Test/bench helper: allocate fresh banks of the given sizes in MB and
    /// return them alongside a window over them.
    pub fn with_bank_sizes(sizes: &[usize]) -> io::Result<(Self, Vec<PpMemory>)> {
        let banks = sizes
            .iter()
            .map(|&mb| PpMemory::try_new(mb))
            .collect::<io::Result<Vec<_>>>()?;
        let space = Self::over(&banks)?;
        Ok((space, banks))
    }

    /// The `u64` the mapped-region bitmap is published into.
    ///
    /// Never null: points at this space's own `u64` until a CPU claims it via
    /// [`MappedMemory::set_bitmap_sink`], and at the CPU's inline field after.
    pub fn bitmap_ptr(&self) -> *const u64 {
        let st = unsafe { &*self.state.get() };
        st.sink as *const u64
    }

    pub fn bank_count(&self) -> usize {
        self.banks.len()
    }

    /// Recompute the 64MB-granularity bitmap from the live mapping list.
    ///
    /// A bit is set only if the whole 64MB region is covered, so a region that
    /// is partially mapped — or mapped alongside MMIO — reads as unmapped and
    /// keeps taking the bus path.
    fn recompute_bitmap(state: &mut SpaceState) {
        const REGION: u64 = 1 << BITMAP_SHIFT;
        let mut bits = 0u64;
        for bit in 0..64u32 {
            let start = (bit as u64) * REGION;
            let end = start + REGION;
            // Covered iff some mapping spans the entire region. Mappings are
            // bank-sized (>= 8MB) and region-aligned in practice, but check
            // the general case rather than assuming.
            let mut covered = start;
            loop {
                let Some(m) = state
                    .mappings
                    .iter()
                    .find(|m| m.at <= covered && m.at + m.len > covered)
                else {
                    break;
                };
                covered = m.at + m.len;
                if covered >= end {
                    bits |= 1u64 << bit;
                    break;
                }
            }
        }
        // Publish unconditionally — `sink` is always a live `u64`, either the
        // CPU's inline field or the one this space owns. Only ever reached
        // from the MEMCFG path on the CPU thread; see `set_bitmap_sink`.
        unsafe { *state.sink = bits };
    }

    fn record(state: &mut SpaceState, at: u64, len: u64) {
        // Drop anything this mapping fully replaces, then record it.
        state
            .mappings
            .retain(|m| !(m.at >= at && m.at + m.len <= at + len));
        state.mappings.push(Mapping { at, len });
        Self::recompute_bitmap(state);
    }
}

impl MappedMemory for PpMemSpace {
    fn clear_mappings(&self) {
        let st = unsafe { &mut *self.state.get() };
        for m in std::mem::take(&mut st.mappings) {
            // Best-effort: a failure here leaves the range mapped, which is
            // still safe — it just isn't reverted.
            let _ = unsafe { self.space.unmap(m.at as usize, m.len as usize) };
            #[cfg(feature = "jitv2")]
            {
                let gen_off = m.at / GEN_RATIO;
                let gen_len = m.len / GEN_RATIO;
                let gran = super::map::granularity() as u64;
                if gen_len >= gran && gen_off % gran == 0 && gen_len % gran == 0 {
                    let _ = unsafe {
                        self.gen_space.unmap(gen_off as usize, gen_len as usize)
                    };
                }
            }
        }
        unsafe { *st.sink = 0 };
    }

    fn map_bank(&self, bank: usize, offset: u64, size: u64, period: u64) -> io::Result<()> {
        let b = &self.banks[bank];
        let bank_size = b.len() as u64;
        assert!(size > 0, "ppmem: map_bank with zero size");
        assert!(period > 0, "ppmem: map_bank with zero period");
        assert!(
            period <= bank_size,
            "ppmem: mirror period {period:#x} exceeds bank size {bank_size:#x}"
        );
        assert!(
            offset + size <= WINDOW_SIZE,
            "ppmem: [{offset:#x}, +{size:#x}) escapes the 4GB window"
        );

        // Lay down consecutive `period`-sized views of the bank's first
        // `period` bytes until the region is full — that *is* SIMM mirroring.
        // A region smaller than one period maps just a prefix and never
        // repeats (the dual-rank case, where the slot is half the bank).
        let mut placed = 0u64;
        while placed < size {
            let chunk = period.min(size - placed);
            let at = offset + placed;
            unsafe {
                self.space
                    .map(at as usize, chunk as usize, b, 0, Prot::ReadWrite)?;
            }
            // Lockstep: the counters for this mirror land at the same offset
            // scaled by GEN_RATIO, from the same bank's gen object. Every
            // mirror therefore shares one physical counter page with the
            // others *and* with the bank's own private gen mapping.
            #[cfg(feature = "jitv2")]
            {
                let gen_off = at / GEN_RATIO;
                let gen_len = chunk / GEN_RATIO;
                let gran = super::map::granularity() as u64;
                if gen_len >= gran && gen_off % gran == 0 && gen_len % gran == 0 {
                    unsafe {
                        self.gen_space.map(
                            gen_off as usize,
                            gen_len as usize,
                            &self.gen_banks[bank],
                            0,
                            Prot::ReadWrite,
                        )?;
                    }
                }
            }
            placed += chunk;
        }

        let st = unsafe { &mut *self.state.get() };
        Self::record(st, offset, size);
        Ok(())
    }

    fn map_alias(&self, bank: usize, offset: u64, size: u64) -> io::Result<()> {
        let b = &self.banks[bank];
        assert!(
            size <= b.len() as u64,
            "ppmem: alias size {size:#x} exceeds bank size {:#x}",
            b.len()
        );
        assert!(
            offset + size <= WINDOW_SIZE,
            "ppmem: alias [{offset:#x}, +{size:#x}) escapes the 4GB window"
        );
        unsafe {
            self.space
                .map(offset as usize, size as usize, b, 0, Prot::ReadWrite)?;
        }
        // Alias the counters too, when the alias is big enough to cover whole
        // granules on the gen side. A 512KB alias maps 1KB of counters, which
        // is below host granularity, so its pages keep whatever the enclosing
        // region already mapped — harmless, since an alias is by definition the
        // same physical pages as the region it mirrors, hence the same
        // counters.
        #[cfg(feature = "jitv2")]
        {
            let gen_off = offset / GEN_RATIO;
            let gen_len = size / GEN_RATIO;
            let gran = super::map::granularity() as u64;
            if gen_len >= gran && gen_off % gran == 0 && gen_len % gran == 0 {
                unsafe {
                    self.gen_space.map(
                        gen_off as usize,
                        gen_len as usize,
                        &self.gen_banks[bank],
                        0,
                        Prot::ReadWrite,
                    )?;
                }
            }
        }
        let st = unsafe { &mut *self.state.get() };
        Self::record(st, offset, size);
        Ok(())
    }

    fn window_base(&self) -> *mut u8 {
        self.space.base()
    }

    #[cfg(feature = "jitv2")]
    fn gen_window_base(&self) -> *mut AtomicU64 {
        self.gen_space.base() as *mut AtomicU64
    }

    unsafe fn set_bitmap_sink(&self, ptr: *mut u64) {
        let st = unsafe { &mut *self.state.get() };
        // Carry the current value across: the boot-time remap runs before the
        // CPU thread starts, so banks are typically already mapped by now.
        let current = unsafe { *st.sink };
        st.sink = ptr;
        unsafe { *ptr = current };
    }

    fn mapped_bitmap(&self) -> u64 {
        // Read the live sink rather than a cached copy, so this agrees with
        // whatever the CPU is testing against.
        let st = unsafe { &*self.state.get() };
        unsafe { *st.sink }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: usize = 1024 * 1024;

    /// Every `BusDevice` access width must produce byte-identical results to
    /// `Memory`. This is the substitutability contract — if this passes,
    /// `PpMemory` can stand in for `Memory` anywhere.
    #[test]
    fn matches_memory_byte_for_byte() {
        use crate::mem::Memory;
        let a = Memory::new(8);
        let b = PpMemory::new(8);

        // Interleave every width, including unaligned-in-word sub-accesses and
        // the byte/halfword XOR-swizzle paths.
        for i in 0..512u32 {
            let addr = i * 8;
            a.write64(addr, 0x0123_4567_89AB_CDEF ^ (i as u64));
            b.write64(addr, 0x0123_4567_89AB_CDEF ^ (i as u64));
            a.write32(addr + 4, 0xDEAD_0000 | i);
            b.write32(addr + 4, 0xDEAD_0000 | i);
            a.write16(addr + 2, i as u16);
            b.write16(addr + 2, i as u16);
            a.write8(addr + 1, i as u8);
            b.write8(addr + 1, i as u8);
        }
        for i in 0..512u32 {
            let addr = i * 8;
            assert_eq!(a.read64(addr).data, b.read64(addr).data, "read64 @{addr:#x}");
            assert_eq!(a.read32(addr).data, b.read32(addr).data, "read32 @{addr:#x}");
            assert_eq!(a.read32(addr + 4).data, b.read32(addr + 4).data);
            for k in 0..8 {
                assert_eq!(
                    a.read8(addr + k).data,
                    b.read8(addr + k).data,
                    "read8 @{:#x}",
                    addr + k
                );
            }
            for k in [0u32, 2, 4, 6] {
                assert_eq!(
                    a.read16(addr + k).data,
                    b.read16(addr + k).data,
                    "read16 @{:#x}",
                    addr + k
                );
            }
        }

        // Block and masked accesses.
        let mut ba = [0u64; 8];
        let mut bb = [0u64; 8];
        a.read_block(0x400, &mut ba);
        b.read_block(0x400, &mut bb);
        assert_eq!(ba, bb, "read_block");

        a.write_block(0x800, &ba);
        b.write_block(0x800, &bb);
        a.write64_masked(0x800, 0xFFFF_FFFF_FFFF_FFFF, 0x0000_FFFF_FFFF_0000);
        b.write64_masked(0x800, 0xFFFF_FFFF_FFFF_FFFF, 0x0000_FFFF_FFFF_0000);
        assert_eq!(a.read64(0x800).data, b.read64(0x800).data, "write64_masked");

        // And the snapshot representation must match exactly.
        assert_eq!(a.snapshot_words(), b.snapshot_words(), "snapshot_words");
    }

    #[test]
    fn addr_mask_wraps_like_memory() {
        use crate::mem::Memory;
        let mut a = Memory::new(8);
        let b = PpMemory::new(8);
        // Mirror an 8MB bank every 1MB.
        a.set_addr_mask(0x000F_FFFF);
        b.set_addr_mask(0x000F_FFFF);
        a.write32(0x40, 0xCAFEBABE);
        b.write32(0x40, 0xCAFEBABE);
        // Reading one mirror period up must see the same word.
        assert_eq!(a.read32(0x10_0040).data, 0xCAFEBABE);
        assert_eq!(b.read32(0x10_0040).data, 0xCAFEBABE, "ppmem mask wrap");
        assert_eq!(a.read32(0x10_0040).data, b.read32(0x10_0040).data);
    }

    #[test]
    fn snapshot_restore_roundtrip() {
        let m = PpMemory::new(8);
        for i in 0..32u32 {
            m.write32(i * 4, 0xDEAD_0000 + i);
        }
        let snap = m.snapshot_words();
        for i in 0..32u32 {
            m.write32(i * 4, 0xCAFEBABE);
        }
        m.restore_words(&snap);
        for i in 0..32u32 {
            assert_eq!(m.read32(i * 4).data, 0xDEAD_0000 + i, "word {i}");
        }
    }

    #[test]
    fn power_on_zeroes() {
        let m = PpMemory::new(8);
        m.write32(0x1000, 0xFFFF_FFFF);
        m.write32(0x20_0000, 0xAAAA_AAAA);
        m.power_on();
        assert_eq!(m.read32(0x1000).data, 0);
        assert_eq!(m.read32(0x20_0000).data, 0);
    }

    #[test]
    fn save_load_bin_roundtrip() {
        let dir = std::env::temp_dir().join(format!("ppmem-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bank.bin");

        let a = PpMemory::new(8);
        for i in 0..64u32 {
            a.write32(i * 4, 0x1234_0000 + i);
        }
        a.save_bin(&path).unwrap();

        let b = PpMemory::new(8);
        b.load_bin(&path).unwrap();
        for i in 0..64u32 {
            assert_eq!(b.read32(i * 4).data, 0x1234_0000 + i, "word {i}");
        }
        // Format must be interchangeable with Memory's.
        let c = crate::mem::Memory::new(8);
        c.load_bin(&path).unwrap();
        assert_eq!(c.read32(0).data, b.read32(0).data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The headline capability: an undersized bank repeats across its slot,
    /// and writes through one mirror are visible through the others because
    /// they are the same physical pages.
    #[test]
    fn undersized_bank_repeats_across_region() {
        let (space, _banks) = PpMemSpace::with_bank_sizes(&[8]).unwrap();
        // 8MB bank filling a 32MB slot => 4 mirrors.
        space.map_bank(0, 0, 32 * MB as u64, 8 * MB as u64).unwrap();

        let w = space.window_base();
        unsafe {
            *(w as *mut u32) = 0x1234_5678;
            for i in 1..4usize {
                assert_eq!(
                    *(w.add(i * 8 * MB) as *const u32),
                    0x1234_5678,
                    "mirror {i} not aliased"
                );
            }
            // Write through the last mirror, observe through the first.
            *(w.add(3 * 8 * MB + 64) as *mut u32) = 0xABCD_EF01;
            assert_eq!(*(w.add(64) as *const u32), 0xABCD_EF01, "mirror not bidirectional");
        }
    }

    /// A bank mapped into the window and the same bank accessed through
    /// `BusDevice` must be the same memory — that is what makes ppmem
    /// pluggable into the existing bus rather than a parallel universe.
    #[test]
    fn window_and_busdevice_see_the_same_memory() {
        let (space, banks) = PpMemSpace::with_bank_sizes(&[8]).unwrap();
        space.map_bank(0, 0x0800_0000, 8 * MB as u64, 8 * MB as u64).unwrap();
        let bank = &banks[0];

        // Write via the bus, read via the window.
        bank.write32(0x100, 0xFEED_FACE);
        unsafe {
            let p = space.window_base().add(0x0800_0000 + 0x100) as *const u32;
            assert_eq!(*p, 0xFEED_FACE, "window does not see bus write");
            // ...and the reverse.
            *(p as *mut u32) = 0x0BAD_C0DE;
        }
        assert_eq!(bank.read32(0x100).data, 0x0BAD_C0DE, "bus does not see window write");
    }

    #[test]
    fn bitmap_marks_only_fully_covered_regions() {
        let (space, _banks) = PpMemSpace::with_bank_sizes(&[64, 8]).unwrap();
        assert_eq!(space.mapped_bitmap(), 0, "nothing mapped yet");

        // 64MB bank at 0x08000000 fills exactly bit 2 (0x08000000 >> 26 == 2).
        space.map_bank(0, 0x0800_0000, 64 * MB as u64, 64 * MB as u64).unwrap();
        let bm = space.mapped_bitmap();
        assert_eq!(bm & (1 << 2), 1 << 2, "bit 2 should be fully covered");
        assert_eq!(bm & (1 << 3), 0, "bit 3 must not be set");

        // An 8MB alias at 0 covers only part of bit 0 -> must stay clear.
        space.map_alias(1, 0, 512 * 1024).unwrap();
        assert_eq!(space.mapped_bitmap() & 1, 0, "partial region must read unmapped");
    }

    /// The bitmap must land in whatever `u64` the CPU handed over, so the hot
    /// path reads it as a plain inline field.
    #[test]
    fn bitmap_publishes_into_the_cpu_field() {
        let (space, _banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();

        // Stand in for MipsCore.ppmem_bitmap. Boxed so its address is stable.
        let cpu_field = Box::new(0u64);
        let ptr = &*cpu_field as *const u64 as *mut u64;

        // Map before registering: the boot-time remap runs before the CPU
        // thread exists, so the current value must carry across.
        space.map_bank(0, 0x0800_0000, 64 * MB as u64, 64 * MB as u64).unwrap();
        let expected = space.mapped_bitmap();
        assert_ne!(expected, 0);

        unsafe { space.set_bitmap_sink(ptr) };
        assert_eq!(*cpu_field, expected, "existing bitmap not carried to the CPU field");

        // Subsequent changes must publish straight through.
        space.map_bank(0, 0x1000_0000, 64 * MB as u64, 64 * MB as u64).unwrap();
        assert_eq!(*cpu_field, space.mapped_bitmap(), "later remap did not publish");

        space.clear_mappings();
        assert_eq!(*cpu_field, 0, "clear_mappings did not publish");
    }

    /// The bit the CPU tests must be the one covering that physical address.
    #[test]
    fn cpu_bitmap_test_selects_the_right_region() {
        let (space, _banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
        space.map_bank(0, 0x0800_0000, 64 * MB as u64, 64 * MB as u64).unwrap();
        let bm = space.mapped_bitmap();

        // This is exactly the test the CPU hot path performs.
        let mapped = |phys: u64| bm & (1u64 << (phys >> BITMAP_SHIFT)) != 0;

        assert!(mapped(0x0800_0000), "start of the mapped bank");
        assert!(mapped(0x0BFF_FFFF), "end of the mapped 64MB region");
        assert!(!mapped(0x0C00_0000), "just past the mapped region");
        assert!(!mapped(0x1FA0_0000), "MC/HPC3/PROM must stay on the bus path");
        assert!(!mapped(0x0000_0000), "low alias region must stay on the bus path");
    }

    /// The direct window path and the `BusDevice` path must produce identical
    /// bytes for block accesses — this is what `Physical::read_block` /
    /// `write_block` rely on when they bypass the device dispatch.
    #[test]
    fn window_block_access_matches_busdevice() {
        let (space, banks) = PpMemSpace::with_bank_sizes(&[8]).unwrap();
        let base = 0x0800_0000u64;
        space.map_bank(0, base, 8 * MB as u64, 8 * MB as u64).unwrap();
        let bank = &banks[0];

        // Seed through the bus, read back through the window the way
        // Physical::read_block does.
        let src: Vec<u64> = (0..16u64).map(|i| 0x0102_0304_0506_0708 ^ (i << 8)).collect();
        bank.write_block(0x2000, &src);

        let w = unsafe { space.window_base().add((base + 0x2000) as usize) as *const u64 };
        for (i, &want) in src.iter().enumerate() {
            let got = unsafe { (*w.add(i)).rotate_left(32) };
            assert_eq!(got, want, "window read_block mismatch at qword {i}");
        }

        // ...and the reverse: write through the window, read through the bus.
        let dst: Vec<u64> = (0..16u64).map(|i| 0xAABB_CCDD_EEFF_0011 ^ i).collect();
        unsafe {
            let wm = w as *mut u64;
            for (i, &v) in dst.iter().enumerate() {
                *wm.add(i) = v.rotate_left(32);
            }
        }
        let mut back = [0u64; 16];
        bank.read_block(0x2000, &mut back);
        assert_eq!(&back[..], &dst[..], "bus does not see window write_block");
    }

    /// A directly-mapped address must resolve to the same host pointer the
    /// cache's line-fill fast path would use, and dereference identically to
    /// `mem_ptr` on the bank.
    #[test]
    fn window_ptr_matches_bank_mem_ptr() {
        let (space, banks) = PpMemSpace::with_bank_sizes(&[8]).unwrap();
        let base = 0x0800_0000u64;
        space.map_bank(0, base, 8 * MB as u64, 8 * MB as u64).unwrap();

        banks[0].write64(0x400, 0xCAFE_F00D_1234_5678);
        let via_bank = banks[0].mem_ptr(0x400).unwrap();
        let via_window = unsafe { space.window_base().add((base + 0x400) as usize) as *const u64 };
        assert_eq!(
            unsafe { *via_bank },
            unsafe { *via_window },
            "window pointer and bank mem_ptr disagree"
        );
    }

    #[test]
    fn clear_mappings_resets_bitmap() {
        let (space, _banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
        space.map_bank(0, 0x0800_0000, 64 * MB as u64, 64 * MB as u64).unwrap();
        assert_ne!(space.mapped_bitmap(), 0);
        space.clear_mappings();
        assert_eq!(space.mapped_bitmap(), 0, "bitmap not cleared");
    }

    /// The low-512KB alias: bank 0's first 512KB visible at physical 0, the
    /// same pages as at LOMEM_BASE. Replaces `physical.rs`'s AliasBus.
    #[test]
    fn low_alias_shares_pages_with_bank0() {
        let (space, _banks) = PpMemSpace::with_bank_sizes(&[8]).unwrap();
        space.map_bank(0, 0x0800_0000, 8 * MB as u64, 8 * MB as u64).unwrap();
        space.map_alias(0, 0, 512 * 1024).unwrap();

        unsafe {
            let w = space.window_base();
            *(w.add(0x0800_0000 + 0x40) as *mut u32) = 0x5EED_1234;
            assert_eq!(
                *(w.add(0x40) as *const u32),
                0x5EED_1234,
                "low alias does not mirror LOMEM"
            );
            *(w.add(0x80) as *mut u32) = 0x9999_8888;
            assert_eq!(
                *(w.add(0x0800_0000 + 0x80) as *const u32),
                0x9999_8888,
                "low alias write not visible at LOMEM"
            );
        }
    }

    #[cfg(feature = "jitv2")]
    #[test]
    fn write_bumps_only_the_touched_page_gen() {
        let m = PpMemory::new(8);
        let g0 = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let g1 = unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) };
        m.write32(0, 0x1234);
        assert_eq!(unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) }, g0 + 1);
        assert_eq!(
            unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) },
            g1,
            "untouched page must not bump"
        );
    }

    #[cfg(feature = "jitv2")]
    #[test]
    fn read_does_not_bump_gen() {
        let m = PpMemory::new(8);
        let before = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let _ = m.read32(0);
        assert_eq!(unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) }, before);
    }

    #[cfg(feature = "jitv2")]
    #[test]
    fn write_block_bumps_every_page_it_spans() {
        let m = PpMemory::new(8);
        let start = (JITV2_PAGE_SIZE - 8) as u32;
        let g0 = unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) };
        let g1 = unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) };
        m.write_block(start, &[0u64, 0u64]);
        assert_eq!(unsafe { (*m.gen_ptr(0)).load(Ordering::Relaxed) }, g0 + 1);
        assert_eq!(
            unsafe { (*m.gen_ptr(JITV2_PAGE_SIZE as u32)).load(Ordering::Relaxed) },
            g1 + 1
        );
    }

    /// All mirrors of a page must share one generation counter — they are one
    /// physical page, so a write through any of them invalidates artifacts for
    /// all of them.
    #[cfg(feature = "jitv2")]
    #[test]
    fn mirrored_pages_share_one_gen_counter() {
        let m = PpMemory::new(8);
        m.set_addr_mask(0x000F_FFFF); // mirror every 1MB
        let a = m.gen_ptr(0x1000);
        let b = m.gen_ptr(0x10_1000); // same page, one mirror up
        assert_eq!(a, b, "mirrored addresses must map to the same gen counter");
    }

    /// The headline property of the window-mapped gen scheme: an undersized
    /// bank repeated across a region has, at every mirror, the *same physical*
    /// counter — because the gen window is mapped in lockstep with the data
    /// window, not because an index was folded by a mask.
    #[cfg(feature = "jitv2")]
    #[test]
    fn gen_window_mirrors_in_lockstep_with_data() {
        let (space, banks) = PpMemSpace::with_bank_sizes(&[8]).unwrap();
        // 8MB bank filling a 32MB slot at LOMEM => 4 mirrors.
        let base = 0x0800_0000u64;
        space.map_bank(0, base, 32 * MB as u64, 8 * MB as u64).unwrap();

        let gw = space.gen_window_base();
        let counter = |phys: u64| unsafe { &*gw.add((phys >> 12) as usize) };

        // Bump the counter for the first page through mirror 0, and observe it
        // through every other mirror.
        let before = counter(base).load(Ordering::Relaxed);
        counter(base).fetch_add(1, Ordering::Relaxed);
        for i in 1..4u64 {
            let mirrored = base + i * 8 * MB as u64;
            assert_eq!(
                counter(mirrored).load(Ordering::Relaxed),
                before + 1,
                "mirror {i} does not share the counter"
            );
        }

        // And the bank's own gen_ptr must be that same physical counter: a
        // write through the bus must be visible in the window.
        let g0 = counter(base).load(Ordering::Relaxed);
        banks[0].write32(0, 0xDEAD_BEEF);
        assert_eq!(
            counter(base).load(Ordering::Relaxed),
            g0 + 1,
            "bus write did not bump the counter visible in the gen window"
        );
    }

    /// `gen_ptr(phys)` must be a pure shift off the physical address — the
    /// same constant relation the data window has.
    #[cfg(feature = "jitv2")]
    #[test]
    fn gen_window_offset_is_a_pure_shift() {
        let (space, _banks) = PpMemSpace::with_bank_sizes(&[8]).unwrap();
        space.map_bank(0, 0x0800_0000, 8 * MB as u64, 8 * MB as u64).unwrap();
        let gw = space.gen_window_base();
        for phys in [0x0800_0000u64, 0x0800_1000, 0x0840_0000, 0x087F_F000] {
            let expect = unsafe { gw.add((phys >> 12) as usize) };
            let got = unsafe { (gw as *mut u8).add((phys >> 12) as usize * 8) } as *mut AtomicU64;
            assert_eq!(expect, got, "gen offset is not (phys >> 12) * 8 at {phys:#x}");
        }
    }

    /// The gen window is sized to cover the whole 4GB space at 4KB per counter.
    #[cfg(feature = "jitv2")]
    #[test]
    fn gen_window_sizing() {
        assert_eq!(GEN_RATIO, 512, "4096 bytes per page / 8 bytes per counter");
        assert_eq!(GEN_WINDOW_SIZE, 8 * 1024 * 1024, "4GB / 512 = 8MB");
        // The smallest supported bank must still produce a whole number of
        // host-granularity-sized gen blocks.
        let min_bank = 8 * MB;
        let gen_bytes = min_bank / GEN_RATIO as usize;
        assert_eq!(gen_bytes, 16 * 1024, "8MB bank => 16KB of counters");
    }
}
