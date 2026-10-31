//! A cache that is visible to software but stays out of the data path.
//!
//! Every load, store and instruction fetch goes straight to memory, exactly as
//! in [`PassthroughCacheOf`](crate::cpu::mips_cache_v2::PassthroughCacheOf). What
//! this adds is a *shadow*: tag and data arrays that only the CACHE
//! instruction ever touches.
//!
//! The reasoning is that a cache's effect on a functional emulator is entirely
//! observational. Its timing is invisible to the guest, and its contents are
//! invisible too as long as they agree with memory — which here they trivially
//! do, because memory is the only store. What *is* visible is the CACHE
//! instruction, the geometry reported through CP0 Config, and the diagnostics
//! a PROM runs against both. So model those, and let the host CPU — which
//! already has real caches, real speculation and real out-of-order execution —
//! get on with it unimpeded.
//!
//! Two things fall out of this beyond speed:
//!
//! - **Coherency is exact and free.** There is no stale line to miss on a
//!   self-modifying store, no L1/L2 inclusion policy to get wrong, and no way
//!   for a missed writeback to lose data.
//! - **Diagnostics round-trip.** The shadow stores whatever bits are written
//!   to it and returns them unchanged, so a walking-1s test over tag or data
//!   SRAM passes without anyone having to know the hardware's field layout.
//!   Where the emulator *must* know the layout — to decide whether a line is
//!   valid, say — that is a separate question from storing the bits. The one
//!   exception is the R10000's secondary tag, whose reserved bits read as zero
//!   (`R10K_L2_TAG_MASK`).
//!
//! The shadow is not consulted by `read`/`write`/`fetch`, with one exception
//! that costs a normal access one predicted branch: on the R10000 family, a
//! secondary line the guest *staged* — a valid tag written by Index Store Tag
//! and data by Index Store Data — is a real hit for later cached loads, stores
//! and fetches, served from the shadow without a bus access. A line staged
//! in the primary instruction cache the same way serves fetches, before the
//! secondary cache. See [`Staged`].
//! Nothing else about the shadow reaches the data path: a line the guest never
//! staged is memory, as before, and ordinary misses never fill or evict. So a
//! staged line lives until an explicit index or hit operation, an Index Store
//! Tag or power-on ends it; a guest that reuses the memory without
//! invalidating reads the stale line, where hardware would have evicted it.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::cpu::mips_cache_v2::{
    cache_op_name, model, r4k_load_taglo, r4k_store_taglo, CpuModel, FetchInstrResult, MipsCache,
    C_HINV, C_HWBINV, C_IINV, C_ILT, C_IST, C_IWBINV, C_R10K_CBARRIER, C_R10K_ILD, C_R10K_ISD, CACH_PD, CACH_PI, CACH_SD, CACH_SI,
};
use crate::traits::{BusDevice, BusRead64, BUS_OK};
use crate::devlog::{LogModule, CACHE_LOG_HIT, CACHE_LOG_OP, devlog_is_active, devlog_mask};

/// Shadow tag and data arrays for one cache.
///
/// Indexed `set * WAYS + way`. The ways are here and nowhere else: a CACHE
/// index operation addresses one *way* of one set, so software can see them,
/// and the IP28 PROM's tag diagnostic depends on it — it writes different tags
/// to the two ways of a set and reads one back. Keeping them costs an array
/// dimension in a structure no load or store ever consults.
struct Shadow {
    /// One raw tag per line, stored and returned verbatim. 64 bits: an
    /// R10000 secondary tag carries a 40-bit physical address.
    tags: Box<[u64]>,
    /// Data array in u64 slots. Empty where the model has no data shadow.
    data: Box<[u64]>,
    /// The check bits stored with each data slot. On an R10000 these ride
    /// with the data: `Index_Store_Data` takes them from CP0 ECC and
    /// `Index_Load_Data` returns them there, so they are storage like the
    /// data itself, not a computed value.
    ecc: Box<[u32]>,
    /// The most-recently-used bit for each set. Shared across the set's ways
    /// — see `MRU_BIT`. Hardware state rather than tag storage, which is why
    /// it is kept here and re-applied on read instead of living in `tags`.
    mru: Box<[bool]>,
}

impl Shadow {
    fn new(lines: usize, data_words: usize) -> Self {
        Self {
            tags: vec![0u64; lines.max(1)].into_boxed_slice(),
            data: vec![0u64; data_words].into_boxed_slice(),
            ecc: vec![0u32; data_words].into_boxed_slice(),
            mru: vec![false; (lines.max(1) / WAYS).max(1)].into_boxed_slice(),
        }
    }
}

/// The most-recently-used bit: TagHi[31], i.e. bit 63 of the assembled tag.
///
/// It is **one bit per set, shared between the ways** — not a per-way tag bit.
/// Writing a tag with it set through either way marks the set, and reading
/// the tag of *either* way reports it. The IP28 PROM's diagnostic tests
/// exactly that: it writes the bit through way 0, writes the next set to
/// clobber the signal line, then reads way 1 back and requires the bit to be
/// there. Storing "which way is MRU" and reporting it only on that way passes
/// nothing, because the way that is read is never the way that was written.
const MRU_BIT: u64 = 1 << 63;

/// R10000 only: ways per set, as CACHE index operations address them. Bit 0
/// of the index selects the way; the set starts above the line offset. The
/// R4400 and R5000 index differently — see `tag_slot`.
const WAYS: usize = 2;

/// Bits an R10000 secondary-cache tag retains: MRU (bit 63), the physical tag
/// (TagHi 3:0 and TagLo 31:14), state, virtual index and ECC. TagLo bits 9, 12
/// and 13 and TagHi 30:4 are reserved and read as zero (R10000 UM p. 270,
/// figure 11-31). Other parts keep every bit; see `CpuModel::SECONDARY_TAG_MASK`.
const R10K_L2_TAG_MASK: u64 = 0x8000_000f_ffff_cdff;

/// The secondary tag fields a cached access is compared against. The tag
/// register pair holds PA[35:18] in TagLo[31:14] and PA[39:36] in TagHi[3:0],
/// which the assembled 64-bit tag keeps at bits 31:14 and 35:32. NEC
/// U10278EJ4V0UM pp.267-270.
const L2_TAG_PA_MASK: u64 = 0xf_ffff_c000;

/// Where the secondary state field sits in the assembled tag: TagLo[11:10],
/// 0 Invalid, 1 Shared, 2 CleanExclusive, 3 DirtyExclusive.
const L2_STATE_SHIFT: u32 = 10;
const L2_STATE_DIRTY_EXCLUSIVE: u64 = 3;

/// The primary instruction-cache tag, from the register pair (NEC
/// U10278EJ4V0UM pp.267-270): TagLo[31:8] is PA[35:12], TagLo[6] the state (1 valid), and
/// TagHi[3:0] PA[39:36], kept at bits 35:32 of the assembled tag.
const L1I_TAG_PA_MASK: u64 = 0xf_ffff_ff00;
const L1I_VALID: u64 = 1 << 6;

#[inline(always)]
fn l1i_tag_of(pa: u64) -> u64 {
    (((pa >> 12) & 0xff_ffff) << 8) | (((pa >> 36) & 0xf) << 32)
}

/// Does the secondary tag hold a valid line? Every state but Invalid does.
#[inline(always)]
fn l2_tag_valid(tag: u64) -> bool {
    (tag >> L2_STATE_SHIFT) & 3 != 0
}

/// The tag bits a physical address is compared with.
#[inline(always)]
fn l2_tag_of(pa: u64) -> u64 {
    (((pa >> 18) & 0x3_ffff) << 14) | (((pa >> 36) & 0xf) << 32)
}

/// **Inferred, not established.** Which word of a staged doubleword a word
/// access at the lower address sees. A secondary quadword's doubleword is
/// staged as TagHi (bits 63:32) and TagLo (bits 31:0), but neither NEC manual
/// says how those halves become the words of a primary refill. The only
/// evidence is software that stages a line this way and then fetches it: it
/// is consistent
/// only with the TagLo word landing at the lower address, and that is a
/// simulation, not a measurement. `false` is the manual-literal order. Index
/// Load Data is unaffected; it returns the registers as they were stored.
const STAGED_TAGLO_AT_LOWER_ADDRESS: bool = true;

/// A staged doubleword as a big-endian memory doubleword: what a `ld` at its
/// address returns. The conversion is its own inverse.
#[inline(always)]
fn staged_to_memory(d: u64) -> u64 {
    if STAGED_TAGLO_AT_LOWER_ADDRESS { d.rotate_left(32) } else { d }
}

/// The secondary lines a guest has made valid, and so what a cached access
/// must consult before memory.
///
/// A line is staged when its tag is valid; its data are whatever Index Store
/// Data put in the data shadow, so the shadow arrays *are* the staged lines and
/// this holds only the index needed to avoid looking: `valid` for the hot paths
/// and `pages` for the page-granular ones. With `valid == 0`, which is every
/// guest that never stages a line, `read`, `write` and `fetch` pay one load and
/// a not-taken branch.
///
/// The page set exists because translations are cached per page. The
/// interpreter's and the JIT's inline loads and stores reach RAM through
/// ppmem's window without calling the cache, and the JIT compiles a code page
/// from memory; both are kept off a page in `pages` (`MipsCache::page_staged`),
/// so every access to it comes through the cache. That is the guard against
/// stale compiled code that SMC needs too, done once at the moment a line
/// becomes valid instead of on every store.
struct Staged {
    /// Secondary tags with a valid state.
    valid: u32,
    /// Pages holding at least one such line, and how many.
    pages: HashMap<u64, u32>,
    /// `pages` gained or lost a member since `take_staged_changed`.
    changed: bool,
}

/// Is the shadow cache's operation trace on, at this level of detail?
///
/// `log l2c mask op` traces the tag operations a diagnostic actually cares
/// about; `log l2c mask op+hit` adds the data-array walk as well. That
/// distinction is load-bearing rather than tidy: the IP28 PROM issues 65k+
/// `Index_Store_Data` ops walking the array, and tracing all of them once
/// slowed the guest so much a run never reached the test it was there to
/// observe.
///
/// This used to be `std::env::var_os("IRIS_SHADOW_CACHEOPS")`, uncached, six
/// times per `cache_op` — a getenv on every CACHE instruction in every build.
/// `devlog` costs two relaxed atomic loads and can be turned on, masked and
/// redirected to a file while the machine runs.
#[inline(always)]
fn l2_op_log(bit: u32) -> bool {
    devlog_is_active(LogModule::L2c) && (devlog_mask(LogModule::L2c) & bit) != 0
}

/// A `MipsCache` whose contents are only ever observed through CACHE ops.
///
/// Const parameters carry the geometry that software can read back, and the
/// processor identity. Nothing here affects the speed of a load.
///
/// `MODEL` is `mips_cache_v2::model`, as for `CpuCache`, and decides
/// everything shape cannot: how a CACHE index op picks its line and way, what
/// a tag keeps, and the R10000's MRU bit, data arrays and op encodings.
pub struct ShadowCache<
    const IC_SIZE: usize,
    const IC_LINE: usize,
    const DC_SIZE: usize,
    const DC_LINE: usize,
    const L2_SIZE: usize,
    const L2_LINE: usize,
    const MIPS4: bool,
    const PRID: u32,
    const FIR: u32,
    const TLB_ENTRIES: usize,
    const MODEL: u8,
> {
    downstream: Arc<dyn BusDevice>,
    llbit: UnsafeCell<bool>,
    lladdr: UnsafeCell<u64>,
    /// CP0 ECC on the way in to a store, and on the way out of a load.
    ecc_in: UnsafeCell<u32>,
    ecc_out: UnsafeCell<u32>,
    ic: UnsafeCell<Shadow>,
    dc: UnsafeCell<Shadow>,
    l2: UnsafeCell<Shadow>,
    /// tcache: ppmem's window base, its mapped-region bitmap (published into
    /// directly on every remap) and the jitv2 generation window. This cache
    /// reads nothing through them — `read`/`write` go to `downstream`, whose
    /// ppmem path uses the same window — they are here only so jitv2's inline
    /// tcache path can serve this model too (`jit_dc_geometry`).
    tc_base: UnsafeCell<*mut u8>,
    tc_bitmap: UnsafeCell<u64>,
    /// The window's `ppmem::window_shift`, set before the CPU runs.
    tc_shift: UnsafeCell<u32>,
    #[cfg(feature = "jitv2")]
    tc_gen: UnsafeCell<*mut std::sync::atomic::AtomicU64>,
    /// Decoded instructions, by physical page. Never touches the shadow
    /// arrays above — see `decode_cache`.
    decode: UnsafeCell<crate::cpu::decode_cache::DecodeCache>,
    /// See [`MipsCache::set_pd_tag_store_hook`]. Consulted by `cache_op` only.
    pd_tag_hook: UnsafeCell<Option<Box<dyn Fn(u32) + Send + Sync>>>,
    /// Secondary lines the guest staged. R10000 family only; empty otherwise.
    staged: UnsafeCell<Staged>,
    /// The translated address of the CACHE operand about to arrive in
    /// `cache_op`: the R10000's secondary index and way come from it.
    op_pa: UnsafeCell<Option<u64>>,
    /// Serve fetches from staged primary I lines. Off unless `IRIS_STAGED_L1I`
    /// is set (or `set_staged_l1i`), because the predecoded form is unknown:
    /// see "Model choices" in the staged primary instruction-cache section.
    l1i_served: UnsafeCell<bool>,
}

// Safety: the CPU thread is the only accessor, as for every other cache model
// in this crate.
unsafe impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8,
    > Send
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL>
{
}
unsafe impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8,
    > Sync
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL>
{
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8,
    > ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL>
{
    const R10K: bool = model::r10k_family(MODEL);
    const R5K: bool = MODEL == model::R5000;
    /// Index Writeback Invalidate (D) also zeroes the primary data tag on the
    /// R12000 and later; the R10000 leaves it as it was. NEC VR12000 user's
    /// manual, Appendix B.20. The R14000 and R16000 are taken to match.
    const IWBI_CLEARS_PD_TAG: bool = Self::R10K && MODEL != model::R10000;
    const IC_LINES: usize = if IC_LINE == 0 { 0 } else { IC_SIZE / IC_LINE };
    const DC_LINES: usize = if DC_LINE == 0 { 0 } else { DC_SIZE / DC_LINE };
    const L2_LINES: usize = if L2_LINE == 0 { 0 } else { L2_SIZE / L2_LINE };

    pub fn new(downstream: Arc<dyn BusDevice>) -> Self {
        Self {
            downstream,
            llbit: UnsafeCell::new(false),
            lladdr: UnsafeCell::new(0),
            ecc_in: UnsafeCell::new(0),
            ecc_out: UnsafeCell::new(0),
            // The R10000 keeps data arrays for all three caches, for the PROM's
            // Index_Load_Data / Index_Store_Data walks. The R4000 family has no
            // index data ops.
            ic: UnsafeCell::new(Shadow::new(Self::IC_LINES, if Self::R10K { IC_SIZE / 4 } else { 0 })),
            dc: UnsafeCell::new(Shadow::new(Self::DC_LINES, if Self::R10K { DC_SIZE / 4 } else { 0 })),
            l2: UnsafeCell::new(Shadow::new(Self::L2_LINES, if Self::R10K { L2_SIZE / 8 } else { 0 })),
            tc_base: UnsafeCell::new(std::ptr::null_mut()),
            tc_bitmap: UnsafeCell::new(0),
            tc_shift: UnsafeCell::new(crate::ppmem::BITMAP_SHIFT),
            #[cfg(feature = "jitv2")]
            tc_gen: UnsafeCell::new(std::ptr::null_mut()),
            decode: UnsafeCell::new(crate::cpu::decode_cache::DecodeCache::new()),
            pd_tag_hook: UnsafeCell::new(None),
            staged: UnsafeCell::new(Staged { valid: 0, pages: HashMap::new(), changed: false }),
            op_pa: UnsafeCell::new(None),
            l1i_served: UnsafeCell::new(Self::R10K_FAMILY && std::env::var_os("IRIS_STAGED_L1I").is_some()),
        }
    }

    /// ppmem's window base if `phys` is RAM the window maps, as the region
    /// bitmap says; `None` for the PROM, MMIO and the low-512KB alias, which go
    /// through the bus. The bitmap stays zero until the window is wired, so a
    /// set bit also means the base is valid.
    #[inline(always)]
    fn ram(&self, phys: u64) -> Option<*mut u8> {
        if crate::ppmem::bitmap_maps_at(unsafe { *self.tc_bitmap.get() }, unsafe { *self.tc_shift.get() }, phys) {
            Some(unsafe { *self.tc_base.get() })
        } else {
            None
        }
    }

    /// A window store skips `BusDevice`, so it bumps the page's jitv2
    /// generation counter itself, as `PpMemory`'s writes do.
    #[inline(always)]
    fn bump_gen(&self, _phys: u64) {
        #[cfg(feature = "jitv2")]
        {
            let g = unsafe { *self.tc_gen.get() };
            if !g.is_null() {
                unsafe { (*g.add((_phys >> 12) as usize)).fetch_add(1, std::sync::atomic::Ordering::Relaxed) };
            }
        }
    }

    #[allow(clippy::mut_from_ref)]
    fn shadow(&self, sel: u32) -> &mut Shadow {
        unsafe {
            match sel {
                CACH_PI => &mut *self.ic.get(),
                CACH_PD => &mut *self.dc.get(),
                _ => &mut *self.l2.get(),
            }
        }
    }

    /// Tag slot for a CACHE index operation.
    ///
    /// Bit 0 of the index selects the way; the set number starts above the
    /// line offset. Observed directly: the PROM initialises the secondary
    /// cache at `…1000`, `…1001`, `…1080`, `…1081`, stepping by the 128-byte
    /// line with the low bit alternating.
    ///
    /// Folding the way bit away instead — on the reasoning that it sits below
    /// line granularity — made the two ways of a set alias onto one slot, so
    /// a tag written to way 1 overwrote way 0 and the PROM read back the
    /// wrong one. That is the whole of the "TAG walking 1s" failure.
    ///
    /// That is the R10000. The R4400 is direct-mapped throughout: the index
    /// is the set. The R5000's 2-way primaries take the way from VA[14], above
    /// the 16 KB a way spans (VR5000 UM 3.4), as `CpuCache` does.
    fn tag_slot(&self, sel: u32, virt_addr: u64) -> usize {
        let (line, lines) = match sel {
            CACH_PI => (IC_LINE, Self::IC_LINES),
            CACH_PD => (DC_LINE, Self::DC_LINES),
            _ => (L2_LINE, Self::L2_LINES),
        };
        if line == 0 || lines == 0 {
            return 0;
        }
        if !Self::R10K {
            let va = virt_addr as usize;
            if Self::R5K && matches!(sel, CACH_PI | CACH_PD) {
                let sets = lines / 2;
                return ((va >> 14) & 1) * sets + (va / line) % sets;
            }
            return (va / line) % lines;
        }
        let way = (virt_addr as usize) & (WAYS - 1);
        let set = ((virt_addr as usize) / line) % (lines / WAYS).max(1);
        (set * WAYS + way) % lines
    }

    /// Data slot for an R10000 `Index_Load_Data` / `Index_Store_Data`.
    ///
    /// Same shape: way in bit 0, the rest addressing the array, with four-byte
    /// primary and eight-byte secondary slots. This is a model choice, enough
    /// for a data walk to read back what it wrote.
    fn data_slot(&self, sel: u32, len: usize, virt_addr: u64) -> usize {
        if len == 0 {
            return 0;
        }
        let way = (virt_addr as usize) & (WAYS - 1);
        let shift = if matches!(sel, CACH_PI | CACH_PD) { 2 } else { 3 };
        let word = (virt_addr as usize) >> shift;
        (word * WAYS + way) % len
    }

    /// R10000 `Index_Store_Data`, kept out of line: it runs only in a PROM's
    /// diagnostic walk, and inlining it grew `cache_op` and moved the code
    /// laid out after it.
    #[cold]
    #[inline(never)]
    fn index_store_data(&self, sel: u32, virt_addr: u64, phys_addr: u64) {
        let s = self.shadow(sel);
        if s.data.is_empty() {
            return;
        }
        let slot = self.data_slot(sel, s.data.len(), virt_addr);
        // Diagnostic field widths: R10000 UM pp. 262, 271-272.
        let (data_mask, ecc_mask) = if sel == CACH_PI {
            (0xf_ffff_ffff, 1)
        } else if sel == CACH_PD {
            (0xffff_ffff, 0xf)
        } else {
            (u64::MAX, 0x3ff)
        };
        let ecc = unsafe { *self.ecc_in.get() } & ecc_mask;
        s.data[slot] = phys_addr & data_mask;
        s.ecc[slot] = ecc;
        if matches!(sel, CACH_SI | CACH_SD) {
            // Index Store Data (S) writes a quadword: the selected
            // doubleword plus a zero partner, sharing the check bits
            // (R10000 UM p. 91, figure 5-7).
            let partner = slot ^ WAYS;
            s.data[partner] = 0;
            s.ecc[partner] = ecc;
        }
        if l2_op_log(CACHE_LOG_HIT) {
            crate::dlog!(LogModule::L2c, "shadow:   -> ISD slot={slot} data={phys_addr:#018x} ecc={:#x}",
                         s.ecc[slot]);
        }
    }

    // ---- Staged secondary lines ---------------------------------------
    //
    // Everything below is R10000-family only and runs only once a guest has
    // made a secondary tag valid. A cached access that misses the primary
    // caches looks up both secondary ways at the access's index, and a way
    // whose tag holds the access's physical address in a valid state supplies
    // the data. Index Store Data (S) wrote a whole quadword, so the other
    // doubleword of it reads as zero rather than as memory.
    //
    // Not modelled, and loudly absent rather than guessed:
    // - the primary data cache. A line staged there with the primary index
    //   operations is not consulted by loads. (A staged primary instruction
    //   line is: see "Staged primary instruction-cache lines".) A valid
    //   primary instruction line is served before the secondary cache;
    // - replacement. An ordinary cached miss never fills, so never evicts:
    //   a staged line lives until an index or hit operation, or an Index
    //   Store Tag, ends it (`clanker/cpu/staged-secondary-cache-lines.md`);
    // - tag ECC and data check bits. Nothing is checked when a line hits, so
    //   a line staged with bad check bits hits without a Cache Error;
    // - the virtual-index field, TagLo[8:7]. A mismatch with the access's
    //   VA[13:12] purges primary lines at the recorded index, and there are
    //   no primary lines here to purge;
    // - a CACHE operand on an uncached page. It is undefined on the R10000;
    //   the operation goes ahead as if the page were cached;
    // - the BadVAddr, Context and EntryHi values after a CACHE-raised TLB
    //   exception, which the manuals do not give;
    // - coherence. Another CPU's or a DMA device's write to memory a CPU
    //   holds in a staged line does not reach the line, and the line's data
    //   are not visible to them;
    // - the part. Only the R10000 is documented. The R12000, R14000 and
    //   R16000 use the same tag layout and rules here, which is an inference;
    //   every part keeps only the bits of `R10K_L2_TAG_MASK`;
    // - undefined encodings. Operations with op[1:0] = 10 (`CACH_SI`) are
    //   undefined on the R10000 and do nothing.

    /// Are there any staged lines? The test every cached access makes.
    #[inline(always)]
    fn staged_any(&self) -> bool {
        Self::R10K_FAMILY && unsafe { (*self.staged.get()).valid } != 0
    }

    /// The tag index and data-shadow slot of the doubleword at `pa` in
    /// whichever way holds it as a valid line, if one does. Same layout as `data_slot`: a way's
    /// doublewords are every `WAYS`-th slot, indexed by `PA[N-2:3]`.
    fn staged_line(&self, pa: u64) -> Option<(usize, usize)> {
        let l2 = unsafe { &*self.l2.get() };
        if l2.data.is_empty() {
            return None;
        }
        let set = (pa as usize / L2_LINE) % (Self::L2_LINES / WAYS).max(1);
        let want = l2_tag_of(pa);
        (0..WAYS).find_map(|way| {
            let tag = l2.tags[set * WAYS + way];
            (l2_tag_valid(tag) && (tag ^ want) & L2_TAG_PA_MASK == 0)
                .then(|| (set * WAYS + way, (((pa as usize) >> 3) % (l2.data.len() / WAYS)) * WAYS + way))
        })
    }

    /// The data-shadow slot of the doubleword at `pa`, if a staged line holds it.
    fn staged_slot(&self, pa: u64) -> Option<usize> {
        self.staged_line(pa).map(|(_, slot)| slot)
    }

    /// A cached load of `SIZE` bytes at `pa` from a staged line, as the
    /// zero-extended value; `None` if no staged line holds it.
    #[inline(never)]
    fn staged_read<const SIZE: usize>(&self, pa: u64) -> Option<u64> {
        let slot = self.staged_slot(pa)?;
        let m = staged_to_memory(unsafe { (*self.l2.get()).data[slot] });
        let shift = 64 - 8 * SIZE as u32 - 8 * (pa & 7) as u32;
        Some(if SIZE == 8 { m } else { (m >> shift) & ((1u64 << (8 * SIZE)) - 1) })
    }

    /// A cached store of `SIZE` bytes at `pa` into a staged line. The line
    /// is a write-back cache's: the store stays in it and memory is not
    /// written. Returns whether a staged line took the store.
    #[inline(never)]
    fn staged_write<const SIZE: usize>(&self, pa: u64, val: u64) -> bool {
        let shift = 64 - 8 * SIZE as u32 - 8 * (pa & 7) as u32;
        let mask = if SIZE == 8 { u64::MAX } else { ((1u64 << (8 * SIZE)) - 1) << shift };
        self.staged_write_masked(pa, val << shift, mask)
    }

    /// `staged_write` for bytes selected by `mask` of the doubleword at `pa`,
    /// `val` already in memory-doubleword position.
    fn staged_write_masked(&self, pa: u64, val: u64, mask: u64) -> bool {
        let Some((idx, slot)) = self.staged_line(pa) else { return false };
        let l2 = unsafe { &mut *self.l2.get() };
        let m = (staged_to_memory(l2.data[slot]) & !mask) | (val & mask);
        l2.data[slot] = staged_to_memory(m);
        // **Inferred.** A store to a CleanExclusive or Shared secondary block
        // asks the secondary cache to make it DirtyExclusive (NEC
        // U10278EJ4V0UM p.246, performance counter events 14 and 15). The
        // manual does not state the transition, so this takes it as
        // immediate. A later writeback invalidate then sends the stored bytes
        // to memory instead of dropping them.
        l2.tags[idx] |= L2_STATE_DIRTY_EXCLUSIVE << L2_STATE_SHIFT;
        true
    }

    /// A staged line moved from page `old` to page `new` (`None`: it was not,
    /// or is no longer, valid). The first line in a page and the last one
    /// leaving it change which reads of that page the decode cache may serve
    /// from the window, and which translations may be cached.
    fn page_move(&self, old: Option<u64>, new: Option<u64>) {
        let st = unsafe { &mut *self.staged.get() };
        let decode = unsafe { &mut *self.decode.get() };
        if let Some(pfn) = old {
            st.valid -= 1;
            if let Some(n) = st.pages.get_mut(&pfn) {
                *n -= 1;
                if *n == 0 {
                    st.pages.remove(&pfn);
                    st.changed = true;
                    decode.set_bus_only(pfn, false);
                }
            }
        }
        if let Some(pfn) = new {
            st.valid += 1;
            let n = st.pages.entry(pfn).or_insert(0);
            *n += 1;
            if *n == 1 {
                st.changed = true;
                decode.set_bus_only(pfn, true);
            }
        }
    }

    /// Store a secondary tag, keeping the staged-line index in step.
    fn set_l2_tag(&self, idx: usize, new: u64) {
        let l2 = unsafe { &mut *self.l2.get() };
        let old = l2.tags[idx];
        l2.tags[idx] = new;
        let page = |tag: u64| l2_tag_valid(tag).then(|| self.l2_line_pa(tag, idx / WAYS) >> 12);
        if l2_tag_valid(old) || l2_tag_valid(new) {
            self.page_move(page(old), page(new));
        }
    }

    // ---- Staged primary instruction-cache lines -------------------------
    //
    // Index Store Tag (I) and Index Store Data (I) make a primary I line
    // exactly as the secondary pair does, and a cached fetch that finds one
    // takes its word from it before the secondary cache is consulted (NEC
    // U10278EJ4V0UM pp.87-88). The primary index is the virtual address:
    // VA[13:6] picks the set, VA[0] the way, VA[13:2] the word. The tag
    // compares the fetch's physical address. Index Store Data (I) writes one
    // word per operation, so a block takes 16.
    //
    // Model choices, each labelled by how well the sources settle it:
    // - a fetch compares both ways at VA[13:6] against the fetch's translated
    //   PA[39:12]. Stated for Hit Invalidate (I), inferred for a fetch;
    // - a fetch hit does not update LRU. Unstated, and no replacement is
    //   modelled;
    // - the instruction is TagLo[31:0] and TagHi[3:0] is ignored. The
    //   predecoded form is unknown (neither manual defines it). This is the
    //   plain-instruction assumption, and it is WRONG for a guest that stages
    //   words it took from Index Load Data of hardware-predecoded lines: the
    //   Fuel PROM stages such words (36-bit values that are not an
    //   instruction in TagLo, with extra bits inside the word) for its own
    //   code, and executing TagLo from them crashes it while it copies its
    //   image. This model refills and predecodes nothing, so it cannot
    //   reproduce those words. Hence the switch below: serving is off unless
    //   asked for.
    //
    // The primary data cache is not served: its tag and data layouts are
    // known, but a dirty line writes back to the secondary cache and the
    // modifier field has no model here.

    /// Physical address of the line a primary I `tag` at `set` names.
    fn l1i_line_pa(&self, tag: u64, set: usize) -> u64 {
        (((tag >> 8) & 0xff_ffff) << 12)
            | (((tag >> 32) & 0xf) << 36)
            | ((set as u64 * IC_LINE as u64) & 0xfff)
    }

    /// Serve fetches from staged primary I lines, from now on. Set before any
    /// line is staged.
    pub fn set_staged_l1i(&self, on: bool) {
        unsafe { *self.l1i_served.get() = on };
    }

    /// Store a primary I tag, keeping the staged-line index in step.
    fn set_l1i_tag(&self, idx: usize, new: u64) {
        let ic = unsafe { &mut *self.ic.get() };
        let old = ic.tags[idx];
        ic.tags[idx] = new;
        if !unsafe { *self.l1i_served.get() } {
            return;
        }
        let page = |tag: u64| (tag & L1I_VALID != 0).then(|| self.l1i_line_pa(tag, idx / WAYS) >> 12);
        if old & L1I_VALID != 0 || new & L1I_VALID != 0 {
            self.page_move(page(old), page(new));
        }
    }

    /// The tag slot of the valid primary I line holding `pa` at the set `va`
    /// indexes, if any.
    fn staged_l1i_line(&self, va: u64, pa: u64) -> Option<usize> {
        let ic = unsafe { &*self.ic.get() };
        if ic.data.is_empty() {
            return None;
        }
        let set = (va as usize / IC_LINE) % (Self::IC_LINES / WAYS).max(1);
        let want = l1i_tag_of(pa);
        (0..WAYS)
            .map(|way| set * WAYS + way)
            .find(|&idx| ic.tags[idx] & L1I_VALID != 0 && (ic.tags[idx] ^ want) & L1I_TAG_PA_MASK == 0)
    }

    /// The instruction word a fetch at `va`/`pa` finds in a staged primary I
    /// line, then (primary before secondary) in a staged secondary line.
    #[inline(never)]
    fn staged_fetch_word(&self, va: u64, pa: u64) -> Option<u32> {
        if let Some(idx) = self.staged_l1i_line(va, pa).filter(|_| unsafe { *self.l1i_served.get() }) {
            let ic = unsafe { &*self.ic.get() };
            let way = idx % WAYS;
            let slot = (((va as usize) >> 2) % (ic.data.len() / WAYS)) * WAYS + way;
            return Some(ic.data[slot] as u32);
        }
        self.staged_read::<4>(pa).map(|w| w as u32)
    }

    /// Clear the valid bit of the primary I line at `idx`.
    fn l1i_invalidate(&self, idx: usize) {
        let tag = unsafe { (*self.ic.get()).tags[idx] };
        // With serving off the primary tags are the diagnostic round-trip
        // store they always were, and an invalidate leaves them alone.
        if unsafe { *self.l1i_served.get() } && tag & L1I_VALID != 0 {
            self.set_l1i_tag(idx, tag & !L1I_VALID);
        }
    }

    /// Physical address of the line a secondary `tag` at `set` names: the
    /// tag's PA[39:18] and the set's own PA[17:B] below it.
    fn l2_line_pa(&self, tag: u64, set: usize) -> u64 {
        (((tag >> 14) & 0x3_ffff) << 18)
            | (((tag >> 32) & 0xf) << 36)
            | ((set as u64 * L2_LINE as u64) & 0x3_ffff)
    }

    /// Invalidate the secondary line at tag slot `idx`: a DirtyExclusive one
    /// is first written back to the system interface, then its state goes to
    /// Invalid (NEC U12754EJ1V0UMJ1 pp.19, 25, 27). The tag keeps its other
    /// fields. The manual also rewrites the tag with the operand's physical
    /// address and generates its ECC; neither is modelled, because the
    /// shadow stores tags verbatim and has no ECC.
    fn l2_invalidate(&self, idx: usize, writeback: bool) {
        let l2 = unsafe { &*self.l2.get() };
        let tag = l2.tags[idx];
        if !l2_tag_valid(tag) {
            return;
        }
        if writeback && (tag >> L2_STATE_SHIFT) & 3 == L2_STATE_DIRTY_EXCLUSIVE {
            self.l2_write_line_back(idx);
        }
        self.set_l2_tag(idx, tag & !(3 << L2_STATE_SHIFT));
    }

    /// Send the block at tag slot `idx` to the system interface, doubleword by
    /// doubleword in memory order, at the address its tag names.
    fn l2_write_line_back(&self, idx: usize) {
        let l2 = unsafe { &*self.l2.get() };
        if l2.data.is_empty() {
            return;
        }
        let (set, way) = (idx / WAYS, idx % WAYS);
        let base = self.l2_line_pa(l2.tags[idx], set);
        let per_way = l2.data.len() / WAYS;
        for i in 0..(L2_LINE / 8) {
            let slot = ((set * L2_LINE / 8 + i) % per_way) * WAYS + way;
            // A bus error on the way out has nowhere to go.
            let _ = self.downstream.write64(base + 8 * i as u64, staged_to_memory(l2.data[slot]));
        }
    }

    /// Hit Invalidate / Hit Writeback Invalidate (S): act on whichever way
    /// holds `pa`. A miss does nothing. Status.CH is not modelled.
    fn l2_hit_invalidate(&self, pa: u64, writeback: bool) {
        let set = (pa as usize / L2_LINE) % (Self::L2_LINES / WAYS).max(1);
        let want = l2_tag_of(pa);
        for way in 0..WAYS {
            let idx = set * WAYS + way;
            let tag = unsafe { (*self.l2.get()).tags[idx] };
            if l2_tag_valid(tag) && (tag ^ want) & L2_TAG_PA_MASK == 0 {
                self.l2_invalidate(idx, writeback);
            }
        }
    }
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8,
    > From<Arc<dyn BusDevice>>
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL>
{
    fn from(downstream: Arc<dyn BusDevice>) -> Self {
        Self::new(downstream)
    }
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8,
    > CpuModel
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL>
{
    const MIPS4: bool = MIPS4;
    const PRID: u32 = PRID;
    const FIR: u32 = FIR;
    const TLB_ENTRIES: usize = TLB_ENTRIES;
    // The R10000 family implements 44 virtual address bits where the R4x00
    // implements 40.
    const VA_BITS: u32 = if Self::R10K { 44 } else { 40 };
    const PA_BITS: u32 = if Self::R10K { 40 } else { 36 };
    // The CPU's name, which snapshots check: the same as the full model's, so
    // a mismatch reads as a cache-kind difference rather than a different CPU.
    // The R10000 keeps its historical "shadow", which existing IP28
    // snapshots carry.
    const NAME: &'static str = match MODEL {
        model::R4400 => "R4400",
        model::R5000 => "R5000",
        model::R12000 => "R12000",
        model::R16000 => "R16000",
        _ => "shadow",
    };
    const CACHE_KIND: &'static str = "shadow";
    const R10K_FAMILY: bool = Self::R10K_CACHE_OPS;
    // The upper word of a partial LWR is implementation-defined. Keeping it
    // is what the IRIX CPU instruction test (olperi) expects and what
    // qemu-sgi-origami models; for the R12000 and later it is a model choice.
    const PARTIAL_LWR_KEEPS_UPPER: bool = Self::R10K_FAMILY;
    const R10K_CACHE_OPS: bool = Self::R10K;
    const POSTS_UNCACHED_STORES: bool = Self::R10K_CACHE_OPS;
    const SECONDARY_TAG_MASK: u64 = if Self::R10K_CACHE_OPS { R10K_L2_TAG_MASK } else { u64::MAX };
    const CONFIG0_RESET: u32 = if MODEL == model::R16000 { R16000_800_CONFIG0 } else { 0 };
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const MODEL: u8,
    > MipsCache
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, MODEL>
{
    const IC_SIZE: usize = IC_SIZE;
    const IC_LINE: usize = IC_LINE;
    const IC_WAYS: usize = 1;
    const DC_SIZE: usize = DC_SIZE;
    const DC_LINE: usize = DC_LINE;
    const DC_WAYS: usize = 1;
    const L2_SIZE: usize = L2_SIZE;
    const L2_LINE: usize = L2_LINE;

    // ppmem's window, in every build: memory is this cache's only store, and
    // reading RAM through the window is one host load instead of a bus call.
    unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe { *self.tc_base.get() = base };
    }

    fn set_tcache_shift(&self, shift: u32) {
        unsafe {
            *self.tc_shift.get() = shift;
            (*self.decode.get()).set_window_shift(shift);
        }
    }

    fn tcache_bitmap_ptr(&self) -> *mut u64 { self.tc_bitmap.get() }

    fn tcache_base_ptr(&self) -> *mut u8 { unsafe { *self.tc_base.get() } }

    fn set_pd_tag_store_hook(&self, hook: Box<dyn Fn(u32) + Send + Sync>) {
        unsafe { *self.pd_tag_hook.get() = Some(hook) };
    }

    /// Tags only: a fetch reads memory through the decode cache and changes
    /// nothing a guest can see.
    fn fetch_is_pure(&self) -> bool { true }

    fn page_staged(&self, phys: u64) -> bool {
        self.staged_any() && unsafe { (*self.staged.get()).pages.contains_key(&(phys >> 12)) }
    }

    fn take_staged_changed(&self) -> bool {
        Self::R10K_FAMILY && std::mem::take(unsafe { &mut (*self.staged.get()).changed })
    }

    #[cfg(feature = "jitv2")]
    fn tcache_gen_ptr(&self) -> *mut u8 { unsafe { *self.tc_gen.get() as *mut u8 } }

    #[cfg(feature = "jitv2")]
    unsafe fn set_tcache_gen_window(&self, gen_base: *mut std::sync::atomic::AtomicU64) {
        unsafe { *self.tc_gen.get() = gen_base };
    }

    /// Under tcache, a tagless geometry: the inline path is the window alone.
    /// Declined until both windows are published, as `CpuCache` does, so the
    /// emitted code can assume them. Without tcache there is no inline path:
    /// every access calls out, exactly as for `PassthroughCache`.
    fn jit_dc_geometry(&self) -> crate::cpu::mips_cache_v2::JitDcGeometry {
        #[cfg(feature = "jitv2")]
        if !unsafe { *self.tc_base.get() }.is_null() && !unsafe { *self.tc_gen.get() }.is_null() {
            return crate::cpu::mips_cache_v2::JitDcGeometry {
                supported: true,
                tagless: true,
                window_shift: unsafe { *self.tc_shift.get() },
                ..crate::cpu::mips_cache_v2::JitDcGeometry::unsupported()
            };
        }
        crate::cpu::mips_cache_v2::JitDcGeometry::unsupported()
    }

    #[inline(always)]
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        // Decoded instructions by physical page — see `decode_cache`.
        let dc = unsafe { &mut *self.decode.get() };
        // Only a page the decode cache reads word by word (`set_bus_only`)
        // comes here when a line is staged, so this costs a RAM page nothing.
        let word = |a: u64| {
            if self.staged_any() {
                // The word being read is at `a`; its virtual address is the
                // fetch's, moved by the same distance.
                let va = virt_addr.wrapping_add(a.wrapping_sub(phys_addr));
                if let Some(w) = self.staged_fetch_word(va, a) {
                    return Ok(w);
                }
            }
            let r = self.downstream.read32(a);
            if r.is_ok() { Ok(r.data) } else { Err(r.status) }
        };
        let (window, bitmap) = unsafe { (*self.tc_base.get() as *const u8, *self.tc_bitmap.get()) };
        match dc.fetch(phys_addr, window, bitmap, IC_LINE, word) {
            Ok(p) => FetchInstrResult::hit(p),
            Err(status) => FetchInstrResult::exception(status),
        }
    }

    fn read<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64) -> BusRead64 {
        const {
            assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE")
        };
        if self.staged_any() {
            if let Some(v) = self.staged_read::<SIZE>(phys_addr) {
                return BusRead64::ok(v);
            }
        }
        // RAM: one load out of ppmem's window. Its layout is u32 words in
        // native order — byte at `off ^ 3`, halfword at `(off >> 1) ^ 1`,
        // u64 as rotate_left(32) — the same as CpuCache::tc_read.
        if let Some(base) = self.ram(phys_addr) {
            let off = phys_addr as usize;
            let data = unsafe {
                if SIZE == 8 {
                    (*(base.add(off) as *const u64)).rotate_left(32)
                } else if SIZE == 4 {
                    *(base.add(off) as *const u32) as u64
                } else if SIZE == 2 {
                    *((base as *const u16).add((off >> 1) ^ 1)) as u64
                } else {
                    *base.add(off ^ 3) as u64
                }
            };
            return BusRead64::ok(data);
        }
        if SIZE == 1 {
            let r = self.downstream.read8(phys_addr);
            BusRead64 { status: r.status, data: r.data as u64 }
        } else if SIZE == 2 {
            let r = self.downstream.read16(phys_addr);
            BusRead64 { status: r.status, data: r.data as u64 }
        } else if SIZE == 4 {
            let r = self.downstream.read32(phys_addr);
            BusRead64 { status: r.status, data: r.data as u64 }
        } else {
            self.downstream.read64(phys_addr)
        }
    }

    fn write<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        const {
            assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE")
        };
        if self.staged_any() && self.staged_write::<SIZE>(phys_addr, val) {
            return BUS_OK;
        }
        if let Some(base) = self.ram(phys_addr) {
            let off = phys_addr as usize;
            unsafe {
                if SIZE == 8 {
                    *(base.add(off) as *mut u64) = val.rotate_left(32);
                } else if SIZE == 4 {
                    *(base.add(off) as *mut u32) = val as u32;
                } else if SIZE == 2 {
                    *((base as *mut u16).add((off >> 1) ^ 1)) = val as u16;
                } else {
                    *base.add(off ^ 3) = val as u8;
                }
            }
            self.bump_gen(phys_addr);
            return BUS_OK;
        }
        if SIZE == 1 {
            self.downstream.write8(phys_addr, val as u8)
        } else if SIZE == 2 {
            self.downstream.write16(phys_addr, val as u16)
        } else if SIZE == 4 {
            self.downstream.write32(phys_addr, val as u32)
        } else {
            self.downstream.write64(phys_addr, val)
        }
    }

    fn write64_masked(&self, _virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        if self.staged_any() && self.staged_write_masked(phys_addr, val, mask) {
            return BUS_OK;
        }
        if let Some(base) = self.ram(phys_addr) {
            // Compare-exchange, not a read and a store: another CPU's store
            // to the rest of this doubleword must not be lost. No lock would
            // do, because plain stores would not take it.
            let p = unsafe { base.add((phys_addr & !7) as usize) as *mut u64 };
            let (v, m) = (val.rotate_left(32), mask.rotate_left(32));
            let word = unsafe { std::sync::atomic::AtomicU64::from_ptr(p) };
            let _ = word.fetch_update(std::sync::atomic::Ordering::AcqRel, std::sync::atomic::Ordering::Relaxed, |cur| {
                Some((cur & !m) | (v & m))
            });
            self.bump_gen(phys_addr);
            return BUS_OK;
        }
        let aligned = phys_addr & !7;
        let r = self.downstream.read64(aligned);
        if !r.is_ok() {
            return r.status;
        }
        self.downstream.write64(aligned, (r.data & !mask) | (val & mask))
    }

    /// The whole point of the type.
    ///
    /// Every operation that only *moves data between cache and memory* —
    /// invalidate, writeback, fill — is a genuine no-op here, because the
    /// cache and memory can never disagree. The operations that move data
    /// between the cache and a register are the ones with observable effects,
    /// and those are served from the shadow.
    fn cache_op(&self, cache_op: u32, virt_addr: u64, phys_addr: u64) -> u64 {
        let sel = cache_op & 3;
        let op = cache_op & 0x1C;
        let op_pa = unsafe { (*self.op_pa.get()).take() };

        if op == C_IST && sel == CACH_PD {
            if let Some(hook) = unsafe { &*self.pd_tag_hook.get() } {
                hook(phys_addr as u32);
            }
        }

        // Tag operations under `op`, the whole data-array walk only under
        // `hit` as well — see `l2_op_log`.
        if l2_op_log(if matches!(op, C_IST | C_ILT) { CACHE_LOG_OP } else { CACHE_LOG_HIT }) {
            crate::dlog!(LogModule::L2c,
                "shadow: {:<22} raw={cache_op:#04x} va={virt_addr:#018x} arg={phys_addr:#018x}",
                cache_op_name(cache_op));
        }

        // op[1:0] = 10 is not defined for the R10000 (NEC U12754EJ1V0UMJ1
        // p.16). Do nothing, so it cannot stage or end a line.
        if Self::R10K_FAMILY && sel == CACH_SI {
            return 0;
        }
        // The secondary set, way and quadword come from the operand's physical
        // address, the primaries' from its virtual address (NEC U12754EJ1V0UMJ1
        // pp.14-15). A caller that gives no physical address (a unit test, or
        // an operand that was not translated) is indexed by the address given.
        let virt_addr = match op_pa {
            Some(pa) if Self::R10K_FAMILY && sel == CACH_SD => pa,
            _ => virt_addr,
        };

        // R4400/R5000: a tag keeps only its fields, and reads back with
        // generated check bits — exactly what the full model returns.
        if !Self::R10K && matches!(op, C_IST | C_ILT) {
            let lines = match sel { CACH_PI => Self::IC_LINES, CACH_PD => Self::DC_LINES, _ => Self::L2_LINES };
            if lines == 0 {
                return 0; // no secondary on the R5000
            }
            let idx = self.tag_slot(sel, virt_addr);
            let s = self.shadow(sel);
            if op == C_IST {
                s.tags[idx] = r4k_store_taglo(Self::R5K, sel, phys_addr as u32) as u64;
                return 0;
            }
            return r4k_load_taglo(Self::R5K, sel, s.tags[idx] as u32) as u64;
        }

        match op {
            // Index_Store_Tag / Index_Load_Tag. Stored and returned verbatim
            // unless the part sets a secondary mask: a tag test is a round trip,
            // and round trips do not require knowing what the bits mean.
            C_IST => {
                let idx = self.tag_slot(sel, virt_addr);
                let mask = if matches!(sel, CACH_SI | CACH_SD) {
                    <Self as CpuModel>::SECONDARY_TAG_MASK
                } else {
                    u64::MAX
                };
                if l2_op_log(CACHE_LOG_OP) {
                    crate::dlog!(LogModule::L2c, "shadow:   -> IST va={virt_addr:#012x} idx={idx} stores {:#018x}",
                                 phys_addr & mask & !MRU_BIT);
                }
                let s = self.shadow(sel);
                if idx < s.tags.len() {
                    // The MRU bit belongs to the set, not to this way, so it
                    // is lifted out and kept separately. Everything else is
                    // stored verbatim: bit 32 in particular is *not* spare —
                    // TagHi[3:0] are tag address bits 35:32, and clearing it
                    // here destroyed real tag bits.
                    let set = (idx / WAYS).min(s.mru.len() - 1);
                    s.mru[set] = phys_addr & MRU_BIT != 0;
                    let tag = phys_addr & mask & !MRU_BIT;
                    if Self::R10K_FAMILY && sel == CACH_SD {
                        self.set_l2_tag(idx, tag);
                    } else if Self::R10K_FAMILY && sel == CACH_PI {
                        self.set_l1i_tag(idx, tag);
                    } else {
                        s.tags[idx] = tag;
                    }
                }
                0
            }
            C_ILT => {
                let idx = self.tag_slot(sel, virt_addr);
                let s = self.shadow(sel);
                let set = (idx / WAYS).min(s.mru.len() - 1);
                let mut v = if idx < s.tags.len() { s.tags[idx] } else { 0 };
                // The bit belongs to the set: whichever way is read reports
                // it.
                if s.mru[set] {
                    v |= MRU_BIT;
                }
                if l2_op_log(CACHE_LOG_OP) {
                    crate::dlog!(LogModule::L2c, "shadow:   -> ILT va={virt_addr:#012x} idx={idx} set={set} mru={} tag={:#018x}",
                                 s.mru[set], if idx < s.tags.len() { s.tags[idx] } else { 0 });
                    crate::dlog!(LogModule::L2c, "shadow:   -> ILT slot={idx} returns {v:#018x}");
                }
                v
            }

            // R10000 reassigns 5/6/7, scoped to particular cache selects.
            C_R10K_CBARRIER if Self::R10K && sel == CACH_PI => 0,
            C_R10K_ILD if Self::R10K && matches!(sel, CACH_PI | CACH_PD | CACH_SD) => {
                let s = self.shadow(sel);
                let slot = self.data_slot(sel, s.data.len(), virt_addr);
                if s.data.is_empty() {
                    0
                } else {
                    unsafe { *self.ecc_out.get() = s.ecc[slot] };
                    if l2_op_log(CACHE_LOG_HIT) {
                        crate::dlog!(LogModule::L2c, "shadow:   -> ILD slot={slot} data={:#018x} ecc={:#x}",
                                     s.data[slot], s.ecc[slot]);
                    }
                    s.data[slot]
                }
            }
            C_R10K_ISD if Self::R10K => {
                self.index_store_data(sel, virt_addr, phys_addr);
                0
            }

            C_IWBINV if Self::IWBI_CLEARS_PD_TAG && sel == CACH_PD => {
                let idx = self.tag_slot(sel, virt_addr);
                let s = self.shadow(sel);
                if idx < s.tags.len() {
                    s.tags[idx] = 0;
                }
                0
            }
            // The secondary operations that end a staged line. The other
            // primary ones, and every other cache's, fall through: they move data
            // between a cache and memory, and here those never disagree.
            C_IWBINV if Self::R10K_FAMILY && sel == CACH_SD => {
                let idx = self.tag_slot(sel, virt_addr);
                self.l2_invalidate(idx, true);
                0
            }
            // Index Invalidate (I) clears the valid bit of the way named, Hit
            // Invalidate (I) of whichever way holds the address (NEC
            // U12754EJ1V0UMJ1 pp.18, 27).
            C_IINV if Self::R10K_FAMILY && sel == CACH_PI => {
                let idx = self.tag_slot(sel, virt_addr);
                self.l1i_invalidate(idx);
                0
            }
            C_HINV if Self::R10K_FAMILY && sel == CACH_PI => {
                if let Some(idx) = self.staged_l1i_line(virt_addr, phys_addr) {
                    self.l1i_invalidate(idx);
                }
                0
            }
            C_HINV if Self::R10K_FAMILY && sel == CACH_SD => {
                self.l2_hit_invalidate(phys_addr, false);
                0
            }
            C_HWBINV if Self::R10K_FAMILY && sel == CACH_SD => {
                self.l2_hit_invalidate(phys_addr, true);
                0
            }

            // Invalidate, writeback, fill, and the R4000 hit operations. All
            // no-ops: there is nothing held that could be stale or dirty.
            _ => 0,
        }
    }

    /// Power-on: tags read back invalid, and nothing decoded survives — RAM
    /// may be reloaded without a write counter moving.
    fn power_on(&self) {
        for sel in [CACH_PI, CACH_PD, CACH_SD] {
            let s = self.shadow(sel);
            s.tags.fill(0);
            s.mru.fill(false);
        }
        let st = unsafe { &mut *self.staged.get() };
        st.changed |= st.valid != 0;
        st.valid = 0;
        st.pages.clear();
        unsafe { (*self.decode.get()).clear_bus_only() };
        unsafe { (*self.decode.get()).flush() };
    }

    /// The staged secondary lines, which cached accesses read: tag, MRU bit,
    /// and the line's doublewords with their check bits. The rest of the shadow
    /// arrays is for diagnostics, which nothing reads back after a restore, and
    /// is not saved. Empty when nothing is staged.
    fn save_cache_state(&self) -> toml::Value {
        let mut t = toml::value::Table::new();
        if !self.staged_any() {
            return toml::Value::Table(t);
        }
        let l2 = unsafe { &*self.l2.get() };
        let per_way = l2.data.len() / WAYS;
        let u = |v: u64| toml::Value::Array(vec![toml::Value::Integer((v & 0xffff_ffff) as i64), toml::Value::Integer((v >> 32) as i64)]);
        let mut lines = Vec::new();
        for (idx, &tag) in l2.tags.iter().enumerate() {
            if !l2_tag_valid(tag) {
                continue;
            }
            let (set, way) = (idx / WAYS, idx % WAYS);
            let slots = (0..L2_LINE / 8).map(|i| ((set * L2_LINE / 8 + i) % per_way) * WAYS + way);
            let mut e = toml::value::Table::new();
            e.insert("idx".into(), toml::Value::Integer(idx as i64));
            e.insert("tag".into(), u(tag));
            e.insert("mru".into(), toml::Value::Boolean(l2.mru[set.min(l2.mru.len() - 1)]));
            e.insert("data".into(), toml::Value::Array(slots.clone().map(|s| u(l2.data[s])).collect()));
            e.insert("ecc".into(), toml::Value::Array(slots.map(|s| toml::Value::Integer(l2.ecc[s] as i64)).collect()));
            lines.push(toml::Value::Table(e));
        }
        t.insert("staged_l2".into(), toml::Value::Array(lines));
        let ic = unsafe { &*self.ic.get() };
        let per_way = (ic.data.len() / WAYS).max(1);
        let mut ilines = Vec::new();
        for (idx, &tag) in ic.tags.iter().enumerate().filter(|(_, &t)| t & L1I_VALID != 0) {
            let (set, way) = (idx / WAYS, idx % WAYS);
            let words = (0..IC_LINE / 4).map(|i| toml::Value::Integer(ic.data[((set * IC_LINE / 4 + i) % per_way) * WAYS + way] as i64));
            let mut e = toml::value::Table::new();
            e.insert("idx".into(), toml::Value::Integer(idx as i64));
            e.insert("tag".into(), u(tag));
            e.insert("words".into(), toml::Value::Array(words.collect()));
            ilines.push(toml::Value::Table(e));
        }
        t.insert("staged_l1i".into(), toml::Value::Array(ilines));
        toml::Value::Table(t)
    }

    /// A restore replaces RAM wholesale: drop every decoded instruction, end
    /// every staged line and bring back the saved ones.
    fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        let l2 = unsafe { &mut *self.l2.get() };
        if Self::R10K_FAMILY {
            for idx in 0..l2.tags.len() {
                if l2_tag_valid(l2.tags[idx]) {
                    self.set_l2_tag(idx, l2.tags[idx] & !(3 << L2_STATE_SHIFT));
                }
            }
        }
        let ic = unsafe { &mut *self.ic.get() };
        if Self::R10K_FAMILY {
            for idx in 0..ic.tags.len() {
                self.l1i_invalidate(idx);
            }
        }
        let lines = v.get("staged_l2").and_then(|l| l.as_array());
        let u = |v: &toml::Value| -> Result<u64, String> {
            let a = v.as_array().filter(|a| a.len() == 2).ok_or("staged line: bad doubleword")?;
            let w = |x: &toml::Value| x.as_integer().map(|i| i as u64 & 0xffff_ffff).ok_or("staged line: bad word");
            Ok(w(&a[0])? | w(&a[1])? << 32)
        };
        for e in lines.into_iter().flatten().filter(|_| Self::R10K_FAMILY) {
            let idx = e.get("idx").and_then(|i| i.as_integer()).ok_or("staged line: no index")? as usize;
            if idx >= l2.tags.len() || l2.data.is_empty() {
                return Err(format!("staged line index {idx} outside the secondary cache"));
            }
            let (set, way) = (idx / WAYS, idx % WAYS);
            let per_way = l2.data.len() / WAYS;
            let data = e.get("data").and_then(|d| d.as_array()).ok_or("staged line: no data")?;
            let ecc = e.get("ecc").and_then(|d| d.as_array()).ok_or("staged line: no check bits")?;
            if data.len() != L2_LINE / 8 || ecc.len() != data.len() {
                return Err("staged line: wrong size for this cache".into());
            }
            for (i, (d, c)) in data.iter().zip(ecc).enumerate() {
                let slot = ((set * L2_LINE / 8 + i) % per_way) * WAYS + way;
                l2.data[slot] = u(d)?;
                l2.ecc[slot] = c.as_integer().ok_or("staged line: bad check bits")? as u32;
            }
            l2.mru[set.min(l2.mru.len() - 1)] = e.get("mru").and_then(|m| m.as_bool()).unwrap_or(false);
            self.set_l2_tag(idx, u(e.get("tag").ok_or("staged line: no tag")?)?);
        }
        for e in v.get("staged_l1i").and_then(|l| l.as_array()).into_iter().flatten().filter(|_| Self::R10K_FAMILY) {
            let idx = e.get("idx").and_then(|i| i.as_integer()).ok_or("staged line: no index")? as usize;
            let words = e.get("words").and_then(|d| d.as_array()).ok_or("staged line: no words")?;
            if idx >= ic.tags.len() || words.len() != IC_LINE / 4 || ic.data.is_empty() {
                return Err("staged primary line does not fit this cache".into());
            }
            let (set, way) = (idx / WAYS, idx % WAYS);
            let per_way = ic.data.len() / WAYS;
            for (i, w) in words.iter().enumerate() {
                ic.data[((set * IC_LINE / 4 + i) % per_way) * WAYS + way] = w.as_integer().ok_or("staged line: bad word")? as u64;
            }
            self.set_l1i_tag(idx, u(e.get("tag").ok_or("staged line: no tag")?)?);
        }
        unsafe { (*self.decode.get()).flush() };
        Ok(())
    }

    /// Dirty staged lines go to memory, and stay staged.
    fn writeback_all(&self) {
        if !self.staged_any() {
            return;
        }
        let l2 = unsafe { &*self.l2.get() };
        for idx in 0..l2.tags.len() {
            if l2_tag_valid(l2.tags[idx]) && (l2.tags[idx] >> L2_STATE_SHIFT) & 3 == L2_STATE_DIRTY_EXCLUSIVE {
                self.l2_write_line_back(idx);
            }
        }
    }

    /// `l2 probe` in the monitor: how many secondary lines are staged.
    fn debug_probe(&self, _cache_name: &str, _virt_addr: u64, _phys_addr: u64) -> String {
        let st = unsafe { &*self.staged.get() };
        format!("staged secondary lines: {} in {} page(s)", st.valid, st.pages.len())
    }

    fn get_config(&self, cache_target: u32) -> (usize, usize) {
        match cache_target {
            CACH_PI => (IC_SIZE, IC_LINE),
            CACH_PD => (DC_SIZE, DC_LINE),
            _ => (L2_SIZE, L2_LINE),
        }
    }

    /// The check bits ride with cache data on an R10000: a store takes them
    /// from CP0 ECC and a load returns them there.
    fn set_cache_ecc(&self, v: u32) {
        unsafe { *self.ecc_in.get() = v };
    }

    fn cache_op_ecc(&self) -> u32 {
        unsafe { *self.ecc_out.get() }
    }

    fn set_cache_op_pa(&self, pa: u64) {
        unsafe { *self.op_pa.get() = Some(pa) };
    }

    fn downstream(&self) -> Arc<dyn BusDevice> {
        self.downstream.clone()
    }

    fn cas_ram32(&self, phys: u64, old: u32, new: u32) -> Option<bool> {
        use std::sync::atomic::{AtomicU32, Ordering};
        // A staged line is not in RAM: the caller stores the ordinary way.
        if self.staged_any() && self.staged_slot(phys).is_some() {
            return None;
        }
        // Memory beyond the window (the Octane's above 4 GB) is on the bus.
        let Some(base) = self.ram(phys) else {
            return self.downstream.cas32(phys, old, new);
        };
        let word = unsafe { AtomicU32::from_ptr(base.add(phys as usize) as *mut u32) };
        let ok = word.compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed).is_ok();
        if ok {
            self.bump_gen(phys);
        }
        Some(ok)
    }

    fn cas_ram64(&self, phys: u64, old: u64, new: u64) -> Option<bool> {
        use std::sync::atomic::{AtomicU64, Ordering};
        if self.staged_any() && self.staged_slot(phys).is_some() {
            return None;
        }
        let Some(base) = self.ram(phys) else {
            return self.downstream.cas64(phys, old, new);
        };
        let word = unsafe { AtomicU64::from_ptr(base.add(phys as usize) as *mut u64) };
        let ok = word.compare_exchange(old.rotate_left(32), new.rotate_left(32), Ordering::AcqRel, Ordering::Relaxed).is_ok();
        if ok {
            self.bump_gen(phys);
        }
        Some(ok)
    }

    fn check_and_clear_llbit(&self, _phys_addr: u64) {
        unsafe { *self.llbit.get() = false };
    }
    fn get_llbit(&self) -> bool {
        unsafe { *self.llbit.get() }
    }
    fn set_llbit(&self, val: bool) {
        unsafe { *self.llbit.get() = val };
    }
    fn get_lladdr(&self) -> u64 {
        unsafe { *self.lladdr.get() }
    }
    fn set_lladdr(&self, addr: u64) {
        unsafe { *self.lladdr.get() = addr };
    }
}

/// SGI Indigo2 IMPACT R10000 (IP28).
///
/// Real geometry as software reads it — 32 KB primaries with 64-byte
/// instruction lines and 32-byte data lines, a 1 MB secondary with 128-byte
/// lines — 64 TLB entries, MIPS IV, and the R10000 cache operation encodings.
/// Nothing of the microarchitecture: no ways, no LRU, no out-of-order.
/// Reports revision 2.5, matching R10000Cache and enabling the IP28 boot tune.
pub type R10000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 1048576, 128, true, 0x0000_0925, 0x0000_0900, 64, { model::R10000 }>;

/// R12000 3.5 as the Octane has it: the R10000's programming model with a
/// 2 MB secondary cache. PRId 0x0e35, FIR 0x900, 128-byte secondary lines
/// (SB: a model choice).
///
/// The Octane's R14000 is this type with `R14000_PRID` set at runtime, as the
/// Octane's own R10000 revision is; only the PRId differs. Both report
/// "R12000" as `NAME`.
pub type R12000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 128, true, 0x0000_0e35, 0x0000_0900, 64, { model::R12000 }>;

/// PRId of the R14000 2.4 in the Octane, to set on an `R12000ShadowCache`.
/// Implementation 0x0f is shared with the R16000, which is told apart by a
/// major revision above 2.
pub const R14000_PRID: u32 = 0x0000_0f24;

/// R12000 3.5 as the O2 has it: the Octane part with 64-byte secondary lines
/// (SB clear). With 128-byte lines the O2 PROM panics before its banner
/// (write TLB miss at 0x0); 64 boots. IRIX on the O2 never reads SB and sizes
/// the secondary cache as 512 KB << SS.
pub type R12000O2ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 64, true, 0x0000_0e35, 0x0000_0900, 64, { model::R12000 }>;

/// R14000, as the Fuel has it: the Octane's part at revision 2.3 (PRId
/// 0x0f23), the revision a public software inventory of a Fuel prints. The
/// raw CP0 word, the FIR and the 128-byte lines are taken from the Octane
/// part (`R12000ShadowCache` with `R14000_PRID`), not read from a Fuel. Like
/// the Octane's R14000 it reports "R12000" as `NAME`.
pub type R14000FuelShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 128, true, 0x0000_0f23, 0x0000_0900, 64, { model::R12000 }>;

/// R16000 2.2, the 800 MHz part, with a 4 MB secondary cache. The Tezro
/// builds it for both its parts, setting the 700 MHz part's PRId and Config
/// word below on it.
///
/// Values and where they come from:
/// - PRId 0x0f42: the low 16 bits are printed in hinv listings (revision 2.2
///   under the R16000 name; the major nibble is the printed major plus 2).
///   Bits 31:16 are zero by inference from every full PRId read on this family.
/// - FIR 0x900: a model choice. No R16000 FIR has been printed; this is the
///   R12000 and R14000 value, kept for compatibility.
/// - Config reset word 0x009baf85 (`R16000_800_CONFIG0`): derived
///   from the PROM's per-part clock table (printed clocks and cache size fit
///   it), not read from a running part. SS (4 MB) is inferred from the printed
///   size, the clock fields from the printed 800 MHz CPU and 400 MHz cache
///   clocks. 128-byte lines (SB) are a model choice, and the K0, DN, PM, SK,
///   CT, PE, DDR and bit 24 fields are unknown, carried as the table has them.
/// - Cache tap 0xa is printed in hinv listings and is not modelled.
pub type R16000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 4194304, 128, true, 0x0000_0f42, 0x0000_0900, 64, { model::R16000 }>;

/// The Config bits an R16000 part's reset word defines: the low 25. IC and DC
/// above them stay the family's.
pub const CONFIG0_RESET_MASK: u32 = 0x01ff_ffff;

/// Config reset word of `R16000ShadowCache`, the 800 MHz 4 MB part. Same
/// grades as listed there.
pub const R16000_800_CONFIG0: u32 = 0x009b_af85;

/// PRId of the R16000 2.1, the 700 MHz part, to set on an `R16000ShadowCache`.
/// The low 16 bits are printed in hinv listings as revision 2.1; bits 31:16 are
/// zero by inference. Nothing else in the model differs: both parts have a
/// 4 MB secondary cache.
pub const R16000_MHZ700_PRID: u32 = 0x0000_0f41;

/// Config reset word of the 700 MHz part. It differs from the 800 MHz word in
/// the EC field only (6 where the 800 MHz part has 7). Derived from the PROM's
/// per-part clock table with the same grades as the 800 MHz word: the 700 MHz
/// CPU clock and 350 MHz cache clock are printed, the other fields unknown.
/// Cache tap 0xc is printed and not modelled.
pub const R16000_MHZ700_CONFIG0: u32 = 0x009b_ad85;

/// SGI O2 R10000 (IP32): the IP28 model with 64-byte secondary-cache lines.
///
/// The O2 PROM (4.18 and earlier) handles only that line size. With Config's
/// SB bit set (128-byte lines, as on an IP28) its sizing code keeps the
/// secondary cache's *size* as the line size, then aligns every DMA buffer to
/// 1 MB, runs out of heap, and panics on the first SCSI command. So O2 R10000
/// modules have SB clear, and so does this model.
pub type R10000O2ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 1048576, 64, true, 0x0000_0925, 0x0000_0900, 64, { model::R10000 }>;

/// SGI Indy / Indigo2 R4400: the full model's geometry and identity — 16 KB
/// direct-mapped primaries with 16-byte lines, a 1 MB secondary with 128-byte
/// lines, PRId/FIR, 48 TLB entries, MIPS III — so CP0 Config and a PROM's
/// cache sizing see the same part. `shadow_matches_the_full_models` pins it.
pub type R4400ShadowCache =
    ShadowCache<16384, 16, 16384, 16, 1048576, 128, false, 0x0000_0440, 0x0000_0500, 48, { model::R4400 }>;

/// SGI Indy R5000: 2-way 32 KB primaries with 32-byte lines, no secondary,
/// MIPS IV, the full model's PRId/FIR and TLB.
pub type R5000ShadowCache =
    ShadowCache<32768, 32, 32768, 32, 0, 128, true, 0x0000_2321, 0x0000_2300, 48, { model::R5000 }>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::mem::Memory;
    use crate::cpu::mips_cache_v2::{CACH_PD, CACH_SD};

    fn cache() -> R10000ShadowCache {
        let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1)); // MB
        R10000ShadowCache::from(mem)
    }

    /// A tag test is a round trip, so the shadow must return exactly the bits
    /// it was given, except the reserved ones the manual reads as zero. The cache
    /// model that preceded this one decoded TagLo into ptag/state/pidx fields
    /// and re-encoded on the way out, which silently dropped the low seven
    /// bits and every state code it did not recognise. A walking-1s test then
    /// failed on its very first bit.
    #[test]
    fn tags_round_trip_every_defined_bit() {
        let c = cache();
        for bit in 0..64 {
            // 32 and 63 are MRU control/state, not tag storage.
            if bit == 32 || bit == 63 { continue; }
            let v = 1u64 << bit;
            c.cache_op(C_IST | CACH_SD, 0, v);
            let kept = <R10000ShadowCache as CpuModel>::SECONDARY_TAG_MASK & v;
            assert_eq!(
                c.cache_op(C_ILT | CACH_SD, 0, 0),
                kept,
                "bit {bit} did not survive a store/load tag round trip"
            );
        }
    }

    /// Index_Store_Data / Index_Load_Data likewise, in the secondary and the
    /// primary caches: a PROM's data walk round-trips.
    #[test]
    fn secondary_data_round_trips() {
        let c = cache();
        c.cache_op(C_R10K_ISD | CACH_SD, 0x40, 0xdeadbeef);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, 0x40, 0), 0xdeadbeef);

        c.cache_op(C_R10K_ISD | CACH_PI, 4, 0x1234);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, 4, 0), 0x1234);
        c.cache_op(C_R10K_ISD | CACH_PD, 8, 0x5678);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PD, 8, 0), 0x5678);
    }

    /// The MRU bit is **shared between the ways of a set**: written through
    /// one way, it must read back through the other. The IP28 PROM's
    /// diagnostic does exactly this and nothing else satisfies it — a model
    /// that records which way was marked reports nothing on the way that is
    /// actually read.
    #[test]
    fn the_mru_bit_is_shared_between_the_ways_of_a_set() {
        let c = cache();
        // Way 0 of set 0, then way 1 of set 0.
        c.cache_op(C_IST | CACH_SD, 0, MRU_BIT);
        assert_ne!(
            c.cache_op(C_ILT | CACH_SD, 1, 0) & MRU_BIT,
            0,
            "written through way 0, must be visible through way 1"
        );
        // And the other direction.
        c.cache_op(C_IST | CACH_SD, 1, MRU_BIT);
        assert_ne!(c.cache_op(C_ILT | CACH_SD, 0, 0) & MRU_BIT, 0);
    }

    /// Writing a tag without the bit clears it, again for the whole set.
    #[test]
    fn writing_a_tag_without_the_mru_bit_clears_it() {
        let c = cache();
        c.cache_op(C_IST | CACH_SD, 0, MRU_BIT);
        c.cache_op(C_IST | CACH_SD, 0, 0);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 1, 0) & MRU_BIT, 0);
    }

    /// Marking one set must not mark its neighbour. The PROM writes the
    /// following set specifically to clobber the signal line between the two
    /// checks, so a model where the bit leaks across sets passes the first
    /// check and fails the next.
    #[test]
    fn the_mru_bit_does_not_leak_between_sets() {
        let c = cache();
        let next_set = 128; // one line
        c.cache_op(C_IST | CACH_SD, 0, MRU_BIT);
        c.cache_op(C_IST | CACH_SD, next_set, 0);
        assert_ne!(c.cache_op(C_ILT | CACH_SD, 1, 0) & MRU_BIT, 0, "set 0 keeps its bit");
        assert_eq!(c.cache_op(C_ILT | CACH_SD, next_set, 0) & MRU_BIT, 0, "set 1 has none");
    }

    /// Check bits are stored with the data and returned on load. The PROM's
    /// ECC walk writes a different value through each way and reads them
    /// back, so both the round trip and the separation matter.
    ///
    /// This exists because the methods carrying ECC were once declared on the
    /// trait with defaults and simply never implemented here: the data
    /// round-tripped perfectly and every check bit read back as zero.
    #[test]
    fn check_bits_travel_with_the_data() {
        let c = cache();
        c.set_cache_ecc(0x001);
        c.cache_op(C_R10K_ISD | CACH_SD, 0, 0x1111_2222_3333_4444);
        c.set_cache_ecc(0x3fe);
        c.cache_op(C_R10K_ISD | CACH_SD, 1, 0xaaaa_bbbb_cccc_dddd);

        c.set_cache_ecc(0);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, 0, 0), 0x1111_2222_3333_4444);
        assert_eq!(c.cache_op_ecc(), 0x001, "way 0's check bits");
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, 1, 0), 0xaaaa_bbbb_cccc_dddd);
        assert_eq!(c.cache_op_ecc(), 0x3fe, "way 1's check bits");
    }

    /// The NEC layout: ten check bits, and a secondary store writes a
    /// quadword whose other doubleword is zero and shares the check bits.
    #[test]
    fn nec_layout_stores_a_quadword_with_ten_check_bits() {
        let c = cache();
        c.set_cache_ecc(0);
        c.cache_op(C_R10K_ISD | CACH_SD, 8, 0xdead_beef);
        c.set_cache_ecc(0xffff_fc01);
        c.cache_op(C_R10K_ISD | CACH_SD, 0, 0x1111_2222_3333_4444);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, 0, 0), 0x1111_2222_3333_4444);
        assert_eq!(c.cache_op_ecc(), 0x001, "ten check bits");
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, 8, 0), 0, "zero partner");
        assert_eq!(c.cache_op_ecc(), 0x001, "shared check bits");
    }

    /// An untouched set reports no MRU, so an ordinary tag read is not
    /// contaminated by it.
    #[test]
    fn an_untouched_set_has_no_mru_bit() {
        let c = cache();
        // Only defined tag bits (TagLo bits 9, 12 and 13 read as zero).
        c.cache_op(C_IST | CACH_SD, 0, 0xdead_8cef);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 0, 0), 0xdead_8cef);
    }

    /// Separate lines must not alias onto one another.
    #[test]
    fn distinct_lines_hold_distinct_tags() {
        let c = cache();
        c.cache_op(C_IST | CACH_SD, 0, 0x1111_0111);
        c.cache_op(C_IST | CACH_SD, 128, 0x2222_0022);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 0, 0), 0x1111_0111);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 128, 0), 0x2222_0022);
    }

    /// Bit 0 of a CACHE index selects the **way**, and the two ways of a set
    /// must not alias.
    ///
    /// This is exactly the IP28 PROM's secondary-cache tag test: it stores one
    /// tag to way 0 and a different one to way 1 of the same set, then reads
    /// way 0 back. Folding the way bit away let the second store clobber the
    /// first, and the PROM reported
    /// `Expected: 0x0000000000000001 ... TAG walking 1s`.
    #[test]
    fn the_two_ways_of_a_set_are_independent() {
        let c = cache();
        c.cache_op(C_IST | CACH_SD, 0x2000_0000, 0x0000_0001);
        c.cache_op(C_IST | CACH_SD, 0x2000_0001, 0xffff_cdfe);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 0x2000_0000, 0), 0x0000_0001,
                   "way 1's tag overwrote way 0's");
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 0x2000_0001, 0), 0xffff_cdfe);
    }

    /// Adjacent sets stay distinct once the way bit is accounted for.
    #[test]
    fn adjacent_sets_do_not_alias_through_the_way_bit() {
        let c = cache();
        for set in 0..4u64 {
            for way in 0..2u64 {
                let va = set * 128 + way;
                c.cache_op(C_IST | CACH_SD, va, 0x4000 + set * 16 + way);
            }
        }
        for set in 0..4u64 {
            for way in 0..2u64 {
                let va = set * 128 + way;
                assert_eq!(c.cache_op(C_ILT | CACH_SD, va, 0),
                           0x4000 + set * 16 + way,
                           "set {set} way {way} aliased");
            }
        }
    }

    /// Nothing is ever held, so a load must see the store that preceded it
    /// with no flush in between. This is the property that makes the design
    /// safe, not merely fast.
    #[test]
    fn memory_is_the_only_store() {
        let c = cache();
        c.write::<4>(0, 0x2000, 0x1234_5678);
        assert_eq!(c.read::<4>(0, 0x2000).data, 0x1234_5678);
        // An invalidate cannot lose it, and a writeback cannot be needed.
        c.cache_op(crate::cpu::mips_cache_v2::C_IINV | CACH_PD, 0x2000, 0);
        assert_eq!(c.read::<4>(0, 0x2000).data, 0x1234_5678);
    }

    /// On the R12000 and later Index Writeback Invalidate (D) zeroes the
    /// line's primary data tag, which Index Load Tag then reads back; the
    /// R10000 keeps the tag.
    #[test]
    fn index_writeback_invalidate_d_clears_the_tag_from_the_r12000_on() {
        fn run<C: MipsCache + From<Arc<dyn BusDevice>>>() -> u64 {
            let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
            let c = C::from(mem);
            let tag = 0x0000_0001_2345_6780u64;
            c.cache_op(crate::cpu::mips_cache_v2::C_IST | CACH_PD, 0x40, tag);
            assert_eq!(c.cache_op(crate::cpu::mips_cache_v2::C_ILT | CACH_PD, 0x40, 0), tag);
            c.cache_op(crate::cpu::mips_cache_v2::C_IWBINV | CACH_PD, 0x40, 0);
            c.cache_op(crate::cpu::mips_cache_v2::C_ILT | CACH_PD, 0x40, 0)
        }
        assert_eq!(run::<R12000ShadowCache>(), 0);
        assert_eq!(run::<R12000O2ShadowCache>(), 0);
        assert_eq!(run::<R10000ShadowCache>(), 0x0000_0001_2345_6780);
    }

    /// Geometry is what CP0 Config is built from, so it must be the real
    /// part's even though the model holds nothing.
    #[test]
    fn reported_geometry_is_the_real_parts() {
        let c = cache();
        assert_eq!(c.get_config(CACH_PI), (32768, 64));
        assert_eq!(c.get_config(CACH_PD), (32768, 32));
        assert_eq!(c.get_config(CACH_SD), (1048576, 128));
    }

    /// The R4400 and R5000 shadows stand in for the full models, so everything
    /// software can read — CP0 Config geometry, PRId/FIR, ISA level, TLB size,
    /// the name snapshots check — must be the full model's.
    #[test]
    fn shadow_matches_the_full_models() {
        use crate::cpu::mips_cache_v2::{R4400Cache, R5000Cache};
        fn same<S: CpuModel, F: CpuModel>() {
            assert_eq!((S::IC_SIZE, S::IC_LINE, S::DC_SIZE, S::DC_LINE, S::L2_SIZE, S::L2_LINE),
                       (F::IC_SIZE, F::IC_LINE, F::DC_SIZE, F::DC_LINE, F::L2_SIZE, F::L2_LINE), "geometry");
            assert_eq!((S::PRID, S::FIR, S::MIPS4, S::TLB_ENTRIES, S::VA_BITS, S::R10K_CACHE_OPS),
                       (F::PRID, F::FIR, F::MIPS4, F::TLB_ENTRIES, F::VA_BITS, F::R10K_CACHE_OPS), "identity");
            assert_eq!(S::NAME, F::NAME, "snapshot name");
            assert_eq!((S::CACHE_KIND, F::CACHE_KIND), ("shadow", "full"));
        }
        same::<R4400ShadowCache, R4400Cache>();
        same::<R5000ShadowCache, R5000Cache>();
        assert_eq!(<R10000ShadowCache as CpuModel>::NAME, "shadow", "existing IP28 snapshots");
    }

    /// The R10000 returns zero for the reserved bits of a secondary tag and
    /// keeps MRU, tag, state, index and ECC (R10000 UM p. 270); the R4400
    /// stores the tag as written.
    #[test]
    fn the_r10000_masks_secondary_tags() {
        fn stored<C: CpuModel + MipsCache + From<Arc<dyn BusDevice>>>(tag: u64) -> u64 {
            let c = C::from(Arc::new(Memory::new(1)) as Arc<dyn BusDevice>);
            c.cache_op(C_IST | CACH_SD, 0, tag);
            c.cache_op(C_ILT | CACH_SD, 0, 0)
        }
        assert_eq!(stored::<R10000ShadowCache>(u64::MAX), 0x8000_000f_ffff_cdff);
        assert_eq!(stored::<R10000O2ShadowCache>(u64::MAX), 0x8000_000f_ffff_cdff);
        assert_eq!(stored::<R4400ShadowCache>(0x3000) & 0x3000, 0x3000, "R4400 keeps TagLo bits 12 and 13");
    }

    /// The O2 R10000 differs from the IP28's only in its secondary line size.
    #[test]
    fn the_o2_r10000_is_the_ip28_part_with_shorter_secondary_lines() {
        type O2 = R10000O2ShadowCache;
        type Ip28 = R10000ShadowCache;
        assert_eq!(<O2 as MipsCache>::L2_LINE, 64);
        assert_eq!((<O2 as MipsCache>::IC_SIZE, <O2 as MipsCache>::IC_LINE, <O2 as MipsCache>::DC_SIZE, <O2 as MipsCache>::DC_LINE, <O2 as MipsCache>::L2_SIZE),
                   (<Ip28 as MipsCache>::IC_SIZE, <Ip28 as MipsCache>::IC_LINE, <Ip28 as MipsCache>::DC_SIZE, <Ip28 as MipsCache>::DC_LINE, <Ip28 as MipsCache>::L2_SIZE));
        assert_eq!((<O2 as CpuModel>::PRID, <O2 as CpuModel>::FIR, <O2 as CpuModel>::MIPS4, <O2 as CpuModel>::TLB_ENTRIES, <O2 as CpuModel>::VA_BITS, <O2 as CpuModel>::R10K_CACHE_OPS),
                   (<Ip28 as CpuModel>::PRID, <Ip28 as CpuModel>::FIR, <Ip28 as CpuModel>::MIPS4, <Ip28 as CpuModel>::TLB_ENTRIES, <Ip28 as CpuModel>::VA_BITS, <Ip28 as CpuModel>::R10K_CACHE_OPS));
    }

    /// Each R10000-family part reports its own identity and geometry and shares
    /// the family's address widths and cache operations; the R5000 does not.
    #[test]
    fn the_r10000_family_parts_report_their_own_identity() {
        fn id<C: CpuModel + MipsCache>() -> (u32, usize, usize, &'static str, u32, u32, bool) {
            (C::PRID, C::L2_SIZE, C::L2_LINE, C::NAME, C::VA_BITS, C::PA_BITS, C::R10K_CACHE_OPS)
        }
        assert_eq!(id::<R10000ShadowCache>(), (0x925, 1 << 20, 128, "shadow", 44, 40, true));
        assert_eq!(id::<R12000ShadowCache>(), (0xe35, 2 << 20, 128, "R12000", 44, 40, true));
        assert_eq!(id::<R12000O2ShadowCache>(), (0xe35, 2 << 20, 64, "R12000", 44, 40, true));
        assert_eq!(id::<R16000ShadowCache>(), (0xf42, 4 << 20, 128, "R16000", 44, 40, true));
        assert_eq!(id::<R5000ShadowCache>().6, false);
        // The 700 MHz part is the same type with a PRId set at build time, as
        // the Octane's R14000 is the R12000 type.
        assert_eq!(R16000_MHZ700_PRID, 0xf41);
        assert_eq!(R16000_MHZ700_PRID >> 4 & 0xf, 4, "major nibble is printed major 2 plus 2");
    }

    /// The tag store hook sees primary data tag stores and nothing else.
    #[test]
    fn the_tag_store_hook_sees_only_primary_data_tag_stores() {
        let c = cache();
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let s = seen.clone();
        c.set_pd_tag_store_hook(Box::new(move |tag| s.lock().unwrap().push(tag)));
        c.cache_op(C_IST | CACH_PI, 0x8000_0000, 0x11);
        c.cache_op(C_IST | CACH_SD, 0x8000_0000, 0x22);
        c.cache_op(C_ILT | CACH_PD, 0x8000_0000, 0x33);
        c.cache_op(C_IST | CACH_PD, 0x8000_0000, 0xc8);
        assert_eq!(*seen.lock().unwrap(), vec![0xc8]);
    }

    /// Index_Store_Tag then Index_Load_Tag returns exactly what the full model
    /// returns, for random TagLo words at random indexes: the shadow must not
    /// be observably different to a PROM, IDE or IRIX reading tags. Both
    /// caches start with every tag invalid, so a store followed by a load is
    /// the whole of what software can see.
    #[test]
    fn tags_read_back_as_the_full_model_does() {
        use crate::cpu::mips_cache_v2::{R4400Cache, R5000Cache};
        fn check<S: MipsCache + From<Arc<dyn BusDevice>>, F: MipsCache + From<Arc<dyn BusDevice>>>(
            name: &str, sels: &[u32]) {
            let mem = || -> Arc<dyn BusDevice> { Arc::new(Memory::new(1)) };
            let (shadow, full) = (S::from(mem()), F::from(mem()));
            let mut x = 0x9e37_79b9_7f4a_7c15u64;
            for _ in 0..20_000 {
                x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                let sel = sels[(x >> 60) as usize % sels.len()];
                let index = 0x8000_0000 | (x >> 20 & 0xF_FFF8);
                let taglo = x as u32 as u64;
                shadow.cache_op(C_IST | sel, index, taglo);
                full.cache_op(C_IST | sel, index, taglo);
                let probe = 0x8000_0000 | (x >> 40 & 0xF_FFF8);
                assert_eq!(shadow.cache_op(C_ILT | sel, probe, probe), full.cache_op(C_ILT | sel, probe, probe),
                           "{name}: sel {sel} TagLo {taglo:#x} stored at {index:#x}, loaded at {probe:#x}");
            }
        }
        check::<R4400ShadowCache, R4400Cache>("R4400", &[CACH_PI, CACH_PD, CACH_SD]);
        check::<R5000ShadowCache, R5000Cache>("R5000", &[CACH_PI, CACH_PD]);
    }

    // ---- Index-operation staging ---------------------------------------

    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;
    use crate::cpu::mips_cache_v2::{C_HINV, C_HWBINV, C_IWBINV};

    /// Memory that reads 0xee everywhere, counts reads and records writes, so
    /// a staged value can never be mistaken for it and a bus access shows.
    #[derive(Default)]
    struct Bus {
        reads: AtomicU32,
        writes: Mutex<Vec<(u64, u64)>>,
    }

    impl BusDevice for Bus {
        fn read8(&self, _: u64) -> crate::traits::BusRead8 {
            self.reads.fetch_add(1, Ordering::Relaxed);
            crate::traits::BusRead8::ok(0xee)
        }
        fn read16(&self, _: u64) -> crate::traits::BusRead16 {
            self.reads.fetch_add(1, Ordering::Relaxed);
            crate::traits::BusRead16::ok(0xeeee)
        }
        fn read32(&self, _: u64) -> crate::traits::BusRead32 {
            self.reads.fetch_add(1, Ordering::Relaxed);
            crate::traits::BusRead32::ok(0xeeee_eeee)
        }
        fn read64(&self, _: u64) -> BusRead64 {
            self.reads.fetch_add(1, Ordering::Relaxed);
            BusRead64::ok(0xeeee_eeee_eeee_eeee)
        }
        fn write8(&self, a: u64, v: u8) -> u32 { self.writes.lock().unwrap().push((a, v as u64)); BUS_OK }
        fn write16(&self, a: u64, v: u16) -> u32 { self.writes.lock().unwrap().push((a, v as u64)); BUS_OK }
        fn write32(&self, a: u64, v: u32) -> u32 { self.writes.lock().unwrap().push((a, v as u64)); BUS_OK }
        fn write64(&self, a: u64, v: u64) -> u32 { self.writes.lock().unwrap().push((a, v)); BUS_OK }
    }

    /// An R10000 secondary cache (1 MiB, 128-byte lines, 4096 sets) over `Bus`.
    fn staged_cache() -> (R10000ShadowCache, Arc<Bus>) {
        let bus = Arc::new(Bus::default());
        let c = R10000ShadowCache::from(bus.clone() as Arc<dyn BusDevice>);
        c.set_staged_l1i(true);
        (c, bus)
    }

    /// A line in set 0x20 (PA[18:7]), tag PA[35:18] = 4.
    const PA: u64 = 0x0010_1000;
    const MEM: u64 = 0xeeee_eeee_eeee_eeee;

    /// The operand VA of an index operation on `pa`'s set, `way` in bit 0.
    fn operand(pa: u64, way: u64) -> u64 {
        0xffff_ffff_8000_0000 | (pa & 0x1fff_fff8) | way
    }

    /// Index Store Tag (S): the line of `pa`, in `way`, with `state`.
    fn stage_tag(c: &R10000ShadowCache, pa: u64, way: u64, state: u64) {
        c.cache_op(C_IST | CACH_SD, operand(pa & !127, way), l2_tag_of(pa) | state << L2_STATE_SHIFT);
    }

    /// Index Store Data (S): the doubleword of `pa` becomes TagHi:TagLo = `d`.
    fn stage_data(c: &R10000ShadowCache, pa: u64, way: u64, d: u64) {
        c.cache_op(C_R10K_ISD | CACH_SD, operand(pa & !7, way), d);
    }

    fn load<const S: usize>(c: &R10000ShadowCache, pa: u64) -> u64 {
        let r = c.read::<S>(operand(pa, 0), pa);
        assert!(r.is_ok());
        r.data
    }

    /// Encoding: the 5-bit op field is `op[4:2]` over `cache[1:0]`, and
    /// the secondary operations are the ones the manual lists. ISUM pp.70-72.
    #[test]
    fn the_secondary_index_and_hit_operations_have_the_listed_encodings() {
        assert_eq!(C_IWBINV | CACH_SD, 0x03);
        assert_eq!(C_ILT | CACH_SD, 0x07);
        assert_eq!(C_IST | CACH_SD, 0x0b);
        assert_eq!(C_HINV | CACH_SD, 0x13);
        assert_eq!(C_HWBINV | CACH_SD, 0x17);
        assert_eq!(C_R10K_ILD | CACH_SD, 0x1b);
        assert_eq!(C_R10K_ISD | CACH_SD, 0x1f);
        // Primary data and instruction, for the neighbours a decoder mixes up.
        assert_eq!((C_IST | CACH_PD, C_R10K_ISD | CACH_PD), (0x09, 0x1d));
        assert_eq!((C_IST | CACH_PI, C_R10K_ISD | CACH_PI), (0x08, 0x1c));
        assert_eq!(cache_op_name(0x0b), "Index_Store_Tag(SD)");
    }

    /// The whole thing through the CACHE instruction (major opcode 0x2f, op in
    /// rt, base register and offset) and the interpreter's cached load: TagLo
    /// and TagHi are what the operations store, and `lw` hits the line.
    #[test]
    fn cache_instructions_stage_a_line_a_cached_load_hits() {
        use crate::cpu::mips_exec::{MipsCpuConfig, MipsExecutor, EXEC_COMPLETE};
        use crate::cpu::mips_isa::{OP_CACHE, OP_LW};
        use crate::cpu::mips_tlb::PassthroughTlb;
        let bus = Arc::new(Bus::default());
        let mut exec = MipsExecutor::<PassthroughTlb, R10000ShadowCache>::new(
            bus.clone() as Arc<dyn BusDevice>, PassthroughTlb::default(),
            &MipsCpuConfig::for_model::<R10000ShadowCache>(Arc::new(crate::ip22_bus::memory_map(false))));
        #[cfg(feature = "jitv2")]
        { exec.jitv2_dispatch_enabled = false; }
        exec.core.cp0_config = (exec.core.cp0_config & !7) | 3; // KSEG0: cacheable
        let i = |op: u32, rs: u32, rt: u32, imm: u32| (op << 26) | (rs << 21) | (rt << 16) | imm;
        exec.core.gpr[4] = operand(PA, 0);
        let tag = l2_tag_of(PA) | 2 << L2_STATE_SHIFT;
        exec.core.cp0_taglo = tag as u32 as u64;
        exec.core.cp0_taghi = (tag >> 32) as u32;
        assert_eq!(exec.exec(i(OP_CACHE, 4, 0x0b, 0)), EXEC_COMPLETE);
        exec.core.cp0_taglo = 0x1111_2222u64; // word at the lower address (inferred)
        exec.core.cp0_taghi = 0x3333_4444;
        assert_eq!(exec.exec(i(OP_CACHE, 4, 0x1f, 0)), EXEC_COMPLETE);
        assert_eq!(exec.exec(i(OP_LW, 4, 5, 0)), EXEC_COMPLETE);
        assert_eq!(exec.core.gpr[5], 0x1111_2222);
        assert_eq!(exec.exec(i(OP_LW, 4, 6, 4)), EXEC_COMPLETE);
        assert_eq!(exec.core.gpr[6], 0x3333_4444);
        assert_eq!(bus.reads.load(Ordering::Relaxed), 0, "no system-bus read");
    }

    /// Way select (ISUM pp.71-72): address bit 0 picks the way for every
    /// index operation, and the two ways of a set hold different lines.
    #[test]
    fn the_way_comes_from_address_bit_0() {
        let (c, bus) = staged_cache();
        let other = PA + (1 << 19); // same set, tag PA[35:18] differs
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 0xaaaa_aaaa_0000_0001);
        stage_tag(&c, other, 1, 2);
        stage_data(&c, other, 1, 0xbbbb_bbbb_0000_0002);
        assert_eq!(load::<8>(&c, PA).rotate_left(32), 0xaaaa_aaaa_0000_0001);
        assert_eq!(load::<8>(&c, other).rotate_left(32), 0xbbbb_bbbb_0000_0002);
        assert_eq!(bus.reads.load(Ordering::Relaxed), 0);
        // The tags are read back from the way that was addressed.
        let tag = |way| c.cache_op(C_ILT | CACH_SD, operand(PA, way), 0);
        assert_eq!(tag(0), l2_tag_of(PA) | 2 << L2_STATE_SHIFT);
        assert_eq!(tag(1), l2_tag_of(other) | 2 << L2_STATE_SHIFT);
        // Ending way 1 leaves way 0.
        stage_tag(&c, other, 1, 0);
        assert_eq!(load::<8>(&c, other), MEM);
        assert_ne!(load::<8>(&c, PA), MEM);
    }

    /// Tag state (UM p.84): Shared, CleanExclusive and DirtyExclusive are
    /// valid; Invalid misses.
    #[test]
    fn every_state_but_invalid_is_valid() {
        for (state, hits) in [(0, false), (1, true), (2, true), (3, true)] {
            let (c, bus) = staged_cache();
            stage_tag(&c, PA, 0, state);
            stage_data(&c, PA, 0, 0x1234_5678_9abc_def0);
            let got = load::<8>(&c, PA);
            assert_eq!(got != MEM, hits, "state {state}");
            assert_eq!(bus.reads.load(Ordering::Relaxed), if hits { 0 } else { 1 }, "state {state}");
        }
    }

    /// The hit after Index Store Tag + Index Store Data: the staged data, at
    /// every access width, with no bus read; the other doubleword of the
    /// quadword reads zero (UM p.91, ISUM p.30), not memory.
    #[test]
    fn a_staged_line_hits_with_its_data_and_no_bus_read() {
        let (c, bus) = staged_cache();
        stage_tag(&c, PA, 0, 2);
        // The word at the lower address is TagLo: the inferred order, below.
        stage_data(&c, PA + 8, 0, 0x0102_0304_a1a2_a3a4);
        let m = 0xa1a2_a3a4_0102_0304u64; // as `ld` at PA+8 sees it
        assert_eq!(load::<8>(&c, PA + 8), m);
        assert_eq!(load::<4>(&c, PA + 8), m >> 32);
        assert_eq!(load::<4>(&c, PA + 12), m & 0xffff_ffff);
        assert_eq!(load::<2>(&c, PA + 10), (m >> 32) & 0xffff);
        assert_eq!(load::<1>(&c, PA + 11), (m >> 32) & 0xff);
        assert_eq!(load::<1>(&c, PA + 15), m & 0xff);
        assert_eq!(load::<8>(&c, PA), 0, "the quadword's other doubleword is zero");
        assert_eq!(bus.reads.load(Ordering::Relaxed), 0);
        // Two Index Store Data to one quadword leave only the second.
        stage_data(&c, PA, 0, 0x5555_5555_6666_6666);
        assert_eq!(load::<8>(&c, PA + 8), 0);
        assert_ne!(load::<8>(&c, PA), 0);
    }

    /// A miss: the tag does not match, the state is invalid, or the access
    /// is in another set. The access reads the bus and sees memory.
    #[test]
    fn a_tag_that_does_not_match_misses() {
        let (c, bus) = staged_cache();
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 0x1234_5678_9abc_def0);
        for bit in 14..32 {
            // TagLo[31:14] is PA[35:18]: a flipped bit is another address.
            let pa = PA ^ (1 << (bit + 4));
            let before = bus.reads.load(Ordering::Relaxed);
            assert_eq!(load::<8>(&c, pa), MEM, "PA bit {}", bit + 4);
            assert_eq!(bus.reads.load(Ordering::Relaxed), before + 1);
        }
        // TagHi[3:0] is PA[39:36].
        let before = bus.reads.load(Ordering::Relaxed);
        assert_eq!(load::<8>(&c, PA | 1 << 36), MEM);
        assert_eq!(bus.reads.load(Ordering::Relaxed), before + 1);
        // Another set, same tag.
        assert_eq!(load::<8>(&c, PA + 128), MEM);
        // The line itself still hits.
        assert_ne!(load::<8>(&c, PA), MEM);
    }

    /// **Inferred.** Staging TagHi = W1 and TagLo = W0 into one doubleword:
    /// a word load at its lower address returns W0 (TagLo lands at the lower
    /// address), the higher address W1, and a doubleword load has W0 on top.
    /// The manual-literal order would return W1 first. Index Load Data
    /// returns the registers as stored under either order.
    #[test]
    fn the_inferred_refill_order_puts_taglo_at_the_lower_address() {
        let (c, _) = staged_cache();
        let (w0, w1) = (0x1111_1111u64, 0x2222_2222u64);
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, w1 << 32 | w0);
        assert!(STAGED_TAGLO_AT_LOWER_ADDRESS, "the inferred order is the one modelled");
        assert_eq!(load::<4>(&c, PA), w0, "lower address: TagLo");
        assert_eq!(load::<4>(&c, PA + 4), w1, "higher address: TagHi");
        assert_eq!(load::<8>(&c, PA), w0 << 32 | w1);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, operand(PA, 0), 0), w1 << 32 | w0,
                   "Index Load Data is independent of the order");
        // The conversion itself, for either order.
        assert_eq!(staged_to_memory(w1 << 32 | w0), w0 << 32 | w1);
        assert_eq!(staged_to_memory(staged_to_memory(7)), 7);
    }

    /// A fetch hits too, and sees the line end when it is invalidated: the
    /// decode cache does not keep serving the staged word.
    #[test]
    fn a_fetch_runs_the_staged_word_until_the_line_is_invalidated() {
        let (c, bus) = staged_cache();
        let word = |c: &R10000ShadowCache| unsafe {
            (*c.fetch(operand(PA, 0), PA).instr).raw
        };
        assert_eq!(word(&c), 0xeeee_eeee, "memory before staging");
        let reads = bus.reads.load(Ordering::Relaxed);
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 0x2402_0022_0000_0000 | 0x0003_e008); // TagHi = addiu, TagLo = jr ra
        assert_eq!(word(&c), 0x0003_e008, "TagLo is the word at the lower address");
        assert_eq!(word(&c), 0x0003_e008);
        let f = c.fetch(operand(PA, 0) + 4, PA + 4);
        assert_eq!(unsafe { (*f.instr).raw }, 0x2402_0022);
        let f = c.fetch(operand(PA, 0) + 8, PA + 8);
        assert_eq!(unsafe { (*f.instr).raw }, 0, "the quadword's pad is a nop");
        assert_eq!(bus.reads.load(Ordering::Relaxed), reads);
        c.cache_op(C_IWBINV | CACH_SD, operand(PA, 0), 0);
        assert_eq!(word(&c), 0xeeee_eeee, "back to memory");
    }

    /// Eviction and invalidation end the hit: Index Writeback Invalidate by
    /// index, Hit Invalidate and Hit Writeback Invalidate by address, an
    /// Index Store Tag that writes another tag or an invalid state, and
    /// power-on. An operation on another line does not.
    #[test]
    fn invalidation_and_eviction_end_the_hit() {
        let ends: [(&str, Box<dyn Fn(&R10000ShadowCache)>); 6] = [
            ("Index Writeback Invalidate", Box::new(|c| { c.cache_op(C_IWBINV | CACH_SD, operand(PA, 0), 0); })),
            ("Hit Invalidate", Box::new(|c| { c.cache_op(C_HINV | CACH_SD, operand(PA, 0), PA); })),
            ("Hit Writeback Invalidate", Box::new(|c| { c.cache_op(C_HWBINV | CACH_SD, operand(PA, 0), PA); })),
            ("Index Store Tag, invalid", Box::new(|c| stage_tag(c, PA, 0, 0))),
            ("Index Store Tag, another line", Box::new(|c| stage_tag(c, PA + (1 << 19), 0, 2))),
            ("power-on", Box::new(|c| c.power_on())),
        ];
        for (name, end) in ends {
            let (c, bus) = staged_cache();
            stage_tag(&c, PA, 0, 2);
            stage_data(&c, PA, 0, 0x1234_5678_9abc_def0);
            assert_ne!(load::<8>(&c, PA), MEM, "{name}: hits before");
            end(&c);
            assert_eq!(load::<8>(&c, PA), MEM, "{name}: misses after");
            assert_eq!(bus.reads.load(Ordering::Relaxed), 1, "{name}");
            assert!(c.take_staged_changed(), "{name}");
            assert!(!c.page_staged(PA) || name == "Index Store Tag, another line");
        }
        // Another line's invalidation, and a hit operation that misses, leave it.
        let (c, _) = staged_cache();
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 0x1234_5678_9abc_def0);
        c.cache_op(C_IWBINV | CACH_SD, operand(PA + 128, 0), 0);
        c.cache_op(C_HINV | CACH_SD, operand(PA, 0), PA + (1 << 19));
        c.cache_op(C_IWBINV | CACH_SD, operand(PA, 1), 0); // the other way
        assert_ne!(load::<8>(&c, PA), MEM);
    }

    /// Writeback (ISUM pp.18-19, 27): invalidating a DirtyExclusive line
    /// sends its block to the system interface at the tag's address; other
    /// states only notify. Staged data and memory order agree.
    #[test]
    fn a_dirty_exclusive_line_is_written_back_when_it_goes() {
        for (state, written) in [(3, true), (2, false), (1, false)] {
            let (c, bus) = staged_cache();
            stage_tag(&c, PA, 0, state);
            stage_data(&c, PA + 16, 0, 0x1111_1111_2222_2222);
            c.cache_op(C_IWBINV | CACH_SD, operand(PA, 0), 0);
            let w = bus.writes.lock().unwrap().clone();
            assert_eq!(!w.is_empty(), written, "state {state}");
            if written {
                assert_eq!(w.len(), 16, "a 128-byte block, doubleword by doubleword");
                assert_eq!(w[0], (PA & !127, 0), "the line at the tag's address");
                assert_eq!(w[2], (PA + 16, 0x2222_2222_1111_1111), "in memory order");
            }
        }
    }

    /// A cached store to a staged line stays in the cache: memory is not
    /// written, and a later load sees it. A store elsewhere is memory's.
    #[test]
    fn a_store_to_a_staged_line_stays_in_the_cache() {
        let (c, bus) = staged_cache();
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 0);
        assert_eq!(c.write::<4>(operand(PA, 0) + 4, PA + 4, 0xdead_beef), BUS_OK);
        assert_eq!(c.write::<1>(operand(PA, 0) + 1, PA + 1, 0x7f), BUS_OK);
        assert_eq!(load::<4>(&c, PA + 4), 0xdead_beef);
        assert_eq!(load::<8>(&c, PA), 0x007f_0000_dead_beef);
        assert!(bus.writes.lock().unwrap().is_empty());
        c.write::<4>(operand(PA, 0) + 0x1000, PA + 0x1000, 5);
        assert_eq!(bus.writes.lock().unwrap().len(), 1);
    }

    /// **Inferred** (NEC U10278EJ4V0UM p.246: a store to a CleanExclusive or
    /// Shared block requests DirtyExclusive). A store moves the line to
    /// DirtyExclusive, so invalidating with writeback sends the stored bytes
    /// to memory instead of dropping them. Hit Invalidate still drops them.
    #[test]
    fn a_store_makes_the_line_dirty_and_the_writeback_carries_it() {
        for state in [1, 2, 3] {
            let (c, bus) = staged_cache();
            stage_tag(&c, PA, 0, state);
            stage_data(&c, PA, 0, 0);
            c.write::<4>(operand(PA, 0), PA, 0x0123_4567);
            assert_eq!((c.cache_op(C_ILT | CACH_SD, operand(PA, 0), 0) >> L2_STATE_SHIFT) & 3, 3, "state {state}");
            c.cache_op(C_IWBINV | CACH_SD, operand(PA, 0), 0);
            let w = bus.writes.lock().unwrap().clone();
            assert_eq!(w.len(), 16, "state {state}");
            assert_eq!(w[0], (PA & !127, 0x0123_4567_0000_0000), "state {state}: the stored word, at the lower address");
        }
        let (c, bus) = staged_cache();
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 0);
        c.write::<4>(operand(PA, 0), PA, 1);
        c.cache_op(C_HINV | CACH_SD, operand(PA, 0), PA);
        assert!(bus.writes.lock().unwrap().is_empty(), "Hit Invalidate does not write back");
        // A load does not dirty a line.
        let (c, bus) = staged_cache();
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 5);
        load::<4>(&c, PA);
        c.cache_op(C_IWBINV | CACH_SD, operand(PA, 0), 0);
        assert!(bus.writes.lock().unwrap().is_empty());
    }

    /// op[1:0] = 10 is undefined on the R10000: it neither stages nor ends a line.
    #[test]
    fn the_undefined_secondary_instruction_encoding_does_nothing() {
        let (c, bus) = staged_cache();
        c.cache_op(C_IST | CACH_SI, operand(PA & !127, 0), l2_tag_of(PA) | 2 << L2_STATE_SHIFT);
        assert!(!c.page_staged(PA));
        assert_eq!(load::<8>(&c, PA), MEM);
        stage_tag(&c, PA, 0, 2);
        stage_data(&c, PA, 0, 7);
        c.cache_op(C_IWBINV | CACH_SI, operand(PA, 0), 0);
        c.cache_op(C_HINV | CACH_SI, operand(PA, 0), PA);
        c.cache_op(C_HWBINV | CACH_SI, operand(PA, 0), PA);
        assert_ne!(load::<8>(&c, PA), MEM, "still staged");
        assert_eq!(bus.reads.load(Ordering::Relaxed), 1);
    }

    /// The load's VA[13:12] may differ from TagLo[8:7].
    /// The lookup is physical, so the line hits whatever virtual address is used.
    #[test]
    fn a_different_virtual_index_still_hits() {
        let (c, bus) = staged_cache();
        c.cache_op(C_IST | CACH_SD, operand(PA & !127, 0), l2_tag_of(PA) | 2 << L2_STATE_SHIFT | 3 << 7);
        stage_data(&c, PA, 0, 0x1111_2222_3333_4444);
        for va_bits in [0u64, 0x1000, 0x2000, 0x3000] {
            let r = c.read::<8>(0xffff_ffff_8000_0000 | (PA & 0xfff) | va_bits, PA);
            assert_eq!(r.data, 0x3333_4444_1111_2222, "VA[13:12] = {}", va_bits >> 12);
        }
        assert_eq!(bus.reads.load(Ordering::Relaxed), 0);
    }

    /// The same rules at every geometry the family ships: 2 MiB with 128-byte
    /// lines (Octane, Fuel), 1 MiB with 64-byte lines (O2), 4 MiB (R16000).
    /// Two lines of one set differ by a way's worth of address.
    #[test]
    fn staging_works_at_every_cache_geometry() {
        use crate::cpu::mips_cache_v2::MipsCache as _;
        fn check<C: MipsCache + CpuModel + From<Arc<dyn BusDevice>>>(name: &str) {
            let bus = Arc::new(Bus::default());
            let c = C::from(bus.clone() as Arc<dyn BusDevice>);
            let line = C::L2_LINE as u64;
            let way_bytes = (C::L2_SIZE / 2) as u64;
            let (a, b) = (0x0020_3000u64 + line, 0x0020_3000u64 + line + way_bytes);
            let op = |pa: u64, way: u64| 0xffff_ffff_8000_0000 | (pa & 0x1fff_fff8) | way;
            let tag = |pa: u64, way: u64, state: u64| {
                c.cache_op(C_IST | CACH_SD, op(pa & !(line - 1), way), l2_tag_of(pa) | state << L2_STATE_SHIFT);
            };
            let data = |pa: u64, way: u64, d: u64| { c.cache_op(C_R10K_ISD | CACH_SD, op(pa & !7, way), d); };
            let load = |pa: u64| c.read::<8>(op(pa, 0), pa).data;
            tag(a, 0, 2); data(a, 0, 0x0a0a_0a0a_0b0b_0b0b);
            tag(b, 1, 2); data(b, 1, 0x0c0c_0c0c_0d0d_0d0d);
            assert_eq!(load(a), 0x0b0b_0b0b_0a0a_0a0a, "{name}: way 0");
            assert_eq!(load(b), 0x0d0d_0d0d_0c0c_0c0c, "{name}: way 1");
            assert_eq!(load(a + 8), 0, "{name}: the line's unstaged doubleword");
            assert_eq!(bus.reads.load(Ordering::Relaxed), 0, "{name}");
            // Flip each tag address bit above the index. The first one is
            // the other line's, which is staged in way 1.
            assert_eq!(load(b), 0x0d0d_0d0d_0c0c_0c0c, "{name}: the other way's line");
            let index_top = (C::L2_SIZE / 2).trailing_zeros();
            for bit in (index_top + 1)..36 {
                assert_eq!(load(a ^ (1 << bit)), MEM, "{name}: PA bit {bit}");
            }
            // Dirty write back covers the whole line, at this cache's line size.
            tag(a, 0, 3);
            c.cache_op(C_IWBINV | CACH_SD, op(a, 0), 0);
            let w = bus.writes.lock().unwrap().clone();
            assert_eq!(w.len() as u64, line / 8, "{name}");
            assert_eq!(w[0].0, a & !(line - 1), "{name}");
            assert_eq!(load(a), MEM, "{name}: ended");
            assert_ne!(load(b), MEM, "{name}: the other way stays");
        }
        check::<R10000ShadowCache>("R10000 1 MiB/128");
        check::<R10000O2ShadowCache>("R10000 O2 1 MiB/64");
        check::<R12000ShadowCache>("R12000 2 MiB/128");
        check::<R14000FuelShadowCache>("R14000 2 MiB/128");
        check::<R16000ShadowCache>("R16000 4 MiB/128");
    }

    /// A snapshot carries the staged lines, and a restore ends any that were
    /// staged before it. Dirty lines also reach memory on `writeback_all`.
    #[test]
    fn staged_lines_survive_a_snapshot_round_trip() {
        let (c, _) = staged_cache();
        assert!(c.save_cache_state().as_table().unwrap().is_empty(), "nothing staged, nothing saved");
        stage_tag(&c, PA, 1, 3);
        stage_data(&c, PA + 8, 1, 0x8000_0000_0000_0001);
        c.cache_op(C_R10K_ISD | CACH_SD, operand(PA + 8, 1), 0x8000_0000_0000_0001);
        let saved = c.save_cache_state();

        let (d, bus) = staged_cache();
        stage_tag(&d, PA + 0x4000, 0, 2); // staged before the restore: must go
        d.load_cache_state(&saved).unwrap();
        assert_eq!(load::<8>(&d, PA + 8), 0x0000_0001_8000_0000);
        assert_eq!(load::<8>(&d, PA + 0x4000), MEM);
        assert!(d.page_staged(PA) && !d.page_staged(PA + 0x4000));
        assert_eq!(bus.reads.load(Ordering::Relaxed), 1);
        assert_eq!(d.cache_op(C_ILT | CACH_SD, operand(PA, 1), 0), l2_tag_of(PA) | 3 << L2_STATE_SHIFT);

        d.writeback_all();
        let w = bus.writes.lock().unwrap().clone();
        assert_eq!(w.len(), 16);
        assert_eq!(w[1], (PA & !127 | 8, 0x0000_0001_8000_0000));
        assert_ne!(load::<8>(&d, PA + 8), MEM, "writeback_all keeps the line");

        let empty = toml::Value::Table(Default::default());
        d.load_cache_state(&empty).unwrap();
        assert_eq!(load::<8>(&d, PA + 8), MEM, "restoring a snapshot with no staged lines ends them");
    }

    /// Index Store Data (S) writes the one
    /// quadword PA[6:4] names and leaves
    /// the line's other quadwords alone, and a load or fetch finds each
    /// quadword by the same bits. A quadword is stored once, the others are
    /// rewritten many times, and the first must still read back what was
    /// stored: for every quadword of the line, on 64- and 128-byte lines.
    #[test]
    fn index_store_data_writes_only_the_quadword_the_address_names() {
        fn check<C: MipsCache + CpuModel + From<Arc<dyn BusDevice>>>(name: &str) {
            let bus = Arc::new(Bus::default());
            let c = C::from(bus.clone() as Arc<dyn BusDevice>);
            let line = C::L2_LINE as u64;
            let pa0 = 0x0020_3000u64 & !(line - 1);
            let op = |pa: u64| 0xffff_ffff_8000_0000 | (pa & 0x1fff_fff8);
            let quads = line / 16;
            for target in 0..quads {
                c.cache_op(C_IST | CACH_SD, op(pa0), l2_tag_of(pa0) | 2 << L2_STATE_SHIFT);
                // Clear the whole line, then store the target quadword once.
                for q in 0..quads {
                    c.cache_op(C_R10K_ISD | CACH_SD, op(pa0 + 16 * q), 0);
                }
                // The upper doubleword of the quadword, to catch a mix-up of PA[3].
                let at = pa0 + 16 * target + 8;
                let (w0, w1) = (0x1000_0000 + target as u32, 0x2000_0000 + target as u32);
                c.cache_op(C_R10K_ISD | CACH_SD, op(at), (w1 as u64) << 32 | w0 as u64);
                // Rewrite every other quadword many times, both halves.
                for round in 0..40u64 {
                    for q in (0..quads).filter(|&q| q != target) {
                        let half = (round + q) & 1;
                        c.cache_op(C_R10K_ISD | CACH_SD, op(pa0 + 16 * q + 8 * half), 0xdead_0000_0000_0000 | round << 8 | q);
                    }
                }
                // Index Load Data (S) of the quadword returns what was stored. The
                // check bits on the rewrites are assumed valid: none are modelled.
                assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, op(at), 0), (w1 as u64) << 32 | w0 as u64,
                           "{name} quadword {target}: Index Load Data");
                let rd = |pa: u64| c.read::<4>(op(pa), pa).data;
                assert_eq!(rd(at), w0 as u64, "{name} quadword {target}: word at the lower address");
                assert_eq!(rd(at + 4), w1 as u64, "{name} quadword {target}: word at +4");
                assert_eq!(rd(at - 8), 0, "{name} quadword {target}: its other doubleword");
                assert_eq!(rd(at - 4), 0, "{name} quadword {target}: its other doubleword");
                let f = c.fetch(op(at), at);
                assert_eq!(unsafe { (*f.instr).raw } as u64, w0 as u64, "{name} quadword {target}: fetch");
                assert_eq!(bus.reads.load(Ordering::Relaxed), 0, "{name}");
                // And the neighbours still hold what they were last given.
                for q in (0..quads).filter(|&q| q != target) {
                    let half = (39 + q) & 1;
                    let v = 0xdead_0000_0000_0000u64 | 39 << 8 | q;
                    assert_eq!(c.read::<8>(op(pa0 + 16 * q + 8 * half), pa0 + 16 * q + 8 * half).data, v.rotate_left(32),
                               "{name} quadword {q}");
                }
            }
        }
        check::<R10000ShadowCache>("128-byte lines");
        check::<R10000O2ShadowCache>("64-byte lines");
        check::<R14000FuelShadowCache>("2 MiB, 128-byte lines");
    }

    // ---- Staged primary instruction-cache lines -------------------------

    /// Index Store Tag (I) on `pa`'s block at the set `va` indexes, `way`.
    fn stage_i_tag(c: &R10000ShadowCache, va: u64, pa: u64, way: u64, valid: bool) {
        let tag = l1i_tag_of(pa) | if valid { L1I_VALID } else { 0 };
        c.cache_op(C_IST | CACH_PI, (va & !63) | way, tag);
    }

    /// Index Store Data (I): the word at `va`, in `way`, is `w`.
    fn stage_i_word(c: &R10000ShadowCache, va: u64, way: u64, w: u32) {
        c.cache_op(C_R10K_ISD | CACH_PI, (va & !3) | way, w as u64);
    }

    fn fetch_raw(c: &R10000ShadowCache, va: u64, pa: u64) -> u32 {
        unsafe { (*c.fetch(va, pa).instr).raw }
    }

    /// Stage whole primary I blocks on way 1 of several sets (16 words each)
    /// and fetch through them: the staged words, with no bus read.
    #[test]
    fn a_staged_primary_instruction_line_serves_fetches() {
        let (c, bus) = staged_cache();
        let base = 0x0030_0000u64;
        for set in 0..4u64 {
            let (va, pa) = (0xffff_ffff_8000_0000 | (base + set * 64), base + set * 64);
            stage_i_tag(&c, va, pa, 1, true);
            for w in 0..16u64 {
                stage_i_word(&c, va + 4 * w, 1, 0x3c00_0000 | (set << 8) as u32 | w as u32);
            }
        }
        for set in 0..4u64 {
            for w in 0..16u64 {
                let (va, pa) = (0xffff_ffff_8000_0000 | (base + set * 64 + 4 * w), base + set * 64 + 4 * w);
                assert_eq!(fetch_raw(&c, va, pa), 0x3c00_0000 | (set << 8) as u32 | w as u32, "set {set} word {w}");
            }
        }
        assert_eq!(bus.reads.load(Ordering::Relaxed), 0);
        // The index is virtual: the same physical block through a VA with
        // other VA[13:6] misses. The tag is physical: another frame misses.
        assert_eq!(fetch_raw(&c, 0xffff_ffff_8000_0000 | (base + 0x400), base), 0xeeee_eeee);
        assert_eq!(fetch_raw(&c, 0xffff_ffff_8000_0000 | base, base + (1 << 20)), 0xeeee_eeee);
        // Loads are not served by an instruction line.
        assert_eq!(load::<4>(&c, base), 0xeeee_eeee);
    }

    /// A secondary staged line served fetches first; ending it leaves the
    /// primary I lines hitting. With both valid the primary wins.
    #[test]
    fn primary_instruction_lines_outlive_the_secondary_line_and_win_over_it() {
        let (c, bus) = staged_cache();
        let (pa, va) = (0x0030_0000u64, 0xffff_ffff_8030_0000u64);
        stage_tag(&c, pa, 0, 2);
        for q in 0..8 {
            stage_data(&c, pa + 16 * q, 0, 0x5555_0000_6666_0000 | q);
        }
        assert_eq!(fetch_raw(&c, va, pa), 0x6666_0000, "the secondary line");
        stage_i_tag(&c, va, pa, 1, true);
        for w in 0..16 {
            stage_i_word(&c, va + 4 * w, 1, 0x7000 + w as u32);
        }
        assert_eq!(fetch_raw(&c, va, pa), 0x7000, "primary before secondary");
        stage_tag(&c, pa, 0, 0); // the secondary line ends
        assert_eq!(fetch_raw(&c, va, pa), 0x7000);
        assert_eq!(fetch_raw(&c, va + 8, pa + 8), 0x7002);
        assert_eq!(bus.reads.load(Ordering::Relaxed), 0);
    }

    /// Ending a primary I line: Index Store Tag with an invalid state, Index
    /// Invalidate (the way named), Hit Invalidate (by address), power-on. A
    /// fetch served before must not be served again. Another way, another
    /// set, and an Index Invalidate of the other way leave the line.
    #[test]
    fn primary_instruction_lines_end() {
        let (pa, va) = (0x0030_0040u64, 0xffff_ffff_8030_0040u64);
        let ends: [(&str, Box<dyn Fn(&R10000ShadowCache)>); 4] = [
            ("Index Store Tag, invalid", Box::new(move |c| stage_i_tag(c, va, pa, 1, false))),
            ("Index Invalidate", Box::new(move |c| { c.cache_op(C_IINV | CACH_PI, va | 1, 0); })),
            ("Hit Invalidate", Box::new(move |c| { c.cache_op(C_HINV | CACH_PI, va, pa); })),
            ("power-on", Box::new(|c| c.power_on())),
        ];
        for (name, end) in ends {
            let (c, _) = staged_cache();
            stage_i_tag(&c, va, pa, 1, true);
            stage_i_word(&c, va, 1, 0x1234);
            assert_eq!(fetch_raw(&c, va, pa), 0x1234, "{name}: before");
            assert!(c.page_staged(pa));
            end(&c);
            assert_eq!(fetch_raw(&c, va, pa), 0xeeee_eeee, "{name}: after");
            assert!(!c.page_staged(pa), "{name}");
        }
        let (c, _) = staged_cache();
        stage_i_tag(&c, va, pa, 1, true);
        stage_i_word(&c, va, 1, 0x1234);
        c.cache_op(C_IINV | CACH_PI, va, 0); // the other way
        c.cache_op(C_IINV | CACH_PI, va + 64, 0); // the next set
        c.cache_op(C_HINV | CACH_PI, va, pa + (1 << 20)); // misses
        assert_eq!(fetch_raw(&c, va, pa), 0x1234);
    }

    /// Staged primary I lines survive a snapshot round trip.
    #[test]
    fn staged_primary_instruction_lines_are_saved() {
        let (c, _) = staged_cache();
        let (pa, va) = (0x0030_0000u64, 0xffff_ffff_8030_0000u64);
        stage_i_tag(&c, va, pa, 1, true);
        for w in 0..16 {
            stage_i_word(&c, va + 4 * w, 1, 0x7000 + w as u32);
        }
        let saved = c.save_cache_state();
        let (d, _) = staged_cache();
        d.load_cache_state(&saved).unwrap();
        assert_eq!(fetch_raw(&d, va + 12, pa + 12), 0x7003);
        assert!(d.page_staged(pa));
        d.load_cache_state(&toml::Value::Table(Default::default())).unwrap();
        assert_eq!(fetch_raw(&d, va + 12, pa + 12), 0xeeee_eeee);
    }

    // ---- Mapped operands ------------------------------------------------

    type MappedExec = crate::cpu::mips_exec::MipsExecutor<crate::cpu::mips_tlb::MipsTlb, R10000ShadowCache>;

    /// An executor, a cached 4 KiB page at 0x200000 mapped to frame `pfn`
    /// with cache algorithm `alg` and dirty bit `dirty`, and the bus it sits on.
    fn mapped(alg: u64, dirty: u64, pfn: u64) -> (Box<MappedExec>, Arc<Bus>) {
        use crate::cpu::mips_exec::{MipsCpuConfig, MipsExecutor, EXEC_COMPLETE};
        let bus = Arc::new(Bus::default());
        let mut exec = Box::new(MipsExecutor::<crate::cpu::mips_tlb::MipsTlb, R10000ShadowCache>::new(
            bus.clone() as Arc<dyn BusDevice>, crate::cpu::mips_tlb::MipsTlb::default(),
            &MipsCpuConfig::for_model::<R10000ShadowCache>(Arc::new(crate::ip22_bus::memory_map(false)))));
        #[cfg(feature = "jitv2")]
        { exec.jitv2_dispatch_enabled = false; }
        exec.core.cp0_config = (exec.core.cp0_config & !7) | 3; // KSEG0 cacheable
        exec.core.cp0_status &= !crate::cpu::mips_core::STATUS_ERL;
        exec.resync_privilege_state();
        exec.core.cp0_index = 3;
        exec.core.cp0_pagemask = 0;
        exec.core.cp0_entryhi = 0x100 << 13;
        exec.core.cp0_entrylo0 = (pfn << 6) | (alg << 3) | (dirty << 2) | (1 << 1);
        exec.core.cp0_entrylo1 = 0;
        use crate::cpu::mips_isa::OP_COP0;
        assert_eq!(exec.exec((OP_COP0 << 26) | (0x10 << 21) | 0x02), EXEC_COMPLETE); // TLBWI
        (exec, bus)
    }

    /// CACHE `op` on `va`, with TagLo/TagHi as given; the instruction's status.
    fn cache_at(exec: &mut MappedExec, op: u32, va: u64, tag: u64) -> u32 {
        exec.core.gpr[4] = va;
        exec.core.cp0_taglo = tag & 0xffff_ffff;
        exec.core.cp0_taghi = (tag >> 32) as u32;
        exec.exec((crate::cpu::mips_isa::OP_CACHE << 26) | (4 << 21) | (op << 16))
    }

    const VA: u64 = 0x20_0000;

    /// The executor is large: run on a big stack, as the other TLB tests do.
    fn big_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new().stack_size(16 * 1024 * 1024).spawn(f).unwrap().join().unwrap();
    }

    /// The secondary set and way come from the physical address of a mapped
    /// operand (NEC U12754EJ1V0UMJ1 pp.14-15). Stage through the mapped page, then a cached
    /// access to the frame's kseg0 address hits, and the tag is at the frame's
    /// set, not the virtual address's.
    #[test]
    fn a_mapped_index_operation_lands_on_the_physical_set() {
        big_stack(|| {
        use crate::cpu::mips_exec::EXEC_COMPLETE;
        let pfn = 0x3a5; // PA 0x3a5000: set 0x4a0; the VA's set is 0
        let (mut exec, bus) = mapped(3, 1, pfn);
        let pa = pfn << 12 | 0x40;
        let tag = l2_tag_of(pa) | 2 << L2_STATE_SHIFT;
        assert_eq!(cache_at(&mut exec, 0x0b, VA + 0x40, tag), EXEC_COMPLETE);
        assert_eq!(cache_at(&mut exec, 0x1f, VA + 0x40, 0x3333_4444_1111_2222), EXEC_COMPLETE);
        let kseg0 = 0xffff_ffff_8000_0000 | pa;
        assert_eq!(exec.cache.read::<4>(kseg0, pa).data, 0x1111_2222);
        assert_eq!(exec.cache.read::<4>(kseg0 + 4, pa + 4).data, 0x3333_4444);
        let f = exec.cache.fetch(kseg0, pa);
        assert_eq!(unsafe { (*f.instr).raw }, 0x1111_2222, "a cached fetch through the frame");
        assert_eq!(bus.reads.load(Ordering::Relaxed), 0);
        // The tag is at the physical set, read back through an unmapped operand.
        assert_eq!(cache_at(&mut exec, 0x07, 0xffff_ffff_8000_0000 | (pa & !127), 0), EXEC_COMPLETE);
        assert_eq!(exec.core.cp0_taglo | (exec.core.cp0_taghi as u64) << 32, tag);
        // And nothing is at the set the virtual address would have chosen.
        assert_eq!(cache_at(&mut exec, 0x07, 0xffff_ffff_8000_0000 | VA, 0), EXEC_COMPLETE);
        assert_eq!(exec.core.cp0_taglo, 0);
        // Index Writeback Invalidate through the mapped operand ends the line.
        assert_eq!(cache_at(&mut exec, 0x03, VA + 0x40, 0), EXEC_COMPLETE);
        assert_eq!(exec.cache.read::<4>(kseg0, pa).data, 0xeeee_eeee);
        })

    }

    /// Two virtual pages on one frame reach one secondary line; the primary
    /// index stays virtual.
    #[test]
    fn aliases_share_a_secondary_line_and_the_primary_stays_virtual() {
        big_stack(|| {
        use crate::cpu::mips_exec::EXEC_COMPLETE;
        let (mut exec, _) = mapped(3, 1, 0x3a5);
        let pa = 0x3a5000u64;
        let tag = l2_tag_of(pa) | 2 << L2_STATE_SHIFT;
        cache_at(&mut exec, 0x0b, VA, tag);
        // Another way to the same line: the unmapped address of the frame.
        assert_eq!(cache_at(&mut exec, 0x07, 0xffff_ffff_8000_0000 | pa, 0), EXEC_COMPLETE);
        assert_eq!(exec.core.cp0_taglo | (exec.core.cp0_taghi as u64) << 32, tag);
        // Primary D: indexed by VA[13:5]. Mapped store, unmapped load at the
        // same low 14 bits but another frame.
        assert_eq!(cache_at(&mut exec, 0x09, VA + 0x40, 0x0000_0abc), EXEC_COMPLETE);
        assert_eq!(cache_at(&mut exec, 0x05, 0xffff_ffff_8000_0000 | 0x40, 0), EXEC_COMPLETE);
        assert_eq!(exec.core.cp0_taglo, 0x0abc);
        })

    }

    /// A mapped operand takes TLB Refill when no entry maps it, and the cache
    /// does not change.
    #[test]
    fn an_unmapped_operand_takes_tlb_refill() {
        big_stack(|| {
        use crate::cpu::mips_exec::{EXEC_IS_EXCEPTION, EXEC_IS_TLB_REFILL};
        let (mut exec, _) = mapped(3, 1, 0x3a5);
        let st = cache_at(&mut exec, 0x0b, 0x60_0000, l2_tag_of(0x3a5000) | 2 << L2_STATE_SHIFT);
        assert!(st & EXEC_IS_EXCEPTION != 0 && st & EXEC_IS_TLB_REFILL != 0, "status {st:#x}");
        assert!(!exec.cache.page_staged(0x3a5000));
        // The same through an index operation on a data cache.
        let st = cache_at(&mut exec, 0x1d, 0x60_0000, 0);
        assert!(st & EXEC_IS_EXCEPTION != 0 && st & EXEC_IS_TLB_REFILL != 0, "status {st:#x}");
        })

    }

    /// A valid entry with the dirty bit clear is accepted: CACHE never takes
    /// TLB Modified.
    #[test]
    fn a_clean_page_takes_no_tlb_modified() {
        big_stack(|| {
        use crate::cpu::mips_exec::EXEC_COMPLETE;
        let (mut exec, _) = mapped(3, 0, 0x3a5);
        let tag = l2_tag_of(0x3a5000) | 2 << L2_STATE_SHIFT;
        assert_eq!(cache_at(&mut exec, 0x0b, VA, tag), EXEC_COMPLETE);
        assert_eq!(cache_at(&mut exec, 0x1f, VA, 7), EXEC_COMPLETE);
        assert_eq!(cache_at(&mut exec, 0x03, VA, 0), EXEC_COMPLETE);
        })

    }

    /// A CACHE operand on an uncached page is undefined on the R10000. This
    /// model translates it and goes ahead as if the page were cached; pinned
    /// here so a change is deliberate.
    #[test]
    fn an_operand_on_an_uncached_page_goes_ahead() {
        big_stack(|| {
        use crate::cpu::mips_exec::EXEC_COMPLETE;
        let (mut exec, _) = mapped(2, 1, 0x3a5);
        let tag = l2_tag_of(0x3a5000) | 2 << L2_STATE_SHIFT;
        assert_eq!(cache_at(&mut exec, 0x0b, VA, tag), EXEC_COMPLETE);
        assert!(exec.cache.page_staged(0x3a5000));
        })

    }

    /// Which pages the page-granular fast paths must leave to the cache:
    /// exactly those with a valid staged line, reported once as a change.
    #[test]
    fn the_staged_page_set_follows_the_valid_lines() {
        let (c, _) = staged_cache();
        assert!(!c.page_staged(PA) && !c.take_staged_changed());
        stage_tag(&c, PA, 0, 2);
        assert!(c.page_staged(PA) && c.page_staged(PA | 0xfff) && !c.page_staged(PA + 0x1000));
        assert!(c.take_staged_changed());
        assert!(!c.take_staged_changed(), "reading clears it");
        stage_tag(&c, PA + 128, 0, 1); // second line, same page: no change
        stage_tag(&c, PA, 0, 0);
        assert!(c.page_staged(PA) && !c.take_staged_changed());
        stage_tag(&c, PA + 128, 0, 0);
        assert!(!c.page_staged(PA) && c.take_staged_changed());
    }

    /// Only the R10000 family stages lines: the R4400's index operations are
    /// as before.
    #[test]
    fn the_r4400_does_not_stage_lines() {
        let c = R4400ShadowCache::from(Arc::new(Bus::default()) as Arc<dyn BusDevice>);
        c.cache_op(C_IST | CACH_SD, 0x8010_1000, 0x0000_0800 | 4 << 14);
        assert!(!c.page_staged(PA));
        assert_eq!(c.read::<8>(0x8010_1000, PA).data, MEM);
    }

    /// Under tcache the JIT serves this model through the window alone, and
    /// only once both windows are published: the emitted code dereferences
    /// them without a null check.
    #[test]
    #[cfg(feature = "jitv2")]
    fn tcache_offers_a_tagless_path_once_both_windows_exist() {
        let c = cache();
        assert!(!c.jit_dc_geometry().supported, "no window yet");
        let mut window = [0u8; 8];
        unsafe { c.set_tcache_window(window.as_mut_ptr()) };
        assert!(!c.jit_dc_geometry().supported, "no generation window yet");
        let mut gens = [std::sync::atomic::AtomicU64::new(0)];
        unsafe { c.set_tcache_gen_window(gens.as_mut_ptr()) };
        let g = c.jit_dc_geometry();
        assert!(g.supported && g.tagless && !g.has_l2,
            "tagless, and no L2 decode slots for a store to invalidate: {g:?}");
    }
}
