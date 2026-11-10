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
//!   (`CpuModel::SECONDARY_TAG_MASK`).
//!
//! The shadow is deliberately not consulted by `read`/`write`/`fetch`. If it
//! ever needs to be, this type is the wrong shape and should say so loudly
//! rather than growing a slow path.
//!
//! The R10000-family PROMs use the caches as memory before memory works: they
//! make lines valid with Index Store Tag and Index Store Data and keep a stack
//! or run code in them, while the uncached path sees memory (NEC VR10000
//! user's manual p. 75). Such a line lives in ppmem's window rather than
//! here: the window this cache reads is the CPU's own cached view
//! (`PpMemSpace::add_cached_view`), and a valid line makes its page private
//! memory in that view, holding the line's bytes
//! ([`ShadowCache::line_enters`]). Loads, stores and fetches read the view as
//! they always did. The CACHE instruction is the only thing that moves a page
//! in or out, and the tags stay the only record of which lines are valid.
//! `clanker/cpu/cache-lines-live-in-the-cpus-view.md` has the limits.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::cpu::mips_cache_v2::{
    cache_op_name, model, r4k_load_taglo, r4k_store_taglo, CpuModel, FetchInstrResult, MipsCache,
    C_ILT, C_IST, C_IWBINV, C_R10K_CBARRIER, C_R10K_ILD, C_R10K_ISD, CACH_PD, CACH_PI, CACH_SD, CACH_SI,
};
use crate::ppmem::PpMemSpace;
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
    /// The line holds its page in the CPU's cached view
    /// (`ShadowCache::line_enters`).
    held: Box<[bool]>,
}

impl Shadow {
    fn new(lines: usize, data_words: usize) -> Self {
        Self {
            held: vec![false; lines.max(1)].into_boxed_slice(),
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

/// R10000-family tag states (NEC U10278EJ3V0UM pp. 285-287): the primary data
/// state in TagLo[7:6] and the secondary state in TagLo[11:10], 0 Invalid.
const PD_STATE_SHIFT: u32 = 6;
const SD_STATE_SHIFT: u32 = 10;

/// The primary instruction state, TagLo[6] (NEC U10278EJ3V0UM p. 285).
const PI_VALID: u64 = 1 << 6;

/// A primary instruction word is the instruction predecoded, 36 bits (NEC
/// U10278EJ3V0UM p. 288). Neither manual gives the layout. This one is
/// inferred from the words PROM 6.210 stores, `p` the word and `f` the
/// instruction: p[25:20] = f[5:0], p[19:15] = f[10:6], p[14:10] = f[25:21],
/// p[28:26] = f[28:26], p[35:29] a class that also gives f[31:29], and p[4:0]
/// and p[9:5] the rt and rd fields in the order the class gives. Only the
/// sampled classes decode, and every row is inferred. `addiu $4,$0,2`
/// (0x24040002) is 0x3_4420_0004.
///
/// Rows: class, f[31:29], whether p[4:0] is rt, the f[28:26] values as a bit
/// mask, and the function fields of the SPECIAL members.
const L1I_CLASSES: &[(u8, u32, bool, u8, &[u8])] = &[
    (0x00, 0, false, 1 << 0, &[0x00]),       // nop
    (0x08, 0, false, 1 << 0, &[0x38, 0x3c]), // dsll, dsll32
    (0x18, 0, false, 1 << 0, &[0x25]),       // or
    (0x1a, 1, true, 0b1010_0011, &[]),       // addi, addiu, ori, lui
    (0x1e, 3, true, 1 << 1, &[]),            // daddiu
    (0x35, 5, false, 1 << 7, &[]),           // cache
    (0x36, 6, true, 1 << 7, &[]),            // ld
    (0x3f, 2, false, 1 << 0, &[]),           // mtc0
    (0x40, 0, false, 1 << 3, &[]),           // jal
    (0x41, 0, false, 0b101, &[0x08]),        // jr, j
    (0x71, 0, false, 1 << 1, &[]),           // REGIMM
];

/// The model's own form, bits 35:32 set over an instruction no class holds.
/// Hardware would never produce it. It takes classes 0x78-0x7f, so a real
/// word of those classes would be read as raw.
const L1I_RAW: u64 = 0xf << 32;

/// The instruction a predecoded word encodes, or the class that is refused.
fn l1i_decode(p: u64) -> Result<u32, u8> {
    if p >> 32 == L1I_RAW >> 32 {
        return Ok(p as u32);
    }
    let (class, op, funct) = ((p >> 29) as u8 & 0x7f, (p >> 26) as u32 & 7, (p >> 20) as u8 & 0x3f);
    let &(_, hi3, rt_low, ops, functs) = L1I_CLASSES.iter().find(|c| c.0 == class).ok_or(class)?;
    if ops >> op & 1 == 0 || (op == 0 && !functs.is_empty() && !functs.contains(&funct)) || (class == 0 && p != 0) {
        return Err(class);
    }
    let (low, high) = ((p & 0x1f) as u32, (p >> 5) as u32 & 0x1f);
    let (rt, rd) = if rt_low { (low, high) } else { (high, low) };
    Ok(hi3 << 29 | op << 26 | ((p >> 10) as u32 & 0x1f) << 21 | rt << 16 | rd << 11 | ((p >> 15) as u32 & 0x1f) << 6 | funct as u32)
}

/// The predecoded word for `f`, or the raw form where no class holds it.
fn l1i_encode(f: u32) -> u64 {
    let w = f as u64;
    let (rt, rd) = (w >> 16 & 0x1f, w >> 11 & 0x1f);
    L1I_CLASSES.iter().find_map(|&(class, _, rt_low, _, _)| {
        let (low, high) = if rt_low { (rt, rd) } else { (rd, rt) };
        let p = (class as u64) << 29 | (w >> 26 & 7) << 26 | (w & 0x3f) << 20 | (w >> 6 & 0x1f) << 15
            | (w >> 21 & 0x1f) << 10 | high << 5 | low;
        (l1i_decode(p) == Ok(f)).then_some(p)
    }).unwrap_or(L1I_RAW | w)
}

/// R10000 only: ways per set, as CACHE index operations address them. Bit 0
/// of the index selects the way; the set starts above the line offset. The
/// R4400 and R5000 index differently — see `tag_slot`.
const WAYS: usize = 2;

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
    tc_bitmap: AtomicU64,
    /// The window's `ppmem::window_shift`, set before the CPU runs.
    tc_shift: UnsafeCell<u32>,
    /// The window's fine table and its length in regions, for an access the
    /// bitmap misses (`PpMemSpace::fine_table`). Zero regions until set.
    tc_fine: UnsafeCell<*const std::sync::atomic::AtomicU64>,
    tc_fine_regions: UnsafeCell<u64>,
    /// The key the decode cache holds its window pointers under, published
    /// on every change of the bitmap or the fine table.
    tc_key: AtomicU64,
    #[cfg(feature = "jitv2")]
    tc_gen: UnsafeCell<*mut std::sync::atomic::AtomicU64>,
    /// Decoded instructions, by physical page. Never touches the shadow
    /// arrays above — see `decode_cache`.
    decode: UnsafeCell<crate::cpu::decode_cache::DecodeCache>,
    /// The cached view `tc_base` is, and its number in `PpMemSpace`: where
    /// valid lines hold their pages. Null without one.
    view_space: UnsafeCell<*const PpMemSpace>,
    view: UnsafeCell<usize>,
    /// The held pages whose generation counters are not ppmem's, with the
    /// counters carrying their `GEN_HELD`: see [`Self::mark_foreign`].
    #[cfg(feature = "jitv2")]
    foreign: UnsafeCell<Vec<(u64, *const std::sync::atomic::AtomicU64)>>,
    /// For each primary instruction set, the 64-byte half of a held
    /// secondary line that a refill there would load (`refilled_word`), or
    /// `u64::MAX`.
    refill: UnsafeCell<Box<[u64]>>,
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
    /// manual, Appendix B.20. The R14000 is taken to match.
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
            tc_bitmap: AtomicU64::new(0),
            tc_shift: UnsafeCell::new(crate::ppmem::BITMAP_SHIFT),
            tc_fine: UnsafeCell::new(std::ptr::null()),
            tc_fine_regions: UnsafeCell::new(0),
            tc_key: AtomicU64::new(0),
            #[cfg(feature = "jitv2")]
            tc_gen: UnsafeCell::new(std::ptr::null_mut()),
            decode: UnsafeCell::new(crate::cpu::decode_cache::DecodeCache::new()),
            view_space: UnsafeCell::new(std::ptr::null()),
            view: UnsafeCell::new(0),
            #[cfg(feature = "jitv2")]
            foreign: UnsafeCell::new(Vec::new()),
            refill: UnsafeCell::new(vec![u64::MAX; Self::PI_SETS].into_boxed_slice()),
        }
    }

    /// ppmem's window base if `phys` is RAM the window maps, as the region
    /// bitmap says; `None` for the PROM, MMIO and the low-512KB alias, which go
    /// through the bus. The bitmap stays zero until the window is wired, so a
    /// set bit also means the base is valid.
    #[inline(always)]
    fn ram(&self, phys: u64) -> Option<*mut u8> {
        if crate::ppmem::bitmap_maps_at(self.tc_bitmap.load(Ordering::Relaxed), unsafe { *self.tc_shift.get() }, phys)
            || self.fine_maps(phys)
        {
            Some(unsafe { *self.tc_base.get() })
        } else {
            None
        }
    }

    /// Whether the fine table holds `phys`'s region whole: for an access the
    /// bitmap missed, before it takes the bus. False until a table is set.
    #[inline(always)]
    fn fine_maps(&self, phys: u64) -> bool {
        let r = phys >> crate::ppmem::FINE_SHIFT;
        r < unsafe { *self.tc_fine_regions.get() }
            && unsafe { (*(*self.tc_fine.get()).add((r / 64) as usize)).load(std::sync::atomic::Ordering::Relaxed) } >> (r % 64) & 1 != 0
    }

    /// A window store skips `BusDevice`, so it bumps the page's jitv2
    /// generation counter itself, as `PpMemory`'s writes do.
    #[inline(always)]
    fn bump_gen(&self, _phys: u64) {
        #[cfg(feature = "jitv2")]
        {
            let g = unsafe { *self.tc_gen.get() };
            if !g.is_null() {
                // Release, after the bytes (`PpMemory::bump_gen`).
                unsafe { (*g.add((_phys >> 12) as usize)).fetch_add(1, std::sync::atomic::Ordering::Release) };
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
        if s.held.get(self.tag_slot(sel, virt_addr)) == Some(&true) {
            self.store_into_held_line(sel, virt_addr, phys_addr);
        }
        if sel == CACH_PI {
            // The set now holds what the guest wrote, not a refill.
            if let Some(r) = unsafe { &mut *self.refill.get() }.get_mut((virt_addr as usize / IC_LINE.max(1)) % Self::PI_SETS.max(1)) {
                *r = u64::MAX;
            }
        }
    }

    // ---- valid lines hold pages in the CPU's cached view ----

    /// Primary instruction sets, as Index Load Data (I) addresses them.
    const PI_SETS: usize = if IC_LINE == 0 { 0 } else { IC_SIZE / WAYS / IC_LINE };

    fn view_space(&self) -> Option<&PpMemSpace> {
        unsafe { (*self.view_space.get()).as_ref() }
    }

    fn view_no(&self) -> usize {
        unsafe { *self.view.get() }
    }

    /// The physical address and length of the line `tag` names, if it is
    /// valid. `va` is an index operand naming the line's slot: the primary
    /// caches take PA[11:5] or PA[11:6] from it, the secondary PA[17:7] (NEC
    /// U10278EJ3V0UM pp. 285-287; the tag holds the rest). The R5000's
    /// primary data tag has the same PTag and PState fields (NEC
    /// U11761EJ6V0UM p. 133).
    fn line_of(&self, sel: u32, tag: u64, va: u64) -> Option<(u64, usize)> {
        let pa_hi = ((tag >> 32) & 0xf) << 36;
        match sel {
            CACH_PD if DC_LINE != 0 && (tag >> PD_STATE_SHIFT) & 3 != 0 => {
                Some((pa_hi | ((tag >> 8) & 0xff_ffff) << 12 | (va & 0xfff & !(DC_LINE as u64 - 1)), DC_LINE))
            }
            CACH_PI if IC_LINE != 0 && tag & PI_VALID != 0 => {
                Some((pa_hi | ((tag >> 8) & 0xff_ffff) << 12 | (va & 0xfff & !(IC_LINE as u64 - 1)), IC_LINE))
            }
            CACH_SD if L2_LINE != 0 && (tag >> SD_STATE_SHIFT) & 3 != 0 => {
                Some((pa_hi | ((tag >> 14) & 0x3_ffff) << 18 | (va & 0x3_ffff & !(L2_LINE as u64 - 1)), L2_LINE))
            }
            _ => None,
        }
    }

    /// An index operand that names slot `i` of `sel`: `tag_slot`'s inverse.
    fn slot_va(&self, sel: u32, i: usize) -> u64 {
        let line = match sel { CACH_PI => IC_LINE, CACH_PD => DC_LINE, _ => L2_LINE };
        if Self::R5K && matches!(sel, CACH_PI | CACH_PD) {
            let sets = if sel == CACH_PI { Self::IC_LINES } else { Self::DC_LINES } / 2;
            return ((i / sets) << 14 | (i % sets) * line) as u64;
        }
        ((i / WAYS) * line + (i % WAYS)) as u64
    }

    /// A CACHE instruction changed slot `idx` of `sel`'s tag from `old` to
    /// `new`; `va` is the operand that named it. A held line that stops being
    /// valid, or moves, gives its page back; one that becomes valid takes one.
    /// The tags stay the only record of which lines are valid.
    #[cold]
    #[inline(never)]
    fn tag_changes(&self, sel: u32, idx: usize, va: u64, old: u64, new: u64) {
        let Some(sp) = self.view_space() else { return };
        let now = self.line_of(sel, new, va);
        if self.shadow(sel).held[idx] {
            let was = self.line_of(sel, old, va).expect("a held line is valid");
            if now == Some(was) {
                return; // the state changed, not the line
            }
            self.line_leaves(sp, sel, idx, was);
        }
        if let Some(line) = now {
            self.line_enters(sp, sel, idx, va, line);
        }
    }

    /// Slot `idx` of `sel` holds a valid line for `[pa, pa + len)`: its page
    /// becomes private in the CPU's view and the line's bytes are written
    /// into it from the data array, since Index Store Tag writes the tag
    /// alone (NEC VR5000/VR10000 instruction manual pp. 24-25). A line with
    /// no memory behind it holds no page and stays in the arrays, as every
    /// line did before, unless the machine allows lines there.
    fn line_enters(&self, sp: &PpMemSpace, sel: u32, idx: usize, va: u64, (pa, len): (u64, usize)) {
        let unbacked = !sp.backs(pa);
        let Some(new) = sp.hold_line_page(self.view_no(), pa) else { return };
        let base = unsafe { *self.tc_base.get() };
        if new {
            self.mark_foreign(sp, pa, true);
        }
        if new && unbacked {
            // No memory behind the page yet: it starts as what the cache's
            // bus shows there.
            let gran = crate::ppmem::granularity() as u64;
            for a in (pa & !(gran - 1)..(pa & !(gran - 1)) + gran).step_by(8) {
                let r = self.downstream.read64(a);
                unsafe { *(base.add(a as usize) as *mut u64) = if r.is_ok() { r.data.rotate_left(32) } else { 0 } };
            }
        }
        if l2_op_log(CACHE_LOG_OP) {
            crate::dlog!(LogModule::L2c, "shadow:   -> line {pa:#x} holds its page (new: {new})");
        }
        let s = self.shadow(sel);
        s.held[idx] = true;
        if s.data.is_empty() {
            return; // the R5000 has no Index Store Data: the page starts as memory
        }
        let (way, first) = (va & 1, va & !(len as u64 - 1));
        let unit = if sel == CACH_SD { 8 } else { 4 };
        for k in 0..len / unit {
            let word = s.data[self.data_slot(sel, s.data.len(), first + (unit * k) as u64 + way)];
            let at = pa + (unit * k) as u64;
            match sel {
                CACH_PI => self.put_instruction(at, word),
                CACH_PD => unsafe { *(base.add(at as usize) as *mut u32) = word as u32 },
                // TagHi is the doubleword's high word and TagLo its low word
                // (MIPS R10000 UM p. 201; NEC VR10000 UM p. 289, Fig. 13-33).
                // Neither manual says which address each word belongs to.
                // TagLo at the lower address is inferred from shipped PROMs:
                // PROM 6.210 stages code pairs with the word at A in TagLo
                // and the one at A + 4 in TagHi, and its staged code equals
                // its image only this way, as does the IP35 boot-flash
                // PROM's staging. The window keeps a doubleword's words in
                // address order, low half first, so this is the data as is.
                _ => unsafe { *(base.add(at as usize) as *mut u64) = word },
            }
        }
        if sel == CACH_SD {
            self.set_refill(s.tags[idx], pa, len, true);
        }
    }

    /// The line in slot `idx` of `sel` leaves, unwritten, as Index Store Tag
    /// leaves it. With the page's last line the view shows memory there
    /// again. Until then the line's bytes stay in the page.
    fn line_leaves(&self, sp: &PpMemSpace, sel: u32, idx: usize, (pa, len): (u64, usize)) {
        let s = self.shadow(sel);
        s.held[idx] = false;
        if sel == CACH_SD {
            self.set_refill(s.tags[idx], pa, len, false);
        }
        let gone = sp.release_line_page(self.view_no(), pa);
        if gone {
            self.mark_foreign(sp, pa, false);
        }
        if l2_op_log(CACHE_LOG_OP) {
            crate::dlog!(LogModule::L2c, "shadow:   -> line {pa:#x} leaves (page released: {gone})");
        }
    }

    /// ppmem puts `GEN_HELD` on its own generation counters. Where the cache's
    /// bus counts a held page's writes on a counter of its own, which compiles
    /// check instead (the Fuel's reset PROM, mapped over memory), the mark
    /// goes there too, and comes off with the page's last line. A page
    /// outside any counter, or on ppmem's, is skipped.
    #[cfg_attr(not(feature = "jitv2"), allow(unused_variables))]
    fn mark_foreign(&self, sp: &PpMemSpace, pa: u64, held: bool) {
        #[cfg(feature = "jitv2")]
        {
            use crate::cpu::jitv2::jitv2::GEN_HELD;
            use crate::ppmem::MappedMemory;
            use std::sync::atomic::{AtomicU64, Ordering};
            let page = pa & !(crate::ppmem::granularity() as u64 - 1);
            let foreign = unsafe { &mut *self.foreign.get() };
            if held {
                for a in (page..page + crate::ppmem::granularity() as u64).step_by(4096) {
                    let g = self.downstream.gen_ptr(a);
                    let own = unsafe { sp.gen_window_base().add((a >> 12) as usize) as *const AtomicU64 };
                    if !g.is_null() && g != own {
                        unsafe { (*g).fetch_add(GEN_HELD, Ordering::Release) };
                        foreign.push((page, g));
                    }
                }
            } else {
                foreign.retain(|&(p, g)| {
                    if p == page {
                        unsafe { (*g).fetch_add(1u64.wrapping_sub(GEN_HELD), Ordering::Release) };
                    }
                    p != page
                });
            }
        }
    }

    /// Index Store Data into a held line writes through to its page, so the
    /// next cached access sees it. Same quadword rule as the data array.
    fn store_into_held_line(&self, sel: u32, va: u64, value: u64) {
        let idx = self.tag_slot(sel, va);
        let Some((pa, len)) = self.line_of(sel, self.shadow(sel).tags[idx], va) else { return };
        let unit = if sel == CACH_SD { 8 } else { 4 };
        let at = pa + (va & (len as u64 - 1) & !(unit - 1));
        let base = unsafe { *self.tc_base.get() };
        match sel {
            CACH_PI => self.put_instruction(at, value & 0xf_ffff_ffff),
            CACH_PD => unsafe { *(base.add(at as usize) as *mut u32) = value as u32 },
            _ => unsafe {
                *(base.add(at as usize) as *mut u64) = value;
                *(base.add((at ^ 8) as usize) as *mut u64) = 0;
            },
        }
        if sel == CACH_SD {
            self.set_refill(self.shadow(sel).tags[idx], pa, len, true);
        }
    }

    /// A primary instruction word goes into its line's page as the
    /// instruction it predecodes, so a fetch runs it. A word of a class the
    /// model does not know leaves the page as it was, where hardware would
    /// run it.
    fn put_instruction(&self, at: u64, word: u64) {
        match l1i_decode(word) {
            Ok(f) => unsafe { *((*self.tc_base.get()).add(at as usize) as *mut u32) = f },
            Err(class) => crate::dlog!(LogModule::L2c,
                "shadow: primary instruction word {word:#011x} at {at:#x}: predecode class {class:#04x} unknown, memory's word runs"),
        }
    }

    /// A held secondary line `[pa, pa + len)` with tag `tag` is what a primary
    /// instruction refill in each set it maps to would load, or stops being.
    /// The set is the line's virtual index, TagLo[8:7] (VA[13:12], NEC
    /// U10278EJ3V0UM p. 287), over PA[11:6]. Two lines for one set: the
    /// later wins.
    fn set_refill(&self, tag: u64, pa: u64, len: usize, held: bool) {
        let refill = unsafe { &mut *self.refill.get() };
        if !held {
            // By address: the tag may already name another line.
            refill.iter_mut().filter(|r| (pa..pa + len as u64).contains(*r)).for_each(|r| *r = u64::MAX);
            return;
        }
        for half in (pa..pa + len as u64).step_by(IC_LINE.max(1)) {
            if let Some(r) = refill.get_mut(((tag >> 7 & 3) << 12 | (half & 0xfff)) as usize / IC_LINE) {
                *r = half;
            }
        }
    }

    /// Index Load Data (I) in a set a held secondary line maps to, which
    /// Index Store Data (I) has not written since and where no held primary
    /// line is: the word a refill from that
    /// line would have left, predecoded. The model never refills on a fetch,
    /// so this assumes the line was refilled, whichever way the operand
    /// names. PROM 6.210, the IP35 machines' PROM, runs a staged secondary
    /// line and reads the primary words back to stage them itself. Index
    /// Load Tag (I) still
    /// reads the stored tag.
    #[cold]
    #[inline(never)]
    fn refilled_word(&self, va: u64) -> Option<u64> {
        let half = *unsafe { &*self.refill.get() }.get((va as usize / IC_LINE.max(1)) % Self::PI_SETS.max(1))?;
        if half == u64::MAX || self.shadow(CACH_PI).held[self.tag_slot(CACH_PI, va)] {
            return None;
        }
        let at = half + (va & (IC_LINE as u64 - 1) & !3);
        Some(l1i_encode(unsafe { *((*self.tc_base.get()).add(at as usize) as *const u32) }))
    }

    /// How many lines hold pages.
    fn held_count(&self) -> usize {
        [CACH_PI, CACH_PD, CACH_SD].iter().map(|&sel| self.shadow(sel).held.iter().filter(|&&h| h).count()).sum()
    }

    /// Every held line goes, with no writeback: power-on, or a snapshot
    /// restore.
    fn drop_held_lines(&self) {
        let Some(sp) = self.view_space() else { return };
        for sel in [CACH_PI, CACH_PD, CACH_SD] {
            for i in 0..self.shadow(sel).held.len() {
                if self.shadow(sel).held[i] {
                    let line = self.line_of(sel, self.shadow(sel).tags[i], self.slot_va(sel, i)).expect("a held line is valid");
                    self.line_leaves(sp, sel, i, line);
                }
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
    const R10K_CACHE_OPS: bool = Self::R10K;
    const CVT_L_51_BITS: bool = MODEL == model::R10000;
    const FETCH_ALIAS: bool = MODEL == model::R16000;
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
        unsafe { *self.tc_shift.get() = shift };
    }

    unsafe fn set_tcache_fine(&self, table: *const std::sync::atomic::AtomicU64, regions: u64) {
        unsafe {
            *self.tc_fine.get() = table;
            *self.tc_fine_regions.get() = regions;
        }
    }

    fn tcache_key_ptr(&self) -> *mut u64 { self.tc_key.as_ptr() }

    fn tcache_bitmap_ptr(&self) -> *mut u64 { self.tc_bitmap.as_ptr() }

    fn tcache_base_ptr(&self) -> *mut u8 { unsafe { *self.tc_base.get() } }

    unsafe fn set_cached_view(&self, sp: *const PpMemSpace, view: usize) {
        unsafe {
            *self.view_space.get() = sp;
            *self.view.get() = view;
        }
    }

    /// `l2 probe`: the lines holding pages in the CPU's cached view.
    fn debug_probe(&self, _cache_name: &str, _virt_addr: u64, _phys_addr: u64) -> String {
        let pages = self.view_space().map(|sp| sp.held_line_pages(self.view_no())).unwrap_or_default();
        let list: Vec<String> = pages.iter().map(|p| format!("{p:#x}")).collect();
        format!("shadow: {} valid lines hold {} pages in the cached view [{}]",
                self.held_count(), pages.len(), list.join(" "))
    }

    /// Tags only: a fetch reads memory through the decode cache and changes
    /// nothing a guest can see.
    fn fetch_is_pure(&self) -> bool { true }

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
    fn fetch(&self, _virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        // Decoded instructions by physical page — see `decode_cache`.
        let dc = unsafe { &mut *self.decode.get() };
        let word = |a: u64| {
            let r = self.downstream.read32(a);
            if r.is_ok() { Ok(r.data) } else { Err(r.status) }
        };
        let key = self.tc_key.load(Ordering::Relaxed);
        // A page's words straight from the window, if it maps the page.
        let host = |page: u64| match self.ram(page) {
            Some(window) if !window.is_null() => unsafe { window.add(page as usize) as *const u32 },
            _ => std::ptr::null(),
        };
        match dc.fetch(phys_addr, key, IC_LINE, host, word) {
            Ok(p) => FetchInstrResult::hit(p),
            Err(status) => FetchInstrResult::fault(status),
        }
    }

    fn read<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64) -> BusRead64 {
        const {
            assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE")
        };
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

        // Tag operations under `op`, the whole data-array walk only under
        // `hit` as well — see `l2_op_log`.
        if l2_op_log(if matches!(op, C_IST | C_ILT) { CACHE_LOG_OP } else { CACHE_LOG_HIT }) {
            crate::dlog!(LogModule::L2c,
                "shadow: {:<22} raw={cache_op:#04x} va={virt_addr:#018x} arg={phys_addr:#018x}",
                cache_op_name(cache_op));
        }

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
                let old = std::mem::replace(&mut s.tags[idx], r4k_store_taglo(Self::R5K, sel, phys_addr as u32) as u64);
                // The O2's POST keeps its stack in R5000 primary data lines
                // as it does in R10000 ones. An instruction line, with no
                // Index Store Data to fill it, could hold only memory.
                if Self::R5K && sel == CACH_PD && old != s.tags[idx] {
                    self.tag_changes(sel, idx, virt_addr, old, s.tags[idx]);
                }
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
                    let old = s.tags[idx];
                    s.tags[idx] = phys_addr & mask & !MRU_BIT;
                    if Self::R10K && matches!(sel, CACH_PI | CACH_PD | CACH_SD) && old != s.tags[idx] {
                        self.tag_changes(sel, idx, virt_addr, old, s.tags[idx]);
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
                    if sel == CACH_PI {
                        if let Some(p) = self.refilled_word(virt_addr) {
                            return p;
                        }
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
                    let old = std::mem::take(&mut s.tags[idx]);
                    self.tag_changes(sel, idx, virt_addr, old, 0);
                }
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
        self.drop_held_lines();
        for sel in [CACH_PI, CACH_PD, CACH_SD] {
            let s = self.shadow(sel);
            s.tags.fill(0);
            s.mru.fill(false);
        }
        unsafe { (*self.decode.get()).flush() };
    }

    /// The shadow arrays are not snapshotted (nothing reads them but
    /// diagnostics). Lines holding pages are, as a count: a PROM's cache
    /// phase is the only time any are held, and a snapshot of it would
    /// restore without them, so it is refused.
    fn save_cache_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        let n = self.held_count();
        if n > 0 {
            t.insert("held_lines".into(), toml::Value::Integer(n as i64));
        }
        toml::Value::Table(t)
    }

    /// A restore replaces RAM wholesale; drop every decoded instruction and
    /// every held line.
    fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        if let Some(n) = crate::snapshot::get_field(v, "held_lines") {
            return Err(format!("cache: the snapshot was taken with {n} cache lines holding memory, which it does not carry"));
        }
        unsafe { (*self.decode.get()).flush() };
        self.drop_held_lines();
        Ok(())
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

    fn downstream(&self) -> Arc<dyn BusDevice> {
        self.downstream.clone()
    }

    fn cas_ram32(&self, phys: u64, old: u32, new: u32) -> Option<bool> {
        use std::sync::atomic::{AtomicU32, Ordering};
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
/// The Octane's R14000 is this type with its own PRId set at build time (see
/// `Part::prid`), as the Octane's own R10000 revision is; only the PRId
/// differs. Both report "R12000" as `NAME`.
pub type R12000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 128, true, 0x0000_0e35, 0x0000_0900, 64, { model::R12000 }>;

/// R12000 as the Origin 200's 270 MHz module has it: the Octane part with the
/// module's 4 MB secondary cache (its Config word says SS = 3). The stock IP27
/// PROM's secondary-cache tag test fails on the Octane type's 2 MB. The 128-byte
/// lines are the Octane part's, not read from the module.
pub type R12000Module4MShadowCache =
    ShadowCache<32768, 64, 32768, 32, 4194304, 128, true, 0x0000_0e35, 0x0000_0900, 64, { model::R12000 }>;

/// The R16000 with a 4 MB secondary cache, the Tezro's 700 and 800 MHz parts:
/// the R12000's geometry and programming model with the larger secondary (SS
/// code 3 in Config) and 128-byte secondary lines (SB: a model choice). The
/// PRId here is a placeholder: the part's own, `Part::prid`, is set at build
/// time, as for the R14000. FIR 0x0900, the one R14000 value on record; a
/// model choice: no R16000 FIR was found in the retained evidence.
pub type R16000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 4194304, 128, true, 0x0000_0f41, 0x0000_0900, 64, { model::R16000 }>;

/// The R16000 with an 8 MB secondary cache, the 900 MHz part's: the 4 MB type
/// with twice the secondary (SS code 4). The size is a type parameter so the
/// executor, the translator and the cache know it when they are built, and no
/// run-time field is read to find it.
pub type R16000Scache8ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 8388608, 128, true, 0x0000_0f50, 0x0000_0900, 64, { model::R16000 }>;

/// The R16000 with a 16 MB secondary cache, the 1 GHz part's (SS code 5);
/// otherwise the 4 MB type.
pub type R16000Scache16ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 16777216, 128, true, 0x0000_0f50, 0x0000_0900, 64, { model::R16000 }>;

/// R12000 3.5 as the O2 has it: the Octane part with 64-byte secondary lines
/// (SB clear). With 128-byte lines the O2 PROM panics before its banner
/// (write TLB miss at 0x0); 64 boots. IRIX on the O2 never reads SB and sizes
/// the secondary cache as 512 KB << SS.
pub type R12000O2ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 64, true, 0x0000_0e35, 0x0000_0900, 64, { model::R12000 }>;

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

    /// A secondary Index Store Data writes a quadword: the other doubleword
    /// becomes zero and shares the ten check bits (R10000 UM p. 91).
    #[test]
    fn nec_layout_stores_a_quadword_with_ten_check_bits() {
        let c = cache();
        c.set_cache_ecc(0);
        c.cache_op(C_R10K_ISD | CACH_SD, 8, 0xdead_beef);
        c.set_cache_ecc(0xffff_fc01);
        c.cache_op(C_R10K_ISD | CACH_SD, 0, 0x1111_2222_3333_4444);
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

    /// On the R12000 and later Index Writeback Invalidate (D) zeroes the line's
    /// primary data tag; the R10000 keeps it.
    #[test]
    fn index_writeback_invalidate_d_clears_the_tag_from_the_r12000_on() {
        fn run<C: MipsCache + From<Arc<dyn BusDevice>>>() -> u64 {
            let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
            let c = C::from(mem);
            c.cache_op(crate::cpu::mips_cache_v2::C_IST | CACH_PD, 0x40, 0x1_2345_6780);
            c.cache_op(crate::cpu::mips_cache_v2::C_IWBINV | CACH_PD, 0x40, 0);
            c.cache_op(crate::cpu::mips_cache_v2::C_ILT | CACH_PD, 0x40, 0)
        }
        assert_eq!(run::<R12000ShadowCache>(), 0);
        assert_eq!(run::<R10000ShadowCache>(), 0x1_2345_6780);
        assert_eq!(run::<R16000ShadowCache>(), 0, "the R16000 follows the R12000 and R14000");
    }

    /// The three R16000 types differ in the secondary cache alone: size, the
    /// SS code Config reads from it, and the placeholder PRId each starts
    /// with; the PRId a part really presents is `Part::prid`'s.
    #[test]
    fn the_r16000_types_differ_only_in_the_secondary_cache() {
        fn id<C: CpuModel + MipsCache>() -> (u32, u32, usize, usize, &'static str, u32, u32, bool) {
            (C::PRID, C::FIR, C::L2_SIZE, C::L2_LINE, C::NAME, C::VA_BITS, C::PA_BITS, C::R10K_CACHE_OPS)
        }
        assert_eq!(id::<R16000ShadowCache>(), (0xf41, 0x900, 4 << 20, 128, "R16000", 44, 40, true));
        assert_eq!(id::<R16000Scache8ShadowCache>(), (0xf50, 0x900, 8 << 20, 128, "R16000", 44, 40, true));
        assert_eq!(id::<R16000Scache16ShadowCache>(), (0xf50, 0x900, 16 << 20, 128, "R16000", 44, 40, true));
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

    /// The R10000 reads the reserved bits of a secondary tag as zero (R10000 UM
    /// p. 270); the R4400 stores the tag as written.
    #[test]
    fn the_r10000_masks_secondary_tags() {
        fn stored<C: CpuModel + MipsCache + From<Arc<dyn BusDevice>>>(tag: u64) -> u64 {
            let c = C::from(Arc::new(Memory::new(1)) as Arc<dyn BusDevice>);
            c.cache_op(C_IST | CACH_SD, 0, tag);
            c.cache_op(C_ILT | CACH_SD, 0, 0)
        }
        assert_eq!(stored::<R10000ShadowCache>(u64::MAX), 0x8000_000f_ffff_cdff);
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

    // ---- valid lines hold pages in the CPU's cached view ----

    use crate::ppmem::{MappedMemory, PpMemory};

    /// A cache reading memory through a cached view of a fresh window, with
    /// a 64 MB bank mapped at 0. Its bus is the bank itself: memory as the
    /// uncached path sees it. Boxed: the view publishes into the cache's
    /// inline bitmap field.
    fn windowed() -> (Box<R10000ShadowCache>, Box<PpMemSpace>, Arc<PpMemory>) {
        let (space, mut banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
        let space = Box::new(space);
        let bank = Arc::new(banks.remove(0));
        let c = Box::new(R10000ShadowCache::from(bank.clone() as Arc<dyn BusDevice>));
        attach(&c, &space);
        space.map_bank(0, 0, 64 << 20, 64 << 20).unwrap();
        (c, space, bank)
    }

    fn attach(c: &R10000ShadowCache, space: &PpMemSpace) -> usize {
        let v = space.add_cached_view(&[c.tcache_bitmap_ptr()]).unwrap();
        unsafe {
            c.set_tcache_window(space.view_base(v));
            c.set_cached_view(space, v);
        }
        v
    }

    /// The page a line for `pa` holds: the host granule around it, which is
    /// 16 KB on Apple silicon and 64 KB on Windows.
    fn page(pa: u64) -> u64 { pa & !(crate::ppmem::granularity() as u64 - 1) }

    /// TagLo of a DirtyExclusive primary data line for `pa`'s page.
    fn pd_dirty(pa: u64) -> u64 { (pa >> 12) << 8 | 3 << PD_STATE_SHIFT }

    /// TagLo of a secondary line for `pa` in `state`.
    fn sd_tag(pa: u64, state: u64) -> u64 { (pa >> 18) << 14 | state << SD_STATE_SHIFT }

    /// The O2's POST: a valid tag for page 0, the page tested uncached while
    /// the cached side keeps its stack, and an invalid tag that drops the
    /// line without a writeback.
    #[test]
    fn a_valid_line_gives_the_cached_side_its_own_page() {
        let (c, space, bank) = windowed();
        bank.write32(0x40, 0x1111_1111);
        c.cache_op(C_R10K_ISD | CACH_PD, 0x8000_0040, 0x1111_1111);
        c.cache_op(C_IST | CACH_PD, 0x8000_0040, pd_dirty(0));
        assert_eq!(space.held_line_pages(0), vec![0]);
        assert_eq!(c.read::<4>(0, 0x40).data, 0x1111_1111);
        c.write::<4>(0, 0x40, 0x2222_2222);
        bank.write32(0x40, 0x3333_3333);
        assert_eq!(c.read::<4>(0, 0x40).data, 0x2222_2222, "cached: the line");
        assert_eq!(bank.read32(0x40).data, 0x3333_3333, "uncached: memory");
        c.cache_op(C_IST | CACH_PD, 0x8000_0040, 0);
        assert!(space.held_line_pages(0).is_empty());
        assert_eq!(c.read::<4>(0, 0x40).data, 0x3333_3333, "dropped, not written back");
    }

    /// Index Store Tag writes the tag alone, so the line holds what Index
    /// Store Data put in the data array before, and Index Store Data after
    /// writes through. Index Load Data reads the array.
    #[test]
    fn a_line_holds_its_data_array_and_takes_later_stores() {
        let (c, _space, bank) = windowed();
        bank.write32(0x2004, 0x5555_5555);
        c.cache_op(C_R10K_ISD | CACH_PD, 0x8000_0004, 0xdead_beef);
        c.cache_op(C_IST | CACH_PD, 0x8000_0000, pd_dirty(0x2000));
        assert_eq!(c.read::<4>(0, 0x2004).data, 0xdead_beef);
        c.cache_op(C_R10K_ISD | CACH_PD, 0x8000_000c, 0x0123_4567);
        assert_eq!(c.read::<4>(0, 0x200c).data, 0x0123_4567);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PD, 0x8000_000c, 0), 0x0123_4567);
        assert_eq!(bank.read32(0x2004).data, 0x5555_5555, "memory never saw the line");
    }

    /// A secondary line takes a doubleword at a time with TagLo at the lower
    /// address. That is how the shipped PROMs stage code: the instruction at
    /// A in TagLo and the one at A + 4 in TagHi, then Index Store Data (S),
    /// and cached accesses see them in that order. Index Store Data (S) also
    /// zeroes the other doubleword of its quadword (R10000 UM p. 202).
    #[test]
    fn a_secondary_line_runs_code_staged_the_way_the_proms_stage_it() {
        let (c, space, _bank) = windowed();
        let (at_a, at_a4) = (0x2404_0002u64, 0x03e0_0008u64); // addiu $4,$0,2; jr $31
        c.cache_op(C_R10K_ISD | CACH_SD, 0x8000_3008, at_a4 << 32 | at_a);
        c.cache_op(C_IST | CACH_SD, 0x8000_3000, sd_tag(0x3000, 2));
        assert_eq!(space.held_line_pages(0), vec![page(0x3000)]);
        assert_eq!(c.read::<4>(0, 0x3008).data, at_a, "the word at A");
        assert_eq!(c.read::<4>(0, 0x300c).data, at_a4, "the word at A + 4");
        c.cache_op(C_R10K_ISD | CACH_SD, 0x8000_3000, 0x9999);
        assert_eq!(c.read::<8>(0, 0x3000).data, 0x9999 << 32);
        assert_eq!(c.read::<8>(0, 0x3008).data, 0, "the partner doubleword is zeroed");
    }

    /// A tag test's patterns name no memory: they hold nothing, and stay
    /// in the arrays as before.
    #[test]
    fn a_line_with_no_memory_behind_holds_no_page() {
        let (c, space, _bank) = windowed();
        c.cache_op(C_IST | CACH_SD, 0x8000_0000, sd_tag(0x55_5554_0000, 3) | 0x5);
        c.cache_op(C_IST | CACH_PD, 0x8000_0000, pd_dirty(0x8000_0000));
        assert!(space.held_line_pages(0).is_empty());
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 0x8000_0000, 0) & 0xffff_ffff, (sd_tag(0x55_5554_0000, 3) | 0x5) & 0xffff_ffff);
    }

    /// The R4400 PROMs keep nothing in their caches: a valid tag there holds
    /// no page, and none of this runs for them.
    #[test]
    fn an_r4400_tag_holds_nothing() {
        let (space, mut banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
        let bank = Arc::new(banks.remove(0));
        let c = Box::new(R4400ShadowCache::from(bank as Arc<dyn BusDevice>));
        let v = space.add_cached_view(&[c.tcache_bitmap_ptr()]).unwrap();
        unsafe {
            c.set_tcache_window(space.view_base(v));
            c.set_cached_view(&space, v);
        }
        space.map_bank(0, 0, 64 << 20, 64 << 20).unwrap();
        c.cache_op(C_IST | CACH_PD, 0x8000_2000, 0x2 << 8 | 3 << 6);
        c.cache_op(C_IST | CACH_SD, 0x8000_2000, 0x1 << 13 | 5 << 10);
        assert!(space.held_line_pages(v).is_empty());
    }

    /// An R5000 cache on a cached view of a fresh window with a 64 MB bank
    /// at 0, as `windowed` builds an R10000 one.
    fn r5000_windowed() -> (Box<R5000ShadowCache>, Box<PpMemSpace>, Arc<PpMemory>) {
        let (space, mut banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
        let space = Box::new(space);
        let bank = Arc::new(banks.remove(0));
        let c = Box::new(R5000ShadowCache::from(bank.clone() as Arc<dyn BusDevice>));
        let v = space.add_cached_view(&[c.tcache_bitmap_ptr()]).unwrap();
        unsafe {
            c.set_tcache_window(space.view_base(v));
            c.set_cached_view(&*space, v);
        }
        space.map_bank(0, 0, 64 << 20, 64 << 20).unwrap();
        (c, space, bank)
    }

    /// The O2's POST on the R5000, as the shipped PROM runs it on iris:
    /// Index Store Tag (D) with TagLo 0xc0 (page 0, Valid) at every index of
    /// 0x8000_0000..=0x8000_1000, the stack stored cached, page 0 tested
    /// uncached, then an invalid tag at each index. The cached side keeps
    /// the stack until the last line goes, which drops it unwritten. The
    /// R5000 has no Index Store Data, so the page starts as memory.
    #[test]
    fn an_r5000_primary_data_line_gives_the_cached_side_its_own_page() {
        let (c, space, bank) = r5000_windowed();
        bank.write32(0xff4, 0x1111_1111);
        for va in (0x8000_0000..=0x8000_1000u64).step_by(32) {
            c.cache_op(C_IST | CACH_PD, va, 0xc0);
        }
        assert_eq!(space.held_line_pages(0), vec![0]);
        assert_eq!(c.read::<4>(0, 0xff4).data, 0x1111_1111, "the page starts as memory");
        c.write::<4>(0, 0xff4, 0xbfc0_4400);
        bank.write32(0xff4, 0xa000_0ff0);
        assert_eq!(c.read::<4>(0, 0xff4).data, 0xbfc0_4400, "cached: the stack");
        assert_eq!(bank.read32(0xff4).data, 0xa000_0ff0, "uncached: the memory test");
        assert_eq!(c.cache_op(C_ILT | CACH_PD, 0x8000_0fe0, 0) & 0xc0, 0xc0, "the tag reads back");
        for va in (0x8000_0020..=0x8000_1000u64).step_by(32) {
            c.cache_op(C_IST | CACH_PD, va, 0);
        }
        assert_eq!(space.held_line_pages(0), vec![0], "until its last line goes");
        c.cache_op(C_IST | CACH_PD, 0x8000_0000, 0);
        assert!(space.held_line_pages(0).is_empty());
        assert_eq!(c.read::<4>(0, 0xff4).data, 0xa000_0ff0, "dropped, not written back");
    }

    /// The R5000's second way, VA[14], holds a page like the first, and
    /// power-on drops it. Instruction lines hold nothing: with no Index
    /// Store Data, one could hold only what memory holds. There is no
    /// secondary.
    #[test]
    fn an_r5000_line_in_either_way_holds_and_power_on_drops_it() {
        let (c, space, _bank) = r5000_windowed();
        c.cache_op(C_IST | CACH_PI, 0x8000_0000, 0x3 << 8 | 0x80);
        c.cache_op(C_IST | CACH_SD, 0x8000_0000, 0x1 << 13 | 5 << 10);
        assert!(space.held_line_pages(0).is_empty());
        c.cache_op(C_IST | CACH_PD, 0x8000_4fe0, 0x5 << 8 | 0xc0);
        assert_eq!(space.held_line_pages(0), vec![page(0x5000)]);
        assert_eq!(c.cache_op(C_ILT | CACH_PD, 0x8000_0fe0, 0) & 0xc0, 0, "way 0 is untouched");
        c.power_on();
        assert!(space.held_line_pages(0).is_empty());
        assert!(c.debug_probe("l2", 0, 0).contains("0 valid lines hold 0 pages"));
    }

    /// Two CPUs running the same PROM keep their stacks in lines for the
    /// same physical page, with different contents, and neither sees the
    /// other's.
    #[test]
    fn two_cpus_hold_the_same_page_apart() {
        let (space, mut banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
        let space = Box::new(space);
        let bank = Arc::new(banks.remove(0));
        let a = Box::new(R10000ShadowCache::from(bank.clone() as Arc<dyn BusDevice>));
        let b = Box::new(R10000ShadowCache::from(bank.clone() as Arc<dyn BusDevice>));
        attach(&a, &space);
        attach(&b, &space);
        space.map_bank(0, 0, 64 << 20, 64 << 20).unwrap();
        bank.write32(0x10_0040, 0x1234_5678);
        for c in [&a, &b] {
            c.cache_op(C_R10K_ISD | CACH_PD, 0x8000_0040, 0x1234_5678);
            c.cache_op(C_IST | CACH_PD, 0x8000_0040, pd_dirty(0x10_0000));
        }
        a.write::<4>(0, 0x10_0040, 0xaaaa_aaaa);
        b.write::<4>(0, 0x10_0040, 0xbbbb_bbbb);
        assert_eq!(a.read::<4>(0, 0x10_0040).data, 0xaaaa_aaaa);
        assert_eq!(b.read::<4>(0, 0x10_0040).data, 0xbbbb_bbbb);
        assert_eq!(bank.read32(0x10_0040).data, 0x1234_5678, "memory is neither");
        assert_eq!(unsafe { *(space.window_base().add(0x10_0040) as *const u32) }, 0x1234_5678,
                   "the shared window, which other masters read, is memory");
        a.cache_op(C_IST | CACH_PD, 0x8000_0040, 0);
        assert_eq!(a.read::<4>(0, 0x10_0040).data, 0x1234_5678);
        assert_eq!(b.read::<4>(0, 0x10_0040).data, 0xbbbb_bbbb);
    }

    /// Two lines of one page: the page stays until the last goes. The R12000
    /// zeroing a primary data tag at Index Writeback Invalidate ends a line
    /// like an invalid tag store.
    #[test]
    fn a_page_goes_with_its_last_line() {
        let (space, mut banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
        let space = Box::new(space);
        let bank = Arc::new(banks.remove(0));
        let c = Box::new(R12000ShadowCache::from(bank.clone() as Arc<dyn BusDevice>));
        let v = space.add_cached_view(&[c.tcache_bitmap_ptr()]).unwrap();
        unsafe {
            c.set_tcache_window(space.view_base(v));
            c.set_cached_view(&*space, v);
        }
        space.map_bank(0, 0, 64 << 20, 64 << 20).unwrap();
        bank.write32(0x6000, 0x600d);
        c.cache_op(C_IST | CACH_PD, 0x8000_6000, pd_dirty(0x6000));
        c.cache_op(C_IST | CACH_PD, 0x8000_6040, pd_dirty(0x6000));
        c.write::<4>(0, 0x6000, 0xbad);
        c.cache_op(C_IST | CACH_PD, 0x8000_6000, 0);
        assert_eq!(space.held_line_pages(v), vec![page(0x6000)]);
        assert_eq!(c.read::<4>(0, 0x6000).data, 0xbad, "the page stays the view's");
        c.cache_op(C_IWBINV | CACH_PD, 0x8000_6040, 0);
        assert!(space.held_line_pages(v).is_empty());
        assert_eq!(c.read::<4>(0, 0x6000).data, 0x600d);
    }

    /// The R16000 parts keep the R10000 family's index operations: a valid
    /// primary data line or secondary line holds its page in the CPU's view,
    /// with 4, 8 and 16 MiB secondary caches, and Index Store Data fills it.
    #[test]
    fn the_r16000_types_hold_lines_in_the_view() {
        fn run<C: MipsCache + From<Arc<dyn BusDevice>>>() {
            let (space, mut banks) = PpMemSpace::with_bank_sizes(&[64]).unwrap();
            let space = Box::new(space);
            let bank = Arc::new(banks.remove(0));
            let c = Box::new(C::from(bank.clone() as Arc<dyn BusDevice>));
            let v = space.add_cached_view(&[c.tcache_bitmap_ptr()]).unwrap();
            unsafe {
                c.set_tcache_window(space.view_base(v));
                c.set_cached_view(&*space, v);
            }
            space.map_bank(0, 0, 64 << 20, 64 << 20).unwrap();
            bank.write32(0x2004, 0x5555_5555);
            c.cache_op(C_R10K_ISD | CACH_PD, 0x8000_0004, 0xdead_beef);
            c.cache_op(C_IST | CACH_PD, 0x8000_0000, pd_dirty(0x2000));
            assert_eq!(space.held_line_pages(v), vec![0x2000]);
            assert_eq!(c.read::<4>(0, 0x2004).data, 0xdead_beef);
            assert_eq!(bank.read32(0x2004).data, 0x5555_5555, "memory never saw the line");
            c.cache_op(C_R10K_ISD | CACH_SD, 0x8000_3008, 0x1111_2222_3333_4444);
            c.cache_op(C_IST | CACH_SD, 0x8000_3000, sd_tag(0x3000, 2));
            assert_eq!(space.held_line_pages(v), vec![0x2000, 0x3000]);
            assert_eq!(c.read::<4>(0, 0x3008).data, 0x3333_4444, "TagLo at the lower address");
            c.cache_op(C_IST | CACH_PD, 0x8000_0000, 0);
            c.cache_op(C_IST | CACH_SD, 0x8000_3000, 0);
            assert!(space.held_line_pages(v).is_empty());
        }
        run::<R16000ShadowCache>();
        run::<R16000Scache8ShadowCache>();
        run::<R16000Scache16ShadowCache>();
    }

    /// Power-on drops every held page. A snapshot taken while lines hold
    /// pages is refused at restore.
    #[test]
    fn power_on_drops_held_pages_and_a_snapshot_of_them_is_refused() {
        let (c, space, _bank) = windowed();
        c.cache_op(C_IST | CACH_PD, 0x8000_0000, pd_dirty(0x7000));
        let saved = c.save_cache_state();
        assert!(c.load_cache_state(&saved).is_err());
        c.power_on();
        assert!(space.held_line_pages(0).is_empty());
        assert!(c.debug_probe("l2", 0, 0).contains("0 valid lines hold 0 pages"));
        assert!(c.load_cache_state(&c.save_cache_state()).is_ok());
    }

    // ---- primary instruction lines ----

    /// Every class of the table, encoded and decoded back, one field moved
    /// at a time for the example word, and what the table refuses.
    #[test]
    fn predecoded_words_round_trip() {
        let p = 0x3_4420_0004; // addiu $4,$0,2
        for (q, f) in [(p, 0x2404_0002), (p | 0x1f << 5, 0x2404_f802), (p | 0x1f << 10, 0x27e4_0002), (p | 0x1f << 15, 0x2404_07c2)] {
            assert_eq!(l1i_decode(q), Ok(f), "{q:#x}");
        }
        for (class, f) in [(0x08, 0x0009_20f8), (0x08, 0x0009_20fc), (0x18, 0x0109_5025), (0x1a, 0x2084_0010),
                           (0x1a, 0x3484_0010), (0x1a, 0x3c04_8000), (0x1e, 0x6484_0010), (0x35, 0xbd04_0040),
                           (0x36, 0xdd04_0008), (0x3f, 0x4084_6000), (0x40, 0x0c00_1234), (0x41, 0x03e0_0008),
                           (0x41, 0x0800_1234), (0x71, 0x0481_0004), (0, 0)] {
            assert_eq!((l1i_encode(f) >> 29, l1i_decode(l1i_encode(f))), (class, Ok(f)), "{f:#x}");
        }
        for f in [0x0009_20c0, 0x2884_0010, 0x4422_3000] { // sll, slti, mtc1: no class
            assert_eq!((l1i_encode(f), l1i_decode(L1I_RAW | f as u64)), (L1I_RAW | f as u64, Ok(f)));
        }
        assert_eq!((l1i_decode(1), l1i_decode(0x2a << 29)), (Err(0), Err(0x2a)));
    }

    /// TagLo of a valid primary instruction line for `pa`'s page.
    fn pi_valid(pa: u64) -> u64 { (pa >> 12) << 8 | PI_VALID }

    /// A valid primary instruction line puts the instructions its words
    /// predecode into the page, leaving memory's word for one it cannot
    /// decode, and Index Store Data (I) writes through.
    #[test]
    fn a_primary_instruction_line_puts_its_instructions_in_the_page() {
        let (c, space, bank) = windowed();
        bank.write32(0x9008, 0x1111_1111);
        let va = 0x8000_9001u64; // way 1
        c.cache_op(C_R10K_ISD | CACH_PI, va, 0x3_4420_0004);
        c.cache_op(C_R10K_ISD | CACH_PI, va + 8, 0x2a << 29);
        c.cache_op(C_IST | CACH_PI, va, pi_valid(0x9000));
        assert_eq!(space.held_line_pages(0), vec![0x9000]);
        assert_eq!([0x9000, 0x9008].map(|a| c.read::<4>(0, a).data), [0x2404_0002, 0x1111_1111]);
        c.cache_op(C_R10K_ISD | CACH_PI, va + 4, l1i_encode(0x03e0_0008));
        assert_eq!(c.read::<4>(0, 0x9004).data, 0x03e0_0008);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, va, 0), 0x3_4420_0004, "Index Load Data (I) reads the array");
        assert_eq!(bank.read32(0x9000).data, 0, "memory never saw it");
        c.cache_op(C_IST | CACH_PI, va, 0);
        assert!(space.held_line_pages(0).is_empty());
    }

    /// A held secondary line answers Index Load Data (I) in the primary sets
    /// it maps to, as the predecoded words a refill would leave, until Index
    /// Store Data (I) writes the set or the line goes. A held primary line
    /// answers for itself.
    #[test]
    fn index_load_data_i_reads_a_refill_until_the_guest_writes_the_set() {
        let (c, _space, _bank) = windowed();
        c.cache_op(C_IST | CACH_SD, 0x8000_4f00, sd_tag(0x4f00, 2) | 1 << 7); // VA[13:12] = 1
        c.cache_op(C_R10K_ISD | CACH_SD, 0x8000_4f10, 0x0140_0008); // jr $10 at 0x4f10
        let set = |off: u64| 0x8000_1f00 | off;
        for va in [set(0x10), set(0x10) | 1] {
            assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, va, 0), l1i_encode(0x0140_0008), "either way");
        }
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, 0x8000_0f10, 0), 0, "another set: the array");
        c.cache_op(C_R10K_ISD | CACH_PI, set(0x10), 0x5_5555_5555);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, set(0x10), 0), 0x5_5555_5555, "the guest's pattern");
        c.cache_op(C_R10K_ISD | CACH_SD, 0x8000_4f10, 0x0140_0008);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, set(0x10), 0), l1i_encode(0x0140_0008), "restaged: a refill again");
        c.cache_op(C_R10K_ISD | CACH_PI, set(0x10), 0x3_4420_0004);
        c.cache_op(C_IST | CACH_PI, set(0x10), pi_valid(0x1_4000));
        c.cache_op(C_R10K_ISD | CACH_SD, 0x8000_4f10, 0x0140_0008);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, set(0x10), 0), 0x3_4420_0004, "a held primary line answers for itself");
        c.cache_op(C_IST | CACH_PI, set(0x10), 0);
        c.cache_op(C_IST | CACH_SD, 0x8000_4f00, 0);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_PI, set(0x10), 0), 0x3_4420_0004, "gone with the line: the array");
    }
}
