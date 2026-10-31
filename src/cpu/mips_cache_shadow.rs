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
//! - **Diagnostics round-trip by default.** The shadow stores whatever bits
//!   are written to it and returns them unchanged, so walking-1s tests pass
//!   without knowing the hardware's field layout. The R14000 alone masks its
//!   secondary tags (`R14000_L2_TAG_MASK`).
//!
//! The shadow is deliberately not consulted by `read`/`write`/`fetch`. If it
//! ever needs to be, this type is the wrong shape and should say so loudly
//! rather than growing a slow path.

use std::cell::UnsafeCell;
use std::sync::Arc;

use crate::cpu::mips_cache_v2::{
    cache_op_name, model, r4k_load_taglo, r4k_store_taglo, CpuModel, FetchInstrResult, MipsCache,
    C_ILT, C_IST, C_R10K_CBARRIER, C_R10K_ILD, C_R10K_ISD, CACH_PD, CACH_PI, CACH_SD, CACH_SI,
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

/// Bits an R14000 secondary-cache tag retains: MRU, physical tag, state,
/// virtual index and ECC, from the R10000 tag format in NEC U10278EJ4V0UM
/// p.270, figure 11-31. It is a compatibility choice for the Fuel PROM's tag
/// check, not a measurement of R14000 silicon. A 36-bit mask once inferred
/// from the IP28 PROM's walking-0s phase broke its diagnostics, so the other
/// parts keep every bit and extending this one needs IP28 and Octane
/// diagnostic comparisons.
const R14000_L2_TAG_MASK: u64 = 0x8000_000f_ffff_cdff;

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
    #[cfg(feature = "jitv2")]
    tc_gen: UnsafeCell<*mut std::sync::atomic::AtomicU64>,
    /// Decoded instructions, by physical page. Never touches the shadow
    /// arrays above — see `decode_cache`.
    decode: UnsafeCell<crate::cpu::decode_cache::DecodeCache>,
    /// See [`MipsCache::set_pd_tag_store_hook`]. Consulted by `cache_op` only.
    pd_tag_hook: UnsafeCell<Option<Box<dyn Fn(u32) + Send + Sync>>>,
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
            // The R10000 secondary keeps a data shadow: it is the one the IP28
            // PROM walks with Index_Store_Data, and a 1 MB array is cheap
            // once. The R4000 family has no index data ops. The primaries keep
            // arrays too, for a PROM's primary-cache data walk.
            ic: UnsafeCell::new(Shadow::new(Self::IC_LINES, if Self::R10K { IC_SIZE / 4 } else { 0 })),
            dc: UnsafeCell::new(Shadow::new(Self::DC_LINES, if Self::R10K { DC_SIZE / 4 } else { 0 })),
            l2: UnsafeCell::new(Shadow::new(Self::L2_LINES, if Self::R10K { L2_SIZE / 8 } else { 0 })),
            tc_base: UnsafeCell::new(std::ptr::null_mut()),
            tc_bitmap: UnsafeCell::new(0),
            #[cfg(feature = "jitv2")]
            tc_gen: UnsafeCell::new(std::ptr::null_mut()),
            decode: UnsafeCell::new(crate::cpu::decode_cache::DecodeCache::new()),
            pd_tag_hook: UnsafeCell::new(None),
        }
    }

    /// ppmem's window base if `phys` is RAM the window maps, as the region
    /// bitmap says; `None` for the PROM, MMIO and the low-512KB alias, which go
    /// through the bus. The bitmap stays zero until the window is wired, so a
    /// set bit also means the base is valid.
    #[inline(always)]
    fn ram(&self, phys: u64) -> Option<*mut u8> {
        if crate::ppmem::bitmap_maps(unsafe { *self.tc_bitmap.get() }, phys) {
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
    /// primary and eight-byte secondary slots, a diagnostic compatibility
    /// approximation of the NEC layout.
    fn data_slot(&self, sel: u32, len: usize, virt_addr: u64) -> usize {
        if len == 0 {
            return 0;
        }
        let way = (virt_addr as usize) & (WAYS - 1);
        let shift = if matches!(sel, CACH_PI | CACH_PD) { 2 } else { 3 };
        let word = (virt_addr as usize) >> shift;
        (word * WAYS + way) % len
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
    // implements 40; key this off the same flag that selects its cache
    // encodings.
    const VA_BITS: u32 = if Self::R10K { 44 } else { 40 };
    const PA_BITS: u32 = if Self::R10K { 40 } else { 36 };
    // The CPU's name, which snapshots check: the same as the full model's, so
    // a mismatch reads as a cache-kind difference rather than a different CPU.
    // The R10000 keeps its historical "shadow", which existing IP28
    // snapshots carry.
    const NAME: &'static str = match MODEL {
        model::R4400 => "R4400",
        model::R5000 => "R5000",
        model::R10000 => "shadow",
        model::R12000 => "R12000",
        model::R14000 => "R14000",
        _ => "R16000",
    };
    const CACHE_KIND: &'static str = "shadow";
    const R10K_CACHE_OPS: bool = Self::R10K;
    const SECONDARY_TAG_MASK: u64 = if MODEL == model::R14000 { R14000_L2_TAG_MASK } else { u64::MAX };
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

    fn tcache_bitmap_ptr(&self) -> *mut u64 { self.tc_bitmap.get() }

    fn tcache_base_ptr(&self) -> *mut u8 { unsafe { *self.tc_base.get() } }

    fn set_pd_tag_store_hook(&self, hook: Box<dyn Fn(u32) + Send + Sync>) {
        unsafe { *self.pd_tag_hook.get() = Some(hook) };
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
                    s.tags[idx] = phys_addr & mask & !MRU_BIT;
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
                let s = self.shadow(sel);
                let slot = self.data_slot(sel, s.data.len(), virt_addr);
                if !s.data.is_empty() {
                    // Primary diagnostic fields: NEC U10278E fourth edition,
                    // pp. 262, 271-272. Preserve supplied predecode/parity.
                    let (data_mask, ecc_mask) = match sel {
                        CACH_PI => (0xf_ffff_ffff, 1),
                        CACH_PD => (0xffff_ffff, 0xf),
                        _ => (u64::MAX, 0x3ff),
                    };
                    s.data[slot] = phys_addr & data_mask;
                    s.ecc[slot] = unsafe { *self.ecc_in.get() } & ecc_mask;
                    if matches!(sel, CACH_SI | CACH_SD) {
                        // Index Store Data(S) writes a quadword: selected VA[3]
                        // doubleword plus a zero partner; check bits are shared.
                        // NEC U10278E fourth edition p.91, figure 5-7.
                        let partner = slot ^ WAYS;
                        s.data[partner] = 0;
                        s.ecc[partner] = s.ecc[slot];
                    }
                    if l2_op_log(CACHE_LOG_HIT) {
                        crate::dlog!(LogModule::L2c, "shadow:   -> ISD slot={slot} data={phys_addr:#018x} ecc={:#x}",
                                     s.ecc[slot]);
                    }
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
        for sel in [CACH_PI, CACH_PD, CACH_SD] {
            let s = self.shadow(sel);
            s.tags.fill(0);
            s.mru.fill(false);
        }
        unsafe { (*self.decode.get()).flush() };
    }

    /// A restore replaces RAM wholesale; drop every decoded instruction. The
    /// shadow arrays are not snapshotted (nothing reads them but diagnostics).
    fn load_cache_state(&self, _v: &toml::Value) -> Result<(), String> {
        unsafe { (*self.decode.get()).flush() };
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
        let base = self.ram(phys)?;
        let word = unsafe { AtomicU32::from_ptr(base.add(phys as usize) as *mut u32) };
        let ok = word.compare_exchange(old, new, Ordering::AcqRel, Ordering::Relaxed).is_ok();
        if ok {
            self.bump_gen(phys);
        }
        Some(ok)
    }

    fn cas_ram64(&self, phys: u64, old: u64, new: u64) -> Option<bool> {
        use std::sync::atomic::{AtomicU64, Ordering};
        let base = self.ram(phys)?;
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

/// SGI O2 R10000 (IP32): the IP28 model with 64-byte secondary-cache lines.
///
/// The O2 PROM (4.18 and earlier) handles only that line size. With Config's
/// SB bit set (128-byte lines, as on an IP28) its sizing code keeps the
/// secondary cache's *size* as the line size, then aligns every DMA buffer to
/// 1 MB, runs out of heap, and panics on the first SCSI command. So O2 R10000
/// modules have SB clear, and so does this model.
pub type R10000O2ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 1048576, 64, true, 0x0000_0925, 0x0000_0900, 64, { model::R10000 }>;

/// SGI Octane2 R12000 (IP30): revision 3.5, 2 MiB secondary cache.
///
/// PRId 0x0e35 and FPU word 0x0900 are the kernel-inventory words of a running
/// Octane2 (origami's `ip30-r12000.md`, O2): not raw CP0 reads, and not proof
/// of every R12000 revision. A 64-entry TLB and MIPS IV are what Linux selects
/// for the implementation field 0x0e (same record, L1), and the primary caches
/// are 32 KiB each from Config (NEC U10278EJ4V0UM p.230). The 128-byte
/// secondary line is the R10000 family's, not established for this module
/// (Config.SB is unknown there).
pub type R12000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 128, true, 0x0000_0e35, 0x0000_0900, 64, { model::R12000 }>;

/// SGI O2 R12000 (IP32): the Octane2 part with a 1 MiB secondary cache and the
/// 64-byte secondary line the O2 PROM needs (see `R10000O2ShadowCache`).
///
/// SGI shipped O2s with a 300 MHz R12000 and 1 MiB of secondary cache
/// (origami's `o2-hardware-findings.md`). Its PRId revision is not
/// established; the Octane2 word is borrowed so the implementation field is
/// right.
pub type R12000O2ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 1048576, 64, true, 0x0000_0e35, 0x0000_0900, 64, { model::R12000 }>;

/// SGI Fuel R14000 (IP34): 2 MiB secondary cache.
///
/// PRId 0x0f23 is revision 2.3, inferred from a public software inventory of
/// a Fuel, not a captured CP0 word (origami's `ip34-fuel.md`). The FPU word
/// is the R14000 compatibility choice from `r14000.md`.
pub type R14000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 128, true, 0x0000_0f23, 0x0000_0900, 64, { model::R14000 }>;

/// SGI Octane R14000 (IP30): revision 2.4, 2 MiB secondary cache with
/// 128-byte lines, as two Linux captures of a dual-R14000 Octane report
/// (origami's `r14000.md` and its IP30 identity and cache evidence review).
pub type R14000Ip30ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 2097152, 128, true, 0x0000_0f24, 0x0000_0900, 64, { model::R14000 }>;

/// SGI Tezro R16000 (IP53), a 4 MiB secondary cache.
///
/// PRId 0x0f50 is what a Tezro PROM capture prints for four processors
/// (origami's `r16000.md`, "PROM CP0 PRId evidence"); it is conditional on
/// those parts, not established for every R16000 revision. The FPU word
/// 0x0900 is an R14000 compatibility approximation. The secondary cache is
/// product-specific, not an R16000 constant: 4 MiB is the 700 MHz Tezro part,
/// and the Origin 3900 part has 8 MiB. Its 128-byte line is unestablished.
/// Linux treats the R16000 secondary cache like the family's (c-r4k.c,
/// 374-390). Nothing else of the R16000 is modelled: its CP0 diagnostic
/// register, denormal handling and CVT.L limit are the R14000's.
pub type R16000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 4194304, 128, true, 0x0000_0f50, 0x0000_0900, 64, { model::R16000 }>;

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
    /// it was given — including any the real layout would not use. The cache
    /// model that preceded this one decoded TagLo into ptag/state/pidx fields
    /// and re-encoded on the way out, which silently dropped the low seven
    /// bits and every state code it did not recognise. A walking-1s test then
    /// failed on its very first bit.
    #[test]
    fn tags_round_trip_every_bit() {
        let c = cache();
        for bit in 0..64 {
            // 32 and 63 are MRU control/state, not tag storage.
            if bit == 32 || bit == 63 { continue; }
            let v = 1u64 << bit;
            c.cache_op(C_IST | CACH_SD, 0, v);
            assert_eq!(
                c.cache_op(C_ILT | CACH_SD, 0, 0),
                v,
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
        c.cache_op(C_IST | CACH_SD, 0, 0xdead_beef);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 0, 0), 0xdead_beef);
    }

    /// Separate lines must not alias onto one another.
    #[test]
    fn distinct_lines_hold_distinct_tags() {
        let c = cache();
        c.cache_op(C_IST | CACH_SD, 0, 0x1111_1111);
        c.cache_op(C_IST | CACH_SD, 128, 0x2222_2222);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 0, 0), 0x1111_1111);
        assert_eq!(c.cache_op(C_ILT | CACH_SD, 128, 0), 0x2222_2222);
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
                c.cache_op(C_IST | CACH_SD, va, 0x1000 + set * 16 + way);
            }
        }
        for set in 0..4u64 {
            for way in 0..2u64 {
                let va = set * 128 + way;
                assert_eq!(c.cache_op(C_ILT | CACH_SD, va, 0),
                           0x1000 + set * 16 + way,
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

    /// Each R10000-family part reports its own identity, and all of them are
    /// the family: 44 virtual and 40 physical address bits, R10000 cache
    /// operations. Only the R14000 masks its secondary tags.
    #[test]
    fn the_r10000_family_parts_report_their_own_identity() {
        fn id<C: CpuModel>() -> (u32, u32, usize, bool, &'static str, u32, u32, u64) {
            (C::PRID, C::FIR, C::TLB_ENTRIES, C::MIPS4, C::NAME, C::VA_BITS, C::PA_BITS, C::SECONDARY_TAG_MASK)
        }
        fn caches<C: MipsCache>() -> (usize, usize, usize, usize, usize, usize) {
            (C::IC_SIZE, C::IC_LINE, C::DC_SIZE, C::DC_LINE, C::L2_SIZE, C::L2_LINE)
        }
        assert_eq!(id::<R10000ShadowCache>(), (0x925, 0x900, 64, true, "shadow", 44, 40, u64::MAX));
        assert_eq!(id::<R12000ShadowCache>(), (0xe35, 0x900, 64, true, "R12000", 44, 40, u64::MAX));
        assert_eq!(id::<R14000ShadowCache>(), (0xf23, 0x900, 64, true, "R14000", 44, 40, R14000_L2_TAG_MASK));
        assert_eq!(id::<R14000Ip30ShadowCache>(), (0xf24, 0x900, 64, true, "R14000", 44, 40, R14000_L2_TAG_MASK));
        assert_eq!(id::<R16000ShadowCache>(), (0xf50, 0x900, 64, true, "R16000", 44, 40, u64::MAX));
        assert_eq!(caches::<R12000ShadowCache>(), (32768, 64, 32768, 32, 2 << 20, 128));
        assert_eq!(caches::<R12000O2ShadowCache>(), (32768, 64, 32768, 32, 1 << 20, 64));
        assert_eq!(caches::<R14000ShadowCache>(), (32768, 64, 32768, 32, 2 << 20, 128));
        assert_eq!(caches::<R16000ShadowCache>(), (32768, 64, 32768, 32, 4 << 20, 128));
        fn family<C: CpuModel>() {
            assert!(C::R10K_CACHE_OPS);
        }
        family::<R10000ShadowCache>();
        family::<R10000O2ShadowCache>();
        family::<R12000ShadowCache>();
        family::<R12000O2ShadowCache>();
        family::<R14000ShadowCache>();
        family::<R14000Ip30ShadowCache>();
        family::<R16000ShadowCache>();
        assert!(!<R5000ShadowCache as CpuModel>::R10K_CACHE_OPS);
        assert!(!<R4400ShadowCache as CpuModel>::R10K_CACHE_OPS);
    }

    /// The secondary tag mask is the one difference between the family's parts
    /// that diagnostics see: the R14000 keeps MRU, tag, state, index and ECC,
    /// and the rest keep every bit.
    #[test]
    fn only_the_r14000_masks_secondary_tags() {
        fn stored<C: CpuModel + MipsCache + From<Arc<dyn BusDevice>>>() -> u64 {
            let c = C::from(Arc::new(Memory::new(1)) as Arc<dyn BusDevice>);
            c.cache_op(C_IST | CACH_SD, 0, u64::MAX);
            c.cache_op(C_ILT | CACH_SD, 0, 0)
        }
        assert_eq!(stored::<R10000ShadowCache>(), u64::MAX);
        assert_eq!(stored::<R12000ShadowCache>(), u64::MAX);
        assert_eq!(stored::<R16000ShadowCache>(), u64::MAX);
        assert_eq!(stored::<R14000ShadowCache>(), 0x8000_000f_ffff_cdff);
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
