//! A cache that is visible to software but stays out of the data path.
//!
//! Every load, store and instruction fetch goes straight to memory, exactly as
//! in [`PassthroughCacheOf`](crate::mips_cache_v2::PassthroughCacheOf). What
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
//!   valid, say — that is a separate question from storing the bits.
//!
//! The shadow is deliberately not consulted by `read`/`write`/`fetch`. If it
//! ever needs to be, this type is the wrong shape and should say so loudly
//! rather than growing a slow path.

use std::cell::UnsafeCell;
use std::sync::Arc;

use crate::mips_cache_v2::{
    cache_op_name, CpuModel, FetchInstrResult, MipsCache, C_ILT, C_IST, C_R10K_CBARRIER,
    C_R10K_ILD, C_R10K_ISD, CACH_PD, CACH_PI, CACH_SD, CACH_SI,
};
use crate::mips_exec::{DecodedInstr, FLAG_NOT_DECODED};
use crate::traits::{BusDevice, BusRead64};
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

/// Ways per set, as CACHE index operations address them. Bit 0 of the index
/// selects the way on an R10000; the set starts above the line offset.
const WAYS: usize = 2;

/// Bits a secondary-cache tag retains.
///
/// All of them, now. A 36-bit mask was inferred here from the PROM's
/// walking-0s phase, which writes an all-ones TagLo and expects back
/// 0x0000000f_ffffcdfe — but that truncation comes from the tag being
/// assembled as `(TagHi << 32) | TagLo[31:0]`, not from the array dropping
/// bits. Once the executor carried TagHi properly the mask did nothing except
/// discard the MRU bit, which the PROM writes as TagHi[31] and reads back.
const L2_TAG_MASK: u64 = u64::MAX;

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
    const R10K_OPS: bool,
> {
    downstream: Arc<dyn BusDevice>,
    llbit: UnsafeCell<bool>,
    lladdr: UnsafeCell<u32>,
    /// Somewhere to decode into. Not a cache line — there is no caching.
    fetch_scratch: UnsafeCell<DecodedInstr>,
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
    #[cfg(feature = "tcache")]
    tc_base: UnsafeCell<*mut u8>,
    #[cfg(feature = "tcache")]
    tc_bitmap: UnsafeCell<u64>,
    #[cfg(feature = "jitv2")]
    tc_gen: UnsafeCell<*mut std::sync::atomic::AtomicU64>,
}

// Safety: the CPU thread is the only accessor, as for every other cache model
// in this crate.
unsafe impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const R10K_OPS: bool,
    > Send
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, R10K_OPS>
{
}
unsafe impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const R10K_OPS: bool,
    > Sync
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, R10K_OPS>
{
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const R10K_OPS: bool,
    > ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, R10K_OPS>
{
    const IC_LINES: usize = if IC_LINE == 0 { 0 } else { IC_SIZE / IC_LINE };
    const DC_LINES: usize = if DC_LINE == 0 { 0 } else { DC_SIZE / DC_LINE };
    const L2_LINES: usize = if L2_LINE == 0 { 0 } else { L2_SIZE / L2_LINE };

    pub fn new(downstream: Arc<dyn BusDevice>) -> Self {
        Self {
            downstream,
            llbit: UnsafeCell::new(false),
            lladdr: UnsafeCell::new(0),
            fetch_scratch: UnsafeCell::new(DecodedInstr::default()),
            ecc_in: UnsafeCell::new(0),
            ecc_out: UnsafeCell::new(0),
            ic: UnsafeCell::new(Shadow::new(Self::IC_LINES, 0)),
            dc: UnsafeCell::new(Shadow::new(Self::DC_LINES, 0)),
            // Only the secondary keeps a data shadow: it is the one a PROM
            // walks with Index_Store_Data, and a 1 MB array is cheap once.
            l2: UnsafeCell::new(Shadow::new(Self::L2_LINES, L2_SIZE / 8)),
            #[cfg(feature = "tcache")]
            tc_base: UnsafeCell::new(std::ptr::null_mut()),
            #[cfg(feature = "tcache")]
            tc_bitmap: UnsafeCell::new(0),
            #[cfg(feature = "jitv2")]
            tc_gen: UnsafeCell::new(std::ptr::null_mut()),
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
    fn tag_slot(&self, sel: u32, virt_addr: u64) -> usize {
        let (line, lines) = match sel {
            CACH_PI => (IC_LINE, Self::IC_LINES),
            CACH_PD => (DC_LINE, Self::DC_LINES),
            _ => (L2_LINE, Self::L2_LINES),
        };
        if line == 0 || lines == 0 {
            return 0;
        }
        let way = (virt_addr as usize) & (WAYS - 1);
        let set = ((virt_addr as usize) / line) % (lines / WAYS).max(1);
        (set * WAYS + way) % lines
    }

    /// Data slot for an R10000 `Index_Load_Data` / `Index_Store_Data`.
    ///
    /// Same shape: way in bit 0, the rest addressing the array. The PROM walks
    /// it at `…00`, `…01`, `…10`, `…11`, `…20` — one doubleword per operation
    /// with the way bit shifted in beneath it.
    fn data_slot(&self, len: usize, virt_addr: u64) -> usize {
        if len == 0 {
            return 0;
        }
        let way = (virt_addr as usize) & (WAYS - 1);
        let word = (virt_addr as usize) >> 4;
        (word * WAYS + way) % len
    }
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const R10K_OPS: bool,
    > From<Arc<dyn BusDevice>>
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, R10K_OPS>
{
    fn from(downstream: Arc<dyn BusDevice>) -> Self {
        Self::new(downstream)
    }
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const R10K_OPS: bool,
    > CpuModel
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, R10K_OPS>
{
    const MIPS4: bool = MIPS4;
    const PRID: u32 = PRID;
    const FIR: u32 = FIR;
    const TLB_ENTRIES: usize = TLB_ENTRIES;
    // The R10000 implements 44 virtual address bits where the R4x00 implements
    // 40; the shadow cache is only used for it, but key this off the same flag
    // that selects its cache encodings rather than asserting it unconditionally.
    const VA_BITS: u32 = if R10K_OPS { 44 } else { 40 };
    const NAME: &'static str = "shadow";
    const R10K_CACHE_OPS: bool = R10K_OPS;
}

impl<
        const IC_SIZE: usize, const IC_LINE: usize, const DC_SIZE: usize, const DC_LINE: usize,
        const L2_SIZE: usize, const L2_LINE: usize, const MIPS4: bool, const PRID: u32,
        const FIR: u32, const TLB_ENTRIES: usize, const R10K_OPS: bool,
    > MipsCache
    for ShadowCache<IC_SIZE, IC_LINE, DC_SIZE, DC_LINE, L2_SIZE, L2_LINE, MIPS4, PRID, FIR, TLB_ENTRIES, R10K_OPS>
{
    const IC_SIZE: usize = IC_SIZE;
    const IC_LINE: usize = IC_LINE;
    const IC_WAYS: usize = 1;
    const DC_SIZE: usize = DC_SIZE;
    const DC_LINE: usize = DC_LINE;
    const DC_WAYS: usize = 1;
    const L2_SIZE: usize = L2_SIZE;
    const L2_LINE: usize = L2_LINE;

    #[cfg(feature = "tcache")]
    unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe { *self.tc_base.get() = base };
    }

    #[cfg(feature = "tcache")]
    fn tcache_bitmap_ptr(&self) -> *mut u64 { self.tc_bitmap.get() }

    #[cfg(feature = "tcache")]
    fn tcache_base_ptr(&self) -> *mut u8 { unsafe { *self.tc_base.get() } }

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
    fn jit_dc_geometry(&self) -> crate::mips_cache_v2::JitDcGeometry {
        #[cfg(feature = "jitv2")]
        if !unsafe { *self.tc_base.get() }.is_null() && !unsafe { *self.tc_gen.get() }.is_null() {
            return crate::mips_cache_v2::JitDcGeometry {
                supported: true,
                tagless: true,
                ..crate::mips_cache_v2::JitDcGeometry::unsupported()
            };
        }
        crate::mips_cache_v2::JitDcGeometry::unsupported()
    }

    fn fetch(&self, _virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        let r = self.downstream.read32(phys_addr as u32);
        if r.is_ok() {
            let slot = unsafe { &mut *self.fetch_scratch.get() };
            slot.flags = FLAG_NOT_DECODED;
            slot.raw = r.data;
            FetchInstrResult::hit(slot as *const DecodedInstr)
        } else {
            FetchInstrResult::exception(r.status)
        }
    }

    fn read<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64) -> BusRead64 {
        const {
            assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE")
        };
        let a = phys_addr as u32;
        if SIZE == 1 {
            let r = self.downstream.read8(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        } else if SIZE == 2 {
            let r = self.downstream.read16(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        } else if SIZE == 4 {
            let r = self.downstream.read32(a);
            BusRead64 { status: r.status, data: r.data as u64 }
        } else {
            self.downstream.read64(a)
        }
    }

    fn write<const SIZE: usize>(&self, _virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        const {
            assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE")
        };
        let a = phys_addr as u32;
        if SIZE == 1 {
            self.downstream.write8(a, val as u8)
        } else if SIZE == 2 {
            self.downstream.write16(a, val as u16)
        } else if SIZE == 4 {
            self.downstream.write32(a, val as u32)
        } else {
            self.downstream.write64(a, val)
        }
    }

    fn write64_masked(&self, _virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        let aligned = (phys_addr & !7) as u32;
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

        match op {
            // Index_Store_Tag / Index_Load_Tag. Stored and returned verbatim:
            // a tag test is a round trip, and round trips do not require
            // knowing what the bits mean.
            C_IST => {
                let idx = self.tag_slot(sel, virt_addr);
                let mask = if matches!(sel, CACH_SI | CACH_SD) { L2_TAG_MASK } else { u64::MAX };
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
            C_R10K_CBARRIER if R10K_OPS && sel == CACH_PI => 0,
            C_R10K_ILD if R10K_OPS && matches!(sel, CACH_PI | CACH_PD | CACH_SD) => {
                let s = self.shadow(sel);
                let slot = self.data_slot(s.data.len(), virt_addr);
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
            C_R10K_ISD if R10K_OPS && matches!(sel, CACH_SI | CACH_SD) => {
                let s = self.shadow(sel);
                let slot = self.data_slot(s.data.len(), virt_addr);
                if !s.data.is_empty() {
                    s.data[slot] = phys_addr;
                    s.ecc[slot] = unsafe { *self.ecc_in.get() };
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

    fn check_and_clear_llbit(&self, _phys_addr: u64) {
        unsafe { *self.llbit.get() = false };
    }
    fn get_llbit(&self) -> bool {
        unsafe { *self.llbit.get() }
    }
    fn set_llbit(&self, val: bool) {
        unsafe { *self.llbit.get() = val };
    }
    fn get_lladdr(&self) -> u32 {
        unsafe { *self.lladdr.get() }
    }
    fn set_lladdr(&self, addr: u32) {
        unsafe { *self.lladdr.get() = addr };
    }
}

/// SGI Indigo2 IMPACT R10000 (IP28).
///
/// Real geometry as software reads it — 32 KB primaries with 64-byte
/// instruction lines and 32-byte data lines, a 1 MB secondary with 128-byte
/// lines — 64 TLB entries, MIPS IV, and the R10000 cache operation encodings.
/// Nothing of the microarchitecture: no ways, no LRU, no out-of-order.
pub type R10000ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 1048576, 128, true, 0x0000_0900, 0x0000_0900, 64, true>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::mem::Memory;
    use crate::mips_cache_v2::{CACH_PD, CACH_SD};

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

    /// Index_Store_Data / Index_Load_Data likewise.
    #[test]
    fn secondary_data_round_trips() {
        let c = cache();
        c.cache_op(C_R10K_ISD | CACH_SD, 0x40, 0xdeadbeef);
        assert_eq!(c.cache_op(C_R10K_ILD | CACH_SD, 0x40, 0), 0xdeadbeef);
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
        c.cache_op(crate::mips_cache_v2::C_IINV | CACH_PD, 0x2000, 0);
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
