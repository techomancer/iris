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
//!   valid, say — that is a separate question from storing the bits.
//!
//! The shadow is deliberately not consulted by `read`/`write`/`fetch`, with
//! one exception, below. If it ever needs to be for anything else, this type
//! is the wrong shape and should say so loudly rather than growing a slow
//! path.
//!
//! # Cache as RAM
//!
//! "Cache and memory agree" fails while a PROM uses the primary data cache
//! as its RAM. It keeps its stack and data in cache lines and, meanwhile,
//! either has no memory behind them at all (the Origin's PROM, before it
//! sets up DRAM) or reprograms and tests the memory behind them with
//! uncached accesses, relying on the cache not seeing those (the O2's PROM
//! sizing memory). Real hardware serves the PROM's cached loads and stores
//! from the lines and never asks memory; a tags-only cache loses the stack.
//!
//! So a machine can give this cache a *cache-as-RAM window*: a physical
//! range, no larger than the primary data cache, that the PROM keeps its
//! data in, and arm it while the PROM runs that way
//! ([`MipsCache::set_cache_as_ram`]). While armed, a cached access to the
//! window is served from a line [`Car`] holds, allocated on first touch as
//! the hardware would: filled from memory if memory answers, zero if not,
//! or unfilled when software makes it valid with Index_Store_Tag or
//! Create_Dirty_Exclusive. The CACHE operations that move data between a
//! line and memory act on held lines as on the hardware: Index_Store_Tag
//! and Hit_Invalidate drop one unwritten, Index_Writeback_Invalidate and
//! Hit_Writeback_Invalidate write it to the bus and drop it, Hit_Writeback
//! (R4000 family) writes it and keeps it. Disarming writes every held line
//! back. Everything outside the window, and everything while disarmed,
//! goes straight to memory as before, and uncached accesses never come here,
//! so they reach memory past a held line just as on the part.
//!
//! The window and when it is armed are the machine's: they are facts about
//! its PROM, not about the cache. A machine without one (the Octane, whose
//! PROM sets up memory before it uses the cache) never arms it and pays one
//! predictable branch per load and store.

use std::cell::UnsafeCell;
use std::sync::Arc;

use crate::cpu::mips_cache_v2::{
    cache_op_name, model, r4k_load_taglo, r4k_store_taglo, CpuModel, FetchInstrResult, MipsCache,
    C_CDX, C_HINV, C_HWB, C_HWBINV, C_ILT, C_IST, C_IWBINV, C_R10K_CBARRIER, C_R10K_ILD, C_R10K_ISD,
    CACH_PD, CACH_PI, CACH_SD, CACH_SI,
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

/// The primary data lines held as cache-as-RAM (see the module notes),
/// indexed like the shadow's primary data tags.
struct Car {
    /// The machine's window, empty when it has none.
    window: std::ops::Range<u64>,
    /// Whether accesses to the window are held.
    armed: bool,
    /// The physical line address each slot holds, or [`Car::EMPTY`].
    line: Box<[u64]>,
    /// The held lines' data, a line's worth of doublewords per slot, as
    /// `read64` returns them.
    data: Box<[u64]>,
    /// Slots holding a line.
    held: usize,
}

impl Car {
    const EMPTY: u64 = u64::MAX;

    fn new(lines: usize, line_bytes: usize) -> Self {
        Self {
            window: 0..0,
            armed: false,
            line: vec![Self::EMPTY; lines].into_boxed_slice(),
            data: vec![0u64; lines * line_bytes / 8].into_boxed_slice(),
            held: 0,
        }
    }

    fn in_window(&self, phys: u64) -> bool {
        self.armed && self.window.contains(&phys)
    }

    fn span(&self, slot: usize) -> std::ops::Range<usize> {
        let n = self.data.len() / self.line.len();
        slot * n..(slot + 1) * n
    }

    fn words(&mut self, slot: usize) -> &mut [u64] {
        let r = self.span(slot);
        &mut self.data[r]
    }

    /// `slot` now holds `line`, its data still to be filled.
    fn hold(&mut self, slot: usize, line: u64) {
        if self.line[slot] == Self::EMPTY {
            self.held += 1;
        }
        self.line[slot] = line;
    }

    fn drop_slot(&mut self, slot: usize) {
        if self.line[slot] != Self::EMPTY {
            self.line[slot] = Self::EMPTY;
            self.held -= 1;
        }
    }

    fn clear(&mut self) {
        self.line.fill(Self::EMPTY);
        self.held = 0;
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
    /// Primary data lines held as cache-as-RAM: see the module notes.
    car: UnsafeCell<Car>,
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
    const R10K: bool = MODEL == model::R10000;
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
            ic: UnsafeCell::new(Shadow::new(Self::IC_LINES, 0)),
            dc: UnsafeCell::new(Shadow::new(Self::DC_LINES, 0)),
            // Only the R10000 secondary keeps a data shadow: it is the one
            // the IP28 PROM walks with Index_Store_Data, and a 1 MB array is
            // cheap once. The R4000 family has no index data ops.
            l2: UnsafeCell::new(Shadow::new(Self::L2_LINES, if Self::R10K { L2_SIZE / 8 } else { 0 })),
            car: UnsafeCell::new(Car::new(Self::DC_LINES.max(1), DC_LINE.max(8))),
            tc_base: UnsafeCell::new(std::ptr::null_mut()),
            tc_bitmap: UnsafeCell::new(0),
            #[cfg(feature = "jitv2")]
            tc_gen: UnsafeCell::new(std::ptr::null_mut()),
            decode: UnsafeCell::new(crate::cpu::decode_cache::DecodeCache::new()),
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

    #[allow(clippy::mut_from_ref)]
    fn car(&self) -> &mut Car {
        unsafe { &mut *self.car.get() }
    }

    /// Whether the window is armed: the one test every load and store pays.
    #[inline(always)]
    fn car_live(&self) -> bool {
        unsafe { (*self.car.get()).armed }
    }

    /// The primary data slots a cached access at `virt_addr` can hit, one per
    /// way of its set, numbered as `tag_slot` numbers them.
    fn car_slots(&self, virt_addr: u64) -> [usize; 2] {
        let (va, line, lines) = (virt_addr as usize, DC_LINE.max(1), Self::DC_LINES.max(1));
        if Self::R10K {
            let set = (va / line) % (lines / WAYS).max(1);
            [set * WAYS, set * WAYS + 1]
        } else if Self::R5K {
            let sets = (lines / 2).max(1);
            let set = (va / line) % sets;
            [set, sets + set]
        } else {
            let s = (va / line) % lines;
            [s, s]
        }
    }

    /// The slot holding the line a cached access at `phys_addr` falls in.
    fn car_find(&self, virt_addr: u64, phys_addr: u64) -> Option<usize> {
        let line = phys_addr & !(DC_LINE as u64 - 1);
        let car = self.car();
        self.car_slots(virt_addr).into_iter().find(|&s| car.line[s] == line)
    }

    /// The held line for a cached access to the armed window, allocated on
    /// first touch and filled from memory where memory answers. The way is
    /// the window offset's bit above a way's span, so a window the size of
    /// the cache fills every line of it once, as the PROM's sequential
    /// stores do on the part; a line it displaces is written back first.
    fn car_line(&self, virt_addr: u64, phys_addr: u64) -> Option<usize> {
        if !self.car().in_window(phys_addr) {
            return None;
        }
        if let Some(slot) = self.car_find(virt_addr, phys_addr) {
            return Some(slot);
        }
        let ways = if Self::R10K || Self::R5K { 2 } else { 1 };
        let car = self.car();
        let way = ((phys_addr - car.window.start) as usize / (DC_SIZE / ways).max(1)) % ways;
        let slot = self.car_slots(virt_addr)[way];
        self.car_writeback(slot);
        let line = phys_addr & !(DC_LINE as u64 - 1);
        let car = self.car();
        car.hold(slot, line);
        for (i, w) in car.words(slot).iter_mut().enumerate() {
            let r = self.downstream.read64(line + 8 * i as u64);
            *w = if r.is_ok() { r.data } else { 0 };
        }
        Some(slot)
    }

    /// The physical line a primary data tag names, if its state is valid:
    /// PTag is PA[35:12] in TagLo[31:8], and on the R10000 PA[39:36] in
    /// TagHi[3:0]. The index address supplies the bits below the page.
    fn pd_tag_line(virt_addr: u64, tag: u64) -> Option<u64> {
        if (tag >> 6) & 3 == 0 {
            return None;
        }
        let mut pa = ((tag >> 8) & 0xFF_FFFF) << 12;
        if Self::R10K {
            pa |= ((tag >> 32) & 0xF) << 36;
        }
        Some(pa | (virt_addr & 0xFFF & !(DC_LINE as u64 - 1)))
    }

    /// `slot` was given a new tag: whatever it held goes, unwritten, and it
    /// holds `line` if that is valid and in the armed window. Its data is
    /// what the SRAM held, which is undefined; zero is as good as any.
    fn car_install(&self, slot: usize, line: Option<u64>) {
        let car = self.car();
        car.drop_slot(slot);
        if let Some(line) = line.filter(|&l| car.in_window(l)) {
            car.hold(slot, line);
            car.words(slot).fill(0);
        }
    }

    /// Arm the window `w`, or disarm (`None`), writing back what it held.
    fn car_arm(&self, w: Option<std::ops::Range<u64>>) {
        if let Some(w) = &w {
            assert!(w.end - w.start <= DC_SIZE as u64, "a cache-as-RAM window larger than the data cache");
        }
        let car = self.car();
        if !car.armed && w.as_ref() == Some(&car.window) {
            car.armed = true;
            return;
        }
        if car.armed && w.as_ref() == Some(&car.window) {
            return;
        }
        for slot in 0..car.line.len() {
            self.car_writeback(slot);
        }
        let car = self.car();
        car.clear();
        car.armed = w.is_some();
        if let Some(w) = w {
            car.window = w;
        }
    }

    /// Write a held slot to the bus. Where nothing answers the write goes
    /// nowhere, as a writeback to a missing address does on the part.
    fn car_writeback(&self, slot: usize) {
        let car = self.car();
        let line = car.line[slot];
        if line == Car::EMPTY {
            return;
        }
        for (i, w) in car.words(slot).iter().enumerate() {
            let _ = self.downstream.write64(line + 8 * i as u64, *w);
        }
    }

    /// The bit position of a `size`-byte access at `phys_addr` within its
    /// doubleword: big-endian, so the lowest address is the top byte.
    fn car_lane(size: usize, phys_addr: u64) -> (usize, u32) {
        let word = (phys_addr as usize % DC_LINE.max(8)) / 8;
        let shift = 8 * (8 - size - (phys_addr as usize & 7)) as u32;
        (word, shift)
    }

    #[cold]
    #[inline(never)]
    fn car_read(&self, size: usize, virt_addr: u64, phys_addr: u64) -> Option<u64> {
        let slot = self.car_line(virt_addr, phys_addr)?;
        let (word, shift) = Self::car_lane(size, phys_addr);
        let w = self.car().words(slot)[word];
        Some(if size == 8 { w } else { (w >> shift) & ((1u64 << (8 * size)) - 1) })
    }

    /// Merge `val` under `mask` (both in the doubleword's own bit positions)
    /// into the held line, if `phys_addr` is in the armed window.
    #[cold]
    #[inline(never)]
    fn car_write(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> bool {
        let Some(slot) = self.car_line(virt_addr, phys_addr) else { return false };
        let (word, _) = Self::car_lane(8, phys_addr & !7);
        let w = &mut self.car().words(slot)[word];
        *w = (*w & !mask) | (val & mask);
        true
    }

    /// Hit operations on the primary data cache: write back and/or drop the
    /// held line `phys_addr` falls in.
    fn car_hit(&self, virt_addr: u64, phys_addr: u64, writeback: bool, invalidate: bool) {
        if let Some(slot) = self.car_find(virt_addr, phys_addr) {
            if writeback {
                self.car_writeback(slot);
            }
            if invalidate {
                self.car().drop_slot(slot);
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
    // The R10000 implements 44 virtual address bits where the R4x00 implements
    // 40; the shadow cache is only used for it, but key this off the same flag
    // that selects its cache encodings rather than asserting it unconditionally.
    const VA_BITS: u32 = if MODEL == model::R10000 { 44 } else { 40 };
    const PA_BITS: u32 = if MODEL == model::R10000 { 40 } else { 36 };
    // The CPU's name, which snapshots check: the same as the full model's, so
    // a mismatch reads as a cache-kind difference rather than a different CPU.
    // The R10000 keeps its historical "shadow", which existing IP28
    // snapshots carry.
    const NAME: &'static str = match MODEL {
        model::R4400 => "R4400",
        model::R5000 => "R5000",
        _ => "shadow",
    };
    const CACHE_KIND: &'static str = "shadow";
    const R10K_CACHE_OPS: bool = MODEL == model::R10000;
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
    ///
    /// Nor while a cache-as-RAM window is armed: the inline path decides by
    /// the RAM bitmap alone and would reach memory past a held line. A
    /// machine re-publishes the geometry when it arms or disarms one.
    fn jit_dc_geometry(&self) -> crate::cpu::mips_cache_v2::JitDcGeometry {
        #[cfg(feature = "jitv2")]
        if !self.car_live() && !unsafe { *self.tc_base.get() }.is_null() && !unsafe { *self.tc_gen.get() }.is_null() {
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

    fn read<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64) -> BusRead64 {
        const {
            assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE")
        };
        // Ahead of RAM: a held line keeps its data even once memory is
        // configured under it.
        if self.car_live() {
            if let Some(v) = self.car_read(SIZE, virt_addr, phys_addr) {
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

    fn write<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        const {
            assert!(SIZE == 1 || SIZE == 2 || SIZE == 4 || SIZE == 8, "invalid memory access SIZE")
        };
        if self.car_live() {
            let (_, shift) = Self::car_lane(SIZE, phys_addr);
            let mask = if SIZE == 8 { u64::MAX } else { ((1u64 << (8 * SIZE)) - 1) << shift };
            if self.car_write(virt_addr, phys_addr, val << (shift & 63), mask) {
                return BUS_OK;
            }
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

    fn write64_masked(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        if self.car_live() && self.car_write(virt_addr, phys_addr, val, mask) {
            return BUS_OK;
        }
        if let Some(base) = self.ram(phys_addr) {
            let p = unsafe { base.add((phys_addr & !7) as usize) as *mut u64 };
            unsafe {
                let cur = (*p).rotate_left(32);
                *p = ((cur & !mask) | (val & mask)).rotate_left(32);
            }
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

        // Cache as RAM: what the primary data operations do to held lines
        // (see the module notes). Index_Store_Tag is below, where the tag
        // is stored.
        if sel == CACH_PD {
            match op {
                C_IWBINV => {
                    let slot = self.tag_slot(sel, virt_addr);
                    self.car_writeback(slot);
                    self.car().drop_slot(slot);
                }
                C_HINV => self.car_hit(virt_addr, phys_addr, false, true),
                C_HWBINV => self.car_hit(virt_addr, phys_addr, true, true),
                C_HWB if !Self::R10K => self.car_hit(virt_addr, phys_addr, true, false),
                // A valid dirty line with no fill, at the access's address:
                // the other way the R4000 family makes a line software owns.
                C_CDX if !Self::R10K && self.car().in_window(phys_addr) => {
                    let line = phys_addr & !(DC_LINE as u64 - 1);
                    let [a, b] = self.car_slots(virt_addr);
                    self.car_install(if self.car().line[b] == line { b } else { a }, Some(line));
                }
                _ => {}
            }
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
                if sel == CACH_PD {
                    self.car_install(idx, Self::pd_tag_line(virt_addr, s.tags[idx]));
                }
                return 0;
            }
            return r4k_load_taglo(Self::R5K, sel, s.tags[idx] as u32) as u64;
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
                if sel == CACH_PD {
                    self.car_install(idx, Self::pd_tag_line(virt_addr, phys_addr));
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
            C_R10K_ISD if Self::R10K && matches!(sel, CACH_SI | CACH_SD) => {
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

    /// Power-on: tags read back invalid, and nothing decoded survives — RAM
    /// may be reloaded without a write counter moving.
    fn power_on(&self) {
        for sel in [CACH_PI, CACH_PD, CACH_SD] {
            let s = self.shadow(sel);
            s.tags.fill(0);
            s.mru.fill(false);
        }
        self.car().clear();
        unsafe { (*self.decode.get()).flush() };
    }

    fn set_cache_as_ram(&self, window: Option<std::ops::Range<u64>>) {
        self.car_arm(window);
    }

    /// The held lines, which are the only data this cache owns: a snapshot
    /// taken while a PROM keeps its stack in them needs them back.
    fn save_cache_state(&self) -> toml::Value {
        let car = &*self.car();
        let lines: Vec<toml::Value> = (0..car.line.len())
            .filter(|&s| car.line[s] != Car::EMPTY)
            .map(|s| {
                let mut t = toml::map::Map::new();
                t.insert("slot".into(), crate::snapshot::hex_u64(s as u64));
                t.insert("line".into(), crate::snapshot::hex_u64(car.line[s]));
                t.insert("data".into(), crate::snapshot::u64_slice_to_toml(&car.data[car.span(s)]));
                toml::Value::Table(t)
            })
            .collect();
        let mut t = toml::map::Map::new();
        t.insert("car_armed".into(), toml::Value::Boolean(car.armed));
        t.insert("car_window".into(), crate::snapshot::u64_slice_to_toml(&[car.window.start, car.window.end]));
        t.insert("car".into(), toml::Value::Array(lines));
        toml::Value::Table(t)
    }

    /// A restore replaces RAM wholesale; drop every decoded instruction. The
    /// shadow arrays are not snapshotted (nothing reads them but diagnostics);
    /// the cache-as-RAM window and its lines are.
    fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        use crate::snapshot::{get_field, load_u64_slice, toml_bool, toml_u64};
        unsafe { (*self.decode.get()).flush() };
        let car = self.car();
        car.clear();
        car.armed = get_field(v, "car_armed").and_then(toml_bool).unwrap_or(false);
        if let Some(w) = get_field(v, "car_window") {
            let mut r = [0u64; 2];
            load_u64_slice(w, &mut r);
            car.window = r[0]..r[1];
        }
        for l in get_field(v, "car").and_then(|a| a.as_array()).into_iter().flatten() {
            let slot = get_field(l, "slot").and_then(toml_u64).ok_or("cache-as-RAM line without a slot")? as usize;
            let line = get_field(l, "line").and_then(toml_u64).ok_or("cache-as-RAM line without an address")?;
            if slot >= car.line.len() {
                return Err(format!("cache-as-RAM slot {slot} out of range"));
            }
            car.hold(slot, line);
            car.words(slot).fill(0);
            if let Some(d) = get_field(l, "data") {
                load_u64_slice(d, car.words(slot));
            }
        }
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

/// SGI O2 R10000 (IP32): the IP28 shadow with 64-byte secondary-cache lines.
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

    /// The O2 R10000 runs this model from reset, so everything software can
    /// read must be the O2 part's: the IP28 identity with 64-byte secondary
    /// lines, the one size the O2 PROM handles.
    #[test]
    fn the_o2_shadow_is_the_ip28_part_with_64_byte_secondary_lines() {
        type O2 = R10000O2ShadowCache;
        type I2 = R10000ShadowCache;
        assert_eq!(<O2 as MipsCache>::L2_LINE, 64);
        assert_eq!((O2::IC_SIZE, O2::IC_LINE, O2::DC_SIZE, O2::DC_LINE, O2::L2_SIZE),
                   (I2::IC_SIZE, I2::IC_LINE, I2::DC_SIZE, I2::DC_LINE, I2::L2_SIZE));
        assert_eq!((<O2 as CpuModel>::PRID, <O2 as CpuModel>::FIR, <O2 as CpuModel>::TLB_ENTRIES, <O2 as CpuModel>::NAME),
                   (<I2 as CpuModel>::PRID, <I2 as CpuModel>::FIR, <I2 as CpuModel>::TLB_ENTRIES, <I2 as CpuModel>::NAME));
    }

    /// The O2 POST case: memory behind the window, which the PROM rewrites
    /// uncached (straight to the bus here) while its stack is in the cache.
    /// The cached copy must survive, and memory keep what was written to it.
    #[test]
    fn an_armed_window_keeps_its_lines_from_uncached_stores() {
        let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
        let c = R10000ShadowCache::from(mem.clone());
        c.set_cache_as_ram(Some(0..0x1000));
        c.write::<8>(0x8000_0f80, 0xf80, 0x1122_3344_5566_7788);
        mem.write64(0xf80, 0xdead_beef_dead_beef);
        assert_eq!(c.read::<8>(0x8000_0f80, 0xf80).data, 0x1122_3344_5566_7788, "the cached stack");
        assert_eq!(mem.read64(0xf80).data, 0xdead_beef_dead_beef, "memory under it");
        // Outside the window nothing changes: memory is the store.
        c.write::<4>(0x8000_2000, 0x2000, 0x1234_5678);
        assert_eq!(mem.read32(0x2000).data, 0x1234_5678);
        // A line is filled from memory on first touch, as a cached load
        // would fill it.
        mem.write32(0x40, 0xcafe_f00d);
        assert_eq!(c.read::<4>(0x8000_0040, 0x40).data, 0xcafe_f00d);
        // Disarmed, the lines go to memory and memory is the store again.
        c.set_cache_as_ram(None);
        assert_eq!(mem.read64(0xf80).data, 0x1122_3344_5566_7788, "written back on disarm");
        mem.write64(0xf80, 7);
        assert_eq!(c.read::<8>(0x8000_0f80, 0xf80).data, 7);
    }

    /// The Origin case: nothing answers behind the window, and the PROM
    /// makes its lines valid with Index_Store_Tag before using them.
    #[test]
    fn a_window_with_no_memory_behind_it_holds_data() {
        let c = R10000ShadowCache::from(Arc::new(crate::dev::mem::UnmappedRam) as Arc<dyn BusDevice>);
        c.set_cache_as_ram(Some(0x10_0000..0x10_8000));
        // Dirty exclusive (PState 3) for PA 0x10_0040, through way 0 of its set.
        c.cache_op(C_IST | CACH_PD, 0x8010_0040, (0x100 << 8) | 3 << 6);
        c.write::<8>(0x8010_0040, 0x10_0040, 0x0123_4567_89ab_cdef);
        assert_eq!(c.read::<8>(0x8010_0040, 0x10_0040).data, 0x0123_4567_89ab_cdef);
        // Every line of a cache-sized window, both ways of every set.
        for a in (0x10_0000..0x10_8000u64).step_by(32) {
            c.write::<8>(0x8000_0000 | a, a, a ^ 0x5a5a);
        }
        for a in (0x10_0000..0x10_8000u64).step_by(32) {
            assert_eq!(c.read::<8>(0x8000_0000 | a, a).data, a ^ 0x5a5a, "line {a:#x}");
        }
    }

    /// The CACHE operations move a held line as they would a real one.
    #[test]
    fn cache_ops_write_back_and_drop_held_lines() {
        let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
        let c = R10000ShadowCache::from(mem.clone());
        c.set_cache_as_ram(Some(0..0x1000));
        let (va, pa) = (0x8000_0100u64, 0x100u64);
        let reload = |v: u64| { c.write::<8>(va, pa, v); mem.write64(pa, 0); };

        // Index_Store_Tag with TagLo 0 invalidates: the data is gone, and
        // the next access fills from memory.
        reload(1);
        c.cache_op(C_IST | CACH_PD, va, 0);
        c.cache_op(C_IST | CACH_PD, va | 1, 0);
        assert_eq!(mem.read64(pa).data, 0, "not written");
        assert_eq!(c.read::<8>(va, pa).data, 0, "dropped");

        // Index_Writeback_Invalidate writes back, through the way that holds it.
        reload(2);
        c.cache_op(C_IWBINV | CACH_PD, va, 0);
        assert_eq!(mem.read64(pa).data, 2);
        mem.write64(pa, 3);
        assert_eq!(c.read::<8>(va, pa).data, 3, "refilled after the invalidate");

        // Hit_Writeback_Invalidate on the R10000.
        reload(4);
        c.cache_op(C_HWBINV | CACH_PD, va, pa);
        assert_eq!(mem.read64(pa).data, 4);

        // Hit_Invalidate drops it unwritten.
        reload(5);
        c.cache_op(C_HINV | CACH_PD, va, pa);
        assert_eq!(mem.read64(pa).data, 0);
    }

    /// Hit_Writeback writes and keeps the line on the R4000 family.
    #[test]
    fn hit_writeback_keeps_the_line_on_the_r5000() {
        let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
        let c = R5000ShadowCache::from(mem.clone());
        c.set_cache_as_ram(Some(0..0x1000));
        c.write::<8>(0x8000_0200, 0x200, 9);
        c.cache_op(C_HWB | CACH_PD, 0x8000_0200, 0x200);
        assert_eq!(mem.read64(0x200).data, 9);
        mem.write64(0x200, 10);
        assert_eq!(c.read::<8>(0x8000_0200, 0x200).data, 9, "still held");
    }

    /// Byte, halfword, word and partial-doubleword stores to a held line land
    /// in the same bytes they would in memory.
    #[test]
    fn held_lines_keep_memorys_byte_lanes() {
        let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
        let plain = R10000ShadowCache::from(Arc::new(Memory::new(1)) as Arc<dyn BusDevice>);
        let c = R10000ShadowCache::from(mem.clone());
        c.set_cache_as_ram(Some(0..0x1000));
        for cache in [&plain, &c] {
            cache.write::<8>(0x8000_0010, 0x10, 0x0011_2233_4455_6677);
            cache.write::<1>(0x8000_0011, 0x11, 0xaa);
            cache.write::<2>(0x8000_0014, 0x14, 0xbbcc);
            cache.write::<4>(0x8000_0018, 0x18, 0xdead_beef);
            cache.write64_masked(0x8000_0018, 0x18, 0x0000_0000_1234_5678, 0x0000_0000_ffff_0000);
        }
        for (a, n) in [(0x10u64, 8), (0x11, 1), (0x12, 2), (0x14, 4), (0x18, 8), (0x1c, 4), (0x1e, 2)] {
            let (p, h) = match n {
                1 => (plain.read::<1>(0x8000_0000 | a, a).data, c.read::<1>(0x8000_0000 | a, a).data),
                2 => (plain.read::<2>(0x8000_0000 | a, a).data, c.read::<2>(0x8000_0000 | a, a).data),
                4 => (plain.read::<4>(0x8000_0000 | a, a).data, c.read::<4>(0x8000_0000 | a, a).data),
                _ => (plain.read::<8>(0x8000_0000 | a, a).data, c.read::<8>(0x8000_0000 | a, a).data),
            };
            assert_eq!(h, p, "{n} bytes at {a:#x}");
        }
        c.set_cache_as_ram(None);
        assert_eq!(mem.read64(0x18).data, plain.read::<8>(0x8000_0018, 0x18).data, "written back in place");
    }

    /// A snapshot taken while a PROM's stack is held brings it back.
    #[test]
    fn held_lines_survive_a_snapshot() {
        let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(1));
        let c = R10000ShadowCache::from(mem.clone());
        c.set_cache_as_ram(Some(0..0x1000));
        c.write::<8>(0x8000_0f00, 0xf00, 0x0f0f);
        let saved = c.save_cache_state();
        let d = R10000ShadowCache::from(mem.clone());
        d.load_cache_state(&saved).unwrap();
        mem.write64(0xf00, 0);
        assert_eq!(d.read::<8>(0x8000_0f00, 0xf00).data, 0x0f0f);
    }

    /// While a window is armed the JIT must call out for every load and
    /// store: its inline path would go past a held line to RAM.
    #[test]
    #[cfg(feature = "jitv2")]
    fn an_armed_window_turns_the_inline_path_off() {
        let c = cache();
        let mut window = [0u8; 8];
        let mut gens = [std::sync::atomic::AtomicU64::new(0)];
        unsafe {
            c.set_tcache_window(window.as_mut_ptr());
            c.set_tcache_gen_window(gens.as_mut_ptr());
        }
        c.set_cache_as_ram(Some(0..0x1000));
        assert!(!c.jit_dc_geometry().supported);
        c.set_cache_as_ram(None);
        assert!(c.jit_dc_geometry().supported);
    }
}
