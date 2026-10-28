//! The O2's CPU cache: the full model until the kernel starts, the shadow one
//! after. Both O2 CPUs, R10000 and R5000, run this way.
//!
//! The O2 PROM sizes memory with its stack in the data cache, before there is
//! any RAM behind it, and only a cache that holds data keeps that stack: the
//! shadow cache (tags only, loads and stores straight to memory) loses it and
//! POST returns to a garbage address. Everything after the PROM is the
//! opposite case. An OS manages the caches itself, and the shadow cache is the
//! model IP28 (and every jitv2 build of the other machines) runs on: faster,
//! and out of the data path, so a cache modelling mistake cannot corrupt
//! memory.
//!
//! So this holds both and forwards to one. [`HandoverCache::hand_over`]
//! writes every dirty line of the full model back to memory, carries the
//! LL/SC link across and switches to the shadow model for good.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::cpu::mips_cache_shadow::{R5000ShadowCache, ShadowCache};
use crate::cpu::mips_cache_v2::{
    model, CpuCache, CpuModel, FetchInstrResult, JitDcGeometry, MipsCache,
};
use crate::cpu::mips_isa::{CACH_PD, C_IST};
use crate::traits::{BusDevice, BusRead64};

/// SGI O2 R10000 (IP32): the IP28 model with 64-byte secondary-cache lines.
///
/// The O2 PROM (4.18 and earlier) handles only that line size. With Config's
/// SB bit set (128-byte lines, as on an IP28) its sizing code keeps the
/// secondary cache's *size* as the line size, then aligns every DMA buffer to
/// 1 MB, runs out of heap, and panics on the first SCSI command. So O2 R10000
/// modules have SB clear, and so does this model.
pub type R10000O2Cache = CpuCache<32768, 64, 1, 512,
                                  32768, 32, 1, 1024, 4096,
                                  1048576, 64, 16384, 131072, 262144, true,
                                  true, 0x0000_0925, 0x0000_0900, 64, { model::R10000 }, false>;

/// SGI O2 R5000 (IP32): the Indy/Indigo2 R5000 model, holding its data like
/// the R10000's (`TC = false`) because it too runs the PROM.
pub type R5000O2Cache = CpuCache<32768, 32, 2, 1024,
                                 32768, 32, 2, 1024, 4096,
                                 128, 128, 1, 16, 0, false,
                                 true, 0x0000_2321, 0x0000_2300, 48, { model::R5000 }, false>;

/// The IP28 shadow with the same 64-byte secondary lines. Not usable from
/// reset on its own (see the module notes), so [`HandoverCache`] switches to
/// it once a kernel starts.
pub type R10000O2ShadowCache =
    ShadowCache<32768, 64, 32768, 32, 1048576, 64, true, 0x0000_0925, 0x0000_0900, 64, { model::R10000 }>;

/// A full model `F` for the PROM and a shadow model `S` for the kernel. `F`
/// is built with `TC = false`: it only ever runs the PROM, which keeps its
/// stack in the cache with nothing behind it.
pub struct HandoverCache<F, S> {
    full: F,
    shadow: S,
    on_shadow: AtomicBool,
    /// The guest began initialising the primary data cache's tags while the
    /// shadow model was live: POST starting over, which needs the full model
    /// before it puts its stack in the cache. See [`Self::take_reset_seen`].
    reset_seen: AtomicBool,
}

/// The two O2 CPUs.
pub type R10000Handover = HandoverCache<R10000O2Cache, R10000O2ShadowCache>;
pub type R5000Handover = HandoverCache<R5000O2Cache, R5000ShadowCache>;

impl<F: From<Arc<dyn BusDevice>>, S: From<Arc<dyn BusDevice>>> From<Arc<dyn BusDevice>> for HandoverCache<F, S> {
    fn from(bus: Arc<dyn BusDevice>) -> Self {
        Self { full: F::from(bus.clone()), shadow: S::from(bus), on_shadow: AtomicBool::new(false), reset_seen: AtomicBool::new(false) }
    }
}

/// Forward to whichever model is live.
macro_rules! live {
    ($s:expr, $c:ident => $e:expr) => {
        if $s.on_shadow.load(Ordering::Relaxed) {
            let $c = &$s.shadow;
            $e
        } else {
            let $c = &$s.full;
            $e
        }
    };
}

impl<F: MipsCache, S: MipsCache> HandoverCache<F, S> {
    /// Whether the shadow model is live.
    #[inline(always)]
    pub fn on_shadow(&self) -> bool {
        self.on_shadow.load(Ordering::Relaxed)
    }

    /// Write the full model's dirty lines back, then switch to the shadow
    /// model for good (until a reboot hands back).
    pub fn hand_over(&self) {
        if self.on_shadow() {
            return;
        }
        self.full.writeback_all();
        self.shadow.set_llbit(self.full.get_llbit());
        self.shadow.set_lladdr(self.full.get_lladdr());
        // Which physical ranges are RAM (the direct-path bitmap): the memory
        // space maintains the full model's; without it the shadow model sent
        // every instruction fetch and helper access the slow way (indirect
        // calls ran at half speed). The bank layout is the PROM's, fixed
        // before the kernel starts.
        unsafe {
            let (from, to) = (self.full.tcache_bitmap_ptr(), self.shadow.tcache_bitmap_ptr());
            if !from.is_null() && !to.is_null() {
                *to = *from;
            }
        }
        self.on_shadow.store(true, Ordering::Relaxed);
    }

    // The full model runs only the PROM: kept out of line, so the shadow
    // model's fetch, load and store inline into the interpreter as they do
    // on IP28 (an outlined forwarder carrying both models cost the O2 about
    // a sixth of its interpreter time).
    #[cold]
    #[inline(never)]
    fn full_fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        self.full.fetch(virt_addr, phys_addr)
    }
    #[cold]
    #[inline(never)]
    fn full_read<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64) -> BusRead64 {
        self.full.read::<SIZE>(virt_addr, phys_addr)
    }
    #[cold]
    #[inline(never)]
    fn full_write<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        self.full.write::<SIZE>(virt_addr, phys_addr, val)
    }

    /// Whether POST has started over since the last call: an Index Store Tag
    /// to the primary data cache under the shadow model. POST sets every tag
    /// that way before it keeps its stack in the cache; an OS calling a PROM
    /// routine, or running, does not. A reboot may simply jump into the PROM,
    /// so this, not the reset vector, is what tells the machine to hand back.
    pub fn take_reset_seen(&self) -> bool {
        self.reset_seen.swap(false, Ordering::Relaxed)
    }

    /// Back to the full model, empty, for a PROM starting over (a reboot):
    /// POST needs a cache that holds its stack again.
    pub fn hand_back(&self) {
        self.reset_seen.store(false, Ordering::Relaxed);
        if !self.on_shadow() {
            return;
        }
        self.full.power_on();
        self.full.set_llbit(self.shadow.get_llbit());
        self.full.set_lladdr(self.shadow.get_lladdr());
        self.on_shadow.store(false, Ordering::Relaxed);
    }
}

impl<F: MipsCache + CpuModel, S: MipsCache> CpuModel for HandoverCache<F, S> {
    const MIPS4: bool = F::MIPS4;
    const PRID: u32 = F::PRID;
    const FIR: u32 = F::FIR;
    const TLB_ENTRIES: usize = F::TLB_ENTRIES;
    const VA_BITS: u32 = F::VA_BITS;
    const PA_BITS: u32 = F::PA_BITS;
    const NAME: &'static str = F::NAME;
    const R10K_CACHE_OPS: bool = F::R10K_CACHE_OPS;
}

impl<F: MipsCache + CpuModel, S: MipsCache> MipsCache for HandoverCache<F, S> {
    const IC_SIZE: usize = F::IC_SIZE;
    const IC_LINE: usize = F::IC_LINE;
    const IC_WAYS: usize = F::IC_WAYS;
    const DC_SIZE: usize = F::DC_SIZE;
    const DC_LINE: usize = F::DC_LINE;
    const DC_WAYS: usize = F::DC_WAYS;
    const L2_SIZE: usize = F::L2_SIZE;
    const L2_LINE: usize = F::L2_LINE;

    fn set_l1i_counters(&mut self, hit: Arc<AtomicU64>, fetch: Arc<AtomicU64>) {
        self.full.set_l1i_counters(hit.clone(), fetch.clone());
        self.shadow.set_l1i_counters(hit, fetch);
    }
    #[inline(always)]
    fn fetch(&self, virt_addr: u64, phys_addr: u64) -> FetchInstrResult {
        if self.on_shadow() { self.shadow.fetch(virt_addr, phys_addr) } else { self.full_fetch(virt_addr, phys_addr) }
    }
    fn fetch_is_pure(&self) -> bool {
        live!(self, c => c.fetch_is_pure())
    }
    #[inline(always)]
    fn read<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64) -> BusRead64 {
        if self.on_shadow() { self.shadow.read::<SIZE>(virt_addr, phys_addr) } else { self.full_read::<SIZE>(virt_addr, phys_addr) }
    }
    #[inline(always)]
    fn write<const SIZE: usize>(&self, virt_addr: u64, phys_addr: u64, val: u64) -> u32 {
        if self.on_shadow() { self.shadow.write::<SIZE>(virt_addr, phys_addr, val) } else { self.full_write::<SIZE>(virt_addr, phys_addr, val) }
    }
    fn write64_masked(&self, virt_addr: u64, phys_addr: u64, val: u64, mask: u64) -> u32 {
        live!(self, c => c.write64_masked(virt_addr, phys_addr, val, mask))
    }
    fn cache_op(&self, cache_op: u32, virt_addr: u64, phys_addr: u64) -> u64 {
        if cache_op & 0x1C == C_IST && cache_op & 3 == CACH_PD && self.on_shadow() {
            self.reset_seen.store(true, Ordering::Relaxed);
        }
        live!(self, c => c.cache_op(cache_op, virt_addr, phys_addr))
    }
    fn set_cache_ecc(&self, v: u32) {
        live!(self, c => c.set_cache_ecc(v))
    }
    fn set_error_control(&self, ce: bool, detect: bool) {
        live!(self, c => c.set_error_control(ce, detect))
    }
    fn take_cache_err(&self) -> u32 {
        live!(self, c => c.take_cache_err())
    }
    fn take_op_status(&self) -> u32 {
        live!(self, c => c.take_op_status())
    }
    fn cache_op_ecc(&self) -> u32 {
        live!(self, c => c.cache_op_ecc())
    }
    fn writeback(&self, virt_addr: Option<u64>, phys_addr: u64, size: u64) -> usize {
        live!(self, c => c.writeback(virt_addr, phys_addr, size))
    }
    fn get_config(&self, cache_target: u32) -> (usize, usize) {
        live!(self, c => c.get_config(cache_target))
    }
    fn downstream(&self) -> Arc<dyn BusDevice> {
        self.full.downstream()
    }
    fn check_and_clear_llbit(&self, phys_addr: u64) {
        live!(self, c => c.check_and_clear_llbit(phys_addr))
    }
    fn get_llbit(&self) -> bool {
        live!(self, c => c.get_llbit())
    }
    fn set_llbit(&self, val: bool) {
        live!(self, c => c.set_llbit(val))
    }
    fn get_lladdr(&self) -> u64 {
        live!(self, c => c.get_lladdr())
    }
    fn set_lladdr(&self, addr: u64) {
        live!(self, c => c.set_lladdr(addr))
    }
    fn debug_probe(&self, cache_name: &str, virt_addr: u64, phys_addr: u64) -> String {
        live!(self, c => c.debug_probe(cache_name, virt_addr, phys_addr))
    }
    fn debug_dump_line(&self, cache_name: &str, idx: usize) -> String {
        live!(self, c => c.debug_dump_line(cache_name, idx))
    }
    fn set_l2_enabled(&mut self, enabled: bool) {
        self.full.set_l2_enabled(enabled);
        self.shadow.set_l2_enabled(enabled);
    }
    fn power_on(&self) {
        self.full.power_on();
        self.shadow.power_on();
        self.on_shadow.store(false, Ordering::Relaxed);
    }
    /// Both models and which one is live: a snapshot taken in the PROM has
    /// its stack in the full model's lines, one taken under a kernel runs on
    /// the shadow model.
    fn save_cache_state(&self) -> toml::Value {
        let mut t = toml::map::Map::new();
        t.insert("on_shadow".into(), toml::Value::Boolean(self.on_shadow()));
        t.insert("full".into(), self.full.save_cache_state());
        t.insert("shadow".into(), self.shadow.save_cache_state());
        toml::Value::Table(t)
    }
    fn load_cache_state(&self, v: &toml::Value) -> Result<(), String> {
        use crate::snapshot::{get_field, toml_bool};
        if let Some(f) = get_field(v, "full") {
            self.full.load_cache_state(f)?;
        }
        if let Some(s) = get_field(v, "shadow") {
            self.shadow.load_cache_state(s)?;
        }
        let shadow = get_field(v, "on_shadow").and_then(toml_bool).unwrap_or(false);
        self.on_shadow.store(shadow, Ordering::Relaxed);
        self.reset_seen.store(false, Ordering::Relaxed);
        Ok(())
    }
    // The ppmem windows go to both models; everything the JIT's inline
    // path reads comes from the live one, so code compiled after the
    // handover gets the shadow model's tagless path (window only).
    unsafe fn set_tcache_window(&self, base: *mut u8) {
        unsafe {
            self.full.set_tcache_window(base);
            self.shadow.set_tcache_window(base);
        }
    }
    #[cfg(feature = "jitv2")]
    unsafe fn set_tcache_gen_window(&self, gen_base: *mut AtomicU64) {
        unsafe {
            self.full.set_tcache_gen_window(gen_base);
            self.shadow.set_tcache_gen_window(gen_base);
        }
    }
    /// The one the machine registers with the memory space, which keeps it
    /// up to date: the full model's under tcache (`hand_over` copies it
    /// across), else the shadow model's own. Without tcache the full model
    /// has none, and the shadow model ran with an empty bitmap: every load,
    /// store and instruction fetch went through the bus.
    fn tcache_bitmap_ptr(&self) -> *mut u64 {
        let p = self.full.tcache_bitmap_ptr();
        if p.is_null() { self.shadow.tcache_bitmap_ptr() } else { p }
    }
    fn tcache_base_ptr(&self) -> *mut u8 {
        live!(self, c => c.tcache_base_ptr())
    }
    #[cfg(feature = "jitv2")]
    fn tcache_gen_ptr(&self) -> *mut u8 {
        live!(self, c => c.tcache_gen_ptr())
    }
    fn jit_fast_read<const SIZE: usize>(&self, virt: u64, phys: u64) -> Option<u64> {
        live!(self, c => c.jit_fast_read::<SIZE>(virt, phys))
    }
    fn jit_fast_write<const SIZE: usize>(&self, virt: u64, phys: u64, val: u64) -> bool {
        live!(self, c => c.jit_fast_write::<SIZE>(virt, phys, val))
    }
    fn jit_l2_tags_ptr(&self) -> *mut u8 {
        live!(self, c => c.jit_l2_tags_ptr())
    }
    fn jit_dc_tags_ptr(&self) -> *mut u8 {
        live!(self, c => c.jit_dc_tags_ptr())
    }
    fn jit_dc_lru_ptr(&self) -> *mut u8 {
        live!(self, c => c.jit_dc_lru_ptr())
    }
    fn writeback_all(&self) {
        live!(self, c => c.writeback_all())
    }
    fn jit_dc_geometry(&self) -> JitDcGeometry {
        live!(self, c => c.jit_dc_geometry())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shadow stands in for the full model once the kernel runs, so
    /// everything software can read must be the full model's: the geometry
    /// CP0 Config reports (64-byte secondary lines, the one size the O2 PROM
    /// handles), PRId/FIR, ISA level and TLB size.
    #[test]
    fn the_o2_shadow_matches_the_o2_full_model() {
        type S = R10000O2ShadowCache;
        type F = R10000O2Cache;
        same::<F, S>();
        assert_eq!(<F as MipsCache>::L2_LINE, 64);
        same::<R5000O2Cache, R5000ShadowCache>();
    }

    /// The memory space publishes the RAM bitmap into whichever pointer this
    /// hands it, and after the handover the shadow model must see it, with or
    /// without tcache: an empty bitmap sends every access through the bus.
    #[test]
    fn after_the_handover_ram_is_read_from_the_window() {
        let c = R10000Handover::from(Arc::new(crate::dev::mem::UnmappedRam) as Arc<dyn BusDevice>);
        let mut window = [0u32; 4];
        window[1] = 0x1234_5678;
        unsafe {
            c.set_tcache_window(window.as_mut_ptr() as *mut u8);
            // What `set_bitmap_sink2` and a remap do: region 0 is RAM.
            *c.tcache_bitmap_ptr() = 1;
        }
        c.hand_over();
        assert_eq!(c.read::<4>(0x8000_0004, 4).data, 0x1234_5678, "a load reads the window, not the bus");
    }

    fn same<F: MipsCache + CpuModel, S: MipsCache + CpuModel>() {
        assert_eq!((S::IC_SIZE, S::IC_LINE, S::DC_SIZE, S::DC_LINE, S::L2_SIZE, S::L2_LINE),
                   (F::IC_SIZE, F::IC_LINE, F::DC_SIZE, F::DC_LINE, F::L2_SIZE, F::L2_LINE), "geometry");
        assert_eq!((<S as CpuModel>::PRID, <S as CpuModel>::FIR, <S as CpuModel>::MIPS4,
                    <S as CpuModel>::TLB_ENTRIES, <S as CpuModel>::VA_BITS, <S as CpuModel>::R10K_CACHE_OPS),
                   (<F as CpuModel>::PRID, <F as CpuModel>::FIR, <F as CpuModel>::MIPS4,
                    <F as CpuModel>::TLB_ENTRIES, <F as CpuModel>::VA_BITS, <F as CpuModel>::R10K_CACHE_OPS),
                   "identity");
    }
}
