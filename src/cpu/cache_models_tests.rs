//! The real caches (`mips_cache_v2`, holding line data) against the shadow
//! caches (`mips_cache_shadow`, memory the only store), access for access.
//!
//! Where both models answer the same, these tests say so for a long seeded
//! run. Where they differ by design, a test names the difference and pins
//! both answers, so a change to either model shows up here rather than in a
//! guest.

use std::sync::Arc;

use crate::cpu::mips_cache_shadow::{R10000ShadowCache, R4400ShadowCache, R5000ShadowCache};
use crate::cpu::mips_cache_v2::{
    MipsCache, RealR10000Cache, RealR4400Cache, RealR5000Cache, CACH_PD, CACH_PI, CACH_SD,
    C_HINV, C_ILT, C_IST,
};
use crate::ppmem::{MappedMemory, PpMemSpace, PpMemory};
use crate::traits::{BusDevice, BUS_OK};

const BANK_MB: usize = 64;
/// The span the access sequence covers: four times the largest secondary
/// cache, so secondary lines are evicted as well as primary ones.
const SPAN: u64 = 4 << 20;

fn kseg0(pa: u64) -> u64 { 0xffff_ffff_8000_0000 | pa }

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// A real cache over its own bank, and a shadow cache reading a second bank
/// through a cached view of a window, the way a machine wires it. Both banks
/// start with the same contents. The shadow is boxed: the view publishes into
/// its inline bitmap field.
struct Pair<R, S> {
    real: R,
    real_mem: Arc<PpMemory>,
    shadow: Box<S>,
    shadow_mem: Arc<PpMemory>,
    _space: Box<PpMemSpace>,
}

impl<R: MipsCache + From<Arc<dyn BusDevice>>, S: MipsCache + From<Arc<dyn BusDevice>>> Pair<R, S> {
    fn new(seed: u64) -> Self {
        let real_mem = Arc::new(PpMemory::new(BANK_MB));
        let real = R::from(real_mem.clone() as Arc<dyn BusDevice>);
        let (space, mut banks) = PpMemSpace::with_bank_sizes(&[BANK_MB]).unwrap();
        let space = Box::new(space);
        let shadow_mem = Arc::new(banks.remove(0));
        let shadow = Box::new(S::from(shadow_mem.clone() as Arc<dyn BusDevice>));
        let v = space.add_cached_view(&[shadow.tcache_bitmap_ptr()]).unwrap();
        unsafe {
            shadow.set_tcache_window(space.view_base(v));
            shadow.set_cached_view(&*space, v);
        }
        space.map_bank(0, 0, (BANK_MB as u64) << 20, (BANK_MB as u64) << 20).unwrap();
        assert!(!shadow.tcache_base_ptr().is_null(), "the shadow must read through the window");
        let mut rng = Rng(seed);
        for pa in (0..SPAN).step_by(8) {
            let v = rng.next();
            real_mem.write64(pa, v);
            shadow_mem.write64(pa, v);
        }
        Self { real, real_mem, shadow, shadow_mem, _space: space }
    }

    /// Both caches' answer to one load, which must be the same.
    fn load<const N: usize>(&self, pa: u64, step: usize, name: &str) {
        let (a, b) = (self.real.read::<N>(kseg0(pa), pa), self.shadow.read::<N>(kseg0(pa), pa));
        assert_eq!((a.status, a.data), (b.status, b.data),
            "{name}: {N}-byte load at {pa:#x}, step {step}: real, shadow");
    }

    fn store<const N: usize>(&self, pa: u64, v: u64, step: usize, name: &str) {
        let (a, b) = (self.real.write::<N>(kseg0(pa), pa, v), self.shadow.write::<N>(kseg0(pa), pa, v));
        assert_eq!((a, b), (BUS_OK, BUS_OK), "{name}: {N}-byte store at {pa:#x}, step {step}");
    }
}

/// An address for the sequence: half anywhere in the span, half on a few
/// lines 32 KB apart, which share a primary set on every model here and so
/// fill both ways of an R5000 set and evict from it.
fn address(rng: &mut Rng) -> u64 {
    let r = rng.next();
    if r & 1 == 0 {
        (r >> 8) % SPAN
    } else {
        (((r >> 8) & 7) << 15 | ((r >> 16) & 0x7f) << 3 | (r >> 24 & 3) << 20) % SPAN
    }
}

/// The same seeded run of 8-, 4-, 2- and 1-byte loads and stores through
/// both models must load the same bytes at every step and, once the real
/// cache has written back its dirty lines, leave the same RAM. Then Index
/// Store Tag and Index Load Tag at random indexes must read back the same
/// bits from both, in the caches `tag_sels` names.
fn agree<R, S>(name: &str, tag_sels: &[u32])
where
    R: MipsCache + From<Arc<dyn BusDevice>>,
    S: MipsCache + From<Arc<dyn BusDevice>>,
{
    let p = Pair::<R, S>::new(0x1234_5678_9abc_def1);
    let mut rng = Rng(0xdead_beef_cafe_0001);
    for step in 0..60_000 {
        let pa = address(&mut rng);
        let v = rng.next();
        match rng.next() % 8 {
            0 => p.load::<8>(pa & !7, step, name),
            1 => p.load::<4>(pa & !3, step, name),
            2 => p.load::<2>(pa & !1, step, name),
            3 => p.load::<1>(pa, step, name),
            4 => p.store::<8>(pa & !7, v, step, name),
            5 => p.store::<4>(pa & !3, v & 0xffff_ffff, step, name),
            6 => p.store::<2>(pa & !1, v & 0xffff, step, name),
            _ => p.store::<1>(pa, v & 0xff, step, name),
        }
    }

    p.real.writeback_all();
    for pa in (0..SPAN).step_by(8) {
        assert_eq!(p.real_mem.read64(pa).data, p.shadow_mem.read64(pa).data,
            "{name}: RAM differs at {pa:#x} after the real cache wrote back");
    }

    for &sel in tag_sels {
        for _ in 0..4096 {
            let index = kseg0(rng.next() & 0xf_fff8);
            let taglo = rng.next() as u32 as u64;
            p.real.cache_op(C_IST | sel, index, taglo);
            p.shadow.cache_op(C_IST | sel, index, taglo);
            assert_eq!(p.real.cache_op(C_ILT | sel, index, index), p.shadow.cache_op(C_ILT | sel, index, index),
                "{name}: tag {taglo:#x} stored at {index:#x} with sel {sel}");
        }
    }
}

#[test]
fn the_r4400_models_agree_access_for_access() {
    agree::<RealR4400Cache, R4400ShadowCache>("R4400", &[CACH_PI, CACH_PD, CACH_SD]);
}

#[test]
fn the_r5000_models_agree_access_for_access() {
    agree::<RealR5000Cache, R5000ShadowCache>("R5000", &[CACH_PI, CACH_PD]);
}

/// Data only: `R10000Cache` keeps the R4400's tag layouts, so its tags are
/// not the R10000's.
#[test]
fn the_r10000_models_agree_access_for_access() {
    agree::<RealR10000Cache, R10000ShadowCache>("R10000", &[]);
}

/// Hit Invalidate drops a dirty line without writing it back, so the real
/// model loses the store, as the hardware does (cpu-tests'
/// `cache/hit_inv_discards` passes on a real Indy). The shadow has no line
/// to drop, and memory keeps the store.
#[test]
fn hit_invalidate_discards_a_dirty_line_only_in_the_real_model() {
    fn store_then_discard<C: MipsCache>(c: &C, pa: u64, has_l2: bool) -> u64 {
        c.write::<4>(kseg0(pa), pa, 0xdead_0000);
        c.cache_op(C_HINV | CACH_PD, kseg0(pa), pa);
        if has_l2 {
            c.cache_op(C_HINV | CACH_SD, kseg0(pa), pa);
        }
        c.read::<4>(kseg0(pa), pa).data
    }
    fn check<R, S>(name: &str, has_l2: bool)
    where
        R: MipsCache + From<Arc<dyn BusDevice>>,
        S: MipsCache + From<Arc<dyn BusDevice>>,
    {
        let p = Pair::<R, S>::new(1);
        let pa = 0x1_0040;
        p.real_mem.write32(pa, 0xba5e_0000);
        p.shadow_mem.write32(pa, 0xba5e_0000);
        assert_eq!(store_then_discard(&p.real, pa, has_l2), 0xba5e_0000, "{name}: the real model kept the store");
        assert_eq!(store_then_discard(&*p.shadow, pa, has_l2), 0xdead_0000, "{name}: the shadow lost the store");
    }
    check::<RealR4400Cache, R4400ShadowCache>("R4400", true);
    check::<RealR5000Cache, R5000ShadowCache>("R5000", false);
}

/// A cached store reaches memory only when the real model writes its line
/// back. In the shadow it is in memory at once, which is what an uncached
/// load or a device's DMA then sees.
#[test]
fn memory_is_stale_until_writeback_only_in_the_real_model() {
    fn check<R, S>(name: &str)
    where
        R: MipsCache + From<Arc<dyn BusDevice>>,
        S: MipsCache + From<Arc<dyn BusDevice>>,
    {
        let p = Pair::<R, S>::new(2);
        let pa = 0x2_0080;
        p.real_mem.write32(pa, 0x1111_1111);
        p.shadow_mem.write32(pa, 0x1111_1111);
        p.real.write::<4>(kseg0(pa), pa, 0x2222_2222);
        p.shadow.write::<4>(kseg0(pa), pa, 0x2222_2222);
        assert_eq!(p.real_mem.read32(pa).data, 0x1111_1111, "{name}: the real model wrote through");
        assert_eq!(p.shadow_mem.read32(pa).data, 0x2222_2222, "{name}: the shadow held the store");
        p.real.writeback_all();
        assert_eq!(p.real_mem.read32(pa).data, 0x2222_2222, "{name}: writeback_all left it in the cache");
    }
    check::<RealR4400Cache, R4400ShadowCache>("R4400");
    check::<RealR5000Cache, R5000ShadowCache>("R5000");
    check::<RealR10000Cache, R10000ShadowCache>("R10000");
}

/// A load brings its line into the R10000's secondary cache, and Index Load
/// Tag shows it there, on the real model. The shadow records no line that a
/// load or store brings in, so a later secondary Hit operation finds nothing
/// to act on. Neither model sets Status.CH: nothing in iris does.
#[test]
fn a_load_makes_a_secondary_line_valid_only_in_the_real_r10000() {
    /// The line's state, TagLo bits 10 and up, in whichever way of its
    /// secondary set holds it. `R10000Cache` reports the R4400's three-bit
    /// state there, the shadow the R10000's two bits.
    fn sd_state<C: MipsCache>(c: &C, pa: u64, mask: u64) -> u64 {
        (0..2).map(|way| c.cache_op(C_ILT | CACH_SD, kseg0(pa) | way, 0) >> 10 & mask).max().unwrap()
    }
    let p = Pair::<RealR10000Cache, R10000ShadowCache>::new(3);
    let pa = 0x3_4000;
    p.real.read::<4>(kseg0(pa), pa);
    p.shadow.read::<4>(kseg0(pa), pa);
    assert_ne!(sd_state(&p.real, pa, 7), 0, "the real R10000 holds the line");
    assert_eq!(sd_state(&*p.shadow, pa, 3), 0, "the shadow records no line for a load");
}
