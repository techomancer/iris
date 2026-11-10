//! Texture memory: texels held as the model chooses, one 32-bit word each, at the address
//! a load names.
//!
//! The board's texture memory is not guest-visible except through loads and
//! binds, and the order and width of a texel inside it is not known for any
//! format, so the model keeps its own layout: one word per texel, `0xAARRGGBB`,
//! 8 bits a component, rows of a level contiguous at the level's pitch. A format
//! narrower than that (5-5-5-1, 4-4-4-4) is reduced to its width at load, so the
//! texel shows what the board would hold.
//!
//! The pipe has no allocator of its own: a load writes where its address words say, a level
//! reloaded at the same place overwrites it, and one a later load overlaps is gone
//! (established). A level is addressed by its bank and an address in units of 256 bits, relative
//! to the block the kernel gave its owner. The store follows that: a level at address `a` of a
//! bank starts at word `a * UNIT_WORDS` of that bank's plane, so there is nothing to place and
//! nothing to fragment, and a load is refused only when its address, with the level's extent,
//! lies outside the bank (`contains`) or would take the store past its ceiling (`fits`). Both
//! refusals are the model's own: the hardware has nothing to refuse with, and a client that goes
//! past its block is the kernel's to prevent. A unit holds sixteen 16-bit lanes, so `UNIT_WORDS`
//! words is room for the most texels a unit can carry, and levels whose units do not overlap
//! never share a word.
//!
//! A plane is one bank of one owner's block: the owner's two banks, and two more for its detail
//! images, which the model keeps apart from the base levels (inferred). The block offsets the
//! kernel writes to the global-offset registers on a reserve are not modelled, so every context
//! has planes of its own, each as large as a bank, where the board has one pool the owners' blocks
//! share. That is approximated: contexts neither contend for memory nor alias, so a texture
//! loaded through one member of a share group (contexts the kernel lets see the same physical
//! block) is not visible to a bind from another member.
//!
//! A plane is sparse: it is a table of fixed pages of `PAGE_WORDS` words (64 KiB), and a page
//! exists once a nonzero word is written into it, so host memory follows what was written, not
//! how far the highest address reaches; a word in a page that does not exist reads as zero. The
//! pages held across every plane may not pass `cap_pages`, the most texels the product's texture
//! memory can carry (a lane is the least a texel takes, and the model keeps a word for it), which
//! is what the real pool cannot exceed whichever owners hold it. Per context planes would
//! otherwise let a guest that loads at high addresses in many contexts take gigabytes of host memory.
//!
//! The producer (`kona-hip`) writes only after the tiles are drained, and a draw's workers only
//! read, so a plane may change under `UnsafeCell` without a lock: no reader runs while it changes.

use crate::dev::kona::defs::CONTEXT_SLOTS;
use std::cell::UnsafeCell;

/// Words of a plane a unit of 256 bits (sixteen 16-bit lanes) addresses: the most texels it can carry.
pub const UNIT_WORDS: u32 = 16;
/// Bytes a unit of 256 bits holds.
const UNIT_BYTES: u64 = 32;
/// The planes a context owns: its two banks and its detail images' two.
pub const PLANES_PER_CONTEXT: usize = 4;
/// Every context slot and the no-context state have their planes.
pub const PLANES: usize = (CONTEXT_SLOTS + 1) * PLANES_PER_CONTEXT;
/// Words of a page: 64 KiB, which is one chunk of a snapshot.
pub const PAGE_WORDS: usize = 1 << 14;
const PAGE_SHIFT: u32 = 14;

/// A page of a plane.
pub type Page = [u32; PAGE_WORDS];

fn new_page() -> Box<Page> {
    // A zeroed vector is a lazily mapped allocation, so the page costs host memory when it is written.
    vec![0u32; PAGE_WORDS].into_boxed_slice().try_into().expect("a page is PAGE_WORDS long")
}

/// One plane: its pages by index, a missing page being zero. Trailing missing pages are not kept.
#[derive(Default)]
pub struct Plane {
    pages: Vec<Option<Box<Page>>>,
}

impl Plane {
    /// Word `at`; 0 where no page is.
    #[inline(always)]
    pub fn get(&self, at: u32) -> u32 {
        let at = at as usize;
        match self.pages.get(at >> PAGE_SHIFT) {
            Some(Some(p)) => p[at & (PAGE_WORDS - 1)],
            _ => 0,
        }
    }

    fn held(&self) -> usize {
        self.pages.iter().flatten().count()
    }

    fn has(&self, page: usize) -> bool {
        matches!(self.pages.get(page), Some(Some(_)))
    }

    fn trim(&mut self) {
        while matches!(self.pages.last(), Some(None)) {
            self.pages.pop();
        }
    }
}

static NO_PLANE: Plane = Plane { pages: Vec::new() };

#[derive(Default)]
struct Planes {
    planes: Vec<Plane>,
    /// Pages held across every plane, and the most there have been since power-on.
    resident: usize,
    peak: usize,
}

pub struct TextureStore {
    inner: UnsafeCell<Planes>,
    /// Units of 256 bits one bank of one raster manager's texture memory holds.
    bank_units: u32,
    /// Pages the planes together may hold.
    cap_pages: usize,
}

impl Default for TextureStore {
    /// A 64 MB raster manager's texture memory: the InfiniteReality2's, which a store has unless its
    /// product says otherwise.
    fn default() -> Self {
        Self::with_capacity_bytes(64 << 20)
    }
}

// SAFETY: see the module doc: writes happen only while no worker runs (the producer
// drains first), and reads never overlap them.
unsafe impl Sync for TextureStore {}

impl TextureStore {
    /// A store for a raster manager whose texture memory is `bytes` long, in two banks.
    pub fn with_capacity_bytes(bytes: u64) -> Self {
        let bank_units = (bytes / 2 / UNIT_BYTES).min(u32::MAX as u64 / UNIT_WORDS as u64) as u32;
        // A lane of two bytes is the least a texel takes, and a word stands for it. A pool is at least a page
        // to each bank, so a toy store can hold both.
        let cap_pages = ((bytes / 2).div_ceil(PAGE_WORDS as u64)).max(2) as usize;
        TextureStore { inner: UnsafeCell::new(Planes::default()), bank_units, cap_pages }
    }

    /// Units of 256 bits one bank holds.
    #[cfg(test)]
    pub fn bank_units(&self) -> u32 {
        self.bank_units
    }

    fn bank_words(&self) -> usize {
        self.bank_units as usize * UNIT_WORDS as usize
    }

    /// Pages the planes together may hold: the product's texture memory, a word to a 16-bit lane.
    pub fn cap_pages(&self) -> usize {
        self.cap_pages
    }

    /// Whether `units` units from `address` lie inside a bank.
    pub fn contains(&self, address: u32, units: u64) -> bool {
        address as u64 + units <= self.bank_units as u64
    }

    /// Whether writing `rows` runs of `run` words, `stride` apart from word `start` of plane `plane`, would
    /// leave the store within its ceiling. Pages already held cost nothing, and a run is taken to need
    /// its pages though a zero word allocates none, so a yes is a guarantee. Rows must not overlap.
    pub fn fits(&self, plane: u16, start: u64, run: u64, stride: u64, rows: u64) -> bool {
        // SAFETY: as for `plane`.
        let st = unsafe { &*self.inner.get() };
        let pl = st.planes.get(plane as usize).unwrap_or(&NO_PLANE);
        let (bank, mut missing, mut next) = (self.bank_words() as u64, 0usize, 0usize);
        for r in 0..rows {
            let first = start + r * stride;
            if run == 0 || first >= bank {
                break;
            }
            let last = (first + run - 1).min(bank - 1);
            let (p0, p1) = ((first >> PAGE_SHIFT) as usize, (last >> PAGE_SHIFT) as usize);
            for pg in p0.max(next)..=p1 {
                missing += !pl.has(pg) as usize;
            }
            next = next.max(p1 + 1);
            if st.resident + missing > self.cap_pages {
                return false;
            }
        }
        st.resident + missing <= self.cap_pages
    }

    /// Zero the `n` words of plane `plane` from `from`. A page the range covers is given back, a page it
    /// part covers is zeroed in the range, and none is made. A range past the bank is cut at its end
    /// (`contains` is the caller's check). Call only with the tiles drained.
    pub fn clear(&self, plane: u16, from: u32, n: usize) {
        // SAFETY: no reader runs (the caller drained the tiles).
        let st = unsafe { &mut *self.inner.get() };
        let end = (from as usize + n).min(self.bank_words());
        let from = from as usize;
        let Some(pl) = st.planes.get_mut(plane as usize).filter(|_| from < end) else { return };
        let mut freed = 0;
        for pg in from >> PAGE_SHIFT..=(end - 1) >> PAGE_SHIFT {
            let (lo, hi) = (pg << PAGE_SHIFT, (pg + 1) << PAGE_SHIFT);
            let Some(slot) = pl.pages.get_mut(pg) else { break };
            if from <= lo && hi <= end {
                freed += slot.take().is_some() as usize;
            } else if let Some(p) = slot {
                p[from.max(lo) - lo..end.min(hi) - lo].fill(0);
            }
        }
        pl.trim();
        st.resident -= freed;
    }

    /// Give plane `plane` back: its words are gone and its memory is returned. Call only with the tiles drained.
    pub fn drop_plane(&self, plane: u16) {
        // SAFETY: as for `clear`.
        let st = unsafe { &mut *self.inner.get() };
        if let Some(pl) = st.planes.get_mut(plane as usize) {
            st.resident -= pl.held();
            pl.pages = Vec::new();
        }
    }

    /// Write word `at` of plane `plane`. False when the word is not written: past the bank, or the page it
    /// needs would pass the ceiling. A zero into a page that is not held writes nothing and is true. Call only
    /// with the tiles drained.
    pub fn set(&self, plane: u16, at: u32, v: u32) -> bool {
        // SAFETY: as for `clear`.
        let st = unsafe { &mut *self.inner.get() };
        if plane as usize >= PLANES || at as usize >= self.bank_words() {
            return false;
        }
        let (pg, i) = (at as usize >> PAGE_SHIFT, at as usize & (PAGE_WORDS - 1));
        if st.planes.len() <= plane as usize {
            st.planes.resize_with(PLANES, Plane::default);
        }
        let pl = &mut st.planes[plane as usize];
        if !pl.has(pg) {
            if v == 0 {
                return true;
            }
            if st.resident >= self.cap_pages {
                return false;
            }
            if pl.pages.len() <= pg {
                pl.pages.resize_with(pg + 1, || None);
            }
            pl.pages[pg] = Some(new_page());
            st.resident += 1;
            st.peak = st.peak.max(st.resident);
        }
        if let Some(Some(p)) = pl.pages.get_mut(pg) {
            p[i] = v;
        }
        true
    }

    /// Word `at` of plane `plane`; 0 where nothing has been written.
    #[inline(always)]
    pub fn get(&self, plane: u16, at: u32) -> u32 {
        self.plane(plane).get(at)
    }

    /// Plane `plane`, so a loop over one level's texels reaches them without a lookup of the plane each.
    #[inline(always)]
    pub fn plane(&self, plane: u16) -> &Plane {
        // SAFETY: readers run while nothing writes.
        unsafe { &*self.inner.get() }.planes.get(plane as usize).unwrap_or(&NO_PLANE)
    }

    /// The planes that hold pages, in order, with the length of each one's page table.
    pub fn used(&self) -> Vec<(u16, usize)> {
        // SAFETY: as for `plane`.
        unsafe { &*self.inner.get() }.planes.iter().enumerate().filter(|(_, d)| d.held() > 0).map(|(i, d)| (i as u16, d.pages.len())).collect()
    }

    /// Page `page` of plane `plane`, if it is held.
    pub fn page(&self, plane: u16, page: usize) -> Option<&Page> {
        self.plane(plane).pages.get(page)?.as_deref()
    }

    /// Pages a plane can have: those a bank's words fill.
    pub fn plane_pages(&self) -> usize {
        self.bank_words().div_ceil(PAGE_WORDS)
    }

    /// Take `pages` as plane `plane`, as a snapshot saved them. False, and nothing changed, for a plane the
    /// store has not, a table longer than a bank, or pages that would take the store past its ceiling. Call
    /// only with the tiles drained.
    pub fn set_pages(&self, plane: u16, pages: Vec<Option<Box<Page>>>) -> bool {
        // SAFETY: as for `clear`.
        let st = unsafe { &mut *self.inner.get() };
        if plane as usize >= PLANES || pages.len() > self.plane_pages() {
            return false;
        }
        if st.planes.len() <= plane as usize {
            st.planes.resize_with(PLANES, Plane::default);
        }
        let mut pl = Plane { pages };
        pl.trim();
        let (old, new) = (st.planes[plane as usize].held(), pl.held());
        if st.resident - old + new > self.cap_pages {
            return false;
        }
        st.resident = st.resident - old + new;
        st.peak = st.peak.max(st.resident);
        st.planes[plane as usize] = pl;
        true
    }

    /// Pages held across all planes.
    pub fn resident_pages(&self) -> usize {
        // SAFETY: as for `plane`.
        unsafe { &*self.inner.get() }.resident
    }

    /// The most pages held at once since power-on.
    pub fn peak_pages(&self) -> usize {
        // SAFETY: as for `plane`.
        unsafe { &*self.inner.get() }.peak
    }

    /// Forget every image (power-on). Call only with the tiles drained.
    pub fn reset(&self) {
        // SAFETY: as for `clear`.
        let st = unsafe { &mut *self.inner.get() };
        st.planes.clear();
        st.resident = 0;
        st.peak = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_is_where_its_address_puts_it_zeroed_and_bounded() {
        let t = TextureStore::default();
        assert!(t.set(0, 17, 0xdead_beef) && t.set(1, 3, 7));
        assert_eq!((t.resident_pages(), t.used()), (2, vec![(0, 1), (1, 1)]));
        assert_eq!((t.get(0, 17), t.get(0, 16), t.get(0, 99), t.get(1, 3), t.get(7, 0)), (0xdead_beef, 0, 0, 7, 0));
        assert!(t.set(0, 20, 0) && t.set(5, 4, 0), "a zero into a page that is not held writes nothing");
        assert_eq!(t.resident_pages(), 2);
        assert!(!t.set(0, t.bank_units() * UNIT_WORDS, 1), "past the bank");
        t.clear(0, 17, 1);
        assert_eq!(t.get(0, 17), 0, "cleared again");
        t.drop_plane(0);
        assert_eq!((t.used(), t.resident_pages()), (vec![(1, 1)], 1));
        t.reset();
        assert_eq!((t.resident_pages(), t.peak_pages()), (0, 0));
    }

    /// A page is made by the first nonzero word in it: a single word at the top of the bank holds one page, not a bank.
    #[test]
    fn a_word_at_the_top_of_a_bank_holds_one_page() {
        let t = TextureStore::default();
        let top = t.bank_units() * UNIT_WORDS - 1;
        assert!(t.set(3, top, 9));
        assert_eq!((t.resident_pages(), t.get(3, top), t.get(3, top - 1)), (1, 9, 0));
        assert!(t.page(3, t.plane_pages() - 1).is_some() && t.page(3, 0).is_none());
        t.clear(3, 0, usize::MAX / 2);
        assert_eq!(t.resident_pages(), 0, "a clear over the plane gives the pages back");
    }

    /// A clear gives back only the pages it covers and zeroes the rest in range, making none.
    #[test]
    fn a_clear_covers_whole_pages_and_part_pages() {
        let t = TextureStore::default();
        let w = PAGE_WORDS as u32;
        for pg in 0..3 {
            assert!(t.set(0, pg * w + 5, 1));
        }
        t.clear(0, 4, PAGE_WORDS + 3);
        assert_eq!((t.resident_pages(), t.get(0, 5), t.get(0, w + 5), t.get(0, 2 * w + 5)), (3, 0, 0, 1), "pages 0 and 1 are only part covered");
        t.clear(0, 0, 2 * PAGE_WORDS);
        assert_eq!((t.resident_pages(), t.used()), (1, vec![(0, 3)]));
        t.clear(1, 0, 10);
        assert_eq!(t.resident_pages(), 1, "a clear makes nothing");
    }

    /// The pages held across all planes stay within the ceiling, `fits` says so beforehand and `set` refuses past it.
    #[test]
    fn the_planes_together_stay_within_the_ceiling() {
        // 4 MiB of texture memory: 2 Mi words of lanes, 128 pages; a bank is 64 pages.
        let t = TextureStore::with_capacity_bytes(4 << 20);
        assert_eq!((t.cap_pages(), t.plane_pages()), (128, 64));
        let w = PAGE_WORDS as u64;
        for (plane, pages) in [(0u16, 64u32), (1, 63)] {
            for pg in 0..pages {
                assert!(t.set(plane, pg * PAGE_WORDS as u32, 1), "plane {plane} page {pg}");
            }
        }
        assert_eq!(t.resident_pages(), 127);
        assert!(t.fits(2, 0, 1, 1, 1) && !t.fits(2, 0, 2 * w, 2 * w, 1) && t.fits(0, 0, 64 * w, 64 * w, 1), "pages held cost nothing");
        assert!(t.set(2, 0, 1) && !t.set(2, PAGE_WORDS as u32, 1), "the 129th page is refused");
        assert!(t.set(2, 5, 2), "a page already held takes more words");
        assert!(!t.fits(1, 63 * w, 1, 1, 1));
        assert_eq!(t.peak_pages(), 128);
        t.drop_plane(0);
        assert!(t.fits(2, w, 1, 1, 1) && t.set(2, PAGE_WORDS as u32, 1));
    }

    /// Rows that share a page cost it once.
    #[test]
    fn rows_that_share_a_page_cost_it_once() {
        // A bank of 3 pages and a pool of 6, with 4 held.
        let t = TextureStore::with_capacity_bytes(12 * PAGE_WORDS as u64);
        assert_eq!((t.plane_pages(), t.cap_pages()), (3, 6));
        for (plane, pages) in [(1u16, 3u32), (2, 1)] {
            for pg in 0..pages {
                assert!(t.set(plane, pg * PAGE_WORDS as u32, 1));
            }
        }
        let w = PAGE_WORDS as u64;
        // 64 rows of 4 words, 256 apart, are all in page 0.
        assert!(t.fits(0, 0, 4, 256, 64));
        assert!(t.fits(0, 0, 1, w, 2) && !t.fits(0, 0, 1, w, 3));
    }

    /// Snapshots set a plane whole and refuse one past a bank or the ceiling.
    #[test]
    fn a_plane_from_a_snapshot_is_checked() {
        let t = TextureStore::with_capacity_bytes(4 << 20);
        let page = || {
            let mut p = new_page();
            p[2] = 3;
            Some(p)
        };
        assert!(t.set_pages(3, vec![None, page()]));
        assert_eq!((t.used(), t.get(3, PAGE_WORDS as u32 + 2), t.resident_pages()), (vec![(3, 2)], 3, 1));
        assert!(!t.set_pages(3, (0..t.plane_pages() + 1).map(|_| None).collect()), "a table past the bank");
        assert!(!t.set_pages(PLANES as u16, vec![page()]), "no such plane");
        assert!(t.set_pages(4, (0..64).map(|_| page()).collect()) && t.set_pages(5, (0..63).map(|_| page()).collect()));
        assert!(!t.set_pages(6, vec![page(), page()]), "past the ceiling");
        assert!(t.set_pages(4, vec![]) && t.set_pages(5, vec![]));
        assert_eq!((t.used(), t.resident_pages()), (vec![(3, 2)], 1), "refused: unchanged");
        assert!(t.set_pages(3, vec![]) && t.used().is_empty());
    }

    /// A product with more texture memory holds addresses a smaller one refuses.
    #[test]
    fn the_capacity_is_the_products() {
        let small = TextureStore::default();
        let big = TextureStore::with_capacity_bytes(256 << 20);
        assert_eq!((small.bank_units(), big.bank_units()), (1 << 20, 4 << 20));
        assert!(!small.contains(1 << 20, 1) && big.contains(1 << 20, 1));
        assert!(!big.contains(4 << 20, 1));
        assert_eq!((small.cap_pages(), big.cap_pages()), (2048, 8192));
        assert_eq!(TextureStore::with_capacity_bytes(64).cap_pages(), 2, "at least a page to a bank");
    }
}
