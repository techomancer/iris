//! Decoded-instruction cache for the shadow cache.
//!
//! The full cache model keeps decoded instructions in its cache lines, so the
//! interpreter decodes each instruction once per fill. The shadow cache has no
//! lines, and without this decodes every instruction on every fetch: about
//! half the interpreter's speed.
//!
//! This is a host-side memo of decode work and nothing more. It is keyed by
//! physical page, never consults or changes the architectural cache state the
//! shadow keeps for CACHE instructions, and is exactly coherent with memory,
//! like the rest of the shadow: a word served from here is always the word
//! memory holds now. Self-modifying code takes effect at once, with no stale
//! I-cache window — stricter than hardware, never wrong for correct software.
//!
//! Coherence is by compare: every fetch re-reads the word (and, for a fused
//! pair, the next one) and reuses the decode only if it is unchanged. Exact,
//! and cheap because the read is one host load through the page's own pointer
//! into ppmem's window, taken when the page is first used. Pages that are not
//! RAM (the PROM, MMIO, the low-512KB alias) have no pointer and read through
//! the bus.
//!
//! jitv2's per-page write-generation counters were tried as the validator in
//! jitv2 builds and measured no faster (raw/gen geomean ×1.05 on R4400, ×1.00
//! on R5000, full-length iris-bench), so compare is the only mode: it works
//! the same in every build and every region.
//!
//! A RAM remap (MEMCFG) changes which regions are RAM and which counters back
//! a page: the cache records the region bitmap it was filled under and starts
//! over when it changes. Snapshot restore and power-on flush everything.

use crate::cpu::mips_exec::{DecodedInstr, FLAG_NOT_DECODED};
#[cfg(feature = "opcodefusion")]
use crate::cpu::mips_exec::FLAG_IMM_IS_NEXT;

/// Decoded pages held at once. Direct-mapped on a hash of the page number
/// ([`index`]).
pub const PAGES: usize = 64;
const SLOTS: usize = 1024; // 4 KB / 4
const WORDS: usize = SLOTS / 64;
const EMPTY: u64 = u64::MAX;

struct Page {
    /// Physical page number, or `EMPTY`.
    pfn: u64,
    /// The page's words in ppmem's window, or null to read through the bus.
    host: *const u32,
    /// Slot holds a word read from memory (decoded or not yet).
    valid: [u64; WORDS],
    /// Raw mode: the slot's decode also depends on the following word
    /// (opcode fusion), recorded in `next`.
    fused: [u64; WORDS],
    next: Box<[u32]>,
    slots: Box<[DecodedInstr]>,
}

impl Page {
    fn new() -> Self {
        Self {
            pfn: EMPTY,
            host: std::ptr::null(),
            valid: [0; WORDS],
            fused: [0; WORDS],
            next: vec![0; SLOTS].into_boxed_slice(),
            slots: (0..SLOTS).map(|_| DecodedInstr::default()).collect(),
        }
    }

    fn reset(&mut self, pfn: u64, host: *const u32) {
        self.pfn = pfn;
        self.host = host;
        self.valid = [0; WORDS];
        self.fused = [0; WORDS];
    }
}

pub struct DecodeCache {
    pages: Box<[Page]>,
    /// The ppmem region bitmap the pages were filled under.
    bitmap: u64,
    /// The bitmap's region size, as `ppmem::window_shift`.
    window_shift: u32,
}

impl Default for DecodeCache {
    fn default() -> Self { Self::new() }
}

impl DecodeCache {
    /// The window is larger than 4GB. Set before anything is fetched.
    pub fn set_window_shift(&mut self, shift: u32) {
        self.window_shift = shift;
    }

    pub fn new() -> Self {
        Self { pages: (0..PAGES).map(|_| Page::new()).collect(), bitmap: 0, window_shift: crate::ppmem::BITMAP_SHIFT }
    }

    /// Forget everything: memory may have changed without any counter moving.
    pub fn flush(&mut self) {
        for p in self.pages.iter_mut() {
            p.pfn = EMPTY;
        }
    }

    /// The decoded slot for the instruction at `phys`, filled on a miss.
    ///
    /// `window` and `bitmap` are ppmem's window base
    /// and region bitmap: a page in a RAM region reads its words straight out
    /// of the window. `ic_line` is the I-cache line size: opcode fusion pairs a
    /// word with the next one only within a line. `word` reads a word through
    /// the bus, returning the bus status on failure.
    #[inline(always)]
    pub fn fetch(
        &mut self,
        phys: u64,
        window: *const u8,
        bitmap: u64,
        ic_line: usize,
        word: impl Fn(u64) -> Result<u32, u32>,
    ) -> Result<*const DecodedInstr, u32> {
        // The common case, kept small enough to inline into the interpreter
        // loop: a RAM page already here, the slot filled, its word (and a
        // fused pair's second word) unchanged. One or two host loads.
        if bitmap == self.bitmap {
            let pfn = phys >> 12;
            let k = (phys as usize >> 2) & (SLOTS - 1);
            let (w, bit) = (k >> 6, 1u64 << (k & 63));
            let page = &self.pages[index(pfn)];
            if page.pfn == pfn && !page.host.is_null() && page.valid[w] & bit != 0 {
                let now = unsafe { *page.host.add(k) };
                if now == page.slots[k].raw
                    && (page.fused[w] & bit == 0 || unsafe { *page.host.add(k + 1) } == page.next[k])
                {
                    return Ok(&page.slots[k] as *const DecodedInstr);
                }
            }
        }
        self.fetch_slow(phys, window, bitmap, ic_line, word)
    }

    /// Everything else: a new bitmap, a page change, a bus page, a fill.
    #[inline(never)]
    #[cold]
    fn fetch_slow(
        &mut self,
        phys: u64,
        window: *const u8,
        bitmap: u64,
        ic_line: usize,
        word: impl Fn(u64) -> Result<u32, u32>,
    ) -> Result<*const DecodedInstr, u32> {
        if bitmap != self.bitmap {
            self.flush();
            self.bitmap = bitmap;
        }
        let pfn = phys >> 12;
        let k = (phys as usize >> 2) & (SLOTS - 1);
        let (w, bit) = (k >> 6, 1u64 << (k & 63));
        let window_shift = self.window_shift;
        let page = &mut self.pages[index(pfn)];
        let _ = ic_line;
        let host_for = || -> *const u32 {
            if !window.is_null() && crate::ppmem::bitmap_maps_at(bitmap, window_shift, phys) {
                unsafe { window.add((pfn << 12) as usize) as *const u32 }
            } else {
                std::ptr::null()
            }
        };

        if page.pfn != pfn {
            page.reset(pfn, host_for());
        }
        if page.valid[w] & bit != 0 {
            let now = read(page.host, k, phys, &word)?;
            let next_ok = page.fused[w] & bit == 0
                || read(page.host, k + 1, phys + 4, &word)? == page.next[k];
            if now == page.slots[k].raw && next_ok {
                return Ok(&page.slots[k] as *const DecodedInstr);
            }
        }

        // Fill: the executor decodes the slot in place on first use.
        let raw = read(page.host, k, phys, &word)?;
        #[cfg(feature = "opcodefusion")]
        let host = page.host;
        let slot = &mut page.slots[k];
        slot.raw = raw;
        slot.flags = FLAG_NOT_DECODED;
        page.fused[w] &= !bit;
        // Opcode fusion: hand decode the following word, as the full model's
        // fill does, unless this is the last word of an I-cache line (fusion
        // never crosses one, so the pair stays in this page).
        #[cfg(feature = "opcodefusion")]
        if (phys as usize) & (ic_line - 1) != ic_line - 4 {
            let next = read(host, k + 1, phys + 4, &word)?;
            slot.imm = next;
            slot.flags |= FLAG_IMM_IS_NEXT;
            page.next[k] = next;
            page.fused[w] |= bit;
        }
        page.valid[w] |= bit;
        Ok(slot as *const DecodedInstr)
    }
}

/// The table entry for page `pfn`: a multiplicative hash, not the low bits.
/// A kernel that colours pages (the O2's IRIX gives a page the physical
/// colour of its virtual address, and program text starts on aligned
/// addresses) hands out text pages that agree in their low bits, and with
/// `pfn % PAGES` Dhrystone's and libc's text all took entry 0 and evicted
/// each other on every call.
#[inline(always)]
fn index(pfn: u64) -> usize {
    (pfn.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - PAGES.trailing_zeros())) as usize
}

/// Word `k` of a page: one load from its window pointer, or a bus read at
/// `phys` when it has none.
#[inline(always)]
fn read(host: *const u32, k: usize, phys: u64, word: &impl Fn(u64) -> Result<u32, u32>) -> Result<u32, u32> {
    if host.is_null() { word(phys) } else { Ok(unsafe { *host.add(k) }) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    fn mem(words: &[(u64, u32)]) -> RefCell<HashMap<u64, u32>> {
        RefCell::new(words.iter().copied().collect())
    }

    /// A cached slot is served while its word is unchanged and refilled once
    /// it changes.
    #[test]
    fn a_changed_word_is_refilled() {
        {
            let m = mem(&[(0x1000, 0x2400_0001), (0x1004, 0)]);
            let read = |a: u64| Ok(*m.borrow().get(&a).unwrap_or(&0));
            let mut dc = DecodeCache::new();

            let p = dc.fetch(0x1000, std::ptr::null(), 0, 16, read).unwrap();
            unsafe { (*(p as *mut DecodedInstr)).flags = 0 }; // the executor decoded it
            let q = dc.fetch(0x1000, std::ptr::null(), 0, 16, read).unwrap();
            assert_eq!(p, q);
            assert_eq!(unsafe { (*q).flags }, 0, "served decoded, not refilled");

            m.borrow_mut().insert(0x1000, 0x2400_0002);
            let r = dc.fetch(0x1000, std::ptr::null(), 0, 16, read).unwrap();
            assert_eq!(unsafe { (*r).raw }, 0x2400_0002, "refilled");
            assert_ne!(unsafe { (*r).flags } & FLAG_NOT_DECODED, 0);
        }
    }

    /// A bus error on the word reaches the caller as its status.
    #[test]
    fn a_bus_error_is_returned() {
        let mut dc = DecodeCache::new();
        assert_eq!(dc.fetch(0x2000, std::ptr::null(), 0, 16, |_| Err(0x0800_001c)).err(), Some(0x0800_001c));
    }

    /// Pages that share a table entry evict each other rather than alias.
    #[test]
    fn colliding_pages_do_not_alias() {
        let a = 0x1000u64;
        let b = (1..).map(|n| a + n * 4096).find(|&b| index(b >> 12) == index(a >> 12)).unwrap();
        let m = mem(&[(a, 0x1111_1111), (b, 0x2222_2222)]);
        let read = |x: u64| Ok(*m.borrow().get(&x).unwrap_or(&0));
        let mut dc = DecodeCache::new();
        assert_eq!(unsafe { (*dc.fetch(a, std::ptr::null(), 0, 16, read).unwrap()).raw }, 0x1111_1111);
        assert_eq!(unsafe { (*dc.fetch(b, std::ptr::null(), 0, 16, read).unwrap()).raw }, 0x2222_2222);
        assert_eq!(unsafe { (*dc.fetch(a, std::ptr::null(), 0, 16, read).unwrap()).raw }, 0x1111_1111);
    }

    /// Coloured pages, which agree in their low page-number bits, spread over
    /// the table instead of sharing one entry.
    #[test]
    fn coloured_pages_spread_over_the_table() {
        const { assert!(PAGES.is_power_of_two()) };
        let used: std::collections::HashSet<usize> = (0..PAGES as u64).map(|n| index(0x50000 + n * 64)).collect();
        assert!(used.len() > PAGES / 2, "{} of {PAGES} entries", used.len());
    }

    /// A page in a RAM region reads its words through the window pointer, not
    /// the bus; a changed region bitmap (a remap) starts the cache over.
    #[test]
    fn ram_pages_read_through_the_window_and_a_remap_flushes() {
        let mut ram = vec![0u32; 2 * 1024];
        ram[1024 + 3] = 0x2400_0007; // phys 0x100c, page 1 word 3
        let window = ram.as_ptr() as *const u8;
        let bus = |_| -> Result<u32, u32> { panic!("RAM must not go through the bus") };
        let mut dc = DecodeCache::new();
        let p = dc.fetch(0x100c, window, 1, 16, bus).unwrap();
        assert_eq!(unsafe { (*p).raw }, 0x2400_0007);

        // Same bitmap: served (and validated) from the window.
        ram[1024 + 3] = 0x2400_0008;
        assert_eq!(unsafe { (*dc.fetch(0x100c, window, 1, 16, bus).unwrap()).raw }, 0x2400_0008);

        // Region no longer RAM: everything goes to the bus now.
        let p = dc.fetch(0x100c, window, 0, 16, |_| Ok(0x2400_0009)).unwrap();
        assert_eq!(unsafe { (*p).raw }, 0x2400_0009);
    }

    /// Raw mode: a fused slot depends on the next word too, so changing only
    /// the delay slot must refill it.
    #[test]
    #[cfg(feature = "opcodefusion")]
    fn raw_mode_rechecks_the_fused_next_word() {
        let m = mem(&[(0x1000, 0x03e0_0008), (0x1004, 0)]); // jr ra; nop
        let read = |a: u64| Ok(*m.borrow().get(&a).unwrap_or(&0));
        let mut dc = DecodeCache::new();
        let p = dc.fetch(0x1000, std::ptr::null(), 0, 16, read).unwrap();
        unsafe { (*(p as *mut DecodedInstr)).flags = 0 };
        m.borrow_mut().insert(0x1004, 0x2402_0001); // the delay slot is no longer a NOP
        let q = dc.fetch(0x1000, std::ptr::null(), 0, 16, read).unwrap();
        assert_ne!(unsafe { (*q).flags } & FLAG_NOT_DECODED, 0, "refilled");
        assert_eq!(unsafe { (*q).imm }, 0x2402_0001);
    }
}
