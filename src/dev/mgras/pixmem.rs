//! Pixel memory: the RDRAM behind one raster engine, as pages.
//!
//! The board has no linear framebuffer. RDRAM bytes are 9 bits wide, and a
//! page (2KB on each of the six chips) is one tile of the screen: 192 x 16
//! pixels of a 36-bit buffer, or 768 x 16 pixels of the 9-bit overlay
//! buffer, four to a 36-bit word. A buffer is a run of pages starting at a
//! page pointer (`DRBpointers` for drawing, the XMAP DIB pointer for
//! scanout); pixel (x, y) is in page `ptr + (y / 16) * xtiles + x / tile
//! width`, `xtiles` coming from `DRBsize`. See ignore/gr4/PP1.h for how the
//! real board spreads a page over its chips; only the raw RDRAM window
//! (the RE4's indirect device space) could see that order, so here a page
//! is 3072 words in plain row order and an overlay word holds four
//! neighbouring pixels, lowest x in the low bits.
//!
//! One raster engine only: with two, scanlines alternate between them and
//! each has its own pages.

/// Pages behind one raster engine (2MB RDRAMs, 2KB pages).
pub const PAGES: usize = 1024;
pub const TILE_W: usize = 192;
pub const TILE_H: usize = 16;
pub const PAGE_WORDS: usize = TILE_W * TILE_H;
/// Overlay tile width: four 9-bit pixels a word.
pub const OVERLAY_TILE_W: usize = 4 * TILE_W;
/// Bits of a word the buffers use (four 9-bit bytes).
pub const WORD_MASK: u64 = 0xF_FFFF_FFFF;

/// `DRBsize` bits 1:0: overlay tiles per row; bits 5:2: 36-bit tiles per
/// row.
pub fn xtiles(drbsize: u32) -> (u32, u32) {
    match ((drbsize >> 2) & 0xF, drbsize & 3) {
        // Not programmed yet: the PROM's 1280x1024 layout (7 and 2).
        (0, _) => (7, 2),
        t => t,
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A 36-bit buffer: one pixel a word.
    Wide,
    /// The 9-bit overlay buffer: four pixels a word, 9 bits each.
    Overlay,
}

/// Where a buffer is and how its words hold pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffer {
    pub ptr: u32,
    pub kind: Kind,
    /// Tiles per row for this kind.
    pub xtiles: u32,
}

impl Buffer {
    pub fn new(ptr: u32, kind: Kind, drbsize: u32) -> Self {
        let (x36, x9) = xtiles(drbsize);
        Buffer { ptr: ptr & 0x3FF, kind, xtiles: if kind == Kind::Wide { x36 } else { x9 } }
    }

    /// Word index, bit shift and value mask of pixel (x, y).
    #[inline]
    pub fn locate(&self, x: u32, y: u32) -> (usize, u32, u64) {
        let (x, y) = (x as usize, y as usize);
        let tile_w = if self.kind == Kind::Wide { TILE_W } else { OVERLAY_TILE_W };
        let page = (self.ptr as usize + (y / TILE_H) * self.xtiles as usize + x / tile_w) % PAGES;
        let base = page * PAGE_WORDS + (y % TILE_H) * TILE_W;
        match self.kind {
            Kind::Wide => (base + x % TILE_W, 0, WORD_MASK),
            Kind::Overlay => (base + (x % OVERLAY_TILE_W) / 4, 9 * (x as u32 % 4), 0x1FF),
        }
    }
}

/// The pages. Plain data, valid zeroed (build in place, never by value).
#[repr(C)]
pub struct PixMem {
    pub words: [u64; PAGES * PAGE_WORDS],
}

impl PixMem {
    #[inline]
    pub fn get(&self, b: &Buffer, x: u32, y: u32) -> u64 {
        let (i, shift, mask) = b.locate(x, y);
        (self.words[i] >> shift) & mask
    }

    #[inline]
    pub fn put(&mut self, b: &Buffer, x: u32, y: u32, v: u64) {
        let (i, shift, mask) = b.locate(x, y);
        self.words[i] = (self.words[i] & !(mask << shift)) | ((v & mask) << shift);
    }

    /// Row `y` of a buffer, pixels `0..out.len()`, as u32s.
    pub fn read_row(&self, b: &Buffer, y: u32, out: &mut [u32]) {
        match b.kind {
            Kind::Wide => {
                for (tx, chunk) in out.chunks_mut(TILE_W).enumerate() {
                    let (i, _, _) = b.locate((tx * TILE_W) as u32, y);
                    let n = chunk.len();
                    for (o, w) in chunk.iter_mut().zip(&self.words[i..i + n]) {
                        *o = *w as u32;
                    }
                }
            }
            Kind::Overlay => {
                for (x, o) in out.iter_mut().enumerate() {
                    *o = self.get(b, x as u32, y) as u32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_pixels_tile_by_192_by_16() {
        let b = Buffer::new(576, Kind::Wide, 0x31E);
        assert_eq!(b.locate(0, 0), (576 * PAGE_WORDS, 0, WORD_MASK));
        assert_eq!(b.locate(191, 15).0, 576 * PAGE_WORDS + 15 * 192 + 191);
        // Next tile across, then the next tile row (7 tiles at 1280).
        assert_eq!(b.locate(192, 0).0, 577 * PAGE_WORDS);
        assert_eq!(b.locate(0, 16).0, 583 * PAGE_WORDS);
        // The last row of a 1280x1024 buffer ends at page 1023.
        assert_eq!(b.locate(1279, 1023).0 / PAGE_WORDS, 1023);
    }

    #[test]
    fn overlay_packs_four_pixels_a_word() {
        let b = Buffer::new(448, Kind::Overlay, 0x31E);
        assert_eq!(b.locate(0, 0), (448 * PAGE_WORDS, 0, 0x1FF));
        assert_eq!(b.locate(3, 0), (448 * PAGE_WORDS, 27, 0x1FF));
        assert_eq!(b.locate(4, 0).0, 448 * PAGE_WORDS + 1);
        assert_eq!(b.locate(768, 0).0, 449 * PAGE_WORDS);
        // 1280x1024 overlay: 2 tiles x 64 rows = 128 pages, up to main's 576.
        assert_eq!(b.locate(1279, 1023).0 / PAGE_WORDS, 575);
        let mut m = crate::dev::mgras::plain::boxed_zeroed::<PixMem>();
        m.put(&b, 1, 5, 0x1AB);
        m.put(&b, 2, 5, 0x0FF);
        assert_eq!((m.get(&b, 0, 5), m.get(&b, 1, 5), m.get(&b, 2, 5)), (0, 0x1AB, 0xFF));
    }
}
