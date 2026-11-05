//! The board's memory as the board addresses it: one array of 64-bit words,
//! of the size the strap reports, and surfaces over it named the way the
//! command stream names them (block 0x017, the texture level descriptors,
//! the copy operands).
//!
//! Memory is words, as main memory is. A field narrower than a word, n bytes
//! at byte address a, is the bit range of its word that a 64-bit load of
//! guest memory gives those addresses: bits (8 - a % 8 - n) * 8 up, the
//! lowest address highest. Nothing converts byte order: DMA moves the
//! guest's words in and out whole, and every field is read and written by
//! shift and mask (`VramPtr::load`, `store`).
//!
//! A surface is a base address in 16-byte units, a pitch in 16 x 16-pixel
//! tiles and a pixel size of 2 << n bytes (n, the frame-buffer depth, from
//! bits 15:13 of the pixel mode's status word). Tiles are stored in row-major order of tiles, pixels in
//! row-major order inside a tile. The order inside a tile is the model's
//! choice: nothing the host can read shows the board's, since data reaches
//! memory only through the board's DMA and copies. Measurements and the
//! format word's decode: `ignore/odsy/docs/measured.md` sections 4 and 5,
//! `clanker/plans/odyssey-board-memory.md`.

/// Bytes in one unit of a base address.
pub(super) const UNIT: u64 = 16;
/// Pixels on a side of a tile.
pub(super) const TILE: u32 = 16;
/// The address bits of a base word; the top three are flags.
pub(super) const ADDR_MASK: u32 = 0x1fff_ffff;
/// Bit 28 of a copy's address word: the operand is the current drawable
/// (0x10000019 its draw buffer, 0x10000010 its read buffer), not memory.
pub(super) const DRAWABLE: u32 = 0x1000_0000;
/// Word 1 of a select: the pitch in tiles.
pub(super) const PITCH_MASK: u32 = 0xfff;

/// Fields of the pixel mode's status word (block 0x017 word 2). The two
/// depths are bytes a pixel, 2 << code.
pub(super) mod fmt {
    pub const FB_DEPTH_SHIFT: u32 = 13; // the memory's pixel
    pub const WIN_DEPTH_SHIFT: u32 = 10; // the part of it the visual uses
    pub const PIXEL_FMT_SHIFT: u32 = 7; // how the visual arranges that part
    pub const FIELD_FMT_SHIFT: u32 = 4; // the colour field's components
    pub const MASK3: u32 = 7;
    pub const PIXEL_INDEX: u32 = 1 << 19; // colour-index visual
    pub const SCRATCH: u32 = 1 << 27; // no buffer select, window clip or offset
}

/// A status word, split into its fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Format {
    /// Bytes a memory pixel is 1 << fb_log2.
    pub fb_log2: u8,
    /// Bytes of it the visual uses, 1 << win_log2.
    pub win_log2: u8,
    pub pixel_fmt: u8,
    pub field_fmt: u8,
    pub index: bool,
    pub scratch: bool,
}

impl Format {
    pub fn decode(w: u32) -> Format {
        let f = |s: u32| ((w >> s) & fmt::MASK3) as u8;
        Format {
            fb_log2: f(fmt::FB_DEPTH_SHIFT) + 1,
            win_log2: f(fmt::WIN_DEPTH_SHIFT) + 1,
            pixel_fmt: f(fmt::PIXEL_FMT_SHIFT),
            field_fmt: f(fmt::FIELD_FMT_SHIFT),
            index: w & fmt::PIXEL_INDEX != 0,
            scratch: w & fmt::SCRATCH != 0,
        }
    }
}

/// A rectangle of board memory as a surface: where it starts, its row of
/// tiles and its pixel size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Surface {
    /// In 16-byte units.
    pub base: u32,
    /// Tiles in a row of tiles.
    pub pitch: u32,
    /// Bytes a pixel is 1 << bpp_log2.
    pub bpp_log2: u8,
}

#[cfg(test)]
impl Surface {
    /// Bytes from the base to the end of row of tiles rows-1.
    pub fn bytes(&self, rows: u32) -> u64 {
        (rows.div_ceil(TILE) as u64 * self.pitch as u64 * (TILE * TILE) as u64) << self.bpp_log2
    }

    pub fn width(&self) -> u32 {
        self.pitch * TILE
    }
}

impl Surface {
    pub fn new(base: u32, pitch: u32, bpp_log2: u8) -> Surface {
        Surface { base: base & ADDR_MASK, pitch, bpp_log2 }
    }

#[cfg(test)]
    /// The surface a pixel mode (block 0x017) selects: words 0 (origin), 1
    /// (stride) and 2 (status).
    pub fn from_select(w: &[u32]) -> Surface {
        Surface::new(w[0], w[1] & PITCH_MASK, Format::decode(w[2]).fb_log2)
    }

    pub fn bytes_per_pixel(&self) -> u32 {
        1 << self.bpp_log2
    }

    /// The byte address of pixel (x, y).
    pub fn offset(&self, x: u32, y: u32) -> u64 {
        let tile = (y / TILE) as u64 * self.pitch as u64 + (x / TILE) as u64;
        let pixel = tile * (TILE * TILE) as u64 + ((y % TILE) * TILE + x % TILE) as u64;
        self.base as u64 * UNIT + (pixel << self.bpp_log2)
    }

}

/// The board's memory.
#[derive(Clone, Default)]
pub(super) struct Vram {
    words: Vec<u64>,
}

impl Vram {
    /// Memory of at least size bytes, a power of two, so an address masks
    /// into it.
    pub fn new(size: u64) -> Vram {
        Vram { words: vec![0; (size.next_power_of_two() / 8).max(1) as usize] }
    }

    /// Its size in bytes.
    pub fn size(&self) -> u64 {
        self.words.len() as u64 * 8
    }

    /// The memory as the raster unit and the display reach it. The array is
    /// sized once, at allocation, and never moves, so the pointer stays
    /// good for the board's life.
    pub fn ptr(&mut self) -> VramPtr {
        VramPtr { ptr: self.words.as_mut_ptr(), mask: self.words.len() - 1 }
    }

    /// ptr() from a shared borrow: the memory is written through the pointer
    /// by units that race as the board's do (see VramPtr).
    pub fn ptr_shared(&self) -> VramPtr {
        VramPtr { ptr: self.words.as_ptr() as *mut u64, mask: self.words.len() - 1 }
    }

    pub fn words_mut(&mut self) -> &mut [u64] {
        &mut self.words
    }
}

/// Board memory shared with the raster unit and the display thread, which
/// read and write it with no lock, as the board's own units do: the front
/// end waits for the raster unit before it touches memory itself
/// (Board::raster_wait), and a display frame that reads a pixel mid-store
/// shows it for one frame.
#[derive(Clone, Copy)]
pub(super) struct VramPtr {
    ptr: *mut u64,
    /// Words less one: a word index masks into the memory.
    mask: usize,
}

// SAFETY: the array outlives every user (Board owns it and stops the raster
// and display threads before it drops), and never moves.
unsafe impl Send for VramPtr {}
unsafe impl Sync for VramPtr {}

impl VramPtr {
    /// The word holding byte address a.
    #[inline(always)]
    fn word(&self, a: u64) -> *mut u64 {
        // SAFETY: masked into the array.
        unsafe { self.ptr.add((a >> 3) as usize & self.mask) }
    }

    /// The value of the n bytes (1 to 8) at byte address a, which lie in
    /// one word: the bits of it they are, the lowest address highest.
    #[inline(always)]
    pub fn load(&self, a: u64, n: u32) -> u64 {
        // SAFETY: the word is in the memory; readers and writers race as
        // the board's units do (see the type).
        let w = unsafe { *self.word(a) };
        let shift = 64 - 8 * ((a & 7) as u32 + n);
        if n == 8 { w } else { (w >> shift) & ((1u64 << (8 * n)) - 1) }
    }

    /// Store v as the n bytes at byte address a (load()'s bits).
    #[inline(always)]
    pub fn store(&self, a: u64, n: u32, v: u64) {
        let p = self.word(a);
        // SAFETY: as load().
        unsafe {
            if n == 8 {
                *p = v;
            } else {
                let shift = 64 - 8 * ((a & 7) as u32 + n);
                let m = ((1u64 << (8 * n)) - 1) << shift;
                *p = (*p & !m) | ((v << shift) & m);
            }
        }
    }

    /// The memory's size in bytes.
    pub fn size(&self) -> u64 {
        (self.mask as u64 + 1) * 8
    }

    /// The memory as bytes in address order, for what is a byte stream by
    /// nature: a dump file, the memory hash.
    pub fn address_bytes(&self) -> impl Iterator<Item = u8> + '_ {
        (0..self.size()).map(move |a| self.load(a, 1) as u8)
    }

    /// No memory: a view of a field the surface lacks, which nothing reads.
    pub fn null() -> VramPtr {
        VramPtr { ptr: std::ptr::null_mut(), mask: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The selects the driver sent in the trace of an X start at depth 24 on a
    // 128 MB board (measured.md section 4).
    const FB: [u32; 3] = [0x2000, 0x50, 0x6950];
    const OVERLAY: [u32; 3] = [0x142000, 0x50, 0x20080080];
    const SCRATCH: [u32; 3] = [0x16b800, 0x20, 0x08104851];
    const LEVEL: [u32; 3] = [0x16f800, 0x02, 0x080024d0];

    #[test]
    fn format_words_decode() {
        let f = Format::decode(FB[2]);
        assert_eq!((f.fb_log2, f.win_log2, f.pixel_fmt, f.field_fmt, f.index, f.scratch), (4, 3, 2, 5, false, false));
        let o = Format::decode(OVERLAY[2]);
        assert_eq!((o.fb_log2, o.win_log2, o.pixel_fmt, o.field_fmt, o.index), (1, 1, 1, 0, true));
        assert_eq!(Format::decode(SCRATCH[2]).fb_log2, 3);
        let l = Format::decode(LEVEL[2]);
        assert_eq!((l.fb_log2, l.win_log2, l.field_fmt, l.scratch), (2, 2, 5, true));
    }

    #[test]
    fn frame_buffer_ends_where_the_overlay_starts() {
        let fb = Surface::from_select(&FB);
        let ov = Surface::from_select(&OVERLAY);
        assert_eq!(fb.bytes_per_pixel(), 16);
        assert_eq!(fb.width(), 1280);
        assert_eq!(fb.base as u64 * UNIT + fb.bytes(1024), ov.base as u64 * UNIT);
        // The overlay's 2 bytes a pixel end before the first texture
        // allocation the trace saw.
        assert!(ov.base as u64 * UNIT + ov.bytes(1024) <= 0x16b000 * UNIT);
    }

    #[test]
    fn tiles_are_row_major_and_pixels_row_major_inside() {
        let fb = Surface::from_select(&FB);
        let o = fb.base as u64 * UNIT;
        assert_eq!(fb.offset(0, 0), o);
        assert_eq!(fb.offset(1, 0), o + 16);
        assert_eq!(fb.offset(0, 1), o + 16 * 16);
        assert_eq!(fb.offset(16, 0), o + 256 * 16);
        assert_eq!(fb.offset(0, 16), o + 80 * 256 * 16);
        assert_eq!(fb.offset(1279, 1023), o + fb.bytes(1024) - 16);
    }

    #[test]
    fn fields_are_bit_ranges_of_words() {
        let mut m = Vram::new(1 << 20);
        let p = m.ptr();
        p.store(16, 8, 0x0011_2233_4455_6677);
        assert_eq!(p.load(16, 4), 0x0011_2233);
        assert_eq!(p.load(20, 4), 0x4455_6677);
        assert_eq!(p.load(17, 3), 0x11_2233);
        assert_eq!(p.load(23, 1), 0x77);
        p.store(18, 2, 0xabcd);
        assert_eq!(p.load(16, 8), 0x0011_abcd_4455_6677);
        // Addresses wrap at the memory's end.
        p.store(1 << 20, 4, 0x5a5a_5a5a);
        assert_eq!(p.load(0, 4), 0x5a5a_5a5a);
    }
}
