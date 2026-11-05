//! The board's memory as the board addresses it: one byte array of the size
//! the strap reports, and surfaces over it named the way the command stream
//! names them (block 0x017, the texture level descriptors, the copy operands).
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
    bytes: Vec<u8>,
}

impl Vram {
    /// Memory of at least size bytes, a power of two, so an address masks
    /// into it (the texture unit's reads, Span::texels()).
    pub fn new(size: u64) -> Vram {
        Vram { bytes: vec![0; size.next_power_of_two() as usize] }
    }

    pub fn size(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// The memory as the raster unit and the display reach it. The array is
    /// sized once, at allocation, and never moves, so the pointer stays
    /// good for the board's life.
    pub fn ptr(&mut self) -> VramPtr {
        VramPtr { ptr: self.bytes.as_mut_ptr(), len: self.bytes.len() }
    }

    /// ptr() from a shared borrow: the memory is written through the pointer
    /// by units that race as the board's do (see VramPtr).
    pub fn ptr_shared(&self) -> VramPtr {
        VramPtr { ptr: self.bytes.as_ptr() as *mut u8, len: self.bytes.len() }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.bytes
    }

#[cfg(test)]
    /// The bytes of pixel (x, y), or None past the end of memory (a guest
    /// bug; the caller reports it).
    pub fn pixel(&self, s: &Surface, x: u32, y: u32) -> Option<&[u8]> {
        let at = s.offset(x, y) as usize;
        self.bytes.get(at..at + s.bytes_per_pixel() as usize)
    }

    #[cfg(test)]
    pub fn pixel_mut(&mut self, s: &Surface, x: u32, y: u32) -> Option<&mut [u8]> {
        let at = s.offset(x, y) as usize;
        self.bytes.get_mut(at..at + s.bytes_per_pixel() as usize)
    }

    #[cfg(test)]
    /// Copy a w x h rectangle between two surfaces of the same pixel size,
    /// byte for byte. Overlapping rectangles copy as if through a temporary.
    pub fn copy_rect(&mut self, src: &Surface, sx: u32, sy: u32, dst: &Surface, dx: u32, dy: u32, w: u32, h: u32) -> bool {
        if src.bpp_log2 != dst.bpp_log2 {
            return false;
        }
        let bpp = src.bytes_per_pixel() as usize;
        let mut rect = vec![0u8; w as usize * h as usize * bpp];
        for y in 0..h {
            for x in 0..w {
                let Some(p) = self.pixel(src, sx + x, sy + y) else { return false };
                let at = (y as usize * w as usize + x as usize) * bpp;
                rect[at..at + bpp].copy_from_slice(p);
            }
        }
        for y in 0..h {
            for x in 0..w {
                let at = (y as usize * w as usize + x as usize) * bpp;
                let Some(p) = self.pixel_mut(dst, dx + x, dy + y) else { return false };
                p.copy_from_slice(&rect[at..at + bpp]);
            }
        }
        true
    }
}

/// Board memory shared with the raster unit and the display thread, which
/// read and write it with no lock, as the board's own units do: the front
/// end waits for the raster unit before it touches memory itself
/// (Board::raster_wait), and a display frame that reads a pixel mid-store
/// shows it for one frame.
#[derive(Clone, Copy)]
pub(super) struct VramPtr {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the array outlives every user (Board owns it and stops the raster
// and display threads before it drops), and never moves.
unsafe impl Send for VramPtr {}
unsafe impl Sync for VramPtr {}

impl VramPtr {
    /// # Safety
    /// The board's memory must still exist; writers and readers race as
    /// the hardware's do (see the type).
    pub unsafe fn bytes<'a>(self) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
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
    fn copy_between_board_addresses() {
        // The signed-texture upload's copy, scratch to level, minus the
        // conversion: the bytes land where the level's select names.
        let mut m = Vram::new(128 << 20);
        let a = Surface::new(SCRATCH[0], SCRATCH[1], 2);
        let b = Surface::from_select(&LEVEL);
        for y in 0..4 {
            for x in 0..4 {
                m.pixel_mut(&a, x, y).unwrap().copy_from_slice(&[x as u8, y as u8, 7, 9]);
            }
        }
        assert!(m.copy_rect(&a, 0, 0, &b, 0, 0, 4, 4));
        assert_eq!(m.pixel(&b, 3, 2).unwrap(), &[3, 2, 7, 9]);
        assert!(m.pixel(&b, 0, 1u32 << 20).is_none());
    }
}
