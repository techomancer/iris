//! The copy, block 0x2031: eight operands that name both rectangles in
//! full, so one operation moves pixels between any two places in board
//! memory: the X server's scrolls and window-id copies, GL's pixel copies,
//! and the library's texture conversions through scratch images.
//!
//! The operand layout is where the GL library's copy builder packs its
//! fields, checked against a raw recording of a signed-texture upload
//! (`ignore/odsy/docs/measured.md` section 10):
//!
//! | word | bits | |
//! |---|---|---|
//! | 0 | 28:0 | source address; bit 28 with the low bits 0x19: the read drawable |
//! | 1 | | source x (15:0), y (31:16) |
//! | 2 | 31 | destination front |
//! | 2 | 27:16, 11:0 | destination and source stride, in tiles |
//! | 2 | 15:12 | the destination field's offset |
//! | 3 | | width (15:0), height (31:16) |
//! | 4 | 31, 27:24, 23:20 | source front, B and A field offsets |
//! | 4 | 19:16, 15:13 | source pixel size and field size, 2 << code bytes |
//! | 4 | 7:0 | the source field's code (7:3) and class (2:0) |
//! | 5 | 28:0 | destination address; bit 28 with 0x10: the draw drawable |
//! | 6 | | destination x, y |
//! | 7 | 23:20 | another field offset, not used (see below) |
//! | 7 | 19:16, 15:13 | destination pixel size and field size |
//!
//! Field offsets count 2-byte units from the start of the pixel. For the
//! source, word 4 bit 31 picks A (front) or B. The destination's field is
//! word 2 bits 15:12, whatever word 2 bit 31 says (INFERRED from two traces
//! that agree on nothing else): the X server's window moves copy a 16-byte
//! pixel as two 8-byte halves, the second with bit 31 set, word 2 at 4 and
//! word 7's bits 23:20 at 0; a glCopyPixels into the back buffer of a
//! double-buffered 8:8:8:8 window has bit 31 clear, word 2 at 2 (the back
//! field) and word 7 at 6 (the front field).
//! Before a copy the library points the pixel mode (block 0x017) at the
//! destination, whose layout gives the destination's packing.
//!
//! Word 7 bits 31 and 30 are set in the X server's window moves, whose
//! destinations lie right of and below their overlapping sources (traced),
//! and read as the walk's direction (INFERRED; the signed upload's copy, to
//! another surface, has 31 set and 30 clear). The model reads the whole
//! source before it writes, which gives what an ordered walk gives, so it
//! does not need them.
//!
//! A destination the drawable bit names is a window: its pixels pass the
//! window clip and id test, as a GL pixel copy's. A destination given by
//! address is not tested: the X server's window moves copy outside the clip
//! rectangles in force (traced), as screen-to-screen copies do.

use super::bits::sextract32;
use super::board::Board;
use super::consts::{tex, ODSY_DID_MAIN_SHIFT, ODSY_PLANE_MASK_SHIFT};
use super::field::{Codec, Field, Layout, FIELD_FOUR, FIELD_ONE, FIELD_PACKED, FIELD_TWO};
use super::view::{Rows, View, MAX_DIM};
use super::vram::{Format, Surface, ADDR_MASK, DRAWABLE, PITCH_MASK};

/// One rectangle of a copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Side {
    /// In 16-byte units; for a drawable, the drawable register's.
    pub base: u32,
    pub drawable: bool,
    pub x: u32,
    pub y: u32,
    pub stride: u32,
    pub pixel_log2: u8,
    /// The field's first byte in the pixel, and its size.
    pub at: u8,
    pub bytes: u8,
    /// The field's code and class (word 4 bits 7:0; the destination's are
    /// not decoded and read 0).
    pub format: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CopyOperands {
    pub w: [u32; 8],
}

impl CopyOperands {
    pub fn decode(op: &[u32]) -> CopyOperands {
        let mut w = [0; 8];
        w.copy_from_slice(&op[..8]);
        CopyOperands { w }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.w[3] & 0xffff, self.w[3] >> 16)
    }

    pub fn src(&self) -> Side {
        let w = &self.w;
        let front = w[4] >> 31 != 0;
        Side {
            base: w[0] & ADDR_MASK,
            drawable: w[0] & DRAWABLE != 0,
            x: w[1] & 0xffff,
            y: w[1] >> 16,
            stride: w[2] & PITCH_MASK,
            pixel_log2: ((w[4] >> 16) & 0xf) as u8 + 1,
            at: 2 * (if front { (w[4] >> 20) & 0xf } else { (w[4] >> 24) & 0xf }) as u8,
            bytes: 2 << ((w[4] >> 13) & 7),
            format: w[4] & 0xff,
        }
    }

    pub fn dst(&self) -> Side {
        let w = &self.w;
        Side {
            base: w[5] & ADDR_MASK,
            drawable: w[5] & DRAWABLE != 0,
            x: w[6] & 0xffff,
            y: w[6] >> 16,
            stride: (w[2] >> 16) & PITCH_MASK,
            pixel_log2: ((w[7] >> 16) & 0xf) as u8 + 1,
            at: 2 * ((w[2] >> 12) & 0xf) as u8,
            bytes: 2 << ((w[7] >> 13) & 7),
            format: 0,
        }
    }
}

/// How a field of `bytes` bytes with code and class `format` holds a word:
/// the pairs the library's reads name (measured.md section 6), four equal
/// components otherwise.
/// Bytes a field of codec c holds.
pub(super) fn codec_bytes(c: Codec) -> u8 {
    match c {
        Codec::Lanes => 8,
        Codec::Byte => 1,
        Codec::Texel { bytes, .. } => bytes,
    }
}

pub(super) fn source_codec(bytes: u8, format: u32) -> Codec {
    let (class, code) = (format & 7, (format >> 3) & 0x1f);
    match (bytes, class, code) {
        (2, 0, 1) => Codec::Texel { bytes: 2, field: FIELD_ONE },
        (2, 1, 0) => Codec::Texel { bytes: 2, field: FIELD_TWO },
        (2, 1, 1) => Codec::Texel { bytes: 2, field: FIELD_PACKED },
        (4, 2, 5) => Codec::Texel { bytes: 4, field: FIELD_TWO },
        (8.., _, _) => Codec::Lanes,
        (b, _, _) => Codec::Texel { bytes: b, field: FIELD_FOUR },
    }
}

impl Board {
    /// How the read drawable's field at offset `at` holds a colour at plane
    /// pixel (x, y): as the display mode of the window the pixel belongs to
    /// (0x2900 + its main id) says, as the scan-out reads it. The pixel mode
    /// names the copy's destination by then, and the copy's own format byte
    /// is the packing it asks for, not the window's. None where the mode
    /// names no field there.
    pub fn drawable_codec(&self, x: i32, y: i32, at: u8) -> Option<Codec> {
        let did = self.dest_mem().did.load1(x, y)? as u32;
        let mode = self.regs().dispmode((did >> ODSY_DID_MAIN_SHIFT) as usize);
        super::frame::mode_fields(mode).into_iter().find(|f| f.at == at).map(|f| f.codec)
    }

    /// Block 0x2031 (see the module).
    pub fn copy_block(&mut self) {
        let c = CopyOperands::decode(&self.cmd.operand);
        let (w, h) = c.size();
        let (src, dst) = (c.src(), c.dst());

        self.raster_wait();
        if w == 0 || h == 0 {
            return;
        }
        // The destination's format is the pixel mode's field at the offset
        // the copy names; word 7's field size is the unit moved, the colour
        // as the source holds it (INFERRED from the traces: the library's
        // reads copy a 4-byte colour into the 8-byte lanes or 2-byte 4:4:4:4
        // of a scratch image, converting it). The window-id buffer is the
        // exception: its two 1-byte fields are one 2-byte pixel to the X
        // server's copies, which move it whole, under the masks.
        let mode = Format::decode(self.pixel_mode[2]);
        let layout = Layout::of(mode);
        let whole_wid = layout.front.codec == Codec::Byte && dst.at == 0 && dst.bytes == 2;
        let dst_codec = [layout.front, layout.back]
            .into_iter()
            .find(|f| f.at == dst.at && !whole_wid)
            .map_or_else(|| source_codec(dst.bytes, 0), |f| f.codec);
        let mem = self.vram.ptr();
        let side_rows = |s: &Side, codec: Codec, base: u32| {
            let surface = Surface::new(base, s.stride, s.pixel_log2);
            Rows::new(mem, View::colour(&surface, Field { at: s.at, codec }, MAX_DIM))
        };
        let src_base = if src.drawable { self.read_word & ADDR_MASK } else { src.base };
        let dst_base = if dst.drawable { self.draw_word & ADDR_MASK } else { dst.base };
        // A drawable source holds its colour as its window's display mode
        // says; memory as the copy's format byte names.
        let src_codec = |at: u8| {
            let named = source_codec(src.bytes, src.format);
            if !src.drawable {
                return named;
            }
            let (x, y) = ((src.x as i32).wrapping_add(self.win.read_origin[0]), (src.y as i32).wrapping_add(self.win.read_origin[1]));
            self.drawable_codec(x, y, at).filter(|c| codec_bytes(*c) == src.bytes).unwrap_or(named)
        };
        let from = side_rows(&src, src_codec(src.at), src_base);
        // A drawable source's front is the buffer its window shows, as a
        // DMA read's is (Board::read_buffer()): buffer 0 the A field, 1 the B.
        let w4 = c.w[4];
        let ab = src.drawable.then(|| {
            [(w4 >> 20) & 0xf, (w4 >> 24) & 0xf].map(|u| side_rows(&Side { at: 2 * u as u8, ..src }, src_codec(2 * u as u8), src_base))
        });
        let front = w4 >> 31 != 0;
        let to = side_rows(&dst, dst_codec, dst_base);
        let (ox, oy) = if src.drawable { (self.win.read_origin[0], self.win.read_origin[1]) } else { (0, 0) };
        let (dx0, dy0) = if dst.drawable { (self.win.origin[0], self.win.origin[1]) } else { (0, 0) };
        // A copy between a drawable and memory lists its rows bottom first
        // (INFERRED from guest traces of glReadPixels, glCopyTexImage and
        // glDrawPixels of the signed types; no known operand bit says so).
        let flip = src.drawable != dst.drawable;
        // A copy of whole window-id pixels writes the id and overlay bytes
        // the drawing mask selects, under the plane mask, as Board::paint_aux()
        // does: the X server fills a menu's overlay by copying a tile under
        // mask 0xc (traced), and the id bytes it passes over keep their ids.
        let byte_masks = whole_wid.then(|| {
            let m = (self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) as u8;
            let b = self.draw_buffers();
            [if b & 1 != 0 { m } else { 0 }, if b & 2 != 0 { m } else { 0 }]
        });
        let sb = &self.tex_regs().scale_bias;
        let filter = (0..4).any(|c| sb[c * 2] != tex::SCALE_BIAS_ONE || sb[c * 2 + 1] != 0);
        // Bytes move as they are when both sides hold a word the same way.
        let raw = src_codec(src.at) == dst_codec && !filter;

        // The whole source first, so an overlapping copy reads what was there.
        let n = w as usize * h as usize;
        let mut words = vec![0u32; n];
        let mut raws = vec![0u64; if raw { n } else { 0 }];
        for y in 0..h {
            for x in 0..w {
                let (sx, sy) = ((src.x + x) as i32 + ox, (src.y + y) as i32 + oy);
                let i = (y * w + x) as usize;
                let from = match &ab {
                    Some(f) => &f[self.read_buffer(front, sx as i64, sy as i64) as usize & 1],
                    None => &from,
                };
                if raw {
                    raws[i] = from.load_raw(sx, sy);
                } else {
                    words[i] = from.load_word(sx, sy).unwrap_or(0);
                }
            }
        }
        if filter {
            self.copy_post_filter(&mut words, !src.drawable);
        }
        for y in 0..h {
            let row = if flip { h - 1 - y } else { y };
            for x in 0..w {
                let (px, py) = ((dst.x + x) as i32 + dx0, (dst.y + row) as i32 + dy0);
                if dst.drawable && !self.pixel_allowed(px as i64, py as i64) {
                    continue;
                }
                let i = (y * w + x) as usize;
                if let (true, Some(m)) = (raw, byte_masks) {
                    // The id byte, then the overlay byte: the field's top two.
                    let n = to.v.len();
                    let mask = (m[0] as u64) << (8 * (n - 1)) | (m[1] as u64) << (8 * (n.max(2) - 2));
                    let v = to.load_raw(px, py);
                    to.store_raw(px, py, (v & !mask) | (raws[i] & mask));
                } else if raw {
                    to.store_raw(px, py, raws[i]);
                } else {
                    to.store_word(px, py, words[i]);
                }
            }
        }
    }

    /// Whether the board reads the bytes of a copy's source as signed values
    /// when it filters them. INFERRED, and nothing board-visible establishes
    /// it: the library sends a signed type's burst and write code as the
    /// unsigned type's, and programs the final pass as ordinary scale and
    /// bias words. The pass the library builds for a signed type is 2x + 1/4096
    /// from memory the burst filled, which gives GL's result only if the bits
    /// are read as signed. So the model reads memory as signed and a
    /// drawable as unsigned.
    ///
    /// The post-filter (block 0x203b) on a copy's words, lane by lane in R,
    /// G, B, A order: byte b of a lane becomes clamp(x * scale + bias, 0, 1)
    /// as a byte, with x = b / 255 (signed: the byte as a signed value over
    /// 255), scale and bias signed 3.12 fixed point (ESTABLISHED as the
    /// layout). An alpha lane whose bias is full scale (0x1000) is the
    /// library's opaque overwrite for a source without alpha: 1.0.
    /// glPixelTransfer's scale and bias on an image pixel drawn into a
    /// window: the library programs block 0x203b before the image (traced,
    /// pixtest draw/RGBA/UNSIGNED_BYTE/scale-bias) and the board applies it
    /// on the way, as for a copy (the V10 passes the case). A colour index
    /// passes as it is.
    pub fn image_transfer(&self, word: u32, is_index: bool) -> u32 {
        let sb = &self.tex_regs().scale_bias;
        if is_index || (0..4).all(|c| sb[c * 2] == tex::SCALE_BIAS_ONE && sb[c * 2 + 1] == 0) {
            return word;
        }
        let mut w = [word];
        self.copy_post_filter(&mut w, false);
        w[0]
    }

    pub fn copy_post_filter(&self, words: &mut [u32], signed: bool) {
        let sb = &self.tex_regs().scale_bias;
        let lane = |c: usize| {
            (sextract32(sb[c * 2], 0, tex::SCALE_BIAS_BITS) as f64 / tex::SCALE_BIAS_ONE as f64,
             sextract32(sb[c * 2 + 1], 0, tex::SCALE_BIAS_BITS) as f64 / tex::SCALE_BIAS_ONE as f64)
        };

        for word in words.iter_mut() {
            let mut out = *word;

            for c in 0..4usize {
                let (scale, bias) = lane(c);
                let shift = 24 - 8 * c as u32;
                let byte = (*word >> shift) as u8;
                let x = if signed { byte as i8 as f64 } else { byte as f64 } / 255.0;
                let v = if c == 3 && bias >= 1.0 { 1.0 } else { (x * scale + bias).clamp(0.0, 1.0) };

                out = (out & !(0xff << shift)) | (((v * 255.0 + 0.5).floor() as u32) << shift);
            }
            *word = out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The signed upload's copy, from the raw recording.
    const SIGNED: [u32; 8] = [0x0016b800, 0, 0x80020020, 0x00040004, 0x8002500b, 0x0016f800, 0, 0xb801204b];

    #[test]
    fn a_window_move_copies_the_second_half_onto_the_second_half() {
        let c = CopyOperands::decode(&[0x2000, 0x019e0110, 0x80504050, 0x00680272, 0x84435004, 0x2000, 0x01cc01ca, 0xf2034000]);
        let (s, d) = (c.src(), c.dst());
        assert_eq!((s.at, s.bytes, 1u32 << s.pixel_log2), (8, 8, 16));
        assert_eq!((d.at, d.bytes, 1u32 << d.pixel_log2), (8, 8, 16));
    }

    #[test]
    fn a_gl_copy_into_the_back_buffer_lands_in_the_back_field() {
        let c = CopyOperands::decode(&[0x3000_0019, 0x00bb_003c, 0x0050_2050, 0x0009_0015, 0x1263_3042, 0x3000_0010, 0x00a2_0064, 0x3863_204b]);
        let (s, d) = (c.src(), c.dst());
        assert!(s.drawable && d.drawable);
        assert_eq!((s.at, d.at, d.bytes), (4, 4, 4), "back to back: field B, bytes 4-7");
    }

    #[test]
    fn the_recorded_copy_decodes() {
        let c = CopyOperands::decode(&SIGNED);
        let (s, d) = (c.src(), c.dst());
        assert_eq!((s.base, s.stride, 1u32 << s.pixel_log2, s.at, s.bytes, s.drawable), (0x16b800, 0x20, 8, 0, 8, false));
        assert_eq!((d.base, d.stride, 1u32 << d.pixel_log2, d.at, d.bytes), (0x16f800, 2, 4, 0, 4));
        assert_eq!(c.size(), (4, 4));
        assert_eq!(source_codec(s.bytes, s.format), Codec::Lanes);
    }
}
