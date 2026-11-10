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
//! | 2 | 15:12 | the destination's B field offset |
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
//! source, word 4 bit 31 picks A (front) or B. A destination in memory takes
//! the field word 2 bits 15:12 names, whatever word 2 bit 31 says (the X
//! server's window moves copy a 16-byte pixel as two 8-byte halves, the
//! second with bit 31 set and word 2 at 4). A drawable destination with bit
//! 31 clear takes it too. With bit 31 set it is the window's front, which is
//! the buffer the window displays, so it follows a swap: A when buffer 0 is
//! displayed, else the B field word 2 names. A's offset is the pixel mode's
//! (the library sends one in word 7 bits 23:20, which is not decoded: the
//! layout lists that bit range as unknown). ESTABLISHED: the destination and
//! the front bit are those words and that bit. INFERRED: that the front bit
//! means the displayed buffer, from the window's draw mask, which the kernel
//! exchanges with a swap, and from one trace of a copy into the front of a
//! colour-index window (cicopy: bit 31 set, word 2 at 2, the back field,
//! and GL reads the copied pixels from the front). UNKNOWN: how the board
//! turns the bit into a buffer, and what a back copy does after a swap, which
//! stays the field word 2 names.
//! Before a copy the library points the pixel mode (block 0x017) at the
//! destination, whose layout gives the destination's packing.
//!
//! Word 7 bits 31 and 30 say the source is walked in decreasing x and y,
//! and a BEGIN-0xa image open around the copy says which way the
//! destination is walked: its box, reversed where x2 < x1 or y2 < y1. On
//! an axis where the two disagree the copy mirrors (traced with pixtest
//! on the GL library's copies: glCopyPixels at zoom 1, -1 sends a box
//! reversed in both axes with bit 31 alone, so x is straight and y
//! mirrored; an overlapping copy whose destination lies right of its
//! source sends a box reversed in x with bit 31, so nothing mirrors and the
//! walk clears the overlap; glCopyTexImage sends a box reversed in y with
//! neither bit, so the texture is flipped). With no image open (the X
//! server's window moves, which set the bits for their overlapping walks)
//! nothing mirrors but a copy between a drawable and memory, whose rows the
//! library lists bottom first. The model reads the whole source before it
//! writes, so the walk's order itself does not matter.
//!
//! A destination the drawable bit names is a window: its pixels pass the
//! window clip and id test, as a GL pixel copy's. A destination given by
//! address is not tested: the X server's window moves copy outside the clip
//! rectangles in force (traced), as screen-to-screen copies do.

use super::gpu_unit::{CopyOp, CopySelect, CopyWindow};
use super::pipe::backend::{widen8, DestMem};
use super::pipe::ring::BackQuad;
use super::regs::buzz::{CopyMixed, ImageWord0, ReadHeader, WriteFormat, WriteHeader};
use super::bits::sextract32;
use super::board::Board;
use super::consts::{tex, ODSY_DID_MAIN_SHIFT, ODSY_PLANE_MASK_SHIFT};
use super::field::{Codec, Field, Layout, FIELD_FOUR, FIELD_INDEX, FIELD_ONE, FIELD_PACKED, FIELD_TWO};
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

    /// Word 4: the source field's description.
    pub fn read_header(&self) -> ReadHeader {
        ReadHeader::new(self.w[4])
    }

    /// Operand 2 bit 31: the copy writes the destination's front buffer.
    pub fn dst_front(&self) -> bool {
        CopyMixed::new(self.w[2]).dst_front()
    }

    pub fn src(&self) -> Side {
        let w = &self.w;
        let h = self.read_header();
        Side {
            base: w[0] & ADDR_MASK,
            drawable: w[0] & DRAWABLE != 0,
            x: w[1] & 0xffff,
            y: w[1] >> 16,
            stride: w[2] & PITCH_MASK,
            pixel_log2: ((w[4] >> 16) & 0xf) as u8 + 1,
            at: 2 * (if h.front() { h.field_a() } else { h.field_b() }) as u8,
            bytes: 2 << h.size_field(),
            format: w[4] & 0xff,
        }
    }

    /// Word 7's write format (not checked against a board).
    pub fn write_format(&self) -> WriteFormat {
        WriteHeader::new(self.w[7]).format()
    }

    /// Word 7 bit 27: the copy is a textured rectangle (a zoom sets it too)
    /// and not the short copy path.
    pub fn transform(&self) -> bool {
        WriteHeader::new(self.w[7]).transform()
    }

    /// The read and write field sizes, bits 15:13 of words 4 and 7.
    pub fn same_width(&self) -> bool {
        self.read_header().size_field() == WriteHeader::new(self.w[7]).size_field()
    }

    /// The destination with its field as operand 2 names it, bits 15:12: the
    /// B field. For a drawable destination `copy_block` picks the buffer
    /// (see the module).
    pub fn dst(&self) -> Side {
        let w = &self.w;
        Side {
            base: ImageWord0::new(w[5]).ref_(),
            drawable: w[5] & DRAWABLE != 0,
            x: w[6] & 0xffff,
            y: w[6] >> 16,
            stride: (w[2] >> 16) & PITCH_MASK,
            pixel_log2: ((w[7] >> 16) & 0xf) as u8 + 1,
            at: 2 * CopyMixed::new(w[2]).dst_field_b() as u8,
            bytes: 2 << WriteHeader::new(w[7]).size_field(),
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
        // The id bytes are the front end's own: no wait (Board::id_compare_passes).
        let did = DestMem::of(self.vram_ptr(), &self.dest_views()).did.load1(x, y)? as u32;
        let mode = self.regs().dispmode((did >> ODSY_DID_MAIN_SHIFT) as usize);
        super::frame::mode_fields(mode).into_iter().find(|f| f.at == at).map(|f| f.codec)
    }

    /// The destination with its buffer chosen: for a drawable whose copy
    /// has its front bit set, the field of the buffer the window displays at
    /// the destination's first pixel (one copy has one window), A at offset
    /// `front_at` for buffer 0, else the B field the operands name. Any other
    /// destination is as `CopyOperands::dst()` has it.
    fn copy_dst_buffer(&mut self, c: &CopyOperands, dst: Side, front_at: u8) -> Side {
        if !dst.drawable || !c.dst_front() {
            return dst;
        }
        let (x, y) = (dst.x as i64 + self.win.origin[0] as i64, dst.y as i64 + self.win.origin[1] as i64);
        match self.draw_buffer(true, x, y) {
            0 => Side { at: front_at, ..dst },
            _ => dst,
        }
    }

    /// Block 0x2031 (see the module).
    pub fn copy_block(&mut self) {
        let c = CopyOperands::decode(&self.cmd.operand);
        let (w, h) = c.size();
        let (src, dst) = (c.src(), c.dst());

        if w == 0 || h == 0 {
            return;
        }
        // The copy's own rectangles come back below; the CPU unit waits
        // for everything.
        let gpu = self.gpu_unit();
        if !gpu {
            self.raster_wait();
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
        let dst = self.copy_dst_buffer(&c, dst, layout.front.at);
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
        let hdr = c.read_header();
        let ab = src.drawable.then(|| {
            [hdr.field_a(), hdr.field_b()].map(|u| side_rows(&Side { at: 2 * u as u8, ..src }, src_codec(2 * u as u8), src_base))
        });
        let front = hdr.front();
        let to = side_rows(&dst, dst_codec, dst_base);
        let (ox, oy) = if src.drawable { (self.win.read_origin[0], self.win.read_origin[1]) } else { (0, 0) };
        let (dx0, dy0) = if dst.drawable { (self.win.origin[0], self.win.origin[1]) } else { (0, 0) };
        // Which axes the copy mirrors (see the module): inside a BEGIN-0xa
        // image, those where word 7's walk bits and the box's direction
        // disagree; outside one, the rows of a copy between a drawable and
        // memory.
        let (mirror_x, mirror_y) = if self.image.active {
            let reversed = |a: u32, b: u32| b < a;
            (
                (c.w[7] >> 31 != 0) != reversed(self.image.x1, self.image.x2),
                ((c.w[7] >> 30) & 1 != 0) != reversed(self.image.y1, self.image.y2),
            )
        } else {
            (false, src.drawable != dst.drawable)
        };
        // A copy of whole window-id pixels writes the id and overlay bytes
        // the drawing mask selects, under the plane mask, as Board::paint_aux()
        // does: the X server fills a menu's overlay by copying a tile under
        // mask 0xc (traced), and the id bytes it passes over keep their ids.
        let byte_masks = whole_wid.then(|| {
            let m = (self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) as u8;
            let b = self.draw_buffers();
            [if b & 1 != 0 { m } else { 0 }, if b & 2 != 0 { m } else { 0 }]
        });
        let filter = {
            let sb = &self.tex_regs().scale_bias;
            (0..4).any(|c| sb[c * 2] != tex::SCALE_BIAS_ONE || sb[c * 2 + 1] != 0)
        };
        // A colour index is not a colour. The X server tiles a colour-index
        // root window by copying a seed block across the screen with read
        // byte 0x08 (traced), the read that names one component in a 2-byte
        // field; the same read serves luminance and alpha, so the index
        // meaning comes from the write side. Unpacked as a luminance texel
        // the 8-bit index, held as 0x00nn, came out as its top byte, 0,
        // twice: a black background. The field moves as it is when all hold
        // (the Odyssey RE notes, not checked against a board):
        //  - the short copy path: word 7 bit 27 clear; a textured rectangle,
        //    a zoom included, keeps converting;
        //  - the copy writes COLOR_INDEX (word 7 bits 4:0);
        //  - read and write fields are the same size (bits 15:13 of words 4
        //    and 7), here the 2-byte one;
        //  - the read is the one-component 2-byte one above;
        //  - the scale and bias (block 0x203b, separate state, not in these
        //    words) are the identity.
        // Known limitation: what a non-identity scale or bias does to an
        // index copy is unknown, so that case converts as before.
        // The seed's own DMA upload names the same format word and already
        // lands as an index, because `dma_fields` takes the pixel-mode
        // surface's layout over the format byte; the copy is the one place
        // that read the byte differently.
        let one_component = Codec::Texel { bytes: 2, field: FIELD_ONE };
        let index = !filter
            && !c.transform()
            && c.write_format() == WriteFormat::Index
            && c.same_width()
            && src_codec(src.at) == one_component
            && matches!(dst_codec, Codec::Texel { field: FIELD_INDEX, bytes: 2 });
        // A copy inside a depth-tested image (glCopyPixels under the depth
        // test, traced: the copy comes inside a BEGIN-0xa image whose box
        // carries the raster depth) goes through the fragment back end
        // pixel by pixel, as the image's own pixels do (Board::image_put).
        // It needs the decoded words, so it is never a raw move.
        let fragment = self.image.active && dst.drawable && self.image.fragment.is_some();
        // Bytes move as they are when both sides hold a word the same way.
        let raw = !fragment && (index || (src_codec(src.at) == dst_codec && !filter));

        if gpu {
            // The unit copies within its own memory, in order with the
            // records and with no readback, when it can take the copy.
            if !fragment {
                let op = self.copy_op(&c, &src, &dst, &from, ab.as_ref(), &to, (ox, oy), (dx0, dy0), (mirror_x, mirror_y), raw, byte_masks, filter);
                if self.raster.run_now(|ctx| ctx.gpu.as_mut().is_some_and(|g| g.copy(&op))) {
                    return;
                }
            }
            // Otherwise the source and destination rectangles come back,
            // current in the mirror, and the CPU copies.
            let region = |rows: &Rows, x: i32, y: i32| {
                self.raster.run_now(|ctx| {
                    if let Some(g) = &mut ctx.gpu {
                        if let Err(e) = g.read_back_view(&rows.v, x, y, x.wrapping_add(w as i32 - 1), y.wrapping_add(h as i32 - 1)) {
                            crate::dlog!(crate::devlog::LogModule::Odyssey, "sgi-odyssey: gpu readback: {e}");
                        }
                    }
                });
            };
            let (sx, sy) = ((src.x as i32).wrapping_add(ox), (src.y as i32).wrapping_add(oy));
            region(&from, sx, sy);
            if let Some(f) = &ab {
                region(&f[0], sx, sy);
                region(&f[1], sx, sy);
            }
            region(&to, (dst.x as i32).wrapping_add(dx0), (dst.y as i32).wrapping_add(dy0));
        }
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
        if filter && !raw {
            self.copy_post_filter(&mut words, !src.drawable);
        }
        for y in 0..h {
            let row = if mirror_y { h - 1 - y } else { y };
            for x in 0..w {
                let col = if mirror_x { w - 1 - x } else { x };
                let (px, py) = ((dst.x + col) as i32 + dx0, (dst.y + row) as i32 + dy0);
                if dst.drawable && !self.pixel_allowed(px as i64, py as i64) {
                    continue;
                }
                let i = (y * w + x) as usize;
                if fragment {
                    let Some(f) = self.image.fragment.as_deref() else { continue };
                    let word = words[i];
                    let src4 = [word >> 24, word >> 16, word >> 8, word].map(|c| widen8(c & 0xff));
                    let zq = f.1;
                    match self.image.gpu_slot {
                        // The unit tests and stores it in order, as a fragment.
                        Some(slot) => {
                            let q = BackQuad { x: px, y: py, cov: 1, zq: [zq, 0, 0, 0], rgba: [src4.map(|c| c as u16), [0; 4], [0; 4], [0; 4]], ..Default::default() };
                            self.raster.run_now(|ctx| ctx.gpu.as_mut().expect("the unit").push_frag(&q, &slot));
                        }
                        None => f.0.fragment_q(&self.dest_mem(), px, py, zq, src4),
                    }
                } else if let (true, Some(m)) = (raw, byte_masks) {
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

    /// The copy as the GPU unit takes it (`GpuRaster::copy`): the sides as
    /// views, the per-pixel buffer choice, the masks, the filter's words and
    /// the destination window's tests.
    #[allow(clippy::too_many_arguments, reason = "the copy's decoded sides")]
    fn copy_op(&mut self, c: &CopyOperands, src: &Side, dst: &Side, from: &Rows, ab: Option<&[Rows; 2]>, to: &Rows, (ox, oy): (i32, i32), (dx0, dy0): (i32, i32), mirror: (bool, bool), raw: bool, byte_masks: Option<[u8; 2]>, filter: bool) -> CopyOp {
        let did = DestMem::of(self.vram_ptr(), &self.dest_views()).did.v;
        let (views, select) = match ab {
            None => ([from.v, from.v], CopySelect::Fixed(0)),
            Some(f) => {
                let front = c.read_header().front();
                let select = if self.read_drawable_offscreen() {
                    CopySelect::Fixed(self.read_buffer(front, 0, 0) as usize & 1)
                } else {
                    CopySelect::PerPixel { did, displayed: self.front.lock().displayed_buffer, front }
                };
                ([f[0].v, f[1].v], select)
            }
        };
        let byte_mask = byte_masks.map(|m| {
            let n = to.v.len() as usize;
            (m[0] as u64) << (8 * (n - 1)) | (m[1] as u64) << (8 * (n.max(2) - 2))
        });
        let filter = filter.then(|| {
            let sb = &self.tex_regs().scale_bias;
            (std::array::from_fn(|i| sextract32(sb[i], 0, tex::SCALE_BIAS_BITS)), !src.drawable)
        });
        let window = dst.drawable.then(|| {
            let mut clip = [[0i32; 4]; 4];
            for (i, r) in self.win.clip_rect.iter().enumerate().take(4) {
                clip[i] = [r.x0, r.y0, r.x1, r.y1];
            }
            CopyWindow { width: self.dest_views().colour[0].width, clip, clip_kept: self.win.clip_kept, id_compare: self.win.id_compare, did }
        });
        CopyOp {
            src: views,
            select,
            dst: to.v,
            size: c.size(),
            src_xy: ((src.x as i32).wrapping_add(ox), (src.y as i32).wrapping_add(oy)),
            dst_xy: ((dst.x as i32).wrapping_add(dx0), (dst.y as i32).wrapping_add(dy0)),
            mirror,
            raw,
            byte_mask,
            filter,
            window,
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
    /// G, B, A order: byte b of a lane becomes b / 255 * scale + bias as a
    /// byte (signed: the byte as a signed value over 255), scale and bias
    /// signed 3.12 fixed point (ESTABLISHED as the layout), clamped to the
    /// byte and rounded to the nearest, ties up; computed exactly in
    /// integers, the same on both raster units. An alpha lane whose bias is full scale (0x1000) is the
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
        let lane = |c: usize| (sextract32(sb[c * 2], 0, tex::SCALE_BIAS_BITS) as i64, sextract32(sb[c * 2 + 1], 0, tex::SCALE_BIAS_BITS) as i64);

        for word in words.iter_mut() {
            let mut out = *word;

            for c in 0..4usize {
                let (scale, bias) = lane(c);
                let shift = 24 - 8 * c as u32;
                let byte = (*word >> shift) as u8;
                let x = if signed { byte as i8 as i64 } else { byte as i64 };
                // x / 255 * scale + bias, times 255 and 4096: the byte times
                // the scale plus 255 times the bias, exact.
                let v = if c == 3 && bias >= tex::SCALE_BIAS_ONE as i64 { 255 } else { ((x * scale + 255 * bias).clamp(0, 255 * 4096) + 2048) >> 12 };

                out = (out & !(0xff << shift)) | ((v as u32) << shift);
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

    /// cicopy's back-to-front glCopyPixels on a 12-bit colour-index window
    /// (traced, V12): word 2 0x80502050 has bit 31 set with bits 15:12 at 2,
    /// the back field, and the destination is word 5, the draw drawable.
    #[test]
    fn a_gl_copy_into_the_front_buffer_is_flagged_front_with_the_back_field_named() {
        let c = CopyOperands::decode(&[0x3000_0019, 0x0094_0014, 0x8050_2050, 0x0020_0040, 0x1233_1071, 0x3000_0010, 0x0044_0096, 0x3833_0061]);
        let (s, d) = (c.src(), c.dst());
        assert!(s.drawable && d.drawable && c.dst_front());
        assert_eq!((s.at, d.at, d.bytes), (4, 4, 2), "back to the back field, 2 bytes, until the board picks the buffer");
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
