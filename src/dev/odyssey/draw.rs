//! The board's 2D drawing: colour-buffer access, clipping and id-plane tests,
//! integer pixel paints, rectangle fill and copy, and pixel bursts.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

/// The clip-rectangle runs of one row (Board::row_runs).
#[derive(Clone, Copy)]
pub struct RowRuns {
    pub n: u32,
    pub start: [i64; SGI_ODYSSEY_CLIP_RECTS as usize],
    pub end: [i64; SGI_ODYSSEY_CLIP_RECTS as usize],
}

impl Default for RowRuns {
    fn default() -> Self {
        RowRuns { n: 0, start: [0; SGI_ODYSSEY_CLIP_RECTS as usize], end: [0; SGI_ODYSSEY_CLIP_RECTS as usize] }
    }
}

/// 2D fills carry no alpha channel in this grammar; the framebuffer's fourth
/// byte is modelled fully opaque (the colour plane's red-high R,G,B,A packing,
/// resolved by resolve_row() below).
pub fn pack_pixel(r: u32, g: u32, b: u32) -> u32 {
    (r << 24) | (g << 16) | (b << 8) | 0xff
}

/// The 0x840 ownership test at an in-plane pixel, against id plane `did`
/// and the 0x840 word `id_compare`.
pub fn id_compare_passes(did: &[u8], id_compare: u32, x: i64, y: i64) -> bool {
    let mask = (id_compare >> ODSY_ID_COMPARE_MASK_SHIFT) as u8;

    id_compare & ODSY_ID_COMPARE_ENABLE == 0
        || ((did[(y * SGI_ODYSSEY_WIDTH as i64 + x) as usize] ^ id_compare as u8) & mask) == 0
}

impl Board {
    /// Colour buffer b's board memory and plane (set up by Board::allocate()), b being 0 or
    /// 1 in the displayed-buffer bits' sense (ODSY_REG_DRAW_MASK).
    ///
    /// The planes are the raster unit's: the accessors here and below wait
    /// for its queued jobs before the front end reads or writes them
    /// (raster_unit.rs).
    pub fn buffer_mem(&self, b: u32) -> &BoardMemory {
        self.raster_wait();
        if b != 0 { &self.buffer1_mem } else { &self.mem }
    }

    pub fn buffer_mem_mut(&mut self, b: u32) -> &mut BoardMemory {
        self.raster_wait();
        if b != 0 { &mut self.buffer1_mem } else { &mut self.mem }
    }

    pub fn buffer_plane(&self, b: u32) -> Option<Plane> {
        if b != 0 {
            self.buffer1_mem.plane(self.buffer1_plane)
        } else {
            self.mem.plane(self.colour_plane)
        }
    }

    pub fn buffer_load(&self, b: u32, x: i64, y: i64) -> u32 {
        match self.buffer_plane(b) {
            Some(p) => self.buffer_mem(b).plane_load(&p, x as i32, y as i32),
            None => 0,
        }
    }

    pub fn buffer_store(&mut self, b: u32, x: i64, y: i64, word: u32) {
        if let Some(p) = self.buffer_plane(b) {
            self.buffer_mem_mut(b).plane_store(&p, x as i32, y as i32, word);
        }
    }

    /// Whether the last 0x1017 select named the id-plane destination, whatever
    /// ODSY_DEST_FLAGS it carried.
    pub fn dest_is_id_planes(&self) -> bool {
        (self.dest_buffer_word & !ODSY_DEST_FLAGS) == (ODSY_DEST_ID_PLANES & !ODSY_DEST_FLAGS)
    }

    /// The buffers a colour write lands in, bit b for buffer b, from the last
    /// 0x0424 drawing mask. A right-eye bit folds into its buffer: stereo is not
    /// modelled.
    pub fn draw_buffers(&self) -> u32 {
        (if self.draw_mask & ODSY_DRAW_MASK_BUFFER0 != 0 { 1 } else { 0 })
            | (if self.draw_mask & ODSY_DRAW_MASK_BUFFER1 != 0 { 2 } else { 0 })
    }

    /// The one completion fence, shared by the standalone 0x0a5a host-write
    /// command and the DMA engine's publish(): write the cookie's 16-bit token,
    /// big-endian, to the guest fence slot at the host-write base plus twice the
    /// slot, and pulse the softsync interrupt when the cookie asks.
    ///
    /// The raster unit writes it, behind the raster jobs queued before it, so
    /// the token lands only once the drawing before it is in the planes.
    pub fn fence(&mut self, cookie: u64) {
        let word = (cookie & ODSY_FENCE_WORD_MASK as u64) as u32;
        let slot = word & ODSY_DMA_FENCE_SLOT_MASK;
        let token = (word >> ODSY_DMA_FENCE_TOKEN_SHIFT) & 0xffff;
        let base = (((self.host_write_addr_upper & 0xffff) as u64) << 32) | self.host_write_addr_lower as u64;
        let (guest, front) = (self.guest.clone(), self.front.clone());

        self.raster.submit(move |_| {
            write_guest(&guest, base.wrapping_add(2 * slot as u64), &(token as u16).to_be_bytes());
            if word & ODSY_DMA_FENCE_INTERRUPT != 0 {
                front.lock().deliver_softsync();
            }
        });
    }
}

impl Board {
    /// Rebuild the kept (non-empty) clip rectangles from the raw 0x204b words.
    pub fn clip_update(&mut self) {
        self.clip_kept = 0;
        for i in 0..SGI_ODYSSEY_CLIP_RECTS as usize {
            let lo = self.clip_words[2 * i];
            let hi = self.clip_words[2 * i + 1];
            let r = ClipRect {
                x0: ((lo >> ODSY_CLIP_X_SHIFT) & ODSY_CLIP_FIELD_MASK) as i32,
                y0: (lo & ODSY_CLIP_FIELD_MASK) as i32,
                x1: ((hi >> ODSY_CLIP_X_SHIFT) & ODSY_CLIP_FIELD_MASK) as i32,
                y1: (hi & ODSY_CLIP_FIELD_MASK) as i32,
            };

            if r.x0 <= r.x1 && r.y0 <= r.y1 {
                self.clip_rect[self.clip_kept as usize] = r;
                self.clip_kept += 1;
            }
        }
    }

    /// Whether clip_words is the off-screen form (ODSY_CMD_WINDOW_RESTORE).
    pub fn clip_is_offscreen(&self) -> bool {
        for (i, &word) in self.clip_words.iter().enumerate().take((2 * SGI_ODYSSEY_CLIP_RECTS) as usize) {
            let want = if i == 1 { ODSY_CLIP_OFFSCREEN_HI } else { 0 };

            if (word & ODSY_CLIP_WORD_MASK) != want {
                return false;
            }
        }
        true
    }

    /// The plane bound. The plane itself drops an out-of-plane store, but the
    /// family checks first so the shared code never has to guess whose error it
    /// is. Board::pixel_allowed() adds the window tests for every write except a
    /// copy destination, which calls this helper directly.
    pub fn pixel_in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && x < SGI_ODYSSEY_WIDTH as i64 && y < SGI_ODYSSEY_HEIGHT as i64
    }

    pub fn id_compare_active(&self) -> bool {
        self.id_compare & ODSY_ID_COMPARE_ENABLE != 0
    }

    /// The 0x840 ownership test at an in-plane pixel.
    pub fn id_compare_passes(&self, x: i64, y: i64) -> bool {
        self.raster_wait();
        id_compare_passes(&self.did, self.id_compare, x, y)
    }

    /// The one pixel test every write but a copy destination passes: the plane
    /// bound and the 0x840 id test, then inside at least one kept clip rectangle
    /// -- with none kept nothing is drawn.
    pub fn pixel_allowed(&self, x: i64, y: i64) -> bool {
        if !self.pixel_in_bounds(x, y) || !self.id_compare_passes(x, y) {
            return false;
        }
        for r in &self.clip_rect[..self.clip_kept as usize] {
            if x >= r.x0 as i64 && x <= r.x1 as i64 && y >= r.y0 as i64 && y <= r.y1 as i64 {
                return true;
            }
        }
        false
    }

    /// The pixels of one row that Board::pixel_allowed() passes before its id
    /// test: the plane bound and the union of the kept clip rectangles, as
    /// ascending, disjoint, non-touching [start, end) runs. The 2D fills walk
    /// these a row at a time; they still apply the id test per pixel.
    pub fn row_runs(&self, y: i64) -> RowRuns {
        let mut runs = RowRuns::default();
        let mut n = 0usize;
        let mut kept = 0usize;

        if y < 0 || y >= SGI_ODYSSEY_HEIGHT as i64 {
            return runs;
        }
        for &r in &self.clip_rect[..self.clip_kept as usize] {
            let a = (r.x0 as i64).max(0);
            let b = (r.x1 as i64 + 1).min(SGI_ODYSSEY_WIDTH as i64);
            let mut j = n;

            n += 1;
            if y < r.y0 as i64 || y > r.y1 as i64 || a >= b {
                n -= 1;
                continue;
            }
            // Insertion by start: there are at most four.
            while j != 0 && runs.start[j - 1] > a {
                runs.start[j] = runs.start[j - 1];
                runs.end[j] = runs.end[j - 1];
                j -= 1;
            }
            runs.start[j] = a;
            runs.end[j] = b;
        }
        for i in 0..n {
            if kept != 0 && runs.start[i] <= runs.end[kept - 1] {
                runs.end[kept - 1] = runs.end[kept - 1].max(runs.end[i]);
            } else {
                runs.start[kept] = runs.start[i];
                runs.end[kept] = runs.end[i];
                kept += 1;
            }
        }
        runs.n = kept as u32;
        runs
    }

    /// Where colour buffer b keeps the n pixels of row y from x, when all of
    /// them lie in its plane and its memory: the plane, and the byte offset
    /// and length of the run in Board::buffer_mem(), the same bytes
    /// BoardMemory::plane_store() and BoardMemory::plane_load() reach one
    /// pixel at a time. None otherwise, and the caller goes a pixel at a time.
    pub fn buffer_row(&self, b: u32, x: i64, y: i64, n: i64) -> Option<(Plane, std::ops::Range<usize>)> {
        let m = self.buffer_mem(b);
        let p = self.buffer_plane(b)?;
        let size = plane_bytes_per_pixel(p.format);

        if m.bytes.is_empty() || n <= 0 || x < 0 || y < 0 || x + n > p.width as i64 || y >= p.height as i64 {
            return None;
        }
        let at = p.base.wrapping_add((y as u64).wrapping_mul(p.stride as u64)).wrapping_add((x as u64).wrapping_mul(size as u64));
        let len = (n as u64).wrapping_mul(size as u64);
        if at > m.size || len > m.size - at {
            return None;
        }
        Some((p, at as usize..(at + len) as usize))
    }

    /// A colour write, into each buffer the drawing mask selects.
    pub fn paint_pixel(&mut self, x: i64, y: i64, pixel: u32) {
        let buffers = self.draw_buffers();

        if !self.pixel_allowed(x, y) {
            return;
        }
        for b in 0..ODSY_BUFFERS {
            if buffers & (1u32 << b) != 0 {
                self.buffer_store(b, x, y, pixel);
            }
        }
    }
}

/// The plane word a fill stores: the packed RGB pixel of an RGB fill, or the
/// raw index of a colour-index fill. No colormap is applied here: scanout
/// resolves each pixel through its own drawable.
pub fn fill_pixel(is_index: bool, value: u32) -> u32 {
    if is_index { value & ODSY_INDEX_MASK } else { value }
}

impl Board {
    /// One pixel's id-plane write: the bits mask selects of the id byte when the
    /// drawing mask selects buffer 0, and of the overlay pixel byte when it
    /// selects buffer 1.
    pub fn aux_store(&mut self, at: u64, buffers: u32, value: u8, mask: u8) {
        let at_i = at as usize;

        self.raster_wait();

        if buffers & (1u32 << ODSY_AUX_ID_BUFFER) != 0 {
            self.did[at_i] = (self.did[at_i] & !mask) | (value & mask);
        }
        if buffers & (1u32 << ODSY_AUX_OVERLAY_BUFFER) != 0 {
            self.overlay[at_i] = (self.overlay[at_i] & !mask) | (value & mask);
        }
        if buffers & ((1u32 << ODSY_AUX_ID_BUFFER) | (1u32 << ODSY_AUX_OVERLAY_BUFFER)) == 0 && !self.aux_mask_reported {
            self.aux_mask_reported = true;
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: id-plane write with drawing mask {:#x} selects neither the id nor the overlay byte; dropped",
                self.draw_mask
            );
        }
    }

    /// One pixel the X server draws at the id-plane destination: the index's low
    /// byte combined through op with the byte it lands on -- the overlay byte when
    /// the drawing mask selects buffer 1, else the id byte -- and stored under the
    /// 0x1023 plane mask.
    pub fn paint_aux_at(&mut self, at: u64, index: u32, op: u8) {
        let buffers = self.draw_buffers();
        let mut byte = index as u8;

        self.raster_wait();
        if op as u32 != SGI_LOGIC_COPY {
            let dst = if buffers & (1u32 << ODSY_AUX_OVERLAY_BUFFER) != 0 { self.overlay[at as usize] } else { self.did[at as usize] };

            byte = logic_op(op, index, dst as u32) as u8;
        }
        self.aux_store(at, buffers, byte, (self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) as u8);
    }

    pub fn paint_aux(&mut self, x: i64, y: i64, index: u32, op: u8) {
        if self.pixel_allowed(x, y) {
            self.paint_aux_at(y as u64 * SGI_ODYSSEY_WIDTH as u64 + x as u64, index, op);
        }
    }

    /// One pixel of an integer primitive -- a QUADS rectangle, a pattern quad, a
    /// line, point or polygon pixel -- combined with each selected buffer through
    /// the logical operation 0x0422 names, when 0x0404 enables one for the
    /// colour's kind: bit 19 for an index, bit 20 for an RGB colour. An index
    /// result keeps the index bits; an RGB result keeps the fourth byte opaque.
    /// Images and copies do not come through here: whether the board applies the
    /// operation to them is unknown.
    ///
    /// At the id-plane destination the pixel goes to the id or overlay byte
    /// instead, under the 0x1023 plane mask. Its logical operation reads the
    /// overlay byte when buffer 1 is selected and the id byte otherwise.
    pub fn paint_integer(&mut self, x: i64, y: i64, is_index: bool, value: u32) {
        let src = fill_pixel(is_index, value);
        let enable = if is_index { ODSY_MODE_INDEX_LOGIC_OP } else { ODSY_MODE_COLOR_LOGIC_OP };
        let op = self.gl_ref().cur.logic_op_mode.wrapping_sub(SGI_GL_CLEAR) as u8;
        let buffers = self.draw_buffers();

        if self.dest_is_id_planes() {
            let use_op = (self.pattern_mode & enable) != 0 && op as u32 != SGI_LOGIC_COPY;

            self.paint_aux(x, y, src, if use_op { op } else { SGI_LOGIC_COPY as u8 });
            return;
        }
        if (self.pattern_mode & enable) == 0 || op as u32 == SGI_LOGIC_COPY {
            self.paint_pixel(x, y, src);
            return;
        }
        if !self.pixel_allowed(x, y) {
            return;
        }
        for b in 0..ODSY_BUFFERS {
            if buffers & (1u32 << b) != 0 {
                let out = logic_op(op, src, self.buffer_load(b, x, y));

                self.buffer_store(b, x, y, if is_index { out & ODSY_INDEX_MASK } else { out | 0xff });
            }
        }
    }

    /// Board::paint_integer() at each pixel of row y from x0 up to x1, given the
    /// row's runs, a run of the buffer's bytes at a time. Each pixel is tested,
    /// combined and stored exactly as the per-pixel path does it. Pixels are
    /// independent, so going a buffer at a time rather than a pixel at a time
    /// stores the same bytes.
    pub fn paint_integer_span(&mut self, runs: &RowRuns, y: i64, x0: i64, x1: i64, is_index: bool, value: u32) {
        let src = fill_pixel(is_index, value);
        let enable = if is_index { ODSY_MODE_INDEX_LOGIC_OP } else { ODSY_MODE_COLOR_LOGIC_OP };
        let op = self.gl_ref().cur.logic_op_mode.wrapping_sub(SGI_GL_CLEAR) as u8;
        let logic = (self.pattern_mode & enable) != 0 && op as u32 != SGI_LOGIC_COPY;
        let id_test = self.id_compare_active();
        let buffers = self.draw_buffers();
        let aux = self.dest_is_id_planes();

        for r in 0..runs.n as usize {
            let a = x0.max(runs.start[r]);
            let e = x1.min(runs.end[r]);

            if a >= e {
                continue;
            }
            if aux {
                for x in a..e {
                    if !id_test || self.id_compare_passes(x, y) {
                        self.paint_aux_at(
                            y as u64 * SGI_ODYSSEY_WIDTH as u64 + x as u64,
                            src,
                            if logic { op } else { SGI_LOGIC_COPY as u8 },
                        );
                    }
                }
                continue;
            }
            for b in 0..ODSY_BUFFERS {
                let mut word = [0u8; 4];

                if buffers & (1u32 << b) == 0 {
                    continue;
                }
                let Some((plane, run)) = self.buffer_row(b, a, y, e - a) else {
                    for x in a..e {
                        let mut out = src;

                        if id_test && !self.id_compare_passes(x, y) {
                            continue;
                        }
                        if logic {
                            out = logic_op(op, src, self.buffer_load(b, x, y));
                            out = if is_index { out & ODSY_INDEX_MASK } else { out | 0xff };
                        }
                        self.buffer_store(b, x, y, out);
                    }
                    continue;
                };
                let bpp = plane_bpp(plane.format) as usize;
                let (did, id_compare) = (&self.did, self.id_compare);
                let mem = if b != 0 { &mut self.buffer1_mem } else { &mut self.mem };

                plane.encode(src, &mut word);
                for (x, px) in (a..e).zip(mem.bytes[run].chunks_exact_mut(bpp)) {
                    if id_test && !id_compare_passes(did, id_compare, x, y) {
                        continue;
                    }
                    if !logic {
                        px.copy_from_slice(&word[..bpp]);
                        continue;
                    }
                    let out = logic_op(op, src, plane.decode(px));
                    plane.encode(if is_index { out & ODSY_INDEX_MASK } else { out | 0xff }, px);
                }
            }
        }
    }

    /// The solid rectangle fill: BEGIN(QUADS)/COLOR_3UB or COLOR_INDEX/four VERTEX_2I
    /// corners/END drives the solid rectangle fill. Board::gl_end() routes a
    /// QUADS fill here when pattern mode is off, and to Board::pattern_fill()
    /// when it is on. is_index selects whether value is a packed RGB pixel or a
    /// PseudoColor index resolved per pixel.
    pub fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, is_index: bool, value: u32) {
        for row in 0..h {
            let runs = self.row_runs(y as i64 + row as i64);
            self.paint_integer_span(&runs, y as i64 + row as i64, x as i64, x as i64 + w as i64, is_index, value);
        }
        self.scanout.invalidate_all();
    }

    /// Board::clear_planes(): the X server's write to the id-plane destination.
    /// For each pixel of the rectangle, replace the plane bits the 0x1023 mask
    /// selects with the value the COLOR_INDEX word carried, in the id byte when
    /// the drawing mask selects buffer 0 and in the overlay pixel byte when it
    /// selects buffer 1, leaving the colour framebuffer untouched.
    pub fn clear_planes(&mut self, x: u32, y: u32, w: u32, h: u32, value: u8, mask: u8) {
        let buffers = self.draw_buffers();
        let id_test = self.id_compare_active();
        for row in 0..h {
            let py = y as i64 + row as i64;
            let runs = self.row_runs(py);
            for r in 0..runs.n as usize {
                let a = (x as i64).max(runs.start[r]);
                let e = (x as i64 + w as i64).min(runs.end[r]);

                for px in a..e {
                    if !id_test || self.id_compare_passes(px, py) {
                        self.aux_store(py as u64 * SGI_ODYSSEY_WIDTH as u64 + px as u64, buffers, value, mask);
                    }
                }
            }
        }
        // A pixel's drawable picks its colour at scanout.
        self.scanout.invalidate_all();
    }

    /// The screen-to-screen blit (Board::copy_rect()), the console-scroll primitive.
    /// Copies in whichever row/column direction keeps a read ahead of the matching
    /// write, so an overlapping source and destination in the same framebuffer --
    /// the case a scroll relies on -- reads the old contents before they are
    /// overwritten.
    ///
    /// Approximation: which buffer the board reads the source from is part of the
    /// uncleared read selection, so the copy runs within each buffer the drawing
    /// mask selects, source and destination in the same buffer.
    ///
    /// Approximation, inferred: a destination pixel passes Board::pixel_in_bounds()
    /// only, not the window-clip rectangles Board::pixel_allowed() also tests.
    pub fn copy_rect_buffer(&mut self, source: u32, dest: u32) {
        let sx = self.copy_src & 0xffff;
        let sy = (self.copy_src >> 16) & 0xffff;
        let w = self.copy_size & 0xffff;
        let h = (self.copy_size >> 16) & 0xffff;
        let dx = self.copy_dst & 0xffff;
        let dy = (self.copy_dst >> 16) & 0xffff;
        let w_clip = if dx < SGI_ODYSSEY_WIDTH { w.min(SGI_ODYSSEY_WIDTH - dx) } else { 0 };
        let h_clip = if dy < SGI_ODYSSEY_HEIGHT { h.min(SGI_ODYSSEY_HEIGHT - dy) } else { 0 };
        let reverse_rows = dy > sy;
        let reverse_cols = dx > sx;

        for row in 0..h_clip {
            let sr = if reverse_rows { h_clip - 1 - row } else { row };
            let src_row = if w_clip != 0 {
                self.buffer_row(source, sx as i64, sy as i64 + sr as i64, w_clip as i64)
            } else {
                None
            };
            let dst_row = if src_row.is_some() {
                self.buffer_row(dest, dx as i64, dy as i64 + sr as i64, w_clip as i64)
            } else {
                None
            };

            // A row wholly inside both planes, in the same encoding, is the byte
            // copy the pixel loop below makes. The loop runs its columns so that
            // a read stays ahead of the write it could meet, which is what a
            // memmove does for the one overlapping case, a row copied along
            // itself.
            if let (Some((src_plane, src)), Some((dst_plane, dst))) = (src_row, dst_row) {
                if src_plane.format == dst_plane.format && src_plane.packing == dst_plane.packing {
                    if (source != 0) == (dest != 0) {
                        self.buffer_mem_mut(dest).bytes.copy_within(src, dst.start);
                    } else {
                        let (from, to) = if source != 0 {
                            (&self.buffer1_mem, &mut self.mem)
                        } else {
                            (&self.mem, &mut self.buffer1_mem)
                        };
                        to.bytes[dst].copy_from_slice(&from.bytes[src]);
                    }
                    continue;
                }
            }
            for col in 0..w_clip {
                let sc = if reverse_cols { w_clip - 1 - col } else { col };
                // An out-of-plane source reads zero, as the guarded read did.
                let pixel = self.buffer_load(source, sx.wrapping_add(sc) as i64, sy.wrapping_add(sr) as i64);

                if self.pixel_in_bounds(dx.wrapping_add(sc) as i64, dy.wrapping_add(sr) as i64) {
                    self.buffer_store(dest, dx.wrapping_add(sc) as i64, dy.wrapping_add(sr) as i64, pixel);
                }
            }
        }
    }

    pub fn copy_rect(&mut self) {
        let source = (self.cmd_operand[2] >> ODSY_COPY_RECT_SOURCE_SHIFT) & 0xf;
        let dest = (self.cmd_operand[7] >> ODSY_COPY_RECT_DEST_SHIFT) & 0xf;
        let format_source = (self.cmd_operand[4] >> ODSY_COPY_RECT_FORMAT_SOURCE_SHIFT) & 0xf;
        let format_dest = (self.cmd_operand[4] >> ODSY_COPY_RECT_DEST_SHIFT) & 0xf;

        if self.cmd_operand[0] == ODSY_DMA_SENTINEL_READ {
            if !self.copy_rect_to_object() {
                self.cmd_unimplemented(ODSY_CMD_COPY_RECT_TRIGGER);
            }
            return;
        }

        if source != format_source || dest != format_dest {
            self.cmd_unimplemented(ODSY_CMD_COPY_RECT_TRIGGER);
            return;
        }

        // The two observed copy blocks select 0 then 4 in both their source and
        // destination fields. This is an experimental model policy: the selector
        // values choose model buffers 0 and 1, but do not identify physical
        // memory banks. Source and destination choices stay separate so a copy
        // never changes another colour buffer.
        if (source == 0 || source == 4) && (dest == 0 || dest == 4) {
            self.copy_rect_buffer(source >> 2, dest >> 2);
        } else if source == dest && (source == 2 || source == 3) {
            // Existing public and modelled templates retain draw-mask behaviour.
            let buffers = self.draw_buffers();

            for b in 0..ODSY_BUFFERS {
                if buffers & (1u32 << b) != 0 {
                    self.copy_rect_buffer(b, b);
                }
            }
        } else {
            self.cmd_unimplemented(ODSY_CMD_COPY_RECT_TRIGGER);
            return;
        }
        self.scanout.invalidate_all();
    }

    /// Pattern mode is on when the 0x0404 mode's bits 3,4 are set and a shape is
    /// selected. A quad drawn in this state paints the current colour through the
    /// pattern instead of as a solid rectangle.
    pub fn pattern_enabled(&mut self) -> bool {
        if (self.pattern_mode & ODSY_PATTERN_MODE_ENABLE) != ODSY_PATTERN_MODE_ENABLE {
            return false;
        }
        matches!(
            self.pattern_shape & !ODSY_PATTERN_SHAPE_OPAQUE,
            ODSY_PATTERN_SHAPE_64X16 | ODSY_PATTERN_SHAPE_32X32 | ODSY_PATTERN_SHAPE_16X64
        )
    }

    /// The pattern bit at quad-relative column u, row v, for the current shape.
    /// Row 0 column 0 is the quad's top-left vertex and bit 31 of a row's first
    /// word is its leftmost pixel. A coordinate past the shape's own extent is
    /// treated as clear: the X server never draws a quad larger than its pattern.
    pub fn pattern_bit(&mut self, u: u32, v: u32) -> bool {

        let (word, bit) = match self.pattern_shape & !ODSY_PATTERN_SHAPE_OPAQUE {
            ODSY_PATTERN_SHAPE_64X16 => {
                if u >= 64 || v >= 16 {
                    return false;
                }
                (self.pattern_buffer[(2 * v + (u >> 5)) as usize], 31 - (u & 31))
            }
            ODSY_PATTERN_SHAPE_32X32 => {
                if u >= 32 || v >= 32 {
                    return false;
                }
                (self.pattern_buffer[v as usize], 31 - u)
            }
            ODSY_PATTERN_SHAPE_16X64 => {
                if u >= 16 || v >= 64 {
                    return false;
                }
                (self.pattern_buffer[(v >> 1) as usize], 31 - 16 * (v & 1) - u)
            }
            _ => return false,
        };
        (word >> bit) & 1 != 0
    }

    /// The opaque pattern's background: the 0x0c21 constant colour, an index from
    /// operand 1 bits 23:12, or an RGB pixel from the 12-bit red, green and blue
    /// fields narrowed to 8 bits.
    pub fn pattern_background(&mut self, is_index: bool) -> u32 {
        let r = (self.const_colour[0] >> 12) & 0xfff;
        let g = self.const_colour[0] & 0xfff;
        let b = (self.const_colour[1] >> 12) & 0xfff;

        if is_index {
            return r;
        }
        pack_pixel(r >> 4, g >> 4, b >> 4)
    }

    /// A pattern quad: paint the current colour at each pixel whose pattern bit is
    /// set, and the background at each clear one when the pattern is opaque,
    /// leaving the rest as it was otherwise. Only the rows and columns the quad
    /// covers are read, so the stale part of the persistent pattern buffer beyond
    /// the tile never contributes.
    #[allow(clippy::too_many_arguments, reason = "a fill rectangle, its pattern origin and its colour")]
    pub fn pattern_fill(&mut self, x: u32, y: u32, w: u32, h: u32, pat_u0: u32, pat_v0: u32, is_index: bool, value: u32) {
        let opaque = self.pattern_shape & ODSY_PATTERN_SHAPE_OPAQUE != 0;
        let background = self.pattern_background(is_index);
        // Each row as spans of equal pattern bits, painted a span at a time.
        for v in 0..h {
            let py = y as i64 + v as i64;

            let runs = self.row_runs(py);
            if runs.n == 0 {
                continue;
            }
            let mut u = 0u32;
            while u < w {
                let bit = self.pattern_bit(pat_u0.wrapping_add(u), pat_v0.wrapping_add(v));
                let first = u;

                loop {
                    u += 1;
                    if !(u < w && self.pattern_bit(pat_u0.wrapping_add(u), pat_v0.wrapping_add(v)) == bit) {
                        break;
                    }
                }
                if bit || opaque {
                    self.paint_integer_span(
                        &runs,
                        py,
                        x as i64 + first as i64,
                        x as i64 + u as i64,
                        is_index,
                        if bit { value } else { background },
                    );
                }
            }
        }
        self.scanout.invalidate_all();
    }
}

/// A raster-box coordinate arrives as an IEEE-754 single the X server built
/// with an integer-to-float convert, so the 32-bit operand is a float bit
/// pattern, not an integer. Truncate toward zero; a NaN or negative clamps to
/// the screen origin.
pub fn float_to_u32(bits: u32) -> u32 {
    let f = f32::from_bits(bits);

    if !(f > 0.0f32) {
        return 0;
    }
    f as u32
}

impl Board {

    /// The object an image transfer writes, or -1 for the screen: the last 0x1017
    /// select, when it named a base other than the colour buffer's.
    pub fn image_object(&mut self) -> i32 {
        if self.dest_base == 0 || self.dest_base == self.dma_colour_base || self.dest_is_id_planes() {
            return -1;
        }
        self.object_find(self.dest_base)
    }

    /// Lay one image pixel, as 0xRRGGBBAA, at the raster cursor in the selected
    /// buffer (Board::image_object()) and advance it. Each axis is a directed
    /// half-open interval: increasing endpoints visit [first, last), while
    /// decreasing endpoints visit [first - 1, last]. Payload advances along X,
    /// then along Y.
    pub fn image_put(&mut self, rgba: u32, is_index: bool, preserve_alpha: bool) {
        let x_decreasing = self.image_x2 < self.image_x1;
        let y_decreasing = self.image_y2 < self.image_y1;
        let y_finished = if y_decreasing {
            self.image_cur_y == u32::MAX || self.image_cur_y < self.image_y2
        } else {
            self.image_cur_y >= self.image_y2
        };

        if self.image_x2 == self.image_x1 || self.image_y2 == self.image_y1 || y_finished {
            return;
        }
        let obj = self.image_object();
        if obj >= 0 {
            // An off-screen object: box coordinates, no window or id tests.
            self.object_store(obj, self.image_cur_x, self.image_cur_y, rgba);
        } else if self.dest_is_id_planes() {
            // Menu cascade arrows: index images into the overlay.
            if is_index {
                self.paint_aux(
                    self.image_cur_x as i64 + self.win_origin[0] as i64,
                    self.image_cur_y as i64 + self.win_origin[1] as i64,
                    rgba,
                    SGI_LOGIC_COPY as u8,
                );
            } else if !self.aux_image_reported {
                self.aux_image_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: an RGB image to the id-plane destination is not modeled; dropped");
            }
        } else {
            // Screen pixels are window-relative; object pixels are not.
            self.paint_pixel(
                self.image_cur_x as i64 + self.win_origin[0] as i64,
                self.image_cur_y as i64 + self.win_origin[1] as i64,
                if is_index || preserve_alpha { rgba } else { (rgba & !0xffu32) | 0xff },
            );
        }
        if (x_decreasing && self.image_cur_x == self.image_x2) || (!x_decreasing && self.image_cur_x == self.image_x2.wrapping_sub(1)) {
            self.image_cur_x = if x_decreasing { self.image_x1.wrapping_sub(1) } else { self.image_x1 };
            if y_decreasing {
                self.image_cur_y = if self.image_cur_y == self.image_y2 { u32::MAX } else { self.image_cur_y.wrapping_sub(1) };
            } else {
                self.image_cur_y = self.image_cur_y.wrapping_add(1);
            }
        } else if x_decreasing {
            self.image_cur_x = self.image_cur_x.wrapping_sub(1);
        } else {
            self.image_cur_x = self.image_cur_x.wrapping_add(1);
        }
        self.scanout.invalidate_all();
    }

    /// One byte-stream texel of the open image at the raster cursor, bounded by
    /// the header's group length in pixels.
    pub fn image_texel(&mut self, rgba: u32, is_index: bool) {
        if self.image_group_len != 0 && self.image_group_painted >= self.image_group_len {
            return;
        }
        self.image_put(rgba, is_index, false);
        self.image_group_painted += 1;
    }
}

pub fn unorm16_to_8(v: u32) -> u32 {
    (v.wrapping_mul(255).wrapping_add(65535 / 2)) / 65535
}

/// Components of a 12-bit texture source code, or 0 for any other code.
pub fn texel12_components(code: u32) -> u32 {
    match code {
        ODSY_IMAGE_CODE_LUM12 => 1,
        ODSY_IMAGE_CODE_LUM_ALPHA12 => 2,
        ODSY_IMAGE_CODE_RGB12 => 3,
        ODSY_IMAGE_CODE_RGBA12 | ODSY_IMAGE_CODE_ABGR12 => 4,
        _ => 0,
    }
}

/// A 12-bit texture source pixel as 0xRRGGBBAA: its components, each a byte or
/// (wide) a big-endian 16-bit word, in the order its code names.
pub fn decode_texel12(code: u32, b: &[u8], wide: bool) -> u32 {
    let mut c = [0u32; 4];
    let n = texel12_components(code) as usize;

    for i in 0..n {
        c[i] = if wide { unorm16_to_8(u16::from_be_bytes(b[2 * i..][..2].try_into().unwrap()) as u32) } else { b[i] as u32 };
    }
    match code {
        ODSY_IMAGE_CODE_LUM12 => pack_pixel(c[0], c[0], c[0]),
        ODSY_IMAGE_CODE_LUM_ALPHA12 => (pack_pixel(c[0], c[0], c[0]) & !0xffu32) | c[1],
        ODSY_IMAGE_CODE_RGB12 => pack_pixel(c[0], c[1], c[2]),
        ODSY_IMAGE_CODE_ABGR12 => (pack_pixel(c[3], c[2], c[1]) & !0xffu32) | c[0],
        _ => (pack_pixel(c[0], c[1], c[2]) & !0xffu32) | c[3],
    }
}

impl Board {
    /// Decode one pixel of a byte-stream source format whose bytes are b[0 ..]:
    /// its 0xRRGGBBAA word, or for 0x41 and 0x61 the raw index. False for a code
    /// this model does not know.
    pub fn decode_pixel(&mut self, code: u32, b: &[u8], rgba: &mut u32) -> bool {
        match code {
            ODSY_IMAGE_CODE_ABGR => {
                *rgba = pack_pixel(b[3] as u32, b[2] as u32, b[1] as u32);
                true
            }
            ODSY_IMAGE_CODE_FALLBACK_XRGB => {
                *rgba = pack_pixel(b[1] as u32, b[2] as u32, b[3] as u32);
                true
            }
            ODSY_IMAGE_CODE_RGBA | ODSY_IMAGE_CODE_RGBA12 => {
                *rgba = u32::from_be_bytes(b[..4].try_into().unwrap());
                true
            }
            ODSY_IMAGE_CODE_RGB => {
                *rgba = pack_pixel(b[0] as u32, b[1] as u32, b[2] as u32);
                true
            }
            ODSY_IMAGE_CODE_LUM => {
                *rgba = pack_pixel(b[0] as u32, b[0] as u32, b[0] as u32);
                true
            }
            ODSY_IMAGE_CODE_LUM_ALPHA => {
                *rgba = (pack_pixel(b[0] as u32, b[0] as u32, b[0] as u32) & !0xffu32) | b[1] as u32;
                true
            }
            ODSY_IMAGE_CODE_LUM12 | ODSY_IMAGE_CODE_LUM_ALPHA12 => {
                // A burst's 16-bit components (pixel_bytes()).
                *rgba = decode_texel12(code, b, true);
                true
            }
            ODSY_IMAGE_CODE_CI => {
                // The raw index: scanout resolves it through the pixel's drawable.
                *rgba = b[0] as u32;
                true
            }
            ODSY_IMAGE_CODE_CI16 => {
                *rgba = u16::from_be_bytes(b[..2].try_into().unwrap()) as u32 & ODSY_INDEX_MASK;
                true
            }
            _ => false,
        }
    }
}

/// Whether a source format carries colour indices rather than colours.
pub fn code_is_index(code: u32) -> bool {
    code == ODSY_IMAGE_CODE_CI || code == ODSY_IMAGE_CODE_CI16
}

/// Bytes a pixel of a source format takes, 0 for an unknown code.
pub fn pixel_bytes(code: u32) -> u32 {
    match code {
        ODSY_IMAGE_CODE_ABGR | ODSY_IMAGE_CODE_FALLBACK_XRGB | ODSY_IMAGE_CODE_RGBA | ODSY_IMAGE_CODE_RGBA12 => 4,
        ODSY_IMAGE_CODE_RGB => 3,
        ODSY_IMAGE_CODE_LUM | ODSY_IMAGE_CODE_CI => 1,
        ODSY_IMAGE_CODE_LUM_ALPHA | ODSY_IMAGE_CODE_LUM12 | ODSY_IMAGE_CODE_CI16 => 2,
        ODSY_IMAGE_CODE_LUM_ALPHA12 => 4,
        _ => 0,
    }
}

pub fn image_pairing_supported(code: u32, fmt: u32) -> bool {
    if code == ODSY_IMAGE_CODE_RGBA12 {
        return fmt == ODSY_IMAGE_FMT_RGB || fmt == ODSY_IMAGE_FMT_RGBA16;
    }
    if code == ODSY_IMAGE_CODE_RGB12 || code == ODSY_IMAGE_CODE_ABGR12 {
        return fmt == ODSY_IMAGE_FMT_RGBA16;
    }
    if code == ODSY_IMAGE_CODE_LUM_ALPHA {
        return fmt == ODSY_IMAGE_FMT_CI;
    }
    fmt != ODSY_IMAGE_FMT_RGBA16
}

impl Board {
    /// One data word of a pixel burst, laid out by the transfer header's format
    /// code: its bytes join a stream that yields a pixel every
    /// pixel_bytes() bytes, so a three-byte pixel may span two words. A
    /// bounded group's trailing padding (past image_group_len) is dropped with any
    /// partial pixel, so the raster position stays exact. A header with a code
    /// this model does not know falls back on the burst tag's format, 0x11 one
    /// 0x00RRGGBB pixel a word and 0x10 four indices, and is reported once.
    pub fn image_burst_word(&mut self, word: u32) {
        let mut code = self.image_code;
        let mut n = pixel_bytes(code);
        let mut b = [0u8; 4];

        if !image_pairing_supported(code, self.image_fmt) {
            return;
        }
        if self.image_fmt == ODSY_IMAGE_FMT_RGBA16 {
            // A stream of 16-bit components, high half first, a pixel every
            // texel12_components() halves: two words for 0x6b and 0x6c, a
            // word and a half for 0x69. At most two halves wait between words;
            // image_nbytes counts them, image_bytes holds them.
            let comps = texel12_components(code) as usize;
            let mut halves = [0u16; 4];
            let mut held = self.image_nbytes.min(2) as usize;

            if self.image_group_len != 0 && self.image_group_painted >= self.image_group_len {
                return;
            }
            for (i, half) in halves.iter_mut().enumerate().take(held) {
                *half = (self.image_bytes >> (16 * (held - 1 - i))) as u16;
            }
            for h in 0..2 {
                let mut px = [0u8; 8];

                halves[held] = if h != 0 { word as u16 } else { (word >> 16) as u16 };
                held += 1;
                if held < comps {
                    continue;
                }
                for (pair, half) in px.as_chunks_mut::<2>().0.iter_mut().zip(&halves[..comps]) {
                    pair.copy_from_slice(&half.to_be_bytes());
                }
                self.image_put(decode_texel12(code, &px, true), false, true);
                held = 0;
            }
            self.image_bytes = if held > 1 {
                ((halves[0] as u32) << 16) | halves[1] as u32
            } else if held != 0 {
                halves[0] as u32
            } else {
                0
            };
            self.image_nbytes = held as u32;
            self.image_group_painted += 1;
            if self.image_group_len != 0 && self.image_group_painted >= self.image_group_len && self.image_nbytes != 0 {
                self.image_bytes = 0;
                self.image_nbytes = 0;
            }
            return;
        }

        if n == 0 {
            if !self.image_code_reported {
                self.image_code_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: image source format {:#x} is not modeled; laid out by the burst tag", code);
            }
            code = if self.image_fmt == ODSY_IMAGE_FMT_RGB { ODSY_IMAGE_CODE_FALLBACK_XRGB } else { ODSY_IMAGE_CODE_CI };
            n = pixel_bytes(code);
        }
        b[..4].copy_from_slice(&word.to_be_bytes());
        for &byte in &b[..4] {
            let mut px = [0u8; 4];
            let mut rgba = 0u32;

            self.image_bytes = (self.image_bytes << 8) | byte as u32;
            self.image_nbytes += 1;
            if self.image_nbytes < n {
                continue;
            }
            self.image_nbytes = 0;
            // The pixel's bytes, first byte highest, at the start of px[].
            px[..4].copy_from_slice(&(self.image_bytes << (8 * (4 - n))).to_be_bytes());
            self.decode_pixel(code, &px, &mut rgba);
            self.image_texel(rgba, code_is_index(code));
        }
    }
}

/// True, and set *fmt, when reg is a pixel-burst register-select word.
pub fn image_burst_fmt(reg: u32, fmt: &mut u32) -> bool {
    let f = reg & ODSY_IMAGE_BURST_FMT_MASK;

    if (reg & ODSY_IMAGE_BURST_BASE) == 0 || (reg & ODSY_IMAGE_BURST_RESERVED_MASK) != 0 {
        return false;
    }
    match f {
        ODSY_IMAGE_FMT_CI | ODSY_IMAGE_FMT_RGB | ODSY_IMAGE_FMT_RGBA16 => {
            *fmt = f;
            true
        }
        // Other documented pixel-burst formats remain unsupported.
        _ => false,
    }
}

/// The family side of the shared OpenGL context and software rasterizer: a
/// thin decoder. The rasterizer rule set is chosen, pending capture: the
/// Octane RE4 set is the only one RasterRules::rules_supported() implements,
/// and a flat single triangle in one colour exercises none of the parameters
/// that distinguish heads. The board's own subpixel grid and
/// interpolated-colour depth are unmeasured. The depth word is the board's
/// 24-bit float (depth_word.rs). (A function, not a static: the struct is not const
/// constructible.)
///
/// One-pixel lines take the centre-sampled rule (SGI_LINE_CENTRE_SAMPLED), as
/// the QEMU sgi-origami branch's board model does (ab9b3f98f25f,
/// c5417243927d). The branch's record is the authority for where that rule
/// is established and where it is a model choice: the x-major tie and the
/// end rule are model choices.
pub fn gl_rules() -> RasterRules {
    RasterRules {
        subpixel_bits: 8,
        color_bits: 8,
        depth_bits: 24,
        stencil_bits: 8,
        rounding: SGI_ROUND_HALF_AWAY as u8,
        tie_rule: SGI_TIE_TOP_LEFT_BOTTOM_ORIGIN as u8,
        arith: SGI_ARITH_DOUBLE as u8,
        line_rule: SGI_LINE_CENTRE_SAMPLED as u8,
        alpha_test_bits: 12,
        smooth_primitives: true,
        // The board's depth word; Board::gl_load_combined().
        depth_float: true,
        ..Default::default()
    }
}
