//! The board's 2D drawing: colour-buffer access, clipping and id-plane tests,
//! integer pixel paints, rectangle fill and copy, and pixel bursts.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.


use super::depth_word::word_for_fraction;
use super::pipe::backend::{widen8, BackendState, DestMem};
use super::view::{Kind, MAX_DIM};
use super::field::{Codec, Layout};
use super::vram::Format;
use super::bits::{be16, be32};
use crate::devlog::LogModule;
use super::board::{Board, ODSY_CLIP_RECTS};
use super::consts::{
    clip, dma, image, mode, pattern, write,
    ODSY_BUFFERS, ODSY_BURST_STENCIL_SHIFT, ODSY_DEPTH24_MAX,
    ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1, ODSY_FENCE_WORD_MASK, ODSY_ID_COMPARE_ENABLE,
    ODSY_ID_COMPARE_MASK_SHIFT, ODSY_INDEX_MASK, ODSY_PLANE_MASK_SHIFT,
};
use super::clip::ClipRect;
use super::logic::{logic_op, LOGIC_COPY};
use super::pixel_dma::write_guest;
use super::pixel_write::{write_form, WriteForm};
/// The clip-rectangle runs of one row (Board::row_runs).
#[derive(Clone, Copy)]
pub(super) struct RowRuns {
    pub n: u32,
    pub start: [i64; ODSY_CLIP_RECTS as usize],
    pub end: [i64; ODSY_CLIP_RECTS as usize],
}

impl Default for RowRuns {
    fn default() -> Self {
        RowRuns { n: 0, start: [0; ODSY_CLIP_RECTS as usize], end: [0; ODSY_CLIP_RECTS as usize] }
    }
}

/// 2D fills carry no alpha channel in this grammar; the framebuffer's fourth
/// byte is modelled fully opaque (the colour plane's red-high R,G,B,A packing,
/// resolved by resolve_row() below).
pub(super) fn pack_pixel(r: u32, g: u32, b: u32) -> u32 {
    (r << 24) | (g << 16) | (b << 8) | 0xff
}

impl Board {
    /// The pixel mode's destination in board memory (view::DestViews), for
    /// the front end's own stores: it waits for the raster unit's queued
    /// jobs first.
    pub fn dest_mem(&self) -> DestMem {
        // Everything the drawing wrote, back: the general path; the hot
        // ones use planes_in() or store through the unit.
        self.gpu_sync();
        DestMem::of(self.vram_ptr(), &self.dest_views())
    }

    /// The destination for a front-end path that reads or writes the
    /// pixels of one box (inclusive): on the GPU unit, only the blocks the
    /// drawing touched under that box come back, and nothing waits when
    /// none did; with the CPU unit, `dest_mem`.
    pub fn planes_in(&self, x0: i64, y0: i64, x1: i64, y1: i64) -> DestMem {
        if !self.gpu_unit() {
            return self.dest_mem();
        }
        let planes = DestMem::of(self.vram_ptr(), &self.dest_views());
        self.gpu_read_back(&planes, x0, y0, x1, y1);
        planes
    }

    /// Bring the mirror current under a box of every plane of `planes`
    /// (GpuRaster::read_back_view).
    pub fn gpu_read_back(&self, planes: &DestMem, x0: i64, y0: i64, x1: i64, y1: i64) {
        let clamp = |v: i64| v.clamp(-1, MAX_DIM as i64) as i32;
        let (x0, y0, x1, y1) = (clamp(x0), clamp(y0), clamp(x1), clamp(y1));
        self.raster.run_now(|ctx| {
            if let Some(g) = &mut ctx.gpu {
                for rows in [&planes.colour[0], &planes.colour[1], &planes.depth, &planes.stencil, &planes.did, &planes.overlay] {
                    if let Err(e) = g.read_back_view(&rows.v, x0, y0, x1, y1) {
                        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: gpu readback: {e}");
                    }
                }
            }
        });
    }

    /// Field b of the destination: 0 the A (front) field, 1 the B (back).
    pub fn buffer_load(&self, b: u32, x: i64, y: i64) -> u32 {
        self.planes_in(x, y, x, y).colour[(b & 1) as usize].load_word(x as i32, y as i32).unwrap_or(0)
    }

    /// Store field b of the destination. A byte field (the window-id
    /// buffer's id and overlay bytes) is stored under the plane mask.
    pub fn buffer_store(&mut self, b: u32, x: i64, y: i64, word: u32) {
        if self.gpu_blit_buffers(1 << (b & 1), x, y, word, None, u32::MAX, 0) {
            return;
        }
        let rows = self.planes_in(x, y, x, y).colour[(b & 1) as usize];
        if rows.v.kind == Kind::Byte {
            let mask = (self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) as u8;
            if let Some(old) = rows.load1(x as i32, y as i32) {
                rows.store1(x as i32, y as i32, (old & !mask) | (word as u8 & mask));
            }
            return;
        }
        rows.store_word(x as i32, y as i32, word);
    }

    /// Whether the pixel mode lays drawing out as the window-id buffer's
    /// two bytes, the id and overlay bytes (field.rs, Layout::of).
    pub fn dest_is_id_planes(&self) -> bool {
        Layout::of(Format::decode(self.pixel_mode[2])).front.codec == Codec::Byte
    }

    /// The buffers a colour write lands in, bit b for buffer b, from the last
    /// 0x0424 drawing mask. A right-eye bit folds into its buffer: stereo is not
    /// modelled.
    pub fn draw_buffers(&self) -> u32 {
        (if self.draw_mask & ODSY_DRAW_MASK_BUFFER0 != 0 { 1 } else { 0 })
            | (if self.draw_mask & ODSY_DRAW_MASK_BUFFER1 != 0 { 2 } else { 0 })
    }

    /// The one completion fence, shared by the 0x0a5a host-write command and
    /// the DMA engine's publish(): write the cookie's 16-bit token,
    /// big-endian, to the guest fence slot at the host-write base plus twice the
    /// slot, and pulse the softsync interrupt when the cookie asks.
    ///
    /// The raster unit writes it, behind the raster jobs queued before it, so
    /// the token lands only once the drawing before it is in the planes.
    pub fn fence(&mut self, cookie: u64) {
        let word = (cookie & ODSY_FENCE_WORD_MASK as u64) as u32;
        let slot = word & dma::FENCE_SLOT_MASK;
        let token = (word >> dma::FENCE_TOKEN_SHIFT) & 0xffff;
        let base = (((self.widget.host_write_addr_upper & 0xffff) as u64) << 32) | self.widget.host_write_addr_lower as u64;
        let (guest, front) = (self.guest.clone(), self.front.clone());

        self.raster.submit_completion(move || {
            write_guest(&guest, base.wrapping_add(2 * slot as u64), &(token as u16).to_be_bytes());
            if word & dma::FENCE_INTERRUPT != 0 {
                front.lock().deliver_softsync();
            }
        });
    }
}

impl Board {
    /// Rebuild the kept (non-empty) clip rectangles from the raw 0x204b words.
    pub fn clip_update(&mut self) {
        self.win.clip_kept = 0;
        for i in 0..ODSY_CLIP_RECTS as usize {
            let lo = self.win.clip_words[2 * i];
            let hi = self.win.clip_words[2 * i + 1];
            let r = ClipRect {
                x0: ((lo >> clip::X_SHIFT) & clip::FIELD_MASK) as i32,
                y0: (lo & clip::FIELD_MASK) as i32,
                x1: ((hi >> clip::X_SHIFT) & clip::FIELD_MASK) as i32,
                y1: (hi & clip::FIELD_MASK) as i32,
            };

            if r.x0 <= r.x1 && r.y0 <= r.y1 {
                self.win.clip_rect[self.win.clip_kept as usize] = r;
                self.win.clip_kept += 1;
            }
        }
    }

    /// Whether clip_words is the off-screen form (cmd::WINDOW_RESTORE).
    pub fn clip_is_offscreen(&self) -> bool {
        for (i, &word) in self.win.clip_words.iter().enumerate().take((2 * ODSY_CLIP_RECTS) as usize) {
            let want = if i == 1 { clip::OFFSCREEN_HI } else { 0 };

            if (word & clip::WORD_MASK) != want {
                return false;
            }
        }
        true
    }

    /// The destination's bound: its width in tiles, and the largest drawable.
    pub fn pixel_in_bounds(&self, x: i64, y: i64) -> bool {
        x >= 0 && y >= 0 && x < self.dest_views().colour[0].width as i64 && y < MAX_DIM as i64
    }

    pub fn id_compare_active(&self) -> bool {
        self.win.id_compare & ODSY_ID_COMPARE_ENABLE != 0
    }

    /// The 0x840 ownership test: the window-id byte at (x, y) under the
    /// test's mask.
    pub fn id_compare_passes(&self, x: i64, y: i64) -> bool {
        let c = self.win.id_compare;
        let mask = (c >> ODSY_ID_COMPARE_MASK_SHIFT) as u8;
        // The id bytes are the X server's, written by the front end alone,
        // never by the raster unit: no wait before reading them.
        c & ODSY_ID_COMPARE_ENABLE == 0 || DestMem::of(self.vram_ptr(), &self.dest_views()).did.load1(x as i32, y as i32).is_some_and(|id| (id ^ c as u8) & mask == 0)
    }

    /// The one pixel test every write but a copy destination passes: the plane
    /// bound and the 0x840 id test, then inside at least one kept clip rectangle
    /// -- with none kept nothing is drawn.
    pub fn pixel_allowed(&self, x: i64, y: i64) -> bool {
        if !self.pixel_in_bounds(x, y) || !self.id_compare_passes(x, y) {
            return false;
        }
        for r in &self.win.clip_rect[..self.win.clip_kept as usize] {
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

        let width = self.dest_views().colour[0].width as i64;
        if y < 0 || y >= MAX_DIM as i64 {
            return runs;
        }
        for &r in &self.win.clip_rect[..self.win.clip_kept as usize] {
            let a = (r.x0 as i64).max(0);
            let b = (r.x1 as i64 + 1).min(width);
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

    /// A colour write, into each buffer the drawing mask selects.
    ///
    /// Marks the pixel's display line to be resolved again.
    pub fn paint_pixel(&mut self, x: i64, y: i64, pixel: u32) {
        let buffers = self.draw_buffers();

        if !self.pixel_allowed(x, y) {
            return;
        }
        if self.gpu_paint(buffers, x, y, pixel) {
            return;
        }
        for b in 0..ODSY_BUFFERS {
            if buffers & (1u32 << b) != 0 {
                self.buffer_store(b, x, y, pixel);
            }
        }
    }

    /// `paint_pixel`'s stores on the GPU unit, in order behind the drawing
    /// and with no wait: the pixel joins a blit record. False without the
    /// unit, or for planes its kernels cannot address.
    fn gpu_paint(&mut self, buffers: u32, x: i64, y: i64, pixel: u32) -> bool {
        self.gpu_blit_buffers(buffers, x, y, pixel, None, u32::MAX, 0)
    }

    /// A store into each colour buffer of `buffers` on the GPU unit, with
    /// `op` combining the pixel with the old word first and its result
    /// and-ed and or-ed (Board::paint_integer, paint_aux); a byte field is
    /// stored under the plane mask (Board::buffer_store). False without
    /// the unit, or for planes its kernels cannot address.
    fn gpu_blit_buffers(&mut self, buffers: u32, x: i64, y: i64, pixel: u32, op: Option<u8>, and: u32, or: u32) -> bool {
        if !self.gpu_unit() {
            return false;
        }
        let planes = DestMem::of(self.vram_ptr(), &self.dest_views());
        let Some(slot) = self.gpu_dest_slot(&planes) else {
            return false;
        };
        let plane_mask = (self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) as u8;
        self.raster.run_now(|ctx| {
            let g = ctx.gpu.as_mut().expect("the unit");
            for b in 0..ODSY_BUFFERS {
                if buffers & (1u32 << b) == 0 {
                    continue;
                }
                let mask = if planes.colour[b as usize].v.kind == Kind::Byte { plane_mask as u32 } else { u32::MAX };
                let key = super::gpu_unit::BlitKey { mask, op: op.map_or(0, |o| 0x100 | o as u32), and, or, ..super::gpu_unit::BlitKey::store(b, mask) };
                g.blit_pixel(&slot, &key, x as i32, y as i32, pixel);
                mirror_byte_store(&planes.colour[b as usize], x, y, pixel, &key);
            }
        });
        true
    }

    /// A store into one view of the destination (`backend.wgsl` V_*) on the
    /// GPU unit, the bits of `mask`. False without the unit.
    pub fn gpu_blit_view(&mut self, view: u32, x: i64, y: i64, word: u32, mask: u32) -> bool {
        if !self.gpu_unit() {
            return false;
        }
        let planes = DestMem::of(self.vram_ptr(), &self.dest_views());
        let Some(slot) = self.gpu_dest_slot(&planes) else {
            return false;
        };
        let key = super::gpu_unit::BlitKey::store(view, mask);
        self.raster.run_now(|ctx| ctx.gpu.as_mut().expect("the unit").blit_pixel(&slot, &key, x as i32, y as i32, word));
        let rows = [planes.colour[0], planes.colour[1], planes.depth, planes.stencil, planes.did, planes.overlay];
        if let Some(r) = rows.get(view as usize) {
            mirror_byte_store(r, x, y, word, &key);
        }
        true
    }

    /// The slot of the destination and window for the front end's own
    /// stores, when the kernels can address the planes.
    /// Built once per destination, clip list and id test: the paths that
    /// store a pixel at a time ask for it at every pixel.
    pub fn gpu_dest_slot(&self, planes: &DestMem) -> Option<super::gpu_unit::Slot> {
        let mut clip = [0i32; 4 * ODSY_CLIP_RECTS as usize];
        for (i, r) in self.win.clip_rect.iter().enumerate() {
            clip[4 * i..4 * i + 4].copy_from_slice(&[r.x0, r.y0, r.x1, r.y1]);
        }
        let views = [&planes.colour[0], &planes.colour[1], &planes.depth, &planes.stencil, &planes.did, &planes.overlay].map(|r| r.v);
        let key = SlotKey { views, clip, kept: self.win.clip_kept, id: self.win.id_compare };
        if let Some((k, s)) = self.slot_cache.get() {
            if k == key {
                return s;
            }
        }
        let words = super::pipe::state::RasterWords { clip_rect: self.win.clip_rect, clip_kept: self.win.clip_kept, id_compare: self.win.id_compare, ..Default::default() };
        let s = super::gpu_unit::Slot::clear(planes, &words).ok();
        self.slot_cache.set(Some((key, s)));
        s
    }
}

/// What Board::gpu_dest_slot() builds its slot from.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlotKey {
    views: [super::view::View; 6],
    clip: [i32; 4 * ODSY_CLIP_RECTS as usize],
    kept: u32,
    id: u32,
}

/// How a 2D fill combines its word with each pixel it passes
/// (Board::fill_box): the word as is, or through `op` with the old word,
/// its result and-ed and or-ed (Board::paint_integer, paint_aux). A byte
/// field is stored under `byte_mask`, or the plane mask without one.
#[derive(Clone, Copy)]
pub(super) struct FillOp {
    pub word: u32,
    pub op: Option<u8>,
    pub and: u32,
    pub or: u32,
    pub byte_mask: Option<u8>,
}

impl FillOp {
    fn value(&self, old: u32) -> u32 {
        match self.op {
            Some(op) => (logic_op(op, self.word, old) & self.and) | self.or,
            None => self.word,
        }
    }

    /// The pixel's new contents in `rows`, from its old ones.
    fn store(&self, rows: &super::view::Rows, x: i32, y: i32, plane_mask: u8) {
        if rows.v.kind == Kind::Byte {
            if let Some(old) = rows.load1(x, y) {
                let m = self.byte_mask.unwrap_or(plane_mask);
                rows.store1(x, y, (old & !m) | (self.value(old as u32) as u8 & m));
            }
            return;
        }
        let val = match self.op {
            Some(_) => self.value(rows.load_word(x, y).unwrap_or(0)),
            None => self.word,
        };
        rows.store_word(x, y, val);
    }
}

/// The window-id buffer's bytes (a byte field: the id and overlay bytes)
/// are read by the front end itself, with no wait for the raster unit
/// (`Board::id_compare_passes`, `drawable_codec`, `main_id`). On the GPU unit
/// the unit stores them, so the front end makes each such store in the
/// mirror too, as the blit kernel makes it (vpro.wgsl `blit`), through an
/// untracked pointer: the mirror's id bytes stay current, nothing is
/// uploaded twice, and a later readback of the block brings back the same
/// bytes. Other fields are left to the unit and its readbacks.
fn mirror_byte_store(rows: &super::view::Rows, x: i64, y: i64, word: u32, key: &super::gpu_unit::BlitKey) {
    if rows.v.kind != Kind::Byte || key.window {
        return;
    }
    let rows = super::view::Rows::new(rows.mem().untracked(), rows.v);
    let (x, y) = (x as i32, y as i32);
    let Some(old) = rows.load1(x, y) else { return };
    let old = old as u32;
    let new = if key.op == 0 && key.mask == u32::MAX {
        word
    } else {
        let val = if key.op != 0 { (logic_op((key.op & 0xf) as u8, word, old) & key.and) | key.or } else { word };
        (old & !key.mask) | (val & key.mask)
    };
    rows.store1(x, y, new as u8);
}

/// The plane word a fill stores: the packed RGB pixel of an RGB fill, or the
/// raw index of a colour-index fill. No colormap is applied here: scanout
/// resolves each pixel through its own drawable.
pub(super) fn fill_pixel(is_index: bool, value: u32) -> u32 {
    if is_index { value & ODSY_INDEX_MASK } else { value }
}

impl Board {
    /// One pixel the X server draws into the window-id buffer: the index's
    /// low byte combined through op with the byte it lands on, in each
    /// field the drawing mask selects (A the id byte, B the overlay byte),
    /// under the plane mask (Board::buffer_store()).
    pub fn paint_aux(&mut self, x: i64, y: i64, index: u32, op: u8) {
        if !self.pixel_allowed(x, y) {
            return;
        }
        let buffers = self.draw_buffers();
        if self.gpu_blit_buffers(buffers, x, y, index, (op as u32 != LOGIC_COPY).then_some(op), 0xff, 0) {
            return;
        }
        for b in 0..ODSY_BUFFERS {
            if buffers & (1u32 << b) != 0 {
                let byte = if op as u32 == LOGIC_COPY { index } else { logic_op(op, index, self.buffer_load(b, x, y)) };
                self.buffer_store(b, x, y, byte & 0xff);
            }
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
        let enable = if is_index { mode::INDEX_LOGIC_OP } else { mode::COLOR_LOGIC_OP };
        let op = (self.logic_op_word & super::consts::ODSY_LOGIC_OP_MASK) as u8;
        let buffers = self.draw_buffers();

        if self.dest_is_id_planes() {
            let use_op = (self.stream.pattern_mode & enable) != 0 && op as u32 != LOGIC_COPY;

            self.paint_aux(x, y, src, if use_op { op } else { LOGIC_COPY as u8 });
            return;
        }
        if (self.stream.pattern_mode & enable) == 0 || op as u32 == LOGIC_COPY {
            self.paint_pixel(x, y, src);
            return;
        }
        if !self.pixel_allowed(x, y) {
            return;
        }
        if self.gpu_blit_buffers(buffers, x, y, src, Some(op), if is_index { ODSY_INDEX_MASK } else { u32::MAX }, if is_index { 0 } else { 0xff }) {
            return;
        }
        for b in 0..ODSY_BUFFERS {
            if buffers & (1u32 << b) != 0 {
                let out = logic_op(op, src, self.buffer_load(b, x, y));

                self.buffer_store(b, x, y, if is_index { out & ODSY_INDEX_MASK } else { out | 0xff });
            }
        }
    }

    /// The solid rectangle fill: BEGIN(QUADS)/COLOR_3UB or COLOR_INDEX/four VERTEX_2I
    /// corners/END drives the solid rectangle fill. Board::gl_end() routes a
    /// QUADS fill here when pattern mode is off, and to Board::pattern_fill()
    /// when it is on. is_index selects whether value is a packed RGB pixel or a
    /// PseudoColor index resolved per pixel.
    pub fn fill_rect(&mut self, x: u32, y: u32, w: u32, h: u32, is_index: bool, value: u32) {
        let f = self.integer_fill(is_index, value);
        self.fill_box(x as i64, y as i64, x as i64 + w as i64, y as i64 + h as i64, f);
    }

    /// How Board::paint_integer() stores `value` at a pixel, as a FillOp.
    pub(super) fn integer_fill(&self, is_index: bool, value: u32) -> FillOp {
        let src = fill_pixel(is_index, value);
        let enable = if is_index { mode::INDEX_LOGIC_OP } else { mode::COLOR_LOGIC_OP };
        let op = (self.logic_op_word & super::consts::ODSY_LOGIC_OP_MASK) as u8;
        let use_op = (self.stream.pattern_mode & enable) != 0 && op as u32 != LOGIC_COPY;
        if self.dest_is_id_planes() {
            // Board::paint_aux: the low byte, through the operation.
            return FillOp { word: src, op: use_op.then_some(op), and: 0xff, or: 0, byte_mask: None };
        }
        if !use_op {
            return FillOp { word: src, op: None, and: u32::MAX, or: 0, byte_mask: None };
        }
        FillOp { word: src, op: Some(op), and: if is_index { ODSY_INDEX_MASK } else { u32::MAX }, or: if is_index { 0 } else { 0xff }, byte_mask: None }
    }

    /// A 2D fill of the box [x0, x1) x [y0, y1): `f` at each pixel
    /// Board::pixel_allowed() passes, in each buffer the drawing mask
    /// selects, as Board::paint_integer() stores it a pixel at a time. The
    /// GPU unit takes one record per buffer and applies the window's tests
    /// itself; the CPU unit walks the rows' clip runs once.
    pub(super) fn fill_box(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, f: FillOp) {
        let buffers = self.draw_buffers();
        if buffers == 0 || x0 >= x1 || y0 >= y1 {
            return;
        }
        let plane_mask = (self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) as u8;
        let id_test = self.id_compare_active();
        if self.gpu_unit() {
            let planes = DestMem::of(self.vram_ptr(), &self.dest_views());
            let byte = |b: u32| planes.colour[b as usize].v.kind == Kind::Byte;
            // Two buffers where one writes the id bytes the test reads:
            // the test must see each pixel before both stores, which one
            // record per buffer does not keep. Nor does the kernel's test
            // fail a pixel with no id byte, as the CPU's does.
            let per_pixel = id_test && ((buffers == 3 && (byte(0) || byte(1))) || planes.did.v.kind == Kind::None);
            match self.gpu_dest_slot(&planes) {
                Some(slot) if !per_pixel => {
                    let (bx0, by0, bx1, by1) = (x0.max(-1) as i32, y0.max(-1) as i32, (x1 - 1).min(MAX_DIM as i64) as i32, (y1 - 1).min(MAX_DIM as i64) as i32);
                    self.raster.run_now(|ctx| {
                        let g = ctx.gpu.as_mut().expect("the unit");
                        for b in 0..ODSY_BUFFERS {
                            if buffers & (1u32 << b) == 0 {
                                continue;
                            }
                            let mask = if byte(b) { f.byte_mask.unwrap_or(plane_mask) as u32 } else { u32::MAX };
                            let key = super::gpu_unit::BlitKey { window: true, mask, op: f.op.map_or(0, |o| 0x100 | o as u32), and: f.and, or: f.or, ..super::gpu_unit::BlitKey::store(b, mask) };
                            g.blit_fill(&slot, &key, bx0, by0, bx1, by1, f.word);
                        }
                    });
                    // The id and overlay bytes in the mirror too
                    // (mirror_byte_store), with the same tests.
                    let rows: [Option<super::view::Rows>; 2] = std::array::from_fn(|b| {
                        (buffers & (1 << b) != 0 && byte(b as u32)).then(|| super::view::Rows::new(planes.colour[b].mem().untracked(), planes.colour[b].v))
                    });
                    if rows.iter().any(Option::is_some) {
                        self.fill_rows(&planes, &rows, x0, y0, x1, y1, f, plane_mask, id_test);
                    }
                    return;
                }
                _ => {
                    // The kernels cannot take it: a pixel at a time.
                    for y in y0..y1 {
                        let runs = self.row_runs(y);
                        for r in 0..runs.n as usize {
                            for x in x0.max(runs.start[r])..x1.min(runs.end[r]) {
                                self.fill_pixel_slow(x, y, f);
                            }
                        }
                    }
                    return;
                }
            }
        }
        let planes = self.dest_mem();
        let rows: [Option<super::view::Rows>; 2] = std::array::from_fn(|b| (buffers & (1 << b) != 0).then_some(planes.colour[b]));
        self.fill_rows(&planes, &rows, x0, y0, x1, y1, f, plane_mask, id_test);
    }

    /// Board::fill_box's stores into `rows` on the CPU: each row's clip
    /// runs, then the id test a pixel at a time when it is on.
    #[allow(clippy::too_many_arguments, reason = "a box, its fill, and the tests")]
    fn fill_rows(&self, planes: &DestMem, rows: &[Option<super::view::Rows>; 2], x0: i64, y0: i64, x1: i64, y1: i64, f: FillOp, plane_mask: u8, id_test: bool) {
        let c = self.win.id_compare;
        let (id_mask, id_value) = ((c >> ODSY_ID_COMPARE_MASK_SHIFT) as u8, c as u8);
        for y in y0.max(0)..y1.min(MAX_DIM as i64) {
            let runs = self.row_runs(y);
            for r in 0..runs.n as usize {
                let (a, e) = (x0.max(runs.start[r]), x1.min(runs.end[r]));
                if !id_test && f.op.is_none() {
                    // A plain store: the word field by field, a run at a time.
                    for rows in rows.iter().flatten() {
                        if rows.v.kind == Kind::Byte {
                            let m = f.byte_mask.unwrap_or(plane_mask);
                            if m == 0xff {
                                rows.fill(y as i32, a as i32, e as i32 - 1, f.word);
                                continue;
                            }
                            for x in a..e {
                                f.store(rows, x as i32, y as i32, plane_mask);
                            }
                        } else {
                            rows.fill(y as i32, a as i32, e as i32 - 1, f.word);
                        }
                    }
                    continue;
                }
                for x in a..e {
                    let (x, y) = (x as i32, y as i32);
                    if id_test && !planes.did.load1(x, y).is_some_and(|id| (id ^ id_value) & id_mask == 0) {
                        continue;
                    }
                    for r in rows.iter().flatten() {
                        f.store(r, x, y, plane_mask);
                    }
                }
            }
        }
    }

    /// One pixel of a fill the GPU unit's kernels cannot take, through the
    /// per-pixel paths.
    fn fill_pixel_slow(&mut self, x: i64, y: i64, f: FillOp) {
        if !self.pixel_allowed(x, y) {
            return;
        }
        let buffers = self.draw_buffers();
        let plane_mask = (self.plane_write_mask >> ODSY_PLANE_MASK_SHIFT) as u8;
        for b in 0..ODSY_BUFFERS {
            if buffers & (1u32 << b) == 0 {
                continue;
            }
            let rows = self.planes_in(x, y, x, y).colour[b as usize];
            let mask = if rows.v.kind == Kind::Byte { f.byte_mask.unwrap_or(plane_mask) as u32 } else { u32::MAX };
            let key = super::gpu_unit::BlitKey { mask, op: f.op.map_or(0, |o| 0x100 | o as u32), and: f.and, or: f.or, ..super::gpu_unit::BlitKey::store(b, mask) };
            let planes = DestMem::of(self.vram_ptr(), &self.dest_views());
            if let Some(slot) = self.gpu_dest_slot(&planes) {
                self.raster.run_now(|ctx| ctx.gpu.as_mut().expect("the unit").blit_pixel(&slot, &key, x as i32, y as i32, f.word));
                mirror_byte_store(&planes.colour[b as usize], x, y, f.word, &key);
            } else {
                f.store(&rows, x as i32, y as i32, plane_mask);
            }
        }
    }

    /// Board::clear_planes(): the X server's write to the id-plane destination.
    /// For each pixel of the rectangle, replace the plane bits the 0x1023 mask
    /// selects with the value the COLOR_INDEX word carried, in the id byte when
    /// the drawing mask selects buffer 0 and in the overlay pixel byte when it
    /// selects buffer 1, leaving the colour framebuffer untouched.
    pub fn clear_planes(&mut self, x: u32, y: u32, w: u32, h: u32, value: u8, mask: u8) {
        let f = FillOp { word: value as u32, op: None, and: u32::MAX, or: 0, byte_mask: Some(mask) };
        self.fill_box(x as i64, y as i64, x as i64 + w as i64, y as i64 + h as i64, f);
    }

    /// Pattern mode is on when the 0x0404 mode's bits 3,4 are set and a shape is
    /// selected. A quad drawn in this state paints the current colour through the
    /// pattern instead of as a solid rectangle.
    pub fn pattern_enabled(&mut self) -> bool {
        if (self.stream.pattern_mode & pattern::MODE_ENABLE) != pattern::MODE_ENABLE {
            return false;
        }
        matches!(
            self.stream.pattern_shape & !pattern::SHAPE_OPAQUE,
            pattern::SHAPE_64X16 | pattern::SHAPE_32X32 | pattern::SHAPE_16X64
        )
    }

    /// The pattern bit at quad-relative column u, row v, for the current shape.
    /// Row 0 column 0 is the quad's top-left vertex and bit 31 of a row's first
    /// word is its leftmost pixel. A coordinate past the shape's own extent is
    /// treated as clear: the X server never draws a quad larger than its pattern.
    pub fn pattern_bit(&mut self, u: u32, v: u32) -> bool {

        let (word, bit) = match self.stream.pattern_shape & !pattern::SHAPE_OPAQUE {
            pattern::SHAPE_64X16 => {
                if u >= 64 || v >= 16 {
                    return false;
                }
                (self.stream.pattern_buffer[(2 * v + (u >> 5)) as usize], 31 - (u & 31))
            }
            pattern::SHAPE_32X32 => {
                if u >= 32 || v >= 32 {
                    return false;
                }
                (self.stream.pattern_buffer[v as usize], 31 - u)
            }
            pattern::SHAPE_16X64 => {
                if u >= 16 || v >= 64 {
                    return false;
                }
                (self.stream.pattern_buffer[(v >> 1) as usize], 31 - 16 * (v & 1) - u)
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
        let opaque = self.stream.pattern_shape & pattern::SHAPE_OPAQUE != 0;
        let background = self.pattern_background(is_index);
        let (fg, bg) = (self.integer_fill(is_index, value), self.integer_fill(is_index, background));
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
                    let f = if bit { fg } else { bg };
                    self.fill_box(x as i64 + first as i64, py, x as i64 + u as i64, py + 1, f);
                }
            }
        }
    }
}

/// A raster-box coordinate arrives as an IEEE-754 single the X server built
/// with an integer-to-float convert, so the 32-bit operand is a float bit
/// pattern, not an integer. Truncate toward zero; a NaN or negative clamps to
/// the screen origin.
pub(super) fn float_to_u32(bits: u32) -> u32 {
    let f = f32::from_bits(bits);

    if !(f > 0.0f32) {
        return 0;
    }
    f as u32
}

impl Board {

    /// Lay one image pixel, as 0xRRGGBBAA, at the raster cursor in the pixel
    /// mode's destination and advance it. Each axis is a directed
    /// half-open interval: increasing endpoints visit [first, last), while
    /// decreasing endpoints visit [first - 1, last]. Payload advances along X,
    /// then along Y. A scratch destination (SCRATCH_BUFFER) takes box
    /// coordinates with no window offset or tests; a window's are window
    /// relative.
    pub fn image_put(&mut self, rgba: u32, is_index: bool, preserve_alpha: bool) {
        let x_decreasing = self.image.x2 < self.image.x1;
        let y_decreasing = self.image.y2 < self.image.y1;
        let y_finished = if y_decreasing {
            self.image.cur_y == u32::MAX || self.image.cur_y < self.image.y2
        } else {
            self.image.cur_y >= self.image.y2
        };

        if self.image.x2 == self.image.x1 || self.image.y2 == self.image.y1 || y_finished {
            return;
        }
        let word = if is_index || preserve_alpha { rgba } else { (rgba & !0xffu32) | 0xff };
        if self.dest_views().scratch {
            // A texture or scratch image: memory no drawing touches. On the
            // GPU unit it goes into the mirror untracked and up as a
            // rectangle when the image is done (GpuRaster::note_region),
            // with no wait.
            let (x, y) = (self.image.cur_x as i32, self.image.cur_y as i32);
            if self.gpu_unit() {
                let rows = DestMem::of(self.vram_ptr().untracked(), &self.dest_views()).colour[0];
                rows.store_word(x, y, word);
                self.raster.run_now(|ctx| ctx.gpu.as_mut().expect("the unit").note_region(&rows.v, x, y));
            } else {
                let rows = self.dest_mem().colour[0];
                rows.store_word(x, y, word);
            }
        } else if let Some(f) = self.image.fragment.as_deref().filter(|_| !is_index) {
            let word = self.image_transfer(word, is_index);
            let (x, y) = (self.image.cur_x as i64 + self.win.origin[0] as i64, self.image.cur_y as i64 + self.win.origin[1] as i64);
            if self.pixel_allowed(x, y) {
                let src = [word >> 24, word >> 16, word >> 8, word].map(|c| widen8(c & 0xff));
                let zq = f.1;
                match self.image.gpu_slot {
                    // The unit tests and stores it in order, as a fragment.
                    Some(slot) => {
                        let q = super::pipe::ring::BackQuad { x: x as i32, y: y as i32, cov: 1, zq: [zq, 0, 0, 0], rgba: [src.map(|c| c as u16), [0; 4], [0; 4], [0; 4]], ..Default::default() };
                        self.raster.run_now(|ctx| ctx.gpu.as_mut().expect("the unit").push_frag(&q, &slot));
                    }
                    None => f.0.fragment_q(&self.dest_mem(), x as i32, y as i32, zq, src),
                }
            }
        } else {
            let word = self.image_transfer(word, is_index);
            self.paint_pixel(
                self.image.cur_x as i64 + self.win.origin[0] as i64,
                self.image.cur_y as i64 + self.win.origin[1] as i64,
                word,
            );
        }
        self.image_advance(x_decreasing, y_decreasing);
    }

    /// A box record has arrived. Its operands 6 and 9 carry the raster
    /// position's depth when the library depth-tests the image (they are
    /// zero otherwise): normalised-device z on the scale the combined
    /// matrix's depth row uses, 2 * DEPTH_S per unit, offset by one unit
    /// (traced: raster z 0, 0.5 and 1 under glOrtho(..., 2, -2) send DEPTH_S
    /// times 1, 1.25 and 1.5, the same under any glDepthRange). The board
    /// maps it through the depth range as it does a vertex (measured, V10:
    /// the stored depths are 0.5, 0.625 and 0.75, and 0.5 and 0.625 under
    /// glDepthRange(0.25, 0.75)), and the image's pixels then go through the
    /// depth test and the rest of the fragment back end. An image with no
    /// depth is stored as before.
    pub fn image_box_loaded(&mut self) {
        let z = f32::from_bits(self.image.box_words[6]) as f64;

        self.image.fragment = None;
        if z == 0.0 || !z.is_finite() || self.dest_views().scratch {
            return;
        }
        // The image runs under the library's own raster state, so the client's
        // depth sign word (Board::gl_depth_scale()) does not apply.
        let z01 = (z / (2.0 * super::consts::gl_cmd::DEPTH_S)).clamp(0.0, 1.0);
        let [n, f] = self.assembly.xf.depth_range.map(f64::from);
        let zq = word_for_fraction((n + (f - n) * z01).clamp(0.0, 1.0));
        let words = self.fragment_words();
        let backend = BackendState::of(&words.backend, true, true);
        // On the GPU unit the image's pixels go through its back end as
        // fragments under this slot.
        self.image.gpu_slot = None;
        if self.gpu_unit() {
            let planes = DestMem::of(self.vram_ptr(), &self.dest_views());
            self.image.gpu_slot = super::gpu_unit::Slot::new(&words.backend, &planes, false, false, &words.raster, None, None).ok();
        }
        if self.trace.on(super::trace::TRACE_FIFO) {
            let depth_kind = self.dest_views().depth.kind;
            self.trace_note(format_args!(
                "image depth {z01} -> word {zq:#x}: depth test {} write {} func {:?}, targets {:?}, depth plane {:?}",
                backend.depth_test, backend.depth_write, backend.depth_func, backend.targets, depth_kind
            ));
        }
        self.image.fragment = Some(Box::new((backend, zq)));
    }

    /// Step the raster cursor to the next pixel of the open image.
    fn image_advance(&mut self, x_decreasing: bool, y_decreasing: bool) {
        if (x_decreasing && self.image.cur_x == self.image.x2) || (!x_decreasing && self.image.cur_x == self.image.x2.wrapping_sub(1)) {
            self.image.cur_x = if x_decreasing { self.image.x1.wrapping_sub(1) } else { self.image.x1 };
            if y_decreasing {
                self.image.cur_y = if self.image.cur_y == self.image.y2 { u32::MAX } else { self.image.cur_y.wrapping_sub(1) };
            } else {
                self.image.cur_y = self.image.cur_y.wrapping_add(1);
            }
        } else if x_decreasing {
            self.image.cur_x = self.image.cur_x.wrapping_sub(1);
        } else {
            self.image.cur_x = self.image.cur_x.wrapping_add(1);
        }
    }

    /// One byte-stream texel of the open image at the raster cursor, bounded by
    /// the header's group length in pixels. A source format that carries
    /// alpha stores it (code_carries_alpha()); any other stores 255.
    pub fn image_texel(&mut self, rgba: u32, code: u32) {
        if self.image.group_len != 0 && self.image.group_painted >= self.image.group_len {
            return;
        }
        self.image_put(rgba, code_is_index(code), code_carries_alpha(code));
        self.image.group_painted += 1;
    }

    /// The same for a pixel of the library's write table: `b` holds its bytes
    /// first byte first. It stores the alpha the source carries.
    pub fn image_texel_form(&mut self, form: &WriteForm, b: &[u8]) {
        if self.image.group_len != 0 && self.image.group_painted >= self.image.group_len {
            return;
        }
        self.image_put(form.decode(b), form.is_index(), form.has_alpha());
        self.image.group_painted += 1;
    }
}

/// Whether a GL image write under source format `code` stores the source's
/// alpha rather than 255.
///
/// ESTABLISHED: the library sends
/// the source's alpha bytes unchanged under 0x4b, 0x48, 0x6b, 0x4c and
/// 0x6c, and no header bit asks the board to replace them. INFERRED:
/// the board stores that alpha. The write mask's alpha enable (block
/// 0x052) is not applied to image writes in this model.
///
/// The X server's on-screen images under 0x4c come by DMA to the colour
/// buffer (Board::dma_put()), which still stores 255: they would, by inference, write the A byte, 0 in every traced upload, but the X server
/// passes client pixels through, and whether alpha is stored at all for a
/// window whose visual has none is unknown. The model keeps one plane for
/// every visual, so it cannot tell such a window apart.
pub(super) fn code_carries_alpha(code: u32) -> bool {
    matches!(
        code,
        image::CODE_RGBA
            | image::CODE_LUM_ALPHA
            | image::CODE_ABGR
            | image::CODE_RGBA12
            | image::CODE_ABGR12
    )
}

pub(super) fn unorm16_to_8(v: u32) -> u32 {
    (v.wrapping_mul(255).wrapping_add(65535 / 2)) / 65535
}

/// Components of a 12-bit texture source code, or 0 for any other code.
pub(super) fn texel12_components(code: u32) -> u32 {
    match code {
        image::CODE_LUM12 => 1,
        image::CODE_LUM_ALPHA12 => 2,
        image::CODE_RGB12 => 3,
        image::CODE_RGBA12 | image::CODE_ABGR12 => 4,
        _ => 0,
    }
}

/// A 12-bit texture source pixel as 0xRRGGBBAA: its components, each a byte or
/// (wide) a big-endian 16-bit word, in the order its code names.
pub(super) fn decode_texel12(code: u32, b: &[u8], wide: bool) -> u32 {
    let mut c = [0u32; 4];
    let n = texel12_components(code) as usize;

    for i in 0..n {
        c[i] = if wide { unorm16_to_8(be16(&b[2 * i..][..2]) as u32) } else { b[i] as u32 };
    }
    match code {
        image::CODE_LUM12 => pack_pixel(c[0], c[0], c[0]),
        image::CODE_LUM_ALPHA12 => (pack_pixel(c[0], c[0], c[0]) & !0xffu32) | c[1],
        image::CODE_RGB12 => pack_pixel(c[0], c[1], c[2]),
        image::CODE_ABGR12 => (pack_pixel(c[3], c[2], c[1]) & !0xffu32) | c[0],
        _ => (pack_pixel(c[0], c[1], c[2]) & !0xffu32) | c[3],
    }
}

impl Board {
    /// Decode one pixel of a byte-stream source format whose bytes are b[0 ..]:
    /// its 0xRRGGBBAA word, or for 0x41 and 0x61 the raw index. None for a code
    /// this model does not know.
    pub fn decode_pixel(&mut self, code: u32, b: &[u8]) -> Option<u32> {
        match code {
            image::CODE_ABGR => Some((pack_pixel(b[3] as u32, b[2] as u32, b[1] as u32) & !0xffu32) | b[0] as u32),
            image::CODE_FALLBACK_XRGB => Some(pack_pixel(b[1] as u32, b[2] as u32, b[3] as u32)),
            image::CODE_RGBA | image::CODE_RGBA12 => Some(be32(&b[..4])),
            image::CODE_RGB => Some(pack_pixel(b[0] as u32, b[1] as u32, b[2] as u32)),
            image::CODE_LUM => Some(pack_pixel(b[0] as u32, b[0] as u32, b[0] as u32)),
            image::CODE_LUM_ALPHA => Some((pack_pixel(b[0] as u32, b[0] as u32, b[0] as u32) & !0xffu32) | b[1] as u32),
            image::CODE_LUM12 | image::CODE_LUM_ALPHA12 => {
                // A burst's 16-bit components (pixel_bytes()).
                Some(decode_texel12(code, b, true))
            }
            image::CODE_CI => {
                // The raw index: scanout resolves it through the pixel's drawable.
                Some(b[0] as u32)
            }
            image::CODE_CI16 => Some(be16(&b[..2]) as u32 & ODSY_INDEX_MASK),
            _ => None,
        }
    }
}

/// Whether a source format carries colour indices rather than colours.
pub(super) fn code_is_index(code: u32) -> bool {
    code == image::CODE_CI || code == image::CODE_CI16
}

/// Bytes a pixel of a source format takes, 0 for an unknown code.
pub(super) fn pixel_bytes(code: u32) -> u32 {
    match code {
        image::CODE_ABGR | image::CODE_FALLBACK_XRGB | image::CODE_RGBA | image::CODE_RGBA12 => 4,
        image::CODE_RGB => 3,
        image::CODE_LUM | image::CODE_CI => 1,
        image::CODE_LUM_ALPHA | image::CODE_LUM12 | image::CODE_CI16 => 2,
        image::CODE_LUM_ALPHA12 => 4,
        _ => 0,
    }
}

pub(super) fn image_pairing_supported(code: u32, fmt: u32) -> bool {
    if code == image::CODE_RGBA12 {
        return fmt == image::FMT_RGB || fmt == image::FMT_RGBA16;
    }
    if code == image::CODE_RGB12 || code == image::CODE_ABGR12 {
        return fmt == image::FMT_RGBA16;
    }
    // The library sends 8-bit luminance-alpha under the colour-index or the
    // RGB tag, and its 16-bit luminance and luminance-alpha as 16-bit
    // components under the RGBA16 tag as well as the other two (observed on
    // 13x5 draws of the GL test suite).
    if code == image::CODE_LUM_ALPHA {
        return fmt != image::FMT_RGBA16;
    }
    if code == image::CODE_LUM12 || code == image::CODE_LUM_ALPHA12 {
        return true;
    }
    fmt != image::FMT_RGBA16
}

impl Board {
    /// Whether the open image's header and a burst's format tag go together:
    /// image_pairing_supported() for the codes the byte-stream tables know,
    /// and for any other code, whatever the write table (pixel_write.rs)
    /// takes, under any tag.
    pub fn image_burst_supported(&self, fmt: u32) -> bool {
        image_pairing_supported(self.image.code, fmt)
            || (pixel_bytes(self.image.code) == 0 && texel12_components(self.image.code) == 0 && write_form(self.image.write_code).is_some())
    }

    /// One word of a format-0, type-4 burst: a pixel of the depth and
    /// stencil word, four bytes.
    ///
    /// ESTABLISHED: the
    /// burst's operand carries no size field and no field codes, so the
    /// header does not say which plane; the library selects the plane by
    /// the write-mask block (0x052) it sends just before the burst. Its depth
    /// download clears the stencil mask (bits 19:12) and keeps the depth-write
    /// bit (24) as the application has it; its stencil download clears bit 24
    /// and leaves the stencil mask. INFERRED: the enables decide which part of
    /// the stored pixel changes, so the model writes the depth plane's word
    /// (bits 23:0) when bit 24 is set, and the stencil plane's bits 19:12 of
    /// block 0x052 select as `(data & mask) | (old & !mask)`. ESTABLISHED:
    /// the library packs
    /// the stencil index at bits 23:16 with no mask (index 0x11 is
    /// 0x0011_0000) and the burst copies the word, so that byte is the data;
    /// the 8-bit mask in bits 19:12 of block 0x052 applies to it, not to the
    /// word as sent. UNKNOWN: how the board aligns the two, and where it
    /// keeps the index inside a stored pixel. UNKNOWN: whether
    /// the board runs the burst through the depth and stencil tests, and the
    /// effect of their enables on it; the model applies neither. The same
    /// clip-rectangle and window-ID tests as a GL fragment apply (model
    /// choice, as for the DMA write).
    pub fn image_depth_stencil_word(&mut self, word: u32) {
        let x_decreasing = self.image.x2 < self.image.x1;
        let y_decreasing = self.image.y2 < self.image.y1;
        let y_finished = if y_decreasing {
            self.image.cur_y == u32::MAX || self.image.cur_y < self.image.y2
        } else {
            self.image.cur_y >= self.image.y2
        };

        if (self.image.group_len != 0 && self.image.group_painted >= self.image.group_len)
            || self.image.x2 == self.image.x1
            || self.image.y2 == self.image.y1
            || y_finished
        {
            return;
        }
        self.image.group_painted += 1;
        let x = self.image.cur_x as i64 + self.win.origin[0] as i64;
        let y = self.image.cur_y as i64 + self.win.origin[1] as i64;

        // Only a destination with a depth field takes the burst's depth and stencil.
        if self.dest_views().depth.kind != Kind::None && !self.dest_views().scratch && self.pixel_allowed(x, y) {
            let mask = (self.gl_write_mask >> write::STENCIL_SHIFT) & write::STENCIL_MASK;

            if self.gl_write_mask & write::DEPTH != 0 {
                self.depth_store(x, y, word & ODSY_DEPTH24_MAX);
            }
            if mask != 0 {
                let old = self.stencil_load(x, y);
                let data = (word >> ODSY_BURST_STENCIL_SHIFT) & write::STENCIL_MASK;

                self.stencil_store(x, y, ((data & mask) | (old & !mask)) as u8);
            }
        }
        self.image_advance(x_decreasing, y_decreasing);
    }

    /// One data word of a pixel burst, laid out by the transfer header's format
    /// code: its bytes join a stream that yields a pixel every
    /// pixel_bytes() bytes, so a three-byte pixel may span two words. A
    /// bounded group's trailing padding (past image_group_len) is dropped with any
    /// partial pixel, so the raster position stays exact. A header with a code
    /// this model does not know falls back on the burst tag's format, 0x11 one
    /// 0x00RRGGBB pixel a word and 0x10 four indices, and is reported once.
    pub fn image_burst_word(&mut self, word: u32) {
        let mut code = self.image.code;
        let mut n = pixel_bytes(code);
        let mut b = [0u8; 4];

        if !self.image_burst_supported(self.image.fmt) {
            return;
        }
        if code == image::CODE_DEPTH_STENCIL {
            self.image_depth_stencil_word(word);
            return;
        }
        if self.image.fmt == image::FMT_RGBA16 && texel12_components(code) != 0 {
            // A stream of 16-bit components, high half first, a pixel every
            // texel12_components() halves: two words for 0x6b and 0x6c, a
            // word and a half for 0x69. At most two halves wait between words;
            // image_nbytes counts them, image_bytes holds them.
            let comps = texel12_components(code) as usize;
            let mut halves = [0u16; 4];
            let mut held = self.image.nbytes.min(2) as usize;

            if self.image.group_len != 0 && self.image.group_painted >= self.image.group_len {
                return;
            }
            for (i, half) in halves.iter_mut().enumerate().take(held) {
                *half = (self.image.bytes >> (16 * (held - 1 - i))) as u16;
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
                let row = self.image.cur_y;

                self.image_put(decode_texel12(code, &px, true), false, true);
                held = 0;
                if h == 0 && self.image.cur_y != row {
                    // The row ended in the word's first half: the second
                    // is pad, and the next row starts on the next word.
                    break;
                }
            }
            self.image.bytes = if held > 1 {
                ((halves[0] as u32) << 16) | halves[1] as u32
            } else if held != 0 {
                halves[0] as u32
            } else {
                0
            };
            self.image.nbytes = held as u32;
            self.image.group_painted += 1;
            if self.image.group_len != 0 && self.image.group_painted >= self.image.group_len && self.image.nbytes != 0 {
                self.image.bytes = 0;
                self.image.nbytes = 0;
            }
            return;
        }

        let mut form = None;

        if n == 0 {
            // A code the byte-stream tables above do not know: the library's
            // write table (pixel_write.rs), for pixels of up to sixteen bytes.
            form = write_form(self.image.write_code).filter(|f| f.bytes <= 16);
            if let Some(f) = &form {
                n = f.bytes;
            } else {
                if !self.diag.image_code_reported {
                    self.diag.image_code_reported = true;
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: image source format {:#x} is not modeled; laid out by the burst tag",
                        self.image.write_code
                    );
                }
                code = if self.image.fmt == image::FMT_RGB { image::CODE_FALLBACK_XRGB } else { image::CODE_CI };
                n = pixel_bytes(code);
            }
        }
        b[..4].copy_from_slice(&word.to_be_bytes());
        if let Some(f) = &form {
            // The write table's pixels: any number of bytes up to a pixel of
            // four 32-bit components, in arrival order.
            for (i, &byte) in b[..4].iter().enumerate() {
                self.image.pixel[self.image.nbytes as usize] = byte;
                self.image.nbytes += 1;
                if self.image.nbytes < f.bytes {
                    continue;
                }
                self.image.nbytes = 0;
                let px = self.image.pixel;
                let row = self.image.cur_y;

                self.image_texel_form(f, &px);
                if i < 3 && self.image.cur_y != row {
                    // The row ended inside the word: the rest is pad.
                    break;
                }
            }
            return;
        }
        // OBSERVED (guest traces of GL draws): each row of 16- and 24-bit
        // pixels ends on a word, padded with 0x5a, and the next row starts
        // on the next word. Eight-bit pixels (the X server's index images)
        // are left as they were: no trace shows their rows padded.
        let padded_rows = (n == 2 || n == 3) && code != image::CODE_CI16;

        for (i, &byte) in b[..4].iter().enumerate() {
            let mut px = [0u8; 4];

            self.image.bytes = (self.image.bytes << 8) | byte as u32;
            self.image.nbytes += 1;
            if self.image.nbytes < n {
                continue;
            }
            self.image.nbytes = 0;
            // The pixel's bytes, first byte highest, at the start of px[].
            px[..4].copy_from_slice(&(self.image.bytes << (8 * (4 - n))).to_be_bytes());
            let rgba = self.decode_pixel(code, &px).unwrap_or(0);
            let row = self.image.cur_y;

            self.image_texel(rgba, code);
            if padded_rows && i < 3 && self.image.cur_y != row {
                break;
            }
        }
    }
}

/// The format of a pixel-burst register-select word, None for any other.
pub(super) fn image_burst_fmt(reg: u32) -> Option<u32> {
    let f = reg & image::BURST_FMT_MASK;

    if (reg & image::BURST_BASE) == 0 || (reg & image::BURST_RESERVED_MASK) != 0 {
        return None;
    }
    match f {
        image::FMT_CI | image::FMT_RGB | image::FMT_RGBA16 => Some(f),
        // Other documented pixel-burst formats remain unsupported.
        _ => None,
    }
}


/// The most colour targets a clear writes.
pub(super) const MAX_TARGETS: u32 = 4;

/// What a clear writes, bits of RasterOp::clear_buffers.
pub(super) const CLEAR_DEPTH: u32 = 0x0100;
pub(super) const CLEAR_STENCIL: u32 = 0x0400;
pub(super) const CLEAR_COLOUR: u32 = 0x4000;


/// A raster operation's kind: a clear of a rectangle.
pub(super) const RASTER_CLEAR: u32 = 1;

/// A clear of a rectangle (the GL rectangle words, Board::gl_rect()), in
/// plane pixels, the rectangle inclusive.
#[derive(Clone, Copy, Default)]
pub(super) struct RasterOp {
    pub kind: u32,
    /// The colour buffers written.
    pub targets: [u32; MAX_TARGETS as usize],
    pub ntargets: u32,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    /// Only pixels inside these are written; at most the rectangle's. An
    /// empty list is how the board says its window has no visible pixel.
    pub clip: super::clip::ClipList,
    /// CLEAR_COLOUR, CLEAR_DEPTH and CLEAR_STENCIL.
    pub clear_buffers: u32,
    /// The clear colour in [0, 1]; the back end quantizes it to the plane.
    pub clear_rgba: [f32; 4],
    /// The components written: red, green, blue, alpha from bit 0.
    pub clear_colour_mask: u32,
    /// In the depth plane's units; the plane keeps its low bits.
    pub clear_depth24: u32,
    /// The stencil clear value, written under the stencil write mask.
    pub clear_stencil: u32,
    pub stencil_mask: u32,
}
