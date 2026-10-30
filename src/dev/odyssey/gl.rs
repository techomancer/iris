//! The colour, depth, stencil, fog and texture hooks the board gives the shared
//! rasterizer, and the state its GL draws are made under.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.


use crate::devlog::LogModule;
use super::bits::sextract32;
use super::board::{
    Board, ODSY_CLIP_RECTS, ODSY_HEIGHT, ODSY_OBJECTS, ODSY_OBJ_POOL_WORDS,
    ODSY_TEX_COLOUR_TABLE_ENTRIES, ODSY_TEX_DETAIL_FUNCTION_ENTRIES, ODSY_TEX_LEVELS,
    ODSY_VOLUME_SLICES, ODSY_WIDTH,
};
use super::consts::{
    clip, colour, gl_cmd, mode, pattern, prog, stencil, tex, write, ODSY_AUX_OVERLAY_BUFFER,
    ODSY_BUFFERS, ODSY_DEPTH24_MAX, ODSY_DEPTH_FUNC_MASK, ODSY_DRAW_WORD_STATE,
    ODSY_ID_COMPARE_ENABLE, ODSY_ID_COMPARE_MASK_SHIFT, ODSY_INDEX_MASK,
    ODSY_VOLUME_SLICE_SHIFT_MAX,
};
use super::draw::{gl_rules, id_compare_passes};
use super::geometry::{f32_bits, gl_transpose, odyssey_f32};
use super::gl_context::{
    op, vtx, ColourTarget, GL_ADD, GL_ALPHA, GL_ALPHA_TEST, GL_BACK, GL_BLEND, GL_CCW, GL_CLAMP,
    GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_CLIP_PLANE0, GL_CLIP_PLANES, GL_COLOR_BUFFER_BIT,
    GL_COLOR_LOGIC_OP, GL_CONSTANT_ALPHA, GL_CONSTANT_COLOR, GL_CULL_FACE, GL_CW, GL_DECAL,
    GL_DECR, GL_DEPTH_TEST, GL_DRAW_BACK, GL_DRAW_FRONT, GL_DRAW_OK, GL_DST_ALPHA, GL_DST_COLOR,
    GL_FILL, GL_FLAT, GL_FOG, GL_FRONT, GL_FRONT_AND_BACK, GL_FUNC_ADD, GL_FUNC_REVERSE_SUBTRACT,
    GL_FUNC_SUBTRACT, GL_INCR, GL_INDEX_LOGIC_OP, GL_INTENSITY, GL_INVERT, GL_KEEP, GL_LINE,
    GL_LINEAR, GL_LINEAR_DETAIL_ALPHA_SGIS, GL_LINEAR_DETAIL_COLOR_SGIS, GL_LINEAR_DETAIL_SGIS,
    GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_NEAREST, GL_LINE_SMOOTH, GL_LINE_STIPPLE,
    GL_LOGIC_OP, GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_MAX, GL_MAX_TARGETS, GL_MIN, GL_MODULATE,
    GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR, GL_NEAREST_MIPMAP_NEAREST, GL_NEVER_ENUM,
    GL_ONE_MINUS_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_COLOR, GL_ONE_MINUS_DST_ALPHA,
    GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR, GL_POINT,
    GL_POINT_SMOOTH, GL_POLYGON_OFFSET_FILL, GL_POLYGON_OFFSET_LINE, GL_POLYGON_OFFSET_POINT,
    GL_POLYGON_STIPPLE, GL_REPEAT, GL_REPLACE, GL_RGB, GL_RGBA, GL_SMOOTH, GL_SRC_ALPHA,
    GL_SRC_ALPHA_SATURATE, GL_SRC_COLOR, GL_STENCIL_TEST, GL_TEXTURE, GlCmd, GlContext, GlDraw,
    GlDrawResult, GlDrawState, GlDrawable, GlOp, GlSink, RASTER_CLEAR, RasterOp,
};
use super::memory::{plane_bpp, word_load, word_store, words_fill, ClipRect, Plane, Rows, RowsMut};
use super::raster::{
    cmax, cmin, GL_TEX_LEVELS_MAX, GlTexImage, GlTexLevel, GlTexPyramid, NoTexels, Planes,
    RasterColour, RasterRules, RasterTarget, Texels,
};
use super::sampler::{texture_level_format, Refusal, TexelFormat, Wrap};
use super::raster_unit::{
    primitive_cost, rect_cost, touch_row, LINE_PHASE_ON_RASTER, PlaneSpans, RasterCtx, Span,
    TOUCHED_OVERLAY, TouchedRows,
};
/// One level's texels in the object store: `width` by `height` words from
/// word `offset`, rows `stride` words apart.
#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(PartialEq, Debug))]
pub(super) struct Texture {
    pub offset: u32,
    pub stride: u32,
    pub width: u32,
    pub height: u32,
    pub base_format: u32,
    pub words: u32,
    /// A 2D level's border rows are stored, so a one-row level is not 1D.
    pub border_rows: bool,
}

/// The detail function table's weights, unsigned 12-bit fractions, entry i
/// at LOD -i/32 (TextureDetailWeight::at()).
#[derive(Clone, Copy)]
#[cfg_attr(test, derive(PartialEq, Debug))]
pub(super) struct TextureDetailWeight {
    pub table: [u16; ODSY_TEX_DETAIL_FUNCTION_ENTRIES as usize],
}

impl Default for TextureDetailWeight {
    fn default() -> Self {
        TextureDetailWeight { table: [0; ODSY_TEX_DETAIL_FUNCTION_ENTRIES as usize] }
    }
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(PartialEq, Debug))]
pub(super) struct TextureVolume {
    pub base: u32,
    pub stride_units: u32,
    pub level_word2: u32,
    /// Its slices in tex_slice_offset/tex_slice_width (texture setup).
    pub first_slice: u32,
    pub slices: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub base_format: u32,
    pub words: u32,
    pub colour_table_alpha: bool,
    pub border: bool,
    pub colour_table: bool,
    pub scale_bias: bool,
}

/// A GlTexPyramid's shape: its levels are the first `count` of
/// TextureChain::levels.
#[derive(Clone, Copy, Default)]
pub(super) struct PyramidShape {
    pub count: u32,
    pub base_level: u32,
    pub max_level: u32,
    pub min_lod: f64,
    pub max_lod: f64,
}

/// Where a draw's texture lives (Board::texture_setup()).
#[derive(Clone, Default)]
pub(super) struct TextureChain {
    /// The level-zero volume, and the post-filter and detail weight state.
    pub volume: TextureVolume,
    pub volumes: [TextureVolume; GL_TEX_LEVELS_MAX as usize],
    pub storage: [Texture; GL_TEX_LEVELS_MAX as usize],
    pub levels: [GlTexLevel; GL_TEX_LEVELS_MAX as usize],
    pub pyramid: Option<PyramidShape>,
    pub detail_storage: Texture,
    pub detail_image: GlTexLevel,
    pub detail_weight: TextureDetailWeight,
    /// The leading entries of `volumes`, `storage` and `levels` written since
    /// the chain was last cleared; the rest are still default.
    filled: usize,
}

impl TextureChain {
    /// Make the chain default again, writing only what a setup wrote. A chain
    /// is about 7 KB, and a textured draw cleared all of it, though a typical
    /// texture fills a few levels.
    pub fn clear(&mut self) {
        let n = self.filled.min(self.levels.len());

        // Every setup in the tests proves `touch` covers what it writes.
        #[cfg(test)]
        {
            assert!(self.volumes[n..].iter().all(|v| *v == TextureVolume::default()), "a written volume was not touched");
            assert!(self.storage[n..].iter().all(|t| *t == Texture::default()), "a written level store was not touched");
            assert!(self.levels[n..].iter().all(|l| *l == GlTexLevel::default()), "a written level was not touched");
        }
        self.volumes[..n].fill(TextureVolume::default());
        self.storage[..n].fill(Texture::default());
        self.levels[..n].fill(GlTexLevel::default());
        self.volume = TextureVolume::default();
        self.pyramid = None;
        self.detail_storage = Texture::default();
        self.detail_image = GlTexLevel::default();
        self.detail_weight = TextureDetailWeight::default();
        self.filled = 0;
    }

    /// Note that level `i`'s entries are about to be written.
    pub fn touch(&mut self, i: u32) {
        self.filled = self.filled.max(i as usize + 1);
    }

    /// image with this chain's pyramid.
    pub fn image<'a>(&'a self, image: &GlTexImage<'static>) -> GlTexImage<'a> {
        let mut image: GlTexImage<'a> = *image;

        image.pyramid = self.pyramid.map(|p| GlTexPyramid {
            levels: &self.levels[..p.count as usize],
            base_level: p.base_level,
            max_level: p.max_level,
            min_lod: p.min_lod,
            max_lod: p.max_lod,
        });
        image
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct TextureDetailSampler {
    pub descriptor: u32,
    pub width: u32,
    pub height: u32,
    pub level: i32,
    pub mode: u32,
}

impl Board {
    /// Whether a GL fragment's colour is an index (prog::COLOUR_MODE).
    pub fn gl_index_mode(&self) -> bool {
        self.colour_mode[0] & colour::MODE_INDEX != 0
    }
}

/// A colour index carried through the shared rasterizer, which knows only
/// RGBA: bits 7:0 of the index ride in blue and bits 11:8 in green, so at
/// the rules' eight colour bits the stored colour's low twelve bits are the
/// index again (gl_store_colour()). Smooth shading then
/// interpolates the index as GL does, within one run of 256.
pub(super) fn gl_index_rgba(index: u32) -> [f32; 4] {
    [0.0, ((index >> 8) & 0xf) as f32 / 255.0f32, (index & 0xff) as f32 / 255.0f32, 1.0]
}

/// A float index (glIndexf, glClearIndex): its integer part, wrapped to the
/// twelve index bits the way a write keeps only the buffer's bits.
pub(super) fn float_index(bits: u32) -> u32 {
    let mut f = f32::from_bits(bits);

    if !f.is_finite() {
        return 0;
    }
    // The integer part, wrapped to the buffer's twelve bits without an
    // out-of-range conversion: fmodf is exact.
    f = f.floor() % 4096.0f32;
    if f < 0.0 {
        f += 4096.0f32;
    }
    (f as u32) & ODSY_INDEX_MASK
}

impl Board {
    /// Whether GL draws go to the id-plane destination: it is selected and the
    /// kernel has bound it as the draw drawable (0x608, bit 31 aside), as it
    /// does for flight's overlay context (0x80142000; OBSERVED). The X server
    /// selects the same destination for its own id writes, which a GL
    /// context's draws do not follow.
    ///
    /// APPROXIMATION: this rule reconciles those two observations; it is not a
    /// known board mechanism. An equally consistent alternative is that GL
    /// draws simply follow the drawable 0x608 binds, whatever 0x1017 last
    /// selected.
    pub fn gl_to_aux(&mut self) -> bool {
        self.dest_is_id_planes() && (self.draw_word & !ODSY_DRAW_WORD_STATE) == self.dest_base
    }

    /// A GL draw aimed at the id-plane destination that has no modelled home.
    pub fn gl_aux_refused(&mut self, what: &str) {
        if !self.diag.gl_aux_reported {
            self.diag.gl_aux_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: {} to the id-plane destination is not modeled; dropped", what);
        }
    }

    /// What GlPlanes declines, reported once, from the draw rather than from
    /// a pixel: the conditions are fixed for the whole draw, and GlPlanes
    /// then only stores. colour says whether the draw writes colour at
    /// all.
    pub fn gl_report_refusals(&mut self, colour: bool) {
        let index_mask = self.gl_write_mask & write::INDEX_MASK;
        let mask = (self.gl_write_mask >> write::COLOUR_SHIFT) & write::COLOUR_MASK;
        let buffers = self.draw_buffers();

        if !colour {
            return;
        }
        if !self.gl_index_mode() {
            if buffers != 0 && self.gl_to_aux() {
                self.gl_aux_refused("an RGBA draw");
            }
            return;
        }
        if index_mask != write::INDEX_MASK && !self.diag.gl_index_mask_logged {
            self.diag.gl_index_mask_logged = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: index write mask {:#05x} is decoded but not applied", index_mask);
        }
        if mask != 0 && (buffers & !(1u32 << ODSY_AUX_OVERLAY_BUFFER)) != 0 && self.gl_to_aux() {
            self.gl_aux_refused("a GL id-byte write");
        }
    }
}

/// The board state a GL draw tests and stores its pixels under, fixed for
/// the whole draw.
#[derive(Clone, Copy)]
pub(super) struct GlPixelState<'a> {
    pub did: &'a [u8],
    pub id_compare: u32,
    pub clip_rect: [ClipRect; ODSY_CLIP_RECTS as usize],
    pub clip_kept: u32,
    pub write_mask: u32,
    /// Board::gl_to_aux() for the draw.
    pub to_aux: bool,
    pub fog: [u32; 6],
    pub colour: [Plane; ODSY_BUFFERS as usize],
    pub depth: Plane,
    pub stencil: Plane,
}

impl GlPixelState<'_> {
    /// Board::pixel_allowed(), from the draw's copy of its state.
    pub fn pixel_allowed(&self, x: i64, y: i64) -> bool {
        x >= 0
            && y >= 0
            && x < ODSY_WIDTH as i64
            && y < ODSY_HEIGHT as i64
            && id_compare_passes(self.did, self.id_compare, x, y)
            && self.clip_rect[..self.clip_kept as usize]
                .iter()
                .any(|r| x >= r.x0 as i64 && x <= r.x1 as i64 && y >= r.y0 as i64 && y <= r.y1 as i64)
    }
}

/// The board's planes as a GL draw borrows them: the two colour buffers,
/// the model depth and stencil planes and the overlay byte.
pub(super) struct GlPixels<'a> {
    pub state: GlPixelState<'a>,
    pub colour: [RowsMut<'a, u8>; ODSY_BUFFERS as usize],
    pub depth: RowsMut<'a, u8>,
    pub stencil: RowsMut<'a, u8>,
    pub overlay: RowsMut<'a, u8>,
    /// The rows the draw stored colour to (see `GlPlanes`), for the raster
    /// unit to hand the display.
    pub touched: TouchedRows,
    /// The colour buffers a lent row's stores may reach, bit b for buffer
    /// b: what the draw's targets name. A lent row marks them all.
    pub row_targets: u32,
    /// GlPlanes::row() lends rows; tests clear it to compare the row loops
    /// with the per-pixel path.
    pub lend_rows: bool,
}

impl GlPixels<'_> {
    /// Hand the rows the draw wrote to the raster unit's record, for the
    /// display to resolve (RasterCtx::drawn).
    pub fn record(&self, ctx: &mut RasterCtx) {
        ctx.drawn.add(&self.touched, self.state.id_compare);
    }

    /// The buffers a list of colour targets names, as `row_targets` has them.
    pub fn target_bits(targets: &[ColourTarget]) -> u32 {
        targets.iter().filter(|&&t| t < ODSY_BUFFERS).fold(0, |bits, &t| bits | 1 << t)
    }
}

impl GlPixels<'_> {
    /// The planes, for drawing on this thread.
    pub fn planes(&mut self) -> GlPlanes<'_> {
        let [c0, c1] = &mut self.colour;

        GlPlanes {
            state: self.state,
            colour: [c0.view(), c1.view()],
            depth: self.depth.view(),
            stencil: self.stencil.view(),
            overlay: self.overlay.view(),
            touched: &self.touched,
            row_targets: self.row_targets,
            lend_rows: self.lend_rows,
        }
    }
}

/// GlPixels as one thread of a draw stores to them. Colour-index draws use
/// store_colour; RGBA draws store_rgba and load_rgba (Board::gl_target()).
/// Their once-only reports are made before the draw
/// (Board::gl_report_refusals()).
pub(super) struct GlPlanes<'a> {
    state: GlPixelState<'a>,
    colour: [Rows<'a, u8>; ODSY_BUFFERS as usize],
    depth: Rows<'a, u8>,
    stencil: Rows<'a, u8>,
    overlay: Rows<'a, u8>,
    /// Each row a colour store (or a lent row, for the draw's
    /// `row_targets`) reached, by buffer: what the display resolves again.
    touched: &'a TouchedRows,
    row_targets: u32,
    lend_rows: bool,
}

/// One row of GlPlanes as a row loop reaches it (GlPlanes::row()): the depth
/// plane's and both colour buffers' words and the stencil plane's bytes for
/// the row's pixels, and of GlPixelState::pixel_allowed() only the id test,
/// which is per pixel; the plane bound and clip rectangles were tested for
/// the whole row. Its
/// methods take plane coordinates on that row and answer as GlPlanes would.
pub(super) struct GlPlaneRow<'a> {
    x0: i32,
    depth: &'a [std::cell::Cell<u8>],
    depth_le: bool,
    colour: [&'a [std::cell::Cell<u8>]; ODSY_BUFFERS as usize],
    colour_le: [bool; ODSY_BUFFERS as usize],
    stencil: &'a [std::cell::Cell<u8>],
    /// Stores reach the planes: not a draw to the id-plane destination.
    store: bool,
    /// The id bytes of the row's pixels, or empty with the id test off.
    did: &'a [u8],
    id_value: u8,
    id_mask: u8,
}

impl GlPlaneRow<'_> {
    #[inline(always)]
    fn allowed(&self, i: usize) -> bool {
        self.did.is_empty() || ((self.did[i] ^ self.id_value) & self.id_mask) == 0
    }

    /// Whether the id test passes the pixels x0 to x1 of the row.
    fn all_allowed(&self, x0: i32, x1: i32) -> bool {
        (x0..=x1).all(|x| self.allowed((x - self.x0) as usize))
    }

    /// store_depth() of one value at the row's pixels x0 to x1: a slice fill
    /// where the id test passes them all, else a pixel at a time.
    pub fn fill_depth(&self, x0: i32, x1: i32, depth: u32) {
        if !self.store {
            return;
        }
        if self.all_allowed(x0, x1) {
            let (i0, i1) = ((x0 - self.x0) as usize, (x1 - self.x0) as usize);

            words_fill(&self.depth[4 * i0..4 * (i1 + 1)], self.depth_le, depth & ODSY_DEPTH24_MAX);
        } else {
            for x in x0..=x1 {
                self.store_depth(x, 0, depth);
            }
        }
    }

    /// store_rgba() of one colour at the row's pixels x0 to x1, as
    /// fill_depth() does.
    pub fn fill_rgba(&self, target: ColourTarget, x0: i32, x1: i32, colour: &RasterColour) {
        if target >= ODSY_BUFFERS || !self.store {
            return;
        }
        if self.all_allowed(x0, x1) {
            let (i0, i1) = ((x0 - self.x0) as usize, (x1 - self.x0) as usize);
            let c = &colour.component;

            words_fill(
                &self.colour[target as usize][4 * i0..4 * (i1 + 1)],
                self.colour_le[target as usize],
                (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3],
            );
        } else {
            for x in x0..=x1 {
                self.store_rgba(target, x, 0, colour);
            }
        }
    }
}

#[cfg(test)]
mod plane_row_tests {
    use super::*;
    use std::cell::Cell;

    const N: usize = 24;

    fn noise(seed: u8) -> Vec<Cell<u8>> {
        (0..4 * N).map(|i| Cell::new((i as u8).wrapping_mul(37).wrapping_add(seed))).collect()
    }

    fn bytes(v: &[Cell<u8>]) -> Vec<u8> {
        v.iter().map(Cell::get).collect()
    }

    // A run fill stores what the pixel stores do, in either byte order, with
    // the id test passing all, some or none of the row's pixels, a draw to
    // the id plane, a target past the buffers, and runs that start and end inside the row.
    #[test]
    fn run_fills_match_the_pixel_stores() {
        let pass = [0u8; N];
        let mut some = [0u8; N];
        let none = [0xffu8; N];
        let mut one_in_range = [0u8; N];

        for i in (0..N).step_by(5) {
            some[i] = 0xff;
        }
        one_in_range[9] = 0xff;
        let colour = RasterColour { component: [0x12, 0x34, 0x56, 0x78] };

        for le in [false, true] {
            for did in [&[][..], &pass[..], &some[..], &none[..], &one_in_range[..]] {
                for store in [true, false] {
                    for (x0, x1) in [(0, N as i32 - 1), (0, 0), (3, 3), (4, 20), (N as i32 - 1, N as i32 - 1), (5, 8)] {
                        let (fd, sd) = (noise(1), noise(1));
                        let (fc, sc) = ([noise(2), noise(4)], [noise(2), noise(4)]);
                        let stencil = noise(3);
                        let row = |depth, colour0, colour1| GlPlaneRow {
                            x0: 100,
                            depth,
                            depth_le: le,
                            colour: [colour0, colour1],
                            colour_le: [le, !le],
                            stencil: &stencil,
                            store,
                            did,
                            id_value: 0,
                            id_mask: 0xff,
                        };
                        let fast = row(&fd, &fc[0], &fc[1]);
                        let slow = row(&sd, &sc[0], &sc[1]);

                        fast.fill_depth(100 + x0, 100 + x1, 0xabcdef);
                        for target in 0..3 {
                            fast.fill_rgba(target, 100 + x0, 100 + x1, &colour);
                        }
                        for x in x0..=x1 {
                            slow.store_depth(100 + x, 0, 0xabcdef);
                            for target in 0..3 {
                                slow.store_rgba(target, 100 + x, 0, &colour);
                            }
                        }
                        assert_eq!(bytes(&fd), bytes(&sd), "depth, le {le}, did {did:?}, store {store}, {x0}..={x1}");
                        assert_eq!(bytes(&fc[0]), bytes(&sc[0]), "colour 0, le {le}, did {did:?}, store {store}, {x0}..={x1}");
                        assert_eq!(bytes(&fc[1]), bytes(&sc[1]), "colour 1, le {le}, did {did:?}, store {store}, {x0}..={x1}");
                    }
                }
            }
        }
    }
}

impl Planes for GlPlaneRow<'_> {
    #[inline(always)]
    fn store_rgba(&self, target: ColourTarget, x: i32, _y: i32, colour: &RasterColour) {
        let i = (x - self.x0) as usize;
        let c = &colour.component;

        if target < ODSY_BUFFERS && self.store && self.allowed(i) {
            let word = (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3];

            word_store(self.colour[target as usize], i, self.colour_le[target as usize], word);
        }
    }

    #[inline(always)]
    fn load_rgba(&self, target: ColourTarget, x: i32, _y: i32) -> Option<RasterColour> {
        if target >= ODSY_BUFFERS {
            return None;
        }
        let word = word_load(self.colour[target as usize], (x - self.x0) as usize, self.colour_le[target as usize]);
        let mut colour = RasterColour::default();
        for i in 0..4u32 {
            colour.component[i as usize] = (word >> (24 - 8 * i)) & 0xff;
        }
        Some(colour)
    }

    #[inline(always)]
    fn load_depth(&self, x: i32, _y: i32) -> u32 {
        word_load(self.depth, (x - self.x0) as usize, self.depth_le)
    }

    #[inline(always)]
    fn store_depth(&self, x: i32, _y: i32, depth: u32) {
        let i = (x - self.x0) as usize;

        if self.store && self.allowed(i) {
            word_store(self.depth, i, self.depth_le, depth & ODSY_DEPTH24_MAX);
        }
    }

    #[inline(always)]
    fn load_stencil(&self, x: i32, _y: i32) -> u32 {
        self.stencil[(x - self.x0) as usize].get() as u32
    }

    /// As GlPlanes::store_stencil(): the pixel test alone, to_aux or not.
    #[inline(always)]
    fn store_stencil(&self, x: i32, _y: i32, stencil: u32) {
        let i = (x - self.x0) as usize;

        if self.allowed(i) {
            self.stencil[i].set((stencil & stencil::MAX) as u8);
        }
    }
}

impl GlPlanes<'_> {
    /// The stencil plane's bytes for pixels x0 to x0 + n - 1 of row y, as
    /// Rows::plane_load() reaches each: None unless all are in the plane.
    fn stencil_bytes(&self, y: i32, x0: i32, n: usize) -> Option<&[std::cell::Cell<u8>]> {
        let p = &self.state.stencil;

        if plane_bpp(p.format) != 1 || !p.contains(x0, y) || !p.contains(x0 + n as i32 - 1, y) {
            return None;
        }
        self.stencil.get(y as usize, x0 as usize, n)
    }
}

impl GlPlanes<'_> {
    /// Plane pixels x0 to x1 of row y as a row loop reaches them, with the
    /// plane bounds and clip rectangles tested once for the row: when the
    /// whole row is on the screen, inside one kept clip rectangle, and in
    /// the depth plane and both colour planes. Otherwise None, and the loop
    /// reaches each pixel through GlPlanes itself, with the same results.
    pub fn row(&self, y: i32, x0: i32, x1: i32) -> Option<GlPlaneRow<'_>> {
        let s = &self.state;
        let n = (x1 as i64 - x0 as i64 + 1) as usize;

        if !self.lend_rows || x0 < 0 || y < 0 || x1 >= ODSY_WIDTH as i32 || y >= ODSY_HEIGHT as i32 || x1 < x0 {
            return None;
        }
        if !s.clip_rect[..s.clip_kept as usize].iter().any(|r| y >= r.y0 && y <= r.y1 && x0 >= r.x0 && x1 <= r.x1) {
            return None;
        }
        // The row's stores reach the buffers the draw targets.
        if !s.to_aux {
            for b in (0..ODSY_BUFFERS as usize).filter(|b| self.row_targets >> b & 1 != 0) {
                touch_row(self.touched, b, y);
            }
        }
        let did = if s.id_compare & ODSY_ID_COMPARE_ENABLE != 0 {
            let at = y as usize * ODSY_WIDTH as usize + x0 as usize;
            s.did.get(at..at + n)?
        } else {
            &[]
        };
        Some(GlPlaneRow {
            x0,
            depth: self.depth.plane_words(&s.depth, y, x0, n)?,
            depth_le: s.depth.is_le(),
            colour: [
                self.colour[0].plane_words(&s.colour[0], y, x0, n)?,
                self.colour[1].plane_words(&s.colour[1], y, x0, n)?,
            ],
            colour_le: [s.colour[0].is_le(), s.colour[1].is_le()],
            stencil: self.stencil_bytes(y, x0, n)?,
            store: !s.to_aux,
            did,
            id_value: s.id_compare as u8,
            id_mask: (s.id_compare >> ODSY_ID_COMPARE_MASK_SHIFT) as u8,
        })
    }

    /// A colour-index fragment's packed index word.
    pub fn store_colour(&self, target: ColourTarget, x: i32, y: i32, colour: u32) {
        let s = &self.state;
        let mask = (s.write_mask >> write::COLOUR_SHIFT) & write::COLOUR_MASK;

        if mask == 0 || target >= ODSY_BUFFERS || !s.pixel_allowed(x as i64, y as i64) {
            return;
        }
        if s.to_aux {
            // A colour-index GL draw to the id-plane destination: an overlay
            // context's drawing (overlay::MODE_BASE). Buffer 1 is the
            // overlay pixel byte and takes the index's low eight bits; the id
            // byte is the X server's alone, so a GL write to buffer 0 is
            // reported and dropped.
            if target == ODSY_AUX_OVERLAY_BUFFER {
                if let Some(px) = self.overlay.get(y as usize, x as usize, 1) {
                    px[0].set((colour & 0xff) as u8);
                    touch_row(self.touched, TOUCHED_OVERLAY, y);
                }
            }
            return;
        }
        touch_row(self.touched, target as usize, y);
        self.colour[target as usize].plane_store(&s.colour[target as usize], x, y, colour & ODSY_INDEX_MASK);
    }

    /// Experimental policy; the board's scale sign and clamps are unmeasured.
    pub fn fog_factor(&self, window_z: f32) -> f32 {
        let w = &self.state.fog;
        let scale = if (w[2] & 0xffffff) != 0 {
            f64::from(odyssey_f32(((w[2] & 0xffffff) << 5).wrapping_add(96u32 << 23)).abs())
        } else {
            0.0
        };
        let mut t = (scale * window_z as f64 + sextract32(w[3], 0, 12) as f64 / 256.0)
            * 2.0f64.powi(((w[3] >> 12) & 7) as i32);
        let p = [
            sextract32(w[4] >> 12, 0, 12) as f64 / 256.0,
            sextract32(w[4], 0, 12) as f64 / 256.0,
            sextract32(w[5] >> 12, 0, 12) as f64 / 256.0,
            sextract32(w[5], 0, 12) as f64 / 256.0,
        ];

        t = t.clamp(0.0, 1.0);
        let u = 1.0 - t;
        let f = p[0] * u * u * u + 3.0 * p[1] * u * u * t + 3.0 * p[2] * u * t * t + p[3] * t * t * t;
        (if f >= 1.0 { 1.0 } else if f > 0.0 { f } else { 0.0 }) as f32
    }
}

impl Planes for GlPlanes<'_> {
    /// The shared renderer supplies all four already quantized channel bytes.
    fn store_rgba(&self, target: ColourTarget, x: i32, y: i32, colour: &RasterColour) {
        let s = &self.state;
        let c = &colour.component;

        if target < ODSY_BUFFERS && s.pixel_allowed(x as i64, y as i64) && !s.to_aux {
            let word = (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3];

            touch_row(self.touched, target as usize, y);
            self.colour[target as usize].plane_store(&s.colour[target as usize], x, y, word);
        }
    }

    /// Storage the plane does not reach reads as nothing, not as zero.
    fn load_rgba(&self, target: ColourTarget, x: i32, y: i32) -> Option<RasterColour> {
        if target >= ODSY_BUFFERS {
            return None;
        }
        let p = &self.state.colour[target as usize];
        let rows = &self.colour[target as usize];
        if !p.contains(x, y) || rows.get(y as usize, x as usize * 4, 4).is_none() {
            return None;
        }
        let word = rows.plane_load(p, x, y);
        let mut colour = RasterColour::default();
        for i in 0..4u32 {
            colour.component[i as usize] = (word >> (24 - 8 * i)) & 0xff;
        }
        Some(colour)
    }

    /// The depth plane (set up by Board::allocate()): the board's 24-bit depth word
    /// (depth_word.rs), 0 nearest,
    /// through the same pixel test as a colour write. It is a model plane
    /// because the board memory the real depth buffer occupies is not known.
    fn load_depth(&self, x: i32, y: i32) -> u32 {
        self.depth.plane_load(&self.state.depth, x, y)
    }

    fn store_depth(&self, x: i32, y: i32, depth: u32) {
        // The overlay has no depth; an overlay clear leaves the main depth.
        if !self.state.to_aux && self.state.pixel_allowed(x as i64, y as i64) {
            self.depth.plane_store(&self.state.depth, x, y, depth & ODSY_DEPTH24_MAX);
        }
    }

    /// The stencil plane (set up by Board::allocate()): 8 bits, through the
    /// same pixel test as a depth write. A model plane: where the board keeps
    /// its stencil planes is not known.
    fn load_stencil(&self, x: i32, y: i32) -> u32 {
        self.stencil.plane_load(&self.state.stencil, x, y)
    }

    fn store_stencil(&self, x: i32, y: i32, stencil: u32) {
        if self.state.pixel_allowed(x as i64, y as i64) {
            self.stencil.plane_store(&self.state.stencil, x, y, stencil & stencil::MAX);
        }
    }
}

/// The board's planes and pixel state as a raster job takes them: the
/// pixel state of the moment it was set up, and spans of the planes the
/// raster unit owns (raster_unit.rs).
#[derive(Clone, Copy)]
pub(super) struct GlPixelJob {
    pub state: GlPixelState<'static>,
    pub spans: PlaneSpans,
}

impl GlPixelJob {
    /// The planes, as GlPixels borrows them.
    ///
    /// # Safety
    /// As `Span::bytes`: only the raster unit, or a caller the raster unit
    /// has finished for, uses the planes meanwhile.
    pub unsafe fn pixels<'a>(&self) -> GlPixels<'a> {
        let s = &self.spans;
        let st = &self.state;
        // SAFETY: the caller's contract (this function's # Safety) is
        // `Span::bytes`'s: the planes belong to the raster unit or to a
        // caller it has finished for, so the spans' bytes are not aliased.
        unsafe {
            GlPixels {
                state: GlPixelState { did: s.did.bytes(), ..*st },
                colour: [
                    RowsMut::new(s.colour[0].bytes(), st.colour[0].stride as usize),
                    RowsMut::new(s.colour[1].bytes(), st.colour[1].stride as usize),
                ],
                depth: RowsMut::new(s.depth.bytes(), st.depth.stride as usize),
                stencil: RowsMut::new(s.stencil.bytes(), st.stencil.stride as usize),
                overlay: RowsMut::new(s.overlay.bytes(), ODSY_WIDTH as usize),
                touched: Default::default(),
                row_targets: 3,
                lend_rows: true,
            }
        }
    }
}

/// The texture tables a textured draw reads, copied when it is set up: a
/// later command may rewrite them before the raster unit draws. Only what
/// the chain uses is copied.
#[derive(Default)]
pub(super) struct TexelTables {
    slice_offset: Vec<u32>,
    slice_width: Vec<u32>,
    colour_table: Option<Box<[[u32; ODSY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2]>>,
    scale_bias: [u32; 8],
}

static NO_COLOUR_TABLE: [[u32; ODSY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2] =
    [[0; ODSY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2];

impl TexelTables {
    fn of(s: &Board, chain: Option<&TextureChain>) -> Self {
        let Some(chain) = chain else {
            return TexelTables::default();
        };
        // Only a volume reads the slice tables (BoardTexels::read_3d()).
        let volume = chain.volume.slices != 0 || chain.volumes.iter().any(|v| v.slices != 0);
        TexelTables {
            slice_offset: if volume { s.tex.slice_offset.to_vec() } else { Vec::new() },
            slice_width: if volume { s.tex.slice_width.to_vec() } else { Vec::new() },
            colour_table: chain.volume.colour_table.then(|| Box::new(s.tex.colour_table)),
            scale_bias: s.tex.scale_bias,
        }
    }

    /// The board's texels, as a raster job reads them.
    ///
    /// # Safety
    /// As `Span::bytes` for the object store.
    unsafe fn texels<'a>(&'a self, chain: &'a TextureChain, obj: Span) -> BoardTexels<'a> {
        let obj: &'a [u8] = unsafe { obj.bytes() };
        BoardTexels {
            chain,
            obj,
            obj_words: obj.len() as u64 / 4,
            slice_offset: &self.slice_offset,
            slice_width: &self.slice_width,
            colour_table: self.colour_table.as_deref().unwrap_or(&NO_COLOUR_TABLE),
            scale_bias: &self.scale_bias,
        }
    }
}

/// A GL draw's fields other than its borrows, for a raster job to own.
#[derive(Clone, Copy)]
struct GlDrawFields {
    context: u32,
    mode: u32,
    count: u32,
    back_colors: bool,
    point_size: f32,
    line_width: f32,
    point_smooth: bool,
    line_smooth: bool,
    polygon_smooth: bool,
    polygon_stipple_enabled: bool,
    polygon_modes: Option<[u32; 2]>,
    fill_by_triangle: bool,
    line_stipple_enabled: bool,
    line_stipple_repeat: u32,
    line_stipple_pattern: u32,
    line_phase: Option<u32>,
    line_phase_reset: bool,
}

impl GlDrawFields {
    fn of(d: &GlDraw) -> Self {
        GlDrawFields {
            context: d.context,
            mode: d.mode,
            count: d.count,
            back_colors: d.back_colors,
            point_size: d.point_size,
            line_width: d.line_width,
            point_smooth: d.point_smooth,
            line_smooth: d.line_smooth,
            polygon_smooth: d.polygon_smooth,
            polygon_stipple_enabled: d.polygon_stipple_enabled,
            polygon_modes: d.polygon_modes,
            fill_by_triangle: d.fill_by_triangle,
            line_stipple_enabled: d.line_stipple_enabled,
            line_stipple_repeat: d.line_stipple_repeat,
            line_stipple_pattern: d.line_stipple_pattern,
            line_phase: d.line_phase,
            line_phase_reset: d.line_phase_reset,
        }
    }

    fn draw<'a>(&self, vertices: &'a [f32], state: &'a GlDrawState, polygon_stipple: &'a [u32; 32]) -> GlDraw<'a> {
        let f = *self;
        GlDraw {
            context: f.context,
            mode: f.mode,
            count: f.count,
            vertices,
            back_colors: f.back_colors,
            state,
            point_size: f.point_size,
            line_width: f.line_width,
            point_smooth: f.point_smooth,
            line_smooth: f.line_smooth,
            polygon_smooth: f.polygon_smooth,
            polygon_stipple,
            polygon_stipple_enabled: f.polygon_stipple_enabled,
            polygon_modes: f.polygon_modes,
            fill_by_triangle: f.fill_by_triangle,
            line_stipple_enabled: f.line_stipple_enabled,
            line_stipple_repeat: f.line_stipple_repeat,
            line_stipple_pattern: f.line_stipple_pattern,
            line_phase: f.line_phase,
            line_phase_reset: f.line_phase_reset,
        }
    }
}

/// One GL primitive set up for the raster unit: everything Board::gl_draw()
/// read of the board, owned.
///
/// The raster unit hands a finished job back (RasterQueue::give_back) and
/// the front end fills it again for a later draw, keeping its vertex
/// buffer and texture chain: allocating them per draw on the front end and
/// freeing them on the raster thread had the two threads contending for
/// the allocator.
pub(super) struct GlDrawJob {
    rules: RasterRules,
    target: RasterTarget<'static>,
    image: GlTexImage<'static>,
    /// Kept across reuse; the draw reads it only when `textured`.
    chain: Option<Box<TextureChain>>,
    textured: bool,
    tables: TexelTables,
    state: GlDrawState,
    vertices: Vec<f32>,
    polygon_stipple: [u32; 32],
    fields: GlDrawFields,
    pixels: GlPixelJob,
}

impl GlDrawJob {
    /// Draw it; the result, and the line-stipple phase the context keeps.
    pub fn run(&self, ctx: &mut RasterCtx) -> (GlDrawResult, Option<u32>) {
        let mut t = self.target.clone();
        let chain = self.chain.as_deref().filter(|_| self.textured);
        t.texture = chain.map(|chain| chain.image(&self.image));
        let mut d = self.fields.draw(&self.vertices, &self.state, &self.polygon_stipple);
        if d.line_phase == Some(LINE_PHASE_ON_RASTER) {
            d.line_phase = Some(ctx.line_phase);
        }
        // SAFETY: the raster unit owns the planes while the job runs
        // (PlaneSpans).
        let mut pixels = unsafe { self.pixels.pixels() };
        let mut targets = [0; GL_MAX_TARGETS as usize];
        let n = self.state.draw_targets(&mut targets) as usize;
        pixels.row_targets = GlPixels::target_bits(&targets[..n]);
        // SAFETY: the raster unit owns the board memory the spans name while
        // the job runs (PlaneSpans), and the chain is the one the job holds.
        let texels = chain.map(|chain| unsafe { self.tables.texels(chain, self.pixels.spans.obj) });
        let texels: &dyn Texels = match &texels {
            Some(texels) => texels,
            None => &NoTexels,
        };
        let result = self.rules.draw(&t, &mut pixels, texels, &mut d);
        pixels.record(ctx);
        if let Some(phase) = d.line_phase {
            ctx.line_phase = phase;
        }
        crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: GL draw path: backend scalar result {} called=1", result);
        (result, d.line_phase)
    }
}

impl Board {
    /// The board state a GL draw's pixels pass under, without the id bytes.
    pub fn gl_pixel_state(&mut self) -> GlPixelState<'static> {
        GlPixelState {
            did: &[],
            id_compare: self.win.id_compare,
            clip_rect: self.win.clip_rect,
            clip_kept: self.win.clip_kept,
            write_mask: self.gl_write_mask,
            to_aux: self.gl_to_aux(),
            fog: self.gl_fog,
            colour: [
                self.planes.colour.plane(self.planes.colour_plane).expect("the colour plane"),
                self.planes.buffer1.plane(self.planes.buffer1_plane).expect("colour buffer 1's plane"),
            ],
            depth: self.planes.depth.plane(self.planes.depth_plane).expect("the depth plane"),
            stencil: self.planes.stencil.plane(self.planes.stencil_plane).expect("the stencil plane"),
        }
    }

    /// The board's planes as a GL draw borrows them, and, for a textured
    /// draw, its texels in chain, for a draw on the caller's thread. It
    /// waits for the raster unit's queued jobs first.
    pub fn gl_parts<'a>(&'a mut self, chain: Option<&'a TextureChain>) -> (GlPixels<'a>, Option<BoardTexels<'a>>) {
        self.raster_wait();
        let state = self.gl_pixel_state();
        let texels = chain.map(|chain| BoardTexels {
            chain,
            obj: &self.objects.mem.bytes,
            obj_words: self.objects.mem.size / 4,
            slice_offset: &self.tex.slice_offset,
            slice_width: &self.tex.slice_width,
            colour_table: &self.tex.colour_table,
            scale_bias: &self.tex.scale_bias,
        });
        let pixels = GlPixels {
            state: GlPixelState { did: &self.planes.did, ..state },
            colour: [self.planes.colour.rows(&state.colour[0]), self.planes.buffer1.rows(&state.colour[1])],
            depth: self.planes.depth.rows(&state.depth),
            stencil: self.planes.stencil.rows(&state.stencil),
            overlay: RowsMut::new(&mut self.planes.overlay, ODSY_WIDTH as usize),
            touched: Default::default(),
            row_targets: 3,
            lend_rows: true,
        };
        (pixels, texels)
    }
}

impl Board {
    /// How far past [0, 1] a fragment's depth may round and still be drawn,
    /// clamped: 16 units in the last place of a single near 1 (gl_cmd::DEPTH_SLACK
    /// in consts.rs). APPROXIMATION: the depth row and block 16 arrive as
    /// singles, and a library that maps the far plane to 0.99999999 of the range
    /// leaves less margin than their rounding takes. A board does its
    /// arithmetic in its own precision and the clip it applies at the far plane
    /// is UNKNOWN.
    ///
    /// The depth row over the block-16 row, in units of the client's [0, 1]
    /// depth (Board::gl_load_combined()). The shared context's model-view is
    /// the identity, so its eye coordinates are the object coordinates the
    /// combined matrix takes. Block 16 is the divisor only when all four of its
    /// words are finite and the fourth is not zero; otherwise the sign-only
    /// fallback divides by a constant.
    pub fn gl_operation_rules(&mut self) -> RasterRules {
        let mut rules = gl_rules().clone();
        let scale = 2.0 * gl_cmd::DEPTH_S * self.gl_depth_scale();
        let mut block16 = [0.0f64; 4];
        let mut divisor = true;

        rules.alpha_bits = if self.gl_index_mode() { 0 } else { 8 };
        rules.depth_row_given = true;
        for k in 0..4usize {
            rules.depth_row[k] = f64::from(odyssey_f32(self.gl_combined[8 + k])) / scale;
            block16[k] = f64::from(odyssey_f32(self.gl_block16[k]));
            divisor &= block16[k].is_finite();
        }
        if divisor && block16[3] != 0.0 {
            rules.depth_divisor_given = true;
            for k in 0..4usize {
                rules.depth_divisor[k] = block16[k] / block16[3];
            }
        }
        rules.depth_slack = gl_cmd::DEPTH_SLACK;
        rules
    }
}

impl Board {
    /// An opaque polygon stipple (0x043f bit 4 with shape 0, the GL stipple):
    /// clear bits take the 0x0c21 constant colour, as the 2D path's opaque
    /// patterns do (Board::pattern_background()). That the board paints the
    /// clear bits here as it does for the 2D quads is INFERRED: it is the only
    /// reading under which the X server's FillOpaqueStippled sequence draws an
    /// opaque stipple.
    pub fn gl_stipple_background(&mut self, t: &mut RasterTarget) {
        if (self.stream.pattern_shape & pattern::SHAPE_OPAQUE) == 0 {
            return;
        }
        t.stipple_opaque = true;
        if self.gl_index_mode() {
            t.stipple_background = gl_index_rgba(self.pattern_background(true));
        } else {
            let pixel = self.pattern_background(false);

            for k in 0..3u32 {
                t.stipple_background[k as usize] = ((pixel >> (24 - 8 * k)) & 0xff) as f32 / 255.0f32;
            }
            t.stipple_background[3] = 1.0;
        }
    }

    /// What the board's planes offer a GL draw (GlPlanes).
    pub fn gl_target(&mut self) -> RasterTarget<'static> {
        let mut t = RasterTarget { index: self.gl_index_mode(), ..Default::default() };

        self.gl_stipple_background(&mut t);
        t
    }
}

/// The board's texture storage as a textured GL draw reads it: the levels
/// Board::texture_setup() found in the object store, and the post-filter
/// state.
pub(super) struct BoardTexels<'a> {
    pub chain: &'a TextureChain,
    pub obj: &'a [u8],
    /// The object store's size in words.
    pub obj_words: u64,
    pub slice_offset: &'a [u32],
    pub slice_width: &'a [u32],
    pub colour_table: &'a [[u32; ODSY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2],
    pub scale_bias: &'a [u32; 8],
}

impl BoardTexels<'_> {
    /// Texel (x, y) of texture, packed RRGGBBAA.
    pub fn read(&self, texture: &Texture, x: u32, y: u32) -> Option<u32> {
        if x >= texture.width || y >= texture.height {
            return None;
        }
        // y * stride cannot overflow 64 bits.
        let word = y as u64 * texture.stride as u64 + texture.offset as u64 + x as u64;
        let end = word + 1;
        if end > texture.words as u64 || end > self.obj_words {
            return None;
        }
        let at = (word * 4) as usize;
        let rgba = u32::from_be_bytes(self.obj.get(at..at + 4)?.try_into().unwrap());
        Some(if texture.base_format == GL_ALPHA { (rgba >> 24) & 0xff } else { rgba })
    }

    /// A border texel of level `level` of the 2D pyramid.
    pub fn read_border_level(&self, level: u32, x: i32, y: i32) -> Option<u32> {
        let mut texture = *self.chain.storage.get(level as usize)?;

        if texture.height == 1 && !texture.border_rows {
            if y != 0
                || texture.offset == 0
                || (x != -1 && x != texture.width as i32)
            {
                return None;
            }
            texture.offset -= 1;
            texture.width += 2;
            return self.read(&texture, (x + 1) as u32, 0);
        }
        if x < -1
            || x > texture.width as i32
            || y < -1
            || y > texture.height as i32
            || (x >= 0 && x < texture.width as i32 && y >= 0 && y < texture.height as i32)
            || (texture.offset as u64) < texture.stride as u64 + 1
        {
            return None;
        }
        // Setup takes a bordered level only when its whole border was written.
        texture.offset -= texture.stride + 1;
        texture.width += 2;
        texture.height += 2;
        self.read(&texture, (x + 1) as u32, (y + 1) as u32)
    }

    /// Texel (x, y, z) of volume, its border included when border is set.
    pub fn read_3d(&self, volume: &TextureVolume, x: i32, y: i32, z: i32, border: bool) -> Option<u32> {
        if (border
            && (x < -1 || y < -1 || z < -1 || x > volume.width as i32 || y > volume.height as i32 || z > volume.depth as i32))
            || (!border
                && (x < 0 || y < 0 || z < 0 || x >= volume.width as i32 || y >= volume.height as i32 || z >= volume.depth as i32))
        {
            return None;
        }
        // Setup found and checked the object of every slice a read in these
        // bounds can reach (Board::texture_setup()), so the slice's place in
        // the object store is read from its table, not looked up per texel.
        let slice = if border { (z + 1) as u32 } else { z as u32 };
        if slice >= volume.slices {
            return None;
        }
        let at = (volume.first_slice + slice) as usize;
        let stride = self.slice_width[at];
        let width = if border { volume.width + 2 } else { volume.width };
        let height = if border { volume.height + 2 } else { volume.height };
        let px = if border { (x + 1) as u32 } else { x as u32 };
        let py = if border { (y + 1) as u32 } else { y as u32 };
        let offset = if border {
            (stride as u64 * 15 + 15).checked_add(self.slice_offset[at] as u64)? as u32
        } else {
            self.slice_offset[at]
        };
        let texture = Texture {
            offset,
            stride,
            width,
            height,
            base_format: volume.base_format,
            words: volume.words,
            border_rows: false,
        };
        self.read(&texture, px, py)
    }

    /// The texture colour table: each normalized component indexes it after
    /// filtering.
    pub fn colour_table(&self, alpha: bool, rgba: &mut [f64; 4]) {
        for (channel, c) in rgba.iter_mut().enumerate().take(3) {
            let index = (cmax(0.0, cmin(*c, 1.0)) * 255.0 + 0.5).floor() as u32;
            let plane = (channel == 2) as usize;
            let word = self.colour_table[plane][index as usize];
            let component = if channel == 1 { word & 0xfff } else { (word >> 12) & 0xfff };

            *c = component as f64 / 4095.0;
        }
        if alpha {
            let index = (cmax(0.0, cmin(rgba[3], 1.0)) * 255.0 + 0.5).floor() as u32;
            let word = self.colour_table[1][index as usize];

            rgba[3] = (word & 0xfff) as f64 / 4095.0;
        }
    }

    /// GL_SGIX_texture_scale_bias applies scale, then bias, then a clamp to
    /// [0, 1] to the filtered texel, before the texture colour table and the
    /// environment. That the board follows the extension's order and clamp is
    /// INFERRED: no source gives the board's arithmetic.
    pub fn scale_bias(&self, rgba: &mut [f64; 4]) {
        for c in 0..4usize {
            let scale = sextract32(self.scale_bias[c * 2], 0, tex::SCALE_BIAS_BITS) as f64;
            let bias = sextract32(self.scale_bias[c * 2 + 1], 0, tex::SCALE_BIAS_BITS) as f64;

            rgba[c] = (rgba[c] * scale + bias) / tex::SCALE_BIAS_ONE as f64;
            rgba[c] = cmin(cmax(rgba[c], 0.0), 1.0);
        }
    }
}

impl Texels for BoardTexels<'_> {
    fn texel(&self, level: Option<&GlTexLevel>, x: u32, y: u32) -> Option<u32> {
        match level {
            Some(l) if l.detail => (l.level == 0).then(|| self.read(&self.chain.detail_storage, x, y))?,
            Some(l) => self.read(self.chain.storage.get(l.level as usize)?, x, y),
            None => None,
        }
    }

    fn border_texel(&self, level: Option<&GlTexLevel>, x: i32, y: i32) -> Option<u32> {
        match level {
            Some(l) if !l.detail => self.read_border_level(l.level, x, y),
            _ => None,
        }
    }

    fn texel_3d(&self, level: Option<&GlTexLevel>, x: u32, y: u32, z: u32) -> Option<u32> {
        let volume = match level {
            Some(l) => self.chain.volumes.get(l.level as usize)?,
            None => &self.chain.volume,
        };
        self.read_3d(volume, x as i32, y as i32, z as i32, volume.border)
    }

    fn border_texel_3d(&self, level: Option<&GlTexLevel>, x: i32, y: i32, z: i32) -> Option<u32> {
        match level {
            None => self.read_3d(&self.chain.volume, x, y, z, true),
            Some(_) => None,
        }
    }

    fn post_filter(&self, rgba: &mut [f64; 4]) {
        let volume = &self.chain.volume;

        if volume.scale_bias {
            self.scale_bias(rgba);
        }
        if volume.colour_table {
            self.colour_table(volume.colour_table_alpha, rgba);
        }
    }

    fn filter_weight(&self, lambda: f64) -> Option<f64> {
        self.chain.detail_weight.at(lambda)
    }
}

/// A volume level's slices sit 2^(word 2) tiles of 256 texels apart, each
/// texel the bytes the library books for the level's format, in 16-byte
/// address units. The library places each slice's upload so and sends the
/// tile count's base-2 ceiling in word 2 (ESTABLISHED). That the board
/// derives its slice stride the same way is INFERRED: it is the only reading
/// under which an upload and a later sample agree, as are the 16-byte unit
/// and the 256-texel tile. For a 4-byte format this gives the spacing seen
/// three ways on the model: 0x80 apart for word 2 1, 0x400 for 4 and 0x1000
/// for 6.
pub(super) fn volume_slice_stride(level_word2: u32, format: TexelFormat) -> u64 {
    (256u64 * format.booked() as u64 / 16).wrapping_shl(level_word2)
}

/// The detail function table (0x3e40-0x3e4f): the GL library fills its 256
/// entries with the application's function at LOD -i/32, interpolated
/// linearly between the points and held beyond the end ones, as unsigned
/// 12-bit fractions in bits 23:12 (ESTABLISHED as the library's table
/// builder). Beyond
/// the table's ends the weight is the end entry's, as SGIS_detail_texture
/// holds the function past its end points. Between two entries the model
/// interpolates linearly, which gives back the application's function
/// wherever its points sit on the 1/32 grid. How the board itself indexes
/// and rounds the table between entries is UNKNOWN: APPROXIMATION.
impl TextureDetailWeight {
    /// The detail and sharpen weight at lambda, where the table gives one.
    pub fn at(&self, lambda: f64) -> Option<f64> {
        let last = (ODSY_TEX_DETAIL_FUNCTION_ENTRIES - 1) as usize;
        let weight = |i: usize| self.table[i] as f64 / 4096.0;

        if lambda.is_nan() {
            return None;
        }
        if lambda >= 0.0 {
            return Some(weight(0));
        }
        if lambda <= tex::DETAIL_FUNCTION_LAST_LOD {
            return Some(weight(last));
        }
        let x = -lambda * 32.0;
        let i = (x as usize).min(last - 1);
        let f = x - i as f64;
        Some(weight(i) + (weight(i + 1) - weight(i)) * f)
    }
}

/// Post-filter scale and bias (0x203b, where texture validation sends a
/// texture's words): eight words, a scale and a bias for each of R, G, B
/// and A in turn, each signed 3.12 in bits 15:0. The GL library clamps both
/// to -8 through 7.999755859375 and defaults them to 4096 (1.0) and 0
/// (ESTABLISHED as the library's conversion). Pairing scale before bias
/// within a channel is INFERRED from the order the library names them.
/// The library never sets bits 31:16, so a word that does is unknown state
/// and drops the draw. The reset value is the library's default, a model
/// choice.
impl Board {
    pub fn texture_scale_bias_known(&self) -> bool {
        self.tex.scale_bias[..tex::SCALE_BIAS_WORDS as usize]
            .iter()
            .all(|&w| (w >> tex::SCALE_BIAS_BITS) == 0)
    }

    pub fn texture_scale_bias_identity(&self) -> bool {
        for c in 0..4usize {
            if self.tex.scale_bias[c * 2] != tex::SCALE_BIAS_ONE || self.tex.scale_bias[c * 2 + 1] != 0 {
                return false;
            }
        }
        true
    }

}

/// The detail texture a sampler word 0 names under a detail magnification
/// filter. Bit 31 is the combine (1 modulate, 0 add) and bits 29:25 the
/// detail level as a positive count, at most 15 (ESTABLISHED as the library's
/// setter); bits 14:0 are the base level's sizes. Bits 24:20 and 19:15 are
/// the detail image's log2 width and height, and the image is level-table
/// entry 64 minus the level (ESTABLISHED from the library's detail branch,
/// and consistent with the captured 2x2 and 4x2 controls at levels -1 to -3,
/// entries 63 to 61, and SGI Quake's 128x128 detail image at the default
/// level -4, whose entry 60 holds a 128-texel row). The library writes bits
/// 24:15 only for a detail magnification filter and leaves them stale after,
/// so the caller reads them only while word 4 carries magnification code 2, 3
/// or 4. Bit 30 has no known meaning and refuses the word.
pub(super) fn texture_detail_sampler(sampler: u32) -> Option<TextureDetailSampler> {
    let level = (sampler >> 25) & 0x1f;

    if (sampler & 0x40000000) != 0 || level == 0 || level > 15 {
        return None;
    }
    Some(TextureDetailSampler {
        descriptor: ODSY_TEX_LEVELS - level,
        width: 1 << ((sampler >> 20) & 0x1f),
        height: 1 << ((sampler >> 15) & 0x1f),
        level: -(level as i32),
        mode: if (sampler & 0x80000000) != 0 { GL_MODULATE } else { GL_ADD },
    })
}

/// A detail function entry's weight, bits 23:12 as an unsigned 12-bit
/// fraction. The library never sets the other bits, so an entry that does
/// is unknown state and drops the draw.
pub(super) fn texture_detail_entry_weight(value: u32) -> Option<u16> {
    if (value & !0x00fff000) != 0 {
        return None;
    }
    Some((value >> 12) as u16)
}

impl Board {
    pub fn texture_detail_function_weight(&self) -> Option<TextureDetailWeight> {
        let table = &self.tex.detail_function;

        if self.tex.detail_function_valid != 0xffff {
            return None;
        }
        let mut weight = TextureDetailWeight::default();
        for (w, &e) in weight.table.iter_mut().zip(table.iter()) {
            *w = texture_detail_entry_weight(e)?;
        }
        Some(weight)
    }
}

impl Board {
    /// The detail image at level-table entry `descriptor`, held as a plain 2D
    /// level is: its row is its width in 16-texel units rounded up to even,
    /// and its entry's word 1 carries no bits outside the format and that
    /// stride (Board::texture_setup()).
    ///
    /// The entry's word 1 bits 31:30 and 25:20 give the image's own index and
    /// size code, which must be a table row; its components are the
    /// primary's, `primary`, since the board is sent only the primary's class
    /// (texture_level_format()). An index the table does not list refuses
    /// the draw.
    pub(super) fn texture_detail_backing_setup(
        &mut self,
        chain: &mut TextureChain,
        primary: u32,
        descriptor: u32,
        width: u32,
        height: u32,
    ) -> Result<(), Refusal> {
        let e = (descriptor * 3) as usize;
        let entry = [self.tex.levels[e], self.tex.levels[e + 1], self.tex.levels[e + 2]];
        let Some(base_format) = texture_level_format(entry[1], primary) else {
            return Err(Refusal("a detail level's format index or size code is not one the library emits"));
        };
        let base = entry[0] & 0x1fffffff;
        let stride_units = (width.div_ceil(16) + 1) & !1u32;

        if base == 0
            || (entry[1] & 0xfff) != stride_units
            || (entry[1] & 0x3c0ff000) != 0
            || entry[2] != 0
            || self.objects.count > ODSY_OBJECTS
        {
            return Err(Refusal("a detail level that is not a plain level"));
        }
        let Some(o) = self.object_find(base) else {
            return Err(Refusal("a detail level with no object"));
        };
        let end = (height.wrapping_sub(1) as u64)
            .wrapping_mul(self.objects.width[o] as u64)
            .wrapping_add(self.objects.offset[o] as u64)
            .wrapping_add(width as u64);
        let region_written = self.object_region_written(o, 0, 0, width, height);
        if self.objects.width[o] != stride_units * 16
            || self.objects.rows[o] < height
            || end > self.objects.used as u64
            || end > self.objects.mem.size / 4
            || !region_written
        {
            return Err(Refusal("a detail level whose object does not hold it written"));
        }
        chain.detail_storage = Texture {
            offset: self.objects.offset[o],
            stride: self.objects.width[o],
            width,
            height,
            base_format,
            words: self.objects.used,
            border_rows: false,
        };
        chain.detail_image = GlTexLevel {
            level: 0,
            width,
            height,
            depth: 1,
            base_format,
            internal_format: base_format,
            detail: true,
            read_texel: true,
            ..Default::default()
        };
        Ok(())
    }
}

/// The four 0x1013 words are 14-bit two's complement with 8 fraction bits:
/// the GL library writes trunc(x * 256) & 0x3fff (ESTABLISHED as the
/// library's conversion). It never sets bits 31:14, so a word that does is
/// unknown state and drops the draw.
pub(super) fn texture_lod_word(word: u32) -> i32 {
    sextract32(word, 0, tex::LOD_BITS)
}

/// Minimum and maximum LOD (words 2 and 3). The library clamps both to
/// +/-15 before writing them (ESTABLISHED); the field holds +/-32. What the
/// board does with a limit past 15 is UNKNOWN, as no stock client sends
/// one: the model clamps it the way the library does, an approximation.
pub(super) fn texture_lod_limit(word: u32) -> i32 {
    cmin(cmax(texture_lod_word(word), -(tex::LOD_LIMIT as i32)), tex::LOD_LIMIT as i32)
}

impl Board {
    /// LOD bias S and T (words 0 and 1). The library clamps each to -32 through
    /// 31.99609375, the field's whole range, so every value is one it can send
    /// (ESTABLISHED). The R bias never reaches a register. The model applies a
    /// bias as GL_SGIX_texture_lod_bias defines it, scaling that axis's
    /// derivatives by 2^bias before the LOD is computed: INFERRED, since the
    /// board's LOD arithmetic is not documented. A zero bias leaves the scale
    /// unset, so an unbiased draw keeps the exact unscaled path.
    pub fn texture_lod_bias(&self, image: &mut GlTexImage) {
        let bias_s = texture_lod_word(self.tex.lod[0]);
        let bias_t = texture_lod_word(self.tex.lod[1]);

        image.lod_scale_s = if bias_s != 0 { (bias_s as f64 / tex::LOD_ONE as f64).exp2() } else { 0.0 };
        image.lod_scale_t = if bias_t != 0 { (bias_t as f64 / tex::LOD_ONE as f64).exp2() } else { 0.0 };
    }
}

impl GlTexImage<'_> {
    /// Sampler words 1 and 2, bits 16:0, hold each axis's maximum clamp in
    /// whole texels of the base level: the library writes trunc(2^log2size *
    /// GL_TEXTURE_MAX_CLAMP_*_SGIX), so the default 1.0 gives the size itself
    /// (ESTABLISHED as the library's setter). A smaller value under GL_CLAMP
    /// (wrap 2) clamps that axis's coordinate to value / size, as
    /// GL_SGIX_texture_coordinate_clamp defines it, and the size comes from
    /// word 0's log2 field. That the board applies it only under GL_CLAMP, and
    /// at every level as the same fraction of the image, is INFERRED from the
    /// extension. Under other wraps a smaller value is ignored. A value above
    /// the size cannot come from the library and refuses the draw
    /// (Board::texture_sampler()). Limitation: the R axis's clamp (word 3) is
    /// not applied; a volume whose word 3 is below its depth stays
    /// unsupported (Board::texture_support()).
    pub(super) fn texture_max_clamp(&mut self, size: [u32; 3], max_clamp: [u32; 3], wrap: [Wrap; 3]) {
        self.max_clamp_s = 1.0;
        self.max_clamp_t = 1.0;
        self.max_clamp_r = 1.0;
        if max_clamp[0] < size[0] && wrap[0] == Wrap::Clamp {
            self.max_clamp = true;
            self.max_clamp_s = max_clamp[0] as f64 / size[0] as f64;
        }
        if max_clamp[1] < size[1] && wrap[1] == Wrap::Clamp {
            self.max_clamp = true;
            self.max_clamp_t = max_clamp[1] as f64 / size[1] as f64;
        }
    }
}

impl Board {
    /// Rebuild shared state from board words; no borrowed image survives a draw.
    pub fn texture_vertex(&mut self) {
        const ENVIRONMENTS: [u32; 5] = [GL_REPLACE, GL_REPLACE, GL_MODULATE, GL_DECAL, GL_BLEND];
        let mut strq: [u32; 4] = [self.tex.coord[0], self.tex.coord[1], self.tex.coord_r, self.tex.coord_q];
        let one_d = (self.tex.mode & (tex::MODE_1D | tex::MODE_2D)) == tex::MODE_1D;
        self.gl().cur.texture_2d = (self.tex.mode & (tex::MODE_1D | tex::MODE_2D)) != 0;
        self.gl().cur.texture_3d = (self.tex.mode & tex::MODE_3D) != 0;
        // Excluded environments are rejected by the board before rasterization.
        self.gl().cur.texture_env_mode = GL_REPLACE;
        if (self.tex.env[0] as usize) < ENVIRONMENTS.len() {
            self.gl().cur.texture_env_mode = ENVIRONMENTS[self.tex.env[0] as usize];
        }
        // Captured linear planes already include the eye-space transform.
        if (self.cmd.tag == gl_cmd::VERTEX_2I
            || self.cmd.tag == gl_cmd::VERTEX_2F
            || self.cmd.tag == gl_cmd::VERTEX_3F
            || self.cmd.tag == gl_cmd::VERTEX_3I
            || self.cmd.tag == gl_cmd::VERTEX_4F)
            && self.geom.linear_texgen != 0
        {
            let mut vertex = [
                f64::from(odyssey_f32(self.cmd.operand[0])),
                f64::from(odyssey_f32(self.cmd.operand[1])),
                if self.cmd.tag != gl_cmd::VERTEX_2I && self.cmd.tag != gl_cmd::VERTEX_2F { f64::from(odyssey_f32(self.cmd.operand[2])) } else { 0.0 },
                if self.cmd.tag == gl_cmd::VERTEX_4F { f64::from(odyssey_f32(self.cmd.operand[3])) } else { 1.0 },
            ];

            if self.cmd.tag == gl_cmd::VERTEX_2I || self.cmd.tag == gl_cmd::VERTEX_3I {
                vertex[0] = (self.cmd.operand[0] as i32) as f64;
                vertex[1] = (self.cmd.operand[1] as i32) as f64;
            }
            if self.cmd.tag == gl_cmd::VERTEX_3I {
                vertex[2] = (self.cmd.operand[2] as i32) as f64;
            }
            for axis in 0..4usize {
                let plane = if axis == 3 {
                    self.tex.linear_plane_q
                } else if axis == 2 {
                    self.tex.linear_plane_r
                } else {
                    self.tex.linear_plane[axis]
                };
                let mut value = 0.0f32;

                if (self.geom.linear_texgen & self.tex.linear_valid & (1u32 << axis)) == 0 {
                    continue;
                }
                for k in 0..4usize {
                    value = (value as f64 + f64::from(odyssey_f32(plane[k])) * vertex[k]) as f32;
                }
                strq[axis] = value.to_bits();
            }
        }
        let rows = self.gl_texture_matrix;
        let mut matrix = gl_transpose(&rows);
        // Input T may transform S, but transformed T cannot affect 1D sampling.
        if one_d {
            for i in 0..4usize {
                matrix[i * 4 + 1] = 0.0;
            }
        }
        self.gl().set_matrix(GL_TEXTURE, &matrix);
        self.gl_feed(op::TEXCOORD4F as u16, &strq);
    }
}

/// A draw's estimated raster work in pixels, from its vertices' window
/// positions (`primitive_cost`).
fn gl_draw_cost(d: &GlDraw) -> u32 {
    let points = d.vertices.chunks_exact(vtx::FLOATS).take(d.count as usize).map(|v| {
        let clip = [v[vtx::POSITION], v[vtx::POSITION + 1], v[vtx::POSITION + 2], v[vtx::POSITION + 3]];
        if !(clip[3] > 0.0) {
            return None;
        }
        let win = d.state.clip_to_window(&clip, (1i32 << gl_rules().subpixel_bits) as f32)?;
        Some((win[0] as f64, win[1] as f64))
    });
    primitive_cost(d.mode, points, d.point_size, d.line_width)
}

impl Board {
    /// The context emits one assembled primitive here; rasterize it into the
    /// planes. The context requires the checked result once per-face polygon
    /// modes are activated (Board::reset_registers): it relays
    /// RasterRules::draw()'s out-of-memory signal from GL_POLYGON's dynamic
    /// vertex assembly back to the context synchronously instead of dropping
    /// it. The native renderer does not exist here, so this is always the
    /// scalar draw.
    pub fn gl_draw(&mut self, d: &mut GlDraw) -> GlDrawResult {
        let mut t = self.gl_target();
        let rules = self.gl_operation_rules();
        let mut state: GlDrawState = *d.state;
        let mut image = GlTexImage::default();
        // A job the raster unit has finished with, if one is back.
        let mut spare = self.raster.take_draw();
        // Some 7 KB that only a textured draw uses, and Board::texture_setup()
        // clears before it fills any of it, so it is not built for every
        // primitive, and a reused job's is reused.
        let mut texture: Option<Box<TextureChain>> = spare.as_mut().and_then(|j| j.chain.take());
        let textured = d.state.texture_2d || d.state.texture_3d;

        if textured {
            let chain = texture.get_or_insert_with(Box::default);

            if !self.texture_setup(&mut image, chain) {
                if let Some(mut j) = spare {
                    j.chain = texture;
                    self.raster.give_back(j);
                }
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: GL draw path: backend scalar-skip result {} called=0", GL_DRAW_OK);
                return GL_DRAW_OK;
            }
        }
        if (state.texture_2d || state.texture_3d) && self.tex.env[0] == 4 {
            // The board words pack two normalized 12-bit components each.
            state.texture_env_color[0] = ((self.tex.env[1] >> 12) & 0xfff) as f32 / 4095.0f32;
            state.texture_env_color[1] = (self.tex.env[1] & 0xfff) as f32 / 4095.0f32;
            state.texture_env_color[2] = ((self.tex.env[2] >> 12) & 0xfff) as f32 / 4095.0f32;
            state.texture_env_color[3] = (self.tex.env[2] & 0xfff) as f32 / 4095.0f32;
        }
        if !self.gl_index_mode() {
            state.colour_mask = (self.gl_write_mask >> write::COLOUR_SHIFT) & write::COLOUR_MASK;
        }
        state.fog_color[0] = ((self.gl_fog[0] >> 12) & 0xfff) as f32 / 4096.0f32;
        state.fog_color[1] = (self.gl_fog[0] & 0xfff) as f32 / 4096.0f32;
        state.fog_color[2] = ((self.gl_fog[1] >> 12) & 0xfff) as f32 / 4096.0f32;
        state.fog_color[3] = (self.gl_fog[1] & 0xfff) as f32 / 4096.0f32;
        self.gl_report_refusals(true);
        t.texture = None;
        let tables = TexelTables::of(self, texture.as_deref().filter(|_| textured));
        let pixels = self.gl_pixel_job();
        let job = match spare {
            Some(mut j) => {
                j.rules = rules;
                j.target = t;
                j.image = image;
                j.chain = texture;
                j.textured = textured;
                j.tables = tables;
                j.state = state;
                j.vertices.clear();
                j.vertices.extend_from_slice(d.vertices);
                j.polygon_stipple = *d.polygon_stipple;
                j.fields = GlDrawFields::of(d);
                j.pixels = pixels;
                j
            }
            None => Box::new(GlDrawJob {
                rules,
                target: t,
                image,
                chain: texture,
                textured,
                tables,
                state,
                vertices: d.vertices.to_vec(),
                polygon_stipple: *d.polygon_stipple,
                fields: GlDrawFields::of(d),
                pixels,
            }),
        };
        let cost = gl_draw_cost(d);
        if !self.raster.threaded() {
            let (result, phase) = self.raster.run_now(|ctx| job.run(ctx));
            d.line_phase = phase;
            self.raster.give_back(job);
            return result;
        }
        // The raster unit draws it later, so the context hears no result:
        // every result but out-of-memory is only reported, and that one
        // needs a polygon of more vertices than the host can address. The
        // raster unit keeps the line-stipple phase the draw advances.
        self.raster.submit_draw(job, cost);
        if d.line_phase.is_some() {
            d.line_phase = Some(LINE_PHASE_ON_RASTER);
        }
        GL_DRAW_OK
    }

    /// The board's planes and pixel state as a raster job takes them.
    pub fn gl_pixel_job(&mut self) -> GlPixelJob {
        GlPixelJob { state: self.gl_pixel_state(), spans: self.plane_spans() }
    }

    /// glClear's raster op, and the pattern rectangle's runs: the raster
    /// unit runs ops in order with the draw state of now.
    pub fn gl_raster_ops(&mut self, ops: Vec<RasterOp>) {
        let t = self.gl_target();
        let rules = self.gl_operation_rules();
        let pixels = self.gl_pixel_job();

        if let Some(op) = ops.first() {
            self.gl_report_refusals(op.kind != RASTER_CLEAR || (op.clear_buffers & GL_COLOR_BUFFER_BIT) != 0);
        }
        let cost = ops.iter().fold(0u32, |c, op| c.saturating_add(rect_cost(op.x0, op.y0, op.x1, op.y1, op.kind == RASTER_CLEAR)));
        self.raster.submit_work(cost, move |ctx| {
            // SAFETY: the job runs before the front end next touches the
            // planes (PlaneSpans).
            let mut planes = unsafe { pixels.pixels() };
            planes.row_targets = ops.iter().fold(0, |bits, op| bits | GlPixels::target_bits(&op.targets[..op.ntargets.min(GL_MAX_TARGETS) as usize]));
            for op in &ops {
                rules.raster(&t, &mut planes, op);
            }
            planes.record(ctx);
        });
    }

    /// glClear's raster op.
    pub fn gl_raster(&mut self, op: &RasterOp) {
        self.gl_raster_ops(vec![*op]);
    }

    /// Run one command in the GL context, with the board as the sink for the
    /// draws it emits. The context runs outside the board for the call, so
    /// that a draw can borrow the rest of the board; a stand-in fills its
    /// field meanwhile. The draw path does not touch the context.
    pub fn gl_command(&mut self, cmd: &GlCmd) {
        let stand_in = self.gl_spare.take().expect("GL commands do not nest");
        let mut ctx = std::mem::replace(&mut self.gl_ctx, stand_in);

        ctx.command(self, cmd);
        self.gl_spare = Some(std::mem::replace(&mut self.gl_ctx, ctx));
    }

    /// The board's GL context. Not for the draw path: while a command runs
    /// the field holds a stand-in.
    pub fn gl(&mut self) -> &mut GlContext {
        assert!(self.gl_spare.is_some(), "the GL context is running a command");
        &mut self.gl_ctx
    }

    pub fn gl_ref(&self) -> &GlContext {
        assert!(self.gl_spare.is_some(), "the GL context is running a command");
        &self.gl_ctx
    }
}

impl GlSink for Board {
    fn draw(&mut self, d: &mut GlDraw) -> GlDrawResult {
        self.gl_draw(d)
    }
}

impl ClipRect {
    fn gl_clip_rect_eq(&self, b: &ClipRect) -> bool {
        self.x0 == b.x0 && self.y0 == b.y0 && self.x1 == b.x1 && self.y1 == b.y1
    }
}

impl GlDrawable {
    // Compares the two drawables field by field.
    pub fn drawable_eq(&self, b: &GlDrawable) -> bool {
        self.bound == b.bound
            && self.origin == b.origin
            && self.width == b.width
            && self.height == b.height
            && self.clip.n == b.clip.n
            && self.clip.empty == b.clip.empty
            // Only the rectangles the list counts: this runs before every
            // command, and a window rarely owns more than one or two of the
            // sixteen. Those past the count are not drawn against.
            && self.clip.r.iter().zip(b.clip.r.iter()).take(self.clip.n as usize).all(|(x, y)| x.gl_clip_rect_eq(y))
            && self.front == b.front
            && self.back == b.back
            && self.extra == b.extra
            && self.overlay == b.overlay
    }
}

impl Board {
    /// Point the shared context at the current window: its origin from 0x841
    /// and the kept 0x204b rectangles as the pixels it owns, in plane
    /// coordinates. Called before every command, so a change to either
    /// register takes effect at the next one. With no rectangle kept, one
    /// inverted rectangle keeps the shared rasterizer drawing nothing rather
    /// than logging an unbound window. The shared context's front and back
    /// targets carry buffers 0 and 1, the raster target callbacks' buffer
    /// numbers, and its draw buffers follow the 0x0424 drawing mask. The kernel
    /// has already turned the client's front and back into buffer numbers
    /// (reg::DRAW_MASK), so the shared names are slots here, not the
    /// client's roles: a client drawing its front after one swap draws the
    /// context's "back", buffer 1.
    ///
    /// The shared rasterizer adds window rows to the origin as they come; the
    /// client's combined matrix already counts them down from the window's top
    /// row, the one 0x841 names (Board::gl_load_combined()), so the origin,
    /// rectangles and id plane all stay in the registers' top-down plane
    /// coordinates.
    pub fn gl_bind(&mut self) {
        let buffers = self.draw_buffers();
        let draw: u32 = (if buffers & 1 != 0 { GL_DRAW_FRONT } else { 0 })
            | (if buffers & 2 != 0 { GL_DRAW_BACK } else { 0 });
        let n = cmax(self.win.clip_kept, 1);
        let origin = self.win.origin;
        let clip_kept = self.win.clip_kept;
        let clip_rect = self.win.clip_rect;

        // This runs before every command, but the drawable changes only with
        // the window registers, so it is built in place and handed over only
        // when it differs from the shared context's. gl_bind_drawable's other
        // fields keep their initial values, and its rectangles past the
        // count stay zero, so it is the drawable a fresh build would make.
        {
            let d = &mut self.gl_bind_drawable;

            d.origin[0] = origin[0];
            d.origin[1] = origin[1];
            if clip_kept != 0 {
                d.clip.r[..clip_kept as usize].copy_from_slice(&clip_rect[..clip_kept as usize]);
            } else {
                // An all-empty clip; ClipList.empty replaces this at rebase.
                d.clip.r[0] = ClipRect { x0: 1, y0: 1, x1: 0, y1: 0 };
            }
            if d.clip.n > n {
                for i in n as usize..d.clip.n as usize {
                    d.clip.r[i] = ClipRect::default();
                }
            }
            d.clip.n = n;
        }
        if !self.gl_bind_drawable.drawable_eq(&self.gl_ref().cur.drawable) {
            let d = self.gl_bind_drawable;
            self.gl().set_drawable(&d);
        }
        if self.gl_ref().cur.draw_buffers != draw {
            self.gl_command(&GlCmd::new(op::DRAW_BUFFERS as u16, std::slice::from_ref(&draw)));
        }
    }

    /// Decode one command into the shared context.
    pub fn gl_feed(&mut self, op: u16, args: &[u32]) {
        self.gl_bind();
        self.gl_command(&GlCmd::new(op, args));
    }

    // Board::gl_feed with the op as the shared layer's u32 enum.
    fn gl_feed32(&mut self, op: GlOp, args: &[u32]) {
        self.gl_feed(op as u16, args);
    }
}

/// 0x404 bits 10:9 to the shared layer's cull-face enum: GL_BACK 01,
/// GL_FRONT 10, GL_FRONT_AND_BACK 11. Returns UINT32_MAX for 00, which no
/// probe word has sent; the shared layer's own glCullFace validation then
/// rejects it, leaving the previous mode in place.
pub(super) fn cull_face_mode(code: u32) -> u32 {
    match code {
        1 => GL_BACK,
        2 => GL_FRONT,
        3 => GL_FRONT_AND_BACK,
        _ => u32::MAX,
    }
}

/// 0x404 bit 23 to the shared layer's glFrontFace sense, inverted -- see
/// the mode::FRONT_FACE_CW comment for why.
pub(super) fn front_face(pattern_mode: u32) -> u32 {
    if (pattern_mode & mode::FRONT_FACE_CW) != 0 { GL_CCW } else { GL_CW }
}

impl Board {
    /// Feed one raster-state register (polygon mode, line width and point
    /// size, line stipple and its enable). A context swap drops the outgoing
    /// primitive and only then re-pushes the owner's polygon mode, sizes and
    /// stipple; on the board no primitive is open by then. The model is
    /// different: it keeps a GL_POLYGON open across the switch, because it
    /// assembles a polygon whole and the board's way of continuing one after a
    /// switch is unknown (Board::cmd_dispatch()'s 0x0444 and 0x0850 cases).
    /// The shared layer would refuse the restored values as GL calls inside
    /// glBegin and keep what the other context left, so while a GL_POLYGON is
    /// held open its begin flag is set aside for the one call. APPROXIMATION:
    /// this follows the model's hold, not any board behaviour; for every other
    /// primitive the shared layer's rule stands. The hold is right only because
    /// the model saves no GL_POLYGON vertices to the shadow, so the kernel's
    /// restart replays nothing: if the model ever saves them, it must stop
    /// holding at the same time, or the polygon draws twice.
    pub fn gl_feed_raster_state(&mut self, op: u16, args: &[u32]) {
        let in_begin = self.gl_ref().in_begin;
        let held_polygon = self.prim.begin_open && self.stream.primitive == 9;

        if held_polygon {
            self.gl().in_begin = false;
        }
        self.gl_feed(op, args);
        self.gl().in_begin = in_begin;
    }

    /// Hand the GL state register 0x0404 carries to the shared context: bit 12,
    /// the shade model, bit 16, the depth test, bit 14, the alpha test, bit 15,
    /// the stencil test, bit 17, blending, bit 19, index logic op, bit 20,
    /// colour logic op, bit 8, GL_CULL_FACE, bits 10:9, the face it culls, and
    /// bit 23, front face (fed inverted; see the mode::FRONT_FACE_CW
    /// comment), and bit 3, float GL line stipple. The shared rasterizer fills
    /// a flat triangle from its provoking vertex, the last one for triangles
    /// and strips as GL specifies. Nothing in the stream says which vertex the
    /// board takes; the library never reorders or repeats colours to steer it,
    /// so GL's own rule is INFERRED. Called on every 0x0404, including the X
    /// server's pattern-mode writes, which also update the separate integer 2D
    /// interpreter.
    ///
    /// Bit 15, the stencil test, gates fragments through the model's stencil
    /// plane (mode::STENCIL_TEST).
    pub fn gl_apply_polygon_stipple(&mut self) {
        let cap = GL_POLYGON_STIPPLE;
        let enable = (self.stream.pattern_mode & gl_cmd::RECT_PATTERN_ENABLE) != 0 && (self.stream.pattern_shape & !pattern::SHAPE_OPAQUE) == 0;

        if self.prim.in_prim {
            return;
        }
        if enable {
            let mut rows = [0u32; 32];

            for i in 0..rows.len() {
                rows[i] = self.stream.pattern_buffer[31 - i];
            }
            if !self.gl().polygon_stipple(&rows) {
                return;
            }
        }
        self.gl_feed32(if enable { op::ENABLE } else { op::DISABLE }, &[cap]);
    }

    /// 0x0404 bit 17 (mode::BLEND) blends only into an RGBA drawable:
    /// OpenGL blends RGBA fragments alone (OpenGL 1.2.1, 4.1.6), and the X
    /// server sets the bit with factors ONE, ZERO for its opaque stipples,
    /// whose background is the 0x0c21 constant colour and needs no destination
    /// (pattern::SHAPE_OPAQUE). Fed again when the colour mode changes.
    pub fn gl_apply_blend(&mut self) {
        let blend = GL_BLEND;
        let on = (self.stream.pattern_mode & mode::BLEND) != 0 && !self.gl_index_mode();

        self.gl_feed32(if on { op::ENABLE } else { op::DISABLE }, &[blend]);
    }

    pub fn gl_apply_mode(&mut self) {
        let shade = if (self.stream.pattern_mode & mode::FLAT_SHADE) != 0 { GL_FLAT } else { GL_SMOOTH };
        let cull_mode = cull_face_mode((self.stream.pattern_mode >> mode::CULL_MODE_SHIFT) & mode::CULL_MODE_MASK);
        let front_face = front_face(self.stream.pattern_mode);
        let pm = self.stream.pattern_mode;
        let en = |bit: u32| if (pm & bit) != 0 { op::ENABLE } else { op::DISABLE };

        self.gl_feed32(op::SHADE_MODEL, &[shade]);
        // Bit assignments follow the library's fragment-state encoding.
        self.gl_feed32(en(mode::POINT_SMOOTH), &[GL_POINT_SMOOTH]);
        self.gl_feed32(en(mode::LINE_SMOOTH), &[GL_LINE_SMOOTH]);
        // The shared layer has no dither rule yet, so the first mode word that
        // enables it is reported and the board draws undithered. Once the
        // shared rule lands this feeds op::ENABLE/DISABLE of GL_DITHER.
        if (pm & mode::DITHER) != 0 && !self.diag.gl_dither_reported {
            self.diag.gl_dither_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: dither not modelled; drawing undithered");
        }
        self.gl_feed32(en(mode::DEPTH_TEST), &[GL_DEPTH_TEST]);
        self.gl_feed32(en(mode::ALPHA_TEST), &[GL_ALPHA_TEST]);
        self.gl_feed32(en(mode::STENCIL_TEST), &[GL_STENCIL_TEST]);
        self.gl_apply_blend();
        self.gl_feed32(en(mode::FOG), &[GL_FOG]);
        self.gl_feed32(en(mode::INDEX_LOGIC_OP), &[GL_INDEX_LOGIC_OP]);
        self.gl_feed32(en(mode::COLOR_LOGIC_OP), &[GL_COLOR_LOGIC_OP]);
        self.gl_feed32(en(mode::OFFSET_POINT), &[GL_POLYGON_OFFSET_POINT]);
        self.gl_feed32(en(mode::OFFSET_LINE), &[GL_POLYGON_OFFSET_LINE]);
        self.gl_feed32(en(mode::OFFSET_FILL), &[GL_POLYGON_OFFSET_FILL]);
        self.gl_feed32(en(mode::CULL_FACE), &[GL_CULL_FACE]);
        self.gl_feed_raster_state(en(mode::GL_LINE_STIPPLE) as u16, &[GL_LINE_STIPPLE]);
        self.gl_feed32(op::CULL_FACE, &[cull_mode]);
        self.gl_feed32(op::FRONT_FACE, &[front_face]);
        self.gl_apply_polygon_stipple();
    }

    pub fn gl_apply_line_stipple(&mut self) {
        // Word 1 remains the integer 2D path's 32-bit rotated pattern.
        let args = [(self.stream.line_stipple[2] & 0xff) + 1, self.stream.line_stipple[0] & 0xffff];

        self.gl_feed_raster_state(op::LINE_STIPPLE as u16, &args);
    }

    /// Push the board-shadowed clip box (reg::CLIP_BOX) to the shared
    /// context's scissor rectangle (see the macro's own comment for the
    /// encoding and the always-on rationale). min > max -- including the
    /// library's own 0x1000, 0x1000 empty encoding -- clips every fragment,
    /// matched here with a zero-area box rather than a negative width or
    /// height, which op::SCISSOR rejects.
    pub fn gl_apply_clip_box(&mut self) {
        let x_min = (self.gl_clip_box[0] >> clip::BOX_MIN_SHIFT) & clip::BOX_FIELD_MASK;
        let x_max = self.gl_clip_box[0] & clip::BOX_FIELD_MASK;
        let y_min = (self.gl_clip_box[1] >> clip::BOX_MIN_SHIFT) & clip::BOX_FIELD_MASK;
        let y_max = self.gl_clip_box[1] & clip::BOX_FIELD_MASK;
        let mut box_ = [x_min, y_min, 0, 0];

        if x_min <= x_max && y_min <= y_max {
            box_[2] = x_max - x_min + 1;
            box_[3] = y_max - y_min + 1;
        }
        self.gl_feed32(op::SCISSOR, &box_);
    }

    /// Feed eye-space clip plane i to the shared context as the object-space
    /// plane the current model-view gives it: p_obj[k] = sum_r p_eye[r]
    /// M[r][k], M row-major as gl_modelview holds it.
    pub fn gl_apply_clip_plane(&mut self, i: u32) {
        let mut args = [0u32; 5];
        let mut eye = [0.0f64; 4];
        let mut obj = [0.0f64; 4];

        args[0] = i;
        for r in 0..4usize {
            eye[r] = f64::from(odyssey_f32(self.geom.clip_eye[i as usize * 4 + r]));
        }
        for r in 0..4usize {
            for k in 0..4usize {
                obj[k] += eye[r] * f64::from(odyssey_f32(self.geom.modelview[r * 4 + k]));
            }
        }
        for k in 0..4usize {
            args[1 + k] = f32_bits(obj[k] as f32);
        }
        self.gl_feed32(op::CLIP_PLANE, &args);
    }

    /// Recompute which of the shared context's six clip-plane slots are
    /// tested, from 0x0405 bits 2:0's thermometer count n (see
    /// tex::MODE_CLIP_COUNT_MASK's comment): slots 0..n-1 enabled, the rest
    /// disabled. Called on every 0x0405 write; does not
    /// touch the loaded coefficients themselves, which a separate load
    /// (gl_cmd::CLIP_PLANE_LOAD) feeds.
    pub fn gl_apply_clip_planes(&mut self) {
        let bits = self.tex.mode & tex::MODE_CLIP_COUNT_MASK;
        let n = bits.count_ones();

        for i in 0..GL_CLIP_PLANES {
            let plane = GL_CLIP_PLANE0 + i;

            self.gl_feed32(if i < n { op::ENABLE } else { op::DISABLE }, &[plane]);
        }
    }
}

/// 0x0c21's 6-bit blend factor code to the shared layer's enum
/// (established): 0 GL_ZERO, 0x08 GL_ONE, 0x10 "the other colour"
/// (GL_DST_COLOR as a source factor, GL_SRC_COLOR as a destination factor
/// -- the board never needs both names at the same position, so one code
/// covers both), 0x18 GL_SRC_ALPHA, 0x20 GL_DST_ALPHA, 0x28
/// GL_CONSTANT_COLOR, 0x30 GL_CONSTANT_ALPHA, 0x38 the "one minus" of the
/// 0x10 pairing, 0x39 through 0x3c the "one minus" of 0x18/0x20/0x28/0x30
/// in turn, and 0x40, a source-only code, GL_SRC_ALPHA_SATURATE. Returns
/// UINT32_MAX for a code with no factor at the requested position.
pub(super) fn blend_factor(code: u32, dst: bool) -> u32 {
    match code {
        0x00 => 0, // GL_ZERO
        0x08 => 1, // GL_ONE
        0x10 => {
            if dst { GL_SRC_COLOR } else { GL_DST_COLOR }
        }
        0x18 => GL_SRC_ALPHA,
        0x20 => GL_DST_ALPHA,
        0x28 => GL_CONSTANT_COLOR,
        0x30 => GL_CONSTANT_ALPHA,
        0x38 => {
            if dst { GL_ONE_MINUS_SRC_COLOR } else { GL_ONE_MINUS_DST_COLOR }
        }
        0x39 => GL_ONE_MINUS_SRC_ALPHA,
        0x3a => GL_ONE_MINUS_DST_ALPHA,
        0x3b => GL_ONE_MINUS_CONSTANT_COLOR,
        0x3c => GL_ONE_MINUS_CONSTANT_ALPHA,
        0x40 => {
            if dst { u32::MAX } else { GL_SRC_ALPHA_SATURATE }
        }
        _ => u32::MAX,
    }
}

/// 0x041f's three 3-bit stencil-op fields (fail, depth-fail, depth-pass)
/// to the shared layer's enum: KEEP 0, ZERO 1, REPLACE 2, INCR 3, DECR 4,
/// INVERT 5. Returns UINT32_MAX for the two codes the board leaves unused.
pub(super) fn stencil_op(code: u32) -> u32 {
    match code {
        0 => GL_KEEP,
        1 => 0, // GL_ZERO
        2 => GL_REPLACE,
        3 => GL_INCR,
        4 => GL_DECR,
        5 => GL_INVERT,
        _ => u32::MAX,
    }
}

/// 0x0415's 2-bit face code to the shared layer's polygon-mode enum,
/// GL_POINT 0 to GL_FILL 2. Returns UINT32_MAX for the one code the board
/// leaves undefined, which the shared layer's own glPolygonMode validation
/// then rejects, leaving the previous mode in place.
pub(super) fn polygon_mode(code: u32) -> u32 {
    match code {
        0 => GL_POINT,
        1 => GL_LINE,
        2 => GL_FILL,
        _ => u32::MAX,
    }
}

/// 0x0c21 word 0 bits 15:13 to the shared layer's blend equation. Codes 3
/// and 4, the SGIX alpha min/max extension, have no shared equivalent
/// (gl_blend_equation_valid() accepts only the other six), so the
/// dispatch case reports and skips them rather than feeding a code the
/// shared layer would reject.
pub(super) static BLEND_EQUATION: [u32; 8] = [
    GL_FUNC_ADD,
    GL_MIN,
    GL_MAX,
    0,
    0,
    GL_LOGIC_OP,
    GL_FUNC_SUBTRACT,
    GL_FUNC_REVERSE_SUBTRACT,
];

impl Board {
    /// 0x0420 and 0x0452 to the shared context's comparison and depth mask.
    pub fn gl_apply_depth(&mut self) {
        let func = GL_NEVER_ENUM + (self.gl_depth_func & ODSY_DEPTH_FUNC_MASK);
        let mask = ((self.gl_write_mask & write::DEPTH) != 0) as u32;

        self.gl_feed32(op::DEPTH_FUNC, &[func]);
        self.gl_feed32(op::DEPTH_MASK, &[mask]);
    }
}

/// The lighting words matched by value (gl_cmd::NORMAL_3F and the rest).
pub(super) static GL_LIGHT_WORDS: [u32; 16] = [
    gl_cmd::NORMAL_3F,
    gl_cmd::MATERIAL_AMBIENT,
    gl_cmd::MATERIAL_DIFFUSE,
    gl_cmd::MATERIAL_SPECULAR,
    gl_cmd::MATERIAL_EMISSION,
    gl_cmd::BACK_AMBIENT,
    gl_cmd::BACK_DIFFUSE,
    gl_cmd::BACK_SPECULAR,
    gl_cmd::BACK_EMISSION,
    gl_cmd::SHININESS_SQRT,
    gl_cmd::BACK_SHININESS_SQRT,
    gl_cmd::SHININESS_DERIVED,
    gl_cmd::LIGHT_COLOUR_LOAD,
    gl_cmd::LIGHT_VEC_LOAD,
    gl_cmd::CLIP_PLANE_LOAD,
    prog::RUN_ONCE,
];

/// The texture-coordinate and generation fragments.
pub(super) static UCODE_TEXGEN: [u32; 14] = [
    0x033, 0x04d, 0x052, 0x057, 0x05c, 0x061, 0x066, 0x06b, 0x06c, 0x077, 0x07b, 0x07f, 0x091, 0x09f,
];

pub(super) fn prog_value(word: u32) -> u32 {
    (word & prog::LOW_MASK) | (((word >> prog::HIGH_SHIFT) & 1) << 10)
}

pub(super) fn is_program_word(word: u32) -> bool {
    let cls = word >> prog::CLASS_SHIFT;

    if (word & prog::SHAPE) != prog::TAG {
        return false;
    }
    if (prog::PROGRAM_HALFWORD..=prog::PROGRAM_HALFWORD_LAST).contains(&cls) {
        return true;
    }
    if (prog::REG_FIRST..=prog::REG_LAST).contains(&cls)
        && (word & gl_cmd::MATRIX_SELECT_SHAPE) != gl_cmd::MATRIX_SELECT_TAG
    {
        return true;
    }
    matches!(
        cls,
        prog::COLOUR_MODE
            | prog::COLOUR_MODE_B
            | prog::SET_UCODE
            | prog::PROGRAM_ADDR
            | prog::CONST_SELECT
            | prog::ENTRY_29
            | prog::ENTRY_2A
            | prog::ENTRY_2C
            | prog::LIGHT_VEC_SELECT
            | prog::LIGHT_COLOUR_SELECT
            | prog::LIST_INDEX
            | prog::COLOUR_MATERIAL
            | prog::CONST_STORE
    )
}

pub(super) fn ucode_is_texgen(a: u32) -> bool {
    UCODE_TEXGEN.contains(&a)
}
