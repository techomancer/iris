//! The colour, depth, stencil, fog and texture hooks the board gives the shared
//! rasterizer, and the state its GL draws are made under.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

// MIN/MAX: the second operand wins ties and NaNs.
fn cmin<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}
fn cmax<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// One level's texels in the object store: `width` by `height` words from
/// word `offset`, rows `stride` words apart.
#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(PartialEq, Debug))]
pub struct Texture {
    pub offset: u32,
    pub stride: u32,
    pub width: u32,
    pub height: u32,
    pub base_format: u32,
    pub words: u32,
    /// A 2D level's border rows are stored, so a one-row level is not 1D.
    pub border_rows: bool,
}

/// First and last detail function weights, and whether the table is flat.
#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(PartialEq, Debug))]
pub struct TextureDetailWeight {
    pub constant: bool,
    pub first: f64,
    pub last: f64,
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(test, derive(PartialEq, Debug))]
pub struct TextureVolume {
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
    pub detail_weight: TextureDetailWeight,
}

/// A GlTexPyramid's shape: its levels are the first `count` of
/// TextureChain::levels.
#[derive(Clone, Copy, Default)]
pub struct PyramidShape {
    pub count: u32,
    pub base_level: u32,
    pub max_level: u32,
    pub min_lod: f64,
    pub max_lod: f64,
}

/// Where a draw's texture lives (Board::texture_setup()).
#[derive(Clone, Default)]
pub struct TextureChain {
    /// The level-zero volume, and the post-filter and detail weight state.
    pub volume: TextureVolume,
    pub volumes: [TextureVolume; SGI_GL_TEX_LEVELS_MAX as usize],
    pub storage: [Texture; SGI_GL_TEX_LEVELS_MAX as usize],
    pub levels: [GlTexLevel; SGI_GL_TEX_LEVELS_MAX as usize],
    pub pyramid: Option<PyramidShape>,
    pub detail_storage: Texture,
    pub detail_image: GlTexLevel,
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
pub struct TextureDetailSampler {
    pub descriptor: u32,
    pub width: u32,
    pub height: u32,
    pub level: i32,
    pub mode: u32,
}

impl Board {
    /// Whether a GL fragment's colour is an index (ODSY_PROG_COLOUR_MODE).
    pub fn gl_index_mode(&mut self) -> bool {
        self.colour_mode[0] & ODSY_COLOUR_MODE_INDEX != 0
    }
}

/// A colour index carried through the shared rasterizer, which knows only
/// RGBA: bits 7:0 of the index ride in blue and bits 11:8 in green, so at
/// the rules' eight colour bits the stored colour's low twelve bits are the
/// index again (gl_store_colour()). Smooth shading then
/// interpolates the index as GL does, within one run of 256.
pub fn gl_index_rgba(index: u32, rgba: &mut [f32; 4]) {
    rgba[0] = 0.0;
    rgba[1] = ((index >> 8) & 0xf) as f32 / 255.0f32;
    rgba[2] = (index & 0xff) as f32 / 255.0f32;
    rgba[3] = 1.0;
}

/// A float index (glIndexf, glClearIndex): its integer part, wrapped to the
/// twelve index bits the way a write keeps only the buffer's bits.
pub fn float_index(bits: u32) -> u32 {
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
        if !self.gl_aux_reported {
            self.gl_aux_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: {} to the id-plane destination is not modeled; dropped", what);
        }
    }

    /// What GlPlanes declines, reported once, from the draw rather than from
    /// a pixel: the conditions are fixed for the whole draw, and GlPlanes
    /// then only stores. colour says whether the draw writes colour at
    /// all.
    pub fn gl_report_refusals(&mut self, colour: bool) {
        let index_mask = self.gl_write_mask & ODSY_WRITE_INDEX_MASK;
        let mask = (self.gl_write_mask >> ODSY_WRITE_COLOUR_SHIFT) & ODSY_WRITE_COLOUR_MASK;
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
        if index_mask != ODSY_WRITE_INDEX_MASK && !self.gl_index_mask_logged {
            self.gl_index_mask_logged = true;
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
pub struct GlPixelState<'a> {
    pub did: &'a [u8],
    pub id_compare: u32,
    pub clip_rect: [ClipRect; SGI_ODYSSEY_CLIP_RECTS as usize],
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
            && x < SGI_ODYSSEY_WIDTH as i64
            && y < SGI_ODYSSEY_HEIGHT as i64
            && id_compare_passes(self.did, self.id_compare, x, y)
            && self.clip_rect[..self.clip_kept as usize]
                .iter()
                .any(|r| x >= r.x0 as i64 && x <= r.x1 as i64 && y >= r.y0 as i64 && y <= r.y1 as i64)
    }
}

/// The board's planes as a GL draw borrows them: the two colour buffers,
/// the model depth and stencil planes and the overlay byte.
pub struct GlPixels<'a> {
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

impl PlaneStore for GlPixels<'_> {
    fn planes(&mut self) -> Box<dyn Planes + '_> {
        let [c0, c1] = &mut self.colour;

        Box::new(GlPlanes {
            state: self.state,
            colour: [c0.view(), c1.view()],
            depth: self.depth.view(),
            stencil: self.stencil.view(),
            overlay: self.overlay.view(),
            touched: &self.touched,
            row_targets: self.row_targets,
        })
    }
}

/// GlPixels as one thread of a draw stores to them. Colour index draws use
/// store_colour; RGBA draws store_rgba and load_rgba (Board::gl_target()).
/// Their once-only reports are made before the draw
/// (Board::gl_report_refusals()).
pub struct GlPlanes<'a> {
    state: GlPixelState<'a>,
    colour: [Rows<'a, u8>; ODSY_BUFFERS as usize],
    depth: Rows<'a, u8>,
    stencil: Rows<'a, u8>,
    overlay: Rows<'a, u8>,
    /// Each row a colour store (or a lent row, for the draw's
    /// `row_targets`) reached, by buffer: what the display resolves again.
    touched: &'a TouchedRows,
    row_targets: u32,
}

/// One row of GlPlanes as a row loop reaches it (Planes::row()): the depth
/// plane's and both colour buffers' words and the stencil plane's bytes for
/// the row's pixels, and of GlPixelState::pixel_allowed() only the id test,
/// which is per pixel; the plane bound and clip rectangles were tested for
/// the whole row. Its
/// methods take plane coordinates on that row and answer as GlPlanes would.
pub struct GlPlaneRow<'a> {
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
            self.stencil[i].set((stencil & ODSY_STENCIL_MAX) as u8);
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

impl Planes for GlPlanes<'_> {
    /// The whole row on the screen and inside one kept clip rectangle, and
    /// in the depth plane and both colour planes; otherwise None, and the
    /// row loop tests each pixel.
    fn row(&self, y: i32, x0: i32, x1: i32) -> Option<GlPlaneRow<'_>> {
        let s = &self.state;
        let n = (x1 as i64 - x0 as i64 + 1) as usize;

        if x0 < 0 || y < 0 || x1 >= SGI_ODYSSEY_WIDTH as i32 || y >= SGI_ODYSSEY_HEIGHT as i32 || x1 < x0 {
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
            let at = y as usize * SGI_ODYSSEY_WIDTH as usize + x0 as usize;
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

    fn store_colour(&self, target: ColourTarget, x: i32, y: i32, colour: u32) {
        let s = &self.state;
        let mask = (s.write_mask >> ODSY_WRITE_COLOUR_SHIFT) & ODSY_WRITE_COLOUR_MASK;

        if mask == 0 || target >= ODSY_BUFFERS || !s.pixel_allowed(x as i64, y as i64) {
            return;
        }
        if s.to_aux {
            // A colour-index GL draw to the id-plane destination: an overlay
            // context's drawing (ODSY_OVERLAY_MODE_BASE). Buffer 1 is the
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
            self.stencil.plane_store(&self.state.stencil, x, y, stencil & ODSY_STENCIL_MAX);
        }
    }

    /// Experimental policy; the board's scale sign and clamps are unmeasured.
    fn fog_factor(&self, window_z: f32, _eye_z: f32) -> f32 {
        let w = &self.state.fog;
        let scale = if (w[2] & 0xffffff) != 0 {
            odyssey_f32(((w[2] & 0xffffff) << 5).wrapping_add(96u32 << 23)).abs()
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

/// The board's planes and pixel state as a raster job takes them: the
/// pixel state of the moment it was set up, and spans of the planes the
/// raster unit owns (raster_unit.rs).
#[derive(Clone, Copy)]
pub struct GlPixelJob {
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
        unsafe {
            GlPixels {
                state: GlPixelState { did: s.did.bytes(), ..*st },
                colour: [
                    RowsMut::new(s.colour[0].bytes(), st.colour[0].stride as usize),
                    RowsMut::new(s.colour[1].bytes(), st.colour[1].stride as usize),
                ],
                depth: RowsMut::new(s.depth.bytes(), st.depth.stride as usize),
                stencil: RowsMut::new(s.stencil.bytes(), st.stencil.stride as usize),
                overlay: RowsMut::new(s.overlay.bytes(), SGI_ODYSSEY_WIDTH as usize),
                touched: Default::default(),
                row_targets: 3,
            }
        }
    }
}

/// The texture tables a textured draw reads, copied when it is set up: a
/// later command may rewrite them before the raster unit draws. Only what
/// the chain uses is copied.
#[derive(Default)]
pub struct TexelTables {
    slice_offset: Vec<u32>,
    slice_width: Vec<u32>,
    colour_table: Option<Box<[[u32; SGI_ODYSSEY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2]>>,
    scale_bias: [u32; 8],
}

static NO_COLOUR_TABLE: [[u32; SGI_ODYSSEY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2] =
    [[0; SGI_ODYSSEY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2];

impl TexelTables {
    fn of(s: &Board, chain: Option<&TextureChain>) -> Self {
        let Some(chain) = chain else {
            return TexelTables::default();
        };
        // Only a volume reads the slice tables (BoardTexels::read_3d()).
        let volume = chain.volume.slices != 0 || chain.volumes.iter().any(|v| v.slices != 0);
        TexelTables {
            slice_offset: if volume { s.tex_slice_offset.to_vec() } else { Vec::new() },
            slice_width: if volume { s.tex_slice_width.to_vec() } else { Vec::new() },
            colour_table: chain.volume.colour_table.then(|| Box::new(s.tex_colour_table)),
            scale_bias: s.tex_scale_bias,
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
pub struct GlDrawJob {
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
        let mut targets = [0; SGI_GL_MAX_TARGETS as usize];
        let n = self.state.draw_targets(&mut targets) as usize;
        pixels.row_targets = GlPixels::target_bits(&targets[..n]);
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
            id_compare: self.id_compare,
            clip_rect: self.clip_rect,
            clip_kept: self.clip_kept,
            write_mask: self.gl_write_mask,
            to_aux: self.gl_to_aux(),
            fog: self.gl_fog,
            colour: [
                self.mem.plane(self.colour_plane).expect("the colour plane"),
                self.buffer1_mem.plane(self.buffer1_plane).expect("colour buffer 1's plane"),
            ],
            depth: self.depth_mem.plane(self.depth_plane).expect("the depth plane"),
            stencil: self.stencil_mem.plane(self.stencil_plane).expect("the stencil plane"),
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
            obj: &self.obj_mem.bytes,
            obj_words: self.obj_mem.size / 4,
            slice_offset: &self.tex_slice_offset,
            slice_width: &self.tex_slice_width,
            colour_table: &self.tex_colour_table,
            scale_bias: &self.tex_scale_bias,
        });
        let pixels = GlPixels {
            state: GlPixelState { did: &self.did, ..state },
            colour: [self.mem.rows(&state.colour[0]), self.buffer1_mem.rows(&state.colour[1])],
            depth: self.depth_mem.rows(&state.depth),
            stencil: self.stencil_mem.rows(&state.stencil),
            overlay: RowsMut::new(&mut self.overlay, SGI_ODYSSEY_WIDTH as usize),
            touched: Default::default(),
            row_targets: 3,
        };
        (pixels, texels)
    }
}

impl Board {
    /// How far past [0, 1] a fragment's depth may round and still be drawn,
    /// clamped: 16 units in the last place of a single near 1 (ODSY_GL_DEPTH_SLACK
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
        let scale = 2.0 * ODSY_GL_DEPTH_S * self.gl_depth_scale();
        let mut block16 = [0.0f64; 4];
        let mut divisor = true;

        rules.alpha_bits = if self.gl_index_mode() { 0 } else { 8 };
        rules.depth_row_given = true;
        for k in 0..4usize {
            rules.depth_row[k] = odyssey_f32(self.gl_combined[8 + k]) / scale;
            block16[k] = odyssey_f32(self.gl_block16[k]);
            divisor &= block16[k].is_finite();
        }
        if divisor && block16[3] != 0.0 {
            rules.depth_divisor_given = true;
            for k in 0..4usize {
                rules.depth_divisor[k] = block16[k] / block16[3];
            }
        }
        rules.depth_slack = ODSY_GL_DEPTH_SLACK;
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
        if (self.pattern_shape & ODSY_PATTERN_SHAPE_OPAQUE) == 0 {
            return;
        }
        t.stipple_opaque = true;
        if self.gl_index_mode() {
            t.stipple_background_index = self.pattern_background(true);
            let idx = t.stipple_background_index;
            gl_index_rgba(idx, &mut t.stipple_background);
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
        let index = self.gl_index_mode();
        let mut t = RasterTarget {
            store_colour: index,
            store_rgba: !index,
            load_rgba: !index,
            stencil: true,
            fog_factor: true,
            mixed_frame: false,
            ..Default::default()
        };

        self.gl_stipple_background(&mut t);
        t
    }
}

/// The board's texture storage as a textured GL draw reads it: the levels
/// Board::texture_setup() found in the object store, and the post-filter
/// state.
pub struct BoardTexels<'a> {
    pub chain: &'a TextureChain,
    pub obj: &'a [u8],
    /// The object store's size in words.
    pub obj_words: u64,
    pub slice_offset: &'a [u32],
    pub slice_width: &'a [u32],
    pub colour_table: &'a [[u32; SGI_ODYSSEY_TEX_COLOUR_TABLE_ENTRIES as usize]; 2],
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
        Some(if texture.base_format == SGI_GL_ALPHA { (rgba >> 24) & 0xff } else { rgba })
    }

    /// A border texel of level `level` of the 2D pyramid.
    pub fn read_border_level(&self, level: u32, x: i32, y: i32) -> Option<u32> {
        let mut texture = *self.chain.storage.get(level as usize)?;

        if texture.height == 1 && !texture.border_rows {
            if (texture.width != 2 && texture.width != 4 && texture.width != 8 && texture.width != 64)
                || y != 0
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
        // Setup validates the complete captured 4-by-4 border region; for a
        // larger bordered image it checks only that the rows exist.
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
            let scale = sextract32(self.scale_bias[c * 2], 0, ODSY_TEX_SCALE_BIAS_BITS) as f64;
            let bias = sextract32(self.scale_bias[c * 2 + 1], 0, ODSY_TEX_SCALE_BIAS_BITS) as f64;

            rgba[c] = (rgba[c] * scale + bias) / ODSY_TEX_SCALE_BIAS_ONE as f64;
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
        self.chain.volume.detail_weight.at(lambda)
    }
}

/// A volume level's slices sit 64 << word 2 address units apart. OBSERVED
/// three ways on the model: the captured 8x8x8 levels (word 2 1, 0x80
/// apart), the captured stored 2x2x2 (word 2 4, 0x400 apart) and a
/// 128x128x64 RGBA volume (word 2 6, 64 slice objects 0x1000 apart). That
/// word 2 is the log2 of the slice allocation is INFERRED.
pub fn volume_slice_stride(level_word2: u32) -> u64 {
    64u64.wrapping_shl(level_word2)
}

/// The stock driver's two-point function (-4, 1/4), (0, 1/2) fills entry 0
/// with 1/2, falls by 1/512 per entry to 1/4 at entry 128 and holds 1/4
/// through entry 255. That is consistent with entry i serving LOD -i/32, so
/// the table runs out at LOD -255/32. SGIS_detail_texture fixes the weight
/// at and beyond each end point, and those are the only weights used here.
/// How the board indexes and interpolates the table at LODs in between is
/// not known, so a nonconstant table refuses them.
impl TextureDetailWeight {
    /// The detail and sharpen weight at lambda, where the table gives one.
    pub fn at(&self, lambda: f64) -> Option<f64> {
        if self.constant || lambda <= ODSY_TEX_DETAIL_FUNCTION_LAST_LOD {
            return Some(self.last);
        }
        if lambda >= 0.0 {
            return Some(self.first);
        }
        None
    }
}

/// Post-filter scale and bias (0x203b, where texture validation sends a
/// texture's words): eight words, a scale and a bias for each of R, G, B
/// and A in turn, each signed 3.12 in bits 15:0. The GL library clamps both
/// to -8 through 7.999755859375 and defaults them to 4096 (1.0) and 0
/// (ESTABLISHED as the library's conversion). Pairing scale before bias
/// within a channel is INFERRED from the order the datasheet names them.
/// The library never sets bits 31:16, so a word that does is unknown state
/// and drops the draw. The reset value is the library's default, a model
/// choice.
impl Board {
    pub fn texture_scale_bias_known(&self) -> bool {
        self.tex_scale_bias[..ODSY_TEX_SCALE_BIAS_WORDS as usize]
            .iter()
            .all(|&w| (w >> ODSY_TEX_SCALE_BIAS_BITS) == 0)
    }

    pub fn texture_scale_bias_identity(&self) -> bool {
        for c in 0..4usize {
            if self.tex_scale_bias[c * 2] != ODSY_TEX_SCALE_BIAS_ONE || self.tex_scale_bias[c * 2 + 1] != 0 {
                return false;
            }
        }
        true
    }

}

pub fn texture_detail_sampler(sampler: u32) -> Option<TextureDetailSampler> {
    let mut detail = TextureDetailSampler { descriptor: 62, width: 2, height: 2, level: -2, mode: SGI_GL_ADD };
    match sampler {
        0x02108000 => {
            detail.descriptor = 63;
            detail.level = -1;
            Some(detail)
        }
        0x04108000 => Some(detail),
        0x06108000 => {
            detail.descriptor = 61;
            detail.level = -3;
            Some(detail)
        }
        0x04208000 => {
            detail.width = 4;
            Some(detail)
        }
        0x04110000 => {
            detail.height = 4;
            Some(detail)
        }
        0x84108000 => {
            detail.mode = SGI_GL_MODULATE;
            Some(detail)
        }
        _ => None,
    }
}

pub fn texture_detail_entry_weight(value: u32) -> Option<f64> {
    match value {
        0x00000000 | 0x00400000 | 0x00555000 | 0x00800000 | 0x00fff000 => Some(value as f64 / 16777216.0),
        _ => None,
    }
}

impl Board {
    pub fn texture_detail_function_weight(&self) -> Option<TextureDetailWeight> {
        let table = &self.tex_detail_function;
        let last = (SGI_ODYSSEY_TEX_DETAIL_FUNCTION_ENTRIES - 1) as usize;

        if self.tex_detail_function_valid != 0xffff {
            return None;
        }
        Some(TextureDetailWeight {
            constant: table[1..=last].iter().all(|&e| e == table[0]),
            first: texture_detail_entry_weight(table[0])?,
            last: texture_detail_entry_weight(table[last])?,
        })
    }
}

/// Sampler word 4 bits 7:5 are the minification code (0 nearest, 1 linear,
/// 2 to 5 the four mipmap modes in GL order) and bits 4:1 the
/// magnification code (2 linear detail, 3 detail alpha, 4 detail colour).
/// With a detail magnification filter the GL library sends GL_NEAREST as
/// code 2 and GL_LINEAR as code 5 (ESTABLISHED as the library's setter), so
/// a detail word carries code 2 or 5. The model samples each code as the
/// mipmap mode it names: INFERRED; what the board does with code 2 or 5
/// under a detail filter is UNKNOWN.
/// Word 4 bit 17: set for GL_LINEAR_MIPMAP_LINEAR and for GL_PHASE_SGIX,
/// clear otherwise (ESTABLISHED as the library's setter). Its effect on the
/// board is UNKNOWN. With minification code 5 the model samples as that
/// code says and ignores the bit; with any other code it can only mean
/// GL_PHASE_SGIX, which the model does not implement, so the draw drops.
pub fn texture_detail_filter(word: u32) -> u32 {
    let min = word & ODSY_TEX_MIN_MASK;

    if min != ODSY_TEX_MIN_NEAREST_MIPMAP_NEAREST && min != ODSY_TEX_MIN_LINEAR_MIPMAP_LINEAR {
        return 0;
    }
    match word & !ODSY_TEX_MIN_MASK {
        0x13000004 => SGI_GL_LINEAR_DETAIL_SGIS,
        0x13000006 => SGI_GL_LINEAR_DETAIL_ALPHA_SGIS,
        0x13000008 => SGI_GL_LINEAR_DETAIL_COLOR_SGIS,
        _ => 0,
    }
}

impl Board {
    pub fn texture_detail_backing_setup(&mut self, chain: &mut TextureChain, descriptor: u32, width: u32, height: u32, base_format: u32) -> bool {
        let e = (descriptor * 3) as usize;
        let entry = [self.tex_levels[e], self.tex_levels[e + 1], self.tex_levels[e + 2]];
        let base = entry[0] & 0x1fffffff;

        if base == 0 || entry[1] != 0x41100002 || entry[2] != 0 || self.obj_count > SGI_ODYSSEY_OBJECTS {
            return false;
        }
        let object = self.object_find(base);
        if object < 0 {
            return false;
        }
        let o = object as usize;
        let end = (height.wrapping_sub(1) as u64)
            .wrapping_mul(self.obj_width[o] as u64)
            .wrapping_add(self.obj_offset[o] as u64)
            .wrapping_add(width as u64);
        let region_written = self.object_region_written(object, 0, 0, width, height);
        if self.obj_width[o] != 32
            || self.obj_rows[o] < height
            || end > self.obj_used as u64
            || end > self.obj_mem.size / 4
            || !region_written
        {
            return false;
        }
        chain.detail_storage = Texture {
            offset: self.obj_offset[o],
            stride: self.obj_width[o],
            width,
            height,
            base_format,
            words: self.obj_used,
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
        true
    }
}

/// The four 0x1013 words are 14-bit two's complement with 8 fraction bits:
/// the GL library writes trunc(x * 256) & 0x3fff (ESTABLISHED as the
/// library's conversion). It never sets bits 31:14, so a word that does is
/// unknown state and drops the draw.
pub fn texture_lod_word(word: u32) -> i32 {
    sextract32(word, 0, ODSY_TEX_LOD_BITS)
}

/// Minimum and maximum LOD (words 2 and 3). The library clamps both to
/// +/-15 before writing them (ESTABLISHED); the field holds +/-32. What the
/// board does with a limit past 15 is UNKNOWN, as no stock client sends
/// one: the model clamps it the way the library does, an approximation.
pub fn texture_lod_limit(word: u32) -> i32 {
    cmin(cmax(texture_lod_word(word), -(ODSY_TEX_LOD_LIMIT as i32)), ODSY_TEX_LOD_LIMIT as i32)
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
        let bias_s = texture_lod_word(self.tex_lod[0]);
        let bias_t = texture_lod_word(self.tex_lod[1]);

        image.lod_scale_s = if bias_s != 0 { (bias_s as f64 / ODSY_TEX_LOD_ONE as f64).exp2() } else { 0.0 };
        image.lod_scale_t = if bias_t != 0 { (bias_t as f64 / ODSY_TEX_LOD_ONE as f64).exp2() } else { 0.0 };
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
    /// the size cannot come from the library and is left to fail the size
    /// check. Limitation: the R axis's clamp (word 3) is not applied; a volume
    /// whose word 3 is below its depth stays unsupported.
    pub fn texture_max_clamp(&mut self, sampler_size: u32, ws: u32, wt: u32, width: &mut u32, height: &mut u32) {
        let span_s = 1u32 << ((sampler_size >> 10) & 31);
        let span_t = 1u32 << ((sampler_size >> 5) & 31);

        self.max_clamp_s = 1.0;
        self.max_clamp_t = 1.0;
        self.max_clamp_r = 1.0;
        if *width < span_s {
            if ws == 2 {
                self.max_clamp = true;
                self.max_clamp_s = *width as f64 / span_s as f64;
            }
            *width = span_s;
        }
        if *height < span_t {
            if wt == 2 {
                self.max_clamp = true;
                self.max_clamp_t = *height as f64 / span_t as f64;
            }
            *height = span_t;
        }
    }
}

impl Board {
    pub fn texture_setup(&mut self, image: &mut GlTexImage<'static>, chain: &mut TextureChain) -> bool {
        let width_word = self.tex_sampler[1];
        let height_word = self.tex_sampler[2];
        let mut width = width_word;
        let mut height = height_word;
        let depth = self.tex_sampler[3];
        let phase = (self.tex_sampler[4] & ODSY_TEX_SAMPLER_BIT17) != 0
            && (self.tex_sampler[4] & ODSY_TEX_MIN_MASK) != ODSY_TEX_MIN_LINEAR_MIPMAP_LINEAR;
        let word = self.tex_sampler[4] & !ODSY_TEX_SAMPLER_BIT17;
        let components = word & 0x0f800000;
        let mut sampler_size = self.tex_sampler[0];
        let detail = texture_detail_sampler(sampler_size);
        let detail_identity = detail.is_some();
        let detail_sampler = detail.unwrap_or_default();
        let detail_weight = if detail_identity { self.texture_detail_function_weight() } else { None };
        let detail_filter = if detail_identity { texture_detail_filter(word) } else { 0 };
        let tex_shape = self.tex_mode & (ODSY_TEX_MODE_1D | ODSY_TEX_MODE_2D | ODSY_TEX_MODE_3D);
        let three_d = (self.tex_mode & ODSY_TEX_MODE_3D) != 0;
        let three_d_shape = tex_shape == ODSY_TEX_MODE_3D
            && width >= 2
            && height >= 2
            && depth >= 2
            && (width as u64).is_power_of_two()
            && (height as u64).is_power_of_two()
            && (depth as u64).is_power_of_two()
            && self.tex_level0 == 0;
        let ws = (word >> 12) & 3;
        let wt = (word >> 10) & 3;
        let clamp = ws == 2 || wt == 2;
        let border = ws == 3 || wt == 3;
        let one_d_shape = tex_shape == ODSY_TEX_MODE_1D && height == 1 && depth == 1 && self.tex_level0 == 0;
        let two_d_shape =
            tex_shape == ODSY_TEX_MODE_2D && (width == 2 || width == 4) && height == 2 && depth == 1 && self.tex_level0 == 0;
        let stored_border = ((one_d_shape
            && (width == 2 || width == 4 || width == 8 || width == 64)
            && (word == 0x03002000 || word == 0x03002022))
            || (two_d_shape && (word == 0x03002800 || word == 0x03002822))
            || (three_d_shape && (word == 0x03002a00 || word == 0x03002a22)))
            && (self.tex_sampler[5] & 0xff000000) == 0
            && (self.tex_sampler[6] & 0xff000000) == 0x01000000;
        let three_d_stored = stored_border && three_d_shape;
        let three_d_clamp = three_d_shape && (three_d_stored || word == 0x13002a00 || word == 0x13002a22);
        let one_d_clamp = one_d_shape && (stored_border || word == 0x13002000 || word == 0x13002022);
        let two_d_clamp = two_d_shape && (stored_border || word == 0x13002800 || word == 0x13002822);
        let captured_clamp = one_d_clamp || two_d_clamp || three_d_clamp;
        let two_d_stored = stored_border && two_d_shape;
        let filter = word & 0xe2;
        let min = filter & 0xe0;
        // A two-dimensional image of any size with its one-texel border stored
        // around it, under any filter: word 4's bit 28 clear and word 6's top
        // byte 1, the marks the captured stored-border uploads carry. Seen from
        // a 512x512 GL_CLAMP linear-mipmap image whose ten levels each sit 16
        // texels in from their allocation's corner, with the border at row and
        // column 15. The captured small shapes keep their own checks.
        let general_border = !stored_border
            && !detail_identity
            && !three_d
            && tex_shape == ODSY_TEX_MODE_2D
            && ws != 3
            && wt != 3
            && word == (components | (ws << 12) | (wt << 10) | filter)
            && (self.tex_sampler[5] & 0xff000000) == 0
            && (self.tex_sampler[6] & 0xff000000) == 0x01000000;
        let bordered = stored_border || general_border;
        // GL_CLAMP (wrap field 2) on both axes outside the captured shapes,
        // under any filter. The board's sampling is assumed to be the OpenGL
        // 1.2.1 rule (section 3.8.4), which the shared sampler implements:
        // linear taps past the edge take the border colour, words 5 and 6.
        // GL_CLAMP on one axis only stays refused; no client has been seen to
        // send it.
        let general_clamp = ws == 2 && wt == 2 && !captured_clamp;
        let colour_table_l8 = word == 0x11500001 || word == 0x11500003 || word == 0x11600001;
        let colour_table_rgba8_rgb = word == 0x13500001;
        let colour_table_rgba8_rgba = word == 0x13600001;
        let colour_table = colour_table_l8 || colour_table_rgba8_rgb || colour_table_rgba8_rgba;
        let colour_table_disabled = word == 0x11500000;
        let first = self.tex_level0 & 0x3f;
        let count: u32;
        let terminal: u32;
        let lo = texture_lod_limit(self.tex_lod[2]);
        let hi = texture_lod_limit(self.tex_lod[3]);
        const MIN_FILTERS: [u32; 6] = [
            SGI_GL_NEAREST,
            SGI_GL_LINEAR,
            SGI_GL_NEAREST_MIPMAP_NEAREST,
            SGI_GL_LINEAR_MIPMAP_NEAREST,
            SGI_GL_NEAREST_MIPMAP_LINEAR,
            SGI_GL_LINEAR_MIPMAP_LINEAR,
        ];
        'unsupported: {
            if detail_identity {
                if detail_filter == 0
                    || detail_weight.is_none()
                    || self.tex_level0 != 0
                    || width != 1
                    || height != 1
                    || depth != 1
                    || tex_shape != ODSY_TEX_MODE_2D
                {
                    break 'unsupported;
                }
                sampler_size = 0x08000000;
            }
            if border {
                width &= 0xffff;
                height &= 0xffff;
            } else {
                // An axis with wrap field 1 (clamp to edge) carries 0x4 in bits
                // 23:16 of its size word, as a Performer-style client sends.
                // Other axes carry none. INFERRED from that pairing, axis by
                // axis, the way clamp to border's axes carry 0xfc.
                if (width_word >> 16) == (if ws == 1 { 0x4u32 } else { 0 }) {
                    width &= 0xffff;
                }
                if (height_word >> 16) == (if wt == 1 { 0x4u32 } else { 0 }) {
                    height &= 0xffff;
                }
            }
            if !detail_identity {
                image.texture_max_clamp(sampler_size, ws, wt, &mut width, &mut height);
            }
            if self.gl_index_mode() {
                break 'unsupported;
            }
            match components {
                0x01000000 => image.base_format = SGI_GL_LUMINANCE,
                0x01800000 => image.base_format = SGI_GL_LUMINANCE_ALPHA,
                0x00800000 => image.base_format = SGI_GL_ALPHA,
                0x02000000 => image.base_format = SGI_GL_INTENSITY,
                0x02800000 => image.base_format = SGI_GL_RGB,
                0x03000000 => image.base_format = SGI_GL_RGBA,
                _ => break 'unsupported,
            }
            if detail_identity && image.base_format != SGI_GL_RGBA {
                break 'unsupported;
            }
            let bf = image.base_format;
            if width == 0
                || height == 0
                || width > i32::MAX as u32
                || height > i32::MAX as u32
                || !(width as u64).is_power_of_two()
                || !(height as u64).is_power_of_two()
                || (three_d
                    && (depth == 0
                        || depth > SGI_ODYSSEY_OBJECTS
                        || !(depth as u64).is_power_of_two()
                        || (self.tex_mode & (ODSY_TEX_MODE_1D | ODSY_TEX_MODE_2D)) != 0))
                || (!three_d && depth != 1)
                || sampler_size
                    != (0x08000000 | (width.trailing_zeros() << 10) | (height.trailing_zeros() << 5) | (if three_d { depth.trailing_zeros() } else { 0 }))
                || ws > 3
                || wt > 3
                || (!border
                    && !captured_clamp
                    && !general_clamp
                    && !general_border
                    && (self.tex_sampler[5] != 0 || self.tex_sampler[6] != 0))
                || (general_clamp && !general_border && ((self.tex_sampler[5] | self.tex_sampler[6]) & 0xff000000) != 0)
                || (captured_clamp && !stored_border && ((self.tex_sampler[5] | self.tex_sampler[6]) & 0xff000000) != 0)
                || (border
                    && (ws != 3
                        || wt != 3
                        || width_word != (0x00fc0000 | width)
                        || height_word != (0x00fc0000 | height)
                        || ((self.tex_sampler[5] | self.tex_sampler[6]) & 0xff000000) != 0))
                || (general_clamp && three_d)
                || (clamp && !general_clamp && !captured_clamp)
                || (border
                    && !captured_clamp
                    && ((clamp && (ws != 2 || wt != 2)) || tex_shape != ODSY_TEX_MODE_2D || bf != SGI_GL_RGBA || filter != 0))
                || (min != 0 && min != 0x20 && min != 0x40 && min != 0x60 && min != 0x80 && min != 0xa0)
                || (!colour_table
                    && !colour_table_disabled
                    && !bordered
                    && !three_d_clamp
                    && word != (0x10000000 | components | (ws << 12) | (wt << 10) | filter)
                    && detail_filter == 0)
                || ((self.tex_lod[0] | self.tex_lod[1] | self.tex_lod[2] | self.tex_lod[3]) & !ODSY_TEX_LOD_MASK) != 0
                || !self.texture_scale_bias_known()
                || phase
                || self.tex_offset[0] != 0
                || self.tex_offset[1] != 0
                || self.tex_offset[2] != 0
                || (min >= 0x40 && lo > hi)
                || (colour_table
                    && ((colour_table_l8 && bf != SGI_GL_LUMINANCE)
                        || ((colour_table_rgba8_rgb || colour_table_rgba8_rgba) && bf != SGI_GL_RGBA)
                        || three_d
                        || tex_shape != ODSY_TEX_MODE_2D
                        || self.tex_colour_table_valid[0] != 0xffff
                        || self.tex_colour_table_valid[1] != 0xffff))
                || (colour_table_disabled && (bf != SGI_GL_LUMINANCE || three_d || tex_shape != ODSY_TEX_MODE_2D))
                || self.tex_env[0] < 1
                || self.tex_env[0] > 4
                || (self.tex_env[0] == 3
                    && (bf == SGI_GL_ALPHA || bf == SGI_GL_LUMINANCE || bf == SGI_GL_LUMINANCE_ALPHA || bf == SGI_GL_INTENSITY))
                || (self.tex_env[0] == 4
                    && (((self.tex_env[1] | self.tex_env[2]) & 0xff000000) != 0 || self.tex_env[3] != 0 || self.tex_env[4] != 0))
            {
                break 'unsupported;
            }
            image.border = bordered as u32;
            image.width = width;
            image.height = height;
            image.wrap_s = if border {
                SGI_GL_CLAMP_TO_BORDER
            } else if ws == 2 {
                SGI_GL_CLAMP
            } else if ws != 0 {
                SGI_GL_CLAMP_TO_EDGE
            } else {
                SGI_GL_REPEAT
            };
            // A 1D image must not acquire a T-axis border tap during filtering.
            image.wrap_t = if one_d_clamp {
                SGI_GL_CLAMP_TO_EDGE
            } else if border {
                SGI_GL_CLAMP_TO_BORDER
            } else if wt == 2 {
                SGI_GL_CLAMP
            } else if wt != 0 {
                SGI_GL_CLAMP_TO_EDGE
            } else {
                SGI_GL_REPEAT
            };
            if border || captured_clamp || general_clamp {
                image.border_color[0] = ((self.tex_sampler[5] >> 12) & 0xfff) as f32 / 4095.0f32;
                image.border_color[1] = (self.tex_sampler[5] & 0xfff) as f32 / 4095.0f32;
                image.border_color[2] = ((self.tex_sampler[6] >> 12) & 0xfff) as f32 / 4095.0f32;
                image.border_color[3] = (self.tex_sampler[6] & 0xfff) as f32 / 4095.0f32;
            }
            image.mag_filter = if detail_filter != 0 {
                detail_filter
            } else if (filter & 2) != 0 {
                SGI_GL_LINEAR
            } else {
                SGI_GL_NEAREST
            };
            image.min_filter = MIN_FILTERS[(min >> 5) as usize];
            self.texture_lod_bias(image);

            if three_d {
                let e = (first * 3) as usize;
                let entry = [self.tex_levels[e], self.tex_levels[e + 1], self.tex_levels[e + 2]];
                let stride_units = entry[1] & 0xfff;
                let expected_units = if three_d_stored { 4 } else { (width.div_ceil(16) + 1) & !1u32 };
                let mut base = entry[0] & 0x1fffffff;
                let mipmap = min >= 0x40;

                // Observed small-volume mip levels follow the preceding level's
                // slices, with one 128-byte address unit per slice. Larger-volume
                // packing and nonzero table selectors are not established. Keep
                // this bounded model policy separate from the level-zero address
                // rule.
                if first != 0
                    || self.tex_level0 != 0
                    || (!mipmap && word != 0x13000000 && word != 0x13000022 && !three_d_clamp)
                    || (mipmap
                        && ((filter != 0x40 && filter != 0xa0)
                            || ws != 0
                            || wt != 0
                            || (bf != SGI_GL_RGB && bf != SGI_GL_RGBA)
                            || width > 16
                            || height > 16
                            || depth > 16
                            || stride_units != 2
                            || entry[2] != 1))
                    || self.tex_env[0] > 2
                    || entry[1] != (0x41100000 | stride_units)
                    || (three_d_stored && entry[2] != height + 2)
                    || entry[2] == 0
                    || stride_units != expected_units
                    || base == 0
                    || (three_d_stored
                        && (bf != SGI_GL_RGBA || width > 48 || height > 30 || depth > 30 || (filter != 0 && filter != 0x22)))
                    || self.obj_count > SGI_ODYSSEY_OBJECTS
                    || entry[2] > ODSY_VOLUME_SLICE_SHIFT_MAX
                {
                    break 'unsupported;
                }
                let slice_stride = volume_slice_stride(entry[2]);
                // The captured 2x2x2 stored upload starts 15 slice strides above
                // the level base. For this one captured profile the rule applies
                // at any level base: a compatibility approximation of model
                // traces that showed it at two bases. Physical V12 texture layout
                // remains unqualified.
                if three_d_stored && width == 2 && height == 2 && depth == 2 {
                    let first_slice = slice_stride
                        .checked_mul(15)
                        .and_then(|s| s.checked_add(base as u64))
                        .and_then(|s| u32::try_from(s).ok());

                    let Some(first_slice) = first_slice else {
                        break 'unsupported;
                    };
                    base = first_slice;
                }
                terminal = cmax(cmax(width.trailing_zeros(), height.trailing_zeros()), depth.trailing_zeros());
                count = if mipmap && hi > 0 { cmin(terminal, ((hi + 255) / 256) as u32) + 1 } else { 1 };
                if count > SGI_GL_TEX_LEVELS_MAX {
                    break 'unsupported;
                }
                chain.clear();
                let mut slice_next: u32 = 0;
                for i in 0..count {
                    let level_width = cmax(1, width >> i);
                    let level_height = cmax(1, height >> i);
                    let level_depth = cmax(1, depth >> i);
                    if i != 0
                        && (self.tex_levels[(i * 3) as usize] != 0
                            || self.tex_levels[(i * 3 + 1) as usize] != 0
                            || self.tex_levels[(i * 3 + 2) as usize] != 0)
                    {
                        break 'unsupported;
                    }
                    let level_slice = slice_next;
                    let mut z = 0u32;
                    while z < level_depth + (if three_d_stored { 2 } else { 0 }) {
                        let slice_base = (z as u64)
                            .checked_mul(slice_stride)
                            .and_then(|o| o.checked_add(base as u64))
                            .and_then(|s| u32::try_from(s).ok());

                        let Some(slice_base) = slice_base else {
                            break 'unsupported;
                        };
                        let object = self.object_find(slice_base);
                        if object < 0 {
                            break 'unsupported;
                        }
                        let o = object as usize;
                        let sb: u32 = if three_d_stored { 15 } else { 0 };
                        let sb2: u32 = if three_d_stored { 2 } else { 0 };
                        let storage_end = (sb as u64 + level_height as u64 - 1 + sb2 as u64)
                            .checked_mul(self.obj_width[o] as u64)
                            .and_then(|e| e.checked_add(self.obj_offset[o] as u64))
                            .and_then(|e| {
                                e.checked_add(if three_d_stored { level_width as u64 + 18 } else { level_width as u64 })
                            });
                        if self.obj_width[o] != stride_units * 16
                            || self.obj_rows[o] < sb + level_height + sb2
                            || !storage_end.is_some_and(|e| e <= self.obj_used as u64 && e <= self.obj_mem.size / 4)
                        {
                            break 'unsupported;
                        }
                        if three_d_stored && !self.object_region_written(object, 15, 15, level_width + 2, level_height + 2) {
                            break 'unsupported;
                        }
                        if slice_next as usize >= self.tex_slice_offset.len() {
                            break 'unsupported;
                        }
                        self.tex_slice_offset[slice_next as usize] = self.obj_offset[o];
                        self.tex_slice_width[slice_next as usize] = self.obj_width[o];
                        slice_next += 1;
                        z += 1;
                    }
                    chain.touch(i);
                    chain.volumes[i as usize] = TextureVolume {
                        base,
                        stride_units,
                        level_word2: entry[2],
                        width: level_width,
                        height: level_height,
                        depth: level_depth,
                        base_format: image.base_format,
                        words: self.obj_used,
                        border: three_d_stored,
                        first_slice: level_slice,
                        slices: slice_next - level_slice,
                        ..Default::default()
                    };
                    chain.touch(i);
                    chain.levels[i as usize] = GlTexLevel {
                        level: i,
                        width: level_width,
                        height: level_height,
                        depth: level_depth,
                        base_format: image.base_format,
                        internal_format: image.base_format,
                        read_texel_3d: true,
                        ..Default::default()
                    };
                    if i + 1 < count {
                        let next_base = (level_depth as u64)
                            .checked_mul(slice_stride)
                            .and_then(|n| n.checked_add(base as u64))
                            .and_then(|n| u32::try_from(n).ok());

                        let Some(next_base) = next_base else {
                            break 'unsupported;
                        };
                        base = next_base;
                    }
                }
                chain.volume = chain.volumes[0];
                image.depth = depth;
                image.volume = true;
                image.wrap_r = if three_d_stored || (three_d_clamp && (word & 0x200) != 0) {
                    SGI_GL_CLAMP
                } else {
                    SGI_GL_REPEAT
                };
                image.read_texel_3d = true;
                image.border = three_d_stored as u32;
                if three_d_stored {
                    image.wrap_s = SGI_GL_CLAMP;
                    image.wrap_t = SGI_GL_CLAMP;
                    image.read_border_texel_3d = true;
                }
                if !self.texture_scale_bias_identity() {
                    chain.volume.scale_bias = true;
                    image.post_filter = true;
                }
                if mipmap {
                    chain.pyramid = Some(PyramidShape {
                        count,
                        base_level: 0,
                        max_level: count - 1,
                        min_lod: lo as f64 / 256.0,
                        max_lod: hi as f64 / 256.0,
                    });
                }
                return true;
            }

            terminal = cmax((sampler_size >> 10) & 31, (sampler_size >> 5) & 31);
            count = if min >= 0x40 && hi > 0 { cmin(terminal, ((hi + 255) / 256) as u32) + 1 } else { 1 };
            if count > SGI_GL_TEX_LEVELS_MAX || first + count > SGI_ODYSSEY_TEX_LEVELS {
                break 'unsupported;
            }
            chain.clear();
            chain.volume.colour_table_alpha = colour_table_rgba8_rgba || word == 0x11600001;
            chain.volume.colour_table = colour_table;
            chain.volume.scale_bias = !self.texture_scale_bias_identity();
            if chain.volume.colour_table || chain.volume.scale_bias {
                image.post_filter = true;
            }
            for i in 0..count {
                let e = ((first + i) * 3) as usize;
                let entry = [self.tex_levels[e], self.tex_levels[e + 1], self.tex_levels[e + 2]];
                let level_width = cmax(1, width >> i);
                let level_height = cmax(1, height >> i);
                let stride_units = entry[1] & 0xfff;
                // A bordered level's row is its width plus 16 texels each side,
                // in 16-texel units rounded up to even, as the unbordered rule.
                let expected_units = if stored_border {
                    if one_d_shape && level_width == 64 { 6 } else { 4 }
                } else if general_border {
                    ((level_width + 32).div_ceil(16) + 1) & !1u32
                } else {
                    (level_width.div_ceil(16) + 1) & !1u32
                };
                let base = entry[0] & 0x1fffffff;
                let interior_x: u32 = if bordered { 16 } else { 0 };
                let interior_y: u32 = if two_d_stored || general_border { 16 } else { 0 };
                let rows = level_height + interior_y + (two_d_stored || general_border) as u32;
                let require_backing = stored_border;

                if (stored_border && (entry[1] != (if expected_units == 6 { 0x41100006 } else { 0x41100004 }) || base == 0))
                    || (two_d_clamp
                        && !stored_border
                        && (filter != 0 || self.tex_sampler[5] != 0 || self.tex_sampler[6] != 0)
                        && entry[1] != 0x41100002)
                    || (entry[1] & 0x3c0ff000) != 0
                    || entry[2] != 0
                    || (i != 0 && (entry[1] & 0xc3f00000) != (self.tex_levels[(first * 3 + 1) as usize] & 0xc3f00000))
                    || (base != 0 && stride_units != expected_units && !(count == 1 && stride_units == 0))
                {
                    break 'unsupported;
                }
                chain.touch(i);
                chain.levels[i as usize] = GlTexLevel {
                    level: i,
                    width: level_width,
                    height: level_height,
                    base_format: image.base_format,
                    internal_format: image.base_format,
                    ..Default::default()
                };
                if base == 0 || self.obj_count > SGI_ODYSSEY_OBJECTS {
                    if require_backing {
                        break 'unsupported;
                    }
                    continue;
                }
                let object = self.object_find(base);
                let backed = object >= 0 && {
                    let o = object as usize;
                    let end = ((rows - 1) as u64)
                        .checked_mul(self.obj_width[o] as u64)
                        .and_then(|e| e.checked_add(self.obj_offset[o] as u64))
                        .and_then(|e| e.checked_add(interior_x as u64))
                        .and_then(|e| e.checked_add(level_width as u64 + bordered as u64));
                    !(self.obj_width[o] < level_width
                        || (stride_units != 0 && self.obj_width[o] != stride_units * 16)
                        || self.obj_rows[o] < rows
                        || !end.is_some_and(|e| e <= self.obj_used as u64 && e <= self.obj_mem.size / 4))
                };
                if !backed {
                    if require_backing {
                        break 'unsupported;
                    }
                    continue;
                }
                let o = object as usize;
                if require_backing {
                    let written = self.object_region_written(
                        object,
                        interior_x - 1,
                        if two_d_stored { interior_y - 1 } else { 0 },
                        level_width + 2,
                        if one_d_shape { 1 } else { level_height + 2 },
                    );

                    if !written {
                        break 'unsupported;
                    }
                }
                chain.touch(i);
                chain.storage[i as usize] = Texture {
                    // Captured stored-border policy uses interior coordinate 16.
                    offset: self.obj_offset[o].wrapping_add(interior_y.wrapping_mul(self.obj_width[o])).wrapping_add(interior_x),
                    stride: self.obj_width[o],
                    width: level_width,
                    height: level_height,
                    base_format: image.base_format,
                    words: self.obj_used,
                    border_rows: two_d_stored || general_border,
                };
                chain.levels[i as usize].read_texel = true;
                if bordered {
                    chain.levels[i as usize].border = 1;
                    chain.levels[i as usize].read_border_texel = true;
                }
            }
            chain.pyramid = Some(PyramidShape {
                count,
                base_level: 0,
                max_level: count - 1,
                min_lod: lo as f64 / 256.0,
                max_lod: if min >= 0x40 { hi as f64 / 256.0 } else { 15.0 },
            });
            if detail_identity {
                if !self.texture_detail_backing_setup(
                    chain,
                    detail_sampler.descriptor,
                    detail_sampler.width,
                    detail_sampler.height,
                    image.base_format,
                ) {
                    break 'unsupported;
                }
                chain.volume.detail_weight = detail_weight.unwrap_or_default();
                image.detail_image = Some(chain.detail_image);
                image.detail_level = detail_sampler.level;
                image.detail_mode = detail_sampler.mode;
                image.read_filter_weight = true;
            }
            return true;
        }
        let index_mode = self.gl_index_mode();
        crate::dlog!(
            LogModule::Odyssey,
            "sgi-odyssey: unsupported texture state sampler {:#x} environment {}{}, draw dropped",
            self.tex_sampler[4],
            self.tex_env[0],
            if index_mode { " colour-index" } else { "" }
        );
        false
    }

    /// Rebuild shared state from family words; no borrowed image survives a draw.
    pub fn texture_vertex(&mut self) {
        const ENVIRONMENTS: [u32; 5] = [SGI_GL_REPLACE, SGI_GL_REPLACE, SGI_GL_MODULATE, SGI_GL_DECAL, SGI_GL_BLEND];
        let mut strq: [u32; 4] = [self.tex_coord[0], self.tex_coord[1], self.tex_coord_r, self.tex_coord_q];
        let one_d = (self.tex_mode & (ODSY_TEX_MODE_1D | ODSY_TEX_MODE_2D)) == ODSY_TEX_MODE_1D;
        let mut matrix = [0.0f32; 16];

        self.gl().cur.texture_2d = ((self.tex_mode & (ODSY_TEX_MODE_1D | ODSY_TEX_MODE_2D)) != 0) as u32;
        self.gl().cur.texture_3d = ((self.tex_mode & ODSY_TEX_MODE_3D) != 0) as u32;
        // Excluded environments are rejected by the family before rasterization.
        self.gl().cur.texture_env_mode = SGI_GL_REPLACE;
        if (self.tex_env[0] as usize) < ENVIRONMENTS.len() {
            self.gl().cur.texture_env_mode = ENVIRONMENTS[self.tex_env[0] as usize];
        }
        // Captured linear planes already include the eye-space transform.
        if (self.cmd_tag == ODSY_GL_VERTEX_2I
            || self.cmd_tag == ODSY_GL_VERTEX_2F
            || self.cmd_tag == ODSY_GL_VERTEX_3F
            || self.cmd_tag == ODSY_GL_VERTEX_3I
            || self.cmd_tag == ODSY_GL_VERTEX_4F)
            && self.gl_linear_texgen != 0
        {
            let mut vertex = [
                odyssey_f32(self.cmd_operand[0]),
                odyssey_f32(self.cmd_operand[1]),
                if self.cmd_tag != ODSY_GL_VERTEX_2I && self.cmd_tag != ODSY_GL_VERTEX_2F { odyssey_f32(self.cmd_operand[2]) } else { 0.0 },
                if self.cmd_tag == ODSY_GL_VERTEX_4F { odyssey_f32(self.cmd_operand[3]) } else { 1.0 },
            ];

            if self.cmd_tag == ODSY_GL_VERTEX_2I || self.cmd_tag == ODSY_GL_VERTEX_3I {
                vertex[0] = (self.cmd_operand[0] as i32) as f64;
                vertex[1] = (self.cmd_operand[1] as i32) as f64;
            }
            if self.cmd_tag == ODSY_GL_VERTEX_3I {
                vertex[2] = (self.cmd_operand[2] as i32) as f64;
            }
            for axis in 0..4usize {
                let plane = if axis == 3 {
                    self.tex_linear_plane_q
                } else if axis == 2 {
                    self.tex_linear_plane_r
                } else {
                    self.tex_linear_plane[axis]
                };
                let mut value = 0.0f32;

                if (self.gl_linear_texgen & self.tex_linear_valid & (1u32 << axis)) == 0 {
                    continue;
                }
                for k in 0..4usize {
                    value = (value as f64 + odyssey_f32(plane[k]) * vertex[k]) as f32;
                }
                strq[axis] = value.to_bits();
            }
        }
        let rows = self.gl_texture_matrix;
        gl_transpose(&rows, &mut matrix);
        // Input T may transform S, but transformed T cannot affect 1D sampling.
        if one_d {
            for i in 0..4usize {
                matrix[i * 4 + 1] = 0.0;
            }
        }
        self.gl().set_matrix(SGI_GL_TEXTURE, &matrix);
        self.gl_feed(SGI_GL_OP_TEXCOORD4F as u16, &strq);
    }
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
        let textured = d.state.texture_2d != 0 || d.state.texture_3d != 0;

        if textured {
            let chain = texture.get_or_insert_with(Box::default);

            if !self.texture_setup(&mut image, chain) {
                if let Some(mut j) = spare {
                    j.chain = texture;
                    self.raster.give_back(j);
                }
                crate::dlog_dev!(LogModule::Odyssey, "sgi-odyssey: GL draw path: backend scalar-skip result {} called=0", SGI_GL_DRAW_OK);
                return SGI_GL_DRAW_OK;
            }
        }
        if (state.texture_2d != 0 || state.texture_3d != 0) && self.tex_env[0] == 4 {
            // The family words pack two normalized 12-bit components each.
            state.texture_env_color[0] = ((self.tex_env[1] >> 12) & 0xfff) as f32 / 4095.0f32;
            state.texture_env_color[1] = (self.tex_env[1] & 0xfff) as f32 / 4095.0f32;
            state.texture_env_color[2] = ((self.tex_env[2] >> 12) & 0xfff) as f32 / 4095.0f32;
            state.texture_env_color[3] = (self.tex_env[2] & 0xfff) as f32 / 4095.0f32;
        }
        if !self.gl_index_mode() {
            state.colour_mask = (self.gl_write_mask >> ODSY_WRITE_COLOUR_SHIFT) & ODSY_WRITE_COLOUR_MASK;
        }
        state.fog_color[0] = ((self.gl_fog[0] >> 12) & 0xfff) as f32 / 4096.0f32;
        state.fog_color[1] = (self.gl_fog[0] & 0xfff) as f32 / 4096.0f32;
        state.fog_color[2] = ((self.gl_fog[1] >> 12) & 0xfff) as f32 / 4096.0f32;
        state.fog_color[3] = (self.gl_fog[1] & 0xfff) as f32 / 4096.0f32;
        state.fog_index = ((self.gl_fog[0] >> 12) & 0xfff) as f32;
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
        self.raster.submit_draw(job);
        if d.line_phase.is_some() {
            d.line_phase = Some(LINE_PHASE_ON_RASTER);
        }
        SGI_GL_DRAW_OK
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
            self.gl_report_refusals(op.kind != SGI_RASTER_CLEAR || (op.clear_buffers & SGI_GL_COLOR_BUFFER_BIT) != 0);
        }
        self.raster.submit(move |ctx| {
            // SAFETY: the job runs before the front end next touches the
            // planes (PlaneSpans).
            let mut planes = unsafe { pixels.pixels() };
            planes.row_targets = ops.iter().fold(0, |bits, op| bits | GlPixels::target_bits(&op.targets[..op.ntargets.min(SGI_GL_MAX_TARGETS) as usize]));
            for op in &ops {
                rules.raster(&t, &mut planes, op);
            }
            planes.record(ctx);
        });
    }

    /// glClear's raster op: not on the first-triangle path, wired for later.
    pub fn gl_raster(&mut self, op: &RasterOp) {
        self.gl_raster_ops(vec![*op]);
    }

    /// Run one command in the GL context, with the board as the sink for the
    /// draws it emits. The context is lifted out of the board for the call,
    /// so that a draw can borrow the rest of the board.
    pub fn gl_command(&mut self, cmd: &GlCmd) {
        let mut ctx = self.gl_ctx.take().expect("the GL context is not in a command");

        ctx.command(self, cmd);
        self.gl_ctx = Some(ctx);
    }

    /// The board's GL context, which is only out of the board during
    /// Board::gl_command().
    pub fn gl(&mut self) -> &mut GlContext {
        self.gl_ctx.as_mut().expect("the GL context is not in a command")
    }

    pub fn gl_ref(&self) -> &GlContext {
        self.gl_ctx.as_ref().expect("the GL context is not in a command")
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
    /// (ODSY_REG_DRAW_MASK), so the shared names are slots here, not the
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
        let draw: u32 = (if buffers & 1 != 0 { SGI_GL_DRAW_FRONT } else { 0 })
            | (if buffers & 2 != 0 { SGI_GL_DRAW_BACK } else { 0 });
        let n = cmax(self.clip_kept, 1);
        let origin = self.win_origin;
        let clip_kept = self.clip_kept;
        let clip_rect = self.clip_rect;

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
            self.gl_command(&GlCmd::new(SGI_GL_OP_DRAW_BUFFERS as u16, std::slice::from_ref(&draw)));
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
pub fn cull_face_mode(code: u32) -> u32 {
    match code {
        1 => SGI_GL_BACK,
        2 => SGI_GL_FRONT,
        3 => SGI_GL_FRONT_AND_BACK,
        _ => u32::MAX,
    }
}

/// 0x404 bit 23 to the shared layer's glFrontFace sense, inverted -- see
/// the ODSY_MODE_FRONT_FACE_CW comment for why.
pub fn front_face(pattern_mode: u32) -> u32 {
    if (pattern_mode & ODSY_MODE_FRONT_FACE_CW) != 0 { SGI_GL_CCW } else { SGI_GL_CW }
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
        let held_polygon = self.gl_begin_open && self.gl_primitive == 9;

        if held_polygon {
            self.gl().in_begin = 0;
        }
        self.gl_feed(op, args);
        self.gl().in_begin = in_begin;
    }

    /// Hand the GL state register 0x0404 carries to the shared context: bit 12,
    /// the shade model, bit 16, the depth test, bit 14, the alpha test, bit 15,
    /// the stencil test, bit 17, blending, bit 19, index logic op, bit 20,
    /// colour logic op, bit 8, GL_CULL_FACE, bits 10:9, the face it culls, and
    /// bit 23, front face (fed inverted; see the ODSY_MODE_FRONT_FACE_CW
    /// comment), and bit 3, float GL line stipple. The shared rasterizer fills
    /// a flat triangle from its provoking vertex, the last one for triangles
    /// and strips as GL specifies. Nothing in the stream says which vertex the
    /// board takes; the library never reorders or repeats colours to steer it,
    /// so GL's own rule is INFERRED. Called on every 0x0404, including the X
    /// server's pattern-mode writes, which also update the separate integer 2D
    /// interpreter.
    ///
    /// Bit 15, the stencil test, gates fragments through the model's stencil
    /// plane (ODSY_MODE_STENCIL_TEST).
    pub fn gl_apply_polygon_stipple(&mut self) {
        let cap = SGI_GL_POLYGON_STIPPLE;
        let enable = (self.pattern_mode & ODSY_GL_RECT_PATTERN_ENABLE) != 0 && (self.pattern_shape & !ODSY_PATTERN_SHAPE_OPAQUE) == 0;

        if self.gl_in_prim {
            return;
        }
        if enable {
            let mut rows = [0u32; 32];

            for i in 0..rows.len() {
                rows[i] = self.pattern_buffer[31 - i];
            }
            if !self.gl().polygon_stipple(&rows) {
                return;
            }
        }
        self.gl_feed32(if enable { SGI_GL_OP_ENABLE } else { SGI_GL_OP_DISABLE }, &[cap]);
    }

    /// 0x0404 bit 17 (ODSY_MODE_BLEND) blends only into an RGBA drawable:
    /// OpenGL blends RGBA fragments alone (OpenGL 1.2.1, 4.1.6), and the X
    /// server sets the bit with factors ONE, ZERO for its opaque stipples,
    /// whose background is the 0x0c21 constant colour and needs no destination
    /// (ODSY_PATTERN_SHAPE_OPAQUE). Fed again when the colour mode changes.
    pub fn gl_apply_blend(&mut self) {
        let blend = SGI_GL_BLEND;
        let on = (self.pattern_mode & ODSY_MODE_BLEND) != 0 && !self.gl_index_mode();

        self.gl_feed32(if on { SGI_GL_OP_ENABLE } else { SGI_GL_OP_DISABLE }, &[blend]);
    }

    pub fn gl_apply_mode(&mut self) {
        let shade = if (self.pattern_mode & ODSY_MODE_FLAT_SHADE) != 0 { SGI_GL_FLAT } else { SGI_GL_SMOOTH };
        let cull_mode = cull_face_mode((self.pattern_mode >> ODSY_MODE_CULL_MODE_SHIFT) & ODSY_MODE_CULL_MODE_MASK);
        let front_face = front_face(self.pattern_mode);
        let pm = self.pattern_mode;
        let en = |bit: u32| if (pm & bit) != 0 { SGI_GL_OP_ENABLE } else { SGI_GL_OP_DISABLE };

        self.gl_feed32(SGI_GL_OP_SHADE_MODEL, &[shade]);
        // Bit assignments follow the library's fragment-state encoding.
        self.gl_feed32(en(ODSY_MODE_POINT_SMOOTH), &[SGI_GL_POINT_SMOOTH]);
        self.gl_feed32(en(ODSY_MODE_LINE_SMOOTH), &[SGI_GL_LINE_SMOOTH]);
        // The shared layer has no dither rule yet, so the bit is kept as family
        // state and reported at the first draw it would affect
        // (Board::gl_vertex_colour()). Once the shared rule lands this feeds
        // SGI_GL_OP_ENABLE/DISABLE of SGI_GL_DITHER instead.
        self.gl_dither = (self.pattern_mode & ODSY_MODE_DITHER) != 0;
        self.gl_feed32(en(ODSY_MODE_DEPTH_TEST), &[SGI_GL_DEPTH_TEST]);
        self.gl_feed32(en(ODSY_MODE_ALPHA_TEST), &[SGI_GL_ALPHA_TEST]);
        self.gl_feed32(en(ODSY_MODE_STENCIL_TEST), &[SGI_GL_STENCIL_TEST]);
        self.gl_apply_blend();
        self.gl_feed32(en(ODSY_MODE_FOG), &[SGI_GL_FOG]);
        self.gl_feed32(en(ODSY_MODE_INDEX_LOGIC_OP), &[SGI_GL_INDEX_LOGIC_OP]);
        self.gl_feed32(en(ODSY_MODE_COLOR_LOGIC_OP), &[SGI_GL_COLOR_LOGIC_OP]);
        self.gl_feed32(en(ODSY_MODE_OFFSET_POINT), &[SGI_GL_POLYGON_OFFSET_POINT]);
        self.gl_feed32(en(ODSY_MODE_OFFSET_LINE), &[SGI_GL_POLYGON_OFFSET_LINE]);
        self.gl_feed32(en(ODSY_MODE_OFFSET_FILL), &[SGI_GL_POLYGON_OFFSET_FILL]);
        self.gl_feed32(en(ODSY_MODE_CULL_FACE), &[SGI_GL_CULL_FACE]);
        self.gl_feed_raster_state(en(ODSY_MODE_GL_LINE_STIPPLE) as u16, &[SGI_GL_LINE_STIPPLE]);
        self.gl_feed32(SGI_GL_OP_CULL_FACE, &[cull_mode]);
        self.gl_feed32(SGI_GL_OP_FRONT_FACE, &[front_face]);
        self.gl_apply_polygon_stipple();
    }

    pub fn gl_apply_line_stipple(&mut self) {
        // Word 1 remains the integer 2D path's 32-bit rotated pattern.
        let args = [(self.line_stipple[2] & 0xff) + 1, self.line_stipple[0] & 0xffff];

        self.gl_feed_raster_state(SGI_GL_OP_LINE_STIPPLE as u16, &args);
    }

    /// Push the family-shadowed clip box (ODSY_REG_CLIP_BOX) to the shared
    /// context's scissor rectangle (see the macro's own comment for the
    /// encoding and the always-on rationale). min > max -- including the
    /// library's own 0x1000, 0x1000 empty encoding -- clips every fragment,
    /// matched here with a zero-area box rather than a negative width or
    /// height, which SGI_GL_OP_SCISSOR rejects.
    pub fn gl_apply_clip_box(&mut self) {
        let x_min = (self.gl_clip_box[0] >> ODSY_CLIP_BOX_MIN_SHIFT) & ODSY_CLIP_BOX_FIELD_MASK;
        let x_max = self.gl_clip_box[0] & ODSY_CLIP_BOX_FIELD_MASK;
        let y_min = (self.gl_clip_box[1] >> ODSY_CLIP_BOX_MIN_SHIFT) & ODSY_CLIP_BOX_FIELD_MASK;
        let y_max = self.gl_clip_box[1] & ODSY_CLIP_BOX_FIELD_MASK;
        let mut box_ = [x_min, y_min, 0, 0];

        if x_min <= x_max && y_min <= y_max {
            box_[2] = x_max - x_min + 1;
            box_[3] = y_max - y_min + 1;
        }
        self.gl_feed32(SGI_GL_OP_SCISSOR, &box_);
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
            eye[r] = odyssey_f32(self.gl_clip_eye[i as usize * 4 + r]);
        }
        for r in 0..4usize {
            for k in 0..4usize {
                obj[k] += eye[r] * odyssey_f32(self.gl_modelview[r * 4 + k]);
            }
        }
        for k in 0..4usize {
            args[1 + k] = f32_bits(obj[k] as f32);
        }
        self.gl_feed32(SGI_GL_OP_CLIP_PLANE, &args);
    }

    /// Recompute which of the shared context's six clip-plane slots are
    /// tested, from 0x0405 bits 2:0's thermometer count n (see
    /// ODSY_TEX_MODE_CLIP_COUNT_MASK's comment): slots 0..n-1 enabled, the rest
    /// disabled. Called on every 0x0405 write; does not
    /// touch the loaded coefficients themselves, which a separate load
    /// (ODSY_GL_CLIP_PLANE_LOAD) feeds.
    pub fn gl_apply_clip_planes(&mut self) {
        let bits = self.tex_mode & ODSY_TEX_MODE_CLIP_COUNT_MASK;
        let n = bits.count_ones();

        for i in 0..SGI_GL_CLIP_PLANES {
            let plane = SGI_GL_CLIP_PLANE0 + i;

            self.gl_feed32(if i < n { SGI_GL_OP_ENABLE } else { SGI_GL_OP_DISABLE }, &[plane]);
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
pub fn blend_factor(code: u32, dst: bool) -> u32 {
    match code {
        0x00 => 0, // GL_ZERO
        0x08 => 1, // GL_ONE
        0x10 => {
            if dst { SGI_GL_SRC_COLOR } else { SGI_GL_DST_COLOR }
        }
        0x18 => SGI_GL_SRC_ALPHA,
        0x20 => SGI_GL_DST_ALPHA,
        0x28 => SGI_GL_CONSTANT_COLOR,
        0x30 => SGI_GL_CONSTANT_ALPHA,
        0x38 => {
            if dst { SGI_GL_ONE_MINUS_SRC_COLOR } else { SGI_GL_ONE_MINUS_DST_COLOR }
        }
        0x39 => SGI_GL_ONE_MINUS_SRC_ALPHA,
        0x3a => SGI_GL_ONE_MINUS_DST_ALPHA,
        0x3b => SGI_GL_ONE_MINUS_CONSTANT_COLOR,
        0x3c => SGI_GL_ONE_MINUS_CONSTANT_ALPHA,
        0x40 => {
            if dst { u32::MAX } else { SGI_GL_SRC_ALPHA_SATURATE }
        }
        _ => u32::MAX,
    }
}

/// 0x041f's three 3-bit stencil-op fields (fail, depth-fail, depth-pass)
/// to the shared layer's enum: KEEP 0, ZERO 1, REPLACE 2, INCR 3, DECR 4,
/// INVERT 5. Returns UINT32_MAX for the two codes the board leaves unused.
pub fn stencil_op(code: u32) -> u32 {
    match code {
        0 => SGI_GL_KEEP,
        1 => 0, // GL_ZERO
        2 => SGI_GL_REPLACE,
        3 => SGI_GL_INCR,
        4 => SGI_GL_DECR,
        5 => SGI_GL_INVERT,
        _ => u32::MAX,
    }
}

/// 0x0415's 2-bit face code to the shared layer's polygon-mode enum,
/// GL_POINT 0 to GL_FILL 2. Returns UINT32_MAX for the one code the board
/// leaves undefined, which the shared layer's own glPolygonMode validation
/// then rejects, leaving the previous mode in place.
pub fn polygon_mode(code: u32) -> u32 {
    match code {
        0 => SGI_GL_POINT,
        1 => SGI_GL_LINE,
        2 => SGI_GL_FILL,
        _ => u32::MAX,
    }
}

/// 0x0c21 word 0 bits 15:13 to the shared layer's blend equation. Codes 3
/// and 4, the SGIX alpha min/max extension, have no shared equivalent
/// (gl_blend_equation_valid() accepts only the other six), so the
/// dispatch case reports and skips them rather than feeding a code the
/// shared layer would reject.
pub static BLEND_EQUATION: [u32; 8] = [
    SGI_GL_FUNC_ADD,
    SGI_GL_MIN,
    SGI_GL_MAX,
    0,
    0,
    SGI_GL_LOGIC_OP,
    SGI_GL_FUNC_SUBTRACT,
    SGI_GL_FUNC_REVERSE_SUBTRACT,
];

impl Board {
    /// 0x0420 and 0x0452 to the shared context's comparison and depth mask.
    pub fn gl_apply_depth(&mut self) {
        let func = SGI_GL_NEVER_ENUM + (self.gl_depth_func & ODSY_DEPTH_FUNC_MASK);
        let mask = ((self.gl_write_mask & ODSY_WRITE_DEPTH) != 0) as u32;

        self.gl_feed32(SGI_GL_OP_DEPTH_FUNC, &[func]);
        self.gl_feed32(SGI_GL_OP_DEPTH_MASK, &[mask]);
    }
}

/// The lighting words matched by value (ODSY_GL_NORMAL_3F and the rest).
pub static GL_LIGHT_WORDS: [u32; 16] = [
    ODSY_GL_NORMAL_3F,
    ODSY_GL_MATERIAL_AMBIENT,
    ODSY_GL_MATERIAL_DIFFUSE,
    ODSY_GL_MATERIAL_SPECULAR,
    ODSY_GL_MATERIAL_EMISSION,
    ODSY_GL_BACK_AMBIENT,
    ODSY_GL_BACK_DIFFUSE,
    ODSY_GL_BACK_SPECULAR,
    ODSY_GL_BACK_EMISSION,
    ODSY_GL_SHININESS_SQRT,
    ODSY_GL_BACK_SHININESS_SQRT,
    ODSY_GL_SHININESS_DERIVED,
    ODSY_GL_LIGHT_COLOUR_LOAD,
    ODSY_GL_LIGHT_VEC_LOAD,
    ODSY_GL_CLIP_PLANE_LOAD,
    ODSY_PROG_RUN_ONCE,
];

/// The texture-coordinate and generation fragments.
pub static UCODE_TEXGEN: [u32; 14] = [
    0x033, 0x04d, 0x052, 0x057, 0x05c, 0x061, 0x066, 0x06b, 0x06c, 0x077, 0x07b, 0x07f, 0x091, 0x09f,
];

pub fn prog_value(word: u32) -> u32 {
    (word & ODSY_PROG_LOW_MASK) | (((word >> ODSY_PROG_HIGH_SHIFT) & 1) << 10)
}

pub fn is_program_word(word: u32) -> bool {
    let cls = word >> ODSY_PROG_CLASS_SHIFT;

    if (word & ODSY_PROG_SHAPE) != ODSY_PROG_TAG {
        return false;
    }
    if (ODSY_PROG_PROGRAM_HALFWORD..=ODSY_PROG_PROGRAM_HALFWORD_LAST).contains(&cls) {
        return true;
    }
    if (ODSY_PROG_SEQ_FIRST..=ODSY_PROG_SEQ_LAST).contains(&cls)
        && (word & ODSY_GL_MATRIX_SELECT_SHAPE) != ODSY_GL_MATRIX_SELECT_TAG
    {
        return true;
    }
    matches!(
        cls,
        ODSY_PROG_COLOUR_MODE
            | ODSY_PROG_COLOUR_MODE_B
            | ODSY_PROG_SET_UCODE
            | ODSY_PROG_PROGRAM_ADDR
            | ODSY_PROG_CONST_SELECT
            | ODSY_PROG_ENTRY_29
            | ODSY_PROG_ENTRY_2A
            | ODSY_PROG_ENTRY_2C
            | ODSY_PROG_LIGHT_VEC_SELECT
            | ODSY_PROG_LIGHT_COLOUR_SELECT
            | ODSY_PROG_LIST_INDEX
            | ODSY_PROG_COLOUR_MATERIAL
            | ODSY_PROG_CONST_STORE
    )
}

pub fn ucode_is_texgen(a: u32) -> bool {
    UCODE_TEXGEN.contains(&a)
}
