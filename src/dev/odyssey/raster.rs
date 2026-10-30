//! The scalar rasteriser's core: the rules a draw follows where OpenGL
//! leaves a choice, the draw and fragment types, the fragment path (window
//! mapping, depth and stencil, blending, logic ops, fog, the colour mask)
//! and the draw entry point. Triangles, lines, pixel work, texturing and
//! banding are in the other raster_*.rs files; raster_fixed_texture.rs is
//! the fixed-point fast path for textured triangles.
//!
//! Ported from include/hw/display/sgi/rasterizer.h in jamesbraid/qemu-sgi-origami.

use super::*;

// The rules a head's rasterizer follows where OpenGL leaves a choice.

pub type Rounding = u32;
pub const SGI_ROUND_FLOOR: Rounding = 0;
pub const SGI_ROUND_HALF_UP: Rounding = 1;
pub const SGI_ROUND_HALF_EVEN: Rounding = 2;
pub const SGI_ROUND_HALF_AWAY: Rounding = 3;

pub type Arith = u32;
pub const SGI_ARITH_DOUBLE: Arith = 1;

pub type TieRule = u32;
pub const SGI_TIE_TOP_LEFT_BOTTOM_ORIGIN: TieRule = 1;

pub type LineRule = u32;
pub const SGI_LINE_COVERAGE_DOUBLE: LineRule = 1;

pub type Dither = u32;
pub const SGI_DITHER_NONE: Dither = 0;
pub const SGI_DITHER_BAYER4_TOP_ROWS: Dither = 1;

pub type DitherAnchor = u32;
pub const SGI_DITHER_SCREEN: DitherAnchor = 0;

pub type SmoothRule = u32;
pub const SGI_SMOOTH_TWO_PIXEL_FILTER: SmoothRule = 1;

// Rounding: how a window coordinate or depth is rounded to its grid
// (SGI_ROUND_HALF_AWAY is round half away from zero). TieRule: which pixel centres on a
// shared triangle edge the edge owns, rows counted from the top or from the
// bottom. LineRule: which pixels a width-1 segment covers. Dither:
// BAYER4_TOP_ROWS is a 4x4 ordered dither, floor(c * max + B / 16) with B
// from the Bayer index matrix at (column, row) mod 4, rows from the top.
// DitherAnchor: the display or the window. SmoothRule: AREA is the
// exact box filter; TWO_PIXEL_FILTER the separable tent filter.

#[derive(Clone, Default)]
pub struct RasterRules {
    /// window coordinates snap to 2^-n pixel
    pub subpixel_bits: u8,
    /// per channel of a stored colour
    pub color_bits: u8,
    /// of a stored depth
    pub depth_bits: u8,
    /// of a stored stencil value, 0 for planes with no stencil buffer
    pub stencil_bits: u8,
    /// Rounding, for snapping and depth
    pub rounding: u8,
    /// TieRule
    pub tie_rule: u8,
    /// Arith
    pub arith: u8,
    /// LineRule
    pub line_rule: u8,
    /// Dither
    pub dither: u8,
    /// DitherAnchor
    pub dither_anchor: u8,
    /// stored A width, zero means no alpha
    pub alpha_bits: u8,
    /// Zero derives precision from color/alpha bits.
    pub alpha_test_bits: u8,
    /// Smooth points, lines and polygons as their enables ask; without it
    /// a smoothed primitive is drawn aliased and reported.
    pub smooth_primitives: bool,
    /// SmoothRule
    pub smooth_rule: u8,
    /// Depth linear in eye distance (a w-buffer): a vertex's window depth
    /// is depth_row . e for its eye coordinates e, perspective-corrected
    /// per fragment; a depth outside [0, 1] is dropped.
    pub depth_row_given: bool,
    pub depth_row: [f64; 4],
    /// With depth_row, a second row whose product with the eye coordinates
    /// divides the first's. Not given, the divisor is one.
    pub depth_divisor_given: bool,
    pub depth_divisor: [f64; 4],
    /// With depth_row, how far outside [0, 1] a depth may lie and be
    /// clamped rather than dropped.
    pub depth_slack: f64,
    /// Stored depth is the board's 24-bit float word of 2S times the window
    /// depth (depth_word.rs), which orders as an integer; needs depth_bits
    /// 24. A polygon offset then counts steps of that word.
    pub depth_float: bool,
}

/// A vertex in clip coordinates, as view-volume clipping takes and makes it.
#[derive(Clone)]
pub struct VolumeVertex {
    pub clip: [f64; 4],
    pub eye: [f64; 4],
    pub attr: [f32; SGI_GL_VERTEX_FLOATS],
}

impl Default for VolumeVertex {
    fn default() -> Self {
        VolumeVertex { clip: [0.0; 4], eye: [0.0; 4], attr: [0.0; SGI_GL_VERTEX_FLOATS] }
    }
}

/// The most vertices clipping a triangle by the six planes leaves. usize
/// so it can size arrays.
pub const SGI_VOLUME_TRIANGLE_MAX: usize = 3 + 6;

pub const SGI_GL_TEX_LEVELS_MAX: u32 = 31;

/// One level of a texture image, as the family describes it. The `read_*`
/// flags say which reads the draw's Texels answer for it.
#[derive(Clone, Copy, Default)]
pub struct GlTexLevel {
    pub level: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub base_format: u32,
    pub internal_format: u32,
    pub border: u32,
    /// The image's detail texture rather than a level of its pyramid.
    pub detail: bool,
    pub read_texel: bool,
    /// Border indices range from -1 through width/height, inclusive.
    pub read_border_texel: bool,
    pub read_texel_3d: bool,
    /// Each signed axis ranges from -1 through its interior dimension.
    pub read_border_texel_3d: bool,
}

#[derive(Clone, Copy, Default)]
pub struct GlTexPyramid<'a> {
    pub levels: &'a [GlTexLevel],
    pub base_level: u32,
    pub max_level: u32,
    pub min_lod: f64,
    pub max_lod: f64,
}

/// One family-owned level-zero image and its texture object's sampler.
/// volume selects 3D sampling, one_dimensional S-only sampling; they are
/// mutually exclusive. Dimensions count interior texels. Texels return
/// canonical RRGGBBAA. Immutable through each synchronous draw.
#[derive(Clone, Copy, Default)]
pub struct GlTexImage<'a> {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub base_format: u32,
    pub border: u32,
    pub min_filter: u32,
    pub mag_filter: u32,
    pub wrap_s: u32,
    pub wrap_t: u32,
    pub wrap_r: u32,
    pub border_color: [f32; 4],
    pub read_texel: bool,
    pub read_texel_3d: bool,
    pub volume: bool,
    pub one_dimensional: bool,
    /// None selects the level-zero image fields.
    pub pyramid: Option<GlTexPyramid<'a>>,
    /// Required for a 2D image with border == 1; outside the interior only.
    pub read_border_texel: bool,
    /// Positive per-axis derivative factors; zero preserves identity.
    pub lod_scale_s: f64,
    pub lod_scale_t: f64,
    /// Texels::post_filter() applies to the expanded RGBA after filtering,
    /// before the texture environment.
    pub post_filter: bool,
    /// Required for a volume with border == 1, including depth one.
    pub read_border_texel_3d: bool,
    /// Positive volume R derivative factor; zero preserves identity.
    pub lod_scale_r: f64,
    /// Texels::filter_weight() replaces the public default curve; lambda is
    /// already clamped.
    pub read_filter_weight: bool,
    /// Level-zero detail image; L <= 0, mode ADD or MODULATE.
    pub detail_image: Option<GlTexLevel>,
    pub detail_level: i32,
    pub detail_mode: u32,
    /// Zero disables comparison. SGIX operators compare depth to projected R.
    pub shadow_compare: u32,
    pub shadow_ambient: f64,
    /// GL_SGIX_texture_coordinate_clamp: with max_clamp set, a CLAMP axis
    /// clamps to [0, max_clamp_*] instead of [0, 1].
    pub max_clamp: bool,
    pub max_clamp_s: f64,
    pub max_clamp_t: f64,
    pub max_clamp_r: f64,
}

/// Where a draw's texels come from: the family's texture storage. `level`
/// is one of the image's pyramid levels or its detail image, or None for
/// the level-zero image itself. A read outside the storage is None. Bands
/// of one draw read through it at once.
pub trait Texels: Sync {
    fn texel(&self, level: Option<&GlTexLevel>, x: u32, y: u32) -> Option<u32>;
    fn border_texel(&self, level: Option<&GlTexLevel>, x: i32, y: i32) -> Option<u32>;
    fn texel_3d(&self, level: Option<&GlTexLevel>, x: u32, y: u32, z: u32) -> Option<u32>;
    fn border_texel_3d(&self, level: Option<&GlTexLevel>, x: i32, y: i32, z: i32) -> Option<u32>;
    /// The image's post-filter (GlTexImage::post_filter).
    fn post_filter(&self, rgba: &mut [f64; 4]) {
        let _ = rgba;
    }
    /// The detail and sharpen weight at lambda (GlTexImage::read_filter_weight).
    fn filter_weight(&self, lambda: f64) -> Option<f64> {
        let _ = lambda;
        None
    }
}

/// A draw without a texture reads no texels.
pub struct NoTexels;

impl Texels for NoTexels {
    fn texel(&self, _: Option<&GlTexLevel>, _: u32, _: u32) -> Option<u32> {
        None
    }
    fn border_texel(&self, _: Option<&GlTexLevel>, _: i32, _: i32) -> Option<u32> {
        None
    }
    fn texel_3d(&self, _: Option<&GlTexLevel>, _: u32, _: u32, _: u32) -> Option<u32> {
        None
    }
    fn border_texel_3d(&self, _: Option<&GlTexLevel>, _: i32, _: i32, _: i32) -> Option<u32> {
        None
    }
}

/// Quantized scalar channels, each normalized by its own declared maximum.
#[derive(Clone, Copy, Default)]
pub struct RasterColour {
    /// R, G, B, A; absent alpha is canonical zero
    pub component: [u32; 4],
}

/// The family's planes as one thread of a draw reaches them, in plane
/// pixels. target is one of the drawable's colour targets, which only the
/// family interprets; a colour is the rules' color_bits a channel, red
/// highest. Only the methods RasterTarget's flags name are called; the
/// planes are shared, so a store goes through interior mutability.
pub trait Planes {
    fn store_colour(&self, target: ColourTarget, x: i32, y: i32, colour: u32) {
        let _ = (target, x, y, colour);
        unreachable!("store_colour without RasterTarget::store_colour");
    }
    /// The colour stored at (x, y), or in colour-index mode the index.
    fn load_colour(&self, target: ColourTarget, x: i32, y: i32) -> u32 {
        let _ = (target, x, y);
        unreachable!("load_colour without RasterTarget::load_colour");
    }
    fn store_index(&self, target: ColourTarget, x: i32, y: i32, index: u32, mask: u32) {
        let _ = (target, x, y, index, mask);
        unreachable!("store_index without RasterTarget::store_index");
    }
    fn store_rgba(&self, target: ColourTarget, x: i32, y: i32, colour: &RasterColour) {
        let _ = (target, x, y, colour);
        unreachable!("store_rgba without RasterTarget::store_rgba");
    }
    /// The colour at (x, y); None where it cannot be read.
    fn load_rgba(&self, target: ColourTarget, x: i32, y: i32) -> Option<RasterColour> {
        let _ = (target, x, y);
        unreachable!("load_rgba without RasterTarget::load_rgba");
    }
    fn load_depth(&self, x: i32, y: i32) -> u32;
    fn store_depth(&self, x: i32, y: i32, depth: u32);
    fn load_stencil(&self, x: i32, y: i32) -> u32 {
        let _ = (x, y);
        unreachable!("load_stencil without RasterTarget::stencil");
    }
    fn store_stencil(&self, x: i32, y: i32, stencil: u32) {
        let _ = (x, y, stencil);
        unreachable!("store_stencil without RasterTarget::stencil");
    }
    /// A fog factor the family supplies, per draw rather than GL state.
    fn fog_factor(&self, window_z: f32, eye_z: f32) -> f32 {
        let _ = (window_z, eye_z);
        unreachable!("fog_factor without RasterTarget::fog_factor");
    }
}

/// The planes a draw writes, as the family lends them for one draw: whole,
/// or split so that each band of a banded draw has its own rows.
pub trait PlaneStore: Send {
    /// The planes, for drawing on this thread.
    fn planes(&mut self) -> Box<dyn Planes + '_>;
    /// One store per band, band b holding the rows SoftDraw::row_in_band()
    /// gives it.
    fn split(&mut self, bands: u32) -> Vec<Box<dyn PlaneStore + '_>>;
}

/// The setup of a draw, before it touches a plane.
pub struct NoPlanes;

impl Planes for NoPlanes {
    fn load_depth(&self, _: i32, _: i32) -> u32 {
        unreachable!("a pixel access before the draw lent its planes");
    }
    fn store_depth(&self, _: i32, _: i32, _: u32) {
        unreachable!("a pixel access before the draw lent its planes");
    }
}

/// What the family's planes are and offer, for one draw. The flags say
/// which Planes methods it answers; load_depth and store_depth always.
#[derive(Clone, Default)]
pub struct RasterTarget<'a> {
    pub store_colour: bool,
    pub load_colour: bool,
    pub store_index: bool,
    pub store_rgba: bool,
    pub load_rgba: bool,
    /// A stencil buffer: load_stencil and store_stencil.
    pub stencil: bool,
    /// Planes::fog_factor() replaces the built-in fog modes.
    pub fog_factor: bool,
    /// Plane row of the display's top line, for a screen-anchored dither.
    pub screen_top: i32,
    /// Whether plane rows below the top line have larger y.
    pub screen_rows_down: bool,
    /// Current binding; resolved once by the family before drawing.
    pub texture: Option<GlTexImage<'a>>,
    /// An opaque polygon stipple: a fragment whose stipple bit is clear
    /// takes stipple_background (or stipple_background_index).
    pub stipple_opaque: bool,
    pub stipple_background: [f32; 4],
    pub stipple_background_index: u32,
    /// Bands run serially.
    pub concurrent_pixels: bool,
    pub band_min_pixels: u32,
    pub mixed_frame: bool,
}

pub type RasterTexturePath = u32;
pub const SGI_RASTER_TEXTURE_SPANS: RasterTexturePath = 0;
pub const SGI_RASTER_TEXTURE_FIXED: RasterTexturePath = 1;
pub const SGI_RASTER_TEXTURE_DOUBLE: RasterTexturePath = 2;

// ---------------------------------------------------------------------
// The draw itself. The fixed-point texture path is in raster_fixed_texture.rs.
// ---------------------------------------------------------------------

pub fn fmax(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}
pub fn fmin(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// The drawable's window: its origin and the bounds of the pixels it owns,
/// in plane pixels, and the rectangles themselves.
#[derive(Clone, Copy, Default)]
pub struct SoftWindow {
    pub ox: i32,
    pub oy: i32,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    pub clip: ClipList,
}

/// Once-a-draw reports, shared by the bands of one draw.
#[derive(Default)]
pub struct SoftReports {
    pub texture: std::sync::atomic::AtomicU32,
    pub rgba: std::sync::atomic::AtomicBool,
}

/// One draw's fixed inputs, the part of a draw its bands share.
#[derive(Clone, Copy)]
pub struct SoftShared<'a> {
    pub rules: &'a RasterRules,
    pub t: &'a RasterTarget<'a>,
    pub st: &'a GlDrawState,
    pub texels: &'a dyn Texels,
    pub reports: &'a SoftReports,
    pub win: SoftWindow,
    pub targets: [ColourTarget; SGI_GL_MAX_TARGETS as usize],
    pub ntargets: u32,
    /// Back-facing polygons take the vertices' back colours.
    pub back_colors: bool,
    /// Fragments blend with the colour they replace.
    pub blend: bool,
    /// Fragments combine with it by the logical operation instead.
    pub logic_op: bool,
    /// In RGBA mode, the packed colour's bits glColorMask lets through.
    pub colour_mask: u32,
    /// The stored colour is dithered: GL_DITHER and the rules' dither.
    pub dither: bool,
    pub texture: Option<&'a GlTexImage<'a>>,
    pub texture_base: Option<&'a GlTexLevel>,
    pub texture_last: u32,
    pub point_size: u32,
    pub line_width: u32,
    pub point_smooth: bool,
    pub line_smooth: bool,
    pub smooth_point_size: f64,
    pub smooth_line_width: f64,
    pub polygon_smooth: bool,
    pub polygon_stipple: Option<&'a [u32; 32]>,
    pub rgba_mask: u32,
    pub rgba_read: bool,
    /// Filled triangles sample in fixed point.
    pub fixed: Option<&'a SoftFixedTex>,
    /// Of bands > 1, the rows this pass draws (SoftDraw::row_in_band()).
    pub band: u32,
    pub bands: u32,
}

/// A draw as one thread runs it: the shared inputs, and the planes and
/// texel cache that are this thread's alone. It derefs to the shared part,
/// so the drawing code reads `self.rules` whichever part a field is in.
#[derive(Clone, Copy)]
pub struct SoftDraw<'a> {
    pub shared: SoftShared<'a>,
    /// The planes; NoPlanes until the draw starts on them.
    pub px: &'a dyn Planes,
    /// This thread's texel cache, when texels read in this draw are kept.
    pub texels_cache: Option<&'a std::cell::RefCell<SoftTexelCache>>,
}

impl<'a> std::ops::Deref for SoftDraw<'a> {
    type Target = SoftShared<'a>;
    fn deref(&self) -> &SoftShared<'a> {
        &self.shared
    }
}

impl std::ops::DerefMut for SoftDraw<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shared
    }
}

impl<'a> SoftDraw<'a> {
    /// A draw under rules into target, in state st, with nothing else
    /// decided yet.
    pub fn new(rules: &'a RasterRules, t: &'a RasterTarget<'a>, st: &'a GlDrawState, reports: &'a SoftReports) -> Self {
        SoftDraw {
            shared: SoftShared {
                rules,
                t,
                st,
                texels: &NoTexels,
                reports,
                win: SoftWindow::default(),
                targets: [0; SGI_GL_MAX_TARGETS as usize],
                ntargets: 0,
                back_colors: false,
                blend: false,
                logic_op: false,
                colour_mask: 0,
                dither: false,
                texture: None,
                texture_base: None,
                texture_last: 0,
                point_size: 0,
                line_width: 0,
                point_smooth: false,
                line_smooth: false,
                smooth_point_size: 0.0,
                smooth_line_width: 0.0,
                polygon_smooth: false,
                polygon_stipple: None,
                rgba_mask: 0,
                rgba_read: false,
                fixed: None,
                band: 0,
                bands: 0,
            },
            px: &NoPlanes,
            texels_cache: None,
        }
    }

    /// The draw's texture, which a caller has checked is there.
    pub fn image(&self) -> &'a GlTexImage<'a> {
        self.texture.expect("a textured draw")
    }

    /// The base level of the draw's pyramid, which a caller has checked is
    /// there.
    pub fn base_level(&self) -> &'a GlTexLevel {
        self.texture_base.expect("a texture pyramid")
    }

    /// The draw on planes px.
    pub fn with_planes<'p>(&self, px: &'p dyn Planes) -> SoftDraw<'p>
    where
        'a: 'p,
    {
        SoftDraw { shared: self.shared, px, texels_cache: self.texels_cache }
    }
}

impl<'a> SoftShared<'a> {
    /// The draw on this thread: on planes px, keeping texels in cache when
    /// it is textured.
    pub fn on<'p>(&self, px: &'p dyn Planes, cache: &'p std::cell::RefCell<SoftTexelCache>) -> SoftDraw<'p>
    where
        'a: 'p,
    {
        SoftDraw { shared: *self, px, texels_cache: self.texture.map(|_| cache) }
    }
}

/// A vertex after transformation, ready to rasterize.
#[derive(Clone, Copy)]
pub struct SoftVertex {
    /// window position in subpixels
    pub x: i64,
    pub y: i64,
    /// normalized depth; [0, 1] inside the view volume
    pub z: f64,
    /// 1 / clip w, for perspective-correct colour
    pub inv_w: f64,
    /// the rules' depth divisor over clip w
    pub depth_w: f64,
    pub rgba: [f64; 4],
    pub texcoord: [f64; 4],
    /// the colour index, in colour-index mode
    pub index: f64,
    /// the fog coordinate, |z| in eye coordinates
    pub fog: f64,
    /// Each client clip plane's value at the vertex, kept where >= 0.
    pub clip: [f64; SGI_GL_CLIP_PLANES as usize],
}

impl Default for SoftVertex {
    fn default() -> Self {
        SoftVertex {
            x: 0,
            y: 0,
            z: 0.0,
            inv_w: 0.0,
            depth_w: 0.0,
            rgba: [0.0; 4],
            texcoord: [0.0; 4],
            index: 0.0,
            fog: 0.0,
            clip: [0.0; SGI_GL_CLIP_PLANES as usize],
        }
    }
}

/// A fragment's values as rasterization leaves them (OpenGL 1.2.1, 3.1),
/// before the per-fragment operations of chapter 4: its colour, each
/// component in [0, 1], and in colour-index mode its index.
#[derive(Clone, Copy, Default)]
pub struct SoftFrag {
    pub rgba: [f64; 4],
    pub texcoord: [f64; 4],
    pub texture_rho: f64,
    /// With fixed set, the texel the fixed-point sampler filtered for the
    /// fragment, which the texture environment applies in place of
    /// sampling texcoord.
    pub fixed: bool,
    pub texel: [f64; 4],
    pub index: f64,
    /// the fog coordinate (3.10)
    pub fog: f64,
}

 // rows interleave in runs of 8

thread_local! {
    /// A band other than the first repeats the primitive work: say it once.
    pub static SOFT_BAND_QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl GlDrawState {
    pub fn window(&self, win: &mut SoftWindow) -> bool {
        let d = &self.drawable;

        if d.bound == 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: draw with no window bound dropped");
            return false;
        }
        if d.clip.n == 0 {
            // Other windows cover all of this one: nothing to draw.
            return false;
        }
        win.ox = d.origin[0];
        win.oy = d.origin[1];
        win.x0 = d.clip.r[0].x0;
        win.y0 = d.clip.r[0].y0;
        win.x1 = d.clip.r[0].x1;
        win.y1 = d.clip.r[0].y1;
        for i in 1..std::cmp::min(d.clip.n as usize, SGI_CLIP_MAX) {
            win.x0 = win.x0.min(d.clip.r[i].x0);
            win.y0 = win.y0.min(d.clip.r[i].y0);
            win.x1 = win.x1.max(d.clip.r[i].x1);
            win.y1 = win.y1.max(d.clip.r[i].y1);
        }
        win.clip = d.clip;
        true
    }
}

impl RasterRules {
    pub fn rules_supported(&self) -> bool {
        self.subpixel_bits >= 1 && self.subpixel_bits <= 12 &&
            self.color_bits >= 1 && self.color_bits <= 10 &&
            self.depth_bits >= 1 && self.depth_bits <= 32 &&
            (!self.depth_float || self.depth_bits == 24) &&
            (!self.depth_row_given ||
             (self.depth_row[0].is_finite() && self.depth_row[1].is_finite() &&
              self.depth_row[2].is_finite() && self.depth_row[3].is_finite())) &&
            (!self.depth_divisor_given ||
             (self.depth_row_given &&
              self.depth_divisor[0].is_finite() && self.depth_divisor[1].is_finite() &&
              self.depth_divisor[2].is_finite() && self.depth_divisor[3].is_finite())) &&
            self.depth_slack >= 0.0 && self.depth_slack < 1.0 &&
            self.stencil_bits <= 16 && self.alpha_bits <= 10 &&
            self.alpha_test_bits <= 16 &&
            (self.rounding as u32) <= SGI_ROUND_HALF_AWAY &&
            (self.tie_rule as u32) == SGI_TIE_TOP_LEFT_BOTTOM_ORIGIN &&
            (self.arith as u32) == SGI_ARITH_DOUBLE &&
            (self.line_rule as u32) == SGI_LINE_COVERAGE_DOUBLE &&
            (self.dither as u32) <= SGI_DITHER_BAYER4_TOP_ROWS &&
            (self.smooth_rule as u32) <= SGI_SMOOTH_TWO_PIXEL_FILTER &&
            (self.dither as u32 == SGI_DITHER_NONE ||
             self.dither_anchor as u32 == SGI_DITHER_SCREEN)
    }

    pub fn check_rules(&self) -> bool {
        if !self.rules_supported() {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: raster rules are not implemented; nothing drawn");
            return false;
        }
        true
    }
}

impl RasterTarget<'_> {
    /// Whether the target can take the colour-index work a draw state asks.
    pub fn soft_check_index(&self, index_bits: u32) -> bool {
        if index_bits != 0 && !self.store_index {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: colour-index drawing is not implemented for these planes; nothing drawn");
            return false;
        }
        true
    }
}

impl RasterRules {
    /// x rounded to an integer by the rules' rounding.
    pub fn round(&self, x: f64) -> i64 {
        let rounding = self.rounding as u32;
        if rounding != SGI_ROUND_FLOOR && rounding != SGI_ROUND_HALF_UP && rounding != SGI_ROUND_HALF_EVEN {
            // Round half away from zero, without a libm call: x less its truncation is exact, so
            // a fraction of a half or more rounds away from zero.
            let t = x as i64;
            let frac = x - t as f64;
            return t + (frac >= 0.5) as i64 - (frac <= -0.5) as i64;
        }
        let f = x.floor();
        let frac = x - f;
        match rounding {
            SGI_ROUND_FLOOR => f as i64,
            SGI_ROUND_HALF_UP => f as i64 + (frac >= 0.5) as i64,
            _ => {
                if frac == 0.5 {
                    return f as i64 + ((f as i64) & 1);
                }
                f as i64 + (frac > 0.5) as i64
            }
        }
    }

    /// The largest stored depth.
    pub fn depth_max(&self) -> u32 {
        u32::MAX.wrapping_shr(32u32.wrapping_sub(self.depth_bits as u32))
    }

    /// The stored code of window depth z in [0, 1]: an integer fraction of
    /// depth_max, or with depth_float the board's depth word for 2S z
    /// (depth_word::word_for_fraction()).
    pub fn depth_code(&self, z: f64) -> u32 {
        if !self.depth_float {
            return self.round(z * self.depth_max() as f64) as u32;
        }
        super::depth_word::word_for_fraction(z)
    }

    /// A polygon offset (the sum of slope, units and bias, in window depth)
    /// split into what adds to the window depth before the code and the steps
    /// that add to the code after it. With depth_float the board adds the
    /// offset field to the encoded word (depth_word::word_with_offset(),
    /// INFERRED placement), and the field's step is 2^-24 of the offset
    /// carried here (ODSY_POLYGON_OFFSET_SCALE), so none of it adds to window
    /// depth.
    pub fn split_offset(&self, offset: f64) -> (f64, i64) {
        if !self.depth_float {
            return (offset, 0);
        }
        (0.0, (offset * (1u64 << 24) as f64).round() as i64)
    }

    /// [`depth_code`](Self::depth_code) of z, then `steps` of polygon offset.
    pub fn depth_code_offset(&self, z: f64, steps: i64) -> u32 {
        let code = self.depth_code(z);

        if self.depth_float {
            super::depth_word::word_with_offset(code, steps)
        } else {
            code
        }
    }

    /// A component clamped to [0, 1] and scaled to the rules' colour bits, a
    /// fraction of a step rounded half up, whatever the rules' rounding: the
    /// sum is truncated (OpenGL 1.2.1, 2.13.9).
    pub fn channel(&self, c: f64) -> u32 {
        let max = (1u32.wrapping_shl(self.color_bits as u32)).wrapping_sub(1);

        if !(c > 0.0) {
            return 0;
        }
        if c >= 1.0 {
            return max;
        }
        (c * max as f64 + 0.5) as u32
    }

    /// Incoming alpha precision is independent of absent or narrower stored A.
    pub fn alpha_max(&self) -> u32 {
        (1u32.wrapping_shl(std::cmp::max(2, std::cmp::max(self.color_bits, self.alpha_bits)) as u32)).wrapping_sub(1)
    }

    pub fn alpha(&self, value: f64) -> u32 {
        let max = self.alpha_max();

        if !(value > 0.0) {
            0
        } else if value >= 1.0 {
            max
        } else {
            (value * max as f64 + 0.5) as u32
        }
    }

    pub fn soft_alpha_test(&self, value: f64) -> u32 {
        let max = if self.alpha_test_bits != 0 {
            (1u32.wrapping_shl(self.alpha_test_bits as u32)).wrapping_sub(1)
        } else {
            self.alpha_max()
        };

        if !(value > 0.0) {
            0
        } else if value >= 1.0 {
            max
        } else {
            (value * max as f64 + 0.5) as u32
        }
    }

    /// Internal nearest conversion precedes storage truncation, even on clears.
    pub fn stored_alpha(&self, value: f64) -> u32 {
        let m = self.alpha_max();
        if m == 0 {
            return 0;
        }
        self.alpha(value)
            .wrapping_mul((1u32.wrapping_shl(self.alpha_bits as u32)).wrapping_sub(1))
            / m
    }

    pub fn rgba_mask(&self, mask: u32) -> u32 {
        mask & if self.alpha_bits != 0 { SGI_GL_COLOUR_MASK_ALL } else { SGI_GL_COLOUR_MASK_ALPHA - 1 }
    }

    /// Reject unsupported capabilities before the operation can write any plane.
    pub fn rgba_capable(&self, t: &RasterTarget, read: bool) -> bool {
        if (self.alpha_bits != 0 && !t.store_rgba) ||
            (t.store_rgba && read && !t.load_rgba) ||
            (!t.store_rgba && t.load_rgba)
        {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: inconsistent RGBA callbacks; operation dropped");
            return false;
        }
        true
    }

    /// Read every selected destination before any colour/depth/stencil write.
    pub fn rgba_load(&self, px: &dyn Planes, targets: &[ColourTarget], count: u32, x: i32, y: i32, dest: &mut [RasterColour; SGI_GL_MAX_TARGETS as usize], reported: &std::sync::atomic::AtomicBool) -> bool {
        for i in 0..count as usize {
            let loaded = px.load_rgba(targets[i], x, y);
            let mut valid = loaded.is_some();
            if let Some(c) = loaded {
                dest[i] = c;
            }

            let mut k = 0;
            while valid && k < 4 {
                let bits = if k == 3 { self.alpha_bits } else { self.color_bits };

                // An absent A is ignored, including its returned representation.
                valid = bits == 0 || dest[i].component[k] < (1u32 << bits);
                k += 1;
            }
            if !valid {
                // Bands of one draw share the flag (soft_bands_run()).
                if !reported.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: RGBA destination read failed; pixel writes dropped");
                }
                return false;
            }
            if self.alpha_bits == 0 {
                dest[i].component[3] = 0;
            }
        }
        true
    }

    /// A colour, each component clamped and rounded to the rules' bits.
    pub fn rgb(&self, c: &[f64]) -> u32 {
        (self.channel(c[0]).wrapping_shl(2 * self.color_bits as u32)) |
            (self.channel(c[1]).wrapping_shl(self.color_bits as u32)) |
            self.channel(c[2])
    }
}

/// SGI_DITHER_BAYER4_TOP_ROWS's thresholds, in sixteenths of a step.
pub static SOFT_BAYER4: [[u8; 4]; 4] = [
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
];

impl SoftDraw<'_> {
    /// A fragment's colour at plane pixel (x, y) as the planes store it:
    /// dithered while the draw dithers, otherwise rounded as RasterRules::rgb()
    /// rounds. A clamped component of 0 or 1 is exact either way, and one
    /// below 1 stays below max + 1 with the largest threshold added.
    pub fn store_rgb(&self, x: i32, y: i32, c: &[f64]) -> u32 {
        let bits = self.rules.color_bits as u32;
        let max: u32 = (1u32 << bits) - 1;
        let mut out: u32 = 0;

        if !self.dither {
            return self.rules.rgb(c);
        }
        let row: i64 = if self.t.screen_rows_down {
            y as i64 - self.t.screen_top as i64
        } else {
            self.t.screen_top as i64 - y as i64
        };
        let b = SOFT_BAYER4[(row & 3) as usize][(x & 3) as usize] as f64 / 16.0;
        for k in 0..3usize {
            let v: u32;

            if !(c[k] > 0.0) {
                v = 0;
            } else if c[k] >= 1.0 {
                v = max;
            } else {
                v = (c[k] * max as f64 + b).floor() as u32;
            }
            out |= v.wrapping_shl((2 - k as u32) * bits);
        }
        out
    }
}

impl SoftVertex {
    pub fn edge(&self, b: &SoftVertex, px: i64, py: i64) -> i64 {
        (b.x.wrapping_sub(self.x)).wrapping_mul(py.wrapping_sub(self.y))
            .wrapping_sub((b.y.wrapping_sub(self.y)).wrapping_mul(px.wrapping_sub(self.x)))
    }

    /// SGI_TIE_TOP_LEFT_BOTTOM_ORIGIN: a counter-clockwise edge owns the pixel
    /// centres on it when it is a left edge (running down) or a top edge
    /// (horizontal, running left), with rows counted up from the bottom of the
    /// plane, so two triangles sharing an edge never both draw its centres, as
    /// OpenGL 1.2.1, 3.5.1 requires.
    pub fn owns(&self, b: &SoftVertex, e: i64) -> bool {
        if e != 0 {
            return e > 0;
        }
        b.y < self.y || (b.y == self.y && b.x < self.x)
    }

    /// Narrow [*lo, *hi], pixel steps along a row from a start whose edge value
    /// is e, to the steps where SoftVertex::owns() holds for the edge from a to b,
    /// whose value moves by step a pixel. The edge function is linear along the
    /// row, so the owned steps are one run: e + k * step >= need, where need is
    /// 0 for an edge that owns the centres on it and 1 for one that does not.
    pub fn owned_run(&self, b: &SoftVertex, e: i64, step: i64, lo: &mut i64, hi: &mut i64) {
        let need: i64 = if self.owns(b, 0) { 0 } else { 1 };
        let d = need.wrapping_sub(e);

        if step > 0 {
            // The first k with k * step >= d, rounding the quotient up.
            let v = if d > 0 { (d + step - 1) / step } else { -(-d / step) };
            *lo = std::cmp::max(*lo, v);
        } else if step < 0 {
            // The last k with k * -step <= -d, rounding the quotient down.
            let v = if -d >= 0 { -d / -step } else { -((d - step - 1) / -step) };
            *hi = std::cmp::min(*hi, v);
        } else if d > 0 {
            *hi = *lo - 1;
        }
    }
}

/// Whether value passes a test comparing it with ref by func, the
/// comparison minus GL_NEVER: the depth test's incoming depth against the
/// stored one (4.1.5), the alpha test's alpha against its reference
/// (4.1.4), the stencil test's masked reference against the masked stored
/// value (4.1.5).
pub fn soft_compare(func: u32, value: u32, ref_: u32) -> bool {
    match func {
        SGI_GL_NEVER => false,
        SGI_GL_LESS => value < ref_,
        SGI_GL_EQUAL => value == ref_,
        SGI_GL_LEQUAL => value <= ref_,
        SGI_GL_GREATER => value > ref_,
        SGI_GL_NOTEQUAL => value != ref_,
        SGI_GL_GEQUAL => value >= ref_,
        _ => true,
    }
}

impl RasterRules {
    pub fn alpha_test(&self, func: u32, alpha: f64, ref_: f64) -> bool {
        soft_compare(func, self.soft_alpha_test(alpha), self.soft_alpha_test(ref_))
    }

    /// Whether the planes have a stencil buffer, and its largest value.
    pub fn stencil_max(&self, t: &RasterTarget, max: &mut u32) -> bool {
        *max = (1u32.wrapping_shl(self.stencil_bits as u32)).wrapping_sub(1);
        self.stencil_bits != 0 && t.stencil
    }
}

impl GlDrawState {
    /// One of glStencilOp's operations on the stencil value at (x, y), the
    /// result written under the stencil write mask (4.1.5, 4.2.2). GL_INCR
    /// and GL_DECR clamp to the buffer's range.
    pub fn stencil_op(&self, px: &dyn Planes, x: i32, y: i32, op: u32, max: u32, ref_: u32) {
        let old = px.load_stencil(x, y) & max;
        let mask = self.stencil_write_mask & max;
        let value: u32;

        match op {
            SGI_GL_KEEP => return,
            SGI_GL_REPLACE => value = ref_,
            SGI_GL_INCR => value = if old < max { old + 1 } else { max },
            SGI_GL_DECR => value = if old != 0 { old - 1 } else { 0 },
            SGI_GL_INVERT => value = !old & max,
            _ => value = 0,
        }
        px.store_stencil(x, y, (old & !mask) | (value & mask));
    }
}

impl RasterRules {
    pub fn depth_stencil(&self, st: &GlDrawState, t: &RasterTarget, px: &dyn Planes, x: i32, y: i32, z: f64, steps: i64) -> bool {
        let mut smax: u32 = 0;
        let mut sref: u32 = 0;

        if !(0.0..=1.0).contains(&z) {
            return false;
        }
        // The reference clamps to the buffer's range; with no stencil buffer
        // the test passes and nothing is written.
        let stencil = st.stencil_test != 0 && self.stencil_max(t, &mut smax);
        if stencil {
            let vmask = st.stencil_value_mask;

            sref = if st.stencil_ref < 0 { 0 } else { std::cmp::min(st.stencil_ref as u32, smax) };
            if !soft_compare(
                st.stencil_func,
                sref & vmask,
                px.load_stencil(x, y) & smax & vmask,
            ) {
                st.stencil_op(px, x, y, st.stencil_op[0], smax, sref);
                return false;
            }
        }
        let zq = self.depth_code_offset(z, steps);
        if st.depth_test != 0 {
            if !soft_compare(st.depth_func, zq, px.load_depth(x, y)) {
                if stencil {
                    st.stencil_op(px, x, y, st.stencil_op[1], smax, sref);
                }
                return false;
            }
            if st.depth_mask != 0 {
                px.store_depth(x, y, zq);
            }
        }
        // A disabled depth test passes (4.1.5).
        if stencil {
            st.stencil_op(px, x, y, st.stencil_op[2], smax, sref);
        }
        true
    }
}

/// One blend factor (OpenGL 1.2.1, tables 4.1 and 4.2) for component k,
/// red to alpha, from the source s, destination d and constant colour c.
pub fn soft_blend_factor(factor: u32, k: usize, s: &[f64; 4], d: &[f64; 4], c: &[f32; 4]) -> f64 {
    match factor {
        1 => 1.0,
        SGI_GL_SRC_COLOR => s[k],
        SGI_GL_ONE_MINUS_SRC_COLOR => 1.0 - s[k],
        SGI_GL_DST_COLOR => d[k],
        SGI_GL_ONE_MINUS_DST_COLOR => 1.0 - d[k],
        SGI_GL_SRC_ALPHA => s[3],
        SGI_GL_ONE_MINUS_SRC_ALPHA => 1.0 - s[3],
        SGI_GL_DST_ALPHA => d[3],
        SGI_GL_ONE_MINUS_DST_ALPHA => 1.0 - d[3],
        SGI_GL_SRC_ALPHA_SATURATE => if k == 3 { 1.0 } else { fmin(s[3], 1.0 - d[3]) },
        SGI_GL_CONSTANT_COLOR => c[k] as f64,
        SGI_GL_ONE_MINUS_CONSTANT_COLOR => 1.0 - c[k] as f64,
        SGI_GL_CONSTANT_ALPHA => c[3] as f64,
        SGI_GL_ONE_MINUS_CONSTANT_ALPHA => 1.0 - c[3] as f64,
        _ => 0.0,
    }
}

impl GlDrawState {
    pub fn blend_component(&self, k: usize, src: &[f64; 4], dst: &[f64; 4]) -> f64 {
        let sf = soft_blend_factor(self.blend_src, k, src, dst, &self.blend_color);
        let df = soft_blend_factor(self.blend_dst, k, src, dst, &self.blend_color);

        match self.blend_equation {
            SGI_GL_MIN => fmin(src[k], dst[k]),
            SGI_GL_MAX => fmax(src[k], dst[k]),
            SGI_GL_FUNC_SUBTRACT => src[k] * sf - dst[k] * df,
            SGI_GL_FUNC_REVERSE_SUBTRACT => dst[k] * df - src[k] * sf,
            _ => src[k] * sf + dst[k] * df,
        }
    }
}

impl RasterRules {
    /// Blend the packed source colour src, with alpha alpha, and the packed
    /// destination dst (4.1.6), each component the fixed point of the rules'
    /// colour bits taken as a fraction of its largest value. The planes hold
    /// no alpha, so the destination's alpha is 1. The result is clamped to
    /// [0, 1] by the subsequent colour conversion.
    pub fn soft_blend_rgb(&self, st: &GlDrawState, src: u32, alpha: f64, dst: u32, out: &mut [f64; 3]) {
        let bits = self.color_bits as u32;
        let max: f64 = ((1u32 << bits) - 1) as f64;
        let mut s = [0.0f64; 4];
        let mut d = [0.0f64; 4];

        for k in 0..3usize {
            let shift = (2 - k as u32) * bits;

            s[k] = ((src >> shift) & (max as u32)) as f64 / max;
            d[k] = ((dst >> shift) & (max as u32)) as f64 / max;
        }
        s[3] = alpha;
        d[3] = 1.0;
        for k in 0..3usize {
            out[k] = st.blend_component(k, &s, &d);
        }
    }

}

impl SoftDraw<'_> {
    pub fn blend(&self, x: i32, y: i32, src: u32, alpha: f64, dst: u32) -> u32 {
        let mut out = [0.0f64; 3];

        self.rules.soft_blend_rgb(&self.st, src, alpha, dst, &mut out);
        self.store_rgb(x, y, &out)
    }
}

/// glLogicOp's operation (4.1.8, table 4.3) on the source s and the
/// destination d, bit by bit.
pub fn soft_logic_op(op: u32, s: u32, d: u32) -> u32 {
    x_logic_op(op.wrapping_sub(SGI_GL_CLEAR), s, d)
}

impl RasterRules {
    /// RGB retains its existing arithmetic; spatial alpha dithering is omitted.
    pub fn rgba_value(&self, rgb: u32, alpha: f64) -> RasterColour {
        let mut c = RasterColour::default();
        let max = (1u32 << self.color_bits) - 1;

        for k in 0..3usize {
            c.component[k] = (rgb >> ((2 - k as u32) * self.color_bits as u32)) & max;
        }
        c.component[3] = self.stored_alpha(alpha);
        c
    }
}

impl SoftDraw<'_> {
    pub fn rgba_result(&self, x: i32, y: i32, frag: &SoftFrag, dest: &RasterColour) -> RasterColour {
        let mut out = RasterColour::default();
        let rules = &*self.rules;
        let max = (1u32 << rules.color_bits) - 1;
        let alpha_max = rules.alpha_max();
        let alpha_q = rules.alpha(frag.rgba[3]);
        let alpha = alpha_q as f64 / alpha_max as f64;
        let rgb = if self.blend { rules.rgb(&frag.rgba) } else { self.store_rgb(x, y, &frag.rgba) };

        if !self.blend && !self.logic_op && self.rgba_mask == rules.rgba_mask(SGI_GL_COLOUR_MASK_ALL) {
            // RasterRules::rgba_value()'s stored alpha, RasterRules::alpha() of
            // alpha_q / alpha_max scaled to the stored bits, without the round
            // trip: alpha_q / alpha_max * alpha_max + 0.5 truncates to alpha_q
            // for every alpha_q up to 1023.
            let stored = (1u32.wrapping_shl(rules.alpha_bits as u32)).wrapping_sub(1);

            for k in 0..3usize {
                out.component[k] = (rgb >> ((2 - k as u32) * rules.color_bits as u32)) & max;
            }
            out.component[3] = if stored == alpha_max { alpha_q } else { alpha_q.wrapping_mul(stored) / alpha_max };
            return out;
        }
        out = rules.rgba_value(rgb, alpha);
        if self.blend {
            let mut src = [0.0f64; 4];
            let mut dst = [0.0f64; 4];
            let mut blended = [0.0f64; 4];

            for k in 0..3usize {
                src[k] = out.component[k] as f64 / max as f64;
                dst[k] = dest.component[k] as f64 / max as f64;
            }
            src[3] = alpha;
            dst[3] = if rules.alpha_bits != 0 {
                dest.component[3] as f64 / ((1u32 << rules.alpha_bits) - 1) as f64
            } else {
                1.0
            };
            for k in 0..4usize {
                blended[k] = self.st.blend_component(k, &src, &dst);
            }
            out = rules.rgba_value(self.store_rgb(x, y, &blended), blended[3]);
        }
        for k in 0..4usize {
            let bits = if k == 3 { rules.alpha_bits } else { rules.color_bits } as u32;

            if self.logic_op {
                out.component[k] = soft_logic_op(self.st.logic_op_mode, out.component[k], dest.component[k])
                    & (1u32.wrapping_shl(bits)).wrapping_sub(1);
            }
            if self.rgba_mask & (1u32 << k) == 0 {
                out.component[k] = dest.component[k];
            }
        }
        out
    }
}

impl RasterRules {
    /// The bits of a packed colour of the rules' colour bits that a
    /// glColorMask channel mask lets through. The planes hold no alpha, so
    /// its bit has no effect.
    pub fn colour_mask(&self, mask: u32) -> u32 {
        let channel = (1u32 << self.color_bits) - 1;
        let mut out = 0u32;

        for k in 0..3u32 {
            if mask & (SGI_GL_COLOUR_MASK_RED << k) != 0 {
                out |= channel << ((2 - k) * self.color_bits as u32);
            }
        }
        out
    }

    /// The packed colour's bits, all of which a full mask writes.
    pub fn colour_bits(&self) -> u32 {
        self.colour_mask(SGI_GL_COLOUR_MASK_ALL)
    }
}

impl SoftDraw<'_> {
    /// Which of the operations that read the stored colour a draw state
    /// enables. The logical operation applies to a colour index while
    /// GL_INDEX_LOGIC_OP is enabled, and to an RGBA colour while
    /// GL_COLOR_LOGIC_OP is, or while blending is enabled with
    /// EXT_blend_logic_op's GL_LOGIC_OP equation; it replaces blending
    /// (4.1.8). A colour mask with some channels clear keeps the stored
    /// bits of those (4.2.2). Planes that cannot be read draw as if the
    /// operations were disabled and the mask were all ones, logged once for
    /// the draw. An RGBA draw is dithered while GL_DITHER is enabled under
    /// rules that name a dither; a colour index never is.
    pub fn setup_ops(&mut self) -> bool {
        let st = self.st;

        if st.index_bits != 0 {
            self.blend = false;
            self.logic_op = st.index_logic_op != 0;
        } else {
            self.logic_op = st.color_logic_op != 0 || (st.blend != 0 && st.blend_equation == SGI_GL_LOGIC_OP);
            self.blend = st.blend != 0 && !self.logic_op;
        }
        self.colour_mask = self.rules.colour_mask(st.colour_mask);
        self.dither = st.dither != 0 && self.rules.dither as u32 != SGI_DITHER_NONE;
        if st.index_bits == 0 {
            self.rgba_mask = self.rules.rgba_mask(st.colour_mask);
            self.rgba_read = self.ntargets != 0 && self.rgba_mask != 0 &&
                (self.blend || self.logic_op ||
                 self.rgba_mask != self.rules.rgba_mask(SGI_GL_COLOUR_MASK_ALL));
            if !self.rules.rgba_capable(&self.t, self.rgba_read) {
                return false;
            }
            if self.t.store_rgba {
                return true;
            }
        }
        if (self.blend || self.logic_op ||
            (st.index_bits == 0 && self.colour_mask != 0 && self.colour_mask != self.rules.colour_bits())) &&
            !self.t.load_colour
        {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: blending, logical operations and partial colour masks need planes that can be read back; drawn without them");
            self.blend = false;
            self.logic_op = false;
            self.colour_mask = self.rules.colour_bits();
        }
        true
    }

}

pub const SGI_SOFT_FIXED_LEVELS: usize = 14;
pub const SGI_SOFT_FIXED_FAILED: u32 = 0x80000000;

/// The fixed-point sampler's texels of one level for one draw, by interior
/// index: each the read tag in the high word and the packed RGBA8 texel,
/// components expanded, in the low (SoftDraw::fixed_texel()).
#[derive(Clone, Default)]
pub struct SoftFixedStore {
    pub cell: Vec<u64>,
    pub size: usize,
    pub plain: Vec<u64>,
}

thread_local! {
    pub static SOFT_TEXEL_CACHE: std::cell::RefCell<SoftTexelCache> = std::cell::RefCell::new(SoftTexelCache::new());
}

/// Run f with this thread's texel cache, begun for a new draw.
pub fn with_texel_cache<R>(f: impl FnOnce(&std::cell::RefCell<SoftTexelCache>) -> R) -> R {
    SOFT_TEXEL_CACHE.with(|cache| {
        cache.borrow_mut().begin();
        f(cache)
    })
}

thread_local! {
    pub static SOFT_TEXTURE_SELECTED: std::cell::Cell<SoftTexSelectionMemo> =
        std::cell::Cell::new(SoftTexSelectionMemo {
            serial: 0,
            rho: 0,
            selected: SoftTexSelection { first: 0, second: 0, filter: 0, alpha: 0.0, lambda: 0.0 },
        });
}

impl SoftDraw<'_> {
    /// Fog (3.10): the fog factor f from the fragment's fog coordinate c by
    /// the mode, clamped to [0, 1], blends an RGBA fragment's red, green and
    /// blue toward the fog colour, f C + (1 - f) Cf, and adds (1 - f) times
    /// the fog index to a colour index. A family's fog_factor, where set, gives
    /// f from the window depth and fog coordinate instead. GL_LINEAR with start
    /// equal to end divides by zero; the model takes f as 1 before the end and
    /// 0 from it, the limit of the equation from either side.
    pub fn fog(&self, frag: &mut SoftFrag, z: f64) {
        let st: &GlDrawState = &self.st;
        let c = frag.fog;
        let mut f: f64;
        let dens = st.fog_density as f64;

        if self.t.fog_factor {
            // A family's own curve replaces the mode; the blend is the same.
            f = self.px.fog_factor(z as f32, c as f32) as f64;
        } else if st.fog_mode == SGI_GL_LINEAR {
            f = if st.fog_end == st.fog_start {
                if c < st.fog_end as f64 { 1.0 } else { 0.0 }
            } else {
                (st.fog_end as f64 - c) / (st.fog_end as f64 - st.fog_start as f64)
            };
        } else if st.fog_mode == SGI_GL_EXP2 {
            f = (-(dens * c) * (dens * c)).exp();
        } else {
            f = (-dens * c).exp();
        }
        f = if f >= 1.0 { 1.0 } else if f > 0.0 { f } else { 0.0 };
        if st.index_bits != 0 {
            frag.index += (1.0 - f) * st.fog_index as f64;
            return;
        }
        for k in 0..3usize {
            frag.rgba[k] = f * frag.rgba[k] + (1.0 - f) * st.fog_color[k] as f64;
        }
    }
}

impl GlDrawState {
    /// Whether a point whose client clip plane values are d lies outside an
    /// enabled plane (2.11).
    pub fn clipped(&self, d: &[f64]) -> bool {
        for i in 0..SGI_GL_CLIP_PLANES as usize {
            if (self.clip_planes & (1u32 << i)) != 0 && !(d[i] >= 0.0) {
                return true;
            }
        }
        false
    }
}

impl SoftDraw<'_> {
    /// A fragment's colour value from an interpolated or flat index.
    pub fn soft_index(&self, index: f64) -> u32 {
        gl_index_value(self.st.index_bits, index)
    }

    pub fn owned_at(&self, px: i32, py: i32) -> bool {
        let x = self.win.ox.wrapping_add(px);
        let y = self.win.oy.wrapping_add(py);
        let st: &GlDrawState = &self.st;

        (self.win.clip.n <= 1 || self.win.clip.contains(x, y)) &&
            (st.scissor_test == 0 ||
             (px >= st.scissor[0] &&
              (px as i64) < st.scissor[0] as i64 + st.scissor[2] as i64 &&
              py >= st.scissor[1] &&
              (py as i64) < st.scissor[1] as i64 + st.scissor[3] as i64))
    }

    /// The colour of a fragment that has passed every test written to the
    /// draw's colour targets at plane pixel (x, y): blended with or combined
    /// by the logical operation with what is there, dithered and masked, from
    /// dest in RGBA mode, which the caller has read where the draw reads it.
    #[inline(always)]
    pub fn fragment_store(&self, x: i32, y: i32, f: &SoftFrag, dest: &[RasterColour; SGI_GL_MAX_TARGETS as usize]) {
        let t: &RasterTarget = &self.t;
        let st: &GlDrawState = &self.st;

        if st.index_bits == 0 && t.store_rgba {
            let mut i = 0;
            while self.rgba_mask != 0 && i < self.ntargets as usize {
                let out = self.rgba_result(x, y, f, &dest[i]);

                self.px.store_rgba(self.targets[i], x, y, &out);
                i += 1;
            }
            return;
        }
        // Dithering follows blending (4.1.7), so a colour to be blended is
        // rounded and the blend's result dithered.
        let colour = if st.index_bits != 0 {
            self.soft_index(f.index)
        } else if self.blend {
            self.rules.rgb(&f.rgba)
        } else {
            self.store_rgb(x, y, &f.rgba)
        };
        for i in 0..self.ntargets as usize {
            let mut out = colour;

            if self.logic_op {
                let bits: u32 = if st.index_bits != 0 { st.index_bits } else { 3 * self.rules.color_bits as u32 };

                out = soft_logic_op(
                    st.logic_op_mode,
                    colour,
                    self.px.load_colour(self.targets[i], x, y),
                ) & (if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 });
            }
            if st.index_bits != 0 {
                self.px.store_index(self.targets[i], x, y, out, st.index_mask);
                continue;
            }
            if self.colour_mask == 0 {
                continue;
            }
            if self.blend {
                out = self.blend(
                    x,
                    y,
                    colour,
                    self.rules.alpha(f.rgba[3]) as f64 / self.rules.alpha_max() as f64,
                    self.px.load_colour(self.targets[i], x, y),
                );
            }
            if self.colour_mask != self.rules.colour_bits() {
                out = (out & self.colour_mask) | (self.px.load_colour(self.targets[i], x, y) & !self.colour_mask);
            }
            self.px.store_colour(self.targets[i], x, y, out);
        }
    }

    pub fn fragment_at(&self, px: i32, py: i32, z: f64, frag: &SoftFrag, coverage: f64, antialias: bool) {
        self.fragment_at_offset(px, py, z, 0, frag, coverage, antialias);
    }

    /// [`fragment_at`](Self::fragment_at) with `steps` of the board's polygon
    /// offset added to the depth code (RasterRules::split_offset()).
    pub fn fragment_at_offset(&self, px: i32, py: i32, z: f64, steps: i64, frag: &SoftFrag, coverage: f64, antialias: bool) {
        let t: &RasterTarget = &self.t;
        let st: &GlDrawState = &self.st;
        let mut fogged = *frag;
        let x = self.win.ox.wrapping_add(px);
        let y = self.win.oy.wrapping_add(py);
        let mut dest = [RasterColour::default(); SGI_GL_MAX_TARGETS as usize];

        if !(z >= 0.0 && z <= 1.0) {
            return;
        }
        if let Some(im) = self.texture {
            if frag.fixed {
                let format = match self.texture_base {
                    Some(b) => b.base_format,
                    None => im.base_format,
                };
                self.texture_apply(format, st.texture_env_mode, &mut fogged.texel, &mut fogged.rgba);
            } else if !self.texture(&mut fogged) {
                return;
            }
        }
        if st.fog != 0 {
            self.fog(&mut fogged, z);
        }
        fogged.rgba[3] *= coverage;
        if antialias && st.index_bits != 0 {
            let bits = std::cmp::min(4, st.index_bits);
            let mask = (1u32 << bits) - 1;
            let index = self.soft_index(fogged.index);
            let value = std::cmp::min((coverage * (1u32 << bits) as f64).floor() as u32, mask);

            fogged.index = ((index & !mask) | value) as f64;
        }
        if !self.owned_at(px, py) {
            return;
        }
        // Incoming alpha and reference use at least two bits and at least the
        // stored alpha width, independently of the colour mask (4.1.4, 2.13.9).
        let f = &fogged;
        if st.alpha_test != 0 && st.index_bits == 0 &&
            !self.rules.alpha_test(st.alpha_func, f.rgba[3], st.alpha_ref as f64)
        {
            return;
        }
        if st.index_bits == 0 && t.store_rgba && self.rgba_read &&
            !self.rules.rgba_load(self.px, &self.targets, self.ntargets, x, y, &mut dest, &self.reports.rgba)
        {
            return;
        }
        if !self.rules.depth_stencil(st, t, self.px, x, y, z, steps) {
            return;
        }
        self.fragment_store(x, y, f, &dest);
    }
}

impl GlDrawState {
    /// The depth range [n, f] normalized depth is mapped through (2.10.1): the
    /// draw state's, or [0, 1] under a window matrix, whose depth is already
    /// window depth.
    pub fn depth_range(&self, n: &mut f64, f: &mut f64) {
        *n = if self.window_matrix_given != 0 { 0.0 } else { self.depth_range[0] as f64 };
        *f = if self.window_matrix_given != 0 { 1.0 } else { self.depth_range[1] as f64 };
    }

    pub fn window_depth(&self, z: f64, offset: f64, window: &mut f64) -> bool {
        let mut n = 0.0;
        let mut f = 0.0;

        if !(z >= 0.0 && z <= 1.0) {
            return false;
        }
        self.depth_range(&mut n, &mut f);
        let zw = n + (f - n) * z + offset;
        *window = if zw >= 1.0 { 1.0 } else if zw > 0.0 { zw } else { 0.0 };
        true
    }
}

impl SoftDraw<'_> {
    /// The window pixels a primitive's fragments may fall on, inclusive: the
    /// bounds of those the window owns, within the viewport unless a window
    /// matrix replaces it. False when there are none.
    pub fn bounds(&self, x0: &mut i32, y0: &mut i32, x1: &mut i32, y1: &mut i32) -> bool {
        let st: &GlDrawState = &self.st;
        let win = &self.win;
        let vx = st.viewport[0];
        let vy = st.viewport[1];
        let vw = st.viewport[2];
        let vh = st.viewport[3];

        *x0 = win.x0.wrapping_sub(win.ox);
        *y0 = win.y0.wrapping_sub(win.oy);
        *x1 = win.x1.wrapping_sub(win.ox);
        *y1 = win.y1.wrapping_sub(win.oy);
        if st.window_matrix_given != 0 {
            return true;
        }
        if vw <= 0 || vh <= 0 {
            return false;
        }
        *x0 = std::cmp::max(*x0, vx);
        *y0 = std::cmp::max(*y0, vy);
        *x1 = std::cmp::min(*x1, vx.wrapping_add(vw).wrapping_sub(1));
        *y1 = std::cmp::min(*y1, vy.wrapping_add(vh).wrapping_sub(1));
        true
    }

    /// Whether a polygon fragment at (px, py) is drawn under the polygon
    /// stipple, and in *background whether it takes the target's opaque
    /// stipple background (RasterTarget's stipple_opaque).
    pub fn stipple_pass(&self, px: i32, py: i32, background: &mut bool) -> bool {
        *background = false;
        if self.polygon_stipple.is_none_or(|rows| rows[((py as u32) & 31) as usize] & (1u32 << ((px as u32) & 31)) != 0) {
            return true;
        }
        *background = self.t.stipple_opaque;
        *background
    }

    pub fn stipple_background(&self, frag: &mut SoftFrag) {
        for k in 0..4usize {
            frag.rgba[k] = self.t.stipple_background[k] as f64;
        }
        frag.index = self.t.stipple_background_index as f64;
    }

    /// A primitive's fragment at normalized depth z, (zd + 1) / 2 for the
    /// normalized device z zd: dropped outside [0, 1], the view volume's near
    /// and far planes, which clipping has already cut the primitive to, and
    /// otherwise mapped to window depth n + (f - n) z through the depth range
    /// [n, f] (OpenGL 1.2.1, 2.10.1). A polygon's offset is then added to the
    /// window depth, and the sum clamped to [0, 1] (3.5.5).
    pub fn fragment_coverage(&self, px: i32, py: i32, z: f64, offset: f64, depth_w: f64, frag: &SoftFrag, coverage: f64, antialias: bool) {
        let mut z = z;
        let (mut offset, steps) = self.rules.split_offset(offset);
        let mut zw = 0.0;

        if self.rules.depth_row_given {
            let slack = self.rules.depth_slack;

            // Depth and its divisor, each over w, interpolate linearly in the
            // window; their ratio is the row's depth at the fragment.
            z = (z + offset) / depth_w;
            offset = 0.0;
            if z < 0.0 && z >= -slack {
                z = 0.0;
            } else if z > 1.0 && z <= 1.0 + slack {
                z = 1.0;
            }
        }
        if !(coverage > 0.0) || !self.st.window_depth(z, offset, &mut zw) {
            return;
        }
        self.fragment_at_offset(px, py, zw, steps, frag, coverage, antialias);
    }

    pub fn fragment(&self, px: i32, py: i32, z: f64, offset: f64, depth_w: f64, frag: &SoftFrag) {
        self.fragment_coverage(px, py, z, offset, depth_w, frag, 1.0, false);
    }
}

impl GlDrawState {
    /// A triangle's polygon offset (3.5.5), o = m factor + r units, where m is
    /// the larger magnitude of its window depth's slopes in x and y, which the
    /// specification allows in place of their length, and r the rules' one
    /// depth step; EXT_polygon_offset's bias adds in window depth. Boundary modes
    /// share this slope calculation and select their own offset enable.
    pub fn depth_slope(&self, window: &[[f64; 3]; 3]) -> f64 {
        let mut n = 0.0;
        let mut f = 0.0;

        self.depth_range(&mut n, &mut f);
        let x1 = window[1][0] - window[0][0];
        let y1 = window[1][1] - window[0][1];
        let x2 = window[2][0] - window[0][0];
        let y2 = window[2][1] - window[0][1];
        let z1 = window[1][2] - window[0][2];
        let z2 = window[2][2] - window[0][2];
        let det = x1 * y2 - x2 * y1;
        // Reviewed model choice for an entirely degenerate boundary: m = 0.
        if det == 0.0 {
            return 0.0;
        }
        fmax((z1 * y2 - z2 * y1).abs(), (x1 * z2 - x2 * z1).abs()) / det.abs() * (f - n).abs()
    }
}

impl SoftDraw<'_> {
    pub fn window_vertices(&self, v: &[SoftVertex; 3], window: &mut [[f64; 3]; 3]) {
        let sub = (1i32 << self.rules.subpixel_bits) as f64;

        for i in 0..3usize {
            window[i][0] = v[i].x as f64 / sub;
            window[i][1] = v[i].y as f64 / sub;
            window[i][2] = v[i].z;
        }
    }

    pub fn slope(&self, v: &[SoftVertex; 3]) -> f64 {
        let mut window = [[0.0f64; 3]; 3];

        self.window_vertices(v, &mut window);
        self.st.depth_slope(&window)
    }
}

impl RasterRules {
    pub fn polygon_offset(&self, st: &GlDrawState, window: &[[f64; 3]; 3]) -> f64 {
        if st.polygon_offset & SGI_GL_OFFSET_FILL == 0 {
            return 0.0;
        }
        st.depth_slope(window) * st.offset_factor as f64
            + st.offset_units as f64 / self.depth_max() as f64
            + st.offset_bias as f64
    }
}

impl SoftDraw<'_> {
    pub fn offset(&self, v: &[SoftVertex; 3]) -> f64 {
        let mut window = [[0.0f64; 3]; 3];

        self.window_vertices(v, &mut window);
        self.rules.polygon_offset(&self.st, &window)
    }
}

impl SoftDraw<'_> {
    /// Map one clipped vertex (2.10): the divide by w and the viewport map,
    /// with depth left normalized to [0, 1]; the depth range maps it per
    /// fragment. Under a window matrix the position is the matrix's and the
    /// depth its window depth. Clipping has put the vertex in the view volume,
    /// so its normalized depth is clamped to [0, 1] against rounding. Under the
    /// rules' depth row the depth is the row's over w instead, unclamped, as
    /// clipping does not bound it. The colour is the front colour; a back-facing
    /// triangle takes its back colour later.
    /// A vertex with w <= 0 is in the volume only at the clip-space origin,
    /// where the divide has no answer, and returns false, as does one mapped
    /// beyond the rasterizer's range.
    pub fn project(&self, cv: &VolumeVertex, out: &mut SoftVertex) -> bool {
        let st: &GlDrawState = &self.st;
        let clip = &cv.clip;
        let eye = &cv.eye;
        let mut win = [0.0f64; 3];
        let quiet = SOFT_BAND_QUIET.with(|q| q.get());

        if !self.texture_vertex_valid(cv) {
            return false;
        }
        if !(clip[3] > 0.0) {
            if !quiet {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: clipped vertex at w {} has no window position, primitive dropped", clip[3]);
            }
            return false;
        }
        st.clip_to_window(clip, &mut win);
        let xw = win[0];
        let yw = win[1];
        if !(xw.abs() < 65536.0 && yw.abs() < 65536.0) {
            if !quiet {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: vertex outside the rasterizer's range, primitive dropped");
            }
            return false;
        }
        for k in 0..4usize {
            out.texcoord[k] = cv.attr[SGI_GL_V_TEXCOORD + k] as f64;
        }
        let sub = (1i32 << self.rules.subpixel_bits) as f64;
        out.x = self.rules.round(xw * sub);
        out.y = self.rules.round(yw * sub);
        if !self.rules.depth_row_given {
            let z = win[2];
            out.z = if z >= 1.0 { 1.0 } else if z > 0.0 { z } else { 0.0 };
        } else {
            let d = &self.rules.depth_row;

            // The head's depth over w, which a window line interpolates.
            out.z = (d[0] * eye[0] + d[1] * eye[1] + d[2] * eye[2] + d[3] * eye[3]) / clip[3];
        }
        out.inv_w = 1.0 / clip[3];
        out.depth_w = out.inv_w;
        if self.rules.depth_divisor_given {
            let d = &self.rules.depth_divisor;

            out.depth_w = (d[0] * eye[0] + d[1] * eye[1] + d[2] * eye[2] + d[3] * eye[3]) / clip[3];
        }
        // The fog coordinate approximates the eye distance by |z|, which
        // 3.10 allows; it is affine in eye space, so interpolating it as a
        // vertex attribute gives it exactly at each fragment.
        out.fog = (if eye[3] != 0.0 { eye[2] / eye[3] } else { eye[2] }).abs();
        for i in 0..SGI_GL_CLIP_PLANES as usize {
            let p = &st.clip_plane[i];

            out.clip[i] = p[0] as f64 * eye[0] + p[1] as f64 * eye[1] + p[2] as f64 * eye[2] + p[3] as f64 * eye[3];
        }
        for k in 0..4usize {
            let a = cv.attr[SGI_GL_V_COLOR + k];
            let m = if a > 0.0 { a } else { 0.0 };
            let n = if m < 1.0 { m } else { 1.0 };
            out.rgba[k] = n as f64;
        }
        out.index = cv.attr[SGI_GL_V_INDEX] as f64;
        true
    }
}

impl RasterRules {
    /// Draw a draw's primitives into the planes, its texels from texels.
    /// A standalone line stipple's phase goes back in d.line_phase.
    pub fn draw(&self, t: &RasterTarget, planes: &mut dyn PlaneStore, texels: &dyn Texels, d: &mut GlDraw) -> GlDrawResult {
        with_texel_cache(|cache| self.draw_cached(t, planes, texels, d, cache))
    }

    fn draw_cached(
        &self,
        t: &RasterTarget,
        planes: &mut dyn PlaneStore,
        texels: &dyn Texels,
        d: &mut GlDraw,
        cache: &std::cell::RefCell<SoftTexelCache>,
    ) -> GlDrawResult {
        let reports = SoftReports::default();
        let mut r = SoftDraw::new(self, t, d.state, &reports);
        let mut line = SoftLineStipple {
            counter: match d.line_phase {
                Some(phase) if !d.line_phase_reset => phase,
                _ => 0,
            },
            enabled: d.line_stipple_enabled,
            repeat: r2_max(1, r2_min(d.line_stipple_repeat, 256)),
            pattern: d.line_stipple_pattern,
            consumed: false,
        };
        let use_stipple = d.line_phase.is_some() || d.line_stipple_enabled;
        let mut fixed = SoftFixedTex::default();

        r.texels = texels;
        r.back_colors = d.back_colors;
        r.point_smooth = self.smooth_primitives && d.point_smooth;
        r.line_smooth = self.smooth_primitives && d.line_smooth;
        r.smooth_point_size = d.point_size as f64;
        r.smooth_line_width = d.line_width as f64;
        r.polygon_smooth = self.smooth_primitives && d.polygon_smooth;
        r.polygon_stipple = if d.polygon_stipple_enabled { Some(d.polygon_stipple) } else { None };
        line.counter %= 16u32.wrapping_mul(line.repeat);
        r.ntargets = d.state.draw_targets(&mut r.targets);
        if !self.check_rules() || !t.soft_check_index(d.state.index_bits) {
            return SGI_GL_DRAW_OK;
        }
        if !d.state.window(&mut r.win) {
            if !use_stipple
                || d.mode < SGI_GL_LINES
                || d.mode > SGI_GL_LINE_STRIP
                || d.state.drawable.bound == 0
                || d.state.drawable.clip.n != 0
            {
                return SGI_GL_DRAW_OK;
            }
            // Empty ownership still counts standalone geometric coverage.
            r.win.x1 = -1;
            r.win.y1 = -1;
        }
        if !r.texture_setup(true) {
            return SGI_GL_DRAW_OK;
        }
        if r.texture.is_some() {
            r.texels_cache = Some(cache);
        }
        if !r.setup_ops() {
            return SGI_GL_DRAW_OK;
        }
        let mut fixed_on = false;
        if d.mode >= SGI_GL_TRIANGLES && d.mode <= SGI_GL_POLYGON && r.fixed_setup(&mut fixed) {
            let mut texels = 0u64;
            let mut first = 0u32;
            let mut last = 0u32;

            fixed.span = r.fixed_span_select(&fixed);
            fixed.fixed_reach(&mut first, &mut last);
            for i in first..=last {
                texels += fixed.level[i as usize].width as u64 * fixed.level[i as usize].height as u64;
            }
            // A draw as large as the smallest that bands share, at least.
            texels = r2_max(texels, SGI_RASTER_PARALLEL_MIN_PIXELS);
            fixed_on = (r.draw_pixels(d, texels) >= texels
                && r.fixed_coordinates_valid(d)
                && r.fixed_prefill(&mut fixed))
                || r.fixed_begin(&fixed);
        }
        if fixed_on {
            r.fixed = Some(&fixed);
        }
        let result = r.draw_primitives(planes, d, &mut line, use_stipple);
        if d.line_phase.is_some() && line.consumed {
            d.line_phase = Some(line.counter);
        }
        if fixed.prefilled {
            // The prefilled arrays go back to this thread's store for the next draw.
            fixed_return(cache, &mut fixed);
        }
        result
    }
}

impl<'a> SoftDraw<'a> {
    /// The primitives of a draw that has passed its setup.
    fn draw_primitives(&self, planes: &mut dyn PlaneStore, d: &GlDraw<'a>, line: &mut SoftLineStipple, use_stipple: bool) -> GlDrawResult {
        let n = d.count as usize;
        let v = |i: usize| -> &[f32] { d.soft_polygon_vertex(None, i) };

        if !self.rules.smooth_primitives {
            let mut smooth: Option<&str> = None;

            if d.mode == SGI_GL_POINTS && d.point_smooth {
                smooth = Some("point");
            } else if d.mode >= SGI_GL_LINES && d.mode <= SGI_GL_LINE_STRIP && d.line_smooth {
                smooth = Some("line");
            } else if d.mode >= SGI_GL_TRIANGLES && d.mode <= SGI_GL_POLYGON && d.polygon_smooth {
                smooth = Some("polygon");
            }
            if let Some(s) = smooth {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: {} smoothing requested but unsupported by raster rules; drawn aliased", s);
            }
        }
        let filled = d.mode >= SGI_GL_TRIANGLES && d.mode <= SGI_GL_POLYGON;
        if filled
            && (self.polygon_smooth || d.polygon_modes.is_some_and(|m| m != [SGI_GL_FILL, SGI_GL_FILL]))
        {
            let px = planes.planes();
            return self.with_planes(&*px).polygon_modes(d);
        }
        let mut r = *self;
        match d.mode {
            SGI_GL_POINTS => {
                if !soft_dimension(d.point_size, &mut r.point_size) {
                    return SGI_GL_DRAW_OK;
                }
                let px = planes.planes();
                let r = r.with_planes(&*px);
                for i in 0..n {
                    r.point(v(i));
                }
                SGI_GL_DRAW_OK
            }
            SGI_GL_LINES => {
                if !soft_dimension(d.line_width, &mut r.line_width) {
                    return SGI_GL_DRAW_OK;
                }
                let px = planes.planes();
                let r = r.with_planes(&*px);
                let mut i = 0usize;
                while i + 1 < n {
                    if d.line_phase.is_none() {
                        line.counter = 0;
                    }
                    r.line(v(i), v(i + 1), v(i + 1), if use_stipple { Some(&mut *line) } else { None });
                    i += 2;
                }
                SGI_GL_DRAW_OK
            }
            SGI_GL_LINE_STRIP | SGI_GL_LINE_LOOP => {
                if !soft_dimension(d.line_width, &mut r.line_width) {
                    return SGI_GL_DRAW_OK;
                }
                let px = planes.planes();
                let r = r.with_planes(&*px);
                let mut i = 0usize;
                while i + 1 < n {
                    r.line(v(i), v(i + 1), v(i + 1), if use_stipple { Some(&mut *line) } else { None });
                    i += 1;
                }
                if d.mode == SGI_GL_LINE_LOOP && n >= 2 {
                    r.line(v(n - 1), v(0), v(0), if use_stipple { Some(&mut *line) } else { None });
                }
                SGI_GL_DRAW_OK
            }
            SGI_GL_TRIANGLES | SGI_GL_TRIANGLE_STRIP | SGI_GL_TRIANGLE_FAN | SGI_GL_POLYGON | SGI_GL_QUADS
            | SGI_GL_QUAD_STRIP => {
                if !r.draw_parallel(d, planes) {
                    let px = planes.planes();
                    r.with_planes(&*px).draw_triangles(d);
                }
                SGI_GL_DRAW_OK
            }
            _ => {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: draw of mode {} ignored", d.mode);
                SGI_GL_DRAW_OK
            }
        }
    }
}
