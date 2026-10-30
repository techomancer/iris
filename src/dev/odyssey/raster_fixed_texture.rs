//! The fixed-point fast path for textured triangles: the sampler's state, its
//! texel reads and filters, and the lambda table. The quad loops that draw a
//! textured triangle's pixels through it are SoftDraw::fill_quads() in
//! raster_fill.rs. Its colours stay within 3/255 of the double-precision
//! path's; it is faster, not more accurate.
//!
//! Ported from hw/display/sgi/rasterizer.c in jamesbraid/qemu-sgi-origami.

use super::*;

/// MAX and MIN as conditionals: the first operand when the comparison fails,
/// so a NaN operand is dropped.
#[inline(always)]
pub fn fx_max<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

#[inline(always)]
pub fn fx_min<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { a } else { b }
}

// Fixed-point texturing for filled triangles.
//
// Approximated: no source gives any family's sub-texel, filter-weight or
// level-of-detail widths, and the double arithmetic of SoftDraw::texture()
// is no closer to a board's than another choice. Where the state allows,
// a filled triangle samples instead with:
//  - each fragment's s and t from the triangle's S/W, T/W and Q/W planes,
//    divided in double, so a far, many-times repeated floor keeps its
//    texel positions (a 32-bit fixed-point numerator does not);
//  - texel positions in 1/256 of a texel: bilinear weights of 8 bits;
//  - lambda, the level of detail, in 1/256, from the squared footprint
//    through a 10-bit table of log2, with no square root; within a few
//    steps of a choice between filters or nearest levels it is computed
//    exactly instead, rounded up, so the choice is the double path's;
//  - a blend between two levels by an 8-bit fraction;
//  - texels as 8-bit components, filtered in integers with no rounding
//    until the texture environment takes the result in double.
// Coverage, depth, stencil and the pixels written are those of the
// double path. Colour differs from it within a bound measured on the
// comparison corpus.
//
// The state it takes: a 2D image or pyramid of base format alpha,
// luminance, luminance-alpha, intensity, RGB or RGBA, read through the
// packed reader or through component readers 1, 2, 4 or 8 bits wide,
// which convert to 8 bits exactly; levels of at most 8192 texels a side
// and 2^22 texels; NEAREST, LINEAR or a mipmap minification filter with
// NEAREST or LINEAR magnification; any wrap, a border image only under
// REPEAT or CLAMP_TO_EDGE, which never reach it; a target that does not
// draw mixed frames (RasterTarget.mixed_frame); no alpha test, whose
// outcome would move with the texel's alpha; and no logical operation,
// which turns a difference in the lowest bit into one in any. Anything
// else, and a fragment the fixed sampler cannot take (a coordinate beyond
// 2^30, Q/W not positive, a failed read), goes to SoftDraw::texture(), so it
// is dropped and reported as before.
pub const SGI_SOFT_FIXED_MAX_SIDE: u32 = 8192;
pub const SGI_SOFT_FIXED_MAX_TEXELS: u32 = 1 << 22;
/// The fewest pixels a draw covers for its texels to be read ahead
/// (SoftDraw::fixed_prefill()).
pub const SGI_SOFT_FIXED_PREFILL_MIN_PIXELS: u64 = 2048;
/// Lambda's bounds, beyond any level count, in 1/256.
pub const SGI_SOFT_FIXED_LAMBDA_MAX: i32 = 1 << 24;

thread_local! {
    pub static SOFT_TEXTURE_PATH: std::cell::Cell<RasterTexturePath> =
        const { std::cell::Cell::new(SGI_RASTER_TEXTURE_SPANS) };
}

/// How filled triangles drawn on this thread sample a texture, for tests.
/// Per thread, so that tests running at once cannot change each other's
/// path.
pub fn raster_set_texture_path(path: RasterTexturePath) {
    SOFT_TEXTURE_PATH.with(|p| p.set(path));
}

#[derive(Clone, Copy, Default)]
pub struct SoftFixedLevel {
    /// None: the image's level-zero fields
    pub level: Option<GlTexLevel>,
    pub width: u32,
    pub height: u32,
    pub width_shift: u32,
    pub height_shift: u32,
}

#[derive(Clone, Default)]
pub struct SoftFixedTex {
    pub level: [SoftFixedLevel; SGI_SOFT_FIXED_LEVELS],
    /// The row loop for textured triangles, or None for the general one.
    pub span: Option<FillRow>,
    /// Every texel read before drawing, by level (SoftDraw::fixed_prefill()).
    pub prefilled: bool,
    /// Lent by the drawing thread's texel store for the draw
    /// (SoftDraw::fixed_return()).
    pub plain: [Vec<u64>; SGI_SOFT_FIXED_LEVELS],
    /// the last level, relative to base
    pub range: u32,
    pub format: u32,
    pub min: u32,
    pub mag: u32,
    pub wrap_s: u32,
    pub wrap_t: u32,
    /// the border colour, in lanes
    pub border: u64,
    /// CLAMP's upper coordinate on each axis (GlTexImage.max_clamp).
    pub upper_s: f64,
    pub upper_t: f64,
    /// Whether the footprint chooses the filter or level at all.
    pub lod: bool,
    /// Whether it chooses between filters or nearest levels.
    pub choice: bool,
    /// in 1/256
    pub min_lod: i32,
    pub max_lod: i32,
    /// lambda at or below it magnifies
    pub mag_below: i32,
    /// the footprint's per-axis scales
    pub ns: f64,
    pub nt: f64,
}

/// A packed RRGGBBAA texel's components each in the low byte of a 16-bit
/// lane, red highest: a component times a weight of at most 256 fits its
/// lane, so one product weights all four.
#[inline(always)]
pub fn soft_fixed_lanes(rgba: u32) -> u64 {
    let t: u64 = ((rgba & 0xffff0000) as u64) << 16 | (rgba & 0xffff) as u64;

    (t & 0x0000ff000000ff00) << 8 | (t & 0x000000ff000000ff)
}

/// A packed RRGGBBAA texel with the components the base format lacks
/// filled in, as soft_texture_components() fills them.
pub fn soft_fixed_expand(format: u32, rgba: u32) -> u32 {
    let r = rgba >> 24;

    match format {
        SGI_GL_ALPHA => 0xffffff00 | (rgba & 0xff),
        SGI_GL_LUMINANCE => r.wrapping_mul(0x01010100) | 0xff,
        SGI_GL_LUMINANCE_ALPHA => r.wrapping_mul(0x01010100) | (rgba & 0xff),
        SGI_GL_INTENSITY => r.wrapping_mul(0x01010101),
        SGI_GL_RGB => rgba | 0xff,
        _ => rgba,
    }
}

pub fn soft_fixed_unit(c: f64) -> u32 {
    (fx_min(fx_max(c, 0.0), 1.0) * 255.0 + 0.5) as u32
}

impl SoftDraw<'_> {
    /// Whether draw r's filled triangles sample its texture in fixed point,
    /// and if so f made ready for them; the tests SoftDraw::texture_setup()
    /// has already made are not repeated.
    pub fn fixed_setup(&self, f: &mut SoftFixedTex) -> bool {
        let base = self.texture_base;
        let mut border = [0.0f64; 4];

        if SOFT_TEXTURE_PATH.with(|p| p.get()) == SGI_RASTER_TEXTURE_DOUBLE
            || self.texture.is_none()
            || self.texels_cache.is_none()
            || self.t.mixed_frame
            || self.image().volume
            || self.image().one_dimensional
            || self.st.index_bits != 0
            || self.st.alpha_test != 0
            || self.logic_op
        {
            return false;
        }
        let im = self.image();
        let p = im.pyramid;
        *f = SoftFixedTex::default();
        f.format = match base {
            Some(b) => b.base_format,
            None => im.base_format,
        };
        f.min = im.min_filter;
        f.mag = im.mag_filter;
        f.wrap_s = im.wrap_s;
        f.wrap_t = im.wrap_t;
        f.upper_s = if im.max_clamp { im.max_clamp_s } else { 1.0 };
        f.upper_t = if im.max_clamp { im.max_clamp_t } else { 1.0 };
        let mipmap: bool = soft_texture_mipmap(f.min);
        if f.format == SGI_GL_DEPTH_COMPONENT
            || (f.mag != SGI_GL_NEAREST && f.mag != SGI_GL_LINEAR)
            || (f.min != SGI_GL_NEAREST && f.min != SGI_GL_LINEAR && !mipmap)
        {
            return false;
        }
        let edge_only: bool = (f.wrap_s == SGI_GL_REPEAT || f.wrap_s == SGI_GL_CLAMP_TO_EDGE)
            && (f.wrap_t == SGI_GL_REPEAT || f.wrap_t == SGI_GL_CLAMP_TO_EDGE);
        f.range = if base.is_some() { self.texture_last.wrapping_sub(p.unwrap().base_level) } else { 0 };
        if f.range as usize >= SGI_SOFT_FIXED_LEVELS {
            return false;
        }
        for i in 0..=f.range {
            let l: Option<&GlTexLevel> = if base.is_some() {
                let p = p.unwrap();
                p.soft_texture_level(p.base_level.wrapping_add(i))
            } else {
                None
            };
            let fl = &mut f.level[i as usize];
            let format = match l {
                Some(l) => l.base_format,
                None => im.base_format,
            };
            fl.level = l.copied();
            fl.width = l.map_or(im.width, |l| l.width);
            fl.height = l.map_or(im.height, |l| l.height);
            fl.width_shift = fl.width.trailing_zeros();
            fl.height_shift = fl.height.trailing_zeros();
            if l.is_none() && i != 0 {
                return false;
            }
            let border_texels = match l {
                Some(l) => l.border,
                None => im.border,
            };
            let readable = l.map_or(im.read_texel, |l| l.read_texel);
            if fl.width > SGI_SOFT_FIXED_MAX_SIDE
                || fl.height > SGI_SOFT_FIXED_MAX_SIDE
                || (fl.width as u64) * (fl.height as u64) > SGI_SOFT_FIXED_MAX_TEXELS as u64
                || !(fl.width as u64).is_power_of_two()
                || !(fl.height as u64).is_power_of_two()
                || format != f.format
                || (border_texels != 0 && !edge_only)
                || !readable
            {
                return false;
            }
        }
        for k in 0..4 {
            border[k] = fx_min(fx_max(im.border_color[k], 0.0), 1.0) as f64;
        }
        soft_texture_components(f.format, &mut border);
        f.border = soft_fixed_lanes(
            soft_fixed_unit(border[0]) << 24
                | soft_fixed_unit(border[1]) << 16
                | soft_fixed_unit(border[2]) << 8
                | soft_fixed_unit(border[3]),
        );
        // An image without a pyramid has one filter and no footprint.
        f.lod = base.is_some() && (mipmap || f.min != f.mag);
        f.choice = f.lod && !(f.min == SGI_GL_LINEAR_MIPMAP_LINEAR && f.mag == SGI_GL_LINEAR);
        if let Some(base) = base {
            let p = p.unwrap();
            let lmax = SGI_SOFT_FIXED_LAMBDA_MAX as f64;

            // Rounded up, as an exact lambda is (SoftFixedTex::fixed_decide()).
            f.min_lod = fx_min(fx_max(p.min_lod * 256.0, -lmax), lmax).ceil() as i32;
            f.max_lod = fx_min(fx_max(p.max_lod * 256.0, -lmax), lmax).ceil() as i32;
            f.ns = base.width as f64 * soft_texture_lod_scale(im.lod_scale_s);
            f.nt = base.height as f64 * soft_texture_lod_scale(im.lod_scale_t);
        }
        // soft_texture_select()'s c, for these filters.
        f.mag_below = if f.mag == SGI_GL_LINEAR
            && (f.min == SGI_GL_NEAREST_MIPMAP_NEAREST || f.min == SGI_GL_NEAREST_MIPMAP_LINEAR)
        {
            128
        } else {
            0
        };
        true
    }
}

/// Allocate zeroed cells for size texels, or None where that fails.
pub fn soft_fixed_alloc(size: usize) -> Option<Vec<u64>> {
    let mut v: Vec<u64> = Vec::new();

    if v.try_reserve_exact(size).is_err() {
        return None;
    }
    v.resize(size, 0);
    Some(v)
}

impl<'a> SoftDraw<'a> {
    /// The draw's fixed-point sampler, which a caller has checked is there.
    pub fn fixed_tex(&self) -> &'a SoftFixedTex {
        self.fixed.expect("a fixed-point draw")
    }

    /// This thread's texel cache, which a textured draw has.
    pub fn texel_cache(&self) -> std::cell::RefMut<'a, SoftTexelCache> {
        self.texels_cache.expect("a texel cache").borrow_mut()
    }

    /// Make this thread's texel store ready for f's levels: false, and the
    /// draw samples in double, where it cannot be allocated.
    pub fn fixed_begin(&self, f: &SoftFixedTex) -> bool {
        let mut cache = self.texel_cache();

        for i in 0..=f.range as usize {
            let store = &mut cache.fixed[i];
            let size = (f.level[i].width as usize) * (f.level[i].height as usize);

            if store.size < size {
                store.cell = Vec::new();
                store.size = 0;
                match soft_fixed_alloc(size) {
                    Some(v) => store.cell = v,
                    None => return false,
                }
                store.size = size;
            }
        }
        true
    }

    /// Interior texel (x, y) of relative level i of the draw's image as a
    /// packed texel, components expanded; None where the read fails.
    pub fn fixed_read(&self, f: &SoftFixedTex, i: u32, x: u32, y: u32) -> Option<u32> {
        let rgba = self.texels.texel(f.level[i as usize].level.as_ref(), x, y)?;

        Some(soft_fixed_expand(f.format, rgba))
    }

    /// Read interior texel (x, y) of relative level i, and return its cell,
    /// tagged with tag, or tagged failed where the read failed.
    #[inline(never)]
    pub fn fixed_fill(&self, i: u32, x: u32, y: u32, tag: u32) -> u64 {
        match self.fixed_read(self.fixed_tex(), i, x, y) {
            Some(texel) => (tag as u64) << 32 | texel as u64,
            None => ((tag | SGI_SOFT_FIXED_FAILED) as u64) << 32,
        }
    }
}

impl SoftFixedTex {
    /// The relative levels a draw's fragments can sample, first to last: the
    /// level clamp keeps lambda within [min_lod, max_lod], and a lambda above
    /// a half never magnifies, so no level outside reaches the sampler.
    pub fn fixed_reach(&self, first: &mut u32, last: &mut u32) {
        *first = 0;
        *last = self.range;
        if !self.lod {
            return;
        }
        if self.min_lod > 128 {
            *first = fx_min((self.min_lod >> 8) as u32, self.range);
        }
        *last = if self.max_lod <= 0 { 0 } else { fx_min((self.max_lod.wrapping_add(255)) >> 8, self.range as i32) as u32 };
        *first = fx_min(*first, *last);
    }
}

impl SoftDraw<'_> {
    /// Read every texel of the levels draw r can sample into this thread's
    /// plain arrays, which f then lends the draw: for a large draw that
    /// covers about as many pixels as those levels have texels, which then
    /// taps them without a tag. False, and the draw reads texels as it reaches them,
    /// where any read fails or the arrays cannot be allocated.
    pub fn fixed_prefill(&self, f: &mut SoftFixedTex) -> bool {
        let mut first = 0u32;
        let mut last = 0u32;

        f.fixed_reach(&mut first, &mut last);
        for i in first..=last {
            let fl = f.level[i as usize];
            let size = (fl.width as usize) * (fl.height as usize);
            let mut plain = std::mem::take(&mut self.texel_cache().fixed[i as usize].plain);

            if plain.len() < size {
                match soft_fixed_alloc(size) {
                    Some(v) => plain = v,
                    None => return false,
                }
            }
            for y in 0..fl.height {
                for x in 0..fl.width {
                    let Some(texel) = self.fixed_read(f, i, x, y) else {
                        self.texel_cache().fixed[i as usize].plain = plain;
                        return false;
                    };
                    plain[((y as usize) << fl.width_shift) + x as usize] = soft_fixed_lanes(texel);
                }
            }
            f.plain[i as usize] = plain;
        }
        f.prefilled = true;
        true
    }

}

/// Give the arrays SoftDraw::fixed_prefill() filled back to this thread's
/// texel store, for the next draw to reuse.
pub fn fixed_return(cache: &std::cell::RefCell<SoftTexelCache>, f: &mut SoftFixedTex) {
    let mut cache = cache.borrow_mut();

    for (store, plain) in cache.fixed.iter_mut().zip(f.plain.iter_mut()) {
        if !plain.is_empty() {
            store.plain = std::mem::take(plain);
        }
    }
}

impl<'a> SoftDraw<'a> {
    /// Interior texel (x, y) of relative level i in lanes, read once a draw.
    #[inline(always)]
    pub fn fixed_texel(&self, i: u32, x: u32, y: u32, lanes: &mut u64) -> bool {
        let f = self.fixed_tex();
        let index = ((y as usize) << f.level[i as usize].width_shift) + x as usize;

        if f.prefilled {
            *lanes = f.plain[i as usize][index];
            return true;
        }
        let (tag, mut v) = {
            let cache = self.texel_cache();
            (cache.tag, cache.fixed[i as usize].cell[index])
        };
        if (v >> 32) as u32 != tag {
            if (v >> 32) as u32 == (tag | SGI_SOFT_FIXED_FAILED) {
                return false;
            }
            v = self.fixed_fill(i, x, y, tag);
            self.texel_cache().fixed[i as usize].cell[index] = v;
            if (v >> 32) as u32 != tag {
                return false;
            }
        }
        *lanes = soft_fixed_lanes(v as u32);
        true
    }

    /// The texel at signed index (x, y) of relative level i after wrapping,
    /// which leaves an index outside the interior only under CLAMP and
    /// CLAMP_TO_BORDER: there the border colour.
    #[inline(always)]
    pub fn fixed_tap<const REPEAT: bool>(&self, i: u32, x: i64, y: i64, lanes: &mut u64) -> bool {
        let f = self.fixed_tex();
        let fl = &f.level[i as usize];

        if !REPEAT && (x < 0 || y < 0 || x >= fl.width as i64 || y >= fl.height as i64) {
            *lanes = f.border;
            return true;
        }
        self.fixed_texel(i, x as u32, y as u32, lanes)
    }
}

/// floor(x) for |x| below 2^62.
#[inline(always)]
pub fn soft_fixed_floor(x: f64) -> i64 {
    let i = x as i64;

    i - ((x < i as f64) as i64)
}

/// One axis of a fixed-point position u, in 1/256 of a texel of a level n
/// texels wide, wrapped as soft_texture_axis() wraps the coordinate,
/// with CLAMP's upper bound upper rather than 1: floor() of the clamped
/// coordinate is the clamped floor(), so the bound is exactly the double
/// path's.
/// for LINEAR, the first tap's index in *x0, the second's in *x1 and the
/// second's weight; for NEAREST, the tap's index in *x0.
#[inline(always)]
pub fn soft_fixed_axis(mut u: i64, n: u32, wrap: u32, upper: f64, linear: bool, x0: &mut i64, x1: &mut i64) -> u32 {
    let whole: i64 = (n as i64) << 8;
    let mask: i64 = n.wrapping_sub(1) as i64;
    let mut a: u32 = 0;

    if wrap == SGI_GL_CLAMP && upper < 1.0 {
        u = fx_min(u, soft_fixed_floor(upper * whole as f64));
    }

    if linear {
        match wrap {
            SGI_GL_REPEAT => {
                u -= 128;
            }
            SGI_GL_CLAMP_TO_EDGE => {
                u = fx_min(fx_max(u - 128, 0), whole - 256);
            }
            SGI_GL_CLAMP_TO_BORDER => {
                u = fx_min(fx_max(u, -128), whole + 128) - 128;
            }
            _ => {
                u = fx_min(fx_max(u, 0), whole) - 128;
            }
        }
        a = (u & 255) as u32;
        *x0 = u >> 8;
        *x1 = *x0 + 1;
        if wrap == SGI_GL_REPEAT {
            *x0 &= mask;
            *x1 &= mask;
        } else if wrap == SGI_GL_CLAMP_TO_EDGE {
            *x1 = fx_min(*x1, mask);
        }
        return a;
    }
    *x0 = u >> 8;
    match wrap {
        SGI_GL_REPEAT => {
            *x0 &= mask;
        }
        SGI_GL_CLAMP_TO_BORDER => {
            *x0 = fx_min(fx_max(*x0, -1), n as i64);
        }
        _ => {
            *x0 = fx_min(fx_max(*x0, 0), n as i64 - 1);
        }
    }
    a
}

/// Two of a texel's lanes, in the low 16 bits of each 32-bit half.
pub const SGI_SOFT_FIXED_PAIRS: u64 = 0x0000ffff0000ffff;

impl SoftDraw<'_> {
    /// Relative level i sampled at position (u, v), in 1/256 of its texels,
    /// with filter, each component in 1/(255 * 2^16): out[0] holds red over
    /// blue and out[1] green over alpha, one to each 32-bit half. Across, one
    /// product weights all four of a texel's components in their 16-bit
    /// lanes; down, the sums need 24 bits, so two products take two each.
    #[inline(always)]
    pub fn fixed_level<const REPEAT: bool>(&self, i: u32, filter: u32, u: i64, v: i64, out: &mut [u64; 2]) -> bool {
        let f = self.fixed_tex();
        let fl = &f.level[i as usize];
        let ws = if REPEAT { SGI_GL_REPEAT } else { f.wrap_s };
        let wt = if REPEAT { SGI_GL_REPEAT } else { f.wrap_t };
        let (mut x0, mut x1, mut y0, mut y1) = (0i64, 0i64, 0i64, 0i64);
        let (mut c00, mut c10, mut c01, mut c11) = (0u64, 0u64, 0u64, 0u64);

        if filter == SGI_GL_NEAREST {
            soft_fixed_axis(u, fl.width, ws, f.upper_s, false, &mut x0, &mut x1);
            soft_fixed_axis(v, fl.height, wt, f.upper_t, false, &mut y0, &mut y1);
            if !SoftDraw::fixed_tap::<REPEAT>(self, i, x0, y0, &mut c00) {
                return false;
            }
            out[0] = ((c00 >> 16) & SGI_SOFT_FIXED_PAIRS) << 16;
            out[1] = (c00 & SGI_SOFT_FIXED_PAIRS) << 16;
            return true;
        }
        let a = soft_fixed_axis(u, fl.width, ws, f.upper_s, true, &mut x0, &mut x1);
        let b = soft_fixed_axis(v, fl.height, wt, f.upper_t, true, &mut y0, &mut y1);
        if !SoftDraw::fixed_tap::<REPEAT>(self, i, x0, y0, &mut c00)
            || !SoftDraw::fixed_tap::<REPEAT>(self, i, x1, y0, &mut c10)
            || !SoftDraw::fixed_tap::<REPEAT>(self, i, x0, y1, &mut c01)
            || !SoftDraw::fixed_tap::<REPEAT>(self, i, x1, y1, &mut c11)
        {
            return false;
        }
        let (a, b) = (a as u64, b as u64);
        let top = c00.wrapping_mul(256 - a).wrapping_add(c10.wrapping_mul(a));
        let bot = c01.wrapping_mul(256 - a).wrapping_add(c11.wrapping_mul(a));
        out[0] = ((top >> 16) & SGI_SOFT_FIXED_PAIRS) * (256 - b) + ((bot >> 16) & SGI_SOFT_FIXED_PAIRS) * b;
        out[1] = (top & SGI_SOFT_FIXED_PAIRS) * (256 - b) + (bot & SGI_SOFT_FIXED_PAIRS) * b;
        true
    }
}

/// round(128 log2(1 + (i + 0.5) / 1024)): lambda's table, in 1/256.
pub static SOFT_FIXED_LOG2: std::sync::OnceLock<[i16; 1024]> = std::sync::OnceLock::new();

/// The table, built on first use (not at start-up).
#[inline(always)]
pub fn soft_fixed_log2_init() -> &'static [i16; 1024] {
    SOFT_FIXED_LOG2.get_or_init(|| {
        let mut t = [0i16; 1024];

        for (i, v) in t.iter_mut().enumerate() {
            *v = (128.0 * (1.0 + (i as f64 + 0.5) / 1024.0).log2()).round() as i16;
        }
        t
    })
}

/// Lambda in 1/256, log2 of the footprint rho, from rho squared, finite
/// and not negative: half its binary exponent plus the table's value for
/// the top ten bits of its significand. Zero and subnormals give the
/// least lambda, which every level clamp raises.
#[inline(always)]
pub fn soft_fixed_lambda(rho2: f64) -> i32 {
    let bits = rho2.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;

    if exponent == 0 {
        return -SGI_SOFT_FIXED_LAMBDA_MAX;
    }
    (exponent - 1023) * 128 + soft_fixed_log2_init()[((bits >> 42) & 1023) as usize] as i32
}

impl SoftFixedTex {
    /// Lambda in 1/256 for a quad whose footprint under the diagonal rule
    /// (soft_quad_rho()) is rho, as SoftFixedTex::fixed_coords() takes a
    /// fragment's own footprint to one; None where rho squared overflows.
    #[inline(always)]
    pub fn quad_lambda(&self, rho: f64) -> Option<i32> {
        let rho2 = rho * rho;

        (rho2 <= f64::MAX).then(|| self.fixed_decide(rho2, soft_fixed_lambda(rho2)))
    }

    /// Lambda for footprint squared rho2 from its table value lambda, or,
    /// within a few steps of a lambda where the filter or nearest level
    /// changes, computed exactly and rounded up. A lambda of ceil(256 l) is at
    /// or below 256 c exactly when l is at or below c, and rounds to the same
    /// nearest level, so those choices are soft_texture_select()'s; only
    /// the blend between levels takes the table's precision.
    #[inline(always)]
    pub fn fixed_decide(&self, rho2: f64, lambda: i32) -> i32 {
        let clamped = fx_min(fx_max(lambda, self.min_lod), self.max_lod);
        let at_edge = (clamped - self.mag_below).abs() <= 3
            || ((self.min == SGI_GL_NEAREST_MIPMAP_NEAREST || self.min == SGI_GL_LINEAR_MIPMAP_NEAREST)
                && (((clamped - 128) & 255) - 128).abs() >= 125);
        let lmax = SGI_SOFT_FIXED_LAMBDA_MAX as f64;

        if !self.choice || !at_edge {
            return lambda;
        }
        if rho2 > 0.0 {
            fx_min(fx_max((128.0 * rho2.log2()).ceil(), -lmax), lmax) as i32
        } else {
            -SGI_SOFT_FIXED_LAMBDA_MAX
        }
    }

    /// A fragment's s, t and lambda from the triangle's planes g at pixel
    /// column px, whose row terms are row; None where the fixed sampler
    /// cannot take it. lambda, where given, is the fragment's: its quad's
    /// (SoftFixedTex::quad_lambda()). Without it the fragment takes its own
    /// footprint's.
    #[inline(always)]
    pub fn fixed_coords(&self, g: &SoftTexPlane, row: &[f64; 4], px: i32, lambda: Option<i32>) -> Option<(f64, f64, i32)> {
        let x = px as f64 + 0.5 - g.x;
        let q = g.value[2] + x * g.dx[2] + row[2];

        if !(q > 0.0) {
            return None;
        }
        let iq = 1.0 / q;
        let s = (g.value[0] + x * g.dx[0] + row[0]) * iq;
        let t = (g.value[1] + x * g.dx[1] + row[1]) * iq;
        if !(s.abs() < 1073741824.0 && t.abs() < 1073741824.0) {
            return None;
        }
        if !self.lod || lambda.is_some() {
            return Some((s, t, lambda.unwrap_or(0)));
        }
        // d(S/Q)/dx is (dS/dx - s dQ/dx) / Q, and so on.
        let a = self.ns * (g.dx[0] - s * g.dx[2]) * iq;
        let b = self.nt * (g.dx[1] - t * g.dx[2]) * iq;
        let c = self.ns * (g.dy[0] - s * g.dy[2]) * iq;
        let e = self.nt * (g.dy[1] - t * g.dy[2]) * iq;
        let rho2 = fx_max(a * a + b * b, c * c + e * e);
        if !(rho2 <= f64::MAX) {
            return None;
        }
        Some((s, t, self.fixed_decide(rho2, soft_fixed_lambda(rho2))))
    }
}

/// The fixed-point sampler's choice of kernel, a constant in the
/// specialised row loops: NEAREST or LINEAR with no footprint, trilinear
/// (LINEAR_MIPMAP_LINEAR minification with LINEAR magnification), or any
/// filter the image has, chosen per fragment.
pub type SoftFixedKernel = u32;
pub const SGI_SOFT_FIXED_ANY: SoftFixedKernel = 0;
pub const SGI_SOFT_FIXED_NEAREST: SoftFixedKernel = 1;
pub const SGI_SOFT_FIXED_LINEAR: SoftFixedKernel = 2;
pub const SGI_SOFT_FIXED_TRILINEAR: SoftFixedKernel = 3;

impl SoftDraw<'_> {
    /// The filtered texel at (s, t) for lambda, each component in [0, 1], as
    /// soft_texture_select() chooses filter, levels and their blend, with
    /// lambda and the blend fraction in 1/256. KERNEL, and REPEAT (both axes
    /// GL_REPEAT), are constants where the caller can name them.
    #[inline(always)]
    pub fn fixed_sample<const KERNEL: u32, const REPEAT: bool>(&self, s: f64, t: f64, mut lambda: i32, texel: &mut [f64; 4]) -> bool {
        let f = self.fixed_tex();
        let unit: f64 = 1.0 / (255.0 * (1u32 << 24) as f64);
        let mut first: u32 = 0;
        let mut second: u32 = 0;
        let mut alpha: u32 = 0;
        let mut filter = if KERNEL == SGI_SOFT_FIXED_NEAREST {
            SGI_GL_NEAREST
        } else if KERNEL == SGI_SOFT_FIXED_ANY {
            f.min
        } else {
            SGI_GL_LINEAR
        };
        let mut out = [0u64; 2];
        let mut other = [0u64; 2];

        if KERNEL == SGI_SOFT_FIXED_TRILINEAR || (KERNEL == SGI_SOFT_FIXED_ANY && f.lod) {
            lambda = fx_min(fx_max(lambda, f.min_lod), f.max_lod);
        }
        if KERNEL == SGI_SOFT_FIXED_TRILINEAR {
            // At or below 0, magnification is the same LINEAR on level 0.
            let rel = fx_min(fx_max(lambda, 0), (f.range << 8) as i32) as u32;

            first = rel >> 8;
            alpha = rel & 255;
            second = fx_min(first + 1, f.range);
        } else if KERNEL == SGI_SOFT_FIXED_ANY && f.lod {
            if lambda <= f.mag_below {
                filter = f.mag;
            } else if soft_texture_mipmap(f.min) {
                filter = if f.min == SGI_GL_NEAREST_MIPMAP_NEAREST || f.min == SGI_GL_NEAREST_MIPMAP_LINEAR {
                    SGI_GL_NEAREST
                } else {
                    SGI_GL_LINEAR
                };
                if f.min == SGI_GL_NEAREST_MIPMAP_NEAREST || f.min == SGI_GL_LINEAR_MIPMAP_NEAREST {
                    // The nearest level: lambda rounded, half down.
                    first = if lambda <= 128 {
                        0
                    } else if lambda > (f.range << 8) as i32 + 128 {
                        f.range
                    } else {
                        (lambda.wrapping_add(127) as u32) >> 8
                    };
                } else {
                    let rel = fx_min(fx_max(lambda, 0), (f.range << 8) as i32) as u32;

                    first = rel >> 8;
                    alpha = rel & 255;
                    second = fx_min(first + 1, f.range);
                }
            }
        }
        let fl = &f.level[first as usize];
        let mut u = soft_fixed_floor(s * (fl.width as f64 * 256.0));
        let mut v = soft_fixed_floor(t * (fl.height as f64 * 256.0));
        if !SoftDraw::fixed_level::<REPEAT>(self, first, filter, u, v, &mut out) {
            return false;
        }
        if alpha != 0 {
            // The next level's position: a level half as wide scales s by an
            // exact half, so its floor is this one's halved, rounded down.
            u >>= fl.width_shift.wrapping_sub(f.level[second as usize].width_shift) & 63;
            v >>= fl.height_shift.wrapping_sub(f.level[second as usize].height_shift) & 63;
            if !SoftDraw::fixed_level::<REPEAT>(self, second, filter, u, v, &mut other) {
                return false;
            }
            let a = alpha as u64;
            out[0] = out[0] * (256 - a) + other[0] * a;
            out[1] = out[1] * (256 - a) + other[1] * a;
        } else {
            out[0] <<= 8;
            out[1] <<= 8;
        }
        texel[0] = (out[0] >> 32) as u32 as f64 * unit;
        texel[1] = (out[1] >> 32) as u32 as f64 * unit;
        texel[2] = out[0] as u32 as f64 * unit;
        texel[3] = out[1] as u32 as f64 * unit;
        true
    }
}

/// One triangle's fixed inputs to a quad loop (SoftDraw::fill_quads()).
#[derive(Clone, Copy)]
pub struct SoftTexSpan<'a> {
    /// three, wound as the rows step
    pub v: &'a [SoftVertex; 3],
    pub g: &'a SoftTexPlane,
    /// lambda where Q/W is flat, the same for every quad, or None
    pub flat_lambda: Option<i32>,
    pub flat: &'a SoftFrag,
    pub offset: f64,
    pub area: i64,
    pub step1: i64,
    pub step2: i64,
    pub smooth: bool,
    /// Smooth colour components that are 0 at every vertex, or 1 at
    /// every vertex: interpolation gives exactly that value, the terms
    /// summing to the weights' own sum, so they are not interpolated.
    pub constant: u8,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Large enough that the draw covers more pixels than the texture path
    // prefills for.
    const W: usize = 128;

    /// A colour and a depth plane of W by W words.
    struct Fb<'a> {
        colour: RowsMut<'a, u32>,
        depth: RowsMut<'a, u32>,
    }

    struct FbView<'a> {
        colour: Rows<'a, u32>,
        depth: Rows<'a, u32>,
    }

    impl PlaneStore for Fb<'_> {
        fn planes(&mut self) -> Box<dyn Planes + '_> {
            Box::new(FbView { colour: self.colour.view(), depth: self.depth.view() })
        }
    }

    impl Planes for FbView<'_> {
        fn store_colour(&self, _t: ColourTarget, x: i32, y: i32, c: u32) {
            self.colour.get(y as usize, x as usize, 1).unwrap()[0].set(c);
        }
        fn load_colour(&self, _t: ColourTarget, x: i32, y: i32) -> u32 {
            self.colour.get(y as usize, x as usize, 1).unwrap()[0].get()
        }
        fn store_depth(&self, x: i32, y: i32, z: u32) {
            self.depth.get(y as usize, x as usize, 1).unwrap()[0].set(z);
        }
        fn load_depth(&self, x: i32, y: i32) -> u32 {
            self.depth.get(y as usize, x as usize, 1).unwrap()[0].get()
        }
    }

    struct Hashed;

    impl Texels for Hashed {
        fn texel(&self, _: Option<&GlTexLevel>, x: u32, y: u32) -> Option<u32> {
            let h = x.wrapping_mul(0x9e3779b1) ^ y.wrapping_mul(0x85ebca77) ^ 0x5bd1e995;
            Some((h ^ (h >> 13)) | 0xff)
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

    fn draw(path: RasterTexturePath, filter: u32) -> (Vec<u32>, Vec<u32>) {
        let rules = RasterRules {
            subpixel_bits: 4,
            color_bits: 8,
            depth_bits: 24,
            rounding: SGI_ROUND_HALF_AWAY as u8,
            tie_rule: SGI_TIE_TOP_LEFT_BOTTOM_ORIGIN as u8,
            arith: SGI_ARITH_DOUBLE as u8,
            line_rule: SGI_LINE_COVERAGE_DOUBLE as u8,
            ..Default::default()
        };
        let mut colour = vec![0u32; W * W];
        let mut depth = vec![0x00ffffffu32; W * W];
        let image = GlTexImage {
            width: 8,
            height: 8,
            base_format: SGI_GL_RGBA,
            min_filter: filter,
            mag_filter: filter,
            wrap_s: SGI_GL_REPEAT,
            wrap_t: SGI_GL_REPEAT,
            read_texel: true,
            ..Default::default()
        };
        let target = RasterTarget {
            store_colour: true,
            load_colour: true,
            texture: Some(image),
            ..Default::default()
        };
        let mut id = [0.0f32; 16];
        for k in 0..4 {
            id[k * 5] = 1.0;
        }
        let mut st = GlDrawState {
            modelview: id,
            projection: id,
            viewport: [0, 0, W as i32, W as i32],
            shade_model: SGI_GL_SMOOTH,
            depth_test: 1,
            depth_func: SGI_GL_LEQUAL,
            depth_mask: 1,
            depth_range: [0.0, 1.0],
            draw_buffers: 1,
            colour_mask: SGI_GL_COLOUR_MASK_ALL,
            texture_2d: 1,
            texture_env_mode: SGI_GL_REPLACE,
            ..Default::default()
        };
        st.drawable.bound = 1;
        st.drawable.front = 1;
        st.drawable.clip.n = 1;
        st.drawable.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: W as i32 - 1, y1: W as i32 - 1 };
        // x, y, z, w, then colour; strq at the texture slot.
        let mut verts = vec![0.0f32; 3 * SGI_GL_VERTEX_FLOATS];
        let corners = [(-0.95f32, -0.95f32, 0.0f32, 0.0f32), (0.95, -0.9, 3.0, 0.0), (-0.9, 0.95, 0.5, 4.0)];
        for (i, (x, y, s, t)) in corners.iter().enumerate() {
            let v = &mut verts[i * SGI_GL_VERTEX_FLOATS..(i + 1) * SGI_GL_VERTEX_FLOATS];
            v[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4].copy_from_slice(&[*x, *y, 0.0, 1.0]);
            v[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4].copy_from_slice(&[1.0; 4]);
            v[SGI_GL_V_TEXCOORD..SGI_GL_V_TEXCOORD + 4].copy_from_slice(&[*s, *t, 0.0, 1.0]);
        }
        let stipple = [0u32; 32];
        let mut d = GlDraw::new(SGI_GL_TRIANGLES, 3, &verts, &st, &stipple);
        let mut fb = Fb { colour: RowsMut::new(&mut colour, W), depth: RowsMut::new(&mut depth, W) };

        raster_set_texture_path(path);
        rules.draw(&target, &mut fb, &Hashed, &mut d);
        raster_set_texture_path(SGI_RASTER_TEXTURE_SPANS);
        (colour, depth)
    }

    // Coverage, depth and the pixels written are the double path's under
    // every fixed-point selection; NEAREST texels are exact in both, and
    // LINEAR ones differ by less than the 8-bit weights allow, but do differ
    // (the weights truncate), which shows the fixed sampler ran. The
    // specialised row loop and the general one take the same texels.
    #[test]
    fn fixed_and_double_paths_agree() {
        for filter in [SGI_GL_NEAREST, SGI_GL_LINEAR] {
            let (dc, dz) = draw(SGI_RASTER_TEXTURE_DOUBLE, filter);
            let (sc, sz) = draw(SGI_RASTER_TEXTURE_SPANS, filter);
            let (fc, fz) = draw(SGI_RASTER_TEXTURE_FIXED, filter);
            let tol = if filter == SGI_GL_NEAREST { 0 } else { 2 };
            let mut differ = 0;

            assert!(dc.iter().any(|&c| c != 0), "triangle drew nothing");
            assert_eq!(sc, fc, "span and general loops differ, filter {filter:#x}");
            assert_eq!(sz, fz);
            assert_eq!(dz, fz, "depth differs, filter {filter:#x}");
            for i in 0..W * W {
                assert_eq!(dc[i] == 0, fc[i] == 0, "coverage differs at {i}");
                differ += (dc[i] != fc[i]) as u32;
                for sh in [0, 8, 16] {
                    let a = ((dc[i] >> sh) & 0xff) as i32;
                    let b = ((fc[i] >> sh) & 0xff) as i32;
                    assert!((a - b).abs() <= tol, "pixel {i} differs: {a} vs {b}, filter {filter:#x}");
                }
            }
            assert_eq!(differ == 0, filter == SGI_GL_NEAREST, "differing pixels: {differ}");
        }
    }
}
