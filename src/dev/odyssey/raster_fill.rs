//! Row loops for filled triangles, specialised by draw state and by the
//! family's planes.
//!
//! One row loop, SoftDraw::fill_row(), is written against the Fill trait,
//! whose implementor FillMode carries in const parameters the draw state
//! each pixel would otherwise test: texturing and the fixed-point sampler's
//! kernel, wrap, environment and format; the depth test, its function and
//! write; flat or smooth shading; blending; fog. A parameter of FILL_DYN
//! reads its state at run time instead. So the same source is both a loop
//! with every test folded away and the fallback that takes any state the
//! loops admit, and the two cannot disagree: the arithmetic is the same
//! expressions in the same order, only the branches around it differ.
//!
//! The loop is also generic over the Planes implementation, through
//! Planes::fill_row(), whose default body each implementation inherits as
//! its own copy: a plane access in the loop is a direct call the compiler
//! can inline, not one through the draw's `dyn Planes`.
//!
//! FillRow names the specialisations, the states measured as common;
//! SoftDraw::fill_select() and SoftDraw::fixed_span_select() choose one per
//! draw, or the fallback. What a loop does per fragment is what
//! SoftDraw::fragment_at() does for the same state, through the same
//! functions and plane calls; only the order of plane reads differs (the
//! depth read before the texture's). The tests at the end compare every
//! loop against the general path pixel for pixel.

use super::*;

/// A FillMode parameter that reads its state at run time.
pub const FILL_DYN: u32 = u32::MAX;

/// BLEND: off, the alpha-over blend (SRC_ALPHA, ONE_MINUS_SRC_ALPHA, added),
/// or FILL_DYN for any blend the draw state names.
pub const FILL_BLEND_OFF: u32 = 0;
pub const FILL_BLEND_OVER: u32 = 1;

/// Which row loops filled triangles take, for tests and the benchmark:
/// TABLE the specialisations where the state has one, DYNAMIC the
/// fallback loops for every state they admit, OFF no loop for untextured
/// fills (the general per-fragment path; textured fills keep their own
/// switch, raster_set_texture_path()).
pub type RasterFillPath = u32;
pub const SGI_RASTER_FILL_TABLE: RasterFillPath = 0;
pub const SGI_RASTER_FILL_DYNAMIC: RasterFillPath = 1;
pub const SGI_RASTER_FILL_OFF: RasterFillPath = 2;

thread_local! {
    pub static SOFT_FILL_PATH: std::cell::Cell<RasterFillPath> = const { std::cell::Cell::new(SGI_RASTER_FILL_TABLE) };
}

/// Select the row loops for draws made on this thread. Per thread, so that
/// tests running at once cannot change each other's path.
pub fn raster_set_fill_path(path: RasterFillPath) {
    SOFT_FILL_PATH.with(|p| p.set(path));
}

/// The draw state a row loop branches on. Under FillMode each accessor is a
/// constant where its parameter is, and a read of the state where it is
/// FILL_DYN.
pub trait Fill {
    /// The loop that tests at run time all the state this one names: the
    /// fallback with the same texturing.
    type Dyn: Fill;
    /// Fragments are textured through the fixed-point sampler.
    const TEX: bool;
    /// The texture environment and base format, or 0 to read them.
    const ENV: u32;
    const FORMAT: u32;
    /// The fixed-point sampler under the loop's kernel and wrap.
    fn sample(r: &SoftDraw, s: f64, t: f64, lambda: i32, texel: &mut [f64; 4]) -> bool;
    fn depth_test(st: &GlDrawState) -> bool;
    fn depth_func(st: &GlDrawState) -> u32;
    fn depth_write(st: &GlDrawState) -> bool;
    fn smooth(sp: &SoftTexSpan) -> bool;
    fn blend(r: &SoftDraw) -> bool;
    fn blend_src(st: &GlDrawState) -> u32;
    fn blend_dst(st: &GlDrawState) -> u32;
    fn blend_equation(st: &GlDrawState) -> u32;
    fn fog(st: &GlDrawState) -> bool;
}

/// The const draw state of one row loop (see the Fill trait). DEPTH, WRITE,
/// SMOOTH and FOG are 0, 1 or FILL_DYN; FUNC a depth function or FILL_DYN;
/// BLEND one of the FILL_BLEND_ values or FILL_DYN.
pub struct FillMode<
    const TEX: bool,
    const KERNEL: u32,
    const REPEAT: bool,
    const ENV: u32,
    const FORMAT: u32,
    const DEPTH: u32,
    const FUNC: u32,
    const WRITE: u32,
    const SMOOTH: u32,
    const BLEND: u32,
    const FOG: u32,
>;

impl<
        const TEX: bool,
        const KERNEL: u32,
        const REPEAT: bool,
        const ENV: u32,
        const FORMAT: u32,
        const DEPTH: u32,
        const FUNC: u32,
        const WRITE: u32,
        const SMOOTH: u32,
        const BLEND: u32,
        const FOG: u32,
    > Fill for FillMode<TEX, KERNEL, REPEAT, ENV, FORMAT, DEPTH, FUNC, WRITE, SMOOTH, BLEND, FOG>
{
    type Dyn = FillMode<TEX, SGI_SOFT_FIXED_ANY, false, 0, 0, FILL_DYN, FILL_DYN, FILL_DYN, FILL_DYN, FILL_DYN, FILL_DYN>;
    const TEX: bool = TEX;
    const ENV: u32 = ENV;
    const FORMAT: u32 = FORMAT;

    #[inline(always)]
    fn sample(r: &SoftDraw, s: f64, t: f64, lambda: i32, texel: &mut [f64; 4]) -> bool {
        SoftDraw::fixed_sample::<KERNEL, REPEAT>(r, s, t, lambda, texel)
    }
    #[inline(always)]
    fn depth_test(st: &GlDrawState) -> bool {
        if DEPTH == FILL_DYN { st.depth_test != 0 } else { DEPTH != 0 }
    }
    #[inline(always)]
    fn depth_func(st: &GlDrawState) -> u32 {
        if FUNC == FILL_DYN { st.depth_func } else { FUNC }
    }
    #[inline(always)]
    fn depth_write(st: &GlDrawState) -> bool {
        if WRITE == FILL_DYN { st.depth_mask != 0 } else { WRITE != 0 }
    }
    #[inline(always)]
    fn smooth(sp: &SoftTexSpan) -> bool {
        if SMOOTH == FILL_DYN { sp.smooth } else { SMOOTH != 0 }
    }
    #[inline(always)]
    fn blend(r: &SoftDraw) -> bool {
        if BLEND == FILL_DYN { r.blend } else { BLEND != FILL_BLEND_OFF }
    }
    #[inline(always)]
    fn blend_src(st: &GlDrawState) -> u32 {
        if BLEND == FILL_BLEND_OVER { SGI_GL_SRC_ALPHA } else { st.blend_src }
    }
    #[inline(always)]
    fn blend_dst(st: &GlDrawState) -> u32 {
        if BLEND == FILL_BLEND_OVER { SGI_GL_ONE_MINUS_SRC_ALPHA } else { st.blend_dst }
    }
    #[inline(always)]
    fn blend_equation(st: &GlDrawState) -> u32 {
        if BLEND == FILL_BLEND_OVER { SGI_GL_FUNC_ADD } else { st.blend_equation }
    }
    #[inline(always)]
    fn fog(st: &GlDrawState) -> bool {
        if FOG == FILL_DYN { st.fog != 0 } else { FOG != 0 }
    }
}

/// A row's pixels as the window's clip rectangles and the scissor leave
/// them (SoftDraw::fill_run()): px0 to px1, the edge values at px0, and
/// whether a pixel must still test the window's rectangles.
#[derive(Clone, Copy)]
pub struct FillRun {
    pub px0: i32,
    pub px1: i32,
    pub e1: i64,
    pub e2: i64,
    pub clipped: bool,
}

impl SoftDraw<'_> {
    /// GlDrawState::blend_component() with the factors and equation M names.
    #[inline(always)]
    pub fn fill_blend_component<M: Fill>(&self, k: usize, src: &[f64; 4], dst: &[f64; 4]) -> f64 {
        let st: &GlDrawState = self.st;
        let sf = soft_blend_factor(M::blend_src(st), k, src, dst, &st.blend_color);
        let df = soft_blend_factor(M::blend_dst(st), k, src, dst, &st.blend_color);

        match M::blend_equation(st) {
            SGI_GL_MIN => fmin(src[k], dst[k]),
            SGI_GL_MAX => fmax(src[k], dst[k]),
            SGI_GL_FUNC_SUBTRACT => src[k] * sf - dst[k] * df,
            SGI_GL_FUNC_REVERSE_SUBTRACT => dst[k] * df - src[k] * sf,
            _ => src[k] * sf + dst[k] * df,
        }
    }

    /// SoftDraw::rgba_result() under M, for a draw with no logical
    /// operation (SoftDraw::fill_admits() admits none).
    #[inline(always)]
    pub fn fill_rgba_result<M: Fill>(&self, x: i32, y: i32, frag: &SoftFrag, dest: &RasterColour) -> RasterColour {
        let mut out = RasterColour::default();
        let rules = self.rules;
        let max = (1u32 << rules.color_bits) - 1;
        let alpha_max = rules.alpha_max();
        let alpha_q = rules.alpha(frag.rgba[3]);
        let alpha = alpha_q as f64 / alpha_max as f64;
        let blend = M::blend(self);
        let rgb = if blend { rules.rgb(&frag.rgba) } else { self.store_rgb(x, y, &frag.rgba) };

        if !blend && self.rgba_mask == rules.rgba_mask(SGI_GL_COLOUR_MASK_ALL) {
            let stored = (1u32.wrapping_shl(rules.alpha_bits as u32)).wrapping_sub(1);

            for (k, comp) in out.component.iter_mut().enumerate().take(3) {
                *comp = (rgb >> ((2 - k as u32) * rules.color_bits as u32)) & max;
            }
            out.component[3] = if stored == alpha_max { alpha_q } else { alpha_q.wrapping_mul(stored) / alpha_max };
            return out;
        }
        out = rules.rgba_value(rgb, alpha);
        if blend {
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
            for (k, b) in blended.iter_mut().enumerate() {
                *b = self.fill_blend_component::<M>(k, &src, &dst);
            }
            out = rules.rgba_value(self.store_rgb(x, y, &blended), blended[3]);
        }
        for k in 0..4usize {
            if self.rgba_mask & (1u32 << k) == 0 {
                out.component[k] = dest.component[k];
            }
        }
        out
    }

    /// Pixels px0 to px1 of row py of a filled triangle, whose edge values
    /// for v[2]-v[0] and v[0]-v[1] at px0 are e1 and e2, as the window's clip
    /// rectangles and the scissor leave them for the whole row; None where
    /// they leave none.
    #[allow(clippy::too_many_arguments, reason = "a row loop takes the row as its parts")]
    #[inline(always)]
    pub fn fill_run(&self, sp: &SoftTexSpan, py: i32, mut px0: i32, mut px1: i32, mut e1: i64, mut e2: i64) -> Option<FillRun> {
        let st: &GlDrawState = self.st;
        let y = self.win.oy.wrapping_add(py);
        let mut clipped = self.win.clip.n > 1;

        let mut i = 0u32;
        while clipped && i < fx_min(self.win.clip.n, SGI_CLIP_MAX as u32) {
            // One of the window's rectangles holds the whole row.
            let c = &self.win.clip.r[i as usize];

            clipped = !(y >= c.y0
                && y <= c.y1
                && (self.win.ox as i64) + px0 as i64 >= c.x0 as i64
                && (self.win.ox as i64) + px1 as i64 <= c.x1 as i64);
            i += 1;
        }
        if st.scissor_test != 0 && !clipped {
            // SoftDraw::owned_at()'s scissor test, decided for the whole row.
            let sx0 = st.scissor[0] as i64;
            let sx1 = st.scissor[0] as i64 + st.scissor[2] as i64 - 1;

            if !(py >= st.scissor[1] && (py as i64) < st.scissor[1] as i64 + st.scissor[3] as i64) {
                return None;
            }
            if sx0 > px0 as i64 {
                if sx0 > px1 as i64 {
                    return None;
                }
                e1 = e1.wrapping_add((sx0 - px0 as i64).wrapping_mul(sp.step1));
                e2 = e2.wrapping_add((sx0 - px0 as i64).wrapping_mul(sp.step2));
                px0 = sx0 as i32;
            }
            px1 = fx_min(px1 as i64, sx1) as i32;
        }
        (px0 <= px1).then_some(FillRun { px0, px1, e1, e2, clipped })
    }

    /// The pixels of run, on row py of a filled triangle, under state
    /// SoftDraw::fill_select() or SoftDraw::fixed_span_select() admits, and
    /// whose varying parts M names, into planes px: self.px, or the row of
    /// them Planes::row() lends. A textured loop stops at the first fragment
    /// the fixed sampler cannot take and returns its column, which the
    /// caller draws through the general loop; otherwise it returns end.
    #[inline(always)]
    pub fn fill_row<M: Fill, P: Planes + ?Sized>(&self, px: &P, sp: &SoftTexSpan, py: i32, run: FillRun, end: i32) -> i32 {
        let rules: &RasterRules = self.rules;
        let st: &GlDrawState = self.st;
        let t: &RasterTarget = self.t;
        let v: &[SoftVertex; 3] = sp.v;
        let y = self.win.oy.wrapping_add(py);
        let FillRun { px0, px1, mut e1, mut e2, clipped } = run;
        let depth_test = M::depth_test(st);
        let depth_func = M::depth_func(st);
        let depth_write = depth_test && M::depth_write(st);
        let fog = M::fog(st);
        let smooth = M::smooth(sp);
        let rgba = t.store_rgba;
        let rgba_read = rgba && self.rgba_read;
        let store = self.rgba_mask != 0 && self.ntargets != 0;
        let env = if !M::TEX { 0 } else if M::ENV != 0 { M::ENV } else { st.texture_env_mode };
        let format = if !M::TEX { 0 } else if M::FORMAT != 0 { M::FORMAT } else { self.fixed_tex().format };
        // REPLACE leaves none of the colour of these formats' fragments.
        let keeps = !M::TEX
            || env != SGI_GL_REPLACE
            || (format != SGI_GL_RGBA && format != SGI_GL_INTENSITY && format != SGI_GL_LUMINANCE_ALPHA);
        let mut frag: SoftFrag = *sp.flat;
        let mut dest = [RasterColour::default(); SGI_GL_MAX_TARGETS as usize];

        // Nothing in an untextured flat colour varies along the row unless it
        // is dithered (SoftDraw::store_rgb()), fogged or blended.
        let uniform = rgba && store && !M::TEX && !smooth && !fog && !self.dither && !rgba_read;
        let flat_out = if uniform {
            self.fill_rgba_result::<M>(self.win.ox.wrapping_add(px0), y, &frag, &dest[0])
        } else {
            RasterColour::default()
        };
        // The polygon offset, split between window depth and depth-word steps
        // (RasterRules::split_offset()).
        let (offset, steps) = rules.split_offset(sp.offset);
        let mut col = px0;
        let mut first_pixel = true;
        loop {
            // The loop increment, run before every pixel but the first so
            // that a continue still steps.
            if !first_pixel {
                col += 1;
                e1 = e1.wrapping_add(sp.step1);
                e2 = e2.wrapping_add(sp.step2);
            }
            first_pixel = false;
            if col > px1 {
                break;
            }
            let x = self.win.ox.wrapping_add(col);
            let b1 = e1 as f64 / sp.area as f64;
            let b2 = e2 as f64 / sp.area as f64;
            let mut z = v[0].z + b1 * (v[1].z - v[0].z) + b2 * (v[2].z - v[0].z);
            let w0 = (1.0 - b1 - b2) * v[0].inv_w;
            let w1 = b1 * v[1].inv_w;
            let w2 = b2 * v[2].inv_w;
            let sum = w0 + w1 + w2;
            let mut off = offset;
            let mut zw = 0.0f64;
            let mut zq = 0u32;

            if rules.depth_row_given {
                let depth_w = (1.0 - b1 - b2) * v[0].depth_w + b1 * v[1].depth_w + b2 * v[2].depth_w;
                let slack = rules.depth_slack;

                z = (z + off) / depth_w;
                off = 0.0;
                if z < 0.0 && z >= -slack {
                    z = 0.0;
                } else if z > 1.0 && z <= 1.0 + slack {
                    z = 1.0;
                }
            }
            if !st.window_depth(z, off, &mut zw) || !(0.0..=1.0).contains(&zw) {
                continue;
            }
            if clipped && !self.owned_at(col, py) {
                continue;
            }
            if depth_test {
                zq = rules.depth_code_offset(zw, steps);
                if !soft_compare(depth_func, zq, px.load_depth(x, y)) {
                    continue;
                }
            }
            if M::TEX {
                match self.fixed_tex().fixed_coords(sp.g, &sp.row, col, sp.flat_lambda) {
                    Some((s, tc, lambda)) if M::sample(self, s, tc, lambda, &mut frag.texel) => {}
                    _ => return col,
                }
            }
            if uniform {
                if depth_write {
                    px.store_depth(x, y, zq);
                }
                for &target in &self.targets[..self.ntargets as usize] {
                    px.store_rgba(target, x, y, &flat_out);
                }
                continue;
            }
            if smooth && keeps {
                for k in 0..4 {
                    frag.rgba[k] = if sp.constant & (1u8 << k) != 0 {
                        v[0].rgba[k]
                    } else {
                        (w0 * v[0].rgba[k] + w1 * v[1].rgba[k] + w2 * v[2].rgba[k]) / sum
                    };
                }
            } else {
                frag.rgba = sp.flat.rgba;
            }
            if M::TEX {
                self.texture_apply(format, env, &mut frag.texel, &mut frag.rgba);
            }
            if fog {
                frag.fog = (w0 * v[0].fog + w1 * v[1].fog + w2 * v[2].fog) / sum;
                self.fog(&mut frag, zw);
            }
            if rgba_read && !rules.rgba_load(px, &self.targets, self.ntargets, x, y, &mut dest, &self.reports.rgba) {
                continue;
            }
            if depth_write {
                px.store_depth(x, y, zq);
            }
            if !rgba {
                // Packed colour planes: the general store.
                self.fragment_store(x, y, &frag, &dest);
            } else if store {
                for i in 0..self.ntargets as usize {
                    let out = self.fill_rgba_result::<M>(x, y, &frag, &dest[i]);

                    px.store_rgba(self.targets[i], x, y, &out);
                }
            }
        }
        end
    }
}

/// Generates FillRow, one variant per specialisation, and the dispatch
/// from a variant to its instance of SoftDraw::fill_row(). Each entry is the
/// texture's kernel, wrap, environment and format, then the depth test,
/// function and write, shading, blend and fog.
macro_rules! fill_rows {
    ($($id:ident = ($tex:expr, $kernel:expr, $repeat:expr, $env:expr, $format:expr,
                    $depth:expr, $func:expr, $write:expr, $smooth:expr, $blend:expr, $fog:expr);)*) => {
        /// A row loop for filled triangles (raster_fill.rs).
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum FillRow {
            $($id,)*
        }

        /// Planes::fill_row()'s body: the row loop row names, on planes px.
        #[allow(clippy::too_many_arguments, reason = "a row loop takes the row as its parts")]
        #[inline]
        pub fn fill_row_dispatch<P: Planes + ?Sized>(
            px: &P,
            r: &SoftDraw,
            row: FillRow,
            sp: &SoftTexSpan,
            py: i32,
            px0: i32,
            px1: i32,
            e1: i64,
            e2: i64,
        ) -> i32 {
            match row {
                $(FillRow::$id => fill_row_instance::<FillMode<
                    { $tex }, { $kernel }, { $repeat }, { $env }, { $format },
                    { $depth }, { $func }, { $write }, { $smooth }, { $blend }, { $fog },
                >, P>(px, r, sp, py, px0, px1, e1, e2),)*
            }
        }
    };
}

/// One specialisation as a function of its own. It writes through the row
/// of planes Planes::row() lends, which tests the plane bounds and clip
/// rectangles once for the row; where the planes cannot lend one, the
/// fallback with the same texturing draws the row a pixel at a time
/// through the planes' own methods, which give the same pixels.
#[allow(clippy::too_many_arguments, reason = "a row loop takes the row as its parts")]
#[inline(never)]
fn fill_row_instance<M: Fill, P: Planes + ?Sized>(
    px: &P,
    r: &SoftDraw,
    sp: &SoftTexSpan,
    py: i32,
    px0: i32,
    px1: i32,
    e1: i64,
    e2: i64,
) -> i32 {
    let end = px1 + 1;
    let Some(run) = r.fill_run(sp, py, px0, px1, e1, e2) else {
        return end;
    };
    let y = r.win.oy.wrapping_add(py);

    match px.row(y, r.win.ox.wrapping_add(run.px0), r.win.ox.wrapping_add(run.px1)) {
        Some(row) => r.fill_row::<M, GlPlaneRow>(&row, sp, py, run, end),
        None => fill_row_pixels::<M::Dyn, P>(px, r, sp, py, run, end),
    }
}

/// A row a pixel at a time through planes px (fill_row_instance()).
#[inline(never)]
fn fill_row_pixels<M: Fill, P: Planes + ?Sized>(px: &P, r: &SoftDraw, sp: &SoftTexSpan, py: i32, run: FillRun, end: i32) -> i32 {
    r.fill_row::<M, P>(px, sp, py, run, end)
}

const D: u32 = FILL_DYN;
const OFF: u32 = FILL_BLEND_OFF;
const OVER: u32 = FILL_BLEND_OVER;
const NEAREST: u32 = SGI_SOFT_FIXED_NEAREST;
const LINEAR: u32 = SGI_SOFT_FIXED_LINEAR;
const TRILINEAR: u32 = SGI_SOFT_FIXED_TRILINEAR;
const REPLACE: u32 = SGI_GL_REPLACE;
const MODULATE: u32 = SGI_GL_MODULATE;
const RGBA: u32 = SGI_GL_RGBA;
const RGB: u32 = SGI_GL_RGB;
const LESS: u32 = SGI_GL_LESS;
const LEQUAL: u32 = SGI_GL_LEQUAL;

fill_rows! {
    // The fallbacks: any state the loops take, tested per pixel.
    TextureAny = (true, SGI_SOFT_FIXED_ANY, false, 0, 0, D, D, D, D, D, D);
    FillAny = (false, 0, false, 0, 0, D, D, D, D, D, D);

    // Textured: a repeated RGB or RGBA image, nearest, bilinear or
    // trilinear, replacing or modulating the fragment's colour, depth
    // tested LESS or LEQUAL and written, without blending or fog.
    NearestReplaceRgbaLess = (true, NEAREST, true, REPLACE, RGBA, 1, LESS, 1, D, OFF, 0);
    NearestReplaceRgbLess = (true, NEAREST, true, REPLACE, RGB, 1, LESS, 1, D, OFF, 0);
    NearestModulateRgbaLess = (true, NEAREST, true, MODULATE, RGBA, 1, LESS, 1, D, OFF, 0);
    NearestModulateRgbLess = (true, NEAREST, true, MODULATE, RGB, 1, LESS, 1, D, OFF, 0);
    LinearReplaceRgbaLess = (true, LINEAR, true, REPLACE, RGBA, 1, LESS, 1, D, OFF, 0);
    LinearReplaceRgbLess = (true, LINEAR, true, REPLACE, RGB, 1, LESS, 1, D, OFF, 0);
    LinearModulateRgbaLess = (true, LINEAR, true, MODULATE, RGBA, 1, LESS, 1, D, OFF, 0);
    LinearModulateRgbLess = (true, LINEAR, true, MODULATE, RGB, 1, LESS, 1, D, OFF, 0);
    TrilinearReplaceRgbaLess = (true, TRILINEAR, true, REPLACE, RGBA, 1, LESS, 1, D, OFF, 0);
    TrilinearReplaceRgbLess = (true, TRILINEAR, true, REPLACE, RGB, 1, LESS, 1, D, OFF, 0);
    TrilinearModulateRgbaLess = (true, TRILINEAR, true, MODULATE, RGBA, 1, LESS, 1, D, OFF, 0);
    TrilinearModulateRgbLess = (true, TRILINEAR, true, MODULATE, RGB, 1, LESS, 1, D, OFF, 0);
    NearestReplaceRgbaLequal = (true, NEAREST, true, REPLACE, RGBA, 1, LEQUAL, 1, D, OFF, 0);
    NearestReplaceRgbLequal = (true, NEAREST, true, REPLACE, RGB, 1, LEQUAL, 1, D, OFF, 0);
    NearestModulateRgbaLequal = (true, NEAREST, true, MODULATE, RGBA, 1, LEQUAL, 1, D, OFF, 0);
    NearestModulateRgbLequal = (true, NEAREST, true, MODULATE, RGB, 1, LEQUAL, 1, D, OFF, 0);
    LinearReplaceRgbaLequal = (true, LINEAR, true, REPLACE, RGBA, 1, LEQUAL, 1, D, OFF, 0);
    LinearReplaceRgbLequal = (true, LINEAR, true, REPLACE, RGB, 1, LEQUAL, 1, D, OFF, 0);
    LinearModulateRgbaLequal = (true, LINEAR, true, MODULATE, RGBA, 1, LEQUAL, 1, D, OFF, 0);
    LinearModulateRgbLequal = (true, LINEAR, true, MODULATE, RGB, 1, LEQUAL, 1, D, OFF, 0);
    TrilinearReplaceRgbaLequal = (true, TRILINEAR, true, REPLACE, RGBA, 1, LEQUAL, 1, D, OFF, 0);
    TrilinearReplaceRgbLequal = (true, TRILINEAR, true, REPLACE, RGB, 1, LEQUAL, 1, D, OFF, 0);
    TrilinearModulateRgbaLequal = (true, TRILINEAR, true, MODULATE, RGBA, 1, LEQUAL, 1, D, OFF, 0);
    TrilinearModulateRgbLequal = (true, TRILINEAR, true, MODULATE, RGB, 1, LEQUAL, 1, D, OFF, 0);

    // Untextured: no depth test, or LESS or LEQUAL written; flat or
    // smooth; unblended or blended alpha-over; no fog.
    NodepthFlat = (false, 0, false, 0, 0, 0, D, 1, 0, OFF, 0);
    NodepthSmooth = (false, 0, false, 0, 0, 0, D, 1, 1, OFF, 0);
    NodepthFlatOver = (false, 0, false, 0, 0, 0, D, 1, 0, OVER, 0);
    NodepthSmoothOver = (false, 0, false, 0, 0, 0, D, 1, 1, OVER, 0);
    LessFlat = (false, 0, false, 0, 0, 1, LESS, 1, 0, OFF, 0);
    LessSmooth = (false, 0, false, 0, 0, 1, LESS, 1, 1, OFF, 0);
    LessFlatOver = (false, 0, false, 0, 0, 1, LESS, 1, 0, OVER, 0);
    LessSmoothOver = (false, 0, false, 0, 0, 1, LESS, 1, 1, OVER, 0);
    LequalFlat = (false, 0, false, 0, 0, 1, LEQUAL, 1, 0, OFF, 0);
    LequalSmooth = (false, 0, false, 0, 0, 1, LEQUAL, 1, 1, OFF, 0);
    LequalFlatOver = (false, 0, false, 0, 0, 1, LEQUAL, 1, 0, OVER, 0);
    LequalSmoothOver = (false, 0, false, 0, 0, 1, LEQUAL, 1, 1, OVER, 0);
}

/// Textured specialisations by depth function (LESS, LEQUAL), kernel less
/// one, then replace or modulate, then RGBA or RGB.
pub const SOFT_TEXTURE_SPANS: [[[[FillRow; 2]; 2]; 3]; 2] = {
    use FillRow::*;
    [
        [
            [[NearestReplaceRgbaLess, NearestReplaceRgbLess], [NearestModulateRgbaLess, NearestModulateRgbLess]],
            [[LinearReplaceRgbaLess, LinearReplaceRgbLess], [LinearModulateRgbaLess, LinearModulateRgbLess]],
            [[TrilinearReplaceRgbaLess, TrilinearReplaceRgbLess], [TrilinearModulateRgbaLess, TrilinearModulateRgbLess]],
        ],
        [
            [[NearestReplaceRgbaLequal, NearestReplaceRgbLequal], [NearestModulateRgbaLequal, NearestModulateRgbLequal]],
            [[LinearReplaceRgbaLequal, LinearReplaceRgbLequal], [LinearModulateRgbaLequal, LinearModulateRgbLequal]],
            [
                [TrilinearReplaceRgbaLequal, TrilinearReplaceRgbLequal],
                [TrilinearModulateRgbaLequal, TrilinearModulateRgbLequal],
            ],
        ],
    ]
};

/// Untextured specialisations by depth (none, LESS, LEQUAL), shading (flat,
/// smooth), blend (off, over).
pub const SOFT_FILL_TABLE: [[[FillRow; 2]; 2]; 3] = {
    use FillRow::*;
    [
        [[NodepthFlat, NodepthFlatOver], [NodepthSmooth, NodepthSmoothOver]],
        [[LessFlat, LessFlatOver], [LessSmooth, LessSmoothOver]],
        [[LequalFlat, LequalFlatOver], [LequalSmooth, LequalSmoothOver]],
    ]
};

impl SoftDraw<'_> {
    /// Whether the draw's blend is the alpha-over one the specialisations name.
    pub fn fill_blend_over(&self) -> bool {
        let st: &GlDrawState = self.st;

        st.blend_src == SGI_GL_SRC_ALPHA
            && st.blend_dst == SGI_GL_ONE_MINUS_SRC_ALPHA
            && st.blend_equation != SGI_GL_MIN
            && st.blend_equation != SGI_GL_MAX
            && st.blend_equation != SGI_GL_FUNC_SUBTRACT
            && st.blend_equation != SGI_GL_FUNC_REVERSE_SUBTRACT
    }

    /// The state every row loop leaves to SoftDraw::fragment_at(): what none
    /// of them does per fragment. RGBA colour, with no alpha test, logical
    /// operation, stencil test on planes with a stencil buffer, client clip
    /// planes or polygon stipple.
    pub fn fill_admits(&self) -> bool {
        let st: &GlDrawState = self.st;

        st.index_bits == 0
            && st.alpha_test == 0
            && !self.logic_op
            && !(st.stencil_test != 0 && self.rules.stencil_max(self.t).is_some())
            && st.clip_planes == 0
            && self.polygon_stipple.is_none()
    }

    /// The row loop for the draw's untextured filled triangles, or None for
    /// the general per-fragment path.
    pub fn fill_select(&self) -> Option<FillRow> {
        let st: &GlDrawState = self.st;
        let path = SOFT_FILL_PATH.with(|p| p.get());

        if path == SGI_RASTER_FILL_OFF || self.texture.is_some() || !self.fill_admits() {
            return None;
        }
        let depth: i32 = if st.depth_test == 0 {
            0
        } else if st.depth_mask == 0 {
            -1
        } else if st.depth_func == SGI_GL_LESS {
            1
        } else if st.depth_func == SGI_GL_LEQUAL {
            2
        } else {
            -1
        };
        let blend: i32 = if !self.blend {
            0
        } else if self.fill_blend_over() {
            1
        } else {
            -1
        };
        if path == SGI_RASTER_FILL_DYNAMIC || depth < 0 || blend < 0 || st.fog != 0 {
            return Some(FillRow::FillAny);
        }
        let smooth = (st.shade_model != SGI_GL_FLAT) as usize;
        Some(SOFT_FILL_TABLE[depth as usize][smooth][blend as usize])
    }

    /// The row loop for the draw's textured triangles, sampled through f, or
    /// None for the general loop: under SGI_RASTER_TEXTURE_SPANS, for state
    /// SoftDraw::fill_admits() takes (the fixed-point sampler already
    /// excludes the rest). A loop of its own for the common states
    /// SOFT_TEXTURE_SPANS names; the others take the loop that tests the
    /// state per pixel.
    pub fn fixed_span_select(&self, f: &SoftFixedTex) -> Option<FillRow> {
        let st: &GlDrawState = self.st;

        if SOFT_TEXTURE_PATH.with(|p| p.get()) != SGI_RASTER_TEXTURE_SPANS || !self.fill_admits() {
            return None;
        }
        let kernel: i32 = if !f.lod && f.min == SGI_GL_NEAREST {
            0
        } else if !f.lod && f.min == SGI_GL_LINEAR {
            1
        } else if f.min == SGI_GL_LINEAR_MIPMAP_LINEAR && f.mag == SGI_GL_LINEAR {
            2
        } else {
            -1
        };
        let env: i32 = if st.texture_env_mode == SGI_GL_REPLACE {
            0
        } else if st.texture_env_mode == SGI_GL_MODULATE {
            1
        } else {
            -1
        };
        let format: i32 = if f.format == SGI_GL_RGBA {
            0
        } else if f.format == SGI_GL_RGB {
            1
        } else {
            -1
        };
        let func: i32 = if st.depth_func == SGI_GL_LESS {
            0
        } else if st.depth_func == SGI_GL_LEQUAL {
            1
        } else {
            -1
        };
        if SOFT_FILL_PATH.with(|p| p.get()) == SGI_RASTER_FILL_DYNAMIC
            || kernel < 0
            || env < 0
            || format < 0
            || func < 0
            || f.wrap_s != SGI_GL_REPEAT
            || f.wrap_t != SGI_GL_REPEAT
            || st.depth_test == 0
            || st.depth_mask == 0
            || self.blend
            || st.fog != 0
        {
            return Some(FillRow::TextureAny);
        }
        Some(SOFT_TEXTURE_SPANS[func as usize][kernel as usize][env as usize][format as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::odyssey::raster_bench::*;

    /// The square of plane pixels the cases draw in.
    const SIDE: i32 = 160;

    fn pick<T: Copy>(rng: &mut Lcg, from: &[T]) -> T {
        from[rng.next_u32() as usize % from.len()]
    }

    /// A random draw state, biased toward the specialised ones, and
    /// triangles in perspective, some crossing the near and far planes.
    fn case(seed: u64, textured: bool) -> Bench {
        let mut b = Bench::empty();
        let mut rng = Lcg(seed.wrapping_mul(0x9e3779b97f4a7c15) | 1);
        let st = &mut b.st;

        st.depth_test = (rng.next_u32() % 4 != 0) as u32;
        st.depth_func = pick(&mut rng, &[SGI_GL_LESS, SGI_GL_LEQUAL, SGI_GL_LESS, SGI_GL_LEQUAL, SGI_GL_GREATER, SGI_GL_ALWAYS, SGI_GL_EQUAL, SGI_GL_NOTEQUAL, SGI_GL_NEVER]);
        st.depth_mask = (rng.next_u32() % 5 != 0) as u32;
        st.shade_model = pick(&mut rng, &[SGI_GL_FLAT, SGI_GL_SMOOTH]);
        st.dither = rng.next_u32() % 2;
        if rng.next_u32() % 3 == 0 {
            b.rules.dither = SGI_DITHER_BAYER4_TOP_ROWS as u8;
        }
        if rng.next_u32() % 2 == 0 {
            st.blend = 1;
            if rng.next_u32() % 2 == 0 {
                st.blend_src = SGI_GL_SRC_ALPHA;
                st.blend_dst = SGI_GL_ONE_MINUS_SRC_ALPHA;
            } else {
                let factors = [0, 1, SGI_GL_SRC_ALPHA, SGI_GL_ONE_MINUS_SRC_ALPHA, SGI_GL_DST_ALPHA, SGI_GL_ONE_MINUS_DST_COLOR, SGI_GL_SRC_ALPHA_SATURATE, SGI_GL_CONSTANT_ALPHA];
                st.blend_src = pick(&mut rng, &factors);
                st.blend_dst = pick(&mut rng, &factors);
                st.blend_color = [0.25, 0.5, 0.75, 0.3];
            }
            st.blend_equation = pick(&mut rng, &[0, SGI_GL_FUNC_ADD, SGI_GL_FUNC_ADD, SGI_GL_MIN, SGI_GL_FUNC_SUBTRACT]);
        }
        if rng.next_u32() % 4 == 0 {
            st.fog = 1;
            st.fog_mode = pick(&mut rng, &[SGI_GL_LINEAR, SGI_GL_EXP, SGI_GL_EXP2]);
            st.fog_start = 0.0;
            st.fog_end = 2.5;
            st.fog_density = 0.7;
            st.fog_color = [0.1, 0.8, 0.3, 1.0];
            b.builtin_fog = rng.next_u32() % 2 == 0;
        }
        if rng.next_u32() % 4 == 0 {
            st.colour_mask = rng.next_u32() % 16;
        }
        if rng.next_u32() % 4 == 0 {
            st.scissor_test = 1;
            st.scissor = [rng.next_u32() as i32 % 60, rng.next_u32() as i32 % 60, 40 + rng.next_u32() as i32 % 100, 40 + rng.next_u32() as i32 % 100];
        }
        if rng.next_u32() % 4 == 0 {
            st.polygon_offset = SGI_GL_OFFSET_FILL;
            st.offset_factor = 1.5;
            st.offset_units = 2.0;
        }
        if rng.next_u32() % 4 == 0 {
            // Two rectangles: rows that neither holds whole test each pixel.
            st.drawable.clip.n = 2;
            st.drawable.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: 99, y1: SIDE - 1 };
            st.drawable.clip.r[1] = ClipRect { x0: 60, y0: 30, x1: SIDE - 1, y1: 120 };
        }
        if rng.next_u32() % 8 == 0 {
            st.stencil_test = 1;
            st.stencil_func = SGI_GL_ALWAYS;
        }
        if textured {
            let min = pick(&mut rng, &[SGI_GL_NEAREST, SGI_GL_LINEAR, SGI_GL_LINEAR_MIPMAP_LINEAR, SGI_GL_LINEAR_MIPMAP_LINEAR, SGI_GL_NEAREST_MIPMAP_NEAREST]);
            let mag = pick(&mut rng, &[SGI_GL_NEAREST, SGI_GL_LINEAR, SGI_GL_LINEAR]);
            let format = pick(&mut rng, &[SGI_GL_RGBA, SGI_GL_RGB, SGI_GL_RGBA, SGI_GL_LUMINANCE]);
            let mut tex = BenchTexture::new(32, min, mag, format);

            if rng.next_u32() % 4 == 0 {
                tex.wrap_t = SGI_GL_CLAMP_TO_EDGE;
            }
            st.texture_2d = 1;
            st.texture_env_mode = pick(&mut rng, &[SGI_GL_REPLACE, SGI_GL_MODULATE, SGI_GL_MODULATE, SGI_GL_DECAL, SGI_GL_BLEND]);
            st.texture_env_color = [0.9, 0.2, 0.4, 0.6];
            if seed % 2 == 0 {
                // Half the cases in a state a specialisation names.
                let kernel = pick(&mut rng, &[(SGI_GL_NEAREST, SGI_GL_NEAREST), (SGI_GL_LINEAR, SGI_GL_LINEAR), (SGI_GL_LINEAR_MIPMAP_LINEAR, SGI_GL_LINEAR)]);
                tex = BenchTexture::new(32, kernel.0, kernel.1, pick(&mut rng, &[SGI_GL_RGBA, SGI_GL_RGB]));
                st.texture_env_mode = pick(&mut rng, &[SGI_GL_REPLACE, SGI_GL_MODULATE]);
                st.depth_test = 1;
                st.depth_mask = 1;
                st.depth_func = pick(&mut rng, &[SGI_GL_LESS, SGI_GL_LEQUAL]);
                st.blend = 0;
                st.fog = 0;
                st.stencil_test = 0;
            }
            b.tex = Some(tex);
        }
        for _ in 0..60 {
            let mut v = vec![0.0f32; 3 * SGI_GL_VERTEX_FLOATS];
            let size = pick(&mut rng, &[3.0f32, 12.0, 40.0, 90.0]);
            let cx = rng.range(0.0, SIDE as f32);
            let cy = rng.range(0.0, SIDE as f32);
            let grey = rng.next_u32() % 3 == 0;
            for i in 0..3 {
                let x = cx + rng.range(-size, size);
                let y = cy + rng.range(-size, size);
                let z = rng.range(-1.2, 1.2);
                let w = rng.range(0.5, 2.0);
                let o = &mut v[i * SGI_GL_VERTEX_FLOATS..(i + 1) * SGI_GL_VERTEX_FLOATS];
                let nx = x / BENCH_W as f32 * 2.0 - 1.0;
                let ny = y / BENCH_H as f32 * 2.0 - 1.0;
                o[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4].copy_from_slice(&[nx * w, ny * w, z * w, w]);
                // Some triangles keep a component 0 or 1 at every vertex.
                let c = if grey {
                    [1.0, 0.0, rng.range(0.0, 1.0), 1.0]
                } else {
                    [rng.range(-0.1, 1.1), rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.0, 1.0)]
                };
                o[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4].copy_from_slice(&c);
                o[SGI_GL_V_TEXCOORD..SGI_GL_V_TEXCOORD + 4]
                    .copy_from_slice(&[rng.range(-2.0, 3.0) * w, rng.range(-2.0, 3.0) * w, 0.0, w]);
            }
            b.prims.push(v);
        }
        b
    }

    fn draw(seed: u64, textured: bool, fill: RasterFillPath, texture: RasterTexturePath) -> (Vec<u32>, Vec<u32>) {
        draw_bench(case(seed, textured), fill, texture)
    }

    fn draw_bench(mut b: Bench, fill: RasterFillPath, texture: RasterTexturePath) -> (Vec<u32>, Vec<u32>) {
        b.prepare();
        raster_set_fill_path(fill);
        raster_set_texture_path(texture);
        b.run();
        raster_set_fill_path(SGI_RASTER_FILL_TABLE);
        raster_set_texture_path(SGI_RASTER_TEXTURE_SPANS);
        b.snapshot(SIDE)
    }

    fn compare(seed: u64, what: &str, want: &(Vec<u32>, Vec<u32>), got: &(Vec<u32>, Vec<u32>)) {
        for i in 0..want.0.len() {
            assert!(
                want.0[i] == got.0[i] && want.1[i] == got.1[i],
                "seed {seed}, {what}: pixel ({}, {}) colour {:#010x} depth {:#08x}, general path {:#010x} {:#08x}",
                i as i32 % SIDE,
                i as i32 / SIDE,
                got.0[i],
                got.1[i],
                want.0[i],
                want.1[i]
            );
        }
    }

    // Untextured fills: the specialised loops and the dynamic one draw what
    // the general per-fragment path draws, pixel for pixel, colour and depth.
    #[test]
    fn fill_rows_match_the_general_path() {
        let mut drawn = 0;
        for seed in 1..=40u64 {
            let want = draw(seed, false, SGI_RASTER_FILL_OFF, SGI_RASTER_TEXTURE_SPANS);
            let dynamic = draw(seed, false, SGI_RASTER_FILL_DYNAMIC, SGI_RASTER_TEXTURE_SPANS);
            let table = draw(seed, false, SGI_RASTER_FILL_TABLE, SGI_RASTER_TEXTURE_SPANS);

            compare(seed, "dynamic loop", &want, &dynamic);
            compare(seed, "specialised loop", &want, &table);
            drawn += want.0.iter().filter(|&&c| c != 0).count();
        }
        assert!(drawn > 10000, "the cases drew too little to compare: {drawn}");
    }

    // Textured fills: the loops draw what the general loop draws with the
    // same fixed-point sampler.
    #[test]
    fn texture_rows_match_the_general_path() {
        for seed in 100..=140u64 {
            let want = draw(seed, true, SGI_RASTER_FILL_TABLE, SGI_RASTER_TEXTURE_FIXED);
            let dynamic = draw(seed, true, SGI_RASTER_FILL_DYNAMIC, SGI_RASTER_TEXTURE_SPANS);
            let table = draw(seed, true, SGI_RASTER_FILL_TABLE, SGI_RASTER_TEXTURE_SPANS);

            compare(seed, "dynamic textured loop", &want, &dynamic);
            compare(seed, "specialised textured loop", &want, &table);
        }
    }

    // Each textured specialisation, in the state it names, smooth shaded,
    // draws what the general loop draws.
    #[test]
    fn every_texture_row_matches_the_general_path() {
        let mut seed = 200u64;
        for func in [SGI_GL_LESS, SGI_GL_LEQUAL] {
            for (min, mag) in [(SGI_GL_NEAREST, SGI_GL_NEAREST), (SGI_GL_LINEAR, SGI_GL_LINEAR), (SGI_GL_LINEAR_MIPMAP_LINEAR, SGI_GL_LINEAR)] {
                for env in [SGI_GL_REPLACE, SGI_GL_MODULATE] {
                    for format in [SGI_GL_RGBA, SGI_GL_RGB] {
                        let make = || {
                            let mut b = case(seed, true);
                            b.st.depth_test = 1;
                            b.st.depth_mask = 1;
                            b.st.depth_func = func;
                            b.st.shade_model = SGI_GL_SMOOTH;
                            b.st.blend = 0;
                            b.st.fog = 0;
                            b.st.stencil_test = 0;
                            b.st.texture_env_mode = env;
                            b.tex = Some(BenchTexture::new(32, min, mag, format));
                            b
                        };
                        let want = draw_bench(make(), SGI_RASTER_FILL_TABLE, SGI_RASTER_TEXTURE_FIXED);
                        let table = draw_bench(make(), SGI_RASTER_FILL_TABLE, SGI_RASTER_TEXTURE_SPANS);

                        compare(seed, "specialised textured loop", &want, &table);
                        seed += 1;
                    }
                }
            }
        }
    }

    // The board's own pixel tests: an id test that passes some pixels of a
    // row and not others, which the row of planes Planes::row() lends
    // applies per pixel, and two clip rectangles that split rows, which it
    // cannot lend and the loops then draw a pixel at a time. Either way the
    // loops draw what the general path draws.
    #[test]
    fn fill_rows_match_under_the_board_pixel_tests() {
        for seed in 300..=330u64 {
            let textured = seed % 2 == 0;
            let make = || {
                let mut b = case(seed, textured);
                let mut rng = Lcg(seed);

                if seed % 3 != 0 {
                    b.board.id_compare = ODSY_ID_COMPARE_ENABLE | (0x0f << ODSY_ID_COMPARE_MASK_SHIFT) | 0x05;
                    for d in b.board.did.iter_mut() {
                        *d = if rng.next_u32() % 4 == 0 { 0x15 } else { 0x06 };
                    }
                }
                if seed % 3 != 1 {
                    b.board.clip_kept = 2;
                    b.board.clip_rect[0] = ClipRect { x0: 0, y0: 0, x1: 70, y1: SIDE - 1 };
                    b.board.clip_rect[1] = ClipRect { x0: 50, y0: 40, x1: SIDE - 1, y1: 110 };
                }
                b
            };
            let (fill, texture) = if textured {
                (SGI_RASTER_FILL_TABLE, SGI_RASTER_TEXTURE_FIXED)
            } else {
                (SGI_RASTER_FILL_OFF, SGI_RASTER_TEXTURE_SPANS)
            };
            let want = draw_bench(make(), fill, texture);
            let table = draw_bench(make(), SGI_RASTER_FILL_TABLE, SGI_RASTER_TEXTURE_SPANS);

            compare(seed, "row loop under the board's pixel tests", &want, &table);
        }
    }

    // Every untextured specialisation is reachable from the state it names.
    #[test]
    fn selection_reaches_each_table_entry() {
        let mut seen = Vec::new();
        for (depth, table) in SOFT_FILL_TABLE.iter().enumerate() {
            for smooth in [SGI_GL_FLAT, SGI_GL_SMOOTH] {
                for over in [false, true] {
                    let mut b = Bench::empty();
                    b.st.depth_test = (depth != 0) as u32;
                    b.st.depth_func = if depth == 2 { SGI_GL_LEQUAL } else { SGI_GL_LESS };
                    b.st.shade_model = smooth;
                    if over {
                        b.st.blend = 1;
                        b.st.blend_src = SGI_GL_SRC_ALPHA;
                        b.st.blend_dst = SGI_GL_ONE_MINUS_SRC_ALPHA;
                    }
                    let t = b.board.gl_target();
                    let reports = SoftReports::default();
                    let mut r = SoftDraw::new(&b.rules, &t, &b.st, &reports);
                    r.ntargets = 1;
                    assert!(r.setup_ops());
                    let f = r.fill_select().expect("a row loop");
                    assert_eq!(f, table[(smooth == SGI_GL_SMOOTH) as usize][over as usize]);
                    if !seen.contains(&f) {
                        seen.push(f);
                    }
                }
            }
        }
        assert_eq!(seen.len(), 12);
    }
}
