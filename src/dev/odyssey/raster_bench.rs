//! A fill-rate benchmark for the drawing pipeline, and the scenes it draws.
//!
//! Each scene drives Board::gl_draw() the way the GL context does: one
//! primitive per draw, into a v12 board's own memory, its state words
//! encoded from a draw state (pipe/encode.rs) and its textures loaded into
//! board memory (load_texture()). No guest is involved, so the numbers are
//! the pipeline's, not the command decoder's or the geometry program's.
//!
//! Run it from a release build, one test thread:
//!
//!     cargo test --release --lib odyssey::raster_bench -- --ignored --nocapture --test-threads=1
//!
//! BENCH_SCENE picks one scene, BENCH_SECS the time each takes (default
//! one second).

use crate::dev::odyssey::draw::{RasterOp, RASTER_CLEAR};
use super::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use super::consts::{gl_cmd, tex};
use super::pipe::encode::{
    vtx, GL_ALWAYS, GL_COLOR_BUFFER_BIT, GL_COLOUR_MASK_ALL, GL_DEPTH_BUFFER_BIT, GL_EQUAL, GL_FLAT, GL_INCR,
    GL_KEEP, GL_LEQUAL, GL_LESS, GL_LINES, GL_ONE_MINUS_SRC_ALPHA, GL_QUADS, GL_SMOOTH, GL_SRC_ALPHA,
    GL_TRIANGLES, LibDraw, LibState,
};
use super::clip::ClipRect;
pub(super) const BENCH_W: i32 = ODSY_WIDTH as i32;
pub(super) const BENCH_H: i32 = ODSY_HEIGHT as i32;

/// A deterministic generator, so every run and every path draws the same.
pub(super) struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// In [lo, hi).
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next_u32() as f32 / (1u64 << 31) as f32)
    }
}

/// Texels that are a hash of their level and position, so a wrong filter or
/// level shows.
// Textures. Each level lives in board memory where its descriptor places
// it, the way the library's uploads leave them.

#[derive(Clone, Copy)]
pub struct Tex {
    /// The sampler word's component code: 0x01000000 luminance, 0x01800000
    /// luminance-alpha, 0x00800000 alpha, 0x02000000 intensity, 0x02800000
    /// RGB, 0x03000000 RGBA.
    pub components: u32,
    pub size: u32,
    /// 0 nearest, 1 linear, 2 to 5 the mipmapped filters in GL's order.
    pub min: u32,
    pub mag_linear: bool,
    /// 0 repeat, 1 clamp to edge.
    pub wrap: u32,
    /// 1 replace, 2 modulate, 3 decal, 4 blend.
    pub env: u32,
}

pub const LUMINANCE: u32 = 0x0100_0000;
pub const LUMINANCE_ALPHA: u32 = 0x0180_0000;
pub const ALPHA: u32 = 0x0080_0000;
pub const INTENSITY: u32 = 0x0200_0000;
pub const RGB: u32 = 0x0280_0000;
pub const RGBA: u32 = 0x0300_0000;

impl Tex {
    pub fn new(components: u32) -> Tex {
        Tex { components, size: 16, min: 0, mag_linear: false, wrap: 0, env: 2 }
    }
}

/// Texel (x, y) of a level: a gradient and a check, tinted by level so a
/// wrong level shows.
pub fn texel(level: u32, x: u32, y: u32, w: u32, h: u32) -> u32 {
    let r = x * 255 / w.max(2).saturating_sub(1).max(1);
    let g = y * 255 / h.max(2).saturating_sub(1).max(1);
    let b = if (x / 2 + y / 2) & 1 == 0 { 230 } else { 40 };
    let a = 255 - ((x + y) * 200 / (w + h)).min(200);
    let tint = [0, 60, 120, 30, 90, 150, 20, 70][level as usize & 7];
    ((r.min(255) ^ tint) << 24) | (g.min(255) << 16) | (b << 8) | a
}

/// Level descriptor word 1's size code and format index for a sampler
/// component code: alpha 8 (index 0), luminance 8 (4), luminance-alpha 8/8
/// (11), intensity 8 (2), and 8/8/8/8 (17) for RGB and RGBA, which the class
/// alone tells apart.
/// Store a texel where level descriptor words (word0, word1) place it.
pub fn put_texel(b: &Board, word0: u32, word1: u32, x: u32, y: u32, texel: u32) {
    use super::view::{Rows, View, MAX_DIM};
    let (surface, field) = super::field::level_surface(word0, word1).expect("a level the library emits");
    Rows::new(b.vram_ptr(), View::colour(&surface, field, MAX_DIM)).store_word(x as i32, y as i32, texel);
}

pub fn texel_format_word(components: u32) -> u32 {
    match components {
        ALPHA => 0x0000_0000,
        LUMINANCE => 0x0040_0000,
        LUMINANCE_ALPHA => 0x00b0_0000,
        INTENSITY => 0x0020_0000,
        _ => 0x4110_0000,
    }
}

/// Load texture t into the board's memory and its texture words, as the
/// library's uploads and texture activation leave them.
pub fn load_texture(b: &mut Board, st: &mut LibState, t: &Tex) {
    {
        let levels = if t.min >= 2 { t.size.trailing_zeros() + 1 } else { 1 };
        let (ws, wt) = (t.wrap, t.wrap);

        let mut base = 0x10_0000;
        for level in 0..levels {
            let w = (t.size >> level).max(1);
            let units = (w.div_ceil(16) + 1) & !1;
            let word1 = texel_format_word(t.components) | units;
            for y in 0..w {
                for x in 0..w {
                    put_texel(b, base, word1, x, y, texel(level, x, y, w, w));
                }
            }
            b.tex_mut().levels[level as usize * 3..][..3].copy_from_slice(&[base, word1, 0]);
            // The next level after this one's rows of tiles (16-byte units).
            base += units * 256 * 8 * w.div_ceil(16) / 16;
        }
        let axis = |w: u32, wrap: u32| w | if wrap == 1 { 0x4_0000 } else { 0 };
        let filter = (t.min << 5) | if t.mag_linear { 2 } else { 0 };
        b.tex_mut().path = tex::MODE_2D;
        b.tex_mut().level0 = 0;
        b.tex_mut().sampler = [
            0x0800_0000 | (t.size.trailing_zeros() << 10) | (t.size.trailing_zeros() << 5),
            axis(t.size, ws),
            axis(t.size, wt),
            1,
            0x1000_0000 | t.components | (ws << 12) | (wt << 10) | filter,
            0,
            0,
        ];
        b.tex_mut().env = [t.env, 0, 0, 0, 0];
        if t.env == 4 {
            // The environment colour: two normalized 12-bit components a word.
            b.tex_mut().env[1] = 0xc00 << 12 | 0x400;
            b.tex_mut().env[2] = 0x200 << 12 | 0xfff;
        }
        st.texture_2d = true;
    }
}

/// What a scene's primitives are and how they are drawn.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Scene {
    /// Flat-shaded triangles, no depth test: 2D-style GL fills.
    FlatFill,
    /// Smooth-shaded, depth-tested triangles.
    SmoothDepth,
    /// Smooth, depth-tested, alpha-blended triangles.
    SmoothBlend,
    /// Textured, depth-tested, modulated, bilinear.
    TexBilinearDepth,
    /// Textured, depth-tested, modulated, trilinear.
    TexTrilinearDepth,
    /// Textured with fog and blending: the textured fallback loop.
    TexFogBlend,
    /// Textured, depth-tested, trilinear, in perspective: each quad takes
    /// its own level of detail.
    TexPerspective,
    /// Smooth, depth-tested triangles of some thousands of pixels.
    MidSmooth,
    /// Screen-filling quads, smooth and depth tested.
    BigQuads,
    /// Screen-filling trilinear, depth-tested quads: GLperf's fill test.
    BigTexTrilinear,
    /// One-pixel lines.
    Lines,
    /// Colour and depth clears of the whole screen.
    Clear,
    /// Textured, bilinear RGB, depth tested, inside a stencil mask: the
    /// stencil EQUAL test, every operation KEEP (Smoke & Mirrors' scene).
    TexStencil,
    /// Untextured, writing only the stencil: ALWAYS, incremented where the
    /// depth test passes, colour and depth masked (a mirror's mask).
    StencilMark,
    /// Trilinear RGB, flat, depth LEQUAL, s clamped to the edge, in
    /// perspective (perfly's ground and buildings).
    TexClampTrilinear,
    /// Trilinear RGBA blended over what is there, depth ALWAYS and not
    /// written (perfly's sky and overlays).
    TexBlendOver,
    /// TexStencil's state under a client clip plane that cuts the screen
    /// (Smoke & Mirrors' reflected scene).
    TexStencilClip,
    /// Small bilinear RGB quads, depth tested: each quad's two triangles
    /// validated as a pair before either draws.
    TexQuads,
    /// Bilinear RGB blended over what is there, depth tested LESS and
    /// written (Smoke & Mirrors' glass).
    TexGlass,
    /// Antialiased triangles, smooth shaded, blended and depth tested.
    PolySmooth,
}

pub(super) const ALL_SCENES: [Scene; 20] = [
    Scene::FlatFill,
    Scene::SmoothDepth,
    Scene::SmoothBlend,
    Scene::TexBilinearDepth,
    Scene::TexTrilinearDepth,
    Scene::TexFogBlend,
    Scene::TexPerspective,
    Scene::MidSmooth,
    Scene::BigQuads,
    Scene::BigTexTrilinear,
    Scene::Lines,
    Scene::Clear,
    Scene::TexStencil,
    Scene::StencilMark,
    Scene::TexClampTrilinear,
    Scene::TexBlendOver,
    Scene::TexStencilClip,
    Scene::TexQuads,
    Scene::TexGlass,
    Scene::PolySmooth,
];

/// One scene, built: the board, the draw state, the vertex lists of its
/// primitives and the pixels they stand for.
pub(super) struct Bench {
    pub board: Box<Board>,
    pub st: LibState,
    pub tex: Option<Tex>,
    /// Each primitive's vertices, vtx::FLOATS apiece.
    pub prims: Vec<Vec<f32>>,
    pub mode: u32,
    /// The pixels the scene covers, by area, for the rate.
    pub pixels: f64,
    pub clears: usize,
    /// Polygons antialiased (the draw's own flag, not the state's).
    pub polygon_smooth: bool,
}

fn identity() -> [f32; 16] {
    let mut m = [0.0f32; 16];
    for k in 0..4 {
        m[k * 5] = 1.0;
    }
    m
}

fn ndc(x: f32, size: i32) -> f32 {
    x / size as f32 * 2.0 - 1.0
}

/// A triangle of about side 2 * size pixels around a random centre, with
/// random colours and depths, its texture coordinates texscale a pixel; and
/// its area in pixels.
pub(super) fn triangle(rng: &mut Lcg, size: f32, texscale: f32) -> (Vec<f32>, f64) {
    let cx = rng.range(0.0, BENCH_W as f32);
    let cy = rng.range(0.0, BENCH_H as f32);
    let mut v = vec![0.0f32; 3 * vtx::FLOATS];
    let mut xy = [(0.0f32, 0.0f32); 3];
    let s0 = rng.range(0.0, 4.0);
    let t0 = rng.range(0.0, 4.0);
    for i in 0..3 {
        let x = cx + rng.range(-size, size);
        let y = cy + rng.range(-size, size);
        let z = rng.range(-0.9, 0.9);
        let o = &mut v[i * vtx::FLOATS..(i + 1) * vtx::FLOATS];
        xy[i] = (x.clamp(0.0, BENCH_W as f32), y.clamp(0.0, BENCH_H as f32));
        o[vtx::POSITION..vtx::POSITION + 4].copy_from_slice(&[ndc(x, BENCH_W), ndc(y, BENCH_H), z, 1.0]);
        o[vtx::COLOR..vtx::COLOR + 4]
            .copy_from_slice(&[rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.2, 1.0)]);
        o[vtx::TEXCOORD..vtx::TEXCOORD + 4]
            .copy_from_slice(&[s0 + (x - cx) * texscale, t0 + (y - cy) * texscale, 0.0, 1.0]);
    }
    let area = ((xy[1].0 - xy[0].0) * (xy[2].1 - xy[0].1) - (xy[2].0 - xy[0].0) * (xy[1].1 - xy[0].1)).abs() / 2.0;
    (v, area as f64)
}

impl Bench {
    /// The board and draw state every scene starts from, with no primitives.
    pub fn empty() -> Bench {
        let mut board = Board::new("v12", 11).expect("board");
        board.test_screen();

        // Every pixel of the screen is the window's.
        board.win.clip_kept = 1;
        board.win.clip_rect[0] = ClipRect { x0: 0, y0: 0, x1: BENCH_W - 1, y1: BENCH_H - 1 };
        // Depth (z + w) / 2 over w: the identity projection's [-1, 1] mapped
        // to [0, 1], in the combined matrix's words as Board::gl_depth_row()
        // unscales them.
        let scale = 2.0 * gl_cmd::DEPTH_S * board.gl_depth_scale();
        for (k, r) in [0.0f64, 0.0, 0.5, 0.5].into_iter().enumerate() {
            board.gl_combined[8 + k] = ((r * scale) as f32).to_bits();
        }

        let st = LibState {
            modelview: identity(),
            projection: identity(),
            viewport: [0, 0, BENCH_W, BENCH_H],
            shade_model: GL_SMOOTH,
            depth_func: GL_LESS,
            depth_mask: true,
            depth_range: [0.0, 1.0],
            draw_buffers: 1,
            ..Default::default()
        };
        Bench {
            board,
            st,
            tex: None,
            prims: Vec::new(),
            mode: GL_TRIANGLES,
            pixels: 0.0,
            clears: 0,
            polygon_smooth: false,
        }
    }

    pub fn new(scene: Scene, seed: u64, count: usize) -> Bench {
        let mut b = Bench::empty();
        let mut rng = Lcg(seed);
        let st = &mut b.st;
        let mut tris = |b: &mut Vec<Vec<f32>>, pixels: &mut f64, size: f32, texscale: f32| {
            for _ in 0..count {
                let (v, a) = triangle(&mut rng, size, texscale);
                b.push(v);
                *pixels += a;
            }
        };

        match scene {
            Scene::FlatFill => {
                st.shade_model = GL_FLAT;
                tris(&mut b.prims, &mut b.pixels, 40.0, 0.0);
            }
            Scene::SmoothDepth | Scene::SmoothBlend | Scene::MidSmooth => {
                st.depth_test = true;
                if scene == Scene::SmoothBlend {
                    st.blend = true;
                    st.blend_src = GL_SRC_ALPHA;
                    st.blend_dst = GL_ONE_MINUS_SRC_ALPHA;
                }
                tris(&mut b.prims, &mut b.pixels, if scene == Scene::MidSmooth { 110.0 } else { 40.0 }, 0.0);
            }
            Scene::TexBilinearDepth | Scene::TexTrilinearDepth | Scene::TexFogBlend => {
                let min = if scene == Scene::TexBilinearDepth { 1 } else { 5 };
                st.depth_test = true;
                st.texture_2d = true;
                if scene == Scene::TexFogBlend {
                    st.fog = true;
                    st.blend = true;
                    st.blend_src = GL_SRC_ALPHA;
                    st.blend_dst = GL_ONE_MINUS_SRC_ALPHA;
                }
                b.tex = Some(Tex { size: 256, min, mag_linear: true, ..Tex::new(RGBA) });
                // Minified by about 1.5: a trilinear draw mixes two levels.
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::TexPerspective | Scene::TexClampTrilinear => {
                st.depth_test = true;
                st.texture_2d = true;
                if scene == Scene::TexPerspective {
                    b.tex = Some(Tex { size: 256, min: 5, mag_linear: true, ..Tex::new(RGBA) });
                } else {
                    b.tex = Some(Tex { size: 256, min: 5, mag_linear: true, wrap: 1, ..Tex::new(RGB) });
                    st.depth_func = GL_LEQUAL;
                    st.shade_model = GL_FLAT;
                }
                for _ in 0..count {
                    let (mut v, a) = triangle(&mut rng, 40.0, 1.5 / 256.0);

                    // Each vertex its own w, the same window position, and
                    // depth (z + w) / 2 kept within [0, 1].
                    for i in 0..3 {
                        let w = rng.range(1.0, 4.0);
                        let d = rng.range(0.05, 0.95);
                        let o = &mut v[i * vtx::FLOATS + vtx::POSITION..];

                        o[0] *= w;
                        o[1] *= w;
                        o[2] = 2.0 * d - w;
                        o[3] = w;
                    }
                    b.prims.push(v);
                    b.pixels += a;
                }
            }
            Scene::TexStencil | Scene::TexStencilClip => {
                if scene == Scene::TexStencilClip {
                    // x + y / 4 >= -0.25: about three fifths of the screen.
                    st.clip_planes = 1;
                    st.clip_plane[0] = [1.0, 0.25, 0.0, 0.25];
                }
                st.depth_test = true;
                st.texture_2d = true;
                st.stencil_test = true;
                st.stencil_func = GL_EQUAL;
                st.stencil_value_mask = u32::MAX;
                st.stencil_write_mask = u32::MAX;
                st.stencil_op = [GL_KEEP; 3];
                b.tex = Some(Tex { size: 256, min: 1, mag_linear: true, ..Tex::new(RGB) });
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::StencilMark => {
                st.depth_test = true;
                st.depth_mask = false;
                st.stencil_test = true;
                st.stencil_func = GL_ALWAYS;
                st.stencil_value_mask = u32::MAX;
                st.stencil_write_mask = u32::MAX;
                st.stencil_op = [GL_KEEP, GL_KEEP, GL_INCR];
                tris(&mut b.prims, &mut b.pixels, 40.0, 0.0);
            }
            Scene::TexBlendOver => {
                st.depth_test = true;
                st.depth_func = GL_ALWAYS;
                st.depth_mask = false;
                st.texture_2d = true;
                st.blend = true;
                st.blend_src = GL_SRC_ALPHA;
                st.blend_dst = GL_ONE_MINUS_SRC_ALPHA;
                b.tex = Some(Tex { size: 256, min: 5, mag_linear: true, ..Tex::new(RGBA) });
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::PolySmooth => {
                st.depth_test = true;
                b.polygon_smooth = true;
                st.blend = true;
                st.blend_src = GL_SRC_ALPHA;
                st.blend_dst = GL_ONE_MINUS_SRC_ALPHA;
                tris(&mut b.prims, &mut b.pixels, 40.0, 0.0);
            }
            Scene::TexGlass => {
                st.depth_test = true;
                st.texture_2d = true;
                st.blend = true;
                st.blend_src = GL_SRC_ALPHA;
                st.blend_dst = GL_ONE_MINUS_SRC_ALPHA;
                // The bilinear texture has no alpha, so the blend weighs the
                // vertices' own.
                b.tex = Some(Tex { size: 256, min: 1, mag_linear: true, ..Tex::new(RGB) });
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::TexQuads => {
                st.depth_test = true;
                st.texture_2d = true;
                b.mode = GL_QUADS;
                b.tex = Some(Tex { size: 256, min: 1, mag_linear: true, ..Tex::new(RGB) });
                for _ in 0..count {
                    // A triangle's three corners and the fourth that makes
                    // a parallelogram of them.
                    let (mut v, a) = triangle(&mut rng, 20.0, 1.5 / 256.0);
                    let f = vtx::FLOATS;
                    let d: Vec<f32> = (0..f).map(|k| v[k] + v[2 * f + k] - v[f + k]).collect();
                    v.extend_from_slice(&d);
                    b.prims.push(v);
                    b.pixels += 2.0 * a;
                }
            }
            Scene::BigQuads | Scene::BigTexTrilinear => {
                st.depth_test = true;
                b.mode = GL_QUADS;
                if scene == Scene::BigTexTrilinear {
                    st.texture_2d = true;
                    b.tex = Some(Tex { size: 256, min: 5, mag_linear: true, ..Tex::new(RGBA) });
                }
                for i in 0..count {
                    let mut v = vec![0.0f32; 4 * vtx::FLOATS];
                    // Each nearer than the last, so every pixel passes.
                    let z = 0.8 - 1.6 * (i as f32 / count as f32);
                    let corners = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
                    for (k, (x, y)) in corners.iter().enumerate() {
                        let o = &mut v[k * vtx::FLOATS..(k + 1) * vtx::FLOATS];
                        o[vtx::POSITION..vtx::POSITION + 4].copy_from_slice(&[*x, *y, z, 1.0]);
                        o[vtx::COLOR..vtx::COLOR + 4]
                            .copy_from_slice(&[rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.0, 1.0), 1.0]);
                        // The 256-texel image some 2.5 times across: minified
                        // by about 2, two levels mixed.
                        o[vtx::TEXCOORD..vtx::TEXCOORD + 4].copy_from_slice(&[(x + 1.0) * 2.5, (y + 1.0) * 2.0, 0.0, 1.0]);
                    }
                    b.prims.push(v);
                    b.pixels += (BENCH_W * BENCH_H) as f64;
                }
            }
            Scene::Lines => {
                b.mode = GL_LINES;
                st.shade_model = GL_FLAT;
                for _ in 0..count {
                    let mut v = vec![0.0f32; 2 * vtx::FLOATS];
                    let x0 = rng.range(0.0, BENCH_W as f32);
                    let y0 = rng.range(0.0, BENCH_H as f32);
                    let x1 = (x0 + rng.range(-100.0, 100.0)).clamp(0.0, BENCH_W as f32 - 1.0);
                    let y1 = (y0 + rng.range(-100.0, 100.0)).clamp(0.0, BENCH_H as f32 - 1.0);
                    let c = [rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.0, 1.0), 1.0];
                    for (k, (x, y)) in [(x0, y0), (x1, y1)].iter().enumerate() {
                        let o = &mut v[k * vtx::FLOATS..(k + 1) * vtx::FLOATS];
                        o[vtx::POSITION..vtx::POSITION + 4].copy_from_slice(&[ndc(*x, BENCH_W), ndc(*y, BENCH_H), 0.0, 1.0]);
                        o[vtx::COLOR..vtx::COLOR + 4].copy_from_slice(&c);
                    }
                    b.prims.push(v);
                    b.pixels += ((x1 - x0).abs().max((y1 - y0).abs()) + 1.0) as f64;
                }
            }
            Scene::Clear => {
                b.clears = count;
                b.pixels = (BENCH_W * BENCH_H) as f64 * count as f64;
            }
        }
        if let Some(t) = b.tex {
            load_texture(&mut b.board, &mut b.st, &t);
        }
        b
    }

    pub(super) fn clear_op(buffers: u32) -> RasterOp {
        let mut op = RasterOp {
            kind: RASTER_CLEAR,
            ntargets: 1,
            x0: 0,
            y0: 0,
            x1: BENCH_W - 1,
            y1: BENCH_H - 1,
            clear_buffers: buffers,
            clear_rgba: [0.2, 0.3, 0.4, 1.0],
            clear_colour_mask: GL_COLOUR_MASK_ALL,
            clear_depth24: 0xffffff,
            ..Default::default()
        };
        op.clip.n = 1;
        op.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: BENCH_W - 1, y1: BENCH_H - 1 };
        op
    }

    /// Reset the depth plane to its farthest, so that each pass of a
    /// depth-tested scene draws the same fragments; outside the timing.
    pub fn prepare(&mut self) {
        self.board.gl_raster(&Bench::clear_op(GL_DEPTH_BUFFER_BIT));
    }

    /// Draw the scene once.
    pub fn run(&mut self) {
        let stipple = [0u32; 32];

        for _ in 0..self.clears {
            self.board.gl_raster(&Bench::clear_op(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
        }
        let floats = vtx::FLOATS as u32;
        for v in self.prims.iter() {
            let mut d = LibDraw::new(self.mode, v.len() as u32 / floats, v, &self.st, &stipple);

            d.polygon_smooth = self.polygon_smooth;
            self.board.lib_draw(&mut d);
        }
        self.board.raster_wait();
    }

    /// Draw d on the board as the library would (Board::lib_draw()).
    pub fn draw(&mut self, d: &mut LibDraw) {
        self.board.lib_draw(d);
        self.board.raster_wait();
    }

    /// The depth plane's words in the square from (0, 0) to side - 1, row
    /// by row.
    pub fn depth(&self, side: i32) -> Vec<u32> {
        self.snapshot(side).1
    }

    /// Set the depth plane's words in the square from (0, 0) to side - 1
    /// from words, row by row.
    pub fn set_depth(&mut self, side: i32, words: &[u32]) {
        let depth = self.board.dest_mem().depth;

        for y in 0..side {
            for x in 0..side {
                depth.store_word(x, y, words[(y * side + x) as usize]);
            }
        }
    }

    /// The colour, depth and stencil planes in the square from (0, 0) to
    /// side - 1.
    pub fn snapshot(&self, side: i32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let m = self.board.dest_mem();
        let mut c = Vec::new();
        let mut z = Vec::new();
        let mut s = Vec::new();

        for y in 0..side {
            for x in 0..side {
                c.push(m.colour[0].load_word(x, y).unwrap_or(0));
                z.push(m.depth.load_word(x, y).unwrap_or(0));
                s.push(m.stencil.load1(x, y).unwrap_or(0) as u32);
            }
        }
        (c, z, s)
    }
}

/// Primitives per scene: enough that a pass takes some milliseconds.
pub(super) fn bench_count(scene: Scene) -> usize {
    match scene {
        Scene::BigQuads | Scene::BigTexTrilinear | Scene::Clear => 8,
        Scene::MidSmooth => 1000,
        Scene::PolySmooth => 1000,
        Scene::Lines => 20000,
        _ => 4000,
    }
}

#[test]
#[ignore]
fn raster_bench() {
    let only = std::env::var("BENCH_SCENE").ok();
    let secs: f64 = std::env::var("BENCH_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    for scene in ALL_SCENES {
        if let Some(o) = &only {
            if !format!("{scene:?}").eq_ignore_ascii_case(o) {
                continue;
            }
        }
        let mut bench = Bench::new(scene, 1, bench_count(scene));
        let mut best = f64::MAX;
        let mut total = 0.0;
        let mut runs = 0;

        bench.prepare();
        bench.run();
        while total < secs || runs < 3 {
            bench.prepare();
            let t0 = std::time::Instant::now();
            bench.run();
            let dt = t0.elapsed().as_secs_f64();
            best = best.min(dt);
            total += dt;
            runs += 1;
        }
        println!(
            "bench {:<18} {:>8.2} Mpix/s best, {:>8.2} mean, {:>8.2} ms/pass, {} passes",
            format!("{scene:?}"),
            bench.pixels / best / 1e6,
            bench.pixels * runs as f64 / total / 1e6,
            best * 1e3,
            runs
        );
    }
}
