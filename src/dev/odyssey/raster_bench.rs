//! A fill-rate benchmark for the rasteriser, and the scenes it draws.
//!
//! Each scene drives RasterRules::draw() and RasterRules::raster() the way
//! Board::gl_draw() does: one primitive per draw, into a v12 board's own
//! planes (Board::gl_parts()), with the board's raster rules (depth row,
//! float depth, RGBA stores). No guest is involved, so the numbers are the
//! rasteriser's and the planes', not the command decoder's or the geometry
//! program's.
//!
//! Run it from a release build, one test thread:
//!
//!     cargo test --release --lib odyssey::raster_bench -- --ignored --nocapture --test-threads=1
//!
//! BENCH_SCENE picks one scene, BENCH_SECS the time each takes (default
//! one second), BENCH_FILL the row loops (table, dynamic or off; see
//! raster_set_fill_path()). The tests in raster_fill.rs draw with the same
//! board and state.


use super::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use super::consts::gl_cmd;
use super::draw::gl_rules;
use super::gl_context::{
    vtx, GL_ALWAYS, GL_CLAMP_TO_EDGE, GL_COLOR_BUFFER_BIT, GL_COLOUR_MASK_ALL, GL_DEPTH_BUFFER_BIT,
    GL_EQUAL, GL_FLAT, GL_INCR, GL_KEEP, GL_LEQUAL, GL_LESS, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR,
    GL_LINES, GL_MODULATE, GL_ONE_MINUS_SRC_ALPHA, GL_QUADS, GL_REPEAT, GL_RGB, GL_RGBA, GL_SMOOTH,
    GL_SRC_ALPHA, GL_TRIANGLES, GlDraw, GlDrawResult, GlDrawState, RASTER_CLEAR, RasterOp,
};
use super::memory::ClipRect;
use super::raster::{GlTexImage, GlTexLevel, GlTexPyramid, RasterRules, RasterTarget, Texels};
use super::raster_fill::{raster_set_fill_path, FillPath};
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
pub(super) struct HashTexels;

impl Texels for HashTexels {
    fn texel(&self, level: Option<&GlTexLevel>, x: u32, y: u32) -> Option<u32> {
        let l = level.map_or(0, |l| l.level);
        let h = x.wrapping_mul(0x9e3779b1) ^ y.wrapping_mul(0x85ebca77) ^ l.wrapping_mul(0xc2b2ae35) ^ 0x5bd1e995;
        Some(h ^ (h >> 13))
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

/// A square mipmapped texture of side size: its levels and sampler.
pub(super) struct BenchTexture {
    pub levels: Vec<GlTexLevel>,
    pub min: u32,
    pub mag: u32,
    pub format: u32,
    pub wrap_s: u32,
    pub wrap_t: u32,
}

impl BenchTexture {
    pub fn new(size: u32, min: u32, mag: u32, format: u32) -> BenchTexture {
        let mut levels = Vec::new();
        let mut s = size;
        let mut level = 0;
        loop {
            levels.push(GlTexLevel {
                level,
                width: s,
                height: s,
                depth: 1,
                base_format: format,
                internal_format: format,
                read_texel: true,
                ..Default::default()
            });
            if s == 1 {
                break;
            }
            s /= 2;
            level += 1;
        }
        BenchTexture { levels, min, mag, format, wrap_s: GL_REPEAT, wrap_t: GL_REPEAT }
    }

    pub fn image(&self) -> GlTexImage<'_> {
        GlTexImage {
            width: self.levels[0].width,
            height: self.levels[0].height,
            base_format: self.format,
            min_filter: self.min,
            mag_filter: self.mag,
            wrap_s: self.wrap_s,
            wrap_t: self.wrap_t,
            read_texel: true,
            pyramid: Some(GlTexPyramid {
                levels: &self.levels,
                base_level: 0,
                max_level: 1000,
                min_lod: -1000.0,
                max_lod: 1000.0,
            }),
            ..Default::default()
        }
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
    pub st: GlDrawState,
    pub rules: RasterRules,
    pub tex: Option<BenchTexture>,
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
fn triangle(rng: &mut Lcg, size: f32, texscale: f32) -> (Vec<f32>, f64) {
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

        // Every pixel of the screen is the window's.
        board.win.clip_kept = 1;
        board.win.clip_rect[0] = ClipRect { x0: 0, y0: 0, x1: BENCH_W - 1, y1: BENCH_H - 1 };
        let mut rules = gl_rules();
        rules.alpha_bits = 8;
        rules.depth_row_given = true;
        // Depth (z + w) / 2 over w: the identity projection's [-1, 1] mapped
        // to [0, 1], as Board::gl_operation_rules() derives it from a
        // combined matrix.
        rules.depth_row = [0.0, 0.0, 0.5, 0.5];
        rules.depth_slack = gl_cmd::DEPTH_SLACK;

        let mut st = GlDrawState {
            modelview: identity(),
            projection: identity(),
            viewport: [0, 0, BENCH_W, BENCH_H],
            shade_model: GL_SMOOTH,
            depth_func: GL_LESS,
            depth_mask: true,
            depth_range: [0.0, 1.0],
            draw_buffers: 1,
            colour_mask: GL_COLOUR_MASK_ALL,
            texture_env_mode: GL_MODULATE,
            ..Default::default()
        };
        st.drawable.bound = true;
        st.drawable.front = 0;
        st.drawable.back = 0;
        st.drawable.clip.n = 1;
        st.drawable.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: BENCH_W - 1, y1: BENCH_H - 1 };
        Bench {
            board,
            st,
            rules,
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
                let (min, mag) = if scene == Scene::TexBilinearDepth {
                    (GL_LINEAR, GL_LINEAR)
                } else {
                    (GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR)
                };
                st.depth_test = true;
                st.texture_2d = true;
                if scene == Scene::TexFogBlend {
                    st.fog = true;
                    st.fog_mode = GL_LINEAR;
                    st.fog_start = 0.0;
                    st.fog_end = 2.0;
                    st.fog_color = [0.5, 0.5, 0.6, 1.0];
                    st.blend = true;
                    st.blend_src = GL_SRC_ALPHA;
                    st.blend_dst = GL_ONE_MINUS_SRC_ALPHA;
                }
                b.tex = Some(BenchTexture::new(256, min, mag, GL_RGBA));
                // Minified by about 1.5: a trilinear draw mixes two levels.
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::TexPerspective | Scene::TexClampTrilinear => {
                st.depth_test = true;
                st.texture_2d = true;
                if scene == Scene::TexPerspective {
                    b.tex = Some(BenchTexture::new(256, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_RGBA));
                } else {
                    let mut tex = BenchTexture::new(256, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_RGB);

                    tex.wrap_s = GL_CLAMP_TO_EDGE;
                    b.tex = Some(tex);
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
                b.tex = Some(BenchTexture::new(256, GL_LINEAR, GL_LINEAR, GL_RGB));
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::StencilMark => {
                st.depth_test = true;
                st.depth_mask = false;
                st.colour_mask = 0;
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
                b.tex = Some(BenchTexture::new(256, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_RGBA));
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
                b.tex = Some(BenchTexture::new(256, GL_LINEAR, GL_LINEAR, GL_RGB));
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::TexQuads => {
                st.depth_test = true;
                st.texture_2d = true;
                b.mode = GL_QUADS;
                b.tex = Some(BenchTexture::new(256, GL_LINEAR, GL_LINEAR, GL_RGB));
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
                    b.tex = Some(BenchTexture::new(256, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR, GL_RGBA));
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
        b
    }

    fn clear_op(buffers: u32) -> RasterOp {
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
        let t = self.board.gl_target();

        self.rules.raster(&t, &mut self.board.gl_parts(None).0, &Bench::clear_op(GL_DEPTH_BUFFER_BIT));
    }

    /// Draw the scene once.
    pub fn run(&mut self) {
        let mut t = self.board.gl_target();
        let stipple = [0u32; 32];

        t.texture = self.tex.as_ref().map(|tex| tex.image());
        for _ in 0..self.clears {
            let op = Bench::clear_op(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            self.rules.raster(&t, &mut self.board.gl_parts(None).0, &op);
        }
        let floats = vtx::FLOATS as u32;
        for v in self.prims.iter() {
            let mut d = GlDraw::new(self.mode, v.len() as u32 / floats, v, &self.st, &stipple);

            d.polygon_smooth = self.polygon_smooth;
            self.rules.draw(&t, &mut self.board.gl_parts(None).0, &HashTexels, &mut d);
        }
    }

    /// Draw d on the board under the bench's rules, as Board::gl_draw()
    /// would, reading texels from texels.
    pub fn draw(&mut self, t: &RasterTarget, texels: &dyn Texels, d: &mut GlDraw) -> GlDrawResult {
        self.rules.draw(t, &mut self.board.gl_parts(None).0, texels, d)
    }

    /// The depth plane's words in the square from (0, 0) to side - 1, row
    /// by row.
    pub fn depth(&self, side: i32) -> Vec<u32> {
        self.snapshot(side).1
    }

    /// Set the depth plane's words in the square from (0, 0) to side - 1
    /// from words, row by row.
    pub fn set_depth(&mut self, side: i32, words: &[u32]) {
        let b = &mut self.board;
        let depth = b.planes.depth.plane(b.planes.depth_plane).expect("the depth plane");

        for y in 0..side {
            for x in 0..side {
                b.planes.depth.plane_store(&depth, x, y, words[(y * side + x) as usize]);
            }
        }
    }

    /// The colour, depth and stencil planes in the square from (0, 0) to
    /// side - 1.
    pub fn snapshot(&self, side: i32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        let b = &self.board;
        let depth = b.planes.depth.plane(b.planes.depth_plane).expect("the depth plane");
        let stencil = b.planes.stencil.plane(b.planes.stencil_plane).expect("the stencil plane");
        let mut c = Vec::new();
        let mut z = Vec::new();
        let mut s = Vec::new();

        for y in 0..side {
            for x in 0..side {
                c.push(b.buffer_load(0, x as i64, y as i64));
                z.push(b.planes.depth.plane_load(&depth, x, y));
                s.push(b.planes.stencil.plane_load(&stencil, x, y));
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
    let fill = match std::env::var("BENCH_FILL").as_deref() {
        Ok("dynamic") => FillPath::Dynamic,
        Ok("off") => FillPath::Off,
        _ => FillPath::Table,
    };

    raster_set_fill_path(fill);
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
    raster_set_fill_path(FillPath::Table);
}
