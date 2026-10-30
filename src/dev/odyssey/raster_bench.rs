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
//! one second).

use super::*;

pub const BENCH_W: i32 = SGI_ODYSSEY_WIDTH as i32;
pub const BENCH_H: i32 = SGI_ODYSSEY_HEIGHT as i32;

/// A deterministic generator, so every run and every path draws the same.
pub struct Lcg(pub u64);

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
pub struct HashTexels;

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
pub struct BenchTexture {
    pub levels: Vec<GlTexLevel>,
    pub min: u32,
    pub mag: u32,
    pub format: u32,
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
        BenchTexture { levels, min, mag, format, wrap_t: SGI_GL_REPEAT }
    }

    pub fn image(&self) -> GlTexImage<'_> {
        GlTexImage {
            width: self.levels[0].width,
            height: self.levels[0].height,
            base_format: self.format,
            min_filter: self.min,
            mag_filter: self.mag,
            wrap_s: SGI_GL_REPEAT,
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
pub enum Scene {
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
}

pub const ALL_SCENES: [Scene; 11] = [
    Scene::FlatFill,
    Scene::SmoothDepth,
    Scene::SmoothBlend,
    Scene::TexBilinearDepth,
    Scene::TexTrilinearDepth,
    Scene::TexFogBlend,
    Scene::MidSmooth,
    Scene::BigQuads,
    Scene::BigTexTrilinear,
    Scene::Lines,
    Scene::Clear,
];

/// One scene, built: the board, the draw state, the vertex lists of its
/// primitives and the pixels they stand for.
pub struct Bench {
    pub board: Box<Board>,
    pub st: GlDrawState,
    pub rules: RasterRules,
    pub tex: Option<BenchTexture>,
    /// Each primitive's vertices, SGI_GL_VERTEX_FLOATS apiece.
    pub prims: Vec<Vec<f32>>,
    pub mode: u32,
    /// The pixels the scene covers, by area, for the rate.
    pub pixels: f64,
    pub clears: usize,
    /// Fog by the draw state's mode rather than the board's own curve.
    pub builtin_fog: bool,
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
    let mut v = vec![0.0f32; 3 * SGI_GL_VERTEX_FLOATS];
    let mut xy = [(0.0f32, 0.0f32); 3];
    let s0 = rng.range(0.0, 4.0);
    let t0 = rng.range(0.0, 4.0);
    for i in 0..3 {
        let x = cx + rng.range(-size, size);
        let y = cy + rng.range(-size, size);
        let z = rng.range(-0.9, 0.9);
        let o = &mut v[i * SGI_GL_VERTEX_FLOATS..(i + 1) * SGI_GL_VERTEX_FLOATS];
        xy[i] = (x.clamp(0.0, BENCH_W as f32), y.clamp(0.0, BENCH_H as f32));
        o[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4].copy_from_slice(&[ndc(x, BENCH_W), ndc(y, BENCH_H), z, 1.0]);
        o[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4]
            .copy_from_slice(&[rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.2, 1.0)]);
        o[SGI_GL_V_TEXCOORD..SGI_GL_V_TEXCOORD + 4]
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
        board.clip_kept = 1;
        board.clip_rect[0] = ClipRect { x0: 0, y0: 0, x1: BENCH_W - 1, y1: BENCH_H - 1 };
        board.raster_bands = false;
        let mut rules = gl_rules();
        rules.alpha_bits = 8;
        rules.depth_row_given = true;
        // Depth (z + w) / 2 over w: the identity projection's [-1, 1] mapped
        // to [0, 1], as Board::gl_operation_rules() derives it from a
        // combined matrix.
        rules.depth_row = [0.0, 0.0, 0.5, 0.5];
        rules.depth_slack = ODSY_GL_DEPTH_SLACK;

        let mut st = GlDrawState {
            modelview: identity(),
            projection: identity(),
            viewport: [0, 0, BENCH_W, BENCH_H],
            shade_model: SGI_GL_SMOOTH,
            depth_func: SGI_GL_LESS,
            depth_mask: 1,
            depth_range: [0.0, 1.0],
            draw_buffers: 1,
            colour_mask: SGI_GL_COLOUR_MASK_ALL,
            texture_env_mode: SGI_GL_MODULATE,
            dither: 1,
            ..Default::default()
        };
        st.drawable.bound = 1;
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
            mode: SGI_GL_TRIANGLES,
            pixels: 0.0,
            clears: 0,
            builtin_fog: false,
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
                st.shade_model = SGI_GL_FLAT;
                st.dither = 0;
                tris(&mut b.prims, &mut b.pixels, 40.0, 0.0);
            }
            Scene::SmoothDepth | Scene::SmoothBlend | Scene::MidSmooth => {
                st.depth_test = 1;
                if scene == Scene::SmoothBlend {
                    st.blend = 1;
                    st.blend_src = SGI_GL_SRC_ALPHA;
                    st.blend_dst = SGI_GL_ONE_MINUS_SRC_ALPHA;
                }
                tris(&mut b.prims, &mut b.pixels, if scene == Scene::MidSmooth { 110.0 } else { 40.0 }, 0.0);
            }
            Scene::TexBilinearDepth | Scene::TexTrilinearDepth | Scene::TexFogBlend => {
                let (min, mag) = if scene == Scene::TexBilinearDepth {
                    (SGI_GL_LINEAR, SGI_GL_LINEAR)
                } else {
                    (SGI_GL_LINEAR_MIPMAP_LINEAR, SGI_GL_LINEAR)
                };
                st.depth_test = 1;
                st.texture_2d = 1;
                if scene == Scene::TexFogBlend {
                    st.fog = 1;
                    st.fog_mode = SGI_GL_LINEAR;
                    st.fog_start = 0.0;
                    st.fog_end = 2.0;
                    st.fog_color = [0.5, 0.5, 0.6, 1.0];
                    st.blend = 1;
                    st.blend_src = SGI_GL_SRC_ALPHA;
                    st.blend_dst = SGI_GL_ONE_MINUS_SRC_ALPHA;
                }
                b.tex = Some(BenchTexture::new(256, min, mag, SGI_GL_RGBA));
                // Minified by about 1.5: a trilinear draw mixes two levels.
                tris(&mut b.prims, &mut b.pixels, 40.0, 1.5 / 256.0);
            }
            Scene::BigQuads | Scene::BigTexTrilinear => {
                st.depth_test = 1;
                b.mode = SGI_GL_QUADS;
                if scene == Scene::BigTexTrilinear {
                    st.texture_2d = 1;
                    b.tex = Some(BenchTexture::new(256, SGI_GL_LINEAR_MIPMAP_LINEAR, SGI_GL_LINEAR, SGI_GL_RGBA));
                }
                for i in 0..count {
                    let mut v = vec![0.0f32; 4 * SGI_GL_VERTEX_FLOATS];
                    // Each nearer than the last, so every pixel passes.
                    let z = 0.8 - 1.6 * (i as f32 / count as f32);
                    let corners = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
                    for (k, (x, y)) in corners.iter().enumerate() {
                        let o = &mut v[k * SGI_GL_VERTEX_FLOATS..(k + 1) * SGI_GL_VERTEX_FLOATS];
                        o[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4].copy_from_slice(&[*x, *y, z, 1.0]);
                        o[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4]
                            .copy_from_slice(&[rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.0, 1.0), 1.0]);
                        // The 256-texel image some 2.5 times across: minified
                        // by about 2, two levels mixed.
                        o[SGI_GL_V_TEXCOORD..SGI_GL_V_TEXCOORD + 4].copy_from_slice(&[(x + 1.0) * 2.5, (y + 1.0) * 2.0, 0.0, 1.0]);
                    }
                    b.prims.push(v);
                    b.pixels += (BENCH_W * BENCH_H) as f64;
                }
            }
            Scene::Lines => {
                b.mode = SGI_GL_LINES;
                st.shade_model = SGI_GL_FLAT;
                st.dither = 0;
                for _ in 0..count {
                    let mut v = vec![0.0f32; 2 * SGI_GL_VERTEX_FLOATS];
                    let x0 = rng.range(0.0, BENCH_W as f32);
                    let y0 = rng.range(0.0, BENCH_H as f32);
                    let x1 = (x0 + rng.range(-100.0, 100.0)).clamp(0.0, BENCH_W as f32 - 1.0);
                    let y1 = (y0 + rng.range(-100.0, 100.0)).clamp(0.0, BENCH_H as f32 - 1.0);
                    let c = [rng.range(0.0, 1.0), rng.range(0.0, 1.0), rng.range(0.0, 1.0), 1.0];
                    for (k, (x, y)) in [(x0, y0), (x1, y1)].iter().enumerate() {
                        let o = &mut v[k * SGI_GL_VERTEX_FLOATS..(k + 1) * SGI_GL_VERTEX_FLOATS];
                        o[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4].copy_from_slice(&[ndc(*x, BENCH_W), ndc(*y, BENCH_H), 0.0, 1.0]);
                        o[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4].copy_from_slice(&c);
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
            kind: SGI_RASTER_CLEAR,
            ntargets: 1,
            x0: 0,
            y0: 0,
            x1: BENCH_W - 1,
            y1: BENCH_H - 1,
            clear_buffers: buffers,
            clear_rgba: [0.2, 0.3, 0.4, 1.0],
            clear_colour_mask: SGI_GL_COLOUR_MASK_ALL,
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

        self.rules.raster(&t, &mut self.board.gl_parts(None).0, &Bench::clear_op(SGI_GL_DEPTH_BUFFER_BIT));
    }

    /// Draw the scene once.
    pub fn run(&mut self) {
        let mut t = self.board.gl_target();
        let stipple = [0u32; 32];

        t.texture = self.tex.as_ref().map(|tex| tex.image());
        if self.builtin_fog {
            t.fog_factor = false;
        }
        for _ in 0..self.clears {
            let op = Bench::clear_op(SGI_GL_COLOR_BUFFER_BIT | SGI_GL_DEPTH_BUFFER_BIT);

            self.rules.raster(&t, &mut self.board.gl_parts(None).0, &op);
        }
        let floats = SGI_GL_VERTEX_FLOATS as u32;
        for v in self.prims.iter() {
            let mut d = GlDraw::new(self.mode, v.len() as u32 / floats, v, &self.st, &stipple);

            self.rules.draw(&t, &mut self.board.gl_parts(None).0, &HashTexels, &mut d);
        }
    }

    /// The colour and depth planes in the square from (0, 0) to side - 1.
    pub fn snapshot(&self, side: i32) -> (Vec<u32>, Vec<u32>) {
        let b = &self.board;
        let depth = b.depth_mem.plane(b.depth_plane).expect("the depth plane");
        let mut c = Vec::new();
        let mut z = Vec::new();

        for y in 0..side {
            for x in 0..side {
                c.push(b.buffer_load(0, x as i64, y as i64));
                z.push(b.depth_mem.plane_load(&depth, x, y));
            }
        }
        (c, z)
    }
}

/// Primitives per scene: enough that a pass takes some milliseconds.
pub fn bench_count(scene: Scene) -> usize {
    match scene {
        Scene::BigQuads | Scene::BigTexTrilinear | Scene::Clear => 8,
        Scene::MidSmooth => 1000,
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
