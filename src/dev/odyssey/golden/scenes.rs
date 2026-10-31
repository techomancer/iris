//! The scenes, and the context that draws them.

use crate::dev::odyssey::board::{Board, ODSY_HEIGHT, ODSY_WIDTH};
use crate::dev::odyssey::consts::{colour, gl_cmd, light, tex, write};
use crate::dev::odyssey::draw::gl_rules;
use crate::dev::odyssey::gl_context::{
    vtx, GL_ALWAYS, GL_BACK, GL_BLEND, GL_CCW, GL_CLEAR, GL_COLOR_BUFFER_BIT, GL_COLOUR_MASK_ALL,
    GL_COLOUR_MASK_ALPHA, GL_COLOUR_MASK_RED, GL_CONSTANT_ALPHA, GL_CONSTANT_COLOR, GL_CW,
    GL_DECAL, GL_DECR, GL_DEPTH_BUFFER_BIT, GL_DRAW_OK, GL_DST_ALPHA, GL_DST_COLOR, GL_EQUAL,
    GL_FILL, GL_FLAT, GL_FRONT, GL_FUNC_ADD, GL_FUNC_REVERSE_SUBTRACT, GL_FUNC_SUBTRACT, GL_GEQUAL,
    GL_GREATER, GL_INCR, GL_INVERT, GL_KEEP, GL_LEQUAL, GL_LESS, GL_LINE, GL_LINES, GL_MAX, GL_MIN,
    GL_MODULATE, GL_NOTEQUAL, GL_OFFSET_FILL, GL_ONE_MINUS_CONSTANT_ALPHA,
    GL_ONE_MINUS_CONSTANT_COLOR, GL_ONE_MINUS_DST_ALPHA, GL_ONE_MINUS_DST_COLOR,
    GL_ONE_MINUS_SRC_ALPHA, GL_ONE_MINUS_SRC_COLOR, GL_POINT, GL_POINTS, GL_POLYGON, GL_QUADS,
    GL_REPLACE, GL_SMOOTH, GL_SRC_ALPHA, GL_SRC_ALPHA_SATURATE, GL_SRC_COLOR,
    GL_STENCIL_BUFFER_BIT, GL_TRIANGLES, GlDraw, GlDrawState, RASTER_CLEAR, RasterOp,
};
use crate::dev::odyssey::golden::{Frame, SIDE};
use crate::dev::odyssey::memory::ClipRect;
use crate::dev::odyssey::raster_bench::Bench;
use super::raster_bench::Lcg;

const W: f32 = SIDE as f32;

pub const RED: [f32; 4] = [1.0, 0.1, 0.1, 1.0];
pub const GREEN: [f32; 4] = [0.1, 0.9, 0.2, 1.0];
pub const BLUE: [f32; 4] = [0.1, 0.2, 1.0, 1.0];
pub const YELLOW: [f32; 4] = [1.0, 0.9, 0.1, 1.0];
pub const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// What every scene starts from.
const CLEAR: [f32; 4] = [0.15, 0.2, 0.3, 1.0];

/// A vertex: window position as a normalized device position, colour,
/// texture coordinates and normal.
#[derive(Clone, Copy)]
pub struct V {
    pub p: [f32; 4],
    pub c: [f32; 4],
    pub t: [f32; 4],
    pub n: [f32; 3],
    pub back: Option<[f32; 4]>,
}

fn nd(px: f32) -> f32 {
    px / W * 2.0 - 1.0
}

/// A vertex at window pixel (x, y) and depth z in [-1, 1].
pub fn v(x: f32, y: f32, z: f32, c: [f32; 4]) -> V {
    V { p: [nd(x), nd(y), z, 1.0], c, t: [0.0, 0.0, 0.0, 1.0], n: [0.0, 0.0, 1.0], back: None }
}

impl V {
    pub fn tex(mut self, s: f32, t: f32) -> V {
        self.t = [s, t, 0.0, 1.0];
        self
    }
    /// The vertex as a perspective division by w would leave it: the same
    /// window position, and window depth d.
    pub fn persp(mut self, w: f32, d: f32) -> V {
        self.p = [self.p[0] * w, self.p[1] * w, 2.0 * d - w, w];
        self.t = [self.t[0] * w, self.t[1] * w, 0.0, w];
        self
    }
    fn floats(&self) -> [f32; vtx::FLOATS] {
        let mut o = [0.0f32; vtx::FLOATS];
        o[vtx::POSITION..][..4].copy_from_slice(&self.p);
        o[vtx::COLOR..][..4].copy_from_slice(&self.c);
        o[vtx::NORMAL..][..3].copy_from_slice(&self.n);
        o[vtx::TEXCOORD..][..4].copy_from_slice(&self.t);
        o[vtx::EDGE] = f32::from_bits(1);
        o[vtx::BACK_COLOR..][..4].copy_from_slice(&self.back.unwrap_or(self.c));
        o
    }
}

/// A board set up to draw into the scene window, the draw state, and the
/// per-primitive flags the GL context would add to a draw.
pub struct Ctx {
    pub board: Box<Board>,
    pub st: GlDrawState,
    pub stipple: [u32; 32],
    pub polygon_stipple_enabled: bool,
    pub polygon_modes: Option<[u32; 2]>,
    pub point_size: f32,
    pub line_width: f32,
    pub point_smooth: bool,
    pub line_smooth: bool,
    pub polygon_smooth: bool,
    pub line_stipple: Option<(u32, u32)>,
    pub back_colors: bool,
    phase: Option<u32>,
    /// A feature to leave out of the scene ("fog", "blend", "depth", "texture"),
    /// so a test can tell the feature changes the frame.
    pub off: Option<&'static str>,
}

impl Ctx {
    /// A v12 board with the scene window cleared: colour, depth to the far
    /// plane, stencil to 0.
    pub fn new() -> Ctx {
        let b = Bench::empty();
        let (mut board, mut st) = (b.board, b.st);
        let rect = ClipRect { x0: 0, y0: 0, x1: SIDE - 1, y1: SIDE - 1 };

        board.win.clip_rect[0] = rect;
        board.gl_write_mask |= write::COLOUR_MASK << write::COLOUR_SHIFT;
        st.drawable.clip.r[0] = rect;
        st.viewport = [0, 0, SIDE, SIDE];
        st.shade_model = GL_SMOOTH;
        let mut c = Ctx {
            board,
            st,
            stipple: [0; 32],
            polygon_stipple_enabled: false,
            polygon_modes: None,
            point_size: 1.0,
            line_width: 1.0,
            point_smooth: false,
            line_smooth: false,
            polygon_smooth: false,
            line_stipple: None,
            back_colors: false,
            phase: Some(0),
            off: None,
        };
        c.depth_row([0.0, 0.0, 0.5, 0.5]);
        c.clear(CLEAR, 0xff_ffff, 0);
        c
    }

    /// The window as `new` leaves it, for telling what a scene drew.
    pub fn blank(&self) -> Frame {
        Frame::of(&Ctx::new().board)
    }

    /// The depth the board derives from its combined matrix: row . eye over
    /// clip w, scaled the way Board::gl_operation_rules() unscales it. The
    /// default row is the identity projection's, (z + w) / 2.
    pub fn depth_row(&mut self, row: [f64; 4]) {
        let scale = 2.0 * gl_cmd::DEPTH_S * self.board.gl_depth_scale();
        for k in 0..4 {
            self.board.gl_combined[8 + k] = ((row[k] * scale) as f32).to_bits();
        }
    }

    pub fn clear(&mut self, rgba: [f32; 4], depth: u32, stencil: u32) {
        let t = self.board.gl_target();
        let mut op = RasterOp {
            kind: RASTER_CLEAR,
            ntargets: 1,
            x1: BENCH_W_MAX,
            y1: BENCH_H_MAX,
            clear_buffers: GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT,
            clear_rgba: rgba,
            clear_colour_mask: GL_COLOUR_MASK_ALL,
            clear_depth24: depth,
            clear_stencil: stencil,
            stencil_mask: 0xff,
            ..Default::default()
        };
        op.clip.n = 1;
        op.clip.r[0] = ClipRect { x0: 0, y0: 0, x1: BENCH_W_MAX, y1: BENCH_H_MAX };
        gl_rules().raster(&t, &mut self.board.gl_parts(None).0, &op);
    }

    /// Draw one primitive through the board's own GL draw.
    pub fn draw(&mut self, mode: u32, vs: &[V]) {
        let floats: Vec<f32> = vs.iter().flat_map(|v| v.floats()).collect();
        let mut d = GlDraw::new(mode, vs.len() as u32, &floats, &self.st, &self.stipple);

        d.back_colors = self.back_colors;
        d.point_size = self.point_size;
        d.line_width = self.line_width;
        d.point_smooth = self.point_smooth;
        d.line_smooth = self.line_smooth;
        d.polygon_smooth = self.polygon_smooth;
        d.polygon_stipple_enabled = self.polygon_stipple_enabled;
        d.polygon_modes = self.polygon_modes;
        if let Some((repeat, pattern)) = self.line_stipple {
            d.line_stipple_enabled = true;
            d.line_stipple_repeat = repeat;
            d.line_stipple_pattern = pattern;
            d.line_phase = self.phase;
        }
        assert_eq!(self.board.gl_draw(&mut d), GL_DRAW_OK);
        if self.line_stipple.is_some() {
            self.phase = d.line_phase;
        }
    }

    pub fn tri(&mut self, a: V, b: V, c: V) {
        self.draw(GL_TRIANGLES, &[a, b, c]);
    }

    /// A rectangle of two triangles' worth, as one GL_QUADS primitive.
    pub fn rect(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, z: f32, c: [f32; 4]) {
        self.draw(GL_QUADS, &[v(x0, y0, z, c), v(x1, y0, z, c), v(x1, y1, z, c), v(x0, y1, z, c)]);
    }

    pub fn depth_test(&mut self, func: u32) {
        if self.off == Some("depth") {
            return;
        }
        self.st.depth_test = true;
        self.st.depth_func = func;
    }

    pub fn blend(&mut self, src: u32, dst: u32) {
        if self.off == Some("blend") {
            return;
        }
        self.st.blend = true;
        self.st.blend_src = src;
        self.st.blend_dst = dst;
    }

    /// The board's fog curve (word 2 the scale, 3 the offset and shift, 4 and
    /// 5 the curve's four control values), a ramp over the window depth, and
    /// the fog colour.
    pub fn fog(&mut self, colour: [u32; 3], steep: u32) {
        if self.off == Some("fog") {
            return;
        }
        let c12 = |f: f32| (f * 4095.0) as u32;
        self.st.fog = true;
        self.board.gl_fog = [
            c12(colour[0] as f32 / 255.0) << 12 | c12(colour[1] as f32 / 255.0),
            c12(colour[2] as f32 / 255.0) << 12 | 0xfff,
            0x7c0000,
            steep << 12,
            85,
            171 << 12 | 256,
        ];
    }
}

const BENCH_W_MAX: i32 = ODSY_WIDTH as i32 - 1;
const BENCH_H_MAX: i32 = ODSY_HEIGHT as i32 - 1;

// Textures. Each level lives in an object of the board's memory, found from
// the sampler words the way the library's uploads leave them.

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
fn texel(level: u32, x: u32, y: u32, w: u32, h: u32) -> u32 {
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
fn texel_format_word(components: u32) -> u32 {
    match components {
        ALPHA => 0x0000_0000,
        LUMINANCE => 0x0040_0000,
        LUMINANCE_ALPHA => 0x00b0_0000,
        INTENSITY => 0x0020_0000,
        _ => 0x4110_0000,
    }
}

impl Ctx {
    pub fn texture(&mut self, t: &Tex) {
        if self.off == Some("texture") {
            return;
        }
        let b = &mut self.board;
        let levels = if t.min >= 2 { t.size.trailing_zeros() + 1 } else { 1 };
        let (ws, wt) = (t.wrap, t.wrap);

        for level in 0..levels {
            let w = (t.size >> level).max(1);
            let units = (w.div_ceil(16) + 1) & !1;
            let base = 0x10_0000 + level * 0x1000;
            let obj = b.object_select(base, units * 16);
            assert!(b.object_rows(obj, w));
            for y in 0..w {
                for x in 0..w {
                    b.object_store(obj, x, y, texel(level, x, y, w, w));
                }
            }
            b.tex.levels[level as usize * 3..][..3].copy_from_slice(&[base, texel_format_word(t.components) | units, 0]);
        }
        let axis = |w: u32, wrap: u32| w | if wrap == 1 { 0x4_0000 } else { 0 };
        let filter = (t.min << 5) | if t.mag_linear { 2 } else { 0 };
        b.tex.mode = tex::MODE_2D;
        b.tex.level0 = 0;
        b.tex.sampler = [
            0x0800_0000 | (t.size.trailing_zeros() << 10) | (t.size.trailing_zeros() << 5),
            axis(t.size, ws),
            axis(t.size, wt),
            1,
            0x1000_0000 | t.components | (ws << 12) | (wt << 10) | filter,
            0,
            0,
        ];
        b.tex.env = [t.env, 0, 0, 0, 0];
        if t.env == 4 {
            // The environment colour: two normalized 12-bit components a word.
            b.tex.env[1] = 0xc00 << 12 | 0x400;
            b.tex.env[2] = 0x200 << 12 | 0xfff;
        }
        self.st.texture_2d = true;
        self.st.texture_env_mode = [GL_REPLACE, GL_REPLACE, GL_MODULATE, GL_DECAL, GL_BLEND][t.env as usize];
    }
}

// Scenes. Each draws into a fresh `Ctx`.

fn tri_flat(c: &mut Ctx) {
    c.st.shade_model = GL_FLAT;
    c.tri(v(10.0, 10.0, 0.0, RED), v(70.0, 20.0, 0.0, GREEN), v(30.0, 80.0, 0.0, BLUE));
    c.tri(v(60.0, 60.0, 0.0, YELLOW), v(120.0, 70.0, 0.0, RED), v(90.0, 120.0, 0.0, GREEN));
    c.draw(GL_QUADS, &[v(5.0, 90.0, 0.0, BLUE), v(40.0, 95.0, 0.0, GREEN), v(45.0, 125.0, 0.0, RED), v(8.0, 120.0, 0.0, YELLOW)]);
}

fn tri_smooth(c: &mut Ctx) {
    c.tri(v(10.0, 10.0, 0.0, RED), v(110.0, 25.0, 0.0, GREEN), v(40.0, 115.0, 0.0, BLUE));
    c.tri(v(70.0, 70.0, 0.0, [1.0, 1.0, 1.0, 0.25]), v(125.0, 80.0, 0.0, [0.0, 0.0, 0.0, 1.0]), v(95.0, 125.0, 0.0, YELLOW));
}

fn tri_tiny_and_thin(c: &mut Ctx) {
    // Sub-pixel and sliver triangles: coverage, tie rule and seam pixels.
    let mut rng = Lcg(7);
    for _ in 0..60 {
        let (x, y) = (rng.range(4.0, 120.0), rng.range(4.0, 120.0));
        let s = rng.range(0.3, 3.0);
        c.tri(v(x, y, 0.0, RED), v(x + s, y + 0.2, 0.0, GREEN), v(x + 0.3, y + s, 0.0, BLUE));
    }
    for i in 0..8 {
        let y = 10.0 + 14.0 * i as f32;
        c.tri(v(5.0, y, 0.0, WHITE), v(120.0, y + 0.4 + 0.1 * i as f32, 0.0, YELLOW), v(5.0, y + 0.6, 0.0, WHITE));
    }
}

fn quad_shared_edges(c: &mut Ctx) {
    // A grid of quads that share edges: no pixel drawn twice, none missed.
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    let n = 6;
    let step = 100.0 / n as f32;
    for i in 0..n {
        for j in 0..n {
            let (x, y) = (12.3 + step * i as f32, 9.7 + step * j as f32);
            let a = [0.5, 0.5, 0.5, 0.5];
            c.draw(GL_QUADS, &[v(x, y, 0.0, a), v(x + step, y, 0.0, a), v(x + step, y + step, 0.0, a), v(x, y + step, 0.0, a)]);
        }
    }
}

fn polygon_convex(c: &mut Ctx) {
    let pts = [(64.0, 8.0), (118.0, 40.0), (100.0, 110.0), (30.0, 118.0), (8.0, 50.0)];
    let cols = [RED, GREEN, BLUE, YELLOW, WHITE];
    let vs: Vec<V> = pts.iter().zip(cols).map(|(&(x, y), col)| v(x, y, 0.0, col)).collect();
    c.draw(GL_POLYGON, &vs);
}

fn polygon_modes_line_point(c: &mut Ctx) {
    c.polygon_modes = Some([GL_LINE, GL_POINT]);
    c.point_size = 3.0;
    c.tri(v(10.0, 10.0, 0.0, RED), v(60.0, 15.0, 0.0, GREEN), v(30.0, 60.0, 0.0, BLUE));
    c.polygon_modes = Some([GL_FILL, GL_LINE]);
    c.tri(v(70.0, 70.0, 0.0, RED), v(120.0, 120.0, 0.0, GREEN), v(70.0, 120.0, 0.0, BLUE));
    c.polygon_modes = Some([GL_POINT, GL_FILL]);
    c.tri(v(70.0, 10.0, 0.0, YELLOW), v(120.0, 20.0, 0.0, YELLOW), v(90.0, 60.0, 0.0, YELLOW));
}

fn polygon_stipple(c: &mut Ctx) {
    for (i, row) in c.stipple.iter_mut().enumerate() {
        *row = if i & 1 == 0 { 0xaaaa_aaaa } else { 0x5555_5555 };
    }
    c.stipple[8] = 0xffff_0000;
    c.polygon_stipple_enabled = true;
    c.tri(v(5.0, 5.0, 0.0, RED), v(123.0, 20.0, 0.0, GREEN), v(40.0, 120.0, 0.0, BLUE));
}

fn polygon_smooth(c: &mut Ctx) {
    c.polygon_smooth = true;
    c.depth_test(GL_LESS);
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    c.tri(v(10.0, 15.0, 0.0, RED), v(115.0, 35.0, 0.0, GREEN), v(50.0, 118.0, 0.0, BLUE));
}

fn cull_and_front_face(c: &mut Ctx) {
    c.st.cull_face = true;
    c.st.cull_face_mode = GL_BACK;
    c.st.front_face = GL_CCW;
    c.tri(v(5.0, 5.0, 0.0, RED), v(55.0, 8.0, 0.0, RED), v(20.0, 50.0, 0.0, RED));
    c.tri(v(5.0, 70.0, 0.0, GREEN), v(20.0, 120.0, 0.0, GREEN), v(55.0, 75.0, 0.0, GREEN));
    c.st.cull_face_mode = GL_FRONT;
    c.tri(v(70.0, 5.0, 0.0, BLUE), v(120.0, 8.0, 0.0, BLUE), v(85.0, 50.0, 0.0, BLUE));
    c.tri(v(70.0, 70.0, 0.0, YELLOW), v(85.0, 120.0, 0.0, YELLOW), v(120.0, 75.0, 0.0, YELLOW));
    c.st.front_face = GL_CW;
    c.st.cull_face_mode = GL_BACK;
    c.tri(v(60.0, 60.0, 0.0, WHITE), v(75.0, 100.0, 0.0, WHITE), v(100.0, 62.0, 0.0, WHITE));
}

fn two_sided_colours(c: &mut Ctx) {
    c.back_colors = true;
    let back = [0.9, 0.1, 0.9, 1.0];
    let mk = |x: f32, y: f32, col| V { back: Some(back), ..v(x, y, 0.0, col) };
    c.tri(mk(10.0, 10.0, RED), mk(60.0, 15.0, GREEN), mk(30.0, 70.0, BLUE));
    c.tri(mk(70.0, 10.0, RED), mk(90.0, 70.0, GREEN), mk(120.0, 15.0, BLUE));
}

fn polygon_offset(c: &mut Ctx) {
    c.depth_test(GL_LEQUAL);
    c.rect(10.0, 10.0, 118.0, 118.0, 0.2, BLUE);
    c.st.polygon_offset = GL_OFFSET_FILL;
    c.st.offset_factor = 1.0;
    c.st.offset_units = -20000.0;
    c.tri(v(10.0, 10.0, 0.2, RED), v(118.0, 20.0, 0.2, RED), v(40.0, 118.0, 0.2, RED));
    c.st.offset_units = 20000.0;
    c.tri(v(118.0, 118.0, 0.2, GREEN), v(10.0, 100.0, 0.2, GREEN), v(100.0, 10.0, 0.2, GREEN));
}

fn lines_flat(c: &mut Ctx) {
    c.st.shade_model = GL_FLAT;
    let mut rng = Lcg(11);
    for _ in 0..24 {
        let col = [rng.range(0.2, 1.0), rng.range(0.2, 1.0), rng.range(0.2, 1.0), 1.0];
        let (x0, y0, x1, y1) = (rng.range(2.0, 126.0), rng.range(2.0, 126.0), rng.range(2.0, 126.0), rng.range(2.0, 126.0));
        c.draw(GL_LINES, &[v(x0, y0, 0.0, col), v(x1, y1, 0.0, col)]);
    }
    // Axis-aligned and diagonal runs, where the end rule shows.
    for (i, (a, b)) in [((5.0, 5.0), (60.0, 5.0)), ((5.0, 8.0), (5.0, 60.0)), ((10.0, 10.0), (50.0, 50.0)), ((60.0, 60.0), (100.0, 20.0))].iter().enumerate() {
        c.draw(GL_LINES, &[v(a.0 + 60.0, a.1 + 60.0 + i as f32, 0.0, WHITE), v(b.0 + 60.0, b.1 + 60.0 + i as f32, 0.0, WHITE)]);
    }
}

fn lines_smooth_colour(c: &mut Ctx) {
    c.draw(GL_LINES, &[v(8.5, 10.5, 0.0, RED), v(120.5, 40.5, 0.0, BLUE)]);
    c.draw(GL_LINES, &[v(8.5, 120.5, 0.0, GREEN), v(120.5, 60.5, 0.0, YELLOW)]);
    c.draw(GL_LINES, &[v(64.5, 5.5, 0.0, WHITE), v(64.5, 125.5, 0.0, RED)]);
}

fn lines_wide(c: &mut Ctx) {
    for (i, w) in [2.0, 3.0, 5.0].iter().enumerate() {
        c.line_width = *w;
        let y = 20.0 + 35.0 * i as f32;
        c.draw(GL_LINES, &[v(10.0, y, 0.0, RED), v(110.0, y + 15.0, 0.0, GREEN)]);
        c.draw(GL_LINES, &[v(20.0 + 40.0 * i as f32, 5.0, 0.0, BLUE), v(30.0 + 40.0 * i as f32, 120.0, 0.0, YELLOW)]);
    }
}

fn lines_smooth(c: &mut Ctx) {
    c.line_smooth = true;
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    for (i, w) in [1.0, 2.0, 4.0].iter().enumerate() {
        c.line_width = *w;
        let y = 15.0 + 35.0 * i as f32;
        c.draw(GL_LINES, &[v(8.0, y, 0.0, WHITE), v(120.0, y + 20.0, 0.0, YELLOW)]);
    }
}

fn lines_stippled(c: &mut Ctx) {
    c.line_stipple = Some((1, 0x0f0f));
    c.draw(GL_LINES, &[v(5.0, 10.5, 0.0, WHITE), v(60.0, 10.5, 0.0, WHITE)]);
    // The second piece continues the first's pattern.
    c.draw(GL_LINES, &[v(60.0, 10.5, 0.0, WHITE), v(123.0, 10.5, 0.0, WHITE)]);
    c.line_stipple = Some((3, 0x00ff));
    c.draw(GL_LINES, &[v(5.0, 30.5, 0.0, YELLOW), v(120.0, 100.5, 0.0, YELLOW)]);
    c.line_width = 3.0;
    c.line_stipple = Some((2, 0xaaaa));
    c.draw(GL_LINES, &[v(5.0, 120.5, 0.0, RED), v(120.0, 60.5, 0.0, RED)]);
}

fn points(c: &mut Ctx) {
    let mut rng = Lcg(5);
    for size in [1.0, 2.0, 3.0, 6.0] {
        c.point_size = size;
        for _ in 0..12 {
            let col = [rng.range(0.2, 1.0), rng.range(0.2, 1.0), rng.range(0.2, 1.0), 1.0];
            c.draw(GL_POINTS, &[v(rng.range(6.0, 122.0), rng.range(6.0, 122.0), 0.0, col)]);
        }
    }
}

fn points_smooth(c: &mut Ctx) {
    c.point_smooth = true;
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    for (i, size) in [1.0, 2.5, 4.0, 7.0, 12.0].iter().enumerate() {
        c.point_size = *size;
        c.draw(GL_POINTS, &[v(15.0 + 22.0 * i as f32, 30.3 + 12.0 * i as f32, 0.0, YELLOW)]);
        c.draw(GL_POINTS, &[v(15.5 + 22.0 * i as f32, 100.5, 0.0, RED)]);
    }
}

/// Three overlapping triangles that cross in depth.
fn crossing(c: &mut Ctx) {
    c.tri(v(5.0, 10.0, -0.6, RED), v(120.0, 20.0, 0.6, RED), v(30.0, 120.0, 0.0, RED));
    c.tri(v(5.0, 110.0, 0.6, GREEN), v(120.0, 100.0, -0.6, GREEN), v(80.0, 5.0, 0.0, GREEN));
    c.tri(v(60.0, 5.0, 0.1, BLUE), v(70.0, 125.0, -0.1, BLUE), v(125.0, 60.0, 0.1, BLUE));
}

fn depth_less(c: &mut Ctx) {
    c.depth_test(GL_LESS);
    crossing(c);
}

fn depth_lequal_greater(c: &mut Ctx) {
    c.depth_test(GL_GREATER);
    c.clear(CLEAR, 0, 0);
    crossing(c);
}

fn depth_always_masked(c: &mut Ctx) {
    c.depth_test(GL_ALWAYS);
    c.st.depth_mask = false;
    c.rect(10.0, 10.0, 100.0, 100.0, 0.3, RED);
    c.st.depth_func = GL_LESS;
    c.st.depth_mask = true;
    c.rect(40.0, 40.0, 120.0, 120.0, 0.4, GREEN);
}

fn depth_equal_notequal(c: &mut Ctx) {
    c.depth_test(GL_LESS);
    c.rect(10.0, 10.0, 80.0, 80.0, 0.25, BLUE);
    c.st.depth_func = GL_EQUAL;
    c.rect(30.0, 30.0, 110.0, 110.0, 0.25, YELLOW);
    c.st.depth_func = GL_NOTEQUAL;
    c.rect(60.0, 5.0, 125.0, 60.0, 0.25, RED);
}

fn depth_range_wide(c: &mut Ctx) {
    // Window depth from the far plane to the near: the depth word's float
    // encoding across its exponents.
    c.depth_test(GL_LESS);
    for i in 0..8 {
        let z = -0.95 + 0.27 * i as f32;
        let x = 5.0 + 14.0 * i as f32;
        c.rect(x, 5.0 + 6.0 * i as f32, x + 40.0, 80.0 + 6.0 * i as f32, z, [0.1 + 0.12 * i as f32, 0.3, 1.0 - 0.1 * i as f32, 1.0]);
    }
}

fn stencil_mark_then_masked(c: &mut Ctx) {
    c.depth_test(GL_LESS);
    c.st.stencil_test = true;
    c.st.stencil_func = GL_ALWAYS;
    c.st.stencil_value_mask = u32::MAX;
    c.st.stencil_write_mask = u32::MAX;
    c.st.stencil_op = [GL_KEEP, GL_KEEP, GL_INCR];
    c.st.colour_mask = 0;
    c.board.gl_write_mask &= !(write::COLOUR_MASK << write::COLOUR_SHIFT);
    c.tri(v(10.0, 10.0, 0.0, RED), v(110.0, 30.0, 0.0, RED), v(40.0, 110.0, 0.0, RED));
    c.tri(v(30.0, 30.0, 0.0, RED), v(120.0, 90.0, 0.0, RED), v(20.0, 100.0, 0.0, RED));
    c.board.gl_write_mask |= write::COLOUR_MASK << write::COLOUR_SHIFT;
    c.st.stencil_func = GL_EQUAL;
    c.st.stencil_ref = 2;
    c.st.stencil_op = [GL_KEEP; 3];
    c.rect(0.0, 0.0, 128.0, 128.0, -0.5, GREEN);
    c.st.stencil_ref = 1;
    c.rect(0.0, 0.0, 128.0, 128.0, -0.5, YELLOW);
}

fn stencil_ops(c: &mut Ctx) {
    c.st.stencil_test = true;
    c.st.stencil_value_mask = u32::MAX;
    c.st.stencil_write_mask = 0xff;
    c.st.stencil_func = GL_ALWAYS;
    c.st.stencil_ref = 0x35;
    for (i, op) in [GL_REPLACE, GL_INCR, GL_DECR, GL_INVERT, GL_KEEP].iter().enumerate() {
        c.st.stencil_op = [GL_KEEP, GL_KEEP, *op];
        let x = 4.0 + 24.0 * i as f32;
        c.rect(x, 10.0, x + 22.0, 60.0, 0.0, RED);
        c.rect(x, 10.0, x + 22.0, 40.0, 0.0, RED);
    }
    c.st.stencil_func = GL_NOTEQUAL;
    c.st.stencil_ref = 0x35;
    c.st.stencil_value_mask = 0x0f;
    c.st.stencil_op = [GL_KEEP; 3];
    c.rect(0.0, 70.0, 128.0, 120.0, 0.0, BLUE);
}

fn stencil_depth_fail(c: &mut Ctx) {
    c.depth_test(GL_LESS);
    c.rect(20.0, 20.0, 100.0, 100.0, -0.5, BLUE);
    c.st.stencil_test = true;
    c.st.stencil_func = GL_ALWAYS;
    c.st.stencil_value_mask = u32::MAX;
    c.st.stencil_write_mask = 0xf0;
    c.st.stencil_ref = 0xff;
    c.st.stencil_op = [GL_KEEP, GL_REPLACE, GL_INCR];
    c.rect(5.0, 5.0, 80.0, 80.0, 0.5, RED);
}

fn alpha_test(c: &mut Ctx) {
    c.st.alpha_test = true;
    c.st.alpha_func = GL_GEQUAL;
    c.st.alpha_ref = 0.5;
    c.tri(v(5.0, 5.0, 0.0, [1.0, 0.0, 0.0, 1.0]), v(120.0, 10.0, 0.0, [0.0, 1.0, 0.0, 0.0]), v(30.0, 120.0, 0.0, [0.0, 0.0, 1.0, 0.5]));
    c.st.alpha_func = GL_LESS;
    c.st.alpha_ref = 0.3;
    c.tri(v(120.0, 120.0, 0.0, YELLOW), v(60.0, 60.0, 0.0, [1.0, 1.0, 1.0, 0.0]), v(120.0, 40.0, 0.0, WHITE));
}

/// A background to blend over: horizontal ramps.
fn ramp_background(c: &mut Ctx) {
    c.draw(GL_QUADS, &[v(0.0, 0.0, 0.9, [0.9, 0.2, 0.1, 0.8]), v(128.0, 0.0, 0.9, [0.1, 0.9, 0.3, 0.6]), v(128.0, 128.0, 0.9, [0.2, 0.3, 1.0, 0.4]), v(0.0, 128.0, 0.9, [1.0, 1.0, 0.1, 0.9])]);
}

/// Every source factor against a few destination factors, in a grid.
fn blend_factor_grid(c: &mut Ctx) {
    ramp_background(c);
    let srcs = [GL_ZERO_F, GL_ONE_F, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_SRC_ALPHA_SATURATE];
    let dsts = [GL_ZERO_F, GL_ONE_F, GL_SRC_COLOR, GL_ONE_MINUS_SRC_COLOR, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA];
    c.board.gl_write_mask |= write::COLOUR_MASK << write::COLOUR_SHIFT;
    for (i, &s) in srcs.iter().enumerate() {
        for (j, &d) in dsts.iter().enumerate() {
            c.blend(s, d);
            let (x, y) = (2.0 + 14.0 * i as f32, 2.0 + 20.0 * j as f32);
            c.draw(GL_QUADS, &[v(x, y, 0.0, [0.8, 0.5, 0.2, 0.6]), v(x + 12.0, y, 0.0, [0.3, 0.9, 0.6, 0.3]), v(x + 12.0, y + 18.0, 0.0, [0.9, 0.2, 0.9, 0.8]), v(x, y + 18.0, 0.0, [0.5, 0.5, 0.5, 0.5])]);
        }
    }
}

const GL_ZERO_F: u32 = 0;
const GL_ONE_F: u32 = 1;

fn blend_equations(c: &mut Ctx) {
    ramp_background(c);
    c.blend(GL_ONE_F, GL_ONE_F);
    for (i, eq) in [GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT, GL_MIN, GL_MAX].iter().enumerate() {
        c.st.blend_equation = *eq;
        let x = 4.0 + 24.0 * i as f32;
        c.draw(GL_QUADS, &[v(x, 10.0, 0.0, [0.6, 0.4, 0.2, 0.7]), v(x + 22.0, 10.0, 0.0, [0.2, 0.7, 0.9, 0.7]), v(x + 22.0, 118.0, 0.0, [0.8, 0.8, 0.1, 0.3]), v(x, 118.0, 0.0, [0.1, 0.1, 0.6, 0.3])]);
    }
}

fn blend_constant_colour(c: &mut Ctx) {
    ramp_background(c);
    c.st.blend_color = [0.25, 0.5, 0.75, 0.4];
    for (i, (s, d)) in [(GL_CONSTANT_COLOR, GL_ONE_MINUS_CONSTANT_COLOR), (GL_CONSTANT_ALPHA, GL_ONE_MINUS_CONSTANT_ALPHA), (GL_ONE_MINUS_CONSTANT_COLOR, GL_CONSTANT_COLOR), (GL_ONE_MINUS_CONSTANT_ALPHA, GL_CONSTANT_ALPHA)].iter().enumerate() {
        c.blend(*s, *d);
        let x = 4.0 + 31.0 * i as f32;
        c.rect(x, 10.0, x + 28.0, 118.0, 0.0, [0.9, 0.3, 0.2, 0.5]);
    }
}

fn blend_triangles_over(c: &mut Ctx) {
    ramp_background(c);
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    c.tri(v(10.0, 10.0, 0.0, [1.0, 0.0, 0.0, 0.8]), v(120.0, 20.0, 0.0, [0.0, 1.0, 0.0, 0.4]), v(40.0, 120.0, 0.0, [0.0, 0.0, 1.0, 0.1]));
    c.tri(v(120.0, 120.0, 0.0, [1.0, 1.0, 1.0, 0.7]), v(10.0, 100.0, 0.0, [1.0, 1.0, 1.0, 0.7]), v(100.0, 10.0, 0.0, [0.0, 0.0, 0.0, 0.7]));
}

fn logic_op_and_colour_mask(c: &mut Ctx) {
    ramp_background(c);
    c.st.color_logic_op = true;
    c.st.logic_op_mode = GL_CLEAR + 6; // XOR
    c.rect(10.0, 10.0, 60.0, 118.0, 0.0, [0.5, 0.7, 0.2, 1.0]);
    c.st.logic_op_mode = GL_CLEAR + 1; // AND
    c.rect(40.0, 40.0, 100.0, 100.0, 0.0, [0.9, 0.5, 0.8, 1.0]);
    c.st.color_logic_op = false;
    c.board.gl_write_mask = (c.board.gl_write_mask & !(write::COLOUR_MASK << write::COLOUR_SHIFT)) | ((GL_COLOUR_MASK_RED | GL_COLOUR_MASK_ALPHA) << write::COLOUR_SHIFT);
    c.rect(70.0, 20.0, 120.0, 110.0, 0.0, [1.0, 1.0, 1.0, 0.5]);
}

fn scissor(c: &mut Ctx) {
    c.st.scissor_test = true;
    c.st.scissor = [20, 30, 70, 50];
    c.tri(v(0.0, 0.0, 0.0, RED), v(128.0, 10.0, 0.0, GREEN), v(10.0, 128.0, 0.0, BLUE));
    c.draw(GL_LINES, &[v(0.0, 40.5, 0.0, WHITE), v(128.0, 60.5, 0.0, WHITE)]);
}

// Clipping.

/// Triangles that leave the window on each side, and one that covers it.
fn clip_frustum_xy(c: &mut Ctx) {
    c.tri(v(-40.0, 20.0, 0.0, RED), v(60.0, -30.0, 0.0, GREEN), v(70.0, 70.0, 0.0, BLUE));
    c.tri(v(100.0, -20.0, 0.0, YELLOW), v(180.0, 60.0, 0.0, YELLOW), v(90.0, 50.0, 0.0, RED));
    c.tri(v(-20.0, 90.0, 0.0, WHITE), v(60.0, 90.0, 0.0, GREEN), v(20.0, 190.0, 0.0, BLUE));
    c.draw(GL_QUADS, &[v(80.0, 80.0, 0.0, RED), v(200.0, 90.0, 0.0, GREEN), v(190.0, 200.0, 0.0, BLUE), v(70.0, 150.0, 0.0, YELLOW)]);
}

/// Depth outside [-w, w]: the near and far planes cut, with w varying.
fn clip_frustum_z(c: &mut Ctx) {
    c.depth_test(GL_LESS);
    // z beyond -w (near) at one corner, beyond w (far) at another.
    c.tri(
        V { p: [-0.9, -0.9, -2.0, 1.0], ..v(0.0, 0.0, 0.0, RED) },
        V { p: [0.9, -0.5, 0.5, 1.0], ..v(0.0, 0.0, 0.0, GREEN) },
        V { p: [0.0, 0.9, 1.9, 1.0], ..v(0.0, 0.0, 0.0, BLUE) },
    );
    // Window depth carried by w: a quad through the near plane in perspective.
    c.draw(
        GL_QUADS,
        &[
            v(70.0, 10.0, 0.0, YELLOW).persp(0.5, 0.2),
            v(120.0, 10.0, 0.0, YELLOW).persp(3.0, 0.9),
            v(120.0, 110.0, 0.0, RED).persp(3.0, 0.9),
            v(70.0, 110.0, 0.0, RED).persp(0.5, 0.2),
        ],
    );
}

/// Vertices at and behind the eye: w zero and negative.
fn clip_w_behind_eye(c: &mut Ctx) {
    c.tri(v(10.0, 10.0, 0.0, RED).persp(1.0, 0.5), v(120.0, 20.0, 0.0, GREEN).persp(-1.0, 0.5), v(40.0, 110.0, 0.0, BLUE).persp(2.0, 0.5));
    c.tri(v(10.0, 120.0, 0.0, YELLOW).persp(1.0, 0.5), v(100.0, 100.0, 0.0, WHITE).persp(0.0, 0.5), v(120.0, 50.0, 0.0, RED).persp(1.5, 0.5));
    c.draw(GL_LINES, &[v(10.0, 64.0, 0.0, WHITE).persp(2.0, 0.5), v(120.0, 64.0, 0.0, WHITE).persp(-0.5, 0.5)]);
}

fn clip_user_plane(c: &mut Ctx) {
    // x + y/4 >= -0.25 in eye space, as the bench's mirror scene cuts.
    c.st.clip_planes = 1;
    c.st.clip_plane[0] = [1.0, 0.25, 0.0, 0.25];
    c.tri(v(5.0, 5.0, 0.0, RED), v(120.0, 20.0, 0.0, GREEN), v(30.0, 120.0, 0.0, BLUE));
    c.draw(GL_LINES, &[v(0.0, 100.5, 0.0, WHITE), v(128.0, 30.5, 0.0, WHITE)]);
    c.point_size = 4.0;
    for i in 0..6 {
        c.draw(GL_POINTS, &[v(8.0 + 20.0 * i as f32, 8.0 + 20.0 * i as f32, 0.0, YELLOW)]);
    }
}

fn clip_user_planes_rotated(c: &mut Ctx) {
    // Two planes under a modelview that rotates 30 degrees and moves: the
    // plane is in eye space, so its edge in the window turns with it.
    let (s, k) = (0.5f32, 0.866_025_4f32);
    c.st.modelview = [k, s, 0.0, 0.0, -s, k, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, -0.05, 0.0, 1.0];
    c.st.clip_planes = 0b101;
    c.st.clip_plane[0] = [0.0, 1.0, 0.0, 0.3];
    c.st.clip_plane[2] = [-1.0, 0.2, 0.3, 0.5];
    c.tri(v(5.0, 5.0, 0.0, RED), v(125.0, 15.0, 0.0, GREEN), v(20.0, 125.0, 0.0, BLUE));
    c.tri(v(125.0, 125.0, 0.0, YELLOW), v(10.0, 110.0, 0.0, WHITE), v(110.0, 10.0, 0.0, RED));
}

fn clip_user_plane_textured_smooth(c: &mut Ctx) {
    c.texture(&Tex { min: 1, mag_linear: true, ..Tex::new(RGBA) });
    c.st.clip_planes = 2;
    c.st.clip_plane[1] = [-0.5, 1.0, 0.0, 0.1];
    c.depth_test(GL_LESS);
    c.tri(v(5.0, 5.0, 0.0, WHITE).tex(0.0, 0.0), v(125.0, 10.0, 0.0, WHITE).tex(1.0, 0.0), v(60.0, 125.0, 0.0, WHITE).tex(0.5, 1.0));
}

// The window-matrix mode: one eye-to-window transform in place of the
// projection and the viewport map.

fn window_matrix(c: &mut Ctx, rows_down: bool) {
    c.st.window_matrix_given = true;
    c.st.window_depth_scale = 1.0;
    c.st.window_rows_down = rows_down;
    // Eye coordinates are window pixels and depth in [0, 1].
    let mut m = [0.0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    m[12] = 0.0;
    c.st.window_matrix = m;
    c.depth_row([0.0, 0.0, 1.0, 0.0]);
    // The window-matrix y is in the draw's own rows.
    if rows_down {
        c.st.drawable.height = SIDE as u32;
    }
}

fn wv(x: f32, y: f32, z: f32, col: [f32; 4]) -> V {
    V { p: [x, y, z, 1.0], ..v(0.0, 0.0, 0.0, col) }
}

fn window_matrix_triangles(c: &mut Ctx) {
    window_matrix(c, false);
    c.depth_test(GL_LESS);
    c.tri(wv(10.0, 10.0, 0.2, RED), wv(110.0, 25.0, 0.4, GREEN), wv(40.0, 115.0, 0.6, BLUE));
    c.tri(wv(100.0, 100.0, 0.1, YELLOW), wv(20.0, 90.0, 0.7, WHITE), wv(110.0, 20.0, 0.5, RED));
}

fn window_matrix_rows_down(c: &mut Ctx) {
    window_matrix(c, true);
    c.st.cull_face = true;
    c.st.cull_face_mode = GL_BACK;
    c.st.front_face = GL_CCW;
    c.tri(wv(10.0, 10.0, 0.5, RED), wv(60.0, 15.0, 0.5, GREEN), wv(20.0, 60.0, 0.5, BLUE));
    c.tri(wv(70.0, 10.0, 0.5, YELLOW), wv(80.0, 60.0, 0.5, YELLOW), wv(120.0, 15.0, 0.5, YELLOW));
}

fn window_matrix_clip_and_lines(c: &mut Ctx) {
    window_matrix(c, false);
    c.depth_test(GL_LESS);
    // Outside the depth range and outside the window.
    c.tri(wv(-30.0, 10.0, 0.5, RED), wv(60.0, 5.0, 1.4, GREEN), wv(30.0, 90.0, -0.4, BLUE));
    c.tri(wv(90.0, 60.0, 0.3, YELLOW), wv(200.0, 70.0, 0.3, YELLOW), wv(120.0, 190.0, 0.3, WHITE));
    c.draw(GL_LINES, &[wv(5.0, 100.5, 0.2, WHITE), wv(125.0, 120.5, 0.8, RED)]);
    c.point_size = 3.0;
    c.draw(GL_POINTS, &[wv(64.0, 64.0, 0.3, GREEN)]);
}

fn window_matrix_scaled(c: &mut Ctx) {
    // A real transform: scale, rotate and offset, with a depth scale below 1.
    window_matrix(c, false);
    c.st.window_depth_scale = 0.5;
    let (s, k) = (0.6f32, 0.8f32);
    c.st.window_matrix = [60.0 * k, 60.0 * s, 0.0, 0.0, -60.0 * s, 60.0 * k, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 64.0, 64.0, 0.0, 1.0];
    c.depth_test(GL_LESS);
    c.tri(V { p: [-0.8, -0.8, 0.3, 1.0], ..wv(0.0, 0.0, 0.0, RED) }, V { p: [0.9, -0.7, 0.5, 1.0], ..wv(0.0, 0.0, 0.0, GREEN) }, V { p: [0.0, 0.9, 0.2, 1.0], ..wv(0.0, 0.0, 0.0, BLUE) });
}

// Perspective and the general transform.

fn projection_frustum(c: &mut Ctx) {
    // glFrustum(-0.5, 0.5, -0.5, 0.5, 1, 20) looking at a floor and a wall,
    // under a modelview that moves the scene 4 units away.
    let (l, r, b, t, n, f) = (-0.5f32, 0.5f32, -0.5f32, 0.5f32, 1.0f32, 20.0f32);
    let p = [2.0 * n / (r - l), 0.0, 0.0, 0.0, 0.0, 2.0 * n / (t - b), 0.0, 0.0, (r + l) / (r - l), (t + b) / (t - b), -(f + n) / (f - n), -1.0, 0.0, 0.0, -2.0 * f * n / (f - n), 0.0];
    c.st.projection = p;
    c.st.modelview = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -4.0, 1.0];
    // The board's window depth is linear in eye distance, 0 at the near plane
    // and 1 at the far: the depth row applied to the eye position.
    c.depth_row([0.0, 0.0, -1.0 / (f - n) as f64, -n as f64 / (f - n) as f64]);
    c.depth_test(GL_LESS);
    let o = |x: f32, y: f32, z: f32, col| V { p: [x, y, z, 1.0], ..v(0.0, 0.0, 0.0, col) };
    // A floor receding, a wall across it, a tilted slab through both.
    c.draw(GL_QUADS, &[o(-2.0, -1.0, 2.0, RED), o(2.0, -1.0, 2.0, GREEN), o(2.0, -1.0, -10.0, BLUE), o(-2.0, -1.0, -10.0, YELLOW)]);
    c.draw(GL_QUADS, &[o(-1.0, -1.0, -2.0, WHITE), o(1.0, -1.0, -2.0, WHITE), o(1.0, 1.0, -2.0, RED), o(-1.0, 1.0, -2.0, RED)]);
    c.tri(o(-1.5, -0.5, 1.0, GREEN), o(1.5, 0.5, -3.0, BLUE), o(0.0, 1.5, -1.0, YELLOW));
}

fn projection_ortho_rotated(c: &mut Ctx) {
    // glOrtho(-2, 2, -2, 2, -1, 1) under a modelview that rotates about z
    // and moves: depth row (a z + b + 1) / 2 for z' = a z + b.
    c.st.projection = [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let (s, k) = (0.7071f32, 0.7071f32);
    c.st.modelview = [k, s, 0.0, 0.0, -s, k, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.3, -0.2, 0.0, 1.0];
    c.depth_row([0.0, 0.0, -0.5, 0.5]);
    c.depth_test(GL_LESS);
    let o = |x: f32, y: f32, z: f32, col| V { p: [x, y, z, 1.0], ..v(0.0, 0.0, 0.0, col) };
    c.tri(o(-1.5, -1.0, -0.5, RED), o(1.5, -1.2, 0.5, GREEN), o(0.0, 1.7, 0.0, BLUE));
    c.draw(GL_QUADS, &[o(-1.0, -1.0, 0.2, YELLOW), o(1.0, -1.0, 0.2, YELLOW), o(1.0, 1.0, -0.2, WHITE), o(-1.0, 1.0, -0.2, WHITE)]);
}

fn perspective_correct_colour(c: &mut Ctx) {
    // A quad that recedes: colour and texture are perspective-corrected.
    c.texture(&Tex { min: 1, mag_linear: true, ..Tex::new(RGB) });
    c.depth_test(GL_LESS);
    c.draw(
        GL_QUADS,
        &[
            v(5.0, 5.0, 0.0, WHITE).tex(0.0, 0.0).persp(1.0, 0.1),
            v(123.0, 5.0, 0.0, WHITE).tex(4.0, 0.0).persp(4.0, 0.9),
            v(123.0, 123.0, 0.0, WHITE).tex(4.0, 4.0).persp(4.0, 0.9),
            v(5.0, 123.0, 0.0, WHITE).tex(0.0, 4.0).persp(1.0, 0.1),
        ],
    );
}

// Textures: formats, environments, filters.

fn textured(c: &mut Ctx, t: Tex, vertex: [f32; 4]) {
    c.texture(&t);
    let col = vertex;
    c.draw(GL_QUADS, &[v(8.0, 8.0, 0.0, col).tex(0.0, 0.0), v(120.0, 8.0, 0.0, col).tex(1.0, 0.0), v(120.0, 120.0, 0.0, col).tex(1.0, 1.0), v(8.0, 120.0, 0.0, col).tex(0.0, 1.0)]);
}

macro_rules! format_scenes {
    ($($name:ident: $fmt:expr,)*) => {$(
        fn $name(c: &mut Ctx) {
            textured(c, Tex { mag_linear: true, min: 1, ..Tex::new($fmt) }, [0.9, 0.8, 0.6, 0.8]);
        }
    )*};
}

format_scenes! {
    tex_format_rgba: RGBA,
    tex_format_rgb: RGB,
    tex_format_luminance: LUMINANCE,
    tex_format_luminance_alpha: LUMINANCE_ALPHA,
    tex_format_alpha: ALPHA,
    tex_format_intensity: INTENSITY,
}

macro_rules! env_scenes {
    ($($name:ident: $env:expr, $fmt:expr,)*) => {$(
        fn $name(c: &mut Ctx) {
            c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            ramp_background(c);
            textured(c, Tex { env: $env, mag_linear: true, min: 1, ..Tex::new($fmt) }, [0.9, 0.5, 0.7, 0.7]);
        }
    )*};
}

env_scenes! {
    tex_env_replace_rgba: 1, RGBA,
    tex_env_modulate_rgba: 2, RGBA,
    tex_env_decal_rgba: 3, RGBA,
    tex_env_decal_rgb: 3, RGB,
    tex_env_blend_rgba: 4, RGBA,
    tex_env_blend_luminance: 4, LUMINANCE,
    tex_env_modulate_luminance_alpha: 2, LUMINANCE_ALPHA,
    tex_env_replace_alpha: 1, ALPHA,
}

fn tex_nearest_repeat(c: &mut Ctx) {
    c.texture(&Tex { min: 0, mag_linear: false, ..Tex::new(RGBA) });
    c.draw(GL_QUADS, &[v(5.0, 5.0, 0.0, WHITE).tex(-1.0, -1.0), v(123.0, 5.0, 0.0, WHITE).tex(2.5, -1.0), v(123.0, 123.0, 0.0, WHITE).tex(2.5, 2.5), v(5.0, 123.0, 0.0, WHITE).tex(-1.0, 2.5)]);
}

fn tex_clamp_to_edge_linear(c: &mut Ctx) {
    c.texture(&Tex { min: 1, mag_linear: true, wrap: 1, ..Tex::new(RGBA) });
    c.draw(GL_QUADS, &[v(5.0, 5.0, 0.0, WHITE).tex(-0.5, -0.5), v(123.0, 5.0, 0.0, WHITE).tex(1.5, -0.5), v(123.0, 123.0, 0.0, WHITE).tex(1.5, 1.5), v(5.0, 123.0, 0.0, WHITE).tex(-0.5, 1.5)]);
}

macro_rules! mip_scenes {
    ($($name:ident: $min:expr,)*) => {$(
        fn $name(c: &mut Ctx) {
            // A texture of 32 texels minified to about a quarter, and a
            // recession through several levels.
            c.texture(&Tex { size: 32, min: $min, mag_linear: true, ..Tex::new(RGB) });
            c.draw(GL_QUADS, &[v(4.0, 4.0, 0.0, WHITE).tex(0.0, 0.0), v(60.0, 4.0, 0.0, WHITE).tex(4.0, 0.0), v(60.0, 60.0, 0.0, WHITE).tex(4.0, 4.0), v(4.0, 60.0, 0.0, WHITE).tex(0.0, 4.0)]);
            c.depth_test(GL_LESS);
            c.draw(GL_QUADS, &[
                v(66.0, 4.0, 0.0, WHITE).tex(0.0, 0.0).persp(1.0, 0.2),
                v(124.0, 4.0, 0.0, WHITE).tex(6.0, 0.0).persp(6.0, 0.8),
                v(124.0, 124.0, 0.0, WHITE).tex(6.0, 6.0).persp(6.0, 0.8),
                v(66.0, 124.0, 0.0, WHITE).tex(0.0, 6.0).persp(1.0, 0.2),
            ]);
        }
    )*};
}

mip_scenes! {
    tex_mip_nearest_nearest: 2,
    tex_mip_linear_nearest: 3,
    tex_mip_nearest_linear: 4,
    tex_mip_linear_linear: 5,
}

/// SGIS_detail_texture under GL_LINEAR_DETAIL_COLOR_SGIS, as SGI Quake draws
/// its world: a mipmapped RGB base with a 16x16 detail image at the default
/// level -4 (level-table entry 60), and the library's ramp from weight 0 at
/// LOD 0 to 1 at LOD -4. Both quads magnify, so the weight comes from
/// between the table's entries.
fn tex_detail_colour(c: &mut Ctx) {
    tex_detail(c, 0);
}

/// The same scene under the modulate combine, word 0 bit 31.
fn tex_detail_colour_modulate(c: &mut Ctx) {
    tex_detail(c, 0x8000_0000);
}

fn tex_detail(c: &mut Ctx, combine: u32) {
    c.texture(&Tex { size: 32, min: 5, ..Tex::new(RGB) });
    let b = &mut c.board;
    let (size, base) = (16u32, 0x18_0000u32);
    let units = (size.div_ceil(16) + 1) & !1;
    let obj = b.object_select(base, units * 16);
    assert!(b.object_rows(obj, size));
    for y in 0..size {
        for x in 0..size {
            let g = if (x / 2 + y) & 1 == 0 { 0xc0 } else { 0x40 } + (x * 3) as u32;
            b.object_store(obj, x, y, (g << 24) | (g << 16) | (g << 8) | 0xff);
        }
    }
    b.tex.levels[60 * 3..][..3].copy_from_slice(&[base, 0x00d0_0000 | units, 0]);
    // Detail level 4 in bits 29:25, log2 16 in 24:20 and 19:15.
    b.tex.sampler[0] = combine | 0x0800_0000 | (4 << 20) | (4 << 15) | (5 << 10) | (5 << 5);
    // Minification code 5 and magnification code 4, detail colour.
    b.tex.sampler[4] = 0x1000_0000 | RGB | (5 << 5) | (4 << 1);
    for (i, e) in b.tex.detail_function.iter_mut().enumerate() {
        *e = ((i as u32 * 32).min(4095)) << 12;
    }
    b.tex.detail_function_valid = 0xffff;
    c.draw(GL_QUADS, &[v(4.0, 4.0, 0.0, WHITE).tex(0.0, 0.0), v(62.0, 4.0, 0.0, WHITE).tex(1.0, 0.0), v(62.0, 124.0, 0.0, WHITE).tex(1.0, 2.0), v(4.0, 124.0, 0.0, WHITE).tex(0.0, 2.0)]);
    c.draw(GL_QUADS, &[v(66.0, 4.0, 0.0, WHITE).tex(0.0, 0.0), v(124.0, 4.0, 0.0, WHITE).tex(0.125, 0.0), v(124.0, 124.0, 0.0, WHITE).tex(0.125, 0.25), v(66.0, 124.0, 0.0, WHITE).tex(0.0, 0.25)]);
}

fn tex_with_fog_blend_depth(c: &mut Ctx) {
    c.depth_test(GL_LESS);
    c.fog([200, 210, 230], 1);
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    c.texture(&Tex { min: 5, size: 32, mag_linear: true, ..Tex::new(RGBA) });
    for i in 0..4 {
        let z = -0.6 + 0.4 * i as f32;
        let x = 6.0 + 20.0 * i as f32;
        c.draw(GL_QUADS, &[v(x, 6.0, z, WHITE).tex(0.0, 0.0), v(x + 70.0, 6.0, z, WHITE).tex(2.0, 0.0), v(x + 70.0, 116.0, z, WHITE).tex(2.0, 2.0), v(x, 116.0, z, WHITE).tex(0.0, 2.0)]);
    }
}

// Fog.

fn fog_ramp_triangles(c: &mut Ctx) {
    c.depth_test(GL_LESS);
    c.fog([180, 190, 210], 1);
    c.draw(GL_QUADS, &[v(5.0, 5.0, -0.9, RED), v(123.0, 5.0, 0.9, GREEN), v(123.0, 60.0, 0.9, GREEN), v(5.0, 60.0, -0.9, RED)]);
    c.tri(v(5.0, 70.0, -0.8, BLUE), v(120.0, 70.0, 0.2, YELLOW), v(60.0, 125.0, 0.8, WHITE));
}

fn fog_lines_and_points(c: &mut Ctx) {
    c.fog([255, 128, 0], 2);
    c.draw(GL_LINES, &[v(5.0, 10.5, -0.9, WHITE), v(120.0, 40.5, 0.9, WHITE)]);
    c.point_size = 5.0;
    for i in 0..6 {
        c.draw(GL_POINTS, &[v(10.0 + 20.0 * i as f32, 80.0, -0.9 + 0.3 * i as f32, GREEN)]);
    }
}

fn fog_over_blend(c: &mut Ctx) {
    ramp_background(c);
    c.fog([40, 40, 40], 1);
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    c.tri(v(5.0, 5.0, -0.9, [1.0, 1.0, 1.0, 0.8]), v(120.0, 20.0, 0.9, [1.0, 1.0, 1.0, 0.8]), v(40.0, 120.0, 0.0, [1.0, 1.0, 1.0, 0.8]));
}

// Lighting. The board lights a vertex from the words the command stream
// loaded (Board::gl_light()); the scenes load them and light a sphere's
// vertices the way the front end would before it hands the draw on.

#[derive(Clone, Copy, Default)]
pub struct Light {
    pub pos: [f32; 3],
    pub directional: bool,
    pub colour: [f32; 3],
    /// Constant, linear, quadratic.
    pub atten: [f32; 3],
    /// Spot direction, cutoff cosine and exponent; a cosine of -1 is no spot.
    pub spot: Option<([f32; 3], f32, f32)>,
}

pub fn load_lights(b: &mut Board, lights: &[(usize, Light)], ambient: [f32; 3], local_viewer: bool, shininess: f32, material: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]), colour_material: u32) {
    let identity: [f32; 16] = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];
    b.geom.modelview = identity.map(f32::to_bits);
    b.geom.normal_matrix = identity.map(f32::to_bits);
    b.geom.local_viewer = local_viewer;
    b.geom.normalize = true;
    b.geom.colour_material = colour_material;
    b.geom.light_colour.fill(0);
    b.geom.light_vec.fill(0);
    let a = gl_cmd::LIGHT_MODEL_AMBIENT as usize * 4;
    for k in 0..3 {
        b.geom.light_colour[a + k] = ambient[k].to_bits();
    }
    let (ma, md, ms, me) = material;
    for k in 0..3 {
        b.geom.material[k] = ma[k].to_bits();
        b.geom.material[4 + k] = md[k].to_bits();
        b.geom.material[8 + k] = ms[k].to_bits();
        b.geom.material[12 + k] = me[k].to_bits();
    }
    b.geom.material[3] = 1.0f32.to_bits();
    b.geom.material[7] = 0.9f32.to_bits();
    b.geom.material[11] = 1.0f32.to_bits();
    b.geom.material[15] = 1.0f32.to_bits();
    b.geom.shininess_sqrt = shininess.sqrt().to_bits();
    for &(i, l) in lights {
        for k in 0..3 {
            b.geom.light_colour[i * 12 + k] = (l.colour[k] * 0.2).to_bits();
            b.geom.light_colour[i * 12 + 4 + k] = l.colour[k].to_bits();
            b.geom.light_colour[i * 12 + 8 + k] = l.colour[k].to_bits();
        }
        let blk = &mut b.geom.light_vec[i * gl_cmd::LIGHT_VEC_FLOATS as usize..];
        for k in 0..3 {
            blk[light::POS + k] = l.pos[k].to_bits();
        }
        if l.directional {
            // The board is given the half vector; any non-zero one marks the
            // light directional.
            let n = (l.pos[0] * l.pos[0] + l.pos[1] * l.pos[1] + (l.pos[2] + 1.0).powi(2)).sqrt().max(1e-6);
            blk[light::HALF] = (l.pos[0] / n).to_bits();
            blk[light::HALF + 1] = (l.pos[1] / n).to_bits();
            blk[light::HALF + 2] = ((l.pos[2] + 1.0) / n).to_bits();
        }
        blk[light::ATTEN] = l.atten[2].to_bits();
        blk[light::ATTEN + 1] = l.atten[0].to_bits();
        blk[light::ATTEN + 2] = l.atten[1].to_bits();
        blk[light::SPOT_COS] = 1.0f32.to_bits();
        if let Some((dir, cos, exp)) = l.spot {
            blk[light::SPOT_COS] = (-cos).to_bits();
            blk[light::SPOT_EXP] = (-exp).to_bits();
            for k in 0..3 {
                blk[light::SPOT_DIR + k] = (-dir[k]).to_bits();
            }
        }
    }
}

/// A sphere of radius 0.8 about the window's middle (and a flat disc behind
/// it), lit vertex by vertex, drawn smooth and depth tested.
fn lit_sphere(c: &mut Ctx, current_colour: [f32; 4]) {
    c.depth_test(GL_LESS);
    c.board.geom.colour = current_colour.map(f32::to_bits);
    let (nlat, nlon) = (10usize, 16usize);
    let point = |i: usize, j: usize| -> [f32; 3] {
        let th = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / nlat as f32;
        let ph = 2.0 * std::f32::consts::PI * j as f32 / nlon as f32;
        [th.cos() * ph.sin(), th.sin(), -th.cos() * ph.cos()]
    };
    let lit = |c: &mut Ctx, p: [f32; 3]| -> V {
        let n = p;
        c.board.geom.normal = n.map(f32::to_bits);
        let obj = [p[0] * 0.8, p[1] * 0.8, p[2] * 0.8, 1.0];
        let rgba = c.board.gl_light(&obj);
        V { p: [p[0] * 0.8, p[1] * 0.8, p[2] * 0.8, 1.0], c: rgba, t: [0.0, 0.0, 0.0, 1.0], n, back: None }
    };
    for i in 0..nlat {
        for j in 0..nlon {
            let q = [point(i, j), point(i + 1, j), point(i + 1, j + 1), point(i, j + 1)];
            let vs: Vec<V> = q.iter().map(|&p| lit(c, p)).collect();
            c.draw(GL_TRIANGLES, &[vs[0], vs[1], vs[2]]);
            c.draw(GL_TRIANGLES, &[vs[0], vs[2], vs[3]]);
        }
    }
}

pub const MATERIAL: ([f32; 3], [f32; 3], [f32; 3], [f32; 3]) = ([0.2, 0.2, 0.25], [0.8, 0.5, 0.4], [0.9, 0.9, 0.9], [0.02, 0.0, 0.05]);

fn light_directional(c: &mut Ctx) {
    let l = Light { pos: [0.5, 0.6, -0.62], directional: true, colour: [1.0, 0.95, 0.85], atten: [1.0, 0.0, 0.0], spot: None };
    load_lights(&mut c.board, &[(0, l)], [0.2, 0.2, 0.2], false, 20.0, MATERIAL, 0);
    lit_sphere(c, WHITE);
}

fn light_positional_attenuated(c: &mut Ctx) {
    let l = Light { pos: [1.5, 1.0, -2.0], directional: false, colour: [1.0, 1.0, 1.0], atten: [1.0, 0.3, 0.15], spot: None };
    load_lights(&mut c.board, &[(0, l)], [0.1, 0.1, 0.1], false, 8.0, MATERIAL, 0);
    lit_sphere(c, WHITE);
}

fn light_spot(c: &mut Ctx) {
    let l = Light { pos: [0.0, 1.5, -2.5], directional: false, colour: [1.0, 1.0, 0.8], atten: [1.0, 0.0, 0.0], spot: Some(([0.0, -0.5, 0.85], 0.8, 4.0)) };
    load_lights(&mut c.board, &[(0, l)], [0.05, 0.05, 0.1], false, 30.0, MATERIAL, 0);
    lit_sphere(c, WHITE);
}

fn light_local_viewer_specular(c: &mut Ctx) {
    let l = Light { pos: [-1.0, 1.0, -1.5], directional: false, colour: [0.9, 0.9, 1.0], atten: [1.0, 0.0, 0.0], spot: None };
    load_lights(&mut c.board, &[(2, l)], [0.15, 0.15, 0.15], true, 64.0, MATERIAL, 0);
    lit_sphere(c, WHITE);
}

fn light_three_lights_colour_material(c: &mut Ctx) {
    let a = Light { pos: [1.0, 0.0, -1.0], directional: true, colour: [1.0, 0.1, 0.1], atten: [1.0, 0.0, 0.0], spot: None };
    let b = Light { pos: [-1.0, 0.5, -1.0], directional: true, colour: [0.1, 0.1, 1.0], atten: [1.0, 0.0, 0.0], spot: None };
    let s = Light { pos: [0.0, -2.0, -1.5], directional: false, colour: [0.2, 1.0, 0.2], atten: [1.0, 0.1, 0.0], spot: Some(([0.0, 1.0, 0.5], 0.7, 2.0)) };
    load_lights(&mut c.board, &[(0, a), (3, b), (7, s)], [0.1, 0.1, 0.1], false, 12.0, MATERIAL, colour::MATERIAL_AMBIENT | colour::MATERIAL_DIFFUSE);
    lit_sphere(c, [0.9, 0.7, 0.3, 0.75]);
}

fn light_unlit_emission_only(c: &mut Ctx) {
    load_lights(&mut c.board, &[], [0.3, 0.3, 0.3], false, 1.0, MATERIAL, colour::MATERIAL_EMISSION);
    lit_sphere(c, [0.2, 0.8, 0.4, 1.0]);
}

fn light_lit_blended_textured(c: &mut Ctx) {
    let l = Light { pos: [0.5, 0.6, -0.62], directional: true, colour: [1.0, 1.0, 1.0], atten: [1.0, 0.0, 0.0], spot: None };
    load_lights(&mut c.board, &[(0, l)], [0.2, 0.2, 0.2], false, 20.0, MATERIAL, 0);
    c.texture(&Tex { min: 1, mag_linear: true, ..Tex::new(RGB) });
    c.blend(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ramp_background(c);
    // Texture coordinates by latitude and longitude.
    c.depth_test(GL_LESS);
    let (nlat, nlon) = (8usize, 12usize);
    let pt = |c: &mut Ctx, i: usize, j: usize| -> V {
        let th = -std::f32::consts::FRAC_PI_2 + std::f32::consts::PI * i as f32 / nlat as f32;
        let ph = 2.0 * std::f32::consts::PI * j as f32 / nlon as f32;
        let p = [th.cos() * ph.sin(), th.sin(), -th.cos() * ph.cos()];
        c.board.geom.normal = p.map(f32::to_bits);
        c.board.geom.colour = WHITE.map(f32::to_bits);
        let rgba = c.board.gl_light(&[p[0] * 0.8, p[1] * 0.8, p[2] * 0.8, 1.0]);
        V { p: [p[0] * 0.8, p[1] * 0.8, p[2] * 0.8, 1.0], c: rgba, t: [j as f32 / nlon as f32 * 2.0, i as f32 / nlat as f32 * 2.0, 0.0, 1.0], n: p, back: None }
    };
    for i in 0..nlat {
        for j in 0..nlon {
            let q = [pt(c, i, j), pt(c, i + 1, j), pt(c, i + 1, j + 1), pt(c, i, j + 1)];
            c.draw(GL_QUADS, &q);
        }
    }
}

macro_rules! scenes {
    ($($name:ident,)*) => {
        /// Every scene, in the order the goldens list them.
        pub const NAMES: &[&str] = &[$(stringify!($name)),*];

        pub fn run_scene(name: &str, c: &mut Ctx) -> bool {
            match name {
                $(stringify!($name) => $name(c),)*
                _ => return false,
            }
            true
        }

        #[cfg(test)]
        mod scene_tests {
            $(#[test] fn $name() { super::super::check(stringify!($name)); })*
        }
    };
}

scenes! {
    tri_flat,
    tri_smooth,
    tri_tiny_and_thin,
    quad_shared_edges,
    polygon_convex,
    polygon_modes_line_point,
    polygon_stipple,
    polygon_smooth,
    cull_and_front_face,
    two_sided_colours,
    polygon_offset,
    lines_flat,
    lines_smooth_colour,
    lines_wide,
    lines_smooth,
    lines_stippled,
    points,
    points_smooth,
    depth_less,
    depth_lequal_greater,
    depth_always_masked,
    depth_equal_notequal,
    depth_range_wide,
    stencil_mark_then_masked,
    stencil_ops,
    stencil_depth_fail,
    alpha_test,
    blend_factor_grid,
    blend_equations,
    blend_constant_colour,
    blend_triangles_over,
    logic_op_and_colour_mask,
    scissor,
    clip_frustum_xy,
    clip_frustum_z,
    clip_w_behind_eye,
    clip_user_plane,
    clip_user_planes_rotated,
    clip_user_plane_textured_smooth,
    window_matrix_triangles,
    window_matrix_rows_down,
    window_matrix_clip_and_lines,
    window_matrix_scaled,
    projection_frustum,
    projection_ortho_rotated,
    perspective_correct_colour,
    tex_format_rgba,
    tex_format_rgb,
    tex_format_luminance,
    tex_format_luminance_alpha,
    tex_format_alpha,
    tex_format_intensity,
    tex_env_replace_rgba,
    tex_env_modulate_rgba,
    tex_env_decal_rgba,
    tex_env_decal_rgb,
    tex_env_blend_rgba,
    tex_env_blend_luminance,
    tex_env_modulate_luminance_alpha,
    tex_env_replace_alpha,
    tex_nearest_repeat,
    tex_clamp_to_edge_linear,
    tex_mip_nearest_nearest,
    tex_mip_linear_nearest,
    tex_mip_nearest_linear,
    tex_mip_linear_linear,
    tex_detail_colour,
    tex_detail_colour_modulate,
    tex_with_fog_blend_depth,
    fog_ramp_triangles,
    fog_lines_and_points,
    fog_over_blend,
    light_directional,
    light_positional_attenuated,
    light_spot,
    light_local_viewer_specular,
    light_three_lights_colour_material,
    light_unlit_emission_only,
    light_lit_blended_textured,
}
