//! The GPU raster unit against the CPU pipeline, word for word: the same
//! triangles, shaded quads and clears through `pipe::raster`,
//! `pipe::interp`, `pipe::fixed` and `pipe::backend` into a `Vram`, and
//! through `GpuRaster` into its GPU memory, then both memories compared.
//! They need a GPU (or lavapipe) and skip without one.

use std::sync::Arc;

use super::super::clip::ClipRect;
use super::super::consts::{gl_cmd, mode, write, ODSY_DEPTH24_MAX};
use super::super::depth_word::word_for_fraction;
use super::super::pipe::backend::{colour12, BackendState, ClearDraw, DestMem};
use super::super::pipe::aux_targets;
use super::super::pipe::fixed;
use super::super::pipe::interp::{self, TriTex};
use super::super::pipe::raster::{triangle_quads, walk_bounds, Quad, RasterState, RasterVertex, Triangle};
use super::super::pipe::ring::BackQuad;
use super::super::pipe::setup::Tri;
use super::super::pipe::shader::ShaderState;
use super::super::pipe::state::{BackendWords, Compare, RasterWords, ShaderWords};
use super::super::pipe::texture::{Latch, TexRegs, Unit, Wrap};
use super::super::consts::tex;
use super::super::view::DestViews;
use super::super::vram::Vram;
use super::{GpuRaster, Rec, Slot, TexSlot};
use crate::wgpu::{Gpu, GpuOptions};

/// The test memory: 4 MB, a power of two.
const VRAM_BYTES: u64 = 1 << 22;
/// The plane the batch grid covers.
const GRID: u32 = 512;
/// The RGBA destination: 16-byte pixels, 256 wide, from byte 0x1000
/// (the pixel mode Board::test_screen() uses, narrowed), and the
/// window-id surface at 2 MB with 2-byte pixels.
const RGBA_MODE: [u32; 4] = [0x100, 16, 0x0040_6d50, 0];
const WID_BASE: u32 = 0x20000;
const WID_STRIDE: u32 = 16;
/// The overlay as a destination: an index visual of 2-byte pixels over
/// the window-id surface (vram.rs tests' OVERLAY select).
const OVERLAY_MODE: [u32; 4] = [WID_BASE, WID_STRIDE, 0x2008_0080, 0];

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }

    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo) as u32) as i32
    }

    fn unit(&mut self) -> f64 {
        self.next() as f64 / 4294967296.0
    }

    fn f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    fn chance(&mut self, n: u32) -> bool {
        self.below(n) == 0
    }
}

/// The tests that call this are `#[ignore]`d: a run that asks for them on a
/// host with no usable GPU fails instead of passing unchecked.
fn gpu() -> Arc<Gpu> {
    Gpu::new(&GpuOptions::default()).unwrap_or_else(|e| panic!("no host GPU: {e}"))
}

/// Both memories and the destination they share.
struct Pair {
    cpu: Vram,
    gpu: GpuRaster,
    dv: DestViews,
    /// The clip list (plane coordinates), the window origin.
    raster: RasterWords,
    ox: i32,
    oy: i32,
}

impl Pair {
    fn new(gpu: &Arc<Gpu>, rng: &mut Lcg, pixel_mode: &[u32; 4]) -> Pair {
        let mut cpu = Vram::new(VRAM_BYTES);
        for w in cpu.words_mut().iter_mut() {
            *w = (rng.next() as u64) << 32 | rng.next() as u64;
        }
        let mut g = GpuRaster::new(gpu, VRAM_BYTES, GRID, GRID).unwrap_or_else(|e| panic!("{e}"));
        g.write_words(0, cpu.words_mut());
        let dv = DestViews::of(pixel_mode, WID_BASE, WID_STRIDE);
        let mut raster = RasterWords::default();
        // Two clip rectangles over the 256-pixel plane, a window origin
        // near the corner.
        raster.clip_kept = 2;
        raster.clip_rect[0] = ClipRect { x0: rng.range(0, 40), y0: rng.range(0, 40), x1: rng.range(120, 255), y1: rng.range(120, 255) };
        raster.clip_rect[1] = ClipRect { x0: rng.range(100, 200), y0: rng.range(0, 100), x1: 255, y1: rng.range(150, 255) };
        raster.origin = [rng.range(-30, 30), rng.range(-30, 30)];
        let (ox, oy) = (raster.origin[0], raster.origin[1]);
        Pair { cpu, gpu: g, dv, raster, ox, oy }
    }

    fn planes(&mut self) -> DestMem {
        DestMem::of(self.cpu.ptr(), &self.dv)
    }

    /// The depth plane at its far value in both memories. The random
    /// memory holds depths about half as far as the test triangles' (near
    /// 0.97), so under GL_LESS or GL_LEQUAL nothing at all was drawn.
    fn clear_depth(&mut self) {
        let planes = self.planes();
        for y in 0..GRID as i32 {
            for x in 0..GRID as i32 {
                planes.depth.store_word(x, y, ODSY_DEPTH24_MAX);
            }
        }
        self.gpu.write_words(0, self.cpu.words_mut());
    }

    /// A raster state over the destination: the clip list, no scissor
    /// beyond the plane, the pipeline's 32-bit depth, no pre-test.
    fn raster_state(&mut self, words: &RasterWords, depth_range: [f64; 2]) -> RasterState {
        let planes = self.planes();
        let mut st = RasterState::of(words, self.dv.colour[0].width, planes.did, depth_range, gl_cmd::DEPTH_SLACK, word_for_fraction, None, false);
        st.scissor = Some((-100_000, -100_000, 100_000, 100_000));
        st.depth32 = true;
        st
    }

    /// Both memories equal, or the first differences.
    fn check(&mut self, what: &str) {
        let got = self.gpu.read_words().unwrap_or_else(|e| panic!("{what}: {e}"));
        let want = self.cpu.words_mut();
        assert_eq!(got.len(), want.len());
        let mut diffs = 0;
        let mut msg = String::new();
        let (base, pitch, log2, _) = self.dv.colour[0].parts();
        let fields = [("colour0", self.dv.colour[0]), ("colour1", self.dv.colour[1]), ("depth", self.dv.depth), ("stencil", self.dv.stencil)];
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g != w {
                diffs += 1;
                if diffs <= 12 {
                    msg += &format!("\n  word {i:#x} (byte {:#x}): gpu {g:#018x} cpu {w:#018x}", i * 8);
                    // The pixel of the destination it falls in, and its fields.
                    let a = i as u64 * 8;
                    if a >= base {
                        let p = (a - base) >> log2;
                        let (tile, inner) = (p >> 8, p & 255);
                        let (x, y) = ((tile % pitch as u64) * 16 + (inner & 15), (tile / pitch as u64) * 16 + (inner >> 4));
                        msg += &format!(" = pixel ({x}, {y}) byte {}", a & ((1 << log2) - 1));
                        for (name, v) in &fields {
                            let (_, _, _, at) = v.parts();
                            msg += &format!(" {name}@{at}");
                        }
                    }
                }
            }
        }
        assert_eq!(diffs, 0, "{what}: {diffs} words differ{msg}");
    }
}

/// A random triangle in window subpixels around the plane, with depth
/// over w, its divisor and 1/w as a perspective draw has them, and
/// colours.
fn vertex(rng: &mut Lcg, cx: i32, cy: i32, size: i32, ox: i32, oy: i32) -> RasterVertex {
    let w = rng.f(0.5, 20.0);
    let inv_w = 1.0 / w;
    let fraction = rng.f(-0.05, 1.05);
    RasterVertex {
        x: ((cx + rng.range(-size, size + 1) - ox) as i64) * 256 + rng.range(0, 256) as i64,
        y: ((cy + rng.range(-size, size + 1) - oy) as i64) * 256 + rng.range(0, 256) as i64,
        z: fraction * inv_w,
        inv_w,
        depth_w: inv_w,
        rgba: [rng.unit(), rng.unit(), rng.unit(), rng.unit()],
        spec: [rng.unit(), rng.unit(), rng.unit()],
        tex: [rng.f(-2.0, 3.0), rng.f(-2.0, 3.0), rng.f(0.0, 1.0), 1.0],
        ..Default::default()
    }
}

fn tri_of(rng: &mut Lcg, st: &RasterState, smooth: bool, ox: i32, oy: i32, size: i32) -> Option<Tri> {
    tri_textured(rng, st, smooth, ox, oy, size, None)
}

fn tri_textured(rng: &mut Lcg, st: &RasterState, smooth: bool, ox: i32, oy: i32, size: i32, regs: Option<&TexRegs>) -> Option<Tri> {
    let (cx, cy) = (rng.range(-40, 300), rng.range(-40, 300));
    let v = [vertex(rng, cx, cy, size, ox, oy), vertex(rng, cx, cy, size, ox, oy), vertex(rng, cx, cy, size, ox, oy)];
    let mut area = v[0].edge(&v[1], v[2].x, v[2].y);
    if area == 0 {
        return None;
    }
    let (front, culled) = st.facing(area);
    if culled {
        return None;
    }
    let mut v = v;
    if area < 0 {
        v.swap(1, 2);
        area = -area;
    }
    let provoking = rng.below(3) as usize;
    let flat = v[provoking].rgba;
    let tex = match regs {
        Some(r) => TriTex::setup(&v.map(|r| super::super::pipe::setup::texture_values(&r)), [r.wrap(0) == Wrap::Repeat, r.wrap(1) == Wrap::Repeat]),
        None => TriTex::default(),
    };
    Some(Tri {
        tri: Triangle::setup(v.map(|r| (r.x, r.y)), v.map(|r| r.z), v.map(|r| r.depth_w), area, front),
        live: std::ptr::null(),
        rgba12: [0, 1, 2].map(|i| v[i].rgba.map(fixed::from_f64)),
        flat12: flat.map(fixed::from_f64),
        spec12: [0, 1, 2].map(|i| v[i].spec.map(fixed::from_f64)),
        flat_spec12: v[provoking].spec.map(fixed::from_f64),
        smooth,
        tex,
        inv_w: v.map(|r| r.inv_w),
        clip: [[0.0; 6]; 3],
        planes: false,
    })
}

/// What a draw shades with, besides the triangle: the alpha test and the
/// polygon stipple.
#[derive(Clone, Copy, Default)]
struct Shade {
    alpha: Option<(Compare, u32)>,
    stipple: Option<[u32; 32]>,
    background: Option<[f64; 4]>,
}

impl Shade {
    fn words(&self) -> (u32, u32) {
        match self.alpha {
            Some((f, r)) => (mode::ALPHA_TEST, (compare_code(f) << 12) | r),
            None => (0, 0),
        }
    }
}

fn compare_code(c: Compare) -> u32 {
    (0..8).find(|&k| Compare::of(k) == c).unwrap()
}

/// Draw::shade_quad without fog or user planes: the stipple, the colour
/// and specular weights, the texture unit, then the shader stage.
fn cpu_shade(t: &Tri, q: &Quad, sh: &Shade, unit: Option<&Unit>, shader: &ShaderState) -> BackQuad {
    let mut cov = q.cov;
    let mut background = 0u8;
    for i in 0..4usize {
        if cov & (1 << i) == 0 {
            continue;
        }
        let (px, py) = (q.x + (i & 1) as i32, q.y + (i >> 1) as i32);
        if let Some(rows) = &sh.stipple {
            if rows[((py as u32) & 31) as usize] & (1u32 << ((px as u32) & 31)) == 0 {
                if sh.background.is_some() {
                    background |= 1 << i;
                } else {
                    cov &= !(1 << i);
                }
            }
        }
    }
    background &= cov;
    let mut c = t.flat12.map(|v| [v; 4]);
    let mut s = t.flat_spec12.map(|v| [v; 4]);
    if t.smooth {
        let iw = t.inv_w.map(|w| w as f32);
        let (w, kept) = interp::weights4(&q.l1, &q.l2, &iw);
        let lanes = |v: [i32; 3]| -> [i32; 4] { std::array::from_fn(|i| fixed::interpolate(&[w[0][i], w[1][i], w[2][i]], v)) };
        cov &= kept | background;
        c = std::array::from_fn(|k| lanes([t.rgba12[0][k], t.rgba12[1][k], t.rgba12[2][k]]));
        if shader.specular {
            s = std::array::from_fn(|k| lanes([t.spec12[0][k], t.spec12[1][k], t.spec12[2][k]]));
        }
    }
    if background != 0 {
        let bg = sh.background.expect("an opaque stipple").map(fixed::from_f64);
        for i in (0..4).filter(|i| background & (1 << i) != 0) {
            for k in 0..4 {
                c[k][i] = bg[k];
            }
            for sk in s.iter_mut() {
                sk[i] = 0;
            }
        }
    }
    let mut texel = [[0i32; 4]; 4];
    if let Some(u) = unit {
        u.sample_quad(&t.tex.quad(&q.l1, &q.l2), q.cov, &mut texel);
    }
    cov &= shader.shade12x4(unit.is_some(), &mut c, &s, &texel, |_| 0.0);
    let mut out = BackQuad { cov, ..BackQuad::default() };
    for i in (0..4).filter(|i| cov & (1 << i) != 0) {
        out.zq[i] = q.zq[i];
        out.rgba[i] = std::array::from_fn(|k| colour12(c[k][i]) as u16);
    }
    out
}

/// One draw's state on both sides.
struct DrawState {
    words: BackendWords,
    index: bool,
    to_aux: bool,
    depth_range: [f64; 2],
    steps: i64,
    shade: Shade,
    /// The draw's texture registers and their latch, when it textures.
    tex: Option<Box<(TexRegs, Latch)>>,
    specular: bool,
}

impl DrawState {
    fn plain() -> DrawState {
        DrawState {
            words: BackendWords { write_mask: write::DEPTH | (0xf << write::COLOUR_SHIFT) | (0xff << write::STENCIL_SHIFT), draw_mask: 1, ..Default::default() },
            index: false,
            to_aux: false,
            depth_range: [0.0, 1.0],
            steps: 0,
            shade: Shade::default(),
            tex: None,
            specular: false,
        }
    }

    fn depth(mut self, func: u32) -> DrawState {
        self.words.mode |= mode::DEPTH_TEST;
        self.words.depth_func = func;
        self
    }
}

/// The codes of the blend factors the board knows.
const FACTOR_CODES: [u32; 13] = [0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x40];

fn random_state(rng: &mut Lcg) -> DrawState {
    let mut d = DrawState::plain();
    if rng.chance(2) {
        d = d.depth(rng.below(8));
        if rng.chance(4) {
            d.words.write_mask &= !write::DEPTH;
        }
    }
    if rng.chance(3) {
        d.words.mode |= mode::BLEND;
        let (src, dst) = (FACTOR_CODES[rng.below(13) as usize], FACTOR_CODES[rng.below(12) as usize]);
        let eq = [0u32, 1, 2, 6, 7][rng.below(5) as usize];
        d.words.blend = [(eq << 13) | (src << 6) | dst, rng.next() & 0x00ff_ffff, rng.next() & 0x00ff_ffff];
    } else if rng.chance(4) {
        d.words.mode |= mode::COLOR_LOGIC_OP;
        d.words.logic_op = rng.below(16);
    }
    if rng.chance(4) {
        d.words.write_mask = (d.words.write_mask & !(0xf << write::COLOUR_SHIFT)) | (rng.below(16) << write::COLOUR_SHIFT);
    }
    if rng.chance(3) {
        d.words.mode |= mode::STENCIL_TEST;
        d.words.stencil = (rng.below(8) << 25) | (rng.below(6) << 22) | (rng.below(6) << 19) | (rng.below(6) << 16) | ((rng.next() & 0xff) << 8) | (rng.next() & 0xff);
        if rng.chance(2) {
            d.words.write_mask = (d.words.write_mask & !(0xff << write::STENCIL_SHIFT)) | ((rng.next() & 0xff) << write::STENCIL_SHIFT);
        }
    }
    d.words.draw_mask = [1, 4, 5][rng.below(3) as usize];
    if rng.chance(3) {
        d.depth_range = [rng.f(-0.1, 0.4), rng.f(0.5, 1.1)];
    }
    if rng.chance(3) {
        d.steps = [0, 1, -1, 1000, -1000, 1 << 24, -(1 << 24), 1 << 30][rng.below(8) as usize];
    }
    if rng.chance(3) {
        d.shade.alpha = Some((Compare::of(rng.below(8)), rng.below(4096)));
    }
    d
}

/// Draw `tris` under `d` on both sides.
fn draw(p: &mut Pair, d: &DrawState, smooth: bool, tris: &[Tri]) {
    let planes = p.planes();
    let mut st = BackendState::of(&d.words, planes.stencil.v.kind != super::super::view::Kind::None, !d.index);
    if d.to_aux {
        st.targets = aux_targets(st.targets, d.index);
    }
    st.store = st.select(&planes);
    let raster_words = RasterWords { mode: if smooth { 0 } else { mode::FLAT_SHADE }, ..p.raster };
    let mut rst = p.raster_state(&raster_words, d.depth_range);
    rst.offset_fill = true;
    rst.offset_steps = d.steps;
    let slot = Slot::new(&d.words, &planes, d.index, d.to_aux, &raster_words, d.shade.stipple.as_ref(), d.shade.background).unwrap_or_else(|e| panic!("{e:?}"));
    let (mode_word, alpha_word) = d.shade.words();
    let (ox, oy) = (p.ox, p.oy);
    let default_regs = TexRegs::default();
    let regs = d.tex.as_ref().map_or(&default_regs, |t| &t.0);
    let shader = ShaderState::of(&ShaderWords { mode: mode_word, alpha: alpha_word, ..Default::default() }, regs);
    let mem = p.cpu.ptr();
    let unit = d.tex.as_ref().map(|t| Unit { regs: &t.0, latch: &t.1, mem });
    let tex_slot = d.tex.as_ref().map(|t| TexSlot::new(&t.0, &t.1));
    for t in tris {
        // The CPU walk and store.
        triangle_quads(&rst, &t.tri, None, |q| {
            let mut out = cpu_shade(t, q, &d.shade, unit.as_ref(), &shader);
            out.x = ox.wrapping_add(q.x);
            out.y = oy.wrapping_add(q.y);
            if out.cov != 0 {
                (st.store)(&st, &planes, &out);
            }
        });
        // The GPU record.
        if let Some((x0, y0, x1, y1)) = walk_bounds(&rst, &t.tri, None) {
            p.gpu.push_tri(t, ox, oy, (x0 + ox, y0 + oy, x1 + ox, y1 + oy), d.steps, d.depth_range, gl_cmd::DEPTH_SLACK, mode_word, alpha_word, &slot, tex_slot.as_ref(), shader.specular);
        }
    }
}

/// Random texture registers: a 1D, 2D or 3D target of a few levels in
/// memory away from the destination, any wrap, filter, class and
/// environment, sometimes a border, a detail image, LOD limits, clamps,
/// a scale and bias or a colour table.
fn random_texture(rng: &mut Lcg, specular: bool) -> (TexRegs, Latch) {
    let mut r = TexRegs::default();
    let target = [tex::MODE_2D, tex::MODE_2D, tex::MODE_2D, tex::MODE_1D, tex::MODE_3D][rng.below(5) as usize];
    r.path = target | if specular { tex::MODE_SEPARATE_SPECULAR } else { 0 };
    let (ls, lt, lr) = (rng.range(1, 7) as u32, rng.range(1, 7) as u32, if target == tex::MODE_3D { rng.range(1, 4) as u32 } else { 0 });
    let detail = rng.chance(5);
    let k = rng.range(1, 4) as u32;
    r.sampler[0] = (ls << 10) | (lt << 5) | lr;
    if detail {
        r.sampler[0] |= (k << 25) | ((ls + k).min(31) << 20) | ((lt + k).min(31) << 15) | if rng.chance(2) { 0x8000_0000 } else { 0 };
    }
    let min = rng.below(6);
    let mag = if detail { 2 + rng.below(3) } else { rng.below(2) };
    let class = 1 + rng.below(6);
    let table = rng.chance(6);
    r.sampler[4] = (class << 23) | (if table { [20, 24][rng.below(2) as usize] } else { 0 } << 18) | (rng.below(4) << 12) | (rng.below(4) << 10) | (rng.below(4) << 8) | (min << 5) | (mag << 1) | table as u32;
    for a in 0..3 {
        r.sampler[1 + a] = if rng.chance(3) { rng.below(1 << 7) } else { 0x1ffff };
    }
    r.sampler[5] = rng.next() & 0x00ff_ffff;
    r.sampler[6] = (rng.next() & 0x00ff_ffff) | if rng.chance(4) { 1 << 24 } else { 0 };
    r.env[0] = 1 + rng.below(5);
    r.env[1] = rng.next() & 0x00ff_ffff;
    r.env[2] = rng.next() & 0x00ff_ffff;
    // LOD: a small bias, the limits around the chain.
    r.lod[0] = (rng.range(-512, 512) as u32) & 0x3fff;
    r.lod[1] = (rng.range(-512, 512) as u32) & 0x3fff;
    r.lod[2] = (rng.range(-15 << 8, 1 << 8) as u32) & 0x3fff;
    r.lod[3] = (rng.range(0, 15 << 8) as u32) & 0x3fff;
    // Levels: entries 0.. in two memory regions the destination never
    // reaches, a storage from the table, the pitch mostly automatic.
    let formats = [0u32, 11, 12, 13, 16, 17, 18, 32];
    let format = formats[rng.below(8) as usize];
    for i in 0..super::super::pipe::texture::LEVEL_ENTRIES as u32 {
        let base = [0x18000u32, 0x30000][rng.below(2) as usize] + rng.below(0x6000);
        let pitch = if rng.chance(4) { rng.range(1, 6) as u32 } else { 0 };
        let e = (i as usize) * 3;
        r.levels[e] = base;
        r.levels[e + 1] = (format << 20) | pitch | if target == tex::MODE_3D { rng.below(4) << 30 } else { 0 };
        r.levels[e + 2] = if target == tex::MODE_3D { rng.below(3) } else { 0 };
    }
    r.level0 = rng.below(4);
    if rng.chance(4) {
        for c in 0..4 {
            r.scale_bias[c * 2] = (rng.range(-8192, 8192) as u32) & 0xffff;
            r.scale_bias[c * 2 + 1] = (rng.range(-4096, 4096) as u32) & 0xffff;
        }
    }
    for t in 0..2 {
        for i in 0..256 {
            r.colour_table[t][i] = rng.next() & 0x00ff_ffff;
        }
    }
    for i in 0..256 {
        r.detail[i] = rng.next() & 0x00ff_f000;
    }
    let l = Latch::of(&r);
    (r, l)
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn textured_triangles_sample_as_the_cpu_unit() {
    let gpu = gpu();
    let mut rng = Lcg(20);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    for round in 0..40 {
        let mut d = DrawState::plain().depth(7);
        if round % 3 == 0 {
            d = random_state(&mut rng);
        }
        d.specular = rng.chance(3);
        let (regs, latch) = random_texture(&mut rng, d.specular);
        d.tex = Some(Box::new((regs, latch)));
        let smooth = rng.chance(2);
        let st = p.raster_state(&p.raster.clone(), [0.0, 1.0]);
        let (ox, oy) = (p.ox, p.oy);
        let regs = &d.tex.as_ref().unwrap().0;
        let size = [12, 40, 150][rng.below(3) as usize];
        let t: Vec<Tri> = (0..15).filter_map(|_| tri_textured(&mut rng, &st, smooth, ox, oy, size, Some(regs))).collect();
        draw(&mut p, &d, smooth, &t);
    }
    p.check("textured triangles");
    let (_, prims) = p.gpu.counts();
    assert!(prims > 300, "{prims} textured records");
}


fn tris(rng: &mut Lcg, p: &mut Pair, smooth: bool, n: usize, size: i32) -> Vec<Tri> {
    let st = p.raster_state(&p.raster.clone(), [0.0, 1.0]);
    let (ox, oy) = (p.ox, p.oy);
    (0..n).filter_map(|_| tri_of(rng, &st, smooth, ox, oy, size)).collect()
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn flat_triangles_store_as_the_cpu_pipeline() {
    let gpu = gpu();
    let mut rng = Lcg(1);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    p.clear_depth();
    let d = DrawState::plain().depth(1);
    for &size in &[6, 40, 300, 3000] {
        let t = tris(&mut rng, &mut p, false, 150, size);
        draw(&mut p, &d, false, &t);
    }
    p.check("flat triangles");
    let (flushes, prims) = p.gpu.counts();
    assert!(prims > 200 && flushes >= 1, "{prims} records in {flushes} batches");
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn smooth_triangles_and_the_alpha_test() {
    let gpu = gpu();
    let mut rng = Lcg(2);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    p.clear_depth();
    for round in 0..6 {
        let mut d = DrawState::plain().depth(3);
        if round % 2 == 1 {
            d.shade.alpha = Some((Compare::of(rng.below(8)), rng.below(4096)));
        }
        if round >= 4 {
            d.depth_range = [0.2, 0.8];
            d.steps = -500;
        }
        let t = tris(&mut rng, &mut p, true, 120, [30, 200][round % 2]);
        draw(&mut p, &d, true, &t);
    }
    p.check("smooth triangles");
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn blends_logic_ops_masks_and_stencil() {
    let gpu = gpu();
    let mut rng = Lcg(3);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    for _ in 0..40 {
        let d = random_state(&mut rng);
        let smooth = rng.chance(2);
        let size = [20, 120][rng.below(2) as usize];
        let t = tris(&mut rng, &mut p, smooth, 25, size);
        draw(&mut p, &d, smooth, &t);
    }
    p.check("random back-end states");
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn polygon_stipples_with_and_without_a_background() {
    let gpu = gpu();
    let mut rng = Lcg(4);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    p.clear_depth();
    for round in 0..4 {
        let mut d = DrawState::plain().depth(1);
        d.shade.stipple = Some(std::array::from_fn(|_| rng.next()));
        if round % 2 == 1 {
            d.shade.background = Some([rng.unit(), rng.unit(), rng.unit(), rng.unit()]);
        }
        let smooth = round >= 2;
        let t = tris(&mut rng, &mut p, smooth, 60, 80);
        draw(&mut p, &d, smooth, &t);
    }
    p.check("stipples");
}

/// A clear on both sides: the CPU as gl_raster_ops() runs it without a
/// ring, the GPU as the ring's rectangles.
fn clear(p: &mut Pair, c: &ClearDraw, x0: i32, y0: i32, x1: i32, y1: i32) {
    let slot = Slot::clear(&c.planes, &p.raster).unwrap_or_else(|e| panic!("{e:?}"));
    let n = p.raster.clip_kept as usize;
    for r in &p.raster.clip_rect[..n] {
        let (rx0, ry0, rx1, ry1) = (x0.max(r.x0), y0.max(r.y0), x1.min(r.x1), y1.min(r.y1));
        if rx0 > rx1 || ry0 > ry1 {
            continue;
        }
        for y in ry0..=ry1 {
            c.span(y, rx0, rx1);
        }
        p.gpu.push(Rec::rect(rx0, ry0, rx1, ry1, c), &slot);
    }
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn clears_and_cpu_shaded_quads_keep_the_order() {
    let gpu = gpu();
    let mut rng = Lcg(5);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    for round in 0..30 {
        let planes = p.planes();
        // A clear of some planes, under a colour mask and an id test now
        // and then.
        let full = rng.chance(2);
        let c = ClearDraw {
            planes,
            targets: [rng.chance(2) || round == 0, rng.chance(2)],
            colour: rng.chance(3).then(|| rng.next()),
            colour_mask: if full { u32::MAX } else { rng.next() },
            depth: rng.chance(2).then(|| rng.next() & ODSY_DEPTH24_MAX),
            stencil: rng.chance(2).then(|| ((rng.next() & 0xff) as u8, (rng.next() & 0xff) as u8)),
            id_value: (rng.next() & 0xff) as u8,
            id_mask: if rng.chance(3) { (rng.next() & 0xff) as u8 } else { 0 },
        };
        let (x0, y0) = (rng.range(-10, 200), rng.range(-10, 200));
        clear(&mut p, &c, x0, y0, x0 + rng.range(1, 200), y0 + rng.range(1, 200));
        // Triangles over it.
        let d = random_state(&mut rng);
        let t = tris(&mut rng, &mut p, false, 10, 60);
        draw(&mut p, &d, false, &t);
        // Quads shaded on the CPU, stored through the same state.
        let mut st = BackendState::of(&d.words, true, true);
        st.store = st.select(&planes);
        let raster_words = RasterWords { mode: mode::FLAT_SHADE, ..p.raster };
        let slot = Slot::new(&d.words, &planes, false, false, &raster_words, None, None).unwrap();
        for _ in 0..40 {
            let q = BackQuad {
                x: rng.range(0, 256) & !1,
                y: rng.range(0, 256) & !1,
                cov: (rng.next() & 0xf) as u8,
                zq: std::array::from_fn(|_| rng.next() & ODSY_DEPTH24_MAX),
                rgba: std::array::from_fn(|_| std::array::from_fn(|_| (rng.next() & 0xfff) as u16)),
                ..BackQuad::default()
            };
            (st.store)(&st, &planes, &q);
            p.gpu.push(Rec::frag(&q), &slot);
        }
    }
    p.check("clears, triangles and shaded quads");
}

#[test]
#[ignore = "requires a conformant host GPU"]
fn index_draws_share_words_between_pixels() {
    let gpu = gpu();
    let mut rng = Lcg(6);
    let mut p = Pair::new(&gpu, &mut rng, &OVERLAY_MODE);
    assert_eq!(p.dv.colour[1].kind, super::super::view::Kind::Byte);
    for round in 0..12 {
        let mut d = DrawState::plain();
        d.index = true;
        d.to_aux = round % 2 == 0;
        d.words.draw_mask = [4, 5][rng.below(2) as usize];
        d.words.write_mask = (d.words.write_mask & !write::INDEX_MASK) | if rng.chance(2) { write::INDEX_MASK } else { rng.next() & write::INDEX_MASK };
        if rng.chance(2) {
            d.words.mode |= mode::INDEX_LOGIC_OP;
            d.words.logic_op = rng.below(16);
        }
        let t = tris(&mut rng, &mut p, false, 40, 50);
        draw(&mut p, &d, false, &t);
        // A clear of the index plane under an id test.
        let planes = p.planes();
        let c = ClearDraw {
            planes,
            targets: [false, true],
            colour: Some(rng.next() & 0xff),
            colour_mask: if rng.chance(2) { u32::MAX } else { rng.next() & 0xff },
            depth: None,
            stencil: None,
            id_value: (rng.next() & 0xff) as u8,
            id_mask: if rng.chance(2) { 0x0f } else { 0 },
        };
        let (x0, y0) = (rng.range(0, 200), rng.range(0, 200));
        clear(&mut p, &c, x0, y0, x0 + rng.range(1, 60), y0 + rng.range(1, 60));
    }
    p.check("index draws into byte planes");
}

/// The depth word under depth ranges other than [0, 1] and with polygon
/// offsets, state by state, so a divergence names its range.
#[test]
#[ignore = "requires a conformant host GPU"]
fn depth_ranges_and_offsets() {
    let gpu = gpu();
    let mut rng = Lcg(7);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    let mut failed = vec![];
    for round in 0..24 {
        let mut d = DrawState::plain().depth(7);
        match round % 4 {
            0 => d.depth_range = [0.0, 1.0],
            1 => d.depth_range = [rng.f(-0.1, 0.4), rng.f(0.5, 1.1)],
            2 => d.depth_range = [rng.f(0.0, 0.4), 1.0],
            _ => d.depth_range = [0.0, rng.f(0.5, 1.1)],
        }
        if round >= 12 {
            d.steps = [0, 1, -1, 1000, -1000, 1 << 24, -(1 << 24), 1 << 30][rng.below(8) as usize];
        }
        let t = tris(&mut rng, &mut p, false, 30, 60);
        draw(&mut p, &d, false, &t);
        let got = p.gpu.read_words().unwrap();
        let n = got.iter().zip(p.cpu.words_mut().iter()).filter(|(g, w)| g != w).count();
        if n != 0 {
            failed.push(format!("round {round}: range {:?} steps {}: {n} words", d.depth_range, d.steps));
            // Resync so the next round stands alone.
            p.gpu.write_words(0, p.cpu.words_mut());
        }
    }
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}

/// The coded colour fields: 10:10:10:2 and 4:4:4:4 and 5:5:5:1 texels, the
/// 12-bit lanes, two-byte indices, under random states, clears and shaded
/// quads, each pixel mode on its own board.
#[test]
#[ignore = "requires a conformant host GPU"]
fn coded_colour_fields_convert_as_the_cpu_codecs() {
    use super::super::view::Kind;
    let gpu = gpu();
    // Status words: 4-byte 10:10:10:2 in an 8-byte window of a 16-byte
    // pixel with depth; 12-bit lanes (16-byte window); 2-byte 4:4:4:4;
    // 2-byte 5:5:5:1; 2-byte index.
    let modes: [(u32, &str); 5] = [(0x6960, "packed10"), (0x6cd0, "lanes"), (0x65d0, "4444"), (0x65e0, "5551"), (0x8_65d0, "index16")];
    for (round, (status, name)) in modes.iter().enumerate() {
        let mut rng = Lcg(10 + round as u64);
        let pixel_mode = [RGBA_MODE[0], RGBA_MODE[1], *status, 0];
        let mut p = Pair::new(&gpu, &mut rng, &pixel_mode);
        assert!(matches!(p.dv.colour[0].kind, Kind::Coded(_)), "{name}: {:?}", p.dv.colour[0].kind);
        let index = *name == "index16";
        for _ in 0..12 {
            let mut d = random_state(&mut rng);
            d.index = index;
            if index {
                d.words.write_mask = (d.words.write_mask & !write::INDEX_MASK) | if rng.chance(2) { write::INDEX_MASK } else { rng.next() & write::INDEX_MASK };
            }
            let smooth = rng.chance(2);
            let t = tris(&mut rng, &mut p, smooth, 20, 70);
            draw(&mut p, &d, smooth, &t);
            let planes = p.planes();
            let c = ClearDraw {
                planes,
                targets: [rng.chance(2), rng.chance(2)],
                colour: Some(rng.next()),
                colour_mask: if rng.chance(2) { u32::MAX } else { rng.next() },
                depth: rng.chance(2).then(|| rng.next() & ODSY_DEPTH24_MAX),
                stencil: None,
                id_value: 0,
                id_mask: 0,
            };
            let (x0, y0) = (rng.range(0, 200), rng.range(0, 200));
            clear(&mut p, &c, x0, y0, x0 + rng.range(1, 60), y0 + rng.range(1, 60));
        }
        p.check(name);
    }
}

/// Blit runs: rows of pixels through `blit_pixel`, into RGBA and byte
/// planes under masks, with and without the window's tests, interleaved
/// with triangles, against the CPU's stores in the same order.
#[test]
#[ignore = "requires a conformant host GPU"]
fn blit_runs_store_as_the_cpu_paint_path() {
    use super::super::view::Kind;
    let gpu = gpu();
    for (round, mode) in [RGBA_MODE, OVERLAY_MODE].iter().enumerate() {
        let mut rng = Lcg(30 + round as u64);
        let mut p = Pair::new(&gpu, &mut rng, mode);
        let byte = p.dv.colour[0].kind == Kind::Byte;
        for _ in 0..30 {
            // A triangle or two between the runs.
            let d = random_state(&mut rng);
            let t = tris(&mut rng, &mut p, false, 2, 60);
            draw(&mut p, &d, false, &t);
            // Rows of pixels: some consecutive, some broken, into one or
            // both buffers, a byte plane under a mask.
            let planes = p.planes();
            let words = RasterWords { clip_rect: p.raster.clip_rect, clip_kept: p.raster.clip_kept, id_compare: p.raster.id_compare, ..Default::default() };
            let slot = Slot::clear(&planes, &words).unwrap();
            let window = rng.chance(2);
            let buffers = 1 + rng.below(3);
            let mask = if byte { rng.next() & 0xff } else { u32::MAX };
            // A logical operation now and then, with the index or RGB
            // result masks of paint_integer.
            let op = rng.chance(3).then(|| rng.below(16) as u8);
            let (and, or) = if rng.chance(2) { (0xfff, 0) } else { (u32::MAX, 0xff) };
            let own = rng.chance(4);
            let y = rng.range(-5, 260);
            let mut x = rng.range(-5, 240);
            for _ in 0..rng.range(1, 80) {
                if rng.chance(10) {
                    x += rng.range(2, 9);
                }
                let word = rng.next();
                let owned = !window || (x >= 0 && y >= 0 && (x as u32) < p.dv.colour[0].width && p.raster.clip_rect[..p.raster.clip_kept as usize].iter().any(|r| x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1));
                for b in 0..2u32 {
                    if buffers & (1 << b) == 0 {
                        continue;
                    }
                    let rows = planes.colour[b as usize];
                    if owned {
                        let old = rows.load_word(x, y);
                        let val = match (op, old) {
                            (Some(o), Some(old)) => (super::super::logic::logic_op(o, word, old) & and) | or,
                            (Some(_), None) => u32::MAX, // nothing stored: the GPU returns too
                            (None, _) => word,
                        };
                        if op.is_some() && old.is_none() {
                        } else if byte {
                            if let Some(old) = rows.load1(x, y) {
                                rows.store1(x, y, (old & !(mask as u8)) | (val as u8 & mask as u8));
                            }
                        } else {
                            rows.store_word(x, y, val);
                        }
                    }
                    let view = if own { super::super::gpu_unit::BlitView::Own(Slot::own_view(&rows).unwrap()) } else { super::super::gpu_unit::BlitView::Slot(b) };
                    let key = super::super::gpu_unit::BlitKey { view, mask, window, op: op.map_or(0, |o| 0x100 | o as u32), and, or };
                    p.gpu.blit_pixel(&slot, &key, x, y, word);
                }
                x += 1;
            }
        }
        p.check(if byte { "blits into byte planes" } else { "blits" });
    }
}

/// The scan-out on the GPU against `Frame::compose`: random display
/// modes, lookup RAM, overlay modes and cursor over random memory, whole
/// frames compared.
#[test]
#[ignore = "requires a conformant host GPU"]
fn scanout_composes_as_the_cpu_frame() {
    use super::super::consts::{cursor, ODSY_GAMMA_GROUP};
    use super::super::frame::{pbj, Frame};
    use super::super::reader::DisplayRegs;
    use super::scanout::VproScanout;
    let gpu = gpu();
    let mut rng = Lcg(40);
    let mut p = Pair::new(&gpu, &mut rng, &RGBA_MODE);
    let (buf, words) = p.gpu.vram_buffer();
    let mut scan = VproScanout::new(&gpu, &buf, words).unwrap_or_else(|e| panic!("{e}"));
    for round in 0..24 {
        let mut regs = DisplayRegs::default();
        // The window-id and frame buffers of the test destination: 128 x 96
        // of 16-byte pixels (or 8 and 4 now and then), stride 16 tiles.
        let depth = [3u32, 3, 2, 1][rng.below(4) as usize];
        regs.pbj[pbj::OVERLAY_BASE] = WID_BASE;
        regs.pbj[pbj::PIXEL_BASE] = RGBA_MODE[0];
        regs.pbj[pbj::WIDTH] = [128, 97, 256][rng.below(3) as usize];
        regs.pbj[pbj::HEIGHT] = [96, 61, 128][rng.below(3) as usize];
        regs.pbj[pbj::STRIDE] = 16;
        regs.pbj[pbj::DEPTH] = depth;
        for id in 0..32 {
            // A mode word with naturally aligned colour fields, as the X
            // server programs them (a field across an 8-byte word is one
            // the model does not load); every fourth is RGB through the
            // gamma group.
            use super::super::frame::mode;
            // No 12-bit lanes in a 4-byte pixel: the model's guard lets a
            // 6-byte field across the pixel through.
            let (format, bytes) = [(mode::RGB4, 2u32), (mode::RGB5, 2), (mode::RGB8, 4), (mode::RGB10, 4), (mode::RGB12, 8), (0, 2), (1, 4)][rng.below(if depth >= 2 { 7 } else { 4 }) as usize];
            let units = if format == 1 { 2 } else if format == 0 { 1 } else { rng.below(8) };
            let at = |rng: &mut Lcg| (rng.below(16 / bytes) * bytes / 2) & 7;
            let group = if rng.chance(4) { ODSY_GAMMA_GROUP } else { rng.below(32) };
            let m = (units << mode::FIELD_UNITS_SHIFT) | (at(&mut rng) << mode::A_SHIFT) | (at(&mut rng) << mode::B_SHIFT) | (format << mode::FORMAT_SHIFT) | (group << 4) | (rng.next() & 0xf) | (rng.next() & 0xfc00_0000);
            regs.dispmode[id] = m;
        }
        for o in 0..8 {
            regs.overlay_mode[o] = rng.next();
        }
        for e in regs.lut.iter_mut() {
            *e = rng.next();
        }
        for c in regs.cursor.iter_mut() {
            *c = rng.next();
        }
        regs.cursor[cursor::POSITION as usize] = ((rng.range(-40, 140) as u32 & 0xffff) << 16) | (rng.range(-40, 110) as u32 & 0xffff);
        regs.cursor[cursor::CONTROL as usize] = match round % 4 {
            0 => 0,
            1 => cursor::CTL_ENABLE,
            2 => cursor::CTL_ENABLE | cursor::CTL_CROSSHAIR,
            _ => cursor::CTL_ENABLE | cursor::CTL_CHANNEL1,
        };
        let displayed = rng.next();
        let frame = Frame::snapshot(&regs, displayed).expect("a frame");
        let (w, h) = (frame.width as usize, frame.height as usize);
        let mut want = vec![0u32; w * h];
        frame.compose(p.cpu.ptr(), &regs, &mut want, w, h);
        let mut got = vec![0u32; w * h];
        scan.run(&frame, &regs, w as u32, h as u32, &mut got).unwrap_or_else(|e| panic!("{e}"));
        let diffs = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        if diffs != 0 {
            let (i, (a, b)) = want.iter().zip(&got).enumerate().find(|(_, (a, b))| a != b).unwrap();
            panic!("round {round}: {diffs} pixels differ, first at ({}, {}): cpu {a:#08x} gpu {b:#08x}", i % w, i / w);
        }
    }
}
