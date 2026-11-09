//! Random register configurations for the raster engine's equivalence
//! tests: the JIT against the interpreter (`rss_jit_tests`) and the GPU
//! unit against the interpreter (`wgpu::tests`). Each draws a primitive
//! kind's mode registers (target, format, masks, clip, the GL fragment and
//! texture state) and its data (colours, geometry, planes).

use super::rss::{self, reg};
use super::te1::reg as te;

/// A primitive kind the generators make registers for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prim {
    Fill,
    Line,
    Stipple,
    Xfer,
    Tri,
    GlLine,
}

impl Prim {
    pub fn is_gl(self) -> bool {
        matches!(self, Prim::Tri | Prim::GlLine)
    }
}

pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }

    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }

    pub fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }

    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + self.below((hi - lo + 1) as u64) as i64
    }

    pub fn f(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * ((self.next() >> 11) as f64 / (1u64 << 53) as f64)
    }

    pub fn pick<T: Copy>(&mut self, v: &[T]) -> T {
        v[self.below(v.len() as u64) as usize]
    }
}


/// One candidate: register writes, in order.
pub type Regs = Vec<(u32, u32)>;

pub fn hilo(regs: &mut Regs, r: u32, v: i64) {
    regs.push((r, (v as u64 >> 32) as u32));
    regs.push((r + 1, v as u32));
}

/// A float position as the RE takes it: biased by 49151.5.
pub fn pos(v: f64) -> u32 {
    ((v + 49151.5) as f32).to_bits()
}

/// The registers every primitive reads: target, format, logic op, clip.
pub fn common_regs(rng: &mut Rng, regs: &mut Regs, gl: bool) {
    let a_page = rng.pick(&[0x40u32, 0x50]);
    let mut b_page = rng.pick(&[0u32, a_page, 0x80, 0x90]);
    let field = match rng.below(if gl { 9 } else { 10 }) {
        0 => 0,
        1 | 2 => 1,
        3 => 2,
        4 => {
            // Both buffers, distinct.
            if b_page == 0 || b_page == a_page {
                b_page = 0x80;
            }
            3
        }
        5 => 3,
        6 | 7 => 0x40 | rng.below(16) as u32,
        8 => rng.pick(&[4u32, 0x10, 0x20, 0x3F]),
        _ => 0x50,
    };
    let drb = if field & 0x70 == 0x40 { 0xC0 } else { a_page | b_page << 10 };
    let mut pp1 = rng.u32() & !(0x7F << 14) & !(1 << 2) & !(0xF << 26);
    pp1 |= field << 14;
    if rng.chance(0.4) {
        pp1 |= 1 << 2 | (rng.below(16) as u32) << 26;
    }
    regs.push((reg::PP1FILLMODE, pp1));
    regs.push((reg::DRBPOINTERS, drb));
    // Tiles per row, and sometimes two raster engines (Maximum Impact).
    regs.push((reg::DRBSIZE, (rng.chance(0.3) as u32) << 6 | (rng.range(1, 3) as u32) << 2 | rng.range(1, 2) as u32));
    let mask = |rng: &mut Rng| {
        let r = rng.u32();
        rng.pick(&[0xFF_FFFFu32, u32::MAX, 0xFFF, 0xFF, 0, r])
    };
    regs.push((reg::COLORMASKLSBSA, mask(rng)));
    regs.push((reg::COLORMASKLSBSB, mask(rng)));
    regs.push((reg::COLORMASKMSBS, { let opts = [0xFFu32, 0xF0, 0x70, rng.u32()]; rng.pick(&opts) }));
    let mut win = rng.u32() & 0x30F;
    if rng.chance(0.4) {
        win |= (rng.range(1, 15) as u32) << 4;
    }
    win |= (rng.below(4) as u32) << 10;
    regs.push((reg::PP1WINMODE, win));
    let mut mode = 0;
    let n = rng.pick(&[0, 0, 1, 1, 2, 3, 4]);
    let mut order = [0u32, 1, 2, 3];
    for i in 0..4 {
        let j = rng.below(4) as usize;
        order.swap(i, j);
    }
    for (i, &m) in order.iter().enumerate() {
        if i < n {
            mode |= 1 << m;
        }
        if rng.chance(0.5) {
            mode |= 0x10 << m;
        }
        let (x0, y0) = (rng.range(-20, 300) as i32, rng.range(-20, 120) as i32);
        let (x1, y1) = (x0 + rng.range(0, 250) as i32, y0 + rng.range(0, 100) as i32);
        regs.push((reg::CLIP_X + 2 * m, (x0 as u32 & 0xFFFF) << 16 | (x1 as u32 & 0xFFFF)));
        regs.push((reg::CLIP_Y + 2 * m, (y0 as u32 & 0xFFFF) << 16 | (y1 as u32 & 0xFFFF)));
    }
    regs.push((reg::CLIP_MODE, mode));
    let yflip = rng.chance(0.4);
    let ox = rng.range(-30, 260) as i32;
    let oy = if yflip { rng.range(20, 200) } else { rng.range(-30, 100) } as i32;
    regs.push((reg::XYWIN, (oy as u32 & 0xFFFF) << 16 | (ox as u32 & 0xFFFF)));
    let config = (rng.u32() & !rss::CONFIG_YFLIP) | if yflip { rss::CONFIG_YFLIP } else { 0 };
    regs.push((reg::CONFIG, config));
}

/// Colours, whichever the primitive uses (data: not in the key).
pub fn colour_regs(rng: &mut Rng, regs: &mut Regs) {
    for r in [reg::FILL_COLOR_R, reg::FILL_COLOR_G, reg::FILL_COLOR_B, reg::FILL_COLOR_B + 1, reg::RED, reg::PACKEDCOLOR, reg::BG_COLOR, reg::BG_COLOR_BLUE] {
        regs.push((r, rng.u32()));
    }
}

/// The GL per-fragment and texture registers.
pub fn gl_regs(rng: &mut Rng, regs: &mut Regs, tex_bias: f64) {
    let te = |b: bool, v: u32| if b { v | rss::TEST_ENABLE } else { v & !rss::TEST_ENABLE };
    regs.push((reg::AFUNCMODE, te(rng.chance(0.5), rng.u32() & 0xFFFF)));
    let ops = |rng: &mut Rng| (rng.below(8) as u32) << 4 | (rng.below(8) as u32) << 8 | (rng.below(8) as u32) << 12;
    regs.push((reg::STENCILMODE, te(rng.chance(0.5), rng.below(8) as u32 | ops(rng) | (rng.u32() & 0xFF) << 16)));
    regs.push((reg::STENCILMASK, { let opts = [0xFFFFu32, rng.u32() & 0xFFFF, 0xFF0F]; rng.pick(&opts) }));
    let zmask = { let opts = [0xFF_FFFFu32, rng.u32() & 0xFF_FFFF, 0]; rng.pick(&opts) };
    regs.push((reg::ZMODE, te(rng.chance(0.6), rng.below(8) as u32 | zmask << 4)));
    let blend = (rng.below(16) as u32)
        | (rng.below(16) as u32) << 4
        | (rng.below(6) as u32) << rss::BLEND_EQ_SHIFT
        | if rng.chance(0.5) { rss_blend_enable() } else { 0 };
    regs.push((reg::BLENDFACTOR, blend));
    regs.push((reg::BLENDCOLOR_RG, rng.u32() & 0x0FFF_0FFF));
    regs.push((reg::BLENDCOLOR_BA, rng.u32() & 0x0FFF_0FFF));
    let tex_on = rng.chance(tex_bias);
    let m1 = (rng.u32() & 0xFFE) | tex_on as u32;
    regs.push((te::TEXMODE1, m1));
    // Detail textures (bit 9) on a quarter of them: the JIT declines them.
    let detail = rng.chance(0.25);
    regs.push((te::TEXMODE2, rng.u32() & 0xFFDFF | (detail as u32) << 9));
    let dlevel = (rng.below(5) as u32 + 2) << 16;
    regs.push((te::TXSIZE, rng.range(0, 9) as u32 | (rng.range(0, 9) as u32) << 4 | dlevel));
    let last = { let opts = [0u32, 15, 31, rng.below(32) as u32]; rng.pick(&opts) };
    regs.push((te::TXLOD, { let opts = [0u32, 1, 3, 7, 15, rng.u32() & 0xF]; rng.pick(&opts) } | last << 11));
    regs.push((te::TXDETAIL, rng.u32() & 0xFF));
    regs.push((te::TXADDR, 0));
    for _ in 0..32 {
        regs.push((te::DETAILSCALE, rng.u32() & 0x1F));
    }
    regs.push((te::TXBCOLOR_RG, rng.u32()));
    regs.push((te::TXBCOLOR_BA, rng.u32()));
    regs.push((te::TXENV_RG, rng.u32()));
    regs.push((te::TXENV_B, rng.u32()));
    regs.push((te::TXADDR, 0));
    for _ in 0..16 {
        regs.push((te::TXMIPMAP, rng.u32() & 0x1FF));
    }
    regs.push((te::TXADDR, 0));
    for _ in 0..16 {
        regs.push((te::TXBORDER, rng.u32() & 0x1FF));
    }
    regs.push((reg::FOG_ON, rng.chance(0.35) as u32));
    regs.push((reg::FOG_RG, rng.u32()));
    regs.push((reg::FOG_B, rng.u32()));
    let one = rss_iter_one();
    hilo(regs, reg::FOG_F, (rng.f(-0.3, 1.3) * one) as i64);
    hilo(regs, reg::FOG_F + 2, (rng.f(-0.02, 0.02) * one) as i64);
    hilo(regs, reg::FOG_F + 4, (rng.f(-0.02, 0.02) * one) as i64);
    for j in 0..32 {
        regs.push((reg::INDIRECT_ADDR, rss::POLY_STIPPLE_RAM + j));
        regs.push((reg::INDIRECT_DATA, if rng.chance(0.2) { u32::MAX } else { rng.u32() }));
    }
}

pub fn rss_blend_enable() -> u32 {
    1 << 8
}

pub fn rss_iter_one() -> f64 {
    super::te1::ITER_ONE
}

/// Colour, depth and texture planes (triangle step registers).
pub fn plane_regs(rng: &mut Rng, regs: &mut Regs) {
    for c in 0..4u32 {
        regs.push((0x05C + c, rng.range(-0x20_0000, 0x110_0000) as u32));
        regs.push((0x060 + 2 * c, rng.range(-0x6_0000, 0x6_0000) as u32));
        regs.push((0x061 + 2 * c, rng.range(-0x6_0000, 0x6_0000) as u32));
    }
    let z = |rng: &mut Rng, lo: f64, hi: f64| (rng.f(lo, hi) * 4096.0) as i64;
    hilo(regs, 0x068, z(rng, -1e5, 16_877_216.0));
    hilo(regs, 0x06A, z(rng, -3e4, 3e4));
    hilo(regs, 0x06C, z(rng, -3e4, 3e4));
    let one = rss_iter_one();
    let w = match rng.below(12) {
        0 => 0.0,
        1 => -rng.f(0.1, 2.0),
        2 => rng.f(1e-6, 1e-3),
        _ => rng.f(0.25, 4.0),
    };
    let scale = 10f64.powf(rng.f(-3.5, -0.3));
    let (s, t) = (rng.f(-1.5, 2.5), rng.f(-1.5, 2.5));
    hilo(regs, te::WI, (w * one) as i64);
    hilo(regs, te::SW, (s * w * one) as i64);
    hilo(regs, te::TW, (t * w * one) as i64);
    for r in [te::DSWE, te::DTWE, te::DSWX, te::DTWX, te::DSWY, te::DTWY] {
        hilo(regs, r, (rng.f(-1.0, 1.0) * scale * one) as i64);
    }
    for r in [te::DWIE, te::DWIX, te::DWIY] {
        hilo(regs, r, (rng.f(-1.0, 1.0) * scale * 0.2 * one) as i64);
    }
}

pub fn triangle_regs(rng: &mut Rng, regs: &mut Regs) -> u32 {
    let ymin = rng.f(-10.0, 120.0);
    let ymax = ymin + { let opts = [rng.f(0.2, 3.0), rng.f(1.0, 24.0), rng.f(10.0, 64.0)]; rng.pick(&opts) };
    let ymid = rng.f(ymin, ymax);
    let x = |rng: &mut Rng| rng.f(-20.0, 380.0);
    for (r, v) in [(0x000, x(rng)), (0x001, x(rng)), (0x002, x(rng)), (0x003, ymax), (0x004, ymid), (0x005, ymin)] {
        regs.push((r, pos(v)));
    }
    if rng.chance(0.03) {
        regs.push((0x005, pos(ymax + 1.0)));
    }
    for r in [0x006, 0x008, 0x00A] {
        hilo(regs, r, (rng.f(-3.0, 3.0) * (1u64 << 24) as f64) as i64);
    }
    plane_regs(rng, regs);
    if rng.chance(0.5) { rss::OP_AREA_LTOR } else { rss::OP_AREA_RTOL }
}

pub fn gl_line_regs(rng: &mut Rng, regs: &mut Regs) -> u32 {
    let (x0, y0) = (rng.f(-10.0, 380.0), rng.f(-10.0, 130.0));
    let len = { let opts = [rng.f(0.0, 3.0), rng.f(1.0, 60.0), rng.f(30.0, 250.0)]; rng.pick(&opts) };
    let a = rng.f(0.0, std::f64::consts::TAU);
    for (r, v) in [(0x00C, x0), (0x00D, y0), (0x00E, x0 + len * a.cos()), (0x00F, y0 + len * a.sin())] {
        regs.push((r, pos(v)));
    }
    plane_regs(rng, regs);
    regs.push((reg::GLINECONFIG, rng.pick(&[0u32, 0, 1, 2, 3])));
    regs.push((reg::LINE_STIPPLE, rng.u32()));
    regs.push((reg::LSCRL, rng.pick(&[0u32, 1, 3])));
    rss::OP_GL_LINE
}

pub fn block_regs(rng: &mut Rng, regs: &mut Regs, max: (i64, i64)) {
    let (xs, ys) = (rng.range(-20, 300) as i32, rng.range(-20, 110) as i32);
    let (w, h) = (rng.range(0, max.0) as i32, rng.range(0, max.1) as i32);
    let (xe, ye) = if rng.chance(0.25) { (xs - w, ys - h) } else { (xs + w, ys + h) };
    regs.push((reg::BLOCKXYSTARTI, (xs as u32 & 0xFFFF) << 16 | (ys as u32 & 0xFFFF)));
    regs.push((reg::BLOCKXYENDI, (xe as u32 & 0xFFFF) << 16 | (ye as u32 & 0xFFFF)));
}

/// Registers that decide `prim`'s key.
pub fn mode_regs(rng: &mut Rng, prim: Prim, tex_bias: f64) -> Regs {
    let mut regs = Regs::new();
    common_regs(rng, &mut regs, prim.is_gl());
    let fm = match prim {
        Prim::Fill => {
            let kind = rng.pick(&[0u32, 1, 6, 7]);
            (rng.u32() & !(7 << 22) & !(1 << 20) & !(1 << 3)) | kind << 22 | if rng.chance(0.5) { 1 << 20 } else { 0 }
        }
        Prim::Line => rng.u32(),
        Prim::Stipple => (rng.u32() & !(7 << 22) & !(1 << 20)) | 1 << 22 | 1 << 3,
        Prim::Xfer => {
            let (f, c) = rng.pick(&[(0u32, 0u32), (0, 1), (8, 8), (8, 10), (8, 0), (2, 3), (7, 0), (8, 1), (7, 1), (5, 5)]);
            // RGBA16 red first (RGBorder, bit 22), as the JIT keys it.
            let order = if (f, c) == (8, 1) { 1 << 22 } else { 0 };
            regs.push((reg::XFRMODE, (rng.u32() & 0x7FF00) | f << 4 | c | order));
            (rng.u32() & !(7 << 22) & !(1 << 20)) | 5 << 22
        }
        Prim::Tri => {
            gl_regs(rng, &mut regs, tex_bias);
            rng.u32() & !(1 << 2) | if rng.chance(0.3) { 1 << 2 } else { 0 }
        }
        Prim::GlLine => {
            gl_regs(rng, &mut regs, tex_bias);
            rng.u32() & !rss::FILL_LINE_STIPPLE | if rng.chance(0.4) { rss::FILL_LINE_STIPPLE } else { 0 }
        }
    };
    regs.push((reg::FILLMODE, fm));
    regs
}

/// Geometry and data for one `prim` (not in the key).
pub fn data_regs(rng: &mut Rng, prim: Prim) -> (Regs, u32) {
    let mut regs = Regs::new();
    colour_regs(rng, &mut regs);
    let op = match prim {
        Prim::Fill => {
            block_regs(rng, &mut regs, (60, 40));
            0x8
        }
        Prim::Line => {
            let p = |rng: &mut Rng| (rng.range(-20, 380) as u32 & 0xFFFF) << 16 | (rng.range(-20, 130) as u32 & 0xFFFF);
            regs.push((reg::LINE_START, p(rng)));
            regs.push((reg::LINE_END, p(rng)));
            regs.push((reg::LINE_STIPPLE, rng.u32()));
            0x5
        }
        Prim::Stipple => {
            block_regs(rng, &mut regs, (40, 8));
            0x8
        }
        Prim::Xfer => {
            block_regs(rng, &mut regs, (0, 6));
            regs.push((reg::XFRSIZE, rng.range(1, 48) as u32));
            0x8
        }
        Prim::Tri => triangle_regs(rng, &mut regs),
        Prim::GlLine => gl_line_regs(rng, &mut regs),
    };
    (regs, op)
}

