//! The raster JIT (`gr4-jit`): RSS/TE1 pixel pipelines compiled to native
//! code with Cranelift, modelled on the REX3 shader JIT (`ng1::rex3_jit`).
//!
//! The interpreter in `rss.rs` decides, pixel by pixel, from the registers,
//! what every stage of the pipeline does. A shader decides it once: the
//! registers that shape the pipeline are reduced to a `PipeKey` (primitive,
//! clipping, draw target and pixel format, logic op, and for GL primitives
//! the alpha, stencil and depth tests, blending, texture environment,
//! texture format, filtering and wrap modes, and fog), and the shader for
//! that key is straight-line code for exactly that pipeline. Everything the
//! pipeline only reads (colours, masks, reference values, page pointers,
//! plane coefficients, the sampler's pages) goes in a `RasterCtx`, built
//! per primitive.
//!
//! The shaders are bit-exact with the interpreter (`rss_jit_tests.rs`
//! proves it by sweeping keys and comparing whole boards): they evaluate
//! the same f64/f32 expressions in the same order, and take their setup —
//! edges, planes, the sampler — from the same Rust code (`Rss::tri_setup`,
//! `Rss::gl_line_setup`, `Te1::sampler`).
//!
//! Shaders are compiled on one background thread and published in a
//! process-wide map; a primitive whose shader is not ready yet runs in the
//! interpreter. `IRIS_GR4_JIT=off|sync|async` picks the mode (`sync`
//! compiles on first use and waits, for tests and comparisons).

pub mod compiler;
pub mod key;

pub use key::{Blend, Draw, PipeKey, Pix, Prim, Stencil, Tex, XFmt};

use super::pixmem::{self, Buffer, Kind};
use super::rss::{self, reg, GlLineSetup, Rss, TriSetup, Xfer};
use super::te1::{self, reg as te_reg, Sampler};
use std::collections::HashMap;
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Mutex, OnceLock, RwLock};

/// The shader ABI: one context, everything else reached through it.
pub type ShaderFn = unsafe extern "C" fn(*mut RasterCtx);

/// Interpreter only.
pub const MODE_OFF: u32 = 1;
/// Compile in the background; the interpreter draws until a shader is ready.
pub const MODE_ASYNC: u32 = 0;
/// Compile on first use and wait (tests, comparisons).
pub const MODE_SYNC: u32 = 2;

/// The mode a new board starts in: `IRIS_GR4_JIT`, else async (off in unit
/// tests, so the interpreter's tests stay deterministic unless asked).
pub fn default_mode() -> u32 {
    match std::env::var("IRIS_GR4_JIT").as_deref() {
        Ok("off") | Ok("0") => MODE_OFF,
        Ok("sync") => MODE_SYNC,
        Ok("async") | Ok("on") | Ok("1") => MODE_ASYNC,
        _ if cfg!(test) => MODE_OFF,
        _ => MODE_ASYNC,
    }
}

/// A primitive's last shader: valid while `epoch` matches and `f` is set.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Memo {
    epoch: u32,
    bits: u32,
    f: usize,
    packed: u64,
}

/// The JIT's state on one board: plain data, part of `Rss`.
///
/// What the registers make of the pipeline is cached: the board's context
/// (`Rss::jit_ctx`) keeps the register-derived fields, and `key2d` /
/// `keygl` (packed) the key fields they decide. Any register write other
/// than to a per-primitive data register (`is_data_reg`) bumps `epoch`,
/// which drops both and every primitive's memoised shader.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct JitState {
    pub mode: u32,
    epoch: u32,
    ok2d: u32,
    okgl: u32,
    key2d: u64,
    keygl: u64,
    memo: [Memo; 6],
    last: u64,
    /// Primitives drawn by a shader / by the interpreter while a shader was
    /// compiling / that no shader covers.
    pub hits: u64,
    pub misses: u64,
    pub declined: u64,
}

impl JitState {
    /// A register that shapes the pipeline changed.
    #[inline]
    pub fn invalidate(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        self.ok2d = 0;
        self.okgl = 0;
    }

    /// The packed key of the last shader run.
    pub fn last_key(&self) -> u64 {
        self.last
    }
}

/// Registers that only feed one primitive (positions, colours, iterators,
/// stipple bits, fill mode, transfer size): writing them leaves the cached
/// pipeline alone. Everything the cached part reads is outside this set.
pub fn is_data_reg(r: u32) -> bool {
    matches!(
        r,
        0x000..=0x00F
            | 0x013
            | 0x040 | 0x041 | 0x045..=0x047
            | 0x05B..=0x06D
            | 0x070 | 0x071
            | 0x080..=0x08B
            | 0x0C0..=0x0CB
            | 0x102
            | 0x110
            | 0x140 | 0x141
            | 0x146
            | 0x153
            | 0x159..=0x15C
            | 0x176..=0x179
            | 0x3F0..=0x3F5
    )
}

// ── context ──────────────────────────────────────────────────────────────────

/// A screen mask: inclusive ranges, and whether pixels inside pass.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ClipRect {
    pub xmin: i32,
    pub xmax: i32,
    pub ymin: i32,
    pub ymax: i32,
    pub keep_inside: u32,
    pub _pad: u32,
}

/// One buffer a primitive writes, as `Rss::put_in` would treat it.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Target {
    pub ptr: u32,
    pub xtiles: u32,
    /// Plane mask merged under (`Pix::Rgba8`, `Pix::Ci12`, overlay).
    pub mask: u32,
    /// What survives a logic op (12, 24 or 32 bits).
    pub lop_width: u32,
}

/// Register-derived context every primitive uses: origin, clipping, the
/// draw targets. Cached until a state register changes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CommonCtx {
    /// Window origin and the y direction (-1 with Y-flip).
    pub ox: i32,
    pub oy: i32,
    pub ysign: i32,
    pub cidmatch: u32,
    pub cidwmask: u32,
    /// Depth buffer tiles per row.
    pub zxtiles: u32,
    /// The window coordinates that land on screen: x in [vis[0], vis[1]),
    /// y in [vis[2], vis[3]). Triangles skip the rest (they draw nothing
    /// there and nothing else depends on visiting them).
    pub vis: [i32; 4],
    pub clip: [ClipRect; 4],
    pub tgt: [Target; 2],
}

/// Register-derived GL context: test references and masks, the polygon
/// stipple, the sampler, environment and fog colours. Cached like
/// `CommonCtx`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GlCtx {
    /// Alpha reference, 12.16.
    pub aref: i32,
    pub sref: u32,
    pub scmask: u32,
    pub swmask: u32,
    pub _pad0: u32,
    pub zmask: u64,
    pub poly: [u32; 32],
    pub smp: Sampler,
    /// Level 0's size in texels.
    pub smp_w: f64,
    pub smp_h: f64,
    /// Shared-page offsets by log2 of a level's larger side (0 above 16).
    pub small_off: [u32; 16],
    pub small_boff: [u32; 16],
    /// Texture environment colour and alpha; fog colour (12.16).
    pub env_c: [i32; 3],
    pub env_a: i32,
    pub fog_c: [i32; 3],
}

/// Everything a shader reads that is not in its key: one per board
/// (`Rss::jit_ctx`), plain data, valid zeroed.
#[repr(C)]
pub struct RasterCtx {
    pub words: *mut u64,
    pub cid: *mut u8,
    pub tram: *const u8,
    /// Transfer line bytes.
    pub src: *const u8,
    pub common: CommonCtx,
    pub gl: GlCtx,
    pub color: u32,
    pub bg: u32,
    /// Block: start, step, extent (fill, stipple, transfer).
    pub bxs: i32,
    pub bys: i32,
    pub bdx: i32,
    pub bdy: i32,
    pub bcols: i32,
    pub brows: i32,
    /// Character stipple chunk: bits (MSB first), count, and the block
    /// position, updated in place.
    pub sbits: u64,
    pub sn: u32,
    pub scol: i32,
    pub srow: i32,
    /// X line end points and stipple pattern.
    pub lx0: i32,
    pub ly0: i32,
    pub lx1: i32,
    pub ly1: i32,
    pub lpattern: u32,
    /// Transfer: pixels in this line, and the line.
    pub xwidth: u32,
    pub xline: u32,
    /// GL line stipple position, updated in place.
    pub stipple_pos: u32,
    pub tri: TriSetup,
    pub gll: GlLineSetup,
}

impl RasterCtx {
    fn set_pointers(rss: &mut Rss) {
        rss.jit_ctx.words = rss.mem.words.as_mut_ptr();
        rss.jit_ctx.cid = rss.cid.as_mut_ptr();
        rss.jit_ctx.tram = rss.te.tram_ptr();
    }
}

// ── key and context from the registers ───────────────────────────────────────

/// What `Rss::put_in` does with a write to `b` (`back`: through
/// ColorMaskLSBsB).
fn target(rss: &Rss, b: Buffer, back: bool) -> Target {
    let pp1 = rss.reg(reg::PP1FILLMODE);
    let lsb = rss.reg(if back { reg::COLORMASKLSBSB } else { reg::COLORMASKLSBSA });
    let msbs = rss.reg(reg::COLORMASKMSBS);
    let wide = rss::rgb_pixtype(pp1) || lsb == 0xFF_FFFF || lsb == u32::MAX;
    let ptype = (pp1 >> 8) & 7;
    let mask = if b.kind == Kind::Overlay {
        if rss::draw_buffer(pp1) == 0x48 { msbs & 0xFF } else if msbs != 0 { 0xFF } else { 0 }
    } else if ptype == 2 {
        if lsb == u32::MAX { lsb } else { lsb & 0xFF_FFFF | (msbs & 0xFF) << 24 }
    } else if matches!(ptype, 4 | 6) {
        lsb & 0xFFF
    } else {
        u32::MAX
    };
    Target {
        ptr: b.ptr,
        xtiles: b.xtiles,
        mask,
        lop_width: if ptype == 2 { u32::MAX } else if wide { 0xFF_FFFF } else { 0xFFF },
    }
}

/// The key fields and context every primitive shares: clipping, the draw
/// target, the pixel format and logic op.
fn common(rss: &mut Rss) -> PipeKey {
    let mut c = std::mem::take(&mut rss.jit_ctx.common);
    let pp1 = rss.reg(reg::PP1FILLMODE);
    let field = rss::draw_buffer(pp1);
    let drbsize = rss.reg(reg::DRBSIZE);
    let b = rss.target();
    c.tgt = [Target::default(); 2];
    c.tgt[0] = target(rss, b, field == rss::DRAW_B);
    let draw = if field == rss::DRAW_CID {
        Draw::Cid
    } else if b.kind == Kind::Overlay {
        Draw::Overlay
    } else {
        match rss.second_buffer() {
            Some(p) if field == rss::DRAW_A_AND_B => {
                let b2 = Buffer::new(p, Kind::Wide, drbsize);
                if b2 != b {
                    c.tgt[1] = target(rss, b2, true);
                    Draw::Dual
                } else {
                    Draw::Single
                }
            }
            _ => Draw::Single,
        }
    };
    let ptype = (pp1 >> 8) & 7;
    let pix = match ptype {
        0 if pp1 & (1 << 13) == 0 => Pix::Rgb12,
        2 => Pix::Rgba8,
        4 | 6 => Pix::Ci12,
        _ => Pix::Plain,
    };
    let winmode = rss.reg(reg::PP1WINMODE);
    c.cidmatch = rss::cid_match(winmode);
    c.cidwmask = rss::cid_write_mask(winmode) as u32;
    let mode = rss.reg(reg::CLIP_MODE);
    let mut nclip = 0;
    c.clip = [ClipRect::default(); 4];
    for n in 0..4 {
        if mode & (1 << n) == 0 {
            continue;
        }
        let (mx, my) = (rss.reg(reg::CLIP_X + 2 * n), rss.reg(reg::CLIP_Y + 2 * n));
        c.clip[nclip] = ClipRect {
            xmin: rss::signed16(mx >> 16),
            xmax: rss::signed16(mx),
            ymin: rss::signed16(my >> 16),
            ymax: rss::signed16(my),
            keep_inside: (mode & (0x10 << n) != 0) as u32,
            _pad: 0,
        };
        nclip += 1;
    }
    let win = rss.reg(reg::XYWIN);
    c.ox = rss::signed16(win);
    c.oy = rss::signed16(win >> 16);
    c.ysign = if rss.reg(reg::CONFIG) & rss::CONFIG_YFLIP != 0 { -1 } else { 1 };
    let (ox, oy) = (c.ox as i64, c.oy as i64);
    let w = rss::WIDTH as i64;
    let (ylo, yhi) = if c.ysign > 0 { (-oy, w - oy) } else { (oy - (w - 1), oy + 1) };
    let cl = |v: i64| v.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
    c.vis = [cl(-ox), cl(w - ox), cl(ylo), cl(yhi)];
    c.zxtiles = pixmem::xtiles(drbsize).0;
    let key = PipeKey {
        cid_test: c.cidmatch != 0,
        nclip: nclip as u8,
        draw,
        pix,
        logic: (pp1 & rss::PP1_LOGIC_OP_ENABLE != 0).then_some(((pp1 >> 26) & 0xF) as u8),
        ..PipeKey::default()
    };
    rss.jit_ctx.common = c;
    key
}

/// The GL per-fragment state (tests, blending, texture, fog). None for
/// the clip-ID planes, which GL fragments never reach.
fn gl_common(rss: &mut Rss, mut k: PipeKey) -> PipeKey {
    let mut c = std::mem::take(&mut rss.jit_ctx.gl);
    k.rgb = rss.rgb_mode();
    let af = rss.reg(reg::AFUNCMODE);
    k.alpha = (af & rss::TEST_ENABLE != 0).then_some((af & 7) as u8);
    c.aref = (((af >> 4) & 0xFFF) << 16) as i32;
    let st = rss.reg(reg::STENCILMODE);
    k.stencil = (st & rss::TEST_ENABLE != 0).then_some(Stencil {
        func: (st & 7) as u8,
        fail: ((st >> 4) & 7) as u8,
        zfail: ((st >> 8) & 7) as u8,
        zpass: ((st >> 12) & 7) as u8,
    });
    let masks = rss.reg(reg::STENCILMASK);
    c.sref = (st >> 16) & 0xFF;
    c.scmask = masks & 0xFF;
    c.swmask = (masks >> 8) & 0xFF;
    let zm = rss.reg(reg::ZMODE);
    k.z = (zm & rss::ZMODE_TEST != 0).then_some((zm & 7) as u8);
    c.zmask = ((zm >> 4) & 0xFF_FFFF) as u64;
    let bf = rss.reg(reg::BLENDFACTOR);
    k.blend = (bf & rss::BLEND_ENABLE != 0).then_some(Blend { src: (bf & 0xF) as u8, dst: ((bf >> 4) & 0xF) as u8 });
    let m1 = rss.reg(te_reg::TEXMODE1);
    if m1 & te1::TEXMODE1_ENABLE != 0 {
        let m2 = rss.reg(te_reg::TEXMODE2);
        k.tex = Some(Tex {
            env: ((m1 >> 1) & 3) as u8,
            nc: ((m1 >> 3) & 3) as u8 + 1,
            slot: ((m1 >> 7) & 3) as u8,
            class: ((m1 >> 9) & 7) as u8,
            mag_linear: m2 & te1::TM2_MAG_LINEAR != 0,
            min_linear: m2 & te1::TM2_MIN_LINEAR != 0,
            mipmap: m2 & te1::TM2_MM_ENABLE != 0 && m2 & te1::TM2_MIPMAP != 0,
            mip_linear: m2 & te1::TM2_MIP_LINEAR != 0,
            clamp_s: m2 & te1::TM2_CLAMP_S != 0,
            clamp_t: m2 & te1::TM2_CLAMP_T != 0,
            gl_clamp: m2 & te1::TM2_GL_CLAMP != 0,
            no_border: m2 & te1::TM2_NO_BORDER != 0,
            depth: te1::depth_nibbles(m2 >> 2) as u8,
        });
        c.smp = rss.sampler();
        let (w, h) = c.smp.size();
        c.smp_w = w;
        c.smp_h = h;
        for l in 0..16 {
            let n = 1usize << l;
            if n <= 16 {
                c.small_off[l] = te1::small_offset(n) as u32;
                c.small_boff[l] = te1::small_border_offset(n) as u32;
            }
        }
        let e = super::fixed::field12;
        let (rg, b) = (rss.reg(te_reg::TXENV_RG), rss.reg(te_reg::TXENV_B));
        c.env_c = [e(rg), e(rg >> 12), e(b)];
        c.env_a = e(b >> 12);
    }
    k.fog = rss.reg(reg::FOG_ON) != 0;
    if k.fog {
        let f = super::fixed::field12;
        let (rg, b) = (rss.reg(reg::FOG_RG), rss.reg(reg::FOG_B));
        c.fog_c = [f(rg), f(rg >> 12), f(b)];
    }
    for j in 0..32 {
        c.poly[j] = rss.poly_stipple_row(j as u32);
    }
    rss.jit_ctx.gl = c;
    k
}

fn set_block(c: &mut RasterCtx, b: &rss::Block) {
    c.bxs = b.xs;
    c.bys = b.ys;
    c.bdx = b.dx();
    c.bdy = b.dy();
    c.bcols = b.cols();
    c.brows = b.rows();
    c.color = b.color;
}

/// A primitive and what it brings besides the registers.
pub(super) enum Args<'a> {
    Fill(&'a rss::Block),
    Line,
    Stipple(&'a rss::Stipple, u64, u32),
    Xfer(&'a Xfer, u32, &'a [u8]),
    Tri(&'a TriSetup),
    GlLine(&'a GlLineSetup),
}

impl Args<'_> {
    fn prim(&self) -> Prim {
        match self {
            Args::Fill(_) => Prim::Fill,
            Args::Line => Prim::Line,
            Args::Stipple(..) => Prim::Stipple,
            Args::Xfer(..) => Prim::Xfer,
            Args::Tri(_) => Prim::Tri,
            Args::GlLine(_) => Prim::GlLine,
        }
    }
}

const BIT_STIPPLE: u32 = 1;
const BIT_OPAQUE: u32 = 2;
const BIT_SKIP_LAST: u32 = 4;

/// Bring the board's context up to date for a primitive: the cached
/// register-derived parts if a state register changed, then the
/// primitive's own fields. Returns the key bits the primitive adds to the
/// cached key (`key`), or None when no shader covers it.
pub(super) fn prepare(rss: &mut Rss, a: &Args) -> Option<u32> {
    RasterCtx::set_pointers(rss);
    if rss.jit.ok2d == 0 {
        // A GL primitive key, to keep the GL fields when packed.
        let k = PipeKey { prim: Prim::Tri, ..common(rss) };
        rss.jit.key2d = k.pack();
        rss.jit.ok2d = 1;
    }
    let gl = a.prim().is_gl();
    if gl {
        let base = PipeKey::unpack(rss.jit.key2d);
        // GL fragments never reach the clip-ID planes.
        if base.draw == Draw::Cid {
            return None;
        }
        if rss.jit.okgl == 0 {
            rss.jit.keygl = gl_common(rss, base).pack();
            rss.jit.okgl = 1;
        }
    }
    let fm = rss.reg(reg::FILLMODE);
    let bits = match *a {
        Args::Fill(b) => {
            set_block(&mut rss.jit_ctx, b);
            0
        }
        Args::Line => {
            let (s, e) = (rss.reg(reg::LINE_START), rss.reg(reg::LINE_END));
            let color = rss.current_block().color;
            let pattern = rss.reg(reg::LINE_STIPPLE);
            let c = &mut rss.jit_ctx;
            c.lx0 = rss::signed16(s >> 16);
            c.ly0 = rss::signed16(s);
            c.lx1 = rss::signed16(e >> 16);
            c.ly1 = rss::signed16(e);
            c.color = color;
            c.lpattern = pattern;
            let stipple = fm & rss::FILL_LINE_STIPPLE != 0;
            let opaque = stipple && fm & rss::FILL_LINE_STIPPLE_OPAQUE != 0;
            if opaque {
                rss.jit_ctx.bg = rss.background();
            }
            let skip = fm & rss::FILL_LINE_SKIP_LAST != 0;
            stipple as u32 * BIT_STIPPLE | opaque as u32 * BIT_OPAQUE | skip as u32 * BIT_SKIP_LAST
        }
        Args::Stipple(s, bits, n) => {
            if s.opaque {
                rss.jit_ctx.bg = rss.background();
            }
            let c = &mut rss.jit_ctx;
            set_block(c, &s.block);
            c.sbits = bits;
            c.sn = n;
            c.scol = s.col;
            c.srow = s.row;
            s.opaque as u32 * BIT_OPAQUE
        }
        Args::Xfer(x, line, bytes) => {
            // A line ending in a partial pixel: the interpreter decodes that
            // pixel from fewer bytes.
            let bpp = x.bpp as usize;
            if bytes.len() % bpp != 0 && bytes.len().div_ceil(bpp) <= x.width as usize {
                return None;
            }
            let fmt = XFmt::of(x.format.0 << 4 | x.format.1);
            let code = key::XBPP.iter().position(|&b| b as u32 == x.bpp)? as u32;
            let c = &mut rss.jit_ctx;
            set_block(c, &x.block);
            c.src = bytes.as_ptr();
            c.xwidth = (x.width as usize).min(bytes.len() / bpp) as u32;
            c.xline = line;
            (fmt as u32) << 3 | code << 6
        }
        Args::Tri(t) => {
            rss.jit_ctx.tri = *t;
            (t.stipple != 0) as u32 * BIT_STIPPLE
        }
        Args::GlLine(l) => {
            rss.jit_ctx.stipple_pos = rss.gl_line_stipple_pos;
            rss.jit_ctx.gll = *l;
            (l.stipple != 0) as u32 * BIT_STIPPLE
        }
    };
    Some(bits)
}

/// The full key of a prepared primitive.
pub(super) fn key(st: &JitState, prim: Prim, bits: u32) -> PipeKey {
    let mut k = PipeKey::unpack(if prim.is_gl() { st.keygl } else { st.key2d });
    k.prim = prim;
    k.stipple = bits & BIT_STIPPLE != 0;
    k.opaque = bits & BIT_OPAQUE != 0;
    k.skip_last = bits & BIT_SKIP_LAST != 0;
    if prim == Prim::Xfer {
        k.xfmt = XFmt::ALL[(bits >> 3 & 7) as usize];
        k.xbpp = key::XBPP[(bits >> 6) as usize];
    }
    k
}

// ── dispatch ─────────────────────────────────────────────────────────────────

/// Prepare and run a primitive's shader. False when there is none (yet):
/// the caller draws with the interpreter.
pub(super) fn run(rss: &mut Rss, a: &Args) -> bool {
    if rss.jit.mode == MODE_OFF {
        return false;
    }
    let prim = a.prim();
    let Some(bits) = prepare(rss, a) else {
        rss.jit.declined += 1;
        return false;
    };
    let st = &mut rss.jit;
    let m = st.memo[prim as usize];
    let f = if m.f != 0 && m.epoch == st.epoch && m.bits == bits {
        st.last = m.packed;
        // SAFETY: memoised from a ShaderFn; shaders are never freed.
        unsafe { std::mem::transmute::<usize, ShaderFn>(m.f) }
    } else {
        let packed = key(st, prim, bits).pack();
        match store().get(packed, st.mode == MODE_SYNC) {
            Some(f) => {
                st.memo[prim as usize] = Memo { epoch: st.epoch, bits, f: f as usize, packed };
                st.last = packed;
                f
            }
            None => {
                st.misses += 1;
                return false;
            }
        }
    };
    // SAFETY: the shader reads and writes only through the context, whose
    // pointers `prepare` took from this board, held mutably here.
    unsafe { f(&mut rss.jit_ctx) };
    rss.jit.hits += 1;
    true
}

pub(super) fn triangle(rss: &mut Rss, t: &TriSetup) -> bool {
    run(rss, &Args::Tri(t))
}

pub(super) fn gl_line(rss: &mut Rss, l: &GlLineSetup) -> bool {
    if run(rss, &Args::GlLine(l)) {
        rss.gl_line_stipple_pos = rss.jit_ctx.stipple_pos;
        return true;
    }
    false
}

pub(super) fn fill(rss: &mut Rss, b: &rss::Block) -> bool {
    run(rss, &Args::Fill(b))
}

pub(super) fn line(rss: &mut Rss) -> bool {
    run(rss, &Args::Line)
}

/// A stipple chunk; on success the stipple's position is updated.
pub(super) fn stipple(rss: &mut Rss, s: &mut rss::Stipple, bits: u64, n: u32) -> bool {
    if run(rss, &Args::Stipple(s, bits, n)) {
        s.col = rss.jit_ctx.scol;
        s.row = rss.jit_ctx.srow;
        return true;
    }
    false
}

pub(super) fn xfer_line(rss: &mut Rss, x: &Xfer, line: u32, bytes: &[u8]) -> bool {
    run(rss, &Args::Xfer(x, line, bytes))
}

// ── shader store and compiler thread ─────────────────────────────────────────

enum Entry {
    Queued,
    Ready(ShaderFn, u32),
    Failed,
}

struct Request {
    key: u64,
    /// Signalled when the key is settled (sync requests).
    done: Option<SyncSender<()>>,
}

/// The process-wide shader map. Shaders depend on nothing but their key,
/// so every board shares them; the compiler thread and its code live for
/// the rest of the process.
pub struct Store {
    map: RwLock<HashMap<u64, Entry>>,
    tx: Mutex<Option<SyncSender<Request>>>,
}

pub fn store() -> &'static Store {
    static STORE: OnceLock<Store> = OnceLock::new();
    STORE.get_or_init(|| Store { map: RwLock::new(HashMap::new()), tx: Mutex::new(None) })
}

impl Store {
    fn sender(&self) -> SyncSender<Request> {
        let mut tx = self.tx.lock().unwrap();
        tx.get_or_insert_with(|| {
            let (tx, rx) = mpsc::sync_channel::<Request>(256);
            std::thread::Builder::new()
                .name("gr4-jit".into())
                .spawn(move || {
                    let mut comp = compiler::Compiler::new();
                    for req in rx {
                        let s = store();
                        let settled = matches!(s.map.read().unwrap().get(&req.key), Some(Entry::Ready(..) | Entry::Failed));
                        if !settled {
                            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| comp.compile(req.key)));
                            let entry = match r {
                                Ok(Some((f, bytes))) => Entry::Ready(f, bytes),
                                Ok(None) => Entry::Failed,
                                Err(_) => {
                                    eprintln!("GR4 JIT: compiler panicked on {}", PipeKey::unpack(req.key));
                                    comp = compiler::Compiler::new();
                                    Entry::Failed
                                }
                            };
                            s.map.write().unwrap().insert(req.key, entry);
                        }
                        if let Some(done) = req.done {
                            let _ = done.send(());
                        }
                    }
                })
                .expect("spawn gr4-jit thread");
            tx
        })
        .clone()
    }

    /// The shader for a packed key. Async: None until it is compiled (the
    /// first miss queues it). Sync: compile now if need be and wait.
    pub fn get(&self, key: u64, sync: bool) -> Option<ShaderFn> {
        match self.map.read().unwrap().get(&key) {
            Some(Entry::Ready(f, _)) => return Some(*f),
            Some(Entry::Failed) => return None,
            Some(Entry::Queued) if !sync => return None,
            _ => {}
        }
        if sync {
            let (dtx, drx) = mpsc::sync_channel(1);
            self.sender().send(Request { key, done: Some(dtx) }).ok()?;
            drx.recv().ok()?;
            return match self.map.read().unwrap().get(&key) {
                Some(Entry::Ready(f, _)) => Some(*f),
                _ => None,
            };
        }
        {
            let mut map = self.map.write().unwrap();
            if map.contains_key(&key) {
                return None;
            }
            map.insert(key, Entry::Queued);
        }
        // Never block the RSS thread: a full queue drops the request, and
        // a later primitive asks again.
        if self.sender().try_send(Request { key, done: None }).is_err() {
            self.map.write().unwrap().remove(&key);
        }
        None
    }

    /// (compiled, queued, failed, code bytes) for `mgras jit`.
    pub fn summary(&self) -> (usize, usize, usize, u64) {
        let map = self.map.read().unwrap();
        let mut s = (0, 0, 0, 0u64);
        for e in map.values() {
            match e {
                Entry::Ready(_, b) => {
                    s.0 += 1;
                    s.3 += *b as u64;
                }
                Entry::Queued => s.1 += 1,
                Entry::Failed => s.2 += 1,
            }
        }
        s
    }

    /// Compiled keys, for listings.
    pub fn compiled(&self) -> Vec<(u64, u32)> {
        let map = self.map.read().unwrap();
        let mut v: Vec<(u64, u32)> = map
            .iter()
            .filter_map(|(k, e)| match e {
                Entry::Ready(_, b) => Some((*k, *b)),
                _ => None,
            })
            .collect();
        v.sort_unstable();
        v
    }
}
