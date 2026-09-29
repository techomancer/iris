//! OpenGL on GR2: HLE of the GE7 geometry microcode.
//!
//! libglcore (EXPRESS) writes GL state and vertices straight into the HQ2
//! FIFO. The FIFO word index is a microcode token (bits 8:0) plus HQ2 input
//! modifiers: USEV (9), LOADV (10), conversion (13:11) and ITOF (14); see
//! HQ2.h "OpenGL token addressing". This module is the GE side: it keeps the
//! transform state, assembles primitives by the vertex routine selected with
//! LOADV, transforms, maps through the viewport and window origin (0x1E5) and
//! rasterizes into RE3 spans.
//!
//! Blending: the real board blends in the GE7 microcode (RE3 has no blend
//! unit); here the GE hands the blend state and a per-pixel alpha plane to
//! an emulator-only RE3 extension (re3::RE3_OP_BLEND / RE3_OP_ALPHA).
//! Alpha test: there is none in the hardware and none here on purpose.
//! libglcore rasterizes alpha-tested primitives on the CPU and sends the
//! surviving fragments (T_FRAGMENT). See GR2.h "BLENDING AND ALPHA TEST".
//!
//! Current scope: RGB windows, no lighting (colours from glColor), no
//! clipping beyond the window/scissor rectangle (vertices behind the eye
//! reject the whole primitive), lines and points without Z.

use super::re3;
use super::{Hq2Engine, Re3Sink, SCREEN_H};

#[path = "gl_light.rs"]
mod light;

// Tokens (FIFO index bits 8:0).
pub const T_VERTEX: u32 = 0x063;
pub const T_COLOR: u32 = 0x182;
pub const T_NORMAL: u32 = 0x10d;
pub const T_MODELVIEW: u32 = 0x037;
pub const T_PROJECTION: u32 = 0x038;
pub const T_NORMAL_MATRIX: u32 = 0x039;
pub const T_TEXTURE_MATRIX: u32 = 0x03a;
pub const T_VIEWPORT: u32 = 0x03b;
pub const T_SCISSOR: u32 = 0x03c;
pub const T_SHADE_MODEL: u32 = 0x013;
pub const T_COLOR_WRITEMASK: u32 = 0x10b;
pub const T_FRONT_FACE: u32 = 0x108;
pub const T_CLEAR_COLOR: u32 = 0x104;
pub const T_CLEAR_COLOR_DEPTH: u32 = 0x0a0;
/// Window rectangle, sent by the kernel at context restore: x0; DATA y0 (GL,
/// bottom-up), w, h, 0x10, 0, 1, (x1 << 11) | x0, (y1 << 10) | y0, 0 x 6.
pub const T_WINDOW: u32 = 0x1e5;
/// MakeCurrent: pixel mode from a kernel ioctl (4 = 24-bit RGB, 2 = 12-bit
/// RGB double buffered; other values (unverified) keep the current format).
pub const T_MAKECURRENT: u32 = 0x004;
/// Cull flags (__glExpPassCullFace): 0x1B = cull front faces, 0x1C = back.
pub const T_CULL_FRONT: u32 = 0x01b;
pub const T_CULL_BACK: u32 = 0x01c;
/// Polygon mode (front): 1 = fill, 2 = point, 3 = line.
pub const T_POLYGON_MODE: u32 = 0x0e2;
/// SwapBuffers: the new swap state (0/1) selects which COLOR_WRITEMASK of
/// the pair applies (the kernel flips the XMAP buffer select).
pub const T_SWAP_BUFFERS: u32 = 0x1e7;
/// glEnable/glDisable(GL_DITHER) (dither_enable, 0x2044); libglcore also
/// turns it off around clears with an integral colour.
pub const T_DITHER: u32 = 0x011;
/// glPolygonStipple + enable (__glExpPassPolygonStipple, 0x207C): 1; DATA
/// 32 words in the layout __glExpConvertStipple builds from the GL mask
/// (row 0 = bottom = window y mod 32, MSB = leftmost = window x mod 32):
/// word i (0..15) = left 16 pixels of row 30-2i << 16 | left 16 pixels of
/// row 31-2i; word 16+i = the same for the right 16 pixels.
pub const T_STIPPLE_ON: u32 = 0x01f;
/// glDisable(GL_POLYGON_STIPPLE) (0x2078): 0.
pub const T_STIPPLE_OFF: u32 = 0x01e;
/// Depth (gr2_depth.c): test enable 0/1; func = testFunc & 7 (NEVER..ALWAYS);
/// write mask ((1 << zbits) - 1, or 0). Window z spans 0 .. 2^zbits - 1 via
/// the viewport zscale/zcenter.
pub const T_DEPTH_TEST: u32 = 0x014;
pub const T_DEPTH_FUNC: u32 = 0x024;
pub const T_DEPTH_MASK: u32 = 0x00a;
/// Depth-only clear (gr2_depth.c Clear, HQ2_FIFO_CI.clear_depth: token 0xA0
/// with ITOF|CP): 0; DATA depth, 0 (unverified: not yet seen in a trace).
pub const T_DEPTH_CLEAR: u32 = 0x68a0;
/// Stencil (gr2_stencil.c): mode = enable; DATA ref, func, mask, fail,
/// zfail, zpass (ops 0 KEEP 1 INVERT 2 ZERO 3 REPLACE 4 INCR 5 DECR);
/// write mask; config = stencil bits (MakeCurrent); clear = value; DATA mask.
pub const T_STENCIL_MODE: u32 = 0x00f;
pub const T_STENCIL_WMASK: u32 = 0x010;
pub const T_STENCIL_CONFIG: u32 = 0x00e;
pub const T_STENCIL_CLEAR: u32 = 0x0a1;
// IRIS GL (libgl.so) on the same microcode: most tokens are shared with
// OpenGL; these are its own (decoded from atlantis traces, IRIX 6.5.22).
/// Per-vertex normal (n3f): 3 floats.
/// IRIS GL mmode(MSINGLE): libgl.so multiplies on the host and loads the
/// whole object-to-clip matrix (16 floats, same order as 0x037) after every
/// matrix call (amesh). Replaces P * MV until the next 0x037 / 0x038.
pub const T_IRIS_MATRIX: u32 = 0x036;
pub const T_IRIS_NORMAL: u32 = 0x035;
/// Current colour: any conversion; packed (cpack) is 0xAABBGGRR (R low).
pub const T_IRIS_COLOR: u32 = 0x113;
/// bgnpolygon; the vertex routine is LOADV|0x1AE; endpolygon.
pub const T_IRIS_BGNPOLYGON: u32 = 0x1a4;
pub const T_IRIS_ENDPOLYGON: u32 = 0x041;
const VR_IRIS_POLYGON: u32 = 0x1ae;
/// color(i) is 0x030 (ITOF|C1 for an integer index). pmv / pdr / pclos
/// polygons (libgl.so gl_i_pmv2 / gl_i_pdr / gl_i_pclos; rectf, circf,
/// showmap): 0x044 then LOADV|0x1AE, points on 0x045 (V3 or ITOF|V3), 0x042
/// then LOADV|0x065.
pub const T_IRIS_INDEX: u32 = 0x030;
pub const T_IRIS_PMV: u32 = 0x044;
pub const T_IRIS_PCLOS: u32 = 0x042;
pub const T_IRIS_PDR: u32 = 0x045;
/// libgl.so (IP12GR232) move / draw: move sends 0x05B = 0, then the point
/// on 0x05D (V3, float or ITOF); draw sends only the point. A line from the
/// previous point to each drawn one (gl_i_move2, gl_i_draw).
pub const T_IRIS_MOVE: u32 = 0x05b;
pub const T_IRIS_DRAW: u32 = 0x05d;
/// cmov: current character position, V3 (gl_c_cmov). getcpos sends 0x068 =
/// 0, Finish, reads the mailbox: x, y (window pixels, stored as shorts) and
/// a word whose sign bit means "invalid, leave the caller's values" (gl_g_getcpos).
pub const T_IRIS_CMOV: u32 = 0x066;
pub const T_IRIS_GETCPOS: u32 = 0x068;
/// Character bitmaps drawn at the character position, which then advances
/// (charstr / fmprstr; gr_osview, IRIX 6.5.22 trace; sender not found in
/// libgl.so, format (inferred) from the glyph data). Four header words:
///   w << 16 | h;  xorig << 16 | yorig (i16);  xmove << 16 | ymove (i16);
///   flags (bit 0 clear: the first 16-bit row slot is padding);
/// then rows top-down, MSB = leftmost pixel:
///   0x069: 9 words, two 16-bit rows each, low half first (h <= 17);
///   0x06A: 17 words, the same (h <= 33);
///   0x06D: 17 words, one 32-bit row each (w <= 32, h <= 17).
pub const T_IRIS_CHAR16: u32 = 0x069;
pub const T_IRIS_CHAR16_TALL: u32 = 0x06a;
pub const T_IRIS_CHAR32: u32 = 0x06d;
/// Frame setup: 0x008 = 0, back buffer (0/1); 0x005 = colour write mask
/// for it (0x000FFF / 0xFFF000 for 12-bit double buffering).
pub const T_IRIS_BUFFER: u32 = 0x008;
pub const T_IRIS_WRITEMASK: u32 = 0x005;
/// clear() with the current colour; zclear(value).
pub const T_IRIS_CLEAR: u32 = 0x09e;
pub const T_IRIS_ZCLEAR: u32 = 0x09f;
/// IRIS GL lighting enable: 0; DATA on (same shape as OpenGL's 0x10E, and
/// likewise followed by 0x0D4 two-sided).
pub const T_IRIS_DB: u32 = 0x0db;
/// Blending (gr2_raster.c __glExpPassBlendFunc): mode = 1 for the
/// SRC_ALPHA / ONE_MINUS_SRC_ALPHA fast path; factor port = 3 words: enable
/// (0 = ONE, ZERO), src code, dst code (re3::RE3_OP_BLEND codes).
pub const T_BLEND_MODE: u32 = 0x026;
pub const T_BLEND_FACTOR: u32 = 0x025;
/// Logic op (__glExpPassLogicOp): GL logicOp & 0xF (3 = COPY; RE3 FUNC order).
pub const T_LOGIC_OP: u32 = 0x01a;
/// Software-pipeline fragment (gr2_rgb.c Store / Store_B, used e.g. with
/// GL_ALPHA_TEST, which GR2 lacks; the fragments have already passed it, so
/// they are written unconditionally apart from Z/stencil/blend/masks):
/// ITOF token = x; ITOF DATA y, z; DATA
/// r, g, b, a (floats 0..1). x, y are window coordinates + 6144
/// (constants.viewportX/YAdjust, __glExpCreateContext).
pub const T_FRAGMENT: u32 = 0x402b;
const FRAGMENT_BIAS: i32 = 6144;
/// Software spans (CPU-rasterized textures, IRIS GL software lines):
/// setup = 6 floats x, dx, y, dy, z, dz (x, y window coordinates + 6144;
/// libglcore __glExpTextureSpan sends dx = 1, dy = 0); then one colour per
/// pixel on 0x02A (float RGBA x4, or ITOF|CP packed 0xAABBGGRR as IRIS GL
/// sends it) which writes the pixel and steps x, y and z; 0x02D steps
/// without writing. 0x02C: same format as 0x02A, seen from IRIS GL (blast)
/// for vertical constant-colour spans; how it differs is (unverified).
pub const T_SPAN_SETUP: u32 = 0x029;
pub const T_SPAN_COLOR: u32 = 0x02a;
pub const T_SPAN_COLOR_B: u32 = 0x02c;
pub const T_SPAN_SKIP: u32 = 0x02d;
/// State readbacks (gr2_get.c / __glExpGetColor ...): token = 0, then the
/// client waits for FIN3 (Finish), reads the mailbox and acks with
/// read_done (0xBD). Mailbox: GL address 0x12088 = board 0x10088 = shram
/// word 0x4022, 4 words.
pub const T_GET_COLOR: u32 = 0x0e9;
pub const T_GET_NORMAL: u32 = 0x0ea;
pub const T_GET_RASTERPOS: u32 = 0x107;
pub const T_READ_DONE: u32 = 0x0bd;
const READBACK_SHRAM: usize = 0x4022;

// Vertex routines (LOADV token) and Begin/End tokens (gr2_prim.c).
const VR_OUTSIDE: u32 = 0x065;
const VR_POINTS: u32 = 0x123;
const VR_LINES: u32 = 0x0eb;
const VR_LSTRIP: u32 = 0x056;
const VR_LLOOP: u32 = 0x059;
const VR_TRIANGLES: u32 = 0x0f2;
const VR_TSTRIP: u32 = 0x047;
const VR_TFAN: u32 = 0x0ed;
const VR_QUADS: u32 = 0x0f5;
const VR_QSTRIP: u32 = 0x04d;
const VR_POLYGON: u32 = 0x0f8;
const BEGIN_TOKENS: [u32; 8] = [0x11d, 0x17c, 0x058, 0x0f0, 0x046, 0x0f3, 0x04c, 0x0f7];
const END_TOKENS: [u32; 9] = [0x168, 0x057, 0x05a, 0x0f1, 0x04a, 0x0f4, 0x051, 0x0fc, 0x0fd];
const END_LLOOP: u32 = 0x05a;
const END_POLYGON: u32 = 0x0fc;

const USEV: u32 = 0x200;
const LOADV: u32 = 0x400;
const ITOF: u32 = 0x4000;

/// A vertex after transform: window coordinates (GL: y up), colour 0..1.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Wv {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub c: [f32; 4],
    /// Back-face colour (two-sided lighting; = c otherwise).
    pub cb: [f32; 4],
    /// 0 when the vertex is behind the eye (w <= 0): its primitive is dropped.
    pub ok: u32,
}

/// GL state held by the HQ2/GE7. Plain data, valid when zeroed.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GlState {
    /// Software span position and steps (T_SPAN_SETUP), window relative.
    /// First: the only 8-byte-aligned field, so the struct has no padding
    /// (asserted below) and its context image holds no uninitialised bytes.
    span: [f64; 6],
    inited: u32,
    /// Window clip from the kernel (0x1E5, Gr2ValidateClip / Gr2PcxSwap):
    /// WID, obscured flag, number of pieces (> 4: clip by the RE3 WID test),
    /// and up to 4 visible rectangles x0, y0, x1, y1 (screen, GL y up,
    /// exclusive).
    clip_wid: u32,
    clip_obscured: u32,
    clip_n: u32,
    clip_rects: [[i32; 4]; 4],
    /// RE3 WID test left enabled by gl_setup.
    wid_sent: u32,
    /// Window: x0, y0 (screen, GL y up), w, h.
    win: [i32; 4],
    /// Viewport (window relative): x0, x1, y0, y1, zscale, zcenter.
    vp: [f32; 6],
    mv: [f32; 16],
    proj: [f32; 16],
    mvp: [f32; 16],
    mvp_dirty: u32,
    smooth: u32,
    color: [f32; 4],
    /// Scissor (window relative, inclusive): x0, y0, x1, y1.
    scissor: [i32; 4],
    colormask: u32,
    /// COLOR_WRITEMASK pair (__glExpPassDrawBuffer): plane mask for swap
    /// state 0 and 1; `swap` is the current state.
    masks: [u32; 2],
    swap: u32,
    /// re3::PIXFMT_* of the current window's visual.
    pixfmt: u32,
    cull_front: u32,
    cull_back: u32,
    dither: u32,
    stipple_on: u32,
    stipple: [u32; 32],
    ztest: u32,
    zfunc: u32,
    zmask: u32,
    /// Stencil: enable, ref, func, mask, fail, zfail, zpass; write mask.
    st: [u32; 7],
    st_wmask: u32,
    /// RE3 Z/stencil control was left non-zero by the last primitive.
    zs_sent: u32,
    blend_on: u32,
    blend_src: u32,
    blend_dst: u32,
    blend_sent: u32,
    /// 0x026: selects the GE's fast SRC_ALPHA / ONE_MINUS_SRC_ALPHA blend
    /// path (0x025 enable selects the general path with its own factors).
    /// Either one turns blending on; with the fast path, 0x025's factors are
    /// ignored (IRIS GL sends 0x025 = off, ONE, ZERO with it).
    blend_fast: u32,
    logic_op: u32,
    normal: [f32; 3],
    /// Lighting and fog (gl_light.rs).
    lt: light::Lighting,
    front_ccw: u32,
    polymode: u32,
    /// Self-streaming port being assembled.
    port: u32,
    port_n: u32,
    port_buf: [u32; 24],
    /// Primitive assembly.
    vroutine: u32,
    nv: u32,
    first: Wv,
    prev: [Wv; 3],
    /// Character position (cmov): window x, y; 0 valid, 1 clipped.
    cpos: [i32; 3],
    /// Colour latched by cmov for the characters drawn there.
    cpos_color: [f32; 4],
    /// move / draw pen: 0 = next 0x05D point is a move.
    pen: u32,
    pen_at: Wv,
    /// Vertices drawn / primitives emitted since the last trace note.
    pub stats_vertices: u32,
}

const IDENT: [f32; 16] = [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.];

impl GlState {
    /// Range of the R channel in GL colour units (see to_fixed).
    fn cmax(&self) -> f32 {
        if self.pixfmt == re3::PIXFMT_CI12 { 4095.0 / 255.0 } else { 1.0 }
    }

    fn ensure_init(&mut self) {
        if self.inited == 0 {
            self.inited = 1;
            self.mv = IDENT;
            self.proj = IDENT;
            self.mvp = IDENT;
            self.color = [1.0; 4];
            self.smooth = 1;
            self.colormask = 0x00ff_ffff;
            self.masks = [0x00ff_ffff; 2];
            self.logic_op = 3;
            self.lt.init();
            self.blend_src = 1;
            self.normal = [0.0, 0.0, 1.0];
            self.front_ccw = 1;
            self.polymode = 1;
            self.scissor = [0, 0, 0x7ff, 0x7ff];
            self.win = [0, 0, re3::FB_W as i32, SCREEN_H];
            self.vp = [0.0, re3::FB_W as f32 - 1.0, 0.0, SCREEN_H as f32 - 1.0, 1.0, 0.0];
        }
    }

    /// Effective blend factors (src, dst), None = blending off.
    fn blend_func(&self) -> Option<(u32, u32)> {
        if self.blend_fast != 0 {
            Some((4, 5))
        } else if self.blend_on != 0 {
            Some((self.blend_src, self.blend_dst))
        } else {
            None
        }
    }

    fn update_mvp(&mut self) {
        if self.mvp_dirty != 0 {
            self.mvp_dirty = 0;
            // Column-major (glLoadMatrix order): clip = P * MV * v.
            let (a, b) = (&self.proj, &self.mv);
            let mut m = [0.0f32; 16];
            for c in 0..4 {
                for r in 0..4 {
                    m[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
                }
            }
            self.mvp = m;
        }
    }

    /// Object coordinates -> window vertex (screen pixels, GL y up).
    fn transform(&mut self, v: [f32; 4]) -> Wv {
        self.update_mvp();
        let m = &self.mvp;
        let clip = |r: usize| m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
        let (cx, cy, cz, cw) = (clip(0), clip(1), clip(2), clip(3));
        let mut out = Wv { c: self.color, cb: self.color, ..Default::default() };
        if cw <= 1e-6 {
            return out;
        }
        let (nx, ny, nz) = (cx / cw, cy / cw, cz / cw);
        let [vx0, vx1, vy0, vy1, zs, zc] = self.vp;
        // Snap to 1/16 pixel, like the hardware's fixed-point subpixel
        // coordinates (removes float noise such as 319.99998 on edges).
        let snap = |v: f32| (v * 16.0).round() / 16.0;
        out.x = snap(self.win[0] as f32 + vx0 + (nx + 1.0) * 0.5 * (vx1 - vx0 + 1.0));
        out.y = snap(self.win[1] as f32 + vy0 + (ny + 1.0) * 0.5 * (vy1 - vy0 + 1.0));
        out.z = zc + nz * zs;
        out.ok = 1;
        out
    }

    /// Window clipping by the RE3 WID test: more visible pieces than the 4
    /// rectangles 0x1E5 can carry.
    fn wid_clip(&self) -> bool {
        self.clip_n > 4
    }

    /// Visible rectangles (screen, exclusive): clip_rect() intersected with
    /// the window's visible pieces. Empty ones are dropped.
    fn visible_rects(&self) -> ([[i32; 4]; 4], usize) {
        let r = self.clip_rect();
        let mut out = [[0i32; 4]; 4];
        // The piece list decides, not the obscured flag: a partly covered
        // window arrives with obscured = 0 and 2 pieces (IRIX 6.5.22 atlantis
        // trace). 0 pieces = the whole window.
        if self.clip_n == 0 || self.wid_clip() {
            if r[0] < r[2] && r[1] < r[3] {
                out[0] = r;
                return (out, 1);
            }
            return (out, 0);
        }
        let mut n = 0;
        for p in &self.clip_rects[..self.clip_n.min(4) as usize] {
            let c = [r[0].max(p[0]), r[1].max(p[1]), r[2].min(p[2]), r[3].min(p[3])];
            if c[0] < c[2] && c[1] < c[3] {
                out[n] = c;
                n += 1;
            }
        }
        (out, n)
    }

    /// The visible pieces of row `y` between x0 and x1 (exclusive), left to
    /// right.
    fn span_pieces(&self, y: i32, x0: i32, x1: i32) -> ([(i32, i32); 4], usize) {
        let (rects, n) = self.visible_rects();
        let mut out = [(0, 0); 4];
        let mut k = 0;
        for r in &rects[..n] {
            if y >= r[1] && y < r[3] {
                let (a, b) = (x0.max(r[0]), x1.min(r[2]));
                if a < b {
                    out[k] = (a, b);
                    k += 1;
                }
            }
        }
        out[..k].sort_unstable_by_key(|p| p.0);
        (out, k)
    }

    fn pixel_visible(&self, x: i32, y: i32) -> bool {
        self.span_pieces(y, x, x + 1).1 != 0
    }

    /// Drawable rectangle in screen pixels: window ∩ scissor ∩ screen,
    /// as x0, y0, x1, y1 (exclusive).
    fn clip_rect(&self) -> [i32; 4] {
        let [wx, wy, ww, wh] = self.win;
        let mut r = [wx, wy, wx + ww, wy + wh];
        let s = self.scissor;
        if !(s[0] == 0 && s[1] == 0 && s[2] >= 0x7ff && s[3] >= 0x7ff) {
            r = [r[0].max(wx + s[0]), r[1].max(wy + s[1]), r[2].min(wx + s[2] + 1), r[3].min(wy + s[3] + 1)];
        }
        [r[0].max(0), r[1].max(0), r[2].min(re3::FB_W as i32), r[3].min(SCREEN_H)]
    }
}

#[inline]
fn f(v: u32) -> f32 {
    f32::from_bits(v)
}

/// Words a self-streaming port takes, for FIFO index `index`.
fn port_words(index: u32) -> Option<u32> {
    let tok = index & 0x1ff;
    let conv = (index >> 11) & 7;
    match tok {
        T_SPAN_COLOR | T_SPAN_COLOR_B if conv == 5 => Some(1),
        T_SPAN_COLOR | T_SPAN_COLOR_B if conv == 0 || conv == 4 => Some(4),
        T_SPAN_SETUP | T_SPAN_SKIP if index & !0x1ff == 0 => Some(if tok == T_SPAN_SETUP { 6 } else { 1 }),
        T_VERTEX | T_IRIS_PDR if index & (USEV | LOADV) != 0 || conv != 0 => Some(match conv { 1 => 3, 2 => 2, _ => 4 }),
        T_IRIS_INDEX if conv == 6 => Some(1),
        T_IRIS_DRAW | T_IRIS_CMOV if conv != 0 => Some(match conv { 1 => 3, 2 => 2, _ => 4 }),
        T_COLOR if index & !0x1ff != 0 => Some(match conv { 3 => 3, 4 => 4, _ => 1 }),
        T_IRIS_COLOR => Some(match conv { 3 => 3, 4 => 4, _ => 1 }),
        _ if index & !0x1ff != 0 => None,
        T_MODELVIEW | T_PROJECTION | T_TEXTURE_MATRIX | T_IRIS_MATRIX => Some(16),
        T_IRIS_CHAR16 => Some(13),
        T_IRIS_CHAR16_TALL | T_IRIS_CHAR32 => Some(21),
        T_NORMAL_MATRIX => Some(9),
        T_NORMAL | T_IRIS_NORMAL => Some(3),
        T_COLOR_WRITEMASK => Some(2),
        T_BLEND_FACTOR => Some(3),
        t => light::Lighting::port_words(t),
    }
}

/// R, G, B as RE3 8.11 iterator values. `max` is the channel range in GL
/// units: 1.0 in RGB; in colour-index mode R carries the index (/255) and
/// may reach 4095/255 (R is 12.11, RE3.h).
fn to_fixed(c: [f32; 4], max: f32) -> [u32; 3] {
    let q = |v: f32, m: f32| ((v.clamp(0.0, m) * 255.0 + 0.5) as u32) << 11;
    [q(c[0], max), q(c[1], 1.0), q(c[2], 1.0)]
}

/// GL context save / restore through kernel memory, as the kernel's
/// Gr2PcxSwap drives it (HQ2.h "KERNEL TOKENS"):
///   GE_HQMSAV (0x1F0) id; DATA state, mode  -> the microcode reports in
///       shram word 0x302 (CX_SIZE_MAIN) how many words the outgoing
///       context's state takes and in word 0x303 whose it is (its id);
///   0x1E1 (save main): the kernel reads that many words from HQ2_GEDMA by
///       VDMA into a kmem buffer of the outgoing context;
///   0x1E2 (restore main) = word count: the kernel writes a saved buffer back
///       through HQ2_GEDMA when the incoming context's state is 2 (saved).
/// The HLE's saved image is the GlState bytes behind a 2-word header. GlState
/// is #[repr(C)] plain data (numbers and arrays only, no pointers), so the
/// image means the same thing when it comes back. A new context (state 0)
/// starts from the defaults: IRIS GL winopen never turns lighting, blending
/// or fog off.
pub const CX_MAGIC: u32 = 0x474c_4358; // "GLCX"
pub const CX_WORDS: usize = 2 + std::mem::size_of::<GlState>() / 4;
const _: () = assert!(std::mem::size_of::<GlState>() % 4 == 0);
// No padding anywhere in GlState: every field after `span` is 4-byte
// aligned, so there is none inside; the last field must end the struct.
const _: () = assert!(std::mem::offset_of!(GlState, inited) == 48);
const _: () = assert!(
    std::mem::offset_of!(GlState, stats_vertices) + 4 == std::mem::size_of::<GlState>()
);

/// Context bookkeeping and the restore stream being received.
#[repr(C)]
pub struct GlCx {
    /// Id (GE_HQMSAV word) of the context whose state is live.
    cur: u32,
    have_cur: u32,
    /// 0x1E2 in progress: words expected / received.
    expect: u32,
    got: u32,
    buf: [u32; CX_WORDS],
    /// Image of the outgoing context, taken at GE_HQMSAV (before a new
    /// context resets the state) and handed out at 0x1E1.
    save: [u32; CX_WORDS],
}

impl Hq2Engine {
    /// 0x1E1 save main: the image the kernel now reads from HQ2_GEDMA.
    pub(super) fn gl_cx_save(&mut self, out: &mut dyn Re3Sink) {
        out.cx_publish(&self.gl_ctx.save);
    }

    /// The live GL state as a context image.
    fn gl_cx_image(&self, out: &mut [u32; CX_WORDS]) {
        out[0] = CX_MAGIC;
        out[1] = std::mem::size_of::<GlState>() as u32;
        // SAFETY: GlState is plain data of size_of / 4 words (asserted above).
        unsafe {
            std::ptr::copy_nonoverlapping(
                &self.gl as *const GlState as *const u32, out[2..].as_mut_ptr(), CX_WORDS - 2);
        }
    }

    /// GE_HQMSAV (context id; DATA state, mode).
    pub(super) fn gl_switch_context(&mut self, id: u32, state: u32, out: &mut dyn Re3Sink) {
        if state == 3 {
            // Detach: the context's live state leaves the GE. Gr2PcxSwap
            // (mode change) saves it into that context's own buffer;
            // Gr2DestroyDDRN (context exit) discards it. Either way the GE has
            // no owner afterwards, so no later switch may name this context
            // for a save: its RRM node may already be gone (kernel NULL
            // dereference at ->unk4C->unk4 otherwise).
            if self.gl_ctx.have_cur != 0 && self.gl_ctx.cur == id {
                let mut img = [0u32; CX_WORDS];
                self.gl_cx_image(&mut img);
                self.gl_ctx.save = img;
                out.shram(super::SHRAM_CX_OWNER, id);
                out.shram(super::SHRAM_CX_SIZE_MAIN, CX_WORDS as u32);
            } else {
                out.shram(super::SHRAM_CX_SIZE_MAIN, 0);
            }
            self.gl_ctx.have_cur = 0;
            return;
        }
        let outgoing = self.gl_ctx.have_cur != 0 && self.gl_ctx.cur != id;
        if outgoing {
            let mut img = [0u32; CX_WORDS];
            self.gl_cx_image(&mut img);
            self.gl_ctx.save = img;
            out.shram(super::SHRAM_CX_OWNER, self.gl_ctx.cur);
            out.shram(super::SHRAM_CX_SIZE_MAIN, CX_WORDS as u32);
        } else {
            out.shram(super::SHRAM_CX_SIZE_MAIN, 0);
        }
        if state == 0 {
            // SAFETY: GlState is plain data, valid when zeroed (ensure_init
            // fills in the defaults on first use).
            self.gl = unsafe { std::mem::zeroed() };
        }
        // State 2: the kernel restores the saved image next (0x1E2). Other
        // states keep the live state (a context the HLE holds no image of:
        // the session or trace started after it was created).
        self.gl_ctx.cur = id;
        self.gl_ctx.have_cur = 1;
    }

    /// 0x1E2 restore main: `words` GEDMA words follow.
    pub(super) fn gl_cx_restore_begin(&mut self, words: u32) {
        self.gl_ctx.expect = words;
        self.gl_ctx.got = 0;
    }

    /// One restore word. Returns true when the image is complete (then the
    /// state is loaded if the image is ours and intact).
    pub(super) fn gl_cx_restore_word(&mut self, val: u32) -> bool {
        let c = &mut self.gl_ctx;
        if (c.got as usize) < CX_WORDS {
            c.buf[c.got as usize] = val;
        }
        c.got += 1;
        if c.got < c.expect {
            return false;
        }
        if c.expect as usize == CX_WORDS && c.buf[0] == CX_MAGIC
            && c.buf[1] == std::mem::size_of::<GlState>() as u32 {
            // SAFETY: same layout as gl_cx_image wrote; GlState is plain data
            // for which every bit pattern of its number fields is valid.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    c.buf[2..].as_ptr(), &mut self.gl as *mut GlState as *mut u32, CX_WORDS - 2);
            }
        }
        true
    }

    /// Is `index` a GL self-streaming port?
    pub(super) fn gl_port_ready(&self, index: u32) -> bool {
        port_words(index).is_some()
    }

    /// Fixed word counts of GL commands that take DATA words.
    pub(super) fn gl_arg_count(cmd: u32) -> Option<u32> {
        Some(match cmd {
            T_VIEWPORT => 7,
            T_SCISSOR => 4,
            T_CLEAR_COLOR => 4,
            T_CLEAR_COLOR_DEPTH => 6,
            T_WINDOW => 15,
            light::T_LIGHTING | light::T_TWO_SIDED => 2,
            light::T_NORMALIZE | light::T_NORMALIZE_B | light::T_FOG_ON | light::T_MATERIAL_COMMIT => 1,
            T_SHADE_MODEL | T_FRONT_FACE | T_MAKECURRENT | T_CULL_FRONT | T_CULL_BACK
            | T_POLYGON_MODE | T_SWAP_BUFFERS | T_DITHER | T_STIPPLE_OFF
            | T_DEPTH_TEST | T_DEPTH_FUNC | T_DEPTH_MASK | T_STENCIL_WMASK | T_STENCIL_CONFIG
            | T_BLEND_MODE | T_LOGIC_OP | T_GET_COLOR | T_GET_NORMAL | T_GET_RASTERPOS | T_READ_DONE => 1,
            T_FRAGMENT => 7,
            T_STENCIL_CLEAR | T_IRIS_BUFFER | T_IRIS_DB => 2,
            T_IRIS_WRITEMASK | T_IRIS_CLEAR | T_IRIS_ZCLEAR | T_IRIS_BGNPOLYGON | T_IRIS_ENDPOLYGON
            | T_IRIS_PMV | T_IRIS_PCLOS | T_IRIS_MOVE | T_IRIS_GETCPOS => 1,
            T_DEPTH_CLEAR => 3,
            T_STENCIL_MODE => 7,
            T_STIPPLE_ON => 33,
            c if BEGIN_TOKENS.contains(&c) || END_TOKENS.contains(&c) => 1,
            c if c & LOADV != 0 && c & !(LOADV | 0x1ff) == 0 => 1,
            _ => return None,
        })
    }

    /// A GE parameter pair (0x080) whose value goes to DATA|C1 instead of the
    /// token: parameters 18..23, colour-index materials (__glExpValidateMaterial;
    /// IRIS GL sends them in RGB mode too). Completes the pending pair.
    pub(super) fn gl_port_wants_data(&mut self, val: u32, out: &mut dyn Re3Sink,
                                     done: &mut Option<&mut dyn FnMut(String)>) -> bool {
        if self.gl.port == light::T_LIGHT_PARAM && self.gl.port_n == 1 {
            return self.gl_port(light::T_LIGHT_PARAM, val, out, done);
        }
        false
    }

    /// Self-streaming GL ports (matrices, vertex, colour, normal): consume one
    /// word. Returns false if `index` is not such a port.
    pub(super) fn gl_port(&mut self, index: u32, val: u32, out: &mut dyn Re3Sink,
                          done: &mut Option<&mut dyn FnMut(String)>) -> bool {
        let Some(n) = port_words(index) else { return false };
        self.gl.ensure_init();
        let g = &mut self.gl;
        if g.port != index || g.port_n >= n {
            g.port = index;
            g.port_n = 0;
        }
        g.port_buf[g.port_n as usize] = val;
        g.port_n += 1;
        if g.port_n < n {
            return true;
        }
        let b = g.port_buf;
        let tok = index & 0x1ff;
        let conv = (index >> 11) & 7;
        let num = |w: u32| if index & ITOF != 0 { w as i32 as f32 } else { f(w) };
        match tok {
            T_MODELVIEW | T_PROJECTION => {
                let mut m = [0.0f32; 16];
                for i in 0..16 {
                    m[i] = f(b[i]);
                }
                if tok == T_MODELVIEW { g.mv = m } else { g.proj = m }
                g.mvp_dirty = 1;
                if let Some(d) = done.as_mut() {
                    d(format!("{} [{:.3} {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} {:.3}]",
                        if tok == T_MODELVIEW { "GL_MODELVIEW" } else { "GL_PROJECTION" },
                        m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]));
                }
            }
            T_SPAN_SETUP => {
                let v = |i: usize| f(b[i]) as f64;
                g.span = [v(0) - FRAGMENT_BIAS as f64, v(1), v(2) - FRAGMENT_BIAS as f64, v(3), v(4), v(5)];
                if let Some(d) = done.as_mut() {
                    let sp = g.span;
                    d(format!("GL_SPAN_SETUP window ({}, {}) step ({}, {}) z {:.0} dz {:.2}", sp[0], sp[2], sp[1], sp[3], sp[4], sp[5]));
                }
            }
            T_SPAN_COLOR | T_SPAN_COLOR_B | T_SPAN_SKIP => {
                let c = if tok == T_SPAN_SKIP {
                    None
                } else if conv == 5 {
                    let u = |sh: u32| ((b[0] >> sh) & 0xff) as f32 / 255.0;
                    Some([u(0), u(8), u(16), u(24)])
                } else if index & ITOF != 0 {
                    let u = |w: u32| (w as i32 as f32) / 255.0;
                    Some([u(b[0]), u(b[1]), u(b[2]), u(b[3])])
                } else {
                    Some([f(b[0]), f(b[1]), f(b[2]), f(b[3])])
                };
                let sp = g.span;
                g.span[0] += sp[1];
                g.span[2] += sp[3];
                g.span[4] += sp[5];
                if let Some(c) = c {
                    let z = (sp[4].round().clamp(-8388608.0, 8388607.0) as i32 as u32) & 0x00ff_ffff;
                    self.gl_fragment(sp[0].floor() as i32, sp[2].floor() as i32, z, c, out);
                }
            }
            T_IRIS_MATRIX => {
                for i in 0..16 {
                    g.mvp[i] = f(b[i]);
                }
                g.mvp_dirty = 0;
                if let Some(d) = done.as_mut() {
                    let m = &g.mvp;
                    d(format!("IRISGL_MATRIX (single) [{:.3} {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} {:.3} | {:.3} {:.3} {:.3} {:.3}]",
                        m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]));
                }
            }
            T_COLOR => {
                g.color = match conv {
                    3 | 4 => {
                        // Integer colours arrive pre-scaled to 0..255 by
                        // libglcore (ub as is, us >> 8, ui >> 24, i >> 23, ...).
                        let s = |w: u32| if index & ITOF != 0 { (w as i32 as f32) / 255.0 } else { f(w) };
                        [s(b[0]), s(b[1]), s(b[2]), if conv == 4 { s(b[3]) } else { 1.0 }]
                    }
                    // Packed colour: byte order (unverified) R in bits 31:24.
                    5 => {
                        let u = |sh: u32| ((b[0] >> sh) & 0xff) as f32 / 255.0;
                        [u(24), u(16), u(8), u(0)]
                    }
                    _ => g.color,
                };
            }
            T_IRIS_INDEX => {
                g.color = [num(b[0]) / 255.0, 0.0, 0.0, 1.0];
            }
            T_IRIS_CMOV => {
                let v = [num(b[0]), num(b[1]), if conv == 2 { 0.0 } else { num(b[2]) }, 1.0];
                let wv = g.transform(v);
                g.cpos = [
                    (wv.x - g.win[0] as f32).floor() as i32,
                    (wv.y - g.win[1] as f32).floor() as i32,
                    (wv.ok == 0) as i32,
                ];
                g.cpos_color = g.color;
            }
            T_IRIS_CHAR16 | T_IRIS_CHAR16_TALL | T_IRIS_CHAR32 => {
                let b = &b[..n as usize];
                self.gl_glyph(tok, b, out);
            }
            T_IRIS_DRAW => {
                let v = [num(b[0]), num(b[1]), if conv == 2 { 0.0 } else { num(b[2]) }, 1.0];
                let wv = g.transform(v);
                let (p, pen) = (g.pen_at, g.pen);
                g.pen_at = wv;
                g.pen = 1;
                if pen != 0 {
                    let c = wv.c;
                    self.gl_line(p, wv, c, out);
                }
            }
            T_VERTEX | T_IRIS_PDR => {
                let v = match conv {
                    1 => [num(b[0]), num(b[1]), num(b[2]), 1.0],
                    2 => [num(b[0]), num(b[1]), 0.0, 1.0],
                    _ => [num(b[0]), num(b[1]), num(b[2]), num(b[3])],
                };
                let mut wv = g.transform(v);
                if g.lt.on != 0 || g.lt.fog_on != 0 {
                    // Eye-space position for lighting and fog.
                    let m = &g.mv;
                    let e = |r: usize| m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
                    let eye = [e(0), e(1), e(2), e(3)];
                    if g.lt.on != 0 {
                        let (fc, bc) = g.lt.light_vertex(eye, g.normal, g.color);
                        wv.c = fc;
                        wv.cb = bc;
                    }
                    wv.c = g.lt.fog_color(eye, wv.c);
                    wv.cb = g.lt.fog_color(eye, wv.cb);
                }
                g.stats_vertices += 1;
                if let Some(d) = done.as_mut() {
                    d(format!("VERTEX ({:.2}, {:.2}, {:.2}, {:.2}) -> window ({:.2}, {:.2}, z {:.0}){} color ({:.3}, {:.3}, {:.3})",
                        v[0], v[1], v[2], v[3], wv.x, wv.y, wv.z, if wv.ok != 0 { "" } else { " CULLED(w<=0)" },
                        wv.c[0], wv.c[1], wv.c[2]));
                }
                self.gl_vertex(wv, out, done);
            }
            T_BLEND_FACTOR => {
                g.blend_on = b[0] & 1;
                g.blend_src = b[1] & 7;
                g.blend_dst = b[2] & 7;
                if let Some(d) = done.as_mut() {
                    d(format!("GL_BLEND on={} src={} dst={}", g.blend_on, g.blend_src, g.blend_dst));
                }
            }
            T_NORMAL | T_IRIS_NORMAL => g.normal = [f(b[0]), f(b[1]), f(b[2])],
            T_IRIS_COLOR => {
                g.color = match conv {
                    3 | 4 => {
                        let s = |w: u32| if index & ITOF != 0 { (w as i32 as f32) / 255.0 } else { f(w) };
                        [s(b[0]), s(b[1]), s(b[2]), if conv == 4 { s(b[3]) } else { 1.0 }]
                    }
                    // cpack: 0xAABBGGRR (atlantis clears with 0x00953535,
                    // its dark blue sea).
                    _ => {
                        let u = |sh: u32| ((b[0] >> sh) & 0xff) as f32 / 255.0;
                        [u(0), u(8), u(16), u(24)]
                    }
                };
            }
            T_COLOR_WRITEMASK => {
                // Word 0 = plane mask for swap state 0, word 1 = state 1.
                // (Swapping them makes ideas flicker from the start: tested.)
                g.masks = [b[0] & 0x00ff_ffff, b[1] & 0x00ff_ffff];
                if let Some(d) = done.as_mut() {
                    d(format!("GL_COLOR_WRITEMASK swap0 {:#08x} swap1 {:#08x}", g.masks[0], g.masks[1]));
                }
            }
            T_NORMAL_MATRIX => {
                for k in 0..9 {
                    g.lt.normal_matrix[k] = f(b[k]);
                }
            }
            T_TEXTURE_MATRIX => {}
            t => g.lt.port(t, &b[..n as usize]),
        }
        true
    }

    /// Fixed-size GL commands. Returns false if `cmd` is not one.
    pub(super) fn gl_execute(&mut self, cmd: u32, out: &mut dyn Re3Sink) -> bool {
        if Self::gl_arg_count(cmd).is_none() {
            return false;
        }
        self.gl.ensure_init();
        let a = self.args;
        let g = &mut self.gl;
        match cmd {
            T_WINDOW => {
                g.win = [a[0] as i32 & 0x7ff, a[1] as i32 & 0x7ff, a[2] as i32, a[3] as i32];
                g.clip_wid = a[4];
                g.clip_obscured = a[5] & 1;
                g.clip_n = a[6];
                // Pieces: (x1 << 11) | x0, (ytop << 10) | ybottom, inclusive,
                // GL y up. Obscured with 0 pieces: one rectangle (the window
                // clamped to the screen) in the first pair.
                let pairs = if g.clip_n == 0 && g.clip_obscured != 0 { 1 } else { g.clip_n.min(4) };
                for k in 0..pairs as usize {
                    let (w0, w1) = (a[7 + 2 * k], a[8 + 2 * k]);
                    g.clip_rects[k] = [(w0 & 0x7ff) as i32, (w1 & 0x3ff) as i32,
                                       ((w0 >> 11) & 0x7ff) as i32 + 1, ((w1 >> 10) & 0x3ff) as i32 + 1];
                }
                if g.clip_n == 0 && g.clip_obscured != 0 {
                    g.clip_n = 1;
                }
            }
            T_VIEWPORT => g.vp = [f(a[1]), f(a[2]), f(a[3]), f(a[4]), f(a[5]), f(a[6])],
            T_SCISSOR => g.scissor = [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32],
            T_SHADE_MODEL => g.smooth = a[0] & 1,
            T_FRONT_FACE => g.front_ccw = a[0] & 1,
            T_CULL_FRONT => g.cull_front = a[0] & 1,
            T_CULL_BACK => g.cull_back = a[0] & 1,
            T_POLYGON_MODE => g.polymode = a[0],
            T_SWAP_BUFFERS => g.swap = a[0] & 1,
            T_DITHER => g.dither = a[0] & 1,
            T_STIPPLE_ON => {
                g.stipple_on = 1;
                let w = &a[1..33];
                for r in 0..32usize {
                    let i = (31 - r) / 2;
                    let half = |x: u32| if r % 2 == 1 { x & 0xffff } else { x >> 16 };
                    g.stipple[r] = (half(w[i]) << 16) | half(w[16 + i]);
                }
            }
            T_STIPPLE_OFF => g.stipple_on = 0,
            T_BLEND_MODE => g.blend_fast = a[0] & 1,
            T_IRIS_DB => g.lt.on = a[1] & 1,
            T_IRIS_BUFFER | T_IRIS_BGNPOLYGON | T_IRIS_PMV => {}
            T_IRIS_WRITEMASK => g.masks = [a[0] & 0x00ff_ffff; 2],
            T_IRIS_CLEAR => {
                let c = g.color;
                self.gl_clear(c, out);
            }
            // IRIS GL zclear(): clears to the far value GD_ZMAX (0x7FFFFF,
            // Z is signed 24-bit); the word is the Z plane mask (atlantis
            // sends 0x00FFFFFF). Explicit values come through czclear, 0x68A0.
            T_IRIS_ZCLEAR => self.gl_zfill(0x007f_ffff, a[0] & 0x00ff_ffff, out),
            T_IRIS_ENDPOLYGON | T_IRIS_PCLOS => {
                let g = &self.gl;
                if g.vroutine == VR_IRIS_POLYGON && g.polymode == 3 && g.nv >= 3 {
                    let (a, b) = (g.prev[0], g.first);
                    let c = if g.smooth != 0 { b.c } else { g.first.c };
                    self.gl_line(a, b, c, out);
                }
            }
            light::T_LIGHTING | light::T_TWO_SIDED | light::T_NORMALIZE | light::T_NORMALIZE_B
            | light::T_FOG_ON | light::T_MATERIAL_COMMIT => {
                g.lt.command(cmd, &a[..]);
            }
            T_LOGIC_OP => g.logic_op = a[0] & 0xf,
            T_READ_DONE => {}
            T_GET_COLOR => {
                let c = g.color;
                for (k, v) in c.iter().enumerate() {
                    out.shram(READBACK_SHRAM + k, v.to_bits());
                }
            }
            T_GET_NORMAL => {
                let n = g.normal;
                for (k, v) in n.iter().enumerate() {
                    out.shram(READBACK_SHRAM + k, v.to_bits());
                }
            }
            T_IRIS_MOVE => g.pen = 0,
            T_IRIS_GETCPOS => {
                out.shram(READBACK_SHRAM, g.cpos[0] as u32);
                out.shram(READBACK_SHRAM + 1, g.cpos[1] as u32);
                out.shram(READBACK_SHRAM + 2, if g.cpos[2] != 0 { 0x8000_0000 } else { 0 });
            }
            T_GET_RASTERPOS => {
                // Raster position is not tracked yet (unverified layout).
                for k in 0..4 {
                    out.shram(READBACK_SHRAM + k, 0);
                }
            }
            T_FRAGMENT => {
                let (x, y) = (a[0] as i32 - FRAGMENT_BIAS, a[1] as i32 - FRAGMENT_BIAS);
                let c = [f(a[3]), f(a[4]), f(a[5]), f(a[6])];
                self.gl_fragment(x, y, a[2], c, out);
            }
            T_DEPTH_TEST => g.ztest = a[0] & 1,
            T_DEPTH_FUNC => g.zfunc = a[0] & 7,
            T_DEPTH_MASK => g.zmask = a[0] & 0x00ff_ffff,
            T_STENCIL_MODE => g.st.copy_from_slice(&a[..7]),
            T_STENCIL_WMASK => g.st_wmask = a[0] & 0xff,
            T_STENCIL_CONFIG => {}
            T_STENCIL_CLEAR => {
                let (v, m) = ((a[0] & 0xff) << re3::ZBUF_STENCIL_SHIFT, (a[1] & 0xff) << re3::ZBUF_STENCIL_SHIFT);
                self.gl_zfill(v, m, out);
            }
            T_DEPTH_CLEAR => {
                // ITOF|CP CZClear: packed colour 0xAABBGGRR; DATA depth;
                // DATA colour plane mask. libglcore's depth-only Clear sends
                // mask 0; IRIS GL czclear sends the drawn buffer's planes
                // (0x000FFF / 0xFFF000).
                let planes = a[2] & 0x00ff_ffff;
                if planes != 0 {
                    let u = |s: u32| ((a[0] >> s) & 0xff) as f32 / 255.0;
                    let saved = g.masks;
                    self.gl.masks = [planes; 2];
                    self.gl_clear([u(0), u(8), u(16), u(24)], out);
                    self.gl.masks = saved;
                }
                let z = a[1] & 0x00ff_ffff;
                self.gl_zfill(z, 0x00ff_ffff, out);
            }
            // Visual of the window being bound: 4 = 24-bit RGB, 2 = 12-bit
            // RGB, 10 = 12-bit colour index (showmap) (inferred from traces).
            T_MAKECURRENT => match a[0] {
                4 => g.pixfmt = 0,
                2 => g.pixfmt = re3::PIXFMT_RGB12,
                10 => g.pixfmt = re3::PIXFMT_CI12,
                _ => {}
            },
            T_CLEAR_COLOR | T_CLEAR_COLOR_DEPTH => {
                let c = [f(a[0]), f(a[1]), f(a[2]), 1.0];
                self.gl_clear(c, out);
                if cmd == T_CLEAR_COLOR_DEPTH {
                    // CZClear: r; DATA g, b, 0, depth, 0xFFFFFFFF.
                    self.gl_zfill(a[4] & 0x00ff_ffff, 0x00ff_ffff, out);
                }
            }
            c if c & LOADV != 0 => {
                let r = c & 0x1ff;
                if self.gl.vroutine == VR_LLOOP && r == VR_OUTSIDE && self.gl.nv > 1 {
                    // (closing handled by END_LLOOP)
                }
                self.gl.vroutine = r;
                self.gl.nv = 0;
            }
            END_LLOOP => {
                if self.gl.vroutine == VR_LLOOP && self.gl.nv > 1 {
                    let (a, b) = (self.gl.prev[0], self.gl.first);
                    self.gl_line(a, b, b.c, out);
                }
            }
            END_POLYGON => {
                // Line mode: the closing edge of the polygon.
                let g = &self.gl;
                if g.vroutine == VR_POLYGON && g.polymode == 3 && g.nv >= 3 {
                    let (a, b, c) = (g.prev[0], g.first, g.first.c);
                    let flat = if g.smooth != 0 { b.c } else { c };
                    self.gl_line(a, b, flat, out);
                }
            }
            _ => {} // other Begin/End tokens: assembly is keyed by LOADV
        }
        true
    }

    pub(super) fn gl_describe(&self, cmd: u32) -> Option<String> {
        let a = &self.args;
        let g = &self.gl;
        Some(match cmd {
            T_WINDOW => format!("GL_WINDOW origin ({}, {}) {}x{}", g.win[0], g.win[1], g.win[2], g.win[3]),
            T_VIEWPORT => format!("GL_VIEWPORT x {}..{} y {}..{} zscale {} zcenter {}", g.vp[0], g.vp[1], g.vp[2], g.vp[3], g.vp[4], g.vp[5]),
            T_SCISSOR => format!("GL_SCISSOR ({}, {})-({}, {})", g.scissor[0], g.scissor[1], g.scissor[2], g.scissor[3]),
            T_SHADE_MODEL => format!("GL_SHADE_MODEL {}", if g.smooth != 0 { "smooth" } else { "flat" }),
            T_CLEAR_COLOR | T_CLEAR_COLOR_DEPTH => format!("GL_CLEAR rgb ({}, {}, {}) rect {:?}", f(a[0]), f(a[1]), f(a[2]), g.clip_rect()),
            c if c & LOADV != 0 => format!("GL vertex routine {:#x}", c & 0x1ff),
            c if BEGIN_TOKENS.contains(&c) || END_TOKENS.contains(&c) => super::index_label(c),
            T_FRONT_FACE => format!("GL_FRONT_FACE {}", if g.front_ccw != 0 { "CCW" } else { "CW" }),
            T_CULL_FRONT | T_CULL_BACK => format!("GL_CULL front={} back={}", g.cull_front, g.cull_back),
            T_POLYGON_MODE => format!("GL_POLYGON_MODE {}", match g.polymode { 1 => "fill", 2 => "point", 3 => "line", _ => "?" }),
            T_DITHER => format!("GL_DITHER {}", if g.dither != 0 { "on" } else { "off" }),
            T_STIPPLE_ON => format!("GL_POLYGON_STIPPLE on rows[0..4] {:08x} {:08x} {:08x} {:08x}",
                g.stipple[0], g.stipple[1], g.stipple[2], g.stipple[3]),
            T_STIPPLE_OFF => "GL_POLYGON_STIPPLE off".to_string(),
            T_FRAGMENT => format!("GL_FRAGMENT window ({}, {}) z {:#x} rgba ({:.3}, {:.3}, {:.3}, {:.3})",
                a[0] as i32 - FRAGMENT_BIAS, a[1] as i32 - FRAGMENT_BIAS, a[2], f(a[3]), f(a[4]), f(a[5]), f(a[6])),
            T_IRIS_MOVE => "IRISGL_MOVE".to_string(),
            T_IRIS_GETCPOS => format!("IRISGL_GETCPOS -> ({}, {}){}", g.cpos[0], g.cpos[1], if g.cpos[2] != 0 { " invalid" } else { "" }),
            T_GET_COLOR | T_GET_NORMAL | T_GET_RASTERPOS => format!("{} -> shram[{:#x}]", super::index_label(cmd), READBACK_SHRAM),
            T_READ_DONE => "GL_READ_DONE".to_string(),
            T_LOGIC_OP => format!("GL_LOGIC_OP {}", g.logic_op),
            light::T_LIGHTING => format!("GL_LIGHTING {}", if g.lt.on != 0 { "on" } else { "off" }),
            light::T_TWO_SIDED => format!("GL_LIGHT_MODEL_TWO_SIDE {}", g.lt.two_sided),
            light::T_NORMALIZE | light::T_NORMALIZE_B => format!("GL_NORMALIZE {}", g.lt.normalize),
            light::T_FOG_ON => format!("GL_FOG {}", if g.lt.fog_on != 0 { "on" } else { "off" }),
            light::T_MATERIAL_COMMIT => "GL_MATERIAL_COMMIT".to_string(),
            T_BLEND_MODE => format!("GL_BLEND_MODE {}", a[0]),
            T_IRIS_BUFFER => format!("IRISGL_BUFFER {} {}", a[0], a[1]),
            T_IRIS_WRITEMASK => format!("IRISGL_WRITEMASK {:#08x}", a[0]),
            T_IRIS_CLEAR => format!("IRISGL_CLEAR rgb ({:.3}, {:.3}, {:.3}) rect {:?}", g.color[0], g.color[1], g.color[2], g.clip_rect()),
            T_IRIS_ZCLEAR => format!("IRISGL_ZCLEAR {:#x}", a[0]),
            T_IRIS_BGNPOLYGON | T_IRIS_PMV => "IRISGL_BGNPOLYGON".to_string(),
            T_IRIS_ENDPOLYGON | T_IRIS_PCLOS => "IRISGL_ENDPOLYGON".to_string(),
            T_IRIS_DB => format!("IRISGL_LIGHTING {}", a[1]),
            T_DEPTH_TEST | T_DEPTH_FUNC | T_DEPTH_MASK => format!("GL_DEPTH test={} func={} mask={:#08x}", g.ztest, g.zfunc, g.zmask),
            T_STENCIL_MODE => format!("GL_STENCIL on={} ref={} func={} mask={:#x} ops fail={} zfail={} zpass={}",
                g.st[0], g.st[1], g.st[2], g.st[3], g.st[4], g.st[5], g.st[6]),
            T_STENCIL_WMASK => format!("GL_STENCIL_WRITEMASK {:#x}", g.st_wmask),
            T_STENCIL_CONFIG => format!("GL_STENCIL_BITS {}", a[0]),
            T_STENCIL_CLEAR => format!("GL_CLEAR_STENCIL {} mask {:#x} rect {:?}", a[0], a[1], g.clip_rect()),
            T_DEPTH_CLEAR => format!("GL_CLEAR_DEPTH {:#x} colour {:#010x} planes {:#08x} rect {:?}", a[1], a[0], a[2], g.clip_rect()),
            T_SWAP_BUFFERS => format!("GL_SWAP_BUFFERS state {} (draw mask {:#08x})", g.swap, g.masks[g.swap as usize & 1]),
            T_MAKECURRENT => format!("GL_MAKECURRENT mode {} -> {}", a[0], match g.pixfmt {
                re3::PIXFMT_RGB12 => "RGB12",
                re3::PIXFMT_CI12 => "CI12",
                _ => "RGB24",
            }),
            _ => return None,
        })
    }

    /// Primitive assembly for one transformed vertex.
    fn gl_vertex(&mut self, v: Wv, out: &mut dyn Re3Sink, done: &mut Option<&mut dyn FnMut(String)>) {
        let g = &mut self.gl;
        let n = g.nv;
        g.nv += 1;
        let (p0, p1, p2) = (g.prev[0], g.prev[1], g.prev[2]);
        let first = g.first;
        if n == 0 {
            g.first = v;
        }
        // prev[0] = newest before v.
        g.prev = [v, p0, p1];
        let flat = |c: Wv| c;
        // Edge masks (bit 0: v0-v1, 1: v1-v2, 2: v2-v0) mark real polygon
        // edges for GL_LINE polygon mode (no quad/polygon diagonals).
        match g.vroutine {
            VR_POINTS => self.gl_point(v, v.c, out),
            VR_LINES => if n % 2 == 1 { self.gl_line(p0, v, v.c, out) },
            VR_LSTRIP | VR_LLOOP => if n >= 1 { self.gl_line(p0, v, v.c, out) },
            VR_TRIANGLES => if n % 3 == 2 { self.gl_tri([p1, p0, v], flat(v), 0b111, out, done) },
            VR_TSTRIP => if n >= 2 {
                let t = if n % 2 == 0 { [p1, p0, v] } else { [p0, p1, v] };
                self.gl_tri(t, flat(v), 0b111, out, done)
            },
            VR_TFAN => if n >= 2 { self.gl_tri([first, p0, v], flat(v), 0b111, out, done) },
            VR_QUADS => if n % 4 == 3 {
                self.gl_tri([p2, p1, p0], flat(v), 0b011, out, done);
                self.gl_tri([p2, p0, v], flat(v), 0b110, out, done);
            },
            VR_QSTRIP => if n >= 3 && n % 2 == 1 {
                // Quad (v0, v1, v3, v2) of the pair (p2, p1) + (p0, v).
                self.gl_tri([p2, p1, v], flat(v), 0b011, out, done);
                self.gl_tri([p2, v, p0], flat(v), 0b110, out, done);
            },
            VR_POLYGON | VR_IRIS_POLYGON => if n >= 2 {
                let edges = 0b010 | if n == 2 { 0b001 } else { 0 };
                self.gl_tri([first, p0, v], flat(first), edges, out, done)
            },
            _ => {}
        }
    }

    /// RE3 state for GL drawing into the colour planes.
    fn gl_setup(&mut self, out: &mut dyn Re3Sink) {
        out.reg(re3::REG_FUNC, self.gl.logic_op);
        if let Some((src, dst)) = self.gl.blend_func() {
            out.op(re3::RE3_OP_BLEND, (1 | (src << 1) | (dst << 4)) as u64);
            self.gl.blend_sent = 1;
        }
        out.reg(re3::REG_NOPUP, 1);
        out.reg(re3::REG_UAUXDATA, 0);
        out.reg(re3::REG_AUXMASK, 0);
        if self.gl.wid_clip() {
            // More visible pieces than 0x1E5 carries: draw only where the
            // CID planes (painted by Xsgi, 2D_CID_WRITE) hold this window's
            // WID; 4-bit compare (FBOPTION bit 0) (unverified: that the
            // kernel's WID numbering is the DDX's CID numbering).
            out.reg(re3::REG_CURWID, self.gl.clip_wid & 0xf);
            out.reg(re3::REG_FBOPTION, 1);
            out.reg(re3::REG_ENABWID, 1);
            self.gl.wid_sent = 1;
        }
        // Stipple patterns are span-relative (see gl_stipple_pattern).
        out.reg(re3::REG_ALIGNPAT, 0);
        let mask = self.gl.masks[self.gl.swap as usize & 1] & self.gl.colormask;
        out.reg(re3::REG_PIXMASK, mask & 0x00ff_ffff);
        if self.gl.pixfmt != 0 {
            out.op(re3::RE3_OP_PIXFMT, self.gl.pixfmt as u64);
            out.reg(re3::REG_ENABDITH, self.gl.dither);
        }
        if self.gl_zs_active() {
            let g = &self.gl;
            let zctl = g.ztest | (g.zfunc << 1) | (g.zmask << 8);
            let st = &g.st;
            let stencil = (st[0] & 1) as u64 | ((st[2] as u64 & 7) << 1) | ((st[1] as u64 & 0xff) << 4)
                | ((st[3] as u64 & 0xff) << 12) | ((g.st_wmask as u64 & 0xff) << 20)
                | ((st[4] as u64 & 0xf) << 28) | ((st[5] as u64 & 0xf) << 32) | ((st[6] as u64 & 0xf) << 36);
            out.op(re3::RE3_OP_ZCTL, zctl as u64);
            out.op(re3::RE3_OP_STENCIL, stencil);
            self.gl.zs_sent = 1;
        }
        // GL clips in software (window ∩ scissor); keep the RE3 scissor at
        // the full screen, which is also what the 2D path assumes.
        out.reg(re3::REG_XMIN, 0);
        out.reg(re3::REG_XMAX, re3::FB_W as u32 - 1);
        out.reg(re3::REG_YMIN, 0);
        out.reg(re3::REG_YMAX, re3::FB_H as u32 - 1);
    }

    /// Undo GL-only RE3 state so 2D drawing sees what it expects.
    fn gl_done(&mut self, out: &mut dyn Re3Sink) {
        if self.gl.pixfmt != 0 {
            out.op(re3::RE3_OP_PIXFMT, 0);
            out.reg(re3::REG_ENABDITH, 0);
        }
        if self.gl.zs_sent != 0 {
            out.op(re3::RE3_OP_ZCTL, 0);
            out.op(re3::RE3_OP_STENCIL, 0);
            self.gl.zs_sent = 0;
        }
        if self.gl.blend_sent != 0 {
            out.op(re3::RE3_OP_BLEND, 0);
            self.gl.blend_sent = 0;
        }
        if self.gl.logic_op != re3::ROP_COPY {
            out.reg(re3::REG_FUNC, re3::ROP_COPY);
        }
        if self.gl.wid_sent != 0 {
            out.reg(re3::REG_ENABWID, 0);
            out.reg(re3::REG_FBOPTION, 0);
            self.gl.wid_sent = 0;
        }
    }

    /// One software-pipeline fragment (window coordinates, GL y up) through
    /// the hardware per-pixel path: Z/stencil, blend, logic op, masks.
    fn gl_fragment(&mut self, x: i32, y: i32, z: u32, c: [f32; 4], out: &mut dyn Re3Sink) {
        let (sx, sy) = (self.gl.win[0] + x, self.gl.win[1] + y);
        if !self.gl.pixel_visible(sx, sy) {
            return;
        }
        self.gl_setup(out);
        let q = |v: f32| ((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u32) << 11;
        let rgb = to_fixed(c, self.gl.cmax());
        self.gl_color_regs(rgb, out);
        out.reg(re3::REG_DR, 0);
        out.reg(re3::REG_DG, 0);
        out.reg(re3::REG_DB, 0);
        if self.gl_zs_active() {
            out.reg(re3::REG_Z, z & 0x00ff_ffff);
        }
        if self.gl.blend_func().is_some() {
            out.op(re3::RE3_OP_ALPHA, q(c[3]) as u64);
        }
        self.gl_span_ir(sx, sy, 1, re3::IR_SHADED, None, out);
        self.gl_done(out);
    }

    /// Depth test or stencil test active: spans must iterate Z per pixel.
    fn gl_zs_active(&self) -> bool {
        self.gl.ztest != 0 || self.gl.st[0] & 1 != 0
    }

    /// Fill the Z/stencil buffer over the window ∩ scissor rectangle.
    fn gl_zfill(&mut self, value: u32, mask: u32, out: &mut dyn Re3Sink) {
        let (rects, n) = self.gl.visible_rects();
        for r in &rects[..n] {
            let rect = (r[0] as u64) | ((r[1] as u64) << 16) | ((r[2] as u64) << 32) | ((r[3] as u64) << 48);
            out.op(re3::RE3_OP_ZFILL_A, rect);
            out.op(re3::RE3_OP_ZFILL_B, value as u64 | ((mask as u64) << 32));
        }
    }

    fn gl_color_regs(&mut self, c: [u32; 3], out: &mut dyn Re3Sink) {
        out.reg(re3::REG_R, c[0]);
        out.reg(re3::REG_G, c[1]);
        out.reg(re3::REG_B, c[2]);
    }

    /// One character bitmap (T_IRIS_CHAR*) at the character position.
    fn gl_glyph(&mut self, tok: u32, b: &[u32], out: &mut dyn Re3Sink) {
        let g = &mut self.gl;
        let hi = |w: u32| (w >> 16) as i16 as i32;
        let lo = |w: u32| w as i16 as i32;
        let (w, h) = (hi(b[0]), lo(b[0]));
        let (xorig, yorig) = (hi(b[1]), lo(b[1]));
        let (cx, cy, invalid) = (g.cpos[0], g.cpos[1], g.cpos[2]);
        g.cpos[0] += hi(b[2]);
        g.cpos[1] += lo(b[2]);
        if invalid != 0 || w <= 0 || h <= 0 {
            return;
        }
        let data = &b[4..];
        let rows: Vec<u32> = if tok == T_IRIS_CHAR32 {
            data.to_vec()
        } else {
            let skip = (b[3] & 1 == 0) as usize;
            data.iter().flat_map(|&v| [(v & 0xffff) << 16, v & 0xffff_0000]).skip(skip).collect()
        };
        let (x0, top) = (g.win[0] + cx - xorig, g.win[1] + cy - yorig + h - 1);
        let wmask = if w >= 32 { u32::MAX } else { !(u32::MAX >> w) };
        let c = to_fixed(g.cpos_color, g.cmax());
        self.gl_setup(out);
        self.gl_color_regs(c, out);
        for (r, &bits) in rows.iter().take(h as usize).enumerate() {
            let bits = bits & wmask;
            if bits == 0 {
                continue;
            }
            let y = top - r as i32;
            let (pieces, np) = self.gl.span_pieces(y, x0, x0 + w);
            for &(a, e) in &pieces[..np] {
                let sh = (a - x0) as u32;
                self.span(a, y, (e - a) as u32, Some(if sh >= 32 { 0 } else { bits << sh }), out);
            }
        }
        self.gl_done(out);
    }

    fn gl_clear(&mut self, c: [f32; 4], out: &mut dyn Re3Sink) {
        let (rects, n) = self.gl.visible_rects();
        if n == 0 {
            return;
        }
        self.gl_setup(out);
        self.gl_color_regs(to_fixed(c, self.gl.cmax()), out);
        for r in &rects[..n] {
            for y in r[1]..r[3] {
                self.span(r[0], y, (r[2] - r[0]) as u32, None, out);
            }
        }
        self.gl_done(out);
    }

    fn gl_point(&mut self, v: Wv, color: [f32; 4], out: &mut dyn Re3Sink) {
        if v.ok == 0 {
            return;
        }
        let (x, y) = (v.x.floor() as i32, v.y.floor() as i32);
        if !self.gl.pixel_visible(x, y) {
            return;
        }
        self.gl_setup(out);
        self.gl_color_regs(to_fixed(color, self.gl.cmax()), out);
        self.span(x, y, 1, None, out);
        self.gl_done(out);
    }

    /// 1-pixel line (DDA along the major axis) in one colour (smooth lines
    /// not yet).
    fn gl_line(&mut self, a: Wv, b: Wv, color: [f32; 4], out: &mut dyn Re3Sink) {
        if a.ok == 0 || b.ok == 0 {
            return;
        }
        self.gl_setup(out);
        self.gl_color_regs(to_fixed(color, self.gl.cmax()), out);
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let steps = dx.abs().max(dy.abs()).ceil().max(1.0) as i32;
        let (sx, sy) = (dx / steps as f32, dy / steps as f32);
        // GL diamond-exit rule approximated: draw steps pixels, excluding the last.
        for i in 0..steps {
            let x = (a.x + sx * i as f32).floor() as i32;
            let y = (a.y + sy * i as f32).floor() as i32;
            if self.gl.pixel_visible(x, y) {
                self.span(x, y, 1, None, out);
            }
        }
        self.gl_done(out);
    }

    /// Scan-convert one triangle (window coordinates, GL y up; pixel centres
    /// at +0.5, top-left style fill: a pixel is in if its centre is in
    /// [left, right) on rows whose centre is in [ymin, ymax)).
    /// `prov` = the provoking vertex (flat shading colour); `edges` = which
    /// edges are polygon edges (bit 0: v0-v1, 1: v1-v2, 2: v2-v0) for
    /// GL_LINE polygon mode.
    fn gl_tri(&mut self, mut v: [Wv; 3], mut prov: Wv, edges: u8, out: &mut dyn Re3Sink,
              done: &mut Option<&mut dyn FnMut(String)>) {
        if v.iter().any(|v| v.ok == 0) {
            return;
        }
        let area = (v[1].x - v[0].x) * (v[2].y - v[0].y) - (v[2].x - v[0].x) * (v[1].y - v[0].y);
        if area.abs() < 1e-6 {
            return;
        }
        // Face culling: window-space counter-clockwise (GL y up) is front
        // when FrontFace is CCW.
        let front = (area > 0.0) == (self.gl.front_ccw != 0);
        if (front && self.gl.cull_front != 0) || (!front && self.gl.cull_back != 0) {
            if let Some(d) = done.as_mut() {
                d(format!("GL triangle culled ({} face)", if front { "front" } else { "back" }));
            }
            return;
        }
        // Two-sided lighting: back-facing polygons use the back colours.
        if !front && self.gl.lt.two_sided != 0 && self.gl.lt.on != 0 {
            for x in v.iter_mut() {
                x.c = x.cb;
            }
            prov.c = prov.cb;
        }
        let flat = Some(prov.c);
        match self.gl.polymode {
            2 => {
                for k in 0..3 {
                    let c = if self.gl.smooth != 0 { v[k].c } else { flat.unwrap_or(v[2].c) };
                    self.gl_point(v[k], c, out);
                }
                return;
            }
            3 => {
                for k in 0..3 {
                    if edges & (1 << k) != 0 {
                        let (a, b) = (v[k], v[(k + 1) % 3]);
                        let c = if self.gl.smooth != 0 { b.c } else { flat.unwrap_or(v[2].c) };
                        self.gl_line(a, b, c, out);
                    }
                }
                return;
            }
            _ => {}
        }
        let smooth = self.gl.smooth != 0;
        let clip = self.gl.clip_rect();
        if let Some(d) = done.as_mut() {
            d(format!("GL triangle ({:.1}, {:.1}) ({:.1}, {:.1}) ({:.1}, {:.1}) {} clip {:?}",
                v[0].x, v[0].y, v[1].x, v[1].y, v[2].x, v[2].y, if smooth { "smooth" } else { "flat" }, clip));
        }
        // Colour plane equations (per channel, in 0..255 units).
        let det = area;
        let grad = |k: usize| {
            let (c0, c1, c2) = (v[0].c[k] * 255.0, v[1].c[k] * 255.0, v[2].c[k] * 255.0);
            let dx = ((c1 - c0) * (v[2].y - v[0].y) - (c2 - c0) * (v[1].y - v[0].y)) / det;
            let dy = ((c2 - c0) * (v[1].x - v[0].x) - (c1 - c0) * (v[2].x - v[0].x)) / det;
            (c0, dx, dy)
        };
        let planes = [grad(0), grad(1), grad(2)];
        // Window z plane (z already in depth-buffer units).
        let zdx = ((v[1].z - v[0].z) * (v[2].y - v[0].y) - (v[2].z - v[0].z) * (v[1].y - v[0].y)) / det;
        let zdy = ((v[2].z - v[0].z) * (v[1].x - v[0].x) - (v[1].z - v[0].z) * (v[2].x - v[0].x)) / det;
        let zs = self.gl_zs_active();
        // SHADED spans iterate colour and Z per pixel: needed for smooth
        // shading and for any depth/stencil work (flat colour = zero deltas).
        let blend = self.gl.blend_func().is_some();
        let iter = smooth || zs || blend;
        let aplane = grad(3);
        let flat_c = flat.unwrap_or(v[2].c);
        let cmax = self.gl.cmax();
        self.gl_setup(out);
        if iter {
            out.reg(re3::REG_DX, 1 << 14);
            out.reg(re3::REG_DY, 0);
        } else {
            self.gl_color_regs(to_fixed(flat_c, self.gl.cmax()), out);
        }
        let ymin = v.iter().map(|v| v.y).fold(f32::MAX, f32::min);
        let ymax = v.iter().map(|v| v.y).fold(f32::MIN, f32::max);
        let y0 = ((ymin - 0.5).ceil() as i32).max(clip[1]);
        let y1 = ((ymax - 0.5).ceil() as i32).min(clip[3]);
        for y in y0..y1 {
            let yc = y as f32 + 0.5;
            // Intersections of the row centre with the three edges.
            let mut xs = [0.0f32; 2];
            let mut nx = 0;
            for i in 0..3 {
                let (p, q) = (v[i], v[(i + 1) % 3]);
                let (lo, hi) = if p.y < q.y { (p, q) } else { (q, p) };
                if yc >= lo.y && yc < hi.y && nx < 2 {
                    xs[nx] = lo.x + (yc - lo.y) * (hi.x - lo.x) / (hi.y - lo.y);
                    nx += 1;
                }
            }
            if nx < 2 {
                continue;
            }
            let (xl, xr) = if xs[0] < xs[1] { (xs[0], xs[1]) } else { (xs[1], xs[0]) };
            // Clip the row to the window's visible pieces: each piece is its
            // own RE3 span with iterators started at its left end.
            let (pieces, np) = self.gl.span_pieces(y, (xl - 0.5).ceil() as i32, (xr - 0.5).ceil() as i32);
            for &(x0, x1) in &pieces[..np] {
                let n = (x1 - x0) as u32;
                if iter {
                    let px = x0 as f32 + 0.5;
                    let mut start = [0u32; 3];
                    let mut step = [0u32; 3];
                    for k in 0..3 {
                        let m = if k == 0 { cmax } else { 1.0 };
                        if !smooth {
                            start[k] = to_fixed(flat_c, cmax)[k];
                            continue;
                        }
                        let (c0, dx, dy) = planes[k];
                        let s = (c0 + dx * (px - v[0].x) + dy * (yc - v[0].y)).clamp(0.0, 255.0 * m);
                        let e = (s + dx * (n - 1) as f32).clamp(0.0, 255.0 * m);
                        let d = if n > 1 { (e - s) / (n - 1) as f32 } else { 0.0 };
                        start[k] = (s * 2048.0) as u32;
                        step[k] = ((d * 2048.0) as i32) as u32;
                    }
                    if blend {
                        // Source alpha for the blender (8.11), flat or smooth.
                        let (a0, adx, ady) = aplane;
                        let (sa, da) = if smooth {
                            let s = (a0 + adx * (px - v[0].x) + ady * (yc - v[0].y)).clamp(0.0, 255.0);
                            let e = (s + adx * (n - 1) as f32).clamp(0.0, 255.0);
                            (s, if n > 1 { (e - s) / (n - 1) as f32 } else { 0.0 })
                        } else {
                            (flat_c[3].clamp(0.0, 1.0) * 255.0, 0.0)
                        };
                        let step = ((da * 2048.0) as i32) as u32 as u64;
                        out.op(re3::RE3_OP_ALPHA, ((sa * 2048.0) as u32 as u64) | (step << 32));
                    }
                    if zs {
                        // RE3 Z iterator: signed 24-bit integer Z, 24.14 step
                        // (IRIS GL uses -0x800000..0x7FFFFF, OpenGL 0..0x7FFFFF).
                        let z = (v[0].z + zdx * (px - v[0].x) + zdy * (yc - v[0].y)).clamp(-8388608.0, 8388607.0);
                        let dz = (zdx as f64 * 16384.0).round() as i64;
                        out.reg(re3::REG_Z, (z as i32 as u32) & 0x00ff_ffff);
                        out.reg(re3::REG_DZI, ((dz >> 14) as u32) & 0x00ff_ffff);
                        out.reg(re3::REG_DZF, (dz as u32) & 0x3fff);
                    }
                    out.reg(re3::REG_R, start[0]);
                    out.reg(re3::REG_G, start[1]);
                    out.reg(re3::REG_B, start[2]);
                    out.reg(re3::REG_DR, step[0] & 0x00ff_ffff);
                    out.reg(re3::REG_DG, step[1] & 0x000f_ffff);
                    out.reg(re3::REG_DB, step[2] & 0x000f_ffff);
                    let pat = self.gl_stipple_pattern(x0, y);
                    self.gl_span_ir(x0, y, n, re3::IR_SHADED, pat, out);
                } else {
                    let pat = self.gl_stipple_pattern(x0, y);
                    self.span(x0, y, n, pat, out);
                }
            }
        }
        if iter {
            // FLAT spans step the colour iterators too: leave them at zero.
            if zs {
                out.reg(re3::REG_DZI, 0);
                out.reg(re3::REG_DZF, 0);
            }
            out.reg(re3::REG_DR, 0);
            out.reg(re3::REG_DG, 0);
            out.reg(re3::REG_DB, 0);
            out.reg(re3::REG_DX, 0);
        }
        self.gl_done(out);
    }

    /// Polygon stipple pattern for a span starting at screen (x, y): the
    /// window-aligned row word rotated so its bit 31 is pixel x (RE3 steps
    /// the pattern once per pixel from the span start, repeating every 32).
    fn gl_stipple_pattern(&self, x: i32, y: i32) -> Option<u32> {
        let g = &self.gl;
        if g.stipple_on == 0 {
            return None;
        }
        let row = g.stipple[((y - g.win[1]) & 31) as usize];
        Some(row.rotate_left(((x - g.win[0]) & 31) as u32))
    }

    fn gl_span_ir(&mut self, x: i32, y: i32, n: u32, ir: u32, pattern: Option<u32>, out: &mut dyn Re3Sink) {
        let want = pattern.is_some() as u32;
        if self.pattern_on != want {
            out.reg(re3::REG_ENABPAT, want);
            self.pattern_on = want;
        }
        if let Some(p) = pattern {
            out.reg(re3::REG_PATH, p >> 16);
            out.reg(re3::REG_PATL, p & 0xffff);
        }
        out.reg(re3::REG_X, x as u32);
        out.reg(re3::REG_Y, y as u32);
        out.reg(re3::REG_NUMPIX, n);
        out.reg(re3::REG_IR, ir);
    }
}
