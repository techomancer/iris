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
/// sboxf(x1, y1, x2, y2): screen-aligned filled box (gl_i_sboxf): x1 on
/// the token, y1, x2, y2 on DATA, f32; sboxfi is ITOF (0x4053 / 0x41DF).
/// The corners are transformed, the box between them filled axis-aligned.
/// twilight draws its stars with it.
/// User clip planes (__glExpEnableClipPlanes / __glExpPassClipPlanes):
/// 0x02E = enabled, then 0x02E = plane (0..5); 0x02F = plane, DATA a, b,
/// c, d in eye space (gc->state.transform.eyeClipPlanes).
pub const T_CLIP_PLANE_ENABLE: u32 = 0x02e;
pub const T_CLIP_PLANE: u32 = 0x02f;
/// Pixel zoom for pixel writes: token x3 = 0.0, zoom x, zoom y (meaning
/// of the first word unverified; lrectwrite sends 0, 1, 1 at zoom 1).
pub const T_PIXEL_ZOOM: u32 = 0x0bb;
pub const T_IRIS_SBOXF: u32 = 0x053;
/// swaptmesh (gl_i_swaptmesh): swap the two vertices a tmesh (bgntmesh =
/// 0x046, LOADV|0x047) keeps; the next vertex makes a triangle with them.
pub const T_IRIS_SWAPTMESH: u32 = 0x04b;
pub const T_IRIS_SBOXFI: u32 = 0x4053;
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
/// Pixel read setup (__glExpReadPixelsKDMA, Fetch, ReadColor): read setup
/// 0x0A7 / 0x0A8 = 0 and read mode 0x0BC (1 before a pixel DMA read, 0
/// before READ_RECT). Meaning unknown; the HLE ignores them.
pub const T_READ_SETUP_A: u32 = 0x0a7;
pub const T_READ_SETUP_B: u32 = 0x0a8;
pub const T_READ_MODE: u32 = 0x0bc;
/// Read source (__glExpSetReadBuffer): two words to the token, kind and
/// buffer: 0, 0 = front; 0, 1 = back; 1, n = with aux buffers; 2, 0 = the
/// Z buffer (depth and stencil reads).
pub const T_READ_BUFFER: u32 = 0x10a;
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
    /// 0 when the vertex is behind the eye (w <= 0): it has no window
    /// position and its primitive must be clipped (gl_poly / gl_line).
    pub ok: u32,
    /// Clip planes the vertex is outside of (bit i = plane i), computed once
    /// at transform; 0 for vertices made by clipping or in window space.
    pub oc: u32,
    /// Clip-space position (frustum clipping) and eye-space position (user
    /// clip planes). Both are linear in the object position, so clipping
    /// interpolates them along an edge exactly.
    pub h: [f32; 4],
    pub e: [f32; 4],
}

/// Clip planes: 0..5 the view volume -w <= x, y, z <= w (GL and IRIS GL
/// alike; their different Z ranges come after clipping, in the viewport
/// transform), 6..11 the user planes (0x02E / 0x02F, eye space).
const CLIP_PLANES: usize = 12;

/// Most visible rectangles a 0x1E5 clip decomposes into (visible_rects).
const MAX_VIS: usize = 32;

/// Vertex buffer (GlState::vb): strips, fans and tmeshes cycle through the
/// first RING slots, never overwriting the vertices they still refer to
/// (the last three and a fan's first); polygons fill all VB slots and are
/// drawn whole at their end. Clipping appends new vertices in a scratch
/// area addressed after the buffer (indices VB..), so polygons are clipped
/// as index lists.
const VB: usize = 32;
const RING: usize = 8;
/// Largest polygon after clipping: each plane adds at most one vertex.
const MAXP: usize = VB + CLIP_PLANES;
const SCRATCH: usize = 2 * CLIP_PLANES;
const ALL_EDGES: u64 = u64::MAX;

/// Point on a-b at the crossing of a plane where the signed distances are
/// da and db: positions and colours interpolated, window position still to
/// be computed (GlState::project).
fn clip_cross(a: &Wv, b: &Wv, da: f32, db: f32) -> Wv {
    let t = da / (da - db);
    let l = |x: f32, y: f32| x + (y - x) * t;
    let mut v = *a;
    v.oc = 0;
    for k in 0..4 {
        v.h[k] = l(a.h[k], b.h[k]);
        v.e[k] = l(a.e[k], b.e[k]);
        v.c[k] = l(a.c[k], b.c[k]);
        v.cb[k] = l(a.cb[k], b.cb[k]);
    }
    v
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
    /// Primitive assembly: the vertex routine (LOADV), vertices since it
    /// started, the vertex buffer, the three newest earlier vertices (pv[0]
    /// newest; swaptmesh swaps pv[0] and pv[1]), the first vertex (fans,
    /// line loops), the ring's next slot, a polygon's vertex count and
    /// whether it continues an earlier full buffer.
    vroutine: u32,
    nv: u32,
    vb: [Wv; VB],
    pv: [u32; 3],
    vfirst: u32,
    vhead: u32,
    pn: u32,
    pcont: u32,
    /// Character position (cmov): window x, y; 0 valid, 1 clipped.
    cpos: [i32; 3],
    /// Colour latched by cmov for the characters drawn there.
    cpos_color: [f32; 4],
    /// move / draw pen: 0 = next 0x05D point is a move.
    pen: u32,
    pen_at: Wv,
    /// Triangle strip / tmesh winding: toggles per triangle and per swap.
    tflip: u32,
    /// Pixel zoom (0x0BB): x, y; 0 = 1.
    pzoom: [f32; 2],
    /// User clip planes (eye space) and their enable bits.
    uclip: [[f32; 4]; 6],
    uclip_on: u32,
    /// Read source from 0x10A: kind, buffer (0, 0 = front).
    read_src: [u32; 2],
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
        let h = [clip(0), clip(1), clip(2), clip(3)];
        let m = &self.mv;
        let eye = |r: usize| m[r] * v[0] + m[4 + r] * v[1] + m[8 + r] * v[2] + m[12 + r] * v[3];
        let e = [eye(0), eye(1), eye(2), eye(3)];
        let mut out = Wv { c: self.color, cb: self.color, h, e, ..Default::default() };
        out.oc = self.outcode(&out);
        self.project(out)
    }

    /// Ring slot for the next strip / fan / tmesh vertex.
    fn vb_alloc(&mut self) -> usize {
        for i in 0..RING {
            let s = (self.vhead as usize + i) % RING;
            let s32 = s as u32;
            if !self.pv.contains(&s32) && s32 != self.vfirst {
                self.vhead = s32 + 1;
                return s;
            }
        }
        0
    }

    /// Signed distance of `v` to clip plane `i` (>= 0 inside).
    fn clip_dist(&self, v: &Wv, i: usize) -> f32 {
        let h = &v.h;
        match i {
            0 => h[3] + h[0],
            1 => h[3] - h[0],
            2 => h[3] + h[1],
            3 => h[3] - h[1],
            4 => h[3] + h[2],
            5 => h[3] - h[2],
            _ => {
                let p = &self.uclip[i - 6];
                p[0] * v.e[0] + p[1] * v.e[1] + p[2] * v.e[2] + p[3] * v.e[3]
            }
        }
    }

    /// Planes in use: the view volume plus the enabled user planes.
    fn clip_mask(&self) -> u32 {
        0x3f | ((self.uclip_on & 0x3f) << 6)
    }

    /// Bit i set when `v` is outside plane i.
    fn outcode(&self, v: &Wv) -> u32 {
        let mask = self.clip_mask();
        (0..CLIP_PLANES).filter(|&i| mask & (1 << i) != 0 && self.clip_dist(v, i) < 0.0)
            .fold(0, |o, i| o | (1 << i))
    }

    /// Window position of `v` from its clip position (`ok` = 0 if w <= 0).
    fn project(&self, mut out: Wv) -> Wv {
        let [cx, cy, cz, cw] = out.h;
        out.ok = 0;
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
    /// rectangles 0x1E5 can carry, or an obscured window sent with no piece
    /// list (Gr2ValidateClip then sends only the bounding box; twilight
    /// drawing the root window under every desktop window, IRIX 6.5.22).
    fn wid_clip(&self) -> bool {
        self.clip_n > 4 || (self.clip_n == 0 && self.clip_obscured != 0)
    }

    /// Visible rectangles (screen, exclusive): clip_rect() intersected with
    /// the window's visible region, as y bands of the piece list's XOR
    /// (see span_pieces). Up to 7 bands of up to 4 intervals.
    fn visible_rects(&self) -> ([[i32; 4]; MAX_VIS], usize) {
        let r = self.clip_rect();
        let mut out = [[0i32; 4]; MAX_VIS];
        // 0 pieces and not obscured = the whole window; the WID test cases
        // are bounded by the window (the box the kernel sends).
        if self.clip_n == 0 || self.wid_clip() {
            let r = if self.clip_n == 0 && self.clip_obscured != 0 {
                let b = self.clip_rects[0];
                [r[0].max(b[0]), r[1].max(b[1]), r[2].min(b[2]), r[3].min(b[3])]
            } else {
                r
            };
            if r[0] < r[2] && r[1] < r[3] {
                out[0] = r;
                return (out, 1);
            }
            return (out, 0);
        }
        // Band edges: every piece's top and bottom inside the clip rect.
        let mut ys = [0i32; 10];
        let mut ny = 0;
        ys[ny] = r[1];
        ny += 1;
        ys[ny] = r[3];
        ny += 1;
        for p in &self.clip_rects[..self.clip_n.min(4) as usize] {
            for y in [p[1], p[3]] {
                if y > r[1] && y < r[3] {
                    ys[ny] = y;
                    ny += 1;
                }
            }
        }
        let ys = &mut ys[..ny];
        ys.sort_unstable();
        let mut n = 0;
        for band in ys.windows(2) {
            let (y0, y1) = (band[0], band[1]);
            if y0 >= y1 {
                continue;
            }
            let (pieces, np) = self.span_pieces(y0, r[0], r[2]);
            for &(a, b) in &pieces[..np] {
                if n < MAX_VIS {
                    out[n] = [a, y0, b, y1];
                    n += 1;
                }
            }
        }
        (out, n)
    }

    /// The visible pieces of row `y` between x0 and x1 (exclusive), left to
    /// right. A pixel is visible when it lies in an odd number of the 0x1E5
    /// pieces (XOR). Xsgi's expValidateClip sends either disjoint visible
    /// rectangles (XOR = union) or, when the region is the window minus one
    /// rectangle, [whole window, hole] with wid = 0 (HQ2.h 0x1E5): atlantis
    /// with ideas over the middle of its side. The union let atlantis draw
    /// over ideas.
    fn span_pieces(&self, y: i32, x0: i32, x1: i32) -> ([(i32, i32); 4], usize) {
        let r = self.clip_rect();
        let mut out = [(0, 0); 4];
        if y < r[1] || y >= r[3] {
            return (out, 0);
        }
        let (x0, x1) = (x0.max(r[0]), x1.min(r[2]));
        if x0 >= x1 {
            return (out, 0);
        }
        if self.clip_n == 0 || self.wid_clip() {
            let (rects, n) = self.visible_rects();
            if n == 1 && y >= rects[0][1] && y < rects[0][3] {
                let (a, b) = (x0.max(rects[0][0]), x1.min(rects[0][2]));
                if a < b {
                    out[0] = (a, b);
                    return (out, 1);
                }
            }
            return (out, 0);
        }
        // Edges of the pieces covering this row; sorted, they pair up into
        // the intervals of odd coverage (equal edges cancel).
        let mut xs = [0i32; 8];
        let mut nx = 0;
        for p in &self.clip_rects[..self.clip_n.min(4) as usize] {
            if y >= p[1] && y < p[3] && p[0] < p[2] {
                xs[nx] = p[0];
                xs[nx + 1] = p[2];
                nx += 2;
            }
        }
        let xs = &mut xs[..nx];
        xs.sort_unstable();
        let mut k = 0;
        for pair in xs.chunks(2) {
            let (a, b) = (pair[0].max(x0), pair[1].min(x1));
            if a < b && k < 4 {
                out[k] = (a, b);
                k += 1;
            }
        }
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
        T_PIXEL_ZOOM => Some(3),
        T_IRIS_CHAR16_TALL | T_IRIS_CHAR32 => Some(21),
        T_NORMAL_MATRIX => Some(9),
        T_NORMAL | T_IRIS_NORMAL => Some(3),
        T_COLOR_WRITEMASK | T_READ_BUFFER => Some(2),
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
    /// Trace notes: owner of the live state before the last GE_HQMSAV
    /// (u32::MAX = none), and why the last restore was rejected (0 = it
    /// loaded; 1 length, 2 magic, 3 GlState size).
    pub(super) prev_owner: u32,
    pub(super) restore_reject: u32,
}

impl Hq2Engine {
    /// 0x1E1 save main: the image the kernel now reads from HQ2_GEDMA.
    pub(super) fn gl_cx_save(&mut self, out: &mut dyn Re3Sink) {
        out.gedma_out(&self.gl_ctx.save);
    }

    /// How 0x0AC reads the current read source (0x10A) in this window's
    /// pixel format. Double-buffered 12-bit: after the swap to state s the
    /// front is bank s and the back bank s ^ 1 (the inverse of GL_BACK's
    /// write masks 0xFFF000 / 0x000FFF for states 0 / 1).
    fn gl_read_decode(&self) -> super::ReadDecode {
        use super::ReadDecode;
        let g = &self.gl;
        if g.read_src[0] == 2 {
            return ReadDecode::Depth;
        }
        let bank = (g.swap ^ (g.read_src[1] & 1)) & 1;
        match g.pixfmt {
            re3::PIXFMT_RGB12 => ReadDecode::Rgb12 { bank },
            re3::PIXFMT_CI12 => ReadDecode::Ci12 { bank },
            _ => ReadDecode::Rgb24,
        }
    }

    pub(super) fn gl_read_desc(&self) -> String {
        format!("src {} {} {:?} window ({}, {})", self.gl.read_src[0], self.gl.read_src[1],
            self.gl_read_decode(), self.gl.win[0], self.gl.win[1])
    }

    /// 0x0AC pixel DMA read (lrectread, glReadPixels KDMA): x on the token;
    /// y, width, height, words/row, flag, 0. Window-relative, y = the
    /// bottom row; rows go out top first, as 0x0B5 takes them in. Pixels
    /// per word from width / words per row, MSB first.
    pub(super) fn gl_dma_read(&mut self, a: &[u32], out: &mut dyn Re3Sink) {
        use super::{ReadDest, ReadImage};
        self.gl.ensure_init();
        let (w, rows, wpr) = (a[2], a[3], a[4].max(1));
        let mut s2d = self.s2d;
        s2d.buf_select = match (w + wpr - 1) / wpr { 4 => 2, 2 => 1, _ => 0 };
        s2d.buf_offset = 0;
        let req = ReadImage {
            x: self.gl.win[0] + a[0] as i32,
            top: self.gl.win[1] + a[1] as i32 + rows as i32 - 1,
            w,
            rows,
            words_per_row: wpr,
            s2d,
            decode: self.gl_read_decode(),
        };
        out.read_image(&req, ReadDest::Gedma);
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
        self.gl_ctx.prev_owner = if self.gl_ctx.have_cur != 0 { self.gl_ctx.cur } else { u32::MAX };
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
        c.restore_reject = if c.expect as usize != CX_WORDS {
            1
        } else if c.buf[0] != CX_MAGIC {
            2
        } else if c.buf[1] != std::mem::size_of::<GlState>() as u32 {
            3
        } else {
            0
        };
        if c.restore_reject != 0 {
            // Not an image of ours (truncated, shifted, or another build's):
            // start from the defaults rather than keep the outgoing
            // context's live state, which would draw into its window.
            // SAFETY: GlState is plain data, valid when zeroed.
            self.gl = unsafe { std::mem::zeroed() };
        }
        let c = &mut self.gl_ctx;
        if c.restore_reject == 0 {
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
            | T_BLEND_MODE | T_LOGIC_OP | T_GET_COLOR | T_GET_NORMAL | T_GET_RASTERPOS | T_READ_DONE
            | T_READ_SETUP_A | T_READ_SETUP_B | T_READ_MODE => 1,
            T_FRAGMENT => 7,
            T_STENCIL_CLEAR | T_IRIS_BUFFER | T_IRIS_DB => 2,
            T_IRIS_WRITEMASK | T_IRIS_CLEAR | T_IRIS_ZCLEAR | T_IRIS_BGNPOLYGON | T_IRIS_ENDPOLYGON
            | T_IRIS_PMV | T_IRIS_PCLOS | T_IRIS_MOVE | T_IRIS_GETCPOS => 1,
            T_DEPTH_CLEAR => 3,
            T_IRIS_SBOXF | T_IRIS_SBOXFI => 4,
            T_IRIS_SWAPTMESH => 1,
            T_CLIP_PLANE_ENABLE => 2,
            T_CLIP_PLANE => 5,
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
            T_PIXEL_ZOOM => g.pzoom = [f(b[1]), f(b[2])],
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
            T_READ_BUFFER => {
                g.read_src = [b[0], b[1]];
                if let Some(d) = done.as_mut() {
                    d(format!("GL_READ_BUFFER {} {}", b[0], b[1]));
                }
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
                // x and y are full signed words (Gr2ValidateClip: xorg, and
                // 0x400 - (yorg + ysize)): negative for a window pushed past
                // the left or bottom edge of the screen. Masking them to 11
                // bits put such windows far off to the right / top.
                g.win = [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32];
                g.clip_wid = a[4];
                g.clip_obscured = a[5] & 1;
                g.clip_n = a[6];
                // Pieces: (x1 << 11) | x0, (ytop << 10) | ybottom, inclusive,
                // GL y up. Obscured with 0 pieces: one rectangle (the window
                // clamped to the screen) in the first pair; only a bound, the
                // visible region comes from the WID test (wid_clip).
                let pairs = if g.clip_n == 0 && g.clip_obscured != 0 { 1 } else { g.clip_n.min(4) };
                for k in 0..pairs as usize {
                    let (w0, w1) = (a[7 + 2 * k], a[8 + 2 * k]);
                    g.clip_rects[k] = [(w0 & 0x7ff) as i32, (w1 & 0x3ff) as i32,
                                       ((w0 >> 11) & 0x7ff) as i32 + 1, ((w1 >> 10) & 0x3ff) as i32 + 1];
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
            // The polygon is drawn whole when LOADV|0x065 follows.
            T_IRIS_ENDPOLYGON | T_IRIS_PCLOS => {}
            light::T_LIGHTING | light::T_TWO_SIDED | light::T_NORMALIZE | light::T_NORMALIZE_B
            | light::T_FOG_ON | light::T_MATERIAL_COMMIT => {
                g.lt.command(cmd, &a[..]);
            }
            T_LOGIC_OP => g.logic_op = a[0] & 0xf,
            T_READ_DONE | T_READ_SETUP_A | T_READ_SETUP_B | T_READ_MODE => {}
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
            T_CLIP_PLANE_ENABLE => if a[1] < 6 {
                let bit = 1 << a[1];
                g.uclip_on = if a[0] & 1 != 0 { g.uclip_on | bit } else { g.uclip_on & !bit };
            },
            T_CLIP_PLANE => if a[0] < 6 {
                g.uclip[a[0] as usize] = [f(a[1]), f(a[2]), f(a[3]), f(a[4])];
            },
            T_IRIS_SWAPTMESH => {
                g.pv.swap(0, 1);
                g.tflip ^= 1;
            }
            T_IRIS_SBOXF | T_IRIS_SBOXFI => {
                let num = |w: u32| if cmd == T_IRIS_SBOXFI { w as i32 as f32 } else { f(w) };
                let p = g.transform([num(a[0]), num(a[1]), 0.0, 1.0]);
                let q = g.transform([num(a[2]), num(a[3]), 0.0, 1.0]);
                if p.ok != 0 && q.ok != 0 {
                    // Window-space box: rasterized directly, no clipping.
                    let (x0, x1) = (p.x.min(q.x), p.x.max(q.x));
                    let (y0, y1) = (p.y.min(q.y), p.y.max(q.y));
                    let at = |x: f32, y: f32| Wv { x, y, oc: 0, ..p };
                    let mut quad = [at(x0, y0), at(x1, y0), at(x1, y1), at(x0, y1)];
                    // Counter-clockwise, so face culling never drops it.
                    let (cf, cb) = (g.cull_front, g.cull_back);
                    g.cull_front = 0;
                    g.cull_back = 0;
                    let mut none: Option<&mut dyn FnMut(String)> = None;
                    self.gl_poly_raster(&mut quad, p, ALL_EDGES, out, &mut none);
                    self.gl.cull_front = cf;
                    self.gl.cull_back = cb;
                }
            }
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
                // A new vertex routine ends the primitive in progress: every
                // End (GL 0x0FC/0x0FD, IRIS 0x041/0x042) is followed by
                // LOADV|0x065. A pending polygon is drawn now, whole.
                let mut none: Option<&mut dyn FnMut(String)> = None;
                self.gl_poly_flush(true, out, &mut none);
                let g = &mut self.gl;
                g.vroutine = c & 0x1ff;
                g.nv = 0;
                g.tflip = 0;
                g.pn = 0;
                g.pcont = 0;
                g.pv = [u32::MAX; 3];
                g.vfirst = u32::MAX;
            }
            END_LLOOP => {
                let g = &self.gl;
                if g.vroutine == VR_LLOOP && g.nv > 1 {
                    let (a, b) = (g.vb[g.pv[0] as usize], g.vb[g.vfirst as usize]);
                    self.gl_line(a, b, b.c, out);
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
            T_READ_SETUP_A | T_READ_SETUP_B => format!("GL_READ_SETUP {:#x} {:#x}", cmd, a[0]),
            T_READ_MODE => format!("GL_READ_MODE {}", a[0]),
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
        let polygon = matches!(self.gl.vroutine, VR_POLYGON | VR_IRIS_POLYGON);
        if polygon {
            // Polygons are convex (GL and IRIS GL): kept whole and drawn as
            // one primitive at their end. A buffer-full polygon is drawn
            // so far and continues from its first and last vertex.
            if self.gl.pn as usize == VB {
                self.gl_poly_flush(false, out, done);
            }
            let g = &mut self.gl;
            g.vb[g.pn as usize] = v;
            g.pn += 1;
            g.nv += 1;
            return;
        }
        let g = &mut self.gl;
        let n = g.nv;
        g.nv += 1;
        let s = g.vb_alloc();
        g.vb[s] = v;
        if n == 0 {
            g.vfirst = s as u32;
        }
        let [p0, p1, p2] = g.pv.map(|x| x as u8);
        g.pv = [s as u32, g.pv[0], g.pv[1]];
        let (s, first) = (s as u8, g.vfirst as u8);
        match g.vroutine {
            VR_POINTS => self.gl_point(v, v.c, out),
            VR_LINES => if n % 2 == 1 { self.gl_line(self.gl.vb[p0 as usize], v, v.c, out) },
            VR_LSTRIP | VR_LLOOP => if n >= 1 { self.gl_line(self.gl.vb[p0 as usize], v, v.c, out) },
            VR_TRIANGLES => if n % 3 == 2 { self.gl_poly(&[p1, p0, s], s, ALL_EDGES, out, done) },
            VR_TSTRIP => if n >= 2 {
                // (older, newer, v); every other triangle reversed to keep
                // one winding. A swaptmesh also flips it, so fans built with
                // swaps keep one winding too.
                let t = if g.tflip == 0 { [p1, p0, s] } else { [p0, p1, s] };
                g.tflip ^= 1;
                self.gl_poly(&t, s, ALL_EDGES, out, done)
            },
            VR_TFAN => if n >= 2 { self.gl_poly(&[first, p0, s], s, ALL_EDGES, out, done) },
            VR_QUADS => if n % 4 == 3 { self.gl_poly(&[p2, p1, p0, s], s, ALL_EDGES, out, done) },
            // Quad (v0, v1, v3, v2) of the pair (p2, p1) + (p0, v).
            VR_QSTRIP => if n >= 3 && n % 2 == 1 { self.gl_poly(&[p2, p1, s, p0], s, ALL_EDGES, out, done) },
            _ => {}
        }
    }

    /// Draw the polygon collected in the vertex buffer. `last`: its end
    /// (closing edge is a real edge); otherwise the buffer is full and the
    /// polygon continues as a fan piece from vertex 0 and the last one.
    fn gl_poly_flush(&mut self, last: bool, out: &mut dyn Re3Sink, done: &mut Option<&mut dyn FnMut(String)>) {
        let g = &self.gl;
        if !matches!(g.vroutine, VR_POLYGON | VR_IRIS_POLYGON) || g.pn < 3 {
            return;
        }
        let n = g.pn as usize;
        let mut edges = if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
        if !last {
            edges &= !(1u64 << (n - 1));
        }
        if g.pcont != 0 {
            edges &= !1;
        }
        let mut idx = [0u8; VB];
        for (i, x) in idx.iter_mut().enumerate().take(n) {
            *x = i as u8;
        }
        self.gl_poly(&idx[..n], 0, edges, out, done);
        if !last {
            let g = &mut self.gl;
            g.vb[1] = g.vb[n - 1];
            g.pn = 2;
            g.pcont = 1;
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
        // Always: a 2D 12-bit draw (MODE 2) may have left RGB12 behind.
        out.op(re3::RE3_OP_PIXFMT, self.gl.pixfmt as u64);
        if self.gl.pixfmt != 0 {
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

    /// One row of a GL pixel DMA (0x0B5 / 0x0B8, lrectwrite): `row` counts
    /// DMA rows, which arrive top row first; (x, y) is the rectangle's
    /// bottom-left corner in window coordinates (GL y up) and `h` its height.
    /// The kernel builds the VDMA "high to low" (_Gr2HtoLmkudmada, from the
    /// end of the bottom-up lrectwrite array) unless the client's rows are
    /// already top-down (_Gr2mkudmada), so the GE always gets rows top-down
    /// (OPART MRI: a 512x512 image in 4 bands of 128 rows, y = 0..384).
    /// `fmt`: 2 =
    /// 4 pixels per word, 1 = 2, 0 = 1; first pixel in the MSB. Colour-index
    /// visuals take the value as the index; RGB visuals take 32-bit pixels as
    /// 0xAABBGGRR (cpack order). Drawn as runs of equal pixels, clipped to
    /// the window, with integer pixel zoom for 0x0B8.
    pub(super) fn gl_pixel_row(&mut self, x: i32, y: i32, row: u32, w: u32, h: u32, fmt: u32, words: &[u32],
                               zoomed: bool, out: &mut dyn Re3Sink) {
        let g = &self.gl;
        let (per_word, bits) = match fmt { 2 => (4usize, 8u32), 1 => (2, 16), _ => (1, 32) };
        let px = |n: usize| -> u32 {
            let wi = n / per_word;
            if wi >= words.len() {
                return 0;
            }
            if bits == 32 { words[wi] } else { (words[wi] >> (32 - bits * (1 + (n % per_word) as u32))) & ((1 << bits) - 1) }
        };
        let ci = g.pixfmt == re3::PIXFMT_CI12 || bits < 32;
        let colour = |v: u32| -> [f32; 4] {
            if ci {
                [v as f32 / 255.0, 0.0, 0.0, 1.0]
            } else {
                let u = |sh: u32| ((v >> sh) & 0xff) as f32 / 255.0;
                [u(0), u(8), u(16), u(24)]
            }
        };
        let z = |v: f32| if zoomed && v >= 1.0 { v.round() as i32 } else { 1 };
        let (zx, zy) = (z(g.pzoom[0]), z(g.pzoom[1]));
        let from_bottom = h.saturating_sub(1 + row) as i32;
        let (wx, wy) = (g.win[0] + x, g.win[1] + y + from_bottom * zy);
        let cmax = g.cmax();
        self.gl_setup(out);
        let mut n = 0usize;
        while n < w as usize {
            let v = px(n);
            let mut e = n + 1;
            while e < w as usize && px(e) == v {
                e += 1;
            }
            self.gl_color_regs(to_fixed(colour(v), cmax), out);
            let (x0, x1) = (wx + n as i32 * zx, wx + e as i32 * zx);
            for dy in 0..zy {
                let (pieces, np) = self.gl.span_pieces(wy + dy, x0, x1);
                for &(a, b) in &pieces[..np] {
                    self.span(a, wy + dy, (b - a) as u32, None, out);
                }
            }
            n = e;
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
        if v.ok == 0 || v.oc != 0 {
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
    fn gl_line(&mut self, mut a: Wv, mut b: Wv, color: [f32; 4], out: &mut dyn Re3Sink) {
        // Parametric clip against every plane in use (Liang-Barsky style).
        let (oa, ob) = (a.oc, b.oc);
        if oa & ob != 0 {
            return;
        }
        if oa | ob != 0 {
            let (mut t0, mut t1) = (0.0f32, 1.0f32);
            for plane in 0..CLIP_PLANES {
                if (oa | ob) & (1 << plane) == 0 {
                    continue;
                }
                let (da, db) = (self.gl.clip_dist(&a, plane), self.gl.clip_dist(&b, plane));
                let t = da / (da - db);
                if da < 0.0 { t0 = t0.max(t) } else { t1 = t1.min(t) }
            }
            if t0 >= t1 {
                return;
            }
            let at = |t: f32| {
                // Distances along the edge are linear: fake them for clip_cross.
                clip_cross(&a, &b, -t, 1.0 - t)
            };
            let (na, nb) = (at(t0), at(t1));
            a = self.gl.project(na);
            b = self.gl.project(nb);
        }
        if a.ok == 0 || b.ok == 0 {
            return;
        }
        // One RE3 SHADED primitive: SHADED is not a horizontal span but a
        // stepper from (X, Y) by (DX, DY) per pixel for NUMPIX pixels (RE3.h;
        // the GR1 RE1 spec: "interpolates pixels along random-angled or
        // horizontal scan lines"), so a line is a single primitive with its
        // colour, Z and alpha iterated along it. The major axis steps whole
        // pixels (+-1.0) through the pixel centres in [start, end) (GL
        // diamond-exit approximated; the last point is not drawn, so strip
        // joints are drawn once); the minor axis starts at its coordinate at
        // the first centre, whole part in X / Y and 1/16 fraction in XYFRAC.
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let xmajor = dx.abs() >= dy.abs();
        let (ma, mb, na) = if xmajor { (a.x, b.x, a.y) } else { (a.y, b.y, a.x) };
        let dmaj = mb - ma;
        if dmaj == 0.0 {
            return;
        }
        let dir = if dmaj > 0.0 { 1.0f32 } else { -1.0 };
        let slope = (if xmajor { dy } else { dx }) / dmaj.abs();
        // First and past-last pixel index along the major axis.
        let (p0, p1) = if dir > 0.0 {
            ((ma - 0.5).ceil(), (mb - 0.5).ceil())
        } else {
            ((ma - 0.5).floor(), (mb - 0.5).floor())
        };
        let total = ((p1 - p0) * dir) as i32;
        if total <= 0 {
            return;
        }
        // Minor coordinate at pixel centre p (along the major axis).
        let minor_at = |p: f32| na + slope * ((p + 0.5 - ma) * dir);
        // Keep to pixels on screen: RE3 X / Y are unsigned.
        let (lim_maj, lim_min) = if xmajor { (re3::FB_W as f32, SCREEN_H as f32) } else { (SCREEN_H as f32, re3::FB_W as f32) };
        let mut i0 = 0i32;
        let mut i1 = total;
        // Major axis.
        let first_ok = |p: f32| p >= 0.0 && p < lim_maj;
        while i0 < i1 && !first_ok(p0 + dir * i0 as f32) {
            i0 += 1;
            if i0 > 4096 { return; }
        }
        while i1 > i0 && !first_ok(p0 + dir * (i1 - 1) as f32) {
            i1 -= 1;
        }
        // Minor axis (slope <= 1, so a bounded walk).
        let minor_ok = |i: i32| { let m = minor_at(p0 + dir * i as f32); m >= 0.0 && m < lim_min };
        while i0 < i1 && !minor_ok(i0) {
            i0 += 1;
        }
        while i1 > i0 && !minor_ok(i1 - 1) {
            i1 -= 1;
        }
        if i0 >= i1 {
            return;
        }
        let n = (i1 - i0) as u32;
        let pmaj = p0 + dir * i0 as f32;
        let m16 = (minor_at(pmaj) * 16.0).floor() as i32; // 1/16 pixel
        let (minor_px, minor_frac) = (m16 >> 4, (m16 & 15) as u32);
        let (x_px, y_px) = if xmajor { (pmaj as i32, minor_px) } else { (minor_px, pmaj as i32) };
        // Parameter along a -> b at the first pixel, and per pixel.
        let t0 = ((pmaj + 0.5 - ma) / dmaj).clamp(0.0, 1.0);
        let dt = 1.0 / dmaj.abs();
        let steps = 1.0 / dt; // pixels per unit parameter, for the deltas
        let smooth = self.gl.smooth != 0;
        let (ca, cb) = if smooth { (a.c, b.c) } else { (color, color) };
        let lerp = |p: f32, q: f32| p + (q - p) * t0;
        let cmax = self.gl.cmax();
        let blend = self.gl.blend_func().is_some();
        let zs = self.gl_zs_active();
        self.gl_setup(out);
        // Colour: start at step i0, per-pixel steps from the ends (8.11).
        let mut start = [0u32; 3];
        let mut step = [0u32; 3];
        for k in 0..3 {
            let m = 255.0 * if k == 0 { cmax } else { 1.0 };
            let c0 = (lerp(ca[k], cb[k]) * 255.0).clamp(0.0, m);
            let d = (cb[k] - ca[k]) * 255.0 / steps;
            start[k] = (c0 * 2048.0) as u32;
            step[k] = ((d * 2048.0) as i32) as u32;
        }
        out.reg(re3::REG_R, start[0]);
        out.reg(re3::REG_G, start[1]);
        out.reg(re3::REG_B, start[2]);
        out.reg(re3::REG_DR, step[0] & 0x00ff_ffff);
        out.reg(re3::REG_DG, step[1] & 0x000f_ffff);
        out.reg(re3::REG_DB, step[2] & 0x000f_ffff);
        if zs {
            let z = lerp(a.z, b.z).clamp(-8388608.0, 8388607.0);
            let dz = (((b.z - a.z) / steps) as f64 * 16384.0).round() as i64;
            out.reg(re3::REG_Z, (z as i32 as u32) & 0x00ff_ffff);
            out.reg(re3::REG_DZI, ((dz >> 14) as u32) & 0x00ff_ffff);
            out.reg(re3::REG_DZF, (dz as u32) & 0x3fff);
        }
        let dxy = |v: f32| ((v * 16384.0).round() as i32 as u32) & 0xffff;
        let (sx, sy) = if xmajor { (dir, slope) } else { (slope, dir) };
        out.reg(re3::REG_DX, dxy(sx));
        out.reg(re3::REG_DY, dxy(sy));
        out.reg(re3::REG_XYFRAC, minor_frac);
        // Clip in RE3: once per visible rectangle with the scissor set to it
        // (the WID test covers the rest in WID mode, set up by gl_setup).
        let (rects, nr) = self.gl.visible_rects();
        for r in &rects[..nr] {
            out.reg(re3::REG_XMIN, r[0] as u32);
            out.reg(re3::REG_XMAX, (r[2] - 1) as u32);
            out.reg(re3::REG_YMIN, r[1] as u32);
            out.reg(re3::REG_YMAX, (r[3] - 1) as u32);
            if blend {
                let a0 = (lerp(ca[3], cb[3]).clamp(0.0, 1.0) * 255.0 * 2048.0) as u32 as u64;
                let da = (((cb[3] - ca[3]) * 255.0 / steps * 2048.0) as i32) as u32 as u64;
                out.op(re3::RE3_OP_ALPHA, a0 | (da << 32));
            }
            self.gl_span_ir(x_px, y_px, n, re3::IR_SHADED, None, out);
        }
        // Back to the full-screen scissor and zero steps (FLAT spans and the
        // 2D path step the iterators too).
        out.reg(re3::REG_XMIN, 0);
        out.reg(re3::REG_XMAX, re3::FB_W as u32 - 1);
        out.reg(re3::REG_YMIN, 0);
        out.reg(re3::REG_YMAX, re3::FB_H as u32 - 1);
        out.reg(re3::REG_DX, 0);
        out.reg(re3::REG_DY, 0);
        out.reg(re3::REG_XYFRAC, 0);
        out.reg(re3::REG_DR, 0);
        out.reg(re3::REG_DG, 0);
        out.reg(re3::REG_DB, 0);
        if zs {
            out.reg(re3::REG_DZI, 0);
            out.reg(re3::REG_DZF, 0);
        }
        self.gl_done(out);
    }

    /// Clip a convex polygon (indices into the vertex buffer) to the view
    /// volume and the enabled user planes, then rasterize it as one
    /// primitive. Vertices carry their outcodes from transform, so most
    /// polygons are accepted or rejected without touching a plane. The rest
    /// are clipped Sutherland-Hodgman style on index lists; new vertices go
    /// to a scratch area addressed after the buffer. `prov` = provoking
    /// vertex (flat shading); `edges` bit k = edge k -> k+1 is a real edge
    /// (GL_LINE polygon mode). Backseat Driver's road and terrain run under
    /// the camera: without clipping they were lost entirely.
    fn gl_poly(&mut self, idx: &[u8], prov: u8, edges: u64, out: &mut dyn Re3Sink,
               done: &mut Option<&mut dyn FnMut(String)>) {
        let g = &self.gl;
        let (mut and, mut or) = (u32::MAX, 0u32);
        for &i in idx {
            let o = g.vb[i as usize].oc;
            and &= o;
            or |= o;
        }
        if and != 0 {
            return; // all outside one plane
        }
        let prov = g.vb[prov as usize];
        let mut rv = [Wv::default(); MAXP];
        let mut redges = edges;
        let n;
        if or == 0 {
            for (d, &i) in rv.iter_mut().zip(idx) {
                *d = g.vb[i as usize];
            }
            n = idx.len();
            if rv[..n].iter().any(|v| v.ok == 0) {
                return;
            }
        } else {
            let mut scratch = [Wv::default(); SCRATCH];
            let mut ns = 0usize;
            let mut cur = [0u8; MAXP];
            let mut cn = idx.len();
            cur[..cn].copy_from_slice(idx);
            let vtx = |scratch: &[Wv; SCRATCH], i: u8| -> Wv {
                if (i as usize) < VB { g.vb[i as usize] } else { scratch[i as usize - VB] }
            };
            for plane in 0..CLIP_PLANES {
                if or & (1 << plane) == 0 || cn < 3 {
                    continue;
                }
                let mut next = [0u8; MAXP];
                let mut ne = 0u64;
                let mut m = 0;
                for i in 0..cn {
                    let (ia, ib) = (cur[i], cur[(i + 1) % cn]);
                    let (a, b) = (vtx(&scratch, ia), vtx(&scratch, ib));
                    let (da, db) = (g.clip_dist(&a, plane), g.clip_dist(&b, plane));
                    let e = redges >> i & 1;
                    if da >= 0.0 && m < MAXP {
                        // The edge from an inside vertex is (part of) edge i.
                        next[m] = ia;
                        ne |= e << m;
                        m += 1;
                    }
                    if (da >= 0.0) != (db >= 0.0) && m < MAXP && ns < SCRATCH {
                        // Entering: the rest of edge i. Leaving: the next
                        // edge runs along the clip plane, not a real edge.
                        scratch[ns] = clip_cross(&a, &b, da, db);
                        next[m] = (VB + ns) as u8;
                        ne |= ((da < 0.0) as u64 & e) << m;
                        ns += 1;
                        m += 1;
                    }
                }
                cur = next;
                redges = ne;
                cn = m;
            }
            if cn < 3 {
                return;
            }
            for v in &mut scratch[..ns] {
                *v = g.project(*v);
            }
            for (d, &i) in rv.iter_mut().zip(&cur[..cn]) {
                *d = vtx(&scratch, i);
                d.oc = 0;
            }
            n = cn;
            if rv[..n].iter().any(|v| v.ok == 0) {
                return;
            }
        }
        self.gl_poly_raster(&mut rv[..n], prov, redges, out, done);
    }

    /// Culling, two-sided colours and polygon mode for one window-space
    /// convex polygon, then the rasterizer.
    fn gl_poly_raster(&mut self, v: &mut [Wv], mut prov: Wv, edges: u64, out: &mut dyn Re3Sink,
                      done: &mut Option<&mut dyn FnMut(String)>) {
        let n = v.len();
        if n < 3 {
            return;
        }
        // Twice the signed area (shoelace): > 0 = counter-clockwise, GL y up.
        let mut area = 0.0f32;
        for i in 0..n {
            let (a, b) = (&v[i], &v[(i + 1) % n]);
            area += a.x * b.y - b.x * a.y;
        }
        if area.abs() < 1e-6 {
            return;
        }
        let front = (area > 0.0) == (self.gl.front_ccw != 0);
        if (front && self.gl.cull_front != 0) || (!front && self.gl.cull_back != 0) {
            if let Some(d) = done.as_mut() {
                d(format!("GL polygon culled ({} face)", if front { "front" } else { "back" }));
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
        let smooth = self.gl.smooth != 0;
        match self.gl.polymode {
            2 => {
                for k in 0..n {
                    let c = if smooth { v[k].c } else { prov.c };
                    self.gl_point(v[k], c, out);
                }
                return;
            }
            3 => {
                for k in 0..n {
                    if k < 64 && edges >> k & 1 != 0 {
                        let (a, b) = (v[k], v[(k + 1) % n]);
                        let c = if smooth { b.c } else { prov.c };
                        self.gl_line(a, b, c, out);
                    }
                }
                return;
            }
            _ => {}
        }
        match (smooth, self.gl_zs_active()) {
            (true, true) => self.raster_poly::<true, true>(v, prov.c, out, done),
            (true, false) => self.raster_poly::<true, false>(v, prov.c, out, done),
            (false, true) => self.raster_poly::<false, true>(v, prov.c, out, done),
            (false, false) => self.raster_poly::<false, false>(v, prov.c, out, done),
        }
    }

    /// Scan-convert a convex polygon (window coordinates, GL y up; pixel
    /// centres at +0.5; a pixel is in if its centre is in [left, right) on
    /// rows whose centre is in [ymin, ymax)). Walks the two edge chains down
    /// from the top vertex; colour (SMOOTH), alpha and Z (ZS) are
    /// interpolated along the chains and then across each row, which is the
    /// plane equation for a triangle and Gouraud scanline interpolation for
    /// a polygon. Each row becomes one RE3 span per visible window piece.
    /// Flat shading takes `flat` (the provoking vertex's colour).
    fn raster_poly<const SMOOTH: bool, const ZS: bool>(&mut self, v: &[Wv], flat: [f32; 4], out: &mut dyn Re3Sink,
                                                       done: &mut Option<&mut dyn FnMut(String)>) {
        let n = v.len();
        let clip = self.gl.clip_rect();
        if let Some(d) = done.as_mut() {
            d(format!("GL polygon {} vertices ({:.1}, {:.1}) ({:.1}, {:.1}) ({:.1}, {:.1}){} {} clip {:?}",
                n, v[0].x, v[0].y, v[1].x, v[1].y, v[2].x, v[2].y, if n > 3 { " ..." } else { "" },
                if SMOOTH { "smooth" } else { "flat" }, clip));
        }
        let blend = self.gl.blend_func().is_some();
        // SHADED spans iterate colour and Z per pixel: smooth shading, depth
        // or stencil work and blending (flat colour = zero deltas).
        let iter = SMOOTH || ZS || blend;
        let cmax = self.gl.cmax();
        let flat_fixed = to_fixed(flat, cmax);
        self.gl_setup(out);
        if iter {
            out.reg(re3::REG_DX, 1 << 14);
            out.reg(re3::REG_DY, 0);
        } else {
            self.gl_color_regs(flat_fixed, out);
        }
        let (mut top, mut ymin, mut ymax) = (0, f32::MAX, f32::MIN);
        for (i, p) in v.iter().enumerate() {
            if p.y > ymax {
                ymax = p.y;
                top = i;
            }
            ymin = ymin.min(p.y);
        }
        let y0 = ((ymin - 0.5).ceil() as i32).max(clip[1]);
        let y1 = ((ymax - 0.5).ceil() as i32).min(clip[3]);
        // Chain state: current edge (upper a, lower b) and edges walked.
        // Chain 0 runs forward from the top vertex, chain 1 backward.
        let mut ch = [(top, (top + 1) % n, 0usize), (top, (top + n - 1) % n, 0usize)];
        let step = [1, n - 1];
        // Attributes along a chain at row centre yc: x, z, r, g, b, a.
        let at = |a: &Wv, b: &Wv, yc: f32| -> [f32; 6] {
            let t = (yc - b.y) / (a.y - b.y);
            let l = |p: f32, q: f32| q + (p - q) * t;
            let mut r = [l(a.x, b.x), 0.0, 0.0, 0.0, 0.0, 0.0];
            if ZS {
                r[1] = l(a.z, b.z);
            }
            if SMOOTH {
                for k in 0..4 {
                    r[2 + k] = l(a.c[k], b.c[k]) * 255.0;
                }
            } else if blend {
                r[5] = flat[3] * 255.0;
            }
            r
        };
        for y in (y0..y1).rev() {
            let yc = y as f32 + 0.5;
            let mut side = [[0.0f32; 6]; 2];
            let mut ok = true;
            for c in 0..2 {
                let (ref mut a, ref mut b, ref mut walked) = ch[c];
                // Advance past edges that end above this row.
                while v[*b].y > yc && *walked < n {
                    *a = *b;
                    *b = (*b + step[c]) % n;
                    *walked += 1;
                }
                if *walked >= n || v[*a].y <= yc {
                    ok = false;
                    break;
                }
                side[c] = at(&v[*a], &v[*b], yc);
            }
            if !ok {
                continue;
            }
            let (l, r) = if side[0][0] <= side[1][0] { (side[0], side[1]) } else { (side[1], side[0]) };
            let w = r[0] - l[0];
            let slope = |k: usize| if w > 1e-6 { (r[k] - l[k]) / w } else { 0.0 };
            let (pieces, np) = self.gl.span_pieces(y, (l[0] - 0.5).ceil() as i32, (r[0] - 0.5).ceil() as i32);
            for &(x0, x1) in &pieces[..np] {
                let cnt = (x1 - x0) as u32;
                let pat = self.gl_stipple_pattern(x0, y);
                if !iter {
                    self.span(x0, y, cnt, pat, out);
                    continue;
                }
                let px = x0 as f32 + 0.5;
                // Start and per-pixel step of attribute k over this span,
                // clamped at both ends like the hardware iterators.
                let run = |k: usize, max: f32| {
                    let d = slope(k);
                    let s0 = (l[k] + d * (px - l[0])).clamp(0.0, max);
                    let e = (s0 + d * (cnt - 1) as f32).clamp(0.0, max);
                    (s0, if cnt > 1 { (e - s0) / (cnt - 1) as f32 } else { 0.0 })
                };
                let (mut start, mut stepc) = ([0u32; 3], [0u32; 3]);
                for k in 0..3 {
                    if SMOOTH {
                        let (s0, d) = run(2 + k, 255.0 * if k == 0 { cmax } else { 1.0 });
                        start[k] = (s0 * 2048.0) as u32;
                        stepc[k] = ((d * 2048.0) as i32) as u32;
                    } else {
                        start[k] = flat_fixed[k];
                    }
                }
                if blend {
                    // Source alpha for the blender (8.11).
                    let (sa, da) = if SMOOTH { run(5, 255.0) } else { ((flat[3].clamp(0.0, 1.0)) * 255.0, 0.0) };
                    let st = ((da * 2048.0) as i32) as u32 as u64;
                    out.op(re3::RE3_OP_ALPHA, ((sa * 2048.0) as u32 as u64) | (st << 32));
                }
                if ZS {
                    // RE3 Z iterator: signed 24-bit integer Z, 24.14 step
                    // (IRIS GL uses -0x800000..0x7FFFFF, OpenGL 0..0x7FFFFF).
                    let dzp = slope(1);
                    let z = (l[1] + dzp * (px - l[0])).clamp(-8388608.0, 8388607.0);
                    let dz = (dzp as f64 * 16384.0).round() as i64;
                    out.reg(re3::REG_Z, (z as i32 as u32) & 0x00ff_ffff);
                    out.reg(re3::REG_DZI, ((dz >> 14) as u32) & 0x00ff_ffff);
                    out.reg(re3::REG_DZF, (dz as u32) & 0x3fff);
                }
                out.reg(re3::REG_R, start[0]);
                out.reg(re3::REG_G, start[1]);
                out.reg(re3::REG_B, start[2]);
                out.reg(re3::REG_DR, stepc[0] & 0x00ff_ffff);
                out.reg(re3::REG_DG, stepc[1] & 0x000f_ffff);
                out.reg(re3::REG_DB, stepc[2] & 0x000f_ffff);
                self.gl_span_ir(x0, y, cnt, re3::IR_SHADED, pat, out);
            }
        }
        if iter {
            // FLAT spans step the colour iterators too: leave them at zero.
            if ZS {
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
