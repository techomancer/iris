//! OpenGL on IMPACT: high-level emulation of the GE11 geometry microcode.
//!
//! libGLcore writes GL commands into the HQ3 command FIFO as command-
//! processor tokens (`hq3::token_name`). On the board the HQ3 hands them to
//! the GE11s, whose microcode keeps the GL state, transforms, lights and
//! clips, and sets up primitives in the RE4. Here the HQ thread does the
//! same with the shared GL core (`crate::dev::gl`): triangles go through the
//! RE4's area registers the way SGI's diagnostic triangle code sets them up
//! (`rss::Rss::triangle`), lines through the GL line registers
//! (`rss::Rss::gl_line`), and the per-fragment state (alpha, stencil, depth,
//! blend, logic op, write masks) into the PP1 registers (layouts in
//! `rss.rs`, provisional where only microcode writes them).
//!
//! The GL context's window (origin, screen masks, buffer pointers) reaches
//! the GE from the kernel (`Hq3Engine::apply_window`). The RE4 also serves
//! the X server, which keeps its own origin, masks, Y-flip and fill modes
//! in the same registers; the RSS saves them when a GL primitive batch
//! starts and puts them back when it ends (`rss::Rss::gl_enter`), the
//! effect of the RE4's separate register contexts.
//!
//! Token data formats, from traces of libGLcore (IRIX 6.5.22): compare
//! functions arrive as indices 0..7 (NEVER..ALWAYS), blend factors and
//! polygon modes as GL enums, logic ops as indices (GL order = X11 order),
//! stencil ops as indices (KEEP ZERO REPLACE INCR DECR INVERT), the alpha
//! reference as ref * 4096, the depth clear value as a 24-bit integer, the
//! colour mask as RGBA bits 0..3, the line stipple as (repeat - 1, pattern
//! bit-reversed: first bit in bit 15).
//!
//! Lighting (tokens per glLight / glMaterial / glLightModel parameter, light
//! positions and clip planes in object space, transformed here by the
//! modelview current when they arrive, as OpenGL specifies), fog (per
//! vertex) and user clip planes use the shared GL core.
//!
//! Texture (2D): texture coordinates go through the texture matrix and are
//! clipped with the vertex; triangles get S/W, T/W and Q/W planes in the
//! TE's iterator registers (`te1.rs`). The context's TE registers arrive in
//! libGLcore's register lists; the GE keeps a shadow of them and loads the
//! TE when a batch starts, so contexts do not see each other's textures.
//! Texture images arrive as SEND_PIXELS with routine 0x511A and go to the
//! texture loader (`tex_load`).
//!
//! Not yet: point size, textured lines and points, texgen, 1D/3D textures.

use super::hq3::Hq3Sink;
use super::te1::{self, reg as te_reg};
use crate::dev::gl::light::{Lighting, MAX_LIGHTS};
use crate::dev::gl::math::{self, Mat4, Stack};
use crate::dev::gl::vertex::{Clip, Viewport, Wv, MAX_POLY};

/// GL tokens the GE handles (numbers as in `hq3::token_name`).
mod tok {
    pub const VERTEX4F: u32 = 0x000;
    /// Context set-up for an RGB visual, or a colour-index one (one word:
    /// 8 or 12, presumably the index bits).
    pub const INIT_RGB: u32 = 0x009;
    pub const INIT_CI: u32 = 0x00A;
    pub const INIT_FORMAT_VALUES: u32 = 0x09A;
    /// glClearIndex.
    pub const CLEAR_INDEX: u32 = 0x0BD;
    /// glRasterPos (object x, y [, z [, w]]).
    pub const RASTER_POS4F: u32 = 0x038;
    /// glRasterPos / glBitmap raster position update (floats dx, dy).
    pub const UPDATE_RASTER_POS: u32 = 0x0D6;
    /// Latch raster position attributes to current attributes.
    pub const LOAD_RASTER_POS_INFO: u32 = 0x0D7;
    /// glIndexMask.
    pub const INDEX_MASK: u32 = 0x03D;
    /// glTexCoord (s [, t [, r [, q]]]).
    pub const TEX_COORD: u32 = 0x008;
    /// Texturing on / off for the primitives that follow (libGLcore
    /// mgrTM_SendTexmode1: on when the bound texture is complete).
    pub const TEX_ON: u32 = 0x00B;
    pub const TEX_OFF: u32 = 0x00C;
    /// GE pixel state (ERAM) writes: [?, ?, bytes, address, 0, 2, count,
    /// values]; the layout PIXEL_STATE also has.
    pub const INIT_PIXEL_STATE_ERAM: u32 = 0x0CD;
    /// One register written through an indexed GE shadow table: [index
    /// shadow, table shadow, register, value] (TXMIPMAP, TXBORDER and
    /// DETAILSCALE, indexed by TXADDR).
    pub const RSS_REG_SHADOW_IDX: u32 = 0x0EC;
    /// glTexGen: [coordinate 0x2000 + i (S T R Q), GL_TEXTURE_GEN_MODE,
    /// mode]; a plane's header [coordinate, GL_OBJECT_PLANE / EYE_PLANE],
    /// then its data (4 floats).
    pub const TEX_GENIV: u32 = 0x05A;
    pub const TEX_GEN_PLANE_HEADER: u32 = 0x05B;
    pub const TEX_GEN_PLANE_DATA: u32 = 0x05C;
    /// glEnable / glDisable of a capability the GE keeps (GL enum; seen:
    /// GL_TEXTURE_GEN_S..Q).
    pub const ENABLE_CAP: u32 = 0x075;
    /// glGetTexImage: [width, height, s and t offsets (floats, 0.5: texel
    /// centres), transfer mode, the raster interface start word]; the host
    /// starts the read itself (raster interface register 5).
    pub const READ_TEXTURE: u32 = 0x0F0;
    /// The texture manager's save/restore of TRAM pages (mgrTM_SR_TexRead
    /// / mgrTM_SR_TexLoad): a raw read of the sampler's level [rows,
    /// xfrmode, 0x400D], and the GE set up for texels by host DMA [?, ?,
    /// ?, GE routine].
    pub const READ_TEXTURE_RAW: u32 = 0x0E5;
    pub const WRITE_DMAGESETUP: u32 = 0x09D;
    /// glPointSize (float).
    pub const POINT_SIZE: u32 = 0x04D;
    pub const DISABLE_CAP: u32 = 0x076;
    /// glBitmap: format, row length in bits, rows, x / y origin and x / y
    /// move (floats), data words; the rows follow as FIFO pixel data.
    pub const BITMAP: u32 = 0x094;
    /// glDrawPixels through the GE's pixel path (libGLcore
    /// __glMgrSendPixels), 8 words: data words per row, two skips, the
    /// pixel-state counts, rows (provisional), the GE routine (0x49D0 for
    /// host to raster), flags; the rows follow as FIFO pixel data.
    pub const SEND_PIXELS: u32 = 0x08D;
    /// A write of GE pixel-path state (libGLcore): words 3 the GE address,
    /// 6 the count, then the values. Address 0x192 holds 1 / pixel zoom
    /// (1.0 in mandel, 1/6 in snoop at zoom 6).
    pub const PIXEL_STATE: u32 = 0x080;
    /// glReadPixels into host memory (libGLcore GetPixels*): 1 word; the
    /// read block comes from RSS_REG_SHADOW, the host DMA follows.
    pub const GET_PIXELS: u32 = 0x0DB;
    /// glReadBuffer: words, the third the GL enum (0x404 front, 0x405 back).
    pub const READ_BUFFER: u32 = 0x044;
    /// End of a pixel operation: the raster registers saved by SAVE_RSS
    /// come back.
    pub const RESTORE_RSS: u32 = 0x0D3;
    /// Start of a pixel operation (libGLcore __glMgrSaveRSS, 5 words).
    pub const SAVE_RSS: u32 = 0x0D2;
    /// glNormal3f (other forms through the HQ's conversion field).
    pub const NORMAL: u32 = 0x007;
    pub const LIGHTING_ON: u32 = 0x00D;
    pub const LIGHTING_OFF: u32 = 0x00E;
    /// glMaterial: face (GL_FRONT/BACK/FRONT_AND_BACK), pname, 4 values.
    pub const MATERIAL: u32 = 0x011;
    /// glLight: light index, values. Position and spot direction are in
    /// object space.
    pub const LIGHT_AMBIENT: u32 = 0x012;
    pub const LIGHT_DIFFUSE: u32 = 0x013;
    pub const LIGHT_SPECULAR: u32 = 0x014;
    pub const LIGHT_POSITION: u32 = 0x04F;
    pub const LIGHT_SPOT_DIRECTION: u32 = 0x050;
    pub const LIGHT_SPOT_EXPONENT: u32 = 0x051;
    pub const LIGHT_SPOT_CUTOFF: u32 = 0x052;
    pub const LIGHT_CONSTANT_ATTENUATION: u32 = 0x053;
    pub const LIGHT_LINEAR_ATTENUATION: u32 = 0x054;
    pub const LIGHT_QUADRATIC_ATTENUATION: u32 = 0x055;
    pub const LIGHT_MODEL_AMBIENT: u32 = 0x057;
    pub const LIGHT_MODEL_LOCAL_VIEWER: u32 = 0x058;
    pub const LIGHT_MODEL_TWO_SIDE: u32 = 0x059;
    /// glEnable / glDisable(GL_LIGHTi): the light index.
    pub const LIGHT_ON: u32 = 0x071;
    pub const LIGHT_OFF: u32 = 0x0A0;
    /// glColorMaterial(face, mode): front, back, front and back.
    pub const COLOR_MATERIAL_FRONT: u32 = 0x0BF;
    pub const COLOR_MATERIAL_BACK: u32 = 0x0C0;
    pub const COLOR_MATERIAL_BOTH: u32 = 0x0C1;
    pub const COLOR_MATERIAL_ON: u32 = 0x0C2;
    pub const COLOR_MATERIAL_OFF: u32 = 0x0C3;
    /// glEnable / glDisable(GL_NORMALIZE): 1 / 0.
    pub const NORMALIZE: u32 = 0x0A3;
    /// glClipPlane: 0x3000 + plane, a, b, c, d (object space); enable /
    /// disable: 0x3000 + plane.
    pub const CLIP_PLANE: u32 = 0x0A5;
    pub const CLIP_PLANE_ON: u32 = 0x0A6;
    pub const CLIP_PLANE_OFF: u32 = 0x0A7;
    /// Fog: enable (1 / 0), mode (GL enum), colour, start, end, density.
    pub const FOG: u32 = 0x068;
    pub const FOG_MODE: u32 = 0x0B9;
    pub const FOG_COLOR: u32 = 0x0B4;
    pub const FOG_START: u32 = 0x0B5;
    pub const FOG_END: u32 = 0x0B6;
    pub const FOG_DENSITY: u32 = 0x0B7;
    /// glDrawBuffer: buffer bits (1 front left, 2 front right, 4 back
    /// left, 8 back right; 0 none), then 2 words not decoded (1, 0 seen).
    pub const DRAW_BUFFER: u32 = 0x049;
    /// Kernel token (MgrasValidateBanks, at every swap and window
    /// validation): the bank GL draws into from now on, main buffers then
    /// the second set (see `Gl::set_draw_bank`).
    pub const VALIDATE_BANKS: u32 = 0x098;
    /// IRIS GL through IGLOO (libGLcore mgras_igloo.c): swaptmesh,
    /// lmcolor(LMC_COLOR) on / off, n3f (a normal, 3 floats).
    pub const SWAPTMESH: u32 = 0x0DC;
    pub const LMC_COLOR_ON: u32 = 0x0DD;
    pub const LMC_COLOR_OFF: u32 = 0x0DE;
    pub const IRIS_NORMAL: u32 = 0x0DF;
    pub const COLOR4F: u32 = 0x002;
    pub const CLEAR: u32 = 0x015;
    pub const BEGIN_POINTS: u32 = 0x016;
    pub const BEGIN_POLYGON: u32 = 0x01F;
    pub const END_POINTS: u32 = 0x020;
    pub const END_POLYGON: u32 = 0x029;
    pub const MATRIX_MODE: u32 = 0x02A;
    pub const LOAD_MATRIXD: u32 = 0x02B;
    pub const LOAD_IDENTITY: u32 = 0x02C;
    pub const MULT_MATRIXD: u32 = 0x02D;
    pub const POP_MATRIX: u32 = 0x02E;
    pub const PUSH_MATRIX: u32 = 0x02F;
    pub const ROTATED: u32 = 0x030;
    pub const SCALED: u32 = 0x031;
    pub const TRANSLATED: u32 = 0x032;
    pub const VIEWPORT: u32 = 0x033;
    pub const FRUSTUM: u32 = 0x034;
    pub const ORTHO: u32 = 0x035;
    pub const FLUSH: u32 = 0x036;
    pub const SCISSOR: u32 = 0x039;
    pub const STENCIL_MASK: u32 = 0x03A;
    pub const COLOR_MASK: u32 = 0x03B;
    pub const DEPTH_MASK: u32 = 0x03C;
    pub const ALPHA_FUNC: u32 = 0x03E;
    pub const BLEND_FUNC: u32 = 0x03F;
    pub const LOGIC_OP: u32 = 0x040;
    pub const STENCIL_FUNC: u32 = 0x041;
    pub const STENCIL_OP: u32 = 0x042;
    pub const DEPTH_FUNC: u32 = 0x043;
    pub const LINE_WIDTH: u32 = 0x045;
    pub const LINE_STIPPLE: u32 = 0x046;
    pub const SHADE_MODEL: u32 = 0x047;
    pub const DEPTH_RANGE: u32 = 0x048;
    /// glClear's depth / stencil parts (sent only when the visual has the
    /// buffer).
    pub const CLEAR_DEPTH_BUFFER: u32 = 0x04A;
    pub const CLEAR_STENCIL_BUFFER: u32 = 0x04C;
    pub const LOAD_POLYGON_STIPPLE: u32 = 0x04E;
    pub const FRONT_FACE: u32 = 0x056;
    /// glEnable / glDisable, 1 / 0 (3 / 0 for dither).
    pub const ALPHA_TEST: u32 = 0x064;
    pub const BLEND: u32 = 0x065;
    pub const DEPTH_TEST: u32 = 0x066;
    pub const DITHER: u32 = 0x067;
    pub const LINE_STIPPLE_ON: u32 = 0x06A;
    pub const LOGIC_OP_ON: u32 = 0x06B;
    pub const POLYGON_STIPPLE_ON: u32 = 0x06D;
    pub const SCISSOR_TEST: u32 = 0x06E;
    pub const STENCIL_TEST: u32 = 0x06F;
    /// glEnable / glDisable(GL_CULL_FACE), no data.
    pub const CULL_ON: u32 = 0x072;
    pub const CULL_OFF: u32 = 0x073;
    pub const CULL_FACE: u32 = 0x074;
    pub const POLYGON_MODE: u32 = 0x0A8;
    pub const CLEAR_COLOR: u32 = 0x0BA;
    pub const CLEAR_DEPTH: u32 = 0x0BC;
    pub const CLEAR_STENCIL: u32 = 0x0BE;
}

/// Primitive kinds, BEGIN_* minus BEGIN_POINTS (GL_POINTS .. GL_POLYGON).
mod prim {
    pub const POINTS: u32 = 0;
    pub const LINES: u32 = 1;
    pub const LSTRIP: u32 = 2;
    pub const LLOOP: u32 = 3;
    pub const TRIANGLES: u32 = 4;
    pub const TSTRIP: u32 = 5;
    pub const TFAN: u32 = 6;
    pub const QUADS: u32 = 7;
    pub const QSTRIP: u32 = 8;
    pub const POLYGON: u32 = 9;
    /// No primitive open.
    pub const NONE: u32 = 0xFF;
}

/// Raster registers the GE writes.
mod re {
    pub const TRI_X0: u32 = 0x000;
    pub const TRI_X2: u32 = 0x002;
    pub const TRI_YMID: u32 = 0x004;
    pub const TRI_DXDY0: u32 = 0x006;
    pub const TRI_DXDY1: u32 = 0x008;
    pub const TRI_DXDY2: u32 = 0x00A;
    pub const GLINE_XSTARTF: u32 = 0x00C;
    pub const IR: u32 = 0x013;
    pub const IR_ALIAS: u32 = 0x045;
    pub const BLOCK_XYSTARTI: u32 = 0x046;
    pub const BLOCK_XYENDI: u32 = 0x047;
    pub const XFRSIZE: u32 = 0x153;
    pub const XFRCONTROL: u32 = 0x102;
    pub const XFRMODE: u32 = 0x159;
    pub const RED: u32 = 0x05C;
    pub const DRE: u32 = 0x060;
    pub const Z: u32 = 0x068;
    pub const FILLMODE: u32 = 0x110;
    pub const XYWIN: u32 = 0x115;
    pub const GLINECONFIG: u32 = 0x146;
    pub const SCRMSK1X: u32 = 0x147;
    pub const WINMODE: u32 = 0x14F;
    pub const PP1WINMODE: u32 = 0x17B;
    pub const LSPAT: u32 = 0x15A;
    pub const LSCRL: u32 = 0x15B;
    pub const DEVICE_ADDR: u32 = 0x15C;
    pub const DEVICE_DATA: u32 = 0x15D;
    pub const PP1FILLMODE: u32 = 0x161;
    pub const COLORMASKMSBS: u32 = 0x162;
    pub const COLORMASKLSBSA: u32 = 0x163;
    pub const COLORMASKLSBSB: u32 = 0x164;
    pub const BLENDFACTOR: u32 = 0x165;
    pub const STENCILMODE: u32 = 0x166;
    pub const STENCILMASK: u32 = 0x167;
    pub const ZMODE: u32 = 0x168;
    pub const AFUNCMODE: u32 = 0x169;
    pub const DRBPOINTERS: u32 = 0x16D;
    pub const FILL_COLOR_R: u32 = 0x176;
}

/// Fill mode: fast fill from the fill colour registers, block type 0.
const FILL_FAST: u32 = 1 << 20;
/// Fill mode: block type 5, a DMA write transfer (the X server's value).
const FILL_DMA_WRITE: u32 = 0x1140_0000;
/// Fill mode: block type 4, a DMA read transfer (SAVE_RSS's value).
const FILL_DMA_READ: u32 = 0x0100_0000;
/// Fill mode: polygon stipple (provisional bit, see rss.rs) / line stipple.
const FILL_POLY_STIPPLE: u32 = 1 << 2;
const FILL_LINE_STIPPLE: u32 = 1 << 5;
/// PP1 fill mode for a 24-bit RGB window drawing buffer A (the 6.5.22 X
/// server's TrueColor value: pixel type 2, logic op copy enabled, draw
/// buffer 0x01); bits 29:26 hold the logic op.
const PP1_RGB24_BUFFER_A: u32 = 0x0C00_6204;
/// Area / GL line IR opcodes (provisional numbers, see rss.rs).
const IR_AREA_LTOR: u32 = super::rss::OP_AREA_LTOR;
const IR_AREA_RTOL: u32 = super::rss::OP_AREA_RTOL;
const IR_GL_LINE: u32 = super::rss::OP_GL_LINE;
/// SEND_PIXELS' routine for a texture image, and glDrawPixels'.
const SEND_PIXELS_TEXTURE: u32 = 0x511A;
const SEND_PIXELS_DRAW: u32 = 0x49D0;
/// XFRCONTROL: start a DMA transfer to the texture side (te1.rs, rss.rs).
const TE_LOAD_START: u32 = 0x5;
/// XFRCONTROL: a DMA read from the texture side (glGetTexImage starts it
/// as 0x400D through the raster interface).
const TE_READ_START: u32 = 0xD;
/// IR for a block with Setup.
const IR_BLOCK_SETUP: u32 = 0x18;
/// The RE4 takes positions as floats of value + this bias: in
/// [32768, 65536) a float's mantissa is fixed point with 8 fractional bits.
const POS_BIAS: f32 = 49151.5;
/// Largest window Z (24-bit depth buffer).
const ZMAX: f32 = 16_777_215.0;

/// The GL window's raster state, from the kernel.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct Window {
    pub valid: u32,
    /// x | y << 16, y bottom-up (window's lower left in framebuffer rows).
    pub origin: u32,
    pub mode: u32,
    /// Screen masks 1-4: (x, y) ranges `min << 16 | max`.
    pub masks: [[u32; 2]; 4],
    pub drb: u32,
    /// PP1 window mode: the origin's low bits and the clip ID the window's
    /// pixels must carry (see rss.rs `cid_match`).
    pub pp1winmode: u32,
}

/// GL state held by the GE. Plain data, valid zeroed (`ensure_init` sets
/// GL's initial values on first use).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Gl {
    inited: u32,
    pub window: Window,
    matrix_mode: u32,
    mv: Stack<32>,
    proj: Stack<4>,
    tex: Stack<4>,
    mvp: Mat4,
    mvp_dirty: u32,
    vp: Viewport,
    depth_range: [f32; 2],
    clip: Clip,
    color: [f32; 4],
    normal: [f32; 3],
    /// Lighting and fog (shared core); enabled lights as a bit mask; the
    /// normal matrix is stale; fog start, end, density.
    lt: Lighting,
    lights_on: u32,
    nm_dirty: u32,
    fog_start: f32,
    fog_end: f32,
    fog_density: f32,
    /// IRIS GL lmcolor(LMC_COLOR): on, and whether a colour has come after
    /// the last normal (what follows is then drawn unlit; IRIS GL
    /// Programming Guide 9.6.4, as GR2's GE7 HLE models it).
    lmc_color: u32,
    unlit: u32,
    /// Triangle strip: winding toggles per triangle and per swaptmesh.
    tflip: u32,
    /// glDrawBuffer bits, and whether the buffers have swapped an odd
    /// number of times (front = B, back = A).
    draw_bits: u32,
    /// DRAW_BUFFER's second word, the buffers drawn after an odd number of
    /// swaps (see `draw_mask`), and its third, the buffers drawn less one;
    /// `draw_words` is set when the token carried them.
    draw_bits_swapped: u32,
    draw_words: u32,
    buffer_count: u32,
    /// The driver's PP1 pixel-format and buffer-size fields.
    pixel_format: u32,
    swapped: u32,
    /// The kernel has named the bank drawn into (`set_draw_bank`): swaps
    /// leave `swapped` to it.
    banks_known: u32,
    smooth: u32,
    clear_color: [f32; 4],
    /// Depth: test on, function index, write mask on, clear value (24-bit).
    depth_test: u32,
    depth_func: u32,
    depth_mask: u32,
    clear_depth: u32,
    /// Culling: on, mode (GL_FRONT / BACK / FRONT_AND_BACK), front face
    /// counter-clockwise; polygon mode front / back (GL_POINT/LINE/FILL).
    cull: u32,
    cull_mode: u32,
    front_ccw: u32,
    polymode: [u32; 2],
    dither: u32,
    scissor_test: u32,
    /// Scissor, window coordinates, inclusive: x0, x1, y0, y1.
    scissor: [i32; 4],
    /// Blend: on, PP1 factor codes (see rss.rs).
    blend: u32,
    blend_src: u32,
    blend_dst: u32,
    /// Alpha test: on, function index, reference * 4096.
    alpha_test: u32,
    alpha_func: u32,
    alpha_ref: u32,
    /// Colour logic op: on, op index.
    logic: u32,
    logic_op: u32,
    /// Stencil: on, function, ref, compare mask, ops (fail, zfail, zpass),
    /// write mask, clear value.
    stencil: u32,
    st_func: u32,
    st_ref: u32,
    st_cmask: u32,
    st_ops: [u32; 3],
    st_wmask: u32,
    clear_stencil: u32,
    /// glColorMask: RGBA in bits 0..3.
    color_mask: u32,
    /// Lines: width, stipple on, repeat - 1, pattern (first bit 15).
    line_width: f32,
    line_stipple: u32,
    line_repeat: u32,
    line_pattern: u32,
    /// Polygon stipple: on, rows (window y mod 32; MSB = window x mod 32 0).
    poly_stipple: u32,
    stipple_rows: [u32; 32],
    /// The current raster position: window x, y, z, its colour, and
    /// whether it is valid (inside the view volume).
    raster: [f32; 3],
    raster_color: [f32; 4],
    raster_valid: u32,
    /// Colour-index context (INIT_CI): colours are indices (red / 4095 in
    /// `color`), drawn as 12-bit colour index pixels; and the clear index.
    ci: u32,
    /// Unnormalized current index for GE state readback.
    current_index: f32,
    clear_index: u32,
    index_mask: u32,
    /// A glBitmap header waiting for its rows (pixel data).
    bitmap: [u32; 8],
    bitmap_pending: u32,
    /// A SEND_PIXELS header waiting for its rows, and the raster engine's
    /// transfer mode as the driver last set it (host pixel format).
    send_pixels: [u32; 8],
    send_pixels_pending: u32,
    pub xfrmode: u32,
    /// 1 / the pixel zoom (GE pixel state 0x192); 0 when never set.
    zoom_inv: f32,
    /// The pixel block (block_xystarti / endi, window coordinates) the
    /// driver last set through RSS_REG_SHADOW / SET, and glReadBuffer back.
    pub pixel_block: [u32; 2],
    read_back: u32,
    /// Between SAVE_RSS and RESTORE_RSS: a GL pixel operation, whose
    /// register lists belong inside the raster bracket. 2 denotes a
    /// framebuffer read, whose DMA bytes need GE component ordering.
    pub pixel_op: u32,
    /// Primitive assembly: the open primitive (`prim`), vertices since its
    /// BEGIN, and the buffered vertices (strips and fans keep the first
    /// and the last two; quads and polygons keep them all).
    prim: u32,
    nv: u32,
    vb: [Wv; MAX_POLY],
    /// A GL batch's raster state is loaded (`begin_raster`), and the fill
    /// mode last written in it.
    raster_loaded: u32,
    fill_loaded: u32,
    /// Texturing: the current texture coordinates; on (TEX_ON / TEX_OFF);
    /// the GE's shadow of the context's TE registers (`te1::CONTEXT_REGS`
    /// order), TXADDR and the tables it indexes; the TE must be loaded from
    /// the shadow at the next batch.
    texcoord: [f32; 4],
    tex_on: u32,
    te: [u32; te1::CONTEXT_REGS.len()],
    te_index: u32,
    te_tables: [[u32; te1::TABLE_LEN]; 3],
    pub te_dirty: u32,
    /// The texture loader's GE pixel state: the sub-image (ERAM 0xDA8: s
    /// and t origin, width, height, transfer size) and where it goes
    /// (0xDB1: level width, ?, level, page | bank << 8, border page, ?).
    tl_rect: [u32; 5],
    tl_dest: [u32; 6],
    /// The host pixel format's scale to 0..1 (ERAM 0xD0A: 1/255 for 8-bit
    /// components, 1/65535 for 16-bit, 1.0 for floats).
    tl_scale: u32,
    /// The host element the GE unpacks (pixel state 0xDBB): 1 bytes, 3 16-bit
    /// (components, or packed 4-4-4-4 / 5-5-5-1), 4 32-bit (packed 8-8-8-8,
    /// 10-10-10-2, or components).
    tl_elem: u32,
    /// Texture coordinate generation, per coordinate S T R Q: on (bits),
    /// mode (0 object linear, 1 eye linear, 2 sphere map), object and eye
    /// planes (eye planes in eye space), and the plane the next
    /// TEX_GEN_PLANE_DATA is for (coordinate | eye << 2).
    texgen_on: u32,
    texgen_mode: [u32; 4],
    texgen_obj: [[f32; 4]; 4],
    texgen_eye: [[f32; 4]; 4],
    texgen_plane: u32,
    /// glPointSize.
    point_size: f32,
    /// A texture read transfer is armed (READ_TEXTURE .. RESTORE_RSS).
    tex_reading: u32,
    pub stats_triangles: u64,
}

impl Gl {
    /// The state that places drawing (window, buffers, viewport, scissor),
    /// for the monitor.
    pub fn describe(&self) -> String {
        let w = &self.window;
        format!(
            "  window: valid {} origin {:#x} mode {:#x} PP1winmode {:#x} DRBpointers {:#x} masks {:x?}\n  \
             draw buffer bits {:#x}, swapped {}, colour mask {:#x}\n  \
             viewport x {}..{} y {}..{}, scissor {} {:?}\n  \
             colour {:?}, clear colour {:?}, lighting {} (lights {:#x}), depth test {}",
            w.valid, w.origin, w.mode, w.pp1winmode, w.drb, w.masks, self.draw_bits, self.swapped, self.color_mask,
            self.vp.x0, self.vp.x1, self.vp.y0, self.vp.y1, self.scissor_test, self.scissor,
            self.color, self.clear_color, self.lt.on, self.lights_on, self.depth_test)
    }

    fn ensure_init(&mut self) {
        if self.inited != 0 {
            return;
        }
        self.inited = 1;
        self.mv.init();
        self.proj.init();
        self.tex.init();
        self.mvp = math::IDENT;
        self.depth_range = [0.0, 1.0];
        self.vp.set(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, ZMAX);
        self.color = [1.0; 4];
        self.current_index = 1.0;
        self.normal = [0.0, 0.0, 1.0];
        self.lt.init();
        self.nm_dirty = 1;
        self.fog_end = 1.0;
        self.fog_density = 1.0;
        self.lt.fog_mode = 1;
        self.update_fog();
        self.smooth = 1;
        self.draw_bits = 1;
        self.depth_func = 1;
        self.depth_mask = 1;
        self.clear_depth = ZMAX as u32;
        self.cull_mode = 0x405;
        self.front_ccw = 1;
        self.polymode = [0x1B02; 2];
        self.dither = 1;
        self.blend_dst = 0;
        self.blend_src = 1;
        self.alpha_func = 7;
        self.logic_op = 3;
        self.st_func = 7;
        self.st_cmask = 0xFF;
        self.st_wmask = 0xFF;
        self.color_mask = 0xF;
        self.index_mask = 0xFFF;
        self.scissor = [0, 1279, 0, 1023];
        self.line_width = 1.0;
        self.line_pattern = 0xFFFF;
        self.prim = prim::NONE;
        self.matrix_mode = 0x1700;
    }

    fn stack(&mut self) -> &mut dyn MatrixStack {
        match self.matrix_mode {
            0x1701 => &mut self.proj,
            0x1702 => &mut self.tex,
            _ => &mut self.mv,
        }
    }

    fn matrix_op(&mut self, f: impl FnOnce(&mut dyn MatrixStack)) {
        f(self.stack());
        self.mvp_dirty = 1;
        self.nm_dirty = 1;
    }

    /// Rebuild the enabled-light chain from `lights_on`.
    fn update_lights(&mut self) {
        let mut prev: i32 = -1;
        self.lt.head = -1;
        self.lt.count = 0;
        for i in 0..MAX_LIGHTS {
            if self.lights_on & (1 << i) == 0 {
                continue;
            }
            if prev < 0 {
                self.lt.head = i as i32;
            } else {
                self.lt.lights[prev as usize].next = i as i32;
            }
            self.lt.lights[i].next = -1;
            prev = i as i32;
            self.lt.count += 1;
        }
    }

    /// The fog parameters in the shared core's form.
    fn update_fog(&mut self) {
        if self.lt.fog_mode == 0 {
            let d = self.fog_end - self.fog_start;
            (self.lt.fog_a, self.lt.fog_b) = (self.fog_end, if d != 0.0 { 1.0 / d } else { 0.0 });
        } else {
            self.lt.fog_a = self.fog_density;
        }
    }

    /// A light parameter token: light index, then values.
    fn light_param(&mut self, cmd: u32, d: &[u32]) {
        let Some(&i) = d.first() else { return };
        if i as usize >= MAX_LIGHTS {
            return;
        }
        let v: Vec<f32> = d[1..].iter().map(|&w| f32::from_bits(w)).collect();
        let g = |k: usize| v.get(k).copied().unwrap_or(0.0);
        let mv = *self.mv.get();
        let l = &mut self.lt.lights[i as usize];
        match cmd {
            tok::LIGHT_AMBIENT => l.ambient = [g(0), g(1), g(2)],
            tok::LIGHT_DIFFUSE => l.diffuse = [g(0), g(1), g(2)],
            tok::LIGHT_SPECULAR => l.specular = [g(0), g(1), g(2)],
            tok::LIGHT_POSITION => l.pos = math::xform(&mv, [g(0), g(1), g(2), g(3)]),
            tok::LIGHT_SPOT_DIRECTION => {
                let e = math::xform(&mv, [g(0), g(1), g(2), 0.0]);
                let n = (e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt();
                l.spot_dir = if n > 1e-20 { [e[0] / n, e[1] / n, e[2] / n] } else { [0.0, 0.0, -1.0] };
            }
            tok::LIGHT_SPOT_EXPONENT => l.spot_exp = g(0),
            tok::LIGHT_SPOT_CUTOFF => {
                l.spot_on = (g(0) != 180.0) as u32;
                l.spot_cos_cutoff = g(0).to_radians().cos();
            }
            tok::LIGHT_CONSTANT_ATTENUATION => l.atten[2] = g(0),
            tok::LIGHT_LINEAR_ATTENUATION => l.atten[0] = g(0),
            _ => l.atten[1] = g(0),
        }
    }

    /// glMaterial: face, pname, values.
    fn material(&mut self, d: &[u32]) {
        if d.len() < 3 {
            return;
        }
        let (face, pname) = (d[0], d[1]);
        let v: Vec<f32> = d[2..].iter().map(|&w| f32::from_bits(w)).collect();
        let g = |k: usize| v.get(k).copied().unwrap_or(0.0);
        let faces: &[usize] = match face {
            0x404 => &[0],
            0x405 => &[1],
            _ => &[0, 1],
        };
        for &f in faces {
            let m = &mut self.lt.mat[f];
            match pname {
                0x1200 => m.ambient = [g(0), g(1), g(2)],
                0x1201 => m.diffuse = [g(0), g(1), g(2), g(3)],
                0x1202 => m.specular = [g(0), g(1), g(2)],
                0x1600 => m.emission = [g(0), g(1), g(2)],
                0x1601 => m.shininess = g(0),
                0x1602 => {
                    m.ambient = [g(0), g(1), g(2)];
                    m.diffuse = [g(0), g(1), g(2), g(3)];
                }
                _ => {}
            }
        }
    }

    fn mvp(&mut self) -> Mat4 {
        if self.mvp_dirty != 0 {
            self.mvp_dirty = 0;
            self.mvp = math::mul(self.proj.get(), self.mv.get());
        }
        self.mvp
    }

    /// A GE state word for a readback (`__MGR_RETURN_MODE` address), where
    /// the HLE knows the layout: 0x1FC the dither enable (glFinish asks for
    /// it); 0x3A-0x49 the current matrix mode's top matrix, 16 floats in
    /// OpenGL's memory order (IRIS GL getmatrix under IGLOO reads them).
    pub fn state_word(&mut self, addr: u32) -> Option<u32> {
        self.ensure_init();
        match addr {
            4 if self.ci != 0 => Some(self.current_index.to_bits()),
            4..=7 => Some(self.color[(addr - 4) as usize].to_bits()),
            // libGLcore glPushAttrib/glPopAttrib read these GE words rather
            // than its software state. Placeholder zeros erase write masks,
            // viewports and scissor boxes when applications restore them.
            0x29 => Some(self.matrix_mode.wrapping_sub(0x1701)),
            0x83 => Some(0x899 + self.mv.top * 16),
            0x896 => Some(0xA99 + self.proj.top * 16),
            0x897 => Some(0xAB9 + self.tex.top * 16),
            0x29F => Some(if self.ci != 0 { self.index_mask } else { self.color_mask }),
            0xB7E..=0xB81 => Some(self.clear_color[(addr - 0xB7E) as usize].to_bits()),
            0xB86 => Some((self.clear_depth as f32 / ZMAX).to_bits()),
            0xB87 => Some((self.clear_index as f32).to_bits()),
            0xB88 => Some(self.clear_stencil),
            0xB91..=0xB94 => Some(match addr {
                0xB91 => self.vp.x0 as i32 as u32,
                0xB92 => self.vp.y0 as i32 as u32,
                0xB93 => (self.vp.x1 - self.vp.x0 + 1.0) as i32 as u32,
                _ => (self.vp.y1 - self.vp.y0 + 1.0) as i32 as u32,
            }),
            0xB95..=0xB96 => Some(self.depth_range[(addr - 0xB95) as usize].to_bits()),
            // glGet(GL_SCISSOR_BOX) reads x, y, xmax, ymax, then computes
            // width/height. SCISSOR tokens carry x, width-1, y, height-1.
            0xB98..=0xB9B => {
                let [x0, x1, y0, y1] = self.scissor_rect();
                Some([x0, y0, x1, y1][(addr - 0xB98) as usize] as u32)
            }
            0xBB1 => Some(self.point_size.to_bits()),
            0xBB2 => Some(self.line_width.to_bits()),
            0xBBA => Some(self.line_repeat + 1),
            0x1FC => Some(self.dither),
            // Scissor box 0 (window coordinates inclusive: xmin, xmax, ymin, ymax):
            // 0x200 returns (xmax << 16) | xmin, 0x202 returns (ymax << 16) | ymin.
            // Read by libGLcore __glMgrim_DrawPixels for scissor box clipping.
            0x200 => Some(((self.scissor[1] as u32) << 16) | (self.scissor[0] as u32 & 0xFFFF)),
            0x202 => Some(((self.scissor[3] as u32) << 16) | (self.scissor[2] as u32 & 0xFFFF)),
            // The current raster position, window x, y, z (floats), and
            // whether it is valid: read by IRIS GL getcpos (gr_osview
            // measures its text with it before and after charstr) and
            // OpenGL glGetIntegerv(GL_CURRENT_RASTER_POSITION) / glDrawPixels.
            // X and Y are biased by RE_COORD_BIAS (49151.5), which libGLcore
            // subtracts when converting GE coordinates back to window space.
            0x2CD..=0x2CF => {
                let v = match addr {
                    0x2CD => self.raster[0] + 49151.5,
                    0x2CE => self.raster[1] + 49151.5,
                    _ => self.raster[2] / ZMAX,
                };
                Some(v.to_bits())
            }
            0x2C4 => Some(1.0f32.to_bits()),
            // Bit 2 (0x4) is the valid flag tested by libGLcore (__glMgrDoGet);
            // bit 0 (0x1) is tested by other callers. Return 0x7 when valid.
            0x2D0 => Some(if self.raster_valid != 0 { 0x7 } else { 0 }),
            0x2BF => Some(0.0f32.to_bits()),
            0x2C5..=0x2C8 => Some(self.raster_color[(addr - 0x2C5) as usize].to_bits()),
            0x3A..=0x49 => {
                let m = match self.matrix_mode {
                    0x1701 => self.proj.get(),
                    0x1702 => self.tex.get(),
                    _ => self.mv.get(),
                };
                Some(m[(addr - 0x3A) as usize].to_bits())
            }
            0x4A..=0x59 => Some(self.proj.get()[(addr - 0x4A) as usize].to_bits()),
            0x6A..=0x79 => Some(self.tex.get()[(addr - 0x6A) as usize].to_bits()),
            _ => None,
        }
    }

    /// One GL token. False when the GE does not handle it (yet).
    pub fn token(&mut self, cmd: u32, d: &[u32], sink: &mut dyn Hq3Sink) -> bool {
        self.ensure_init();
        let w0 = d.first().copied().unwrap_or(0);
        let on = (w0 != 0) as u32;
        match cmd {
            tok::MATRIX_MODE => self.matrix_mode = if d.is_empty() { 0x1700 } else { w0 },
            tok::LOAD_IDENTITY => self.matrix_op(|s| s.load(math::IDENT)),
            tok::PUSH_MATRIX => self.matrix_op(|s| { s.push(); }),
            tok::POP_MATRIX => self.matrix_op(|s| { s.pop(); }),
            tok::LOAD_MATRIXD | tok::MULT_MATRIXD => {
                let a = args(d, 16);
                let mut m = [0.0f32; 16];
                m.copy_from_slice(&a[..16]);
                if cmd == tok::LOAD_MATRIXD {
                    self.matrix_op(|s| s.load(m));
                } else {
                    self.matrix_op(|s| s.mult(&m));
                }
            }
            tok::ROTATED => {
                let a = args(d, 4);
                self.matrix_op(|s| s.mult(&math::rotate(a[0], a[1], a[2], a[3])));
            }
            tok::TRANSLATED => {
                let a = args(d, 3);
                self.matrix_op(|s| s.mult(&math::translate(a[0], a[1], a[2])));
            }
            tok::SCALED => {
                let a = args(d, 3);
                self.matrix_op(|s| s.mult(&math::scale(a[0], a[1], a[2])));
            }
            tok::ORTHO | tok::FRUSTUM => {
                let a = args(d, 6);
                let m = if cmd == tok::ORTHO {
                    math::ortho(a[0], a[1], a[2], a[3], a[4], a[5])
                } else {
                    math::frustum(a[0], a[1], a[2], a[3], a[4], a[5])
                };
                self.matrix_op(|s| s.mult(&m));
            }
            tok::VIEWPORT if d.len() >= 4 => {
                let v: Vec<f32> = d[..4].iter().map(|&w| w as i32 as f32).collect();
                let [n, f] = self.depth_range;
                self.vp.set(v[0], v[1], v[2], v[3], n, f, ZMAX);
            }
            tok::DEPTH_RANGE => {
                let a = args(d, 2);
                self.depth_range = [a[0], a[1]];
                let vp = self.vp;
                self.vp.set(vp.x0, vp.y0, vp.x1 - vp.x0 + 1.0, vp.y1 - vp.y0 + 1.0, a[0], a[1], ZMAX);
            }
            tok::SHADE_MODEL => self.smooth = (w0 != 0x1D00) as u32,
            tok::COLOR4F => {
                self.color = args_f32(d);
                if self.ci != 0 {
                    // An index: red carries it to the raster engine, which
                    // writes red * 4095 in colour-index pixels.
                    self.current_index = self.color[0];
                    self.color = [self.current_index / 4095.0, 0.0, 0.0, 1.0];
                }
                // The raster colour follows the current colour: IRIS GL
                // programs (gr_osview's colour-keyed legend, through IGLOO)
                // set a colour between bitmaps without a new raster
                // position, and libGLcore sends no colour for glBitmap.
                // Our reading; OpenGL's latch at glRasterPos would then be
                // libGLcore's business (UPDATE_RASTER_POS and friends).
                self.raster_color = self.color;
                self.lt.track_color(self.color);
                if self.lmc_color != 0 && self.lt.cmat_on == 0 {
                    self.unlit = 1;
                }
            }
            tok::NORMAL | tok::IRIS_NORMAL => {
                let a = args_f32(d);
                self.normal = [a[0], a[1], a[2]];
                self.unlit = 0;
            }
            tok::LMC_COLOR_ON => self.lmc_color = 1,
            tok::LMC_COLOR_OFF => {
                self.lmc_color = 0;
                self.unlit = 0;
            }
            tok::SWAPTMESH => {
                self.vb.swap(0, 1);
                self.tflip ^= 1;
            }
            tok::LIGHTING_ON => self.lt.on = 1,
            tok::LIGHTING_OFF => self.lt.on = 0,
            tok::MATERIAL => self.material(d),
            tok::LIGHT_AMBIENT | tok::LIGHT_DIFFUSE | tok::LIGHT_SPECULAR
            | tok::LIGHT_POSITION..=tok::LIGHT_QUADRATIC_ATTENUATION => self.light_param(cmd, d),
            tok::LIGHT_MODEL_AMBIENT => {
                let a = args_f32(d);
                self.lt.amb_sum = [a[0], a[1], a[2]];
            }
            tok::LIGHT_MODEL_LOCAL_VIEWER => self.lt.local_viewer = on,
            tok::LIGHT_MODEL_TWO_SIDE => self.lt.two_sided = on,
            tok::LIGHT_ON | tok::LIGHT_OFF if (w0 as usize) < MAX_LIGHTS => {
                if cmd == tok::LIGHT_ON { self.lights_on |= 1 << w0 } else { self.lights_on &= !(1 << w0) }
                self.update_lights();
            }
            tok::COLOR_MATERIAL_FRONT | tok::COLOR_MATERIAL_BACK | tok::COLOR_MATERIAL_BOTH => {
                self.lt.cmat_face = cmd - tok::COLOR_MATERIAL_FRONT;
                self.lt.cmat_param = match w0 {
                    0x1600 => 1,
                    0x1200 => 2,
                    0x1201 => 3,
                    0x1202 => 4,
                    _ => 5,
                };
            }
            tok::COLOR_MATERIAL_ON => {
                self.lt.cmat_on = 1;
                self.lt.track_color(self.color);
            }
            tok::COLOR_MATERIAL_OFF => self.lt.cmat_on = 0,
            tok::NORMALIZE => self.lt.normalize = on,
            tok::CLIP_PLANE if d.len() >= 5 && (w0.wrapping_sub(0x3000) as usize) < 6 => {
                // Object space to eye space: the plane times the inverse of
                // the current modelview.
                let p = [f32::from_bits(d[1]), f32::from_bits(d[2]), f32::from_bits(d[3]), f32::from_bits(d[4])];
                let inv = math::invert(self.mv.get()).unwrap_or(math::IDENT);
                let col = |c: usize| (0..4).map(|r| p[r] * inv[c * 4 + r]).sum::<f32>();
                self.clip.user[(w0 - 0x3000) as usize] = [col(0), col(1), col(2), col(3)];
            }
            tok::CLIP_PLANE_ON | tok::CLIP_PLANE_OFF if (w0.wrapping_sub(0x3000) as usize) < 6 => {
                let bit = 1 << (w0 - 0x3000);
                if cmd == tok::CLIP_PLANE_ON { self.clip.user_on |= bit } else { self.clip.user_on &= !bit }
            }
            tok::FOG => self.lt.fog_on = on,
            tok::FOG_MODE => {
                self.lt.fog_mode = match w0 { 0x2601 => 0, 0x801 => 2, _ => 1 };
                self.update_fog();
            }
            tok::FOG_COLOR => {
                let a = args_f32(d);
                self.lt.fog_color = [a[0], a[1], a[2]];
            }
            tok::FOG_START | tok::FOG_END | tok::FOG_DENSITY => {
                let v = f32::from_bits(w0);
                match cmd {
                    tok::FOG_START => self.fog_start = v,
                    tok::FOG_END => self.fog_end = v,
                    _ => self.fog_density = v,
                }
                self.update_fog();
            }
            // IRIS GL clear() clears to the current colour: IGLOO sends GE
            // state addresses instead of floats (4, 5, 6, 7: the current
            // colour's words; as floats these would be denormals).
            tok::CLEAR_COLOR if d.len() >= 4 && d[..4].iter().all(|&w| w < 0x100) => {
                let mut c = [0.0f32; 4];
                for (k, &a) in d[..4].iter().enumerate() {
                    c[k] = match a {
                        4..=7 => self.color[(a - 4) as usize],
                        _ => 0.0,
                    };
                }
                self.clear_color = c;
            }
            tok::CLEAR_COLOR => self.clear_color = args_f32(d),
            tok::CLEAR_DEPTH => self.clear_depth = w0 & 0xFF_FFFF,
            tok::CLEAR_STENCIL => self.clear_stencil = w0 & 0xFF,
            tok::CLEAR => self.clear_color_buffer(sink),
            tok::CLEAR_DEPTH_BUFFER => self.clear_depth_stencil(true, sink),
            tok::CLEAR_STENCIL_BUFFER => self.clear_depth_stencil(false, sink),
            tok::DEPTH_TEST => self.set_state(|g| g.depth_test = on, sink),
            tok::DEPTH_FUNC => self.set_state(|g| g.depth_func = w0 & 7, sink),
            tok::DEPTH_MASK => self.set_state(|g| g.depth_mask = on, sink),
            tok::ALPHA_TEST => self.set_state(|g| g.alpha_test = on, sink),
            tok::ALPHA_FUNC if d.len() >= 2 => self.set_state(|g| (g.alpha_func, g.alpha_ref) = (w0 & 7, d[1].min(0xFFF)), sink),
            tok::BLEND => self.set_state(|g| g.blend = on, sink),
            tok::BLEND_FUNC if d.len() >= 2 => self.set_state(|g| (g.blend_src, g.blend_dst) = (blend_code(w0), blend_code(d[1])), sink),
            tok::LOGIC_OP_ON => self.set_state(|g| g.logic = on, sink),
            tok::LOGIC_OP => self.set_state(|g| g.logic_op = w0 & 0xF, sink),
            tok::STENCIL_TEST => self.set_state(|g| g.stencil = on, sink),
            tok::STENCIL_FUNC if d.len() >= 3 => self.set_state(|g| (g.st_func, g.st_ref, g.st_cmask) = (w0 & 7, d[1] & 0xFF, d[2] & 0xFF), sink),
            tok::STENCIL_OP if d.len() >= 3 => self.set_state(|g| g.st_ops = [w0 & 7, d[1] & 7, d[2] & 7], sink),
            tok::STENCIL_MASK => self.set_state(|g| g.st_wmask = w0 & 0xFF, sink),
            tok::COLOR_MASK => self.set_state(|g| g.color_mask = w0 & 0xF, sink),
            tok::DITHER => self.dither = on,
            tok::SCISSOR_TEST => self.scissor_test = on,
            tok::SCISSOR if d.len() >= 4 => self.scissor = [d[0] as i32, d[1] as i32, d[2] as i32, d[3] as i32],
            tok::CULL_ON => self.cull = 1,
            tok::CULL_OFF => self.cull = 0,
            tok::CULL_FACE => self.cull_mode = if d.is_empty() { 0x405 } else { w0 },
            tok::FRONT_FACE => self.front_ccw = (w0 != 0x900) as u32,
            tok::POLYGON_MODE if d.len() >= 2 => {
                let m = d[1];
                match w0 {
                    0x404 => self.polymode[0] = m,
                    0x405 => self.polymode[1] = m,
                    _ => self.polymode = [m; 2],
                }
            }
            tok::LINE_WIDTH => self.line_width = f32::from_bits(w0).max(1.0),
            tok::LINE_STIPPLE_ON => self.set_state(|g| g.line_stipple = on, sink),
            tok::LINE_STIPPLE if d.len() >= 2 => self.set_state(|g| (g.line_repeat, g.line_pattern) = (w0 & 0xFF, d[1] & 0xFFFF), sink),
            tok::POLYGON_STIPPLE_ON => self.set_state(|g| g.poly_stipple = on, sink),
            tok::LOAD_POLYGON_STIPPLE if d.len() >= 32 => {
                let mut rows = [0u32; 32];
                rows.copy_from_slice(&d[..32]);
                self.set_state(|g| g.stipple_rows = rows, sink);
            }
            tok::FLUSH => self.end_raster(sink),
            tok::VALIDATE_BANKS => self.set_draw_bank(w0, sink),
            tok::DRAW_BUFFER => {
                self.end_raster(sink);
                self.draw_bits = w0;
                self.draw_words = (d.len() >= 3) as u32;
                self.draw_bits_swapped = d.get(1).copied().unwrap_or(1) & 0x7F;
                self.buffer_count = d.get(2).copied().unwrap_or(0) & 1;
            }
            tok::BEGIN_POINTS..=tok::BEGIN_POLYGON => {
                self.prim = cmd - tok::BEGIN_POINTS;
                self.nv = 0;
            }
            tok::END_POINTS..=tok::END_POLYGON => {
                self.end_primitive(sink);
                self.prim = prim::NONE;
                self.end_raster(sink);
            }
            tok::VERTEX4F => self.vertex(args_f32(d), sink),
            tok::INIT_FORMAT_VALUES => self.set_state(|g| g.pixel_format = w0 & 0x2700, sink),
            tok::INIT_RGB => self.ci = 0,
            tok::INIT_CI => self.ci = 1,
            tok::CLEAR_INDEX => self.clear_index = if w0 >> 16 == 0 { w0 } else { f32::from_bits(w0) as u32 },
            tok::RASTER_POS4F => {
                let w = self.transform(args_f32(d));
                let h = w.h;
                let inside = (0..3).all(|i| h[i].abs() <= h[3]);
                self.raster = [w.x, w.y, w.z];
                self.raster_color = w.c;
                self.raster_valid = (w.ok != 0 && inside) as u32;
            }
            tok::UPDATE_RASTER_POS => {
                let a = args_f32(d);
                self.raster[0] += a[0];
                self.raster[1] += a[1];
            }
            tok::LOAD_RASTER_POS_INFO => {
                self.color = self.raster_color;
                if self.ci != 0 { self.current_index = self.color[0] * 4095.0; }
            }
            tok::INDEX_MASK => self.set_state(|g| g.index_mask = w0 & 0xFFF, sink),
            tok::READ_BUFFER => self.read_back = (d.get(2) == Some(&0x405)) as u32,
            tok::SAVE_RSS => self.pixel_op = 1,
            tok::READ_TEXTURE if d.len() >= 5 => {
                self.begin_raster(sink);
                sink.rss_write(re::XFRMODE, d[4], false);
                sink.rss_write(re::XFRSIZE, d[1] << 16 | (d[0] & 0xFFFF), false);
                sink.rss_write(re::XFRCONTROL, TE_READ_START, false);
                self.tex_reading = 1;
            }
            // TRAM saved raw: the level the sampler points at (64 texels of
            // 32 bits a row from its page), byte for byte.
            tok::READ_TEXTURE_RAW if d.len() >= 3 => {
                self.begin_raster(sink);
                let size = self.te[te1::CONTEXT_REGS.iter().position(|&r| r == te_reg::TXSIZE).unwrap_or(0)];
                sink.rss_write(re::XFRMODE, d[1], false);
                sink.rss_write(re::XFRSIZE, d[0] << 16 | 1 << (size & 0xF), false);
                sink.rss_write(super::rss::reg::TE_RAW, 1, false);
                sink.rss_write(re::XFRCONTROL, TE_READ_START, false);
                self.tex_reading = 1;
            }
            // TRAM restored: the texels of the 0xDA8 rectangle come by host
            // DMA as they were saved.
            tok::WRITE_DMAGESETUP if d.len() >= 4 && d[3] == SEND_PIXELS_TEXTURE => {
                // Rows of the transfer mode's texels (RGB 3 bytes, RGBA 4).
                let (w, h) = (self.tl_rect[2], self.tl_rect[3]);
                let words = (w * te1::Te1::read_texel_bytes(self.xfrmode)).div_ceil(4);
                self.send_pixels = [words, 0, 0, h, 0, 1, SEND_PIXELS_TEXTURE, 0];
                self.send_pixels_pending = 1;
            }
            // glDrawPixels whose image comes by host DMA in one transfer:
            // the write half of glCopyPixels (Maya copies the front buffer
            // to the back after a full redraw, then redraws only what
            // changes: 0xDA8 rectangle 780 x 490, CI 16 bits a pixel, one
            // DMA line of 0xBA9F0 bytes, start 0xA7). The 0xDA8 transfer
            // size gives the rows and their width in transfer mode pixels.
            tok::WRITE_DMAGESETUP if d.len() >= 4 && d[3] == SEND_PIXELS_DRAW => {
                let xs = self.tl_rect[4];
                let (w, h) = (xs & 0xFFFF, xs >> 16);
                let words = (w * super::rss::bytes_per_pixel(self.xfrmode)).div_ceil(4);
                self.send_pixels = [words, 0, 0, h, 0, 1, SEND_PIXELS_DRAW, 0];
                self.send_pixels_pending = 1;
            }
            tok::RESTORE_RSS => {
                if self.tex_reading != 0 {
                    self.tex_reading = 0;
                    sink.rss_write(re::XFRCONTROL, 0, false);
                    sink.rss_write(super::rss::reg::TE_RAW, 0, false);
                }
                self.pixel_op = 0;
                self.end_raster(sink);
            }
            tok::GET_PIXELS => self.get_pixels(sink),
            tok::PIXEL_STATE | tok::INIT_PIXEL_STATE_ERAM if d.len() >= 7 => {
                let (addr, n) = (d[3], d[6] as usize);
                for (k, &v) in d[7..].iter().take(n).enumerate() {
                    self.pixel_state(addr + k as u32, v);
                }
            }
            tok::TEX_COORD => self.texcoord = args_f32(d),
            tok::POINT_SIZE => self.point_size = f32::from_bits(w0),
            tok::TEX_GENIV if d.len() >= 3 && d[0].wrapping_sub(0x2000) < 4 && d[1] == 0x2500 => {
                self.texgen_mode[(d[0] - 0x2000) as usize] = match d[2] {
                    0x2401 => 0,
                    0x2402 => 2,
                    _ => 1,
                };
            }
            tok::TEX_GEN_PLANE_HEADER if d.len() >= 2 && d[0].wrapping_sub(0x2000) < 4 => {
                self.texgen_plane = (d[0] - 0x2000) | ((d[1] == 0x2502) as u32) << 2;
            }
            tok::TEX_GEN_PLANE_DATA if d.len() >= 4 => {
                let p = args_f32(d);
                let i = (self.texgen_plane & 3) as usize;
                if self.texgen_plane & 4 != 0 {
                    // Eye planes are kept in eye space: times the inverse
                    // of the modelview current now (OpenGL), as clip planes.
                    let inv = math::invert(self.mv.get()).unwrap_or(math::IDENT);
                    let col = |c: usize| (0..4).map(|r| p[r] * inv[c * 4 + r]).sum::<f32>();
                    self.texgen_eye[i] = [col(0), col(1), col(2), col(3)];
                } else {
                    self.texgen_obj[i] = p;
                }
            }
            tok::ENABLE_CAP | tok::DISABLE_CAP if w0.wrapping_sub(0xC60) < 4 => {
                let bit = 1 << (w0 - 0xC60);
                if cmd == tok::ENABLE_CAP { self.texgen_on |= bit } else { self.texgen_on &= !bit }
            }
            tok::TEX_ON | tok::TEX_OFF => {
                self.tex_on = (cmd == tok::TEX_ON) as u32;
                if self.raster_loaded != 0 {
                    sink.rss_write(te_reg::TEXMODE1, self.texmode1(), false);
                }
            }
            tok::RSS_REG_SHADOW_IDX if d.len() >= 4 => {
                self.te_write(d[2] & 0x3FF, d[3], sink);
            }
            tok::SEND_PIXELS if d.len() >= 8 => {
                self.send_pixels.copy_from_slice(&d[..8]);
                self.send_pixels_pending = 1;
            }
            tok::BITMAP if d.len() >= 8 => {
                self.bitmap.copy_from_slice(&d[..8]);
                self.bitmap_pending = 1;
            }
            _ => return false,
        }
        true
    }

    /// Object coordinates to a vertex with clip, eye and window positions
    /// (window relative, y up).
    fn transform(&mut self, v: [f32; 4]) -> Wv {
        let mvp = self.mvp();
        let h = math::xform(&mvp, v);
        let e = math::xform(self.mv.get(), v);
        let (mut c, mut cb) = (self.color, self.color);
        if self.lt.on != 0 && self.unlit == 0 {
            if self.nm_dirty != 0 {
                self.nm_dirty = 0;
                self.lt.normal_matrix = math::normal_matrix(self.mv.get());
            }
            (c, cb) = self.lt.light_vertex(e, self.normal, self.color);
        }
        // With texturing, fog comes after the texture environment: the
        // fog factor goes with the vertex instead (`fog_after_texture`).
        let mut f = 1.0;
        if self.lt.fog_on != 0 {
            if self.fog_after_texture() {
                let z = if e[3] != 0.0 { (e[2] / e[3]).abs() } else { e[2].abs() };
                f = self.lt.fog_factor(z);
            } else {
                c = self.lt.fog_color(e, c);
                cb = self.lt.fog_color(e, cb);
            }
        }
        let mut w = Wv { c, cb, h, e, f, ..Default::default() };
        if self.tex_on != 0 {
            let tc = self.texgen(v, e);
            w.t = math::xform(self.tex.get(), tc);
        }
        w.oc = self.clip.outcode(&w);
        self.vp.project(w, [0.0, 0.0], 256.0)
    }

    /// The vertex's texture coordinates before the texture matrix: the
    /// current ones, with the generated ones where texgen is on (object
    /// or eye linear: the plane times the position; sphere map: from the
    /// reflection of the eye direction about the eye-space normal).
    fn texgen(&mut self, obj: [f32; 4], eye: [f32; 4]) -> [f32; 4] {
        let mut tc = self.texcoord;
        if self.texgen_on == 0 {
            return tc;
        }
        let dot = |p: &[f32; 4], v: [f32; 4]| p[0] * v[0] + p[1] * v[1] + p[2] * v[2] + p[3] * v[3];
        let mut sphere: Option<[f32; 2]> = None;
        for i in 0..4 {
            if self.texgen_on & (1 << i) == 0 {
                continue;
            }
            tc[i] = match self.texgen_mode[i] {
                0 => dot(&self.texgen_obj[i], obj),
                2 if i < 2 => {
                    let st = *sphere.get_or_insert_with(|| {
                        if self.nm_dirty != 0 {
                            self.nm_dirty = 0;
                            self.lt.normal_matrix = math::normal_matrix(self.mv.get());
                        }
                        let nm = &self.lt.normal_matrix;
                        let n0 = self.normal;
                        let n = [
                            nm[0] * n0[0] + nm[3] * n0[1] + nm[6] * n0[2],
                            nm[1] * n0[0] + nm[4] * n0[1] + nm[7] * n0[2],
                            nm[2] * n0[0] + nm[5] * n0[1] + nm[8] * n0[2],
                        ];
                        let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
                        let ln = len(n);
                        let n = [n[0] / ln, n[1] / ln, n[2] / ln];
                        let w = if eye[3] != 0.0 { eye[3] } else { 1.0 };
                        let u = [eye[0] / w, eye[1] / w, eye[2] / w];
                        let lu = len(u);
                        let u = [u[0] / lu, u[1] / lu, u[2] / lu];
                        let nu = n[0] * u[0] + n[1] * u[1] + n[2] * u[2];
                        let r = [u[0] - 2.0 * n[0] * nu, u[1] - 2.0 * n[1] * nu, u[2] - 2.0 * n[2] * nu];
                        let m = 2.0 * (r[0] * r[0] + r[1] * r[1] + (r[2] + 1.0) * (r[2] + 1.0)).sqrt().max(1e-20);
                        [r[0] / m + 0.5, r[1] / m + 0.5]
                    });
                    st[i]
                }
                2 => tc[i],
                _ => dot(&self.texgen_eye[i], eye),
            };
        }
        tc
    }

    fn vertex(&mut self, obj: [f32; 4], sink: &mut dyn Hq3Sink) {
        if self.prim == prim::NONE {
            return;
        }
        let v = self.transform(obj);
        let n = self.nv as usize;
        self.nv += 1;
        let mut poly = [Wv::default(); MAX_POLY];
        match self.prim {
            prim::POINTS => self.point(v, sink),
            prim::LINES => {
                self.vb[n % 2] = v;
                if n % 2 == 1 {
                    self.line(self.vb[0], v, true, sink);
                }
            }
            prim::LSTRIP | prim::LLOOP => {
                if n == 0 {
                    self.vb[0] = v;
                } else {
                    self.line(self.vb[1 + (n - 1) % 2], v, n == 1, sink);
                }
                self.vb[1 + n % 2] = v;
            }
            prim::TRIANGLES => {
                self.vb[n % 3] = v;
                if n % 3 == 2 {
                    poly[..3].copy_from_slice(&self.vb[..3]);
                    self.polygon(&mut poly, 3, v, sink);
                }
            }
            prim::TSTRIP => {
                // vb[0] older, vb[1] newer. Every other triangle is
                // reversed to keep the strip's winding; swaptmesh swaps the
                // pair and flips it too. The newest vertex provokes.
                if n == 0 {
                    self.tflip = 0;
                }
                if n >= 2 {
                    let (a, b) = (self.vb[0], self.vb[1]);
                    poly[..3].copy_from_slice(&if self.tflip == 0 { [a, b, v] } else { [b, a, v] });
                    self.tflip ^= 1;
                    self.polygon(&mut poly, 3, v, sink);
                }
                self.vb[0] = self.vb[1];
                self.vb[1] = v;
            }
            prim::TFAN => {
                if n == 0 {
                    self.vb[0] = v;
                } else {
                    if n >= 2 {
                        poly[..3].copy_from_slice(&[self.vb[0], self.vb[1], v]);
                        self.polygon(&mut poly, 3, v, sink);
                    }
                    self.vb[1] = v;
                }
            }
            prim::QUADS => {
                self.vb[n % 4] = v;
                if n % 4 == 3 {
                    poly[..4].copy_from_slice(&self.vb[..4]);
                    self.polygon(&mut poly, 4, v, sink);
                }
            }
            prim::QSTRIP => {
                self.vb[n % 4] = v;
                if n >= 3 && n % 2 == 1 {
                    let q = |k: usize| self.vb[(n - k) % 4];
                    poly[..4].copy_from_slice(&[q(3), q(2), q(0), q(1)]);
                    self.polygon(&mut poly, 4, v, sink);
                }
            }
            prim::POLYGON => {
                if n < MAX_POLY - 12 {
                    self.vb[n] = v;
                }
            }
            _ => {}
        }
    }

    /// END: close line loops, draw polygons.
    fn end_primitive(&mut self, sink: &mut dyn Hq3Sink) {
        let n = self.nv as usize;
        match self.prim {
            prim::LLOOP if n >= 2 => {
                let (last, first) = (self.vb[1 + (n - 1) % 2], self.vb[0]);
                self.line(last, first, false, sink);
            }
            prim::POLYGON if n >= 3 => {
                let n = n.min(MAX_POLY - 12);
                let mut v = [Wv::default(); MAX_POLY];
                v[..n].copy_from_slice(&self.vb[..n]);
                let prov = v[0];
                self.polygon(&mut v, n, prov, sink);
            }
            _ => {}
        }
    }

    /// Clip a convex polygon, cull it or draw it in its polygon mode (as
    /// a fan of triangles, its edges, or its vertices). `prov` carries the
    /// flat-shading colour.
    fn polygon(&mut self, v: &mut [Wv; MAX_POLY], n: usize, prov: Wv, sink: &mut dyn Hq3Sink) {
        let n = self.clip.clip_polygon(v, n);
        // Vertices made by clipping carry a parent's window position:
        // project them all again from their clip positions.
        for w in v[..n].iter_mut() {
            *w = self.vp.project(*w, [0.0, 0.0], 256.0);
        }
        if n < 3 || v[..n].iter().any(|w| w.ok == 0) {
            return;
        }
        // Window-space signed area: > 0 counter-clockwise (y up).
        let area: f32 = (0..n).map(|i| {
            let (a, b) = (&v[i], &v[(i + 1) % n]);
            a.x * b.y - b.x * a.y
        }).sum();
        let front = (area > 0.0) == (self.front_ccw != 0);
        if self.cull != 0 {
            let culled = match self.cull_mode {
                0x404 => front,
                0x408 => true,
                _ => !front,
            };
            if culled || area == 0.0 {
                return;
            }
        }
        // Two-sided lighting: back faces show the back colours.
        let mut prov = prov;
        if self.lt.on != 0 && self.lt.two_sided != 0 && !front {
            for w in v[..n].iter_mut() {
                w.c = w.cb;
            }
            prov.c = prov.cb;
        }
        let flat = (self.smooth == 0).then_some(prov.c);
        if flat.is_some() {
            for w in v[..n].iter_mut() {
                w.c = prov.c;
            }
        }
        match self.polymode[!front as usize] {
            0x1B00 => {
                for w in v[..n].to_vec() {
                    self.point_window(w, sink);
                }
                return;
            }
            0x1B01 => {
                for i in 0..n {
                    self.line_window(v[i], v[(i + 1) % n], i == 0, sink);
                }
                return;
            }
            _ => {}
        }
        let n = self.scissor_polygon(v, n);
        if n < 3 {
            return;
        }
        self.begin_raster(sink);
        self.fill_mode(if self.poly_stipple != 0 { FILL_POLY_STIPPLE } else { 0 }, sink);
        for k in 1..n - 1 {
            self.triangle(&v[0], &v[k], &v[k + 1], flat, sink);
        }
    }

    /// The scissor box as window-space half-planes, if the test is on:
    /// x0, x1 + 1, y0, y1 + 1.
    fn scissor_box(&self) -> Option<[f32; 4]> {
        let [x0, x1, y0, y1] = self.scissor_rect();
        (self.scissor_test != 0).then_some([x0 as f32, (x1 + 1) as f32, y0 as f32, (y1 + 1) as f32])
    }

    /// The scissor box, inclusive window coordinates [x0, x1, y0, y1]. The
    /// token carries [x, width - 1, y, height - 1] (traced: glScissor(30,
    /// 40, 100, 50) sends 30, 99, 40, 49), kept as sent for the readback.
    fn scissor_rect(&self) -> [i32; 4] {
        let [x, w1, y, h1] = self.scissor;
        [x, x + w1, y, y + h1]
    }

    /// Clip a window-space polygon to the scissor box (colours and depth
    /// interpolated linearly in window space).
    fn scissor_polygon(&self, v: &mut [Wv; MAX_POLY], n: usize) -> usize {
        let Some(b) = self.scissor_box() else { return n };
        let mut n = n;
        let mut tmp = [Wv::default(); MAX_POLY];
        for edge in 0..4 {
            let dist = |w: &Wv| match edge {
                0 => w.x - b[0],
                1 => b[1] - w.x,
                2 => w.y - b[2],
                _ => b[3] - w.y,
            };
            let mut m = 0;
            for i in 0..n {
                let (a, c) = (v[i], v[(i + 1) % n]);
                let (da, dc) = (dist(&a), dist(&c));
                if da >= 0.0 && m < MAX_POLY {
                    tmp[m] = a;
                    m += 1;
                }
                if (da >= 0.0) != (dc >= 0.0) && m < MAX_POLY {
                    tmp[m] = lerp_window(&a, &c, da / (da - dc));
                    m += 1;
                }
            }
            v[..m].copy_from_slice(&tmp[..m]);
            n = m;
            if n == 0 {
                break;
            }
        }
        n
    }

    /// A line from object-space vertices already transformed: clip, then
    /// draw. `restart` restarts the stipple pattern (each GL_LINES segment,
    /// the first segment of a strip or loop).
    fn line(&mut self, a: Wv, b: Wv, restart: bool, sink: &mut dyn Hq3Sink) {
        let Some((a, b)) = self.clip.clip_line(a, b) else { return };
        let (mut a, b) = (self.vp.project(a, [0.0, 0.0], 256.0), self.vp.project(b, [0.0, 0.0], 256.0));
        if a.ok == 0 || b.ok == 0 {
            return;
        }
        // Flat lines take the segment's second vertex colour.
        if self.smooth == 0 {
            a.c = b.c;
        }
        self.line_window(a, b, restart, sink);
    }

    /// A window-space line: scissor, then the RE4 GL line registers.
    fn line_window(&mut self, a: Wv, b: Wv, restart: bool, sink: &mut dyn Hq3Sink) {
        let (mut a, mut b) = (a, b);
        if let Some(bx) = self.scissor_box() {
            // Liang-Barsky against the box.
            let (dx, dy) = (b.x - a.x, b.y - a.y);
            let (mut t0, mut t1) = (0.0f32, 1.0f32);
            for (p, q) in [(-dx, a.x - bx[0]), (dx, bx[1] - a.x), (-dy, a.y - bx[2]), (dy, bx[3] - a.y)] {
                if p == 0.0 {
                    if q < 0.0 {
                        return;
                    }
                } else {
                    let r = q / p;
                    if p < 0.0 { t0 = t0.max(r) } else { t1 = t1.min(r) }
                }
            }
            if t0 > t1 {
                return;
            }
            let (oa, ob) = (a, b);
            a = lerp_window(&oa, &ob, t0);
            b = lerp_window(&oa, &ob, t1);
        }
        self.begin_raster(sink);
        self.fill_mode(if self.line_stipple != 0 { FILL_LINE_STIPPLE } else { 0 }, sink);
        if restart {
            sink.rss_write(re::LSCRL, self.line_repeat, false);
        }
        let len = (b.x - a.x).abs().max((b.y - a.y).abs());
        if len == 0.0 {
            return;
        }
        let pos = |f: f32| (f + POS_BIAS).to_bits();
        sink.rss_write(re::GLINE_XSTARTF, pos(a.x), false);
        sink.rss_write(re::GLINE_XSTARTF + 1, pos(a.y), false);
        sink.rss_write(re::GLINE_XSTARTF + 2, pos(b.x), false);
        sink.rss_write(re::GLINE_XSTARTF + 3, pos(b.y), false);
        sink.rss_write(re::GLINECONFIG, (self.line_width.round() as u32).max(1) - 1, false);
        let fixc = |c: f32| (c as f64 * 0xFF_F000 as f64) as i32 as u32;
        for k in 0..4 {
            sink.rss_write(re::RED + k as u32, fixc(a.c[k]), false);
            sink.rss_write(re::DRE + 2 * k as u32, fixc((b.c[k] - a.c[k]) / len), false);
        }
        let z64 = |v: f64| (v * 4096.0) as i64 as u64;
        let (z, dz) = (z64(a.z as f64), z64((b.z as f64 - a.z as f64) / len as f64));
        sink.rss_write(re::Z, (z >> 32) as u32, false);
        sink.rss_write(re::Z + 1, z as u32, false);
        sink.rss_write(re::Z + 4, (dz >> 32) as u32, false);
        sink.rss_write(re::Z + 5, dz as u32, false);
        if self.texturing() {
            // S/W, T/W, Q/W at the start and their steps per pixel along
            // the major axis (in the edge step registers).
            let [qa, qb] = normalised_q([tex_q(&a), tex_q(&b)]);
            for k in 0..3 {
                let (r, dr) = [(te_reg::SW, te_reg::DSWE), (te_reg::TW, te_reg::DTWE), (te_reg::WI, te_reg::DWIE)][k];
                put64(sink, r, qa[k] * te1::ITER_ONE);
                put64(sink, dr, (qb[k] - qa[k]) / len as f64 * te1::ITER_ONE);
            }
        }
        self.fog_plane(a.f as f64, (b.f as f64 - a.f as f64) / len as f64, 0.0, sink);
        sink.rss_write(re::IR, IR_GL_LINE, true);
    }

    /// A point (size 1): the pixel containing it, as a one-pixel GL line.
    fn point(&mut self, v: Wv, sink: &mut dyn Hq3Sink) {
        if v.oc != 0 || v.ok == 0 {
            return;
        }
        self.point_window(v, sink);
    }

    /// Whether a glBitmap header waits for its rows.
    pub fn wants_bitmap_rows(&self) -> bool {
        self.bitmap_pending != 0 || self.send_pixels_pending != 0
    }

    /// Words a pending SEND_PIXELS image needs before it can be drawn or
    /// loaded (words per row times rows, when the header says; 0 for
    /// glBitmap, drawn as it comes).
    pub fn pixel_words_needed(&self) -> usize {
        if self.send_pixels_pending == 0 {
            return 0;
        }
        let rows = (self.send_pixels[3] + 3 * self.send_pixels[4]) as usize;
        (self.send_pixels[0] as usize * rows).min(1 << 22)
    }

    /// glReadPixels: a DMA read block over the pixel block, in the read
    /// buffer (the depth buffer for the depth format, transfer mode format
    /// 2 / type 3, our reading); the host starts the transfer itself
    /// (raster interface register 5, see hq3.rs). The raster state stays
    /// loaded until RESTORE_RSS.
    fn get_pixels(&mut self, sink: &mut dyn Hq3Sink) {
        self.pixel_op = 2;
        self.begin_raster(sink);
        let depth = (self.xfrmode >> 4) & 0xF == 2 && self.xfrmode & 0xF == 3;
        let drb = if depth {
            super::rss::ZST_PAGE
        } else {
            let w = self.window.drb;
            let (a, b) = (w & 0x3FF, (w >> 10) & 0x3FF);
            (w & !0x3FF) | if (self.read_back != 0) != (self.swapped != 0) { b } else { a }
        };
        sink.rss_write(re::DRBPOINTERS, drb, false);
        self.fill_mode(FILL_DMA_READ, sink);
        sink.rss_write(re::XFRMODE, self.xfrmode, false);
        sink.rss_write(re::IR_ALIAS, IR_BLOCK_SETUP, false);
        sink.rss_write(re::BLOCK_XYSTARTI, self.pixel_block[0], false);
        sink.rss_write(re::BLOCK_XYENDI, self.pixel_block[1], true);
    }

    /// glDrawPixels' rows (FIFO pixel data, `send_pixels[0]` words a row,
    /// bottom row first) at the raster position, as the GE streams them:
    /// a raster-engine DMA write block the size of the image, the host
    /// format from the transfer mode (the driver sets `xfrmode` itself), so
    /// the pixels go through the raster engine's format conversion, plane
    /// masks and clipping. Pixel transfer stages (index shift/offset, maps,
    /// zoom) are taken as identity.
    fn draw_pixels(&mut self, words: &[u32], sink: &mut dyn Hq3Sink) {
        self.send_pixels_pending = 0;
        let wpr = self.send_pixels[0] as usize;
        if self.raster_valid == 0 || wpr == 0 || words.len() < wpr {
            return;
        }
        // Rows: header words 3 + 3 * 4 (mandel 1 + 0, snoop 2 + 3 * 18 =
        // 56), else what came.
        let hdr_rows = (self.send_pixels[3] + 3 * self.send_pixels[4]) as usize;
        let rows = if hdr_rows > 0 { hdr_rows.min(words.len() / wpr) } else { words.len() / wpr };
        let bpp = super::rss::bytes_per_pixel(self.xfrmode) as usize;
        // The host's pixels may carry fewer components than the raster
        // engine's format: glCopyPixels draws RGB16 back in RGBA16 (the GE's
        // pixel pipeline adds alpha). The image's size (pixel state 0xDA8's
        // transfer size, when its rows match) gives the host's pixel size.
        let xs = self.tl_rect[4];
        let host_bpp = match (xs & 0xFFFF) as usize {
            w if w > 0 && (xs >> 16) as usize == rows && wpr * 4 % w == 0 => wpr * 4 / w,
            _ => bpp,
        };
        let comp = if self.xfrmode & 0xF == 1 { 2 } else { 1 };
        let words: std::borrow::Cow<[u32]> = if host_bpp != bpp && host_bpp % comp == 0 && host_bpp < bpp {
            // Widen each pixel: its components, then ones (alpha).
            let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
            let mut out: Vec<u8> = Vec::with_capacity(bytes.len() / host_bpp * bpp + 8);
            for row in bytes.chunks(wpr * 4).take(rows) {
                let mut line: Vec<u8> = Vec::new();
                for px in row.chunks_exact(host_bpp) {
                    line.extend_from_slice(px);
                    line.resize(line.len() + bpp - host_bpp, 0xFF);
                }
                line.resize(line.len().div_ceil(4) * 4, 0);
                out.extend(line);
            }
            let wpr2 = out.len() / rows.max(1) / 4;
            let ws: Vec<u32> = out.chunks(4).map(|c| u32::from_be_bytes([c[0], c[1], c[2], c[3]])).collect();
            return self.draw_pixels_words(&ws, wpr2, rows, bpp, sink);
        } else {
            std::borrow::Cow::Borrowed(words)
        };
        self.draw_pixels_words(&words, wpr, rows, bpp, sink);
    }

    /// glDrawPixels' rows, `wpr` words each, in the raster engine's format
    /// (`bpp` bytes a pixel).
    fn draw_pixels_words(&mut self, words: &[u32], wpr: usize, rows: usize, bpp: usize, sink: &mut dyn Hq3Sink) {
        // Pixel zoom (same both ways, any factor): output pixel i takes
        // source pixel floor(i / zoom), as OpenGL's zoomed rectangles do
        // (the desks overview shrinks its miniatures, snoop magnifies 6x).
        let zoom = if self.zoom_inv > 0.0 { (1.0 / self.zoom_inv).clamp(1.0 / 64.0, 64.0) } else { 1.0 };
        // A large image comes in chunks (glCopyPixels draws back 80 pixels
        // wide): pixel state 0xDA8 holds the chunk's place in the image
        // (x0, y0, x1, y1, transfer size), relative to the raster position.
        let xs = self.tl_rect[4];
        let (ox, oy) = if xs != 0 && (xs >> 16) as usize == rows && (xs & 0xFFFF) as usize == wpr * 4 / bpp.max(1) {
            (self.tl_rect[0] as i32 as f32 * zoom, self.tl_rect[1] as i32 as f32 * zoom)
        } else {
            (0.0, 0.0)
        };
        let (x0, y0) = ((self.raster[0] + ox).floor() as i32, (self.raster[1] + oy).floor() as i32);
        let src_w = wpr * 4 / bpp;
        let out_w = ((src_w as f32 * zoom).round() as i32).max(1);
        let out_h = ((rows as f32 * zoom).round() as i32).max(1);
        // The scissor box (window coordinates, inclusive) cuts the
        // rectangle, as it does every fragment.
        let (mut cx0, mut cx1, mut cy0, mut cy1) = (x0, x0 + out_w - 1, y0, y0 + out_h - 1);
        if self.scissor_test != 0 {
            let [sx0, sx1, sy0, sy1] = self.scissor_rect();
            (cx0, cx1, cy0, cy1) = (cx0.max(sx0), cx1.min(sx1), cy0.max(sy0), cy1.min(sy1));
        }
        if cx0 > cx1 || cy0 > cy1 {
            return;
        }
        let (w, h) = (cx1 - cx0 + 1, cy1 - cy0 + 1);
        self.begin_raster(sink);
        if (self.xfrmode >> 4) & 0xF == 2 && self.xfrmode & 0xF == 3 {
            // Depth pixels (glCopyPixels of depth draws them back).
            sink.rss_write(re::DRBPOINTERS, super::rss::ZST_PAGE, false);
        }
        self.fill_mode(FILL_DMA_WRITE, sink);
        sink.rss_write(re::XFRMODE, self.xfrmode, false);
        sink.rss_write(re::XFRSIZE, (h as u32) << 16 | w as u32, false);
        sink.rss_write(re::IR_ALIAS, IR_BLOCK_SETUP, false);
        sink.rss_write(re::BLOCK_XYSTARTI, xy(cx0, cy0), false);
        sink.rss_write(re::BLOCK_XYENDI, xy(cx1, cy1), true);
        let src_px = |r: usize, c: usize| -> [u8; 8] {
            let mut px = [0u8; 8];
            for (k, b) in px.iter_mut().enumerate().take(bpp) {
                let i = r * wpr * 4 + c * bpp + k;
                *b = words.get(i / 4).map_or(0, |w| (w >> (24 - 8 * (i % 4))) as u8);
            }
            // GE pixels have component order RGBA after HQ formatting;
            // the RSS's byte RGBA transfer is a packed ABGR X pixel.
            if bpp == 4 && self.xfrmode & 0xFF == 0x80 {
                px[..4].reverse();
            }
            px
        };
        for (line, oy) in (cy0 - y0..=cy1 - y0).enumerate() {
            let sr = ((oy as f32 / zoom) as usize).min(rows - 1);
            let mut bytes = Vec::with_capacity(w as usize * bpp);
            for ox in cx0 - x0..=cx1 - x0 {
                let sc = ((ox as f32 / zoom) as usize).min(src_w - 1);
                bytes.extend_from_slice(&src_px(sr, sc)[..bpp]);
            }
            sink.dma_write_line(line as u32, &bytes);
        }
        self.end_raster(sink);
    }

    /// glBitmap's rows (FIFO pixel data words, big-endian bytes): rows of
    /// the header's length in bits, padded to bytes, bottom row first, most
    /// significant bit leftmost. Set bits are fragments of the raster
    /// colour and depth at the raster position less the origin; the raster
    /// position then moves. Each run of set bits goes through the GL
    /// fragment path as a one-pixel-high rectangle.
    pub fn bitmap_rows(&mut self, words: &[u32], sink: &mut dyn Hq3Sink) {
        self.ensure_init();
        if self.send_pixels_pending != 0 {
            if self.send_pixels[6] == SEND_PIXELS_TEXTURE {
                self.tex_load(words, sink);
            } else {
                self.draw_pixels(words, sink);
            }
            return;
        }
        self.bitmap_pending = 0;
        let h = self.bitmap;
        let (bits, rows) = (h[1] as usize, h[2] as usize);
        let f = |i: usize| f32::from_bits(h[i]);
        let (xorig, yorig, xmove, ymove) = (f(3), f(4), f(5), f(6));
        if self.raster_valid == 0 {
            return;
        }
        let x0 = (self.raster[0] - xorig).floor();
        let y0 = (self.raster[1] - yorig).floor();
        let stride = bits.div_ceil(8);
        let byte = |i: usize| words.get(i / 4).map_or(0, |w| (w >> (24 - 8 * (i % 4))) as u8);
        let mut v = Wv { z: self.raster[2], c: self.raster_color, cb: self.raster_color, ok: 1, ..Default::default() };
        for r in 0..rows {
            let set = |c: usize| byte(r * stride + c / 8) >> (7 - c % 8) & 1 != 0;
            let mut c = 0;
            while c < bits {
                if !set(c) {
                    c += 1;
                    continue;
                }
                let start = c;
                while c < bits && set(c) {
                    c += 1;
                }
                let (xa, xb, ya) = (x0 + start as f32, x0 + c as f32, y0 + r as f32);
                let mut quad = [Wv::default(); MAX_POLY];
                for (k, (x, y)) in [(xa, ya), (xb, ya), (xb, ya + 1.0), (xa, ya + 1.0)].into_iter().enumerate() {
                    v.x = x;
                    v.y = y;
                    quad[k] = v;
                }
                let n = self.scissor_polygon(&mut quad, 4);
                if n < 3 {
                    continue;
                }
                self.begin_raster(sink);
                self.fill_mode(0, sink);
                for k in 1..n - 1 {
                    self.triangle(&quad[0], &quad[k], &quad[k + 1], Some(self.raster_color), sink);
                }
            }
        }
        self.raster[0] += xmove;
        self.raster[1] += ymove;
    }

    fn point_window(&mut self, v: Wv, sink: &mut dyn Hq3Sink) {
        let size = self.point_size.round().max(1.0);
        let (mut a, mut b) = (v, v);
        if size <= 1.0 {
            let x = v.x.floor();
            a.x = x + 0.25;
            b.x = x + 0.75;
            self.line_window(a, b, true, sink);
            return;
        }
        // A square of size x size pixels, centred on the nearest pixel
        // corner (even sizes) or on the pixel's centre (odd), as OpenGL
        // places non-antialiased points: a horizontal line over those
        // pixels, that wide.
        let even = size as i32 % 2 == 0;
        let (cx, cy) = if even { ((v.x + 0.5).floor(), (v.y + 0.5).floor()) } else { (v.x.floor() + 0.5, v.y.floor() + 0.5) };
        let left = cx - size / 2.0;
        a.x = left + 0.25;
        b.x = left + size - 0.25;
        a.y = cy;
        b.y = cy;
        let width = std::mem::replace(&mut self.line_width, size);
        self.line_window(a, b, true, sink);
        self.line_width = width;
    }

    /// The PP1 per-fragment registers for the current state (layouts in
    /// rss.rs, provisional).
    fn load_fragment_state(&mut self, sink: &mut dyn Hq3Sink) {
        let zmask = if self.depth_mask != 0 { 0xFF_FFFF << 4 } else { 0 };
        sink.rss_write(re::ZMODE, if self.depth_test != 0 { zmask | 1 << 3 | self.depth_func } else { 0 }, false);
        sink.rss_write(re::AFUNCMODE, if self.alpha_test != 0 { self.alpha_ref << 4 | 1 << 3 | self.alpha_func } else { 7 }, false);
        let st = if self.stencil != 0 {
            self.st_ref << 16 | self.st_ops[2] << 12 | self.st_ops[1] << 8 | self.st_ops[0] << 4 | 1 << 3 | self.st_func
        } else {
            7
        };
        sink.rss_write(re::STENCILMODE, st, false);
        sink.rss_write(re::STENCILMASK, self.st_wmask << 8 | self.st_cmask, false);
        sink.rss_write(re::BLENDFACTOR, if self.blend != 0 { 1 << 8 | self.blend_dst << 4 | self.blend_src } else { 0 }, false);
        let op = if self.logic != 0 { self.logic_op } else { 3 };
        sink.rss_write(re::PP1FILLMODE, (self.pp1_base() & !(0xF << 26)) | op << 26, false);
        let (lsb, msb) = self.color_write_masks();
        sink.rss_write(re::COLORMASKLSBSA, lsb, false);
        sink.rss_write(re::COLORMASKLSBSB, lsb, false);
        sink.rss_write(re::COLORMASKMSBS, msb, false);
        sink.rss_write(re::LSPAT, self.line_pattern, false);
        if self.poly_stipple != 0 {
            for (i, row) in self.stipple_rows.iter().enumerate() {
                sink.rss_write(re::DEVICE_ADDR, super::rss::POLY_STIPPLE_RAM + i as u32, false);
                sink.rss_write(re::DEVICE_DATA, *row, false);
            }
        }
    }

    /// Fragment state changed: a batch already loaded picks it up now.
    fn set_state(&mut self, f: impl FnOnce(&mut Self), sink: &mut dyn Hq3Sink) {
        f(self);
        if self.raster_loaded != 0 {
            self.load_fragment_state(sink);
        }
    }

    /// The fill mode for the primitives that follow, written when it
    /// changes within a batch.
    fn fill_mode(&mut self, fm: u32, sink: &mut dyn Hq3Sink) {
        if self.fill_loaded != fm {
            self.fill_loaded = fm;
            sink.rss_write(re::FILLMODE, fm, false);
        }
    }

    /// SCHEDULE_SWAP reached the GE: front and back trade places for what
    /// follows (the display flips at the next retrace, by the kernel's
    /// BUF_SELECT write).
    pub fn swap_buffers(&mut self, sink: &mut dyn Hq3Sink) {
        self.ensure_init();
        self.end_raster(sink);
        if self.banks_known == 0 {
            self.swapped ^= 1;
        }
    }

    /// The bank GL draws into, as the kernel tracks it (MgrasValidateBanks:
    /// the window's displayed bank xor 1, sent as VALIDATE_BANKS before
    /// each SCHEDULE_SWAP for the frame after it, or stored in a parked
    /// context's image, words 16-17). 1 is B (DRBpointers bits 19:10),
    /// where GL_BACK draws unswapped (DRAW_BUFFER [4, 1]); 0 is A. Traced
    /// with Maya: [1, 1], swap, frame drawn in B; [0, 0], swap, frame in
    /// A. Counting swaps instead loses the phase for good on any swap the
    /// GE does not see, and every other frame then lands in the shown
    /// buffer.
    pub fn set_draw_bank(&mut self, bank: u32, sink: &mut dyn Hq3Sink) {
        self.end_raster(sink);
        self.swapped = (bank & 1 == 0) as u32;
        self.banks_known = 1;
    }

    /// The buffers drawn into, from DRAW_BUFFER (libGLcore's
    /// __glMgrasDrawBuffer always sends three words): the first word
    /// before an odd number of swaps, the second after. 1 is buffer A; B is
    /// 2 in 12-bit visuals and 4 in 24-bit ones (GL_FRONT [1, 2] or [1, 4],
    /// GL_BACK [2, 1] or [4, 1], both [3, 3] or [5, 5], traced: every
    /// double-buffered demo sends [4, 1]); 0x4N the overlay planes (aux
    /// buffers, [0x48, 0x48]); 0 nothing. None for one-word tokens.
    fn draw_mask(&self) -> Option<u32> {
        (self.draw_words != 0).then(|| if self.swapped != 0 { self.draw_bits_swapped } else { self.draw_bits & 0x7F })
    }

    /// The DRBpointers value for the colour buffer drawn into: the window's
    /// pointers with bits 9:0 set to the page drawn (A in bits 9:0, B in
    /// 19:10 of the window's value); unchanged for both buffers (PP1 draw
    /// field 3 writes both pages) and for the overlay.
    fn draw_pointers(&self) -> u32 {
        let drb = self.window.drb;
        let a = drb & 0x3FF;
        // A single-buffered window has no second page: B is A.
        let b = match (drb >> 10) & 0x3FF {
            0 => a,
            b => b,
        };
        let page = match self.draw_mask() {
            Some(m) if m & 0x70 == 0x40 => return drb,
            Some(m) if m & 1 != 0 && m & 0xE != 0 => return drb,
            Some(m) => if m & 0xE != 0 { b } else { a },
            None => {
                // One-word tokens: the back bits, from the swap state.
                let back = self.draw_bits & 0xC != 0;
                if back != (self.swapped != 0) { b } else { a }
            }
        };
        (drb & !0x3FF) | page
    }

    /// Load the GL window's raster state for a batch of GL primitives (the
    /// RSS saves the X server's first and turns Y-flip off).
    pub fn begin_raster(&mut self, sink: &mut dyn Hq3Sink) {
        if self.raster_loaded != 0 {
            return;
        }
        self.raster_loaded = 1;
        sink.gl_bracket(true);
        let w = self.window;
        if w.valid != 0 {
            sink.rss_write(re::XYWIN, w.origin, false);
            for (n, [x, y]) in w.masks.iter().enumerate() {
                sink.rss_write(re::SCRMSK1X + 2 * n as u32, *x, false);
                sink.rss_write(re::SCRMSK1X + 2 * n as u32 + 1, *y, false);
            }
            sink.rss_write(re::WINMODE, w.mode, false);
            sink.rss_write(re::PP1WINMODE, w.pp1winmode, false);
            sink.rss_write(re::DRBPOINTERS, self.draw_pointers(), false);
        }
        self.fill_loaded = 0;
        sink.rss_write(re::FILLMODE, 0, false);
        self.load_fragment_state(sink);
        self.load_te(sink);
    }

    /// Put the X server's raster state back after a GL batch.
    pub fn end_raster(&mut self, sink: &mut dyn Hq3Sink) {
        if self.raster_loaded == 0 {
            return;
        }
        self.raster_loaded = 0;
        sink.gl_bracket(false);
    }

    /// A fast-fill block over the window's first screen mask, cut to the
    /// scissor box (window coordinates), in `color` (12-bit components:
    /// red, green, blue, alpha) through plane masks `lsb` / `msb`, into
    /// buffer page `drb`. The fragment tests are off for clears.
    fn clear_block(&mut self, color: [u32; 4], lsb: u32, msb: u32, drb: u32, sink: &mut dyn Hq3Sink) {
        let w = self.window;
        if w.valid == 0 {
            sink.note("GL clear before the kernel gave the window".into());
            return;
        }
        let (ox, oy) = ((w.origin & 0xFFFF) as i32, (w.origin >> 16) as i32);
        let [mx, my] = w.masks[0];
        let (mut x0, mut x1) = ((mx >> 16) as i32 - ox, (mx & 0xFFFF) as i32 - ox);
        let (mut y0, mut y1) = ((my >> 16) as i32 - oy, (my & 0xFFFF) as i32 - oy);
        if self.scissor_test != 0 {
            let [sx0, sx1, sy0, sy1] = self.scissor_rect();
            (x0, x1, y0, y1) = (x0.max(sx0), x1.min(sx1), y0.max(sy0), y1.min(sy1));
            if x0 > x1 || y0 > y1 {
                return;
            }
        }
        self.begin_raster(sink);
        sink.rss_write(re::ZMODE, 0, false);
        sink.rss_write(re::AFUNCMODE, 7, false);
        sink.rss_write(re::STENCILMODE, 7, false);
        sink.rss_write(re::BLENDFACTOR, 0, false);
        sink.rss_write(re::PP1FILLMODE, self.pp1_base(), false);
        sink.rss_write(re::DRBPOINTERS, drb, false);
        sink.rss_write(re::COLORMASKLSBSA, lsb, false);
        sink.rss_write(re::COLORMASKLSBSB, lsb, false);
        sink.rss_write(re::COLORMASKMSBS, msb, false);
        self.fill_mode(FILL_FAST, sink);
        for (k, c) in color.iter().enumerate() {
            sink.rss_write(re::FILL_COLOR_R + k as u32, *c, false);
        }
        sink.rss_write(re::IR_ALIAS, IR_BLOCK_SETUP, false);
        sink.rss_write(re::BLOCK_XYSTARTI, xy(x0, y0), false);
        sink.rss_write(re::BLOCK_XYENDI, xy(x1, y1), true);
        sink.rss_write(re::DRBPOINTERS, self.draw_pointers(), false);
        self.load_fragment_state(sink);
        self.end_raster(sink);
    }

    /// The PP1 fill mode GL drawing uses: 24-bit RGB, or 12-bit colour
    /// index (pixel type 6) in a colour-index context. Main pixels use our
    /// canonical storage format; overlays retain the driver's CI8 format.
    fn pp1_base(&self) -> u32 {
        let pp1 = if self.ci != 0 { (PP1_RGB24_BUFFER_A & !0x700) | 0x600 } else { PP1_RGB24_BUFFER_A };
        match self.draw_mask() {
            // The overlay: its draw field, buffer count and the driver's
            // pixel format.
            Some(m) if m & 0x70 == 0x40 => {
                let pp1 = (pp1 & !((0x7F << 14) | (1 << 11))) | m << 14 | self.buffer_count << 11;
                (pp1 & !0x2700) | self.pixel_format
            }
            // A and B: draw field 3.
            Some(m) if m & 1 != 0 && m & 0xE != 0 => (pp1 & !(0x7F << 14)) | 3 << 14,
            _ => pp1,
        }
    }

    /// Plane masks in the RSS storage layout for the selected GL buffer.
    fn color_write_masks(&self) -> (u32, u32) {
        match self.draw_mask() {
            Some(0) => return (0, 0),
            Some(m) if m & 0x70 == 0x40 => return (0, self.index_mask & 0xFF),
            _ => {}
        }
        if self.ci != 0 { return (self.index_mask & 0xFFF, 0); }
        let cm = self.color_mask;
        ((cm & 1) * 0xFF | (cm >> 1 & 1) * 0xFF00 | (cm >> 2 & 1) * 0xFF_0000, (cm >> 3 & 1) * 0xFF)
    }

    /// glClear's colour part: the clear colour through the colour mask.
    fn clear_color_buffer(&mut self, sink: &mut dyn Hq3Sink) {
        let drb = self.draw_pointers();
        let (lsb, msb) = self.color_write_masks();
        if self.ci != 0 {
            self.clear_block([self.clear_index & 0xFFF, 0, 0, 0], lsb, msb, drb, sink);
            return;
        }
        let q = |c: f32| ((c.clamp(0.0, 1.0) * 255.0).round() as u32) << 4;
        let c = self.clear_color;
        self.clear_block([q(c[0]), q(c[1]), q(c[2]), q(c[3])], lsb, msb, drb, sink);
    }

    /// glClear's depth (24-bit Z, planes 23:0) or stencil (planes 31:24,
    /// through the stencil write mask) part, in the ZST buffer.
    fn clear_depth_stencil(&mut self, depth: bool, sink: &mut dyn Hq3Sink) {
        let zst = super::rss::ZST_PAGE;
        if depth {
            if self.depth_mask == 0 {
                return;
            }
            let z = self.clear_depth;
            let c = [(z & 0xFF) << 4, ((z >> 8) & 0xFF) << 4, ((z >> 16) & 0xFF) << 4, 0];
            self.clear_block(c, 0xFF_FFFF, 0, zst, sink);
        } else {
            let s = self.clear_stencil << 4;
            let m = self.st_wmask;
            self.clear_block([0, 0, 0, s], 0, m, zst, sink);
        }
    }

    fn triangle(&mut self, p: &Wv, q: &Wv, r: &Wv, flat: Option<[f32; 4]>, sink: &mut dyn Hq3Sink) {
        let mut v = [p, q, r];
        v.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal));
        let (cw, bw, aw) = (v[0], v[1], v[2]);
        let major_y = cw.y - aw.y;
        if major_y <= 0.0 {
            return;
        }
        self.stats_triangles += 1;
        let major_x = aw.x - cw.x;
        let minor_y = cw.y - bw.y;
        let dxdy0 = major_x / major_y;
        let perc = minor_y / major_y;
        let dxdy1 = if minor_y != 0.0 { (bw.x - cw.x) / minor_y } else { 0.0 };
        let dxdy2 = if bw.y - aw.y != 0.0 { (aw.x - bw.x) / (bw.y - aw.y) } else { 0.0 };
        let (ymax, ymid) = (cw.y.floor(), bw.y.floor());
        let xmid = cw.x + perc * major_x;
        let ltor = xmid <= bw.x;
        let span_x = bw.x - xmid;
        let dy = cw.y - ymax;
        let x0 = cw.x + dxdy0 * dy;
        let x1 = cw.x + dxdy1 * dy;
        let x2 = bw.x + dxdy2 * (bw.y - ymid);
        let pos = |f: f32| (f + POS_BIAS).to_bits();
        sink.rss_write(re::TRI_YMID, pos(bw.y), false);
        sink.rss_write(re::TRI_YMID + 1, pos(aw.y), false);
        for (reg, s) in [(re::TRI_DXDY0, dxdy0), (re::TRI_DXDY1, dxdy1), (re::TRI_DXDY2, dxdy2)] {
            let f = (s as f64 * (1u64 << 24) as f64) as i64 as u64;
            sink.rss_write(reg, (f >> 32) as u32, false);
            sink.rss_write(reg + 1, f as u32, false);
        }
        sink.rss_write(re::TRI_X0, pos(x0), false);
        sink.rss_write(re::TRI_X0 + 1, pos(x1), false);
        sink.rss_write(re::TRI_X2, pos(x2), false);
        sink.rss_write(re::TRI_X2 + 1, pos(cw.y), false);
        // Colour: start value at the first pixel, steps along the span and
        // down the major edge.
        let dx_x = (if ltor { x0.ceil() } else { x0.floor() }) - cw.x;
        let n1 = dxdy0.trunc();
        let fixc = |c: f32| (c as f64 * 0xFF_F000 as f64) as i32 as u32;
        for k in 0..4 {
            let (start, dre, drx) = match flat {
                Some(c) => (c[k], 0.0, 0.0),
                None => {
                    let (a, b, c) = (aw.c[k], bw.c[k], cw.c[k]);
                    let mid = c + perc * (a - c);
                    let drx = if span_x != 0.0 { (b - mid) / span_x } else { 0.0 };
                    let drdy = (a - c - drx * major_x) / major_y;
                    let start = c + drdy * dy + drx * dx_x;
                    let mut dre = drdy + n1 * drx;
                    let drx_out = if ltor { drx } else { -drx };
                    if ltor ^ (dxdy0 >= 0.0) {
                        dre -= drx_out;
                    }
                    (start, dre, drx_out)
                }
            };
            sink.rss_write(re::RED + k as u32, fixc(start), false);
            sink.rss_write(re::DRE + 2 * k as u32, fixc(dre), false);
            sink.rss_write(re::DRE + 2 * k as u32 + 1, fixc(drx), false);
        }
        if self.depth_test != 0 {
            // Z: the same plane setup, 64-bit, value * 0x1000.
            let (a, b, c) = (aw.z as f64, bw.z as f64, cw.z as f64);
            let (major_x, major_y) = (major_x as f64, major_y as f64);
            let mid = c + perc as f64 * (a - c);
            let dzx = if span_x != 0.0 { (b - mid) / span_x as f64 } else { 0.0 };
            let dzdy = (a - c - dzx * major_x) / major_y;
            let z = c + dzdy * dy as f64 + dzx * dx_x as f64;
            let mut dze = dzdy + n1 as f64 * dzx;
            let dzx_out = if ltor { dzx } else { -dzx };
            if ltor ^ (dxdy0 >= 0.0) {
                dze -= dzx_out;
            }
            for (k, val) in [z, dzx_out, dze].into_iter().enumerate() {
                let f = (val * 4096.0) as i64 as u64;
                sink.rss_write(re::Z + 2 * k as u32, (f >> 32) as u32, false);
                sink.rss_write(re::Z + 2 * k as u32 + 1, f as u32, false);
            }
        }
        if self.fog_after_texture() {
            let (a, b, c) = (aw.f as f64, bw.f as f64, cw.f as f64);
            let mid = c + perc as f64 * (a - c);
            let dx = if span_x != 0.0 { (b - mid) / span_x as f64 } else { 0.0 };
            let ddown = (a - c - dx * major_x as f64) / major_y as f64;
            let start = c + ddown * dy as f64 + dx * dx_x as f64;
            let mut de = ddown + n1 as f64 * dx;
            let dx_out = if ltor { dx } else { -dx };
            if ltor ^ (dxdy0 >= 0.0) {
                de -= dx_out;
            }
            self.fog_plane(start, dx_out, de, sink);
        } else {
            self.fog_plane(1.0, 0.0, 0.0, sink);
        }
        if self.texturing() {
            // S/W, T/W, Q/W: perspective-correct texture coordinates are
            // linear in window space divided by clip w. Each gets the same
            // plane setup as Z, at 2^32 (te1::ITER_ONE), plus d/dy.
            let [qa, qb, qc] = normalised_q([tex_q(aw), tex_q(bw), tex_q(cw)]);
            let (major_x, major_y) = (major_x as f64, major_y as f64);
            let mut planes = [[0.0f64; 4]; 3];
            for (k, pl) in planes.iter_mut().enumerate() {
                let (a, b, c) = (qa[k], qb[k], qc[k]);
                let mid = c + perc as f64 * (a - c);
                let dx = if span_x != 0.0 { (b - mid) / span_x as f64 } else { 0.0 };
                let ddown = (a - c - dx * major_x) / major_y;
                let start = c + ddown * dy as f64 + dx * dx_x as f64;
                let mut de = ddown + n1 as f64 * dx;
                let dx_out = if ltor { dx } else { -dx };
                if ltor ^ (dxdy0 >= 0.0) {
                    de -= dx_out;
                }
                *pl = [start, dx_out, de, -ddown];
            }
            let put = |r: u32, v: f64, sink: &mut dyn Hq3Sink| put64(sink, r, v * te1::ITER_ONE);
            let [s, t, w] = planes;
            for (r, v) in [
                (te_reg::SW, s[0]), (te_reg::TW, t[0]), (te_reg::WI, w[0]),
                (te_reg::DWIE, w[2]), (te_reg::DWIX, w[1]), (te_reg::DWIY, w[3]),
                (te_reg::DSWE, s[2]), (te_reg::DTWE, t[2]), (te_reg::DSWX, s[1]),
                (te_reg::DTWX, t[1]), (te_reg::DSWY, s[3]), (te_reg::DTWY, t[3]),
            ] {
                put(r, v, sink);
            }
        }
        sink.rss_write(re::IR, if ltor { IR_AREA_LTOR } else { IR_AREA_RTOL }, true);
    }

    /// Texturing is on for the primitives that follow (GE on, TE enabled).
    fn texturing(&self) -> bool {
        self.tex_on != 0 && self.texmode1() & te1::TEXMODE1_ENABLE != 0
    }

    /// Fog is applied per fragment, after the texture environment (OpenGL's
    /// order), rather than to vertex colours.
    fn fog_after_texture(&self) -> bool {
        self.lt.fog_on != 0 && self.texturing()
    }

    /// The per-fragment fog of the next primitive (model-private registers,
    /// `rss::reg::FOG_*`): the fog colour, the fog factor's plane (start,
    /// step along the span, step down the edge; along a line, its step per
    /// pixel), or off.
    fn fog_plane(&mut self, start: f64, dx: f64, de: f64, sink: &mut dyn Hq3Sink) {
        use super::rss::reg as rr;
        if !self.fog_after_texture() {
            sink.rss_write(rr::FOG_ON, 0, false);
            return;
        }
        let c = self.lt.fog_color.map(|v| (v.clamp(0.0, 1.0) * 4095.0).round() as u32);
        sink.rss_write(rr::FOG_RG, c[1] << 12 | c[0], false);
        sink.rss_write(rr::FOG_B, c[2], false);
        put64(sink, rr::FOG_F, start * te1::ITER_ONE);
        put64(sink, rr::FOG_F + 2, dx * te1::ITER_ONE);
        put64(sink, rr::FOG_F + 4, de * te1::ITER_ONE);
        sink.rss_write(rr::FOG_ON, 1, false);
    }

    /// A GE pixel state word (PIXEL_STATE / INIT_PIXEL_STATE_ERAM) at
    /// `addr`: the pixel zoom and the texture loader's blocks are kept.
    fn pixel_state(&mut self, addr: u32, v: u32) {
        match addr {
            0x192 => self.zoom_inv = f32::from_bits(v),
            0xDA8..=0xDAC => self.tl_rect[(addr - 0xDA8) as usize] = v,
            0xDB1..=0xDB6 => self.tl_dest[(addr - 0xDB1) as usize] = v,
            0xD0A => self.tl_scale = v,
            0xDBB => self.tl_elem = v,
            _ => {}
        }
    }

    /// TEXMODE1 as the TE gets it: the shadow's, with texturing off while
    /// the GE has it off.
    fn texmode1(&self) -> u32 {
        let m = self.te[0];
        if self.tex_on != 0 { m } else { m & !te1::TEXMODE1_ENABLE }
    }

    /// A TE register from the driver (register lists, RSS_REG_SHADOW_IDX):
    /// into the context's shadow, and through to the TE while a batch has
    /// it loaded. False for other registers.
    pub fn te_write(&mut self, r: u32, v: u32, sink: &mut dyn Hq3Sink) -> bool {
        self.ensure_init();
        if r == te_reg::TXADDR {
            self.te_index = v;
        } else if let Some(i) = te1::CONTEXT_REGS.iter().position(|&x| x == r) {
            self.te[i] = v;
        } else if let Some(k) = te1::TABLE_REGS.iter().position(|&x| x == r) {
            self.te_tables[k][self.te_index as usize % te1::TABLE_LEN] = v;
            self.te_index = self.te_index.wrapping_add(1);
        } else {
            return false;
        }
        if self.raster_loaded != 0 {
            sink.rss_write(r, if r == te_reg::TEXMODE1 { self.texmode1() } else { v }, false);
        } else {
            self.te_dirty = 1;
        }
        true
    }

    /// Load the TE from the context's shadow (all of it when it changed or
    /// another context may have used the TE, else TEXMODE1 alone).
    fn load_te(&mut self, sink: &mut dyn Hq3Sink) {
        if self.te_dirty == 0 {
            sink.rss_write(te_reg::TEXMODE1, self.texmode1(), false);
            return;
        }
        self.te_dirty = 0;
        for (i, &r) in te1::CONTEXT_REGS.iter().enumerate() {
            sink.rss_write(r, if i == 0 { self.texmode1() } else { self.te[i] }, false);
        }
        for (k, &r) in te1::TABLE_REGS.iter().enumerate() {
            sink.rss_write(te_reg::TXADDR, 0, false);
            for &v in &self.te_tables[k] {
                sink.rss_write(r, v, false);
            }
        }
        sink.rss_write(te_reg::TXADDR, self.te_index, false);
    }

    /// A texture image (SEND_PIXELS routine 0x511A): the texels to the
    /// texture loader. The GE pixel state says which sub-image (0xDA8: s
    /// and t origin, signed, -1 for a border; width, height) and which TRAM
    /// page (0xDB1); the context's TL_MODE / TL_SPEC (in the TE once the
    /// batch is loaded) say level size, components and depth. Large images
    /// come as 64-texel-wide strips (a TRAM page's width). Rows are
    /// SEND_PIXELS' words-per-row apart when it sends a row per texel row,
    /// else packed; host components are 8 or 16 bits (the pixel format's
    /// scale, 0xD0A) or floats, and become the internal format's
    /// (luminance replicated, alpha picked out, missing alpha 1).
    fn tex_load(&mut self, words: &[u32], sink: &mut dyn Hq3Sink) {
        self.send_pixels_pending = 0;
        let [s0, t0, w, h, _] = self.tl_rect;
        let (w, h) = (w as usize, h as usize);
        if w == 0 || h == 0 || w > 4096 || h > 4096 {
            return;
        }
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
        let wpr = self.send_pixels[0] as usize * 4;
        let hdr_rows = (self.send_pixels[3] + 3 * self.send_pixels[4]) as usize;
        let (pitch, bpt) = if hdr_rows == h && wpr >= w { (wpr, wpr / w) } else { (0, bytes.len() / (w * h)) };
        let pitch = if pitch == 0 { w * bpt } else { pitch };
        let scale = f32::from_bits(self.tl_scale);
        // GL_UNSIGNED_SHORT_4_4_4_4_EXT: 16-bit elements with an 8-bit
        // scale. Widen each nibble to a byte (RGBA, R in the top nibble).
        // A restore of saved TRAM (no destination level: the loader
        // registers are the driver's own) is copied as it comes.
        let raw = self.tl_dest[0] == 0;
        let packed4 = !raw && self.tl_elem == 3 && bpt == 2 && scale > 1.0 / 1000.0 && scale < 0.5;
        let (bytes, pitch, bpt) = if packed4 {
            let mut out = Vec::with_capacity(w * h * 4);
            for row in 0..h {
                for col in 0..w {
                    let i = row * pitch + col * 2;
                    let v = bytes.get(i..i + 2).map_or(0, |b| u16::from_be_bytes([b[0], b[1]]));
                    out.extend((0..4).map(|k| ((v >> (12 - 4 * k)) & 0xF) as u8 * 0x11));
                }
            }
            (out, w * 4, 4)
        } else {
            (bytes, pitch, bpt)
        };
        let (comp, float) = if raw {
            (1, false)
        } else if scale >= 0.5 {
            (4, true)
        } else if scale > 0.0 && scale < 1.0 / 1000.0 {
            (if scale < 1.0 / 100_000.0 { 4 } else { 2 }, false)
        } else {
            (1, false)
        };
        let n_ext = (bpt / comp).clamp(1, 4);
        let tl_mode = self.te[te1::CONTEXT_REGS.iter().position(|&r| r == te_reg::TL_MODE).unwrap_or(0)];
        let nc = ((tl_mode >> 1) & 3) as usize + 1;
        let alpha_class = !raw && (self.te[0] >> 9) & 7 == 1;
        if sink.tracing_tex() {
            let spec = self.te[te1::CONTEXT_REGS.iter().position(|&r| r == te_reg::TL_SPEC).unwrap_or(0)];
            sink.trace(format!(
                "TEX load rect {:?} dest {:?} scale {scale} elem {} hdr {:x?} bytes {} pitch {pitch} bpt {bpt} n_ext {n_ext} \
                 tl_mode {tl_mode:#x} tl_spec {spec:#x} texmode1 {:#x} tl_mipmap {:#x}",
                self.tl_rect.map(|v| v as i32), self.tl_dest, self.tl_elem, &self.send_pixels[..6], bytes.len(), self.te[0],
                self.te[te1::CONTEXT_REGS.iter().position(|&r| r == te_reg::TL_MIPMAP).unwrap_or(0)]
            ));
        }
        self.begin_raster(sink);
        if !raw {
            sink.rss_write(te_reg::TL_MIPMAP, self.tl_dest[3], false);
            sink.rss_write(te_reg::TL_BORDER, self.tl_dest[4], false);
        }
        sink.rss_write(te_reg::TL_ADDR, 0, false);
        sink.rss_write(te_reg::TL_S_SIZE, w as u32, false);
        sink.rss_write(te_reg::TL_T_SIZE, h as u32, false);
        sink.rss_write(te_reg::TL_S_LEFT, s0, false);
        sink.rss_write(te_reg::TL_T_BOTTOM, t0, false);
        sink.rss_write(re::XFRSIZE, (h as u32) << 16 | w as u32, false);
        sink.rss_write(re::XFRCONTROL, TE_LOAD_START, false);
        for row in 0..h {
            let mut line = Vec::with_capacity(w * 4);
            for col in 0..w {
                let i = row * pitch + col * bpt;
                // Component k's top 8 bits (big-endian), or a float's.
                let src = |k: usize| {
                    let at = i + k * comp;
                    if float {
                        let b = bytes.get(at..at + 4).map_or(0, |b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
                        (f32::from_bits(b).clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        bytes.get(at).copied().unwrap_or(0)
                    }
                };
                let (lum, alpha) = (src(0), if n_ext == 2 || n_ext == 4 { src(n_ext - 1) } else { 0xFF });
                let px = match (n_ext, nc) {
                    _ if raw => [src(0), src(1), src(2), src(3)],
                    _ if alpha_class => [alpha, 0, 0, 0],
                    (n, c) if n == c => [src(0), src(1), src(2), src(3)],
                    (1 | 2, 4) => [lum, lum, lum, alpha],
                    (1 | 2, 3) => [lum, lum, lum, 0],
                    (_, 2) => [lum, alpha, 0, 0],
                    (3, 4) => [src(0), src(1), src(2), 0xFF],
                    _ => [src(0), src(1), src(2), src(3)],
                };
                line.extend_from_slice(&px);
            }
            sink.dma_write_line(row as u32, &line);
        }
        sink.rss_write(re::XFRCONTROL, 0, false);
    }
}

/// A 64-bit register pair (high word first), value rounded toward zero.
fn put64(sink: &mut dyn Hq3Sink, r: u32, v: f64) {
    let f = v as i64 as u64;
    sink.rss_write(r, (f >> 32) as u32, false);
    sink.rss_write(r + 1, f as u32, false);
}

/// A vertex's S/W, T/W, Q/W (texture coordinates over clip w, linear in
/// window space).
fn tex_q(v: &Wv) -> [f64; 3] {
    let wi = if v.h[3] != 0.0 { 1.0 / v.h[3] as f64 } else { 0.0 };
    [v.t[0] as f64 * wi, v.t[1] as f64 * wi, v.t[3] as f64 * wi]
}

/// A primitive's S/W, T/W, Q/W scaled by one factor so the largest Q/W is
/// 1. The TE divides by Q/W, so a common scale changes neither the
/// coordinates nor their derivatives; what it changes is precision in the
/// fixed-point planes. Far geometry has tiny 1/W (blast's nebula at W ~
/// 5e5: 1/W ~ 2e-6, its steps per pixel one or two LSBs at 2^32), and
/// each triangle of a clipped polygon then mapped the texture differently.
/// The IDE's planes (1/W = 1.0 at 2^20) suggest the GE normalises too.
fn normalised_q<const N: usize>(q: [[f64; 3]; N]) -> [[f64; 3]; N] {
    let m = q.iter().map(|v| v[2].abs()).fold(0.0, f64::max);
    if m == 0.0 || !m.is_finite() {
        return q;
    }
    q.map(|v| v.map(|c| c / m))
}

/// x | y << 16 for the RE4's integer xy registers.
fn xy(x: i32, y: i32) -> u32 {
    (x as u32 & 0xFFFF) << 16 | (y as u32 & 0xFFFF)
}

/// The point at `t` along window-space a-b (position, depth, colours).
fn lerp_window(a: &Wv, b: &Wv, t: f32) -> Wv {
    let l = |p: f32, q: f32| p + (q - p) * t;
    let mut v = *a;
    v.x = l(a.x, b.x);
    v.y = l(a.y, b.y);
    v.z = l(a.z, b.z);
    for k in 0..4 {
        v.c[k] = l(a.c[k], b.c[k]);
        v.cb[k] = l(a.cb[k], b.cb[k]);
        v.t[k] = l(a.t[k], b.t[k]);
    }
    v.f = l(a.f, b.f);
    v
}

/// A GL blend factor enum as the PP1 code (see rss.rs).
fn blend_code(e: u32) -> u32 {
    match e {
        0 => 0,
        1 => 1,
        0x300..=0x308 => e - 0x300 + 2,
        _ => 1,
    }
}

/// A token's float arguments: missing ones read as GL's defaults for
/// vertices and colours (0, 0, 0, 1).
fn args_f32(d: &[u32]) -> [f32; 4] {
    let mut v = [0.0, 0.0, 0.0, 1.0];
    for (o, w) in v.iter_mut().zip(d) {
        *o = f32::from_bits(*w);
    }
    v
}

/// `n` arguments of a *D token: doubles (high word first) when there are
/// two words each, floats otherwise.
fn args(d: &[u32], n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = if d.len() >= 2 * n {
        d.chunks(2).take(n).map(|p| f64::from_bits((p[0] as u64) << 32 | p[1] as u64) as f32).collect()
    } else {
        d.iter().take(n).map(|&w| f32::from_bits(w)).collect()
    };
    v.resize(n.max(v.len()), 0.0);
    v
}

/// The matrix stacks behind one interface (they differ only in depth).
trait MatrixStack {
    fn load(&mut self, m: Mat4);
    fn mult(&mut self, m: &Mat4);
    fn push(&mut self) -> bool;
    fn pop(&mut self) -> bool;
}

impl<const N: usize> MatrixStack for Stack<N> {
    fn load(&mut self, m: Mat4) {
        Stack::load(self, m)
    }
    fn mult(&mut self, m: &Mat4) {
        Stack::mult(self, m)
    }
    fn push(&mut self) -> bool {
        Stack::push(self)
    }
    fn pop(&mut self) -> bool {
        Stack::pop(self)
    }
}
