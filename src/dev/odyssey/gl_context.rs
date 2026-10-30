//! One OpenGL context's state. The board decodes its own command words
//! (geometry.rs, gl.rs) and feeds them to a context as GL operations. A
//! context holds the GL state a draw is made under (GlDrawState), the
//! matrices, the current vertex attributes, the clear values and the
//! primitive being assembled. Colours, matrices and vertex components are IEEE single
//! floats holding exactly the bits the client sent; arithmetic is double.

//!
//! Ported from include/hw/display/sgi/gl-context.h in jamesbraid/qemu-sgi-origami.

use crate::devlog::LogModule;
use super::bits::deposit32;
use super::memory::ClipList;
use crate::dev::gl::math::{invert, xform, IDENT};
use crate::dev::gl::vertex::{Geometry, NearW, Viewport, Wv};

/// A raster operation: pixel work on a head's planes that is not a GL
/// primitive, queued beside draws in the order the head received it.
pub(super) type RasterKind = u32;

/// Where colour is stored: a value the board's drawable gives and only
/// the board's store hook reads. Shared code copies it, never interprets it.
pub(super) type ColourTarget = u32;

pub(super) type GlDrawResult = u32;
pub(super) type GlOp = u32;

// Constants
pub(super) const GL_MAX_TARGETS: u32 = 4;

// The channels glColorMask lets a draw or clear write (4.2.2).
pub(super) const GL_COLOUR_MASK_RED: u32 = 0x1;
pub(super) const GL_COLOUR_MASK_ALPHA: u32 = 0x8;
pub(super) const GL_COLOUR_MASK_ALL: u32 = 0xf;

// The buffers a clear names, glClear's mask bits.
pub(super) const GL_DEPTH_BUFFER_BIT: u32 = 0x0100;
pub(super) const GL_STENCIL_BUFFER_BIT: u32 = 0x0400;
pub(super) const GL_COLOR_BUFFER_BIT: u32 = 0x4000;

/// Float slots of a captured vertex, `vtx::FLOATS` of them.
pub(super) mod vtx {
    pub const FLOATS: usize = 22;
    pub const POSITION: usize = 0;
    pub const COLOR: usize = 4;
    pub const NORMAL: usize = 8;
    pub const MATERIAL: usize = 11;
    pub const TEXCOORD: usize = 12;
    pub const EDGE: usize = 17;
    pub const BACK_COLOR: usize = 18;
}

// OpenGL 1.2 enumerants the state holds.
pub(super) const GL_POINTS: u32 = 0;
pub(super) const GL_LINES: u32 = 1;
pub(super) const GL_LINE_LOOP: u32 = 2;
pub(super) const GL_LINE_STRIP: u32 = 3;
pub(super) const GL_TRIANGLES: u32 = 4;
pub(super) const GL_TRIANGLE_STRIP: u32 = 5;
pub(super) const GL_TRIANGLE_FAN: u32 = 6;
pub(super) const GL_QUADS: u32 = 7;
pub(super) const GL_QUAD_STRIP: u32 = 8;
pub(super) const GL_POLYGON: u32 = 9;
pub(super) const GL_MODELVIEW: u32 = 0x1700;
pub(super) const GL_PROJECTION: u32 = 0x1701;
pub(super) const GL_TEXTURE: u32 = 0x1702;
pub(super) const GL_EYE_LINEAR: u32 = 0x2400;
pub(super) const GL_MODULATE: u32 = 0x2100;
pub(super) const GL_DECAL: u32 = 0x2101;
pub(super) const GL_NEAREST: u32 = 0x2600;
pub(super) const GL_NEAREST_MIPMAP_NEAREST: u32 = 0x2700;
pub(super) const GL_LINEAR_MIPMAP_NEAREST: u32 = 0x2701;
pub(super) const GL_NEAREST_MIPMAP_LINEAR: u32 = 0x2702;
pub(super) const GL_LINEAR_MIPMAP_LINEAR: u32 = 0x2703;
pub(super) const GL_LINEAR_SHARPEN_SGIS: u32 = 0x80ad;
pub(super) const GL_LINEAR_SHARPEN_ALPHA_SGIS: u32 = 0x80ae;
pub(super) const GL_LINEAR_SHARPEN_COLOR_SGIS: u32 = 0x80af;
pub(super) const GL_LINEAR_DETAIL_SGIS: u32 = 0x8097;
pub(super) const GL_LINEAR_DETAIL_ALPHA_SGIS: u32 = 0x8098;
pub(super) const GL_LINEAR_DETAIL_COLOR_SGIS: u32 = 0x8099;
pub(super) const GL_ADD: u32 = 0x0104;
pub(super) const GL_REPEAT: u32 = 0x2901;
pub(super) const GL_CLAMP: u32 = 0x2900;
pub(super) const GL_CLAMP_TO_EDGE: u32 = 0x812f;
pub(super) const GL_CLAMP_TO_BORDER: u32 = 0x812d;
pub(super) const GL_ALPHA4: u32 = 0x803b;
pub(super) const GL_ALPHA8: u32 = 0x803c;
pub(super) const GL_ALPHA12: u32 = 0x803d;
pub(super) const GL_ALPHA16: u32 = 0x803e;
pub(super) const GL_LUMINANCE4: u32 = 0x803f;
pub(super) const GL_LUMINANCE8: u32 = 0x8040;
pub(super) const GL_LUMINANCE12: u32 = 0x8041;
pub(super) const GL_LUMINANCE16: u32 = 0x8042;
pub(super) const GL_LUMINANCE4_ALPHA4: u32 = 0x8043;
pub(super) const GL_LUMINANCE6_ALPHA2: u32 = 0x8044;
pub(super) const GL_LUMINANCE8_ALPHA8: u32 = 0x8045;
pub(super) const GL_LUMINANCE12_ALPHA4: u32 = 0x8046;
pub(super) const GL_LUMINANCE12_ALPHA12: u32 = 0x8047;
pub(super) const GL_LUMINANCE16_ALPHA16: u32 = 0x8048;
pub(super) const GL_INTENSITY4: u32 = 0x804a;
pub(super) const GL_INTENSITY8: u32 = 0x804b;
pub(super) const GL_INTENSITY12: u32 = 0x804c;
pub(super) const GL_INTENSITY16: u32 = 0x804d;
pub(super) const GL_R3_G3_B2: u32 = 0x2a10;
pub(super) const GL_RGB4: u32 = 0x804f;
pub(super) const GL_RGB5: u32 = 0x8050;
pub(super) const GL_RGB8: u32 = 0x8051;
pub(super) const GL_RGB10: u32 = 0x8052;
pub(super) const GL_RGB12: u32 = 0x8053;
pub(super) const GL_RGB16: u32 = 0x8054;
pub(super) const GL_RGBA2: u32 = 0x8055;
pub(super) const GL_RGBA4: u32 = 0x8056;
pub(super) const GL_RGB5_A1: u32 = 0x8057;
pub(super) const GL_RGBA8: u32 = 0x8058;
pub(super) const GL_RGB10_A2: u32 = 0x8059;
pub(super) const GL_RGBA12: u32 = 0x805a;
pub(super) const GL_RGBA16: u32 = 0x805b;
pub(super) const GL_ALPHA: u32 = 0x1906;
pub(super) const GL_RGB: u32 = 0x1907;
pub(super) const GL_RGBA: u32 = 0x1908;
pub(super) const GL_DEPTH_COMPONENT: u32 = 0x1902;
pub(super) const GL_DEPTH_COMPONENT16_SGIX: u32 = 0x81a5;
pub(super) const GL_DEPTH_COMPONENT24_SGIX: u32 = 0x81a6;
pub(super) const GL_DEPTH_COMPONENT32_SGIX: u32 = 0x81a7;
pub(super) const GL_TEXTURE_LEQUAL_R_SGIX: u32 = 0x819c;
pub(super) const GL_TEXTURE_GEQUAL_R_SGIX: u32 = 0x819d;
pub(super) const GL_LUMINANCE: u32 = 0x1909;
pub(super) const GL_LUMINANCE_ALPHA: u32 = 0x190a;
pub(super) const GL_INTENSITY: u32 = 0x8049;
pub(super) const GL_FLAT: u32 = 0x1d00;
pub(super) const GL_SMOOTH: u32 = 0x1d01;
// The comparisons in GL_NEVER's order, as depth_func holds them.
pub(super) const GL_NEVER: u32 = 0;
pub(super) const GL_LESS: u32 = 1;
pub(super) const GL_EQUAL: u32 = 2;
pub(super) const GL_LEQUAL: u32 = 3;
pub(super) const GL_GREATER: u32 = 4;
pub(super) const GL_NOTEQUAL: u32 = 5;
pub(super) const GL_GEQUAL: u32 = 6;
pub(super) const GL_ALWAYS: u32 = 7;
pub(super) const GL_DRAW_FRONT: u32 = 0x1;
pub(super) const GL_DRAW_BACK: u32 = 0x2;
//
// The board's overlay planes, a layer of their own: set, the draw goes
// to the drawable's overlay target alone and the other bits name nothing.
//
pub(super) const GL_DRAW_OVERLAY: u32 = 0x40;
pub(super) const GL_NEVER_ENUM: u32 = 0x0200;
pub(super) const GL_CURRENT_COLOR: u32 = 0x0b00;
pub(super) const GL_POINT_SMOOTH: u32 = 0x0b10;
pub(super) const GL_LINE_SMOOTH: u32 = 0x0b20;
pub(super) const GL_LINE_STIPPLE: u32 = 0x0b24;
pub(super) const GL_LINE_STIPPLE_PATTERN: u32 = 0x0b25;
pub(super) const GL_LINE_STIPPLE_REPEAT: u32 = 0x0b26;
pub(super) const GL_POLYGON_SMOOTH: u32 = 0x0b41;
pub(super) const GL_POLYGON_STIPPLE: u32 = 0x0b42;
pub(super) const GL_CULL_FACE: u32 = 0x0b44;
pub(super) const GL_FOG: u32 = 0x0b60;
pub(super) const GL_DEPTH_TEST: u32 = 0x0b71;
pub(super) const GL_STENCIL_TEST: u32 = 0x0b90;
pub(super) const GL_DONT_CARE: u32 = 0x1100;
pub(super) const GL_HINTS: u32 = 5;
pub(super) const GL_SPRITE_AXIAL_SGIX: u32 = 0x814c;
pub(super) const GL_ALPHA_TEST: u32 = 0x0bc0;
pub(super) const GL_BLEND: u32 = 0x0be2;
pub(super) const GL_INDEX_LOGIC_OP: u32 = 0x0bf1;
pub(super) const GL_COLOR_LOGIC_OP: u32 = 0x0bf2;
pub(super) const GL_SCISSOR_TEST: u32 = 0x0c11;
pub(super) const GL_CLIP_PLANE0: u32 = 0x3000;
pub(super) const GL_POLYGON_OFFSET_POINT: u32 = 0x2a01;
pub(super) const GL_POLYGON_OFFSET_LINE: u32 = 0x2a02;
pub(super) const GL_POLYGON_OFFSET_FILL: u32 = 0x8037;
// The polygon_offset bits: the enable of each polygon mode's offset.
pub(super) const GL_OFFSET_FILL: u32 = 0x1;
pub(super) const GL_OFFSET_LINE: u32 = 0x2;
pub(super) const GL_OFFSET_POINT: u32 = 0x4;

//
// glBlendFunc's factors (4.1.6, and the imaging subset's constant
// factors) and glBlendEquation's equations. GL_ZERO is 0 and GL_ONE 1;
// GL_LOGIC_OP is EXT_blend_logic_op's equation.
//
pub(super) const GL_SRC_COLOR: u32 = 0x0300;
pub(super) const GL_ONE_MINUS_SRC_COLOR: u32 = 0x0301;
pub(super) const GL_SRC_ALPHA: u32 = 0x0302;
pub(super) const GL_ONE_MINUS_SRC_ALPHA: u32 = 0x0303;
pub(super) const GL_DST_ALPHA: u32 = 0x0304;
pub(super) const GL_ONE_MINUS_DST_ALPHA: u32 = 0x0305;
pub(super) const GL_DST_COLOR: u32 = 0x0306;
pub(super) const GL_ONE_MINUS_DST_COLOR: u32 = 0x0307;
pub(super) const GL_SRC_ALPHA_SATURATE: u32 = 0x0308;
pub(super) const GL_CONSTANT_COLOR: u32 = 0x8001;
pub(super) const GL_ONE_MINUS_CONSTANT_COLOR: u32 = 0x8002;
pub(super) const GL_CONSTANT_ALPHA: u32 = 0x8003;
pub(super) const GL_ONE_MINUS_CONSTANT_ALPHA: u32 = 0x8004;
pub(super) const GL_FUNC_ADD: u32 = 0x8006;
pub(super) const GL_MIN: u32 = 0x8007;
pub(super) const GL_MAX: u32 = 0x8008;
pub(super) const GL_FUNC_SUBTRACT: u32 = 0x800a;
pub(super) const GL_FUNC_REVERSE_SUBTRACT: u32 = 0x800b;
pub(super) const GL_LOGIC_OP: u32 = 0x0bf1;
// glLogicOp's operations run from GL_CLEAR to GL_SET (4.1.8).
pub(super) const GL_CLEAR: u32 = 0x1500;
pub(super) const GL_COPY: u32 = 0x1503;
pub(super) const GL_SET: u32 = 0x150f;

// Faces, and the light and material parameters (OpenGL 1.2.1, 2.13.2).
pub(super) const GL_FRONT: u32 = 0x0404;
pub(super) const GL_BACK: u32 = 0x0405;
pub(super) const GL_FRONT_AND_BACK: u32 = 0x0408;
// glPolygonMode's modes (3.5.4).
pub(super) const GL_POINT: u32 = 0x1b00;
pub(super) const GL_LINE: u32 = 0x1b01;
pub(super) const GL_FILL: u32 = 0x1b02;
//
// The rasterization enables a context keeps but does not draw with, as
// bits of raster_enables.
//
pub(super) const GL_RASTER_POINT_SMOOTH: u32 = 0x01;
pub(super) const GL_RASTER_LINE_SMOOTH: u32 = 0x02;
pub(super) const GL_RASTER_LINE_STIPPLE: u32 = 0x04;
pub(super) const GL_RASTER_POLYGON_SMOOTH: u32 = 0x08;
pub(super) const GL_RASTER_POLYGON_STIPPLE: u32 = 0x10;
// glFog's modes (3.10).
pub(super) const GL_EXP: u32 = 0x0800;
pub(super) const GL_EXP2: u32 = 0x0801;
pub(super) const GL_LINEAR: u32 = 0x2601;
// glStencilOp's operations (4.1.5); GL_ZERO is 0.
pub(super) const GL_INVERT: u32 = 0x150a;
pub(super) const GL_KEEP: u32 = 0x1e00;
pub(super) const GL_REPLACE: u32 = 0x1e01;
pub(super) const GL_INCR: u32 = 0x1e02;
pub(super) const GL_DECR: u32 = 0x1e03;
// glFrontFace's directions (2.13.1).
pub(super) const GL_CW: u32 = 0x0900;
pub(super) const GL_CCW: u32 = 0x0901;

// The client clip planes a context keeps, GL_CLIP_PLANE0 to 5 (2.11).
pub(super) const GL_CLIP_PLANES: u32 = 6;

//
// The most vertices a context holds between pieces of a primitive: a quad
// is drawn once its fourth arrives, so three wait.
//
pub(super) const GL_HELD_VERTICES: u32 = 3;

/// Half the extent, in pixels, of the guard band the window-matrix path clips
/// x and y to, about the drawable's origin: h0 and h1 are divided by it so
/// that the shared clip planes (|x| <= w) fall on it. It is iris's choice,
/// a limit of the integer edge arithmetic and not of the board, and is
/// separate from `WINDOW_REACH` (8192, the combined-matrix path's band about
/// the window) because a window position here is measured from the
/// drawable's origin and a window can sit anywhere in it. Both are far
/// inside the hard limit, the +-65536 check in `Board::project()`.
pub(super) const GL_WINDOW_GUARD: i32 = 16384;

/// Reports, once per process (a machine has one VPro board), that the w
/// plane clipped a primitive: preclip is off by default, the library stores
/// positions untested, and the board's behaviour at w <= 0 is unknown. iris
/// clips there so nothing reaches the integer edge functions. Call it from
/// a clip's slow path only.
pub(super) fn report_w_plane_clip() {
    static REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::dlog!(LogModule::Odyssey, "sgi-gl: a primitive reached w <= 0 and was clipped; the board's behaviour there is unknown");
    }
}

pub(super) const GL_OUT_OF_MEMORY: u32 = 0x0505;

pub(super) const RASTER_CLEAR: RasterKind = 1;

pub(super) const GL_DRAW_OK: GlDrawResult = 0;
pub(super) const GL_DRAW_OUT_OF_MEMORY: GlDrawResult = 1;
pub(super) const GL_DRAW_UNSUPPORTED: GlDrawResult = 2;
pub(super) const GL_DRAW_RESOURCE_ERROR: GlDrawResult = 3;
pub(super) const GL_DRAW_HOST_ERROR: GlDrawResult = 4;

/// The operations the board feeds the shared GL context (`Board::gl_feed32()`).
pub(super) mod op {
    use super::GlOp;

    pub const VIEWPORT: GlOp = 0;
    pub const CLEAR_COLOR: GlOp = 1;
    pub const CLEAR_DEPTH: GlOp = 2;
    pub const SHADE_MODEL: GlOp = 3;
    pub const MATRIX_MODE: GlOp = 4;
    pub const LOAD_MATRIX: GlOp = 5;
    pub const LOAD_IDENTITY: GlOp = 6;
    pub const MULT_MATRIX: GlOp = 7;
    pub const FRUSTUM: GlOp = 8;
    pub const ORTHO: GlOp = 9;
    pub const PUSH_MATRIX: GlOp = 10;
    pub const POP_MATRIX: GlOp = 11;
    pub const ROTATE: GlOp = 12;
    pub const SCALE: GlOp = 13;
    pub const TRANSLATE: GlOp = 14;
    pub const ENABLE: GlOp = 15;
    pub const DISABLE: GlOp = 16;
    pub const DEPTH_FUNC: GlOp = 17;
    pub const DEPTH_MASK: GlOp = 18;
    pub const DRAW_BUFFERS: GlOp = 19;
    pub const COLOR3F: GlOp = 20;
    pub const COLOR4F: GlOp = 21;
    pub const COLOR3UB: GlOp = 22;
    pub const COLOR4UB: GlOp = 23;
    pub const NORMAL3F: GlOp = 24;
    pub const EDGE_FLAG: GlOp = 25;
    pub const BEGIN: GlOp = 26;
    pub const END: GlOp = 27;
    pub const VERTEX2F: GlOp = 28;
    pub const VERTEX3F: GlOp = 29;
    pub const VERTEX4F: GlOp = 30;
    pub const VERTEX2I: GlOp = 31;
    pub const FLUSH: GlOp = 32;
    pub const CLEAR: GlOp = 33;
    pub const FINISH: GlOp = 34;
    pub const RASTER_POS2F: GlOp = 35;
    pub const RASTER_POS3F: GlOp = 36;
    pub const RASTER_POS4F: GlOp = 37;
    pub const INDEXF: GlOp = 38;
    pub const INDEX_MASK: GlOp = 39;
    pub const CLEAR_INDEX: GlOp = 40;
    pub const DEPTH_RANGE: GlOp = 41;

    //
    // glLightfv: the light's number, 0 for super::GL_LIGHT0, the parameter and
    // four floats, of which the parameter uses as many as it takes.
    //
    pub const LIGHT: GlOp = 42;

    // glMaterialfv: the face, the parameter and four floats, likewise.
    pub const MATERIAL: GlOp = 43;

    //
    // glLightModelfv: the parameter and four floats; a boolean parameter
    // is true for a nonzero first float.
    //
    pub const LIGHT_MODEL: GlOp = 44;
    pub const COLOR_MATERIAL: GlOp = 45;
    pub const CULL_FACE: GlOp = 46;
    pub const FRONT_FACE: GlOp = 47;
    pub const SCISSOR: GlOp = 48;
    pub const ALPHA_FUNC: GlOp = 49;

    // super::GL_NEVER to super::GL_ALWAYS, the reference as int32, the value mask
    pub const STENCIL_FUNC: GlOp = 50;
    pub const STENCIL_OP: GlOp = 51;
    pub const STENCIL_MASK: GlOp = 52;
    pub const CLEAR_STENCIL: GlOp = 53;
    pub const BLEND_FUNC: GlOp = 54;
    pub const BLEND_EQUATION: GlOp = 55;
    pub const BLEND_COLOR: GlOp = 56;
    pub const LOGIC_OP: GlOp = 57;
    pub const COLOR_MASK: GlOp = 58;

    //
    // glFogfv: the parameter and four floats, of which it uses as many as
    // it takes; super::GL_FOG_MODE's is the mode's enumerant as a float.
    //
    pub const FOG: GlOp = 59;
    pub const POLYGON_OFFSET: GlOp = 60;
    pub const POLYGON_OFFSET_EXT: GlOp = 61;
    pub const CLIP_PLANE: GlOp = 62;
    pub const POINT_SIZE: GlOp = 63;
    pub const LINE_WIDTH: GlOp = 64;
    pub const LINE_STIPPLE: GlOp = 65;
    pub const POLYGON_MODE: GlOp = 66;
    pub const TEXCOORD2F: GlOp = 67;
    pub const TEXCOORD3F: GlOp = 68;
    pub const TEXCOORD4F: GlOp = 69;

    // Parameter and four floats; mode's enumerant as a float.
    pub const TEX_ENV: GlOp = 70;

    //
    // IRIS GL swaptmesh as the OpenGL client passes it through an open
    // triangle strip: the next vertex makes its triangle with the strip's
    // oldest retained vertex instead of its middle one.
    //
    pub const SWAP_TMESH: GlOp = 71;
    pub const HINT: GlOp = 72;

    //
    // glTexGenfv: the coordinate (super::GL_S to super::GL_Q), the parameter and four
    // floats; super::GL_TEXTURE_GEN_MODE's mode is its enumerant as a float.
    //
    pub const TEX_GEN: GlOp = 73;
}

/// Only the fields a kind the rasterizer executes needs are here.
/// Coordinates are plane pixels, the rectangle inclusive.
#[derive(Clone, Copy, Default)]
pub(super) struct RasterOp {
    /// RasterKind.
    pub kind: u32,
    /// The colour targets written, as the drawable names them.
    pub targets: [ColourTarget; GL_MAX_TARGETS as usize],
    pub ntargets: u32,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    /// Only pixels inside these are written; at most the rectangle's. An
    /// empty list is how the board says its window has no visible pixel.
    pub clip: ClipList,
    /// CLEAR: GL_COLOR_BUFFER_BIT and GL_DEPTH_BUFFER_BIT.
    pub clear_buffers: u32,
    /// The GL clear colour; the rasterizer quantizes it to the plane.
    pub clear_rgba: [f32; 4],
    /// glColorMask's channels written: GL_COLOUR_MASK_RED and so on.
    pub clear_colour_mask: u32,
    /// In the depth plane's units; the plane keeps its low bits.
    pub clear_depth24: u32,
    /// GL_STENCIL_BUFFER_BIT: the clear value, written under the
    /// stencil write mask; the plane keeps its low bits.
    pub clear_stencil: u32,
    pub stencil_mask: u32,
}

/// Where a context draws, as the board's context switch, window clip and
/// buffer selection give it. Coordinates are plane pixels.
#[derive(Clone, Copy, Default)]
pub(super) struct GlDrawable {
    /// False while the board has no window for the context: nothing draws,
    /// and each draw, clear or bitmap is logged as dropped.
    pub bound: bool,
    /// The window's lower-left pixel, where window coordinates start; its
    /// top-left under a window matrix whose rows run down the display.
    pub origin: [i32; 2],
    /// 0/0 until the board supplies a validated drawable extent.
    pub width: u32,
    pub height: u32,
    /// The pixels the window owns. A bound window with no rectangle is one
    /// other windows wholly cover: nothing draws and nothing is logged. The
    /// list's empty flag plays no part here.
    pub clip: ClipList,
    /// The colour targets glDrawBuffer's front and back buffers name.
    pub front: ColourTarget,
    pub back: ColourTarget,
    /// The targets draw-buffer bits 2 and 3 name; zero where the board has
    /// none, and the bit is then ignored.
    pub extra: [ColourTarget; 2],
    /// The target GL_DRAW_OVERLAY names; zero where none.
    pub overlay: ColourTarget,
}

/// The GL state a draw is made under.
#[derive(Clone, Copy, Default)]
pub(super) struct GlDrawState {
    /// Column-major, as glLoadMatrixf lays a matrix out.
    pub modelview: [f32; 16],
    pub projection: [f32; 16],
    pub viewport: [i32; 4],
    /// GL_FLAT or GL_SMOOTH.
    pub shade_model: u32,
    pub depth_test: bool,
    /// The comparison minus GL_NEVER, 0 to 7.
    pub depth_func: u32,
    pub depth_mask: bool,
    /// glDepthRange's near and far, each clamped to [0, 1] (2.10.1).
    pub depth_range: [f32; 2],
    /// GL_DRAW_* bits; those the drawable has no target for are ignored.
    pub draw_buffers: u32,
    /// glColorMask's channels, GL_COLOUR_MASK_RED and so on (4.2.2).
    pub colour_mask: u32,
    /// Polygon culling (3.5.1): GL_CULL_FACE's enable,
    /// glCullFace's mode, and glFrontFace's GL_CW or GL_CCW. Any
    /// front_face but GL_CW is GL_CCW.
    pub cull_face: bool,
    pub cull_face_mode: u32,
    pub front_face: u32,
    /// The scissor test (4.1.2): its enable and glScissor's box,
    /// x, y, width and height in window coordinates.
    pub scissor_test: bool,
    pub scissor: [i32; 4],
    /// The alpha test (4.1.4): its enable, glAlphaFunc's
    /// comparison minus GL_NEVER, and its reference clamped to [0, 1].
    pub alpha_test: bool,
    pub alpha_func: u32,
    pub alpha_ref: f32,
    /// The stencil test (4.1.5): its enable, glStencilFunc's
    /// comparison minus GL_NEVER, reference (kept as sent) and value mask,
    /// glStencilMask's write mask, and glStencilOp's three operations as
    /// enumerants.
    pub stencil_test: bool,
    pub stencil_func: u32,
    pub stencil_ref: i32,
    pub stencil_value_mask: u32,
    pub stencil_write_mask: u32,
    pub stencil_op: [u32; 3],
    /// Blending (4.1.6): its enable, glBlendFunc's factors,
    /// glBlendEquation's equation, and glBlendColor clamped to [0, 1].
    pub blend: bool,
    pub blend_src: u32,
    pub blend_dst: u32,
    pub blend_equation: u32,
    pub blend_color: [f32; 4],
    /// The logical operation (4.1.8): the GL_COLOR_LOGIC_OP enable and
    /// glLogicOp's operation.
    pub color_logic_op: bool,
    pub logic_op_mode: u32,
    /// Fog (3.10): its enable, mode, density, start and end, the
    /// fog colour clamped to [0, 1]. The board applies its own curve
    /// (GlPlanes::fog_factor()) and reads only the enable and the colour.
    pub fog: bool,
    pub fog_mode: u32,
    pub fog_density: f32,
    pub fog_start: f32,
    pub fog_end: f32,
    pub fog_color: [f32; 4],
    /// Polygon offset (3.5.5): the enables as GL_OFFSET_FILL and so
    /// on, the factor, and the constant term, either glPolygonOffset's
    /// units or EXT_polygon_offset's bias; the command last given sets its
    /// own and zeroes the other.
    pub polygon_offset: u32,
    pub offset_factor: f32,
    pub offset_units: f32,
    pub offset_bias: f32,
    /// Client clip planes (2.11): bit i enables GL_CLIP_PLANEi; each plane
    /// is kept in eye coordinates.
    pub clip_planes: u32,
    pub clip_plane: [[f32; 4]; GL_CLIP_PLANES as usize],
    pub drawable: GlDrawable,
    /// A window matrix, for a board that takes one eye-to-window
    /// transform in place of a projection and a viewport; none of it is GL
    /// state (GlContext::set_window_matrix()). While window_matrix_given is set,
    /// an eye position e goes to h = W e by window_matrix W, column-major,
    /// and h replaces the projection, the divide and the viewport map:
    /// window x and y are h0 / h3 and h1 / h3, unrounded, in pixels from
    /// the drawable's origin, and window depth is
    /// window_depth_scale * h2 / h3. With window_rows_down set, window y
    /// counts rows down the display from the drawable's origin (then the
    /// window's top-left pixel). Clipping bounds x and y by GL_WINDOW_GUARD and
    /// w by the near-w plane; near and far are no clip, the raster drops a
    /// fragment by its window depth (0 and 1). This
    /// departs from OpenGL in raster position validity, bitmap rows, the tie
    /// rule and the scissor, which is measured from the top.
    pub window_matrix_given: bool,
    pub window_matrix: [f32; 16],
    pub window_depth_scale: f32,
    pub window_rows_down: bool,
    /// The texture environment belongs to the context, not its objects.
    pub texture_1d: bool,
    pub texture_2d: bool,
    pub texture_3d: bool,
    pub texture_env_mode: u32,
    pub texture_env_color: [f32; 4],
}

/// Host-owned deferred boundary. `vertices` was a malloc'd float array of
/// `capacity` floats with `count` vertices used; here a Vec sized by
/// `capacity`. Never shallow-copy a live context.
#[derive(Clone, Default)]
pub(super) struct GlPolygonAssembly {
    pub vertices: Vec<f32>,
    pub count: u32,
    pub capacity: u32,
    pub failed: bool,
    pub error: u32,
    /// Host activation marker; not part of the context's GL state.
    pub enabled: bool,
    pub loaded: bool,
}

#[derive(Clone, Default)]
pub(super) struct GlContext {
    /// The hardware identity the board keys this context by.
    pub hw_id: u32,
    pub cur: GlDrawState,
    /// The texture matrix (2.10.2), applied to each vertex before clipping.
    pub texture: [f32; 16],
    pub cur_color: [f32; 4],
    pub cur_normal: [f32; 3],
    pub cur_edge: u32,
    /// The current raster position (2.12): window x, y and z and clip w,
    /// whether it is valid, its eye distance and its colour.
    pub raster_pos: [f32; 4],
    pub raster_valid: bool,
    pub raster_color: [f32; 4],
    pub raster_index: f32,
    /// In the depth plane's units, as the board delivered it; a clear
    /// keeps the low bits the plane holds, so all ones is 1.0 at any depth.
    pub clear_depth: u32,
    /// Aliased sizes and retained polygon modes (3.3 to 3.5).
    pub point_size: f32,
    pub line_width: f32,
    pub line_stipple_repeat: u32,
    pub line_stipple_pattern: u32,
    pub polygon_mode: [u32; 2],
    pub raster_enables: u32,
    /// Canonical window rows: bit x of row y (OpenGL 1.2.1, 3.5.2).
    pub polygon_stipple: [u32; 32],
    pub polygon: GlPolygonAssembly,
    /// The primitive open since glBegin, GL_POINTS to GL_POLYGON.
    pub begin_mode: u32,
    pub in_begin: bool,
    /// Vertices waiting for the rest of their piece, nheld of them.
    pub held: [[f32; vtx::FLOATS]; GL_HELD_VERTICES as usize],
    pub nheld: u32,
    /// Set while the next triangle of an open strip is an odd one, whose
    /// first two vertices are sent swapped (2.6.1).
    pub strip_odd: bool,
    /// Set while held[2] keeps the open triangle strip's oldest retained
    /// vertex, which a swap makes the next triangle's first vertex.
    pub strip_prev: bool,
    /// Set from a swap until the next vertex of the strip.
    pub strip_swap: bool,
    pub cur_texcoord: [f32; 4],
    pub raster_texcoord: [f32; 4],
    /// Host policy and presence markers; not part of the context's GL state.
    pub standalone_stipple: bool,
    pub line_phase_loaded: bool,
    pub draw_error_loaded: bool,
    pub line_phase: u32,
    /// Advisory modes are retained even though rendering ignores them.
    pub hints: [u32; GL_HINTS as usize],
    pub sprite_mode: u32,
    pub sprite_axis: [f32; 3],
    pub tex_gen_mode: [u32; 4],
    pub tex_gen_object_plane: [[f32; 4]; 4],
    pub tex_gen_eye_plane: [[f32; 4]; 4],
    /// Draw-result failures already reported, one bit per DrawFailure.
    draw_failures_reported: u32,
}

/// The kinds of checked-draw failure that have no consumer, each reported
/// once per context.
#[derive(Clone, Copy)]
enum DrawFailure {
    Unsupported,
    Resource,
    Host,
    Invalid,
}

/// A decoded command: its entry point and its scalar arguments as 32-bit
/// words in the specification's order, a float argument as the IEEE single
/// bits the client sent.
#[derive(Clone, Copy)]
pub(super) struct GlCmd<'a> {
    pub op: u16,
    pub args: &'a [u32],
}

impl<'a> GlCmd<'a> {
    pub fn new(op: u16, args: &'a [u32]) -> GlCmd<'a> {
        GlCmd { op, args }
    }
}

/// A batch of vertices drawn as one primitive mode under one draw state.
/// The context lights each vertex before it is drawn, so no draw carries
/// materials: a vertex's material index is 0, the draw state's. Everything
/// it points at is the context's, borrowed for the call that carries it.
#[derive(Clone, Copy)]
pub(super) struct GlDraw<'a> {
    /// The context's hw_id.
    pub context: u32,
    /// GL_POINTS to GL_POLYGON.
    pub mode: u32,
    pub count: u32,
    /// count * vtx::FLOATS floats, texture-matrix-transformed strq
    /// included, ready for clipping.
    pub vertices: &'a [f32],
    /// True when the vertices carry back-face colour or index values for a
    /// back-facing polygon: two-sided lighting (2.13.1).
    pub back_colors: bool,
    pub state: &'a GlDrawState,
    /// Borrowed dimensions; zero in direct submissions means one.
    pub point_size: f32,
    pub line_width: f32,
    pub point_smooth: bool,
    pub line_smooth: bool,
    pub polygon_smooth: bool,
    /// 32 canonical rows, used while polygon_stipple_enabled is set.
    pub polygon_stipple: &'a [u32; 32],
    pub polygon_stipple_enabled: bool,
    /// Front and back modes; None keeps legacy FILL.
    pub polygon_modes: Option<[u32; 2]>,
    pub fill_by_triangle: bool,
    pub line_stipple_enabled: bool,
    pub line_stipple_repeat: u32,
    pub line_stipple_pattern: u32,
    /// The context's standalone stipple phase, which the draw advances in
    /// place; None counts a phase local to the draw.
    pub line_phase: Option<u32>,
    /// With a borrowed phase, reset a local candidate before counting.
    pub line_phase_reset: bool,
}

impl<'a> GlDraw<'a> {
    /// A draw of count vertices in mode, the rest of it off.
    pub fn new(mode: u32, count: u32, vertices: &'a [f32], state: &'a GlDrawState, polygon_stipple: &'a [u32; 32]) -> Self {
        GlDraw {
            context: 0,
            mode,
            count,
            vertices,
            back_colors: false,
            state,
            point_size: 0.0,
            line_width: 0.0,
            point_smooth: false,
            line_smooth: false,
            polygon_smooth: false,
            polygon_stipple,
            polygon_stipple_enabled: false,
            polygon_modes: None,
            fill_by_triangle: false,
            line_stipple_enabled: false,
            line_stipple_repeat: 0,
            line_stipple_pattern: 0,
            line_phase: None,
            line_phase_reset: false,
        }
    }
}

/// Where a context's draws go. Every draw is executed before the call
/// returns, so what it points at need only live that long.
pub(super) trait GlSink {
    /// Rasterize one draw; a line stipple's phase comes back in
    /// d.line_phase.
    fn draw(&mut self, d: &mut GlDraw) -> GlDrawResult;
}

// Functions.

// The last client clip plane and light enumerants.
pub(super) const GL_CLIP_PLANE_LAST: u32 = GL_CLIP_PLANE0 + GL_CLIP_PLANES - 1;

/// The word `i` of a command's arguments (a short list reads as 0 rather
/// than past the end).
pub(super) fn gl_arg(p: &[u32], i: usize) -> u32 {
    p.get(i).copied().unwrap_or(0)
}

pub(super) fn gl_float(bits: u32) -> f32 {
    f32::from_bits(bits)
}

pub(super) fn gl_bits(value: f32) -> u32 {
    value.to_bits()
}

impl GlContext {

    /// OpenGL 1.2.1 initial state (2.7, 2.10, 2.13.7, 4.1.5, 4.2.1-4.2.3).
    pub fn init(&mut self, hw_id: u32) {
        *self = GlContext::default();
        self.hw_id = hw_id;
        self.hints[..GL_HINTS as usize].fill(GL_DONT_CARE);
        self.cur_color = [1.0; 4];
        self.cur.shade_model = GL_SMOOTH;
        self.sprite_mode = GL_SPRITE_AXIAL_SGIX;
        self.sprite_axis[2] = 1.0;
        self.cur.modelview = IDENT;
        self.cur.projection = IDENT;
        self.texture = IDENT;
        self.cur.depth_func = GL_LESS;
        self.cur.alpha_func = GL_ALWAYS;
        self.cur.stencil_func = GL_ALWAYS;
        self.cur.stencil_value_mask = u32::MAX;
        self.cur.stencil_write_mask = u32::MAX;
        self.cur.stencil_op[..3].fill(GL_KEEP);
        self.cur.blend_src = 1;
        self.cur.blend_equation = GL_FUNC_ADD;
        self.cur.logic_op_mode = GL_COPY;
        self.cur.texture_env_mode = GL_MODULATE;
        self.cur_texcoord[3] = 1.0;
        self.raster_texcoord[3] = 1.0;
        self.tex_gen_mode[..4].fill(GL_EYE_LINEAR);
        self.tex_gen_object_plane[0][0] = 1.0;
        self.tex_gen_eye_plane[0][0] = 1.0;
        self.tex_gen_object_plane[1][1] = 1.0;
        self.tex_gen_eye_plane[1][1] = 1.0;
        self.cur.fog_mode = GL_EXP;
        self.cur.fog_density = 1.0;
        self.cur.fog_end = 1.0;
        self.point_size = 1.0;
        self.line_width = 1.0;
        self.line_stipple_repeat = 1;
        self.line_stipple_pattern = 0xffff;
        self.polygon_mode[0] = GL_FILL;
        self.polygon_mode[1] = GL_FILL;
        self.polygon_stipple = [0xffff_ffff; 32];
        self.cur.depth_mask = true;
        self.cur.depth_range[1] = 1.0;
        self.clear_depth = u32::MAX;
        self.cur.draw_buffers = GL_DRAW_FRONT;
        self.cur.colour_mask = GL_COLOUR_MASK_ALL;
        self.cur.cull_face_mode = GL_BACK;
        self.cur.front_face = GL_CCW;
        self.cur_normal[2] = 1.0;
        self.cur_edge = 1;
        self.raster_pos[3] = 1.0;
        self.raster_valid = true;
        self.raster_color[..4].fill(1.0);
        self.raster_index = 1.0;
    }

}

impl GlPolygonAssembly {
    pub fn polygon_release(&mut self) {
        self.vertices = Vec::new();
        self.count = 0;
        self.capacity = 0;
        self.failed = false;
        self.loaded = false;
    }
}

impl GlContext {
    /// Audited host consumers opt in; reset preserves that host policy.
    pub fn enable_polygon_modes(&mut self) -> bool {
        if self.in_begin {
            return false;
        }
        self.polygon.enabled = true;
        true
    }

    pub fn enable_line_stipple(&mut self) -> bool {
        if self.in_begin {
            return false;
        }
        self.standalone_stipple = true;
        true
    }

    pub fn cleanup(&mut self) {
        self.polygon.polygon_release();
        self.polygon.enabled = false;
        self.polygon.error = 0;
        self.standalone_stipple = false;
        self.line_phase_loaded = false;
        self.draw_error_loaded = false;
        self.line_phase = 0;
    }

    pub fn reset(&mut self, hw_id: u32) {
        let enabled = self.polygon.enabled;
        let standalone = self.standalone_stipple;

        self.cleanup();
        self.init(hw_id);
        self.polygon.enabled = enabled;
        self.standalone_stipple = standalone;
    }
}

impl GlPolygonAssembly {
    /// Count and bytes are representation bounds, not hardware size limits.
    /// `vertices` is kept `capacity * vtx::FLOATS` floats long.
    pub fn polygon_reserve(&mut self, count: usize) -> bool {
        let mut capacity: usize = std::cmp::max(self.capacity as usize, 8);

        if count > u32::MAX as usize || count > usize::MAX / vtx::FLOATS / 4 {
            return false;
        }
        if count <= self.capacity as usize {
            return true;
        }
        while capacity < count {
            if capacity > (u32::MAX / 2) as usize
                || capacity > usize::MAX / vtx::FLOATS / 4 / 2
            {
                capacity = count;
                break;
            }
            capacity *= 2;
        }
        let floats = capacity * vtx::FLOATS;
        if self.vertices.try_reserve_exact(floats.saturating_sub(self.vertices.len())).is_err() {
            return false;
        }
        self.vertices.resize(floats, 0.0);
        self.capacity = capacity as u32;
        true
    }
}

impl GlContext {

    /// Whether the context consumes a draw's result: with per-face polygon
    /// modes or the standalone line stipple active.
    pub fn draws_checked(&self) -> bool {
        self.polygon.enabled || self.standalone_stipple
    }

    /// Account for a draw's result, for a draw made while draws_checked() was
    /// `checked`.
    pub fn draw_result(&mut self, checked: bool, result: GlDrawResult) {
        if !checked {
            return;
        }
        if result == GL_DRAW_OUT_OF_MEMORY && self.polygon.error == 0 {
            self.polygon.error = GL_OUT_OF_MEMORY;
        }
        if result == GL_DRAW_OK {
            return;
        }
        match result {
            GL_DRAW_OUT_OF_MEMORY => {}
            GL_DRAW_UNSUPPORTED => {
                self.report_draw_failure(DrawFailure::Unsupported, "unsupported checked draw");
            }
            GL_DRAW_RESOURCE_ERROR => {
                self.report_draw_failure(DrawFailure::Resource, "checked draw resource failure");
            }
            GL_DRAW_HOST_ERROR => {
                self.report_draw_failure(DrawFailure::Host, "checked draw host failure");
            }
            _ => {
                self.report_draw_failure(DrawFailure::Invalid, &format!("invalid checked draw result {}", result));
            }
        }
    }

    /// Say that a draw failed with nobody to tell, the first time for each
    /// kind of failure.
    fn report_draw_failure(&mut self, kind: DrawFailure, what: &str) {
        let bit = 1u32 << kind as u32;

        if self.draw_failures_reported & bit == 0 {
            self.draw_failures_reported |= bit;
            eprintln!("sgi-gl: {what} has no result consumer");
        }
    }

    /// Canonical 32x32 pattern, copied on upload. Bit x in row y names window
    /// (x, y). Rejects use inside Begin/End.
    pub fn polygon_stipple(&mut self, rows: &[u32; 32]) -> bool {
        if self.in_begin {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon stipple inside glBegin: invalid operation");
            return false;
        }
        self.polygon_stipple = *rows;
        true
    }

    /// Point the context at a drawable. A bound half-known extent is refused
    /// without changing the current drawable. Unbinding clears both dimensions.
    pub fn set_drawable(&mut self, d: &GlDrawable) -> bool {
        if d.bound && (d.width != 0) != (d.height != 0) {
            return false;
        }
        self.cur.drawable = *d;
        if !d.bound {
            self.cur.drawable.width = 0;
            self.cur.drawable.height = 0;
        }
        true
    }

}

impl GlContext {
    /// The current matrix of the stack a matrix mode names, or None.
    pub fn slot_matrix(&mut self, slot: u32) -> Option<&mut [f32; 16]> {
        match slot {
            GL_MODELVIEW => Some(&mut self.cur.modelview),
            GL_PROJECTION => Some(&mut self.cur.projection),
            GL_TEXTURE => Some(&mut self.texture),
            _ => {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-gl: matrix mode {:#x} is not modelled; matrix command ignored",
                    slot
                );
                None
            }
        }
    }

    /// Load m, column-major, as the current matrix of the stack slot names:
    /// GL_MODELVIEW, GL_PROJECTION or GL_TEXTURE, whatever the
    /// context's matrix mode. False, and logged, for a slot the context does
    /// not keep.
    pub fn set_matrix(&mut self, slot: u32, m: &[f32; 16]) -> bool {
        match self.slot_matrix(slot) {
            None => false,
            Some(dst) => {
                *dst = *m;
                true
            }
        }
    }
}

/// The arguments each command takes.
pub(super) const GL_NARGS_LEN: usize = 74;
pub(super) static GL_NARGS: [u8; GL_NARGS_LEN] = {
    let mut t = [0u8; GL_NARGS_LEN];
    t[op::HINT as usize] = 2;
    t[op::VIEWPORT as usize] = 4;
    t[op::CLEAR_COLOR as usize] = 4;
    t[op::CLEAR_DEPTH as usize] = 1;
    t[op::SHADE_MODEL as usize] = 1;
    t[op::MATRIX_MODE as usize] = 1;
    t[op::LOAD_MATRIX as usize] = 16;
    t[op::LOAD_IDENTITY as usize] = 0;
    t[op::MULT_MATRIX as usize] = 16;
    t[op::FRUSTUM as usize] = 6;
    t[op::ORTHO as usize] = 6;
    t[op::PUSH_MATRIX as usize] = 0;
    t[op::POP_MATRIX as usize] = 0;
    t[op::ROTATE as usize] = 4;
    t[op::SCALE as usize] = 3;
    t[op::TRANSLATE as usize] = 3;
    t[op::ENABLE as usize] = 1;
    t[op::DISABLE as usize] = 1;
    t[op::DEPTH_FUNC as usize] = 1;
    t[op::DEPTH_MASK as usize] = 1;
    t[op::DRAW_BUFFERS as usize] = 1;
    t[op::COLOR3F as usize] = 3;
    t[op::COLOR4F as usize] = 4;
    t[op::COLOR3UB as usize] = 3;
    t[op::COLOR4UB as usize] = 4;
    t[op::NORMAL3F as usize] = 3;
    t[op::EDGE_FLAG as usize] = 1;
    t[op::BEGIN as usize] = 1;
    t[op::END as usize] = 0;
    t[op::VERTEX2F as usize] = 2;
    t[op::VERTEX3F as usize] = 3;
    t[op::VERTEX4F as usize] = 4;
    t[op::VERTEX2I as usize] = 2;
    t[op::FLUSH as usize] = 0;
    t[op::CLEAR as usize] = 1;
    t[op::FINISH as usize] = 0;
    t[op::RASTER_POS2F as usize] = 2;
    t[op::RASTER_POS3F as usize] = 3;
    t[op::RASTER_POS4F as usize] = 4;
    t[op::INDEXF as usize] = 1;
    t[op::INDEX_MASK as usize] = 1;
    t[op::CLEAR_INDEX as usize] = 1;
    t[op::DEPTH_RANGE as usize] = 2;
    t[op::LIGHT as usize] = 6;
    t[op::MATERIAL as usize] = 6;
    t[op::LIGHT_MODEL as usize] = 5;
    t[op::COLOR_MATERIAL as usize] = 2;
    t[op::CULL_FACE as usize] = 1;
    t[op::FRONT_FACE as usize] = 1;
    t[op::SCISSOR as usize] = 4;
    t[op::ALPHA_FUNC as usize] = 2;
    t[op::STENCIL_FUNC as usize] = 3;
    t[op::STENCIL_OP as usize] = 3;
    t[op::STENCIL_MASK as usize] = 1;
    t[op::CLEAR_STENCIL as usize] = 1;
    t[op::BLEND_FUNC as usize] = 2;
    t[op::BLEND_EQUATION as usize] = 1;
    t[op::BLEND_COLOR as usize] = 4;
    t[op::LOGIC_OP as usize] = 1;
    t[op::COLOR_MASK as usize] = 4;
    t[op::FOG as usize] = 5;
    t[op::POLYGON_OFFSET as usize] = 2;
    t[op::POLYGON_OFFSET_EXT as usize] = 2;
    t[op::CLIP_PLANE as usize] = 5;
    t[op::POINT_SIZE as usize] = 1;
    t[op::LINE_WIDTH as usize] = 1;
    t[op::LINE_STIPPLE as usize] = 2;
    t[op::POLYGON_MODE as usize] = 2;
    t[op::TEXCOORD2F as usize] = 2;
    t[op::TEXCOORD3F as usize] = 3;
    t[op::TEXCOORD4F as usize] = 4;
    t[op::TEX_ENV as usize] = 5;
    t[op::TEX_GEN as usize] = 6;
    t[op::SWAP_TMESH as usize] = 0;
    t
};

impl GlDrawState {
    /// The colour targets the draw state's draw buffers select, in bit order
    /// and each once, into out; returns how many.
    pub fn draw_targets(&self, out: &mut [ColourTarget; GL_MAX_TARGETS as usize]) -> u32 {
        let pick: [ColourTarget; GL_MAX_TARGETS as usize] = [
            self.drawable.front,
            self.drawable.back,
            self.drawable.extra[0],
            self.drawable.extra[1],
        ];
        let mut n: u32 = 0;

        if (self.draw_buffers & GL_DRAW_OVERLAY) != 0 && self.drawable.overlay != 0 {
            out[0] = self.drawable.overlay;
            return 1;
        }
        for (bit, &target) in pick.iter().enumerate().take(GL_MAX_TARGETS as usize) {
            if self.draw_buffers & (1u32 << bit) == 0 {
                continue;
            }
            if bit >= 2 && target == 0 {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-gl: draw buffer bit {} of {:#x} has no buffer; ignored",
                    bit,
                    self.draw_buffers
                );
                continue;
            }
            // A drawable whose buffers are one target, as a single-buffered one
            // may be, is stored to once: a read-modify-write stage in front of
            // the store must not apply twice.
            if !out[..n as usize].contains(&target) {
                out[n as usize] = target;
                n += 1;
            }
        }
        if self.draw_buffers & !0xfu32 != 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: draw buffers {:#x} beyond bit 3 ignored", self.draw_buffers);
        }
        n
    }
}

impl GlContext {

    /// Send one piece of the open primitive: count of v0 to v3.
    #[allow(clippy::too_many_arguments, reason = "the primitive and up to four vertices of it")]
    pub fn submit(&mut self, sink: &mut dyn GlSink, mode: u32, count: u32, v0: Option<&[f32; vtx::FLOATS]>, v1: Option<&[f32; vtx::FLOATS]>, v2: Option<&[f32; vtx::FLOATS]>, v3: Option<&[f32; vtx::FLOATS]>) {
        let stipple = if mode == GL_LINES { self.standalone_stipple } else { self.polygon.enabled };
        let v = [v0, v1, v2, v3];
        let mut verts = [0.0f32; 4 * vtx::FLOATS];

        for i in 0..count as usize {
            if let Some(src) = v[i] {
                verts[i * vtx::FLOATS..][..vtx::FLOATS].copy_from_slice(src);
            }
        }
        let checked = self.draws_checked();
        let mut d = GlDraw::new(mode, count, &verts, &self.cur, &self.polygon_stipple);

        d.context = self.hw_id;
        d.point_size = self.point_size;
        d.line_width = self.line_width;
        d.point_smooth = self.raster_enables & GL_RASTER_POINT_SMOOTH != 0;
        d.line_smooth = self.raster_enables & GL_RASTER_LINE_SMOOTH != 0;
        d.polygon_smooth = self.raster_enables & GL_RASTER_POLYGON_SMOOTH != 0;
        d.polygon_modes = (self.polygon.enabled).then_some(self.polygon_mode);
        d.line_stipple_enabled = stipple && (self.raster_enables & GL_RASTER_LINE_STIPPLE != 0);
        d.line_stipple_repeat = if stipple { self.line_stipple_repeat } else { 0 };
        d.line_stipple_pattern = if stipple { self.line_stipple_pattern } else { 0 };
        d.line_phase = (mode == GL_LINES && self.standalone_stipple).then_some(self.line_phase);
        d.line_phase_reset = self.begin_mode == GL_LINES;
        d.polygon_stipple_enabled = self.raster_enables & GL_RASTER_POLYGON_STIPPLE != 0;

        let result = sink.draw(&mut d);
        let phase = d.line_phase;
        if let Some(phase) = phase {
            self.line_phase = phase;
        }
        self.draw_result(checked, result);
    }

    /// A vertex's texture coordinates: the current ones through the
    /// texture matrix, before clipping (OpenGL 1.2.1, 2.10.2). The board
    /// generates none of its own.
    pub fn texture_coord(&self) -> [f32; 4] {
        xform(&self.texture, self.cur_texcoord)
    }

    /// Hold a vertex in slot n of the three a context keeps.
    pub fn hold(&mut self, n: u32, v: &[f32; vtx::FLOATS]) {
        self.held[n as usize] = *v;
    }
}

impl GlContext {

    /// Capture one vertex with the current attributes, without
    /// drawing. `bits` holds `components` words.
    pub fn capture_vertex(&self, mode: u32, bits: &[u32], components: u32) -> [f32; vtx::FLOATS] {
        let mut out = [0.0; vtx::FLOATS];
        out[vtx::POSITION + 3] = 1.0;
        for i in 0..std::cmp::min(components as usize, bits.len()) {
            out[vtx::POSITION + i] = gl_float(bits[i]);
        }
        out[vtx::COLOR..vtx::COLOR + 4].copy_from_slice(&self.cur_color);
        out[vtx::NORMAL..vtx::NORMAL + 3].copy_from_slice(&self.cur_normal);
        out[vtx::TEXCOORD..vtx::TEXCOORD + 4].copy_from_slice(&self.texture_coord());
        out[vtx::EDGE] = f32::from_bits(self.cur_edge);
        // Material and lighting belong to the instant this vertex arrived.
        out[vtx::BACK_COLOR..vtx::BACK_COLOR + 4].copy_from_slice(&self.cur_color);

        if self.polygon.enabled
            && (mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN || mode == GL_QUAD_STRIP)
        {
            out[vtx::EDGE] = f32::from_bits(1);
        }
        out
    }

    /// One vertex of the primitive open since glBegin (OpenGL 1.2.1, 2.6.1),
    /// carrying the current attributes. The context holds at most three
    /// earlier vertices, in held[] with nheld of them in use, and every piece
    /// is sent as its own draw as soon as its last vertex arrives, so a
    /// context switch between two vertices loses nothing. A line loop's
    /// closing segment is sent at glEnd. Vertices that complete no piece
    /// before glEnd draw nothing.
    pub fn vertex(&mut self, sink: &mut dyn GlSink, pos: &[f32; 4], components: u32) {
        let h0 = self.held[0];
        let h1 = self.held[1];
        let h2 = self.held[2];
        let n = self.nheld;

        if !self.in_begin {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: vertex outside glBegin ignored");
            return;
        }
        let bits = [pos[0].to_bits(), pos[1].to_bits(), pos[2].to_bits(), pos[3].to_bits()];
        let mode = self.begin_mode;
        let cur = self.capture_vertex(mode, &bits, components);
        if self.polygon.enabled && self.begin_mode == GL_POLYGON {
            let p = &mut self.polygon;

            if p.failed {
                return;
            }
            if p.count == u32::MAX || !p.polygon_reserve(p.count as usize + 1) {
                p.polygon_release();
                p.failed = true;
                if p.error == 0 {
                    p.error = GL_OUT_OF_MEMORY;
                }
                return;
            }
            let at = p.count as usize * vtx::FLOATS;
            p.vertices[at..at + vtx::FLOATS].copy_from_slice(&cur);
            p.count += 1;
            return;
        }

        match self.begin_mode {
            GL_POINTS => {
                self.submit(sink, GL_POINTS, 1, Some(&cur), None, None, None);
                return;
            }
            GL_LINES => {
                if n >= 1 {
                    self.submit(sink, GL_LINES, 2, Some(&h0), Some(&cur), None, None);
                    self.nheld = 0;
                    return;
                }
            }
            GL_LINE_STRIP => {
                if n >= 1 {
                    self.submit(sink, GL_LINES, 2, Some(&h0), Some(&cur), None, None);
                    self.hold(0, &cur);
                    return;
                }
            }
            GL_LINE_LOOP => {
                // Slot 0 keeps the first vertex for the closing segment.
                if n >= 1 {
                    let prev = self.held[n as usize - 1];
                    self.submit(sink, GL_LINES, 2, Some(&prev), Some(&cur), None, None);
                    self.hold(1, &cur);
                    self.nheld = 2;
                    return;
                }
            }
            GL_TRIANGLES => {
                if n >= 2 {
                    self.submit(sink, GL_TRIANGLES, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    self.nheld = 0;
                    return;
                }
            }
            GL_TRIANGLE_STRIP => {
                if n >= 2 {
                    if self.strip_swap && self.strip_prev {
                        // After a swap the vertex replaces the middle one of the
                        // three the strip keeps, so its triangle shares the
                        // strip's oldest and newest vertices, and keeps the
                        // facing of the triangle before it, as a fan's does. The
                        // oldest stays the oldest for the next swap.
                        if self.strip_odd {
                            self.submit(sink, GL_TRIANGLES, 3, Some(&h2), Some(&h1), Some(&cur), None);
                        } else {
                            self.submit(sink, GL_TRIANGLES, 3, Some(&h1), Some(&h2), Some(&cur), None);
                        }
                        self.strip_swap = false;
                        self.hold(0, &h1);
                        self.hold(1, &cur);
                        return;
                    }
                    self.strip_swap = false;
                    if self.strip_odd {
                        self.submit(sink, GL_TRIANGLES, 3, Some(&h1), Some(&h0), Some(&cur), None);
                    } else {
                        self.submit(sink, GL_TRIANGLES, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    }
                    self.strip_odd = !self.strip_odd;
                    self.hold(2, &h0);
                    self.strip_prev = true;
                    self.hold(0, &h1);
                    self.hold(1, &cur);
                    return;
                }
            }
            GL_TRIANGLE_FAN => {
                if n >= 2 {
                    self.submit(sink, GL_TRIANGLES, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    self.hold(1, &cur);
                    return;
                }
            }
            GL_POLYGON => {
                if n >= 2 {
                    self.submit(sink, GL_POLYGON, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    self.hold(1, &cur);
                    return;
                }
            }
            GL_QUADS => {
                if n >= 3 {
                    self.submit(sink, GL_QUADS, 4, Some(&h0), Some(&h1), Some(&h2), Some(&cur));
                    self.nheld = 0;
                    return;
                }
            }
            GL_QUAD_STRIP => {
                if n >= 3 {
                    self.submit(sink, GL_QUAD_STRIP, 4, Some(&h0), Some(&h1), Some(&h2), Some(&cur));
                    self.hold(0, &h2);
                    self.hold(1, &cur);
                    self.nheld = 2;
                    return;
                }
            }
            _ => unreachable!("sgi-gl: begin mode past GL_POLYGON"),
        }
        self.hold(n, &cur);
        self.nheld = n + 1;
    }

}

impl GlDrawState {
    /// An eye position's clip coordinates (2.10) under the draw state, in
    /// single precision through the shared transform (dev::gl::math): by the
    /// projection, or under a window matrix the window matrix's h with x and
    /// y scaled by 1 / GL_WINDOW_GUARD and z by 2 window_depth_scale, less w.
    pub fn eye_to_clip(&self, eye: [f32; 4]) -> [f32; 4] {
        if !self.window_matrix_given {
            return xform(&self.projection, eye);
        }
        let h = xform(&self.window_matrix, eye);
        let guard = GL_WINDOW_GUARD as f32;
        [h[0] / guard, h[1] / guard, 2.0 * self.window_depth_scale * h[2] - h[3], h[3]]
    }

    /// The viewport as the shared viewport mapping takes it, depth [0, 1]
    /// left normalized: the depth range maps it per fragment.
    pub fn gl_viewport(&self) -> Viewport {
        let [x, y, w, h] = self.viewport.map(|v| v as f32);
        let mut vp = Viewport::default();

        vp.set(x, y, w, h, 0.0, 1.0, 1.0);
        vp
    }

    /// Clip coordinates as GlDrawState::eye_to_clip() gives them to window x
    /// and y on the grid of 1 / subpixel of a pixel, and normalized depth,
    /// (z / w + 1) / 2, in single precision through the shared VPro geometry
    /// (dev::gl::vertex::NearW): the viewport mapping
    /// (dev::gl::vertex::Viewport::project_by()), or under a window matrix the
    /// window position the matrix gives. The divide is a nearest-rounded
    /// reciprocal and a multiply, and the position truncates onto the grid.
    /// None for a w of zero or less.
    pub fn clip_to_window(&self, clip: &[f32; 4], subpixel: f32) -> Option<[f32; 3]> {
        if !self.window_matrix_given {
            let v = self.gl_viewport().project_by::<NearW, 0, 0>(Wv::<0, 0> { h: *clip, ..Default::default() }, [0.0, 0.0], subpixel);
            return (v.ok != 0).then_some([v.x, v.y, v.z]);
        }
        let [x, y, z] = NearW::divide(clip)?;
        // The guard is a power of two, so scaling by it is exact.
        let guard = GL_WINDOW_GUARD as f32;
        Some([NearW::snap(x * guard, subpixel), NearW::snap(y * guard, subpixel), (z + 1.0) * 0.5])
    }
}

/// Whether op is one of glStencilOp's operations (4.1.5).
pub(super) fn gl_stencil_op_valid(op: u32) -> bool {
    matches!(op, 0 | GL_KEEP | GL_REPLACE | GL_INCR | GL_DECR | GL_INVERT)
}

/// Whether glBlendFunc accepts factor as a source factor (OpenGL 1.2.1,
/// table 4.1, with the imaging subset's constant factors) or, with dst,
/// as a destination one (table 4.2).
pub(super) fn gl_blend_factor_valid(factor: u32, dst: bool) -> bool {
    match factor {
        0
        | 1
        | GL_SRC_ALPHA
        | GL_ONE_MINUS_SRC_ALPHA
        | GL_DST_ALPHA
        | GL_ONE_MINUS_DST_ALPHA
        | GL_CONSTANT_COLOR
        | GL_ONE_MINUS_CONSTANT_COLOR
        | GL_CONSTANT_ALPHA
        | GL_ONE_MINUS_CONSTANT_ALPHA => true,
        GL_SRC_COLOR | GL_ONE_MINUS_SRC_COLOR => dst,
        GL_DST_COLOR | GL_ONE_MINUS_DST_COLOR | GL_SRC_ALPHA_SATURATE => !dst,
        _ => false,
    }
}

/// Whether glBlendEquation accepts equation.
pub(super) fn gl_blend_equation_valid(equation: u32) -> bool {
    matches!(
        equation,
        GL_FUNC_ADD
            | GL_MIN
            | GL_MAX
            | GL_FUNC_SUBTRACT
            | GL_FUNC_REVERSE_SUBTRACT
            | GL_LOGIC_OP
    )
}

impl GlContext {
    /// glClipPlane (2.11): the plane p in object coordinates becomes p M^-1 in
    /// eye coordinates, M the modelview current at the call. A plane number
    /// the context does not keep, or a modelview with no inverse, which the
    /// specification leaves undefined, changes nothing.
    pub fn clip_plane(&mut self, plane: u32, p: &[f32; 4]) {
        let inverse = invert(&self.cur.modelview).filter(|m| m.iter().all(|v| v.is_finite()));
        let Some(inv) = inverse.filter(|_| plane < GL_CLIP_PLANES) else {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: glClipPlane of plane {} ignored", plane);
            return;
        };
        // (p M^-1)_j = sum_i p_i M^-1[i][j], M^-1[i][j] at inv[j * 4 + i].
        self.cur.clip_plane[plane as usize] = std::array::from_fn(|j| (0..4).map(|i| p[i] * inv[j * 4 + i]).sum());
    }
}

/// The rasterization enables kept as raster_enables bits; 0 for any other
/// capability.
pub(super) fn gl_raster_enable(cap: u32) -> u32 {
    match cap {
        GL_POINT_SMOOTH => GL_RASTER_POINT_SMOOTH,
        GL_LINE_SMOOTH => GL_RASTER_LINE_SMOOTH,
        GL_LINE_STIPPLE => GL_RASTER_LINE_STIPPLE,
        GL_POLYGON_SMOOTH => GL_RASTER_POLYGON_SMOOTH,
        GL_POLYGON_STIPPLE => GL_RASTER_POLYGON_STIPPLE,
        _ => 0,
    }
}

impl GlContext {
    /// glPointSize, glLineWidth, glLineStipple and glPolygonMode (3.3 to 3.5):
    /// a size or width not above 0 or a face or mode the specification lacks
    /// is an error and changes nothing, and the repeat clamps to [1, 256].
    /// The state is kept and answered, and a value the rasterizer does not
    /// draw is logged.
    pub fn raster_state(&mut self, op: u16, a: &[u32]) {
        let f = gl_float(gl_arg(a, 0));

        match op as u32 {
            op::POINT_SIZE | op::LINE_WIDTH => {
                if self.in_begin {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: size setter inside glBegin: invalid operation");
                    return;
                }
                if !f.is_finite() || !(f > 0.0) {
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-gl: {}({}) ignored",
                        if op as u32 == op::POINT_SIZE { "glPointSize" } else { "glLineWidth" },
                        f
                    );
                    return;
                }
                if op as u32 == op::POINT_SIZE {
                    self.point_size = f;
                } else {
                    self.line_width = f;
                }
            }
            op::LINE_STIPPLE => {
                if self.in_begin {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: line stipple inside glBegin: invalid operation");
                    return;
                }
                let repeat = gl_arg(a, 0) as i32;
                self.line_stipple_repeat = repeat.clamp(1, 256) as u32;
                self.line_stipple_pattern = gl_arg(a, 1) & 0xffff;
            }
            _ => {
                if self.in_begin {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon mode inside glBegin: invalid operation");
                    return;
                }
                let a0 = gl_arg(a, 0);
                let a1 = gl_arg(a, 1);
                let faces: u32 = if a0 == GL_FRONT_AND_BACK {
                    3
                } else if a0 == GL_FRONT {
                    1
                } else if a0 == GL_BACK {
                    2
                } else {
                    0
                };
                if faces == 0 || a1.wrapping_sub(GL_POINT) > GL_FILL - GL_POINT {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glPolygonMode({:#x}, {:#x}) ignored", a0, a1);
                    return;
                }
                for (i, mode) in self.polygon_mode.iter_mut().enumerate() {
                    if faces & (1u32 << i) != 0 {
                        *mode = a1;
                    }
                }
                if a1 != GL_FILL && !self.polygon.enabled {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon mode {:#x} is kept; polygons are drawn filled", a1);
                }
            }
        }
    }

    pub fn capability(&mut self, cap: u32, on: bool) {
        let raster = gl_raster_enable(cap);
        let onv = on as u32;

        if raster != 0 {
            if (cap == GL_POLYGON_STIPPLE || cap == GL_LINE_STIPPLE) && self.in_begin {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: stipple enable inside glBegin: invalid operation");
                return;
            }
            self.raster_enables = if on { self.raster_enables | raster } else { self.raster_enables & !raster };
            if on && cap == GL_LINE_STIPPLE && !self.standalone_stipple {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: standalone line stipple is kept; standalone lines are drawn without it");
            }
            return;
        }
        match cap {
            GL_DEPTH_TEST => self.cur.depth_test = on,
            GL_CULL_FACE => self.cur.cull_face = on,
            GL_SCISSOR_TEST => self.cur.scissor_test = on,
            GL_ALPHA_TEST => self.cur.alpha_test = on,
            GL_STENCIL_TEST => self.cur.stencil_test = on,
            GL_BLEND => self.cur.blend = on,
            // Colour-index drawing stores the index unchanged; no logical
            // operation is modelled for it.
            GL_INDEX_LOGIC_OP => {}
            GL_COLOR_LOGIC_OP => self.cur.color_logic_op = on,
            GL_FOG => self.cur.fog = on,
            GL_POLYGON_OFFSET_FILL => self.cur.polygon_offset = deposit32(self.cur.polygon_offset, 0, 1, onv),
            GL_POLYGON_OFFSET_LINE => self.cur.polygon_offset = deposit32(self.cur.polygon_offset, 1, 1, onv),
            GL_POLYGON_OFFSET_POINT => self.cur.polygon_offset = deposit32(self.cur.polygon_offset, 2, 1, onv),
            GL_CLIP_PLANE0..=GL_CLIP_PLANE_LAST => {
                self.cur.clip_planes = deposit32(self.cur.clip_planes, cap - GL_CLIP_PLANE0, 1, onv);
            }
            _ => {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-gl: capability {:#x} is not modelled; {} ignored",
                    cap,
                    if on { "glEnable" } else { "glDisable" }
                );
            }
        }
    }

}

/// Four floats from command words `a[from..from + 4]`.
pub(super) fn gl_words4(a: &[u32], from: usize) -> [f32; 4] {
    [
        gl_float(gl_arg(a, from)),
        gl_float(gl_arg(a, from + 1)),
        gl_float(gl_arg(a, from + 2)),
        gl_float(gl_arg(a, from + 3)),
    ]
}

impl GlContext {
    /// Execute one command against the context, sending any work to sink.
    pub fn command(&mut self, sink: &mut dyn GlSink, cmd: &GlCmd) {
        let a = cmd.args;
        let mut pos = [0.0f32; 4];

        if cmd.op as usize >= GL_NARGS.len() || a.len() != GL_NARGS[cmd.op as usize] as usize {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: command {} with {} arguments ignored", cmd.op, a.len());
            return;
        }
        match cmd.op as u32 {
            op::TEXCOORD2F | op::TEXCOORD3F | op::TEXCOORD4F => {
                self.cur_texcoord[2] = 0.0;
                self.cur_texcoord[3] = 1.0;
                for i in 0..a.len() {
                    self.cur_texcoord[i] = gl_float(a[i]);
                }
            }
            op::SHADE_MODEL => self.cur.shade_model = a[0],
            op::ENABLE | op::DISABLE => {
                self.capability(a[0], cmd.op as u32 == op::ENABLE);
            }
            op::DEPTH_FUNC => {
                if a[0].wrapping_sub(GL_NEVER_ENUM) > GL_ALWAYS {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: depth function {:#x} ignored", a[0]);
                    return;
                }
                self.cur.depth_func = a[0] - GL_NEVER_ENUM;
            }
            op::DEPTH_MASK => self.cur.depth_mask = a[0] != 0,
            op::CULL_FACE => {
                if gl_faces(a[0]) == 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glCullFace({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.cull_face_mode = a[0];
            }
            op::FRONT_FACE => {
                if a[0] != GL_CW && a[0] != GL_CCW {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glFrontFace({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.front_face = a[0];
            }
            op::DRAW_BUFFERS => self.cur.draw_buffers = a[0],
            op::SCISSOR => {
                // A negative width or height is an error and changes nothing.
                if (a[2] as i32) < 0 || (a[3] as i32) < 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glScissor of {} by {} ignored", a[2] as i32, a[3] as i32);
                    return;
                }
                for i in 0..4 {
                    self.cur.scissor[i] = a[i] as i32;
                }
            }
            op::ALPHA_FUNC => {
                let r = gl_float(a[1]);

                if a[0].wrapping_sub(GL_NEVER_ENUM) > GL_ALWAYS {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: alpha function {:#x} ignored", a[0]);
                    return;
                }
                self.cur.alpha_func = a[0] - GL_NEVER_ENUM;
                // Clamped to [0, 1] (4.1.4); not a number clamps to 0.
                self.cur.alpha_ref = if r >= 1.0 {
                    1.0
                } else if r > 0.0 {
                    r
                } else {
                    0.0
                };
            }
            op::STENCIL_FUNC => {
                if a[0].wrapping_sub(GL_NEVER_ENUM) > GL_ALWAYS {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: stencil function {:#x} ignored", a[0]);
                    return;
                }
                self.cur.stencil_func = a[0] - GL_NEVER_ENUM;
                self.cur.stencil_ref = a[1] as i32;
                self.cur.stencil_value_mask = a[2];
            }
            op::STENCIL_OP => {
                for i in 0..3 {
                    if !gl_stencil_op_valid(a[i]) {
                        crate::dlog!(LogModule::Odyssey, "sgi-gl: glStencilOp({:#x}, {:#x}, {:#x}) ignored", a[0], a[1], a[2]);
                        return;
                    }
                }
                self.cur.stencil_op.copy_from_slice(&a[..3]);
            }
            op::STENCIL_MASK => self.cur.stencil_write_mask = a[0],
            op::BLEND_FUNC => {
                if !gl_blend_factor_valid(a[0], false) || !gl_blend_factor_valid(a[1], true) {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glBlendFunc({:#x}, {:#x}) ignored", a[0], a[1]);
                    return;
                }
                self.cur.blend_src = a[0];
                self.cur.blend_dst = a[1];
            }
            op::BLEND_EQUATION => {
                if !gl_blend_equation_valid(a[0]) {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glBlendEquation({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.blend_equation = a[0];
            }
            op::BLEND_COLOR => {
                for i in 0..4 {
                    let v = gl_float(a[i]);

                    self.cur.blend_color[i] = if v >= 1.0 {
                        1.0
                    } else if v > 0.0 {
                        v
                    } else {
                        0.0
                    };
                }
            }
            op::LOGIC_OP => {
                if a[0].wrapping_sub(GL_CLEAR) > GL_SET - GL_CLEAR {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glLogicOp({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.logic_op_mode = a[0];
            }
            op::POLYGON_OFFSET | op::POLYGON_OFFSET_EXT => {
                self.cur.offset_factor = gl_float(a[0]);
                self.cur.offset_units =
                    if cmd.op as u32 == op::POLYGON_OFFSET { gl_float(a[1]) } else { 0.0 };
                self.cur.offset_bias =
                    if cmd.op as u32 == op::POLYGON_OFFSET_EXT { gl_float(a[1]) } else { 0.0 };
            }
            op::CLIP_PLANE => {
                pos = gl_words4(a, 1);
                self.clip_plane(a[0], &pos);
            }
            op::POINT_SIZE | op::LINE_WIDTH | op::LINE_STIPPLE | op::POLYGON_MODE => {
                self.raster_state(cmd.op, a);
            }
            op::COLOR3F => {
                for i in 0..3 {
                    self.cur_color[i] = gl_float(a[i]);
                }
                self.cur_color[3] = 1.0;
            }
            op::COLOR4F => {
                for i in 0..4 {
                    self.cur_color[i] = gl_float(a[i]);
                }
            }
            op::COLOR3UB | op::COLOR4UB => {
                // A byte c is c / 255 (OpenGL 1.2.1, table 2.6); 3ub's alpha is 1.
                for i in 0..4usize {
                    self.cur_color[i] =
                        if i == a.len() { 1.0 } else { (a[i] & 0xff) as f32 / 255.0 };
                }
            }
            op::BEGIN => {
                if self.in_begin {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: nested glBegin ignored");
                    return;
                }
                if a[0] > GL_POLYGON {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glBegin of mode {} ignored", a[0]);
                    return;
                }
                self.line_phase = 0;
                self.in_begin = true;
                self.begin_mode = a[0];
                self.nheld = 0;
                self.strip_odd = false;
                self.strip_prev = false;
                self.strip_swap = false;
            }
            op::END => {
                if !self.in_begin && self.polygon.enabled {
                    return;
                }
                if self.polygon.enabled && self.begin_mode == GL_POLYGON {
                    if !self.polygon.failed && self.polygon.count >= 3 {
                        let checked = self.draws_checked();
                        let mut d = GlDraw::new(
                            GL_POLYGON,
                            self.polygon.count,
                            &self.polygon.vertices,
                            &self.cur,
                            &self.polygon_stipple,
                        );

                        d.context = self.hw_id;
                        d.point_size = self.point_size;
                        d.line_width = self.line_width;
                        d.point_smooth = self.raster_enables & GL_RASTER_POINT_SMOOTH != 0;
                        d.line_smooth = self.raster_enables & GL_RASTER_LINE_SMOOTH != 0;
                        d.polygon_smooth = self.raster_enables & GL_RASTER_POLYGON_SMOOTH != 0;
                        d.polygon_modes = Some(self.polygon_mode);
                        d.fill_by_triangle = true;
                        d.polygon_stipple_enabled = self.raster_enables & GL_RASTER_POLYGON_STIPPLE != 0;
                        d.line_stipple_enabled = self.raster_enables & GL_RASTER_LINE_STIPPLE != 0;
                        d.line_stipple_repeat = self.line_stipple_repeat;
                        d.line_stipple_pattern = self.line_stipple_pattern;

                        let result = sink.draw(&mut d);
                        self.draw_result(checked, result);
                    }
                    self.polygon.polygon_release();
                }
                // A line loop closes from its last vertex to its first, in the
                // first vertex's colour when flat (2.13.7).
                if self.in_begin && self.begin_mode == GL_LINE_LOOP && self.nheld == 2 {
                    let h1 = self.held[1];
                    let h0 = self.held[0];
                    self.submit(sink, GL_LINES, 2, Some(&h1), Some(&h0), None, None);
                }
                self.line_phase = 0;
                self.in_begin = false;
                self.begin_mode = 0;
                self.nheld = 0;
                self.strip_odd = false;
                self.strip_prev = false;
                self.strip_swap = false;
            }
            op::VERTEX2F | op::VERTEX3F | op::VERTEX4F => {
                for i in 0..a.len() {
                    pos[i] = gl_float(a[i]);
                }
                self.vertex(sink, &pos, a.len() as u32);
            }
            op::VERTEX2I => {
                // Integer components convert to float unscaled (2.7).
                pos[0] = (a[0] as i32) as f32;
                pos[1] = (a[1] as i32) as f32;
                self.vertex(sink, &pos, 2);
            }
            _ => {
                // Unreachable: every op the table lists is handled above.
            }
        }
    }

}

impl GlContext {
    /// glGet of GL_CURRENT_COLOR: the current colour as float bits (6.1.2).
    pub fn current_color(&self) -> [u32; 4] {
        self.cur_color.map(gl_bits)
    }
}

/// Whether a primitive of this mode can hold k vertices after count vertices
/// since glBegin. Only count mod 2,
/// 3 and 4 and k itself decide it, never count's absolute size, so a
/// count that is right only modulo 12 is enough: once a mode has all it
/// will ever hold, k alone says so.
pub(super) fn gl_resume_held_ok(mode: u32, k: u32, count: u32) -> bool {
    match mode {
        GL_POINTS => k == 0,
        GL_LINES => k == count % 2,
        GL_TRIANGLES => k == count % 3,
        GL_QUADS => k == count % 4,
        GL_QUAD_STRIP => {
            if k >= 2 {
                k == 2 + count % 2
            } else {
                k == count
            }
        }
        GL_LINE_STRIP => k <= 1 && (k == 1 || count == 0),
        // Strips of triangles, fans, loops, polygons: first and last.
        _ => k <= 2 && (k == 2 || k == count),
    }
}

impl GlContext {
    /// Resume a primitive a context switch interrupted inside glBegin/glEnd:
    /// begin mode and take k of the vertices the primitive held, oldest first,
    /// without drawing. count is how many vertices had been sent since glBegin.
    /// Refused, with a log, inside glBegin, for a (k, count, mode) that cannot
    /// occur, and for glPolygon while polygon mode assembly is on. `v` holds k
    /// vertices; None when k is 0.
    pub fn resume_primitive(&mut self, mode: u32, v: Option<&[[f32; vtx::FLOATS]]>, k: u32, count: u32) -> bool {
        if self.in_begin {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: resume inside glBegin ignored");
            return false;
        }
        if mode > GL_POLYGON
            || k > GL_HELD_VERTICES
            || (k != 0 && v.is_none_or(|s| s.len() < k as usize))
            || !gl_resume_held_ok(mode, k, count)
            || (self.polygon.enabled && mode == GL_POLYGON)
        {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: resume of mode {} with {} held of {} sent refused", mode, k, count);
            return false;
        }
        // line_phase is deliberately left alone, unlike glBegin.
        self.in_begin = true;
        self.begin_mode = mode;
        self.nheld = k;
        for i in 0..k {
            self.hold(i, &v.unwrap()[i as usize]);
        }
        // The strip's next triangle faces as its count's parity says.
        self.strip_odd = mode == GL_TRIANGLE_STRIP && k == 2 && count & 1 != 0;
        // The oldest retained vertex of a resumed strip is not sent.
        self.strip_prev = false;
        self.strip_swap = false;
        true
    }

    /// Close an interrupted primitive without drawing a line-loop edge.
    pub fn discard_primitive(&mut self) {
        if self.polygon.enabled && self.begin_mode == GL_POLYGON {
            self.polygon.polygon_release();
        }
        self.in_begin = false;
        self.begin_mode = 0;
        self.nheld = 0;
        self.strip_odd = false;
        self.strip_prev = false;
        self.strip_swap = false;
    }

}

/// The faces a glMaterial, glColorMaterial or glCullFace face names, as a
/// mask of material[] entries, or 0 for a value that names none.
pub(super) fn gl_faces(face: u32) -> u32 {
    match face {
        GL_FRONT => 1,
        GL_BACK => 2,
        GL_FRONT_AND_BACK => 3,
        _ => 0,
    }
}
