//! One OpenGL context's state, in the shape the SGI graphics heads share.
//! A head's family code frames and decodes its own command words and keeps
//! its own table of contexts. What one table entry holds is here: the GL
//! state a draw is made under (GlDrawState), the matrix stacks, the
//! current vertex attributes, the clear values and the primitive being
//! assembled. Colours, matrices and vertex components are IEEE single
//! floats holding exactly the bits the client sent; arithmetic is double.
//!
//! Ported from hw/display/sgi/gl-context.h in jamesbraid/qemu-sgi-origami.

use super::*;

use crate::dev::gl::math::{xform, IDENT};

/// A raster operation: pixel work on a head's planes that is not a GL
/// primitive, queued beside draws in the order the head received it.
pub type RasterKind = u32;

/// Where colour is stored: a value the family's drawable gives and only
/// the family's store hook reads. Shared code copies it, never interprets it.
pub type ColourTarget = u32;

pub type GlDrawResult = u32;
pub type GlOp = u32;

// Constants
pub const SGI_GL_MAX_TARGETS: u32 = 4;

// The channels glColorMask lets a draw or clear write (4.2.2).
pub const SGI_GL_COLOUR_MASK_RED: u32 = 0x1;
pub const SGI_GL_COLOUR_MASK_ALPHA: u32 = 0x8;
pub const SGI_GL_COLOUR_MASK_ALL: u32 = 0xf;

// The buffers a clear names, glClear's mask bits.
pub const SGI_GL_DEPTH_BUFFER_BIT: u32 = 0x0100;
pub const SGI_GL_STENCIL_BUFFER_BIT: u32 = 0x0400;
pub const SGI_GL_COLOR_BUFFER_BIT: u32 = 0x4000;

pub const SGI_GL_VERTEX_FLOATS: usize = 24;
pub const SGI_GL_V_POSITION: usize = 0;
pub const SGI_GL_V_COLOR: usize = 4;
pub const SGI_GL_V_NORMAL: usize = 8;
pub const SGI_GL_V_MATERIAL: usize = 11;
pub const SGI_GL_V_TEXCOORD: usize = 12;
pub const SGI_GL_V_EDGE: usize = 17;
pub const SGI_GL_V_INDEX: usize = 18;
pub const SGI_GL_V_BACK_INDEX: usize = 19;
pub const SGI_GL_V_BACK_COLOR: usize = 20;

// OpenGL 1.2 enumerants the state holds.
pub const SGI_GL_POINTS: u32 = 0;
pub const SGI_GL_LINES: u32 = 1;
pub const SGI_GL_LINE_LOOP: u32 = 2;
pub const SGI_GL_LINE_STRIP: u32 = 3;
pub const SGI_GL_TRIANGLES: u32 = 4;
pub const SGI_GL_TRIANGLE_STRIP: u32 = 5;
pub const SGI_GL_TRIANGLE_FAN: u32 = 6;
pub const SGI_GL_QUADS: u32 = 7;
pub const SGI_GL_QUAD_STRIP: u32 = 8;
pub const SGI_GL_POLYGON: u32 = 9;
pub const SGI_GL_MODELVIEW: u32 = 0x1700;
pub const SGI_GL_PROJECTION: u32 = 0x1701;
pub const SGI_GL_TEXTURE: u32 = 0x1702;
pub const SGI_GL_EYE_LINEAR: u32 = 0x2400;
pub const SGI_GL_MODULATE: u32 = 0x2100;
pub const SGI_GL_DECAL: u32 = 0x2101;
pub const SGI_GL_NEAREST: u32 = 0x2600;
pub const SGI_GL_NEAREST_MIPMAP_NEAREST: u32 = 0x2700;
pub const SGI_GL_LINEAR_MIPMAP_NEAREST: u32 = 0x2701;
pub const SGI_GL_NEAREST_MIPMAP_LINEAR: u32 = 0x2702;
pub const SGI_GL_LINEAR_MIPMAP_LINEAR: u32 = 0x2703;
pub const SGI_GL_LINEAR_SHARPEN_SGIS: u32 = 0x80ad;
pub const SGI_GL_LINEAR_SHARPEN_ALPHA_SGIS: u32 = 0x80ae;
pub const SGI_GL_LINEAR_SHARPEN_COLOR_SGIS: u32 = 0x80af;
pub const SGI_GL_LINEAR_DETAIL_SGIS: u32 = 0x8097;
pub const SGI_GL_LINEAR_DETAIL_ALPHA_SGIS: u32 = 0x8098;
pub const SGI_GL_LINEAR_DETAIL_COLOR_SGIS: u32 = 0x8099;
pub const SGI_GL_ADD: u32 = 0x0104;
pub const SGI_GL_REPEAT: u32 = 0x2901;
pub const SGI_GL_CLAMP: u32 = 0x2900;
pub const SGI_GL_CLAMP_TO_EDGE: u32 = 0x812f;
pub const SGI_GL_CLAMP_TO_BORDER: u32 = 0x812d;
pub const SGI_GL_ALPHA4: u32 = 0x803b;
pub const SGI_GL_ALPHA8: u32 = 0x803c;
pub const SGI_GL_ALPHA12: u32 = 0x803d;
pub const SGI_GL_ALPHA16: u32 = 0x803e;
pub const SGI_GL_LUMINANCE4: u32 = 0x803f;
pub const SGI_GL_LUMINANCE8: u32 = 0x8040;
pub const SGI_GL_LUMINANCE12: u32 = 0x8041;
pub const SGI_GL_LUMINANCE16: u32 = 0x8042;
pub const SGI_GL_LUMINANCE4_ALPHA4: u32 = 0x8043;
pub const SGI_GL_LUMINANCE6_ALPHA2: u32 = 0x8044;
pub const SGI_GL_LUMINANCE8_ALPHA8: u32 = 0x8045;
pub const SGI_GL_LUMINANCE12_ALPHA4: u32 = 0x8046;
pub const SGI_GL_LUMINANCE12_ALPHA12: u32 = 0x8047;
pub const SGI_GL_LUMINANCE16_ALPHA16: u32 = 0x8048;
pub const SGI_GL_INTENSITY4: u32 = 0x804a;
pub const SGI_GL_INTENSITY8: u32 = 0x804b;
pub const SGI_GL_INTENSITY12: u32 = 0x804c;
pub const SGI_GL_INTENSITY16: u32 = 0x804d;
pub const SGI_GL_R3_G3_B2: u32 = 0x2a10;
pub const SGI_GL_RGB4: u32 = 0x804f;
pub const SGI_GL_RGB5: u32 = 0x8050;
pub const SGI_GL_RGB8: u32 = 0x8051;
pub const SGI_GL_RGB10: u32 = 0x8052;
pub const SGI_GL_RGB12: u32 = 0x8053;
pub const SGI_GL_RGB16: u32 = 0x8054;
pub const SGI_GL_RGBA2: u32 = 0x8055;
pub const SGI_GL_RGBA4: u32 = 0x8056;
pub const SGI_GL_RGB5_A1: u32 = 0x8057;
pub const SGI_GL_RGBA8: u32 = 0x8058;
pub const SGI_GL_RGB10_A2: u32 = 0x8059;
pub const SGI_GL_RGBA12: u32 = 0x805a;
pub const SGI_GL_RGBA16: u32 = 0x805b;
pub const SGI_GL_ALPHA: u32 = 0x1906;
pub const SGI_GL_RGB: u32 = 0x1907;
pub const SGI_GL_RGBA: u32 = 0x1908;
pub const SGI_GL_DEPTH_COMPONENT: u32 = 0x1902;
pub const SGI_GL_DEPTH_COMPONENT16_SGIX: u32 = 0x81a5;
pub const SGI_GL_DEPTH_COMPONENT24_SGIX: u32 = 0x81a6;
pub const SGI_GL_DEPTH_COMPONENT32_SGIX: u32 = 0x81a7;
pub const SGI_GL_TEXTURE_LEQUAL_R_SGIX: u32 = 0x819c;
pub const SGI_GL_TEXTURE_GEQUAL_R_SGIX: u32 = 0x819d;
pub const SGI_GL_LUMINANCE: u32 = 0x1909;
pub const SGI_GL_LUMINANCE_ALPHA: u32 = 0x190a;
pub const SGI_GL_INTENSITY: u32 = 0x8049;
pub const SGI_GL_FLAT: u32 = 0x1d00;
pub const SGI_GL_SMOOTH: u32 = 0x1d01;
// The comparisons in GL_NEVER's order, as depth_func holds them.
pub const SGI_GL_NEVER: u32 = 0;
pub const SGI_GL_LESS: u32 = 1;
pub const SGI_GL_EQUAL: u32 = 2;
pub const SGI_GL_LEQUAL: u32 = 3;
pub const SGI_GL_GREATER: u32 = 4;
pub const SGI_GL_NOTEQUAL: u32 = 5;
pub const SGI_GL_GEQUAL: u32 = 6;
pub const SGI_GL_ALWAYS: u32 = 7;
pub const SGI_GL_DRAW_FRONT: u32 = 0x1;
pub const SGI_GL_DRAW_BACK: u32 = 0x2;
//
// The family's overlay planes, a layer of their own: set, the draw goes
// to the drawable's overlay target alone and the other bits name nothing.
//
pub const SGI_GL_DRAW_OVERLAY: u32 = 0x40;
pub const SGI_GL_NEVER_ENUM: u32 = 0x0200;
pub const SGI_GL_CURRENT_COLOR: u32 = 0x0b00;
pub const SGI_GL_POINT_SMOOTH: u32 = 0x0b10;
pub const SGI_GL_LINE_SMOOTH: u32 = 0x0b20;
pub const SGI_GL_LINE_STIPPLE: u32 = 0x0b24;
pub const SGI_GL_LINE_STIPPLE_PATTERN: u32 = 0x0b25;
pub const SGI_GL_LINE_STIPPLE_REPEAT: u32 = 0x0b26;
pub const SGI_GL_POLYGON_SMOOTH: u32 = 0x0b41;
pub const SGI_GL_POLYGON_STIPPLE: u32 = 0x0b42;
pub const SGI_GL_CULL_FACE: u32 = 0x0b44;
pub const SGI_GL_FOG: u32 = 0x0b60;
pub const SGI_GL_DEPTH_TEST: u32 = 0x0b71;
pub const SGI_GL_STENCIL_TEST: u32 = 0x0b90;
pub const SGI_GL_DONT_CARE: u32 = 0x1100;
pub const SGI_GL_HINTS: u32 = 5;
pub const SGI_GL_SPRITE_AXIAL_SGIX: u32 = 0x814c;
pub const SGI_GL_ALPHA_TEST: u32 = 0x0bc0;
pub const SGI_GL_BLEND: u32 = 0x0be2;
pub const SGI_GL_INDEX_LOGIC_OP: u32 = 0x0bf1;
pub const SGI_GL_COLOR_LOGIC_OP: u32 = 0x0bf2;
pub const SGI_GL_SCISSOR_TEST: u32 = 0x0c11;
pub const SGI_GL_CLIP_PLANE0: u32 = 0x3000;
pub const SGI_GL_POLYGON_OFFSET_POINT: u32 = 0x2a01;
pub const SGI_GL_POLYGON_OFFSET_LINE: u32 = 0x2a02;
pub const SGI_GL_POLYGON_OFFSET_FILL: u32 = 0x8037;
// The polygon_offset bits: the enable of each polygon mode's offset.
pub const SGI_GL_OFFSET_FILL: u32 = 0x1;
pub const SGI_GL_OFFSET_LINE: u32 = 0x2;
pub const SGI_GL_OFFSET_POINT: u32 = 0x4;

//
// glBlendFunc's factors (4.1.6, and the imaging subset's constant
// factors) and glBlendEquation's equations. GL_ZERO is 0 and GL_ONE 1;
// GL_LOGIC_OP is EXT_blend_logic_op's equation.
//
pub const SGI_GL_SRC_COLOR: u32 = 0x0300;
pub const SGI_GL_ONE_MINUS_SRC_COLOR: u32 = 0x0301;
pub const SGI_GL_SRC_ALPHA: u32 = 0x0302;
pub const SGI_GL_ONE_MINUS_SRC_ALPHA: u32 = 0x0303;
pub const SGI_GL_DST_ALPHA: u32 = 0x0304;
pub const SGI_GL_ONE_MINUS_DST_ALPHA: u32 = 0x0305;
pub const SGI_GL_DST_COLOR: u32 = 0x0306;
pub const SGI_GL_ONE_MINUS_DST_COLOR: u32 = 0x0307;
pub const SGI_GL_SRC_ALPHA_SATURATE: u32 = 0x0308;
pub const SGI_GL_CONSTANT_COLOR: u32 = 0x8001;
pub const SGI_GL_ONE_MINUS_CONSTANT_COLOR: u32 = 0x8002;
pub const SGI_GL_CONSTANT_ALPHA: u32 = 0x8003;
pub const SGI_GL_ONE_MINUS_CONSTANT_ALPHA: u32 = 0x8004;
pub const SGI_GL_FUNC_ADD: u32 = 0x8006;
pub const SGI_GL_MIN: u32 = 0x8007;
pub const SGI_GL_MAX: u32 = 0x8008;
pub const SGI_GL_FUNC_SUBTRACT: u32 = 0x800a;
pub const SGI_GL_FUNC_REVERSE_SUBTRACT: u32 = 0x800b;
pub const SGI_GL_LOGIC_OP: u32 = 0x0bf1;
// glLogicOp's operations run from GL_CLEAR to GL_SET (4.1.8).
pub const SGI_GL_CLEAR: u32 = 0x1500;
pub const SGI_GL_COPY: u32 = 0x1503;
pub const SGI_GL_SET: u32 = 0x150f;

// Faces, and the light and material parameters (OpenGL 1.2.1, 2.13.2).
pub const SGI_GL_FRONT: u32 = 0x0404;
pub const SGI_GL_BACK: u32 = 0x0405;
pub const SGI_GL_FRONT_AND_BACK: u32 = 0x0408;
// glPolygonMode's modes (3.5.4).
pub const SGI_GL_POINT: u32 = 0x1b00;
pub const SGI_GL_LINE: u32 = 0x1b01;
pub const SGI_GL_FILL: u32 = 0x1b02;
//
// The rasterization enables a context keeps but does not draw with, as
// bits of raster_enables.
//
pub const SGI_GL_RASTER_POINT_SMOOTH: u32 = 0x01;
pub const SGI_GL_RASTER_LINE_SMOOTH: u32 = 0x02;
pub const SGI_GL_RASTER_LINE_STIPPLE: u32 = 0x04;
pub const SGI_GL_RASTER_POLYGON_SMOOTH: u32 = 0x08;
pub const SGI_GL_RASTER_POLYGON_STIPPLE: u32 = 0x10;
// glFog's modes (3.10).
pub const SGI_GL_EXP: u32 = 0x0800;
pub const SGI_GL_EXP2: u32 = 0x0801;
pub const SGI_GL_LINEAR: u32 = 0x2601;
// glStencilOp's operations (4.1.5); GL_ZERO is 0.
pub const SGI_GL_INVERT: u32 = 0x150a;
pub const SGI_GL_KEEP: u32 = 0x1e00;
pub const SGI_GL_REPLACE: u32 = 0x1e01;
pub const SGI_GL_INCR: u32 = 0x1e02;
pub const SGI_GL_DECR: u32 = 0x1e03;
// glFrontFace's directions (2.13.1).
pub const SGI_GL_CW: u32 = 0x0900;
pub const SGI_GL_CCW: u32 = 0x0901;

// The client clip planes a context keeps, GL_CLIP_PLANE0 to 5 (2.11).
pub const SGI_GL_CLIP_PLANES: u32 = 6;

//
// The most vertices a context holds between pieces of a primitive: a quad
// is drawn once its fourth arrives, so three wait.
//
pub const SGI_GL_HELD_VERTICES: u32 = 3;

pub const SGI_GL_WINDOW_GUARD: i32 = 16384;

pub const SGI_GL_OUT_OF_MEMORY: u32 = 0x0505;

pub const SGI_RASTER_CLEAR: RasterKind = 1;

pub const SGI_GL_DRAW_OK: GlDrawResult = 0;
pub const SGI_GL_DRAW_OUT_OF_MEMORY: GlDrawResult = 1;
pub const SGI_GL_DRAW_UNSUPPORTED: GlDrawResult = 2;
pub const SGI_GL_DRAW_RESOURCE_ERROR: GlDrawResult = 3;
pub const SGI_GL_DRAW_HOST_ERROR: GlDrawResult = 4;

pub const SGI_GL_OP_VIEWPORT: GlOp = 0;
pub const SGI_GL_OP_CLEAR_COLOR: GlOp = 1;
pub const SGI_GL_OP_CLEAR_DEPTH: GlOp = 2;
pub const SGI_GL_OP_SHADE_MODEL: GlOp = 3;
pub const SGI_GL_OP_MATRIX_MODE: GlOp = 4;
pub const SGI_GL_OP_LOAD_MATRIX: GlOp = 5;
pub const SGI_GL_OP_LOAD_IDENTITY: GlOp = 6;
pub const SGI_GL_OP_MULT_MATRIX: GlOp = 7;
pub const SGI_GL_OP_FRUSTUM: GlOp = 8;
pub const SGI_GL_OP_ORTHO: GlOp = 9;
pub const SGI_GL_OP_PUSH_MATRIX: GlOp = 10;
pub const SGI_GL_OP_POP_MATRIX: GlOp = 11;
pub const SGI_GL_OP_ROTATE: GlOp = 12;
pub const SGI_GL_OP_SCALE: GlOp = 13;
pub const SGI_GL_OP_TRANSLATE: GlOp = 14;
pub const SGI_GL_OP_ENABLE: GlOp = 15;
pub const SGI_GL_OP_DISABLE: GlOp = 16;
pub const SGI_GL_OP_DEPTH_FUNC: GlOp = 17;
pub const SGI_GL_OP_DEPTH_MASK: GlOp = 18;
pub const SGI_GL_OP_DRAW_BUFFERS: GlOp = 19;
pub const SGI_GL_OP_COLOR3F: GlOp = 20;
pub const SGI_GL_OP_COLOR4F: GlOp = 21;
pub const SGI_GL_OP_COLOR3UB: GlOp = 22;
pub const SGI_GL_OP_COLOR4UB: GlOp = 23;
pub const SGI_GL_OP_NORMAL3F: GlOp = 24;
pub const SGI_GL_OP_EDGE_FLAG: GlOp = 25;
pub const SGI_GL_OP_BEGIN: GlOp = 26;
pub const SGI_GL_OP_END: GlOp = 27;
pub const SGI_GL_OP_VERTEX2F: GlOp = 28;
pub const SGI_GL_OP_VERTEX3F: GlOp = 29;
pub const SGI_GL_OP_VERTEX4F: GlOp = 30;
pub const SGI_GL_OP_VERTEX2I: GlOp = 31;
pub const SGI_GL_OP_FLUSH: GlOp = 32;
pub const SGI_GL_OP_CLEAR: GlOp = 33;
pub const SGI_GL_OP_FINISH: GlOp = 34;
pub const SGI_GL_OP_RASTER_POS2F: GlOp = 35;
pub const SGI_GL_OP_RASTER_POS3F: GlOp = 36;
pub const SGI_GL_OP_RASTER_POS4F: GlOp = 37;
pub const SGI_GL_OP_INDEXF: GlOp = 38;
pub const SGI_GL_OP_INDEX_MASK: GlOp = 39;
pub const SGI_GL_OP_CLEAR_INDEX: GlOp = 40;
pub const SGI_GL_OP_DEPTH_RANGE: GlOp = 41;
//
// glLightfv: the light's number, 0 for GL_LIGHT0, the parameter and
// four floats, of which the parameter uses as many as it takes.
//
pub const SGI_GL_OP_LIGHT: GlOp = 42;
// glMaterialfv: the face, the parameter and four floats, likewise.
pub const SGI_GL_OP_MATERIAL: GlOp = 43;
//
// glLightModelfv: the parameter and four floats; a boolean parameter
// is true for a nonzero first float.
//
pub const SGI_GL_OP_LIGHT_MODEL: GlOp = 44;
pub const SGI_GL_OP_COLOR_MATERIAL: GlOp = 45;
pub const SGI_GL_OP_CULL_FACE: GlOp = 46;
pub const SGI_GL_OP_FRONT_FACE: GlOp = 47;
pub const SGI_GL_OP_SCISSOR: GlOp = 48;
pub const SGI_GL_OP_ALPHA_FUNC: GlOp = 49;
// GL_NEVER to GL_ALWAYS, the reference as int32, the value mask
pub const SGI_GL_OP_STENCIL_FUNC: GlOp = 50;
pub const SGI_GL_OP_STENCIL_OP: GlOp = 51;
pub const SGI_GL_OP_STENCIL_MASK: GlOp = 52;
pub const SGI_GL_OP_CLEAR_STENCIL: GlOp = 53;
pub const SGI_GL_OP_BLEND_FUNC: GlOp = 54;
pub const SGI_GL_OP_BLEND_EQUATION: GlOp = 55;
pub const SGI_GL_OP_BLEND_COLOR: GlOp = 56;
pub const SGI_GL_OP_LOGIC_OP: GlOp = 57;
pub const SGI_GL_OP_COLOR_MASK: GlOp = 58;
//
// glFogfv: the parameter and four floats, of which it uses as many as
// it takes; GL_FOG_MODE's is the mode's enumerant as a float.
//
pub const SGI_GL_OP_FOG: GlOp = 59;
pub const SGI_GL_OP_POLYGON_OFFSET: GlOp = 60;
pub const SGI_GL_OP_POLYGON_OFFSET_EXT: GlOp = 61;
pub const SGI_GL_OP_CLIP_PLANE: GlOp = 62;
pub const SGI_GL_OP_POINT_SIZE: GlOp = 63;
pub const SGI_GL_OP_LINE_WIDTH: GlOp = 64;
pub const SGI_GL_OP_LINE_STIPPLE: GlOp = 65;
pub const SGI_GL_OP_POLYGON_MODE: GlOp = 66;
pub const SGI_GL_OP_TEXCOORD2F: GlOp = 67;
pub const SGI_GL_OP_TEXCOORD3F: GlOp = 68;
pub const SGI_GL_OP_TEXCOORD4F: GlOp = 69;
// Parameter and four floats; mode's enumerant as a float.
pub const SGI_GL_OP_TEX_ENV: GlOp = 70;
//
// IRIS GL swaptmesh as the OpenGL client passes it through an open
// triangle strip: the next vertex makes its triangle with the strip's
// oldest retained vertex instead of its middle one.
//
pub const SGI_GL_OP_SWAP_TMESH: GlOp = 71;
pub const SGI_GL_OP_HINT: GlOp = 72;
//
// glTexGenfv: the coordinate (GL_S to GL_Q), the parameter and four
// floats; GL_TEXTURE_GEN_MODE's mode is its enumerant as a float.
//
pub const SGI_GL_OP_TEX_GEN: GlOp = 73;

/// Only the fields a kind the rasterizer executes needs are here.
/// Coordinates are plane pixels, the rectangle inclusive.
#[derive(Clone, Copy, Default)]
pub struct RasterOp {
    /// RasterKind.
    pub kind: u32,
    /// The colour targets written, as the drawable names them.
    pub targets: [ColourTarget; SGI_GL_MAX_TARGETS as usize],
    pub ntargets: u32,
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
    /// Only pixels inside these are written; at most the rectangle's. An
    /// empty list is how a family says its window has no visible pixel.
    pub clip: ClipList,
    /// CLEAR: SGI_GL_COLOR_BUFFER_BIT and SGI_GL_DEPTH_BUFFER_BIT.
    pub clear_buffers: u32,
    /// The GL clear colour; the rasterizer quantizes it to the plane.
    pub clear_rgba: [f32; 4],
    /// glColorMask's channels written: SGI_GL_COLOUR_MASK_RED and so on.
    pub clear_colour_mask: u32,
    /// In the depth plane's units; the plane keeps its low bits.
    pub clear_depth24: u32,
    /// SGI_GL_STENCIL_BUFFER_BIT: the clear value, written under the
    /// stencil write mask; the plane keeps its low bits.
    pub clear_stencil: u32,
    pub stencil_mask: u32,
    /// CLEAR in colour-index mode: nonzero index_bits, and the clear index
    /// already masked to them, written under index_mask.
    pub index_bits: u32,
    pub clear_index: u32,
    pub index_mask: u32,
}

/// Where a context draws, as the family's context switch, window clip and
/// buffer selection give it. Coordinates are plane pixels.
#[derive(Clone, Copy, Default)]
pub struct GlDrawable {
    /// 0 while the family has no window for the context: nothing draws, and
    /// each draw, clear or bitmap is logged as dropped. 0 or 1.
    pub bound: u32,
    /// The window's lower-left pixel, where window coordinates start; its
    /// top-left under a window matrix whose rows run down the display.
    pub origin: [i32; 2],
    /// 0/0 until the family supplies a validated drawable extent.
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
    /// The target SGI_GL_DRAW_OVERLAY names; zero where none.
    pub overlay: ColourTarget,
}

/// The GL state a draw is made under.
#[derive(Clone, Copy, Default)]
pub struct GlDrawState {
    /// Column-major, as glLoadMatrixf lays a matrix out.
    pub modelview: [f32; 16],
    pub projection: [f32; 16],
    pub viewport: [i32; 4],
    /// GL_FLAT or GL_SMOOTH.
    pub shade_model: u32,
    /// 0 or 1.
    pub depth_test: u32,
    /// The comparison minus GL_NEVER, 0 to 7.
    pub depth_func: u32,
    /// 0 or 1.
    pub depth_mask: u32,
    /// glDepthRange's near and far, each clamped to [0, 1] (2.10.1).
    pub depth_range: [f32; 2],
    /// SGI_GL_DRAW_* bits; those the drawable has no target for are ignored.
    pub draw_buffers: u32,
    /// 0 in RGBA mode; in colour-index mode the bits a colour index holds
    /// (GlContext::set_index_mode()).
    pub index_bits: u32,
    /// glIndexMask's write mask (4.2.2).
    pub index_mask: u32,
    /// glColorMask's channels, SGI_GL_COLOUR_MASK_RED and so on (4.2.2).
    pub colour_mask: u32,
    /// Polygon culling (3.5.1): GL_CULL_FACE's enable (0 or 1),
    /// glCullFace's mode, and glFrontFace's GL_CW or GL_CCW. Any
    /// front_face but GL_CW is GL_CCW.
    pub cull_face: u32,
    pub cull_face_mode: u32,
    pub front_face: u32,
    /// The scissor test (4.1.2): its enable (0 or 1) and glScissor's box,
    /// x, y, width and height in window coordinates.
    pub scissor_test: u32,
    pub scissor: [i32; 4],
    /// The alpha test (4.1.4): its enable (0 or 1), glAlphaFunc's
    /// comparison minus GL_NEVER, and its reference clamped to [0, 1].
    pub alpha_test: u32,
    pub alpha_func: u32,
    pub alpha_ref: f32,
    /// The stencil test (4.1.5): its enable (0 or 1), glStencilFunc's
    /// comparison minus GL_NEVER, reference (kept as sent) and value mask,
    /// glStencilMask's write mask, and glStencilOp's three operations as
    /// enumerants.
    pub stencil_test: u32,
    pub stencil_func: u32,
    pub stencil_ref: i32,
    pub stencil_value_mask: u32,
    pub stencil_write_mask: u32,
    pub stencil_op: [u32; 3],
    /// Blending (4.1.6): its enable (0 or 1), glBlendFunc's factors,
    /// glBlendEquation's equation, and glBlendColor clamped to [0, 1].
    pub blend: u32,
    pub blend_src: u32,
    pub blend_dst: u32,
    pub blend_equation: u32,
    pub blend_color: [f32; 4],
    /// The logical operation (4.1.8): the GL_INDEX_LOGIC_OP and
    /// GL_COLOR_LOGIC_OP enables (0 or 1) and glLogicOp's operation.
    pub index_logic_op: u32,
    pub color_logic_op: u32,
    pub logic_op_mode: u32,
    /// Fog (3.10): its enable (0 or 1), mode, density, start and end, the
    /// fog colour clamped to [0, 1] and the fog index.
    pub fog: u32,
    pub fog_mode: u32,
    pub fog_density: f32,
    pub fog_start: f32,
    pub fog_end: f32,
    pub fog_color: [f32; 4],
    pub fog_index: f32,
    /// Polygon offset (3.5.5): the enables as SGI_GL_OFFSET_FILL and so
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
    pub clip_plane: [[f32; 4]; SGI_GL_CLIP_PLANES as usize],
    pub drawable: GlDrawable,
    /// GL_DITHER's enable (0 or 1), initially on (4.1.7).
    pub dither: u32,
    /// A window matrix, for a family whose board takes one eye-to-window
    /// transform in place of a projection and a viewport; none of it is GL
    /// state (GlContext::set_window_matrix()). While window_matrix_given is 1,
    /// an eye position e goes to h = W e by window_matrix W, column-major,
    /// and h replaces the projection, the divide and the viewport map:
    /// window x and y are h0 / h3 and h1 / h3, unrounded, in pixels from
    /// the drawable's origin, and window depth is
    /// window_depth_scale * h2 / h3. With window_rows_down 1 window y
    /// counts rows down the display from the drawable's origin (then the
    /// window's top-left pixel). Clipping keeps near and far planes, window
    /// depth 0 and 1, and bounds x and y by SGI_GL_WINDOW_GUARD. This
    /// departs from OpenGL in raster position validity, bitmap rows, the tie
    /// rule and the scissor, which is measured from the top.
    pub window_matrix_given: u32,
    pub window_matrix: [f32; 16],
    pub window_depth_scale: f32,
    pub window_rows_down: u32,
    /// The texture environment belongs to the context, not its objects.
    pub texture_1d: u32,
    pub texture_2d: u32,
    pub texture_3d: u32,
    pub texture_env_mode: u32,
    pub texture_env_color: [f32; 4],
}

/// Host-owned deferred boundary. `vertices` was a malloc'd float array of
/// `capacity` floats with `count` vertices used; here a Vec sized by
/// `capacity`. Never shallow-copy a live context.
#[derive(Clone, Default)]
pub struct GlPolygonAssembly {
    pub vertices: Vec<f32>,
    pub count: u32,
    pub capacity: u32,
    pub failed: u32,
    pub error: u32,
    /// Host activation marker; not part of the context's GL state.
    pub enabled: u32,
    pub loaded: u32,
}

#[derive(Clone, Default)]
pub struct GlContext {
    /// The hardware identity the family keys this context by.
    pub hw_id: u32,
    pub cur: GlDrawState,
    /// The texture matrix (2.10.2), applied to each vertex before clipping.
    pub texture: [f32; 16],
    pub cur_color: [f32; 4],
    pub cur_index: f32,
    pub cur_normal: [f32; 3],
    /// 0 or 1.
    pub cur_edge: u32,
    /// The current raster position (2.12): window x, y and z and clip w,
    /// whether it is valid (0 or 1), its eye distance and its colour.
    pub raster_pos: [f32; 4],
    pub raster_valid: u32,
    pub raster_color: [f32; 4],
    pub raster_index: f32,
    /// In the depth plane's units, as the family delivered it; a clear
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
    /// The primitive open since glBegin, SGI_GL_POINTS to SGI_GL_POLYGON.
    pub begin_mode: u32,
    /// 0 or 1.
    pub in_begin: u32,
    /// Vertices waiting for the rest of their piece, nheld of them.
    pub held: [[f32; SGI_GL_VERTEX_FLOATS]; SGI_GL_HELD_VERTICES as usize],
    pub nheld: u32,
    /// 1 while the next triangle of an open strip is an odd one, whose
    /// first two vertices are sent swapped (2.6.1).
    pub strip_odd: u32,
    /// 1 while held[2] keeps the open triangle strip's oldest retained
    /// vertex, which a swap makes the next triangle's first vertex.
    pub strip_prev: u32,
    /// 1 from a swap until the next vertex of the strip.
    pub strip_swap: u32,
    pub cur_texcoord: [f32; 4],
    pub raster_texcoord: [f32; 4],
    /// Host policy and presence markers; not part of the context's GL state.
    pub standalone_stipple: bool,
    pub line_phase_loaded: bool,
    pub draw_error_loaded: bool,
    pub line_phase: u32,
    /// Advisory modes are retained even though rendering ignores them.
    pub hints: [u32; SGI_GL_HINTS as usize],
    pub sprite_mode: u32,
    pub sprite_axis: [f32; 3],
    pub tex_gen_mode: [u32; 4],
    pub tex_gen_object_plane: [[f32; 4]; 4],
    pub tex_gen_eye_plane: [[f32; 4]; 4],
}

/// A decoded command: its entry point and its scalar arguments as 32-bit
/// words in the specification's order, a float argument as the IEEE single
/// bits the client sent.
#[derive(Clone, Copy)]
pub struct GlCmd<'a> {
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
pub struct GlDraw<'a> {
    /// The context's hw_id.
    pub context: u32,
    /// SGI_GL_POINTS to SGI_GL_POLYGON.
    pub mode: u32,
    pub count: u32,
    /// count * SGI_GL_VERTEX_FLOATS floats, texture-matrix-transformed strq
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
pub trait GlSink {
    /// Rasterize one draw; a line stipple's phase comes back in
    /// d.line_phase.
    fn draw(&mut self, d: &mut GlDraw) -> GlDrawResult;
}

/// The most words one GlContext::get() answer takes: a matrix's.
pub const SGI_GL_GET_WORDS: usize = 16;

// Functions.

use std::sync::atomic::{AtomicU32, Ordering};

// The last client clip plane and light enumerants.
pub const SGI_GL_CLIP_PLANE_LAST: u32 = SGI_GL_CLIP_PLANE0 + SGI_GL_CLIP_PLANES - 1;

/// Print a message once per site.
pub fn gl_report_once(site: u32, msg: &str) {
    static REPORTED: AtomicU32 = AtomicU32::new(0);
    let bit = 1u32 << site;

    if REPORTED.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        eprintln!("{}", msg);
    }
}

/// The word `i` of a command's arguments (a short list reads as 0 rather
/// than past the end).
pub fn gl_arg(p: &[u32], i: usize) -> u32 {
    p.get(i).copied().unwrap_or(0)
}

pub fn gl_float(bits: u32) -> f32 {
    f32::from_bits(bits)
}

pub fn gl_bits(value: f32) -> u32 {
    value.to_bits()
}

impl GlContext {

    /// OpenGL 1.2.1 initial state (2.7, 2.10, 2.13.7, 4.1.5, 4.2.1-4.2.3).
    pub fn init(&mut self, hw_id: u32) {
        *self = GlContext::default();
        self.hw_id = hw_id;
        for i in 0..SGI_GL_HINTS as usize {
            self.hints[i] = SGI_GL_DONT_CARE;
        }
        for i in 0..4 {
            self.cur_color[i] = 1.0;
        }
        self.cur.shade_model = SGI_GL_SMOOTH;
        self.sprite_mode = SGI_GL_SPRITE_AXIAL_SGIX;
        self.sprite_axis[2] = 1.0;
        self.cur.dither = 1;
        self.cur.modelview = IDENT;
        self.cur.projection = IDENT;
        self.texture = IDENT;
        self.cur.depth_func = SGI_GL_LESS;
        self.cur.alpha_func = SGI_GL_ALWAYS;
        self.cur.stencil_func = SGI_GL_ALWAYS;
        self.cur.stencil_value_mask = u32::MAX;
        self.cur.stencil_write_mask = u32::MAX;
        for i in 0..3 {
            self.cur.stencil_op[i] = SGI_GL_KEEP;
        }
        self.cur.blend_src = 1;
        self.cur.blend_equation = SGI_GL_FUNC_ADD;
        self.cur.logic_op_mode = SGI_GL_COPY;
        self.cur.texture_env_mode = SGI_GL_MODULATE;
        self.cur_texcoord[3] = 1.0;
        self.raster_texcoord[3] = 1.0;
        for i in 0..4 {
            self.tex_gen_mode[i] = SGI_GL_EYE_LINEAR;
        }
        self.tex_gen_object_plane[0][0] = 1.0;
        self.tex_gen_eye_plane[0][0] = 1.0;
        self.tex_gen_object_plane[1][1] = 1.0;
        self.tex_gen_eye_plane[1][1] = 1.0;
        self.cur.fog_mode = SGI_GL_EXP;
        self.cur.fog_density = 1.0;
        self.cur.fog_end = 1.0;
        self.point_size = 1.0;
        self.line_width = 1.0;
        self.line_stipple_repeat = 1;
        self.line_stipple_pattern = 0xffff;
        self.polygon_mode[0] = SGI_GL_FILL;
        self.polygon_mode[1] = SGI_GL_FILL;
        self.polygon_stipple = [0xffff_ffff; 32];
        self.cur.depth_mask = 1;
        self.cur.depth_range[1] = 1.0;
        self.clear_depth = u32::MAX;
        self.cur.draw_buffers = SGI_GL_DRAW_FRONT;
        self.cur.index_mask = u32::MAX;
        self.cur.colour_mask = SGI_GL_COLOUR_MASK_ALL;
        self.cur.cull_face_mode = SGI_GL_BACK;
        self.cur.front_face = SGI_GL_CCW;
        self.cur_index = 1.0;
        self.cur_normal[2] = 1.0;
        self.cur_edge = 1;
        self.raster_pos[3] = 1.0;
        self.raster_valid = 1;
        for i in 0..4 {
            self.raster_color[i] = 1.0;
        }
        self.raster_index = 1.0;
    }

}

impl GlPolygonAssembly {
    pub fn polygon_release(&mut self) {
        self.vertices = Vec::new();
        self.count = 0;
        self.capacity = 0;
        self.failed = 0;
        self.loaded = 0;
    }
}

impl GlContext {
    /// Audited host consumers opt in; reset preserves that host policy.
    pub fn enable_polygon_modes(&mut self) -> bool {
        if self.in_begin != 0 {
            return false;
        }
        self.polygon.enabled = 1;
        true
    }

    pub fn enable_line_stipple(&mut self) -> bool {
        if self.in_begin != 0 {
            return false;
        }
        self.standalone_stipple = true;
        true
    }

    pub fn cleanup(&mut self) {
        self.polygon.polygon_release();
        self.polygon.enabled = 0;
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
    /// `vertices` is kept `capacity * SGI_GL_VERTEX_FLOATS` floats long.
    pub fn polygon_reserve(&mut self, count: usize) -> bool {
        let mut capacity: usize = std::cmp::max(self.capacity as usize, 8);

        if count > u32::MAX as usize || count > usize::MAX / SGI_GL_VERTEX_FLOATS / 4 {
            return false;
        }
        if count <= self.capacity as usize {
            return true;
        }
        while capacity < count {
            if capacity > (u32::MAX / 2) as usize
                || capacity > usize::MAX / SGI_GL_VERTEX_FLOATS / 4 / 2
            {
                capacity = count;
                break;
            }
            capacity *= 2;
        }
        let floats = capacity * SGI_GL_VERTEX_FLOATS;
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
        self.polygon.enabled != 0 || self.standalone_stipple
    }

    /// Account for a draw's result, for a draw made while draws_checked() was
    /// `checked`.
    pub fn draw_result(&mut self, checked: bool, result: GlDrawResult) {
        if !checked {
            return;
        }
        if result == SGI_GL_DRAW_OUT_OF_MEMORY && self.polygon.error == 0 {
            self.polygon.error = SGI_GL_OUT_OF_MEMORY;
        }
        if result == SGI_GL_DRAW_OK {
            return;
        }
        match result {
            SGI_GL_DRAW_OUT_OF_MEMORY => {}
            SGI_GL_DRAW_UNSUPPORTED => {
                gl_report_once(1, "sgi-gl: unsupported checked draw has no result consumer");
            }
            SGI_GL_DRAW_RESOURCE_ERROR => {
                gl_report_once(2, "sgi-gl: checked draw resource failure has no result consumer");
            }
            SGI_GL_DRAW_HOST_ERROR => {
                gl_report_once(3, "sgi-gl: checked draw host failure has no result consumer");
            }
            _ => {
                gl_report_once(
                    4,
                    &format!(
                        "sgi-gl: invalid checked draw result {} has no result consumer",
                        result
                    ),
                );
            }
        }
    }

    /// Canonical 32x32 pattern, copied on upload. Bit x in row y names window
    /// (x, y). Rejects use inside Begin/End.
    pub fn polygon_stipple(&mut self, rows: &[u32; 32]) -> bool {
        if self.in_begin != 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon stipple inside glBegin: invalid operation");
            return false;
        }
        self.polygon_stipple = *rows;
        true
    }

    /// Point the context at a drawable. A bound half-known extent is refused
    /// without changing the current drawable. Unbinding clears both dimensions.
    pub fn set_drawable(&mut self, d: &GlDrawable) -> bool {
        if d.bound != 0 && (d.width != 0) != (d.height != 0) {
            return false;
        }
        self.cur.drawable = *d;
        if d.bound == 0 {
            self.cur.drawable.width = 0;
            self.cur.drawable.height = 0;
        }
        true
    }

}

/// A colour index as a colour-index buffer of index_bits bits stores it
/// (OpenGL 1.2.1, 2.13.9): the integer part, masked to the buffer's bits.
/// 2.13.9 converts an index to fixed point with an unspecified number of
/// fraction bits and keeps the integer part, which for a value already in
/// fixed point is its floor; a value beyond 32 bits, or not a number,
/// keeps nothing a buffer could hold and stores 0.
pub fn gl_index_value(index_bits: u32, index: f64) -> u32 {
    let mask: u32 = if index_bits >= 32 { u32::MAX } else { (1u32 << index_bits) - 1 };
    let f = index.floor();

    if !(f >= -2147483648.0 && f < 4294967296.0) {
        return 0;
    }
    (if f < 0.0 { (f as i32) as u32 } else { f as u32 }) & mask
}

/// inv = m^-1, both column-major, by cofactors in double precision; false,
/// with inv unset, for a singular or non-finite m.
pub fn gl_invert(m: &[f32; 16], inv: &mut [f64; 16]) -> bool {
    let mut a = [0.0f64; 16];
    let mut det = 0.0f64;

    for i in 0..16 {
        a[i] = m[i] as f64;
    }
    for col in 0..4usize {
        for row in 0..4usize {
            let mut minor = [0.0f64; 9];
            let mut n = 0;

            for c in 0..4usize {
                for r in 0..4usize {
                    if c != col && r != row {
                        minor[n] = a[c * 4 + r];
                        n += 1;
                    }
                }
            }
            // The cofactor of a[col][row] is the adjugate's [row][col].
            let sign: f64 = if (row + col) & 1 != 0 { -1.0 } else { 1.0 };
            inv[row * 4 + col] = sign
                * (minor[0] * (minor[4] * minor[8] - minor[5] * minor[7])
                    - minor[3] * (minor[1] * minor[8] - minor[2] * minor[7])
                    + minor[6] * (minor[1] * minor[5] - minor[2] * minor[4]));
        }
    }
    for k in 0..4 {
        det += a[k * 4] * inv[k];
    }
    if det == 0.0 || !det.is_finite() {
        return false;
    }
    for i in 0..16 {
        inv[i] /= det;
    }
    true
}

impl GlContext {
    /// The current matrix of the stack a matrix mode names, or None.
    pub fn slot_matrix(&mut self, slot: u32) -> Option<&mut [f32; 16]> {
        match slot {
            SGI_GL_MODELVIEW => Some(&mut self.cur.modelview),
            SGI_GL_PROJECTION => Some(&mut self.cur.projection),
            SGI_GL_TEXTURE => Some(&mut self.texture),
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
    /// SGI_GL_MODELVIEW, SGI_GL_PROJECTION or SGI_GL_TEXTURE, whatever the
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
pub const SGI_GL_NARGS_LEN: usize = 74;
pub static GL_NARGS: [u8; SGI_GL_NARGS_LEN] = {
    let mut t = [0u8; SGI_GL_NARGS_LEN];
    t[SGI_GL_OP_HINT as usize] = 2;
    t[SGI_GL_OP_VIEWPORT as usize] = 4;
    t[SGI_GL_OP_CLEAR_COLOR as usize] = 4;
    t[SGI_GL_OP_CLEAR_DEPTH as usize] = 1;
    t[SGI_GL_OP_SHADE_MODEL as usize] = 1;
    t[SGI_GL_OP_MATRIX_MODE as usize] = 1;
    t[SGI_GL_OP_LOAD_MATRIX as usize] = 16;
    t[SGI_GL_OP_LOAD_IDENTITY as usize] = 0;
    t[SGI_GL_OP_MULT_MATRIX as usize] = 16;
    t[SGI_GL_OP_FRUSTUM as usize] = 6;
    t[SGI_GL_OP_ORTHO as usize] = 6;
    t[SGI_GL_OP_PUSH_MATRIX as usize] = 0;
    t[SGI_GL_OP_POP_MATRIX as usize] = 0;
    t[SGI_GL_OP_ROTATE as usize] = 4;
    t[SGI_GL_OP_SCALE as usize] = 3;
    t[SGI_GL_OP_TRANSLATE as usize] = 3;
    t[SGI_GL_OP_ENABLE as usize] = 1;
    t[SGI_GL_OP_DISABLE as usize] = 1;
    t[SGI_GL_OP_DEPTH_FUNC as usize] = 1;
    t[SGI_GL_OP_DEPTH_MASK as usize] = 1;
    t[SGI_GL_OP_DRAW_BUFFERS as usize] = 1;
    t[SGI_GL_OP_COLOR3F as usize] = 3;
    t[SGI_GL_OP_COLOR4F as usize] = 4;
    t[SGI_GL_OP_COLOR3UB as usize] = 3;
    t[SGI_GL_OP_COLOR4UB as usize] = 4;
    t[SGI_GL_OP_NORMAL3F as usize] = 3;
    t[SGI_GL_OP_EDGE_FLAG as usize] = 1;
    t[SGI_GL_OP_BEGIN as usize] = 1;
    t[SGI_GL_OP_END as usize] = 0;
    t[SGI_GL_OP_VERTEX2F as usize] = 2;
    t[SGI_GL_OP_VERTEX3F as usize] = 3;
    t[SGI_GL_OP_VERTEX4F as usize] = 4;
    t[SGI_GL_OP_VERTEX2I as usize] = 2;
    t[SGI_GL_OP_FLUSH as usize] = 0;
    t[SGI_GL_OP_CLEAR as usize] = 1;
    t[SGI_GL_OP_FINISH as usize] = 0;
    t[SGI_GL_OP_RASTER_POS2F as usize] = 2;
    t[SGI_GL_OP_RASTER_POS3F as usize] = 3;
    t[SGI_GL_OP_RASTER_POS4F as usize] = 4;
    t[SGI_GL_OP_INDEXF as usize] = 1;
    t[SGI_GL_OP_INDEX_MASK as usize] = 1;
    t[SGI_GL_OP_CLEAR_INDEX as usize] = 1;
    t[SGI_GL_OP_DEPTH_RANGE as usize] = 2;
    t[SGI_GL_OP_LIGHT as usize] = 6;
    t[SGI_GL_OP_MATERIAL as usize] = 6;
    t[SGI_GL_OP_LIGHT_MODEL as usize] = 5;
    t[SGI_GL_OP_COLOR_MATERIAL as usize] = 2;
    t[SGI_GL_OP_CULL_FACE as usize] = 1;
    t[SGI_GL_OP_FRONT_FACE as usize] = 1;
    t[SGI_GL_OP_SCISSOR as usize] = 4;
    t[SGI_GL_OP_ALPHA_FUNC as usize] = 2;
    t[SGI_GL_OP_STENCIL_FUNC as usize] = 3;
    t[SGI_GL_OP_STENCIL_OP as usize] = 3;
    t[SGI_GL_OP_STENCIL_MASK as usize] = 1;
    t[SGI_GL_OP_CLEAR_STENCIL as usize] = 1;
    t[SGI_GL_OP_BLEND_FUNC as usize] = 2;
    t[SGI_GL_OP_BLEND_EQUATION as usize] = 1;
    t[SGI_GL_OP_BLEND_COLOR as usize] = 4;
    t[SGI_GL_OP_LOGIC_OP as usize] = 1;
    t[SGI_GL_OP_COLOR_MASK as usize] = 4;
    t[SGI_GL_OP_FOG as usize] = 5;
    t[SGI_GL_OP_POLYGON_OFFSET as usize] = 2;
    t[SGI_GL_OP_POLYGON_OFFSET_EXT as usize] = 2;
    t[SGI_GL_OP_CLIP_PLANE as usize] = 5;
    t[SGI_GL_OP_POINT_SIZE as usize] = 1;
    t[SGI_GL_OP_LINE_WIDTH as usize] = 1;
    t[SGI_GL_OP_LINE_STIPPLE as usize] = 2;
    t[SGI_GL_OP_POLYGON_MODE as usize] = 2;
    t[SGI_GL_OP_TEXCOORD2F as usize] = 2;
    t[SGI_GL_OP_TEXCOORD3F as usize] = 3;
    t[SGI_GL_OP_TEXCOORD4F as usize] = 4;
    t[SGI_GL_OP_TEX_ENV as usize] = 5;
    t[SGI_GL_OP_TEX_GEN as usize] = 6;
    t[SGI_GL_OP_SWAP_TMESH as usize] = 0;
    t
};

impl GlDrawState {
    /// The colour targets the draw state's draw buffers select, in bit order
    /// and each once, into out; returns how many.
    pub fn draw_targets(&self, out: &mut [ColourTarget; SGI_GL_MAX_TARGETS as usize]) -> u32 {
        let pick: [ColourTarget; SGI_GL_MAX_TARGETS as usize] = [
            self.drawable.front,
            self.drawable.back,
            self.drawable.extra[0],
            self.drawable.extra[1],
        ];
        let mut n: u32 = 0;

        if (self.draw_buffers & SGI_GL_DRAW_OVERLAY) != 0 && self.drawable.overlay != 0 {
            out[0] = self.drawable.overlay;
            return 1;
        }
        for bit in 0..SGI_GL_MAX_TARGETS as usize {
            let mut dup = false;

            if self.draw_buffers & (1u32 << bit) == 0 {
                continue;
            }
            if bit >= 2 && pick[bit] == 0 {
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
            for i in 0..n as usize {
                dup |= out[i] == pick[bit];
            }
            if !dup {
                out[n as usize] = pick[bit];
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
    pub fn submit(&mut self, sink: &mut dyn GlSink, mode: u32, count: u32, v0: Option<&[f32; SGI_GL_VERTEX_FLOATS]>, v1: Option<&[f32; SGI_GL_VERTEX_FLOATS]>, v2: Option<&[f32; SGI_GL_VERTEX_FLOATS]>, v3: Option<&[f32; SGI_GL_VERTEX_FLOATS]>) {
        let stipple = if mode == SGI_GL_LINES { self.standalone_stipple } else { self.polygon.enabled != 0 };
        let v = [v0, v1, v2, v3];
        let mut verts = [0.0f32; 4 * SGI_GL_VERTEX_FLOATS];

        for i in 0..count as usize {
            if let Some(src) = v[i] {
                verts[i * SGI_GL_VERTEX_FLOATS..][..SGI_GL_VERTEX_FLOATS].copy_from_slice(src);
            }
        }
        let checked = self.draws_checked();
        let mut d = GlDraw::new(mode, count, &verts, &self.cur, &self.polygon_stipple);

        d.context = self.hw_id;
        d.point_size = self.point_size;
        d.line_width = self.line_width;
        d.point_smooth = self.raster_enables & SGI_GL_RASTER_POINT_SMOOTH != 0;
        d.line_smooth = self.raster_enables & SGI_GL_RASTER_LINE_SMOOTH != 0;
        d.polygon_smooth = self.raster_enables & SGI_GL_RASTER_POLYGON_SMOOTH != 0;
        d.polygon_modes = (self.polygon.enabled != 0).then_some(self.polygon_mode);
        d.line_stipple_enabled = stipple && (self.raster_enables & SGI_GL_RASTER_LINE_STIPPLE != 0);
        d.line_stipple_repeat = if stipple { self.line_stipple_repeat } else { 0 };
        d.line_stipple_pattern = if stipple { self.line_stipple_pattern } else { 0 };
        d.line_phase = (mode == SGI_GL_LINES && self.standalone_stipple).then_some(self.line_phase);
        d.line_phase_reset = self.begin_mode == SGI_GL_LINES;
        d.polygon_stipple_enabled = self.raster_enables & SGI_GL_RASTER_POLYGON_STIPPLE != 0;

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
    pub fn texture_coord(&self, obj: &[f32; 4], pos: &[f32; 4], n: &[f32; 3], out: &mut [f32; 4]) {
        *out = xform(&self.texture, self.cur_texcoord);
    }

    /// Hold a vertex in slot n of the three a context keeps.
    pub fn hold(&mut self, n: u32, v: &[f32; SGI_GL_VERTEX_FLOATS]) {
        self.held[n as usize] = *v;
    }
}

impl GlContext {

    /// Capture one vertex with the current attributes, without
    /// drawing. `bits` holds `components` words.
    pub fn capture_vertex(&mut self, mode: u32, bits: &[u32], components: u32, out: &mut [f32; SGI_GL_VERTEX_FLOATS]) {
        let normal: [f32; 3];
        let mut obj = [0.0f32; 4];

        *out = [0.0; SGI_GL_VERTEX_FLOATS];
        out[SGI_GL_V_POSITION + 3] = 1.0;
        for i in 0..std::cmp::min(components as usize, bits.len()) {
            out[SGI_GL_V_POSITION + i] = gl_float(bits[i]);
        }
        out[SGI_GL_V_COLOR..SGI_GL_V_COLOR + 4].copy_from_slice(&self.cur_color);
        normal = self.cur_normal;
        obj.copy_from_slice(&out[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4]);
        out[SGI_GL_V_NORMAL..SGI_GL_V_NORMAL + 3].copy_from_slice(&normal);
        let mut pos = [0.0f32; 4];
        pos.copy_from_slice(&out[SGI_GL_V_POSITION..SGI_GL_V_POSITION + 4]);
        let mut tc = [0.0f32; 4];
        self.texture_coord(&obj, &pos, &normal, &mut tc);
        out[SGI_GL_V_TEXCOORD..SGI_GL_V_TEXCOORD + 4].copy_from_slice(&tc);
        out[SGI_GL_V_EDGE] = f32::from_bits(self.cur_edge);
        out[SGI_GL_V_INDEX] = self.cur_index;
        // Material and lighting belong to the instant this vertex arrived.
        out[SGI_GL_V_BACK_COLOR..SGI_GL_V_BACK_COLOR + 4].copy_from_slice(&self.cur_color);
        out[SGI_GL_V_BACK_INDEX] = out[SGI_GL_V_INDEX];
    
        if self.polygon.enabled != 0
            && (mode == SGI_GL_TRIANGLE_STRIP || mode == SGI_GL_TRIANGLE_FAN || mode == SGI_GL_QUAD_STRIP)
        {
            out[SGI_GL_V_EDGE] = f32::from_bits(1);
        }
    }

    /// One vertex of the primitive open since glBegin (OpenGL 1.2.1, 2.6.1),
    /// carrying the current attributes. The context holds at most three
    /// earlier vertices, in held[] with nheld of them in use, and every piece
    /// is sent as its own draw as soon as its last vertex arrives, so a
    /// context switch between two vertices loses nothing. A line loop's
    /// closing segment is sent at glEnd. Vertices that complete no piece
    /// before glEnd draw nothing.
    pub fn vertex(&mut self, sink: &mut dyn GlSink, pos: &[f32; 4], components: u32) {
        let mut cur = [0.0f32; SGI_GL_VERTEX_FLOATS];
        let h0 = self.held[0];
        let h1 = self.held[1];
        let h2 = self.held[2];
        let n = self.nheld;

        if self.in_begin == 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: vertex outside glBegin ignored");
            return;
        }
        let bits = [pos[0].to_bits(), pos[1].to_bits(), pos[2].to_bits(), pos[3].to_bits()];
        let mode = self.begin_mode;
        self.capture_vertex(mode, &bits, components, &mut cur);
        if self.polygon.enabled != 0 && self.begin_mode == SGI_GL_POLYGON {
            let p = &mut self.polygon;

            if p.failed != 0 {
                return;
            }
            if p.count == u32::MAX || !p.polygon_reserve(p.count as usize + 1) {
                p.polygon_release();
                p.failed = 1;
                if p.error == 0 {
                    p.error = SGI_GL_OUT_OF_MEMORY;
                }
                return;
            }
            let at = p.count as usize * SGI_GL_VERTEX_FLOATS;
            p.vertices[at..at + SGI_GL_VERTEX_FLOATS].copy_from_slice(&cur);
            p.count += 1;
            return;
        }

        match self.begin_mode {
            SGI_GL_POINTS => {
                self.submit(sink, SGI_GL_POINTS, 1, Some(&cur), None, None, None);
                return;
            }
            SGI_GL_LINES => {
                if n >= 1 {
                    self.submit(sink, SGI_GL_LINES, 2, Some(&h0), Some(&cur), None, None);
                    self.nheld = 0;
                    return;
                }
            }
            SGI_GL_LINE_STRIP => {
                if n >= 1 {
                    self.submit(sink, SGI_GL_LINES, 2, Some(&h0), Some(&cur), None, None);
                    self.hold(0, &cur);
                    return;
                }
            }
            SGI_GL_LINE_LOOP => {
                // Slot 0 keeps the first vertex for the closing segment.
                if n >= 1 {
                    let prev = self.held[n as usize - 1];
                    self.submit(sink, SGI_GL_LINES, 2, Some(&prev), Some(&cur), None, None);
                    self.hold(1, &cur);
                    self.nheld = 2;
                    return;
                }
            }
            SGI_GL_TRIANGLES => {
                if n >= 2 {
                    self.submit(sink, SGI_GL_TRIANGLES, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    self.nheld = 0;
                    return;
                }
            }
            SGI_GL_TRIANGLE_STRIP => {
                if n >= 2 {
                    if self.strip_swap != 0 && self.strip_prev != 0 {
                        // After a swap the vertex replaces the middle one of the
                        // three the strip keeps, so its triangle shares the
                        // strip's oldest and newest vertices, and keeps the
                        // facing of the triangle before it, as a fan's does. The
                        // oldest stays the oldest for the next swap.
                        if self.strip_odd != 0 {
                            self.submit(sink, SGI_GL_TRIANGLES, 3, Some(&h2), Some(&h1), Some(&cur), None);
                        } else {
                            self.submit(sink, SGI_GL_TRIANGLES, 3, Some(&h1), Some(&h2), Some(&cur), None);
                        }
                        self.strip_swap = 0;
                        self.hold(0, &h1);
                        self.hold(1, &cur);
                        return;
                    }
                    self.strip_swap = 0;
                    if self.strip_odd != 0 {
                        self.submit(sink, SGI_GL_TRIANGLES, 3, Some(&h1), Some(&h0), Some(&cur), None);
                    } else {
                        self.submit(sink, SGI_GL_TRIANGLES, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    }
                    self.strip_odd = (self.strip_odd == 0) as u32;
                    self.hold(2, &h0);
                    self.strip_prev = 1;
                    self.hold(0, &h1);
                    self.hold(1, &cur);
                    return;
                }
            }
            SGI_GL_TRIANGLE_FAN => {
                if n >= 2 {
                    self.submit(sink, SGI_GL_TRIANGLES, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    self.hold(1, &cur);
                    return;
                }
            }
            SGI_GL_POLYGON => {
                if n >= 2 {
                    self.submit(sink, SGI_GL_POLYGON, 3, Some(&h0), Some(&h1), Some(&cur), None);
                    self.hold(1, &cur);
                    return;
                }
            }
            SGI_GL_QUADS => {
                if n >= 3 {
                    self.submit(sink, SGI_GL_QUADS, 4, Some(&h0), Some(&h1), Some(&h2), Some(&cur));
                    self.nheld = 0;
                    return;
                }
            }
            SGI_GL_QUAD_STRIP => {
                if n >= 3 {
                    self.submit(sink, SGI_GL_QUAD_STRIP, 4, Some(&h0), Some(&h1), Some(&h2), Some(&cur));
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
    /// An eye position's clip coordinates (2.10) under the draw state: by the
    /// projection, or under a window matrix the window matrix's h with x and y
    /// scaled by 1 / SGI_GL_WINDOW_GUARD and z by 2 window_depth_scale, less w.
    pub fn eye_to_clip(&self, eye: &[f64; 4], clip: &mut [f64; 4]) {
        let m = if self.window_matrix_given != 0 { &self.window_matrix } else { &self.projection };

        for row in 0..4 {
            clip[row] = 0.0;
            for k in 0..4 {
                clip[row] += (m[k * 4 + row] as f64) * eye[k];
            }
        }
        if self.window_matrix_given != 0 {
            clip[0] /= SGI_GL_WINDOW_GUARD as f64;
            clip[1] /= SGI_GL_WINDOW_GUARD as f64;
            clip[2] = 2.0 * self.window_depth_scale as f64 * clip[2] - clip[3];
        }
    }

    /// Clip coordinates as GlDrawState::eye_to_clip() gives them to window x and y
    /// and normalized depth, (z / w + 1) / 2, in out. w must be positive.
    pub fn clip_to_window(&self, clip: &[f64; 4], out: &mut [f64; 3]) {
        let vp = &self.viewport;

        if self.window_matrix_given != 0 {
            out[0] = clip[0] * SGI_GL_WINDOW_GUARD as f64 / clip[3];
            out[1] = clip[1] * SGI_GL_WINDOW_GUARD as f64 / clip[3];
        } else {
            // 2.10.1: xw = (px / 2) xd + ox, ox = x + w / 2.
            out[0] = (clip[0] / clip[3] + 1.0) * vp[2] as f64 / 2.0 + vp[0] as f64;
            out[1] = (clip[1] / clip[3] + 1.0) * vp[3] as f64 / 2.0 + vp[1] as f64;
        }
        out[2] = (clip[2] / clip[3] + 1.0) / 2.0;
    }
}

/// Whether op is one of glStencilOp's operations (4.1.5).
pub fn gl_stencil_op_valid(op: u32) -> bool {
    matches!(op, 0 | SGI_GL_KEEP | SGI_GL_REPLACE | SGI_GL_INCR | SGI_GL_DECR | SGI_GL_INVERT)
}

/// Whether glBlendFunc accepts factor as a source factor (OpenGL 1.2.1,
/// table 4.1, with the imaging subset's constant factors) or, with dst,
/// as a destination one (table 4.2).
pub fn gl_blend_factor_valid(factor: u32, dst: bool) -> bool {
    match factor {
        0
        | 1
        | SGI_GL_SRC_ALPHA
        | SGI_GL_ONE_MINUS_SRC_ALPHA
        | SGI_GL_DST_ALPHA
        | SGI_GL_ONE_MINUS_DST_ALPHA
        | SGI_GL_CONSTANT_COLOR
        | SGI_GL_ONE_MINUS_CONSTANT_COLOR
        | SGI_GL_CONSTANT_ALPHA
        | SGI_GL_ONE_MINUS_CONSTANT_ALPHA => true,
        SGI_GL_SRC_COLOR | SGI_GL_ONE_MINUS_SRC_COLOR => dst,
        SGI_GL_DST_COLOR | SGI_GL_ONE_MINUS_DST_COLOR | SGI_GL_SRC_ALPHA_SATURATE => !dst,
        _ => false,
    }
}

/// Whether glBlendEquation accepts equation.
pub fn gl_blend_equation_valid(equation: u32) -> bool {
    matches!(
        equation,
        SGI_GL_FUNC_ADD
            | SGI_GL_MIN
            | SGI_GL_MAX
            | SGI_GL_FUNC_SUBTRACT
            | SGI_GL_FUNC_REVERSE_SUBTRACT
            | SGI_GL_LOGIC_OP
    )
}

impl GlContext {
    /// glClipPlane (2.11): the plane p in object coordinates becomes p M^-1 in
    /// eye coordinates, M the modelview current at the call. A plane number
    /// the context does not keep, or a modelview with no inverse, which the
    /// specification leaves undefined, changes nothing.
    pub fn clip_plane(&mut self, plane: u32, p: &[f32; 4]) {
        let mut inv = [0.0f64; 16];

        if plane >= SGI_GL_CLIP_PLANES || !gl_invert(&self.cur.modelview, &mut inv) {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: glClipPlane of plane {} ignored", plane);
            return;
        }
        for j in 0..4 {
            let mut v = 0.0f64;

            // (p M^-1)_j = sum_i p_i M^-1[i][j], M^-1[i][j] at inv[j * 4 + i].
            for i in 0..4 {
                v += p[i] as f64 * inv[j * 4 + i];
            }
            self.cur.clip_plane[plane as usize][j] = v as f32;
        }
    }
}

/// The rasterization enables kept as raster_enables bits; 0 for any other
/// capability.
pub fn gl_raster_enable(cap: u32) -> u32 {
    match cap {
        SGI_GL_POINT_SMOOTH => SGI_GL_RASTER_POINT_SMOOTH,
        SGI_GL_LINE_SMOOTH => SGI_GL_RASTER_LINE_SMOOTH,
        SGI_GL_LINE_STIPPLE => SGI_GL_RASTER_LINE_STIPPLE,
        SGI_GL_POLYGON_SMOOTH => SGI_GL_RASTER_POLYGON_SMOOTH,
        SGI_GL_POLYGON_STIPPLE => SGI_GL_RASTER_POLYGON_STIPPLE,
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
            SGI_GL_OP_POINT_SIZE | SGI_GL_OP_LINE_WIDTH => {
                if self.in_begin != 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: size setter inside glBegin: invalid operation");
                    return;
                }
                if !f.is_finite() || !(f > 0.0) {
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-gl: {}({}) ignored",
                        if op as u32 == SGI_GL_OP_POINT_SIZE { "glPointSize" } else { "glLineWidth" },
                        f
                    );
                    return;
                }
                if op as u32 == SGI_GL_OP_POINT_SIZE {
                    self.point_size = f;
                } else {
                    self.line_width = f;
                }
            }
            SGI_GL_OP_LINE_STIPPLE => {
                if self.in_begin != 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: line stipple inside glBegin: invalid operation");
                    return;
                }
                let repeat = gl_arg(a, 0) as i32;
                self.line_stipple_repeat = std::cmp::min(std::cmp::max(repeat, 1), 256) as u32;
                self.line_stipple_pattern = gl_arg(a, 1) & 0xffff;
            }
            _ => {
                if self.in_begin != 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon mode inside glBegin: invalid operation");
                    return;
                }
                let a0 = gl_arg(a, 0);
                let a1 = gl_arg(a, 1);
                let faces: u32 = if a0 == SGI_GL_FRONT_AND_BACK {
                    3
                } else if a0 == SGI_GL_FRONT {
                    1
                } else if a0 == SGI_GL_BACK {
                    2
                } else {
                    0
                };
                if faces == 0 || a1.wrapping_sub(SGI_GL_POINT) > SGI_GL_FILL - SGI_GL_POINT {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glPolygonMode({:#x}, {:#x}) ignored", a0, a1);
                    return;
                }
                for i in 0..2usize {
                    if faces & (1u32 << i) != 0 {
                        self.polygon_mode[i] = a1;
                    }
                }
                if a1 != SGI_GL_FILL && self.polygon.enabled == 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: polygon mode {:#x} is kept; polygons are drawn filled", a1);
                }
            }
        }
    }

    pub fn capability(&mut self, cap: u32, on: bool) {
        let raster = gl_raster_enable(cap);
        let onv = on as u32;

        if raster != 0 {
            if (cap == SGI_GL_POLYGON_STIPPLE || cap == SGI_GL_LINE_STIPPLE) && self.in_begin != 0 {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: stipple enable inside glBegin: invalid operation");
                return;
            }
            self.raster_enables = if on { self.raster_enables | raster } else { self.raster_enables & !raster };
            if on && cap == SGI_GL_LINE_STIPPLE && !self.standalone_stipple {
                crate::dlog!(LogModule::Odyssey, "sgi-gl: standalone line stipple is kept; standalone lines are drawn without it");
            }
            return;
        }
        match cap {
            SGI_GL_DEPTH_TEST => self.cur.depth_test = onv,
            SGI_GL_CULL_FACE => self.cur.cull_face = onv,
            SGI_GL_SCISSOR_TEST => self.cur.scissor_test = onv,
            SGI_GL_ALPHA_TEST => self.cur.alpha_test = onv,
            SGI_GL_STENCIL_TEST => self.cur.stencil_test = onv,
            SGI_GL_BLEND => self.cur.blend = onv,
            SGI_GL_INDEX_LOGIC_OP => self.cur.index_logic_op = onv,
            SGI_GL_COLOR_LOGIC_OP => self.cur.color_logic_op = onv,
            SGI_GL_FOG => self.cur.fog = onv,
            SGI_GL_POLYGON_OFFSET_FILL => self.cur.polygon_offset = deposit32(self.cur.polygon_offset, 0, 1, onv),
            SGI_GL_POLYGON_OFFSET_LINE => self.cur.polygon_offset = deposit32(self.cur.polygon_offset, 1, 1, onv),
            SGI_GL_POLYGON_OFFSET_POINT => self.cur.polygon_offset = deposit32(self.cur.polygon_offset, 2, 1, onv),
            SGI_GL_CLIP_PLANE0..=SGI_GL_CLIP_PLANE_LAST => {
                self.cur.clip_planes = deposit32(self.cur.clip_planes, cap - SGI_GL_CLIP_PLANE0, 1, onv);
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
pub fn gl_words4(a: &[u32], from: usize) -> [f32; 4] {
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
            SGI_GL_OP_TEXCOORD2F | SGI_GL_OP_TEXCOORD3F | SGI_GL_OP_TEXCOORD4F => {
                self.cur_texcoord[2] = 0.0;
                self.cur_texcoord[3] = 1.0;
                for i in 0..a.len() {
                    self.cur_texcoord[i] = gl_float(a[i]);
                }
            }
            SGI_GL_OP_SHADE_MODEL => self.cur.shade_model = a[0],
            SGI_GL_OP_ENABLE | SGI_GL_OP_DISABLE => {
                self.capability(a[0], cmd.op as u32 == SGI_GL_OP_ENABLE);
            }
            SGI_GL_OP_DEPTH_FUNC => {
                if a[0].wrapping_sub(SGI_GL_NEVER_ENUM) > SGI_GL_ALWAYS {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: depth function {:#x} ignored", a[0]);
                    return;
                }
                self.cur.depth_func = a[0] - SGI_GL_NEVER_ENUM;
            }
            SGI_GL_OP_DEPTH_MASK => self.cur.depth_mask = (a[0] != 0) as u32,
            SGI_GL_OP_CULL_FACE => {
                if gl_faces(a[0]) == 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glCullFace({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.cull_face_mode = a[0];
            }
            SGI_GL_OP_FRONT_FACE => {
                if a[0] != SGI_GL_CW && a[0] != SGI_GL_CCW {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glFrontFace({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.front_face = a[0];
            }
            SGI_GL_OP_DRAW_BUFFERS => self.cur.draw_buffers = a[0],
            SGI_GL_OP_SCISSOR => {
                // A negative width or height is an error and changes nothing.
                if (a[2] as i32) < 0 || (a[3] as i32) < 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glScissor of {} by {} ignored", a[2] as i32, a[3] as i32);
                    return;
                }
                for i in 0..4 {
                    self.cur.scissor[i] = a[i] as i32;
                }
            }
            SGI_GL_OP_ALPHA_FUNC => {
                let r = gl_float(a[1]);

                if a[0].wrapping_sub(SGI_GL_NEVER_ENUM) > SGI_GL_ALWAYS {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: alpha function {:#x} ignored", a[0]);
                    return;
                }
                self.cur.alpha_func = a[0] - SGI_GL_NEVER_ENUM;
                // Clamped to [0, 1] (4.1.4); not a number clamps to 0.
                self.cur.alpha_ref = if r >= 1.0 {
                    1.0
                } else if r > 0.0 {
                    r
                } else {
                    0.0
                };
            }
            SGI_GL_OP_STENCIL_FUNC => {
                if a[0].wrapping_sub(SGI_GL_NEVER_ENUM) > SGI_GL_ALWAYS {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: stencil function {:#x} ignored", a[0]);
                    return;
                }
                self.cur.stencil_func = a[0] - SGI_GL_NEVER_ENUM;
                self.cur.stencil_ref = a[1] as i32;
                self.cur.stencil_value_mask = a[2];
            }
            SGI_GL_OP_STENCIL_OP => {
                for i in 0..3 {
                    if !gl_stencil_op_valid(a[i]) {
                        crate::dlog!(LogModule::Odyssey, "sgi-gl: glStencilOp({:#x}, {:#x}, {:#x}) ignored", a[0], a[1], a[2]);
                        return;
                    }
                }
                self.cur.stencil_op.copy_from_slice(&a[..3]);
            }
            SGI_GL_OP_STENCIL_MASK => self.cur.stencil_write_mask = a[0],
            SGI_GL_OP_BLEND_FUNC => {
                if !gl_blend_factor_valid(a[0], false) || !gl_blend_factor_valid(a[1], true) {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glBlendFunc({:#x}, {:#x}) ignored", a[0], a[1]);
                    return;
                }
                self.cur.blend_src = a[0];
                self.cur.blend_dst = a[1];
            }
            SGI_GL_OP_BLEND_EQUATION => {
                if !gl_blend_equation_valid(a[0]) {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glBlendEquation({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.blend_equation = a[0];
            }
            SGI_GL_OP_BLEND_COLOR => {
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
            SGI_GL_OP_LOGIC_OP => {
                if a[0].wrapping_sub(SGI_GL_CLEAR) > SGI_GL_SET - SGI_GL_CLEAR {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glLogicOp({:#x}) ignored", a[0]);
                    return;
                }
                self.cur.logic_op_mode = a[0];
            }
            SGI_GL_OP_POLYGON_OFFSET | SGI_GL_OP_POLYGON_OFFSET_EXT => {
                self.cur.offset_factor = gl_float(a[0]);
                self.cur.offset_units =
                    if cmd.op as u32 == SGI_GL_OP_POLYGON_OFFSET { gl_float(a[1]) } else { 0.0 };
                self.cur.offset_bias =
                    if cmd.op as u32 == SGI_GL_OP_POLYGON_OFFSET_EXT { gl_float(a[1]) } else { 0.0 };
            }
            SGI_GL_OP_CLIP_PLANE => {
                pos = gl_words4(a, 1);
                self.clip_plane(a[0], &pos);
            }
            SGI_GL_OP_POINT_SIZE | SGI_GL_OP_LINE_WIDTH | SGI_GL_OP_LINE_STIPPLE | SGI_GL_OP_POLYGON_MODE => {
                self.raster_state(cmd.op, a);
            }
            SGI_GL_OP_COLOR3F => {
                for i in 0..3 {
                    self.cur_color[i] = gl_float(a[i]);
                }
                self.cur_color[3] = 1.0;
            }
            SGI_GL_OP_COLOR4F => {
                for i in 0..4 {
                    self.cur_color[i] = gl_float(a[i]);
                }
            }
            SGI_GL_OP_COLOR3UB | SGI_GL_OP_COLOR4UB => {
                // A byte c is c / 255 (OpenGL 1.2.1, table 2.6); 3ub's alpha is 1.
                for i in 0..4usize {
                    self.cur_color[i] =
                        if i == a.len() { 1.0 } else { (a[i] & 0xff) as f32 / 255.0 };
                }
            }
            SGI_GL_OP_BEGIN => {
                if self.in_begin != 0 {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: nested glBegin ignored");
                    return;
                }
                if a[0] > SGI_GL_POLYGON {
                    crate::dlog!(LogModule::Odyssey, "sgi-gl: glBegin of mode {} ignored", a[0]);
                    return;
                }
                self.line_phase = 0;
                self.in_begin = 1;
                self.begin_mode = a[0];
                self.nheld = 0;
                self.strip_odd = 0;
                self.strip_prev = 0;
                self.strip_swap = 0;
            }
            SGI_GL_OP_END => {
                if self.in_begin == 0 && self.polygon.enabled != 0 {
                    return;
                }
                if self.polygon.enabled != 0 && self.begin_mode == SGI_GL_POLYGON {
                    if self.polygon.failed == 0 && self.polygon.count >= 3 {
                        let checked = self.draws_checked();
                        let mut d = GlDraw::new(
                            SGI_GL_POLYGON,
                            self.polygon.count,
                            &self.polygon.vertices,
                            &self.cur,
                            &self.polygon_stipple,
                        );

                        d.context = self.hw_id;
                        d.point_size = self.point_size;
                        d.line_width = self.line_width;
                        d.point_smooth = self.raster_enables & SGI_GL_RASTER_POINT_SMOOTH != 0;
                        d.line_smooth = self.raster_enables & SGI_GL_RASTER_LINE_SMOOTH != 0;
                        d.polygon_smooth = self.raster_enables & SGI_GL_RASTER_POLYGON_SMOOTH != 0;
                        d.polygon_modes = Some(self.polygon_mode);
                        d.fill_by_triangle = true;
                        d.polygon_stipple_enabled = self.raster_enables & SGI_GL_RASTER_POLYGON_STIPPLE != 0;
                        d.line_stipple_enabled = self.raster_enables & SGI_GL_RASTER_LINE_STIPPLE != 0;
                        d.line_stipple_repeat = self.line_stipple_repeat;
                        d.line_stipple_pattern = self.line_stipple_pattern;

                        let result = sink.draw(&mut d);
                        self.draw_result(checked, result);
                    }
                    self.polygon.polygon_release();
                }
                // A line loop closes from its last vertex to its first, in the
                // first vertex's colour when flat (2.13.7).
                if self.in_begin != 0 && self.begin_mode == SGI_GL_LINE_LOOP && self.nheld == 2 {
                    let h1 = self.held[1];
                    let h0 = self.held[0];
                    self.submit(sink, SGI_GL_LINES, 2, Some(&h1), Some(&h0), None, None);
                }
                self.line_phase = 0;
                self.in_begin = 0;
                self.begin_mode = 0;
                self.nheld = 0;
                self.strip_odd = 0;
                self.strip_prev = 0;
                self.strip_swap = 0;
            }
            SGI_GL_OP_VERTEX2F | SGI_GL_OP_VERTEX3F | SGI_GL_OP_VERTEX4F => {
                for i in 0..a.len() {
                    pos[i] = gl_float(a[i]);
                }
                self.vertex(sink, &pos, a.len() as u32);
            }
            SGI_GL_OP_VERTEX2I => {
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
    /// The glGet answer for pname as *n 32-bit words into out, which holds
    /// SGI_GL_GET_WORDS; floats as their bits, enumerants and integers as
    /// themselves, a matrix column-major. False for a pname the context does
    /// not keep.
    ///
    /// Texture coordinates are not kept, so GL_CURRENT_TEXTURE_COORDS and
    /// GL_CURRENT_RASTER_TEXTURE_COORDS answer OpenGL's initial (0, 0, 0, 1)
    /// (2.7, 2.12), wrong once a client has sent glTexCoord.
    pub fn get(&self, pname: u32, out: &mut [u32], n: &mut u32) -> bool {
        if self.in_begin != 0
            && (pname == SGI_GL_LINE_STIPPLE
                || pname == SGI_GL_LINE_STIPPLE_REPEAT
                || pname == SGI_GL_LINE_STIPPLE_PATTERN)
        {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: line stipple query inside glBegin: invalid operation");
            return false;
        }
        match pname {
            SGI_GL_CURRENT_COLOR => {
                for i in 0..4 {
                    out[i] = gl_bits(self.cur_color[i]);
                }
                *n = 4;
                true
            }
            // Column-major, the order glLoadMatrixf takes (6.1.2).
            // The indices as floats, the form the context keeps them in.
            // glGetBooleanv of a capability is glIsEnabled (6.1.1).
            // The matrices a stack holds, the current one included (6.1.2).
            _ => false,
        }
    }
}

/// How many vertices a primitive of this mode holds after count vertices
/// since glBegin, or -1 when k cannot be that number. Only count mod 2,
/// 3 and 4 and k itself decide it, never count's absolute size, so a
/// count that is right only modulo 12 is enough: once a mode has all it
/// will ever hold, k alone says so.
pub fn gl_resume_held_ok(mode: u32, k: u32, count: u32) -> bool {
    match mode {
        SGI_GL_POINTS => k == 0,
        SGI_GL_LINES => k == count % 2,
        SGI_GL_TRIANGLES => k == count % 3,
        SGI_GL_QUADS => k == count % 4,
        SGI_GL_QUAD_STRIP => {
            if k >= 2 {
                k == 2 + count % 2
            } else {
                k == count
            }
        }
        SGI_GL_LINE_STRIP => k <= 1 && (k == 1 || count == 0),
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
    pub fn resume_primitive(&mut self, mode: u32, v: Option<&[[f32; SGI_GL_VERTEX_FLOATS]]>, k: u32, count: u32) -> bool {
        if self.in_begin != 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: resume inside glBegin ignored");
            return false;
        }
        if mode > SGI_GL_POLYGON
            || k > SGI_GL_HELD_VERTICES
            || (k != 0 && v.map_or(true, |s| s.len() < k as usize))
            || !gl_resume_held_ok(mode, k, count)
            || (self.polygon.enabled != 0 && mode == SGI_GL_POLYGON)
        {
            crate::dlog!(LogModule::Odyssey, "sgi-gl: resume of mode {} with {} held of {} sent refused", mode, k, count);
            return false;
        }
        // line_phase is deliberately left alone, unlike glBegin.
        self.in_begin = 1;
        self.begin_mode = mode;
        self.nheld = k;
        for i in 0..k {
            self.hold(i, &v.unwrap()[i as usize]);
        }
        // The strip's next triangle faces as its count's parity says.
        self.strip_odd = if mode == SGI_GL_TRIANGLE_STRIP && k == 2 { count & 1 } else { 0 };
        // The oldest retained vertex of a resumed strip is not sent.
        self.strip_prev = 0;
        self.strip_swap = 0;
        true
    }

    /// Close an interrupted primitive without drawing a line-loop edge.
    pub fn discard_primitive(&mut self) {
        if self.polygon.enabled != 0 && self.begin_mode == SGI_GL_POLYGON {
            self.polygon.polygon_release();
        }
        self.in_begin = 0;
        self.begin_mode = 0;
        self.nheld = 0;
        self.strip_odd = 0;
        self.strip_prev = 0;
        self.strip_swap = 0;
    }

}

/// The faces a glMaterial, glColorMaterial or glCullFace face names, as a
/// mask of material[] entries, or 0 for a value that names none.
pub fn gl_faces(face: u32) -> u32 {
    match face {
        SGI_GL_FRONT => 1,
        SGI_GL_BACK => 2,
        SGI_GL_FRONT_AND_BACK => 3,
        _ => 0,
    }
}
