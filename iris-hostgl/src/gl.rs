//! The OpenGL entry points and enumerants the service calls by name.
//!
//! Only GL here -- nothing platform-specific (that is `backend`). The entry
//! points a guest's command names are not declared at all: they are looked up
//! by name at run time (`Exec::func`) and called through the signature
//! generated from IRIX's gl.h (calls.rs).

#![allow(non_snake_case, dead_code)]

use std::ffi::c_void;

#[cfg_attr(target_os = "macos", link(name = "OpenGL", kind = "framework"))]
extern "C" {
    pub fn glFlush();
    pub fn glFinish();
    pub fn glClear(mask: u32);
    pub fn glGetString(name: u32) -> *const u8;
    pub fn glGetError() -> u32;
    pub fn glGetIntegerv(pname: u32, out: *mut i32);
    pub fn glIsEnabled(cap: u32) -> u8;
    pub fn glEnable(cap: u32);
    pub fn glDisable(cap: u32);
    pub fn glViewport(x: i32, y: i32, w: i32, h: i32);
    pub fn glPixelStorei(pname: u32, v: i32);
    pub fn glPixelStoref(pname: u32, v: f32);
    pub fn glReadBuffer(mode: u32);
    pub fn glDrawBuffer(mode: u32);
    pub fn glReadPixels(x: i32, y: i32, w: i32, h: i32, fmt: u32, ty: u32, data: *mut c_void);
    pub fn glPolygonOffset(factor: f32, units: f32);
    pub fn glTexEnvi(target: u32, pname: u32, v: i32);
    pub fn glActiveTexture(texture: u32);
    pub fn glClientActiveTexture(texture: u32);
    pub fn glMultiTexCoord2f(target: u32, s: f32, t: f32);
    pub fn glBindTexture(target: u32, texture: u32);
    pub fn glBegin(mode: u32);
    pub fn glEnd();
    pub fn glEnableClientState(cap: u32);
    pub fn glDisableClientState(cap: u32);
    pub fn glVertexPointer(size: i32, ty: u32, stride: i32, p: *const c_void);
    pub fn glNormalPointer(ty: u32, stride: i32, p: *const c_void);
    pub fn glColorPointer(size: i32, ty: u32, stride: i32, p: *const c_void);
    pub fn glIndexPointer(ty: u32, stride: i32, p: *const c_void);
    pub fn glTexCoordPointer(size: i32, ty: u32, stride: i32, p: *const c_void);
    pub fn glEdgeFlagPointer(stride: i32, p: *const c_void);
    pub fn glDrawArrays(mode: u32, first: i32, count: i32);
    pub fn glDrawElements(mode: u32, count: i32, ty: u32, indices: *const c_void);
    pub fn glFeedbackBuffer(size: i32, ty: u32, buffer: *mut f32);
    pub fn glSelectBuffer(size: i32, buffer: *mut u32);
    pub fn glRenderMode(mode: u32) -> i32;
    pub fn glGetTexLevelParameteriv(target: u32, level: i32, pname: u32, out: *mut i32);
    pub fn glGetColorTableParameteriv(target: u32, pname: u32, out: *mut i32);
    pub fn glGetConvolutionParameteriv(target: u32, pname: u32, out: *mut i32);
    pub fn glGetHistogramParameteriv(target: u32, pname: u32, out: *mut i32);
    pub fn glGetMapiv(target: u32, query: u32, out: *mut i32);

    // EXT_framebuffer_object, EXT_framebuffer_blit, EXT_framebuffer_multisample.
    pub fn glGenFramebuffersEXT(n: i32, ids: *mut u32);
    pub fn glDeleteFramebuffersEXT(n: i32, ids: *const u32);
    pub fn glBindFramebufferEXT(target: u32, id: u32);
    pub fn glGenRenderbuffersEXT(n: i32, ids: *mut u32);
    pub fn glDeleteRenderbuffersEXT(n: i32, ids: *const u32);
    pub fn glBindRenderbufferEXT(target: u32, id: u32);
    pub fn glRenderbufferStorageEXT(target: u32, fmt: u32, w: i32, h: i32);
    pub fn glRenderbufferStorageMultisampleEXT(target: u32, samples: i32, fmt: u32, w: i32, h: i32);
    pub fn glFramebufferRenderbufferEXT(target: u32, att: u32, rbt: u32, rb: u32);
    pub fn glFramebufferTexture2DEXT(target: u32, attach: u32, textarget: u32, tex: u32, level: i32);
    pub fn glCheckFramebufferStatusEXT(target: u32) -> u32;
    pub fn glBlitFramebufferEXT(sx0: i32, sy0: i32, sx1: i32, sy1: i32, dx0: i32, dy0: i32, dx1: i32, dy1: i32, mask: u32, filter: u32);
}

pub const GL_FRAMEBUFFER: u32 = 0x8D40;
pub const GL_RENDERBUFFER: u32 = 0x8D41;
pub const GL_COLOR_ATTACHMENT0: u32 = 0x8CE0;
pub const GL_DEPTH_ATTACHMENT: u32 = 0x8D00;
pub const GL_STENCIL_ATTACHMENT: u32 = 0x8D20;
pub const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8CD5;
pub const GL_RGBA8: u32 = 0x8058;
pub const GL_DEPTH24_STENCIL8: u32 = 0x88F0;
pub const GL_RGBA: u32 = 0x1908;
pub const GL_BGRA: u32 = 0x80E1;
pub const GL_UNSIGNED_BYTE: u32 = 0x1401;
pub const GL_READ_FRAMEBUFFER: u32 = 0x8CA8;
pub const GL_DRAW_FRAMEBUFFER: u32 = 0x8CA9;
pub const GL_DRAW_FRAMEBUFFER_BINDING: u32 = 0x8CA6;
pub const GL_COLOR_BUFFER_BIT: u32 = 0x4000;
pub const GL_NEAREST: u32 = 0x2600;
pub const GL_SCISSOR_TEST: u32 = 0x0C11;
pub const GL_MULTISAMPLE: u32 = 0x809D;
pub const GL_UNPACK_SWAP_BYTES: u32 = 0x0CF0;
pub const GL_UNPACK_LSB_FIRST: u32 = 0x0CF1;
pub const GL_UNPACK_ROW_LENGTH: u32 = 0x0CF2;
pub const GL_UNPACK_SKIP_ROWS: u32 = 0x0CF3;
pub const GL_UNPACK_SKIP_PIXELS: u32 = 0x0CF4;
pub const GL_UNPACK_ALIGNMENT: u32 = 0x0CF5;
pub const GL_PACK_SWAP_BYTES: u32 = 0x0D00;
pub const GL_PACK_LSB_FIRST: u32 = 0x0D01;
pub const GL_PACK_ROW_LENGTH: u32 = 0x0D02;
pub const GL_PACK_SKIP_ROWS: u32 = 0x0D03;
pub const GL_PACK_SKIP_PIXELS: u32 = 0x0D04;
pub const GL_PACK_ALIGNMENT: u32 = 0x0D05;
pub const GL_PACK_SKIP_IMAGES: u32 = 0x806B;
pub const GL_PACK_IMAGE_HEIGHT: u32 = 0x806C;
pub const GL_UNPACK_SKIP_IMAGES: u32 = 0x806D;
pub const GL_UNPACK_IMAGE_HEIGHT: u32 = 0x806E;
pub const GL_FEEDBACK: u32 = 0x1C01;
pub const GL_SELECT: u32 = 0x1C02;
pub const GL_BITMAP: u32 = 0x1A00;
pub const GL_VENDOR: u32 = 0x1F00;
pub const GL_RENDERER: u32 = 0x1F01;
pub const GL_VERSION: u32 = 0x1F02;
pub const GL_EXTENSIONS: u32 = 0x1F03;

// ARB_multitexture. The unit enumerants are consecutive from GL_TEXTURE0, and
// the ARB names have the same values as the core ones.
pub const GL_TEXTURE0_ARB: u32 = 0x84C0;
pub const GL_TEXTURE31_ARB: u32 = 0x84DF;
pub const GL_ACTIVE_TEXTURE_ARB: u32 = 0x84E0;
pub const GL_CLIENT_ACTIVE_TEXTURE_ARB: u32 = 0x84E1;
pub const GL_MAX_TEXTURE_UNITS_ARB: u32 = 0x84E2;
/*
 * SGIS_multitexture's unit enumerants, from the extension's own numbers --
 * this IRIX image's gl.h does not declare the extension at all, so a program
 * using it (Quake 2's SGIS path) carries its own definitions, and these are
 * the values such programs use.
 */
pub const GL_TEXTURE0_SGIS: u32 = 0x835E;
pub const GL_TEXTURE1_SGIS: u32 = 0x835F;
