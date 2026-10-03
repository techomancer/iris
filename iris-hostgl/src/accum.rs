//! The accumulation buffer, which the host cannot give a drawable.
//!
//! Every drawable is a framebuffer object (draw.rs), and OpenGL attaches an
//! accumulation buffer only to the window system's framebuffer, never to an
//! object: glAccum on the host does nothing at all. IRIX programs use it for
//! antialiasing by jittered passes, motion blur and depth of field, and IRIS
//! GL's acbuf is the same operation, so it is kept here instead -- per
//! drawable, a pair of floating-point textures the size of the drawable.
//!
//! Each operation is one pass of a small shader over the whole buffer,
//!
//! ```text
//!     new = old * a + colour * b + c
//! ```
//!
//! where `colour` is the read buffer copied into a texture first, and a, b
//! and c are what the operation makes them (`Accum::op`). The pass writes the
//! other texture of the pair, which then becomes the buffer: reading and
//! writing one texture at once is undefined, and blending into a 32-bit float
//! target, which would avoid the pair, is not something every GPU can do.
//! GL_RETURN draws `old * value` into the drawable instead, where the target's
//! own fixed-point storage clamps it to 0..1.
//!
//! As the specification has it, every operation is limited to the scissor
//! box, GL_RETURN honours the colour mask (and nothing else of the fragment
//! stage), and the values are kept unclamped between operations. The program's
//! state is saved around each pass and put back, so nothing it can see moves.

use std::collections::HashMap;
use std::ffi::{c_void, CString};

use crate::gl::*;

#[link(name = "OpenGL", kind = "framework")]
extern "C" {
    fn glCreateShader(kind: u32) -> u32;
    fn glShaderSource(s: u32, count: i32, strings: *const *const i8, lengths: *const i32);
    fn glCompileShader(s: u32);
    fn glGetShaderiv(s: u32, pname: u32, out: *mut i32);
    fn glCreateProgram() -> u32;
    fn glAttachShader(p: u32, s: u32);
    fn glLinkProgram(p: u32);
    fn glGetProgramiv(p: u32, pname: u32, out: *mut i32);
    fn glUseProgram(p: u32);
    fn glGetUniformLocation(p: u32, name: *const i8) -> i32;
    fn glUniform1i(loc: i32, v: i32);
    fn glUniform4f(loc: i32, a: f32, b: f32, c: f32, d: f32);
    fn glGenTextures(n: i32, ids: *mut u32);
    fn glDeleteTextures(n: i32, ids: *const u32);
    fn glTexImage2D(target: u32, level: i32, internal: i32, w: i32, h: i32, border: i32, fmt: u32, ty: u32, data: *const c_void);
    fn glTexParameteri(target: u32, pname: u32, v: i32);
    fn glCopyTexSubImage2D(target: u32, level: i32, xoff: i32, yoff: i32, x: i32, y: i32, w: i32, h: i32);
    fn glPushAttrib(mask: u32);
    fn glPopAttrib();
    fn glMatrixMode(mode: u32);
    fn glPushMatrix();
    fn glPopMatrix();
    fn glLoadIdentity();
    fn glColorMask(r: u8, g: u8, b: u8, a: u8);
    fn glPolygonMode(face: u32, mode: u32);
    fn glTexCoord2f(s: f32, t: f32);
    fn glVertex2f(x: f32, y: f32);
}

pub const GL_ACCUM: u32 = 0x0100;
pub const GL_LOAD: u32 = 0x0101;
pub const GL_RETURN: u32 = 0x0102;
pub const GL_MULT: u32 = 0x0103;
pub const GL_ADD: u32 = 0x0104;
pub const GL_ACCUM_BUFFER_BIT: u32 = 0x0200;
pub const GL_ACCUM_CLEAR_VALUE: u32 = 0x0B80;
pub const GL_ACCUM_RED_BITS: u32 = 0x0D58;
pub const GL_ACCUM_ALPHA_BITS: u32 = 0x0D5B;
/// What the accumulation buffer reports per component: its textures hold
/// 32-bit floats, and 16 is what SGI's machines with one reported.
pub const ACCUM_BITS: i64 = 16;

const GL_TEXTURE_2D: u32 = 0x0DE1;
const GL_RGBA32F_ARB: i32 = 0x8814;
const GL_FLOAT: u32 = 0x1406;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;
const GL_CLAMP_TO_EDGE: i32 = 0x812F;
const GL_PROJECTION: u32 = 0x1701;
const GL_MODELVIEW: u32 = 0x1700;
const GL_TEXTURE: u32 = 0x1702;
const GL_ALL_ATTRIB_BITS: u32 = 0x000F_FFFF;
const GL_FRONT_AND_BACK: u32 = 0x0408;
const GL_FILL: u32 = 0x1B02;
const GL_QUADS: u32 = 0x0007;
const GL_CURRENT_PROGRAM: u32 = 0x8B8D;
const GL_READ_FRAMEBUFFER_BINDING: u32 = 0x8CAA;
const GL_VERTEX_SHADER: u32 = 0x8B31;
const GL_FRAGMENT_SHADER: u32 = 0x8B30;
const GL_COMPILE_STATUS: u32 = 0x8B81;
const GL_LINK_STATUS: u32 = 0x8B82;

/// Everything a pass turns off, the fragment stage's tests and the vertex
/// stage's extras; GL_SCISSOR_TEST stays as the program set it.
const OFF: [u32; 22] = [
    0x0B71, // GL_DEPTH_TEST
    0x0BE2, // GL_BLEND
    0x0BC0, // GL_ALPHA_TEST
    0x0B90, // GL_STENCIL_TEST
    0x0BF2, // GL_COLOR_LOGIC_OP
    0x0B44, // GL_CULL_FACE
    0x0B50, // GL_LIGHTING
    0x0B60, // GL_FOG
    0x0DE0, // GL_TEXTURE_1D
    0x0DE1, // GL_TEXTURE_2D
    0x806F, // GL_TEXTURE_3D
    0x3000, 0x3001, 0x3002, 0x3003, 0x3004, 0x3005, // GL_CLIP_PLANE0..5
    0x0B42, // GL_POLYGON_STIPPLE
    0x8037, // GL_POLYGON_OFFSET_FILL
    0x0C60, 0x0C61, // GL_TEXTURE_GEN_S, _T
    0x8513, // GL_TEXTURE_CUBE_MAP
];

const VERTEX: &str = "varying vec2 uv;
void main() { gl_Position = gl_Vertex; uv = gl_MultiTexCoord0.xy; }
";

const FRAGMENT: &str = "uniform sampler2D acc;
uniform sampler2D src;
uniform vec4 a;
uniform vec4 b;
uniform vec4 c;
varying vec2 uv;
void main() { gl_FragColor = texture2D(acc, uv) * a + texture2D(src, uv) * b + c; }
";

/// One drawable's accumulation buffer.
struct Buffer {
    w: i32,
    h: i32,
    /// The pair: `tex[cur]` is the buffer, the other is the next pass's target.
    tex: [u32; 2],
    fbo: [u32; 2],
    cur: usize,
    /// The read buffer's pixels, copied for GL_ACCUM and GL_LOAD to sample.
    colour: u32,
}

/// A context's accumulation buffers, by drawable, and the clear value
/// (glClearAccum), which is context state.
#[derive(Default)]
pub struct Accum {
    buffers: HashMap<u32, Buffer>,
    clear: [f32; 4],
    program: u32,
    /// The shader would not build: glAccum does nothing, as it did before.
    broken: bool,
}

impl Accum {
    pub fn clear_value(&self) -> [f32; 4] {
        self.clear
    }

    /// glClearAccum: clamped to -1..1, as the specification clamps it.
    pub fn set_clear_value(&mut self, v: [f32; 4]) {
        self.clear = v.map(|c| c.clamp(-1.0, 1.0));
    }

    /// Forget the buffers of drawables that are gone.
    pub fn retain(&mut self, alive: impl Fn(u32) -> bool) {
        let gone: Vec<u32> = self.buffers.keys().copied().filter(|id| !alive(*id)).collect();
        for id in gone {
            self.free(id);
        }
    }

    fn free(&mut self, id: u32) {
        if let Some(b) = self.buffers.remove(&id) {
            // SAFETY: objects of the current context's share group.
            unsafe {
                glDeleteFramebuffersEXT(2, b.fbo.as_ptr());
                glDeleteTextures(2, b.tex.as_ptr());
                glDeleteTextures(1, &b.colour);
            }
        }
    }

    /// glClear's GL_ACCUM_BUFFER_BIT, on drawable `draw` (`w` x `h`).
    pub fn clear(&mut self, draw: u32, w: i32, h: i32) {
        let c = self.clear;
        self.pass(draw, w, h, Pass::Buffer { a: 0.0, b: 0.0, c, colour: false });
    }

    /// glAccum(op, value) on drawable `draw`. False for an operation that is
    /// not one, which the caller turns into GL_INVALID_ENUM.
    pub fn op(&mut self, draw: u32, w: i32, h: i32, op: u32, value: f32) -> bool {
        let pass = match op {
            GL_ACCUM => Pass::Buffer { a: 1.0, b: value, c: [0.0; 4], colour: true },
            GL_LOAD => Pass::Buffer { a: 0.0, b: value, c: [0.0; 4], colour: true },
            GL_ADD => Pass::Buffer { a: 1.0, b: 0.0, c: [value; 4], colour: false },
            GL_MULT => Pass::Buffer { a: value, b: 0.0, c: [0.0; 4], colour: false },
            GL_RETURN => Pass::Return { value },
            _ => return false,
        };
        self.pass(draw, w, h, pass);
        true
    }

    fn pass(&mut self, draw: u32, w: i32, h: i32, pass: Pass) {
        if w <= 0 || h <= 0 || self.broken {
            return;
        }
        // SAFETY: GL calls on the current context, whose state is saved first
        // and put back after; the objects are its share group's.
        unsafe {
            if self.program == 0 {
                self.program = build();
                if self.program == 0 {
                    self.broken = true;
                    return;
                }
            }
            let saved = Saved::save();
            let fresh = self.ensure(draw, w, h);
            for cap in OFF {
                glDisable(cap);
            }
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glViewport(0, 0, w, h);
            glUseProgram(self.program);
            let p = self.program;
            let loc = |n: &str| {
                let n = CString::new(n).unwrap();
                glGetUniformLocation(p, n.as_ptr())
            };
            glUniform1i(loc("acc"), 0);
            glUniform1i(loc("src"), 1);
            let b = self.buffers.get_mut(&draw).unwrap();
            glActiveTexture(GL_TEXTURE0_ARB);
            glBindTexture(GL_TEXTURE_2D, b.tex[b.cur]);
            match pass {
                Pass::Buffer { a, b: k, c, colour } => {
                    // A buffer just made holds nothing yet: what it is read
                    // as is zero, whatever the pass makes of it.
                    let a = if fresh { 0.0 } else { a };
                    if colour {
                        glActiveTexture(GL_TEXTURE0_ARB + 1);
                        glBindTexture(GL_TEXTURE_2D, b.colour);
                        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, w, h);
                        glActiveTexture(GL_TEXTURE0_ARB);
                    }
                    glUniform4f(loc("a"), a, a, a, a);
                    glUniform4f(loc("b"), k, k, k, k);
                    glUniform4f(loc("c"), c[0], c[1], c[2], c[3]);
                    let next = 1 - b.cur;
                    glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, b.fbo[next]);
                    glColorMask(1, 1, 1, 1);
                    quad();
                    b.cur = next;
                }
                Pass::Return { value } => {
                    let v = if fresh { 0.0 } else { value };
                    glUniform4f(loc("a"), v, v, v, v);
                    glUniform4f(loc("b"), 0.0, 0.0, 0.0, 0.0);
                    glUniform4f(loc("c"), 0.0, 0.0, 0.0, 0.0);
                    // Into the drawable as it is bound, through the program's
                    // own draw buffer and colour mask.
                    glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, saved.draw_fbo);
                    quad();
                }
            }
            saved.restore();
        }
    }

    /// Make or resize `draw`'s buffer. True when it was just made (or made
    /// again for a new size): its contents are then undefined, which reads
    /// here as zero.
    unsafe fn ensure(&mut self, draw: u32, w: i32, h: i32) -> bool {
        if let Some(b) = self.buffers.get(&draw) {
            if b.w == w && b.h == h {
                return false;
            }
        }
        self.free(draw);
        let mut b = Buffer { w, h, tex: [0; 2], fbo: [0; 2], cur: 0, colour: 0 };
        let mut was_draw = 0i32;
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &mut was_draw);
        glGenTextures(2, b.tex.as_mut_ptr());
        glGenFramebuffersEXT(2, b.fbo.as_mut_ptr());
        for i in 0..2 {
            texture(b.tex[i], GL_RGBA32F_ARB, w, h, GL_FLOAT);
            glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, b.fbo[i]);
            glFramebufferTexture2DEXT(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, b.tex[i], 0);
        }
        glGenTextures(1, &mut b.colour);
        texture(b.colour, GL_RGBA8 as i32, w, h, GL_UNSIGNED_BYTE);
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, was_draw as u32);
        self.buffers.insert(draw, b);
        true
    }
}

enum Pass {
    /// Into the accumulation buffer: old * a + colour * b + c.
    Buffer { a: f32, b: f32, c: [f32; 4], colour: bool },
    /// Out to the drawable: old * value.
    Return { value: f32 },
}

/// A texture of `w` x `h` in `internal`, sampled texel for texel. The
/// binding on unit 0 is the caller's to restore.
unsafe fn texture(t: u32, internal: i32, w: i32, h: i32, ty: u32) {
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, GL_RGBA, ty, std::ptr::null());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST as i32);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST as i32);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

/// The whole viewport, texel centres on pixel centres.
unsafe fn quad() {
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0);
    glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0);
    glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0);
    glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0);
    glVertex2f(-1.0, 1.0);
    glEnd();
}

/// The pass's shader program, or 0 if the host would not build it.
unsafe fn build() -> u32 {
    let stage = |kind: u32, src: &str| -> u32 {
        let s = glCreateShader(kind);
        let text = CString::new(src).unwrap();
        let p = text.as_ptr();
        glShaderSource(s, 1, &p, std::ptr::null());
        glCompileShader(s);
        let mut ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &mut ok);
        if ok == 0 { 0 } else { s }
    };
    let (v, f) = (stage(GL_VERTEX_SHADER, VERTEX), stage(GL_FRAGMENT_SHADER, FRAGMENT));
    if v == 0 || f == 0 {
        return 0;
    }
    let p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    let mut ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &mut ok);
    if ok == 0 { 0 } else { p }
}

/// What a pass changes, as it was.
struct Saved {
    program: i32,
    active: i32,
    draw_fbo: u32,
    read_fbo: u32,
}

impl Saved {
    unsafe fn save() -> Saved {
        let (mut program, mut active, mut draw, mut read) = (0, 0, 0, 0);
        glGetIntegerv(GL_CURRENT_PROGRAM, &mut program);
        glGetIntegerv(GL_ACTIVE_TEXTURE_ARB, &mut active);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &mut draw);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &mut read);
        // Every attribute group, the bindings of every texture unit among
        // them; the matrices separately, unit 0's texture matrix included.
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glActiveTexture(GL_TEXTURE0_ARB);
        for m in [GL_PROJECTION, GL_MODELVIEW, GL_TEXTURE] {
            glMatrixMode(m);
            glPushMatrix();
            glLoadIdentity();
        }
        Saved { program, active, draw_fbo: draw as u32, read_fbo: read as u32 }
    }

    unsafe fn restore(self) {
        glActiveTexture(GL_TEXTURE0_ARB);
        for m in [GL_TEXTURE, GL_MODELVIEW, GL_PROJECTION] {
            glMatrixMode(m);
            glPopMatrix();
        }
        glPopAttrib();
        glUseProgram(self.program as u32);
        glActiveTexture(self.active as u32);
        glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER, self.draw_fbo);
        glBindFramebufferEXT(GL_READ_FRAMEBUFFER, self.read_fbo);
    }
}
