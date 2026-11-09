//! The GL client's state as the pipe keeps it for one rendering context:
//! matrix stacks, the viewport, scissor and depth range, the lights and
//! material, the current colour, normal and texture coordinate, and the few
//! switches the vertex path reads.
//!
//! The pipe, not the library, keeps the matrices and lights the vertices
//! (software expectation from the senders: the client only writes its
//! arguments). Word layouts are the decoder's in QEMU's public KONA model
//! (jamesbraid/qemu-sgi-origami, `hw/misc/sgi-kona.c`); enumerant values are
//! OpenGL's. A float argument is a 32-bit word holding the float's bits. Colour
//! tables are not taken (counted).

use crate::dev::kona::back::raster::fog::Fog;
use crate::dev::kona::back::raster::ops::{BlendEq, BlendFactor, Compare, LogicOp, Ownership, StencilOp};
use crate::dev::kona::back::raster::texel::{EnvMode, TexDesc};
use crate::dev::kona::back::raster::{AlphaTest, BlendState, DrawState, Rect, Smooth, StencilState};
use crate::dev::kona::defs::{gl, *};
use crate::dev::kona::stream::Stream;
use crate::dev::gl::light::{Lighting, MAX_LIGHTS};
use crate::dev::gl::math::{self, Mat4, Stack};
use crate::dev::gl::texgen::{self, TexGen};

/// Matrix stack depths: OpenGL's minimums for the view stacks and texture.
pub const MV_DEPTH: usize = 32;
pub const PROJ_DEPTH: usize = 4;
pub const TEX_DEPTH: usize = 4;

pub fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

/// Where a draw lands: the window's rectangle in plane pixels, the colour
/// words of the buffers drawn to and the depth word. From the buffer
/// configuration (the layout), the window the kernel placed and glDrawBuffer.
/// A draw with no colour word (no layout yet, an overlay, draw buffer none)
/// writes no colour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Target {
    pub colour: [u32; 2],
    pub colours: u8,
    /// The depth word and the bits of depth in it; 0 bits: no depth plane.
    pub depth_word: u32,
    pub depth_bits: u32,
    /// The stencil word and its bits (8, or 0: none).
    pub stencil_word: u32,
    pub stencil_bits: u32,
}

/// The fog parameters the pipe keeps: mode, density, start, end and colour, with the enable (the
/// generic state query lists a row for each, and for the enable). The fog index is the colour-index
/// mode's and is not kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FogState {
    pub on: bool,
    /// `gl::LINEAR`, `gl::EXP` or `gl::EXP2`.
    pub mode: u32,
    pub density: f32,
    pub start: f32,
    pub end: f32,
    pub colour: [f32; 3],
}

impl Default for FogState {
    /// OpenGL's initial values (the table lists no start-up contents for the rows).
    fn default() -> Self {
        FogState { on: false, mode: gl::EXP, density: 1.0, start: 0.0, end: 1.0, colour: [0.0; 3] }
    }
}

/// The switches a draw reads, grouped so a snapshot changes only when one does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Raster {
    pub viewport: [f32; 4],
    pub depth_range: [f32; 2],
    pub scissor: [i32; 4],
    pub scissor_test: bool,
    pub depth_test: bool,
    pub depth_func: u32,
    pub cull: bool,
    pub cull_mode: u32,
    pub front_cw: bool,
    pub flat: bool,
    pub texture_2d: bool,
    pub texture_3d: bool,
    pub lighting: bool,
    pub target: Target,
    // The fragment operations. The tokens' argument layouts are GL's argument orders
    // (inferred: no recording sends one of them, but the blend enable).
    pub alpha_test: bool,
    pub alpha_func: u32,
    pub alpha_ref: f32,
    pub stencil_test: bool,
    pub stencil_func: u32,
    pub stencil_ref: u32,
    pub stencil_read_mask: u32,
    pub stencil_write_mask: u32,
    pub stencil_ops: [u32; 3],
    pub blend: bool,
    pub blend_factors: [u32; 2],
    pub blend_colour: [f32; 4],
    pub logic_op_on: bool,
    pub logic_op: u32,
    pub colour_mask: [bool; 4],
    pub depth_mask: bool,
    pub dither: bool,
    pub offset_fill: bool,
    pub offset: [f32; 2],
    pub line_width: f32,
    pub point_size: f32,
    pub line_stipple_on: bool,
    /// The line stipple's repeat count and 16-bit pattern.
    pub line_stipple: (u32, u16),
    pub polygon_stipple_on: bool,
    pub polygon_stipple: [u32; 32],
    /// The texture environment mode number and colour.
    pub env_mode: u32,
    pub env_colour: [f32; 4],
    pub fog: FogState,
    /// How the front and the back faces of a polygon are drawn: `gl::POINT`, `gl::LINE` or `gl::FILL`.
    pub polygon_mode: [u32; 2],
    /// The smoothing enables (see `raster::smooth`).
    pub point_smooth: bool,
    pub line_smooth: bool,
    pub polygon_smooth: bool,
    pub ms: crate::dev::kona::back::raster::multisample::Multisample,
}

impl Default for Raster {
    fn default() -> Self {
        Raster {
            viewport: [0.0; 4],
            depth_range: [0.0, 1.0],
            scissor: [0; 4],
            scissor_test: false,
            depth_test: false,
            depth_func: gl::LESS,
            cull: false,
            cull_mode: gl::BACK,
            front_cw: false,
            flat: false,
            texture_2d: false,
            texture_3d: false,
            lighting: false,
            target: Target::default(),
            alpha_test: false,
            alpha_func: gl::ALWAYS,
            alpha_ref: 0.0,
            stencil_test: false,
            stencil_func: gl::ALWAYS,
            stencil_ref: 0,
            stencil_read_mask: !0,
            stencil_write_mask: !0,
            stencil_ops: [gl::KEEP; 3],
            blend: false,
            blend_factors: [1, 0],
            blend_colour: [0.0; 4],
            logic_op_on: false,
            logic_op: gl::LOGIC_OP_CLEAR + 3,
            colour_mask: [true; 4],
            depth_mask: true,
            dither: true,
            offset_fill: false,
            offset: [0.0; 2],
            line_width: 1.0,
            point_size: 1.0,
            line_stipple_on: false,
            line_stipple: (1, 0xffff),
            polygon_stipple_on: false,
            polygon_stipple: [!0; 32],
            env_mode: 0x2100,
            fog: FogState::default(),
            polygon_mode: [gl::FILL; 2],
            point_smooth: false,
            line_smooth: false,
            polygon_smooth: false,
            ms: Default::default(),
            env_colour: [0.0; 4],
        }
    }
}

/// Plain data: a context copy is a `clone`.
#[derive(Clone)]
pub struct Gl {
    pub mode: u32,
    pub mv: Stack<MV_DEPTH>,
    pub proj: Stack<PROJ_DEPTH>,
    pub tex: Stack<TEX_DEPTH>,
    pub light: Lighting,
    /// The normal matrix follows the model-view matrix; recomputed when used
    /// after a change.
    pub normal_stale: bool,
    pub color: [f32; 4],
    pub normal: [f32; 3],
    pub texcoord: [f32; 4],
    pub edge_flag: bool,
    /// The user clip planes in eye space, and which are enabled (bit n is plane n).
    pub user_plane: [[f32; 4]; gl::CLIP_PLANES],
    pub user_on: u32,
    /// Texture coordinate generation.
    pub texgen: TexGen,
    pub raster: Raster,
    pub in_begin: bool,
    /// The begin token of the open primitive.
    pub prim: u32,
    pub error: u32,
    /// Enabled lights, as bits.
    pub light_bits: u32,
    /// The raster group changed since the last snapshot.
    pub dirty: bool,
    /// The buffer configuration's layout (level 0), the window the kernel placed
    /// (its origin and clip rectangle in plane pixels), glDrawBuffer's mask (1 front, 2
    /// back; start-up: the back buffer of a double-buffered window, else the front),
    /// the double-buffer flag and the clear values.
    pub layout: Option<crate::dev::kona::back::store::layout::Layout>,
    pub double: bool,
    pub origin: [i32; 2],
    pub clip: Rect,
    pub placed: bool,
    /// The window ID the kernel placed, as the signed word, and whether its test is on.
    pub window_id: i32,
    pub id_test: bool,
    /// The display id the kernel gave the window, and which colour buffer the
    /// display shows (0 the first word, 1 the second): drawing goes to the other
    /// one, and a swap flips them.
    pub display_id: u32,
    pub shown: u8,
    pub draw: Option<u32>,
    pub clear_colour: [f32; 4],
    pub clear_depth: f32,
    pub clear_stencil: u32,
    /// The bound texture, set by a bind and read when the draw state is built.
    pub texture: Option<TexDesc>,
    /// Pixel-store, transfer and zoom state, the raster position, maps and the accumulation clear value.
    pub pix: super::pixels::Pixels,
    /// The render mode and what a select or feedback pass has collected.
    pub feed: super::query::Feed,
    /// How many attribute-stack entries a push has made and a pop has not taken back.
    pub attrib_depth: u32,
    /// The draw state built from the above, kept until a state command changes it.
    ds: Option<DrawState>,
}

impl std::fmt::Debug for Gl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gl").field("mode", &self.mode).field("raster", &self.raster).field("in_begin", &self.in_begin).finish_non_exhaustive()
    }
}

impl Default for Gl {
    fn default() -> Self {
        let mut g = Gl {
            mode: gl::MODELVIEW,
            mv: Stack { m: [math::IDENT; MV_DEPTH], top: 0 },
            proj: Stack { m: [math::IDENT; PROJ_DEPTH], top: 0 },
            tex: Stack { m: [math::IDENT; TEX_DEPTH], top: 0 },
            light: Lighting::default(),
            normal_stale: false,
            color: [1.0; 4],
            normal: [0.0, 0.0, 1.0],
            texcoord: [0.0, 0.0, 0.0, 1.0],
            edge_flag: true,
            user_plane: [[0.0; 4]; gl::CLIP_PLANES],
            user_on: 0,
            texgen: TexGen::default(),
            raster: Raster::default(),
            in_begin: false,
            prim: BEGIN_PRIMITIVE_POINTS,
            error: 0,
            light_bits: 0,
            dirty: true,
            layout: None,
            double: false,
            origin: [0; 2],
            clip: Rect::default(),
            placed: false,
            window_id: 0,
            id_test: false,
            display_id: u32::MAX,
            shown: 0,
            draw: None,
            clear_colour: [0.0; 4],
            clear_depth: 1.0,
            clear_stencil: 0,
            texture: None,
            pix: Default::default(),
            feed: Default::default(),
            attrib_depth: 0,
            ds: None,
        };
        g.light.init();
        g
    }
}

/// What a state command did that the caller counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Taken {
    Yes,
    /// A form or enumerant the model does not carry out.
    Unmodelled,
    /// Illegal here (a matrix command between begin and end, a stack error).
    Error(u32),
}


/// A matrix stack, as the stream module reads and writes it.
fn visit_stack<S: Stream, const N: usize>(s: &mut S, st: &mut Stack<N>) {
    for m in st.m.iter_mut() {
        s.f32s(m);
    }
    s.u32(&mut st.top);
}

fn visit_lighting<S: Stream>(s: &mut S, l: &mut Lighting) {
    for v in [&mut l.on, &mut l.two_sided, &mut l.local_viewer, &mut l.ambient_folded, &mut l.count, &mut l.cmat_param, &mut l.cmat_face, &mut l.cmat_on, &mut l.normalize, &mut l.fog_on, &mut l.fog_mode] {
        s.u32(v);
    }
    s.f32s(&mut l.amb_sum);
    for m in l.mat.iter_mut() {
        s.f32s(&mut m.emission);
        s.f32s(&mut m.ambient);
        s.f32s(&mut m.diffuse);
        s.f32s(&mut m.specular);
        s.f32(&mut m.shininess);
        s.u32(&mut m.spec_table);
        s.f32(&mut m.spec_scale);
        s.f32(&mut m.spec_start);
        s.f32s(&mut m.spec);
    }
    for g in l.lights.iter_mut() {
        s.f32s(&mut g.ambient);
        s.f32s(&mut g.diffuse);
        s.f32s(&mut g.specular);
        s.f32s(&mut g.pos);
        s.f32s(&mut g.spot_dir);
        s.f32s(&mut g.atten);
        s.u32(&mut g.spot_on);
        s.f32(&mut g.spot_exp);
        s.f32(&mut g.spot_cos_cutoff);
        s.u32(&mut g.spot_table);
        s.f32(&mut g.spot_scale);
        s.f32(&mut g.spot_start);
        s.f32s(&mut g.spot);
        s.i32(&mut g.next);
    }
    s.i32(&mut l.head);
    s.f32s(&mut l.normal_matrix);
    s.f32(&mut l.fog_a);
    s.f32(&mut l.fog_b);
    s.f32s(&mut l.fog_color);
}

impl Raster {
    /// The switches, as the stream module reads and writes them. The target is made again
    /// from the layout and the window when the draw state is built.
    fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { viewport: _, depth_range: _, scissor: _, scissor_test: _, depth_test: _, depth_func: _, cull: _, cull_mode: _, front_cw: _, flat: _, texture_2d: _, texture_3d: _, lighting: _, target: _, alpha_test: _, alpha_func: _, alpha_ref: _, stencil_test: _, stencil_func: _, stencil_ref: _, stencil_read_mask: _, stencil_write_mask: _, stencil_ops: _, blend: _, blend_factors: _, blend_colour: _, logic_op_on: _, logic_op: _, colour_mask: _, depth_mask: _, dither: _, offset_fill: _, offset: _, line_width: _, point_size: _, line_stipple_on: _, line_stipple: _, polygon_stipple_on: _, polygon_stipple: _, env_mode: _, env_colour: _, fog: _, polygon_mode: _, point_smooth: _, line_smooth: _, polygon_smooth: _, ms: _ } = self;
        s.f32s(&mut self.viewport);
        s.f32s(&mut self.depth_range);
        s.i32s(&mut self.scissor);
        for b in [&mut self.scissor_test, &mut self.depth_test, &mut self.cull, &mut self.front_cw, &mut self.flat, &mut self.texture_2d, &mut self.texture_3d, &mut self.lighting, &mut self.alpha_test, &mut self.stencil_test, &mut self.blend, &mut self.logic_op_on, &mut self.depth_mask, &mut self.dither, &mut self.offset_fill, &mut self.line_stipple_on, &mut self.polygon_stipple_on, &mut self.point_smooth, &mut self.line_smooth, &mut self.polygon_smooth] {
            s.bool(b);
        }
        for v in [&mut self.depth_func, &mut self.cull_mode, &mut self.alpha_func, &mut self.stencil_func, &mut self.stencil_ref, &mut self.stencil_read_mask, &mut self.stencil_write_mask, &mut self.logic_op, &mut self.env_mode] {
            s.u32(v);
        }
        s.u32s(&mut self.stencil_ops);
        s.u32s(&mut self.blend_factors);
        s.f32(&mut self.alpha_ref);
        s.f32s(&mut self.blend_colour);
        for b in self.colour_mask.iter_mut() {
            s.bool(b);
        }
        s.f32s(&mut self.offset);
        s.f32(&mut self.line_width);
        s.f32(&mut self.point_size);
        s.u32(&mut self.line_stipple.0);
        s.u16(&mut self.line_stipple.1);
        s.u32s(&mut self.polygon_stipple);
        s.f32s(&mut self.env_colour);
        let fog = &mut self.fog;
        s.bool(&mut fog.on);
        s.u32(&mut fog.mode);
        for v in [&mut fog.density, &mut fog.start, &mut fog.end] {
            s.f32(v);
        }
        s.f32s(&mut fog.colour);
        s.u32s(&mut self.polygon_mode);
        self.ms.visit(s);
    }
}

impl Gl {
    /// Whether a primitive is open, which a snapshot does not carry.
    pub fn busy(&self) -> bool {
        self.in_begin
    }

    /// The context's GL state, as the stream module reads and writes it. The draw state
    /// is made again (`dirty`) on a load.
    pub fn visit<S: Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { mode: _, mv: _, proj: _, tex: _, light: _, normal_stale: _, color: _, normal: _, texcoord: _, edge_flag: _, user_plane: _, user_on: _, texgen: _, raster: _, in_begin: _, prim: _, error: _, light_bits: _, dirty: _, layout: _, double: _, origin: _, clip: _, placed: _, window_id: _, id_test: _, display_id: _, shown: _, draw: _, clear_colour: _, clear_depth: _, clear_stencil: _, texture: _, pix: _, feed: _, attrib_depth: _, ds: _ } = self;
        s.u32(&mut self.mode);
        visit_stack(s, &mut self.mv);
        visit_stack(s, &mut self.proj);
        visit_stack(s, &mut self.tex);
        visit_lighting(s, &mut self.light);
        s.bool(&mut self.normal_stale);
        s.f32s(&mut self.color);
        s.f32s(&mut self.normal);
        s.f32s(&mut self.texcoord);
        s.bool(&mut self.edge_flag);
        for p in &mut self.user_plane {
            s.f32s(p);
        }
        s.u32(&mut self.user_on);
        s.u32(&mut self.texgen.enabled);
        for i in 0..4 {
            s.u32(&mut self.texgen.mode[i]);
            s.f32s(&mut self.texgen.object[i]);
            s.f32s(&mut self.texgen.eye[i]);
        }
        self.raster.visit(s);
        s.u32(&mut self.prim);
        s.u32(&mut self.error);
        s.u32(&mut self.light_bits);
        let mut has_layout = self.layout.is_some();
        s.bool(&mut has_layout);
        if s.loading() {
            self.layout = has_layout.then(|| crate::dev::kona::back::store::layout::Layout { colour: 0..0, depth: 0..0, stencil: 0..0, multisample: 0..0, accumulation: 0..0, last: 0 });
        }
        if let Some(l) = &mut self.layout {
            for r in [&mut l.colour, &mut l.depth, &mut l.stencil, &mut l.multisample, &mut l.accumulation] {
                s.u32(&mut r.start);
                s.u32(&mut r.end);
            }
            s.u32(&mut l.last);
        }
        s.bool(&mut self.double);
        s.i32s(&mut self.origin);
        for v in [&mut self.clip.x0, &mut self.clip.y0, &mut self.clip.x1, &mut self.clip.y1] {
            s.i32(v);
        }
        s.bool(&mut self.placed);
        s.i32(&mut self.window_id);
        s.bool(&mut self.id_test);
        s.u32(&mut self.display_id);
        s.u8(&mut self.shown);
        s.opt(&mut self.draw, |s, v| s.u32(v));
        s.f32s(&mut self.clear_colour);
        s.f32(&mut self.clear_depth);
        s.u32(&mut self.clear_stencil);
        s.opt(&mut self.texture, |s, t| t.visit(s));
        self.pix.visit(s);
        self.feed.visit(s);
        s.u32(&mut self.attrib_depth);
        if s.loading() {
            self.in_begin = false;
            self.dirty = true;
            self.ds = None;
        }
    }

    fn matrix_op(&mut self, op: impl FnOnce(&mut Mat4)) {
        match self.mode {
            gl::MODELVIEW => {
                let mut m = *self.mv.get();
                op(&mut m);
                self.mv.load(m);
                self.normal_stale = true;
            }
            gl::PROJECTION => {
                let mut m = *self.proj.get();
                op(&mut m);
                self.proj.load(m);
            }
            gl::TEXTURE => {
                let mut m = *self.tex.get();
                op(&mut m);
                self.tex.load(m);
            }
            _ => {}
        }
    }

    fn push_pop(&mut self, push: bool) -> Taken {
        let ok = match (self.mode, push) {
            (gl::MODELVIEW, true) => self.mv.push(),
            (gl::MODELVIEW, false) => self.mv.pop(),
            (gl::PROJECTION, true) => self.proj.push(),
            (gl::PROJECTION, false) => self.proj.pop(),
            (gl::TEXTURE, true) => self.tex.push(),
            (gl::TEXTURE, false) => self.tex.pop(),
            _ => return Taken::Unmodelled,
        };
        if self.mode == gl::MODELVIEW {
            self.normal_stale = true;
        }
        if ok {
            Taken::Yes
        } else {
            Taken::Error(if push { gl::STACK_OVERFLOW } else { gl::STACK_UNDERFLOW })
        }
    }

    /// The matrix family: matrix mode, load, mult, push, pop, rotate, scale,
    /// translate, in float and double forms, and frustum and orthographic projection
    /// (the double forms' word order is inferred from the depth clear's). Between a begin and its end it is an error and changes nothing.
    pub fn matrix(&mut self, selector: u32, a: &[u32]) -> Taken {
        if self.in_begin {
            return Taken::Error(gl::INVALID_OPERATION);
        }
        let fl = |i: usize| f(a[i]);
        // A double is two words, the high one first (as the depth clear's is, established).
        let db = |i: usize| f64::from_bits((a[2 * i] as u64) << 32 | a[2 * i + 1] as u64) as f32;
        match (selector, a.len()) {
            (SELECT_CURRENT_MATRIX_STACK, 1) => self.mode = a[0],
            // Load identity, push and pop carry one word, which means nothing.
            (LOAD_IDENTITY_MATRIX, 1) => self.matrix_op(|m| *m = math::IDENT),
            (LOAD_MATRIX_16_FLOATS, 16) => {
                let m: Mat4 = std::array::from_fn(|i| fl(i));
                self.matrix_op(|t| *t = m);
            }
            (MULTIPLY_BY_MATRIX_16_FLOATS, 16) => {
                let m: Mat4 = std::array::from_fn(|i| fl(i));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (LOAD_MATRIX_16_DOUBLES, 32) => {
                let m: Mat4 = std::array::from_fn(db);
                self.matrix_op(|t| *t = m);
            }
            (MULTIPLY_BY_MATRIX_16_DOUBLES, 32) => {
                let m: Mat4 = std::array::from_fn(db);
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (ROTATE_DOUBLES, 8) => {
                let m = math::rotate(db(0), db(1), db(2), db(3));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (SCALE_DOUBLES, 6) => {
                let m = math::scale(db(0), db(1), db(2));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (TRANSLATE_DOUBLES, 6) => {
                let m = math::translate(db(0), db(1), db(2));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (MULTIPLY_CURRENT_MATRIX_BY_FRUSTUM, 12) => {
                let m = math::frustum(db(0), db(1), db(2), db(3), db(4), db(5));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (MULTIPLY_CURRENT_MATRIX_BY_ORTHOGRAPHIC_PROJECTION, 12) => {
                let m = math::ortho(db(0), db(1), db(2), db(3), db(4), db(5));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (POP_MATRIX_STACK, 1) => return self.push_pop(false),
            (PUSH_MATRIX_STACK, 1) => return self.push_pop(true),
            (ROTATE_FLOATS, 4) => {
                let m = math::rotate(fl(0), fl(1), fl(2), fl(3));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (SCALE_FLOATS, 3) => {
                let m = math::scale(fl(0), fl(1), fl(2));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            (TRANSLATE_FLOATS, 3) => {
                let m = math::translate(fl(0), fl(1), fl(2));
                self.matrix_op(|t| *t = math::mul(t, &m));
            }
            _ => return Taken::Unmodelled,
        }
        Taken::Yes
    }

    /// glEnable and glDisable of the capabilities the vertex path reads.
    pub fn enable(&mut self, cap: u32, on: bool) -> Taken {
        let r = &mut self.raster;
        match cap {
            gl::LIGHTING => {
                r.lighting = on;
                self.light.on = on as u32;
            }
            gl::CULL_FACE => r.cull = on,
            gl::DEPTH_TEST => r.depth_test = on,
            gl::TEXTURE_2D => r.texture_2d = on,
            gl::TEXTURE_3D => r.texture_3d = on,
            gl::SCISSOR_TEST => r.scissor_test = on,
            gl::ALPHA_TEST => r.alpha_test = on,
            gl::BLEND => r.blend = on,
            gl::STENCIL_TEST => r.stencil_test = on,
            gl::COLOR_LOGIC_OP => r.logic_op_on = on,
            gl::DITHER => r.dither = on,
            gl::POLYGON_OFFSET_FILL => r.offset_fill = on,
            gl::FOG => r.fog.on = on,
            gl::POINT_SMOOTH => r.point_smooth = on,
            gl::LINE_SMOOTH => r.line_smooth = on,
            gl::POLYGON_SMOOTH => r.polygon_smooth = on,
            gl::LINE_STIPPLE => r.line_stipple_on = on,
            gl::POLYGON_STIPPLE => r.polygon_stipple_on = on,
            gl::MULTISAMPLE => r.ms.enable = on,
            gl::SAMPLE_ALPHA_TO_MASK => r.ms.alpha_to_mask = on,
            gl::SAMPLE_ALPHA_TO_ONE => r.ms.alpha_to_one = on,
            gl::SAMPLE_MASK => r.ms.mask_on = on,
            gl::NORMALIZE => self.light.normalize = on as u32,
            gl::COLOR_MATERIAL => self.light.cmat_on = on as u32,
            // A clip plane's enable goes through the generic enable token, which takes the names 0x3000 to
            // 0x3005 (established). Not part of the draw state:
            // the vertex path reads the planes at begin.
            gl::CLIP_PLANE0..=gl::CLIP_PLANE5 => {
                let bit = 1 << (cap - gl::CLIP_PLANE0);
                self.user_on = if on { self.user_on | bit } else { self.user_on & !bit };
                return Taken::Yes;
            }
            // Each coordinate's enable goes through the generic enable token (established).
            gl::TEXTURE_GEN_S..=gl::TEXTURE_GEN_Q => {
                let bit = 1 << (cap - gl::TEXTURE_GEN_S);
                self.texgen.enabled = if on { self.texgen.enabled | bit } else { self.texgen.enabled & !bit };
                return Taken::Yes;
            }
            gl::LIGHT0..=gl::LIGHT7 => {
                let bit = 1 << (cap - gl::LIGHT0);
                if on {
                    self.light_bits |= bit;
                } else {
                    self.light_bits &= !bit;
                }
                self.rechain_lights();
                return Taken::Yes;
            }
            // The enable tokens take only the six planes in 0x3000 to 0x3fff; the rest of that range is an
            // invalid enumerant (established).
            gl::CLIP_PLANE_RANGE_FIRST..=gl::CLIP_PLANE_RANGE_LAST => return Taken::Error(gl::INVALID_ENUM),
            _ => return Taken::Unmodelled,
        }
        self.dirty = true;
        Taken::Yes
    }

    /// The chain of enabled lights, lowest first.
    fn rechain_lights(&mut self) {
        let on: Vec<usize> = (0..MAX_LIGHTS).filter(|i| self.light_bits >> i & 1 != 0).collect();
        self.light.head = on.first().map_or(-1, |&i| i as i32);
        self.light.count = on.len() as u32;
        for (k, &i) in on.iter().enumerate() {
            self.light.lights[i].next = on.get(k + 1).map_or(-1, |&n| n as i32);
        }
    }

    /// The other state commands, by selector.
    pub fn state(&mut self, selector: u32, a: &[u32]) -> Taken {
        match (selector, a.len()) {
            (SET_VIEWPORT, 4) => {
                self.raster.viewport = [f(a[0]), f(a[1]), f(a[2]), f(a[3])];
                // A guess: the viewport words are integers or floats, and a word
                // below 0x10000 is read as an integer (no float of an ordinary
                // size has that bit pattern). The recorded streams send
                // integers (0, 0, 500, 400); a capture of a client that sends
                // a fractional viewport, or the sender's own code, would settle it.
                for (v, w) in self.raster.viewport.iter_mut().zip(a) {
                    if *w < 0x0001_0000 {
                        *v = *w as f32;
                    }
                }
            }
            (SET_SCISSOR_BOX, 4) => self.raster.scissor = [a[0] as i32, a[1] as i32, a[2] as i32, a[3] as i32],
            (ENABLE_CAPABILITY, 1) => return self.enable(a[0], true),
            (DISABLE_CAPABILITY, 1) => return self.enable(a[0], false),
            (SET_FLAT_OR_SMOOTH_SHADING, 1) => self.raster.flat = a[0] == gl::FLAT,
            (SET_FACE_CULLING, 1) => self.raster.cull_mode = a[0],
            (SET_FRONT_FACE_WINDING, 1) => self.raster.front_cw = a[0] == gl::CW,
            (SET_DEPTH_TEST, 1) => self.raster.depth_func = a[0],
            (SET_ALPHA_TEST, 2) => {
                self.raster.alpha_func = a[0];
                self.raster.alpha_ref = f(a[1]);
            }
            (SET_STENCIL_TEST, 3) => {
                self.raster.stencil_func = a[0];
                self.raster.stencil_ref = a[1];
                self.raster.stencil_read_mask = a[2];
            }
            (SET_STENCIL_OPERATIONS, 3) => self.raster.stencil_ops = [a[0], a[1], a[2]],
            (SET_STENCIL_WRITE_MASK, 1) => self.raster.stencil_write_mask = a[0],
            (SET_BLEND_FACTORS, 2) => self.raster.blend_factors = [a[0], a[1]],
            (SET_CONSTANT_BLEND_COLOUR, 4) => self.raster.blend_colour = [f(a[0]), f(a[1]), f(a[2]), f(a[3])],
            (SET_LOGIC_OPERATION, 1) => self.raster.logic_op = a[0],
            (SET_COLOUR_WRITE_MASK, 4) => self.raster.colour_mask = [a[0] != 0, a[1] != 0, a[2] != 0, a[3] != 0],
            (SET_DEPTH_WRITE_MASK, 1) => self.raster.depth_mask = a[0] != 0,
            (SET_POLYGON_OFFSET, 2) => self.raster.offset = [f(a[0]), f(a[1])],
            // The polygon fill mode: two words, the face then the mode (established: the pipe reads
            // the face from the first word and the mode from the last). It validates neither: a face
            // other than front, back or both is ignored without an error, and a mode is kept as sent
            // (the client checks both). A triangle's mode then reads as line when it equals GL's line
            // mode, point when below it and fill when above (established), see `GlPath::polygon_mode`.
            // The pipe also writes a per-context flag from each call's mode alone (1 for line); nothing
            // in the model reads it, so it is not kept.
            (SET_POLYGON_FILL_MODE, 2) => {
                let faces = match a[0] {
                    gl::FRONT => [true, false],
                    gl::BACK => [false, true],
                    gl::FRONT_AND_BACK => [true; 2],
                    _ => return Taken::Yes,
                };
                for (m, _) in self.raster.polygon_mode.iter_mut().zip(faces).filter(|&(_, on)| on) {
                    *m = a[1];
                }
            }
            // The first word is the float (inferred from GL's one-float call; the second is unread).
            (SET_USER_CLIP_PLANE, 9) => return self.clip_plane(a),
            (TEXTURE_COORDINATE_GENERATION_FLOAT_FORM, 9) => return self.tex_gen(a),
            (SET_LINE_WIDTH, 2) => self.raster.line_width = f(a[0]),
            (SET_POINT_SIZE, 2) => self.raster.point_size = f(a[0]),
            // GL's argument order: the repeat count, then the pattern.
            (SET_LINE_STIPPLE, 2) => self.raster.line_stipple = (a[0].clamp(1, 256), a[1] as u16),
            (SET_STENCIL_CLEAR_VALUE, 1) => self.clear_stencil = a[0],
            // The texture environment: the parameter's name, then four values. The mode's number
            // is a float in the float form (inferred from its start-up value), the colour four floats.
            (SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, 5) if a[0] == 0x2200 => self.raster.env_mode = f(a[1]) as u32,
            (SET_TEXTURE_ENVIRONMENT_PARAMETER_INTEGER_FORM, 5) if a[0] == 0x2200 => self.raster.env_mode = a[1],
            (SET_TEXTURE_ENVIRONMENT_PARAMETER_FLOAT_FORM, 5) if a[0] == 0x2201 => self.raster.env_colour = [f(a[1]), f(a[2]), f(a[3]), f(a[4])],
            (SELECT_DRAW_BUFFER, 1) => {
                // glDrawBuffer: front gl::FRONT, back gl::BACK, front and back gl::FRONT_AND_BACK, none 0.
                self.draw = Some(match a[0] {
                    gl::FRONT | gl::FRONT_LEFT | gl::RIGHT => 1,
                    gl::BACK | gl::BACK_LEFT => 2,
                    gl::FRONT_AND_BACK => 3,
                    0 => 0,
                    _ => return Taken::Error(gl::INVALID_ENUM),
                });
            }
            (SET_MULTISAMPLE_PATTERN, 1) => {
                if !self.raster.ms.set_pattern(a[0]) {
                    return Taken::Error(gl::INVALID_ENUM);
                }
            }
            // The value (clamped to 0 to 1 by the client) and the invert flag's low byte.
            (SET_MULTISAMPLE_COVERAGE_MASK, 2) => {
                self.raster.ms.mask_value = f(a[0]).clamp(0.0, 1.0);
                self.raster.ms.mask_invert = a[1] & 0xff != 0;
            }
            (SET_COLOUR_CLEAR_VALUE, 4) => self.clear_colour = [f(a[0]), f(a[1]), f(a[2]), f(a[3])],
            (SET_DEPTH_CLEAR_VALUE, 2) => self.clear_depth = f64::from_bits((a[0] as u64) << 32 | a[1] as u64) as f32,
            (SET_DEPTH_RANGE, 4) => {
                // Two doubles, high word first (the senders' order is assumed).
                let d = |i: usize| f64::from_bits((a[i] as u64) << 32 | a[i + 1] as u64) as f32;
                self.raster.depth_range = [d(0), d(2)];
            }
            (SET_FOG_PARAMETER_FLOAT_FORM, 5) | (SET_FOG_PARAMETER_INTEGER_FORM, 5) => return self.fog(selector == SET_FOG_PARAMETER_INTEGER_FORM, a),
            (SET_LIGHT_PARAMETER_FLOAT_FORM, 6) | (SET_LIGHT_PARAMETER_INTEGER_FORM, 6) => return self.light(selector == SET_LIGHT_PARAMETER_INTEGER_FORM, a),
            (SET_LIGHTING_MODEL_PARAMETER_FLOAT_FORM, 5) | (SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM, 5) => return self.light_model(selector == SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM, a),
            (SET_MATERIAL_PARAMETER_FLOAT_FORM, 6) => return self.material(a),
            _ => return Taken::Unmodelled,
        }
        self.dirty = true;
        Taken::Yes
    }

    /// A user clip plane: the plane name, 0x3000 + n, then the four coefficients as doubles, high word
    /// first (nine words; established). The pipe's engine multiplies them at the call by a 16-word block
    /// and keeps the eye-space plane (established); that the block is the inverse of the model-view
    /// matrix, as OpenGL specifies, is inferred. A model-view matrix with no inverse leaves the plane as it was (what the
    /// pipe does is unknown).
    /// No draw state changes: the vertex path reads the planes when a primitive begins.
    fn clip_plane(&mut self, a: &[u32]) -> Taken {
        let Some(n) = (a[0] as usize).checked_sub(gl::CLIP_PLANE0 as usize).filter(|&n| n < gl::CLIP_PLANES) else {
            return Taken::Error(gl::INVALID_ENUM);
        };
        let d = |i: usize| f64::from_bits((a[1 + 2 * i] as u64) << 32 | a[2 + 2 * i] as u64) as f32;
        if let Some(plane) = self.to_eye(std::array::from_fn(d)) {
            self.user_plane[n] = plane;
        }
        Taken::Yes
    }

    /// Texture coordinate generation (established): the
    /// coordinate (0x2000 to 0x2003), the parameter name, then seven value words. The mode name
    /// takes one integer word; the object and eye planes four floats. The pipe accepts the coordinates
    /// 0x2000 to 0x2003, the names 0x2500 to 0x2502 and 0x81f4 to 0x81f7 and the modes 0x2400 to 0x2402
    /// and 0x81f0 to 0x81f3, and refuses any other with an invalid enumerant (established). The eye
    /// plane is multiplied at the call by the block a clip plane uses (established; that the block
    /// is the inverse model-view is inferred) and the object plane is stored unchanged; the
    /// equations are OpenGL's (inferred). The distance names and modes are accepted and
    /// not modelled: they are counted and change nothing.
    fn tex_gen(&mut self, a: &[u32]) -> Taken {
        let Some(i) = (a[0] as usize).checked_sub(gl::TEXTURE_COORD_S as usize).filter(|&i| i < 4) else {
            return Taken::Error(gl::INVALID_ENUM);
        };
        let plane: [f32; 4] = std::array::from_fn(|k| f(a[2 + k]));
        match a[1] {
            // The pipe accepts the three modes for any coordinate and the four distance modes; sphere map
            // on r or q is not refused and the engine appears to ignore it (inferred): the mode is kept
            // and the generator leaves the coordinate as it was. The distance modes are accepted and
            // not modelled. Any other value is an invalid enumerant (established).
            gl::TEXTURE_GEN_MODE => match a[2] {
                texgen::EYE_LINEAR..=texgen::SPHERE_MAP => self.texgen.mode[i] = a[2],
                gl::DISTANCE_MODE_FIRST..=gl::DISTANCE_MODE_LAST => return Taken::Unmodelled,
                _ => return Taken::Error(gl::INVALID_ENUM),
            },
            gl::OBJECT_PLANE => self.texgen.object[i] = plane,
            gl::EYE_PLANE => {
                if let Some(p) = self.to_eye(plane) {
                    self.texgen.eye[i] = p;
                }
            }
            gl::DISTANCE_PLANE_FIRST..=gl::DISTANCE_PLANE_LAST => return Taken::Unmodelled,
            _ => return Taken::Error(gl::INVALID_ENUM),
        }
        Taken::Yes
    }

    /// The current normal in eye space, not normalised: what a sphere-mapped coordinate reflects about.
    pub fn eye_normal(&mut self) -> [f32; 3] {
        self.refresh_normal_matrix();
        let (nm, n) = (&self.light.normal_matrix, self.normal);
        [nm[0] * n[0] + nm[3] * n[1] + nm[6] * n[2], nm[1] * n[0] + nm[4] * n[1] + nm[7] * n[2], nm[2] * n[0] + nm[5] * n[1] + nm[8] * n[2]]
    }

    /// A plane given in object coordinates, in eye coordinates: its coefficients, as a row
    /// vector, times the inverse of the current model-view matrix. None if that has no inverse.
    fn to_eye(&self, p: [f32; 4]) -> Option<[f32; 4]> {
        let inv = math::invert(self.mv.get())?;
        Some(std::array::from_fn(|j| (0..4).map(|k| p[k] * inv[j * 4 + k]).sum()))
    }

    fn params(int: bool, a: &[u32]) -> [f32; 4] {
        std::array::from_fn(|i| if int { a[i] as i32 as f32 } else { f(a[i]) })
    }

    /// A fog parameter: the parameter's name, then four values, in float or integer form. The float
    /// form is five words and the integer form a one-word token and a four-word continuation; both
    /// layouts are established, though how the name word reaches the engine on the integer path is
    /// unknown. The pipe accepts seven names (index, density, start,
    /// end, mode, colour and the offset extension) and four modes (linear, exp, exp2 and the function
    /// table extension), and answers any other name or mode with an invalid enumerant, keeping the old
    /// mode (established). It does not range-check start, end or density (established). The index, the
    /// offset and the function-table mode are accepted and not modelled. The integer colour scale is
    /// 2^-31, clamped to 0..1 (the scale is established; reading it as a value mapping is inferred).
    fn fog(&mut self, int: bool, a: &[u32]) -> Taken {
        let fog = &mut self.raster.fog;
        let v = |k: usize| if int { a[1 + k] as i32 as f32 } else { f(a[1 + k]) };
        match a[0] {
            gl::FOG_MODE => {
                let mode = if int { a[1] } else { f(a[1]) as u32 };
                match mode {
                    gl::LINEAR | gl::EXP | gl::EXP2 => fog.mode = mode,
                    gl::FOG_FUNC => return Taken::Unmodelled,
                    _ => return Taken::Error(gl::INVALID_ENUM),
                }
            }
            gl::FOG_DENSITY => fog.density = v(0),
            gl::FOG_START => fog.start = v(0),
            gl::FOG_END => fog.end = v(0),
            gl::FOG_COLOR => {
                let c = |k: usize| if int { (a[1 + k] as i32 as f64 / 2147483648.0).clamp(0.0, 1.0) as f32 } else { f(a[1 + k]) };
                fog.colour = [c(0), c(1), c(2)];
            }
            gl::FOG_INDEX | gl::FOG_OFFSET => return Taken::Unmodelled,
            _ => return Taken::Error(gl::INVALID_ENUM),
        }
        self.dirty = true;
        Taken::Yes
    }

    fn light(&mut self, int: bool, a: &[u32]) -> Taken {
        let Some(n) = (a[0] as usize).checked_sub(gl::LIGHT0 as usize).filter(|&n| n < MAX_LIGHTS) else {
            return Taken::Error(gl::INVALID_ENUM);
        };
        let p = Self::params(int, &a[2..6]);
        let (mv, l) = (*self.mv.get(), &mut self.light.lights[n]);
        let c3 = [p[0], p[1], p[2]];
        match a[1] {
            gl::AMBIENT => l.ambient = c3,
            gl::DIFFUSE => l.diffuse = c3,
            gl::SPECULAR => l.specular = c3,
            gl::POSITION => l.pos = math::xform(&mv, p),
            gl::SPOT_DIRECTION => {
                let e = math::xform(&mv, [p[0], p[1], p[2], 0.0]);
                l.spot_dir = [e[0], e[1], e[2]];
            }
            gl::SPOT_EXPONENT => l.spot_exp = p[0],
            gl::SPOT_CUTOFF => {
                l.spot_on = (p[0] != 180.0) as u32;
                l.spot_cos_cutoff = if p[0] == 180.0 { -1.0 } else { p[0].to_radians().cos() };
            }
            gl::CONSTANT_ATTENUATION => l.atten[2] = p[0],
            gl::LINEAR_ATTENUATION => l.atten[0] = p[0],
            gl::QUADRATIC_ATTENUATION => l.atten[1] = p[0],
            _ => return Taken::Unmodelled,
        }
        Taken::Yes
    }

    fn light_model(&mut self, int: bool, a: &[u32]) -> Taken {
        let p = Self::params(int, &a[1..5]);
        match a[0] {
            gl::LIGHT_MODEL_AMBIENT => self.light.amb_sum = [p[0], p[1], p[2]],
            gl::LIGHT_MODEL_LOCAL_VIEWER => self.light.local_viewer = (p[0] != 0.0) as u32,
            gl::LIGHT_MODEL_TWO_SIDE => self.light.two_sided = (p[0] != 0.0) as u32,
            _ => return Taken::Unmodelled,
        }
        Taken::Yes
    }

    fn material(&mut self, a: &[u32]) -> Taken {
        let p = [f(a[2]), f(a[3]), f(a[4]), f(a[5])];
        let c3 = [p[0], p[1], p[2]];
        for face in 0..2 {
            // gl::FRONT front, gl::BACK back, gl::FRONT_AND_BACK both.
            if (a[0] == gl::FRONT && face != 0) || (a[0] == gl::BACK && face != 1) {
                continue;
            }
            let m = &mut self.light.mat[face];
            match a[1] {
                gl::AMBIENT => m.ambient = c3,
                gl::DIFFUSE => m.diffuse = p,
                gl::SPECULAR => m.specular = c3,
                gl::EMISSION => m.emission = c3,
                gl::SHININESS => m.shininess = p[0],
                gl::AMBIENT_AND_DIFFUSE => {
                    m.ambient = c3;
                    m.diffuse = p;
                }
                _ => return Taken::Unmodelled,
            }
        }
        Taken::Yes
    }

    /// A buffer configuration for level 0: the layout the planes follow, and
    /// whether the window is double buffered. A level above 0 draws overlay
    /// planes, which are not drawn (the target stays as it was).
    pub fn configure(&mut self, level: i32, double: bool, layout: crate::dev::kona::back::store::layout::Layout, samples: u32) {
        if level == 0 {
            self.layout = Some(layout);
            self.double = double;
            // A configuration resets the mask, the pattern and the alpha options.
            self.raster.ms.configure(samples);
            self.dirty = true;
        }
    }

    /// The window the kernel placed, in plane pixels, under display id `display_id`
    /// whose swaps so far last showed buffer `shown`: a context placed after a swap
    /// draws into the buffer the display is not showing, as one placed before does.
    /// `origin` is where window coordinates start; `clip` is the rectangle the
    /// kernel allows drawing in, which need not start at the origin or equal
    /// the window.
    pub fn place(&mut self, origin: [i32; 2], clip: Rect, display_id: u32, shown: u8, id_test: bool) {
        self.origin = origin;
        self.clip = clip;
        self.placed = true;
        self.window_id = display_id as i32;
        self.id_test = id_test;
        self.display_id = display_id;
        self.shown = shown;
        self.dirty = true;
    }

    /// A swap of display id `id` that shows buffer `bank`.
    pub fn swapped(&mut self, id: u32, bank: u8) {
        if self.display_id == id && self.shown != bank {
            self.shown = bank;
            self.dirty = true;
        }
    }

    /// The window moved to `x`, `y`: its origin and its clip rectangle move by the
    /// same amount.
    pub fn move_to(&mut self, x: i32, y: i32) {
        if self.placed {
            let (dx, dy) = (x.saturating_sub(self.origin[0]), y.saturating_sub(self.origin[1]));
            self.origin = [x, y];
            let c = &mut self.clip;
            *c = Rect { x0: c.x0.saturating_add(dx), y0: c.y0.saturating_add(dy), x1: c.x1.saturating_add(dx), y1: c.y1.saturating_add(dy) };
            self.dirty = true;
        }
    }

    /// A bind made `d` the bound texture.
    pub fn set_texture(&mut self, d: Option<TexDesc>) {
        self.texture = d;
        self.dirty = true;
    }

    /// Load the polygon stipple's 32 rows.
    pub fn set_polygon_stipple(&mut self, rows: [u32; 32]) {
        self.raster.polygon_stipple = rows;
        self.dirty = true;
    }

    /// The target, from the layout, window and draw buffer, into the raster group.
    pub fn retarget(&mut self) {
        let mut t = Target::default();
        if let Some(l) = &self.layout {
            let mask = self.draw.unwrap_or(if self.double { 2 } else { 1 });
            // One word a buffer: the first is shown at start-up, and a swap flips
            // which; the front buffer is the one shown, the back the other.
            let shown = if self.double { self.shown as u32 } else { 0 };
            for (bit, word) in [(1u32, l.colour.start + shown), (2, l.colour.start + 1 - shown)] {
                if mask & bit != 0 && word < l.colour.end && (bit == 1 || self.double) {
                    t.colour[t.colours as usize] = word;
                    t.colours += 1;
                }
            }
            if !l.depth.is_empty() {
                t.depth_word = l.depth.start;
                t.depth_bits = 32;
            }
            if !l.stencil.is_empty() {
                t.stencil_word = l.stencil.start;
                t.stencil_bits = 8;
            }
        }
        self.raster.target = t;
    }

    /// The draw state for the raster group as it stands, built again when a
    /// state command has changed it. `screen` is the managed area.
    pub fn draw_state(&mut self, screen: Rect) -> DrawState {
        if self.dirty || self.ds.is_none() {
            self.retarget();
            let r = &self.raster;
            let t = &r.target;
            let [ox, oy] = self.origin;
            // Nothing is drawn before the kernel has placed the window.
            let mut clip = if self.placed { self.clip } else { Rect::default() };
            if r.scissor_test {
                let s = r.scissor;
                let (x0, y0) = (ox.saturating_add(s[0]), oy.saturating_add(s[1]));
                clip = clip.intersect(&Rect { x0, y0, x1: x0.saturating_add(s[2]), y1: y0.saturating_add(s[3]) });
            }
            let [vx, vy, vw, vh] = r.viewport;
            // The viewport rectangle holds every pixel the viewport touches.
            let (ox, oy) = (ox as f64, oy as f64);
            let cut = |v: f64| v.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
            let viewport = Rect { x0: cut(ox + (vx as f64).floor()), y0: cut(oy + (vy as f64).floor()), x1: cut(ox + (vx as f64 + vw as f64).ceil()), y1: cut(oy + (vy as f64 + vh as f64).ceil()) };
            let samples = self.layout.as_ref().filter(|l| !l.multisample.is_empty()).and_then(|l| {
                let run = |r: &std::ops::Range<u32>| (r.start, r.end - r.start);
                r.ms.draw(run(&l.multisample), run(&l.depth), run(&l.stencil))
            });
            self.ds = Some(DrawState {
                origin: self.origin,
                clip: clip.intersect(&screen),
                viewport,
                colour: t.colour,
                colours: t.colours,
                depth_word: t.depth_word,
                depth_bits: t.depth_bits,
                depth_test: r.depth_test,
                depth_func: Compare::from_gl(r.depth_func).unwrap_or_default(),
                depth_mask: r.depth_mask,
                stencil_word: t.stencil_word,
                stencil_bits: t.stencil_bits,
                stencil: StencilState {
                    enabled: r.stencil_test,
                    func: Compare::from_gl(r.stencil_func).unwrap_or_default(),
                    reference: r.stencil_ref as u8,
                    read_mask: r.stencil_read_mask as u8,
                    write_mask: r.stencil_write_mask as u8,
                    fail: StencilOp::from_gl(r.stencil_ops[0]).unwrap_or_default(),
                    zfail: StencilOp::from_gl(r.stencil_ops[1]).unwrap_or_default(),
                    zpass: StencilOp::from_gl(r.stencil_ops[2]).unwrap_or_default(),
                },
                // The alpha reference is the token's float in 13-bit units, rounded: the model's own.
                alpha: AlphaTest { enabled: r.alpha_test, func: Compare::from_gl(r.alpha_func).unwrap_or_default(), reference: (r.alpha_ref.clamp(0.0, 1.0) * 8191.0 + 0.5) as u32 },
                blend: BlendState {
                    enabled: r.blend,
                    src: BlendFactor::from_gl(r.blend_factors[0]).unwrap_or(BlendFactor::One),
                    dst: BlendFactor::from_gl(r.blend_factors[1]).unwrap_or(BlendFactor::Zero),
                    eq: BlendEq::Add,
                    constant: r.blend_colour.map(|c| (c.clamp(0.0, 1.0) * 8191.0 + 0.5) as u32),
                },
                logic: r.logic_op_on.then(|| LogicOp::from_gl(r.logic_op)).flatten(),
                colour_mask: r.colour_mask,
                dither: r.dither,
                offset: r.offset_fill.then_some(r.offset),
                ownership: self.id_test.then(|| Ownership::encode(self.window_id).0),
                line_stipple: r.line_stipple_on.then_some((r.line_stipple.1, r.line_stipple.0)),
                polygon_stipple: r.polygon_stipple_on.then_some(r.polygon_stipple),
                samples,
                // Lines and points smooth in a multisampled window and polygons multisample there (the
                // sampling word and the point handler's order of tests, established; see
                // `raster::smooth`).
                smooth: Smooth { points: r.point_smooth, lines: r.line_smooth, polygons: r.polygon_smooth && samples.is_none() },
                fog: Fog::new(r.fog.on, r.fog.mode, r.fog.density, r.fog.start, r.fog.end, r.fog.colour),
                texture: self.texture.filter(|d| if d.volume { r.texture_3d } else { r.texture_2d }).map(|d| TexDesc { env: EnvMode::from_gl(r.env_mode).unwrap_or_default(), env_colour: r.env_colour.map(|c| (c.clamp(0.0, 1.0) * 8191.0 + 0.5) as u32), ..d }),
            });
            self.dirty = false;
        }
        self.ds.expect("just built")
    }

    /// The normal matrix, brought up to date with the model-view matrix.
    pub fn refresh_normal_matrix(&mut self) {
        if self.normal_stale {
            self.light.normal_matrix = math::normal_matrix(self.mv.get());
            self.normal_stale = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn matrix_commands_compose_on_the_current_stack() {
        let mut g = Gl::default();
        assert_eq!(g.matrix(TRANSLATE_FLOATS, &bits(&[1.0, 2.0, 3.0])), Taken::Yes);
        assert_eq!(g.mv.get()[12..15], [1.0, 2.0, 3.0]);
        assert_eq!(g.matrix(PUSH_MATRIX_STACK, &[0]), Taken::Yes);
        g.matrix(SCALE_FLOATS, &bits(&[2.0, 2.0, 2.0]));
        assert_eq!(g.mv.get()[0], 2.0);
        assert_eq!(g.matrix(POP_MATRIX_STACK, &[0]), Taken::Yes);
        assert_eq!(g.mv.get()[0], 1.0);
        assert_eq!(g.matrix(POP_MATRIX_STACK, &[0]), Taken::Error(gl::STACK_UNDERFLOW));
        g.matrix(SELECT_CURRENT_MATRIX_STACK, &[gl::PROJECTION]);
        g.matrix(LOAD_MATRIX_16_FLOATS, &bits(&math::IDENT.map(|x| x * 3.0)));
        assert_eq!(g.proj.get()[0], 3.0);
        assert_eq!(g.mv.get()[0], 1.0, "the model-view matrix is untouched");
        for _ in 0..PROJ_DEPTH - 1 {
            assert_eq!(g.matrix(PUSH_MATRIX_STACK, &[0]), Taken::Yes);
        }
        assert_eq!(g.matrix(PUSH_MATRIX_STACK, &[0]), Taken::Error(gl::STACK_OVERFLOW));
        g.in_begin = true;
        assert_eq!(g.matrix(LOAD_IDENTITY_MATRIX, &[0]), Taken::Error(gl::INVALID_OPERATION));
    }

    #[test]
    fn lights_are_set_up_in_eye_space_and_chained_when_enabled() {
        let mut g = Gl::default();
        g.matrix(TRANSLATE_FLOATS, &bits(&[0.0, 0.0, -5.0]));
        let pos = [gl::LIGHT0 + 1, gl::POSITION, 0, 0, 0, 0];
        let mut a = pos;
        a[2..].copy_from_slice(&bits(&[1.0, 2.0, 3.0, 1.0]));
        assert_eq!(g.state(SET_LIGHT_PARAMETER_FLOAT_FORM, &a), Taken::Yes);
        assert_eq!(g.light.lights[1].pos, [1.0, 2.0, -2.0, 1.0]);
        g.enable(gl::LIGHT0 + 1, true);
        g.enable(gl::LIGHT0, true);
        assert_eq!((g.light.head, g.light.count, g.light.lights[0].next), (0, 2, 1));
        g.enable(gl::LIGHT0, false);
        assert_eq!((g.light.head, g.light.count), (1, 1));
        assert_eq!(g.state(SET_LIGHT_PARAMETER_FLOAT_FORM, &[gl::LIGHT0 + 9, gl::AMBIENT, 0, 0, 0, 0]), Taken::Error(gl::INVALID_ENUM));
        assert_eq!(g.state(SET_LIGHT_PARAMETER_FLOAT_FORM, &[gl::LIGHT0, 0x9999, 0, 0, 0, 0]), Taken::Unmodelled);
    }

    #[test]
    fn a_material_or_model_word_lands_on_the_faces_it_names() {
        let mut g = Gl::default();
        let a = [gl::FRONT, gl::SHININESS].into_iter().chain(bits(&[30.0, 0.0, 0.0, 0.0])).collect::<Vec<_>>();
        g.state(SET_MATERIAL_PARAMETER_FLOAT_FORM, &a);
        assert_eq!((g.light.mat[0].shininess, g.light.mat[1].shininess), (30.0, 0.0));
        let a = [gl::LIGHT_MODEL_TWO_SIDE].into_iter().chain(bits(&[1.0, 0.0, 0.0, 0.0])).collect::<Vec<_>>();
        g.state(SET_LIGHTING_MODEL_PARAMETER_FLOAT_FORM, &a);
        assert_eq!(g.light.two_sided, 1);
        let a = [gl::LIGHT_MODEL_TWO_SIDE, 0, 0, 0, 0];
        g.state(SET_LIGHTING_MODEL_PARAMETER_INTEGER_FORM, &a);
        assert_eq!(g.light.two_sided, 0, "the integer form");
    }

    /// The double forms of the matrix tokens equal the float forms, and the frustum and
    /// orthographic tokens multiply GL's own matrices (a double is two words, high first).
    #[test]
    fn matrix_double_forms_and_projections_match_the_float_ones() {
        let d = |v: f64| [(v.to_bits() >> 32) as u32, v.to_bits() as u32];
        let words = |vs: &[f64]| vs.iter().flat_map(|&v| d(v)).collect::<Vec<u32>>();
        let mut a = Gl::default();
        let mut b = Gl::default();
        a.matrix(LOAD_MATRIX_16_FLOATS, &(0..16).map(|i| (i as f32 * 0.5).to_bits()).collect::<Vec<_>>());
        b.matrix(LOAD_MATRIX_16_DOUBLES, &words(&(0..16).map(|i| i as f64 * 0.5).collect::<Vec<_>>()));
        assert_eq!(a.mv.get(), b.mv.get());
        a.matrix(TRANSLATE_FLOATS, &[1.0f32.to_bits(), 2.0f32.to_bits(), 3.0f32.to_bits()]);
        b.matrix(TRANSLATE_DOUBLES, &words(&[1.0, 2.0, 3.0]));
        a.matrix(SCALE_FLOATS, &[2.0f32.to_bits(), 3.0f32.to_bits(), 4.0f32.to_bits()]);
        b.matrix(SCALE_DOUBLES, &words(&[2.0, 3.0, 4.0]));
        a.matrix(ROTATE_FLOATS, &[30.0f32.to_bits(), 0.0f32.to_bits(), 1.0f32.to_bits(), 0.0f32.to_bits()]);
        b.matrix(ROTATE_DOUBLES, &words(&[30.0, 0.0, 1.0, 0.0]));
        assert_eq!(a.mv.get(), b.mv.get());
        for g in [&mut a, &mut b] {
            g.matrix(SELECT_CURRENT_MATRIX_STACK, &[gl::PROJECTION]);
        }
        b.matrix(MULTIPLY_CURRENT_MATRIX_BY_FRUSTUM, &words(&[-1.0, 1.0, -1.0, 1.0, 1.0, 10.0]));
        a.matrix(MULTIPLY_BY_MATRIX_16_FLOATS, &math::frustum(-1.0, 1.0, -1.0, 1.0, 1.0, 10.0).map(f32::to_bits));
        assert_eq!(a.proj.get(), b.proj.get());
        let mut c = Gl::default();
        c.matrix(SELECT_CURRENT_MATRIX_STACK, &[gl::PROJECTION]);
        c.matrix(MULTIPLY_CURRENT_MATRIX_BY_ORTHOGRAPHIC_PROJECTION, &words(&[0.0, 4.0, 0.0, 2.0, -1.0, 1.0]));
        assert_eq!(c.proj.get(), &math::ortho(0.0, 4.0, 0.0, 2.0, -1.0, 1.0));
    }

    /// The words a double-buffered window draws into are the ones the display
    /// composes: it shows `COLOUR_WORD + bank` and the draw goes to the other.
    #[test]
    fn the_draw_target_is_the_word_the_display_is_not_showing() {
        use crate::dev::kona::back::store::layout::{BufferConfig, Layout, MAIN_COLOUR_WORD};
        let cfg = BufferConfig::from_words(&[0, 8, 0, 1, 0, 0, 24, 0, 0, 0]);
        let layout = Layout::new(&cfg, 0, 64).expect("a plain double-buffered layout");
        assert_eq!(layout.colour.start, MAIN_COLOUR_WORD, "the layout starts where the display reads");
        let mut g = Gl::default();
        g.configure(0, true, layout, 0);
        g.place([0, 0], Rect { x0: 0, y0: 0, x1: 8, y1: 8 }, 5, 0, false);
        g.retarget();
        assert_eq!((g.raster.target.colours, g.raster.target.colour[0]), (1, MAIN_COLOUR_WORD + 1), "bank 0 shown: the back word");
        g.swapped(5, 1);
        g.retarget();
        assert_eq!(g.raster.target.colour[0], MAIN_COLOUR_WORD, "bank 1 shown: the first word");
        // A context placed after the swap starts from the buffer the display shows now.
        let mut late = Gl::default();
        late.configure(0, true, Layout::new(&cfg, 0, 64).expect("layout"), 0);
        late.place([0, 0], Rect { x0: 0, y0: 0, x1: 8, y1: 8 }, 5, 1, false);
        late.retarget();
        assert_eq!(late.raster.target.colour[0], MAIN_COLOUR_WORD);
    }
}
