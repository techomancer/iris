//! Tests: a client library's part. A test describes a draw the way a GL
//! library's client does (`LibState`, `LibDraw`, and the library's
//! enumerants below) and `Board::lib_draw` turns it into what the library
//! would send the board: the state words the drawing pipeline reads, and the
//! transform state primitive assembly hands it. The encodings are the
//! inverses of the board's decoders (`pipe/state.rs`); where a word is
//! narrower than the library's value (the constant colour's 12 bits, the
//! stencil reference's 8) the word holds what the library would send.

pub(in crate::dev::odyssey) use super::super::assembly::vtx;
use super::super::assembly::{self, Prim, XfState, CLIP_PLANES};
use super::super::board::Board;
use super::super::consts::{alpha, blend, gl_cmd, line, mode, pattern, stencil, write, ODSY_DRAW_MASK_BUFFER0, ODSY_DRAW_MASK_BUFFER1};
use super::super::draw;

// The library's enumerants the tests describe draws with.
pub(crate) const GL_POINTS: u32 = assembly::POINTS;
pub(crate) const GL_LINES: u32 = assembly::LINES;
pub(crate) const GL_TRIANGLES: u32 = assembly::TRIANGLES;
pub(crate) const GL_QUADS: u32 = assembly::QUADS;
pub(crate) const GL_POLYGON: u32 = assembly::POLYGON;
pub(crate) const GL_DEPTH_BUFFER_BIT: u32 = draw::CLEAR_DEPTH;
pub(crate) const GL_STENCIL_BUFFER_BIT: u32 = draw::CLEAR_STENCIL;
pub(crate) const GL_COLOR_BUFFER_BIT: u32 = draw::CLEAR_COLOUR;
pub(crate) const GL_COLOUR_MASK_RED: u32 = 0x1;
pub(crate) const GL_COLOUR_MASK_ALPHA: u32 = 0x8;
pub(crate) const GL_COLOUR_MASK_ALL: u32 = 0xf;
pub(crate) const GL_LESS: u32 = 1;
pub(crate) const GL_EQUAL: u32 = 2;
pub(crate) const GL_LEQUAL: u32 = 3;
pub(crate) const GL_GREATER: u32 = 4;
pub(crate) const GL_NOTEQUAL: u32 = 5;
pub(crate) const GL_GEQUAL: u32 = 6;
pub(crate) const GL_ALWAYS: u32 = 7;
pub(crate) const GL_FLAT: u32 = 0x1d00;
pub(crate) const GL_SMOOTH: u32 = 0x1d01;
pub(crate) const GL_POINT: u32 = 0x1b00;
pub(crate) const GL_LINE: u32 = 0x1b01;
pub(crate) const GL_FILL: u32 = 0x1b02;
pub(crate) const GL_FRONT: u32 = 0x0404;
pub(crate) const GL_BACK: u32 = 0x0405;
pub(crate) const GL_CW: u32 = 0x0900;
pub(crate) const GL_CCW: u32 = 0x0901;
pub(crate) const GL_OFFSET_FILL: u32 = 0x1;
pub(crate) const GL_OFFSET_LINE: u32 = 0x2;
pub(crate) const GL_OFFSET_POINT: u32 = 0x4;
pub(crate) const GL_KEEP: u32 = 0x1e00;
pub(crate) const GL_REPLACE: u32 = 0x1e01;
pub(crate) const GL_INCR: u32 = 0x1e02;
pub(crate) const GL_DECR: u32 = 0x1e03;
pub(crate) const GL_INVERT: u32 = 0x150a;
pub(crate) const GL_SRC_COLOR: u32 = 0x0300;
pub(crate) const GL_ONE_MINUS_SRC_COLOR: u32 = 0x0301;
pub(crate) const GL_SRC_ALPHA: u32 = 0x0302;
pub(crate) const GL_ONE_MINUS_SRC_ALPHA: u32 = 0x0303;
pub(crate) const GL_DST_ALPHA: u32 = 0x0304;
pub(crate) const GL_ONE_MINUS_DST_ALPHA: u32 = 0x0305;
pub(crate) const GL_DST_COLOR: u32 = 0x0306;
pub(crate) const GL_ONE_MINUS_DST_COLOR: u32 = 0x0307;
pub(crate) const GL_SRC_ALPHA_SATURATE: u32 = 0x0308;
pub(crate) const GL_CONSTANT_COLOR: u32 = 0x8001;
pub(crate) const GL_ONE_MINUS_CONSTANT_COLOR: u32 = 0x8002;
pub(crate) const GL_CONSTANT_ALPHA: u32 = 0x8003;
pub(crate) const GL_ONE_MINUS_CONSTANT_ALPHA: u32 = 0x8004;
pub(crate) const GL_FUNC_ADD: u32 = 0x8006;
pub(crate) const GL_MIN: u32 = 0x8007;
pub(crate) const GL_MAX: u32 = 0x8008;
pub(crate) const GL_FUNC_SUBTRACT: u32 = 0x800a;
pub(crate) const GL_FUNC_REVERSE_SUBTRACT: u32 = 0x800b;
pub(crate) const GL_LOGIC_OP: u32 = 0x0bf1;
pub(crate) const GL_CLEAR: u32 = 0x1500;
pub(crate) const GL_COPY: u32 = 0x1503;

/// A draw's state as the client sets it: the transform, the fragment
/// state, and the primitive's sizes, smoothing and stipples.
#[derive(Clone, Copy)]
pub(crate) struct LibState {
    /// Column-major.
    pub modelview: [f32; 16],
    pub projection: [f32; 16],
    pub viewport: [i32; 4],
    pub depth_range: [f32; 2],
    pub clip_planes: u32,
    pub clip_plane: [[f32; 4]; CLIP_PLANES as usize],
    pub shade_model: u32,
    pub depth_test: bool,
    pub depth_func: u32,
    pub depth_mask: bool,
    /// Bit 0 the front buffer, bit 1 the back.
    pub draw_buffers: u32,
    pub cull_face: bool,
    pub cull_face_mode: u32,
    pub front_face: u32,
    pub scissor_test: bool,
    pub scissor: [i32; 4],
    pub alpha_test: bool,
    pub alpha_func: u32,
    pub alpha_ref: f32,
    pub stencil_test: bool,
    pub stencil_func: u32,
    pub stencil_ref: i32,
    pub stencil_value_mask: u32,
    pub stencil_write_mask: u32,
    pub stencil_op: [u32; 3],
    pub blend: bool,
    pub blend_src: u32,
    pub blend_dst: u32,
    pub blend_equation: u32,
    pub blend_color: [f32; 4],
    pub color_logic_op: bool,
    pub logic_op_mode: u32,
    pub fog: bool,
    pub polygon_offset: u32,
    /// The polygon offset in depth-word steps over 2^24.
    pub offset_bias: f32,
    pub texture_2d: bool,
}

impl Default for LibState {
    fn default() -> Self {
        let id = assembly::XfState::default().modelview;
        LibState {
            modelview: id,
            projection: id,
            viewport: [0; 4],
            depth_range: [0.0, 1.0],
            clip_planes: 0,
            clip_plane: [[0.0; 4]; CLIP_PLANES as usize],
            shade_model: GL_SMOOTH,
            depth_test: false,
            depth_func: GL_LESS,
            depth_mask: true,
            draw_buffers: 1,
            cull_face: false,
            cull_face_mode: GL_BACK,
            front_face: GL_CCW,
            scissor_test: false,
            scissor: [0; 4],
            alpha_test: false,
            alpha_func: GL_ALWAYS,
            alpha_ref: 0.0,
            stencil_test: false,
            stencil_func: GL_ALWAYS,
            stencil_ref: 0,
            stencil_value_mask: u32::MAX,
            stencil_write_mask: u32::MAX,
            stencil_op: [GL_KEEP; 3],
            blend: false,
            blend_src: 1,
            blend_dst: 0,
            blend_equation: GL_FUNC_ADD,
            blend_color: [0.0; 4],
            color_logic_op: false,
            logic_op_mode: GL_COPY,
            fog: false,
            polygon_offset: 0,
            offset_bias: 0.0,
            texture_2d: false,
        }
    }
}

impl LibState {
    /// The transform state assembly would hand the pipeline.
    pub fn xf(&self) -> XfState {
        XfState {
            modelview: self.modelview,
            projection: self.projection,
            viewport: self.viewport,
            depth_range: self.depth_range,
            clip_planes: self.clip_planes,
            clip_plane: self.clip_plane,
        }
    }
}

/// One primitive as a client draws it.
pub(crate) struct LibDraw<'a> {
    pub mode: u32,
    pub count: u32,
    pub vertices: &'a [f32],
    pub state: &'a LibState,
    pub back_colours: bool,
    pub point_size: f32,
    pub line_width: f32,
    pub point_smooth: bool,
    pub line_smooth: bool,
    pub polygon_smooth: bool,
    pub polygon_stipple: &'a [u32; 32],
    pub polygon_stipple_enabled: bool,
    pub polygon_modes: Option<[u32; 2]>,
    pub line_stipple_enabled: bool,
    pub line_stipple_repeat: u32,
    pub line_stipple_pattern: u32,
    /// The standalone line stipple's phase, carried between draws.
    pub line_phase: Option<u32>,
}

impl<'a> LibDraw<'a> {
    pub fn new(mode: u32, count: u32, vertices: &'a [f32], state: &'a LibState, polygon_stipple: &'a [u32; 32]) -> Self {
        LibDraw {
            mode,
            count,
            vertices,
            state,
            back_colours: false,
            point_size: 0.0,
            line_width: 0.0,
            point_smooth: false,
            line_smooth: false,
            polygon_smooth: false,
            polygon_stipple,
            polygon_stipple_enabled: false,
            polygon_modes: None,
            line_stipple_enabled: false,
            line_stipple_repeat: 0,
            line_stipple_pattern: 0,
            line_phase: None,
        }
    }
}

/// The board's code of a blend factor.
fn factor_code(f: u32, dst: bool) -> u32 {
    match f {
        0 => 0x00,
        1 => 0x08,
        GL_SRC_COLOR if dst => 0x10,
        GL_DST_COLOR if !dst => 0x10,
        GL_SRC_ALPHA => 0x18,
        GL_DST_ALPHA => 0x20,
        GL_CONSTANT_COLOR => 0x28,
        GL_CONSTANT_ALPHA => 0x30,
        GL_ONE_MINUS_SRC_COLOR if dst => 0x38,
        GL_ONE_MINUS_DST_COLOR if !dst => 0x38,
        GL_ONE_MINUS_SRC_ALPHA => 0x39,
        GL_ONE_MINUS_DST_ALPHA => 0x3a,
        GL_ONE_MINUS_CONSTANT_COLOR => 0x3b,
        GL_ONE_MINUS_CONSTANT_ALPHA => 0x3c,
        GL_SRC_ALPHA_SATURATE => 0x40,
        _ => 0x7f,
    }
}

/// The board's code of a blend equation.
fn equation_code(e: u32) -> u32 {
    match e {
        GL_MIN => 1,
        GL_MAX => 2,
        GL_LOGIC_OP => 5,
        GL_FUNC_SUBTRACT => 6,
        GL_FUNC_REVERSE_SUBTRACT => 7,
        _ => 0,
    }
}

/// The board's code of a stencil operation.
fn stencil_code(op: u32) -> u32 {
    match op {
        GL_KEEP => 0,
        0 => 1,
        GL_REPLACE => 2,
        GL_INCR => 3,
        GL_DECR => 4,
        GL_INVERT => 5,
        _ => 7,
    }
}

/// A component in [0, 1] as a 12-bit field over 4096, 1.0 sent as 4095.
fn field12(c: f32) -> u32 {
    if !(c > 0.0) {
        0
    } else if c >= 1.0 {
        0xfff
    } else {
        ((c * 4096.0) as u32).min(0xfff)
    }
}

impl Board {
    /// Draw d as the library would: its state words set, then its vertices
    /// through the drawing pipeline under its transform state. The line
    /// stipple's phase comes back in d.line_phase.
    pub fn lib_draw(&mut self, d: &mut LibDraw) {
        self.encode_lib_draw(d);
        self.lib_draw_encoded(d);
    }

    /// lib_draw() with the state words already set (encode_lib_draw()).
    pub fn lib_draw_encoded(&mut self, d: &mut LibDraw) {
        let xf = d.state.xf();
        let mut p = Prim::new(d.mode, d.count, d.vertices, &xf);

        p.back_colours = d.back_colours;
        p.whole_polygon = d.mode == GL_POLYGON;
        p.line_phase = d.line_phase;
        self.draw_prim(&mut p);
        d.line_phase = p.line_phase;
    }

    /// Set the state words the drawing pipeline reads from a draw.
    pub fn encode_lib_draw(&mut self, d: &LibDraw) {
        self.encode_lib_state(d.state);
        let set = |on: bool, bit: u32| if on { bit } else { 0 };
        let mut m = self.stream.pattern_mode & !(mode::POINT_SMOOTH | mode::LINE_SMOOTH | (1 << 2) | mode::GL_LINE_STIPPLE | gl_cmd::RECT_PATTERN_ENABLE);

        m |= set(d.point_smooth, mode::POINT_SMOOTH);
        m |= set(d.line_smooth, mode::LINE_SMOOTH);
        m |= set(d.polygon_smooth, 1 << 2);
        m |= set(d.line_stipple_enabled, mode::GL_LINE_STIPPLE);
        m |= set(d.polygon_stipple_enabled, gl_cmd::RECT_PATTERN_ENABLE);
        self.stream.pattern_mode = m;
        let size = |v: f32| ((if v > 0.0 { v } else { 1.0 }) * line::POINT_SIZE_SCALE) as u32 & line::POINT_SIZE_MASK;
        self.line_point_words = [size(d.line_width), size(d.point_size)];
        self.stream.line_stipple[0] = d.line_stipple_pattern & 0xffff;
        self.stream.line_stipple[line::STIPPLE_CONTROL] = d.line_stipple_repeat.clamp(1, 256) - 1;
        if d.polygon_stipple_enabled {
            self.stream.pattern_shape &= pattern::SHAPE_OPAQUE;
            for i in 0..32 {
                self.stream.pattern_buffer[31 - i] = d.polygon_stipple[i];
            }
        }
        let code = |m: u32| match m {
            GL_POINT => 0,
            GL_LINE => 1,
            _ => 2,
        };
        let [front, back] = d.polygon_modes.unwrap_or([GL_FILL, GL_FILL]);
        self.polygon_mode_word = code(front) << 2 | code(back);
    }

    /// Set the state words the drawing pipeline reads from st.
    pub fn encode_lib_state(&mut self, st: &LibState) {
        let mut m = self.stream.pattern_mode
            & !(mode::DEPTH_TEST | mode::STENCIL_TEST | mode::BLEND | mode::FOG | mode::ALPHA_TEST | mode::FLAT_SHADE
                | mode::CULL_FACE | (mode::CULL_MODE_MASK << mode::CULL_MODE_SHIFT) | mode::FRONT_FACE_CW
                | mode::OFFSET_FILL | mode::OFFSET_LINE | mode::OFFSET_POINT | mode::COLOR_LOGIC_OP | mode::INDEX_LOGIC_OP);
        let set = |on: bool, bit: u32| if on { bit } else { 0 };

        m |= set(st.depth_test, mode::DEPTH_TEST);
        m |= set(st.stencil_test, mode::STENCIL_TEST);
        m |= set(st.blend && !st.color_logic_op, mode::BLEND);
        m |= set(st.fog, mode::FOG);
        m |= set(st.alpha_test, mode::ALPHA_TEST);
        m |= set(st.shade_model == GL_FLAT, mode::FLAT_SHADE);
        // The board's bit is set for the library's counter-clockwise front
        // (pipe::raster::RasterState::of()).
        m |= set(st.front_face != GL_CW, mode::FRONT_FACE_CW);
        m |= set(st.color_logic_op, mode::COLOR_LOGIC_OP);
        m |= set(st.polygon_offset & GL_OFFSET_FILL != 0, mode::OFFSET_FILL);
        m |= set(st.polygon_offset & GL_OFFSET_LINE != 0, mode::OFFSET_LINE);
        m |= set(st.polygon_offset & GL_OFFSET_POINT != 0, mode::OFFSET_POINT);
        if st.cull_face {
            let faces = match st.cull_face_mode {
                GL_BACK => 1,
                GL_FRONT => 2,
                _ => 3,
            };
            m |= mode::CULL_FACE | faces << mode::CULL_MODE_SHIFT;
        }
        self.stream.pattern_mode = m;
        self.gl_depth_func = st.depth_func & 7;
        // The colour mask stays the board's, as a stream sets it.
        self.gl_write_mask = (self.gl_write_mask & !(write::DEPTH | (write::STENCIL_MASK << write::STENCIL_SHIFT)))
            | if st.depth_mask { write::DEPTH } else { 0 }
            | (st.stencil_write_mask & write::STENCIL_MASK) << write::STENCIL_SHIFT;
        self.alpha_word = (st.alpha_func & alpha::FUNC_MASK) << alpha::FUNC_SHIFT | field12(st.alpha_ref);
        let sref = st.stencil_ref.clamp(0, 0xff) as u32;
        self.stencil_word = (st.stencil_func & stencil::FUNC_MASK) << stencil::FUNC_SHIFT
            | stencil_code(st.stencil_op[0]) << stencil::OP_FAIL_SHIFT
            | stencil_code(st.stencil_op[1]) << stencil::OP_ZFAIL_SHIFT
            | stencil_code(st.stencil_op[2]) << stencil::OP_ZPASS_SHIFT
            | (st.stencil_value_mask & stencil::VALUE_MASK_MASK) << stencil::VALUE_MASK_SHIFT
            | sref;
        self.blend_word = equation_code(st.blend_equation) << blend::EQ_SHIFT | factor_code(st.blend_src, false) << blend::SRC_SHIFT | factor_code(st.blend_dst, true);
        let c = st.blend_color.map(field12);
        self.const_colour = [c[0] << 12 | c[1], c[2] << 12 | c[3]];
        self.logic_op_word = st.logic_op_mode.wrapping_sub(GL_CLEAR) & 0xf;
        // The scissor as the clip box: x then y, lower bound in bits 23:12.
        let (x0, y0, x1, y1) = if st.scissor_test {
            (st.scissor[0], st.scissor[1], st.scissor[0] + st.scissor[2] - 1, st.scissor[1] + st.scissor[3] - 1)
        } else {
            (0, 0, 0xfff, 0xfff)
        };
        let f = |v: i32| (v.clamp(0, 0xfff)) as u32;
        self.gl_clip_box = [f(x0) << 12 | f(x1), f(y0) << 12 | f(y1)];
        if x1 < x0 || y1 < y0 {
            // An empty scissor: a box whose bounds cross.
            self.gl_clip_box = [1 << 12, 1 << 12];
        }
        self.polygon_offset_word = ((st.offset_bias as f64 * (1u64 << 24) as f64).round() as i64 as u32) & 0x1ff_ffff;
        self.draw_mask = (if st.draw_buffers & 1 != 0 { ODSY_DRAW_MASK_BUFFER0 } else { 0 })
            | (if st.draw_buffers & 2 != 0 { ODSY_DRAW_MASK_BUFFER1 } else { 0 });
    }
}
