//! The fragment pipeline's state, as the board's own words give it, grouped
//! by the stage that reads it: the rasteriser, the shader and the back end.
//!
//! Each record holds the words themselves, as the command FIFO delivered
//! them, and its accessors decode the fields `ignore/odsy/docs/tokens.md`
//! and `words.md` establish (each accessor says which) and nothing more. A
//! stage branches on these accessors; there is no GL state in between.
//! `Board::fragment_words()` captures the three records for a draw.
//!
//! Comparisons are the GL enum minus 0x200 (established), blend factors
//! and stencil operations the board's own codes (tables below), the
//! logical operation the GL enum minus 0x1500 (established).

use super::super::board::Board;
use super::super::bits::sextract32;
use super::super::consts::{alpha, blend, clip, line, mode, pattern, polygon, stencil, write, ODSY_DEPTH_FUNC_MASK, ODSY_LOGIC_OP_MASK};
use super::super::board::ODSY_CLIP_RECTS;
use super::super::clip::ClipRect;
use super::super::view::DestViews;

/// A comparison: bits 2:0 of the depth word 0x420, 14:12 of the alpha
/// word 0x41e, 27:25 of the stencil word 0x41f, the GL enum minus 0x200
/// (established).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Compare {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
}

impl Compare {
    pub fn of(code: u32) -> Compare {
        match code & 7 {
            0 => Compare::Never,
            1 => Compare::Less,
            2 => Compare::Equal,
            3 => Compare::LessEqual,
            4 => Compare::Greater,
            5 => Compare::NotEqual,
            6 => Compare::GreaterEqual,
            _ => Compare::Always,
        }
    }

    /// Whether a passes against b.
    #[inline(always)]
    pub fn test(self, a: u32, b: u32) -> bool {
        match self {
            Compare::Never => false,
            Compare::Less => a < b,
            Compare::Equal => a == b,
            Compare::LessEqual => a <= b,
            Compare::Greater => a > b,
            Compare::NotEqual => a != b,
            Compare::GreaterEqual => a >= b,
            Compare::Always => true,
        }
    }
}

/// A stencil operation, a 3-bit field of the stencil word 0x41f: the
/// board's codes 0 to 5 (established from the library's setter); 6 and 7
/// are unused by the library.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum StencilOp {
    Keep,
    Zero,
    Replace,
    Incr,
    Decr,
    Invert,
    Unknown(u32),
}

impl StencilOp {
    pub fn of(code: u32) -> StencilOp {
        match code & 7 {
            0 => StencilOp::Keep,
            1 => StencilOp::Zero,
            2 => StencilOp::Replace,
            3 => StencilOp::Incr,
            4 => StencilOp::Decr,
            5 => StencilOp::Invert,
            c => StencilOp::Unknown(c),
        }
    }
}

/// A blend factor, the board's code in the blend word 0xc21: bits 5:0 for
/// the destination factor, 12:6 for the source factor (established from the
/// library's setter). `OtherColour` is the other operand's colour, the
/// destination's for the source factor and the source's for the
/// destination factor, as the one code 0x10 means on both sides.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BlendFactor {
    Zero,
    One,
    OtherColour,
    SrcAlpha,
    DstAlpha,
    ConstColour,
    ConstAlpha,
    OneMinusOtherColour,
    OneMinusSrcAlpha,
    OneMinusDstAlpha,
    OneMinusConstColour,
    OneMinusConstAlpha,
    /// Source factor only.
    SrcAlphaSaturate,
    Unknown(u32),
}

impl BlendFactor {
    pub fn of(code: u32) -> BlendFactor {
        match code {
            0x00 => BlendFactor::Zero,
            0x08 => BlendFactor::One,
            0x10 => BlendFactor::OtherColour,
            0x18 => BlendFactor::SrcAlpha,
            0x20 => BlendFactor::DstAlpha,
            0x28 => BlendFactor::ConstColour,
            0x30 => BlendFactor::ConstAlpha,
            0x38 => BlendFactor::OneMinusOtherColour,
            0x39 => BlendFactor::OneMinusSrcAlpha,
            0x3a => BlendFactor::OneMinusDstAlpha,
            0x3b => BlendFactor::OneMinusConstColour,
            0x3c => BlendFactor::OneMinusConstAlpha,
            0x40 => BlendFactor::SrcAlphaSaturate,
            c => BlendFactor::Unknown(c),
        }
    }
}

/// The blend equation, bits 15:13 of the blend word: add, min, max, the
/// SGIX alpha min and max, the logical operation, subtract and reverse
/// subtract (established for the six the library sends; 3 and 4 are the
/// extension's).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BlendEquation {
    Add,
    Min,
    Max,
    AlphaMin,
    AlphaMax,
    LogicOp,
    Subtract,
    ReverseSubtract,
}

impl BlendEquation {
    pub fn of(code: u32) -> BlendEquation {
        match code & 7 {
            0 => BlendEquation::Add,
            1 => BlendEquation::Min,
            2 => BlendEquation::Max,
            3 => BlendEquation::AlphaMin,
            4 => BlendEquation::AlphaMax,
            5 => BlendEquation::LogicOp,
            6 => BlendEquation::Subtract,
            _ => BlendEquation::ReverseSubtract,
        }
    }
}

/// A polygon face's mode, a 2-bit field of the polygon-mode word 0x415
/// (established): point, line, fill; 3 is unused by the library.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PolygonMode {
    Point,
    Line,
    Fill,
    Unknown,
}

impl PolygonMode {
    pub fn of(code: u32) -> PolygonMode {
        match code & 3 {
            0 => PolygonMode::Point,
            1 => PolygonMode::Line,
            2 => PolygonMode::Fill,
            _ => PolygonMode::Unknown,
        }
    }
}

/// Which faces are culled: bits 10:9 of the mode word with bit 8 set
/// (established): 01 back, 10 front, 11 both.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Cull {
    None,
    Back,
    Front,
    Both,
}

/// A 12-bit colour component scaled from a word's 12-bit field.
#[inline(always)]
fn field12(word: u32, shift: u32) -> u16 {
    ((word >> shift) & 0xfff) as u16
}

/// What the rasteriser reads: the primitive's walk, its window and the
/// depth pre-test.
#[derive(Clone, Copy, Default, Debug)]
#[repr(C, align(64))]
pub(crate) struct RasterWords {
    /// 0x404, the raster enable word.
    pub mode: u32,
    /// 0x415.
    pub polygon_mode: u32,
    /// 0x458.
    pub polygon_offset: u32,
    /// 0x81d, two operands.
    pub clip_box: [u32; 2],
    /// 0x841, sign-extended as the board keeps it.
    pub origin: [i32; 2],
    /// 0x204b, four rectangles as the board decoded them, and how many are kept.
    pub clip_rect: [ClipRect; ODSY_CLIP_RECTS as usize],
    pub clip_kept: u32,
    /// 0x840 operand 1.
    pub id_compare: u32,
    /// 0x84a: line width then point size, each x128 in bits 23:0.
    pub line_point: [u32; 2],
    /// 0xc48.
    pub line_stipple: [u32; 3],
    /// 0x3c4e and 0x3c4f: rows 31 down to 0 of the polygon stipple.
    pub pattern: [u32; 32],
    /// 0x43f.
    pub pattern_shape: u32,
    /// 0x420, for the depth pre-test.
    pub depth_func: u32,
}

impl RasterWords {
    pub fn cull(&self) -> Cull {
        if self.mode & mode::CULL_FACE == 0 {
            return Cull::None;
        }
        match (self.mode >> mode::CULL_MODE_SHIFT) & mode::CULL_MODE_MASK {
            1 => Cull::Back,
            2 => Cull::Front,
            3 => Cull::Both,
            _ => Cull::None,
        }
    }

    /// Bit 23: a clockwise winding is the front face.
    pub fn front_cw(&self) -> bool {
        self.mode & mode::FRONT_FACE_CW != 0
    }

    /// Bit 12: flat shading, the provoking vertex's colour.
    pub fn flat(&self) -> bool {
        self.mode & mode::FLAT_SHADE != 0
    }

    pub fn point_smooth(&self) -> bool {
        self.mode & mode::POINT_SMOOTH != 0
    }

    pub fn line_smooth(&self) -> bool {
        self.mode & mode::LINE_SMOOTH != 0
    }

    /// Bit 2 (established).
    pub fn polygon_smooth(&self) -> bool {
        self.mode & (1 << 2) != 0
    }

    pub fn line_stipple_on(&self) -> bool {
        self.mode & mode::GL_LINE_STIPPLE != 0
    }

    /// Bit 4 (established).
    pub fn polygon_stipple_on(&self) -> bool {
        self.mode & (1 << 4) != 0
    }

    /// Bits 5, 6 and 7: the polygon offset applies to polygons drawn as
    /// points, lines and fills.
    pub fn offset_point(&self) -> bool {
        self.mode & mode::OFFSET_POINT != 0
    }

    pub fn offset_line(&self) -> bool {
        self.mode & mode::OFFSET_LINE != 0
    }

    pub fn offset_fill(&self) -> bool {
        self.mode & mode::OFFSET_FILL != 0
    }

    pub fn polygon_mode_front(&self) -> PolygonMode {
        PolygonMode::of(self.polygon_mode >> polygon::MODE_FRONT_SHIFT)
    }

    pub fn polygon_mode_back(&self) -> PolygonMode {
        PolygonMode::of(self.polygon_mode >> polygon::MODE_BACK_SHIFT)
    }

    /// The polygon offset in steps of the 24-bit depth word, bits 24:0
    /// signed (established).
    pub fn offset_steps(&self) -> i32 {
        sextract32(self.polygon_offset, 0, polygon::OFFSET_BITS)
    }

    /// The scissor box as the two operands carry it: x in operand 0 and y
    /// in operand 1, the lower bound in bits 23:12 and the upper in bits
    /// 11:0 (established as two 12-bit fields; which is which follows the
    /// board's own clip-box decoding, `Board::gl_apply_clip_box`).
    pub fn scissor(&self) -> (i32, i32, i32, i32) {
        let lo = |w: u32| ((w >> clip::BOX_MIN_SHIFT) & clip::BOX_FIELD_MASK) as i32;
        let hi = |w: u32| (w & clip::BOX_FIELD_MASK) as i32;

        (lo(self.clip_box[0]), lo(self.clip_box[1]), hi(self.clip_box[0]), hi(self.clip_box[1]))
    }

    /// The line width in pixels times 128 (established).
    pub fn line_width_x128(&self) -> u32 {
        self.line_point[0] & line::POINT_SIZE_MASK
    }

    /// The point size in pixels times 128 (established).
    pub fn point_size_x128(&self) -> u32 {
        self.line_point[1] & line::POINT_SIZE_MASK
    }

    /// The line stipple's 16-bit pattern, bits 15:0 of operand 0 (operand
    /// 1 is the integer 2D path's 32-bit rotated pattern), and its repeat
    /// factor, operand 2's low byte plus one (established as the factor
    /// minus one).
    pub fn line_stipple_pattern(&self) -> u16 {
        (self.line_stipple[0] & 0xffff) as u16
    }

    pub fn line_stipple_repeat(&self) -> u32 {
        (self.line_stipple[line::STIPPLE_CONTROL] & 0xff) + 1
    }

    /// The polygon stipple: bit 4 with a pattern shape of 0 (the shape's
    /// opaque bit aside), and its rows bottom up, as the library converts
    /// them (the pattern blocks hold rows 31 down to 0).
    pub fn polygon_stipple(&self) -> Option<[u32; 32]> {
        (self.polygon_stipple_on() && self.pattern_shape & !pattern::SHAPE_OPAQUE == 0)
            .then(|| std::array::from_fn(|i| self.pattern[31 - i]))
    }

    pub fn depth_func(&self) -> Compare {
        Compare::of(self.depth_func & ODSY_DEPTH_FUNC_MASK)
    }

}

#[cfg(test)]
impl RasterWords {
    /// Whether the window's clip list or the scissor keeps window pixel
    /// (x, y), in plane coordinates.
    pub fn owned(&self, x: i32, y: i32) -> bool {
        let n = self.clip_kept.min(ODSY_CLIP_RECTS) as usize;

        // No rectangle kept: the window owns nothing.
        self.clip_rect[..n].iter().any(|r| x >= r.x0 && x <= r.x1 && y >= r.y0 && y <= r.y1)
    }
}

/// What the shader reads: texturing, fog and the alpha test.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(64))]
pub(crate) struct ShaderWords {
    /// 0x404.
    pub mode: u32,
    /// 0x181a.
    pub fog: [u32; 6],
    /// 0x41e.
    pub alpha: u32,
}

impl ShaderWords {
    pub fn fog_on(&self) -> bool {
        self.mode & mode::FOG != 0
    }

    pub fn alpha_test_on(&self) -> bool {
        self.mode & mode::ALPHA_TEST != 0
    }

    /// The fog colour, two 12-bit components a word, R G then B A
    /// (established).
    pub fn fog_colour(&self) -> [u16; 4] {
        [field12(self.fog[0], 12), field12(self.fog[0], 0), field12(self.fog[1], 12), field12(self.fog[1], 0)]
    }

    pub fn alpha_func(&self) -> Compare {
        Compare::of((self.alpha >> alpha::FUNC_SHIFT) & alpha::FUNC_MASK)
    }

    /// The alpha reference, bits 11:0: the value times 4096 with 1.0 sent
    /// as 4095 (established).
    pub fn alpha_ref(&self) -> u16 {
        (self.alpha & alpha::REF_MASK) as u16
    }
}

/// What the back end reads: the tests, the blend, the masks and the
/// planes they write.
#[derive(Clone, Copy, Default, Debug)]
#[repr(C, align(64))]
pub(crate) struct BackendWords {
    /// 0x404.
    pub mode: u32,
    /// 0x420.
    pub depth_func: u32,
    /// 0x452.
    pub write_mask: u32,
    /// 0x41f.
    pub stencil: u32,
    /// 0xc21: the factors and equation, then the constant colour R G, B A.
    pub blend: [u32; 3],
    /// 0x422.
    pub logic_op: u32,
    /// 0x1017's select operand.
    pub dest_select: u32,
    /// 0x1023.
    pub plane_mask: u32,
    /// 0x424: bit b selects colour buffer b (the displayed-buffer sense).
    pub draw_mask: u32,
    /// The colour-mode words, class 0x22 then 0x23, as last sent.
    pub colour_mode: [u32; 2],
    /// Where the draw lands: the pixel mode's fields over board memory.
    pub dest: DestViews,
}

impl BackendWords {
    pub fn depth_test_on(&self) -> bool {
        self.mode & mode::DEPTH_TEST != 0
    }

    pub fn depth_func(&self) -> Compare {
        Compare::of(self.depth_func & ODSY_DEPTH_FUNC_MASK)
    }

    /// Bit 24 of the write mask (established).
    pub fn depth_write(&self) -> bool {
        self.write_mask & write::DEPTH != 0
    }

    pub fn stencil_test_on(&self) -> bool {
        self.mode & mode::STENCIL_TEST != 0
    }

    pub fn stencil_func(&self) -> Compare {
        Compare::of((self.stencil >> stencil::FUNC_SHIFT) & stencil::FUNC_MASK)
    }

    pub fn stencil_ref(&self) -> u32 {
        self.stencil & stencil::REF_MASK
    }

    pub fn stencil_value_mask(&self) -> u32 {
        (self.stencil >> stencil::VALUE_MASK_SHIFT) & stencil::VALUE_MASK_MASK
    }

    /// The operations on stencil fail, depth fail and pass.
    pub fn stencil_ops(&self) -> [StencilOp; 3] {
        [
            StencilOp::of(self.stencil >> stencil::OP_FAIL_SHIFT),
            StencilOp::of(self.stencil >> stencil::OP_ZFAIL_SHIFT),
            StencilOp::of(self.stencil >> stencil::OP_ZPASS_SHIFT),
        ]
    }

    /// Bits 19:12 of the write mask (established).
    pub fn stencil_write_mask(&self) -> u32 {
        (self.write_mask >> write::STENCIL_SHIFT) & write::STENCIL_MASK
    }

    /// Bits 23:20 of the write mask: the colour mask's four bits
    /// (established as the field; the bit order is the GL mask's).
    pub fn colour_mask_bits(&self) -> u32 {
        (self.write_mask >> write::COLOUR_SHIFT) & write::COLOUR_MASK
    }

    /// Bits 11:0: the index mask (established).
    pub fn index_mask(&self) -> u32 {
        self.write_mask & write::INDEX_MASK
    }

    /// Bit 17: blending in use, which the library sets only with the
    /// colour logic op off (established).
    pub fn blend_on(&self) -> bool {
        self.mode & mode::BLEND != 0
    }

    pub fn blend_src(&self) -> BlendFactor {
        BlendFactor::of((self.blend[0] >> blend::SRC_SHIFT) & blend::SRC_MASK)
    }

    pub fn blend_dst(&self) -> BlendFactor {
        BlendFactor::of(self.blend[0] & blend::DST_MASK)
    }

    pub fn blend_equation(&self) -> BlendEquation {
        BlendEquation::of((self.blend[0] >> blend::EQ_SHIFT) & blend::EQ_MASK)
    }

    /// The constant colour, 12 bits a component (established).
    pub fn const_colour(&self) -> [u16; 4] {
        [field12(self.blend[1], 12), field12(self.blend[1], 0), field12(self.blend[2], 12), field12(self.blend[2], 0)]
    }

    pub fn colour_logic_op_on(&self) -> bool {
        self.mode & mode::COLOR_LOGIC_OP != 0
    }

    pub fn index_logic_op_on(&self) -> bool {
        self.mode & mode::INDEX_LOGIC_OP != 0
    }

    /// Bits 3:0: the GL logical operation minus 0x1500, GL_CLEAR 0 to
    /// GL_SET 15 (established).
    pub fn logic_op(&self) -> u32 {
        self.logic_op & ODSY_LOGIC_OP_MASK
    }
}

/// The three records of one draw.
#[derive(Clone, Copy, Default, Debug)]
#[repr(C, align(64))]
pub(crate) struct FragmentWords {
    pub raster: RasterWords,
    pub shader: ShaderWords,
    pub backend: BackendWords,
}

impl Board {
    /// The fragment pipeline's state as the board holds it now.
    pub fn fragment_words(&self) -> FragmentWords {
        let m = self.stream.pattern_mode;
        let raster = RasterWords {
            mode: m,
            polygon_mode: self.polygon_mode_word,
            polygon_offset: self.polygon_offset_word,
            clip_box: self.gl_clip_box,
            origin: self.win.origin,
            clip_rect: self.win.clip_rect,
            clip_kept: self.win.clip_kept,
            id_compare: self.win.id_compare,
            line_point: self.line_point_words,
            line_stipple: self.stream.line_stipple,
            pattern: self.stream.pattern_buffer,
            pattern_shape: self.stream.pattern_shape,
            depth_func: self.gl_depth_func,
        };
        let shader = ShaderWords {
            mode: m,
            fog: self.gl_fog,
            alpha: self.alpha_word,
        };
        let backend = BackendWords {
            mode: m,
            depth_func: self.gl_depth_func,
            write_mask: self.gl_write_mask,
            stencil: self.stencil_word,
            blend: [self.blend_word, self.const_colour[0], self.const_colour[1]],
            logic_op: self.logic_op_word,
            dest_select: self.dest_buffer_word,
            plane_mask: self.plane_write_mask,
            draw_mask: self.draw_mask,
            colour_mode: self.colour_mode,
            dest: self.dest_views(),
        };
        FragmentWords { raster, shader, backend }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mode_word_bits_land_in_each_stage() {
        let m = mode::DEPTH_TEST | mode::STENCIL_TEST | mode::BLEND | mode::FOG | mode::ALPHA_TEST | mode::FLAT_SHADE
            | mode::DITHER | mode::CULL_FACE | (2 << mode::CULL_MODE_SHIFT) | mode::FRONT_FACE_CW | mode::OFFSET_FILL
            | (1 << 2) | (1 << 4);
        let r = RasterWords { mode: m, ..Default::default() };
        let s = ShaderWords { mode: m, ..Default::default() };
        let b = BackendWords { mode: m, ..Default::default() };

        assert_eq!(r.cull(), Cull::Front);
        assert!(r.front_cw() && r.flat() && r.offset_fill() && r.polygon_smooth() && r.polygon_stipple_on());
        assert!(!r.offset_point() && !r.line_smooth());
        assert!(s.fog_on() && s.alpha_test_on());
        assert!(b.depth_test_on() && b.stencil_test_on() && b.blend_on());
        assert!(!b.colour_logic_op_on());
        assert_eq!(RasterWords { mode: mode::CULL_FACE | (1 << mode::CULL_MODE_SHIFT), ..Default::default() }.cull(), Cull::Back);
        assert_eq!(RasterWords { mode: 3 << mode::CULL_MODE_SHIFT, ..Default::default() }.cull(), Cull::None);
    }

    #[test]
    fn the_test_words_decode_as_the_notes_establish() {
        // The odydiags alpha test: LEQUAL at 0.5.
        let s = ShaderWords { alpha: (3 << 12) | 0x800, ..Default::default() };
        assert_eq!(s.alpha_func(), Compare::LessEqual);
        assert_eq!(s.alpha_ref(), 0x800);
        // Depth: GL_LESS is 1.
        let b = BackendWords { depth_func: 1, write_mask: write::DEPTH | (0xf << 20) | (0xa5 << 12) | 0x123, ..Default::default() };
        assert_eq!(b.depth_func(), Compare::Less);
        assert!(b.depth_write());
        assert_eq!(b.colour_mask_bits(), 0xf);
        assert_eq!(b.stencil_write_mask(), 0xa5);
        assert_eq!(b.index_mask(), 0x123);
        // Stencil: GEQUAL, ref 0x7f, value mask 0xf0, ops KEEP / INCR / REPLACE.
        let word = (6 << 25) | (3 << 19) | (2 << 16) | (0xf0 << 8) | 0x7f; // fail op 0 (KEEP) in bits 24:22
        let b = BackendWords { stencil: word, ..Default::default() };
        assert_eq!(b.stencil_func(), Compare::GreaterEqual);
        assert_eq!(b.stencil_ref(), 0x7f);
        assert_eq!(b.stencil_value_mask(), 0xf0);
        assert_eq!(b.stencil_ops(), [StencilOp::Keep, StencilOp::Incr, StencilOp::Replace]);
        assert_eq!(StencilOp::of(7), StencilOp::Unknown(7));
        assert!(Compare::LessEqual.test(3, 3) && !Compare::Less.test(3, 3) && Compare::NotEqual.test(1, 2));
    }

    #[test]
    fn the_blend_words_decode_as_the_notes_establish() {
        // SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ADD, with a constant colour.
        let w0 = (0x18 << 6) | 0x39;
        let b = BackendWords { blend: [w0, (0x123 << 12) | 0x456, (0x789 << 12) | 0xabc], logic_op: 0x3, ..Default::default() };
        assert_eq!(b.blend_src(), BlendFactor::SrcAlpha);
        assert_eq!(b.blend_dst(), BlendFactor::OneMinusSrcAlpha);
        assert_eq!(b.blend_equation(), BlendEquation::Add);
        assert_eq!(b.const_colour(), [0x123, 0x456, 0x789, 0xabc]);
        assert_eq!(b.logic_op(), 3);
        assert_eq!(BlendFactor::of(0x10), BlendFactor::OtherColour);
        assert_eq!(BlendFactor::of(0x40), BlendFactor::SrcAlphaSaturate);
        assert_eq!(BlendFactor::of(0x41), BlendFactor::Unknown(0x41));
        assert_eq!(BlendEquation::of(6), BlendEquation::Subtract);
        assert_eq!(BlendEquation::of(3), BlendEquation::AlphaMin);
    }

    #[test]
    fn the_raster_words_decode_as_the_notes_establish() {
        let r = RasterWords {
            polygon_mode: (2 << 2) | 1,
            polygon_offset: 0x1ff_ffff, // -1 in 25 bits
            clip_box: [(10 << 12) | 1279, (20 << 12) | 1023],
            line_point: [3 * 128, 5 * 128 + 64],
            line_stipple: [0xaaaa, 0xf0f0_aaaa, 7],
            depth_func: 3,
            ..Default::default()
        };
        assert_eq!(r.polygon_mode_front(), PolygonMode::Fill);
        assert_eq!(r.polygon_mode_back(), PolygonMode::Line);
        assert_eq!(r.offset_steps(), -1);
        assert_eq!(r.scissor(), (10, 20, 1279, 1023));
        assert_eq!(r.line_width_x128(), 384);
        assert_eq!(r.point_size_x128(), 704);
        assert_eq!(r.line_stipple_pattern(), 0xaaaa);
        assert_eq!(r.line_stipple_repeat(), 8);
        assert_eq!(r.depth_func(), Compare::LessEqual);
        assert!(!r.owned(5, 5), "no rectangle kept owns nothing");
        let mut r = r;
        r.clip_kept = 1;
        r.clip_rect[0] = ClipRect { x0: 100, y0: 100, x1: 200, y1: 200 };
        assert!(r.owned(150, 150) && !r.owned(99, 150));
    }

    #[test]
    fn the_shader_words_decode_as_the_notes_establish() {
        let s = ShaderWords { fog: [(0xfff << 12) | 0x800, (0x400 << 12) | 0x200, 0, 0, 0, 0], ..Default::default() };
        assert_eq!(s.fog_colour(), [0xfff, 0x800, 0x400, 0x200]);
    }

    #[test]
    fn the_texture_registers_decode_as_the_notes_establish() {
        use super::super::texture::{Env, Target, TexRegs};

        let mut t = TexRegs { path: (1 << 10) | (1 << 4) | 0b110, ..Default::default() };
        t.env = [4, (0x111 << 12) | 0x222, (0x333 << 12) | 0x444, 0, 0];
        assert_eq!(t.target(), Target::TwoD);
        assert_eq!(t.clip_planes(), 2);
        assert_eq!(t.env(), Env::Blend);
        assert_eq!(t.env_colour(), [0x111, 0x222, 0x333, 0x444]);
        t.path = 1 << 10;
        assert_eq!(t.target(), Target::None);
        t.path = (1 << 10) | (1 << 5) | (1 << 4);
        assert_eq!(t.target(), Target::ThreeD);
    }
}
