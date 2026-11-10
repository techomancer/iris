// generated from odsy 4e187ebeace7 BUZZ_DECODE.h - do not edit, regenerate in odsy
//!
//! Command opcodes and the image, copy, pixel-mode, texture-level and sampler
//! words of the Odyssey geometry and pixel engine's command stream.
//!
//! Grades on each item say what is known about it: `established` is what the
//! shipped driver software does, read in full; `inferred` is a reasoned
//! reading; `unknown` is not derived. None of them is a measurement of the
//! silicon.

#![allow(dead_code)]

/// established.
pub const CMD_CLASS1_PREFIX: u32 = 0x10000;
/// established.
pub const OP_COPY: u32 = 0x2031;
pub const OP_COPY_WORDS: u32 = 8;
/// established.
pub const OP_POSTFILTER: u32 = 0x203b;
pub const OP_POSTFILTER_WORDS: u32 = 8;
pub const TEXLEVEL_COUNT: u32 = 64;
pub const OP_TEXMODE: u32 = 0x1c16;
pub const RD_CODE_INDEX_BYTE: u32 = 1;
pub const HDR_W7_TYPE_INPLACE: u32 = 0x1e0;
pub const PF_FRAC_BITS: u32 = 12;
/// established.
pub const PF_ONE: u32 = 0x1000;
pub const PF_MAX_X4096: u32 = 0x7fff;

/// `op_texlevel(i)` is the opcode of descriptor `i`: `0x0e80` plus `i`.
pub const fn op_texlevel(i: u32) -> u32 {
    0x0e80 + i
}

/// Write header format code (bits 4:0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteFormat {
    /// GL_STENCIL_INDEX, GL_DEPTH_COMPONENT.
    DepthStencil,
    /// GL_COLOR_INDEX.
    Index,
    /// GL_LUMINANCE; GL_ALPHA into dest class 0/3.
    Luminance,
    /// GL_INTENSITY.
    Intensity,
    /// GL_RED.
    Red,
    /// GL_GREEN.
    Green,
    /// GL_BLUE.
    Blue,
    /// GL_ALPHA into other destination classes.
    Alpha,
    /// GL_LUMINANCE_ALPHA.
    LumAlpha,
    /// GL_RGB.
    Rgb,
    /// GL_RGBA; 0xc with packed REV types.
    Rgba,
    /// GL_ABGR_EXT; 0xb with 4_4_4_4_REV or 8_8_8_8_REV.
    Abgr,
    /// GL_BGR; 9 with 5_6_5_REV.
    Bgr,
    /// GL_BGRA; 0x10 with the four REV types.
    Bgra,
    /// A value the notes do not name.
    Other(u8),
}

impl WriteFormat {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::DepthStencil,
            0x1 => Self::Index,
            0x2 => Self::Luminance,
            0x3 => Self::Intensity,
            0x4 => Self::Red,
            0x5 => Self::Green,
            0x6 => Self::Blue,
            0x7 => Self::Alpha,
            0x8 => Self::LumAlpha,
            0x9 => Self::Rgb,
            0xb => Self::Rgba,
            0xc => Self::Abgr,
            0xd => Self::Bgr,
            0x11 => Self::Bgra,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::DepthStencil => 0x0,
            Self::Index => 0x1,
            Self::Luminance => 0x2,
            Self::Intensity => 0x3,
            Self::Red => 0x4,
            Self::Green => 0x5,
            Self::Blue => 0x6,
            Self::Alpha => 0x7,
            Self::LumAlpha => 0x8,
            Self::Rgb => 0x9,
            Self::Rgba => 0xb,
            Self::Abgr => 0xc,
            Self::Bgr => 0xd,
            Self::Bgra => 0x11,
            Self::Other(v) => v as u32,
        }
    }
}

/// Write header type code (bits 8:5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteType {
    /// GL_BYTE, GL_UNSIGNED_BYTE, 8_8_8_8 and REV.
    Byte,
    /// short types.
    Short,
    /// int types; the only depth/stencil source types.
    Int,
    /// UNSIGNED_BYTE_3_3_2 and 2_3_3_REV.
    Packed332,
    /// UNSIGNED_SHORT_4_4_4_4 and REV.
    Packed4444,
    /// UNSIGNED_SHORT_5_5_5_1 and REV.
    Packed5551,
    /// UNSIGNED_INT_10_10_10_2 and REV.
    Packed1010102,
    /// UNSIGNED_SHORT_5_6_5 and REV.
    Packed565,
    /// byte types into a 12-bit-per-channel dest.
    Byte12,
    /// 5_6_5 types into destination class 10.
    Packed565Class10,
    /// A value the notes do not name.
    Other(u8),
}

impl WriteType {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x2 => Self::Byte,
            0x3 => Self::Short,
            0x4 => Self::Int,
            0x5 => Self::Packed332,
            0x6 => Self::Packed4444,
            0x7 => Self::Packed5551,
            0x8 => Self::Packed1010102,
            0xa => Self::Packed565,
            0xb => Self::Byte12,
            0xc => Self::Packed565Class10,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Byte => 0x2,
            Self::Short => 0x3,
            Self::Int => 0x4,
            Self::Packed332 => 0x5,
            Self::Packed4444 => 0x6,
            Self::Packed5551 => 0x7,
            Self::Packed1010102 => 0x8,
            Self::Packed565 => 0xa,
            Self::Byte12 => 0xb,
            Self::Packed565Class10 => 0xc,
            Self::Other(v) => v as u32,
        }
    }
}

/// Pixel-size code (three bits, 18:16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelSize {
    /// 2 bytes - established.
    Bytes2,
    /// 4 bytes - established.
    Bytes4,
    /// 8 bytes - established.
    Bytes8,
    /// 16 bytes - established.
    Bytes16,
    /// 12 bytes - established.
    Bytes12,
    /// 1 byte - established.
    Bytes1,
    /// 6 bytes - established.
    Bytes6,
    /// 3 bytes - established.
    Bytes3,
    /// A value the notes do not name.
    Other(u8),
}

impl PixelSize {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::Bytes2,
            0x1 => Self::Bytes4,
            0x2 => Self::Bytes8,
            0x3 => Self::Bytes16,
            0x4 => Self::Bytes12,
            0x5 => Self::Bytes1,
            0x6 => Self::Bytes6,
            0x7 => Self::Bytes3,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Bytes2 => 0x0,
            Self::Bytes4 => 0x1,
            Self::Bytes8 => 0x2,
            Self::Bytes16 => 0x3,
            Self::Bytes12 => 0x4,
            Self::Bytes1 => 0x5,
            Self::Bytes6 => 0x6,
            Self::Bytes3 => 0x7,
            Self::Other(v) => v as u32,
        }
    }
}

/// Pixel-mode field format (bits 6:4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldFormat {
    /// classes 16, 17 (colour index).
    Index,
    /// classes 0-5: one channel.
    One,
    /// classes 6, 9: two channels.
    Two,
    /// classes 7, 10, 12, 13.
    Rgba,
    /// classes 8, 11: short alpha.
    RgbaShortA,
    /// A value the notes do not name.
    Other(u8),
}

impl FieldFormat {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::Index,
            0x2 => Self::One,
            0x3 => Self::Two,
            0x5 => Self::Rgba,
            0x6 => Self::RgbaShortA,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Index => 0x0,
            Self::One => 0x2,
            Self::Two => 0x3,
            Self::Rgba => 0x5,
            Self::RgbaShortA => 0x6,
            Self::Other(v) => v as u32,
        }
    }
}

/// Sampler minification filter (bits 7:5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerMin {
    /// established.
    Nearest,
    /// established.
    Linear,
    /// established.
    NearestMipNearest,
    /// established.
    LinearMipNearest,
    /// established.
    NearestMipLinear,
    /// established.
    LinearMipLinear,
    /// A value the notes do not name.
    Other(u8),
}

impl SamplerMin {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::Nearest,
            0x1 => Self::Linear,
            0x2 => Self::NearestMipNearest,
            0x3 => Self::LinearMipNearest,
            0x4 => Self::NearestMipLinear,
            0x5 => Self::LinearMipLinear,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Nearest => 0x0,
            Self::Linear => 0x1,
            Self::NearestMipNearest => 0x2,
            Self::LinearMipNearest => 0x3,
            Self::NearestMipLinear => 0x4,
            Self::LinearMipLinear => 0x5,
            Self::Other(v) => v as u32,
        }
    }
}

/// Sampler magnification mode (bits 4:1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerMag {
    /// established.
    Nearest,
    /// established.
    Linear,
    /// established.
    Detail,
    /// established.
    DetailAlpha,
    /// established.
    DetailColor,
    /// A value the notes do not name.
    Other(u8),
}

impl SamplerMag {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::Nearest,
            0x1 => Self::Linear,
            0x2 => Self::Detail,
            0x3 => Self::DetailAlpha,
            0x4 => Self::DetailColor,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Nearest => 0x0,
            Self::Linear => 0x1,
            Self::Detail => 0x2,
            Self::DetailAlpha => 0x3,
            Self::DetailColor => 0x4,
            Self::Other(v) => v as u32,
        }
    }
}

/// Sampler wrap mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerWrap {
    /// established.
    Repeat,
    /// established: clamp to edge.
    Edge,
    /// established.
    Clamp,
    /// established: clamp to border.
    Border,
    /// A value the notes do not name.
    Other(u8),
}

impl SamplerWrap {
    pub const fn from_bits(v: u32) -> Self {
        match v {
            0x0 => Self::Repeat,
            0x1 => Self::Edge,
            0x2 => Self::Clamp,
            0x3 => Self::Border,
            _ => Self::Other(v as u8),
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::Repeat => 0x0,
            Self::Edge => 0x1,
            Self::Clamp => 0x2,
            Self::Border => 0x3,
            Self::Other(v) => v as u32,
        }
    }
}

/// Image header word 0: board reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageWord0(u32);

impl ImageWord0 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 28:0, established.
    pub const fn ref_(self) -> u32 {
        self.0 & 0x1fffffff
    }

    /// Bit 29, established.
    pub const fn window_rel(self) -> bool {
        (self.0 >> 29) & 1 != 0
    }

    /// Bits 31:30, inferred.
    pub const fn kept(self) -> u32 {
        self.0 >> 30
    }
}

/// Image header word 1: x and y of the smaller corner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageWord1(u32);

impl ImageWord1 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 15:0, established.
    pub const fn x(self) -> u32 {
        self.0 & 0xffff
    }

    /// Bits 31:16, established.
    pub const fn y(self) -> u32 {
        self.0 >> 16
    }
}

/// Image header word 3: width and height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageWord3(u32);

impl ImageWord3 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 15:0, established.
    pub const fn width(self) -> u32 {
        self.0 & 0xffff
    }

    /// Bits 31:16, established.
    pub const fn height(self) -> u32 {
        self.0 >> 16
    }
}

/// Image header word 2: the window word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageWord2(u32);

impl ImageWord2 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 11:0, established.
    pub const fn window_word(self) -> u32 {
        self.0 & 0xfff
    }
}

/// Image header word 6: row bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageWord6(u32);

impl ImageWord6 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 19:0, established.
    pub const fn row_bytes(self) -> u32 {
        self.0 & 0xfffff
    }
}

/// Image header word 4: class and code, field geometry, pixel size.
/// The read header's word 4, and the COPY source description.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadHeader(u32);

impl ReadHeader {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 2:0, established.
    pub const fn class(self) -> u32 {
        self.0 & 0x7
    }

    /// Bits 7:3, established.
    pub const fn code(self) -> u32 {
        (self.0 >> 3) & 0x1f
    }

    /// Bit 12, established.
    pub const fn bit12(self) -> bool {
        (self.0 >> 12) & 1 != 0
    }

    /// Bits 15:13, established.
    pub const fn size_field(self) -> u32 {
        (self.0 >> 13) & 0x7
    }

    /// Bits 18:16, inferred.
    pub const fn pixel_size(self) -> PixelSize {
        PixelSize::from_bits((self.0 >> 16) & 0x7)
    }

    /// Bit 19, unknown.
    pub const fn pixel_size_b19(self) -> bool {
        (self.0 >> 19) & 1 != 0
    }

    /// Bits 23:20, established.
    pub const fn field_a(self) -> u32 {
        (self.0 >> 20) & 0xf
    }

    /// Bits 27:24, established.
    pub const fn field_b(self) -> u32 {
        (self.0 >> 24) & 0xf
    }

    /// Bit 28, established.
    pub const fn drawable(self) -> bool {
        (self.0 >> 28) & 1 != 0
    }

    /// Bit 31, established.
    pub const fn front(self) -> bool {
        (self.0 >> 31) & 1 != 0
    }
}

/// Image header word 7: format, type, size and path control.
/// The write header's word 7, and the COPY destination control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WriteHeader(u32);

impl WriteHeader {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 4:0, established.
    pub const fn format(self) -> WriteFormat {
        WriteFormat::from_bits(self.0 & 0x1f)
    }

    /// Bits 8:5, established.
    pub const fn type_(self) -> WriteType {
        WriteType::from_bits((self.0 >> 5) & 0xf)
    }

    /// Bit 9, established.
    pub const fn expand(self) -> bool {
        (self.0 >> 9) & 1 != 0
    }

    /// Bit 10, inferred.
    pub const fn swap(self) -> bool {
        (self.0 >> 10) & 1 != 0
    }

    /// Bits 15:13, established.
    pub const fn size_field(self) -> u32 {
        (self.0 >> 13) & 0x7
    }

    /// Bits 18:16, established.
    pub const fn psize(self) -> PixelSize {
        PixelSize::from_bits((self.0 >> 16) & 0x7)
    }

    /// Bit 24, established.
    pub const fn read_dir(self) -> bool {
        (self.0 >> 24) & 1 != 0
    }

    /// Bit 27, established.
    pub const fn transform(self) -> bool {
        (self.0 >> 27) & 1 != 0
    }

    /// Bit 30, established.
    pub const fn differs(self) -> bool {
        (self.0 >> 30) & 1 != 0
    }
}

/// COPY operand 2: the two window words and the destination's second field code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyMixed(u32);

impl CopyMixed {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 11:0, established.
    pub const fn src_window(self) -> u32 {
        self.0 & 0xfff
    }

    /// Bits 15:12, established.
    pub const fn dst_field_b(self) -> u32 {
        (self.0 >> 12) & 0xf
    }

    /// Bits 27:16, established.
    pub const fn dst_window(self) -> u32 {
        (self.0 >> 16) & 0xfff
    }

    /// Bit 31, established.
    pub const fn dst_front(self) -> bool {
        (self.0 >> 31) & 1 != 0
    }
}

/// COPY operand 7 beyond the write header: flags set around the walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyControl(u32);

impl CopyControl {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 23:19, unknown.
    pub const fn reserved_23_19(self) -> u32 {
        (self.0 >> 19) & 0x1f
    }

    /// Bits 29:28, inferred.
    pub const fn clamp_off(self) -> u32 {
        (self.0 >> 28) & 0x3
    }

    /// Bit 30, inferred.
    pub const fn dst_y_in_src(self) -> bool {
        (self.0 >> 30) & 1 != 0
    }

    /// Bit 31, inferred.
    pub const fn dst_x_in_src(self) -> bool {
        (self.0 >> 31) & 1 != 0
    }
}

/// A scale or bias word of the post-filter block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostFilterWord(u32);

impl PostFilterWord {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 15:0, established.
    pub const fn value(self) -> u32 {
        self.0 & 0xffff
    }
}

/// Pixel-mode descriptor, second word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelModeW1(u32);

impl PixelModeW1 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 6:4, established.
    pub const fn field_format(self) -> FieldFormat {
        FieldFormat::from_bits((self.0 >> 4) & 0x7)
    }

    /// Bits 9:7, established.
    pub const fn layout(self) -> u32 {
        (self.0 >> 7) & 0x7
    }

    /// Bits 12:10, established.
    pub const fn storage(self) -> u32 {
        (self.0 >> 10) & 0x7
    }

    /// Bits 15:13, established.
    pub const fn size_field(self) -> u32 {
        (self.0 >> 13) & 0x7
    }

    /// Bit 19, established.
    pub const fn index_class(self) -> bool {
        (self.0 >> 19) & 1 != 0
    }

    /// Bit 21, unknown.
    pub const fn bit21(self) -> bool {
        (self.0 >> 21) & 1 != 0
    }

    /// Bit 22, inferred.
    pub const fn storage3(self) -> bool {
        (self.0 >> 22) & 1 != 0
    }

    /// Bit 27, unknown.
    pub const fn bit27(self) -> bool {
        (self.0 >> 27) & 1 != 0
    }
}

/// Pixel-mode descriptor, first word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PixelModeW0(u32);

impl PixelModeW0 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 11:0, established.
    pub const fn window_word(self) -> u32 {
        self.0 & 0xfff
    }
}

/// Texture level descriptor, word 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexLevelW0(u32);

impl TexLevelW0 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 28:0, established.
    pub const fn addr(self) -> u32 {
        self.0 & 0x1fffffff
    }

    /// Bits 31:29, unknown.
    pub const fn kept(self) -> u32 {
        self.0 >> 29
    }
}

/// Texture level descriptor, word 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexLevelW1(u32);

impl TexLevelW1 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 11:0, established.
    pub const fn pitch(self) -> u32 {
        self.0 & 0xfff
    }

    /// Bits 25:20, established.
    pub const fn format_index(self) -> u32 {
        (self.0 >> 20) & 0x3f
    }

    /// Bits 28:26, established.
    pub const fn offset_code(self) -> u32 {
        (self.0 >> 26) & 0x7
    }

    /// Bits 31:30, established.
    pub const fn size_code(self) -> u32 {
        self.0 >> 30
    }
}

/// Texture level descriptor, word 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexLevelW2(u32);

impl TexLevelW2 {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bits 11:0, established.
    pub const fn bits(self) -> u32 {
        self.0 & 0xfff
    }
}

/// The sampler word: operand 4 of the texture-mode block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerWord(u32);

impl SamplerWord {
    pub const fn new(word: u32) -> Self {
        Self(word)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Bit 0, established.
    pub const fn colour_table(self) -> bool {
        self.0 & 1 != 0
    }

    /// Bits 4:1, established.
    pub const fn mag(self) -> SamplerMag {
        SamplerMag::from_bits((self.0 >> 1) & 0xf)
    }

    /// Bits 7:5, established.
    pub const fn min(self) -> SamplerMin {
        SamplerMin::from_bits((self.0 >> 5) & 0x7)
    }

    /// Bits 9:8, established.
    pub const fn wrap_r(self) -> SamplerWrap {
        SamplerWrap::from_bits((self.0 >> 8) & 0x3)
    }

    /// Bits 11:10, established.
    pub const fn wrap_t(self) -> SamplerWrap {
        SamplerWrap::from_bits((self.0 >> 10) & 0x3)
    }

    /// Bits 13:12, established.
    pub const fn wrap_s(self) -> SamplerWrap {
        SamplerWrap::from_bits((self.0 >> 12) & 0x3)
    }

    /// Bit 14, established.
    pub const fn image_flag14(self) -> bool {
        (self.0 >> 14) & 1 != 0
    }

    /// Bit 17, established.
    pub const fn trilinear(self) -> bool {
        (self.0 >> 17) & 1 != 0
    }

    /// Bit 18, established.
    pub const fn image_flag18(self) -> bool {
        (self.0 >> 18) & 1 != 0
    }

    /// Bits 22:20, established.
    pub const fn table_fmt(self) -> u32 {
        (self.0 >> 20) & 0x7
    }

    /// Bits 25:23, established.
    pub const fn activation(self) -> u32 {
        (self.0 >> 23) & 0x7
    }

    /// Bit 28, established.
    pub const fn no_offset(self) -> bool {
        (self.0 >> 28) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testvec_hdr_w7_41() {
        let w = WriteHeader::new(0x41);
        assert_eq!(w.format().bits(), 1, "format");
        assert_eq!(w.type_().bits(), 2, "type");
        assert_eq!(w.expand() as u32, 0, "expand");
    }

    #[test]
    fn testvec_hdr_w7_42() {
        let w = WriteHeader::new(0x42);
        assert_eq!(w.format().bits(), 2, "format");
        assert_eq!(w.type_().bits(), 2, "type");
        assert_eq!(w.expand() as u32, 0, "expand");
    }

    #[test]
    fn testvec_hdr_w7_4b() {
        let w = WriteHeader::new(0x4b);
        assert_eq!(w.format().bits(), 11, "format");
        assert_eq!(w.type_().bits(), 2, "type");
        assert_eq!(w.expand() as u32, 0, "expand");
    }

    #[test]
    fn testvec_hdr_w7_61() {
        let w = WriteHeader::new(0x61);
        assert_eq!(w.format().bits(), 1, "format");
        assert_eq!(w.type_().bits(), 3, "type");
        assert_eq!(w.expand() as u32, 0, "expand");
    }

    #[test]
    fn testvec_hdr_w7_6b() {
        let w = WriteHeader::new(0x6b);
        assert_eq!(w.format().bits(), 11, "format");
        assert_eq!(w.type_().bits(), 3, "type");
        assert_eq!(w.expand() as u32, 0, "expand");
    }

    #[test]
    fn testvec_hdr_w7_16c() {
        let w = WriteHeader::new(0x16c);
        assert_eq!(w.format().bits(), 12, "format");
        assert_eq!(w.type_().bits(), 11, "type");
        assert_eq!(w.expand() as u32, 0, "expand");
    }

    #[test]
    fn testvec_hdr_w7_32050041() {
        let w = WriteHeader::new(0x32050041);
        assert_eq!(w.format().bits(), 1, "format");
        assert_eq!(w.type_().bits(), 2, "type");
        assert_eq!(w.psize().bits(), 5, "psize");
        assert_eq!(w.read_dir() as u32, 0, "read_dir");
        assert_eq!(w.transform() as u32, 0, "transform");
        assert_eq!(w.differs() as u32, 0, "differs");
    }

    #[test]
    fn testvec_hdr_w4_8() {
        let w = ReadHeader::new(0x8);
        assert_eq!(w.class(), 0, "class");
        assert_eq!(w.code(), 1, "code");
    }

    #[test]
    fn testvec_hdr_w4_83331008() {
        let w = ReadHeader::new(0x83331008);
        assert_eq!(w.class(), 0, "class");
        assert_eq!(w.code(), 1, "code");
        assert_eq!(w.bit12() as u32, 1, "bit12");
        assert_eq!(w.size_field(), 0, "size_field");
        assert_eq!(w.pixel_size().bits(), 3, "pixel_size");
        assert_eq!(w.field_a(), 3, "field_a");
        assert_eq!(w.field_b(), 3, "field_b");
        assert_eq!(w.drawable() as u32, 0, "drawable");
        assert_eq!(w.front() as u32, 1, "front");
    }

    #[test]
    fn testvec_sampler_82() {
        let w = SamplerWord::new(0x82);
        assert_eq!(w.colour_table() as u32, 0, "colour_table");
        assert_eq!(w.mag().bits(), 1, "mag");
        assert_eq!(w.min().bits(), 4, "min");
    }
}
