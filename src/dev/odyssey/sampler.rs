//! Texture setup: the board's texture words decoded into one sampler
//! description, the check of what the model can draw from it, and the level
//! chain built from that description.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.
//!
//! The words are the sampler block (0x1c16, seven words), the LOD block
//! (0x1013), the level table (0x0e80 + i, three words an entry), the first
//! level index (block 0x006), the path selection word's target bits (block
//! 0x005) and the environment block (0x141b). The field layout is what the
//! GL library writes (ESTABLISHED as the library's setters unless a comment
//! says otherwise). What the board does with each field is mostly UNKNOWN:
//! the model samples with OpenGL's rules through the shared sampler, and a
//! comment marks each place where it picks a reading the library alone does
//! not settle.

use crate::devlog::LogModule;
use super::board::{Board, ODSY_OBJECTS, ODSY_OBJ_POOL_WORDS, ODSY_TEX_LEVELS, ODSY_VOLUME_SLICES};
use super::consts::{tex, ODSY_VOLUME_SLICE_SHIFT_MAX};
use super::gl::{
    texture_detail_sampler, texture_lod_limit, volume_slice_stride, PyramidShape, Texture, TextureChain,
    TextureDetailSampler, TextureVolume,
};
use super::gl_context::{
    GL_ALPHA, GL_CLAMP, GL_CLAMP_TO_BORDER, GL_CLAMP_TO_EDGE, GL_INTENSITY, GL_LINEAR, GL_LINEAR_DETAIL_ALPHA_SGIS,
    GL_LINEAR_DETAIL_COLOR_SGIS, GL_LINEAR_DETAIL_SGIS, GL_LINEAR_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_NEAREST,
    GL_LUMINANCE, GL_LUMINANCE_ALPHA, GL_NEAREST, GL_NEAREST_MIPMAP_LINEAR, GL_NEAREST_MIPMAP_NEAREST, GL_REPEAT,
    GL_RGB, GL_RGBA,
};
use super::raster::{cmax, cmin, GlTexImage, GlTexLevel, GL_TEX_LEVELS_MAX};

/// Sampler word 4 bits 25:23 are the class of the texel format: 1 alpha, 2
/// luminance, 3 luminance-alpha, 4 intensity, 5 RGB, 6 RGBA. What the board
/// does with the class is UNKNOWN. MODEL CHOICE: the class is read as the
/// components the texel words hold, so class 5 samples with alpha 1 through
/// the base format GL_RGB; that the board forces alpha to 1 for class 5 is
/// INFERRED, not established. `components` is the word masked by
/// tex::COMPONENTS_MASK.
pub fn texture_class_format(components: u32) -> Option<u32> {
    match components {
        0x00800000 => Some(GL_ALPHA),
        0x01000000 => Some(GL_LUMINANCE),
        0x01800000 => Some(GL_LUMINANCE_ALPHA),
        0x02000000 => Some(GL_INTENSITY),
        0x02800000 => Some(GL_RGB),
        0x03000000 => Some(GL_RGBA),
        _ => None,
    }
}

/// One row of the board's texel format index table: what a level
/// descriptor's word 1 bits 25:20 names. The index gives the layout and
/// storage, and index 12 is the 4/4/4/4 layout in every use; the component
/// set comes from the class in sampler word 4, not from the index.
/// ESTABLISHED as the library's whole mapping: the sixteen indices it emits,
/// with the size code and the classes texture activation pairs with each.
/// No software checks the class against the index, and the pixel-to-texture
/// move sends class 6 with indices the table pairs with other classes (index
/// 10 among them), so `classes` records the library's pairing only: MODEL
/// CHOICE, the model never refuses a pair. The board's own packing of a
/// texel, and whether it checks the pair, are UNKNOWN; the model keeps whole
/// texel words, so a row decides what the model accepts, not a bit layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TexelFormat {
    /// Word 1 bits 31:30, the size code: log2 of the booked bytes a texel
    /// minus 1. Texture activation and the pixel-to-texture move both send
    /// it (ESTABLISHED for activation, INFERRED for every move source).
    pub size: u32,
    /// The classes texture activation pairs with the index, one bit at each
    /// class value.
    pub classes: u8,
}

impl TexelFormat {
    /// Whether the library pairs the index with sampler class `class`.
    pub const fn pairs_with(&self, class: u32) -> bool {
        class < 8 && (self.classes & (1 << class)) != 0
    }


    /// Bytes the library books for a texel of this format, `2 << size`.
    /// A bound on the packing, not the packing itself.
    pub const fn booked(&self) -> u32 {
        2 << self.size
    }
}

const fn texel_classes(list: &[u32]) -> u8 {
    let mut m = 0u8;
    let mut i = 0;

    while i < list.len() {
        m |= 1 << list[i];
        i += 1;
    }
    m
}

/// The format index table. None for every index the library does not emit:
/// 1, 3, 5, 6 and 7 have width entries that nothing sends, and 14, 15, 19 to
/// 31, 34 to 47 and 50 to 63 have none (the library leaves the size code
/// unwritten and returns class 0). UNKNOWN: whether the board accepts them,
/// so a level that names one is refused.
///
/// MODEL CHOICES: 32 and 33 are the same 12-bit RGBA layout and read alike
/// (the twin pair is unexplained; only the library's pixel-to-texture move
/// sends 32), and 48 and 49, the ICC pair, are read as their component sets
/// alone (their stored widths are INFERRED from the enumerant names).
pub const fn texel_format(index: u32) -> Option<TexelFormat> {
    const A: u8 = texel_classes(&[1]);
    const L: u8 = texel_classes(&[2]);
    const LA: u8 = texel_classes(&[3]);
    const I: u8 = texel_classes(&[4]);
    const RGB: u8 = texel_classes(&[5]);
    const RGBA: u8 = texel_classes(&[6]);

    Some(match index {
        0 | 8 => TexelFormat { size: 0, classes: A },
        2 | 9 => TexelFormat { size: 0, classes: I },
        4 | 10 => TexelFormat { size: 0, classes: L },
        11 => TexelFormat { size: 0, classes: LA },
        12 | 13 => TexelFormat { size: 0, classes: RGB | RGBA },
        16 => TexelFormat { size: 1, classes: LA },
        17 | 18 => TexelFormat { size: 1, classes: RGB | RGBA },
        32 | 33 => TexelFormat { size: 2, classes: RGB | RGBA },
        48 => TexelFormat { size: 0, classes: RGB },
        49 => TexelFormat { size: 1, classes: RGBA },
        _ => return None,
    })
}

/// The format a level descriptor's word 1 names, from its bits 31:30 (size
/// code) and 25:20 (format index; ESTABLISHED as one six-bit field). None
/// for an index the table does not list, or a size code that is not the
/// index's.
pub fn texture_level_texel_format(word1: u32) -> Option<TexelFormat> {
    texel_format(level_index(word1)).filter(|f| f.size == word1 >> 30)
}

/// The texel components of a detail level whose entry word 1 is `word1`,
/// under the primary's base format `primary`: the primary's, since only the
/// bound texture's mode word, with its class, reaches the board (ESTABLISHED),
/// while the detail level carries its own index. None for an index the table
/// does not list or a size code that is not the index's.
pub fn texture_level_format(word1: u32, primary: u32) -> Option<u32> {
    texture_level_texel_format(word1).map(|_| primary)
}

fn level_index(word1: u32) -> u32 {
    (word1 >> 20) & 0x3f
}

/// Level descriptor word 1 bits 31:30 and 25:20, the size code and format
/// index.
pub(super) const LEVEL_FORMAT_MASK: u32 = 0xc3f00000;
/// Level descriptor word 1 bits 11:0, the row in 16-texel units.
const LEVEL_STRIDE_MASK: u32 = 0xfff;
/// Word 1 bits 28:26 are the texture-offset code, which only the texture move
/// writes, and then 0 or 2 (ESTABLISHED). Bit 29 and bits 19:12 are never
/// written. The model has no reading for offset code 2, so all of these
/// must be clear.
const LEVEL_WORD1_UNMODELLED: u32 = 0x3c0ff000;
/// Word 0 bits 28:0 are the level's address. Bits 31:29 are kept from the
/// object by the texture move and have no known meaning.
const LEVEL_ADDRESS_MASK: u32 = 0x1fffffff;

/// A level's row, in 16-texel units rounded up to even, for a level `width`
/// texels wide (ESTABLISHED as the library's rule for a primary level).
pub(super) fn level_stride_units(width: u32) -> u32 {
    (width.div_ceil(16) + 1) & !1u32
}

/// A level stored with a texel offset sits this many texels in from its
/// allocation's corner on each axis it has, its border at one less
/// (TexelOffset).
const OFFSET_TEXELS: u32 = 16;

/// Why a draw's texture state is refused. Every refusal is logged with the
/// state's words (Board::texture_setup()).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Refusal(pub &'static str);

/// The image's dimensionality, from the path selection word's texture bits
/// (block 0x005 bits 5:3: 3D, 2D, 1D). A 3D image with a 1D or 2D bit as
/// well is not a state the library sets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Target {
    OneD,
    TwoD,
    ThreeD,
}

impl Target {
    /// The axes the image has: S, then T, then R.
    fn axes(self) -> usize {
        match self {
            Target::OneD => 1,
            Target::TwoD => 2,
            Target::ThreeD => 3,
        }
    }
}

/// Sampler word 4's wrap code for one axis: 0 repeat, 1 clamp to edge, 2
/// clamp, 3 clamp to border.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Wrap {
    Repeat,
    ClampToEdge,
    Clamp,
    ClampToBorder,
}

impl Wrap {
    fn from_code(code: u32) -> Wrap {
        match code & 3 {
            0 => Wrap::Repeat,
            1 => Wrap::ClampToEdge,
            2 => Wrap::Clamp,
            _ => Wrap::ClampToBorder,
        }
    }

    /// Words 1 to 3 bits 23:18, the value the library writes with each wrap
    /// mode: 0 repeat or clamp, 1 clamp to edge, 63 clamp to border. Its
    /// meaning on the board is UNKNOWN; the model reads only that it agrees
    /// with the wrap.
    fn size_word_mark(self) -> u32 {
        match self {
            Wrap::Repeat | Wrap::Clamp => 0,
            Wrap::ClampToEdge => 1,
            Wrap::ClampToBorder => 63,
        }
    }

    fn gl(self) -> u32 {
        match self {
            Wrap::Repeat => GL_REPEAT,
            Wrap::ClampToEdge => GL_CLAMP_TO_EDGE,
            Wrap::Clamp => GL_CLAMP,
            Wrap::ClampToBorder => GL_CLAMP_TO_BORDER,
        }
    }
}

/// Sampler word 4 bits 22:18 and bit 0, which the library takes from context
/// state rather than the texture object. What they select on the board is
/// UNKNOWN. The model reads the values it has seen and refuses any other:
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SamplerContext {
    /// Bits 22:18 0, or 20 or 24 with bit 0 clear: an ordinary draw.
    Plain,
    /// Bits 22:18 25 (bits 22, 21 and 18) with bit 0 clear: the image path's
    /// default block word, which the texture move samples under (ESTABLISHED:
    /// the move runs with that block's word, 0x03640000 with its class forced
    /// to 6) and the mip-level builder sends too. MODEL CHOICE: sampled as
    /// the plain image.
    Image,
    /// Bits 22:18 20 or 24 with bit 0 set: a colour table looked up after
    /// filtering, the 0x3e00 and 0x3e10 tables, and with 24 its alpha table
    /// too. INFERRED from the draws that send these values while those tables
    /// are loaded; what the bits do on the board is UNKNOWN.
    ColourTable { alpha: bool },
}

impl SamplerContext {
    fn decode(word: u32) -> Option<SamplerContext> {
        let enable = (word & 1) != 0;

        match ((word >> 18) & 0x1f, enable) {
            (0, false) | (20, false) | (24, false) => Some(SamplerContext::Plain),
            (25, false) => Some(SamplerContext::Image),
            (20, true) => Some(SamplerContext::ColourTable { alpha: false }),
            (24, true) => Some(SamplerContext::ColourTable { alpha: true }),
            _ => None,
        }
    }
}

/// Whether sampler word 4 is the mip-level builder's. The builder samples
/// under the image path's context (SamplerContext::Image) with bit 28 set,
/// and writes no wrap, filter or bit 17: the words seen from it differ only
/// in the class. MODEL CHOICE: any class is taken, against any index, since
/// the copy keeps whole texel words and no software checks the pair.
pub(super) fn texture_mip_build_word(word4: u32) -> bool {
    !( (word4 & MODE_WORD_UNKNOWN) != 0
        || (word4 & CORNER_FLAG) == 0
        || (word4 & 0x3fffe) != 0
        || SamplerContext::decode(word4) != Some(SamplerContext::Image)
        || texture_class_format(word4 & tex::COMPONENTS_MASK).is_none())
}

/// How a level is placed in its allocation, from word 4 bit 28 and word 6
/// bit 24, which the library sets as complements from the object's
/// texel-offset flag (ESTABLISHED).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Placement {
    /// Bit 28 set and bit 24 clear: the level starts at its allocation's
    /// corner.
    Corner,
    /// Bit 28 clear and bit 24 set: the level sits 16 texels in from its
    /// allocation's corner on each of its axes, with its one-texel border
    /// stored around it. INFERRED from the captured stored-border uploads
    /// (1D, 2D and a 2x2x2 volume) and a 512x512 ten-level 2D image whose
    /// every level sits there, border at row and column 15; MODEL CHOICE for
    /// every other size and for the volume's slices.
    TexelOffset,
    /// Both clear: the texture move's form. The move samples under the image
    /// path's default block, whose bit 28 is clear (ESTABLISHED), and writes
    /// no border colour. Read as Corner, with the move's rows (Sampler::rows).
    Move,
}

/// The texture state of one draw, decoded field by field.
#[derive(Clone, Copy)]
pub(super) struct Sampler {
    pub target: Target,
    /// Word 0 bits 14:10, 9:5 and 4:0: log2 of the base level's size on S, T
    /// and R.
    pub log2: [u32; 3],
    /// Words 1 to 3 bits 16:0: each axis's maximum clamp in whole texels.
    pub max_clamp: [u32; 3],
    pub wrap: [Wrap; 3],
    /// The class's components (texture_class_format()).
    pub base_format: u32,
    pub context: SamplerContext,
    pub placement: Placement,
    /// Word 4 bits 7:5: 0 nearest, 1 linear, 2 to 5 the four mipmap modes in
    /// GL order.
    pub min: u32,
    /// Word 4 bits 4:1: 0 nearest, 1 linear, 2 linear detail, 3 detail
    /// alpha, 4 detail colour.
    pub mag: u32,
    /// Word 4 bit 17 under a minification code other than 5: GL_PHASE_SGIX.
    pub phase: bool,
    /// The detail image word 0 names, under magnification codes 2 to 4.
    pub detail: Option<TextureDetailSampler>,
    /// Words 5 and 6 bits 23:0: the border colour, four 12-bit channels.
    pub border_colour: [u32; 2],
    /// Block 0x006 bits 5:0: the level table entry of level 0.
    pub first: u32,
    /// The minimum and maximum LOD in 1/256 (texture_lod_limit()).
    pub lod_min: i32,
    pub lod_max: i32,
    /// The environment mode, 0x141b word 0 bits 2:0: 1 replace, 2 modulate,
    /// 3 decal, 4 blend, 5 add.
    pub env: u32,
}

const MIN_FILTERS: [u32; 6] = [
    GL_NEAREST,
    GL_LINEAR,
    GL_NEAREST_MIPMAP_NEAREST,
    GL_LINEAR_MIPMAP_NEAREST,
    GL_NEAREST_MIPMAP_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
];

/// Word 4's minification code for GL_LINEAR_MIPMAP_LINEAR.
const MIN_CODE_LINEAR_MIPMAP_LINEAR: u32 = 5;
/// Word 4's minification code for GL_NEAREST_MIPMAP_NEAREST.
const MIN_CODE_NEAREST_MIPMAP_NEAREST: u32 = 2;

/// Word 0 bit 30 and word 4 bits 31:26 and 16:14 are never written.
const SIZE_WORD_UNKNOWN: u32 = 0x40000000;
const MODE_WORD_UNKNOWN: u32 = 0xec01c000;
/// Words 1 to 3: bits 31:24 and 17 are never written.
const AXIS_WORD_UNKNOWN: u32 = 0xff020000;
/// Words 5 and 6 bits 31:25, and word 5 bit 24, are never written.
const BORDER_WORD_UNKNOWN: u32 = 0xfe000000;
const OFFSET_FLAG: u32 = 1 << 24;
const CORNER_FLAG: u32 = 1 << 28;
/// The largest log2 size whose maximum clamp, 2^log2 texels, fits the 17-bit
/// field.
const LOG2_SIZE_MAX: u32 = 16;

impl Sampler {
    pub fn size(&self) -> [u32; 3] {
        self.log2.map(|l| 1u32 << l)
    }

    pub fn mipmap(&self) -> bool {
        self.min >= 2
    }

    /// The axes of the image, in S, T, R order.
    fn axis_wraps(&self) -> &[Wrap] {
        &self.wrap[..self.target.axes()]
    }

    pub fn detail_filter(&self) -> u32 {
        match self.mag {
            2 => GL_LINEAR_DETAIL_SGIS,
            3 => GL_LINEAR_DETAIL_ALPHA_SGIS,
            4 => GL_LINEAR_DETAIL_COLOR_SGIS,
            _ => 0,
        }
    }

    /// How many levels the chain has: one unless a mipmap filter and a
    /// positive maximum LOD, then down to the terminal level or the maximum
    /// LOD, whichever comes first. The library sends a maximum LOD of at
    /// most MAX_LEVEL - BASE_LEVEL (ESTABLISHED).
    fn level_count(&self) -> u32 {
        let terminal = match self.target {
            Target::ThreeD => cmax(cmax(self.log2[0], self.log2[1]), self.log2[2]),
            _ => cmax(self.log2[0], self.log2[1]),
        };

        if self.mipmap() && self.lod_max > 0 {
            cmin(terminal, ((self.lod_max + 255) / 256) as u32) + 1
        } else {
            1
        }
    }
}

impl Board {
    /// The draw's texture words decoded. A field holding a value the library
    /// never writes is unknown state, and refuses the draw.
    pub(super) fn texture_sampler(&self) -> Result<Sampler, Refusal> {
        let t = &self.tex;
        let [size, s_word, t_word, r_word, mode, border0, border1] = t.sampler;

        let target = match t.mode & (tex::MODE_1D | tex::MODE_2D | tex::MODE_3D) {
            tex::MODE_3D => Target::ThreeD,
            m if (m & tex::MODE_3D) != 0 => return Err(Refusal("a 3D target with a 1D or 2D bit")),
            m if (m & tex::MODE_2D) != 0 => Target::TwoD,
            m if (m & tex::MODE_1D) != 0 => Target::OneD,
            _ => return Err(Refusal("no texture target")),
        };
        if (size & SIZE_WORD_UNKNOWN) != 0 {
            return Err(Refusal("sampler word 0 bit 30"));
        }
        if (mode & MODE_WORD_UNKNOWN) != 0 {
            return Err(Refusal("sampler word 4 bits the library never writes"));
        }
        let log2 = [(size >> 10) & 31, (size >> 5) & 31, size & 31];
        if log2.iter().any(|&l| l > LOG2_SIZE_MAX) {
            return Err(Refusal("a size past the maximum clamp's field"));
        }
        if target != Target::ThreeD && log2[2] != 0 {
            return Err(Refusal("an R size on a 1D or 2D image"));
        }
        let wrap = [Wrap::from_code(mode >> 12), Wrap::from_code(mode >> 10), Wrap::from_code(mode >> 8)];
        // The library writes repeat for R unless the target is 3D.
        if target != Target::ThreeD && wrap[2] != Wrap::Repeat {
            return Err(Refusal("an R wrap on a 1D or 2D image"));
        }
        let mut max_clamp = [0u32; 3];
        for (axis, &word) in [s_word, t_word, r_word].iter().enumerate() {
            if (word & AXIS_WORD_UNKNOWN) != 0 {
                return Err(Refusal("sampler size word bits the library never writes"));
            }
            if (word >> 18) & 0x3f != wrap[axis].size_word_mark() {
                return Err(Refusal("a size word's wrap mark that disagrees with its wrap"));
            }
            max_clamp[axis] = word & 0x1ffff;
            // trunc(2^log2size * GL_TEXTURE_MAX_CLAMP_*_SGIX) with the clamp
            // at most 1.0 (ESTABLISHED as the library's setter).
            if max_clamp[axis] > 1 << log2[axis] {
                return Err(Refusal("a maximum clamp past the image's size"));
            }
        }
        let Some(base_format) = texture_class_format(mode & tex::COMPONENTS_MASK) else {
            return Err(Refusal("sampler word 4 class"));
        };
        let Some(context) = SamplerContext::decode(mode) else {
            return Err(Refusal("sampler word 4 bits 22:18 and 0"));
        };
        if ((border0 | border1) & BORDER_WORD_UNKNOWN) != 0 || (border0 & OFFSET_FLAG) != 0 {
            return Err(Refusal("border colour word bits the library never writes"));
        }
        let placement = match ((mode & CORNER_FLAG) != 0, (border1 & OFFSET_FLAG) != 0) {
            (true, false) => Placement::Corner,
            (false, true) => Placement::TexelOffset,
            (false, false) if context == SamplerContext::Image => Placement::Move,
            _ => return Err(Refusal("word 4 bit 28 and word 6 bit 24 are not complements")),
        };
        let min = (mode >> 5) & 7;
        let mag = (mode >> 1) & 0xf;
        if min > 5 {
            return Err(Refusal("minification code"));
        }
        if mag > 4 {
            return Err(Refusal("magnification code"));
        }
        let detail = if (2..=4).contains(&mag) {
            // With a detail magnification filter the library sends GL_NEAREST
            // as code 2 and GL_LINEAR as code 5 (ESTABLISHED).
            if min != MIN_CODE_NEAREST_MIPMAP_NEAREST && min != MIN_CODE_LINEAR_MIPMAP_LINEAR {
                return Err(Refusal("a detail filter with a minification code the library never pairs with one"));
            }
            match texture_detail_sampler(size) {
                Some(d) => Some(d),
                None => return Err(Refusal("a detail filter whose sampler word 0 names no detail level")),
            }
        } else {
            // Word 0 bits 29:25 are the detail level the setter clamps to 15;
            // the move writes 0. Bits 31 and 24:15 are stale without a detail
            // filter (ESTABLISHED), so they are not read here.
            if (size >> 25) & 0x1f > 15 {
                return Err(Refusal("a detail level past 15"));
            }
            None
        };
        if (t.level0 & !0x3f) != 0 {
            return Err(Refusal("first level index bits past 5:0"));
        }
        if t.lod.iter().any(|&w| (w & !tex::LOD_MASK) != 0) {
            return Err(Refusal("LOD word bits past 13:0"));
        }
        if !self.texture_scale_bias_known() {
            return Err(Refusal("post-filter scale or bias bits past 15:0"));
        }
        // The library always sends 0, 0, 0 (ESTABLISHED); their meaning is
        // UNKNOWN.
        if t.offset.iter().any(|&w| w != 0) {
            return Err(Refusal("texture offset words"));
        }
        let env = t.env[0];
        if !(1..=5).contains(&env) {
            return Err(Refusal("environment mode"));
        }
        // MODEL CHOICE: the environment colour's spare bits and words 3 and
        // 4 are checked only where the colour is used, under blend.
        if env == 4 && (((t.env[1] | t.env[2]) & 0xff000000) != 0 || t.env[3] != 0 || t.env[4] != 0) {
            return Err(Refusal("environment colour bits the library never writes"));
        }
        Ok(Sampler {
            target,
            log2,
            max_clamp,
            wrap,
            base_format,
            context,
            placement,
            min,
            mag,
            phase: (mode & tex::SAMPLER_BIT17) != 0 && min != MIN_CODE_LINEAR_MIPMAP_LINEAR,
            detail,
            border_colour: [border0, border1],
            first: t.level0,
            lod_min: texture_lod_limit(t.lod[2]),
            lod_max: texture_lod_limit(t.lod[3]),
            env,
        })
    }

    /// Whether the model can draw with sampler `s`. Each refusal names what
    /// is missing from the model, not a fault in the state.
    pub(super) fn texture_support(&self, s: &Sampler) -> Result<(), Refusal> {
        let axes = s.axis_wraps();
        let bf = s.base_format;

        if self.gl_index_mode() {
            return Err(Refusal("texturing in colour-index mode is not modelled"));
        }
        if s.phase {
            return Err(Refusal("GL_PHASE_SGIX is not modelled"));
        }
        if s.env == 5 {
            return Err(Refusal("the add environment is not modelled"));
        }
        // GL leaves decal undefined for these components.
        if s.env == 3 && bf != GL_RGB && bf != GL_RGBA {
            return Err(Refusal("decal of a texture without colour"));
        }
        if s.mipmap() && s.lod_min > s.lod_max {
            return Err(Refusal("a minimum LOD above the maximum"));
        }
        // MODEL CHOICE: GL_CLAMP and clamp to border are taken only on every
        // axis of the image or none. No client has been seen to mix them
        // with other wraps, and what the board does then is UNKNOWN.
        for wrap in [Wrap::Clamp, Wrap::ClampToBorder] {
            if axes.contains(&wrap) && !axes.iter().all(|&a| a == wrap) {
                return Err(Refusal("GL_CLAMP or clamp to border on some axes only"));
            }
        }
        if axes.contains(&Wrap::ClampToBorder) && s.placement == Placement::TexelOffset {
            return Err(Refusal("clamp to border on an image with a stored border"));
        }
        if let SamplerContext::ColourTable { .. } = s.context {
            if s.target == Target::ThreeD {
                return Err(Refusal("a colour table on a volume is not modelled"));
            }
            if self.tex.colour_table_valid.iter().any(|&v| v != 0xffff) {
                return Err(Refusal("a colour table lookup before both tables are loaded"));
            }
        }
        if s.detail.is_some() {
            if s.target != Target::TwoD || s.first != 0 {
                return Err(Refusal("a detail texture on other than a 2D image from entry 0"));
            }
            if s.placement != Placement::Corner || s.context != SamplerContext::Plain {
                return Err(Refusal("a detail texture with a stored border or a lookup table is not modelled"));
            }
            if axes.iter().any(|&a| a != Wrap::Repeat) {
                return Err(Refusal("a detail texture under a wrap other than repeat is not modelled"));
            }
            if self.texture_detail_function_weight().is_none() {
                return Err(Refusal("a detail texture before its whole function table is loaded"));
            }
        }
        if s.target == Target::ThreeD {
            // The volume path draws replace and modulate only.
            if s.env > 2 {
                return Err(Refusal("a volume under decal or blend is not modelled"));
            }
            if s.placement == Placement::Move {
                return Err(Refusal("the texture move on a volume"));
            }
            if axes.contains(&Wrap::ClampToBorder) {
                return Err(Refusal("clamp to border on a volume is not modelled"));
            }
            if s.size()[2] > ODSY_VOLUME_SLICES {
                return Err(Refusal("a volume deeper than the model's slice table"));
            }
            // Limitation: the R axis's maximum clamp is not applied.
            if s.max_clamp[2] != s.size()[2] {
                return Err(Refusal("a volume's R maximum clamp below its depth is not modelled"));
            }
            // The model reads a volume's levels only from entry 0; a nonzero
            // selector has no captured layout.
            if s.first != 0 {
                return Err(Refusal("a volume from a level table entry other than 0"));
            }
            // A volume without mipmaps is drawn as one image, which the shared
            // sampler filters with a single filter.
            if !s.mipmap() && s.min != s.mag {
                return Err(Refusal("a volume with different minification and magnification filters is not modelled"));
            }
        }
        Ok(())
    }

    /// Build the draw's texture from its texture words: decode them, check
    /// the model can draw them, and find each level in the object store. A
    /// refused state is logged and drops the draw.
    pub fn texture_setup(&mut self, image: &mut GlTexImage<'static>, chain: &mut TextureChain) -> bool {
        let result = self
            .texture_sampler()
            .and_then(|s| self.texture_support(&s).map(|()| s))
            .and_then(|s| self.texture_chain(&s, image, chain));

        match result {
            Ok(()) => true,
            Err(Refusal(why)) => {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: texture state refused, {}: sampler {:08x?} mode {:#x} first level {} environment {}, draw dropped",
                    why,
                    self.tex.sampler,
                    self.tex.mode,
                    self.tex.level0,
                    self.tex.env[0]
                );
                false
            }
        }
    }

    /// The image fields every target shares.
    fn texture_image(&self, s: &Sampler, image: &mut GlTexImage<'static>) {
        let [width, height, _] = s.size();

        image.base_format = s.base_format;
        if s.detail.is_none() {
            image.texture_max_clamp(s.size(), s.max_clamp, s.wrap);
        }
        image.width = width;
        image.height = height;
        image.border = (s.placement == Placement::TexelOffset) as u32;
        image.wrap_s = s.wrap[0].gl();
        // MODEL CHOICE: a 1D image has no T axis, so T never takes a border
        // tap whatever its wrap word says.
        image.wrap_t = if s.target == Target::OneD { GL_CLAMP_TO_EDGE } else { s.wrap[1].gl() };
        if s.axis_wraps().iter().any(|&w| w == Wrap::Clamp || w == Wrap::ClampToBorder) {
            let [c0, c1] = s.border_colour;
            image.border_color = [
                ((c0 >> 12) & 0xfff) as f32 / 4095.0f32,
                (c0 & 0xfff) as f32 / 4095.0f32,
                ((c1 >> 12) & 0xfff) as f32 / 4095.0f32,
                (c1 & 0xfff) as f32 / 4095.0f32,
            ];
        }
        image.mag_filter = match s.mag {
            0 => GL_NEAREST,
            1 => GL_LINEAR,
            _ => s.detail_filter(),
        };
        image.min_filter = MIN_FILTERS[s.min as usize];
        self.texture_lod_bias(image);
    }

    /// Level table entry `i`'s three words.
    fn level_entry(&self, i: u32) -> [u32; 3] {
        let e = (i * 3) as usize;
        [self.tex.levels[e], self.tex.levels[e + 1], self.tex.levels[e + 2]]
    }

    /// The format a level's word 1 names, when the model can read it: an
    /// index the table lists, with its own size code, and no texture-offset
    /// code. The size code is checked the same way for the texture move,
    /// which sends the same quantity. The class is never checked against the
    /// index (TexelFormat).
    fn level_format_check(word1: u32) -> Result<TexelFormat, Refusal> {
        if (word1 & LEVEL_WORD1_UNMODELLED) != 0 {
            return Err(Refusal("a level descriptor's word 1 bits the model has no reading for"));
        }
        texture_level_texel_format(word1)
            .ok_or(Refusal("a level's format index or size code is not one the library emits"))
    }

    fn texture_chain(&mut self, s: &Sampler, image: &mut GlTexImage<'static>, chain: &mut TextureChain) -> Result<(), Refusal> {
        self.texture_image(s, image);
        if s.target == Target::ThreeD {
            return self.texture_volume_chain(s, image, chain);
        }
        let [width, height, _] = s.size();
        let first = s.first;
        let count = s.level_count();
        let offset = s.placement == Placement::TexelOffset;
        let two_d = s.target == Target::TwoD;

        if count > GL_TEX_LEVELS_MAX || first + count > ODSY_TEX_LEVELS {
            return Err(Refusal("a chain past the level table"));
        }
        // An object holds only the rows a transfer wrote, and a client may
        // define a level with no data and then write part of it, as SDL
        // streams a frame into a larger texture. The rest of the level is
        // still its storage, with contents GL leaves undefined: grow the
        // object to the level's rows, the new ones reading 0. Before any
        // level's place is taken, since growing may move objects. Growth
        // that fits the store always succeeds (object_rows() forgets others
        // to make room), and one that cannot is not tried, so a draw never
        // waits on the raster unit for a growth that fails. MODEL POLICY:
        // only an ordinary level that holds some rows grows; an object that
        // holds none stays missing, and a level with a stored border keeps
        // its own check below. Sampling is not use, so finding a level here
        // does not touch its object (Board::object_touch()).
        if s.placement == Placement::Corner {
            for i in 0..count {
                let entry = self.level_entry(first + i);
                let object = self.object_find(entry[0] & LEVEL_ADDRESS_MASK);
                let stride_units = entry[1] & LEVEL_STRIDE_MASK;
                let (level_width, level_height) = (cmax(1, width >> i), cmax(1, height >> i));

                if let Some(o) = object {
                    if self.objects.width[o] >= level_width
                        && (stride_units == 0 || self.objects.width[o] == stride_units * 16)
                        && self.objects.rows[o] != 0
                        && self.objects.rows[o] < level_height
                        && self.objects.width[o] as u64 * level_height as u64 <= ODSY_OBJ_POOL_WORDS as u64
                    {
                        self.object_rows(o, level_height);
                    }
                }
            }
        }
        chain.clear();
        if let SamplerContext::ColourTable { alpha } = s.context {
            chain.volume.colour_table = true;
            chain.volume.colour_table_alpha = alpha;
        }
        chain.volume.scale_bias = !self.texture_scale_bias_identity();
        if chain.volume.colour_table || chain.volume.scale_bias {
            image.post_filter = true;
        }
        let format_word = self.level_entry(first)[1] & LEVEL_FORMAT_MASK;
        // A stored border's interior starts 16 texels in on each axis the
        // image has (Placement::TexelOffset).
        let interior_x = if offset { OFFSET_TEXELS } else { 0 };
        let interior_y = if offset && two_d { OFFSET_TEXELS } else { 0 };
        for i in 0..count {
            let entry = self.level_entry(first + i);
            let level_width = cmax(1, width >> i);
            let level_height = cmax(1, height >> i);
            let stride_units = entry[1] & LEVEL_STRIDE_MASK;
            // A stored border's row is its width plus 16 texels each side,
            // rounded as an ordinary row: INFERRED from the captured strides.
            let expected_units = level_stride_units(level_width + 2 * interior_x);
            let base = entry[0] & LEVEL_ADDRESS_MASK;
            let rows = level_height + interior_y + (offset && two_d) as u32;

            Self::level_format_check(entry[1])?;
            // The library writes the primary's level-0 format into every
            // primary entry (ESTABLISHED). Word 2 is not written for a 1D or
            // 2D level, so its value is not this texture's and is not read.
            if (entry[1] & LEVEL_FORMAT_MASK) != format_word {
                return Err(Refusal("levels of one chain with different formats"));
            }
            if base != 0 && stride_units != expected_units && !(count == 1 && stride_units == 0) {
                return Err(Refusal("a level row that is not the library's for its width"));
            }
            chain.touch(i);
            chain.levels[i as usize] = GlTexLevel {
                level: i,
                width: level_width,
                height: level_height,
                base_format: s.base_format,
                internal_format: s.base_format,
                ..Default::default()
            };
            if base == 0 || self.objects.count > ODSY_OBJECTS {
                continue;
            }
            let Some(o) = self.object_find(base) else {
                continue;
            };
            // MODEL CHOICE: the texture move draws only the rows it wrote,
            // from an object sized to them, so a moved level needs only the
            // rows its object holds; the rectangle's coordinates reach no
            // further.
            let rows = if s.placement == Placement::Move { rows.min(self.objects.rows[o]).max(1) } else { rows };
            let end = ((rows - 1) as u64)
                .checked_mul(self.objects.width[o] as u64)
                .and_then(|e| e.checked_add(self.objects.offset[o] as u64))
                .and_then(|e| e.checked_add(interior_x as u64))
                .and_then(|e| e.checked_add(level_width as u64 + offset as u64));
            if self.objects.width[o] < level_width
                || (stride_units != 0 && self.objects.width[o] != stride_units * 16)
                || self.objects.rows[o] < rows
                || !end.is_some_and(|e| e <= self.objects.used as u64 && e <= self.objects.mem.size / 4)
            {
                continue;
            }
            // A stored border is read, so all of it and the level must have
            // been written; MODEL POLICY: otherwise the level is missing.
            if offset
                && !self.object_region_written(
                    o,
                    interior_x - 1,
                    if two_d { interior_y - 1 } else { 0 },
                    level_width + 2,
                    if two_d { level_height + 2 } else { 1 },
                )
            {
                continue;
            }
            chain.touch(i);
            chain.storage[i as usize] = Texture {
                offset: self.objects.offset[o].wrapping_add(interior_y.wrapping_mul(self.objects.width[o])).wrapping_add(interior_x),
                stride: self.objects.width[o],
                width: level_width,
                height: level_height,
                base_format: s.base_format,
                words: self.objects.used,
                border_rows: offset && two_d,
            };
            chain.levels[i as usize].read_texel = true;
            if offset {
                chain.levels[i as usize].border = 1;
                chain.levels[i as usize].read_border_texel = true;
            }
        }
        chain.pyramid = Some(PyramidShape {
            count,
            base_level: 0,
            max_level: count - 1,
            min_lod: s.lod_min as f64 / 256.0,
            max_lod: if s.mipmap() { s.lod_max as f64 / 256.0 } else { 15.0 },
        });
        if let Some(detail) = s.detail {
            // The detail image's entry lies past the base chain's.
            if first + count > detail.descriptor {
                return Err(Refusal("a detail image whose entry overlaps the base chain"));
            }
            self.texture_detail_backing_setup(chain, s.base_format, detail.descriptor, detail.width, detail.height)?;
            chain.detail_weight = self.texture_detail_function_weight().unwrap_or_default();
            image.detail_image = Some(chain.detail_image);
            image.detail_level = detail.level;
            image.detail_mode = detail.mode;
            image.read_filter_weight = true;
        }
        Ok(())
    }

    /// A volume's levels. Level 0 is entry 0; each slice is its own object,
    /// the slices 64 << word 2 address units apart (volume_slice_stride()).
    ///
    /// MODEL POLICY for mipmaps: the library fills entry 0 alone, and the
    /// model derives each further level as following the one before, one
    /// slice stride per slice. That rule is taken only for what it was seen
    /// on: repeat-wrapped volumes of at most 16 on each axis, with 128-byte
    /// slices (word 2 1) and rows of 32 texels (stride 2), and for any
    /// minification filter the shared sampler draws. Whether the board
    /// mipmaps volumes at all is UNKNOWN (published accounts say it does not).
    fn texture_volume_chain(&mut self, s: &Sampler, image: &mut GlTexImage<'static>, chain: &mut TextureChain) -> Result<(), Refusal> {
        let [width, height, depth] = s.size();
        let entry = self.level_entry(0);
        let stride_units = entry[1] & LEVEL_STRIDE_MASK;
        let offset = s.placement == Placement::TexelOffset;
        let mipmap = s.mipmap();
        // A stored border's slices hold the level 16 texels in, border at 15.
        let interior = if offset { OFFSET_TEXELS } else { 0 };
        let border_texels = if offset { 2 } else { 0 };

        let format = Self::level_format_check(entry[1])?;
        if stride_units != level_stride_units(width + 2 * interior) {
            return Err(Refusal("a volume row that is not the library's for its width"));
        }
        if (entry[0] & LEVEL_ADDRESS_MASK) == 0 || entry[2] == 0 || entry[2] > ODSY_VOLUME_SLICE_SHIFT_MAX {
            return Err(Refusal("a volume level with no address or slice spacing"));
        }
        if mipmap
            && (s.wrap.iter().any(|&w| w != Wrap::Repeat)
                || width > 16
                || height > 16
                || depth > 16
                || stride_units != 2
                || entry[2] != 1)
        {
            return Err(Refusal("a mipmapped volume outside the derived-level policy"));
        }
        if self.objects.count > ODSY_OBJECTS {
            return Err(Refusal("the object table is past its size"));
        }
        let slice_stride = volume_slice_stride(entry[2], format);
        // The border slice sits at 15 slice strides from the level's base
        // and the level's own slices from 16: INFERRED from the captured
        // stored 2x2x2 volume, whose first upload is 15 strides above the
        // level base at two bases; MODEL CHOICE for other sizes, as the
        // same 16-texel rule as rows and columns.
        let start = slice_stride
            .checked_mul((interior - offset as u32) as u64)
            .and_then(|s| s.checked_add((entry[0] & LEVEL_ADDRESS_MASK) as u64))
            .and_then(|s| u32::try_from(s).ok());
        let Some(mut base) = start else {
            return Err(Refusal("a volume past the address space"));
        };
        let count = s.level_count();
        if count > GL_TEX_LEVELS_MAX {
            return Err(Refusal("a chain past the level table"));
        }
        chain.clear();
        let mut slice_next: u32 = 0;
        for i in 0..count {
            let level_width = cmax(1, width >> i);
            let level_height = cmax(1, height >> i);
            let level_depth = cmax(1, depth >> i);
            if i != 0 && self.level_entry(i) != [0; 3] {
                return Err(Refusal("a mipmapped volume with level table entries past 0"));
            }
            let level_slice = slice_next;
            for z in 0..level_depth + border_texels {
                let slice_base = (z as u64)
                    .checked_mul(slice_stride)
                    .and_then(|o| o.checked_add(base as u64))
                    .and_then(|s| u32::try_from(s).ok());
                let Some(slice_base) = slice_base else {
                    return Err(Refusal("a volume past the address space"));
                };
                let Some(o) = self.object_find(slice_base) else {
                    return Err(Refusal("a volume slice with no object"));
                };
                let first_row = interior.saturating_sub(offset as u32);
                let storage_end = (first_row as u64 + level_height as u64 - 1 + border_texels as u64)
                    .checked_mul(self.objects.width[o] as u64)
                    .and_then(|e| e.checked_add(self.objects.offset[o] as u64))
                    .and_then(|e| e.checked_add(level_width as u64 + if offset { interior as u64 + 2 } else { 0 }));
                if self.objects.width[o] != stride_units * 16
                    || self.objects.rows[o] < first_row + level_height + border_texels
                    || !storage_end.is_some_and(|e| e <= self.objects.used as u64 && e <= self.objects.mem.size / 4)
                {
                    return Err(Refusal("a volume slice whose object does not hold it"));
                }
                if offset && !self.object_region_written(o, first_row, first_row, level_width + 2, level_height + 2) {
                    return Err(Refusal("a stored volume slice not wholly written"));
                }
                if slice_next as usize >= self.tex.slice_offset.len() {
                    return Err(Refusal("a volume deeper than the model's slice table"));
                }
                self.tex.slice_offset[slice_next as usize] = self.objects.offset[o];
                self.tex.slice_width[slice_next as usize] = self.objects.width[o];
                slice_next += 1;
            }
            chain.touch(i);
            chain.volumes[i as usize] = TextureVolume {
                base,
                stride_units,
                level_word2: entry[2],
                width: level_width,
                height: level_height,
                depth: level_depth,
                base_format: s.base_format,
                words: self.objects.used,
                border: offset,
                first_slice: level_slice,
                slices: slice_next - level_slice,
                ..Default::default()
            };
            chain.levels[i as usize] = GlTexLevel {
                level: i,
                width: level_width,
                height: level_height,
                depth: level_depth,
                base_format: s.base_format,
                internal_format: s.base_format,
                read_texel_3d: true,
                ..Default::default()
            };
            if i + 1 < count {
                let next_base = (level_depth as u64)
                    .checked_mul(slice_stride)
                    .and_then(|n| n.checked_add(base as u64))
                    .and_then(|n| u32::try_from(n).ok());
                let Some(next_base) = next_base else {
                    return Err(Refusal("a volume past the address space"));
                };
                base = next_base;
            }
        }
        chain.volume = chain.volumes[0];
        image.depth = depth;
        image.volume = true;
        image.wrap_r = s.wrap[2].gl();
        image.read_texel_3d = true;
        image.read_border_texel_3d = offset;
        if !self.texture_scale_bias_identity() {
            chain.volume.scale_bias = true;
            image.post_filter = true;
        }
        if mipmap {
            chain.pyramid = Some(PyramidShape {
                count,
                base_level: 0,
                max_level: count - 1,
                min_lod: s.lod_min as f64 / 256.0,
                max_lod: s.lod_max as f64 / 256.0,
            });
        }
        Ok(())
    }
}
