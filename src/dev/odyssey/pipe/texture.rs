//! The texture unit: the board's texture registers, what a draw latches
//! from them, and the sampling of a quad (`clanker/plans/odyssey-quad-pipeline.md`).
//!
//! The registers are the sampler block (0x1c16), the environment block
//! (0x141b), the LOD block (0x1013), the first level's entry (block 0x006),
//! the level table (0x0e80 + i, three words an entry), the post-filter
//! scale and bias (0x203b), the colour tables (0x3e00, 0x3e10) and the
//! detail function (0x3e40), and the path selection word (block 0x005)
//! whose bits 5:3 name the target. The field layouts are what the GL
//! library writes (ESTABLISHED as the library's setters); what the board
//! does with them is mostly UNKNOWN, and the unit samples by OpenGL's rules.
//!
//! The unit reads board memory where the level table points, masked to the
//! memory's size: a level that points at garbage samples garbage.
//!
//! Approximated: no source gives the board's sub-texel, filter-weight or
//! level-of-detail widths. The unit takes positions in 1/256 of a texel,
//! filter weights of 8 bits, lambda in 1/256 from a quad's diagonals, a
//! blend between levels by an 8-bit fraction, and texels as 12-bit
//! components, filtered with 16 fraction bits and handed to the shader in
//! 12.12 (`fixed`).

use super::fixed;
use super::super::bits::sextract32;
use super::super::consts::tex;
use super::super::field::{texel_storage, texel_unpack, value_len, FIELD_FOUR, FIELD_ONE, FIELD_PACKED, FIELD_TWO};
use super::super::vram::VramPtr;

/// Level table entries (0x0e80 .. 0x0ebf).
pub(crate) const LEVEL_ENTRIES: usize = 64;
/// Colour table and detail function entries.
pub(crate) const TABLE_ENTRIES: usize = 256;
/// The most levels a chain has: one per bit of a 5-bit log2 size.
pub(crate) const LEVELS: usize = 32;
/// Lambda's least value, below every level clamp, in 1/256.
pub(crate) const LAMBDA_MIN: i32 = -(1 << 24);

/// The texture registers as the command FIFO wrote them.
#[derive(Clone)]
#[repr(C)]
pub(crate) struct TexRegs {
    /// Block 0x005, the transform and texture path selection: bits 5:3
    /// the target, bits 2:0 the client clip planes.
    pub path: u32,
    /// 0x1c16.
    pub sampler: [u32; 7],
    /// 0x141b.
    pub env: [u32; 5],
    /// 0x1013: the S and T LOD bias, the minimum and the maximum LOD.
    pub lod: [u32; 4],
    /// Block 0x006: the level table entry of level 0.
    pub level0: u32,
    /// 0x0e80 + i, three words an entry.
    pub levels: [u32; LEVEL_ENTRIES * 3],
    /// 0x203b: scale then bias for R, G, B and A.
    pub scale_bias: [u32; 8],
    /// 0x0c3e.
    pub offset: [u32; 3],
    /// 0x3e00 and 0x3e10.
    pub colour_table: [[u32; TABLE_ENTRIES]; 2],
    /// 0x3e40.
    pub detail: [u32; TABLE_ENTRIES],
}

impl Default for TexRegs {
    /// The registers out of reset: the library's defaults for the LOD
    /// limits (-15 and 15) and the scale (1.0), a model choice.
    fn default() -> Self {
        let mut scale_bias = [0; 8];
        for c in 0..4 {
            scale_bias[c * 2] = tex::SCALE_BIAS_ONE;
        }
        TexRegs {
            path: 0,
            sampler: [0; 7],
            env: [0; 5],
            lod: [0, 0, 0x3100, 0x0f00],
            level0: 0,
            levels: [0; LEVEL_ENTRIES * 3],
            scale_bias,
            offset: [0; 3],
            colour_table: [[0; TABLE_ENTRIES]; 2],
            detail: [0; TABLE_ENTRIES],
        }
    }
}

/// The path word's target, bits 5:3 (ESTABLISHED; 3D wins over 2D over 1D).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub(crate) enum Target {
    None,
    OneD,
    TwoD,
    ThreeD,
}

impl Target {
    pub fn axes(self) -> usize {
        match self {
            Target::None => 0,
            Target::OneD => 1,
            Target::TwoD => 2,
            Target::ThreeD => 3,
        }
    }
}

/// Sampler word 4's wrap code for one axis: bits 13:12 S, 11:10 T, 9:8 R.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub(crate) enum Wrap {
    Repeat,
    ClampToEdge,
    Clamp,
    ClampToBorder,
}

/// Sampler word 4 bits 7:5, in GL's order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub(crate) enum Min {
    Nearest,
    Linear,
    NearestMipNearest,
    LinearMipNearest,
    NearestMipLinear,
    LinearMipLinear,
}

impl Min {
    pub fn mipmap(self) -> bool {
        !matches!(self, Min::Nearest | Min::Linear)
    }

    /// The filter within a level.
    fn linear(self) -> bool {
        matches!(self, Min::Linear | Min::LinearMipNearest | Min::LinearMipLinear)
    }

    /// The nearest level rather than a blend of two.
    fn nearest_level(self) -> bool {
        matches!(self, Min::NearestMipNearest | Min::LinearMipNearest)
    }
}

/// Sampler word 4 bits 4:1: nearest, linear, and the three detail filters
/// (SGIS_detail_texture: all components, alpha only, colour only).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub(crate) enum Mag {
    Nearest,
    Linear,
    Detail,
    DetailAlpha,
    DetailColour,
}

impl Mag {
    fn detail(self) -> bool {
        matches!(self, Mag::Detail | Mag::DetailAlpha | Mag::DetailColour)
    }
}

/// The texel class, sampler word 4 bits 25:23: the components the texels
/// hold. MODEL CHOICE: class 5 samples with alpha 1.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub(crate) enum Class {
    Alpha,
    Luminance,
    LuminanceAlpha,
    Intensity,
    Rgb,
    Rgba,
}

/// The environment block's word 0 bits 2:0: 1 replace, 2 modulate, 3
/// decal, 4 blend, 5 add.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub(crate) enum Env {
    Replace,
    Modulate,
    Decal,
    Blend,
    Add,
}

/// What the latch found it does not decode or the model does not draw:
/// one bit per reason, logged once each (Board::latch_state()).
pub(crate) mod unimplemented {
    pub const MIN_CODE: u32 = 1 << 0;
    pub const MAG_CODE: u32 = 1 << 1;
    pub const CLASS: u32 = 1 << 2;
    pub const FORMAT: u32 = 1 << 3;
    pub const PHASE: u32 = 1 << 4;
    pub const OFFSET: u32 = 1 << 5;
    pub const CONTEXT: u32 = 1 << 6;
    pub const NAMES: [&str; 7] = [
        "a minification code past 5",
        "a magnification code past 4",
        "a texel class past 6",
        "a level format index with no storage",
        "GL_PHASE_SGIX (word 4 bit 17)",
        "the texture offset words",
        "sampler word 4 bits 22:18",
    ];
}

impl TexRegs {
    pub fn target(&self) -> Target {
        if self.path & tex::MODE_3D != 0 {
            Target::ThreeD
        } else if self.path & tex::MODE_2D != 0 {
            Target::TwoD
        } else if self.path & tex::MODE_1D != 0 {
            Target::OneD
        } else {
            Target::None
        }
    }

    /// Bits 2:0 of the path word, a thermometer code: how many client clip
    /// planes the hardware tests (ESTABLISHED).
    pub fn clip_planes(&self) -> u32 {
        (self.path & 7).count_ones()
    }

    /// Word 0 bits 14:10, 9:5 and 4:0: log2 of the base level's size on S,
    /// T and R.
    pub fn log2(&self) -> [u32; 3] {
        let w = self.sampler[0];
        [(w >> 10) & 31, (w >> 5) & 31, w & 31]
    }

    pub fn wrap(&self, axis: usize) -> Wrap {
        match (self.sampler[4] >> (12 - 2 * axis)) & 3 {
            0 => Wrap::Repeat,
            1 => Wrap::ClampToEdge,
            2 => Wrap::Clamp,
            _ => Wrap::ClampToBorder,
        }
    }

    pub fn min(&self) -> Min {
        match (self.sampler[4] >> 5) & 7 {
            0 => Min::Nearest,
            1 => Min::Linear,
            2 => Min::NearestMipNearest,
            3 => Min::LinearMipNearest,
            4 => Min::NearestMipLinear,
            _ => Min::LinearMipLinear,
        }
    }

    pub fn mag(&self) -> Mag {
        match (self.sampler[4] >> 1) & 0xf {
            0 => Mag::Nearest,
            1 => Mag::Linear,
            2 => Mag::Detail,
            3 => Mag::DetailAlpha,
            4 => Mag::DetailColour,
            _ => Mag::Linear,
        }
    }

    pub fn class(&self) -> Class {
        match (self.sampler[4] >> 23) & 7 {
            1 => Class::Alpha,
            2 => Class::Luminance,
            3 => Class::LuminanceAlpha,
            4 => Class::Intensity,
            5 => Class::Rgb,
            _ => Class::Rgba,
        }
    }

    /// Word 6 bit 24: the level sits 16 texels in from its allocation's
    /// corner on each axis it has, its one-texel border stored around it
    /// (INFERRED from captured stored-border uploads).
    pub fn stored_border(&self) -> bool {
        self.sampler[6] & (1 << 24) != 0
    }

    /// Word 4 bit 0 with bits 22:18 20 or 24: a colour table looked up after
    /// filtering, and with 24 its alpha table too (INFERRED from the draws
    /// that send these while the tables are loaded).
    pub fn colour_table(&self) -> (bool, bool) {
        let on = self.sampler[4] & 1 != 0;
        (on, on && (self.sampler[4] >> 18) & 0x1f == 24)
    }

    /// Words 5 and 6 bits 23:0: the border colour, four 12-bit channels R,
    /// G, B, A.
    pub fn border(&self) -> [u32; 4] {
        let [c0, c1] = [self.sampler[5], self.sampler[6]];
        [(c0 >> 12) & 0xfff, c0 & 0xfff, (c1 >> 12) & 0xfff, c1 & 0xfff]
    }

    pub fn env(&self) -> Env {
        match self.env[0] & 7 {
            2 => Env::Modulate,
            3 => Env::Decal,
            4 => Env::Blend,
            5 => Env::Add,
            _ => Env::Replace,
        }
    }

    /// The environment colour, two 12-bit components a word, R G then B A
    /// (ESTABLISHED as the packing).
    pub fn env_colour(&self) -> [u32; 4] {
        [(self.env[1] >> 12) & 0xfff, self.env[1] & 0xfff, (self.env[2] >> 12) & 0xfff, self.env[2] & 0xfff]
    }

    /// LOD word i: 14-bit two's complement with 8 fraction bits.
    pub fn lod(&self, i: usize) -> i32 {
        sextract32(self.lod[i], 0, tex::LOD_BITS)
    }

    /// Level table entry i (wrapping the table).
    pub fn entry(&self, i: u32) -> [u32; 3] {
        let e = (i as usize % LEVEL_ENTRIES) * 3;
        [self.levels[e], self.levels[e + 1], self.levels[e + 2]]
    }

    /// The detail function at lambda in 1/256, in 12.12: entry i at LOD
    /// -i/32, bits 23:12 an unsigned 12-bit fraction (ESTABLISHED as the
    /// library's table builder), held past the ends and linear between
    /// entries (APPROXIMATION: the board's indexing is UNKNOWN).
    pub fn detail_weight(&self, lambda: i32) -> i32 {
        let weight = |i: usize| ((self.detail[i] >> 12) & 0xfff) as i32;

        if lambda >= 0 {
            return weight(0) << 12;
        }
        // -lambda / 8: entries in 1/32 of a LOD over 1/256.
        let x = lambda.unsigned_abs() as usize;
        let i = x >> 3;
        if i >= TABLE_ENTRIES - 1 {
            return weight(TABLE_ENTRIES - 1) << 12;
        }
        let f = (x & 7) as i32;
        (weight(i) << 12) + (((weight(i + 1) - weight(i)) << 12) * f >> 3)
    }

    /// What the latch does not decode in these registers.
    fn unimplemented(&self) -> u32 {
        use unimplemented::*;
        let w4 = self.sampler[4];
        let mut u = 0;

        if (w4 >> 5) & 7 > 5 {
            u |= MIN_CODE;
        }
        if (w4 >> 1) & 0xf > 4 {
            u |= MAG_CODE;
        }
        if !(1..=6).contains(&((w4 >> 23) & 7)) {
            u |= CLASS;
        }
        if w4 & tex::SAMPLER_BIT17 != 0 && self.min() != Min::LinearMipLinear {
            u |= PHASE;
        }
        if self.offset != [0; 3] {
            u |= OFFSET;
        }
        if !matches!((w4 >> 18) & 0x1f, 0 | 20 | 24 | 25) {
            u |= CONTEXT;
        }
        u
    }
}

/// One level as the unit reads it: where its texel (0, 0) is, its row of
/// 16-texel tiles, its texel storage and its log2 size; a volume's slices
/// are `slice` bytes apart.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct Level {
    /// Decodes the level's storage and the texture's class.
    pub gather: GatherFn,
    pub addr: u64,
    pub pitch: u32,
    pub bpp: u8,
    pub field: u8,
    /// The interior's offset in the allocation, a stored border's 16.
    pub ox: u32,
    pub oy: u32,
    pub wl: u32,
    pub hl: u32,
    pub dl: u32,
    pub slice: u64,
}

impl Default for Level {
    fn default() -> Self {
        Level { gather: gather_for(4, FIELD_FOUR, Class::Rgba), addr: 0, pitch: 0, bpp: 0, field: 0, ox: 0, oy: 0, wl: 0, hl: 0, dl: 0, slice: 0 }
    }
}

/// A level's row in 16-texel units rounded up to even, for a level `width`
/// texels wide (ESTABLISHED as the library's rule), where its entry gives
/// none.
fn row_units(width: u64) -> u32 {
    ((width.div_ceil(16) + 1) & !1) as u32
}

/// The texture state a draw takes from the registers once they change: the
/// levels the table names, the LOD clamps and scales, and the border
/// colour, each worked out once rather than per quad.
#[derive(Clone)]
#[repr(C)]
pub(crate) struct Latch {
    pub count: u32,
    pub level: [Level; LEVELS],
    /// The address function of each axis, for its wrap.
    pub addr: [AddrFn; 3],
    /// The detail image, under a detail magnification filter.
    pub detail: Level,
    /// Whether lambda decides anything, its clamps, and where magnification
    /// starts, in 1/256.
    pub lod: bool,
    pub min_lod: i32,
    pub max_lod: i32,
    pub mag_below: i32,
    /// The LOD bias per axis, as a scale of the footprint.
    pub lod_scale: [f32; 3],
    /// The base level's size per axis in 1/256 of a texel, and the detail
    /// image's per unit s and t.
    pub scale: [f32; 3],
    pub detail_scale: [f32; 2],
    /// CLAMP's upper bound per axis in 1/256 of a base texel.
    pub clamp: [i64; 3],
    /// The border colour, 12-bit components with the class's filled in.
    pub border: [u32; 4],
    /// unimplemented::* bits.
    pub unimplemented: u32,
}

impl Default for Latch {
    fn default() -> Self {
        Latch {
            count: 1,
            level: [Level::default(); LEVELS],
            addr: [addr_for(Wrap::Repeat, 0), addr_for(Wrap::Repeat, 1), addr_for(Wrap::Repeat, 2)],
            detail: Level::default(),
            lod: false,
            min_lod: 0,
            max_lod: 0,
            mag_below: 0,
            lod_scale: [1.0; 3],
            scale: [0.0; 3],
            detail_scale: [0.0; 2],
            clamp: [i64::MAX; 3],
            border: [0; 4],
            unimplemented: 0,
        }
    }
}

impl Latch {
    /// The state the registers r give.
    pub fn of(r: &TexRegs) -> Latch {
        let mut l = Latch { unimplemented: r.unimplemented(), ..Latch::default() };
        let target = r.target();
        let log2 = r.log2();
        let min = r.min();
        let mag = r.mag();
        let three = target == Target::ThreeD;
        let border = r.stored_border();
        let class = r.class();
        // A level's interior sits 16 texels in on each of its axes
        // (TexRegs::stored_border()).
        let inset = if border { 16 } else { 0 };

        // The chain: one level unless a mipmap filter and a positive
        // maximum LOD, then down to the terminal level or the maximum LOD.
        let terminal = if three { log2[0].max(log2[1]).max(log2[2]) } else { log2[0].max(log2[1]) };
        l.min_lod = r.lod(2);
        l.max_lod = if min.mipmap() { r.lod(3) } else { 15 << 8 };
        l.count = if min.mipmap() && l.max_lod > 0 { terminal.min(((l.max_lod + 255) >> 8) as u32) + 1 } else { 1 };

        let storage = |w1: u32, u: &mut u32| {
            texel_storage((w1 >> 20) & 0x3f).unwrap_or_else(|| {
                *u |= unimplemented::FORMAT;
                (4, FIELD_FOUR)
            })
        };
        if three {
            // A volume's levels follow entry 0's, each a slice stride per
            // slice past the one before (MODEL POLICY: the library fills
            // entry 0 alone). Its slices are 2^(word 2) tiles of 256 texels
            // apart, each texel the bytes the size code books (INFERRED);
            // a stored border's slice sits one below the interior's first,
            // which is 16 strides up.
            let e = r.entry(0);
            let (bytes, field) = storage(e[1], &mut l.unimplemented);
            let booked = 2u64 << (e[1] >> 30);
            let slice = (256 * booked).wrapping_shl(e[2]);
            let mut addr = ((e[0] & 0x1fff_ffff) as u64 * 16).wrapping_add(slice.wrapping_mul(inset as u64));
            let pitch = match e[1] & 0xfff {
                0 => row_units((1u64 << log2[0]) + 2 * inset as u64),
                p => p,
            };
            for i in 0..l.count as usize {
                let lv = Level {
                    gather: gather_for(bytes, field, class),
                    addr,
                    pitch,
                    bpp: bytes.trailing_zeros() as u8,
                    field,
                    ox: inset,
                    oy: inset,
                    wl: log2[0].saturating_sub(i as u32),
                    hl: log2[1].saturating_sub(i as u32),
                    dl: log2[2].saturating_sub(i as u32),
                    slice,
                };
                addr = addr.wrapping_add(slice.wrapping_mul(1 << lv.dl));
                l.level[i] = lv;
            }
        } else {
            let oy = if target == Target::TwoD { inset } else { 0 };
            for i in 0..l.count {
                let e = r.entry((r.level0 & 0x3f) + i);
                let (bytes, field) = storage(e[1], &mut l.unimplemented);
                let wl = log2[0].saturating_sub(i);

                l.level[i as usize] = Level {
                    gather: gather_for(bytes, field, class),
                    addr: (e[0] & 0x1fff_ffff) as u64 * 16,
                    pitch: match e[1] & 0xfff {
                        0 => row_units((1u64 << wl) + 2 * inset as u64),
                        p => p,
                    },
                    bpp: bytes.trailing_zeros() as u8,
                    field,
                    ox: inset,
                    oy,
                    wl,
                    hl: log2[1].saturating_sub(i),
                    dl: 0,
                    slice: 0,
                };
            }
        }
        if mag.detail() {
            // Word 0 bits 29:25 are the detail level as a positive count,
            // the image entry 64 minus it, 24:20 and 19:15 its log2 width
            // and height (ESTABLISHED from the library's detail branch).
            let w0 = r.sampler[0];
            let k = (w0 >> 25) & 0x1f;
            let e = r.entry(LEVEL_ENTRIES as u32 - k);
            let (bytes, field) = storage(e[1], &mut l.unimplemented);
            let wl = (w0 >> 20) & 0x1f;

            l.detail = Level {
                    gather: gather_for(bytes, field, class),
                addr: (e[0] & 0x1fff_ffff) as u64 * 16,
                pitch: match e[1] & 0xfff {
                    0 => row_units(1u64 << wl),
                    p => p,
                },
                bpp: bytes.trailing_zeros() as u8,
                field,
                wl,
                hl: (w0 >> 15) & 0x1f,
                ..Level::default()
            };
            // The detail image's texels per unit s and t, 2^k as fine as
            // the base level's.
            l.detail_scale = [(log2[0] + k) as f32, (log2[1] + k) as f32].map(|e| e.exp2() * 256.0);
        }
        // Lambda decides the filter or the level where there is a choice.
        l.lod = min.mipmap() || min.linear() != (mag != Mag::Nearest) || mag.detail();
        l.mag_below = if mag.detail() {
            if min.nearest_level() { 128 } else { 0 }
        } else if mag != Mag::Nearest && matches!(min, Min::NearestMipNearest | Min::NearestMipLinear) {
            128
        } else {
            0
        };
        let bias = |b: i32| (b as f32 / tex::LOD_ONE as f32).exp2();
        l.lod_scale = [bias(r.lod(0)), if target == Target::OneD { 0.0 } else { bias(r.lod(1)) }, if three { 1.0 } else { 0.0 }];
        l.scale = log2.map(|e| (e as f32).exp2() * 256.0);
        // Words 1 to 3 bits 16:0: each axis's maximum clamp in whole
        // texels, taken under CLAMP alone.
        for a in 0..3 {
            let max = (r.sampler[1 + a] & 0x1ffff) as i64;
            if r.wrap(a) == Wrap::Clamp && max < 1 << log2[a] {
                l.clamp[a] = max << 8;
            }
        }
        l.border = expand(class, r.border());
        // A 1D image's one row is every row.
        let wrap_t = if target == Target::OneD { Wrap::ClampToEdge } else { r.wrap(1) };
        l.addr = [addr_for(r.wrap(0), 0), addr_for(wrap_t, 1), addr_for(r.wrap(2), 2)];
        l
    }
}

/// A texel's components as the class fills them: luminance over red, green
/// and blue, alpha 1 (4095) where there is none, colour 1 for an alpha
/// texture.
#[inline(always)]
fn expand(class: Class, c: [u32; 4]) -> [u32; 4] {
    const ONE: u32 = 0xfff;
    match class {
        Class::Alpha => [ONE, ONE, ONE, c[3]],
        Class::Luminance => [c[0], c[0], c[0], ONE],
        Class::LuminanceAlpha => [c[0], c[0], c[0], c[3]],
        Class::Intensity => [c[0]; 4],
        Class::Rgb => [c[0], c[1], c[2], ONE],
        Class::Rgba => c,
    }
}

/// Texels a window holds on each of x and y, and slices on z.
pub(crate) const WIN: usize = 8;
pub(crate) const WIN_SLICES: usize = 2;

/// A gathered window: texel (x, y, z) at (z * WIN + y) * WIN + x, 12-bit
/// components with the class's filled in.
pub(crate) type Win = [[u32; 4]; WIN * WIN * WIN_SLICES];

/// One axis of a window: each index's part of the texel's byte offset, and
/// which indices lie outside the level and take the border colour.
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub(crate) struct Axis {
    pub off: [u64; WIN],
    pub outside: u32,
    pub n: u32,
}

/// Fill an axis for indices lo .. lo + n of a level, under its wrap.
pub(crate) type AddrFn = fn(lv: &Level, lo: i64, n: u32, border: bool, a: &mut Axis);

/// Decode a window's texels from board memory.
///
/// # Safety
/// `out` holds a window (`Win`).
pub(crate) type GatherFn = unsafe fn(mem: VramPtr, base: u64, xs: &Axis, ys: &Axis, zs: &Axis, border: &[u32; 4], out: *mut [u32; 4]);

/// The address part of indices along one axis, wrapped: a tiled offset is
/// a column part plus a row part (16-texel tiles of 256 texels, `pitch`
/// tiles a row), and a slice part for a volume. Repeat wraps the index,
/// clamp to edge clamps it, and the two clamps leave an index outside the
/// level for the border colour, unless the border is stored and sits at -1
/// and the size.
fn addr<const WRAP: u32, const AXIS: u32>(lv: &Level, lo: i64, n: u32, border: bool, a: &mut Axis) {
    let (l, inset) = match AXIS {
        0 => (lv.wl, lv.ox),
        1 => (lv.hl, lv.oy),
        // A volume's slice inset is in its address already (Latch::of()).
        _ => (lv.dl, 0),
    };
    let size = 1i64 << l.min(62);
    let mask = size - 1;

    a.n = n;
    a.outside = 0;
    for i in 0..n as usize {
        let mut x = lo + i as i64;
        if WRAP == Wrap::Repeat as u32 {
            x &= mask;
        } else if WRAP == Wrap::ClampToEdge as u32 {
            x = x.clamp(0, mask);
        } else if !border && (x < 0 || x > mask) {
            a.outside |= 1 << i;
        }
        let x = x.wrapping_add(inset as i64) as u64;
        a.off[i] = match AXIS {
            0 => ((x >> 4) << 8 | (x & 15)) << lv.bpp,
            1 => ((x >> 4).wrapping_mul(lv.pitch as u64) << 8 | (x & 15) << 4) << lv.bpp,
            _ => x.wrapping_mul(lv.slice),
        };
    }
}

/// The address function for a wrap on an axis.
pub(crate) fn addr_for(wrap: Wrap, axis: u32) -> AddrFn {
    macro_rules! on_axis {
        ($w:expr) => {
            match axis {
                0 => addr::<{ $w as u32 }, 0>,
                1 => addr::<{ $w as u32 }, 1>,
                _ => addr::<{ $w as u32 }, 2>,
            }
        };
    }
    match wrap {
        Wrap::Repeat => on_axis!(Wrap::Repeat),
        Wrap::ClampToEdge => on_axis!(Wrap::ClampToEdge),
        Wrap::Clamp => on_axis!(Wrap::Clamp),
        Wrap::ClampToBorder => on_axis!(Wrap::ClampToBorder),
    }
}

/// The texel at byte address a as 12-bit components of the class: the
/// storage's value unpacked, each 8-bit value widened by repeating its top
/// bits. An alpha texture's one component is its alpha.
#[inline(always)]
fn texel<const BYTES: u8, const FIELD: u8, const CLASS: u32>(mem: VramPtr, a: u64) -> [u32; 4] {
    let w = texel_unpack(mem.load(a, value_len(BYTES)), BYTES, FIELD);
    let c = |s: u32| {
        let v = (w >> s) & 0xff;
        (v << 4) | (v >> 4)
    };
    let class = CLASSES[CLASS as usize];
    let rgba = if class == Class::Alpha { [0, 0, 0, c(24)] } else { [c(24), c(16), c(8), c(0)] };

    expand(class, rgba)
}

const CLASSES: [Class; 6] = [Class::Alpha, Class::Luminance, Class::LuminanceAlpha, Class::Intensity, Class::Rgb, Class::Rgba];

/// A window's texels for one storage and class.
unsafe fn gather<const BYTES: u8, const FIELD: u8, const CLASS: u32>(mem: VramPtr, base: u64, xs: &Axis, ys: &Axis, zs: &Axis, border: &[u32; 4], out: *mut [u32; 4]) {
    for z in 0..zs.n as usize {
        for y in 0..ys.n as usize {
            let row = base.wrapping_add(zs.off[z]).wrapping_add(ys.off[y]);
            let edge = (zs.outside >> z | ys.outside >> y) & 1 != 0;
            for x in 0..xs.n as usize {
                let t = if edge || (xs.outside >> x) & 1 != 0 {
                    *border
                } else {
                    texel::<BYTES, FIELD, CLASS>(mem, row.wrapping_add(xs.off[x]))
                };
                // SAFETY: the window holds WIN x WIN x WIN_SLICES texels.
                unsafe { out.add((z * WIN + y) * WIN + x).write(t) };
            }
        }
    }
}

/// The gather function for a level's storage (texel bytes and field) and
/// the texture's class. A storage the table has no row for reads as four
/// 8-bit components (the latch reports it).
pub(crate) fn gather_for(bytes: u8, field: u8, class: Class) -> GatherFn {
    macro_rules! classes {
        ($b:literal, $f:expr) => {
            match class {
                Class::Alpha => gather::<$b, { $f }, 0> as GatherFn,
                Class::Luminance => gather::<$b, { $f }, 1>,
                Class::LuminanceAlpha => gather::<$b, { $f }, 2>,
                Class::Intensity => gather::<$b, { $f }, 3>,
                Class::Rgb => gather::<$b, { $f }, 4>,
                Class::Rgba => gather::<$b, { $f }, 5>,
            }
        };
    }
    match (bytes, field) {
        (2, FIELD_ONE) => classes!(2, FIELD_ONE),
        (2, FIELD_TWO) => classes!(2, FIELD_TWO),
        (2, FIELD_FOUR) => classes!(2, FIELD_FOUR),
        (2, FIELD_PACKED) => classes!(2, FIELD_PACKED),
        (4, FIELD_TWO) => classes!(4, FIELD_TWO),
        (4, FIELD_PACKED) => classes!(4, FIELD_PACKED),
        (8, FIELD_FOUR) => classes!(8, FIELD_FOUR),
        _ => classes!(4, FIELD_FOUR),
    }
}

/// The position along one axis, in 1/256 of a texel of a level 2^l texels
/// wide, as the first tap's index before wrapping and the second tap's
/// weight (LINEAR), or the tap's index (NEAREST): the clamps act on the
/// position here, the wraps on the index (addr()).
#[inline(always)]
fn pos(mut u: i64, l: u32, wrap: Wrap, linear: bool) -> (i64, u32) {
    let whole = 1i64.wrapping_shl(l + 8);
    let last = (1i64 << l.min(62)) - 1;

    if linear {
        u = match wrap {
            Wrap::Repeat => u - 128,
            Wrap::ClampToEdge => (u - 128).max(0).min(whole - 256),
            Wrap::ClampToBorder => u.max(-128).min(whole + 128) - 128,
            Wrap::Clamp => u.max(0).min(whole) - 128,
        };
        return (u >> 8, (u & 255) as u32);
    }
    let x = u >> 8;
    let x = match wrap {
        Wrap::Repeat => x,
        Wrap::ClampToBorder => x.max(-1).min(last + 1),
        _ => x.max(0).min(last),
    };
    (x, 0)
}

/// floor(x) for |x| below 2^31; beyond, the cast saturates.
#[inline(always)]
fn floor_i32(x: f32) -> i32 {
    let i = x as i32;
    i - (x < i as f32) as i32
}

/// A quad's texture coordinates: pixel i (at x + (i & 1), y + (i >> 1))
/// has homogeneous S, T, R and Q in c[i]. The pixels a primitive does not
/// cover are still its helpers for the level of detail.
#[derive(Clone, Copy, Default, Debug)]
#[repr(C)]
pub(crate) struct TexQuad {
    pub c: [[f32; 4]; 4],
}

/// A draw's texture: its registers and latch, and board memory.
#[derive(Clone, Copy)]
#[repr(C)]
pub(crate) struct Unit<'a> {
    pub regs: &'a TexRegs,
    pub latch: &'a Latch,
    pub mem: VramPtr,
}

/// One pixel's taps on one level: the first tap's index per axis before
/// wrapping, and the second taps' weights.
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Taps {
    at: [i64; 3],
    w: [u32; 3],
}

impl Unit<'_> {
    /// Sample a quad: each pixel's position in each axis, the quad's level
    /// of detail from its diagonals, then the texel of each pixel in `cov`
    /// into out in 12.12.
    #[inline]
    pub fn sample_quad(&self, q: &TexQuad, cov: u8, out: &mut [[i32; 4]; 4]) {
        let l = self.latch;
        let r = self.regs;
        let axes = r.target().axes();
        let mut p = [[0i64; 3]; 4];
        let mut d = [[0i64; 2]; 4];

        for i in 0..4 {
            let c = &q.c[i];
            let rq = 1.0 / c[3];
            p[i] = [0, 1, 2].map(|a| if a < axes { floor_i32(c[a] * rq * l.scale[a]) as i64 } else { 0 });
            d[i] = [0, 1].map(|a| floor_i32(c[a] * rq * l.detail_scale[a]) as i64);
        }
        // The diagonal rule: the largest difference along the quad's two
        // diagonals, in 1/256 of a base texel, scaled by the bias.
        let mut lambda = if l.lod {
            let mut rho = 0.0f32;
            for (a, b) in [(0, 3), (1, 2)] {
                for k in 0..3 {
                    rho = rho.max((p[b][k] - p[a][k]).unsigned_abs() as f32 * l.lod_scale[k]);
                }
            }
            let rho = rho.min(u32::MAX as f32) as u32;
            if rho == 0 { LAMBDA_MIN } else { fixed::log2_q8(rho) - (8 << 8) }
        } else {
            0
        };
        // The filter, the levels and their blend lambda chooses, one for
        // the quad.
        let min = r.min();
        let mag = r.mag();
        let mut linear = min.linear();
        let (mut first, mut second, mut frac) = (0u32, 0u32, 0u32);
        let mut detail = false;
        let range = l.count - 1;
        if l.lod {
            lambda = lambda.max(l.min_lod).min(l.max_lod);
            if lambda <= l.mag_below {
                linear = mag != Mag::Nearest;
                detail = mag.detail();
            } else if min.mipmap() {
                if min.nearest_level() {
                    // The nearest level: lambda rounded, half down.
                    first = if lambda <= 128 { 0 } else { ((lambda as u32).wrapping_add(127) >> 8).min(range) };
                } else {
                    let rel = lambda.max(0).min((range << 8) as i32) as u32;
                    first = rel >> 8;
                    frac = rel & 255;
                    second = (first + 1).min(range);
                }
            }
        }
        for pi in p.iter_mut() {
            for a in 0..3 {
                pi[a] = pi[a].min(l.clamp[a]);
            }
        }
        let mut acc = [[0u32; 4]; 4];
        self.level_quad(first, linear, &p, cov, &mut acc);
        if frac != 0 {
            let mut other = [[0u32; 4]; 4];
            self.level_quad(second, linear, &p, cov, &mut other);
            for i in 0..4 {
                for k in 0..4 {
                    acc[i][k] = ((acc[i][k] as u64 * (256 - frac) as u64 + other[i][k] as u64 * frac as u64) >> 8) as u32;
                }
            }
        }
        let mut dt = [[0u32; 4]; 4];
        if detail {
            self.detail_quad(&d, cov, &mut dt);
        }
        let weight = if detail { r.detail_weight(lambda) } else { 0 };
        for i in 0..4 {
            if cov & (1 << i) == 0 {
                continue;
            }
            // 12 bits with 16 fraction bits over 4095, as 12.12 over 4096.
            let mut c = acc[i].map(|v| ((v as u64 * 4097) >> 16) as i32);
            if detail {
                self.apply_detail(weight, &dt[i], &mut c);
            }
            self.post(&mut c);
            out[i] = c;
        }
    }

    /// Each covered pixel's taps on level lv at base positions p.
    #[inline(always)]
    fn taps(&self, lv: &Level, linear: bool, p: &[[i64; 3]; 4], cov: u8) -> [Taps; 4] {
        let r = self.regs;
        let l0 = &self.latch.level[0];
        let target = r.target();
        let mut t = [Taps::default(); 4];

        for i in 0..4 {
            if cov & (1 << i) == 0 {
                continue;
            }
            let (x, a) = pos(p[i][0] >> (l0.wl - lv.wl), lv.wl, r.wrap(0), linear);
            let (y, b) = if target == Target::OneD { (0, 0) } else { pos(p[i][1] >> (l0.hl - lv.hl), lv.hl, r.wrap(1), linear) };
            let (z, c) = if target == Target::ThreeD { pos(p[i][2] >> (l0.dl - lv.dl), lv.dl, r.wrap(2), linear) } else { (0, 0) };
            t[i] = Taps { at: [x, y, z], w: [a, b, c] };
        }
        t
    }

    /// Level i sampled at each covered pixel's base position, each component
    /// 12 bits with 16 fraction bits: the quad's taps gathered once into a
    /// window where they fit one, else each pixel's own.
    fn level_quad(&self, i: u32, linear: bool, p: &[[i64; 3]; 4], cov: u8, out: &mut [[u32; 4]; 4]) {
        let lv = &self.latch.level[i as usize];
        let t = self.taps(lv, linear, p, cov);
        let target = self.regs.target();
        // Taps a pixel takes past its first, per axis.
        let more = [linear as i64, (linear && target != Target::OneD) as i64, (linear && target == Target::ThreeD) as i64];
        let mut lo = [i64::MAX; 3];
        let mut hi = [i64::MIN; 3];
        for i in (0..4).filter(|i| cov & (1 << i) != 0) {
            for a in 0..3 {
                lo[a] = lo[a].min(t[i].at[a]);
                hi[a] = hi[a].max(t[i].at[a] + more[a]);
            }
        }
        if lo[0] > hi[0] {
            return;
        }
        let mut win = std::mem::MaybeUninit::<Win>::uninit();
        let fits = hi[0] - lo[0] < WIN as i64 && hi[1] - lo[1] < WIN as i64 && hi[2] - lo[2] < WIN_SLICES as i64;
        if fits {
            self.gather(lv, lo, [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1], win.as_mut_ptr().cast());
        }
        for i in (0..4).filter(|i| cov & (1 << i) != 0) {
            let base = if fits {
                lo
            } else {
                self.gather(lv, t[i].at, [1 + more[0], 1 + more[1], 1 + more[2]], win.as_mut_ptr().cast());
                t[i].at
            };
            let at = [0, 1, 2].map(|a| (t[i].at[a] - base[a]) as usize);
            // SAFETY: the taps lie in what gather() wrote.
            out[i] = unsafe { filter(win.as_ptr().cast(), at, t[i].w, linear, target == Target::ThreeD) };
        }
    }

    /// Gather level lv's texels from index lo, n on each axis, into win.
    #[inline(always)]
    fn gather(&self, lv: &Level, lo: [i64; 3], n: [i64; 3], win: *mut [u32; 4]) {
        let l = self.latch;
        let border = self.regs.stored_border();
        let mut axes = [Axis::default(); 3];

        for a in 0..3 {
            if a < 2 || self.regs.target() == Target::ThreeD {
                (l.addr[a])(lv, lo[a], n[a] as u32, border, &mut axes[a]);
            } else {
                axes[a].n = 1;
            }
        }
        // SAFETY: win holds a window (the callers' Win).
        unsafe { (lv.gather)(self.mem, lv.addr, &axes[0], &axes[1], &axes[2], &l.border, win) };
    }

    /// The detail image at each covered pixel's position, LINEAR and
    /// repeated, 12 bits with 16 fraction bits.
    fn detail_quad(&self, d: &[[i64; 2]; 4], cov: u8, out: &mut [[u32; 4]; 4]) {
        let lv = &self.latch.detail;
        let mut win = std::mem::MaybeUninit::<Win>::uninit();
        let repeat = [addr_for(Wrap::Repeat, 0), addr_for(Wrap::Repeat, 1)];

        for i in (0..4).filter(|i| cov & (1 << i) != 0) {
            let (x, a) = pos(d[i][0], lv.wl, Wrap::Repeat, true);
            let (y, b) = pos(d[i][1], lv.hl, Wrap::Repeat, true);
            let mut axes = [Axis::default(); 3];
            repeat[0](lv, x, 2, false, &mut axes[0]);
            repeat[1](lv, y, 2, false, &mut axes[1]);
            axes[2].n = 1;
            // SAFETY: win is a window.
            unsafe { (lv.gather)(self.mem, lv.addr, &axes[0], &axes[1], &axes[2], &self.latch.border, win.as_mut_ptr().cast()) };
            // SAFETY: gather() wrote the 2x2.
            out[i] = unsafe { filter(win.as_ptr().cast(), [0; 3], [a, b, 0], true, false) };
        }
    }

    /// The detail image's change to a magnified texel: its signed offset
    /// weighted by the detail function and added or multiplied in (word 0
    /// bit 31 clear adds), per component the filter names.
    fn apply_detail(&self, weight: i32, d: &[u32; 4], texel: &mut [i32; 4]) {
        let r = self.regs;
        let add = r.sampler[0] & 0x8000_0000 == 0;
        let mag = r.mag();

        for k in 0..4 {
            if (k == 3 && mag == Mag::DetailColour) || (k != 3 && mag == Mag::DetailAlpha) {
                continue;
            }
            let other = ((d[k] as u64 * 4097) >> 16) as i32;
            // weight (2 d - 1), in [-1, 1].
            let signed = fixed::smul(2 * other - fixed::ONE, weight);
            let c = if add { texel[k] + signed } else { texel[k] + fixed::smul(signed, texel[k]) };

            texel[k] = fixed::clamp(c);
        }
    }

    /// The post-filter: scale and bias (signed 3.12 each, clamped after,
    /// SGIX_texture_scale_bias's order), then the colour table, each
    /// component indexing its table (red and green the first table's high
    /// and low fields, blue the second's high, alpha the second's low).
    #[inline(always)]
    fn post(&self, c: &mut [i32; 4]) {
        let r = self.regs;
        let sb = &r.scale_bias;

        for k in 0..4 {
            let scale = sextract32(sb[k * 2], 0, tex::SCALE_BIAS_BITS) as i64;
            let bias = sextract32(sb[k * 2 + 1], 0, tex::SCALE_BIAS_BITS) as i64;
            if scale != tex::SCALE_BIAS_ONE as i64 || bias != 0 {
                c[k] = ((c[k] as i64 * scale >> 12) + (bias << 12)).clamp(0, fixed::MAX as i64) as i32;
            }
        }
        let (on, alpha) = r.colour_table();
        if !on {
            return;
        }
        let index = |v: i32| ((v as i64 * 255 + (1 << 23)) >> 24).clamp(0, 255) as usize;
        for k in 0..3 {
            let word = r.colour_table[(k == 2) as usize][index(c[k])];
            c[k] = fixed::field12_full(if k == 1 { word } else { word >> 12 });
        }
        if alpha {
            c[3] = fixed::field12_full(r.colour_table[1][index(c[3])]);
        }
    }

    /// Texel (x, y, z) of level i as the unit reads it, in 12-bit
    /// components, a stored border's at -1 and at the size.
    #[cfg(test)]
    pub fn fetch(&self, i: usize, x: i64, y: i64, z: i64) -> [u32; 4] {
        let mut win = std::mem::MaybeUninit::<Win>::uninit();
        self.gather(&self.latch.level[i], [x, y, z], [1, 1, 1], win.as_mut_ptr().cast());
        // SAFETY: gather() wrote texel 0.
        unsafe { win.assume_init_ref()[0] }
    }
}

/// A pixel's texel from a window: its first tap at window index `at`, the
/// second taps' weights w, LINEAR or NEAREST, over two slices for a volume;
/// 12-bit components with 16 fraction bits.
///
/// # Safety
/// win holds a window, written at every texel the taps read.
#[inline(always)]
unsafe fn filter(win: *const [u32; 4], at: [usize; 3], w: [u32; 3], linear: bool, three: bool) -> [u32; 4] {
    // SAFETY: the caller's taps were gathered.
    let at_ = |x: usize, y: usize, z: usize| unsafe { *win.add(((at[2] + z) * WIN + at[1] + y) * WIN + at[0] + x) };

    if !linear {
        return at_(0, 0, 0).map(|v| v << 16);
    }
    let [a, b, c] = w;
    let plane = |z: usize| -> [u32; 4] {
        let (c00, c10, c01, c11) = (at_(0, 0, z), at_(1, 0, z), at_(0, 1, z), at_(1, 1, z));
        std::array::from_fn(|k| {
            let top = c00[k] * (256 - a) + c10[k] * a;
            let bot = c01[k] * (256 - a) + c11[k] * a;
            top * (256 - b) + bot * b
        })
    };
    if !three {
        return plane(0);
    }
    let (s0, s1) = (plane(0), plane(1));
    std::array::from_fn(|k| ((s0[k] as u64 * (256 - c) as u64 + s1[k] as u64 * c as u64) >> 8) as u32)
}
