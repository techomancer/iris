//! Pure functions on typed values for the per-pixel module: comparisons and colour
//! packing. No state, no store access: each is a function of its arguments, so a
//! GPU kernel could reproduce it.

use crate::dev::kona::defs::{gl, WINDOW_ID_GROUP_FIRST, WINDOW_ID_GROUP_LAST, WINDOW_ID_ORDINARY_LAST, WINDOW_ID_POPUP};

/// A GL comparison function, typed when the draw state is built so nothing per
/// pixel matches an enumerant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Compare {
    Never,
    Less,
    Equal,
    Lequal,
    Greater,
    Notequal,
    Gequal,
    #[default]
    Always,
}

impl Compare {
    /// The function GL names, or None for a value that is not one.
    pub fn from_gl(v: u32) -> Option<Compare> {
        Some(match v {
            gl::NEVER => Compare::Never,
            gl::LESS => Compare::Less,
            gl::EQUAL => Compare::Equal,
            gl::LEQUAL => Compare::Lequal,
            gl::GREATER => Compare::Greater,
            gl::NOTEQUAL => Compare::Notequal,
            gl::GEQUAL => Compare::Gequal,
            gl::ALWAYS => Compare::Always,
            _ => return None,
        })
    }

    /// Whether `new` passes against `old`.
    #[inline(always)]
    pub fn passes(self, new: u32, old: u32) -> bool {
        match self {
            Compare::Never => false,
            Compare::Less => new < old,
            Compare::Equal => new == old,
            Compare::Lequal => new <= old,
            Compare::Greater => new > old,
            Compare::Notequal => new != old,
            Compare::Gequal => new >= old,
            Compare::Always => true,
        }
    }
}

/// What a stencil test does to the stored value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StencilOp {
    #[default]
    Keep,
    Zero,
    Replace,
    Incr,
    Decr,
    Invert,
}

impl StencilOp {
    pub fn from_gl(v: u32) -> Option<StencilOp> {
        Some(match v {
            gl::KEEP => StencilOp::Keep,
            gl::ZERO => StencilOp::Zero,
            gl::REPLACE => StencilOp::Replace,
            gl::INCR => StencilOp::Incr,
            gl::DECR => StencilOp::Decr,
            gl::INVERT => StencilOp::Invert,
            _ => return None,
        })
    }

    /// The new value of stored `s`, with reference `r`; increment and decrement
    /// saturate at the 8-bit range.
    #[inline(always)]
    pub fn apply(self, s: u8, r: u8) -> u8 {
        match self {
            StencilOp::Keep => s,
            StencilOp::Zero => 0,
            StencilOp::Replace => r,
            StencilOp::Incr => s.saturating_add(1),
            StencilOp::Decr => s.saturating_sub(1),
            StencilOp::Invert => !s,
        }
    }
}

/// A blend factor, by what it multiplies by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    #[default]
    One,
    SrcColour,
    OneMinusSrcColour,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    DstColour,
    OneMinusDstColour,
    SrcAlphaSaturate,
    ConstantColour,
    OneMinusConstantColour,
    ConstantAlpha,
    OneMinusConstantAlpha,
}

/// The full-intensity value of a 13-bit component.
pub const FULL: u32 = 8191;

impl BlendFactor {
    pub fn from_gl(v: u32) -> Option<BlendFactor> {
        Some(match v {
            gl::ZERO => BlendFactor::Zero,
            gl::ONE => BlendFactor::One,
            gl::SRC_COLOR => BlendFactor::SrcColour,
            gl::ONE_MINUS_SRC_COLOR => BlendFactor::OneMinusSrcColour,
            gl::SRC_ALPHA => BlendFactor::SrcAlpha,
            gl::ONE_MINUS_SRC_ALPHA => BlendFactor::OneMinusSrcAlpha,
            gl::DST_ALPHA => BlendFactor::DstAlpha,
            gl::ONE_MINUS_DST_ALPHA => BlendFactor::OneMinusDstAlpha,
            gl::DST_COLOR => BlendFactor::DstColour,
            gl::ONE_MINUS_DST_COLOR => BlendFactor::OneMinusDstColour,
            gl::SRC_ALPHA_SATURATE => BlendFactor::SrcAlphaSaturate,
            gl::CONSTANT_COLOR => BlendFactor::ConstantColour,
            gl::ONE_MINUS_CONSTANT_COLOR => BlendFactor::OneMinusConstantColour,
            gl::CONSTANT_ALPHA => BlendFactor::ConstantAlpha,
            gl::ONE_MINUS_CONSTANT_ALPHA => BlendFactor::OneMinusConstantAlpha,
            _ => return None,
        })
    }

    /// The factor's weight on each component (red, green, blue, alpha), in 13-bit
    /// units, for source `s`, destination `d` and constant `k`.
    #[inline(always)]
    pub fn weights(self, s: [u32; 4], d: [u32; 4], k: [u32; 4]) -> [u32; 4] {
        let one_minus = |v: [u32; 4]| v.map(|c| FULL - c);
        match self {
            BlendFactor::Zero => [0; 4],
            BlendFactor::One => [FULL; 4],
            BlendFactor::SrcColour => s,
            BlendFactor::OneMinusSrcColour => one_minus(s),
            BlendFactor::SrcAlpha => [s[3]; 4],
            BlendFactor::OneMinusSrcAlpha => [FULL - s[3]; 4],
            BlendFactor::DstAlpha => [d[3]; 4],
            BlendFactor::OneMinusDstAlpha => [FULL - d[3]; 4],
            BlendFactor::DstColour => d,
            BlendFactor::OneMinusDstColour => one_minus(d),
            BlendFactor::SrcAlphaSaturate => {
                let f = s[3].min(FULL - d[3]);
                [f, f, f, FULL]
            }
            BlendFactor::ConstantColour => k,
            BlendFactor::OneMinusConstantColour => one_minus(k),
            BlendFactor::ConstantAlpha => [k[3]; 4],
            BlendFactor::OneMinusConstantAlpha => [FULL - k[3]; 4],
        }
    }
}

/// `src * sf + dst * df` on each of four 13-bit components. The rounding (to
/// nearest), the clamp and the 13-bit units are the model's own: no source gives them.
#[inline(always)]
pub fn blend(src: [u32; 4], dst: [u32; 4], sf: BlendFactor, df: BlendFactor, eq: BlendEq, k: [u32; 4]) -> [u32; 4] {
    let BlendEq::Add = eq;
    let (sw, dw) = (sf.weights(src, dst, k), df.weights(src, dst, k));
    std::array::from_fn(|i| ((src[i] * sw[i] + dst[i] * dw[i] + FULL / 2) / FULL).min(FULL))
}

/// The blend equation. Only add is modelled: the board's equation token has no
/// layout read, so this is a model limit, and the type is here so a second
/// equation is a variant and not a flag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendEq {
    #[default]
    Add,
}

/// A logic operation: GL's sixteen, in the order the X server numbers them. Shared
/// with the X path, which numbers them 0 to 15 and sends anything above as a copy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LogicOp {
    Clear,
    And,
    AndReverse,
    #[default]
    Copy,
    AndInverted,
    Noop,
    Xor,
    Or,
    Nor,
    Equiv,
    Invert,
    OrReverse,
    CopyInverted,
    OrInverted,
    Nand,
    Set,
}

impl LogicOp {
    /// The operation GL names, or None for a value that is not one.
    pub fn from_gl(v: u32) -> Option<LogicOp> {
        v.checked_sub(gl::LOGIC_OP_CLEAR).filter(|&n| n < 16).map(LogicOp::from_index)
    }

    /// The operation the X server's number `n` names; above 15 is a copy.
    #[inline(always)]
    pub fn from_index(n: u32) -> LogicOp {
        match n {
            0 => LogicOp::Clear,
            1 => LogicOp::And,
            2 => LogicOp::AndReverse,
            3 => LogicOp::Copy,
            4 => LogicOp::AndInverted,
            5 => LogicOp::Noop,
            6 => LogicOp::Xor,
            7 => LogicOp::Or,
            8 => LogicOp::Nor,
            9 => LogicOp::Equiv,
            10 => LogicOp::Invert,
            11 => LogicOp::OrReverse,
            12 => LogicOp::CopyInverted,
            13 => LogicOp::OrInverted,
            14 => LogicOp::Nand,
            15 => LogicOp::Set,
            _ => LogicOp::Copy,
        }
    }

    /// The operation on a source `s` and destination `d`.
    #[inline(always)]
    pub fn apply(self, s: u32, d: u32) -> u32 {
        match self {
            LogicOp::Clear => 0,
            LogicOp::And => s & d,
            LogicOp::AndReverse => s & !d,
            LogicOp::Copy => s,
            LogicOp::AndInverted => !s & d,
            LogicOp::Noop => d,
            LogicOp::Xor => s ^ d,
            LogicOp::Or => s | d,
            LogicOp::Nor => !(s | d),
            LogicOp::Equiv => !(s ^ d),
            LogicOp::Invert => !d,
            LogicOp::OrReverse => s | !d,
            LogicOp::CopyInverted => !s,
            LogicOp::OrInverted => !s | d,
            LogicOp::Nand => !(s & d),
            LogicOp::Set => !0,
        }
    }
}

/// The window-ID test the board's register holds: the reference ID and the mask of the bits
/// compared. A fragment passes when `((plane ^ reference) & mask) == 0` (the polarity is inferred
/// from the values: every pair the encoder and the kernel make has its ID bits inside its mask
/// bits, and the other polarity would compare no window's ID at all).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ownership {
    pub reference: u32,
    pub mask: u32,
}

impl Ownership {
    /// What the board encodes the window-placement request's word 9 as, and whether
    /// that word was one of the ranges it names (anything else is encoded as the
    /// ordinary pair 0x3f, 0x3f and counted by the caller).
    pub fn encode(word9: i32) -> (Ownership, bool) {
        match word9 {
            0..=WINDOW_ID_ORDINARY_LAST => (Ownership { reference: word9 as u32, mask: 0x3f }, true),
            WINDOW_ID_POPUP => (Ownership { reference: 0x800, mask: 0x800 }, true),
            WINDOW_ID_GROUP_FIRST..=WINDOW_ID_GROUP_LAST => (Ownership { reference: ((word9 - WINDOW_ID_GROUP_FIRST) as u32) << 6, mask: 0x7c0 }, true),
            // The kernel's null clip sends -1 with the test on, to make a pipe own nothing (established): the
            // board encodes it as the ordinary pair, which no pixel the shipped software paints matches.
            -1 => (Ownership { reference: 0x3f, mask: 0x3f }, true),
            _ => (Ownership { reference: 0x3f, mask: 0x3f }, false),
        }
    }

    /// The window-ID test register's value for this ownership with the enable set.
    #[cfg(test)]
    pub fn register(&self) -> u32 {
        use crate::dev::kona::defs::WINDOW_ID_TEST as T;
        T::MASK.put(T::REFERENCE_ID.put(T::ENABLE.put(0, 1), self.reference), self.mask)
    }

    /// Whether a pixel whose compared plane value is `plane` is the window's.
    #[inline(always)]
    pub fn owns(&self, plane: u32) -> bool {
        (plane ^ self.reference) & self.mask == 0
    }
}

/// The 12-bit value the ownership test compares for a pixel: the painted value shifted right
/// by 2 (inferred). The X server paints main window ID n as `n << 2` and an overlay or popup group g
/// as `g << 8`, so the compared value holds the main ID in bits 5:0 and the group in bits 10:6 (the
/// ID-plane word of the store holds the painted value, both fields).
///
/// One reading is the model's: the encoder compares a popup window at bit 11 (reference and mask
/// 0x800), which the painter never sets, since it paints the popup as group 31. Taking a group
/// field of all ones as the popup (the table's leading reading, inferred) sets bit 11 for it. An
/// overlay window of group 31 and a popup window then own the same pixels, as the painter gives them
/// the same pattern.
#[inline(always)]
pub fn ownership_plane(painted: u32) -> u32 {
    (painted >> 2 & 0xfff) | ((painted >> 8 & 0x1f == 0x1f) as u32) << 11
}

/// A plane word `0xAARRGGBB` as four 13-bit components (red, green, blue, alpha).
#[inline(always)]
pub fn unpack_colour(w: u32) -> [u32; 4] {
    let c = |shift: u32| ((w >> shift & 0xff) * FULL + 127) / 255;
    [c(16), c(8), c(0), c(24)]
}

/// A colour component of 13 bits as 8, rounded.
#[inline(always)]
pub fn to8(c13: u32) -> u32 {
    (c13 * 255 + 4095) / 8191
}

/// Red, green, blue and alpha of 13 bits each as the plane word `0xAARRGGBB`.
#[inline(always)]
pub fn pack_colour(c: [u32; 4]) -> u32 {
    to8(c[3]) << 24 | to8(c[0]) << 16 | to8(c[1]) << 8 | to8(c[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparisons_follow_gl_and_an_unknown_function_passes() {
        let c = |f: u32| Compare::from_gl(f).unwrap();
        assert!(c(gl::LESS).passes(1, 2) && !c(gl::LESS).passes(2, 2));
        assert!(c(gl::LEQUAL).passes(2, 2) && c(gl::EQUAL).passes(3, 3) && !c(gl::NEVER).passes(0, 0));
        assert!(c(gl::GREATER).passes(3, 2) && c(gl::GEQUAL).passes(2, 2) && c(gl::NOTEQUAL).passes(1, 2));
        assert!(c(gl::ALWAYS).passes(0, 9));
        assert_eq!(Compare::from_gl(0x1234), None);
    }

    #[test]
    fn colour_packs_to_eight_bits_a_component_rounded() {
        assert_eq!(pack_colour([8191, 0, 0, 8191]), 0xffff_0000);
        assert_eq!(pack_colour([0, 8191, 4096, 0]), 0x00_00_ff_80);
        assert_eq!(to8(0), 0);
    }

    #[test]
    fn stencil_operations_saturate_at_the_eight_bit_range() {
        assert_eq!((StencilOp::Incr.apply(255, 9), StencilOp::Incr.apply(3, 9)), (255, 4));
        assert_eq!((StencilOp::Decr.apply(0, 9), StencilOp::Decr.apply(3, 9)), (0, 2));
        assert_eq!((StencilOp::Replace.apply(3, 9), StencilOp::Zero.apply(3, 9), StencilOp::Keep.apply(3, 9), StencilOp::Invert.apply(0x0f, 0)), (9, 0, 3, 0xf0));
        assert_eq!(StencilOp::from_gl(0x1e02), Some(StencilOp::Incr));
        assert_eq!(StencilOp::from_gl(0x1234), None);
    }

    /// Source-over: (alpha, 1 - alpha) gives the weighted average, rounded; (one, zero)
    /// returns the source; (zero, one) the destination; additive saturates.
    #[test]
    fn blending_weighs_source_and_destination() {
        let (s, d, k) = ([FULL, 0, 4096, 4096], [0, FULL, 0, FULL], [0; 4]);
        assert_eq!(blend(s, d, BlendFactor::One, BlendFactor::Zero, BlendEq::Add, k), s);
        assert_eq!(blend(s, d, BlendFactor::Zero, BlendFactor::One, BlendEq::Add, k), d);
        let over = blend(s, d, BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendEq::Add, k);
        assert_eq!(over, [4096, 4095, 2048, 6143], "{over:?}: alpha 4096 of 8191 is just over half");
        assert_eq!(blend(s, s, BlendFactor::One, BlendFactor::One, BlendEq::Add, k), [FULL, 0, FULL, FULL], "additive saturates");
        let constant = blend(s, d, BlendFactor::ConstantColour, BlendFactor::OneMinusConstantColour, BlendEq::Add, [FULL, 0, FULL, 0]);
        assert_eq!(constant[..3], [FULL, FULL, 4096], "a constant of (1, 0, 1) takes source red and blue, destination green");
        // Saturate weighs by min(source alpha, 1 - destination alpha): none left of an opaque destination.
        assert_eq!(blend(s, d, BlendFactor::SrcAlphaSaturate, BlendFactor::Zero, BlendEq::Add, k)[0], 0);
        assert_eq!(blend(s, [0; 4], BlendFactor::SrcAlphaSaturate, BlendFactor::Zero, BlendEq::Add, k)[0], (FULL * 4096 + FULL / 2) / FULL);
    }

    #[test]
    fn the_stored_word_widens_to_thirteen_bits_and_packs_back_exactly() {
        for v in 0..=255u32 {
            let w = v << 24 | v << 16 | (255 - v) << 8 | v;
            assert_eq!(pack_colour(unpack_colour(w).map(|c| c)), w & 0xffff_ffff, "{v}");
        }
        assert_eq!(unpack_colour(0xff00_ff00), [0, FULL, 0, FULL]);
    }

    /// The sixteen operations by their truth tables over (source, destination), in the
    /// order the X server numbers them and GL names them.
    #[test]
    fn logic_operations_follow_their_truth_tables() {
        const TRUTH: [u32; 16] = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
        let (s, d) = (0x963c_a569u32, 0x5a81_c30fu32);
        for (op, t) in TRUTH.iter().enumerate() {
            let want = (0..32).fold(0u32, |acc, bit| {
                let index = ((s >> bit) & 1) << 1 | (d >> bit) & 1;
                acc | ((t >> index) & 1) << bit
            });
            assert_eq!(LogicOp::from_index(op as u32).apply(s, d), want, "operation {op}");
        }
        assert_eq!(LogicOp::from_index(u32::MAX).apply(s, d), s);
        assert_eq!(LogicOp::from_index(3).apply(0xf0, 0x0f), 0xf0);
        assert_eq!(LogicOp::from_index(6).apply(0xf0, 0xff), 0x0f);
        assert_eq!(LogicOp::from_index(16).apply(5, 6), 5, "above 15 is a copy");
        assert_eq!(LogicOp::from_gl(gl::LOGIC_OP_CLEAR + 6), Some(LogicOp::Xor));
        assert_eq!(LogicOp::from_gl(0x1510), None);
    }

    /// The encoder's worked values: the recorded word 9 of 0xe and 0xf give the
    /// registers 0x7e01d and 0x7e01f; the overlay context's 0x44 is group 4 and
    /// gives reference 0x100 with mask 0x7c0; the popup and the all-ones case.
    #[test]
    fn word_nine_encodes_as_the_board_does() {
        assert_eq!(Ownership::encode(0xe).0.register(), 0x7e01d);
        assert_eq!(Ownership::encode(0xf).0.register(), 0x7e01f);
        assert_eq!(Ownership::encode(0x44).0, Ownership { reference: 0x100, mask: 0x7c0 });
        assert_eq!(Ownership::encode(0x3f).0, Ownership { reference: 0x800, mask: 0x800 });
        assert_eq!(Ownership::encode(-1), (Ownership { reference: 0x3f, mask: 0x3f }, true), "the null clip: the ordinary pair, not counted odd");
        assert_eq!(Ownership::encode(0x60), (Ownership { reference: 0x3f, mask: 0x3f }, false), "outside every range: the ordinary pair, flagged");
        assert_eq!(Ownership::encode(-5).1, false);
    }

    /// An ordinary window owns the pixels painted with its ID and no other; a group window owns its
    /// group whatever the main field holds; the popup owns a group of all ones; all ones, the null clip, owns
    /// a pixel only where the main field is 0x3f, which the painter never writes.
    #[test]
    fn ownership_compares_the_bits_the_mask_names() {
        let own = |w: i32, painted: u32| Ownership::encode(w).0.owns(ownership_plane(painted));
        // Painted values: main ID n is n << 2, an overlay group g is g << 8.
        assert!(own(5, 5 << 2) && !own(5, 6 << 2));
        assert!(own(5, 5 << 2 | 2 << 8), "the overlay group beside the main ID does not matter to an ordinary window");
        assert!(own(0x44, 4 << 8) && own(0x44, 37 << 2 | 4 << 8), "group 4 whatever the main ID");
        assert!(!own(0x44, 5 << 8) && !own(0x44, 37 << 2), "no group is not group 4");
        assert!(own(0x40, 0), "group 0 is the pixels with no overlay painted");
        assert!(own(0x3f, 0x1f << 8) && !own(0x3f, 5 << 2), "the popup owns the group of all ones");
        assert!(own(0x5f, 0x1f << 8), "and so does an overlay window of group 31");
        assert!(!(0..0x3f).any(|n| own(-1, n << 2)), "no main ID the painter writes is 0x3f");
        assert!(own(-1, 0x3f << 2), "a pixel whose main field is 0x3f is the only one the null clip owns");
    }
}
