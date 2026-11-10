//! The X server's graphics context, as the thirteen words it loads.
//!
//! The words say what the next fills draw: a colour, a background colour, a
//! write mask, a logic operation, and a target (which planes). Only some forms
//! are understood. A form that is not is reported with why, and a draw using
//! it is refused and counted, never guessed.
//!
//! Ported from hw/display/sgi/x-gc.c in jamesbraid/qemu-sgi-origami. Its
//! grades carry over: the field layouts are software expectations read from
//! the senders. What the ported model marks inferred stays inferred here: that the second
//! buffer selector, when it equals the first, is the same selection; that the
//! overlay-both form (config 0x28, selector 3) is the geometry GC loader's
//! form for an 8-bit overlay window drawn into both overlay buffers; and that
//! the 30-bit TrueColor form is told apart by format 9. The canonical colour
//! subsets (8-bit index, 8-bit-per-component RGB) are model subsets, not
//! visual identities.

use crate::dev::kona::back::raster::ops::LogicOp;
use crate::dev::kona::defs::*;

/// Which planes a context draws into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// An 8-bit colour index in the main planes.
    MainCi8,
    /// The native 16-bit main-plane index.
    MainIndex16,
    /// 24-bit RGB in the main planes.
    MainRgb8,
    Overlay0,
    Overlay1,
    Popup,
    /// The display-ID plane.
    Did,
    /// Both overlay buffers at once, the same 8-bit index in each.
    OverlayBoth,
}

/// Why a context was not decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    BadFields,
    BadColour,
    BadMask,
    MixedDid,
    /// A form whose meaning is not known.
    Deferred,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gc {
    pub raw: [u32; GC_WORDS],
    pub target: Target,
    pub foreground: u32,
    pub background: u32,
    pub mask: u32,
    pub width_mask: u32,
    /// The logic operation, 0 to 15 in the GL numbering (clear, and, and
    /// reverse, copy, ...).
    pub gx: u8,
    pub first_selector: u8,
    pub second_selector: u8,
    /// Word 4's bit 31: the X server's intent of an opaque stippled fill (fill style 3) or a tiled
    /// fill with a pipe tile, the only difference on the wire between an opaque and a transparent
    /// stipple (inferred). Whether the board honours it is unknown: the engine forwards the word
    /// whole and the chip probably applies the 0x00ffffff mask to it, which would leave the bit no
    /// effect. The model keys an opaque fill on it, which is an approximation that contradicts that
    /// mask; a raster-side observation of where a recording's opaque pixels come from would settle
    /// it.
    pub opaque_or_tiled: bool,
}

fn index_value(high: u32) -> u32 {
    (high >> 4) & 0xfff0 | (high >> 3) & 0xf
}

fn index_mask(word: u32) -> u32 {
    (word & 0xfff) << 4 | (word >> 20) & 0xf
}

/// Whether a 12-bit colour field is a widened component: an 8-bit one
/// `b << 4 | b >> 4` for 24-bit TrueColor, a 10-bit one `c << 2 | c >> 8` for
/// 30-bit TrueColor, as the X server's context builder widens them.
fn canonical_component(component: u32, rgb30: bool) -> bool {
    if rgb30 {
        let c = component >> 2;
        component == (c << 2 | c >> 8)
    } else {
        let byte = component >> 4;
        component == (byte << 4 | byte >> 4)
    }
}

/// The model keeps eight bits a component: a 30-bit component's low two bits
/// are dropped.
fn canonical_rgb(colour: u64, rgb30: bool) -> Option<u32> {
    const FIELDS: u64 = 0xfff << 40 | 0xfff << 27 | 0xfff << 14 | 0x1ffe;
    let alpha = (colour >> 1) & 0xfff;
    if colour & !FIELDS != 0 || (alpha != 0 && alpha != 0xfff) {
        return None;
    }
    let mut out = 0;
    for k in 1..=3u32 {
        let component = ((colour >> (13 * k + 1)) & 0xfff) as u32;
        if !canonical_component(component, rgb30) {
            return None;
        }
        out |= (component >> 4) << (8 * (k - 1));
    }
    Some(out)
}

fn decode_did(w: &[u32; GC_WORDS], mut gc: Gc) -> Result<Gc, Status> {
    let (high, low) = (w[GC_FOREGROUND_HIGH] != 0 || w[GC_WRITE_MASK_FIRST] != 0, w[GC_FOREGROUND_LOW] != 0 || w[GC_WRITE_MASK_SECOND] != 0);
    gc.target = Target::Did;
    gc.width_mask = 0xffff;
    if w[GC_FRAMEBUFFER_CONFIG] != 8 || w[GC_OPERATION] != 1 || w[GC_FRAMEBUFFER_BASES] != 0 || w[GC_BLEND_AND_LOGIC] != 6 || w[GC_PIXEL_PACK_CODE] != 0 || w[GC_TEXTURE_FILTER_COLOUR_FLAG] != 1 || w[GC_TEXTURE_FILTER_PIXEL_FLAG] != 1 || w[GC_BACKGROUND_HIGH] != 0 || w[GC_BACKGROUND_LOW] != 0 {
        return Err(Status::BadFields);
    }
    if high && low {
        return Err(Status::MixedDid);
    }
    if w[GC_FOREGROUND_HIGH] & !0xfff78 != 0 || w[GC_FOREGROUND_LOW] & !0x3ffde00 != 0 {
        return Err(Status::BadColour);
    }
    if w[GC_WRITE_MASK_FIRST] & !0xf0_0fff != 0 || w[GC_WRITE_MASK_SECOND] & !0xfff_00f != 0 {
        return Err(Status::BadMask);
    }
    if high {
        gc.foreground = index_value(w[GC_FOREGROUND_HIGH]);
        gc.mask = index_mask(w[GC_WRITE_MASK_FIRST]);
    } else {
        gc.foreground = (w[GC_FOREGROUND_LOW] >> 9) & 0xf | (w[GC_FOREGROUND_LOW] >> 10) & 0xfff0;
        gc.mask = (w[GC_WRITE_MASK_SECOND] >> 12) & 0xfff | (w[GC_WRITE_MASK_SECOND] & 0xf) << 12;
    }
    Ok(gc)
}

impl Gc {
    /// Decode the words the way the X fills do: the native 16-bit index form
    /// for configurations 0xc and 0x180c, the 8-bit form otherwise.
    pub fn decode_for_fill(w: &[u32; GC_WORDS]) -> Result<Gc, Status> {
        if w[GC_FRAMEBUFFER_CONFIG] == 0xc || w[GC_FRAMEBUFFER_CONFIG] == 0x180c {
            Gc::decode_main_index(w)
        } else {
            Gc::decode(w)
        }
    }

    pub fn decode(w: &[u32; GC_WORDS]) -> Result<Gc, Status> {
        decode(w, false)
    }

    /// Keep the full native main-plane index and mask.
    pub fn decode_main_index(w: &[u32; GC_WORDS]) -> Result<Gc, Status> {
        let gc = decode(w, true)?;
        if gc.target == Target::MainIndex16 { Ok(gc) } else { Err(Status::BadFields) }
    }

    /// The write that changes nothing: no bits enabled, or the no-op operation.
    pub fn is_noop(&self) -> bool {
        self.mask == 0 || self.gx == 5
    }

    /// Whether a write depends on the pixel it replaces.
    pub fn needs_destination(&self) -> bool {
        if self.is_noop() {
            return false;
        }
        self.mask != self.width_mask || !matches!(self.gx, 0 | 3 | 12 | 15)
    }

    /// The word written for `source` over `destination`.
    pub fn apply(&self, source: u32, destination: u32) -> u32 {
        ((LogicOp::from_index(self.gx as u32).apply(source, destination) & self.mask) | (destination & !self.mask)) & self.width_mask
    }
}

fn decode(w: &[u32; GC_WORDS], native_index: bool) -> Result<Gc, Status> {
    let plane = w[GC_FRAMEBUFFER_CONFIG] & 0x3c;
    let mask_word = w[GC_WRITE_MASK_FIRST] & !0x8000_0000;
    // The X server's context builder copies the buffer selector in op bits
    // 1:0 into bits 3:2 for a format with bit 0x80000. What the second copy
    // selects is unknown: a copy equal to the first reads as the same
    // selection (inferred), any other value is deferred.
    let op = if (w[GC_OPERATION] >> 2) & 3 == w[GC_OPERATION] & 3 { w[GC_OPERATION] & !0xc } else { w[GC_OPERATION] };
    let residual = op & !0xfc_0003;
    let geometry = w[GC_PIXEL_PACK_CODE] == 0;
    let rgb30 = w[GC_FRAMEBUFFER_CONFIG] == 0 && (w[GC_PIXEL_PACK_CODE] == 0 || w[GC_PIXEL_PACK_CODE] == 9);

    if w[GC_BLEND_AND_LOGIC] > 30 || w[GC_BLEND_AND_LOGIC] & 1 != 0 {
        return Err(Status::BadFields);
    }
    let mut gc = Gc {
        raw: *w,
        target: Target::MainCi8,
        foreground: 0,
        background: 0,
        mask: 0,
        width_mask: 0,
        gx: (w[GC_BLEND_AND_LOGIC] / 2) as u8,
        first_selector: (w[GC_OPERATION] & 3) as u8,
        second_selector: ((w[GC_OPERATION] >> 18) & 0x3f) as u8,
        opaque_or_tiled: w[GC_WRITE_MASK_FIRST] & 0x8000_0000 != 0,
    };
    if w[GC_FRAMEBUFFER_CONFIG] == 8 {
        return decode_did(w, gc);
    }
    // Config 0x28 with buffer selector 3 is the geometry GC loader's form for
    // an 8-bit overlay window drawn into both overlay buffers (inferred); a
    // 15-bit overlay format also gets 0x28 whatever its selector, and those
    // forms stay deferred.
    if (plane == 0x28 && !(w[GC_FRAMEBUFFER_CONFIG] == 0x28 && op == 3)) || (w[GC_FRAMEBUFFER_CONFIG] == 0x1008 && w[GC_OPERATION] == 0x10001 && w[GC_PIXEL_PACK_CODE] == 5) {
        return Err(Status::Deferred);
    }
    match w[GC_FRAMEBUFFER_CONFIG] {
        0 => {
            // 30-bit TrueColor: config 0 from every builder, told apart by
            // format 9, or 0 from the geometry builder (inferred).
            if !rgb30 {
                return Err(Status::BadFields);
            }
            gc.target = Target::MainRgb8;
            gc.width_mask = 0xff_ffff;
        }
        0xc | 0x180c => {
            gc.target = if native_index { Target::MainIndex16 } else { Target::MainCi8 };
            gc.width_mask = if native_index { 0xffff } else { 0xff };
        }
        4 | 0x2804 | 0x3804 => {
            gc.target = Target::MainRgb8;
            gc.width_mask = 0xff_ffff;
        }
        0x2c | 0x182c | 0x102c => {
            gc.target = Target::Overlay0;
            gc.width_mask = 0xffff;
        }
        0x30 | 0x1030 => return Err(Status::Deferred),
        0x1830 => {
            gc.target = Target::Overlay1;
            gc.width_mask = 0xffff;
        }
        0x28 => {
            gc.target = Target::OverlayBoth;
            gc.width_mask = 0xff;
        }
        0x20 | 0x1820 | 0x1020 => {
            gc.target = Target::Popup;
            gc.width_mask = 3;
        }
        _ => return Err(Status::BadFields),
    }
    if op & 4 != 0 {
        return Err(Status::Deferred);
    }
    if !rgb30 && geometry != (w[GC_FRAMEBUFFER_CONFIG] < 0x1000) {
        return Err(Status::BadFields);
    }
    if matches!(gc.target, Target::MainCi8 | Target::MainIndex16 | Target::MainRgb8) {
        let rgb = gc.target == Target::MainRgb8;
        let tail = if rgb { 0 } else { 1 };
        if w[GC_FRAMEBUFFER_BASES] != 1 || w[GC_TEXTURE_FILTER_COLOUR_FLAG] != tail || w[GC_TEXTURE_FILTER_PIXEL_FLAG] != tail {
            return Err(Status::BadFields);
        }
        if geometry {
            if op != 1 && !(rgb && w[GC_FRAMEBUFFER_CONFIG] == 4 && op == 3) {
                return Err(Status::BadFields);
            }
        } else if rgb30 {
            // Selector 2 has no config form to show it in: deferred.
            if op != 0x64001 {
                return Err(Status::Deferred);
            }
        } else if w[GC_PIXEL_PACK_CODE] != if rgb { 8 } else { 5 }
            || !(1..=2).contains(&gc.first_selector)
            || !(1..=2).contains(&gc.second_selector)
            || residual != if rgb { 0x2_0000 } else { 0 }
            || (rgb && ((w[GC_FRAMEBUFFER_CONFIG] == 0x3804) != (gc.second_selector == 2)))
        {
            return Err(Status::BadFields);
        }
    } else if w[GC_FRAMEBUFFER_BASES] != 0
        || w[GC_TEXTURE_FILTER_COLOUR_FLAG] != 1
        || w[GC_TEXTURE_FILTER_PIXEL_FLAG] != 1
        || (!geometry && w[GC_PIXEL_PACK_CODE] != 5)
        || op != match gc.target {
            Target::Overlay1 => 2,
            Target::OverlayBoth => 3,
            _ => 1,
        }
    {
        return Err(Status::BadFields);
    }

    if gc.target == Target::MainRgb8 {
        let (red, green, blue) = (mask_word & 0xfff, (mask_word >> 12) & 0xfff, w[GC_WRITE_MASK_SECOND] & 0xfff);
        let (Some(fg), Some(bg)) = (canonical_rgb((w[GC_FOREGROUND_HIGH] as u64) << 32 | w[GC_FOREGROUND_LOW] as u64, rgb30), canonical_rgb((w[GC_BACKGROUND_HIGH] as u64) << 32 | w[GC_BACKGROUND_LOW] as u64, rgb30)) else {
            return Err(Status::BadColour);
        };
        gc.foreground = fg;
        gc.background = bg;
        if mask_word & !0xff_ffff != 0
            || w[GC_WRITE_MASK_SECOND] & !0xff_ffff != 0
            || w[GC_WRITE_MASK_SECOND] & 0xfff000 != 0xfff000
            || !canonical_component(red, rgb30)
            || !canonical_component(green, rgb30)
            || !canonical_component(blue, rgb30)
        {
            return Err(Status::BadMask);
        }
        gc.mask = (red >> 4) << 16 | (green >> 4) << 8 | (blue >> 4);
    } else {
        if w[GC_FOREGROUND_HIGH] & !0xfff78 != 0 || w[GC_BACKGROUND_HIGH] & !0xfff78 != 0 || (w[GC_FOREGROUND_LOW] != 0 && w[GC_FOREGROUND_LOW] != 0x1ffe) || (w[GC_BACKGROUND_LOW] != 0 && w[GC_BACKGROUND_LOW] != 0x1ffe) {
            return Err(Status::BadColour);
        }
        if mask_word & !0xf0_0fff != 0 || w[GC_WRITE_MASK_SECOND] != 0 {
            return Err(Status::BadMask);
        }
        gc.foreground = index_value(w[GC_FOREGROUND_HIGH]);
        gc.background = index_value(w[GC_BACKGROUND_HIGH]);
        gc.mask = index_mask(mask_word);
        match gc.target {
            Target::MainCi8 => {
                if (gc.foreground | gc.background) & 0xff != 0 {
                    return Err(Status::BadColour);
                }
                if gc.mask & 0xff != 0xff {
                    return Err(Status::BadMask);
                }
                gc.foreground >>= 8;
                gc.background >>= 8;
                gc.mask >>= 8;
            }
            Target::OverlayBoth => {
                // The two bytes are one index repeated; keep one of them.
                if gc.foreground >> 8 != gc.foreground & 0xff || gc.background >> 8 != gc.background & 0xff {
                    return Err(Status::BadColour);
                }
                if gc.mask >> 8 != gc.mask & 0xff {
                    return Err(Status::BadMask);
                }
                gc.foreground &= 0xff;
                gc.background &= 0xff;
                gc.mask &= 0xff;
            }
            Target::Popup => {
                if mask_word & !0x30_0000 != 0 {
                    return Err(Status::BadMask);
                }
                gc.foreground &= 3;
                gc.background &= 3;
                gc.mask = (mask_word >> 20) & 3;
            }
            _ => {}
        }
    }
    Ok(gc)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CI: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];
    const RGB: [u32; 13] = [0, 0, 0, 0, 0xffffff, 0xffffff, 0x2804, 0x60001, 1, 6, 8, 0, 0];
    const DID: [u32; 13] = [0, 0, 0, 0, 0, 0xfc000, 8, 1, 0, 6, 0, 1, 1];

    fn ok(w: &[u32; 13]) -> Gc {
        Gc::decode(w).unwrap()
    }

    /// The engine applies a mask to each of words 4 to 12 before the raster register sees it, so a form
    /// the decoder takes has no bit outside its word's mask.
    #[test]
    fn the_forms_the_decoder_takes_fit_the_engines_masks() {
        for w in [CI, RGB, DID] {
            ok(&w);
            for (k, mask) in GC_WORD_MASKS.iter().enumerate() {
                assert_eq!(w[GC_WRITE_MASK_FIRST + k] & !mask, 0, "word {} of {w:x?}", GC_WRITE_MASK_FIRST + k);
            }
        }
    }

    #[test]
    fn an_index_context_decodes() {
        let gc = ok(&CI);
        assert_eq!((gc.target, gc.foreground, gc.background, gc.mask), (Target::MainCi8, 0x5a, 0xa5, 0xff));
        assert_eq!(gc.raw, CI);
        let mut w = CI;
        w[GC_WRITE_MASK_FIRST] = 0xf0053f; // logical mask 0x53, alignment-fill bits 0xff
        let gc = ok(&w);
        assert_eq!(gc.mask, 0x53);
        assert_eq!(gc.apply(0x5a, 0xa5), 0xf6);
        w[GC_WRITE_MASK_FIRST] = 0xf0000f;
        assert!(ok(&w).is_noop());
        w[GC_WRITE_MASK_FIRST] = CI[4] ^ 1;
        assert_eq!(Gc::decode(&w), Err(Status::BadMask));
        w[GC_WRITE_MASK_FIRST] = CI[4];
        w[GC_FOREGROUND_HIGH] |= 0x100;
        assert_eq!(Gc::decode(&w), Err(Status::BadColour));
    }

    #[test]
    fn an_rgb_context_decodes_every_widened_byte() {
        let mut w = RGB;
        for b in 0..256u64 {
            let component = b << 4 | b >> 4;
            let colour = component << 40 | component << 27 | component << 14 | 0x1ffe;
            w[GC_FOREGROUND_HIGH] = (colour >> 32) as u32;
            w[GC_FOREGROUND_LOW] = colour as u32;
            w[GC_WRITE_MASK_FIRST] = (component | component << 12) as u32;
            w[GC_WRITE_MASK_SECOND] = 0xfff000 | component as u32;
            let gc = ok(&w);
            assert_eq!((gc.foreground, gc.mask), (b as u32 * 0x010101, b as u32 * 0x010101));
        }
        let mut w = RGB;
        for m in 0..4096u32 {
            let byte = m >> 4;
            let canonical = m == (byte << 4 | byte >> 4);
            w[GC_WRITE_MASK_FIRST] = 0xfff000 | m;
            assert_eq!(Gc::decode(&w).map(|_| ()), if canonical { Ok(()) } else { Err(Status::BadMask) }, "{m:#x}");
        }
        let mut w = RGB;
        w[GC_FOREGROUND_HIGH] = 0xff000;
        assert_eq!(Gc::decode(&w), Err(Status::BadColour));
        w[GC_FOREGROUND_HIGH] = 0;
        w[GC_BACKGROUND_HIGH] = 0xff000;
        assert_eq!(Gc::decode(&w), Err(Status::BadColour));
    }

    #[test]
    fn a_display_id_context_decodes_both_halves() {
        for v in (0..=0xffffu32).step_by(7).chain([0xffff]) {
            let mut w = DID;
            w[GC_FOREGROUND_HIGH] = (v & 15) << 3 | (v & 0xfff0) << 4;
            w[GC_WRITE_MASK_FIRST] = (v >> 4) & 0xfff | (v & 15) << 20;
            w[GC_WRITE_MASK_SECOND] = 0;
            let gc = ok(&w);
            assert_eq!((gc.target, gc.foreground, gc.mask), (Target::Did, v, v));
            w[GC_FOREGROUND_HIGH] = 0;
            w[GC_WRITE_MASK_FIRST] = 0;
            w[GC_FOREGROUND_LOW] = (v & 15) << 9 | (v & 0xfff0) << 10;
            w[GC_WRITE_MASK_SECOND] = (v >> 12) & 15 | (v & 0xfff) << 12;
            let gc = ok(&w);
            assert_eq!((gc.foreground, gc.mask), (v, v));
        }
        let mut w = DID;
        assert_eq!(ok(&w).apply(0, 0xa5c3), 0xa503);
        w[GC_WRITE_MASK_FIRST] = 0x300a5c;
        assert_eq!(Gc::decode(&w), Err(Status::MixedDid));
        w[GC_WRITE_MASK_SECOND] = 0;
        w[GC_FOREGROUND_HIGH] = 0x80;
        assert_eq!(Gc::decode(&w), Err(Status::BadColour));
        let mut w = DID;
        w[GC_BLEND_AND_LOGIC] = 10;
        assert_eq!(Gc::decode(&w), Err(Status::BadFields));
    }

    #[test]
    fn the_ancillary_plane_sets_decode_to_their_targets() {
        for (cfg, target) in [
            (0x182c, Target::Overlay0),
            (0x2c, Target::Overlay0),
            (0x102c, Target::Overlay0),
            (0x1830, Target::Overlay1),
            (0x1820, Target::Popup),
            (0x20, Target::Popup),
            (0x1020, Target::Popup),
        ] {
            let mut w = CI;
            w[GC_FOREGROUND_HIGH] = 0x78;
            w[GC_BACKGROUND_HIGH] = 0x38;
            w[GC_WRITE_MASK_FIRST] = if target == Target::Popup { 0x300000 } else { 0xf00000 };
            w[GC_FRAMEBUFFER_CONFIG] = cfg;
            w[GC_OPERATION] = if cfg & 0x3c == 0x30 { 2 } else { 1 };
            w[GC_FRAMEBUFFER_BASES] = 0;
            if cfg < 0x1000 {
                w[GC_PIXEL_PACK_CODE] = 0;
            }
            let gc = ok(&w);
            let (fg, m) = if target == Target::Popup { (3, 3) } else { (15, 15) };
            assert_eq!((gc.target, gc.foreground, gc.mask), (target, fg, m), "{cfg:#x}");
        }
        let mut w = CI;
        w[GC_FRAMEBUFFER_CONFIG] = 0xc;
        w[GC_OPERATION] = 1;
        w[GC_PIXEL_PACK_CODE] = 0;
        assert_eq!(ok(&w).target, Target::MainCi8);
        w[GC_FOREGROUND_HIGH] = 0x93000;
        assert_eq!(ok(&w).target, Target::MainCi8);
        w[GC_WRITE_MASK_FIRST] |= 0x8000_0000;
        assert!(ok(&w).opaque_or_tiled);
        let mut w = RGB;
        w[GC_OPERATION] = 0x7c001;
        assert_eq!(Gc::decode(&w), Err(Status::BadFields));
        w[GC_FRAMEBUFFER_CONFIG] = 0;
        w[GC_OPERATION] = 0x64001;
        w[GC_PIXEL_PACK_CODE] = 9;
        assert_eq!(ok(&w).target, Target::MainRgb8);
        w[GC_OPERATION] = 0xa4001;
        assert_eq!(Gc::decode(&w), Err(Status::Deferred));
        assert_eq!(Gc::decode(&[0; 13]), Err(Status::BadFields));
    }

    /// The 30-bit packing: each 10-bit component to `c << 2 | c >> 8`; the
    /// model keeps the top eight bits.
    #[test]
    fn a_thirty_bit_context_keeps_eight_bits_a_component() {
        fn words(pixel: u32) -> (u32, u32) {
            let mut colour = 0x1ffeu64;
            for k in 1..=3u32 {
                let c = (pixel >> (10 * (k - 1))) & 0x3ff;
                colour |= ((c << 2 | c >> 8) as u64) << (13 * k + 1);
            }
            ((colour >> 32) as u32, colour as u32)
        }
        let mut w = [0, 0, 0, 0, 0xffffff, 0xffffff, 0, 0x64001, 1, 6, 9, 0, 0];
        (w[GC_FOREGROUND_HIGH], w[GC_FOREGROUND_LOW]) = words(0x123 << 20 | 0x2c7 << 10 | 0x0a5);
        (w[GC_BACKGROUND_HIGH], w[GC_BACKGROUND_LOW]) = words(0x3ff << 20 | 0x001 << 10 | 0x200);
        let gc = ok(&w);
        assert_eq!((gc.target, gc.foreground, gc.background, gc.mask), (Target::MainRgb8, 0x48b129, 0xff0080, 0xffffff));
        w[GC_OPERATION] = 1;
        w[GC_PIXEL_PACK_CODE] = 0;
        assert_eq!(ok(&w).foreground, 0x48b129);
        w[GC_FOREGROUND_LOW] ^= 1 << 14;
        assert_eq!(Gc::decode(&w), Err(Status::BadColour));
        w[GC_FOREGROUND_LOW] ^= 1 << 14;
        w[GC_PIXEL_PACK_CODE] = 5;
        assert_eq!(Gc::decode(&w), Err(Status::BadFields));
    }

    #[test]
    fn a_repeated_buffer_selector_reads_as_the_same_selection() {
        let mut w = RGB;
        w[GC_OPERATION] = 0x60005;
        let gc = ok(&w);
        assert_eq!((gc.target, gc.first_selector), (Target::MainRgb8, 1));
        w[GC_FRAMEBUFFER_CONFIG] = 4;
        w[GC_OPERATION] = 5;
        w[GC_PIXEL_PACK_CODE] = 0;
        assert!(Gc::decode(&w).is_ok());
        w[GC_OPERATION] = 6;
        assert_eq!(Gc::decode(&w), Err(Status::Deferred));
        let mut w = CI;
        w[GC_OPERATION] = 0x40005;
        assert!(Gc::decode(&w).is_ok());
        w[GC_OPERATION] = 0x40006;
        assert_eq!(Gc::decode(&w), Err(Status::Deferred));
    }

    #[test]
    fn masks_and_operations_apply_as_the_gc_says() {
        for gx in 0..16u32 {
            let mut w = CI;
            w[GC_BLEND_AND_LOGIC] = 2 * gx;
            w[GC_WRITE_MASK_FIRST] = 0xf0053f;
            let gc = ok(&w);
            assert_eq!(gc.gx as u32, gx);
            assert_eq!(gc.apply(0x195, 0x26a), (LogicOp::from_index(gx).apply(0x95, 0x6a) & 0x53) | (0x6a & 0xac));
            assert_eq!(gc.is_noop(), gx == 5);
            assert_eq!(gc.needs_destination(), gx != 5);
        }
        let mut w = CI;
        for gx in 0..16u32 {
            w[GC_BLEND_AND_LOGIC] = 2 * gx;
            assert_eq!(ok(&w).needs_destination(), !matches!(gx, 0 | 3 | 5 | 12 | 15));
        }
        for bl in (1..32).step_by(2) {
            w[GC_BLEND_AND_LOGIC] = bl;
            assert_eq!(Gc::decode(&w), Err(Status::BadFields));
        }
        w[GC_BLEND_AND_LOGIC] = 32;
        assert_eq!(Gc::decode(&w), Err(Status::BadFields));
    }

    #[test]
    fn reserved_bits_refuse() {
        let mut w = CI;
        w[GC_FRAMEBUFFER_CONFIG] = 0x1820;
        w[GC_OPERATION] = 1;
        w[GC_FRAMEBUFFER_BASES] = 0;
        for bit in 22..=23 {
            w[GC_WRITE_MASK_FIRST] = 0x300000 | 1 << bit;
            assert_eq!(Gc::decode(&w), Err(Status::BadMask));
        }
        for pair in [0, 2] {
            for gap in [0, 13, 26, 39, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63] {
                let mut w = RGB;
                let v = 1u64 << gap;
                w[pair] = (v >> 32) as u32;
                w[pair + 1] = v as u32;
                assert_eq!(Gc::decode(&w), Err(Status::BadColour), "pair {pair} gap {gap}");
            }
            let mut w = RGB;
            w[pair + 1] = 2;
            assert_eq!(Gc::decode(&w), Err(Status::BadColour));
            w[pair + 1] = 0x1ffe;
            assert!(Gc::decode(&w).is_ok());
        }
        let mut w = CI;
        (w[GC_FRAMEBUFFER_CONFIG], w[GC_OPERATION], w[GC_FRAMEBUFFER_BASES], w[GC_PIXEL_PACK_CODE]) = (0x30, 2, 0, 0);
        assert_eq!(Gc::decode(&w), Err(Status::Deferred));
        (w[GC_FRAMEBUFFER_CONFIG], w[GC_PIXEL_PACK_CODE]) = (0x1030, 5);
        assert_eq!(Gc::decode(&w), Err(Status::Deferred));
    }

    #[test]
    fn the_overlay_layer_clear_decodes_and_its_siblings_defer() {
        let mut w = [0, 0x1ffe, 0, 0x1ffe, 0xf00fff, 0, 0x28, 3, 0, 6, 0, 1, 1];
        let gc = ok(&w);
        assert_eq!((gc.target, gc.foreground, gc.mask, gc.width_mask), (Target::OverlayBoth, 0, 0xff, 0xff));
        w[GC_FOREGROUND_HIGH] = 0x5a5 << 8 | 0xa << 3;
        assert_eq!(ok(&w).foreground, 0x5a);
        w[GC_FOREGROUND_HIGH] = 0x5a000;
        assert_eq!(Gc::decode(&w), Err(Status::BadColour));
        w[GC_FOREGROUND_HIGH] = 0;
        w[GC_WRITE_MASK_FIRST] = 0xfff;
        assert_eq!(Gc::decode(&w), Err(Status::BadMask));
        w[GC_WRITE_MASK_FIRST] = 0xf00fff;
        w[GC_OPERATION] = 1;
        assert_eq!(Gc::decode(&w), Err(Status::Deferred));
        (w[GC_FRAMEBUFFER_CONFIG], w[GC_PIXEL_PACK_CODE], w[GC_OPERATION]) = (0x1828, 5, 3);
        assert_eq!(Gc::decode(&w), Err(Status::Deferred));
    }

    /// The fills read configurations 0xc and 0x180c as the native 16-bit index.
    #[test]
    fn a_fill_reads_the_native_index_form_for_its_two_configurations() {
        let mut w = CI;
        w[GC_FRAMEBUFFER_CONFIG] = 0xc;
        w[GC_OPERATION] = 1;
        w[GC_PIXEL_PACK_CODE] = 0;
        assert_eq!(Gc::decode_for_fill(&w).unwrap().target, Target::MainIndex16);
        assert_eq!(Gc::decode_main_index(&CI).unwrap().target, Target::MainIndex16);
        assert_eq!(Gc::decode_main_index(&RGB), Err(Status::BadFields));
        assert_eq!(Gc::decode_for_fill(&RGB).unwrap().target, Target::MainRgb8);
    }
}
