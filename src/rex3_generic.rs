//! Generic REX3 draw path.
//!
//! **One expression of the draw logic, usable two ways.** The pipeline is
//! written once against the [`Mode`] trait; a [`ConstMode`] implementor carries
//! the draw-mode fields as const generics (so every `if m.blend() != 0` folds at
//! monomorphisation and LLVM deletes the dead arms — the same specialisation
//! Cranelift gets), while [`DynMode`] holds them as plain fields read from the
//! registers.
//!
//! That is what makes a generated table possible without a second
//! implementation: a table entry instantiates the pipeline with `ConstMode`, and
//! anything not in the table runs the *same source* with `DynMode`. Const
//! generic arguments must be compile-time literals, so the table is the bridge
//! from runtime register values to constants — but the logic it dispatches into
//! is shared, not duplicated.

use crate::rex3::{
    Rex3, Rex3Context, OCTANT_XDEC, OCTANT_YDEC, REX3_BRES_OCTANTS, CLIPMODE_CIDMATCH_SHIFT, CLIPMODE_ENSMASK_MASK,
    CLIPMODE_ENSMASK_SMASK0, CLIPMODE_ENSMASK_SMASK1_4,
    DRAWMODE0_ADRMODE_A_LINE, DRAWMODE0_ADRMODE_BLOCK, DRAWMODE0_ADRMODE_F_LINE,
    DRAWMODE0_ADRMODE_I_LINE, DRAWMODE0_ADRMODE_SPAN, DRAWMODE0_OPCODE_DRAW,
    DRAWMODE0_OPCODE_NOOP, DRAWMODE0_OPCODE_READ, DRAWMODE0_OPCODE_SCR2SCR,
    DRAWMODE1_PLANES_CID, DRAWMODE1_PLANES_OLAY, DRAWMODE1_PLANES_PUP,
    DRAWMODE1_PLANES_RGB, DRAWMODE1_PLANES_RGBA,
    REX3_COORD_BIAS, REX3_SCREEN_HEIGHT, REX3_SCREEN_WIDTH,
    DRAWMODE1_BF_MOC, DRAWMODE1_BF_MSA, DRAWMODE1_BF_OC, DRAWMODE1_BF_ONE,
    DRAWMODE1_BF_SA, DRAWMODE1_BF_ZERO,
    DRAWMODE1_COMPARE_EQ, DRAWMODE1_COMPARE_GE, DRAWMODE1_COMPARE_GT,
    DRAWMODE1_COMPARE_LE, DRAWMODE1_COMPARE_LT, DRAWMODE1_COMPARE_NE,
    DRAWMODE1_COMPARE_NEVER,
    DRAWMODE1_DRAWDEPTH_12, DRAWMODE1_DRAWDEPTH_24, DRAWMODE1_DRAWDEPTH_4,
    DRAWMODE1_DRAWDEPTH_8, DRAWMODE1_HOSTDEPTH_12, DRAWMODE1_HOSTDEPTH_8,
    DRAWMODE1_HOSTDEPTH_4,
    DRAWMODE1_LOGICOP_ZERO,
    DRAWMODE1_LOGICOP_AND,
    DRAWMODE1_LOGICOP_ANDR,
    DRAWMODE1_LOGICOP_SRC,
    DRAWMODE1_LOGICOP_ANDI,
    DRAWMODE1_LOGICOP_DST,
    DRAWMODE1_LOGICOP_XOR,
    DRAWMODE1_LOGICOP_OR,
    DRAWMODE1_LOGICOP_NOR,
    DRAWMODE1_LOGICOP_XNOR,
    DRAWMODE1_LOGICOP_NDST,
    DRAWMODE1_LOGICOP_ORR,
    DRAWMODE1_LOGICOP_NSRC,
    DRAWMODE1_LOGICOP_ORI,
    DRAWMODE1_LOGICOP_NAND,
};
use crate::rex3_shape as sh;

/// The two framebuffer planes, as the JIT shader receives them.
///
/// Matching the compiled shader's `(ctx, fb_rgb, fb_aux)` ABI means the pixel
/// path needs no `&Rex3` at all: everything it touches is the context plus these
/// two pointers.
#[derive(Clone, Copy)]
pub struct Framebuffers {
    pub rgb: *mut u32,
    pub aux: *mut u32,
}

// ── Mode ─────────────────────────────────────────────────────────────────────

/// The draw-mode fields the pipeline branches on.
///
/// Implemented twice: once with const generics, once with runtime fields. Every
/// accessor is `#[inline(always)]`, so under [`ConstMode`] each call becomes a
/// literal and the surrounding branch disappears; under [`DynMode`] it is a
/// field read.
pub trait Mode: Copy {
    fn opcode(&self) -> u32;
    fn planes(&self) -> u32;
    fn drawdepth(&self) -> u32;
    fn dblsrc(&self) -> u32;
    fn rgbmode(&self) -> u32;
    fn dither(&self) -> u32;
    fn logicop(&self) -> u32;
    fn compare(&self) -> u32;
    fn blend(&self) -> u32;
    fn backblend(&self) -> u32;
    fn blendalpha(&self) -> u32;
    fn fastclear(&self) -> u32;
    fn colorhost(&self) -> u32;
    fn alphahost(&self) -> u32;
    fn enzpattern(&self) -> u32;
    fn enlspattern(&self) -> u32;
    fn zpopaque(&self) -> u32;
    fn lsopaque(&self) -> u32;
    fn cidtest(&self) -> u32;
    fn shade(&self) -> u32;
    fn ciclamp(&self) -> u32;
    fn stoponx(&self) -> u32;
    fn stopony(&self) -> u32;
    fn skipfirst(&self) -> u32;
    fn skiplast(&self) -> u32;
    fn lsadvlast(&self) -> u32;
    fn length32(&self) -> u32;
    fn lronly(&self) -> u32;
    fn ystride(&self) -> u32;
    fn endptfilter(&self) -> u32;
    fn adrmode(&self) -> u32;
    fn xyoffset(&self) -> u32;
    fn yflip(&self) -> u32;
    fn ensmask(&self) -> u32;
    /// CIDMATCH nibble. Data rather than a code selector — `cidtest` decides
    /// whether the test is emitted — but it lives here so there is one mode
    /// object rather than two.
    fn cid_mask(&self) -> u32;
    fn sfactor(&self) -> u32;
    fn dfactor(&self) -> u32;
    fn rwpacked(&self) -> u32;
    fn hostdepth(&self) -> u32;
    fn rwdouble(&self) -> u32;
    fn swapendian(&self) -> u32;
}

/// Compile-time mode: every field a const generic, so the pipeline specialises.
#[derive(Clone, Copy)]
pub struct ConstMode<
    const OPCODE: u32,
    const PLANES: u32,
    const DRAWDEPTH: u32,
    const DBLSRC: u32,
    const RGBMODE: u32,
    const DITHER: u32,
    const LOGICOP: u32,
    const COMPARE: u32,
    const BLEND: u32,
    const BACKBLEND: u32,
    const BLENDALPHA: u32,
    const FASTCLEAR: u32,
    const COLORHOST: u32,
    const ALPHAHOST: u32,
    const ENZPATTERN: u32,
    const ENLSPATTERN: u32,
    const ZPOPAQUE: u32,
    const LSOPAQUE: u32,
    const CIDTEST: u32,
    const SHADE: u32,
    const CICLAMP: u32,
    const STOPONX: u32,
    const STOPONY: u32,
    const SKIPFIRST: u32,
    const SKIPLAST: u32,
    const LSADVLAST: u32,
    const LENGTH32: u32,
    const LRONLY: u32,
    const YSTRIDE: u32,
    const ENDPTFILTER: u32,
    const ADRMODE: u32,
    const XYOFFSET: u32,
    const YFLIP: u32,
    const ENSMASK: u32,
    const CID_MASK: u32,
    const SFACTOR: u32,
    const DFACTOR: u32,
    const RWPACKED: u32,
    const HOSTDEPTH: u32,
    const RWDOUBLE: u32,
    const SWAPENDIAN: u32,
>;

#[rustfmt::skip]
impl<
    const OPCODE: u32, const PLANES: u32, const DRAWDEPTH: u32, const DBLSRC: u32,
    const RGBMODE: u32, const DITHER: u32, const LOGICOP: u32, const COMPARE: u32,
    const BLEND: u32, const BACKBLEND: u32, const BLENDALPHA: u32, const FASTCLEAR: u32,
    const COLORHOST: u32, const ALPHAHOST: u32, const ENZPATTERN: u32,
    const ENLSPATTERN: u32, const ZPOPAQUE: u32, const LSOPAQUE: u32, const CIDTEST: u32,
    const SHADE: u32, const CICLAMP: u32, const STOPONX: u32, const STOPONY: u32,
    const SKIPFIRST: u32, const SKIPLAST: u32, const LSADVLAST: u32, const LENGTH32: u32,
    const LRONLY: u32, const YSTRIDE: u32, const ENDPTFILTER: u32, const ADRMODE: u32,
    const XYOFFSET: u32, const YFLIP: u32, const ENSMASK: u32, const CID_MASK: u32,
    const SFACTOR: u32, const DFACTOR: u32, const RWPACKED: u32, const HOSTDEPTH: u32,
    const RWDOUBLE: u32, const SWAPENDIAN: u32,
> Mode for ConstMode<
    OPCODE, PLANES, DRAWDEPTH, DBLSRC, RGBMODE, DITHER, LOGICOP, COMPARE, BLEND,
    BACKBLEND, BLENDALPHA, FASTCLEAR, COLORHOST, ALPHAHOST, ENZPATTERN, ENLSPATTERN,
    ZPOPAQUE, LSOPAQUE, CIDTEST, SHADE, CICLAMP, STOPONX, STOPONY, SKIPFIRST, SKIPLAST,
    LSADVLAST, LENGTH32, LRONLY, YSTRIDE, ENDPTFILTER, ADRMODE, XYOFFSET, YFLIP,
    ENSMASK, CID_MASK, SFACTOR, DFACTOR, RWPACKED, HOSTDEPTH, RWDOUBLE, SWAPENDIAN,
> {
    #[inline(always)] fn opcode(&self)      -> u32 { OPCODE }
    #[inline(always)] fn planes(&self)      -> u32 { PLANES }
    #[inline(always)] fn drawdepth(&self)   -> u32 { DRAWDEPTH }
    #[inline(always)] fn dblsrc(&self)      -> u32 { DBLSRC }
    #[inline(always)] fn rgbmode(&self)     -> u32 { RGBMODE }
    #[inline(always)] fn dither(&self)      -> u32 { DITHER }
    #[inline(always)] fn logicop(&self)     -> u32 { LOGICOP }
    #[inline(always)] fn compare(&self)     -> u32 { COMPARE }
    #[inline(always)] fn blend(&self)       -> u32 { BLEND }
    #[inline(always)] fn backblend(&self)   -> u32 { BACKBLEND }
    #[inline(always)] fn blendalpha(&self)  -> u32 { BLENDALPHA }
    #[inline(always)] fn fastclear(&self)   -> u32 { FASTCLEAR }
    #[inline(always)] fn colorhost(&self)   -> u32 { COLORHOST }
    #[inline(always)] fn alphahost(&self)   -> u32 { ALPHAHOST }
    #[inline(always)] fn enzpattern(&self)  -> u32 { ENZPATTERN }
    #[inline(always)] fn enlspattern(&self) -> u32 { ENLSPATTERN }
    #[inline(always)] fn zpopaque(&self)    -> u32 { ZPOPAQUE }
    #[inline(always)] fn lsopaque(&self)    -> u32 { LSOPAQUE }
    #[inline(always)] fn cidtest(&self)     -> u32 { CIDTEST }
    #[inline(always)] fn shade(&self)       -> u32 { SHADE }
    #[inline(always)] fn ciclamp(&self)     -> u32 { CICLAMP }
    #[inline(always)] fn stoponx(&self)     -> u32 { STOPONX }
    #[inline(always)] fn stopony(&self)     -> u32 { STOPONY }
    #[inline(always)] fn skipfirst(&self)   -> u32 { SKIPFIRST }
    #[inline(always)] fn skiplast(&self)    -> u32 { SKIPLAST }
    #[inline(always)] fn lsadvlast(&self)   -> u32 { LSADVLAST }
    #[inline(always)] fn length32(&self)    -> u32 { LENGTH32 }
    #[inline(always)] fn lronly(&self)      -> u32 { LRONLY }
    #[inline(always)] fn ystride(&self)     -> u32 { YSTRIDE }
    #[inline(always)] fn endptfilter(&self) -> u32 { ENDPTFILTER }
    #[inline(always)] fn adrmode(&self)     -> u32 { ADRMODE }
    #[inline(always)] fn xyoffset(&self)    -> u32 { XYOFFSET }
    #[inline(always)] fn yflip(&self)       -> u32 { YFLIP }
    #[inline(always)] fn ensmask(&self)     -> u32 { ENSMASK }
    #[inline(always)] fn cid_mask(&self)    -> u32 { CID_MASK }
    #[inline(always)] fn sfactor(&self)     -> u32 { SFACTOR }
    #[inline(always)] fn dfactor(&self)     -> u32 { DFACTOR }
    #[inline(always)] fn rwpacked(&self)    -> u32 { RWPACKED }
    #[inline(always)] fn hostdepth(&self)   -> u32 { HOSTDEPTH }
    #[inline(always)] fn rwdouble(&self)    -> u32 { RWDOUBLE }
    #[inline(always)] fn swapendian(&self)  -> u32 { SWAPENDIAN }
}

/// Runtime mode: the same fields, read from the registers.
///
/// Instantiating the pipeline with this gives a single correct-for-everything
/// instance — the permanent fallback for shapes no table entry covers.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
#[rustfmt::skip]
pub struct DynMode {
    pub opcode: u32, pub planes: u32, pub drawdepth: u32, pub dblsrc: u32,
    pub rgbmode: u32, pub dither: u32, pub logicop: u32, pub compare: u32,
    pub blend: u32, pub backblend: u32, pub blendalpha: u32, pub fastclear: u32,
    pub colorhost: u32, pub alphahost: u32, pub enzpattern: u32, pub enlspattern: u32,
    pub zpopaque: u32, pub lsopaque: u32, pub cidtest: u32, pub shade: u32,
    pub ciclamp: u32, pub stoponx: u32, pub stopony: u32, pub skipfirst: u32,
    pub skiplast: u32, pub lsadvlast: u32, pub length32: u32, pub lronly: u32,
    pub ystride: u32, pub endptfilter: u32, pub adrmode: u32, pub xyoffset: u32,
    pub yflip: u32, pub ensmask: u32, pub cid_mask: u32,
    // Blend inputs and host-transfer shape. Not read by the pixel path directly
    // (blend goes through Rex3::blend, host setup through host_setup), but they
    // are part of the canonical mode: unpack folds them to 0 when dead, which is
    // what keeps two equivalent draws producing one identical mode.
    pub sfactor: u32, pub dfactor: u32,
    pub rwpacked: u32, pub hostdepth: u32, pub rwdouble: u32, pub swapendian: u32,
}

#[rustfmt::skip]
impl Mode for DynMode {
    #[inline(always)] fn opcode(&self)      -> u32 { self.opcode }
    #[inline(always)] fn planes(&self)      -> u32 { self.planes }
    #[inline(always)] fn drawdepth(&self)   -> u32 { self.drawdepth }
    #[inline(always)] fn dblsrc(&self)      -> u32 { self.dblsrc }
    #[inline(always)] fn rgbmode(&self)     -> u32 { self.rgbmode }
    #[inline(always)] fn dither(&self)      -> u32 { self.dither }
    #[inline(always)] fn logicop(&self)     -> u32 { self.logicop }
    #[inline(always)] fn compare(&self)     -> u32 { self.compare }
    #[inline(always)] fn blend(&self)       -> u32 { self.blend }
    #[inline(always)] fn backblend(&self)   -> u32 { self.backblend }
    #[inline(always)] fn blendalpha(&self)  -> u32 { self.blendalpha }
    #[inline(always)] fn fastclear(&self)   -> u32 { self.fastclear }
    #[inline(always)] fn colorhost(&self)   -> u32 { self.colorhost }
    #[inline(always)] fn alphahost(&self)   -> u32 { self.alphahost }
    #[inline(always)] fn enzpattern(&self)  -> u32 { self.enzpattern }
    #[inline(always)] fn enlspattern(&self) -> u32 { self.enlspattern }
    #[inline(always)] fn zpopaque(&self)    -> u32 { self.zpopaque }
    #[inline(always)] fn lsopaque(&self)    -> u32 { self.lsopaque }
    #[inline(always)] fn cidtest(&self)     -> u32 { self.cidtest }
    #[inline(always)] fn shade(&self)       -> u32 { self.shade }
    #[inline(always)] fn ciclamp(&self)     -> u32 { self.ciclamp }
    #[inline(always)] fn stoponx(&self)     -> u32 { self.stoponx }
    #[inline(always)] fn stopony(&self)     -> u32 { self.stopony }
    #[inline(always)] fn skipfirst(&self)   -> u32 { self.skipfirst }
    #[inline(always)] fn skiplast(&self)    -> u32 { self.skiplast }
    #[inline(always)] fn lsadvlast(&self)   -> u32 { self.lsadvlast }
    #[inline(always)] fn length32(&self)    -> u32 { self.length32 }
    #[inline(always)] fn lronly(&self)      -> u32 { self.lronly }
    #[inline(always)] fn ystride(&self)     -> u32 { self.ystride }
    #[inline(always)] fn endptfilter(&self) -> u32 { self.endptfilter }
    #[inline(always)] fn adrmode(&self)     -> u32 { self.adrmode }
    #[inline(always)] fn xyoffset(&self)    -> u32 { self.xyoffset }
    #[inline(always)] fn yflip(&self)       -> u32 { self.yflip }
    #[inline(always)] fn ensmask(&self)     -> u32 { self.ensmask }
    #[inline(always)] fn cid_mask(&self)    -> u32 { self.cid_mask }
    #[inline(always)] fn sfactor(&self)     -> u32 { self.sfactor }
    #[inline(always)] fn dfactor(&self)     -> u32 { self.dfactor }
    #[inline(always)] fn rwpacked(&self)    -> u32 { self.rwpacked }
    #[inline(always)] fn hostdepth(&self)   -> u32 { self.hostdepth }
    #[inline(always)] fn rwdouble(&self)    -> u32 { self.rwdouble }
    #[inline(always)] fn swapendian(&self)  -> u32 { self.swapendian }
}

// ── Colour conversion and DDA primitives ─────────────────────────────────────
// Moved here from `impl Rex3`: none of these ever touched `self`. They were
// associated functions only because the function-pointer draw path selected
// among them by pointer, and reaching them from here then needed a `_pub`
// forwarding shim per function. Both layers are gone.

// ── Shade iterate functions ────────────────────────────────────────────────
// Called once per pixel after drawing.  Advance color DDAs and, when
// CICLAMP is set, clamp the result to legal range.
//
// The spec (§3.8 / DRAWMODE0 bit CICLAMP) says:
//   • RGB mode: each component is in o12.11 format (integer part bits[22:11]).
//     Clamp: if negative (bit 31 set) or integer >= 0x180 → 0; if > 0xFF → 0x7FFFF.
//   • CI mode: only colorred clamped; depth-specific overflow bit check.
//     8bpp: clamp if bit 19 set.  12bpp: clamp if bit 21 set.
//     (4bpp: bit 15; 24bpp: no clamp per spec.)

#[inline(always)]
pub(crate) fn shade_add(ctx: &mut Rex3Context) {
    ctx.colorred   = ctx.colorred.wrapping_add(ctx.slopered   as u32);
    ctx.colorgrn   = ctx.colorgrn.wrapping_add(ctx.slopegrn   as u32);
    ctx.colorblue  = ctx.colorblue.wrapping_add(ctx.slopeblue  as u32);
    ctx.coloralpha = ctx.coloralpha.wrapping_add(ctx.slopealpha as u32);
}

// ── Pattern iterate functions ──────────────────────────────────────────────
// Called once per pixel after drawing.  Advance lspattern and/or zpattern
// bit counters.
//
// ZPATTERN: always 32-bit, simple rotate — zpat_bit = (zpat_bit - 1) & 31.
//
// LSPATTERN (via lsmode):
//   LSRCOUNT  (down counter, 0..LSREPEAT-1): decremented each pixel.
//   When LSRCOUNT == 0 after decrement → advance pat_bit, reload LSRCOUNT = LSREPEAT-1.
//   LSLENGTH  (4 bits): pattern length = lslength + 17 (range 17..32).
//   pat_bit wraps: when it would go below (32 - length), reset to 31.
//   LSREPEAT==0 is treated as 1 (no-repeat is the degenerate case).
//
// Cursors start at 31 (MSB) on DOSETUP/row start (see execute_go) and walk
// DOWN toward the wrap point — a contiguous MSB-to-LSB sweep. f2d0bff
// briefly flipped this to increment-from-31 (+ an lspattern rotate-on-wrap
// in place of the reset), but incrementing from 31 immediately wraps to 0
// after a single step (`(31+1)&31 == 0`), producing a discontinuous,
// backwards bit walk — confirmed as the cause of garbled/mirrored PROM
// text. Reverted back to decrement/reset-to-31.

#[inline(always)]
pub(crate) fn advance_zpat(ctx: &mut Rex3Context) {
    ctx.zpat_bit = ctx.zpat_bit.wrapping_sub(1) & 31;
}

#[inline(always)]
pub(crate) fn advance_lspat(ctx: &mut Rex3Context) {
    let lsrepeat = ctx.lsmode.lsrepeat() as u8;
    let repeat = if lsrepeat == 0 { 1 } else { lsrepeat };
    if ctx.lsmode.lsrcount() == 0 {
        // Reload counter, advance bit
        ctx.lsmode.set_lsrcount((repeat - 1) as u32);
        let length = ctx.lsmode.lslength() as u8 + 17; // 17..=32
        let wrap_point = 32u8.saturating_sub(length);  // bit index of pattern end
        if ctx.pat_bit == wrap_point {
            ctx.pat_bit = 31; // recirculate
        } else {
            ctx.pat_bit = ctx.pat_bit.wrapping_sub(1) & 31;
        }
    } else {
        ctx.lsmode.set_lsrcount(ctx.lsmode.lsrcount() - 1);
    }
}

pub(crate) fn rgb4_to_rgb24(val: u32) -> u32 {
    let r = if (val & 1) != 0 { 0xFF } else { 0 };
    let g_raw = (val >> 1) & 3;
    let g = (g_raw << 6) | (g_raw << 4) | (g_raw << 2) | g_raw;
    let b = if (val & 8) != 0 { 0xFF } else { 0 };
    (b << 16) | (g << 8) | r
}

pub(crate) fn rgb24_to_rgb4(val: u32) -> u32 {
    let r = (val >> 7) & 1;
    let g = (val >> 14) & 3;
    let b = (val >> 23) & 1;
    (b << 3) | (g << 1) | r
}

pub(crate) fn rgb8_to_rgb24(val: u32) -> u32 {
    let r_raw = val & 7;
    let r = (r_raw << 5) | (r_raw << 2) | (r_raw >> 1);
    let g_raw = (val >> 3) & 7;
    let g = (g_raw << 5) | (g_raw << 2) | (g_raw >> 1);
    let b_raw = (val >> 6) & 3;
    let b = (b_raw << 6) | (b_raw << 4) | (b_raw << 2) | b_raw;
    (b << 16) | (g << 8) | r
}

pub(crate) fn rgb24_to_rgb8(val: u32) -> u32 {
    let r = (val >> 5) & 7;
    let g = (val >> 13) & 7;
    let b = (val >> 22) & 3;
    (b << 6) | (g << 3) | r
}

pub(crate) fn rgb12_to_rgb24(val: u32) -> u32 {
    let r_raw = val & 0xF;
    let r = (r_raw << 4) | r_raw;
    let g_raw = (val >> 4) & 0xF;
    let g = (g_raw << 4) | g_raw;
    let b_raw = (val >> 8) & 0xF;
    let b = (b_raw << 4) | b_raw;
    (b << 16) | (g << 8) | r
}

pub(crate) fn rgb24_to_rgb12(val: u32) -> u32 {
    let r = (val >> 4) & 0xF;
    let g = (val >> 12) & 0xF;
    let b = (val >> 20) & 0xF;
    (b << 8) | (g << 4) | r
}

// Bayer 4x4 dither matrix packed as 16 nibbles in a u64.
// Indexed by (y&3)<<2|(x&3): threshold = (BAYER_PACKED >> (idx*4)) & 0xF.
// Table: [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5]
const BAYER_PACKED: u64 = 0x5D7F91B36E4CA280;

/// Pack bayer index into bits 27:24 of color value (top byte unused by 24-bit BGR).
/// Encoding: bits[3:2] = y&3, bits[1:0] = x&3 → index = (y&3)<<2|(x&3).
/// Non-dither compress variants ignore these bits.
#[inline(always)]
pub(crate) fn bayer_pack(color: u32, x: i32, y: i32) -> u32 {
    (color & 0x00FFFFFF) | (((y as u32 & 3) << 2 | (x as u32 & 3)) << 24)
}

#[inline(always)]
pub(crate) fn bayer_threshold(idx: u32) -> u32 {
    ((BAYER_PACKED >> (idx * 4)) & 0xF) as u32
}

pub(crate) fn rgb24_to_rgb4_dither(val: u32) -> u32 {
    let bayer = bayer_threshold(val >> 24);
    let r = (val & 0xFF) as u8;
    let g = ((val >> 8) & 0xFF) as u8;
    let b = ((val >> 16) & 0xFF) as u8;
    // 4bpp: 1-2-1 BGR. Each channel dithered down from 8-bit.
    let sr = (r >> 3).wrapping_sub(r >> 4);
    let sg = (g >> 2).wrapping_sub(g >> 4);
    let sb = (b >> 3).wrapping_sub(b >> 4);
    let mut dr = (sr >> 4) & 1;
    let mut dg = (sg >> 4) & 3;
    let mut db = (sb >> 4) & 1;
    if (sr & 0xf) as u32 > bayer { dr = (dr + 1).min(1); }
    if (sg & 0xf) as u32 > bayer { dg = (dg + 1).min(3); }
    if (sb & 0xf) as u32 > bayer { db = (db + 1).min(1); }
    ((db << 3) | (dg << 1) | dr) as u32
}

pub(crate) fn rgb24_to_rgb8_dither(val: u32) -> u32 {
    let bayer = bayer_threshold(val >> 24);
    let r = (val & 0xFF) as u8;
    let g = ((val >> 8) & 0xFF) as u8;
    let b = ((val >> 16) & 0xFF) as u8;
    // 8bpp: 3-3-2 BGR.
    let sr = (r >> 1).wrapping_sub(r >> 4);
    let sg = (g >> 1).wrapping_sub(g >> 4);
    let sb = (b >> 2).wrapping_sub(b >> 4);
    let mut dr = (sr >> 4) & 7;
    let mut dg = (sg >> 4) & 7;
    let mut db = (sb >> 4) & 3;
    if (sr & 0xf) as u32 > bayer { dr = (dr + 1).min(7); }
    if (sg & 0xf) as u32 > bayer { dg = (dg + 1).min(7); }
    if (sb & 0xf) as u32 > bayer { db = (db + 1).min(3); }
    ((db << 6) | (dg << 3) | dr) as u32
}

pub(crate) fn rgb24_to_rgb12_dither(val: u32) -> u32 {
    let bayer = bayer_threshold(val >> 24);
    let r = (val & 0xFF) as u32;
    let g = ((val >> 8) & 0xFF) as u32;
    let b = ((val >> 16) & 0xFF) as u32;
    // 12bpp: 4-4-4 BGR.
    let sr = r - (r >> 4);
    let sg = g - (g >> 4);
    let sb = b - (b >> 4);
    let mut dr = (sr >> 4) & 15;
    let mut dg = (sg >> 4) & 15;
    let mut db = (sb >> 4) & 15;
    if (sr & 0xf) > bayer { dr = (dr + 1).min(15); }
    if (sg & 0xf) > bayer { dg = (dg + 1).min(15); }
    if (sb & 0xf) > bayer { db = (db + 1).min(15); }
    (db << 8) | (dg << 4) | dr
}


// ── Colour helpers ───────────────────────────────────────────────────────────

/// Compress 24-bit BGR to plane depth, optionally dithering.
///
/// Delegates to the existing `rgb24_to_rgb{4,8,12}[_dither]` bodies rather than
/// restating them: the packings are irregular (1-2-1 at 4bpp, 3-3-2 at 8bpp,
/// 4-4-4 at 12bpp) and the dither variants carry a specific error-diffusion
/// form. In CI mode the source is already plane-depth, so this is the identity.
#[inline(always)]
fn compress<M: Mode>(m: &M, val: u32, x: i32, y: i32) -> u32 {
    if m.rgbmode() == 0 {
        return val;
    }
    if m.dither() != 0 {
        // The `_dither` bodies expect the bayer cell index in bits 27:24.
        let packed = bayer_pack(val, x, y);
        match m.drawdepth() {
            DRAWMODE1_DRAWDEPTH_4 => rgb24_to_rgb4_dither(packed),
            DRAWMODE1_DRAWDEPTH_8 => rgb24_to_rgb8_dither(packed),
            DRAWMODE1_DRAWDEPTH_12 => rgb24_to_rgb12_dither(packed),
            // 24bpp: nothing to quantise, so nothing to dither.
            _ => val,
        }
    } else {
        match m.drawdepth() {
            DRAWMODE1_DRAWDEPTH_4 => rgb24_to_rgb4(val),
            DRAWMODE1_DRAWDEPTH_8 => rgb24_to_rgb8(val),
            DRAWMODE1_DRAWDEPTH_12 => rgb24_to_rgb12(val),
            _ => val,
        }
    }
}

/// Expand a plane-depth pixel to 24-bit BGR (blend destination, scr2scr source).
#[inline(always)]
fn expand<M: Mode>(m: &M, val: u32) -> u32 {
    if m.rgbmode() == 0 {
        return val;
    }
    match m.drawdepth() {
        DRAWMODE1_DRAWDEPTH_4 => rgb4_to_rgb24(val),
        DRAWMODE1_DRAWDEPTH_8 => rgb8_to_rgb24(val),
        DRAWMODE1_DRAWDEPTH_12 => rgb12_to_rgb24(val),
        _ => val,
    }
}

/// Replicate a plane-depth value into the slots WRMASK may select (dblsrc
/// packing).
#[inline(always)]
fn amplify<M: Mode>(m: &M, val: u32) -> u32 {
    match m.planes() {
        DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => match m.drawdepth() {
            DRAWMODE1_DRAWDEPTH_4 => val | (val << 4),
            DRAWMODE1_DRAWDEPTH_8 => val | (val << 8),
            DRAWMODE1_DRAWDEPTH_12 => val | (val << 12),
            // 24bpp fills the word; nothing to replicate.
            _ => val,
        },
        DRAWMODE1_PLANES_OLAY => (val << 8) | (val << 16),
        DRAWMODE1_PLANES_CID => val | (val << 4),
        DRAWMODE1_PLANES_PUP => (val << 2) | (val << 6),
        _ => 0,
    }
}

/// Apply the DRAWMODE1 LOGICOP selector.
#[inline(always)]
fn logic_op<M: Mode>(m: &M, src: u32, dst: u32) -> u32 {
    match m.logicop() {
        DRAWMODE1_LOGICOP_ZERO => 0,
        DRAWMODE1_LOGICOP_AND => src & dst,
        DRAWMODE1_LOGICOP_ANDR => src & !dst,
        DRAWMODE1_LOGICOP_SRC => src,
        DRAWMODE1_LOGICOP_ANDI => !src & dst,
        DRAWMODE1_LOGICOP_DST => dst,
        DRAWMODE1_LOGICOP_XOR => src ^ dst,
        DRAWMODE1_LOGICOP_OR => src | dst,
        DRAWMODE1_LOGICOP_NOR => !(src | dst),
        DRAWMODE1_LOGICOP_XNOR => !(src ^ dst),
        DRAWMODE1_LOGICOP_NDST => !dst,
        DRAWMODE1_LOGICOP_ORR => src | !dst,
        DRAWMODE1_LOGICOP_NSRC => !src,
        DRAWMODE1_LOGICOP_ORI => !src | dst,
        DRAWMODE1_LOGICOP_NAND => !(src & dst),
        // ONE (0xF) and any unreachable encoding.
        _ => !0,
    }
}

/// Alpha-vs-ALPHAREF compare; 0x7 means disabled.
#[inline(always)]
fn afunc<M: Mode>(m: &M, sa: u32, aref: u32) -> bool {
    match m.compare() {
        DRAWMODE1_COMPARE_NEVER => false,
        DRAWMODE1_COMPARE_LT => sa < aref,
        DRAWMODE1_COMPARE_EQ => sa == aref,
        DRAWMODE1_COMPARE_LE => sa <= aref,
        DRAWMODE1_COMPARE_GT => sa > aref,
        DRAWMODE1_COMPARE_NE => sa != aref,
        DRAWMODE1_COMPARE_GE => sa >= aref,
        // DISABLE (0x7): every relation set, so everything passes.
        _ => true,
    }
}

/// The `(shift, mask)` this plane selection reads with, or `None` for an
/// unmapped plane (the old `read_zero` / `write_nop` case).
#[inline(always)]
fn plane_shift_mask<M: Mode>(m: &M) -> Option<(u32, u32)> {
    let dbl = m.dblsrc() != 0;
    match m.planes() {
        DRAWMODE1_PLANES_RGB | DRAWMODE1_PLANES_RGBA => match m.drawdepth() {
            DRAWMODE1_DRAWDEPTH_4 => Some((if dbl { 4 } else { 0 }, 0xF)),
            DRAWMODE1_DRAWDEPTH_8 => Some((if dbl { 8 } else { 0 }, 0xFF)),
            DRAWMODE1_DRAWDEPTH_12 => Some((if dbl { 12 } else { 0 }, 0xFFF)),
            DRAWMODE1_DRAWDEPTH_24 => Some((0, 0xFFFFFF)),
            _ => None,
        },
        DRAWMODE1_PLANES_OLAY => Some((if dbl { 16 } else { 8 }, 0xFF)),
        DRAWMODE1_PLANES_PUP => Some((if dbl { 6 } else { 2 }, 0x3)),
        DRAWMODE1_PLANES_CID => Some((if dbl { 4 } else { 0 }, 0x3)),
        _ => None,
    }
}

/// True when this plane lives in the aux framebuffer rather than RGB.
#[inline(always)]
fn plane_is_aux<M: Mode>(m: &M) -> bool {
    matches!(
        m.planes(),
        DRAWMODE1_PLANES_OLAY | DRAWMODE1_PLANES_PUP | DRAWMODE1_PLANES_CID
    )
}

/// Read one plane-depth pixel.
#[inline(always)]
fn read_plane<M: Mode>(fb: &Framebuffers, m: &M, addr: u32) -> u32 {
    let Some((shift, mask)) = plane_shift_mask(m) else {
        return 0;
    };
    let plane = if plane_is_aux(m) { fb.aux } else { fb.rgb };
    let raw = unsafe { *plane.add(addr as usize) };
    (raw >> shift) & mask
}

/// Masked write to the plane this mode selects.
#[inline(always)]
fn write_plane<M: Mode>(fb: &Framebuffers, m: &M, ctx: &Rex3Context, addr: u32, val: u32) {
    if plane_shift_mask(m).is_none() {
        return; // unmapped plane: write_nop
    }
    let plane = if plane_is_aux(m) { fb.aux } else { fb.rgb };
    let slot = unsafe { &mut *plane.add(addr as usize) };
    let mask = ctx.wrmask;
    *slot = (*slot & !mask) | (val & mask);
}

/// CIDMATCH is a mask of permitted CIDs: the 2-bit CID from the aux plane
/// indexes a bit of it. 0xF (all permitted) is handled by `cidtest() == 0`, so
/// this is only reached when the test is live.
#[inline(always)]
fn cid_allows_write(fb: &Framebuffers, cid_mask: u32, addr: u32) -> bool {
    let cid = unsafe { *fb.aux.add(addr as usize) } & 3;
    cid_mask & (1 << cid) != 0
}


// ── Addressing and blending ──────────────────────────────────────────────────
//
// Moved off `Rex3`: none of these touched `self`, they only ever read the
// context. As free functions they sit next to the pipeline that uses them, and
// the draw path no longer needs a `&Rex3` to reach them — which is what lets its
// signature converge on the JIT shader's (ctx, fb_rgb, fb_aux).

pub fn calculate_fb_address<M: Mode, const IS_WRITE: bool>(
    m: &M,
    x: i32,
    y: i32,
    ctx: &Rex3Context,
) -> Option<u32> {
    // Mode fields come from `m`, so a const instance folds these away: a draw
    // with no scissor enabled drops both clip blocks entirely, and one without
    // YFLIP loses that branch.
    let is_scr2scr = m.opcode() == DRAWMODE0_OPCODE_SCR2SCR;
    
    let mut x_curr = x;
    let mut y_curr = y;

    // In scr2scr mode xymove is unconditional (it offsets the destination).
    // In regular paint mode it is conditional on the xyoffset flag.
    let apply_xymove = is_scr2scr || m.xyoffset() != 0;
    if apply_xymove {
        let x_move = (ctx.xymove >> 16) as i16 as i32;
        let y_move = (ctx.xymove & 0xFFFF) as i16 as i32;
        x_curr += x_move;
        y_curr += y_move;
    }

    // Apply XYWIN offset
    // XYWIN is 16,16. Assuming High=X, Low=Y.
    // Treated as signed 16-bit integers for coordinate biasing.
    let x_off = ((ctx.xywin >> 16) & 0xFFFF) as i16 as i32;
    let y_off = (ctx.xywin & 0xFFFF) as i16 as i32;

    let x_abs = x_curr + x_off;
    let y_abs = y_curr + y_off;

    if IS_WRITE {
        let ensmask = m.ensmask();

        // SMASK0 (Window Relative) — high16=min, low16=max
        if (ensmask & CLIPMODE_ENSMASK_SMASK0) != 0 {
            let min_x = ((ctx.smask0x >> 16) & 0xFFFF) as i16 as i32;
            let max_x = (ctx.smask0x & 0xFFFF) as i16 as i32;
            let min_y = ((ctx.smask0y >> 16) & 0xFFFF) as i16 as i32;
            let max_y = (ctx.smask0y & 0xFFFF) as i16 as i32;

            if x_curr < min_x || x_curr > max_x || y_curr < min_y || y_curr > max_y {
                return None;
            }
        }

        // SMASK1-4 (Screen Absolute, unaffected by XYWIN per spec).
        // Host pre-biases smask values with the 4K,4K offset. The equivalent
        // pixel coordinate is x_phys + 0x1000 = (x_abs - 0x1000) + 0x1000 = x_abs.
        // So compare x_abs/y_abs directly against raw smask values. high16=min, low16=max.
        // Logic: pixel must be inside at least one enabled mask.
        let smask_enabled = (ensmask & CLIPMODE_ENSMASK_SMASK1_4) != 0;
        if smask_enabled {
            let smasks = [
                (ctx.smask1x, ctx.smask1y),
                (ctx.smask2x, ctx.smask2y),
                (ctx.smask3x, ctx.smask3y),
                (ctx.smask4x, ctx.smask4y),
            ];
            let mut inside_any = false;
            for (bit, (sx, sy)) in smasks.iter().enumerate() {
                if (ensmask & (1 << (bit + 1))) == 0 { continue; }
                let min_x = (sx >> 16) as i16 as i32;
                let max_x = (sx & 0xFFFF) as i16 as i32;
                let min_y = (sy >> 16) as i16 as i32;
                let max_y = (sy & 0xFFFF) as i16 as i32;
                if x_abs >= min_x && x_abs <= max_x && y_abs >= min_y && y_abs <= max_y {
                    inside_any = true;
                    break;
                }
            }
            if !inside_any {
                return None;
            }
        }
    }
    // Physical Address Calculation
    let x_phys = x_abs - REX3_COORD_BIAS;
    let y_phys = if m.yflip() != 0 { 0x23FF - y_abs } else { y_abs - REX3_COORD_BIAS };

    // Sector Clipping (VRAM bounds)
    // Reads are not culled (but must be within VRAM allocation)
    let width_limit = if IS_WRITE { REX3_SCREEN_WIDTH } else { 2048 };
    if x_phys < 0 || x_phys >= width_limit || y_phys < 0 || y_phys >= REX3_SCREEN_HEIGHT {
        return None;
    }

    Some((y_phys as u32) * 2048 + (x_phys as u32))
}


pub fn calculate_src_address<M: Mode>(m: &M, x: i32, y: i32, ctx: &Rex3Context) -> Option<u32> {
    // xymove affects the destination, not the source.
    // Source is just (x, y) + xywin.
    let x_win = ((ctx.xywin >> 16) & 0xFFFF) as i16 as i32;
    let y_win = (ctx.xywin & 0xFFFF) as i16 as i32;

    let x_abs = x + x_win;
    let y_abs = y + y_win;

    let x_phys = x_abs - REX3_COORD_BIAS;
    let y_phys = if m.yflip() != 0 { 0x23FF - y_abs } else { y_abs - REX3_COORD_BIAS };

    if x_phys < 0 || x_phys >= REX3_SCREEN_WIDTH || y_phys < 0 || y_phys >= REX3_SCREEN_HEIGHT {
        return None;
    }

    Some((y_phys as u32) * 2048 + (x_phys as u32))
}


pub fn blend<M: Mode>(m: &M, src: u32, dst: u32) -> u32 {
    let s_factor_sel = m.sfactor();
    let d_factor_sel = m.dfactor();

    // BLENDALPHA (DRAWMODE1 bit 27) substitutes the SOURCE multiplier only.
    // Spec §3.8: "When source multiplier is set to source alpha (SFACTOR=4) ...
    // When BLENDALPHA is set to 0, the source multiplier for blending alpha is
    // one instead of source alpha AND DESTINATION MULTIPLIER IS DEFINED BY
    // DFACTOR." The trailing clause is load-bearing: DFACTOR keeps its own
    // definition, so a DFACTOR of BF_MSA still evaluates 1 - source alpha
    // against the real alpha. Substituting in both factors would zero BF_MSA
    // and discard the destination entirely, which the spec does not say.
    let sa_real = (src >> 24) & 0xFF;
    let sa_src = if m.blendalpha() != 0 { sa_real } else { 255 };

    // `c` is the *other* operand's channel — destination when computing the
    // source factor, source when computing the destination factor — which is
    // why BF_OC/BF_MOC read as BF_DC/BF_MDC for SFACTOR and BF_SC/BF_MSC for
    // DFACTOR (spec Tables 13/14).
    let get_factor = |sel: u32, c: u32, a: u32| -> u32 {
        match sel {
            DRAWMODE1_BF_ZERO => 0,
            DRAWMODE1_BF_ONE => 255,
            DRAWMODE1_BF_OC => c,
            DRAWMODE1_BF_MOC => 255 - c,
            DRAWMODE1_BF_SA => a,
            DRAWMODE1_BF_MSA => 255 - a,
            // 110/111 are undefined by the hardware.
            _ => 0,
        }
    };

    let mut res = 0;
    for i in 0..4 {
        let shift = i * 8;
        let s_c = (src >> shift) & 0xFF;
        let d_c = (dst >> shift) & 0xFF;
        
        let sf = get_factor(s_factor_sel, d_c, sa_src);
        let df = get_factor(d_factor_sel, s_c, sa_real);
        
        let val = (s_c * sf + d_c * df) / 255;
        let val_clamped = if val > 255 { 255 } else { val };
        
        res |= val_clamped << shift;
    }
    res
}


// ── Per-pixel iteration ──────────────────────────────────────────────────────

/// Advance the shade DDAs.
///
/// RGB mode clamps every iteration per spec §3.8; CI mode clamps only under
/// CICLAMP, and only at 8/12bpp. The DDA advances whether or not the pixel was
/// drawn. Whether the clamp writes back into the accumulator (as here) or only
/// affects the value sent downstream is an open hardware question — see
/// `rules/rex3/shade-dda-clamp-open-question.md`.
#[inline(always)]
pub fn iterate_shade<M: Mode>(ctx: &mut Rex3Context, m: &M) {
    if m.shade() == 0 {
        return;
    }
    shade_add(ctx);
    if m.rgbmode() != 0 {
        ctx.colorred = clamp_shade(ctx.colorred);
        ctx.colorgrn = clamp_shade(ctx.colorgrn);
        ctx.colorblue = clamp_shade(ctx.colorblue);
        ctx.coloralpha = clamp_shade(ctx.coloralpha);
    } else if m.ciclamp() != 0 {
        match m.drawdepth() {
            DRAWMODE1_DRAWDEPTH_8 => {
                if ctx.colorred & (1 << 19) != 0 {
                    ctx.colorred = 0x0007_FFFF;
                }
            }
            DRAWMODE1_DRAWDEPTH_12 => {
                if ctx.colorred & (1 << 21) != 0 {
                    ctx.colorred = 0x001F_FFFF;
                }
            }
            // 4bpp / 24bpp: no CI clamp per spec.
            _ => {}
        }
    }
}

/// integer = bits[22:11] & 0x1FF; negative or >= 0x180 clamps to 0, > 0xFF to
/// the saturation constant.
#[inline(always)]
fn clamp_shade(c: u32) -> u32 {
    let val = (c >> 11) & 0x1FF;
    if c & (1 << 31) != 0 || val >= 0x180 {
        0
    } else if val > 0xFF {
        0x0007_FFFF
    } else {
        c
    }
}

/// Advance the pattern bit counters.
#[inline(always)]
pub fn iterate_pattern<M: Mode>(ctx: &mut Rex3Context, m: &M) {
    if m.enzpattern() != 0 {
        advance_zpat(ctx);
    }
    if m.enlspattern() != 0 {
        advance_lspat(ctx);
    }
}

// ── Pixel bodies ─────────────────────────────────────────────────────────────

/// One pixel of a DRAW primitive.
///
/// The pattern prologue follows the old `process_pixel_noblend` and the rex-jit
/// shader: the lspattern test is evaluated unconditionally, with no suppression
/// on the zpopaque path. `process_pixel_draw` suppressed it; two of the three
/// implementations did not, and the JIT is one of them, so that is what the
/// comparison tests enforce. Whether hardware suppresses is open.
#[inline(always)]
fn pixel_draw<M: Mode>(
    fb: &Framebuffers,
    ctx: &mut Rex3Context,
    m: &M,
    fc_color: u32,
    x: i32,
    y: i32,
) {
    let mut use_bg = false;

    if m.enzpattern() != 0 && (ctx.zpattern >> ctx.zpat_bit) & 1 == 0 {
        if m.zpopaque() != 0 {
            use_bg = true;
        } else {
            return;
        }
    }

    if m.enlspattern() != 0 && (ctx.lspattern >> ctx.pat_bit) & 1 == 0 {
        if m.lsopaque() != 0 {
            use_bg = true;
        } else {
            return;
        }
    }

    // Host pixel is consumed only when the pattern passed and the source is the
    // host FIFO — matching MAME's get_host_color() placement.
    let host_pixel = if !use_bg && (m.alphahost() != 0 || m.colorhost() != 0) {
        fetch_host_pixel(ctx, m)
    } else {
        0
    };

    let Some(addr) = calculate_fb_address::<M, true>(m, x, y, ctx) else {
        return;
    };

    if m.cidtest() != 0 && !cid_allows_write(fb, m.cid_mask(), addr) {
        return;
    }

    // FASTCLEAR: flat fill from COLORVRAM, no per-pixel operation at all
    // (rex3.pdf §3.5.5). Only set where hardware honours it — DRAW opcode, CID
    // checking off, not overridden by colorhost.
    if m.fastclear() != 0 {
        // Hoisted by the caller: depends only on the mode and COLORVRAM, both
        // fixed for the primitive, so recomputing it per pixel is pure waste.
        write_plane(fb, m, ctx, addr, fc_color);
        return;
    }

    let raw_src = if use_bg {
        ctx.colorback
    } else {
        combine_host_dda(ctx, m, host_pixel)
    };

    if m.compare() != 0x7 && !afunc(m, (raw_src >> 24) & 0xFF, ctx.alpharef & 0xFF) {
        return;
    }

    let res = if m.blend() != 0 {
        let dst_raw = if m.backblend() != 0 {
            ctx.colorback
        } else {
            expand(m, read_plane(fb, m, addr))
        };
        let blended = blend(m, raw_src, dst_raw);
        amplify(m, compress(m, blended, x, y))
    } else {
        let src = amplify(m, compress(m, raw_src, x, y));
        let dst = amplify(m, read_plane(fb, m, addr));
        logic_op(m, src, dst)
    };

    write_plane(fb, m, ctx, addr, res);
}

/// One pixel of a READ (HOSTR) primitive: framebuffer to host FIFO. No write, so
/// no CID check and no write-side clipping.
#[inline(always)]
fn pixel_read<M: Mode>(fb: &Framebuffers, ctx: &mut Rex3Context, m: &M, x: i32, y: i32) {
    let pixel = match calculate_fb_address::<M, false>(m, x, y, ctx) {
        // Expand to 24-bit BGR before packing: host_pack_*_rgb expects that,
        // and expand is the identity in CI mode.
        Some(addr) => expand(m, read_plane(fb, m, addr)),
        None => 0,
    };
    store_host_pixel(ctx, m, pixel);
}

/// One pixel of a SCR2SCR primitive.
///
/// Two deliberate differences from the DRAW path, both matching the original
/// `process_pixel_scr2scr`: the blend arm does not amplify after compressing,
/// and there is no pattern or afunction handling.
#[inline(always)]
fn pixel_scr2scr<M: Mode>(fb: &Framebuffers, ctx: &mut Rex3Context, m: &M, x: i32, y: i32) {
    let raw_src = match calculate_src_address(m, x, y, ctx) {
        Some(src_addr) => expand(m, read_plane(fb, m, src_addr)),
        None => 0,
    };

    let Some(dst_addr) = calculate_fb_address::<M, true>(m, x, y, ctx) else {
        return;
    };
    if m.cidtest() != 0 && !cid_allows_write(fb, m.cid_mask(), dst_addr) {
        return;
    }

    let res = if m.blend() != 0 {
        let dst_raw = if m.backblend() != 0 {
            ctx.colorback
        } else {
            expand(m, read_plane(fb, m, dst_addr))
        };
        // No amplify here — matches process_pixel_scr2scr.
        compress(m, blend(m, raw_src, dst_raw), x, y)
    } else {
        let src = amplify(m, compress(m, raw_src, x, y));
        let dst = amplify(m, read_plane(fb, m, dst_addr));
        logic_op(m, src, dst)
    };

    write_plane(fb, m, ctx, dst_addr, res);
}

/// Dispatch one pixel to the body this opcode selects.
#[inline(always)]
pub fn pixel<M: Mode>(
    fb: &Framebuffers,
    ctx: &mut Rex3Context,
    m: &M,
    fc_color: u32,
    x: i32,
    y: i32,
) {
    match m.opcode() {
        DRAWMODE0_OPCODE_READ => pixel_read(fb, ctx, m, x, y),
        DRAWMODE0_OPCODE_SCR2SCR => pixel_scr2scr(fb, ctx, m, x, y),
        DRAWMODE0_OPCODE_NOOP => {}
        _ => pixel_draw(fb, ctx, m, fc_color, x, y),
    }
}

/// Set/clear a `rexdiag` activity bit. No-op without the feature.
#[inline(always)]
pub fn diag_set(ctx: &Rex3Context, bits: u64, on: bool) {
    let _ = (ctx, bits, on);
    #[cfg(feature = "rexdiag")]
    {
        let host = ctx.host;
        if !host.is_null() {
            use std::sync::atomic::Ordering;
            let d = unsafe { &(*host).diag };
            if on {
                d.fetch_or(bits, Ordering::Relaxed);
            } else {
                d.fetch_and(!bits, Ordering::Relaxed);
            }
        }
    }
}

/// Developer block/span trace. Compiles to nothing without the feature.
#[inline(always)]
fn log_block(ctx: &Rex3Context, opcode: u32) {
    let host = ctx.host;
    if host.is_null() {
        return;
    }
    unsafe { (*host).log_block(ctx, opcode) }
}


// ── Host FIFO ────────────────────────────────────────────────────────────────
//
// HOSTRW transfers, moved off `Rex3`: the pack/unpack variant is selected from
// the mode rather than through the `host_pack`/`host_unpack` function pointers
// these replaced, and the slot geometry comes from `Mode::hostdepth`/`rwpacked`/
// `rwdouble` instead of the `host_count`/`host_shift` cells.

/// Pixels per 64-bit host word.
#[inline(always)]
fn host_count<M: Mode>(m: &M) -> u32 {
    if m.rwpacked() == 0 {
        return 1;
    }
    if m.rwdouble() != 0 {
        match m.hostdepth() {
            DRAWMODE1_HOSTDEPTH_12 | DRAWMODE1_HOSTDEPTH_8 => 8,
            DRAWMODE1_HOSTDEPTH_4 => 4,
            _ => 2,
        }
    } else {
        match m.hostdepth() {
            DRAWMODE1_HOSTDEPTH_12 | DRAWMODE1_HOSTDEPTH_8 => 4,
            DRAWMODE1_HOSTDEPTH_4 => 2,
            _ => 1,
        }
    }
}

/// Bits per host pixel slot. 4bpp pixels sit in an 8-bit slot, so the shift is
/// 8 rather than 4.
#[inline(always)]
fn host_shift<M: Mode>(m: &M) -> u32 {
    if m.rwpacked() == 0 {
        return 0;
    }
    match m.hostdepth() {
        DRAWMODE1_HOSTDEPTH_12 | DRAWMODE1_HOSTDEPTH_8 => 8,
        DRAWMODE1_HOSTDEPTH_4 => 16,
        _ => 32,
    }
}

/// Extract the leading pixel from the host shifter, expanding to 24-bit BGR in
/// RGB mode and leaving a plane-depth index in CI mode.
#[inline(always)]
fn host_unpack<M: Mode>(m: &M, val: u64) -> u32 {
    let rgb = m.rgbmode() != 0;
    match m.hostdepth() {
        DRAWMODE1_HOSTDEPTH_12 => {
            let v = ((val >> 56) & 0xF) as u32;
            if rgb { Rex3::expand_4_rgb(v) } else { v }
        }
        DRAWMODE1_HOSTDEPTH_8 => {
            let v = ((val >> 56) & 0xFF) as u32;
            if rgb { Rex3::expand_8_rgb(v) } else { v }
        }
        DRAWMODE1_HOSTDEPTH_4 => {
            let v = ((val >> 48) & 0xFFF) as u32;
            if rgb { Rex3::expand_12_rgb(v) } else { v }
        }
        _ => {
            let v = (val >> 32) as u32;
            if rgb { Rex3::expand_32_rgb(v) } else { v }
        }
    }
}

/// Inverse of [`host_unpack`]: shift the accumulator up by one slot and insert.
#[inline(always)]
fn host_pack<M: Mode>(m: &M, acc: u64, pixel: u32) -> u64 {
    let rgb = m.rgbmode() != 0;
    match m.hostdepth() {
        DRAWMODE1_HOSTDEPTH_12 => {
            let v = if rgb { Rex3::compress_4_rgb(pixel) } else { pixel & 0xF };
            (acc << 8) | v as u64
        }
        DRAWMODE1_HOSTDEPTH_8 => {
            let v = if rgb { Rex3::compress_8_rgb(pixel) } else { pixel & 0xFF };
            (acc << 8) | v as u64
        }
        DRAWMODE1_HOSTDEPTH_4 => {
            let v = if rgb { Rex3::compress_12_rgb(pixel) } else { pixel & 0xFFF };
            (acc << 16) | v as u64
        }
        _ => {
            let v = if rgb { Rex3::compress_32_rgb(pixel) } else { pixel };
            (acc << 32) | v as u64
        }
    }
}

/// Has the current transfer been fully consumed?
///
/// Host mode stops after one *word* per GO on real hardware, because the CPU
/// feeds HOSTRW one word at a time and each write carries its own GO. With the
/// port as an array the rule generalises to "one *transfer* per GO": the
/// walker keeps going while `host_cursor` has not reached `host_len`. A PIO
/// write is `host_len == 1`, which reproduces one-word-per-GO exactly.
#[inline(always)]
fn host_batch_drained(ctx: &Rex3Context) -> bool {
    ctx.hostrw_drained()
}

/// Pull one pixel from the HOSTRW FIFO, refilling the shifter when empty.
#[inline(always)]
pub fn fetch_host_pixel<M: Mode>(ctx: &mut Rex3Context, m: &M) -> u32 {
    if ctx.hostcnt == 0 {
        // Load the word at the cursor, then step past it. For a single-word
        // PIO access (host_len == 1) the step is a no-op and the cursor stays
        // on element 0, exactly as the scalar port behaved; for a longer
        // transfer it walks the array.
        // 32-bit writes land in the high half [63:32] via HOSTRW0, so the data
        // is already at the MSB.
        ctx.host_shifter = ctx.hostrw_get();
        ctx.hostrw_advance();
        if m.swapendian() != 0 {
            ctx.host_shifter = if m.rwdouble() != 0 {
                ctx.host_shifter.swap_bytes()
            } else {
                // Swap only the high 32-bit word.
                let hi = (ctx.host_shifter >> 32) as u32;
                ((hi.swap_bytes() as u64) << 32) | (ctx.host_shifter & 0xFFFF_FFFF)
            };
        }
        ctx.hostcnt = host_count(m);
    }

    let pixel = host_unpack(m, ctx.host_shifter);
    ctx.host_shifter <<= host_shift(m);
    ctx.hostcnt -= 1;
    pixel
}

/// Publish the assembled shifter back to HOSTRW for the CPU to read.
///
/// Writes into `hostrw[host_cursor]` and steps, so a multi-word read fills
/// the array and is available after one pipeline drain instead of one per
/// qword. For `host_len == 1` the cursor stays on element 0.
#[inline(always)]
fn send_host_word<M: Mode>(ctx: &mut Rex3Context, m: &M) {
    let mut val = ctx.host_shifter;
    if m.swapendian() != 0 {
        val = val.swap_bytes();
    } else if m.rwdouble() == 0 {
        val <<= 32;
    }
    ctx.hostrw_set(val);
    // Step so the next word lands in the next slot. For host_len == 1 the
    // cursor stays put and the CPU reads element 0, as before.
    ctx.hostrw_advance();
}

/// Append one pixel to the host word, publishing when it fills.
#[inline(always)]
pub fn store_host_pixel<M: Mode>(ctx: &mut Rex3Context, m: &M, pixel: u32) {
    if ctx.hostcnt == 0 {
        ctx.hostcnt = host_count(m);
        ctx.host_shifter = 0;
    }
    ctx.host_shifter = host_pack(m, ctx.host_shifter, pixel);
    ctx.hostcnt -= 1;
    if ctx.hostcnt == 0 {
        send_host_word(ctx, m);
    }
}

/// Publish a partially-filled word at the end of a primitive.
#[inline(always)]
pub fn flush_host_pixel<M: Mode>(ctx: &mut Rex3Context, m: &M) {
    if ctx.hostcnt > 0 {
        // Align the remaining bits to the MSB, since packing runs LSB to MSB.
        ctx.host_shifter <<= ctx.hostcnt * host_shift(m);
        send_host_word(ctx, m);
        ctx.hostcnt = 0;
    }
}

/// Combine the host pixel with the DDA colour and alpha.
#[inline(always)]
fn combine_host_dda<M: Mode>(ctx: &Rex3Context, m: &M, host_pixel: u32) -> u32 {
    let color = if m.colorhost() != 0 { host_pixel } else { ctx.get_colori() };
    // Afunction's source alpha comes from "either DDA or host" (selected by
    // ALPHAHOST) independent of plane format (spec §3.8.1) — a CI-mode app can
    // stream a colour index alongside a host alpha byte. Always overlay bits
    // 31:24 so afunction sees the right value in both modes; the CI write paths
    // mask down to the plane-depth index and never look at these bits.
    let a = if m.alphahost() != 0 {
        (host_pixel >> 24) & 0xFF
    } else {
        Rex3Context::clamp_color_component(ctx.coloralpha)
    };
    (color & 0x00FF_FFFF) | (a << 24)
}

/// Replicate COLORVRAM into every plane slot for a fastclear fill.
///
/// Depends only on the mode and COLORVRAM, so the draw loops hoist it out of
/// the pixel loop rather than recomputing it per pixel.
#[inline(always)]
pub fn fastclear_color<M: Mode>(ctx: &Rex3Context, m: &M) -> u32 {
    let v = ctx.colorvram;
    match m.drawdepth() {
        DRAWMODE1_DRAWDEPTH_4 => {
            let c = v & 0xf;
            c | (c << 4) | (c << 8) | (c << 16)
        }
        DRAWMODE1_DRAWDEPTH_8 => {
            let c = v & 0xff;
            c | (c << 8) | (c << 16)
        }
        DRAWMODE1_DRAWDEPTH_12 => {
            // 12bpp: RGB mode takes nibbles from colorvram, CI the low 12 bits.
            let c = if m.rgbmode() != 0 {
                ((v & 0xf00000) >> 12) | ((v & 0xf000) >> 8) | ((v & 0xf0) >> 4)
            } else {
                v & 0x000fff
            };
            c | (c << 12)
        }
        _ => v & 0xffffff,
    }
}

// ── Draw ─────────────────────────────────────────────────────────────────────

/// The draw pipeline, written once.
///
/// Instantiated with [`ConstMode`] every `m.*()` call is a literal and the dead
/// arms vanish; instantiated with [`DynMode`] the same source reads fields at
/// runtime. That is the point: one expression, two ways of running it.
#[inline]
pub fn draw<M: Mode>(ctx: &mut Rex3Context, m: &M) {
    // Derive the framebuffers from the device, then run the shared core. A
    // generated shader skips this and calls draw_with_fb directly, because the
    // compiled-shader ABI hands it the two pointers already.
    let host = ctx.host;
    if host.is_null() {
        return;
    }
    let rex: &Rex3 = unsafe { &*host };
    let fb = Framebuffers {
        rgb: unsafe { (*rex.fb_rgb.get()).as_mut_ptr() },
        aux: unsafe { (*rex.fb_aux.get()).as_mut_ptr() },
    };
    draw_with_fb(ctx, &fb, m);
}

/// The draw core, taking the framebuffers the caller already holds.
#[inline]
pub fn draw_with_fb<M: Mode>(ctx: &mut Rex3Context, fb: &Framebuffers, m: &M) {
    if m.opcode() == DRAWMODE0_OPCODE_NOOP {
        return;
    }
    match m.adrmode() {
        DRAWMODE0_ADRMODE_I_LINE => draw_iline_g(fb, ctx, m),
        DRAWMODE0_ADRMODE_F_LINE => draw_fline_g(fb, ctx, m),
        DRAWMODE0_ADRMODE_A_LINE => draw_aline_g(fb, ctx, m),
        DRAWMODE0_ADRMODE_BLOCK => {
            log_block(ctx, m.opcode());
            draw_block_g(fb, ctx, m);
        }
        DRAWMODE0_ADRMODE_SPAN => {
            log_block(ctx, m.opcode());
            draw_span_g(fb, ctx, m);
        }
        _ => {}
    }
}

// ── Entry point ──────────────────────────────────────────────────────────────

/// Draw one primitive, reading the mode registers directly.
///
/// Builds a [`DynMode`] from DRAWMODE0/DRAWMODE1/CLIPMODE and runs the pipeline
/// with it. A generated table will sit in front of this, matching the extracted
/// values against known shapes and instantiating [`ConstMode`] for the hits —
/// the const path and this one share the same `draw`, so a table entry is a
/// specialisation, never a second implementation.
#[inline]// ── Primitive walkers ────────────────────────────────────────────────────────
// Moved here from `impl Rex3`: like the colour helpers these never touched
// `self`, and their only callers are in this module. They walk the primitive
// (block/span/line) and call into the pixel bodies above.

pub fn draw_block_g<M: Mode>(fb: &Framebuffers, ctx: &mut Rex3Context, m: &M) {
    // The rex3_simd pre-loop bailouts used to live here. They were removed:
    // despite the name they contained no vector code (objdump: zero xmm/ymm
    // instructions), and they intercepted the primitive *before* execute_go's
    // pixel-processor selection — so their own mode gates silently shadowed
    // the interpreter's. try_fastclear_block never checked CIDMATCH, which
    // made execute_go's `no_cid` term unreachable for BLOCK draws and let a
    // fastclear block ignore CID checking entirely, against the spec. See
    // rules/rex3/fastclear-cid-divergence.md.

    let _w = (ctx.xend - ctx.xstart).abs();
    let _h = (ctx.yend - ctx.ystart).abs();

    // Draw-loop control flow reads the decoded shape, not the raw registers:
    // one decode per primitive, and the same values the pixel path uses.
    let stopony = m.stopony() != 0;
    let length32 = m.length32() != 0;
    let ystride = m.ystride() != 0;
    let opcode = m.opcode();
    let colorhost = m.colorhost() != 0;
    let mut first = true;
    let skipfirst = m.skipfirst() != 0;
    let skiplast = m.skiplast() != 0;


    // In host mode (READ or DRAW+colorhost), each GO processes exactly one word's worth
    // of pixels (host_count pixels). stop_on_word causes the loop to exit after the
    // word boundary so the next GO picks up where we left off.
    let stop_on_word = opcode == DRAWMODE0_OPCODE_READ || colorhost;

    // stop_on_word takes priority over stoponx — host mode governs its own stop.
    let stoponx = m.stoponx() != 0 || stop_on_word;

    let octant = ctx.bresoctinc1.octant();
    let lronly = m.lronly() != 0;
    let x_dec = (octant & OCTANT_XDEC) != 0;
    let y_dec = (octant & OCTANT_YDEC) != 0;
    let lrskip = lronly && x_dec;
    // it is important to note that lrskip still performs y advance operations otherwise some triangles grow weird tails
    // Coordinate steps in 21.11 fixed-point: ±1 integer = ±2048
    let stepx: i32 = if x_dec { -(1 << 11) } else { 1 << 11 };
    let y_inc: i32 = if ystride { 2 } else { 1 };
    let stepy: i32 = if y_dec { -(y_inc << 11) } else { y_inc << 11 };

    // length32 only clamps if span is >= 32 pixels wide
    let span_len = ((ctx.xend - ctx.xstart).abs()) >> 11;
    let xstop = if length32 && span_len >= 32 { Some(ctx.xstart + stepx * 32) } else { None };



    // Loop-invariant: fastclear writes COLORVRAM replicated across plane
    // slots, and neither input changes within a primitive.
    let fc_color = fastclear_color(ctx, m);

    ctx.mid_primitive = true;
    #[cfg(feature = "rexdiag")]
    diag_set(ctx, Rex3::DIAG_LOOP_DRAW_BLOCK, true);
    loop {
        let x = ctx.xstart >> 11;
        let y = ctx.ystart >> 11;

        ctx.xstart += stepx;

        let x_end_reached = if x_dec { ctx.xstart < ctx.xend } else { ctx.xstart > ctx.xend };

        if !(first && skipfirst || x_end_reached && skiplast || lrskip) {
            pixel(fb, ctx, m, fc_color, x, y);
        }

        iterate_shade(ctx, m);
        iterate_pattern(ctx, m);

        if x_end_reached {
            // advance y, wrap x; reset pattern bits so each row starts at bit 31
            ctx.ystart += stepy;
            ctx.xstart = ctx.xsave;
            ctx.pat_bit  = 31;
            ctx.zpat_bit = 31;
            // lsrcount continues across rows — iterate_pattern_ls manages it per-pixel.

            if !stopony {
                // Without STOPONY, hardware doesn't auto-advance rows — each row is
                // its own complete primitive and the next GO starts a fresh one.
                // Mirrors the JIT's emit_shader (!stopony branch in end_x_block),
                // which already clears mid_primitive here. Leaving this true (as
                // before) permanently wedged log_block()'s "primitive start" guard
                // after the first non-stopony block ever ran, hiding every
                // subsequent block/READ/DRAW header from block.log.
                ctx.mid_primitive = false;
                break;
            }

            let y_end_reached = if y_dec { ctx.ystart < ctx.yend } else { ctx.ystart > ctx.yend };

            if y_end_reached {
                // All rows consumed — primitive done.
                ctx.mid_primitive = false;
                break;
            }

            // Host mode: a row boundary is always a forced word boundary too,
            // even if the row's width doesn't divide evenly into host_count
            // (e.g. a 17px-wide CI8 row's last word only has 1 of 4 slots
            // filled). Without this, a still-open partial word from this row
            // would keep accumulating pixels from the next row instead of
            // being flushed — hardware sends one word per GO in host mode,
            // full or not. Checked here (not just via the hostcnt==0 check
            // below) because that check alone never fires for a word that
            // never reaches host_count pixels.
            // Unconditional: a partial word at a row boundary must be sent
            // even mid-transfer. Gating this on the transfer being drained
            // merges the partial word into the next row, which is exactly the
            // divergence the JIT/interpreter HOSTR stress test catches.
            if stop_on_word && ctx.hostcnt > 0 {
                break;
            }

            first = true; // pixel in next row will be first
        } else if let Some(limit) = xstop {
            let limit_reached = if x_dec { ctx.xstart <= limit } else { ctx.xstart >= limit };
            if limit_reached {
                break;
            }
        }

        // Host mode: stop after one word (after y-advance so row boundary is handled first).
        // Primitive continues on next GO — mid_primitive stays true.
        // A multi-word transfer carries N words behind one GO, so it keeps
        // going until the array is drained. For host_len == 1 that is after a
        // single word, which is exactly the old one-word-per-GO rule.
        if stop_on_word && ctx.hostcnt == 0 && host_batch_drained(ctx) {
            break;
        }

        // stoponx/stopony: next GO advances to next step — mid_primitive stays true.
        if !stoponx {
            break;
        }
    }


    #[cfg(feature = "rexdiag")]
    diag_set(ctx, Rex3::DIAG_LOOP_DRAW_BLOCK, false);

    if opcode == DRAWMODE0_OPCODE_READ {
        flush_host_pixel(ctx, m);
    }
}

pub fn draw_span_g<M: Mode>(fb: &Framebuffers, ctx: &mut Rex3Context, m: &M) {
    if  m.lronly() != 0 && (ctx.bresoctinc1.octant() & OCTANT_XDEC) != 0{
        return;
    }
    let length32 = m.length32() != 0;
    let ystride = m.ystride() != 0;
    let opcode = m.opcode();
    let colorhost = m.colorhost() != 0;
    let mut first = true;
    let skipfirst = m.skipfirst() != 0;
    let skiplast = m.skiplast() != 0;

    // In host mode (READ or DRAW+colorhost), each GO processes exactly one word's worth
    // of pixels (host_count pixels). stop_on_word causes the loop to exit after the
    // word boundary so the next GO picks up where we left off.
    let stop_on_word = opcode == DRAWMODE0_OPCODE_READ || colorhost;

    // stop_on_word takes priority over stoponx — host mode governs its own stop.
    let stoponx = m.stoponx() != 0 || stop_on_word;

    // Spans always advance left-to-right (+1 in 21.11 fixed-point).
    // length32 only clamps if span is >= 32 pixels wide.
    let span_len = (ctx.xend - ctx.xstart) >> 11;
    let xstop = if length32 && span_len >= 32 { Some(ctx.xstart + (32 << 11)) } else { None };



    // Loop-invariant: fastclear writes COLORVRAM replicated across plane
    // slots, and neither input changes within a primitive.
    let fc_color = fastclear_color(ctx, m);

    ctx.mid_primitive = true;
    #[cfg(feature = "rexdiag")]
    diag_set(ctx, Rex3::DIAG_LOOP_DRAW_BLOCK, true);
    let x_end_reached = loop {
        let x = ctx.xstart >> 11;
        let y = ctx.ystart >> 11;

        ctx.xstart += 1 << 11;

        let x_end_reached = ctx.xstart > ctx.xend;

        if !(first && skipfirst || x_end_reached && skiplast) {
            pixel(fb, ctx, m, fc_color, x, y);
        }

        iterate_shade(ctx, m);
        iterate_pattern(ctx, m);

        if x_end_reached {
            break true;
        } else if let Some(limit) = xstop {
            if ctx.xstart >= limit {
                break false;
            }
        }

        // Host mode: stop after one word.
        // Primitive continues on next GO — mid_primitive stays true.
        // See the block walker: a batch runs to completion under one GO.
        if stop_on_word && ctx.hostcnt == 0 && host_batch_drained(ctx) {
            break false;
        }

        // stoponx: next GO advances to next step — mid_primitive stays true.
        if !stoponx {
            break false;
        }

        first = false;
    };

    if x_end_reached {
        // Span fully consumed — advance to next row and reset state.
        //let y_inc: i32 = if ystride { 2 } else { 1 };
        //ctx.ystart += y_inc << 11;
        //ctx.xstart = ctx.xsave;
        ctx.pat_bit  = 31;
        ctx.zpat_bit = 31;
        ctx.mid_primitive = false;
    }


    #[cfg(feature = "rexdiag")]
    diag_set(ctx, Rex3::DIAG_LOOP_DRAW_BLOCK, false);

    if opcode == DRAWMODE0_OPCODE_READ {
        flush_host_pixel(ctx, m);
    }
}

/// Fractional d correction for F_LINE/A_LINE from 21.11 endpoint sub-pixel position.
/// Fractional nibble is bits [10:7] of the 21.11 coordinate (16.4(7) layout).
pub fn fline_apply_fract(
    ctx: &Rex3Context,
    d: &mut i32,
    x: &mut i32,
    y: &mut i32,
    incrx2: i32,
    incry2: i32,
    y_major: bool,
) {
    let octant = ctx.bresoctinc1.octant() & 7;
    let x1p = ctx.xstart >> 11;
    let y1p = ctx.ystart >> 11;
    let x2p = ctx.xend >> 11;
    let y2p = ctx.yend >> 11;
    let mut dx = (x1p - x2p).abs();
    let mut dy = (y1p - y2p).abs();
    let mut xf = ((ctx.xstart >> 7) & 0xF) as i32;
    let mut yf = ((ctx.ystart >> 7) & 0xF) as i32;

    match octant {
        1 => {
            std::mem::swap(&mut xf, &mut yf);
            std::mem::swap(&mut dx, &mut dy);
        }
        3 => {
            xf = 0x10 - xf;
            std::mem::swap(&mut xf, &mut yf);
            std::mem::swap(&mut dx, &mut dy);
        }
        7 => { xf = 0x10 - xf; }
        6 => {
            xf = 0x10 - xf;
            yf = 0x10 - yf;
        }
        2 => {
            let t = 0x10 - xf;
            xf = 0x10 - yf;
            yf = t;
            std::mem::swap(&mut dx, &mut dy);
        }
        0 => {
            let t = 0x10 - yf;
            yf = xf;
            xf = t;
            std::mem::swap(&mut dx, &mut dy);
        }
        4 => { yf = 0x10 - yf; }
        _ => {}
    }

    // `*d` arrives holding the I_LINE decision variable (2*minor - major,
    // computed by setup() and shared by all line adrmodes). The REX3 spec
    // (rex3_pdf.md 3.6.1.2/3.6.2.1) defines F_LINE/A_LINE's base d as
    // 3*minor - 2*major instead — confirmed against MAME's do_fline,
    // which independently derives the same 3dy-2dx formula. The two
    // formulas differ by exactly (minor - major); apply that correction
    // before adding the fractional term below. Without it, the fractional
    // term is added to the wrong baseline and can flip d's sign on the
    // very first step for near-degenerate (small minor-axis) lines,
    // producing a spurious extra step in the minor-axis direction that
    // the line never recovers from (confirmed via a real R4400/REX3
    // fractional-line test that undershot its endpoint by one row).
    *d += dy - dx;
    *d += 2 * (((dx * yf) >> 4) - ((dy * xf) >> 4));
    let major_delta = if y_major { dy } else { dx };
    let e = *d - 2 * major_delta;
    if e > 0 {
        *d = e;
        let x_major = !y_major;
        if x_major {
            *y -= incry2;
        } else {
            *x += incrx2;
        }
    }
}

pub fn draw_iline_g<M: Mode>(fb: &Framebuffers, ctx: &mut Rex3Context, m: &M) {
    draw_line_bresenham(fb, ctx, m, false, false, false);
}

pub fn draw_fline_g<M: Mode>(fb: &Framebuffers, ctx: &mut Rex3Context, m: &M) {
    draw_line_bresenham(fb, ctx, m, true, false, false);
}

pub fn draw_aline_g<M: Mode>(fb: &Framebuffers, ctx: &mut Rex3Context, m: &M) {
    let mut extra_skip_first = false;
    let mut extra_skip_last = false;
    if m.endptfilter() != 0 {
        // Basic endpoint filter: consult AWEIGHT LUT for sub-pixel coverage.
        let xsf = (ctx.xstart >> 7) & 0xF;
        let ysf = (ctx.ystart >> 7) & 0xF;
        let xef = (ctx.xend >> 7) & 0xF;
        let yef = (ctx.yend >> 7) & 0xF;
        if xsf != 0 || ysf != 0 {
            let wi = ((xsf + ysf) as usize).min(15);
            let w = (ctx.aweight0 >> (wi * 4)) & 0xF;
            if w == 0 {
                extra_skip_first = true;
            }
        }
        if xef != 0 || yef != 0 {
            let wi = ((xef + yef) as usize).min(15);
            let w = (ctx.aweight1 >> (wi * 4)) & 0xF;
            if w == 0 {
                extra_skip_last = true;
            }
        }
    }
    draw_line_bresenham(fb, ctx, m, true, extra_skip_first, extra_skip_last);
}

fn draw_line_bresenham<M: Mode>(
    fb: &Framebuffers,
    ctx: &mut Rex3Context,
    m: &M,
    fract: bool,
    extra_skip_first: bool,
    extra_skip_last: bool,
) {
    let octant = (ctx.bresoctinc1.octant() & 7) as usize;
    let (incrx1, incrx2, incry1, incry2, y_major) = REX3_BRES_OCTANTS[octant];

    let x2 = ctx.xend >> 11;
    let y2 = ctx.yend >> 11;
    let mut x = ctx.xstart >> 11;
    let mut y = ctx.ystart >> 11;

    // All Bresenham state comes from registers — set by setup() or restored across GOs.
    // incr1: 20-bit, always positive (no sign extension needed).
    let incr1 = ctx.bresoctinc1.incr1() as i32;
    // incr2: 21-bit signed — sign-extend from bit 20.
    let incr2 = {
        let raw = ctx.bresrndinc2.incr2();
        if raw & (1 << 20) != 0 { (raw | 0xFFE0_0000) as i32 } else { raw as i32 }
    };
    // d: 27-bit signed — sign-extend from bit 26.  Persisted across step-mode GOs.
    // For F_LINE/A_LINE (fract=true), the fractional-endpoint correction was
    // already applied in setup() (see setup()'s doc comment) — bresd/xstart/ystart
    // read here are already correct, no further adjustment needed.
    let mut d = {
        let raw = ctx.bresd & 0x7FF_FFFF;
        if raw & (1 << 26) != 0 { (raw | 0xF800_0000) as i32 } else { raw as i32 }
    };

    // pixel_count = major_axis_length + 1 (both endpoints inclusive).
    // max(|dx|,|dy|): continuation GOs (dosetup clear) must still walk the full
    // segment when persisted octant y_major disagrees with start→end (e.g. a
    // degenerate setup GO followed by a horizontal stipple continuation).
    let adx = (x2 - x).abs();
    let ady = (y2 - y).abs();
    let major = adx.max(ady);
    let mut pixel_count = major + 1;
    if m.length32() != 0 && pixel_count > 32 {
        pixel_count = 32;
    }

    let iterate_one = m.stoponx() == 0 && m.stopony() == 0;
    let mut skip_first = m.skipfirst() != 0 || extra_skip_first;
    let mut skip_last = m.skiplast() != 0 || extra_skip_last;
    if iterate_one {
        pixel_count = 1;
        skip_first = false;
        skip_last = false;
    }


    let lsadvlast  = m.lsadvlast() != 0;

    macro_rules! bres_step {
        () => {
            if d < 0 {
                x += incrx1; y -= incry1; d += incr1;
            } else {
                x += incrx2; y -= incry2; d += incr2;
            }
        };
    }

    // Loop-invariant: fastclear writes COLORVRAM replicated across plane
    // slots, and neither input changes within a primitive.
    let fc_color = fastclear_color(ctx, m);

    for i in 0..pixel_count {
        let is_first = i == 0;
        let is_last  = i == pixel_count - 1;

        // Write pixel unless suppressed by skip_first/skip_last.
        // iterate_one overrides skip_first so single-step mode always draws.
        let draw = (!is_first || !skip_first) && (!is_last || !skip_last);
        if draw {
            pixel(fb, ctx, m, fc_color, x, y);
        }

        iterate_shade(ctx, m);
        if !is_last || lsadvlast {
            iterate_pattern(ctx, m);
        }

        // On the last pixel of a full I_LINE draw, verify Bresenham landed on x2,y2 —
        // integer Bresenham is exact, so this is a real invariant for I_LINE.
        // F_LINE/A_LINE do NOT get this check: a fractional start biases the initial
        // error term but the loop still steps by whole pixels along the major axis,
        // so the minor-axis position at the final major-axis step is the closest
        // integer approximation to the true (fractional) line, not necessarily the
        // literal requested endpoint — this is expected behavior for fractional
        // Bresenham (confirmed independently: real hardware/software fractional-DDA
        // implementations only guarantee landing in the endpoint's pixel *column/row*,
        // not its exact minor-axis coordinate).
        debug_assert!(
            fract || iterate_one || !is_last || (x == x2 && y == y2),
            "I_LINE bres mismatch: pos ({},{}) != end ({},{})", x, y, x2, y2
        );

        // In full-line mode: do NOT step after the last pixel — that would leave
        // xstart/ystart one position beyond the endpoint, breaking the next XYENDI GO
        // (dosetup re-derives Bresenham from xstart).
        // In step mode: always step so the next single-step GO starts at the next position.
        if !is_last || iterate_one {
            bres_step!();
        }
    }

    ctx.xstart = x << 11;
    ctx.ystart = y << 11;
    // Persist d so the next GO (step mode) picks up where we left off.
    ctx.bresd = (d as u32) & 0x7FF_FFFF;
}



pub fn draw_primitive(ctx: &mut Rex3Context) {
    let dm0 = ctx.drawmode0.0;
    let dm1 = ctx.drawmode1.0;
    let clipmode = ctx.clipmode;

    // One decoder. `unpack` owns the canonicalisation — the FASTCLEAR fold, the
    // dead blend inputs, the host fields — so it is not restated here.
    let m = sh::unpack(dm0, dm1, clipmode);

    draw(ctx, &m);
}
