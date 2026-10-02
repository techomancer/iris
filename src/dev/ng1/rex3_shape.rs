//! Canonical decoding of the REX3 draw-mode registers.
//!
//! One place that answers "what shape of draw is this?", so the interpreter, the
//! Cranelift JIT and (later) the generated monomorphised draw table all agree by
//! construction instead of by three hand-kept copies.
//!
//! Today this holds only `normalize_dm1`. The field extractors and the
//! `unpack()` that feeds the const-generic draw path land in the next stage.

use crate::dev::ng1::rex3::{
    CLIPMODE_CIDMATCH_SHIFT, CLIPMODE_ENSMASK_MASK, DRAWMODE0_OPCODE_DRAW,
    DRAWMODE0_OPCODE_READ, DRAWMODE0_OPCODE_SCR2SCR,
};

/// DRAWMODE1 bit 16 — DITHER.
const DM1_DITHER: u32 = 1 << 16;
/// DRAWMODE1 bit 17 — FASTCLEAR.
const DM1_FASTCLEAR: u32 = 1 << 17;
/// DRAWMODE1 bit 18 — BLEND.
const DM1_BLEND: u32 = 1 << 18;

// ── DRAWMODE0 field extractors ───────────────────────────────────────────────
// Bit positions mirror the `DrawMode0` bitfield in rex3.rs. These are `const fn`
// so they work in const contexts, in the JIT's runtime code, and as the source
// of the constants baked into the generated draw table.

#[inline(always)] pub const fn dm0_opcode(v: u32)      -> u32 { v & 0x3 }
#[inline(always)] pub const fn dm0_adrmode(v: u32)     -> u32 { (v >> 2) & 0x7 }
#[inline(always)] pub const fn dm0_dosetup(v: u32)     -> u32 { (v >> 5) & 1 }
#[inline(always)] pub const fn dm0_colorhost(v: u32)   -> u32 { (v >> 6) & 1 }
#[inline(always)] pub const fn dm0_alphahost(v: u32)   -> u32 { (v >> 7) & 1 }
#[inline(always)] pub const fn dm0_stoponx(v: u32)     -> u32 { (v >> 8) & 1 }
#[inline(always)] pub const fn dm0_stopony(v: u32)     -> u32 { (v >> 9) & 1 }
#[inline(always)] pub const fn dm0_skipfirst(v: u32)   -> u32 { (v >> 10) & 1 }
#[inline(always)] pub const fn dm0_skiplast(v: u32)    -> u32 { (v >> 11) & 1 }
#[inline(always)] pub const fn dm0_enzpattern(v: u32)  -> u32 { (v >> 12) & 1 }
#[inline(always)] pub const fn dm0_enlspattern(v: u32) -> u32 { (v >> 13) & 1 }
#[inline(always)] pub const fn dm0_lsadvlast(v: u32)   -> u32 { (v >> 14) & 1 }
#[inline(always)] pub const fn dm0_length32(v: u32)    -> u32 { (v >> 15) & 1 }
#[inline(always)] pub const fn dm0_zpopaque(v: u32)    -> u32 { (v >> 16) & 1 }
#[inline(always)] pub const fn dm0_lsopaque(v: u32)    -> u32 { (v >> 17) & 1 }
#[inline(always)] pub const fn dm0_shade(v: u32)       -> u32 { (v >> 18) & 1 }
#[inline(always)] pub const fn dm0_lronly(v: u32)      -> u32 { (v >> 19) & 1 }
#[inline(always)] pub const fn dm0_xyoffset(v: u32)    -> u32 { (v >> 20) & 1 }
#[inline(always)] pub const fn dm0_ciclamp(v: u32)     -> u32 { (v >> 21) & 1 }
#[inline(always)] pub const fn dm0_endptfilter(v: u32) -> u32 { (v >> 22) & 1 }
#[inline(always)] pub const fn dm0_ystride(v: u32)     -> u32 { (v >> 23) & 1 }

// ── DRAWMODE1 field extractors ───────────────────────────────────────────────

#[inline(always)] pub const fn dm1_planes(v: u32)     -> u32 { v & 0x7 }
#[inline(always)] pub const fn dm1_drawdepth(v: u32)  -> u32 { (v >> 3) & 0x3 }
#[inline(always)] pub const fn dm1_dblsrc(v: u32)     -> u32 { (v >> 5) & 1 }
#[inline(always)] pub const fn dm1_yflip(v: u32)      -> u32 { (v >> 6) & 1 }
#[inline(always)] pub const fn dm1_rwpacked(v: u32)   -> u32 { (v >> 7) & 1 }
#[inline(always)] pub const fn dm1_hostdepth(v: u32)  -> u32 { (v >> 8) & 0x3 }
#[inline(always)] pub const fn dm1_rwdouble(v: u32)   -> u32 { (v >> 10) & 1 }
#[inline(always)] pub const fn dm1_swapendian(v: u32) -> u32 { (v >> 11) & 1 }
#[inline(always)] pub const fn dm1_compare(v: u32)    -> u32 { (v >> 12) & 0x7 }
#[inline(always)] pub const fn dm1_rgbmode(v: u32)    -> u32 { (v >> 15) & 1 }
#[inline(always)] pub const fn dm1_dither(v: u32)     -> u32 { (v >> 16) & 1 }
#[inline(always)] pub const fn dm1_fastclear(v: u32)  -> u32 { (v >> 17) & 1 }
#[inline(always)] pub const fn dm1_blend(v: u32)      -> u32 { (v >> 18) & 1 }
#[inline(always)] pub const fn dm1_sfactor(v: u32)    -> u32 { (v >> 19) & 0x7 }
#[inline(always)] pub const fn dm1_dfactor(v: u32)    -> u32 { (v >> 22) & 0x7 }
#[inline(always)] pub const fn dm1_backblend(v: u32)  -> u32 { (v >> 25) & 1 }
#[inline(always)] pub const fn dm1_prefetch(v: u32)   -> u32 { (v >> 26) & 1 }
#[inline(always)] pub const fn dm1_blendalpha(v: u32) -> u32 { (v >> 27) & 1 }
#[inline(always)] pub const fn dm1_logicop(v: u32)    -> u32 { (v >> 28) & 0xF }

/// Host pixels per 64-bit host word. Lifted from `Rex3::host_setup`, which the
/// JIT's `Dm1::host_count` used to duplicate by hand.
#[inline(always)]
pub const fn host_count(dm1: u32) -> u32 {
    if dm1_rwpacked(dm1) == 0 {
        return 1;
    }
    let depth = dm1_hostdepth(dm1);
    if dm1_rwdouble(dm1) != 0 {
        match depth { 0 | 1 => 8, 2 => 4, 3 => 2, _ => 1 }
    } else {
        match depth { 0 | 1 => 4, 2 => 2, 3 => 1, _ => 1 }
    }
}

/// Bits per host pixel slot. Lifted from `Rex3::host_setup`.
#[inline(always)]
pub const fn host_shift(dm1: u32) -> u32 {
    if dm1_rwpacked(dm1) == 0 {
        return 0;
    }
    match dm1_hostdepth(dm1) { 0 | 1 => 8, 2 => 16, 3 => 32, _ => 0 }
}

/// Canonical DRAWMODE1 for keying and specialization.
///
/// Folds out fields that are redundant with other state, so a draw that behaves
/// identically always produces the same key — whichever engine asks:
///
/// - **FASTCLEAR clears BLEND.** Hardware ignores blending when fastclear is set.
/// - **SCR2SCR clears DITHER.** A screen-to-screen copy moves already-quantized
///   pixels; dithering them again would corrupt the copy.
///
/// The two are ordered, not combined: fastclear wins, matching the JIT's
/// `compile_shader` and the dispatch key it has to agree with.
///
/// **Scope note.** FASTCLEAR kills much more than BLEND. Per `rex3.pdf` §3.5.5:
/// "No support for any per pixel operations, such as shade, stipple, dither,
/// blend. Flat fill only, via value previously written by host into the
/// COLORVRAM register." Restricted to OPCODE=draw with ADRMODE=block or span,
/// left-to-right spans only.
///
/// IRIS matches: `process_pixel_fastclear` is address-then-write, with the
/// colour from `ctx.colorvram` (see `fastclear_color`) — no destination read,
/// logic op, afunction, host fetch, shade DDA or dither. The JIT agrees and adds
/// that `colorhost` beats fastclear (`dm1.fastclear() && !is_hostw`) and that
/// fastclear forces `compare = 7`.
///
/// MAME (`newport.cpp` ~3392) does **not** match the spec here: it uses the
/// fastclear bit only to suppress the shade/rgb colour path, falls back to
/// COLORI rather than COLORVRAM, and still evaluates stipple and the opaque
/// path. Not a usable reference for this mode.
///
/// Only the BLEND fold lives here, because this function is currently the key
/// for `planes_setup`, whose other inputs (logicop, dither, compare, drawdepth,
/// rgbmode) still have to be honoured for non-fastclear draws sharing the cache
/// slot. The rest of the fastclear folding belongs in `unpack()`, where it can
/// zero LOGICOP/COMPARE/DITHER/SFACTOR/DFACTOR/host fields for the
/// monomorphisation key and collapse every fastclear variant onto one body.
#[inline]
pub const fn normalize_dm1(dm1: u32, opcode: u32) -> u32 {
    if dm1 & DM1_FASTCLEAR != 0 {
        dm1 & !DM1_BLEND
    } else if opcode == DRAWMODE0_OPCODE_SCR2SCR {
        dm1 & !DM1_DITHER
    } else {
        dm1
    }
}

/// Decode `(dm0, dm1, clipmode)` into the canonical shape.
///
/// Canonicalisation — folding dead fields to a fixed value — is what keeps the
/// table small. Without it, drawmodes that generate byte-identical code become
/// distinct monomorphisations, because drivers leave stale bits in fields the
/// current mode ignores.
pub const fn unpack(dm0: u32, dm1: u32, clipmode: u32) -> crate::dev::ng1::rex3_generic::DynMode {
    let opcode = dm0_opcode(dm0);
    let dm1 = normalize_dm1(dm1, opcode);

    let colorhost = dm0_colorhost(dm0);
    let alphahost = dm0_alphahost(dm0);
    let is_host = colorhost != 0 || alphahost != 0 || opcode == DRAWMODE0_OPCODE_READ;

    let cidtest = if (clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF == 0xF { 0 } else { 1 };

    // FASTCLEAR collapses the whole pixel pipeline (rex3.pdf §3.5.5: "No support
    // for any per pixel operation ... Flat fill only, via COLORVRAM"), but only
    // where hardware actually honours it: DRAW + block/span, CID checking off,
    // and not overridden by colorhost. Outside that it is an ordinary draw and
    // every field below stays live.
    let fastclear_active = dm1_fastclear(dm1) != 0
        && opcode == DRAWMODE0_OPCODE_DRAW
        && cidtest == 0
        && colorhost == 0;

    // Blend inputs are dead unless blending; logicop is dead when blending,
    // since the blended value is written directly.
    let blend = if fastclear_active { 0 } else { dm1_blend(dm1) };
    let blending = blend != 0;

    crate::dev::ng1::rex3_generic::DynMode {
        opcode,
        planes: dm1_planes(dm1),
        drawdepth: dm1_drawdepth(dm1),
        dblsrc: dm1_dblsrc(dm1),
        rgbmode: dm1_rgbmode(dm1),
        dither: if fastclear_active { 0 } else { dm1_dither(dm1) },
        fastclear: if fastclear_active { 1 } else { 0 },
        blend,
        backblend: if blending { dm1_backblend(dm1) } else { 0 },
        blendalpha: if blending { dm1_blendalpha(dm1) } else { 0 },
        // Fastclear writes COLORVRAM unconditionally; blending bypasses the
        // logic op. Either way the selector stops selecting anything.
        logicop: if fastclear_active || blending { 0 } else { dm1_logicop(dm1) },
        // Fastclear bypasses afunction (the JIT forces compare=7 for it).
        // 7 is "always pass", i.e. the disabled encoding.
        compare: if fastclear_active { 7 } else { dm1_compare(dm1) },
        colorhost,
        alphahost,
        enzpattern: if fastclear_active { 0 } else { dm0_enzpattern(dm0) },
        enlspattern: if fastclear_active { 0 } else { dm0_enlspattern(dm0) },
        zpopaque: if fastclear_active { 0 } else { dm0_zpopaque(dm0) },
        lsopaque: if fastclear_active { 0 } else { dm0_lsopaque(dm0) },
        shade: if fastclear_active { 0 } else { dm0_shade(dm0) },
        // CICLAMP only governs the CI shade paths.
        ciclamp: if dm0_shade(dm0) != 0 && !fastclear_active { dm0_ciclamp(dm0) } else { 0 },
        ystride: dm0_ystride(dm0),
        lronly: dm0_lronly(dm0),
        length32: dm0_length32(dm0),
        stoponx: dm0_stoponx(dm0),
        stopony: dm0_stopony(dm0),
        skipfirst: dm0_skipfirst(dm0),
        skiplast: dm0_skiplast(dm0),
        // Stipple-advance and endpoint filtering only mean anything on lines.
        lsadvlast: if dm0_enlspattern(dm0) != 0 && !fastclear_active { dm0_lsadvlast(dm0) } else { 0 },
        endptfilter: dm0_endptfilter(dm0),
        adrmode: dm0_adrmode(dm0),
        xyoffset: dm0_xyoffset(dm0),
        yflip: dm1_yflip(dm1),
        ensmask: clipmode & CLIPMODE_ENSMASK_MASK,
        cid_mask: (clipmode >> CLIPMODE_CIDMATCH_SHIFT) & 0xF,
        sfactor: if blending { dm1_sfactor(dm1) } else { 0 },
        dfactor: if blending { dm1_dfactor(dm1) } else { 0 },
        rwpacked: if is_host { dm1_rwpacked(dm1) } else { 0 },
        hostdepth: if is_host { dm1_hostdepth(dm1) } else { 0 },
        rwdouble: if is_host { dm1_rwdouble(dm1) } else { 0 },
        swapendian: if is_host { dm1_swapendian(dm1) } else { 0 },
        // Host transfer shape is dead when no host pixels move.
        cidtest,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::ng1::rex3::{DrawMode0, DrawMode1, CLIPMODE_CIDMATCH_MASK};

    /// The expression `execute_go` and `compile_shader` each spelled out inline
    /// before this module existed. Kept verbatim as the oracle so the shared
    /// function is pinned to the behaviour the JIT key already depended on.
    fn legacy_jit_norm(dm1: u32, opcode: u32) -> u32 {
        if dm1 & (1 << 17) != 0 {
            dm1 & !(1 << 18)
        } else if opcode == DRAWMODE0_OPCODE_SCR2SCR {
            dm1 & !(1 << 16)
        } else {
            dm1
        }
    }

    #[test]
    fn matches_legacy_jit_normalization() {
        // Walk the three bits that matter against every opcode, plus a spread of
        // unrelated bit patterns to prove nothing else is disturbed.
        for extra in [0x0000_0000, 0xFFFF_FFFF, 0x3000_F019, 0x5A5A_5A5A, 0x0007_0000] {
            for bits in 0..8u32 {
                let dm1 = extra
                    | if bits & 1 != 0 { DM1_DITHER } else { 0 }
                    | if bits & 2 != 0 { DM1_FASTCLEAR } else { 0 }
                    | if bits & 4 != 0 { DM1_BLEND } else { 0 };
                for opcode in [
                    DRAWMODE0_OPCODE_DRAW,
                    DRAWMODE0_OPCODE_SCR2SCR,
                    DRAWMODE0_OPCODE_READ,
                ] {
                    assert_eq!(
                        normalize_dm1(dm1, opcode),
                        legacy_jit_norm(dm1, opcode),
                        "dm1={dm1:#010x} opcode={opcode}"
                    );
                }
            }
        }
    }

    #[test]
    fn fastclear_beats_scr2scr() {
        // Both conditions true: fastclear must win, so BLEND clears and DITHER
        // survives. Getting this order wrong is invisible until a fastclear
        // scr2scr draw keys differently in two engines.
        let dm1 = DM1_FASTCLEAR | DM1_BLEND | DM1_DITHER;
        let got = normalize_dm1(dm1, DRAWMODE0_OPCODE_SCR2SCR);
        assert_eq!(got & DM1_BLEND, 0, "fastclear must clear blend");
        assert_eq!(got & DM1_DITHER, DM1_DITHER, "dither must survive fastclear");
    }

    #[test]
    fn leaves_unrelated_modes_alone() {
        let dm1 = 0x3000_F019;
        assert_eq!(normalize_dm1(dm1, DRAWMODE0_OPCODE_DRAW), dm1);
    }

    /// Sample values that exercise every bit position: all-clear, all-set, each
    /// single bit alone, and a few real drawmodes from the saved corpus.
    fn samples() -> Vec<u32> {
        let mut v = vec![0x0000_0000, 0xFFFF_FFFF, 0x3000_F019, 0x0306, 0x0327, 0x032A];
        for bit in 0..32 {
            v.push(1 << bit);
            v.push(!(1u32 << bit));
        }
        v
    }

    /// The extractors must agree with the `bitfield!` accessors they are going to
    /// replace. This is the entire correctness basis for the migration: if a
    /// shift or mask is off by one, the generated table bakes the wrong constant
    /// into every shader and the failure is silent.
    #[test]
    fn extractors_match_bitfield_accessors() {
        for v in samples() {
            let d0 = DrawMode0(v);
            assert_eq!(dm0_opcode(v),      d0.opcode(),           "opcode {v:#010x}");
            assert_eq!(dm0_adrmode(v),     d0.adrmode(),          "adrmode {v:#010x}");
            assert_eq!(dm0_dosetup(v),     d0.dosetup() as u32,   "dosetup {v:#010x}");
            assert_eq!(dm0_colorhost(v),   d0.colorhost() as u32, "colorhost {v:#010x}");
            assert_eq!(dm0_alphahost(v),   d0.alphahost() as u32, "alphahost {v:#010x}");
            assert_eq!(dm0_stoponx(v),     d0.stoponx() as u32,   "stoponx {v:#010x}");
            assert_eq!(dm0_stopony(v),     d0.stopony() as u32,   "stopony {v:#010x}");
            assert_eq!(dm0_skipfirst(v),   d0.skipfirst() as u32, "skipfirst {v:#010x}");
            assert_eq!(dm0_skiplast(v),    d0.skiplast() as u32,  "skiplast {v:#010x}");
            assert_eq!(dm0_enzpattern(v),  d0.enzpattern() as u32,  "enzpattern {v:#010x}");
            assert_eq!(dm0_enlspattern(v), d0.enlspattern() as u32, "enlspattern {v:#010x}");
            assert_eq!(dm0_lsadvlast(v),   d0.lsadvlast() as u32, "lsadvlast {v:#010x}");
            assert_eq!(dm0_length32(v),    d0.length32() as u32,  "length32 {v:#010x}");
            assert_eq!(dm0_zpopaque(v),    d0.zpopaque() as u32,  "zpopaque {v:#010x}");
            assert_eq!(dm0_lsopaque(v),    d0.lsopaque() as u32,  "lsopaque {v:#010x}");
            assert_eq!(dm0_shade(v),       d0.shade() as u32,     "shade {v:#010x}");
            assert_eq!(dm0_lronly(v),      d0.lronly() as u32,    "lronly {v:#010x}");
            assert_eq!(dm0_xyoffset(v),    d0.xyoffset() as u32,  "xyoffset {v:#010x}");
            assert_eq!(dm0_ciclamp(v),     d0.ciclamp() as u32,   "ciclamp {v:#010x}");
            assert_eq!(dm0_endptfilter(v), d0.endptfilter() as u32, "endptfilter {v:#010x}");
            assert_eq!(dm0_ystride(v),     d0.ystride() as u32,   "ystride {v:#010x}");

            let d1 = DrawMode1(v);
            assert_eq!(dm1_planes(v),     d1.planes(),            "planes {v:#010x}");
            assert_eq!(dm1_drawdepth(v),  d1.drawdepth(),         "drawdepth {v:#010x}");
            assert_eq!(dm1_dblsrc(v),     d1.dblsrc() as u32,     "dblsrc {v:#010x}");
            assert_eq!(dm1_yflip(v),      d1.yflip() as u32,      "yflip {v:#010x}");
            assert_eq!(dm1_rwpacked(v),   d1.rwpacked() as u32,   "rwpacked {v:#010x}");
            assert_eq!(dm1_hostdepth(v),  d1.hostdepth(),         "hostdepth {v:#010x}");
            assert_eq!(dm1_rwdouble(v),   d1.rwdouble() as u32,   "rwdouble {v:#010x}");
            assert_eq!(dm1_swapendian(v), d1.swapendian() as u32, "swapendian {v:#010x}");
            assert_eq!(dm1_compare(v),    d1.compare(),           "compare {v:#010x}");
            assert_eq!(dm1_rgbmode(v),    d1.rgbmode() as u32,    "rgbmode {v:#010x}");
            assert_eq!(dm1_dither(v),     d1.dither() as u32,     "dither {v:#010x}");
            assert_eq!(dm1_fastclear(v),  d1.fastclear() as u32,  "fastclear {v:#010x}");
            assert_eq!(dm1_blend(v),      d1.blend() as u32,      "blend {v:#010x}");
            assert_eq!(dm1_sfactor(v),    d1.sfactor(),           "sfactor {v:#010x}");
            assert_eq!(dm1_dfactor(v),    d1.dfactor(),           "dfactor {v:#010x}");
            assert_eq!(dm1_backblend(v),  d1.backblend() as u32,  "backblend {v:#010x}");
            assert_eq!(dm1_prefetch(v),   d1.prefetch() as u32,   "prefetch {v:#010x}");
            assert_eq!(dm1_blendalpha(v), d1.blendalpha() as u32, "blendalpha {v:#010x}");
            assert_eq!(dm1_logicop(v),    d1.logicop(),           "logicop {v:#010x}");
        }
    }

    const CID_OFF: u32 = 0xF << CLIPMODE_CIDMATCH_SHIFT;

    /// Dead blend inputs fold to zero, so drawmodes differing only in stale
    /// sfactor/dfactor bits share one monomorphisation.
    #[test]
    fn blend_inputs_fold_when_not_blending() {
        let dm1_a = 0 << 19; // sfactor 0
        let dm1_b = 5 << 19; // sfactor 5, blend still off
        let a = unpack(DRAWMODE0_OPCODE_DRAW, dm1_a, CID_OFF);
        let b = unpack(DRAWMODE0_OPCODE_DRAW, dm1_b, CID_OFF);
        assert_eq!(a, b, "stale sfactor must not create a second shape");
        assert_eq!(a.sfactor, 0);
    }

    /// Per rex3.pdf §3.5.5 fastclear supports no per-pixel operation, so every
    /// variant collapses onto one shape regardless of the stale bits around it.
    #[test]
    fn fastclear_collapses_the_pipeline() {
        let plain = DM1_FASTCLEAR;
        let noisy = DM1_FASTCLEAR
            | DM1_BLEND
            | DM1_DITHER
            | (0x6 << 28)          // logicop XOR
            | (0x3 << 12)          // compare LE
            | (5 << 19) | (2 << 22); // sfactor/dfactor
        let dm0_noisy = DRAWMODE0_OPCODE_DRAW
            | (1 << 12)  // enzpattern
            | (1 << 18); // shade

        let a = unpack(DRAWMODE0_OPCODE_DRAW, plain, CID_OFF);
        let b = unpack(dm0_noisy, noisy, CID_OFF);
        assert_eq!(a, b, "fastclear variants must share one shape");
        assert_eq!(a.fastclear, 1);
        assert_eq!(a.blend, 0);
        assert_eq!(a.dither, 0);
        assert_eq!(a.logicop, 0);
        assert_eq!(a.shade, 0);
        assert_eq!(a.enzpattern, 0);
        assert_eq!(a.compare, 7, "fastclear bypasses afunction");
    }

    /// Hardware disables FASTCLEAR when CID checking is on (rex3.pdf bit 17,
    /// §3.5.5, and the programming notes), so the fold must not apply there —
    /// the draw is an ordinary one and its fields stay live.
    /// See rules/rex3/fastclear-cid-divergence.md.
    #[test]
    fn fastclear_does_not_fold_when_cid_checking_enabled() {
        let dm1 = DM1_FASTCLEAR | (0x6 << 28); // logicop XOR left live
        let cid_on = 0x7 << CLIPMODE_CIDMATCH_SHIFT; // != 0xF
        let f = unpack(DRAWMODE0_OPCODE_DRAW, dm1, cid_on);
        assert_eq!(f.fastclear, 0, "CID checking disables fastclear");
        assert_eq!(f.logicop, 0x6, "ordinary draw keeps its logic op");
        assert_eq!(f.cidtest, 1);
    }

    /// colorhost beats fastclear, matching the JIT's `!is_hostw` guard.
    #[test]
    fn colorhost_beats_fastclear() {
        let dm0 = DRAWMODE0_OPCODE_DRAW | (1 << 6); // colorhost
        let f = unpack(dm0, DM1_FASTCLEAR, CID_OFF);
        assert_eq!(f.fastclear, 0);
        assert_eq!(f.colorhost, 1);
    }

    /// Host-transfer shape is dead when no host pixels move.
    #[test]
    fn host_fields_fold_for_non_host_draws() {
        let dm1 = (1 << 7) | (0x2 << 8) | (1 << 10); // rwpacked, hostdepth, rwdouble
        let f = unpack(DRAWMODE0_OPCODE_DRAW, dm1, CID_OFF);
        assert_eq!((f.rwpacked, f.hostdepth, f.rwdouble), (0, 0, 0));
        // …but a READ moves host pixels, so they stay live.
        let r = unpack(DRAWMODE0_OPCODE_READ, dm1, CID_OFF);
        assert_eq!((r.rwpacked, r.hostdepth, r.rwdouble), (1, 2, 1));
    }

    #[test]
    fn cidtest_is_on_off_not_the_nibble() {
        let off = unpack(DRAWMODE0_OPCODE_DRAW, 0, CID_OFF);
        assert_eq!(off.cidtest, 0, "0xF permits every CID — no test");
        // Every other nibble, including 0x0 (permit nothing), emits the test.
        for nib in 0..0xFu32 {
            let f = unpack(DRAWMODE0_OPCODE_DRAW, 0, nib << CLIPMODE_CIDMATCH_SHIFT);
            assert_eq!(f.cidtest, 1, "nibble {nib:#x} must emit the test");
        }
    }

    /// host_count/host_shift must match the Rex3::host_setup logic they were
    /// lifted from — the JIT had its own hand-copy of this.
    #[test]
    fn host_count_shift_cover_all_depths() {
        // Unpacked: always one pixel, no shift.
        assert_eq!((host_count(0), host_shift(0)), (1, 0));
        for depth in 0..4u32 {
            let packed = (1 << 7) | (depth << 8);
            let dbl = packed | (1 << 10);
            let expect_count = match depth { 0 | 1 => 4, 2 => 2, _ => 1 };
            let expect_dbl = match depth { 0 | 1 => 8, 2 => 4, _ => 2 };
            let expect_shift = match depth { 0 | 1 => 8, 2 => 16, _ => 32 };
            assert_eq!(host_count(packed), expect_count, "depth {depth}");
            assert_eq!(host_count(dbl), expect_dbl, "depth {depth} rwdouble");
            assert_eq!(host_shift(packed), expect_shift, "depth {depth}");
        }
    }
}

#[cfg(test)]
mod corpus_check {
    use super::*;

    /// Decode the developer's saved JIT profile (if present) and report how far
    /// canonicalisation collapses it. Prints rather than asserts — no particular
    /// machine's profile is a correctness property — but the number is what
    /// table sizing rests on, so it is worth being able to see.
    /// Run with: cargo test --features rex-jit --lib corpus_check -- --nocapture
    #[test]
    fn report_corpus_collapse() {
        let path = match std::env::var_os("HOME") {
            Some(h) => std::path::PathBuf::from(h).join(".iris/rex-jit-profile.bin"),
            None => return,
        };
        let data = match std::fs::read(&path) { Ok(d) => d, Err(_) => return };
        if data.len() < 9 || &data[0..4] != b"IRXP" || data[4] != 2 {
            return;
        }
        let count = u32::from_le_bytes(data[5..9].try_into().unwrap()) as usize;

        let mut raw = std::collections::HashSet::new();
        let mut shapes = std::collections::HashSet::new();
        for i in 0..count {
            let o = 9 + i * 12;
            if o + 12 > data.len() { break; }
            let dm0 = u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
            let dm1 = u32::from_le_bytes(data[o + 4..o + 8].try_into().unwrap());
            let cm = u32::from_le_bytes(data[o + 8..o + 12].try_into().unwrap());
            raw.insert((dm0, dm1, cm));
            shapes.insert(unpack(dm0, dm1, cm));
        }
        println!(
            "corpus: {} raw triples -> {} canonical shapes",
            raw.len(),
            shapes.len()
        );
    }
}

// ── Shape-key hashing ────────────────────────────────────────────────────────

/// A fast hash for `(dm0, dm1, cm)` shape keys.
///
/// The default `HashMap` hasher is SipHash, which is ~27 ns for this 12-byte key
/// — measured, and the dominant cost of a dispatch lookup. SipHash buys DoS
/// resistance against adversarial keys; these keys come from a guest's draw-mode
/// registers, are bounded by the reachable draw-mode space, and land in a map the
/// guest cannot enumerate. That protection has nothing to defend here.
///
/// This is the FxHash construction (rustc's own): multiply by a constant with
/// good bit-mixing and rotate, folding each word in turn. No allocation, no
/// finalisation, a handful of instructions.
#[derive(Default, Clone, Copy)]
pub struct ShapeHasher(u64);

impl ShapeHasher {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

    #[inline(always)]
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(Self::SEED);
    }
}

impl std::hash::Hasher for ShapeHasher {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        // Shape keys are written as u32s; this path exists only to satisfy the
        // trait and is not on the hot path.
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(buf));
        }
    }

    #[inline(always)]
    fn write_u32(&mut self, v: u32) {
        self.add(v as u64);
    }

    #[inline(always)]
    fn write_u64(&mut self, v: u64) {
        self.add(v);
    }

    #[inline(always)]
    fn write_usize(&mut self, v: usize) {
        self.add(v as u64);
    }
}

/// `BuildHasher` for [`ShapeHasher`].
#[derive(Default, Clone, Copy)]
pub struct ShapeHashBuilder;

impl std::hash::BuildHasher for ShapeHashBuilder {
    type Hasher = ShapeHasher;
    #[inline(always)]
    fn build_hasher(&self) -> ShapeHasher {
        ShapeHasher(0)
    }
}

/// A `HashMap` keyed by shape triples.
pub type ShapeMap<V> = std::collections::HashMap<(u32, u32, u32), V, ShapeHashBuilder>;

/// A `HashSet` of shape triples.
pub type ShapeSet = std::collections::HashSet<(u32, u32, u32), ShapeHashBuilder>;
