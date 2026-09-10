//! SIMD-friendly fast paths for common REX3 interpreter draws (pre rex-jit).

use crate::rex3::{Rex3, Rex3Context};

// ctx.xstart/ystart/xend/yend (>>11) are WINDOW-RELATIVE coordinates —
// calculate_fb_address adds XYWIN's offset (plus xymove, when applicable)
// and only THEN subtracts REX3_COORD_BIAS to get the physical framebuffer
// position (see calculate_fb_address in rex3.rs: `x_abs = x_curr + x_off`
// then `x_phys = x_abs - REX3_COORD_BIAS`). There is deliberately no
// pre-loop bounding-box clamp here — any clamp applied directly to
// xstart/xend before XYWIN is added would reject legitimate coordinates
// for windows not positioned at the absolute screen origin (confirmed:
// an earlier attempt to clamp to a fixed biased screen range broke solid
// fills for most on-screen windows). calculate_fb_address already performs
// the real, XYWIN-aware bounds check per pixel — same as the interpreter's
// unclamped draw_block/draw_span fallback loop — so no pre-clamp is needed;
// xstart/xend/ystart/yend are bounded 16-bit register fields regardless,
// so the loop range can't run away even without one.

/// Try to fast-fill a fastclear BLOCK without per-pixel interpreter overhead.
/// Returns true when the entire primitive was handled.
pub fn try_fastclear_block(rex: &Rex3, ctx: &Rex3Context) -> bool {
    if !ctx.drawmode1.fastclear() {
        return false;
    }
    if ctx.drawmode0.enzpattern() || ctx.drawmode0.enlspattern() {
        return false;
    }
    if ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost() {
        return false;
    }
    if ctx.drawmode0.adrmode() != 1 {
        return false; // BLOCK only
    }

    let x0 = ctx.xstart >> 11;
    let x1 = ctx.xend >> 11;
    let y0 = ctx.ystart >> 11;
    let y1 = ctx.yend >> 11;
    let (x_lo, x_hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    let (y_lo, y_hi) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };

    let color = Rex3::fastclear_color(ctx);
    let wr_fn = unsafe { *rex.px_wr.get() };
    let mut rows = 0u64;

    for y in y_lo..=y_hi {
        for x in x_lo..=x_hi {
            if let Some(addr) = rex.calculate_fb_address(x, y, ctx, true) {
                wr_fn(rex, addr, color);
            }
        }
        rows += 1;
    }

    rex.simd_fill_rows.fetch_add(rows, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Fast SRC logicop horizontal span (solid RGB, no blend/pattern/host).
pub fn try_src_span_rgb(rex: &Rex3, ctx: &Rex3Context) -> bool {
    use crate::rex3::{DRAWMODE0_ADRMODE_SPAN, DRAWMODE1_LOGICOP_SRC};
    if ctx.drawmode0.adrmode() << 2 != DRAWMODE0_ADRMODE_SPAN {
        return false;
    }
    if ctx.drawmode1.logicop() != DRAWMODE1_LOGICOP_SRC >> 28 {
        return false;
    }
    if ctx.drawmode0.enzpattern() || ctx.drawmode0.enlspattern() {
        return false;
    }
    if ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost() || ctx.drawmode0.shade() {
        return false;
    }
    if ctx.drawmode1.planes() != 0 {
        return false; // RGB/RGBA planes only
    }
    // Blending and the alpha-vs-ALPHAREF compare are both per-pixel decisions this
    // path cannot make: it precomputes one solid colour and stores it unconditionally,
    // which would bypass the blend entirely and write pixels the alpha test should
    // have killed. COMPARE == 0x7 means the test is disabled (always pass).
    if ctx.drawmode1.blend() || ctx.drawmode1.compare() != 0x7 {
        return false;
    }
    // Dithering needs a per-pixel bayer(x,y) threshold — not representable as a
    // single precomputed fill color, so fall back to the interpreter for that case.
    if ctx.drawmode1.dither() {
        return false;
    }

    let x0 = ctx.xstart >> 11;
    let x1 = ctx.xend >> 11;
    let y = ctx.ystart >> 11;
    let (x_lo, x_hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };

    // Match process_pixel_noblend's src formatting: compress raw 24-bit BGR
    // (get_colori()) down to the framebuffer's plane depth, then amplify for
    // dblsrc packing (replicates into both packed-pixel slots so whichever
    // slot wrmask selects gets the right bits) — without this, a raw
    // get_colori() write is only correct for drawdepth==3 (24bpp)/no-dblsrc,
    // and lands unpacked/misaligned bits for every other depth.
    let compress_fn = unsafe { *rex.px_compress.get() };
    let amp_fn = unsafe { *rex.px_amp.get() };
    let color = amp_fn(compress_fn(ctx.get_colori()));
    let wr_fn = unsafe { *rex.px_wr.get() };
    for x in x_lo..=x_hi {
        if let Some(addr) = rex.calculate_fb_address(x, y, ctx, true) {
            wr_fn(rex, addr, color);
        }
    }
    rex.simd_fill_rows.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    true
}

/// Fast SRC logicop axis-aligned BLOCK fill (solid RGB, no blend/pattern/host).
pub fn try_src_block_rgb(rex: &Rex3, ctx: &Rex3Context) -> bool {
    use crate::rex3::{DRAWMODE1_LOGICOP_SRC};
    if ctx.drawmode0.adrmode() != 1 {
        return false;
    }
    if ctx.drawmode1.logicop() != DRAWMODE1_LOGICOP_SRC >> 28 {
        return false;
    }
    if ctx.drawmode0.enzpattern() || ctx.drawmode0.enlspattern() {
        return false;
    }
    if ctx.drawmode0.colorhost() || ctx.drawmode0.alphahost() || ctx.drawmode0.shade() {
        return false;
    }
    if ctx.drawmode1.planes() != 0 || ctx.drawmode1.fastclear() {
        return false;
    }
    // See try_src_span_rgb: blend and the alpha compare are per-pixel decisions
    // this precomputed-colour path cannot honour.
    if ctx.drawmode1.blend() || ctx.drawmode1.compare() != 0x7 {
        return false;
    }
    // See try_src_span_rgb: dithering needs a per-pixel bayer(x,y) threshold,
    // not representable as a single precomputed fill color.
    if ctx.drawmode1.dither() {
        return false;
    }

    let x0 = ctx.xstart >> 11;
    let x1 = ctx.xend >> 11;
    let y0 = ctx.ystart >> 11;
    let y1 = ctx.yend >> 11;
    let (x_lo, x_hi) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
    let (y_lo, y_hi) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };

    // See try_src_span_rgb: compress to plane depth + amplify for dblsrc
    // packing, matching process_pixel_noblend's src formatting.
    let compress_fn = unsafe { *rex.px_compress.get() };
    let amp_fn = unsafe { *rex.px_amp.get() };
    let color = amp_fn(compress_fn(ctx.get_colori()));
    let wr_fn = unsafe { *rex.px_wr.get() };
    let mut rows = 0u64;

    for y in y_lo..=y_hi {
        for x in x_lo..=x_hi {
            if let Some(addr) = rex.calculate_fb_address(x, y, ctx, true) {
                wr_fn(rex, addr, color);
            }
        }
        rows += 1;
    }

    rex.simd_fill_rows.fetch_add(rows, std::sync::atomic::Ordering::Relaxed);
    true
}
