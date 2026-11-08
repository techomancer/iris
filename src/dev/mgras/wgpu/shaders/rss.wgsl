// IMPACT's raster engine (RE4) as hooks for the common tile pass: one lane
// a framebuffer pixel decides whether the record covers it, then runs it
// through the pixel processors (pp1.wgsl) and the texture engine
// (te1.wgsl). No spans are walked: coverage is a per-pixel test of the
// same quantities the interpreter (rss.rs) iterates.
//
// - fill: the block's rectangle (`Rss::fill`, `Rss::point`);
// - X line: Bresenham's step along the major axis in closed form
//   (`xline_step`; brute-forced against `Rss::line`'s loop in the tests);
// - character stipple and transfer lines: the block row and column the
//   pixel is at (`Rss::block_px` inverted);
// - GL triangle: the row's edges in exact doubles (f64.wgsl), as
//   `Rss::triangle` computes them, and the half-open span test;
// - GL line: the pixel's step along the major axis and the minor extent
//   at it, likewise in doubles (`Rss::gl_line`).
//
// Record layout: wgpu/mod.rs `Rec`; offsets arrive as defines.

#include "f64.wgsl"
#include "te1.wgsl"

struct Rec {
    kind: u32,
    slot: u32,
    flags: u32,
    pad: u32,
    w: array<u32, REC_WORDS>,
}

@group(1) @binding(0) var<storage, read> recs: array<Rec>;
// Transfer pixels, as the CPU decoded them (`GpuRaster::transfer_line`).
@group(2) @binding(3) var<storage, read> stage: array<u32>;

fn ru(pi: u32, i: u32) -> u32 {
    return recs[pi].w[i];
}

fn ri(pi: u32, i: u32) -> i32 {
    return bitcast<i32>(recs[pi].w[i]);
}

fn r64(pi: u32, i: u32) -> I64 {
    return I64(recs[pi].w[i], recs[pi].w[i + 1u]);
}

const F64_HALF: I64 = I64(0u, 0x3fe00000u);

// Framebuffer (x, y) to window coordinates (`Rss::to_fb` inverted).
fn window_of(si: u32, x: i32, y: i32) -> vec2<i32> {
    let win = sr(si, SL_XYWIN);
    let ox = s16(win);
    let oy = s16(win >> 16u);
    if (sr(si, SL_CONFIG) & CONFIG_YFLIP) != 0u {
        return vec2<i32>(x - ox, oy - y);
    }
    return vec2<i32>(x - ox, y - oy);
}

fn fill(pi: u32, si: u32, x: i32, y: i32) {
    if x < ri(pi, F_BOX) || y < ri(pi, F_BOX + 1u) || x > ri(pi, F_BOX + 2u) || y > ri(pi, F_BOX + 3u) {
        return;
    }
    if (recs[pi].flags & FILL36) != 0u {
        put36(si, x, y, vec2<u32>(ru(pi, F_V36), ru(pi, F_V36 + 1u)));
    } else {
        put(si, x, y, ru(pi, F_COLOUR));
    }
}

// The step along an X line (Bresenham from p0 to p1, `Rss::line`) at which
// it draws window (x, y), or -1. Along the major axis k steps; by then the
// minor axis has taken floor((2 minor k + major) / (2 major)) steps.
fn xline_step(x0: i32, y0: i32, x1: i32, y1: i32, skip_last: bool, x: i32, y: i32) -> i32 {
    let dx = abs(x1 - x0);
    let dy = abs(y1 - y0);
    let sx = select(-1, 1, x0 < x1);
    let sy = select(-1, 1, y0 < y1);
    var major = dy;
    var minor = dx;
    var k = (y - y0) * sy;
    var at = x;
    var start = x0;
    var s = sx;
    if dx >= dy {
        major = dx;
        minor = dy;
        k = (x - x0) * sx;
        at = y;
        start = y0;
        s = sy;
    }
    if k < 0 || k > major || (skip_last && k == major) {
        return -1;
    }
    var n = 0u;
    if major != 0 {
        let p = u32(minor) * u32(k);
        let q = p / u32(major);
        let rem = p % u32(major);
        n = q + (2u * rem + u32(major)) / (2u * u32(major));
    }
    if at != start + s * i32(n) {
        return -1;
    }
    return k;
}

fn xline(pi: u32, si: u32, x: i32, y: i32) {
    let w = window_of(si, x, y);
    let flags = recs[pi].flags;
    let k = xline_step(ri(pi, L_P0), ri(pi, L_P0 + 1u), ri(pi, L_P1), ri(pi, L_P1 + 1u), (flags & LINE_SKIP_LAST) != 0u, w.x, w.y);
    if k < 0 {
        return;
    }
    var c = ru(pi, L_COLOUR);
    if (flags & LINE_STIPPLE) != 0u && (ru(pi, L_PATTERN) & (1u << (31u - u32(k) % 32u))) == 0u {
        if (flags & LINE_OPAQUE) == 0u {
            return;
        }
        c = ru(pi, L_BG);
    }
    put(si, x, y, c);
}

// The block column and row of window (x, y) (`Rss::block_px` inverted).
fn block_cell(pi: u32, w: vec2<i32>) -> vec2<i32> {
    return vec2<i32>((w.x - ri(pi, B_XS)) * ri(pi, B_DX), (w.y - ri(pi, B_YS)) * ri(pi, B_DY));
}

fn stipple(pi: u32, si: u32, x: i32, y: i32) {
    let cell = block_cell(pi, window_of(si, x, y));
    let c0 = ri(pi, S_COL);
    if cell.y != ri(pi, B_ROW) || cell.x < c0 || cell.x >= c0 + ri(pi, S_COUNT) {
        return;
    }
    let i = u32(cell.x - c0);
    var bit: u32;
    if i < 32u {
        bit = (ru(pi, S_BITS) >> (31u - i)) & 1u;
    } else {
        bit = (ru(pi, S_BITS + 1u) >> (63u - i)) & 1u;
    }
    var c = ru(pi, S_COLOUR);
    if bit == 0u {
        if (recs[pi].flags & STIPPLE_OPAQUE) == 0u {
            return;
        }
        c = ru(pi, S_BG);
    }
    put(si, x, y, c);
}

fn transfer(pi: u32, si: u32, x: i32, y: i32) {
    let cell = block_cell(pi, window_of(si, x, y));
    if cell.y != ri(pi, B_ROW) || cell.x < 0 || cell.x >= ri(pi, X_COUNT) {
        return;
    }
    let v = stage[ru(pi, X_OFF) + u32(cell.x)];
    let flags = recs[pi].flags;
    switch flags & 3u {
        case XFER_PUT: {
            put(si, x, y, v);
        }
        case XFER_FRAG: {
            put_fragment(si, x, y, v, (flags & XFER_ALPHA) != 0u);
        }
        case XFER_DEPTH: {
            if visible(si, x, y) {
                let zb = buf_new(ZST_PAGE, KIND_WIDE, sr(si, SL_DRBSIZE));
                let old = pix_get(zb, u32(x), u32(y));
                pix_put(zb, u32(x), u32(y), vec2<u32>((old.x & ~0xffffffu) | v, old.y));
            }
        }
        default: {
            if visible(si, x, y) {
                let zb = buf_new(ZST_PAGE, KIND_WIDE, sr(si, SL_DRBSIZE));
                let wmask = (sr(si, SL_STMASK) >> 8u) & 0xffu;
                let old = pix_get(zb, u32(x), u32(y));
                pix_put(zb, u32(x), u32(y), vec2<u32>((old.x & ~(wmask << 24u)) | ((v & wmask) << 24u), old.y));
            }
        }
    }
}

// A fixed plane at (xs + di, yref - dj) (`fixed::at`), wrapping.
fn plane_at(pi: u32, i: u32, di: i32, dj: i32) -> i32 {
    return ri(pi, i) + ri(pi, i + 1u) * di + ri(pi, i + 2u) * dj;
}

fn plane64_at(pi: u32, i: u32, di: i32, dj: i32) -> I64 {
    return i64_madd(i64_madd(r64(pi, i), r64(pi, i + 2u), di), r64(pi, i + 4u), dj);
}

// An edge's x at the centre of row j: x + s (yref - (j + 0.5)).
fn edge_x(pi: u32, x: u32, s: u32, yref: u32, yc: I64) -> I64 {
    return f64_add(r64(pi, x), f64_mul(r64(pi, s), f64_sub(r64(pi, yref), yc)));
}

fn triangle(pi: u32, si: u32, x: i32, y: i32) {
    let w = window_of(si, x, y);
    let i = w.x;
    let j = w.y;
    if j < ri(pi, T_J0) || j >= ri(pi, T_J1) {
        return;
    }
    let flags = recs[pi].flags;
    let yc = f64_add(f64_from_i32(j), F64_HALF);
    let major = edge_x(pi, T_X0, T_S0, T_YREF, yc);
    var minor: I64;
    if f64_ge(yc, r64(pi, T_YMID)) {
        minor = edge_x(pi, T_X1, T_S1, T_YREF, yc);
    } else {
        minor = edge_x(pi, T_X2, T_S2, T_YREF2, yc);
    }
    var l = minor;
    var r = major;
    if (flags & TRI_LTOR) != 0u {
        l = major;
        r = minor;
    }
    if i < f64_ceil_i32(f64_sub(l, F64_HALF)) || i >= f64_ceil_i32(f64_sub(r, F64_HALF)) {
        return;
    }
    if (flags & TRI_STIPPLE) != 0u && ((sr(si, SL_STIPPLE + (u32(j) & 31u)) >> (31u - (u32(i) & 31u))) & 1u) == 0u {
        return;
    }
    let di = i - ri(pi, T_XS);
    let dj = ri(pi, T_YREFI) - j;
    var rgba = vec4<i32>(plane_at(pi, T_C, di, dj), plane_at(pi, T_C + 3u, di, dj), plane_at(pi, T_C + 6u, di, dj), plane_at(pi, T_C + 9u, di, dj));
    if (flags & TRI_TEX) != 0u {
        let sw = plane64_at(pi, T_T, di, dj);
        let tw = plane64_at(pi, T_T + 6u, di, dj);
        let wi = plane64_at(pi, T_T + 12u, di, dj);
        let ds = array<I64, 2>(r64(pi, T_T + 2u), r64(pi, T_T + 4u));
        let dt = array<I64, 2>(r64(pi, T_T + 8u), r64(pi, T_T + 10u));
        let dw = array<I64, 2>(r64(pi, T_T + 14u), r64(pi, T_T + 16u));
        rgba = textured(si, sw, tw, wi, ds, dt, dw, rgba);
    }
    if (flags & TRI_FOG) != 0u {
        rgba = fogged(si, rgba, plane_at(pi, T_FOG, di, dj));
    }
    gl_fragment(si, x, y, rgba, plane64_at(pi, T_Z, di, dj));
}

fn gl_line(pi: u32, si: u32, x: i32, y: i32) {
    let w = window_of(si, x, y);
    let flags = recs[pi].flags;
    let xmajor = (flags & GL_XMAJOR) != 0u;
    let p = select(w.y, w.x, xmajor);
    let q = select(w.x, w.y, xmajor);
    let dir = ri(pi, G_DIR);
    let p0 = ri(pi, G_P0);
    let k = (p - p0) * dir;
    if k < 0 || k >= (ri(pi, G_P1) - p0) * dir {
        return;
    }
    if (flags & GL_STIPPLE) != 0u {
        let bit = ((ru(pi, G_POS) + u32(k)) / ru(pi, G_REPEAT)) % 16u;
        if ((ru(pi, G_PATTERN) >> (15u - bit)) & 1u) == 0u {
            return;
        }
    }
    let width = ri(pi, G_WIDTH);
    let pc = f64_add(f64_from_i32(p), F64_HALF);
    let b = f64_add(r64(pi, G_B0), f64_mul(r64(pi, G_SLOPE), f64_sub(pc, r64(pi, G_A0))));
    let q0 = f64_floor_i32(f64_add(f64_sub(b, f64_mul(F64_HALF, f64_from_i32(width))), F64_HALF));
    let qe = q0 + width;
    if qe < q0 || q < q0 || q >= qe {
        return;
    }
    var rgba = vec4<i32>(0);
    for (var c = 0u; c < 4u; c++) {
        rgba[c] = ri(pi, G_CBASE + c) + ri(pi, G_CSTEP + c) * k;
    }
    if (flags & GL_TEX) != 0u {
        let sw = i64_madd(r64(pi, G_TBASE), r64(pi, G_TSTEP), k);
        let tw = i64_madd(r64(pi, G_TBASE + 2u), r64(pi, G_TSTEP + 2u), k);
        let wi = i64_madd(r64(pi, G_TBASE + 4u), r64(pi, G_TSTEP + 4u), k);
        let zero = I64(0u, 0u);
        let ds = array<I64, 2>(r64(pi, G_TSTEP), zero);
        let dt = array<I64, 2>(r64(pi, G_TSTEP + 2u), zero);
        let dw = array<I64, 2>(r64(pi, G_TSTEP + 4u), zero);
        rgba = textured(si, sw, tw, wi, ds, dt, dw, rgba);
    }
    if (flags & GL_FOG) != 0u {
        rgba = fogged(si, rgba, ri(pi, G_FBASE) + ri(pi, G_FSTEP) * k);
    }
    gl_fragment(si, x, y, rgba, i64_madd(r64(pi, G_ZBASE), r64(pi, G_ZSTEP), k));
}

fn arch_tile_begin(tile: u32, px: i32, py: i32, lane: u32) {
}

fn arch_prim(pi: u32, px: i32, py: i32, lane: u32) {
    let si = recs[pi].slot;
    switch recs[pi].kind {
        case REC_FILL: {
            fill(pi, si, px, py);
        }
        case REC_LINE: {
            xline(pi, si, px, py);
        }
        case REC_STIPPLE: {
            stipple(pi, si, px, py);
        }
        case REC_XFER: {
            transfer(pi, si, px, py);
        }
        case REC_TRI: {
            triangle(pi, si, px, py);
        }
        case REC_GLLINE: {
            gl_line(pi, si, px, py);
        }
        default: {}
    }
}

fn arch_tile_end(tile: u32, px: i32, py: i32, lane: u32) {
}
