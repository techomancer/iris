// IMPACT's records as the bin and raster kernels read them, and what
// both need of a GL triangle: the window position of a framebuffer pixel
// and the edges' span bounds. Layout: wgpu/mod.rs `Rec`; offsets arrive as
// defines.

#include "pp1.wgsl"

struct Rec {
    kind: u32,
    slot: u32,
    flags: u32,
    pad: u32,
    w: array<u32, REC_WORDS>,
}

@group(1) @binding(0) var<storage, read> recs: array<Rec>;

fn ru(pi: u32, i: u32) -> u32 {
    return recs[pi].w[i];
}

fn ri(pi: u32, i: u32) -> i32 {
    return bitcast<i32>(recs[pi].w[i]);
}

fn r64(pi: u32, i: u32) -> I64 {
    return I64(recs[pi].w[i], recs[pi].w[i + 1u]);
}

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

// `rss::edge_bound`: the first pixel right of edge k at row j,
// -((B j + C) >> 25) saturated to i32.
fn edge_bound(pi: u32, k: u32, j: i32) -> i32 {
    let e = T_EDGES + 4u * k;
    let v = i64_madd(r64(pi, e), r64(pi, e + 2u), j);
    return i64_low_i32(i64_clamp32(i64_neg(i64_shr(v, 25u)), -2147483647 - 1, 2147483647));
}
