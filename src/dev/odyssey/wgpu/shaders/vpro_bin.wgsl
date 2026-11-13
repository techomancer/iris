// VPro's bin hook: a triangle record goes to tiles only when each of its
// three edges passes at some pixel centre of them within the walk's box.
// An edge function is linear, so over a rectangle of pixel centres its
// largest value is at the corner its x and y steps point to; if even that
// fails the edge's test (raster.rs: >= 1 where the edge does not own its
// pixels, >= 0 where it does), no pixel of the tiles is drawn. Three edges
// passing at three different corners still accept the tiles: the test is
// conservative, never exact. Every other kind takes its box.

#include "i64.wgsl"
#include "vpro_rec.wgsl"

fn bin_edge(pi: u32, k: u32, x0: u32, y0: u32, x1: u32, y1: u32, need: u32) -> bool {
    let dx = I64(recs[pi].w[T_DX + 2u * k], recs[pi].w[T_DX + 2u * k + 1u]);
    let dy = I64(recs[pi].w[T_DY + 2u * k], recs[pi].w[T_DY + 2u * k + 1u]);
    let x = select(x1, x0, i64_is_neg(dx));
    let y = select(y1, y0, i64_is_neg(dy));
    var v = I64(recs[pi].w[T_E0 + 2u * k], recs[pi].w[T_E0 + 2u * k + 1u]);
    v = i64_add(v, i64_mul_u16(dx, x));
    v = i64_add(v, i64_mul_u16(dy, y));
    return i64_ge_need(v, need);
}

fn bin_accept(pi: u32, tx0: u32, ty0: u32, tx1: u32, ty1: u32) -> bool {
    if recs[pi].kind != REC_TRI {
        return true;
    }
    // The tiles' pixels in plane coordinates, cut to the walk's box.
    let bx = vec4<i32>(
        bitcast<i32>(recs[pi].w[T_BOX]),
        bitcast<i32>(recs[pi].w[T_BOX + 1u]),
        bitcast<i32>(recs[pi].w[T_BOX + 2u]),
        bitcast<i32>(recs[pi].w[T_BOX + 3u]),
    );
    let x0 = max(batch.origin_x + i32(tx0 * TILE_W), bx.x);
    let y0 = max(batch.origin_y + i32(ty0 * TILE_H), bx.y);
    let x1 = min(batch.origin_x + i32(tx1 * TILE_W + TILE_W - 1u), bx.z);
    let y1 = min(batch.origin_y + i32(ty1 * TILE_H + TILE_H - 1u), bx.w);
    if x0 > x1 || y0 > y1 {
        return false;
    }
    let need = recs[pi].w[T_NEED];
    let r = vec4<u32>(vec4<i32>(x0, y0, x1, y1));
    return bin_edge(pi, 0u, r.x, r.y, r.z, r.w, need & 1u)
        && bin_edge(pi, 1u, r.x, r.y, r.z, r.w, (need >> 1u) & 1u)
        && bin_edge(pi, 2u, r.x, r.y, r.z, r.w, (need >> 2u) & 1u);
}
