// IMPACT's bin hook: a GL triangle goes to tiles only when some row of
// them within the triangle's rows has a span (rss.wgsl `triangle`) that
// can reach their columns. A span bound is monotonic in the row, so over a
// run of rows on one minor edge its extremes are at the run's ends: the
// tiles are rejected when the leftmost left bound is right of them or the
// rightmost right bound is not past their left column. The two extremes
// may come from different rows, so the test is conservative. Every other
// kind takes its box.

#include "rss_rec.wgsl"

// Whether rows ja..=jb, on minor edge `minor`, may draw in columns
// ia..=ib.
fn bin_rows(pi: u32, minor: u32, ja: i32, jb: i32, ia: i32, ib: i32) -> bool {
    if ja > jb {
        return false;
    }
    let ma = edge_bound(pi, 0u, ja);
    let mb = edge_bound(pi, 0u, jb);
    let na = edge_bound(pi, minor, ja);
    let nb = edge_bound(pi, minor, jb);
    let ltor = (recs[pi].flags & TRI_LTOR) != 0u;
    let left = select(min(na, nb), min(ma, mb), ltor);
    let right = select(max(ma, mb), max(na, nb), ltor);
    return left <= ib && right > ia;
}

fn bin_accept(pi: u32, tx0: u32, ty0: u32, tx1: u32, ty1: u32) -> bool {
    if recs[pi].kind != REC_TRI {
        return true;
    }
    let si = recs[pi].slot;
    let a = window_of(si, batch.origin_x + i32(tx0 * TILE_W), batch.origin_y + i32(ty0 * TILE_H));
    let b = window_of(si, batch.origin_x + i32(tx1 * TILE_W + TILE_W - 1u), batch.origin_y + i32(ty1 * TILE_H + TILE_H - 1u));
    let ja = max(min(a.y, b.y), ri(pi, T_J0));
    let jb = min(max(a.y, b.y), ri(pi, T_J1) - 1);
    let jmid = ri(pi, T_JMID);
    return bin_rows(pi, 2u, ja, min(jb, jmid - 1), a.x, b.x)
        || bin_rows(pi, 1u, max(ja, jmid), jb, a.x, b.x);
}
