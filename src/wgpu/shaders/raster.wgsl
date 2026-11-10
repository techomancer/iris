// The raster pass driver: one 64-lane workgroup per busy tile, one pixel
// a lane, dispatched indirectly from the busy list. The workgroup is the
// only writer of its tile for the batch: it walks the tile's run of the
// list (bin.wgsl), which holds its primitives in order, and calls the
// board's hooks, which store pixels themselves. Nothing is left to clear:
// the next batch's bin pass writes its runs and entries afresh.
//
// Lanes take the tile's pixels a quad at a time: lane bits 0 and 1 are
// the pixel's x and y within its 2x2 quad (even tile coordinates, and
// tiles start at even target coordinates), bits 2 and 3 the quad's x and
// bits 4 and 5 its y. The four lanes of a quad exchange values with
// quad_swap_x and quad_swap_diagonal (quad_subgroup.wgsl or
// quad_shared.wgsl), the way a GPU's quads share derivatives.
//
// The board provides:
//   fn arch_tile_begin(tile: u32, px: i32, py: i32, lane: u32)
//   fn arch_prim(p: u32, px: i32, py: i32, lane: u32)
//   fn arch_tile_end(tile: u32, px: i32, py: i32, lane: u32)
// (px, py) is the lane's pixel in target coordinates. Hooks run in
// workgroup-uniform control flow (the busy entry is read with
// workgroupUniformLoad, the run from read-only storage), so they may use
// barriers and the quad exchanges as long as they keep every lane in the
// same branches and gate only their stores on coverage.

#include "common.wgsl"

// The busy tiles' primitives, a run per tile (bin.wgsl).
@group(0) @binding(1) var<storage, read> tile_prims: array<u32>;

// The workgroup's busy entry: tile, first, count.
var<workgroup> wg_work: vec4<u32>;

const NO_TILE: u32 = 0xffffffffu;

// The lane's index in the tile, for the quad exchanges.
var<private> raster_lane: u32;

// The tile pixel lane `lane` takes: x from lane bits 0, 2 and 3, y from
// bits 1, 4 and 5.
fn lane_x(lane: u32) -> u32 {
    return (lane & 1u) | ((lane >> 1u) & 6u);
}

fn lane_y(lane: u32) -> u32 {
    return ((lane >> 1u) & 1u) | ((lane >> 3u) & 6u);
}

// The tile of workgroup wg (the entry point's) for lane `lane`.
fn raster_tile(wg: vec3<u32>, lane: u32) {
    raster_lane = lane;
    if lane == 0u {
        let idx = wg.x + wg.y * MAX_GROUPS;
        wg_work = select(vec4<u32>(NO_TILE, 0u, 0u, 0u), busy.tiles[idx], idx < atomicLoad(&busy.count));
    }
    let work = workgroupUniformLoad(&wg_work);
    let tile = work.x;
    if tile == NO_TILE {
        return;
    }
    let tx = tile % batch.tile_cols;
    let ty = tile / batch.tile_cols;
    let px = batch.origin_x + i32(tx * TILE_W + lane_x(lane));
    let py = batch.origin_y + i32(ty * TILE_H + lane_y(lane));

    arch_tile_begin(tile, px, py, lane);
    let end = min(work.y + work.z, LIST_CAP);
    for (var i = work.y; i < end; i++) {
        arch_prim(tile_prims[i], px, py, lane);
    }
    arch_tile_end(tile, px, py, lane);
}
