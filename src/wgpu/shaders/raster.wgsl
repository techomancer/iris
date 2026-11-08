// The raster pass driver: one 64-lane workgroup per busy tile, one pixel
// a lane, dispatched indirectly from the busy list. The workgroup is the
// only writer of its tile for the batch: it walks the tile's mask in
// primitive order and calls the board's hooks, which store pixels
// themselves. On the way it clears the mask words and the busy flag, so
// the next batch starts clean without a buffer clear.
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
// workgroup-uniform control flow (the tile and the mask words are read
// with workgroupUniformLoad), so they may use barriers and the quad
// exchanges as long as they keep every lane in the same branches and
// gate only their stores on coverage.

#include "common.wgsl"

var<workgroup> wg_tile: u32;
var<workgroup> wg_summary: u32;
var<workgroup> wg_bits: u32;

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
        wg_tile = select(NO_TILE, busy.tiles[idx], idx < atomicLoad(&busy.count));
    }
    let tile = workgroupUniformLoad(&wg_tile);
    if tile == NO_TILE {
        return;
    }
    let tx = tile % batch.tile_cols;
    let ty = tile / batch.tile_cols;
    let px = batch.origin_x + i32(tx * TILE_W + lane_x(lane));
    let py = batch.origin_y + i32(ty * TILE_H + lane_y(lane));

    arch_tile_begin(tile, px, py, lane);
    let base = tile * MASK_STRIDE;
    // The summary's words that cover the batch, then in each the mask
    // words that have a bit: the rest stay clear.
    let words = (batch.prim_count + 31u) / 32u;
    for (var g = 0u; g < (words + 31u) / 32u; g++) {
        if lane == 0u {
            wg_summary = atomicExchange(&masks[base + MASK_WORDS + g], 0u);
        }
        var summary = workgroupUniformLoad(&wg_summary);
        while summary != 0u {
            let w = g * 32u + firstTrailingBit(summary);
            summary &= summary - 1u;
            if lane == 0u {
                wg_bits = atomicExchange(&masks[base + w], 0u);
            }
            var bits = workgroupUniformLoad(&wg_bits);
            while bits != 0u {
                let p = w * 32u + firstTrailingBit(bits);
                bits &= bits - 1u;
                arch_prim(p, px, py, lane);
            }
        }
    }
    arch_tile_end(tile, px, py, lane);
    if lane == 0u {
        atomicStore(&busy_flags[tile], 0u);
    }
}
