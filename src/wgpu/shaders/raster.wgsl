// The raster pass driver: one 64-lane workgroup per busy tile, one pixel
// a lane, dispatched indirectly from the busy list. The workgroup is the
// only writer of its tile for the batch: it walks the tile's mask in
// primitive order and calls the board's hooks, which store pixels
// themselves. On the way it clears the mask words and the busy flag, so
// the next batch starts clean without a buffer clear.
//
// The board provides:
//   fn arch_tile_begin(tile: u32, px: i32, py: i32, lane: u32)
//   fn arch_prim(p: u32, px: i32, py: i32, lane: u32)
//   fn arch_tile_end(tile: u32, px: i32, py: i32, lane: u32)
// (px, py) is the lane's pixel in target coordinates. Hooks run in
// workgroup-uniform control flow (the tile and the mask words are read
// with workgroupUniformLoad), so they may use barriers and subgroup
// operations as long as they keep every lane in the same branches and
// gate only their stores on coverage.

#include "common.wgsl"

var<workgroup> wg_tile: u32;
var<workgroup> wg_bits: u32;

const NO_TILE: u32 = 0xffffffffu;

@compute @workgroup_size(64)
fn raster(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
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
    let px = batch.origin_x + i32(tx * TILE_W + lane % TILE_W);
    let py = batch.origin_y + i32(ty * TILE_H + lane / TILE_W);

    arch_tile_begin(tile, px, py, lane);
    let base = tile * MASK_WORDS;
    for (var w = 0u; w < MASK_WORDS; w++) {
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
    arch_tile_end(tile, px, py, lane);
    if lane == 0u {
        atomicStore(&busy_flags[tile], 0u);
    }
}
