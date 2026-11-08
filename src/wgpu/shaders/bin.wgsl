// The bin pass: one workgroup per primitive, lanes over the rows of its
// tile box. A lane sets the primitive's bit in each tile's mask and puts
// a tile it is the first to touch on the busy list. Nothing here depends
// on order: the bit position is the primitive index.
//
// `bin_accept(p, tx, ty)` is the board's hook: true when primitive p may
// cover tile (tx, ty). The default accepts the whole box; a board can
// test its edge functions at the tile's corners.

#include "common.wgsl"

// The raster pass's indirect dispatch arguments, bound only here: a
// buffer cannot be read-write storage and the indirect source in one
// dispatch. x and y are folded so the busy count fits under MAX_GROUPS.
struct DispatchArgs {
    x: atomic<u32>,
    y: atomic<u32>,
    z: u32,
    pad: u32,
}
@group(0) @binding(5) var<storage, read_write> args: DispatchArgs;

@compute @workgroup_size(64)
fn bin(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let p = wg.x;
    if p >= batch.prim_count {
        return;
    }
    let b = prim_boxes[p];
    let word = p / 32u;
    let bit = 1u << (p % 32u);
    for (var ty = b.y + lane; ty <= b.w; ty += 64u) {
        for (var tx = b.x; tx <= b.z; tx++) {
            if !bin_accept(p, tx, ty) {
                continue;
            }
            let t = tile_index(tx, ty);
            // The first bit in a word marks the word in the tile's summary.
            if atomicOr(&masks[t * MASK_STRIDE + word], bit) == 0u {
                atomicOr(&masks[t * MASK_STRIDE + MASK_WORDS + word / 32u], 1u << (word % 32u));
            }
            if atomicExchange(&busy_flags[t], 1u) == 0u {
                let i = atomicAdd(&busy.count, 1u);
                busy.tiles[i] = t;
                atomicMax(&args.x, min(i + 1u, MAX_GROUPS));
                atomicMax(&args.y, i / MAX_GROUPS + 1u);
            }
        }
    }
}
