// The bin pass: one workgroup per coarse tile (COARSE x COARSE tiles, 32x32
// pixels) of the rectangle the batch's boxes cover. It finds the batch's
// primitives whose boxes touch the coarse tile, then for each of its 16
// tiles the ones that touch that tile, and writes each busy tile's
// primitives as one run of indices, in primitive order, in the shared list,
// with the tile's busy entry (tile, first, count). The masks it works from
// live in workgroup memory: no global atomics but one allocation of list
// entries and one of busy entries a coarse tile, and nothing to clear for
// the next batch.
//
// `bin_accept(p, tx0, ty0, tx1, ty1)` is the board's hook: true when
// primitive p may cover a pixel of tiles tx0..=tx1, ty0..=ty1, whose
// rectangle its box overlaps. The default accepts the whole box; a board
// can test its edge functions at the rectangle's corners. The coarse pass
// asks it of the coarse tile, the fine pass of each tile.

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
@group(0) @binding(1) var<storage, read_write> tile_prims: array<u32>;

// Tiles of a coarse tile, and lanes per tile in the fine pass: each takes a
// quarter of the mask's words, in order.
const FINE: u32 = COARSE * COARSE;
const SPLIT: u32 = 64u / FINE;

// The primitives touching the coarse tile.
var<workgroup> wg_coarse: array<atomic<u32>, MASK_WORDS>;
// Per tile, the primitives touching it.
var<workgroup> wg_fine: array<u32, FINE * MASK_WORDS>;
// Per lane, the primitives its words hold.
var<workgroup> wg_count: array<u32, 64>;
// The coarse tile's first list entry and first busy entry.
var<workgroup> wg_first: u32;
var<workgroup> wg_slot: u32;

fn overlaps(b: vec4<u32>, x0: u32, y0: u32, x1: u32, y1: u32) -> bool {
    return b.x <= x1 && b.z >= x0 && b.y <= y1 && b.w >= y0;
}

// Primitives of tile g of the coarse tile.
fn tile_total(g: u32) -> u32 {
    var t = 0u;
    for (var r = 0u; r < SPLIT; r++) {
        t += wg_count[g + r * FINE];
    }
    return t;
}

@compute @workgroup_size(64)
fn bin(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let x0 = (batch.coarse_x0 + wg.x) * COARSE;
    let y0 = (batch.coarse_y0 + wg.y) * COARSE;
    let n = batch.prim_count;
    let words = (n + 31u) / 32u;

    // The coarse mask: the boxes that touch any of the 16 tiles.
    for (var w = lane; w < words; w += 64u) {
        atomicStore(&wg_coarse[w], 0u);
    }
    workgroupBarrier();
    for (var p = lane; p < n; p += 64u) {
        let cx1 = min(x0 + COARSE, batch.tile_cols) - 1u;
        let cy1 = min(y0 + COARSE, batch.tile_rows) - 1u;
        let b = prim_boxes[p];
        if overlaps(b, x0, y0, cx1, cy1) && bin_accept(p, max(x0, b.x), max(y0, b.y), min(cx1, b.z), min(cy1, b.w)) {
            atomicOr(&wg_coarse[p / 32u], 1u << (p % 32u));
        }
    }
    workgroupBarrier();

    // The fine masks: lane = tile f of the coarse tile, and the quarter q
    // of the mask's words it tests.
    let f = lane % FINE;
    let q = lane / FINE;
    let tx = x0 + f % COARSE;
    let ty = y0 + f / COARSE;
    let inside = tx < batch.tile_cols && ty < batch.tile_rows;
    let lo = q * words / SPLIT;
    let hi = (q + 1u) * words / SPLIT;
    var count = 0u;
    for (var w = lo; w < hi; w++) {
        var bits = select(0u, atomicLoad(&wg_coarse[w]), inside);
        var fine = 0u;
        while bits != 0u {
            let b = firstTrailingBit(bits);
            bits &= bits - 1u;
            let p = w * 32u + b;
            if overlaps(prim_boxes[p], tx, ty, tx, ty) && bin_accept(p, tx, ty, tx, ty) {
                fine |= 1u << b;
            }
        }
        wg_fine[f * MASK_WORDS + w] = fine;
        count += countOneBits(fine);
    }
    wg_count[lane] = count;
    workgroupBarrier();

    // One allocation of list entries and busy entries for the coarse tile.
    if lane == 0u {
        var pairs = 0u;
        var tiles = 0u;
        for (var g = 0u; g < FINE; g++) {
            let t = tile_total(g);
            pairs += t;
            tiles += select(0u, 1u, t != 0u);
        }
        var first = 0u;
        var slot = 0u;
        if pairs != 0u {
            first = atomicAdd(&busy.pairs, pairs);
            slot = atomicAdd(&busy.count, tiles);
            let end = slot + tiles;
            atomicMax(&args.x, min(end, MAX_GROUPS));
            atomicMax(&args.y, (end - 1u) / MAX_GROUPS + 1u);
        }
        wg_first = first;
        wg_slot = slot;
    }
    workgroupBarrier();

    // Tile f's run starts after the runs of the tiles before it; this
    // lane's words after those of the lanes before it on the same tile.
    var first = wg_first;
    var slot = wg_slot;
    var total = 0u;
    for (var g = 0u; g < FINE; g++) {
        let t = tile_total(g);
        if g < f {
            first += t;
            slot += select(0u, 1u, t != 0u);
        }
        if g == f {
            total = t;
        }
    }
    var at = first;
    for (var r = 0u; r < q; r++) {
        at += wg_count[f + r * FINE];
    }
    for (var w = lo; w < hi; w++) {
        var fine = wg_fine[f * MASK_WORDS + w];
        while fine != 0u {
            let b = firstTrailingBit(fine);
            fine &= fine - 1u;
            if at < LIST_CAP {
                tile_prims[at] = w * 32u + b;
            }
            at += 1u;
        }
    }
    if q == 0u && total != 0u {
        busy.tiles[slot] = vec4<u32>(tile_index(tx, ty), first, total, 0u);
    }
}
