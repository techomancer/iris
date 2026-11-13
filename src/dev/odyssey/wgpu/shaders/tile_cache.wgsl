// The tile cache: the raster kernel's word operations (registered as
// vram_mem.wgsl when the cache is on). The workgroup owns its 8x8 tile for
// the whole batch, so it can hold the tile's block of each surface its
// primitives draw to in workgroup memory, from the first primitive that
// names the surface to the end of the tile, and store the blocks it
// changed back once. Fragments then read and write workgroup memory, not
// board memory through the L2.
//
// A block. Tiles start at multiples of 8 in plane coordinates, and a
// surface keeps 16 x 16 pixel tiles row-major with the pixels row-major
// inside (views.wgsl), so a raster tile is a quarter of a memory tile:
// 8 rows of 8 pixels, each row one run of 8 << log2 bytes. The block holds
// the rows one after another, every field of the pixel (both colour
// buffers, depth and stencil, at their bytes), and its words keep board
// memory's layout, so vram.wgsl's field arithmetic works on it unchanged:
// a byte address with TC_FLAG set is a byte of the cache. Pixels of 1 and
// 2 bytes share words between lanes, so the cache's words are atomics too.
//
// The directory. Entries are keyed by the surface's base and geometry
// (pitch, pixel size), not the field: one block serves every view of the
// surface. Surfaces never overlap on the board, so two entries never hold
// the same bytes. Lane 0 decides at the start of each primitive which of
// its slot's surfaces (and a blit's own view) are missing and which entry
// each takes, evicting round the ring past the ones the primitive uses;
// the workgroup then stores the evicted blocks it changed and loads the new
// ones, each lane the same words of both, between barriers. A surface that
// does not fit (more than TC_ENTRIES in one primitive, pixels wider than
// 16 bytes, a base off a word, a tile outside the view) goes to board
// memory as before, for as long as it is not resident.
//
// Coherency. Nothing else reads or writes board memory while the raster
// pass runs: copies, readbacks and texture builds flush the batch first
// (GpuRaster::copy, read_back), and the end of the tile writes the blocks
// back. Texture reads (texture.wgsl) use plain addresses and go to board
// memory, as before.

const TC_FLAG: u32 = 0x80000000u;
const TC_ENTRIES: u32 = 4u;
// Pixels up to 16 bytes.
const TC_LOG2_MAX: u32 = 4u;
// u32 words an entry holds: 64 pixels of 16 bytes.
const TC_ENTRY_U32: u32 = 256u;
// Surfaces one primitive can name: its slot's six views and a blit's own.
const TC_OPS: u32 = 7u;
const TC_NONE: u32 = 0xffffffffu;

struct TcOp {
    e: u32,
    // The entry held another surface, to be stored back if changed.
    evict: u32,
    old_base: u32,
    old_geom: u32,
    base: u32,
    geom: u32,
}

var<workgroup> tc_data: array<atomic<u32>, 1024>;
var<workgroup> tc_dirty: array<atomic<u32>, TC_ENTRIES>;
var<workgroup> tc_base: array<u32, TC_ENTRIES>;
var<workgroup> tc_geom: array<u32, TC_ENTRIES>;
var<workgroup> tc_count: u32;
// Lane 0's: the next eviction candidate, the slot prepared last.
var<workgroup> tc_victim: u32;
var<workgroup> tc_last: u32;
// The plan lane 0 makes for a primitive.
var<workgroup> tc_ops: array<TcOp, TC_OPS>;
var<workgroup> tc_nops: u32;
var<workgroup> tc_new_count: u32;

// The tile (raster.wgsl's index). Its origin is derived from it when
// needed: a value from the tile index stays in scalar registers, while one
// derived from the lane's pixel would be a vector register held across all
// the fragment code (the kernel went 96 -> 128 VGPRs over three of them).
var<private> tc_tile: u32;

fn tc_x0() -> i32 {
    return batch.origin_x + i32((tc_tile % batch.tile_cols) * TILE_W);
}

fn tc_y0() -> i32 {
    return batch.origin_y + i32((tc_tile / batch.tile_cols) * TILE_H);
}

// The lane of tile pixel (x, y) (raster.wgsl lane_x, lane_y inverted).
// The hooks recompute it from the pixel, which is live anyway, instead of
// holding the lane index across the fragment code.
fn tc_lane(x: i32, y: i32) -> u32 {
    let ux = u32(x);
    let uy = u32(y);
    return (ux & 1u) | ((uy & 1u) << 1u) | ((ux & 6u) << 1u) | ((uy & 6u) << 3u);
}

fn vram_word(a: u32) -> u32 {
    if (a & TC_FLAG) != 0u {
        return TC_FLAG | (((a & ~TC_FLAG) >> 3u) * 2u);
    }
    return ((a >> 3u) & VRAM_WORD_MASK) * 2u;
}

fn mem_load(i: u32) -> u32 {
    if (i & TC_FLAG) != 0u {
        return atomicLoad(&tc_data[i & ~TC_FLAG]);
    }
    return atomicLoad(&vram[i]);
}

fn tc_touch(j: u32) {
    atomicStore(&tc_dirty[j / TC_ENTRY_U32], 1u);
}

fn mem_store(i: u32, v: u32) {
    if (i & TC_FLAG) != 0u {
        let j = i & ~TC_FLAG;
        atomicStore(&tc_data[j], v);
        tc_touch(j);
        return;
    }
    atomicStore(&vram[i], v);
}

fn mem_and(i: u32, m: u32) {
    if (i & TC_FLAG) != 0u {
        let j = i & ~TC_FLAG;
        atomicAnd(&tc_data[j], m);
        tc_touch(j);
        return;
    }
    atomicAnd(&vram[i], m);
}

fn mem_or(i: u32, m: u32) {
    if (i & TC_FLAG) != 0u {
        let j = i & ~TC_FLAG;
        atomicOr(&tc_data[j], m);
        tc_touch(j);
        return;
    }
    atomicOr(&vram[i], m);
}

// The key of a view's surface: its pitch and pixel size, not the field.
fn tc_geom_of(v: View) -> u32 {
    return v.geom & 0xffffffu;
}

fn tc_log2(geom: u32) -> u32 {
    return (geom >> 16u) & 0xffu;
}

// The block's u32 words: 8 rows of 8 << log2 bytes.
fn tc_block_words(geom: u32) -> u32 {
    return 16u << tc_log2(geom);
}

// The u32 index in board memory of the block's word k.
fn tc_mem_index(base: u32, geom: u32, k: u32) -> u32 {
    let log2 = tc_log2(geom);
    let per_row = 2u << log2;
    let r = k / per_row;
    let c = k % per_row;
    let x = u32(tc_x0());
    let y = u32(tc_y0()) + r;
    let tile = (y >> 4u) * (geom & 0xffffu) + (x >> 4u);
    let row = base + (((tile << 8u) | ((y & 15u) << 4u) | (x & 15u)) << log2);
    return (((row >> 3u) + (c >> 1u)) & VRAM_WORD_MASK) * 2u + (c & 1u);
}

fn tc_cacheable(v: View) -> bool {
    return v.kind != KIND_NONE && tc_log2(v.geom) <= TC_LOG2_MAX && (v.base & 7u) == 0u
        && u32(tc_x0()) < view_width(v) && u32(tc_y0()) < view_height(v);
}

// Where view v's field at (x, y), a pixel of this tile at byte address a,
// is held: in the cache when its surface is resident. The entries are
// compared unrolled (an entry not in use holds base TC_NONE, which no
// cacheable view has): a loop to tc_count at every field access cost 24
// VGPRs.
fn tile_addr(v: View, a: u32, x: i32, y: i32) -> u32 {
    let g = tc_geom_of(v);
    var e = TC_NONE;
    e = select(e, 3u, tc_base[3] == v.base && tc_geom[3] == g);
    e = select(e, 2u, tc_base[2] == v.base && tc_geom[2] == g);
    e = select(e, 1u, tc_base[1] == v.base && tc_geom[1] == g);
    e = select(e, 0u, tc_base[0] == v.base && tc_geom[0] == g);
    if e == TC_NONE {
        return a;
    }
    let o = ((u32(y & 7) * 8u + u32(x & 7)) << tc_log2(g)) + (v.geom >> 24u);
    return TC_FLAG | (e * TC_ENTRY_U32 * 4u + o);
}

fn tc_store_block(e: u32, base: u32, geom: u32, lane: u32) {
    let n = tc_block_words(geom);
    for (var k = lane; k < n; k += 64u) {
        atomicStore(&vram[tc_mem_index(base, geom, k)], atomicLoad(&tc_data[e * TC_ENTRY_U32 + k]));
    }
}

fn tc_load_block(e: u32, base: u32, geom: u32, lane: u32) {
    let n = tc_block_words(geom);
    for (var k = lane; k < n; k += 64u) {
        atomicStore(&tc_data[e * TC_ENTRY_U32 + k], atomicLoad(&vram[tc_mem_index(base, geom, k)]));
    }
}

fn tc_begin(tile: u32, px: i32, py: i32) {
    tc_tile = tile;
    if tc_lane(px, py) == 0u {
        tc_count = 0u;
        tc_victim = 0u;
        tc_last = TC_NONE;
        for (var e = 0u; e < TC_ENTRIES; e++) {
            tc_base[e] = TC_NONE;
        }
    }
}

// Lane 0: the surfaces primitive pi names that are not resident, into
// tc_ops. The directory itself changes only between the barriers of
// tc_prepare, as lanes may still be reading it for the last primitive.
fn tc_plan(pi: u32) {
    tc_nops = 0u;
    let own = recs[pi].kind == REC_BLIT && (recs[pi].flags & BLIT_OWN_VIEW) != 0u;
    let si = recs[pi].slot;
    if si == tc_last && !own {
        return;
    }
    var base: array<u32, TC_ENTRIES>;
    var geom: array<u32, TC_ENTRIES>;
    for (var e = 0u; e < TC_ENTRIES; e++) {
        base[e] = tc_base[e];
        geom[e] = tc_geom[e];
    }
    var count = tc_count;
    var victim = tc_victim;
    var wanted = 0u;
    var nops = 0u;
    for (var k = 0u; k < TC_OPS; k++) {
        var v: View;
        if k < 6u {
            v = slots[si].views[k];
        } else if own {
            v = View(ru(pi, B_OWN), ru(pi, B_OWN + 1u), ru(pi, B_OWN + 2u), ru(pi, B_OWN + 3u));
        } else {
            break;
        }
        if !tc_cacheable(v) {
            continue;
        }
        let g = tc_geom_of(v);
        var hit = TC_NONE;
        for (var e = 0u; e < count; e++) {
            if base[e] == v.base && geom[e] == g {
                hit = e;
            }
        }
        if hit != TC_NONE {
            wanted |= 1u << hit;
            continue;
        }
        var e = count;
        var evict = 0u;
        if count < TC_ENTRIES {
            count += 1u;
        } else {
            // The next entry round the ring that this primitive does not use;
            // none left, and the surface stays in board memory.
            e = TC_NONE;
            for (var i = 0u; i < TC_ENTRIES; i++) {
                let c = (victim + i) % TC_ENTRIES;
                if (wanted & (1u << c)) == 0u {
                    e = c;
                    break;
                }
            }
            if e == TC_NONE {
                continue;
            }
            victim = (e + 1u) % TC_ENTRIES;
            evict = 1u;
        }
        tc_ops[nops] = TcOp(e, evict, base[e], geom[e], v.base, g);
        nops += 1u;
        base[e] = v.base;
        geom[e] = g;
        wanted |= 1u << e;
    }
    tc_nops = nops;
    tc_new_count = count;
    tc_victim = victim;
    tc_last = select(si, TC_NONE, own);
}

// Make the surfaces of primitive pi resident; workgroup-uniform. (px, py)
// is the lane's pixel.
fn tc_prepare(pi: u32, px: i32, py: i32) {
    let lane = tc_lane(px, py);
    if lane == 0u {
        tc_plan(pi);
    }
    let n = workgroupUniformLoad(&tc_nops);
    if n == 0u {
        return;
    }
    // Board memory a lane stored directly (a surface not resident until
    // now) is read by other lanes below.
    storageBarrier();
    for (var i = 0u; i < n; i++) {
        let op = tc_ops[i];
        if op.evict != 0u && atomicLoad(&tc_dirty[op.e]) != 0u {
            tc_store_block(op.e, op.old_base, op.old_geom, lane);
        }
    }
    // Every lane has read the dirty flags before they are cleared, and
    // the directory changes while nobody reads it.
    workgroupBarrier();
    if lane == 0u {
        for (var i = 0u; i < n; i++) {
            tc_base[tc_ops[i].e] = tc_ops[i].base;
            tc_geom[tc_ops[i].e] = tc_ops[i].geom;
        }
        tc_count = tc_new_count;
    }
    for (var i = 0u; i < n; i++) {
        let op = tc_ops[i];
        atomicStore(&tc_dirty[op.e], 0u);
        tc_load_block(op.e, op.base, op.geom, lane);
    }
    workgroupBarrier();
}

// Store back the blocks the tile changed; workgroup-uniform.
fn tc_end(px: i32, py: i32) {
    let lane = tc_lane(px, py);
    let n = workgroupUniformLoad(&tc_count);
    for (var e = 0u; e < n; e++) {
        if atomicLoad(&tc_dirty[e]) != 0u {
            tc_store_block(e, tc_base[e], tc_geom[e], lane);
        }
    }
}
