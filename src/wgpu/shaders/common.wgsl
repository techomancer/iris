// Batch layout shared by the bin and raster kernels. src/wgpu/tiles.rs
// defines the same structs; TILE_W, TILE_H, MASK_WORDS, COARSE, LIST_CAP
// and MAX_GROUPS arrive as defines.

struct BatchHeader {
    prim_count: u32,
    tile_cols: u32,
    tile_rows: u32,
    // The bin pass's grid: coarse tiles from (coarse_x0, coarse_y0),
    // coarse_cols a row.
    coarse_cols: u32,
    origin_x: i32,
    origin_y: i32,
    coarse_x0: u32,
    coarse_y0: u32,
}

// The busy count, the list entries used, and per busy tile: the tile
// index, its first entry in the list, its count.
struct BusyList {
    count: atomic<u32>,
    pairs: atomic<u32>,
    pad1: u32,
    pad2: u32,
    tiles: array<vec4<u32>>,
}

@group(0) @binding(0) var<uniform> batch: BatchHeader;
// Binding 1, the list of each busy tile's primitives (indices, a run per
// tile in primitive order), is declared by each kernel: written by the bin
// pass, read by the raster pass.
@group(0) @binding(2) var<storage, read_write> busy: BusyList;
// Per primitive, its inclusive tile box: x0, y0, x1, y1.
@group(0) @binding(4) var<storage, read> prim_boxes: array<vec4<u32>>;

fn tile_index(tx: u32, ty: u32) -> u32 {
    return ty * batch.tile_cols + tx;
}
