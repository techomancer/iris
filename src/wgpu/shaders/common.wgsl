// Batch layout shared by the bin and raster kernels. src/wgpu/tiles.rs
// defines the same structs; TILE_W, TILE_H, MASK_WORDS, MASK_STRIDE and MAX_GROUPS
// arrive as defines.

struct BatchHeader {
    prim_count: u32,
    tile_cols: u32,
    tile_rows: u32,
    mask_words: u32,
    origin_x: i32,
    origin_y: i32,
    pad0: u32,
    pad1: u32,
}

// The busy count and the busy tiles.
struct BusyList {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    tiles: array<u32>,
}

@group(0) @binding(0) var<uniform> batch: BatchHeader;
// MASK_STRIDE words per tile: MASK_WORDS words with one bit per primitive
// of the batch, then the summary, one bit per mask word that has any.
@group(0) @binding(1) var<storage, read_write> masks: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> busy: BusyList;
// One word per tile: set when the tile is on the busy list.
@group(0) @binding(3) var<storage, read_write> busy_flags: array<atomic<u32>>;
// Per primitive, its inclusive tile box: x0, y0, x1, y1.
@group(0) @binding(4) var<storage, read> prim_boxes: array<vec4<u32>>;

fn tile_index(tx: u32, ty: u32) -> u32 {
    return ty * batch.tile_cols + tx;
}
