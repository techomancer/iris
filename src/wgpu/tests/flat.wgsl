// The test board: flat-coloured triangles with a write mask into a
// plane of u32 pixels, STRIDE words a row, WIDTH x HEIGHT visible.
// Coordinates are 1/16 pixel; the sample is the pixel centre; the fill
// rule is top-left. src/wgpu/tests/mod.rs has the same arithmetic in Rust.

struct Rec {
    x0: i32, y0: i32,
    x1: i32, y1: i32,
    x2: i32, y2: i32,
    colour: u32,
    slot: u32,
}

struct Slot {
    mask: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(1) @binding(0) var<storage, read> recs: array<Rec>;
@group(1) @binding(1) var<storage, read> slots: array<Slot>;
@group(2) @binding(0) var<storage, read_write> vram: array<u32>;

var<private> addr: u32;
var<private> pixel: u32;
var<private> in_target: bool;

fn edge(ax: i32, ay: i32, bx: i32, by: i32, px: i32, py: i32) -> i32 {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

// Top-left rule for a positively oriented triangle: a top edge runs
// right with no rise, a left edge runs up.
fn inside(e: i32, dx: i32, dy: i32) -> bool {
    return e > 0 || (e == 0 && ((dy == 0 && dx > 0) || dy < 0));
}

fn arch_tile_begin(tile: u32, px: i32, py: i32, lane: u32) {
    in_target = px >= 0 && py >= 0 && u32(px) < WIDTH && u32(py) < HEIGHT;
    addr = u32(py) * STRIDE + u32(px);
    pixel = select(0u, vram[addr], in_target);
}

fn arch_prim(p: u32, px: i32, py: i32, lane: u32) {
    let r = recs[p];
    let sx = px * 16 + 8;
    let sy = py * 16 + 8;
    let e0 = edge(r.x1, r.y1, r.x2, r.y2, sx, sy);
    let e1 = edge(r.x2, r.y2, r.x0, r.y0, sx, sy);
    let e2 = edge(r.x0, r.y0, r.x1, r.y1, sx, sy);
    let covered = inside(e0, r.x2 - r.x1, r.y2 - r.y1)
        && inside(e1, r.x0 - r.x2, r.y0 - r.y2)
        && inside(e2, r.x1 - r.x0, r.y1 - r.y0);
    if covered {
        let m = slots[r.slot].mask;
        pixel = (pixel & ~m) | (r.colour & m);
    }
}

fn arch_tile_end(tile: u32, px: i32, py: i32, lane: u32) {
    if in_target {
        vram[addr] = pixel;
    }
}
