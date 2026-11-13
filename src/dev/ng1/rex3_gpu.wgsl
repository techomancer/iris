// REX3 flat fills, binned into 32x32 screen tiles on the CPU.
//
// One workgroup owns one tile and each invocation owns four pixels of it, so
// no two invocations ever touch the same word: no barriers are needed, and
// each invocation applies its tile's fills in submission order with the
// pixels held in registers. `bins` holds TILE_COUNT + 1 offsets into itself,
// then the fill indices of every tile in order.
//
// `rgb` is tile-major: tile t's 1,024 pixels start at t * 1024, row by row.

struct FlatFill {
    min_x: u32,
    min_y: u32,
    width: u32,
    height: u32,
    color: u32,
    write_mask: u32,
    stride: u32,
    plane_len: u32,
}

@group(0) @binding(0)
var<storage, read_write> rgb: array<u32>;

@group(0) @binding(1)
var<storage, read> fills: array<FlatFill>;

@group(0) @binding(2)
var<storage, read> bins: array<u32>;

const TILE: u32 = 32u;
const TILES_X: u32 = 42u;
const TILE_PIXELS: u32 = 1024u;
const ROW_STEP: u32 = 8u;
// Four rows of 32 pixels apart: the next pixel this invocation owns.
const STEP: u32 = 256u;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let tile = group.x;
    let begin = bins[tile];
    let end = bins[tile + 1u];
    if begin == end {
        return;
    }
    let x = (tile % TILES_X) * TILE + lane % TILE;
    let y0 = (tile / TILES_X) * TILE + lane / TILE;
    let base = tile * TILE_PIXELS + lane;

    var p0 = rgb[base];
    var p1 = rgb[base + STEP];
    var p2 = rgb[base + 2u * STEP];
    var p3 = rgb[base + 3u * STEP];
    for (var i = begin; i < end; i += 1u) {
        let fill = fills[bins[i]];
        // Unsigned wrap: x below min_x becomes huge and fails the test.
        if x - fill.min_x >= fill.width {
            continue;
        }
        let keep = ~fill.write_mask;
        let value = fill.color & fill.write_mask;
        if y0 - fill.min_y < fill.height {
            p0 = (p0 & keep) | value;
        }
        if y0 + ROW_STEP - fill.min_y < fill.height {
            p1 = (p1 & keep) | value;
        }
        if y0 + 2u * ROW_STEP - fill.min_y < fill.height {
            p2 = (p2 & keep) | value;
        }
        if y0 + 3u * ROW_STEP - fill.min_y < fill.height {
            p3 = (p3 & keep) | value;
        }
    }
    rgb[base] = p0;
    rgb[base + STEP] = p1;
    rgb[base + 2u * STEP] = p2;
    rgb[base + 3u * STEP] = p3;
}
