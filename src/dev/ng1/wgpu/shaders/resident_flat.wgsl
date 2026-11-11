// One lane owns one linear pixel throughout a tile's ordered fills.
struct FillRecord {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    slot: u32,
}

struct FillSlot {
    color: u32,
    mask: u32,
    plane: u32,
    pad: u32,
}

@group(1) @binding(0) var<storage, read> fills: array<FillRecord>;
@group(1) @binding(1) var<storage, read> states: array<FillSlot>;
@group(2) @binding(0) var<storage, read_write> rgb: array<u32>;
@group(2) @binding(1) var<storage, read_write> aux: array<u32>;

var<private> address: u32;
var<private> rgb_value: u32;
var<private> aux_value: u32;
var<private> in_bounds: bool;
var<private> rgb_written: bool;
var<private> aux_written: bool;

fn arch_tile_begin(tile: u32, px: i32, py: i32, lane: u32) {
    in_bounds = px >= 0 && py >= 0 && u32(px) < WRITE_WIDTH && u32(py) < PLANE_HEIGHT;
    rgb_written = false;
    aux_written = false;
    if in_bounds {
        address = u32(py) * PLANE_STRIDE + u32(px);
        rgb_value = rgb[address];
        aux_value = aux[address];
    }
}

fn arch_prim(p: u32, px: i32, py: i32, lane: u32) {
    let fill = fills[p];
    if in_bounds && u32(px) >= fill.min_x && u32(px) <= fill.max_x
        && u32(py) >= fill.min_y && u32(py) <= fill.max_y {
        let state = states[fill.slot];
        if state.plane == 0u {
            rgb_value = (rgb_value & ~state.mask) | (state.color & state.mask);
            rgb_written = true;
        } else {
            aux_value = (aux_value & ~state.mask) | (state.color & state.mask);
            aux_written = true;
        }
    }
}

fn arch_tile_end(tile: u32, px: i32, py: i32, lane: u32) {
    if rgb_written {
        rgb[address] = rgb_value;
    }
    if aux_written {
        aux[address] = aux_value;
    }
}
