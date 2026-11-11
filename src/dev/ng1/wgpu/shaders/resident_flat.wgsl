// One lane owns one linear pixel throughout a tile's ordered commands.
struct RexCommand {
    kind: u32,
    state_slot: u32,
    flags: u32,
    payload_offset: u32,
    payload_words: u32,
    dst_x0: i32,
    dst_y0: i32,
    dst_x1: i32,
    dst_y1: i32,
    src_x0: i32,
    src_y0: i32,
    src_x1: i32,
    src_y1: i32,
    data: array<u32, 3>,
}

struct RexStateSlot {
    words: array<u32, REX_STATE_WORDS>,
}

@group(1) @binding(0) var<storage, read> commands: array<RexCommand>;
@group(1) @binding(1) var<storage, read> states: array<RexStateSlot>;
@group(2) @binding(0) var<storage, read_write> rgb: array<u32>;
@group(2) @binding(1) var<storage, read_write> aux: array<u32>;
@group(2) @binding(2) var<storage, read> payload: array<u32>;

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
    let command = commands[p];
    let fill = command.kind == REX_COMMAND_BLOCK
        || command.kind == REX_COMMAND_SPAN
        || command.kind == REX_COMMAND_CLEAR;
    if fill && in_bounds
        && px >= command.dst_x0 && px <= command.dst_x1
        && py >= command.dst_y0 && py <= command.dst_y1 {
        let state = states[command.state_slot];
        let mask = state.words[REX_STATE_WRITE_MASK];
        let color = state.words[REX_STATE_SOURCE_COLOR];
        if (command.flags & REX_FLAG_WRITE_RGB) != 0u {
            rgb_value = (rgb_value & ~mask) | (color & mask);
            rgb_written = true;
        }
        if (command.flags & REX_FLAG_WRITE_AUX) != 0u {
            aux_value = (aux_value & ~mask) | (color & mask);
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
