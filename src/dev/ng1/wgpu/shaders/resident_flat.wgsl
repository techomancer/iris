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

fn high_i16(value: u32) -> i32 {
    return bitcast<i32>(value) >> 16;
}

fn low_i16(value: u32) -> i32 {
    return bitcast<i32>(value << 16u) >> 16;
}

fn inside_rect(x: i32, y: i32, sx: u32, sy: u32) -> bool {
    return x >= high_i16(sx) && x <= low_i16(sx)
        && y >= high_i16(sy) && y <= low_i16(sy);
}

fn clip_allows(state: RexStateSlot, px: i32, py: i32) -> bool {
    let clipmode = state.words[REX_STATE_CLIP_MODE];
    let enabled = clipmode & 0x1fu;
    let xywin = state.words[REX_STATE_XYWIN];
    let x_abs = px + 4096;
    let y_abs = py + 4096;
    let x_curr = x_abs - high_i16(xywin);
    let y_curr = y_abs - low_i16(xywin);
    if (enabled & 1u) != 0u
        && !inside_rect(
            x_curr,
            y_curr,
            state.words[REX_STATE_SMASK0X],
            state.words[REX_STATE_SMASK0Y],
        ) {
        return false;
    }

    if (enabled & 0x1eu) != 0u {
        var inside_any = false;
        for (var index = 0u; index < 4u; index++) {
            if (enabled & (1u << (index + 1u))) != 0u {
                let offset = REX_STATE_SMASK1X + index * 2u;
                inside_any = inside_any || inside_rect(
                    x_abs,
                    y_abs,
                    state.words[offset],
                    state.words[offset + 1u],
                );
            }
        }
        if !inside_any {
            return false;
        }
    }
    return true;
}

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
    let line = command.kind == REX_COMMAND_I_LINE
        || command.kind == REX_COMMAND_F_LINE
        || command.kind == REX_COMMAND_A_LINE;
    var covered = fill;
    if line {
        let y_major = (command.flags & REX_FLAG_LINE_Y_MAJOR) != 0u;
        let decreasing = (command.flags & REX_FLAG_LINE_MAJOR_DECREASING) != 0u;
        let major = select(px, py, y_major);
        let delta = select(major - command.src_x0, command.src_x0 - major, decreasing);
        if delta >= 0 && u32(delta) < command.payload_words {
            let minor = select(py, px, y_major);
            let expected = bitcast<i32>(payload[command.payload_offset + u32(delta)]);
            covered = minor == expected;
        }
    }
    if covered && in_bounds
        && px >= command.dst_x0 && px <= command.dst_x1
        && py >= command.dst_y0 && py <= command.dst_y1 {
        let state = states[command.state_slot];
        let cid_mask = (state.words[REX_STATE_CLIP_MODE] >> 9u) & 0xfu;
        let cid_allows = (command.flags & REX_FLAG_READ_AUX) == 0u
            || (cid_mask & (1u << (aux_value & 3u))) != 0u;
        if !clip_allows(state, px, py) || !cid_allows {
            return;
        }
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
