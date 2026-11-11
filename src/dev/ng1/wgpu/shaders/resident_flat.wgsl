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

const REX_COORD_BIAS: i32 = 0x1000;
const REX_YFLIP_BASE: i32 = 0x23ff;

const REX_PLANES_RGB: u32 = 1u;
const REX_PLANES_RGBA: u32 = 2u;
const REX_PLANES_OLAY: u32 = 4u;
const REX_PLANES_PUP: u32 = 5u;
const REX_PLANES_CID: u32 = 6u;

const REX_DRAWDEPTH_4: u32 = 0u;
const REX_DRAWDEPTH_8: u32 = 1u;
const REX_DRAWDEPTH_12: u32 = 2u;
const REX_LINE_BACKGROUND: u32 = 0x80000000u;
const REX_LINE_SKIPPED: u32 = 0xffffffffu;
const REX_BAYER: array<u32, 16> = array<u32, 16>(
    0u, 8u, 2u, 10u,
    12u, 4u, 14u, 6u,
    3u, 11u, 1u, 9u,
    15u, 7u, 13u, 5u,
);

struct ShadeDda {
    red: u32,
    alpha: u32,
    green: u32,
    blue: u32,
}

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

fn plane(state: RexStateSlot) -> u32 {
    return state.words[REX_STATE_DRAW_MODE1] & 0x7u;
}

fn draw_depth(state: RexStateSlot) -> u32 {
    return (state.words[REX_STATE_DRAW_MODE1] >> 3u) & 0x3u;
}

fn read_plane(state: RexStateSlot, value: u32) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    let selected_plane = mode & 0x7u;
    let depth = (mode >> 3u) & 0x3u;
    let double_source = ((mode >> 5u) & 1u) != 0u;

    switch selected_plane {
        case REX_PLANES_RGB, REX_PLANES_RGBA: {
            switch depth {
                case REX_DRAWDEPTH_4: {
                    return (value >> select(0u, 4u, double_source)) & 0xfu;
                }
                case REX_DRAWDEPTH_8: {
                    return (value >> select(0u, 8u, double_source)) & 0xffu;
                }
                case REX_DRAWDEPTH_12: {
                    return (value >> select(0u, 12u, double_source)) & 0xfffu;
                }
                default: {
                    return value & 0x00ffffffu;
                }
            }
        }
        case REX_PLANES_OLAY: {
            return (value >> select(8u, 16u, double_source)) & 0xffu;
        }
        case REX_PLANES_PUP: {
            return (value >> select(2u, 6u, double_source)) & 0x3u;
        }
        case REX_PLANES_CID: {
            return (value >> select(0u, 4u, double_source)) & 0x3u;
        }
        default: {
            return 0u;
        }
    }
}

fn dither_channel(value: u32, shift: u32, maximum: u32, threshold: u32) -> u32 {
    let scaled = (value >> shift) - (value >> 4u);
    var result = (scaled >> 4u) & maximum;
    if (scaled & 0xfu) > threshold {
        result = min(result + 1u, maximum);
    }
    return result;
}

fn walker_position(state: RexStateSlot, px: i32, py: i32) -> vec2<i32> {
    let xywin = state.words[REX_STATE_XYWIN];
    let xymove = state.words[REX_STATE_XYMOVE];
    let xy_offset = ((state.words[REX_STATE_DRAW_MODE0] >> 20u) & 1u) != 0u;
    let y_flip = ((state.words[REX_STATE_DRAW_MODE1] >> 6u) & 1u) != 0u;
    let absolute_y = select(py + REX_COORD_BIAS, REX_YFLIP_BASE - py, y_flip);
    return vec2<i32>(
        px + REX_COORD_BIAS - high_i16(xywin) - select(0, high_i16(xymove), xy_offset),
        absolute_y - low_i16(xywin) - select(0, low_i16(xymove), xy_offset),
    );
}

fn compress_source(
    state: RexStateSlot,
    source: u32,
    px: i32,
    py: i32,
) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    if ((mode >> 15u) & 1u) == 0u {
        return source;
    }
    let dither = ((mode >> 16u) & 1u) != 0u;
    let walker = walker_position(state, px, py);
    let bayer_index = ((u32(walker.y) & 3u) << 2u) | (u32(walker.x) & 3u);
    let threshold = REX_BAYER[bayer_index];
    let red = source & 0xffu;
    let green = (source >> 8u) & 0xffu;
    let blue = (source >> 16u) & 0xffu;
    switch (mode >> 3u) & 0x3u {
        case REX_DRAWDEPTH_4: {
            if dither {
                let dr = dither_channel(red, 3u, 1u, threshold);
                let dg = dither_channel(green, 2u, 3u, threshold);
                let db = dither_channel(blue, 3u, 1u, threshold);
                return (db << 3u) | (dg << 1u) | dr;
            }
            return ((blue >> 7u) << 3u) | ((green >> 6u) << 1u) | (red >> 7u);
        }
        case REX_DRAWDEPTH_8: {
            if dither {
                let dr = dither_channel(red, 1u, 7u, threshold);
                let dg = dither_channel(green, 1u, 7u, threshold);
                let db = dither_channel(blue, 2u, 3u, threshold);
                return (db << 6u) | (dg << 3u) | dr;
            }
            return ((blue >> 6u) << 6u) | ((green >> 5u) << 3u) | (red >> 5u);
        }
        case REX_DRAWDEPTH_12: {
            if dither {
                let dr = dither_channel(red, 0u, 0xfu, threshold);
                let dg = dither_channel(green, 0u, 0xfu, threshold);
                let db = dither_channel(blue, 0u, 0xfu, threshold);
                return (db << 8u) | (dg << 4u) | dr;
            }
            return ((blue >> 4u) << 8u) | ((green >> 4u) << 4u) | (red >> 4u);
        }
        default: {
            return source;
        }
    }
}

fn amplify(state: RexStateSlot, value: u32) -> u32 {
    switch plane(state) {
        case REX_PLANES_RGB, REX_PLANES_RGBA: {
            switch draw_depth(state) {
                case REX_DRAWDEPTH_4: {
                    return value | (value << 4u);
                }
                case REX_DRAWDEPTH_8: {
                    return value | (value << 8u);
                }
                case REX_DRAWDEPTH_12: {
                    return value | (value << 12u);
                }
                default: {
                    return value;
                }
            }
        }
        case REX_PLANES_OLAY: {
            return (value << 8u) | (value << 16u);
        }
        case REX_PLANES_PUP: {
            return (value << 2u) | (value << 6u);
        }
        case REX_PLANES_CID: {
            return value | (value << 4u);
        }
        default: {
            return 0u;
        }
    }
}

fn expand_destination(state: RexStateSlot, value: u32) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    if ((mode >> 15u) & 1u) == 0u {
        return value;
    }
    switch (mode >> 3u) & 0x3u {
        case REX_DRAWDEPTH_4: {
            let red = select(0u, 0xffu, (value & 1u) != 0u);
            let green_raw = (value >> 1u) & 3u;
            let green = (green_raw << 6u) | (green_raw << 4u)
                | (green_raw << 2u) | green_raw;
            let blue = select(0u, 0xffu, (value & 8u) != 0u);
            return (blue << 16u) | (green << 8u) | red;
        }
        case REX_DRAWDEPTH_8: {
            let red_raw = value & 7u;
            let red = (red_raw << 5u) | (red_raw << 2u) | (red_raw >> 1u);
            let green_raw = (value >> 3u) & 7u;
            let green = (green_raw << 5u) | (green_raw << 2u) | (green_raw >> 1u);
            let blue_raw = (value >> 6u) & 3u;
            let blue = (blue_raw << 6u) | (blue_raw << 4u)
                | (blue_raw << 2u) | blue_raw;
            return (blue << 16u) | (green << 8u) | red;
        }
        case REX_DRAWDEPTH_12: {
            let red_raw = value & 0xfu;
            let red = (red_raw << 4u) | red_raw;
            let green_raw = (value >> 4u) & 0xfu;
            let green = (green_raw << 4u) | green_raw;
            let blue_raw = (value >> 8u) & 0xfu;
            let blue = (blue_raw << 4u) | blue_raw;
            return (blue << 16u) | (green << 8u) | red;
        }
        default: {
            return value;
        }
    }
}

fn blend_factor(selector: u32, other: u32, source_alpha: u32) -> u32 {
    switch selector {
        case 0u: {
            return 0u;
        }
        case 1u: {
            return 0xffu;
        }
        case 2u: {
            return other;
        }
        case 3u: {
            return 0xffu - other;
        }
        case 4u: {
            return source_alpha;
        }
        case 5u: {
            return 0xffu - source_alpha;
        }
        default: {
            return 0u;
        }
    }
}

fn blend_colors(state: RexStateSlot, source: u32, destination: u32) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    let source_factor = (mode >> 19u) & 0x7u;
    let destination_factor = (mode >> 22u) & 0x7u;
    let source_alpha = (source >> 24u) & 0xffu;
    let blend_alpha = ((mode >> 27u) & 1u) != 0u;
    var result = 0u;
    for (var channel = 0u; channel < 4u; channel++) {
        let shift = channel * 8u;
        let source_component = (source >> shift) & 0xffu;
        let destination_component = (destination >> shift) & 0xffu;
        let alpha_factor = select(0xffu, source_alpha, channel != 3u || blend_alpha);
        let sf = blend_factor(source_factor, destination_component, alpha_factor);
        let df = blend_factor(destination_factor, source_component, source_alpha);
        let component = min(
            (source_component * sf + destination_component * df) / 0xffu,
            0xffu,
        );
        result |= component << shift;
    }
    return result;
}

fn alpha_allows(state: RexStateSlot, source: u32) -> bool {
    let source_alpha = (source >> 24u) & 0xffu;
    let reference = state.words[REX_STATE_ALPHA_REF] & 0xffu;
    switch (state.words[REX_STATE_DRAW_MODE1] >> 12u) & 0x7u {
        case 0u: {
            return false;
        }
        case 1u: {
            return source_alpha < reference;
        }
        case 2u: {
            return source_alpha == reference;
        }
        case 3u: {
            return source_alpha <= reference;
        }
        case 4u: {
            return source_alpha > reference;
        }
        case 5u: {
            return source_alpha != reference;
        }
        case 6u: {
            return source_alpha >= reference;
        }
        default: {
            return true;
        }
    }
}

fn logic_operation(state: RexStateSlot, source: u32, destination: u32) -> u32 {
    switch (state.words[REX_STATE_DRAW_MODE1] >> 28u) & 0xfu {
        case 0x0u: {
            return 0u;
        }
        case 0x1u: {
            return source & destination;
        }
        case 0x2u: {
            return source & ~destination;
        }
        case 0x3u: {
            return source;
        }
        case 0x4u: {
            return ~source & destination;
        }
        case 0x5u: {
            return destination;
        }
        case 0x6u: {
            return source ^ destination;
        }
        case 0x7u: {
            return source | destination;
        }
        case 0x8u: {
            return ~(source | destination);
        }
        case 0x9u: {
            return ~(source ^ destination);
        }
        case 0xau: {
            return ~destination;
        }
        case 0xbu: {
            return source | ~destination;
        }
        case 0xcu: {
            return ~source;
        }
        case 0xdu: {
            return ~source | destination;
        }
        case 0xeu: {
            return ~(source & destination);
        }
        default: {
            return 0xffffffffu;
        }
    }
}

fn pixel_result(
    state: RexStateSlot,
    source: u32,
    destination_word: u32,
    px: i32,
    py: i32,
) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    if ((mode >> 18u) & 1u) != 0u {
        var destination = expand_destination(state, read_plane(state, destination_word));
        if ((mode >> 25u) & 1u) != 0u {
            destination = state.words[REX_STATE_COLOR_BACK];
        }
        return amplify(
            state,
            compress_source(state, blend_colors(state, source, destination), px, py),
        );
    }
    let compressed_source = amplify(state, compress_source(state, source, px, py));
    let destination = amplify(state, read_plane(state, destination_word));
    return logic_operation(state, compressed_source, destination);
}

struct PatternResult {
    draws: bool,
    background: bool,
}

fn sequence_mod(row: u32, width: u32, column: u32, modulus: u32) -> u32 {
    return (
        (row % modulus) * (width % modulus)
        + column % modulus
    ) % modulus;
}

fn sequence_capped(row: u32, width: u32, column: u32) -> u32 {
    let row_part = min(row, 256u);
    let width_part = min(width, 256u);
    let base = min(row_part * width_part, 256u);
    return min(base + min(column, 256u), 256u);
}

fn line_pattern_state_after(
    count: u32,
    repeat: u32,
    length: u32,
    bit: u32,
    advances_mod: u32,
    advances_capped: u32,
) -> vec2<u32> {
    if advances_capped <= count {
        return vec2<u32>(count - advances_capped, bit);
    }
    let cycle = repeat * length;
    let delta = (advances_mod + cycle - ((count + 1u) % cycle)) % cycle;
    let bit_advances = 1u + delta / repeat;
    let next_count = repeat - 1u - delta % repeat;
    let offset = (31u - bit + bit_advances) % length;
    return vec2<u32>(next_count, 31u - offset);
}

fn clamp_shade(value: u32) -> u32 {
    let integer = (value >> 11u) & 0x1ffu;
    if (value & 0x80000000u) != 0u || integer >= 0x180u {
        return 0u;
    }
    if integer > 0xffu {
        return 0x0007ffffu;
    }
    return value;
}

fn advance_shade(state: RexStateSlot, value: ShadeDda) -> ShadeDda {
    var next = ShadeDda(
        value.red + state.words[REX_STATE_SLOPE_RED],
        value.alpha + state.words[REX_STATE_SLOPE_ALPHA],
        value.green + state.words[REX_STATE_SLOPE_GREEN],
        value.blue + state.words[REX_STATE_SLOPE_BLUE],
    );
    let mode = state.words[REX_STATE_DRAW_MODE1];
    if ((mode >> 15u) & 1u) != 0u {
        next.red = clamp_shade(next.red);
        next.alpha = clamp_shade(next.alpha);
        next.green = clamp_shade(next.green);
        next.blue = clamp_shade(next.blue);
    } else if ((state.words[REX_STATE_DRAW_MODE0] >> 21u) & 1u) != 0u {
        switch (mode >> 3u) & 3u {
            case REX_DRAWDEPTH_8: {
                if (next.red & (1u << 19u)) != 0u {
                    next.red = 0x0007ffffu;
                }
            }
            case REX_DRAWDEPTH_12: {
                if (next.red & (1u << 21u)) != 0u {
                    next.red = 0x001fffffu;
                }
            }
            default: {}
        }
    }
    return next;
}

fn shade_component(value: u32) -> u32 {
    let integer = (value >> 11u) & 0x1ffu;
    if (value & 0x80000000u) != 0u || integer >= 0x180u {
        return 0u;
    }
    return min(integer, 0xffu);
}

fn shade_source(
    state: RexStateSlot,
    command: RexCommand,
    line: bool,
    px: i32,
    py: i32,
) -> u32 {
    var index: u32;
    if line {
        let y_major = (command.flags & REX_FLAG_LINE_Y_MAJOR) != 0u;
        let decreasing = (command.flags & REX_FLAG_LINE_MAJOR_DECREASING) != 0u;
        let major = select(px, py, y_major);
        index = u32(select(major - command.src_x0, command.src_x0 - major, decreasing));
    } else {
        let walker = walker_position(state, px, py);
        let column = u32(abs(walker.x - command.src_x0));
        let row = u32(abs(walker.y - command.src_y0));
        let width = u32(abs(command.src_x1 - command.src_x0)) + 1u;
        index = row * width + column;
    }
    let line_words = select(0u, command.payload_words, line);
    let checkpoint = command.payload_offset + line_words + (index >> 5u) * 4u;
    var value = ShadeDda(
        payload[checkpoint],
        payload[checkpoint + 1u],
        payload[checkpoint + 2u],
        payload[checkpoint + 3u],
    );
    var offset = 0u;
    while offset < (index & 31u) {
        value = advance_shade(state, value);
        offset += 1u;
    }
    let mode = state.words[REX_STATE_DRAW_MODE1];
    var color = value.red >> 11u;
    if ((mode >> 15u) & 1u) != 0u {
        color = (shade_component(value.blue) << 16u)
            | (shade_component(value.green) << 8u)
            | shade_component(value.red);
    }
    let alpha = select(
        shade_component(value.alpha),
        0xffu,
        command.kind == REX_COMMAND_A_LINE,
    );
    return (color & 0x00ffffffu) | (alpha << 24u);
}

fn fill_pattern(
    state: RexStateSlot,
    command: RexCommand,
    px: i32,
    py: i32,
) -> PatternResult {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    let z_enabled = ((mode >> 12u) & 1u) != 0u;
    let line_enabled = ((mode >> 13u) & 1u) != 0u;
    if !z_enabled && !line_enabled {
        return PatternResult(true, false);
    }

    let walker = walker_position(state, px, py);
    let column = u32(abs(walker.x - command.src_x0));
    let row = u32(abs(walker.y - command.src_y0));
    let width = u32(abs(command.src_x1 - command.src_x0)) + 1u;
    var background = false;

    if z_enabled {
        let initial_bit = (state.words[REX_STATE_PATTERN_BITS] >> 8u) & 0xffu;
        let row_bit = select(31u, initial_bit, row == 0u);
        let bit = (row_bit + 32u - (column & 31u)) & 31u;
        if ((state.words[REX_STATE_Z_PATTERN] >> bit) & 1u) == 0u {
            if ((mode >> 16u) & 1u) != 0u {
                background = true;
            } else {
                return PatternResult(false, false);
            }
        }
    }

    if line_enabled {
        let line_mode = state.words[REX_STATE_LINE_MODE];
        let initial_count = line_mode & 0xffu;
        let repeat = max((line_mode >> 8u) & 0xffu, 1u);
        let length = ((line_mode >> 24u) & 0xffu) + 17u;
        let cycle = repeat * length;
        let row_advances_mod = sequence_mod(row, width, 0u, cycle);
        let row_advances_capped = sequence_capped(row, width, 0u);
        let row_state = line_pattern_state_after(
            initial_count,
            repeat,
            length,
            31u,
            row_advances_mod,
            row_advances_capped,
        );
        let initial_bit = state.words[REX_STATE_PATTERN_BITS] & 0xffu;
        let row_bit = select(31u, initial_bit, row == 0u);
        let column_state = line_pattern_state_after(
            row_state.x,
            repeat,
            length,
            row_bit,
            column % cycle,
            min(column, 256u),
        );
        if ((state.words[REX_STATE_LINE_PATTERN] >> column_state.y) & 1u) == 0u {
            if ((mode >> 17u) & 1u) != 0u {
                background = true;
            } else {
                return PatternResult(false, false);
            }
        }
    }

    return PatternResult(true, background);
}

fn clip_allows(state: RexStateSlot, px: i32, py: i32) -> bool {
    let clipmode = state.words[REX_STATE_CLIP_MODE];
    let enabled = clipmode & 0x1fu;
    let xywin = state.words[REX_STATE_XYWIN];
    let y_flip = ((state.words[REX_STATE_DRAW_MODE1] >> 6u) & 1u) != 0u;
    let x_abs = px + REX_COORD_BIAS;
    let y_abs = select(py + REX_COORD_BIAS, REX_YFLIP_BASE - py, y_flip);
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
    var use_background = false;
    let state = states[command.state_slot];
    if fill && command.kind != REX_COMMAND_CLEAR {
        let pattern = fill_pattern(state, command, px, py);
        covered = pattern.draws;
        use_background = pattern.background;
    }
    if line {
        let y_major = (command.flags & REX_FLAG_LINE_Y_MAJOR) != 0u;
        let decreasing = (command.flags & REX_FLAG_LINE_MAJOR_DECREASING) != 0u;
        let major = select(px, py, y_major);
        let delta = select(major - command.src_x0, command.src_x0 - major, decreasing);
        if delta >= 0 && u32(delta) < command.payload_words {
            let minor = select(py, px, y_major);
            let pattern_word = payload[command.payload_offset + u32(delta)];
            if pattern_word != REX_LINE_SKIPPED {
                let expected = i32(pattern_word & ~REX_LINE_BACKGROUND);
                covered = minor == expected;
                use_background = (pattern_word & REX_LINE_BACKGROUND) != 0u;
            }
        }
    }
    if covered && in_bounds
        && px >= command.dst_x0 && px <= command.dst_x1
        && py >= command.dst_y0 && py <= command.dst_y1 {
        let cid_mask = (state.words[REX_STATE_CLIP_MODE] >> 9u) & 0xfu;
        let cid_allows = (command.flags & REX_FLAG_READ_AUX) == 0u
            || (cid_mask & (1u << (aux_value & 3u))) != 0u;
        if !clip_allows(state, px, py) || !cid_allows {
            return;
        }
        let mask = state.words[REX_STATE_WRITE_MASK];
        var foreground_source = state.words[REX_STATE_SOURCE_COLOR];
        if (command.flags & REX_FLAG_SHADED) != 0u {
            foreground_source = shade_source(state, command, line, px, py);
        }
        let raw_source = select(
            foreground_source,
            state.words[REX_STATE_COLOR_BACK],
            use_background,
        );
        if (command.flags & REX_FLAG_WRITE_RGB) != 0u {
            if command.kind == REX_COMMAND_CLEAR {
                rgb_value = (rgb_value & ~mask) | (raw_source & mask);
                rgb_written = true;
            } else if alpha_allows(state, raw_source) {
                let result = pixel_result(state, raw_source, rgb_value, px, py);
                rgb_value = (rgb_value & ~mask) | (result & mask);
                rgb_written = true;
            }
        }
        if (command.flags & REX_FLAG_WRITE_AUX) != 0u {
            if command.kind == REX_COMMAND_CLEAR {
                aux_value = (aux_value & ~mask) | (raw_source & mask);
                aux_written = true;
            } else if alpha_allows(state, raw_source) {
                let result = pixel_result(state, raw_source, aux_value, px, py);
                aux_value = (aux_value & ~mask) | (result & mask);
                aux_written = true;
            }
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
