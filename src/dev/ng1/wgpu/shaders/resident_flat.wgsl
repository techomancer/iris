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

fn compress_source_at(
    state: RexStateSlot,
    source: u32,
    walker: vec2<i32>,
) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    if ((mode >> 15u) & 1u) == 0u {
        return source;
    }
    let dither = ((mode >> 16u) & 1u) != 0u;
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

fn compress_source(
    state: RexStateSlot,
    source: u32,
    px: i32,
    py: i32,
) -> u32 {
    return compress_source_at(state, source, walker_position(state, px, py));
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

struct FillPosition {
    covered: bool,
    row: u32,
    column: u32,
    index: u32,
    last: bool,
}

fn fill_position(
    state: RexStateSlot,
    command: RexCommand,
    px: i32,
    py: i32,
) -> FillPosition {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    let octant = (state.words[34] >> 24u) & 7u;
    let block = (command.flags & REX_FLAG_FILL_BLOCK) != 0u;
    let x_decreasing = block && (octant & REX_OCTANT_XDEC) != 0u;
    let y_decreasing = block && (octant & REX_OCTANT_YDEC) != 0u;
    let y_stride = select(1u, 2u, block && ((mode >> 23u) & 1u) != 0u);
    let walker = walker_position(state, px, py);
    let y_delta = select(
        walker.y - command.src_y0,
        command.src_y0 - walker.y,
        y_decreasing,
    );
    if y_delta < 0 || u32(y_delta) % y_stride != 0u {
        return FillPosition(false, 0u, 0u, 0u, false);
    }
    let row = u32(y_delta) / y_stride;
    let rows = u32(abs(command.src_y1 - command.src_y0)) / y_stride + 1u;
    if row >= rows {
        return FillPosition(false, 0u, 0u, 0u, false);
    }
    let first_width = command.data[1];
    let full_width = command.data[2];
    let width = select(first_width, full_width, row > 0u);
    let row_start = select(command.src_x0, bitcast<i32>(command.data[0]), row > 0u);
    let x_delta = select(
        walker.x - row_start,
        row_start - walker.x,
        x_decreasing,
    );
    if x_delta < 0 || u32(x_delta) >= width {
        return FillPosition(false, 0u, 0u, 0u, false);
    }
    let column = u32(x_delta);
    var before_row = 0u;
    if row > 0u {
        before_row = first_width + (row - 1u) * full_width;
    }
    let first_complete =
        (command.flags & REX_FLAG_FILL_FIRST_ROW_COMPLETE) != 0u;
    let last = column + 1u == width && (row > 0u || first_complete);
    let skip_first = ((mode >> 10u) & 1u) != 0u;
    let skip_last = ((mode >> 11u) & 1u) != 0u;
    let lr_skip = block
        && ((mode >> 19u) & 1u) != 0u
        && x_decreasing;
    let suppressed = select(
        (column == 0u && skip_first) || (last && skip_last),
        skip_first || (last && skip_last) || lr_skip,
        block,
    );
    return FillPosition(
        !suppressed,
        row,
        column,
        before_row + column,
        last,
    );
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
        index = fill_position(state, command, px, py).index;
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
    position: FillPosition,
) -> PatternResult {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    let z_enabled = ((mode >> 12u) & 1u) != 0u;
    let line_enabled = ((mode >> 13u) & 1u) != 0u;
    if !z_enabled && !line_enabled {
        return PatternResult(true, false);
    }

    let column = position.column;
    let row = position.row;
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
        let before_row = position.index - column;
        let row_advances_mod = before_row % cycle;
        let row_advances_capped = min(before_row, 256u);
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

struct HostStream {
    high: u32,
    low: u32,
    count: u32,
    cursor: u32,
    length: u32,
}

struct HostFetch {
    pixel: u32,
    stream: HostStream,
}

struct HostWalk {
    xstart: i32,
    ystart: i32,
    mid_primitive: bool,
    stream: HostStream,
    pat_bit: u32,
    zpat_bit: u32,
    lsrcount: u32,
    shade: ShadeDda,
}

struct HostLineWalk {
    x: i32,
    y: i32,
    d: i32,
    stream: HostStream,
    pat_bit: u32,
    zpat_bit: u32,
    lsrcount: u32,
    shade: ShadeDda,
}

fn host_swap32(value: u32) -> u32 {
    return ((value & 0x000000ffu) << 24u)
        | ((value & 0x0000ff00u) << 8u)
        | ((value & 0x00ff0000u) >> 8u)
        | ((value & 0xff000000u) >> 24u);
}

fn host_count(mode: u32) -> u32 {
    if ((mode >> 7u) & 1u) == 0u {
        return 1u;
    }
    let depth = (mode >> 8u) & 3u;
    if ((mode >> 10u) & 1u) != 0u {
        return select(select(8u, 4u, depth == 2u), 2u, depth == 3u);
    }
    return select(select(4u, 2u, depth == 2u), 1u, depth == 3u);
}

fn host_shift(mode: u32) -> u32 {
    if ((mode >> 7u) & 1u) == 0u {
        return 0u;
    }
    let depth = (mode >> 8u) & 3u;
    return select(select(8u, 16u, depth == 2u), 32u, depth == 3u);
}

fn host_expand(depth: u32, rgb_mode: bool, value: u32) -> u32 {
    if !rgb_mode {
        return value;
    }
    switch depth {
        case 0u: {
            let red = select(0u, 0xffu, (value & 1u) != 0u);
            let green_raw = (value >> 1u) & 3u;
            let green = (green_raw << 6u) | (green_raw << 4u)
                | (green_raw << 2u) | green_raw;
            let blue = select(0u, 0xffu, (value & 8u) != 0u);
            return (blue << 16u) | (green << 8u) | red;
        }
        case 1u: {
            let red_raw = value & 7u;
            let red = (red_raw << 5u) | (red_raw << 2u) | (red_raw >> 1u);
            let green_raw = (value >> 3u) & 7u;
            let green = (green_raw << 5u) | (green_raw << 2u) | (green_raw >> 1u);
            let blue_raw = (value >> 6u) & 3u;
            let blue = (blue_raw << 6u) | (blue_raw << 4u)
                | (blue_raw << 2u) | blue_raw;
            return (blue << 16u) | (green << 8u) | red;
        }
        case 2u: {
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

fn host_shift_stream(stream: HostStream, amount: u32) -> HostStream {
    var result = stream;
    switch amount {
        case 8u: {
            result.high = (stream.high << 8u) | (stream.low >> 24u);
            result.low = stream.low << 8u;
        }
        case 16u: {
            result.high = (stream.high << 16u) | (stream.low >> 16u);
            result.low = stream.low << 16u;
        }
        case 32u: {
            result.high = stream.low;
            result.low = 0u;
        }
        default: {}
    }
    return result;
}

fn host_fetch(
    state: RexStateSlot,
    command: RexCommand,
    input: HostStream,
) -> HostFetch {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    var stream = input;
    if stream.count == 0u {
        let last = select(0u, stream.length - 1u, stream.length > 0u);
        let index = min(stream.cursor, last);
        let offset = command.payload_offset + index * 2u;
        stream.high = payload[offset];
        stream.low = payload[offset + 1u];
        if ((mode >> 11u) & 1u) != 0u {
            if ((mode >> 10u) & 1u) != 0u {
                let old_high = stream.high;
                stream.high = host_swap32(stream.low);
                stream.low = host_swap32(old_high);
            } else {
                stream.high = host_swap32(stream.high);
            }
        }
        if stream.cursor < stream.length {
            stream.cursor += 1u;
        }
        stream.count = host_count(mode);
    }

    let depth = (mode >> 8u) & 3u;
    var packed = stream.high;
    switch depth {
        case 0u: {
            packed = (stream.high >> 24u) & 0xfu;
        }
        case 1u: {
            packed = stream.high >> 24u;
        }
        case 2u: {
            packed = (stream.high >> 16u) & 0xfffu;
        }
        default: {}
    }
    var pixel = host_expand(depth, ((mode >> 15u) & 1u) != 0u, packed);
    let draw_mode = state.words[REX_STATE_DRAW_MODE0];
    let alpha_host = ((draw_mode >> 7u) & 1u) != 0u;
    let color_host = ((draw_mode >> 6u) & 1u) != 0u;
    if alpha_host && !color_host && (depth == 0u || depth == 1u) {
        pixel = (pixel & 0x00ffffffu) | (stream.high & 0xff000000u);
    }

    stream = host_shift_stream(stream, host_shift(mode));
    stream.count -= 1u;
    return HostFetch(pixel, stream);
}

fn host_physical(state: RexStateSlot, walker: vec2<i32>) -> vec2<i32> {
    let draw_mode = state.words[REX_STATE_DRAW_MODE0];
    let xy_offset = ((draw_mode >> 20u) & 1u) != 0u;
    let xywin = state.words[REX_STATE_XYWIN];
    let xymove = state.words[REX_STATE_XYMOVE];
    let x = walker.x + select(0, high_i16(xymove), xy_offset) + high_i16(xywin);
    let y = walker.y + select(0, low_i16(xymove), xy_offset) + low_i16(xywin);
    let y_flip = ((state.words[REX_STATE_DRAW_MODE1] >> 6u) & 1u) != 0u;
    return vec2<i32>(
        x - REX_COORD_BIAS,
        select(y - REX_COORD_BIAS, REX_YFLIP_BASE - y, y_flip),
    );
}

fn host_write_pixel(
    state: RexStateSlot,
    command: RexCommand,
    walker: vec2<i32>,
    source: u32,
) {
    let destination = host_physical(state, walker);
    if destination.x < 0 || destination.y < 0
        || u32(destination.x) >= WRITE_WIDTH
        || u32(destination.y) >= PLANE_HEIGHT
        || !clip_allows(state, destination.x, destination.y) {
        return;
    }
    let destination_address =
        u32(destination.y) * PLANE_STRIDE + u32(destination.x);
    let destination_aux = aux[destination_address];
    let cid_mask = (state.words[REX_STATE_CLIP_MODE] >> 9u) & 0xfu;
    if (command.flags & REX_FLAG_READ_AUX) != 0u
        && (cid_mask & (1u << (destination_aux & 3u))) == 0u {
        return;
    }
    if !alpha_allows(state, source) {
        return;
    }

    let mask = state.words[REX_STATE_WRITE_MASK];
    if (command.flags & REX_FLAG_WRITE_RGB) != 0u {
        let old = rgb[destination_address];
        let result = pixel_result(
            state,
            source,
            old,
            destination.x,
            destination.y,
        );
        rgb[destination_address] = (old & ~mask) | (result & mask);
    }
    if (command.flags & REX_FLAG_WRITE_AUX) != 0u {
        let old = destination_aux;
        let result = pixel_result(
            state,
            source,
            old,
            destination.x,
            destination.y,
        );
        aux[destination_address] = (old & ~mask) | (result & mask);
    }
}

fn host_pattern_bits(
    state: RexStateSlot,
    pat_bit: u32,
    zpat_bit: u32,
) -> PatternResult {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    var background = false;
    if ((mode >> 12u) & 1u) != 0u
        && ((state.words[REX_STATE_Z_PATTERN] >> zpat_bit) & 1u) == 0u {
        if ((mode >> 16u) & 1u) != 0u {
            background = true;
        } else {
            return PatternResult(false, false);
        }
    }
    if ((mode >> 13u) & 1u) != 0u
        && ((state.words[REX_STATE_LINE_PATTERN] >> pat_bit) & 1u) == 0u {
        if ((mode >> 17u) & 1u) != 0u {
            background = true;
        } else {
            return PatternResult(false, false);
        }
    }
    return PatternResult(true, background);
}

fn host_pattern(state: RexStateSlot, walk: HostWalk) -> PatternResult {
    return host_pattern_bits(state, walk.pat_bit, walk.zpat_bit);
}

fn host_shade_source(state: RexStateSlot, shade: ShadeDda) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    var color = shade.red >> 11u;
    if ((mode >> 15u) & 1u) != 0u {
        color = (shade_component(shade.blue) << 16u)
            | (shade_component(shade.green) << 8u)
            | shade_component(shade.red);
    }
    return (color & 0x00ffffffu) | (shade_component(shade.alpha) << 24u);
}

fn host_advance(state: RexStateSlot, input: HostWalk) -> HostWalk {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    var walk = input;
    if ((mode >> 18u) & 1u) != 0u {
        walk.shade = advance_shade(state, walk.shade);
    }
    if ((mode >> 12u) & 1u) != 0u {
        walk.zpat_bit = (walk.zpat_bit + 31u) & 31u;
    }
    if ((mode >> 13u) & 1u) != 0u {
        let line_mode = state.words[REX_STATE_LINE_MODE];
        let repeat = max((line_mode >> 8u) & 0xffu, 1u);
        if walk.lsrcount == 0u {
            walk.lsrcount = repeat - 1u;
            let wrap = 32u - (((line_mode >> 24u) & 0xfu) + 17u);
            walk.pat_bit = select(
                (walk.pat_bit + 31u) & 31u,
                31u,
                walk.pat_bit == wrap,
            );
        } else {
            walk.lsrcount -= 1u;
        }
    }
    return walk;
}

fn host_invocation(
    state: RexStateSlot,
    command: RexCommand,
    block: bool,
    input: HostWalk,
) -> HostWalk {
    let draw_mode = state.words[REX_STATE_DRAW_MODE0];
    let color_host = ((draw_mode >> 6u) & 1u) != 0u;
    let alpha_host = ((draw_mode >> 7u) & 1u) != 0u;
    let stop_on_y = ((draw_mode >> 9u) & 1u) != 0u;
    let stop_on_word = color_host && (!block || !stop_on_y);
    let stop_on_x = ((draw_mode >> 8u) & 1u) != 0u || stop_on_word;
    let octant = (state.words[34] >> 24u) & 7u;
    let x_decreasing = block && (octant & REX_OCTANT_XDEC) != 0u;
    let y_decreasing = block && (octant & REX_OCTANT_YDEC) != 0u;
    if !block
        && ((draw_mode >> 19u) & 1u) != 0u
        && (octant & REX_OCTANT_XDEC) != 0u {
        return input;
    }
    let x_step = select(1 << 11, -(1 << 11), x_decreasing);
    var y_step = select(1 << 11, -(1 << 11), y_decreasing);
    if block && ((draw_mode >> 23u) & 1u) != 0u {
        y_step *= 2;
    }
    let xend = bitcast<i32>(state.words[REX_STATE_HOST_XEND]);
    let yend = bitcast<i32>(state.words[REX_STATE_HOST_YEND]);
    let xsave = bitcast<i32>(state.words[REX_STATE_HOST_XSAVE]);
    let skip_first = ((draw_mode >> 10u) & 1u) != 0u;
    let skip_last = ((draw_mode >> 11u) & 1u) != 0u;
    let lr_skip = block
        && ((draw_mode >> 19u) & 1u) != 0u
        && x_decreasing;

    var walk = input;
    let span_length = select(
        (xend - walk.xstart) >> 11,
        abs(xend - walk.xstart) >> 11,
        block,
    );
    let has_xstop = ((draw_mode >> 15u) & 1u) != 0u
        && span_length >= 32;
    let xstop = walk.xstart + x_step * 32;
    var first = true;
    walk.mid_primitive = true;
    loop {
        let walker = vec2<i32>(walk.xstart >> 11, walk.ystart >> 11);
        walk.xstart += x_step;
        let x_end_reached = select(
            walk.xstart > xend,
            walk.xstart < xend,
            x_decreasing,
        );
        let suppressed = select(
            (first && skip_first) || (x_end_reached && skip_last),
            skip_first || (x_end_reached && skip_last) || lr_skip,
            block,
        );
        if !suppressed {
            let pattern = host_pattern(state, walk);
            if pattern.draws {
                var source = state.words[REX_STATE_SOURCE_COLOR];
                if ((draw_mode >> 18u) & 1u) != 0u {
                    source = host_shade_source(state, walk.shade);
                }
                if pattern.background {
                    source = state.words[REX_STATE_COLOR_BACK];
                } else {
                    let fetched = host_fetch(state, command, walk.stream);
                    walk.stream = fetched.stream;
                    var color = source;
                    var alpha = source >> 24u;
                    if color_host {
                        color = fetched.pixel;
                    }
                    if alpha_host {
                        alpha = fetched.pixel >> 24u;
                    }
                    source = (color & 0x00ffffffu) | (alpha << 24u);
                }
                host_write_pixel(state, command, walker, source);
            }
        }
        walk = host_advance(state, walk);

        if x_end_reached {
            if !block {
                walk.mid_primitive = false;
                break;
            }
            walk.ystart += y_step;
            walk.xstart = xsave;
            if !stop_on_y {
                walk.mid_primitive = false;
                break;
            }
            let y_end_reached = select(
                walk.ystart > yend,
                walk.ystart < yend,
                y_decreasing,
            );
            if y_end_reached {
                walk.mid_primitive = false;
                break;
            }
            if stop_on_word && walk.stream.count > 0u {
                break;
            }
            if color_host && !stop_on_word {
                walk.stream.count = 0u;
            }
        } else if has_xstop {
            let xstop_reached = select(
                walk.xstart >= xstop,
                walk.xstart <= xstop,
                x_decreasing,
            );
            if xstop_reached {
                break;
            }
        }
        if stop_on_word
            && walk.stream.count == 0u
            && walk.stream.cursor >= walk.stream.length {
            break;
        }
        if !stop_on_x {
            break;
        }
        if !block {
            first = false;
        }
    }
    return walk;
}

fn host_fill(state: RexStateSlot, command: RexCommand) {
    var walk = HostWalk(
        bitcast<i32>(state.words[REX_STATE_HOST_XSTART]),
        bitcast<i32>(state.words[REX_STATE_HOST_YSTART]),
        (state.words[REX_STATE_PATTERN_BITS] & (1u << 16u)) != 0u,
        HostStream(
            state.words[46],
            state.words[45],
            state.words[44],
            state.words[REX_STATE_HOST_CURSOR],
            state.words[REX_STATE_HOST_LENGTH],
        ),
        state.words[REX_STATE_PATTERN_BITS] & 0xffu,
        (state.words[REX_STATE_PATTERN_BITS] >> 8u) & 0xffu,
        state.words[REX_STATE_LINE_MODE] & 0xffu,
        ShadeDda(
            state.words[REX_STATE_COLOR_RED],
            state.words[REX_STATE_COLOR_ALPHA],
            state.words[REX_STATE_COLOR_GREEN],
            state.words[REX_STATE_COLOR_BLUE],
        ),
    );
    let block = command.kind == REX_COMMAND_HOST_BLOCK;
    walk = host_invocation(state, command, block, walk);
    loop {
        if walk.stream.length <= 1u || walk.stream.cursor >= walk.stream.length {
            break;
        }
        let before = walk.stream.cursor;
        walk.stream.count = 0u;
        walk = host_invocation(state, command, block, walk);
        if walk.stream.cursor == before {
            break;
        }
    }
}

fn host_line_steps(octant: u32) -> vec4<i32> {
    switch octant {
        case 0u: { return vec4<i32>(0, 1, -1, -1); }
        case 1u: { return vec4<i32>(0, 1, 1, 1); }
        case 2u: { return vec4<i32>(0, -1, -1, -1); }
        case 3u: { return vec4<i32>(0, -1, 1, 1); }
        case 4u: { return vec4<i32>(1, 1, 0, -1); }
        case 5u: { return vec4<i32>(1, 1, 0, 1); }
        case 6u: { return vec4<i32>(-1, -1, 0, -1); }
        default: { return vec4<i32>(-1, -1, 0, 1); }
    }
}

fn host_line_round_bit(octant: u32) -> u32 {
    switch octant {
        case 0u: { return 1u; }
        case 1u: { return 6u; }
        case 2u: { return 2u; }
        case 3u: { return 5u; }
        case 4u: { return 0u; }
        case 5u: { return 7u; }
        case 6u: { return 3u; }
        default: { return 4u; }
    }
}

fn host_line_empty_weight(weights: u32, x_fraction: u32, y_fraction: u32) -> bool {
    if x_fraction == 0u && y_fraction == 0u {
        return false;
    }
    let index = min(x_fraction + y_fraction, 15u);
    if index >= 8u {
        return true;
    }
    return ((weights >> (index * 4u)) & 0xfu) == 0u;
}

fn host_line_endpoint_skips(
    state: RexStateSlot,
    antialiased: bool,
    first_invocation: bool,
) -> vec2<bool> {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    if !antialiased || ((mode >> 22u) & 1u) == 0u {
        return vec2<bool>(false, false);
    }
    let xstart = bitcast<i32>(state.words[REX_STATE_HOST_XSTART]);
    let ystart = bitcast<i32>(state.words[REX_STATE_HOST_YSTART]);
    let xend = bitcast<i32>(state.words[REX_STATE_HOST_XEND]);
    let yend = bitcast<i32>(state.words[REX_STATE_HOST_YEND]);
    return vec2<bool>(
        first_invocation
            && host_line_empty_weight(
                state.words[38],
                u32((xstart >> 7) & 0xf),
                u32((ystart >> 7) & 0xf),
            ),
        host_line_empty_weight(
            state.words[39],
            u32((xend >> 7) & 0xf),
            u32((yend >> 7) & 0xf),
        ),
    );
}

fn host_line_advance_pattern(
    state: RexStateSlot,
    input: HostLineWalk,
) -> HostLineWalk {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    var walk = input;
    if ((mode >> 12u) & 1u) != 0u {
        walk.zpat_bit = (walk.zpat_bit + 31u) & 31u;
    }
    if ((mode >> 13u) & 1u) != 0u {
        let line_mode = state.words[REX_STATE_LINE_MODE];
        let repeat = max((line_mode >> 8u) & 0xffu, 1u);
        if walk.lsrcount == 0u {
            walk.lsrcount = repeat - 1u;
            let wrap = 32u - (((line_mode >> 24u) & 0xfu) + 17u);
            walk.pat_bit = select(
                (walk.pat_bit + 31u) & 31u,
                31u,
                walk.pat_bit == wrap,
            );
        } else {
            walk.lsrcount -= 1u;
        }
    }
    return walk;
}

fn host_line_invocation(
    state: RexStateSlot,
    command: RexCommand,
    input: HostLineWalk,
    first_invocation: bool,
) -> HostLineWalk {
    let draw_mode = state.words[REX_STATE_DRAW_MODE0];
    let color_host = ((draw_mode >> 6u) & 1u) != 0u;
    let alpha_host = ((draw_mode >> 7u) & 1u) != 0u;
    let shaded = ((draw_mode >> 18u) & 1u) != 0u;
    let antialiased = command.kind == REX_COMMAND_HOST_A_LINE;
    let fractional = command.kind != REX_COMMAND_HOST_I_LINE;
    let octant = (state.words[34] >> 24u) & 7u;
    let steps = host_line_steps(octant);
    let y_major = octant < 4u;
    let xend = bitcast<i32>(state.words[REX_STATE_HOST_XEND]) >> 11;
    let yend = bitcast<i32>(state.words[REX_STATE_HOST_YEND]) >> 11;
    let adx = abs(xend - input.x);
    let ady = abs(yend - input.y);
    var major = max(adx, ady);
    if fractional {
        major = select(adx, ady, y_major);
    }
    var pixel_count = u32(major) + 1u;
    if ((draw_mode >> 15u) & 1u) != 0u {
        pixel_count = min(pixel_count, 32u);
    }
    let single_step = ((draw_mode >> 8u) & 3u) == 0u;
    if single_step {
        pixel_count = 1u;
    }
    let endpoint_skips =
        host_line_endpoint_skips(state, antialiased, first_invocation);
    let skip_first = !single_step
        && (((draw_mode >> 10u) & 1u) != 0u || endpoint_skips.x);
    let skip_last = !single_step
        && (((draw_mode >> 11u) & 1u) != 0u || endpoint_skips.y);
    let incr1 = i32(state.words[34] & 0x000fffffu);
    let incr2 = bitcast<i32>((state.words[35] & 0x001fffffu) << 11u) >> 11u;
    let round = (state.words[35] >> 24u) & 0xffu;
    let straight_on_zero =
        ((round >> host_line_round_bit(octant)) & 1u) == 0u;

    var walk = input;
    for (var index = 0u; index < pixel_count; index++) {
        let last = index + 1u == pixel_count;
        let suppressed = (index == 0u && skip_first) || (last && skip_last);
        if !suppressed {
            let pattern = host_pattern_bits(state, walk.pat_bit, walk.zpat_bit);
            if pattern.draws {
                var source = state.words[REX_STATE_SOURCE_COLOR];
                if shaded {
                    source = host_shade_source(state, walk.shade);
                }
                if antialiased && !alpha_host {
                    source = (source & 0x00ffffffu) | 0xff000000u;
                }
                if pattern.background {
                    source = state.words[REX_STATE_COLOR_BACK];
                } else {
                    let fetched = host_fetch(state, command, walk.stream);
                    walk.stream = fetched.stream;
                    var color = source;
                    var alpha = source >> 24u;
                    if color_host {
                        color = fetched.pixel;
                    }
                    if alpha_host {
                        alpha = fetched.pixel >> 24u;
                    }
                    source = (color & 0x00ffffffu) | (alpha << 24u);
                }
                host_write_pixel(
                    state,
                    command,
                    vec2<i32>(walk.x, walk.y),
                    source,
                );
            }
        }

        if shaded {
            walk.shade = advance_shade(state, walk.shade);
        }
        if !last || ((draw_mode >> 14u) & 1u) != 0u {
            walk = host_line_advance_pattern(state, walk);
        }
        if !last || single_step {
            let straight = walk.d < 0
                || (walk.d == 0 && straight_on_zero);
            if straight {
                walk.x += steps.x;
                walk.y -= steps.z;
                walk.d += incr1;
            } else {
                walk.x += steps.y;
                walk.y -= steps.w;
                walk.d += incr2;
            }
        }
    }
    return walk;
}

fn host_line(state: RexStateSlot, command: RexCommand) {
    var walk = HostLineWalk(
        bitcast<i32>(state.words[REX_STATE_HOST_XSTART]) >> 11,
        bitcast<i32>(state.words[REX_STATE_HOST_YSTART]) >> 11,
        bitcast<i32>((state.words[32] & 0x07ffffffu) << 5u) >> 5u,
        HostStream(
            state.words[46],
            state.words[45],
            state.words[44],
            state.words[REX_STATE_HOST_CURSOR],
            state.words[REX_STATE_HOST_LENGTH],
        ),
        state.words[REX_STATE_PATTERN_BITS] & 0xffu,
        (state.words[REX_STATE_PATTERN_BITS] >> 8u) & 0xffu,
        state.words[REX_STATE_LINE_MODE] & 0xffu,
        ShadeDda(
            state.words[REX_STATE_COLOR_RED],
            state.words[REX_STATE_COLOR_ALPHA],
            state.words[REX_STATE_COLOR_GREEN],
            state.words[REX_STATE_COLOR_BLUE],
        ),
    );
    walk = host_line_invocation(state, command, walk, true);
    loop {
        if walk.stream.length <= 1u || walk.stream.cursor >= walk.stream.length {
            break;
        }
        let before = walk.stream.cursor;
        walk.stream.count = 0u;
        walk = host_line_invocation(state, command, walk, false);
        if walk.stream.cursor == before {
            break;
        }
    }
}

fn copy_physical(
    state: RexStateSlot,
    walker: vec2<i32>,
    moved: bool,
) -> vec2<i32> {
    let xywin = state.words[REX_STATE_XYWIN];
    let xymove = state.words[REX_STATE_XYMOVE];
    let move_x = select(0, high_i16(xymove), moved);
    let move_y = select(0, low_i16(xymove), moved);
    let absolute_x = walker.x + high_i16(xywin) + move_x;
    let absolute_y = walker.y + low_i16(xywin) + move_y;
    let y_flip = ((state.words[REX_STATE_DRAW_MODE1] >> 6u) & 1u) != 0u;
    return vec2<i32>(
        absolute_x - REX_COORD_BIAS,
        select(absolute_y - REX_COORD_BIAS, REX_YFLIP_BASE - absolute_y, y_flip),
    );
}

fn copy_walker_position(state: RexStateSlot, px: i32, py: i32) -> vec2<i32> {
    let xywin = state.words[REX_STATE_XYWIN];
    let xymove = state.words[REX_STATE_XYMOVE];
    let y_flip = ((state.words[REX_STATE_DRAW_MODE1] >> 6u) & 1u) != 0u;
    let absolute_y = select(py + REX_COORD_BIAS, REX_YFLIP_BASE - py, y_flip);
    return vec2<i32>(
        px + REX_COORD_BIAS - high_i16(xywin) - high_i16(xymove),
        absolute_y - low_i16(xywin) - low_i16(xymove),
    );
}

fn copy_pixel_result(
    state: RexStateSlot,
    source_word: u32,
    destination_word: u32,
    walker: vec2<i32>,
) -> u32 {
    let mode = state.words[REX_STATE_DRAW_MODE1];
    let source = expand_destination(state, read_plane(state, source_word));
    if ((mode >> 18u) & 1u) != 0u {
        var destination = expand_destination(state, read_plane(state, destination_word));
        if ((mode >> 25u) & 1u) != 0u {
            destination = state.words[REX_STATE_COLOR_BACK];
        }
        return compress_source_at(state, blend_colors(state, source, destination), walker);
    }
    let compressed_source = amplify(state, compress_source_at(state, source, walker));
    let destination = amplify(state, read_plane(state, destination_word));
    return logic_operation(state, compressed_source, destination);
}

fn copy_pixel(state: RexStateSlot, command: RexCommand, walker: vec2<i32>) {
    let destination = copy_physical(state, walker, true);
    if destination.x < 0 || destination.y < 0
        || u32(destination.x) >= WRITE_WIDTH
        || u32(destination.y) >= PLANE_HEIGHT
        || !clip_allows(state, destination.x, destination.y) {
        return;
    }
    let destination_address =
        u32(destination.y) * PLANE_STRIDE + u32(destination.x);
    let destination_aux = aux[destination_address];
    let cid_mask = (state.words[REX_STATE_CLIP_MODE] >> 9u) & 0xfu;
    if cid_mask != 0xfu
        && (cid_mask & (1u << (destination_aux & 3u))) == 0u {
        return;
    }

    let source = copy_physical(state, walker, false);
    var source_word = 0u;
    if source.x >= 0 && source.y >= 0
        && u32(source.x) < WRITE_WIDTH
        && u32(source.y) < PLANE_HEIGHT {
        let source_address = u32(source.y) * PLANE_STRIDE + u32(source.x);
        source_word = select(
            aux[source_address],
            rgb[source_address],
            plane(state) == REX_PLANES_RGB || plane(state) == REX_PLANES_RGBA,
        );
    }

    let mask = state.words[REX_STATE_WRITE_MASK];
    if (command.flags & REX_FLAG_WRITE_RGB) != 0u {
        let old = rgb[destination_address];
        let result = copy_pixel_result(state, source_word, old, walker);
        rgb[destination_address] = (old & ~mask) | (result & mask);
    }
    if (command.flags & REX_FLAG_WRITE_AUX) != 0u {
        let old = destination_aux;
        let result = copy_pixel_result(state, source_word, old, walker);
        aux[destination_address] = (old & ~mask) | (result & mask);
    }
}

fn copy_fill(state: RexStateSlot, command: RexCommand) {
    let mode = state.words[REX_STATE_DRAW_MODE0];
    let octant = (state.words[34] >> 24u) & 7u;
    let block = command.kind == REX_COMMAND_COPY_BLOCK;
    let x_step = select(1, -1, block && (octant & REX_OCTANT_XDEC) != 0u);
    var y_step = select(1, -1, block && (octant & REX_OCTANT_YDEC) != 0u);
    if block && ((mode >> 23u) & 1u) != 0u {
        y_step *= 2;
    }
    let skip_first = ((mode >> 10u) & 1u) != 0u;
    let skip_last = ((mode >> 11u) & 1u) != 0u;
    let lr_skip = block
        && ((mode >> 19u) & 1u) != 0u
        && (octant & REX_OCTANT_XDEC) != 0u;

    let y_stride = u32(abs(y_step));
    let rows = u32(abs(command.src_y1 - command.src_y0)) / y_stride + 1u;
    for (var row = 0u; row < rows; row++) {
        let row_start = select(
            command.src_x0,
            bitcast<i32>(command.data[0]),
            row > 0u,
        );
        let width = select(command.data[1], command.data[2], row > 0u);
        let row_complete = row > 0u
            || (command.flags & REX_FLAG_FILL_FIRST_ROW_COMPLETE) != 0u;
        let y = command.src_y0 + i32(row) * y_step;
        for (var column = 0u; column < width; column++) {
            let x = row_start + i32(column) * x_step;
            let last = column + 1u == width && row_complete;
            let suppressed = select(
                (column == 0u && skip_first) || (last && skip_last),
                skip_first || (last && skip_last) || lr_skip,
                block,
            );
            if !suppressed {
                copy_pixel(state, command, vec2<i32>(x, y));
            }
        }
    }
}

fn copy_line(state: RexStateSlot, command: RexCommand) {
    let y_major = (command.flags & REX_FLAG_LINE_Y_MAJOR) != 0u;
    let decreasing = (command.flags & REX_FLAG_LINE_MAJOR_DECREASING) != 0u;
    for (var index = 0u; index < command.payload_words; index++) {
        let word = payload[command.payload_offset + index];
        if word != REX_LINE_SKIPPED {
            let major_delta = select(i32(index), -i32(index), decreasing);
            let major = command.src_x0 + major_delta;
            let minor = i32(word & ~REX_LINE_BACKGROUND);
            let destination = select(
                vec2<i32>(major, minor),
                vec2<i32>(minor, major),
                y_major,
            );
            copy_pixel(
                state,
                command,
                copy_walker_position(state, destination.x, destination.y),
            );
        }
    }
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
    let host_command = command.kind == REX_COMMAND_HOST_BLOCK
        || command.kind == REX_COMMAND_HOST_SPAN
        || command.kind == REX_COMMAND_HOST_I_LINE
        || command.kind == REX_COMMAND_HOST_F_LINE
        || command.kind == REX_COMMAND_HOST_A_LINE;
    if host_command {
        if lane == 0u {
            let state = states[command.state_slot];
            if command.kind == REX_COMMAND_HOST_BLOCK
                || command.kind == REX_COMMAND_HOST_SPAN {
                host_fill(state, command);
            } else {
                host_line(state, command);
            }
        }
        return;
    }
    let copy_fill_command = command.kind == REX_COMMAND_COPY_BLOCK
        || command.kind == REX_COMMAND_COPY_SPAN;
    let copy_line_command = command.kind == REX_COMMAND_COPY_I_LINE
        || command.kind == REX_COMMAND_COPY_F_LINE
        || command.kind == REX_COMMAND_COPY_A_LINE;
    if copy_fill_command || copy_line_command {
        if lane == 0u {
            let state = states[command.state_slot];
            if copy_fill_command {
                copy_fill(state, command);
            } else {
                copy_line(state, command);
            }
        }
        return;
    }
    let fill = command.kind == REX_COMMAND_BLOCK
        || command.kind == REX_COMMAND_SPAN
        || command.kind == REX_COMMAND_CLEAR;
    let line = command.kind == REX_COMMAND_I_LINE
        || command.kind == REX_COMMAND_F_LINE
        || command.kind == REX_COMMAND_A_LINE;
    var covered = false;
    var use_background = false;
    let state = states[command.state_slot];
    if fill {
        let position = fill_position(state, command, px, py);
        covered = position.covered;
        if covered && command.kind != REX_COMMAND_CLEAR {
            let pattern = fill_pattern(state, command, position);
            covered = pattern.draws;
            use_background = pattern.background;
        }
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
