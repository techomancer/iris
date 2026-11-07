// The back end (pipe/backend.rs) from the board's words: the stencil and
// depth tests and their operations, the destination read, the blend or
// logical operation, the colour mask, the stores; and a clear's pixel.
// The colour arithmetic is integer at 12 bits a component, as there.
//
// A slot holds the back-end words as the command FIFO delivered them
// (pipe/state.rs `BackendWords`), the facts about the draw the words do
// not carry (stencil planes, index mode, an overlay context), the
// destination's views, the window's clip rectangles and id test, and the
// polygon stipple. `backend_of` decodes the words as `BackendState::of`
// does; `fragment_q` is the generic fragment path, which the
// monomorphised stores on the CPU equal for the pixels inside a window.

#include "views.wgsl"

// Slot flags.
const SLOT_HAS_STENCIL: u32 = 1u;
const SLOT_RGBA: u32 = 2u;
const SLOT_TO_AUX: u32 = 4u;
const SLOT_STIPPLE: u32 = 8u;
const SLOT_BACKGROUND: u32 = 16u;

// View indices.
const V_COLOUR0: u32 = 0u;
const V_COLOUR1: u32 = 1u;
const V_DEPTH: u32 = 2u;
const V_STENCIL: u32 = 3u;
const V_DID: u32 = 4u;
const V_OVERLAY: u32 = 5u;

struct Slot {
    mode: u32,        // 0x404
    depth_func: u32,  // 0x420
    write_mask: u32,  // 0x452
    stencil: u32,     // 0x41f
    blend0: u32,      // 0xc21
    blend1: u32,
    blend2: u32,
    logic_op: u32,    // 0x422
    draw_mask: u32,   // 0x424
    flags: u32,
    clip_kept: u32,
    id_compare: u32,  // 0x840 operand 1
    pad0: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
    views: array<View, 6>,
    clip: array<vec4<i32>, 4>,
    pattern: array<u32, 32>,
    bg12: vec4<i32>,
}

@group(1) @binding(1) var<storage, read> slots: array<Slot>;

// Register bits (consts.rs).
const MODE_DEPTH_TEST: u32 = 1u << 16u;
const MODE_STENCIL_TEST: u32 = 1u << 15u;
const MODE_BLEND: u32 = 1u << 17u;
const MODE_INDEX_LOGIC_OP: u32 = 1u << 19u;
const MODE_COLOR_LOGIC_OP: u32 = 1u << 20u;
const WRITE_DEPTH: u32 = 1u << 24u;
const WRITE_COLOUR_SHIFT: u32 = 20u;
const WRITE_STENCIL_SHIFT: u32 = 12u;
const WRITE_INDEX_MASK: u32 = 0xfffu;
const STENCIL_FUNC_SHIFT: u32 = 25u;
const STENCIL_VALUE_MASK_SHIFT: u32 = 8u;
const STENCIL_OP_FAIL_SHIFT: u32 = 22u;
const STENCIL_OP_ZFAIL_SHIFT: u32 = 19u;
const STENCIL_OP_ZPASS_SHIFT: u32 = 16u;
const BLEND_DST_MASK: u32 = 0x3fu;
const BLEND_SRC_SHIFT: u32 = 6u;
const BLEND_SRC_MASK: u32 = 0x7fu;
const BLEND_EQ_SHIFT: u32 = 13u;
const DRAW_MASK_BUFFER0: u32 = 0x3u;
const DRAW_MASK_BUFFER1: u32 = 0xcu;
const ID_COMPARE_ENABLE: u32 = 1u << 16u;
const ID_COMPARE_MASK_SHIFT: u32 = 8u;
const DEPTH24_MAX: u32 = 0xffffffu;

// Blend factors, in the table order of backend.rs FACTORS.
const BF_ZERO: u32 = 0u;
const BF_ONE: u32 = 1u;
const BF_OTHER: u32 = 2u;
const BF_SRC_ALPHA: u32 = 3u;
const BF_DST_ALPHA: u32 = 4u;
const BF_CONST: u32 = 5u;
const BF_CONST_ALPHA: u32 = 6u;
const BF_ONE_MINUS_OTHER: u32 = 7u;
const BF_ONE_MINUS_SRC_ALPHA: u32 = 8u;
const BF_ONE_MINUS_DST_ALPHA: u32 = 9u;
const BF_ONE_MINUS_CONST: u32 = 10u;
const BF_ONE_MINUS_CONST_ALPHA: u32 = 11u;
const BF_SRC_ALPHA_SATURATE: u32 = 12u;
const BF_UNKNOWN: u32 = 13u;

// Blend equations (BlendEquation::of).
const BE_ADD: u32 = 0u;
const BE_MIN: u32 = 1u;
const BE_MAX: u32 = 2u;
const BE_LOGIC_OP: u32 = 5u;
const BE_SUBTRACT: u32 = 6u;
const BE_REVERSE_SUBTRACT: u32 = 7u;

// Stencil operations (StencilOp::of).
const SO_KEEP: u32 = 0u;
const SO_ZERO: u32 = 1u;
const SO_REPLACE: u32 = 2u;
const SO_INCR: u32 = 3u;
const SO_DECR: u32 = 4u;
const SO_INVERT: u32 = 5u;

// The board's blend factor code to the table index (BlendFactor::of).
fn blend_factor(code: u32) -> u32 {
    var r: u32;
    switch code {
        case 0x00u: { r = BF_ZERO; }
        case 0x08u: { r = BF_ONE; }
        case 0x10u: { r = BF_OTHER; }
        case 0x18u: { r = BF_SRC_ALPHA; }
        case 0x20u: { r = BF_DST_ALPHA; }
        case 0x28u: { r = BF_CONST; }
        case 0x30u: { r = BF_CONST_ALPHA; }
        case 0x38u: { r = BF_ONE_MINUS_OTHER; }
        case 0x39u: { r = BF_ONE_MINUS_SRC_ALPHA; }
        case 0x3au: { r = BF_ONE_MINUS_DST_ALPHA; }
        case 0x3bu: { r = BF_ONE_MINUS_CONST; }
        case 0x3cu: { r = BF_ONE_MINUS_CONST_ALPHA; }
        case 0x40u: { r = BF_SRC_ALPHA_SATURATE; }
        default: { r = BF_UNKNOWN; }
    }
    return r;
}

// Compare::test for the 3-bit code.
fn compare(code: u32, a: u32, b: u32) -> bool {
    var r: bool;
    switch code & 7u {
        case 0u: { r = false; }
        case 1u: { r = a < b; }
        case 2u: { r = a == b; }
        case 3u: { r = a <= b; }
        case 4u: { r = a > b; }
        case 5u: { r = a != b; }
        case 6u: { r = a >= b; }
        default: { r = true; }
    }
    return r;
}

// logic.rs logic_op.
fn logic_op(op: u32, s: u32, d: u32) -> u32 {
    var r: u32;
    switch op & 0xfu {
        case 0u: { r = 0u; }
        case 1u: { r = s & d; }
        case 2u: { r = s & ~d; }
        case 3u: { r = s; }
        case 4u: { r = ~s & d; }
        case 5u: { r = d; }
        case 6u: { r = s ^ d; }
        case 7u: { r = s | d; }
        case 8u: { r = ~(s | d); }
        case 9u: { r = ~(s ^ d); }
        case 10u: { r = ~d; }
        case 11u: { r = s | ~d; }
        case 12u: { r = ~s; }
        case 13u: { r = ~s | d; }
        case 14u: { r = ~(s & d); }
        default: { r = 0xffffffffu; }
    }
    return r;
}

// What the back end decides once per draw (BackendState).
struct Backend {
    depth_test: bool,
    depth_func: u32,
    depth_write: bool,
    stencil: bool,
    stencil_func: u32,
    stencil_ref: u32,
    stencil_value_mask: u32,
    stencil_ops: vec3<u32>,
    stencil_write_mask: u32,
    blend: bool,
    blend_src: u32,
    blend_dst: u32,
    blend_eq: u32,
    const_colour: vec4<u32>,
    has_logic: bool,
    logic: u32,
    colour_mask: u32,
    index_mask: u32,
    rgba: bool,
    target0: bool,
    target1: bool,
    reads: bool,
}

fn field12(word: u32, shift: u32) -> u32 {
    return (word >> shift) & 0xfffu;
}

// BackendState::of(words, has_stencil, rgba) with aux_targets.
fn backend_of(si: u32) -> Backend {
    var b: Backend;
    let has_stencil = (slots[si].flags & SLOT_HAS_STENCIL) != 0u;
    let rgba = (slots[si].flags & SLOT_RGBA) != 0u;
    let blend_on = (slots[si].mode & MODE_BLEND) != 0u;
    let eq = (slots[si].blend0 >> BLEND_EQ_SHIFT) & 7u;
    let index_mask = slots[si].write_mask & WRITE_INDEX_MASK;

    b.depth_test = (slots[si].mode & MODE_DEPTH_TEST) != 0u;
    b.depth_func = slots[si].depth_func & 7u;
    b.depth_write = b.depth_test && (slots[si].write_mask & WRITE_DEPTH) != 0u;
    b.stencil = has_stencil && (slots[si].mode & MODE_STENCIL_TEST) != 0u;
    b.stencil_func = (slots[si].stencil >> STENCIL_FUNC_SHIFT) & 7u;
    b.stencil_ref = min(slots[si].stencil & 0xffu, 0xffu);
    b.stencil_value_mask = (slots[si].stencil >> STENCIL_VALUE_MASK_SHIFT) & 0xffu;
    b.stencil_ops = vec3<u32>((slots[si].stencil >> STENCIL_OP_FAIL_SHIFT) & 7u, (slots[si].stencil >> STENCIL_OP_ZFAIL_SHIFT) & 7u, (slots[si].stencil >> STENCIL_OP_ZPASS_SHIFT) & 7u);
    b.stencil_write_mask = (slots[si].write_mask >> WRITE_STENCIL_SHIFT) & 0xffu;
    b.has_logic = (rgba && (slots[si].mode & MODE_COLOR_LOGIC_OP) != 0u) || (!rgba && (slots[si].mode & MODE_INDEX_LOGIC_OP) != 0u) || (blend_on && eq == BE_LOGIC_OP);
    b.logic = slots[si].logic_op & 0xfu;
    b.blend = rgba && blend_on && !b.has_logic;
    b.blend_src = blend_factor((slots[si].blend0 >> BLEND_SRC_SHIFT) & BLEND_SRC_MASK);
    b.blend_dst = blend_factor(slots[si].blend0 & BLEND_DST_MASK);
    b.blend_eq = eq;
    b.const_colour = vec4<u32>(min(field12(slots[si].blend1, 12u), 4095u), min(field12(slots[si].blend1, 0u), 4095u), min(field12(slots[si].blend2, 12u), 4095u), min(field12(slots[si].blend2, 0u), 4095u));
    if rgba {
        b.colour_mask = (slots[si].write_mask >> WRITE_COLOUR_SHIFT) & 0xfu;
    } else {
        b.colour_mask = select(0u, 0xfu, index_mask != 0u);
    }
    b.index_mask = index_mask;
    b.rgba = rgba;
    b.target0 = (slots[si].draw_mask & DRAW_MASK_BUFFER0) != 0u;
    b.target1 = (slots[si].draw_mask & DRAW_MASK_BUFFER1) != 0u;
    if (slots[si].flags & SLOT_TO_AUX) != 0u {
        // An overlay context's draw: an index goes to buffer B; RGBA has
        // no home there.
        b.target0 = false;
        b.target1 = !rgba && b.target1;
    }
    if rgba {
        b.reads = b.blend || b.has_logic || b.colour_mask != 0xfu;
    } else {
        b.reads = b.has_logic || index_mask != WRITE_INDEX_MASK;
    }
    return b;
}

// The window's ownership of plane pixel (x, y): the destination's width,
// the clip list, and the id test (RasterState::owned).
fn owned(si: u32, x: i32, y: i32) -> bool {
    let width = view_width(slots[si].views[V_COLOUR0]);
    if x < 0 || y < 0 || u32(x) >= width || u32(y) >= MAX_DIM {
        return false;
    }
    let n = min(slots[si].clip_kept, 4u);
    var inside = false;
    for (var i = 0u; i < n; i++) {
        let r = slots[si].clip[i];
        inside = inside || (x >= r.x && x <= r.z && y >= r.y && y <= r.w);
    }
    if !inside {
        return false;
    }
    if (slots[si].id_compare & ID_COMPARE_ENABLE) != 0u {
        let id_mask = (slots[si].id_compare >> ID_COMPARE_MASK_SHIFT) & 0xffu;
        if id_mask != 0u {
            let id = view_load(slots[si].views[V_DID], x, y);
            let value = slots[si].id_compare & 0xffu;
            if ((id.v ^ value) & id_mask) != 0u {
                return false;
            }
        }
    }
    return true;
}

// Packers.
fn pack_rgba8(c: vec4<u32>) -> u32 {
    return ((c.x & 0xffu) << 24u) | ((c.y & 0xffu) << 16u) | ((c.z & 0xffu) << 8u) | (c.w & 0xffu);
}

fn unpack_rgba8(w: u32) -> vec4<u32> {
    return vec4<u32>(w >> 24u, (w >> 16u) & 0xffu, (w >> 8u) & 0xffu, w & 0xffu);
}

fn index_of(src: vec4<u32>) -> u32 {
    return (((src.y >> 4u) & 0xfu) << 8u) | (src.z >> 4u);
}

fn widen8(v: u32) -> u32 {
    return (v << 4u) | (v >> 4u);
}

fn mul12(a: u32, b: u32) -> u32 {
    let p = a * b;
    return (p + (p >> 12u) + 2048u) >> 12u;
}

// A blend factor's 12-bit value for component k.
fn factor(f: u32, k: u32, other_is_dst: bool, s: vec4<u32>, d: vec4<u32>, c: vec4<u32>) -> u32 {
    let other = select(s, d, other_is_dst);
    var r: u32;
    switch f {
        case BF_ZERO: { r = 0u; }
        case BF_ONE: { r = 4095u; }
        case BF_OTHER: { r = other[k]; }
        case BF_SRC_ALPHA: { r = s.w; }
        case BF_DST_ALPHA: { r = d.w; }
        case BF_CONST: { r = c[k]; }
        case BF_CONST_ALPHA: { r = c.w; }
        case BF_ONE_MINUS_OTHER: { r = 4095u - other[k]; }
        case BF_ONE_MINUS_SRC_ALPHA: { r = 4095u - s.w; }
        case BF_ONE_MINUS_DST_ALPHA: { r = 4095u - d.w; }
        case BF_ONE_MINUS_CONST: { r = 4095u - c[k]; }
        case BF_ONE_MINUS_CONST_ALPHA: { r = 4095u - c.w; }
        case BF_SRC_ALPHA_SATURATE: { r = select(min(s.w, 4095u - d.w), 4095u, k == 3u); }
        default: { r = 0u; }
    }
    return r;
}

// The blend of the fragment's 12-bit components over the destination's.
fn blend(b: Backend, s: vec4<u32>, d: vec4<u32>) -> vec4<u32> {
    var out: vec4<u32>;
    for (var k = 0u; k < 4u; k++) {
        let sf = factor(b.blend_src, k, true, s, d, b.const_colour);
        let df = factor(b.blend_dst, k, false, s, d, b.const_colour);
        let a = i32(mul12(s[k], sf));
        let bb = i32(mul12(d[k], df));
        var v: i32;
        switch b.blend_eq {
            case BE_MIN: { v = i32(min(s[k], d[k])); }
            case BE_MAX: { v = i32(max(s[k], d[k])); }
            case BE_SUBTRACT: { v = a - bb; }
            case BE_REVERSE_SUBTRACT: { v = bb - a; }
            default: { v = a + bb; }
        }
        out[k] = u32(clamp(v, 0, 4095));
    }
    return out;
}

// The 8-bit colour a fragment of 12-bit components stores over the 8-bit
// destination.
fn colour_result(b: Backend, src: vec4<u32>, dst: vec4<u32>) -> vec4<u32> {
    var out: vec4<u32>;
    if b.blend {
        out = blend(b, src, vec4<u32>(widen8(dst.x), widen8(dst.y), widen8(dst.z), widen8(dst.w)));
    } else {
        out = src;
    }
    out = out >> vec4<u32>(4u);
    for (var k = 0u; k < 4u; k++) {
        if b.has_logic {
            out[k] = logic_op(b.logic, out[k], dst[k]) & 0xffu;
        }
        if (b.colour_mask & (1u << k)) == 0u {
            out[k] = dst[k];
        }
    }
    return out;
}

// The stencil operation op on old under the write mask; `ok` false keeps.
fn stencil_value(b: Backend, op: u32, old: u32) -> Loaded {
    var value: u32;
    switch op {
        case SO_ZERO: { value = 0u; }
        case SO_REPLACE: { value = b.stencil_ref; }
        case SO_INCR: { value = select(0xffu, old + 1u, old < 0xffu); }
        case SO_DECR: { value = select(old - 1u, 0u, old == 0u); }
        case SO_INVERT: { value = ~old & 0xffu; }
        default: { return Loaded(false, 0u); }
    }
    let mask = b.stencil_write_mask & 0xffu;
    return Loaded(true, (old & ~mask) | (value & mask));
}

fn store_index(b: Backend, si: u32, x: i32, y: i32, index: u32) {
    for (var t = 0u; t < 2u; t++) {
        if !select(b.target0, b.target1, t == 1u) {
            continue;
        }
        let v = slots[si].views[V_COLOUR0 + t];
        if !b.reads {
            view_store(v, x, y, index);
            continue;
        }
        let old = view_load(v, x, y);
        if !old.ok {
            continue;
        }
        var val = index;
        if b.has_logic {
            val = logic_op(b.logic, index, old.v);
        }
        view_store(v, x, y, (val & b.index_mask) | (old.v & ~b.index_mask));
    }
}

// One fragment at plane pixel (x, y) with depth word zq and colour src in
// 12 bits a component, through the tests and into memory
// (BackendState::fragment_q).
fn fragment_q(b: Backend, si: u32, x: i32, y: i32, zq: u32, src: vec4<u32>) {
    var outcome = 2u;
    var old_stencil = 0u;
    if b.stencil {
        old_stencil = view_load(slots[si].views[V_STENCIL], x, y).v;
        if !compare(b.stencil_func, b.stencil_ref & b.stencil_value_mask, old_stencil & b.stencil_value_mask) {
            outcome = 0u;
        }
    }
    if outcome == 2u && b.depth_test {
        let stored = view_load(slots[si].views[V_DEPTH], x, y).v;
        if !compare(b.depth_func, zq, stored) {
            outcome = 1u;
        }
    }
    if outcome != 2u {
        if b.stencil {
            let v = stencil_value(b, b.stencil_ops[outcome], old_stencil);
            if v.ok {
                view_store(slots[si].views[V_STENCIL], x, y, v.v);
            }
        }
        return;
    }
    if b.depth_write {
        view_store(slots[si].views[V_DEPTH], x, y, zq & DEPTH24_MAX);
    }
    if b.stencil {
        let v = stencil_value(b, b.stencil_ops[2], old_stencil);
        if v.ok {
            view_store(slots[si].views[V_STENCIL], x, y, v.v);
        }
    }
    if b.colour_mask == 0u {
        return;
    }
    if !b.rgba {
        store_index(b, si, x, y, index_of(src));
        return;
    }
    for (var t = 0u; t < 2u; t++) {
        if !select(b.target0, b.target1, t == 1u) {
            continue;
        }
        let v = slots[si].views[V_COLOUR0 + t];
        if !b.reads {
            view_store(v, x, y, pack_rgba8(src >> vec4<u32>(4u)));
            continue;
        }
        let d = view_load(v, x, y);
        if !d.ok {
            continue;
        }
        view_store(v, x, y, pack_rgba8(colour_result(b, src, unpack_rgba8(d.v))));
    }
}

// A clear's pixel (ClearDraw::span). The record carries the values; the
// slot the planes and the id test.
const CLEAR_COLOUR: u32 = 1u;
const CLEAR_DEPTH: u32 = 2u;
const CLEAR_STENCIL: u32 = 4u;
const CLEAR_TARGET0: u32 = 8u;
const CLEAR_TARGET1: u32 = 16u;

fn clear_pixel(si: u32, flags: u32, x: i32, y: i32, colour: u32, colour_mask: u32, depth: u32, stencil_vm: u32, id_vm: u32) {
    let id_mask = id_vm >> 8u;
    if id_mask != 0u {
        let id = view_load(slots[si].views[V_DID], x, y);
        if !id.ok || ((id.v ^ (id_vm & 0xffu)) & id_mask) != 0u {
            return;
        }
    }
    if (flags & CLEAR_DEPTH) != 0u {
        view_store(slots[si].views[V_DEPTH], x, y, depth & DEPTH24_MAX);
    }
    if (flags & CLEAR_STENCIL) != 0u {
        let old = view_load(slots[si].views[V_STENCIL], x, y);
        if old.ok {
            let m = stencil_vm >> 8u;
            view_store(slots[si].views[V_STENCIL], x, y, (old.v & ~m) | (stencil_vm & m));
        }
    }
    if (flags & CLEAR_COLOUR) != 0u {
        for (var t = 0u; t < 2u; t++) {
            if (flags & (CLEAR_TARGET0 << t)) == 0u {
                continue;
            }
            let v = slots[si].views[V_COLOUR0 + t];
            var out = colour;
            if colour_mask != 0xffffffffu {
                let d = view_load(v, x, y);
                if !d.ok {
                    continue;
                }
                out = (colour & colour_mask) | (d.v & ~colour_mask);
            }
            view_store(v, x, y, out);
        }
    }
}
