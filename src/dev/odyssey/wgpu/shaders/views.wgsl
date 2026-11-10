// A field of a surface in board memory (view.rs): the surface's base, its
// pitch in 16 x 16 tiles, its pixel size and the field's byte inside the
// pixel, and the field's kind. Tiles are row-major, pixels row-major
// inside a tile. A view arrives packed in four words (wgpu/mod.rs
// `View::pack`).

#include "vram.wgsl"

const KIND_NONE: u32 = 0u;
const KIND_RGBA8: u32 = 1u;
const KIND_Z24: u32 = 2u;
const KIND_BYTE: u32 = 3u;
// A texel field (field.rs): the byte count in bits 15:8 of the kind word
// and FIELD_FMT in 23:16.
const KIND_TEXEL: u32 = 4u;
// Four 12-bit lanes in the field's first 6 bytes.
const KIND_LANES: u32 = 5u;

const FIELD_INDEX: u32 = 0u;
const FIELD_ONE: u32 = 2u;
const FIELD_TWO: u32 = 3u;
const FIELD_FOUR: u32 = 5u;
const FIELD_PACKED: u32 = 6u;

struct View {
    base: u32,
    // pitch in bits 15:0, log2 of the pixel size in 23:16, the field's
    // byte in 31:24.
    geom: u32,
    kind: u32,
    // width in bits 15:0, height in 31:16.
    dims: u32,
}

fn view_width(v: View) -> u32 {
    return v.dims & 0xffffu;
}

fn view_height(v: View) -> u32 {
    return v.dims >> 16u;
}

// Whether (x, y) is in the view.
fn view_has(v: View, x: i32, y: i32) -> bool {
    return v.kind != KIND_NONE && x >= 0 && y >= 0 && u32(x) < view_width(v) && u32(y) < view_height(v);
}

// The byte address of the field at (x, y), unchecked.
fn view_addr(v: View, x: i32, y: i32) -> u32 {
    let ux = u32(x);
    let uy = u32(y);
    let pitch = v.geom & 0xffffu;
    let log2 = (v.geom >> 16u) & 0xffu;
    let at = v.geom >> 24u;
    let tile = (uy >> 4u) * pitch + (ux >> 4u);
    return v.base + (((tile << 8u) | ((uy & 15u) << 4u) | (ux & 15u)) << log2) + at;
}

// Bytes the field's value is held in.
fn view_len(v: View) -> u32 {
    var r: u32;
    switch v.kind & 0xffu {
        case KIND_RGBA8: { r = 4u; }
        case KIND_Z24: { r = 3u; }
        case KIND_BYTE: { r = 1u; }
        case KIND_TEXEL: { r = (v.kind >> 8u) & 0xffu; }
        case KIND_LANES: { r = 6u; }
        default: { r = 0u; }
    }
    return r;
}

// Texel codecs (field.rs texel_pack / texel_unpack): a 0xRRGGBBAA word
// to and from the field's value, the field's first byte the value's top.

fn narrow(c: u32, n: u32) -> u32 {
    return c >> (8u - n);
}

// An n-bit value widened to 8 bits by repeating its bits.
fn widen(v: u32, n: u32) -> u32 {
    var out = 0u;
    var have = 0u;
    while have < 8u {
        out = (out << n) | v;
        have += n;
    }
    return out >> (have - 8u);
}

fn chan(word: u32, k: u32) -> u32 {
    return (word >> (24u - 8u * k)) & 0xffu;
}

fn word4(c: vec4<u32>) -> u32 {
    return (c.x << 24u) | (c.y << 16u) | (c.z << 8u) | c.w;
}

fn texel_pack(word: u32, bytes: u32, field: u32) -> u32 {
    let c0 = chan(word, 0u);
    let c1 = chan(word, 1u);
    let c2 = chan(word, 2u);
    let c3 = chan(word, 3u);
    if bytes == 2u {
        switch field {
            case FIELD_INDEX: { return word & 0xffffu; }
            case FIELD_ONE: { return (c0 << 8u) | c0; }
            case FIELD_TWO: { return (c0 << 8u) | c3; }
            case FIELD_FOUR: { return (narrow(c0, 4u) << 12u) | (narrow(c1, 4u) << 8u) | (narrow(c2, 4u) << 4u) | narrow(c3, 4u); }
            case FIELD_PACKED: { return (narrow(c0, 5u) << 11u) | (narrow(c1, 5u) << 6u) | (narrow(c2, 5u) << 1u) | narrow(c3, 1u); }
            default: {}
        }
    } else if bytes == 4u {
        switch field {
            case FIELD_INDEX: { return word; }
            case FIELD_TWO: { return (((c0 << 8u) | c0) << 16u) | (c3 << 8u) | c3; }
            case FIELD_PACKED: { return (((c0 << 2u) | (c0 >> 6u)) << 22u) | (((c1 << 2u) | (c1 >> 6u)) << 12u) | (((c2 << 2u) | (c2 >> 6u)) << 2u) | narrow(c3, 2u); }
            default: {}
        }
    }
    // The word's top bytes.
    return word >> (32u - 8u * min(bytes, 4u));
}

fn texel_unpack(w: u32, bytes: u32, field: u32) -> u32 {
    if bytes == 2u {
        switch field {
            case FIELD_INDEX: { return w; }
            case FIELD_ONE: {
                let v = (w >> 8u) & 0xffu;
                return word4(vec4<u32>(v, v, v, v));
            }
            case FIELD_TWO: {
                let b0 = (w >> 8u) & 0xffu;
                return word4(vec4<u32>(b0, b0, b0, w & 0xffu));
            }
            case FIELD_FOUR: { return word4(vec4<u32>(widen((w >> 12u) & 0xfu, 4u), widen((w >> 8u) & 0xfu, 4u), widen((w >> 4u) & 0xfu, 4u), widen(w & 0xfu, 4u))); }
            case FIELD_PACKED: { return word4(vec4<u32>(widen((w >> 11u) & 0x1fu, 5u), widen((w >> 6u) & 0x1fu, 5u), widen((w >> 1u) & 0x1fu, 5u), select(0u, 0xffu, (w & 1u) != 0u))); }
            default: {}
        }
    } else if bytes == 4u {
        switch field {
            case FIELD_INDEX: { return w; }
            case FIELD_TWO: {
                let b0 = (w >> 24u) & 0xffu;
                return word4(vec4<u32>(b0, b0, b0, (w >> 8u) & 0xffu));
            }
            case FIELD_PACKED: { return word4(vec4<u32>((w >> 24u) & 0xffu, (w >> 14u) & 0xffu, (w >> 4u) & 0xffu, widen(w & 3u, 2u))); }
            default: {}
        }
    }
    // The value as the word's top bytes.
    return w << (32u - 8u * min(bytes, 4u));
}

// Four 12-bit lanes (48 bits, R highest) from a word: each byte at the top
// of its lane, its top bits repeated below; and back.
fn lanes_of(word: u32) -> vec2<u32> {
    var l: vec4<u32>;
    for (var k = 0u; k < 4u; k++) {
        let c = chan(word, k);
        l[k] = (c << 4u) | (c >> 4u);
    }
    // bits 47:36 l0, 35:24 l1, 23:12 l2, 11:0 l3.
    let lo = (l.y << 24u) | (l.z << 12u) | l.w;
    let hi = (l.x << 4u) | (l.y >> 8u);
    return vec2<u32>(lo, hi);
}

fn word_of_lanes(v: vec2<u32>) -> u32 {
    let l0 = (v.y >> 4u) & 0xfffu;
    let l1 = ((v.y << 8u) | (v.x >> 24u)) & 0xfffu;
    let l2 = (v.x >> 12u) & 0xfffu;
    let l3 = v.x & 0xfffu;
    return word4(vec4<u32>((l0 >> 4u) & 0xffu, (l1 >> 4u) & 0xffu, (l2 >> 4u) & 0xffu, (l3 >> 4u) & 0xffu));
}

// The field's word at (x, y), when it is in the view; `ok` says so.
struct Loaded {
    ok: bool,
    v: u32,
}

// A field's word (see the module) from its value at byte address a, for
// a field of kind `kind` (the View kind word).
fn load_kind(a: u32, kind: u32) -> u32 {
    let k = kind & 0xffu;
    if k == KIND_LANES {
        return word_of_lanes(vram_load64(a, 6u));
    }
    var n: u32;
    switch k {
        case KIND_RGBA8: { n = 4u; }
        case KIND_Z24: { n = 3u; }
        case KIND_BYTE: { n = 1u; }
        case KIND_TEXEL: { n = (kind >> 8u) & 0xffu; }
        default: { n = 0u; }
    }
    if n == 0u {
        return 0u;
    }
    let raw = vram_load(a, n);
    if k == KIND_TEXEL {
        return texel_unpack(raw, n, (kind >> 16u) & 0xffu);
    }
    return raw;
}

fn view_load(v: View, x: i32, y: i32) -> Loaded {
    if !view_has(v, x, y) {
        return Loaded(false, 0u);
    }
    return Loaded(true, load_kind(tile_addr(v, view_addr(v, x, y), x, y), v.kind));
}

fn view_store(v: View, x: i32, y: i32, w: u32) {
    if !view_has(v, x, y) {
        return;
    }
    let a = tile_addr(v, view_addr(v, x, y), x, y);
    let kind = v.kind & 0xffu;
    if kind == KIND_LANES {
        vram_store64(a, 6u, lanes_of(w));
        return;
    }
    var val = w;
    if kind == KIND_Z24 {
        val = w & 0xffffffu;
    } else if kind == KIND_BYTE {
        val = w & 0xffu;
    } else if kind == KIND_TEXEL {
        val = texel_pack(w, (v.kind >> 8u) & 0xffu, (v.kind >> 16u) & 0xffu);
    }
    vram_store(a, view_len(v), val);
}
