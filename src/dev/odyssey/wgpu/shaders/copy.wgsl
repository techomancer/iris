// The copy, block 0x2031, on the GPU unit (copy.rs `Board::copy_block`):
// two passes over the copy's w x h pixels, one lane a pixel. `gather`
// reads the source into the scratch buffer, a field's raw bytes or its
// word; `scatter` stores the scratch into the destination, mirrored where
// the copy mirrors, through the window's tests, the scale-and-bias filter
// and the destination's codec. The whole source is read before anything
// is written, so an overlapping copy gives what an ordered walk gives, as
// the CPU model does. The operands arrive packed (wgpu/mod.rs
// `CopyUniform`).

#include "views.wgsl"

const COPY_MIRROR_X: u32 = 1u;
const COPY_MIRROR_Y: u32 = 2u;
// Bytes move as they are (both sides hold a word the same way).
const COPY_RAW: u32 = 4u;
const COPY_FILTER: u32 = 8u;
// The filter reads the source bytes as signed.
const COPY_SIGNED: u32 = 16u;
// The destination is a window: its pixels pass the clip list and id test.
const COPY_WINDOW: u32 = 32u;
// The source view is chosen per pixel from the window-id plane: the
// buffer the pixel's window shows (COPY_SELECT_FRONT) or the other.
const COPY_SELECT_PIXEL: u32 = 64u;
// Otherwise: source B when set, A when clear.
const COPY_SELECT_B: u32 = 128u;
// A raw store under byte masks (the X server's window-id copies).
const COPY_BYTE_MASKS: u32 = 256u;
const COPY_SELECT_FRONT: u32 = 512u;

const ID_COMPARE_ENABLE: u32 = 1u << 16u;
const ID_COMPARE_MASK_SHIFT: u32 = 8u;

struct CopyOp {
    src_a: vec4<u32>,
    src_b: vec4<u32>,
    dst: vec4<u32>,
    did: vec4<u32>,
    // width, height, source x, source y (plane pixels)
    geo: vec4<i32>,
    // destination x, y, the flags, the display's shown-buffer bits
    geo2: vec4<i32>,
    // clip rectangles kept, the id compare word, the byte mask low and high
    misc: vec4<u32>,
    // the destination plane's width
    misc2: vec4<u32>,
    clip: array<vec4<i32>, 4>,
    // scale and bias for R, G, B, A, signed 3.12: (sR, bR, sG, bG), (sB, bB, sA, bA)
    sb: array<vec4<i32>, 2>,
}

@group(0) @binding(0) var<uniform> op: CopyOp;
// One entry a pixel: a raw value (low half first) or a word in .x.
@group(0) @binding(1) var<storage, read_write> scratch: array<vec2<u32>>;

fn view_of(v: vec4<u32>) -> View {
    return View(v.x, v.y, v.z, v.w);
}

// Bytes a raw copy moves of a field (view.rs View::len): the lanes field
// is eight, its lanes in the first six.
fn view_raw_len(v: View) -> u32 {
    if (v.kind & 0xffu) == KIND_LANES {
        return 8u;
    }
    return view_len(v);
}

fn flags() -> u32 {
    return u32(op.geo2.z);
}

// The source view that holds plane pixel (x, y): A or B
// (Board::read_buffer for a window's front and back).
fn source(x: i32, y: i32) -> View {
    let f = flags();
    if (f & COPY_SELECT_PIXEL) != 0u {
        let did = view_load(view_of(op.did), x, y);
        var id = 0u;
        if did.ok {
            id = (did.v >> DID_MAIN_SHIFT) & (DISPMODE_COUNT - 1u);
        }
        let shown = (u32(op.geo2.w) >> id) & 1u;
        var b = shown;
        if (f & COPY_SELECT_FRONT) == 0u {
            b = shown ^ 1u;
        }
        if b != 0u {
            return view_of(op.src_b);
        }
        return view_of(op.src_a);
    }
    if (f & COPY_SELECT_B) != 0u {
        return view_of(op.src_b);
    }
    return view_of(op.src_a);
}

// Board::pixel_allowed for the destination window: the plane bound, the
// id test, then inside a kept clip rectangle.
fn allowed(x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || u32(x) >= op.misc2.x || u32(y) >= MAX_DIM {
        return false;
    }
    let idc = op.misc.y;
    if (idc & ID_COMPARE_ENABLE) != 0u {
        let id = view_load(view_of(op.did), x, y);
        if !id.ok {
            return false;
        }
        let mask = (idc >> ID_COMPARE_MASK_SHIFT) & 0xffu;
        if ((id.v ^ (idc & 0xffu)) & mask) != 0u {
            return false;
        }
    }
    let n = min(op.misc.x, 4u);
    for (var i = 0u; i < n; i++) {
        let r = op.clip[i];
        if x >= r.x && x <= r.z && y >= r.y && y <= r.w {
            return true;
        }
    }
    return false;
}

// Board::copy_post_filter: byte b of lane c becomes b * scale + bias in
// 3.12 fixed point (255 times the bias, so the sum is the result times
// 255 and 4096), clamped and rounded to a byte, ties up; an alpha bias of
// one or more is an opaque overwrite.
fn filter_word(word: u32) -> u32 {
    let signed = (flags() & COPY_SIGNED) != 0u;
    var out = word;
    for (var c = 0u; c < 4u; c++) {
        let s = op.sb[c / 2u][(c % 2u) * 2u];
        let b = op.sb[c / 2u][(c % 2u) * 2u + 1u];
        let shift = 24u - 8u * c;
        let byte = (word >> shift) & 0xffu;
        var x = i32(byte);
        if signed && byte >= 128u {
            x = x - 256;
        }
        var v: i32;
        if c == 3u && b >= 4096 {
            v = 255;
        } else {
            let n = clamp(x * s + 255 * b, 0, 255 * 4096);
            v = (n + 2048) >> 12u;
        }
        out = (out & ~(0xffu << shift)) | (u32(v) << shift);
    }
    return out;
}

@compute @workgroup_size(64)
fn gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(op.geo.x);
    let h = u32(op.geo.y);
    let i = gid.x;
    if i >= w * h {
        return;
    }
    let x = op.geo.z + i32(i % w);
    let y = op.geo.w + i32(i / w);
    let v = source(x, y);
    var r = vec2<u32>(0u, 0u);
    if (flags() & COPY_RAW) != 0u {
        if view_has(v, x, y) {
            r = vram_load64(view_addr(v, x, y), view_raw_len(v));
        }
    } else {
        let l = view_load(v, x, y);
        if l.ok {
            r.x = l.v;
        }
    }
    scratch[i] = r;
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(op.geo.x);
    let h = u32(op.geo.y);
    let i = gid.x;
    if i >= w * h {
        return;
    }
    let f = flags();
    var cx = i32(i % w);
    var cy = i32(i / w);
    if (f & COPY_MIRROR_X) != 0u {
        cx = i32(w) - 1 - cx;
    }
    if (f & COPY_MIRROR_Y) != 0u {
        cy = i32(h) - 1 - cy;
    }
    let x = op.geo2.x + cx;
    let y = op.geo2.y + cy;
    if (f & COPY_WINDOW) != 0u && !allowed(x, y) {
        return;
    }
    let d = view_of(op.dst);
    let r = scratch[i];
    if (f & COPY_RAW) != 0u {
        if !view_has(d, x, y) {
            return;
        }
        let a = view_addr(d, x, y);
        let n = view_raw_len(d);
        if (f & COPY_BYTE_MASKS) != 0u {
            let m = vec2<u32>(op.misc.z, op.misc.w);
            let old = vram_load64(a, n);
            vram_store64(a, n, (old & ~m) | (r & m));
        } else {
            vram_store64(a, n, r);
        }
        return;
    }
    var word = r.x;
    if (f & COPY_FILTER) != 0u {
        word = filter_word(word);
    }
    view_store(d, x, y, word);
}
