// The texture unit (pipe/texture.rs) for one pixel of a quad: the
// pixel's position in each axis, the quad's level of detail from the
// diagonals (the four lanes of a quad share their positions, or a line's
// lane makes up its quad), the
// filter and the levels lambda chooses, the taps gathered from board
// memory and filtered with 8-bit weights to 12 bits with 16 fraction
// bits, the detail image, the post-filter, and the texel in 12.12. The
// arithmetic is the CPU unit's operation for operation; where the CPU
// uses 64-bit products they are split here.
//
// A texture slot (wgpu/mod.rs `TexSlot`) is the draw's texture registers
// and what the CPU latches from them once (`Latch`): the levels, the LOD
// clamps and scales, the border colour. Slots are in their own buffer.

#include "float.wgsl"
#include "views.wgsl"

struct TexLevel {
    addr: u32,
    pitch: u32,
    bpp: u32,     // log2 of the texel size
    field: u32,
    bytes: u32,
    ox: u32,
    oy: u32,
    wl: u32,
    hl: u32,
    dl: u32,
    slice: u32,
    pad: u32,
}

struct TexSlot {
    sampler0: u32,
    sampler4: u32,
    axes: u32,
    flags: u32,
    count: u32,
    min_lod: i32,
    max_lod: i32,
    mag_below: i32,
    lod_scale: array<f32, 4>,
    scale: array<f32, 4>,
    detail_scale: array<f32, 4>,
    clamp: array<i32, 4>,
    border: array<u32, 4>,
    scale_bias: array<i32, 8>,
    // The environment (ENV_*) and its colour in 12.12.
    env: u32,
    env_colour: array<i32, 4>,
    pad: array<u32, 3>,
    levels: array<TexLevel, 32>,
    detail: TexLevel,
    colour_table: array<u32, 512>,
    detail_fn: array<u32, 256>,
}

@group(2) @binding(1) var<storage, read> tex_slots: array<TexSlot>;
// log2(1 + i/256) in 1/256 (fixed::log2_q8's table), i in 0..256.
@group(2) @binding(3) var<storage, read> log2_table: array<u32, 256>;

const TEX_LOD: u32 = 1u;
const TEX_POST: u32 = 2u;
const TEX_STORED_BORDER: u32 = 4u;
const TEX_TABLE: u32 = 8u;
const TEX_TABLE_ALPHA: u32 = 16u;
// The clamp word's "no clamp".
const NO_CLAMP: i32 = 0x7fffffff;
const LAMBDA_MIN: i32 = -(1 << 24);
const FIXED_ONE: i32 = 1 << 24;

const WRAP_REPEAT: u32 = 0u;
const WRAP_CLAMP_TO_EDGE: u32 = 1u;
const WRAP_CLAMP: u32 = 2u;
const WRAP_CLAMP_TO_BORDER: u32 = 3u;

const ENV_REPLACE: u32 = 0u;
const ENV_MODULATE: u32 = 1u;
const ENV_DECAL: u32 = 2u;
const ENV_BLEND: u32 = 3u;
const ENV_ADD: u32 = 4u;

const CLASS_ALPHA: u32 = 0u;
const CLASS_LUMINANCE: u32 = 1u;
const CLASS_LUMINANCE_ALPHA: u32 = 2u;
const CLASS_INTENSITY: u32 = 3u;
const CLASS_RGB: u32 = 4u;
const CLASS_RGBA: u32 = 5u;

// Sampler word 4's fields.
fn tex_wrap(ti: u32, axis: u32) -> u32 {
    return (tex_slots[ti].sampler4 >> (12u - 2u * axis)) & 3u;
}

fn tex_min(ti: u32) -> u32 {
    return (tex_slots[ti].sampler4 >> 5u) & 7u;
}

fn tex_mag(ti: u32) -> u32 {
    let m = (tex_slots[ti].sampler4 >> 1u) & 0xfu;
    return select(m, 1u, m > 4u);
}

fn tex_class(ti: u32) -> u32 {
    let c = (tex_slots[ti].sampler4 >> 23u) & 7u;
    switch c {
        case 1u: { return CLASS_ALPHA; }
        case 2u: { return CLASS_LUMINANCE; }
        case 3u: { return CLASS_LUMINANCE_ALPHA; }
        case 4u: { return CLASS_INTENSITY; }
        case 5u: { return CLASS_RGB; }
        default: { return CLASS_RGBA; }
    }
}

fn min_mipmap(m: u32) -> bool {
    return m >= 2u;
}

fn min_linear(m: u32) -> bool {
    return m == 1u || m == 3u || m == 5u;
}

fn min_nearest_level(m: u32) -> bool {
    return m == 2u || m == 3u;
}

fn mag_detail(m: u32) -> bool {
    return m >= 2u;
}

// A texel's components as the cls fills them.
fn expand(cls: u32, c: vec4<u32>) -> vec4<u32> {
    let one = 0xfffu;
    switch cls {
        case CLASS_ALPHA: { return vec4<u32>(one, one, one, c.w); }
        case CLASS_LUMINANCE: { return vec4<u32>(c.x, c.x, c.x, one); }
        case CLASS_LUMINANCE_ALPHA: { return vec4<u32>(c.x, c.x, c.x, c.w); }
        case CLASS_INTENSITY: { return vec4<u32>(c.x, c.x, c.x, c.x); }
        case CLASS_RGB: { return vec4<u32>(c.x, c.y, c.z, one); }
        default: { return c; }
    }
}

// Rust's `as i32` then the step down for a value below it: floor for |x|
// below 2^31, saturating past.
fn floor_i32(x: f32) -> i32 {
    let i = sat_i32(x);
    return i - select(0, 1, x < f32(i));
}

fn sat_u32(x: f32) -> u32 {
    let c = clamp(x, 0.0, 4294967040.0);
    return select(u32(c), 0u, x != x);
}

// log2(v) in 1/256 for v > 0 (fixed::log2_q8): the leading one's position
// and a table of the eight bits after it (LOG2_TABLE comes with the
// pipeline's defines).
fn log2_q8(v: u32) -> i32 {
    let n = 31u - countLeadingZeros(v);
    var f: u32;
    if n >= 8u {
        f = (v >> (n - 8u)) & 0xffu;
    } else {
        f = (v << (8u - n)) & 0xffu;
    }
    return i32(n << 8u) + i32(log2_table[f]);
}

// (a * (256 - f) + b * f) >> 8 for a, b below 2^28, exactly.
fn blend8(a: u32, b: u32, f: u32) -> u32 {
    let g = 256u - f;
    let lo = (a & 0xffu) * g + (b & 0xffu) * f;
    let hi = (a >> 8u) * g + (b >> 8u) * f;
    return hi + (lo >> 8u);
}

// (v * 4097) >> 16 for v below 2^28: 12 bits with 16 fraction bits over
// 4095 as 12.12 over 4096.
fn to_fixed(v: u32) -> i32 {
    let vl = v & 0xffffu;
    let vh = v >> 16u;
    return i32(vh * 4097u + ((vl * 4097u) >> 16u));
}

// The position along one axis in 1/256 of a texel of a level 2^l texels
// wide: the first tap's index before wrapping and the second tap's weight
// (LINEAR), or the tap's index (NEAREST). `big` levels (2^(l + 8) past
// i32) make the clamps no-ops.
struct Pos {
    at: i32,
    w: u32,
}

fn pos(u_in: i32, l: u32, wrap: u32, linear: bool) -> Pos {
    let big = l + 8u >= 31u;
    let whole = select(1 << (l + 8u), 0x7fffffff, big);
    let last = i32((1u << min(l, 31u)) - 1u);
    var u = u_in;
    if linear {
        switch wrap {
            case WRAP_REPEAT: { u = u - 128; }
            case WRAP_CLAMP_TO_EDGE: {
                u = max(u - 128, 0);
                if !big {
                    u = min(u, whole - 256);
                }
            }
            case WRAP_CLAMP_TO_BORDER: {
                u = max(u, -128);
                if !big {
                    u = min(u, whole + 128);
                }
                u = u - 128;
            }
            default: {
                u = max(u, 0);
                if !big {
                    u = min(u, whole);
                }
                u = u - 128;
            }
        }
        return Pos(u >> 8u, u32(u & 255));
    }
    var x = u >> 8u;
    switch wrap {
        case WRAP_REPEAT: {}
        case WRAP_CLAMP_TO_BORDER: { x = min(max(x, -1), last + 1); }
        default: { x = min(max(x, 0), last); }
    }
    return Pos(x, 0u);
}

// One tap's part of the texel's byte offset along an axis, wrapped, and
// whether it lies outside the level (the border colour).
struct Tap {
    off: u32,
    outside: bool,
}

fn axis_tap(lv: TexLevel, axis: u32, x_in: i32, wrap: u32, border: bool) -> Tap {
    var l: u32;
    var inset: u32;
    if axis == 0u {
        l = lv.wl;
        inset = lv.ox;
    } else if axis == 1u {
        l = lv.hl;
        inset = lv.oy;
    } else {
        l = lv.dl;
        inset = 0u;
    }
    let mask = i32((1u << min(l, 31u)) - 1u);
    var x = x_in;
    var outside = false;
    if wrap == WRAP_REPEAT {
        x = x & mask;
    } else if wrap == WRAP_CLAMP_TO_EDGE {
        x = min(max(x, 0), mask);
    } else if !border && (x < 0 || x > mask) {
        outside = true;
    }
    let ux = u32(x) + inset;
    var off: u32;
    if axis == 0u {
        off = (((ux >> 4u) << 8u) | (ux & 15u)) << lv.bpp;
    } else if axis == 1u {
        off = (((ux >> 4u) * lv.pitch << 8u) | ((ux & 15u) << 4u)) << lv.bpp;
    } else {
        off = ux * lv.slice;
    }
    return Tap(off, outside);
}

// The texel at byte address a of a level, as 12-bit components of the
// cls.
fn texel_at(lv: TexLevel, cls: u32, a: u32) -> vec4<u32> {
    var w: u32;
    if lv.bytes >= 8u {
        w = word_of_lanes(vram_load64(a, 6u));
    } else {
        w = texel_unpack(vram_load(a, lv.bytes), lv.bytes, lv.field);
    }
    var c: vec4<u32>;
    for (var k = 0u; k < 4u; k++) {
        let v = (w >> (24u - 8u * k)) & 0xffu;
        c[k] = (v << 4u) | (v >> 4u);
    }
    if cls == CLASS_ALPHA {
        c = vec4<u32>(0u, 0u, 0u, c.w);
    }
    return expand(cls, c);
}

// A pixel's texel from level lv at first taps `at` with weights w: LINEAR
// over 2 (4, 8 for a volume) taps or NEAREST; 12-bit components with 16
// fraction bits.
fn filter_level(ti: u32, lv: TexLevel, cls: u32, axes: u32, at: vec3<i32>, w: vec3<u32>, linear: bool, wraps: vec3<u32>, border: bool) -> vec4<u32> {
    let three = axes == 3u;
    let bc = vec4<u32>(tex_slots[ti].border[0], tex_slots[ti].border[1], tex_slots[ti].border[2], tex_slots[ti].border[3]);
    let nx = select(1u, 2u, linear);
    let ny = select(1u, 2u, linear && axes >= 2u);
    let nz = select(1u, 2u, linear && three);
    // Each tap is weighted as it is read instead of kept for the filter:
    // the bilinear sum (c00 (256 - a) + c10 a) (256 - b) + (c01 (256 - a)
    // + c11 a) b is, in wrapping 32-bit arithmetic, exactly the sum of each
    // texel times its two weights. An axis with one tap gives it both
    // weights (256), and the nearest filter is one texel at 256 x 256.
    // One loop over the taps, x fastest, so that one tap is live at a time.
    let a = select(0u, w.x, linear);
    let b = select(0u, w.y, linear && axes >= 2u);
    let n = nx * ny * nz;
    var acc = vec4<u32>(0u);
    var plane0 = vec4<u32>(0u);
    for (var t = 0u; t < n; t++) {
        let x = t % nx;
        let y = (t / nx) % ny;
        let z = t / (nx * ny);
        var tz = Tap(0u, false);
        if three {
            tz = axis_tap(lv, 2u, at.z + i32(z), wraps.z, border);
        }
        var ty = Tap(0u, false);
        if axes >= 2u {
            ty = axis_tap(lv, 1u, at.y + i32(y), wraps.y, border);
        }
        let tx = axis_tap(lv, 0u, at.x + i32(x), wraps.x, border);
        let wx = select(select(256u - a, a, x == 1u), 256u, nx == 1u);
        let wy = select(select(256u - b, b, y == 1u), 256u, ny == 1u);
        var c = bc;
        if !(tz.outside || ty.outside || tx.outside) {
            c = texel_at(lv, cls, lv.addr + tz.off + ty.off + tx.off);
        }
        acc += c * (wx * wy);
        if t + 1u == nx * ny && nz == 2u {
            plane0 = acc;
            acc = vec4<u32>(0u);
        }
    }
    if nz == 1u {
        return acc;
    }
    var out: vec4<u32>;
    for (var k = 0u; k < 4u; k++) {
        out[k] = blend8(plane0[k], acc[k], w.z);
    }
    return out;
}

// The quad's texture coordinates: c[a] holds S, T, R or Q at pixels 0 to
// 3 (pixel i at x + (i & 1), y + (i >> 1)).
struct TexQuad {
    c: array<vec4<f32>, 4>,
}

// A pixel's position in each of the first `axes` axes in 1/256 of a
// texel, from its texture coordinates c (S, T, R, Q): S, T or R over Q
// times the axis's scale, floored. The others are zero.
fn tex_position(c: vec4<f32>, scale: array<f32, 4>, axes: u32) -> vec3<i32> {
    var p = vec3<i32>(0);
    let rq = div_exact(1.0, c[3]);
    for (var a = 0u; a < axes; a++) {
        let v = (c[a] * rq) * scale[a];
        p[a] = floor_i32(v);
    }
    return p;
}

// The magnitude of the difference of two positions per axis, which can
// pass i32: the larger less the smaller in u32.
fn tex_diff(p: vec3<i32>, q: vec3<i32>) -> vec3<u32> {
    return vec3<u32>(max(p, q)) - vec3<u32>(min(p, q));
}

// The post-filter: the scale and bias, then the colour table.
fn tex_post(ti: u32, c_in: vec4<i32>) -> vec4<i32> {
    var c = c_in;
    for (var k = 0u; k < 4u; k++) {
        let scale = tex_slots[ti].scale_bias[k * 2u];
        let bias = tex_slots[ti].scale_bias[k * 2u + 1u];
        if scale != 4096 || bias != 0 {
            // c * scale >> 12 in 64 bits on the CPU: c is below 2^24 and
            // scale a signed 16-bit value, so split c.
            let ch = c[k] >> 12u;
            let cl = c[k] & 0xfff;
            let v = ch * scale + ((cl * scale) >> 12u);
            c[k] = clamp(v + (bias << 12u), 0, FIXED_ONE - 1);
        }
    }
    if (tex_slots[ti].flags & TEX_TABLE) == 0u {
        return c;
    }
    for (var k = 0u; k < 3u; k++) {
        let idx = table_index(c[k]);
        let word = tex_slots[ti].colour_table[select(0u, 256u, k == 2u) + idx];
        c[k] = field12_full(select(word >> 12u, word, k == 1u));
    }
    if (tex_slots[ti].flags & TEX_TABLE_ALPHA) != 0u {
        c.w = field12_full(tex_slots[ti].colour_table[256u + table_index(c.w)]);
    }
    return c;
}

// (v * 255 + 2^23) >> 24 clamped to 0..255, for v in 12.12 (below 2^24
// after the clamp above, so the product fits with a split).
fn table_index(v: i32) -> u32 {
    let vv = clamp(v, 0, FIXED_ONE - 1);
    let vh = vv >> 12u;
    let vl = vv & 0xfff;
    // (vh * 2^12 + vl) * 255 + 2^23, over 2^24.
    let low = vl * 255 + (1 << 23);
    let total = vh * 255 + (low >> 12u);
    return u32(clamp(total >> 12u, 0, 255));
}

fn field12_full(v: u32) -> i32 {
    let x = i32(v & 0xfffu);
    return (x << 12u) + x;
}

fn fixed_clamp(v: i32) -> i32 {
    return clamp(v, 0, FIXED_ONE - 1);
}

// fixed::smul: a signed difference times a fraction in [0, ONE].
fn smul(d: i32, b: i32) -> i32 {
    return ((d >> 9u) * (b >> 9u)) >> 6u;
}

// The detail function at lambda, in 12.12.
fn detail_weight(ti: u32, lambda: i32) -> i32 {
    if lambda >= 0 {
        return i32((tex_slots[ti].detail_fn[0] >> 12u) & 0xfffu) << 12u;
    }
    let x = u32(-lambda);
    let i = x >> 3u;
    if i >= 255u {
        return i32((tex_slots[ti].detail_fn[255] >> 12u) & 0xfffu) << 12u;
    }
    let w0 = i32((tex_slots[ti].detail_fn[i] >> 12u) & 0xfffu);
    let w1 = i32((tex_slots[ti].detail_fn[i + 1u] >> 12u) & 0xfffu);
    let f = i32(x & 7u);
    return (w0 << 12u) + ((((w1 - w0) << 12u) * f) >> 3u);
}

// The level of detail by the diagonal rule: the largest difference along
// the quad's two diagonals (d0 between pixels 0 and 3, d1 between 1 and
// 2, per axis), in 1/256 of a base texel, scaled by the bias. Zero for a
// texture without levels.
fn tex_lambda(ti: u32, d0: vec3<u32>, d1: vec3<u32>) -> i32 {
    if (tex_slots[ti].flags & TEX_LOD) == 0u {
        return 0;
    }
    var rho = 0.0;
    for (var k = 0u; k < 3u; k++) {
        rho = max(rho, i64_to_f32(I64(d0[k], 0u)) * tex_slots[ti].lod_scale[k]);
        rho = max(rho, i64_to_f32(I64(d1[k], 0u)) * tex_slots[ti].lod_scale[k]);
    }
    let r = sat_u32(min(rho, 4294967295.0));
    return select(log2_q8(r) - (8 << 8u), LAMBDA_MIN, r == 0u);
}

// A quad's level of detail from its four pixels' texture coordinates, all
// in one lane (a line's pixel, whose quad is made up).
fn tex_lambda_quad(ti: u32, q: TexQuad) -> i32 {
    let axes = tex_slots[ti].axes;
    var p = array<vec3<i32>, 4>();
    for (var j = 0u; j < 4u; j++) {
        p[j] = tex_position(vec4<f32>(q.c[0][j], q.c[1][j], q.c[2][j], q.c[3][j]), tex_slots[ti].scale, axes);
    }
    return tex_lambda(ti, tex_diff(p[3], p[0]), tex_diff(p[2], p[1]));
}

// Sample one pixel whose texture coordinates are tc, whose position is
// p_in (tex_position()) and whose quad's level of detail is lambda_in
// (tex_lambda()): the texel in 12.12.
fn sample_pixel(ti: u32, tc: vec4<f32>, p_in: vec3<i32>, lambda_in: i32) -> vec4<i32> {
    let axes = tex_slots[ti].axes;
    var p = p_in;
    let has_lod = (tex_slots[ti].flags & TEX_LOD) != 0u;
    var lambda = lambda_in;
    let mn = tex_min(ti);
    let mg = tex_mag(ti);
    var linear = min_linear(mn);
    var first = 0u;
    var second = 0u;
    var frac = 0u;
    var detail = false;
    let range = tex_slots[ti].count - 1u;
    if has_lod {
        lambda = min(max(lambda, tex_slots[ti].min_lod), tex_slots[ti].max_lod);
        if lambda <= tex_slots[ti].mag_below {
            linear = mg != 0u;
            detail = mag_detail(mg);
        } else if min_mipmap(mn) {
            if min_nearest_level(mn) {
                first = select(min((u32(lambda) + 127u) >> 8u, range), 0u, lambda <= 128);
            } else {
                let rel = u32(min(max(lambda, 0), i32(range << 8u)));
                first = rel >> 8u;
                frac = rel & 255u;
                second = min(first + 1u, range);
            }
        }
    }
    for (var a = 0u; a < 3u; a++) {
        if tex_slots[ti].clamp[a] != NO_CLAMP {
            p[a] = min(p[a], tex_slots[ti].clamp[a]);
        }
    }
    let cls = tex_class(ti);
    let three = axes == 3u;
    let wraps = vec3<u32>(tex_wrap(ti, 0u), select(tex_wrap(ti, 1u), WRAP_CLAMP_TO_EDGE, axes == 1u), tex_wrap(ti, 2u));
    let l0 = tex_slots[ti].levels[0];
    // This pixel's taps on a level: the base position shifted down to the
    // level, through pos().
    var acc = level_sample(ti, first, l0, cls, p, linear, wraps, three);
    if frac != 0u {
        let other = level_sample(ti, second, l0, cls, p, linear, wraps, three);
        for (var k = 0u; k < 4u; k++) {
            acc[k] = blend8(acc[k], other[k], frac);
        }
    }
    var c = vec4<i32>(to_fixed(acc.x), to_fixed(acc.y), to_fixed(acc.z), to_fixed(acc.w));
    if detail {
        let dscale = array<f32, 4>(tex_slots[ti].detail_scale[0], tex_slots[ti].detail_scale[1], 0.0, 0.0);
        let d = tex_position(tc, dscale, 2u);
        let lv = tex_slots[ti].detail;
        let px = pos(d[0], lv.wl, WRAP_REPEAT, true);
        let py = pos(d[1], lv.hl, WRAP_REPEAT, true);
        let dt = filter_level(ti, lv, cls, 2u, vec3<i32>(px.at, py.at, 0), vec3<u32>(px.w, py.w, 0u), true, vec3<u32>(WRAP_REPEAT, WRAP_REPEAT, WRAP_REPEAT), false);
        let weight = detail_weight(ti, lambda);
        let add = (tex_slots[ti].sampler0 & 0x80000000u) == 0u;
        for (var k = 0u; k < 4u; k++) {
            if (k == 3u && mg == 4u) || (k != 3u && mg == 3u) {
                continue;
            }
            let o = to_fixed(dt[k]);
            let signed = smul(2 * o - FIXED_ONE, weight);
            var v: i32;
            if add {
                v = c[k] + signed;
            } else {
                v = c[k] + smul(signed, c[k]);
            }
            c[k] = fixed_clamp(v);
        }
    }
    if (tex_slots[ti].flags & TEX_POST) != 0u {
        c = tex_post(ti, c);
    }
    return c;
}

// Level `li` at pixel i's base position.
fn level_sample(ti: u32, li: u32, l0: TexLevel, cls: u32, p: vec3<i32>, linear: bool, wraps: vec3<u32>, three: bool) -> vec4<u32> {
    let lv = tex_slots[ti].levels[li];
    let n = select(select(2u, 1u, tex_slots[ti].axes == 1u), 3u, three);
    var at = vec3<i32>(0);
    var w = vec3<u32>(0u);
    let shifts = vec3<u32>(l0.wl - lv.wl, l0.hl - lv.hl, l0.dl - lv.dl);
    let sizes = vec3<u32>(lv.wl, lv.hl, lv.dl);
    for (var a = 0u; a < n; a++) {
        let r = pos(p[a] >> shifts[a], sizes[a], wraps[a], linear);
        at[a] = r.at;
        w[a] = r.w;
    }
    return filter_level(ti, lv, cls, tex_slots[ti].axes, at, w, linear, wraps, (tex_slots[ti].flags & TEX_STORED_BORDER) != 0u);
}

// fixed::mul: two colours' product, 4096.0 times 4096.0 is 4096.0.
fn fixed_mul(a: i32, b: i32) -> i32 {
    return i32(((u32(a) >> 8u) * (u32(b) >> 8u)) >> 8u);
}

fn fixed_lerp(a: i32, b: i32, f: i32) -> i32 {
    return a + smul(b - a, f);
}

// Which of a texel's components the environment takes.
fn takes(cls: u32, k: u32) -> bool {
    let alpha = k == 3u;
    return !((!alpha && cls == CLASS_ALPHA) || (alpha && (cls == CLASS_RGB || cls == CLASS_LUMINANCE)));
}

// ShaderState::apply12x4: the texel by the environment into the colour,
// in 12.12.
fn tex_env(ti: u32, texel: vec4<i32>, rgba_in: vec4<i32>) -> vec4<i32> {
    let cls = tex_class(ti);
    var rgba = rgba_in;
    for (var k = 0u; k < 4u; k++) {
        if !takes(cls, k) {
            continue;
        }
        let alpha = k == 3u;
        let tx = texel[k];
        let c = rgba[k];
        var r: i32;
        switch tex_slots[ti].env {
            case ENV_MODULATE: { r = fixed_mul(c, tx); }
            case ENV_ADD: {
                if alpha && cls != CLASS_INTENSITY {
                    r = fixed_mul(c, tx);
                } else {
                    r = fixed_clamp(c + tx);
                }
            }
            case ENV_DECAL: {
                if alpha {
                    r = c;
                } else if cls == CLASS_RGB {
                    r = tx;
                } else {
                    r = fixed_lerp(c, tx, texel.w);
                }
            }
            case ENV_BLEND: {
                if alpha && cls != CLASS_INTENSITY {
                    r = fixed_mul(c, tx);
                } else {
                    r = fixed_lerp(c, tex_slots[ti].env_colour[k], tx);
                }
            }
            default: { r = tx; }
        }
        rgba[k] = r;
    }
    return rgba;
}
