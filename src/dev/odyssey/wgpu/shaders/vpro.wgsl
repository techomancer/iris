// The VPro raster unit's hooks for the common raster pass: the quad ring's
// three commands (pipe/ring.rs) as records. A triangle record is the
// walk's setup (raster::Triangle) and the shading state of its record
// (setup::Tri); a fragment record is a quad the CPU shaded itself; a
// rectangle record is a clear's. The arithmetic follows pipe/raster.rs,
// pipe/interp.rs, pipe/fixed.rs and pipe/backend.rs operation for
// operation; where a product feeds a sum, it is multiplied by a one the
// compiler cannot see through (`nc`), so no driver fuses it into a
// multiply-add the CPU does not do, and quotients go through
// `div_exact` (src/wgpu/shaders/float.wgsl), since a GPU's division need
// not be correctly rounded.
//
// Record layout: src/dev/odyssey/wgpu/mod.rs `Rec` writes it.

#include "i64.wgsl"
#include "float.wgsl"
#include "backend.wgsl"
#include "texture.wgsl"

const REC_TRI: u32 = 1u;
const REC_FRAG: u32 = 2u;
const REC_RECT: u32 = 3u;
const REC_BLIT: u32 = 4u;

const REC_WORDS: u32 = 84u;

struct Rec {
    kind: u32,
    slot: u32,
    flags: u32,
    // 1.0 as bits: the one that stops contraction.
    one: u32,
    w: array<u32, REC_WORDS>,
}

@group(1) @binding(0) var<storage, read> recs: array<Rec>;

// Triangle flags.
const TRI_SMOOTH: u32 = 1u;
const TRI_TEXTURED: u32 = 2u;
const TRI_SPECULAR: u32 = 4u;

// Triangle word offsets.
const T_E0: u32 = 0u;      // 3 x I64: e0, e1, e2 at the centre of plane pixel (0, 0)
const T_DX: u32 = 6u;      // 3 x I64
const T_DY: u32 = 12u;     // 3 x I64
const T_NEED: u32 = 18u;   // bit k: edge k needs 1 (not owning)
const T_INV_AREA: u32 = 19u;
const T_Z: u32 = 20u;      // z0, z1 - z0, z2 - z0
const T_D: u32 = 23u;      // d0, d1 - d0, d2 - d0
const T_IW: u32 = 26u;     // 1/w at the vertices
const T_SLACK: u32 = 29u;  // depth slack, near, far - near
const T_RGBA12: u32 = 32u; // 3 x 4 in 12.12
const T_FLAT12: u32 = 44u; // 4
const T_STEPS: u32 = 48u;  // polygon offset, clamped to +-2^25
const T_BOX: u32 = 49u;    // x0, y0, x1, y1 in plane pixels, inclusive
const T_ORIGIN: u32 = 53u; // ox, oy
const T_ALPHA: u32 = 55u;  // 0x41e: the alpha test's word
const T_MODE: u32 = 56u;   // 0x404: its enable
const T_TEX: u32 = 57u;    // S/W, T/W, R/W, Q/W at the three vertices
const T_TEXSLOT: u32 = 69u;
const T_SPEC12: u32 = 70u; // 3 x 3 in 12.12
const T_FLAT_SPEC: u32 = 79u;

// Fragment word offsets.
const F_X: u32 = 0u;
const F_Y: u32 = 1u;
const F_COV: u32 = 2u;
const F_ZQ: u32 = 3u;      // 4
const F_RGBA: u32 = 7u;    // 4 pixels x 2 words: r | g << 16, b | a << 16

// Rectangle word offsets.
const R_BOX: u32 = 0u;     // x0, y0, x1, y1
const R_COLOUR: u32 = 4u;
const R_COLOUR_MASK: u32 = 5u;
const R_DEPTH: u32 = 6u;
const R_STENCIL: u32 = 7u; // value | mask << 8
const R_ID: u32 = 8u;      // value | mask << 8

// Blit word offsets: a row run of pixels from the stage buffer into one
// of the slot's views, under a mask, through the window's tests or not.
const B_BOX: u32 = 0u;      // x0, y0, x1, y1 (one row)
const B_OFF: u32 = 4u;      // the first pixel's word in the stage
const B_MASK: u32 = 5u;     // the bits stored (all ones: a plain store)
const B_VIEW: u32 = 6u;     // the slot's view index
const B_OP: u32 = 7u;       // 0, or 0x100 | the logical operation's code
const B_AND: u32 = 8u;      // the operation's result is and-ed and or-ed
const B_OR: u32 = 9u;
const B_OWN: u32 = 10u;     // the record's own view, with BLIT_OWN_VIEW
const BLIT_WINDOW: u32 = 1u;
const BLIT_OWN_VIEW: u32 = 2u;

// Pixels the blits of the batch store (wgpu/mod.rs `GpuRaster::stage`).
@group(2) @binding(2) var<storage, read> stage: array<u32>;

const MODE_ALPHA_TEST: u32 = 1u << 14u;
const MODE_FLAT_SHADE: u32 = 1u << 12u;

const W_ONE: i32 = 1 << 15;
const FIXED_MAX: i32 = (1 << 24) - 1;
const DEPTH_2S: f32 = 8589901824.0;
const DEPTH_MIN_VALUE: f32 = 1.0 / 2147483648.0;

var<private> nc: f32;

// Record words, read in place: a record or slot copied by value sits in
// each lane's private memory, and enough of that gave wrong results on
// RADV (clanker/wgpu/read-storage-in-place.md).
fn ru(pi: u32, i: u32) -> u32 {
    return recs[pi].w[i];
}

fn rf(pi: u32, i: u32) -> f32 {
    return bitcast<f32>(recs[pi].w[i]);
}

fn ri(pi: u32, i: u32) -> i32 {
    return bitcast<i32>(recs[pi].w[i]);
}

fn r64(pi: u32, i: u32) -> I64 {
    return I64(recs[pi].w[i], recs[pi].w[i + 1u]);
}

// x * 1.0 through a value the compiler cannot fold: the product is
// rounded on its own before it joins a sum.
fn p(x: f32) -> f32 {
    return x * nc;
}

// Rust's `as i32`: truncation, saturating, NaN to 0.
fn sat_i32(x: f32) -> i32 {
    let c = clamp(x, -2147483648.0, 2147483520.0);
    return select(i32(c), 0, x != x);
}

// word_for_fraction32: the 24-bit depth word of a window depth.
fn depth_word(z_in: f32) -> u32 {
    let z = select(select(0.0, z_in, z_in > 0.0), 1.0, z_in > 1.0);
    let v = z * DEPTH_2S;
    let bits = bitcast<u32>(v);
    let exp = (((bits >> 23u) & 0xffu) + 31u) - 127u;
    let mantissa = (bits >> 5u) & 0x3ffffu;
    let word = ((exp << 18u) | mantissa) & DEPTH24_MAX;
    return select(0u, word, v >= DEPTH_MIN_VALUE);
}

// fixed::interpolate: weights p (0.15) over vertex values c (12.12).
fn interpolate(p0: i32, p1: i32, p2: i32, c0: i32, c1: i32, c2: i32) -> i32 {
    let s = (c0 >> 8u) * p0 + (c1 >> 8u) * p1 + (c2 >> 8u) * p2;
    return s >> 7u;
}

fn colour12(c: i32) -> u32 {
    return u32(clamp(c, 0, FIXED_MAX) >> 12u);
}

fn tri(pi: u32, si: u32, px: i32, py: i32) {
    // The walk's bounds, every clip already folded in.
    if px < ri(pi, T_BOX) || px > ri(pi, T_BOX + 2u) || py < ri(pi, T_BOX + 1u) || py > ri(pi, T_BOX + 3u) {
        return;
    }
    // The three edge values at the pixel centre, exact.
    var e: array<I64, 3>;
    let need = ru(pi, T_NEED);
    for (var k = 0u; k < 3u; k++) {
        var v = r64(pi, T_E0 + 2u * k);
        v = i64_add(v, i64_mul_u16(r64(pi, T_DX + 2u * k), u32(px)));
        v = i64_add(v, i64_mul_u16(r64(pi, T_DY + 2u * k), u32(py)));
        if !i64_ge_need(v, (need >> k) & 1u) {
            return;
        }
        e[k] = v;
    }
    if !owned(si, px, py) {
        return;
    }
    let inv_area = rf(pi, T_INV_AREA);
    let l1 = p(i64_to_f32(e[1]) * inv_area);
    let l2 = p(i64_to_f32(e[2]) * inv_area);
    // The pixel's quad (2x2, even window coordinates) and its index in
    // it: the texture unit reads the whole quad's positions.
    let wx = px - ri(pi, T_ORIGIN);
    let wy = py - ri(pi, T_ORIGIN + 1u);
    let qi = u32(wx & 1) | (u32(wy & 1) << 1u);

    // Triangle::depth4: the planes, the ratio, the slack clamps, the range
    // and the word with the polygon offset.
    let zz = (rf(pi, T_Z) + p(l1 * rf(pi, T_Z + 1u))) + p(l2 * rf(pi, T_Z + 2u));
    let dd = (rf(pi, T_D) + p(l1 * rf(pi, T_D + 1u))) + p(l2 * rf(pi, T_D + 2u));
    var z = div_exact(zz, dd);
    let slack = rf(pi, T_SLACK);
    if z < 0.0 && z >= -slack {
        z = 0.0;
    } else if z > 1.0 && z <= 1.0 + slack {
        z = 1.0;
    }
    if !(z >= 0.0 && z <= 1.0) {
        return;
    }
    var zw = rf(pi, T_SLACK + 1u) + p(rf(pi, T_SLACK + 2u) * z);
    zw = select(select(0.0, zw, zw > 0.0), 1.0, zw >= 1.0);
    let zq = u32(clamp(i32(depth_word(zw)) + ri(pi, T_STEPS), 0, i32(DEPTH24_MAX)));

    // The polygon stipple, in window pixels.
    var background = false;
    if (slots[si].flags & SLOT_STIPPLE) != 0u {
        let wx = u32(px - ri(pi, T_ORIGIN));
        let wy = u32(py - ri(pi, T_ORIGIN + 1u));
        if (slots[si].pattern[wy & 31u] & (1u << (wx & 31u))) == 0u {
            if (slots[si].flags & SLOT_BACKGROUND) == 0u {
                return;
            }
            background = true;
        }
    }

    // The colour: the provoking vertex's, or interpolated by the
    // perspective-correct weights (interp::weights4); the separate
    // specular likewise.
    var c: vec4<i32>;
    var spec = vec3<i32>(ri(pi, T_FLAT_SPEC), ri(pi, T_FLAT_SPEC + 1u), ri(pi, T_FLAT_SPEC + 2u));
    let specular = (recs[pi].flags & TRI_SPECULAR) != 0u;
    if background {
        c = slots[si].bg12;
        spec = vec3<i32>(0);
    } else if (recs[pi].flags & TRI_SMOOTH) != 0u {
        let iw0 = rf(pi, T_IW);
        let w1 = p(l1 * rf(pi, T_IW + 1u));
        let w2 = p(l2 * rf(pi, T_IW + 2u));
        let sum = (p(((1.0 - l1) - l2) * iw0) + w1) + w2;
        if sum == 0.0 || !(abs(sum) < bitcast<f32>(0x7f800000u)) {
            return;
        }
        let rr = div_exact(f32(W_ONE), sum);
        let p1 = clamp(sat_i32(p(w1 * rr) + 0.5), 0, W_ONE);
        let rest = W_ONE - p1;
        let p2 = clamp(sat_i32(p(w2 * rr) + 0.5), 0, rest);
        let p0 = rest - p2;
        for (var k = 0u; k < 4u; k++) {
            c[k] = interpolate(p0, p1, p2, ri(pi, T_RGBA12 + k), ri(pi, T_RGBA12 + 4u + k), ri(pi, T_RGBA12 + 8u + k));
        }
        if specular {
            for (var k = 0u; k < 3u; k++) {
                spec[k] = interpolate(p0, p1, p2, ri(pi, T_SPEC12 + k), ri(pi, T_SPEC12 + 3u + k), ri(pi, T_SPEC12 + 6u + k));
            }
        }
    } else {
        c = vec4<i32>(ri(pi, T_FLAT12), ri(pi, T_FLAT12 + 1u), ri(pi, T_FLAT12 + 2u), ri(pi, T_FLAT12 + 3u));
    }

    // The texture: the quad's coordinates from its four pixels'
    // barycentrics (interp::TriTex::quad), sampled for this pixel, then
    // the environment.
    if (recs[pi].flags & TRI_TEXTURED) != 0u {
        var tq: TexQuad;
        for (var j = 0u; j < 4u; j++) {
            let dx = i32(j & 1u) - i32(qi & 1u);
            let dy = i32(j >> 1u) - i32(qi >> 1u);
            var l1j = l1;
            var l2j = l2;
            if j != qi {
                var e1 = e[1];
                var e2 = e[2];
                if dx > 0 {
                    e1 = i64_add(e1, r64(pi, T_DX + 2u));
                    e2 = i64_add(e2, r64(pi, T_DX + 4u));
                } else if dx < 0 {
                    e1 = i64_add(e1, i64_neg(r64(pi, T_DX + 2u)));
                    e2 = i64_add(e2, i64_neg(r64(pi, T_DX + 4u)));
                }
                if dy > 0 {
                    e1 = i64_add(e1, r64(pi, T_DY + 2u));
                    e2 = i64_add(e2, r64(pi, T_DY + 4u));
                } else if dy < 0 {
                    e1 = i64_add(e1, i64_neg(r64(pi, T_DY + 2u)));
                    e2 = i64_add(e2, i64_neg(r64(pi, T_DY + 4u)));
                }
                l1j = p(i64_to_f32(e1) * inv_area);
                l2j = p(i64_to_f32(e2) * inv_area);
            }
            for (var a = 0u; a < 4u; a++) {
                let v0 = rf(pi, T_TEX + 3u * a);
                let v1 = rf(pi, T_TEX + 3u * a + 1u);
                let v2 = rf(pi, T_TEX + 3u * a + 2u);
                tq.c[a][j] = (v0 + p(l1j * (v1 - v0))) + p(l2j * (v2 - v0));
            }
        }
        let ti = ru(pi, T_TEXSLOT);
        let texel = sample_quad(ti, tq, qi);
        c = tex_env(ti, texel, c);
    }
    if specular {
        for (var k = 0u; k < 3u; k++) {
            c[k] = fixed_clamp(c[k] + spec[k]);
        }
    }

    // The alpha test on the 12-bit alpha.
    if (ru(pi, T_MODE) & MODE_ALPHA_TEST) != 0u {
        let a = ru(pi, T_ALPHA);
        if !compare((a >> 12u) & 7u, colour12(c.w), a & 0xfffu) {
            return;
        }
    }
    let src = vec4<u32>(colour12(c.x), colour12(c.y), colour12(c.z), colour12(c.w));
    fragment_q(backend_of(si), si, px, py, zq, src);
}

fn frag(pi: u32, si: u32, px: i32, py: i32) {
    let x = ri(pi, F_X);
    let y = ri(pi, F_Y);
    let dx = px - x;
    let dy = py - y;
    if dx < 0 || dx > 1 || dy < 0 || dy > 1 {
        return;
    }
    let i = u32(dx) + 2u * u32(dy);
    if (ru(pi, F_COV) & (1u << i)) == 0u {
        return;
    }
    let a = ru(pi, F_RGBA + 2u * i);
    let b = ru(pi, F_RGBA + 2u * i + 1u);
    let src = vec4<u32>(a & 0xffffu, a >> 16u, b & 0xffffu, b >> 16u);
    fragment_q(backend_of(si), si, px, py, ru(pi, F_ZQ + i), src);
}

fn rect(pi: u32, si: u32, px: i32, py: i32) {
    if px < ri(pi, R_BOX) || px > ri(pi, R_BOX + 2u) || py < ri(pi, R_BOX + 1u) || py > ri(pi, R_BOX + 3u) {
        return;
    }
    clear_pixel(si, recs[pi].flags, px, py, ru(pi, R_COLOUR), ru(pi, R_COLOUR_MASK), ru(pi, R_DEPTH), ru(pi, R_STENCIL), ru(pi, R_ID));
}

fn blit(pi: u32, si: u32, px: i32, py: i32) {
    if px < ri(pi, B_BOX) || px > ri(pi, B_BOX + 2u) || py < ri(pi, B_BOX + 1u) || py > ri(pi, B_BOX + 3u) {
        return;
    }
    if (recs[pi].flags & BLIT_WINDOW) != 0u && !owned(si, px, py) {
        return;
    }
    var v: View;
    if (recs[pi].flags & BLIT_OWN_VIEW) != 0u {
        v = View(ru(pi, B_OWN), ru(pi, B_OWN + 1u), ru(pi, B_OWN + 2u), ru(pi, B_OWN + 3u));
    } else {
        v = slots[si].views[ru(pi, B_VIEW)];
    }
    let w = stage[ru(pi, B_OFF) + u32(px - ri(pi, B_BOX))];
    let mask = ru(pi, B_MASK);
    let op = ru(pi, B_OP);
    if op == 0u && mask == 0xffffffffu {
        view_store(v, px, py, w);
        return;
    }
    // The destination is read once: the operation and a byte plane's mask
    // both see the same old word (Board::paint_integer, buffer_store).
    let old = view_load(v, px, py);
    if !old.ok {
        return;
    }
    var val = w;
    if op != 0u {
        val = (logic_op(op & 0xfu, w, old.v) & ru(pi, B_AND)) | ru(pi, B_OR);
    }
    if mask == 0xffffffffu {
        view_store(v, px, py, val);
    } else {
        view_store(v, px, py, (old.v & ~mask) | (val & mask));
    }
}

fn arch_tile_begin(tile: u32, px: i32, py: i32, lane: u32) {
}

fn arch_prim(pi: u32, px: i32, py: i32, lane: u32) {
    let si = recs[pi].slot;
    nc = bitcast<f32>(recs[pi].one);
    switch recs[pi].kind {
        case REC_TRI: { tri(pi, si, px, py); }
        case REC_FRAG: { frag(pi, si, px, py); }
        case REC_RECT: { rect(pi, si, px, py); }
        case REC_BLIT: { blit(pi, si, px, py); }
        default: {}
    }
}

fn arch_tile_end(tile: u32, px: i32, py: i32, lane: u32) {
}
