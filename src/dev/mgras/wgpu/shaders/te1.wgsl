// IMPACT's texture engine (TE1) on the GPU: te1.rs's `Sampler` (texel
// addressing in TRAM, wrap and clamp, borders, the 8-bit bilinear weights,
// the mip blend), the reciprocal unit and level of detail of fixed.rs,
// `Rss::textured`, `tex_env` and `Rss::fogged`, operation for operation.
// The sampler comes resolved in the slot (`Te1::sampler` on the CPU).
//
// TRAM is its own buffer: bytes, two nibbles each, the low nibble first.
// Only the CPU's texture loader writes it (uploaded in FIFO order), so the
// kernels read it plainly.

#include "pp1.wgsl"

@group(2) @binding(2) var<storage, read> tram: array<u32>;
// fixed::RECIP (1024 words), then log2(1 + i/256) in Q8 (256), then
// round(256 2^(i/256)) (256).
@group(2) @binding(4) var<storage, read> tables: array<u32>;

const LOD_NONE: i32 = -(1 << 24);
const TM2_MAG_LINEAR: u32 = 128u;
const TM2_MIN_LINEAR: u32 = 256u;
const TM2_MIP_LINEAR: u32 = 64u;
const TM2_CLAMP_S: u32 = 2048u;
const TM2_CLAMP_T: u32 = 4096u;
const TM2_GL_CLAMP: u32 = 16384u;
const TM2_NO_BORDER: u32 = 65536u;

fn nibble(a: u32) -> u32 {
    let byte = a >> 1u;
    return (tram[byte >> 2u] >> (8u * (byte & 3u) + 4u * (a & 1u))) & 0xfu;
}

// A component of d nibbles at nibble address a, widened to 12 bits.
fn component(a: u32, d: u32) -> u32 {
    var v = 0u;
    for (var i = 0u; i < d; i++) {
        v |= nibble((a + i) % TRAM_NIBBLES) << (4u * i);
    }
    if d == 1u {
        return v * 0x111u;
    }
    if d == 2u {
        return (v << 4u) | (v >> 4u);
    }
    return v;
}

fn cell_nibbles(d: u32) -> u32 {
    return select(select(16u, 8u, d == 2u), 4u, d == 1u);
}

fn comp_offset(sel: u32, nc: u32, c: u32, d: u32) -> u32 {
    var slot = 0u;
    if nc == 1u {
        slot = sel & 3u;
    } else if nc == 2u {
        slot = (sel & 1u) * 2u;
    }
    return min(slot + c, 3u) * d;
}

fn small_offset(n: u32) -> u32 {
    var off = 0u;
    var k = 16u;
    while k > n {
        off += k * k;
        k /= 2u;
    }
    return off;
}

fn small_border_offset(n: u32) -> u32 {
    var off = 0u;
    var k = 16u;
    while k > n {
        off += 4u * k + 4u;
        k /= 2u;
    }
    return off;
}

fn texel_addr(page: u32, w: u32, h: u32, s: u32, t: u32, tn: u32) -> u32 {
    let side = max(w, h);
    let off = select(0u, small_offset(side), side <= 16u);
    return (page * TRAM_PAGE_NIBBLES + (off + t * w + s) * tn) % TRAM_NIBBLES;
}

// s in -1..=w, t in -1..=h, on the border.
fn border_addr(page: u32, w: u32, h: u32, s: i32, t: i32, tn: u32) -> u32 {
    let wi = i32(w);
    let hi = i32(h);
    var idx: i32;
    if t < 0 {
        idx = s + 1;
    } else if t >= hi {
        idx = wi + 2 + s + 1;
    } else if s < 0 {
        idx = 2 * (wi + 2) + t;
    } else {
        idx = 2 * (wi + 2) + hi + t;
    }
    let side = max(w, h);
    let off = select(0u, small_border_offset(side), side <= 16u);
    return (page * TRAM_PAGE_NIBBLES + TRAM_PAGE_NIBBLES / 2u + (off + u32(idx)) * tn) % TRAM_NIBBLES;
}

// A coordinate along an axis of n texels (a power of two): clamped to
// -1..=n, or wrapped.
fn axis(v: I64, n: u32, clamp_on: bool) -> i32 {
    if clamp_on {
        return i64_low_i32(i64_clamp32(v, -1, i32(n)));
    }
    return i32(v.lo & (n - 1u));
}

// `Sampler::texel`: texel (i, j) of `level`, 12-bit components.
fn smp_texel(si: u32, level: u32, i: I64, j: I64) -> vec4<u32> {
    let ls = sr(si, SL_LS);
    let lt = sr(si, SL_LT);
    let w = 1u << select(0u, ls - level, ls > level);
    let h = 1u << select(0u, lt - level, lt > level);
    let m2 = sr(si, SL_MODE2);
    let s = axis(i, w, (m2 & TM2_CLAMP_S) != 0u);
    let t = axis(j, h, (m2 & TM2_CLAMP_T) != 0u);
    let nc = sr(si, SL_NC);
    let sel = sr(si, SL_SEL);
    let d = sr(si, SL_D);
    let tn = cell_nibbles(d);
    var a: u32;
    if s < 0 || t < 0 || s >= i32(w) || t >= i32(h) {
        if (m2 & TM2_NO_BORDER) != 0u {
            return vec4<u32>(sr(si, SL_BORDER), sr(si, SL_BORDER + 1u), sr(si, SL_BORDER + 2u), sr(si, SL_BORDER + 3u));
        }
        a = border_addr(sr(si, SL_BPAGES + (level & 15u)), w, h, s, t, tn);
    } else {
        a = texel_addr(sr(si, SL_PAGES + (level & 15u)), w, h, u32(s), u32(t), tn);
    }
    var c = vec4<u32>(0u);
    for (var k = 0u; k < nc; k++) {
        c[k] = component(a + comp_offset(sel, nc, k, d), d);
    }
    return c;
}

const Q31_ONE: I64 = I64(0x80000000u, 0u);

fn i64_clamp_q31(v: I64) -> I64 {
    if i64_is_neg(v) {
        return I64(0u, 0u);
    }
    if i64_lt(Q31_ONE, v) {
        return Q31_ONE;
    }
    return v;
}

// `Sampler::level`: a level at (s, t) in Q31, nearest or bilinear, 12.16.
fn smp_level(si: u32, level: u32, s_in: I64, t_in: I64, linear: bool) -> vec4<i32> {
    let ls = sr(si, SL_LS);
    let lt = sr(si, SL_LT);
    let lw = select(0u, ls - level, ls > level);
    let lh = select(0u, lt - level, lt > level);
    let m2 = sr(si, SL_MODE2);
    let gl_clamp = (m2 & TM2_GL_CLAMP) != 0u;
    let cs = (m2 & TM2_CLAMP_S) != 0u;
    let ct = (m2 & TM2_CLAMP_T) != 0u;
    var s = s_in;
    var t = t_in;
    if gl_clamp && cs {
        s = i64_clamp_q31(s);
    }
    if gl_clamp && ct {
        t = i64_clamp_q31(t);
    }
    let u = i64_shr(s, 15u - lw);
    let v = i64_shr(t, 15u - lh);
    if !linear {
        var iu = i64_shr(u, 16u);
        var iv = i64_shr(v, 16u);
        if cs && gl_clamp && i64_lt(i64_from_i32(i32((1u << lw) - 1u)), iu) {
            iu = i64_from_i32(i32((1u << lw) - 1u));
        }
        if ct && gl_clamp && i64_lt(i64_from_i32(i32((1u << lh) - 1u)), iv) {
            iv = i64_from_i32(i32((1u << lh) - 1u));
        }
        let c = smp_texel(si, level, iu, iv);
        return vec4<i32>(bitcast<vec4<i32>>(c << vec4<u32>(16u)));
    }
    let uu = i64_sub(u, I64(0x8000u, 0u));
    let vv = i64_sub(v, I64(0x8000u, 0u));
    let i = i64_shr(uu, 16u);
    let j = i64_shr(vv, 16u);
    let a = i32((uu.lo >> 8u) & 0xffu);
    let b = i32((vv.lo >> 8u) & 0xffu);
    let i1 = i64_add(i, I64(1u, 0u));
    let j1 = i64_add(j, I64(1u, 0u));
    let t00 = vec4<i32>(smp_texel(si, level, i, j));
    let t10 = vec4<i32>(smp_texel(si, level, i1, j));
    let t01 = vec4<i32>(smp_texel(si, level, i, j1));
    let t11 = vec4<i32>(smp_texel(si, level, i1, j1));
    let w00 = (256 - a) * (256 - b);
    let w10 = a * (256 - b);
    let w01 = (256 - a) * b;
    let w11 = a * b;
    return t00 * w00 + t10 * w10 + t01 * w01 + t11 * w11;
}

// `Sampler::sample` at level of detail lambda (Q8).
fn smp_sample(si: u32, s: I64, t: I64, lambda: i32) -> vec4<i32> {
    let m = sr(si, SL_MODE2);
    if lambda <= 0 {
        let base = smp_level(si, 0u, s, t, (m & TM2_MAG_LINEAR) != 0u);
        if sr(si, SL_DETAIL) != 0u {
            return smp_detail(si, base, s, t, lambda);
        }
        return base;
    }
    let linear = (m & TM2_MIN_LINEAR) != 0u;
    let max_level = sr(si, SL_MAXLEVEL);
    if max_level == 0u {
        return smp_level(si, 0u, s, t, linear);
    }
    let top = i32(max_level);
    if (m & TM2_MIP_LINEAR) == 0u {
        let l = clamp(((lambda + 128 + 255) >> 8u) - 1, 0, top);
        return smp_level(si, u32(l), s, t, linear);
    }
    let l0 = clamp(lambda >> 8u, 0, top);
    let l1 = min(l0 + 1, top);
    let f = clamp(lambda - (l0 << 8u), 0, 256);
    let a = smp_level(si, u32(l0), s, t, linear);
    let b = smp_level(si, u32(l1), s, t, linear);
    var out = vec4<i32>(0);
    for (var k = 0u; k < 4u; k++) {
        out[k] = a[k] + i64_low_i32(i64_shr(i64_mul_i32(b[k] - a[k], f), 8u));
    }
    return out;
}

// `Sampler::detail_weight`: F at lambda (<= 0, Q8) in 12.16.
fn detail_weight(si: u32, lambda: i32) -> i32 {
    let e = max(lambda, -(1 << 24)) + 5 * 256;
    var x = tables[1280u + u32(e & 0xff)];
    let k = e >> 8u;
    if k >= 0 {
        x = x << u32(min(k, 8));
    } else {
        x = x >> u32(min(-k, 31));
    }
    var i = 31u;
    var f = 0u;
    if x < (31u << 8u) {
        i = x >> 8u;
        f = x & 0xffu;
    }
    var t1 = 0u;
    if i < 31u {
        t1 = sr(si, SL_DTABLE + i + 1u);
    }
    let w = sr(si, SL_DTABLE + i) * (256u - f) + t1 * f;
    return i32(w * 33825u);
}

// A detail coordinate: Q31 to Q16 detail texels.
fn detail_coord(v: I64, sh: u32) -> I64 {
    if sh <= 15u {
        return i64_shr(v, 15u - sh);
    }
    return u64_shl(v, sh - 15u);
}

// A texel cell of tn nibbles (4, 8 or 16) at nibble address a, a multiple
// of tn: its bits, the first nibble lowest (as `nibble` numbers them).
fn cell_bits(a: u32, tn: u32) -> vec2<u32> {
    let w = a >> 3u;
    let lo = tram[w];
    if tn == 16u {
        return vec2<u32>(lo, tram[w + 1u]);
    }
    if tn == 8u {
        return vec2<u32>(lo, 0u);
    }
    return vec2<u32>((lo >> (4u * (a & 7u))) & 0xffffu, 0u);
}

// `component` of a cell from `cell_bits`: d nibbles at nibble o, widened.
fn cell_comp(c: vec2<u32>, o: u32, d: u32) -> u32 {
    let b = 4u * o;
    var v: u32;
    if b >= 32u {
        v = c.y >> (b - 32u);
    } else if b == 0u {
        v = c.x;
    } else {
        v = (c.x >> b) | (c.y << (32u - b));
    }
    v &= (1u << (4u * d)) - 1u;
    if d == 1u {
        return v * 0x111u;
    }
    if d == 2u {
        return (v << 4u) | (v >> 4u);
    }
    return v;
}

// Texel (i, j) of the detail image (the left half of a 2n x n level at
// `page`), its components widened to 12 bits.
fn detail_texel(page: u32, n: u32, i: u32, j: u32, nc: u32, sel: u32, d: u32) -> vec4<i32> {
    let tn = cell_nibbles(d);
    let c = cell_bits((page * TRAM_PAGE_NIBBLES + (j * 2u * n + i) * tn) % TRAM_NIBBLES, tn);
    var t = vec4<i32>(0);
    t.x = i32(cell_comp(c, comp_offset(sel, nc, 0u, d), d));
    if nc > 1u {
        t.y = i32(cell_comp(c, comp_offset(sel, nc, 1u, d), d));
    }
    if nc > 2u {
        t.z = i32(cell_comp(c, comp_offset(sel, nc, 2u, d), d));
    }
    if nc > 3u {
        t.w = i32(cell_comp(c, comp_offset(sel, nc, 3u, d), d));
    }
    return t;
}

// `Sampler::detail`: base + F (2 detail - 1), the detail image bilinear and
// repeating at the level-0 coordinate scaled by 2^-L.
fn smp_detail(si: u32, base: vec4<i32>, s: I64, t: I64, lambda: i32) -> vec4<i32> {
    let page = sr(si, SL_DETAIL + 1u);
    let n = 1u << sr(si, SL_DETAIL + 2u);
    let uu = i64_sub(detail_coord(s, sr(si, SL_DETAIL + 3u)), I64(0x8000u, 0u));
    let vv = i64_sub(detail_coord(t, sr(si, SL_DETAIL + 4u)), I64(0x8000u, 0u));
    let i = i64_shr(uu, 16u).lo & (n - 1u);
    let j = i64_shr(vv, 16u).lo & (n - 1u);
    let a = i32((uu.lo >> 8u) & 0xffu);
    let b = i32((vv.lo >> 8u) & 0xffu);
    let i1 = (i + 1u) & (n - 1u);
    let j1 = (j + 1u) & (n - 1u);
    let nc = sr(si, SL_NC);
    let sel = sr(si, SL_SEL);
    let d = sr(si, SL_D);
    let td = detail_texel(page, n, i, j, nc, sel, d) * ((256 - a) * (256 - b))
        + detail_texel(page, n, i1, j, nc, sel, d) * (a * (256 - b))
        + detail_texel(page, n, i, j1, nc, sel, d) * ((256 - a) * b)
        + detail_texel(page, n, i1, j1, nc, sel, d) * (a * b);
    let f = detail_weight(si, lambda);
    let x = 2 * td - vec4<i32>(FIX_ONE);
    return clamp(base + vec4<i32>(fmul(f, x.x), fmul(f, x.y), fmul(f, x.z), fmul(f, x.w)), vec4<i32>(0), vec4<i32>(FIX_MAX));
}

// ── fixed.rs: the reciprocal unit and the level of detail ────────────────────

// `fixed::recip` for wi > 0: (y, n), y = 1/m in Q31, wi = m 2^n.
fn recip(wi: I64) -> vec2<u32> {
    let n = u64_top_bit(wi);
    var x: I64;
    if n >= 30u {
        x = u64_shr(wi, n - 30u);
    } else {
        x = u64_shl(wi, 30u - n);
    }
    let y0 = tables[(x.lo >> 20u) & 0x3ffu];
    let e = u64_shr(mul32x32(x.lo, y0), 30u).lo;
    // (y0 * (2^32 - e)) >> 31.
    var p: I64;
    if e == 0u {
        p = I64(0u, y0);
    } else {
        p = mul32x32(y0, 0u - e);
    }
    return vec2<u32>(u64_shr(p, 31u).lo, n);
}

fn persp(v: I64, y: u32, n: u32) -> I64 {
    return mul_shr_s(v, I64(y, 0u), n);
}

fn deriv(a: I64, y: u32, n: u32, l: u32) -> I64 {
    let r = mul_shr_s(a, I64(y, 0u), n + 15u - l);
    let lo = I64(0x80000000u, 0xffffffffu);
    let hi = I64(0x80000000u, 0u);
    if i64_lt(r, lo) {
        return lo;
    }
    if i64_lt(hi, r) {
        return hi;
    }
    return r;
}

fn scale31(v: I64, c: I64) -> I64 {
    return mul_shr_s(v, c, 31u);
}

fn log2_q8(v: I64) -> i32 {
    let n = u64_top_bit(v);
    var f: u32;
    if n >= 8u {
        f = u64_shr(v, n - 8u).lo & 0xffu;
    } else {
        f = u64_shl(v, 8u - n).lo & 0xffu;
    }
    return (i32(n) << 8u) + i32(tables[1024u + f]);
}

fn lod_q8(dsx: I64, dtx: I64, dsy: I64, dty: I64) -> i32 {
    let ax = i64_add(i64_mul(dsx, dsx), i64_mul(dtx, dtx));
    let ay = i64_add(i64_mul(dsy, dsy), i64_mul(dty, dty));
    var m = ax;
    if u64_lt(ax, ay) {
        m = ay;
    }
    if i64_is_zero(m) {
        return LOD_NONE;
    }
    return (log2_q8(m) - (32 << 8u)) >> 1u;
}

// ── the texture environment and fog ─────────────────────────────────────────

fn tex_env(mode1: u32, env_rg: u32, env_b: u32, f: vec4<i32>, tex: vec4<i32>) -> vec4<i32> {
    let nc = ((mode1 >> 3u) & 3u) + 1u;
    let tclass = (mode1 >> 9u) & 7u;
    var has_c = true;
    var ct = vec3<i32>(tex.x);
    var has_a = false;
    var at = 0;
    if tclass == 1u {
        has_c = false;
        has_a = true;
        at = tex.x;
    } else if tclass == 2u {
        has_a = nc == 2u;
        at = tex.y;
    } else if tclass == 3u {
        has_a = true;
        at = tex.x;
    } else if nc >= 3u {
        ct = tex.xyz;
        has_a = nc == 4u;
        at = tex.w;
    } else {
        has_a = nc == 2u;
        at = tex.y;
    }
    let cc = vec3<i32>(field12(env_rg), field12(env_rg >> 12u), field12(env_b));
    let ac = field12(env_b >> 12u);
    let intensity = tclass == 3u;
    var out = f;
    switch (mode1 >> 1u) & 3u {
        case 1u: {
            if has_c {
                let a = select(FIX_ONE, at, has_a);
                for (var k = 0u; k < 3u; k++) {
                    out[k] = fmul(f[k], FIX_ONE - a) + fmul(ct[k], a);
                }
            }
        }
        case 2u: {
            if has_c {
                for (var k = 0u; k < 3u; k++) {
                    out[k] = fmul(f[k], FIX_ONE - ct[k]) + fmul(cc[k], ct[k]);
                }
            }
            if has_a {
                out.w = select(fmul(f.w, at), fmul(f.w, FIX_ONE - at) + fmul(ac, at), intensity);
            }
        }
        case 3u: {
            if has_a {
                out.w = fmul(f.w, at);
            }
        }
        default: {
            if has_c {
                for (var k = 0u; k < 3u; k++) {
                    out[k] = fmul(f[k], ct[k]);
                }
            }
            if has_a {
                out.w = fmul(f.w, at);
            }
        }
    }
    return out;
}

// `Rss::textured`: S/W, T/W, 1/W (2^32) and their steps across x (d0)
// and down y (d1).
fn textured(si: u32, sw: I64, tw: I64, wi: I64, ds: array<I64, 2>, dt: array<I64, 2>, dw: array<I64, 2>, rgba: vec4<i32>) -> vec4<i32> {
    if i64_is_neg(wi) || i64_is_zero(wi) {
        return rgba;
    }
    let r = recip(wi);
    let y = r.x;
    let n = r.y;
    let s = persp(sw, y, n);
    let t = persp(tw, y, n);
    let ls = sr(si, SL_LS);
    let lt = sr(si, SL_LT);
    let dsx = i64_sub(ds[0], scale31(s, dw[0]));
    let dtx = i64_sub(dt[0], scale31(t, dw[0]));
    let dsy = i64_sub(scale31(s, dw[1]), ds[1]);
    let dty = i64_sub(scale31(t, dw[1]), dt[1]);
    let lambda = lod_q8(deriv(dsx, y, n, ls), deriv(dtx, y, n, lt), deriv(dsy, y, n, ls), deriv(dty, y, n, lt));
    let texel = smp_sample(si, s, t, lambda);
    return tex_env(sr(si, SL_TM1), sr(si, SL_TXENVRG), sr(si, SL_TXENVB), rgba, texel);
}

fn fogged(si: u32, rgba: vec4<i32>, f_in: i32) -> vec4<i32> {
    let f = clamp(f_in, 0, FIX_ONE);
    let rg = sr(si, SL_FOGRG);
    let b = sr(si, SL_FOGB);
    let fc = vec3<i32>(field12(rg), field12(rg >> 12u), field12(b));
    var out = rgba;
    for (var k = 0u; k < 3u; k++) {
        out[k] = fmul(f, rgba[k]) + fmul(FIX_ONE - f, fc[k]);
    }
    return out;
}
