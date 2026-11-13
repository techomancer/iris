// IMPACT's pixel processors (PP1) on the GPU: rss.rs's `visible`, `put`,
// `put_in`, `put_in36`, `put36`, the GL fragment's tests (`gl_fragment`,
// `depth_stencil`), colour (`fragment_color`, `fragment_color36`,
// `blend_rgba`) and the transfer fragment (`put_fragment`), operation for
// operation, from the registers a record's slot holds (wgpu/mod.rs
// `Slot::of`). Colour is 12.16 fixed point (fixed.rs).

#include "pixmem.wgsl"
#include "wide.wgsl"

struct Slot {
    w: array<u32, SLOT_WORDS>,
}

@group(1) @binding(1) var<storage, read> slots: array<Slot>;

// Slot words, read in place (clanker/wgpu/read-storage-in-place.md).
fn sr(si: u32, k: u32) -> u32 {
    return slots[si].w[k];
}

fn s16(v: u32) -> i32 {
    return (bitcast<i32>(v << 16u)) >> 16u;
}

const FIX_ONE: i32 = 1 << 28;
const FIX_MAX: i32 = (1 << 28) - 1;
const TEST_ENABLE: u32 = 8u;
const ZMODE_TEST: u32 = 8u;
const BLEND_ENABLE: u32 = 256u;
const LOGIC_ENABLE: u32 = 4u;
const PP1_DITHER: u32 = 8u;
const CONFIG_YFLIP: u32 = 8u;
const DRAW_CID: u32 = 0x50u;
const BAYER_LO: u32 = 0x6E4CA280u;
const BAYER_HI: u32 = 0x5D7F91B3u;

fn draw_buffer(pp1: u32) -> u32 {
    return (pp1 >> 14u) & 0x7fu;
}

fn rgb_pixtype(pp1: u32) -> bool {
    return ((pp1 >> 8u) & 7u) <= 3u;
}

fn rgb36(pp1: u32) -> bool {
    return ((pp1 >> 8u) & 7u) == 3u && (pp1 & (1u << 13u)) != 0u;
}

fn rgb12_pair(pp1: u32) -> bool {
    return ((pp1 >> 8u) & 7u) <= 1u && (pp1 & (1u << 13u)) == 0u;
}

fn draws_b(f: u32) -> bool {
    return f == 2u || f == 4u;
}

fn draws_a_and_b(f: u32) -> bool {
    return f == 3u || f == 5u;
}

fn draws_overlay(pp1: u32) -> bool {
    return (draw_buffer(pp1) & 0x70u) == 0x40u;
}

fn deep36(si: u32) -> bool {
    let lsb = sr(si, SL_LSBA) | sr(si, SL_LSBB);
    return rgb36(sr(si, SL_PP1)) && ((lsb & 0xff000000u) != 0u || (sr(si, SL_MSBS) & 0xfu) != 0u);
}

fn second_buffer(si: u32) -> u32 {
    return (sr(si, SL_DRB) >> 10u) & 0x3ffu;
}

// `Rss::target`.
fn target_buf(si: u32) -> Buf {
    let pp1 = sr(si, SL_PP1);
    let drb = sr(si, SL_DRB);
    let kind = select(KIND_WIDE, KIND_OVERLAY, draws_overlay(pp1));
    let b = second_buffer(si);
    var ptr = drb;
    if b != 0u && kind == KIND_WIDE && draws_b(draw_buffer(pp1)) && !rgb12_pair(pp1) && rgb_pixtype(pp1) {
        ptr = b;
    }
    return buf_new(ptr, kind, sr(si, SL_DRBSIZE));
}

// `Rss::visible` (one unit owns every row).
fn visible(si: u32, x: i32, y: i32) -> bool {
    if x < 0 || y < 0 || x >= i32(FB_W) || y >= i32(FB_H) {
        return false;
    }
    let m = (sr(si, SL_WINMODE) >> 4u) & 0xfu;
    if m != 0u && ((m >> cid_get(u32(x), u32(y))) & 1u) == 0u {
        return false;
    }
    let mode = sr(si, SL_CLIPMODE);
    for (var n = 0u; n < 4u; n++) {
        if (mode & (1u << n)) == 0u {
            continue;
        }
        let mx = sr(si, SL_CLIP + 2u * n);
        let my = sr(si, SL_CLIP + 2u * n + 1u);
        let inside = s16(mx >> 16u) <= x && x <= s16(mx) && s16(my >> 16u) <= y && y <= s16(my);
        if inside != ((mode & (0x10u << n)) != 0u) {
            return false;
        }
    }
    return true;
}

fn logic_op(op: u32, s: u32, d: u32) -> u32 {
    switch op & 0xfu {
        case 0u: { return 0u; }
        case 1u: { return s & d; }
        case 2u: { return s & ~d; }
        case 3u: { return s; }
        case 4u: { return ~s & d; }
        case 5u: { return d; }
        case 6u: { return s ^ d; }
        case 7u: { return s | d; }
        case 8u: { return ~(s | d); }
        case 9u: { return ~(s ^ d); }
        case 10u: { return ~d; }
        case 11u: { return s | ~d; }
        case 12u: { return ~s; }
        case 13u: { return ~s | d; }
        case 14u: { return ~(s & d); }
        default: { return 0xffffffffu; }
    }
}

// ── 12:12:12 ────────────────────────────────────────────────────────────────

fn pack36(c: vec3<u32>) -> vec2<u32> {
    let lo = ((c.x >> 4u) & 0xffu) | (((c.y >> 4u) & 0xffu) << 8u) | (((c.z >> 4u) & 0xffu) << 16u) | ((c.x & 0xfu) << 24u) | ((c.y & 0xfu) << 28u);
    return vec2<u32>(lo, c.z & 0xfu);
}

fn unpack36(w: vec2<u32>) -> vec3<u32> {
    return vec3<u32>(
        ((w.x & 0xffu) << 4u) | ((w.x >> 24u) & 0xfu),
        (((w.x >> 8u) & 0xffu) << 4u) | ((w.x >> 28u) & 0xfu),
        (((w.x >> 16u) & 0xffu) << 4u) | (w.y & 0xfu),
    );
}

fn rgb36_from8(v: u32) -> vec2<u32> {
    let r = v & 0xffu;
    let g = (v >> 8u) & 0xffu;
    let b = (v >> 16u) & 0xffu;
    return pack36(vec3<u32>((r << 4u) | (r >> 4u), (g << 4u) | (g >> 4u), (b << 4u) | (b >> 4u)));
}

// ── pixel pairs (rss.rs `Pair`) ──────────────────────────────────────────────

const PAIR_NONE: u32 = 0u;
const PAIR_444: u32 = 1u;
const PAIR_555: u32 = 2u;

fn pair_of(pp1: u32) -> u32 {
    if !rgb12_pair(pp1) {
        return PAIR_NONE;
    }
    return select(PAIR_444, PAIR_555, ((pp1 >> 8u) & 7u) == 1u);
}

fn pair_bits(p: u32) -> u32 {
    return select(4u, 5u, p == PAIR_555);
}

// Buffer b's alpha: (position, width).
fn pair_alpha(p: u32, b: bool) -> vec2<u32> {
    if p == PAIR_555 {
        return vec2<u32>(select(30u, 31u, b), 1u);
    }
    return vec2<u32>(select(24u, 28u, b), 4u);
}

fn bayer(idx: u32) -> u32 {
    let w = select(BAYER_LO, BAYER_HI, idx >= 8u);
    return (w >> (4u * (idx & 7u))) & 0xfu;
}

// An 8-bit component to n bits, truncated (t < 0) or dithered.
fn quantise(ch: u32, n: u32, t: i32) -> u32 {
    if t < 0 {
        return ch >> (8u - n);
    }
    let s = ch - (ch >> n);
    let f = s & ((1u << (8u - n)) - 1u);
    return min((s >> (8u - n)) + select(0u, 1u, f > (u32(t) >> (n - 4u))), (1u << n) - 1u);
}

// An n-bit component to 8 bits, its bits repeated.
fn widen8(v: u32, n: u32) -> u32 {
    if n == 0u {
        return 0u;
    }
    if n == 1u {
        return v * 0xffu;
    }
    return ((v << (8u - n)) | (v >> (2u * n - 8u))) & 0xffu;
}

fn pair_pack(p: u32, pp1: u32, v: u32, x: i32, y: i32) -> u32 {
    var t = -1;
    if (pp1 & PP1_DITHER) != 0u {
        t = i32(bayer(((u32(y) & 3u) << 2u) | (u32(x) & 3u)));
    }
    let n = pair_bits(p);
    var c = 0u;
    for (var k = 0u; k < 3u; k++) {
        c |= quantise((v >> (8u * k)) & 0xffu, n, t) << (n * k);
    }
    let a0 = pair_alpha(p, false);
    let a1 = pair_alpha(p, true);
    let a = (v >> 24u) >> (8u - a0.y);
    return c | (c << (3u * n)) | (a << a0.x) | (a << a1.x);
}

fn pair_unpack(p: u32, w: u32, b: bool) -> u32 {
    let n = pair_bits(p);
    let c = select(w, w >> (3u * n), b);
    var rgb = 0u;
    for (var k = 0u; k < 3u; k++) {
        rgb |= widen8((c >> (n * k)) & ((1u << n) - 1u), n) << (8u * k);
    }
    let a = pair_alpha(p, b);
    return rgb | (widen8((w >> a.x) & ((1u << a.y) - 1u), a.y) << 24u);
}

// ── stores ──────────────────────────────────────────────────────────────────

fn put_in36(si: u32, b: Buf, x: i32, y: i32, v: vec2<u32>, back: bool) {
    let pp1 = sr(si, SL_PP1);
    let lsb = sr(si, select(SL_LSBA, SL_LSBB, back));
    let mhi = sr(si, SL_MSBS) & 0xfu;
    let old = pix_get(b, u32(x), u32(y));
    var nv = v;
    if (pp1 & LOGIC_ENABLE) != 0u {
        let op = pp1 >> 26u;
        nv = vec2<u32>(logic_op(op, v.x, old.x), logic_op(op, v.y, old.y));
    }
    pix_put(b, u32(x), u32(y), vec2<u32>((old.x & ~lsb) | (nv.x & lsb), (old.y & ~mhi) | (nv.y & mhi)));
}

fn put_in(si: u32, b: Buf, x: i32, y: i32, v: u32, back: bool) {
    let pp1 = sr(si, SL_PP1);
    if b.kind == KIND_WIDE && deep36(si) {
        put_in36(si, b, x, y, rgb36_from8(v), back);
        return;
    }
    let lsb = sr(si, select(SL_LSBA, SL_LSBB, back));
    let wide = rgb_pixtype(pp1);
    let old = pix_get(b, u32(x), u32(y)).x;
    let pair = pair_of(pp1);
    if b.kind == KIND_WIDE && pair != PAIR_NONE {
        var pv = pair_pack(pair, pp1, v, x, y);
        if (pp1 & LOGIC_ENABLE) != 0u {
            pv = logic_op(pp1 >> 26u, pv, old);
        }
        pix_put(b, u32(x), u32(y), vec2<u32>((old & ~lsb) | (pv & lsb), 0u));
        return;
    }
    let ci = b.kind == KIND_WIDE && !rgb_pixtype(pp1);
    var val = select(v, (v & 0xfffu) * 0x1001u, ci);
    if (pp1 & LOGIC_ENABLE) != 0u {
        var width = 0xfffu;
        if ((pp1 >> 8u) & 7u) == 2u {
            width = 0xffffffffu;
        } else if wide || ci {
            width = 0xffffffu;
        }
        val = logic_op(pp1 >> 26u, val, old) & width;
    }
    if b.kind == KIND_OVERLAY {
        let msbs = sr(si, SL_MSBS);
        var mask = select(0u, 0xffu, msbs != 0u);
        if draw_buffer(pp1) == 0x48u {
            mask = msbs & 0xffu;
        }
        val = (old & ~mask) | (val & mask);
    } else if b.kind == KIND_WIDE && ((pp1 >> 8u) & 7u) == 2u {
        val = (old & ~lsb) | (val & lsb);
    } else if ci {
        let mask = lsb & 0xffffffu;
        val = (old & ~mask) | (val & mask);
    }
    pix_put(b, u32(x), u32(y), vec2<u32>(val, 0u));
}

// `Rss::put`: the target buffer, B as well for draw fields 3 and 5, or
// the clip-ID planes.
fn put(si: u32, x: i32, y: i32, v: u32) {
    let pp1 = sr(si, SL_PP1);
    let field = draw_buffer(pp1);
    if !visible(si, x, y) {
        return;
    }
    if field == DRAW_CID {
        cid_put(u32(x), u32(y), v, (sr(si, SL_WINMODE) >> 10u) & 3u);
        return;
    }
    let b = target_buf(si);
    put_in(si, b, x, y, v, draws_b(field));
    if draws_a_and_b(field) && !(b.kind == KIND_WIDE && (rgb12_pair(pp1) || !rgb_pixtype(pp1))) {
        let p = second_buffer(si);
        if p != 0u {
            let b2 = buf_new(p, KIND_WIDE, sr(si, SL_DRBSIZE));
            if !buf_eq(b2, b) {
                put_in(si, b2, x, y, v, true);
            }
        }
    }
}

fn put36(si: u32, x: i32, y: i32, v: vec2<u32>) {
    let field = draw_buffer(sr(si, SL_PP1));
    if !visible(si, x, y) || field == DRAW_CID {
        return;
    }
    let b = target_buf(si);
    put_in36(si, b, x, y, v, draws_b(field));
    if draws_a_and_b(field) {
        let p = second_buffer(si);
        if p != 0u {
            let b2 = buf_new(p, KIND_WIDE, sr(si, SL_DRBSIZE));
            if !buf_eq(b2, b) {
                put_in36(si, b2, x, y, v, true);
            }
        }
    }
}

// ── the GL fragment ─────────────────────────────────────────────────────────

// (a * b) >> 28, 12.16.
fn fmul(a: i32, b: i32) -> i32 {
    let p = i64_mul_i32(a, b);
    return bitcast<i32>((p.lo >> 28u) | (p.hi << 4u));
}

fn fclamp(v: i32) -> i32 {
    return clamp(v, 0, FIX_MAX);
}

fn to_byte(c: i32) -> u32 {
    return u32(fclamp(c) >> 20u);
}

fn to_index(c: i32) -> u32 {
    return u32(fclamp(c) >> 16u);
}

fn field12(v: u32) -> i32 {
    return bitcast<i32>((v & 0xfffu) << 16u);
}

// `a` OP `b` for an OpenGL compare function.
fn compare(func: u32, a: i32, b: i32) -> bool {
    switch func & 7u {
        case 0u: { return false; }
        case 1u: { return a < b; }
        case 2u: { return a == b; }
        case 3u: { return a <= b; }
        case 4u: { return a > b; }
        case 5u: { return a != b; }
        case 6u: { return a >= b; }
        default: { return true; }
    }
}

fn blend_factor(code: u32, k: u32, s: vec4<i32>, d: vec4<i32>, kc: vec4<i32>) -> i32 {
    switch code {
        case 0u: { return 0; }
        case 1u: { return FIX_ONE; }
        case 2u: { return s[k]; }
        case 3u: { return FIX_ONE - s[k]; }
        case 4u: { return s.w; }
        case 5u: { return FIX_ONE - s.w; }
        case 6u: { return d.w; }
        case 7u: { return FIX_ONE - d.w; }
        case 8u: { return d[k]; }
        case 9u: { return FIX_ONE - d[k]; }
        case 10u: { return select(min(s.w, FIX_ONE - d.w), FIX_ONE, k == 3u); }
        case 11u: { return kc[k]; }
        case 12u: { return FIX_ONE - kc[k]; }
        case 13u: { return kc.w; }
        case 14u: { return FIX_ONE - kc.w; }
        default: { return FIX_ONE; }
    }
}

fn blend_rgba(bf: u32, kc: vec4<i32>, s: vec4<i32>, d: vec4<i32>) -> vec4<i32> {
    let sf = bf & 0xfu;
    let df = (bf >> 4u) & 0xfu;
    let eq = (bf >> 9u) & 7u;
    var out = vec4<i32>(0);
    for (var k = 0u; k < 4u; k++) {
        var v: i32;
        switch eq {
            case 1u: { v = fmul(s[k], blend_factor(sf, k, s, d, kc)) - fmul(d[k], blend_factor(df, k, s, d, kc)); }
            case 2u: { v = fmul(d[k], blend_factor(df, k, s, d, kc)) - fmul(s[k], blend_factor(sf, k, s, d, kc)); }
            case 3u: { v = min(s[k], d[k]); }
            case 4u: { v = max(s[k], d[k]); }
            default: { v = fmul(s[k], blend_factor(sf, k, s, d, kc)) + fmul(d[k], blend_factor(df, k, s, d, kc)); }
        }
        out[k] = fclamp(v);
    }
    return out;
}

fn blend_constant(si: u32) -> vec4<i32> {
    let rg = sr(si, SL_BLENDRG);
    let ba = sr(si, SL_BLENDBA);
    return vec4<i32>(field12(rg >> 16u), field12(rg), field12(ba >> 16u), field12(ba));
}

// A z.12 depth rounded to the 24-bit buffer value (`fixed::zbuf`).
fn zbuf(z: I64) -> u32 {
    return i64_clamp32(i64_shr(i64_add(z, I64(2048u, 0u)), 12u), 0, 0xffffff).lo;
}

// `Rss::depth_stencil`.
fn depth_stencil(si: u32, x: u32, y: u32, z: I64) -> bool {
    let st = sr(si, SL_STMODE);
    let zm = sr(si, SL_ZMODE);
    let sten = (st & TEST_ENABLE) != 0u;
    let ztest = (zm & ZMODE_TEST) != 0u;
    if !(sten || ztest) {
        return true;
    }
    let zb = buf_new(ZST_PAGE, KIND_WIDE, sr(si, SL_DRBSIZE));
    let zst = pix_get(zb, x, y);
    let s_old = (zst.x >> 24u) & 0xffu;
    let masks = sr(si, SL_STMASK);
    let cmask = masks & 0xffu;
    let wmask = (masks >> 8u) & 0xffu;
    let sref = (st >> 16u) & 0xffu;
    let zq = zbuf(z);
    let zold = zst.x & 0xffffffu;
    let spass = !sten || compare(st & 7u, i32(sref & cmask), i32(s_old & cmask));
    let zpass = spass && (!ztest || compare(zm & 7u, i32(zq), i32(zold)));
    var nw = zst;
    if sten {
        var op: u32;
        if !spass {
            op = (st >> 4u) & 7u;
        } else if !zpass {
            op = (st >> 8u) & 7u;
        } else {
            op = (st >> 12u) & 7u;
        }
        var s_new = s_old;
        switch op {
            case 1u: { s_new = 0u; }
            case 2u: { s_new = sref; }
            case 3u: { s_new = min(s_old + 1u, 0xffu); }
            case 4u: { s_new = select(0u, s_old - 1u, s_old > 0u); }
            case 5u: { s_new = ~s_old & 0xffu; }
            default: {}
        }
        let s = (s_old & ~wmask) | (s_new & wmask);
        nw.x = (nw.x & ~(0xffu << 24u)) | (s << 24u);
    }
    if zpass && ztest {
        let m = (zm >> 4u) & 0xffffffu;
        nw.x = (nw.x & ~m) | (zq & m);
    }
    if nw.x != zst.x || nw.y != zst.y {
        pix_put(zb, x, y, nw);
    }
    return zpass;
}

fn fragment_color36(si: u32, b: Buf, x: u32, y: u32, rgba: vec4<i32>, pp1: u32, back: bool) {
    let dst = pix_get(b, x, y);
    let logic = (pp1 & LOGIC_ENABLE) != 0u && ((pp1 >> 26u) & 0xfu) != 3u;
    let blend = sr(si, SL_BLEND);
    var c = rgba;
    if (blend & BLEND_ENABLE) != 0u && !logic {
        let u = unpack36(dst);
        let d = vec4<i32>(bitcast<i32>(u.x << 16u), bitcast<i32>(u.y << 16u), bitcast<i32>(u.z << 16u), FIX_ONE);
        c = blend_rgba(blend, blend_constant(si), rgba, d);
    }
    let v = pack36(vec3<u32>(u32(fclamp(c.x) >> 16u), u32(fclamp(c.y) >> 16u), u32(fclamp(c.z) >> 16u)));
    put_in36(si, b, i32(x), i32(y), v, back);
}

// `Rss::fragment_color`.
fn fragment_color(si: u32, b: Buf, x: u32, y: u32, rgba: vec4<i32>, pp1: u32, back: bool) {
    if b.kind == KIND_WIDE && deep36(si) {
        fragment_color36(si, b, x, y, rgba, pp1, back);
        return;
    }
    var dst = pix_get(b, x, y).x;
    let pair = pair_of(pp1);
    if b.kind == KIND_WIDE && pair != PAIR_NONE {
        dst = pair_unpack(pair, dst, back);
    }
    let logic = (pp1 & LOGIC_ENABLE) != 0u && ((pp1 >> 26u) & 0xfu) != 3u;
    var src: u32;
    if rgb_pixtype(pp1) {
        let blend = sr(si, SL_BLEND);
        var c = rgba;
        if (blend & BLEND_ENABLE) != 0u && !logic {
            var d = vec4<i32>(0);
            for (var k = 0u; k < 4u; k++) {
                let ch = (dst >> (8u * k)) & 0xffu;
                d[k] = bitcast<i32>(((ch << 4u) | (ch >> 4u)) << 16u);
            }
            c = blend_rgba(blend, blend_constant(si), rgba, d);
        }
        src = to_byte(c.x) | (to_byte(c.y) << 8u) | (to_byte(c.z) << 16u) | (to_byte(c.w) << 24u);
    } else {
        src = to_index(rgba.x) & 0xfffu;
    }
    put_in(si, b, i32(x), i32(y), src, back);
}

// The colour stage into the target buffer, and into B for fields 3 and 5.
fn fragment_out(si: u32, x: u32, y: u32, rgba: vec4<i32>) {
    let pp1 = sr(si, SL_PP1);
    let field = draw_buffer(pp1);
    let b = target_buf(si);
    fragment_color(si, b, x, y, rgba, pp1, draws_b(field));
    if draws_a_and_b(field) && !rgb12_pair(pp1) && rgb_pixtype(pp1) {
        let p = second_buffer(si);
        if p != 0u {
            let b2 = buf_new(p, KIND_WIDE, sr(si, SL_DRBSIZE));
            if !buf_eq(b2, b) {
                fragment_color(si, b2, x, y, rgba, pp1, true);
            }
        }
    }
}

// `Rss::gl_fragment` at framebuffer (x, y).
fn gl_fragment(si: u32, x: i32, y: i32, rgba_in: vec4<i32>, z: I64) {
    if draw_buffer(sr(si, SL_PP1)) == DRAW_CID || !visible(si, x, y) {
        return;
    }
    let rgba = vec4<i32>(fclamp(rgba_in.x), fclamp(rgba_in.y), fclamp(rgba_in.z), fclamp(rgba_in.w));
    let af = sr(si, SL_AFUNC);
    if (af & TEST_ENABLE) != 0u && !compare(af & 7u, rgba.w, bitcast<i32>(((af >> 4u) & 0xfffu) << 16u)) {
        return;
    }
    if !depth_stencil(si, u32(x), u32(y), z) {
        return;
    }
    fragment_out(si, u32(x), u32(y), rgba);
}

// `Rss::put_fragment`: a transferred pixel through the per-fragment stage.
fn put_fragment(si: u32, x: i32, y: i32, v: u32, alpha: bool) {
    if !visible(si, x, y) {
        return;
    }
    let rgb = rgb_pixtype(sr(si, SL_PP1));
    var rgba: vec4<i32>;
    if rgb {
        let vv = select(v | (0xffu << 24u), v, alpha);
        for (var k = 0u; k < 4u; k++) {
            let ch = (vv >> (8u * k)) & 0xffu;
            rgba[k] = bitcast<i32>(((ch << 4u) | (ch >> 4u)) << 16u);
        }
    } else {
        rgba = vec4<i32>(bitcast<i32>((v & 0xfffu) << 16u), 0, 0, 0);
    }
    let af = sr(si, SL_AFUNC);
    if rgb && (af & TEST_ENABLE) != 0u && !compare(af & 7u, rgba.w, bitcast<i32>(((af >> 4u) & 0xfffu) << 16u)) {
        return;
    }
    let xz = sr(si, SL_XFERZ);
    if !depth_stencil(si, u32(x), u32(y), I64(xz << 4u, xz >> 28u)) {
        return;
    }
    fragment_out(si, u32(x), u32(y), rgba);
}
