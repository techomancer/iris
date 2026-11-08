// IMPACT's pixel memory (pixmem.rs) and clip-ID planes on the GPU.
//
// The pages are the model's u64 words as bytes on a little-endian host:
// word i is u32 pair (2i, 2i + 1), the low half first. A 36-bit buffer
// keeps a pixel in a word's low 36 bits; the overlay keeps four 9-bit
// pixels a word, lowest x in the low bits, so pixel 3's cell crosses the
// halves. The clip-ID planes are a byte a framebuffer pixel, four to a
// u32. Every access is atomic: four overlay pixels (and four clip IDs)
// share a word, and the tile's lanes read-modify-write their own bits of
// it (clanker/wgpu/vram-accesses-are-atomic.md).

@group(2) @binding(0) var<storage, read_write> pix: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> cids: array<atomic<u32>>;

const KIND_WIDE: u32 = 0u;
const KIND_OVERLAY: u32 = 1u;

// pixmem.rs `Buffer`.
struct Buf {
    ptr: u32,
    kind: u32,
    xtiles: u32,
    yshift: u32,
}

// DRBsize bits 5:2 / 1:0: 36-bit and overlay tiles a row (the PROM's
// 1280x1024 layout until programmed).
fn buf_new(ptr: u32, kind: u32, drbsize: u32) -> Buf {
    var x36 = (drbsize >> 2u) & 0xfu;
    var x9 = drbsize & 3u;
    if x36 == 0u {
        x36 = 7u;
        x9 = 2u;
    }
    let yshift = select(0u, 1u, ((drbsize >> 6u) & 3u) != 0u);
    return Buf(ptr & 0x3ffu, kind, select(x9, x36, kind == KIND_WIDE), yshift);
}

fn buf_eq(a: Buf, b: Buf) -> bool {
    return a.ptr == b.ptr && a.kind == b.kind && a.xtiles == b.xtiles && a.yshift == b.yshift;
}

// The u64 word index of pixel (x, y), and the cell's shift (`Buffer::locate`).
fn buf_locate(b: Buf, x: u32, y: u32) -> vec2<u32> {
    let re = y & ((1u << b.yshift) - 1u);
    let ry = y >> b.yshift;
    let tile_w = select(OVERLAY_TILE_W, PIX_TILE_W, b.kind == KIND_WIDE);
    let page = (b.ptr + (ry / PIX_TILE_H) * b.xtiles + x / tile_w) % PAGES;
    let base = re * RE_WORDS + page * PAGE_WORDS + (ry % PIX_TILE_H) * PIX_TILE_W;
    if b.kind == KIND_WIDE {
        return vec2<u32>(base + x % PIX_TILE_W, 0u);
    }
    return vec2<u32>(base + (x % OVERLAY_TILE_W) / 4u, 9u * (x % 4u));
}

// The pixel's value: 36 bits (low 32, high 4) or 9.
fn pix_get(b: Buf, x: u32, y: u32) -> vec2<u32> {
    let l = buf_locate(b, x, y);
    let w = 2u * l.x;
    let lo = atomicLoad(&pix[w]);
    if b.kind == KIND_WIDE {
        return vec2<u32>(lo, atomicLoad(&pix[w + 1u]) & 0xfu);
    }
    if l.y + 9u <= 32u {
        return vec2<u32>((lo >> l.y) & 0x1ffu, 0u);
    }
    let hi = atomicLoad(&pix[w + 1u]);
    return vec2<u32>(((lo >> l.y) | (hi << (32u - l.y))) & 0x1ffu, 0u);
}

// Store the pixel's value (masked to its 36 or 9 bits).
fn pix_put(b: Buf, x: u32, y: u32, v: vec2<u32>) {
    let l = buf_locate(b, x, y);
    let w = 2u * l.x;
    if b.kind == KIND_WIDE {
        atomicStore(&pix[w], v.x);
        atomicAnd(&pix[w + 1u], ~0xfu);
        atomicOr(&pix[w + 1u], v.y & 0xfu);
        return;
    }
    let c = v.x & 0x1ffu;
    let m_lo = 0x1ffu << l.y;
    atomicAnd(&pix[w], ~m_lo);
    atomicOr(&pix[w], c << l.y);
    if l.y + 9u > 32u {
        let s = 32u - l.y;
        let m_hi = 0x1ffu >> s;
        atomicAnd(&pix[w + 1u], ~m_hi);
        atomicOr(&pix[w + 1u], c >> s);
    }
}

fn cid_get(x: u32, y: u32) -> u32 {
    let i = y * FB_W + x;
    return (atomicLoad(&cids[i >> 2u]) >> (8u * (i & 3u))) & 0xffu;
}

// Bits `m` of the pixel's clip ID from `v`.
fn cid_put(x: u32, y: u32, v: u32, m: u32) {
    let i = y * FB_W + x;
    let s = 8u * (i & 3u);
    atomicAnd(&cids[i >> 2u], ~(m << s));
    atomicOr(&cids[i >> 2u], (v & m) << s);
}
