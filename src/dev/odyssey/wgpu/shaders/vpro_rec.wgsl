// The VPro raster unit's records as the bin and raster kernels read them:
// the kinds, the record, and a triangle record's word offsets.
// src/dev/odyssey/wgpu/mod.rs `Rec` writes them.

const REC_TRI: u32 = 1u;
const REC_FRAG: u32 = 2u;
const REC_RECT: u32 = 3u;
const REC_BLIT: u32 = 4u;
const REC_FRAGS: u32 = 5u;
const REC_LINE: u32 = 6u;

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
