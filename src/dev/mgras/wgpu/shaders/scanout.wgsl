// IMPACT's scan-out: frame.rs's `Frame::pixel` for one display pixel (y = 0
// the top row), from the GPU unit's pixel pages and the frame's slot
// (wgpu/scanout.rs `MgrasScanout::build`): cursor, then the overlay through
// its window ID's mode, then the main planes through theirs (pairs and the
// buffer select included), then the DAC gamma. Out: 0x00RRGGBB.

#include "pp1.wgsl"

@group(1) @binding(0) var<storage, read> dslot: array<u32>;

fn ds(k: u32) -> u32 {
    return dslot[k];
}

// The window ID at x on row y of run list `list` (0 main, 1 overlay), as
// `fill_did_row` fills it: each run lasts to the next one's x, later runs
// over earlier ones.
fn did_at(list: u32, y: u32, x: u32) -> u32 {
    let rows = D_ROWS + list * (D_ROW_COUNT + 1u);
    let first = ds(rows + y);
    let end = ds(rows + y + 1u);
    var did = 0u;
    for (var k = first; k < end; k++) {
        let r = ds(D_RUNS + k);
        let x0 = min(r & 0xffffu, FB_W);
        var x1 = FB_W;
        if k + 1u < end {
            x1 = min(ds(D_RUNS + k + 1u) & 0xffffu, FB_W);
        }
        if x0 <= x && x < x1 {
            did = (r >> 16u) & 0xffu;
        }
    }
    return did;
}

fn gamma_rgb(r: u32, g: u32, b: u32) -> u32 {
    let lr = ds(D_GAMMA + (r & 0xffu)) & 0xffu;
    let lg = (ds(D_GAMMA + (g & 0xffu)) >> 8u) & 0xffu;
    let lb = (ds(D_GAMMA + (b & 0xffu)) >> 16u) & 0xffu;
    return (lr << 16u) | (lg << 8u) | lb;
}

// A colormap entry (0x00RRGGBB) through the gamma.
fn gamma_cmap(c: u32) -> u32 {
    return gamma_rgb(c >> 16u, c >> 8u, c);
}

fn cursor_value(x: u32, y: u32) -> u32 {
    if ds(D_CURSOR_ON) == 0u {
        return 0u;
    }
    let cx = i32(x) - bitcast<i32>(ds(D_CURSOR_X));
    let cy = i32(y) - bitcast<i32>(ds(D_CURSOR_Y));
    let size = i32(ds(D_CURSOR_SIZE));
    if cx < 0 || cy < 0 || cx >= size || cy >= size {
        return 0u;
    }
    let i = u32(cy) * CURSOR_MAX + u32(cx);
    return (ds(D_GLYPH + i / 16u) >> (2u * (i % 16u))) & 3u;
}

fn scanout_pixel(x: u32, y: u32) -> u32 {
    if ds(D_BLANK) != 0u {
        return 0u;
    }
    let c = cursor_value(x, y);
    if c != 0u {
        return gamma_cmap(ds(D_CURSOR_RGB + c));
    }
    let h = ds(D_HEIGHT);
    let fb_y = h - 1u - y;
    let drbsize = ds(D_DRBSIZE);
    let optr = ds(D_OVERLAY);
    if optr != NO_BUFFER {
        let omode = ds(D_OVERLAY_MODE + (did_at(1u, y, x) & 31u));
        let v = pix_get(buf_new(optr, KIND_OVERLAY, drbsize), x, fb_y).x & 0xffu;
        if (omode & 1u) != 0u && v != 0u {
            return gamma_cmap(ds(D_CMAP + (((omode >> 16u) + v) & 0x1fffu)));
        }
    }
    let did = did_at(0u, y, x) & 31u;
    let m = ds(D_MAIN_MODE + did);
    let rgb = (m & 1u) != 0u;
    let pair = (m >> 1u) & 3u;
    let b = ((ds(D_SELECT) >> did) & 1u) != 0u;
    var v = pix_get(buf_new(ds(D_MAIN), KIND_WIDE, drbsize), x, fb_y).x;
    if pair != 0u {
        v = pair_unpack(pair, v, b) & 0xffffffu;
    } else if b && !rgb {
        v = (v >> 12u) & 0xfffu;
    } else if b {
        v = pix_get(buf_new(ds(D_BACK), KIND_WIDE, drbsize), x, fb_y).x;
    }
    if rgb {
        return gamma_rgb(v, v >> 8u, v >> 16u);
    }
    let mask = (m >> 4u) & 0xfffu;
    return gamma_cmap(ds(D_CMAP + (((m >> 16u) + (v & mask)) & 0x1fffu)));
}
