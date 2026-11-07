// The VPro scan-out for one screen pixel (frame.rs `Frame::pixel` and
// reader.rs `DisplayRegs::cursor_row`): the window-id byte picks the
// overlay or the window's display mode, the colour field of the buffer the
// window shows is decoded and sent through the gamma tables or the colour
// lookup RAM, and the cursor goes over it. The display slot
// (wgpu/scanout.rs `DisplaySlot`) is what the CPU resolves once per
// frame from the registers.

#include "views.wgsl"

struct DisplaySlot {
    // fb base (bytes), fb pitch (tiles), fb log2 bytes a pixel, fb bytes a
    // pixel, wid base, wid pitch, width, height, cursor x, cursor y,
    // cursor control (bit 0 shown, bit 1 crosshair), wid front byte, wid
    // back byte, pad.
    head: array<u32, 16>,
    // Per main window id, 8 words: buffer shown, rgb, keep, down, base,
    // mask, field A, field B (a field is its kind word with the byte
    // offset in bits 31:24).
    ids: array<u32, 256>,
    // Per overlay id: enabled, entry base.
    overlay: array<u32, 16>,
    // Each 8-bit channel value through the gamma table, in its byte of
    // the pixel: red, green, blue.
    gamma: array<u32, 768>,
    // The cursor's backend entries 0x2500-0x25ff.
    cursor: array<u32, 256>,
    // Every lookup-RAM entry as 0x00RRGGBB.
    rgb: array<u32, 8192>,
}

@group(1) @binding(0) var<storage, read> disp: DisplaySlot;

const DID_MAIN_SHIFT: u32 = 3u;
const DID_OVERLAY_MASK: u32 = 7u;
const CURSOR_SIZE: i32 = 32;
const CURSOR_GLYPH: u32 = 0x00u;
const CURSOR_COLOURS: u32 = 0x90u;
const CURSOR_PLANE: u32 = 0xc0u;

// The byte address of pixel (x, y) of a tiled surface (vram.rs
// Surface::offset).
fn surface_offset(base: u32, pitch: u32, log2: u32, x: u32, y: u32) -> u32 {
    let tile = (y >> 4u) * pitch + (x >> 4u);
    return base + (((tile << 8u) | ((y & 15u) << 4u) | (x & 15u)) << log2);
}

fn cursor_rgb(word: u32) -> u32 {
    return (((word >> 22u) & 0xffu) << 16u) | (((word >> 12u) & 0xffu) << 8u) | ((word >> 2u) & 0xffu);
}

fn scanout_pixel(x: u32, y: u32) -> u32 {
    let fb_base = disp.head[0];
    let fb_pitch = disp.head[1];
    let fb_log2 = disp.head[2];
    let fb_bytes = disp.head[3];
    let w = surface_offset(disp.head[4], disp.head[5], 1u, x, y);
    let p = surface_offset(fb_base, fb_pitch, fb_log2, x, y);
    let did = vram_load(w + disp.head[11], 1u);
    let ov = vram_load(w + disp.head[12], 1u);
    var rgb = 0u;
    let o = did & DID_OVERLAY_MASK;
    if disp.overlay[o * 2u] != 0u && ov != 0u {
        rgb = disp.rgb[(disp.overlay[o * 2u + 1u] + ov) & 8191u];
    } else {
        let id = (did >> DID_MAIN_SHIFT) & 31u;
        let i = id * 8u;
        let buf = disp.ids[i] & 1u;
        let field = disp.ids[i + 6u + buf];
        let at = field >> 24u;
        if at + 2u <= fb_bytes {
            let word = load_kind(p + at, field & 0xffffffu);
            if disp.ids[i + 1u] != 0u {
                let keep = disp.ids[i + 2u];
                let down = disp.ids[i + 3u];
                let r = ((((word >> 24u) & keep) | ((word >> 24u) >> down))) & 0xffu;
                let g = (((((word >> 16u) & 0xffu) & keep) | (((word >> 16u) & 0xffu) >> down))) & 0xffu;
                let b = (((((word >> 8u) & 0xffu) & keep) | (((word >> 8u) & 0xffu) >> down))) & 0xffu;
                rgb = disp.gamma[r] | disp.gamma[256u + g] | disp.gamma[512u + b];
            } else {
                rgb = disp.rgb[(disp.ids[i + 4u] + (word & disp.ids[i + 5u])) & 8191u];
            }
        }
    }
    // The cursor over it.
    let ctl = disp.head[10];
    if (ctl & 1u) == 0u {
        return rgb;
    }
    let cx = bitcast<i32>(disp.head[8]);
    let cy = bitcast<i32>(disp.head[9]);
    let row = i32(y) - cy;
    if (ctl & 2u) != 0u {
        if row == 0 || i32(x) == cx {
            return cursor_rgb(disp.cursor[CURSOR_COLOURS + 1u]);
        }
        return rgb;
    }
    let col = i32(x) - cx;
    if row < 0 || row >= CURSOR_SIZE || col < 0 || col >= CURSOR_SIZE {
        return rgb;
    }
    let ur = u32(row);
    let uc = u32(col);
    let plane = (disp.cursor[CURSOR_PLANE + ur * 2u + uc / 16u] >> ((uc % 16u) * 2u)) & 3u;
    let glyph = (disp.cursor[CURSOR_GLYPH + ur * 4u + uc / 8u] >> ((uc % 8u) * 4u)) & 0xfu;
    if plane != 3u || glyph == 0xfu {
        return rgb;
    }
    return cursor_rgb(disp.cursor[CURSOR_COLOURS + glyph]);
}
