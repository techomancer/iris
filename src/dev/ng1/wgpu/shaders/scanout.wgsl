// Newport's display on the GPU: per screen pixel, the window id picks an
// XMAP9 mode; the VC2 cursor, then a popup, then the overlay, then the
// main pixel (through the CMAP, or expanded RGB) give the colour; the
// Bt445 RAMDAC's gamma tables finish it. The same rules as
// `SwCompositor::compose_pixels` (src/compositor.rs), word for word.
//
// The slot (group 1) holds the frame's registers and tables. The included
// bindings select either CPU-uploaded planes and display data together, or
// the resident raster's read-only plane group and separate display data.
// The RGB/aux planes are 2048 x 1024, the DID map has one byte per display
// pixel, and VC2 RAM has two 16-bit words per word (lower address first).

const H_CURSOR_X: u32 = 0u;     // the cursor's hot spot, as i32 bits
const H_CURSOR_Y: u32 = 1u;
const H_CURSOR_ENTRY: u32 = 2u;
const H_FLAGS: u32 = 3u;        // bit 0: cursor on; bit 1: 64 x 64 cursor
const H_CURSOR_CMAP: u32 = 4u;
const H_POPUP_CMAP: u32 = 5u;
const H_TOPSCAN: u32 = 6u;      // the first row shown, plus one
const H_VC2_WORDS: u32 = 7u;

struct Ng1Slot {
    head: array<u32, 16>,
    xmap: array<u32, 32>,
    ramdac: array<u32, 256>,
    cmap: array<u32, 8192>,
}

@group(1) @binding(0) var<storage, read> slot: Ng1Slot;
#include "ng1_bindings.wgsl"

fn vc2_word(a: u32) -> u32 {
    let w = vc2_ram[a >> 1u];
    return select(w & 0xffffu, w >> 16u, (a & 1u) != 0u);
}

fn did_at(i: u32) -> u32 {
    return (did_map[i >> 2u] >> (8u * (i & 3u))) & 0xffu;
}

// A CMAP entry, opaque; black past the table.
fn cmap_at(a: u32) -> u32 {
    if a < 8192u {
        return slot.cmap[a] | 0xff000000u;
    }
    return 0xff000000u;
}

// The RGB expansions (Rex3::expand_4_rgb, _8_, _12_): 0x00BBGGRR.
fn expand4(v: u32) -> u32 {
    let b = (v >> 3u) & 1u;
    let g = (0xaau * ((v >> 2u) & 1u)) | (0x55u * ((v >> 1u) & 1u));
    let r = v & 1u;
    return ((b * 255u) << 16u) | (g << 8u) | (r * 255u);
}

fn expand8(v: u32) -> u32 {
    let b = (0xaau * ((v >> 7u) & 1u)) | (0x55u * ((v >> 6u) & 1u));
    let g = (0x92u * ((v >> 5u) & 1u)) | (0x49u * ((v >> 4u) & 1u)) | (0x24u * ((v >> 3u) & 1u));
    let r = (0x92u * ((v >> 2u) & 1u)) | (0x49u * ((v >> 1u) & 1u)) | (0x24u * (v & 1u));
    return (b << 16u) | (g << 8u) | r;
}

fn expand12(v: u32) -> u32 {
    return ((((v >> 8u) & 0xfu) * 0x11u) << 16u) | ((((v >> 4u) & 0xfu) * 0x11u) << 8u) | ((v & 0xfu) * 0x11u);
}

// The VC2 cursor's colour index at screen (x, y); 0 where it shows nothing.
fn cursor_index(x: u32, y: u32) -> u32 {
    let flags = slot.head[H_FLAGS];
    if (flags & 1u) == 0u {
        return 0u;
    }
    let cx = i32(x) - bitcast<i32>(slot.head[H_CURSOR_X]);
    let cy = i32(y) - bitcast<i32>(slot.head[H_CURSOR_Y]);
    if cx < 0 || cy < 0 {
        return 0u;
    }
    let ux = u32(cx);
    let uy = u32(cy);
    let shift = 15u - (ux % 16u);
    let entry = slot.head[H_CURSOR_ENTRY];
    let words = slot.head[H_VC2_WORDS];
    if (flags & 2u) != 0u {
        if ux < 64u && uy < 64u {
            let a = entry + uy * 4u + ux / 16u;
            if a < words {
                return (vc2_word(a) >> shift) & 1u;
            }
        }
        return 0u;
    }
    if ux < 32u && uy < 32u {
        let a = entry + uy * 2u + ux / 16u;
        if a + 64u < words {
            return ((vc2_word(a) >> shift) & 1u) | (((vc2_word(a + 64u) >> shift) & 1u) << 1u);
        }
    }
    return 0u;
}

fn scanout_pixel(x: u32, y: u32) -> u32 {
    let fb_y = (slot.head[H_TOPSCAN] + y) & 0x3ffu;
    let idx = fb_y * 2048u + x;
    let mode = slot.xmap[did_at(y * 2048u + x) & 31u];

    let buf_sel = (mode & 1u) != 0u;
    let ovl_buf_sel = ((mode >> 1u) & 1u) != 0u;
    let msb_cmap = (mode >> 3u) & 0x1fu;
    let pix_mode = (mode >> 8u) & 3u;
    let pix_size = (mode >> 10u) & 3u;
    let aux_pix_mode = (mode >> 16u) & 7u;
    let aux_msb_cmap = (mode >> 19u) & 0x1fu;

    let raw_rgb = fb_rgb[idx];
    let raw_aux = fb_aux[idx];
    let pup = (raw_aux >> 2u) & 3u;
    let overlay = select((raw_aux >> 8u) & 0xffu, (raw_aux >> 16u) & 0xffu, ovl_buf_sel);

    var pixel = 0u;
    switch pix_size {
        case 0u: { pixel = (raw_rgb >> select(0u, 4u, buf_sel)) & 0xfu; }
        case 1u: { pixel = (raw_rgb >> select(0u, 8u, buf_sel)) & 0xffu; }
        case 2u: { pixel = (raw_rgb >> select(0u, 12u, buf_sel)) & 0xfffu; }
        default: { pixel = raw_rgb & 0xffffffu; }
    }

    let cursor = cursor_index(x, y);
    var colour = 0xff000000u;
    if cursor != 0u {
        colour = cmap_at((slot.head[H_CURSOR_CMAP] << 5u) | cursor);
    } else if pup != 0u {
        colour = cmap_at((slot.head[H_POPUP_CMAP] << 5u) | pup);
    } else if (aux_pix_mode == 2u || aux_pix_mode == 6u || aux_pix_mode == 7u) && overlay != 0u {
        colour = cmap_at((aux_msb_cmap << 8u) | overlay);
    } else if pix_mode == 0u {
        var a = pixel;
        if pix_size <= 1u {
            a = (msb_cmap << 8u) | pixel;
        } else if pix_size == 2u {
            a = ((msb_cmap & 0x10u) << 8u) | pixel;
        }
        colour = cmap_at(a);
    } else {
        switch pix_size {
            case 0u: { colour = expand4(pixel) | 0xff000000u; }
            case 1u: { colour = expand8(pixel) | 0xff000000u; }
            case 2u: { colour = expand12(pixel) | 0xff000000u; }
            default: { colour = pixel | 0xff000000u; }
        }
    }

    // The RAMDAC's per-channel gamma. The CMAP's word holds blue in bits
    // 23:16 and red in 7:0; the frame's word is 0x00RRGGBB.
    let b = (slot.ramdac[(colour >> 16u) & 0xffu] >> 16u) & 0xffu;
    let g = (slot.ramdac[(colour >> 8u) & 0xffu] >> 8u) & 0xffu;
    let r = slot.ramdac[colour & 0xffu] & 0xffu;
    return (r << 16u) | (g << 8u) | b;
}
