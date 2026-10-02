//! GR2 software compositor: VRAM + VC1 + XMAP5 + Bt457 -> a finished RGB frame.
//!
//! Runs on the display thread. The output uses the same format as
//! `SwCompositor`: stride 2048, `0xAABBGGRR` (R in the low byte, i.e. R,G,B,A
//! in memory, which is what the GL RGBA upload expects). Either UI compositor
//! can then present it verbatim (`Rex3Screen::prebuilt`). Internally the
//! XMAP5 CLUT holds `0x00RRGGBB`; only this final pack swaps to screen order.
//!
//! Per pixel, highest priority first (VC1.h "screen composition"): cursor,
//! popup (aux[3:2]), overlay (aux[1:0]), main visual chosen by the DID's
//! XMAP5 mode word, blackout. The Bt457 read mask and gamma ramp apply last.
//! Several details are unverified (see XMAP5.h/VC1.h). The textport path
//! (8-bit CI, buffer 0, CLUT page 16) is the one the PROM and kernel set up.

use super::bt457::Bt457;
use super::re3::{FB_H, FB_W};
use super::vc1::Vc1;
use super::xmap5::Xmap5;

pub const OUT_STRIDE: usize = 2048;
/// Cursor colour n (1..3) is CLUT entry 0x1C00 + n (XMAP5.h).
const CURSOR_CLUT_BASE: usize = 0x1c00;

// XMAP5 mode word fields.
const PD_RGB: u32 = 1;
const PD_BLACKOUT: u32 = 3;

/// Compose one 1280x1024 frame into `out` (stride `OUT_STRIDE`).
/// `vram` rows are bottom-up; display row 0 is the top of the screen.
pub fn compose(vram: &[u32], vc1: &Vc1, xmap: &[Xmap5; 5], dac: &[Bt457; 3], out: &mut [u32]) {
    let cursor_on = vc1.cursor_visible();
    let (cur_x, cur_y) = vc1.cursor_pos();

    let mut runs = [(0u16, 0u8); Vc1::MAX_DID_RUNS];
    for dy in 0..FB_H {
        // Per-pixel DID from the line's transition list.
        let nruns = vc1.line_did_runs(dy, &mut runs);
        let mut run = 0;
        let mut did = runs[0].1 as usize;
        let mut next_x = if nruns > 1 { runs[1].0 as usize } else { usize::MAX };
        let src_row = &vram[(FB_H - 1 - dy) * FB_W..(FB_H - dy) * FB_W];
        let out_row = &mut out[dy * OUT_STRIDE..dy * OUT_STRIDE + FB_W];
        let cy = dy as i32 - cur_y;
        let cursor_row = cursor_on && (0..32).contains(&cy);

        for (dx, (&p, o)) in src_row.iter().zip(out_row.iter_mut()).enumerate() {
            while dx >= next_x {
                run += 1;
                did = runs[run].1 as usize;
                next_x = if run + 1 < nruns { runs[run + 1].0 as usize } else { usize::MAX };
            }
            let xm = &xmap[dx % 5];
            let mut rgb = None;

            if cursor_row {
                let cx = dx as i32 - cur_x;
                if (0..32).contains(&cx) {
                    let c = vc1.cursor_pixel(cx as usize, cy as usize);
                    if c != 0 {
                        rgb = Some(xm.clut[CURSOR_CLUT_BASE + c as usize]);
                    }
                }
            }

            let rgb = rgb.unwrap_or_else(|| pixel_rgb(p, xm.mode[did], xm));
            let r = dac[0].lookup((rgb >> 16) as u8) as u32;
            let g = dac[1].lookup((rgb >> 8) as u8) as u32;
            let b = dac[2].lookup(rgb as u8) as u32;
            *o = 0xff00_0000 | (b << 16) | (g << 8) | r;
        }
    }
}

/// Resolve one VRAM word through an XMAP5 mode word to 0x00RRGGBB.
#[inline]
pub fn pixel_rgb(p: u32, mode: u32, xm: &Xmap5) -> u32 {
    let pix_pg = (mode >> 27) as usize & 0x1f;
    let pix_mode = (mode >> 24) & 7;
    let olayen = (mode >> 20) & 0xf;
    let pd_mode = (mode >> 16) & 3;
    let aux_pg = (mode >> 9) as usize & 0x1f;
    let pou_mode = (mode >> 5) & 3;
    let pou_pg = mode as usize & 0x1f;
    let aux = (p >> 24) & 0xf;

    // Popup planes (aux[3:2]) in a separate popup mode (POU_MODE); unused by
    // Xsgi, which leaves POU_MODE 0 and treats popups as part of the aux index.
    let pup = aux >> 2;
    if pou_mode != 0 && pup != 0 {
        return xm.clut[0x1c00 + pou_pg * 16 + pup as usize];
    }
    // Aux planes gated by OLAYEN: the enabled aux bits form one index into
    // the 16-entry aux block CLUT[0x1C00 + AUX_PG * 16] (Xsgi: AUX_PG 4 ->
    // 0x1C40, overlay/popup maps; the cursor block is AUX_PG 0).
    let olay = aux & olayen;
    if olay != 0 {
        return xm.clut[0x1c00 + aux_pg * 16 + olay as usize];
    }

    // Buffer select: bank 1 starts at the visual's depth, as libglcore's
    // double-buffer write masks are the bank-0 mask shifted by the depth:
    // 12-bit 11:0 / 23:12 (0x000FFF / 0xFFF000), 8-bit 7:0 / 15:8 (0x00FF
    // / 0xFF00), 4-bit 3:0 / 7:4.
    let bank1 = pix_mode & 1 == 1 && pix_mode < 6;
    let shift = match pix_mode { 0 | 1 => 4, 2 | 3 => 8, _ => 12 };
    let buf = (if bank1 { p >> shift } else { p }) & 0xfff;

    match pd_mode {
        PD_RGB => match pix_mode {
            7 | 6 => {
                // 24-bit: R in bits 7:0, G 15:8, B 23:16 (MAME RE2 layout).
                ((p & 0xff) << 16) | (p & 0xff00) | ((p >> 16) & 0xff)
            }
            4 | 5 => {
                // 12-bit 4:4:4, nibbles expanded (C8 = C4 << 4 | C4).
                let r = buf & 0xf;
                let g = (buf >> 4) & 0xf;
                let b = (buf >> 8) & 0xf;
                ((r * 0x11) << 16) | ((g * 0x11) << 8) | (b * 0x11)
            }
            _ => {
                // 8-bit 3:3:2: R 7:5, B 4:3, G 2:0 (the Xsgi TrueColor
                // visual's masks: red 0xE0, green 0x07, blue 0x18).
                let v = buf & 0xff;
                let r = ((v >> 5) & 7) * 255 / 7;
                let g = (v & 7) * 255 / 7;
                let b = ((v >> 3) & 3) * 255 / 3;
                (r << 16) | (g << 8) | b
            }
        },
        PD_BLACKOUT => ((xm.misc[0] as u32) << 16) | ((xm.misc[1] as u32) << 8) | xm.misc[2] as u32,
        _ => {
            // PD_CI / PD_CIMM: colour index through the CLUT.
            let idx = match pix_mode {
                0 | 1 => pix_pg * 256 + (buf & 0xf) as usize,
                2 | 3 => pix_pg * 256 + (buf & 0xff) as usize,
                _ => buf as usize,
            };
            xm.clut[idx & 0x1fff]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set16(b: &mut [u8], a: usize, v: u16) {
        b[a] = (v >> 8) as u8;
        b[a + 1] = v as u8;
    }

    /// 8-bit TrueColor: R 7:5, B 4:3, G 2:0 (Xsgi visual 0x27 masks red
    /// 0xE0, green 0x07, blue 0x18; 4Dwm icon images are drawn in it).
    #[test]
    fn rgb8_layout_matches_x_visual() {
        let xm: Box<Xmap5> = unsafe { Box::new_zeroed().assume_init() };
        let mode = (2 << 24) | (PD_RGB << 16);
        assert_eq!(pixel_rgb(0xe0, mode, &xm), 0x00ff_0000, "red");
        assert_eq!(pixel_rgb(0x07, mode, &xm), 0x0000_ff00, "green");
        assert_eq!(pixel_rgb(0x18, mode, &xm), 0x0000_00ff, "blue");
    }

    /// Bank 1 sits at the depth: 8-bit buffers at 7:0 and 15:8 (libglcore's
    /// 8-bit double-buffer masks 0x00FF / 0xFF00), 12-bit at 11:0 / 23:12.
    #[test]
    fn bank1_starts_at_the_depth() {
        let mut xm: Box<Xmap5> = unsafe { Box::new_zeroed().assume_init() };
        xm.clut[0x12] = 0x0000_0012;
        xm.clut[0x34] = 0x0000_0034;
        let ci8 = |pix_mode: u32| pix_mode << 24; // PD_MODE 0: colour index
        let p = 0x0056_3412;
        assert_eq!(pixel_rgb(p, ci8(2), &xm), 0x12, "8-bit bank 0: bits 7:0");
        assert_eq!(pixel_rgb(p, ci8(3), &xm), 0x34, "8-bit bank 1: bits 15:8");
        let rgb12 = |pix_mode: u32| (pix_mode << 24) | (PD_RGB << 16);
        assert_eq!(pixel_rgb(0x000f_000f, rgb12(4), &xm), 0x00ff_0000, "12-bit bank 0: red");
        assert_eq!(pixel_rgb(0x000f_000f, rgb12(5), &xm), 0x0000_ff00, "12-bit bank 1: green");
    }

    /// A line table with several DIDs (Xsgi login window: DID 3, DID 9 from
    /// x 326, DID 3 from x 940) switches the mode word per pixel.
    #[test]
    fn did_runs_switch_visual_mid_line() {
        let mut vc1: Box<Vc1> = unsafe { Box::new_zeroed().assume_init() };
        let mut xmap: Box<[Xmap5; 5]> = unsafe { Box::new_zeroed().assume_init() };
        let mut dac: Box<[Bt457; 3]> = unsafe { Box::new_zeroed().assume_init() };
        set16(&mut vc1.regs, 0x40, 0xb00);
        for y in 0..FB_H {
            set16(&mut vc1.sram, 0xb00 + y * 2, 0x7ff4);
        }
        for (k, v) in [3u16, 0x0003, 0x28c9, 0x7583].into_iter().enumerate() {
            set16(&mut vc1.sram, 0x7ff4 + k * 2, v);
        }
        for xm in xmap.iter_mut() {
            xm.mode[3] = 0x8af0_0800; // 8-bit CI, page 17
            xm.mode[9] = 0x07f1_0800; // 24-bit RGB
            xm.clut[17 * 256 + 5] = 0x00ff_0000;
        }
        for d in dac.iter_mut() {
            d.readmask = 0xff;
            for i in 0..256 {
                d.palette[i] = i as u8;
            }
        }
        let mut vram = vec![0u32; FB_W * FB_H];
        let row = FB_H - 1; // display line 0
        for x in 0..FB_W {
            vram[row * FB_W + x] = if (326..940).contains(&x) { 0x0030_2010 } else { 5 };
        }
        let mut out = vec![0u32; OUT_STRIDE * FB_H];
        compose(&vram, &vc1, &xmap, &dac, &mut out);
        assert_eq!(out[325], 0xff00_00ff, "DID 3: CLUT page 17 red");
        assert_eq!(out[326], 0xff30_2010, "DID 9: 24-bit RGB");
        assert_eq!(out[939], 0xff30_2010);
        assert_eq!(out[940], 0xff00_00ff, "back to DID 3");
    }
}
