//! Prebuilt Newport (VC2) video timing tables for host-selected display modes.
//!
//! IRIX normally programs VC2 via the X/setmon path; these presets let the GUI
//! force standard 4:3 / 5:4 Indy resolutions at VM start before the guest runs.

use serde::{Deserialize, Serialize};

use crate::vc2::{
    Vc2, VC2_CTRL_BLACKOUT, VC2_CTRL_CURSOR_EN, VC2_CTRL_CURSOR_FUNC_EN,
    VC2_CTRL_VIDEO_TIMING_EN, VC2_REG_DISPLAY_CONTROL, VC2_REG_SCANLINE_LEN,
    VC2_REG_VIDEO_ENTRY_PTR, VC2_REG_VT_FRAME_PTR, VT_CBLANK_XMAP_N, VT_HPOS_VC_N,
};

/// Standard Newport modes that fit the 1280×1024 framebuffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NewportResolution {
    /// Leave VC2 untouched — IRIX/PROM programs the mode.
    #[default]
    Guest,
    /// 1024×768 @ ~60 Hz (4:3).
    #[serde(rename = "1024x768")]
    Res1024x768,
    /// 1280×960 (4:3 within 1024-line VRAM).
    #[serde(rename = "1280x960")]
    Res1280x960,
    /// 1280×1024 @ ~72 Hz (5:4 Indy default).
    #[serde(rename = "1280x1024")]
    Res1280x1024,
}

impl NewportResolution {
    /// All selectable modes, in display order. Single source of truth for the
    /// GUI dropdowns (Config tab + New Machine dialog) so they never drift.
    pub const ALL: [Self; 4] = [
        Self::Guest,
        Self::Res1024x768,
        Self::Res1280x960,
        Self::Res1280x1024,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Guest => "Guest (IRIX / setmon)",
            Self::Res1024x768 => "1024×768",
            Self::Res1280x960 => "1280×960",
            Self::Res1280x1024 => "1280×1024 (default)",
        }
    }

    pub fn is_guest(self) -> bool {
        matches!(self, Self::Guest)
    }

    /// Visible pixel size for host window sizing; `None` when guest-controlled.
    pub fn visible_size(self) -> Option<(u32, u32)> {
        match self {
            Self::Guest => None,
            Self::Res1024x768 => Some((1024, 768)),
            Self::Res1280x960 => Some((1280, 960)),
            Self::Res1280x1024 => Some((1280, 1024)),
        }
    }

    fn geometry(self) -> Option<(u16, u16, u16)> {
        // (width, height, horizontal blank before active in pixels)
        match self {
            Self::Guest => None,
            Self::Res1024x768 => Some((1024, 768, 320)),
            Self::Res1280x960 => Some((1280, 960, 408)),
            Self::Res1280x1024 => Some((1280, 1024, 408)),
        }
    }
}

/// Decode visible width/height from VC2 register file + RAM (shared with compositor).
pub fn decode_vc2_visible_size(regs: &[u16; 32], ram: &[u16]) -> (usize, usize) {
    use crate::disp::decode_vc2_timings;
    let (w, h, _) = decode_vc2_timings(regs, ram);
    (w, h)
}

fn pack_line_word(duration: u8, state_a: u8, state_c: u8, eol: bool) -> [u16; 2] {
    let w1 = ((eol as u16) << 15) | ((duration as u16) << 8) | (state_a as u16);
    [w1, state_c as u16]
}

fn push_pixels(
    ram: &mut [u16],
    ptr: &mut usize,
    pixels: u16,
    state_a: u8,
    state_c: u8,
    eol: bool,
) {
    let mut left = pixels;
    while left > 0 {
        let chunk_px = left.min(254);
        let dur = (chunk_px / 2).max(1) as u8;
        let words = pack_line_word(dur, state_a, state_c, eol && left <= chunk_px);
        ram[*ptr] = words[0];
        ram[*ptr + 1] = words[1];
        *ptr += 2;
        left = left.saturating_sub(chunk_px);
    }
}

fn build_mode_ram(width: u16, height: u16, h_blank_lead: u16) -> ([u16; 32], Vec<u16>) {
    const FRAME_PTR: usize = 0x100;
    const LINE_SEQ_PTR: usize = 0x200;

    let mut ram = vec![0u16; 32768];
    let mut seq = LINE_SEQ_PTR;

    // Horizontal blanking before active video.
    if h_blank_lead > 0 {
        push_pixels(&mut ram, &mut seq, h_blank_lead, 0, 0, false);
    }
    // HPOS pulse so cursor timing decode finds a leading edge.
    {
        let words = pack_line_word(2, VT_HPOS_VC_N, 0, false);
        ram[seq] = words[0];
        ram[seq + 1] = words[1];
        seq += 2;
        let words = pack_line_word(2, 0, 0, false);
        ram[seq] = words[0];
        ram[seq + 1] = words[1];
        seq += 2;
    }
    // Gap between the HPOS leading edge and the start of active video. decode_vc2_timings
    // computes cursor_x_adjust = (visible_pixel - hpos_pixel) - 31, valid over [0, 64];
    // without this gap the two land right next to each other and the result is negative
    // (out of range), so the decoder falls back to a hardcoded default. 36 pixels here
    // yields hpos_to_visible=40 -> cursor_x_adjust=9, comfortably in range.
    push_pixels(&mut ram, &mut seq, 36, 0, 0, false);
    // Active video: composite blank deasserted to XMAP.
    push_pixels(&mut ram, &mut seq, width, 0, VT_CBLANK_XMAP_N, true);

    // Frame table: one run of `height` identical lines.
    ram[FRAME_PTR] = LINE_SEQ_PTR as u16;
    ram[FRAME_PTR + 1] = height;

    let mut regs = [0u16; 32];
    regs[VC2_REG_VIDEO_ENTRY_PTR as usize] = FRAME_PTR as u16;
    regs[VC2_REG_VT_FRAME_PTR as usize] = FRAME_PTR as u16;
    regs[VC2_REG_SCANLINE_LEN as usize] = width << 5;
    regs[VC2_REG_DISPLAY_CONTROL as usize] = VC2_CTRL_BLACKOUT
        | VC2_CTRL_VIDEO_TIMING_EN
        | VC2_CTRL_CURSOR_FUNC_EN
        | VC2_CTRL_CURSOR_EN;

    (regs, ram)
}

/// Program VC2 RAM/regs for a standard Newport resolution.
pub fn apply_newport_resolution(vc2: &mut Vc2, mode: NewportResolution) {
    let Some((width, height, h_blank)) = mode.geometry() else {
        return;
    };
    let (regs, ram) = build_mode_ram(width, height, h_blank);
    vc2.regs = regs;
    vc2.ram = ram;
    vc2.dirty = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_decode_to_requested_size() {
        for mode in [
            NewportResolution::Res1024x768,
            NewportResolution::Res1280x960,
            NewportResolution::Res1280x1024,
        ] {
            let Some((ew, eh, _)) = mode.geometry().map(|(w, h, b)| (w as usize, h as usize, b)) else {
                continue;
            };
            let (regs, ram) = {
                let (r, ram) = build_mode_ram(
                    mode.geometry().unwrap().0,
                    mode.geometry().unwrap().1,
                    mode.geometry().unwrap().2,
                );
                (r, ram)
            };
            let (w, h) = decode_vc2_visible_size(&regs, &ram);
            assert_eq!(h, eh, "mode {:?} height", mode);
            assert!(
                (w as i32 - ew as i32).abs() <= 4,
                "mode {:?}: width {} (expected {})",
                mode,
                w,
                ew
            );
        }
    }

    #[test]
    fn presets_hpos_to_visible_gap_stays_in_range() {
        // build_mode_ram lays out: [h_blank_lead][HPOS pulse: 4px][gap][active video].
        // decode_vc2_timings computes hpos_to_visible as the pixel distance from the
        // HPOS falling edge to the first active-video pixel, then cursor_x_adjust =
        // hpos_to_visible - 31, which must land in [0, 64] or the decoder falls back
        // to a hardcoded default (see disp.rs). That distance is fixed by the HPOS
        // pulse width (4px) plus the gap below, independent of h_blank_lead/mode.
        const HPOS_PULSE_PIXELS: i32 = 4;
        const GAP_PIXELS: i32 = 36;
        let hpos_to_visible = HPOS_PULSE_PIXELS + GAP_PIXELS;
        let cursor_x_adjust = hpos_to_visible - 31;
        assert!(
            (0..=64).contains(&cursor_x_adjust),
            "hpos_to_visible={} gives out-of-range cursor_x_adjust={}",
            hpos_to_visible,
            cursor_x_adjust
        );

        // Cross-check against the real decoder for every preset mode.
        use crate::disp::decode_vc2_timings;
        for mode in [
            NewportResolution::Res1024x768,
            NewportResolution::Res1280x960,
            NewportResolution::Res1280x1024,
        ] {
            let (width, height, h_blank) = mode.geometry().unwrap();
            let (regs, ram) = build_mode_ram(width, height, h_blank);
            let (_, _, decoded_adjust) = decode_vc2_timings(&regs, &ram);
            assert_eq!(
                decoded_adjust,
                cursor_x_adjust - 2, // decode_vc2_timings applies a further -2 bias
                "mode {:?}: cursor_x_adjust {} did not match expected in-range value",
                mode,
                decoded_adjust
            );
        }
    }
}
