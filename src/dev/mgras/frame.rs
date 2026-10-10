//! Display composition in two steps, the way Newport's compositor works:
//!
//! 1. `Frame::snapshot`, taken by the display thread just after the vertical
//!    blank (so the guest's retrace-time updates are in): copies the planes
//!    and decodes every piece of board state into flat buffers and small
//!    tables, in display order. All board-specific decisions are made here:
//!    the VC3 window-ID runs become a window ID per pixel (main and overlay),
//!    the XMAP mode words become a descriptor per window ID, the DAC gamma
//!    becomes an 8-bit table, the cursor glyph a 2-bit-per-pixel image.
//! 2. `Frame::compose`: one pass over the pixels, cursor first, then the
//!    overlay, then the main planes through their window ID's descriptor,
//!    then gamma. Each pixel depends only on the buffers at its own position
//!    and on the small tables, so the same pass can run as a fragment shader
//!    with the buffers uploaded as textures.
//!
//! The displayed size comes from the VC3 (see `display_size`): IRIX runs
//! the same board from 1024x768 to 1600x1200, with the screen's top row at
//! framebuffer row height - 1 (the window origin X uses). W x H is the
//! whole framebuffer, of which the screen is the bottom-left corner.
//!
//! Output: stride `OUT_STRIDE`, `0xFFBBGGRR` (red in the low byte), what the
//! renderers take as a prebuilt frame.

use super::dcb::Dcb;
use super::pixmem::{Buffer, Kind};
use super::rss::{self, Rss};

pub const W: usize = rss::WIDTH;
pub const H: usize = rss::HEIGHT;
pub const OUT_STRIDE: usize = crate::disp::FB_STRIDE;
/// What the board displays before IRIX loads its VC3 tables.
pub const DEFAULT_SIZE: (usize, usize) = (1280, 1024);
/// Largest cursor glyph (64x64).
pub const CURSOR_MAX: usize = 64;

/// How a window ID's main planes display. `rgb`: the pixel is `0x00BBGGRR`;
/// otherwise its low 12 bits index the colormap from `cmap_base`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MainMode {
    pub rgb: bool,
    pub cmap_base: u16,
    /// 12-bit RGB pixel pairs (see `rss::rgb12_pair`): 0 no, 1 buffer A
    /// (bits 11:0) always, 2 A or B (bits 23:12) by BUF_SELECT.
    pub rgb12: u8,
}

/// XMAP main mode formats (bits 4:0) that display 12-bit pixel pairs:
/// 7 double-buffered (the kernel's swap flips BUF_SELECT: traced, GL's
/// 12-bit double-buffered visual on a HighImpact at 1280x1024), 5 single.
/// Provisional: inferred from the 6.5.22 PseudoColor server's window IDs
/// (formats 4, 5, 7, 8) and the GL window's 7; 0x15 is GL's 24-bit
/// double-buffered visual (two pages).
pub(super) fn rgb12_format(mode: u32) -> u8 {
    match mode & 0x1F {
        7 => 2,
        5 => 1,
        _ => 0,
    }
}

/// How a window ID's overlay planes display: off, or a non-zero 8-bit value
/// indexing the colormap from `cmap_base` (0 shows the main planes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OverlayMode {
    pub on: bool,
    pub cmap_base: u16,
}

/// One displayed frame's worth of decoded board state. Plain data, valid
/// zeroed; build with `plain::boxed_zeroed` and fill with `snapshot`.
#[repr(C)]
pub struct Frame {
    /// Displayed size (at most W x H).
    pub width: usize,
    pub height: usize,
    /// Main planes, display order (row 0 at the top).
    pub main: [u32; W * H],
    /// Overlay planes (8 bits used), display order.
    pub overlay: [u8; W * H],
    /// Window ID of each pixel, main and overlay planes, display order.
    pub did_main: [u8; W * H],
    pub did_overlay: [u8; W * H],
    pub main_mode: [MainMode; 32],
    pub overlay_mode: [OverlayMode; 32],
    /// Colormap 0, `0x00RRGGBB`.
    pub cmap: [u32; 8192],
    /// DAC gamma reduced to 8 bits, per input level: [red, green, blue].
    pub gamma: [[u8; 3]; 256],
    /// DAC pixel mask is zero: the screen is blanked.
    pub blank: bool,
    pub cursor_on: bool,
    /// Top-left corner on screen, and the glyph's size (32 or 64).
    pub cursor_x: i32,
    pub cursor_y: i32,
    pub cursor_size: usize,
    /// Glyph, 2 bits a pixel (0 transparent), CURSOR_MAX x CURSOR_MAX.
    pub cursor: [u8; CURSOR_MAX * CURSOR_MAX],
    /// Colours of cursor values 1..3 (index 0 unused), `0x00RRGGBB`.
    pub cursor_rgb: [u32; 4],
}

impl Frame {
    /// Take the frame: copy the planes and decode the display state.
    pub fn snapshot(&mut self, rss: &Rss, dcb: &Dcb) {
        (self.width, self.height) = display_size(dcb);
        let (w, h) = (self.width, self.height);
        let (main, overlay) = scanout_buffers(rss, dcb);
        // Double buffering: window IDs with their BUF_SELECT bit show
        // buffer B. The window-ID runs are decoded below; this pass needs
        // them first, so decode them before the planes.
        let select = dcb.xmap.buf_select();
        let back = Buffer::new(dcb.xmap.dib_pointers() >> 10, Kind::Wide, rss.reg(rss::reg::DRBSIZE));
        let mut runs = Vec::with_capacity(64);
        for y in 0..h {
            dcb.vc3.main_did_runs(y, &mut runs);
            fill_did_row(&mut self.did_main[y * W..(y + 1) * W], &mut runs);
        }
        for did in 0..32u32 {
            let m = dcb.xmap.main_mode(did);
            self.main_mode[did as usize] = MainMode { rgb: m & 0x1F >= 4, cmap_base: (((m >> 5) & 0x1F) * 256) as u16, rgb12: rgb12_format(m) };
            let o = dcb.xmap.overlay_mode(did);
            self.overlay_mode[did as usize] = OverlayMode { on: o != 0, cmap_base: (((o >> 3) & 0x1F) * 256) as u16 };
        }
        let mut row = [0u32; W];
        let mut row_b = [0u32; W];
        for y in 0..h {
            let dst = y * W;
            let fb_y = (h - 1 - y) as u32;
            rss.mem.read_row(&main, fb_y, &mut self.main[dst..dst + w]);
            if select != 0 {
                rss.mem.read_row(&back, fb_y, &mut row_b[..w]);
            }
            for x in 0..w {
                let did = self.did_main[dst + x] & 31;
                let b = select >> did & 1 != 0;
                match self.main_mode[did as usize].rgb12 {
                    0 if b => self.main[dst + x] = row_b[x],
                    0 => {}
                    k => {
                        let half = if k == 2 && b { 12 } else { 0 };
                        self.main[dst + x] = rss::from_rgb12(self.main[dst + x] >> half & 0xFFF);
                    }
                }
            }
            match overlay {
                Some(o) => {
                    rss.mem.read_row(&o, fb_y, &mut row[..w]);
                    for (d, s) in self.overlay[dst..dst + w].iter_mut().zip(&row[..w]) {
                        *d = *s as u8;
                    }
                }
                None => self.overlay[dst..dst + w].fill(0),
            }
        }
        for y in 0..h {
            dcb.vc3.overlay_did_runs(y, &mut runs);
            fill_did_row(&mut self.did_overlay[y * W..(y + 1) * W], &mut runs);
        }
        self.cmap.copy_from_slice(&dcb.cmap[0].pal);
        for (g, d) in self.gamma.iter_mut().zip(dcb.dac.gamma.iter()) {
            *g = [(d[0] >> 2) as u8, (d[1] >> 2) as u8, (d[2] >> 2) as u8];
        }
        self.blank = dcb.dac.pixmask() == 0;
        self.cursor_on = false;
        if let Some((cx, cy, size, glyph)) = dcb.vc3.cursor() {
            self.cursor_on = true;
            (self.cursor_x, self.cursor_y, self.cursor_size) = (cx, cy, size);
            let sram = &dcb.vc3.sram;
            let words_per_row = size / 16;
            let plane_words = size * words_per_row;
            let bit = |plane: usize, row: usize, col: usize| -> u8 {
                let w = sram[(glyph + plane * plane_words + row * words_per_row + col / 16) & 0x7FFF];
                (w >> (15 - col % 16)) as u8 & 1
            };
            for row in 0..size {
                for col in 0..size {
                    self.cursor[row * CURSOR_MAX + col] = bit(0, row, col) | bit(1, row, col) << 1;
                }
            }
            // Cursor colours sit in the colormap at the XMAP's cursor base;
            // past its end they read as white.
            let base = dcb.xmap.cursor_cmap_base();
            for c in 1..4 {
                self.cursor_rgb[c] = self.cmap.get(base + c).copied().unwrap_or(0xFF_FFFF);
            }
        }
    }

    /// One pixel, `0xFFBBGGRR`. `x`, `y` in display coordinates.
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> u32 {
        if self.blank {
            return 0xFF00_0000;
        }
        let i = y * W + x;
        let rgb = self.cursor_value(x, y).map(|c| self.cursor_rgb[c as usize])
            .or_else(|| {
                let o = self.overlay_mode[self.did_overlay[i] as usize & 31];
                let v = self.overlay[i];
                (o.on && v != 0).then(|| self.cmap[(o.cmap_base as usize + v as usize) & 0x1FFF])
            });
        match rgb {
            Some(c) => self.gamma_rgb(c >> 16, c >> 8, c),
            None => {
                let m = self.main_mode[self.did_main[i] as usize & 31];
                let v = self.main[i];
                if m.rgb {
                    self.gamma_rgb(v, v >> 8, v >> 16)
                } else {
                    let c = self.cmap[(m.cmap_base as usize + (v & 0xFFF) as usize) & 0x1FFF];
                    self.gamma_rgb(c >> 16, c >> 8, c)
                }
            }
        }
    }

    /// The cursor's 2-bit value at a screen position, if the cursor covers
    /// it with a non-transparent pixel.
    #[inline]
    fn cursor_value(&self, x: usize, y: usize) -> Option<u8> {
        if !self.cursor_on {
            return None;
        }
        let (cx, cy) = (x as i32 - self.cursor_x, y as i32 - self.cursor_y);
        let size = self.cursor_size as i32;
        if !(0..size).contains(&cx) || !(0..size).contains(&cy) {
            return None;
        }
        let c = self.cursor[cy as usize * CURSOR_MAX + cx as usize];
        (c != 0).then_some(c)
    }

    #[inline]
    fn gamma_rgb(&self, r: u32, g: u32, b: u32) -> u32 {
        let lvl = |v: u32, comp: usize| self.gamma[(v & 0xFF) as usize][comp] as u32;
        0xFF00_0000 | lvl(b, 2) << 16 | lvl(g, 1) << 8 | lvl(r, 0)
    }

    /// The whole frame into `out` (stride `OUT_STRIDE`, `height` rows of
    /// `width` pixels).
    pub fn compose(&self, out: &mut [u32]) {
        for y in 0..self.height {
            let row = &mut out[y * OUT_STRIDE..y * OUT_STRIDE + self.width];
            for (x, o) in row.iter_mut().enumerate() {
                *o = self.pixel(x, y);
            }
        }
    }
}

/// The buffers scanout reads: the XMAP DIB pointers' main buffer and, when
/// its pointer differs, the overlay.
pub fn scanout_buffers(rss: &Rss, dcb: &Dcb) -> (Buffer, Option<Buffer>) {
    let dib = dcb.xmap.dib_pointers();
    let drbsize = rss.reg(rss::reg::DRBSIZE);
    let main = Buffer::new(dib, Kind::Wide, drbsize);
    let overlay = Buffer::new(dib >> 20, Kind::Overlay, drbsize);
    (main, (overlay.ptr != main.ptr).then_some(overlay))
}

/// The displayed size the VC3 is programmed for. Height: the scanlines in
/// the main window-ID frame table (one entry per visible line). Width: by
/// height, every IMPACT video format having its own (1280x492 stereo,
/// 1280x960 and 1920x1035 HDTV included). Only for a height not in the list,
/// the timing tables decoded with the VC2 layout, snapped to a standard
/// width; that decode is loose on the VC3 (978 for 1024, 1260 or 1680 for
/// 1280), its blanking bits not being pinned down. Before IRIX loads its
/// tables: `DEFAULT_SIZE`.
pub fn display_size(dcb: &Dcb) -> (usize, usize) {
    const WIDTHS: [usize; 7] = [640, 800, 1024, 1152, 1280, 1600, 1920];
    let lines = dcb.vc3.did_lines();
    let Some(h) = (1..=H.min(crate::disp::FB_MAX_H)).contains(&lines).then_some(lines) else {
        return DEFAULT_SIZE;
    };
    let w = match h {
        480 => 640,
        492 | 960 | 1024 => 1280,
        576 => 768,
        600 => 800,
        768 => 1024,
        864 => 1152,
        1035 => 1920,
        1200 => 1600,
        _ => match dcb.vc3.timing_size() {
            Some((tw, _)) => *WIDTHS.iter().min_by_key(|&&s| s.abs_diff(tw)).unwrap(),
            None => DEFAULT_SIZE.0,
        },
    };
    (w.min(W), h)
}

/// Window IDs of one display row from its VC3 runs `(first x, did)`. A row
/// with no run at x = 0 starts in window ID 0; each run lasts until the
/// next one's x (clamped to the screen); runs out of order overwrite the
/// pixels earlier ones set.
fn fill_did_row(row: &mut [u8], runs: &mut Vec<(u16, u8)>) {
    row.fill(0);
    if runs.is_empty() || runs[0].0 != 0 {
        runs.insert(0, (0, 0));
    }
    for (k, &(x0, did)) in runs.iter().enumerate() {
        let x0 = (x0 as usize).min(W);
        let x1 = runs.get(k + 1).map_or(W, |r| (r.0 as usize).min(W));
        if x1 > x0 {
            row[x0..x1].fill(did);
        }
    }
}
