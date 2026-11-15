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
/// otherwise its low `index_mask` bits index the colormap from `cmap_base`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MainMode {
    pub rgb: bool,
    pub cmap_base: u16,
    /// Colour index planes shown: 8 for format 1 (the X server's CI8
    /// windows draw through mask 0xFF and leave the planes above alone,
    /// so they hold whatever an earlier window wrote), 12 otherwise.
    pub index_mask: u16,
    /// Pixel pairs (see `rss::Pair`) in this layout: buffer A, or B when
    /// the window ID's BUF_SELECT bit is set.
    pub pair: Option<rss::Pair>,
}

impl MainMode {
    /// Decode an XMAP main mode word: format in bits 4:0 (4 and up RGB),
    /// colormap block in 9:5.
    pub fn of(m: u32) -> Self {
        MainMode {
            rgb: m & 0x1F >= 4,
            cmap_base: (((m >> 5) & 0x1F) * 256) as u16,
            index_mask: if m & 0x1F == 1 { 0xFF } else { 0xFFF },
            pair: pair_format(m),
        }
    }
}

/// XMAP main mode formats (bits 4:0) that display pixel pairs, traced
/// with GL's 12-bit double-buffered visuals, the kernel's swaps flipping
/// BUF_SELECT: 5 the 4:4:4 layout (RGBA 4,4,4,4: electropaint, PP1 pixel
/// type 0) and 7 the 5:5:5 one (RGBA 5,5,5,1: octahedra, pixel type 1).
/// 0x15 is GL's 24-bit double-buffered visual (two pages).
pub(super) fn pair_format(mode: u32) -> Option<rss::Pair> {
    match mode & 0x1F {
        5 => Some(rss::Pair::P444),
        7 => Some(rss::Pair::P555),
        _ => None,
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
        for did in 0..32u32 {
            let m = dcb.xmap.main_mode(did);
            self.main_mode[did as usize] = MainMode::of(m);
            let o = dcb.xmap.overlay_mode(did);
            self.overlay_mode[did as usize] = OverlayMode { on: o != 0, cmap_base: (((o >> 3) & 0x1F) * 256) as u16 };
        }
        // Window IDs whose pixels are the main buffer's words as they are:
        // a row of only those needs no pass over its pixels.
        let plain: [bool; 32] = std::array::from_fn(|did| self.main_mode[did].pair.is_none() && select >> did & 1 == 0);
        let mut runs = Vec::with_capacity(64);
        let mut row = [0u32; W];
        let mut row_b = [0u32; W];
        for y in 0..h {
            let dst = y * W;
            let fb_y = (h - 1 - y) as u32;
            dcb.vc3.main_did_runs(y, &mut runs);
            fill_did_row(&mut self.did_main[dst..dst + W], &mut runs);
            rss.mem.read_row(&main, fb_y, &mut self.main[dst..dst + w]);
            if runs.iter().any(|r| !plain[r.1 as usize & 31]) {
                if select != 0 {
                    rss.mem.read_row(&back, fb_y, &mut row_b[..w]);
                }
                for x in 0..w {
                    let did = self.did_main[dst + x] & 31;
                    let b = select >> did & 1 != 0;
                    let mode = self.main_mode[did as usize];
                    if let Some(pair) = mode.pair {
                        self.main[dst + x] = pair.unpack(self.main[dst + x], b) & 0xFF_FFFF;
                    } else if b && !mode.rgb {
                        // Colour index: B is the word's high 12 bits
                        // (`rss::ci_pixtype`).
                        self.main[dst + x] = (self.main[dst + x] >> 12) & 0xFFF;
                    } else if b {
                        self.main[dst + x] = row_b[x];
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
                    let c = self.cmap[(m.cmap_base as usize + (v & m.index_mask as u32) as usize) & 0x1FFF];
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
    /// `width` pixels): `pixel` for every pixel, in three passes a row. The
    /// main planes go a span of one window ID at a time, through its mode
    /// decided once; the overlay shows over them where it is on and not 0;
    /// the cursor over both, in its own rows.
    pub fn compose(&self, out: &mut [u32]) {
        let (w, h) = (self.width, self.height);
        if self.blank {
            for y in 0..h {
                out[y * OUT_STRIDE..y * OUT_STRIDE + w].fill(0xFF00_0000);
            }
            return;
        }
        // The gamma table a channel, each entry in its byte of the pixel.
        let mut gamma = [[0u32; 256]; 3];
        for (v, g) in self.gamma.iter().enumerate() {
            for c in 0..3 {
                gamma[c][v] = (g[c] as u32) << (8 * c);
            }
        }
        // A 0x00RRGGBB colour through gamma.
        let lit = |c: u32| 0xFF00_0000 | gamma[0][(c >> 16) as usize & 0xFF] | gamma[1][(c >> 8) as usize & 0xFF] | gamma[2][c as usize & 0xFF];
        let overlay_on = self.overlay_mode.iter().any(|o| o.on);
        for y in 0..h {
            let at = y * W;
            let row = &mut out[y * OUT_STRIDE..y * OUT_STRIDE + w];
            let (main, dids) = (&self.main[at..at + w], &self.did_main[at..at + w]);
            let mut x = 0;
            while x < w {
                let did = dids[x];
                let end = x + dids[x..].iter().position(|&d| d != did).unwrap_or(w - x);
                let m = self.main_mode[did as usize & 31];
                if m.rgb {
                    for (o, &v) in row[x..end].iter_mut().zip(&main[x..end]) {
                        *o = 0xFF00_0000 | gamma[0][v as usize & 0xFF] | gamma[1][(v >> 8) as usize & 0xFF] | gamma[2][(v >> 16) as usize & 0xFF];
                    }
                } else {
                    let (base, mask) = (m.cmap_base as usize, m.index_mask as u32);
                    for (o, &v) in row[x..end].iter_mut().zip(&main[x..end]) {
                        *o = lit(self.cmap[(base + (v & mask) as usize) & 0x1FFF]);
                    }
                }
                x = end;
            }
            if overlay_on {
                let (over, dids) = (&self.overlay[at..at + w], &self.did_overlay[at..at + w]);
                for (x, &v) in over.iter().enumerate() {
                    if v != 0 {
                        let o = self.overlay_mode[dids[x] as usize & 31];
                        if o.on {
                            row[x] = lit(self.cmap[(o.cmap_base as usize + v as usize) & 0x1FFF]);
                        }
                    }
                }
            }
            let cy = y as i32 - self.cursor_y;
            if self.cursor_on && (0..self.cursor_size as i32).contains(&cy) {
                let x0 = self.cursor_x.max(0) as usize;
                let x1 = ((self.cursor_x + self.cursor_size as i32).max(0) as usize).min(w);
                for x in x0..x1 {
                    if let Some(c) = self.cursor_value(x, y) {
                        row[x] = lit(self.cursor_rgb[c as usize]);
                    }
                }
            }
        }
    }
}

/// The hardware cursor as a frame shows it.
struct Cursor {
    /// Top-left corner on screen, and the glyph's size (32 or 64).
    x: i32,
    y: i32,
    size: usize,
    /// Glyph, 2 bits a pixel (0 transparent), CURSOR_MAX x CURSOR_MAX.
    glyph: [u8; CURSOR_MAX * CURSOR_MAX],
    /// Colours of values 1..3, `0x00RRGGBB`.
    rgb: [u32; 4],
}

impl Cursor {
    /// The cursor the VC3 shows, its colours from colormap `cmap`.
    fn read(dcb: &Dcb, cmap: &[u32]) -> Option<Cursor> {
        let (x, y, size, at) = dcb.vc3.cursor()?;
        let mut c = Cursor { x, y, size, glyph: [0; CURSOR_MAX * CURSOR_MAX], rgb: [0; 4] };
        let sram = &dcb.vc3.sram;
        let words_per_row = size / 16;
        let plane_words = size * words_per_row;
        let bit = |plane: usize, row: usize, col: usize| -> u8 {
            let w = sram[(at + plane * plane_words + row * words_per_row + col / 16) & 0x7FFF];
            (w >> (15 - col % 16)) as u8 & 1
        };
        for row in 0..size {
            for col in 0..size {
                c.glyph[row * CURSOR_MAX + col] = bit(0, row, col) | bit(1, row, col) << 1;
            }
        }
        // Cursor colours sit in the colormap at the XMAP's cursor base;
        // past its end they read as white.
        let base = dcb.xmap.cursor_cmap_base();
        for k in 1..4 {
            c.rgb[k] = cmap.get(base + k).copied().unwrap_or(0xFF_FFFF);
        }
        Some(c)
    }
}

/// How a window ID's pixels are read from the pages and shown.
#[derive(Clone, Copy)]
enum Show {
    /// The low 24 planes of the main buffer's word, or the back buffer's.
    Rgb { back: bool },
    /// Half a pixel pair, through table `pair` of the frame: the index is
    /// the word shifted down and masked.
    Pair { table: usize, shift: u32, mask: u32 },
    /// A colour index: the word shifted down and masked, from `base`.
    Index { shift: u32, mask: u32, base: usize },
}

/// Compose the displayed frame straight from the pixel pages into `out`
/// (stride `OUT_STRIDE`; `0xFFBBGGRR`), with no copy of the planes between:
/// what `Frame::snapshot` and `Frame::compose` make, in one pass, for the
/// display's own frames. Each window ID's way to a colour is decided once
/// (`Show`), through tables that already hold the gamma: the colormap, and
/// each pixel-pair half that some window ID uses. A row goes a span of one
/// window ID at a time, a tile's worth of contiguous words at a go; the
/// overlay shows over it where a word of its planes is not 0; the cursor
/// over both. The pages are read while the engines draw, as the board's
/// scan-out reads them. Returns the displayed size.
pub fn compose_direct(rss: &Rss, dcb: &Dcb, out: &mut [u32]) -> (usize, usize) {
    use super::pixmem::{OVERLAY_TILE_W, TILE_W};
    let (w, h) = display_size(dcb);
    if dcb.dac.pixmask() == 0 {
        for y in 0..h {
            out[y * OUT_STRIDE..y * OUT_STRIDE + w].fill(0xFF00_0000);
        }
        return (w, h);
    }
    let (main, overlay) = scanout_buffers(rss, dcb);
    let select = dcb.xmap.buf_select();
    let back = Buffer::new(dcb.xmap.dib_pointers() >> 10, Kind::Wide, rss.reg(rss::reg::DRBSIZE));
    // The gamma table a channel, each entry in its byte of the pixel.
    let mut gamma = [[0u32; 256]; 3];
    for (v, g) in dcb.dac.gamma.iter().enumerate() {
        for c in 0..3 {
            gamma[c][v] = ((g[c] >> 2) as u8 as u32) << (8 * c);
        }
    }
    // A 0x00BBGGRR pixel, and a 0x00RRGGBB colour, through gamma.
    let lit_pixel = |v: u32| 0xFF00_0000 | gamma[0][v as usize & 0xFF] | gamma[1][(v >> 8) as usize & 0xFF] | gamma[2][(v >> 16) as usize & 0xFF];
    let lit = |c: u32| 0xFF00_0000 | gamma[0][(c >> 16) as usize & 0xFF] | gamma[1][(c >> 8) as usize & 0xFF] | gamma[2][c as usize & 0xFF];
    let cmap = &dcb.cmap[0].pal;
    let mut lit_cmap = [0u32; 8192];
    for (o, &c) in lit_cmap.iter_mut().zip(cmap.iter()) {
        *o = lit(c);
    }
    // Per window ID. A pair half's table is built when an ID first needs it.
    let mut pair_tables: [Vec<u32>; 2] = [Vec::new(), Vec::new()];
    let mut overlay_modes = [OverlayMode::default(); 32];
    let shows: [Show; 32] = std::array::from_fn(|did| {
        let mode = MainMode::of(dcb.xmap.main_mode(did as u32));
        let o = dcb.xmap.overlay_mode(did as u32);
        overlay_modes[did] = OverlayMode { on: o != 0, cmap_base: (((o >> 3) & 0x1F) * 256) as u16 };
        let b = select >> did & 1 != 0;
        match mode.pair {
            Some(pair) => {
                let table = (pair == rss::Pair::P555) as usize;
                let bits = pair.shift_b();
                if pair_tables[table].is_empty() {
                    pair_tables[table] = (0..1u32 << bits).map(|i| lit_pixel(pair.unpack(i, false) & 0xFF_FFFF)).collect();
                }
                Show::Pair { table, shift: if b { bits } else { 0 }, mask: (1 << bits) - 1 }
            }
            None if mode.rgb => Show::Rgb { back: b },
            // Colour index: B is the word's high 12 bits (`rss::ci_pixtype`).
            None => Show::Index { shift: if b { 12 } else { 0 }, mask: if b { 0xFFF & mode.index_mask as u32 } else { mode.index_mask as u32 }, base: mode.cmap_base as usize },
        }
    });
    let overlay = overlay.filter(|_| overlay_modes.iter().any(|o| o.on));
    let cursor = Cursor::read(dcb, cmap);
    let mut runs = Vec::with_capacity(64);
    let mut dids = [0u8; W];
    let mut over_dids = [0u8; W];
    let words = &rss.mem.words;
    for y in 0..h {
        let fb_y = (h - 1 - y) as u32;
        let row = &mut out[y * OUT_STRIDE..y * OUT_STRIDE + w];
        dcb.vc3.main_did_runs(y, &mut runs);
        fill_did_row(&mut dids, &mut runs);
        let mut x = 0;
        while x < w {
            let did = dids[x];
            let end = x + dids[x..w].iter().position(|&d| d != did).unwrap_or(w - x);
            let show = shows[did as usize & 31];
            let buffer = if matches!(show, Show::Rgb { back: true }) { &back } else { &main };
            // A tile's row of the span at a time: contiguous words.
            while x < end {
                let n = (end - x).min(TILE_W - x % TILE_W);
                let at = buffer.locate(x as u32, fb_y).0;
                let (dst, src) = (&mut row[x..x + n], &words[at..at + n]);
                match show {
                    Show::Rgb { .. } => {
                        for (o, &v) in dst.iter_mut().zip(src) {
                            *o = lit_pixel(v as u32);
                        }
                    }
                    Show::Pair { table, shift, mask } => {
                        let table = &pair_tables[table];
                        for (o, &v) in dst.iter_mut().zip(src) {
                            *o = table[((v as u32 >> shift) & mask) as usize];
                        }
                    }
                    Show::Index { shift, mask, base } => {
                        for (o, &v) in dst.iter_mut().zip(src) {
                            *o = lit_cmap[(base + ((v as u32 >> shift) & mask) as usize) & 0x1FFF];
                        }
                    }
                }
                x += n;
            }
        }
        if let Some(over) = &overlay {
            // Four 9-bit pixels a word (`Buffer::locate`); mostly 0.
            let mut have_dids = false;
            for x0 in (0..w).step_by(OVERLAY_TILE_W) {
                let at = over.locate(x0 as u32, fb_y).0;
                let n = (w - x0).min(OVERLAY_TILE_W);
                for (k, &word) in words[at..at + n.div_ceil(4)].iter().enumerate() {
                    if word & 0xF_FFFF_FFFF == 0 {
                        continue;
                    }
                    for j in 0..4 {
                        let (x, v) = (x0 + 4 * k + j, (word >> (9 * j)) as u8);
                        if v == 0 || x >= w {
                            continue;
                        }
                        if !have_dids {
                            have_dids = true;
                            dcb.vc3.overlay_did_runs(y, &mut runs);
                            fill_did_row(&mut over_dids, &mut runs);
                        }
                        let o = overlay_modes[over_dids[x] as usize & 31];
                        if o.on {
                            row[x] = lit_cmap[(o.cmap_base as usize + v as usize) & 0x1FFF];
                        }
                    }
                }
            }
        }
        if let Some(c) = &cursor {
            let cy = y as i32 - c.y;
            if (0..c.size as i32).contains(&cy) {
                let x0 = c.x.max(0) as usize;
                let x1 = ((c.x + c.size as i32).max(0) as usize).min(w);
                for x in x0..x1 {
                    let v = c.glyph[cy as usize * CURSOR_MAX + (x as i32 - c.x) as usize];
                    if v != 0 {
                        row[x] = lit(c.rgb[v as usize]);
                    }
                }
            }
        }
    }
    (w, h)
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
pub(super) fn fill_did_row(row: &mut [u8], runs: &mut Vec<(u16, u8)>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::plain;

    /// `compose` is `pixel` for every pixel: over window-ID spans of every
    /// mode, overlay pixels on and off, and a cursor half off each edge.
    #[test]
    fn compose_matches_the_pixel_reference() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 16) as u32
        };
        for trial in 0..24 {
            let mut f = plain::boxed_zeroed::<Frame>();
            (f.width, f.height) = (300 + (rnd() % 40) as usize, 80);
            for did in 0..32 {
                f.main_mode[did] = MainMode { rgb: rnd() % 2 == 0, cmap_base: ((rnd() % 32) * 256) as u16, index_mask: if rnd() % 2 == 0 { 0xFF } else { 0xFFF }, pair: None };
                f.overlay_mode[did] = OverlayMode { on: trial % 3 != 0 && rnd() % 2 == 0, cmap_base: ((rnd() % 32) * 256) as u16 };
            }
            for y in 0..f.height {
                // Runs of a window ID, a few pixels to the whole row.
                let (mut x, row) = (0, y * W);
                while x < f.width {
                    let n = (1 + rnd() as usize % if rnd() % 4 == 0 { 400 } else { 20 }).min(f.width - x);
                    let (dm, dov) = ((rnd() % 32) as u8, (rnd() % 32) as u8);
                    for i in row + x..row + x + n {
                        f.did_main[i] = dm;
                        f.did_overlay[i] = dov;
                        f.main[i] = rnd() & 0xFF_FFFF;
                        f.overlay[i] = if rnd() % 5 == 0 { rnd() as u8 } else { 0 };
                    }
                    x += n;
                }
            }
            for c in f.cmap.iter_mut() {
                *c = rnd() & 0xFF_FFFF;
            }
            for g in f.gamma.iter_mut() {
                *g = [rnd() as u8, rnd() as u8, rnd() as u8];
            }
            f.blank = trial == 23;
            f.cursor_on = trial % 2 == 0;
            f.cursor_size = if trial % 4 == 0 { 64 } else { 32 };
            (f.cursor_x, f.cursor_y) = [(-10, -10), (290, 60), (100, 20), (f.width as i32 - 5, 75)][trial / 2 % 4];
            for c in f.cursor.iter_mut() {
                *c = (rnd() % 4) as u8;
            }
            f.cursor_rgb = [0, rnd() & 0xFF_FFFF, rnd() & 0xFF_FFFF, rnd() & 0xFF_FFFF];
            let mut out = vec![0x5555_5555u32; OUT_STRIDE * f.height];
            f.compose(&mut out);
            for y in 0..f.height {
                for x in 0..f.width {
                    assert_eq!(out[y * OUT_STRIDE + x], f.pixel(x, y), "trial {trial}: ({x}, {y})");
                }
                assert_eq!(out[y * OUT_STRIDE + f.width], 0x5555_5555, "trial {trial}: row {y} past the width");
            }
        }
    }

    /// `compose_direct` makes the frame `snapshot` and `compose` make, pixel
    /// for pixel, over random pixel memory and random display state:
    /// window-ID tables (runs out of order, past the edge, rows with none),
    /// every main format (colour index, RGB, both pixel-pair layouts),
    /// buffer selects, overlays, colormap blocks, gamma, blanking, cursors,
    /// one and two raster engines.
    #[test]
    fn the_direct_compose_is_the_snapshot_s() {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u64 {
                self.0 ^= self.0 >> 12;
                self.0 ^= self.0 << 25;
                self.0 ^= self.0 >> 27;
                self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
            }
            fn u32(&mut self) -> u32 {
                (self.next() >> 32) as u32
            }
            fn below(&mut self, n: u64) -> u64 {
                self.next() % n.max(1)
            }
            fn chance(&mut self, p: f64) -> bool {
                (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
            }
        }
        let mut rng = Rng(0x5CA2_0002);
        let mut r = Rss::new_boxed();
        for w in r.mem.words.iter_mut() {
            *w = rng.next();
        }
        let mut dcb = plain::boxed_zeroed::<Dcb>();
        dcb.init([0, 0], [0, 0]);
        let mut snap = plain::boxed_zeroed::<Frame>();
        let mut want = vec![0u32; OUT_STRIDE * H];
        let mut got = vec![0u32; OUT_STRIDE * H];
        for trial in 0..12 {
            r.write(rss::reg::DRBSIZE, (rng.chance(0.5) as u32) << 6 | (5 + rng.below(3) as u32) << 2 | 2, false);
            for did in 0..32 {
                let format = match rng.below(6) {
                    0 => 1,
                    1 => 2,
                    2 => 5,
                    3 => 7,
                    4 => 0x15,
                    _ => rng.below(32) as u32,
                };
                let overlay = if rng.chance(0.5) { 0 } else { (rng.below(32) as u32) << 3 | (1 + rng.below(7) as u32) };
                dcb.xmap.set_modes(did, format | (rng.below(32) as u32) << 5, overlay);
            }
            let page = |rng: &mut Rng| rng.below(1024) as u32;
            let (a, b) = (page(&mut rng), page(&mut rng));
            let o = if rng.chance(0.2) { a } else { page(&mut rng) };
            dcb.xmap.set_scanout(rng.u32(), a | b << 10 | o << 20, rng.u32());
            for c in dcb.cmap[0].pal.iter_mut() {
                *c = rng.u32() & 0xFF_FFFF;
            }
            for gm in dcb.dac.gamma.iter_mut() {
                *gm = [0, 1, 2].map(|_| rng.below(1024) as u16);
            }
            dcb.dac.set_pixmask(if trial == 11 { 0 } else { 0xFF });
            // Window-ID tables: frame tables at 0 (main) and 0x800
            // (overlay), 32 line tables from 0x2000 the rows share.
            let h = if rng.chance(0.5) { 1024 } else { 768 };
            let vc3 = &mut dcb.vc3;
            vc3.regs[4] = 0;
            vc3.regs[5] = 0x800;
            for k in 0..32usize {
                let base = 0x2000 + 64 * k;
                let n = rng.below(10) as usize;
                for e in 0..n {
                    let x = if rng.chance(0.1) { 1300 + rng.below(800) as u16 } else if e == 0 && rng.chance(0.5) { 0 } else { rng.below(1300) as u16 };
                    vc3.sram[base + e] = x << 5 | rng.below(32) as u16;
                }
                vc3.sram[base + n] = 0x7FF << 5;
            }
            for y in 0..h {
                vc3.sram[y] = 0x2000 + 64 * rng.below(32) as u16;
                vc3.sram[0x800 + y] = if rng.chance(0.05) { 0xFFFF } else { 0x2000 + 64 * rng.below(32) as u16 };
            }
            vc3.sram[h] = 0xFFFF;
            // A cursor, mostly.
            vc3.regs[0x1D] = if rng.chance(0.8) { 3 | if rng.chance(0.5) { 8 } else { 0 } } else { 0 };
            vc3.regs[1] = 0x6000;
            vc3.regs[2] = rng.below(1400) as u16;
            vc3.regs[3] = rng.below(1100) as u16;
            for w in vc3.sram[0x6000..0x6200].iter_mut() {
                *w = rng.u32() as u16;
            }
            snap.snapshot(&r, &dcb);
            snap.compose(&mut want);
            got.fill(0x5555_5555);
            assert_eq!(compose_direct(&r, &dcb, &mut got), (snap.width, snap.height), "trial {trial}: size");
            for y in 0..snap.height {
                for x in 0..snap.width {
                    let i = y * OUT_STRIDE + x;
                    assert_eq!(got[i], want[i], "trial {trial}: ({x}, {y}), window ID {}", snap.did_main[y * W + x]);
                }
                assert_eq!(got[y * OUT_STRIDE + snap.width], 0x5555_5555, "trial {trial}: row {y} past the width");
            }
        }
    }
}
