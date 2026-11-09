//! What the board shows: the tables the scan-out reads and the composition of a
//! frame from the pixel store.
//!
//! The display reads the store as it is, without waiting for drawing (a frame
//! can show a pixel being drawn; the next frame reads it again). It composes
//! the whole managed area per row from the store and the tables, choosing what
//! to do with a run of pixels once and not per pixel.
//!
//! Deferred, as speed work the model does not do: composing by runs of equal
//! (main, overlay) ID pairs, and presenting only the changed lines (the board
//! presents every frame, as VPro's does). The popup and overlay planes stay word
//! positions beside the display-ID plane; splitting them into a main-ID and an
//! overlay-ID plane would move the recorded stores' hashes and is not done.
//!
//! # What a pixel word is
//!
//! A pixel's colour word means what its window's display ID says. The ID plane
//! (bits 7:2 of each pixel's word there; "bits 7:2" is not confirmed, and the
//! board has two layouts for the ID in a pixel, chosen by its depth) names a
//! window; the tables the X server loads (`tables`) say, from the ID's entry and
//! its mode byte together, whether that window's words are an 8-bit or a 12-bit
//! index into the X colour map, or direct colour. Colour map, then gamma:
//! inferred, solid. A pixel whose ID has no entry, and a board that has been
//! sent no table, use the default descriptor: the front colour buffer at level 0,
//! one word a pixel, read as a colour index into the textport's colour map. That
//! is what the PROM's console needs.
//!
//! The popup and overlay planes go over the main window: popup over overlay over
//! main, index 0 letting the layer below through (inferred, `compose_layers`).
//!
//! # The cursor
//!
//! The glyph cursor here is the XMAP's (cursor mode 0x103). The XMAP injects it
//! ahead of the colour map, so it replaces the pixel's colour before gamma and
//! gamma applies to it (inferred, solid; whether gamma is bypassed for it is
//! unknown). Its colours therefore go through the same gamma as the colour map's
//! (`Tables::through_gamma`) and the glyph is drawn over the composed frame, which
//! comes to the same pixels. The DAC cursor is a different thing: it sits inside
//! the video DAC, after gamma (established), and the kernel enables exactly one
//! of the two. Its load is decoded and kept by channel (`dac_cursor`), and channel 0's is drawn
//! over the composed frame, past the gamma RAM, in X11 mode when enabled (the other modes, and
//! what is inferred, are in that module's header). The crosshair, whose shape is unknown, is not
//! drawn.
//!
//! # Blanking
//!
//! A blanked channel 0, or a temporary blank of every channel, shows black at scan-out
//! (`Baked::is_blank`); the pixel store is untouched and drawing goes on under it. Other channels'
//! blanking is kept and not shown, as the board shows one channel's frame.
//!
//! Not composed: stereo, the crosshair, the 16-plane
//! luminance and 15-bit RGB windows (counted by reason), and anything but the
//! first channel's gamma page. Widening a 10-bit entry to the gamma RAM's 12
//! bits is inferred (the padding rule is unknown); the recorded desktop and PROM
//! frames match their captures to the pixel.
//!
//! That a textport pixel is a colour index, resolved at scan-out and not when
//! drawn, is inferred from the textport's colour-select command taking an
//! index. The textport colour map's entries are 10-bit components the sender
//! scales up by four, narrowed to eight bits as they are stored.

pub mod channels;
pub mod clock;
pub mod dac_cursor;
pub mod screen;
pub mod tables;

use super::store::{PixelStore, DID_PLANE};
use super::store::{OVERLAY0_PLANE, OVERLAY1_PLANE, POPUP_PLANE};
use crate::dev::kona::defs::{OVERLAY_ID_FIRST, POPUP_ID};
use tables::{Baked, Tables, Window};
use parking_lot::Mutex;
use screen::Display;
use std::io::{self, Write};
use std::sync::Arc;

/// Entries in the textport's colour map.
pub const COLOUR_MAP_ENTRIES: usize = 256;

/// The textport's colour map: an index to a colour, held as the bytes red,
/// green, blue and a full alpha, the order the screens in the emulator use
/// (`0xAABBGGRR` read as a little-endian word).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColourMap {
    entries: [u32; COLOUR_MAP_ENTRIES],
}

impl Default for ColourMap {
    fn default() -> Self {
        ColourMap { entries: [0xff00_0000; COLOUR_MAP_ENTRIES] }
    }
}

impl ColourMap {
    /// The entries, as the stream module reads and writes them.
    pub fn visit<S: crate::dev::kona::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { entries: _ } = self;
        s.u32s(&mut self.entries);
    }

    /// A colour-map load: an index and three 10-bit components, each the
    /// 8-bit value scaled up by four (so the top eight bits are kept).
    pub fn load(&mut self, args: &[u32]) {
        let [index, r, g, b] = *args else { return };
        let c = |v: u32| (v >> 2) & 0xff;
        self.entries[(index & 0xff) as usize] = 0xff00_0000 | c(b) << 16 | c(g) << 8 | c(r);
    }

    pub fn get(&self, index: u32) -> u32 {
        self.entries[(index & 0xff) as usize]
    }

    pub fn reset(&mut self) {
        *self = ColourMap::default();
    }
}

/// Compose the frame the default descriptor gives into `out`, `stride` pixels
/// a row, top row first. The store's rows run up from the bottom, so row `y` of
/// the frame is the store's row `height - 1 - y`. A word position with no array
/// yet is a store of zeros: index 0.
pub fn compose_default(store: &PixelStore, word: usize, map: &ColourMap, out: &mut [u32], stride: usize) {
    let (w, h) = (store.width() as usize, store.height() as usize);
    assert!(out.len() >= stride * h.saturating_sub(1) + w, "the frame buffer is too small");
    let Some(array) = store.array(word) else {
        let black = map.get(0);
        for y in 0..h {
            out[y * stride..y * stride + w].fill(black);
        }
        return;
    };
    let pitch = array.pitch();
    for y in 0..h {
        let src = (h - 1 - y) * pitch;
        let row = &mut out[y * stride..y * stride + w];
        for (x, px) in row.iter_mut().enumerate() {
            // SAFETY: `src + x` is inside the array: `x < w <= pitch` and the row is in range.
            *px = map.entries[(unsafe { array.load(src + x) } & 0xff) as usize];
        }
    }
}

/// Compose the frame with the tables the X server loaded: each pixel's display
/// ID (the ID plane's bits 7:2) names what its colour word is. A run of pixels
/// with the same ID is resolved to its window kind, and for an index window to
/// its slice of the baked colour map, once; each pixel is then a load, a shift
/// and a lookup. With nothing loaded this is `compose_default`, and the glyph
/// cursor goes over the top; the DAC cursor goes over that, and a blanked frame
/// is black. Each is chosen once for the frame, not per pixel.
pub fn compose(store: &PixelStore, word: usize, map: &ColourMap, tables: &Baked, out: &mut [u32], stride: usize) {
    if tables.is_blank() {
        // The scan-out shows black; the store is left as it is.
        let (w, h) = (store.width() as usize, store.height() as usize);
        assert!(out.len() >= stride * h.saturating_sub(1) + w, "the frame buffer is too small");
        (0..h).for_each(|y| out[y * stride..y * stride + w].fill(0xff00_0000));
        return;
    }
    if tables.is_default() {
        return compose_default(store, word, map, out, stride);
    }
    let (w, h) = (store.width() as usize, store.height() as usize);
    assert!(out.len() >= stride * h.saturating_sub(1) + w, "the frame buffer is too small");
    let (colour, back, did) = (store.array(word), store.array(word + 1), store.array(DID_PLANE));
    let pitch = store.width() as usize;
    let id_at = |at: usize| -> usize {
        // SAFETY: `at` is inside the managed area, which the array covers.
        did.as_ref().map_or(0, |d| ((unsafe { d.load(at) } >> 2) & 0x3f) as usize)
    };
    for y in 0..h {
        let src = (h - 1 - y) * pitch;
        let word_at = |i: usize| -> u32 {
            // SAFETY: `i` is inside the managed area.
            colour.as_ref().map_or(0, |c| unsafe { c.load(src + i) })
        };
        let row = &mut out[y * stride..y * stride + w];
        let mut x = 0;
        while x < w {
            let id = id_at(src + x);
            let mut end = x + 1;
            while end < w && id_at(src + end) == id {
                end += 1;
            }
            match tables.window(id) {
                Window::Textport => (x..end).for_each(|i| row[i] = map.entries[(word_at(i) & 0xff) as usize]),
                Window::Index8 { base } => {
                    let t = tables.index8(base);
                    (x..end).for_each(|i| row[i] = t[(word_at(i) >> 8) as usize & 0xff]);
                }
                Window::Index12 { base } => {
                    let t = tables.index12(base);
                    (x..end).for_each(|i| row[i] = t[(word_at(i) >> 4) as usize & 0xfff]);
                }
                Window::Direct { bank } => {
                    // The buffer the window's swaps last showed.
                    let shown = if bank == 0 { &colour } else { &back };
                    (x..end).for_each(|i| {
                        // SAFETY: `i` is inside the managed area.
                        row[i] = tables.direct(shown.as_ref().map_or(0, |c| unsafe { c.load(src + i) }));
                    })
                }
                Window::Cleared | Window::Unsupported(_) | Window::Layer { .. } => row[x..end].fill(0xff00_0000),
            }
            x = end;
        }
        if tables.has_layers() {
            compose_layers(store, tables, &mut row[..w], src);
        }
    }
    tables.draw_cursor(out, stride, w, h);
    tables.draw_dac_cursor(out, stride, w, h);
}

/// The overlay and the popup over a composed row, `src` being the row's first
/// pixel in the store. Popup over overlay over main, a layer's index 0 letting
/// what is under it through and any other index replacing it (inferred, from the
/// contract the X server's visuals advertise; the hardware's own choice is read
/// nowhere in software). An opaque overlay replaces the pixel whatever its index.
/// Which overlay window a pixel belongs to is read from the ID plane's bits 12:8,
/// as ID 64 plus that field (inferred: the overlay IDs run 64 to 95, and the
/// model's ID plane keeps those bits beside the main field); the overlay's two
/// planes are its two buffers, as the main colour words are.
fn compose_layers(store: &PixelStore, tables: &Baked, row: &mut [u32], src: usize) {
    let (did, popup) = (store.array(DID_PLANE), store.array(POPUP_PLANE));
    let overlay = [store.array(OVERLAY0_PLANE), store.array(OVERLAY1_PLANE)];
    let at = |a: &Option<super::store::WordArray<'_>>, i: usize| -> u32 {
        // SAFETY: `i` is inside the managed area, which the array covers.
        a.as_ref().map_or(0, |a| unsafe { a.load(src + i) })
    };
    for (i, px) in row.iter_mut().enumerate() {
        let id = OVERLAY_ID_FIRST + ((at(&did, i) >> 8) & 0x1f) as usize;
        if let Window::Layer { base, planes, opaque, bank } = tables.window(id) {
            let v = at(&overlay[bank as usize & 1], i) & ((1 << planes) - 1);
            if v != 0 || opaque {
                *px = tables.layer(base, v);
            }
        }
        if let Window::Layer { base, planes, .. } = tables.window(POPUP_ID) {
            let v = at(&popup, i) & ((1 << planes) - 1);
            if v != 0 {
                *px = tables.layer(base, v);
            }
        }
    }
}

/// The display thread's frame: compose the store through the tables every
/// retrace and present it, as VPro's display does. The store is not told when
/// drawing changes it, so a frame composed from a half-drawn store is simply
/// replaced by the next.
pub struct Compositor {
    store: Arc<PixelStore>,
    word: usize,
    map: Arc<Mutex<ColourMap>>,
    tables: Arc<Mutex<Tables>>,
    display: Arc<Display>,
    frame: Vec<u32>,
}

impl Compositor {
    pub fn new(store: Arc<PixelStore>, word: usize, map: Arc<Mutex<ColourMap>>, tables: Arc<Mutex<Tables>>, display: Arc<Display>) -> Compositor {
        let n = store.width() as usize * store.height() as usize;
        Compositor { store, word, map, tables, display, frame: vec![0; n] }
    }
}

impl Compositor {
    /// Compose the frame and present it.
    pub fn frame(&mut self) {
        // The requests this frame answers are noted before it is composed.
        let noted = self.display.note();
        // The tables are copied once per frame, under their lock; the store
        // is read as it is, without waiting for drawing.
        let map = self.map.lock().clone();
        let (w, h) = (self.store.width() as usize, self.store.height() as usize);
        let tables = self.tables.lock().snapshot();
        compose(&self.store, self.word, &map, &tables, &mut self.frame, w);
        self.display.present(&self.frame, w, h, noted);
    }

    /// The display thread's last act.
    pub fn finish(&mut self) {
        self.display.finish();
    }
}

/// The monitor's pixel reads, `hash`, `pix` and `shot`: they read the store and
/// the colour map through the storage the back end shares with the display, so
/// they run while the command stream does and never hold it up (a screenshot
/// composes 1.3 million pixels, and the pipe's state is not taken out for that).
pub struct Pixels {
    store: Arc<PixelStore>,
    map: Arc<parking_lot::Mutex<ColourMap>>,
    tables: Arc<parking_lot::Mutex<Tables>>,
}

impl Pixels {
    pub fn new(store: Arc<PixelStore>, map: Arc<parking_lot::Mutex<ColourMap>>, tables: Arc<parking_lot::Mutex<Tables>>) -> Pixels {
        Pixels { store, map, tables }
    }

    /// The words of one pixel that have an array, as the planes hold them.
    fn pix(&self, x: u32, y: u32, w: &mut dyn Write) -> io::Result<()> {
        if self.store.word_slot(x, y, 0).is_none() {
            return writeln!(w, "({x}, {y}) is outside the {} x {} managed area", self.store.width(), self.store.height());
        }
        let words = self.store.allocated();
        for i in &words {
            writeln!(w, "word {i:2}  {:#010x}", self.store.read(x, y, *i))?;
        }
        writeln!(w, "{} word arrays allocated", words.len())
    }

    /// The frame the display would show now, composed here from the store as it
    /// stands, without waiting for a display thread: rows from the top, one word a
    /// pixel (0xAABBGGRR).
    pub fn frame(&self) -> Vec<u32> {
        let w = self.store.width() as usize;
        let mut frame = vec![0u32; w * self.store.height() as usize];
        let map = self.map.lock().clone();
        let tables = self.tables.lock().snapshot();
        compose(&self.store, super::textport::COLOUR_WORD, &map, &tables, &mut frame, w);
        frame
    }

    /// The frame written as a PNG.
    fn shot(&self, path: &str) -> Result<(usize, usize), String> {
        let (w, h) = (self.store.width() as usize, self.store.height() as usize);
        crate::hostinput::save_png(path, &self.frame(), w, h, None)
    }
}

impl Pixels {
    /// A pixel monitor command; false if `args` is not one.
    pub fn read(&self, args: &[&str], w: &mut dyn Write) -> io::Result<bool> {
        match args {
            ["hash"] => writeln!(w, "{:016x}", self.store.checksum())?,
            ["shot", path] => match self.shot(path) {
                Ok((w_, h_)) => writeln!(w, "{path}: {w_} x {h_}")?,
                Err(e) => writeln!(w, "kona: {e}")?,
            },
            ["pix", x, y] => {
                let (Ok(x), Ok(y)) = (x.parse::<u32>(), y.parse::<u32>()) else { return Ok(false) };
                self.pix(x, y, w)?
            }
            _ => return Ok(false),
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_load_keeps_the_top_eight_bits_of_each_ten() {
        let mut m = ColourMap::default();
        m.load(&[5, 1020, 4, 0x3ff]);
        assert_eq!(m.get(5), 0xffff_01ff);
        assert_eq!(m.get(5) >> 24, 0xff, "alpha is full");
        assert_eq!(m.get(5) & 0xff, 0xff, "red 1020 is 255");
        assert_eq!((m.get(5) >> 8) & 0xff, 1, "green 4 is 1");
        assert_eq!((m.get(5) >> 16) & 0xff, 0xff, "blue 1023 is 255");
        m.load(&[0x105, 0, 0, 0]);
        assert_eq!(m.get(5) & 0xff_ffff, 0, "the index is a byte");
        m.load(&[1, 2, 3]);
        assert_eq!(m.get(1), 0xff00_0000, "a short load is ignored");
        m.reset();
        assert_eq!(m, ColourMap::default());
    }

    /// The frame is top row first: the store's bottom row is the frame's last.
    #[test]
    fn the_frame_is_composed_top_row_first_through_the_colour_map() {
        let s = PixelStore::new(4, 3);
        let mut m = ColourMap::default();
        m.load(&[1, 1020, 0, 0]); // red
        m.load(&[2, 0, 1020, 0]); // green
        s.write(0, 0, 1, 1); // bottom-left
        s.write(3, 2, 1, 2); // top-right
        let mut out = vec![0u32; 4 * 3];
        compose_default(&s, 1, &m, &mut out, 4);
        assert_eq!(out[2 * 4], m.get(1), "the store's row 0 is the frame's last row");
        assert_eq!(out[3], m.get(2), "the store's top row is the frame's first");
        assert_eq!(out[5], m.get(0));
        // A stride past the width leaves the gap alone.
        let mut wide = vec![0xdead_beefu32; 8 * 3];
        compose_default(&s, 1, &m, &mut wide, 8);
        assert_eq!(wide[4], 0xdead_beef);
        assert_eq!(wide[8 * 2], m.get(1));
        // Only the low byte of a word is an index.
        s.write(1, 1, 1, 0x1_0001);
        compose_default(&s, 1, &m, &mut out, 4);
        assert_eq!(out[4 + 1], m.get(1));
    }

    /// With no array the whole frame is index 0.
    #[test]
    fn an_unallocated_word_position_is_index_zero() {
        let s = PixelStore::new(3, 2);
        let mut m = ColourMap::default();
        m.load(&[0, 400, 800, 100]);
        let mut out = vec![0u32; 6];
        compose_default(&s, 1, &m, &mut out, 3);
        assert!(out.iter().all(|&p| p == m.get(0)));
    }

    /// Each pixel's display ID, in the ID plane's bits 7:2, picks what its colour
    /// word is: an 8-bit index in the top byte, a 12-bit one in bits 15:4, direct
    /// colour, a window the model does not read, or (with no entry) the textport.
    #[test]
    fn the_display_id_picks_how_a_word_becomes_a_colour() {
        let s = PixelStore::new(6, 1);
        let mut t = Tables::default();
        t.load_colours(0x100, &[0x3ff << 20, 0, 0x3ff << 10]);
        t.load_colours(0x200, &[0, 0x3ff]);
        t.load_did(1, 0x100 << 7 | 7 << 3);
        t.load_did(2, 0x200 << 7 | 11 << 3);
        t.load_did(4, 0x140808);
        t.load_mode(3, 4);
        // Pixels: ID 1 index 2, ID 2 index 1 (word 0x10), ID 3 direct, ID 4, ID 5, and ID 1 again.
        for (x, (id, word)) in [(1u32, 2u32 << 8), (2, 0x10), (3, 0x00_aa_bb_cc), (4, 0x1234), (5, 3), (1, 0)].into_iter().enumerate() {
            s.write(x as u32, 0, 1, word);
            s.write(x as u32, 0, DID_PLANE, id << 2 | 0x100);
        }
        let mut m = ColourMap::default();
        m.load(&[3, 1020, 0, 0]);
        let mut out = vec![0u32; 6];
        compose(&s, 1, &m, &t.snapshot(), &mut out, 6);
        assert_eq!(out[0], 0xff00_ff00, "index 2 of the map at 0x100 is green");
        assert_eq!(out[1], 0xffff_0000, "the 12-bit index 1 at 0x200 is blue");
        assert_eq!(out[2], 0xffcc_bbaa, "direct colour: red aa, green bb, blue cc");
        assert_eq!(out[3], 0xff00_0000, "a window the model does not read is black");
        assert_eq!(out[4], m.get(3), "no entry for the ID: the textport's colour map");
        assert_eq!(out[5], 0xff00_00ff, "index 0 of the map at 0x100 is red");
    }

    /// A cursor image for channel 0 of 64 x 64: row 0 is foreground then background then clear.
    fn dac_load() -> dac_cursor::DacCursor {
        let mut w = vec![0u32; dac_cursor::WORDS];
        w[1..7].copy_from_slice(&[0x3ff, 0, 0, 0, 0, 0x3ff]);
        // Pixels 0 to 3 of row 0 are the one byte at RAM offset 12 (the low byte of word 3), pixel 0
        // in its top two bits.
        w[7 + 3] = 0b11_10_00_00;
        dac_cursor::DacCursor::from_request(&w).unwrap()
    }

    /// The DAC cursor is drawn over the composed frame, in the DAC's colours and past the gamma
    /// RAM, only once channel 0's control register enables it in X11 mode.
    #[test]
    fn the_dac_cursor_is_drawn_after_the_frame() {
        let s = PixelStore::new(8, 2);
        let mut t = Tables::default();
        let mut m = ColourMap::default();
        m.load(&[0, 1020, 1020, 1020]);
        s.write(0, 0, 1, 0);
        let frame = |t: &Tables| {
            let mut out = vec![0u32; 16];
            compose(&s, 1, &m, &t.snapshot(), &mut out, 8);
            out
        };
        t.load_dac_cursor(dac_load());
        assert!(t.snapshot().is_default(), "loaded and not enabled changes nothing");
        assert!(frame(&t).iter().all(|&p| p == m.get(0)));
        // Control 5 at index 0x204, then the position bytes at 0x200: x = 2, y = 1.
        for (selector, value) in [(0xb00, 0x04), (0xb00, 0x02), (0xb02, 5), (0xb00, 0x00), (0xb00, 0x02), (0xb02, 2), (0xb02, 0), (0xb02, 1), (0xb02, 0)] {
            t.indirect(selector, value);
        }
        assert!(!t.snapshot().is_default());
        let shown = frame(&t);
        // Foreground is full red (10 bits), background full blue.
        assert_eq!((shown[8 + 2], shown[8 + 3]), (0xff00_00ff, 0xffff_0000));
        assert_eq!(shown.iter().filter(|&&p| p != m.get(0)).count(), 2);
        // A cursor that sits over a blanked frame is blanked with it.
        t.set_blank(true);
        assert!(frame(&t).iter().all(|&p| p == 0xff00_0000));
    }

    /// A blanked frame is black and leaves the store alone; showing it again restores the frame.
    #[test]
    fn blanking_blacks_the_frame_and_leaves_the_store() {
        let s = PixelStore::new(8, 2);
        let mut t = Tables::default();
        let mut m = ColourMap::default();
        m.load(&[0, 1020, 1020, 1020]);
        s.write(0, 0, 1, 0);
        let frame = |t: &Tables| {
            let mut out = vec![0u32; 16];
            compose(&s, 1, &m, &t.snapshot(), &mut out, 8);
            out
        };
        let shown = frame(&t);
        assert!(shown.iter().all(|&p| p == 0xffff_ffff));
        t.set_blank(true);
        assert!(!t.snapshot().is_default());
        assert!(frame(&t).iter().all(|&p| p == 0xff00_0000));
        assert_eq!(s.read(0, 0, 1), 0, "the store is not touched");
        t.set_blank(false);
        assert!(t.snapshot().is_default());
        assert_eq!(frame(&t), shown);
    }

    /// The popup lies over the overlay and the overlay over the main window; an
    /// index of 0 lets what is under it through, an opaque overlay does not, and
    /// the overlay's second buffer shows after its swap.
    #[test]
    fn the_popup_lies_over_the_overlay_over_the_main_window() {
        use crate::dev::kona::back::store::{OVERLAY0_PLANE, OVERLAY1_PLANE, POPUP_PLANE};
        let s = PixelStore::new(6, 1);
        let mut t = Tables::default();
        // Main window ID 1: direct colour. Overlay ID 64 + 2, 4 planes, map at 0x100; popup ID 63, map at 0x200.
        t.load_mode(1, 4);
        t.load_did(66, 0x100 << 7 | 3 << 3);
        t.load_mode(66, 1);
        t.load_did(63, 0x200 << 7 | 1 << 3);
        t.load_colours(0x100, &[0, 0x3ff << 20, 0x3ff << 10, 0x3ff]);
        t.load_colours(0x200, &[0, 0x3ff << 20 | 0x3ff << 10, 0x3ff << 10 | 0x3ff, 0]);
        // Six pixels: main only; overlay 1; overlay 0 (transparent); popup 1 over overlay 1; popup 0 (transparent) over overlay 2; no overlay window.
        let main = 0x00_11_22_33u32;
        for x in 0..6u32 {
            s.write(x, 0, 1, main);
            s.write(x, 0, 2, 0x00_44_55_66);
            s.write(x, 0, DID_PLANE, 1 << 2 | if x == 5 { 0 } else { 2 << 8 });
        }
        s.write(1, 0, OVERLAY0_PLANE, 1);
        s.write(2, 0, OVERLAY0_PLANE, 0);
        s.write(3, 0, OVERLAY0_PLANE, 1);
        s.write(3, 0, POPUP_PLANE, 1);
        s.write(4, 0, OVERLAY0_PLANE, 2);
        s.write(5, 0, OVERLAY0_PLANE, 1);
        let mut out = vec![0u32; 6];
        compose(&s, 1, &ColourMap::default(), &t.snapshot(), &mut out, 6);
        let main_px = 0xff33_2211;
        assert_eq!(out[0], main_px, "no layer: the main window");
        assert_eq!(out[1], 0xff00_00ff, "overlay index 1 is red");
        assert_eq!(out[2], main_px, "overlay index 0 lets the main window through");
        assert_eq!(out[3], 0xff00_ffff, "the popup over the overlay: its index 1 is red and green");
        assert_eq!(out[4], 0xff00_ff00, "popup index 0 shows the overlay's index 2, green");
        assert_eq!(out[5], main_px, "a pixel whose ID plane names no decoded overlay");
        // The overlay's second buffer after a swap, and an opaque overlay replacing even index 0.
        t.load_bank(66, 1);
        s.write(1, 0, OVERLAY1_PLANE, 3);
        compose(&s, 1, &ColourMap::default(), &t.snapshot(), &mut out, 6);
        assert_eq!(out[1], 0xffff_0000, "buffer 1, index 3 is blue");
        t.load_mode(66, 0x11);
        compose(&s, 1, &ColourMap::default(), &t.snapshot(), &mut out, 6);
        assert_eq!(out[2], 0xff00_0000, "opaque: index 0 of buffer 1 is the map's first entry, black");
    }

    /// With no table loaded the composition is the default one.
    #[test]
    fn with_no_tables_the_frame_is_the_default() {
        let s = PixelStore::new(3, 1);
        s.write(1, 0, 1, 2);
        let mut m = ColourMap::default();
        m.load(&[2, 1020, 0, 0]);
        let (mut a, mut b) = (vec![0u32; 3], vec![0u32; 3]);
        compose(&s, 1, &m, &Tables::default().snapshot(), &mut a, 3);
        compose_default(&s, 1, &m, &mut b, 3);
        assert_eq!(a, b);
    }
}
